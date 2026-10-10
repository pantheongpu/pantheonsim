// What libvgpucudnn's three sources share: the descriptor registry, the last
// error message, and tensors on the host -- strided device memory of any of
// cuDNN's element types read into logical order as doubles and written back --
// plus the one convolution all of the classic API, and the backend API, run.
//
// Internal to the library: the namespace is hidden, so nothing here is
// exported next to cuDNN's own names.
#pragma once

#include <cudnn.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace vgpu_cudnn __attribute__((visibility("hidden"))) {

inline bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}
inline bool trace() {
  const char* t = std::getenv("VGPU_TRACE");
  return t && t[0] == '1';
}

// Every classic descriptor (handles, tensors, filters, convolutions, ...) is
// registered when created, so a stale or foreign pointer is refused rather
// than dereferenced. One registry for the whole library: the RNN API takes the
// dropout descriptor the classic API's dropout calls take. (cudnn_api.cpp)
void* track_raw(void* p);
bool known(const void* p);
void untrack(const void* p);
template <class T> T* track(T* p) { return static_cast<T*>(track_raw(p)); }

// cudnnGetLastErrorString's message: the reason the last call on this thread
// failed. fail() records it and returns the status; a NOT_SUPPORTED is also
// printed (unless VGPU_QUIET=1), since it is this library declining work the
// hardware would do.
void set_last_error(const std::string& msg);
std::string last_error();
cudnnStatus_t fail(cudnnStatus_t s, const char* fn, const std::string& why);

/* ---- element types ---- */

inline size_t type_bytes(cudnnDataType_t t) {
  switch (t) {
    case CUDNN_DATA_HALF:
    case CUDNN_DATA_BFLOAT16: return 2;
    case CUDNN_DATA_DOUBLE:
    case CUDNN_DATA_INT64: return 8;
    case CUDNN_DATA_INT8:
    case CUDNN_DATA_UINT8:
    case CUDNN_DATA_FP8_E4M3:
    case CUDNN_DATA_FP8_E5M2:
    case CUDNN_DATA_BOOLEAN: return 1;
    default: return 4;  // FLOAT, INT32, and the packed INT8x4/UINT8x4
  }
}
// Types this library reads and writes element by element.
inline bool storable(cudnnDataType_t t) {
  switch (t) {
    case CUDNN_DATA_FLOAT: case CUDNN_DATA_DOUBLE: case CUDNN_DATA_HALF: case CUDNN_DATA_BFLOAT16:
    case CUDNN_DATA_INT8: case CUDNN_DATA_UINT8: case CUDNN_DATA_INT32: case CUDNN_DATA_INT64:
    case CUDNN_DATA_BOOLEAN: case CUDNN_DATA_FP8_E4M3: case CUDNN_DATA_FP8_E5M2:
      return true;
    default: return false;
  }
}
inline bool floating(cudnnDataType_t t) {
  return t == CUDNN_DATA_FLOAT || t == CUDNN_DATA_DOUBLE || t == CUDNN_DATA_HALF || t == CUDNN_DATA_BFLOAT16;
}
const char* type_name(cudnnDataType_t t);

float half_to_float(uint16_t h);
uint16_t float_to_half(float f);  // round to nearest even
float bf16_to_float(uint16_t b);
uint16_t float_to_bf16(float f);  // round to nearest even

// One element, decoded from / encoded to its bytes. Encoding rounds as the
// hardware's conversions do: to nearest even, and integers saturate.
double decode(cudnnDataType_t t, const uint8_t* p);
void encode(cudnnDataType_t t, double v, uint8_t* p);
// v as the type holds it.
double round_to(cudnnDataType_t t, double v);

// A scaling factor (alpha, beta): a double for double data, else a float.
inline double scale_of(const void* p, cudnnDataType_t data, double dflt = 1.0) {
  if (!p) return dflt;
  if (data == CUDNN_DATA_DOUBLE) return *static_cast<const double*>(p);
  return *static_cast<const float*>(p);
}

/* ---- tensors ---- */

constexpr int kMaxRank = CUDNN_DIM_MAX;

// A tensor's element type and shape. dims are logical (N, C, then spatial,
// whatever the memory order); strides say where each lands, in elements.
struct Layout {
  cudnnDataType_t type = CUDNN_DATA_FLOAT;
  int rank = 0;
  int64_t dims[kMaxRank] = {};
  int64_t strides[kMaxRank] = {};

  size_t count() const {
    if (!rank) return 0;
    size_t n = 1;
    for (int i = 0; i < rank; ++i) n *= static_cast<size_t>(dims[i]);
    return n;
  }
  // Elements from the first to one past the last, as the strides lay them out.
  size_t span() const {
    if (!count()) return 0;
    size_t s = 1;
    for (int i = 0; i < rank; ++i) s += static_cast<size_t>((dims[i] - 1) * strides[i]);
    return s;
  }
  bool same_dims(const Layout& o) const {
    if (rank != o.rank) return false;
    for (int i = 0; i < rank; ++i)
      if (dims[i] != o.dims[i]) return false;
    return true;
  }
};

// Packed strides for dims in NCHW order (row-major), or channels-last (NHWC:
// C innermost, then the spatial dimensions, then N).
void packed_strides(int rank, const int64_t* dims, bool channels_last, int64_t* strides);

// Calls f(logical index, element offset) for every element in logical
// (row-major) order.
template <class F>
void each(const Layout& t, F&& f) {
  const size_t n = t.count();
  int64_t idx[kMaxRank] = {};
  size_t off = 0;
  for (size_t i = 0; i < n; ++i) {
    f(i, off);
    for (int d = t.rank; d-- > 0;) {
      if (++idx[d] < t.dims[d]) { off += static_cast<size_t>(t.strides[d]); break; }
      off -= static_cast<size_t>((t.dims[d] - 1) * t.strides[d]);
      idx[d] = 0;
    }
  }
}

// Device memory in logical order: read() copies the tensor's span to the host
// and decodes it; write() encodes the values back over the span, leaving any
// gaps between strided elements as they were.
bool read(const Layout& t, const void* dev, std::vector<double>* out);
bool write(const Layout& t, void* dev, const std::vector<double>& v);

// out = alpha * r + beta * out, reading the prior output only when beta is
// nonzero (cuDNN's rule, so a NaN-filled output with beta = 0 is fine). r is
// rounded to the output's type by write().
bool blend_write(const Layout& t, void* dev, const std::vector<double>& r, double alpha, double beta);

/* ---- convolution ---- */

// Up to three spatial dimensions, right-aligned into three (fewer are padded
// with extent 1), groups, padding, stride, dilation, and convolution (filter
// flipped) or cross-correlation.
struct ConvGeom {
  int64_t N = 0, C = 0, K = 0, G = 1, Cg = 0, Kg = 0;
  int64_t in[3] = {1, 1, 1}, out[3] = {1, 1, 1}, flt[3] = {1, 1, 1};
  int64_t pad[3] = {0, 0, 0}, str[3] = {1, 1, 1}, dil[3] = {1, 1, 1};
  bool flip = false;
};

// The output extent one spatial dimension of a convolution gives, padded by
// pre before and post after.
inline int64_t conv_out(int64_t in, int64_t pre, int64_t post, int64_t flt, int64_t str, int64_t dil) {
  return 1 + (in + pre + post - ((flt - 1) * dil + 1)) / str;
}
inline int64_t conv_out(int64_t in, int64_t pad, int64_t flt, int64_t str, int64_t dil) {
  return conv_out(in, pad, pad, flt, str, dil);
}

// x: [N, C, sp...], w: [K, C/G, sp...], y: [N, K, sp...] (logical dims), and
// the convolution's per-dimension settings; post paddings, if given, differ
// from the pre paddings (pad). False, with the reason, when they do not fit
// together.
bool conv_geometry(const Layout& x, const Layout& w, const Layout& y, int nsp, const int64_t* pad,
                   const int64_t* str, const int64_t* dil, bool flip, ConvGeom* g, std::string* why,
                   const int64_t* post = nullptr);

enum class ConvDir { Forward, Data, Filter };
// How the products are summed: exactly (in double, then rounded once when the
// result is stored -- what float and double compute give, up to their own
// summation order); in half precision, rounding after every step, as a
// TRUE_HALF configuration does on the hardware; or in integers (INT8 data).
enum class Accum { Exact, Half, Int };

// Every output element of one direction, each as one sum, before alpha/beta:
//   Forward: y[n,k,o]  = sum x[n,c,p] w[k,c',f]       (a = x,  b = w)
//   Data:    dx[n,c,p] = sum dy[n,k,o] w[k,c',f]      (a = dy, b = w)
//   Filter:  dw[k,c',f] = sum dy[n,k,o] x[n,c,p]      (a = dy, b = x)
// over the taps that land inside the input. Inputs and the result are in
// logical (packed NCHW) order.
void convolve(const ConvGeom& g, ConvDir dir, const std::vector<double>& a, const std::vector<double>& b,
              std::vector<double>* out, Accum acc);

// A dropout descriptor, which the classic API's dropout calls and the RNN
// API both take. The generator lives in the caller's states buffer, as
// cuDNN keeps it there: one XORWOW state (cuRAND's, 48 bytes) per thread of
// the dropout kernel, each seeded as curand_init(seed, thread, 0) seeds it.
// cudnnRestoreDropoutDescriptor then resumes where the buffer says, and a
// fresh cudnnSetDropoutDescriptor starts over.
struct DropoutDesc {
  float p = 0.0f;
  void* states = nullptr;
  size_t state_bytes = 0;
  unsigned long long seed = 0;
  size_t threads = 0;            // the kernel's generators, for the device it was set on
  std::vector<uint32_t> local;   // their states (12 words each), when there is no states buffer
};

// The generators cuDNN's dropout kernel runs on the current device, and the
// bytes of states they take (cudnnDropoutGetStatesSize). Measured on an RTX
// 3060 (28 SMs): 21504 generators, 1032192 bytes, that is 768 per SM; other
// devices follow the same rule here without a measurement to confirm it.
size_t dropout_threads();
inline size_t dropout_states_bytes() { return dropout_threads() * 48; }
// Seeds the descriptor's generators: into its states buffer when it has one
// (the words cuDNN writes; its padding word is left as it was), else into
// the descriptor. False if the buffer cannot be written.
bool dropout_seed(DropoutDesc* d);
// Draws n keep (1) / drop (0) decisions, each kept with probability 1 - p,
// from the descriptor's generators, and advances them -- cuDNN's own draw,
// measured on an RTX 3060 against its states buffer: element i is thread
// i % threads's next XORWOW output, kept when curand_uniform of it exceeds p.
// (cudnn_common.cpp)
bool dropout_draw(DropoutDesc* d, size_t n, std::vector<uint8_t>* keep);

// Waits for what the program queued on the handle's stream: this library
// computes on the host, and the inputs must be there first.
void sync_handle(cudnnHandle_t h);

// Leaves a tensor descriptor with no dimensions, as cuDNN reports a weight
// matrix or bias that does not exist (cudnnGetRNNWeightParams).
// (cudnn_api.cpp)
void clear_tensor(cudnnTensorDescriptor_t d);

// cudnnReorderFilterAndBias's permutation of an INT8x32 filter, measured on
// the RTX 3060 by reordering filters whose bytes count their own positions
// (K 4 to 96, C 32 and 64, 1x1 to 3x3): the filter is a [K][L] byte matrix
// (L = C/32 * R * S * 32) cut into 32-byte column chunks; the output holds
// chunk 0 of every row first, then chunk 1, ..., each as groups of 8 rows,
// and within a group output vector j (0..7), lane l takes row
// 8g + j/4 + 2((l%16)/4), byte (j%4)*8 + (l/16)*4 + l%4 of the chunk. Rows
// past K (K not a multiple of 8) come out as zeros. Calls f(output byte,
// source byte or -1).
template <class F>
inline void x32_filter_map(int64_t K, int64_t L, size_t bytes, F&& f) {
  const int64_t G = (K + 7) / 8;
  for (size_t d = 0; d < bytes; ++d) {
    const int64_t l = static_cast<int64_t>(d % 32), t = static_cast<int64_t>(d / 32);
    const int64_t j = t % 8, g = (t / 8) % G, c = t / 8 / G;
    const int64_t row = g * 8 + j / 4 + 2 * ((l % 16) / 4), col = c * 32 + (j % 4) * 8 + (l / 16) * 4 + l % 4;
    f(d, row < K && col < L ? row * L + col : -1);
  }
}
}  // namespace vgpu_cudnn
