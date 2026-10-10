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
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
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
//
// A descriptor can also be copied (Snapshot, below), which is how a call made
// during stream capture keeps what it was given: the caller destroys its
// descriptors as soon as the capture function returns, and the graph runs the
// call later.
class Snapshot;
using CloneFn = void* (*)(const void*, Snapshot&);
void* track_raw(void* p, CloneFn clone = nullptr, void (*del)(void*) = nullptr);
bool known(const void* p);
void untrack(const void* p);
// `fix`, when given, copies an object whose members point at other descriptors
// (an RNN's dropout descriptor) and says which copies those pointers must name.
template <class T>
T* track(T* p, CloneFn fix = nullptr) {
  void (*del)(void*) = [](void* q) { delete static_cast<T*>(q); };
  if (fix) return static_cast<T*>(track_raw(p, fix, del));
  if constexpr (std::is_copy_constructible_v<T>)
    return static_cast<T*>(track_raw(p, [](const void* q, Snapshot&) -> void* { return new T(*static_cast<const T*>(q)); }, del));
  return static_cast<T*>(track_raw(p, nullptr, del));
}

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
// computes on the host, and the inputs must be there first. (During a probe, or
// when a call is replayed from a graph, see defer_call below.)
void sync_handle(cudnnHandle_t h);

/* ---- stream capture ------------------------------------------------------------- */

// On NVIDIA's cuDNN a call made while its handle's stream is capturing is recorded
// into the graph, and runs at each launch over what the graph's own kernels
// have written by then -- with the descriptors it was given as they were at the
// call, though the caller destroys them at once. This library computes on the
// host, so such a call is handed to the graph as a closure over copies.
//
// defer_call is the first line of an entry point (VGPU_DEFER). While the
// handle's stream is capturing it runs the entry point once as a PROBE, with
// the caller's own arguments: the validation happens, in the entry point's
// order, and every status it would return is returned. The first thing past
// validation is sync_handle, which during a probe throws ProbeCommit instead of
// waiting; that is the answer "this call is good and would touch the device".
// The arguments are then copied (every argument that is a registered
// descriptor, by Snapshot, and the host arrays and scalars named by Host) and the
// entry point is recorded to run again at each launch of the graph, on the
// copies, with sync_handle a no-op.
struct ProbeCommit {};
bool replaying();   // the call is running from a graph launch

// A host-side input whose extent only the call knows -- a scalar, a host array --
// as the entry point reads it: `bytes()` says how many bytes, and is asked only
// once the probe has validated the call (so what it reads is sound).
struct Host {
  const void* p;
  std::function<size_t()> bytes;
  template <class T> operator const T*() const { return static_cast<const T*>(p); }
};
// A scalar (alpha, beta) in the type of the tensor or filter `layout_owner`
// describes: double for double data, else float.
Host scalar(const void* p, const void* layout_owner);

class Snapshot {
 public:
  Snapshot() = default;
  Snapshot(const Snapshot&) = delete;
  Snapshot& operator=(const Snapshot&) = delete;
  ~Snapshot();
  // The copy of a registered object (made once); anything else as it is.
  void* of(const void* p);
  // A host buffer kept for the life of the snapshot.
  const void* keep(const void* p, size_t bytes);

 private:
  struct Copy { void* p; void (*del)(void*); };
  std::vector<std::pair<const void*, Copy>> done_;
  std::vector<std::shared_ptr<std::vector<uint8_t>>> bufs_;
};

namespace detail {
bool capturing_stream(cudnnHandle_t h, cudaStream_t* stream);
bool probe_active();
void set_probe(bool on);
bool record_closure(cudaStream_t stream, std::function<void()> op);
void set_replaying(bool on);

template <class P, class A>
P snap_arg(Snapshot& s, const A& a) {
  if constexpr (std::is_same_v<A, Host>) {
    return (P)(a.p ? s.keep(a.p, a.bytes()) : nullptr);
  } else if constexpr (std::is_pointer_v<P> && std::is_convertible_v<A, const void*>) {
    return (P)s.of(static_cast<const void*>(a));
  } else {
    return static_cast<P>(a);
  }
}
}  // namespace detail

template <class... P, class... A>
std::optional<cudnnStatus_t> defer_call(cudnnHandle_t h, cudnnStatus_t (*fn)(P...), A... a) {
  static_assert(sizeof...(P) == sizeof...(A), "defer_call: one argument per parameter");
  if (detail::probe_active() || replaying()) return std::nullopt;
  cudaStream_t stream = nullptr;
  if (!detail::capturing_stream(h, &stream)) return std::nullopt;
  {
    struct ProbeScope {
      ProbeScope() { detail::set_probe(true); }
      ~ProbeScope() { detail::set_probe(false); }
    } probe;
    try {
      return fn(a...);   // refused, or nothing to do on the device
    } catch (const ProbeCommit&) {
    }
  }
  auto snap = std::make_shared<Snapshot>();
  auto args = std::make_shared<std::tuple<std::decay_t<P>...>>(detail::snap_arg<std::decay_t<P>>(*snap, a)...);
  if (!detail::record_closure(stream, [fn, args, snap] {
        detail::set_replaying(true);
        std::apply(fn, *args);
        detail::set_replaying(false);
      }))
    return std::nullopt;   // the capture ended meanwhile: run it now
  return CUDNN_STATUS_SUCCESS;
}

#define VGPU_DEFER(h, fn, ...)                                                                         \
  do {                                                                                                 \
    if (auto vgpu_deferred_ = ::vgpu_cudnn::defer_call((h), &fn, __VA_ARGS__)) return *vgpu_deferred_; \
  } while (0)

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
