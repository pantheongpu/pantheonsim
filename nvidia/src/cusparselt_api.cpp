// libvgpucusparselt -- VirtualGPU's cuSPARSELt, presented as libcusparseLt.so.0.
//
// NVIDIA's libcusparseLt cannot run on a simulated GPU: it carries a
// statically linked CUDA runtime, which reaches the driver through NVIDIA's
// undocumented internal interface. This is the documented API (cuSPARSELt
// 0.10) implemented the way the simulator's other vendor libraries are: a
// call reads its operands out of simulated device memory, does the
// arithmetic on the host, and writes the result back. Application kernels
// are simulated; vendor library calls are implemented.
//
// cuSPARSELt multiplies D = act(alpha op(A) op(B) + beta C + bias) where one
// of A and B is "structured": 2 of every 4 consecutive values along the
// reduction dimension K are zero (1 of every 2 for 32-bit values). The
// structured operand is pruned to that shape (SpMMAPrune), then compressed
// (SpMMACompress) into its kept values and 2-bit position metadata, and
// Matmul takes the compressed form.
//
// What follows the card. Everything an application can observe was measured
// against NVIDIA's libcusparseLt 0.10.0 on an RTX 3060 (sm_86), and each rule
// below says so where it is not in the documentation:
//   - the descriptor checks and which of INVALID_VALUE and NOT_SUPPORTED each
//     refusal returns, the attribute defaults, sizes and accepted values;
//   - the type and layout combinations sm_86 accepts (fp16, bf16 and tf32
//     with fp32 compute; int8 with int32 compute to int8, int32, fp16 or bf16,
//     with both operands K-contiguous), and where each refusal happens;
//   - which values the STRIP and TILE pruning keep, ties included;
//   - the compressed buffer: its size, the kept values region, and the
//     metadata layout for the shapes measured (see "Compression" below);
//   - Matmul's rounding: fp32 accumulation for 16-bit inputs, operands
//     rounded to tf32 (to nearest, ties away) for fp32, int32 accumulation
//     for int8, round-to-nearest-even into every output type, saturation into
//     integers, ReLU's signed zero, the bias type, and alpha-vector scaling.
// What is not the card's: the configuration a search picks (the simulator
// has one kernel), and the workspace a plan asks for is the card's for the
// default split-K but is never used. Refused, with a message: FP8/FP4 inputs
// and fp16 compute (the card refuses both on sm_86); scale modes are
// accepted and ignored, as the card accepts them on these types;
// GELU outside int8 output is INVALID_VALUE, as on the card. A call on a
// stream that is capturing a graph is recorded and runs at each launch.
#include "../include/vgpu_cusparselt.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "vgpu/runtime/capture.hpp"

namespace {

using Status = cusparseStatus_t;

// Numeric values the CUDA 12.0 headers may not name.
constexpr int kR8F_E4M3 = 28, kR8F_E5M2 = 29, kR4F_E2M1 = 33;

// ---- logging ----
//
// NVIDIA's library reports a bad argument on stderr as
//   " ** On entry to cusparseLtX() parameter number N (name) had an illegal value: ..."
// and so does this one. Refusals of something the simulator does not do are
// "[vgpu] ..." lines. VGPU_QUIET=1 silences both.
bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

Status bad_arg(const char* api, int num, const char* name, const std::string& what) {
  if (!quiet())
    std::fprintf(stderr, " ** On entry to %s() parameter number %d (%s) had an illegal value: %s\n\n", api,
                 num, name, what.c_str());
  return CUSPARSE_STATUS_INVALID_VALUE;
}

Status refuse(const char* api, const std::string& why) {
  if (!quiet()) std::fprintf(stderr, "[vgpu] %s: %s\n", api, why.c_str());
  return CUSPARSE_STATUS_NOT_SUPPORTED;
}

// ---- the opaque objects ----
//
// Every object lives inside the caller's 512 bytes, so descriptors can be
// copied, and destroying one only marks it. A magic number tells an
// initialized object from a zeroed or destroyed one, as NVIDIA's library does
// ("bad initialization or already destroyed").
constexpr uint32_t kHandleMagic = 0x564c5448;  // "HTLV"
constexpr uint32_t kMatMagic = 0x564c544d;
constexpr uint32_t kMatmulMagic = 0x564c5444;
constexpr uint32_t kAlgMagic = 0x564c5441;
constexpr uint32_t kPlanMagic = 0x564c5450;

struct HandleImpl {
  uint32_t magic;
  int32_t sm;
};

struct MatImpl {
  uint32_t magic;
  uint8_t structured;
  uint8_t order;  // cusparseOrder_t
  int32_t type;   // cudaDataType
  uint32_t alignment;
  int32_t batches;
  int64_t rows, cols, ld, stride;
};

struct MatmulImpl {
  uint32_t magic;
  int32_t opA, opB, compute;
  MatImpl A, B, C, D;
  int32_t relu, gelu, alpha_vec, beta_vec;
  float relu_ub, relu_th, gelu_scale;
  int64_t bias_stride;
  const void* bias;
  const void* sparse_ptr;
  int32_t scale_mode[5];
  const void* scale_ptr[5];
};

struct AlgImpl {
  uint32_t magic;
  int32_t config_id, max_id, search_iters, split_k, split_k_mode, split_k_buffers;
};

struct PlanImpl {
  uint32_t magic;
  MatmulImpl md;
  AlgImpl alg;
};

static_assert(sizeof(HandleImpl) <= 512 && sizeof(MatImpl) <= 512 && sizeof(MatmulImpl) <= 512 &&
                  sizeof(AlgImpl) <= 512 && sizeof(PlanImpl) <= 512,
              "every cuSPARSELt object must fit its 512 opaque bytes");

template <class T, class O>
T* impl(O* o) { return reinterpret_cast<T*>(o->data); }
template <class T, class O>
const T* impl(const O* o) { return reinterpret_cast<const T*>(o->data); }

// The handle checks every call starts with: parameter 1, NULL or not
// initialized (measured: a zeroed or destroyed handle is INVALID_VALUE).
Status check_handle(const char* api, const cusparseLtHandle_t* h) {
  if (!h) return bad_arg(api, 1, "handle", "NULL pointer");
  if (impl<HandleImpl>(h)->magic != kHandleMagic)
    return bad_arg(api, 1, "handle", "bad initialization or already destroyed");
  return CUSPARSE_STATUS_SUCCESS;
}

// ---- types ----

int type_bits(int t) {
  switch (t) {
    case CUDA_R_32F: case CUDA_R_32I: return 32;
    case CUDA_R_16F: case CUDA_R_16BF: return 16;
    case CUDA_R_8I: case kR8F_E4M3: case kR8F_E5M2: return 8;
    case kR4F_E2M1: return 4;
    default: return 0;
  }
}
size_t type_bytes(int t) { return (size_t)type_bits(t) / 8; }
// Measured: a descriptor takes these value types, and refuses fp64, complex
// and unsigned with INVALID_VALUE. FP8 and FP4 descriptors are accepted; the
// matmul descriptor is what refuses them on sm_86.
bool descriptor_type(int t) { return type_bits(t) != 0; }

const char* type_name(int t) {
  switch (t) {
    case CUDA_R_32F: return "CUDA_R_32F";
    case CUDA_R_32I: return "CUDA_R_32I";
    case CUDA_R_16F: return "CUDA_R_16F";
    case CUDA_R_16BF: return "CUDA_R_16BF";
    case CUDA_R_8I: return "CUDA_R_8I";
    case CUDA_R_64F: return "CUDA_R_64F";
    case CUDA_C_32F: return "CUDA_C_32F";
    case CUDA_C_64F: return "CUDA_C_64F";
    case CUDA_R_8U: return "CUDA_R_8U";
    default: return "UNKNOWN";
  }
}

float half_to_float(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  int e = (h >> 10) & 0x1f;
  uint32_t m = h & 0x3ff, bits;
  if (e == 0x1f) bits = sign | 0x7f800000u | (m << 13);
  else if (e == 0) {
    if (!m) bits = sign;
    else {  // subnormal: normalize
      e = 1;
      while (!(m & 0x400)) { m <<= 1; --e; }
      bits = sign | (uint32_t)(e + 112) << 23 | ((m & 0x3ff) << 13);
    }
  } else bits = sign | (uint32_t)(e + 112) << 23 | (m << 13);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// fp32 to fp16, round to nearest even (measured: 2049 -> 2048, 2051 -> 2052).
uint16_t float_to_half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000;
  const uint32_t ex = (x >> 23) & 0xff;
  uint32_t m = x & 0x7fffff;
  if (ex == 0xff) return (uint16_t)(sign | 0x7c00 | (m ? 0x200 : 0));
  const int e = (int)ex - 112;
  if (e >= 31) return (uint16_t)(sign | 0x7c00);
  if (e <= 0) {
    if (e < -10) return (uint16_t)sign;
    m |= 0x800000;
    const int shift = 14 - e;
    uint32_t h = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1))) ++h;
    return (uint16_t)(sign | h);
  }
  uint32_t h = ((uint32_t)e << 10) | (m >> 13);
  const uint32_t rem = m & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) ++h;
  return (uint16_t)(sign | h);
}

float bf16_to_float(uint16_t b) {
  const uint32_t x = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

uint16_t float_to_bf16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  if ((x & 0x7fffffff) > 0x7f800000) return (uint16_t)((x >> 16) | 0x40);
  x += 0x7fff + ((x >> 16) & 1);
  return (uint16_t)(x >> 16);
}

// tf32 as the card makes it (measured): round to nearest with ties away from
// zero, by adding half a tf32 unit to the bits and dropping the low 13. The
// compressed values region stores the bits with the half unit added and the
// low bits left in place; the tensor core drops them.
uint32_t tf32_bias(uint32_t bits) { return bits + 0x1000u; }
float tf32_round(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  if ((x & 0x7f800000) != 0x7f800000) x = tf32_bias(x) & 0xffffe000u;
  std::memcpy(&f, &x, 4);
  return f;
}
float tf32_truncate_bits(uint32_t x) {
  if ((x & 0x7f800000) != 0x7f800000) x &= 0xffffe000u;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

// One element at a byte address, as float (int8/int32 exactly representable
// in the ranges that matter, and kept as integers where it counts).
float load(const uint8_t* p, int t) {
  switch (t) {
    case CUDA_R_32F: { float f; std::memcpy(&f, p, 4); return f; }
    case CUDA_R_32I: { int32_t i; std::memcpy(&i, p, 4); return (float)i; }
    case CUDA_R_16F: { uint16_t h; std::memcpy(&h, p, 2); return half_to_float(h); }
    case CUDA_R_16BF: { uint16_t h; std::memcpy(&h, p, 2); return bf16_to_float(h); }
    case CUDA_R_8I: return (float)(int8_t)*p;
    default: return 0.f;
  }
}

// Integers round to nearest even and saturate (measured: -17.5 -> -18,
// -16.5 -> -16, -4065 -> -128 in int8; the same rounding into int32).
void store(uint8_t* p, int t, float v) {
  switch (t) {
    case CUDA_R_32F: std::memcpy(p, &v, 4); break;
    case CUDA_R_16F: { const uint16_t h = float_to_half(v); std::memcpy(p, &h, 2); break; }
    case CUDA_R_16BF: { const uint16_t h = float_to_bf16(v); std::memcpy(p, &h, 2); break; }
    case CUDA_R_8I: {
      float r = std::isnan(v) ? 0.f : std::nearbyint(v);
      r = std::min(127.f, std::max(-128.f, r));
      *p = (uint8_t)(int8_t)r;
      break;
    }
    case CUDA_R_32I: {
      double r = std::isnan(v) ? 0.0 : std::nearbyint((double)v);
      r = std::min(2147483647.0, std::max(-2147483648.0, r));
      const int32_t i = (int32_t)r;
      std::memcpy(p, &i, 4);
      break;
    }
    default: break;
  }
}

// ---- simulated device memory ----

bool read_bytes(void* dst, const void* src, size_t n) {
  if (!n) return true;
  const bool ok = cudaMemcpy(dst, src, n, cudaMemcpyDefault) == cudaSuccess;
  cudaGetLastError();
  return ok;
}

bool write_bytes(void* dst, const void* src, size_t n) {
  if (!n) return true;
  const bool ok = cudaMemcpy(dst, src, n, cudaMemcpyDefault) == cudaSuccess;
  cudaGetLastError();
  return ok;
}

bool capturing(cudaStream_t s) {
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  const bool on = cudaStreamIsCapturing(s, &st) == cudaSuccess && st == cudaStreamCaptureStatusActive;
  cudaGetLastError();
  return on;
}

// Runs `work` in stream order: recorded into the graph when the stream is
// capturing (it then re-reads device memory at each launch, as the card's
// kernels would), otherwise after the stream's earlier work has finished.
Status in_stream_order(cudaStream_t s, std::function<Status()> work, const char* api) {
  if (capturing(s)) {
    std::string name = api;
    std::function<void()> op = [work, name] {
      if (work() != CUSPARSE_STATUS_SUCCESS && !quiet())
        std::fprintf(stderr, "[vgpu] %s (graph): a recorded call failed\n", name.c_str());
    };
    return vgpu_record_host_op_if_capturing(reinterpret_cast<CUstream_st*>(s), std::move(op))
               ? CUSPARSE_STATUS_SUCCESS
               : CUSPARSE_STATUS_EXECUTION_FAILED;
  }
  cudaGetLastError();
  if (cudaStreamSynchronize(s) != cudaSuccess) return CUSPARSE_STATUS_EXECUTION_FAILED;
  return work();
}

// ---- matrix geometry ----

int64_t lines(const MatImpl& m) { return m.order == CUSPARSE_ORDER_ROW ? m.rows : m.cols; }
int64_t matrix_elems(const MatImpl& m) { return lines(m) * m.ld; }
int64_t offset_of(const MatImpl& m, int64_t r, int64_t c) {
  return m.order == CUSPARSE_ORDER_ROW ? r * m.ld + c : c * m.ld + r;
}

// The structured operand seen logically: `nk` rows along the non-reduction
// dimension, `kk` columns along K. A with op N, and B with op T, have K along
// their columns; the other two along their rows.
struct SparseView {
  const MatImpl* m;
  bool k_along_cols;
  int64_t nk, k;
  int64_t at(int64_t i, int64_t kk) const {
    return k_along_cols ? offset_of(*m, i, kk) : offset_of(*m, kk, i);
  }
  // The values region keeps the input's memory orientation (measured): K
  // contiguous when the input has K contiguous, otherwise the other way.
  bool k_contiguous() const { return k_along_cols == (m->order == CUSPARSE_ORDER_ROW); }
};

SparseView sparse_view(const MatImpl& m, bool isA, int op) {
  SparseView v{&m, false, 0, 0};
  v.k_along_cols = isA ? op == CUSPARSE_OPERATION_NON_TRANSPOSE : op != CUSPARSE_OPERATION_NON_TRANSPOSE;
  v.nk = v.k_along_cols ? m.rows : m.cols;
  v.k = v.k_along_cols ? m.cols : m.rows;
  return v;
}

int64_t round_up(int64_t x, int64_t a) { return (x + a - 1) / a * a; }

// ---- compression ----
//
// The compressed matrix, per batch, is [kept values][metadata]:
//   values: the 2 kept of every 4 (1 of 2 for 32-bit), nk x k/2, dense, in
//     the input's memory orientation. fp32 values are stored with the tf32
//     rounding half-unit added to their bits (measured).
//   metadata: for 8- and 16-bit values a 4-bit code per group of 4, the two
//     kept positions i0 | i1 << 2; for fp32 a 4-bit code per pair, 0x4 for
//     the first and 0xE for the second.
// The positions kept are the nonzero ones; with fewer than two nonzeros the
// card pads from the right (measured: x000 -> (0,3), 0x00 -> (1,3),
// 00x0 -> (2,3), 000x -> (2,3), 0000 -> (2,3); a zero pair -> second).
//
// The metadata layout is the card's for every shape measured below 256 rows
// (fp16/bf16 in 64-row by 32-K blocks, int8 in 64 by 64 tiles, fp32 as a
// plain row-major copy followed by the fp16 block layout of the same codes),
// and the code 0xE fills its padding. NVIDIA's library switches the 8- and
// 16-bit layouts for larger shapes (seen at 256 x 64), which this one does
// not follow; Matmul here reads this layout, so results are unaffected.

// Sizes, measured over every shape of a grid up to 320 x 320 in each type
// and orientation (nk, k logical): bytes of metadata, and of the buffer
// SpMMACompress asks for.
size_t meta_bytes(int t, int64_t nk, int64_t k) {
  switch (t) {
    case CUDA_R_16F: case CUDA_R_16BF:
      return (size_t)std::max(round_up(nk, 32) * round_up(k, 64), round_up(nk, 64) * round_up(k, 32)) / 8;
    case CUDA_R_8I: return (size_t)(round_up(nk, 64) * round_up(k, 128) / 4);
    case CUDA_R_32F:
      return (size_t)std::max(round_up(nk, 16) * round_up(k, 64), round_up(nk, 64) * round_up(k, 16)) / 2;
    default: return 0;
  }
}
size_t buffer_bytes(int t, int64_t nk, int64_t k) {
  switch (t) {
    case CUDA_R_16F: case CUDA_R_16BF: return meta_bytes(t, nk, k);
    case CUDA_R_8I: return meta_bytes(t, nk, k) / 2;
    case CUDA_R_32F: return meta_bytes(t, nk, k) / 2;
    default: return 0;
  }
}
bool compressible_type(int t) {
  return t == CUDA_R_16F || t == CUDA_R_16BF || t == CUDA_R_8I || t == CUDA_R_32F;
}
size_t values_bytes(int t, int64_t nk, int64_t k) { return (size_t)(nk * (k / 2)) * type_bytes(t); }
// One batch's compressed bytes for a structured descriptor, which does not
// say which operand it is: K is taken to be the contiguous dimension
// (measured: CompressedSize2 and the plan's CompressedSize agree, and for
// int8 -- whose sizes are not symmetric in the two dimensions, and whose
// operands must be K-contiguous -- both give the K-contiguous sizes of a
// column-major A or B).
int64_t line_length(const MatImpl& m) { return m.order == CUSPARSE_ORDER_ROW ? m.cols : m.rows; }
size_t descriptor_meta_bytes(const MatImpl& m) { return meta_bytes(m.type, lines(m), line_length(m)); }
size_t descriptor_batch_bytes(const MatImpl& m) { return values_bytes(m.type, m.rows, m.cols) + descriptor_meta_bytes(m); }
// A batch stride of 0 is one matrix for every batch, so one compressed copy.
int64_t compressed_batches(const MatImpl& m) { return m.stride == 0 ? 1 : m.batches; }

// Nibble index of row r's group g in the 16-bit layout (64 rows x 32 K
// blocks of 128 16-bit words; rows padded to 32, a last 32-row tile packed).
size_t nibble16(int64_t r, int64_t g, int64_t nk) {
  const int64_t rows32 = round_up(nk, 32);
  const int64_t words_per_kblock = (rows32 / 64) * 128 + (rows32 % 64) * 2;
  const int64_t kblock = g / 8, kc = (g % 8) / 4, q = g % 4;
  const int64_t rt = r / 64, rr = r % 64;
  const int64_t w = ((rr >> 3) & 1) | (kc << 1) | (((rr >> 4) & 1) << 2) | ((rr & 7) << 3) | (((rr >> 5) & 1) << 6);
  return (size_t)((kblock * words_per_kblock + rt * 128 + w) * 4 + q);
}
// The int8 layout: 64 x 64 tiles of 256 words, rows padded to 64, row tiles
// first.
size_t nibble8(int64_t r, int64_t g, int64_t nk) {
  const int64_t row_tiles = round_up(nk, 64) / 64;
  const int64_t kt = g / 16, kc = (g % 16) / 4, q = g % 4;
  const int64_t rt = r / 64, rr = r % 64;
  const int64_t w = (kc & 1) | ((rr & 1) << 1) | ((kc >> 1) << 2) | (((rr >> 3) & 7) << 3) | (((rr >> 1) & 3) << 6);
  return (size_t)(((kt * row_tiles + rt) * 256 + w) * 4 + q);
}

// Where each 4-bit code goes. The region's size comes from the descriptor
// alone (see descriptor_meta_bytes), which for int8 can be smaller than the
// card's layout of the logical operand needs when K is not the contiguous
// dimension -- a layout no int8 product accepts; such a matrix keeps its
// codes row-major instead.
struct MetaCodec {
  int t;
  int64_t nk, k;
  size_t bytes;
  bool plain;
  MetaCodec(int t_, int64_t nk_, int64_t k_, size_t bytes_) : t(t_), nk(nk_), k(k_), bytes(bytes_), plain(false) {
    const int64_t groups = t == CUDA_R_32F ? k / 2 : k / 4;
    size_t top = 0;
    for (int64_t r = 0; r < nk; ++r)
      for (int64_t g = 0; g < groups; ++g)
        top = std::max(top, index(r, g, t == CUDA_R_32F ? 1 : 0));
    plain = top >= bytes * 2;
  }
  // fp32: the plain copy fills the first half, the blocked copy the second.
  size_t index(int64_t r, int64_t g, int copy) const {
    if (plain) return (size_t)(r * (t == CUDA_R_32F ? k / 2 : k / 4) + g) + (copy ? bytes : 0);
    if (t == CUDA_R_8I) return nibble8(r, g, nk);
    if (t == CUDA_R_32F) return copy == 0 ? (size_t)(r * (k / 2) + g) : bytes + nibble16(r, g, nk);
    return nibble16(r, g, nk);
  }
};

uint8_t get_nibble(const std::vector<uint8_t>& v, size_t n) {
  return n / 2 < v.size() ? (uint8_t)((v[n / 2] >> (4 * (n & 1))) & 15) : 0xE;
}
void set_nibble(std::vector<uint8_t>& v, size_t n, uint8_t c) {
  if (n / 2 >= v.size()) return;
  uint8_t& b = v[n / 2];
  b = (n & 1) ? (uint8_t)((b & 0x0f) | (c << 4)) : (uint8_t)((b & 0xf0) | c);
}

// The positions compression keeps in one group (see above).
void kept_positions(const float* g, int& i0, int& i1) {
  int nz[4], n = 0;
  for (int j = 0; j < 4; ++j)
    if (g[j] != 0.f || std::isnan(g[j])) nz[n++] = j;
  if (n >= 2) { i0 = nz[0]; i1 = nz[1]; }
  else if (n == 1) { i0 = nz[0] == 3 ? 2 : nz[0]; i1 = 3; }
  else { i0 = 2; i1 = 3; }
}

// ---- pruning ----

// The 90 ways to keep 2 of 4 in every row and column of a 4x4 tile, in the
// order the card prefers them when they tie: bits 4r..4r+3 hold row r's kept
// columns, rows being the tile's memory lines. The order is a linear
// extension of 2281 precedences measured on the card (tiles built so that
// chosen patterns tie for the largest L1 norm); 587 pairs of patterns never
// tie on their own, and their relative order here is unmeasured.
constexpr uint16_t kTileOrder[90] = {
    0xc3c3, 0xa5c3, 0xc3a5, 0xa5a5, 0x69c3, 0x96c3, 0xc369, 0xc396, 0x69a5, 0x96a5,
    0xa569, 0xa596, 0x5ac3, 0xc35a, 0x5aa5, 0xa55a, 0x6969, 0x9669, 0x6996, 0x9696,
    0x3cc3, 0x5a69, 0x5a96, 0x695a, 0x965a, 0xc33c, 0x3ca5, 0xa53c, 0x5a5a, 0x3c69,
    0x3c96, 0x693c, 0x963c, 0x3c5a, 0x5a3c, 0x3c3c, 0xcc33, 0xaa55, 0x6699, 0x9966,
    0xac35, 0xac53, 0xca35, 0x55aa, 0xca53, 0x6c39, 0x9c36, 0x6c93, 0x9c63, 0xc639,
    0xc936, 0x33cc, 0xc693, 0xc963, 0x5c3a, 0x5ca3, 0xc53a, 0x6a59, 0x9a56, 0xc5a3,
    0x6a95, 0x9a65, 0xa659, 0xa956, 0xa695, 0xa965, 0x3a5c, 0x3ac5, 0xa35c, 0x569a,
    0x596a, 0xa3c5, 0x56a9, 0x59a6, 0x659a, 0x956a, 0x65a9, 0x95a6, 0x369c, 0x396c,
    0x36c9, 0x39c6, 0x639c, 0x936c, 0x63c9, 0x93c6, 0x35ac, 0x35ca, 0x53ac, 0x53ca,
};

// STRIP keeps the larger magnitudes of each group, the lower position on a
// tie (measured over all 625 groups of {-2..2} and on random data in every
// type and orientation). Groups holding NaN or an infinity follow the same
// rule with NaN smallest; the card does something else with those (not
// modelled). Returns the kept positions as a bit mask.
unsigned strip_mask(const float* g, int n) {
  float mag[4];
  int idx[4];
  for (int j = 0; j < n; ++j) {
    mag[j] = std::isnan(g[j]) ? -1.f : std::fabs(g[j]);
    idx[j] = j;
  }
  std::stable_sort(idx, idx + n, [&](int a, int b) { return mag[a] > mag[b]; });
  unsigned keep = 0;
  for (int j = 0; j < n / 2; ++j) keep |= 1u << idx[j];
  return keep;
}

// TILE: the 4x4 pattern of largest L1 norm, the first in kTileOrder on a tie
// (measured on 3328 random and 4096 constructed tiles, and in every type and
// orientation, with the tile's rows the stored matrix's memory lines). For
// fp32 the tile is 2x2 and keeps its diagonal when that is strictly larger,
// the anti-diagonal otherwise (measured: the anti-diagonal on a tie).
// Returns the kept cells, bit 4i+j (2i+j for 2x2).
unsigned tile4_mask(const float t[4][4]) {
  float a[4][4];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) a[i][j] = std::isnan(t[i][j]) ? 0.f : std::fabs(t[i][j]);
  int best = 0;
  float best_sum = -1.f;
  for (int p = 0; p < 90; ++p) {
    float s = 0.f;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        if (kTileOrder[p] >> (4 * i + j) & 1) s += a[i][j];
    if (s > best_sum) { best_sum = s; best = p; }
  }
  return kTileOrder[best];
}
unsigned tile2_mask(const float t[2][2]) {
  const float d = std::fabs(t[0][0]) + std::fabs(t[1][1]), a = std::fabs(t[0][1]) + std::fabs(t[1][0]);
  return d > a ? 0x9u : 0x6u;  // bits (0,0),(1,1) or (0,1),(1,0)
}

// A whole stored matrix (one batch), as raw bytes: pruning copies the kept
// elements bit for bit and writes +0 over the rest (measured: pruned
// positions read back 0x0000, a kept -0 stays -0).
struct Dense {
  std::vector<uint8_t> raw;
  int t;
  float get(int64_t off) const { return load(raw.data() + off * type_bytes(t), t); }
  void zero(int64_t off) { std::memset(raw.data() + off * type_bytes(t), 0, type_bytes(t)); }
};

void prune_batch(const SparseView& v, int alg, Dense& m) {
  const int g = m.t == CUDA_R_32F ? 2 : 4;
  if (alg == CUSPARSELT_PRUNE_SPMMA_STRIP) {
    for (int64_t i = 0; i < v.nk; ++i)
      for (int64_t kk = 0; kk < v.k; kk += g) {
        float vals[4];
        for (int j = 0; j < g; ++j) vals[j] = m.get(v.at(i, kk + j));
        const unsigned keep = strip_mask(vals, g);
        for (int j = 0; j < g; ++j)
          if (!(keep >> j & 1)) m.zero(v.at(i, kk + j));
      }
    return;
  }
  const MatImpl& d = *v.m;
  const int64_t nl = lines(d), len = d.order == CUSPARSE_ORDER_ROW ? d.cols : d.rows;
  for (int64_t l = 0; l < nl; l += g)
    for (int64_t p = 0; p < len; p += g) {
      unsigned keep;
      if (g == 4) {
        float tile[4][4];
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 4; ++j) tile[i][j] = m.get((l + i) * d.ld + p + j);
        keep = tile4_mask(tile);
      } else {
        float tile[2][2];
        for (int i = 0; i < 2; ++i)
          for (int j = 0; j < 2; ++j) tile[i][j] = m.get((l + i) * d.ld + p + j);
        keep = tile2_mask(tile);
      }
      for (int i = 0; i < g; ++i)
        for (int j = 0; j < g; ++j)
          if (!(keep >> (g * i + j) & 1)) m.zero((l + i) * d.ld + p + j);
    }
}

bool batch_valid(const SparseView& v, const Dense& m) {
  const int g = m.t == CUDA_R_32F ? 2 : 4;
  for (int64_t i = 0; i < v.nk; ++i)
    for (int64_t kk = 0; kk < v.k; kk += g) {
      int nz = 0;
      for (int j = 0; j < g; ++j) {
        const float x = m.get(v.at(i, kk + j));
        nz += x != 0.f || std::isnan(x);
      }
      if (nz > g / 2) return false;
    }
  return true;
}

// The bytes a stored matrix of one batch spans, from its first element.
size_t span_bytes(const MatImpl& m) { return (size_t)matrix_elems(m) * type_bytes(m.type); }

bool load_batch(const MatImpl& m, const void* base, int64_t b, Dense& out) {
  out.t = m.type;
  out.raw.assign(span_bytes(m), 0);
  const uint8_t* p = static_cast<const uint8_t*>(base) + (size_t)(b * m.stride) * type_bytes(m.type);
  return read_bytes(out.raw.data(), p, out.raw.size());
}

// Writes back only the matrix's elements: the gaps between leading-dimension
// rows and between batches are left alone (measured: Prune2 leaves them).
bool store_batch(const MatImpl& m, void* base, int64_t b, const Dense& in) {
  uint8_t* p = static_cast<uint8_t*>(base) + (size_t)(b * m.stride) * type_bytes(m.type);
  const size_t es = type_bytes(m.type);
  const int64_t nl = lines(m), len = m.order == CUSPARSE_ORDER_ROW ? m.cols : m.rows;
  if (len == m.ld) return write_bytes(p, in.raw.data(), in.raw.size());
  for (int64_t l = 0; l < nl; ++l)
    if (!write_bytes(p + (size_t)(l * m.ld) * es, in.raw.data() + (size_t)(l * m.ld) * es, (size_t)len * es))
      return false;
  return true;
}

std::vector<uint8_t> compress_batch(const SparseView& v, const Dense& m) {
  const int t = m.t;
  const size_t es = type_bytes(t);
  const size_t vb = values_bytes(t, v.nk, v.k);
  std::vector<uint8_t> out(descriptor_batch_bytes(*v.m), 0);
  std::vector<uint8_t> meta(descriptor_meta_bytes(*v.m), 0xEE);
  MetaCodec mc{t, v.nk, v.k, meta.size()};
  const int64_t half = v.k / 2;
  auto value_slot = [&](int64_t i, int64_t c) -> uint8_t* {
    const int64_t off = v.k_contiguous() ? i * half + c : c * v.nk + i;
    return out.data() + (size_t)off * es;
  };
  for (int64_t i = 0; i < v.nk; ++i) {
    if (t == CUDA_R_32F) {
      for (int64_t p = 0; p < half; ++p) {
        const int64_t o0 = v.at(i, 2 * p), o1 = v.at(i, 2 * p + 1);
        const float x0 = m.get(o0);
        const int pick = (x0 != 0.f || std::isnan(x0)) ? 0 : 1;
        uint32_t bits;
        std::memcpy(&bits, m.raw.data() + (size_t)(pick ? o1 : o0) * 4, 4);
        bits = tf32_bias(bits);
        std::memcpy(value_slot(i, p), &bits, 4);
        const uint8_t code = pick ? 0xE : 0x4;
        set_nibble(meta, mc.index(i, p, 0), code);
        set_nibble(meta, mc.index(i, p, 1), code);
      }
      continue;
    }
    for (int64_t gi = 0; gi < v.k / 4; ++gi) {
      float g[4];
      int64_t off[4];
      for (int j = 0; j < 4; ++j) { off[j] = v.at(i, 4 * gi + j); g[j] = m.get(off[j]); }
      int i0, i1;
      kept_positions(g, i0, i1);
      std::memcpy(value_slot(i, 2 * gi), m.raw.data() + (size_t)off[i0] * es, es);
      std::memcpy(value_slot(i, 2 * gi + 1), m.raw.data() + (size_t)off[i1] * es, es);
      set_nibble(meta, mc.index(i, gi, 0), (uint8_t)(i0 | (i1 << 2)));
    }
  }
  std::memcpy(out.data() + vb, meta.data(), meta.size());
  return out;
}

// The logical structured operand (nk x k, K contiguous) back out of one
// batch's compressed bytes, as floats ready to multiply (tf32-truncated for
// fp32, which is what the stored half-unit bias rounds).
std::vector<float> decompress_batch(const SparseView& v, const uint8_t* c) {
  const int t = v.m->type;
  const int64_t nk = v.nk, k = v.k;
  const bool k_contiguous = v.k_contiguous();
  const size_t es = type_bytes(t);
  const size_t vb = values_bytes(t, nk, k);
  std::vector<uint8_t> meta(c + vb, c + vb + descriptor_meta_bytes(*v.m));
  MetaCodec mc{t, nk, k, meta.size()};
  const int64_t half = k / 2;
  std::vector<float> L((size_t)(nk * k), 0.f);
  auto value = [&](int64_t i, int64_t col) -> const uint8_t* {
    const int64_t off = k_contiguous ? i * half + col : col * nk + i;
    return c + (size_t)off * es;
  };
  for (int64_t i = 0; i < nk; ++i) {
    if (t == CUDA_R_32F) {
      for (int64_t p = 0; p < half; ++p) {
        const uint8_t code = get_nibble(meta, mc.index(i, p, 0));
        uint32_t bits;
        std::memcpy(&bits, value(i, p), 4);
        L[(size_t)(i * k + 2 * p + (code == 0xE ? 1 : 0))] = tf32_truncate_bits(bits);
      }
      continue;
    }
    for (int64_t gi = 0; gi < k / 4; ++gi) {
      const uint8_t code = get_nibble(meta, mc.index(i, gi, 0));
      const int i0 = code & 3, i1 = (code >> 2) & 3;
      L[(size_t)(i * k + 4 * gi + i0)] += load(value(i, 2 * gi), t);
      L[(size_t)(i * k + 4 * gi + i1)] += load(value(i, 2 * gi + 1), t);
    }
  }
  return L;
}

// ---- type combinations (measured on sm_86) ----

bool k_contiguous_dense(const MatImpl& m, bool isA, int op) {
  // A: op(A) is m x k; B: op(B) is k x n. K is contiguous when the stored
  // matrix's lines run along K.
  const bool k_along_cols = isA ? op == CUSPARSE_OPERATION_NON_TRANSPOSE : op != CUSPARSE_OPERATION_NON_TRANSPOSE;
  return k_along_cols == (m.order == CUSPARSE_ORDER_ROW);
}

// NOT_SUPPORTED reasons at MatmulDescriptorInit, empty when supported.
std::string unsupported_combination(const MatmulImpl& d) {
  const int ab = d.A.type, cd = d.C.type;
  if (d.A.type != d.B.type) return "A and B of different types";
  if (ab == kR8F_E4M3 || ab == kR8F_E5M2 || ab == kR4F_E2M1)
    return "FP8 and FP4 inputs (sm_89 and later on NVIDIA's library; not simulated)";
  if (ab == CUDA_R_16F && cd == CUDA_R_16F && (d.compute == CUSPARSE_COMPUTE_32F || d.compute == CUSPARSE_COMPUTE_16F))
    return "";
  if (ab == CUDA_R_16BF && cd == CUDA_R_16BF && d.compute == CUSPARSE_COMPUTE_32F) return "";
  if (ab == CUDA_R_32F && cd == CUDA_R_32F && d.compute == CUSPARSE_COMPUTE_32F) return "";
  if (ab == CUDA_R_8I && d.compute == CUSPARSE_COMPUTE_32I &&
      (cd == CUDA_R_8I || cd == CUDA_R_32I || cd == CUDA_R_16F || cd == CUDA_R_16BF)) {
    if (!k_contiguous_dense(d.A, true, d.opA) || !k_contiguous_dense(d.B, false, d.opB))
      return "int8 operands that are not both contiguous along K (NVIDIA's library takes only that layout)";
    return "";
  }
  return std::string("this combination of types (A/B ") + type_name(ab) + ", C/D " + type_name(cd) +
         ", compute " + std::to_string(d.compute) + ")";
}

bool same_layout(const MatImpl& a, const MatImpl& b) {
  return a.rows == b.rows && a.cols == b.cols && a.ld == b.ld && a.type == b.type && a.order == b.order;
}

Status init_descriptor(const char* api, const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* d,
                       int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, cudaDataType type,
                       cusparseOrder_t order, bool structured, int sparsity) {
  if (Status s = check_handle(api, handle)) return s;
  if (!d) return bad_arg(api, 2, "static_cast<void*>(matDescr)", "NULL pointer");
  if (rows <= 0) return bad_arg(api, 3, "rows", std::to_string(rows));
  if (cols <= 0) return bad_arg(api, 4, "cols", std::to_string(cols));
  if (order != CUSPARSE_ORDER_ROW && order != CUSPARSE_ORDER_COL)
    return bad_arg(api, 8, "order", "(cusparseOrder_t) " + std::to_string((int)order));
  const int64_t line = order == CUSPARSE_ORDER_ROW ? cols : rows;
  if (ld < line) return bad_arg(api, 5, "ld", std::to_string(ld));
  if (!descriptor_type(type)) return bad_arg(api, 7, "valueType", std::string("(cudaDataType) ") + type_name(type));
  if (structured && sparsity != CUSPARSELT_SPARSITY_50_PERCENT)
    return bad_arg(api, 9, "sparsity", "(cusparseLtSparsity_t) UNKNOWN=(cusparseLtSparsity_t) " +
                                           std::to_string(sparsity));
  // Measured: rows, cols and ld must each be a multiple of 16 bytes' worth of
  // elements (32 bytes' for a structured matrix), the alignment a positive
  // multiple of 16; NOT_SUPPORTED otherwise.
  const int64_t mult = (structured ? 256 : 128) / type_bits(type);
  if (rows % mult || cols % mult || ld % mult)
    return refuse(api, "rows, columns and leading dimension must be multiples of " + std::to_string(mult) +
                           " for this type");
  if (alignment == 0 || alignment % 16) return refuse(api, "the alignment must be a multiple of 16 bytes");
  MatImpl* m = impl<MatImpl>(d);
  std::memset(d, 0, sizeof *d);
  m->magic = kMatMagic;
  m->structured = structured;
  m->order = (uint8_t)order;
  m->type = type;
  m->alignment = alignment;
  m->batches = 1;
  m->rows = rows;
  m->cols = cols;
  m->ld = ld;
  m->stride = (order == CUSPARSE_ORDER_ROW ? rows : cols) * ld;  // measured default
  return CUSPARSE_STATUS_SUCCESS;
}

const MatImpl* mat(const cusparseLtMatDescriptor_t* d) {
  const MatImpl* m = d ? impl<MatImpl>(d) : nullptr;
  return m && m->magic == kMatMagic ? m : nullptr;
}
const MatmulImpl* matmul(const cusparseLtMatmulDescriptor_t* d) {
  const MatmulImpl* m = d ? impl<MatmulImpl>(d) : nullptr;
  return m && m->magic == kMatmulMagic ? m : nullptr;
}
const PlanImpl* plan_of(const cusparseLtMatmulPlan_t* p) {
  const PlanImpl* m = p ? impl<PlanImpl>(p) : nullptr;
  return m && m->magic == kPlanMagic ? m : nullptr;
}

// Attribute data sizes, and the message NVIDIA prints for a wrong one.
Status check_size(const char* api, size_t got, size_t want, const char* ctype) {
  if (got == want) return CUSPARSE_STATUS_SUCCESS;
  return bad_arg(api, 5, "dataSize", "expected " + std::to_string(want) + " bytes(sizeof(" + ctype +
                                         ")), current size " + std::to_string(got) + " bytes");
}

// The workspace NVIDIA's plan asks for with the default split-K, measured on
// square problems: an eighth of a byte per output element (a quarter for
// fp32), at least 256 bytes for 16-bit, 512 for fp32 and 1024 for int8. The
// simulator uses none of it.
size_t workspace_bytes(const MatmulImpl& d) {
  const int64_t mn = d.C.rows * d.C.cols;
  switch (d.A.type) {
    case CUDA_R_32F: return (size_t)std::max<int64_t>(mn / 4, 512);
    case CUDA_R_8I: return (size_t)std::max<int64_t>(mn / 8, 1024);
    default: return (size_t)std::max<int64_t>(mn / 8, 256);
  }
}


// ---- the multiply ----

struct Operands {
  const void* A;
  const void* B;
  const void* C;
  void* D;
};

// Scalars as Matmul received them: plain floats (read when the call is
// made, as the card reads host scalars), or the device vectors of the
// vector-scaling modes (read when the work runs).
struct Scalars {
  float alpha, beta;
  const void* alpha_vec;
  const void* beta_vec;
};

// op(X) of one batch of a dense operand, row-major, as floats; fp32 rounded
// to tf32 as the card rounds it.
bool load_dense_op(const MatImpl& m, int op, const void* base, int64_t b, std::vector<float>& out) {
  Dense d;
  if (!load_batch(m, base, b, d)) return false;
  const bool tr = op != CUSPARSE_OPERATION_NON_TRANSPOSE;
  const int64_t r = tr ? m.cols : m.rows, c = tr ? m.rows : m.cols;
  out.assign((size_t)(r * c), 0.f);
  for (int64_t i = 0; i < r; ++i)
    for (int64_t j = 0; j < c; ++j) {
      float x = d.get(tr ? offset_of(m, j, i) : offset_of(m, i, j));
      if (m.type == CUDA_R_32F) x = tf32_round(x);
      out[(size_t)(i * c + j)] = x;
    }
  return true;
}

float gelu(float x) {
  // The tanh form (measured: it matches the card's int8 output where the erf
  // form does not; the card's fast tanh differs in the last bits where
  // 1 + tanh cancels, x below about -2.9).
  const float u = 0.7978845608f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.f + std::tanh(u));
}

Status run_matmul(const PlanImpl& plan, const Scalars& sc, const Operands& o) {
  const MatmulImpl& d = plan.md;
  const bool sparseA = d.A.structured;
  const MatImpl& S = sparseA ? d.A : d.B;
  const SparseView sv = sparse_view(S, sparseA, sparseA ? d.opA : d.opB);
  const int64_t m = d.C.rows, n = d.C.cols, k = sv.k;
  const int t = d.A.type, ot = d.C.type;
  const bool int8 = t == CUDA_R_8I;
  const size_t cbytes = descriptor_batch_bytes(S);
  // Per-row alpha and beta: a device vector of m floats each. Measured: with
  // alpha-vector scaling and no beta vector, beta is not applied at all; a
  // beta vector without an alpha vector reads beta as a device vector (the
  // card faults on a host scalar there).
  std::vector<float> av((size_t)m, sc.alpha), bv((size_t)m, sc.beta);
  if (d.alpha_vec && !read_bytes(av.data(), sc.alpha_vec, (size_t)m * 4)) return CUSPARSE_STATUS_EXECUTION_FAILED;
  if (d.beta_vec) {
    if (!read_bytes(bv.data(), sc.beta_vec, (size_t)m * 4)) return CUSPARSE_STATUS_EXECUTION_FAILED;
  } else if (d.alpha_vec) {
    std::fill(bv.begin(), bv.end(), 0.f);
  }
  // The bias: one value per row of D, of D's type for floating-point inputs
  // and float for int8 inputs (measured: an fp16 bias for fp16, a bf16 one
  // for bf16, a float one for int8 into any output type).
  const int bias_type = int8 ? CUDA_R_32F : ot;
  std::vector<uint8_t> compressed(cbytes);
  std::vector<float> Af, Bf, bias((size_t)m, 0.f);
  std::vector<double> acc((size_t)n);
  std::vector<int64_t> iacc((size_t)n);
  for (int64_t b = 0; b < d.C.batches; ++b) {
    // The structured operand, decompressed from its batch.
    const int64_t cb = S.stride == 0 ? 0 : b;
    const uint8_t* src = static_cast<const uint8_t*>(sparseA ? o.A : o.B) + (size_t)cb * cbytes;
    if (!read_bytes(compressed.data(), src, cbytes)) return CUSPARSE_STATUS_EXECUTION_FAILED;
    std::vector<float> L = decompress_batch(sv, compressed.data());
    // op(A) as m x k and op(B) as k x n, row-major.
    if (sparseA) {
      Af.swap(L);
      if (!load_dense_op(d.B, d.opB, o.B, b, Bf)) return CUSPARSE_STATUS_EXECUTION_FAILED;
    } else {
      if (!load_dense_op(d.A, d.opA, o.A, b, Af)) return CUSPARSE_STATUS_EXECUTION_FAILED;
      Bf.assign((size_t)(k * n), 0.f);  // L is n x k
      for (int64_t j = 0; j < n; ++j)
        for (int64_t kk = 0; kk < k; ++kk) Bf[(size_t)(kk * n + j)] = L[(size_t)(j * k + kk)];
    }
    if (d.bias) {
      std::vector<uint8_t> raw((size_t)m * type_bytes(bias_type));
      const uint8_t* bp = static_cast<const uint8_t*>(d.bias) + (size_t)(b * d.bias_stride) * type_bytes(bias_type);
      if (!read_bytes(raw.data(), bp, raw.size())) return CUSPARSE_STATUS_EXECUTION_FAILED;
      for (int64_t i = 0; i < m; ++i) bias[(size_t)i] = load(raw.data() + (size_t)i * type_bytes(bias_type), bias_type);
    }
    Dense C, D;
    if (!load_batch(d.C, o.C, b, C)) return CUSPARSE_STATUS_EXECUTION_FAILED;
    if (o.D == o.C && d.C.stride == d.D.stride) D = C;
    else if (!load_batch(d.D, o.D, b, D)) return CUSPARSE_STATUS_EXECUTION_FAILED;
    const size_t oes = type_bytes(ot);
    for (int64_t i = 0; i < m; ++i) {
      const float* arow = &Af[(size_t)(i * k)];
      if (int8) {
        std::fill(iacc.begin(), iacc.end(), 0);
        for (int64_t kk = 0; kk < k; ++kk) {
          const int64_t a = (int64_t)arow[kk];
          if (!a) continue;
          const float* brow = &Bf[(size_t)(kk * n)];
          for (int64_t j = 0; j < n; ++j) iacc[(size_t)j] += a * (int64_t)brow[j];
        }
      } else {
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int64_t kk = 0; kk < k; ++kk) {
          const double a = arow[kk];
          if (a == 0.0) continue;
          const float* brow = &Bf[(size_t)(kk * n)];
          for (int64_t j = 0; j < n; ++j) acc[(size_t)j] += a * (double)brow[j];
        }
      }
      for (int64_t j = 0; j < n; ++j) {
        // int32 accumulation wraps; fp32 accumulation is rounded once here.
        const float x = int8 ? (float)(int32_t)(uint32_t)(uint64_t)iacc[(size_t)j] : (float)acc[(size_t)j];
        const float c = C.get(offset_of(d.C, i, j));
        float v = av[(size_t)i] * x;
        if (bv[(size_t)i] != 0.f) v = std::fma(bv[(size_t)i], c, v);
        v += bias[(size_t)i];
        if (d.gelu) {  // measured: with both set, GELU applies and ReLU does not
          v = d.gelu_scale * gelu(v);
        } else if (d.relu) {
          // Measured: at or below the threshold the result is a zero of the
          // value's sign (-0 for a negative value in fp16), and above it the
          // upper bound clamps.
          if (v <= d.relu_th || std::isnan(v)) v = std::copysign(0.f, v);
          else v = std::min(v, d.relu_ub);
        }
        store(D.raw.data() + (size_t)offset_of(d.D, i, j) * oes, ot, v);
      }
    }
    if (!store_batch(d.D, o.D, b, D)) return CUSPARSE_STATUS_EXECUTION_FAILED;
  }
  return CUSPARSE_STATUS_SUCCESS;
}

// The stream a call's work belongs to.
cudaStream_t first_stream(cudaStream_t* streams, int32_t n) { return n > 0 && streams ? streams[0] : nullptr; }

Status matmul_call(const char* api, const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan,
                   const void* alpha, const void* d_A, const void* d_B, const void* beta, const void* d_C,
                   void* d_D, cudaStream_t* streams, int32_t numStreams) {
  if (Status s = check_handle(api, handle)) return s;
  const PlanImpl* p = plan_of(plan);
  if (!plan) return bad_arg(api, 2, "plan", "NULL pointer");
  if (!p) return bad_arg(api, 2, "plan", "bad initialization or already destroyed");
  if (!alpha) return bad_arg(api, 3, "alpha", "NULL pointer");
  if (!d_A) return bad_arg(api, 4, "d_A", "NULL pointer");
  if (!d_B) return bad_arg(api, 5, "d_B", "NULL pointer");
  if (!beta) return bad_arg(api, 6, "beta", "NULL pointer");
  if (!d_C) return bad_arg(api, 7, "d_C", "NULL pointer");
  if (!d_D) return bad_arg(api, 8, "d_D", "NULL pointer");
  if (numStreams < 0) return bad_arg(api, 11, "numStreams", std::to_string(numStreams));
  if (numStreams > 0 && !streams) return bad_arg(api, 10, "streams", "NULL pointer");
  Scalars sc{1.f, 0.f, nullptr, nullptr};
  if (p->md.alpha_vec) sc.alpha_vec = alpha;
  else if (!read_bytes(&sc.alpha, alpha, 4)) return CUSPARSE_STATUS_EXECUTION_FAILED;
  if (p->md.beta_vec) sc.beta_vec = beta;
  else if (!p->md.alpha_vec && !read_bytes(&sc.beta, beta, 4)) return CUSPARSE_STATUS_EXECUTION_FAILED;
  const PlanImpl copy = *p;
  const Operands o{d_A, d_B, d_C, d_D};
  return in_stream_order(first_stream(streams, numStreams), [copy, sc, o] { return run_matmul(copy, sc, o); }, api);
}

// The structured descriptor a matmul descriptor prunes and compresses.
struct SparseTarget {
  MatImpl m;
  bool isA;
  int op;
};
SparseTarget target_of(const MatmulImpl& d) {
  return d.A.structured ? SparseTarget{d.A, true, d.opA} : SparseTarget{d.B, false, d.opB};
}

// Batches of a pruned or checked input: a zero stride is one matrix.
int64_t input_batches(const MatImpl& m) { return m.stride == 0 ? 1 : m.batches; }

Status prune_call(const char* api, const SparseTarget& tg, const void* d_in, void* d_out, int alg,
                  cudaStream_t stream) {
  if (!compressible_type(tg.m.type)) return refuse(api, std::string("pruning ") + type_name(tg.m.type) + " values");
  const SparseTarget g = tg;
  return in_stream_order(stream, [g, d_in, d_out, alg] {
    const SparseView v = sparse_view(g.m, g.isA, g.op);
    for (int64_t b = 0; b < input_batches(g.m); ++b) {
      Dense m;
      if (!load_batch(g.m, d_in, b, m)) return CUSPARSE_STATUS_EXECUTION_FAILED;
      prune_batch(v, alg, m);
      if (!store_batch(g.m, d_out, b, m)) return CUSPARSE_STATUS_EXECUTION_FAILED;
    }
    return CUSPARSE_STATUS_SUCCESS;
  }, api);
}

// PruneCheck writes 0 when every batch is 2:4 (1:2) along K and 1 otherwise,
// into device memory (measured).
Status check_call(const char* api, const SparseTarget& tg, const void* d_in, int* d_valid, cudaStream_t stream) {
  if (!compressible_type(tg.m.type)) return refuse(api, std::string("checking ") + type_name(tg.m.type) + " values");
  const SparseTarget g = tg;
  return in_stream_order(stream, [g, d_in, d_valid] {
    const SparseView v = sparse_view(g.m, g.isA, g.op);
    int invalid = 0;
    for (int64_t b = 0; b < input_batches(g.m) && !invalid; ++b) {
      Dense m;
      if (!load_batch(g.m, d_in, b, m)) return CUSPARSE_STATUS_EXECUTION_FAILED;
      invalid = !batch_valid(v, m);
    }
    return write_bytes(d_valid, &invalid, sizeof invalid) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
  }, api);
}

Status compress_call(const char* api, const SparseTarget& tg, const void* d_dense, void* d_compressed,
                     cudaStream_t stream) {
  if (!compressible_type(tg.m.type)) return refuse(api, std::string("compressing ") + type_name(tg.m.type) + " values");
  const SparseTarget g = tg;
  return in_stream_order(stream, [g, d_dense, d_compressed] {
    const SparseView v = sparse_view(g.m, g.isA, g.op);
    const size_t bytes = descriptor_batch_bytes(g.m);
    for (int64_t b = 0; b < compressed_batches(g.m); ++b) {
      Dense m;
      if (!load_batch(g.m, d_dense, b, m)) return CUSPARSE_STATUS_EXECUTION_FAILED;
      const std::vector<uint8_t> c = compress_batch(v, m);
      if (!write_bytes(static_cast<uint8_t*>(d_compressed) + (size_t)b * bytes, c.data(), c.size()))
        return CUSPARSE_STATUS_EXECUTION_FAILED;
    }
    return CUSPARSE_STATUS_SUCCESS;
  }, api);
}

Status sizes(const char* api, const SparseTarget& g, size_t* compressedSize, size_t* bufferSize) {
  if (!compressedSize) return bad_arg(api, 3, "compressedSize", "NULL pointer");
  if (!bufferSize) return bad_arg(api, 4, "compressBufferSize", "NULL pointer");
  if (!compressible_type(g.m.type)) return refuse(api, std::string("compressing ") + type_name(g.m.type) + " values");
  // Measured, for a descriptor used in a plan: one compressed copy per batch
  // (none extra for a zero stride). NVIDIA's CompressedSize2 counts a single
  // batch for a batched descriptor that no plan has used yet; this one counts
  // them all.
  *compressedSize = descriptor_batch_bytes(g.m) * (size_t)compressed_batches(g.m);
  *bufferSize = buffer_bytes(g.m.type, lines(g.m), line_length(g.m)) * (size_t)compressed_batches(g.m);
  return CUSPARSE_STATUS_SUCCESS;
}

int sm_of_current_device() {
  int dev = 0, major = 0, minor = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
    cudaGetLastError();
    return -1;
  }
  return major * 10 + minor;
}

}  // namespace

extern "C" {

// ---- library ----

cusparseStatus_t cusparseLtInit(cusparseLtHandle_t* handle) {
  const char* api = "cusparseLtInit";
  if (!handle) return bad_arg(api, 1, "handle", "NULL pointer");
  const int sm = sm_of_current_device();
  if (sm < 0) return CUSPARSE_STATUS_NOT_INITIALIZED;
  // cuSPARSELt's kernels need sparse tensor cores, compute capability 8.0 on
  // (the documented requirement; what NVIDIA's library returns below that was
  // not measured).
  if (sm < 80) {
    if (!quiet()) std::fprintf(stderr, "[vgpu] %s: cuSPARSELt needs compute capability 8.0 or later; the simulated device is %d.%d\n", api, sm / 10, sm % 10);
    return CUSPARSE_STATUS_ARCH_MISMATCH;
  }
  std::memset(handle, 0, sizeof *handle);
  HandleImpl* h = impl<HandleImpl>(handle);
  h->magic = kHandleMagic;
  h->sm = sm;
  return CUSPARSE_STATUS_SUCCESS;
}

// Measured: destroying twice, or a handle never initialized, is
// INVALID_VALUE. (NVIDIA's crashes on a NULL handle; this returns
// INVALID_VALUE.)
cusparseStatus_t cusparseLtDestroy(const cusparseLtHandle_t* handle) {
  if (Status s = check_handle("cusparseLtDestroy", handle)) return s;
  const_cast<HandleImpl*>(impl<HandleImpl>(handle))->magic = 0;
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseLtGetVersion(const cusparseLtHandle_t* handle, int* version) {
  const char* api = "cusparseLtGetVersion";
  if (Status s = check_handle(api, handle)) return s;
  if (!version) return bad_arg(api, 2, "version", "NULL pointer");
  *version = CUSPARSELT_VERSION;
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseLtGetProperty(libraryPropertyType propertyType, int* value) {
  const char* api = "cusparseLtGetProperty";
  if (!value) return bad_arg(api, 2, "value", "NULL pointer");
  switch (propertyType) {
    case MAJOR_VERSION: *value = CUSPARSELT_VER_MAJOR; return CUSPARSE_STATUS_SUCCESS;
    case MINOR_VERSION: *value = CUSPARSELT_VER_MINOR; return CUSPARSE_STATUS_SUCCESS;
    case PATCH_LEVEL: *value = CUSPARSELT_VER_PATCH; return CUSPARSE_STATUS_SUCCESS;
    default:
      return bad_arg(api, 1, "propertyType", "(libraryPropertyType) UNKNOWN=(libraryPropertyType) " +
                                                 std::to_string((int)propertyType));
  }
}

// The names and strings NVIDIA's library returns (measured).
const char* cusparseLtGetErrorName(cusparseStatus_t status) {
  switch ((int)status) {
    case 0: return "CUSPARSE_STATUS_SUCCESS";
    case 1: return "CUSPARSE_STATUS_NOT_INITIALIZED";
    case 2: return "CUSPARSE_STATUS_ALLOC_FAILED";
    case 3: return "CUSPARSE_STATUS_INVALID_VALUE";
    case 4: return "CUSPARSE_STATUS_ARCH_MISMATCH";
    case 5: return "CUSPARSE_STATUS_MAPPING_ERROR";
    case 6: return "CUSPARSE_STATUS_EXECUTION_FAILED";
    case 7: return "CUSPARSE_STATUS_INTERNAL_ERROR";
    case 8: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case 9: return "CUSPARSE_STATUS_ZERO_PIVOT";
    case 10: return "CUSPARSE_STATUS_NOT_SUPPORTED";
    case 11: return "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
    default: return "unrecognized error code";
  }
}

const char* cusparseLtGetErrorString(cusparseStatus_t status) {
  switch ((int)status) {
    case 0: return "success";
    case 1: return "initialization error";
    case 2: return "out of memory";
    case 3: return "invalid value";
    case 4: return "architecture mismatch";
    case 5: return "texture memory mapping error";
    case 6: return "kernel launch failure";
    case 7: return "internal error";
    case 8: return "matrix type not supported";
    case 9: return "zero pivot";
    case 10: return "operation not supported";
    case 11: return "insufficient resources";
    default: return "unrecognized error code";
  }
}

// ---- matrix descriptors ----

cusparseStatus_t cusparseLtDenseDescriptorInit(const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* matDescr,
                                               int64_t rows, int64_t cols, int64_t ld, uint32_t alignment,
                                               cudaDataType valueType, cusparseOrder_t order) {
  return init_descriptor("cusparseLtDenseDescriptorInit", handle, matDescr, rows, cols, ld, alignment, valueType,
                         order, false, 0);
}

cusparseStatus_t cusparseLtStructuredDescriptorInit(const cusparseLtHandle_t* handle,
                                                    cusparseLtMatDescriptor_t* matDescr, int64_t rows, int64_t cols,
                                                    int64_t ld, uint32_t alignment, cudaDataType valueType,
                                                    cusparseOrder_t order, cusparseLtSparsity_t sparsity) {
  return init_descriptor("cusparseLtStructuredDescriptorInit", handle, matDescr, rows, cols, ld, alignment,
                         valueType, order, true, (int)sparsity);
}

cusparseStatus_t cusparseLtMatDescriptorDestroy(const cusparseLtMatDescriptor_t* matDescr) {
  const char* api = "cusparseLtMatDescriptorDestroy";
  if (!matDescr) return bad_arg(api, 1, "matDescr", "NULL pointer");
  if (!mat(matDescr)) return bad_arg(api, 1, "matDescr", "bad initialization or already destroyed");
  const_cast<MatImpl*>(impl<MatImpl>(matDescr))->magic = 0;
  return CUSPARSE_STATUS_SUCCESS;
}

// Measured: NUM_BATCHES is an int of at least 1; BATCH_STRIDE an int64_t,
// either 0 (one matrix for every batch) or at least the matrix's span
// (rows or columns times ld); anything else INVALID_VALUE.
cusparseStatus_t cusparseLtMatDescSetAttribute(const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* matmulDescr,
                                               cusparseLtMatDescAttribute_t matAttribute, const void* data,
                                               size_t dataSize) {
  const char* api = "cusparseLtMatDescSetAttribute";
  if (Status s = check_handle(api, handle)) return s;
  MatImpl* m = matmulDescr ? impl<MatImpl>(matmulDescr) : nullptr;
  if (!m || m->magic != kMatMagic) return bad_arg(api, 2, "matDescr", m ? "bad initialization or already destroyed" : "NULL pointer");
  if (!data) return bad_arg(api, 4, "data", "NULL pointer");
  switch (matAttribute) {
    case CUSPARSELT_MAT_NUM_BATCHES: {
      if (Status s = check_size(api, dataSize, 4, "int")) return s;
      int v;
      std::memcpy(&v, data, 4);
      if (v < 1) return CUSPARSE_STATUS_INVALID_VALUE;
      m->batches = v;
      return CUSPARSE_STATUS_SUCCESS;
    }
    case CUSPARSELT_MAT_BATCH_STRIDE: {
      if (Status s = check_size(api, dataSize, 8, "int64_t")) return s;
      int64_t v;
      std::memcpy(&v, data, 8);
      if (v < 0 || (v > 0 && v < matrix_elems(*m))) return CUSPARSE_STATUS_INVALID_VALUE;
      m->stride = v;
      return CUSPARSE_STATUS_SUCCESS;
    }
    default:
      return bad_arg(api, 3, "attribute", "(cusparseLtMatmulDescAttribute_t) UNKNOWN=(cusparseLtMatDescAttribute_t) " +
                                              std::to_string((int)matAttribute));
  }
}

cusparseStatus_t cusparseLtMatDescGetAttribute(const cusparseLtHandle_t* handle,
                                               const cusparseLtMatDescriptor_t* matmulDescr,
                                               cusparseLtMatDescAttribute_t matAttribute, void* data, size_t dataSize) {
  const char* api = "cusparseLtMatDescGetAttribute";
  if (Status s = check_handle(api, handle)) return s;
  const MatImpl* m = mat(matmulDescr);
  if (!m) return bad_arg(api, 2, "matDescr", matmulDescr ? "bad initialization or already destroyed" : "NULL pointer");
  if (!data) return bad_arg(api, 4, "data", "NULL pointer");
  switch (matAttribute) {
    case CUSPARSELT_MAT_NUM_BATCHES:
      if (Status s = check_size(api, dataSize, 4, "int")) return s;
      std::memcpy(data, &m->batches, 4);
      return CUSPARSE_STATUS_SUCCESS;
    case CUSPARSELT_MAT_BATCH_STRIDE:
      if (Status s = check_size(api, dataSize, 8, "int64_t")) return s;
      std::memcpy(data, &m->stride, 8);
      return CUSPARSE_STATUS_SUCCESS;
    default:
      return bad_arg(api, 3, "attribute", "(cusparseLtMatmulDescAttribute_t) UNKNOWN=(cusparseLtMatDescAttribute_t) " +
                                              std::to_string((int)matAttribute));
  }
}

// ---- matmul descriptor ----

cusparseStatus_t cusparseLtMatmulDescriptorInit(const cusparseLtHandle_t* handle,
                                                cusparseLtMatmulDescriptor_t* matmulDescr, cusparseOperation_t opA,
                                                cusparseOperation_t opB, const cusparseLtMatDescriptor_t* matA,
                                                const cusparseLtMatDescriptor_t* matB,
                                                const cusparseLtMatDescriptor_t* matC,
                                                const cusparseLtMatDescriptor_t* matD,
                                                cusparseComputeType computeType) {
  const char* api = "cusparseLtMatmulDescriptorInit";
  if (Status s = check_handle(api, handle)) return s;
  if (!matmulDescr) return bad_arg(api, 2, "static_cast<void*>(matmulDescr)", "NULL pointer");
  auto valid_op = [](int op) {
    return op == CUSPARSE_OPERATION_NON_TRANSPOSE || op == CUSPARSE_OPERATION_TRANSPOSE;
  };
  if (!valid_op(opA)) return bad_arg(api, 3, "opA", "(cusparseOperation_t) UNKNOWN=(cusparseOperation_t) " + std::to_string((int)opA));
  if (!valid_op(opB)) return bad_arg(api, 4, "opB", "(cusparseOperation_t) UNKNOWN=(cusparseOperation_t) " + std::to_string((int)opB));
  const cusparseLtMatDescriptor_t* ds[4] = {matA, matB, matC, matD};
  const char* names[4] = {"matA", "matB", "matC", "matD"};
  for (int i = 0; i < 4; ++i) {
    if (!ds[i]) return bad_arg(api, 5 + i, names[i], "NULL pointer");
    if (!mat(ds[i])) return bad_arg(api, 5 + i, names[i], "bad initialization or already destroyed");
  }
  if (computeType != CUSPARSE_COMPUTE_32I && computeType != CUSPARSE_COMPUTE_16F && computeType != CUSPARSE_COMPUTE_32F)
    return bad_arg(api, 9, "computeType", "(cusparseComputeType) UNKNOWN=(cusparseComputeType) " + std::to_string((int)computeType));
  const MatImpl &A = *mat(matA), &B = *mat(matB), &C = *mat(matC), &D = *mat(matD);
  // Measured: exactly one of A and B structured, C and D dense; the shapes
  // agree; every operand has the same number of batches -- INVALID_VALUE
  // otherwise. D must be laid out exactly as C (NOT_SUPPORTED otherwise).
  if (A.structured == B.structured || C.structured || D.structured)
    return bad_arg(api, A.structured ? 6 : 5, A.structured ? "matB" : "matA", "exactly one of A and B must be structured");
  const int64_t am = opA == CUSPARSE_OPERATION_NON_TRANSPOSE ? A.rows : A.cols;
  const int64_t ak = opA == CUSPARSE_OPERATION_NON_TRANSPOSE ? A.cols : A.rows;
  const int64_t bk = opB == CUSPARSE_OPERATION_NON_TRANSPOSE ? B.rows : B.cols;
  const int64_t bn = opB == CUSPARSE_OPERATION_NON_TRANSPOSE ? B.cols : B.rows;
  if (ak != bk || C.rows != am || C.cols != bn || D.rows != am || D.cols != bn)
    return bad_arg(api, 7, "matC", "the dimensions of op(A), op(B), C and D do not agree");
  if (A.batches != C.batches || B.batches != C.batches || D.batches != C.batches)
    return bad_arg(api, 7, "matC", "the operands have different numbers of batches");
  if (!same_layout(C, D)) return refuse(api, "D laid out differently from C (NVIDIA's library refuses it too)");
  MatmulImpl md{};
  md.magic = kMatmulMagic;
  md.opA = opA;
  md.opB = opB;
  md.compute = computeType;
  md.A = A;
  md.B = B;
  md.C = C;
  md.D = D;
  md.relu_ub = 3.40282347e38f;  // measured defaults: FLT_MAX, 0, GELU scaling 1
  md.relu_th = 0.f;
  md.gelu_scale = 1.f;
  const std::string why = unsupported_combination(md);
  if (!why.empty()) return refuse(api, why);
  std::memset(matmulDescr, 0, sizeof *matmulDescr);
  std::memcpy(matmulDescr->data, &md, sizeof md);
  return CUSPARSE_STATUS_SUCCESS;
}

// Measured sizes, defaults and refusals: the flags and floats are 4 bytes,
// the stride 8, the pointers 8; GELU and its scaling are only for int8 into
// int8 (INVALID_VALUE otherwise); ReLU reads back as 0 or 1; a negative bias
// stride is INVALID_VALUE; reading SPARSE_MAT_POINTER back is INTERNAL_ERROR
// on NVIDIA's library, and here too.
cusparseStatus_t cusparseLtMatmulDescSetAttribute(const cusparseLtHandle_t* handle,
                                                  cusparseLtMatmulDescriptor_t* matmulDescr,
                                                  cusparseLtMatmulDescAttribute_t matmulAttribute, const void* data,
                                                  size_t dataSize) {
  const char* api = "cusparseLtMatmulDescSetAttribute";
  if (Status s = check_handle(api, handle)) return s;
  MatmulImpl* d = matmulDescr ? impl<MatmulImpl>(matmulDescr) : nullptr;
  if (!d || d->magic != kMatmulMagic)
    return bad_arg(api, 2, "matmulDescr", d ? "bad initialization or already destroyed" : "NULL pointer");
  if (!data) return bad_arg(api, 4, "data", "NULL pointer");
  const int a = (int)matmulAttribute;
  auto get_int = [&](int32_t& out) -> Status {
    if (Status s = check_size(api, dataSize, 4, "int")) return s;
    std::memcpy(&out, data, 4);
    return CUSPARSE_STATUS_SUCCESS;
  };
  auto get_float = [&](float& out) -> Status {
    if (Status s = check_size(api, dataSize, 4, "float")) return s;
    std::memcpy(&out, data, 4);
    return CUSPARSE_STATUS_SUCCESS;
  };
  auto get_ptr = [&](const void*& out) -> Status {
    if (Status s = check_size(api, dataSize, sizeof(void*), "void*")) return s;
    std::memcpy(&out, data, sizeof(void*));
    return CUSPARSE_STATUS_SUCCESS;
  };
  const bool gelu_ok = d->A.type == CUDA_R_8I && d->C.type == CUDA_R_8I;
  int32_t iv = 0;
  switch (a) {
    case CUSPARSELT_MATMUL_ACTIVATION_RELU:
      if (Status s = get_int(iv)) return s;
      d->relu = iv != 0;
      return CUSPARSE_STATUS_SUCCESS;
    case CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND: return get_float(d->relu_ub);
    case CUSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD: return get_float(d->relu_th);
    case CUSPARSELT_MATMUL_ACTIVATION_GELU:
      if (Status s = get_int(iv)) return s;
      if (!gelu_ok) return CUSPARSE_STATUS_INVALID_VALUE;
      d->gelu = iv != 0;
      return CUSPARSE_STATUS_SUCCESS;
    case CUSPARSELT_MATMUL_ACTIVATION_GELU_SCALING: {
      float f;
      if (Status s = get_float(f)) return s;
      if (!gelu_ok) return CUSPARSE_STATUS_INVALID_VALUE;
      d->gelu_scale = f;
      return CUSPARSE_STATUS_SUCCESS;
    }
    case CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING:
      if (Status s = get_int(iv)) return s;
      d->alpha_vec = iv != 0;
      return CUSPARSE_STATUS_SUCCESS;
    case CUSPARSELT_MATMUL_BETA_VECTOR_SCALING:
      if (Status s = get_int(iv)) return s;
      d->beta_vec = iv != 0;
      return CUSPARSE_STATUS_SUCCESS;
    case CUSPARSELT_MATMUL_BIAS_STRIDE: {
      if (Status s = check_size(api, dataSize, 8, "int64_t")) return s;
      int64_t v;
      std::memcpy(&v, data, 8);
      if (v < 0) return CUSPARSE_STATUS_INVALID_VALUE;
      d->bias_stride = v;
      return CUSPARSE_STATUS_SUCCESS;
    }
    case CUSPARSELT_MATMUL_BIAS_POINTER: return get_ptr(d->bias);
    case CUSPARSELT_MATMUL_SPARSE_MAT_POINTER: return get_ptr(d->sparse_ptr);
    case CUSPARSELT_MATMUL_A_SCALE_MODE: case CUSPARSELT_MATMUL_B_SCALE_MODE: case CUSPARSELT_MATMUL_C_SCALE_MODE:
    case CUSPARSELT_MATMUL_D_SCALE_MODE: case CUSPARSELT_MATMUL_D_OUT_SCALE_MODE:
      return get_int(d->scale_mode[a - CUSPARSELT_MATMUL_A_SCALE_MODE]);
    case CUSPARSELT_MATMUL_A_SCALE_POINTER: case CUSPARSELT_MATMUL_B_SCALE_POINTER: case CUSPARSELT_MATMUL_C_SCALE_POINTER:
    case CUSPARSELT_MATMUL_D_SCALE_POINTER: case CUSPARSELT_MATMUL_D_OUT_SCALE_POINTER:
      return get_ptr(d->scale_ptr[a - CUSPARSELT_MATMUL_A_SCALE_POINTER]);
    default:
      return bad_arg(api, 3, "attribute", "(cusparseLtMatmulDescAttribute_t) " + std::to_string(a));
  }
}

cusparseStatus_t cusparseLtMatmulDescGetAttribute(const cusparseLtHandle_t* handle,
                                                  const cusparseLtMatmulDescriptor_t* matmulDescr,
                                                  cusparseLtMatmulDescAttribute_t matmulAttribute, void* data,
                                                  size_t dataSize) {
  const char* api = "cusparseLtMatmulDescGetAttribute";
  if (Status s = check_handle(api, handle)) return s;
  const MatmulImpl* d = matmul(matmulDescr);
  if (!d) return bad_arg(api, 2, "matmulDescr", matmulDescr ? "bad initialization or already destroyed" : "NULL pointer");
  if (!data) return bad_arg(api, 4, "data", "NULL pointer");
  const int a = (int)matmulAttribute;
  auto put = [&](const void* v, size_t want, const char* ctype) -> Status {
    if (Status s = check_size(api, dataSize, want, ctype)) return s;
    std::memcpy(data, v, want);
    return CUSPARSE_STATUS_SUCCESS;
  };
  switch (a) {
    case CUSPARSELT_MATMUL_ACTIVATION_RELU: return put(&d->relu, 4, "int");
    case CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND: return put(&d->relu_ub, 4, "float");
    case CUSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD: return put(&d->relu_th, 4, "float");
    case CUSPARSELT_MATMUL_ACTIVATION_GELU: return put(&d->gelu, 4, "int");
    case CUSPARSELT_MATMUL_ACTIVATION_GELU_SCALING: return put(&d->gelu_scale, 4, "float");
    case CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING: return put(&d->alpha_vec, 4, "int");
    case CUSPARSELT_MATMUL_BETA_VECTOR_SCALING: return put(&d->beta_vec, 4, "int");
    case CUSPARSELT_MATMUL_BIAS_STRIDE: return put(&d->bias_stride, 8, "int64_t");
    case CUSPARSELT_MATMUL_BIAS_POINTER: return put(&d->bias, sizeof(void*), "void*");
    case CUSPARSELT_MATMUL_SPARSE_MAT_POINTER: return CUSPARSE_STATUS_INTERNAL_ERROR;
    case CUSPARSELT_MATMUL_A_SCALE_MODE: case CUSPARSELT_MATMUL_B_SCALE_MODE: case CUSPARSELT_MATMUL_C_SCALE_MODE:
    case CUSPARSELT_MATMUL_D_SCALE_MODE: case CUSPARSELT_MATMUL_D_OUT_SCALE_MODE:
      return put(&d->scale_mode[a - CUSPARSELT_MATMUL_A_SCALE_MODE], 4, "int");
    case CUSPARSELT_MATMUL_A_SCALE_POINTER: case CUSPARSELT_MATMUL_B_SCALE_POINTER: case CUSPARSELT_MATMUL_C_SCALE_POINTER:
    case CUSPARSELT_MATMUL_D_SCALE_POINTER: case CUSPARSELT_MATMUL_D_OUT_SCALE_POINTER:
      return put(&d->scale_ptr[a - CUSPARSELT_MATMUL_A_SCALE_POINTER], sizeof(void*), "void*");
    default:
      return bad_arg(api, 3, "attribute", "(cusparseLtMatmulDescAttribute_t) " + std::to_string(a));
  }
}

// ---- algorithm selection ----

// Measured defaults on sm_86, the same for every type and size tried: four
// configurations (CONFIG_MAX_ID 4), 5 search iterations, split-K 1 in the
// one-kernel mode, no split-K buffers. fp16 compute is refused here.
cusparseStatus_t cusparseLtMatmulAlgSelectionInit(const cusparseLtHandle_t* handle,
                                                  cusparseLtMatmulAlgSelection_t* algSelection,
                                                  const cusparseLtMatmulDescriptor_t* matmulDescr,
                                                  cusparseLtMatmulAlg_t alg) {
  const char* api = "cusparseLtMatmulAlgSelectionInit";
  if (Status s = check_handle(api, handle)) return s;
  if (!algSelection) return bad_arg(api, 2, "algSelection", "NULL pointer");
  const MatmulImpl* d = matmul(matmulDescr);
  if (!d) return bad_arg(api, 3, "matmulDescr", matmulDescr ? "bad initialization or already destroyed" : "NULL pointer");
  if (alg != CUSPARSELT_MATMUL_ALG_DEFAULT)
    return bad_arg(api, 4, "alg", "(cusparseLtMatmulAlg_t) UNKNOWN=(cusparseLtMatmulAlg_t) " + std::to_string((int)alg));
  // Measured: a scale mode set on an fp16 product is accepted here and by
  // the plan; the scaling modes belong to FP8/FP4, which the matmul
  // descriptor has already refused, and this library ignores them.
  if (d->compute == CUSPARSE_COMPUTE_16F) return refuse(api, "fp16 compute (NVIDIA's library has no sm_86 kernel for it)");
  std::memset(algSelection, 0, sizeof *algSelection);
  AlgImpl* a = impl<AlgImpl>(algSelection);
  a->magic = kAlgMagic;
  a->config_id = 0;
  a->max_id = 4;
  a->search_iters = 5;
  a->split_k = 1;
  a->split_k_mode = CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL;
  a->split_k_buffers = 0;
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseLtMatmulAlgSelectionDestroy(const cusparseLtMatmulAlgSelection_t* algSelection) {
  const char* api = "cusparseLtMatmulAlgSelectionDestroy";
  if (!algSelection) return bad_arg(api, 1, "alg_sel_ptr", "NULL pointer");
  const_cast<AlgImpl*>(impl<AlgImpl>(algSelection))->magic = 0;
  return CUSPARSE_STATUS_SUCCESS;
}

// Measured: CONFIG_ID in [0, CONFIG_MAX_ID), CONFIG_MAX_ID read-only, any
// iteration count, split-K at least 1, its mode one or two kernels, a
// non-negative number of buffers; every value an int.
cusparseStatus_t cusparseLtMatmulAlgSetAttribute(const cusparseLtHandle_t* handle,
                                                 cusparseLtMatmulAlgSelection_t* algSelection,
                                                 cusparseLtMatmulAlgAttribute_t attribute, const void* data,
                                                 size_t dataSize) {
  const char* api = "cusparseLtMatmulAlgSetAttribute";
  if (Status s = check_handle(api, handle)) return s;
  AlgImpl* a = algSelection ? impl<AlgImpl>(algSelection) : nullptr;
  if (!a || a->magic != kAlgMagic) return bad_arg(api, 2, "algSelection", a ? "bad initialization or already destroyed" : "NULL pointer");
  if (!data) return bad_arg(api, 4, "data", "NULL pointer");
  if ((int)attribute < 0 || (int)attribute > CUSPARSELT_MATMUL_SPLIT_K_BUFFERS)
    return bad_arg(api, 3, "attribute", "(cusparseLtMatmulAlgAttribute_t) " + std::to_string((int)attribute));
  if (Status s = check_size(api, dataSize, 4, "int")) return s;
  int v;
  std::memcpy(&v, data, 4);
  switch (attribute) {
    case CUSPARSELT_MATMUL_ALG_CONFIG_ID:
      if (v < 0 || v >= a->max_id) return CUSPARSE_STATUS_INVALID_VALUE;
      a->config_id = v;
      return CUSPARSE_STATUS_SUCCESS;
    case CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID: return CUSPARSE_STATUS_INVALID_VALUE;
    case CUSPARSELT_MATMUL_SEARCH_ITERATIONS: a->search_iters = v; return CUSPARSE_STATUS_SUCCESS;
    case CUSPARSELT_MATMUL_SPLIT_K:
      if (v < 1) return CUSPARSE_STATUS_INVALID_VALUE;
      a->split_k = v;
      return CUSPARSE_STATUS_SUCCESS;
    case CUSPARSELT_MATMUL_SPLIT_K_MODE:
      if (v != CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL && v != CUSPARSELT_SPLIT_K_MODE_TWO_KERNELS)
        return CUSPARSE_STATUS_INVALID_VALUE;
      a->split_k_mode = v;
      return CUSPARSE_STATUS_SUCCESS;
    case CUSPARSELT_MATMUL_SPLIT_K_BUFFERS:
      if (v < 0) return CUSPARSE_STATUS_INVALID_VALUE;
      a->split_k_buffers = v;
      return CUSPARSE_STATUS_SUCCESS;
  }
  return CUSPARSE_STATUS_INVALID_VALUE;
}

cusparseStatus_t cusparseLtMatmulAlgGetAttribute(const cusparseLtHandle_t* handle,
                                                 const cusparseLtMatmulAlgSelection_t* algSelection,
                                                 cusparseLtMatmulAlgAttribute_t attribute, void* data,
                                                 size_t dataSize) {
  const char* api = "cusparseLtMatmulAlgGetAttribute";
  if (Status s = check_handle(api, handle)) return s;
  const AlgImpl* a = algSelection ? impl<AlgImpl>(algSelection) : nullptr;
  if (!a || a->magic != kAlgMagic) return bad_arg(api, 2, "algSelection", a ? "bad initialization or already destroyed" : "NULL pointer");
  if (!data) return bad_arg(api, 4, "data", "NULL pointer");
  if ((int)attribute < 0 || (int)attribute > CUSPARSELT_MATMUL_SPLIT_K_BUFFERS)
    return bad_arg(api, 3, "attribute", "(cusparseLtMatmulAlgAttribute_t) " + std::to_string((int)attribute));
  if (Status s = check_size(api, dataSize, 4, "int")) return s;
  const int32_t vals[6] = {a->config_id, a->max_id, a->search_iters, a->split_k, a->split_k_mode, a->split_k_buffers};
  std::memcpy(data, &vals[(int)attribute], 4);
  return CUSPARSE_STATUS_SUCCESS;
}

// ---- plan ----

cusparseStatus_t cusparseLtMatmulGetWorkspace(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan,
                                              size_t* workspaceSize) {
  const char* api = "cusparseLtMatmulGetWorkspace";
  if (Status s = check_handle(api, handle)) return s;
  const PlanImpl* p = plan_of(plan);
  if (!p) return bad_arg(api, 2, "plan", plan ? "bad initialization or already destroyed" : "NULL pointer");
  if (!workspaceSize) return bad_arg(api, 3, "workspaceSize", "NULL pointer");
  *workspaceSize = workspace_bytes(p->md);
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseLtMatmulPlanInit(const cusparseLtHandle_t* handle, cusparseLtMatmulPlan_t* plan,
                                          const cusparseLtMatmulDescriptor_t* matmulDescr,
                                          const cusparseLtMatmulAlgSelection_t* algSelection) {
  const char* api = "cusparseLtMatmulPlanInit";
  if (Status s = check_handle(api, handle)) return s;
  if (!plan) return bad_arg(api, 2, "plan", "NULL pointer");
  const MatmulImpl* d = matmul(matmulDescr);
  if (!d) return bad_arg(api, 3, "matmulDescr", matmulDescr ? "bad initialization or already destroyed" : "NULL pointer");
  const AlgImpl* a = algSelection ? impl<AlgImpl>(algSelection) : nullptr;
  if (!a || a->magic != kAlgMagic)
    return bad_arg(api, 4, "algSelection", a ? "bad initialization or already destroyed" : "NULL pointer");
  PlanImpl p{};
  p.magic = kPlanMagic;
  p.md = *d;
  p.alg = *a;
  std::memset(plan, 0, sizeof *plan);
  std::memcpy(plan->data, &p, sizeof p);
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseLtMatmulPlanDestroy(const cusparseLtMatmulPlan_t* plan) {
  const char* api = "cusparseLtMatmulPlanDestroy";
  if (!plan) return bad_arg(api, 1, "plan", "NULL pointer");
  const_cast<PlanImpl*>(impl<PlanImpl>(plan))->magic = 0;
  return CUSPARSE_STATUS_SUCCESS;
}

// ---- execution ----

cusparseStatus_t cusparseLtMatmul(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan,
                                  const void* alpha, const void* d_A, const void* d_B, const void* beta,
                                  const void* d_C, void* d_D, void* workspace, cudaStream_t* streams,
                                  int32_t numStreams) {
  (void)workspace;  // measured: NULL is accepted
  return matmul_call("cusparseLtMatmul", handle, plan, alpha, d_A, d_B, beta, d_C, d_D, streams, numStreams);
}

// The search times every configuration and keeps the fastest in the plan.
// The simulator has one kernel, so it runs the product once and keeps the
// configuration the plan had (NVIDIA's runs it CONFIG_MAX_ID times
// (SEARCH_ITERATIONS + 1), which an in-place C == D with a nonzero beta can
// see; that is not reproduced).
cusparseStatus_t cusparseLtMatmulSearch(const cusparseLtHandle_t* handle, cusparseLtMatmulPlan_t* plan,
                                        const void* alpha, const void* d_A, const void* d_B, const void* beta,
                                        const void* d_C, void* d_D, void* workspace, cudaStream_t* streams,
                                        int32_t numStreams) {
  (void)workspace;
  return matmul_call("cusparseLtMatmulSearch", handle, plan, alpha, d_A, d_B, beta, d_C, d_D, streams, numStreams);
}

// ---- pruning ----

cusparseStatus_t cusparseLtSpMMAPrune(const cusparseLtHandle_t* handle, const cusparseLtMatmulDescriptor_t* matmulDescr,
                                      const void* d_in, void* d_out, cusparseLtPruneAlg_t pruneAlg,
                                      cudaStream_t stream) {
  const char* api = "cusparseLtSpMMAPrune";
  if (Status s = check_handle(api, handle)) return s;
  const MatmulImpl* d = matmul(matmulDescr);
  if (!d) return bad_arg(api, 2, "matmulDescr", matmulDescr ? "bad initialization or already destroyed" : "NULL pointer");
  if (!d_in) return bad_arg(api, 3, "d_in", "NULL pointer");
  if (!d_out) return bad_arg(api, 4, "d_out", "NULL pointer");
  if (pruneAlg != CUSPARSELT_PRUNE_SPMMA_TILE && pruneAlg != CUSPARSELT_PRUNE_SPMMA_STRIP)
    return bad_arg(api, 5, "pruneAlg", "(cusparseLtPruneAlg_t) UNKNOWN=(cusparseLtPruneAlg_t) " + std::to_string((int)pruneAlg));
  return prune_call(api, target_of(*d), d_in, d_out, pruneAlg, stream);
}

cusparseStatus_t cusparseLtSpMMAPruneCheck(const cusparseLtHandle_t* handle,
                                           const cusparseLtMatmulDescriptor_t* matmulDescr, const void* d_in,
                                           int* valid, cudaStream_t stream) {
  const char* api = "cusparseLtSpMMAPruneCheck";
  if (Status s = check_handle(api, handle)) return s;
  const MatmulImpl* d = matmul(matmulDescr);
  if (!d) return bad_arg(api, 2, "matmulDescr", matmulDescr ? "bad initialization or already destroyed" : "NULL pointer");
  if (!d_in) return bad_arg(api, 3, "d_in", "NULL pointer");
  if (!valid) return bad_arg(api, 4, "d_valid", "NULL pointer");
  return check_call(api, target_of(*d), d_in, valid, stream);
}

// The descriptor-only forms take any matrix descriptor (measured: a dense
// one is accepted) and any nonzero isSparseA as A.
cusparseStatus_t cusparseLtSpMMAPrune2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr,
                                       int isSparseA, cusparseOperation_t op, const void* d_in, void* d_out,
                                       cusparseLtPruneAlg_t pruneAlg, cudaStream_t stream) {
  const char* api = "cusparseLtSpMMAPrune2";
  if (Status s = check_handle(api, handle)) return s;
  const MatImpl* m = mat(sparseMatDescr);
  if (!m) return bad_arg(api, 2, "sparseMatDescr", sparseMatDescr ? "bad initialization or already destroyed" : "NULL pointer");
  if (op != CUSPARSE_OPERATION_NON_TRANSPOSE && op != CUSPARSE_OPERATION_TRANSPOSE)
    return bad_arg(api, 3, "op", "(cusparseOperation_t) UNKNOWN=(cusparseOperation_t) " + std::to_string((int)op));
  if (!d_in) return bad_arg(api, 5, "d_in", "NULL pointer");
  if (!d_out) return bad_arg(api, 6, "d_out", "NULL pointer");
  if (pruneAlg != CUSPARSELT_PRUNE_SPMMA_TILE && pruneAlg != CUSPARSELT_PRUNE_SPMMA_STRIP)
    return bad_arg(api, 7, "pruneAlg", "(cusparseLtPruneAlg_t) UNKNOWN=(cusparseLtPruneAlg_t) " + std::to_string((int)pruneAlg));
  return prune_call(api, SparseTarget{*m, isSparseA != 0, op}, d_in, d_out, pruneAlg, stream);
}

cusparseStatus_t cusparseLtSpMMAPruneCheck2(const cusparseLtHandle_t* handle,
                                            const cusparseLtMatDescriptor_t* sparseMatDescr, int isSparseA,
                                            cusparseOperation_t op, const void* d_in, int* d_valid,
                                            cudaStream_t stream) {
  const char* api = "cusparseLtSpMMAPruneCheck2";
  if (Status s = check_handle(api, handle)) return s;
  const MatImpl* m = mat(sparseMatDescr);
  if (!m) return bad_arg(api, 2, "sparseMatDescr", sparseMatDescr ? "bad initialization or already destroyed" : "NULL pointer");
  if (op != CUSPARSE_OPERATION_NON_TRANSPOSE && op != CUSPARSE_OPERATION_TRANSPOSE)
    return bad_arg(api, 3, "op", "(cusparseOperation_t) UNKNOWN=(cusparseOperation_t) " + std::to_string((int)op));
  if (!d_in) return bad_arg(api, 5, "d_in", "NULL pointer");
  if (!d_valid) return bad_arg(api, 6, "d_valid", "NULL pointer");
  return check_call(api, SparseTarget{*m, isSparseA != 0, op}, d_in, d_valid, stream);
}

// ---- compression ----

cusparseStatus_t cusparseLtSpMMACompressedSize(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan,
                                               size_t* compressedSize, size_t* compressedBufferSize) {
  const char* api = "cusparseLtSpMMACompressedSize";
  if (Status s = check_handle(api, handle)) return s;
  const PlanImpl* p = plan_of(plan);
  if (!p) return bad_arg(api, 2, "plan", plan ? "bad initialization or already destroyed" : "NULL pointer");
  return sizes(api, target_of(p->md), compressedSize, compressedBufferSize);
}

// Measured: the scratch buffer may be NULL (the simulator never uses it).
cusparseStatus_t cusparseLtSpMMACompress(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan,
                                         const void* d_dense, void* d_compressed, void* d_compressed_buffer,
                                         cudaStream_t stream) {
  const char* api = "cusparseLtSpMMACompress";
  (void)d_compressed_buffer;
  if (Status s = check_handle(api, handle)) return s;
  const PlanImpl* p = plan_of(plan);
  if (!p) return bad_arg(api, 2, "plan", plan ? "bad initialization or already destroyed" : "NULL pointer");
  if (!d_dense) return bad_arg(api, 3, "d_dense", "NULL pointer");
  if (!d_compressed) return bad_arg(api, 4, "d_compressed", "NULL pointer");
  return compress_call(api, target_of(p->md), d_dense, d_compressed, stream);
}

cusparseStatus_t cusparseLtSpMMACompressedSize2(const cusparseLtHandle_t* handle,
                                                const cusparseLtMatDescriptor_t* sparseMatDescr,
                                                size_t* compressedSize, size_t* compressedBufferSize) {
  const char* api = "cusparseLtSpMMACompressedSize2";
  if (Status s = check_handle(api, handle)) return s;
  const MatImpl* m = mat(sparseMatDescr);
  if (!m) return bad_arg(api, 2, "sparseMatDescr", sparseMatDescr ? "bad initialization or already destroyed" : "NULL pointer");
  return sizes(api, SparseTarget{*m, true, CUSPARSE_OPERATION_NON_TRANSPOSE}, compressedSize, compressedBufferSize);
}

cusparseStatus_t cusparseLtSpMMACompress2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr,
                                          int isSparseA, cusparseOperation_t op, const void* d_dense,
                                          void* d_compressed, void* d_compressed_buffer, cudaStream_t stream) {
  const char* api = "cusparseLtSpMMACompress2";
  (void)d_compressed_buffer;
  if (Status s = check_handle(api, handle)) return s;
  const MatImpl* m = mat(sparseMatDescr);
  if (!m) return bad_arg(api, 2, "sparseMatDescr", sparseMatDescr ? "bad initialization or already destroyed" : "NULL pointer");
  if (op != CUSPARSE_OPERATION_NON_TRANSPOSE && op != CUSPARSE_OPERATION_TRANSPOSE)
    return bad_arg(api, 4, "op", "(cusparseOperation_t) UNKNOWN=(cusparseOperation_t) " + std::to_string((int)op));
  if (!d_dense) return bad_arg(api, 5, "d_dense", "NULL pointer");
  if (!d_compressed) return bad_arg(api, 6, "d_compressed", "NULL pointer");
  return compress_call(api, SparseTarget{*m, isSparseA != 0, op}, d_dense, d_compressed, stream);
}

}  // extern "C"
