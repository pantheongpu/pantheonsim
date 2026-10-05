// libvgpucutensor -- VirtualGPU's cuTENSOR, presented as libcutensor.so.2.
//
// NVIDIA's libcutensor cannot run on a simulated GPU: it carries a statically
// linked CUDA runtime, which reaches the driver through NVIDIA's
// undocumented internal interface. This is the documented cuTENSOR 2 API
// (nvidia/include/vgpu_cutensor.h) implemented the way the simulator's other
// vendor libraries are: an execute call reads its operands out of simulated
// device memory, computes on the host in double precision (rounding operands
// to the precision the compute descriptor names), and writes the result back.
// Application kernels are simulated; vendor library calls are implemented.
//
// What it covers: tensor descriptors over every element type the RTX 3060
// takes, contractions (binary and trinary), reductions, permutations and
// elementwise binary and trinary operations with their unary and binary
// operators, block-sparse contractions, plan preferences, plans and their
// cache (written to and read from a file of this library's own format),
// workspace estimation, the logger, and graph capture of every execute call.
//
// What is measured and what is our own. Statuses -- which call refuses what,
// with which code -- attribute sizes and defaults, the scalar type each
// operation takes, and which type and compute combinations are accepted
// follow NVIDIA's library on an RTX 3060 (cuTENSOR 2.8.1, CUDA 13.0), where
// the documentation leaves them open; each is noted where it is decided.
// Kernel selection is not modelled: every algorithm, kernel rank and JIT mode
// computes the same way here, workspace estimates are this library's own
// (it needs none), and a plan cache entry remembers the problem, not a
// kernel.
#include "../include/vgpu_cutensor.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include "vgpu/runtime/capture.hpp"

namespace {

using cd = std::complex<double>;
using Status = cutensorStatus_t;

// ---- logging ----
//
// The library's own messages are its refusals. They go to stderr as every
// VirtualGPU library's do (VGPU_QUIET=1 silences them), and to the caller's
// logger callback or file when one is set and its level admits errors.
struct Logger {
  cutensorLoggerCallback_t callback = nullptr;
  FILE* file = nullptr;
  bool own_file = false, disabled = false;
  int level = 0, mask = 0;
};
Logger& logger() {
  static Logger* l = new Logger;
  return *l;
}
std::mutex& log_mu() {
  static std::mutex* m = new std::mutex;
  return *m;
}

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

void say(const char* api, const std::string& why) {
  if (!quiet()) std::fprintf(stderr, "[vgpu] %s: %s\n", api, why.c_str());
  std::lock_guard<std::mutex> g(log_mu());
  Logger& l = logger();
  if (l.disabled || (l.level < 1 && !(l.mask & 1))) return;
  if (l.callback) l.callback(1, api, why.c_str());
  if (l.file) std::fprintf(l.file, "[cuTENSOR][Error][%s] %s\n", api, why.c_str());
}

Status refuse(const char* api, const std::string& why) {
  say(api, why);
  return CUTENSOR_STATUS_NOT_SUPPORTED;
}

// ---- element types ----

enum class Elem { R16F, R16BF, R32F, R64F, C16F, C32F, C64F, R8I, R8U, R32I, R32U, None };

Elem elem_of(cudaDataType_t t) {
  switch (t) {
    case CUDA_R_16F: return Elem::R16F;
    case CUDA_R_16BF: return Elem::R16BF;
    case CUDA_R_32F: return Elem::R32F;
    case CUDA_R_64F: return Elem::R64F;
    case CUDA_C_16F: return Elem::C16F;
    case CUDA_C_32F: return Elem::C32F;
    case CUDA_C_64F: return Elem::C64F;
    case CUDA_R_8I: return Elem::R8I;
    case CUDA_R_8U: return Elem::R8U;
    case CUDA_R_32I: return Elem::R32I;
    case CUDA_R_32U: return Elem::R32U;
    default: return Elem::None;
  }
}

size_t elem_bytes(cudaDataType_t t) {
  switch (elem_of(t)) {
    case Elem::R8I: case Elem::R8U: return 1;
    case Elem::R16F: case Elem::R16BF: return 2;
    case Elem::R32F: case Elem::C16F: case Elem::R32I: case Elem::R32U: return 4;
    case Elem::R64F: case Elem::C32F: return 8;
    case Elem::C64F: return 16;
    default: return 0;
  }
}

bool is_complex(cudaDataType_t t) { return t == CUDA_C_16F || t == CUDA_C_32F || t == CUDA_C_64F; }
bool is_double(cudaDataType_t t) { return t == CUDA_R_64F || t == CUDA_C_64F; }

// IEEE half and bfloat16, rounded to nearest even.
double half_to_double(uint16_t h) {
  const int s = h >> 15, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  double v;
  if (e == 0) v = std::ldexp((double)m, -24);
  else if (e == 31) v = m ? NAN : INFINITY;
  else v = std::ldexp((double)(m | 0x400), e - 25);
  return s ? -v : v;
}
uint16_t float_to_half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000;
  const uint32_t ax = x & 0x7fffffff;
  if (ax >= 0x7f800000) return (uint16_t)(sign | 0x7c00 | (ax > 0x7f800000 ? 0x200 : 0));
  if (ax >= 0x477ff000) return (uint16_t)(sign | 0x7c00);  // rounds past 65504
  if (ax < 0x38800000) {                                  // subnormal half (or zero)
    const float a = std::fabs(f) * 16777216.0f;              // in units of 2^-24
    const uint32_t q = (uint32_t)std::nearbyint(a);
    return (uint16_t)(sign | q);
  }
  uint32_t m = ax + 0xfff + ((ax >> 13) & 1);  // round to nearest even at bit 13
  m -= 0x38000000;                             // rebias 127 -> 15
  return (uint16_t)(sign | (m >> 13));
}
double bf16_to_double(uint16_t b) {
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
double round_half(double v) { return half_to_double(float_to_half((float)v)); }
double round_bf16(double v) { return bf16_to_double(float_to_bf16((float)v)); }
// TensorFloat-32: ten mantissa bits, rounded to nearest with ties away from
// zero, as the hardware's cvt.rna.tf32.f32 does.
double round_tf32(double v) {
  float f = (float)v;
  uint32_t x;
  std::memcpy(&x, &f, 4);
  if ((x & 0x7f800000) != 0x7f800000) {
    x += 0x1000;
    x &= 0xffffe000u;
  }
  std::memcpy(&f, &x, 4);
  return f;
}

// ---- compute descriptors ----
//
// The nine documented ones are this library's own objects; the exported
// constants point at them.
enum class Compute { F16, BF16, TF32, X3TF32, F32, F64, X9BF16, X8INT8, X4F16 };

}  // namespace

struct cutensorComputeDescriptor {
  Compute kind;
};

namespace {
cutensorComputeDescriptor g_compute[9] = {{Compute::F16},  {Compute::BF16},   {Compute::TF32},
                                          {Compute::X3TF32}, {Compute::F32},  {Compute::F64},
                                          {Compute::X9BF16}, {Compute::X8INT8}, {Compute::X4F16}};
}  // namespace

extern "C" {
const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_16F = &g_compute[0];
const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_16BF = &g_compute[1];
const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_TF32 = &g_compute[2];
const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_3XTF32 = &g_compute[3];
const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_32F = &g_compute[4];
const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_64F = &g_compute[5];
const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_9X16BF = &g_compute[6];
const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_8XINT8 = &g_compute[7];
const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_4X16F = &g_compute[8];
}

namespace {

bool known_compute(cutensorComputeDescriptor_t c) {
  for (auto& g : g_compute)
    if (c == &g) return true;
  return false;
}

// The precision an operand is rounded to before the arithmetic.
double round_compute(Compute c, double v) {
  switch (c) {
    case Compute::F16: return round_half(v);
    case Compute::BF16: return round_bf16(v);
    case Compute::TF32: return round_tf32(v);
    case Compute::F64: case Compute::X8INT8: return v;
    default: return (double)(float)v;
  }
}
cd round_compute(Compute c, cd v) { return {round_compute(c, v.real()), round_compute(c, v.imag())}; }

}  // namespace

// ---- handle, descriptors, preferences, plans ----

struct cutensorTensorDescriptor {
  std::vector<int64_t> extent, stride;
  cudaDataType_t type = CUDA_R_32F;
  uint32_t alignment = 0;
};

struct cutensorBlockSparseTensorDescriptor {
  uint32_t modes = 0;
  std::vector<uint32_t> sections;          // per mode
  std::vector<std::vector<int64_t>> ext;   // per mode, per section
  std::vector<std::vector<int32_t>> coord; // per block, per mode
  std::vector<std::vector<int64_t>> stride;// per block, per mode
  cudaDataType_t type = CUDA_R_32F;
};

namespace {

enum class Kind { Contraction, Reduction, Permutation, Binary, Trinary, Contraction3, BlockSparse };

bool elementwise_kind(Kind k) { return k == Kind::Permutation || k == Kind::Binary || k == Kind::Trinary; }


struct Operand {
  cutensorTensorDescriptor desc;
  const cutensorTensorDescriptor* id = nullptr;  // the caller's descriptor, for identity checks
  std::vector<int32_t> modes;
  cutensorOperator_t op = CUTENSOR_OP_IDENTITY;
};

struct BlockOperand {
  cutensorBlockSparseTensorDescriptor desc;
  std::vector<int32_t> modes;
};

}  // namespace

struct cutensorOperationDescriptor {
  Kind kind = Kind::Contraction;
  // Contraction: A, B, C, D. Reduction: A, C, D. Permutation: A, B(out).
  // Binary: A, C, D. Trinary: A, B, C, D. Trinary contraction: A, B, C, D, E.
  std::vector<Operand> t;
  std::vector<BlockOperand> bs;
  cutensorOperator_t op1 = CUTENSOR_OP_ADD, op2 = CUTENSOR_OP_ADD;  // opAB/opAC/opReduce, opABC
  Compute compute = Compute::F32;
  cudaDataType_t scalar = CUDA_R_32F;
  int32_t tag = 0;
  int32_t reproducible = 0;
  std::vector<uint32_t> pad_left, pad_right;
  std::vector<uint8_t> pad_value;
};

struct cutensorPlanPreference {
  int32_t autotune = CUTENSOR_AUTOTUNE_MODE_NONE;
  int32_t cache = CUTENSOR_CACHE_MODE_PEDANTIC;
  int32_t incremental = 4;
  int32_t algo = CUTENSOR_ALGO_DEFAULT;
  int32_t kernel_rank = 0;
  int32_t jit = CUTENSOR_JIT_MODE_NONE;
  int32_t gpu_arch = -1;
};

struct cutensorPlan {
  std::shared_ptr<const cutensorOperationDescriptor> op;
  uint64_t workspace = 0;
  int device = 0;
};

namespace {

// The plan cache. A cacheline is a problem the handle has planned -- its
// operation, operands, modes and tag in a canonical text form -- kept in
// least-recently-used order. Nothing is gained by a hit here (there is no
// kernel search to skip), but the entries are real so the cache's size, its
// file and the count read back from that file behave as documented.
struct PlanCache {
  uint32_t capacity = 64;  // the documented default
  std::list<std::string> lines;
  void touch(const std::string& key) {
    if (!capacity) return;
    auto it = std::find(lines.begin(), lines.end(), key);
    if (it != lines.end()) lines.erase(it);
    lines.push_front(key);
    while (lines.size() > capacity) lines.pop_back();
  }
};

}  // namespace

struct cutensorHandle {
  int device = 0;
  PlanCache cache;
  std::mutex mu;
};

namespace {

// ---- simulated device memory ----
//
// Operands are GPU-accessible memory: device, managed or mapped host memory.
// Every access is a cudaMemcpy with cudaMemcpyDefault, so the runtime decides
// which it is and a device side goes through the simulator's bounds-checked
// memory: a copy past the end of an allocation fails the call.
bool read_bytes(void* dst, const void* src, size_t n) {
  if (!n) return true;
  if (!src || !dst) return false;
  const bool ok = cudaMemcpy(dst, src, n, cudaMemcpyDefault) == cudaSuccess;
  cudaGetLastError();
  return ok;
}

bool write_bytes(void* dst, const void* src, size_t n) {
  if (!n) return true;
  if (!dst || !src) return false;
  const bool ok = cudaMemcpy(dst, src, n, cudaMemcpyDefault) == cudaSuccess;
  cudaGetLastError();
  return ok;
}

// Elements from `first` up to and including the last one the layout touches.
size_t span(const cutensorTensorDescriptor& d) {
  size_t last = 0;
  for (size_t i = 0; i < d.extent.size(); ++i) last += (size_t)(d.extent[i] - 1) * (size_t)d.stride[i];
  return last + 1;
}

template <class V>
V load_elem(const uint8_t* p, cudaDataType_t t) {
  double re = 0, im = 0;
  switch (elem_of(t)) {
    case Elem::R16F: { uint16_t h; std::memcpy(&h, p, 2); re = half_to_double(h); break; }
    case Elem::R16BF: { uint16_t h; std::memcpy(&h, p, 2); re = bf16_to_double(h); break; }
    case Elem::R32F: { float f; std::memcpy(&f, p, 4); re = f; break; }
    case Elem::R64F: std::memcpy(&re, p, 8); break;
    case Elem::C16F: {
      uint16_t h[2];
      std::memcpy(h, p, 4);
      re = half_to_double(h[0]);
      im = half_to_double(h[1]);
      break;
    }
    case Elem::C32F: { float f[2]; std::memcpy(f, p, 8); re = f[0]; im = f[1]; break; }
    case Elem::C64F: { double f[2]; std::memcpy(f, p, 16); re = f[0]; im = f[1]; break; }
    case Elem::R8I: re = (int8_t)p[0]; break;
    case Elem::R8U: re = p[0]; break;
    case Elem::R32I: { int32_t v; std::memcpy(&v, p, 4); re = v; break; }
    case Elem::R32U: { uint32_t v; std::memcpy(&v, p, 4); re = v; break; }
    default: break;
  }
  if constexpr (std::is_same_v<V, cd>) return cd(re, im);
  else return re;
}

// Integers saturate and round to nearest even, as the device's cvt.rni.sat does.
template <class I>
I to_int(double v) {
  if (std::isnan(v)) return 0;
  v = std::nearbyint(v);
  if (v <= (double)std::numeric_limits<I>::min()) return std::numeric_limits<I>::min();
  if (v >= (double)std::numeric_limits<I>::max()) return std::numeric_limits<I>::max();
  return (I)v;
}

template <class V>
void store_elem(uint8_t* p, cudaDataType_t t, V v) {
  double re, im = 0;
  if constexpr (std::is_same_v<V, cd>) re = v.real(), im = v.imag();
  else re = v;
  switch (elem_of(t)) {
    case Elem::R16F: { uint16_t h = float_to_half((float)re); std::memcpy(p, &h, 2); break; }
    case Elem::R16BF: { uint16_t h = float_to_bf16((float)re); std::memcpy(p, &h, 2); break; }
    case Elem::R32F: { float f = (float)re; std::memcpy(p, &f, 4); break; }
    case Elem::R64F: std::memcpy(p, &re, 8); break;
    case Elem::C16F: {
      uint16_t h[2] = {float_to_half((float)re), float_to_half((float)im)};
      std::memcpy(p, h, 4);
      break;
    }
    case Elem::C32F: { float f[2] = {(float)re, (float)im}; std::memcpy(p, f, 8); break; }
    case Elem::C64F: { double f[2] = {re, im}; std::memcpy(p, f, 16); break; }
    case Elem::R8I: { int8_t x = to_int<int8_t>(re); std::memcpy(p, &x, 1); break; }
    case Elem::R8U: { uint8_t x = to_int<uint8_t>(re); std::memcpy(p, &x, 1); break; }
    case Elem::R32I: { int32_t x = to_int<int32_t>(re); std::memcpy(p, &x, 4); break; }
    case Elem::R32U: { uint32_t x = to_int<uint32_t>(re); std::memcpy(p, &x, 4); break; }
    default: break;
  }
}

// A tensor's elements, widened, indexed by element offset from its base.
template <class V>
bool load_tensor(const void* base, const cutensorTensorDescriptor& d, std::vector<V>& out) {
  const size_t n = span(d), eb = elem_bytes(d.type);
  std::vector<uint8_t> raw(n * eb);
  if (!read_bytes(raw.data(), base, raw.size())) return false;
  out.resize(n);
  for (size_t i = 0; i < n; ++i) out[i] = load_elem<V>(raw.data() + i * eb, d.type);
  return true;
}

// A scalar of the operation's scalar type, from host memory.
template <class V>
V load_scalar(const void* p, cudaDataType_t t) {
  if (!p) return V{};
  return load_elem<V>(static_cast<const uint8_t*>(p), t);
}

bool is_zero(double v) { return v == 0; }
bool is_zero(cd v) { return v == cd{}; }

// ---- operators ----

bool unary_op(cutensorOperator_t op) {
  switch (op) {
    case CUTENSOR_OP_IDENTITY: case CUTENSOR_OP_SQRT: case CUTENSOR_OP_RELU: case CUTENSOR_OP_CONJ:
    case CUTENSOR_OP_RCP: case CUTENSOR_OP_SIGMOID: case CUTENSOR_OP_TANH: case CUTENSOR_OP_EXP:
    case CUTENSOR_OP_LOG: case CUTENSOR_OP_ABS: case CUTENSOR_OP_NEG: case CUTENSOR_OP_SIN:
    case CUTENSOR_OP_COS: case CUTENSOR_OP_TAN: case CUTENSOR_OP_SINH: case CUTENSOR_OP_COSH:
    case CUTENSOR_OP_ASIN: case CUTENSOR_OP_ACOS: case CUTENSOR_OP_ATAN: case CUTENSOR_OP_ASINH:
    case CUTENSOR_OP_ACOSH: case CUTENSOR_OP_ATANH: case CUTENSOR_OP_CEIL: case CUTENSOR_OP_FLOOR:
    case CUTENSOR_OP_MISH: case CUTENSOR_OP_SWISH: case CUTENSOR_OP_SOFT_PLUS: case CUTENSOR_OP_SOFT_SIGN:
      return true;
    default: return false;
  }
}
bool binary_op(cutensorOperator_t op) {
  return op == CUTENSOR_OP_ADD || op == CUTENSOR_OP_MUL || op == CUTENSOR_OP_MAX || op == CUTENSOR_OP_MIN;
}

double apply_unary(cutensorOperator_t op, double x) {
  switch (op) {
    case CUTENSOR_OP_IDENTITY: case CUTENSOR_OP_CONJ: return x;
    case CUTENSOR_OP_SQRT: return std::sqrt(x);
    case CUTENSOR_OP_RELU: return x > 0 ? x : 0;
    case CUTENSOR_OP_RCP: return 1 / x;
    case CUTENSOR_OP_SIGMOID: return 1 / (1 + std::exp(-x));
    case CUTENSOR_OP_TANH: return std::tanh(x);
    case CUTENSOR_OP_EXP: return std::exp(x);
    case CUTENSOR_OP_LOG: return std::log(x);
    case CUTENSOR_OP_ABS: return std::fabs(x);
    case CUTENSOR_OP_NEG: return -x;
    case CUTENSOR_OP_SIN: return std::sin(x);
    case CUTENSOR_OP_COS: return std::cos(x);
    case CUTENSOR_OP_TAN: return std::tan(x);
    case CUTENSOR_OP_SINH: return std::sinh(x);
    case CUTENSOR_OP_COSH: return std::cosh(x);
    case CUTENSOR_OP_ASIN: return std::asin(x);
    case CUTENSOR_OP_ACOS: return std::acos(x);
    case CUTENSOR_OP_ATAN: return std::atan(x);
    case CUTENSOR_OP_ASINH: return std::asinh(x);
    case CUTENSOR_OP_ACOSH: return std::acosh(x);
    case CUTENSOR_OP_ATANH: return std::atanh(x);
    case CUTENSOR_OP_CEIL: return std::ceil(x);
    case CUTENSOR_OP_FLOOR: return std::floor(x);
    case CUTENSOR_OP_MISH: return x * std::tanh(std::log1p(std::exp(x)));
    case CUTENSOR_OP_SWISH: return x / (1 + std::exp(-x));
    case CUTENSOR_OP_SOFT_PLUS: return std::log1p(std::exp(x));
    case CUTENSOR_OP_SOFT_SIGN: return x / (std::fabs(x) + 1);
    default: return x;
  }
}
cd apply_unary(cutensorOperator_t op, cd x) {
  switch (op) {
    case CUTENSOR_OP_IDENTITY: return x;
    case CUTENSOR_OP_CONJ: return std::conj(x);
    default: return x;  // refused at creation; see complex_unary_ok
  }
}

// MAX and MIN propagate a NaN from either side (measured: max(NaN, -5) is
// NaN on an RTX 3060).
double apply_binary(cutensorOperator_t op, double a, double b) {
  switch (op) {
    case CUTENSOR_OP_ADD: return a + b;
    case CUTENSOR_OP_MUL: return a * b;
    case CUTENSOR_OP_MAX: return std::isnan(a) || std::isnan(b) ? NAN : (a > b ? a : b);
    case CUTENSOR_OP_MIN: return std::isnan(a) || std::isnan(b) ? NAN : (a < b ? a : b);
    default: return a + b;
  }
}

// The unary operators an elementwise operation or a reduction takes
// (measured): any of the documented ones but CONJ for real data; only the
// identity and CONJ for complex data.
bool unary_ok(cutensorOperator_t op, cudaDataType_t t) {
  if (is_complex(t)) return op == CUTENSOR_OP_IDENTITY || op == CUTENSOR_OP_CONJ;
  return unary_op(op) && op != CUTENSOR_OP_CONJ;
}
// Binary operators: ADD, MUL, MAX, MIN for real data; ADD and MUL for complex.
bool binary_ok(cutensorOperator_t op, cudaDataType_t t) {
  if (is_complex(t)) return op == CUTENSOR_OP_ADD || op == CUTENSOR_OP_MUL;
  return binary_op(op);
}
cd apply_binary(cutensorOperator_t op, cd a, cd b) {
  switch (op) {
    case CUTENSOR_OP_MUL: return a * b;
    default: return a + b;  // MAX and MIN are refused for complex types
  }
}

}  // namespace

namespace {

// ---- the arithmetic ----
//
// Every operation is one loop nest. The output's modes are walked in the
// output's order; the modes no output has (contracted or reduced) are walked
// inside, from a table of each input's offsets built once. A mode an operand
// lacks has stride 0 there, which is broadcasting.

struct Walk {
  std::vector<int64_t> ext;                  // per mode
  std::vector<std::vector<int64_t>> stride;  // [operand][mode]
};

// A mode an operand repeats is its diagonal: the strides of every occurrence
// add up. (NVIDIA's library plans contractions with a repeated mode; this is
// the Einstein-notation meaning.)
int64_t stride_of(const Operand& t, int32_t mode) {
  int64_t s = 0;
  for (size_t i = 0; i < t.modes.size(); ++i)
    if (t.modes[i] == mode) s += t.desc.stride[i];
  return s;
}

int64_t extent_of(const std::vector<const Operand*>& ts, int32_t mode) {
  for (const Operand* t : ts)
    for (size_t i = 0; i < t->modes.size(); ++i)
      if (t->modes[i] == mode) return t->desc.extent[i];
  return 1;
}

Walk make_walk(const std::vector<int32_t>& modes, const std::vector<const Operand*>& ts) {
  Walk w;
  for (int32_t m : modes) w.ext.push_back(extent_of(ts, m));
  w.stride.resize(ts.size());
  for (size_t o = 0; o < ts.size(); ++o)
    for (int32_t m : modes) w.stride[o].push_back(stride_of(*ts[o], m));
  return w;
}

// Calls f(offsets) for every index tuple of `w`, offsets per operand.
template <class F>
void walk(const Walk& w, F&& f) {
  const size_t nm = w.ext.size(), no = w.stride.size();
  for (int64_t e : w.ext)
    if (e <= 0) return;
  std::vector<int64_t> idx(nm, 0), off(no, 0);
  for (;;) {
    f(off.data());
    size_t m = 0;
    for (; m < nm; ++m) {
      if (++idx[m] < w.ext[m]) {
        for (size_t o = 0; o < no; ++o) off[o] += w.stride[o][m];
        break;
      }
      for (size_t o = 0; o < no; ++o) off[o] -= w.stride[o][m] * (w.ext[m] - 1);
      idx[m] = 0;
    }
    if (m == nm) return;
  }
}

std::vector<int32_t> modes_not_in(const std::vector<const Operand*>& ins, const std::vector<int32_t>& out) {
  std::vector<int32_t> r;
  for (const Operand* t : ins)
    for (int32_t m : t->modes)
      if (std::find(out.begin(), out.end(), m) == out.end() && std::find(r.begin(), r.end(), m) == r.end())
        r.push_back(m);
  return r;
}

// An input after its unary operator and the compute precision.
template <class V>
bool load_input(const void* p, const Operand& t, Compute c, std::vector<V>& out) {
  if (!load_tensor(p, t.desc, out)) return false;
  for (V& v : out) v = round_compute(c, apply_unary(t.op, v));
  return true;
}

// The output is read whole (its span) so that elements the layout skips are
// written back unchanged.
template <class V>
struct Output {
  const cutensorTensorDescriptor* d = nullptr;
  std::vector<uint8_t> raw;
  size_t eb = 0;
  bool open(void* p, const cutensorTensorDescriptor& desc, size_t elements = 0) {
    d = &desc;
    eb = elem_bytes(desc.type);
    raw.resize((elements ? elements : span(desc)) * eb);
    return read_bytes(raw.data(), p, raw.size());
  }
  void put(int64_t off, V v) { store_elem(raw.data() + off * eb, d->type, v); }
  bool close(void* p) { return write_bytes(p, raw.data(), raw.size()); }
};

// D = alpha * prod(inputs) summed over the inner modes + beta * C, for a
// contraction of two or three inputs. C has D's layout (cuTENSOR requires
// descC == descD).
template <class V>
Status run_contraction(const cutensorOperationDescriptor& op, const void* alpha_p,
                       const std::vector<const void*>& in, const void* beta_p, const void* C, void* D) {
  const size_t nin = in.size();
  const Operand& out = op.t[nin + 1];
  const Operand& cop = op.t[nin];
  const V alpha = load_scalar<V>(alpha_p, op.scalar), beta = load_scalar<V>(beta_p, op.scalar);
  std::vector<std::vector<V>> x(nin);
  const bool use_in = !is_zero(alpha);
  if (use_in)
    for (size_t i = 0; i < nin; ++i)
      if (!load_input(in[i], op.t[i], op.compute, x[i])) return CUTENSOR_STATUS_EXECUTION_FAILED;
  std::vector<V> c;
  if (!is_zero(beta) && !load_input(C, cop, Compute::F64, c)) return CUTENSOR_STATUS_EXECUTION_FAILED;
  Output<V> o;
  if (!o.open(D, out.desc)) return CUTENSOR_STATUS_EXECUTION_FAILED;
  std::vector<const Operand*> ins;
  for (size_t i = 0; i < nin; ++i) ins.push_back(&op.t[i]);
  const std::vector<int32_t> inner_modes = modes_not_in(ins, out.modes);
  // Offsets of every input across the inner modes, once.
  std::vector<int64_t> inner;
  if (use_in) walk(make_walk(inner_modes, ins), [&](const int64_t* off) { inner.insert(inner.end(), off, off + nin); });
  const size_t nk = use_in ? inner.size() / nin : 0;
  std::vector<const Operand*> outer_ops = ins;
  outer_ops.push_back(&out);
  walk(make_walk(out.modes, outer_ops), [&](const int64_t* off) {
    V acc{};
    if (use_in) {
      if (nin == 2) {
        const V* a = x[0].data() + off[0];
        const V* b = x[1].data() + off[1];
        for (size_t k = 0; k < nk; ++k) acc += a[inner[2 * k]] * b[inner[2 * k + 1]];
      } else {
        const V* a = x[0].data() + off[0];
        const V* b = x[1].data() + off[1];
        const V* cc = x[2].data() + off[2];
        for (size_t k = 0; k < nk; ++k) acc += a[inner[3 * k]] * b[inner[3 * k + 1]] * cc[inner[3 * k + 2]];
      }
      acc *= alpha;
    }
    if (!is_zero(beta)) acc += beta * c[off[nin]];
    o.put(off[nin], acc);
  });
  return o.close(D) ? CUTENSOR_STATUS_SUCCESS : CUTENSOR_STATUS_EXECUTION_FAILED;
}

template <class V>
V reduce_identity(cutensorOperator_t op) {
  if (op == CUTENSOR_OP_MUL) return V(1);
  if constexpr (std::is_same_v<V, double>) {
    if (op == CUTENSOR_OP_MAX) return -INFINITY;
    if (op == CUTENSOR_OP_MIN) return INFINITY;
  }
  return V{};
}

// D = alpha * opReduce over the modes C lacks of opA(A) + beta * opC(C).
template <class V>
Status run_reduction(const cutensorOperationDescriptor& op, const void* alpha_p, const void* A,
                     const void* beta_p, const void* C, void* D) {
  const Operand &a = op.t[0], &cop = op.t[1], &out = op.t[2];
  const V alpha = load_scalar<V>(alpha_p, op.scalar), beta = load_scalar<V>(beta_p, op.scalar);
  std::vector<V> x, c;
  const bool use_a = !is_zero(alpha);
  if (use_a && !load_input(A, a, op.compute, x)) return CUTENSOR_STATUS_EXECUTION_FAILED;
  if (!is_zero(beta) && !load_input(C, cop, op.compute, c)) return CUTENSOR_STATUS_EXECUTION_FAILED;
  Output<V> o;
  if (!o.open(D, out.desc)) return CUTENSOR_STATUS_EXECUTION_FAILED;
  std::vector<int64_t> inner;
  if (use_a) walk(make_walk(modes_not_in({&a}, out.modes), {&a}), [&](const int64_t* off) { inner.push_back(off[0]); });
  walk(make_walk(out.modes, {&a, &out}), [&](const int64_t* off) {
    V acc{};
    if (use_a) {
      acc = reduce_identity<V>(op.op1);
      const V* p = x.data() + off[0];
      for (int64_t k : inner) acc = apply_binary(op.op1, acc, p[k]);
      acc *= alpha;
    }
    if (!is_zero(beta)) acc += beta * c[off[1]];
    o.put(off[1], round_compute(op.compute, acc));
  });
  return o.close(D) ? CUTENSOR_STATUS_SUCCESS : CUTENSOR_STATUS_EXECUTION_FAILED;
}

// Permutation (one input), binary (two) and trinary (three) elementwise
// operations: D = opABC(opAB(alpha opA(A), beta opB(B)), gamma opC(C)).
// A zero scalar leaves its tensor unread and contributes zero.
template <class V>
Status run_elementwise(const cutensorOperationDescriptor& op, const std::vector<const void*>& scal,
                       const std::vector<const void*>& in, void* D) {
  const size_t nin = in.size();
  const Operand& out = op.t.back();
  std::vector<V> s(nin);
  std::vector<std::vector<V>> x(nin);
  for (size_t i = 0; i < nin; ++i) {
    s[i] = load_scalar<V>(scal[i], op.scalar);
    if (!is_zero(s[i]) && !load_input(in[i], op.t[i], op.compute, x[i])) return CUTENSOR_STATUS_EXECUTION_FAILED;
  }
  // Padding (measured on an RTX 3060): the output descriptor keeps the
  // unpadded extents and its strides lay out the padded buffer, which spans
  // extent + left + right along each mode from D; the result lands at the
  // left padding's offset and every other element of the padded box takes
  // the padding value (zero unless set). Where strides make the two overlap
  // the result wins.
  const size_t nm = out.modes.size();
  const bool padded = !op.pad_left.empty() || !op.pad_right.empty();
  cutensorTensorDescriptor box = out.desc;
  int64_t base_out = 0;
  for (size_t m = 0; m < nm && padded; ++m) {
    const int64_t l = op.pad_left.empty() ? 0 : op.pad_left[m], r = op.pad_right.empty() ? 0 : op.pad_right[m];
    box.extent[m] += l + r;
    base_out += l * out.desc.stride[m];
  }
  Output<V> o;
  if (!o.open(D, out.desc, span(box))) return CUTENSOR_STATUS_EXECUTION_FAILED;
  if (padded) {
    V pad{};
    if (!op.pad_value.empty()) pad = load_elem<V>(op.pad_value.data(), out.desc.type);
    Operand whole = out;
    whole.desc = box;
    walk(make_walk(out.modes, {&whole}), [&](const int64_t* off) { o.put(off[0], pad); });
  }
  std::vector<const Operand*> ops;
  for (size_t i = 0; i < nin; ++i) ops.push_back(&op.t[i]);
  ops.push_back(&out);
  Walk w = make_walk(out.modes, ops);
  walk(w, [&](const int64_t* off) {
    V v[3];
    for (size_t i = 0; i < nin; ++i) v[i] = is_zero(s[i]) ? V{} : s[i] * x[i][off[i]];
    V r = v[0];
    if (nin == 2) r = apply_binary(op.op1, v[0], v[1]);
    if (nin == 3) r = apply_binary(op.op2, apply_binary(op.op1, v[0], v[1]), v[2]);
    o.put(base_out + off[nin], round_compute(op.compute, r));
  });
  return o.close(D) ? CUTENSOR_STATUS_SUCCESS : CUTENSOR_STATUS_EXECUTION_FAILED;
}

}  // namespace

namespace {

// ---- creating operations ----

Status copy_operand(const cutensorTensorDescriptor_t d, const int32_t* modes, cutensorOperator_t op,
                    Operand& out) {
  if (!d) return CUTENSOR_STATUS_INVALID_VALUE;
  if (!modes && !d->extent.empty()) return CUTENSOR_STATUS_INVALID_VALUE;
  out.desc = *d;
  out.id = d;
  out.modes.assign(modes, modes + d->extent.size());
  out.op = op;
  return CUTENSOR_STATUS_SUCCESS;
}

bool has_mode(const Operand& t, int32_t m) {
  return std::find(t.modes.begin(), t.modes.end(), m) != t.modes.end();
}

// Every mode has one extent wherever it appears.
bool extents_agree(const std::vector<Operand>& ts) {
  for (size_t a = 0; a < ts.size(); ++a)
    for (size_t i = 0; i < ts[a].modes.size(); ++i)
      for (size_t b = a; b < ts.size(); ++b)
        for (size_t j = 0; j < ts[b].modes.size(); ++j)
          if (ts[a].modes[i] == ts[b].modes[j] && ts[a].desc.extent[i] != ts[b].desc.extent[j]) return false;
  return true;
}

bool same_layout(const cutensorTensorDescriptor& a, const cutensorTensorDescriptor& b) {
  return a.extent == b.extent && a.stride == b.stride && a.type == b.type;
}

// The scalar type, as NVIDIA's library reports it on an RTX 3060: real or
// complex after the output, double precision when the output is double or
// the compute descriptor is 64F (or 8XINT8), single otherwise.
cudaDataType_t scalar_type(cudaDataType_t out, Compute c) {
  const bool dbl = is_double(out) || c == Compute::F64 || c == Compute::X8INT8;
  if (is_complex(out)) return dbl ? CUDA_C_64F : CUDA_C_32F;
  return dbl ? CUDA_R_64F : CUDA_R_32F;
}

double elements(const cutensorTensorDescriptor& d) {
  double n = 1;
  for (int64_t e : d.extent) n *= (double)e;
  return n;
}

}  // namespace

extern "C" {

// ---- version, errors, logging ----

size_t cutensorGetVersion(void) { return CUTENSOR_VERSION; }

// The runtime this library was built against: the simulator's own.
size_t cutensorGetCudartVersion(void) {
  int v = 0;
  cudaRuntimeGetVersion(&v);
  cudaGetLastError();
  return (size_t)v;
}

const char* cutensorGetErrorString(const cutensorStatus_t error) {
  switch (error) {
    case CUTENSOR_STATUS_SUCCESS: return "CUTENSOR_STATUS_SUCCESS";
    case CUTENSOR_STATUS_NOT_INITIALIZED: return "CUTENSOR_STATUS_NOT_INITIALIZED";
    case CUTENSOR_STATUS_ALLOC_FAILED: return "CUTENSOR_STATUS_ALLOC_FAILED";
    case CUTENSOR_STATUS_INVALID_VALUE: return "CUTENSOR_STATUS_INVALID_VALUE";
    case CUTENSOR_STATUS_ARCH_MISMATCH: return "CUTENSOR_STATUS_ARCH_MISMATCH";
    case CUTENSOR_STATUS_MAPPING_ERROR: return "CUTENSOR_STATUS_MAPPING_ERROR";
    case CUTENSOR_STATUS_EXECUTION_FAILED: return "CUTENSOR_STATUS_EXECUTION_FAILED";
    case CUTENSOR_STATUS_INTERNAL_ERROR: return "CUTENSOR_STATUS_INTERNAL_ERROR";
    case CUTENSOR_STATUS_NOT_SUPPORTED: return "CUTENSOR_STATUS_NOT_SUPPORTED";
    case CUTENSOR_STATUS_LICENSE_ERROR: return "CUTENSOR_STATUS_LICENSE_ERROR";
    case CUTENSOR_STATUS_CUBLAS_ERROR: return "CUTENSOR_STATUS_CUBLAS_ERROR";
    case CUTENSOR_STATUS_CUDA_ERROR: return "CUTENSOR_STATUS_CUDA_ERROR";
    case CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE: return "CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE";
    case CUTENSOR_STATUS_INSUFFICIENT_DRIVER: return "CUTENSOR_STATUS_INSUFFICIENT_DRIVER";
    case CUTENSOR_STATUS_IO_ERROR: return "CUTENSOR_STATUS_IO_ERROR";
    default: return "<unknown>";  // as NVIDIA's library answers an unknown code
  }
}

cutensorStatus_t cutensorLoggerSetCallback(cutensorLoggerCallback_t callback) {
  std::lock_guard<std::mutex> g(log_mu());
  logger().callback = callback;
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorLoggerSetFile(FILE* file) {
  std::lock_guard<std::mutex> g(log_mu());
  Logger& l = logger();
  if (l.own_file && l.file) std::fclose(l.file);
  l.file = file;
  l.own_file = false;
  return CUTENSOR_STATUS_SUCCESS;
}

// A file that cannot be opened is INVALID_VALUE (measured).
cutensorStatus_t cutensorLoggerOpenFile(const char* logFile) {
  if (!logFile) return CUTENSOR_STATUS_INVALID_VALUE;
  FILE* f = std::fopen(logFile, "w");
  if (!f) return CUTENSOR_STATUS_INVALID_VALUE;
  std::lock_guard<std::mutex> g(log_mu());
  Logger& l = logger();
  if (l.own_file && l.file) std::fclose(l.file);
  l.file = f;
  l.own_file = true;
  return CUTENSOR_STATUS_SUCCESS;
}

// Any level from 0 up is taken; a negative one is INVALID_VALUE (measured).
cutensorStatus_t cutensorLoggerSetLevel(int32_t level) {
  if (level < 0) return CUTENSOR_STATUS_INVALID_VALUE;
  std::lock_guard<std::mutex> g(log_mu());
  logger().level = level;
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorLoggerSetMask(int32_t mask) {
  std::lock_guard<std::mutex> g(log_mu());
  logger().mask = mask;
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorLoggerForceDisable(void) {
  std::lock_guard<std::mutex> g(log_mu());
  logger().disabled = true;
  return CUTENSOR_STATUS_SUCCESS;
}

// ---- handle ----

cutensorStatus_t cutensorCreate(cutensorHandle_t* handle) {
  if (!handle) return CUTENSOR_STATUS_INVALID_VALUE;
  auto* h = new cutensorHandle;
  cudaGetDevice(&h->device);
  cudaGetLastError();
  *handle = h;
  return CUTENSOR_STATUS_SUCCESS;
}

// A null handle is a no-op success (measured).
cutensorStatus_t cutensorDestroy(cutensorHandle_t handle) {
  delete handle;
  return CUTENSOR_STATUS_SUCCESS;
}

// Resizing to zero detaches the cache; resizing empties it.
cutensorStatus_t cutensorHandleResizePlanCache(cutensorHandle_t handle, const uint32_t numEntries) {
  if (!handle) return CUTENSOR_STATUS_INVALID_VALUE;
  std::lock_guard<std::mutex> g(handle->mu);
  handle->cache.capacity = numEntries;
  handle->cache.lines.clear();
  return CUTENSOR_STATUS_SUCCESS;
}

}  // extern "C"

namespace {

// The plan cache file. Our own format: a header naming this library and its
// version, the line count, then each line as a length and its bytes. An
// empty or foreign file is IO_ERROR, as NVIDIA's library answers one on an
// RTX 3060.
const char kCacheMagic[] = "VirtualGPU cuTENSOR plan cache 2.8\n";

bool write_u32(FILE* f, uint32_t v) { return std::fwrite(&v, 4, 1, f) == 1; }
bool read_u32(FILE* f, uint32_t& v) { return std::fread(&v, 4, 1, f) == 1; }

}  // namespace

extern "C" {

cutensorStatus_t cutensorHandleWritePlanCacheToFile(const cutensorHandle_t handle, const char filename[]) {
  if (!handle || !filename) return CUTENSOR_STATUS_INVALID_VALUE;
  std::lock_guard<std::mutex> g(handle->mu);
  if (handle->cache.capacity == 0) return CUTENSOR_STATUS_INVALID_VALUE;  // no cache attached (measured)
  FILE* f = std::fopen(filename, "wb");
  if (!f) return CUTENSOR_STATUS_IO_ERROR;
  bool ok = std::fwrite(kCacheMagic, sizeof kCacheMagic - 1, 1, f) == 1 &&
            write_u32(f, (uint32_t)handle->cache.lines.size());
  for (auto it = handle->cache.lines.rbegin(); ok && it != handle->cache.lines.rend(); ++it)
    ok = write_u32(f, (uint32_t)it->size()) && (it->empty() || std::fwrite(it->data(), it->size(), 1, f) == 1);
  ok = std::fclose(f) == 0 && ok;
  return ok ? CUTENSOR_STATUS_SUCCESS : CUTENSOR_STATUS_IO_ERROR;
}

cutensorStatus_t cutensorHandleReadPlanCacheFromFile(cutensorHandle_t handle, const char filename[],
                                                     uint32_t* numCachelinesRead) {
  if (!handle || !filename || !numCachelinesRead) return CUTENSOR_STATUS_INVALID_VALUE;
  {
    std::lock_guard<std::mutex> g(handle->mu);
    if (handle->cache.capacity == 0) return CUTENSOR_STATUS_INVALID_VALUE;  // no cache attached (measured)
  }
  FILE* f = std::fopen(filename, "rb");
  if (!f) return CUTENSOR_STATUS_IO_ERROR;
  std::vector<char> magic(sizeof kCacheMagic - 1);
  uint32_t n = 0;
  std::vector<std::string> lines;
  bool ok = std::fread(magic.data(), magic.size(), 1, f) == 1 &&
            std::memcmp(magic.data(), kCacheMagic, magic.size()) == 0 && read_u32(f, n);
  for (uint32_t i = 0; ok && i < n; ++i) {
    uint32_t len = 0;
    ok = read_u32(f, len) && len < (1u << 24);
    std::string s(ok ? len : 0, '\0');
    ok = ok && (len == 0 || std::fread(&s[0], len, 1, f) == 1);
    lines.push_back(std::move(s));
  }
  std::fclose(f);
  if (!ok) return CUTENSOR_STATUS_IO_ERROR;  // an empty or foreign file (measured)
  std::lock_guard<std::mutex> g(handle->mu);
  if (n > handle->cache.capacity) {
    *numCachelinesRead = n;
    return CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE;
  }
  handle->cache.lines.clear();
  for (auto& s : lines) handle->cache.touch(s);
  *numCachelinesRead = n;
  return CUTENSOR_STATUS_SUCCESS;
}

// There is no just-in-time compilation, so the kernel cache is always empty:
// writing it writes an empty file (NVIDIA's library succeeds the same way with
// nothing compiled) and reading an empty file adds nothing. Anything else is
// not a cache this library wrote: INTERNAL_ERROR, as NVIDIA's library
// answers a foreign file (measured).
cutensorStatus_t cutensorWriteKernelCacheToFile(const cutensorHandle_t handle, const char filename[]) {
  if (!handle || !filename) return CUTENSOR_STATUS_INVALID_VALUE;
  FILE* f = std::fopen(filename, "wb");
  if (!f) return CUTENSOR_STATUS_IO_ERROR;
  std::fclose(f);
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorReadKernelCacheFromFile(cutensorHandle_t handle, const char filename[]) {
  if (!handle || !filename) return CUTENSOR_STATUS_INVALID_VALUE;
  FILE* f = std::fopen(filename, "rb");
  if (!f) return CUTENSOR_STATUS_IO_ERROR;
  const bool empty = std::fgetc(f) == EOF;
  std::fclose(f);
  return empty ? CUTENSOR_STATUS_SUCCESS : CUTENSOR_STATUS_INTERNAL_ERROR;
}

// ---- tensor descriptors ----
//
// Measured on an RTX 3060: extents must be positive and strides
// non-negative (zero is taken); null strides mean the packed column-major
// layout; 64 or more modes are NOT_SUPPORTED; the alignment is 0 or a power
// of two no smaller than the element; the complex integer types are
// NOT_SUPPORTED and types outside the list below INVALID_VALUE.
cutensorStatus_t cutensorCreateTensorDescriptor(const cutensorHandle_t handle, cutensorTensorDescriptor_t* desc,
                                                const uint32_t numModes, const int64_t extent[],
                                                const int64_t stride[], cudaDataType_t dataType,
                                                uint32_t alignmentRequirement) {
  if (!handle || !desc) return CUTENSOR_STATUS_INVALID_VALUE;
  if (numModes && !extent) return CUTENSOR_STATUS_INVALID_VALUE;
  if (dataType == CUDA_C_8I || dataType == CUDA_C_8U || dataType == CUDA_C_32I || dataType == CUDA_C_32U)
    return CUTENSOR_STATUS_NOT_SUPPORTED;
  if (elem_of(dataType) == Elem::None) return CUTENSOR_STATUS_INVALID_VALUE;
  if (numModes >= 64) return CUTENSOR_STATUS_NOT_SUPPORTED;
  for (uint32_t i = 0; i < numModes; ++i) {
    if (extent[i] <= 0) return CUTENSOR_STATUS_INVALID_VALUE;
    if (stride && stride[i] < 0) return CUTENSOR_STATUS_INVALID_VALUE;
  }
  const uint32_t al = alignmentRequirement;
  if (al != 0 && ((al & (al - 1)) != 0 || al < elem_bytes(dataType))) return CUTENSOR_STATUS_INVALID_VALUE;
  auto* d = new cutensorTensorDescriptor;
  d->type = dataType;
  d->alignment = al;
  // Packed strides, multiplied in unsigned arithmetic: the product past the
  // last mode is never used and may not fit an int64_t (63 modes of 2).
  uint64_t packed = 1;
  for (uint32_t i = 0; i < numModes; ++i) {
    d->extent.push_back(extent[i]);
    d->stride.push_back(stride ? stride[i] : (int64_t)packed);
    packed *= (uint64_t)extent[i];
  }
  *desc = d;
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorDestroyTensorDescriptor(cutensorTensorDescriptor_t desc) {
  delete desc;
  return CUTENSOR_STATUS_SUCCESS;
}

}  // extern "C"

namespace {

// ---- which combinations the RTX 3060 takes ----
//
// Measured with NVIDIA's cuTENSOR 2.8.1 by creating every operation over
// every pairing of R16F, R16BF, R32F, R64F, C16F, C32F, C64F, R8I and R32I
// with each of the nine compute descriptors and planning it. Contractions,
// trinary contractions and reductions take one type throughout (or R64F
// mixed with C64F in a contraction) from R16F, R16BF, R32F, C32F with any
// compute descriptor but 64F and 8XINT8, and R64F, C64F with any of the
// nine. The 3060 takes the Hopper and Blackwell descriptors (9X16BF, 4X16F,
// 8XINT8) for these; they compute at single precision (8XINT8 at double).
// Elementwise operations and permutations take the pairs listed below.

bool reduction_like_ok(cudaDataType_t t, Compute c) {
  switch (t) {
    case CUDA_R_16F: case CUDA_R_16BF: case CUDA_R_32F: case CUDA_C_32F:
      return c != Compute::F64 && c != Compute::X8INT8;
    case CUDA_R_64F: case CUDA_C_64F: return true;
    default: return false;
  }
}

bool odd_type(cudaDataType_t t) { return t == CUDA_C_16F || t == CUDA_R_32I || t == CUDA_R_32U; }

// (input, output, compute) for permutations and binary elementwise
// operations; trinary ones take fewer (the input pair is A and B, which
// share a type).
struct EwRule {
  cudaDataType_t in, out;
  Compute c;
  bool trinary;
};
const EwRule kElementwise[] = {
    {CUDA_R_16F, CUDA_R_16F, Compute::F16, true},   {CUDA_R_16F, CUDA_R_16F, Compute::F32, true},
    {CUDA_R_16F, CUDA_R_32F, Compute::F32, false},  {CUDA_R_16BF, CUDA_R_16BF, Compute::BF16, true},
    {CUDA_R_16BF, CUDA_R_16BF, Compute::F32, true}, {CUDA_R_32F, CUDA_R_16F, Compute::F32, true},
    {CUDA_R_32F, CUDA_R_32F, Compute::F32, true},   {CUDA_R_32F, CUDA_R_64F, Compute::F64, false},
    {CUDA_R_64F, CUDA_R_32F, Compute::F64, true},   {CUDA_R_64F, CUDA_R_64F, Compute::F64, true},
    {CUDA_C_32F, CUDA_C_32F, Compute::F32, true},   {CUDA_C_32F, CUDA_C_64F, Compute::F64, false},
    {CUDA_C_64F, CUDA_C_32F, Compute::F64, true},   {CUDA_C_64F, CUDA_C_64F, Compute::F64, true},
};

bool elementwise_ok(const cutensorOperationDescriptor& op) {
  const cudaDataType_t out = op.t.back().desc.type;
  const size_t nin = op.kind == Kind::Permutation ? 1 : op.kind == Kind::Binary ? 2 : 3;
  // In a binary operation C has D's type; in a trinary one B has A's.
  if (op.kind == Kind::Trinary && op.t[1].desc.type != op.t[0].desc.type) return false;
  if (nin >= 2 && op.t[nin - 1].desc.type != out) return false;
  const cudaDataType_t in = op.t[0].desc.type;
  for (const EwRule& r : kElementwise)
    if (r.in == in && r.out == out && r.c == op.compute && (r.trinary || op.kind != Kind::Trinary)) return true;
  return false;
}

// The status planning an operation gets on the RTX 3060.
Status plan_support(const cutensorOperationDescriptor& op) {
  switch (op.kind) {
    case Kind::Permutation: case Kind::Binary: case Kind::Trinary:
      return elementwise_ok(op) ? CUTENSOR_STATUS_SUCCESS : CUTENSOR_STATUS_NOT_SUPPORTED;
    case Kind::BlockSparse: return CUTENSOR_STATUS_SUCCESS;  // checked at creation
    default: break;
  }
  for (const Operand& t : op.t)
    if (odd_type(t.desc.type)) return CUTENSOR_STATUS_INTERNAL_ERROR;  // measured: C16F, R32I
  const cudaDataType_t out = op.t.back().desc.type;
  if (op.kind == Kind::Contraction && op.t[0].desc.type != op.t[1].desc.type) {
    const cudaDataType_t a = op.t[0].desc.type, b = op.t[1].desc.type;
    const bool mixed = out == CUDA_C_64F && ((a == CUDA_R_64F && b == CUDA_C_64F) || (a == CUDA_C_64F && b == CUDA_R_64F));
    return mixed ? CUTENSOR_STATUS_SUCCESS : CUTENSOR_STATUS_NOT_SUPPORTED;
  }
  for (const Operand& t : op.t)
    if (t.desc.type != out) return CUTENSOR_STATUS_NOT_SUPPORTED;
  return reduction_like_ok(out, op.compute) ? CUTENSOR_STATUS_SUCCESS : CUTENSOR_STATUS_NOT_SUPPORTED;
}

// What an elementwise operation's creation refuses outright (measured): the
// TF32 descriptors are INVALID_VALUE, the Hopper and Blackwell ones
// NOT_SUPPORTED, 16BF over complex data INVALID_VALUE, and C16F or R32I
// operands INTERNAL_ERROR.
Status elementwise_create_check(const cutensorOperationDescriptor& op) {
  if (op.compute == Compute::TF32 || op.compute == Compute::X3TF32) return CUTENSOR_STATUS_INVALID_VALUE;
  if (op.compute == Compute::X9BF16 || op.compute == Compute::X8INT8 || op.compute == Compute::X4F16)
    return CUTENSOR_STATUS_NOT_SUPPORTED;
  for (const Operand& t : op.t)
    if (is_complex(t.desc.type) && op.compute == Compute::BF16) return CUTENSOR_STATUS_INVALID_VALUE;
  for (const Operand& t : op.t)
    if (odd_type(t.desc.type)) return CUTENSOR_STATUS_INTERNAL_ERROR;
  return CUTENSOR_STATUS_SUCCESS;
}

// An elementwise operation's scalars have the compute descriptor's type.
cudaDataType_t elementwise_scalar(cudaDataType_t out, Compute c) {
  const bool cx = is_complex(out);
  switch (c) {
    case Compute::F16: return cx ? CUDA_C_16F : CUDA_R_16F;
    case Compute::BF16: return CUDA_R_16BF;
    case Compute::F64: return cx ? CUDA_C_64F : CUDA_R_64F;
    default: return cx ? CUDA_C_32F : CUDA_R_32F;
  }
}

Status common_checks(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc,
                     const cutensorComputeDescriptor_t descCompute) {
  if (!handle || !desc || !descCompute) return CUTENSOR_STATUS_INVALID_VALUE;
  if (!known_compute(descCompute)) return CUTENSOR_STATUS_INVALID_VALUE;
  return CUTENSOR_STATUS_SUCCESS;
}

// Contraction-style operations: every output mode must come from an input
// ("Mode m only occurs in the output": INVALID_VALUE); C and D must be the
// same modes and layout (NOT_SUPPORTED, "current limitation"); opC must be
// the identity and opA, opB the identity or, for complex data, CONJ
// (NOT_SUPPORTED). Measured on an RTX 3060.
Status contraction_checks(const cutensorOperationDescriptor& op, size_t nin) {
  // A contraction refuses an operand whose alignment is left at 0
  // ("Alignment requirement (0) is not met", INVALID_VALUE); other
  // operations take it (measured).
  for (const Operand& t : op.t)
    if (t.desc.alignment == 0) return CUTENSOR_STATUS_INVALID_VALUE;
  const Operand &c = op.t[nin], &d = op.t[nin + 1];
  if (c.modes != d.modes || !same_layout(c.desc, d.desc)) return CUTENSOR_STATUS_NOT_SUPPORTED;
  if (c.op != CUTENSOR_OP_IDENTITY) return CUTENSOR_STATUS_NOT_SUPPORTED;
  for (size_t i = 0; i < nin; ++i) {
    const cutensorOperator_t o = op.t[i].op;
    if (o != CUTENSOR_OP_IDENTITY && !(o == CUTENSOR_OP_CONJ && is_complex(op.t[i].desc.type)))
      return CUTENSOR_STATUS_NOT_SUPPORTED;
  }
  for (int32_t m : d.modes) {
    bool found = false;
    for (size_t i = 0; i < nin; ++i) found = found || has_mode(op.t[i], m);
    if (!found) return CUTENSOR_STATUS_INVALID_VALUE;
  }
  return CUTENSOR_STATUS_SUCCESS;
}

// Elementwise operations and permutations (measured): an operator that does
// not apply to the data is INVALID_VALUE; C and D of a binary or trinary
// operation must have the same modes and layout (NOT_SUPPORTED, "current
// limitation"); then the type and compute checks above.
Status elementwise_checks(const cutensorOperationDescriptor& op) {
  const size_t nin = op.t.size() - 1;
  for (size_t i = 0; i < nin; ++i)
    if (!unary_ok(op.t[i].op, op.t[i].desc.type)) return CUTENSOR_STATUS_INVALID_VALUE;
  const cudaDataType_t out = op.t.back().desc.type;
  if (op.kind == Kind::Binary && !binary_ok(op.op1, out)) return CUTENSOR_STATUS_INVALID_VALUE;
  if (op.kind == Kind::Trinary && (!binary_ok(op.op1, out) || !binary_ok(op.op2, out)))
    return CUTENSOR_STATUS_INVALID_VALUE;
  if (nin >= 2) {
    const Operand &c = op.t[nin - 1], &d = op.t[nin];
    if (!same_layout(c.desc, d.desc)) return CUTENSOR_STATUS_NOT_SUPPORTED;
    if (c.modes != d.modes) return CUTENSOR_STATUS_NOT_SUPPORTED;
  }
  return elementwise_create_check(op);
}

// Reductions (measured): the reduction operator must be ADD, MUL, MAX or
// MIN (ADD or MUL for complex data), else NOT_SUPPORTED.
Status reduction_create_check(const cutensorOperationDescriptor& op) {
  const cudaDataType_t out = op.t.back().desc.type;
  if (!binary_ok(op.op1, out)) return CUTENSOR_STATUS_NOT_SUPPORTED;
  for (size_t i = 0; i < 2; ++i)
    if (!unary_ok(op.t[i].op, op.t[i].desc.type)) return CUTENSOR_STATUS_INVALID_VALUE;
  const Operand &c = op.t[1], &d = op.t[2];
  if (!same_layout(c.desc, d.desc) || c.modes != d.modes) return CUTENSOR_STATUS_NOT_SUPPORTED;
  return CUTENSOR_STATUS_SUCCESS;
}

// What planning checks before the type support (measured): every mode has
// one extent wherever it appears -- INTERNAL_ERROR for a contraction,
// INVALID_VALUE otherwise -- and a reduction's output modes all come from A
// (NOT_SUPPORTED: broadcasting is not supported there).
Status plan_checks(const cutensorOperationDescriptor& op) {
  if (op.kind == Kind::BlockSparse) return CUTENSOR_STATUS_SUCCESS;
  const std::vector<Operand>& ts = op.t;
  if (!extents_agree(ts))
    return op.kind == Kind::Contraction || op.kind == Kind::Contraction3 ? CUTENSOR_STATUS_INTERNAL_ERROR
                                                                         : CUTENSOR_STATUS_INVALID_VALUE;
  if (op.kind == Kind::Reduction)
    for (int32_t m : op.t[2].modes)
      if (!has_mode(op.t[0], m)) return CUTENSOR_STATUS_NOT_SUPPORTED;
  // An elementwise input with a mode its output lacks: a binary or trinary
  // operation is INTERNAL_ERROR on an RTX 3060; a permutation plans there and
  // writes zeros, which this library refuses to imitate (NOT_SUPPORTED).
  if (elementwise_kind(op.kind))
    for (size_t i = 0; i + 1 < op.t.size(); ++i)
      for (int32_t m : op.t[i].modes)
        if (!has_mode(op.t.back(), m)) {
          if (op.kind != Kind::Permutation) return CUTENSOR_STATUS_INTERNAL_ERROR;
          return refuse("cutensorCreatePlan", "the permutation's input has a mode its output lacks");
        }
  return CUTENSOR_STATUS_SUCCESS;
}

Status finish_create(cutensorOperationDescriptor* op, cutensorOperationDescriptor_t* desc) {
  *desc = op;
  return CUTENSOR_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

cutensorStatus_t cutensorCreateContraction(
    const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA,
    const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[],
    cutensorOperator_t opB, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC,
    const cutensorTensorDescriptor_t descD, const int32_t modeD[], const cutensorComputeDescriptor_t descCompute) {
  if (Status s = common_checks(handle, desc, descCompute)) return s;
  auto op = std::make_unique<cutensorOperationDescriptor>();
  op->kind = Kind::Contraction;
  op->compute = descCompute->kind;
  op->t.resize(4);
  if (Status s = copy_operand(descA, modeA, opA, op->t[0])) return s;
  if (Status s = copy_operand(descB, modeB, opB, op->t[1])) return s;
  if (Status s = copy_operand(descC, modeC, opC, op->t[2])) return s;
  if (Status s = copy_operand(descD, modeD, CUTENSOR_OP_IDENTITY, op->t[3])) return s;
  if (Status s = contraction_checks(*op, 2)) return s;
  if (op->compute == Compute::X8INT8 && !is_double(descD->type)) return CUTENSOR_STATUS_INTERNAL_ERROR;
  op->scalar = scalar_type(descD->type, op->compute);
  return finish_create(op.release(), desc);
}

cutensorStatus_t cutensorCreateContractionTrinary(
    const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA,
    const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[],
    cutensorOperator_t opB, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC,
    const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opD,
    const cutensorTensorDescriptor_t descE, const int32_t modeE[], const cutensorComputeDescriptor_t descCompute) {
  if (Status s = common_checks(handle, desc, descCompute)) return s;
  auto op = std::make_unique<cutensorOperationDescriptor>();
  op->kind = Kind::Contraction3;
  op->compute = descCompute->kind;
  op->t.resize(5);
  if (Status s = copy_operand(descA, modeA, opA, op->t[0])) return s;
  if (Status s = copy_operand(descB, modeB, opB, op->t[1])) return s;
  if (Status s = copy_operand(descC, modeC, opC, op->t[2])) return s;
  if (Status s = copy_operand(descD, modeD, opD, op->t[3])) return s;
  if (Status s = copy_operand(descE, modeE, CUTENSOR_OP_IDENTITY, op->t[4])) return s;
  if (Status s = contraction_checks(*op, 3)) return s;
  if (op->compute == Compute::X8INT8 && !is_double(descE->type)) return CUTENSOR_STATUS_INTERNAL_ERROR;
  op->scalar = scalar_type(descE->type, op->compute);
  return finish_create(op.release(), desc);
}

cutensorStatus_t cutensorCreateReduction(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc,
                                         const cutensorTensorDescriptor_t descA, const int32_t modeA[],
                                         cutensorOperator_t opA, const cutensorTensorDescriptor_t descC,
                                         const int32_t modeC[], cutensorOperator_t opC,
                                         const cutensorTensorDescriptor_t descD, const int32_t modeD[],
                                         cutensorOperator_t opReduce, const cutensorComputeDescriptor_t descCompute) {
  if (Status s = common_checks(handle, desc, descCompute)) return s;
  auto op = std::make_unique<cutensorOperationDescriptor>();
  op->kind = Kind::Reduction;
  op->compute = descCompute->kind;
  op->op1 = opReduce;
  op->t.resize(3);
  if (Status s = copy_operand(descA, modeA, opA, op->t[0])) return s;
  if (Status s = copy_operand(descC, modeC, opC, op->t[1])) return s;
  if (Status s = copy_operand(descD, modeD, CUTENSOR_OP_IDENTITY, op->t[2])) return s;
  if (Status s = reduction_create_check(*op)) return s;
  if (op->compute == Compute::X8INT8 && !is_double(descD->type)) return CUTENSOR_STATUS_INTERNAL_ERROR;
  op->scalar = scalar_type(descD->type, op->compute);
  return finish_create(op.release(), desc);
}

cutensorStatus_t cutensorCreatePermutation(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc,
                                           const cutensorTensorDescriptor_t descA, const int32_t modeA[],
                                           cutensorOperator_t opA, const cutensorTensorDescriptor_t descB,
                                           const int32_t modeB[], const cutensorComputeDescriptor_t descCompute) {
  if (Status s = common_checks(handle, desc, descCompute)) return s;
  auto op = std::make_unique<cutensorOperationDescriptor>();
  op->kind = Kind::Permutation;
  op->compute = descCompute->kind;
  op->t.resize(2);
  if (Status s = copy_operand(descA, modeA, opA, op->t[0])) return s;
  if (Status s = copy_operand(descB, modeB, CUTENSOR_OP_IDENTITY, op->t[1])) return s;
  if (Status s = elementwise_checks(*op)) return s;
  op->scalar = elementwise_scalar(descB->type, op->compute);
  return finish_create(op.release(), desc);
}

cutensorStatus_t cutensorCreateElementwiseBinary(
    const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA,
    const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descC, const int32_t modeC[],
    cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opAC,
    const cutensorComputeDescriptor_t descCompute) {
  if (Status s = common_checks(handle, desc, descCompute)) return s;
  auto op = std::make_unique<cutensorOperationDescriptor>();
  op->kind = Kind::Binary;
  op->compute = descCompute->kind;
  op->op1 = opAC;
  op->t.resize(3);
  if (Status s = copy_operand(descA, modeA, opA, op->t[0])) return s;
  if (Status s = copy_operand(descC, modeC, opC, op->t[1])) return s;
  if (Status s = copy_operand(descD, modeD, CUTENSOR_OP_IDENTITY, op->t[2])) return s;
  if (Status s = elementwise_checks(*op)) return s;
  op->scalar = elementwise_scalar(descD->type, op->compute);
  return finish_create(op.release(), desc);
}

cutensorStatus_t cutensorCreateElementwiseTrinary(
    const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA,
    const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[],
    cutensorOperator_t opB, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC,
    const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opAB, cutensorOperator_t opABC,
    const cutensorComputeDescriptor_t descCompute) {
  if (Status s = common_checks(handle, desc, descCompute)) return s;
  auto op = std::make_unique<cutensorOperationDescriptor>();
  op->kind = Kind::Trinary;
  op->compute = descCompute->kind;
  op->op1 = opAB;
  op->op2 = opABC;
  op->t.resize(4);
  if (Status s = copy_operand(descA, modeA, opA, op->t[0])) return s;
  if (Status s = copy_operand(descB, modeB, opB, op->t[1])) return s;
  if (Status s = copy_operand(descC, modeC, opC, op->t[2])) return s;
  if (Status s = copy_operand(descD, modeD, CUTENSOR_OP_IDENTITY, op->t[3])) return s;
  if (Status s = elementwise_checks(*op)) return s;
  op->scalar = elementwise_scalar(descD->type, op->compute);
  return finish_create(op.release(), desc);
}

cutensorStatus_t cutensorDestroyOperationDescriptor(cutensorOperationDescriptor_t desc) {
  delete desc;
  return CUTENSOR_STATUS_SUCCESS;
}

}  // extern "C"

namespace {

// ---- operation attributes ----

// FLOPS and MOVED_BYTES as NVIDIA's library reports them (measured on an
// RTX 3060): a contraction costs 2 flops per multiply-add over every mode
// (8 for complex), and moves every operand once, C read and D written.
double mode_volume(const std::vector<int32_t>& modes, const std::vector<const Operand*>& ts) {
  double n = 1;
  for (int32_t m : modes) n *= (double)extent_of(ts, m);
  return n;
}

std::vector<int32_t> mode_union(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  std::vector<int32_t> u = a;
  for (int32_t m : b)
    if (std::find(u.begin(), u.end(), m) == u.end()) u.push_back(m);
  return u;
}

// Multiply-adds count 2 flops with real operands, 4 with one complex and 8
// with two (measured: R64F x C64F counts 4 per multiply-add).
double pair_factor(cudaDataType_t a, cudaDataType_t b) { return 2.0 * (is_complex(a) ? 2 : 1) * (is_complex(b) ? 2 : 1); }

float op_flops(const cutensorOperationDescriptor& op) {
  std::vector<const Operand*> ts;
  for (const Operand& t : op.t) ts.push_back(&t);
  std::vector<int32_t> all;
  for (const Operand& t : op.t) all = mode_union(all, t.modes);
  const double n = mode_volume(all, ts);
  const bool cx = is_complex(op.t.back().desc.type);
  switch (op.kind) {
    case Kind::Contraction:
      return (float)(n * pair_factor(op.t[0].desc.type, op.t[1].desc.type));
    case Kind::Contraction3: {
      // As two pairwise contractions in the cheapest order (measured: m4 k6
      // l3 n5 counts 2*4*6*3 + 2*4*3*5 = 264).
      const Operand* in[3] = {&op.t[0], &op.t[1], &op.t[2]};
      const std::vector<int32_t>& out = op.t[4].modes;
      double best = -1;
      for (int skip = 2; skip >= 0; --skip) {
        const Operand *x = in[(skip + 1) % 3], *y = in[(skip + 2) % 3], *z = in[skip];
        std::vector<int32_t> keep;
        for (int32_t m : mode_union(x->modes, y->modes))
          if (std::find(out.begin(), out.end(), m) != out.end() ||
              std::find(z->modes.begin(), z->modes.end(), m) != z->modes.end())
            keep.push_back(m);
        const cudaDataType_t xy = is_complex(x->desc.type) || is_complex(y->desc.type) ? CUDA_C_32F : CUDA_R_32F;
        const double f = mode_volume(mode_union(x->modes, y->modes), ts) * pair_factor(x->desc.type, y->desc.type) +
                         mode_volume(mode_union(keep, z->modes), ts) * pair_factor(xy, z->desc.type);
        if (best < 0 || f < best) best = f;
      }
      return (float)best;
    }
    default: return (float)(n * (cx ? 4 : 2));  // 2 per element (4 complex), measured
  }
}

float op_moved_bytes(const cutensorOperationDescriptor& op) {
  double b = 0;
  for (const Operand& t : op.t) b += elements(t.desc) * (double)elem_bytes(t.desc.type);
  return (float)b;
}

}  // namespace

extern "C" {

// Sizes must be exact (measured). The padding attributes belong to
// elementwise operations and permutations only.
cutensorStatus_t cutensorOperationDescriptorSetAttribute(const cutensorHandle_t handle,
                                                         cutensorOperationDescriptor_t desc,
                                                         cutensorOperationDescriptorAttribute_t attr,
                                                         const void* buf, size_t sizeInBytes) {
  if (!handle || !desc || !buf) return CUTENSOR_STATUS_INVALID_VALUE;
  const size_t nmodes = desc->t.empty() ? 0 : desc->t.back().modes.size();
  switch (attr) {
    case CUTENSOR_OPERATION_DESCRIPTOR_TAG:
      if (sizeInBytes != 4) return CUTENSOR_STATUS_INVALID_VALUE;
      std::memcpy(&desc->tag, buf, 4);
      return CUTENSOR_STATUS_SUCCESS;
    case CUTENSOR_OPERATION_DESCRIPTOR_BLOCKSPARSE_REPRODUCIBLE:
      if (desc->kind != Kind::BlockSparse || sizeInBytes != 4) return CUTENSOR_STATUS_INVALID_VALUE;
      std::memcpy(&desc->reproducible, buf, 4);
      return CUTENSOR_STATUS_SUCCESS;
    case CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT:
    case CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT: {
      if (!elementwise_kind(desc->kind) || sizeInBytes != nmodes * 4) return CUTENSOR_STATUS_INVALID_VALUE;
      auto& v = attr == CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT ? desc->pad_left : desc->pad_right;
      v.resize(nmodes);
      std::memcpy(v.data(), buf, sizeInBytes);
      return CUTENSOR_STATUS_SUCCESS;
    }
    case CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE:
      if (!elementwise_kind(desc->kind) || sizeInBytes != elem_bytes(desc->t.back().desc.type))
        return CUTENSOR_STATUS_INVALID_VALUE;
      desc->pad_value.assign((const uint8_t*)buf, (const uint8_t*)buf + sizeInBytes);
      return CUTENSOR_STATUS_SUCCESS;
    default: return CUTENSOR_STATUS_INVALID_VALUE;  // the rest are read-only (measured)
  }
}

cutensorStatus_t cutensorOperationDescriptorGetAttribute(const cutensorHandle_t handle,
                                                         cutensorOperationDescriptor_t desc,
                                                         cutensorOperationDescriptorAttribute_t attr, void* buf,
                                                         size_t sizeInBytes) {
  if (!handle || !desc || !buf) return CUTENSOR_STATUS_INVALID_VALUE;
  const size_t nmodes = desc->t.empty() ? 0 : desc->t.back().modes.size();
  switch (attr) {
    case CUTENSOR_OPERATION_DESCRIPTOR_TAG:
      if (sizeInBytes != 4) return CUTENSOR_STATUS_INVALID_VALUE;
      std::memcpy(buf, &desc->tag, 4);
      return CUTENSOR_STATUS_SUCCESS;
    case CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE:
      if (sizeInBytes != 4) return CUTENSOR_STATUS_INVALID_VALUE;
      std::memcpy(buf, &desc->scalar, 4);
      return CUTENSOR_STATUS_SUCCESS;
    case CUTENSOR_OPERATION_DESCRIPTOR_FLOPS:
    case CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES: {
      if (sizeInBytes != 4) return CUTENSOR_STATUS_INVALID_VALUE;
      if (desc->kind == Kind::BlockSparse && attr == CUTENSOR_OPERATION_DESCRIPTOR_FLOPS)
        return CUTENSOR_STATUS_NOT_SUPPORTED;  // measured
      const float v = attr == CUTENSOR_OPERATION_DESCRIPTOR_FLOPS ? op_flops(*desc) : op_moved_bytes(*desc);
      std::memcpy(buf, &v, 4);
      return CUTENSOR_STATUS_SUCCESS;
    }
    case CUTENSOR_OPERATION_DESCRIPTOR_BLOCKSPARSE_REPRODUCIBLE:
      if (desc->kind != Kind::BlockSparse || sizeInBytes != 4) return CUTENSOR_STATUS_INVALID_VALUE;
      std::memcpy(buf, &desc->reproducible, 4);
      return CUTENSOR_STATUS_SUCCESS;
    case CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT:
    case CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT: {
      if (!elementwise_kind(desc->kind) || sizeInBytes != nmodes * 4) return CUTENSOR_STATUS_INVALID_VALUE;
      const auto& v = attr == CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT ? desc->pad_left : desc->pad_right;
      if (v.empty()) std::memset(buf, 0, sizeInBytes);
      else std::memcpy(buf, v.data(), sizeInBytes);
      return CUTENSOR_STATUS_SUCCESS;
    }
    case CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE:
      if (!elementwise_kind(desc->kind) || sizeInBytes != elem_bytes(desc->t.back().desc.type))
        return CUTENSOR_STATUS_INVALID_VALUE;
      if (desc->pad_value.empty()) std::memset(buf, 0, sizeInBytes);
      else std::memcpy(buf, desc->pad_value.data(), sizeInBytes);
      return CUTENSOR_STATUS_SUCCESS;
    default:
      return CUTENSOR_STATUS_SUCCESS;  // an unknown attribute reads as success, nothing written (measured)
  }
}

// ---- plan preferences ----
//
// Defaults and accepted values measured on an RTX 3060: autotune mode 0,
// cache mode 1 (PEDANTIC), incremental count 4, algorithm -1, kernel rank 0,
// JIT off, GPU architecture -1 (the device's). Every attribute is an int32_t
// and the size must be exactly 4.

cutensorStatus_t cutensorCreatePlanPreference(const cutensorHandle_t handle, cutensorPlanPreference_t* pref,
                                              cutensorAlgo_t algo, cutensorJitMode_t jitMode) {
  if (!handle || !pref) return CUTENSOR_STATUS_INVALID_VALUE;
  if (jitMode != CUTENSOR_JIT_MODE_NONE && jitMode != CUTENSOR_JIT_MODE_DEFAULT) return CUTENSOR_STATUS_INVALID_VALUE;
  if ((int)algo >= 0) return CUTENSOR_STATUS_NOT_SUPPORTED;  // a specific GEMM-like kernel (measured)
  if (algo != -7 && algo != -6 && algo != -4 && algo != -3 && algo != -2 && algo != -1)
    return CUTENSOR_STATUS_INVALID_VALUE;
  auto* p = new cutensorPlanPreference;
  p->algo = algo;
  p->jit = jitMode;
  *pref = p;
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorDestroyPlanPreference(cutensorPlanPreference_t pref) {
  delete pref;
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorPlanPreferenceGetAttribute(const cutensorHandle_t handle, cutensorPlanPreference_t pref,
                                                    cutensorPlanPreferenceAttribute_t attr, void* buf,
                                                    size_t sizeInBytes) {
  if (!handle || !pref || !buf || sizeInBytes != 4) return CUTENSOR_STATUS_INVALID_VALUE;
  int32_t v;
  switch (attr) {
    case CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE: v = pref->autotune; break;
    case CUTENSOR_PLAN_PREFERENCE_CACHE_MODE: v = pref->cache; break;
    case CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT: v = pref->incremental; break;
    case CUTENSOR_PLAN_PREFERENCE_ALGO: v = pref->algo; break;
    case CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK: v = pref->kernel_rank; break;
    case CUTENSOR_PLAN_PREFERENCE_JIT: v = pref->jit; break;
    case CUTENSOR_PLAN_PREFERENCE_GPU_ARCH: v = pref->gpu_arch; break;
    default: return CUTENSOR_STATUS_INVALID_VALUE;
  }
  std::memcpy(buf, &v, 4);
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorPlanPreferenceSetAttribute(const cutensorHandle_t handle, cutensorPlanPreference_t pref,
                                                    cutensorPlanPreferenceAttribute_t attr, const void* buf,
                                                    size_t sizeInBytes) {
  if (!handle || !pref || !buf || sizeInBytes != 4) return CUTENSOR_STATUS_INVALID_VALUE;
  int32_t v;
  std::memcpy(&v, buf, 4);
  switch (attr) {
    case CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE:
      if (v < 0) return CUTENSOR_STATUS_INVALID_VALUE;
      pref->autotune = v;
      return CUTENSOR_STATUS_SUCCESS;
    case CUTENSOR_PLAN_PREFERENCE_CACHE_MODE:
      if (v < 0) return CUTENSOR_STATUS_INVALID_VALUE;
      pref->cache = v;
      return CUTENSOR_STATUS_SUCCESS;
    case CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT:
      if (v < 1) return CUTENSOR_STATUS_INVALID_VALUE;
      pref->incremental = v;
      return CUTENSOR_STATUS_SUCCESS;
    case CUTENSOR_PLAN_PREFERENCE_ALGO:
      if (v != -7 && v != -6 && v != -4 && v != -3 && v != -2 && v != -1) return CUTENSOR_STATUS_INVALID_VALUE;
      pref->algo = v;
      return CUTENSOR_STATUS_SUCCESS;
    case CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK:
      if (v < 0) return CUTENSOR_STATUS_INVALID_VALUE;
      pref->kernel_rank = v;
      return CUTENSOR_STATUS_SUCCESS;
    case CUTENSOR_PLAN_PREFERENCE_JIT:
      if (v != 0 && v != 1) return CUTENSOR_STATUS_INVALID_VALUE;
      pref->jit = v;
      return CUTENSOR_STATUS_SUCCESS;
    case CUTENSOR_PLAN_PREFERENCE_GPU_ARCH:
      if (v != 80 && v != 90 && v != 100) return CUTENSOR_STATUS_INVALID_VALUE;
      pref->gpu_arch = v;
      return CUTENSOR_STATUS_SUCCESS;
    default: return CUTENSOR_STATUS_INVALID_VALUE;
  }
}

// ---- workspace and plans ----
//
// Nothing here needs a workspace, so every estimate and every plan's
// requirement is zero; a caller's workspace is accepted and left alone.
cutensorStatus_t cutensorEstimateWorkspaceSize(const cutensorHandle_t handle,
                                               const cutensorOperationDescriptor_t desc,
                                               const cutensorPlanPreference_t planPref,
                                               const cutensorWorksizePreference_t workspacePref,
                                               uint64_t* workspaceSizeEstimate) {
  if (!handle || !desc || !planPref || !workspaceSizeEstimate) return CUTENSOR_STATUS_INVALID_VALUE;
  if (workspacePref != CUTENSOR_WORKSPACE_MIN && workspacePref != CUTENSOR_WORKSPACE_DEFAULT &&
      workspacePref != CUTENSOR_WORKSPACE_MAX)
    return CUTENSOR_STATUS_INVALID_VALUE;
  *workspaceSizeEstimate = 0;
  return CUTENSOR_STATUS_SUCCESS;
}

}  // extern "C"

namespace {

// A problem's identity for the plan cache.
std::string cache_key(const cutensorOperationDescriptor& op) {
  std::string k = std::to_string((int)op.kind) + "|" + std::to_string((int)op.compute) + "|" +
                  std::to_string(op.tag) + "|" + std::to_string((int)op.op1) + "," + std::to_string((int)op.op2);
  for (const Operand& t : op.t) {
    k += "|" + std::to_string((int)t.desc.type) + ":" + std::to_string((int)t.op) + ":";
    for (size_t i = 0; i < t.modes.size(); ++i)
      k += std::to_string(t.modes[i]) + "/" + std::to_string(t.desc.extent[i]) + "/" +
           std::to_string(t.desc.stride[i]) + ";";
  }
  return k;
}

}  // namespace

extern "C" {

cutensorStatus_t cutensorCreatePlan(const cutensorHandle_t handle, cutensorPlan_t* plan,
                                    const cutensorOperationDescriptor_t desc, const cutensorPlanPreference_t pref,
                                    uint64_t workspaceSizeLimit) {
  (void)workspaceSizeLimit;
  if (!handle || !plan || !desc || !pref) return CUTENSOR_STATUS_INVALID_VALUE;  // a null pref too (measured)
  if (desc->kind == Kind::BlockSparse)
    return refuse("cutensorCreatePlan", "block-sparse contractions are not implemented (an RTX 3060 cannot plan "
                                        "them either)");
  if (Status s = plan_checks(*desc)) return s;
  if (Status s = plan_support(*desc)) return s;
  auto* p = new cutensorPlan;
  p->op = std::make_shared<const cutensorOperationDescriptor>(*desc);
  p->workspace = 0;
  p->device = handle->device;
  if (pref->cache != CUTENSOR_CACHE_MODE_NONE) {
    std::lock_guard<std::mutex> g(handle->mu);
    handle->cache.touch(cache_key(*desc));
  }
  *plan = p;
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorDestroyPlan(cutensorPlan_t plan) {
  delete plan;
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorPlanGetAttribute(const cutensorHandle_t handle, const cutensorPlan_t plan,
                                          cutensorPlanAttribute_t attr, void* buf, size_t sizeInBytes) {
  if (!handle || !plan || !buf) return CUTENSOR_STATUS_INVALID_VALUE;
  if (attr != CUTENSOR_PLAN_REQUIRED_WORKSPACE || sizeInBytes != 8) return CUTENSOR_STATUS_INVALID_VALUE;
  std::memcpy(buf, &plan->workspace, 8);
  return CUTENSOR_STATUS_SUCCESS;
}

}  // extern "C"

namespace {

// ---- executing ----

bool capturing(cudaStream_t s) {
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  const bool on = cudaStreamIsCapturing(s, &st) == cudaSuccess && st == cudaStreamCaptureStatusActive;
  cudaGetLastError();
  return on;
}

// A host scalar is consumed when the call is made; keep a copy for a graph
// that replays later.
std::shared_ptr<std::vector<uint8_t>> hold_scalar(const void* p, cudaDataType_t t) {
  if (!p) return nullptr;
  const auto* b = static_cast<const uint8_t*>(p);
  return std::make_shared<std::vector<uint8_t>>(b, b + elem_bytes(t));
}
const void* held(const std::shared_ptr<std::vector<uint8_t>>& h) { return h ? h->data() : nullptr; }

// Runs `work` now, after the stream's earlier work, or records it into the
// graph the stream is capturing.
Status submit(const char* api, cudaStream_t stream, std::function<Status()> work) {
  if (capturing(stream)) {
    std::function<void()> op = [work, api] {
      if (work()) say(api, "a recorded call failed when its graph ran");
    };
    return vgpu_record_host_op_if_capturing(stream, std::move(op)) ? CUTENSOR_STATUS_SUCCESS
                                                                     : CUTENSOR_STATUS_EXECUTION_FAILED;
  }
  cudaGetLastError();
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    cudaGetLastError();
    return CUTENSOR_STATUS_CUDA_ERROR;
  }
  return work();
}

bool complex_problem(const cutensorOperationDescriptor& op) {
  for (const Operand& t : op.t)
    if (is_complex(t.desc.type)) return true;
  return is_complex(op.scalar);
}

Status exec_checks(const cutensorHandle_t handle, const cutensorPlan_t plan, Kind want, void* workspace,
                   uint64_t workspaceSize) {
  if (!handle || !plan) return CUTENSOR_STATUS_INVALID_VALUE;
  if (plan->op->kind != want) return CUTENSOR_STATUS_INVALID_VALUE;
  if (!workspace && workspaceSize > 0) return CUTENSOR_STATUS_INVALID_VALUE;  // measured
  return CUTENSOR_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

cutensorStatus_t cutensorContract(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha,
                                  const void* A, const void* B, const void* beta, const void* C, void* D,
                                  void* workspace, uint64_t workspaceSize, cudaStream_t stream) {
  if (Status s = exec_checks(handle, plan, Kind::Contraction, workspace, workspaceSize)) return s;
  if (!alpha || !A || !B || !beta || !C || !D) return CUTENSOR_STATUS_INVALID_VALUE;
  auto op = plan->op;
  auto a = hold_scalar(alpha, op->scalar), b = hold_scalar(beta, op->scalar);
  return submit("cutensorContract", stream, [=]() -> Status {
    return complex_problem(*op) ? run_contraction<cd>(*op, held(a), {A, B}, held(b), C, D)
                                : run_contraction<double>(*op, held(a), {A, B}, held(b), C, D);
  });
}

cutensorStatus_t cutensorContractTrinary(const cutensorHandle_t handle, const cutensorPlan_t plan,
                                         const void* alpha, const void* A, const void* B, const void* C,
                                         const void* beta, const void* D, void* E, void* workspace,
                                         uint64_t workspaceSize, cudaStream_t stream) {
  if (Status s = exec_checks(handle, plan, Kind::Contraction3, workspace, workspaceSize)) return s;
  if (!alpha || !A || !B || !C || !beta || !D || !E) return CUTENSOR_STATUS_INVALID_VALUE;
  auto op = plan->op;
  auto a = hold_scalar(alpha, op->scalar), b = hold_scalar(beta, op->scalar);
  return submit("cutensorContractTrinary", stream, [=]() -> Status {
    return complex_problem(*op) ? run_contraction<cd>(*op, held(a), {A, B, C}, held(b), D, E)
                                : run_contraction<double>(*op, held(a), {A, B, C}, held(b), D, E);
  });
}

cutensorStatus_t cutensorReduce(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha,
                                const void* A, const void* beta, const void* C, void* D, void* workspace,
                                uint64_t workspaceSize, cudaStream_t stream) {
  if (Status s = exec_checks(handle, plan, Kind::Reduction, workspace, workspaceSize)) return s;
  if (!alpha || !A || !beta || !C || !D) return CUTENSOR_STATUS_INVALID_VALUE;
  auto op = plan->op;
  auto a = hold_scalar(alpha, op->scalar), b = hold_scalar(beta, op->scalar);
  return submit("cutensorReduce", stream, [=]() -> Status {
    return complex_problem(*op) ? run_reduction<cd>(*op, held(a), A, held(b), C, D)
                                : run_reduction<double>(*op, held(a), A, held(b), C, D);
  });
}

cutensorStatus_t cutensorPermute(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha,
                                 const void* A, void* B, const cudaStream_t stream) {
  if (Status s = exec_checks(handle, plan, Kind::Permutation, nullptr, 0)) return s;
  if (!alpha || !A || !B) return CUTENSOR_STATUS_INVALID_VALUE;
  auto op = plan->op;
  auto a = hold_scalar(alpha, op->scalar);
  return submit("cutensorPermute", stream, [=]() -> Status {
    return complex_problem(*op) ? run_elementwise<cd>(*op, {held(a)}, {A}, B)
                                : run_elementwise<double>(*op, {held(a)}, {A}, B);
  });
}

cutensorStatus_t cutensorElementwiseBinaryExecute(const cutensorHandle_t handle, const cutensorPlan_t plan,
                                                  const void* alpha, const void* A, const void* gamma,
                                                  const void* C, void* D, cudaStream_t stream) {
  if (Status s = exec_checks(handle, plan, Kind::Binary, nullptr, 0)) return s;
  if (!alpha || !A || !gamma || !C || !D) return CUTENSOR_STATUS_INVALID_VALUE;
  auto op = plan->op;
  auto a = hold_scalar(alpha, op->scalar), g = hold_scalar(gamma, op->scalar);
  return submit("cutensorElementwiseBinaryExecute", stream, [=]() -> Status {
    return complex_problem(*op) ? run_elementwise<cd>(*op, {held(a), held(g)}, {A, C}, D)
                                : run_elementwise<double>(*op, {held(a), held(g)}, {A, C}, D);
  });
}

cutensorStatus_t cutensorElementwiseTrinaryExecute(const cutensorHandle_t handle, const cutensorPlan_t plan,
                                                   const void* alpha, const void* A, const void* beta,
                                                   const void* B, const void* gamma, const void* C, void* D,
                                                   cudaStream_t stream) {
  if (Status s = exec_checks(handle, plan, Kind::Trinary, nullptr, 0)) return s;
  if (!alpha || !A || !beta || !B || !gamma || !C || !D) return CUTENSOR_STATUS_INVALID_VALUE;
  auto op = plan->op;
  auto a = hold_scalar(alpha, op->scalar), b = hold_scalar(beta, op->scalar), g = hold_scalar(gamma, op->scalar);
  return submit("cutensorElementwiseTrinaryExecute", stream, [=]() -> Status {
    return complex_problem(*op) ? run_elementwise<cd>(*op, {held(a), held(b), held(g)}, {A, B, C}, D)
                                : run_elementwise<double>(*op, {held(a), held(b), held(g)}, {A, B, C}, D);
  });
}

}  // extern "C"

// ---- block-sparse contractions ----
//
// Descriptors and operations are created and checked as NVIDIA's library
// does on an RTX 3060 (a section coordinate out of range or a zero extent is
// INVALID_VALUE; types other than R32F, R64F, C32F, C64F are NOT_SUPPORTED).
// Planning one is not: the RTX 3060 itself cannot (cutensorCreatePlan
// answers INTERNAL_ERROR there), so there is no card behaviour to check a
// block-sparse kernel against, and this library refuses the plan with
// NOT_SUPPORTED.

extern "C" {

cutensorStatus_t cutensorCreateBlockSparseTensorDescriptor(
    cutensorHandle_t handle, cutensorBlockSparseTensorDescriptor_t* desc, const uint32_t numModes,
    const uint64_t numNonZeroBlocks, const uint32_t numSectionsPerMode[], const int64_t extent[],
    const int32_t nonZeroCoordinates[], const int64_t stride[], cudaDataType_t dataType) {
  if (!handle || !desc) return CUTENSOR_STATUS_INVALID_VALUE;
  if (numModes && (!numSectionsPerMode || !extent)) return CUTENSOR_STATUS_INVALID_VALUE;
  if (numNonZeroBlocks && numModes && !nonZeroCoordinates) return CUTENSOR_STATUS_INVALID_VALUE;
  if (dataType != CUDA_R_32F && dataType != CUDA_R_64F && dataType != CUDA_C_32F && dataType != CUDA_C_64F)
    return CUTENSOR_STATUS_NOT_SUPPORTED;
  if (numModes > 32) return CUTENSOR_STATUS_NOT_SUPPORTED;
  auto d = std::make_unique<cutensorBlockSparseTensorDescriptor>();
  d->modes = numModes;
  d->type = dataType;
  size_t at = 0;
  for (uint32_t m = 0; m < numModes; ++m) {
    if (numSectionsPerMode[m] == 0) return CUTENSOR_STATUS_INVALID_VALUE;
    d->sections.push_back(numSectionsPerMode[m]);
    std::vector<int64_t> e;
    for (uint32_t s = 0; s < numSectionsPerMode[m]; ++s, ++at) {
      if (extent[at] <= 0) return CUTENSOR_STATUS_INVALID_VALUE;
      e.push_back(extent[at]);
    }
    d->ext.push_back(std::move(e));
  }
  for (uint64_t b = 0; b < numNonZeroBlocks; ++b) {
    std::vector<int32_t> c(numModes);
    std::vector<int64_t> st(numModes);
    int64_t packed = 1;
    for (uint32_t m = 0; m < numModes; ++m) {
      c[m] = nonZeroCoordinates[b * numModes + m];
      if (c[m] < 0 || (uint32_t)c[m] >= d->sections[m]) return CUTENSOR_STATUS_INVALID_VALUE;
      st[m] = stride ? stride[b * numModes + m] : packed;
      packed *= d->ext[m][c[m]];
    }
    d->coord.push_back(std::move(c));
    d->stride.push_back(std::move(st));
  }
  *desc = d.release();
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorDestroyBlockSparseTensorDescriptor(cutensorBlockSparseTensorDescriptor_t desc) {
  delete desc;
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorCreateBlockSparseContraction(
    const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc,
    const cutensorBlockSparseTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA,
    const cutensorBlockSparseTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB,
    const cutensorBlockSparseTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC,
    const cutensorBlockSparseTensorDescriptor_t descD, const int32_t modeD[],
    const cutensorComputeDescriptor_t descCompute) {
  if (!handle || !desc || !descA || !descB || !descC || !descD || !descCompute) return CUTENSOR_STATUS_INVALID_VALUE;
  if (!known_compute(descCompute)) return CUTENSOR_STATUS_INVALID_VALUE;
  if ((descA->modes && !modeA) || (descB->modes && !modeB) || (descC->modes && !modeC) || (descD->modes && !modeD))
    return CUTENSOR_STATUS_INVALID_VALUE;
  if (descC != descD) return CUTENSOR_STATUS_NOT_SUPPORTED;  // documented: the same descriptor, for now
  if (opA != CUTENSOR_OP_IDENTITY || opB != CUTENSOR_OP_IDENTITY || opC != CUTENSOR_OP_IDENTITY)
    return CUTENSOR_STATUS_NOT_SUPPORTED;
  if (descA->type != descB->type || descA->type != descC->type) return CUTENSOR_STATUS_NOT_SUPPORTED;
  auto op = std::make_unique<cutensorOperationDescriptor>();
  op->kind = Kind::BlockSparse;
  op->compute = descCompute->kind;
  op->scalar = descC->type;
  const cutensorBlockSparseTensorDescriptor* ds[4] = {descA, descB, descC, descD};
  const int32_t* ms[4] = {modeA, modeB, modeC, modeD};
  for (int i = 0; i < 4; ++i) {
    BlockOperand b;
    b.desc = *ds[i];
    b.modes.assign(ms[i], ms[i] + ds[i]->modes);
    op->bs.push_back(std::move(b));
  }
  // Each tensor's dense extent, for MOVED_BYTES.
  for (int i = 0; i < 4; ++i) {
    Operand t;
    t.desc.type = ds[i]->type;
    int64_t elems = 0;
    for (size_t b = 0; b < ds[i]->coord.size(); ++b) {
      int64_t v = 1;
      for (uint32_t m = 0; m < ds[i]->modes; ++m) v *= ds[i]->ext[m][ds[i]->coord[b][m]];
      elems += v;
    }
    t.desc.extent = {std::max<int64_t>(elems, 1)};
    t.desc.stride = {1};
    op->t.push_back(std::move(t));
  }
  *desc = op.release();
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorBlockSparseContract(const cutensorHandle_t handle, const cutensorPlan_t plan,
                                             const void* alpha, const void* const A[], const void* const B[],
                                             const void* beta, const void* const C[], void* const D[],
                                             void* workspace, uint64_t workspaceSize, cudaStream_t stream) {
  (void)alpha, (void)A, (void)B, (void)beta, (void)C, (void)D, (void)stream;
  if (Status s = exec_checks(handle, plan, Kind::BlockSparse, workspace, workspaceSize)) return s;
  return refuse("cutensorBlockSparseContract", "block-sparse contractions are not implemented");
}

// Exported by NVIDIA's library but not documented: present so that a
// program linked against it loads, and refused.
#define VGPU_CUTENSOR_UNDOCUMENTED(name)                                         \
  cutensorStatus_t name(...) {                                                   \
    return refuse(#name, "this undocumented entry point is not implemented");  \
  }
VGPU_CUTENSOR_UNDOCUMENTED(cutensorComputeDescriptorGetAttribute)
VGPU_CUTENSOR_UNDOCUMENTED(cutensorComputeDescriptorSetAttribute)
VGPU_CUTENSOR_UNDOCUMENTED(cutensorCreateComputeDescriptor)
VGPU_CUTENSOR_UNDOCUMENTED(cutensorDestroyComputeDescriptor)
VGPU_CUTENSOR_UNDOCUMENTED(cutensorOperationEstimateRuntime)
VGPU_CUTENSOR_UNDOCUMENTED(cutensorOperationNumAlgos)
VGPU_CUTENSOR_UNDOCUMENTED(cutensorTensorDescriptorSetAttribute)
VGPU_CUTENSOR_UNDOCUMENTED(cutensorCreateExtraction)
VGPU_CUTENSOR_UNDOCUMENTED(cutensorCreateInsertion)
VGPU_CUTENSOR_UNDOCUMENTED(cutensorExtract)
VGPU_CUTENSOR_UNDOCUMENTED(cutensorInsert)
#undef VGPU_CUTENSOR_UNDOCUMENTED

}  // extern "C"
