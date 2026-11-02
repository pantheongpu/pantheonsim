// The simulator's special-function (MUFU) results, as scalar host code, for deciding whether a MUFU instruction may join the host-GPU exact set
// (docs/perf-design/09-host-gpu-spike.md section 6, step 4). The card side is tools/probes/hostgpu_sfu_exhaustive.cu, which runs every one of the 2^32 float
// patterns through the card's instruction and compares the bits with these functions; tests/unit/test_host_gpu_logic.cpp pins these functions to the
// simulator's own SIMD kernels (src/simd/kernels.inc mufu_k) and to the interpreter's libm forms (src/sass/exec_ops.inc, case Op::MUFU), so that what the
// card run compares against is what the simulator computes. Host-only; no CUDA.
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>

namespace vgpu::hostgpu::ref {

// A subnormal is read and written as a zero of its own sign (the .FTZ forms the compiler emits for rcp/rsq/sqrt/ex2/lg2.approx.ftz).
inline float flush(float f) {
  const uint32_t b = std::bit_cast<uint32_t>(f);
  return (b & 0x7f800000u) == 0 ? std::bit_cast<float>(b & 0x80000000u) : f;
}
inline uint32_t finish(float v, bool flush_result) {
  if (v != v) return 0x7fffffffu;   // a NaN is the canonical one
  return std::bit_cast<uint32_t>(flush_result ? flush(v) : v);
}

// MUFU kinds, as the SASS encodes them (Instr::f[0]).
enum class Fn : unsigned { Cos = 0, Sin = 1, Ex2 = 2, Lg2 = 3, Rcp = 4, Rsq = 5, Sqrt = 8, Tanh = 9 };

// The bits the simulator gives MUFU.<fn> of the float with bits `in`: the input flushed, the answer in double rounded once to float, flushed.
inline uint32_t mufu(Fn fn, uint32_t in) {
  const bool flush_io = fn != Fn::Tanh;
  const float f = flush_io ? flush(std::bit_cast<float>(in)) : std::bit_cast<float>(in);
  const double x = f;
  float v;
  switch (fn) {
    case Fn::Cos: v = static_cast<float>(std::cos(x * 6.28318530717958647692)); break;   // the argument is in turns
    case Fn::Sin: v = static_cast<float>(std::sin(x * 6.28318530717958647692)); break;
    case Fn::Ex2: v = static_cast<float>(std::exp2(x)); break;
    case Fn::Lg2: v = static_cast<float>(std::log2(x)); break;
    case Fn::Rcp: v = static_cast<float>(1.0 / x); break;
    case Fn::Rsq: v = static_cast<float>(1.0 / std::sqrt(x)); break;
    case Fn::Sqrt: v = static_cast<float>(std::sqrt(x)); break;
    default: v = static_cast<float>(std::tanh(x)); break;
  }
  return finish(v, flush_io);
}

// ulp distance between two float bit patterns of the same sign (otherwise UINT32_MAX); NaN against NaN is 0.
inline uint32_t ulp_distance(uint32_t a, uint32_t b) {
  const auto nan = [](uint32_t x) { return (x & 0x7f800000u) == 0x7f800000u && (x & 0x7fffffu); };
  if (nan(a) || nan(b)) return nan(a) && nan(b) ? 0 : UINT32_MAX;
  if ((a ^ b) >> 31) return (a << 1) == 0 && (b << 1) == 0 ? 0 : UINT32_MAX;
  return a > b ? a - b : b - a;
}

// The IEEE quotient of two floats as the host computes it (SSE, correctly rounded, subnormals kept): what div.rn.f32 must give on the card and in the simulator.
inline uint32_t fdiv_rn(uint32_t a, uint32_t b) {
  const float q = std::bit_cast<float>(a) / std::bit_cast<float>(b);
  return q != q ? 0x7fffffffu : std::bit_cast<uint32_t>(q);
}

}  // namespace vgpu::hostgpu::ref
