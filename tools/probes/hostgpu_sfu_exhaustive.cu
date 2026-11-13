// Card run for deciding whether an SFU instruction may join the host-GPU exact set (docs/perf-design/09-host-gpu-spike.md section 6 step 4; plan in
// docs/perf-design/rounds/08-rounds-sp11-sp15.md "Round sp15"). Every one of the 2^32 float bit patterns goes through the card's instruction and the bits are compared with the
// simulator's (include/vgpu/host_gpu_ref.hpp, which tests/unit/test_host_gpu_ref.cpp pins to the simulator's own kernels).
//
//   usage: hostgpu_sfu_exhaustive <op> [stride]
//   op:  rcp rsq sqrt ex2 lg2   the .approx.ftz.f32 instruction against the simulator's MUFU          (expected to differ: they are NOT in the set)
//        div                    IEEE a / b (div.rn.f32) for 12 dividends (8 finite, 0, +-inf, NaN) and every divisor           (the division idiom: must be 0 different)
//        sqrtrn, rcprn          IEEE sqrt (sqrt.rn.f32, __fsqrt_rn) and IEEE reciprocal (rcp.rn.f32, __frcp_rn) of every pattern    (the sqrt and reciprocal idioms: must be 0 different; sp43)
//        idiv, sidiv            32-bit integer x / y, unsigned and signed, for 12 dividends and every nonzero divisor            (the integer division idiom: must be 0 different; sp43)
//        sin cos tanh           sin.approx.ftz.f32, cos.approx.ftz.f32, tanh.approx.f32 against the simulator's             (turbo mode, docs/perf-design/12-host-gpu-turbo.md:
//                               expected to differ; the count, the share and the largest distance are what turbo mode's tolerance has to cover)
//                               The compiler turns sin and cos into FMUL.RZ by 1/(2 pi) (the float 0x3e22f983, rounded toward zero) and MUFU.SIN / MUFU.COS, whose
//                               argument is in turns; the simulator side here is that product and then the simulator's MUFU (the SASS was read on an A10G rig
//                               with VGPU_HOST_GPU_LOG=2; a toolkit that compiles it otherwise needs the model changed).
//   stride: test every stride-th pattern (default 1 = all 2^32; a smoke run uses 4099)
//
// Output, one line per op and dividend: "RESULT op=<op> [a=0x...] tested=N differ=N max_ulp=N first=0x<in>:card=0x<bits>:sim=0x<bits> share=<percent> outside_tol=N max_abs=X" (outside_tol: results outside turbo mode's default tolerance,
// atol 1e-8 and rtol 1e-5; max_abs: the largest difference of two finite results; ordinary_*: the same two over the inputs with 2^-20 <= |x| < 2^10,
// 30/256 of the patterns); the exit status is 0 when
// differ=0 for every line, 1 otherwise. A mutation of the card side (e.g. a one-ulp nudge of the quotient) must make `div` report differ>0; run with
// HOSTGPU_SFU_MUTATE=1 to check the harness itself.
//   build: nvcc -std=c++20 -O2 -arch=sm_86 -I include tools/probes/hostgpu_sfu_exhaustive.cu -o hostgpu_sfu_exhaustive      (no -ftz, no -use_fast_math)
#include <cuda_runtime.h>

#include <algorithm>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "vgpu/host_gpu_ref.hpp"
#include "vgpu/host_gpu_tol.hpp"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

__device__ uint32_t card_op(int op, uint32_t in, uint32_t a, bool mutate) {
  const float x = __uint_as_float(in);
  float r;
  switch (op) {
    case 0: asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); break;
    case 1: asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); break;
    case 2: asm volatile("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); break;
    case 3: asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); break;
    case 4: asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); break;
    case 5: r = __fdiv_rn(__uint_as_float(a), x); break;   // div.rn.f32: the idiom (MUFU.RCP, FFMA refinement, FCHK, slow path)
    case 6: r = __fsqrt_rn(x); break;                      // sqrt.rn.f32: MUFU.RSQ, FFMA refinement, a slow path for the extremes
    case 7: r = __frcp_rn(x); break;                       // rcp.rn.f32: MUFU.RCP, FFMA refinement, FCHK, slow path
    case 8: r = __uint_as_float(in ? a / in : 0u); break;  // u32 x / y: I2F.RP, MUFU.RCP, IADD3 0xffffffe, F2I.TRUNC, a Newton step, two corrections
    case 10: asm volatile("sin.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); break;
    case 11: asm volatile("cos.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); break;
    case 12: asm volatile("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); break;
    default: r = __uint_as_float(in && !(a == 0x80000000u && in == 0xffffffffu) ? (uint32_t)((int)a / (int)in) : 0u); break;   // s32 (INT_MIN / -1 is undefined)
  }
  uint32_t b = __float_as_uint(r);
  const bool is_float = (b & 0x7f800000u) != 0x7f800000u;
  if (mutate && op >= 5 && op <= 9 && (op >= 8 || is_float) && (in & 0x3ffu) == 0 && in) b += 1;   // a nudge the harness has to notice
  return b;
}
__global__ void run(int op, uint32_t a, uint64_t first, uint64_t count, uint64_t stride, bool mutate, uint32_t* out) {
  const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (i >= count) return;
  out[i] = card_op(op, (uint32_t)((first + i) * stride), a, mutate);
}

// The argument of MUFU.SIN and MUFU.COS for sin.approx.ftz.f32(x), as nvcc 13.0 compiles it for sm_86 (read from the kernel's SASS on an A10G rig, Round sp59):
// FMUL.RZ R, x, 0.15915493667125701904 (the float 0x3e22f983, the product rounded TOWARD ZERO, no flush), then MUFU. The product of two floats is exact in a
// double, so it is rounded here by hand.
static uint32_t turns(uint32_t in) {
  uint32_t c = 0x3e22f983u, out;
  float inv_two_pi, x;
  std::memcpy(&inv_two_pi, &c, 4);
  std::memcpy(&x, &in, 4);
  const double exact = (double)x * (double)inv_two_pi;
  float t = (float)exact;
  if (t == t && std::fabs((double)t) > std::fabs(exact)) t = std::nextafterf(t, 0.0f);
  std::memcpy(&out, &t, 4);
  return out;
}

int main(int argc, char** argv) {
  using namespace vgpu::hostgpu;
  if (argc < 2) { std::fprintf(stderr, "usage: %s rcp|rsq|sqrt|ex2|lg2|div|sqrtrn|rcprn|idiv|sidiv|sin|cos|tanh [stride]\n", argv[0]); return 2; }
  const std::string name = argv[1];
  const char* names[] = {"rcp", "rsq", "sqrt", "ex2", "lg2", "div", "sqrtrn", "rcprn", "idiv", "sidiv", "sin", "cos", "tanh"};
  const ref::Fn fns[] = {ref::Fn::Rcp, ref::Fn::Rsq, ref::Fn::Sqrt, ref::Fn::Ex2, ref::Fn::Lg2, ref::Fn::Rcp, ref::Fn::Rcp, ref::Fn::Rcp, ref::Fn::Rcp, ref::Fn::Rcp,   // (ops 5 to 9: the exact forms below)
                         ref::Fn::Sin, ref::Fn::Cos, ref::Fn::Tanh};
  int op = -1;
  for (int i = 0; i < 13; ++i) if (name == names[i]) op = i;
  if (op < 0) { std::fprintf(stderr, "unknown op %s\n", argv[1]); return 2; }
  const uint64_t stride = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1;
  const bool mutate = std::getenv("HOSTGPU_SFU_MUTATE") != nullptr;
  const uint64_t total = ((1ull << 32) + stride - 1) / stride;
  // (0, +-inf and a NaN as dividends reach the slow path's NaN generator, MUFU.RSQ of an immediate NaN, for 0/0, inf/inf and NaN operands.)
  const uint32_t dividends[] = {0x3f800000u, 0x40400000u, 0x3dcccccdu, 0xc0200000u, 0x7bf00000u, 0x0a000000u, 0x3f9e3779u, 0x7f7fffffu, 0x00000000u, 0x7f800000u, 0xff800000u, 0x7fc00000u};
  const uint32_t int_dividends[] = {0u, 1u, 2u, 3u, 0x7fffffffu, 0x80000000u, 0xfffffffeu, 0xffffffffu, 12345u, 0x01234567u, 0x9e3779b9u, 0xdeadbeefu};
  const int n_div = (op == 5 || op == 8 || op == 9) ? 12 : 1;
  const uint64_t chunk = 1ull << 26;
  uint32_t* d = nullptr;
  CK(cudaMalloc(&d, chunk * 4));
  std::vector<uint32_t> h(chunk);
  int status = 0;
  for (int di = 0; di < n_div; ++di) {
    const uint32_t a = (op == 8 || op == 9) ? int_dividends[di] : dividends[di];
    uint64_t differ = 0, outside = 0, mid_outside = 0;
    double max_abs = 0, mid_max_abs = 0;
    uint32_t max_ulp = 0, first_in = 0, first_card = 0, first_sim = 0;
    for (uint64_t first = 0; first < total; first += chunk) {
      const uint64_t count = total - first < chunk ? total - first : chunk;
      run<<<(unsigned)((count + 255) / 256), 256>>>(op, a, first, count, stride, mutate, d);
      CK(cudaMemcpy(h.data(), d, count * 4, cudaMemcpyDeviceToHost));
      const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
      std::vector<std::thread> th;
      std::vector<uint64_t> bad(nt, 0), out_tol(nt, 0), mid_out(nt, 0);
      std::vector<double> mabs(nt, 0), mid_abs(nt, 0);
      std::vector<uint32_t> mx(nt, 0), fi(nt, 0), fc(nt, 0), fs(nt, 0);
      std::vector<uint64_t> fidx(nt, UINT64_MAX);
      for (unsigned t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
          for (uint64_t i = t; i < count; i += nt) {
            const uint32_t in = (uint32_t)((first + i) * stride);
            const uint32_t sim = op == 8 ? (in ? a / in : 0u) : op == 9 ? (in && !(a == 0x80000000u && in == 0xffffffffu) ? (uint32_t)((int)a / (int)in) : 0u) : op == 5 ? ref::fdiv_rn(a, in) : op == 6 ? ref::fsqrt_rn(in) : op == 7 ? ref::frcp_rn(in) :
                                 (op == 10 || op == 11) ? ref::mufu(fns[op], turns(in)) : ref::mufu(fns[op], in);
            // (a NaN of the exact float forms and of sin, cos and tanh is compared as the canonical one: its payload is not what these ops are about)
            const uint32_t card = ((op >= 5 && op <= 7) || op >= 10) && (h[i] & 0x7f800000u) == 0x7f800000u && (h[i] & 0x7fffffu) ? 0x7fffffffu : h[i];
            if (card == sim) continue;
            if (!bad[t]++ || i < fidx[t]) { fidx[t] = i; fi[t] = in; fc[t] = card; fs[t] = sim; }
            const uint32_t u = ref::ulp_distance(card, sim);
            if (u > mx[t]) mx[t] = u;
            // (the same pair by turbo mode's default tolerance: the ulp distance of two results near zero says little)
            float cf, sf;
            std::memcpy(&cf, &card, 4);
            std::memcpy(&sf, &sim, 4);
            double a = 0;
            const bool close = values_close(cf, sf, Tolerance{}, &a);
            if (!close) ++out_tol[t];
            if (a > mabs[t]) mabs[t] = a;
            // ... and over the inputs of ordinary size alone, 2^-20 <= |x| < 2^10 (exponent fields 107 to 136): an argument of 2^40 turns is not what
            // a program takes the sine of
            const uint32_t e = (in >> 23) & 0xff;
            if (e >= 107 && e <= 136) {
              if (!close) ++mid_out[t];
              if (a > mid_abs[t]) mid_abs[t] = a;
            }
          }
        });
      for (auto& x : th) x.join();
      uint64_t best = UINT64_MAX;
      for (unsigned t = 0; t < nt; ++t) {
        differ += bad[t];
        outside += out_tol[t];
        mid_outside += mid_out[t];
        if (mabs[t] > max_abs) max_abs = mabs[t];
        if (mid_abs[t] > mid_max_abs) mid_max_abs = mid_abs[t];
        if (mx[t] > max_ulp) max_ulp = mx[t];
        if (bad[t] && fidx[t] < best && !(differ - bad[t])) { best = fidx[t]; first_in = fi[t]; first_card = fc[t]; first_sim = fs[t]; }
      }
    }
    char ds[32] = "";
    if (op == 5 || op == 8 || op == 9) std::snprintf(ds, sizeof ds, " a=0x%08x", a);
    std::printf("RESULT op=%s%s tested=%llu differ=%llu max_ulp=%u first=0x%08x:card=0x%08x:sim=0x%08x share=%.3f%% outside_tol=%llu max_abs=%.9g ordinary_outside_tol=%llu ordinary_max_abs=%.9g\n", name.c_str(), ds, (unsigned long long)total,
                (unsigned long long)differ, max_ulp, first_in, first_card, first_sim, 100.0 * (double)differ / (double)total, (unsigned long long)outside, max_abs,
                (unsigned long long)mid_outside, mid_max_abs);
    std::fflush(stdout);
    if (differ) status = 1;
  }
  CK(cudaFree(d));
  return status;
}
