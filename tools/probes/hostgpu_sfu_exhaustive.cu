// Card run for deciding whether an SFU instruction may join the host-GPU exact set (docs/perf-design/09-host-gpu-spike.md section 6 step 4; plan in
// docs/performance.md "Round sp15"). Every one of the 2^32 float bit patterns goes through the card's instruction and the bits are compared with the
// simulator's (include/vgpu/host_gpu_ref.hpp, which tests/unit/test_host_gpu_ref.cpp pins to the simulator's own kernels).
//
//   usage: hostgpu_sfu_exhaustive <op> [stride]
//   op:  rcp rsq sqrt ex2 lg2   the .approx.ftz.f32 instruction against the simulator's MUFU          (expected to differ: they are NOT in the set)
//        div                    IEEE a / b (div.rn.f32) for 8 dividends and every divisor           (the division idiom: must be 0 different)
//   stride: test every stride-th pattern (default 1 = all 2^32; a smoke run uses 4099)
//
// Output, one line per op and dividend: "RESULT op=<op> [a=0x...] tested=N differ=N max_ulp=N first=0x<in>:card=0x<bits>:sim=0x<bits>"; the exit status is 0 when
// differ=0 for every line, 1 otherwise. A mutation of the card side (e.g. a one-ulp nudge of the quotient) must make `div` report differ>0; run with
// HOSTGPU_SFU_MUTATE=1 to check the harness itself.
//   build: nvcc -std=c++20 -O2 -arch=sm_86 -I include tools/probes/hostgpu_sfu_exhaustive.cu -o hostgpu_sfu_exhaustive      (no -ftz, no -use_fast_math)
#include <cuda_runtime.h>

#include <algorithm>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "vgpu/host_gpu_ref.hpp"

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
    default: r = __fdiv_rn(__uint_as_float(a), x); break;   // div.rn.f32: the idiom (MUFU.RCP, FFMA refinement, FCHK, slow path)
  }
  uint32_t b = __float_as_uint(r);
  if (mutate && op == 5 && (b & 0x7f800000u) != 0x7f800000u && (in & 0x3ffu) == 0 && in) b += 1;   // a nudge the harness has to notice
  return b;
}
__global__ void run(int op, uint32_t a, uint64_t first, uint64_t count, uint64_t stride, bool mutate, uint32_t* out) {
  const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (i >= count) return;
  out[i] = card_op(op, (uint32_t)((first + i) * stride), a, mutate);
}

int main(int argc, char** argv) {
  using namespace vgpu::hostgpu;
  if (argc < 2) { std::fprintf(stderr, "usage: %s rcp|rsq|sqrt|ex2|lg2|div [stride]\n", argv[0]); return 2; }
  const std::string name = argv[1];
  const char* names[] = {"rcp", "rsq", "sqrt", "ex2", "lg2", "div"};
  const ref::Fn fns[] = {ref::Fn::Rcp, ref::Fn::Rsq, ref::Fn::Sqrt, ref::Fn::Ex2, ref::Fn::Lg2, ref::Fn::Rcp};
  int op = -1;
  for (int i = 0; i < 6; ++i) if (name == names[i]) op = i;
  if (op < 0) { std::fprintf(stderr, "unknown op %s\n", argv[1]); return 2; }
  const uint64_t stride = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1;
  const bool mutate = std::getenv("HOSTGPU_SFU_MUTATE") != nullptr;
  const uint64_t total = ((1ull << 32) + stride - 1) / stride;
  const uint32_t dividends[] = {0x3f800000u, 0x40400000u, 0x3dcccccdu, 0xc0200000u, 0x7bf00000u, 0x0a000000u, 0x3f9e3779u, 0x7f7fffffu};
  const int n_div = op == 5 ? 8 : 1;
  const uint64_t chunk = 1ull << 26;
  uint32_t* d = nullptr;
  CK(cudaMalloc(&d, chunk * 4));
  std::vector<uint32_t> h(chunk);
  int status = 0;
  for (int di = 0; di < n_div; ++di) {
    const uint32_t a = dividends[di];
    uint64_t differ = 0;
    uint32_t max_ulp = 0, first_in = 0, first_card = 0, first_sim = 0;
    for (uint64_t first = 0; first < total; first += chunk) {
      const uint64_t count = total - first < chunk ? total - first : chunk;
      run<<<(unsigned)((count + 255) / 256), 256>>>(op, a, first, count, stride, mutate, d);
      CK(cudaMemcpy(h.data(), d, count * 4, cudaMemcpyDeviceToHost));
      const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
      std::vector<std::thread> th;
      std::vector<uint64_t> bad(nt, 0);
      std::vector<uint32_t> mx(nt, 0), fi(nt, 0), fc(nt, 0), fs(nt, 0);
      std::vector<uint64_t> fidx(nt, UINT64_MAX);
      for (unsigned t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
          for (uint64_t i = t; i < count; i += nt) {
            const uint32_t in = (uint32_t)((first + i) * stride);
            const uint32_t sim = op == 5 ? ref::fdiv_rn(a, in) : ref::mufu(fns[op], in);
            const uint32_t card = (op == 5 && (h[i] & 0x7f800000u) == 0x7f800000u && (h[i] & 0x7fffffu)) ? 0x7fffffffu : h[i];
            if (card == sim) continue;
            if (!bad[t]++ || i < fidx[t]) { fidx[t] = i; fi[t] = in; fc[t] = card; fs[t] = sim; }
            const uint32_t u = ref::ulp_distance(card, sim);
            if (u > mx[t]) mx[t] = u;
          }
        });
      for (auto& x : th) x.join();
      uint64_t best = UINT64_MAX;
      for (unsigned t = 0; t < nt; ++t) {
        differ += bad[t];
        if (mx[t] > max_ulp) max_ulp = mx[t];
        if (bad[t] && fidx[t] < best && !(differ - bad[t])) { best = fidx[t]; first_in = fi[t]; first_card = fc[t]; first_sim = fs[t]; }
      }
    }
    char ds[32] = "";
    if (op == 5) std::snprintf(ds, sizeof ds, " a=0x%08x", a);
    std::printf("RESULT op=%s%s tested=%llu differ=%llu max_ulp=%u first=0x%08x:card=0x%08x:sim=0x%08x\n", name.c_str(), ds, (unsigned long long)total,
                (unsigned long long)differ, max_ulp, first_in, first_card, first_sim);
    std::fflush(stdout);
    if (differ) status = 1;
  }
  CK(cudaFree(d));
  return status;
}
