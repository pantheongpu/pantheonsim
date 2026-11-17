// The bfloat16 PTX approximate forms on a real card (sm_90 and later), exhaustively: ex2.approx.ftz.bf16, ex2.approx.ftz.bf16x2,
// tanh.approx.bf16, tanh.approx.bf16x2, each over every bfloat16 pattern, as one hash a form. The sibling of mufu_f16_forms.cu.
//
//   nvcc -std=c++17 -arch=sm_90 mufu_bf16_forms.cu -o mufu_bf16_forms && ./mufu_bf16_forms
//
// Output, one line a form: "<form> <hash>". The .f16 forms run the 65,536 patterns 0 .. 65535 in order; the .f16x2 forms the
// 65,536 registers (x, y) with x = 0 .. 65535 in the low half and y = (x * 40503 + 12345) mod 65536 in the high half (a
// bijection, so every pattern is also a high-half operand). The hash is FNV-1a 64 over the results, a 32-bit word a step (the
// .bf16 result zero-extended). nvcc's SASS for each form is one or two MUFU.<EX2|TANH>.BF16.
#include <cstdint>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define F16FORM(name, ptx) \
  __global__ void name(uint32_t* o, const uint32_t* i) { \
    const int t = blockIdx.x * blockDim.x + threadIdx.x; \
    uint16_t r; asm(ptx " %0, %1;" : "=h"(r) : "h"(static_cast<uint16_t>(i[t]))); \
    o[t] = r; \
  }
#define F16X2FORM(name, ptx) \
  __global__ void name(uint32_t* o, const uint32_t* i) { \
    const int t = blockIdx.x * blockDim.x + threadIdx.x; \
    uint32_t r; asm(ptx " %0, %1;" : "=r"(r) : "r"(i[t])); \
    o[t] = r; \
  }
F16FORM(k_ex2_bf16, "ex2.approx.ftz.bf16")
F16FORM(k_tanh_bf16, "tanh.approx.bf16")
F16X2FORM(k_ex2_bf16x2, "ex2.approx.ftz.bf16x2")
F16X2FORM(k_tanh_bf16x2, "tanh.approx.bf16x2")

static uint64_t fnv(const std::vector<uint32_t>& v) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (uint32_t x : v) h = (h ^ x) * 0x100000001b3ull;
  return h;
}

int main() {
  const int n = 65536;
  std::vector<uint32_t> a(n), b(n), o(n);
  for (int i = 0; i < n; ++i) { a[i] = i; b[i] = static_cast<uint32_t>(i) | (((static_cast<uint32_t>(i) * 40503u + 12345u) & 0xffffu) << 16); }
  uint32_t *da, *db;
  cudaMalloc(&da, n * 4); cudaMalloc(&db, n * 4);
  const auto run = [&](const char* name, void (*k)(uint32_t*, const uint32_t*), const std::vector<uint32_t>& in) {
    cudaMemcpy(da, in.data(), n * 4, cudaMemcpyHostToDevice);
    k<<<n / 256, 256>>>(db, da);
    cudaMemcpy(o.data(), db, n * 4, cudaMemcpyDeviceToHost);
    printf("%s %016llx\n", name, static_cast<unsigned long long>(fnv(o)));
  };
  run("ex2.approx.bf16", k_ex2_bf16, a);
  run("tanh.approx.bf16", k_tanh_bf16, a);
  run("ex2.approx.bf16x2", k_ex2_bf16x2, b);
  run("tanh.approx.bf16x2", k_tanh_bf16x2, b);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); return 1; }
  return 0;
}
