// atomicAdd with subnormal operands and results: does the device flush them, in global and in shared memory, for float and double,
// with the old value used (ATOM) and unused (RED)? One line a case, bits in hex:
//
//     <space> <type> <use> <case> init=<bits> add=<bits> old=<bits> mem=<bits>
//
// space: g (global) or s (shared); type: f32 or f64; use: atom (the returned old value is stored) or red (it is not).
// Build: nvcc -std=c++17 -arch=sm_<NN> -cudart shared [-ftz=true] atomic_subnormal.cu -o atomic_subnormal
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

struct F32Case { uint32_t init, add; };
struct F64Case { uint64_t init, add; };
// f32: a subnormal operand, a subnormal sum of subnormals, a normal sum of subnormals (input flushing makes it 0), a subnormal result
// of normal operands, and a subnormal accumulator plus zero.
__constant__ F32Case k32[] = {
  {0x00000000u, 0x00000001u}, {0x00000003u, 0x00000004u}, {0x007fffffu, 0x007fffffu}, {0x00c00000u, 0x80800000u},
  {0x00800000u, 0x80000001u}, {0x00000005u, 0x00000000u}, {0x80000000u, 0x00000001u}, {0x3f800000u, 0x00000001u},
};
__constant__ F64Case k64[] = {
  {0ull, 1ull}, {3ull, 4ull}, {0x000fffffffffffffull, 0x000fffffffffffffull}, {0x0018000000000000ull, 0x8010000000000000ull},
  {0x0010000000000000ull, 0x8000000000000001ull}, {5ull, 0ull},
};
constexpr int N32 = 8, N64 = 6;

__global__ void run32(uint32_t* out) {   // out[(space * 2 + use) * N32 * 2 + case * 2 + {old, mem}]
  __shared__ float s;
  for (int c = 0; c < N32; ++c) {
    for (int use = 0; use < 2; ++use) {
      for (int space = 0; space < 2; ++space) {
        float* p = space ? &s : reinterpret_cast<float*>(out + 4 * N32 * 2 + (use * N32 + c));   // global cell past the results
        float init = __uint_as_float(k32[c].init), add = __uint_as_float(k32[c].add);
        *p = init;
        __syncthreads();
        float old = 0.0f;
        if (use == 0) old = atomicAdd(p, add); else atomicAdd(p, add);
        uint32_t* o = out + ((space * 2 + use) * N32 + c) * 2;
        o[0] = use == 0 ? __float_as_uint(old) : 0xffffffffu;
        o[1] = __float_as_uint(*p);
        __syncthreads();
      }
    }
  }
}

__global__ void run64(uint64_t* out) {
  __shared__ double s;
  for (int c = 0; c < N64; ++c)
    for (int use = 0; use < 2; ++use)
      for (int space = 0; space < 2; ++space) {
        double* p = space ? &s : reinterpret_cast<double*>(out + 4 * N64 * 2 + (use * N64 + c));
        double init = __longlong_as_double(k64[c].init), add = __longlong_as_double(k64[c].add);
        *p = init;
        __syncthreads();
        double old = 0.0;
        if (use == 0) old = atomicAdd(p, add); else atomicAdd(p, add);
        uint64_t* o = out + ((space * 2 + use) * N64 + c) * 2;
        o[0] = use == 0 ? __double_as_longlong(old) : ~0ull;
        o[1] = __double_as_longlong(*p);
        __syncthreads();
      }
}

int main() {
  uint32_t h32[4 * N32 * 2 + 2 * N32] = {};
  uint64_t h64[4 * N64 * 2 + 2 * N64] = {};
  F32Case c32[N32]; F64Case c64[N64];
  uint32_t* d32; uint64_t* d64;
  CK(cudaMalloc(&d32, sizeof h32)); CK(cudaMalloc(&d64, sizeof h64));
  CK(cudaMemset(d32, 0, sizeof h32)); CK(cudaMemset(d64, 0, sizeof h64));
  run32<<<1, 1>>>(d32); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  run64<<<1, 1>>>(d64); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h32, d32, sizeof h32, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h64, d64, sizeof h64, cudaMemcpyDeviceToHost));
  CK(cudaMemcpyFromSymbol(c32, k32, sizeof c32)); CK(cudaMemcpyFromSymbol(c64, k64, sizeof c64));
  const char* sp[2] = {"g", "s"}; const char* us[2] = {"atom", "red"};
  for (int space = 0; space < 2; ++space)
    for (int use = 0; use < 2; ++use)
      for (int c = 0; c < N32; ++c) {
        const uint32_t* o = h32 + ((space * 2 + use) * N32 + c) * 2;
        printf("%s f32 %s %d init=%08x add=%08x old=%08x mem=%08x\n", sp[space], us[use], c, c32[c].init, c32[c].add, o[0], o[1]);
      }
  for (int space = 0; space < 2; ++space)
    for (int use = 0; use < 2; ++use)
      for (int c = 0; c < N64; ++c) {
        const uint64_t* o = h64 + ((space * 2 + use) * N64 + c) * 2;
        printf("%s f64 %s %d init=%016llx add=%016llx old=%016llx mem=%016llx\n", sp[space], us[use], c, (unsigned long long)c64[c].init,
               (unsigned long long)c64[c].add, (unsigned long long)o[0], (unsigned long long)o[1]);
      }
  return 0;
}
