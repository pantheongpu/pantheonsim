#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
typedef unsigned u4 __attribute__((ext_vector_type(4)));
// One wave: every lane does iters LDS accesses of `width` bytes at base + lane * stride_bytes (+ a fixed offset).
__global__ void probe(unsigned stride_bytes, unsigned width, unsigned iters, unsigned write, unsigned* out) {
  __shared__ unsigned lds[16384];
  const unsigned lane = threadIdx.x;
  for (unsigned i = lane; i < 16384; i += blockDim.x) lds[i] = i;
  __syncthreads();
  const unsigned addr = (unsigned)(uintptr_t)lds + (lane * stride_bytes) % 65536;
  unsigned acc = 0;
  if (!write) {
    if (width == 4) for (unsigned i = 0; i < iters; ++i) { unsigned v; asm volatile("ds_read_b32 %0, %1\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(addr) : "memory"); acc += v; }
    if (width == 8) for (unsigned i = 0; i < iters; ++i) { unsigned long long v; asm volatile("ds_read_b64 %0, %1\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(addr) : "memory"); acc += (unsigned)v; }
    if (width == 16) for (unsigned i = 0; i < iters; ++i) { u4 v; asm volatile("ds_read_b128 %0, %1\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(addr) : "memory"); acc += v[0]; }
  } else {
    if (width == 4) for (unsigned i = 0; i < iters; ++i) asm volatile("ds_write_b32 %0, %1\n s_waitcnt lgkmcnt(0)" :: "v"(addr), "v"(lane) : "memory");
    if (width == 8) { unsigned long long d = lane; for (unsigned i = 0; i < iters; ++i) asm volatile("ds_write_b64 %0, %1\n s_waitcnt lgkmcnt(0)" :: "v"(addr), "v"(d) : "memory"); }
    if (width == 16) { u4 d = {lane, lane, lane, lane}; for (unsigned i = 0; i < iters; ++i) asm volatile("ds_write_b128 %0, %1\n s_waitcnt lgkmcnt(0)" :: "v"(addr), "v"(d) : "memory"); }
  }
  if (acc == 0x12345678u) out[0] = acc;
}
int main(int argc, char** argv) {
  const unsigned stride = argc > 1 ? atoi(argv[1]) : 4, width = argc > 2 ? atoi(argv[2]) : 4, wr = argc > 3 ? atoi(argv[3]) : 0, threads = argc > 4 ? atoi(argv[4]) : 32, iters = 1000;
  unsigned* out; hipMalloc(&out, 64);
  probe<<<1, threads>>>(stride, width, iters, wr, out);
  hipDeviceSynchronize();
  hipFree(out);
  return 0;
}
