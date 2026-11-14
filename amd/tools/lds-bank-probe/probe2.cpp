#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
typedef unsigned u4 __attribute__((ext_vector_type(4)));
typedef unsigned u3 __attribute__((ext_vector_type(3)));
// One wave: every lane does iters LDS accesses at lds + (lane * stride_bytes) % 65536. `form` picks the instruction.
__global__ void probe(unsigned stride_bytes, unsigned form, unsigned iters, unsigned* out) {
  __shared__ unsigned lds[16384];
  const unsigned lane = threadIdx.x;
  for (unsigned i = lane; i < 16384; i += blockDim.x) lds[i] = i;
  __syncthreads();
  const unsigned addr = (unsigned)(uintptr_t)lds + (lane * stride_bytes) % 65536;
  unsigned acc = 0;
  switch (form) {
    case 1:  for (unsigned i = 0; i < iters; ++i) { unsigned v; asm volatile("ds_read_u8 %0, %1\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(addr) : "memory"); acc += v; } break;
    case 2:  for (unsigned i = 0; i < iters; ++i) { unsigned v; asm volatile("ds_read_u16 %0, %1\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(addr) : "memory"); acc += v; } break;
    case 11: for (unsigned i = 0; i < iters; ++i) asm volatile("ds_write_b8 %0, %1\n s_waitcnt lgkmcnt(0)" :: "v"(addr), "v"(lane) : "memory"); break;
    case 12: for (unsigned i = 0; i < iters; ++i) { u3 v; asm volatile("ds_read_b96 %0, %1\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(addr) : "memory"); acc += v[0]; } break;
    case 13: { u3 d = {lane, lane, lane}; for (unsigned i = 0; i < iters; ++i) asm volatile("ds_write_b96 %0, %1\n s_waitcnt lgkmcnt(0)" :: "v"(addr), "v"(d) : "memory"); } break;
    case 104: for (unsigned i = 0; i < iters; ++i) asm volatile("ds_add_u32 %0, %1\n s_waitcnt lgkmcnt(0)" :: "v"(addr), "v"(lane) : "memory"); break;
    case 105: for (unsigned i = 0; i < iters; ++i) { unsigned v; asm volatile("ds_add_rtn_u32 %0, %1, %2\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(addr), "v"(lane) : "memory"); acc += v; } break;
    case 106: for (unsigned i = 0; i < iters; ++i) { unsigned long long d = lane; asm volatile("ds_add_u64 %0, %1\n s_waitcnt lgkmcnt(0)" :: "v"(addr), "v"(d) : "memory"); } break;
    case 202: for (unsigned i = 0; i < iters; ++i) { unsigned long long v; asm volatile("ds_read2_b32 %0, %1 offset0:0 offset1:1\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(addr) : "memory"); acc += (unsigned)v; } break;
    case 203: for (unsigned i = 0; i < iters; ++i) { unsigned long long v; asm volatile("ds_read2_b32 %0, %1 offset0:0 offset1:32\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(addr) : "memory"); acc += (unsigned)v; } break;
    case 204: for (unsigned i = 0; i < iters; ++i) { u4 v; asm volatile("ds_read2_b64 %0, %1 offset0:0 offset1:1\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(addr) : "memory"); acc += v[0]; } break;
    case 205: for (unsigned i = 0; i < iters; ++i) { unsigned long long v; asm volatile("ds_read2st64_b32 %0, %1 offset0:0 offset1:1\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(addr) : "memory"); acc += (unsigned)v; } break;
    case 206: for (unsigned i = 0; i < iters; ++i) asm volatile("ds_write2_b32 %0, %1, %2 offset0:0 offset1:1\n s_waitcnt lgkmcnt(0)" :: "v"(addr), "v"(lane), "v"(lane + 1) : "memory"); break;
  }
  if (acc == 0x12345678u) out[0] = acc;
}
int main(int argc, char** argv) {
  const unsigned stride = argc > 1 ? atoi(argv[1]) : 4, form = argc > 2 ? atoi(argv[2]) : 1, threads = argc > 3 ? atoi(argv[3]) : 32, iters = 1000;
  unsigned* out; (void)hipMalloc(&out, 64);
  probe<<<1, threads>>>(stride, form, iters, out);
  (void)hipDeviceSynchronize();
  (void)hipFree(out);
  return 0;
}
