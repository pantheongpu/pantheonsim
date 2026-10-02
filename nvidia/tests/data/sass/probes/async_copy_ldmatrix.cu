// Decoder probe: cp.async (LDGSTS with and without zero-fill), ldmatrix
// (LDSM x1/x2/x4, transposed) and the CAS forms of ATOM/ATOMG. sm_80+.
#include <cstdint>
__global__ void k(const int4* g, int4* out, int n, unsigned long long* p64, unsigned* p32) {
  __shared__ int4 s[256];
  unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(&s[threadIdx.x]));
  const int4* src = g + threadIdx.x;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(sa), "l"(src));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(sa + 16), "l"(src + 1));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(sa + 32), "l"(src + 2));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa + 48), "l"(src + 3));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(sa + 64), "l"(src + 4), "r"(n));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(sa + 80), "l"(src + 5), "r"(n));
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  unsigned r0, r1, r2, r3;
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(r0) : "r"(sa));
  out[threadIdx.x].x = r0;
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(sa));
  out[threadIdx.x].y = r0 + r1;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(sa));
  out[threadIdx.x].z = r0 + r1 + r2 + r3;
  out[threadIdx.x].w = atomicCAS(p32, n, 5) + (unsigned)atomicCAS(p64, n, 7);
  unsigned long long* gen = (unsigned long long*)(n > 3 ? (void*)p64 : (void*)s);
  out[threadIdx.x + 1].w = atomicCAS(gen, n, 9);
  out[threadIdx.x + 2].w = atomicCAS((unsigned*)gen, n, 9);
}
