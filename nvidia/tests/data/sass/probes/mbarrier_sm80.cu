// Decoder probe: mbarrier as sm_80 has it (init, arrive, arrive.noComplete,
// arrive_drop, test_wait, pending_count, inval, cp.async's arrive), which
// before sm_90 compiles to ATOMS.ARRIVE and plain shared loads. sm_80+.
#include <cstdint>
__global__ void k(int* out, int cnt, const int* g) {
  __shared__ uint64_t bar[2];
  __shared__ int buf[32];
  unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(&bar[0]));
  unsigned b = static_cast<unsigned>(__cvta_generic_to_shared(&bar[1]));
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(a), "r"(cnt));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(b), "r"(64));
  }
  __syncthreads();
  uint64_t st, st2;
  asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(st) : "r"(a));
  asm volatile("mbarrier.arrive.noComplete.shared.b64 %0, [%1], %2;" : "=l"(st2) : "r"(b), "r"(2));
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(&buf[threadIdx.x]));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(s), "l"(g + threadIdx.x));
  asm volatile("cp.async.mbarrier.arrive.shared.b64 [%0];" :: "r"(b));
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(b));
  uint32_t done;
  asm volatile("{ .reg .pred p; mbarrier.test_wait.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(done) : "r"(a), "l"(st));
  out[0] = done;
  asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(done) : "r"(b), "r"(1));
  out[1] = done;
  asm volatile("mbarrier.arrive_drop.shared.b64 _, [%0];" :: "r"(a));
  asm volatile("mbarrier.arrive_drop.noComplete.shared.b64 _, [%0], %1;" :: "r"(b), "r"(1));
  uint32_t pc; asm volatile("mbarrier.pending_count.b64 %0, %1;" : "=r"(pc) : "l"(st2)); out[2] = pc + buf[threadIdx.x];
  asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(a));
}
