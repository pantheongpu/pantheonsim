// Decoder probe: mbarrier init/arrive/expect_tx/complete_tx/test_wait/
// try_wait/arrive_drop/inval (SYNCS). sm_90+.
#include <cstdint>
__global__ void k(int* out, int cnt, unsigned tx) {
  __shared__ uint64_t bar[4];
  unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(&bar[0]));
  unsigned b = static_cast<unsigned>(__cvta_generic_to_shared(&bar[1]));
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(a), "r"(1));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(b), "r"(cnt));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(a + 16), "r"(32));
  }
  __syncthreads();
  uint64_t st;
  asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(st) : "r"(b));
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(a), "r"(tx));
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(tx));
  asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(tx));
  uint32_t done;
  asm volatile("{ .reg .pred p; mbarrier.test_wait.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(done) : "r"(b), "l"(st));
  out[0] = done;
  asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(done) : "r"(a), "r"(0));
  out[1] = done;
  asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(done) : "r"(a), "r"(1));
  out[2] = done + (int)st;
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(3));
  asm volatile("mbarrier.arrive_drop.shared.b64 _, [%0];" :: "r"(b));
  asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(a + 16));
  uint32_t pc; asm volatile("mbarrier.pending_count.b64 %0, %1;" : "=r"(pc) : "l"(st)); out[3] = pc;
}
