// Decoder probe: stores and reductions into another block's shared memory that
// complete on its mbarrier (st.async, red.async), and cluster launch control
// (clusterlaunchcontrol.try_cancel / query_cancel). st.async and red.async
// are sm_90+; cluster launch control is sm_100+.
#include <cstdint>
__global__ void k(unsigned remote, unsigned rbar, uint32_t v, uint64_t w, int* out) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" :: "r"(remote), "r"(v), "r"(rbar) : "memory");
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b64 [%0], %1, [%2];" :: "r"(remote), "l"(w), "r"(rbar) : "memory");
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];" :: "r"(remote), "r"(v), "r"(v + 1), "r"(rbar) : "memory");
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" :: "r"(remote), "r"(v), "r"(v + 1), "r"(v + 2), "r"(v + 3), "r"(rbar) : "memory");
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b64 [%0], {%1, %2}, [%3];" :: "r"(remote), "l"(w), "l"(w + 1), "r"(rbar) : "memory");
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];" :: "r"(remote), "f"(1.0f), "f"(2.0f), "r"(rbar) : "memory");
#define RED(op, ty, c) asm volatile("red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes." #op "." #ty " [%0], %1, [%2];" :: "r"(remote), c(v), "r"(rbar) : "memory")
  RED(add, u32, "r"); RED(min, u32, "r"); RED(max, u32, "r"); RED(min, s32, "r"); RED(max, s32, "r");
  RED(inc, u32, "r"); RED(dec, u32, "r"); RED(and, b32, "r"); RED(or, b32, "r"); RED(xor, b32, "r");
  asm volatile("red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u64 [%0], %1, [%2];" :: "r"(remote), "l"(w), "r"(rbar) : "memory");
#if __CUDA_ARCH__ >= 1000
  __shared__ __align__(16) uint64_t resp[2];
  __shared__ __align__(8) uint64_t bar;
  unsigned r = static_cast<unsigned>(__cvta_generic_to_shared(resp)), b = static_cast<unsigned>(__cvta_generic_to_shared(&bar));
  asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];" :: "r"(r), "r"(b) : "memory");
  asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];" :: "r"(r), "r"(b) : "memory");
  __syncthreads();
  uint32_t ok, x, y, z;
  asm volatile("{ .reg .b128 q; .reg .pred p; ld.shared.b128 q, [%4]; clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, q; selp.u32 %0, 1, 0, p;"
               " clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%1, %2, %3, _}, q; }"
               : "=r"(ok), "=r"(x), "=r"(y), "=r"(z) : "r"(r) : "memory");
  out[0] = ok + x + y + z;
#endif
}
