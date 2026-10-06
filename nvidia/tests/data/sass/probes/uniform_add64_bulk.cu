// sm_90a, sm_100a, sm_120a: a bulk copy from a parameter pointer plus a constant
// or a parameter offset, which CUDA 13's ptxas computes with UIADD3.64 in
// uniform registers (and the UBLKCP forms around it). nvcc -arch=sm_XXa -cubin.
__global__ void k(const char* g, char* o, long long off) {
  __shared__ __align__(128) char s[256];
  __shared__ __align__(8) unsigned long long bar;
  const unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(s)), ba = static_cast<unsigned>(__cvta_generic_to_shared(&bar));
  if (threadIdx.x == 0) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 16, [%2];" :: "r"(sa), "l"(g + 0x30), "r"(ba) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 16, [%2];" :: "r"(sa + 16), "l"(g - 0x1000), "r"(ba) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 16, [%2];" :: "r"(sa + 32), "l"(g + off), "r"(ba) : "memory");
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 16;" :: "l"(o + off * 2), "r"(sa) : "memory");
  }
}
