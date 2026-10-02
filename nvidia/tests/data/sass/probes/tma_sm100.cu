// Decoder probe: Blackwell's TMA forms -- gather4/scatter4 rows, the CTA-pair
// (.cta_group::2) loads, and im2col's ::w modes. sm_100a.
#include <cstdint>
__global__ void k(uint64_t tmap, int c, int r0, int r1, int r2, int r3, unsigned short mask, unsigned short o) {
  __shared__ alignas(128) uint32_t buf[4096];
  __shared__ uint64_t bar;
  unsigned sb = static_cast<unsigned>(__cvta_generic_to_shared(buf));
  unsigned mb = static_cast<unsigned>(__cvta_generic_to_shared(&bar));
  if (threadIdx.x == 0) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                 :: "r"(sb), "l"(tmap), "r"(c), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(mb) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6];"
                 :: "l"(tmap), "r"(c), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(sb) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(sb), "l"(tmap), "r"(c), "r"(r0), "r"(mb) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
                 :: "r"(sb), "l"(tmap), "r"(c), "r"(r0), "r"(mb), "h"(mask) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.cta_group::1.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                 :: "r"(sb), "l"(tmap), "r"(c), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(mb) : "memory");
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.im2col::w.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5], {%6, %6};"
                 :: "r"(sb), "l"(tmap), "r"(c), "r"(r0), "r"(r1), "r"(mb), "h"(o) : "memory");
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.im2col::w::128.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5], {%6, %6};"
                 :: "r"(sb), "l"(tmap), "r"(c), "r"(r0), "r"(r1), "r"(mb), "h"(o) : "memory");
    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4 [%0, {%1, %2, %3, %4, %5}];" :: "l"(tmap), "r"(c), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
  }
  __syncthreads();
}
