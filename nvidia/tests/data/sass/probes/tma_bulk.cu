// Decoder probe: TMA tensor loads/stores/reductions (UTMALDG/UTMASTG/
// UTMAREDG), bulk copies (UBLKCP), tensormap prefetch. sm_90a.
#include <cstdint>
__global__ void k(uint64_t tmap, int x, int y, int z, float* out) {
  __shared__ alignas(128) float buf[1024];
  __shared__ uint64_t bar;
  unsigned sb = static_cast<unsigned>(__cvta_generic_to_shared(buf));
  unsigned mb = static_cast<unsigned>(__cvta_generic_to_shared(&bar));
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mb));
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(sb), "l"(tmap), "r"(x), "r"(y), "r"(mb) : "memory");
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                 :: "r"(sb + 1024), "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(mb) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" :: "l"(tmap), "r"(x), "r"(y), "r"(sb) : "memory");
    asm volatile("cp.async.bulk.commit_group;");
    asm volatile("cp.async.bulk.wait_group.read 0;");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 256, [%2];" :: "r"(sb), "l"(out), "r"(mb) : "memory");
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];" :: "l"(tmap), "r"(x), "r"(y), "r"(sb) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(tmap));
  }
  out[threadIdx.x] = buf[threadIdx.x];
}
