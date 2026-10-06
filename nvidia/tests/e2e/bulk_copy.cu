// The bulk copies that are not tensor copies (cp.async.bulk, SASS UBLKCP and
// UBLKRED), as CuTe's SM90 bulk-copy tests use them: global to shared
// completing on an mbarrier, the same multicast to both CTAs of a cluster,
// shared to shared, shared to global, and a bulk add-reduction into global.
// The SASS form keeps the size in 16-byte units in the low half of its
// register and a multicast's CTA mask in the high half; 1040 bytes (65
// units) tells units from bytes. Each block checks its data, and the sums
// are printed, so the SASS run and the PTX run must print the same.
#include <cstdio>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

constexpr unsigned kBytes = 1040;
constexpr unsigned kWords = kBytes / 4;

__device__ void wait_phase(unsigned bar, unsigned phase) {
  asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W; }"
               :: "r"(bar), "r"(phase) : "memory");
}

__global__ void __cluster_dims__(2, 1, 1) bulk(const unsigned* src, unsigned* out, unsigned* red, unsigned* sums) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  __shared__ __align__(128) unsigned a[kWords + 4];
  __shared__ __align__(128) unsigned b[kWords + 4];
  __shared__ __align__(128) unsigned m[kWords + 4];
  __shared__ __align__(8) unsigned long long bar[2];
  cg::cluster_group cluster = cg::this_cluster();
  const unsigned rank = cluster.block_rank();
  const unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(a));
  const unsigned sb = static_cast<unsigned>(__cvta_generic_to_shared(b));
  const unsigned sm = static_cast<unsigned>(__cvta_generic_to_shared(m));
  const unsigned b0 = static_cast<unsigned>(__cvta_generic_to_shared(&bar[0]));
  const unsigned b1 = static_cast<unsigned>(__cvta_generic_to_shared(&bar[1]));
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(b0));
    asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(b1));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cluster.sync();
  if (threadIdx.x == 0) {
    // Global to this CTA's shared memory.
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(b0), "r"(kBytes) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(sa), "l"(src + rank * kWords), "r"(kBytes), "r"(b0) : "memory");
    // Rank 0 multicasts one more block of global to both CTAs' m[].
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(b1), "r"(kBytes) : "memory");
    if (rank == 0) {
      const unsigned short mask = 3;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
                   :: "r"(sm), "l"(src + 2 * kWords), "r"(kBytes), "r"(b1), "h"(mask) : "memory");
    }
  }
  wait_phase(b0, 0);
  wait_phase(b1, 0);
  __syncthreads();
  if (threadIdx.x == 0) {
    // a to b in shared memory, then b back out to global, and a added into red.
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(b0), "r"(kBytes) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(sb), "r"(sa), "r"(kBytes), "r"(b0) : "memory");
  }
  wait_phase(b0, 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(out + rank * kWords), "r"(sb), "r"(kBytes) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [%0], [%1], %2;" :: "l"(red), "r"(sa), "r"(kBytes) : "memory");
    asm volatile("cp.async.bulk.commit_group;");
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
  unsigned bad = 0, sum_m = 0;
  for (unsigned i = threadIdx.x; i < kWords; i += blockDim.x) {
    bad += a[i] != src[rank * kWords + i];
    bad += m[i] != src[2 * kWords + i];
    sum_m += m[i];
  }
  atomicAdd(&sums[rank * 2], bad);
  atomicAdd(&sums[rank * 2 + 1], sum_m);
  cluster.sync();
#else
  (void)src; (void)out; (void)red; (void)sums;
#endif
}

int main() {
  static unsigned host[3 * kWords];
  for (unsigned i = 0; i < 3 * kWords; ++i) host[i] = i * 2654435761u + 7;
  unsigned *src, *out, *red, *sums;
  cudaMalloc(&src, sizeof host);
  cudaMalloc(&out, 2 * kBytes);
  cudaMalloc(&red, kBytes);
  cudaMalloc(&sums, 4 * sizeof(unsigned));
  cudaMemcpy(src, host, sizeof host, cudaMemcpyHostToDevice);
  cudaMemset(out, 0, 2 * kBytes);
  cudaMemset(red, 0, kBytes);
  cudaMemset(sums, 0, 4 * sizeof(unsigned));
  bulk<<<2, 64>>>(src, out, red, sums);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { std::printf("FAIL: %s\n", cudaGetErrorString(e)); return 1; }
  static unsigned o[2 * kWords], r[kWords];
  unsigned s[4];
  cudaMemcpy(o, out, sizeof o, cudaMemcpyDeviceToHost);
  cudaMemcpy(r, red, sizeof r, cudaMemcpyDeviceToHost);
  cudaMemcpy(s, sums, sizeof s, cudaMemcpyDeviceToHost);
  unsigned bad = s[0] + s[2];
  for (unsigned i = 0; i < 2 * kWords; ++i) bad += o[i] != host[i];
  for (unsigned i = 0; i < kWords; ++i) bad += r[i] != host[i] + host[kWords + i];
  std::printf("%s: %u wrong; multicast sums %08x %08x, out[%u] %08x, red[%u] %08x\n", bad ? "FAIL" : "ok", bad, s[1], s[3],
              kWords + 3, o[kWords + 3], kWords - 1, r[kWords - 1]);
  return bad ? 1 : 0;
}
