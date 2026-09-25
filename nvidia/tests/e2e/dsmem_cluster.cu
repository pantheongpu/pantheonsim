// Distributed shared memory through the CUDA C++ API a program uses for it:
// cooperative_groups::this_cluster(), map_shared_rank(), cluster.sync(), and
// clusters set both at compile time (__cluster_dims__) and at launch
// (cudaLaunchKernelEx with cudaLaunchAttributeClusterDimension).
//
// 1. A ring: each block of a 4-block cluster fills its shared array, then
//    reads its neighbour's through map_shared_rank and checks which block a
//    pointer belongs to with __cluster_query_shared_rank.
// 2. A histogram whose bins are spread over the shared memory of a cluster's
//    blocks: every thread adds into whichever block owns its bin, the way a
//    histogram too large for one block's shared memory is built on Hopper.
//
// Both are checked exactly against the host. Prints PASS on the last line.
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

#define CHECK(x)                                                                      \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      std::printf("FAIL: %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__); \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

constexpr int kRing = 64;

__global__ void __cluster_dims__(4, 1, 1) ring(int* out) {
  __shared__ int buf[kRing];
  cg::cluster_group cluster = cg::this_cluster();
  const unsigned rank = cluster.block_rank();
  const unsigned n = cluster.num_blocks();
  for (int i = threadIdx.x; i < kRing; i += blockDim.x) buf[i] = int(rank * 1000 + i);
  cluster.sync();   // every block's buf is written before anyone reads it
  const unsigned next = (rank + 1) % n;
  int* theirs = cluster.map_shared_rank(buf, next);
  const int base = blockIdx.x * kRing * 2;
  for (int i = threadIdx.x; i < kRing; i += blockDim.x) {
    out[base + 2 * i] = theirs[i];
    out[base + 2 * i + 1] = int(__cluster_query_shared_rank(theirs + i));
  }
  cluster.sync();   // and nobody exits while a neighbour may still read its buf
}

__global__ void histogram(int* bins, const int* data, int n, int nbins) {
  extern __shared__ int smem[];
  cg::cluster_group cluster = cg::this_cluster();
  const int per_block = nbins / int(cluster.num_blocks());
  for (int i = threadIdx.x; i < per_block; i += blockDim.x) smem[i] = 0;
  cluster.sync();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const int b = data[i];
    int* owner = cluster.map_shared_rank(smem, b / per_block);
    atomicAdd(owner + b % per_block, 1);
  }
  cluster.sync();
  // Each block's share of the bins, from this cluster, into the total.
  int* global_bins = bins + cluster.block_rank() * per_block;
  for (int i = threadIdx.x; i < per_block; i += blockDim.x) atomicAdd(&global_bins[i], smem[i]);
}

int main() {
  bool ok = true;
  {
    const int blocks = 8;
    int* d = nullptr;
    CHECK(cudaMalloc(&d, blocks * kRing * 2 * sizeof(int)));
    ring<<<blocks, 32>>>(d);
    CHECK(cudaGetLastError());
    std::vector<int> h(blocks * kRing * 2);
    CHECK(cudaMemcpy(h.data(), d, h.size() * sizeof(int), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int b = 0; b < blocks; ++b)
      for (int i = 0; i < kRing; ++i) {
        const int next = (b % 4 + 1) % 4;
        if (h[b * kRing * 2 + 2 * i] != next * 1000 + i || h[b * kRing * 2 + 2 * i + 1] != next) ++bad;
      }
    std::printf("ring: %d of %d wrong\n", bad, blocks * kRing);
    ok = ok && bad == 0;
    CHECK(cudaFree(d));
  }
  {
    const int n = 1 << 16, nbins = 1024, cluster = 4, blocks = 16, threads = 128;
    std::vector<int> data(n), want(nbins, 0);
    unsigned s = 12345;
    for (int i = 0; i < n; ++i) {
      s = s * 1664525u + 1013904223u;
      data[i] = int((s >> 8) % nbins);
      ++want[data[i]];
    }
    int *d_data = nullptr, *d_bins = nullptr;
    CHECK(cudaMalloc(&d_data, n * sizeof(int)));
    CHECK(cudaMalloc(&d_bins, nbins * sizeof(int)));
    CHECK(cudaMemcpy(d_data, data.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_bins, 0, nbins * sizeof(int)));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(blocks);
    cfg.blockDim = dim3(threads);
    cfg.dynamicSmemBytes = nbins / cluster * sizeof(int);
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = cluster;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    CHECK(cudaLaunchKernelEx(&cfg, histogram, d_bins, (const int*)d_data, n, nbins));
    CHECK(cudaGetLastError());
    std::vector<int> got(nbins);
    CHECK(cudaMemcpy(got.data(), d_bins, nbins * sizeof(int), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < nbins; ++i) bad += got[i] != want[i];
    std::printf("histogram: %d of %d bins wrong\n", bad, nbins);
    ok = ok && bad == 0;
    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_bins));
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
