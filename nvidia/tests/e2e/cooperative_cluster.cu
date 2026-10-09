// A cooperative launch of clusters: cudaLaunchKernelEx with both
// cudaLaunchAttributeCooperative and a cluster dimension. Every block is
// resident, so a grid-wide barrier holds, and each cluster still has its own
// -- a block reads its cluster peer's shared memory, and the grid-wide
// barrier then lets it read what a block of another cluster wrote.
// Checked exactly against the host. Prints PASS on the last line.
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

constexpr int kThreads = 64;

__global__ void coop_cluster(int* shared_out, int* grid_out, int* flags) {
  __shared__ int buf[kThreads];
  cg::grid_group grid = cg::this_grid();
  cg::cluster_group cluster = cg::this_cluster();
  const unsigned rank = cluster.block_rank();
  const int t = threadIdx.x;
  buf[t] = int(blockIdx.x * 1000 + t);
  cluster.sync();   // the cluster's shared memories are written
  // A cluster peer's shared memory, through distributed shared memory.
  int* theirs = cluster.map_shared_rank(buf, (rank + 1) % cluster.num_blocks());
  shared_out[blockIdx.x * kThreads + t] = theirs[t];
  cluster.sync();   // and read, before any block of the cluster may exit
  // Every block of the grid writes a flag, waits for the grid, and reads the flag
  // of a block in the next cluster.
  if (t == 0) flags[blockIdx.x] = int(blockIdx.x) + 1;
  grid.sync();
  const unsigned other = (blockIdx.x + cluster.num_blocks()) % gridDim.x;
  grid_out[blockIdx.x * kThreads + t] = flags[other] * 100 + t;
}

int main() {
  constexpr int kCluster = 2, kBlocks = 8;
  int *d_shared, *d_grid, *d_flags;
  CHECK(cudaMalloc(&d_shared, kBlocks * kThreads * sizeof(int)));
  CHECK(cudaMalloc(&d_grid, kBlocks * kThreads * sizeof(int)));
  CHECK(cudaMalloc(&d_flags, kBlocks * sizeof(int)));
  CHECK(cudaMemset(d_flags, 0, kBlocks * sizeof(int)));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(kBlocks);
  cfg.blockDim = dim3(kThreads);
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeCooperative;
  attr[0].val.cooperative = 1;
  attr[1].id = cudaLaunchAttributeClusterDimension;
  attr[1].val.clusterDim.x = kCluster;
  attr[1].val.clusterDim.y = 1;
  attr[1].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 2;
  CHECK(cudaLaunchKernelEx(&cfg, coop_cluster, d_shared, d_grid, d_flags));
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  std::vector<int> shared(kBlocks * kThreads), grid(kBlocks * kThreads);
  CHECK(cudaMemcpy(shared.data(), d_shared, shared.size() * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(grid.data(), d_grid, grid.size() * sizeof(int), cudaMemcpyDeviceToHost));
  int bad_shared = 0, bad_grid = 0;
  for (int b = 0; b < kBlocks; ++b)
    for (int t = 0; t < kThreads; ++t) {
      const int peer = (b / kCluster) * kCluster + (b % kCluster + 1) % kCluster;   // the next rank of b's cluster
      bad_shared += shared[b * kThreads + t] != peer * 1000 + t;
      const int other = (b + kCluster) % kBlocks;
      bad_grid += grid[b * kThreads + t] != (other + 1) * 100 + t;
    }
  std::printf("cluster shared memory: %d of %d wrong\n", bad_shared, kBlocks * kThreads);
  std::printf("grid barrier: %d of %d wrong\n", bad_grid, kBlocks * kThreads);
  const bool ok = bad_shared == 0 && bad_grid == 0;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
