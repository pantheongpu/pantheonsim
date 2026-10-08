// A block-wide allocation inside an `if (i < n)` that the last threads of a
// block skip -- the shape of CUB's for_each over a partial tile with Cupoch's
// kd-tree builder (SplitNodes) as the operator. The operator calls
// __syncthreads(), which ptxas compiles inside the divergent region to
//   WARPSYNC 0xffffffff ; BAR.SYNC
// while the threads that skipped the region park at the BSYNC that closes it,
// waiting for the ones inside. The WARPSYNC wants all 32 lanes, the BSYNC wants
// lane 0, and the block barrier wants every live lane; each waits for another.
// An RTX 3080 Ti runs it, so the simulator, which reported "every warp ... is
// waiting (a barrier some threads never reach)", lets the lanes inside go on
// once nothing else can move. Every count is computed on the host and the
// program passes on the card.
#include <cstdio>
#include <cuda_runtime.h>
struct Node { int a, b, c; float x, y; };
__device__ __noinline__ void work(Node* nodes, int i, int* cnt, int* alloc) {
  // the shape of a block-wide allocation: a shared counter, barriers, a branchy body
  __shared__ int block_nodes;
  __shared__ int base;
  if (threadIdx.x == 0) block_nodes = 0;
  __syncthreads();
  int off = 0;
  bool split = nodes[i].a > 2 && nodes[i].x != nodes[i].y;
  if (split) off = atomicAdd(&block_nodes, 2);
  __syncthreads();
  if (threadIdx.x == 0) base = atomicAdd(alloc, block_nodes);
  __syncthreads();
  if (split) {
    for (int j = 0; j < 2; ++j) {
      nodes[i].b = base + off + j;
      if (nodes[i].x > nodes[i].y) nodes[i].c += 1; else nodes[i].c -= 1;
      atomicAdd(cnt, 1);
    }
  }
}
__global__ void k(int n, Node* nodes, int* cnt, int* alloc, int* done) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) work(nodes, i, cnt, alloc);   // threads past n skip it, barriers included
  if (i < 1 << 20) atomicAdd(done, 1);
}
int main() {
  int bad = 0;
  for (int n : {1, 2, 5, 31, 33, 100, 129, 257}) {
    Node h[512];
    for (int i = 0; i < 512; ++i) h[i] = {3 + i % 3, -1, 0, float(i % 7), float(i % 5)};
    Node* d; int *c, *a, *dn;
    cudaMalloc(&d, sizeof h); cudaMalloc(&c, 4); cudaMalloc(&a, 4); cudaMalloc(&dn, 4);
    cudaMemcpy(d, h, sizeof h, cudaMemcpyHostToDevice);
    cudaMemset(c, 0, 4); cudaMemset(a, 0, 4); cudaMemset(dn, 0, 4);
    const int blocks = (n + 127) / 128;
    k<<<blocks, 128>>>(n, d, c, a, dn);
    cudaError_t e = cudaDeviceSynchronize();
    int hc = -1, ha = -1, hd = -1;
    cudaMemcpy(&hc, c, 4, cudaMemcpyDeviceToHost); cudaMemcpy(&ha, a, 4, cudaMemcpyDeviceToHost); cudaMemcpy(&hd, dn, 4, cudaMemcpyDeviceToHost);
    int want = 0;
    for (int i = 0; i < n; ++i) if (h[i].a > 2 && h[i].x != h[i].y) want += 2;
    const bool ok = e == cudaSuccess && hc == want && ha == want && hd == blocks * 128;
    printf("n=%d err=%d splits=%d/%d alloc=%d done=%d %s\n", n, (int)e, hc, want, ha, hd, ok ? "ok" : "BAD");
    bad += !ok;
  }
  printf(bad ? "FAIL\n" : "PASS\n");
  return bad != 0;
}
