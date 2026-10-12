// Launches that run blocks on the simulator's pool of host threads
// (include/vgpu/exec/worker_pool.hpp): thousands of small and medium grids in
// a row, on several streams, from several host threads at once, and a
// cooperative one. Everything it computes is an integer, so the answer is the
// same on any schedule, and it prints a digest of each result. run_sched_pool.sh
// runs it at VGPU_THREADS=1, 2, 7 and the default, on SASS and on PTX, and
// requires every run to print the same lines.
#include <cooperative_groups.h>

#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>
namespace cg = cooperative_groups;

// out[i] = f(i), with a block-wide scan through shared memory.
__global__ void scan_blocks(uint32_t* out, uint32_t* block_sums, int n, uint32_t salt) {
  extern __shared__ uint32_t tile[];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t v = i < n ? (uint32_t(i) * 2654435761u + salt) >> 7 : 0;
  tile[threadIdx.x] = v;
  __syncthreads();
  for (int d = 1; d < blockDim.x; d <<= 1) {
    uint32_t add = threadIdx.x >= d ? tile[threadIdx.x - d] : 0;
    __syncthreads();
    tile[threadIdx.x] += add;
    __syncthreads();
  }
  if (i < n) out[i] = tile[threadIdx.x];
  if (threadIdx.x == blockDim.x - 1) block_sums[blockIdx.x] = tile[threadIdx.x];
}

// An integer histogram: atomics across every block.
__global__ void histogram(const uint32_t* in, uint32_t* bins, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&bins[in[i] & 63], 1u);
}

// A tiny kernel for the 10k-launch loop.
__global__ void bump(uint32_t* x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += (i & 3) + 1;
}

__global__ void gridsync(uint32_t* x, uint32_t* y, int n) {
  cg::grid_group grid = cg::this_grid();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += 3;
  grid.sync();
  if (i < n) y[i] = x[(i + 40) % n] * 2;
}

static uint64_t fnv(const std::vector<uint32_t>& v) {
  uint64_t h = 1469598103934665603ull;
  for (uint32_t x : v) {
    h ^= x;
    h *= 1099511628211ull;
  }
  return h;
}

static std::vector<uint32_t> fetch(const uint32_t* d, size_t n) {
  std::vector<uint32_t> h(n);
  cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost);
  return h;
}

static int g_bad = 0;
static void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL %s\n", what);
    ++g_bad;
  }
}

// One host thread's share: its own buffers and stream, a mix of launches.
static uint64_t host_work(int id, int rounds) {
  const int n = 1000 + 37 * id;
  uint32_t *out, *sums, *bins, *x;
  cudaStream_t s;
  cudaStreamCreate(&s);
  const int block = 64 << (id % 3);          // 64, 128, 256
  const int grid = (n + block - 1) / block;
  cudaMalloc(&out, n * 4);
  cudaMalloc(&sums, grid * 4);
  cudaMalloc(&bins, 64 * 4);
  cudaMalloc(&x, n * 4);
  cudaMemset(bins, 0, 64 * 4);
  cudaMemset(x, 0, n * 4);
  // Through cudaLaunchKernel, not <<<>>>: nvcc's generated stub for a kernel
  // stores into a static function pointer on every launch, which is a data race
  // (benign, and ThreadSanitizer reports it) when two host threads launch the
  // same kernel. The launch API itself is what is under test here.
  const uint32_t* cin = out;
  for (int r = 0; r < rounds; ++r) {
    uint32_t salt = 1000u * id + r;
    void* a_scan[] = {&out, &sums, (void*)&n, &salt};
    void* a_hist[] = {&cin, &bins, (void*)&n};
    void* a_bump[] = {&x, (void*)&n};
    cudaLaunchKernel((const void*)scan_blocks, dim3(grid), dim3(block), a_scan, block * 4, s);
    cudaLaunchKernel((const void*)histogram, dim3(grid), dim3(block), a_hist, 0, s);
    cudaLaunchKernel((const void*)bump, dim3(grid), dim3(block), a_bump, 0, s);
  }
  cudaStreamSynchronize(s);
  const std::vector<uint32_t> b = fetch(bins, 64);
  uint64_t total = 0;
  for (uint32_t v : b) total += v;
  check(total == uint64_t(n) * rounds, "histogram total");
  const std::vector<uint32_t> hx = fetch(x, n);
  bool ok = true;
  for (int i = 0; i < n; ++i) ok = ok && hx[i] == uint32_t((i & 3) + 1) * rounds;
  check(ok, "bump result");
  const uint64_t d = fnv(fetch(out, n)) ^ (fnv(fetch(sums, grid)) * 31) ^ (fnv(b) * 131);
  cudaFree(out); cudaFree(sums); cudaFree(bins); cudaFree(x);
  cudaStreamDestroy(s);
  return d;
}

int main() {
  // 1. Many launches of assorted shapes on the default stream.
  const int shapes[][2] = {{1, 32}, {3, 64}, {8, 32}, {17, 96}, {64, 128}, {200, 64}, {5, 1024}};
  for (const auto& sh : shapes) {
    const int grid = sh[0], block = sh[1], n = grid * block - 3;
    uint32_t *out, *sums, *bins, *x;
    cudaMalloc(&out, n * 4);
    cudaMalloc(&sums, grid * 4);
    cudaMalloc(&bins, 64 * 4);
    cudaMalloc(&x, n * 4);
    cudaMemset(bins, 0, 64 * 4);
    cudaMemset(x, 0, n * 4);
    const int reps = grid * block > 4000 ? 6 : 40;
    for (int r = 0; r < reps; ++r) {
      scan_blocks<<<grid, block, block * 4>>>(out, sums, n, r);
      histogram<<<grid, block>>>(out, bins, n);
      bump<<<grid, block>>>(x, n);
    }
    cudaDeviceSynchronize();
    const std::vector<uint32_t> b = fetch(bins, 64);
    uint64_t total = 0;
    for (uint32_t v : b) total += v;
    check(total == uint64_t(n) * reps, "shape histogram total");
    std::printf("shape %dx%d scan %016llx sums %016llx bins %016llx\n", grid, block,
                (unsigned long long)fnv(fetch(out, n)), (unsigned long long)fnv(fetch(sums, grid)),
                (unsigned long long)fnv(b));
    cudaFree(out); cudaFree(sums); cudaFree(bins); cudaFree(x);
  }

  // 2. A loop of thousands of tiny launches: the case the pool is for.
  {
    const int n = 8 * 32, loops = 2000;
    uint32_t* x;
    cudaMalloc(&x, n * 4);
    cudaMemset(x, 0, n * 4);
    for (int i = 0; i < loops; ++i) bump<<<8, 32>>>(x, n);
    cudaDeviceSynchronize();
    const std::vector<uint32_t> hx = fetch(x, n);
    bool ok = true;
    for (int i = 0; i < n; ++i) ok = ok && hx[i] == uint32_t((i & 3) + 1) * loops;
    check(ok, "tiny launch loop");
    std::printf("tiny loop %016llx\n", (unsigned long long)fnv(hx));
    cudaFree(x);
  }

  // 3. Several host threads launching at once, each on a stream of its own.
  {
    const int threads = 4;
    std::vector<uint64_t> digest(threads);
    std::vector<std::thread> hosts;
    for (int t = 0; t < threads; ++t) hosts.emplace_back([&, t] { digest[t] = host_work(t, 25); });
    for (auto& h : hosts) h.join();
    for (int t = 0; t < threads; ++t) std::printf("host thread %d %016llx\n", t, (unsigned long long)digest[t]);
  }

  // 4. A cooperative launch between ordinary ones.
  {
    const int n = 256, threads = 64;
    std::vector<uint32_t> h(n);
    for (int i = 0; i < n; ++i) h[i] = i;
    uint32_t *d, *o;
    cudaMalloc(&d, n * 4);
    cudaMalloc(&o, n * 4);
    cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice);
    int np = n;
    void* args[] = {&d, &o, &np};
    for (int r = 0; r < 3; ++r) {
      bump<<<4, 64>>>(d, 0);
      const cudaError_t e = cudaLaunchCooperativeKernel((void*)gridsync, dim3(n / threads), dim3(threads), args, 0, 0);
      check(e == cudaSuccess, "cooperative launch");
      cudaDeviceSynchronize();
    }
    const std::vector<uint32_t> y = fetch(o, n);
    bool ok = true;
    for (int i = 0; i < n; ++i) ok = ok && y[i] == ((i + 40) % n + 9) * 2u;
    check(ok, "grid sync result");
    std::printf("cooperative %016llx\n", (unsigned long long)fnv(y));
    cudaFree(d);
    cudaFree(o);
  }

  const cudaError_t e = cudaGetLastError();
  check(e == cudaSuccess, "no error left behind");
  std::printf(g_bad ? "FAIL\n" : "PASS\n");
  return g_bad ? 1 : 0;
}
