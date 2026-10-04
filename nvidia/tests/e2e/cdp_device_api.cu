// The device runtime as CUDA Samples' dynamic-parallelism programs use it
// (cdpSimplePrint, cdpSimpleQuicksort, cdpQuadtree), built as they are built:
// -rdc=true, linked with cudadevrt. On the SASS engine the device runtime's
// entry points are builtins (src/sass/module.cpp), and the linked cubin
// carries relocations a plain build does not: a kernel's function descriptor
// in a UMOV pair, and in the device runtime library's own tables.
//
// 1. printf from a grid launched by a grid launched by the host, recursing
//    as cdpSimplePrint does (each thread of each level launches the next).
// 2. A recursive sort, each half in a device-created stream (cdpSimpleQuicksort).
// 3. A child with dynamic shared memory (cdpQuadtree).
// 4. The device-side last error: a launch with too many threads per block
//    fails with cudaErrorInvalidConfiguration, which cudaPeekAtLastError
//    reports and leaves and cudaGetLastError reports and clears (0 9 9 0 on
//    an RTX 3060, CUDA 13.0).
// 5. cudaGetDevice and cudaGetDeviceCount in a child grid.
//
// Every result is checked exactly; the output does not depend on the order
// grids run in. Prints PASS on the last line.
#include <cstdio>

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                           \
      std::printf("FAIL: %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);   \
      return false;                                                                    \
    }                                                                                  \
  } while (0)

// 1. Each level's threads launch the next level; level `depth` counts its
// blocks, and only the deepest level prints, once.
__global__ void levels(int* count, int depth, int max_depth) {
  if (threadIdx.x == 0) atomicAdd(&count[depth], 1);
  if (depth + 1 < max_depth) {
    levels<<<2, 2>>>(count, depth + 1, max_depth);
  } else if (threadIdx.x == 0 && blockIdx.x == 0 && atomicAdd(&count[max_depth], 1) == 0) {
    printf("deepest level %d reached\n", depth);
  }
}

// 2. Quicksort with one thread per launch, as cdpSimpleQuicksort.
__global__ void quicksort(unsigned* data, int left, int right, int depth) {
  if (depth >= 16 || right - left <= 8) {
    for (int i = left; i <= right; ++i)
      for (int j = i + 1; j <= right; ++j)
        if (data[j] < data[i]) {
          unsigned t = data[i];
          data[i] = data[j];
          data[j] = t;
        }
    return;
  }
  unsigned* lptr = data + left;
  unsigned* rptr = data + right;
  const unsigned pivot = data[(left + right) / 2];
  while (lptr <= rptr) {
    unsigned lval = *lptr, rval = *rptr;
    while (lval < pivot) lval = *++lptr;
    while (rval > pivot) rval = *--rptr;
    if (lptr <= rptr) {
      *lptr++ = rval;
      *rptr-- = lval;
    }
  }
  const int nright = static_cast<int>(rptr - data), nleft = static_cast<int>(lptr - data);
  if (left < nright) {
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    quicksort<<<1, 1, 0, s>>>(data, left, nright, depth + 1);
    cudaStreamDestroy(s);
  }
  if (nleft < right) {
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    quicksort<<<1, 1, 0, s>>>(data, nleft, right, depth + 1);
    cudaStreamDestroy(s);
  }
}

// 3. Dynamic shared memory in a child: each block sums its slice through it.
__global__ void block_sums(const int* in, int* out) {
  extern __shared__ int part[];
  part[threadIdx.x] = in[blockIdx.x * blockDim.x + threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0) {
    int s = 0;
    for (unsigned i = 0; i < blockDim.x; ++i) s += part[i];
    out[blockIdx.x] = s;
  }
}
__global__ void launch_sums(const int* in, int* out) { block_sums<<<4, 64, 64 * sizeof(int)>>>(in, out); }

// 4 and 5.
__global__ void noop() {}
__global__ void ask_device(int* out) {
  int dev = -1, count = -1;
  out[4] = cudaGetDevice(&dev);
  out[5] = dev;
  out[6] = cudaGetDeviceCount(&count);
  out[7] = count;
}
__global__ void errors(int* out) {
  out[0] = cudaGetLastError();        // nothing yet
  noop<<<1, 4096>>>();                // more threads than a block may have
  out[1] = cudaPeekAtLastError();     // reported, kept
  out[2] = cudaGetLastError();        // reported, cleared
  out[3] = cudaGetLastError();
  ask_device<<<1, 1>>>(out);
}

bool run() {
  bool ok = true;
  {
    int* d;
    const int max_depth = 3;
    CK(cudaMalloc(&d, (max_depth + 1) * sizeof(int)));
    CK(cudaMemset(d, 0, (max_depth + 1) * sizeof(int)));
    levels<<<2, 2>>>(d, 0, max_depth);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    int h[max_depth + 1];
    CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
    // 2 blocks at level 0; each of their 4 threads launches 2 blocks. The
    // deepest level is 16 grids, each with a block 0 (one of them prints).
    const bool good = h[0] == 2 && h[1] == 8 && h[2] == 32 && h[3] == 16;
    std::printf("recursive launches: %d %d %d, %d grids at the bottom%s\n", h[0], h[1], h[2], h[3],
                good ? "" : " (want 2 8 32, 16)");
    ok = ok && good;
    CK(cudaFree(d));
  }
  {
    const int n = 512;
    unsigned h[n];
    unsigned x = 12345;
    for (int i = 0; i < n; ++i) {
      x = x * 1103515245u + 12345u;
      h[i] = (x >> 8) % 1000;
    }
    unsigned* d;
    CK(cudaMalloc(&d, sizeof h));
    CK(cudaMemcpy(d, h, sizeof h, cudaMemcpyHostToDevice));
    quicksort<<<1, 1>>>(d, 0, n - 1, 0);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
    unsigned long long sum = 0;
    bool sorted = true;
    for (int i = 0; i < n; ++i) {
      sum += h[i];
      if (i && h[i - 1] > h[i]) sorted = false;
    }
    std::printf("quicksort in device streams: %s, sum %llu\n", sorted ? "sorted" : "NOT SORTED", sum);
    ok = ok && sorted;
    CK(cudaFree(d));
  }
  {
    int *in, *out;
    int h[256], s[4];
    for (int i = 0; i < 256; ++i) h[i] = i;
    CK(cudaMalloc(&in, sizeof h));
    CK(cudaMalloc(&out, sizeof s));
    CK(cudaMemcpy(in, h, sizeof h, cudaMemcpyHostToDevice));
    launch_sums<<<1, 1>>>(in, out);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(s, out, sizeof s, cudaMemcpyDeviceToHost));
    bool good = true;
    for (int b = 0; b < 4; ++b) good = good && s[b] == 64 * 64 * b + 64 * 63 / 2;
    std::printf("child dynamic shared memory: %d %d %d %d%s\n", s[0], s[1], s[2], s[3], good ? "" : " (want 2016 6112 10208 14304)");
    ok = ok && good;
    CK(cudaFree(in));
    CK(cudaFree(out));
  }
  {
    int* d;
    int h[8];
    CK(cudaMalloc(&d, sizeof h));
    CK(cudaMemset(d, 0xff, sizeof h));
    errors<<<1, 1>>>(d);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
    int dev = -1, count = -1;
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceCount(&count));
    const bool good = h[0] == cudaSuccess && h[1] == cudaErrorInvalidConfiguration &&
                      h[2] == cudaErrorInvalidConfiguration && h[3] == cudaSuccess;
    std::printf("device last error: %d %d %d %d%s\n", h[0], h[1], h[2], h[3], good ? "" : " (want 0 9 9 0)");
    const bool asked = h[4] == cudaSuccess && h[5] == dev && h[6] == cudaSuccess && h[7] == count;
    std::printf("device-side cudaGetDevice: %s\n", asked ? "as the host's" : "WRONG");
    ok = ok && good && asked;
    CK(cudaFree(d));
  }
  return ok;
}

int main() {
  const bool ok = run();
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
