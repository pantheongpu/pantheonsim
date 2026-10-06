// Memory-test patterns over simulated device memory, written for this
// repository from the textbook algorithms (walking ones and zeros, address in
// address, checkerboard, moving inversions, seeded random data, block copies
// and strided sweeps). It knows nothing of VirtualGPU: a kernel writes a
// pattern, a second launch verifies it, and the program prints how many words
// read back wrong. A clean machine reports 0 for every test.
//
//   memtest_patterns            the whole suite on a cudaMalloc buffer
//   memtest_patterns mapped     the same suite on mapped, pinned host memory
//   memtest_patterns flip       one bit flipped by the host between a write
//                               and its verify: the verify must find exactly it
//   memtest_patterns wild       a verify kernel handed a pointer to nothing
//
// The buffer is the program's first allocation, so it sits at the start of
// device memory, where `vgpu fault stuck --offset` can aim at it.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

static const int kThreads = 128;
static const size_t kWords = 16384;  // 64 KiB: small, the simulator is a CPU

// A hash of (index, seed): a cheap stand-in for a random sequence that the
// host and the device both compute, so the expected value is never stored.
__host__ __device__ static inline uint32_t mix(uint32_t i, uint32_t seed) {
  uint32_t x = i * 0x9E3779B1u + seed;
  x ^= x >> 16; x *= 0x85EBCA6Bu;
  x ^= x >> 13; x *= 0xC2B2AE35u;
  x ^= x >> 16;
  return x;
}

// Every checker adds to *err and remembers the lowest bad index.
__device__ static inline void bad(unsigned long long* err, unsigned long long* first, size_t i) {
  atomicAdd(err, 1ull);
  atomicMin(first, static_cast<unsigned long long>(i));
}

__global__ void fill_const(uint32_t* b, size_t n, uint32_t p) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) b[i] = p;
}
__global__ void check_const(const uint32_t* b, size_t n, uint32_t p, unsigned long long* err,
                            unsigned long long* first) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n && b[i] != p) bad(err, first, i);
}

// Checkerboard: even words hold p, odd words its inverse.
__global__ void fill_checker(uint32_t* b, size_t n, uint32_t p) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) b[i] = (i & 1) ? ~p : p;
}
__global__ void check_checker(const uint32_t* b, size_t n, uint32_t p, unsigned long long* err,
                              unsigned long long* first) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n && b[i] != ((i & 1) ? ~p : p)) bad(err, first, i);
}

// Address in address: each 64-bit word holds its own address.
__global__ void fill_addr(unsigned long long* b, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) b[i] = reinterpret_cast<unsigned long long>(b + i);
}
__global__ void check_addr(const unsigned long long* b, size_t n, unsigned long long* err,
                           unsigned long long* first) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n && b[i] != reinterpret_cast<unsigned long long>(b + i)) bad(err, first, i);
}

__global__ void fill_rand(uint32_t* b, size_t n, uint32_t seed) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) b[i] = mix(static_cast<uint32_t>(i), seed);
}
__global__ void check_rand(const uint32_t* b, size_t n, uint32_t seed, unsigned long long* err,
                           unsigned long long* first) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n && b[i] != mix(static_cast<uint32_t>(i), seed)) bad(err, first, i);
}

// One pass of a moving inversion: each word is checked against `expect`, then
// overwritten with `next`. `down` sweeps from the top of the buffer.
__global__ void invert_pass(uint32_t* b, size_t n, uint32_t expect, uint32_t next, int down,
                            unsigned long long* err, unsigned long long* first) {
  size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= n) return;
  size_t i = down ? n - 1 - t : t;
  if (b[i] != expect) bad(err, first, i);
  b[i] = next;
}

// Strided sweeps: thread t touches word (t * stride) mod n, a permutation when
// stride is odd and n a power of two, so the same words are reached in a
// different order than a linear sweep.
__global__ void fill_stride(uint32_t* b, size_t n, size_t stride, uint32_t seed) {
  size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= n) return;
  size_t i = (t * stride) & (n - 1);
  b[i] = mix(static_cast<uint32_t>(i), seed);
}
__global__ void check_stride(const uint32_t* b, size_t n, size_t stride, uint32_t seed,
                             unsigned long long* err, unsigned long long* first) {
  size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= n) return;
  size_t i = (t * stride) & (n - 1);
  if (b[i] != mix(static_cast<uint32_t>(i), seed)) bad(err, first, i);
}

__global__ void copy_words(const uint32_t* from, uint32_t* to, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) to[i] = from[i];
}

static unsigned grid(size_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

struct Harness {
  uint32_t* buf = nullptr;    // the memory under test, n words
  uint32_t* spare = nullptr;  // a second buffer of n words, for the copy tests
  unsigned long long* cnt = nullptr;  // [0] errors, [1] lowest bad index
  size_t n = kWords;
  unsigned long long total = 0;
  bool launch_failed = false;

  bool sync(const char* what) {
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      printf("launch failed: %d %s (%s)\n", static_cast<int>(e), cudaGetErrorName(e), what);
      launch_failed = true;
      return false;
    }
    return true;
  }
  void clear() {
    unsigned long long init[2] = {0, ~0ull};
    cudaMemcpy(cnt, init, sizeof init, cudaMemcpyHostToDevice);
  }
  // Reads the counters; prints the line for one test and adds to the total.
  unsigned long long report(const char* name) {
    unsigned long long r[2] = {0, 0};
    cudaMemcpy(r, cnt, sizeof r, cudaMemcpyDeviceToHost);
    if (r[0]) printf("test %-22s errors: %llu (first at word %llu)\n", name, r[0], r[1]);
    else      printf("test %-22s errors: 0\n", name);
    total += r[0];
    clear();
    return r[0];
  }

  void walking(bool ones) {
    for (int bit = 0; bit < 32; ++bit) {
      const uint32_t p = ones ? (1u << bit) : ~(1u << bit);
      fill_const<<<grid(n), kThreads>>>(buf, n, p);
      check_const<<<grid(n), kThreads>>>(buf, n, p, cnt, cnt + 1);
    }
    sync("walking");
    report(ones ? "walking ones" : "walking zeros");
  }
  void checkerboard() {
    for (uint32_t p : {0xAAAAAAAAu, 0x55555555u}) {
      fill_checker<<<grid(n), kThreads>>>(buf, n, p);
      check_checker<<<grid(n), kThreads>>>(buf, n, p, cnt, cnt + 1);
    }
    // Plain solid patterns as well: all zeros, all ones, and a byte pattern the
    // runtime's memset writes rather than a kernel.
    for (uint32_t p : {0u, 0xFFFFFFFFu}) {
      fill_const<<<grid(n), kThreads>>>(buf, n, p);
      check_const<<<grid(n), kThreads>>>(buf, n, p, cnt, cnt + 1);
    }
    cudaMemset(buf, 0xA5, n * sizeof(uint32_t));
    check_const<<<grid(n), kThreads>>>(buf, n, 0xA5A5A5A5u, cnt, cnt + 1);
    sync("checkerboard");
    report("checkerboard/inverse");
  }
  void address_in_address() {
    const size_t words64 = n / 2;
    fill_addr<<<grid(words64), kThreads>>>(reinterpret_cast<unsigned long long*>(buf), words64);
    check_addr<<<grid(words64), kThreads>>>(reinterpret_cast<unsigned long long*>(buf), words64,
                                            cnt, cnt + 1);
    sync("address");
    report("address in address");
  }
  void moving_inversions() {
    for (uint32_t p : {0x00000000u, 0x80808080u, 0xFEFEFEFEu}) {
      fill_const<<<grid(n), kThreads>>>(buf, n, p);
      invert_pass<<<grid(n), kThreads>>>(buf, n, p, ~p, 0, cnt, cnt + 1);   // up: p -> ~p
      invert_pass<<<grid(n), kThreads>>>(buf, n, ~p, p, 1, cnt, cnt + 1);   // down: ~p -> p
      invert_pass<<<grid(n), kThreads>>>(buf, n, p, p, 0, cnt, cnt + 1);    // a last read of p
    }
    sync("moving inversions");
    report("moving inversions");
  }
  void random_seeded() {
    for (uint32_t seed : {1u, 0xC0FFEEu, 0xDEADBEEFu}) {
      fill_rand<<<grid(n), kThreads>>>(buf, n, seed);
      check_rand<<<grid(n), kThreads>>>(buf, n, seed, cnt, cnt + 1);
    }
    sync("random");
    report("random seeded");
    // The same sequence checked on the host from what memory holds.
    uint32_t* host = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
    unsigned long long host_bad = 0;
    if (cudaMemcpy(host, buf, n * sizeof(uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess)
      for (size_t i = 0; i < n; ++i) host_bad += host[i] != mix(static_cast<uint32_t>(i), 0xDEADBEEFu);
    else
      host_bad = n;
    free(host);
    printf("test %-22s errors: %llu\n", "random, host verify", host_bad);
    total += host_bad;
  }
  void block_copy() {
    fill_rand<<<grid(n), kThreads>>>(buf, n, 77u);
    cudaMemset(spare, 0, n * sizeof(uint32_t));
    copy_words<<<grid(n), kThreads>>>(buf, spare, n);  // by kernel
    check_rand<<<grid(n), kThreads>>>(spare, n, 77u, cnt, cnt + 1);
    cudaMemset(spare, 0, n * sizeof(uint32_t));
    cudaMemcpy(spare, buf, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice);  // by the runtime
    check_rand<<<grid(n), kThreads>>>(spare, n, 77u, cnt, cnt + 1);
    sync("block copy");
    report("block copy");
    // The top half of buf copied to the bottom of spare, then that half
    // copied up by a kernel; the host checks both halves.
    cudaMemcpy(spare, buf + n / 2, (n / 2) * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    copy_words<<<grid(n / 2), kThreads>>>(spare, spare + n / 2, n / 2);
    sync("halves");
    uint32_t* host = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
    unsigned long long host_bad = 0;
    if (cudaMemcpy(host, spare, n * sizeof(uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess) {
      for (size_t i = 0; i < n; ++i)
        host_bad += host[i] != mix(static_cast<uint32_t>(i % (n / 2) + n / 2), 77u);
    } else {
      host_bad = n;
    }
    free(host);
    printf("test %-22s errors: %llu\n", "block copy (halves)", host_bad);
    total += host_bad;
  }
  void strides() {
    const size_t strides[] = {1, 3, 17, 129, 1021, n - 1};  // all odd
    for (size_t s : strides) {
      fill_stride<<<grid(n), kThreads>>>(buf, n, s, 5u);
      check_stride<<<grid(n), kThreads>>>(buf, n, s, 5u, cnt, cnt + 1);
    }
    // Written in one order, read back in another: a stride-17 fill, verified
    // by a linear sweep of the same function of the index.
    fill_stride<<<grid(n), kThreads>>>(buf, n, 17, 9u);
    check_rand<<<grid(n), kThreads>>>(buf, n, 9u, cnt, cnt + 1);
    sync("strides");
    report("block/stride");
  }
};

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "all";
  Harness h;
  const size_t bytes = h.n * sizeof(uint32_t);
  bool mapped = !strcmp(mode, "mapped");
  if (mapped) {
    cudaSetDeviceFlags(cudaDeviceMapHost);
    void* p = nullptr;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocMapped) != cudaSuccess) { printf("SKIP: no mapped memory\n"); return 0; }
    void* d = nullptr;
    if (cudaHostGetDevicePointer(&d, p, 0) != cudaSuccess) { printf("device pointer failed\n"); return 2; }
    h.buf = static_cast<uint32_t*>(d);
  } else {
    cudaMalloc(&h.buf, bytes);  // first, so it is at the start of device memory
  }
  cudaMalloc(&h.spare, bytes);
  cudaMalloc(&h.cnt, 2 * sizeof(unsigned long long));
  if (!h.buf || !h.spare || !h.cnt) { printf("allocation failed\n"); return 2; }
  h.clear();

  int rc = 0;
  if (!strcmp(mode, "flip")) {
    // The host plays a failing cell: it flips one bit of one word between the
    // write and the verify, and the verify has to find that word and no other.
    const size_t word = 4321;
    for (int bit : {0, 17, 31}) {
      fill_rand<<<grid(h.n), kThreads>>>(h.buf, h.n, 3u);
      h.sync("fill");
      uint32_t v = 0;
      cudaMemcpy(&v, h.buf + word, sizeof v, cudaMemcpyDeviceToHost);
      v ^= 1u << bit;
      cudaMemcpy(h.buf + word, &v, sizeof v, cudaMemcpyHostToDevice);
      check_rand<<<grid(h.n), kThreads>>>(h.buf, h.n, 3u, h.cnt, h.cnt + 1);
      h.sync("verify");
      unsigned long long r[2];
      cudaMemcpy(r, h.cnt, sizeof r, cudaMemcpyDeviceToHost);
      printf("flip bit %2d of word %zu: %llu error(s), first at %llu\n", bit, word, r[0], r[1]);
      if (r[0] != 1 || r[1] != word) rc = 1;
      h.clear();
    }
    printf(rc ? "flip: FAIL\n" : "flip: PASS\n");
  } else if (!strcmp(mode, "wild")) {
    // A verify kernel pointed at memory that was never allocated.
    check_const<<<grid(h.n), kThreads>>>(h.buf + (1ull << 30), h.n, 0, h.cnt, h.cnt + 1);
    h.sync("wild");
    rc = h.launch_failed ? 3 : 0;
  } else {
    h.walking(true);
    h.walking(false);
    h.address_in_address();
    h.checkerboard();
    h.moving_inversions();
    h.random_seeded();
    h.block_copy();
    h.strides();
    printf("total errors: %llu\n", h.total);
    if (h.launch_failed) rc = 3;
    else if (h.total) rc = 1;
    if (!rc) printf("PASS\n");
  }
  if (!mapped) cudaFree(h.buf);
  cudaFree(h.spare);
  cudaFree(h.cnt);
  return rc;
}
