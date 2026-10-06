// Memory-test patterns over simulated AMD device memory: walking ones and
// zeros, address in address, checkerboard and its inverse, solid fills, moving
// inversions, seeded random data, block copies and strided sweeps. Written
// for this repository, in HIP, from the textbook algorithms. It knows nothing
// of VirtualGPU. A kernel writes a pattern, a second launch verifies it, and
// the program prints how many words read back wrong; a healthy machine
// reports 0 for every test, on a wave of 32 or of 64 alike (nothing here
// depends on the wave size, which the program only reports).
//
//   memtest            the whole suite on a hipMalloc buffer
//   memtest mapped     the same suite on mapped, pinned host memory
//   memtest flip       one bit flipped by the host between a write and its
//                      verify: the verify must find exactly that word
//   memtest wild       a verify kernel handed a pointer to nothing
//
// The buffer is the program's first allocation, so it sits at the start of
// device memory, where `vgpu fault stuck --offset` can aim at it.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr unsigned kBlock = 256;
constexpr size_t kWords = 16384;  // 64 KiB: small, the simulator is a CPU

// A hash of (index, seed) that host and device both compute, so the expected
// value of a "random" word is never stored anywhere.
__host__ __device__ inline uint32_t scramble(uint32_t i, uint32_t seed) {
  uint32_t x = (i ^ 0x5bd1e995u) * 0x9E3779B1u + seed;
  x = (x ^ (x >> 15)) * 0x2c1b3c6du;
  x = (x ^ (x >> 12)) * 0x297a2d39u;
  return x ^ (x >> 15);
}

// The word a pattern puts at index i. Each is a type, so the same fill and
// verify kernels serve every pattern.
struct Solid {
  uint32_t v;
  __host__ __device__ uint32_t at(size_t) const { return v; }
};
struct Checker {  // even words hold v, odd words its complement
  uint32_t v;
  __host__ __device__ uint32_t at(size_t i) const { return (i & 1) ? ~v : v; }
};
struct Random {
  uint32_t seed;
  __host__ __device__ uint32_t at(size_t i) const { return scramble(static_cast<uint32_t>(i), seed); }
};

struct Tally {  // lives in device memory: [0] bad words, [1] lowest bad index
  unsigned long long count;
  unsigned long long first;
};

__device__ inline void note(Tally* t, size_t i) {
  atomicAdd(&t->count, 1ull);
  atomicMin(&t->first, static_cast<unsigned long long>(i));
}

__device__ inline size_t lane_index() { return static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; }

template <class P>
__global__ void fill(uint32_t* m, size_t n, P p) {
  const size_t i = lane_index();
  if (i < n) m[i] = p.at(i);
}
template <class P>
__global__ void verify(const uint32_t* m, size_t n, P p, Tally* t) {
  const size_t i = lane_index();
  if (i < n && m[i] != p.at(i)) note(t, i);
}

// Strided order: thread k handles word (k * stride) & (n - 1), a permutation
// for an odd stride and a power-of-two n, so the words are reached in an order
// other than a linear sweep.
template <class P>
__global__ void fill_strided(uint32_t* m, size_t n, size_t stride, P p) {
  const size_t k = lane_index();
  if (k >= n) return;
  const size_t i = (k * stride) & (n - 1);
  m[i] = p.at(i);
}
template <class P>
__global__ void verify_strided(const uint32_t* m, size_t n, size_t stride, P p, Tally* t) {
  const size_t k = lane_index();
  if (k >= n) return;
  const size_t i = (k * stride) & (n - 1);
  if (m[i] != p.at(i)) note(t, i);
}

// Every 64-bit word holds its own address.
__global__ void fill_self(unsigned long long* m, size_t n) {
  const size_t i = lane_index();
  if (i < n) m[i] = reinterpret_cast<unsigned long long>(m + i);
}
__global__ void verify_self(const unsigned long long* m, size_t n, Tally* t) {
  const size_t i = lane_index();
  if (i < n && m[i] != reinterpret_cast<unsigned long long>(m + i)) note(t, i);
}

// A moving inversion's sweep: each word must read `want`, and is then
// rewritten as `then`. `from_top` sweeps downwards from the last word.
__global__ void sweep(uint32_t* m, size_t n, uint32_t want, uint32_t then, bool from_top, Tally* t) {
  const size_t k = lane_index();
  if (k >= n) return;
  const size_t i = from_top ? n - 1 - k : k;
  if (m[i] != want) note(t, i);
  m[i] = then;
}

__global__ void copy_words(const uint32_t* from, uint32_t* to, size_t n) {
  const size_t i = lane_index();
  if (i < n) to[i] = from[i];
}

unsigned blocks_for(size_t n) { return static_cast<unsigned>((n + kBlock - 1) / kBlock); }

#define LAUNCH(kernel, n, ...) hipLaunchKernelGGL(kernel, dim3(blocks_for(n)), dim3(kBlock), 0, 0, __VA_ARGS__)

struct Suite {
  uint32_t* mem = nullptr;    // the memory under test
  uint32_t* spare = nullptr;  // same size, for the copy tests
  Tally* tally = nullptr;
  size_t n = kWords;
  unsigned long long total = 0;
  bool launch_failed = false;

  void sync(const char* what) {
    const hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
      printf("launch failed: %d %s (%s)\n", static_cast<int>(e), hipGetErrorName(e), what);
      launch_failed = true;
    }
  }
  void reset() {
    const Tally zero = {0, ~0ull};
    (void)hipMemcpy(tally, &zero, sizeof zero, hipMemcpyHostToDevice);
  }
  Tally read() {
    Tally t = {0, 0};
    (void)hipMemcpy(&t, tally, sizeof t, hipMemcpyDeviceToHost);
    return t;
  }
  void finish(const char* name) {
    const Tally t = read();
    if (t.count) printf("test %-22s errors: %llu (first at word %llu)\n", name, t.count, t.first);
    else printf("test %-22s errors: 0\n", name);
    total += t.count;
    reset();
  }
  // The host's own check of a buffer's contents against a pattern.
  template <class P>
  unsigned long long host_check(const uint32_t* dev, size_t words, P p, size_t base = 0) {
    std::vector<uint32_t> h(words);
    if (hipMemcpy(h.data(), dev, words * sizeof(uint32_t), hipMemcpyDeviceToHost) != hipSuccess) return words;
    unsigned long long bad = 0;
    for (size_t i = 0; i < words; ++i) bad += h[i] != p.at(base + i);
    return bad;
  }

  void walking(bool ones) {
    for (int bit = 0; bit < 32; ++bit) {
      const Solid p{ones ? (1u << bit) : ~(1u << bit)};
      LAUNCH(fill<Solid>, n, mem, n, p);
      LAUNCH(verify<Solid>, n, mem, n, p, tally);
    }
    sync("walking");
    finish(ones ? "walking ones" : "walking zeros");
  }
  void self_address() {
    auto* m64 = reinterpret_cast<unsigned long long*>(mem);
    const size_t w = n / 2;
    LAUNCH(fill_self, w, m64, w);
    LAUNCH(verify_self, w, m64, w, tally);
    sync("address");
    finish("address in address");
  }
  void checkerboard_and_solid() {
    for (uint32_t v : {0xAAAAAAAAu, 0x55555555u}) {
      LAUNCH(fill<Checker>, n, mem, n, Checker{v});
      LAUNCH(verify<Checker>, n, mem, n, Checker{v}, tally);
    }
    for (uint32_t v : {0u, 0xFFFFFFFFu}) {
      LAUNCH(fill<Solid>, n, mem, n, Solid{v});
      LAUNCH(verify<Solid>, n, mem, n, Solid{v}, tally);
    }
    // A fill the runtime does, not a kernel.
    (void)hipMemset(mem, 0x5A, n * sizeof(uint32_t));
    LAUNCH(verify<Solid>, n, mem, n, Solid{0x5A5A5A5Au}, tally);
    sync("checkerboard");
    finish("checkerboard/inverse");
  }
  void moving_inversions() {
    for (uint32_t v : {0x00000000u, 0x01010101u, 0x7F7F7F7Fu}) {
      LAUNCH(fill<Solid>, n, mem, n, Solid{v});
      LAUNCH(sweep, n, mem, n, v, ~v, false, tally);   // upward: v -> ~v
      LAUNCH(sweep, n, mem, n, ~v, v, true, tally);    // downward: ~v -> v
      LAUNCH(sweep, n, mem, n, v, v, false, tally);    // a last read of v
    }
    sync("moving inversions");
    finish("moving inversions");
  }
  void random_data() {
    for (uint32_t seed : {2u, 0xBADC0DEu, 0x13579BDFu}) {
      LAUNCH(fill<Random>, n, mem, n, Random{seed});
      LAUNCH(verify<Random>, n, mem, n, Random{seed}, tally);
    }
    sync("random");
    finish("random seeded");
    const unsigned long long bad = host_check(mem, n, Random{0x13579BDFu});
    printf("test %-22s errors: %llu\n", "random, host verify", bad);
    total += bad;
  }
  void copies() {
    const Random p{99u};
    LAUNCH(fill<Random>, n, mem, n, p);
    (void)hipMemset(spare, 0, n * sizeof(uint32_t));
    LAUNCH(copy_words, n, mem, spare, n);  // by kernel
    LAUNCH(verify<Random>, n, spare, n, p, tally);
    (void)hipMemset(spare, 0, n * sizeof(uint32_t));
    (void)hipMemcpy(spare, mem, n * sizeof(uint32_t), hipMemcpyDeviceToDevice);  // by the runtime
    LAUNCH(verify<Random>, n, spare, n, p, tally);
    sync("copies");
    finish("block copy");
    // The top half of mem to the bottom of spare, then up again by a kernel;
    // the host checks both halves against what the top half of mem held.
    (void)hipMemcpy(spare, mem + n / 2, (n / 2) * sizeof(uint32_t), hipMemcpyDeviceToDevice);
    LAUNCH(copy_words, n / 2, spare, spare + n / 2, n / 2);
    sync("halves");
    unsigned long long bad = host_check(spare, n / 2, p, n / 2) + host_check(spare + n / 2, n / 2, p, n / 2);
    printf("test %-22s errors: %llu\n", "block copy (halves)", bad);
    total += bad;
  }
  void strides() {
    for (size_t s : {size_t(1), size_t(3), size_t(17), size_t(129), size_t(1021), n - 1}) {
      LAUNCH(fill_strided<Random>, n, mem, n, s, Random{7u});
      LAUNCH(verify_strided<Random>, n, mem, n, s, Random{7u}, tally);
    }
    // Written in one order and read back in another.
    LAUNCH(fill_strided<Random>, n, mem, n, size_t(17), Random{11u});
    LAUNCH(verify<Random>, n, mem, n, Random{11u}, tally);
    sync("strides");
    finish("block/stride");
  }
};

}  // namespace

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "all";
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) == hipSuccess)
    printf("device %s warp %d\n", prop.name, prop.warpSize);

  Suite s;
  const size_t bytes = s.n * sizeof(uint32_t);
  const bool mapped = !strcmp(mode, "mapped");
  void* host_mem = nullptr;
  if (mapped) {
    if (hipHostMalloc(&host_mem, bytes, hipHostMallocMapped) != hipSuccess) {
      printf("SKIP: no mapped memory\n");
      return 0;
    }
    void* dev = nullptr;
    if (hipHostGetDevicePointer(&dev, host_mem, 0) != hipSuccess) {
      printf("device pointer failed\n");
      return 2;
    }
    s.mem = static_cast<uint32_t*>(dev);
  } else {
    (void)hipMalloc(reinterpret_cast<void**>(&s.mem), bytes);  // first: the start of device memory
  }
  (void)hipMalloc(reinterpret_cast<void**>(&s.spare), bytes);
  (void)hipMalloc(reinterpret_cast<void**>(&s.tally), sizeof(Tally));
  if (!s.mem || !s.spare || !s.tally) {
    printf("allocation failed\n");
    return 2;
  }
  s.reset();

  int rc = 0;
  if (!strcmp(mode, "flip")) {
    // The host plays a failing cell: it flips one bit of one word between the
    // write and the verify, and the verify must find that word and no other.
    const size_t word = 3001;
    for (int bit : {0, 13, 31}) {
      LAUNCH(fill<Random>, s.n, s.mem, s.n, Random{4u});
      s.sync("fill");
      uint32_t v = 0;
      (void)hipMemcpy(&v, s.mem + word, sizeof v, hipMemcpyDeviceToHost);
      v ^= 1u << bit;
      (void)hipMemcpy(s.mem + word, &v, sizeof v, hipMemcpyHostToDevice);
      LAUNCH(verify<Random>, s.n, s.mem, s.n, Random{4u}, s.tally);
      s.sync("verify");
      const Tally t = s.read();
      printf("flip bit %2d of word %zu: %llu error(s), first at %llu\n", bit, word, t.count, t.first);
      if (t.count != 1 || t.first != word) rc = 1;
      s.reset();
    }
    printf(rc ? "flip: FAIL\n" : "flip: PASS\n");
  } else if (!strcmp(mode, "wild")) {
    // A verify kernel pointed at memory that was never allocated.
    LAUNCH(verify<Solid>, s.n, s.mem + (1ull << 30), s.n, Solid{0}, s.tally);
    s.sync("wild");
    rc = s.launch_failed ? 3 : 0;
  } else {
    s.walking(true);
    s.walking(false);
    s.self_address();
    s.checkerboard_and_solid();
    s.moving_inversions();
    s.random_data();
    s.copies();
    s.strides();
    printf("total errors: %llu\n", s.total);
    if (s.launch_failed) rc = 3;
    else if (s.total) rc = 1;
    if (!rc) printf("PASS\n");
  }
  if (!mapped) (void)hipFree(s.mem);
  else (void)hipHostFree(host_mem);
  (void)hipFree(s.spare);
  (void)hipFree(s.tally);
  return rc;
}
