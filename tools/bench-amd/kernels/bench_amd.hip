// The AMD engine's benchmark workloads: small self-checking HIP programs, one
// binary, `bench_amd <workload> <quick|full>`. Built by tools/bench-amd/bench.py
// with hipcc, once per target (gfx942 for the simulated MI300X, gfx1030 for the
// RX 6800). Every workload computes on the device and compares with a host
// reference bit for bit (floating point by memcmp, integer-valued data wherever
// a summation order could change bits); the last line is `PASS` only when every
// result matched, so a change that breaks a result fails instead of looking fast.
//
// Before the last line there is one `BENCH device_cpu_s=.. device_wall_s=..
// units=.. unit=..` line: the CPU time (all threads) and wall time of the
// device phase only, which excludes process start-up, allocation and the host
// reference, so instructions over device_cpu_s is the interpreter's rate.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#define HC(x)                                                                                  \
  do {                                                                                         \
    hipError_t e_ = (x);                                                                       \
    if (e_ != hipSuccess) {                                                                    \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
      std::exit(3);                                                                            \
    }                                                                                          \
  } while (0)

// A launch whose error is not looked at turns into a wrong answer later; check it where it happens.
#define LAUNCH(...)         \
  do {                      \
    hipLaunchKernelGGL(__VA_ARGS__); \
    HC(hipGetLastError());  \
  } while (0)

static bool g_ok = true;
static void expect_eq(const char* what, const void* a, const void* b, size_t bytes) {
  if (std::memcmp(a, b, bytes) != 0) {
    std::fprintf(stderr, "MISMATCH in %s\n", what);
    g_ok = false;
  }
}
static void expect_true(const char* what, bool v) {
  if (!v) {
    std::fprintf(stderr, "FAILED %s\n", what);
    g_ok = false;
  }
}

static double cpu_s() {
  timespec t;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static double wall_s() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
struct Phase {
  double c0, w0;
  Phase() : c0(cpu_s()), w0(wall_s()) {}
  void finish(uint64_t units, const char* unit) const {
    std::printf("BENCH device_cpu_s=%.6f device_wall_s=%.6f units=%llu unit=%s\n", cpu_s() - c0, wall_s() - w0,
                static_cast<unsigned long long>(units), unit);
  }
};

template <typename T>
static T* dev_copy(const std::vector<T>& h) {
  T* d;
  HC(hipMalloc(&d, h.size() * sizeof(T)));
  HC(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
  return d;
}
template <typename T>
static std::vector<T> dev_read(const T* d, size_t n) {
  std::vector<T> h(n);
  HC(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
  return h;
}

// ---------------------------------------------------------------- valu

__host__ __device__ static inline uint32_t rotl32(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }
__host__ __device__ static inline uint32_t popc32(uint32_t x) {
  x = x - ((x >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  return (((x + (x >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24;
}
__host__ __device__ static inline uint32_t int_step(uint32_t x, uint32_t i) {
  x = x * 1664525u + 1013904223u;
  x ^= x >> 13;
  x = rotl32(x, 7) + (x & 0xFF00FFu);
  x = (x << 3) ^ (x >> 5) ^ i;
  x += popc32(x) * 2654435761u;
  return x;
}
__global__ void k_valu_int(uint32_t* out, uint32_t iters) {
  uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t x = t * 2654435761u + 1;
  for (uint32_t i = 0; i < iters; ++i) x = int_step(x, i);
  out[t] = x;
}

__host__ __device__ static inline float f32_step(float x, float y) {
  x = fmaf(x, 0.999f, y);
  y = fmaf(y, 0.5f, x * 0.001f);
  x = fmaf(x, y, 0.25f);
  x = x * 0.5f + 0.125f;
  return x + y * 0.0625f;
}
__global__ void k_valu_f32(float* out, uint32_t iters) {
  uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  float x = 1.0f + (t & 15) * 0.01f, y = 0.5f + (t & 7) * 0.02f;
  for (uint32_t i = 0; i < iters; ++i) {
    x = f32_step(x, y);
    y = fmaf(y, 0.75f, 0.1f);
  }
  out[t] = x;
}

__global__ void k_valu_f64(double* out, uint32_t iters) {
  uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  double x = 1.0 + (t & 15) * 0.01, y = 0.5 + (t & 7) * 0.02;
  for (uint32_t i = 0; i < iters; ++i) {
    x = fma(x, 0.999, y);
    y = fma(y, 0.5, x * 0.001);
    x = fma(x, 0.5, 0.125);
  }
  out[t] = x + y;
}

// ---------------------------------------------------------------- matrix

#if defined(__gfx942__) || defined(__gfx90a__) || defined(__gfx950__)
#define DEVICE_MFMA 1
#endif
typedef __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16 half4;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float float4_v;
// D = n * (A x B) with A,B 16x16 holding small integers (exact in f16 and f32),
// accumulated n times with v_mfma_f32_16x16x16_f16. Lane l holds row l%16 of A
// and column l%16 of B at k = 4*(l/16)..+3, and returns D rows 4*(l/16)..+3 of
// column l%16.
__global__ void k_mfma(const _Float16* A, const _Float16* B, float* D, uint32_t n) {
#ifdef DEVICE_MFMA
  const uint32_t l = threadIdx.x & 63, wave = blockIdx.x;
  half4 a, b;
  for (int i = 0; i < 4; ++i) {
    a[i] = A[(l % 16) * 16 + (l / 16) * 4 + i];
    b[i] = B[((l / 16) * 4 + i) * 16 + (l % 16)];
  }
  float4_v acc = {0, 0, 0, 0};
  for (uint32_t i = 0; i < n; ++i) acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);
  for (int i = 0; i < 4; ++i) D[wave * 256 + ((l / 16) * 4 + i) * 16 + (l % 16)] = acc[i];
#endif
}


// The other matrix forms (docs/performance.md, "The matrix instructions"): each computes D = n * (A x B) for a 16x16 tile of small
// integers, which every form represents and sums exactly, and compares with the host's. MI300X: integer, 8-bit float, bfloat16,
// double and sparse MFMA. RX 7900 XTX and RX 9070 XT: WMMA (binary16 into float, and 8-bit integers). Lane layouts as in k_mfma:
// A row l % 16 and B column l % 16, the lane's run of K after (l / 16) * (K / 4); D rows 4 * (l / 16) .. +3 of column l % 16 (the
// double form: rows l / 16 + 4r). gfx11's WMMA lane holds all of K, gfx12's the half of it (l / 16); D rows are 2i + l / 16 (gfx11) or
// i + 8 * (l / 16) (gfx12).
#if defined(__gfx942__) || defined(__gfx950__)
#define DEVICE_MFMA_MORE 1
#endif
typedef __attribute__((__vector_size__(4 * sizeof(int)))) int int4_v;
typedef __attribute__((__vector_size__(4 * sizeof(short)))) short short4_v;
typedef __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16 half8_v;
typedef __attribute__((__vector_size__(4 * sizeof(double)))) double double4_v;
typedef __attribute__((__vector_size__(8 * sizeof(float)))) float float8_v;
typedef __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16 half16_v;
typedef __attribute__((__vector_size__(8 * sizeof(int)))) int int8_v;

__global__ void k_mfma_i8(const int8_t* A, const int8_t* B, int* D, uint32_t n) {
#ifdef DEVICE_MFMA_MORE
  const uint32_t l = threadIdx.x & 63, wave = blockIdx.x;
  long a = 0, b = 0;
  for (int i = 0; i < 8; ++i) {
    a |= static_cast<long>(static_cast<uint8_t>(A[(l % 16) * 32 + (l / 16) * 8 + i])) << (8 * i);
    b |= static_cast<long>(static_cast<uint8_t>(B[((l / 16) * 8 + i) * 16 + (l % 16)])) << (8 * i);
  }
  int4_v acc = {0, 0, 0, 0};
  for (uint32_t i = 0; i < n; ++i) acc = __builtin_amdgcn_mfma_i32_16x16x32_i8(a, b, acc, 0, 0, 0);
  for (int i = 0; i < 4; ++i) D[wave * 256 + ((l / 16) * 4 + i) * 16 + (l % 16)] = acc[i];
#endif
}
__global__ void k_mfma_fp8(const uint8_t* A, const uint8_t* B, float* D, uint32_t n) {
#ifdef DEVICE_MFMA_MORE
  const uint32_t l = threadIdx.x & 63, wave = blockIdx.x;
  long a = 0, b = 0;
  for (int i = 0; i < 8; ++i) {
    a |= static_cast<long>(A[(l % 16) * 32 + (l / 16) * 8 + i]) << (8 * i);
    b |= static_cast<long>(B[((l / 16) * 8 + i) * 16 + (l % 16)]) << (8 * i);
  }
  float4_v acc = {0, 0, 0, 0};
  for (uint32_t i = 0; i < n; ++i) acc = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, acc, 0, 0, 0);
  for (int i = 0; i < 4; ++i) D[wave * 256 + ((l / 16) * 4 + i) * 16 + (l % 16)] = acc[i];
#endif
}
__global__ void k_mfma_bf16(const uint16_t* A, const uint16_t* B, float* D, uint32_t n) {
#ifdef DEVICE_MFMA_MORE
  const uint32_t l = threadIdx.x & 63, wave = blockIdx.x;
  short4_v a, b;
  for (int i = 0; i < 4; ++i) {
    a[i] = static_cast<short>(A[(l % 16) * 16 + (l / 16) * 4 + i]);
    b[i] = static_cast<short>(B[((l / 16) * 4 + i) * 16 + (l % 16)]);
  }
  float4_v acc = {0, 0, 0, 0};
  for (uint32_t i = 0; i < n; ++i) acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
  for (int i = 0; i < 4; ++i) D[wave * 256 + ((l / 16) * 4 + i) * 16 + (l % 16)] = acc[i];
#endif
}
__global__ void k_mfma_f64(const double* A, const double* B, double* D, uint32_t n) {
#ifdef DEVICE_MFMA_MORE
  const uint32_t l = threadIdx.x & 63, wave = blockIdx.x;
  const double a = A[(l % 16) * 4 + l / 16], b = B[(l / 16) * 16 + l % 16];
  double4_v acc = {0, 0, 0, 0};
  for (uint32_t i = 0; i < n; ++i) acc = __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, acc, 0, 0, 0);
  for (int i = 0; i < 4; ++i) D[wave * 256 + (l / 16 + 4 * i) * 16 + (l % 16)] = acc[i];
#endif
}
// 2:4 sparse A: a lane's four stored halves are the dense k offsets 0, 1, 4, 5 of its run of eight (index bits 0b01000100 ...).
__global__ void k_smfmac(const _Float16* A, const _Float16* B, float* D, uint32_t n) {
#ifdef DEVICE_MFMA_MORE
  const uint32_t l = threadIdx.x & 63, wave = blockIdx.x;
  half4 a;
  half8_v b;
  static const int kOff[4] = {0, 1, 4, 5};
  for (int i = 0; i < 4; ++i) a[i] = A[(l % 16) * 32 + (l / 16) * 8 + kOff[i]];
  for (int i = 0; i < 8; ++i) b[i] = B[((l / 16) * 8 + i) * 16 + (l % 16)];
  float4_v acc = {0, 0, 0, 0};
  for (uint32_t i = 0; i < n; ++i) acc = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a, b, acc, 0x44444444, 0, 0);
  for (int i = 0; i < 4; ++i) D[wave * 256 + ((l / 16) * 4 + i) * 16 + (l % 16)] = acc[i];
#endif
}
__global__ void k_wmma_f16(const _Float16* A, const _Float16* B, float* D, uint32_t n) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
  const uint32_t l = threadIdx.x & 31, wave = blockIdx.x;
  half16_v a, b;
  for (int i = 0; i < 16; ++i) a[i] = A[(l % 16) * 16 + i], b[i] = B[i * 16 + (l % 16)];
  float8_v acc = {0, 0, 0, 0, 0, 0, 0, 0};
  for (uint32_t i = 0; i < n; ++i) acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
  for (int i = 0; i < 8; ++i) D[wave * 256 + (2 * i + l / 16) * 16 + (l % 16)] = acc[i];
#elif defined(__gfx1200__) || defined(__gfx1201__)
  const uint32_t l = threadIdx.x & 31, wave = blockIdx.x;
  half8_v a, b;
  for (int i = 0; i < 8; ++i) a[i] = A[(l % 16) * 16 + (l / 16) * 8 + i], b[i] = B[((l / 16) * 8 + i) * 16 + (l % 16)];
  float8_v acc = {0, 0, 0, 0, 0, 0, 0, 0};
  for (uint32_t i = 0; i < n; ++i) acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
  for (int i = 0; i < 8; ++i) D[wave * 256 + (i + 8 * (l / 16)) * 16 + (l % 16)] = acc[i];
#endif
}
__global__ void k_wmma_iu8(const uint8_t* A, const uint8_t* B, int* D, uint32_t n) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
  const uint32_t l = threadIdx.x & 31, wave = blockIdx.x;
  int4_v a = {0, 0, 0, 0}, b = {0, 0, 0, 0};
  for (int i = 0; i < 16; ++i) {
    a[i / 4] |= static_cast<int>(A[(l % 16) * 16 + i]) << (8 * (i % 4));
    b[i / 4] |= static_cast<int>(B[i * 16 + (l % 16)]) << (8 * (i % 4));
  }
  int8_v acc = {0, 0, 0, 0, 0, 0, 0, 0};
  for (uint32_t i = 0; i < n; ++i) acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, false, b, acc, false);
  for (int i = 0; i < 8; ++i) D[wave * 256 + (2 * i + l / 16) * 16 + (l % 16)] = acc[i];
#elif defined(__gfx1200__) || defined(__gfx1201__)
  const uint32_t l = threadIdx.x & 31, wave = blockIdx.x;
  __attribute__((__vector_size__(2 * sizeof(int)))) int a = {0, 0}, b = {0, 0};
  for (int i = 0; i < 8; ++i) {
    a[i / 4] |= static_cast<int>(A[(l % 16) * 16 + (l / 16) * 8 + i]) << (8 * (i % 4));
    b[i / 4] |= static_cast<int>(B[((l / 16) * 8 + i) * 16 + (l % 16)]) << (8 * (i % 4));
  }
  int8_v acc = {0, 0, 0, 0, 0, 0, 0, 0};
  for (uint32_t i = 0; i < n; ++i) acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, false, b, acc, false);
  for (int i = 0; i < 8; ++i) D[wave * 256 + (i + 8 * (l / 16)) * 16 + (l % 16)] = acc[i];
#endif
}

// ---------------------------------------------------------------- memory

__global__ void k_mem_copy(const uint4* in, uint4* out, uint32_t n4, uint32_t reps) {
  uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t r = 0; r < reps; ++r)
    for (uint32_t i = t; i < n4; i += gridDim.x * blockDim.x) {
      uint4 v = in[i];
      v.x += r;
      v.y ^= v.x;
      v.z += v.y;
      v.w ^= v.z;
      out[i] = v;
    }
}
__global__ void k_mem_scalar(const uint32_t* in, uint32_t* out, uint32_t n, uint32_t reps) {
  uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t r = 0; r < reps; ++r)
    for (uint32_t i = t; i < n; i += gridDim.x * blockDim.x) out[i] = in[i] * 3u + r;
}
__global__ void k_mem_gather(const uint32_t* in, const uint32_t* idx, uint32_t* out, uint32_t n, uint32_t reps) {
  uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t r = 0; r < reps; ++r)
    for (uint32_t i = t; i < n; i += gridDim.x * blockDim.x) out[i] = in[idx[i]] + r;
}

// ---------------------------------------------------------------- LDS and barriers

__global__ void k_lds_reduce(const uint32_t* in, uint32_t* out, uint32_t reps) {
  __shared__ uint32_t s[256];
  uint32_t acc = 0;
  for (uint32_t r = 0; r < reps; ++r) {
    s[threadIdx.x] = in[blockIdx.x * 256 + threadIdx.x] + r;
    __syncthreads();
    for (uint32_t st = 128; st > 0; st >>= 1) {
      if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st];
      __syncthreads();
    }
    acc += s[0];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = acc;
}
__global__ void k_lds_transpose(const uint32_t* in, uint32_t* out, uint32_t reps) {
  __shared__ uint32_t tile[32][33];
  const uint32_t tx = threadIdx.x & 31, ty = threadIdx.x >> 5;  // 256 threads: 32 x 8
  const uint32_t base = blockIdx.x * 1024;
  for (uint32_t r = 0; r < reps; ++r) {
    for (uint32_t j = 0; j < 32; j += 8) tile[ty + j][tx] = in[base + (ty + j) * 32 + tx] + r;
    __syncthreads();
    for (uint32_t j = 0; j < 32; j += 8) out[base + (ty + j) * 32 + tx] = tile[tx][ty + j];
    __syncthreads();
  }
}

// ---------------------------------------------------------------- atomics

__global__ void k_atomics(uint32_t* one, uint32_t* bins, uint32_t* maxv, uint32_t* cas, uint32_t iters) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = 0; i < iters; ++i) {
    atomicAdd(one, 1u);
    atomicAdd(&bins[(t * 7 + i) & 255], 2u);
    atomicMax(maxv, t ^ i);
    uint32_t old = *cas, assumed;
    do {
      assumed = old;
      old = atomicCAS(cas, assumed, assumed + 3u);
    } while (old != assumed);
  }
}

// ---------------------------------------------------------------- control

__host__ __device__ static inline uint32_t collatz_mix(uint32_t t, uint32_t seedoff) {
  uint32_t acc = 0;
  for (uint32_t rep = 0; rep < 4; ++rep) {
    uint64_t x = t + 1 + seedoff + rep * 977u;  // collatz: the trip count differs per lane
    uint32_t steps = 0;
    while (x != 1) {
      x = (x & 1) ? 3 * x + 1 : x >> 1;
      ++steps;
    }
    acc += steps;
    switch ((t + rep) & 3) {
      case 0: acc ^= t * 31u; break;
      case 1: acc += rotl32(t, 5); break;
      case 2: acc = acc * 3u + 1u; break;
      default: acc -= t; break;
    }
  }
  return acc;
}
__global__ void k_divergent(uint32_t* out, uint32_t n, uint32_t seedoff) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) out[t] = collatz_mix(t, seedoff);
}

// ---------------------------------------------------------------- wave level

__global__ void k_wave(uint32_t* out, uint32_t iters) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t lane = threadIdx.x % warpSize;
  uint32_t v = t * 2654435761u + 7;
  uint32_t acc = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    uint32_t a = __shfl_xor(v, 1);
    uint32_t b = __shfl_xor(v, 16);
    uint32_t c = __shfl(v, (lane + i) % warpSize);
    unsigned long long m = __ballot((v + i) & 1);
    acc += __popcll(m) + (uint32_t)__any(v & 4) + (uint32_t)__all(v | 1);
    v = rotl32(v ^ a, 3) + b - c + i;
  }
  out[t] = v + acc;
}

// ---------------------------------------------------------------- launch and API rates

__global__ void k_tiny(uint32_t* p, uint32_t v) {
  if (threadIdx.x == 0) p[blockIdx.x] = v + blockIdx.x;
}

// ---------------------------------------------------------------- the workloads

static uint32_t scale(const std::string& size, uint32_t quick, uint32_t full) { return size == "quick" ? quick : full; }
static int g_wave = 64;
static char g_arch[256] = "";

static void run_valu_int(const std::string& size) {
  const uint32_t blocks = 64, threads = 256, iters = scale(size, 200, 2000), n = blocks * threads;
  uint32_t* d;
  HC(hipMalloc(&d, n * 4));
  Phase p;
  LAUNCH(k_valu_int, dim3(blocks), dim3(threads), 0, 0, d, iters);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> got = dev_read(d, n), ref(n);
  for (uint32_t t = 0; t < n; ++t) {
    uint32_t x = t * 2654435761u + 1;
    for (uint32_t i = 0; i < iters; ++i) x = int_step(x, i);
    ref[t] = x;
  }
  expect_eq("valu_int", got.data(), ref.data(), n * 4);
}
static void run_valu_f32(const std::string& size) {
  const uint32_t blocks = 64, threads = 256, iters = scale(size, 200, 2000), n = blocks * threads;
  float* d;
  HC(hipMalloc(&d, n * 4));
  Phase p;
  LAUNCH(k_valu_f32, dim3(blocks), dim3(threads), 0, 0, d, iters);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<float> got = dev_read(d, n), ref(n);
  for (uint32_t t = 0; t < n; ++t) {
    float x = 1.0f + (t & 15) * 0.01f, y = 0.5f + (t & 7) * 0.02f;
    for (uint32_t i = 0; i < iters; ++i) {
      x = f32_step(x, y);
      y = fmaf(y, 0.75f, 0.1f);
    }
    ref[t] = x;
  }
  expect_eq("valu_f32", got.data(), ref.data(), n * 4);
}
static void run_valu_f64(const std::string& size) {
  const uint32_t blocks = 64, threads = 256, iters = scale(size, 200, 4000), n = blocks * threads;
  double* d;
  HC(hipMalloc(&d, n * 8));
  Phase p;
  LAUNCH(k_valu_f64, dim3(blocks), dim3(threads), 0, 0, d, iters);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<double> got = dev_read(d, n), ref(n);
  for (uint32_t t = 0; t < n; ++t) {
    double x = 1.0 + (t & 15) * 0.01, y = 0.5 + (t & 7) * 0.02;
    for (uint32_t i = 0; i < iters; ++i) {
      x = fma(x, 0.999, y);
      y = fma(y, 0.5, x * 0.001);
      x = fma(x, 0.5, 0.125);
    }
    ref[t] = x + y;
  }
  expect_eq("valu_f64", got.data(), ref.data(), n * 8);
}
static void run_mfma(const std::string& size) {
  if (std::string(g_arch).find("gfx9") == std::string::npos) {
    std::printf("SKIP no matrix instructions (MFMA) for %s\n", g_arch);
    std::exit(77);
  }
  const uint32_t waves = 64, n = scale(size, 400, 8000);
  std::vector<_Float16> A(256), B(256);
  for (int i = 0; i < 256; ++i) {
    A[i] = (_Float16)((i * 5) % 7 - 3);
    B[i] = (_Float16)((i * 3) % 5 - 2);
  }
  _Float16 *dA = dev_copy(A), *dB = dev_copy(B);
  float* dD;
  HC(hipMalloc(&dD, waves * 256 * 4));
  Phase p;
  LAUNCH(k_mfma, dim3(waves), dim3(64), 0, 0, dA, dB, dD, n);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<float> got = dev_read(dD, waves * 256), ref(waves * 256);
  for (uint32_t w = 0; w < waves; ++w)
    for (int i = 0; i < 16; ++i)
      for (int j = 0; j < 16; ++j) {
        float s = 0;
        for (int k = 0; k < 16; ++k) s += (float)A[i * 16 + k] * (float)B[k * 16 + j];
        ref[w * 256 + i * 16 + j] = s * (float)n;
      }
  expect_eq("mfma", got.data(), ref.data(), waves * 256 * 4);
}
// ---- the other matrix forms: host side

// D[i][j] = n * sum_k A[i][k] * B[k][j], for A 16 x K and B K x 16 of small integers (exact in every format used).
template <typename TA, typename TB>
static std::vector<double> matrix_ref(const std::vector<TA>& A, const std::vector<TB>& B, int K, uint32_t n) {
  std::vector<double> ref(256);
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) {
      double s = 0;
      for (int k = 0; k < K; ++k) s += static_cast<double>(A[i * K + k]) * static_cast<double>(B[k * 16 + j]);
      ref[i * 16 + j] = s * n;
    }
  return ref;
}
static int small_a(int i, int k) { return (i * 5 + k * 3) % 7 - 3; }
static int small_b(int k, int j) { return (k * 3 + j * 5) % 5 - 2; }
// An integer in -3..3 as an 8-bit float (E4M3: bias 8 on gfx942's FNUZ format, 7 on gfx950's OCP one).
static uint8_t fp8_of(int v, bool ocp) {
  const int bias = ocp ? 7 : 8, m = v < 0 ? -v : v;
  uint8_t code = 0;
  if (m == 1) code = static_cast<uint8_t>(bias << 3);
  else if (m == 2) code = static_cast<uint8_t>((bias + 1) << 3);
  else if (m == 3) code = static_cast<uint8_t>(((bias + 1) << 3) | 4);
  return v < 0 ? static_cast<uint8_t>(code | 0x80) : code;
}
static uint16_t bf16_of(int v) {
  const float f = static_cast<float>(v);
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return static_cast<uint16_t>(b >> 16);
}
static bool has_arch(const char* what) { return std::string(g_arch).find(what) != std::string::npos; }
static void need_mfma_more() {
  if (!has_arch("gfx942") && !has_arch("gfx950")) {
    std::printf("SKIP no such matrix instructions for %s\n", g_arch);
    std::exit(77);
  }
}
static void need_wmma() {
  if (!has_arch("gfx11") && !has_arch("gfx12")) {
    std::printf("SKIP no WMMA for %s\n", g_arch);
    std::exit(77);
  }
}
static void run_mfma_i8(const std::string& size) {
  need_mfma_more();
  const uint32_t waves = 64, n = scale(size, 200, 4000);
  std::vector<int8_t> A(16 * 32), B(32 * 16);
  for (int i = 0; i < 16; ++i)
    for (int k = 0; k < 32; ++k) A[i * 32 + k] = static_cast<int8_t>(small_a(i, k)), B[k * 16 + i] = static_cast<int8_t>(small_b(k, i));
  int8_t *dA = dev_copy(A), *dB = dev_copy(B);
  int* dD;
  HC(hipMalloc(&dD, waves * 256 * 4));
  Phase p;
  LAUNCH(k_mfma_i8, dim3(waves), dim3(64), 0, 0, dA, dB, dD, n);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  const std::vector<int> got = dev_read(dD, waves * 256);
  const std::vector<double> ref = matrix_ref(A, B, 32, n);
  for (uint32_t w = 0; w < waves; ++w)
    for (int e = 0; e < 256; ++e) expect_true("mfma_i8", got[w * 256 + e] == static_cast<int>(ref[e]));
}
static void run_mfma_fp8(const std::string& size) {
  need_mfma_more();
  const bool ocp = has_arch("gfx950");
  const uint32_t waves = 64, n = scale(size, 200, 4000);
  std::vector<uint8_t> A(16 * 32), B(32 * 16);
  std::vector<int8_t> Av(16 * 32), Bv(32 * 16);
  for (int i = 0; i < 16; ++i)
    for (int k = 0; k < 32; ++k) {
      Av[i * 32 + k] = static_cast<int8_t>(small_a(i, k)), Bv[k * 16 + i] = static_cast<int8_t>(small_b(k, i));
      A[i * 32 + k] = fp8_of(Av[i * 32 + k], ocp), B[k * 16 + i] = fp8_of(Bv[k * 16 + i], ocp);
    }
  uint8_t *dA = dev_copy(A), *dB = dev_copy(B);
  float* dD;
  HC(hipMalloc(&dD, waves * 256 * 4));
  Phase p;
  LAUNCH(k_mfma_fp8, dim3(waves), dim3(64), 0, 0, dA, dB, dD, n);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  const std::vector<float> got = dev_read(dD, waves * 256);
  const std::vector<double> ref = matrix_ref(Av, Bv, 32, n);
  for (uint32_t w = 0; w < waves; ++w)
    for (int e = 0; e < 256; ++e) expect_true("mfma_fp8", got[w * 256 + e] == static_cast<float>(ref[e]));
}
static void run_mfma_bf16(const std::string& size) {
  need_mfma_more();
  const uint32_t waves = 64, n = scale(size, 400, 8000);
  std::vector<uint16_t> A(16 * 16), B(16 * 16);
  std::vector<int8_t> Av(16 * 16), Bv(16 * 16);
  for (int i = 0; i < 16; ++i)
    for (int k = 0; k < 16; ++k) {
      Av[i * 16 + k] = static_cast<int8_t>(small_a(i, k)), Bv[k * 16 + i] = static_cast<int8_t>(small_b(k, i));
      A[i * 16 + k] = bf16_of(Av[i * 16 + k]), B[k * 16 + i] = bf16_of(Bv[k * 16 + i]);
    }
  uint16_t *dA = dev_copy(A), *dB = dev_copy(B);
  float* dD;
  HC(hipMalloc(&dD, waves * 256 * 4));
  Phase p;
  LAUNCH(k_mfma_bf16, dim3(waves), dim3(64), 0, 0, dA, dB, dD, n);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  const std::vector<float> got = dev_read(dD, waves * 256);
  const std::vector<double> ref = matrix_ref(Av, Bv, 16, n);
  for (uint32_t w = 0; w < waves; ++w)
    for (int e = 0; e < 256; ++e) expect_true("mfma_bf16", got[w * 256 + e] == static_cast<float>(ref[e]));
}
static void run_mfma_f64(const std::string& size) {
  need_mfma_more();
  const uint32_t waves = 64, n = scale(size, 400, 8000);
  std::vector<double> A(16 * 4), B(4 * 16);
  for (int i = 0; i < 16; ++i)
    for (int k = 0; k < 4; ++k) A[i * 4 + k] = small_a(i, k) * 0.25, B[k * 16 + i] = small_b(k, i) * 0.5;
  double *dA = dev_copy(A), *dB = dev_copy(B);
  double* dD;
  HC(hipMalloc(&dD, waves * 256 * 8));
  Phase p;
  LAUNCH(k_mfma_f64, dim3(waves), dim3(64), 0, 0, dA, dB, dD, n);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  const std::vector<double> got = dev_read(dD, waves * 256);
  const std::vector<double> ref = matrix_ref(A, B, 4, n);
  for (uint32_t w = 0; w < waves; ++w)
    for (int e = 0; e < 256; ++e) expect_true("mfma_f64", got[w * 256 + e] == ref[e]);
}
static void run_smfmac(const std::string& size) {
  need_mfma_more();
  const uint32_t waves = 64, n = scale(size, 200, 4000);
  std::vector<_Float16> A(16 * 32), B(32 * 16);
  std::vector<int8_t> Ad(16 * 32, 0), Bv(32 * 16);
  for (int i = 0; i < 16; ++i)
    for (int k = 0; k < 32; ++k) {
      A[i * 32 + k] = static_cast<_Float16>(small_a(i, k));
      if (k % 4 < 2) Ad[i * 32 + k] = static_cast<int8_t>(small_a(i, k));   // (the dense A: k offsets 0, 1 of each four)
      B[k * 16 + i] = static_cast<_Float16>(small_b(k, i)), Bv[k * 16 + i] = static_cast<int8_t>(small_b(k, i));
    }
  _Float16 *dA = dev_copy(A), *dB = dev_copy(B);
  float* dD;
  HC(hipMalloc(&dD, waves * 256 * 4));
  Phase p;
  LAUNCH(k_smfmac, dim3(waves), dim3(64), 0, 0, dA, dB, dD, n);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  const std::vector<float> got = dev_read(dD, waves * 256);
  const std::vector<double> ref = matrix_ref(Ad, Bv, 32, n);
  for (uint32_t w = 0; w < waves; ++w)
    for (int e = 0; e < 256; ++e) expect_true("smfmac", got[w * 256 + e] == static_cast<float>(ref[e]));
}
static void run_wmma(const std::string& size) {
  need_wmma();
  const uint32_t waves = 64, n = scale(size, 200, 4000);
  std::vector<_Float16> A(16 * 16), B(16 * 16);
  std::vector<int8_t> Av(16 * 16), Bv(16 * 16);
  for (int i = 0; i < 16; ++i)
    for (int k = 0; k < 16; ++k) {
      Av[i * 16 + k] = static_cast<int8_t>(small_a(i, k)), Bv[k * 16 + i] = static_cast<int8_t>(small_b(k, i));
      A[i * 16 + k] = static_cast<_Float16>(Av[i * 16 + k]), B[k * 16 + i] = static_cast<_Float16>(Bv[k * 16 + i]);
    }
  _Float16 *dA = dev_copy(A), *dB = dev_copy(B);
  float* dD;
  HC(hipMalloc(&dD, waves * 256 * 4));
  Phase p;
  LAUNCH(k_wmma_f16, dim3(waves), dim3(32), 0, 0, dA, dB, dD, n);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  const std::vector<float> got = dev_read(dD, waves * 256);
  const std::vector<double> ref = matrix_ref(Av, Bv, 16, n);
  for (uint32_t w = 0; w < waves; ++w)
    for (int e = 0; e < 256; ++e) expect_true("wmma", got[w * 256 + e] == static_cast<float>(ref[e]));
}
static void run_wmma_iu8(const std::string& size) {
  need_wmma();
  const uint32_t waves = 64, n = scale(size, 200, 4000);
  std::vector<uint8_t> A(16 * 16), B(16 * 16);   // (unsigned: 0..5)
  for (int i = 0; i < 16; ++i)
    for (int k = 0; k < 16; ++k) A[i * 16 + k] = static_cast<uint8_t>((i * 5 + k * 3) % 6), B[k * 16 + i] = static_cast<uint8_t>((k * 3 + i * 5) % 4);
  uint8_t *dA = dev_copy(A), *dB = dev_copy(B);
  int* dD;
  HC(hipMalloc(&dD, waves * 256 * 4));
  Phase p;
  LAUNCH(k_wmma_iu8, dim3(waves), dim3(32), 0, 0, dA, dB, dD, n);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  const std::vector<int> got = dev_read(dD, waves * 256);
  const std::vector<double> ref = matrix_ref(A, B, 16, n);
  for (uint32_t w = 0; w < waves; ++w)
    for (int e = 0; e < 256; ++e) expect_true("wmma_iu8", got[w * 256 + e] == static_cast<int>(ref[e]));
}
static void run_mem(const std::string& size) {
  const uint32_t n4 = 1u << 16, n = n4 * 4, reps = scale(size, 4, 40);
  std::vector<uint32_t> in(n), idx(n);
  for (uint32_t i = 0; i < n; ++i) {
    in[i] = i * 2654435761u;
    idx[i] = (i * 40503u + 17u) % n;
  }
  uint32_t *din = dev_copy(in), *didx = dev_copy(idx), *o1, *o2, *o3;
  HC(hipMalloc(&o1, n * 4));
  HC(hipMalloc(&o2, n * 4));
  HC(hipMalloc(&o3, n * 4));
  Phase p;
  LAUNCH(k_mem_copy, dim3(64), dim3(256), 0, 0, (const uint4*)din, (uint4*)o1, n4, reps);
  LAUNCH(k_mem_scalar, dim3(64), dim3(256), 0, 0, (const uint32_t*)din, o2, n, reps);
  LAUNCH(k_mem_gather, dim3(64), dim3(256), 0, 0, (const uint32_t*)din, (const uint32_t*)didx, o3, n, reps);
  HC(hipDeviceSynchronize());
  p.finish(3, "launch");
  std::vector<uint32_t> g1 = dev_read(o1, n), g2 = dev_read(o2, n), g3 = dev_read(o3, n);
  std::vector<uint32_t> r1(n), r2(n), r3(n);
  const uint32_t last = reps - 1;
  for (uint32_t i = 0; i < n4; ++i) {
    uint32_t v0 = in[4 * i + 0] + last, v1 = in[4 * i + 1] ^ v0, v2 = in[4 * i + 2] + v1, v3 = in[4 * i + 3] ^ v2;
    r1[4 * i] = v0, r1[4 * i + 1] = v1, r1[4 * i + 2] = v2, r1[4 * i + 3] = v3;
  }
  for (uint32_t i = 0; i < n; ++i) {
    r2[i] = in[i] * 3u + last;
    r3[i] = in[idx[i]] + last;
  }
  expect_eq("mem_copy", g1.data(), r1.data(), n * 4);
  expect_eq("mem_scalar", g2.data(), r2.data(), n * 4);
  expect_eq("mem_gather", g3.data(), r3.data(), n * 4);
}
static void run_lds(const std::string& size) {
  const uint32_t blocks = 64, reps = scale(size, 10, 200);
  std::vector<uint32_t> in(blocks * 1024);
  for (size_t i = 0; i < in.size(); ++i) in[i] = (uint32_t)(i * 2654435761u) >> 20;
  uint32_t *din = dev_copy(in), *o1, *o2;
  HC(hipMalloc(&o1, blocks * 4));
  HC(hipMalloc(&o2, blocks * 1024 * 4));
  Phase p;
  LAUNCH(k_lds_reduce, dim3(blocks), dim3(256), 0, 0, (const uint32_t*)din, o1, reps);
  LAUNCH(k_lds_transpose, dim3(blocks), dim3(256), 0, 0, (const uint32_t*)din, o2, reps);
  HC(hipDeviceSynchronize());
  p.finish(2, "launch");
  std::vector<uint32_t> g1 = dev_read(o1, blocks), g2 = dev_read(o2, blocks * 1024), r1(blocks), r2(blocks * 1024);
  for (uint32_t b = 0; b < blocks; ++b) {
    uint32_t acc = 0;
    for (uint32_t r = 0; r < reps; ++r) {
      uint32_t s = 0;
      for (uint32_t i = 0; i < 256; ++i) s += in[b * 256 + i] + r;  // k_lds_reduce reads in[blockIdx.x * 256 + tid]
      acc += s;
    }
    r1[b] = acc;
    for (uint32_t y = 0; y < 32; ++y)
      for (uint32_t x = 0; x < 32; ++x) r2[b * 1024 + y * 32 + x] = in[b * 1024 + x * 32 + y] + (reps - 1);
  }
  expect_eq("lds_reduce", g1.data(), r1.data(), blocks * 4);
  expect_eq("lds_transpose", g2.data(), r2.data(), blocks * 1024 * 4);
}
static void run_atomics(const std::string& size) {
  const uint32_t blocks = 16, threads = 256, iters = scale(size, 4, 200), n = blocks * threads;
  uint32_t* d;
  HC(hipMalloc(&d, (3 + 256) * 4));
  HC(hipMemset(d, 0, (3 + 256) * 4));
  Phase p;
  LAUNCH(k_atomics, dim3(blocks), dim3(threads), 0, 0, d, d + 3, d + 1, d + 2, iters);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> got = dev_read(d, 3 + 256), ref(3 + 256, 0);
  ref[0] = n * iters;
  uint32_t mx = 0;
  for (uint32_t t = 0; t < n; ++t)
    for (uint32_t i = 0; i < iters; ++i) {
      ref[3 + ((t * 7 + i) & 255)] += 2;
      mx = std::max(mx, t ^ i);
    }
  ref[1] = mx;
  ref[2] = 3 * n * iters;
  expect_eq("atomics", got.data(), ref.data(), got.size() * 4);
}
static void run_divergent(const std::string& size) {
  const uint32_t n = scale(size, 4096, 65536), seedoff = 1000;
  uint32_t* d;
  HC(hipMalloc(&d, n * 4));
  Phase p;
  LAUNCH(k_divergent, dim3(n / 256), dim3(256), 0, 0, d, n, seedoff);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> got = dev_read(d, n), ref(n);
  for (uint32_t t = 0; t < n; ++t) ref[t] = collatz_mix(t, seedoff);
  expect_eq("divergent", got.data(), ref.data(), n * 4);
}
static void run_wave(const std::string& size) {
  const uint32_t blocks = 32, threads = 256, iters = scale(size, 100, 2000), n = blocks * threads;
  uint32_t* d;
  HC(hipMalloc(&d, n * 4));
  Phase p;
  LAUNCH(k_wave, dim3(blocks), dim3(threads), 0, 0, d, iters);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> got = dev_read(d, n), ref(n);
  const uint32_t ws = g_wave;
  for (uint32_t base = 0; base < n; base += ws) {
    std::vector<uint32_t> v(ws), nv(ws), acc(ws, 0);
    for (uint32_t l = 0; l < ws; ++l) v[l] = (base + l) * 2654435761u + 7;
    for (uint32_t i = 0; i < iters; ++i) {
      uint64_t m = 0;
      bool any = false, all = true;
      for (uint32_t l = 0; l < ws; ++l) {
        m |= (uint64_t)((v[l] + i) & 1) << l;
        any = any || (v[l] & 4);
        all = all && (v[l] | 1);
      }
      for (uint32_t l = 0; l < ws; ++l) {
        uint32_t a = v[l ^ 1], b = v[l ^ 16], c = v[(l + i) % ws];
        acc[l] += (uint32_t)__builtin_popcountll(m) + (uint32_t)any + (uint32_t)all;
        nv[l] = rotl32(v[l] ^ a, 3) + b - c + i;
      }
      v = nv;
    }
    for (uint32_t l = 0; l < ws; ++l) ref[base + l] = v[l] + acc[l];
  }
  expect_eq("wave", got.data(), ref.data(), n * 4);
}
static void run_launch(const std::string& size) {
  const uint32_t launches = scale(size, 2000, 100000);
  uint32_t* d;
  HC(hipMalloc(&d, 64 * 4));
  hipStream_t s[2];
  HC(hipStreamCreate(&s[0]));
  HC(hipStreamCreate(&s[1]));
  Phase p;
  for (uint32_t i = 0; i < launches; ++i) LAUNCH(k_tiny, dim3(1), dim3(64), 0, s[i & 1], d, i);
  HC(hipDeviceSynchronize());
  p.finish(launches, "launch");
  std::vector<uint32_t> got = dev_read(d, 1);
  // the last launch on each stream writes p[0]; the order across the two streams is not defined
  expect_true("launch", got[0] == launches - 1 || got[0] == launches - 2);
}
static void run_api(const std::string& size) {
  const uint32_t rounds = scale(size, 1000, 100000);
  std::vector<uint8_t> h(4096, 7), back(4096);
  hipEvent_t e0, e1;
  HC(hipEventCreate(&e0));
  HC(hipEventCreate(&e1));
  hipStream_t s;
  HC(hipStreamCreate(&s));
  uint32_t good = 0;
  Phase p;
  for (uint32_t r = 0; r < rounds; ++r) {
    void* d;
    HC(hipMalloc(&d, 4096));
    HC(hipMemcpy(d, h.data(), 4096, hipMemcpyHostToDevice));
    HC(hipMemset(d, (int)(r & 0xff), 2048));
    HC(hipEventRecord(e0, s));
    HC(hipMemcpyAsync(back.data(), d, 4096, hipMemcpyDeviceToHost, s));
    HC(hipEventRecord(e1, s));
    HC(hipStreamSynchronize(s));
    HC(hipEventSynchronize(e1));
    float ms;
    HC(hipEventElapsedTime(&ms, e0, e1));
    size_t fr, tot;
    HC(hipMemGetInfo(&fr, &tot));
    HC(hipFree(d));
    good += back[0] == (r & 0xff) && back[3000] == 7;
  }
  p.finish(rounds * 13ull, "call");
  expect_true("api", good == rounds);
}
static void run_startup(const std::string&) {
  uint32_t* d;
  HC(hipMalloc(&d, 64 * 4));
  Phase p;
  LAUNCH(k_tiny, dim3(1), dim3(64), 0, 0, d, 5);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> got = dev_read(d, 1);
  expect_true("startup", got[0] == 5);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <workload> <quick|full>\n", argv[0]);
    return 2;
  }
  const std::string w = argv[1], size = argv[2];
  int dev = 0;
  HC(hipGetDevice(&dev));
  hipDeviceProp_t prop;
  HC(hipGetDeviceProperties(&prop, dev));
  g_wave = prop.warpSize;
  std::snprintf(g_arch, sizeof g_arch, "%s", prop.gcnArchName);
  if (w == "valu_int") run_valu_int(size);
  else if (w == "valu_f32") run_valu_f32(size);
  else if (w == "valu_f64") run_valu_f64(size);
  else if (w == "mfma") run_mfma(size);
  else if (w == "mfma_i8") run_mfma_i8(size);
  else if (w == "mfma_fp8") run_mfma_fp8(size);
  else if (w == "mfma_bf16") run_mfma_bf16(size);
  else if (w == "mfma_f64") run_mfma_f64(size);
  else if (w == "smfmac") run_smfmac(size);
  else if (w == "wmma") run_wmma(size);
  else if (w == "wmma_iu8") run_wmma_iu8(size);
  else if (w == "mem") run_mem(size);
  else if (w == "lds") run_lds(size);
  else if (w == "atomics") run_atomics(size);
  else if (w == "divergent") run_divergent(size);
  else if (w == "wave") run_wave(size);
  else if (w == "launch") run_launch(size);
  else if (w == "api") run_api(size);
  else if (w == "startup") run_startup(size);
  else {
    std::fprintf(stderr, "unknown workload %s\n", w.c_str());
    return 2;
  }
  std::printf(g_ok ? "PASS\n" : "FAIL\n");
  return g_ok ? 0 : 1;
}
