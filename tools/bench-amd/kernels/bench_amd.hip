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
#include <array>
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


// ---------------------------------------------------------------- memory, the other instruction forms
// (sub-dword global accesses, scalar loads, buffer loads with offsets, wide LDS accesses and LDS atomics)

__global__ void k_narrow(const uint8_t* in8, const int8_t* ins8, const uint16_t* in16, uint8_t* o8, uint16_t* o16, int32_t* osum,
                         uint32_t n, uint32_t reps) {
  uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t r = 0; r < reps; ++r)
    for (uint32_t i = t; i < n; i += gridDim.x * blockDim.x) {
      o8[i] = (uint8_t)(in8[i] + r);
      o16[i] = (uint16_t)(in16[i] * 3u + r);
      osum[i] = (int32_t)ins8[i] + (int32_t)r;
    }
}
// Every lane of a wave takes the same table entries: scalar loads (s_load_dword, s_load_dwordx4).
__global__ void k_smem(const uint32_t* __restrict__ tab, uint32_t* __restrict__ out, uint32_t n, uint32_t reps) {
  uint32_t acc = threadIdx.x;
  for (uint32_t r = 0; r < reps; ++r)
    for (uint32_t i = 0; i < n; i += 4) {
      const uint4 v = *reinterpret_cast<const uint4*>(tab + i);
      acc = acc * 31u + v.x;
      acc ^= v.y + r;
      acc += v.z * 7u;
      acc ^= v.w;
      acc += tab[(i * 5u + r) % n];
    }
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
// Raw buffer loads and stores with a per-lane offset, some of them past the end of the resource (which read zero).
__global__ void k_buf(const uint32_t* in, uint32_t* out, uint32_t n, uint32_t records_words, uint32_t reps, uint32_t flags) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  __amdgpu_buffer_rsrc_t ri = __builtin_amdgcn_make_buffer_rsrc(const_cast<uint32_t*>(in), 0, records_words * 4, flags);
  __amdgpu_buffer_rsrc_t ro = __builtin_amdgcn_make_buffer_rsrc(out, 0, n * 4, flags);
  for (uint32_t r = 0; r < reps; ++r)
    for (uint32_t i = t; i < n; i += gridDim.x * blockDim.x) {
      const uint32_t a = __builtin_amdgcn_raw_buffer_load_b32(ri, i * 4, 0, 0);
      const uint32_t b = __builtin_amdgcn_raw_buffer_load_b32(ri, (i + 3) * 4, 0, 0);
      __builtin_amdgcn_raw_buffer_store_b32(a * 3u + b + r, ro, i * 4, 0, 0);
    }
}
__global__ void k_ldsx(const uint32_t* in, uint32_t* out, uint32_t* hist, uint32_t reps) {
  __shared__ uint4 v[256];
  __shared__ uint32_t bins[64];
  uint32_t acc = 0;
  if (threadIdx.x < 64) bins[threadIdx.x] = 0;
  for (uint32_t r = 0; r < reps; ++r) {
    const uint32_t x = in[blockIdx.x * 256 + threadIdx.x] + r;
    v[threadIdx.x] = make_uint4(x, x ^ 1u, x + 2u, x ^ 3u);
    __syncthreads();
    const uint4 a = v[threadIdx.x ^ 1], b = v[(threadIdx.x + 17) & 255];
    acc += a.x + a.y * 3u + a.z + b.w + b.x;
    atomicAdd(&bins[(x >> 3) & 63], 1u);
    __syncthreads();
  }
  out[blockIdx.x * 256 + threadIdx.x] = acc;
  if (threadIdx.x < 64) hist[blockIdx.x * 64 + threadIdx.x] = bins[threadIdx.x];
}

// Generic pointers: flat loads and stores that reach global memory (the aperture checks and then the access).
__global__ void k_flat(const uint32_t* in, uint32_t* out, uint32_t n, uint32_t reps, uint32_t sel) {
  __shared__ uint32_t s[256];
  s[threadIdx.x] = threadIdx.x * 7u;
  __syncthreads();
  const uint32_t* src = sel ? (const uint32_t*)s : in;   // (sel is 0: the global array; the compiler cannot know)
  uint32_t* dst = sel ? (uint32_t*)s : out;
  uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t r = 0; r < reps; ++r)
    for (uint32_t i = t; i < n; i += gridDim.x * blockDim.x) dst[i] = src[(sel ? threadIdx.x : i)] * 3u + r;
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
static void run_narrow(const std::string& size) {
  const uint32_t n = 1u << 16, reps = scale(size, 4, 400);
  std::vector<uint8_t> in8(n);
  std::vector<int8_t> ins8(n);
  std::vector<uint16_t> in16(n);
  for (uint32_t i = 0; i < n; ++i) in8[i] = (uint8_t)(i * 2654435761u >> 24), ins8[i] = (int8_t)(i * 40503u >> 3), in16[i] = (uint16_t)(i * 2654435761u >> 11);
  uint8_t *d8 = dev_copy(in8), *o8;
  int8_t* ds8 = dev_copy(ins8);
  uint16_t *d16 = dev_copy(in16), *o16;
  int32_t* os;
  HC(hipMalloc(&o8, n));
  HC(hipMalloc(&o16, n * 2));
  HC(hipMalloc(&os, n * 4));
  Phase p;
  LAUNCH(k_narrow, dim3(64), dim3(256), 0, 0, (const uint8_t*)d8, (const int8_t*)ds8, (const uint16_t*)d16, o8, o16, os, n, reps);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint8_t> g8 = dev_read(o8, n), r8(n);
  std::vector<uint16_t> g16 = dev_read(o16, n), r16(n);
  std::vector<int32_t> gs = dev_read(os, n), rs(n);
  const uint32_t last = reps - 1;
  for (uint32_t i = 0; i < n; ++i) {
    r8[i] = (uint8_t)(in8[i] + last);
    r16[i] = (uint16_t)(in16[i] * 3u + last);
    rs[i] = (int32_t)ins8[i] + (int32_t)last;
  }
  expect_eq("narrow_u8", g8.data(), r8.data(), n);
  expect_eq("narrow_u16", g16.data(), r16.data(), n * 2);
  expect_eq("narrow_s8", gs.data(), rs.data(), n * 4);
}
static void run_smem(const std::string& size) {
  const uint32_t n = 1024, blocks = 16, threads = 256, reps = scale(size, 4, 60);
  std::vector<uint32_t> tab(n);
  for (uint32_t i = 0; i < n; ++i) tab[i] = i * 2654435761u + 7;
  uint32_t *dt = dev_copy(tab), *o;
  HC(hipMalloc(&o, blocks * threads * 4));
  Phase p;
  LAUNCH(k_smem, dim3(blocks), dim3(threads), 0, 0, (const uint32_t*)dt, o, n, reps);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> got = dev_read(o, blocks * threads), ref(blocks * threads);
  for (uint32_t b = 0; b < blocks; ++b)
    for (uint32_t t = 0; t < threads; ++t) {
      uint32_t acc = t;
      for (uint32_t r = 0; r < reps; ++r)
        for (uint32_t i = 0; i < n; i += 4) {
          acc = acc * 31u + tab[i];
          acc ^= tab[i + 1] + r;
          acc += tab[i + 2] * 7u;
          acc ^= tab[i + 3];
          acc += tab[(i * 5u + r) % n];
        }
      ref[b * threads + t] = acc;
    }
  expect_eq("smem", got.data(), ref.data(), blocks * threads * 4);
}
static void run_buf(const std::string& size) {
  const uint32_t n = 1u << 16, records = n - 2, reps = scale(size, 4, 400);   // the last two words of the input are outside the resource
  std::vector<uint32_t> in(n);
  for (uint32_t i = 0; i < n; ++i) in[i] = i * 2654435761u + 5;
  uint32_t *din = dev_copy(in), *o;
  HC(hipMalloc(&o, n * 4));
  // The resource's last descriptor word: gfx9 (CDNA) takes a data format; RDNA2 a format and a raw out-of-range check.
  const uint32_t flags = std::string(g_arch).find("gfx9") != std::string::npos ? 0x00020000u : 0x31014000u;
  Phase p;
  LAUNCH(k_buf, dim3(64), dim3(256), 0, 0, (const uint32_t*)din, o, n, records, reps, flags);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> got = dev_read(o, n), ref(n);
  const uint32_t last = reps - 1;
  const auto word = [&](uint32_t i) { return i < records ? in[i] : 0u; };
  for (uint32_t i = 0; i < n; ++i) ref[i] = word(i) * 3u + word(i + 3) + last;
  expect_eq("buf", got.data(), ref.data(), n * 4);
}
static void run_ldsx(const std::string& size) {
  const uint32_t blocks = 64, reps = scale(size, 10, 2000);
  std::vector<uint32_t> in(blocks * 256);
  for (size_t i = 0; i < in.size(); ++i) in[i] = (uint32_t)(i * 2654435761u) >> 20;
  uint32_t *din = dev_copy(in), *o, *h;
  HC(hipMalloc(&o, blocks * 256 * 4));
  HC(hipMalloc(&h, blocks * 64 * 4));
  Phase p;
  LAUNCH(k_ldsx, dim3(blocks), dim3(256), 0, 0, (const uint32_t*)din, o, h, reps);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> go = dev_read(o, blocks * 256), gh = dev_read(h, blocks * 64), ro(blocks * 256), rh(blocks * 64, 0);
  for (uint32_t b = 0; b < blocks; ++b) {
    for (uint32_t t = 0; t < 256; ++t) {
      uint32_t acc = 0;
      for (uint32_t r = 0; r < reps; ++r) {
        const auto vec = [&](uint32_t u) {
          const uint32_t x = in[b * 256 + u] + r;
          return std::array<uint32_t, 4>{x, x ^ 1u, x + 2u, x ^ 3u};
        };
        const auto a = vec(t ^ 1), bb = vec((t + 17) & 255);
        acc += a[0] + a[1] * 3u + a[2] + bb[3] + bb[0];
      }
      ro[b * 256 + t] = acc;
    }
    for (uint32_t r = 0; r < reps; ++r)
      for (uint32_t t = 0; t < 256; ++t) ++rh[b * 64 + (((in[b * 256 + t] + r) >> 3) & 63)];
  }
  expect_eq("ldsx_out", go.data(), ro.data(), blocks * 256 * 4);
  expect_eq("ldsx_hist", gh.data(), rh.data(), blocks * 64 * 4);
}
static void run_flat(const std::string& size) {
  const uint32_t n = 1u << 16, reps = scale(size, 4, 400);
  std::vector<uint32_t> in(n);
  for (uint32_t i = 0; i < n; ++i) in[i] = i * 2654435761u + 3;
  uint32_t *din = dev_copy(in), *o;
  HC(hipMalloc(&o, n * 4));
  Phase p;
  LAUNCH(k_flat, dim3(64), dim3(256), 0, 0, (const uint32_t*)din, o, n, reps, 0u);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> got = dev_read(o, n), ref(n);
  for (uint32_t i = 0; i < n; ++i) ref[i] = in[i] * 3u + (reps - 1);
  expect_eq("flat", got.data(), ref.data(), n * 4);
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

// ---- launch, stream, copy and graph rates (speedup4-amdmt) ----
// Many small work-groups: what the work-group hand-off between the calling thread and the workers costs.
__global__ void k_wide(uint32_t* out, uint32_t v) {
  uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  out[t] = (t * 2654435761u) ^ v;
}
static void run_launch_grid(uint32_t groups, uint32_t launches, const char* what) {
  const uint32_t block = 64;
  uint32_t* d;
  HC(hipMalloc(&d, groups * block * 4));
  hipStream_t s;
  HC(hipStreamCreate(&s));
  Phase p;
  for (uint32_t i = 0; i < launches; ++i) LAUNCH(k_wide, dim3(groups), dim3(block), 0, s, d, i);
  HC(hipDeviceSynchronize());
  p.finish(launches, "launch");
  std::vector<uint32_t> got = dev_read(d, groups * block), ref(groups * block);
  for (uint32_t t = 0; t < groups * block; ++t) ref[t] = (t * 2654435761u) ^ (launches - 1);
  expect_eq(what, got.data(), ref.data(), got.size() * 4);
}
static void run_launch_wide(const std::string& size) { run_launch_grid(1024, scale(size, 100, 1000), "launch_wide"); }
// The same with 128 work-groups: about 40 microseconds of work to each of 8 host threads, where starting or waking
// the threads is a good part of the launch.
static void run_launch_mid(const std::string& size) { run_launch_grid(128, scale(size, 400, 6000), "launch_mid"); }
// A chain of launches, each reading what the one before wrote, synchronised every few launches.
__global__ void k_chain(uint32_t* p, uint32_t v) {
  if (blockIdx.x == 0 && threadIdx.x == 0) p[0] = p[0] * 31u + v;
}
static void run_launch_sync(const std::string& size) {
  const uint32_t rounds = scale(size, 200, 20000);
  uint32_t* d;
  HC(hipMalloc(&d, 64 * 4));
  HC(hipMemset(d, 0, 4));
  hipStream_t s;
  HC(hipStreamCreate(&s));
  uint32_t ref = 0;
  Phase p;
  for (uint32_t r = 0; r < rounds; ++r) {
    for (uint32_t j = 0; j < 4; ++j) {
      LAUNCH(k_chain, dim3(1), dim3(64), 0, s, d, r + j);
      ref = ref * 31u + (r + j);
    }
    HC(hipStreamSynchronize(s));
  }
  p.finish(rounds * 5ull, "call");
  std::vector<uint32_t> got = dev_read(d, 1);
  expect_true("launch_sync", got[0] == ref);
}
static void run_memcpy(const std::string& size) {
  const uint32_t rounds = scale(size, 200, 20000);
  const size_t small = 16 << 10, big = 64 << 10;
  std::vector<uint8_t> h(big), back(big);
  for (size_t i = 0; i < big; ++i) h[i] = static_cast<uint8_t>(i * 7 + 1);
  void *a, *b;
  HC(hipMalloc(&a, big));
  HC(hipMalloc(&b, big));
  hipStream_t s;
  HC(hipStreamCreate(&s));
  uint32_t good = 0;
  Phase p;
  for (uint32_t r = 0; r < rounds; ++r) {
    HC(hipMemcpy(a, h.data(), small, hipMemcpyHostToDevice));
    HC(hipMemcpy(b, a, small, hipMemcpyDeviceToDevice));
    HC(hipMemcpy(back.data(), b, small, hipMemcpyDeviceToHost));
    HC(hipMemcpy(a, h.data(), big, hipMemcpyHostToDevice));
    HC(hipMemcpyAsync(b, a, big, hipMemcpyDeviceToDevice, s));
    HC(hipMemsetAsync(b, (int)(r & 0xff), small, s));
    HC(hipMemcpyAsync(back.data() + small, b, big - small, hipMemcpyDeviceToHost, s));
    HC(hipStreamSynchronize(s));
    HC(hipMemset(a, 1, small));
    good += back[0] == h[0] && back[small - 1] == h[small - 1] && back[small] == (r & 0xff) && back[big - 1] == h[big - 1];
  }
  p.finish(rounds * 9ull, "call");
  expect_true("memcpy", good == rounds);
}
static void run_stream(const std::string& size) {
  const uint32_t rounds = scale(size, 100, 8000);
  uint32_t* d;
  HC(hipMalloc(&d, 64 * 4));
  HC(hipMemset(d, 0, 4));
  hipStream_t s0;
  HC(hipStreamCreate(&s0));
  hipEvent_t e;
  HC(hipEventCreateWithFlags(&e, hipEventDisableTiming));
  uint32_t ref = 0;
  Phase p;
  for (uint32_t r = 0; r < rounds; ++r) {
    hipStream_t s1;
    HC(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
    LAUNCH(k_chain, dim3(1), dim3(64), 0, s0, d, r);
    ref = ref * 31u + r;
    HC(hipEventRecord(e, s0));
    HC(hipStreamWaitEvent(s1, e, 0));
    LAUNCH(k_chain, dim3(1), dim3(64), 0, s1, d, r + 1);
    ref = ref * 31u + (r + 1);
    HC(hipStreamSynchronize(s1));
    (void)hipEventQuery(e);
    (void)hipStreamQuery(s0);
    HC(hipStreamSynchronize(s0));
    HC(hipStreamDestroy(s1));
  }
  p.finish(rounds * 11ull, "call");
  std::vector<uint32_t> got = dev_read(d, 1);
  expect_true("stream", got[0] == ref);
}
static void run_graph(const std::string& size) {
  const uint32_t launches = scale(size, 100, 6000);
  uint32_t* d;
  HC(hipMalloc(&d, 64 * 4));
  HC(hipMemset(d, 0, 4));
  hipStream_t s;
  HC(hipStreamCreate(&s));
  hipGraph_t g;
  hipGraphExec_t ge;
  HC(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
  for (uint32_t j = 0; j < 4; ++j) LAUNCH(k_chain, dim3(1), dim3(64), 0, s, d, j + 1);
  HC(hipStreamEndCapture(s, &g));
  HC(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
  uint32_t ref = 0;
  Phase p;
  for (uint32_t i = 0; i < launches; ++i) {
    HC(hipGraphLaunch(ge, s));
    for (uint32_t j = 0; j < 4; ++j) ref = ref * 31u + (j + 1);
    if ((i & 15) == 15) HC(hipStreamSynchronize(s));
  }
  HC(hipStreamSynchronize(s));
  p.finish(launches, "launch");
  std::vector<uint32_t> got = dev_read(d, 1);
  expect_true("graph", got[0] == ref);
  HC(hipGraphExecDestroy(ge));
  HC(hipGraphDestroy(g));
}
static void run_apicheap(const std::string& size) {
  const uint32_t rounds = scale(size, 20000, 1000000);
  int dev = -1, warp = 0, ok = 0;
  Phase p;
  for (uint32_t r = 0; r < rounds; ++r) {
    HC(hipGetDevice(&dev));
    HC(hipGetLastError());
    HC(hipPeekAtLastError());
    HC(hipDeviceGetAttribute(&warp, hipDeviceAttributeWarpSize, dev));
    ok += dev == 0 && warp == g_wave;
  }
  p.finish(rounds * 4ull, "call");
  expect_true("apicheap", ok == (int)rounds);
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
  else if (w == "narrow") run_narrow(size);
  else if (w == "smem") run_smem(size);
  else if (w == "buf") run_buf(size);
  else if (w == "ldsx") run_ldsx(size);
  else if (w == "flat") run_flat(size);
  else if (w == "atomics") run_atomics(size);
  else if (w == "divergent") run_divergent(size);
  else if (w == "wave") run_wave(size);
  else if (w == "launch") run_launch(size);
  else if (w == "api") run_api(size);
  else if (w == "startup") run_startup(size);
  else if (w == "apicheap") run_apicheap(size);
  else if (w == "launch_mid") run_launch_mid(size);
  else if (w == "launch_wide") run_launch_wide(size);
  else if (w == "launch_sync") run_launch_sync(size);
  else if (w == "memcpy") run_memcpy(size);
  else if (w == "stream") run_stream(size);
  else if (w == "graph") run_graph(size);
  else {
    std::fprintf(stderr, "unknown workload %s\n", w.c_str());
    return 2;
  }
  std::printf(g_ok ? "PASS\n" : "FAIL\n");
  return g_ok ? 0 : 1;
}
