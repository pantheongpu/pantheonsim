// Self-checking benchmark workloads for the simulator (tools/bench/bench.py).
//
// One binary, one workload per run:   bench_kernels <workload> <quick|full>
//
// Every workload sets up its data, runs the device phase (which is what gets
// timed), then checks the answer against a host reference and prints PASS only
// when every checked value matches bit for bit. A speed-up that breaks a result
// therefore fails the run instead of looking like a faster one.
//
// The line the harness reads:
//   BENCH workload=<n> size=<s> device_cpu_s=<x> device_wall_s=<y> units=<u> unit=<label>
// device_* cover the device phase only (launches, copies, API calls); setup and
// the host-side check are outside it. units is a workload-specific count
// (launches, API calls) for workloads whose cost is not instructions.
//
// Rules for adding a workload:
//  * deterministic: no clocks, no random seeds, nothing that depends on thread
//    scheduling (atomics are fine if the result is order independent);
//  * integer-valued floats wherever a sum order could change the bits;
//  * a host reference that is bit-exact (fmaf, __fmul_rn and friends are
//    correctly rounded on both sides; a plain a*b+c is not: nvcc contracts it);
//  * check a sample of threads when the host reference would cost more than a
//    few percent of the simulated run.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <time.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace nvcuda;

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                           \
      std::fprintf(stderr, "FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x,            \
                   cudaGetErrorName(e_));                                              \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

#define REQUIRE(cond, ...)                                                             \
  do {                                                                                 \
    if (!(cond)) {                                                                     \
      std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                        \
      std::fprintf(stderr, __VA_ARGS__);                                               \
      std::fprintf(stderr, "\n");                                                      \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

// ---- the timed phase --------------------------------------------------------
static double now_cpu() {
  timespec t;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
  return double(t.tv_sec) + double(t.tv_nsec) * 1e-9;
}
static double now_wall() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return double(t.tv_sec) + double(t.tv_nsec) * 1e-9;
}
static struct {
  double c0 = 0, w0 = 0, cpu = 0, wall = 0;
  unsigned long long units = 0;
  const char* unit = "";
} g_phase;
static void phase_begin() {
  g_phase.c0 = now_cpu();
  g_phase.w0 = now_wall();
}
static void phase_end() {
  g_phase.cpu += now_cpu() - g_phase.c0;
  g_phase.wall += now_wall() - g_phase.w0;
}

static bool g_full = false;
// A workload's size: quick is for the counter check and CI, full for timing.
template <class T>
static T pick(T quick, T full) {
  return g_full ? full : quick;
}
// A thread is checked on the host when this holds (about 1 in 13, spread over
// every lane and warp since 13 is coprime with 32).
static bool sampled(uint32_t i) { return i % 13u == 0; }

// ---- integer ALU ------------------------------------------------------------
__host__ __device__ static inline uint32_t rotl32(uint32_t v, int s) {
  return (v << s) | (v >> (32 - s));
}
__host__ __device__ static inline uint32_t popc32(uint32_t v) {
#ifdef __CUDA_ARCH__
  return __popc(v);
#else
  return uint32_t(__builtin_popcount(v));
#endif
}
// One definition for host and device: integer arithmetic is exact on both.
__host__ __device__ static inline uint32_t int_chain(uint32_t i, int iters) {
  uint32_t x = i * 2654435761u + 12345u, y = i ^ 0x9e3779b9u;
  for (int k = 0; k < iters; ++k) {
    x = x * 1664525u + 1013904223u;
    y ^= x >> 13;
    y += x << 7;
    y = rotl32(y, 5);
    x ^= y;
    x += popc32(y) * 0x01010101u;
    y = (y & 0x0f0f0f0fu) | (x & 0xf0f0f0f0u);
    if ((k & 7) == 7) y += (x | 1u) % 65521u;
  }
  return x ^ rotl32(y, 11);
}
__global__ void k_int_alu(uint32_t* out, int iters) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = int_chain(i, iters);
}
static int w_int_alu() {
  const int blocks = 64, threads = 128, n = blocks * threads;
  const int iters = pick(300, 3000);
  uint32_t* d;
  CK(cudaMalloc(&d, n * 4));
  phase_begin();
  k_int_alu<<<blocks, threads>>>(d, iters);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<uint32_t> h(n);
  CK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
  phase_end();
  for (int i = 0; i < n; ++i)
    if (sampled(i)) REQUIRE(h[i] == int_chain(i, iters), "thread %d", i);
  CK(cudaFree(d));
  return 0;
}

// Correctly rounded add and multiply that nvcc will not contract into an FMA.
// The intrinsics are device-only, so the host side uses the plain operators
// (the harness compiles the host code with -ffp-contract=off).
__host__ __device__ static inline float add_rn(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fadd_rn(a, b);
#else
  return a + b;
#endif
}
__host__ __device__ static inline float mul_rn(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fmul_rn(a, b);
#else
  return a * b;
#endif
}
__host__ __device__ static inline double add_rn(double a, double b) {
#ifdef __CUDA_ARCH__
  return __dadd_rn(a, b);
#else
  return a + b;
#endif
}
__host__ __device__ static inline double mul_rn(double a, double b) {
#ifdef __CUDA_ARCH__
  return __dmul_rn(a, b);
#else
  return a * b;
#endif
}

// ---- f32 FMA ----------------------------------------------------------------
// Normal numbers throughout: every chain converges to a fixed point well
// inside the f32 range (a -> c/(1-b), b -> 2d, ...).
__host__ __device__ static inline float f32_chain(uint32_t i, int iters) {
  float a = 1.0f + float(i & 15) * 0.0625f, b = 0.75f + float((i >> 4) & 7) * 0.03125f;
  float c = 0.015625f, d = 0.0078125f, e = 2.0f, f = 0.5f;
  for (int k = 0; k < iters; ++k) {
#pragma unroll 8
    for (int j = 0; j < 8; ++j) {
      a = fmaf(a, b, c);
      e = fmaf(e, 0.5f, d);
      f = fmaf(f, b, a);
      b = fmaf(b, 0.9375f, d);
    }
    // plain add and multiply, never contracted: the intrinsics are exact on both sides
    c = add_rn(mul_rn(c, 0.5f), 0.0078125f);
  }
  return a + e + f + b + c;
}
__global__ void k_f32_fma(float* out, int iters) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = f32_chain(i, iters);
}
static int w_f32_fma() {
  const int blocks = 64, threads = 128, n = blocks * threads;
  const int iters = pick(60, 600);
  float* d;
  CK(cudaMalloc(&d, n * 4));
  phase_begin();
  k_f32_fma<<<blocks, threads>>>(d, iters);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<float> h(n);
  CK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
  phase_end();
  for (int i = 0; i < n; ++i)
    if (sampled(i)) {
      float want = f32_chain(i, iters);
      REQUIRE(std::memcmp(&h[i], &want, 4) == 0, "thread %d: %a vs %a", i, h[i], want);
    }
  CK(cudaFree(d));
  return 0;
}

// ---- f64 FMA ----------------------------------------------------------------
__host__ __device__ static inline double f64_chain(uint32_t i, int iters) {
  double a = 1.0 + double(i & 15) * 0.0625, b = 0.75 + double((i >> 4) & 7) * 0.03125;
  double c = 0.015625, d = 0.0078125, e = 2.0, f = 0.5;
  for (int k = 0; k < iters; ++k) {
#pragma unroll 8
    for (int j = 0; j < 8; ++j) {
      a = fma(a, b, c);
      e = fma(e, 0.5, d);
      f = fma(f, b, a);
      b = fma(b, 0.9375, d);
    }
    c = add_rn(mul_rn(c, 0.5), 0.0078125);
  }
  return a + e + f + b + c;
}
__global__ void k_f64_fma(double* out, int iters) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = f64_chain(i, iters);
}
static int w_f64_fma() {
  const int blocks = 64, threads = 128, n = blocks * threads;
  const int iters = pick(25, 250);
  double* d;
  CK(cudaMalloc(&d, n * 8));
  phase_begin();
  k_f64_fma<<<blocks, threads>>>(d, iters);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<double> h(n);
  CK(cudaMemcpy(h.data(), d, n * 8, cudaMemcpyDeviceToHost));
  phase_end();
  for (int i = 0; i < n; ++i)
    if (sampled(i)) {
      double want = f64_chain(i, iters);
      REQUIRE(std::memcmp(&h[i], &want, 8) == 0, "thread %d: %a vs %a", i, h[i], want);
    }
  CK(cudaFree(d));
  return 0;
}

// The pantheon fp64_virus chain (kernels/fp64_virus/fp64_virus.cpp). Its four
// coupled chains decay into the subnormal range within ~45 outer iterations
// (measured on the host: the first subnormal appears at i=43), after which
// every FMA has a subnormal operand or result, which costs a microcode assist
// per operation on Intel cores (TODO.md, Performance). Same instruction mix as
// f64_fma, very different host cost: compare the two.
__host__ __device__ static inline double f64_decay_chain(int iters) {
  double a = 1.0, b = 0.999, c = 0.001, d = -0.001;
  for (int i = 0; i < iters; ++i) {
#pragma unroll 32
    for (int j = 0; j < 32; ++j) {
      a = fma(a, b, c);
      b = fma(b, c, d);
      c = fma(c, d, a);
      d = fma(d, a, b);
    }
    if ((i & 0xFF) == 0) {
      a = (a > 2.0 || a < -2.0) ? 1.0 : -a;
      b = (b > 2.0 || b < -2.0) ? 0.999 : -b;
    }
  }
  return a + b + c + d;
}
__global__ void k_f64_decay(double* out, int iters) {
  out[blockIdx.x * blockDim.x + threadIdx.x] = f64_decay_chain(iters);
}
static int w_f64_subnormal() {
  const int blocks = 16, threads = 64, n = blocks * threads;
  const int iters = pick(60, 200);
  double* d;
  CK(cudaMalloc(&d, n * 8));
  phase_begin();
  k_f64_decay<<<blocks, threads>>>(d, iters);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<double> h(n);
  CK(cudaMemcpy(h.data(), d, n * 8, cudaMemcpyDeviceToHost));
  phase_end();
  const double want = f64_decay_chain(iters);  // every thread runs the same chain
  for (int i = 0; i < n; ++i)
    REQUIRE(std::memcmp(&h[i], &want, 8) == 0, "thread %d: %a vs %a", i, h[i], want);
  CK(cudaFree(d));
  return 0;
}

// ---- tensor cores -----------------------------------------------------------
// Small non-negative integers in half precision: every product and every sum is
// exact in f32 whatever order the unit adds in, so the bits are well defined.
static inline half h_of(int v) { return __float2half(float(v)); }
static inline int a_val(int w, int r, int c) { return (w * 7 + r * 3 + c * 5) % 3; }
static inline int b_val(int w, int r, int c) { return (w * 5 + r * 2 + c * 7) % 3; }

__global__ void k_wmma(const half* A, const half* B, float* C, int iters) {
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
  wmma::fill_fragment(fc, 0.0f);
  wmma::load_matrix_sync(fa, A + w * 256, 16);
  wmma::load_matrix_sync(fb, B + w * 256, 16);
  for (int k = 0; k < iters; ++k) wmma::mma_sync(fc, fa, fb, fc);
  wmma::store_matrix_sync(C + w * 256, fc, 16, wmma::mem_row_major);
}
static int w_wmma() {
  const int blocks = 8, threads = 128, warps = blocks * threads / 32;
  const int iters = pick(60, 3000);
  std::vector<half> hA(warps * 256), hB(warps * 256);
  for (int w = 0; w < warps; ++w)
    for (int r = 0; r < 16; ++r)
      for (int c = 0; c < 16; ++c) {
        hA[w * 256 + r * 16 + c] = h_of(a_val(w, r, c));  // A(r,c) row major
        hB[w * 256 + c * 16 + r] = h_of(b_val(w, r, c));  // B(r,c) column major
      }
  half *dA, *dB;
  float* dC;
  CK(cudaMalloc(&dA, hA.size() * 2));
  CK(cudaMalloc(&dB, hB.size() * 2));
  CK(cudaMalloc(&dC, warps * 256 * 4));
  CK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice));
  phase_begin();
  k_wmma<<<blocks, threads>>>(dA, dB, dC, iters);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<float> h(warps * 256);
  CK(cudaMemcpy(h.data(), dC, h.size() * 4, cudaMemcpyDeviceToHost));
  phase_end();
  for (int w = 0; w < warps; ++w)
    for (int r = 0; r < 16; ++r)
      for (int c = 0; c < 16; ++c) {
        int s = 0;
        for (int k = 0; k < 16; ++k) s += a_val(w, r, k) * b_val(w, k, c);
        float want = float(s * iters);
        REQUIRE(h[w * 256 + r * 16 + c] == want, "warp %d C(%d,%d)=%g want %g", w, r, c,
                h[w * 256 + r * 16 + c], want);
      }
  CK(cudaFree(dA));
  CK(cudaFree(dB));
  CK(cudaFree(dC));
  return 0;
}

// mma.sync m16n8k16 (f16 in, f32 accumulate) written as PTX, fragments loaded
// by the layout the PTX ISA documents (groupID = lane>>2, threadID_in_group =
// lane&3). On SASS this is HMMA.16816.F32.
__device__ static inline uint32_t pack2(const half* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}
__global__ void k_mma_ptx(const half* A, const half* B, float* C, int iters) {
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const half* a = A + w * 256;  // 16x16 row major
  const half* b = B + w * 128;  // 16x8 column major: B(k,n) at b[n*16+k]
  uint32_t a0 = pack2(a + g * 16 + t * 2), a1 = pack2(a + (g + 8) * 16 + t * 2);
  uint32_t a2 = pack2(a + g * 16 + t * 2 + 8), a3 = pack2(a + (g + 8) * 16 + t * 2 + 8);
  uint32_t b0 = pack2(b + g * 16 + t * 2), b1 = pack2(b + g * 16 + t * 2 + 8);
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  for (int k = 0; k < iters; ++k)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  float* c = C + w * 128;  // 16x8 row major
  c[g * 8 + t * 2] = c0;
  c[g * 8 + t * 2 + 1] = c1;
  c[(g + 8) * 8 + t * 2] = c2;
  c[(g + 8) * 8 + t * 2 + 1] = c3;
}
static int w_mma_ptx() {
  const int blocks = 8, threads = 128, warps = blocks * threads / 32;
  const int iters = pick(60, 3000);
  std::vector<half> hA(warps * 256), hB(warps * 128);
  for (int w = 0; w < warps; ++w) {
    for (int r = 0; r < 16; ++r)
      for (int c = 0; c < 16; ++c) hA[w * 256 + r * 16 + c] = h_of(a_val(w, r, c));
    for (int k = 0; k < 16; ++k)
      for (int n = 0; n < 8; ++n) hB[w * 128 + n * 16 + k] = h_of(b_val(w, k, n));
  }
  half *dA, *dB;
  float* dC;
  CK(cudaMalloc(&dA, hA.size() * 2));
  CK(cudaMalloc(&dB, hB.size() * 2));
  CK(cudaMalloc(&dC, warps * 128 * 4));
  CK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice));
  phase_begin();
  k_mma_ptx<<<blocks, threads>>>(dA, dB, dC, iters);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<float> h(warps * 128);
  CK(cudaMemcpy(h.data(), dC, h.size() * 4, cudaMemcpyDeviceToHost));
  phase_end();
  for (int w = 0; w < warps; ++w)
    for (int r = 0; r < 16; ++r)
      for (int c = 0; c < 8; ++c) {
        int s = 0;
        for (int k = 0; k < 16; ++k) s += a_val(w, r, k) * b_val(w, k, c);
        float want = float(s * iters);
        REQUIRE(h[w * 128 + r * 8 + c] == want, "warp %d C(%d,%d)=%g want %g", w, r, c,
                h[w * 128 + r * 8 + c], want);
      }
  CK(cudaFree(dA));
  CK(cudaFree(dB));
  CK(cudaFree(dC));
  return 0;
}

// ---- global memory ----------------------------------------------------------
// Coalesced scalar loads and stores, grid-stride.
__global__ void k_stream(const uint32_t* in, uint32_t* out, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    out[i] = in[i] * 3u + 7u;
}
// Coalesced 16-byte loads and stores.
__global__ void k_stream4(const uint4* in, uint4* out, size_t n4) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n4;
       i += size_t(gridDim.x) * blockDim.x) {
    uint4 v = in[i];
    v.x = v.x * 3u + 7u;
    v.y = v.y * 3u + 7u;
    v.z = v.z * 3u + 7u;
    v.w = v.w * 3u + 7u;
    out[i] = v;
  }
}
// One word per lane at a stride of 33 words: every lane lands in its own sector.
__global__ void k_gather(const uint32_t* in, uint32_t* out, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    out[i] = in[(i * 33u + 7u) & (n - 1)];
}
static int w_mem() {
  const size_t n = pick<size_t>(size_t(1) << 18, size_t(1) << 20);  // words, power of two
  const int passes = pick(2, 4);
  std::vector<uint32_t> h(n);
  for (size_t i = 0; i < n; ++i) h[i] = uint32_t(i * 2654435761u);
  uint32_t *din, *dout, *dout4, *dg;
  CK(cudaMalloc(&din, n * 4));
  CK(cudaMalloc(&dout, n * 4));
  CK(cudaMalloc(&dout4, n * 4));
  CK(cudaMalloc(&dg, n * 4));
  CK(cudaMemcpy(din, h.data(), n * 4, cudaMemcpyHostToDevice));
  phase_begin();
  for (int p = 0; p < passes; ++p) {
    k_stream<<<64, 256>>>(din, dout, n);
    k_stream4<<<64, 256>>>(reinterpret_cast<const uint4*>(din), reinterpret_cast<uint4*>(dout4),
                           n / 4);
    k_gather<<<64, 256>>>(din, dg, n);
  }
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<uint32_t> o(n), o4(n), og(n);
  CK(cudaMemcpy(o.data(), dout, n * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(o4.data(), dout4, n * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(og.data(), dg, n * 4, cudaMemcpyDeviceToHost));
  phase_end();
  for (size_t i = 0; i < n; ++i) {
    REQUIRE(o[i] == h[i] * 3u + 7u, "stream[%zu]", i);
    REQUIRE(o4[i] == h[i] * 3u + 7u, "stream4[%zu]", i);
    REQUIRE(og[i] == h[(i * 33u + 7u) & (n - 1)], "gather[%zu]", i);
  }
  CK(cudaFree(din));
  CK(cudaFree(dout));
  CK(cudaFree(dout4));
  CK(cudaFree(dg));
  return 0;
}

// ---- shared memory and barriers ---------------------------------------------
__global__ void k_sred(const uint32_t* in, uint32_t* out, int reps) {
  __shared__ uint32_t s[256];
  const int t = threadIdx.x;
  uint32_t acc = 0;
  for (int r = 0; r < reps; ++r) {
    s[t] = in[blockIdx.x * 256 + t] + uint32_t(r);
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
      if (t < st) s[t] += s[t + st];
      __syncthreads();
    }
    acc += s[0];
    __syncthreads();
  }
  if (t == 0) out[blockIdx.x] = acc;
}
// 32x32 tile transpose through shared memory: unpadded (a 32-way bank conflict
// on the read) and padded (conflict free).
template <int PAD>
__global__ void k_stranspose(const uint32_t* in, uint32_t* out, int width, int reps) {
  __shared__ uint32_t tile[32][32 + PAD];
  const int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y;
  for (int r = 0; r < reps; ++r) {
    for (int j = 0; j < 32; j += 8) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
    __syncthreads();
    const int ox = blockIdx.y * 32 + threadIdx.x, oy = blockIdx.x * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8) out[(oy + j) * width + ox] = tile[threadIdx.x][threadIdx.y + j];
    __syncthreads();
  }
}
static int w_shared() {
  const int blocks = pick(32, 64), reps = pick(6, 40);
  const int total = blocks * 256;
  std::vector<uint32_t> h(total);
  for (int i = 0; i < total; ++i) h[i] = uint32_t(i) * 2246822519u;
  uint32_t *d, *dr;
  CK(cudaMalloc(&d, total * 4));
  CK(cudaMalloc(&dr, blocks * 4));
  CK(cudaMemcpy(d, h.data(), total * 4, cudaMemcpyHostToDevice));
  const int W = 256;  // transpose a 256x256 matrix
  std::vector<uint32_t> m(W * W);
  for (int i = 0; i < W * W; ++i) m[i] = uint32_t(i) * 2654435761u + 1u;
  uint32_t *dm, *dt0, *dt1;
  CK(cudaMalloc(&dm, W * W * 4));
  CK(cudaMalloc(&dt0, W * W * 4));
  CK(cudaMalloc(&dt1, W * W * 4));
  CK(cudaMemcpy(dm, m.data(), W * W * 4, cudaMemcpyHostToDevice));
  const int treps = pick(2, 12);
  phase_begin();
  k_sred<<<blocks, 256>>>(d, dr, reps);
  k_stranspose<0><<<dim3(W / 32, W / 32), dim3(32, 8)>>>(dm, dt0, W, treps);
  k_stranspose<1><<<dim3(W / 32, W / 32), dim3(32, 8)>>>(dm, dt1, W, treps);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<uint32_t> r(blocks), t0(W * W), t1(W * W);
  CK(cudaMemcpy(r.data(), dr, blocks * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(t0.data(), dt0, W * W * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(t1.data(), dt1, W * W * 4, cudaMemcpyDeviceToHost));
  phase_end();
  for (int b = 0; b < blocks; ++b) {
    uint32_t acc = 0;
    for (int rep = 0; rep < reps; ++rep) {
      uint32_t s = 0;
      for (int t = 0; t < 256; ++t) s += h[b * 256 + t] + uint32_t(rep);
      acc += s;
    }
    REQUIRE(r[b] == acc, "block %d reduction", b);
  }
  for (int y = 0; y < W; ++y)
    for (int x = 0; x < W; ++x) {
      REQUIRE(t0[x * W + y] == m[y * W + x], "unpadded transpose (%d,%d)", y, x);
      REQUIRE(t1[x * W + y] == m[y * W + x], "padded transpose (%d,%d)", y, x);
    }
  CK(cudaFree(d));
  CK(cudaFree(dr));
  CK(cudaFree(dm));
  CK(cudaFree(dt0));
  CK(cudaFree(dt1));
  return 0;
}

// ---- atomics ----------------------------------------------------------------
// Every result is independent of the order the atomics land in.
__global__ void k_atomics(unsigned* single, unsigned* hist, unsigned* mx, unsigned* casctr,
                          float* fsum, unsigned long long* wide, int iters) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  for (int k = 0; k < iters; ++k) {
    atomicAdd(single, 1u);                              // one address, every lane
    atomicAdd(&hist[(i * 7u + uint32_t(k)) & 255u], 1u);  // 256 bins
    atomicMax(mx, (i * 2654435761u) ^ uint32_t(k));
    atomicAdd(fsum, float((i + uint32_t(k)) & 7u));
    atomicAdd(wide, (unsigned long long)(i) << 20);
  }
  // a compare-and-swap increment loop, one per thread
  unsigned old = *casctr, assumed;
  do {
    assumed = old;
    old = atomicCAS(casctr, assumed, assumed + 1u);
  } while (old != assumed);
}
static int w_atomics() {
  const int blocks = pick(16, 32), threads = 128, n = blocks * threads;
  const int iters = pick(8, 100);
  unsigned *single, *hist, *mx, *cas;
  float* fs;
  unsigned long long* wide;
  CK(cudaMalloc(&single, 4));
  CK(cudaMalloc(&hist, 256 * 4));
  CK(cudaMalloc(&mx, 4));
  CK(cudaMalloc(&cas, 4));
  CK(cudaMalloc(&fs, 4));
  CK(cudaMalloc(&wide, 8));
  CK(cudaMemset(single, 0, 4));
  CK(cudaMemset(hist, 0, 256 * 4));
  CK(cudaMemset(mx, 0, 4));
  CK(cudaMemset(cas, 0, 4));
  CK(cudaMemset(fs, 0, 4));
  CK(cudaMemset(wide, 0, 8));
  phase_begin();
  k_atomics<<<blocks, threads>>>(single, hist, mx, cas, fs, wide, iters);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  unsigned h_single, h_mx, h_cas;
  std::vector<unsigned> h_hist(256);
  float h_fs;
  unsigned long long h_wide;
  CK(cudaMemcpy(&h_single, single, 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h_hist.data(), hist, 1024, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&h_mx, mx, 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&h_cas, cas, 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&h_fs, fs, 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&h_wide, wide, 8, cudaMemcpyDeviceToHost));
  phase_end();
  std::vector<unsigned> w_hist(256, 0);
  unsigned w_mx = 0;
  double w_fs = 0;
  unsigned long long w_wide = 0;
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < iters; ++k) {
      w_hist[(uint32_t(i) * 7u + uint32_t(k)) & 255u]++;
      unsigned v = (uint32_t(i) * 2654435761u) ^ uint32_t(k);
      if (v > w_mx) w_mx = v;
      w_fs += double((uint32_t(i) + uint32_t(k)) & 7u);
      w_wide += (unsigned long long)(uint32_t(i)) << 20;
    }
  REQUIRE(h_single == unsigned(n) * unsigned(iters), "single counter %u", h_single);
  for (int b = 0; b < 256; ++b) REQUIRE(h_hist[b] == w_hist[b], "hist[%d]", b);
  REQUIRE(h_mx == w_mx, "max %u vs %u", h_mx, w_mx);
  REQUIRE(h_cas == unsigned(n), "cas counter %u", h_cas);
  REQUIRE(double(h_fs) == w_fs && w_fs < 16777216.0, "float sum %g vs %g", h_fs, w_fs);
  REQUIRE(h_wide == w_wide, "64-bit sum");
  CK(cudaFree(single));
  CK(cudaFree(hist));
  CK(cudaFree(mx));
  CK(cudaFree(cas));
  CK(cudaFree(fs));
  CK(cudaFree(wide));
  return 0;
}

// ---- textures ---------------------------------------------------------------
__global__ void k_tex(cudaTextureObject_t pt, cudaTextureObject_t lin, cudaTextureObject_t lin1d,
                      float* out, int w, int h, int iters) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.0f;
  for (int k = 0; k < iters; ++k) {
    int x = (i * 13u + uint32_t(k) * 7u) % uint32_t(w), y = (i * 5u + uint32_t(k) * 3u) % uint32_t(h);
    acc += tex2D<float>(pt, float(x) + 0.5f, float(y) + 0.5f);   // point filter
    acc += tex2D<float>(lin, float(x) + 0.5f, float(y) + 0.5f);  // linear filter at a texel centre
    acc += tex1Dfetch<float>(lin1d, y * w + x);
  }
  out[i] = acc;
}
static float tex_val(int x, int y) { return float((x * 7 + y * 13) % 251); }
static int w_texture() {
  const int W = 128, H = 128;
  std::vector<float> img(W * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) img[y * W + x] = tex_val(x, y);
  cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
  cudaArray_t arr;
  CK(cudaMallocArray(&arr, &fmt, W, H));
  CK(cudaMemcpy2DToArray(arr, 0, 0, img.data(), W * 4, W * 4, H, cudaMemcpyHostToDevice));
  float* lin;
  CK(cudaMalloc(&lin, W * H * 4));
  CK(cudaMemcpy(lin, img.data(), W * H * 4, cudaMemcpyHostToDevice));
  cudaResourceDesc ard{};
  ard.resType = cudaResourceTypeArray;
  ard.res.array.array = arr;
  cudaTextureDesc td{};
  td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
  td.readMode = cudaReadModeElementType;
  td.filterMode = cudaFilterModePoint;
  cudaTextureObject_t pt = 0, bil = 0, t1d = 0;
  CK(cudaCreateTextureObject(&pt, &ard, &td, nullptr));
  td.filterMode = cudaFilterModeLinear;
  CK(cudaCreateTextureObject(&bil, &ard, &td, nullptr));
  cudaResourceDesc lrd{};
  lrd.resType = cudaResourceTypeLinear;
  lrd.res.linear.devPtr = lin;
  lrd.res.linear.desc = fmt;
  lrd.res.linear.sizeInBytes = W * H * 4;
  cudaTextureDesc ltd{};
  ltd.readMode = cudaReadModeElementType;
  CK(cudaCreateTextureObject(&t1d, &lrd, &ltd, nullptr));
  const int blocks = pick(16, 32), threads = 128, n = blocks * threads, iters = pick(20, 120);
  float* d;
  CK(cudaMalloc(&d, n * 4));
  phase_begin();
  k_tex<<<blocks, threads>>>(pt, bil, t1d, d, W, H, iters);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<float> o(n);
  CK(cudaMemcpy(o.data(), d, n * 4, cudaMemcpyDeviceToHost));
  phase_end();
  for (int i = 0; i < n; ++i) {
    float acc = 0.0f;
    for (int k = 0; k < iters; ++k) {
      int x = int((uint32_t(i) * 13u + uint32_t(k) * 7u) % uint32_t(W));
      int y = int((uint32_t(i) * 5u + uint32_t(k) * 3u) % uint32_t(H));
      acc += tex_val(x, y) * 3.0f;
    }
    REQUIRE(o[i] == acc, "thread %d: %g vs %g", i, o[i], acc);
  }
  CK(cudaDestroyTextureObject(pt));
  CK(cudaDestroyTextureObject(bil));
  CK(cudaDestroyTextureObject(t1d));
  CK(cudaFreeArray(arr));
  CK(cudaFree(lin));
  CK(cudaFree(d));
  return 0;
}

// ---- divergent control flow --------------------------------------------------
__host__ __device__ static inline uint32_t collatz_steps(uint32_t n) {
  uint32_t steps = 0;
  while (n != 1u && steps < 1000u) {
    n = (n & 1u) ? 3u * n + 1u : n >> 1;
    ++steps;
  }
  return steps;
}
__host__ __device__ static inline uint32_t branchy(uint32_t i, int iters) {
  uint32_t v = i * 2654435761u;
  for (int k = 0; k < iters; ++k) {
    switch ((v >> 3) & 3u) {
      case 0: v = v * 5u + 1u; break;
      case 1: v = (v ^ (v >> 7)) + uint32_t(k); break;
      case 2: v = rotl32(v, 9) - 3u; break;
      default:
        if (v & 0x100u) v += 0x9e3779b9u; else v ^= 0x85ebca6bu;
        break;
    }
  }
  return v;
}
__global__ void k_divergent(uint32_t* steps, uint32_t* br, int iters) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  steps[i] = collatz_steps(i % 6000u + 1u);
  br[i] = branchy(i, iters);
}
static int w_divergent() {
  const int blocks = pick(16, 64), threads = 128, n = blocks * threads, iters = pick(200, 1500);
  uint32_t *ds, *db;
  CK(cudaMalloc(&ds, n * 4));
  CK(cudaMalloc(&db, n * 4));
  phase_begin();
  k_divergent<<<blocks, threads>>>(ds, db, iters);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<uint32_t> s(n), b(n);
  CK(cudaMemcpy(s.data(), ds, n * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(b.data(), db, n * 4, cudaMemcpyDeviceToHost));
  phase_end();
  for (int i = 0; i < n; ++i) {
    REQUIRE(s[i] == collatz_steps(uint32_t(i) % 6000u + 1u), "collatz thread %d", i);
    if (sampled(i)) REQUIRE(b[i] == branchy(i, iters), "branchy thread %d", i);
  }
  CK(cudaFree(ds));
  CK(cudaFree(db));
  return 0;
}

// ---- warp-level operations ---------------------------------------------------
// Shuffles, votes and ballots. The host reference runs the same steps on a
// 32-element array standing for one warp.
__global__ void k_warp(uint32_t* out, int iters) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned full = 0xffffffffu;
  uint32_t v = i * 2654435761u + 1u;
  for (int k = 0; k < iters; ++k) {
    v += __shfl_xor_sync(full, v, 1 << (k & 3));
    uint32_t b = __ballot_sync(full, (v & 4u) != 0u);
    v ^= __popc(b) * 0x01000193u;
    v += __shfl_down_sync(full, v, 3);
    if (__any_sync(full, (v & 0xffu) == 0u)) v += 1u;
    if (__all_sync(full, (v & 1u) == 1u)) v ^= 0x55u;
    v = __shfl_sync(full, v, (k * 5) & 31) ^ (v << 1);
  }
  out[i] = v;
}
static void warp_ref(uint32_t* v, uint32_t base, int iters) {
  for (int l = 0; l < 32; ++l) v[l] = (base + l) * 2654435761u + 1u;
  uint32_t t[32];
  for (int k = 0; k < iters; ++k) {
    for (int l = 0; l < 32; ++l) t[l] = v[l ^ (1 << (k & 3))];
    for (int l = 0; l < 32; ++l) v[l] += t[l];
    uint32_t b = 0;
    for (int l = 0; l < 32; ++l) b |= uint32_t((v[l] & 4u) != 0u) << l;
    for (int l = 0; l < 32; ++l) v[l] ^= uint32_t(__builtin_popcount(b)) * 0x01000193u;
    for (int l = 0; l < 32; ++l) t[l] = (l + 3 < 32) ? v[l + 3] : v[l];  // __shfl_down: out of range keeps own
    for (int l = 0; l < 32; ++l) v[l] += t[l];
    bool any = false;
    for (int l = 0; l < 32; ++l) any |= (v[l] & 0xffu) == 0u;
    if (any) for (int l = 0; l < 32; ++l) v[l] += 1u;
    bool all = true;
    for (int l = 0; l < 32; ++l) all &= (v[l] & 1u) == 1u;
    if (all) for (int l = 0; l < 32; ++l) v[l] ^= 0x55u;
    const uint32_t src = v[(k * 5) & 31];
    for (int l = 0; l < 32; ++l) v[l] = src ^ (v[l] << 1);
  }
}
static int w_warp() {
  const int blocks = pick(16, 64), threads = 128, n = blocks * threads, iters = pick(80, 800);
  uint32_t* d;
  CK(cudaMalloc(&d, n * 4));
  phase_begin();
  k_warp<<<blocks, threads>>>(d, iters);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<uint32_t> o(n);
  CK(cudaMemcpy(o.data(), d, n * 4, cudaMemcpyDeviceToHost));
  phase_end();
  for (int w = 0; w < n / 32; ++w)
    if (w % 5 == 0) {
      uint32_t v[32];
      warp_ref(v, uint32_t(w) * 32, iters);
      for (int l = 0; l < 32; ++l) REQUIRE(o[w * 32 + l] == v[l], "warp %d lane %d", w, l);
    }
  CK(cudaFree(d));
  return 0;
}

// ---- local memory ------------------------------------------------------------
// A per-thread array indexed by a value the compiler cannot resolve, so it
// lives in local memory (ld.local/st.local, LDL/STL).
__host__ __device__ static inline uint32_t local_chain(uint32_t i, int iters) {
  uint32_t buf[64];
  for (int j = 0; j < 64; ++j) buf[j] = i + uint32_t(j) * 977u;
  uint32_t acc = 0;
  for (int k = 0; k < iters; ++k) {
    const uint32_t a = (i + uint32_t(k) * 5u) & 63u, b = (acc + uint32_t(k) * 3u) & 63u;
    buf[a] = buf[b] + uint32_t(k) + acc;
    acc ^= buf[(a + 17u) & 63u] * 2654435761u;
  }
  uint32_t s = acc;
  for (int j = 0; j < 64; ++j) s += buf[j] * uint32_t(j + 1);
  return s;
}
__global__ void k_local(uint32_t* out, int iters) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = local_chain(i, iters);
}
static int w_local() {
  const int blocks = pick(16, 64), threads = 128, n = blocks * threads, iters = pick(300, 1500);
  uint32_t* d;
  CK(cudaMalloc(&d, n * 4));
  phase_begin();
  k_local<<<blocks, threads>>>(d, iters);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<uint32_t> o(n);
  CK(cudaMemcpy(o.data(), d, n * 4, cudaMemcpyDeviceToHost));
  phase_end();
  for (int i = 0; i < n; ++i)
    if (sampled(i)) REQUIRE(o[i] == local_chain(i, iters), "thread %d", i);
  CK(cudaFree(d));
  return 0;
}

// ---- launch overhead ---------------------------------------------------------
// Thousands of one-warp launches: the cost is the launch path, not the kernel.
__global__ void k_tiny(uint32_t* p) { p[threadIdx.x] += 1u; }
static int w_launch() {
  const int launches = pick(1500, 40000);
  uint32_t* d;
  CK(cudaMalloc(&d, 32 * 4));
  CK(cudaMemset(d, 0, 32 * 4));
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  phase_begin();
  for (int i = 0; i < launches; ++i) k_tiny<<<1, 32, 0, (i & 1) ? s : 0>>>(d);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<uint32_t> o(32);
  CK(cudaMemcpy(o.data(), d, 128, cudaMemcpyDeviceToHost));
  phase_end();
  g_phase.units = (unsigned long long)launches;
  g_phase.unit = "launches";
  for (int i = 0; i < 32; ++i) REQUIRE(o[i] == uint32_t(launches), "lane %d: %u", i, o[i]);
  CK(cudaStreamDestroy(s));
  CK(cudaFree(d));
  return 0;
}

// ---- API-call overhead -------------------------------------------------------
static int w_api() {
  const int rounds = pick(3000, 60000);
  std::vector<uint32_t> src(1024), back(1024);
  for (int i = 0; i < 1024; ++i) src[i] = uint32_t(i) * 2654435761u;
  unsigned long long calls = 0;
  int bad = 0;
  uint32_t* keep;
  CK(cudaMalloc(&keep, 4096));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  phase_begin();
  for (int r = 0; r < rounds; ++r) {
    void* p;
    CK(cudaMalloc(&p, 4096 + (r & 7) * 256));
    CK(cudaMemcpy(p, src.data(), 4096, cudaMemcpyHostToDevice));
    CK(cudaMemset(keep, r & 0xff, 4096));
    CK(cudaMemcpy(keep, p, 4096, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(back.data(), keep, 4096, cudaMemcpyDeviceToHost));
    CK(cudaEventRecord(e0));
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaFree(p));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    CK(cudaStreamSynchronize(s));
    CK(cudaStreamDestroy(s));
    int dev;
    CK(cudaGetDevice(&dev));
    CK(cudaGetLastError());
    calls += 15;
    if (std::memcmp(back.data(), src.data(), 4096) != 0) ++bad;
  }
  phase_end();
  REQUIRE(bad == 0, "%d round trips came back wrong", bad);
  g_phase.units = calls;
  g_phase.unit = "api_calls";
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  CK(cudaFree(keep));
  return 0;
}

// ---- startup -----------------------------------------------------------------
// Context creation, module registration and one launch: what a program pays
// before it does any work.
static int w_startup() {
  uint32_t* d;
  phase_begin();
  CK(cudaMalloc(&d, 128));
  CK(cudaMemset(d, 0, 128));
  k_tiny<<<1, 32>>>(d);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<uint32_t> o(32);
  CK(cudaMemcpy(o.data(), d, 128, cudaMemcpyDeviceToHost));
  phase_end();
  for (int i = 0; i < 32; ++i) REQUIRE(o[i] == 1u, "lane %d", i);
  CK(cudaFree(d));
  return 0;
}

struct Entry {
  const char* name;
  int (*fn)();
};
static const Entry kWorkloads[] = {
    {"int_alu", w_int_alu},         {"f32_fma", w_f32_fma},     {"f64_fma", w_f64_fma},
    {"f64_subnormal", w_f64_subnormal}, {"wmma", w_wmma},       {"mma_ptx", w_mma_ptx},
    {"mem", w_mem},                 {"shared", w_shared},       {"atomics", w_atomics},
    {"texture", w_texture},         {"divergent", w_divergent}, {"warp", w_warp},
    {"local", w_local},             {"launch", w_launch},       {"api", w_api},
    {"startup", w_startup},
};

int main(int argc, char** argv) {
  if (argc >= 2 && std::strcmp(argv[1], "--list") == 0) {
    for (const Entry& e : kWorkloads) std::printf("%s\n", e.name);
    return 0;
  }
  if (argc != 3 || (std::strcmp(argv[2], "quick") && std::strcmp(argv[2], "full"))) {
    std::fprintf(stderr, "usage: %s <workload> <quick|full>   (or --list)\n", argv[0]);
    return 2;
  }
  g_full = std::strcmp(argv[2], "full") == 0;
  for (const Entry& e : kWorkloads) {
    if (std::strcmp(e.name, argv[1])) continue;
    const int rc = e.fn();
    if (rc) {
      std::printf("FAIL %s\n", e.name);
      return 1;
    }
    std::printf("BENCH workload=%s size=%s device_cpu_s=%.6f device_wall_s=%.6f units=%llu unit=%s\n",
                e.name, argv[2], g_phase.cpu, g_phase.wall, g_phase.units,
                g_phase.unit[0] ? g_phase.unit : "none");
    std::printf("PASS\n");
    return 0;
  }
  std::fprintf(stderr, "unknown workload '%s' (try --list)\n", argv[1]);
  return 2;
}
