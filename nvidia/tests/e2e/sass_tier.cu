// The SASS executor's pre-decoded tier (src/sass/tier.hpp) against the interpreter (VGPU_TIER=0).
// run_sass_tier.sh runs this program both ways with the performance counters on and requires byte-for-byte the same
// output: every result's hash, and every counter of every launch.
//
// Each kernel computes a spread of expressions from three input arrays whose values break arithmetic -- zeros of both
// signs, the extremes, NaNs quiet and signalling with payloads, infinities, subnormals, shift counts past the width,
// carries and borrows at every word boundary -- mixed with random bits, so that ptxas emits the forms the tier has a
// handler for (IADD3 with carries, IMAD and its WIDE/HI/X forms, LOP3, SHF, LEA, SEL, ISETP and its .EX form, POPC,
// I2FP, FFMA/FADD/FMUL with |x| and -x, FSETP, FSEL, DFMA/DADD/DMUL, global/shared/local loads and stores of every size)
// and the forms it leaves to the interpreter (division, conversions, transcendentals, atomics, shuffles, warp votes).
// Launches use blocks that are not whole warps and guards that leave lanes idle, so the handlers see partial masks.
//
// The program prints no verdict of its own on the arithmetic: the interpreter is the reference here, and it is held
// to the card by the other SASS tests.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t i32;
typedef int64_t i64;

#define CK(x) do { cudaError_t e_ = (x); if (e_) { std::printf("error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

__device__ __forceinline__ float F(u32 b) { return __uint_as_float(b); }
__device__ __forceinline__ u32 B(float f) { return __float_as_uint(f); }
__device__ __forceinline__ double D(u64 b) { return __longlong_as_double((long long)b); }
__device__ __forceinline__ u64 Bd(double d) { return (u64)__double_as_longlong(d); }

__constant__ u32 kc[64];

// ---- integer ------------------------------------------------------------------------------------
constexpr int kInt = 64;
__global__ void k_int(const u32* A, const u32* Bv, const u32* C, u32* out, int n, u32 p0, u32 p1) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  u32 a = A[i], b = Bv[i], c = C[i];
  i32 sa = (i32)a, sb = (i32)b, sc = (i32)c;
  u64 x = ((u64)a << 32) | b, y = ((u64)c << 32) | a;
  u32 r[kInt];
  int k = 0;
  r[k++] = a + b + c;            r[k++] = a - b - c;           r[k++] = a + b - c;          r[k++] = -a + b - c;
  r[k++] = a * b + c;            r[k++] = a * b - c;           r[k++] = c - a * b;          r[k++] = a * 37u + 1000003u;
  r[k++] = (a << 3) + b;         r[k++] = (a << 7) - b;        r[k++] = a * p0 + p1;        r[k++] = (a + p0) * (b ^ p1);
  r[k++] = (a & b) ^ c;          r[k++] = (a | b) & ~c;        r[k++] = ~(a ^ b) | c;       r[k++] = a & ~b;
  r[k++] = ~a;                   r[k++] = a ^ b ^ c;           r[k++] = (a & b) | (~a & c); r[k++] = (a | b) ^ (b & c);
  r[k++] = __funnelshift_l(a, b, c);  r[k++] = __funnelshift_r(a, b, c);
  r[k++] = __funnelshift_lc(a, b, c); r[k++] = __funnelshift_rc(a, b, c);
  r[k++] = (u32)(x >> (c & 63));  r[k++] = (u32)((i64)x >> (c & 63)); r[k++] = (u32)(x << (c & 63));
  r[k++] = (u32)(((u64)a << (b & 31)) >> 32); r[k++] = a >> (b & 31); r[k++] = (u32)(sa >> (b & 31));
  r[k++] = (a < b) ? c : a;      r[k++] = (sa < sb) ? sb : sc; r[k++] = (a == b) ? c : b;   r[k++] = (a != c) ? a : b;
  r[k++] = (sa >= sc) ? a : b;   r[k++] = (a <= 0x80000000u) ? b : c;
  r[k++] = (u32)(x + y);         r[k++] = (u32)((x + y) >> 32); r[k++] = (u32)(x - y);        r[k++] = (u32)((x - y) >> 32);
  r[k++] = (u32)(x * y);         r[k++] = (u32)((x * y) >> 32); r[k++] = (u32)((u64)a * b);   r[k++] = (u32)(((u64)a * b) >> 32);
  r[k++] = (u32)((i64)sa * sb);  r[k++] = (u32)(((i64)sa * sb) >> 32); r[k++] = __umulhi(a, b); r[k++] = (u32)__mulhi(sa, sb);
  r[k++] = (u32)__umul64hi(x, y); r[k++] = (u32)(__mul64hi((i64)x, (i64)y));
  r[k++] = (u32)(x < y);         r[k++] = (u32)((i64)x < (i64)y); r[k++] = (u32)(x == y);     r[k++] = (u32)(x >= y);
  r[k++] = (u32)(x + c);         r[k++] = (u32)((x + (u64)a * 8) >> 32);
  r[k++] = __popc(a);            r[k++] = __clz(a);            r[k++] = __brev(a);          r[k++] = __ffs(a);
  r[k++] = __byte_perm(a, b, c);  r[k++] = __byte_perm(a, b, 0x5410);
  r[k++] = (a >> 5) & 0x1ff;     r[k++] = (a & 0xfff0f00fu) | (b & 0x000f0ff0u);
  r[k++] = a / 7u;               r[k++] = a % 13u;             r[k++] = (u32)(sa / -5);     r[k++] = a / (b | 1u);
  // r[] holds kInt values at most.
  for (int j = 0; j < kInt && j < k; ++j) out[(size_t)i * kInt + j] = r[j];
  for (int j = k; j < kInt; ++j) out[(size_t)i * kInt + j] = 0;
}

// Branches on data, so that lanes diverge and the handlers see partial masks of every shape.
__global__ void k_branch(const u32* A, const u32* Bv, const u32* C, u32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  u32 a = A[i], b = Bv[i], c = C[i];
  u32 acc = a;
  for (int it = 0; it < 12 + (int)(c & 7); ++it) {
    if (acc & 1) acc = acc * 3u + b;
    else if (acc & 2) acc = (acc >> 1) ^ c;
    else acc = acc + (b << (it & 7)) - 5u;
    if ((acc & 0xff) == (b & 0xff)) acc += it;
    switch (acc & 3) {
      case 0: acc ^= 0x9e3779b9u; break;
      case 1: acc = acc * 5u + 1u; break;
      case 2: acc -= c; break;
      default: acc = (acc << 5) | (acc >> 27);
    }
  }
  out[i] = acc;
}

// ---- float and double ---------------------------------------------------------------------------
constexpr int kFlt = 64;
__global__ void k_float(const u32* A, const u32* Bv, const u32* C, u32* out, int n, u32 p0) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = F(A[i]), y = F(Bv[i]), z = F(C[i]);
  float q = F(p0);
  u32 r[kFlt];
  int k = 0;
  r[k++] = B(fmaf(x, y, z));     r[k++] = B(fmaf(-x, y, z));   r[k++] = B(fmaf(x, y, -z));  r[k++] = B(fmaf(-x, y, -z));
  r[k++] = B(fmaf(fabsf(x), y, z)); r[k++] = B(fmaf(x, fabsf(y), -fabsf(z))); r[k++] = B(fmaf(x, 2.5f, y)); r[k++] = B(fmaf(x, y, 0.5f));
  r[k++] = B(fmaf(x, q, z));     r[k++] = B(fmaf(q, y, z));    r[k++] = B(fmaf(x, y, q));   r[k++] = B(fmaf(-fabsf(x), q, y));
  r[k++] = B(x + y);             r[k++] = B(x - y);            r[k++] = B(-x - y);          r[k++] = B(x + 1.0f);
  r[k++] = B(fabsf(x) + fabsf(y)); r[k++] = B(x + q);          r[k++] = B(x - q);           r[k++] = B(__fadd_rn(x, -y));
  r[k++] = B(x * y);             r[k++] = B(-x * y);           r[k++] = B(x * 3.0f);        r[k++] = B(x * q);
  r[k++] = B(fabsf(x) * -fabsf(y)); r[k++] = B(__fmul_rn(x, y)); r[k++] = B(x * 0.5f);      r[k++] = B(x * -2.0f);
  r[k++] = (x < y) ? B(x) : B(y);   r[k++] = (x >= z) ? B(z) : B(x); r[k++] = (x == y) ? B(z) : B(x); r[k++] = (x != y) ? B(y) : B(z);
  r[k++] = (x > y && y > z) ? 1u : 0u; r[k++] = (x < y || z != z) ? 1u : 0u; r[k++] = (fabsf(x) < 1.0f) ? B(-x) : B(fabsf(y));
  r[k++] = (x > q) ? B(x) : B(q);    r[k++] = (x != x) ? 7u : 9u;
  r[k++] = B(fminf(x, y));       r[k++] = B(fmaxf(x, y));      r[k++] = B(copysignf(x, y)); r[k++] = B(fabsf(x));
  r[k++] = B(truncf(x));         r[k++] = B(floorf(x));        r[k++] = B(ceilf(x));        r[k++] = B(rintf(x));
  r[k++] = (u32)(i32)x;          r[k++] = (u32)(unsigned)fabsf(x); r[k++] = B((float)(i32)A[i]); r[k++] = B((float)A[i]);
  r[k++] = B((float)(i32)Bv[i]); r[k++] = B((float)(u32)(Bv[i] >> 3)); r[k++] = B(sqrtf(fabsf(x)));  r[k++] = B(rsqrtf(fabsf(y)));
  r[k++] = B(1.0f / z);          r[k++] = B(x / y);            r[k++] = B(__fdividef(x, y)); r[k++] = B(__expf(x * 0.01f));
  r[k++] = B(__logf(fabsf(x))); r[k++] = B(__sinf(x * 0.1f)); r[k++] = B(__cosf(y * 0.1f)); r[k++] = B(__saturatef(x));
  for (int j = 0; j < kFlt; ++j) out[(size_t)i * kFlt + j] = j < k ? r[j] : 0;
}

constexpr int kDbl = 24;
__global__ void k_double(const u64* A, const u64* Bv, const u64* C, u64* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double x = D(A[i]), y = D(Bv[i]), z = D(C[i]);
  u64 r[kDbl];
  int k = 0;
  r[k++] = Bd(fma(x, y, z));     r[k++] = Bd(fma(-x, y, z));   r[k++] = Bd(fma(x, y, -z));  r[k++] = Bd(fma(fabs(x), y, z));
  r[k++] = Bd(fma(x, 1.0000001, y)); r[k++] = Bd(fma(x, y, 0.25)); r[k++] = Bd(x + y);       r[k++] = Bd(x - y);
  r[k++] = Bd(x * y);            r[k++] = Bd(-x * y);          r[k++] = Bd(x * 3.5);        r[k++] = Bd(x + 1.5);
  r[k++] = Bd(fabs(x) + fabs(y)); r[k++] = (x < y) ? 1u : 0u;  r[k++] = (x >= z) ? 3u : 4u; r[k++] = (x != x) ? 5u : 6u;
  r[k++] = Bd(fmin(x, y));       r[k++] = Bd(fmax(x, y));      r[k++] = Bd(sqrt(fabs(x)));  r[k++] = Bd(x / y);
  r[k++] = Bd((double)(float)x); r[k++] = (u64)(i64)x;         r[k++] = Bd((double)(i64)A[i]); r[k++] = Bd(__dmul_rn(x, y));
  for (int j = 0; j < kDbl; ++j) out[(size_t)i * kDbl + j] = j < k ? r[j] : 0;
}

// ---- memory -------------------------------------------------------------------------------------
template <class T>
__global__ void k_copy(const T* in, T* out, int n, int lane_limit, int stride, int shift) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && (threadIdx.x % 32) < lane_limit) {
    out[i] = in[((long)i * stride + shift) % n];
    out[n + i] = in[(i ^ shift) % n];   // a second store stream: scattered and coalesced
  }
}

template <class T>
__global__ void k_shared(const T* in, T* out, int mul, int add) {
  __shared__ T s[512];
  int t = threadIdx.x;
  s[(t * mul + add) % 512] = in[blockIdx.x * blockDim.x + t];
  __syncthreads();
  out[blockIdx.x * blockDim.x + t] = s[(t * mul) % 512] ^ s[(t * 5 + 1) % 512] ;
}

struct __align__(16) Q { u32 a, b, c, d; };
__device__ __forceinline__ Q operator^(Q x, Q y) { return Q{x.a ^ y.a, x.b ^ y.b, x.c ^ y.c, x.d ^ y.d}; }

// Local memory: a per-thread array indexed by data (LDL/STL, byte, word and 16-byte forms).
__global__ void k_local(const u32* A, const u32* Bv, u32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  u32 arr[24];
  unsigned char bytes[40];
  Q q[6];
  for (int j = 0; j < 24; ++j) arr[j] = A[(i + j) % n] * (u32)(j + 1);
  for (int j = 0; j < 40; ++j) bytes[j] = (unsigned char)(Bv[(i + j) % n] >> (j & 24));
  for (int j = 0; j < 6; ++j) q[j] = Q{A[(i + j) % n], Bv[(i + j) % n], (u32)j, (u32)i};
  u32 b = Bv[i], acc = 0;
  for (int it = 0; it < 24; ++it) {
    arr[(b + it) % 24] += arr[(b >> 3) % 24] ^ it;
    bytes[(b >> 1) % 40] += (unsigned char)arr[it];
    q[b % 6].c += q[(b >> 4) % 6].a;
    acc += arr[(it * 5) % 24] + bytes[(it * 7) % 40] + q[it % 6].b + q[(it + 1) % 6].c;
  }
  out[i] = acc;
}

// Constant memory (read through the constant bank or the global window) and kernel arguments as operands.
__global__ void k_const(const u32* A, u32* out, int n, u32 p0, u32 p1, u32 p2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  u32 a = A[i];
  out[i] = kc[a & 63] + kc[(a >> 6) & 63] * p0 + (a ^ p1) + p2 + blockDim.x * 3u + gridDim.x + blockIdx.x;
}

// Generic loads and stores, and atomics, left to the interpreter.
__global__ void k_generic(u32* g, const u32* A, int n) {
  __shared__ u32 s[256];
  int t = threadIdx.x;
  s[t] = A[blockIdx.x * blockDim.x + t];
  __syncthreads();
  u32* p = (t & 1) ? &s[(t * 7) & 255] : &g[blockIdx.x * blockDim.x + t];
  u32 v = *p;
  *p = v * 3u + t;
  atomicAdd(&g[(t + blockIdx.x) & 63], v | 1u);
  g[blockIdx.x * blockDim.x + t] ^= __shfl_xor_sync(0xffffffffu, v, 3) + __popc(__ballot_sync(0xffffffffu, v & 4));
}

// ---- driver -------------------------------------------------------------------------------------
static u64 rng = 88172645463325252ull;
static u64 next_rand() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static const u32 kSpecial32[] = {0u, 1u, 2u, 3u, 31u, 32u, 33u, 63u, 64u, 0x7fffffffu, 0x80000000u, 0x80000001u, 0xffffffffu, 0xfffffffeu,
                                 0xffffu, 0x10000u, 0xdeadbeefu, 0x55555555u, 0xaaaaaaaau, 0x00ff00ffu,
                                 // floats
                                 0x3f800000u, 0xbf800000u, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7f800001u, 0xffc00001u, 0x7fffffffu,
                                 0x00000001u, 0x007fffffu, 0x80000001u, 0x807fffffu, 0x00800000u, 0x7f7fffffu, 0xff7fffffu, 0x4b000000u,
                                 0x33800000u, 0x3f800001u, 0x2f800000u, 0x5f800000u, 0x42280000u, 0xc2c80000u, 0x3dcccccdu, 0x40490fdbu};
static const u64 kSpecial64[] = {0ull, 1ull, 0x8000000000000000ull, 0x7fffffffffffffffull, 0xffffffffffffffffull, 0x7ff0000000000000ull,
                                 0xfff0000000000000ull, 0x7ff8000000000000ull, 0x7ff0000000000001ull, 0x0000000000000001ull,
                                 0x000fffffffffffffull, 0x0010000000000000ull, 0x3ff0000000000000ull, 0xbff0000000000000ull,
                                 0x7fefffffffffffffull, 0x4330000000000000ull, 0x3cb0000000000000ull, 0x3ff0000000000001ull,
                                 0x00000000ffffffffull, 0xffffffff00000000ull, 0x4059000000000000ull, 0x3fb999999999999aull};

static void fill32(std::vector<u32>& v, int salt) {
  const int m = sizeof kSpecial32 / sizeof kSpecial32[0];
  for (size_t i = 0; i < v.size(); ++i) {
    const u64 r = next_rand();
    switch ((i + salt) % 4) {
      case 0: v[i] = kSpecial32[(i * 7 + salt * 13 + (r >> 60)) % m]; break;
      case 1: v[i] = (u32)r; break;
      case 2: v[i] = kSpecial32[r % m] ^ ((r >> 32) & 1 ? 0u : 0x80000000u); break;
      default: v[i] = (u32)(r >> 32) & ((r & 8) ? 0xffffffffu : 0xffu); break;
    }
  }
}
static void fill64(std::vector<u64>& v, int salt) {
  const int m = sizeof kSpecial64 / sizeof kSpecial64[0];
  for (size_t i = 0; i < v.size(); ++i) {
    const u64 r = next_rand();
    switch ((i + salt) % 3) {
      case 0: v[i] = kSpecial64[(i * 5 + salt) % m]; break;
      case 1: v[i] = r; break;
      default: v[i] = (r & 1) ? kSpecial64[r % m] : (0x3ff0000000000000ull | (r >> 12)); break;   // normal doubles near 1
    }
  }
}

static u64 hash_bytes(const void* p, size_t n) {
  u64 h = 1469598103934665603ull;
  const unsigned char* c = (const unsigned char*)p;
  for (size_t i = 0; i < n; ++i) h = (h ^ c[i]) * 1099511628211ull;
  return h;
}

template <class T>
static T* dev_copy(const std::vector<T>& h) {
  T* d;
  if (cudaMalloc(&d, h.size() * sizeof(T) + 64) || cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)) return nullptr;
  return d;
}

template <class T>
static int report(const char* name, T* d, size_t count) {
  std::vector<T> h(count);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h.data(), d, count * sizeof(T), cudaMemcpyDeviceToHost));
  std::printf("%-28s %016llx\n", name, (unsigned long long)hash_bytes(h.data(), count * sizeof(T)));
  return 0;
}

template <class T>
static int run_copy(const char* name, int n) {
  std::vector<T> h(n);
  for (int i = 0; i < n; ++i) {
    u64 x = next_rand();
    std::memcpy(&h[i], &x, sizeof(T) < 8 ? sizeof(T) : 8);
  }
  T* din = dev_copy(h);
  T* dout;
  CK(cudaMalloc(&dout, 2 * (size_t)n * sizeof(T) + 64));
  for (int limit : {32, 7, 1, 20})
    for (int stride : {1, 3}) {
      CK(cudaMemset(dout, 0, 2 * (size_t)n * sizeof(T)));
      k_copy<T><<<(n + 127) / 128, 128>>>(din, dout, n, limit, stride, stride == 1 ? 0 : 2);
      char label[64];
      std::snprintf(label, sizeof label, "copy %s n=%d lim=%d st=%d", name, n, limit, stride);
      if (report(label, dout, 2 * (size_t)n)) return 1;
    }
  CK(cudaFree(din));
  CK(cudaFree(dout));
  return 0;
}

template <class T>
static int run_shared(const char* name) {
  const int blocks = 3, threads = 192;
  std::vector<T> h(blocks * threads);
  for (size_t i = 0; i < h.size(); ++i) {
    u64 x = next_rand();
    std::memcpy(&h[i], &x, sizeof(T) < 8 ? sizeof(T) : 8);
  }
  T* din = dev_copy(h);
  T* dout;
  CK(cudaMalloc(&dout, h.size() * sizeof(T) + 64));
  for (int mul : {1, 3, 33, 129, 2}) {
    k_shared<T><<<blocks, threads>>>(din, dout, mul, mul & 3);
    char label[64];
    std::snprintf(label, sizeof label, "shared %s mul=%d", name, mul);
    if (report(label, dout, h.size())) return 1;
  }
  CK(cudaFree(din));
  CK(cudaFree(dout));
  return 0;
}

int main() {
  const int n = 4096 + 77;
  std::vector<u32> a(n), b(n), c(n);
  fill32(a, 0);
  fill32(b, 1);
  fill32(c, 2);
  u32 *da = dev_copy(a), *db = dev_copy(b), *dc = dev_copy(c);
  if (!da || !db || !dc) return 1;
  {
    u32* out;
    CK(cudaMalloc(&out, (size_t)n * kInt * 4));
    for (int threads : {128, 48, 33}) {
      k_int<<<(n + threads - 1) / threads, threads>>>(da, db, dc, out, n, 0x9e3779b9u, 0x80000000u);
      char label[64];
      std::snprintf(label, sizeof label, "int threads=%d", threads);
      if (report(label, out, (size_t)n * kInt)) return 1;
    }
    CK(cudaFree(out));
  }
  {
    u32* out;
    CK(cudaMalloc(&out, (size_t)n * 4));
    for (int threads : {64, 40}) {
      k_branch<<<(n + threads - 1) / threads, threads>>>(da, db, dc, out, n);
      char label[64];
      std::snprintf(label, sizeof label, "branch threads=%d", threads);
      if (report(label, out, n)) return 1;
    }
    CK(cudaFree(out));
  }
  {
    u32* out;
    CK(cudaMalloc(&out, (size_t)n * kFlt * 4));
    for (int threads : {128, 50}) {
      k_float<<<(n + threads - 1) / threads, threads>>>(da, db, dc, out, n, 0x40490fdbu);
      char label[64];
      std::snprintf(label, sizeof label, "float threads=%d", threads);
      if (report(label, out, (size_t)n * kFlt)) return 1;
    }
    CK(cudaFree(out));
  }
  {
    std::vector<u64> xa(n), xb(n), xc(n);
    fill64(xa, 0);
    fill64(xb, 1);
    fill64(xc, 2);
    u64 *dxa = dev_copy(xa), *dxb = dev_copy(xb), *dxc = dev_copy(xc), *out;
    CK(cudaMalloc(&out, (size_t)n * kDbl * 8));
    for (int threads : {64, 37}) {
      k_double<<<(n + threads - 1) / threads, threads>>>(dxa, dxb, dxc, out, n);
      char label[64];
      std::snprintf(label, sizeof label, "double threads=%d", threads);
      if (report(label, out, (size_t)n * kDbl)) return 1;
    }
    CK(cudaFree(dxa)); CK(cudaFree(dxb)); CK(cudaFree(dxc)); CK(cudaFree(out));
  }
  for (int m : {1, 31, 32, 33, 1000, 4096 + 5}) {
    if (run_copy<unsigned char>("u8", m) || run_copy<unsigned short>("u16", m) || run_copy<u32>("u32", m) ||
        run_copy<u64>("u64", m) || run_copy<Q>("q16", m))
      return 1;
  }
  if (run_shared<unsigned char>("u8") || run_shared<unsigned short>("u16") || run_shared<u32>("u32") ||
      run_shared<u64>("u64") || run_shared<Q>("q16"))
    return 1;
  {
    u32* out;
    CK(cudaMalloc(&out, (size_t)n * 4));
    for (int threads : {64, 48}) {
      k_local<<<(n + threads - 1) / threads, threads>>>(da, db, out, n);
      char label[64];
      std::snprintf(label, sizeof label, "local threads=%d", threads);
      if (report(label, out, n)) return 1;
    }
    u32 table[64];
    for (int i = 0; i < 64; ++i) table[i] = (u32)next_rand();
    CK(cudaMemcpyToSymbol(kc, table, sizeof table));
    k_const<<<(n + 127) / 128, 128>>>(da, out, n, 17u, 0xabcdu, 99u);
    if (report("const", out, n)) return 1;
    CK(cudaFree(out));
  }
  {
    u32* g;
    CK(cudaMalloc(&g, 1024 * 4));
    CK(cudaMemset(g, 1, 1024 * 4));
    k_generic<<<4, 256>>>(g, da, n);
    if (report("generic", g, 1024)) return 1;
    CK(cudaFree(g));
  }
  CK(cudaFree(da)); CK(cudaFree(db)); CK(cudaFree(dc));
  std::printf("done\n");
  return 0;
}
