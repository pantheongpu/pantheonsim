// Hopper's DPX intrinsics (CUDA Math API, "SIMD intrinsics"): __vimax/__vimin, the three-operand
// __vimax3/__vimin3, the fused __viaddmax/__viaddmin and the __vibmax/__vibmin that also give the
// choice, in s32, u32, s16x2 and u16x2, with and without _relu. They are the dynamic-programming
// inner loops of StringZilla's edit distances and of every Smith-Waterman kernel.
//
// NVIDIA's own header defines each of them for the host too (crt/device_functions.hpp), as the
// plain C arithmetic the documentation describes, so this program has the oracle in the toolkit:
// every function runs on a spread of operand triples on the device and on the host, and the two
// must agree bit for bit. Built for sm_90 the device side is PTX's max/min(.relu) and
// add.s16x2 (and, from SASS, VIMNMX / VIADDMNMX / VIMNMX3 with .RELU and the packed lanes);
// built for an older architecture it is the emulation the header falls back on. Also the PTX
// ISA's scalar and packed min/max .relu written out in both modifier orders, against a host model.
//
// Prints PASS on the last line.
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---- the PTX min/max with .relu, against a host model ----------------------------------------
__host__ __device__ static unsigned relu_s32(unsigned a, unsigned b, bool mx) {
  int x = static_cast<int>(a), y = static_cast<int>(b);
  int r = mx ? (x > y ? x : y) : (x < y ? x : y);
  return static_cast<unsigned>(r < 0 ? 0 : r);
}
// Per 16-bit half, signed or unsigned, with or without .relu (signed only).
__host__ __device__ static unsigned lanes16(unsigned a, unsigned b, bool mx, bool sgn, bool relu) {
  unsigned out = 0;
  for (int h = 0; h < 2; ++h) {
    const unsigned ua = (a >> (16 * h)) & 0xffff, ub = (b >> (16 * h)) & 0xffff;
    int x = sgn ? static_cast<int>(static_cast<short>(ua)) : static_cast<int>(ua);
    int y = sgn ? static_cast<int>(static_cast<short>(ub)) : static_cast<int>(ub);
    int r = mx ? (x > y ? x : y) : (x < y ? x : y);
    if (relu && r < 0) r = 0;
    out |= (static_cast<unsigned>(r) & 0xffffu) << (16 * h);
  }
  return out;
}
__host__ __device__ static unsigned relu_s16x2(unsigned a, unsigned b, bool mx) { return lanes16(a, b, mx, true, true); }
__host__ __device__ static unsigned s16x2_plain(unsigned a, unsigned b, bool mx) { return lanes16(a, b, mx, true, false); }
__host__ __device__ static unsigned u16x2_plain(unsigned a, unsigned b, bool mx) { return lanes16(a, b, mx, false, false); }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define PTX_RELU(name, ptx, a, b, model) \
  __device__ static unsigned name(unsigned x, unsigned y) { unsigned r; asm volatile(ptx : "=r"(r) : "r"(x), "r"(y)); return r; }
#else
#define PTX_RELU(name, ptx, a, b, model) \
  __host__ __device__ static unsigned name(unsigned x, unsigned y) { return model(x, y, a); }
#endif
PTX_RELU(ptx_max_relu_s32, "max.relu.s32 %0, %1, %2;", true, 0, relu_s32)
PTX_RELU(ptx_max_s32_relu, "max.s32.relu %0, %1, %2;", true, 0, relu_s32)
PTX_RELU(ptx_min_relu_s32, "min.relu.s32 %0, %1, %2;", false, 0, relu_s32)
PTX_RELU(ptx_min_s32_relu, "min.s32.relu %0, %1, %2;", false, 0, relu_s32)
PTX_RELU(ptx_max_relu_s16x2, "max.relu.s16x2 %0, %1, %2;", true, 0, relu_s16x2)
PTX_RELU(ptx_max_s16x2_relu, "max.s16x2.relu %0, %1, %2;", true, 0, relu_s16x2)
PTX_RELU(ptx_min_relu_s16x2, "min.relu.s16x2 %0, %1, %2;", false, 0, relu_s16x2)
PTX_RELU(ptx_min_s16x2_relu, "min.s16x2.relu %0, %1, %2;", false, 0, relu_s16x2)
PTX_RELU(ptx_max_s16x2, "max.s16x2 %0, %1, %2;", true, 0, s16x2_plain)
PTX_RELU(ptx_min_s16x2, "min.s16x2 %0, %1, %2;", false, 0, s16x2_plain)
PTX_RELU(ptx_max_u16x2, "max.u16x2 %0, %1, %2;", true, 0, u16x2_plain)
PTX_RELU(ptx_min_u16x2, "min.u16x2 %0, %1, %2;", false, 0, u16x2_plain)

// The plain two-operand forms are the C++ min and max; the packed ones are PTX's, below.
__host__ __device__ static int s32_max(int a, int b) { return a > b ? a : b; }
__host__ __device__ static int s32_min(int a, int b) { return a < b ? a : b; }
__host__ __device__ static unsigned u32_max(unsigned a, unsigned b) { return a > b ? a : b; }
__host__ __device__ static unsigned u32_min(unsigned a, unsigned b) { return a < b ? a : b; }

// ---- the intrinsics ---------------------------------------------------------------------------
// S2/U2: two operands (signed ints / unsigned, which also carry the packed types); S3/U3: three;
// SB/UB: the choice as a predicate; B16: the choice per half; M: a PTX form, against the model.
#define DPX_LIST(S2, U2, S3, U3, SB, UB, B16, M)                                                                  \
  S2(s32_max) S2(s32_min) U2(u32_max) U2(u32_min) S2(__vimax_s32_relu) S2(__vimin_s32_relu)                       \
  U2(__vimax_s16x2_relu) U2(__vimin_s16x2_relu)                                                                   \
  S3(__vimax3_s32) S3(__vimin3_s32) S3(__vimax3_s32_relu) S3(__vimin3_s32_relu)                                   \
  U3(__vimax3_u32) U3(__vimin3_u32)                                                                               \
  U3(__vimax3_s16x2) U3(__vimin3_s16x2) U3(__vimax3_s16x2_relu) U3(__vimin3_s16x2_relu)                           \
  U3(__vimax3_u16x2) U3(__vimin3_u16x2)                                                                           \
  S3(__viaddmax_s32) S3(__viaddmin_s32) S3(__viaddmax_s32_relu) S3(__viaddmin_s32_relu)                           \
  U3(__viaddmax_u32) U3(__viaddmin_u32)                                                                           \
  U3(__viaddmax_s16x2) U3(__viaddmin_s16x2) U3(__viaddmax_s16x2_relu) U3(__viaddmin_s16x2_relu)                   \
  U3(__viaddmax_u16x2) U3(__viaddmin_u16x2)                                                                       \
  SB(__vibmax_s32) SB(__vibmin_s32) UB(__vibmax_u32) UB(__vibmin_u32)                                             \
  B16(__vibmax_s16x2) B16(__vibmin_s16x2) B16(__vibmax_u16x2) B16(__vibmin_u16x2)                                 \
  M(ptx_max_relu_s32) M(ptx_max_s32_relu) M(ptx_min_relu_s32) M(ptx_min_s32_relu)                                 \
  M(ptx_max_relu_s16x2) M(ptx_max_s16x2_relu) M(ptx_min_relu_s16x2) M(ptx_min_s16x2_relu)           \
  M(ptx_max_s16x2) M(ptx_min_s16x2) M(ptx_max_u16x2) M(ptx_min_u16x2)

// One function, one operand triple -> two words: the result and the predicates.
#define CASE_S2(f) case __COUNTER__ - kBase: o0 = (unsigned)f((int)a, (int)b); break;
#define CASE_U2(f) case __COUNTER__ - kBase: o0 = (unsigned)f(a, b); break;
#define CASE_S3(f) case __COUNTER__ - kBase: o0 = (unsigned)f((int)a, (int)b, (int)c); break;
#define CASE_U3(f) case __COUNTER__ - kBase: o0 = (unsigned)f(a, b, c); break;
#define CASE_SB(f) case __COUNTER__ - kBase: { bool p; o0 = (unsigned)f((int)a, (int)b, &p); o1 = p; } break;
#define CASE_UB(f) case __COUNTER__ - kBase: { bool p; o0 = (unsigned)f(a, b, &p); o1 = p; } break;
#define CASE_B16(f) case __COUNTER__ - kBase: { bool h, l; o0 = (unsigned)f(a, b, &h, &l); o1 = (h ? 2u : 0u) | (l ? 1u : 0u); } break;
#define CASE_M(f) case __COUNTER__ - kBase: o0 = f(a, b); break;
#define NAME(f) #f,
#define COUNT(f) +1

static const char* const kNames[] = {DPX_LIST(NAME, NAME, NAME, NAME, NAME, NAME, NAME, NAME)};
constexpr int kFuncs = 0 DPX_LIST(COUNT, COUNT, COUNT, COUNT, COUNT, COUNT, COUNT, COUNT);
constexpr int kBase = __COUNTER__ + 1;

__host__ __device__ static void eval(int f, unsigned a, unsigned b, unsigned c, unsigned* out) {
  unsigned o0 = 0, o1 = 0;
  switch (f) {
    DPX_LIST(CASE_S2, CASE_U2, CASE_S3, CASE_U3, CASE_SB, CASE_UB, CASE_B16, CASE_M)
    default: break;
  }
  out[0] = o0;
  out[1] = o1;
}

__global__ void kernel(const unsigned* in, unsigned* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  for (int f = 0; f < kFuncs; ++f) eval(f, in[3 * i], in[3 * i + 1], in[3 * i + 2], &out[2 * (size_t(f) * n + i)]);
}

int main() {
  // Operand values that break arithmetic: the ends of each lane's range, zero and the sign
  // boundary of the lanes and of the word, a lane holding one and another a different one.
  static const unsigned edge[] = {0u,          1u,          2u,          0x7fffu,     0x8000u,
                                  0xffffu,     0x00018000u, 0x7fff8000u, 0x8000ffffu, 0xfffe0001u,
                                  0x7fffffffu, 0x80000000u, 0x80000001u, 0xfffffffeu, 0xffffffffu,
                                  0x00010001u};
  constexpr int kEdge = sizeof edge / sizeof edge[0];
  constexpr int kRandom = 1500;
  constexpr int n = kEdge * kEdge * kEdge + kRandom;
  static unsigned h[3 * n];
  int t = 0;
  for (int x = 0; x < kEdge; ++x)
    for (int y = 0; y < kEdge; ++y)
      for (int z = 0; z < kEdge; ++z, ++t) { h[3 * t] = edge[x]; h[3 * t + 1] = edge[y]; h[3 * t + 2] = edge[z]; }
  unsigned long long s = 0x9e3779b97f4a7c15ull;
  auto next = [&] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return static_cast<unsigned>(s >> 16); };
  for (; t < n; ++t) {
    // Half of the random operands have small lanes, where a sum, a minimum and a maximum are
    // all in play rather than saturated at one end.
    const bool small = t & 1;
    for (int k = 0; k < 3; ++k) {
      unsigned v = next();
      if (small) v = ((v & 0x7f) - 0x40) & 0xffff | ((((v >> 16) & 0x7f) - 0x40) << 16);
      h[3 * t + k] = v;
    }
  }
  unsigned *din, *dout;
  cudaMalloc(&din, sizeof h);
  cudaMalloc(&dout, size_t(kFuncs) * n * 2 * sizeof(unsigned));
  cudaMemcpy(din, h, sizeof h, cudaMemcpyHostToDevice);
  kernel<<<(n + 127) / 128, 128>>>(din, dout, n);
  if (cudaDeviceSynchronize() != cudaSuccess) { printf("the kernel failed: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
  static unsigned d[size_t(kFuncs) * n * 2];
  cudaMemcpy(d, dout, sizeof d, cudaMemcpyDeviceToHost);
  int bad = 0, badfuncs = 0;
  for (int f = 0; f < kFuncs; ++f) {
    int wrong = 0;
    for (int i = 0; i < n; ++i) {
      unsigned want[2];
      eval(f, h[3 * i], h[3 * i + 1], h[3 * i + 2], want);
      const unsigned* got = &d[2 * (size_t(f) * n + i)];
      if (got[0] != want[0] || got[1] != want[1]) {
        if (wrong++ < 2)
          printf("%s(%08x, %08x, %08x): got %08x/%u, the host gives %08x/%u\n", kNames[f], h[3 * i], h[3 * i + 1],
                 h[3 * i + 2], got[0], got[1], want[0], want[1]);
      }
    }
    if (wrong) { ++badfuncs; bad += wrong; }
  }
  printf("%d DPX forms over %d operand triples, %d forms differ\n", kFuncs, n, badfuncs);
  puts(bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
