// The AMD engine's probe workloads: the instruction classes bench_amd.hip does not exercise. A second program (its
// own cache key in tools/bench-amd/bench.py, so the cached bench_amd binaries stay valid), `bench_amd_probe <workload>
// <quick|full>`, one small self-checking kernel per class: DPP, ds_swizzle / ds_permute / ds_bpermute, v_permlane,
// readlane / readfirstlane / writelane, 16-bit and packed 16-bit math, sub-dword (SDWA) forms, dual-issue (VOPD)
// pairs, compare and select shapes, 64-bit integer and double ops, scalar loads of every width and s_buffer_load.
// The host reference is bit for bit (integer-valued data wherever float rounding could differ), and the last line is
// `PASS` only when every lane matched. Built once per target (gfx942, gfx1030, gfx1100, gfx1201).
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
#define LAUNCH(...)                  \
  do {                               \
    hipLaunchKernelGGL(__VA_ARGS__); \
    HC(hipGetLastError());           \
  } while (0)

static bool g_ok = true;
static void expect_eq(const char* what, const void* a, const void* b, size_t bytes) {
  if (std::memcmp(a, b, bytes) != 0) {
    const uint32_t* x = static_cast<const uint32_t*>(a);
    const uint32_t* y = static_cast<const uint32_t*>(b);
    size_t i = 0;
    while (i * 4 < bytes && x[i] == y[i]) ++i;
    std::fprintf(stderr, "MISMATCH in %s (word %zu: got %08x want %08x)\n", what, i, i * 4 < bytes ? x[i] : 0, i * 4 < bytes ? y[i] : 0);
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
static uint32_t scale(const std::string& size, uint32_t quick, uint32_t full) { return size == "quick" ? quick : full; }
static int g_wave = 64;
static char g_arch[256] = "";
static bool has_arch(const char* what) { return std::string(g_arch).find(what) != std::string::npos; }

__host__ __device__ static inline uint32_t rotl32(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }
__host__ __device__ static inline uint64_t rotl64(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

// ---------------------------------------------------------------- DPP

// One DPP move of `src` into a lane whose old value is `old`: the control, row and bank masks and bound_ctrl are
// constants of the instruction.
template <int CTRL, int RM, int BM, int BC>
__device__ static inline uint32_t dpp(uint32_t old, uint32_t src) {
  return (uint32_t)__builtin_amdgcn_update_dpp((int)old, (int)src, CTRL, RM, BM, BC != 0);
}
// The host's twin: lane `lane` of a wave of ws lanes.
static uint32_t dpp_host(const uint32_t* old, const uint32_t* src, int lane, int ctrl, int rm, int bm, int bc, int ws) {
  const int row = lane >> 4, r = lane & 15, bank = r >> 2;
  if (!((rm >> row) & 1) || !((bm >> bank) & 1)) return old[lane];
  int sl = -1;
  if (ctrl <= 0xff) sl = (lane & ~3) | ((ctrl >> (2 * (lane & 3))) & 3);
  else if (ctrl >= 0x101 && ctrl <= 0x10f) sl = r + (ctrl & 15) < 16 ? lane + (ctrl & 15) : -1;         // row_shl
  else if (ctrl >= 0x111 && ctrl <= 0x11f) sl = r - (ctrl & 15) >= 0 ? lane - (ctrl & 15) : -1;         // row_shr
  else if (ctrl >= 0x121 && ctrl <= 0x12f) sl = (row << 4) | ((r - (ctrl & 15)) & 15);                  // row_ror
  else if (ctrl == 0x130) sl = lane + 1 < ws ? lane + 1 : -1;                                           // wave_shl:1
  else if (ctrl == 0x134) sl = (lane + 1) % ws;                                                         // wave_rol:1
  else if (ctrl == 0x138) sl = lane - 1 >= 0 ? lane - 1 : -1;                                           // wave_shr:1
  else if (ctrl == 0x13c) sl = (lane + ws - 1) % ws;                                                    // wave_ror:1
  else if (ctrl == 0x140) sl = (row << 4) | (15 - r);                                                   // row_mirror
  else if (ctrl == 0x141) sl = (row << 4) | (r & 8) | (7 - (r & 7));                                    // row_half_mirror
  else if (ctrl == 0x142) sl = row > 0 ? (row - 1) * 16 + 15 : -1;                                      // row_bcast:15
  else if (ctrl == 0x143) sl = row >= 2 ? 31 : -1;                                                      // row_bcast:31
  else if (ctrl >= 0x150 && ctrl <= 0x15f) sl = (row << 4) | (ctrl & 15);                               // row_share
  else if (ctrl >= 0x160 && ctrl <= 0x16f) sl = (row << 4) | (r ^ (ctrl & 15));                         // row_xmask
  if (sl < 0 || sl >= ws) return bc ? 0u : old[lane];
  return src[sl];
}

// The controls both families (GCN/CDNA DPP and RDNA's DPP16) have, with bound_ctrl and the masks set both ways.
#define DPP_COMMON(X)                                                                                              \
  X(0x1B, 0xf, 0xf, 0) X(0xB1, 0xf, 0xf, 0) X(0x111, 0xf, 0xf, 1) X(0x113, 0xf, 0xf, 0) X(0x121, 0xf, 0xf, 0)    \
  X(0x12D, 0xf, 0xf, 0) X(0x141, 0xf, 0xf, 0) X(0x140, 0xf, 0xf, 0) X(0x105, 0xa, 0xf, 0) X(0x10E, 0xf, 0x5, 1)  \
  X(0x4E, 0xf, 0xf, 0) X(0x101, 0xf, 0xf, 1)
// The wave-wide forms only GCN/CDNA has: shifts and rotates by one lane across the whole wave, and the row broadcasts
// a reduction uses.
#define DPP_ARCH9(X)                                                                                         \
  X(0x138, 0xf, 0xf, 1) X(0x130, 0xf, 0xf, 0) X(0x13C, 0xf, 0xf, 0) X(0x134, 0xf, 0xf, 0) X(0x142, 0xa, 0xf, 0) \
  X(0x143, 0xc, 0xf, 0)
// RDNA's: a row's lane shared with the row, or xor-ed with a mask.
#define DPP_ARCHR(X)                                                                                         \
  X(0x153, 0xf, 0xf, 0) X(0x15C, 0xf, 0xf, 1) X(0x161, 0xf, 0xf, 0) X(0x16F, 0xf, 0xf, 0) X(0x168, 0xf, 0xf, 0) \
  X(0x157, 0x5, 0xf, 0)
template <int WHICH>
__global__ void k_dpp(uint32_t* out, uint32_t iters) {
  uint32_t s = (blockIdx.x * blockDim.x + threadIdx.x) * 2654435761u + 7;
  for (uint32_t i = 0; i < iters; ++i) {
    if (WHICH == 0) {
#define STEP(c, rm, bm, bc) s = dpp<c, rm, bm, bc>(s, s) * 33u + i;
      DPP_COMMON(STEP)
#undef STEP
    } else {
#define STEP(c, rm, bm, bc) s = dpp<c, rm, bm, bc>(s, s) * 33u + i;
#if defined(__GFX9__)
      DPP_ARCH9(STEP)
#else
      DPP_ARCHR(STEP)
#endif
#undef STEP
    }
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
static void run_dpp(const std::string& size, int which) {
  const uint32_t blocks = 32, threads = 256, iters = scale(size, 20, 300), n = blocks * threads;
  uint32_t* d;
  HC(hipMalloc(&d, n * 4));
  Phase p;
  if (which == 0) LAUNCH(k_dpp<0>, dim3(blocks), dim3(threads), 0, 0, d, iters);
  else LAUNCH(k_dpp<1>, dim3(blocks), dim3(threads), 0, 0, d, iters);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> got = dev_read(d, n), ref(n);
  const int ws = g_wave;
  for (uint32_t base = 0; base < n; base += ws) {
    std::vector<uint32_t> s(ws), t(ws);
    for (int l = 0; l < ws; ++l) s[l] = (base + l) * 2654435761u + 7;
    for (uint32_t i = 0; i < iters; ++i) {
      const auto step = [&](int c, int rm, int bm, int bc) {
        for (int l = 0; l < ws; ++l) t[l] = dpp_host(s.data(), s.data(), l, c, rm, bm, bc, ws) * 33u + i;
        s = t;
      };
#define STEP(c, rm, bm, bc) step(c, rm, bm, bc);
      if (which == 0) {
        DPP_COMMON(STEP)
      } else if (has_arch("gfx9")) {
        DPP_ARCH9(STEP)
      } else {
        DPP_ARCHR(STEP)
      }
#undef STEP
    }
    for (int l = 0; l < ws; ++l) ref[base + l] = s[l];
  }
  expect_eq(which == 0 ? "dpp" : "dpp_arch", got.data(), ref.data(), n * 4);
}

// DPP folded into arithmetic: v_add_u32_dpp, v_max_u32_dpp, v_and_b32_dpp, v_mul_f32_dpp (the compiler combines a move
// into the instruction that uses it), and the row-wise inclusive scan (+ row_shr:1,2,3 then 4, 8 with bank masks).
__global__ void k_dpp_scan(uint32_t* out, uint32_t iters) {
  uint32_t v = (blockIdx.x * blockDim.x + threadIdx.x) * 2654435761u + 11;
  for (uint32_t i = 0; i < iters; ++i) {
    uint32_t x = (v >> 20) + i;
    x += dpp<0x111, 0xf, 0xf, 1>(0, x);
    x += dpp<0x112, 0xf, 0xf, 1>(0, x);
    x += dpp<0x114, 0xf, 0xf, 1>(0, x);
    x += dpp<0x118, 0xf, 0xf, 1>(0, x);
    uint32_t m = x;
    m = max(m, dpp<0x141, 0xf, 0xf, 0>(m, x));
    m &= ~dpp<0xB1, 0xf, 0xf, 1>(0, x) | 0x00ffffffu;
    v = rotl32(v, 5) ^ x ^ m;
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
static void run_dpp_scan(const std::string& size) {
  const uint32_t blocks = 32, threads = 256, iters = scale(size, 20, 400), n = blocks * threads;
  uint32_t* d;
  HC(hipMalloc(&d, n * 4));
  Phase p;
  LAUNCH(k_dpp_scan, dim3(blocks), dim3(threads), 0, 0, d, iters);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> got = dev_read(d, n), ref(n);
  const int ws = g_wave;
  for (uint32_t base = 0; base < n; base += ws) {
    std::vector<uint32_t> v(ws), x(ws), y(ws), m(ws), z(ws);
    for (int l = 0; l < ws; ++l) v[l] = (base + l) * 2654435761u + 11;
    for (uint32_t i = 0; i < iters; ++i) {
      for (int l = 0; l < ws; ++l) x[l] = (v[l] >> 20) + i;
      for (int c : {0x111, 0x112, 0x114, 0x118}) {
        for (int l = 0; l < ws; ++l) z[l] = 0;
        for (int l = 0; l < ws; ++l) y[l] = x[l] + dpp_host(z.data(), x.data(), l, c, 0xf, 0xf, 1, ws);
        x = y;
      }
      m = x;
      for (int l = 0; l < ws; ++l) y[l] = std::max(m[l], dpp_host(m.data(), x.data(), l, 0x141, 0xf, 0xf, 0, ws));
      m = y;
      for (int l = 0; l < ws; ++l) z[l] = 0;
      for (int l = 0; l < ws; ++l) m[l] &= ~dpp_host(z.data(), x.data(), l, 0xB1, 0xf, 0xf, 1, ws) | 0x00ffffffu;
      for (int l = 0; l < ws; ++l) v[l] = rotl32(v[l], 5) ^ x[l] ^ m[l];
    }
    for (int l = 0; l < ws; ++l) ref[base + l] = v[l];
  }
  expect_eq("dpp_scan", got.data(), ref.data(), n * 4);
}

// ---------------------------------------------------------------- ds_swizzle, ds_permute, ds_bpermute

__global__ void k_perm(uint32_t* out, uint32_t iters) {
  const uint32_t lane = threadIdx.x % warpSize;
  uint32_t v = (blockIdx.x * blockDim.x + threadIdx.x) * 2654435761u + 3;
  uint32_t acc = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    // bit-mask mode: and 0x1f, or 0, xor 0x03 / and 0x1c, or 0x01, xor 0x08;  quad-perm mode 0x1B and 0xE4 (identity)
    const uint32_t a = __builtin_amdgcn_ds_swizzle((int)v, 0x0C1F);
    const uint32_t b = __builtin_amdgcn_ds_swizzle((int)v, 0x001c | (0x01 << 5) | (0x08 << 10));
    const uint32_t c = __builtin_amdgcn_ds_swizzle((int)v, 0x8000 | 0x1B);
    const uint32_t d = __builtin_amdgcn_ds_bpermute((int)(((lane * 5 + i) % warpSize) * 4), (int)v);
    const uint32_t e = __builtin_amdgcn_ds_permute((int)(((lane * 3 + i) % warpSize) * 4), (int)v);
    acc += a ^ c;
    v = rotl32(v ^ b, 7) + d - e + i;
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = v + acc;
}
static void run_perm(const std::string& size) {
  const uint32_t blocks = 32, threads = 256, iters = scale(size, 40, 800), n = blocks * threads;
  uint32_t* d;
  HC(hipMalloc(&d, n * 4));
  Phase p;
  LAUNCH(k_perm, dim3(blocks), dim3(threads), 0, 0, d, iters);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> got = dev_read(d, n), ref(n);
  const int ws = g_wave;
  for (uint32_t base = 0; base < n; base += ws) {
    std::vector<uint32_t> v(ws), nv(ws), acc(ws, 0), dd(ws), ee(ws);
    for (int l = 0; l < ws; ++l) v[l] = (base + l) * 2654435761u + 3;
    for (uint32_t i = 0; i < iters; ++i) {
      for (int l = 0; l < ws; ++l) {
        const int g = l & ~31;
        const int la = g + ((((l & 31) & 0x1f) | 0) ^ 0x03);                              // 0x041f | 0x0c00: and 0x1f, or 0x00, xor 0x03
        const int lb = g + ((((l & 31) & 0x1c) | 0x01) ^ 0x08);
        const int lc = (l & ~3) | ((0x1B >> (2 * (l & 3))) & 3);
        const uint32_t a = v[la], b = v[lb], c = v[lc];
        dd[l] = v[(l * 5 + i) % ws];
        ee[(l * 3 + i) % ws] = v[l];                                                       // ds_permute: lane l writes lane ((3l+i) % ws)
        acc[l] += a ^ c;
        nv[l] = b;
      }
      for (int l = 0; l < ws; ++l) nv[l] = rotl32(v[l] ^ nv[l], 7) + dd[l] - ee[l] + i;
      v = nv;
    }
    for (int l = 0; l < ws; ++l) ref[base + l] = v[l] + acc[l];
  }
  expect_eq("perm", got.data(), ref.data(), n * 4);
}

// ---------------------------------------------------------------- v_permlane16 / v_permlanex16 (RDNA)

#if defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__)
#define HAVE_PERMLANE 1
#endif
__global__ void k_permlane(uint32_t* out, uint32_t iters) {
  uint32_t v = (blockIdx.x * blockDim.x + threadIdx.x) * 2654435761u + 5;
#ifdef HAVE_PERMLANE
  for (uint32_t i = 0; i < iters; ++i) {
    // selects: lane j of a row reads lane nibble j: 0x76543210 / 0xFEDCBA98 is the identity; these reverse and rotate.
    const uint32_t a = __builtin_amdgcn_permlane16(v, v, 0x01234567u, 0x89ABCDEFu, false, false);
    const uint32_t b = __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xFEDCBA98u, false, false);
    const uint32_t c = __builtin_amdgcn_permlane16(0, v, 0x32107654u, 0xBA98FEDCu, false, true);
    v = rotl32(v ^ a, 3) + b - c + i;
  }
#endif
  out[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
static uint32_t nibble(uint32_t lo, uint32_t hi, int j) { return ((j < 8 ? lo : hi) >> (4 * (j & 7))) & 15; }
static void run_permlane(const std::string& size) {
  const uint32_t blocks = 32, threads = 256, iters = scale(size, 40, 800), n = blocks * threads;
  if (has_arch("gfx9")) {   // (the macro is the device pass's: the host decides by the architecture's name)
    std::printf("SKIP no v_permlane16 for %s\n", g_arch);
    return;
  }
  uint32_t* d;
  HC(hipMalloc(&d, n * 4));
  Phase p;
  LAUNCH(k_permlane, dim3(blocks), dim3(threads), 0, 0, d, iters);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> got = dev_read(d, n), ref(n);
  const int ws = g_wave;
  for (uint32_t base = 0; base < n; base += ws) {
    std::vector<uint32_t> v(ws), nv(ws);
    for (int l = 0; l < ws; ++l) v[l] = (base + l) * 2654435761u + 5;
    for (uint32_t i = 0; i < iters; ++i) {
      for (int l = 0; l < ws; ++l) {
        const int row = l >> 4, j = l & 15, grp = row & ~1;
        const uint32_t a = v[(row << 4) | nibble(0x01234567u, 0x89ABCDEFu, j)];
        const uint32_t b = v[((row ^ 1) << 4) | nibble(0x76543210u, 0xFEDCBA98u, j)];
        const uint32_t c = v[(row << 4) | nibble(0x32107654u, 0xBA98FEDCu, j)];
        (void)grp;
        nv[l] = rotl32(v[l] ^ a, 3) + b - c + i;
      }
      v = nv;
    }
    for (int l = 0; l < ws; ++l) ref[base + l] = v[l];
  }
  expect_eq("permlane", got.data(), ref.data(), n * 4);
}

// ---------------------------------------------------------------- readlane, readfirstlane, writelane

// (the lane of v_writelane_b32 is an inline constant here: GFX9 reads one scalar operand a VOP3 instruction.)
#define LANE_STEP(k)                                                                         \
  {                                                                                          \
    const uint32_t i = o + k;                                                                \
    const uint32_t f = __builtin_amdgcn_readfirstlane(v);                                    \
    const uint32_t r = __builtin_amdgcn_readlane((int)v, (i * 7) & 31);                      \
    v = v * 3u + i;                                                                          \
    const uint32_t val = __builtin_amdgcn_readfirstlane(f + r);                              \
    asm volatile("v_writelane_b32 %0, %1, %2" : "+v"(v) : "s"(val), "i"(k));                 \
    acc += v ^ f;                                                                            \
  }
__global__ void k_lane(uint32_t* out, uint32_t iters) {
  uint32_t v = (blockIdx.x * blockDim.x + threadIdx.x) * 2654435761u + 13;
  uint32_t acc = 0;
  for (uint32_t o = 0; o < iters; o += 16) {
    LANE_STEP(0) LANE_STEP(1) LANE_STEP(2) LANE_STEP(3) LANE_STEP(4) LANE_STEP(5) LANE_STEP(6) LANE_STEP(7)
    LANE_STEP(8) LANE_STEP(9) LANE_STEP(10) LANE_STEP(11) LANE_STEP(12) LANE_STEP(13) LANE_STEP(14) LANE_STEP(15)
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = v + acc;
}
static void run_lane(const std::string& size) {
  const uint32_t blocks = 32, threads = 256, iters = scale(size, 96, 2000), n = blocks * threads;
  uint32_t* d;
  HC(hipMalloc(&d, n * 4));
  Phase p;
  LAUNCH(k_lane, dim3(blocks), dim3(threads), 0, 0, d, iters);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> got = dev_read(d, n), ref(n);
  const int ws = g_wave;
  for (uint32_t base = 0; base < n; base += ws) {
    std::vector<uint32_t> v(ws), acc(ws, 0);
    for (int l = 0; l < ws; ++l) v[l] = (base + l) * 2654435761u + 13;
    for (uint32_t i = 0; i < iters; ++i) {
      const uint32_t f = v[0], r = v[(i * 7) & 31];
      for (int l = 0; l < ws; ++l) v[l] = v[l] * 3u + i;
      v[i & 15] = f + r;
      for (int l = 0; l < ws; ++l) acc[l] += v[l] ^ f;
    }
    for (int l = 0; l < ws; ++l) ref[base + l] = v[l] + acc[l];
  }
  expect_eq("lane", got.data(), ref.data(), n * 4);
}

// ---------------------------------------------------------------- 16-bit and packed 16-bit math

typedef uint16_t u16x2 __attribute__((ext_vector_type(2)));
typedef int16_t i16x2 __attribute__((ext_vector_type(2)));
typedef _Float16 f16x2 __attribute__((ext_vector_type(2)));
__host__ __device__ static inline u16x2 pk_step(u16x2 x, u16x2 y, uint32_t i) {
  x = x * y + (u16x2){(uint16_t)i, (uint16_t)(i >> 3)};             // v_pk_mad_u16
  x = x ^ (x >> 3);                                                 // v_pk_lshrrev_b16, xor
  x = x + (y << 1);                                                 // v_pk_lshlrev_b16, v_pk_add_u16
  x = __builtin_elementwise_max(x, y) - __builtin_elementwise_min(x ^ y, y);   // v_pk_max_u16, v_pk_min_u16, v_pk_sub_u16
  return x;
}
__global__ void k_pk16(uint32_t* out, uint32_t iters) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  u16x2 x = (u16x2){(uint16_t)(t * 31u + 1), (uint16_t)(t * 17u + 5)};
  u16x2 y = (u16x2){(uint16_t)(t | 1), (uint16_t)((t >> 2) | 1)};
  // scalar 16-bit: v_add_u16, v_mad_u16, v_lshlrev_b16, v_max_i16, v_ashrrev_i16 with the halves picked by op_sel
  int16_t s = (int16_t)(t * 7u);
  for (uint32_t i = 0; i < iters; ++i) {
    x = pk_step(x, y, i);
    uint16_t lo = x[0], hi = x[1];
    lo = (uint16_t)((uint32_t)lo * hi + (uint16_t)i);
    hi = (uint16_t)((hi << 3) ^ (lo >> 2));
    { const int16_t u = (int16_t)(s + (int16_t)hi), w = (int16_t)(s >> 3); s = u > w ? u : w; }
    x = (u16x2){hi, (uint16_t)(lo + (uint16_t)s)};
    y = y + (u16x2){(uint16_t)3, (uint16_t)5};
  }
  out[t] = (uint32_t)x[0] | (((uint32_t)x[1] << 16) ^ ((uint32_t)(uint16_t)s << 8));
}
static void run_pk16(const std::string& size) {
  const uint32_t blocks = 32, threads = 256, iters = scale(size, 100, 2000), n = blocks * threads;
  uint32_t* d;
  HC(hipMalloc(&d, n * 4));
  Phase p;
  LAUNCH(k_pk16, dim3(blocks), dim3(threads), 0, 0, d, iters);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> got = dev_read(d, n), ref(n);
  for (uint32_t t = 0; t < n; ++t) {
    u16x2 x = (u16x2){(uint16_t)(t * 31u + 1), (uint16_t)(t * 17u + 5)};
    u16x2 y = (u16x2){(uint16_t)(t | 1), (uint16_t)((t >> 2) | 1)};
    int16_t s = (int16_t)(t * 7u);
    for (uint32_t i = 0; i < iters; ++i) {
      x = pk_step(x, y, i);
      uint16_t lo = x[0], hi = x[1];
      lo = (uint16_t)((uint32_t)lo * hi + (uint16_t)i);
      hi = (uint16_t)((hi << 3) ^ (lo >> 2));
      { const int16_t u = (int16_t)(s + (int16_t)hi), w = (int16_t)(s >> 3); s = u > w ? u : w; }
      x = (u16x2){hi, (uint16_t)(lo + (uint16_t)s)};
      y = y + (u16x2){(uint16_t)3, (uint16_t)5};
    }
    ref[t] = (uint32_t)x[0] | (((uint32_t)x[1] << 16) ^ ((uint32_t)(uint16_t)s << 8));
  }
  expect_eq("pk16", got.data(), ref.data(), n * 4);
}

// Packed half-precision: v_pk_add_f16, v_pk_mul_f16, v_pk_fma_f16 on integer-valued and dyadic data (every result exact).
__global__ void k_pkf16(uint32_t* out, uint32_t iters) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  f16x2 x = (f16x2){(_Float16)(1 + (t & 7)), (_Float16)(2 + ((t >> 3) & 7))};
  const f16x2 half = (f16x2){(_Float16)0.5f, (_Float16)0.75f}, c = (f16x2){(_Float16)0.125f, (_Float16)0.25f};
  for (uint32_t i = 0; i < iters; ++i) {
    x = __builtin_elementwise_fma(x, half, c);       // v_pk_fma_f16
    x = x + (f16x2){(_Float16)1, (_Float16)2};       // v_pk_add_f16
    x = x * half;                                    // v_pk_mul_f16
    x = __builtin_elementwise_max(x, c) - c;         // v_pk_max_f16 (and an add of the negated)
  }
  uint16_t a, b;
  const _Float16 x0 = x[0], x1 = x[1];
  __builtin_memcpy(&a, &x0, 2);
  __builtin_memcpy(&b, &x1, 2);
  out[t] = a | ((uint32_t)b << 16);
}
static void run_pkf16(const std::string& size) {
  const uint32_t blocks = 32, threads = 256, iters = scale(size, 100, 2000), n = blocks * threads;
  uint32_t* d;
  HC(hipMalloc(&d, n * 4));
  Phase p;
  LAUNCH(k_pkf16, dim3(blocks), dim3(threads), 0, 0, d, iters);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> got = dev_read(d, n), ref(n);
  for (uint32_t t = 0; t < n; ++t) {
    float x[2] = {(float)(1 + (t & 7)), (float)(2 + ((t >> 3) & 7))};
    const float half[2] = {0.5f, 0.75f}, c[2] = {0.125f, 0.25f}, one[2] = {1.f, 2.f};
    const auto r16 = [](double v) { return (float)(_Float16)v; };
    for (uint32_t i = 0; i < iters; ++i) {
      for (int k = 0; k < 2; ++k) {
        x[k] = r16(std::fma((double)x[k], (double)half[k], (double)c[k]));
        x[k] = r16((double)x[k] + one[k]);
        x[k] = r16((double)x[k] * half[k]);
        x[k] = r16((double)std::max(x[k], c[k]) - c[k]);
      }
    }
    uint16_t a, b;
    const _Float16 ha = (_Float16)x[0], hb = (_Float16)x[1];
    std::memcpy(&a, &ha, 2);
    std::memcpy(&b, &hb, 2);
    ref[t] = a | ((uint32_t)b << 16);
  }
  expect_eq("pkf16", got.data(), ref.data(), n * 4);
}

// ---------------------------------------------------------------- sub-dword operands (SDWA on GCN/RDNA1-2)

__host__ __device__ static inline uint32_t bytes_step(uint32_t x, uint32_t y) {
  const uint32_t b0 = (x & 0xff) + (y >> 24), b1 = ((x >> 8) & 0xff) ^ ((y >> 16) & 0xff);
  const uint32_t b2 = (x >> 16) & 0xff, b3 = x >> 24;
  const uint32_t w = (uint16_t)x + (uint16_t)(y >> 16);
  const uint32_t m = b2 > ((y >> 8) & 0xff) ? b2 : ((y >> 8) & 0xff);
  return (b0 & 0xff) | (b1 << 8) | ((b3 + m) << 16 & 0xff0000u) | (w << 24) ^ (b3 << 24);
}
__global__ void k_sdwa(uint32_t* out, uint32_t iters) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t x = t * 2654435761u + 1, y = t * 40503u + 99;
  for (uint32_t i = 0; i < iters; ++i) {
    x = bytes_step(x, y) + i;
    y = rotl32(y, 5) + (x >> 7);
  }
  out[t] = x ^ y;
}
static void run_sdwa(const std::string& size) {
  const uint32_t blocks = 32, threads = 256, iters = scale(size, 100, 2000), n = blocks * threads;
  uint32_t* d;
  HC(hipMalloc(&d, n * 4));
  Phase p;
  LAUNCH(k_sdwa, dim3(blocks), dim3(threads), 0, 0, d, iters);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> got = dev_read(d, n), ref(n);
  for (uint32_t t = 0; t < n; ++t) {
    uint32_t x = t * 2654435761u + 1, y = t * 40503u + 99;
    for (uint32_t i = 0; i < iters; ++i) {
      x = bytes_step(x, y) + i;
      y = rotl32(y, 5) + (x >> 7);
    }
    ref[t] = x ^ y;
  }
  expect_eq("sdwa", got.data(), ref.data(), n * 4);
}

// ---------------------------------------------------------------- independent pairs (VOPD on RDNA3/4)

__global__ void k_vopd(float* out, uint32_t iters) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  float a0 = 1.0f + (t & 15) * 0.01f, a1 = 0.5f + (t & 7) * 0.02f, a2 = 2.0f + (t & 3) * 0.1f, a3 = 0.25f + (t & 31) * 0.005f;
  float b0 = 0.9f, b1 = 1.1f, b2 = 0.95f, b3 = 1.05f;
  for (uint32_t i = 0; i < iters; ++i) {
    a0 = fmaf(a0, 0.999f, b0);
    a1 = fmaf(a1, 0.5f, b1);
    a2 = a2 * 0.75f + b2;
    a3 = a3 * 0.875f - b3;
    b0 = fmaf(b0, 0.9f, a1 * 0.01f);
    b1 = b1 * 0.8f + a2 * 0.02f;
    b2 = b2 + a3 * 0.03f - 0.5f * b2;
    b3 = fmaf(b3, 0.7f, a0 * 0.04f);
  }
  out[t] = a0 + a1 + a2 + a3 + b0 + b1 + b2 + b3;
}
static void run_vopd(const std::string& size) {
  const uint32_t blocks = 32, threads = 256, iters = scale(size, 100, 2000), n = blocks * threads;
  float* d;
  HC(hipMalloc(&d, n * 4));
  Phase p;
  LAUNCH(k_vopd, dim3(blocks), dim3(threads), 0, 0, d, iters);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<float> got = dev_read(d, n), ref(n);
  for (uint32_t t = 0; t < n; ++t) {
    float a0 = 1.0f + (t & 15) * 0.01f, a1 = 0.5f + (t & 7) * 0.02f, a2 = 2.0f + (t & 3) * 0.1f, a3 = 0.25f + (t & 31) * 0.005f;
    float b0 = 0.9f, b1 = 1.1f, b2 = 0.95f, b3 = 1.05f;
    for (uint32_t i = 0; i < iters; ++i) {
      a0 = fmaf(a0, 0.999f, b0);
      a1 = fmaf(a1, 0.5f, b1);
      a2 = a2 * 0.75f + b2;
      a3 = a3 * 0.875f - b3;
      b0 = fmaf(b0, 0.9f, a1 * 0.01f);
      b1 = b1 * 0.8f + a2 * 0.02f;
      b2 = b2 + a3 * 0.03f - 0.5f * b2;
      b3 = fmaf(b3, 0.7f, a0 * 0.04f);
    }
    ref[t] = a0 + a1 + a2 + a3 + b0 + b1 + b2 + b3;
  }
  expect_eq("vopd", got.data(), ref.data(), n * 4);
}

// ---------------------------------------------------------------- compare and select shapes

__host__ __device__ static inline uint32_t cmp_step(uint32_t x, int32_t y, float f, double dd, uint32_t i) {
  uint32_t acc = 0;
  acc += x < 0x80000000u ? 3u : 5u;                    // v_cmp_gt_u32 + v_cndmask with two constants
  acc += y < -(int32_t)(i & 7) ? (uint32_t)y : x;      // v_cmp_lt_i32 + v_cndmask of registers
  acc += f > 1.5f ? 7u : 0u;                           // v_cmp_gt_f32
  acc += f == f ? 11u : 13u;                           // v_cmp_o_f32 / eq
  acc += (__builtin_isinf(f) ? 17u : 0u) + (__builtin_isnan(f) ? 19u : 0u);   // v_cmp_class_f32
  acc += dd < 0.5 ? 23u : 29u;                         // v_cmp_lt_f64
  acc += (x & 1u) != 0 && y > 0 ? 31u : 37u;           // s_and of two compares
  acc ^= (x == i) ? 0xA5A5u : 0u;                      // v_cmp_eq_u32
  return acc;
}
__host__ __device__ static inline float sel_f32(float f, uint32_t x) { return (x & 2u) ? f * 0.5f : f + 1.0f; }
__host__ __device__ static inline double sel_f64(double dd, uint32_t x) { return (x & 4u) ? dd * 1.25 : dd - 0.5; }
__global__ void k_cmp(uint32_t* out, uint32_t iters) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t x = t * 2654435761u + 1, acc = 0;
  float f = (float)(int32_t)(t & 0xff) * 0.0625f;
  double dd = (double)(t & 0x3f) * 0.03125;
  for (uint32_t i = 0; i < iters; ++i) {
    uint32_t bits = x;
    float g = (t & 15) == 7 ? __uint_as_float(0x7f800000u | ((t >> 4) & 1 ? 0 : 0x400000u)) : f;   // inf, or a quiet NaN
    acc += cmp_step(x, (int32_t)x >> 3, g, dd, i);
    f = sel_f32(f, x);
    dd = sel_f64(dd, x);
    if (dd > 1000.0) dd -= 999.0;
    if (f > 1000.0f) f -= 999.0f;
    x = rotl32(x, 9) + bits + acc;
  }
  out[t] = acc ^ x ^ (uint32_t)(int32_t)f ^ (uint32_t)(int32_t)dd;
}
static void run_cmp(const std::string& size) {
  const uint32_t blocks = 32, threads = 256, iters = scale(size, 100, 2000), n = blocks * threads;
  uint32_t* d;
  HC(hipMalloc(&d, n * 4));
  Phase p;
  LAUNCH(k_cmp, dim3(blocks), dim3(threads), 0, 0, d, iters);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> got = dev_read(d, n), ref(n);
  for (uint32_t t = 0; t < n; ++t) {
    uint32_t x = t * 2654435761u + 1, acc = 0;
    float f = (float)(int32_t)(t & 0xff) * 0.0625f;
    double dd = (double)(t & 0x3f) * 0.03125;
    for (uint32_t i = 0; i < iters; ++i) {
      uint32_t bits = x;
      uint32_t gb = 0;
      const bool special = (t & 15) == 7;
      if (special) gb = 0x7f800000u | (((t >> 4) & 1) ? 0 : 0x400000u);
      float g;
      if (special) std::memcpy(&g, &gb, 4); else g = f;
      acc += cmp_step(x, (int32_t)x >> 3, g, dd, i);
      f = sel_f32(f, x);
      dd = sel_f64(dd, x);
      if (dd > 1000.0) dd -= 999.0;
      if (f > 1000.0f) f -= 999.0f;
      x = rotl32(x, 9) + bits + acc;
    }
    ref[t] = acc ^ x ^ (uint32_t)(int32_t)f ^ (uint32_t)(int32_t)dd;
  }
  expect_eq("cmp", got.data(), ref.data(), n * 4);
}

// ---------------------------------------------------------------- 64-bit integers and doubles

__host__ __device__ static inline uint64_t i64_step(uint64_t x, uint32_t i) {
  x = x * 0x9E3779B97F4A7C15ull + i;                            // v_mad_u64_u32 / v_mul_lo, v_mul_hi
  x ^= x >> 29;                                                 // v_lshrrev_b64
  x += x << 11;                                                 // v_lshlrev_b64, v_add_co / addc
  x = rotl64(x, 17);
  const int64_t sx = (int64_t)x >> 7;                           // v_ashrrev_i64
  x ^= (uint64_t)sx;
  const uint64_t y = x * 3u;
  x = x > y ? x : y;                                            // 64-bit compare + cndmask
  x += (uint64_t)(int64_t)(int32_t)(x >> 40) * (uint64_t)(uint32_t)(x >> 3);   // v_mad_i64_i32 shapes
  x ^= (uint64_t)__builtin_popcountll(x) << 20;                 // v_bcnt_u32_b32 x2
  x += (uint64_t)__builtin_clzll(x | 1) << 5;                   // v_ffbh_u32 x2
  return x;
}
__global__ void k_i64(uint64_t* out, uint32_t iters) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t x = ((uint64_t)t << 32 | (t * 2654435761u)) + 1;
  for (uint32_t i = 0; i < iters; ++i) x = i64_step(x, i);
  out[t] = x;
}
static void run_i64(const std::string& size) {
  const uint32_t blocks = 32, threads = 256, iters = scale(size, 100, 1500), n = blocks * threads;
  uint64_t* d;
  HC(hipMalloc(&d, n * 8));
  Phase p;
  LAUNCH(k_i64, dim3(blocks), dim3(threads), 0, 0, d, iters);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint64_t> got = dev_read(d, n), ref(n);
  for (uint32_t t = 0; t < n; ++t) {
    uint64_t x = ((uint64_t)t << 32 | (t * 2654435761u)) + 1;
    for (uint32_t i = 0; i < iters; ++i) x = i64_step(x, i);
    ref[t] = x;
  }
  expect_eq("i64", got.data(), ref.data(), n * 8);
}

// Doubles beyond fma: ldexp, frexp, floor, trunc, rint, conversions to and from 32-bit, min, max, abs, negate.
__global__ void k_f64x(double* out, uint32_t iters) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  double x = 1.0 + (t & 15) * 0.0625, y = 0.5 + (t & 7) * 0.125, acc = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    int e;
    const double m = frexp(x * 3.0 + y, &e);                    // v_frexp_mant_f64, v_frexp_exp_i32_f64
    acc += ldexp(m, (int)(i & 7) - 3) + (double)e;              // v_ldexp_f64, v_cvt_f64_i32
    acc += floor(y * 1.7) + trunc(x * 2.3) + rint(m * 9.0);     // v_floor_f64, v_trunc_f64, v_rndne_f64
    acc += (double)(float)(x * 0.1) + (double)(int)(y * 100.0) + (double)(unsigned)(x * 50.0);   // v_cvt_f32_f64, f64_f32, i32, u32
    acc = fmin(acc, 1.0e6) + fmax(-fabs(y), -1.0) * 0.0625;     // v_min_f64, v_max_f64, abs/neg modifiers
    x = fabs(x * 0.9990234375 + 0.001953125) + (i & 1) * 0.0009765625;
    y = y * 0.875 + 0.0625 * m;
  }
  out[t] = acc + x + y;
}
static void run_f64x(const std::string& size) {
  const uint32_t blocks = 32, threads = 256, iters = scale(size, 100, 1500), n = blocks * threads;
  double* d;
  HC(hipMalloc(&d, n * 8));
  Phase p;
  LAUNCH(k_f64x, dim3(blocks), dim3(threads), 0, 0, d, iters);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<double> got = dev_read(d, n), ref(n);
  for (uint32_t t = 0; t < n; ++t) {
    double x = 1.0 + (t & 15) * 0.0625, y = 0.5 + (t & 7) * 0.125, acc = 0;
    for (uint32_t i = 0; i < iters; ++i) {
      int e;
      const double m = frexp(x * 3.0 + y, &e);
      acc += ldexp(m, (int)(i & 7) - 3) + (double)e;
      acc += floor(y * 1.7) + trunc(x * 2.3) + rint(m * 9.0);
      acc += (double)(float)(x * 0.1) + (double)(int)(y * 100.0) + (double)(unsigned)(x * 50.0);
      acc = fmin(acc, 1.0e6) + fmax(-fabs(y), -1.0) * 0.0625;
      x = fabs(x * 0.9990234375 + 0.001953125) + (i & 1) * 0.0009765625;
      y = y * 0.875 + 0.0625 * m;
    }
    ref[t] = acc + x + y;
  }
  expect_eq("f64x", got.data(), ref.data(), n * 8);
}

// ---------------------------------------------------------------- scalar memory of every width, and s_buffer_load

typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x16 __attribute__((ext_vector_type(16)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
__global__ void k_smem_wide(const uint32_t* __restrict__ tab, uint32_t* __restrict__ out, uint32_t n, uint32_t reps) {
  uint32_t acc = threadIdx.x;
  for (uint32_t r = 0; r < reps; ++r)
    for (uint32_t i = 0; i < n; i += 32) {
      const u32x16 a = *reinterpret_cast<const u32x16*>(tab + i);         // s_load_dwordx16
      const u32x8 b = *reinterpret_cast<const u32x8*>(tab + i + 16);      // s_load_dwordx8
      const u32x2 c = *reinterpret_cast<const u32x2*>(tab + i + 24);      // s_load_dwordx2
      acc = acc * 31u + a[0] + a[5] * 3u + a[15];
      acc ^= b[1] + b[7] + r;
      acc += c[0] * 7u + c[1];
    }
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
static void run_smem_wide(const std::string& size) {
  const uint32_t n = 1024, blocks = 32, threads = 256, reps = scale(size, 20, 600);
  std::vector<uint32_t> tab(n);
  for (uint32_t i = 0; i < n; ++i) tab[i] = i * 2654435761u + 9;
  uint32_t *dt = dev_copy(tab), *o;
  HC(hipMalloc(&o, blocks * threads * 4));
  Phase p;
  LAUNCH(k_smem_wide, dim3(blocks), dim3(threads), 0, 0, (const uint32_t*)dt, o, n, reps);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> got = dev_read(o, blocks * threads), ref(blocks * threads);
  for (uint32_t t = 0; t < blocks * threads; ++t) {
    uint32_t acc = t % threads;
    for (uint32_t r = 0; r < reps; ++r)
      for (uint32_t i = 0; i < n; i += 32) {
        acc = acc * 31u + tab[i] + tab[i + 5] * 3u + tab[i + 15];
        acc ^= tab[i + 17] + tab[i + 23] + r;
        acc += tab[i + 24] * 7u + tab[i + 25];
      }
    ref[t] = acc;
  }
  expect_eq("smem_wide", got.data(), ref.data(), blocks * threads * 4);
}

// s_buffer_load_dword / dwordx4 through the scalar buffer path (inline asm: the compiler turns only some buffer loads into it).
typedef int i32x4 __attribute__((ext_vector_type(4)));
#if defined(__GFX9__) || defined(__GFX10__)
#define SBUF_B32 "s_buffer_load_dword %0, %1, %2\n s_waitcnt lgkmcnt(0)"
#define SBUF_B128 "s_buffer_load_dwordx4 %0, %1, %2\n s_waitcnt lgkmcnt(0)"
#else
#define SBUF_B32 "s_buffer_load_b32 %0, %1, %2\n s_wait_kmcnt 0"
#define SBUF_B128 "s_buffer_load_b128 %0, %1, %2\n s_wait_kmcnt 0"
#endif
#if defined(__GFX12__)
#undef SBUF_B32
#undef SBUF_B128
#define SBUF_B32 "s_buffer_load_b32 %0, %1, %2\n s_wait_kmcnt 0"
#define SBUF_B128 "s_buffer_load_b128 %0, %1, %2\n s_wait_kmcnt 0"
#elif defined(__GFX11__)
#undef SBUF_B32
#undef SBUF_B128
#define SBUF_B32 "s_buffer_load_b32 %0, %1, %2\n s_waitcnt lgkmcnt(0)"
#define SBUF_B128 "s_buffer_load_b128 %0, %1, %2\n s_waitcnt lgkmcnt(0)"
#endif
__global__ void k_sbuf(const uint32_t* tab, uint32_t* out, uint32_t records, uint32_t reps, uint32_t flags) {
  const uint64_t base = reinterpret_cast<uint64_t>(tab);
  const i32x4 rs = {(int)(uint32_t)base, (int)(uint32_t)((base >> 32) & 0xffff), (int)(records * 4), (int)flags};
  uint32_t acc = threadIdx.x;
  for (uint32_t r = 0; r < reps; ++r) {
    uint32_t a, b;
    i32x4 c;
    asm volatile(SBUF_B32 : "=s"(a) : "s"(rs), "i"(16));
    asm volatile(SBUF_B32 : "=s"(b) : "s"(rs), "i"(400));
    asm volatile(SBUF_B128 : "=s"(c) : "s"(rs), "i"(64));
    acc = acc * 31u + a + b * 3u + r;
    acc ^= (uint32_t)c[0] + (uint32_t)c[3] * 5u;
    acc += (uint32_t)c[1] ^ (uint32_t)c[2];
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
static void run_sbuf(const std::string& size) {
  const uint32_t n = 256, blocks = 32, threads = 256, reps = scale(size, 100, 4000);
  std::vector<uint32_t> tab(n);
  for (uint32_t i = 0; i < n; ++i) tab[i] = i * 2654435761u + 21;
  uint32_t *dt = dev_copy(tab), *o;
  HC(hipMalloc(&o, blocks * threads * 4));
  if (!has_arch("gfx9") && !has_arch("gfx103")) {
    std::printf("SKIP no verified s_buffer_load descriptor for %s\n", g_arch);
    return;
  }
  const uint32_t flags = has_arch("gfx9") ? 0x00020000u : 0x31014000u;
  Phase p;
  LAUNCH(k_sbuf, dim3(blocks), dim3(threads), 0, 0, (const uint32_t*)dt, o, n, reps, flags);
  HC(hipDeviceSynchronize());
  p.finish(1, "launch");
  std::vector<uint32_t> got = dev_read(o, blocks * threads), ref(blocks * threads);
  for (uint32_t t = 0; t < blocks * threads; ++t) {
    uint32_t acc = t % threads;
    for (uint32_t r = 0; r < reps; ++r) {
      acc = acc * 31u + tab[4] + tab[100] * 3u + r;
      acc ^= tab[16] + tab[19] * 5u;
      acc += tab[17] ^ tab[18];
    }
    ref[t] = acc;
  }
  expect_eq("sbuf", got.data(), ref.data(), blocks * threads * 4);
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
  if (w == "dpp") run_dpp(size, 0);
  else if (w == "dpp_arch") run_dpp(size, 1);
  else if (w == "dpp_scan") run_dpp_scan(size);
  else if (w == "perm") run_perm(size);
  else if (w == "permlane") run_permlane(size);
  else if (w == "lane") run_lane(size);
  else if (w == "pk16") run_pk16(size);
  else if (w == "pkf16") run_pkf16(size);
  else if (w == "sdwa") run_sdwa(size);
  else if (w == "vopd") run_vopd(size);
  else if (w == "cmp") run_cmp(size);
  else if (w == "i64") run_i64(size);
  else if (w == "f64x") run_f64x(size);
  else if (w == "smem_wide") run_smem_wide(size);
  else if (w == "sbuf") run_sbuf(size);
  else {
    std::fprintf(stderr, "unknown workload %s\n", w.c_str());
    return 2;
  }
  std::printf(g_ok ? "PASS\n" : "FAIL\n");
  return g_ok ? 0 : 1;
}
