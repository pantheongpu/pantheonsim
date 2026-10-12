// The plain ALU and tensor-core forms the simulator's SIMD kernels (include/vgpu/simd) run -- IADD3, IMAD, LOP3,
// ISETP, shifts, FFMA, FADD, FMUL, HFMA2, DFMA/DADD/DMUL, and the f16/bf16/tf32 tensor core with f32 and f16
// accumulators -- over values that break arithmetic (zeros of both signs, subnormals, the extremes, infinities,
// NaNs, rounding edges). One line per form: an FNV hash of every result's bits. The expected file is an RTX 3060's
// (nvidia/tests/data/simd_lane_forms.rtx3060.txt); run_simd_lane_forms.sh runs this on the simulator's two engines
// at every SIMD level and compares, or on a real GPU with --card.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

constexpr int kN = 4096;

__device__ __host__ inline uint32_t mix32(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x;
}

static const uint32_t kWords[] = {
    0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x007FFFFFu, 0x807FFFFFu, 0x00800000u, 0x80800000u,
    0x3F800000u, 0xBF800000u, 0x3F800001u, 0x3F7FFFFFu, 0x40000000u, 0xC0000000u, 0x7F7FFFFFu, 0xFF7FFFFFu,
    0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00000u, 0x7F800001u, 0x7FBFFFFFu, 0xFFA00000u, 0x7FC12345u,
    0x33800000u, 0x4B000000u, 0x4B800001u, 0x1F800000u, 0x5F800000u, 0x3E800000u, 0xFFFFFFFFu, 0x7FFFFFFFu,
    0xDEADBEEFu, 0x12345678u, 0x0000FFFFu, 0x00010000u, 0x55555555u, 0xAAAAAAAAu, 0x3F801000u, 0x3F800FFFu};
static const uint16_t kHalves[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF,
                                   0x7C00, 0xFC00, 0x7E00, 0x7C01, 0xFE01, 0x3555, 0x4248, 0x0200, 0x5640, 0xD640,
                                   0x0100, 0x0010, 0x3C01, 0x3BFF};
static const uint16_t kBf16[] = {0x0000, 0x8000, 0x0001, 0x007F, 0x0080, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F, 0x7F80,
                                 0xFF80, 0x7FC0, 0x7F81, 0x3EAB, 0x4049, 0x1F00, 0x6F00, 0x3F81, 0x3F7F};

__global__ void k_int(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* out, int form) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t x = a[i], y = b[i], z = c[i];
  uint32_t r = 0;
  switch (form) {
    case 0: r = x + y + z; break;                                  // IADD3
    case 1: r = x - y + z; break;                                  // IADD3 with a negated operand
    case 2: r = x * y + z; break;                                  // IMAD
    case 3: r = (x & y) | (~x & z); break;                         // LOP3
    case 4: r = x ^ y ^ z; break;                                  // LOP3 0x96
    case 5: r = (x | y) & ~z; break;                               // LOP3
    case 6: r = (y < 32 ? x << y : 0u); break;                     // shift with a count past the width
    case 7: r = (y < 32 ? x >> y : 0u); break;
    case 8: r = (uint32_t)((int)x >> (y & 31)); break;
    case 9: r = __funnelshift_l(x, z, y); break;                   // SHF, wrapping count
    case 10: r = __funnelshift_rc(x, z, y); break;                 // SHF, clamped count
    case 11: r = (x < y ? 1u : 0u) | ((x <= y ? 1u : 0u) << 1) | ((x == y ? 1u : 0u) << 2) | ((x != y ? 1u : 0u) << 3) |
                 ((x >= y ? 1u : 0u) << 4) | ((x > y ? 1u : 0u) << 5);                               // ISETP, unsigned
    break;
    case 12: r = ((int)x < (int)y ? 1u : 0u) | (((int)x <= (int)y ? 1u : 0u) << 1) | (((int)x == (int)y ? 1u : 0u) << 2) |
                 (((int)x != (int)y ? 1u : 0u) << 3) | (((int)x >= (int)y ? 1u : 0u) << 4) | (((int)x > (int)y ? 1u : 0u) << 5);
    break;                                                         // ISETP, signed
    case 13: r = __popc(x) + (__clz(y) << 8); break;
    case 14: r = min(x, y) ^ max((int)y, (int)z); break;
  }
  out[i] = r;
}

__global__ void k_float(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* out, int form) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const float x = __uint_as_float(a[i]), y = __uint_as_float(b[i]), z = __uint_as_float(c[i]);
  float r = 0;
  switch (form) {
    case 0: r = __fmaf_rn(x, y, z); break;
    case 1: r = __fadd_rn(x, y); break;
    case 2: r = __fmul_rn(x, y); break;
    case 3: r = __fmaf_rn(-x, y, fabsf(z)); break;
  }
  out[i] = __float_as_uint(r);
}

__global__ void k_half2(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* out, int form) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  __half2 x = *reinterpret_cast<const __half2*>(&a[i]), y = *reinterpret_cast<const __half2*>(&b[i]),
          z = *reinterpret_cast<const __half2*>(&c[i]);
  __half2 r;
  if (form == 0) r = __hfma2(x, y, z);
  else if (form == 1) r = __hadd2(x, y);
  else r = __hmul2(x, y);
  out[i] = *reinterpret_cast<uint32_t*>(&r);
}

__global__ void k_double(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* out, int form) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = (i * 7 + 3) % kN;
  const double x = __longlong_as_double((long long)a[i] << 32 | b[j]), y = __longlong_as_double((long long)b[i] << 32 | c[j]),
               z = __longlong_as_double((long long)c[i] << 32 | a[j]);
  double r = 0;
  switch (form) {
    case 0: r = __fma_rn(x, y, z); break;
    case 1: r = __dadd_rn(x, y); break;
    case 2: r = __dmul_rn(x, y); break;
    case 3: r = __fma_rn(-x, fabs(y), z); break;
  }
  // (Which NaN comes out of an operation with two NaN operands depends on the operand order ptxas picked, which
  // differs between toolkits: a NaN is hashed as the default one. The unit test pins the simulator's choice.)
  const long long u = r != r ? 0x7ff8000000000000ll : __double_as_longlong(r);
  out[2 * i] = (uint32_t)u;
  out[2 * i + 1] = (uint32_t)(u >> 32);
}

// One 16x16x16 tile per block.
template <typename T, typename AccT>
__global__ void k_wmma(const T* a, const T* b, const AccT* c, AccT* d) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 16, AccT> fc;
  const int t = blockIdx.x;
  wmma::load_matrix_sync(fa, a + t * 256, 16);
  wmma::load_matrix_sync(fb, b + t * 256, 16);
  wmma::load_matrix_sync(fc, c + t * 256, 16, wmma::mem_row_major);
  wmma::mma_sync(fc, fa, fb, fc);
  wmma::store_matrix_sync(d + t * 256, fc, 16, wmma::mem_row_major);
}
__global__ void k_wmma_tf32(const float* a, const float* b, const float* c, float* d) {
  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> fc;
  const int t = blockIdx.x;
  wmma::load_matrix_sync(fa, a + t * 128, 8);
  wmma::load_matrix_sync(fb, b + t * 128, 8);
  wmma::load_matrix_sync(fc, c + t * 256, 16, wmma::mem_row_major);
  wmma::mma_sync(fc, fa, fb, fc);
  wmma::store_matrix_sync(d + t * 256, fc, 16, wmma::mem_row_major);
}

static uint64_t fnv(const void* p, size_t bytes) {
  uint64_t h = 1469598103934665603ull;
  const uint8_t* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < bytes; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

static std::vector<uint32_t> words(uint32_t seed, int n, int style) {
  std::vector<uint32_t> v(n);
  for (int i = 0; i < n; ++i) {
    const uint32_t h = mix32(seed * 0x9E3779B9u + i);
    v[i] = style == 0 ? h : kWords[h % (sizeof kWords / sizeof kWords[0])];
  }
  return v;
}
static std::vector<uint32_t> half_pairs(uint32_t seed, int n) {
  std::vector<uint32_t> v(n);
  for (int i = 0; i < n; ++i) {
    const uint32_t h = mix32(seed * 0x9E3779B9u + i), g = mix32(h);
    const size_t m = sizeof kHalves / sizeof kHalves[0];
    v[i] = (uint32_t)kHalves[h % m] | ((uint32_t)kHalves[g % m] << 16);
  }
  return v;
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

template <typename T>
static T* upload(const std::vector<T>& v) {
  T* d;
  cudaMalloc(&d, v.size() * sizeof(T));
  cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

int main() {
  int fails = 0;
  for (int style = 0; style < 2; ++style) {
    const char* sn = style == 0 ? "random" : "edges";
    const auto a = words(1 + style, kN, style), b = words(11 + style, kN, style == 0 ? 0 : 1), c = words(21 + style, kN, style);
    uint32_t *da = upload(a), *db = upload(b), *dc = upload(c), *dout;
    CK(cudaMalloc(&dout, kN * 2 * sizeof(uint32_t)));
    std::vector<uint32_t> out(kN * 2);
    const auto run = [&](const char* name, auto launch, size_t words_out) -> int {
      cudaMemset(dout, 0, kN * 2 * sizeof(uint32_t));
      launch();
      CK(cudaGetLastError());
      CK(cudaMemcpy(out.data(), dout, words_out * 4, cudaMemcpyDeviceToHost));
      std::printf("%-22s %s %016llx\n", name, sn, (unsigned long long)fnv(out.data(), words_out * 4));
      return 0;
    };
    static const char* int_names[] = {"iadd3", "iadd3_neg", "imad", "lop3_mux", "lop3_xor3", "lop3_orandnot", "shl_clamp",
                                      "shr_clamp", "sar", "funnel_l", "funnel_rc", "isetp_u", "isetp_s", "popc_clz", "minmax"};
    for (int f = 0; f < 15; ++f) {
      if (run(int_names[f], [&] { k_int<<<kN / 128, 128>>>(da, db, dc, dout, f); }, kN)) return 1;
    }
    static const char* fn[] = {"ffma", "fadd", "fmul", "ffma_negabs"};
    for (int f = 0; f < 4; ++f)
      if (run(fn[f], [&] { k_float<<<kN / 128, 128>>>(da, db, dc, dout, f); }, kN)) return 1;
    // half2 operands from half pairs
    {
      const auto ha = half_pairs(31 + style, kN), hb = half_pairs(41 + style, kN), hc = half_pairs(51 + style, kN);
      uint32_t *xa = upload(ha), *xb = upload(hb), *xc = upload(hc);
      static const char* hn[] = {"hfma2", "hadd2", "hmul2"};
      for (int f = 0; f < 3; ++f)
        if (run(hn[f], [&] { k_half2<<<kN / 128, 128>>>(xa, xb, xc, dout, f); }, kN)) return 1;
    }
    static const char* dn[] = {"dfma", "dadd", "dmul", "dfma_negabs"};
    for (int f = 0; f < 4; ++f)
      if (run(dn[f], [&] { k_double<<<kN / 128, 128>>>(da, db, dc, dout, f); }, 2 * kN)) return 1;
    // tensor core: 64 tiles
    {
      const int tiles = 64;
      const auto ha = half_pairs(61 + style, tiles * 128), hb = half_pairs(71 + style, tiles * 128), hc = half_pairs(81 + style, tiles * 128);
      std::vector<uint32_t> bfa(tiles * 128), bfb(tiles * 128);
      for (int i = 0; i < tiles * 128; ++i) {
        const uint32_t h = mix32(91 + style + i * 3), g = mix32(h + 1), m = sizeof kBf16 / sizeof kBf16[0];
        bfa[i] = (uint32_t)kBf16[h % m] | ((uint32_t)kBf16[g % m] << 16);
        const uint32_t h2 = mix32(101 + style + i * 5), g2 = mix32(h2 + 1);
        bfb[i] = (uint32_t)kBf16[h2 % m] | ((uint32_t)kBf16[g2 % m] << 16);
      }
      const auto fc = words(111 + style, tiles * 512, style);
      uint32_t *pa = upload(ha), *pb = upload(hb), *pc = upload(hc), *pfc = upload(fc), *pba = upload(bfa), *pbb = upload(bfb);
      uint32_t* pd;
      CK(cudaMalloc(&pd, tiles * 256 * 4));
      std::vector<uint32_t> d(tiles * 256);
      // f32out: the result is f32 words, and a second line hashes them with every NaN replaced by the canonical one.
      // Which NaN comes out of an add or multiply of two NaNs is the host instruction's choice, and the compiler
      // chooses the order of the operands of the simulator's own vector adds, so the raw line can differ between a
      // sanitizer build and an optimised one while every other element is the same (run_simd_lane_forms.sh).
      const auto show = [&](const char* name, size_t n, bool f32out) {
        cudaDeviceSynchronize();
        cudaMemcpy(d.data(), pd, n * 4, cudaMemcpyDeviceToHost);
        std::printf("%-22s %s %016llx\n", name, sn, (unsigned long long)fnv(d.data(), n * 4));
        if (f32out) {
          for (size_t i = 0; i < n; ++i)
            if ((d[i] & 0x7fffffffu) > 0x7f800000u) d[i] = 0x7fffffffu;
          std::printf("%-22s %s %016llx\n", (std::string(name) + "_nonan").c_str(), sn, (unsigned long long)fnv(d.data(), n * 4));
        }
      };
      k_wmma<__half, float><<<tiles, 32>>>((const __half*)pa, (const __half*)pb, (const float*)pfc, (float*)pd);
      CK(cudaGetLastError());
      show("wmma_f16_f32acc", tiles * 256, true);
      k_wmma<__half, __half><<<tiles, 32>>>((const __half*)pa, (const __half*)pb, (const __half*)pc, (__half*)pd);
      CK(cudaGetLastError());
      show("wmma_f16_f16acc", tiles * 128, false);
      k_wmma<__nv_bfloat16, float><<<tiles, 32>>>((const __nv_bfloat16*)pba, (const __nv_bfloat16*)pbb, (const float*)pfc, (float*)pd);
      CK(cudaGetLastError());
      show("wmma_bf16_f32acc", tiles * 256, true);
      k_wmma_tf32<<<tiles, 32>>>((const float*)pfc, (const float*)(pfc + 1024), (const float*)(pfc + 2048), (float*)pd);
      CK(cudaGetLastError());
      show("wmma_tf32_f32acc", tiles * 256, true);
    }
  }
  return fails;
}
