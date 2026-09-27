// gfx950's (CDNA4's) block-scaled matrix instructions and its two new VOP1
// ones, through the compiler's builtins, checked against what the CDNA4 ISA
// reference guide says they compute.
//
// v_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4 multiplies matrices of 8-, 6-
// or 4-bit floats (each operand's format its own), each lane's one row and
// 32 values along K scaled by 2^(e - 127) -- e an E8M0 byte of the lane's
// scale register, the byte its selector names. The host works the same
// product out from the ISA's format table and layouts. v_prng_b32 is one
// step of the ISA's LFSR, and v_permlane32_swap_b32 trades the upper 32 lanes
// of one register for the lower 32 of another. Built for gfx950 only
// (gfx950.gfx950), and run on a simulated MI350X.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(x)                                                                        \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef float v16f __attribute__((ext_vector_type(16)));

// One scaled product per lane's operands: A's and B's eight dwords, the
// addend, and each lane's scale register. The formats and selectors are the
// instruction's own constants.
template <int CBSZ, int BLGP, int SELA, int SELB>
__global__ void scaled16(const v8i* a, const v8i* b, const v4f* c, const unsigned* sa, const unsigned* sb, v4f* d) {
  const int l = threadIdx.x;
  d[l] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a[l], b[l], c[l], CBSZ, BLGP, SELA, (int)sa[l], SELB,
                                                          (int)sb[l]);
}
template <int CBSZ, int BLGP, int SELA, int SELB>
__global__ void scaled32(const v8i* a, const v8i* b, const v16f* c, const unsigned* sa, const unsigned* sb, v16f* d) {
  const int l = threadIdx.x;
  d[l] = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a[l], b[l], c[l], CBSZ, BLGP, SELA, (int)sa[l], SELB,
                                                         (int)sb[l]);
}
__global__ void prng(const unsigned* in, unsigned* out) { out[threadIdx.x] = __builtin_amdgcn_prng_b32(in[threadIdx.x]); }
__global__ void swap32(unsigned* d, unsigned* s) {
  const int l = threadIdx.x;
  auto r = __builtin_amdgcn_permlane32_swap(d[l], s[l], false, false);
  d[l] = r[0];
  s[l] = r[1];
}

// ---- The host's side, from the ISA's tables ------------------------------------

// Bits, exponent bias and mantissa bits of each format (CBSZ/BLGP code 0-4):
// E4M3 (no infinity, S.1111.111 NaN), E5M2 (IEEE-like), E2M3, E3M2, E2M1.
struct Format {
  int bits, mant, bias;
};
const Format kFormats[5] = {{8, 3, 7}, {8, 2, 15}, {6, 3, 1}, {6, 2, 3}, {4, 1, 1}};

double small_float(uint32_t v, int fmt) {
  const Format f = kFormats[fmt];
  const uint32_t sign = 1u << (f.bits - 1), mag = v & (sign - 1);
  if (fmt == 0 && mag == 0x7F) return NAN;
  if (fmt == 1 && (mag >> 2) == 0x1F) return mag & 3 ? NAN : (v & sign ? -INFINITY : INFINITY);
  const int e = static_cast<int>(mag >> f.mant), m = static_cast<int>(mag & ((1u << f.mant) - 1));
  const double x = e == 0 ? std::ldexp(m, 1 - f.bias - f.mant) : std::ldexp((1 << f.mant) | m, e - f.bias - f.mant);
  return v & sign ? -x : x;
}
// Element e of a lane's run of 32, packed from the lowest bit up.
uint32_t element(const int* words, int bits, int e) {
  const int at = e * bits, w = at / 32, sh = at % 32;
  uint64_t v = static_cast<uint32_t>(words[w]);
  if (sh + bits > 32 && w + 1 < 8) v |= static_cast<uint64_t>(static_cast<uint32_t>(words[w + 1])) << 32;
  return static_cast<uint32_t>(v >> sh) & ((1u << bits) - 1);
}
double scale(unsigned reg, int sel) {
  const unsigned e = (reg >> (8 * sel)) & 0xFF;
  return e == 0xFF ? NAN : std::ldexp(1.0, static_cast<int>(e) - 127);
}

uint32_t rng = 12345;
uint32_t next() { return rng = rng * 1664525u + 1013904223u; }

// A random operand of a format: every value finite, so the product is too.
void fill(int* words, int fmt) {
  std::memset(words, 0, 32);
  const int bits = kFormats[fmt].bits;
  for (int e = 0; e < 32; ++e) {
    uint32_t v;
    do v = next() >> (32 - bits);
    while (!std::isfinite(small_float(v, fmt)));
    const int at = e * bits;
    for (int b = 0; b < bits; ++b)
      if (v >> b & 1) words[(at + b) / 32] |= 1 << ((at + b) % 32);
  }
}

// M x N outputs from K = 64 * 32 / M values a lane; lane l holds row (or
// column) l % M, K block l / M. Output (i, j) is in lane j + M * (block of
// four rows' place), as every 16- and 32-wide matrix instruction lays it.
template <int M>
int check_scaled(const char* what, int cbsz, int blgp, int sela, int selb, void (*launch)(const v8i*, const v8i*,
                                                                                          const float*, const unsigned*,
                                                                                          const unsigned*, float*)) {
  constexpr int K = 64 * 32 / M, outs = M * M / 64;
  std::vector<v8i> a(64), b(64);
  std::vector<float> c(64 * outs), d(64 * outs);
  std::vector<unsigned> sa(64), sb(64);
  for (int l = 0; l < 64; ++l) {
    fill(reinterpret_cast<int*>(&a[l]), cbsz);
    fill(reinterpret_cast<int*>(&b[l]), blgp);
    sa[l] = next();
    sb[l] = next();
    // Scales near 1 in the selected bytes, the rest noise.
    reinterpret_cast<unsigned char*>(&sa[l])[sela] = static_cast<unsigned char>(124 + next() % 7);
    reinterpret_cast<unsigned char*>(&sb[l])[selb] = static_cast<unsigned char>(124 + next() % 7);
    for (int r = 0; r < outs; ++r) c[l * outs + r] = static_cast<float>(static_cast<int>(next() % 17) - 8);
  }
  launch(a.data(), b.data(), c.data(), sa.data(), sb.data(), d.data());
  int wrong = 0;
  for (int l = 0; l < 64; ++l)
    for (int r = 0; r < outs; ++r) {
      // Output register r of lane l: column j = l % M; row i from the lane's
      // group of four and the register.
      const int j = l % M;
      const int i = M == 16 ? 4 * (l / 16) + r : 8 * (r / 4) + 4 * (l / 32) + r % 4;
      double want = c[l * outs + r];
      for (int k = 0; k < K; ++k) {
        const int la = i + M * (k / 32), lb = j + M * (k / 32);
        want += small_float(element(reinterpret_cast<const int*>(&a[la]), kFormats[cbsz].bits, k % 32), cbsz) *
                scale(sa[la], sela) *
                small_float(element(reinterpret_cast<const int*>(&b[lb]), kFormats[blgp].bits, k % 32), blgp) *
                scale(sb[lb], selb);
      }
      const float got = d[l * outs + r];
      if (std::fabs(got - want) > 1e-5 * std::fmax(1.0, std::fabs(want)) && wrong++ < 2)
        std::printf("  %s: D[%d][%d] is %g, not %g\n", what, i, j, got, want);
    }
  std::printf("%s: %d of %d wrong\n", what, wrong, 64 * outs);
  return wrong;
}

template <typename Kernel, typename Out>
void run(Kernel k, const v8i* a, const v8i* b, const float* c, const unsigned* sa, const unsigned* sb, float* d,
         int outs) {
  v8i *da, *db;
  Out *dc, *dd;
  unsigned *dsa, *dsb;
  (void)hipMalloc(&da, 64 * sizeof(v8i));
  (void)hipMalloc(&db, 64 * sizeof(v8i));
  (void)hipMalloc(&dc, 64 * sizeof(Out));
  (void)hipMalloc(&dd, 64 * sizeof(Out));
  (void)hipMalloc(&dsa, 256);
  (void)hipMalloc(&dsb, 256);
  (void)hipMemcpy(da, a, 64 * sizeof(v8i), hipMemcpyHostToDevice);
  (void)hipMemcpy(db, b, 64 * sizeof(v8i), hipMemcpyHostToDevice);
  (void)hipMemcpy(dc, c, 64 * outs * 4, hipMemcpyHostToDevice);
  (void)hipMemcpy(dsa, sa, 256, hipMemcpyHostToDevice);
  (void)hipMemcpy(dsb, sb, 256, hipMemcpyHostToDevice);
  k<<<1, 64>>>(da, db, dc, dsa, dsb, dd);
  (void)hipMemcpy(d, dd, 64 * outs * 4, hipMemcpyDeviceToHost);
  for (void* p : {static_cast<void*>(da), static_cast<void*>(db), static_cast<void*>(dc), static_cast<void*>(dd),
                  static_cast<void*>(dsa), static_cast<void*>(dsb)})
    (void)hipFree(p);
}

#define SCALED16(CB, BL, SA, SB)                                                                               \
  check_scaled<16>("16x16x128 formats " #CB "/" #BL ", scale bytes " #SA "/" #SB, CB, BL, SA, SB,             \
                   [](const v8i* a, const v8i* b, const float* c, const unsigned* sa, const unsigned* sb,     \
                      float* d) { run<decltype(&scaled16<CB, BL, SA, SB>), v4f>(scaled16<CB, BL, SA, SB>, a, b, \
                                                                                 c, sa, sb, d, 4); })
#define SCALED32(CB, BL, SA, SB)                                                                                \
  check_scaled<32>("32x32x64 formats " #CB "/" #BL ", scale bytes " #SA "/" #SB, CB, BL, SA, SB,               \
                   [](const v8i* a, const v8i* b, const float* c, const unsigned* sa, const unsigned* sb,      \
                      float* d) { run<decltype(&scaled32<CB, BL, SA, SB>), v16f>(scaled32<CB, BL, SA, SB>, a, \
                                                                                  b, c, sa, sb, d, 16); })

int main() {
  SCALED16(0, 0, 0, 0);   // fp8 x fp8
  SCALED16(1, 0, 1, 2);   // bf8 x fp8
  SCALED16(2, 3, 3, 1);   // fp6 x bf6
  SCALED16(4, 4, 2, 3);   // fp4 x fp4
  SCALED32(0, 1, 0, 3);   // fp8 x bf8
  SCALED32(4, 2, 1, 0);   // fp4 x fp6

  // v_prng_b32: (x << 1) ^ (x's top bit ? 197 : 0).
  std::vector<unsigned> in(64), out(64);
  for (int l = 0; l < 64; ++l) in[l] = next() ^ (l == 0 ? 0u : 0x80000000u * (l & 1));
  in[1] = 0;
  unsigned *din, *dout;
  CHECK(hipMalloc(&din, 256));
  CHECK(hipMalloc(&dout, 256));
  CHECK(hipMemcpy(din, in.data(), 256, hipMemcpyHostToDevice));
  prng<<<1, 64>>>(din, dout);
  CHECK(hipMemcpy(out.data(), dout, 256, hipMemcpyDeviceToHost));
  int wrong = 0;
  for (int l = 0; l < 64; ++l) wrong += out[l] != ((in[l] << 1) ^ (in[l] >> 31 ? 197u : 0u));
  std::printf("v_prng_b32: %d of 64 wrong\n", wrong);

  // v_permlane32_swap_b32: the first register's upper 32 lanes and the
  // second's lower 32 trade places.
  std::vector<unsigned> vd(64), vs(64);
  for (int l = 0; l < 64; ++l) vd[l] = 1000 + l, vs[l] = 2000 + l;
  CHECK(hipMemcpy(din, vd.data(), 256, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dout, vs.data(), 256, hipMemcpyHostToDevice));
  swap32<<<1, 64>>>(din, dout);
  CHECK(hipMemcpy(vd.data(), din, 256, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(vs.data(), dout, 256, hipMemcpyDeviceToHost));
  wrong = 0;
  for (int l = 0; l < 64; ++l) {
    wrong += vd[l] != (l < 32 ? 1000u + l : 2000u + (l - 32));
    wrong += vs[l] != (l < 32 ? 1000u + (l + 32) : 2000u + l);
  }
  std::printf("v_permlane32_swap_b32: %d of 128 wrong\n", wrong);
  return 0;
}
