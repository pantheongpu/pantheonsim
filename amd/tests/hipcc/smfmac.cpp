// The sparse matrix instructions (v_smfmac_*), through the compiler's
// builtins, checked against what the CDNA3 and CDNA4 ISA reference guides
// say they compute.
//
// A is 2:4 sparse along K: of each four values of a lane's run only two are
// held, packed, and the index register says -- two bits a value -- which of
// the four each is. B is dense, and the destination is the addend too. The
// index register holds 32 / (the lane's dense run) sets of indices; with CBSZ
// zero ABID picks one, otherwise the first. Each product is run once with
// the other sets filled with noise, so a wrong set shows.
//
// Built for gfx942 (smfmac.gfx942: its fourteen forms, 8-bit floats the FNUZ
// ones) and gfx950 (smfmac.gfx950, -DVGPU_GFX950: its own fourteen with K
// doubled, and some of gfx942's, 8-bit floats the OCP ones).
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

typedef __fp16 v4h __attribute__((vector_size(8)));
typedef __fp16 v8h __attribute__((vector_size(16)));
typedef __fp16 v16h __attribute__((vector_size(32)));
typedef short v4s __attribute__((vector_size(8)));
typedef short v8s __attribute__((vector_size(16)));
typedef __bf16 v8y __attribute__((vector_size(16)));
typedef __bf16 v16y __attribute__((vector_size(32)));
typedef int v2i __attribute__((vector_size(8)));
typedef int v4i __attribute__((vector_size(16)));
typedef int v8i __attribute__((vector_size(32)));
typedef int v16i __attribute__((vector_size(64)));
typedef float v4f __attribute__((vector_size(16)));
typedef float v16f __attribute__((vector_size(64)));

// One kernel a form: lane l's A, B and addend from a run of eight words each
// (the form using as many as it needs), its index register, and the result.
#define FORM(NAME, VA, VB, VC)                                                                    \
  template <int CBSZ, int ABID>                                                                   \
  __global__ void NAME(const int* a, const int* b, const int* c, const int* idx, int* d) {        \
    const int l = threadIdx.x;                                                                    \
    VA va;                                                                                        \
    VB vb;                                                                                        \
    VC vc;                                                                                        \
    __builtin_memcpy(&va, a + 8 * l, sizeof va);                                                  \
    __builtin_memcpy(&vb, b + 8 * l, sizeof vb);                                                  \
    __builtin_memcpy(&vc, c + 16 * l, sizeof vc);                                                 \
    const VC r = __builtin_amdgcn_smfmac_##NAME(va, vb, vc, idx[l], CBSZ, ABID);                  \
    __builtin_memcpy(d + 16 * l, &r, sizeof r);                                                   \
  }

FORM(f32_16x16x32_f16, v4h, v8h, v4f)
FORM(f32_32x32x16_f16, v4h, v8h, v16f)
FORM(f32_16x16x32_bf16, v4s, v8s, v4f)
FORM(f32_32x32x16_bf16, v4s, v8s, v16f)
FORM(i32_16x16x64_i8, v2i, v4i, v4i)
FORM(i32_32x32x32_i8, v2i, v4i, v16i)
FORM(f32_16x16x64_fp8_fp8, v2i, v4i, v4f)
FORM(f32_16x16x64_bf8_fp8, v2i, v4i, v4f)
FORM(f32_32x32x32_fp8_bf8, v2i, v4i, v16f)
FORM(f32_32x32x32_bf8_bf8, v2i, v4i, v16f)
#ifdef VGPU_GFX950
FORM(f32_16x16x64_f16, v8h, v16h, v4f)
FORM(f32_32x32x32_f16, v8h, v16h, v16f)
FORM(f32_16x16x64_bf16, v8y, v16y, v4f)
FORM(f32_32x32x32_bf16, v8y, v16y, v16f)
FORM(i32_16x16x128_i8, v4i, v8i, v4i)
FORM(i32_32x32x64_i8, v4i, v8i, v16i)
FORM(f32_16x16x128_fp8_bf8, v4i, v8i, v4f)
FORM(f32_16x16x128_bf8_bf8, v4i, v8i, v4f)
FORM(f32_32x32x64_fp8_fp8, v4i, v8i, v16f)
FORM(f32_32x32x64_bf8_fp8, v4i, v8i, v16f)
#endif

// ---- The host's side, from the ISA's tables ------------------------------------

// Element e of a lane's words: 'h' a half, 'b' a bfloat16, 'c' a signed byte,
// 'e' an fp8 (E4M3), 'g' a bf8 (E5M2).
double fp8(uint32_t v, int mant) {
  const int bits_e = 7 - mant;
#ifdef VGPU_GFX950   // OCP: E4M3 has no infinity and S.1111.111 NaNs, E5M2 is IEEE's
  const int bias = (1 << (bits_e - 1)) - 1;
  const uint32_t mag = v & 0x7F;
  if (mant == 3 && mag == 0x7F) return NAN;
  if (mant == 2 && (mag >> 2) == 0x1F) return mag & 3 ? NAN : (v & 0x80 ? -INFINITY : INFINITY);
#else                // FNUZ: one NaN, 0x80, and the bias one more
  const int bias = 1 << (bits_e - 1);
  const uint32_t mag = v & 0x7F;
  if (v == 0x80) return NAN;
#endif
  const int e = static_cast<int>(mag >> mant), m = static_cast<int>(mag & ((1u << mant) - 1));
  const double x = e == 0 ? std::ldexp(m, 1 - bias - mant) : std::ldexp((1 << mant) | m, e - bias - mant);
  return v & 0x80 ? -x : x;
}
double element(const int* words, char type, int e) {
  const uint32_t w16 = static_cast<uint32_t>(words[e / 2]) >> (16 * (e % 2)) & 0xFFFF;
  const uint32_t w8 = static_cast<uint32_t>(words[e / 4]) >> (8 * (e % 4)) & 0xFF;
  switch (type) {
    case 'h': {
      _Float16 h;
      const uint16_t bits = static_cast<uint16_t>(w16);
      std::memcpy(&h, &bits, 2);
      return static_cast<double>(h);
    }
    case 'b': {
      const uint32_t bits = w16 << 16;
      float f;
      std::memcpy(&f, &bits, 4);
      return f;
    }
    case 'c': return static_cast<int8_t>(w8);
    case 'e': return fp8(w8, 3);
    default: return fp8(w8, 2);
  }
}

uint32_t rng = 2468;
uint32_t next() { return rng = rng * 1664525u + 1013904223u; }

// `n` random values of a type into a lane's words: small and finite. The
// halves and bfloat16s are half-integers, so their sums are exact.
void fill(int* words, char type, int n) {
  std::memset(words, 0, 32);
  for (int e = 0; e < n; ++e) {
    uint32_t v;
    if (type == 'h' || type == 'b') {
      // A small integer or half-integer, as a half or a bfloat16.
      const float f = static_cast<float>(static_cast<int>(next() % 33) - 16) / 2;
      if (type == 'h') {
        const _Float16 h = static_cast<_Float16>(f);
        uint16_t bits;
        std::memcpy(&bits, &h, 2);
        v = bits;
      } else {
        uint32_t bits;
        std::memcpy(&bits, &f, 4);
        v = bits >> 16;
      }
      words[e / 2] |= static_cast<int>(v << (16 * (e % 2)));
    } else {
      do v = next() >> 24;
      while (type != 'c' && (!std::isfinite(element(reinterpret_cast<int*>(&v), type, 0)) ||
                             std::fabs(element(reinterpret_cast<int*>(&v), type, 0)) > 64));
      words[e / 4] |= static_cast<int>(v << (8 * (e % 4)));
    }
  }
}

// An M x N (M = N) product with K = 64 * per_lane / M; lane l holds row (or
// column) l % M and per_lane values of K: A's the run of block l / M, B's the
// same, or on gfx950's forms two half-runs K/2 apart. A lane's A holds
// per_lane / 2 values, value v the one at 4 * (v / 2) + its index in that run.
// Output register r of lane l is D[i][l % M], i as every 16- and 32-wide
// matrix instruction lays it out.
using Launch = void (*)(const int*, const int*, const int*, const int*, int*);
int check(const char* what, int m, int per_lane, char ta, char tb, bool integer, int cbsz, int abid, Launch launch) {
  const int outs = m * m / 64, sets = 32 / per_lane, set = (cbsz == 0 ? abid : 0) % sets;
  std::vector<int> a(64 * 8), b(64 * 8), c(64 * 16), idx(64), d(64 * 16);
  for (int l = 0; l < 64; ++l) {
    fill(&a[8 * l], ta, per_lane / 2);
    fill(&b[8 * l], tb, per_lane);
    // Two different places of four, in order, for each pair of A's values;
    // every set, so the one chosen is the one read.
    uint32_t reg = 0;
    for (int p = 0; p < 16; ++p) {
      static const int pairs[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
      const int* q = pairs[next() % 6];
      reg |= static_cast<uint32_t>(q[0] | q[1] << 2) << (4 * p);
    }
    idx[l] = static_cast<int>(reg);
    for (int r = 0; r < outs; ++r) {
      const int v = static_cast<int>(next() % 17) - 8;
      if (integer) c[16 * l + r] = v;
      else {
        const float f = static_cast<float>(v);
        std::memcpy(&c[16 * l + r], &f, 4);
      }
    }
  }
  launch(a.data(), b.data(), c.data(), idx.data(), d.data());
  // A lane's dense run of A, from its values and the chosen indices.
  const auto dense_a = [&](int lane, int e) -> double {
    const uint32_t ix = static_cast<uint32_t>(idx[lane]) >> (set * per_lane);
    double x = 0;
    for (int v = 0; v < per_lane / 2; ++v)
      if (4 * (v / 2) + static_cast<int>(ix >> (2 * v) & 3) == e) x = element(&a[8 * lane], ta, v);
    return x;
  };
  int wrong = 0;
  for (int l = 0; l < 64; ++l)
    for (int r = 0; r < outs; ++r) {
      const int j = l % m, i = m == 16 ? 4 * (l / 16) + r : 8 * (r / 4) + 4 * (l / 32) + r % 4;
      double want;
      if (integer) want = c[16 * l + r];
      else {
        float f;
        std::memcpy(&f, &c[16 * l + r], 4);
        want = f;
      }
      // K value kk of row i is A's lane group kk / per_lane, and of column j
      // B's: the same where B is four registers; where it is eight (gfx950's
      // forms) the first four hold a group's run in the first half of K and
      // the last four the same run K/2 on.
      const int groups = 64 / m, K = groups * per_lane, half = per_lane / 2;
      const bool split = per_lane * (ta == 'h' || ta == 'b' ? 2 : 1) == 32;
      for (int kk = 0; kk < K; ++kk) {
        const int ga = kk / per_lane, ea = kk % per_lane;
        const int gb = split ? kk % (K / 2) / half : ga, eb = split ? kk / (K / 2) * half + kk % half : ea;
        want += dense_a(i + m * ga, ea) * element(&b[8 * (j + m * gb)], tb, eb);
      }
      double got;
      if (integer) got = d[16 * l + r];
      else {
        float f;
        std::memcpy(&f, &d[16 * l + r], 4);
        got = f;
      }
      // Integers, and sums of halves and bfloat16s, are exact; 8-bit floats'
      // products carry more bits than a float's sum can keep.
      const bool right = integer || ta == 'h' || ta == 'b' ? got == want
                                                            : std::fabs(got - want) <= 1e-6 * std::fmax(1.0, std::fabs(want));
      if (!right && wrong++ < 2) std::printf("  %s: D[%d][%d] is %g, not %g\n", what, i, j, got, want);
    }
  std::printf("%s: %d of %d wrong\n", what, wrong, 64 * outs);
  return wrong;
}

template <typename Kernel>
void run(Kernel k, const int* a, const int* b, const int* c, const int* idx, int* d) {
  int *da, *db, *dc, *di, *dd;
  (void)hipMalloc(&da, 64 * 32);
  (void)hipMalloc(&db, 64 * 32);
  (void)hipMalloc(&dc, 64 * 64);
  (void)hipMalloc(&di, 64 * 4);
  (void)hipMalloc(&dd, 64 * 64);
  (void)hipMemcpy(da, a, 64 * 32, hipMemcpyHostToDevice);
  (void)hipMemcpy(db, b, 64 * 32, hipMemcpyHostToDevice);
  (void)hipMemcpy(dc, c, 64 * 64, hipMemcpyHostToDevice);
  (void)hipMemcpy(di, idx, 64 * 4, hipMemcpyHostToDevice);
  k<<<1, 64>>>(da, db, dc, di, dd);
  (void)hipMemcpy(d, dd, 64 * 64, hipMemcpyDeviceToHost);
  for (int* p : {da, db, dc, di, dd}) (void)hipFree(p);
}

// A form, M, the dense run a lane has, A's and B's types, and CBSZ and ABID.
#define CHECK_FORM(NAME, M, PER_LANE, TA, TB, INT, CBSZ, ABID)                                                      \
  wrong += check(#NAME " cbsz:" #CBSZ " abid:" #ABID, M, PER_LANE, TA, TB, INT, CBSZ, ABID,                         \
                 [](const int* a, const int* b, const int* c, const int* i, int* d) { run(NAME<CBSZ, ABID>, a, b, c, i, d); })

int main() {
  int wrong = 0;
  CHECK_FORM(f32_16x16x32_f16, 16, 8, 'h', 'h', false, 0, 0);
  CHECK_FORM(f32_16x16x32_f16, 16, 8, 'h', 'h', false, 0, 2);
  CHECK_FORM(f32_16x16x32_f16, 16, 8, 'h', 'h', false, 1, 3);   // CBSZ set: the first set
  CHECK_FORM(f32_32x32x16_f16, 32, 8, 'h', 'h', false, 0, 1);
  CHECK_FORM(f32_16x16x32_bf16, 16, 8, 'b', 'b', false, 0, 3);
  CHECK_FORM(f32_32x32x16_bf16, 32, 8, 'b', 'b', false, 0, 0);
  CHECK_FORM(i32_16x16x64_i8, 16, 16, 'c', 'c', true, 0, 1);
  CHECK_FORM(i32_32x32x32_i8, 32, 16, 'c', 'c', true, 2, 1);
  CHECK_FORM(f32_16x16x64_fp8_fp8, 16, 16, 'e', 'e', false, 0, 0);
  CHECK_FORM(f32_16x16x64_bf8_fp8, 16, 16, 'g', 'e', false, 0, 1);
  CHECK_FORM(f32_32x32x32_fp8_bf8, 32, 16, 'e', 'g', false, 0, 1);
  CHECK_FORM(f32_32x32x32_bf8_bf8, 32, 16, 'g', 'g', false, 0, 0);
#ifdef VGPU_GFX950
  CHECK_FORM(f32_16x16x64_f16, 16, 16, 'h', 'h', false, 0, 1);
  CHECK_FORM(f32_32x32x32_f16, 32, 16, 'h', 'h', false, 0, 0);
  CHECK_FORM(f32_16x16x64_bf16, 16, 16, 'b', 'b', false, 1, 1);
  CHECK_FORM(f32_32x32x32_bf16, 32, 16, 'b', 'b', false, 0, 1);
  CHECK_FORM(i32_16x16x128_i8, 16, 32, 'c', 'c', true, 0, 1);   // one set: ABID changes nothing
  CHECK_FORM(i32_32x32x64_i8, 32, 32, 'c', 'c', true, 0, 0);
  CHECK_FORM(f32_16x16x128_fp8_bf8, 16, 32, 'e', 'g', false, 0, 0);
  CHECK_FORM(f32_16x16x128_bf8_bf8, 16, 32, 'g', 'g', false, 0, 0);
  CHECK_FORM(f32_32x32x64_fp8_fp8, 32, 32, 'e', 'e', false, 0, 0);
  CHECK_FORM(f32_32x32x64_bf8_fp8, 32, 32, 'g', 'e', false, 0, 0);
#endif
  std::printf("%s\n", wrong ? "FAIL" : "all sparse products right");
  return wrong != 0;
}
