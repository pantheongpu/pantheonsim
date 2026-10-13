// gfx1250 (CDNA 5, MI455X) operand forms of ordinary instructions (forms1250.gfx1250, wave32), each written in inline assembly
// and checked against the host. Found by running every form the assembler accepts through the simulator: the CLAMP and OMOD
// bits on half-precision, packed and 32-bit results; clamped 16-bit and packed integer sums; OPSEL on a scalar source; the
// byte or half of an 8-bit float conversion OPSEL picks; DPP on a three-source (VOP3) instruction; DPP8 on a compare; and
// the VOPD dot products. The oracle is AMD's CDNA5 ISA: 7.1 (CLAMP: "Float arithmetic: clamp result to [0, 1.0]; -0 is clamped
// to +0"; signed and unsigned integers to their range; OMOD scales a half, single or double result by 0.5, 2 or 4 and a
// packed one is not scaled; OPSEL works for VGPR, SGPR and literal sources) and the pseudocode of V_CVT_F32_FP8 / V_CVT_PK_F32_FP8.
// Then instructions whose pseudocode a second reading of the document found the simulator departing from: the "_num" minimum and
// maximum (a number beats a NaN, -0 < +0), v_fmamk_f16, v_cvt_i32_f64 (saturates), v_bfe_i32 and s_bfe_* (an arithmetic shift),
// v_lshl_add_u64 (a count over 4 is zero), the integer dot products' per-source signs, v_cmp_class on a signaling NaN,
// v_cvt_f16_fp8's byte, and the scalar absolute difference, maximum and 64-bit bit tests.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#pragma clang diagnostic ignored "-Winline-asm"

#define CHECK(x)                                                                        \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)
#define U(x) static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<int>(x)))

// One lane per case: a, b are per-lane inputs, o the output.
#define V2(NAME, INSTR)                                                                  \
  __global__ void NAME(const uint32_t* a, const uint32_t* b, uint32_t* o) {              \
    const int i = threadIdx.x;                                                           \
    uint32_t r = 0;                                                                      \
    asm volatile(INSTR : "+v"(r) : "v"(a[i]), "v"(b[i]));                                \
    o[i] = r;                                                                            \
  }
#define V3(NAME, INSTR)                                                                  \
  __global__ void NAME(const uint32_t* a, const uint32_t* b, uint32_t* o) {              \
    const int i = threadIdx.x;                                                           \
    uint32_t r = 0;                                                                      \
    asm volatile(INSTR : "+v"(r) : "v"(a[i]), "v"(b[i]), "v"(a[(i + 5) & 31]));          \
    o[i] = r;                                                                            \
  }
#define V1(NAME, INSTR)                                                                  \
  __global__ void NAME(const uint32_t* a, const uint32_t*, uint32_t* o) {                \
    const int i = threadIdx.x;                                                           \
    uint32_t r = 0;                                                                      \
    asm volatile(INSTR : "+v"(r) : "v"(a[i]));                                           \
    o[i] = r;                                                                            \
  }

V2(k_add_f16_clamp, "v_add_f16 %0, %1, %2 clamp")
V2(k_add_f16_mul2, "v_add_f16 %0, %1, %2 mul:2")
V2(k_add_f16_div2, "v_add_f16 %0, %1, %2 div:2")
V2(k_add_f32_clamp, "v_add_f32 %0, %1, %2 clamp")
V3(k_fma_f32_clamp, "v_fma_f32 %0, %1, %2, %3 clamp")
V1(k_trunc_f32_mul2, "v_trunc_f32 %0, %1 mul:2")
V2(k_pk_add_f16_clamp, "v_pk_add_f16 %0, %1, %2 clamp")
V2(k_pk_add_i16, "v_pk_add_i16 %0, %1, %2")
V2(k_pk_add_i16_clamp, "v_pk_add_i16 %0, %1, %2 clamp")
V2(k_pk_sub_i16, "v_pk_sub_i16 %0, %1, %2")
V2(k_pk_sub_i16_clamp, "v_pk_sub_i16 %0, %1, %2 clamp")
V2(k_add_nc_u16_clamp, "v_add_nc_u16 %0, %1, %2 clamp")
V2(k_sub_nc_u16_clamp, "v_sub_nc_u16 %0, %1, %2 clamp")
V2(k_add_nc_i16_clamp, "v_add_nc_i16 %0, %1, %2 clamp")
V2(k_sub_nc_i16_clamp, "v_sub_nc_i16 %0, %1, %2 clamp")

V1(k_sin_f32, "v_sin_f32 %0, %1")
V1(k_cos_f32, "v_cos_f32 %0, %1")
// rocPRIM's radix sort: min(rank - 256, 256) as one v_add_min_u32 whose first addend is 0xffffff00 (the sum wraps).
V2(k_add_min_u32_wrap, "v_add_min_u32 %0, %1, %2, 256")
// gfx1250's scratch with a VGPR and an SGPR offset (SVS) adds them as signed numbers: here a negative VGPR offset (lane * 4 - 64)
// and a positive SGPR one (64) cancel into the lane's own word. (The volatile array makes the kernel reserve scratch.)
__global__ void k_scratch_svs(const uint32_t* a, const uint32_t*, uint32_t* o) {
  volatile int pad[64];
  pad[threadIdx.x & 63] = 0;
  const int i = threadIdx.x;
  const uint32_t voff = static_cast<uint32_t>(i * 4 - 64), soff = U(64), val = a[i];
  uint32_t r = 0;
  asm volatile("scratch_store_b32 %0, %1, %2\n s_wait_storecnt 0x0" ::"v"(voff), "v"(val), "s"(soff) : "memory");
  asm volatile("scratch_load_b32 %0, %1, %2\n s_wait_loadcnt 0x0" : "=v"(r) : "v"(voff), "s"(soff) : "memory");
  o[i] = r;
}
V2(k_min_num_f32, "v_min_num_f32 %0, %1, %2")
V2(k_max_num_f32, "v_max_num_f32 %0, %1, %2")
V2(k_fmamk_f16, "v_fmamk_f16 %0, %1, 0x4100, %2")
V2(k_dot4_iu8_00, "v_dot4_i32_iu8 %0, %1, %2, 7")
V2(k_dot4_iu8_11, "v_dot4_i32_iu8 %0, %1, %2, 7 neg_lo:[1,1,0]")
V2(k_dot4_iu8_10, "v_dot4_i32_iu8 %0, %1, %2, 7 neg_lo:[1,0,0]")
V2(k_dot4_iu8_01, "v_dot4_i32_iu8 %0, %1, %2, 7 neg_lo:[0,1,0]")
V2(k_dot8_iu4_00, "v_dot8_i32_iu4 %0, %1, %2, 7")
V2(k_dot8_iu4_11, "v_dot8_i32_iu4 %0, %1, %2, 7 neg_lo:[1,1,0]")
V2(k_dot8_iu4_10, "v_dot8_i32_iu4 %0, %1, %2, 7 neg_lo:[1,0,0]")
V2(k_dot8_iu4_01, "v_dot8_i32_iu4 %0, %1, %2, 7 neg_lo:[0,1,0]")
// v_cmp_class_f32: the lane mask (the same for every lane) of a[i] against the classes in b[i].
__global__ void k_class(const uint32_t* a, const uint32_t* b, uint32_t* o) {
  const int i = threadIdx.x;
  uint32_t m = 0;
  asm volatile("v_cmp_class_f32 vcc_lo, %1, %2\n s_mov_b32 %0, vcc_lo" : "=s"(m) : "v"(a[i]), "v"(b[i]) : "vcc");
  o[i] = m;
}
__global__ void k_cvt_i32_f64(const double* a, uint32_t* o) {
  const int i = threadIdx.x;
  uint32_t r = 0;
  asm volatile("v_cvt_i32_f64 %0, %1" : "+v"(r) : "v"(a[i]));
  o[i] = r;
}
__global__ void k_lshl_add_u64(const uint64_t* a, const uint32_t* b, const uint64_t* c, uint64_t* o) {
  const int i = threadIdx.x;
  uint64_t r = 0;
  asm volatile("v_lshl_add_u64 %0, %1, %2, %3" : "+v"(r) : "v"(a[i]), "v"(b[i]), "v"(c[i]));
  o[i] = r;
}
// v_bfe_i32 over (value, start, width) with the start and width taken from the low and high halves of b.
__global__ void k_bfe_i32(const uint32_t* a, const uint32_t* b, uint32_t* o) {
  const int i = threadIdx.x;
  uint32_t r = 0;
  asm volatile("v_bfe_i32 %0, %1, %2, %3" : "+v"(r) : "v"(a[i]), "v"(b[i] & 31), "v"((b[i] >> 16) & 31));
  o[i] = r;
}
// The scalar instructions, one case per iteration (each from lane 0's values): o[k] the result, o[32 + k] the SCC.
__global__ void k_scalar(const uint32_t* a, const uint32_t* b, uint32_t* o) {
  for (int k = 0; k < 32; ++k) {
    const uint32_t x = U(a[k]), y = U(b[k]);
    uint32_t r0, c0, r1, c1, r2, c2, r3, c3;
    asm volatile("s_absdiff_i32 %0, %4, %5\n s_cselect_b32 %1, 1, 0\n"
                 "s_max_i32 %2, %4, %5\n s_cselect_b32 %3, 1, 0" : "=s"(r0), "=s"(c0), "=s"(r1), "=s"(c1) : "s"(x), "s"(y) : "scc");
    asm volatile("s_max_u32 %0, %2, %3\n s_cselect_b32 %1, 1, 0" : "=s"(r2), "=s"(c2) : "s"(x), "s"(y) : "scc");
    const uint64_t wide = (static_cast<uint64_t>(y) << 32) | x;
    asm volatile("s_bitcmp0_b64 %2, %3\n s_cselect_b32 %0, 1, 0\n s_bitcmp1_b64 %2, %3\n s_cselect_b32 %1, 1, 0"
                 : "=s"(r3), "=s"(c3) : "s"(wide), "s"(x) : "scc");
    uint32_t bfe, bfc;
    asm volatile("s_bfe_i32 %0, %2, %3\n s_cselect_b32 %1, 1, 0" : "=s"(bfe), "=s"(bfc) : "s"(x), "s"(y) : "scc");
    uint64_t bfe64;
    asm volatile("s_bfe_i64 %0, %1, %2" : "=s"(bfe64) : "s"(wide), "s"(y) : "scc");
    if (threadIdx.x == 0) {
      o[k * 12 + 0] = r0, o[k * 12 + 1] = c0, o[k * 12 + 2] = r1, o[k * 12 + 3] = c1, o[k * 12 + 4] = r2, o[k * 12 + 5] = c2;
      o[k * 12 + 6] = r3, o[k * 12 + 7] = c3, o[k * 12 + 8] = bfe, o[k * 12 + 9] = bfc;
      o[k * 12 + 10] = static_cast<uint32_t>(bfe64), o[k * 12 + 11] = static_cast<uint32_t>(bfe64 >> 32);
    }
  }
}
// The byte byte_sel picks of v_cvt_f16_fp8, against the plain conversion of that byte moved down.
template <int SEL>
__global__ void k_f16_fp8_byte(const uint32_t* a, const uint32_t*, uint32_t* o) {
  const int i = threadIdx.x;
  uint32_t r = 0, plain = 0;
  if constexpr (SEL == 1) asm volatile("v_cvt_f16_fp8 %0, %1 byte_sel:1" : "+v"(r) : "v"(a[i]));
  if constexpr (SEL == 2) asm volatile("v_cvt_f16_fp8 %0, %1 byte_sel:2" : "+v"(r) : "v"(a[i]));
  if constexpr (SEL == 3) asm volatile("v_cvt_f16_fp8 %0, %1 byte_sel:3" : "+v"(r) : "v"(a[i]));
  const uint32_t moved = a[i] >> (8 * SEL);
  asm volatile("v_cvt_f16_fp8 %0, %1" : "+v"(plain) : "v"(moved));
  o[i] = r & 0xFFFF;
  o[32 + i] = plain & 0xFFFF;
}

// OPSEL on a scalar source: its high half.
__global__ void k_opsel_sgpr(const uint32_t* a, const uint32_t* b, uint32_t* o) {
  const int i = threadIdx.x;
  const uint32_t s = U(a[0]);
  uint32_t r = 0;
  asm volatile("v_add_nc_u16 %0, %1, %2 op_sel:[1,0,0]" : "+v"(r) : "s"(s), "v"(b[i]));
  o[i] = r;
}
// The byte OPSEL picks of an 8-bit float conversion, against the plain conversion of that byte moved down.
template <int SEL>
__global__ void k_fp8_byte(const uint32_t* a, const uint32_t*, uint32_t* o) {
  const int i = threadIdx.x;
  uint32_t r = 0, plain = 0;
  if constexpr (SEL == 0) asm volatile("v_cvt_f32_fp8 %0, %1" : "+v"(r) : "v"(a[i]));
  if constexpr (SEL == 1) asm volatile("v_cvt_f32_fp8_e64 %0, %1 byte_sel:1" : "+v"(r) : "v"(a[i]));
  if constexpr (SEL == 2) asm volatile("v_cvt_f32_fp8_e64 %0, %1 byte_sel:2" : "+v"(r) : "v"(a[i]));
  if constexpr (SEL == 3) asm volatile("v_cvt_f32_fp8_e64 %0, %1 byte_sel:3" : "+v"(r) : "v"(a[i]));
  const uint32_t moved = a[i] >> (8 * SEL);
  asm volatile("v_cvt_f32_fp8 %0, %1" : "+v"(plain) : "v"(moved));
  o[i] = r;
  o[32 + i] = plain;
}
// DPP on a three-source instruction: the first source mirrored within a row of sixteen lanes.
__global__ void k_fma_dpp(const uint32_t* a, const uint32_t* b, uint32_t* o) {
  const int i = threadIdx.x;
  uint32_t r = 0;
  asm volatile("v_fma_f32_e64_dpp %0, %1, %2, %3 row_mirror row_mask:0xf bank_mask:0xf"
               : "+v"(r) : "v"(a[i]), "v"(b[i]), "v"(a[(i + 5) & 31]));
  o[i] = r;
}
// DPP8 on a compare: lane i of each group of eight reads the first source from lane 7 - i of its group.
__global__ void k_cmp_dpp8(const uint32_t* a, const uint32_t* b, uint32_t* o) {
  const int i = threadIdx.x;
  uint32_t m = 0;
  asm volatile("v_cmp_ge_f32_dpp vcc_lo, %1, %2 dpp8:[7,6,5,4,3,2,1,0]\n s_mov_b32 %0, vcc_lo"
               : "=s"(m) : "v"(a[i]), "v"(b[i]) : "vcc");
  o[i] = m;
}
// The VOPD dot product: the destination is also the addend; the other half is a move.
template <bool BF16>
__global__ void k_dual_dot2(const uint32_t* a, const uint32_t* b, uint32_t* o) {
  const int i = threadIdx.x;
  uint32_t acc = 0x3f800000u, mv = 0;   // 1.0f
  if constexpr (!BF16)
    asm volatile("v_dual_dot2acc_f32_f16 %0, %2, %3 :: v_dual_mov_b32 %1, %2" : "+v"(acc), "=v"(mv) : "v"(a[i]), "v"(b[i]));
  else
    asm volatile("v_dual_dot2acc_f32_bf16 %0, %2, %3 :: v_dual_mov_b32 %1, %2" : "+v"(acc), "=v"(mv) : "v"(a[i]), "v"(b[i]));
  o[i] = acc;
  o[32 + i] = mv;
}

static float h2f(uint16_t h) {
  _Float16 x;
  std::memcpy(&x, &h, 2);
  return static_cast<float>(x);
}
static uint16_t f2h(float f) {
  const _Float16 x = static_cast<_Float16>(f);
  uint16_t h;
  std::memcpy(&h, &x, 2);
  return h;
}
static float bf2f(uint16_t h) {
  const uint32_t u = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
static uint32_t fbits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
static float bitsf(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

static int failed = 0;
template <typename K>
static std::vector<uint32_t> launch(K kernel, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, size_t n_out) {
  uint32_t *da = nullptr, *db = nullptr, *dout = nullptr;
  std::vector<uint32_t> out(n_out, 0xDEADBEEFu);
  if (hipMalloc(&da, 128) != hipSuccess || hipMalloc(&db, 128) != hipSuccess || hipMalloc(&dout, n_out * 4) != hipSuccess) return out;
  if (hipMemcpy(da, a.data(), 128, hipMemcpyHostToDevice) != hipSuccess) return out;
  if (hipMemcpy(db, b.data(), 128, hipMemcpyHostToDevice) != hipSuccess) return out;
  hipLaunchKernelGGL(kernel, dim3(1), dim3(32), 0, 0, da, db, dout);
  if (hipDeviceSynchronize() != hipSuccess) return out;
  if (hipMemcpy(out.data(), dout, n_out * 4, hipMemcpyDeviceToHost) != hipSuccess) return out;
  hipFree(da), hipFree(db), hipFree(dout);
  return out;
}
// What the result of lane l should be, checked over all 32 lanes.
template <typename K, typename F>
static void check(const char* what, K kernel, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, F expect,
                  uint32_t mask = 0xFFFFFFFFu) {
  const auto out = launch(kernel, a, b, 32);
  int wrong = 0;
  for (int l = 0; l < 32; ++l) {
    const uint32_t want = expect(l) & mask;
    if ((out[l] & mask) != want) {
      if (wrong < 3) std::printf("  %s lane %d: got %08x want %08x\n", what, l, out[l] & mask, want);
      ++wrong;
    }
  }
  std::printf("%s: %d of 32 wrong\n", what, wrong);
  failed += wrong != 0;
}

int main() {
  // Half-precision inputs: a lane's pair of values from a small table, cycling.
  static const float kv[8] = {-1.0f, -0.5f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 2.0f};
  std::vector<uint32_t> ha(32), hb(32), fa(32), fb(32);
  for (int l = 0; l < 32; ++l) {
    ha[l] = f2h(kv[l & 7]) | static_cast<uint32_t>(f2h(kv[(l * 3 + 1) & 7])) << 16;
    hb[l] = f2h(kv[(l / 4 + l) & 7]) | static_cast<uint32_t>(f2h(kv[(l * 5 + 2) & 7])) << 16;
    fa[l] = fbits(kv[l & 7]);
    fb[l] = fbits(kv[(l / 4 + l) & 7]);
  }
  const auto sat = [](float v) { return v != v || v <= 0.0f ? 0.0f : v > 1.0f ? 1.0f : v; };
  const auto H = [&](const std::vector<uint32_t>& v, int l, int half) { return h2f(static_cast<uint16_t>(v[l] >> (16 * half))); };

  check("v_add_f16 clamp", k_add_f16_clamp, ha, hb, [&](int l) { return f2h(sat(H(ha, l, 0) + H(hb, l, 0))); }, 0xFFFF);
  check("v_add_f16 mul:2", k_add_f16_mul2, ha, hb, [&](int l) { return f2h((H(ha, l, 0) + H(hb, l, 0)) * 2.0f); }, 0xFFFF);
  check("v_add_f16 div:2", k_add_f16_div2, ha, hb, [&](int l) { return f2h((H(ha, l, 0) + H(hb, l, 0)) * 0.5f); }, 0xFFFF);
  check("v_add_f32 clamp (a -0 becomes +0)", k_add_f32_clamp, fa, fb, [&](int l) { return fbits(sat(bitsf(fa[l]) + bitsf(fb[l]))); });
  check("v_fma_f32 clamp", k_fma_f32_clamp, fa, fb, [&](int l) {
    return fbits(sat(__builtin_fmaf(bitsf(fa[l]), bitsf(fb[l]), bitsf(fa[(l + 5) & 31]))));
  });
  check("v_trunc_f32 mul:2", k_trunc_f32_mul2, fa, fb, [&](int l) { return fbits(__builtin_truncf(bitsf(fa[l])) * 2.0f); });
  check("v_pk_add_f16 clamp", k_pk_add_f16_clamp, ha, hb, [&](int l) {
    return f2h(sat(H(ha, l, 0) + H(hb, l, 0))) | static_cast<uint32_t>(f2h(sat(H(ha, l, 1) + H(hb, l, 1)))) << 16;
  });

  // Packed 16-bit integers: lanes cover the edges of the signed range in each half.
  static const int16_t iv[8] = {-32768, -32767, -1, 0, 1, 255, 32766, 32767};
  std::vector<uint32_t> ia(32), ib(32);
  for (int l = 0; l < 32; ++l) {
    ia[l] = static_cast<uint16_t>(iv[l & 7]) | static_cast<uint32_t>(static_cast<uint16_t>(iv[(l * 3 + 1) & 7])) << 16;
    ib[l] = static_cast<uint16_t>(iv[(l / 4 + l) & 7]) | static_cast<uint32_t>(static_cast<uint16_t>(iv[(l * 5 + 2) & 7])) << 16;
  }
  const auto S = [&](const std::vector<uint32_t>& v, int l, int half) { return static_cast<int32_t>(static_cast<int16_t>(v[l] >> (16 * half))); };
  const auto sclamp = [](int32_t v) { return v < -32768 ? -32768 : v > 32767 ? 32767 : v; };
  const auto pk = [](int32_t lo, int32_t hi) { return static_cast<uint16_t>(lo) | static_cast<uint32_t>(static_cast<uint16_t>(hi)) << 16; };
  check("v_pk_add_i16", k_pk_add_i16, ia, ib, [&](int l) { return pk(S(ia, l, 0) + S(ib, l, 0), S(ia, l, 1) + S(ib, l, 1)); });
  check("v_pk_add_i16 clamp", k_pk_add_i16_clamp, ia, ib, [&](int l) {
    return pk(sclamp(S(ia, l, 0) + S(ib, l, 0)), sclamp(S(ia, l, 1) + S(ib, l, 1)));
  });
  check("v_pk_sub_i16", k_pk_sub_i16, ia, ib, [&](int l) { return pk(S(ia, l, 0) - S(ib, l, 0), S(ia, l, 1) - S(ib, l, 1)); });
  check("v_pk_sub_i16 clamp", k_pk_sub_i16_clamp, ia, ib, [&](int l) {
    return pk(sclamp(S(ia, l, 0) - S(ib, l, 0)), sclamp(S(ia, l, 1) - S(ib, l, 1)));
  });
  const auto ulo = [](const std::vector<uint32_t>& v, int l) { return static_cast<int32_t>(v[l] & 0xFFFF); };
  const auto uclamp = [](int32_t v) { return v < 0 ? 0 : v > 65535 ? 65535 : v; };
  check("v_add_nc_u16 clamp", k_add_nc_u16_clamp, ia, ib, [&](int l) { return uclamp(ulo(ia, l) + ulo(ib, l)); }, 0xFFFF);
  check("v_sub_nc_u16 clamp", k_sub_nc_u16_clamp, ia, ib, [&](int l) { return uclamp(ulo(ia, l) - ulo(ib, l)); }, 0xFFFF);
  check("v_add_nc_i16 clamp", k_add_nc_i16_clamp, ia, ib, [&](int l) { return sclamp(S(ia, l, 0) + S(ib, l, 0)); }, 0xFFFF);
  check("v_sub_nc_i16 clamp", k_sub_nc_i16_clamp, ia, ib, [&](int l) { return sclamp(S(ia, l, 0) - S(ib, l, 0)); }, 0xFFFF);

  // OPSEL on a scalar source reads its high half: the scalar is lane 0 of `a`.
  check("v_add_nc_u16 with op_sel on an SGPR", k_opsel_sgpr, ia, ib, [&](int l) { return (ia[0] >> 16) + (ib[l] & 0xFFFF); }, 0xFFFF);

  // The byte OPSEL picks, against the plain conversion of that byte.
  std::vector<uint32_t> ba(32), none(32, 0);
  for (int l = 0; l < 32; ++l) ba[l] = 0x01020304u * (l + 1) ^ (0x11223344u << (l & 7)) ^ static_cast<uint32_t>(l) * 0x9E3779B9u;
  const auto check_byte = [&](const char* what, auto kernel) {
    const auto out = launch(kernel, ba, none, 64);
    int wrong = 0;
    for (int l = 0; l < 32; ++l) wrong += out[l] != out[32 + l] || out[l] == 0xDEADBEEFu;   // (a kernel that never ran leaves the fill)
    std::printf("%s: %d of 32 wrong\n", what, wrong);
    failed += wrong != 0;
  };
  check_byte("v_cvt_f32_fp8 byte 0", k_fp8_byte<0>);
  check_byte("v_cvt_f32_fp8 byte_sel:1", k_fp8_byte<1>);
  check_byte("v_cvt_f32_fp8 byte_sel:2", k_fp8_byte<2>);
  check_byte("v_cvt_f32_fp8 byte_sel:3", k_fp8_byte<3>);

  // DPP on a three-source instruction: lane l reads lane 15 - (l & 15) of its row for the first source.
  check("v_fma_f32 with DPP row_mirror", k_fma_dpp, fa, fb, [&](int l) {
    const int from = (l & ~15) + 15 - (l & 15);
    return fbits(__builtin_fmaf(bitsf(fa[from]), bitsf(fb[l]), bitsf(fa[(l + 5) & 31])));
  });
  // DPP8 on a compare: the mask has bit i set where a[from(i)] >= b[i].
  {
    const auto out = launch(k_cmp_dpp8, fa, fb, 32);
    uint32_t want = 0;
    for (int l = 0; l < 32; ++l) {
      const int from = (l & ~7) + 7 - (l & 7);
      want |= static_cast<uint32_t>(bitsf(fa[from]) >= bitsf(fb[l])) << l;
    }
    int wrong = 0;
    for (int l = 0; l < 32; ++l) wrong += out[l] != want;
    std::printf("v_cmp_ge_f32 with DPP8: %d of 32 wrong (mask %08x want %08x)\n", wrong, out[0], want);
    failed += wrong != 0;
  }
  // The VOPD dot products: acc = 1.0 + a.lo * b.lo + a.hi * b.hi, and the move beside it.
  {
    std::vector<uint32_t> da(32), db(32);
    for (int l = 0; l < 32; ++l) {
      da[l] = f2h(kv[l & 7]) | static_cast<uint32_t>(f2h(kv[(l + 3) & 7])) << 16;
      db[l] = f2h(kv[(l * 3) & 7]) | static_cast<uint32_t>(f2h(kv[(l + 1) & 7])) << 16;
    }
    const auto out = launch(k_dual_dot2<false>, da, db, 64);
    int wrong = 0;
    for (int l = 0; l < 32; ++l) {
      const double sum = 1.0 + static_cast<double>(H(da, l, 0)) * H(db, l, 0) + static_cast<double>(H(da, l, 1)) * H(db, l, 1);
      wrong += out[l] != fbits(static_cast<float>(sum)) || out[32 + l] != da[l];
    }
    std::printf("v_dual_dot2acc_f32_f16 :: v_dual_mov_b32: %d of 32 wrong\n", wrong);
    failed += wrong != 0;
    for (int l = 0; l < 32; ++l) {
      da[l] = static_cast<uint32_t>(fbits(kv[l & 7]) >> 16) | (fbits(kv[(l + 3) & 7]) & 0xFFFF0000u);
      db[l] = static_cast<uint32_t>(fbits(kv[(l * 3) & 7]) >> 16) | (fbits(kv[(l + 1) & 7]) & 0xFFFF0000u);
    }
    const auto out2 = launch(k_dual_dot2<true>, da, db, 64);
    wrong = 0;
    for (int l = 0; l < 32; ++l) {
      const double sum = 1.0 + static_cast<double>(bf2f(da[l] & 0xFFFF)) * bf2f(db[l] & 0xFFFF) +
                         static_cast<double>(bf2f(da[l] >> 16)) * bf2f(db[l] >> 16);
      wrong += out2[l] != fbits(static_cast<float>(sum)) || out2[32 + l] != da[l];
    }
    std::printf("v_dual_dot2acc_f32_bf16 :: v_dual_mov_b32: %d of 32 wrong\n", wrong);
    failed += wrong != 0;
  }


  // ---- A second reading of the document ----------------------------------------------------------------------------------
  // v_min_num_f32 / v_max_num_f32 (15.14): a number beats a NaN; two NaNs give the first, quieted; -0 is below +0.
  {
    static const uint32_t tab[6] = {0x00000000u, 0x80000000u, 0x3f800000u, 0x7f800001u, 0x7fc00000u, 0xbf800000u};
    std::vector<uint32_t> a(32), b(32);
    for (int l = 0; l < 32; ++l) a[l] = tab[l % 6], b[l] = tab[(l / 6) % 6];
    const auto isnan = [](uint32_t v) { return (v & 0x7f800000u) == 0x7f800000u && (v & 0x7fffffu) != 0; };
    const auto bothzero = [](uint32_t x, uint32_t y) { return (x << 1) == 0 && (y << 1) == 0; };
    const auto lt = [&](uint32_t x, uint32_t y) { return bitsf(x) < bitsf(y); };
    check("v_min_num_f32", k_min_num_f32, a, b, [&](int l) {
      const uint32_t x = a[l], y = b[l];
      if (isnan(x) && isnan(y)) return x | 0x400000u;
      if (isnan(x)) return y;
      if (isnan(y)) return x;
      return lt(x, y) || (bothzero(x, y) && (x >> 31) && !(y >> 31)) ? x : y;
    });
    check("v_max_num_f32", k_max_num_f32, a, b, [&](int l) {
      const uint32_t x = a[l], y = b[l];
      if (isnan(x) && isnan(y)) return x | 0x400000u;
      if (isnan(x)) return y;
      if (isnan(y)) return x;
      return lt(y, x) || (bothzero(x, y) && !(x >> 31) && (y >> 31)) ? x : y;
    });
  }
  // v_fmamk_f16 d, a, K, b = a * K + b (the constant is the second factor).
  check("v_fmamk_f16", k_fmamk_f16, ha, hb, [&](int l) {
    return f2h(static_cast<float>(static_cast<double>(H(ha, l, 0)) * 2.5 + static_cast<double>(H(hb, l, 0))));
  }, 0xFFFF);
  // The integer dot products: NEG[0] and NEG[1] (neg_lo) say whether each source is signed; unsigned without them.
  {
    std::vector<uint32_t> a(32), b(32);
    uint32_t seed = 0x1234567u;
    for (int l = 0; l < 32; ++l) {
      seed = seed * 1664525u + 1013904223u, a[l] = seed;
      seed = seed * 1664525u + 1013904223u, b[l] = seed;
    }
    a[0] = b[0] = 0xFFFFFFFFu;   // all ones: 255 unsigned, -1 signed
    const auto dot = [&](int l, int bits, bool sa, bool sb) {
      int64_t sum = 7;
      for (int k = 0; k < 32 / bits; ++k) {
        const uint32_t m = (1u << bits) - 1, ua = (a[l] >> (bits * k)) & m, ub = (b[l] >> (bits * k)) & m;
        const int64_t x = sa && (ua >> (bits - 1)) ? int64_t{ua} - (1 << bits) : ua, y = sb && (ub >> (bits - 1)) ? int64_t{ub} - (1 << bits) : ub;
        sum += x * y;
      }
      return static_cast<uint32_t>(sum);
    };
    check("v_dot4_i32_iu8 unsigned", k_dot4_iu8_00, a, b, [&](int l) { return dot(l, 8, false, false); });
    check("v_dot4_i32_iu8 neg_lo:[1,1,0]", k_dot4_iu8_11, a, b, [&](int l) { return dot(l, 8, true, true); });
    check("v_dot4_i32_iu8 neg_lo:[1,0,0]", k_dot4_iu8_10, a, b, [&](int l) { return dot(l, 8, true, false); });
    check("v_dot4_i32_iu8 neg_lo:[0,1,0]", k_dot4_iu8_01, a, b, [&](int l) { return dot(l, 8, false, true); });
    check("v_dot8_i32_iu4 unsigned", k_dot8_iu4_00, a, b, [&](int l) { return dot(l, 4, false, false); });
    check("v_dot8_i32_iu4 neg_lo:[1,1,0]", k_dot8_iu4_11, a, b, [&](int l) { return dot(l, 4, true, true); });
    check("v_dot8_i32_iu4 neg_lo:[1,0,0]", k_dot8_iu4_10, a, b, [&](int l) { return dot(l, 4, true, false); });
    check("v_dot8_i32_iu4 neg_lo:[0,1,0]", k_dot8_iu4_01, a, b, [&](int l) { return dot(l, 4, false, true); });
  }
  // v_cmp_class_f32 (bit 0 a signaling NaN, bit 1 a quiet one, then -inf, -normal, -denormal, -0, +0, +denormal, +normal, +inf).
  {
    static const uint32_t tab[16] = {0x7f800001u, 0xff800001u, 0x7fc00000u, 0xffc00001u, 0xff800000u, 0xbf800000u, 0x80000001u, 0x80000000u,
                                     0x00000000u, 0x00000001u, 0x3f800000u, 0x7f800000u, 0x7fa00000u, 0x7fffffffu, 0xffbfffffu, 0x00400000u};
    const auto cls = [](uint32_t v) {
      const bool neg = v >> 31, expo = (v & 0x7f800000u) == 0x7f800000u, den = (v & 0x7f800000u) == 0, mant = (v & 0x7fffffu) != 0;
      if (expo && mant) return (v & 0x400000u) ? 1 : 0;
      if (expo) return neg ? 2 : 9;
      if (den && !mant) return neg ? 5 : 6;
      if (den) return neg ? 4 : 7;
      return neg ? 3 : 8;
    };
    std::vector<uint32_t> a(32), b(32);
    for (int l = 0; l < 32; ++l) a[l] = tab[l % 16], b[l] = l < 16 ? 1u << cls(a[l]) : 0x3FFu & ~(1u << cls(a[l]));
    uint32_t want = 0;
    for (int l = 0; l < 32; ++l) want |= ((b[l] >> cls(a[l])) & 1u) << l;
    check("v_cmp_class_f32 on NaNs of both kinds", k_class, a, b, [&](int) { return want; });
  }
  // v_cvt_i32_f64: out-of-range values, infinities saturate and a NaN is zero.
  {
    static const double dv[16] = {1e10, -1e10, __builtin_inf(), -__builtin_inf(), __builtin_nan(""), 2147483647.0, 2147483648.0, -2147483648.0,
                                  -2147483649.0, 0.0, 1.0, -1.0, 123456789.0, -123456789.0, 4294967296.0, -0.0};
    double* da = nullptr;
    uint32_t* dout = nullptr;
    std::vector<double> in(32);
    for (int l = 0; l < 32; ++l) in[l] = dv[l % 16];
    std::vector<uint32_t> out(32, 0xDEADBEEFu);
    CHECK(hipMalloc(&da, 256));
    CHECK(hipMalloc(&dout, 128));
    CHECK(hipMemcpy(da, in.data(), 256, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_cvt_i32_f64, dim3(1), dim3(32), 0, 0, da, dout);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(out.data(), dout, 128, hipMemcpyDeviceToHost));
    int wrong = 0;
    for (int l = 0; l < 32; ++l) {
      const double d = in[l];
      const uint32_t want = d != d ? 0u : d >= 2147483647.0 ? 0x7fffffffu : d <= -2147483648.0 ? 0x80000000u : static_cast<uint32_t>(static_cast<int32_t>(d));
      wrong += out[l] != want;
    }
    std::printf("v_cvt_i32_f64: %d of 32 wrong\n", wrong);
    failed += wrong != 0;
  }
  // v_lshl_add_u64: the count is S1[2:0] and a count over 4 is a shift of zero.
  {
    uint64_t *da = nullptr, *dc = nullptr, *dout = nullptr;
    uint32_t* db = nullptr;
    std::vector<uint64_t> a(32), c(32), out(32, ~0ull);
    std::vector<uint32_t> b(32);
    for (int l = 0; l < 32; ++l) a[l] = 0x0123456789abcdefull * (l + 3), c[l] = 0xfedcba9876543210ull ^ (l * 0x9e3779b97f4a7c15ull), b[l] = l + (l >> 3) * 0x100;
    CHECK(hipMalloc(&da, 256));
    CHECK(hipMalloc(&db, 128));
    CHECK(hipMalloc(&dc, 256));
    CHECK(hipMalloc(&dout, 256));
    CHECK(hipMemcpy(da, a.data(), 256, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db, b.data(), 128, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dc, c.data(), 256, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_lshl_add_u64, dim3(1), dim3(32), 0, 0, da, db, dc, dout);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(out.data(), dout, 256, hipMemcpyDeviceToHost));
    int wrong = 0;
    for (int l = 0; l < 32; ++l) {
      const uint32_t n = b[l] & 7, shift = n > 4 ? 0 : n;
      wrong += out[l] != (a[l] << shift) + c[l];
    }
    std::printf("v_lshl_add_u64: %d of 32 wrong\n", wrong);
    failed += wrong != 0;
  }
  // v_bfe_i32: the shift is arithmetic, so a field reaching past bit 31 reads the sign there.
  {
    std::vector<uint32_t> a(32), b(32);
    uint32_t seed = 99;
    for (int l = 0; l < 32; ++l) {
      seed = seed * 1664525u + 1013904223u, a[l] = seed;
      b[l] = static_cast<uint32_t>(l) | (static_cast<uint32_t>((l * 7 + 3) % 32) << 16);
    }
    a[0] = 0xFFFFFFFFu, b[0] = 2u | (31u << 16);
    check("v_bfe_i32", k_bfe_i32, a, b, [&](int l) {
      const uint32_t start = b[l] & 31, width = (b[l] >> 16) & 31;
      if (width == 0) return 0u;
      const uint32_t tmp = static_cast<uint32_t>(static_cast<int32_t>(a[l]) >> start) & ((1u << width) - 1);
      return static_cast<uint32_t>(static_cast<int32_t>(tmp << (32 - width)) >> (32 - width));
    });
  }
  // The scalar instructions: absolute difference (in 32 bits), maximum (SCC set where the first source is kept, a tie included),
  // the 64-bit bit tests, and the signed bitfield extracts (an arithmetic shift).
  {
    std::vector<uint32_t> a(32), b(32);
    uint32_t seed = 4242;
    for (int l = 0; l < 32; ++l) {
      seed = seed * 1664525u + 1013904223u, a[l] = seed;
      seed = seed * 1664525u + 1013904223u, b[l] = (seed & 63) | (((seed >> 8) % 70) << 16);
    }
    a[0] = 1, b[0] = 0x80000000u;
    a[1] = 0xFFFFFFFFu, b[1] = 2u | (31u << 16);
    a[2] = 5, b[2] = 5;
    a[3] = 0x80000000u, b[3] = 0x80000000u;
    uint32_t *da = nullptr, *db = nullptr, *dout = nullptr;
    std::vector<uint32_t> out(32 * 12, 0xDEADBEEFu);
    CHECK(hipMalloc(&da, 128));
    CHECK(hipMalloc(&db, 128));
    CHECK(hipMalloc(&dout, out.size() * 4));
    CHECK(hipMemcpy(da, a.data(), 128, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db, b.data(), 128, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_scalar, dim3(1), dim3(32), 0, 0, da, db, dout);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
    int wrong[6] = {};
    for (int k = 0; k < 32; ++k) {
      const uint32_t x = a[k], y = b[k], *o = &out[k * 12];
      uint32_t d = x - y;
      if (static_cast<int32_t>(d) < 0) d = 0u - d;
      wrong[0] += o[0] != d || o[1] != (d != 0);
      const bool smax = static_cast<int32_t>(x) >= static_cast<int32_t>(y), umax = x >= y;
      wrong[1] += o[2] != (smax ? x : y) || o[3] != smax;
      wrong[2] += o[4] != (umax ? x : y) || o[5] != umax;
      const uint64_t wide = (static_cast<uint64_t>(y) << 32) | x;
      const bool bit = (wide >> (x & 63)) & 1;
      wrong[3] += o[6] != !bit || o[7] != bit;
      {
        const uint32_t start = y & 31, width = (y >> 16) & 0x7F;
        uint32_t want = 0;
        if (width) {
          const uint32_t sh = static_cast<uint32_t>(static_cast<int32_t>(x) >> start);
          want = width >= 32 ? sh : static_cast<uint32_t>(static_cast<int32_t>((sh & ((1u << width) - 1)) << (32 - width)) >> (32 - width));
        }
        wrong[4] += o[8] != want || o[9] != (want != 0);
      }
      {
        const uint32_t start = y & 63, width = (y >> 16) & 0x7F;
        uint64_t want = 0;
        if (width) {
          const uint64_t sh = static_cast<uint64_t>(static_cast<int64_t>(wide) >> start);
          want = width >= 64 ? sh : static_cast<uint64_t>(static_cast<int64_t>((sh & ((1ull << width) - 1)) << (64 - width)) >> (64 - width));
        }
        wrong[5] += o[10] != static_cast<uint32_t>(want) || o[11] != static_cast<uint32_t>(want >> 32);
      }
    }
    static const char* names[6] = {"s_absdiff_i32", "s_max_i32 (SCC set on a tie)", "s_max_u32 (SCC set on a tie)", "s_bitcmp0_b64 / s_bitcmp1_b64",
                                   "s_bfe_i32", "s_bfe_i64"};
    for (int i = 0; i < 6; ++i) {
      std::printf("%s: %d of 32 wrong\n", names[i], wrong[i]);
      failed += wrong[i] != 0;
    }
  }
  // v_sin_f32 and v_cos_f32 over the full range of a float: the examples the document gives (a large float is a whole number of
  // turns), and infinities giving 0xffc00000.
  {
    static const uint32_t in[8] = {0xff800000u, 0xff7fffffu, 0x7f7fffffu, 0x80000000u, 0x3e800000u, 0x7f800000u, 0x4b000001u, 0x00000000u};
    static const uint32_t sin_want[8] = {0xffc00000u, 0x00000000u, 0x00000000u, 0x80000000u, 0x3f800000u, 0xffc00000u, 0x00000000u, 0x00000000u};
    static const uint32_t cos_want[8] = {0xffc00000u, 0x3f800000u, 0x3f800000u, 0x3f800000u, 0x00000000u, 0xffc00000u, 0x3f800000u, 0x3f800000u};
    std::vector<uint32_t> a(32);
    for (int l = 0; l < 32; ++l) a[l] = in[l & 7];
    // (cos(0.25 turns) is zero to the last place a float shows; allow the tiny residue by comparing magnitudes below 1e-6.)
    check("v_sin_f32 at the document's examples", k_sin_f32, a, a, [&](int l) { return sin_want[l & 7]; });
    const auto out = launch(k_cos_f32, a, a, 32);
    int wrong = 0;
    for (int l = 0; l < 32; ++l) {
      const uint32_t want = cos_want[l & 7];
      wrong += (l & 7) == 4 ? !(std::fabs(bitsf(out[l])) < 1e-6f) : out[l] != want;
    }
    std::printf("v_cos_f32 at the document's examples: %d of 32 wrong\n", wrong);
    failed += wrong != 0;
  }
  // v_add_min_u32 wraps its sum (the Expression is v_min_u32(v_add_nc_u32(...)), and AMD's compiler emits it for
  // `min(rank - 256, 256)`): rocPRIM's onesweep radix sort depends on it. Ranks over 256 carry out of 32 bits.
  {
    std::vector<uint32_t> a(32), b(32, 0xFFFFFF00u);
    for (int l = 0; l < 32; ++l) a[l] = 250 + l * 3;
    check("v_add_min_u32 (the sum wraps)", k_add_min_u32_wrap, a, b, [&](int l) {
      const uint32_t sum = a[l] + b[l];
      return sum < 256u ? sum : 256u;
    });
  }
  // Scratch with signed VGPR + SGPR offsets: each lane gets back what it stored.
  check("scratch_store_b32 / scratch_load_b32 with a VGPR and an SGPR offset", k_scratch_svs, ia, ib, [&](int l) { return ia[l]; });
  // The byte byte_sel picks of v_cvt_f16_fp8, against the plain conversion of that byte.
  {
    const auto check_f16_byte = [&](const char* what, auto kernel) {
      const auto out = launch(kernel, ba, none, 64);
      int wrong = 0;
      for (int l = 0; l < 32; ++l) wrong += out[l] != out[32 + l] || out[l] == 0xDEADBEEFu;
      std::printf("%s: %d of 32 wrong\n", what, wrong);
      failed += wrong != 0;
    };
    check_f16_byte("v_cvt_f16_fp8 byte_sel:1", k_f16_fp8_byte<1>);
    check_f16_byte("v_cvt_f16_fp8 byte_sel:2", k_f16_fp8_byte<2>);
    check_f16_byte("v_cvt_f16_fp8 byte_sel:3", k_f16_fp8_byte<3>);
  }

  std::printf("forms1250: %d failed\n", failed);
  return failed != 0;
}
