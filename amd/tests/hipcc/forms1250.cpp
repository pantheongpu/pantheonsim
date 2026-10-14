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
// MODE holds s_set_vgpr_msb's eight bits (DST 13:12, SRC0 15:14, SRC1 17:16, SRC2 19:18), so writing MODE moves the registers the
// next instructions name: written to v773 with MODE's DST bits set to 3, then read back with s_set_vgpr_msb's SRC0 = 3. v5 must not
// have been touched, and reading MODE after s_set_vgpr_msb shows the same bits. The scalar prefetches do nothing.
__global__ void k_mode_msb(const uint32_t* a, const uint32_t*, uint32_t* o) {
  const int i = threadIdx.x;
  uint32_t r = 0, plain = 0, mode = 0;
  const uint32_t x = a[i];
  asm volatile(
      "s_prefetch_inst_pc_rel 0x0, null, 0\n"
      "v_mov_b32 v5, 0xAAAA\n"
      "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 12, 2), 3\n"
      "v_mov_b32 v5, %2\n"
      "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 12, 2), 0\n"
      "s_set_vgpr_msb 0x03\n"
      "v_mov_b32 %0, v5\n"
      "s_set_vgpr_msb 0x1B\n"
      "s_getreg_b32 %3, hwreg(HW_REG_MODE, 12, 8)\n"
      "s_set_vgpr_msb 0\n"
      "v_mov_b32 %1, v5\n"
      : "=&v"(r), "=&v"(plain), "+v"(const_cast<uint32_t&>(x)), "=&s"(mode)
      :
      : "v5");
  o[i] = r;
  o[32 + i] = plain;
  o[64 + i] = mode;
}
__global__ void k_scalar_prefetch(const uint32_t* a, const uint32_t*, uint32_t* o) {
  const uint64_t addr = reinterpret_cast<uint64_t>(a);
  const uint64_t base = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr)) | static_cast<uint64_t>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr >> 32))) << 32;
  asm volatile("s_wait_xcnt 0x0\n s_prefetch_data %0, 0x0, null, 0\n s_wait_xcnt 0x0\n s_prefetch_inst %0, 0x40, null, 1" ::"s"(base));
  o[threadIdx.x] = a[threadIdx.x] + 1;
}
// global_load_tr6_b96 and global_load_tr4_b64: lane s supplies the address of its column of the tile (16 elements along M).
typedef unsigned v3u __attribute__((ext_vector_type(3)));
typedef unsigned v2u __attribute__((ext_vector_type(2)));
__global__ void k_tr6(const uint8_t* mem, uint32_t* o) {
  const int i = threadIdx.x;
  v3u r = {0, 0, 0};
  const uint8_t* p = mem + 12 * i;
  asm volatile("global_load_tr6_b96 %0, %1, off\n s_wait_loadcnt 0x0" : "=&v"(r) : "v"(p) : "memory");
  o[3 * i + 0] = r[0], o[3 * i + 1] = r[1], o[3 * i + 2] = r[2];
}
__global__ void k_tr4(const uint8_t* mem, uint32_t* o) {
  const int i = threadIdx.x;
  v2u r = {0, 0};
  const uint8_t* p = mem + 8 * i;
  asm volatile("global_load_tr4_b64 %0, %1, off\n s_wait_loadcnt 0x0" : "=&v"(r) : "v"(p) : "memory");
  o[2 * i + 0] = r[0], o[2 * i + 1] = r[1];
}
V1(k_cvt_pk_i16, "v_cvt_pk_i16_f32 %0, %1, 4.0")
V2(k_cvt_pk_u16, "v_cvt_pk_u16_f32 %0, %1, %2")
// gfx1250's LDS barrier atomics: a 64-bit word whose state is a pending count (low bits) and a phase (above), the initial count in bits
// 47:32. All 32 lanes arrive at one barrier initialised to 31 (so the last arrival is the 32nd, which goes below zero) with the _RTN form,
// which returns the word as it was, and at another with the async form (an update of 1). Each lane's returned pending count is 31 - lane;
// afterwards both words are met: the pending count reloaded from the initial count, the phase (the borrow) one lower (assumes WIDTH = 31).
__global__ void k_lds_barrier(const uint32_t*, const uint32_t*, uint32_t* o) {
  __shared__ uint64_t bar[2];
  const int i = threadIdx.x;
  if (i == 0) bar[0] = bar[1] = (31ull << 32) | 31ull;
  __syncthreads();
  const uint32_t a0 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>((__attribute__((address_space(3))) uint64_t*)&bar[0]));
  const uint32_t a1 = a0 + 8;
  uint64_t ret = 0;
  asm volatile("ds_atomic_barrier_arrive_rtn_b64 %0, %1, %2\n s_wait_dscnt 0x0" : "=v"(ret) : "v"(a0), "v"(static_cast<uint64_t>(1)) : "memory");
  asm volatile("ds_atomic_async_barrier_arrive_b64 %0\n s_wait_asynccnt 0x0" ::"v"(a1) : "memory");
  __syncthreads();
  o[2 * i] = static_cast<uint32_t>(ret);
  o[2 * i + 1] = static_cast<uint32_t>(ret >> 32);
  if (i == 0) {
    o[64] = static_cast<uint32_t>(bar[0]), o[65] = static_cast<uint32_t>(bar[0] >> 32);
    o[66] = static_cast<uint32_t>(bar[1]), o[67] = static_cast<uint32_t>(bar[1] >> 32);
  }
}

// MODE.FP16_OVFL (bit 23): with it set an overflow to infinity from finite operands is the largest finite value; infinite operands stay
// infinite. Each lane adds a[l] + b[l] in half precision, and converts two floats to fp8, with the bit set and then clear.
__global__ void k_fp16_ovfl(const uint32_t* a, const uint32_t* b, uint32_t* o) {
  const int i = threadIdx.x;
  uint32_t on = 0, off = 0, c_on = 0, c_off = 0, f_on = 0, f_off = 0;
  asm volatile(
      "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1\n"
      "v_add_f16 %0, %6, %7\n"
      "v_cvt_pk_fp8_f32 %2, %8, %9\n"
      "v_cvt_f16_f32 %4, %8\n"
      "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 0\n"
      "v_add_f16 %1, %6, %7\n"
      "v_cvt_pk_fp8_f32 %3, %8, %9\n"
      "v_cvt_f16_f32 %5, %8\n"
      : "=&v"(on), "=&v"(off), "=&v"(c_on), "=&v"(c_off), "=&v"(f_on), "=&v"(f_off)
      : "v"(a[i]), "v"(b[i]), "v"(0x447A0000u), "v"(0x7F800000u));   // 1000.0f and +infinity
  o[i] = on & 0xFFFF;
  o[32 + i] = off & 0xFFFF;
  o[64 + i] = c_on & 0xFFFF;
  o[96 + i] = c_off & 0xFFFF;
  o[128 + i] = f_on & 0xFFFF;
  o[160 + i] = f_off & 0xFFFF;
}
// v_cvt_scale_*: 4-, 6- and 8-bit floats widened and scaled by a block scale; the kernels differ in scale_sel.
typedef unsigned v2u __attribute__((ext_vector_type(2)));
typedef unsigned v4u __attribute__((ext_vector_type(4)));
typedef unsigned v3u __attribute__((ext_vector_type(3)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef float v16f __attribute__((ext_vector_type(16)));
#define SCALE_F4_F32(N) \
  __global__ void k_sc_f4_f32_##N(const uint32_t* s0, const uint32_t* s1, uint32_t* o) { \
    const int i = threadIdx.x; v8f r; \
    asm volatile("v_cvt_scale_pk8_f32_fp4 %0, %1, %2 scale_sel:" #N : "=&v"(r) : "v"(s0[i]), "v"(s1[i])); \
    *reinterpret_cast<v8f*>(o + 8 * i) = r; }
SCALE_F4_F32(0) SCALE_F4_F32(1) SCALE_F4_F32(2) SCALE_F4_F32(3) SCALE_F4_F32(4) SCALE_F4_F32(5) SCALE_F4_F32(6) SCALE_F4_F32(7)
#define SCALE_F8_F16(N) \
  __global__ void k_sc_f8_f16_##N(const uint32_t* s0, const uint32_t* s1, uint32_t* o) { \
    const int i = threadIdx.x; v4u r; v2u in = {s0[2 * i], s0[2 * i + 1]}; \
    asm volatile("v_cvt_scale_pk8_f16_fp8 %0, %1, %2 scale_sel:" #N : "=&v"(r) : "v"(in), "v"(s1[i])); \
    for (int k = 0; k < 4; ++k) o[4 * i + k] = r[k]; }
SCALE_F8_F16(0) SCALE_F8_F16(1) SCALE_F8_F16(2) SCALE_F8_F16(3) SCALE_F8_F16(4) SCALE_F8_F16(5) SCALE_F8_F16(6) SCALE_F8_F16(7)
SCALE_F8_F16(8) SCALE_F8_F16(9) SCALE_F8_F16(10) SCALE_F8_F16(11)
#define SCALE_F6_F32(N) \
  __global__ void k_sc_f6_f32_##N(const uint32_t* s0, const uint32_t* s1, uint32_t* o) { \
    const int i = threadIdx.x; v16f r; v3u in = {s0[3 * i], s0[3 * i + 1], s0[3 * i + 2]}; \
    asm volatile("v_cvt_scale_pk16_f32_fp6 %0, %1, %2 scale_sel:" #N : "=&v"(r) : "v"(in), "v"(s1[i])); \
    *reinterpret_cast<v16f*>(o + 16 * i) = r; }
SCALE_F6_F32(0) SCALE_F6_F32(1) SCALE_F6_F32(2) SCALE_F6_F32(4) SCALE_F6_F32(5) SCALE_F6_F32(7)
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
  // MODE and the VGPR MSB bits, and the scalar prefetches.
  {
    std::vector<uint32_t> a(32);
    for (int l = 0; l < 32; ++l) a[l] = 0x1000u + l;
    const auto out = launch(k_mode_msb, a, a, 96);
    int wrong = 0;
    for (int l = 0; l < 32; ++l) wrong += out[l] != a[l] || out[32 + l] != 0xAAAAu || out[64 + l] != 0x6Cu;
    std::printf("s_setreg of MODE's VGPR MSB bits and s_getreg: %d of 32 wrong (mode field %02x, want 6c)\n", wrong, out[64]);
    failed += wrong != 0;
    const auto out2 = launch(k_scalar_prefetch, a, a, 32);
    wrong = 0;
    for (int l = 0; l < 32; ++l) wrong += out2[l] != a[l] + 1;
    std::printf("s_prefetch_data, s_prefetch_inst, s_prefetch_inst_pc_rel: %d of 32 wrong\n", wrong);
    failed += wrong != 0;
  }
  // v_cvt_pk_i16_f32 / v_cvt_pk_u16_f32: toward zero, saturating, a NaN is zero (assumed: the document's helpers are undefined).
  {
    static const float fv[16] = {0.0f, 1.9f, -1.9f, 40000.0f, -40000.0f, 70000.0f, -0.5f, 32767.5f, __builtin_nanf(""), __builtin_inff(), -__builtin_inff(), 65535.9f, -32768.9f, 123.0f, 2.5f, -2.5f};
    std::vector<uint32_t> a(32), b(32);
    for (int l = 0; l < 32; ++l) a[l] = fbits(fv[l % 16]), b[l] = fbits(fv[(l * 5 + 3) % 16]);
    const auto si = [](float f) -> uint32_t { return f != f ? 0u : static_cast<uint16_t>(static_cast<int16_t>(f >= 32767.0f ? 32767.0 : f <= -32768.0f ? -32768.0 : __builtin_trunc(f))); };
    const auto su = [](float f) -> uint32_t { return f != f ? 0u : static_cast<uint16_t>(f >= 65535.0f ? 65535.0 : f <= 0.0f ? 0.0 : __builtin_trunc(f)); };
    check("v_cvt_pk_i16_f32 (second source the constant 4.0)", k_cvt_pk_i16, a, a, [&](int l) { return si(bitsf(a[l])) | si(4.0f) << 16; });
    check("v_cvt_pk_u16_f32", k_cvt_pk_u16, a, b, [&](int l) { return su(bitsf(a[l])) | su(bitsf(b[l])) << 16; });
  }
  // The 6- and 4-bit transposed loads, from the figure's source lanes (see the executor): lane o ends with M = o % 16 and the
  // 16 elements K = 16 * (o / 16) + j, each taken from the lane the figure gives for that K.
  {
    uint8_t* dmem = nullptr;
    uint32_t* dout = nullptr;
    CHECK(hipMalloc(&dmem, 512));
    CHECK(hipMalloc(&dout, 512));
    for (int bits : {6, 4}) {
      const int bytes = bits == 6 ? 12 : 8;
      std::vector<uint8_t> mem(32 * bytes);
      uint32_t seed = 7 + bits;
      for (auto& b : mem) seed = seed * 1664525u + 1013904223u, b = static_cast<uint8_t>(seed >> 24);
      std::vector<uint32_t> out(32 * (bits == 6 ? 3 : 2), 0xDEADBEEFu);
      CHECK(hipMemcpy(dmem, mem.data(), mem.size(), hipMemcpyHostToDevice));
      if (bits == 6) hipLaunchKernelGGL(k_tr6, dim3(1), dim3(32), 0, 0, dmem, dout);
      else hipLaunchKernelGGL(k_tr4, dim3(1), dim3(32), 0, 0, dmem, dout);
      CHECK(hipDeviceSynchronize());
      CHECK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
      static const int base6[8] = {0, 8, 4, 12, 16, 24, 20, 28}, base4[4] = {0, 16, 8, 24};
      const auto elem = [&](int lane, int m) -> uint32_t {   // element m of a source lane's bytes
        const int at = m * bits;
        uint32_t v = mem[lane * bytes + at / 8] >> (at % 8);
        if (at % 8 + bits > 8) v |= static_cast<uint32_t>(mem[lane * bytes + at / 8 + 1]) << (8 - at % 8);
        return v & ((1u << bits) - 1);
      };
      int wrong = 0;
      for (int o = 0; o < 32; ++o)
        for (int j = 0; j < 16; ++j) {
          const int k = 16 * (o / 16) + j, src = bits == 6 ? base6[k / 4] + k % 4 : base4[k / 8] + k % 8;
          const int at = j * bits, w0 = o * (bits == 6 ? 3 : 2);
          uint32_t v = out[w0 + at / 32] >> (at % 32);
          if (at % 32 + bits > 32) v |= out[w0 + at / 32 + 1] << (32 - at % 32);
          wrong += (v & ((1u << bits) - 1)) != elem(src, o % 16);
        }
      std::printf("global_load_tr%d_b%d: %d of 512 wrong\n", bits, bits == 6 ? 96 : 64, wrong);
      failed += wrong != 0;
    }
  }
  // The LDS barrier atomics (see the kernel).
  {
    const auto out = launch(k_lds_barrier, ia, ib, 68);
    int wrong = 0;
    for (int l = 0; l < 32; ++l) wrong += out[2 * l] != static_cast<uint32_t>(31 - l) || out[2 * l + 1] != 31u;
    for (int w = 0; w < 2; ++w) wrong += out[64 + 2 * w] != 0x8000001Fu || out[65 + 2 * w] != 31u;
    std::printf("ds_atomic_barrier_arrive_rtn_b64 and ds_atomic_async_barrier_arrive_b64: %d of 68 wrong\n", wrong);
    failed += wrong != 0;
  }
  // MODE.FP16_OVFL.
  {
    static const float av[8] = {60000.0f, -60000.0f, __builtin_inff(), 1.0f, 65504.0f, 30000.0f, -__builtin_inff(), 3.0f};
    static const float bv[8] = {60000.0f, -60000.0f, 1.0f, 2.0f, 1.0f, 30000.0f, -5.0f, __builtin_inff()};
    std::vector<uint32_t> a(32), b(32);
    for (int l = 0; l < 32; ++l) a[l] = f2h(av[l & 7]), b[l] = f2h(bv[l & 7]);
    const auto out = launch(k_fp16_ovfl, a, b, 192);
    int wrong = 0;
    for (int l = 0; l < 32; ++l) {
      const float sum = h2f(a[l]) + h2f(b[l]);
      const bool overflow = std::isfinite(h2f(a[l])) && std::isfinite(h2f(b[l])) && !std::isfinite(static_cast<float>(static_cast<_Float16>(sum)));
      const uint32_t plain = f2h(sum), clamped = overflow ? (sum < 0 ? 0xFBFFu : 0x7BFFu) : plain;
      wrong += out[l] != clamped || out[32 + l] != plain;
      // 1000.0f and +inf into fp8 (E4M3: largest 448, NaN 0x7F): the bit set gives the largest for both, clear gives NaN for both;
      // 1000.0f to half is an ordinary 1000 either way.
      wrong += out[64 + l] != 0x7E7Eu || out[96 + l] != 0x7F7Fu || out[128 + l] != f2h(1000.0f) || out[160 + l] != f2h(1000.0f);
    }
    std::printf("MODE.FP16_OVFL on v_add_f16, v_cvt_pk_fp8_f32 and v_cvt_f16_f32: %d of 32 wrong\n", wrong);
    failed += wrong != 0;
  }
  // v_cvt_scale_pk*: Tables 36 and 37 of the document, transcribed here as arrays (scale byte, and whether it is from lanes 0-15 or 16-31
  // of S1), against the executor's formulas for them.
  {
    struct Sel { int lanes16, byte; };   // lanes16: 0 reads lanes 0-15 of S1, 1 reads lanes 16-31
    // Table 36 (4- and 6-bit), [OPSEL[2:0]][half of the data lanes]
    static const Sel t36[8][2] = {{{0, 0}, {0, 1}}, {{1, 0}, {1, 1}}, {{0, 2}, {0, 3}}, {{1, 2}, {1, 3}},
                                  {{0, 0}, {0, 2}}, {{1, 0}, {1, 2}}, {{0, 1}, {0, 3}}, {{1, 1}, {1, 3}}};
    // Table 37 (8-bit), [OPSEL[3:0]] for OPSEL[3] = 0: both halves alike; for 1: OPSEL[2] = 0 only (rows 000 and 001 alike, 010 and 011 alike)
    static const Sel t37_32[8] = {{0, 0}, {1, 0}, {0, 2}, {1, 2}, {0, 1}, {1, 1}, {0, 3}, {1, 3}};
    static const Sel t37_16[4][2] = {{{0, 0}, {0, 1}}, {{0, 0}, {0, 1}}, {{1, 2}, {1, 3}}, {{1, 2}, {1, 3}}};
    const auto fp_val = [](uint32_t code, int ebits, int mbits, int bias) {   // a small float, no infinity or NaN used
      const int sign = (code >> (ebits + mbits)) & 1, e = (code >> mbits) & ((1 << ebits) - 1), m = code & ((1 << mbits) - 1);
      const double mag = e == 0 ? std::ldexp(static_cast<double>(m), 1 - bias - mbits) : std::ldexp(static_cast<double>((1 << mbits) | m), e - bias - mbits);
      return sign ? -mag : mag;
    };
    uint32_t seed = 31337;
    const auto rnd32 = [&] { seed = seed * 1664525u + 1013904223u; return seed; };
    const auto reg_of = [](std::vector<uint32_t>& v, int words) { (void)words; (void)v; };
    (void)reg_of;
    // Inputs: S1 differs across lanes, so the lane and the byte both show.
    std::vector<uint32_t> s1(32);
    for (int l = 0; l < 32; ++l) s1[l] = (118u + l % 7) | (121u + l % 5) << 8 | (124u + l % 3) << 16 | (127u + l % 4) << 24;
    const auto scale_of = [&](const Sel& sel, int lane) { return std::ldexp(1.0, static_cast<int>((s1[(lane & 15) + 16 * sel.lanes16] >> (8 * sel.byte)) & 0xFF) - 127); };
    // 4-bit to f32, PK8
    {
      std::vector<uint32_t> s0(32);
      for (auto& v : s0) v = rnd32();
      int wrong = 0;
      std::vector<uint32_t> none(32);
      for (int N : {0, 1, 2, 3, 4, 5, 6, 7}) {
        std::vector<uint32_t> out;
        switch (N) {
          case 0: out = launch(k_sc_f4_f32_0, s0, s1, 256); break;
          case 1: out = launch(k_sc_f4_f32_1, s0, s1, 256); break;
          case 2: out = launch(k_sc_f4_f32_2, s0, s1, 256); break;
          case 3: out = launch(k_sc_f4_f32_3, s0, s1, 256); break;
          case 4: out = launch(k_sc_f4_f32_4, s0, s1, 256); break;
          case 5: out = launch(k_sc_f4_f32_5, s0, s1, 256); break;
          case 6: out = launch(k_sc_f4_f32_6, s0, s1, 256); break;
          default: out = launch(k_sc_f4_f32_7, s0, s1, 256); break;
        }
        for (int l = 0; l < 32; ++l)
          for (int k = 0; k < 8; ++k) {
            const double want = fp_val((s0[l] >> (4 * k)) & 15, 2, 1, 1) * scale_of(t36[N][l / 16], l);
            wrong += out[8 * l + k] != fbits(static_cast<float>(want));
          }
      }
      std::printf("v_cvt_scale_pk8_f32_fp4, every scale_sel 0-7: %d of 2048 wrong\n", wrong);
      failed += wrong != 0;
    }
    // 8-bit (E4M3) to f16, PK8: 64-bit source, f16 pairs in four registers
    {
      std::vector<uint32_t> s0(64);
      for (auto& v : s0) {
        v = rnd32();
        for (int k = 0; k < 4; ++k)
          if (((v >> (8 * k)) & 0x7F) == 0x7F) v &= ~(1u << (8 * k));   // no NaN
      }
      int wrong = 0;
      for (int N : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) {
        const uint32_t* s0p = s0.data();
        (void)s0p;
        uint32_t *d0 = nullptr, *d1 = nullptr, *dout = nullptr;
        std::vector<uint32_t> out(128, 0xDEADBEEFu);
        CHECK(hipMalloc(&d0, 256));
        CHECK(hipMalloc(&d1, 128));
        CHECK(hipMalloc(&dout, 512));
        CHECK(hipMemcpy(d0, s0.data(), 256, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(d1, s1.data(), 128, hipMemcpyHostToDevice));
        switch (N) {
          case 0: hipLaunchKernelGGL(k_sc_f8_f16_0, dim3(1), dim3(32), 0, 0, d0, d1, dout); break;
          case 1: hipLaunchKernelGGL(k_sc_f8_f16_1, dim3(1), dim3(32), 0, 0, d0, d1, dout); break;
          case 2: hipLaunchKernelGGL(k_sc_f8_f16_2, dim3(1), dim3(32), 0, 0, d0, d1, dout); break;
          case 3: hipLaunchKernelGGL(k_sc_f8_f16_3, dim3(1), dim3(32), 0, 0, d0, d1, dout); break;
          case 4: hipLaunchKernelGGL(k_sc_f8_f16_4, dim3(1), dim3(32), 0, 0, d0, d1, dout); break;
          case 5: hipLaunchKernelGGL(k_sc_f8_f16_5, dim3(1), dim3(32), 0, 0, d0, d1, dout); break;
          case 6: hipLaunchKernelGGL(k_sc_f8_f16_6, dim3(1), dim3(32), 0, 0, d0, d1, dout); break;
          case 7: hipLaunchKernelGGL(k_sc_f8_f16_7, dim3(1), dim3(32), 0, 0, d0, d1, dout); break;
          case 8: hipLaunchKernelGGL(k_sc_f8_f16_8, dim3(1), dim3(32), 0, 0, d0, d1, dout); break;
          case 9: hipLaunchKernelGGL(k_sc_f8_f16_9, dim3(1), dim3(32), 0, 0, d0, d1, dout); break;
          case 10: hipLaunchKernelGGL(k_sc_f8_f16_10, dim3(1), dim3(32), 0, 0, d0, d1, dout); break;
          default: hipLaunchKernelGGL(k_sc_f8_f16_11, dim3(1), dim3(32), 0, 0, d0, d1, dout); break;
        }
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(out.data(), dout, 512, hipMemcpyDeviceToHost));
        hipFree(d0), hipFree(d1), hipFree(dout);
        for (int l = 0; l < 32; ++l) {
          const Sel sel = (N & 8) ? t37_16[(N & 7) < 4 ? N & 3 : 0][l / 16] : t37_32[N & 7];
          if ((N & 8) && (N & 4)) continue;   // reserved rows are refused (not run here)
          for (int k = 0; k < 8; ++k) {
            const uint32_t code = (s0[2 * l + k / 4] >> (8 * (k % 4))) & 0xFF;
            const double want = fp_val(code, 4, 3, 7) * scale_of(sel, l);
            const uint32_t got = (out[4 * l + k / 2] >> (16 * (k % 2))) & 0xFFFF;
            wrong += got != f2h(static_cast<float>(want));
          }
        }
      }
      std::printf("v_cvt_scale_pk8_f16_fp8, scale_sel 0-7 and 8-11 (the 8-bit block16 rows): %d of 3072 wrong\n", wrong);
      failed += wrong != 0;
    }
    // 6-bit (E2M3) to f32, PK16: 96-bit source, sixteen f32 results
    {
      std::vector<uint32_t> s0(96);
      for (auto& v : s0) v = rnd32();
      int wrong = 0;
      for (int N : {0, 1, 2, 4, 5, 7}) {
        uint32_t *d0 = nullptr, *d1 = nullptr, *dout = nullptr;
        std::vector<uint32_t> out(512, 0xDEADBEEFu);
        CHECK(hipMalloc(&d0, 384));
        CHECK(hipMalloc(&d1, 128));
        CHECK(hipMalloc(&dout, 2048));
        CHECK(hipMemcpy(d0, s0.data(), 384, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(d1, s1.data(), 128, hipMemcpyHostToDevice));
        switch (N) {
          case 0: hipLaunchKernelGGL(k_sc_f6_f32_0, dim3(1), dim3(32), 0, 0, d0, d1, dout); break;
          case 1: hipLaunchKernelGGL(k_sc_f6_f32_1, dim3(1), dim3(32), 0, 0, d0, d1, dout); break;
          case 2: hipLaunchKernelGGL(k_sc_f6_f32_2, dim3(1), dim3(32), 0, 0, d0, d1, dout); break;
          case 4: hipLaunchKernelGGL(k_sc_f6_f32_4, dim3(1), dim3(32), 0, 0, d0, d1, dout); break;
          case 5: hipLaunchKernelGGL(k_sc_f6_f32_5, dim3(1), dim3(32), 0, 0, d0, d1, dout); break;
          default: hipLaunchKernelGGL(k_sc_f6_f32_7, dim3(1), dim3(32), 0, 0, d0, d1, dout); break;
        }
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(out.data(), dout, 2048, hipMemcpyDeviceToHost));
        hipFree(d0), hipFree(d1), hipFree(dout);
        for (int l = 0; l < 32; ++l)
          for (int k = 0; k < 16; ++k) {
            const int at = 6 * k;
            uint32_t code = s0[3 * l + at / 32] >> (at % 32);
            if (at % 32 + 6 > 32) code |= s0[3 * l + at / 32 + 1] << (32 - at % 32);
            const double want = fp_val(code & 63, 2, 3, 1) * scale_of(t36[N][l / 16], l);
            wrong += out[16 * l + k] != fbits(static_cast<float>(want));
          }
      }
      std::printf("v_cvt_scale_pk16_f32_fp6, scale_sel 0, 1, 2, 4, 5, 7: %d of 3072 wrong\n", wrong);
      failed += wrong != 0;
    }
  }
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
