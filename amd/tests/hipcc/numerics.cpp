// The MODE register's floating-point modes, as a kernel starts with them and
// as it sets them: the round mode and the denormal mode of single and double
// precision.
//
// Built twice: numerics.gfx942 as hipcc builds by default (round to nearest
// even, denormals kept -- what every kernel of ROCm's libraries asks for) and
// numerics.flush.gfx942 with -fgpu-flush-denormals-to-zero, whose kernel
// descriptors ask for single-precision denormals to be flushed, in and out.
// Each kernel reads MODE back, works out values whose answer depends on the
// modes, and the host checks each against IEEE arithmetic done its own way.
// Each directed rounding is set with s_setreg around the instruction it is
// for, in the kernel's own assembly so the compiler cannot move it.
#include <hip/hip_runtime.h>

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#define CHECK(x)                                                                        \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

struct In {
  float tiny, one, zero, big, small, three, a, b, c;
  double da, db;
};
struct Out {
  unsigned mode;
  float tiny_times_one, tiny_plus_zero, underflow, third, quotient;
  float add[4], mul[4], fma[4];
  double dadd[4];
  float flushed_by_setreg;
};

// The float instructions under each round mode, set around each one.
#define ROUNDED(r, op, out, ...)                                                        \
  asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 2), " #r "\n\ts_nop 2\n\t" op \
               "\n\ts_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 2), 0\n\ts_nop 2"          \
               : "=v"(out)                                                              \
               : __VA_ARGS__)
#define DROUNDED(r, out, x, y)                                                          \
  asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), " #r "\n\ts_nop 2\n\t"     \
               "v_add_f64 %0, %1, %2\n\ts_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0\n\ts_nop 2" \
               : "=v"(out)                                                              \
               : "v"(x), "v"(y))
#define ALL_ROUNDINGS(r)                                                                \
  ROUNDED(r, "v_add_f32 %0, %1, %2", o->add[r], "v"(in->a), "v"(in->b));               \
  ROUNDED(r, "v_mul_f32 %0, %1, %2", o->mul[r], "v"(in->c), "v"(in->c));               \
  ROUNDED(r, "v_fma_f32 %0, %1, %2, %3", o->fma[r], "v"(in->a), "v"(in->c), "v"(in->b)); \
  DROUNDED(r, o->dadd[r], in->da, in->db)

__global__ void modes(const In* in, Out* o) {
  if (threadIdx.x != 0) return;
  o->mode = __builtin_amdgcn_s_getreg(1 | (31 << 11));   // hwreg(HW_REG_MODE, 0, 32)
  o->tiny_times_one = in->tiny * in->one;
  o->tiny_plus_zero = in->tiny + in->zero;
  o->underflow = in->small * in->small;
  o->third = in->one / in->three;
  o->quotient = in->big / in->three;
  ALL_ROUNDINGS(0);
  ALL_ROUNDINGS(1);
  ALL_ROUNDINGS(2);
  ALL_ROUNDINGS(3);
  // Flushing set by the kernel itself: single-precision denormals off, in
  // and out, for one multiply.
  float f;
  asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0\n\ts_nop 2\n\tv_mul_f32 %0, %1, %2\n\t"
               "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3\n\ts_nop 2"
               : "=v"(f)
               : "v"(in->tiny), "v"(in->one));
  o->flushed_by_setreg = f;
}

// The host's answers, each under the round mode asked for.
template <typename F>
auto rounded(int mode, F f) {
  static const int kRound[4] = {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO};
  const int saved = std::fegetround();
  std::fesetround(kRound[mode]);
  const auto r = f();
  std::fesetround(saved);
  return r;
}

int wrong = 0;
template <typename T>
void expect(const char* what, T got, T want) {
  const bool same = std::memcmp(&got, &want, sizeof got) == 0;
  std::printf("%s %s: %a%s%a\n", same ? "ok  " : "FAIL", what, static_cast<double>(got), same ? " == " : " != ",
              static_cast<double>(want));
  wrong += !same;
}

int main() {
#ifdef VGPU_FLUSH
  const bool flush = true;
#else
  const bool flush = false;
#endif
  volatile In v;
  v.tiny = 0x1p-140f;            // a denormal
  v.one = 1.0f;
  v.zero = 0.0f;
  v.big = 0x1p100f;
  v.small = 0x1p-70f;            // squared: 2^-140, a denormal
  v.three = 3.0f;
  v.a = 1.0f;
  v.b = 0x1.8p-24f;              // a + b lies between two floats, nearer the upper
  v.c = 0x1.555556p-1f;          // c * c, and a * c + b, round differently each way
  v.da = 1.0;
  v.db = -0x1.8p-53;             // da + db lies between two doubles
  In in;
  std::memcpy(&in, const_cast<const In*>(&v), sizeof in);
  In* din;
  Out* dout;
  Out out;
  CHECK(hipMalloc(&din, sizeof(In)));
  CHECK(hipMalloc(&dout, sizeof(Out)));
  CHECK(hipMemcpy(din, &in, sizeof(In), hipMemcpyHostToDevice));
  modes<<<1, 64>>>(din, dout);
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(&out, dout, sizeof(Out), hipMemcpyDeviceToHost));

  // MODE as the descriptor set it: round to nearest even, DX10 clamp and IEEE
  // on, doubles' denormals kept, and floats' kept or flushed as built.
  expect("MODE the kernel starts with", out.mode & 0x3FF, flush ? 0x3C0u : 0x3F0u);
  expect("a denormal times one", out.tiny_times_one, flush ? 0.0f : v.tiny * v.one);
  expect("a denormal plus zero", out.tiny_plus_zero, flush ? 0.0f : v.tiny + v.zero);
  expect("a product too small to be normal", out.underflow, flush ? 0.0f : v.small * v.small);
  expect("one divided by three", out.third, v.one / v.three);
  expect("2^100 divided by three", out.quotient, v.big / v.three);
  static const char* kName[4] = {"to nearest even", "toward +inf", "toward -inf", "toward zero"};
  char what[96];
  for (int r = 0; r < 4; ++r) {
    std::snprintf(what, sizeof what, "v_add_f32 rounded %s", kName[r]);
    expect(what, out.add[r], rounded(r, [&] { return v.a + v.b; }));
    std::snprintf(what, sizeof what, "v_mul_f32 rounded %s", kName[r]);
    expect(what, out.mul[r], rounded(r, [&] { return v.c * v.c; }));
    std::snprintf(what, sizeof what, "v_fma_f32 rounded %s", kName[r]);
    expect(what, out.fma[r], rounded(r, [&] { return std::fma(v.a, v.c, v.b); }));
    std::snprintf(what, sizeof what, "v_add_f64 rounded %s", kName[r]);
    expect(what, out.dadd[r], rounded(r, [&] { return v.da + v.db; }));
  }
  expect("a denormal times one, flushed by s_setreg", out.flushed_by_setreg, 0.0f);
  std::printf("%d wrong\n", wrong);
  return wrong != 0;
}
