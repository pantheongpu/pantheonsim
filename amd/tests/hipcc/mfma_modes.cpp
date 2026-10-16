// The matrix instructions' modifiers and modes, against what AMD's CDNA3 and CDNA4 ISA guides say (mfma_modes.gfx942 and .gfx950, wave64):
// BLGP's permutations of B's lanes (Table 29, 7.1.5.2), the negation bits of the double-precision forms (7.1.5.3), the execution mask the
// matrix core ignores ("forces it to 1 for all threads", 7.1.2), MODE's denormal flag for the 32-bit float forms (7.3: "honor the
// denormal-handling flags in MODE"; C and the result do not), and the sparse forms multiplying only the B values A's indices select.
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

typedef float v4f __attribute__((ext_vector_type(4)));
typedef double v4d __attribute__((ext_vector_type(4)));
typedef _Float16 v4h __attribute__((ext_vector_type(4)));
typedef _Float16 v8h __attribute__((ext_vector_type(8)));

// 16x16x4, one block: A[i][k] in lane i + 16 k, B[k][j] in lane j + 16 k, D[row][col] in lane col + 16 (row / 4), register row % 4.
template <int BLGP>
__global__ void k_blgp(const float* a, const float* b, float* d) {
  const int l = threadIdx.x;
  const v4f zero = {0, 0, 0, 0};
  const v4f r = __builtin_amdgcn_mfma_f32_16x16x4f32(a[l], b[l], zero, 0, 0, BLGP);
  for (int i = 0; i < 4; ++i) d[4 * l + i] = r[i];
}
template <int NEG>
__global__ void k_f64_neg(const double* a, const double* b, const double* c, double* d) {
  const int l = threadIdx.x;
  const v4d cc = {c[4 * l], c[4 * l + 1], c[4 * l + 2], c[4 * l + 3]};
  const v4d r = __builtin_amdgcn_mfma_f64_16x16x4f64(a[l], b[l], cc, 0, 0, NEG);
  for (int i = 0; i < 4; ++i) d[4 * l + i] = r[i];
}
// With only the first 32 lanes on (EXEC's high half cleared inside the asm), every lane still takes part.
__global__ void k_exec(const float* a, const float* b, float* d) {
  const int l = threadIdx.x;
  v4f r = {0, 0, 0, 0};
  uint64_t saved;
  float x = a[l], y = b[l];
  asm volatile(
      "s_mov_b64 %3, exec\n s_mov_b32 exec_hi, 0\n v_mfma_f32_16x16x4_f32 %0, %1, %2, %0\n s_mov_b64 exec, %3"
      : "+v"(r), "+v"(x), "+v"(y), "=&s"(saved));
  for (int i = 0; i < 4; ++i) d[4 * l + i] = r[i];
}
// MODE.FP_DENORM's input bit (bit 4) cleared round the instruction: 32-bit float inputs are flushed.
template <int FLUSH>
__global__ void k_denorm(const float* a, const float* b, float* d) {
  const int l = threadIdx.x;
  v4f r = {0, 0, 0, 0};
  float x = a[l], y = b[l];
  if (FLUSH) {
    asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 1), 0\n v_mfma_f32_16x16x4_f32 %0, %1, %2, %0\n s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 1), 1"
                 : "+v"(r), "+v"(x), "+v"(y));
  } else {
    asm volatile("v_mfma_f32_16x16x4_f32 %0, %1, %2, %0" : "+v"(r), "+v"(x), "+v"(y));
  }
  for (int i = 0; i < 4; ++i) d[4 * l + i] = r[i];
}
// v_smfmac_f32_16x16x32_f16: A (4 halves a lane) holds two of every four of the K values, the index register says which; B's other two are
// NaNs that must not poison the sum.
__global__ void k_sparse(const float* nanv, float* d) {
  const int l = threadIdx.x;
  v4h a = {1, 1, 1, 1};
  v8h b;
  for (int i = 0; i < 8; ++i) b[i] = (i % 4) >= 2 ? static_cast<_Float16>(nanv[0]) : static_cast<_Float16>(1);
  v4f c = {0, 0, 0, 0};
  const v4f r = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a, b, c, 0x44444444, 0, 0);
  for (int i = 0; i < 4; ++i) d[4 * l + i] = r[i];
}

static int failed = 0;
static void report(const char* what, int wrong, int of) {
  std::printf("%s: %d of %d wrong\n", what, wrong, of);
  failed += wrong != 0;
}

int main() {
  float *da, *db, *dd;
  double *dad, *dbd, *dcd, *ddd;
  CHECK(hipMalloc(&da, 256));
  CHECK(hipMalloc(&db, 256));
  CHECK(hipMalloc(&dd, 1024));
  CHECK(hipMalloc(&dad, 512));
  CHECK(hipMalloc(&dbd, 512));
  CHECK(hipMalloc(&dcd, 2048));
  CHECK(hipMalloc(&ddd, 2048));
  std::vector<float> a(64), b(64), d(256);
  for (int l = 0; l < 64; ++l) a[l] = 1.0f + (l % 16) + 3 * (l / 16), b[l] = 1.0f + (l * 7) % 13;
  CHECK(hipMemcpy(da, a.data(), 256, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(db, b.data(), 256, hipMemcpyHostToDevice));
  const auto lane_b = [](int blgp, int l) {
    switch (blgp) {
      case 1: return l % 32;
      case 2: return l % 32 + 32;
      case 3: return (l + 16) % 64;
      case 4: return l % 16;
      case 5: return l % 16 + 16;
      case 6: return l % 16 + 32;
      case 7: return l % 16 + 48;
      default: return l;
    }
  };
  const auto expect_d = [&](int blgp, std::vector<float>& want) {
    want.assign(256, 0);
    for (int l = 0; l < 64; ++l)
      for (int r = 0; r < 4; ++r) {
        const int row = 4 * (l / 16) + r, col = l % 16;
        double sum = 0;
        for (int k = 0; k < 4; ++k) sum += static_cast<double>(a[row + 16 * k]) * b[lane_b(blgp, col + 16 * k)];
        want[4 * l + r] = static_cast<float>(sum);
      }
  };
  const auto run = [&](auto kernel, const char* what, int blgp) -> int {
    CHECK(hipMemset(dd, 0xFF, 1024));
    hipLaunchKernelGGL(kernel, dim3(1), dim3(64), 0, 0, da, db, dd);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(d.data(), dd, 1024, hipMemcpyDeviceToHost));
    std::vector<float> want;
    expect_d(blgp, want);
    int wrong = 0;
    for (int i = 0; i < 256; ++i) wrong += std::memcmp(&d[i], &want[i], 4) != 0;
    report(what, wrong, 256);
    return 0;
  };
  if (run(k_blgp<0>, "v_mfma_f32_16x16x4_f32 blgp:0", 0)) return 1;
  if (run(k_blgp<1>, "v_mfma_f32_16x16x4_f32 blgp:1 (first 32 lanes)", 1)) return 1;
  if (run(k_blgp<2>, "v_mfma_f32_16x16x4_f32 blgp:2 (second 32 lanes)", 2)) return 1;
  if (run(k_blgp<3>, "v_mfma_f32_16x16x4_f32 blgp:3 (rotate 16 lanes left)", 3)) return 1;
  if (run(k_blgp<4>, "v_mfma_f32_16x16x4_f32 blgp:4 (first 16 lanes)", 4)) return 1;
  if (run(k_blgp<5>, "v_mfma_f32_16x16x4_f32 blgp:5 (second 16 lanes)", 5)) return 1;
  if (run(k_blgp<6>, "v_mfma_f32_16x16x4_f32 blgp:6 (third 16 lanes)", 6)) return 1;
  if (run(k_blgp<7>, "v_mfma_f32_16x16x4_f32 blgp:7 (fourth 16 lanes)", 7)) return 1;
  // The execution mask is ignored.
  if (run(k_exec, "v_mfma_f32_16x16x4_f32 with the upper 32 lanes off in EXEC", 0)) return 1;

  // Double precision: BLGP negates A, B, C.
  {
    std::vector<double> ad(64), bd(64), cd(256), dd(256);
    for (int l = 0; l < 64; ++l) ad[l] = a[l] * 0.5, bd[l] = b[l] * 0.25;
    for (int i = 0; i < 256; ++i) cd[i] = 0.125 * (i % 37) - 1.0;
    CHECK(hipMemcpy(dad, ad.data(), 512, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dbd, bd.data(), 512, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dcd, cd.data(), 2048, hipMemcpyHostToDevice));
    const auto f64 = [&](auto kernel, int neg) -> int {
      CHECK(hipMemset(ddd, 0xFF, 2048));
      hipLaunchKernelGGL(kernel, dim3(1), dim3(64), 0, 0, dad, dbd, dcd, ddd);
      CHECK(hipDeviceSynchronize());
      CHECK(hipMemcpy(dd.data(), ddd, 2048, hipMemcpyDeviceToHost));
      const double sa = neg & 1 ? -1 : 1, sb = neg & 2 ? -1 : 1, sc = neg & 4 ? -1 : 1;
      int wrong = 0;
      for (int l = 0; l < 64; ++l)
        for (int r = 0; r < 4; ++r) {
          const int row = l / 16 + 4 * r, col = l % 16;   // a double's block: one row a lane group, the next four rows on
          double sum = sc * cd[4 * l + r];
          for (int k = 0; k < 4; ++k) sum = std::fma(sa * ad[row + 16 * k], sb * bd[col + 16 * k], sum);
          wrong += std::memcmp(&dd[4 * l + r], &sum, 8) != 0;
        }
      char name[80];
      std::snprintf(name, sizeof name, "v_mfma_f64_16x16x4_f64 blgp:%d (negates %s%s%s)", neg, neg & 1 ? "A " : "", neg & 2 ? "B " : "", neg & 4 ? "C" : "");
      report(name, wrong, 256);
      return 0;
    };
    if (f64(k_f64_neg<0>, 0) || f64(k_f64_neg<1>, 1) || f64(k_f64_neg<2>, 2) || f64(k_f64_neg<4>, 4) || f64(k_f64_neg<7>, 7)) return 1;
  }

  // MODE's denormal flag.
  {
    std::vector<float> den(64, 1e-40f), one(64, 1.0f);
    CHECK(hipMemcpy(da, den.data(), 256, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db, one.data(), 256, hipMemcpyHostToDevice));
    int wrong = 0;
    for (int flush = 0; flush < 2; ++flush) {
      CHECK(hipMemset(dd, 0xFF, 1024));
      if (flush) hipLaunchKernelGGL(k_denorm<1>, dim3(1), dim3(64), 0, 0, da, db, dd);
      else hipLaunchKernelGGL(k_denorm<0>, dim3(1), dim3(64), 0, 0, da, db, dd);
      CHECK(hipDeviceSynchronize());
      CHECK(hipMemcpy(d.data(), dd, 1024, hipMemcpyDeviceToHost));
      const float want = flush ? 0.0f : static_cast<float>(4 * static_cast<double>(1e-40f));
      for (int i = 0; i < 256; ++i) wrong += std::memcmp(&d[i], &want, 4) != 0;
    }
    report("v_mfma_f32_16x16x4_f32 on denormal inputs, with MODE's input-denormal flag set and cleared", wrong, 512);
  }

  // Sparse: NaNs at the pruned positions of B.
  {
    float nan = std::nanf("");
    float* dn;
    CHECK(hipMalloc(&dn, 4));
    CHECK(hipMemcpy(dn, &nan, 4, hipMemcpyHostToDevice));
    CHECK(hipMemset(dd, 0xFF, 1024));
    hipLaunchKernelGGL(k_sparse, dim3(1), dim3(64), 0, 0, dn, dd);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(d.data(), dd, 1024, hipMemcpyDeviceToHost));
    int wrong = 0;
    for (int i = 0; i < 256; ++i) wrong += d[i] != 16.0f;
    report("v_smfmac_f32_16x16x32_f16 with NaNs in B at the pruned positions", wrong, 256);
  }
  std::printf("mfma_modes: %d failed\n", failed);
  return failed != 0;
}
