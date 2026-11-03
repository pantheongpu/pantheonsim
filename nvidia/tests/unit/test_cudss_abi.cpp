// The cuDSS shim's two ABIs. One source, built twice (CMakeLists.txt):
//
//   test_cudss_abi_v0   VGPU_CUDSS_API_0_8, linked against libcudss.so.0 (cuDSS 0.8)
//   test_cudss_abi_v1   the default,        linked against libcudss.so.1 (cuDSS 1.0)
//
// Each solves a small symmetric indefinite system the way SCS's cuDSS backend
// does (lower-view zero-based CSR, ANALYSIS, FACTORIZATION, inertia, SOLVE,
// then new values and REFACTORIZATION, 32- and 64-bit indices), and pins what
// the two majors differ in. Every 1.0 expectation below was measured on
// NVIDIA's libcudss 1.0.0.35 (config defaults and the ranges the settings
// accept) or read from its header (enum values, signatures, the symbol
// version); the 0.8 ones are what the shim has answered since it was written.
#ifdef VGPU_CUDSS_API_0_8
#define EXPECT_V1 0
#else
#define EXPECT_V1 1
#endif

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "../../include/vgpu_cudss.h"
#include "vtest.hpp"

extern "C" const char* __asan_default_options() { return "detect_leaks=1:detect_odr_violation=0"; }

// ---- the declarations themselves (compile time) ----
#if EXPECT_V1
static_assert(CUDSS_VERSION_MAJOR == 1 && CUDSS_VERSION_MINOR == 0 && CUDSS_VERSION_PATCH == 0, "1.0.0");
static_assert((int)CUDSS_DATA_TYPE_UNSET == 63, "1.0: UNSET is 63");
static_assert((int)CUDSS_R_64F_MP2 == 62 && (int)CUDSS_R_64F_64F == 62, "1.0: double-double is 62, old name kept");
static_assert(CUDSS_R_32F == CUDA_R_32F && CUDSS_R_64F == CUDA_R_64F && CUDSS_C_32F == CUDA_C_32F &&
                  CUDSS_C_64F == CUDA_C_64F && CUDSS_R_32I == CUDA_R_32I && CUDSS_R_64I == CUDA_R_64I,
              "1.0: the CUDSS_ names are CUDA's");
static_assert(CUDSS_CONFIG_EMULATION_MANTISSA_BIT_COUNT == 26 && CUDSS_CONFIG_FORCE_NONSYMMETRIC_MATCHING == 27 &&
                  CUDSS_CONFIG_DOWNCAST_FACTORS == 28 && CUDSS_CONFIG_MEMORY_OPTIMIZATION_SETTING == 29,
              "1.0 appends four config parameters after ND_UBFACTOR (25)");
// cudssDataType_t is cudaDataType_t again.
static_assert(std::is_same_v<cudssDataType_t, cudaDataType_t>, "1.0: one type");
#else
static_assert(CUDSS_VERSION_MAJOR == 0 && CUDSS_VERSION_MINOR == 8, "0.8");
static_assert((int)CUDSS_DATA_TYPE_UNSET == 1024 && (int)CUDSS_R_64F_64F == 1025 + CUDA_R_64F, "0.8 values");
static_assert(!std::is_same_v<cudssDataType_t, cudaDataType_t>, "0.8: a type of its own");
#endif

// ---- a system to solve ----
// M = [4 1 0 1; 1 3 2 0; 0 2 -1 1; 1 0 1 5], inertia (3 positive, 1 negative);
// M2 is M with its diagonal entries 4 and 5 raised to 6 and 7. x = (1, 2, 3, 4).
static const int kN = 4;
static const int kRowPtr[] = {0, 1, 3, 5, 8};
static const int kCol[] = {0, 0, 1, 1, 2, 0, 2, 3};
static const double kVal[] = {4, 1, 3, 2, -1, 1, 1, 5};
static const double kVal2[] = {6, 1, 3, 2, -1, 1, 1, 7};
static const double kB[] = {10, 13, 5, 24};
static const double kB2[] = {12, 13, 5, 32};
static const double kX[] = {1, 2, 3, 4};

template <class T>
static T* to_device(const T* h, size_t n) {
  T* d = nullptr;
  cudaMalloc((void**)&d, n * sizeof(T) + 16);
  cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

static double max_error(const double* got, const double* want, int n) {
  double e = 0;
  for (int i = 0; i < n; ++i) e = std::fmax(e, std::fabs(got[i] - want[i]));
  return e;
}

// SCS's sequence, in the index width given (the index arrays are int32 or int64 on the device).
template <class I>
static void solve_like_scs(cudssDataType_t itype) {
  std::vector<I> rp(kRowPtr, kRowPtr + kN + 1), ci(kCol, kCol + 8);
  I* d_rp = to_device(rp.data(), rp.size());
  I* d_ci = to_device(ci.data(), ci.size());
  double* d_val = to_device(kVal, 8);
  double* d_b = to_device(kB, kN);
  double zeros[kN] = {0, 0, 0, 0};
  double* d_x = to_device(zeros, kN);

  cudssHandle_t h = nullptr;
  cudssConfig_t cfg = nullptr;
  cudssData_t data = nullptr;
  VCHECK_EQ((int)cudssCreate(&h), (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ((int)cudssConfigCreate(&cfg), (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ((int)cudssDataCreate(h, &data), (int)CUDSS_STATUS_SUCCESS);
  cudssMatrix_t A = nullptr, B = nullptr, X = nullptr;
  VCHECK_EQ((int)cudssMatrixCreateCsr(&A, kN, kN, 8, d_rp, nullptr, d_ci, d_val, itype, itype, CUDSS_R_64F,
                                      CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO),
            (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ((int)cudssMatrixCreateDn(&B, kN, 1, kN, d_b, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
            (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ((int)cudssMatrixCreateDn(&X, kN, 1, kN, d_x, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
            (int)CUDSS_STATUS_SUCCESS);

  VCHECK_EQ((int)cudssExecute(h, CUDSS_PHASE_ANALYSIS, cfg, data, A, X, B), (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ((int)cudssExecute(h, CUDSS_PHASE_FACTORIZATION, cfg, data, A, X, B), (int)CUDSS_STATUS_SUCCESS);
  I inertia[2] = {0, 0};
  size_t written = 0;
  VCHECK_EQ((int)cudssDataGet(h, data, CUDSS_DATA_INERTIA, inertia, sizeof inertia, &written),
            (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ((int)inertia[0], 3);
  VCHECK_EQ((int)inertia[1], 1);
  VCHECK_EQ((int)cudssExecute(h, CUDSS_PHASE_SOLVE, cfg, data, A, X, B), (int)CUDSS_STATUS_SUCCESS);
  cudaDeviceSynchronize();
  double x[kN];
  cudaMemcpy(x, d_x, sizeof x, cudaMemcpyDeviceToHost);
  VCHECK(max_error(x, kX, kN) < 1e-12);

  // New values in the same arrays: refactorize, solve again.
  cudaMemcpy(d_val, kVal2, sizeof kVal2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, kB2, sizeof kB2, cudaMemcpyHostToDevice);
  VCHECK_EQ((int)cudssMatrixSetCsrPointers(A, d_rp, nullptr, d_ci, d_val), (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ((int)cudssExecute(h, CUDSS_PHASE_REFACTORIZATION, cfg, data, A, X, B), (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ((int)cudssExecute(h, CUDSS_PHASE_SOLVE, cfg, data, A, X, B), (int)CUDSS_STATUS_SUCCESS);
  cudaDeviceSynchronize();
  cudaMemcpy(x, d_x, sizeof x, cudaMemcpyDeviceToHost);
  VCHECK(max_error(x, kX, kN) < 1e-12);

  // What the matrix says about itself, through the getter (a cudaDataType_t under 1.0).
  cudssDataType_t ot, it, vt;
  cudssMatrixType_t mt;
  cudssMatrixViewType_t mv;
  cudssIndexBase_t ib;
  int64_t nr, nc, nz;
  void *r0, *r1, *c0, *v0;
  VCHECK_EQ((int)cudssMatrixGetCsr(A, &nr, &nc, &nz, &r0, &r1, &c0, &v0, &ot, &it, &vt, &mt, &mv, &ib),
            (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ((int)ot, (int)itype);
  VCHECK_EQ((int)it, (int)itype);
  VCHECK_EQ((int)vt, (int)CUDA_R_64F);
  VCHECK_EQ(nz, (int64_t)8);

  VCHECK_EQ((int)cudssDataDestroy(h, data), (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ((int)cudssConfigDestroy(cfg), (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ((int)cudssMatrixDestroy(A), (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ((int)cudssMatrixDestroy(B), (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ((int)cudssMatrixDestroy(X), (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ((int)cudssDestroy(h), (int)CUDSS_STATUS_SUCCESS);
  cudaFree(d_rp);
  cudaFree(d_ci);
  cudaFree(d_val);
  cudaFree(d_b);
  cudaFree(d_x);
}

VTEST(solves_a_small_symmetric_system_with_32_bit_indices) { solve_like_scs<int32_t>(CUDSS_R_32I); }
VTEST(solves_a_small_symmetric_system_with_64_bit_indices) { solve_like_scs<int64_t>(CUDSS_R_64I); }

VTEST(the_library_reports_the_version_it_is_presented_as) {
  int major = -1, minor = -1, patch = -1;
  VCHECK_EQ((int)cudssGetProperty(MAJOR_VERSION, &major), (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ((int)cudssGetProperty(MINOR_VERSION, &minor), (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ((int)cudssGetProperty(PATCH_LEVEL, &patch), (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ(major, EXPECT_V1 ? 1 : 0);
  VCHECK_EQ(minor, EXPECT_V1 ? 0 : 8);
  VCHECK_EQ(patch, 0);
}

static int get_int(cudssConfig_t c, cudssConfigParam_t p) {
  int v = -12345;
  size_t w = 0;
  VCHECK_EQ((int)cudssConfigGet(c, p, &v, sizeof v, &w), (int)CUDSS_STATUS_SUCCESS);
  return v;
}
static int set_int(cudssConfig_t c, cudssConfigParam_t p, int v) { return (int)cudssConfigSet(c, p, &v, sizeof v); }

VTEST(matching_and_pivoting_defaults_and_ranges_follow_the_major) {
  cudssConfig_t c = nullptr;
  VCHECK_EQ((int)cudssConfigCreate(&c), (int)CUDSS_STATUS_SUCCESS);
  // 1.0 matches by default (AUTO = 6); 0.8 did not.
  VCHECK_EQ(get_int(c, CUDSS_CONFIG_MATCHING_ALG), EXPECT_V1 ? 6 : 0);
  VCHECK_EQ(get_int(c, CUDSS_CONFIG_PIVOT_TYPE), 0);
  // Bunch-Kaufman: reserved (NOT_SUPPORTED) in 0.8, a pivot type in 1.0.
  VCHECK_EQ(set_int(c, CUDSS_CONFIG_PIVOT_TYPE, CUDSS_PIVOT_BUNCH_KAUFMAN),
            (int)(EXPECT_V1 ? CUDSS_STATUS_SUCCESS : CUDSS_STATUS_NOT_SUPPORTED));
  VCHECK_EQ(set_int(c, CUDSS_CONFIG_PIVOT_TYPE, 7), (int)CUDSS_STATUS_INVALID_VALUE);
  VCHECK_EQ((int)cudssConfigDestroy(c), (int)CUDSS_STATUS_SUCCESS);
}

#if EXPECT_V1
VTEST(the_four_new_config_parameters_have_nvidias_defaults_and_ranges) {
  cudssConfig_t c = nullptr;
  VCHECK_EQ((int)cudssConfigCreate(&c), (int)CUDSS_STATUS_SUCCESS);
  const cudssConfigParam_t params[] = {CUDSS_CONFIG_EMULATION_MANTISSA_BIT_COUNT,
                                       CUDSS_CONFIG_FORCE_NONSYMMETRIC_MATCHING, CUDSS_CONFIG_DOWNCAST_FACTORS,
                                       CUDSS_CONFIG_MEMORY_OPTIMIZATION_SETTING};
  for (auto p : params) {
    VCHECK_EQ(get_int(c, p), 0);  // all default to 0, as ints
    size_t w = 0;
    VCHECK_EQ((int)cudssConfigGet(c, p, nullptr, 0, &w), (int)CUDSS_STATUS_SUCCESS);
    VCHECK_EQ((int)w, 4);
    int v = 0;
    VCHECK_EQ((int)cudssConfigSet(c, p, &v, 8), (int)CUDSS_STATUS_INVALID_VALUE);  // exactly an int
    VCHECK_EQ(set_int(c, p, -1), (int)CUDSS_STATUS_INVALID_VALUE);
  }
  // Ranges: mantissa bits >= 0 (53 and 100 are fine), the others small enums.
  VCHECK_EQ(set_int(c, CUDSS_CONFIG_EMULATION_MANTISSA_BIT_COUNT, 53), (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ(get_int(c, CUDSS_CONFIG_EMULATION_MANTISSA_BIT_COUNT), 53);
  VCHECK_EQ(set_int(c, CUDSS_CONFIG_EMULATION_MANTISSA_BIT_COUNT, 100), (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ(set_int(c, CUDSS_CONFIG_FORCE_NONSYMMETRIC_MATCHING, 1), (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ(set_int(c, CUDSS_CONFIG_FORCE_NONSYMMETRIC_MATCHING, 2), (int)CUDSS_STATUS_INVALID_VALUE);
  VCHECK_EQ(set_int(c, CUDSS_CONFIG_DOWNCAST_FACTORS, CUDSS_DOWNCAST_FACTORS_MODE_KEEP_STRICT_DIAGONAL),
            (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ(get_int(c, CUDSS_CONFIG_DOWNCAST_FACTORS), 2);
  VCHECK_EQ(set_int(c, CUDSS_CONFIG_DOWNCAST_FACTORS, 3), (int)CUDSS_STATUS_INVALID_VALUE);
  VCHECK_EQ(set_int(c, CUDSS_CONFIG_MEMORY_OPTIMIZATION_SETTING, CUDSS_MEMORY_OPTIMIZATION_SOLVE_ONLY),
            (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ(set_int(c, CUDSS_CONFIG_MEMORY_OPTIMIZATION_SETTING, 2), (int)CUDSS_STATUS_INVALID_VALUE);
  // A parameter 1.0 does not have.
  VCHECK_EQ(set_int(c, (cudssConfigParam_t)30, 0), (int)CUDSS_STATUS_INTERNAL_ERROR);
  VCHECK_EQ((int)cudssConfigDestroy(c), (int)CUDSS_STATUS_SUCCESS);
}

VTEST(the_new_settings_do_not_change_the_answer) {
  // Stored and handed back, never acted on: a solve with every one of them set matches the default one.
  cudssConfig_t c = nullptr;
  cudssConfigCreate(&c);
  set_int(c, CUDSS_CONFIG_EMULATION_MANTISSA_BIT_COUNT, 30);
  set_int(c, CUDSS_CONFIG_FORCE_NONSYMMETRIC_MATCHING, 1);
  set_int(c, CUDSS_CONFIG_DOWNCAST_FACTORS, 1);
  set_int(c, CUDSS_CONFIG_MEMORY_OPTIMIZATION_SETTING, 1);
  set_int(c, CUDSS_CONFIG_PIVOT_TYPE, CUDSS_PIVOT_BUNCH_KAUFMAN);
  std::vector<int> rp(kRowPtr, kRowPtr + kN + 1), ci(kCol, kCol + 8);
  int* d_rp = to_device(rp.data(), rp.size());
  int* d_ci = to_device(ci.data(), ci.size());
  double* d_val = to_device(kVal, 8);
  double* d_b = to_device(kB, kN);
  double zeros[kN] = {};
  double* d_x = to_device(zeros, kN);
  cudssHandle_t h;
  cudssData_t data;
  cudssCreate(&h);
  cudssDataCreate(h, &data);
  cudssMatrix_t A, B, X;
  cudssMatrixCreateCsr(&A, kN, kN, 8, d_rp, nullptr, d_ci, d_val, CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
                       CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO);
  cudssMatrixCreateDn(&B, kN, 1, kN, d_b, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
  cudssMatrixCreateDn(&X, kN, 1, kN, d_x, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
  VCHECK_EQ((int)cudssExecute(h, CUDSS_PHASE_ANALYSIS | CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE, c, data, A,
                              X, B),
            (int)CUDSS_STATUS_SUCCESS);
  double x[kN];
  cudaDeviceSynchronize();
  cudaMemcpy(x, d_x, sizeof x, cudaMemcpyDeviceToHost);
  VCHECK(max_error(x, kX, kN) < 1e-12);
  cudssDataDestroy(h, data);
  cudssConfigDestroy(c);
  cudssMatrixDestroy(A);
  cudssMatrixDestroy(B);
  cudssMatrixDestroy(X);
  cudssDestroy(h);
  cudaFree(d_rp);
  cudaFree(d_ci);
  cudaFree(d_val);
  cudaFree(d_b);
  cudaFree(d_x);
}

VTEST(a_transposed_solve_is_refused_not_answered_with_the_plain_solution) {
  cudssConfig_t c = nullptr;
  cudssConfigCreate(&c);
  // 1.0's range for SOLVE_MODE is 0..2; outside it NVIDIA answers NOT_SUPPORTED.
  VCHECK_EQ(set_int(c, CUDSS_CONFIG_SOLVE_MODE, 3), (int)CUDSS_STATUS_NOT_SUPPORTED);
  VCHECK_EQ(set_int(c, CUDSS_CONFIG_SOLVE_MODE, -1), (int)CUDSS_STATUS_NOT_SUPPORTED);
  VCHECK_EQ(set_int(c, CUDSS_CONFIG_SOLVE_MODE, CUDSS_SOLVE_MODE_TRANSPOSE), (int)CUDSS_STATUS_SUCCESS);
  std::vector<int> rp(kRowPtr, kRowPtr + kN + 1), ci(kCol, kCol + 8);
  int* d_rp = to_device(rp.data(), rp.size());
  int* d_ci = to_device(ci.data(), ci.size());
  double* d_val = to_device(kVal, 8);
  double* d_b = to_device(kB, kN);
  double zeros[kN] = {};
  double* d_x = to_device(zeros, kN);
  cudssHandle_t h;
  cudssData_t data;
  cudssCreate(&h);
  cudssDataCreate(h, &data);
  cudssMatrix_t A, B, X;
  cudssMatrixCreateCsr(&A, kN, kN, 8, d_rp, nullptr, d_ci, d_val, CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
                       CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO);
  cudssMatrixCreateDn(&B, kN, 1, kN, d_b, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
  cudssMatrixCreateDn(&X, kN, 1, kN, d_x, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
  VCHECK_EQ((int)cudssExecute(h, CUDSS_PHASE_ANALYSIS | CUDSS_PHASE_FACTORIZATION, c, data, A, X, B),
            (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ((int)cudssExecute(h, CUDSS_PHASE_SOLVE, c, data, A, X, B), (int)CUDSS_STATUS_NOT_SUPPORTED);
  set_int(c, CUDSS_CONFIG_SOLVE_MODE, CUDSS_SOLVE_MODE_NORMAL);
  VCHECK_EQ((int)cudssExecute(h, CUDSS_PHASE_SOLVE, c, data, A, X, B), (int)CUDSS_STATUS_SUCCESS);
  cudssDataDestroy(h, data);
  cudssConfigDestroy(c);
  cudssMatrixDestroy(A);
  cudssMatrixDestroy(B);
  cudssMatrixDestroy(X);
  cudssDestroy(h);
  cudaFree(d_rp);
  cudaFree(d_ci);
  cudaFree(d_val);
  cudaFree(d_b);
  cudaFree(d_x);
}

static int pinned_alloc(void*, void**, size_t, cudaStream_t) { return 0; }
static int pinned_free(void*, void*, size_t, cudaStream_t) { return 0; }

VTEST(the_three_new_functions_keep_what_they_are_given) {
  cudssHandle_t h = nullptr;
  VCHECK_EQ((int)cudssCreate(&h), (int)CUDSS_STATUS_SUCCESS);
  cudssPinnedMemHandler_t got{};
  VCHECK_EQ((int)cudssGetPinnedMemHandler(h, &got), (int)CUDSS_STATUS_INVALID_VALUE);  // none set yet
  cudssPinnedMemHandler_t mine{};
  int tag = 7;
  mine.ctx = &tag;
  mine.pinned_alloc = pinned_alloc;
  mine.pinned_free = pinned_free;
  std::strcpy(mine.name, "mine");
  VCHECK_EQ((int)cudssSetPinnedMemHandler(h, &mine), (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ((int)cudssGetPinnedMemHandler(h, &got), (int)CUDSS_STATUS_SUCCESS);
  VCHECK(got.ctx == &tag && got.pinned_alloc == pinned_alloc && got.pinned_free == pinned_free);
  VCHECK_EQ(std::string(got.name), std::string("mine"));
  VCHECK_EQ((int)cudssSetJitCacheDir(h, "/tmp/vgpu-cudss-jit-unused"), (int)CUDSS_STATUS_SUCCESS);
  VCHECK_EQ((int)cudssSetJitCacheDir(h, nullptr), (int)CUDSS_STATUS_INVALID_VALUE);
  VCHECK_EQ((int)cudssSetJitCacheDir(nullptr, "/tmp/x"), (int)CUDSS_STATUS_INVALID_VALUE);
  VCHECK_EQ((int)cudssDestroy(h), (int)CUDSS_STATUS_SUCCESS);
}
#endif

VTEST(symbols_carry_nvidias_version_tag_only_in_1_0) {
  // NVIDIA's 1.0 tags every export "libcudss.so.1" (a program linked against it asks for
  // cudssCreate@libcudss.so.1); its 0.8 tags none. This shim does the same, so a binary built
  // against either resolves, and a 1.0 program never binds to a 0.8 body.
  void* tagged = dlvsym(RTLD_DEFAULT, "cudssCreate", "libcudss.so.1");
  void* plain = dlsym(RTLD_DEFAULT, "cudssCreate");
  VCHECK(plain != nullptr);
  VCHECK_EQ(tagged != nullptr, EXPECT_V1 != 0);
  // The three functions 1.0 added are not in 0.8.
  VCHECK_EQ(dlsym(RTLD_DEFAULT, "cudssSetJitCacheDir") != nullptr, EXPECT_V1 != 0);
  VCHECK_EQ(dlsym(RTLD_DEFAULT, "cudssGetPinnedMemHandler") != nullptr, EXPECT_V1 != 0);
  VCHECK_EQ(dlsym(RTLD_DEFAULT, "cudssSetPinnedMemHandler") != nullptr, EXPECT_V1 != 0);
}

VTEST_MAIN
