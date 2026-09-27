// hipSPARSELt's 2:4 structured-sparse GEMMs, run as a program would: its own
// kernels prune a dense matrix to two values of each four, compress it into
// the values and their indices, and multiply -- on gfx942 and gfx950 with
// the sparse matrix instructions (v_smfmac_*). The host checks the pruned
// matrix is 2:4 and works the same product out densely from it.
//
// Each case is a type (half, bfloat16 or int8), which of A and B is the
// sparse one, and whether each is transposed. The values are small integers,
// so every sum is exact and the answer must match to the last bit.
//
// Host code only, built against hipSPARSELt's headers (MIT) and the library
// PyTorch's ROCm wheel ships; amd/tests/e2e/run_hipsparselt.sh runs it with
// the wheel's libraries and the simulator's HIP.
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipsparselt/hipsparselt.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define HIP(x)                                                                          \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)
#define LT(x)                                                                           \
  do {                                                                                  \
    hipsparseStatus_t s_ = (x);                                                         \
    if (s_ != HIPSPARSE_STATUS_SUCCESS) {                                               \
      std::printf("FAIL %s: status %d\n", #x, static_cast<int>(s_));                    \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

uint32_t rng = 97531;
uint32_t next() { return rng = rng * 1664525u + 1013904223u; }

// A value of each type from a small integer, and back.
struct Half {
  static constexpr hipDataType type = HIP_R_16F;
  using T = __half;
  static T from(int v) { return __float2half(static_cast<float>(v)); }
  static double to(T v) { return __half2float(v); }
};
struct Bf16 {
  static constexpr hipDataType type = HIP_R_16BF;
  using T = __hip_bfloat16;
  static T from(int v) { return __float2bfloat16(static_cast<float>(v)); }
  static double to(T v) { return __bfloat162float(v); }
};
struct Int8 {
  static constexpr hipDataType type = HIP_R_8I;
  using T = int8_t;
  static T from(int v) { return static_cast<T>(v); }
  static double to(T v) { return v; }
};

// D = A B + C, A m x k and B k x n (as op() leaves them), column-major.
template <typename Ty>
int run(const char* name, bool sparse_a, bool trans_a, bool trans_b, int m, int n, int k) {
  using T = typename Ty::T;
  const bool integer = Ty::type == HIP_R_8I;
  // Small enough that no int8 sum leaves its range.
  const int span = integer ? 1 : 3;
  const hipsparseOperation_t op_a = trans_a ? HIPSPARSE_OPERATION_TRANSPOSE : HIPSPARSE_OPERATION_NON_TRANSPOSE;
  const hipsparseOperation_t op_b = trans_b ? HIPSPARSE_OPERATION_TRANSPOSE : HIPSPARSE_OPERATION_NON_TRANSPOSE;
  // Stored shapes: A is m x k, or k x m transposed; B is k x n, or n x k.
  const int rows_a = trans_a ? k : m, cols_a = trans_a ? m : k, rows_b = trans_b ? n : k, cols_b = trans_b ? k : n;
  std::vector<T> a(static_cast<size_t>(rows_a) * cols_a), b(static_cast<size_t>(rows_b) * cols_b), c(static_cast<size_t>(m) * n), d(c.size());
  for (auto& x : a) x = Ty::from(static_cast<int>(next() % (2 * span + 1)) - span);
  for (auto& x : b) x = Ty::from(static_cast<int>(next() % (2 * span + 1)) - span);
  for (auto& x : c) x = Ty::from(static_cast<int>(next() % (2 * span + 1)) - span);

  T *da, *db, *dc, *dd;
  HIP(hipMalloc(&da, a.size() * sizeof(T)));
  HIP(hipMalloc(&db, b.size() * sizeof(T)));
  HIP(hipMalloc(&dc, c.size() * sizeof(T)));
  HIP(hipMalloc(&dd, d.size() * sizeof(T)));
  HIP(hipMemcpy(da, a.data(), a.size() * sizeof(T), hipMemcpyHostToDevice));
  HIP(hipMemcpy(db, b.data(), b.size() * sizeof(T), hipMemcpyHostToDevice));
  HIP(hipMemcpy(dc, c.data(), c.size() * sizeof(T), hipMemcpyHostToDevice));

  hipsparseLtHandle_t handle;
  hipsparseLtMatDescriptor_t mat_a, mat_b, mat_c, mat_d;
  hipsparseLtMatmulDescriptor_t matmul;
  hipsparseLtMatmulAlgSelection_t alg;
  hipsparseLtMatmulPlan_t plan;
  LT(hipsparseLtInit(&handle));
  const auto describe = [&](hipsparseLtMatDescriptor_t* desc, int rows, int cols, bool sparse) {
    return sparse ? hipsparseLtStructuredDescriptorInit(&handle, desc, rows, cols, rows, 16, Ty::type,
                                                        HIPSPARSE_ORDER_COL, HIPSPARSELT_SPARSITY_50_PERCENT)
                  : hipsparseLtDenseDescriptorInit(&handle, desc, rows, cols, rows, 16, Ty::type, HIPSPARSE_ORDER_COL);
  };
  LT(describe(&mat_a, rows_a, cols_a, sparse_a));
  LT(describe(&mat_b, rows_b, cols_b, !sparse_a));
  LT(describe(&mat_c, m, n, false));
  LT(describe(&mat_d, m, n, false));
  LT(hipsparseLtMatmulDescriptorInit(&handle, &matmul, op_a, op_b, &mat_a, &mat_b, &mat_c, &mat_d,
                                     integer ? HIPSPARSELT_COMPUTE_32I : HIPSPARSELT_COMPUTE_32F));
  LT(hipsparseLtMatmulAlgSelectionInit(&handle, &alg, &matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT));
  LT(hipsparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg));

  // Prune the sparse operand in place, and check the library agrees it is 2:4.
  T* dsparse = sparse_a ? da : db;
  hipStream_t stream = nullptr;
  LT(hipsparseLtSpMMAPrune(&handle, &matmul, dsparse, dsparse, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream));
  int* dvalid;
  int valid = 1;
  HIP(hipMalloc(&dvalid, sizeof(int)));
  LT(hipsparseLtSpMMAPruneCheck(&handle, &matmul, dsparse, dvalid, stream));
  HIP(hipMemcpy(&valid, dvalid, sizeof(int), hipMemcpyDeviceToHost));
  std::vector<T>& pruned = sparse_a ? a : b;
  HIP(hipMemcpy(pruned.data(), dsparse, pruned.size() * sizeof(T), hipMemcpyDeviceToHost));

  size_t compressed_size = 0, buffer_size = 0, workspace_size = 0;
  LT(hipsparseLtSpMMACompressedSize(&handle, &plan, &compressed_size, &buffer_size));
  void *dcompressed, *dbuffer = nullptr, *dworkspace = nullptr;
  HIP(hipMalloc(&dcompressed, compressed_size));
  if (buffer_size) HIP(hipMalloc(&dbuffer, buffer_size));
  LT(hipsparseLtSpMMACompress(&handle, &plan, dsparse, dcompressed, dbuffer, stream));
  LT(hipsparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));
  if (workspace_size) HIP(hipMalloc(&dworkspace, workspace_size));
  const float alpha = 1, beta = 1;
  LT(hipsparseLtMatmul(&handle, &plan, &alpha, sparse_a ? dcompressed : da, sparse_a ? db : dcompressed, &beta, dc,
                       dd, dworkspace, &stream, 1));
  HIP(hipDeviceSynchronize());
  HIP(hipMemcpy(d.data(), dd, d.size() * sizeof(T), hipMemcpyDeviceToHost));

  // The pruned operand must be 2:4 along K: at most two non-zeros in each
  // run of four.
  int not_24 = 0;
  const auto at_a = [&](int i, int kk) { return Ty::to(a[trans_a ? static_cast<size_t>(i) * rows_a + kk : static_cast<size_t>(kk) * rows_a + i]); };
  const auto at_b = [&](int kk, int j) { return Ty::to(b[trans_b ? static_cast<size_t>(kk) * rows_b + j : static_cast<size_t>(j) * rows_b + kk]); };
  for (int outer = 0; outer < (sparse_a ? m : n); ++outer)
    for (int k0 = 0; k0 < k; k0 += 4) {
      int nz = 0;
      for (int kk = k0; kk < k0 + 4; ++kk) nz += (sparse_a ? at_a(outer, kk) : at_b(kk, outer)) != 0;
      not_24 += nz > 2;
    }
  int wrong = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      double want = Ty::to(c[static_cast<size_t>(j) * m + i]);
      for (int kk = 0; kk < k; ++kk) want += at_a(i, kk) * at_b(kk, j);
      const double got = Ty::to(d[static_cast<size_t>(j) * m + i]);
      if (got != want && wrong++ < 2) std::printf("  %s: D[%d][%d] is %g, not %g\n", name, i, j, got, want);
    }
  std::printf("%s: pruned %s, %d runs of four not 2:4, %d of %d wrong\n", name, valid == 0 ? "2:4" : "WRONGLY",
              not_24, wrong, m * n);

  for (void* p : {static_cast<void*>(da), static_cast<void*>(db), static_cast<void*>(dc), static_cast<void*>(dd),
                  static_cast<void*>(dvalid), dcompressed, dbuffer, dworkspace})
    if (p) (void)hipFree(p);
  hipsparseLtMatmulPlanDestroy(&plan);
  for (auto* desc : {&mat_a, &mat_b, &mat_c, &mat_d}) hipsparseLtMatDescriptorDestroy(desc);
  hipsparseLtDestroy(&handle);
  return wrong || not_24 || valid != 0;
}

int main() {
  int failed = 0;
  failed += run<Half>("half, sparse A, NN", true, false, false, 64, 64, 64);
  failed += run<Half>("half, sparse A, TN", true, true, false, 64, 96, 128);
  failed += run<Half>("half, sparse B, NT", false, false, true, 96, 64, 64);
  failed += run<Bf16>("bfloat16, sparse A, NN", true, false, false, 64, 64, 64);
  failed += run<Bf16>("bfloat16, sparse B, TN", false, true, false, 64, 64, 128);
  failed += run<Int8>("int8, sparse A, TN", true, true, false, 64, 64, 128);
  std::printf("%s\n", failed ? "FAIL" : "every sparse GEMM right");
  return failed != 0;
}
