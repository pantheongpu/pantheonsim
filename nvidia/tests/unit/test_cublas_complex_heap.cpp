// The complex and symmetric cuBLAS routines on the host's blocked kernels (cublas_complex.inc, host_gemm.hpp): the heap
// one call holds, and a few outputs against the plain loops.
//
// The routines hold the matrices once, as std::complex<double> (they always did). The kernels read those through accessors
// a panel at a time, so the sums are the only thing beside them: no planar copy of either operand (4 doubles per element of
// A and B) and no second set of planes for the sums, which the first form of this code made (about 28 MiB at 512^3
// where this holds 16 MiB). Each bound below is what the routine holds, plus a few MiB for the panels.
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "alloc_counter.hpp"
#include "vtest.hpp"

namespace {

using cd = std::complex<double>;

template <class T>
struct Dev {
  T* p = nullptr;
  explicit Dev(const std::vector<T>& host) {
    cudaMalloc(reinterpret_cast<void**>(&p), host.size() * sizeof(T));
    cudaMemcpy(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
  }
  ~Dev() { cudaFree(p); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
};

constexpr double kMiB = 1048576.0;

}  // namespace

VTEST(cublas_zgemm_holds_no_planar_copies) {
  const int n = 512;
  std::mt19937_64 rng(5);
  std::uniform_real_distribution<double> u(-2.0, 2.0);
  const size_t cnt = static_cast<size_t>(n) * n;
  std::vector<cuDoubleComplex> A(cnt), B(cnt), C(cnt);
  for (auto* v : {&A, &B, &C})
    for (auto& x : *v) x = make_cuDoubleComplex(u(rng), u(rng));
  cublasHandle_t h = nullptr;
  VCHECK_EQ(cublasCreate(&h), CUBLAS_STATUS_SUCCESS);
  Dev<cuDoubleComplex> dA(A), dB(B), dC(C);
  const cuDoubleComplex alpha = make_cuDoubleComplex(0.5, -1.25), beta = make_cuDoubleComplex(2.0, 0.5);
  const size_t base = alloc_counter::live();
  alloc_counter::reset_peak();
  const cublasStatus_t st = cublasZgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &alpha, dA.p, n, dB.p, n, &beta, dC.p, n);
  const size_t peak = alloc_counter::peak(), after = alloc_counter::live();
  VCHECK_EQ(st, CUBLAS_STATUS_SUCCESS);
  const size_t mat = cnt * sizeof(cuDoubleComplex);
  std::printf("  zgemm 512^3: peak extra heap %.1f MiB (one matrix %.1f MiB), left over %.2f MiB\n", (peak - base) / kMiB, mat / kMiB,
              (after - base) / kMiB);
  VCHECK(peak - base >= 3 * mat);          // the counter sees the library (A, B and C as complex<double>)
  VCHECK(peak - base <= 5 * mat + (6u << 20));   // A, B, C, the sums (and the matrix being read in); no planes
  VCHECK(after - base <= (2u << 20));
  std::vector<cuDoubleComplex> got(cnt);
  cudaMemcpy(got.data(), dC.p, mat, cudaMemcpyDeviceToHost);
  const int points[][2] = {{0, 0}, {1, 511}, {511, 1}, {300, 200}, {511, 511}};
  for (const auto& ij : points) {
    cd s = 0;
    for (int p = 0; p < n; ++p)
      s += cd(A[ij[0] + static_cast<size_t>(p) * n].x, A[ij[0] + static_cast<size_t>(p) * n].y) *
           cd(B[ij[1] + static_cast<size_t>(p) * n].x, B[ij[1] + static_cast<size_t>(p) * n].y);   // B transposed
    const size_t q = ij[0] + static_cast<size_t>(ij[1]) * n;
    const cd want = cd(alpha.x, alpha.y) * s + cd(beta.x, beta.y) * cd(C[q].x, C[q].y);
    const double gr = got[q].x, gi = got[q].y, wr = want.real(), wi = want.imag();
    VCHECK(std::memcmp(&gr, &wr, 8) == 0 && std::memcmp(&gi, &wi, 8) == 0);
  }
  cublasDestroy(h);
}

VTEST(cublas_dsyrk_holds_no_planar_copies) {
  const int n = 768, k = 512;
  std::mt19937_64 rng(6);
  std::uniform_real_distribution<double> u(-2.0, 2.0);
  std::vector<double> A(static_cast<size_t>(n) * k), C(static_cast<size_t>(n) * n);
  for (auto& x : A) x = u(rng);
  for (auto& x : C) x = u(rng);
  cublasHandle_t h = nullptr;
  VCHECK_EQ(cublasCreate(&h), CUBLAS_STATUS_SUCCESS);
  Dev<double> dA(A), dC(C);
  const double alpha = 1.5, beta = 0.25;
  const size_t base = alloc_counter::live();
  alloc_counter::reset_peak();
  const cublasStatus_t st = cublasDsyrk(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &alpha, dA.p, n, &beta, dC.p, n);
  const size_t peak = alloc_counter::peak(), after = alloc_counter::live();
  VCHECK_EQ(st, CUBLAS_STATUS_SUCCESS);
  const size_t ma = A.size() * 16, mc = C.size() * 16;   // as complex<double>
  std::printf("  dsyrk 768x512: peak extra heap %.1f MiB (A and C as complex %.1f MiB), left over %.2f MiB\n", (peak - base) / kMiB,
              (ma + mc) / kMiB, (after - base) / kMiB);
  VCHECK(peak - base >= ma + mc);
  VCHECK(peak - base <= ma + 2 * mc + (8u << 20));   // the two matrices, the sums, the matrix being read in (planes would add 10 MiB)
  VCHECK(after - base <= (2u << 20));
  std::vector<double> got(C.size());
  cudaMemcpy(got.data(), dC.p, got.size() * 8, cudaMemcpyDeviceToHost);
  const int points[][2] = {{0, 0}, {767, 0}, {767, 767}, {400, 300}, {300, 400}};
  for (const auto& ij : points) {
    const size_t q = ij[0] + static_cast<size_t>(ij[1]) * n;
    if (ij[0] < ij[1]) {   // the upper triangle is left as it was
      VCHECK(std::memcmp(&got[q], &C[q], 8) == 0);
      continue;
    }
    cd s = 0;
    for (int p = 0; p < k; ++p) s += cd(A[ij[0] + static_cast<size_t>(p) * n]) * cd(A[ij[1] + static_cast<size_t>(p) * n]);
    const cd want = cd(alpha) * s + cd(beta) * cd(C[q]);
    const double wr = want.real();
    VCHECK(std::memcmp(&got[q], &wr, 8) == 0);
  }
  cublasDestroy(h);
}

VTEST_MAIN
