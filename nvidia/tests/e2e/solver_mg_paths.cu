// cusolverMg: a matrix spread over the devices in NVIDIA's layout -- column
// blocks dealt round-robin, block b on device b % G as its local block b / G,
// each device's columns contiguous with leading dimension = the row count --
// factored and solved, and each result checked by its defining property after
// gathering it back by the same rule:
//
//   getrf + getrs   P L U = A, with IPIV spread like a 1 x N row of the same
//                   blocks; A X = B, and op T refused, as NVIDIA's refuses it
//   potrf + potrs   L L^H = A in the lower triangle, the upper untouched;
//                   A X = B; the upper triangle refused, as NVIDIA's
//                   refuses it
//   potri           A A^-1 = I from the lower triangle
//   syevd           A v = w v, eigenvalues ascending, W in host memory
//
// For S, D, C and Z, on every device there is (two at most): the RTX 3060
// pair this was written against, or the two simulated devices CI gives it.
// N is not a multiple of the block size, so the last block is short.
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusolverMg.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

using cdouble = std::complex<double>;

static int failures = 0;

static void check(bool ok, const char* what, double err) {
  std::printf("%-4s %s (%.2e)\n", ok ? "ok" : "FAIL", what, err);
  if (!ok) ++failures;
}

#define CK(x)                                                                    \
  do {                                                                           \
    const int r_ = (int)(x);                                                     \
    if (r_ != 0) {                                                               \
      std::printf("FAIL %s returned %d\n", #x, r_);                             \
      ++failures;                                                                \
      return;                                                                    \
    }                                                                            \
  } while (0)

template <class T> struct Ty;
template <> struct Ty<float> {
  static constexpr cudaDataType type = CUDA_R_32F, real = CUDA_R_32F;
  static constexpr const char* name = "S";
  static constexpr double tol = 1e-5;
  using R = float;
  static float make(cdouble v) { return (float)v.real(); }
  static cdouble get(float v) { return v; }
};
template <> struct Ty<double> {
  static constexpr cudaDataType type = CUDA_R_64F, real = CUDA_R_64F;
  static constexpr const char* name = "D";
  static constexpr double tol = 1e-12;
  using R = double;
  static double make(cdouble v) { return v.real(); }
  static cdouble get(double v) { return v; }
};
template <> struct Ty<cuComplex> {
  static constexpr cudaDataType type = CUDA_C_32F, real = CUDA_R_32F;
  static constexpr const char* name = "C";
  static constexpr double tol = 1e-5;
  using R = float;
  static cuComplex make(cdouble v) { return make_cuComplex((float)v.real(), (float)v.imag()); }
  static cdouble get(cuComplex v) { return {v.x, v.y}; }
};
template <> struct Ty<cuDoubleComplex> {
  static constexpr cudaDataType type = CUDA_C_64F, real = CUDA_R_64F;
  static constexpr const char* name = "Z";
  static constexpr double tol = 1e-12;
  using R = double;
  static cuDoubleComplex make(cdouble v) { return make_cuDoubleComplex(v.real(), v.imag()); }
  static cdouble get(cuDoubleComplex v) { return {v.x, v.y}; }
};

static int G = 1;
static std::vector<int> ids;
static cusolverMgHandle_t h;
static cudaLibMgGrid_t grid;
static const int NB = 4;

// A rows x cols matrix spread over the devices.
template <class T> struct Dist {
  int rows, cols;
  std::vector<void*> parts;
  std::vector<int> local;  // columns on each device
  cudaLibMgMatrixDesc_t desc = nullptr;
  Dist(int r, int c) : rows(r), cols(c), parts(G), local(G, 0) {
    for (int b = 0; b * NB < c; ++b) local[b % G] += std::min(NB, c - b * NB);
    for (int d = 0; d < G; ++d) {
      cudaSetDevice(ids[d]);
      cudaMalloc(&parts[d], sizeof(T) * (size_t)r * std::max(local[d], 1));
    }
    cudaSetDevice(ids[0]);
    cusolverMgCreateMatrixDesc(&desc, r, c, r, NB, Ty<T>::type, grid);
  }
  ~Dist() {
    for (void* p : parts) cudaFree(p);
    cusolverMgDestroyMatrixDesc(desc);
  }
  static std::pair<int, int> where(int j) { return {(j / NB) % G, ((j / NB) / G) * NB + j % NB}; }
  void put(const std::vector<T>& a) {  // column-major, ld = rows
    for (int j = 0; j < cols; ++j) {
      const auto [d, c] = where(j);
      cudaMemcpy(static_cast<T*>(parts[d]) + (size_t)c * rows, &a[(size_t)j * rows], sizeof(T) * rows,
                 cudaMemcpyHostToDevice);
    }
  }
  std::vector<T> get() const {
    std::vector<T> a((size_t)rows * cols);
    for (int j = 0; j < cols; ++j) {
      const auto [d, c] = where(j);
      cudaMemcpy(&a[(size_t)j * rows], static_cast<T*>(parts[d]) + (size_t)c * rows, sizeof(T) * rows,
                 cudaMemcpyDeviceToHost);
    }
    return a;
  }
};
// Workspace on every device.
template <class T> struct Work {
  std::vector<void*> parts;
  explicit Work(int64_t lwork) : parts(G) {
    for (int d = 0; d < G; ++d) {
      cudaSetDevice(ids[d]);
      cudaMalloc(&parts[d], sizeof(T) * (size_t)std::max<int64_t>(lwork, 1));
    }
    cudaSetDevice(ids[0]);
  }
  ~Work() {
    for (void* p : parts) cudaFree(p);
  }
};

template <class T> static std::vector<cdouble> widen(const std::vector<T>& v) {
  std::vector<cdouble> o;
  for (const T& x : v) o.push_back(Ty<T>::get(x));
  return o;
}

template <class T> static void run() {
  const bool cplx = Ty<T>::type == CUDA_C_32F || Ty<T>::type == CUDA_C_64F;
  const int N = 13, NRHS = 3;
  std::vector<T> a((size_t)N * N), spd((size_t)N * N), b((size_t)N * NRHS);
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
      a[(size_t)j * N + i] = Ty<T>::make(cdouble(std::sin(1.0 + 1.7 * i + 0.3 * j * j) + (i == j ? 0.5 : 0.0),
                                                 cplx ? std::cos(0.9 * i - j) : 0.0));
  const auto aw = widen(a);
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i) {
      cdouble s = i == j ? (double)N : 0.0;
      for (int k = 0; k < N; ++k) s += std::conj(aw[(size_t)i * N + k]) * aw[(size_t)j * N + k];
      spd[(size_t)j * N + i] = Ty<T>::make(s);
    }
  for (size_t i = 0; i < b.size(); ++i) b[i] = Ty<T>::make(cdouble(std::cos(0.3 * (double)i), cplx ? 0.2 : 0.0));
  const auto sw = widen(spd), bw = widen(b);
  double scale = 0;
  for (const auto& v : sw) scale = std::max(scale, std::abs(v));
  const cudaDataType t = Ty<T>::type;
  char line[200];

  {  // getrf, with IPIV spread like a 1 x N row.
    Dist<T> A(N, N);
    A.put(a);
    std::vector<int*> ipiv(G);
    std::vector<int> ilocal(G, 0);
    for (int b2 = 0; b2 * NB < N; ++b2) ilocal[b2 % G] += std::min(NB, N - b2 * NB);
    for (int d = 0; d < G; ++d) {
      cudaSetDevice(ids[d]);
      cudaMalloc(&ipiv[d], sizeof(int) * std::max(ilocal[d], 1));
    }
    cudaSetDevice(ids[0]);
    int64_t lwork = -1;
    CK(cusolverMgGetrf_bufferSize(h, N, N, A.parts.data(), 1, 1, A.desc, ipiv.data(), t, &lwork));
    Work<T> w(lwork);
    int info = -1;
    CK(cusolverMgGetrf(h, N, N, A.parts.data(), 1, 1, A.desc, ipiv.data(), t, w.parts.data(), lwork, &info));
    cudaDeviceSynchronize();
    const auto f = widen(A.get());
    std::vector<int> piv(N);
    for (int j = 0; j < N; ++j) {
      const auto [d, c] = Dist<T>::where(j);
      cudaMemcpy(&piv[j], ipiv[d] + c, sizeof(int), cudaMemcpyDeviceToHost);
    }
    // P L U: L U, then the interchanges undone in reverse.
    std::vector<cdouble> lu((size_t)N * N);
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < N; ++i) {
        cdouble s = 0;
        for (int k = 0; k <= std::min(i, j); ++k) s += (k == i ? 1.0 : f[(size_t)k * N + i]) * f[(size_t)j * N + k];
        lu[(size_t)j * N + i] = s;
      }
    bool valid = true;
    for (int k = N - 1; k >= 0; --k) {
      valid = valid && piv[k] >= 1 && piv[k] <= N;
      if (valid && piv[k] - 1 != k)
        for (int j = 0; j < N; ++j) std::swap(lu[(size_t)j * N + k], lu[(size_t)j * N + piv[k] - 1]);
    }
    double e = 0, sa = 0;
    for (size_t i = 0; i < lu.size(); ++i) {
      e = std::max(e, std::abs(lu[i] - aw[i]));
      sa = std::max(sa, std::abs(aw[i]));
    }
    std::snprintf(line, sizeof line, "%sgetrf over %d device(s): P L U = A, IPIV spread by column blocks", Ty<T>::name,
                  G);
    check(info == 0 && valid && e / sa < 10 * Ty<T>::tol, line, e / sa);

    {
      const cublasOperation_t op = CUBLAS_OP_N;
      Dist<T> B(N, NRHS);
      B.put(b);
      CK(cusolverMgGetrs_bufferSize(h, op, N, NRHS, A.parts.data(), 1, 1, A.desc, ipiv.data(), B.parts.data(), 1, 1,
                                    B.desc, t, &lwork));
      Work<T> w2(lwork);
      CK(cusolverMgGetrs(h, op, N, NRHS, A.parts.data(), 1, 1, A.desc, ipiv.data(), B.parts.data(), 1, 1, B.desc, t,
                         w2.parts.data(), lwork, &info));
      cudaDeviceSynchronize();
      const auto x = widen(B.get());
      double r = 0, xn = 0;
      for (int j = 0; j < NRHS; ++j)
        for (int i = 0; i < N; ++i) {
          cdouble s = 0;
          for (int k = 0; k < N; ++k)
            s += (op == CUBLAS_OP_N ? aw[(size_t)k * N + i] : aw[(size_t)i * N + k]) * x[(size_t)j * N + k];
          r = std::max(r, std::abs(s - bw[(size_t)j * N + i]));
          xn = std::max(xn, std::abs(x[(size_t)j * N + i]));
        }
      std::snprintf(line, sizeof line, "%sgetrs: A X = B", Ty<T>::name);
      check(info == 0 && r / (sa * xn * N) < Ty<T>::tol, line, r / (sa * xn * N));
      const int st = cusolverMgGetrs(h, CUBLAS_OP_T, N, NRHS, A.parts.data(), 1, 1, A.desc, ipiv.data(),
                                     B.parts.data(), 1, 1, B.desc, t, w2.parts.data(), lwork, &info);
      std::snprintf(line, sizeof line, "%sgetrs refuses op T (INVALID_VALUE)", Ty<T>::name);
      check(st == CUSOLVER_STATUS_INVALID_VALUE, line, st);
    }
    for (int* p : ipiv) cudaFree(p);
  }

  {  // potrf + potrs, lower: NVIDIA's takes no other
    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    const bool lower = true;
    Dist<T> A(N, N);
    A.put(spd);
    int64_t lwork = -1;
    CK(cusolverMgPotrf_bufferSize(h, uplo, N, A.parts.data(), 1, 1, A.desc, t, &lwork));
    Work<T> w(lwork);
    int info = -1;
    CK(cusolverMgPotrf(h, uplo, N, A.parts.data(), 1, 1, A.desc, t, w.parts.data(), lwork, &info));
    cudaDeviceSynchronize();
    const auto f = widen(A.get());
    double e = 0;
    bool other = true;
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < N; ++i) {
        if (lower ? i < j : i > j) {
          other = other && f[(size_t)j * N + i] == sw[(size_t)j * N + i];
          continue;
        }
        cdouble s = 0;  // (L L^H)(i, j) or (U^H U)(i, j)
        for (int k = 0; k <= std::min(i, j); ++k)
          s += lower ? f[(size_t)k * N + i] * std::conj(f[(size_t)k * N + j])
                     : std::conj(f[(size_t)i * N + k]) * f[(size_t)j * N + k];
        e = std::max(e, std::abs(s - sw[(size_t)j * N + i]));
      }
    std::snprintf(line, sizeof line, "%spotrf, %s: the factor reproduces A, the other triangle untouched", Ty<T>::name,
                  lower ? "lower" : "upper");
    check(info == 0 && other && e / scale < 10 * Ty<T>::tol, line, e / scale);

    Dist<T> B(N, NRHS);
    B.put(b);
    CK(cusolverMgPotrs_bufferSize(h, uplo, N, NRHS, A.parts.data(), 1, 1, A.desc, B.parts.data(), 1, 1, B.desc, t,
                                  &lwork));
    Work<T> w2(lwork);
    CK(cusolverMgPotrs(h, uplo, N, NRHS, A.parts.data(), 1, 1, A.desc, B.parts.data(), 1, 1, B.desc, t,
                       w2.parts.data(), lwork, &info));
    cudaDeviceSynchronize();
    const auto x = widen(B.get());
    double r = 0, xn = 0;
    for (int j = 0; j < NRHS; ++j)
      for (int i = 0; i < N; ++i) {
        cdouble s = 0;
        for (int k = 0; k < N; ++k) s += sw[(size_t)k * N + i] * x[(size_t)j * N + k];
        r = std::max(r, std::abs(s - bw[(size_t)j * N + i]));
        xn = std::max(xn, std::abs(x[(size_t)j * N + i]));
      }
    std::snprintf(line, sizeof line, "%spotrs, %s: A X = B", Ty<T>::name, lower ? "lower" : "upper");
    check(info == 0 && r / (scale * xn * N) < Ty<T>::tol, line, r / (scale * xn * N));

    {  // The upper triangle, refused: by potrf's _bufferSize, and by potrs and syevd themselves.
      const auto U = CUBLAS_FILL_MODE_UPPER;
      int64_t lw2 = 0;
      std::vector<typename Ty<T>::R> wv(N);
      const int s1 = cusolverMgPotrf_bufferSize(h, U, N, A.parts.data(), 1, 1, A.desc, t, &lw2);
      const int s2 = cusolverMgPotrs(h, U, N, NRHS, A.parts.data(), 1, 1, A.desc, B.parts.data(), 1, 1, B.desc, t,
                                     w2.parts.data(), lwork, &info);
      const int s3 = cusolverMgSyevd(h, CUSOLVER_EIG_MODE_VECTOR, U, N, A.parts.data(), 1, 1, A.desc, wv.data(),
                                     Ty<T>::real, t, w2.parts.data(), lwork, &info);
      std::snprintf(line, sizeof line, "%spotrf, potrs and syevd refuse the upper triangle (INVALID_VALUE)",
                    Ty<T>::name);
      check(s1 == 3 && s2 == 3 && s3 == 3, line, s1 * 100 + s2 * 10 + s3);
    }
    if (lower) {  // potri on the factor: the lower triangle of A^-1
      CK(cusolverMgPotri_bufferSize(h, uplo, N, A.parts.data(), 1, 1, A.desc, t, &lwork));
      Work<T> w3(lwork);
      CK(cusolverMgPotri(h, uplo, N, A.parts.data(), 1, 1, A.desc, t, w3.parts.data(), lwork, &info));
      cudaDeviceSynchronize();
      const auto inv = widen(A.get());
      double ie = 0, in = 0;
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
          cdouble s = 0;
          for (int k = 0; k < N; ++k) {
            const cdouble v = i >= k ? inv[(size_t)k * N + i] : std::conj(inv[(size_t)i * N + k]);
            s += v * sw[(size_t)j * N + k];
          }
          ie = std::max(ie, std::abs(s - (i == j ? 1.0 : 0.0)));
          in = std::max(in, std::abs(inv[(size_t)j * N + i]));
        }
      std::snprintf(line, sizeof line, "%spotri, lower: A A^-1 = I", Ty<T>::name);
      check(info == 0 && ie / (scale * in * N) < Ty<T>::tol, line, ie / (scale * in * N));
    }
  }

  {  // syevd, W in host memory
    Dist<T> A(N, N);
    A.put(spd);
    std::vector<typename Ty<T>::R> wv(N);
    int64_t lwork = -1;
    CK(cusolverMgSyevd_bufferSize(h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, N, A.parts.data(), 1, 1, A.desc,
                                  wv.data(), Ty<T>::real, t, &lwork));
    Work<T> w(lwork);
    int info = -1;
    CK(cusolverMgSyevd(h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, N, A.parts.data(), 1, 1, A.desc, wv.data(),
                       Ty<T>::real, t, w.parts.data(), lwork, &info));
    cudaDeviceSynchronize();
    const auto v = widen(A.get());
    double e = 0;
    bool ascending = true;
    for (int j = 0; j < N; ++j) {
      if (j) ascending = ascending && wv[j - 1] <= wv[j];
      for (int i = 0; i < N; ++i) {
        cdouble s = 0;
        for (int k = 0; k < N; ++k) s += sw[(size_t)k * N + i] * v[(size_t)j * N + k];
        e = std::max(e, std::abs(s - (double)wv[j] * v[(size_t)j * N + i]));
      }
    }
    std::snprintf(line, sizeof line, "%ssyevd: A v = w v, eigenvalues ascending", Ty<T>::name);
    check(info == 0 && ascending && e / scale < 10 * Ty<T>::tol, line, e / scale);
  }
}

int main() {
  int count = 0;
  cudaGetDeviceCount(&count);
  G = std::min(count, 2);
  if (G < 1) {
    std::printf("FAIL: no device\n");
    return 1;
  }
  for (int i = 0; i < G; ++i) ids.push_back(i);
  if (cusolverMgCreate(&h) || cusolverMgDeviceSelect(h, G, ids.data()) ||
      cusolverMgCreateDeviceGrid(&grid, 1, G, ids.data(), CUDALIBMG_GRID_MAPPING_COL_MAJOR)) {
    std::printf("FAIL: cusolverMg setup\n");
    return 1;
  }
  run<float>();
  run<double>();
  run<cuComplex>();
  run<cuDoubleComplex>();
  cusolverMgDestroyGrid(grid);
  cusolverMgDestroy(h);
  std::printf(failures ? "FAIL: %d cusolverMg checks\n" : "PASS: every cusolverMg check\n", failures);
  return failures ? 1 : 0;
}
