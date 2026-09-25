// AMD's rocBLAS, unmodified, on a simulated MI300X: a level-1 call, a
// reduction, and single- and double-precision GEMMs, each checked against the
// same arithmetic done on the host. rocBLAS brings its own kernels -- some
// built into the library, the GEMMs from the Tensile code objects it loads
// at run time -- so what runs is AMD's code, not ours.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cmath>
#include <cstdio>
#include <vector>

#define HIP(x)                                                                      \
  do {                                                                              \
    hipError_t e_ = (x);                                                            \
    if (e_ != hipSuccess) {                                                         \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                      \
      return 1;                                                                     \
    }                                                                               \
  } while (0)
#define BLAS(x)                                                                     \
  do {                                                                              \
    rocblas_status s_ = (x);                                                        \
    if (s_ != rocblas_status_success) {                                             \
      std::printf("FAIL %s: %s\n", #x, rocblas_status_to_string(s_));               \
      return 1;                                                                     \
    }                                                                               \
  } while (0)

template <typename T>
T* to_device(const std::vector<T>& h) {
  T* d = nullptr;
  if (hipMalloc(&d, h.size() * sizeof(T)) != hipSuccess) return nullptr;
  if (hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice) != hipSuccess) return nullptr;
  return d;
}

template <typename T>
std::vector<T> to_host(const T* d, size_t n) {
  std::vector<T> h(n);
  if (hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost) != hipSuccess) h.clear();
  return h;
}

// C = alpha * op(A) * B + beta * C in column-major order, as BLAS has it, on
// the host and in double.
template <typename T>
std::vector<double> gemm_on_host(bool transpose_a, int m, int n, int k, double alpha, const std::vector<T>& a,
                                 int lda, const std::vector<T>& b, int ldb, double beta, const std::vector<T>& c,
                                 int ldc) {
  std::vector<double> out(c.begin(), c.end());
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      double s = 0;
      for (int l = 0; l < k; ++l) s += double(transpose_a ? a[i * lda + l] : a[l * lda + i]) * double(b[j * ldb + l]);
      out[j * ldc + i] = alpha * s + beta * double(c[j * ldc + i]);
    }
  return out;
}

// How many of got's elements are within tolerance of want's, relative to the
// size of what was summed.
template <typename T>
int close(const std::vector<T>& got, const std::vector<double>& want, double tolerance) {
  int n = 0;
  for (size_t i = 0; i < want.size() && i < got.size(); ++i)
    n += std::fabs(double(got[i]) - want[i]) <= tolerance * (1 + std::fabs(want[i]));
  return n;
}

template <typename T>
std::vector<T> filled(size_t n, int seed) {
  std::vector<T> v(n);
  unsigned x = 2463534242u + seed;
  for (auto& e : v) {
    x ^= x << 13, x ^= x >> 17, x ^= x << 5;
    e = T(int(x % 2001) - 1000) / T(1000);
  }
  return v;
}

int main() {
  rocblas_handle handle;
  BLAS(rocblas_create_handle(&handle));
  hipDeviceProp_t p;
  HIP(hipGetDeviceProperties(&p, 0));
  char version[64] = {};
  BLAS(rocblas_get_version_string(version, sizeof version));
  std::printf("rocBLAS %s on %s\n", version, p.gcnArchName);

  // y = alpha * x + y.
  const int n = 1000;
  const std::vector<float> x = filled<float>(n, 1), y = filled<float>(n, 2);
  float *dx = to_device(x), *dy = to_device(y);
  if (!dx || !dy) return 1;
  const float alpha = 2.5f;
  BLAS(rocblas_saxpy(handle, n, &alpha, dx, 1, dy, 1));
  const std::vector<float> axpy = to_host(dy, n);
  std::vector<double> want(n);
  for (int i = 0; i < n; ++i) want[i] = double(alpha) * x[i] + y[i];
  std::printf("saxpy: %d of %d elements right\n", close(axpy, want, 1e-6), n);

  // x . y, reduced on the device, the result back on the host.
  float dot = 0;
  BLAS(rocblas_sdot(handle, n, dx, 1, dx, 1, &dot));
  double want_dot = 0;
  for (int i = 0; i < n; ++i) want_dot += double(x[i]) * x[i];
  std::printf("sdot: %s\n", std::fabs(dot - want_dot) <= 1e-4 * want_dot ? "right" : "wrong");

  // GEMMs of a size Tensile has kernels for, one with A transposed.
  const int m = 96, nn = 80, k = 64;
  {
    const std::vector<float> a = filled<float>(size_t(m) * k, 3), b = filled<float>(size_t(k) * nn, 4),
                             c = filled<float>(size_t(m) * nn, 5);
    float *da = to_device(a), *db = to_device(b), *dc = to_device(c);
    if (!da || !db || !dc) return 1;
    const float al = 1.5f, be = -0.5f;
    BLAS(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, m, nn, k, &al, da, m, db, k, &be, dc,
                       m));
    const std::vector<float> got = to_host(dc, size_t(m) * nn);
    std::printf("sgemm %dx%dx%d: %d of %d elements right\n", m, nn, k,
                close(got, gemm_on_host(false, m, nn, k, al, a, m, b, k, be, c, m), 1e-4), m * nn);
  }
  {
    const std::vector<double> a = filled<double>(size_t(k) * m, 6), b = filled<double>(size_t(k) * nn, 7),
                              c = filled<double>(size_t(m) * nn, 8);
    double *da = to_device(a), *db = to_device(b), *dc = to_device(c);
    if (!da || !db || !dc) return 1;
    const double al = 0.75, be = 2.0;
    BLAS(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, m, nn, k, &al, da, k, db, k, &be,
                       dc, m));
    const std::vector<double> got = to_host(dc, size_t(m) * nn);
    std::printf("dgemm %dx%dx%d, A transposed: %d of %d elements right\n", m, nn, k,
                close(got, gemm_on_host(true, m, nn, k, al, a, k, b, k, be, c, m), 1e-12), m * nn);
  }
  // GEMMs of ragged sizes, every transpose of A and B: where Tensile's
  // kernels lean on a buffer's bounds to read zeroes past a matrix's edge.
  int shapes = 0, right = 0;
  const int sizes[][3] = {{1, 1, 1}, {7, 3, 1}, {33, 17, 5}, {130, 1, 33}, {64, 65, 66}, {200, 130, 70}};
  for (const auto& sz : sizes)
    for (int t = 0; t < 4; ++t) {
      const int m = sz[0], n2 = sz[1], k2 = sz[2];
      const bool ta = t & 1, tb = t & 2;
      const int lda = ta ? k2 : m, ldb = tb ? n2 : k2;
      const auto a = filled<double>(size_t(lda) * (ta ? m : k2), 10 + t), b = filled<double>(size_t(ldb) * (tb ? k2 : n2), 20 + t),
                 c = filled<double>(size_t(m) * n2, 30 + t);
      double *da = to_device(a), *db = to_device(b), *dc = to_device(c);
      const auto fa = std::vector<float>(a.begin(), a.end()), fb = std::vector<float>(b.begin(), b.end()),
                 fc = std::vector<float>(c.begin(), c.end());
      float *sa = to_device(fa), *sb = to_device(fb), *sc = to_device(fc);
      if (!da || !db || !dc || !sa || !sb || !sc) return 1;
      const double al = 1.25, be = -0.5;
      const float fal = 1.25f, fbe = -0.5f;
      const auto op = [](bool x) { return x ? rocblas_operation_transpose : rocblas_operation_none; };
      BLAS(rocblas_dgemm(handle, op(ta), op(tb), m, n2, k2, &al, da, lda, db, ldb, &be, dc, m));
      BLAS(rocblas_sgemm(handle, op(ta), op(tb), m, n2, k2, &fal, sa, lda, sb, ldb, &fbe, sc, m));
      std::vector<double> want(size_t(m) * n2);
      for (int j = 0; j < n2; ++j)
        for (int i = 0; i < m; ++i) {
          double sum = 0;
          for (int l = 0; l < k2; ++l)
            sum += (ta ? a[i * lda + l] : a[l * lda + i]) * (tb ? b[l * ldb + j] : b[j * ldb + l]);
          want[j * m + i] = al * sum + be * c[j * m + i];
        }
      const int n_elems = m * n2;
      right += close(to_host(dc, n_elems), want, 1e-12) == n_elems;
      right += close(to_host(sc, n_elems), want, 1e-4) == n_elems;
      shapes += 2;
      for (void* p : {static_cast<void*>(da), static_cast<void*>(db), static_cast<void*>(dc), static_cast<void*>(sa),
                      static_cast<void*>(sb), static_cast<void*>(sc)})
        HIP(hipFree(p));
    }
  std::printf("ragged GEMMs, every transpose, float and double: %d of %d right\n", right, shapes);
  BLAS(rocblas_destroy_handle(handle));
  return 0;
}
