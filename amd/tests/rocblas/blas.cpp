// AMD's rocBLAS, unmodified, on a simulated MI300X: a level-1 call, a
// reduction, single- and double-precision GEMMs, and gemm_ex on halves,
// bfloat16s and bytes, each checked against the same arithmetic done on the
// host. rocBLAS brings its own kernels -- some built into the library, the
// GEMMs from the Tensile code objects it loads at run time -- so what runs
// is AMD's code, not ours.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

// The narrow types rocBLAS's gemm_ex takes, held as their bits: a half, a
// bfloat16 (a float's top half), a byte, and the float and int32 they
// accumulate into.
struct Type {
  const char* name;
  rocblas_datatype type;
  int bytes;
};
const Type f16{"f16", rocblas_datatype_f16_r, 2}, bf16{"bf16", rocblas_datatype_bf16_r, 2},
    f32{"f32", rocblas_datatype_f32_r, 4}, i8{"i8", rocblas_datatype_i8_r, 1}, i32{"i32", rocblas_datatype_i32_r, 4};

void store(const Type& t, void* at, double v) {
  if (t.type == rocblas_datatype_f16_r) {
    const _Float16 h = static_cast<_Float16>(v);
    std::memcpy(at, &h, 2);
  } else if (t.type == rocblas_datatype_bf16_r) {
    const float f = static_cast<float>(v);
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u = (u + 0x7fff + ((u >> 16) & 1)) >> 16;  // to nearest, ties to even
    std::memcpy(at, &u, 2);
  } else if (t.type == rocblas_datatype_f32_r) {
    const float f = static_cast<float>(v);
    std::memcpy(at, &f, 4);
  } else if (t.type == rocblas_datatype_i8_r) {
    const int8_t b = static_cast<int8_t>(v);
    std::memcpy(at, &b, 1);
  } else {
    const int32_t i = static_cast<int32_t>(v);
    std::memcpy(at, &i, 4);
  }
}

double load(const Type& t, const void* at) {
  if (t.type == rocblas_datatype_f16_r) {
    _Float16 h;
    std::memcpy(&h, at, 2);
    return h;
  }
  if (t.type == rocblas_datatype_bf16_r) {
    uint32_t u = 0;
    std::memcpy(&u, at, 2);
    u <<= 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  }
  if (t.type == rocblas_datatype_f32_r) {
    float f;
    std::memcpy(&f, at, 4);
    return f;
  }
  if (t.type == rocblas_datatype_i8_r) return *static_cast<const int8_t*>(at);
  int32_t i;
  std::memcpy(&i, at, 4);
  return i;
}

// D = alpha * op(A) * op(B) + beta * C through rocblas_gemm_ex, with A and B
// of type in, C and D of type out, summed in compute. The inputs are small
// integers, which every type holds exactly, so each product and sum is exact
// and only storing D in a half or bfloat16 rounds. Returns whether all of D
// is right, or -1 when rocBLAS refuses.
int gemm_ex(rocblas_handle handle, const Type& in, const Type& out, const Type& compute, bool ta, bool tb, int m,
            int n, int k) {
  const int lda = ta ? k : m, ldb = tb ? n : k;
  const size_t na = size_t(lda) * (ta ? m : k), nb = size_t(ldb) * (tb ? k : n), nc = size_t(m) * n;
  std::vector<double> a(na), b(nb), c(nc);
  for (size_t i = 0; i < na; ++i) a[i] = int(i * 7 % 9) - 4;
  for (size_t i = 0; i < nb; ++i) b[i] = int(i * 5 % 7) - 3;
  for (size_t i = 0; i < nc; ++i) c[i] = int(i * 3 % 11) - 5;
  std::vector<unsigned char> ha(na * in.bytes), hb(nb * in.bytes), hc(nc * out.bytes);
  for (size_t i = 0; i < na; ++i) store(in, &ha[i * in.bytes], a[i]);
  for (size_t i = 0; i < nb; ++i) store(in, &hb[i * in.bytes], b[i]);
  for (size_t i = 0; i < nc; ++i) store(out, &hc[i * out.bytes], c[i]);
  unsigned char *da = to_device(ha), *db = to_device(hb), *dc = to_device(hc), *dd = to_device(hc);
  if (!da || !db || !dc || !dd) return -1;
  const float falpha = 2, fbeta = -1;
  const int32_t ialpha = 2, ibeta = -1;
  const bool integer = compute.type == rocblas_datatype_i32_r;
  const void* alpha = integer ? static_cast<const void*>(&ialpha) : &falpha;
  const void* beta = integer ? static_cast<const void*>(&ibeta) : &fbeta;
  const auto op = [](bool x) { return x ? rocblas_operation_transpose : rocblas_operation_none; };
  const rocblas_status s =
      rocblas_gemm_ex(handle, op(ta), op(tb), m, n, k, alpha, da, in.type, lda, db, in.type, ldb, beta, dc, out.type,
                      m, dd, out.type, m, compute.type, rocblas_gemm_algo_standard, 0, 0);
  const std::vector<unsigned char> got = to_host(dd, nc * out.bytes);
  for (void* p : {da, db, dc, dd}) (void)hipFree(p);
  if (s != rocblas_status_success) {
    std::printf("  gemm_ex %s->%s %dx%dx%d: %s\n", in.name, out.name, m, n, k, rocblas_status_to_string(s));
    return -1;
  }
  const bool narrow = out.type == rocblas_datatype_f16_r || out.type == rocblas_datatype_bf16_r;
  const double tolerance = out.type == rocblas_datatype_bf16_r ? 1.0 / 128 : 1.0 / 1024;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      double sum = 0;
      for (int l = 0; l < k; ++l) sum += (ta ? a[i * lda + l] : a[l * lda + i]) * (tb ? b[l * ldb + j] : b[j * ldb + l]);
      const double want = 2 * sum - c[j * m + i];
      const double v = load(out, &got[(size_t(j) * m + i) * out.bytes]);
      if (narrow ? std::fabs(v - want) > tolerance * std::fabs(want) : v != want) {
        std::printf("  gemm_ex %s->%s %dx%dx%d%s%s: D[%d][%d] is %g, not %g\n", in.name, out.name, m, n, k,
                    ta ? " A^T" : "", tb ? " B^T" : "", i, j, v, want);
        return 0;
      }
    }
  return 1;
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

  // Halves, bfloat16s and bytes through gemm_ex, into their own type or a
  // wider one: Tensile's kernels for these use the matrix instructions that
  // take them, and d16 loads and stores that fill half a register.
  const struct {
    const Type &in, &out, &compute;
  } kinds[] = {{f16, f16, f32}, {f16, f32, f32}, {bf16, bf16, f32}, {bf16, f32, f32}, {i8, i32, i32}};
  const int mixed_sizes[][3] = {{96, 80, 64}, {1, 1, 1}, {33, 17, 5}, {64, 65, 66}, {200, 130, 70}};
  int mixed = 0, mixed_right = 0;
  for (const auto& kind : kinds)
    for (const auto& sz : mixed_sizes)
      for (int t = 0; t < 4; ++t) {
        mixed_right += gemm_ex(handle, kind.in, kind.out, kind.compute, t & 1, t & 2, sz[0], sz[1], sz[2]) == 1;
        ++mixed;
      }
  std::printf("gemm_ex, halves, bfloat16s and bytes, every transpose: %d of %d right\n", mixed_right, mixed);
  BLAS(rocblas_destroy_handle(handle));
  return 0;
}
