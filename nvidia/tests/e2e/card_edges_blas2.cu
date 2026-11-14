// cuBLAS's level-2 argument checks and the messages it prints for them, against an RTX 3060's cuBLAS 13.0.
//
//   - ?gemv refuses nothing but a leading dimension below the rows of a non-empty A (INVALID_VALUE, no message).
//     A negative m or n is an empty product (SUCCESS, y untouched), a zero increment is a stride of zero (incx = 0
//     reads x[0] for every element; incy = 0 keeps row 0's result in y[0]), and an operation that is none of
//     the known ones is a transpose. CUBLAS_OP_CONJG (3) is conj(A) for the complex types and A for the real.
//   - every other level-2 routine but ?symv/?hemv refuses its bad arguments as reference BLAS does and prints
//     reference BLAS's line for the first one, on stdout: " ** On entry to SGER   parameter number 5 had an
//     illegal value". So do the GEMMs. ?symv and ?hemv refuse silently.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int fails = 0;
#define WANT(x, want) do { const long long got_ = static_cast<long long>(x); if (got_ != static_cast<long long>(want)) { \
  std::printf("FAIL %s -> %lld, expected %lld\n", #x, got_, static_cast<long long>(want)); ++fails; } } while (0)

// Runs f with stdout going to a pipe and returns what it printed.
template <class F>
static std::string printed(F f, int* status) {
  std::fflush(stdout);
  int p[2];
  if (pipe(p) != 0) return "<pipe failed>";
  const int saved = dup(1);
  dup2(p[1], 1);
  *status = static_cast<int>(f());
  std::fflush(stdout);
  dup2(saved, 1);
  close(saved);
  close(p[1]);
  std::string out;
  char buf[4096];
  for (ssize_t n; (n = read(p[0], buf, sizeof buf)) > 0;) out.append(buf, static_cast<size_t>(n));
  close(p[0]);
  return out;
}

static std::string line(const char* field, int param) {
  char b[160];
  std::snprintf(b, sizeof b, " ** On entry to %s parameter number %d had an illegal value\n", field, param);
  return b;
}

// One refused (or accepted) call: its status and the line it prints, if any (field null: none).
#define CASE(label, want_status, field, param, call) do { int st_ = -1; \
  const std::string got_ = printed([&] { return (call); }, &st_); \
  const std::string want_ = (field) ? line((field), (param)) : std::string(); \
  if (st_ != static_cast<int>(want_status) || got_ != want_) { ++fails; \
    std::printf("FAIL %s: status %d, expected %d; printed \"%s\", expected \"%s\"\n", label, st_, \
                static_cast<int>(want_status), got_.c_str(), want_.c_str()); } } while (0)

static const char* const kNone = nullptr;

int main() {
  cublasHandle_t h;
  WANT(cublasCreate(&h), CUBLAS_STATUS_SUCCESS);
  const size_t bytes = 4096 * sizeof(double2);
  void *pa = nullptr, *px = nullptr, *py = nullptr;
  WANT(cudaMalloc(&pa, bytes), cudaSuccess);
  WANT(cudaMalloc(&px, bytes), cudaSuccess);
  WANT(cudaMalloc(&py, bytes), cudaSuccess);
  WANT(cudaMemset(pa, 0, bytes), cudaSuccess);
  WANT(cudaMemset(px, 0, bytes), cudaSuccess);
  WANT(cudaMemset(py, 0, bytes), cudaSuccess);
  float *A = static_cast<float*>(pa), *x = static_cast<float*>(px), *y = static_cast<float*>(py);
  double *dA = static_cast<double*>(pa), *dx = static_cast<double*>(px), *dy = static_cast<double*>(py);
  cuComplex *cA = static_cast<cuComplex*>(pa), *cx = static_cast<cuComplex*>(px), *cy = static_cast<cuComplex*>(py);
  cuDoubleComplex *zA = static_cast<cuDoubleComplex*>(pa), *zx = static_cast<cuDoubleComplex*>(px),
                  *zy = static_cast<cuDoubleComplex*>(py);
  const float al = 1, be = 0, ra = 1;
  const double dal = 1, dbe = 0, dra = 1;
  const cuComplex cal{1, 0}, cbe{0, 0};
  const cuDoubleComplex zal{1, 0}, zbe{0, 0};
  const auto N = CUBLAS_OP_N;
  const auto U = CUBLAS_FILL_MODE_UPPER;
  const auto L = CUBLAS_SIDE_LEFT;
  const auto D = CUBLAS_DIAG_NON_UNIT;
  const auto badop = static_cast<cublasOperation_t>(7);
  const auto badfill = static_cast<cublasFillMode_t>(5);
  const auto baddiag = static_cast<cublasDiagType_t>(9);
  const int OK = CUBLAS_STATUS_SUCCESS, BAD = CUBLAS_STATUS_INVALID_VALUE;
  (void)L;

  // gemv: silent, and only a short lda is refused.
  CASE("sgemv m=-1", OK, kNone, 0, cublasSgemv(h, N, -1, 4, &al, A, 4, x, 1, &be, y, 1));
  CASE("sgemv n=-1", OK, kNone, 0, cublasSgemv(h, N, 4, -1, &al, A, 4, x, 1, &be, y, 1));
  CASE("sgemv m=-1 lda=0", OK, kNone, 0, cublasSgemv(h, N, -1, 4, &al, A, 0, x, 1, &be, y, 1));
  CASE("sgemv n=0 lda=3", OK, kNone, 0, cublasSgemv(h, N, 4, 0, &al, A, 3, x, 1, &be, y, 1));
  CASE("sgemv lda=3", BAD, kNone, 0, cublasSgemv(h, N, 4, 4, &al, A, 3, x, 1, &be, y, 1));
  CASE("sgemv T lda=3", BAD, kNone, 0, cublasSgemv(h, CUBLAS_OP_T, 4, 4, &al, A, 3, x, 1, &be, y, 1));
  CASE("sgemv incx=0", OK, kNone, 0, cublasSgemv(h, N, 4, 4, &al, A, 4, x, 0, &be, y, 1));
  CASE("sgemv incy=0", OK, kNone, 0, cublasSgemv(h, N, 4, 4, &al, A, 4, x, 1, &be, y, 0));
  CASE("sgemv trans=7", OK, kNone, 0, cublasSgemv(h, badop, 4, 4, &al, A, 4, x, 1, &be, y, 1));
  CASE("dgemv m=-1", OK, kNone, 0, cublasDgemv(h, N, -1, 4, &dal, dA, 4, dx, 1, &dbe, dy, 1));
  CASE("dgemv incy=0", OK, kNone, 0, cublasDgemv(h, N, 4, 4, &dal, dA, 4, dx, 1, &dbe, dy, 0));
  CASE("dgemv lda=3", BAD, kNone, 0, cublasDgemv(h, N, 4, 4, &dal, dA, 3, dx, 1, &dbe, dy, 1));
  CASE("cgemv m=-1", OK, kNone, 0, cublasCgemv(h, N, -1, 4, &cal, cA, 4, cx, 1, &cbe, cy, 1));
  CASE("cgemv incx=0", OK, kNone, 0, cublasCgemv(h, N, 4, 4, &cal, cA, 4, cx, 0, &cbe, cy, 1));
  CASE("cgemv lda=3", BAD, kNone, 0, cublasCgemv(h, N, 4, 4, &cal, cA, 3, cx, 1, &cbe, cy, 1));
  CASE("zgemv n=-1", OK, kNone, 0, cublasZgemv(h, N, 4, -1, &zal, zA, 4, zx, 1, &zbe, zy, 1));
  CASE("zgemv incy=0", OK, kNone, 0, cublasZgemv(h, N, 4, 4, &zal, zA, 4, zx, 1, &zbe, zy, 0));
  CASE("sgemv_64 incx=0", OK, kNone, 0, cublasSgemv_64(h, N, 4, 4, &al, A, 4, x, 0, &be, y, 1));
  CASE("sgemv_64 m=-1", OK, kNone, 0, cublasSgemv_64(h, N, -1, 4, &al, A, 4, x, 1, &be, y, 1));

  // symv and hemv: refused, silently.
  CASE("ssymv n=-1", BAD, kNone, 0, cublasSsymv(h, U, -1, &al, A, 4, x, 1, &be, y, 1));
  CASE("ssymv incx=0", BAD, kNone, 0, cublasSsymv(h, U, 4, &al, A, 4, x, 0, &be, y, 1));
  CASE("ssymv uplo=5", BAD, kNone, 0, cublasSsymv(h, badfill, 4, &al, A, 4, x, 1, &be, y, 1));
  CASE("dsymv lda=3", BAD, kNone, 0, cublasDsymv(h, U, 4, &dal, dA, 3, dx, 1, &dbe, dy, 1));
  CASE("csymv n=-1", BAD, kNone, 0, cublasCsymv(h, U, -1, &cal, cA, 4, cx, 1, &cbe, cy, 1));
  CASE("chemv incy=0", BAD, kNone, 0, cublasChemv(h, U, 4, &cal, cA, 4, cx, 1, &cbe, cy, 0));
  CASE("zhemv n=-1", BAD, kNone, 0, cublasZhemv(h, U, -1, &zal, zA, 4, zx, 1, &zbe, zy, 1));

  // gbmv: trans 1, m 2, n 3, kl 4, ku 5, lda 8, incx 10, incy 13.
  CASE("sgbmv trans=7", BAD, "SGBMV ", 1, cublasSgbmv(h, badop, 4, 4, 1, 1, &al, A, 3, x, 1, &be, y, 1));
  CASE("sgbmv m=-1", BAD, "SGBMV ", 2, cublasSgbmv(h, N, -1, 4, 1, 1, &al, A, 3, x, 1, &be, y, 1));
  CASE("sgbmv n=-1", BAD, "SGBMV ", 3, cublasSgbmv(h, N, 4, -1, 1, 1, &al, A, 3, x, 1, &be, y, 1));
  CASE("sgbmv kl=-1", BAD, "SGBMV ", 4, cublasSgbmv(h, N, 4, 4, -1, 1, &al, A, 3, x, 1, &be, y, 1));
  CASE("sgbmv ku=-1", BAD, "SGBMV ", 5, cublasSgbmv(h, N, 4, 4, 1, -1, &al, A, 3, x, 1, &be, y, 1));
  CASE("sgbmv lda=2", BAD, "SGBMV ", 8, cublasSgbmv(h, N, 4, 4, 1, 1, &al, A, 2, x, 1, &be, y, 1));
  CASE("sgbmv incx=0", BAD, "SGBMV ", 10, cublasSgbmv(h, N, 4, 4, 1, 1, &al, A, 3, x, 0, &be, y, 1));
  CASE("sgbmv incy=0", BAD, "SGBMV ", 13, cublasSgbmv(h, N, 4, 4, 1, 1, &al, A, 3, x, 1, &be, y, 0));
  CASE("sgbmv kl=-1 incx=0", BAD, "SGBMV ", 4, cublasSgbmv(h, N, 4, 4, -1, 1, &al, A, 3, x, 0, &be, y, 1));
  CASE("sgbmv m=0 incx=0", BAD, "SGBMV ", 10, cublasSgbmv(h, N, 0, 4, 1, 1, &al, A, 3, x, 0, &be, y, 1));
  CASE("dgbmv m=-1", BAD, "DGBMV ", 2, cublasDgbmv(h, N, -1, 4, 1, 1, &dal, dA, 3, dx, 1, &dbe, dy, 1));
  CASE("cgbmv lda=2", BAD, "CGBMV ", 8, cublasCgbmv(h, N, 4, 4, 1, 1, &cal, cA, 2, cx, 1, &cbe, cy, 1));
  CASE("zgbmv incy=0", BAD, "ZGBMV ", 13, cublasZgbmv(h, N, 4, 4, 1, 1, &zal, zA, 3, zx, 1, &zbe, zy, 0));

  // sbmv/hbmv: uplo 1, n 2, k 3, lda 6, incx 8, incy 11.
  CASE("ssbmv uplo=5", BAD, "SSBMV ", 1, cublasSsbmv(h, badfill, 4, 1, &al, A, 2, x, 1, &be, y, 1));
  CASE("ssbmv n=-1", BAD, "SSBMV ", 2, cublasSsbmv(h, U, -1, 1, &al, A, 2, x, 1, &be, y, 1));
  CASE("ssbmv k=-1", BAD, "SSBMV ", 3, cublasSsbmv(h, U, 4, -1, &al, A, 2, x, 1, &be, y, 1));
  CASE("ssbmv lda=1", BAD, "SSBMV ", 6, cublasSsbmv(h, U, 4, 1, &al, A, 1, x, 1, &be, y, 1));
  CASE("ssbmv incx=0", BAD, "SSBMV ", 8, cublasSsbmv(h, U, 4, 1, &al, A, 2, x, 0, &be, y, 1));
  CASE("ssbmv incy=0", BAD, "SSBMV ", 11, cublasSsbmv(h, U, 4, 1, &al, A, 2, x, 1, &be, y, 0));
  CASE("dsbmv n=-1", BAD, "DSBMV ", 2, cublasDsbmv(h, U, -1, 1, &dal, dA, 2, dx, 1, &dbe, dy, 1));
  CASE("chbmv k=-1", BAD, "CHBMV ", 3, cublasChbmv(h, U, 4, -1, &cal, cA, 2, cx, 1, &cbe, cy, 1));
  CASE("zhbmv incx=0", BAD, "ZHBMV ", 8, cublasZhbmv(h, U, 4, 1, &zal, zA, 2, zx, 0, &zbe, zy, 1));

  // spmv/hpmv: uplo 1, n 2, incx 6, incy 9.
  CASE("sspmv uplo=5", BAD, "SSPMV ", 1, cublasSspmv(h, badfill, 4, &al, A, x, 1, &be, y, 1));
  CASE("sspmv n=-1", BAD, "SSPMV ", 2, cublasSspmv(h, U, -1, &al, A, x, 1, &be, y, 1));
  CASE("sspmv incx=0", BAD, "SSPMV ", 6, cublasSspmv(h, U, 4, &al, A, x, 0, &be, y, 1));
  CASE("dspmv incy=0", BAD, "DSPMV ", 9, cublasDspmv(h, U, 4, &dal, dA, dx, 1, &dbe, dy, 0));
  CASE("chpmv incx=0", BAD, "CHPMV ", 6, cublasChpmv(h, U, 4, &cal, cA, cx, 0, &cbe, cy, 1));
  CASE("zhpmv n=-1", BAD, "ZHPMV ", 2, cublasZhpmv(h, U, -1, &zal, zA, zx, 1, &zbe, zy, 1));

  // trmv/trsv: uplo 1, trans 2, diag 3, n 4, lda 6, incx 8.
  CASE("strmv uplo=5", BAD, "STRMV ", 1, cublasStrmv(h, badfill, N, D, 4, A, 4, x, 1));
  CASE("strmv trans=7", BAD, "STRMV ", 2, cublasStrmv(h, U, badop, D, 4, A, 4, x, 1));
  CASE("strmv diag=9", BAD, "STRMV ", 3, cublasStrmv(h, U, N, baddiag, 4, A, 4, x, 1));
  CASE("strmv n=-1", BAD, "STRMV ", 4, cublasStrmv(h, U, N, D, -1, A, 4, x, 1));
  CASE("strmv lda=3", BAD, "STRMV ", 6, cublasStrmv(h, U, N, D, 4, A, 3, x, 1));
  CASE("strmv incx=0", BAD, "STRMV ", 8, cublasStrmv(h, U, N, D, 4, A, 4, x, 0));
  CASE("strmv n=0 lda=0", BAD, "STRMV ", 6, cublasStrmv(h, U, N, D, 0, A, 0, x, 1));
  CASE("strmv n=0 incx=0", BAD, "STRMV ", 8, cublasStrmv(h, U, N, D, 0, A, 4, x, 0));
  CASE("dtrmv lda=3", BAD, "DTRMV ", 6, cublasDtrmv(h, U, N, D, 4, dA, 3, dx, 1));
  CASE("ctrmv incx=0", BAD, "CTRMV ", 8, cublasCtrmv(h, U, N, D, 4, cA, 4, cx, 0));
  CASE("strsv trans=7", BAD, "STRSV ", 2, cublasStrsv(h, U, badop, D, 4, A, 4, x, 1));
  CASE("strsv n=-1", BAD, "STRSV ", 4, cublasStrsv(h, U, N, D, -1, A, 4, x, 1));
  CASE("strsv lda=3", BAD, "STRSV ", 6, cublasStrsv(h, U, N, D, 4, A, 3, x, 1));
  CASE("ztrsv incx=0", BAD, "ZTRSV ", 8, cublasZtrsv(h, U, N, D, 4, zA, 4, zx, 0));

  // tbmv/tbsv: uplo 1, trans 2, diag 3, n 4, k 5, lda 7, incx 9.
  CASE("stbmv diag=9", BAD, "STBMV ", 3, cublasStbmv(h, U, N, baddiag, 4, 1, A, 2, x, 1));
  CASE("stbmv k=-1", BAD, "STBMV ", 5, cublasStbmv(h, U, N, D, 4, -1, A, 2, x, 1));
  CASE("stbmv lda=1", BAD, "STBMV ", 7, cublasStbmv(h, U, N, D, 4, 1, A, 1, x, 1));
  CASE("stbmv incx=0", BAD, "STBMV ", 9, cublasStbmv(h, U, N, D, 4, 1, A, 2, x, 0));
  CASE("dtbsv n=-1", BAD, "DTBSV ", 4, cublasDtbsv(h, U, N, D, -1, 1, dA, 2, dx, 1));
  CASE("ctbsv k=-1", BAD, "CTBSV ", 5, cublasCtbsv(h, U, N, D, 4, -1, cA, 2, cx, 1));
  CASE("ztbsv incx=0", BAD, "ZTBSV ", 9, cublasZtbsv(h, U, N, D, 4, 1, zA, 2, zx, 0));

  // tpmv/tpsv: uplo 1, trans 2, diag 3, n 4, incx 7.
  CASE("stpmv uplo=5", BAD, "STPMV ", 1, cublasStpmv(h, badfill, N, D, 4, A, x, 1));
  CASE("stpmv n=-1", BAD, "STPMV ", 4, cublasStpmv(h, U, N, D, -1, A, x, 1));
  CASE("stpmv incx=0", BAD, "STPMV ", 7, cublasStpmv(h, U, N, D, 4, A, x, 0));
  CASE("stpsv trans=7", BAD, "STPSV ", 2, cublasStpsv(h, U, badop, D, 4, A, x, 1));
  CASE("ctpsv n=-1", BAD, "CTPSV ", 4, cublasCtpsv(h, U, N, D, -1, cA, cx, 1));
  CASE("dtpsv incx=0", BAD, "DTPSV ", 7, cublasDtpsv(h, U, N, D, 4, dA, dx, 0));

  // ger, geru, gerc: m 1, n 2, incx 5, incy 7, lda 9.
  CASE("sger m=-1", BAD, "SGER  ", 1, cublasSger(h, -1, 4, &al, x, 1, y, 1, A, 4));
  CASE("sger n=-1", BAD, "SGER  ", 2, cublasSger(h, 4, -1, &al, x, 1, y, 1, A, 4));
  CASE("sger incx=0", BAD, "SGER  ", 5, cublasSger(h, 4, 4, &al, x, 0, y, 1, A, 4));
  CASE("sger incy=0", BAD, "SGER  ", 7, cublasSger(h, 4, 4, &al, x, 1, y, 0, A, 4));
  CASE("sger lda=3", BAD, "SGER  ", 9, cublasSger(h, 4, 4, &al, x, 1, y, 1, A, 3));
  CASE("sger m=-1 incx=0", BAD, "SGER  ", 1, cublasSger(h, -1, 4, &al, x, 0, y, 1, A, 4));
  CASE("sger m=0 incx=0", BAD, "SGER  ", 5, cublasSger(h, 0, 4, &al, x, 0, y, 1, A, 4));
  CASE("sger m=0 lda=0", BAD, "SGER  ", 9, cublasSger(h, 0, 4, &al, x, 1, y, 1, A, 0));
  CASE("dger m=-1", BAD, "DGER  ", 1, cublasDger(h, -1, 4, &dal, dx, 1, dy, 1, dA, 4));
  CASE("cgeru incx=0", BAD, "CGERU ", 5, cublasCgeru(h, 4, 4, &cal, cx, 0, cy, 1, cA, 4));
  CASE("cgerc m=-1", BAD, "CGERC ", 1, cublasCgerc(h, -1, 4, &cal, cx, 1, cy, 1, cA, 4));
  CASE("zgeru lda=3", BAD, "ZGERU ", 9, cublasZgeru(h, 4, 4, &zal, zx, 1, zy, 1, zA, 3));
  CASE("zgerc incy=0", BAD, "ZGERC ", 7, cublasZgerc(h, 4, 4, &zal, zx, 1, zy, 0, zA, 4));
  CASE("sger_64 incx=0", BAD, "SGER  ", 5, cublasSger_64(h, 4, 4, &al, x, 0, y, 1, A, 4));

  // syr, her: uplo 1, n 2, incx 5, lda 7.
  CASE("ssyr uplo=5", BAD, "SSYR  ", 1, cublasSsyr(h, badfill, 4, &al, x, 1, A, 4));
  CASE("ssyr n=-1", BAD, "SSYR  ", 2, cublasSsyr(h, U, -1, &al, x, 1, A, 4));
  CASE("ssyr incx=0", BAD, "SSYR  ", 5, cublasSsyr(h, U, 4, &al, x, 0, A, 4));
  CASE("ssyr lda=3", BAD, "SSYR  ", 7, cublasSsyr(h, U, 4, &al, x, 1, A, 3));
  CASE("dsyr n=-1", BAD, "DSYR  ", 2, cublasDsyr(h, U, -1, &dal, dx, 1, dA, 4));
  CASE("csyr n=-1", BAD, "CSYR  ", 2, cublasCsyr(h, U, -1, &cal, cx, 1, cA, 4));
  CASE("zsyr lda=3", BAD, "ZSYR  ", 7, cublasZsyr(h, U, 4, &zal, zx, 1, zA, 3));
  CASE("cher incx=0", BAD, "CHER  ", 5, cublasCher(h, U, 4, &ra, cx, 0, cA, 4));
  CASE("zher lda=3", BAD, "ZHER  ", 7, cublasZher(h, U, 4, &dra, zx, 1, zA, 3));

  // syr2, her2: uplo 1, n 2, incx 5, incy 7, lda 9.
  CASE("ssyr2 n=-1", BAD, "SSYR2  ", 2, cublasSsyr2(h, U, -1, &al, x, 1, y, 1, A, 4));
  CASE("ssyr2 incy=0", BAD, "SSYR2  ", 7, cublasSsyr2(h, U, 4, &al, x, 1, y, 0, A, 4));
  CASE("ssyr2 lda=3", BAD, "SSYR2  ", 9, cublasSsyr2(h, U, 4, &al, x, 1, y, 1, A, 3));
  CASE("dsyr2 n=-1", BAD, "DSYR2  ", 2, cublasDsyr2(h, U, -1, &dal, dx, 1, dy, 1, dA, 4));
  CASE("csyr2 incx=0", BAD, "CSYR2  ", 5, cublasCsyr2(h, U, 4, &cal, cx, 0, cy, 1, cA, 4));
  CASE("cher2 incx=0", BAD, "CHER2  ", 5, cublasCher2(h, U, 4, &cal, cx, 0, cy, 1, cA, 4));
  CASE("zher2 n=-1", BAD, "ZHER2  ", 2, cublasZher2(h, U, -1, &zal, zx, 1, zy, 1, zA, 4));

  // spr, hpr: uplo 1, n 2, incx 5.  spr2, hpr2: uplo 1, n 2, incx 5, incy 7.
  CASE("sspr uplo=5", BAD, "SSPR  ", 1, cublasSspr(h, badfill, 4, &al, x, 1, A));
  CASE("sspr n=-1", BAD, "SSPR  ", 2, cublasSspr(h, U, -1, &al, x, 1, A));
  CASE("dspr incx=0", BAD, "DSPR  ", 5, cublasDspr(h, U, 4, &dal, dx, 0, dA));
  CASE("chpr incx=0", BAD, "CHPR  ", 5, cublasChpr(h, U, 4, &ra, cx, 0, cA));
  CASE("zhpr n=-1", BAD, "ZHPR  ", 2, cublasZhpr(h, U, -1, &dra, zx, 1, zA));
  CASE("sspr2 n=-1", BAD, "SSPR2  ", 2, cublasSspr2(h, U, -1, &al, x, 1, y, 1, A));
  CASE("sspr2 incx=0", BAD, "SSPR2  ", 5, cublasSspr2(h, U, 4, &al, x, 0, y, 1, A));
  CASE("dspr2 incy=0", BAD, "DSPR2  ", 7, cublasDspr2(h, U, 4, &dal, dx, 1, dy, 0, dA));
  CASE("chpr2 incy=0", BAD, "CHPR2  ", 7, cublasChpr2(h, U, 4, &cal, cx, 1, cy, 0, cA));
  CASE("zhpr2 n=-1", BAD, "ZHPR2  ", 2, cublasZhpr2(h, U, -1, &zal, zx, 1, zy, 1, zA));

  // GEMM: transa 1, transb 2, m 3, n 4, k 5, lda 8, ldb 10, ldc 13.
  CASE("sgemm transa=7", BAD, "SGEMM ", 1, cublasSgemm(h, badop, N, 4, 4, 4, &al, A, 4, x, 4, &be, y, 4));
  CASE("sgemm transb=7", BAD, "SGEMM ", 2, cublasSgemm(h, N, badop, 4, 4, 4, &al, A, 4, x, 4, &be, y, 4));
  CASE("sgemm m=-1", BAD, "SGEMM ", 3, cublasSgemm(h, N, N, -1, 4, 4, &al, A, 4, x, 4, &be, y, 4));
  CASE("sgemm n=-1", BAD, "SGEMM ", 4, cublasSgemm(h, N, N, 4, -1, 4, &al, A, 4, x, 4, &be, y, 4));
  CASE("sgemm k=-1", BAD, "SGEMM ", 5, cublasSgemm(h, N, N, 4, 4, -1, &al, A, 4, x, 4, &be, y, 4));
  CASE("sgemm lda=3", BAD, "SGEMM ", 8, cublasSgemm(h, N, N, 4, 4, 4, &al, A, 3, x, 4, &be, y, 4));
  CASE("sgemm T lda=3", BAD, "SGEMM ", 8, cublasSgemm(h, CUBLAS_OP_T, N, 4, 4, 4, &al, A, 3, x, 4, &be, y, 4));
  CASE("sgemm ldb=3", BAD, "SGEMM ", 10, cublasSgemm(h, N, N, 4, 4, 4, &al, A, 4, x, 3, &be, y, 4));
  CASE("sgemm ldc=3", BAD, "SGEMM ", 13, cublasSgemm(h, N, N, 4, 4, 4, &al, A, 4, x, 4, &be, y, 3));
  CASE("sgemm m=-1 lda=0", BAD, "SGEMM ", 3, cublasSgemm(h, N, N, -1, 4, 4, &al, A, 0, x, 4, &be, y, 4));
  CASE("sgemm m=0 lda=0", BAD, "SGEMM ", 8, cublasSgemm(h, N, N, 0, 4, 4, &al, A, 0, x, 4, &be, y, 1));
  CASE("sgemm null C", BAD, kNone, 0, cublasSgemm(h, N, N, 2, 2, 0, &al, nullptr, 2, nullptr, 1, &be, nullptr, 2));
  CASE("sgemm ok", OK, kNone, 0, cublasSgemm(h, N, N, 4, 4, 4, &al, A, 4, x, 4, &be, y, 4));
  CASE("sgemm_64 m=-1", BAD, "SGEMM ", 3, cublasSgemm_64(h, N, N, -1, 4, 4, &al, A, 4, x, 4, &be, y, 4));
  CASE("dgemm ldc=3", BAD, "DGEMM ", 13, cublasDgemm(h, N, N, 4, 4, 4, &dal, dA, 4, dx, 4, &dbe, dy, 3));
  CASE("cgemm m=-1", BAD, "CGEMM ", 3, cublasCgemm(h, N, N, -1, 4, 4, &cal, cA, 4, cx, 4, &cbe, cy, 4));
  CASE("zgemm transb=7", BAD, "ZGEMM ", 2, cublasZgemm(h, N, badop, 4, 4, 4, &zal, zA, 4, zx, 4, &zbe, zy, 4));
  CASE("cgemm3m m=-1", BAD, "CGEMM ", 3, cublasCgemm3m(h, N, N, -1, 4, 4, &cal, cA, 4, cx, 4, &cbe, cy, 4));
  CASE("zgemm3m ldb=3", BAD, "ZGEMM3M ", 10, cublasZgemm3m(h, N, N, 4, 4, 4, &zal, zA, 4, zx, 3, &zbe, zy, 4));
  {
    const __half hal = __float2half(1.0f), hbe = __float2half(0.0f);
    __half* hA = static_cast<__half*>(pa);
    CASE("hgemm m=-1", BAD, "HGEMM ", 3, cublasHgemm(h, N, N, -1, 4, 4, &hal, hA, 4, hA, 4, &hbe, hA, 4));
    CASE("hgemm lda=3", BAD, "HGEMM ", 8, cublasHgemm(h, N, N, 4, 4, 4, &hal, hA, 3, hA, 4, &hbe, hA, 4));
  }

  // What gemv computes with the arguments it takes. A = 1..16 by columns, x = 1..4, y = 1..4, alpha 1, beta 2.
  {
    float hA[16], hx[4], hy[4];
    for (int i = 0; i < 16; ++i) hA[i] = static_cast<float>(i + 1);
    for (int i = 0; i < 4; ++i) hx[i] = hy[i] = static_cast<float>(i + 1);
    WANT(cudaMemcpy(A, hA, sizeof hA, cudaMemcpyHostToDevice), cudaSuccess);
    WANT(cudaMemcpy(x, hx, sizeof hx, cudaMemcpyHostToDevice), cudaSuccess);
    const float one = 1, two = 2;
    auto run = [&](cublasOperation_t t, int m, int incx, int incy, const float (&want)[4]) {
      WANT(cudaMemcpy(y, hy, sizeof hy, cudaMemcpyHostToDevice), cudaSuccess);
      WANT(cublasSgemv(h, t, m, 4, &one, A, 4, x, incx, &two, y, incy), CUBLAS_STATUS_SUCCESS);
      float got[4];
      WANT(cudaMemcpy(got, y, sizeof got, cudaMemcpyDeviceToHost), cudaSuccess);
      for (int i = 0; i < 4; ++i)
        if (got[i] != want[i]) {
          std::printf("FAIL sgemv trans %d m %d incx %d incy %d: y[%d] = %g, expected %g\n", static_cast<int>(t), m,
                      incx, incy, i, got[i], want[i]);
          ++fails;
        }
    };
    run(N, 4, 1, 1, {92, 104, 116, 128});
    run(CUBLAS_OP_T, 4, 1, 1, {32, 74, 116, 158});
    run(CUBLAS_OP_C, 4, 1, 1, {32, 74, 116, 158});
    run(static_cast<cublasOperation_t>(3), 4, 1, 1, {92, 104, 116, 128});  // CONJG: A itself
    run(badop, 4, 1, 1, {32, 74, 116, 158});                               // unknown: a transpose
    run(static_cast<cublasOperation_t>(-1), 4, 1, 1, {32, 74, 116, 158});
    run(N, 4, 0, 1, {30, 36, 42, 48});    // x[0] for every element: y_i = sum_j A(i,j) + 2 y_i
    run(CUBLAS_OP_T, 4, 0, 1, {12, 30, 48, 66});
    run(N, 4, 1, 0, {92, 2, 3, 4});       // row 0's result in y[0]
    run(N, -1, 1, 1, {1, 2, 3, 4});       // empty
  }
  {
    // Complex: OP_CONJG is conj(A) not transposed; an unknown operation is a transpose. A(i,j) = (i+1) + (j+1)i, 2x2.
    cuComplex hA[4] = {{1, 1}, {2, 1}, {1, 2}, {2, 2}}, hx[2] = {{1, 0}, {0, 1}};
    WANT(cudaMemcpy(cA, hA, sizeof hA, cudaMemcpyHostToDevice), cudaSuccess);
    WANT(cudaMemcpy(cx, hx, sizeof hx, cudaMemcpyHostToDevice), cudaSuccess);
    const cuComplex one{1, 0}, zero{0, 0};
    auto run = [&](int t, const cuComplex (&want)[2]) {
      WANT(cudaMemset(cy, 0, 2 * sizeof(cuComplex)), cudaSuccess);
      WANT(cublasCgemv(h, static_cast<cublasOperation_t>(t), 2, 2, &one, cA, 2, cx, 1, &zero, cy, 1),
           CUBLAS_STATUS_SUCCESS);
      cuComplex got[2];
      WANT(cudaMemcpy(got, cy, sizeof got, cudaMemcpyDeviceToHost), cudaSuccess);
      for (int i = 0; i < 2; ++i)
        if (got[i].x != want[i].x || got[i].y != want[i].y) {
          std::printf("FAIL cgemv trans %d: y[%d] = (%g, %g), expected (%g, %g)\n", t, i, got[i].x, got[i].y,
                      want[i].x, want[i].y);
          ++fails;
        }
    };
    // y = op(A) x with x = (1, i).
    run(0, {{-1, 2}, {0, 3}});   // A x: (1+i) + i(1+2i) = -1+2i; (2+i) + i(2+2i) = 0+3i
    run(1, {{0, 3}, {-1, 4}});   // A^T x: (1+i) + i(2+i) = 0+3i; (1+2i) + i(2+2i) = -1+4i
    run(2, {{2, 1}, {3, 0}});    // A^H x: (1-i) + i(2-i) = 2+i; (1-2i) + i(2-2i) = 3+0i
    run(3, {{3, 0}, {4, 1}});    // conj(A) x: (1-i) + i(1-2i) = 3+0i; (2-i) + i(2-2i) = 4+i
    run(7, {{0, 3}, {-1, 4}});   // unknown: A^T
  }

  WANT(cudaFree(pa), cudaSuccess);
  WANT(cudaFree(px), cudaSuccess);
  WANT(cudaFree(py), cudaSuccess);
  WANT(cublasDestroy(h), CUBLAS_STATUS_SUCCESS);
  std::printf("%s\n", fails ? "FAILED" : "PASS");
  return fails ? 1 : 0;
}
