// cuSOLVER's sparse API (cusolverSp) inside a captured CUDA graph (see graph_capture_common.h).
// run_graph_capture.sh solver_sp cusolver,cusparse --card runs the same program on NVIDIA's cuSOLVER.
//
// Measured on an RTX 3060 (CUDA 13.0), the device entry points that work on the stream are captured -- the
// low-level Cholesky's csrcholFactor, csrcholSolve and csrcholDiag, the low-level QR's csrqrSetup, csrqrFactor and
// csrqrSolve, and csrqrsvBatched -- and run at each launch over what the graph's own kernels wrote. The buffer
// size queries (csrcholBufferInfo, csrqrBufferInfo, csrqrBufferInfoBatched) and the host entry points (the ...Host
// forms) do not touch the stream, and leave the capture as it was. What waits for the device is refused: the
// answer is CUSOLVER_STATUS_INTERNAL_ERROR (7) and the capture is invalidated -- csrlsvqr, csrlsvchol, csreigvsi,
// csrcholZeroPivot, csrqrZeroPivot and the three analyses (csrcholAnalysis, csrqrAnalysis, csrqrAnalysisBatched).
// After a refused capture the library's later calls in the process are not reliable (they answer errors, or
// fault), so each refusal runs in a process of its own: this program starts itself once for each.
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <functional>

#include "graph_capture_common.h"

using namespace gc;

#define OK(x) do { cusolverStatus_t s_ = (x); if (s_ != CUSOLVER_STATUS_SUCCESS) { \
  std::printf("     %s -> %d\n", #x, (int)s_); ok = false; } } while (0)

constexpr int n = 12;

// The values of a symmetric positive definite tridiagonal matrix (CSR, rows' columns ascending), the diagonal
// from the counter; `batch` matrices one after the other.
__global__ void make_values(float* val, const int* rowptr, const int* colind, int rows, int nnz, int batch, const int* counter) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nnz * batch) return;
  const int e = k % nnz, which = k / nnz;
  int row = 0;
  while (rowptr[row + 1] <= e) ++row;
  val[k] = colind[e] == row ? 4.0f + 0.25f * pattern(static_cast<size_t>(row), *counter, 1 + which, 1) : -1.0f;
}

struct Csr {
  std::vector<int> rowptr, colind;
  int nnz = 0;
  int *d_rowptr = nullptr, *d_colind = nullptr;
  Csr() {
    rowptr.push_back(0);
    for (int i = 0; i < n; ++i) {
      for (int j = std::max(0, i - 1); j <= std::min(n - 1, i + 1); ++j) colind.push_back(j);
      rowptr.push_back(static_cast<int>(colind.size()));
    }
    nnz = static_cast<int>(colind.size());
    cudaMalloc(&d_rowptr, rowptr.size() * 4);
    cudaMalloc(&d_colind, colind.size() * 4);
    cudaMemcpy(d_rowptr, rowptr.data(), rowptr.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_colind, colind.data(), colind.size() * 4, cudaMemcpyHostToDevice);
  }
  ~Csr() { cudaFree(d_rowptr); cudaFree(d_colind); }
};

static void values(cudaStream_t st, const int* counter, const Csr& m, float* val, int batch) {
  make_values<<<(m.nnz * batch + 127) / 128, 128, 0, st>>>(val, m.d_rowptr, m.d_colind, n, m.nnz, batch, counter);
}

/* ---- the captured calls -------------------------------------------------------------------- */

static void captured(Runner& r) {
  Csr m;
  cusolverSpHandle_t h;
  cusolverSpCreate(&h);
  cusolverSpSetStream(h, r.st);
  cusparseMatDescr_t d;
  cusparseCreateMatDescr(&d);
  float* val = r.alloc<float>(m.nnz * 2);
  float *b = r.alloc<float>(n * 2), *x = r.alloc<float>(n * 2), *dg = r.alloc<float>(n);
  {
    csrcholInfo_t info;
    cusolverSpCreateCsrcholInfo(&info);
    cusolverSpXcsrcholAnalysis(h, n, m.nnz, d, m.d_rowptr, m.d_colind, info);
    size_t internal = 0, work = 0;
    cusolverSpScsrcholBufferInfo(h, n, m.nnz, d, val, m.d_rowptr, m.d_colind, info, &internal, &work);
    void* buf = r.alloc<char>(work + 1024);
    // csrcholFactor allocates its scratch memory on every call, which a capture in the global mode refuses (checked
    // below, in a process of its own): so this one is captured in the relaxed mode.
    gc::capture_mode = cudaStreamCaptureModeRelaxed;
    r.run("cusolverSpScsrcholFactor + csrcholSolve + csrcholDiag", [&] {
      bool ok = true;
      values(r.st, r.counter, m, val, 1);
      fill(r.st, r.counter, b, n, 1.0f, 0, 5);
      OK(cusolverSpScsrcholFactor(h, n, m.nnz, d, val, m.d_rowptr, m.d_colind, info, buf));
      OK(cusolverSpScsrcholSolve(h, n, b, x, info, buf));
      OK(cusolverSpScsrcholDiag(h, info, dg));
      return ok;
    }, {{x, (size_t)n, Dt::F32}, {dg, (size_t)n, Dt::F32}}, 1e-4);
    gc::capture_mode = cudaStreamCaptureModeGlobal;
    cusolverSpDestroyCsrcholInfo(info);
  }
  {
    csrqrInfo_t info;
    cusolverSpCreateCsrqrInfo(&info);
    cusolverSpXcsrqrAnalysis(h, n, n, m.nnz, d, m.d_rowptr, m.d_colind, info);
    size_t internal = 0, work = 0;
    cusolverSpScsrqrBufferInfo(h, n, n, m.nnz, d, val, m.d_rowptr, m.d_colind, info, &internal, &work);
    void* buf = r.alloc<char>(internal + work + 1024);
    r.run("cusolverSpScsrqrSetup + csrqrFactor + csrqrSolve", [&] {
      bool ok = true;
      values(r.st, r.counter, m, val, 1);
      fill(r.st, r.counter, b, n, 1.0f, 0, 5);
      OK(cusolverSpScsrqrSetup(h, n, n, m.nnz, d, val, m.d_rowptr, m.d_colind, 0.5f, info));
      OK(cusolverSpScsrqrFactor(h, n, n, m.nnz, nullptr, nullptr, info, buf));
      OK(cusolverSpScsrqrSolve(h, n, n, b, x, info, buf));
      return ok;
    }, {{x, (size_t)n, Dt::F32}}, 1e-4);
    // The factor and the solve in one call, on b and x.
    r.run("cusolverSpScsrqrSetup + csrqrFactor (solving b)", [&] {
      bool ok = true;
      values(r.st, r.counter, m, val, 1);
      fill(r.st, r.counter, b, n, 1.0f, 0, 5);
      OK(cusolverSpScsrqrSetup(h, n, n, m.nnz, d, val, m.d_rowptr, m.d_colind, 0.0f, info));
      OK(cusolverSpScsrqrFactor(h, n, n, m.nnz, b, x, info, buf));
      return ok;
    }, {{x, (size_t)n, Dt::F32}}, 1e-4);
    cusolverSpDestroyCsrqrInfo(info);
  }
  {
    csrqrInfo_t info;
    cusolverSpCreateCsrqrInfo(&info);
    cusolverSpXcsrqrAnalysisBatched(h, n, n, m.nnz, d, m.d_rowptr, m.d_colind, info);
    size_t internal = 0, work = 0;
    cusolverSpScsrqrBufferInfoBatched(h, n, n, m.nnz, d, val, m.d_rowptr, m.d_colind, 2, info, &internal, &work);
    void* buf = r.alloc<char>(internal + work + 1024);
    r.run("cusolverSpScsrqrsvBatched", [&] {
      bool ok = true;
      values(r.st, r.counter, m, val, 2);
      fill(r.st, r.counter, b, n * 2, 1.0f, 0, 5);
      OK(cusolverSpScsrqrsvBatched(h, n, n, m.nnz, d, val, m.d_rowptr, m.d_colind, b, x, 2, info, buf));
      return ok;
    }, {{x, (size_t)n * 2, Dt::F32}}, 1e-4);
    cusolverSpDestroyCsrqrInfo(info);
  }
  // The calls that do not touch the stream leave a capture as it was.
  {
    csrcholInfo_t info;
    cusolverSpCreateCsrcholInfo(&info);
    cusolverSpXcsrcholAnalysis(h, n, m.nnz, d, m.d_rowptr, m.d_colind, info);
    size_t internal = 0, work = 0;
    std::vector<float> hv(m.nnz, 1.0f), hb(n, 1.0f), hx(n);
    int sing = 0;
    cudaGraph_t g = nullptr;
    cudaStreamBeginCapture(r.st, cudaStreamCaptureModeGlobal);
    const int s1 = (int)cusolverSpScsrcholBufferInfo(h, n, m.nnz, d, val, m.d_rowptr, m.d_colind, info, &internal, &work);
    for (int i = 0; i < m.nnz; ++i) hv[i] = m.colind[i] == 0 ? 4.0f : -1.0f;
    for (int i = 0, k = 0; i < n; ++i)
      for (int j = std::max(0, i - 1); j <= std::min(n - 1, i + 1); ++j, ++k) hv[k] = i == j ? 4.0f : -1.0f;
    const int s2 = (int)cusolverSpScsrlsvcholHost(h, n, m.nnz, d, hv.data(), m.rowptr.data(), m.colind.data(), hb.data(), 1e-6f, 0, hx.data(), &sing);
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(r.st, &status);
    const cudaError_t e = cudaStreamEndCapture(r.st, &g);
    if (g) cudaGraphDestroy(g);
    expect("csrcholBufferInfo and the Host forms in a capture: they answer, and leave the capture valid",
           s1 == 0 && s2 == 0 && sing == -1 && status == cudaStreamCaptureStatusActive && e == cudaSuccess, s1 * 100 + s2);
    cusolverSpDestroyCsrcholInfo(info);
  }
  cusparseDestroyMatDescr(d);
  cusolverSpDestroy(h);
}

/* ---- the refused calls, each in a process of its own --------------------------------------- */

static const char* const kRefused[] = {"cusolverSpScsrlsvqr", "cusolverSpScsrlsvchol", "cusolverSpScsreigvsi",
                                       "cusolverSpScsrcholZeroPivot", "cusolverSpScsrqrZeroPivot",
                                       "cusolverSpXcsrcholAnalysis", "cusolverSpXcsrqrAnalysis", "cusolverSpXcsrqrAnalysisBatched",
                                       "cusolverSpScsrcholFactor (the global capture mode)"};
constexpr int kRefusedCount = sizeof(kRefused) / sizeof(kRefused[0]);

// The child: one refusal. Prints "eager=<status of the call made eagerly, or -1> status=<in a capture> state=<the stream's
// capture status afterwards> end=<cudaStreamEndCapture>".
static int refusal_child(int idx) {
  cudaFree(nullptr);
  Csr m;
  cusolverSpHandle_t h;
  cusolverSpCreate(&h);
  cudaStream_t st;
  cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
  cusolverSpSetStream(h, st);
  cusparseMatDescr_t d;
  cusparseCreateMatDescr(&d);
  std::vector<float> hv(m.nnz), hb(n, 1.0f);
  for (int i = 0, k = 0; i < n; ++i)
    for (int j = std::max(0, i - 1); j <= std::min(n - 1, i + 1); ++j, ++k) hv[k] = i == j ? 4.0f : -1.0f;
  float *v, *bb, *xx;
  cudaMalloc(&v, m.nnz * 4);
  cudaMalloc(&bb, n * 4);
  cudaMalloc(&xx, n * 4);
  cudaMemcpy(v, hv.data(), m.nnz * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(bb, hb.data(), n * 4, cudaMemcpyHostToDevice);
  cudaMemset(xx, 0, n * 4);
  void* buf = nullptr;
  cudaMalloc(&buf, 1 << 22);
  csrcholInfo_t ci;
  csrqrInfo_t qi, bi;
  cusolverSpCreateCsrcholInfo(&ci);
  cusolverSpCreateCsrqrInfo(&qi);
  cusolverSpCreateCsrqrInfo(&bi);
  int sing = 0, pivot = 0;
  float* mu = nullptr;   // on the device, as csreigvsi writes it
  cudaMalloc(&mu, 4);
  size_t internal = 0, work = 0;
  // What each call needs before it, made eagerly.
  if (idx == 3 || idx == 8) {
    cusolverSpXcsrcholAnalysis(h, n, m.nnz, d, m.d_rowptr, m.d_colind, ci);
    cusolverSpScsrcholBufferInfo(h, n, m.nnz, d, v, m.d_rowptr, m.d_colind, ci, &internal, &work);
    cusolverSpScsrcholFactor(h, n, m.nnz, d, v, m.d_rowptr, m.d_colind, ci, buf);
  } else if (idx == 4) {
    cusolverSpXcsrqrAnalysis(h, n, n, m.nnz, d, m.d_rowptr, m.d_colind, qi);
    cusolverSpScsrqrBufferInfo(h, n, n, m.nnz, d, v, m.d_rowptr, m.d_colind, qi, &internal, &work);
    cusolverSpScsrqrSetup(h, n, n, m.nnz, d, v, m.d_rowptr, m.d_colind, 0.0f, qi);
    cusolverSpScsrqrFactor(h, n, n, m.nnz, nullptr, nullptr, qi, buf);
  }
  cudaStreamSynchronize(st);
  auto call = [&]() -> int {
    switch (idx) {
      case 0: return (int)cusolverSpScsrlsvqr(h, n, m.nnz, d, v, m.d_rowptr, m.d_colind, bb, 1e-6f, 0, xx, &sing);
      case 1: return (int)cusolverSpScsrlsvchol(h, n, m.nnz, d, v, m.d_rowptr, m.d_colind, bb, 1e-6f, 0, xx, &sing);
      case 2: return (int)cusolverSpScsreigvsi(h, n, m.nnz, d, v, m.d_rowptr, m.d_colind, 1.0f, bb, 10, 1e-5f, mu, xx);
      case 3: return (int)cusolverSpScsrcholZeroPivot(h, ci, 1e-6f, &pivot);
      case 4: return (int)cusolverSpScsrqrZeroPivot(h, qi, 1e-6f, &pivot);
      case 5: return (int)cusolverSpXcsrcholAnalysis(h, n, m.nnz, d, m.d_rowptr, m.d_colind, ci);
      case 6: return (int)cusolverSpXcsrqrAnalysis(h, n, n, m.nnz, d, m.d_rowptr, m.d_colind, qi);
      case 7: return (int)cusolverSpXcsrqrAnalysisBatched(h, n, n, m.nnz, d, m.d_rowptr, m.d_colind, bi);
      default: return (int)cusolverSpScsrcholFactor(h, n, m.nnz, d, v, m.d_rowptr, m.d_colind, ci, buf);
    }
  };
  // The analyses are made once on an info: the eager call is on the info the capture then gets a fresh twin of.
  int eager = -1;
  if (idx < 5 || idx == 8) {
    eager = call();
    cudaStreamSynchronize(st);
  } else {
    cusolverSpDestroyCsrcholInfo(ci);
    cusolverSpDestroyCsrqrInfo(qi);
    cusolverSpDestroyCsrqrInfo(bi);
    cusolverSpCreateCsrcholInfo(&ci);
    cusolverSpCreateCsrqrInfo(&qi);
    cusolverSpCreateCsrqrInfo(&bi);
  }
  cudaGraph_t g = nullptr;
  const cudaError_t begin = cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
  const int rc = call();
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(st, &status);
  const cudaError_t end = cudaStreamEndCapture(st, &g);
  std::printf("RESULT begin=%d eager=%d status=%d state=%d end=%d\n", (int)begin, eager, rc, (int)status, (int)end);
  return 0;
}

// Runs this program as the child for refusal `idx` and returns what it printed after "RESULT ".
static std::string run_child(int idx) {
  char self[4096] = {0};
  const ssize_t len = readlink("/proc/self/exe", self, sizeof self - 1);
  if (len <= 0) return "";
  const std::string cmd = "GC_SP_CASE=" + std::to_string(idx) + " '" + std::string(self, static_cast<size_t>(len)) + "' 2>&1";
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) return "";
  std::string line, result;
  char buf[512];
  while (std::fgets(buf, sizeof buf, p)) {
    line = buf;
    if (line.compare(0, 7, "RESULT ") == 0) result = line.substr(7);
  }
  pclose(p);
  while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) result.pop_back();
  return result;
}

int main() {
  if (const char* c = std::getenv("GC_SP_CASE")) return refusal_child(std::atoi(c));
  {
    Runner r;
    captured(r);
  }
  for (int i = 0; i < kRefusedCount; ++i) {
    const std::string res = run_child(i);
    int begin = -1, eager = -2, status = -1, state = -1, end = -1;
    const bool parsed = std::sscanf(res.c_str(), "begin=%d eager=%d status=%d state=%d end=%d", &begin, &eager, &status, &state, &end) == 5;
    // 7 is CUSOLVER_STATUS_INTERNAL_ERROR, 2 ALLOC_FAILED; cudaStreamCaptureStatusInvalidated is 2; 901 is
    // cudaErrorStreamCaptureInvalidated.
    const int answer = i == 8 ? 2 : 7;
    expect(std::string(kRefused[i]) + " in a capture: " + (i == 8 ? "ALLOC_FAILED" : "INTERNAL_ERROR") + ", and the capture is invalidated",
           parsed && begin == 0 && status == answer && state == cudaStreamCaptureStatusInvalidated && end == cudaErrorStreamCaptureInvalidated &&
               (i >= 5 && i != 8 ? true : eager == 0),
           parsed ? status : -1);
    if (!parsed) std::printf("     the child answered: '%s'\n", res.c_str());
  }
  return finish();
}
