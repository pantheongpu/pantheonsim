// GENERATED -- entry points cublas exports that VirtualGPU does not implement.
//
// A framework resolves every symbol in a library at load time, so one missing
// name stops the import before any work happens. These exist so that loading
// succeeds and anything actually exercised either runs for real (the
// implemented entry points elsewhere in this shim take precedence) or fails by
// name, loudly, here. Returning a success status instead would let a model
// carry on with whatever happened to be in its output buffer, which is the one
// outcome this project treats as worse than a crash.
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

namespace {
void vgpu_report_unimplemented(const char* fn) {
  static std::mutex mu;
  static std::set<std::string> seen;
  std::lock_guard<std::mutex> lock(mu);
  if (!seen.insert(fn).second) return;
  // =1 silences, as nvidia/include/vgpu_cuda.h documents and every hand-written shim
  // reads it. This used to test for presence, so VGPU_QUIET=0 meant quiet here
  // and not quiet everywhere else.
  if (const char* q = std::getenv("VGPU_QUIET"); q && q[0] == '1') return;
  std::fprintf(stderr,
               "[vgpu] %s is not implemented by VirtualGPU; returning CUBLAS_STATUS_NOT_SUPPORTED.\n"
               "       The call site will see a failure rather than a wrong answer.\n", fn);
}
}  // namespace

VGPU_EXPORT int cublasCgeqrfBatched() { vgpu_report_unimplemented("cublasCgeqrfBatched"); return 15; }
VGPU_EXPORT int cublasDgeqrfBatched() { vgpu_report_unimplemented("cublasDgeqrfBatched"); return 15; }
VGPU_EXPORT int cublasLtMatmul() { vgpu_report_unimplemented("cublasLtMatmul"); return 15; }
VGPU_EXPORT int cublasLtMatmulAlgoGetHeuristic() { vgpu_report_unimplemented("cublasLtMatmulAlgoGetHeuristic"); return 15; }
VGPU_EXPORT int cublasLtMatmulDescCreate() { vgpu_report_unimplemented("cublasLtMatmulDescCreate"); return 15; }
VGPU_EXPORT int cublasLtMatmulDescDestroy() { vgpu_report_unimplemented("cublasLtMatmulDescDestroy"); return 15; }
VGPU_EXPORT int cublasLtMatmulDescSetAttribute() { vgpu_report_unimplemented("cublasLtMatmulDescSetAttribute"); return 15; }
VGPU_EXPORT int cublasLtMatmulPreferenceCreate() { vgpu_report_unimplemented("cublasLtMatmulPreferenceCreate"); return 15; }
VGPU_EXPORT int cublasLtMatmulPreferenceDestroy() { vgpu_report_unimplemented("cublasLtMatmulPreferenceDestroy"); return 15; }
VGPU_EXPORT int cublasLtMatmulPreferenceSetAttribute() { vgpu_report_unimplemented("cublasLtMatmulPreferenceSetAttribute"); return 15; }
VGPU_EXPORT int cublasLtMatrixLayoutCreate() { vgpu_report_unimplemented("cublasLtMatrixLayoutCreate"); return 15; }
VGPU_EXPORT int cublasLtMatrixLayoutDestroy() { vgpu_report_unimplemented("cublasLtMatrixLayoutDestroy"); return 15; }
VGPU_EXPORT int cublasLtMatrixLayoutSetAttribute() { vgpu_report_unimplemented("cublasLtMatrixLayoutSetAttribute"); return 15; }
VGPU_EXPORT int cublasSgeqrfBatched() { vgpu_report_unimplemented("cublasSgeqrfBatched"); return 15; }
VGPU_EXPORT int cublasZgeqrfBatched() { vgpu_report_unimplemented("cublasZgeqrfBatched"); return 15; }
