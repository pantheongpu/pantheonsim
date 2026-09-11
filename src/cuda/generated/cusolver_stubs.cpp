// GENERATED -- entry points cusolver exports that VirtualGPU does not implement.
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
  // =1 silences, as include/vgpu_cuda.h documents and every hand-written shim
  // reads it. This used to test for presence, so VGPU_QUIET=0 meant quiet here
  // and not quiet everywhere else.
  if (const char* q = std::getenv("VGPU_QUIET"); q && q[0] == '1') return;
  std::fprintf(stderr,
               "[vgpu] %s is not implemented by VirtualGPU; returning CUSOLVER_STATUS_NOT_SUPPORTED.\n"
               "       The call site will see a failure rather than a wrong answer.\n", fn);
}
}  // namespace

VGPU_EXPORT int cusolverDnCgeqrf() { vgpu_report_unimplemented("cusolverDnCgeqrf"); return 31; }
VGPU_EXPORT int cusolverDnCgeqrf_bufferSize() { vgpu_report_unimplemented("cusolverDnCgeqrf_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnCgesvd() { vgpu_report_unimplemented("cusolverDnCgesvd"); return 31; }
VGPU_EXPORT int cusolverDnCgesvd_bufferSize() { vgpu_report_unimplemented("cusolverDnCgesvd_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnCgesvdaStridedBatched() { vgpu_report_unimplemented("cusolverDnCgesvdaStridedBatched"); return 31; }
VGPU_EXPORT int cusolverDnCgesvdaStridedBatched_bufferSize() { vgpu_report_unimplemented("cusolverDnCgesvdaStridedBatched_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnCgesvdj() { vgpu_report_unimplemented("cusolverDnCgesvdj"); return 31; }
VGPU_EXPORT int cusolverDnCgesvdjBatched() { vgpu_report_unimplemented("cusolverDnCgesvdjBatched"); return 31; }
VGPU_EXPORT int cusolverDnCgesvdjBatched_bufferSize() { vgpu_report_unimplemented("cusolverDnCgesvdjBatched_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnCgesvdj_bufferSize() { vgpu_report_unimplemented("cusolverDnCgesvdj_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnCgetrf() { vgpu_report_unimplemented("cusolverDnCgetrf"); return 31; }
VGPU_EXPORT int cusolverDnCgetrf_bufferSize() { vgpu_report_unimplemented("cusolverDnCgetrf_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnCgetrs() { vgpu_report_unimplemented("cusolverDnCgetrs"); return 31; }
VGPU_EXPORT int cusolverDnCheevd() { vgpu_report_unimplemented("cusolverDnCheevd"); return 31; }
VGPU_EXPORT int cusolverDnCheevd_bufferSize() { vgpu_report_unimplemented("cusolverDnCheevd_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnCheevj() { vgpu_report_unimplemented("cusolverDnCheevj"); return 31; }
VGPU_EXPORT int cusolverDnCheevjBatched() { vgpu_report_unimplemented("cusolverDnCheevjBatched"); return 31; }
VGPU_EXPORT int cusolverDnCheevjBatched_bufferSize() { vgpu_report_unimplemented("cusolverDnCheevjBatched_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnCheevj_bufferSize() { vgpu_report_unimplemented("cusolverDnCheevj_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnCpotrf() { vgpu_report_unimplemented("cusolverDnCpotrf"); return 31; }
VGPU_EXPORT int cusolverDnCpotrfBatched() { vgpu_report_unimplemented("cusolverDnCpotrfBatched"); return 31; }
VGPU_EXPORT int cusolverDnCpotrf_bufferSize() { vgpu_report_unimplemented("cusolverDnCpotrf_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnCpotrs() { vgpu_report_unimplemented("cusolverDnCpotrs"); return 31; }
VGPU_EXPORT int cusolverDnCpotrsBatched() { vgpu_report_unimplemented("cusolverDnCpotrsBatched"); return 31; }
VGPU_EXPORT int cusolverDnCreateGesvdjInfo() { vgpu_report_unimplemented("cusolverDnCreateGesvdjInfo"); return 31; }
VGPU_EXPORT int cusolverDnCreateParams() { vgpu_report_unimplemented("cusolverDnCreateParams"); return 31; }
VGPU_EXPORT int cusolverDnCreateSyevjInfo() { vgpu_report_unimplemented("cusolverDnCreateSyevjInfo"); return 31; }
VGPU_EXPORT int cusolverDnCsytrf() { vgpu_report_unimplemented("cusolverDnCsytrf"); return 31; }
VGPU_EXPORT int cusolverDnCsytrf_bufferSize() { vgpu_report_unimplemented("cusolverDnCsytrf_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnCungqr() { vgpu_report_unimplemented("cusolverDnCungqr"); return 31; }
VGPU_EXPORT int cusolverDnCungqr_bufferSize() { vgpu_report_unimplemented("cusolverDnCungqr_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnCunmqr() { vgpu_report_unimplemented("cusolverDnCunmqr"); return 31; }
VGPU_EXPORT int cusolverDnCunmqr_bufferSize() { vgpu_report_unimplemented("cusolverDnCunmqr_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnDestroyGesvdjInfo() { vgpu_report_unimplemented("cusolverDnDestroyGesvdjInfo"); return 31; }
VGPU_EXPORT int cusolverDnDestroyParams() { vgpu_report_unimplemented("cusolverDnDestroyParams"); return 31; }
VGPU_EXPORT int cusolverDnDestroySyevjInfo() { vgpu_report_unimplemented("cusolverDnDestroySyevjInfo"); return 31; }
VGPU_EXPORT int cusolverDnDgesvdaStridedBatched() { vgpu_report_unimplemented("cusolverDnDgesvdaStridedBatched"); return 31; }
VGPU_EXPORT int cusolverDnDgesvdaStridedBatched_bufferSize() { vgpu_report_unimplemented("cusolverDnDgesvdaStridedBatched_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnDgesvdj() { vgpu_report_unimplemented("cusolverDnDgesvdj"); return 31; }
VGPU_EXPORT int cusolverDnDgesvdjBatched() { vgpu_report_unimplemented("cusolverDnDgesvdjBatched"); return 31; }
VGPU_EXPORT int cusolverDnDgesvdjBatched_bufferSize() { vgpu_report_unimplemented("cusolverDnDgesvdjBatched_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnDgesvdj_bufferSize() { vgpu_report_unimplemented("cusolverDnDgesvdj_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnDpotrfBatched() { vgpu_report_unimplemented("cusolverDnDpotrfBatched"); return 31; }
VGPU_EXPORT int cusolverDnDpotrsBatched() { vgpu_report_unimplemented("cusolverDnDpotrsBatched"); return 31; }
VGPU_EXPORT int cusolverDnDsyevj() { vgpu_report_unimplemented("cusolverDnDsyevj"); return 31; }
VGPU_EXPORT int cusolverDnDsyevjBatched() { vgpu_report_unimplemented("cusolverDnDsyevjBatched"); return 31; }
VGPU_EXPORT int cusolverDnDsyevjBatched_bufferSize() { vgpu_report_unimplemented("cusolverDnDsyevjBatched_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnDsyevj_bufferSize() { vgpu_report_unimplemented("cusolverDnDsyevj_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnDsytrf() { vgpu_report_unimplemented("cusolverDnDsytrf"); return 31; }
VGPU_EXPORT int cusolverDnDsytrf_bufferSize() { vgpu_report_unimplemented("cusolverDnDsytrf_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnSgesvdaStridedBatched() { vgpu_report_unimplemented("cusolverDnSgesvdaStridedBatched"); return 31; }
VGPU_EXPORT int cusolverDnSgesvdaStridedBatched_bufferSize() { vgpu_report_unimplemented("cusolverDnSgesvdaStridedBatched_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnSgesvdj() { vgpu_report_unimplemented("cusolverDnSgesvdj"); return 31; }
VGPU_EXPORT int cusolverDnSgesvdjBatched() { vgpu_report_unimplemented("cusolverDnSgesvdjBatched"); return 31; }
VGPU_EXPORT int cusolverDnSgesvdjBatched_bufferSize() { vgpu_report_unimplemented("cusolverDnSgesvdjBatched_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnSgesvdj_bufferSize() { vgpu_report_unimplemented("cusolverDnSgesvdj_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnSpotrfBatched() { vgpu_report_unimplemented("cusolverDnSpotrfBatched"); return 31; }
VGPU_EXPORT int cusolverDnSpotrsBatched() { vgpu_report_unimplemented("cusolverDnSpotrsBatched"); return 31; }
VGPU_EXPORT int cusolverDnSsyevj() { vgpu_report_unimplemented("cusolverDnSsyevj"); return 31; }
VGPU_EXPORT int cusolverDnSsyevjBatched() { vgpu_report_unimplemented("cusolverDnSsyevjBatched"); return 31; }
VGPU_EXPORT int cusolverDnSsyevjBatched_bufferSize() { vgpu_report_unimplemented("cusolverDnSsyevjBatched_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnSsyevj_bufferSize() { vgpu_report_unimplemented("cusolverDnSsyevj_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnSsytrf() { vgpu_report_unimplemented("cusolverDnSsytrf"); return 31; }
VGPU_EXPORT int cusolverDnSsytrf_bufferSize() { vgpu_report_unimplemented("cusolverDnSsytrf_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnXgeqrf() { vgpu_report_unimplemented("cusolverDnXgeqrf"); return 31; }
VGPU_EXPORT int cusolverDnXgeqrf_bufferSize() { vgpu_report_unimplemented("cusolverDnXgeqrf_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnXgesvdjSetMaxSweeps() { vgpu_report_unimplemented("cusolverDnXgesvdjSetMaxSweeps"); return 31; }
VGPU_EXPORT int cusolverDnXgesvdjSetSortEig() { vgpu_report_unimplemented("cusolverDnXgesvdjSetSortEig"); return 31; }
VGPU_EXPORT int cusolverDnXgesvdjSetTolerance() { vgpu_report_unimplemented("cusolverDnXgesvdjSetTolerance"); return 31; }
VGPU_EXPORT int cusolverDnXpotrf() { vgpu_report_unimplemented("cusolverDnXpotrf"); return 31; }
VGPU_EXPORT int cusolverDnXpotrf_bufferSize() { vgpu_report_unimplemented("cusolverDnXpotrf_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnXpotrs() { vgpu_report_unimplemented("cusolverDnXpotrs"); return 31; }
VGPU_EXPORT int cusolverDnXsyevd() { vgpu_report_unimplemented("cusolverDnXsyevd"); return 31; }
VGPU_EXPORT int cusolverDnXsyevd_bufferSize() { vgpu_report_unimplemented("cusolverDnXsyevd_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnXsyevjSetSortEig() { vgpu_report_unimplemented("cusolverDnXsyevjSetSortEig"); return 31; }
VGPU_EXPORT int cusolverDnXsytrs() { vgpu_report_unimplemented("cusolverDnXsytrs"); return 31; }
VGPU_EXPORT int cusolverDnXsytrs_bufferSize() { vgpu_report_unimplemented("cusolverDnXsytrs_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnZgeqrf() { vgpu_report_unimplemented("cusolverDnZgeqrf"); return 31; }
VGPU_EXPORT int cusolverDnZgeqrf_bufferSize() { vgpu_report_unimplemented("cusolverDnZgeqrf_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnZgesvd() { vgpu_report_unimplemented("cusolverDnZgesvd"); return 31; }
VGPU_EXPORT int cusolverDnZgesvd_bufferSize() { vgpu_report_unimplemented("cusolverDnZgesvd_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnZgesvdaStridedBatched() { vgpu_report_unimplemented("cusolverDnZgesvdaStridedBatched"); return 31; }
VGPU_EXPORT int cusolverDnZgesvdaStridedBatched_bufferSize() { vgpu_report_unimplemented("cusolverDnZgesvdaStridedBatched_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnZgesvdj() { vgpu_report_unimplemented("cusolverDnZgesvdj"); return 31; }
VGPU_EXPORT int cusolverDnZgesvdjBatched() { vgpu_report_unimplemented("cusolverDnZgesvdjBatched"); return 31; }
VGPU_EXPORT int cusolverDnZgesvdjBatched_bufferSize() { vgpu_report_unimplemented("cusolverDnZgesvdjBatched_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnZgesvdj_bufferSize() { vgpu_report_unimplemented("cusolverDnZgesvdj_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnZgetrf() { vgpu_report_unimplemented("cusolverDnZgetrf"); return 31; }
VGPU_EXPORT int cusolverDnZgetrf_bufferSize() { vgpu_report_unimplemented("cusolverDnZgetrf_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnZgetrs() { vgpu_report_unimplemented("cusolverDnZgetrs"); return 31; }
VGPU_EXPORT int cusolverDnZheevd() { vgpu_report_unimplemented("cusolverDnZheevd"); return 31; }
VGPU_EXPORT int cusolverDnZheevd_bufferSize() { vgpu_report_unimplemented("cusolverDnZheevd_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnZheevj() { vgpu_report_unimplemented("cusolverDnZheevj"); return 31; }
VGPU_EXPORT int cusolverDnZheevjBatched() { vgpu_report_unimplemented("cusolverDnZheevjBatched"); return 31; }
VGPU_EXPORT int cusolverDnZheevjBatched_bufferSize() { vgpu_report_unimplemented("cusolverDnZheevjBatched_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnZheevj_bufferSize() { vgpu_report_unimplemented("cusolverDnZheevj_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnZpotrf() { vgpu_report_unimplemented("cusolverDnZpotrf"); return 31; }
VGPU_EXPORT int cusolverDnZpotrfBatched() { vgpu_report_unimplemented("cusolverDnZpotrfBatched"); return 31; }
VGPU_EXPORT int cusolverDnZpotrf_bufferSize() { vgpu_report_unimplemented("cusolverDnZpotrf_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnZpotrs() { vgpu_report_unimplemented("cusolverDnZpotrs"); return 31; }
VGPU_EXPORT int cusolverDnZpotrsBatched() { vgpu_report_unimplemented("cusolverDnZpotrsBatched"); return 31; }
VGPU_EXPORT int cusolverDnZsytrf() { vgpu_report_unimplemented("cusolverDnZsytrf"); return 31; }
VGPU_EXPORT int cusolverDnZsytrf_bufferSize() { vgpu_report_unimplemented("cusolverDnZsytrf_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnZungqr() { vgpu_report_unimplemented("cusolverDnZungqr"); return 31; }
VGPU_EXPORT int cusolverDnZungqr_bufferSize() { vgpu_report_unimplemented("cusolverDnZungqr_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnZunmqr() { vgpu_report_unimplemented("cusolverDnZunmqr"); return 31; }
VGPU_EXPORT int cusolverDnZunmqr_bufferSize() { vgpu_report_unimplemented("cusolverDnZunmqr_bufferSize"); return 31; }
