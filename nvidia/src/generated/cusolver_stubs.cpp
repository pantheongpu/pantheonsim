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
  // =1 silences, as nvidia/include/vgpu_cuda.h documents and every hand-written shim
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
VGPU_EXPORT int cusolverDnCsytrf() { vgpu_report_unimplemented("cusolverDnCsytrf"); return 31; }
VGPU_EXPORT int cusolverDnCsytrf_bufferSize() { vgpu_report_unimplemented("cusolverDnCsytrf_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnCungqr() { vgpu_report_unimplemented("cusolverDnCungqr"); return 31; }
VGPU_EXPORT int cusolverDnCungqr_bufferSize() { vgpu_report_unimplemented("cusolverDnCungqr_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnCunmqr() { vgpu_report_unimplemented("cusolverDnCunmqr"); return 31; }
VGPU_EXPORT int cusolverDnCunmqr_bufferSize() { vgpu_report_unimplemented("cusolverDnCunmqr_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnDsytrf() { vgpu_report_unimplemented("cusolverDnDsytrf"); return 31; }
VGPU_EXPORT int cusolverDnDsytrf_bufferSize() { vgpu_report_unimplemented("cusolverDnDsytrf_bufferSize"); return 31; }
VGPU_EXPORT int cusolverDnSsytrf() { vgpu_report_unimplemented("cusolverDnSsytrf"); return 31; }
VGPU_EXPORT int cusolverDnSsytrf_bufferSize() { vgpu_report_unimplemented("cusolverDnSsytrf_bufferSize"); return 31; }
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
