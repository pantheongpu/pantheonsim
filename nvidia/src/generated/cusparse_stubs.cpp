// GENERATED -- entry points cusparse exports that VirtualGPU does not implement.
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
               "[vgpu] %s is not implemented by VirtualGPU; returning CUSPARSE_STATUS_NOT_SUPPORTED.\n"
               "       The call site will see a failure rather than a wrong answer.\n", fn);
}
}  // namespace

VGPU_EXPORT int cusparseLtDenseDescriptorInit() { vgpu_report_unimplemented("cusparseLtDenseDescriptorInit"); return 10; }
VGPU_EXPORT int cusparseLtInit() { vgpu_report_unimplemented("cusparseLtInit"); return 10; }
VGPU_EXPORT int cusparseLtMatDescriptorDestroy() { vgpu_report_unimplemented("cusparseLtMatDescriptorDestroy"); return 10; }
VGPU_EXPORT int cusparseLtMatmul() { vgpu_report_unimplemented("cusparseLtMatmul"); return 10; }
VGPU_EXPORT int cusparseLtMatmulAlgGetAttribute() { vgpu_report_unimplemented("cusparseLtMatmulAlgGetAttribute"); return 10; }
VGPU_EXPORT int cusparseLtMatmulAlgSelectionInit() { vgpu_report_unimplemented("cusparseLtMatmulAlgSelectionInit"); return 10; }
VGPU_EXPORT int cusparseLtMatmulAlgSetAttribute() { vgpu_report_unimplemented("cusparseLtMatmulAlgSetAttribute"); return 10; }
VGPU_EXPORT int cusparseLtMatmulDescSetAttribute() { vgpu_report_unimplemented("cusparseLtMatmulDescSetAttribute"); return 10; }
VGPU_EXPORT int cusparseLtMatmulDescriptorInit() { vgpu_report_unimplemented("cusparseLtMatmulDescriptorInit"); return 10; }
VGPU_EXPORT int cusparseLtMatmulGetWorkspace() { vgpu_report_unimplemented("cusparseLtMatmulGetWorkspace"); return 10; }
VGPU_EXPORT int cusparseLtMatmulPlanDestroy() { vgpu_report_unimplemented("cusparseLtMatmulPlanDestroy"); return 10; }
VGPU_EXPORT int cusparseLtMatmulPlanInit() { vgpu_report_unimplemented("cusparseLtMatmulPlanInit"); return 10; }
VGPU_EXPORT int cusparseLtMatmulSearch() { vgpu_report_unimplemented("cusparseLtMatmulSearch"); return 10; }
VGPU_EXPORT int cusparseLtSpMMACompress2() { vgpu_report_unimplemented("cusparseLtSpMMACompress2"); return 10; }
VGPU_EXPORT int cusparseLtSpMMACompressedSize2() { vgpu_report_unimplemented("cusparseLtSpMMACompressedSize2"); return 10; }
VGPU_EXPORT int cusparseLtStructuredDescriptorInit() { vgpu_report_unimplemented("cusparseLtStructuredDescriptorInit"); return 10; }
