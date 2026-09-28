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

VGPU_EXPORT int cusparseCbsrmm() { vgpu_report_unimplemented("cusparseCbsrmm"); return 10; }
VGPU_EXPORT int cusparseCbsrmv() { vgpu_report_unimplemented("cusparseCbsrmv"); return 10; }
VGPU_EXPORT int cusparseCbsrsm2_analysis() { vgpu_report_unimplemented("cusparseCbsrsm2_analysis"); return 10; }
VGPU_EXPORT int cusparseCbsrsm2_bufferSize() { vgpu_report_unimplemented("cusparseCbsrsm2_bufferSize"); return 10; }
VGPU_EXPORT int cusparseCbsrsm2_solve() { vgpu_report_unimplemented("cusparseCbsrsm2_solve"); return 10; }
VGPU_EXPORT int cusparseCbsrsv2_analysis() { vgpu_report_unimplemented("cusparseCbsrsv2_analysis"); return 10; }
VGPU_EXPORT int cusparseCbsrsv2_bufferSize() { vgpu_report_unimplemented("cusparseCbsrsv2_bufferSize"); return 10; }
VGPU_EXPORT int cusparseCbsrsv2_solve() { vgpu_report_unimplemented("cusparseCbsrsv2_solve"); return 10; }
VGPU_EXPORT int cusparseCcsrgeam2() { vgpu_report_unimplemented("cusparseCcsrgeam2"); return 10; }
VGPU_EXPORT int cusparseCcsrgeam2_bufferSizeExt() { vgpu_report_unimplemented("cusparseCcsrgeam2_bufferSizeExt"); return 10; }
VGPU_EXPORT int cusparseCreateBsrsm2Info() { vgpu_report_unimplemented("cusparseCreateBsrsm2Info"); return 10; }
VGPU_EXPORT int cusparseCreateBsrsv2Info() { vgpu_report_unimplemented("cusparseCreateBsrsv2Info"); return 10; }
VGPU_EXPORT int cusparseDbsrmm() { vgpu_report_unimplemented("cusparseDbsrmm"); return 10; }
VGPU_EXPORT int cusparseDbsrmv() { vgpu_report_unimplemented("cusparseDbsrmv"); return 10; }
VGPU_EXPORT int cusparseDbsrsm2_analysis() { vgpu_report_unimplemented("cusparseDbsrsm2_analysis"); return 10; }
VGPU_EXPORT int cusparseDbsrsm2_bufferSize() { vgpu_report_unimplemented("cusparseDbsrsm2_bufferSize"); return 10; }
VGPU_EXPORT int cusparseDbsrsm2_solve() { vgpu_report_unimplemented("cusparseDbsrsm2_solve"); return 10; }
VGPU_EXPORT int cusparseDbsrsv2_analysis() { vgpu_report_unimplemented("cusparseDbsrsv2_analysis"); return 10; }
VGPU_EXPORT int cusparseDbsrsv2_bufferSize() { vgpu_report_unimplemented("cusparseDbsrsv2_bufferSize"); return 10; }
VGPU_EXPORT int cusparseDbsrsv2_solve() { vgpu_report_unimplemented("cusparseDbsrsv2_solve"); return 10; }
VGPU_EXPORT int cusparseDestroyBsrsm2Info() { vgpu_report_unimplemented("cusparseDestroyBsrsm2Info"); return 10; }
VGPU_EXPORT int cusparseDestroyBsrsv2Info() { vgpu_report_unimplemented("cusparseDestroyBsrsv2Info"); return 10; }
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
VGPU_EXPORT int cusparseSbsrmm() { vgpu_report_unimplemented("cusparseSbsrmm"); return 10; }
VGPU_EXPORT int cusparseSbsrmv() { vgpu_report_unimplemented("cusparseSbsrmv"); return 10; }
VGPU_EXPORT int cusparseSbsrsm2_analysis() { vgpu_report_unimplemented("cusparseSbsrsm2_analysis"); return 10; }
VGPU_EXPORT int cusparseSbsrsm2_bufferSize() { vgpu_report_unimplemented("cusparseSbsrsm2_bufferSize"); return 10; }
VGPU_EXPORT int cusparseSbsrsm2_solve() { vgpu_report_unimplemented("cusparseSbsrsm2_solve"); return 10; }
VGPU_EXPORT int cusparseSbsrsv2_analysis() { vgpu_report_unimplemented("cusparseSbsrsv2_analysis"); return 10; }
VGPU_EXPORT int cusparseSbsrsv2_bufferSize() { vgpu_report_unimplemented("cusparseSbsrsv2_bufferSize"); return 10; }
VGPU_EXPORT int cusparseSbsrsv2_solve() { vgpu_report_unimplemented("cusparseSbsrsv2_solve"); return 10; }
VGPU_EXPORT int cusparseXbsrsm2_zeroPivot() { vgpu_report_unimplemented("cusparseXbsrsm2_zeroPivot"); return 10; }
VGPU_EXPORT int cusparseXbsrsv2_zeroPivot() { vgpu_report_unimplemented("cusparseXbsrsv2_zeroPivot"); return 10; }
VGPU_EXPORT int cusparseZbsrmm() { vgpu_report_unimplemented("cusparseZbsrmm"); return 10; }
VGPU_EXPORT int cusparseZbsrmv() { vgpu_report_unimplemented("cusparseZbsrmv"); return 10; }
VGPU_EXPORT int cusparseZbsrsm2_analysis() { vgpu_report_unimplemented("cusparseZbsrsm2_analysis"); return 10; }
VGPU_EXPORT int cusparseZbsrsm2_bufferSize() { vgpu_report_unimplemented("cusparseZbsrsm2_bufferSize"); return 10; }
VGPU_EXPORT int cusparseZbsrsm2_solve() { vgpu_report_unimplemented("cusparseZbsrsm2_solve"); return 10; }
VGPU_EXPORT int cusparseZbsrsv2_analysis() { vgpu_report_unimplemented("cusparseZbsrsv2_analysis"); return 10; }
VGPU_EXPORT int cusparseZbsrsv2_bufferSize() { vgpu_report_unimplemented("cusparseZbsrsv2_bufferSize"); return 10; }
VGPU_EXPORT int cusparseZbsrsv2_solve() { vgpu_report_unimplemented("cusparseZbsrsv2_solve"); return 10; }
VGPU_EXPORT int cusparseZcsrgeam2() { vgpu_report_unimplemented("cusparseZcsrgeam2"); return 10; }
VGPU_EXPORT int cusparseZcsrgeam2_bufferSizeExt() { vgpu_report_unimplemented("cusparseZcsrgeam2_bufferSizeExt"); return 10; }
