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
  if (std::getenv("VGPU_QUIET")) return;
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
VGPU_EXPORT int cusparseCreateIdentityPermutation() { vgpu_report_unimplemented("cusparseCreateIdentityPermutation"); return 10; }
VGPU_EXPORT int cusparseCreateMatDescr() { vgpu_report_unimplemented("cusparseCreateMatDescr"); return 10; }
VGPU_EXPORT int cusparseCsrSetStridedBatch() { vgpu_report_unimplemented("cusparseCsrSetStridedBatch"); return 10; }
VGPU_EXPORT int cusparseDbsrmm() { vgpu_report_unimplemented("cusparseDbsrmm"); return 10; }
VGPU_EXPORT int cusparseDbsrmv() { vgpu_report_unimplemented("cusparseDbsrmv"); return 10; }
VGPU_EXPORT int cusparseDbsrsm2_analysis() { vgpu_report_unimplemented("cusparseDbsrsm2_analysis"); return 10; }
VGPU_EXPORT int cusparseDbsrsm2_bufferSize() { vgpu_report_unimplemented("cusparseDbsrsm2_bufferSize"); return 10; }
VGPU_EXPORT int cusparseDbsrsm2_solve() { vgpu_report_unimplemented("cusparseDbsrsm2_solve"); return 10; }
VGPU_EXPORT int cusparseDbsrsv2_analysis() { vgpu_report_unimplemented("cusparseDbsrsv2_analysis"); return 10; }
VGPU_EXPORT int cusparseDbsrsv2_bufferSize() { vgpu_report_unimplemented("cusparseDbsrsv2_bufferSize"); return 10; }
VGPU_EXPORT int cusparseDbsrsv2_solve() { vgpu_report_unimplemented("cusparseDbsrsv2_solve"); return 10; }
VGPU_EXPORT int cusparseDcsrgeam2() { vgpu_report_unimplemented("cusparseDcsrgeam2"); return 10; }
VGPU_EXPORT int cusparseDcsrgeam2_bufferSizeExt() { vgpu_report_unimplemented("cusparseDcsrgeam2_bufferSizeExt"); return 10; }
VGPU_EXPORT int cusparseDestroyBsrsm2Info() { vgpu_report_unimplemented("cusparseDestroyBsrsm2Info"); return 10; }
VGPU_EXPORT int cusparseDestroyBsrsv2Info() { vgpu_report_unimplemented("cusparseDestroyBsrsv2Info"); return 10; }
VGPU_EXPORT int cusparseDestroyMatDescr() { vgpu_report_unimplemented("cusparseDestroyMatDescr"); return 10; }
VGPU_EXPORT int cusparseDnMatSetStridedBatch() { vgpu_report_unimplemented("cusparseDnMatSetStridedBatch"); return 10; }
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
VGPU_EXPORT int cusparseSDDMM() { vgpu_report_unimplemented("cusparseSDDMM"); return 10; }
VGPU_EXPORT int cusparseSDDMM_bufferSize() { vgpu_report_unimplemented("cusparseSDDMM_bufferSize"); return 10; }
VGPU_EXPORT int cusparseSDDMM_preprocess() { vgpu_report_unimplemented("cusparseSDDMM_preprocess"); return 10; }
VGPU_EXPORT int cusparseSbsrmm() { vgpu_report_unimplemented("cusparseSbsrmm"); return 10; }
VGPU_EXPORT int cusparseSbsrmv() { vgpu_report_unimplemented("cusparseSbsrmv"); return 10; }
VGPU_EXPORT int cusparseSbsrsm2_analysis() { vgpu_report_unimplemented("cusparseSbsrsm2_analysis"); return 10; }
VGPU_EXPORT int cusparseSbsrsm2_bufferSize() { vgpu_report_unimplemented("cusparseSbsrsm2_bufferSize"); return 10; }
VGPU_EXPORT int cusparseSbsrsm2_solve() { vgpu_report_unimplemented("cusparseSbsrsm2_solve"); return 10; }
VGPU_EXPORT int cusparseSbsrsv2_analysis() { vgpu_report_unimplemented("cusparseSbsrsv2_analysis"); return 10; }
VGPU_EXPORT int cusparseSbsrsv2_bufferSize() { vgpu_report_unimplemented("cusparseSbsrsv2_bufferSize"); return 10; }
VGPU_EXPORT int cusparseSbsrsv2_solve() { vgpu_report_unimplemented("cusparseSbsrsv2_solve"); return 10; }
VGPU_EXPORT int cusparseScsrgeam2() { vgpu_report_unimplemented("cusparseScsrgeam2"); return 10; }
VGPU_EXPORT int cusparseScsrgeam2_bufferSizeExt() { vgpu_report_unimplemented("cusparseScsrgeam2_bufferSizeExt"); return 10; }
VGPU_EXPORT int cusparseSetMatDiagType() { vgpu_report_unimplemented("cusparseSetMatDiagType"); return 10; }
VGPU_EXPORT int cusparseSetMatFillMode() { vgpu_report_unimplemented("cusparseSetMatFillMode"); return 10; }
VGPU_EXPORT int cusparseSetMatIndexBase() { vgpu_report_unimplemented("cusparseSetMatIndexBase"); return 10; }
VGPU_EXPORT int cusparseSetMatType() { vgpu_report_unimplemented("cusparseSetMatType"); return 10; }
VGPU_EXPORT int cusparseSetPointerMode() { vgpu_report_unimplemented("cusparseSetPointerMode"); return 10; }
VGPU_EXPORT int cusparseSpGEMM_compute() { vgpu_report_unimplemented("cusparseSpGEMM_compute"); return 10; }
VGPU_EXPORT int cusparseSpGEMM_copy() { vgpu_report_unimplemented("cusparseSpGEMM_copy"); return 10; }
VGPU_EXPORT int cusparseSpGEMM_createDescr() { vgpu_report_unimplemented("cusparseSpGEMM_createDescr"); return 10; }
VGPU_EXPORT int cusparseSpGEMM_destroyDescr() { vgpu_report_unimplemented("cusparseSpGEMM_destroyDescr"); return 10; }
VGPU_EXPORT int cusparseSpGEMM_workEstimation() { vgpu_report_unimplemented("cusparseSpGEMM_workEstimation"); return 10; }
VGPU_EXPORT int cusparseSpMatSetAttribute() { vgpu_report_unimplemented("cusparseSpMatSetAttribute"); return 10; }
VGPU_EXPORT int cusparseSpSM_analysis() { vgpu_report_unimplemented("cusparseSpSM_analysis"); return 10; }
VGPU_EXPORT int cusparseSpSM_bufferSize() { vgpu_report_unimplemented("cusparseSpSM_bufferSize"); return 10; }
VGPU_EXPORT int cusparseSpSM_createDescr() { vgpu_report_unimplemented("cusparseSpSM_createDescr"); return 10; }
VGPU_EXPORT int cusparseSpSM_destroyDescr() { vgpu_report_unimplemented("cusparseSpSM_destroyDescr"); return 10; }
VGPU_EXPORT int cusparseSpSM_solve() { vgpu_report_unimplemented("cusparseSpSM_solve"); return 10; }
VGPU_EXPORT int cusparseSpSV_analysis() { vgpu_report_unimplemented("cusparseSpSV_analysis"); return 10; }
VGPU_EXPORT int cusparseSpSV_bufferSize() { vgpu_report_unimplemented("cusparseSpSV_bufferSize"); return 10; }
VGPU_EXPORT int cusparseSpSV_createDescr() { vgpu_report_unimplemented("cusparseSpSV_createDescr"); return 10; }
VGPU_EXPORT int cusparseSpSV_destroyDescr() { vgpu_report_unimplemented("cusparseSpSV_destroyDescr"); return 10; }
VGPU_EXPORT int cusparseSpSV_solve() { vgpu_report_unimplemented("cusparseSpSV_solve"); return 10; }
VGPU_EXPORT int cusparseXbsrsm2_zeroPivot() { vgpu_report_unimplemented("cusparseXbsrsm2_zeroPivot"); return 10; }
VGPU_EXPORT int cusparseXbsrsv2_zeroPivot() { vgpu_report_unimplemented("cusparseXbsrsv2_zeroPivot"); return 10; }
VGPU_EXPORT int cusparseXcoo2csr() { vgpu_report_unimplemented("cusparseXcoo2csr"); return 10; }
VGPU_EXPORT int cusparseXcoosortByRow() { vgpu_report_unimplemented("cusparseXcoosortByRow"); return 10; }
VGPU_EXPORT int cusparseXcoosort_bufferSizeExt() { vgpu_report_unimplemented("cusparseXcoosort_bufferSizeExt"); return 10; }
VGPU_EXPORT int cusparseXcsrgeam2Nnz() { vgpu_report_unimplemented("cusparseXcsrgeam2Nnz"); return 10; }
VGPU_EXPORT int cusparseXcsrsort() { vgpu_report_unimplemented("cusparseXcsrsort"); return 10; }
VGPU_EXPORT int cusparseXcsrsort_bufferSizeExt() { vgpu_report_unimplemented("cusparseXcsrsort_bufferSizeExt"); return 10; }
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
