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
  if (std::getenv("VGPU_QUIET")) return;
  std::fprintf(stderr,
               "[vgpu] %s is not implemented by VirtualGPU; returning CUBLAS_STATUS_NOT_SUPPORTED.\n"
               "       The call site will see a failure rather than a wrong answer.\n", fn);
}
}  // namespace

VGPU_EXPORT int cublasCaxpy_v2() { vgpu_report_unimplemented("cublasCaxpy_v2"); return 15; }
VGPU_EXPORT int cublasCcopy_v2() { vgpu_report_unimplemented("cublasCcopy_v2"); return 15; }
VGPU_EXPORT int cublasCdotc_v2() { vgpu_report_unimplemented("cublasCdotc_v2"); return 15; }
VGPU_EXPORT int cublasCdotu_v2() { vgpu_report_unimplemented("cublasCdotu_v2"); return 15; }
VGPU_EXPORT int cublasCgelsBatched() { vgpu_report_unimplemented("cublasCgelsBatched"); return 15; }
VGPU_EXPORT int cublasCgemmBatched() { vgpu_report_unimplemented("cublasCgemmBatched"); return 15; }
VGPU_EXPORT int cublasCgemmStridedBatched() { vgpu_report_unimplemented("cublasCgemmStridedBatched"); return 15; }
VGPU_EXPORT int cublasCgemm_v2() { vgpu_report_unimplemented("cublasCgemm_v2"); return 15; }
VGPU_EXPORT int cublasCgemv_v2() { vgpu_report_unimplemented("cublasCgemv_v2"); return 15; }
VGPU_EXPORT int cublasCgeqrfBatched() { vgpu_report_unimplemented("cublasCgeqrfBatched"); return 15; }
VGPU_EXPORT int cublasCgerc_v2() { vgpu_report_unimplemented("cublasCgerc_v2"); return 15; }
VGPU_EXPORT int cublasCgeru_v2() { vgpu_report_unimplemented("cublasCgeru_v2"); return 15; }
VGPU_EXPORT int cublasCgetrfBatched() { vgpu_report_unimplemented("cublasCgetrfBatched"); return 15; }
VGPU_EXPORT int cublasCgetrsBatched() { vgpu_report_unimplemented("cublasCgetrsBatched"); return 15; }
VGPU_EXPORT int cublasChemm_v2() { vgpu_report_unimplemented("cublasChemm_v2"); return 15; }
VGPU_EXPORT int cublasChemv_v2() { vgpu_report_unimplemented("cublasChemv_v2"); return 15; }
VGPU_EXPORT int cublasCher2_v2() { vgpu_report_unimplemented("cublasCher2_v2"); return 15; }
VGPU_EXPORT int cublasCher2k_v2() { vgpu_report_unimplemented("cublasCher2k_v2"); return 15; }
VGPU_EXPORT int cublasCher_v2() { vgpu_report_unimplemented("cublasCher_v2"); return 15; }
VGPU_EXPORT int cublasCherk_v2() { vgpu_report_unimplemented("cublasCherk_v2"); return 15; }
VGPU_EXPORT int cublasCrot_v2() { vgpu_report_unimplemented("cublasCrot_v2"); return 15; }
VGPU_EXPORT int cublasCrotg_v2() { vgpu_report_unimplemented("cublasCrotg_v2"); return 15; }
VGPU_EXPORT int cublasCscal_v2() { vgpu_report_unimplemented("cublasCscal_v2"); return 15; }
VGPU_EXPORT int cublasCsrot_v2() { vgpu_report_unimplemented("cublasCsrot_v2"); return 15; }
VGPU_EXPORT int cublasCsscal_v2() { vgpu_report_unimplemented("cublasCsscal_v2"); return 15; }
VGPU_EXPORT int cublasCswap_v2() { vgpu_report_unimplemented("cublasCswap_v2"); return 15; }
VGPU_EXPORT int cublasCsymm_v2() { vgpu_report_unimplemented("cublasCsymm_v2"); return 15; }
VGPU_EXPORT int cublasCsymv_v2() { vgpu_report_unimplemented("cublasCsymv_v2"); return 15; }
VGPU_EXPORT int cublasCsyr2_v2() { vgpu_report_unimplemented("cublasCsyr2_v2"); return 15; }
VGPU_EXPORT int cublasCsyr2k_v2() { vgpu_report_unimplemented("cublasCsyr2k_v2"); return 15; }
VGPU_EXPORT int cublasCsyr_v2() { vgpu_report_unimplemented("cublasCsyr_v2"); return 15; }
VGPU_EXPORT int cublasCsyrk_v2() { vgpu_report_unimplemented("cublasCsyrk_v2"); return 15; }
VGPU_EXPORT int cublasCtrmm_v2() { vgpu_report_unimplemented("cublasCtrmm_v2"); return 15; }
VGPU_EXPORT int cublasCtrmv_v2() { vgpu_report_unimplemented("cublasCtrmv_v2"); return 15; }
VGPU_EXPORT int cublasCtrsmBatched() { vgpu_report_unimplemented("cublasCtrsmBatched"); return 15; }
VGPU_EXPORT int cublasCtrsm_v2() { vgpu_report_unimplemented("cublasCtrsm_v2"); return 15; }
VGPU_EXPORT int cublasCtrsv_v2() { vgpu_report_unimplemented("cublasCtrsv_v2"); return 15; }
VGPU_EXPORT int cublasDasum_v2() { vgpu_report_unimplemented("cublasDasum_v2"); return 15; }
VGPU_EXPORT int cublasDaxpy_v2() { vgpu_report_unimplemented("cublasDaxpy_v2"); return 15; }
VGPU_EXPORT int cublasDcopy_v2() { vgpu_report_unimplemented("cublasDcopy_v2"); return 15; }
VGPU_EXPORT int cublasDdot_v2() { vgpu_report_unimplemented("cublasDdot_v2"); return 15; }
VGPU_EXPORT int cublasDgelsBatched() { vgpu_report_unimplemented("cublasDgelsBatched"); return 15; }
VGPU_EXPORT int cublasDgemmBatched() { vgpu_report_unimplemented("cublasDgemmBatched"); return 15; }
VGPU_EXPORT int cublasDgemmStridedBatched() { vgpu_report_unimplemented("cublasDgemmStridedBatched"); return 15; }
VGPU_EXPORT int cublasDgemv_v2() { vgpu_report_unimplemented("cublasDgemv_v2"); return 15; }
VGPU_EXPORT int cublasDgeqrfBatched() { vgpu_report_unimplemented("cublasDgeqrfBatched"); return 15; }
VGPU_EXPORT int cublasDger_v2() { vgpu_report_unimplemented("cublasDger_v2"); return 15; }
VGPU_EXPORT int cublasDgetrfBatched() { vgpu_report_unimplemented("cublasDgetrfBatched"); return 15; }
VGPU_EXPORT int cublasDgetrsBatched() { vgpu_report_unimplemented("cublasDgetrsBatched"); return 15; }
VGPU_EXPORT int cublasDnrm2_v2() { vgpu_report_unimplemented("cublasDnrm2_v2"); return 15; }
VGPU_EXPORT int cublasDotEx() { vgpu_report_unimplemented("cublasDotEx"); return 15; }
VGPU_EXPORT int cublasDrot_v2() { vgpu_report_unimplemented("cublasDrot_v2"); return 15; }
VGPU_EXPORT int cublasDrotg_v2() { vgpu_report_unimplemented("cublasDrotg_v2"); return 15; }
VGPU_EXPORT int cublasDrotm_v2() { vgpu_report_unimplemented("cublasDrotm_v2"); return 15; }
VGPU_EXPORT int cublasDrotmg_v2() { vgpu_report_unimplemented("cublasDrotmg_v2"); return 15; }
VGPU_EXPORT int cublasDscal_v2() { vgpu_report_unimplemented("cublasDscal_v2"); return 15; }
VGPU_EXPORT int cublasDswap_v2() { vgpu_report_unimplemented("cublasDswap_v2"); return 15; }
VGPU_EXPORT int cublasDsymm_v2() { vgpu_report_unimplemented("cublasDsymm_v2"); return 15; }
VGPU_EXPORT int cublasDsymv_v2() { vgpu_report_unimplemented("cublasDsymv_v2"); return 15; }
VGPU_EXPORT int cublasDsyr2_v2() { vgpu_report_unimplemented("cublasDsyr2_v2"); return 15; }
VGPU_EXPORT int cublasDsyr2k_v2() { vgpu_report_unimplemented("cublasDsyr2k_v2"); return 15; }
VGPU_EXPORT int cublasDsyr_v2() { vgpu_report_unimplemented("cublasDsyr_v2"); return 15; }
VGPU_EXPORT int cublasDsyrk_v2() { vgpu_report_unimplemented("cublasDsyrk_v2"); return 15; }
VGPU_EXPORT int cublasDtrmm_v2() { vgpu_report_unimplemented("cublasDtrmm_v2"); return 15; }
VGPU_EXPORT int cublasDtrmv_v2() { vgpu_report_unimplemented("cublasDtrmv_v2"); return 15; }
VGPU_EXPORT int cublasDtrsmBatched() { vgpu_report_unimplemented("cublasDtrsmBatched"); return 15; }
VGPU_EXPORT int cublasDtrsm_v2() { vgpu_report_unimplemented("cublasDtrsm_v2"); return 15; }
VGPU_EXPORT int cublasDtrsv_v2() { vgpu_report_unimplemented("cublasDtrsv_v2"); return 15; }
VGPU_EXPORT int cublasDzasum_v2() { vgpu_report_unimplemented("cublasDzasum_v2"); return 15; }
VGPU_EXPORT int cublasDznrm2_v2() { vgpu_report_unimplemented("cublasDznrm2_v2"); return 15; }
VGPU_EXPORT int cublasGetMatrixAsync() { vgpu_report_unimplemented("cublasGetMatrixAsync"); return 15; }
VGPU_EXPORT int cublasGetVectorAsync() { vgpu_report_unimplemented("cublasGetVectorAsync"); return 15; }
VGPU_EXPORT int cublasIcamax_v2() { vgpu_report_unimplemented("cublasIcamax_v2"); return 15; }
VGPU_EXPORT int cublasIcamin_v2() { vgpu_report_unimplemented("cublasIcamin_v2"); return 15; }
VGPU_EXPORT int cublasIdamax_v2() { vgpu_report_unimplemented("cublasIdamax_v2"); return 15; }
VGPU_EXPORT int cublasIdamin_v2() { vgpu_report_unimplemented("cublasIdamin_v2"); return 15; }
VGPU_EXPORT int cublasIsamax_v2() { vgpu_report_unimplemented("cublasIsamax_v2"); return 15; }
VGPU_EXPORT int cublasIsamin_v2() { vgpu_report_unimplemented("cublasIsamin_v2"); return 15; }
VGPU_EXPORT int cublasIzamax_v2() { vgpu_report_unimplemented("cublasIzamax_v2"); return 15; }
VGPU_EXPORT int cublasIzamin_v2() { vgpu_report_unimplemented("cublasIzamin_v2"); return 15; }
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
VGPU_EXPORT int cublasSasum_v2() { vgpu_report_unimplemented("cublasSasum_v2"); return 15; }
VGPU_EXPORT int cublasScasum_v2() { vgpu_report_unimplemented("cublasScasum_v2"); return 15; }
VGPU_EXPORT int cublasScnrm2_v2() { vgpu_report_unimplemented("cublasScnrm2_v2"); return 15; }
VGPU_EXPORT int cublasScopy_v2() { vgpu_report_unimplemented("cublasScopy_v2"); return 15; }
VGPU_EXPORT int cublasSetMatrixAsync() { vgpu_report_unimplemented("cublasSetMatrixAsync"); return 15; }
VGPU_EXPORT int cublasSetVectorAsync() { vgpu_report_unimplemented("cublasSetVectorAsync"); return 15; }
VGPU_EXPORT int cublasSgelsBatched() { vgpu_report_unimplemented("cublasSgelsBatched"); return 15; }
VGPU_EXPORT int cublasSgemmEx() { vgpu_report_unimplemented("cublasSgemmEx"); return 15; }
VGPU_EXPORT int cublasSgeqrfBatched() { vgpu_report_unimplemented("cublasSgeqrfBatched"); return 15; }
VGPU_EXPORT int cublasSger_v2() { vgpu_report_unimplemented("cublasSger_v2"); return 15; }
VGPU_EXPORT int cublasSgetrfBatched() { vgpu_report_unimplemented("cublasSgetrfBatched"); return 15; }
VGPU_EXPORT int cublasSgetrsBatched() { vgpu_report_unimplemented("cublasSgetrsBatched"); return 15; }
VGPU_EXPORT int cublasSrot_v2() { vgpu_report_unimplemented("cublasSrot_v2"); return 15; }
VGPU_EXPORT int cublasSrotg_v2() { vgpu_report_unimplemented("cublasSrotg_v2"); return 15; }
VGPU_EXPORT int cublasSrotm_v2() { vgpu_report_unimplemented("cublasSrotm_v2"); return 15; }
VGPU_EXPORT int cublasSrotmg_v2() { vgpu_report_unimplemented("cublasSrotmg_v2"); return 15; }
VGPU_EXPORT int cublasSswap_v2() { vgpu_report_unimplemented("cublasSswap_v2"); return 15; }
VGPU_EXPORT int cublasSsymm_v2() { vgpu_report_unimplemented("cublasSsymm_v2"); return 15; }
VGPU_EXPORT int cublasSsymv_v2() { vgpu_report_unimplemented("cublasSsymv_v2"); return 15; }
VGPU_EXPORT int cublasSsyr2_v2() { vgpu_report_unimplemented("cublasSsyr2_v2"); return 15; }
VGPU_EXPORT int cublasSsyr2k_v2() { vgpu_report_unimplemented("cublasSsyr2k_v2"); return 15; }
VGPU_EXPORT int cublasSsyr_v2() { vgpu_report_unimplemented("cublasSsyr_v2"); return 15; }
VGPU_EXPORT int cublasSsyrk_v2() { vgpu_report_unimplemented("cublasSsyrk_v2"); return 15; }
VGPU_EXPORT int cublasStrmm_v2() { vgpu_report_unimplemented("cublasStrmm_v2"); return 15; }
VGPU_EXPORT int cublasStrmv_v2() { vgpu_report_unimplemented("cublasStrmv_v2"); return 15; }
VGPU_EXPORT int cublasStrsmBatched() { vgpu_report_unimplemented("cublasStrsmBatched"); return 15; }
VGPU_EXPORT int cublasStrsm_v2() { vgpu_report_unimplemented("cublasStrsm_v2"); return 15; }
VGPU_EXPORT int cublasStrsv_v2() { vgpu_report_unimplemented("cublasStrsv_v2"); return 15; }
VGPU_EXPORT int cublasZaxpy_v2() { vgpu_report_unimplemented("cublasZaxpy_v2"); return 15; }
VGPU_EXPORT int cublasZcopy_v2() { vgpu_report_unimplemented("cublasZcopy_v2"); return 15; }
VGPU_EXPORT int cublasZdotc_v2() { vgpu_report_unimplemented("cublasZdotc_v2"); return 15; }
VGPU_EXPORT int cublasZdotu_v2() { vgpu_report_unimplemented("cublasZdotu_v2"); return 15; }
VGPU_EXPORT int cublasZdrot_v2() { vgpu_report_unimplemented("cublasZdrot_v2"); return 15; }
VGPU_EXPORT int cublasZdscal_v2() { vgpu_report_unimplemented("cublasZdscal_v2"); return 15; }
VGPU_EXPORT int cublasZgelsBatched() { vgpu_report_unimplemented("cublasZgelsBatched"); return 15; }
VGPU_EXPORT int cublasZgemmBatched() { vgpu_report_unimplemented("cublasZgemmBatched"); return 15; }
VGPU_EXPORT int cublasZgemmStridedBatched() { vgpu_report_unimplemented("cublasZgemmStridedBatched"); return 15; }
VGPU_EXPORT int cublasZgemm_v2() { vgpu_report_unimplemented("cublasZgemm_v2"); return 15; }
VGPU_EXPORT int cublasZgemv_v2() { vgpu_report_unimplemented("cublasZgemv_v2"); return 15; }
VGPU_EXPORT int cublasZgeqrfBatched() { vgpu_report_unimplemented("cublasZgeqrfBatched"); return 15; }
VGPU_EXPORT int cublasZgerc_v2() { vgpu_report_unimplemented("cublasZgerc_v2"); return 15; }
VGPU_EXPORT int cublasZgeru_v2() { vgpu_report_unimplemented("cublasZgeru_v2"); return 15; }
VGPU_EXPORT int cublasZgetrfBatched() { vgpu_report_unimplemented("cublasZgetrfBatched"); return 15; }
VGPU_EXPORT int cublasZgetrsBatched() { vgpu_report_unimplemented("cublasZgetrsBatched"); return 15; }
VGPU_EXPORT int cublasZhemm_v2() { vgpu_report_unimplemented("cublasZhemm_v2"); return 15; }
VGPU_EXPORT int cublasZhemv_v2() { vgpu_report_unimplemented("cublasZhemv_v2"); return 15; }
VGPU_EXPORT int cublasZher2_v2() { vgpu_report_unimplemented("cublasZher2_v2"); return 15; }
VGPU_EXPORT int cublasZher2k_v2() { vgpu_report_unimplemented("cublasZher2k_v2"); return 15; }
VGPU_EXPORT int cublasZher_v2() { vgpu_report_unimplemented("cublasZher_v2"); return 15; }
VGPU_EXPORT int cublasZherk_v2() { vgpu_report_unimplemented("cublasZherk_v2"); return 15; }
VGPU_EXPORT int cublasZrot_v2() { vgpu_report_unimplemented("cublasZrot_v2"); return 15; }
VGPU_EXPORT int cublasZrotg_v2() { vgpu_report_unimplemented("cublasZrotg_v2"); return 15; }
VGPU_EXPORT int cublasZscal_v2() { vgpu_report_unimplemented("cublasZscal_v2"); return 15; }
VGPU_EXPORT int cublasZswap_v2() { vgpu_report_unimplemented("cublasZswap_v2"); return 15; }
VGPU_EXPORT int cublasZsymm_v2() { vgpu_report_unimplemented("cublasZsymm_v2"); return 15; }
VGPU_EXPORT int cublasZsymv_v2() { vgpu_report_unimplemented("cublasZsymv_v2"); return 15; }
VGPU_EXPORT int cublasZsyr2_v2() { vgpu_report_unimplemented("cublasZsyr2_v2"); return 15; }
VGPU_EXPORT int cublasZsyr2k_v2() { vgpu_report_unimplemented("cublasZsyr2k_v2"); return 15; }
VGPU_EXPORT int cublasZsyr_v2() { vgpu_report_unimplemented("cublasZsyr_v2"); return 15; }
VGPU_EXPORT int cublasZsyrk_v2() { vgpu_report_unimplemented("cublasZsyrk_v2"); return 15; }
VGPU_EXPORT int cublasZtrmm_v2() { vgpu_report_unimplemented("cublasZtrmm_v2"); return 15; }
VGPU_EXPORT int cublasZtrmv_v2() { vgpu_report_unimplemented("cublasZtrmv_v2"); return 15; }
VGPU_EXPORT int cublasZtrsmBatched() { vgpu_report_unimplemented("cublasZtrsmBatched"); return 15; }
VGPU_EXPORT int cublasZtrsm_v2() { vgpu_report_unimplemented("cublasZtrsm_v2"); return 15; }
VGPU_EXPORT int cublasZtrsv_v2() { vgpu_report_unimplemented("cublasZtrsv_v2"); return 15; }
VGPU_EXPORT int cublasGemmBatchedEx() { vgpu_report_unimplemented("cublasGemmBatchedEx"); return 15; }
