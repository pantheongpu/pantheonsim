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

VGPU_EXPORT int cublasCgemm3mEx() { vgpu_report_unimplemented("cublasCgemm3mEx"); return 15; }
VGPU_EXPORT int cublasCgemm3mEx_64() { vgpu_report_unimplemented("cublasCgemm3mEx_64"); return 15; }
VGPU_EXPORT int cublasCgemmEx() { vgpu_report_unimplemented("cublasCgemmEx"); return 15; }
VGPU_EXPORT int cublasCgemmEx_64() { vgpu_report_unimplemented("cublasCgemmEx_64"); return 15; }
VGPU_EXPORT int cublasCgeqrfBatched() { vgpu_report_unimplemented("cublasCgeqrfBatched"); return 15; }
VGPU_EXPORT int cublasCher2_v2() { vgpu_report_unimplemented("cublasCher2_v2"); return 15; }
VGPU_EXPORT int cublasCher2k_v2() { vgpu_report_unimplemented("cublasCher2k_v2"); return 15; }
VGPU_EXPORT int cublasCher_v2() { vgpu_report_unimplemented("cublasCher_v2"); return 15; }
VGPU_EXPORT int cublasCherk3mEx() { vgpu_report_unimplemented("cublasCherk3mEx"); return 15; }
VGPU_EXPORT int cublasCherk3mEx_64() { vgpu_report_unimplemented("cublasCherk3mEx_64"); return 15; }
VGPU_EXPORT int cublasCherkEx() { vgpu_report_unimplemented("cublasCherkEx"); return 15; }
VGPU_EXPORT int cublasCherkEx_64() { vgpu_report_unimplemented("cublasCherkEx_64"); return 15; }
VGPU_EXPORT int cublasCrot_v2() { vgpu_report_unimplemented("cublasCrot_v2"); return 15; }
VGPU_EXPORT int cublasCrotg_v2() { vgpu_report_unimplemented("cublasCrotg_v2"); return 15; }
VGPU_EXPORT int cublasCsrot_v2() { vgpu_report_unimplemented("cublasCsrot_v2"); return 15; }
VGPU_EXPORT int cublasCsyr2_v2() { vgpu_report_unimplemented("cublasCsyr2_v2"); return 15; }
VGPU_EXPORT int cublasCsyr2k_v2() { vgpu_report_unimplemented("cublasCsyr2k_v2"); return 15; }
VGPU_EXPORT int cublasCsyr_v2() { vgpu_report_unimplemented("cublasCsyr_v2"); return 15; }
VGPU_EXPORT int cublasCsyrk3mEx() { vgpu_report_unimplemented("cublasCsyrk3mEx"); return 15; }
VGPU_EXPORT int cublasCsyrk3mEx_64() { vgpu_report_unimplemented("cublasCsyrk3mEx_64"); return 15; }
VGPU_EXPORT int cublasCsyrkEx() { vgpu_report_unimplemented("cublasCsyrkEx"); return 15; }
VGPU_EXPORT int cublasCsyrkEx_64() { vgpu_report_unimplemented("cublasCsyrkEx_64"); return 15; }
VGPU_EXPORT int cublasDcopy_v2() { vgpu_report_unimplemented("cublasDcopy_v2"); return 15; }
VGPU_EXPORT int cublasDgemmBatched() { vgpu_report_unimplemented("cublasDgemmBatched"); return 15; }
VGPU_EXPORT int cublasDgemmStridedBatched() { vgpu_report_unimplemented("cublasDgemmStridedBatched"); return 15; }
VGPU_EXPORT int cublasDgeqrfBatched() { vgpu_report_unimplemented("cublasDgeqrfBatched"); return 15; }
VGPU_EXPORT int cublasDger_v2() { vgpu_report_unimplemented("cublasDger_v2"); return 15; }
VGPU_EXPORT int cublasDrot_v2() { vgpu_report_unimplemented("cublasDrot_v2"); return 15; }
VGPU_EXPORT int cublasDrotg_v2() { vgpu_report_unimplemented("cublasDrotg_v2"); return 15; }
VGPU_EXPORT int cublasDrotm_v2() { vgpu_report_unimplemented("cublasDrotm_v2"); return 15; }
VGPU_EXPORT int cublasDrotmg_v2() { vgpu_report_unimplemented("cublasDrotmg_v2"); return 15; }
VGPU_EXPORT int cublasDswap_v2() { vgpu_report_unimplemented("cublasDswap_v2"); return 15; }
VGPU_EXPORT int cublasDsymm_v2() { vgpu_report_unimplemented("cublasDsymm_v2"); return 15; }
VGPU_EXPORT int cublasDsymv_v2() { vgpu_report_unimplemented("cublasDsymv_v2"); return 15; }
VGPU_EXPORT int cublasDsyr2_v2() { vgpu_report_unimplemented("cublasDsyr2_v2"); return 15; }
VGPU_EXPORT int cublasDsyr2k_v2() { vgpu_report_unimplemented("cublasDsyr2k_v2"); return 15; }
VGPU_EXPORT int cublasDsyr_v2() { vgpu_report_unimplemented("cublasDsyr_v2"); return 15; }
VGPU_EXPORT int cublasDsyrk_v2() { vgpu_report_unimplemented("cublasDsyrk_v2"); return 15; }
VGPU_EXPORT int cublasDtrmm_v2() { vgpu_report_unimplemented("cublasDtrmm_v2"); return 15; }
VGPU_EXPORT int cublasDtrmv_v2() { vgpu_report_unimplemented("cublasDtrmv_v2"); return 15; }
VGPU_EXPORT int cublasDtrsv_v2() { vgpu_report_unimplemented("cublasDtrsv_v2"); return 15; }
VGPU_EXPORT int cublasDzasum_v2() { vgpu_report_unimplemented("cublasDzasum_v2"); return 15; }
VGPU_EXPORT int cublasDznrm2_v2() { vgpu_report_unimplemented("cublasDznrm2_v2"); return 15; }
VGPU_EXPORT int cublasGemmGroupedBatchedEx() { vgpu_report_unimplemented("cublasGemmGroupedBatchedEx"); return 15; }
VGPU_EXPORT int cublasGemmGroupedBatchedEx_64() { vgpu_report_unimplemented("cublasGemmGroupedBatchedEx_64"); return 15; }
VGPU_EXPORT int cublasIcamax_v2() { vgpu_report_unimplemented("cublasIcamax_v2"); return 15; }
VGPU_EXPORT int cublasIcamin_v2() { vgpu_report_unimplemented("cublasIcamin_v2"); return 15; }
VGPU_EXPORT int cublasIdamin_v2() { vgpu_report_unimplemented("cublasIdamin_v2"); return 15; }
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
VGPU_EXPORT int cublasRotEx() { vgpu_report_unimplemented("cublasRotEx"); return 15; }
VGPU_EXPORT int cublasRotEx_64() { vgpu_report_unimplemented("cublasRotEx_64"); return 15; }
VGPU_EXPORT int cublasRotgEx() { vgpu_report_unimplemented("cublasRotgEx"); return 15; }
VGPU_EXPORT int cublasRotmEx() { vgpu_report_unimplemented("cublasRotmEx"); return 15; }
VGPU_EXPORT int cublasRotmEx_64() { vgpu_report_unimplemented("cublasRotmEx_64"); return 15; }
VGPU_EXPORT int cublasRotmgEx() { vgpu_report_unimplemented("cublasRotmgEx"); return 15; }
VGPU_EXPORT int cublasScasum_v2() { vgpu_report_unimplemented("cublasScasum_v2"); return 15; }
VGPU_EXPORT int cublasScnrm2_v2() { vgpu_report_unimplemented("cublasScnrm2_v2"); return 15; }
VGPU_EXPORT int cublasScopy_v2() { vgpu_report_unimplemented("cublasScopy_v2"); return 15; }
VGPU_EXPORT int cublasSgemmEx() { vgpu_report_unimplemented("cublasSgemmEx"); return 15; }
VGPU_EXPORT int cublasSgeqrfBatched() { vgpu_report_unimplemented("cublasSgeqrfBatched"); return 15; }
VGPU_EXPORT int cublasSger_v2() { vgpu_report_unimplemented("cublasSger_v2"); return 15; }
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
VGPU_EXPORT int cublasStrsv_v2() { vgpu_report_unimplemented("cublasStrsv_v2"); return 15; }
VGPU_EXPORT int cublasZdrot_v2() { vgpu_report_unimplemented("cublasZdrot_v2"); return 15; }
VGPU_EXPORT int cublasZgeqrfBatched() { vgpu_report_unimplemented("cublasZgeqrfBatched"); return 15; }
VGPU_EXPORT int cublasZher2_v2() { vgpu_report_unimplemented("cublasZher2_v2"); return 15; }
VGPU_EXPORT int cublasZher2k_v2() { vgpu_report_unimplemented("cublasZher2k_v2"); return 15; }
VGPU_EXPORT int cublasZher_v2() { vgpu_report_unimplemented("cublasZher_v2"); return 15; }
VGPU_EXPORT int cublasZrot_v2() { vgpu_report_unimplemented("cublasZrot_v2"); return 15; }
VGPU_EXPORT int cublasZrotg_v2() { vgpu_report_unimplemented("cublasZrotg_v2"); return 15; }
VGPU_EXPORT int cublasZsyr2_v2() { vgpu_report_unimplemented("cublasZsyr2_v2"); return 15; }
VGPU_EXPORT int cublasZsyr2k_v2() { vgpu_report_unimplemented("cublasZsyr2k_v2"); return 15; }
VGPU_EXPORT int cublasZsyr_v2() { vgpu_report_unimplemented("cublasZsyr_v2"); return 15; }
