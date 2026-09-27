// GENERATED -- entry points cudnn exports that VirtualGPU does not implement.
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
               "[vgpu] %s is not implemented by VirtualGPU; returning CUDNN_STATUS_NOT_SUPPORTED.\n"
               "       The call site will see a failure rather than a wrong answer.\n", fn);
}
}  // namespace

VGPU_EXPORT int cudnnCTCLoss() { vgpu_report_unimplemented("cudnnCTCLoss"); return 9; }
VGPU_EXPORT int cudnnCTCLoss_v8() { vgpu_report_unimplemented("cudnnCTCLoss_v8"); return 9; }
VGPU_EXPORT int cudnnConvolutionBackwardData() { vgpu_report_unimplemented("cudnnConvolutionBackwardData"); return 9; }
VGPU_EXPORT int cudnnConvolutionBackwardFilter() { vgpu_report_unimplemented("cudnnConvolutionBackwardFilter"); return 9; }
VGPU_EXPORT int cudnnConvolutionBiasActivationForward() { vgpu_report_unimplemented("cudnnConvolutionBiasActivationForward"); return 9; }
VGPU_EXPORT int cudnnCreateCTCLossDescriptor() { vgpu_report_unimplemented("cudnnCreateCTCLossDescriptor"); return 9; }
VGPU_EXPORT int cudnnCreateSpatialTransformerDescriptor() { vgpu_report_unimplemented("cudnnCreateSpatialTransformerDescriptor"); return 9; }
VGPU_EXPORT int cudnnDestroyCTCLossDescriptor() { vgpu_report_unimplemented("cudnnDestroyCTCLossDescriptor"); return 9; }
VGPU_EXPORT int cudnnDestroySpatialTransformerDescriptor() { vgpu_report_unimplemented("cudnnDestroySpatialTransformerDescriptor"); return 9; }
VGPU_EXPORT int cudnnFindConvolutionBackwardDataAlgorithmEx() { vgpu_report_unimplemented("cudnnFindConvolutionBackwardDataAlgorithmEx"); return 9; }
VGPU_EXPORT int cudnnFindConvolutionBackwardFilterAlgorithmEx() { vgpu_report_unimplemented("cudnnFindConvolutionBackwardFilterAlgorithmEx"); return 9; }
VGPU_EXPORT int cudnnFindConvolutionForwardAlgorithmEx() { vgpu_report_unimplemented("cudnnFindConvolutionForwardAlgorithmEx"); return 9; }
VGPU_EXPORT int cudnnGetCTCLossWorkspaceSize() { vgpu_report_unimplemented("cudnnGetCTCLossWorkspaceSize"); return 9; }
VGPU_EXPORT int cudnnGetCTCLossWorkspaceSize_v8() { vgpu_report_unimplemented("cudnnGetCTCLossWorkspaceSize_v8"); return 9; }
VGPU_EXPORT int cudnnGetConvolutionBackwardDataAlgorithm_v7() { vgpu_report_unimplemented("cudnnGetConvolutionBackwardDataAlgorithm_v7"); return 9; }
VGPU_EXPORT int cudnnGetConvolutionBackwardDataWorkspaceSize() { vgpu_report_unimplemented("cudnnGetConvolutionBackwardDataWorkspaceSize"); return 9; }
VGPU_EXPORT int cudnnGetConvolutionBackwardFilterAlgorithm_v7() { vgpu_report_unimplemented("cudnnGetConvolutionBackwardFilterAlgorithm_v7"); return 9; }
VGPU_EXPORT int cudnnGetConvolutionBackwardFilterWorkspaceSize() { vgpu_report_unimplemented("cudnnGetConvolutionBackwardFilterWorkspaceSize"); return 9; }
VGPU_EXPORT int cudnnGetFilterNdDescriptor() { vgpu_report_unimplemented("cudnnGetFilterNdDescriptor"); return 9; }
VGPU_EXPORT int cudnnGetLastErrorString() { vgpu_report_unimplemented("cudnnGetLastErrorString"); return 9; }
VGPU_EXPORT int cudnnSetCTCLossDescriptorEx() { vgpu_report_unimplemented("cudnnSetCTCLossDescriptorEx"); return 9; }
VGPU_EXPORT int cudnnSetCTCLossDescriptor_v9() { vgpu_report_unimplemented("cudnnSetCTCLossDescriptor_v9"); return 9; }
VGPU_EXPORT int cudnnSetConvolutionNdDescriptor() { vgpu_report_unimplemented("cudnnSetConvolutionNdDescriptor"); return 9; }
VGPU_EXPORT int cudnnSetFilterNdDescriptor() { vgpu_report_unimplemented("cudnnSetFilterNdDescriptor"); return 9; }
VGPU_EXPORT int cudnnSetSpatialTransformerNdDescriptor() { vgpu_report_unimplemented("cudnnSetSpatialTransformerNdDescriptor"); return 9; }
VGPU_EXPORT int cudnnSpatialTfGridGeneratorBackward() { vgpu_report_unimplemented("cudnnSpatialTfGridGeneratorBackward"); return 9; }
VGPU_EXPORT int cudnnSpatialTfGridGeneratorForward() { vgpu_report_unimplemented("cudnnSpatialTfGridGeneratorForward"); return 9; }
VGPU_EXPORT int cudnnSpatialTfSamplerBackward() { vgpu_report_unimplemented("cudnnSpatialTfSamplerBackward"); return 9; }
VGPU_EXPORT int cudnnSpatialTfSamplerForward() { vgpu_report_unimplemented("cudnnSpatialTfSamplerForward"); return 9; }
