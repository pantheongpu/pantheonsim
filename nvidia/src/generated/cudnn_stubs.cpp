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

VGPU_EXPORT int cudnnBackendCreateDescriptor() { vgpu_report_unimplemented("cudnnBackendCreateDescriptor"); return 9; }
VGPU_EXPORT int cudnnBackendDestroyDescriptor() { vgpu_report_unimplemented("cudnnBackendDestroyDescriptor"); return 9; }
VGPU_EXPORT int cudnnBackendExecute() { vgpu_report_unimplemented("cudnnBackendExecute"); return 9; }
VGPU_EXPORT int cudnnBackendFinalize() { vgpu_report_unimplemented("cudnnBackendFinalize"); return 9; }
VGPU_EXPORT int cudnnBackendGetAttribute() { vgpu_report_unimplemented("cudnnBackendGetAttribute"); return 9; }
VGPU_EXPORT int cudnnBackendSetAttribute() { vgpu_report_unimplemented("cudnnBackendSetAttribute"); return 9; }
VGPU_EXPORT int cudnnBatchNormalizationBackwardEx() { vgpu_report_unimplemented("cudnnBatchNormalizationBackwardEx"); return 9; }
VGPU_EXPORT int cudnnBatchNormalizationForwardTrainingEx() { vgpu_report_unimplemented("cudnnBatchNormalizationForwardTrainingEx"); return 9; }
VGPU_EXPORT int cudnnCTCLoss() { vgpu_report_unimplemented("cudnnCTCLoss"); return 9; }
VGPU_EXPORT int cudnnCTCLoss_v8() { vgpu_report_unimplemented("cudnnCTCLoss_v8"); return 9; }
VGPU_EXPORT int cudnnConvolutionBackwardData() { vgpu_report_unimplemented("cudnnConvolutionBackwardData"); return 9; }
VGPU_EXPORT int cudnnConvolutionBackwardFilter() { vgpu_report_unimplemented("cudnnConvolutionBackwardFilter"); return 9; }
VGPU_EXPORT int cudnnConvolutionBiasActivationForward() { vgpu_report_unimplemented("cudnnConvolutionBiasActivationForward"); return 9; }
VGPU_EXPORT int cudnnCreateCTCLossDescriptor() { vgpu_report_unimplemented("cudnnCreateCTCLossDescriptor"); return 9; }
VGPU_EXPORT int cudnnCreateDropoutDescriptor() { vgpu_report_unimplemented("cudnnCreateDropoutDescriptor"); return 9; }
VGPU_EXPORT int cudnnCreateRNNDataDescriptor() { vgpu_report_unimplemented("cudnnCreateRNNDataDescriptor"); return 9; }
VGPU_EXPORT int cudnnCreateRNNDescriptor() { vgpu_report_unimplemented("cudnnCreateRNNDescriptor"); return 9; }
VGPU_EXPORT int cudnnCreateSpatialTransformerDescriptor() { vgpu_report_unimplemented("cudnnCreateSpatialTransformerDescriptor"); return 9; }
VGPU_EXPORT int cudnnDestroyCTCLossDescriptor() { vgpu_report_unimplemented("cudnnDestroyCTCLossDescriptor"); return 9; }
VGPU_EXPORT int cudnnDestroyDropoutDescriptor() { vgpu_report_unimplemented("cudnnDestroyDropoutDescriptor"); return 9; }
VGPU_EXPORT int cudnnDestroyRNNDataDescriptor() { vgpu_report_unimplemented("cudnnDestroyRNNDataDescriptor"); return 9; }
VGPU_EXPORT int cudnnDestroyRNNDescriptor() { vgpu_report_unimplemented("cudnnDestroyRNNDescriptor"); return 9; }
VGPU_EXPORT int cudnnDestroySpatialTransformerDescriptor() { vgpu_report_unimplemented("cudnnDestroySpatialTransformerDescriptor"); return 9; }
VGPU_EXPORT int cudnnDropoutGetStatesSize() { vgpu_report_unimplemented("cudnnDropoutGetStatesSize"); return 9; }
VGPU_EXPORT int cudnnFindConvolutionBackwardDataAlgorithmEx() { vgpu_report_unimplemented("cudnnFindConvolutionBackwardDataAlgorithmEx"); return 9; }
VGPU_EXPORT int cudnnFindConvolutionBackwardFilterAlgorithmEx() { vgpu_report_unimplemented("cudnnFindConvolutionBackwardFilterAlgorithmEx"); return 9; }
VGPU_EXPORT int cudnnFindConvolutionForwardAlgorithmEx() { vgpu_report_unimplemented("cudnnFindConvolutionForwardAlgorithmEx"); return 9; }
VGPU_EXPORT int cudnnGetBatchNormalizationBackwardExWorkspaceSize() { vgpu_report_unimplemented("cudnnGetBatchNormalizationBackwardExWorkspaceSize"); return 9; }
VGPU_EXPORT int cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize() { vgpu_report_unimplemented("cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize"); return 9; }
VGPU_EXPORT int cudnnGetBatchNormalizationTrainingExReserveSpaceSize() { vgpu_report_unimplemented("cudnnGetBatchNormalizationTrainingExReserveSpaceSize"); return 9; }
VGPU_EXPORT int cudnnGetCTCLossWorkspaceSize() { vgpu_report_unimplemented("cudnnGetCTCLossWorkspaceSize"); return 9; }
VGPU_EXPORT int cudnnGetCTCLossWorkspaceSize_v8() { vgpu_report_unimplemented("cudnnGetCTCLossWorkspaceSize_v8"); return 9; }
VGPU_EXPORT int cudnnGetConvolutionBackwardDataAlgorithm_v7() { vgpu_report_unimplemented("cudnnGetConvolutionBackwardDataAlgorithm_v7"); return 9; }
VGPU_EXPORT int cudnnGetConvolutionBackwardDataWorkspaceSize() { vgpu_report_unimplemented("cudnnGetConvolutionBackwardDataWorkspaceSize"); return 9; }
VGPU_EXPORT int cudnnGetConvolutionBackwardFilterAlgorithm_v7() { vgpu_report_unimplemented("cudnnGetConvolutionBackwardFilterAlgorithm_v7"); return 9; }
VGPU_EXPORT int cudnnGetConvolutionBackwardFilterWorkspaceSize() { vgpu_report_unimplemented("cudnnGetConvolutionBackwardFilterWorkspaceSize"); return 9; }
VGPU_EXPORT int cudnnGetFilterNdDescriptor() { vgpu_report_unimplemented("cudnnGetFilterNdDescriptor"); return 9; }
VGPU_EXPORT int cudnnGetLastErrorString() { vgpu_report_unimplemented("cudnnGetLastErrorString"); return 9; }
VGPU_EXPORT int cudnnGetRNNTempSpaceSizes() { vgpu_report_unimplemented("cudnnGetRNNTempSpaceSizes"); return 9; }
VGPU_EXPORT int cudnnGetRNNWeightParams() { vgpu_report_unimplemented("cudnnGetRNNWeightParams"); return 9; }
VGPU_EXPORT int cudnnGetRNNWeightSpaceSize() { vgpu_report_unimplemented("cudnnGetRNNWeightSpaceSize"); return 9; }
VGPU_EXPORT int cudnnGetTensorNdDescriptor() { vgpu_report_unimplemented("cudnnGetTensorNdDescriptor"); return 9; }
VGPU_EXPORT int cudnnRNNBackwardData_v8() { vgpu_report_unimplemented("cudnnRNNBackwardData_v8"); return 9; }
VGPU_EXPORT int cudnnRNNBackwardWeights_v8() { vgpu_report_unimplemented("cudnnRNNBackwardWeights_v8"); return 9; }
VGPU_EXPORT int cudnnRNNForward() { vgpu_report_unimplemented("cudnnRNNForward"); return 9; }
VGPU_EXPORT int cudnnRestoreDropoutDescriptor() { vgpu_report_unimplemented("cudnnRestoreDropoutDescriptor"); return 9; }
VGPU_EXPORT int cudnnSetCTCLossDescriptorEx() { vgpu_report_unimplemented("cudnnSetCTCLossDescriptorEx"); return 9; }
VGPU_EXPORT int cudnnSetCTCLossDescriptor_v9() { vgpu_report_unimplemented("cudnnSetCTCLossDescriptor_v9"); return 9; }
VGPU_EXPORT int cudnnSetConvolutionNdDescriptor() { vgpu_report_unimplemented("cudnnSetConvolutionNdDescriptor"); return 9; }
VGPU_EXPORT int cudnnSetDropoutDescriptor() { vgpu_report_unimplemented("cudnnSetDropoutDescriptor"); return 9; }
VGPU_EXPORT int cudnnSetFilterNdDescriptor() { vgpu_report_unimplemented("cudnnSetFilterNdDescriptor"); return 9; }
VGPU_EXPORT int cudnnSetRNNDataDescriptor() { vgpu_report_unimplemented("cudnnSetRNNDataDescriptor"); return 9; }
VGPU_EXPORT int cudnnSetRNNDescriptor_v8() { vgpu_report_unimplemented("cudnnSetRNNDescriptor_v8"); return 9; }
VGPU_EXPORT int cudnnSetSpatialTransformerNdDescriptor() { vgpu_report_unimplemented("cudnnSetSpatialTransformerNdDescriptor"); return 9; }
VGPU_EXPORT int cudnnSetTensorNdDescriptor() { vgpu_report_unimplemented("cudnnSetTensorNdDescriptor"); return 9; }
VGPU_EXPORT int cudnnSpatialTfGridGeneratorBackward() { vgpu_report_unimplemented("cudnnSpatialTfGridGeneratorBackward"); return 9; }
VGPU_EXPORT int cudnnSpatialTfGridGeneratorForward() { vgpu_report_unimplemented("cudnnSpatialTfGridGeneratorForward"); return 9; }
VGPU_EXPORT int cudnnSpatialTfSamplerBackward() { vgpu_report_unimplemented("cudnnSpatialTfSamplerBackward"); return 9; }
VGPU_EXPORT int cudnnSpatialTfSamplerForward() { vgpu_report_unimplemented("cudnnSpatialTfSamplerForward"); return 9; }
