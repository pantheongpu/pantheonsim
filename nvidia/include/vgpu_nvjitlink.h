/*
 * vgpu_nvjitlink.h -- VirtualGPU's declarations of the nvJitLink API.
 *
 * Written from NVIDIA's publicly documented nvJitLink API
 * (docs.nvidia.com/cuda/nvjitlink) so that VirtualGPU's libnvJitLink can be
 * built on a machine with no CUDA toolkit. It contains no NVIDIA code. The
 * numeric values of the result codes and input types follow the documented
 * ABI, so a program compiled with NVIDIA's nvJitLink.h calls this library the
 * same way it calls NVIDIA's.
 *
 * NVIDIA's header turns every call but nvJitLinkVersion into an inline
 * wrapper around a versioned entry point (__nvJitLinkCreate_12_0 under CUDA
 * 12, __nvJitLinkCreate_13_0 under CUDA 13), which is what a compiled program
 * imports. The library defines both families and the plain names; this
 * header declares the plain names, which is all a caller of it needs.
 */
#ifndef VGPU_NVJITLINK_H_
#define VGPU_NVJITLINK_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  NVJITLINK_SUCCESS = 0,
  NVJITLINK_ERROR_UNRECOGNIZED_OPTION = 1,
  NVJITLINK_ERROR_MISSING_ARCH = 2,
  NVJITLINK_ERROR_INVALID_INPUT = 3,
  NVJITLINK_ERROR_PTX_COMPILE = 4,
  NVJITLINK_ERROR_NVVM_COMPILE = 5,
  NVJITLINK_ERROR_INTERNAL = 6,
  NVJITLINK_ERROR_THREADPOOL = 7,
  NVJITLINK_ERROR_UNRECOGNIZED_INPUT = 8,
  NVJITLINK_ERROR_FINALIZE = 9,
  NVJITLINK_ERROR_NULL_INPUT = 10,
  NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS = 11,
  NVJITLINK_ERROR_INCORRECT_INPUT_TYPE = 12,
  NVJITLINK_ERROR_ARCH_MISMATCH = 13,
  NVJITLINK_ERROR_OUTDATED_LIBRARY = 14,
  NVJITLINK_ERROR_MISSING_FATBIN = 15,
  NVJITLINK_ERROR_UNRECOGNIZED_ARCH = 16,
  NVJITLINK_ERROR_UNSUPPORTED_ARCH = 17,
  NVJITLINK_ERROR_LTO_NOT_ENABLED = 18
} nvJitLinkResult;

typedef enum {
  NVJITLINK_INPUT_NONE = 0,
  NVJITLINK_INPUT_CUBIN = 1,
  NVJITLINK_INPUT_PTX = 2,
  NVJITLINK_INPUT_LTOIR = 3,
  NVJITLINK_INPUT_FATBIN = 4,
  NVJITLINK_INPUT_OBJECT = 5,
  NVJITLINK_INPUT_LIBRARY = 6,
  NVJITLINK_INPUT_INDEX = 7,
  NVJITLINK_INPUT_ANY = 10
} nvJitLinkInputType;

typedef struct nvJitLink* nvJitLinkHandle;

nvJitLinkResult nvJitLinkCreate(nvJitLinkHandle* handle, uint32_t numOptions, const char** options);
nvJitLinkResult nvJitLinkDestroy(nvJitLinkHandle* handle);
nvJitLinkResult nvJitLinkAddData(nvJitLinkHandle handle, nvJitLinkInputType inputType,
                                 const void* data, size_t size, const char* name);
nvJitLinkResult nvJitLinkAddFile(nvJitLinkHandle handle, nvJitLinkInputType inputType,
                                 const char* fileName);
nvJitLinkResult nvJitLinkComplete(nvJitLinkHandle handle);
nvJitLinkResult nvJitLinkGetLinkedCubinSize(nvJitLinkHandle handle, size_t* size);
nvJitLinkResult nvJitLinkGetLinkedCubin(nvJitLinkHandle handle, void* cubin);
nvJitLinkResult nvJitLinkGetLinkedPtxSize(nvJitLinkHandle handle, size_t* size);
nvJitLinkResult nvJitLinkGetLinkedPtx(nvJitLinkHandle handle, char* ptx);
nvJitLinkResult nvJitLinkGetErrorLogSize(nvJitLinkHandle handle, size_t* size);
nvJitLinkResult nvJitLinkGetErrorLog(nvJitLinkHandle handle, char* log);
nvJitLinkResult nvJitLinkGetInfoLogSize(nvJitLinkHandle handle, size_t* size);
nvJitLinkResult nvJitLinkGetInfoLog(nvJitLinkHandle handle, char* log);
nvJitLinkResult nvJitLinkVersion(unsigned int* major, unsigned int* minor);

#ifdef __cplusplus
}
#endif

#endif /* VGPU_NVJITLINK_H_ */
