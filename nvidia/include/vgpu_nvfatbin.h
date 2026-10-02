/*
 * vgpu_nvfatbin.h -- VirtualGPU's declarations of the nvFatbin API.
 *
 * Written from NVIDIA's publicly documented nvFatbin API
 * (docs.nvidia.com/cuda/nvfatbin) so that VirtualGPU's libnvfatbin can be
 * built, and called, on a machine whose toolkit predates nvFatbin (CUDA 12.0
 * has none). It contains no NVIDIA code. The numeric values of the result
 * codes follow the documented ABI, so a program compiled with NVIDIA's
 * nvFatbin.h and one compiled with this header call the same library the
 * same way.
 */
#ifndef VGPU_NVFATBIN_H_
#define VGPU_NVFATBIN_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  NVFATBIN_SUCCESS = 0,
  NVFATBIN_ERROR_INTERNAL = 1,
  NVFATBIN_ERROR_ELF_ARCH_MISMATCH = 2,
  NVFATBIN_ERROR_ELF_SIZE_MISMATCH = 3,
  NVFATBIN_ERROR_MISSING_PTX_VERSION = 4,
  NVFATBIN_ERROR_NULL_POINTER = 5,
  NVFATBIN_ERROR_COMPRESSION_FAILED = 6,
  NVFATBIN_ERROR_COMPRESSED_SIZE_EXCEEDED = 7,
  NVFATBIN_ERROR_UNRECOGNIZED_OPTION = 8,
  NVFATBIN_ERROR_INVALID_ARCH = 9,
  NVFATBIN_ERROR_INVALID_NVVM = 10,
  NVFATBIN_ERROR_EMPTY_INPUT = 11,
  NVFATBIN_ERROR_MISSING_PTX_ARCH = 12,
  NVFATBIN_ERROR_PTX_ARCH_MISMATCH = 13,
  NVFATBIN_ERROR_MISSING_FATBIN = 14,
  NVFATBIN_ERROR_INVALID_INDEX = 15,
  NVFATBIN_ERROR_IDENTIFIER_REUSE = 16,
  NVFATBIN_ERROR_INTERNAL_PTX_OPTION = 17
} nvFatbinResult;

typedef struct _nvFatbinHandle* nvFatbinHandle;

const char* nvFatbinGetErrorString(nvFatbinResult result);
nvFatbinResult nvFatbinCreate(nvFatbinHandle* handle_indirect, const char** options,
                              size_t optionsCount);
nvFatbinResult nvFatbinDestroy(nvFatbinHandle* handle_indirect);
nvFatbinResult nvFatbinAddPTX(nvFatbinHandle handle, const char* code, size_t size,
                              const char* arch, const char* identifier, const char* optionsCmdLine);
nvFatbinResult nvFatbinAddCubin(nvFatbinHandle handle, const void* code, size_t size,
                                const char* arch, const char* identifier);
nvFatbinResult nvFatbinAddLTOIR(nvFatbinHandle handle, const void* code, size_t size,
                                const char* arch, const char* identifier,
                                const char* optionsCmdLine);
nvFatbinResult nvFatbinAddIndex(nvFatbinHandle handle, const void* code, size_t size,
                                const char* identifier);
nvFatbinResult nvFatbinAddReloc(nvFatbinHandle handle, const void* code, size_t size);
nvFatbinResult nvFatbinSize(nvFatbinHandle handle, size_t* size);
nvFatbinResult nvFatbinGet(nvFatbinHandle handle, void* buffer);
nvFatbinResult nvFatbinVersion(unsigned int* major, unsigned int* minor);

#ifdef __cplusplus
}
#endif

#endif /* VGPU_NVFATBIN_H_ */
