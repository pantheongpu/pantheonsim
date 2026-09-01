/*
 * vgpu_cuda.h — VirtualGPU's clean-room subset of the CUDA Driver API.
 *
 * This header is written from NVIDIA's *publicly documented* driver API
 * (docs.nvidia.com/cuda/cuda-driver-api) so that programs written against the
 * documented interface can link against VirtualGPU's libvgpucuda instead of a
 * real libcuda. It contains no NVIDIA code. Numeric values (error codes,
 * attribute ids) follow the documented ABI so binaries and headers agree.
 *
 * Only the subset VirtualGPU implements is declared. Calling anything with
 * semantics we do not implement returns CUDA_ERROR_NOT_SUPPORTED and prints a
 * precise diagnostic to stderr (set VGPU_QUIET=1 to silence).
 *
 * Environment:
 *   VGPU_GPU=<vendor/model>   virtual GPU profile (default: nvidia/h100)
 *   VGPU_DEVICE_COUNT=<n>     number of identical virtual devices (default: 1)
 *   VGPU_QUIET=1              suppress stderr diagnostics
 */
#ifndef VGPU_CUDA_H_
#define VGPU_CUDA_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cudaError_enum {
  CUDA_SUCCESS = 0,
  CUDA_ERROR_INVALID_VALUE = 1,
  CUDA_ERROR_OUT_OF_MEMORY = 2,
  CUDA_ERROR_NOT_INITIALIZED = 3,
  CUDA_ERROR_INVALID_DEVICE = 101,
  CUDA_ERROR_INVALID_CONTEXT = 201,
  CUDA_ERROR_INVALID_PTX = 218,
  CUDA_ERROR_NOT_FOUND = 500,
  CUDA_ERROR_ILLEGAL_ADDRESS = 700,
  CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
  CUDA_ERROR_LAUNCH_TIMEOUT = 702,
  CUDA_ERROR_NOT_SUPPORTED = 801,
  CUDA_ERROR_UNKNOWN = 999,
} CUresult;

typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;

typedef enum CUdevice_attribute_enum {
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
  CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
  CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
  CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
  CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
} CUdevice_attribute;

/* Initialization / errors */
CUresult cuInit(unsigned int flags);
CUresult cuDriverGetVersion(int* driverVersion);
CUresult cuGetErrorName(CUresult error, const char** pStr);
CUresult cuGetErrorString(CUresult error, const char** pStr);

/* Device discovery */
CUresult cuDeviceGetCount(int* count);
CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUresult cuDeviceGetName(char* name, int len, CUdevice dev);
CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev);
CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev);
CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev);
CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev);

/* Contexts (primary-context model; a global current-context stack) */
CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy(CUcontext ctx);
CUresult cuCtxDestroy_v2(CUcontext ctx);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuCtxGetCurrent(CUcontext* pctx);
CUresult cuCtxGetDevice(CUdevice* device);
CUresult cuCtxSynchronize(void);
CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev);
CUresult cuDevicePrimaryCtxRelease(CUdevice dev);
CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev);

/* Memory */
CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);
CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemFree_v2(CUdeviceptr dptr);
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemGetInfo(size_t* free, size_t* total);
CUresult cuMemGetInfo_v2(size_t* free, size_t* total);

/* Modules and kernels. cuModuleLoadData expects NUL-terminated PTX text
 * (cubin/fatbin images are not supported yet — documented limitation). */
CUresult cuModuleLoadData(CUmodule* module, const void* image);
CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions,
                            void* options, void** optionValues);
CUresult cuModuleUnload(CUmodule hmod);
CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                        unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
                        void** kernelParams, void** extra);

#ifdef __cplusplus
}
#endif

#endif /* VGPU_CUDA_H_ */
