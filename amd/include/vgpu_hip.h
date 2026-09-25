/*
 * vgpu_hip.h — VirtualGPU's clean-room subset of the HIP runtime API.
 *
 * Written from AMD's *publicly documented* HIP API (rocm.docs.amd.com) so
 * that programs written against the documented interface can link against
 * VirtualGPU's libamdhip64 instead of a real one. It contains no AMD code.
 * Numeric values (error codes, attribute ids) follow the documented ABI so
 * binaries and headers agree.
 *
 * Only the subset VirtualGPU implements is declared: devices, memory, and the
 * module API, which loads a code object and launches the kernels in it.
 * Anything else returns hipErrorNotSupported with a precise diagnostic on
 * stderr (set VGPU_QUIET=1 to silence).
 *
 * Environment:
 *   VGPU_GPU=<vendor/model>   virtual GPU profile (default: amd/mi300x)
 *   VGPU_DEVICE_COUNT=<n>     number of identical virtual devices (default: 1)
 *   VGPU_QUIET=1              suppress stderr diagnostics
 */
#ifndef VGPU_HIP_H_
#define VGPU_HIP_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum hipError_t {
  hipSuccess = 0,
  hipErrorInvalidValue = 1,
  hipErrorOutOfMemory = 2,
  hipErrorNotInitialized = 3,
  hipErrorDeinitialized = 4,
  hipErrorInvalidConfiguration = 9,
  hipErrorInvalidSymbol = 13,
  hipErrorInvalidPitchValue = 12,
  hipErrorInvalidDevicePointer = 17,
  hipErrorInvalidDeviceFunction = 98,
  hipErrorInvalidMemcpyDirection = 21,
  hipErrorInvalidDevice = 101,
  hipErrorInvalidImage = 200,
  hipErrorNoBinaryForGpu = 209,
  hipErrorInvalidContext = 201,
  hipErrorFileNotFound = 301,
  hipErrorInvalidHandle = 400,
  hipErrorIllegalState = 401,
  hipErrorNotFound = 500,
  hipErrorNotReady = 600,
  hipErrorPeerAccessAlreadyEnabled = 704,
  hipErrorPeerAccessNotEnabled = 705,
  hipErrorLaunchFailure = 719,
  hipErrorCooperativeLaunchTooLarge = 720,
  hipErrorStreamCaptureUnsupported = 900,
  hipErrorStreamCaptureUnmatched = 903,
  hipErrorNotSupported = 801,
  hipErrorUnknown = 999,
} hipError_t;

typedef enum hipMemcpyKind {
  hipMemcpyHostToHost = 0,
  hipMemcpyHostToDevice = 1,
  hipMemcpyDeviceToHost = 2,
  hipMemcpyDeviceToDevice = 3,
  hipMemcpyDefault = 4,
} hipMemcpyKind;

typedef struct ihipModule_t* hipModule_t;
typedef struct ihipModuleSymbol_t* hipFunction_t;
typedef struct ihipStream_t* hipStream_t;
typedef struct ihipEvent_t* hipEvent_t;
typedef int hipDevice_t;

/* What hipModuleLaunchKernel takes in `extra`, as HIP documents it: the
 * argument buffer and its size, ended by HIP_LAUNCH_PARAM_END. */
#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE ((void*)0x02)
#define HIP_LAUNCH_PARAM_END ((void*)0x03)

/* hipDeviceProp_t as hipGetDeviceProperties fills it: the layout ROCm 5 gave
 * it, which the unsuffixed call keeps in every later release for programs
 * built to it (a program built with ROCm 6 or 7's headers calls
 * hipGetDevicePropertiesR0600 and reads that layout instead). The field list
 * is ROCm 5.7's hip_runtime_api.h (MIT licence, (c) Advanced Micro Devices);
 * the architecture flags, a word of bits there, are one opaque word here. */
typedef struct hipDeviceProp_t {
  char name[256];
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  int memoryClockRate;
  int memoryBusWidth;
  size_t totalConstMem;
  int major;
  int minor;
  int multiProcessorCount;
  int l2CacheSize;
  int maxThreadsPerMultiProcessor;
  int computeMode;
  int clockInstructionRate;
  struct { unsigned int bits; } arch;
  int concurrentKernels;
  int pciDomainID;
  int pciBusID;
  int pciDeviceID;
  size_t maxSharedMemoryPerMultiProcessor;
  int isMultiGpuBoard;
  int canMapHostMemory;
  int gcnArch;
  char gcnArchName[256];
  int integrated;
  int cooperativeLaunch;
  int cooperativeMultiDeviceLaunch;
  int maxTexture1DLinear;
  int maxTexture1D;
  int maxTexture2D[2];
  int maxTexture3D[3];
  unsigned int* hdpMemFlushCntl;
  unsigned int* hdpRegFlushCntl;
  size_t memPitch;
  size_t textureAlignment;
  size_t texturePitchAlignment;
  int kernelExecTimeoutEnabled;
  int ECCEnabled;
  int tccDriver;
  int cooperativeMultiDeviceUnmatchedFunc;
  int cooperativeMultiDeviceUnmatchedGridDim;
  int cooperativeMultiDeviceUnmatchedBlockDim;
  int cooperativeMultiDeviceUnmatchedSharedMem;
  int isLargeBar;
  int asicRevision;
  int managedMemory;
  int directManagedMemAccessFromHost;
  int concurrentManagedAccess;
  int pageableMemoryAccess;
  int pageableMemoryAccessUsesHostPageTables;
} hipDeviceProp_t;

hipError_t hipInit(unsigned int flags);
hipError_t hipGetDeviceCount(int* count);
hipError_t hipSetDevice(int device);
hipError_t hipGetDevice(int* device);
hipError_t hipGetDeviceProperties(hipDeviceProp_t* props, int device);
hipError_t hipDeviceSynchronize(void);
hipError_t hipDeviceReset(void);

hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipFree(void* ptr);
hipError_t hipMemcpy(void* dst, const void* src, size_t bytes, hipMemcpyKind kind);
hipError_t hipMemset(void* dst, int value, size_t bytes);
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t bytes, hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemsetAsync(void* dst, int value, size_t bytes, hipStream_t stream);
hipError_t hipMemGetInfo(size_t* free, size_t* total);
hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device);
hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device);
hipError_t hipDeviceGet(hipDevice_t* device, int ordinal);

hipError_t hipModuleLoad(hipModule_t* module, const char* path);
hipError_t hipModuleLoadData(hipModule_t* module, const void* image);
hipError_t hipModuleUnload(hipModule_t module);
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* name);
hipError_t hipModuleGetGlobal(void** dptr, size_t* bytes, hipModule_t module, const char* name);
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
                                 unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                                 unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream,
                                 void** kernelParams, void** extra);

const char* hipGetErrorString(hipError_t error);
const char* hipGetErrorName(hipError_t error);
hipError_t hipGetLastError(void);
hipError_t hipPeekAtLastError(void);
int hipGetStreamDeviceId(hipStream_t stream);
hipError_t hipStreamCreate(hipStream_t* stream);
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags);
hipError_t hipStreamDestroy(hipStream_t stream);
hipError_t hipStreamSynchronize(hipStream_t stream);
/* Events, which a program uses to time what the device did. */
#define hipEventDefault 0x0
#define hipEventBlockingSync 0x1
#define hipEventDisableTiming 0x2
#define hipStreamDefault 0x0
#define hipStreamNonBlocking 0x1

hipError_t hipEventCreate(hipEvent_t* event);
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int flags);
hipError_t hipEventDestroy(hipEvent_t event);
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream);
hipError_t hipEventSynchronize(hipEvent_t event);
hipError_t hipEventQuery(hipEvent_t event);
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t end);

hipError_t hipRuntimeGetVersion(int* version);
hipError_t hipDriverGetVersion(int* version);

#ifdef __cplusplus
}
#endif

#endif /* VGPU_HIP_H_ */
