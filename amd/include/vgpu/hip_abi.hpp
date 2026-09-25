// The structures a program built by hipcc hands the runtime, laid out as the
// HIP headers lay them out -- not as VirtualGPU's own vgpu_hip.h does, which is
// a subset for programs written against it. A binary compiled against the real
// headers reads fields at the real offsets, so these have to match them field
// for field. test_amd_hip_abi checks that against the headers themselves
// wherever ROCm is installed.
//
// The device-property structure and the architecture flags inside it are
// carried over from AMD's hip_runtime_api.h, which is distributed under the
// MIT licence below; the comments are VirtualGPU's own.
//
// Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#pragma once

#include <cstddef>
#include <cstdint>

namespace vgpu::amd::abi {

// A launch's grid or block size, as hipcc's chevron syntax passes it.
struct Dim3 {
  uint32_t x, y, z;
};

// What __hipRegisterFatBinary is handed: a small wrapper in the executable's
// .hipFatBinSegment section, pointing at the clang offload bundle in its
// .hip_fatbin section that holds the device code for each target.
struct FatbinWrapper {
  uint32_t magic;     // "HIPF"
  uint32_t version;   // 1
  const void* binary;
  const void* unused;
};
inline constexpr uint32_t kFatbinMagic = 0x48495046;

struct Uuid {
  char bytes[16];
};

struct DeviceArch {
  unsigned hasGlobalInt32Atomics : 1;
  unsigned hasGlobalFloatAtomicExch : 1;
  unsigned hasSharedInt32Atomics : 1;
  unsigned hasSharedFloatAtomicExch : 1;
  unsigned hasFloatAtomicAdd : 1;
  unsigned hasGlobalInt64Atomics : 1;
  unsigned hasSharedInt64Atomics : 1;
  unsigned hasDoubles : 1;
  unsigned hasWarpVote : 1;
  unsigned hasWarpBallot : 1;
  unsigned hasWarpShuffle : 1;
  unsigned hasFunnelShift : 1;
  unsigned hasThreadFenceSystem : 1;
  unsigned hasSyncThreadsExt : 1;
  unsigned hasSurfaceFuncs : 1;
  unsigned has3dGrid : 1;
  unsigned hasDynamicParallelism : 1;
};

// hipDeviceProp_tR0600: what hipGetDeviceProperties fills in from ROCm 6.0 on.
struct DevicePropR0600 {
  char name[256];
  Uuid uuid;
  char luid[8];
  unsigned int luidDeviceNodeMask;
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  size_t memPitch;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  size_t totalConstMem;
  int major;
  int minor;
  size_t textureAlignment;
  size_t texturePitchAlignment;
  int deviceOverlap;
  int multiProcessorCount;
  int kernelExecTimeoutEnabled;
  int integrated;
  int canMapHostMemory;
  int computeMode;
  int maxTexture1D;
  int maxTexture1DMipmap;
  int maxTexture1DLinear;
  int maxTexture2D[2];
  int maxTexture2DMipmap[2];
  int maxTexture2DLinear[3];
  int maxTexture2DGather[2];
  int maxTexture3D[3];
  int maxTexture3DAlt[3];
  int maxTextureCubemap;
  int maxTexture1DLayered[2];
  int maxTexture2DLayered[3];
  int maxTextureCubemapLayered[2];
  int maxSurface1D;
  int maxSurface2D[2];
  int maxSurface3D[3];
  int maxSurface1DLayered[2];
  int maxSurface2DLayered[3];
  int maxSurfaceCubemap;
  int maxSurfaceCubemapLayered[2];
  size_t surfaceAlignment;
  int concurrentKernels;
  int ECCEnabled;
  int pciBusID;
  int pciDeviceID;
  int pciDomainID;
  int tccDriver;
  int asyncEngineCount;
  int unifiedAddressing;
  int memoryClockRate;
  int memoryBusWidth;
  int l2CacheSize;
  int persistingL2CacheMaxSize;
  int maxThreadsPerMultiProcessor;
  int streamPrioritiesSupported;
  int globalL1CacheSupported;
  int localL1CacheSupported;
  size_t sharedMemPerMultiprocessor;
  int regsPerMultiprocessor;
  int managedMemory;
  int isMultiGpuBoard;
  int multiGpuBoardGroupID;
  int hostNativeAtomicSupported;
  int singleToDoublePrecisionPerfRatio;
  int pageableMemoryAccess;
  int concurrentManagedAccess;
  int computePreemptionSupported;
  int canUseHostPointerForRegisteredMem;
  int cooperativeLaunch;
  int cooperativeMultiDeviceLaunch;
  size_t sharedMemPerBlockOptin;
  int pageableMemoryAccessUsesHostPageTables;
  int directManagedMemAccessFromHost;
  int maxBlocksPerMultiProcessor;
  int accessPolicyMaxWindowSize;
  size_t reservedSharedMemPerBlock;
  int hostRegisterSupported;
  int sparseHipArraySupported;
  int hostRegisterReadOnlySupported;
  int timelineSemaphoreInteropSupported;
  int memoryPoolsSupported;
  int gpuDirectRDMASupported;
  unsigned int gpuDirectRDMAFlushWritesOptions;
  int gpuDirectRDMAWritesOrdering;
  unsigned int
      memoryPoolSupportedHandleTypes;
  int deferredMappingHipArraySupported;
  int ipcEventSupported;
  int clusterLaunch;
  int unifiedFunctionPointers;
  int reserved[63];
  int hipReserved[32];
  char gcnArchName[256];
  size_t maxSharedMemoryPerMultiProcessor;
  int clockInstructionRate;
  DeviceArch arch;
  unsigned int* hdpMemFlushCntl;
  unsigned int* hdpRegFlushCntl;
  int cooperativeMultiDeviceUnmatchedFunc;
  int cooperativeMultiDeviceUnmatchedGridDim;
  int cooperativeMultiDeviceUnmatchedBlockDim;
  int cooperativeMultiDeviceUnmatchedSharedMem;
  int isLargeBar;
  int asicRevision;
};

// hipDeviceAttribute_t: what hipDeviceGetAttribute is asked for, by the
// numbers hip_runtime_api.h gives them (the ones answered here; run_hip_abi.sh
// checks each against the header). Each is answered from the same device
// properties hipGetDeviceProperties returns.
enum class DeviceAttribute : int {
  kEccEnabled = 0,
  kAsyncEngineCount = 2,
  kCanMapHostMemory = 3,
  kCanUseHostPointerForRegisteredMem = 4,
  kClockRate = 5,
  kComputeMode = 6,
  kComputePreemptionSupported = 7,
  kConcurrentKernels = 8,
  kConcurrentManagedAccess = 9,
  kCooperativeLaunch = 10,
  kCooperativeMultiDeviceLaunch = 11,
  kDeviceOverlap = 12,
  kDirectManagedMemAccessFromHost = 13,
  kGlobalL1CacheSupported = 14,
  kHostNativeAtomicSupported = 15,
  kIntegrated = 16,
  kIsMultiGpuBoard = 17,
  kKernelExecTimeout = 18,
  kL2CacheSize = 19,
  kLocalL1CacheSupported = 20,
  kComputeCapabilityMajor = 23,
  kManagedMemory = 24,
  kMaxBlocksPerMultiProcessor = 25,
  kMaxBlockDimX = 26,
  kMaxBlockDimY = 27,
  kMaxBlockDimZ = 28,
  kMaxGridDimX = 29,
  kMaxGridDimY = 30,
  kMaxGridDimZ = 31,
  kMaxThreadsPerBlock = 56,
  kMaxThreadsPerMultiProcessor = 57,
  kMaxPitch = 58,
  kMemoryBusWidth = 59,
  kMemoryClockRate = 60,
  kComputeCapabilityMinor = 61,
  kMultiGpuBoardGroupID = 62,
  kMultiprocessorCount = 63,
  kPageableMemoryAccess = 65,
  kPageableMemoryAccessUsesHostPageTables = 66,
  kPciBusId = 67,
  kPciDeviceId = 68,
  kPciDomainID = 69,
  kPersistingL2CacheMaxSize = 70,
  kMaxRegistersPerBlock = 71,
  kMaxRegistersPerMultiprocessor = 72,
  kReservedSharedMemPerBlock = 73,
  kMaxSharedMemoryPerBlock = 74,
  kSharedMemPerBlockOptin = 75,
  kSharedMemPerMultiprocessor = 76,
  kSingleToDoublePrecisionPerfRatio = 77,
  kStreamPrioritiesSupported = 78,
  kSurfaceAlignment = 79,
  kTccDriver = 80,
  kTextureAlignment = 81,
  kTexturePitchAlignment = 82,
  kTotalConstantMemory = 83,
  kTotalGlobalMem = 84,
  kUnifiedAddressing = 85,
  kWarpSize = 87,
  kMemoryPoolsSupported = 88,
  kHostRegisterSupported = 90,
  kClockInstructionRate = 10000,
  kMaxSharedMemoryPerMultiprocessor = 10002,
  kCooperativeMultiDeviceUnmatchedFunc = 10007,
  kCooperativeMultiDeviceUnmatchedGridDim = 10008,
  kCooperativeMultiDeviceUnmatchedBlockDim = 10009,
  kCooperativeMultiDeviceUnmatchedSharedMem = 10010,
  kIsLargeBar = 10011,
  kAsicRevision = 10012,
  kPhysicalMultiProcessorCount = 10015,
  kCanUseStreamWaitValue = 10013,
  kImageSupport = 10014,
  kFineGrainSupport = 10016,
};

}  // namespace vgpu::amd::abi
