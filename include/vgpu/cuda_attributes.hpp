// What CUDA tells a program about a device: cuDeviceGetAttribute,
// cudaDeviceGetAttribute and cudaDeviceProp, from one table.
//
// Three places used to answer these -- the driver, the runtime's attribute
// switch and its cudaGetDeviceProperties -- each with its own constants, and
// they disagreed: the L2 was 8 MiB in all three whatever the card, the memory
// bus 256 bits, the memory clock a million kHz, and about a hundred attributes
// the driver answered, the runtime refused (cudaErrorInvalidValue), among them
// every texture and surface limit. A program that sizes work by the L2, or
// prints deviceQuery, saw numbers no card has.
//
// Where a value is a fact of the card it comes from the profile (the `cuda:`
// section and `limits:`), which says where each was read. Where it is a rule
// of the compute capability it is that rule, as the CUDA C Programming Guide's
// technical-specification tables give it. Where it says whether a capability
// exists, it says yes only if the simulator implements it: a card's answer is
// not copied for a feature this does not have (sparse arrays, deferred array
// mapping, host memory pools), because a program that sees the capability
// takes the path that uses it.
#pragma once

#include <cstdint>

#include "vgpu/profile.hpp"

namespace vgpu::cuda {

// CUdevice_attribute's numbers, which are also cudaDeviceAttr's for the same
// attribute (the runtime's "Reserved" ones are driver attributes it does not
// name). From CUDA 13.2's cuda.h; these are ABI, so they do not depend on the
// toolkit the shim is built with.
enum Attr : int {
  kMaxThreadsPerBlock = 1,   // MAX_THREADS_PER_BLOCK
  kMaxBlockDimX = 2,   // MAX_BLOCK_DIM_X
  kMaxBlockDimY = 3,   // MAX_BLOCK_DIM_Y
  kMaxBlockDimZ = 4,   // MAX_BLOCK_DIM_Z
  kMaxGridDimX = 5,   // MAX_GRID_DIM_X
  kMaxGridDimY = 6,   // MAX_GRID_DIM_Y
  kMaxGridDimZ = 7,   // MAX_GRID_DIM_Z
  kMaxSharedMemoryPerBlock = 8,   // MAX_SHARED_MEMORY_PER_BLOCK
  kTotalConstantMemory = 9,   // TOTAL_CONSTANT_MEMORY
  kWarpSize = 10,   // WARP_SIZE
  kMaxPitch = 11,   // MAX_PITCH
  kMaxRegistersPerBlock = 12,   // MAX_REGISTERS_PER_BLOCK
  kClockRate = 13,   // CLOCK_RATE
  kTextureAlignment = 14,   // TEXTURE_ALIGNMENT
  kGpuOverlap = 15,   // GPU_OVERLAP
  kMultiprocessorCount = 16,   // MULTIPROCESSOR_COUNT
  kKernelExecTimeout = 17,   // KERNEL_EXEC_TIMEOUT
  kIntegrated = 18,   // INTEGRATED
  kCanMapHostMemory = 19,   // CAN_MAP_HOST_MEMORY
  kComputeMode = 20,   // COMPUTE_MODE
  kMaximumTexture1dWidth = 21,   // MAXIMUM_TEXTURE1D_WIDTH
  kMaximumTexture2dWidth = 22,   // MAXIMUM_TEXTURE2D_WIDTH
  kMaximumTexture2dHeight = 23,   // MAXIMUM_TEXTURE2D_HEIGHT
  kMaximumTexture3dWidth = 24,   // MAXIMUM_TEXTURE3D_WIDTH
  kMaximumTexture3dHeight = 25,   // MAXIMUM_TEXTURE3D_HEIGHT
  kMaximumTexture3dDepth = 26,   // MAXIMUM_TEXTURE3D_DEPTH
  kMaximumTexture2dLayeredWidth = 27,   // MAXIMUM_TEXTURE2D_LAYERED_WIDTH
  kMaximumTexture2dLayeredHeight = 28,   // MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
  kMaximumTexture2dLayeredLayers = 29,   // MAXIMUM_TEXTURE2D_LAYERED_LAYERS
  kSurfaceAlignment = 30,   // SURFACE_ALIGNMENT
  kConcurrentKernels = 31,   // CONCURRENT_KERNELS
  kEccEnabled = 32,   // ECC_ENABLED
  kPciBusId = 33,   // PCI_BUS_ID
  kPciDeviceId = 34,   // PCI_DEVICE_ID
  kTccDriver = 35,   // TCC_DRIVER
  kMemoryClockRate = 36,   // MEMORY_CLOCK_RATE
  kGlobalMemoryBusWidth = 37,   // GLOBAL_MEMORY_BUS_WIDTH
  kL2CacheSize = 38,   // L2_CACHE_SIZE
  kMaxThreadsPerMultiprocessor = 39,   // MAX_THREADS_PER_MULTIPROCESSOR
  kAsyncEngineCount = 40,   // ASYNC_ENGINE_COUNT
  kUnifiedAddressing = 41,   // UNIFIED_ADDRESSING
  kMaximumTexture1dLayeredWidth = 42,   // MAXIMUM_TEXTURE1D_LAYERED_WIDTH
  kMaximumTexture1dLayeredLayers = 43,   // MAXIMUM_TEXTURE1D_LAYERED_LAYERS
  kCanTex2dGather = 44,   // CAN_TEX2D_GATHER
  kMaximumTexture2dGatherWidth = 45,   // MAXIMUM_TEXTURE2D_GATHER_WIDTH
  kMaximumTexture2dGatherHeight = 46,   // MAXIMUM_TEXTURE2D_GATHER_HEIGHT
  kMaximumTexture3dWidthAlternate = 47,   // MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE
  kMaximumTexture3dHeightAlternate = 48,   // MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE
  kMaximumTexture3dDepthAlternate = 49,   // MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE
  kPciDomainId = 50,   // PCI_DOMAIN_ID
  kTexturePitchAlignment = 51,   // TEXTURE_PITCH_ALIGNMENT
  kMaximumTexturecubemapWidth = 52,   // MAXIMUM_TEXTURECUBEMAP_WIDTH
  kMaximumTexturecubemapLayeredWidth = 53,   // MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH
  kMaximumTexturecubemapLayeredLayers = 54,   // MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS
  kMaximumSurface1dWidth = 55,   // MAXIMUM_SURFACE1D_WIDTH
  kMaximumSurface2dWidth = 56,   // MAXIMUM_SURFACE2D_WIDTH
  kMaximumSurface2dHeight = 57,   // MAXIMUM_SURFACE2D_HEIGHT
  kMaximumSurface3dWidth = 58,   // MAXIMUM_SURFACE3D_WIDTH
  kMaximumSurface3dHeight = 59,   // MAXIMUM_SURFACE3D_HEIGHT
  kMaximumSurface3dDepth = 60,   // MAXIMUM_SURFACE3D_DEPTH
  kMaximumSurface1dLayeredWidth = 61,   // MAXIMUM_SURFACE1D_LAYERED_WIDTH
  kMaximumSurface1dLayeredLayers = 62,   // MAXIMUM_SURFACE1D_LAYERED_LAYERS
  kMaximumSurface2dLayeredWidth = 63,   // MAXIMUM_SURFACE2D_LAYERED_WIDTH
  kMaximumSurface2dLayeredHeight = 64,   // MAXIMUM_SURFACE2D_LAYERED_HEIGHT
  kMaximumSurface2dLayeredLayers = 65,   // MAXIMUM_SURFACE2D_LAYERED_LAYERS
  kMaximumSurfacecubemapWidth = 66,   // MAXIMUM_SURFACECUBEMAP_WIDTH
  kMaximumSurfacecubemapLayeredWidth = 67,   // MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH
  kMaximumSurfacecubemapLayeredLayers = 68,   // MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS
  kMaximumTexture1dLinearWidth = 69,   // MAXIMUM_TEXTURE1D_LINEAR_WIDTH
  kMaximumTexture2dLinearWidth = 70,   // MAXIMUM_TEXTURE2D_LINEAR_WIDTH
  kMaximumTexture2dLinearHeight = 71,   // MAXIMUM_TEXTURE2D_LINEAR_HEIGHT
  kMaximumTexture2dLinearPitch = 72,   // MAXIMUM_TEXTURE2D_LINEAR_PITCH
  kMaximumTexture2dMipmappedWidth = 73,   // MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH
  kMaximumTexture2dMipmappedHeight = 74,   // MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT
  kComputeCapabilityMajor = 75,   // COMPUTE_CAPABILITY_MAJOR
  kComputeCapabilityMinor = 76,   // COMPUTE_CAPABILITY_MINOR
  kMaximumTexture1dMipmappedWidth = 77,   // MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH
  kStreamPrioritiesSupported = 78,   // STREAM_PRIORITIES_SUPPORTED
  kGlobalL1CacheSupported = 79,   // GLOBAL_L1_CACHE_SUPPORTED
  kLocalL1CacheSupported = 80,   // LOCAL_L1_CACHE_SUPPORTED
  kMaxSharedMemoryPerMultiprocessor = 81,   // MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
  kMaxRegistersPerMultiprocessor = 82,   // MAX_REGISTERS_PER_MULTIPROCESSOR
  kManagedMemory = 83,   // MANAGED_MEMORY
  kMultiGpuBoard = 84,   // MULTI_GPU_BOARD
  kMultiGpuBoardGroupId = 85,   // MULTI_GPU_BOARD_GROUP_ID
  kHostNativeAtomicSupported = 86,   // HOST_NATIVE_ATOMIC_SUPPORTED
  kSingleToDoublePrecisionPerfRatio = 87,   // SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO
  kPageableMemoryAccess = 88,   // PAGEABLE_MEMORY_ACCESS
  kConcurrentManagedAccess = 89,   // CONCURRENT_MANAGED_ACCESS
  kComputePreemptionSupported = 90,   // COMPUTE_PREEMPTION_SUPPORTED
  kCanUseHostPointerForRegisteredMem = 91,   // CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM
  kCanUseStreamMemOpsV1 = 92,   // CAN_USE_STREAM_MEM_OPS_V1
  kCanUse64BitStreamMemOpsV1 = 93,   // CAN_USE_64_BIT_STREAM_MEM_OPS_V1
  kCanUseStreamWaitValueNorV1 = 94,   // CAN_USE_STREAM_WAIT_VALUE_NOR_V1
  kCooperativeLaunch = 95,   // COOPERATIVE_LAUNCH
  kCooperativeMultiDeviceLaunch = 96,   // COOPERATIVE_MULTI_DEVICE_LAUNCH
  kMaxSharedMemoryPerBlockOptin = 97,   // MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
  kCanFlushRemoteWrites = 98,   // CAN_FLUSH_REMOTE_WRITES
  kHostRegisterSupported = 99,   // HOST_REGISTER_SUPPORTED
  kPageableMemoryAccessUsesHostPageTables = 100,   // PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
  kDirectManagedMemAccessFromHost = 101,   // DIRECT_MANAGED_MEM_ACCESS_FROM_HOST
  kVirtualAddressManagementSupported = 102,   // VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
  kHandleTypePosixFileDescriptorSupported = 103,   // HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED
  kHandleTypeWin32HandleSupported = 104,   // HANDLE_TYPE_WIN32_HANDLE_SUPPORTED
  kHandleTypeWin32KmtHandleSupported = 105,   // HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED
  kMaxBlocksPerMultiprocessor = 106,   // MAX_BLOCKS_PER_MULTIPROCESSOR
  kGenericCompressionSupported = 107,   // GENERIC_COMPRESSION_SUPPORTED
  kMaxPersistingL2CacheSize = 108,   // MAX_PERSISTING_L2_CACHE_SIZE
  kMaxAccessPolicyWindowSize = 109,   // MAX_ACCESS_POLICY_WINDOW_SIZE
  kGpuDirectRdmaWithCudaVmmSupported = 110,   // GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED
  kReservedSharedMemoryPerBlock = 111,   // RESERVED_SHARED_MEMORY_PER_BLOCK
  kSparseCudaArraySupported = 112,   // SPARSE_CUDA_ARRAY_SUPPORTED
  kReadOnlyHostRegisterSupported = 113,   // READ_ONLY_HOST_REGISTER_SUPPORTED
  kTimelineSemaphoreInteropSupported = 114,   // TIMELINE_SEMAPHORE_INTEROP_SUPPORTED
  kMemoryPoolsSupported = 115,   // MEMORY_POOLS_SUPPORTED
  kGpuDirectRdmaSupported = 116,   // GPU_DIRECT_RDMA_SUPPORTED
  kGpuDirectRdmaFlushWritesOptions = 117,   // GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS
  kGpuDirectRdmaWritesOrdering = 118,   // GPU_DIRECT_RDMA_WRITES_ORDERING
  kMempoolSupportedHandleTypes = 119,   // MEMPOOL_SUPPORTED_HANDLE_TYPES
  kClusterLaunch = 120,   // CLUSTER_LAUNCH
  kDeferredMappingCudaArraySupported = 121,   // DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED
  kCanUse64BitStreamMemOps = 122,   // CAN_USE_64_BIT_STREAM_MEM_OPS
  kCanUseStreamWaitValueNor = 123,   // CAN_USE_STREAM_WAIT_VALUE_NOR
  kDmaBufSupported = 124,   // DMA_BUF_SUPPORTED
  kIpcEventSupported = 125,   // IPC_EVENT_SUPPORTED
  kMemSyncDomainCount = 126,   // MEM_SYNC_DOMAIN_COUNT
  kTensorMapAccessSupported = 127,   // TENSOR_MAP_ACCESS_SUPPORTED
  kHandleTypeFabricSupported = 128,   // HANDLE_TYPE_FABRIC_SUPPORTED
  kUnifiedFunctionPointers = 129,   // UNIFIED_FUNCTION_POINTERS
  kNumaConfig = 130,   // NUMA_CONFIG
  kNumaId = 131,   // NUMA_ID
  kMulticastSupported = 132,   // MULTICAST_SUPPORTED
  kMpsEnabled = 133,   // MPS_ENABLED
  kHostNumaId = 134,   // HOST_NUMA_ID
  kD3d12CigSupported = 135,   // D3D12_CIG_SUPPORTED
  kMemDecompressAlgorithmMask = 136,   // MEM_DECOMPRESS_ALGORITHM_MASK
  kMemDecompressMaximumLength = 137,   // MEM_DECOMPRESS_MAXIMUM_LENGTH
  kVulkanCigSupported = 138,   // VULKAN_CIG_SUPPORTED
  kGpuPciDeviceId = 139,   // GPU_PCI_DEVICE_ID
  kGpuPciSubsystemId = 140,   // GPU_PCI_SUBSYSTEM_ID
  kHostNumaVirtualMemoryManagementSupported = 141,   // HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
  kHostNumaMemoryPoolsSupported = 142,   // HOST_NUMA_MEMORY_POOLS_SUPPORTED
  kHostNumaMultinodeIpcSupported = 143,   // HOST_NUMA_MULTINODE_IPC_SUPPORTED
  kHostMemoryPoolsSupported = 144,   // HOST_MEMORY_POOLS_SUPPORTED
  kHostVirtualMemoryManagementSupported = 145,   // HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
  kHostAllocDmaBufSupported = 146,   // HOST_ALLOC_DMA_BUF_SUPPORTED
  kOnlyPartialHostNativeAtomicSupported = 147,   // ONLY_PARTIAL_HOST_NATIVE_ATOMIC_SUPPORTED
  kAtomicReductionSupported = 148,   // ATOMIC_REDUCTION_SUPPORTED
};

// The value of attribute `id` for a device of `profile`, the machine's
// `physical` device (its PCI slot and identity come from that). False, with
// `*value` untouched, for an id that is not an attribute of this CUDA.
bool device_attribute(const DeviceProfile& profile, int physical, int id, int* value);

// The device's identity as CUDA reports it, the same as NVML's for the same
// device (telemetry::describe_device is the one source): its UUID's sixteen
// bytes, and its PCI address.
struct Identity {
  unsigned char uuid[16];
  int domain, bus, device, function;
};
Identity identity(const DeviceProfile& profile, int physical);

// The priorities a stream can have: 0 is the default, and -5 the highest
// (cudaDeviceGetStreamPriorityRange on an RTX 3060 says 0 and -5, as does
// every compute capability from 7.0). Streams here run in order, so a
// priority is kept and read back and nothing more.
inline constexpr int kLeastStreamPriority = 0;
inline constexpr int kGreatestStreamPriority = -5;

// "0000:01:00.0", as cudaDeviceGetPCIBusId and cuDeviceGetPCIBusId write it
// (NVML writes the domain with eight digits).
void pci_bus_id(const Identity& id, char* out, int len);

}  // namespace vgpu::cuda
