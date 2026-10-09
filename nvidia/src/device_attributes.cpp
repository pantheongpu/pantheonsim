// What CUDA tells a program about a device, from one table; see
// vgpu/cuda_attributes.hpp for why there is one.
//
// Measured on an RTX 3060 (driver 13.0, GPU 1 of two) with
// nvidia/tests/e2e/device_attributes.cu --dump, which prints every
// cudaDeviceAttr, every CUdevice_attribute and every cudaDeviceProp member
// (the card's output is nvidia/tests/data/cuda_attributes_rtx3060.card.txt): the texture and surface
// limits, the alignments, the pitch, the 3D "alternate" extents, the access
// policy window and the stream priority range below are that card's, and
// those that are the same across compute capability 7.5 to 12.0 in the CUDA C
// Programming Guide's tables are taken to be.
#include "vgpu/cuda_attributes.hpp"

#include <cstdio>
#include <cstring>

#include "vgpu/telemetry.hpp"

namespace vgpu::cuda {

namespace {

// The ratio of single- to double-precision throughput (the Programming
// Guide's arithmetic-instruction table: fp32 results per clock per SM over fp64).
int single_to_double_ratio(int cc) {
  switch (cc) {
    case 70: case 80: case 90: case 100: case 103: return 2;   // the data-centre parts
    case 75: return 32;                                        // Turing
    default: return 64;                                        // consumer Ampere, Ada, Blackwell
  }
}

}  // namespace

Identity identity(const DeviceProfile& p, int physical) {
  telemetry::DeviceSample d{};
  telemetry::describe_device(p, physical, &d);
  Identity id{};
  // "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx": sixteen bytes in hex.
  int n = 0;
  for (const char* c = d.uuid + 4; *c && n < 32; ++c) {
    if (*c == '-') continue;
    const int v = *c >= 'a' ? *c - 'a' + 10 : *c >= 'A' ? *c - 'A' + 10 : *c - '0';
    id.uuid[n / 2] = static_cast<unsigned char>(n % 2 ? id.uuid[n / 2] | v : v << 4);
    ++n;
  }
  unsigned domain = 0, bus = 0, dev = 0, fn = 0;
  std::sscanf(d.bus_id, "%x:%x:%x.%x", &domain, &bus, &dev, &fn);
  id.domain = static_cast<int>(domain), id.bus = static_cast<int>(bus);
  id.device = static_cast<int>(dev), id.function = static_cast<int>(fn);
  return id;
}

void pci_bus_id(const Identity& id, char* out, int len) {
  std::snprintf(out, static_cast<size_t>(len), "%04x:%02x:%02x.%x", id.domain, id.bus, id.device, id.function);
}

bool device_attribute(const DeviceProfile& p, int physical, int id, int* out) {
  const int cc = p.cc_major * 10 + p.cc_minor;
  const bool hopper = cc >= 90;
  const bool ampere = cc >= 80;
  const Limits& lim = p.limits;
  const auto sat = [](uint64_t v) { return static_cast<int>(v > 2147483647u ? 2147483647u : v); };
  int v = 0;
  switch (id) {
    case kMaxThreadsPerBlock: v = static_cast<int>(lim.max_threads_per_block); break;
    case kMaxBlockDimX: v = static_cast<int>(lim.max_block_dim[0]); break;
    case kMaxBlockDimY: v = static_cast<int>(lim.max_block_dim[1]); break;
    case kMaxBlockDimZ: v = static_cast<int>(lim.max_block_dim[2]); break;
    case kMaxGridDimX: v = static_cast<int>(lim.max_grid_dim[0]); break;
    case kMaxGridDimY: v = static_cast<int>(lim.max_grid_dim[1]); break;
    case kMaxGridDimZ: v = static_cast<int>(lim.max_grid_dim[2]); break;
    case kMaxSharedMemoryPerBlock: v = static_cast<int>(lim.shared_mem_per_block); break;
    case kTotalConstantMemory: v = 65536; break;
    case kWarpSize: v = static_cast<int>(p.warp_size); break;
    case kMaxPitch: v = 2147483647; break;
    case kMaxRegistersPerBlock: v = static_cast<int>(lim.registers_per_block); break;
    // The boost clock a driver reports; see CudaClass.
    case kClockRate:
      v = sat(p.cuda.clock_khz ? p.cuda.clock_khz : uint64_t{p.telemetry.sm_clock_max_mhz} * 1000);
      break;
    case kTextureAlignment: v = 512; break;
    case kGpuOverlap: v = 1; break;
    case kMultiprocessorCount: v = static_cast<int>(lim.multiprocessors); break;
    // No watchdog stops a kernel: the card under WSL answers 1 because its
    // display driver has one, and a headless Linux GPU answers 0.
    case kKernelExecTimeout: v = 0; break;
    case kIntegrated: v = 0; break;
    case kCanMapHostMemory: v = 1; break;
    case kComputeMode: v = 0; break;   // CU_COMPUTEMODE_DEFAULT
    case kMaximumTexture1dWidth: v = 131072; break;
    case kMaximumTexture2dWidth: v = 131072; break;
    case kMaximumTexture2dHeight: v = 65536; break;
    case kMaximumTexture3dWidth: v = 16384; break;
    case kMaximumTexture3dHeight: v = 16384; break;
    case kMaximumTexture3dDepth: v = 16384; break;
    case kMaximumTexture2dLayeredWidth: v = 32768; break;
    case kMaximumTexture2dLayeredHeight: v = 32768; break;
    case kMaximumTexture2dLayeredLayers: v = 2048; break;
    case kSurfaceAlignment: v = 512; break;
    case kConcurrentKernels: v = 1; break;
    // On when the profile's card ships with ECC on, as NVML's ECC mode says.
    case kEccEnabled: v = p.telemetry.ecc ? 1 : 0; break;
    case kPciBusId: v = identity(p, physical).bus; break;
    case kPciDeviceId: v = identity(p, physical).device; break;
    case kTccDriver: v = 0; break;   // Linux has none
    case kMemoryClockRate: v = sat(uint64_t{p.telemetry.mem_clock_max_mhz} * 1000); break;
    case kGlobalMemoryBusWidth: v = static_cast<int>(p.cuda.memory_bus_width_bits); break;
    case kL2CacheSize: v = sat(lim.l2_cache_bytes); break;
    case kMaxThreadsPerMultiprocessor: v = static_cast<int>(lim.max_threads_per_sm); break;
    // Copy engines: the card's where the profile says; two, which every
    // current data-centre and consumer part has at least, where it does not.
    case kAsyncEngineCount: v = p.cuda.async_engine_count ? static_cast<int>(p.cuda.async_engine_count) : 2; break;
    case kUnifiedAddressing: v = 1; break;
    case kMaximumTexture1dLayeredWidth: v = 32768; break;
    case kMaximumTexture1dLayeredLayers: v = 2048; break;
    // tex2Dgather is PTX's tld4, which the interpreter and the SASS engine run.
    case kCanTex2dGather: v = 1; break;
    case kMaximumTexture2dGatherWidth: v = 32768; break;
    case kMaximumTexture2dGatherHeight: v = 32768; break;
    // The 3D extents a texture may have if it takes the 2D-array-like
    // alternative: 8192 x 8192 x 32768.
    case kMaximumTexture3dWidthAlternate: v = 8192; break;
    case kMaximumTexture3dHeightAlternate: v = 8192; break;
    case kMaximumTexture3dDepthAlternate: v = 32768; break;
    case kPciDomainId: v = identity(p, physical).domain; break;
    case kTexturePitchAlignment: v = 32; break;
    case kMaximumTexturecubemapWidth: v = 32768; break;
    case kMaximumTexturecubemapLayeredWidth: v = 32768; break;
    case kMaximumTexturecubemapLayeredLayers: v = 2046; break;
    case kMaximumSurface1dWidth: v = 32768; break;
    case kMaximumSurface2dWidth: v = 131072; break;
    case kMaximumSurface2dHeight: v = 65536; break;
    case kMaximumSurface3dWidth: v = 16384; break;
    case kMaximumSurface3dHeight: v = 16384; break;
    case kMaximumSurface3dDepth: v = 16384; break;
    case kMaximumSurface1dLayeredWidth: v = 32768; break;
    case kMaximumSurface1dLayeredLayers: v = 2048; break;
    case kMaximumSurface2dLayeredWidth: v = 32768; break;
    case kMaximumSurface2dLayeredHeight: v = 32768; break;
    case kMaximumSurface2dLayeredLayers: v = 2048; break;
    case kMaximumSurfacecubemapWidth: v = 32768; break;
    case kMaximumSurfacecubemapLayeredWidth: v = 32768; break;
    case kMaximumSurfacecubemapLayeredLayers: v = 2046; break;
    case kMaximumTexture1dLinearWidth: v = 268435456; break;
    case kMaximumTexture2dLinearWidth: v = 131072; break;
    case kMaximumTexture2dLinearHeight: v = 65000; break;
    case kMaximumTexture2dLinearPitch: v = 2097120; break;
    case kMaximumTexture2dMipmappedWidth: v = 32768; break;
    case kMaximumTexture2dMipmappedHeight: v = 32768; break;
    case kComputeCapabilityMajor: v = p.cc_major; break;
    case kComputeCapabilityMinor: v = p.cc_minor; break;
    case kMaximumTexture1dMipmappedWidth: v = 32768; break;
    case kStreamPrioritiesSupported: v = 1; break;
    case kGlobalL1CacheSupported: v = 1; break;
    case kLocalL1CacheSupported: v = 1; break;
    case kMaxSharedMemoryPerMultiprocessor: v = static_cast<int>(lim.shared_mem_per_sm); break;
    case kMaxRegistersPerMultiprocessor: v = static_cast<int>(lim.registers_per_sm); break;
    case kManagedMemory: v = 1; break;
    case kMultiGpuBoard: v = 0; break;
    case kMultiGpuBoardGroupId: v = 0; break;
    case kHostNativeAtomicSupported: v = 0; break;
    case kSingleToDoublePrecisionPerfRatio: v = single_to_double_ratio(cc); break;
    case kPageableMemoryAccess: v = 0; break;
    // The host and the device may touch managed memory at once, as on Linux
    // since Pascal (Windows and WSL answer 0): it is one host allocation here.
    case kConcurrentManagedAccess: v = 1; break;
    case kComputePreemptionSupported: v = 1; break;
    case kCanUseHostPointerForRegisteredMem: v = 0; break;
    // The stream memory operations 92-94 were version 1's, which no current
    // card answers; 122 and 123 are the 64-bit and NOR forms of version 2.
    // cuStreamWriteValue32 and the 32-bit wait exist; the 64-bit write and
    // wait, the NOR flag and cuStreamBatchMemOp do not, so these are 0.
    case kCanUseStreamMemOpsV1: v = 0; break;
    case kCanUse64BitStreamMemOpsV1: v = 0; break;
    case kCanUseStreamWaitValueNorV1: v = 0; break;
    // Grid-wide sync works under cudaLaunchCooperativeKernel, because the
    // scheduler can hold every block resident; the multi-device form needs
    // grids on separate devices waiting on each other, and is not offered.
    case kCooperativeLaunch: v = 1; break;
    case kCooperativeMultiDeviceLaunch: v = 0; break;
    case kMaxSharedMemoryPerBlockOptin: v = static_cast<int>(lim.shared_mem_per_block_optin); break;
    case kCanFlushRemoteWrites: v = 0; break;
    case kHostRegisterSupported: v = 1; break;
    case kPageableMemoryAccessUsesHostPageTables: v = 0; break;
    case kDirectManagedMemAccessFromHost: v = 0; break;
    // cuMemAddressReserve, cuMemCreate, cuMemMap and cuMemSetAccess work.
    case kVirtualAddressManagementSupported: v = 1; break;
    // cuMemCreate does not hand out file descriptors, and nothing else here
    // is shared between processes through a handle.
    case kHandleTypePosixFileDescriptorSupported: v = 0; break;
    case kHandleTypeWin32HandleSupported: v = 0; break;
    case kHandleTypeWin32KmtHandleSupported: v = 0; break;
    // A real quantity: a divisor in occupancy arithmetic (CUB's scan launched
    // no blocks when it read zero).
    case kMaxBlocksPerMultiprocessor: v = static_cast<int>(lim.max_blocks_per_sm); break;
    // cuMemCreate ignores the compression flag, so the capability is not
    // claimed: a program that sees it asks for a compressed allocation and
    // reads back that it is not.
    case kGenericCompressionSupported: v = 0; break;
    case kMaxPersistingL2CacheSize: v = sat(p.cuda.persisting_l2_bytes); break;
    // 128 MiB less a page, from compute capability 8.0 (an RTX 3060 answers it;
    // there is no access policy window before).
    case kMaxAccessPolicyWindowSize: v = ampere ? 134213632 : 0; break;
    case kGpuDirectRdmaWithCudaVmmSupported: v = 0; break;
    case kReservedSharedMemoryPerBlock: v = static_cast<int>(p.reserved_smem_per_block()); break;
    // Sparse and deferred-mapped CUDA arrays are not implemented: a card
    // says 1, and a program that sees it makes one.
    case kSparseCudaArraySupported: v = 0; break;
    // cuMemHostRegister takes CU_MEMHOSTREGISTER_READ_ONLY: a kernel's store or atomic to the
    // range faults, as on an RTX 3060, which answers 1.
    case kReadOnlyHostRegisterSupported: v = 1; break;
    case kTimelineSemaphoreInteropSupported: v = 0; break;
    // The stream-ordered allocator is implemented (cudaMallocAsync, the
    // cudaMemPool* API), and the runtime says so.
    case kMemoryPoolsSupported: v = 1; break;
    case kGpuDirectRdmaSupported: v = 0; break;
    case kGpuDirectRdmaFlushWritesOptions: v = 0; break;
    case kGpuDirectRdmaWritesOrdering: v = 0; break;
    // No pool can be exported to another process, so no handle types.
    case kMempoolSupportedHandleTypes: v = 0; break;
    // Thread-block clusters run (cudaLaunchAttributeClusterDimension, the
    // cluster barriers and distributed shared memory), from Hopper.
    case kClusterLaunch: v = hopper ? 1 : 0; break;
    case kDeferredMappingCudaArraySupported: v = 0; break;
    case kCanUse64BitStreamMemOps: v = 0; break;
    case kCanUseStreamWaitValueNor: v = 0; break;
    case kDmaBufSupported: v = 0; break;
    case kIpcEventSupported: v = 1; break;
    // Memory synchronization domains: one before Hopper, which has four.
    case kMemSyncDomainCount: v = hopper ? 4 : 1; break;
    // The tensor map API (cuTensorMapEncodeTiled and its TMA loads) is Hopper's.
    case kTensorMapAccessSupported: v = hopper ? 1 : 0; break;
    case kHandleTypeFabricSupported: v = 0; break;
    case kUnifiedFunctionPointers: v = 0; break;
    case kNumaConfig: v = 0; break;      // CU_DEVICE_NUMA_CONFIG_NONE
    case kNumaId: v = -1; break;         // a PCIe GPU has no NUMA node of its own
    case kMulticastSupported: v = 0; break;
    case kMpsEnabled: v = 0; break;
    case kHostNumaId: v = 0; break;
    case kD3d12CigSupported: v = 0; break;
    case kMemDecompressAlgorithmMask: v = 0; break;
    case kMemDecompressMaximumLength: v = 0; break;
    case kVulkanCigSupported: v = 0; break;
    case kGpuPciDeviceId: v = static_cast<int>((p.telemetry.pci_device_id << 16) | p.telemetry.pci_vendor_id); break;
    // The same as the device id: the simulated board has no vendor of its own
    // (NVML reports the same, from telemetry::describe_device).
    case kGpuPciSubsystemId: v = static_cast<int>((p.telemetry.pci_device_id << 16) | p.telemetry.pci_vendor_id); break;
    // Memory pools and virtual memory on host memory are not implemented.
    case kHostNumaVirtualMemoryManagementSupported: v = 0; break;
    case kHostNumaMemoryPoolsSupported: v = 0; break;
    case kHostNumaMultinodeIpcSupported: v = 0; break;
    case kHostMemoryPoolsSupported: v = 0; break;
    case kHostVirtualMemoryManagementSupported: v = 0; break;
    case kHostAllocDmaBufSupported: v = 0; break;
    case kOnlyPartialHostNativeAtomicSupported: v = 0; break;
    case kAtomicReductionSupported: v = 0; break;
    default: return false;
  }
  *out = v;
  return true;
}

}  // namespace vgpu::cuda
