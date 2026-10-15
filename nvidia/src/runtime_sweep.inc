// The runtime functions the toolkit's cuda_runtime_api.h declares and this library lacked, found by
// tests/lint/check_header_exports.py (the sweep of 2026-10-10). Included by runtime_api.cpp, whose helpers
// they use. Each is exported under the toolkit's name and answers as an RTX 3060 under NVIDIA's CUDA 13.0
// runtime does where that was measured (nvidia/tests/data/exports_sweep_runtime.rtx3060.expected, made by
// run_exports_sweep.sh --card); where it was not, the comment says what the answer rests on. A function this
// simulator cannot honour is still exported and answers the library's own "not supported" status, once said on
// stderr (never a plausible wrong answer).
//
// Calls the card crashes on -- a garbage handle, a callback unregistered twice -- are not imitated: they are
// refused with the status the documentation gives for a bad handle.

#include <memory>
#include <set>

namespace {

// A function that is not simulated: its status, and one line on stderr the first time it is called.
cudaError_t refuse(const char* api, const char* why, cudaError_t code = cudaErrorNotSupported) {
  static std::mutex mu;
  static std::set<std::string> said;
  std::lock_guard<std::mutex> lock(mu);
  if (!quiet() && said.insert(api).second)
    std::fprintf(stderr, "[vgpu] %s is not implemented by VirtualGPU: %s; returning %s\n", api, why, cudaGetErrorName(code));
  return code;
}

}  // namespace

// ---- shared memory configuration ---------------------------------------------------------------------------
// Shared memory banks are four bytes wide on every GPU this simulates, as an RTX 3060 answers: a
// configuration in the enumeration is accepted and has no effect, and the answer stays the four-byte bank size
// (cudaSharedMemBankSizeFourByte, 1) whatever was set. Deprecated in 12.4. A configuration travels as an int:
// an out-of-range value must not be loaded as the enum (UBSan), and traced_call copies its arguments on.
static cudaError_t cudaDeviceSetSharedMemConfig_traced(int c) {
  return c >= 0 && c <= 2 ? cudaSuccess : cudaErrorInvalidValue;
}

VGPU_EXPORT cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config) {
  return traced_call("cudaDeviceSetSharedMemConfig", cudaDeviceSetSharedMemConfig_traced, enum_value(config));
}

static cudaError_t cudaDeviceGetSharedMemConfig_traced(cudaSharedMemConfig* config) {
  if (!config) return cudaErrorInvalidValue;
  *config = cudaSharedMemBankSizeFourByte;
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* config) {
  return traced_call("cudaDeviceGetSharedMemConfig", cudaDeviceGetSharedMemConfig_traced, config);
}

// A function that was not registered is cudaErrorInvalidDeviceFunction when it is null and
// cudaErrorInvalidResourceHandle for any other address (measured).
static cudaError_t cudaFuncSetSharedMemConfig_traced(const void* func, int c) {
  if (c < 0 || c > 2) return cudaErrorInvalidValue;
  return guard("cudaFuncSetSharedMemConfig", [&](State& s) -> cudaError_t {
    if (!func) return cudaErrorInvalidDeviceFunction;
    return s.kernels.count(func) ? cudaSuccess : cudaErrorInvalidResourceHandle;
  });
}

VGPU_EXPORT cudaError_t cudaFuncSetSharedMemConfig(const void* func, cudaSharedMemConfig config) {
  return traced_call("cudaFuncSetSharedMemConfig", cudaFuncSetSharedMemConfig_traced, func, enum_value(config));
}

// ---- cudaSetValidDevices -------------------------------------------------------------------------------------
// The list of devices a program is willing to run on, in priority order (deprecated). Accepted, and nothing
// is done with it: the two RTX 3060s answer cudaSuccess for a valid list and for an empty one, and
// cudaErrorInvalidValue for a null list with a length, a negative length and a list with a device that does not exist.
static cudaError_t cudaSetValidDevices_traced(int* device_arr, int len) {
  return guard("cudaSetValidDevices", [&](State& s) -> cudaError_t {
    if (len < 0 || (len > 0 && !device_arr)) return cudaErrorInvalidValue;
    if (len > s.rt->device_count()) return cudaErrorInvalidValue;
    for (int i = 0; i < len; ++i)
      if (device_arr[i] < 0 || device_arr[i] >= s.rt->device_count()) return cudaErrorInvalidValue;
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaSetValidDevices(int* device_arr, int len) {
  return traced_call("cudaSetValidDevices", cudaSetValidDevices_traced, device_arr, len);
}

// ---- the widest linear texture --------------------------------------------------------------------------------
// The most elements a 1D texture bound to linear memory can have for a channel format, on a device. An
// RTX 3060 answers 2^30 for elements of 1, 2 and 4 bytes and 2^28 for 16 bytes: 4 GiB of linear memory over the
// element size, at most 2^30. A format with three channels, or none, is cudaErrorInvalidChannelDescriptor.
static cudaError_t cudaDeviceGetTexture1DLinearMaxWidth_traced(size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int device) {
  return guard("cudaDeviceGetTexture1DLinearMaxWidth", [&](State& s) -> cudaError_t {
    if (!maxWidthInElements || !fmtDesc) return cudaErrorInvalidValue;
    if (device < 0 || device >= s.rt->device_count()) return cudaErrorInvalidDevice;
    // One, two or four channels, each of the same width, filling the leading channels.
    const int bits[4] = {fmtDesc->x, fmtDesc->y, fmtDesc->z, fmtDesc->w};
    int channels = 0;
    while (channels < 4 && bits[channels] > 0) ++channels;
    for (int i = channels; i < 4; ++i)
      if (bits[i] != 0) return cudaErrorInvalidChannelDescriptor;
    if (channels != 1 && channels != 2 && channels != 4) return cudaErrorInvalidChannelDescriptor;
    for (int i = 1; i < channels; ++i)
      if (bits[i] != bits[0]) return cudaErrorInvalidChannelDescriptor;
    if (bits[0] != 8 && bits[0] != 16 && bits[0] != 32) return cudaErrorInvalidChannelDescriptor;
    const uint64_t element = static_cast<uint64_t>(channels) * static_cast<uint64_t>(bits[0]) / 8;
    *maxWidthInElements = static_cast<size_t>(std::min<uint64_t>(uint64_t{1} << 30, (uint64_t{1} << 32) / element));
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int device) {
  return traced_call("cudaDeviceGetTexture1DLinearMaxWidth", cudaDeviceGetTexture1DLinearMaxWidth_traced, maxWidthInElements, fmtDesc, device);
}

// ---- dynamic shared memory per block, for a given occupancy ----------------------------------------------------
// How much dynamic shared memory each of `numBlocks` resident blocks of `blockSize` threads may ask for: the
// SM's shared memory over the blocks in 128-byte units, less the driver's reserved shared memory and the kernel's
// static shared memory, at most 48 KiB (the default per-block limit) less the static part. Measured on an RTX
// 3060 against kernels with 0 and 256 bytes of static shared memory: 1 block 49152 and 48896; 8 blocks of
// 128 threads 11776 and 11520; 12 blocks 7168; 16 blocks of 64 threads 5376 and 5120. A launch that cannot be
// that resident (more blocks than the SM holds, more threads than it holds, a block past the limit) and a
// count or size below 1 are cudaErrorInvalidValue.
static cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock_traced(size_t* dynamicSmemSize, const void* func, int numBlocks, int blockSize) {
  return guard("cudaOccupancyAvailableDynamicSMemPerBlock", [&](State& s) -> cudaError_t {
    if (!dynamicSmemSize) return cudaErrorInvalidValue;
    auto it = s.kernels.find(func);
    if (it == s.kernels.end()) {
      if (!quiet() && func)
        std::fprintf(stderr, "[vgpu] cudaOccupancyAvailableDynamicSMemPerBlock: unregistered kernel stub %p\n", func);
      return cudaErrorInvalidDeviceFunction;
    }
    KernelInfo& ki = it->second;
    if (!ki.mod || !registered_ptx(s, *ki.mod)) return cudaErrorInvalidDeviceFunction;
    if (numBlocks <= 0 || blockSize <= 0) return cudaErrorInvalidValue;
    vgpu::runtime::Device& dev = current(s);
    const vgpu::DeviceProfile& p = dev.profile();
    const vgpu::ptx::EntryFn* fn = dev.get_function(module_on_current(s, *ki.mod), ki.entry_name);
    if (static_cast<uint32_t>(blockSize) > p.limits.max_threads_per_block) return cudaErrorInvalidValue;
    // Resident, with no dynamic shared memory: the occupancy rules of the launch path.
    const auto res = vgpu::exec::kernel_resources(*fn, p, static_cast<uint32_t>(blockSize), 0);
    if (static_cast<uint32_t>(numBlocks) > res.occupancy.blocks_per_sm) return cudaErrorInvalidValue;
    const uint64_t granule = p.cc_major >= 8 ? 128 : 256;
    const uint64_t reserved = p.reserved_smem_per_block();
    uint64_t share = p.limits.shared_mem_per_sm / static_cast<uint64_t>(numBlocks) / granule * granule;
    share = share > reserved ? share - reserved : 0;
    const uint64_t fixed = fn->static_shared_size;
    // At most what the kernel may ask for: 48 KiB less its static shared memory until it opts in to more
    // (cudaFuncSetAttribute), then the figure it set (measured: 60000 for a kernel that asked for 60000 with
    // one block resident, 101376 after asking for the most the part allows).
    const auto at = ki.attrs.find(t_current_device);
    const vgpu_funcattr::State st = at == ki.attrs.end() ? vgpu_funcattr::State{} : at->second;
    *dynamicSmemSize = static_cast<size_t>(std::min<uint64_t>(share > fixed ? share - fixed : 0,
                                                              static_cast<uint64_t>(vgpu_funcattr::max_dynamic_shared(st, p, fixed))));
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* func, int numBlocks, int blockSize) {
  return traced_call("cudaOccupancyAvailableDynamicSMemPerBlock", cudaOccupancyAvailableDynamicSMemPerBlock_traced, dynamicSmemSize, func, numBlocks, blockSize);
}

// ---- NvSci, the export table ---------------------------------------------------------------------------------------
// NvSciSync is for Tegra; a PCIe GPU has no attribute list to hand out. An RTX 3060 answers
// cudaErrorInvalidValue for an attribute list it is given, cudaErrorInvalidResourceHandle for none and
// cudaErrorInvalidDevice for a device that does not exist; this keeps the last two and refuses the first by name.
static cudaError_t cudaDeviceGetNvSciSyncAttributes_traced(void* nvSciSyncAttrList, int device, int flags) {
  return guard("cudaDeviceGetNvSciSyncAttributes", [&](State& s) -> cudaError_t {
    (void)flags;
    if (device < 0 || device >= s.rt->device_count()) return cudaErrorInvalidDevice;
    if (!nvSciSyncAttrList) return cudaErrorInvalidResourceHandle;
    return refuse("cudaDeviceGetNvSciSyncAttributes", "NvSciSync is for Tegra; this device has none", cudaErrorInvalidValue);
  });
}

VGPU_EXPORT cudaError_t cudaDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, int device, int flags) {
  return traced_call("cudaDeviceGetNvSciSyncAttributes", cudaDeviceGetNvSciSyncAttributes_traced, nvSciSyncAttrList, device, flags);
}

// The export table is NVIDIA's private interface between its libraries, looked up by UUID; none is offered. An
// RTX 3060 answers cudaErrorInvalidValue for a UUID it does not know, which is every UUID a program can make up.
static cudaError_t cudaGetExportTable_traced(const void** ppExportTable, const cudaUUID_t* pExportTableId) {
  (void)ppExportTable;
  (void)pExportTableId;
  return cudaErrorInvalidValue;
}

VGPU_EXPORT cudaError_t cudaGetExportTable(const void** ppExportTable, const cudaUUID_t* pExportTableId) {
  return traced_call("cudaGetExportTable", cudaGetExportTable_traced, ppExportTable, pExportTableId);
}

// ---- a kernel's function handle -----------------------------------------------------------------------------------
// The driver's CUfunction for a kernel this runtime registered, so the program can hand it to the driver API
// (cuFuncGetAttribute, cuLaunchKernel, cuFuncGetName). It is made by the driver shim from the module the runtime
// loaded (vgpu_driver_function_for_v1), and the same kernel gives the same handle. A symbol that is not a
// kernel is cudaErrorInvalidDeviceFunction and leaves the result alone; a null result pointer is cudaSuccess, as
// on an RTX 3060.
static cudaError_t cudaGetFuncBySymbol_traced(cudaFunction_t* functionPtr, const void* symbolPtr) {
  return guard("cudaGetFuncBySymbol", [&](State& s) -> cudaError_t {
    auto it = s.kernels.find(symbolPtr);
    if (it == s.kernels.end()) return cudaErrorInvalidDeviceFunction;
    KernelInfo& ki = it->second;
    if (!ki.mod || !registered_ptx(s, *ki.mod)) return cudaErrorInvalidDeviceFunction;
    if (!functionPtr) return cudaSuccess;
    using Bind = int (*)(int, uint64_t, const char*, void**, void**);
    static const Bind fn = reinterpret_cast<Bind>(dlsym(RTLD_DEFAULT, "vgpu_driver_function_for_v1"));
    if (!fn) return refuse("cudaGetFuncBySymbol", "the driver shim (libcuda) is not loaded, and a CUfunction is its handle");
    void* function = nullptr;
    void* module = nullptr;
    const uint64_t mid = module_on_current(s, *ki.mod);
    if (const int rc = fn(t_current_device, mid, ki.entry_name.c_str(), &function, &module); rc != 0)
      return static_cast<cudaError_t>(rc);
    *functionPtr = reinterpret_cast<cudaFunction_t>(function);
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaGetFuncBySymbol(cudaFunction_t* functionPtr, const void* symbolPtr) {
  return traced_call("cudaGetFuncBySymbol", cudaGetFuncBySymbol_traced, functionPtr, symbolPtr);
}

// CUDA 12.1 and later: the kernel handle (cudaKernel_t) of a registered kernel. The handle here is the stub's
// address, which every runtime call that takes a kernel (launches, attributes, occupancy) accepts in place of
// it, as the real handle is; the driver's cuKernel* calls do not know it.
#if CUDART_VERSION >= 12010
static cudaError_t cudaGetKernel_traced(cudaKernel_t* kernelPtr, const void* entryFuncAddr) {
  return guard("cudaGetKernel", [&](State& s) -> cudaError_t {
    if (!kernelPtr) return cudaErrorInvalidValue;
    if (!s.kernels.count(entryFuncAddr)) return cudaErrorInvalidDeviceFunction;
    *kernelPtr = reinterpret_cast<cudaKernel_t>(const_cast<void*>(entryFuncAddr));
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaGetKernel(cudaKernel_t* kernelPtr, const void* entryFuncAddr) {
  return traced_call("cudaGetKernel", cudaGetKernel_traced, kernelPtr, entryFuncAddr);
}
#endif

// ---- atomics over a link ------------------------------------------------------------------------------------------------
// Which atomic operations the host and the peers can do on this device's memory, as a bit mask per operation
// (cudaAtomicOperationCapability). The host: an RTX 3060 under WSL answers 0 for every operation, which is
// what cudaDevAttrHostNativeAtomicSupported (0, here and there) says. A peer: 0 for every operation without a
// peer path (measured, cudaErrorInvalidDevice for a device with itself); over one, derived from the
// documentation and not checked against a card (the machine has no such pair): the integer operations are
// signed, unsigned, reduction, 32- and 64-bit; exchange and compare-and-swap return a value, so they are not
// reductions; the floating-point ones are 32- and 64-bit reductions.
#if CUDART_VERSION >= 13000
namespace {
cudaError_t atomic_caps(const char* api, unsigned* caps, const cudaAtomicOperation* ops, unsigned count, bool path) {
  if (!caps || !ops || count == 0) return cudaErrorInvalidValue;
  for (unsigned i = 0; i < count; ++i) {
    const int op = enum_value(ops[i]);
    if (op < 0 || op > 12) return cudaErrorInvalidValue;
    unsigned c = 0;
    if (path) {
      const unsigned scalars = cudaAtomicCapabilityScalar32 | cudaAtomicCapabilityScalar64;
      if (op <= cudaAtomicOperationXOR) c = cudaAtomicCapabilitySigned | cudaAtomicCapabilityUnsigned | cudaAtomicCapabilityReduction | scalars;
      else if (op <= cudaAtomicOperationCAS) c = cudaAtomicCapabilitySigned | cudaAtomicCapabilityUnsigned | scalars;
      else c = cudaAtomicCapabilityReduction | scalars;
    }
    caps[i] = c;
  }
  (void)api;
  return cudaSuccess;
}
}  // namespace

static cudaError_t cudaDeviceGetHostAtomicCapabilities_traced(unsigned int* capabilities, const cudaAtomicOperation* operations, unsigned int count, int device) {
  return guard("cudaDeviceGetHostAtomicCapabilities", [&](State& s) -> cudaError_t {
    if (!capabilities || !operations || count == 0) return cudaErrorInvalidValue;
    if (device < 0 || device >= s.rt->device_count()) return cudaErrorInvalidDevice;
    return atomic_caps("cudaDeviceGetHostAtomicCapabilities", capabilities, operations, count, false);
  });
}

VGPU_EXPORT cudaError_t cudaDeviceGetHostAtomicCapabilities(unsigned int* capabilities, const cudaAtomicOperation* operations, unsigned int count, int device) {
  return traced_call("cudaDeviceGetHostAtomicCapabilities", cudaDeviceGetHostAtomicCapabilities_traced, capabilities, operations, count, device);
}

static cudaError_t cudaDeviceGetP2PAtomicCapabilities_traced(unsigned int* capabilities, const cudaAtomicOperation* operations, unsigned int count, int srcDevice, int dstDevice) {
  return guard("cudaDeviceGetP2PAtomicCapabilities", [&](State& s) -> cudaError_t {
    const int n = s.rt->device_count();
    if (!capabilities || !operations || count == 0) return cudaErrorInvalidValue;
    if (srcDevice < 0 || srcDevice >= n || dstDevice < 0 || dstDevice >= n || srcDevice == dstDevice) return cudaErrorInvalidDevice;
    return atomic_caps("cudaDeviceGetP2PAtomicCapabilities", capabilities, operations, count, peer_path(s, srcDevice, dstDevice));
  });
}

VGPU_EXPORT cudaError_t cudaDeviceGetP2PAtomicCapabilities(unsigned int* capabilities, const cudaAtomicOperation* operations, unsigned int count, int srcDevice, int dstDevice) {
  return traced_call("cudaDeviceGetP2PAtomicCapabilities", cudaDeviceGetP2PAtomicCapabilities_traced, capabilities, operations, count, srcDevice, dstDevice);
}
#endif

// ---- CUDA arrays: what a program can ask of one that is not sparse or deferred-mapped ---------------------------------
// Memory requirements and sparse properties exist for arrays made sparse (cudaArraySparse) or with deferred
// mapping; any other array, which is every array this simulator makes, is cudaErrorInvalidValue (measured), and
// a handle that is not an array cudaErrorInvalidResourceHandle. cudaArrayGetPlane is for multi-planar
// arrays; a single-planar one is cudaErrorInvalidValue. The null-pointer cases come before the handle.
static cudaError_t cudaArrayGetMemoryRequirements_traced(cudaArrayMemoryRequirements* memoryRequirements, cudaArray_t array, int device) {
  return guard("cudaArrayGetMemoryRequirements", [&](State& s) -> cudaError_t {
    if (!memoryRequirements) return cudaErrorInvalidValue;
    if (!g_arrays.count(reinterpret_cast<uint64_t>(array))) return cudaErrorInvalidResourceHandle;
    (void)device;
    (void)s;
    return cudaErrorInvalidValue;   // not a sparse or deferred-mapped array
  });
}

VGPU_EXPORT cudaError_t cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaArray_t array, int device) {
  return traced_call("cudaArrayGetMemoryRequirements", cudaArrayGetMemoryRequirements_traced, memoryRequirements, array, device);
}

static cudaError_t cudaArrayGetSparseProperties_traced(cudaArraySparseProperties* sparseProperties, cudaArray_t array) {
  return guard("cudaArrayGetSparseProperties", [&](State&) -> cudaError_t {
    if (!sparseProperties) return cudaErrorInvalidValue;
    if (!g_arrays.count(reinterpret_cast<uint64_t>(array))) return cudaErrorInvalidResourceHandle;
    return cudaErrorInvalidValue;
  });
}

VGPU_EXPORT cudaError_t cudaArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaArray_t array) {
  return traced_call("cudaArrayGetSparseProperties", cudaArrayGetSparseProperties_traced, sparseProperties, array);
}

static cudaError_t cudaArrayGetPlane_traced(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx) {
  return guard("cudaArrayGetPlane", [&](State&) -> cudaError_t {
    (void)planeIdx;
    if (!pPlaneArray) return cudaErrorInvalidValue;
    if (!g_arrays.count(reinterpret_cast<uint64_t>(hArray))) return cudaErrorInvalidResourceHandle;
    return cudaErrorInvalidValue;   // no array here has planes
  });
}

VGPU_EXPORT cudaError_t cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx) {
  return traced_call("cudaArrayGetPlane", cudaArrayGetPlane_traced, pPlaneArray, hArray, planeIdx);
}

static cudaError_t cudaMipmappedArrayGetMemoryRequirements_traced(cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device) {
  return guard("cudaMipmappedArrayGetMemoryRequirements", [&](State&) -> cudaError_t {
    (void)device;
    if (!memoryRequirements) return cudaErrorInvalidValue;
    if (!mipmap || !g_mipmapped.count(reinterpret_cast<uint64_t>(mipmap))) return cudaErrorInvalidResourceHandle;
    return cudaErrorInvalidValue;
  });
}

VGPU_EXPORT cudaError_t cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device) {
  return traced_call("cudaMipmappedArrayGetMemoryRequirements", cudaMipmappedArrayGetMemoryRequirements_traced, memoryRequirements, mipmap, device);
}

static cudaError_t cudaMipmappedArrayGetSparseProperties_traced(cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap) {
  return guard("cudaMipmappedArrayGetSparseProperties", [&](State&) -> cudaError_t {
    if (!sparseProperties) return cudaErrorInvalidValue;
    if (!mipmap || !g_mipmapped.count(reinterpret_cast<uint64_t>(mipmap))) return cudaErrorInvalidResourceHandle;
    return cudaErrorInvalidValue;
  });
}

VGPU_EXPORT cudaError_t cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap) {
  return traced_call("cudaMipmappedArrayGetSparseProperties", cudaMipmappedArrayGetSparseProperties_traced, sparseProperties, mipmap);
}

// ---- graphics interop -----------------------------------------------------------------------------------------------------
// OpenGL, Direct3D, EGL and VDPAU resources cannot be registered (the cudaGraphics*Register* calls are in
// cuda_gl_interop.h and the like, which this does not export), so no cudaGraphicsResource_t is ever valid: every
// call on one is cudaErrorInvalidResourceHandle. (An RTX 3060 crashes on a garbage handle and was not
// measured on a null one; both are refused the same way.)
static cudaError_t cudaGraphicsUnregisterResource_traced(cudaGraphicsResource_t resource) {
  return guard("cudaGraphicsUnregisterResource", [&](State&) -> cudaError_t { (void)resource; return cudaErrorInvalidResourceHandle; });
}

VGPU_EXPORT cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) {
  return traced_call("cudaGraphicsUnregisterResource", cudaGraphicsUnregisterResource_traced, resource);
}

static cudaError_t cudaGraphicsResourceSetMapFlags_traced(cudaGraphicsResource_t resource, unsigned int flags) {
  return guard("cudaGraphicsResourceSetMapFlags", [&](State&) -> cudaError_t { (void)resource; (void)flags; return cudaErrorInvalidResourceHandle; });
}

VGPU_EXPORT cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags) {
  return traced_call("cudaGraphicsResourceSetMapFlags", cudaGraphicsResourceSetMapFlags_traced, resource, flags);
}

static cudaError_t cudaGraphicsMapResources_traced(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) {
  return guard("cudaGraphicsMapResources", [&](State&) -> cudaError_t {
    (void)stream;
    if (count < 0 || (count > 0 && !resources)) return cudaErrorInvalidValue;
    return count == 0 ? cudaSuccess : cudaErrorInvalidResourceHandle;
  });
}

VGPU_EXPORT cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) {
  return traced_call("cudaGraphicsMapResources", cudaGraphicsMapResources_traced, count, resources, stream);
}

static cudaError_t cudaGraphicsUnmapResources_traced(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) {
  return guard("cudaGraphicsUnmapResources", [&](State&) -> cudaError_t {
    (void)stream;
    if (count < 0 || (count > 0 && !resources)) return cudaErrorInvalidValue;
    return count == 0 ? cudaSuccess : cudaErrorInvalidResourceHandle;
  });
}

VGPU_EXPORT cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) {
  return traced_call("cudaGraphicsUnmapResources", cudaGraphicsUnmapResources_traced, count, resources, stream);
}

static cudaError_t cudaGraphicsResourceGetMappedPointer_traced(void** devPtr, size_t* size, cudaGraphicsResource_t resource) {
  return guard("cudaGraphicsResourceGetMappedPointer", [&](State&) -> cudaError_t {
    (void)size;
    (void)resource;
    if (!devPtr) return cudaErrorInvalidValue;
    return cudaErrorInvalidResourceHandle;
  });
}

VGPU_EXPORT cudaError_t cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource) {
  return traced_call("cudaGraphicsResourceGetMappedPointer", cudaGraphicsResourceGetMappedPointer_traced, devPtr, size, resource);
}

static cudaError_t cudaGraphicsSubResourceGetMappedArray_traced(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) {
  return guard("cudaGraphicsSubResourceGetMappedArray", [&](State&) -> cudaError_t {
    (void)resource;
    (void)arrayIndex;
    (void)mipLevel;
    if (!array) return cudaErrorInvalidValue;
    return cudaErrorInvalidResourceHandle;
  });
}

VGPU_EXPORT cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) {
  return traced_call("cudaGraphicsSubResourceGetMappedArray", cudaGraphicsSubResourceGetMappedArray_traced, array, resource, arrayIndex, mipLevel);
}

static cudaError_t cudaGraphicsResourceGetMappedMipmappedArray_traced(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource) {
  return guard("cudaGraphicsResourceGetMappedMipmappedArray", [&](State&) -> cudaError_t {
    (void)resource;
    if (!mipmappedArray) return cudaErrorInvalidValue;
    return cudaErrorInvalidResourceHandle;
  });
}

VGPU_EXPORT cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource) {
  return traced_call("cudaGraphicsResourceGetMappedMipmappedArray", cudaGraphicsResourceGetMappedMipmappedArray_traced, mipmappedArray, resource);
}

// ---- memory pools by location -----------------------------------------------------------------------------------
// CUDA 13's way to ask for a device's pools by (location, allocation type) rather than by device number:
// cudaMemGetDefaultMemPool is the pool the device was born with (cudaDeviceGetDefaultMemPool's), cudaMemGetMemPool
// the one cudaMallocAsync draws from now (cudaDeviceGetMemPool's), and cudaMemSetMemPool changes that
// (cudaDeviceSetMemPool's). Measured on an RTX 3060: an allocation type other than pinned is
// cudaErrorInvalidValue, except managed, which is cudaErrorNotSupported; a device location that does not exist,
// an invalid location and a null pointer are cudaErrorInvalidValue; a host or host-NUMA location has a pool of its
// own there, which is not modelled here (cudaErrorNotSupported, once said).
#if CUDART_VERSION >= 13000
namespace {
cudaError_t pool_device_of(State& s, const char* api, const cudaMemLocation* location, int type, int* device) {
  if (!location) return cudaErrorInvalidValue;
  const int loc_type = enum_value(location->type);
  const int alloc_type = type;
  if (alloc_type == cudaMemAllocationTypeManaged) return cudaErrorNotSupported;
  if (alloc_type != cudaMemAllocationTypePinned) return cudaErrorInvalidValue;
  if (loc_type == cudaMemLocationTypeHost || loc_type == cudaMemLocationTypeHostNuma)
    return refuse(api, "memory pools in host memory are not modelled");
  if (loc_type != cudaMemLocationTypeDevice) return cudaErrorInvalidValue;
  if (location->id < 0 || location->id >= s.rt->device_count()) return cudaErrorInvalidValue;
  *device = location->id;
  return cudaSuccess;
}
}  // namespace

static cudaError_t cudaMemGetDefaultMemPool_traced(cudaMemPool_t* memPool, cudaMemLocation* location, int type) {
  return guard("cudaMemGetDefaultMemPool", [&](State& s) -> cudaError_t {
    if (!memPool) return cudaErrorInvalidValue;
    int device = 0;
    if (const cudaError_t e = pool_device_of(s, "cudaMemGetDefaultMemPool", location, type, &device); e != cudaSuccess) return e;
    *memPool = reinterpret_cast<cudaMemPool_t>(&default_pool_for(s, device));
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaMemGetDefaultMemPool(cudaMemPool_t* memPool, cudaMemLocation* location, cudaMemAllocationType type) {
  return traced_call("cudaMemGetDefaultMemPool", cudaMemGetDefaultMemPool_traced, memPool, location, enum_value(type));
}

static cudaError_t cudaMemGetMemPool_traced(cudaMemPool_t* memPool, cudaMemLocation* location, int type) {
  return guard("cudaMemGetMemPool", [&](State& s) -> cudaError_t {
    if (!memPool) return cudaErrorInvalidValue;
    int device = 0;
    if (const cudaError_t e = pool_device_of(s, "cudaMemGetMemPool", location, type, &device); e != cudaSuccess) return e;
    *memPool = reinterpret_cast<cudaMemPool_t>(&pool_for(s, device));
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaMemGetMemPool(cudaMemPool_t* memPool, cudaMemLocation* location, cudaMemAllocationType type) {
  return traced_call("cudaMemGetMemPool", cudaMemGetMemPool_traced, memPool, location, enum_value(type));
}

static cudaError_t cudaMemSetMemPool_traced(cudaMemLocation* location, int type, cudaMemPool_t memPool) {
  if (const cudaError_t unsafe_ = capture_unsafe_gate("cudaMemSetMemPool")) return unsafe_;
  return guard("cudaMemSetMemPool", [&](State& s) -> cudaError_t {
    int device = 0;
    if (const cudaError_t e = pool_device_of(s, "cudaMemSetMemPool", location, type, &device); e != cudaSuccess) return e;
    MemPool* p = from_handle(s, memPool);
    if (!p || p->device != device) return cudaErrorInvalidValue;
    s.current_pool[device] = p;
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaMemSetMemPool(cudaMemLocation* location, cudaMemAllocationType type, cudaMemPool_t memPool) {
  return traced_call("cudaMemSetMemPool", cudaMemSetMemPool_traced, location, enum_value(type), memPool);
}
#endif

// ---- graph copies to and from a symbol ------------------------------------------------------------------------------
// The copy node of a __device__ variable, by the variable's name instead of its address: the symbol's address on
// the current device, plus the offset, at the time the node is made or changed. As cudaMemcpyToSymbol checks
// them, measured on an RTX 3060: a symbol that is not one is cudaErrorInvalidSymbol, a copy past its end
// cudaErrorInvalidValue, and the wrong direction (device to host into a symbol) cudaErrorInvalidMemcpyDirection;
// the default kind is accepted.
namespace {
cudaError_t symbol_copy_address(State& s, const void* symbol, size_t count, size_t offset, int kind, bool to_symbol, void** address) {
  uint64_t addr = 0;
  size_t size = 0;
  if (const cudaError_t e = symbol_address(s, symbol, &addr, &size); e != cudaSuccess) return e;
  if (offset > size || count > size - offset) return cudaErrorInvalidValue;
  const bool ok = kind == cudaMemcpyDeviceToDevice || kind == cudaMemcpyDefault ||
                  kind == (to_symbol ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost);
  if (!ok) return cudaErrorInvalidMemcpyDirection;
  *address = reinterpret_cast<void*>(addr + offset);
  return cudaSuccess;
}
}  // namespace

static cudaError_t cudaGraphAddMemcpyNodeToSymbol_traced(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const void* symbol, const void* src, size_t count, size_t offset, int kind) {
  return guard("cudaGraphAddMemcpyNodeToSymbol", [&](State& s) -> cudaError_t {
    void* dst = nullptr;
    if (!pGraphNode) return cudaErrorInvalidValue;
    if (const cudaError_t e = symbol_copy_address(s, symbol, count, offset, kind, true, &dst); e != cudaSuccess) return e;
    return cudaGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, static_cast<cudaMemcpyKind>(kind));
  });
}

VGPU_EXPORT cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind) {
  return traced_call("cudaGraphAddMemcpyNodeToSymbol", cudaGraphAddMemcpyNodeToSymbol_traced, pGraphNode, graph, pDependencies, numDependencies, symbol, src, count, offset, enum_value(kind));
}

static cudaError_t cudaGraphAddMemcpyNodeFromSymbol_traced(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* symbol, size_t count, size_t offset, int kind) {
  return guard("cudaGraphAddMemcpyNodeFromSymbol", [&](State& s) -> cudaError_t {
    void* src = nullptr;
    if (!pGraphNode) return cudaErrorInvalidValue;
    if (const cudaError_t e = symbol_copy_address(s, symbol, count, offset, kind, false, &src); e != cudaSuccess) return e;
    return cudaGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, static_cast<cudaMemcpyKind>(kind));
  });
}

VGPU_EXPORT cudaError_t cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind) {
  return traced_call("cudaGraphAddMemcpyNodeFromSymbol", cudaGraphAddMemcpyNodeFromSymbol_traced, pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count, offset, enum_value(kind));
}

static cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol_traced(cudaGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, int kind) {
  return guard("cudaGraphMemcpyNodeSetParamsToSymbol", [&](State& s) -> cudaError_t {
    void* dst = nullptr;
    if (const cudaError_t e = symbol_copy_address(s, symbol, count, offset, kind, true, &dst); e != cudaSuccess) return e;
    return cudaGraphMemcpyNodeSetParams1D(node, dst, src, count, static_cast<cudaMemcpyKind>(kind));
  });
}

VGPU_EXPORT cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind) {
  return traced_call("cudaGraphMemcpyNodeSetParamsToSymbol", cudaGraphMemcpyNodeSetParamsToSymbol_traced, node, symbol, src, count, offset, enum_value(kind));
}

static cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol_traced(cudaGraphNode_t node, void* dst, const void* symbol, size_t count, size_t offset, int kind) {
  return guard("cudaGraphMemcpyNodeSetParamsFromSymbol", [&](State& s) -> cudaError_t {
    void* src = nullptr;
    if (const cudaError_t e = symbol_copy_address(s, symbol, count, offset, kind, false, &src); e != cudaSuccess) return e;
    return cudaGraphMemcpyNodeSetParams1D(node, dst, src, count, static_cast<cudaMemcpyKind>(kind));
  });
}

VGPU_EXPORT cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node, void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind) {
  return traced_call("cudaGraphMemcpyNodeSetParamsFromSymbol", cudaGraphMemcpyNodeSetParamsFromSymbol_traced, node, dst, symbol, count, offset, enum_value(kind));
}

static cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol_traced(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, int kind) {
  return guard("cudaGraphExecMemcpyNodeSetParamsToSymbol", [&](State& s) -> cudaError_t {
    void* dst = nullptr;
    if (const cudaError_t e = symbol_copy_address(s, symbol, count, offset, kind, true, &dst); e != cudaSuccess) return e;
    return cudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, static_cast<cudaMemcpyKind>(kind));
  });
}

VGPU_EXPORT cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind) {
  return traced_call("cudaGraphExecMemcpyNodeSetParamsToSymbol", cudaGraphExecMemcpyNodeSetParamsToSymbol_traced, hGraphExec, node, symbol, src, count, offset, enum_value(kind));
}

static cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol_traced(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* symbol, size_t count, size_t offset, int kind) {
  return guard("cudaGraphExecMemcpyNodeSetParamsFromSymbol", [&](State& s) -> cudaError_t {
    void* src = nullptr;
    if (const cudaError_t e = symbol_copy_address(s, symbol, count, offset, kind, false, &src); e != cudaSuccess) return e;
    return cudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, static_cast<cudaMemcpyKind>(kind));
  });
}

VGPU_EXPORT cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind) {
  return traced_call("cudaGraphExecMemcpyNodeSetParamsFromSymbol", cudaGraphExecMemcpyNodeSetParamsFromSymbol_traced, hGraphExec, node, dst, symbol, count, offset, enum_value(kind));
}

// ---- one call to change any node (CUDA 12.2) --------------------------------------------------------------------------
// cudaGraphNodeSetParams changes a node of a graph and cudaGraphExecNodeSetParams the node of an instantiated
// graph, by the union of every node type's parameters. A kernel, copy, fill, host and event node goes to the
// call that node type already has; the rest -- allocation, free, empty, child graph, external semaphores,
// conditional -- are cudaErrorNotSupported here (once said). A null node, null parameters and a type that does
// not exist are cudaErrorInvalidValue, as on an RTX 3060.
#if CUDART_VERSION >= 12020
namespace {
cudaError_t node_params_dispatch(const char* api, cudaGraphExec_t exec, cudaGraphNode_t node, cudaGraphNodeParams* np) {
  if (!node || !np) return cudaErrorInvalidValue;
  const int type = enum_value(np->type);
  switch (type) {
    case cudaGraphNodeTypeKernel: {
      cudaKernelNodeParams k{};
      k.func = np->kernel.func;
      k.gridDim = np->kernel.gridDim;
      k.blockDim = np->kernel.blockDim;
      k.sharedMemBytes = np->kernel.sharedMemBytes;
      k.kernelParams = np->kernel.kernelParams;
      k.extra = np->kernel.extra;
      return exec ? cudaGraphExecKernelNodeSetParams(exec, node, &k) : cudaGraphKernelNodeSetParams(node, &k);
    }
    case cudaGraphNodeTypeMemcpy:
      if (np->memcpy.flags != 0) return cudaErrorInvalidValue;
      return exec ? cudaGraphExecMemcpyNodeSetParams(exec, node, &np->memcpy.copyParams)
                  : cudaGraphMemcpyNodeSetParams(node, &np->memcpy.copyParams);
    case cudaGraphNodeTypeMemset: {
      cudaMemsetParams m{};
      m.dst = np->memset.dst;
      m.pitch = np->memset.pitch;
      m.value = np->memset.value;
      m.elementSize = np->memset.elementSize;
      m.width = np->memset.width;
      m.height = np->memset.height;
      return exec ? cudaGraphExecMemsetNodeSetParams(exec, node, &m) : cudaGraphMemsetNodeSetParams(node, &m);
    }
    case cudaGraphNodeTypeHost: {
      cudaHostNodeParams h{};
      h.fn = np->host.fn;
      h.userData = np->host.userData;
      return exec ? cudaGraphExecHostNodeSetParams(exec, node, &h) : cudaGraphHostNodeSetParams(node, &h);
    }
    case cudaGraphNodeTypeEventRecord:
      return exec ? cudaGraphExecEventRecordNodeSetEvent(exec, node, np->eventRecord.event)
                  : cudaGraphEventRecordNodeSetEvent(node, np->eventRecord.event);
    case cudaGraphNodeTypeWaitEvent:
      return exec ? cudaGraphExecEventWaitNodeSetEvent(exec, node, np->eventWait.event)
                  : cudaGraphEventWaitNodeSetEvent(node, np->eventWait.event);
    case cudaGraphNodeTypeGraph: case cudaGraphNodeTypeEmpty: case cudaGraphNodeTypeExtSemaphoreSignal:
    case cudaGraphNodeTypeExtSemaphoreWait: case cudaGraphNodeTypeMemAlloc: case cudaGraphNodeTypeMemFree:
#if CUDART_VERSION >= 12030
    case cudaGraphNodeTypeConditional:
#endif
      return refuse(api, "only kernel, copy, fill, host and event nodes can be changed this way");
    default:
      return cudaErrorInvalidValue;
  }
}
}  // namespace

static cudaError_t cudaGraphNodeSetParams_traced(cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) {
  return node_params_dispatch("cudaGraphNodeSetParams", nullptr, node, nodeParams);
}

VGPU_EXPORT cudaError_t cudaGraphNodeSetParams(cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) {
  return traced_call("cudaGraphNodeSetParams", cudaGraphNodeSetParams_traced, node, nodeParams);
}

static cudaError_t cudaGraphExecNodeSetParams_traced(cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) {
  if (!graphExec) return cudaErrorInvalidValue;
  return node_params_dispatch("cudaGraphExecNodeSetParams", graphExec, node, nodeParams);
}

VGPU_EXPORT cudaError_t cudaGraphExecNodeSetParams(cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) {
  return traced_call("cudaGraphExecNodeSetParams", cudaGraphExecNodeSetParams_traced, graphExec, node, nodeParams);
}
#endif

// ---- the driver's log of errors (CUDA 13) -----------------------------------------------------------------------------
// cudaLogsCurrent / DumpToMemory / DumpToFile read the runtime's error log, and cudaLogsRegisterCallback
// subscribes to it. Nothing is logged here (an RTX 3060 logs nothing either unless asked with CUDA_LOG_FILE): the
// iterator is 0, a dump writes nothing (a file is created empty, memory gets a size of 0), and a callback
// is never called. Measured: a null iterator for the current position, a null buffer or size for a dump and a
// null path are cudaErrorInvalidValue, a null iterator for a dump is not (it dumps all); registering needs
// somewhere to put the handle, and unregistering a handle twice crashes the card, so it is refused here.
#if CUDART_VERSION >= 13000
namespace {
std::mutex g_log_mu;
std::set<uintptr_t> g_log_callbacks;
uintptr_t g_next_log_callback = 1;
}  // namespace

static cudaError_t cudaLogsRegisterCallback_traced(cudaLogsCallback_t callbackFunc, void* userData, cudaLogsCallbackHandle* callback_out) {
  (void)userData;
  if (!callbackFunc) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_log_mu);
  const uintptr_t h = g_next_log_callback++;
  g_log_callbacks.insert(h);
  if (callback_out) *callback_out = reinterpret_cast<cudaLogsCallbackHandle>(h);
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaLogsRegisterCallback(cudaLogsCallback_t callbackFunc, void* userData, cudaLogsCallbackHandle* callback_out) {
  return traced_call("cudaLogsRegisterCallback", cudaLogsRegisterCallback_traced, callbackFunc, userData, callback_out);
}

static cudaError_t cudaLogsUnregisterCallback_traced(cudaLogsCallbackHandle callback) {
  if (!callback) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_log_mu);
  return g_log_callbacks.erase(reinterpret_cast<uintptr_t>(callback)) ? cudaSuccess : cudaErrorInvalidValue;
}

VGPU_EXPORT cudaError_t cudaLogsUnregisterCallback(cudaLogsCallbackHandle callback) {
  return traced_call("cudaLogsUnregisterCallback", cudaLogsUnregisterCallback_traced, callback);
}

static cudaError_t cudaLogsCurrent_traced(cudaLogIterator* iterator_out, unsigned int flags) {
  (void)flags;
  if (!iterator_out) return cudaErrorInvalidValue;
  *iterator_out = 0;
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaLogsCurrent(cudaLogIterator* iterator_out, unsigned int flags) {
  return traced_call("cudaLogsCurrent", cudaLogsCurrent_traced, iterator_out, flags);
}

static cudaError_t cudaLogsDumpToFile_traced(cudaLogIterator* iterator, const char* pathToFile, unsigned int flags) {
  (void)flags;
  if (!pathToFile) return cudaErrorInvalidValue;
  if (std::FILE* f = std::fopen(pathToFile, "w")) {
    std::fclose(f);
    return cudaSuccess;
  }
  (void)iterator;
  return cudaErrorInvalidValue;
}

VGPU_EXPORT cudaError_t cudaLogsDumpToFile(cudaLogIterator* iterator, const char* pathToFile, unsigned int flags) {
  return traced_call("cudaLogsDumpToFile", cudaLogsDumpToFile_traced, iterator, pathToFile, flags);
}

static cudaError_t cudaLogsDumpToMemory_traced(cudaLogIterator* iterator, char* buffer, size_t* size, unsigned int flags) {
  (void)iterator;
  (void)flags;
  if (!buffer || !size) return cudaErrorInvalidValue;
  *size = 0;
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaLogsDumpToMemory(cudaLogIterator* iterator, char* buffer, size_t* size, unsigned int flags) {
  return traced_call("cudaLogsDumpToMemory", cudaLogsDumpToMemory_traced, iterator, buffer, size, flags);
}
#endif

// ---- notifications about the device (CUDA 12.4) -------------------------------------------------------------------------
// A callback told when the process goes over its device memory budget (cudaAsyncNotificationTypeOverBudget).
// Registering gives a handle and unregistering takes it back; no notification is ever sent, because nothing here
// is over a budget it does not have. A null callback, a null place for the handle and a device that does not exist
// are errors (the card crashes on the second; cudaErrorInvalidValue here); a handle unregistered twice is too.
#if CUDART_VERSION >= 12040
namespace {
std::mutex g_async_mu;
std::set<uintptr_t> g_async_callbacks;
uintptr_t g_next_async_callback = 1;
}  // namespace

static cudaError_t cudaDeviceRegisterAsyncNotification_traced(int device, cudaAsyncCallback callbackFunc, void* userData, cudaAsyncCallbackHandle_t* callback) {
  return guard("cudaDeviceRegisterAsyncNotification", [&](State& s) -> cudaError_t {
    (void)userData;
    if (device < 0 || device >= s.rt->device_count()) return cudaErrorInvalidDevice;
    if (!callbackFunc || !callback) return cudaErrorInvalidValue;
    std::lock_guard<std::mutex> lock(g_async_mu);
    const uintptr_t h = g_next_async_callback++;
    g_async_callbacks.insert(h);
    *callback = reinterpret_cast<cudaAsyncCallbackHandle_t>(h);
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaDeviceRegisterAsyncNotification(int device, cudaAsyncCallback callbackFunc, void* userData, cudaAsyncCallbackHandle_t* callback) {
  return traced_call("cudaDeviceRegisterAsyncNotification", cudaDeviceRegisterAsyncNotification_traced, device, callbackFunc, userData, callback);
}

static cudaError_t cudaDeviceUnregisterAsyncNotification_traced(int device, cudaAsyncCallbackHandle_t callback) {
  return guard("cudaDeviceUnregisterAsyncNotification", [&](State& s) -> cudaError_t {
    if (device < 0 || device >= s.rt->device_count()) return cudaErrorInvalidDevice;
    if (!callback) return cudaErrorInvalidValue;
    std::lock_guard<std::mutex> lock(g_async_mu);
    return g_async_callbacks.erase(reinterpret_cast<uintptr_t>(callback)) ? cudaSuccess : cudaErrorInvalidValue;
  });
}

VGPU_EXPORT cudaError_t cudaDeviceUnregisterAsyncNotification(int device, cudaAsyncCallbackHandle_t callback) {
  return traced_call("cudaDeviceUnregisterAsyncNotification", cudaDeviceUnregisterAsyncNotification_traced, device, callback);
}
#endif

// ---- libraries: the rest of the driver's (CUDA 12.8) ---------------------------------------------------------------------
// The runtime's library API is the driver's under the runtime's names (see the first of these in
// runtime_api.cpp): one simulated machine behind both. These forward to the driver's cuLibrary* functions of the
// same job, so the answers are the driver's, which are the card's (measured through the runtime).
#if CUDART_VERSION >= 12080
static cudaError_t cudaLibraryLoadFromFile_traced(cudaLibrary_t* library, const char* fileName, cudaJitOption* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) {
  using Fn = int (*)(void**, const char*, void*, void**, unsigned, void*, void**, unsigned);
  auto fn = reinterpret_cast<Fn>(driver_symbol("cuLibraryLoadFromFile"));
  if (!fn) return cudaErrorInitializationError;
  return guard("cudaLibraryLoadFromFile", [&](State&) -> cudaError_t {
    return from_driver(fn(reinterpret_cast<void**>(library), fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions));
  });
}

VGPU_EXPORT cudaError_t cudaLibraryLoadFromFile(cudaLibrary_t* library, const char* fileName, cudaJitOption* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) {
  return traced_call("cudaLibraryLoadFromFile", cudaLibraryLoadFromFile_traced, library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);
}

static cudaError_t cudaLibraryGetKernelCount_traced(unsigned int* count, cudaLibrary_t lib) {
  using Fn = int (*)(unsigned*, void*);
  auto fn = reinterpret_cast<Fn>(driver_symbol("cuLibraryGetKernelCount"));
  if (!fn) return cudaErrorInitializationError;
  return guard("cudaLibraryGetKernelCount", [&](State&) -> cudaError_t { return from_driver(fn(count, lib)); });
}

VGPU_EXPORT cudaError_t cudaLibraryGetKernelCount(unsigned int* count, cudaLibrary_t lib) {
  return traced_call("cudaLibraryGetKernelCount", cudaLibraryGetKernelCount_traced, count, lib);
}

static cudaError_t cudaLibraryEnumerateKernels_traced(cudaKernel_t* kernels, unsigned int numKernels, cudaLibrary_t lib) {
  using Fn = int (*)(void**, unsigned, void*);
  auto fn = reinterpret_cast<Fn>(driver_symbol("cuLibraryEnumerateKernels"));
  if (!fn) return cudaErrorInitializationError;
  return guard("cudaLibraryEnumerateKernels", [&](State&) -> cudaError_t { return from_driver(fn(reinterpret_cast<void**>(kernels), numKernels, lib)); });
}

VGPU_EXPORT cudaError_t cudaLibraryEnumerateKernels(cudaKernel_t* kernels, unsigned int numKernels, cudaLibrary_t lib) {
  return traced_call("cudaLibraryEnumerateKernels", cudaLibraryEnumerateKernels_traced, kernels, numKernels, lib);
}

static cudaError_t cudaLibraryGetGlobal_traced(void** dptr, size_t* bytes, cudaLibrary_t library, const char* name) {
  using Fn = int (*)(unsigned long long*, size_t*, void*, const char*);
  auto fn = reinterpret_cast<Fn>(driver_symbol("cuLibraryGetGlobal"));
  if (!fn) return cudaErrorInitializationError;
  return guard("cudaLibraryGetGlobal", [&](State&) -> cudaError_t {
    unsigned long long p = 0;
    const int rc = fn(&p, bytes, library, name);
    if (rc == 0 && dptr) *dptr = reinterpret_cast<void*>(p);
    return from_driver(rc);
  });
}

VGPU_EXPORT cudaError_t cudaLibraryGetGlobal(void** dptr, size_t* bytes, cudaLibrary_t library, const char* name) {
  return traced_call("cudaLibraryGetGlobal", cudaLibraryGetGlobal_traced, dptr, bytes, library, name);
}

static cudaError_t cudaLibraryGetManaged_traced(void** dptr, size_t* bytes, cudaLibrary_t library, const char* name) {
  using Fn = int (*)(unsigned long long*, size_t*, void*, const char*);
  auto fn = reinterpret_cast<Fn>(driver_symbol("cuLibraryGetManaged"));
  if (!fn) return cudaErrorInitializationError;
  return guard("cudaLibraryGetManaged", [&](State&) -> cudaError_t {
    unsigned long long p = 0;
    const int rc = fn(&p, bytes, library, name);
    if (rc == 0 && dptr) *dptr = reinterpret_cast<void*>(p);
    return from_driver(rc);
  });
}

VGPU_EXPORT cudaError_t cudaLibraryGetManaged(void** dptr, size_t* bytes, cudaLibrary_t library, const char* name) {
  return traced_call("cudaLibraryGetManaged", cudaLibraryGetManaged_traced, dptr, bytes, library, name);
}

static cudaError_t cudaLibraryGetUnifiedFunction_traced(void** fptr, cudaLibrary_t library, const char* symbol) {
  using Fn = int (*)(void**, void*, const char*);
  auto fn = reinterpret_cast<Fn>(driver_symbol("cuLibraryGetUnifiedFunction"));
  if (!fn) return cudaErrorInitializationError;
  return guard("cudaLibraryGetUnifiedFunction", [&](State&) -> cudaError_t { return from_driver(fn(fptr, library, symbol)); });
}

VGPU_EXPORT cudaError_t cudaLibraryGetUnifiedFunction(void** fptr, cudaLibrary_t library, const char* symbol) {
  return traced_call("cudaLibraryGetUnifiedFunction", cudaLibraryGetUnifiedFunction_traced, fptr, library, symbol);
}
#endif

// ---- copies and prefetches in batches (CUDA 12.8 and 13) -----------------------------------------------------------------
// cudaMemcpyBatchAsync is `count` copies on one stream, with attributes for runs of them (attrsIdxs gives the
// first copy of each run, from 0, increasing). Measured on an RTX 3060: a count, an attribute count or a size
// of 0 and a null array are cudaErrorInvalidValue, as is an attribute list that does not start at copy 0 or an
// access order that is invalid (0 or past 3) or flags other than 0 and 1; each copy is then a cudaMemcpyAsync
// of kind default. CUDA 12.8 and 12.9 take a `failIdx` too, which a failed batch sets to the copy it stopped at.
#if CUDART_VERSION >= 12080
namespace {
cudaError_t memcpy_batch(const char* api, void** dsts, void** srcs, const size_t* sizes, size_t count, const cudaMemcpyAttributes* attrs,
                         const size_t* attrsIdxs, size_t numAttrs, size_t* failIdx, cudaStream_t stream) {
  if (failIdx) *failIdx = static_cast<size_t>(-1);
  // The legacy default stream is refused: an RTX 3060 answers cudaErrorInvalidValue for stream 0 and takes any stream
  // made with cudaStreamCreate.
  if (!stream) return cudaErrorInvalidValue;
  if (!count || !dsts || !srcs || !sizes || !attrs || !attrsIdxs || !numAttrs) return cudaErrorInvalidValue;
  if (attrsIdxs[0] != 0) return cudaErrorInvalidValue;
  for (size_t i = 0; i < numAttrs; ++i) {
    if (attrsIdxs[i] >= count || (i && attrsIdxs[i] <= attrsIdxs[i - 1])) return cudaErrorInvalidValue;
    const int order = enum_value(attrs[i].srcAccessOrder);
    if (order < cudaMemcpySrcAccessOrderStream || order > cudaMemcpySrcAccessOrderAny) return cudaErrorInvalidValue;
    if (attrs[i].flags & ~1u) return cudaErrorInvalidValue;
  }
  for (size_t i = 0; i < count; ++i)
    if (!sizes[i]) return cudaErrorInvalidValue;
  for (size_t i = 0; i < count; ++i) {
    if (const cudaError_t e = cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyDefault, stream); e != cudaSuccess) {
      if (failIdx) *failIdx = i;
      return e;
    }
  }
  (void)api;
  return cudaSuccess;
}
}  // namespace
#endif

#if CUDART_VERSION >= 13000
static cudaError_t cudaMemcpyBatchAsync_traced(void* const* dsts, const void* const* srcs, const size_t* sizes, size_t count, cudaMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, cudaStream_t stream) {
  return memcpy_batch("cudaMemcpyBatchAsync", const_cast<void**>(dsts), const_cast<void**>(srcs), sizes, count, attrs, attrsIdxs, numAttrs, nullptr, stream);
}

VGPU_EXPORT cudaError_t cudaMemcpyBatchAsync(void* const* dsts, const void* const* srcs, const size_t* sizes, size_t count, cudaMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, cudaStream_t stream) {
  return traced_call("cudaMemcpyBatchAsync", cudaMemcpyBatchAsync_traced, dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, stream);
}
#elif CUDART_VERSION >= 12080
static cudaError_t cudaMemcpyBatchAsync_traced(void** dsts, void** srcs, size_t* sizes, size_t count, cudaMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, size_t* failIdx, cudaStream_t stream) {
  return memcpy_batch("cudaMemcpyBatchAsync", dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, failIdx, stream);
}

VGPU_EXPORT cudaError_t cudaMemcpyBatchAsync(void** dsts, void** srcs, size_t* sizes, size_t count, cudaMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, size_t* failIdx, cudaStream_t stream) {
  return traced_call("cudaMemcpyBatchAsync", cudaMemcpyBatchAsync_traced, dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, failIdx, stream);
}
#endif

// cudaMemcpy3DBatchAsync: the same for boxes. Each operand is a pointer (with the length of a row and the
// height of a layer, in elements, 0 for the box's own) or an array (with an offset); each copy goes to
// cudaMemcpy3DAsync. Flags other than 0, a count of 0, a null list, an extent with a dimension of 0, an
// operand type that does not exist and an invalid access order are cudaErrorInvalidValue.
#if CUDART_VERSION >= 12080
namespace {
cudaError_t memcpy3d_batch(size_t numOps, cudaMemcpy3DBatchOp* opList, size_t* failIdx, unsigned long long flags, cudaStream_t stream) {
  if (failIdx) *failIdx = static_cast<size_t>(-1);
  if (!stream || !numOps || !opList || flags != 0) return cudaErrorInvalidValue;   // the legacy default stream too
  for (size_t i = 0; i < numOps; ++i) {
    const cudaMemcpy3DBatchOp& op = opList[i];
    if (!op.extent.width || !op.extent.height || !op.extent.depth) return cudaErrorInvalidValue;
    const int order = enum_value(op.srcAccessOrder);
    if (order < cudaMemcpySrcAccessOrderStream || order > cudaMemcpySrcAccessOrderAny) return cudaErrorInvalidValue;
    if (op.flags & ~1u) return cudaErrorInvalidValue;
    for (const cudaMemcpy3DOperand* o : {&op.src, &op.dst}) {
      const int t = enum_value(o->type);
      if (t != cudaMemcpyOperandTypePointer && t != cudaMemcpyOperandTypeArray) return cudaErrorInvalidValue;
    }
  }
  for (size_t i = 0; i < numOps; ++i) {
    const cudaMemcpy3DBatchOp& op = opList[i];
    cudaMemcpy3DParms p{};
    p.extent = op.extent;
    p.kind = cudaMemcpyDefault;
    auto operand = [&](const cudaMemcpy3DOperand& o, cudaPitchedPtr* ptr, cudaArray_t* array, cudaPos* pos) {
      if (enum_value(o.type) == cudaMemcpyOperandTypePointer) {
        const size_t row = o.op.ptr.rowLength ? o.op.ptr.rowLength : op.extent.width;
        const size_t layer = o.op.ptr.layerHeight ? o.op.ptr.layerHeight : op.extent.height;
        *ptr = [&] { cudaPitchedPtr pp{}; pp.ptr = o.op.ptr.ptr; pp.pitch = row; pp.xsize = row; pp.ysize = layer; return pp; }();
      } else {
        *array = o.op.array.array;
        *pos = [&] { cudaPos q{}; q.x = o.op.array.offset.x; q.y = o.op.array.offset.y; q.z = o.op.array.offset.z; return q; }();
      }
    };
    operand(op.src, &p.srcPtr, &p.srcArray, &p.srcPos);
    operand(op.dst, &p.dstPtr, &p.dstArray, &p.dstPos);
    if (const cudaError_t e = cudaMemcpy3DAsync(&p, stream); e != cudaSuccess) {
      if (failIdx) *failIdx = i;
      return e;
    }
  }
  return cudaSuccess;
}
}  // namespace

#if CUDART_VERSION >= 13000
static cudaError_t cudaMemcpy3DBatchAsync_traced(size_t numOps, cudaMemcpy3DBatchOp* opList, unsigned long long flags, cudaStream_t stream) {
  return memcpy3d_batch(numOps, opList, nullptr, flags, stream);
}

VGPU_EXPORT cudaError_t cudaMemcpy3DBatchAsync(size_t numOps, cudaMemcpy3DBatchOp* opList, unsigned long long flags, cudaStream_t stream) {
  return traced_call("cudaMemcpy3DBatchAsync", cudaMemcpy3DBatchAsync_traced, numOps, opList, flags, stream);
}
#else
static cudaError_t cudaMemcpy3DBatchAsync_traced(size_t numOps, cudaMemcpy3DBatchOp* opList, size_t* failIdx, unsigned long long flags, cudaStream_t stream) {
  return memcpy3d_batch(numOps, opList, failIdx, flags, stream);
}

VGPU_EXPORT cudaError_t cudaMemcpy3DBatchAsync(size_t numOps, cudaMemcpy3DBatchOp* opList, size_t* failIdx, unsigned long long flags, cudaStream_t stream) {
  return traced_call("cudaMemcpy3DBatchAsync", cudaMemcpy3DBatchAsync_traced, numOps, opList, failIdx, flags, stream);
}
#endif
#endif

// Prefetch and discard in batches (CUDA 13): a prefetch of each range to the location its index names, a discard
// (its contents become undefined) and both. Measured on an RTX 3060, which has no concurrent managed access: all
// three are cudaErrorInvalidValue, even for a valid managed range, as is a count of 0, a null list and flags.
// With concurrent managed access a range must be managed memory (else cudaErrorInvalidValue); a prefetch is
// recorded as the single prefetch call records it, and a discard has nothing to do, as nothing here is ever resident
// on a device to be thrown away.
#if CUDART_VERSION >= 13000
namespace {
cudaError_t managed_batch(State& s, void** dptrs, const size_t* sizes, size_t count, const cudaMemLocation* locs,
                          const size_t* idxs, size_t numLocs, unsigned long long flags, bool prefetch_too, bool need_locs,
                          cudaStream_t stream) {
  if (!stream || !count || !dptrs || !sizes || flags != 0) return cudaErrorInvalidValue;
  if (need_locs && (!locs || !idxs || !numLocs || idxs[0] != 0)) return cudaErrorInvalidValue;
  if (!pages_managed_memory(s)) return cudaErrorInvalidValue;
  for (size_t i = 0; i < numLocs && need_locs; ++i)
    if (idxs[i] >= count || (i && idxs[i] <= idxs[i - 1])) return cudaErrorInvalidValue;
  for (size_t i = 0; i < count; ++i)
    if (!sizes[i] || !managed_range(s, dptrs[i], sizes[i])) return cudaErrorInvalidValue;
  if (prefetch_too) {
    size_t loc = 0;
    for (size_t i = 0; i < count; ++i) {
      while (loc + 1 < numLocs && idxs[loc + 1] <= i) ++loc;
      const int type = enum_value(locs[loc].type);
      if (type != cudaMemLocationTypeDevice && type != cudaMemLocationTypeHost) return cudaErrorInvalidValue;
      if (const cudaError_t e = prefetch(s, dptrs[i], sizes[i], type == cudaMemLocationTypeHost ? cudaCpuDeviceId : locs[loc].id); e != cudaSuccess) return e;
    }
  }
  return cudaSuccess;
}
}  // namespace

static cudaError_t cudaMemPrefetchBatchAsync_traced(void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream) {
  return guard("cudaMemPrefetchBatchAsync", [&](State& s) -> cudaError_t {
    return managed_batch(s, dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, true, true, stream);
  });
}

VGPU_EXPORT cudaError_t cudaMemPrefetchBatchAsync(void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream) {
  return traced_call("cudaMemPrefetchBatchAsync", cudaMemPrefetchBatchAsync_traced, dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream);
}

static cudaError_t cudaMemDiscardBatchAsync_traced(void** dptrs, size_t* sizes, size_t count, unsigned long long flags, cudaStream_t stream) {
  return guard("cudaMemDiscardBatchAsync", [&](State& s) -> cudaError_t {
    return managed_batch(s, dptrs, sizes, count, nullptr, nullptr, 0, flags, false, false, stream);
  });
}

VGPU_EXPORT cudaError_t cudaMemDiscardBatchAsync(void** dptrs, size_t* sizes, size_t count, unsigned long long flags, cudaStream_t stream) {
  return traced_call("cudaMemDiscardBatchAsync", cudaMemDiscardBatchAsync_traced, dptrs, sizes, count, flags, stream);
}

static cudaError_t cudaMemDiscardAndPrefetchBatchAsync_traced(void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream) {
  return guard("cudaMemDiscardAndPrefetchBatchAsync", [&](State& s) -> cudaError_t {
    return managed_batch(s, dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, true, true, stream);
  });
}

VGPU_EXPORT cudaError_t cudaMemDiscardAndPrefetchBatchAsync(void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream) {
  return traced_call("cudaMemDiscardAndPrefetchBatchAsync", cudaMemDiscardAndPrefetchBatchAsync_traced, dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream);
}
#endif
