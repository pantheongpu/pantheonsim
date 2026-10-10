// The driver functions the toolkit's cuda.h declares and this library lacked, found by
// tests/lint/check_header_exports.py (the sweep of 2026-10-10). Included at the end of driver_api.cpp, whose
// helpers they use. Each answers as an RTX 3060 under NVIDIA's CUDA 13.0 driver does where that was measured
// (nvidia/tests/data/exports_sweep_driver.rtx3060.expected, made by run_exports_sweep.sh --card); where it was
// not, the comment says what the answer rests on. A function this simulator cannot honour is still exported and
// answers CUDA_ERROR_NOT_SUPPORTED, said once on stderr (never a plausible wrong answer).
//
// This library is built with no toolkit, so the structures it reads are declared here with the sizes and
// offsets cuda.h gives them (nvidia/tests/unit/test_driver_abi_sweep.cpp checks them against the toolkit's header
// where one is at hand).

#include <signal.h>
#include <sys/syscall.h>
#include <sys/types.h>

#include <array>

#include "driver_sweep_abi.hpp"

#define COMMA ,

namespace {

// A function that is not simulated: its status, and one line on stderr the first time it is called.
CUresult sweep_refuse(const char* api, const char* why, CUresult code = CUDA_ERROR_NOT_SUPPORTED) {
  static std::mutex mu;
  static std::set<std::string> said;
  std::lock_guard<std::mutex> lock(mu);
  if (!quiet() && said.insert(api).second)
    std::fprintf(stderr, "[vgpu] %s is not implemented by VirtualGPU: %s; returning %d\n", api, why, static_cast<int>(code));
  return code;
}

constexpr CUresult kIllegalState = static_cast<CUresult>(401);           // CUDA_ERROR_ILLEGAL_STATE
constexpr CUresult kOperatingSystem = static_cast<CUresult>(304);        // CUDA_ERROR_OPERATING_SYSTEM
constexpr CUresult kInvalidResourceType = static_cast<CUresult>(914);    // CUDA_ERROR_INVALID_RESOURCE_TYPE

using vgpu::abi::MemcpyAttributesABI;
using vgpu::abi::Memcpy3DOperandABI;
using vgpu::abi::Memcpy3DBatchOpABI;
using vgpu::abi::Memcpy3DPeerABI;

}  // namespace

// ---- a runtime kernel's function handle ------------------------------------------------------------------------------
// Called by the runtime shim's cudaGetFuncBySymbol (found through the process's global scope): the driver's
// CUfunction, and the CUmodule it belongs to, for a kernel of a module the runtime loaded (device, module id and
// the kernel's name). The same kernel gives the same handles, as on the card. The module handle is the runtime's
// module under a driver name: cuModuleUnload of it is accepted and does nothing, since the runtime owns it.
VGPU_EXPORT int vgpu_driver_function_for_v1(int dev, uint64_t mid, const char* name, void** func_out, void** module_out) {
  if (cuInit_impl(0) != CUDA_SUCCESS) return CUDA_ERROR_NOT_INITIALIZED;
  return api("vgpu_driver_function_for", true, false, [&](ShimState& s) -> CUresult {
    if (!func_out || !name || dev < 0 || dev >= s.rt->device_count()) return CUDA_ERROR_INVALID_VALUE;
    uintptr_t& mh = s.runtime_modules[{dev, mid}];
    if (!mh) {
      mh = make_handle(s, kTagModule);
      s.modules[mh] = {dev, mid};
    }
    uintptr_t& fh = s.runtime_functions[{mh, std::string(name)}];
    if (!fh) {
      fh = make_handle(s, kTagFunc);
      s.functions[fh] = {dev, mh, s.rt->device(dev).get_function(mid, name), s.rt->device(dev).symbols(mid), mid};
    }
    *func_out = reinterpret_cast<void*>(fh);
    if (module_out) *module_out = reinterpret_cast<void*>(mh);
    return CUDA_SUCCESS;
  });
}

// ---- device queries --------------------------------------------------------------------------------------------------------
// cuDeviceGetLuid: the Windows LUID of the adapter; a null buffer is CUDA_ERROR_INVALID_VALUE and a device that
// does not exist CUDA_ERROR_INVALID_DEVICE, as on an RTX 3060 under WSL, which also hands out a LUID. A Linux
// GPU has none, which is what is simulated, so a valid call is CUDA_ERROR_NOT_SUPPORTED.
static CUresult cuDeviceGetLuid_impl(char* luid, unsigned int* deviceNodeMask, CUdevice dev);
VGPU_EXPORT CUresult cuDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, CUdevice dev) { return traced("cuDeviceGetLuid", cuDeviceGetLuid_impl, luid, deviceNodeMask, dev); }
static CUresult cuDeviceGetLuid_impl(char* luid, unsigned int* deviceNodeMask, CUdevice dev) {
  return api("cuDeviceGetLuid", true, false, [&](ShimState& s) -> CUresult {
    if (!luid || !deviceNodeMask) return CUDA_ERROR_INVALID_VALUE;
    if (dev < 0 || dev >= s.rt->device_count()) return CUDA_ERROR_INVALID_DEVICE;
    return sweep_refuse("cuDeviceGetLuid", "a LUID is Windows'; a Linux GPU has none");
  });
}

// The widest 1D texture over linear memory for a format and a channel count: 4 GiB over the element size,
// at most 2^30 elements; one, two or four channels of a format that exists (measured on an RTX 3060: 2^30 for a
// float and for four bytes, 3 channels and a format that does not exist are CUDA_ERROR_INVALID_VALUE).
static CUresult cuDeviceGetTexture1DLinearMaxWidth_impl(size_t* maxWidthInElements, int format, unsigned numChannels, CUdevice dev);
VGPU_EXPORT CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, int format, unsigned numChannels, CUdevice dev) { return traced("cuDeviceGetTexture1DLinearMaxWidth", cuDeviceGetTexture1DLinearMaxWidth_impl, maxWidthInElements, format, numChannels, dev); }
static CUresult cuDeviceGetTexture1DLinearMaxWidth_impl(size_t* maxWidthInElements, int format, unsigned numChannels, CUdevice dev) {
  return api("cuDeviceGetTexture1DLinearMaxWidth", true, false, [&](ShimState& s) -> CUresult {
    if (!maxWidthInElements) return CUDA_ERROR_INVALID_VALUE;
    if (dev < 0 || dev >= s.rt->device_count()) return CUDA_ERROR_INVALID_DEVICE;
    unsigned bytes = 0;
    switch (format) {   // CUarray_format
      case 0x01: case 0x08: bytes = 1; break;                  // unsigned / signed 8
      case 0x02: case 0x09: case 0x10: bytes = 2; break;       // unsigned / signed 16, half
      case 0x03: case 0x0a: case 0x20: bytes = 4; break;       // unsigned / signed 32, float
      default: return CUDA_ERROR_INVALID_VALUE;
    }
    if (numChannels != 1 && numChannels != 2 && numChannels != 4) return CUDA_ERROR_INVALID_VALUE;
    const uint64_t element = uint64_t{bytes} * numChannels;
    *maxWidthInElements = static_cast<size_t>(std::min<uint64_t>(uint64_t{1} << 30, (uint64_t{1} << 32) / element));
    return CUDA_SUCCESS;
  });
}

// Which atomic operations the host and the peers can do on a device's memory (see the runtime's
// cudaDeviceGet*AtomicCapabilities in runtime_sweep.inc, which has the same operations, numbered the same):
// 0 for every operation for the host (cudaDevAttrHostNativeAtomicSupported is 0) and for a pair without a peer path,
// both measured on an RTX 3060; over a peer path derived from the documentation, not checked against a card.
namespace {
CUresult atomic_capabilities(unsigned* caps, const int* ops, unsigned count, bool path) {
  if (!caps || !ops || count == 0) return CUDA_ERROR_INVALID_VALUE;
  for (unsigned i = 0; i < count; ++i) {
    if (ops[i] < 0 || ops[i] > 12) return CUDA_ERROR_INVALID_VALUE;
    unsigned c = 0;
    if (path) {
      const unsigned scalars = (1u << 3) | (1u << 4);   // CU_ATOMIC_CAPABILITY_SCALAR_32 | _64
      const unsigned sign = (1u << 0) | (1u << 1), reduction = 1u << 2;
      if (ops[i] <= 7) c = sign | reduction | scalars;
      else if (ops[i] <= 9) c = sign | scalars;
      else c = reduction | scalars;
    }
    caps[i] = c;
  }
  return CUDA_SUCCESS;
}
}  // namespace

static CUresult cuDeviceGetHostAtomicCapabilities_impl(unsigned int* capabilities, const int* operations, unsigned int count, CUdevice dev);
VGPU_EXPORT CUresult cuDeviceGetHostAtomicCapabilities(unsigned int* capabilities, const int* operations, unsigned int count, CUdevice dev) { return traced("cuDeviceGetHostAtomicCapabilities", cuDeviceGetHostAtomicCapabilities_impl, capabilities, operations, count, dev); }
static CUresult cuDeviceGetHostAtomicCapabilities_impl(unsigned int* capabilities, const int* operations, unsigned int count, CUdevice dev) {
  return api("cuDeviceGetHostAtomicCapabilities", true, false, [&](ShimState& s) -> CUresult {
    if (!capabilities || !operations || count == 0) return CUDA_ERROR_INVALID_VALUE;
    if (dev < 0 || dev >= s.rt->device_count()) return CUDA_ERROR_INVALID_DEVICE;
    return atomic_capabilities(capabilities, operations, count, false);
  });
}

static CUresult cuDeviceGetP2PAtomicCapabilities_impl(unsigned int* capabilities, const int* operations, unsigned int count, CUdevice srcDevice, CUdevice dstDevice);
VGPU_EXPORT CUresult cuDeviceGetP2PAtomicCapabilities(unsigned int* capabilities, const int* operations, unsigned int count, CUdevice srcDevice, CUdevice dstDevice) { return traced("cuDeviceGetP2PAtomicCapabilities", cuDeviceGetP2PAtomicCapabilities_impl, capabilities, operations, count, srcDevice, dstDevice); }
static CUresult cuDeviceGetP2PAtomicCapabilities_impl(unsigned int* capabilities, const int* operations, unsigned int count, CUdevice srcDevice, CUdevice dstDevice) {
  return api("cuDeviceGetP2PAtomicCapabilities", true, false, [&](ShimState& s) -> CUresult {
    const int n = s.rt->device_count();
    if (!capabilities || !operations || count == 0) return CUDA_ERROR_INVALID_VALUE;
    if (srcDevice < 0 || srcDevice >= n || dstDevice < 0 || dstDevice >= n || srcDevice == dstDevice) return CUDA_ERROR_INVALID_DEVICE;
    return atomic_capabilities(capabilities, operations, count, peer_path(s, srcDevice, dstDevice));
  });
}

// NvSciSync is Tegra's: an RTX 3060 answers CUDA_ERROR_INVALID_VALUE for an attribute list it is given,
// CUDA_ERROR_INVALID_HANDLE for none, CUDA_ERROR_INVALID_DEVICE for a device that does not exist.
static CUresult cuDeviceGetNvSciSyncAttributes_impl(void* nvSciSyncAttrList, CUdevice dev, int flags);
VGPU_EXPORT CUresult cuDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, CUdevice dev, int flags) { return traced("cuDeviceGetNvSciSyncAttributes", cuDeviceGetNvSciSyncAttributes_impl, nvSciSyncAttrList, dev, flags); }
static CUresult cuDeviceGetNvSciSyncAttributes_impl(void* nvSciSyncAttrList, CUdevice dev, int flags) {
  return api("cuDeviceGetNvSciSyncAttributes", true, false, [&](ShimState& s) -> CUresult {
    (void)flags;
    if (dev < 0 || dev >= s.rt->device_count()) return CUDA_ERROR_INVALID_DEVICE;
    if (!nvSciSyncAttrList) return CUDA_ERROR_INVALID_HANDLE;
    return sweep_refuse("cuDeviceGetNvSciSyncAttributes", "NvSciSync is for Tegra; this device has none", CUDA_ERROR_INVALID_VALUE);
  });
}

// Flushing GPUDirect RDMA writes needs a device with the capability (cudaDevAttrGPUDirectRDMAFlushWritesOptions
// is 0 here): CUDA_ERROR_NOT_SUPPORTED for a valid target and scope, CUDA_ERROR_INVALID_VALUE for a target
// other than the current context (0) or a scope other than the owner (100) and all devices (200).
static CUresult cuFlushGPUDirectRDMAWrites_impl(int target, int scope);
VGPU_EXPORT CUresult cuFlushGPUDirectRDMAWrites(int target, int scope) { return traced("cuFlushGPUDirectRDMAWrites", cuFlushGPUDirectRDMAWrites_impl, target, scope); }
static CUresult cuFlushGPUDirectRDMAWrites_impl(int target, int scope) {
  return api("cuFlushGPUDirectRDMAWrites", true, false, [&](ShimState&) -> CUresult {
    if (target != 0 || (scope != 100 && scope != 200)) return CUDA_ERROR_INVALID_VALUE;
    return sweep_refuse("cuFlushGPUDirectRDMAWrites", "this device has no GPUDirect RDMA");
  });
}

// ---- contexts ------------------------------------------------------------------------------------------------------------------
// CUDA 13's forms that take the context: cuCtxGetDevice_v2 and cuCtxSynchronize_v2 with a null context mean the
// current one; one that does not exist is CUDA_ERROR_INVALID_CONTEXT. cuCtxRecordEvent and cuCtxWaitEvent need a
// context (null is CUDA_ERROR_INVALID_CONTEXT) and an event (null is CUDA_ERROR_INVALID_HANDLE); a wait on an
// event that was never recorded returns at once.
static CUresult cuCtxGetDevice_v2_impl(CUdevice* device, CUcontext ctx);
VGPU_EXPORT CUresult cuCtxGetDevice_v2(CUdevice* device, CUcontext ctx) { return traced("cuCtxGetDevice_v2", cuCtxGetDevice_v2_impl, device, ctx); }
static CUresult cuCtxGetDevice_v2_impl(CUdevice* device, CUcontext ctx) {
  return api("cuCtxGetDevice_v2", true, false, [&](ShimState& s) -> CUresult {
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    if (!ctx) {
      *device = current_device(s);
      return CUDA_SUCCESS;
    }
    const auto it = s.contexts.find(reinterpret_cast<uintptr_t>(ctx));
    if (it == s.contexts.end()) return CUDA_ERROR_INVALID_CONTEXT;
    *device = it->second;
    return CUDA_SUCCESS;
  });
}

static CUresult cuCtxSynchronize_v2_impl(CUcontext ctx);
VGPU_EXPORT CUresult cuCtxSynchronize_v2(CUcontext ctx) { return traced("cuCtxSynchronize_v2", cuCtxSynchronize_v2_impl, ctx); }
static CUresult cuCtxSynchronize_v2_impl(CUcontext ctx) {
  if (ctx) {
    ShimState& s = state();
    std::lock_guard<std::recursive_mutex> lock(s.mu);
    if (!s.contexts.count(reinterpret_cast<uintptr_t>(ctx))) return CUDA_ERROR_INVALID_CONTEXT;
  }
  return cuCtxSynchronize();
}

static CUresult cuCtxResetPersistingL2Cache_impl();
VGPU_EXPORT CUresult cuCtxResetPersistingL2Cache() { return traced("cuCtxResetPersistingL2Cache", cuCtxResetPersistingL2Cache_impl); }
static CUresult cuCtxResetPersistingL2Cache_impl() {
  return api("cuCtxResetPersistingL2Cache", true, false, [&](ShimState&) -> CUresult { return CUDA_SUCCESS; });
}

static CUresult cuCtxRecordEvent_impl(CUcontext hCtx, void* hEvent);
VGPU_EXPORT CUresult cuCtxRecordEvent(CUcontext hCtx, void* hEvent) { return traced("cuCtxRecordEvent", cuCtxRecordEvent_impl, hCtx, hEvent); }
static CUresult cuCtxRecordEvent_impl(CUcontext hCtx, void* hEvent) {
  {
    ShimState& s = state();
    std::lock_guard<std::recursive_mutex> lock(s.mu);
    if (!hCtx || !s.contexts.count(reinterpret_cast<uintptr_t>(hCtx))) return CUDA_ERROR_INVALID_CONTEXT;
    if (!hEvent) return CUDA_ERROR_INVALID_HANDLE;
  }
  return cuEventRecord(hEvent, nullptr);
}

static CUresult cuCtxWaitEvent_impl(CUcontext hCtx, void* hEvent);
VGPU_EXPORT CUresult cuCtxWaitEvent(CUcontext hCtx, void* hEvent) { return traced("cuCtxWaitEvent", cuCtxWaitEvent_impl, hCtx, hEvent); }
static CUresult cuCtxWaitEvent_impl(CUcontext hCtx, void* hEvent) {
  {
    ShimState& s = state();
    std::lock_guard<std::recursive_mutex> lock(s.mu);
    if (!hCtx || !s.contexts.count(reinterpret_cast<uintptr_t>(hCtx))) return CUDA_ERROR_INVALID_CONTEXT;
    if (!hEvent) return CUDA_ERROR_INVALID_HANDLE;
  }
  return cuStreamWaitEvent(nullptr, hEvent, 0);
}

// ---- the functions of a module, the kernels of a library -----------------------------------------------------------
// A module's functions as handles (cuModuleEnumerateFunctions fills as many as it is given, up to the count the
// module has; the null array, the null module and the null count are CUDA_ERROR_INVALID_VALUE, measured).
static CUresult cuModuleGetFunctionCount_impl(unsigned int* count, CUmodule mod);
VGPU_EXPORT CUresult cuModuleGetFunctionCount(unsigned int* count, CUmodule mod) { return traced("cuModuleGetFunctionCount", cuModuleGetFunctionCount_impl, count, mod); }
static CUresult cuModuleGetFunctionCount_impl(unsigned int* count, CUmodule mod) {
  return api("cuModuleGetFunctionCount", true, false, [&](ShimState& s) -> CUresult {
    if (!count || !mod) return CUDA_ERROR_INVALID_VALUE;
    const auto it = s.modules.find(reinterpret_cast<uintptr_t>(mod));
    if (it == s.modules.end()) return CUDA_ERROR_INVALID_VALUE;
    *count = static_cast<unsigned>(s.rt->device(it->second.first).kernel_names(it->second.second).size());
    return CUDA_SUCCESS;
  });
}

static CUresult cuModuleEnumerateFunctions_impl(CUfunction* functions, unsigned int numFunctions, CUmodule mod);
VGPU_EXPORT CUresult cuModuleEnumerateFunctions(CUfunction* functions, unsigned int numFunctions, CUmodule mod) { return traced("cuModuleEnumerateFunctions", cuModuleEnumerateFunctions_impl, functions, numFunctions, mod); }
static CUresult cuModuleEnumerateFunctions_impl(CUfunction* functions, unsigned int numFunctions, CUmodule mod) {
  return api("cuModuleEnumerateFunctions", true, false, [&](ShimState& s) -> CUresult {
    if (!functions || !mod) return CUDA_ERROR_INVALID_VALUE;
    const auto it = s.modules.find(reinterpret_cast<uintptr_t>(mod));
    if (it == s.modules.end()) return CUDA_ERROR_INVALID_VALUE;
    const auto [dev, mid] = it->second;
    const std::vector<std::string> names = s.rt->device(dev).kernel_names(mid);
    for (unsigned i = 0; i < numFunctions && i < names.size(); ++i) {
      const uintptr_t mh = reinterpret_cast<uintptr_t>(mod);
      uintptr_t& fh = s.runtime_functions[{mh, names[i]}];
      if (!fh || !s.functions.count(fh)) {
        fh = make_handle(s, kTagFunc);
        s.functions[fh] = {dev, mh, s.rt->device(dev).get_function(mid, names[i]), s.rt->device(dev).symbols(mid), mid};
      }
      functions[i] = reinterpret_cast<CUfunction>(fh);
    }
    return CUDA_SUCCESS;
  });
}

// cuLibraryLoadFromFile: the file's bytes, as cuLibraryLoadData takes them (a file that is not an image is
// taken for PTX text and fails when it is used, as on the card, which loads /etc/hostname without complaint); a
// file that is not there is CUDA_ERROR_FILE_NOT_FOUND.
static CUresult cuLibraryLoadFromFile_impl(void** library, const char* fileName, void* jitOptions, void** jitOptionValues, unsigned int numJitOptions, void* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions);
VGPU_EXPORT CUresult cuLibraryLoadFromFile(void** library, const char* fileName, void* jitOptions, void** jitOptionValues, unsigned int numJitOptions, void* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) { return traced("cuLibraryLoadFromFile", cuLibraryLoadFromFile_impl, library, fileName, jitOptions, jitOptionValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions); }
static CUresult cuLibraryLoadFromFile_impl(void** library, const char* fileName, void* jitOptions, void** jitOptionValues, unsigned int numJitOptions, void* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) {
  if (!library || !fileName) return CUDA_ERROR_INVALID_VALUE;
  std::ifstream in(fileName, std::ios::binary);
  if (!in) return CUDA_ERROR_FILE_NOT_FOUND;
  std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (bytes.size() < 4) bytes.resize(4, '\0');
  bytes.push_back('\0');   // text is NUL-terminated for cuLibraryLoadData
  return cuLibraryLoadData(library, bytes.data(), jitOptions, jitOptionValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);
}

static CUresult cuLibraryGetKernelCount_impl(unsigned int* count, void* lib);
VGPU_EXPORT CUresult cuLibraryGetKernelCount(unsigned int* count, void* lib) { return traced("cuLibraryGetKernelCount", cuLibraryGetKernelCount_impl, count, lib); }
static CUresult cuLibraryGetKernelCount_impl(unsigned int* count, void* lib) {
  return api("cuLibraryGetKernelCount", true, false, [&](ShimState& s) -> CUresult {
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    if (!lib) return CUDA_ERROR_INVALID_HANDLE;
    const uintptr_t h = check_handle(reinterpret_cast<uintptr_t>(lib), kTagLibrary, "library");
    if (!s.libraries.count(h)) return CUDA_ERROR_INVALID_HANDLE;
    const int dev = ctx_stack().empty() ? 0 : current_device(s);
    *count = static_cast<unsigned>(s.rt->device(dev).kernel_names(library_module_on(s, h, dev)).size());
    return CUDA_SUCCESS;
  });
}

static CUresult cuLibraryEnumerateKernels_impl(void** kernels, unsigned int numKernels, void* lib);
VGPU_EXPORT CUresult cuLibraryEnumerateKernels(void** kernels, unsigned int numKernels, void* lib) { return traced("cuLibraryEnumerateKernels", cuLibraryEnumerateKernels_impl, kernels, numKernels, lib); }
static CUresult cuLibraryEnumerateKernels_impl(void** kernels, unsigned int numKernels, void* lib) {
  return api("cuLibraryEnumerateKernels", true, false, [&](ShimState& s) -> CUresult {
    if (!kernels) return CUDA_ERROR_INVALID_VALUE;
    if (!lib) return CUDA_ERROR_INVALID_HANDLE;
    const uintptr_t h = check_handle(reinterpret_cast<uintptr_t>(lib), kTagLibrary, "library");
    if (!s.libraries.count(h)) return CUDA_ERROR_INVALID_HANDLE;
    const int dev = ctx_stack().empty() ? 0 : current_device(s);
    const std::vector<std::string> names = s.rt->device(dev).kernel_names(library_module_on(s, h, dev));
    for (unsigned i = 0; i < numKernels && i < names.size(); ++i) {
      uintptr_t kh = 0;
      for (const auto& [handle, rec] : s.kernels)
        if (rec.library == h && rec.name == names[i]) kh = handle;
      if (!kh) {
        kh = make_handle(s, kTagKernel);
        s.kernels[kh] = {h, names[i]};
      }
      kernels[i] = reinterpret_cast<void*>(kh);
    }
    return CUDA_SUCCESS;
  });
}

static CUresult cuKernelGetLibrary_impl(void** pLib, void* kernel);
VGPU_EXPORT CUresult cuKernelGetLibrary(void** pLib, void* kernel) { return traced("cuKernelGetLibrary", cuKernelGetLibrary_impl, pLib, kernel); }
static CUresult cuKernelGetLibrary_impl(void** pLib, void* kernel) {
  return api("cuKernelGetLibrary", true, false, [&](ShimState& s) -> CUresult {
    if (!pLib) return CUDA_ERROR_INVALID_VALUE;
    if (!kernel) return CUDA_ERROR_INVALID_HANDLE;
    const uintptr_t kh = check_handle(reinterpret_cast<uintptr_t>(kernel), kTagKernel, "kernel");
    const auto it = s.kernels.find(kh);
    if (it == s.kernels.end()) return CUDA_ERROR_INVALID_HANDLE;
    *pLib = reinterpret_cast<void*>(it->second.library);
    return CUDA_SUCCESS;
  });
}

// A library's __device__ variable and __managed__ variable, as cuModuleGetGlobal and a module's managed variable
// answer: either output may be omitted but not both; a name the library does not declare is NOT_FOUND, a kernel's
// name is CUDA_ERROR_INVALID_VALUE (GetGlobal) or NOT_FOUND (GetManaged), and a variable that is not managed is
// NOT_FOUND to GetManaged. GetUnifiedFunction is for devices with unified function pointers
// (cudaDevAttrUnifiedFunctionPointers, 0 for an RTX 3060): it is CUDA_ERROR_INVALID_VALUE for everything there.
static CUresult cuLibraryGetGlobal_impl(unsigned long long* dptr, size_t* bytes, void* library, const char* name);
VGPU_EXPORT CUresult cuLibraryGetGlobal(unsigned long long* dptr, size_t* bytes, void* library, const char* name) { return traced("cuLibraryGetGlobal", cuLibraryGetGlobal_impl, dptr, bytes, library, name); }
static CUresult cuLibraryGetGlobal_impl(unsigned long long* dptr, size_t* bytes, void* library, const char* name) {
  return api("cuLibraryGetGlobal", true, false, [&](ShimState& s) -> CUresult {
    if ((!dptr && !bytes) || !name) return CUDA_ERROR_INVALID_VALUE;
    if (!library) return CUDA_ERROR_INVALID_HANDLE;
    const uintptr_t h = check_handle(reinterpret_cast<uintptr_t>(library), kTagLibrary, "library");
    if (!s.libraries.count(h)) return CUDA_ERROR_INVALID_HANDLE;
    const int dev = ctx_stack().empty() ? 0 : current_device(s);
    const uint64_t mid = library_module_on(s, h, dev);
    uint64_t addr = 0, size = 0;
    if (!s.rt->device(dev).global(mid, name, &addr, &size))
      return s.rt->device(dev).has_kernel(mid, name) ? CUDA_ERROR_INVALID_VALUE : CUDA_ERROR_NOT_FOUND;
    if (dptr) *dptr = addr;
    if (bytes) *bytes = static_cast<size_t>(size);
    return CUDA_SUCCESS;
  });
}

static CUresult cuLibraryGetManaged_impl(unsigned long long* dptr, size_t* bytes, void* library, const char* name);
VGPU_EXPORT CUresult cuLibraryGetManaged(unsigned long long* dptr, size_t* bytes, void* library, const char* name) { return traced("cuLibraryGetManaged", cuLibraryGetManaged_impl, dptr, bytes, library, name); }
static CUresult cuLibraryGetManaged_impl(unsigned long long* dptr, size_t* bytes, void* library, const char* name) {
  return api("cuLibraryGetManaged", true, false, [&](ShimState& s) -> CUresult {
    if ((!dptr && !bytes) || !name) return CUDA_ERROR_INVALID_VALUE;
    if (!library) return CUDA_ERROR_INVALID_HANDLE;
    const uintptr_t h = check_handle(reinterpret_cast<uintptr_t>(library), kTagLibrary, "library");
    if (!s.libraries.count(h)) return CUDA_ERROR_INVALID_HANDLE;
    const int dev = ctx_stack().empty() ? 0 : current_device(s);
    const uint64_t mid = library_module_on(s, h, dev);
    const std::vector<std::string> managed = s.rt->device(dev).managed_globals(mid);
    uint64_t addr = 0, size = 0;
    if (std::find(managed.begin(), managed.end(), std::string(name)) == managed.end() ||
        !s.rt->device(dev).global(mid, name, &addr, &size))
      return CUDA_ERROR_NOT_FOUND;
    if (dptr) *dptr = addr;
    if (bytes) *bytes = static_cast<size_t>(size);
    return CUDA_SUCCESS;
  });
}

static CUresult cuLibraryGetUnifiedFunction_impl(void** fptr, void* library, const char* symbol);
VGPU_EXPORT CUresult cuLibraryGetUnifiedFunction(void** fptr, void* library, const char* symbol) { return traced("cuLibraryGetUnifiedFunction", cuLibraryGetUnifiedFunction_impl, fptr, library, symbol); }
static CUresult cuLibraryGetUnifiedFunction_impl(void** fptr, void* library, const char* symbol) {
  return api("cuLibraryGetUnifiedFunction", true, false, [&](ShimState&) -> CUresult {
    (void)fptr;
    (void)symbol;
    if (!library) return CUDA_ERROR_INVALID_HANDLE;
    return CUDA_ERROR_INVALID_VALUE;   // cudaDevAttrUnifiedFunctionPointers is 0 for every profile below sm_120
  });
}

// A kernel's cache preference (accepted, like cuFuncSetCacheConfig: there is no cache to apply it to): the
// device is the one the setting is for, not -1; a configuration past the four is CUDA_ERROR_INVALID_VALUE.
static CUresult cuKernelSetCacheConfig_impl(void* kernel, int config, CUdevice dev);
VGPU_EXPORT CUresult cuKernelSetCacheConfig(void* kernel, int config, CUdevice dev) { return traced("cuKernelSetCacheConfig", cuKernelSetCacheConfig_impl, kernel, config, dev); }
static CUresult cuKernelSetCacheConfig_impl(void* kernel, int config, CUdevice dev) {
  return api("cuKernelSetCacheConfig", true, false, [&](ShimState& s) -> CUresult {
    if (config < 0 || config > 3) return CUDA_ERROR_INVALID_VALUE;
    if (!kernel) return CUDA_ERROR_INVALID_HANDLE;
    if (dev < 0 || dev >= s.rt->device_count()) return CUDA_ERROR_INVALID_DEVICE;
    const uintptr_t kh = check_handle(reinterpret_cast<uintptr_t>(kernel), kTagKernel, "kernel");
    return s.kernels.count(kh) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
  });
}

// Where a parameter of a kernel sits in its parameter block (the layout of a launch: each parameter at the next
// offset its alignment allows) and its size: cuKernelGetParamInfo and cuFuncGetParamInfo, as the runtime's
// cudaFuncGetParamInfo. An index past the last, a null offset and a null kernel are CUDA_ERROR_INVALID_VALUE
// (measured); the size is optional.
namespace {
CUresult param_info(const vgpu::ptx::EntryFn* fn, size_t paramIndex, size_t* paramOffset, size_t* paramSize) {
  if (!fn || !paramOffset || paramIndex >= fn->params.size()) return CUDA_ERROR_INVALID_VALUE;
  uint64_t end = 0, offset = 0, size = 0;
  for (size_t i = 0; i <= paramIndex; ++i) {
    size = fn->params[i].size;
    const uint64_t align = fn->params[i].align ? fn->params[i].align : (size < 8 ? size : 8);
    offset = align ? (end + align - 1) / align * align : end;
    end = offset + size;
  }
  *paramOffset = static_cast<size_t>(offset);
  if (paramSize) *paramSize = static_cast<size_t>(size);
  return CUDA_SUCCESS;
}
}  // namespace

static CUresult cuKernelGetParamInfo_impl(void* kernel, size_t paramIndex, size_t* paramOffset, size_t* paramSize);
VGPU_EXPORT CUresult cuKernelGetParamInfo(void* kernel, size_t paramIndex, size_t* paramOffset, size_t* paramSize) { return traced("cuKernelGetParamInfo", cuKernelGetParamInfo_impl, kernel, paramIndex, paramOffset, paramSize); }
static CUresult cuKernelGetParamInfo_impl(void* kernel, size_t paramIndex, size_t* paramOffset, size_t* paramSize) {
  return api("cuKernelGetParamInfo", true, false, [&](ShimState& s) -> CUresult {
    if (!kernel) return CUDA_ERROR_INVALID_VALUE;
    const uintptr_t kh = check_handle(reinterpret_cast<uintptr_t>(kernel), kTagKernel, "kernel");
    const auto it = s.kernels.find(kh);
    if (it == s.kernels.end()) return CUDA_ERROR_INVALID_VALUE;
    const int dev = ctx_stack().empty() ? 0 : current_device(s);
    const uint64_t mid = library_module_on(s, it->second.library, dev);
    return param_info(s.rt->device(dev).get_function(mid, it->second.name), paramIndex, paramOffset, paramSize);
  });
}

static CUresult cuFuncGetParamInfo_impl(CUfunction func, size_t paramIndex, size_t* paramOffset, size_t* paramSize);
VGPU_EXPORT CUresult cuFuncGetParamInfo(CUfunction func, size_t paramIndex, size_t* paramOffset, size_t* paramSize) { return traced("cuFuncGetParamInfo", cuFuncGetParamInfo_impl, func, paramIndex, paramOffset, paramSize); }
static CUresult cuFuncGetParamInfo_impl(CUfunction func, size_t paramIndex, size_t* paramOffset, size_t* paramSize) {
  return api("cuFuncGetParamInfo", true, false, [&](ShimState& s) -> CUresult {
    const auto it = s.functions.find(reinterpret_cast<uintptr_t>(func));
    if (it == s.functions.end()) return CUDA_ERROR_INVALID_VALUE;
    return param_info(it->second.fn, paramIndex, paramOffset, paramSize);
  });
}

// A function's name (the mangled one, as the module has it), the module it came from and, for a function of a
// library's kernel, the module of that library on the function's device. A null function or a null result
// is CUDA_ERROR_INVALID_VALUE.
static CUresult cuFuncGetName_impl(const char** name, CUfunction hfunc);
VGPU_EXPORT CUresult cuFuncGetName(const char** name, CUfunction hfunc) { return traced("cuFuncGetName", cuFuncGetName_impl, name, hfunc); }
static CUresult cuFuncGetName_impl(const char** name, CUfunction hfunc) {
  return api("cuFuncGetName", true, false, [&](ShimState& s) -> CUresult {
    if (!name) return CUDA_ERROR_INVALID_VALUE;
    const auto it = s.functions.find(reinterpret_cast<uintptr_t>(hfunc));
    if (it == s.functions.end() || !it->second.fn) return CUDA_ERROR_INVALID_VALUE;
    *name = it->second.fn->name.c_str();
    return CUDA_SUCCESS;
  });
}

static CUresult cuFuncGetModule_impl(CUmodule* hmod, CUfunction hfunc);
VGPU_EXPORT CUresult cuFuncGetModule(CUmodule* hmod, CUfunction hfunc) { return traced("cuFuncGetModule", cuFuncGetModule_impl, hmod, hfunc); }
static CUresult cuFuncGetModule_impl(CUmodule* hmod, CUfunction hfunc) {
  return api("cuFuncGetModule", true, false, [&](ShimState& s) -> CUresult {
    if (!hmod) return CUDA_ERROR_INVALID_VALUE;
    const auto it = s.functions.find(reinterpret_cast<uintptr_t>(hfunc));
    if (it == s.functions.end()) return CUDA_ERROR_INVALID_VALUE;
    const uintptr_t owner = it->second.module_handle;
    if ((owner & 7) == kTagModule) {
      *hmod = reinterpret_cast<CUmodule>(owner);
      return CUDA_SUCCESS;
    }
    // A kernel of a library: the library's module on the function's device, as cuLibraryGetModule hands it out.
    if ((owner & 7) == kTagLibrary && s.libraries.count(owner)) {
      const int dev = it->second.device;
      const uint64_t mid = library_module_on(s, owner, dev);
      uintptr_t& mh = s.runtime_modules[{dev, mid}];
      if (!mh) {
        mh = make_handle(s, kTagModule);
        s.modules[mh] = {dev, mid};
      }
      *hmod = reinterpret_cast<CUmodule>(mh);
      return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_VALUE;
  });
}

// Dynamic shared memory each of `numBlocks` resident blocks may ask for: see the runtime's
// cudaOccupancyAvailableDynamicSMemPerBlock (runtime_sweep.inc) for the rule and the measurements.
static CUresult cuOccupancyAvailableDynamicSMemPerBlock_impl(size_t* dynamicSmemSize, CUfunction func, int numBlocks, int blockSize);
VGPU_EXPORT CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, CUfunction func, int numBlocks, int blockSize) { return traced("cuOccupancyAvailableDynamicSMemPerBlock", cuOccupancyAvailableDynamicSMemPerBlock_impl, dynamicSmemSize, func, numBlocks, blockSize); }
static CUresult cuOccupancyAvailableDynamicSMemPerBlock_impl(size_t* dynamicSmemSize, CUfunction func, int numBlocks, int blockSize) {
  return api("cuOccupancyAvailableDynamicSMemPerBlock", true, false, [&](ShimState& s) -> CUresult {
    if (!dynamicSmemSize) return CUDA_ERROR_INVALID_VALUE;
    if (!func) return CUDA_ERROR_INVALID_HANDLE;
    const auto it = s.functions.find(reinterpret_cast<uintptr_t>(func));
    if (it == s.functions.end() || !it->second.fn) return CUDA_ERROR_INVALID_HANDLE;
    if (numBlocks <= 0 || blockSize <= 0) return CUDA_ERROR_INVALID_VALUE;
    const vgpu::DeviceProfile& p = s.rt->device(it->second.device).profile();
    if (static_cast<uint32_t>(blockSize) > p.limits.max_threads_per_block) return CUDA_ERROR_INVALID_VALUE;
    const auto res = vgpu::exec::kernel_resources(*it->second.fn, p, static_cast<uint32_t>(blockSize), 0);
    if (static_cast<uint32_t>(numBlocks) > res.occupancy.blocks_per_sm) return CUDA_ERROR_INVALID_VALUE;
    const uint64_t granule = p.cc_major >= 8 ? 128 : 256;
    uint64_t share = p.limits.shared_mem_per_sm / static_cast<uint64_t>(numBlocks) / granule * granule;
    const uint64_t reserved = p.reserved_smem_per_block();
    share = share > reserved ? share - reserved : 0;
    share = std::min<uint64_t>(share, p.limits.shared_mem_per_block);
    const uint64_t fixed = it->second.fn->static_shared_size;
    *dynamicSmemSize = static_cast<size_t>(share > fixed ? share - fixed : 0);
    return CUDA_SUCCESS;
  });
}

// ---- the callbacks about the device (CUDA 12.8) --------------------------------------------------------------------------
// See the runtime's cudaDeviceRegisterAsyncNotification: a handle, no notification ever sent.
namespace {
std::mutex g_async_mu;
std::set<uintptr_t> g_async;
uintptr_t g_next_async = 1;
}  // namespace

static CUresult cuDeviceRegisterAsyncNotification_impl(CUdevice device, void* callbackFunc, void* userData, void** callback);
VGPU_EXPORT CUresult cuDeviceRegisterAsyncNotification(CUdevice device, void* callbackFunc, void* userData, void** callback) { return traced("cuDeviceRegisterAsyncNotification", cuDeviceRegisterAsyncNotification_impl, device, callbackFunc, userData, callback); }
static CUresult cuDeviceRegisterAsyncNotification_impl(CUdevice device, void* callbackFunc, void* userData, void** callback) {
  return api("cuDeviceRegisterAsyncNotification", true, false, [&](ShimState& s) -> CUresult {
    (void)userData;
    if (device < 0 || device >= s.rt->device_count()) return CUDA_ERROR_INVALID_DEVICE;
    if (!callbackFunc || !callback) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::mutex> lock(g_async_mu);
    const uintptr_t h = g_next_async++;
    g_async.insert(h);
    *callback = reinterpret_cast<void*>(h);
    return CUDA_SUCCESS;
  });
}

static CUresult cuDeviceUnregisterAsyncNotification_impl(CUdevice device, void* callback);
VGPU_EXPORT CUresult cuDeviceUnregisterAsyncNotification(CUdevice device, void* callback) { return traced("cuDeviceUnregisterAsyncNotification", cuDeviceUnregisterAsyncNotification_impl, device, callback); }
static CUresult cuDeviceUnregisterAsyncNotification_impl(CUdevice device, void* callback) {
  return api("cuDeviceUnregisterAsyncNotification", true, false, [&](ShimState& s) -> CUresult {
    if (device < 0 || device >= s.rt->device_count()) return CUDA_ERROR_INVALID_DEVICE;
    if (!callback) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::mutex> lock(g_async_mu);
    return g_async.erase(reinterpret_cast<uintptr_t>(callback)) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
  });
}

// ---- copies ------------------------------------------------------------------------------------------------------------------
// A copy between the memories of two contexts (cuMemcpy3DPeer): device addresses are unique across the devices
// here, so it is the ordinary 3D copy with the contexts dropped. An empty box (a width, height or depth of 0)
// is CUDA_SUCCESS and touches nothing, as the card has it; a null structure is CUDA_ERROR_INVALID_VALUE.
static CUresult cuMemcpy3DPeer_impl(const void* pCopy);
VGPU_EXPORT CUresult cuMemcpy3DPeer(const void* pCopy) { return traced("cuMemcpy3DPeer", cuMemcpy3DPeer_impl, pCopy); }
static CUresult cuMemcpy3DPeer_impl(const void* pCopy) {
  if (!pCopy) return CUDA_ERROR_INVALID_VALUE;
  const auto& p = *static_cast<const Memcpy3DPeerABI*>(pCopy);
  CUDA_MEMCPY3D c{};
  c.srcXInBytes = p.srcXInBytes; c.srcY = p.srcY; c.srcZ = p.srcZ; c.srcLOD = p.srcLOD;
  c.srcMemoryType = static_cast<CUmemorytype>(p.srcMemoryType);
  c.srcHost = p.srcHost; c.srcDevice = p.srcDevice; c.srcArray = static_cast<CUarray>(p.srcArray);
  c.srcPitch = p.srcPitch; c.srcHeight = p.srcHeight;
  c.dstXInBytes = p.dstXInBytes; c.dstY = p.dstY; c.dstZ = p.dstZ; c.dstLOD = p.dstLOD;
  c.dstMemoryType = static_cast<CUmemorytype>(p.dstMemoryType);
  c.dstHost = p.dstHost; c.dstDevice = p.dstDevice; c.dstArray = static_cast<CUarray>(p.dstArray);
  c.dstPitch = p.dstPitch; c.dstHeight = p.dstHeight;
  c.WidthInBytes = p.WidthInBytes; c.Height = p.Height; c.Depth = p.Depth;
  return cuMemcpy3D_v2(&c);
}

static CUresult cuMemcpy3DPeerAsync_impl(const void* pCopy, CUstream hStream);
VGPU_EXPORT CUresult cuMemcpy3DPeerAsync(const void* pCopy, CUstream hStream) { return traced("cuMemcpy3DPeerAsync", cuMemcpy3DPeerAsync_impl, pCopy, hStream); }
static CUresult cuMemcpy3DPeerAsync_impl(const void* pCopy, CUstream hStream) {
  (void)hStream;
  return cuMemcpy3DPeer(pCopy);
}

// Copies in a batch (CUDA 13: no failure index): `count` copies, with attributes for runs of them. As the
// runtime's cudaMemcpyBatchAsync, and the card's, for a count or an attribute count of 0, a null array, an
// attribute list that does not start at copy 0, a source access order outside the three and flags past bit 0
// (CUDA_ERROR_INVALID_VALUE); each copy is then an async copy of the default kind.
static CUresult cuMemcpyBatchAsync_v2_impl(unsigned long long* dsts, unsigned long long* srcs, size_t* sizes, size_t count, void* attrs, size_t* attrsIdxs, size_t numAttrs, CUstream hStream);
VGPU_EXPORT CUresult cuMemcpyBatchAsync_v2(unsigned long long* dsts, unsigned long long* srcs, size_t* sizes, size_t count, void* attrs, size_t* attrsIdxs, size_t numAttrs, CUstream hStream) { return traced("cuMemcpyBatchAsync_v2", cuMemcpyBatchAsync_v2_impl, dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, hStream); }
static CUresult cuMemcpyBatchAsync_v2_impl(unsigned long long* dsts, unsigned long long* srcs, size_t* sizes, size_t count, void* attrs, size_t* attrsIdxs, size_t numAttrs, CUstream hStream) {
  // The legacy default stream is refused (CUDA_ERROR_INVALID_VALUE on an RTX 3060, which takes any stream made with
  // cuStreamCreate); the runtime's cudaMemcpyBatchAsync takes it.
  if (!hStream) return CUDA_ERROR_INVALID_VALUE;
  if (!count || !dsts || !srcs || !sizes || !attrs || !attrsIdxs || !numAttrs) return CUDA_ERROR_INVALID_VALUE;
  if (attrsIdxs[0] != 0) return CUDA_ERROR_INVALID_VALUE;
  const auto* a = static_cast<const MemcpyAttributesABI*>(attrs);
  for (size_t i = 0; i < numAttrs; ++i) {
    if (attrsIdxs[i] >= count || (i && attrsIdxs[i] <= attrsIdxs[i - 1])) return CUDA_ERROR_INVALID_VALUE;
    if (a[i].srcAccessOrder < 1 || a[i].srcAccessOrder > 3 || (a[i].flags & ~1u)) return CUDA_ERROR_INVALID_VALUE;
  }
  for (size_t i = 0; i < count; ++i)
    if (!sizes[i]) return CUDA_ERROR_INVALID_VALUE;
  for (size_t i = 0; i < count; ++i)
    if (const CUresult r = cuMemcpyAsync(dsts[i], srcs[i], sizes[i], hStream); r != CUDA_SUCCESS) return r;
  return CUDA_SUCCESS;
}

// The same for boxes, each operand a pointer (with the length of a row and the height of a layer, in elements)
// or an array (with an offset): flags other than 0, a count of 0, a null list, a box with a dimension of 0, an
// operand type that does not exist and an invalid access order are CUDA_ERROR_INVALID_VALUE.
static CUresult cuMemcpy3DBatchAsync_v2_impl(size_t numOps, void* opList, unsigned long long flags, CUstream hStream);
VGPU_EXPORT CUresult cuMemcpy3DBatchAsync_v2(size_t numOps, void* opList, unsigned long long flags, CUstream hStream) { return traced("cuMemcpy3DBatchAsync_v2", cuMemcpy3DBatchAsync_v2_impl, numOps, opList, flags, hStream); }
static CUresult cuMemcpy3DBatchAsync_v2_impl(size_t numOps, void* opList, unsigned long long flags, CUstream hStream) {
  if (!hStream || !numOps || !opList || flags != 0) return CUDA_ERROR_INVALID_VALUE;   // the legacy default stream too, as above
  const auto* ops = static_cast<const Memcpy3DBatchOpABI*>(opList);
  for (size_t i = 0; i < numOps; ++i) {
    const Memcpy3DBatchOpABI& op = ops[i];
    if (!op.width || !op.height || !op.depth) return CUDA_ERROR_INVALID_VALUE;
    if (op.srcAccessOrder < 1 || op.srcAccessOrder > 3 || (op.flags & ~1u)) return CUDA_ERROR_INVALID_VALUE;
    for (const Memcpy3DOperandABI* o : {&op.src, &op.dst})
      if (o->type != 1 && o->type != 2) return CUDA_ERROR_INVALID_VALUE;
  }
  for (size_t i = 0; i < numOps; ++i) {
    const Memcpy3DBatchOpABI& op = ops[i];
    CUDA_MEMCPY3D c{};
    c.WidthInBytes = op.width; c.Height = op.height; c.Depth = op.depth;
    if (op.src.type == 1) {
      c.srcMemoryType = CU_MEMORYTYPE_UNIFIED;
      c.srcDevice = op.src.op.ptr.ptr;
      c.srcPitch = op.src.op.ptr.rowLength ? op.src.op.ptr.rowLength : op.width;
      c.srcHeight = op.src.op.ptr.layerHeight ? op.src.op.ptr.layerHeight : op.height;
    } else {
      c.srcMemoryType = CU_MEMORYTYPE_ARRAY;
      c.srcArray = static_cast<CUarray>(op.src.op.array.array);
      c.srcXInBytes = op.src.op.array.x; c.srcY = op.src.op.array.y; c.srcZ = op.src.op.array.z;
    }
    if (op.dst.type == 1) {
      c.dstMemoryType = CU_MEMORYTYPE_UNIFIED;
      c.dstDevice = op.dst.op.ptr.ptr;
      c.dstPitch = op.dst.op.ptr.rowLength ? op.dst.op.ptr.rowLength : op.width;
      c.dstHeight = op.dst.op.ptr.layerHeight ? op.dst.op.ptr.layerHeight : op.height;
    } else {
      c.dstMemoryType = CU_MEMORYTYPE_ARRAY;
      c.dstArray = static_cast<CUarray>(op.dst.op.array.array);
      c.dstXInBytes = op.dst.op.array.x; c.dstY = op.dst.op.array.y; c.dstZ = op.dst.op.array.z;
    }
    if (const CUresult r = cuMemcpy3DAsync_v2(&c, hStream); r != CUDA_SUCCESS) return r;
  }
  return CUDA_SUCCESS;
}

// ---- arrays -------------------------------------------------------------------------------------------------------------------
// As the runtime's: memory requirements and sparse properties are for sparse and deferred-mapped arrays, so any
// other array (every array here) is CUDA_ERROR_INVALID_VALUE, a null array CUDA_ERROR_INVALID_HANDLE and a
// null result CUDA_ERROR_INVALID_VALUE; a plane is for multi-planar arrays.
static CUresult cuArrayGetMemoryRequirements_impl(void* memoryRequirements, CUarray array, CUdevice device);
VGPU_EXPORT CUresult cuArrayGetMemoryRequirements(void* memoryRequirements, CUarray array, CUdevice device) { return traced("cuArrayGetMemoryRequirements", cuArrayGetMemoryRequirements_impl, memoryRequirements, array, device); }
static CUresult cuArrayGetMemoryRequirements_impl(void* memoryRequirements, CUarray array, CUdevice device) {
  return api("cuArrayGetMemoryRequirements", true, false, [&](ShimState& s) -> CUresult {
    (void)device;
    if (!memoryRequirements) return CUDA_ERROR_INVALID_VALUE;
    if (!array || !s.arrays.count(reinterpret_cast<uintptr_t>(array))) return CUDA_ERROR_INVALID_HANDLE;
    return CUDA_ERROR_INVALID_VALUE;
  });
}

static CUresult cuMipmappedArrayGetMemoryRequirements_impl(void* memoryRequirements, CUmipmappedArray mipmap, CUdevice device);
VGPU_EXPORT CUresult cuMipmappedArrayGetMemoryRequirements(void* memoryRequirements, CUmipmappedArray mipmap, CUdevice device) { return traced("cuMipmappedArrayGetMemoryRequirements", cuMipmappedArrayGetMemoryRequirements_impl, memoryRequirements, mipmap, device); }
static CUresult cuMipmappedArrayGetMemoryRequirements_impl(void* memoryRequirements, CUmipmappedArray mipmap, CUdevice device) {
  return api("cuMipmappedArrayGetMemoryRequirements", true, false, [&](ShimState& s) -> CUresult {
    (void)device;
    if (!mipmap || !s.mipmaps.count(reinterpret_cast<uintptr_t>(mipmap))) return CUDA_ERROR_INVALID_HANDLE;
    if (!memoryRequirements) return CUDA_ERROR_INVALID_HANDLE;
    return CUDA_ERROR_INVALID_VALUE;
  });
}

static CUresult cuArrayGetSparseProperties_impl(void* sparseProperties, CUarray array);
VGPU_EXPORT CUresult cuArrayGetSparseProperties(void* sparseProperties, CUarray array) { return traced("cuArrayGetSparseProperties", cuArrayGetSparseProperties_impl, sparseProperties, array); }
static CUresult cuArrayGetSparseProperties_impl(void* sparseProperties, CUarray array) {
  return api("cuArrayGetSparseProperties", true, false, [&](ShimState& s) -> CUresult {
    if (!array || !s.arrays.count(reinterpret_cast<uintptr_t>(array))) return CUDA_ERROR_INVALID_HANDLE;
    (void)sparseProperties;
    return CUDA_ERROR_INVALID_VALUE;
  });
}

static CUresult cuMipmappedArrayGetSparseProperties_impl(void* sparseProperties, CUmipmappedArray mipmap);
VGPU_EXPORT CUresult cuMipmappedArrayGetSparseProperties(void* sparseProperties, CUmipmappedArray mipmap) { return traced("cuMipmappedArrayGetSparseProperties", cuMipmappedArrayGetSparseProperties_impl, sparseProperties, mipmap); }
static CUresult cuMipmappedArrayGetSparseProperties_impl(void* sparseProperties, CUmipmappedArray mipmap) {
  return api("cuMipmappedArrayGetSparseProperties", true, false, [&](ShimState& s) -> CUresult {
    if (!mipmap || !s.mipmaps.count(reinterpret_cast<uintptr_t>(mipmap))) return CUDA_ERROR_INVALID_HANDLE;
    (void)sparseProperties;
    return CUDA_ERROR_INVALID_VALUE;
  });
}

static CUresult cuArrayGetPlane_impl(CUarray* pPlaneArray, CUarray hArray, unsigned int planeIdx);
VGPU_EXPORT CUresult cuArrayGetPlane(CUarray* pPlaneArray, CUarray hArray, unsigned int planeIdx) { return traced("cuArrayGetPlane", cuArrayGetPlane_impl, pPlaneArray, hArray, planeIdx); }
static CUresult cuArrayGetPlane_impl(CUarray* pPlaneArray, CUarray hArray, unsigned int planeIdx) {
  return api("cuArrayGetPlane", true, false, [&](ShimState& s) -> CUresult {
    (void)planeIdx;
    if (!hArray || !s.arrays.count(reinterpret_cast<uintptr_t>(hArray))) return CUDA_ERROR_INVALID_HANDLE;
    if (!pPlaneArray) return CUDA_ERROR_INVALID_VALUE;
    return CUDA_ERROR_INVALID_VALUE;   // no array here has planes
  });
}

// Sparse arrays (cuMemMapArrayAsync) do not exist here, so no map is valid: CUDA_ERROR_INVALID_VALUE for any, as an
// RTX 3060 answers a null list, an empty one and one with an entry. DMA-BUF export of an address range and
// the hardware decompression engine are not on the card (CUDA_ERROR_NOT_SUPPORTED, a null handle
// CUDA_ERROR_INVALID_VALUE).
static CUresult cuMemMapArrayAsync_impl(void* mapInfoList, unsigned int count, CUstream hStream);
VGPU_EXPORT CUresult cuMemMapArrayAsync(void* mapInfoList, unsigned int count, CUstream hStream) { return traced("cuMemMapArrayAsync", cuMemMapArrayAsync_impl, mapInfoList, count, hStream); }
static CUresult cuMemMapArrayAsync_impl(void* mapInfoList, unsigned int count, CUstream hStream) {
  return api("cuMemMapArrayAsync", true, false, [&](ShimState&) -> CUresult {
    (void)mapInfoList; (void)count; (void)hStream;
    return CUDA_ERROR_INVALID_VALUE;
  });
}

static CUresult cuMemGetHandleForAddressRange_impl(void* handle, unsigned long long dptr, size_t size, int handleType, unsigned long long flags);
VGPU_EXPORT CUresult cuMemGetHandleForAddressRange(void* handle, unsigned long long dptr, size_t size, int handleType, unsigned long long flags) { return traced("cuMemGetHandleForAddressRange", cuMemGetHandleForAddressRange_impl, handle, dptr, size, handleType, flags); }
static CUresult cuMemGetHandleForAddressRange_impl(void* handle, unsigned long long dptr, size_t size, int handleType, unsigned long long flags) {
  return api("cuMemGetHandleForAddressRange", true, false, [&](ShimState&) -> CUresult {
    (void)dptr; (void)size; (void)handleType; (void)flags;
    if (!handle) return CUDA_ERROR_INVALID_VALUE;
    return sweep_refuse("cuMemGetHandleForAddressRange", "there is no DMA-BUF to export memory as");
  });
}

static CUresult cuMemBatchDecompressAsync_impl(void* paramsArray, size_t count, unsigned int flags, size_t* errorIndex, CUstream stream);
VGPU_EXPORT CUresult cuMemBatchDecompressAsync(void* paramsArray, size_t count, unsigned int flags, size_t* errorIndex, CUstream stream) { return traced("cuMemBatchDecompressAsync", cuMemBatchDecompressAsync_impl, paramsArray, count, flags, errorIndex, stream); }
static CUresult cuMemBatchDecompressAsync_impl(void* paramsArray, size_t count, unsigned int flags, size_t* errorIndex, CUstream stream) {
  return api("cuMemBatchDecompressAsync", true, false, [&](ShimState&) -> CUresult {
    (void)paramsArray; (void)count; (void)flags; (void)errorIndex; (void)stream;
    return sweep_refuse("cuMemBatchDecompressAsync", "this device has no decompression engine (cudaDevAttrMemDecompressAlgorithmMask is 0)");
  });
}

// ---- memory pools by location, managed memory in batches ---------------------------------------------------------------
// As the runtime's cudaMemGetDefaultMemPool / GetMemPool / SetMemPool and the batch prefetch and discard calls
// (runtime_sweep.inc has the measurements): the pool is the device's, found by (location, allocation type); an
// allocation type other than pinned is CUDA_ERROR_INVALID_VALUE (managed: CUDA_ERROR_NOT_SUPPORTED), a device
// location that does not exist and a null pointer CUDA_ERROR_INVALID_VALUE; a host location has a pool on the card
// that is not modelled here.
namespace {
CUresult pool_location_device(ShimState& s, const char* api, const CUmemLocation* location, int type, int* device) {
  if (!location) return CUDA_ERROR_INVALID_VALUE;
  if (type == 2) return CUDA_ERROR_NOT_SUPPORTED;           // CU_MEM_ALLOCATION_TYPE_MANAGED
  if (type != 1) return CUDA_ERROR_INVALID_VALUE;           // pinned only
  if (location->type == CU_MEM_LOCATION_TYPE_HOST || location->type == CU_MEM_LOCATION_TYPE_HOST_NUMA)
    return sweep_refuse(api, "memory pools in host memory are not modelled");
  if (location->type != CU_MEM_LOCATION_TYPE_DEVICE) return CUDA_ERROR_INVALID_VALUE;
  if (location->id < 0 || location->id >= s.rt->device_count()) return CUDA_ERROR_INVALID_VALUE;
  *device = location->id;
  return CUDA_SUCCESS;
}
}  // namespace

static CUresult cuMemGetDefaultMemPool_impl(void** pool_out, CUmemLocation* location, int type);
VGPU_EXPORT CUresult cuMemGetDefaultMemPool(void** pool_out, CUmemLocation* location, int type) { return traced("cuMemGetDefaultMemPool", cuMemGetDefaultMemPool_impl, pool_out, location, type); }
static CUresult cuMemGetDefaultMemPool_impl(void** pool_out, CUmemLocation* location, int type) {
  return api("cuMemGetDefaultMemPool", true, false, [&](ShimState& s) -> CUresult {
    if (!pool_out) return CUDA_ERROR_INVALID_VALUE;
    int dev = 0;
    if (const CUresult r = pool_location_device(s, "cuMemGetDefaultMemPool", location, type, &dev); r != CUDA_SUCCESS) return r;
    *pool_out = &default_pool_for(s, dev);
    return CUDA_SUCCESS;
  });
}

static CUresult cuMemGetMemPool_impl(void** pool, CUmemLocation* location, int type);
VGPU_EXPORT CUresult cuMemGetMemPool(void** pool, CUmemLocation* location, int type) { return traced("cuMemGetMemPool", cuMemGetMemPool_impl, pool, location, type); }
static CUresult cuMemGetMemPool_impl(void** pool, CUmemLocation* location, int type) {
  return api("cuMemGetMemPool", true, false, [&](ShimState& s) -> CUresult {
    if (!pool) return CUDA_ERROR_INVALID_VALUE;
    int dev = 0;
    if (const CUresult r = pool_location_device(s, "cuMemGetMemPool", location, type, &dev); r != CUDA_SUCCESS) return r;
    *pool = &pool_for(s, dev);
    return CUDA_SUCCESS;
  });
}

static CUresult cuMemSetMemPool_impl(CUmemLocation* location, int type, void* pool);
VGPU_EXPORT CUresult cuMemSetMemPool(CUmemLocation* location, int type, void* pool) { return traced("cuMemSetMemPool", cuMemSetMemPool_impl, location, type, pool); }
static CUresult cuMemSetMemPool_impl(CUmemLocation* location, int type, void* pool) {
  return api("cuMemSetMemPool", true, false, [&](ShimState& s) -> CUresult {
    int dev = 0;
    if (const CUresult r = pool_location_device(s, "cuMemSetMemPool", location, type, &dev); r != CUDA_SUCCESS) return r;
    PoolRec* p = pool_from_handle(s, pool);
    if (!p || p->device != dev) return CUDA_ERROR_INVALID_VALUE;
    s.current_pool[dev] = p;
    return CUDA_SUCCESS;
  });
}

// Batches of prefetches and discards of managed memory: a device without concurrent managed access (an RTX 3060
// under WSL) answers all three CUDA_ERROR_INVALID_VALUE, even for valid ranges (measured). With it, the
// ranges must be managed memory (CUDA_ERROR_INVALID_VALUE otherwise), and a prefetch of a range to a location
// is accepted: nothing here is ever resident on a device to be moved or thrown away.
namespace {
CUresult managed_batch(ShimState& s, unsigned long long* dptrs, const size_t* sizes, size_t count, const CUmemLocation* locs,
                       const size_t* idxs, size_t numLocs, unsigned long long flags, bool need_locs) {
  if (!count || !dptrs || !sizes || flags != 0) return CUDA_ERROR_INVALID_VALUE;
  if (need_locs && (!locs || !idxs || !numLocs || idxs[0] != 0)) return CUDA_ERROR_INVALID_VALUE;
  if (!s.rt->device(current_device(s)).profile().cuda.concurrent_managed_access) return CUDA_ERROR_INVALID_VALUE;
  for (size_t i = 0; i < numLocs && need_locs; ++i)
    if (idxs[i] >= count || (i && idxs[i] <= idxs[i - 1])) return CUDA_ERROR_INVALID_VALUE;
  for (size_t i = 0; i < count; ++i)
    if (!sizes[i] || !managed_range(s, dptrs[i], sizes[i])) return CUDA_ERROR_INVALID_VALUE;
  if (need_locs)
    for (size_t i = 0; i < numLocs; ++i)
      if (const CUresult r = check_location(s, locs[i]); r != CUDA_SUCCESS) return r;
  return CUDA_SUCCESS;
}
}  // namespace

static CUresult cuMemPrefetchBatchAsync_impl(unsigned long long* dptrs, size_t* sizes, size_t count, CUmemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, CUstream hStream);
VGPU_EXPORT CUresult cuMemPrefetchBatchAsync(unsigned long long* dptrs, size_t* sizes, size_t count, CUmemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, CUstream hStream) { return traced("cuMemPrefetchBatchAsync", cuMemPrefetchBatchAsync_impl, dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, hStream); }
static CUresult cuMemPrefetchBatchAsync_impl(unsigned long long* dptrs, size_t* sizes, size_t count, CUmemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, CUstream hStream) {
  (void)hStream;
  return api("cuMemPrefetchBatchAsync", true, false, [&](ShimState& s) -> CUresult {
    return managed_batch(s, dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, true);
  });
}

static CUresult cuMemDiscardBatchAsync_impl(unsigned long long* dptrs, size_t* sizes, size_t count, unsigned long long flags, CUstream hStream);
VGPU_EXPORT CUresult cuMemDiscardBatchAsync(unsigned long long* dptrs, size_t* sizes, size_t count, unsigned long long flags, CUstream hStream) { return traced("cuMemDiscardBatchAsync", cuMemDiscardBatchAsync_impl, dptrs, sizes, count, flags, hStream); }
static CUresult cuMemDiscardBatchAsync_impl(unsigned long long* dptrs, size_t* sizes, size_t count, unsigned long long flags, CUstream hStream) {
  (void)hStream;
  return api("cuMemDiscardBatchAsync", true, false, [&](ShimState& s) -> CUresult {
    return managed_batch(s, dptrs, sizes, count, nullptr, nullptr, 0, flags, false);
  });
}

static CUresult cuMemDiscardAndPrefetchBatchAsync_impl(unsigned long long* dptrs, size_t* sizes, size_t count, CUmemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, CUstream hStream);
VGPU_EXPORT CUresult cuMemDiscardAndPrefetchBatchAsync(unsigned long long* dptrs, size_t* sizes, size_t count, CUmemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, CUstream hStream) { return traced("cuMemDiscardAndPrefetchBatchAsync", cuMemDiscardAndPrefetchBatchAsync_impl, dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, hStream); }
static CUresult cuMemDiscardAndPrefetchBatchAsync_impl(unsigned long long* dptrs, size_t* sizes, size_t count, CUmemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, CUstream hStream) {
  (void)hStream;
  return api("cuMemDiscardAndPrefetchBatchAsync", true, false, [&](ShimState& s) -> CUresult {
    return managed_batch(s, dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, true);
  });
}

// What the range of managed memory has been advised and prefetched: on a device without concurrent managed access
// nothing is kept, so a range reads back the defaults the card reads back (measured on an RTX 3060: read-mostly 0,
// the preferred location, the last prefetch and every accessed-by entry the invalid device, -2). With it the driver
// shim would have to remember the advice it is given, which it does not: refused by name.
namespace {
CUresult range_attribute(ShimState& s, void* data, size_t size, int attribute, unsigned long long devPtr, size_t count) {
  if (!data) return CUDA_ERROR_INVALID_VALUE;
  if (!managed_range(s, devPtr, count)) return CUDA_ERROR_INVALID_VALUE;
  if (!s.rt->device(current_device(s)).profile().cuda.concurrent_managed_access) {
    switch (attribute) {   // CUmem_range_attribute: 1 read-mostly, 2 preferred, 3 accessed-by, 4 last prefetch
      case 1: if (size != sizeof(int)) return CUDA_ERROR_INVALID_VALUE; *static_cast<int*>(data) = 0; return CUDA_SUCCESS;
      case 2: case 4: if (size != sizeof(int)) return CUDA_ERROR_INVALID_VALUE; *static_cast<int*>(data) = -2; return CUDA_SUCCESS;
      case 3:
        if (size % sizeof(int)) return CUDA_ERROR_INVALID_VALUE;
        for (size_t i = 0; i < size / sizeof(int); ++i) static_cast<int*>(data)[i] = -2;
        return CUDA_SUCCESS;
      default: return CUDA_ERROR_INVALID_VALUE;
    }
  }
  return sweep_refuse("cuMemRangeGetAttribute", "the driver shim keeps no record of managed-memory advice");
}
}  // namespace

static CUresult cuMemRangeGetAttribute_impl(void* data, size_t dataSize, int attribute, unsigned long long devPtr, size_t count);
VGPU_EXPORT CUresult cuMemRangeGetAttribute(void* data, size_t dataSize, int attribute, unsigned long long devPtr, size_t count) { return traced("cuMemRangeGetAttribute", cuMemRangeGetAttribute_impl, data, dataSize, attribute, devPtr, count); }
static CUresult cuMemRangeGetAttribute_impl(void* data, size_t dataSize, int attribute, unsigned long long devPtr, size_t count) {
  return api("cuMemRangeGetAttribute", true, false, [&](ShimState& s) -> CUresult { return range_attribute(s, data, dataSize, attribute, devPtr, count); });
}

static CUresult cuMemRangeGetAttributes_impl(void** data, size_t* dataSizes, int* attributes, size_t numAttributes, unsigned long long devPtr, size_t count);
VGPU_EXPORT CUresult cuMemRangeGetAttributes(void** data, size_t* dataSizes, int* attributes, size_t numAttributes, unsigned long long devPtr, size_t count) { return traced("cuMemRangeGetAttributes", cuMemRangeGetAttributes_impl, data, dataSizes, attributes, numAttributes, devPtr, count); }
static CUresult cuMemRangeGetAttributes_impl(void** data, size_t* dataSizes, int* attributes, size_t numAttributes, unsigned long long devPtr, size_t count) {
  return api("cuMemRangeGetAttributes", true, false, [&](ShimState& s) -> CUresult {
    if (!data || !dataSizes || !attributes || !numAttributes) return CUDA_ERROR_INVALID_VALUE;
    for (size_t i = 0; i < numAttributes; ++i)
      if (const CUresult r = range_attribute(s, data[i], dataSizes[i], attributes[i], devPtr, count); r != CUDA_SUCCESS) return r;
    return CUDA_SUCCESS;
  });
}

// ---- streams -----------------------------------------------------------------------------------------------------------------
// The device and the id of a stream (the null stream is the current context's default: its device, and an id of its own),
// and the attributes a stream carries (the same five as the runtime's: access policy window, synchronization
// policy, priority, memory-sync domain and its map). Measured on an RTX 3060: reading an attribute that was
// never set gives zeros, except the synchronization policy, which reads -1; priority reads back what the stream
// was made with; a value pointer that is null is CUDA_ERROR_INVALID_VALUE (here; the card crashes), and so is an
// attribute that is not one of the five.
namespace {
struct StreamAttrs {
  std::map<int, std::array<uint8_t, 64>> values;
};
std::mutex g_stream_attr_mu;
std::map<uintptr_t, StreamAttrs> g_stream_attrs;   // by stream handle; 0 is the default stream
bool stream_attr_id(int a) { return a == 1 || a == 3 || a == 8 || a == 9 || a == 10; }
}  // namespace

static CUresult cuStreamGetDevice_impl(CUstream hStream, CUdevice* device);
VGPU_EXPORT CUresult cuStreamGetDevice(CUstream hStream, CUdevice* device) { return traced("cuStreamGetDevice", cuStreamGetDevice_impl, hStream, device); }
static CUresult cuStreamGetDevice_impl(CUstream hStream, CUdevice* device) {
  return api("cuStreamGetDevice", true, false, [&](ShimState& s) -> CUresult {
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    const uintptr_t h = reinterpret_cast<uintptr_t>(hStream);
    if (h && (h & 7) == kTagStream) {
      const auto it = s.stream_device.find(h);
      if (it == s.stream_device.end() && !s.streams.count(h)) return CUDA_ERROR_INVALID_HANDLE;
      *device = it != s.stream_device.end() ? it->second : current_device(s);
      return CUDA_SUCCESS;
    }
    if (h) return CUDA_ERROR_INVALID_HANDLE;
    *device = current_device(s);
    return CUDA_SUCCESS;
  });
}

static CUresult cuStreamGetId_impl(CUstream hStream, unsigned long long* streamId);
VGPU_EXPORT CUresult cuStreamGetId(CUstream hStream, unsigned long long* streamId) { return traced("cuStreamGetId", cuStreamGetId_impl, hStream, streamId); }
static CUresult cuStreamGetId_impl(CUstream hStream, unsigned long long* streamId) {
  return api("cuStreamGetId", true, false, [&](ShimState& s) -> CUresult {
    if (!streamId) return CUDA_ERROR_INVALID_VALUE;
    const uintptr_t h = reinterpret_cast<uintptr_t>(hStream);
    if (h && (h & 7) != kTagStream) return CUDA_ERROR_INVALID_HANDLE;
    if (h && !s.streams.count(h)) return CUDA_ERROR_INVALID_HANDLE;
    *streamId = h >> 3;   // the handle's own number: distinct for every stream, 0 for the default one
    return CUDA_SUCCESS;
  });
}

static CUresult cuStreamGetAttribute_impl(CUstream hStream, int attr, void* value_out);
VGPU_EXPORT CUresult cuStreamGetAttribute(CUstream hStream, int attr, void* value_out) { return traced("cuStreamGetAttribute", cuStreamGetAttribute_impl, hStream, attr, value_out); }
static CUresult cuStreamGetAttribute_impl(CUstream hStream, int attr, void* value_out) {
  return api("cuStreamGetAttribute", true, false, [&](ShimState& s) -> CUresult {
    if (!value_out || !stream_attr_id(attr)) return CUDA_ERROR_INVALID_VALUE;
    const uintptr_t h = reinterpret_cast<uintptr_t>(hStream);
    if (h && !s.streams.count(h)) return CUDA_ERROR_INVALID_HANDLE;
    std::array<uint8_t, 64> v{};
    if (attr == 3) {   // synchronization policy: -1 until set
      const int unset = -1;
      std::memcpy(v.data(), &unset, sizeof unset);
    }
    if (attr == 8) {   // priority: the stream's
      const auto p = s.stream_priority.find(h);
      const int prio = p == s.stream_priority.end() ? 0 : p->second;
      std::memcpy(v.data(), &prio, sizeof prio);
    }
    {
      std::lock_guard<std::mutex> lock(g_stream_attr_mu);
      const auto rec = g_stream_attrs.find(h);
      if (rec != g_stream_attrs.end()) {
        const auto a = rec->second.values.find(attr);
        if (a != rec->second.values.end()) v = a->second;
      }
    }
    std::memcpy(value_out, v.data(), v.size());
    return CUDA_SUCCESS;
  });
}

static CUresult cuStreamSetAttribute_impl(CUstream hStream, int attr, const void* value);
VGPU_EXPORT CUresult cuStreamSetAttribute(CUstream hStream, int attr, const void* value) { return traced("cuStreamSetAttribute", cuStreamSetAttribute_impl, hStream, attr, value); }
static CUresult cuStreamSetAttribute_impl(CUstream hStream, int attr, const void* value) {
  return api("cuStreamSetAttribute", true, false, [&](ShimState& s) -> CUresult {
    if (!value || !stream_attr_id(attr)) return CUDA_ERROR_INVALID_VALUE;
    const uintptr_t h = reinterpret_cast<uintptr_t>(hStream);
    if (h && !s.streams.count(h)) return CUDA_ERROR_INVALID_HANDLE;
    std::array<uint8_t, 64> v{};
    std::memcpy(v.data(), value, v.size());
    if (attr == 8) {   // priority: clamped to the range, as cuStreamCreateWithPriority clamps it
      int prio = 0;
      std::memcpy(&prio, value, sizeof prio);
      s.stream_priority[h] = std::min(vgpu::cuda::kLeastStreamPriority, std::max(vgpu::cuda::kGreatestStreamPriority, prio));
      return CUDA_SUCCESS;
    }
    std::lock_guard<std::mutex> lock(g_stream_attr_mu);
    g_stream_attrs[h].values[attr] = v;
    return CUDA_SUCCESS;
  });
}

// Copies the attributes of one stream to another (the null stream is the default stream, whose attributes it also
// has); both must exist.
static CUresult cuStreamCopyAttributes_impl(CUstream dst, CUstream src);
VGPU_EXPORT CUresult cuStreamCopyAttributes(CUstream dst, CUstream src) { return traced("cuStreamCopyAttributes", cuStreamCopyAttributes_impl, dst, src); }
static CUresult cuStreamCopyAttributes_impl(CUstream dst, CUstream src) {
  return api("cuStreamCopyAttributes", true, false, [&](ShimState& s) -> CUresult {
    const uintptr_t d = reinterpret_cast<uintptr_t>(dst), f = reinterpret_cast<uintptr_t>(src);
    if ((d && !s.streams.count(d)) || (f && !s.streams.count(f))) return CUDA_ERROR_INVALID_HANDLE;
    if (d == f) return CUDA_SUCCESS;
    StreamAttrs from;
    {
      std::lock_guard<std::mutex> lock(g_stream_attr_mu);
      const auto it = g_stream_attrs.find(f);
      if (it != g_stream_attrs.end()) from = it->second;
      g_stream_attrs[d] = from;
    }
    const auto p = s.stream_priority.find(f);
    if (p != s.stream_priority.end()) s.stream_priority[d] = p->second;
    else s.stream_priority.erase(d);
    return CUDA_SUCCESS;
  });
}

// ---- texture references: the mipmap state, the anisotropy, the border colour ----------------------------------
// What cuTexRefSet* records and cuTexRefGet* reads back (TexRefRec), as the card does: a mipmapped array set on the
// reference is the one read back, and reading it before is CUDA_ERROR_INVALID_VALUE and leaves the result alone; the
// mipmap filter mode is point (0) or linear (1); the bias, the clamp, the anisotropy (0 until set) and the border
// colour (zeros until set) are kept as given. A reference that does not exist is CUDA_ERROR_INVALID_HANDLE. None of
// it acts on a fetch: nothing launches with a texture reference (see TexRefRec).
static CUresult cuTexRefSetMipmappedArray_impl(CUtexref t, CUmipmappedArray m, unsigned int flags);
VGPU_EXPORT CUresult cuTexRefSetMipmappedArray(CUtexref t, CUmipmappedArray m, unsigned int flags) { return traced("cuTexRefSetMipmappedArray", cuTexRefSetMipmappedArray_impl, t, m, flags); }
static CUresult cuTexRefSetMipmappedArray_impl(CUtexref t, CUmipmappedArray m, unsigned int flags) {
  return with_texref("cuTexRefSetMipmappedArray", t, [&](ShimState& s, TexRefRec& r) -> CUresult {
    const unsigned known = CU_TRSF_READ_AS_INTEGER | CU_TRSF_NORMALIZED_COORDINATES | CU_TRSF_SRGB |
                           CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION | CU_TRSF_SEAMLESS_CUBEMAP;
    if (!m || !s.mipmaps.count(reinterpret_cast<uintptr_t>(m))) return CUDA_ERROR_INVALID_HANDLE;
    if (flags & ~known) return CUDA_ERROR_INVALID_VALUE;
    r.mip_array = m;
    r.flags = flags;
    return CUDA_SUCCESS;
  });
}

static CUresult cuTexRefGetMipmappedArray_impl(CUmipmappedArray* out, CUtexref t);
VGPU_EXPORT CUresult cuTexRefGetMipmappedArray(CUmipmappedArray* out, CUtexref t) { return traced("cuTexRefGetMipmappedArray", cuTexRefGetMipmappedArray_impl, out, t); }
static CUresult cuTexRefGetMipmappedArray_impl(CUmipmappedArray* out, CUtexref t) {
  return with_texref("cuTexRefGetMipmappedArray", t, [&](ShimState&, TexRefRec& r) -> CUresult {
    if (!out || !r.mip_array) return CUDA_ERROR_INVALID_VALUE;
    *out = r.mip_array;
    return CUDA_SUCCESS;
  });
}

static CUresult cuTexRefSetMipmapFilterMode_impl(CUtexref t, int mode);
VGPU_EXPORT CUresult cuTexRefSetMipmapFilterMode(CUtexref t, int mode) { return traced("cuTexRefSetMipmapFilterMode", cuTexRefSetMipmapFilterMode_impl, t, mode); }
static CUresult cuTexRefSetMipmapFilterMode_impl(CUtexref t, int mode) {
  return with_texref("cuTexRefSetMipmapFilterMode", t, [&](ShimState&, TexRefRec& r) -> CUresult {
    if (mode < 0 || mode > 1) return CUDA_ERROR_INVALID_VALUE;
    r.mip_filter = mode;
    return CUDA_SUCCESS;
  });
}

static CUresult cuTexRefGetMipmapFilterMode_impl(int* mode, CUtexref t);
VGPU_EXPORT CUresult cuTexRefGetMipmapFilterMode(int* mode, CUtexref t) { return traced("cuTexRefGetMipmapFilterMode", cuTexRefGetMipmapFilterMode_impl, mode, t); }
static CUresult cuTexRefGetMipmapFilterMode_impl(int* mode, CUtexref t) {
  return with_texref("cuTexRefGetMipmapFilterMode", t, [&](ShimState&, TexRefRec& r) -> CUresult {
    if (!mode) return CUDA_ERROR_INVALID_VALUE;
    *mode = r.mip_filter;
    return CUDA_SUCCESS;
  });
}

static CUresult cuTexRefSetMipmapLevelBias_impl(CUtexref t, float bias);
VGPU_EXPORT CUresult cuTexRefSetMipmapLevelBias(CUtexref t, float bias) { return traced("cuTexRefSetMipmapLevelBias", cuTexRefSetMipmapLevelBias_impl, t, bias); }
static CUresult cuTexRefSetMipmapLevelBias_impl(CUtexref t, float bias) {
  return with_texref("cuTexRefSetMipmapLevelBias", t, [&](ShimState&, TexRefRec& r) -> CUresult {
    r.mip_bias = bias;
    return CUDA_SUCCESS;
  });
}

static CUresult cuTexRefGetMipmapLevelBias_impl(float* bias, CUtexref t);
VGPU_EXPORT CUresult cuTexRefGetMipmapLevelBias(float* bias, CUtexref t) { return traced("cuTexRefGetMipmapLevelBias", cuTexRefGetMipmapLevelBias_impl, bias, t); }
static CUresult cuTexRefGetMipmapLevelBias_impl(float* bias, CUtexref t) {
  return with_texref("cuTexRefGetMipmapLevelBias", t, [&](ShimState&, TexRefRec& r) -> CUresult {
    if (!bias) return CUDA_ERROR_INVALID_VALUE;
    *bias = r.mip_bias;
    return CUDA_SUCCESS;
  });
}

static CUresult cuTexRefSetMipmapLevelClamp_impl(CUtexref t, float minClamp, float maxClamp);
VGPU_EXPORT CUresult cuTexRefSetMipmapLevelClamp(CUtexref t, float minMipmapLevelClamp, float maxMipmapLevelClamp) { return traced("cuTexRefSetMipmapLevelClamp", cuTexRefSetMipmapLevelClamp_impl, t, minMipmapLevelClamp, maxMipmapLevelClamp); }
static CUresult cuTexRefSetMipmapLevelClamp_impl(CUtexref t, float minClamp, float maxClamp) {
  return with_texref("cuTexRefSetMipmapLevelClamp", t, [&](ShimState&, TexRefRec& r) -> CUresult {
    r.mip_clamp[0] = minClamp;
    r.mip_clamp[1] = maxClamp;
    return CUDA_SUCCESS;
  });
}

static CUresult cuTexRefGetMipmapLevelClamp_impl(float* minClamp, float* maxClamp, CUtexref t);
VGPU_EXPORT CUresult cuTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref t) { return traced("cuTexRefGetMipmapLevelClamp", cuTexRefGetMipmapLevelClamp_impl, pminMipmapLevelClamp, pmaxMipmapLevelClamp, t); }
static CUresult cuTexRefGetMipmapLevelClamp_impl(float* minClamp, float* maxClamp, CUtexref t) {
  return with_texref("cuTexRefGetMipmapLevelClamp", t, [&](ShimState&, TexRefRec& r) -> CUresult {
    if (!minClamp || !maxClamp) return CUDA_ERROR_INVALID_VALUE;
    *minClamp = r.mip_clamp[0];
    *maxClamp = r.mip_clamp[1];
    return CUDA_SUCCESS;
  });
}

static CUresult cuTexRefSetMaxAnisotropy_impl(CUtexref t, unsigned int maxAniso);
VGPU_EXPORT CUresult cuTexRefSetMaxAnisotropy(CUtexref t, unsigned int maxAniso) { return traced("cuTexRefSetMaxAnisotropy", cuTexRefSetMaxAnisotropy_impl, t, maxAniso); }
static CUresult cuTexRefSetMaxAnisotropy_impl(CUtexref t, unsigned int maxAniso) {
  return with_texref("cuTexRefSetMaxAnisotropy", t, [&](ShimState&, TexRefRec& r) -> CUresult {
    r.max_anisotropy = maxAniso;
    return CUDA_SUCCESS;
  });
}

static CUresult cuTexRefGetMaxAnisotropy_impl(int* maxAniso, CUtexref t);
VGPU_EXPORT CUresult cuTexRefGetMaxAnisotropy(int* pmaxAniso, CUtexref t) { return traced("cuTexRefGetMaxAnisotropy", cuTexRefGetMaxAnisotropy_impl, pmaxAniso, t); }
static CUresult cuTexRefGetMaxAnisotropy_impl(int* maxAniso, CUtexref t) {
  return with_texref("cuTexRefGetMaxAnisotropy", t, [&](ShimState&, TexRefRec& r) -> CUresult {
    if (!maxAniso) return CUDA_ERROR_INVALID_VALUE;
    *maxAniso = static_cast<int>(r.max_anisotropy);
    return CUDA_SUCCESS;
  });
}

static CUresult cuTexRefSetBorderColor_impl(CUtexref t, float* color);
VGPU_EXPORT CUresult cuTexRefSetBorderColor(CUtexref t, float* pBorderColor) { return traced("cuTexRefSetBorderColor", cuTexRefSetBorderColor_impl, t, pBorderColor); }
static CUresult cuTexRefSetBorderColor_impl(CUtexref t, float* color) {
  return with_texref("cuTexRefSetBorderColor", t, [&](ShimState&, TexRefRec& r) -> CUresult {
    if (!color) return CUDA_ERROR_INVALID_VALUE;
    std::memcpy(r.border, color, sizeof r.border);
    return CUDA_SUCCESS;
  });
}

static CUresult cuTexRefGetBorderColor_impl(float* color, CUtexref t);
VGPU_EXPORT CUresult cuTexRefGetBorderColor(float* pBorderColor, CUtexref t) { return traced("cuTexRefGetBorderColor", cuTexRefGetBorderColor_impl, pBorderColor, t); }
static CUresult cuTexRefGetBorderColor_impl(float* color, CUtexref t) {
  return with_texref("cuTexRefGetBorderColor", t, [&](ShimState&, TexRefRec& r) -> CUresult {
    if (!color) return CUDA_ERROR_INVALID_VALUE;
    std::memcpy(color, r.border, sizeof r.border);
    return CUDA_SUCCESS;
  });
}

// ---- texture and surface objects: what they were made from -------------------------------------------------------------
// The descriptors the program passed to cuTexObjectCreate / cuSurfObjectCreate, handed back. A texture made
// without a view descriptor reads back zeros from cuTexObjectGetResourceViewDesc, and one made without a texture
// descriptor from cuTexObjectGetTextureDesc (measured). An object that does not exist, or a null result, is
// CUDA_ERROR_INVALID_VALUE.
static CUresult cuTexObjectGetResourceDesc_impl(void* pResDesc, unsigned long long texObject);
VGPU_EXPORT CUresult cuTexObjectGetResourceDesc(void* pResDesc, unsigned long long texObject) { return traced("cuTexObjectGetResourceDesc", cuTexObjectGetResourceDesc_impl, pResDesc, texObject); }
static CUresult cuTexObjectGetResourceDesc_impl(void* pResDesc, unsigned long long texObject) {
  return api("cuTexObjectGetResourceDesc", true, false, [&](ShimState&) -> CUresult {
    if (!pResDesc) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::mutex> lock(g_object_mu);
    const auto it = g_object_descs.find(texObject);
    if (it == g_object_descs.end()) return CUDA_ERROR_INVALID_VALUE;
    std::memcpy(pResDesc, &it->second.res, sizeof it->second.res);
    return CUDA_SUCCESS;
  });
}

static CUresult cuTexObjectGetTextureDesc_impl(void* pTexDesc, unsigned long long texObject);
VGPU_EXPORT CUresult cuTexObjectGetTextureDesc(void* pTexDesc, unsigned long long texObject) { return traced("cuTexObjectGetTextureDesc", cuTexObjectGetTextureDesc_impl, pTexDesc, texObject); }
static CUresult cuTexObjectGetTextureDesc_impl(void* pTexDesc, unsigned long long texObject) {
  return api("cuTexObjectGetTextureDesc", true, false, [&](ShimState&) -> CUresult {
    if (!pTexDesc) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::mutex> lock(g_object_mu);
    const auto it = g_object_descs.find(texObject);
    if (it == g_object_descs.end()) return CUDA_ERROR_INVALID_VALUE;
    std::memcpy(pTexDesc, &it->second.tex, sizeof it->second.tex);
    return CUDA_SUCCESS;
  });
}

static CUresult cuTexObjectGetResourceViewDesc_impl(void* pResViewDesc, unsigned long long texObject);
VGPU_EXPORT CUresult cuTexObjectGetResourceViewDesc(void* pResViewDesc, unsigned long long texObject) { return traced("cuTexObjectGetResourceViewDesc", cuTexObjectGetResourceViewDesc_impl, pResViewDesc, texObject); }
static CUresult cuTexObjectGetResourceViewDesc_impl(void* pResViewDesc, unsigned long long texObject) {
  return api("cuTexObjectGetResourceViewDesc", true, false, [&](ShimState&) -> CUresult {
    if (!pResViewDesc) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::mutex> lock(g_object_mu);
    const auto it = g_object_descs.find(texObject);
    if (it == g_object_descs.end()) return CUDA_ERROR_INVALID_VALUE;
    std::memcpy(pResViewDesc, &it->second.view, sizeof it->second.view);
    return CUDA_SUCCESS;
  });
}

static CUresult cuSurfObjectGetResourceDesc_impl(void* pResDesc, unsigned long long surfObject);
VGPU_EXPORT CUresult cuSurfObjectGetResourceDesc(void* pResDesc, unsigned long long surfObject) { return traced("cuSurfObjectGetResourceDesc", cuSurfObjectGetResourceDesc_impl, pResDesc, surfObject); }
static CUresult cuSurfObjectGetResourceDesc_impl(void* pResDesc, unsigned long long surfObject) {
  return cuTexObjectGetResourceDesc_impl(pResDesc, surfObject);
}

// ---- Blackwell's wide im2col tensor map ---------------------------------------------------------------------------------------
// cuTensorMapEncodeIm2colWide is sm_100's: an RTX 3060 answers CUDA_ERROR_NOT_SUPPORTED (a null tensor map or global
// address CUDA_ERROR_INVALID_VALUE). cuTensorMapEncodeIm2col is implemented; the wide form's box layout is not
// derived here, so it is refused on every profile.
static CUresult cuTensorMapEncodeIm2colWide_impl(void* tensorMap, int tensorDataType, unsigned tensorRank, void* globalAddress, const unsigned long long* globalDim, const unsigned long long* globalStrides, int pixelBoxLowerCornerWidth, int pixelBoxUpperCornerWidth, unsigned channelsPerPixel, unsigned pixelsPerColumn, const unsigned* elementStrides, int interleave, int mode, int swizzle, int l2Promotion, int oobFill);
VGPU_EXPORT CUresult cuTensorMapEncodeIm2colWide(void* tensorMap, int tensorDataType, unsigned int tensorRank, void* globalAddress, const unsigned long long* globalDim, const unsigned long long* globalStrides, int pixelBoxLowerCornerWidth, int pixelBoxUpperCornerWidth, unsigned int channelsPerPixel, unsigned int pixelsPerColumn, const unsigned int* elementStrides, int interleave, int mode, int swizzle, int l2Promotion, int oobFill) { return traced("cuTensorMapEncodeIm2colWide", cuTensorMapEncodeIm2colWide_impl, tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCornerWidth, pixelBoxUpperCornerWidth, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, mode, swizzle, l2Promotion, oobFill); }
static CUresult cuTensorMapEncodeIm2colWide_impl(void* tensorMap, int tensorDataType, unsigned tensorRank, void* globalAddress, const unsigned long long* globalDim, const unsigned long long* globalStrides, int pixelBoxLowerCornerWidth, int pixelBoxUpperCornerWidth, unsigned channelsPerPixel, unsigned pixelsPerColumn, const unsigned* elementStrides, int interleave, int mode, int swizzle, int l2Promotion, int oobFill) {
  return api("cuTensorMapEncodeIm2colWide", true, false, [&](ShimState&) -> CUresult {
    (void)tensorDataType; (void)tensorRank; (void)pixelBoxLowerCornerWidth; (void)pixelBoxUpperCornerWidth;
    (void)channelsPerPixel; (void)pixelsPerColumn; (void)interleave; (void)mode; (void)swizzle; (void)l2Promotion; (void)oobFill;
    if (!tensorMap || !globalAddress || !globalDim || !globalStrides || !elementStrides) return CUDA_ERROR_INVALID_VALUE;
    return sweep_refuse("cuTensorMapEncodeIm2colWide", "the wide im2col tensor map is Blackwell's and is not simulated");
  });
}

// ---- multi-device cooperative launches (deprecated) ----------------------------------------------------------------------------
// cudaDevAttrCooperativeMultiDeviceLaunch is 0 for every profile here: a count of 0 and a null list are
// CUDA_ERROR_INVALID_VALUE, a function that is not one CUDA_ERROR_INVALID_HANDLE (both measured), and the rest
// CUDA_ERROR_NOT_SUPPORTED.
static CUresult cuLaunchCooperativeKernelMultiDevice_impl(void* launchParamsList, unsigned int numDevices, unsigned int flags);
VGPU_EXPORT CUresult cuLaunchCooperativeKernelMultiDevice(void* launchParamsList, unsigned int numDevices, unsigned int flags) { return traced("cuLaunchCooperativeKernelMultiDevice", cuLaunchCooperativeKernelMultiDevice_impl, launchParamsList, numDevices, flags); }
static CUresult cuLaunchCooperativeKernelMultiDevice_impl(void* launchParamsList, unsigned int numDevices, unsigned int flags) {
  return api("cuLaunchCooperativeKernelMultiDevice", true, false, [&](ShimState& s) -> CUresult {
    (void)flags;
    if (!launchParamsList || numDevices == 0) return CUDA_ERROR_INVALID_VALUE;
    struct LaunchParams { CUfunction function; unsigned gx, gy, gz, bx, by, bz, shared; CUstream stream; void** params; };
    const auto* lp = static_cast<const LaunchParams*>(launchParamsList);
    for (unsigned i = 0; i < numDevices; ++i)
      if (!lp[i].function || !s.functions.count(reinterpret_cast<uintptr_t>(lp[i].function))) return CUDA_ERROR_INVALID_HANDLE;
    return sweep_refuse("cuLaunchCooperativeKernelMultiDevice", "multi-device cooperative launch is deprecated and cudaDevAttrCooperativeMultiDeviceLaunch is 0");
  });
}

// ---- external memory and semaphores -----------------------------------------------------------------------------------------------
// Interop with Vulkan and Direct3D: an RTX 3060 under WSL answers CUDA_ERROR_NOT_SUPPORTED for an opaque-descriptor
// import and CUDA_ERROR_INVALID_VALUE for a Windows handle type, a type that does not exist, a null descriptor and a
// null result (measured); the calls that take an imported object are CUDA_ERROR_INVALID_VALUE for a null one
// (a garbage handle crashes the card; here CUDA_ERROR_INVALID_HANDLE). No object is ever made here.
static CUresult cuImportExternalMemory_impl(void** extMem_out, const void* memHandleDesc);
VGPU_EXPORT CUresult cuImportExternalMemory(void** extMem_out, const void* memHandleDesc) { return traced("cuImportExternalMemory", cuImportExternalMemory_impl, extMem_out, memHandleDesc); }
static CUresult cuImportExternalMemory_impl(void** extMem_out, const void* memHandleDesc) {
  return api("cuImportExternalMemory", true, false, [&](ShimState&) -> CUresult {
    if (!extMem_out || !memHandleDesc) return CUDA_ERROR_INVALID_VALUE;
    const int type = *static_cast<const int*>(memHandleDesc);   // CUexternalMemoryHandleType is the first field
    if (type != 1) return CUDA_ERROR_INVALID_VALUE;               // only the opaque file descriptor is a Linux type
    return sweep_refuse("cuImportExternalMemory", "external memory (Vulkan, Direct3D) is not simulated");
  });
}

static CUresult cuExternalMemoryGetMappedBuffer_impl(unsigned long long* devPtr, void* extMem, const void* bufferDesc);
VGPU_EXPORT CUresult cuExternalMemoryGetMappedBuffer(unsigned long long* devPtr, void* extMem, const void* bufferDesc) { return traced("cuExternalMemoryGetMappedBuffer", cuExternalMemoryGetMappedBuffer_impl, devPtr, extMem, bufferDesc); }
static CUresult cuExternalMemoryGetMappedBuffer_impl(unsigned long long* devPtr, void* extMem, const void* bufferDesc) {
  return api("cuExternalMemoryGetMappedBuffer", true, false, [&](ShimState&) -> CUresult {
    (void)bufferDesc;
    return !devPtr || !extMem ? CUDA_ERROR_INVALID_VALUE : CUDA_ERROR_INVALID_HANDLE;
  });
}

static CUresult cuExternalMemoryGetMappedMipmappedArray_impl(CUmipmappedArray* mipmap, void* extMem, const void* mipmapDesc);
VGPU_EXPORT CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray* mipmap, void* extMem, const void* mipmapDesc) { return traced("cuExternalMemoryGetMappedMipmappedArray", cuExternalMemoryGetMappedMipmappedArray_impl, mipmap, extMem, mipmapDesc); }
static CUresult cuExternalMemoryGetMappedMipmappedArray_impl(CUmipmappedArray* mipmap, void* extMem, const void* mipmapDesc) {
  return api("cuExternalMemoryGetMappedMipmappedArray", true, false, [&](ShimState&) -> CUresult {
    (void)mipmapDesc;
    return !mipmap || !extMem ? CUDA_ERROR_INVALID_VALUE : CUDA_ERROR_INVALID_HANDLE;
  });
}

static CUresult cuDestroyExternalMemory_impl(void* extMem);
VGPU_EXPORT CUresult cuDestroyExternalMemory(void* extMem) { return traced("cuDestroyExternalMemory", cuDestroyExternalMemory_impl, extMem); }
static CUresult cuDestroyExternalMemory_impl(void* extMem) {
  return api("cuDestroyExternalMemory", true, false, [&](ShimState&) -> CUresult { return extMem ? CUDA_ERROR_INVALID_HANDLE : CUDA_ERROR_INVALID_VALUE; });
}

static CUresult cuImportExternalSemaphore_impl(void** extSem_out, const void* semHandleDesc);
VGPU_EXPORT CUresult cuImportExternalSemaphore(void** extSem_out, const void* semHandleDesc) { return traced("cuImportExternalSemaphore", cuImportExternalSemaphore_impl, extSem_out, semHandleDesc); }
static CUresult cuImportExternalSemaphore_impl(void** extSem_out, const void* semHandleDesc) {
  return api("cuImportExternalSemaphore", true, false, [&](ShimState&) -> CUresult {
    if (!extSem_out || !semHandleDesc) return CUDA_ERROR_INVALID_VALUE;
    const int type = *static_cast<const int*>(semHandleDesc);
    if (type != 1) return CUDA_ERROR_INVALID_VALUE;
    return sweep_refuse("cuImportExternalSemaphore", "external semaphores (Vulkan, Direct3D) are not simulated");
  });
}

static CUresult cuSignalExternalSemaphoresAsync_impl(const void* const* extSemArray, const void* paramsArray, unsigned int numExtSems, CUstream stream);
VGPU_EXPORT CUresult cuSignalExternalSemaphoresAsync(const void* const* extSemArray, const void* paramsArray, unsigned int numExtSems, CUstream stream) { return traced("cuSignalExternalSemaphoresAsync", cuSignalExternalSemaphoresAsync_impl, extSemArray, paramsArray, numExtSems, stream); }
static CUresult cuSignalExternalSemaphoresAsync_impl(const void* const* extSemArray, const void* paramsArray, unsigned int numExtSems, CUstream stream) {
  return api("cuSignalExternalSemaphoresAsync", true, false, [&](ShimState&) -> CUresult {
    (void)stream;
    if (!extSemArray || !paramsArray || numExtSems == 0) return CUDA_ERROR_INVALID_VALUE;
    return CUDA_ERROR_INVALID_HANDLE;
  });
}

static CUresult cuWaitExternalSemaphoresAsync_impl(const void* const* extSemArray, const void* paramsArray, unsigned int numExtSems, CUstream stream);
VGPU_EXPORT CUresult cuWaitExternalSemaphoresAsync(const void* const* extSemArray, const void* paramsArray, unsigned int numExtSems, CUstream stream) { return traced("cuWaitExternalSemaphoresAsync", cuWaitExternalSemaphoresAsync_impl, extSemArray, paramsArray, numExtSems, stream); }
static CUresult cuWaitExternalSemaphoresAsync_impl(const void* const* extSemArray, const void* paramsArray, unsigned int numExtSems, CUstream stream) {
  return api("cuWaitExternalSemaphoresAsync", true, false, [&](ShimState&) -> CUresult {
    (void)stream;
    if (!extSemArray || !paramsArray || numExtSems == 0) return CUDA_ERROR_INVALID_VALUE;
    return CUDA_ERROR_INVALID_HANDLE;
  });
}

static CUresult cuDestroyExternalSemaphore_impl(void* extSem);
VGPU_EXPORT CUresult cuDestroyExternalSemaphore(void* extSem) { return traced("cuDestroyExternalSemaphore", cuDestroyExternalSemaphore_impl, extSem); }
static CUresult cuDestroyExternalSemaphore_impl(void* extSem) {
  return api("cuDestroyExternalSemaphore", true, false, [&](ShimState&) -> CUresult { return extSem ? CUDA_ERROR_INVALID_HANDLE : CUDA_ERROR_INVALID_VALUE; });
}

// The external-semaphore graph nodes: no such node exists to read or change (cuGraphAddNode cannot make one here), so a
// null node is CUDA_ERROR_INVALID_VALUE, as measured, and any other CUDA_ERROR_NOT_SUPPORTED.
#define VGPU_EXT_SEM_NODE(NAME, SIG, ARGS, NODEARG)                                                              \
  static CUresult NAME##_impl SIG;                                                                              \
  VGPU_EXPORT CUresult NAME SIG { return traced(#NAME, NAME##_impl, ARGS); }                                    \
  static CUresult NAME##_impl SIG {                                                                             \
    return api(#NAME, true, false, [&](ShimState&) -> CUresult {                                                \
      if (!(NODEARG)) return CUDA_ERROR_INVALID_VALUE;                                                          \
      return sweep_refuse(#NAME, "external-semaphore graph nodes are not simulated");                           \
    });                                                                                                         \
  }
VGPU_EXT_SEM_NODE(cuGraphExternalSemaphoresSignalNodeGetParams, (void* hNode, void* params_out), hNode COMMA params_out, hNode)
VGPU_EXT_SEM_NODE(cuGraphExternalSemaphoresSignalNodeSetParams, (void* hNode, const void* nodeParams), hNode COMMA nodeParams, hNode)
VGPU_EXT_SEM_NODE(cuGraphExternalSemaphoresWaitNodeGetParams, (void* hNode, void* params_out), hNode COMMA params_out, hNode)
VGPU_EXT_SEM_NODE(cuGraphExternalSemaphoresWaitNodeSetParams, (void* hNode, const void* nodeParams), hNode COMMA nodeParams, hNode)
VGPU_EXT_SEM_NODE(cuGraphExecExternalSemaphoresSignalNodeSetParams, (void* hGraphExec, void* hNode, const void* nodeParams), hGraphExec COMMA hNode COMMA nodeParams, hGraphExec && hNode)
VGPU_EXT_SEM_NODE(cuGraphExecExternalSemaphoresWaitNodeSetParams, (void* hGraphExec, void* hNode, const void* nodeParams), hGraphExec COMMA hNode COMMA nodeParams, hGraphExec && hNode)
#undef VGPU_EXT_SEM_NODE

// A conditional-node handle: the conditional nodes it is for (cuGraphAddNode with a conditional node, the driver
// graph's if/while) are not offered by this driver graph, so a handle would be one nothing can use. Arguments
// are checked as the card checks them (a null graph or result, flags past the default-assign bit,
// CUDA_ERROR_INVALID_VALUE; a null context CUDA_ERROR_INVALID_CONTEXT) and the call is then refused.
static CUresult cuGraphConditionalHandleCreate_impl(unsigned long long* pHandle_out, void* hGraph, CUcontext ctx, unsigned int defaultLaunchValue, unsigned int flags);
VGPU_EXPORT CUresult cuGraphConditionalHandleCreate(unsigned long long* pHandle_out, void* hGraph, CUcontext ctx, unsigned int defaultLaunchValue, unsigned int flags) { return traced("cuGraphConditionalHandleCreate", cuGraphConditionalHandleCreate_impl, pHandle_out, hGraph, ctx, defaultLaunchValue, flags); }
static CUresult cuGraphConditionalHandleCreate_impl(unsigned long long* pHandle_out, void* hGraph, CUcontext ctx, unsigned int defaultLaunchValue, unsigned int flags) {
  return api("cuGraphConditionalHandleCreate", true, false, [&](ShimState& s) -> CUresult {
    (void)defaultLaunchValue;
    if (!pHandle_out || !hGraph || (flags & ~1u)) return CUDA_ERROR_INVALID_VALUE;
    if (!ctx || !s.contexts.count(reinterpret_cast<uintptr_t>(ctx))) return CUDA_ERROR_INVALID_CONTEXT;
    return sweep_refuse("cuGraphConditionalHandleCreate", "the driver graph API offers no conditional nodes to use the handle with");
  });
}

// ---- GPU core dumps ----------------------------------------------------------------------------------------------------------------
// The coredump settings. An RTX 3060 under WSL has the file name only (CU_COREDUMP_FILE, 5): it reads back empty and a
// name can be set; every other setting is CUDA_ERROR_INVALID_VALUE to read and to write, as is an attribute that does not exist.
// (The global form before cuInit is not exercised by the card here; it behaves as the per-context one.) A
// null value is CUDA_SUCCESS to read, as the card has it.
namespace {
std::mutex g_coredump_mu;
std::string g_coredump_file;
CUresult coredump_get(int attrib, void* value, size_t* size) {
  if (!size) return CUDA_ERROR_INVALID_VALUE;
  if (!value) return CUDA_SUCCESS;   // before the attribute is looked at, as the card has it
  if (attrib != 5) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(g_coredump_mu);
  if (*size > g_coredump_file.size()) std::memcpy(value, g_coredump_file.c_str(), g_coredump_file.size() + 1);
  return CUDA_SUCCESS;
}
CUresult coredump_set(int attrib, void* value, size_t* size) {
  if (!value || !size) return CUDA_ERROR_INVALID_VALUE;
  if (attrib != 5) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(g_coredump_mu);
  g_coredump_file.assign(static_cast<const char*>(value), strnlen(static_cast<const char*>(value), *size));
  return CUDA_SUCCESS;
}
}  // namespace

static CUresult cuCoredumpGetAttribute_impl(int attrib, void* value, size_t* size);
VGPU_EXPORT CUresult cuCoredumpGetAttribute(int attrib, void* value, size_t* size) { return traced("cuCoredumpGetAttribute", cuCoredumpGetAttribute_impl, attrib, value, size); }
static CUresult cuCoredumpGetAttribute_impl(int attrib, void* value, size_t* size) {
  return api("cuCoredumpGetAttribute", true, false, [&](ShimState&) -> CUresult { return coredump_get(attrib, value, size); });
}
static CUresult cuCoredumpGetAttributeGlobal_impl(int attrib, void* value, size_t* size);
VGPU_EXPORT CUresult cuCoredumpGetAttributeGlobal(int attrib, void* value, size_t* size) { return traced("cuCoredumpGetAttributeGlobal", cuCoredumpGetAttributeGlobal_impl, attrib, value, size); }
static CUresult cuCoredumpGetAttributeGlobal_impl(int attrib, void* value, size_t* size) { return coredump_get(attrib, value, size); }
static CUresult cuCoredumpSetAttribute_impl(int attrib, void* value, size_t* size);
VGPU_EXPORT CUresult cuCoredumpSetAttribute(int attrib, void* value, size_t* size) { return traced("cuCoredumpSetAttribute", cuCoredumpSetAttribute_impl, attrib, value, size); }
static CUresult cuCoredumpSetAttribute_impl(int attrib, void* value, size_t* size) {
  return api("cuCoredumpSetAttribute", true, false, [&](ShimState&) -> CUresult { return coredump_set(attrib, value, size); });
}
static CUresult cuCoredumpSetAttributeGlobal_impl(int attrib, void* value, size_t* size);
VGPU_EXPORT CUresult cuCoredumpSetAttributeGlobal(int attrib, void* value, size_t* size) { return traced("cuCoredumpSetAttributeGlobal", cuCoredumpSetAttributeGlobal_impl, attrib, value, size); }
static CUresult cuCoredumpSetAttributeGlobal_impl(int attrib, void* value, size_t* size) { return coredump_set(attrib, value, size); }

// ---- the driver's log (CUDA 13) -----------------------------------------------------------------------------------------------------
// As the runtime's cudaLogs*: nothing is logged here (an RTX 3060's log is empty unless CUDA_LOG_FILE asks), so the iterator is
// 0, a dump writes nothing (a file is created empty, memory gets a size of 0) and a callback is never called.
namespace {
std::mutex g_log_mu;
std::set<uintptr_t> g_log_callbacks;
uintptr_t g_next_log_callback = 1;
}  // namespace

static CUresult cuLogsRegisterCallback_impl(void* callbackFunc, void* userData, void** callback_out);
VGPU_EXPORT CUresult cuLogsRegisterCallback(void* callbackFunc, void* userData, void** callback_out) { return traced("cuLogsRegisterCallback", cuLogsRegisterCallback_impl, callbackFunc, userData, callback_out); }
static CUresult cuLogsRegisterCallback_impl(void* callbackFunc, void* userData, void** callback_out) {
  (void)userData;
  if (!callbackFunc || !callback_out) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(g_log_mu);
  const uintptr_t h = g_next_log_callback++;
  g_log_callbacks.insert(h);
  *callback_out = reinterpret_cast<void*>(h);
  return CUDA_SUCCESS;
}
static CUresult cuLogsUnregisterCallback_impl(void* callback);
VGPU_EXPORT CUresult cuLogsUnregisterCallback(void* callback) { return traced("cuLogsUnregisterCallback", cuLogsUnregisterCallback_impl, callback); }
static CUresult cuLogsUnregisterCallback_impl(void* callback) {
  if (!callback) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(g_log_mu);
  return g_log_callbacks.erase(reinterpret_cast<uintptr_t>(callback)) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
static CUresult cuLogsCurrent_impl(unsigned int* iterator_out, unsigned int flags);
VGPU_EXPORT CUresult cuLogsCurrent(unsigned int* iterator_out, unsigned int flags) { return traced("cuLogsCurrent", cuLogsCurrent_impl, iterator_out, flags); }
static CUresult cuLogsCurrent_impl(unsigned int* iterator_out, unsigned int flags) {
  (void)flags;
  if (!iterator_out) return CUDA_ERROR_INVALID_VALUE;
  *iterator_out = 0;
  return CUDA_SUCCESS;
}
static CUresult cuLogsDumpToFile_impl(unsigned int* iterator, const char* pathToFile, unsigned int flags);
VGPU_EXPORT CUresult cuLogsDumpToFile(unsigned int* iterator, const char* pathToFile, unsigned int flags) { return traced("cuLogsDumpToFile", cuLogsDumpToFile_impl, iterator, pathToFile, flags); }
static CUresult cuLogsDumpToFile_impl(unsigned int* iterator, const char* pathToFile, unsigned int flags) {
  (void)iterator; (void)flags;
  if (!pathToFile) return CUDA_ERROR_INVALID_VALUE;
  if (std::FILE* f = std::fopen(pathToFile, "w")) {
    std::fclose(f);
    return CUDA_SUCCESS;
  }
  return CUDA_ERROR_INVALID_VALUE;
}
static CUresult cuLogsDumpToMemory_impl(unsigned int* iterator, char* buffer, size_t* size, unsigned int flags);
VGPU_EXPORT CUresult cuLogsDumpToMemory(unsigned int* iterator, char* buffer, size_t* size, unsigned int flags) { return traced("cuLogsDumpToMemory", cuLogsDumpToMemory_impl, iterator, buffer, size, flags); }
static CUresult cuLogsDumpToMemory_impl(unsigned int* iterator, char* buffer, size_t* size, unsigned int flags) {
  (void)iterator; (void)flags;
  if (!buffer || !size) return CUDA_ERROR_INVALID_VALUE;
  *size = 0;
  return CUDA_SUCCESS;
}

// ---- checkpointing a process ------------------------------------------------------------------------------------------------------------
// The checkpoint API (a Linux driver's, to freeze a process's GPU state): an RTX 3060 under WSL answers the state of a
// process of this machine as running (0), a process that does not exist CUDA_ERROR_OPERATING_SYSTEM, locking
// CUDA_ERROR_NOT_SUPPORTED, and checkpointing, restoring and unlocking a process that was not locked
// CUDA_ERROR_ILLEGAL_STATE; a null result is CUDA_ERROR_INVALID_VALUE (all measured). Locking is what a checkpoint
// begins with, and it is refused, so nothing past it can happen.
static CUresult cuCheckpointProcessGetState_impl(int pid, int* state);
VGPU_EXPORT CUresult cuCheckpointProcessGetState(int pid, int* state) { return traced("cuCheckpointProcessGetState", cuCheckpointProcessGetState_impl, pid, state); }
static CUresult cuCheckpointProcessGetState_impl(int pid, int* state) {
  return api("cuCheckpointProcessGetState", true, false, [&](ShimState&) -> CUresult {
    if (!state) return CUDA_ERROR_INVALID_VALUE;
    if (pid <= 0 || ::kill(pid, 0) != 0) return kOperatingSystem;
    *state = 0;   // CU_PROCESS_STATE_RUNNING
    return CUDA_SUCCESS;
  });
}
static CUresult cuCheckpointProcessGetRestoreThreadId_impl(int pid, int* tid);
VGPU_EXPORT CUresult cuCheckpointProcessGetRestoreThreadId(int pid, int* tid) { return traced("cuCheckpointProcessGetRestoreThreadId", cuCheckpointProcessGetRestoreThreadId_impl, pid, tid); }
static CUresult cuCheckpointProcessGetRestoreThreadId_impl(int pid, int* tid) {
  return api("cuCheckpointProcessGetRestoreThreadId", true, false, [&](ShimState&) -> CUresult {
    if (!tid) return CUDA_ERROR_INVALID_VALUE;
    if (pid <= 0 || ::kill(pid, 0) != 0) return kOperatingSystem;
    *tid = pid;   // the thread that restores is the process's main thread
    return CUDA_SUCCESS;
  });
}
static CUresult cuCheckpointProcessLock_impl(int pid, void* args);
VGPU_EXPORT CUresult cuCheckpointProcessLock(int pid, void* args) { return traced("cuCheckpointProcessLock", cuCheckpointProcessLock_impl, pid, args); }
static CUresult cuCheckpointProcessLock_impl(int pid, void* args) {
  return api("cuCheckpointProcessLock", true, false, [&](ShimState&) -> CUresult {
    (void)args;
    if (pid <= 0 || ::kill(pid, 0) != 0) return kOperatingSystem;
    return sweep_refuse("cuCheckpointProcessLock", "checkpointing a process's GPU state is not simulated");
  });
}
static CUresult cuCheckpointProcessCheckpoint_impl(int pid, void* args);
VGPU_EXPORT CUresult cuCheckpointProcessCheckpoint(int pid, void* args) { return traced("cuCheckpointProcessCheckpoint", cuCheckpointProcessCheckpoint_impl, pid, args); }
static CUresult cuCheckpointProcessCheckpoint_impl(int pid, void* args) {
  return api("cuCheckpointProcessCheckpoint", true, false, [&](ShimState&) -> CUresult {
    (void)args;
    if (pid <= 0 || ::kill(pid, 0) != 0) return kOperatingSystem;
    return kIllegalState;   // the process was never locked
  });
}
static CUresult cuCheckpointProcessRestore_impl(int pid, void* args);
VGPU_EXPORT CUresult cuCheckpointProcessRestore(int pid, void* args) { return traced("cuCheckpointProcessRestore", cuCheckpointProcessRestore_impl, pid, args); }
static CUresult cuCheckpointProcessRestore_impl(int pid, void* args) {
  return api("cuCheckpointProcessRestore", true, false, [&](ShimState&) -> CUresult {
    (void)args;
    if (pid <= 0 || ::kill(pid, 0) != 0) return kOperatingSystem;
    return kIllegalState;
  });
}
static CUresult cuCheckpointProcessUnlock_impl(int pid, void* args);
VGPU_EXPORT CUresult cuCheckpointProcessUnlock(int pid, void* args) { return traced("cuCheckpointProcessUnlock", cuCheckpointProcessUnlock_impl, pid, args); }
static CUresult cuCheckpointProcessUnlock_impl(int pid, void* args) {
  return api("cuCheckpointProcessUnlock", true, false, [&](ShimState&) -> CUresult {
    (void)args;
    if (pid <= 0 || ::kill(pid, 0) != 0) return kOperatingSystem;
    return kIllegalState;
  });
}

// ---- green contexts and device resources ----------------------------------------------------------------------------------------------
// Partitioning a device's SMs into green contexts is not simulated: every entry point is refused by name, once said.
// (An RTX 3060 has them: cuDeviceGetDevResource reports 28 SMs, and a green context can be made from a split of them.
// A null green context is CUDA_ERROR_INVALID_CONTEXT there, which is the answer given for it here too.)
// The device's resources: the SM count is the only resource type there is (CU_DEV_RESOURCE_TYPE_SM, 1), another type
// is CUDA_ERROR_INVALID_RESOURCE_TYPE and a device that does not exist CUDA_ERROR_INVALID_DEVICE (measured); the
// SM resource itself is the green-context machinery, refused below.
static CUresult cuDeviceGetDevResource_impl(CUdevice device, void* resource, int type);
VGPU_EXPORT CUresult cuDeviceGetDevResource(CUdevice device, void* resource, int type) { return traced("cuDeviceGetDevResource", cuDeviceGetDevResource_impl, device, resource, type); }
static CUresult cuDeviceGetDevResource_impl(CUdevice device, void* resource, int type) {
  return api("cuDeviceGetDevResource", true, false, [&](ShimState& s) -> CUresult {
    (void)resource;
    if (device < 0 || device >= s.rt->device_count()) return CUDA_ERROR_INVALID_DEVICE;
    if (type != 1) return kInvalidResourceType;
    return sweep_refuse("cuDeviceGetDevResource", "green contexts and device resources are not simulated");
  });
}

#define VGPU_GREEN(NAME, SIG, ARGS, NULLCHECK)                                                                   \
  static CUresult NAME##_impl SIG;                                                                              \
  VGPU_EXPORT CUresult NAME SIG { return traced(#NAME, NAME##_impl, ARGS); }                                    \
  static CUresult NAME##_impl SIG {                                                                             \
    return api(#NAME, true, false, [&](ShimState&) -> CUresult {                                                \
      if (NULLCHECK) return CUDA_ERROR_INVALID_CONTEXT;                                                         \
      return sweep_refuse(#NAME, "green contexts and device resources are not simulated");                      \
    });                                                                                                         \
  }
VGPU_GREEN(cuCtxGetDevResource, (CUcontext hCtx, void* resource, int type), hCtx COMMA resource COMMA type, false)
VGPU_GREEN(cuGreenCtxGetDevResource, (void* hCtx, void* resource, int type), hCtx COMMA resource COMMA type, !hCtx)
VGPU_GREEN(cuDevSmResourceSplitByCount, (void* result, unsigned int* nbGroups, const void* input, void* remaining, unsigned int useFlags, unsigned int minCount), result COMMA nbGroups COMMA input COMMA remaining COMMA useFlags COMMA minCount, false)
VGPU_GREEN(cuDevResourceGenerateDesc, (void** phDesc, void* resources, unsigned int nbResources), phDesc COMMA resources COMMA nbResources, false)
VGPU_GREEN(cuGreenCtxCreate, (void** phCtx, void* desc, CUdevice dev, unsigned int flags), phCtx COMMA desc COMMA dev COMMA flags, false)
VGPU_GREEN(cuGreenCtxDestroy, (void* hCtx), hCtx, !hCtx)
VGPU_GREEN(cuCtxFromGreenCtx, (CUcontext* pContext, void* hCtx), pContext COMMA hCtx, !hCtx)
VGPU_GREEN(cuGreenCtxRecordEvent, (void* hCtx, void* hEvent), hCtx COMMA hEvent, !hCtx)
VGPU_GREEN(cuGreenCtxWaitEvent, (void* hCtx, void* hEvent), hCtx COMMA hEvent, !hCtx)
VGPU_GREEN(cuGreenCtxStreamCreate, (CUstream* phStream, void* greenCtx, unsigned int flags, int priority), phStream COMMA greenCtx COMMA flags COMMA priority, !greenCtx)
VGPU_GREEN(cuGreenCtxGetId, (void* greenCtx, unsigned long long* greenCtxId), greenCtx COMMA greenCtxId, !greenCtx)
#undef VGPU_GREEN

// A stream made on a green context belongs to one; a plain stream is on none (CUDA_SUCCESS and a null result,
// measured).
static CUresult cuStreamGetGreenCtx_impl(CUstream hStream, void** phCtx);
VGPU_EXPORT CUresult cuStreamGetGreenCtx(CUstream hStream, void** phCtx) { return traced("cuStreamGetGreenCtx", cuStreamGetGreenCtx_impl, hStream, phCtx); }
static CUresult cuStreamGetGreenCtx_impl(CUstream hStream, void** phCtx) {
  return api("cuStreamGetGreenCtx", true, false, [&](ShimState& s) -> CUresult {
    if (!phCtx) return CUDA_ERROR_INVALID_VALUE;
    const uintptr_t h = reinterpret_cast<uintptr_t>(hStream);
    if (h && !s.streams.count(h)) return CUDA_ERROR_INVALID_HANDLE;
    *phCtx = nullptr;
    return CUDA_SUCCESS;
  });
}
