// libvgpucuda — VirtualGPU's implementation of a CUDA Driver API subset.
//
// Clean-room: implemented from NVIDIA's public documentation only (see
// include/vgpu_cuda.h). This is the C ABI surface external programs link
// against; it forwards into the C++ runtime and maps vgpu::Error codes onto
// documented CUresult values. Because a CUresult can't carry rich messages,
// the full diagnostic is printed to stderr (VGPU_QUIET=1 disables).
//
// Threading: one global lock around all state. Coarse but correct; the
// current-context stack is process-global rather than thread-local for now
// (documented MVP simplification — see TODO.md).
#include "vgpu_cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"

namespace {

constexpr int kDriverVersion = 12080;  // reported as CUDA 12.8

// Handle tagging: low 3 bits encode the handle type so passing e.g. a module
// where a context belongs is caught instead of misbehaving.
constexpr uintptr_t kTagCtx = 1, kTagModule = 2, kTagFunc = 3;

struct FuncRec {
  int device = 0;
  uintptr_t module_handle = 0;
  const vgpu::ptx::EntryFn* fn = nullptr;
};

struct ShimState {
  std::recursive_mutex mu;
  bool initialized = false;
  std::unique_ptr<vgpu::runtime::Runtime> rt;
  uintptr_t next_id = 8;

  std::unordered_map<uintptr_t, int> contexts;         // ctx handle -> device ordinal
  std::unordered_map<int, uintptr_t> primary_ctx;      // device -> primary ctx handle
  std::vector<uintptr_t> ctx_stack;                    // current-context stack (global)
  std::unordered_map<uintptr_t, std::pair<int, uint64_t>> modules;  // handle -> (dev, module id)
  std::unordered_map<uintptr_t, FuncRec> functions;
};

ShimState& state() {
  static ShimState s;
  return s;
}

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

void report(const char* api, const std::string& msg) {
  if (!quiet()) std::fprintf(stderr, "[vgpu] %s: %s\n", api, msg.c_str());
}

CUresult map_error(const vgpu::Error& e, bool kernel_context) {
  using vgpu::Err;
  switch (e.code()) {
    case Err::OutOfMemory: return CUDA_ERROR_OUT_OF_MEMORY;
    case Err::UnknownGpu: return CUDA_ERROR_INVALID_DEVICE;
    case Err::PtxParse: return CUDA_ERROR_INVALID_PTX;
    case Err::UnsupportedPtx: return CUDA_ERROR_NOT_SUPPORTED;
    case Err::Unsupported: return CUDA_ERROR_NOT_SUPPORTED;
    case Err::NotFound: return CUDA_ERROR_NOT_FOUND;
    case Err::ExecLimit: return CUDA_ERROR_LAUNCH_TIMEOUT;
    case Err::InvalidPointer:
    case Err::UseAfterFree:
    case Err::OutOfBounds:
    case Err::MisalignedAccess:
      // Inside a kernel these are the moral equivalent of a device-side fault.
      return kernel_context ? CUDA_ERROR_ILLEGAL_ADDRESS : CUDA_ERROR_INVALID_VALUE;
    case Err::UninitializedRegister: return CUDA_ERROR_ILLEGAL_ADDRESS;
    case Err::DoubleFree:
    case Err::InvalidFree:
    case Err::InvalidValue:
    case Err::LaunchConfig:
    case Err::ProfileParse: return CUDA_ERROR_INVALID_VALUE;
    case Err::Internal: return CUDA_ERROR_UNKNOWN;
  }
  return CUDA_ERROR_UNKNOWN;
}

// Wraps an API body: locks, checks init, catches and maps errors.
template <class F>
CUresult api(const char* name, bool needs_init, bool kernel_context, F&& body) {
  ShimState& s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (needs_init && !s.initialized) {
    report(name, "cuInit has not been called");
    return CUDA_ERROR_NOT_INITIALIZED;
  }
  try {
    return body(s);
  } catch (const vgpu::Error& e) {
    report(name, e.what());
    return map_error(e, kernel_context);
  } catch (const std::exception& e) {
    report(name, std::string("unexpected: ") + e.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

int current_device(ShimState& s) {
  if (s.ctx_stack.empty())
    throw vgpu::Error::make(vgpu::Err::InvalidValue,
                            "no current context (create one with cuCtxCreate or "
                            "cuDevicePrimaryCtxRetain + cuCtxSetCurrent)");
  return s.contexts.at(s.ctx_stack.back());
}

vgpu::runtime::Device& current(ShimState& s) { return s.rt->device(current_device(s)); }

void check_device(ShimState& s, CUdevice dev) {
  if (dev < 0 || dev >= s.rt->device_count())
    throw vgpu::Error::make(vgpu::Err::UnknownGpu, "invalid device ordinal ", dev, " (have ",
                            s.rt->device_count(), ")");
}

uintptr_t make_handle(ShimState& s, uintptr_t tag) { return (s.next_id++ << 3) | tag; }

uintptr_t check_handle(uintptr_t h, uintptr_t tag, const char* what) {
  if ((h & 7) != tag)
    throw vgpu::Error::make(vgpu::Err::InvalidValue, "handle ", h, " is not a valid ", what,
                            " handle (wrong type or corrupted)");
  return h;
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

/* ---- init / version / errors ---- */

VGPU_EXPORT CUresult cuInit(unsigned int flags) {
  return api("cuInit", false, false, [&](ShimState& s) {
    if (flags != 0) return CUDA_ERROR_INVALID_VALUE;
    if (s.initialized) return CUDA_SUCCESS;
    const char* gpu = std::getenv("VGPU_GPU");
    std::string id = gpu && gpu[0] ? gpu : "nvidia/h100";
    int count = 1;
    if (const char* c = std::getenv("VGPU_DEVICE_COUNT"); c && c[0]) count = std::atoi(c);
    vgpu::DeviceProfile profile = vgpu::load_gpu(id);
    s.rt = std::make_unique<vgpu::runtime::Runtime>(profile, count);
    s.initialized = true;
    if (!quiet())
      std::fprintf(stderr, "[vgpu] virtual GPU platform initialized: %d x %s (%s)\n", count,
                   profile.id.c_str(), profile.model.c_str());
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuDriverGetVersion(int* driverVersion) {
  if (!driverVersion) return CUDA_ERROR_INVALID_VALUE;
  *driverVersion = kDriverVersion;
  return CUDA_SUCCESS;
}

namespace {
struct ErrEntry {
  CUresult code;
  const char* name;
  const char* str;
};
constexpr ErrEntry kErrTable[] = {
    {CUDA_SUCCESS, "CUDA_SUCCESS", "no error"},
    {CUDA_ERROR_INVALID_VALUE, "CUDA_ERROR_INVALID_VALUE", "invalid argument"},
    {CUDA_ERROR_OUT_OF_MEMORY, "CUDA_ERROR_OUT_OF_MEMORY", "out of memory"},
    {CUDA_ERROR_NOT_INITIALIZED, "CUDA_ERROR_NOT_INITIALIZED", "initialization error"},
    {CUDA_ERROR_INVALID_DEVICE, "CUDA_ERROR_INVALID_DEVICE", "invalid device ordinal"},
    {CUDA_ERROR_INVALID_CONTEXT, "CUDA_ERROR_INVALID_CONTEXT", "invalid device context"},
    {CUDA_ERROR_INVALID_PTX, "CUDA_ERROR_INVALID_PTX", "a PTX JIT compilation failed"},
    {CUDA_ERROR_NOT_FOUND, "CUDA_ERROR_NOT_FOUND", "named symbol not found"},
    {CUDA_ERROR_ILLEGAL_ADDRESS, "CUDA_ERROR_ILLEGAL_ADDRESS",
     "an illegal memory access was encountered"},
    {CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
     "too many resources requested for launch"},
    {CUDA_ERROR_LAUNCH_TIMEOUT, "CUDA_ERROR_LAUNCH_TIMEOUT", "the launch timed out and was terminated"},
    {CUDA_ERROR_NOT_SUPPORTED, "CUDA_ERROR_NOT_SUPPORTED", "operation not supported"},
    {CUDA_ERROR_UNKNOWN, "CUDA_ERROR_UNKNOWN", "unknown error"},
};
}  // namespace

VGPU_EXPORT CUresult cuGetErrorName(CUresult error, const char** pStr) {
  if (!pStr) return CUDA_ERROR_INVALID_VALUE;
  for (const auto& e : kErrTable)
    if (e.code == error) {
      *pStr = e.name;
      return CUDA_SUCCESS;
    }
  *pStr = nullptr;
  return CUDA_ERROR_INVALID_VALUE;
}

VGPU_EXPORT CUresult cuGetErrorString(CUresult error, const char** pStr) {
  if (!pStr) return CUDA_ERROR_INVALID_VALUE;
  for (const auto& e : kErrTable)
    if (e.code == error) {
      *pStr = e.str;
      return CUDA_SUCCESS;
    }
  *pStr = nullptr;
  return CUDA_ERROR_INVALID_VALUE;
}

/* ---- device discovery ---- */

VGPU_EXPORT CUresult cuDeviceGetCount(int* count) {
  return api("cuDeviceGetCount", true, false, [&](ShimState& s) {
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    *count = s.rt->device_count();
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuDeviceGet(CUdevice* device, int ordinal) {
  return api("cuDeviceGet", true, false, [&](ShimState& s) {
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, ordinal);
    *device = ordinal;
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
  return api("cuDeviceGetName", true, false, [&](ShimState& s) {
    if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    const std::string& model = s.rt->device(dev).profile().model;
    std::snprintf(name, static_cast<size_t>(len), "%s", model.c_str());
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev) {
  return api("cuDeviceTotalMem", true, false, [&](ShimState& s) {
    if (!bytes) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    *bytes = static_cast<size_t>(s.rt->device(dev).profile().vram_bytes);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev) {
  return cuDeviceTotalMem_v2(bytes, dev);
}

VGPU_EXPORT CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
  return api("cuDeviceGetAttribute", true, false, [&](ShimState& s) {
    if (!pi) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    const vgpu::DeviceProfile& p = s.rt->device(dev).profile();
    switch (attrib) {
      case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: *pi = (int)p.limits.max_threads_per_block; break;
      case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: *pi = (int)p.limits.max_block_dim[0]; break;
      case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: *pi = (int)p.limits.max_block_dim[1]; break;
      case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: *pi = (int)p.limits.max_block_dim[2]; break;
      case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: *pi = (int)p.limits.max_grid_dim[0]; break;
      case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: *pi = (int)p.limits.max_grid_dim[1]; break;
      case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: *pi = (int)p.limits.max_grid_dim[2]; break;
      case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: *pi = (int)p.limits.shared_mem_per_block; break;
      case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: *pi = 65536; break;
      case CU_DEVICE_ATTRIBUTE_WARP_SIZE: *pi = (int)p.warp_size; break;
      case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: *pi = (int)p.limits.registers_per_block; break;
      case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
        // VirtualGPU does not model performance; this is a documented placeholder.
        *pi = 1000000;
        break;
      case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: *pi = (int)p.limits.multiprocessors; break;
      case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: *pi = p.cc_major; break;
      case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: *pi = p.cc_minor; break;
      default:
        report("cuDeviceGetAttribute",
               "attribute " + std::to_string(attrib) + " is not implemented yet");
        return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev) {
  return api("cuDeviceComputeCapability", true, false, [&](ShimState& s) {
    if (!major || !minor) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    *major = s.rt->device(dev).profile().cc_major;
    *minor = s.rt->device(dev).profile().cc_minor;
    return CUDA_SUCCESS;
  });
}

/* ---- contexts ---- */

VGPU_EXPORT CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev) {
  return api("cuCtxCreate", true, false, [&](ShimState& s) {
    (void)flags;  // scheduling flags are performance hints; functionally inert here
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    uintptr_t h = make_handle(s, kTagCtx);
    s.contexts[h] = dev;
    s.ctx_stack.push_back(h);  // cuCtxCreate makes the new context current
    *pctx = reinterpret_cast<CUcontext>(h);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
  return cuCtxCreate_v2(pctx, flags, dev);
}

VGPU_EXPORT CUresult cuCtxDestroy_v2(CUcontext ctx) {
  return api("cuCtxDestroy", true, false, [&](ShimState& s) {
    uintptr_t h = check_handle(reinterpret_cast<uintptr_t>(ctx), kTagCtx, "context");
    if (!s.contexts.erase(h)) return CUDA_ERROR_INVALID_CONTEXT;
    std::erase(s.ctx_stack, h);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuCtxDestroy(CUcontext ctx) { return cuCtxDestroy_v2(ctx); }

VGPU_EXPORT CUresult cuCtxSetCurrent(CUcontext ctx) {
  return api("cuCtxSetCurrent", true, false, [&](ShimState& s) {
    if (!ctx) {  // NULL pops/clears the current context binding
      if (!s.ctx_stack.empty()) s.ctx_stack.pop_back();
      return CUDA_SUCCESS;
    }
    uintptr_t h = check_handle(reinterpret_cast<uintptr_t>(ctx), kTagCtx, "context");
    if (!s.contexts.count(h)) return CUDA_ERROR_INVALID_CONTEXT;
    if (!s.ctx_stack.empty())
      s.ctx_stack.back() = h;
    else
      s.ctx_stack.push_back(h);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuCtxGetCurrent(CUcontext* pctx) {
  return api("cuCtxGetCurrent", true, false, [&](ShimState& s) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    *pctx = s.ctx_stack.empty() ? nullptr : reinterpret_cast<CUcontext>(s.ctx_stack.back());
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuCtxGetDevice(CUdevice* device) {
  return api("cuCtxGetDevice", true, false, [&](ShimState& s) {
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    *device = current_device(s);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuCtxSynchronize(void) {
  return api("cuCtxSynchronize", true, false, [&](ShimState& s) {
    (void)current_device(s);  // requires a current context
    return CUDA_SUCCESS;      // everything is synchronous today
  });
}

VGPU_EXPORT CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
  return api("cuDevicePrimaryCtxRetain", true, false, [&](ShimState& s) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    auto it = s.primary_ctx.find(dev);
    if (it == s.primary_ctx.end()) {
      uintptr_t h = make_handle(s, kTagCtx);
      s.contexts[h] = dev;
      it = s.primary_ctx.emplace(dev, h).first;
    }
    *pctx = reinterpret_cast<CUcontext>(it->second);  // NOTE: does not make it current
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
  return api("cuDevicePrimaryCtxRelease", true, false, [&](ShimState& s) {
    check_device(s, dev);
    return CUDA_SUCCESS;  // refcounting is a no-op while contexts share the device state
  });
}
VGPU_EXPORT CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
  return cuDevicePrimaryCtxRelease_v2(dev);
}

/* ---- memory ---- */

VGPU_EXPORT CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
  return api("cuMemAlloc", true, false, [&](ShimState& s) {
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;
    *dptr = current(s).memory().alloc(bytesize);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
  return cuMemAlloc_v2(dptr, bytesize);
}

VGPU_EXPORT CUresult cuMemFree_v2(CUdeviceptr dptr) {
  return api("cuMemFree", true, false, [&](ShimState& s) {
    current(s).memory().free(dptr);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemFree(CUdeviceptr dptr) { return cuMemFree_v2(dptr); }

VGPU_EXPORT CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
  return api("cuMemcpyHtoD", true, false, [&](ShimState& s) {
    if (!srcHost && ByteCount) return CUDA_ERROR_INVALID_VALUE;
    current(s).memory().write(dstDevice, srcHost, ByteCount);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemcpyHtoD(CUdeviceptr d, const void* h, size_t n) {
  return cuMemcpyHtoD_v2(d, h, n);
}

VGPU_EXPORT CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  return api("cuMemcpyDtoH", true, false, [&](ShimState& s) {
    if (!dstHost && ByteCount) return CUDA_ERROR_INVALID_VALUE;
    current(s).memory().read(srcDevice, dstHost, ByteCount);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemcpyDtoH(void* h, CUdeviceptr d, size_t n) {
  return cuMemcpyDtoH_v2(h, d, n);
}

VGPU_EXPORT CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
  return api("cuMemcpyDtoD", true, false, [&](ShimState& s) {
    std::vector<uint8_t> tmp(ByteCount);
    current(s).memory().read(srcDevice, tmp.data(), ByteCount);
    current(s).memory().write(dstDevice, tmp.data(), ByteCount);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemcpyDtoD(CUdeviceptr d, CUdeviceptr sptr, size_t n) {
  return cuMemcpyDtoD_v2(d, sptr, n);
}

VGPU_EXPORT CUresult cuMemGetInfo_v2(size_t* free_out, size_t* total) {
  return api("cuMemGetInfo", true, false, [&](ShimState& s) {
    if (!free_out || !total) return CUDA_ERROR_INVALID_VALUE;
    vgpu::MemoryManager& mm = current(s).memory();
    *total = static_cast<size_t>(mm.capacity());
    *free_out = static_cast<size_t>(mm.capacity() - mm.used());
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemGetInfo(size_t* f, size_t* t) { return cuMemGetInfo_v2(f, t); }

/* ---- modules / launch ---- */

VGPU_EXPORT CUresult cuModuleLoadData(CUmodule* module, const void* image) {
  return api("cuModuleLoadData", true, false, [&](ShimState& s) {
    if (!module || !image) return CUDA_ERROR_INVALID_VALUE;
    const char* text = static_cast<const char*>(image);
    // Only NUL-terminated PTX text is supported (no cubin/fatbin). ELF images
    // start with 0x7f 'E' 'L' 'F'; give a precise error for those.
    if (text[0] == 0x7f)
      throw vgpu::Error::make(vgpu::Err::Unsupported,
                              "cuModuleLoadData received a cubin/ELF image; VirtualGPU currently "
                              "only loads NUL-terminated PTX text (compile with "
                              "-ptx or embed PTX)");
    int dev = current_device(s);
    uint64_t mid = s.rt->device(dev).load_module(text);
    uintptr_t h = make_handle(s, kTagModule);
    s.modules[h] = {dev, mid};
    *module = reinterpret_cast<CUmodule>(h);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions,
                                        void* options, void** optionValues) {
  (void)numOptions;
  (void)options;
  (void)optionValues;  // JIT options are performance/verbosity hints; ignored
  return cuModuleLoadData(module, image);
}

VGPU_EXPORT CUresult cuModuleUnload(CUmodule hmod) {
  return api("cuModuleUnload", true, false, [&](ShimState& s) {
    uintptr_t h = check_handle(reinterpret_cast<uintptr_t>(hmod), kTagModule, "module");
    auto it = s.modules.find(h);
    if (it == s.modules.end()) return CUDA_ERROR_NOT_FOUND;
    auto [dev, mid] = it->second;
    s.rt->device(dev).unload_module(mid);
    s.modules.erase(it);
    for (auto fit = s.functions.begin(); fit != s.functions.end();) {
      // Function handles from this module are now dangling; drop them.
      fit = fit->second.module_handle == h ? s.functions.erase(fit) : std::next(fit);
    }
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
  return api("cuModuleGetFunction", true, false, [&](ShimState& s) {
    if (!hfunc || !name) return CUDA_ERROR_INVALID_VALUE;
    uintptr_t h = check_handle(reinterpret_cast<uintptr_t>(hmod), kTagModule, "module");
    auto it = s.modules.find(h);
    if (it == s.modules.end()) return CUDA_ERROR_NOT_FOUND;
    auto [dev, mid] = it->second;
    const vgpu::ptx::EntryFn* fn = s.rt->device(dev).get_function(mid, name);
    uintptr_t fh = make_handle(s, kTagFunc);
    s.functions[fh] = {dev, h, fn};
    *hfunc = reinterpret_cast<CUfunction>(fh);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                                    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                                    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
                                    void** kernelParams, void** extra) {
  return api("cuLaunchKernel", true, true, [&](ShimState& s) {
    uintptr_t fh = check_handle(reinterpret_cast<uintptr_t>(f), kTagFunc, "function");
    auto it = s.functions.find(fh);
    if (it == s.functions.end()) return CUDA_ERROR_NOT_FOUND;
    const FuncRec& rec = it->second;
    if (hStream != nullptr)
      throw vgpu::Error::make(vgpu::Err::Unsupported,
                              "streams are not implemented yet; pass the default stream (0)");
    if (extra != nullptr)
      throw vgpu::Error::make(vgpu::Err::Unsupported,
                              "the `extra` parameter-packing path of cuLaunchKernel is not "
                              "implemented; use kernelParams");
    const auto& params = rec.fn->params;
    if (!kernelParams && !params.empty()) return CUDA_ERROR_INVALID_VALUE;
    std::vector<std::vector<uint8_t>> args(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
      if (!kernelParams[i])
        throw vgpu::Error::make(vgpu::Err::InvalidValue, "kernelParams[", i, "] is NULL (kernel '",
                                rec.fn->name, "' takes ", params.size(), " parameters)");
      uint32_t size = params[i].ty.bytes();
      args[i].resize(size);
      std::memcpy(args[i].data(), kernelParams[i], size);
    }
    vgpu::exec::LaunchConfig cfg;
    cfg.grid = {gridDimX, gridDimY, gridDimZ};
    cfg.block = {blockDimX, blockDimY, blockDimZ};
    cfg.shared_bytes = sharedMemBytes;
    s.rt->device(rec.device).launch(*rec.fn, cfg, args);
    return CUDA_SUCCESS;
  });
}
