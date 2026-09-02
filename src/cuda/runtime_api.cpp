// libvgpucudart — VirtualGPU's implementation of the CUDA Runtime API.
//
// This is what unmodified nvcc-compiled applications link against. It replaces
// NVIDIA's libcudart: the same documented C ABI, backed by the VirtualGPU CPU
// engine instead of a GPU. It implements two documented interfaces:
//
//   1. The public CUDA Runtime API (cudaMalloc, cudaMemcpy, cudaLaunchKernel,
//      cudaGetDeviceProperties, ...), from docs.nvidia.com.
//   2. The nvcc host boilerplate ABI (__cudaRegisterFatBinary,
//      __cudaRegisterFunction, __cudaPushCallConfiguration, ...) that the
//      "<<<grid, block>>>" launch syntax lowers to, documented in the toolkit's
//      crt/host_runtime.h. These let a chevron launch route to cudaLaunchKernel.
//
// Building this file requires the CUDA toolkit's ABI headers (driver_types.h,
// vector_types.h) for the exact cudaDeviceProp / enum layout — they are used at
// build time only and not redistributed. The rest of VirtualGPU stays
// dependency-free; this shim is an optional target (see CMakeLists.txt).
//
// Everything is synchronous and deterministic, matching the engine.
// The vendor header is included so every entry point below is checked against
// NVIDIA's own declaration at compile time. Without it the shim's signatures
// were only as right as they looked, and several were not: cudaGraphInstantiate
// still had its CUDA 11 arity, and half the graph and capture calls took int
// where the API takes an enum or an opaque handle.
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_types.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "fatbin.hpp"
// cudaDeviceProp is filled in by this shim and read by the application, so
// both sides must agree on its layout. The original failure was a stale
// /usr/include copy of the header winning over the toolkit's, which silently
// smashed the caller's stack -- so the guard checks that the header in use is
// the one that belongs to the CUDA runtime version it claims to be. Sizes are
// recorded per toolkit; an unrecognised version compiles (a newer toolkit is
// not automatically wrong) but says so, rather than refusing to build.
#if CUDART_VERSION >= 13000 && CUDART_VERSION < 14000
static_assert(sizeof(cudaDeviceProp) == 1008,
              "cudaDeviceProp is not the size CUDA 13 defines: the header in use is probably not "
              "the toolkit's");
#elif CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
static_assert(sizeof(cudaDeviceProp) == 1032,
              "cudaDeviceProp is not the size CUDA 12 defines: the header in use is probably not "
              "the toolkit's");
#else
#warning "unrecognised CUDA runtime version: cudaDeviceProp layout is unchecked"
#endif
#include "vgpu/error.hpp"
#include "vgpu/telemetry.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"

namespace {

constexpr int kRuntimeVersion = 13000;  // CUDA 13.0
constexpr int kDriverVersion = 13000;

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}
bool trace() {
  const char* t = std::getenv("VGPU_TRACE");
  return t && t[0] == '1';
}

// A fatbin registered by the host boilerplate. PTX is parsed into a runtime
// module lazily on first use for the active device.
struct RegisteredModule {
  std::string ptx;
  std::unordered_map<int, uint64_t> module_per_device;  // device -> runtime module id
};

struct KernelInfo {
  RegisteredModule* mod = nullptr;
  std::string entry_name;
};

struct State {
  std::recursive_mutex mu;
  std::unique_ptr<vgpu::runtime::Runtime> rt;
  int current_device = 0;
  std::vector<std::unique_ptr<RegisteredModule>> modules;
  std::unordered_map<const void*, KernelInfo> kernels;  // host stub ptr -> kernel
  std::map<void*, size_t> host_allocs;
  bool initialized = false;
};

State& st() {
  static State s;
  return s;
}

// The driver shim keeps its own initialization flag, and a program that mixes
// the two APIs -- which is what every framework does -- reaches a driver entry
// point without ever calling cuInit itself. Real CUDA hides this because the
// runtime initializes the driver on first use; do the same. Resolved
// dynamically so libcudart keeps no link-time dependency on libcuda: if the
// driver shim is not loaded, there is nothing to initialize and nothing to do.
void init_driver_shim_if_loaded() {
  using CuInit = int (*)(unsigned int);
  if (auto fn = reinterpret_cast<CuInit>(dlsym(RTLD_DEFAULT, "cuInit"))) fn(0);
}

void ensure_init(State& s) {
  if (s.initialized) return;
  const char* gpu = std::getenv("VGPU_GPU");
  std::string id = gpu && gpu[0] ? gpu : "nvidia/h100";
  int count = 1;
  if (const char* c = std::getenv("VGPU_DEVICE_COUNT"); c && c[0]) count = std::atoi(c);
  vgpu::DeviceProfile profile = vgpu::load_gpu(id);
  // Optional: shrink advertised VRAM so VRAM-proportional stress tests run at
  // laptop scale (their size is a % of device memory). Functional behavior is
  // unchanged; only the working-set size the app chooses shrinks.
  if (const char* mb = std::getenv("VGPU_VRAM_MB"); mb && mb[0])
    profile.vram_bytes = static_cast<uint64_t>(std::strtoull(mb, nullptr, 10)) * 1024ull * 1024ull;
  s.rt = std::make_unique<vgpu::runtime::Runtime>(profile, count);
  s.initialized = true;
  init_driver_shim_if_loaded();
  if (!quiet())
    std::fprintf(stderr, "[vgpu] virtual GPU platform initialized: %d x %s (%s)\n", count,
                 profile.id.c_str(), profile.model.c_str());
}

// Sticky last error, per the runtime API contract.
thread_local cudaError_t g_last_error = cudaSuccess;

cudaError_t set_error(State& s, const vgpu::Error& e, const char* api) {
  using vgpu::Err;
  cudaError_t code;
  switch (e.code()) {
    case Err::OutOfMemory: code = cudaErrorMemoryAllocation; break;
    case Err::UnknownGpu: code = cudaErrorInvalidDevice; break;
    case Err::PtxParse: code = cudaErrorInvalidPtx; break;
    case Err::UnsupportedPtx:
    case Err::Unsupported: code = cudaErrorNotSupported; break;
    case Err::NotFound: code = cudaErrorInvalidDeviceFunction; break;
    case Err::ExecLimit: code = cudaErrorLaunchTimeout; break;
    case Err::InvalidPointer:
    case Err::UseAfterFree:
    case Err::OutOfBounds:
    case Err::MisalignedAccess:
    case Err::UninitializedRegister: code = cudaErrorIllegalAddress; break;
    default: code = cudaErrorInvalidValue; break;
  }
  (void)s;
  if (!quiet()) std::fprintf(stderr, "[vgpu] %s: %s\n", api, e.what());
  g_last_error = code;
  return code;
}

vgpu::runtime::Device& current(State& s) { return s.rt->device(s.current_device); }

// The memory of whichever device owns this pointer. Device VA windows are
// disjoint, so the address alone identifies the owner; falling back to the
// current device keeps the error message about an unmapped address rather than
// about the wrong device.
vgpu::MemoryManager& owner_memory(State& s, const void* p) {
  const uint64_t addr = reinterpret_cast<uint64_t>(p);
  for (int d = 0; d < s.rt->device_count(); ++d)
    if (s.rt->device(d).memory().owns(addr)) return s.rt->device(d).memory();
  return current(s).memory();
}

// Loads (once) the runtime module for a registered fatbin on the current device.
uint64_t module_on_current(State& s, RegisteredModule& m) {
  int dev = s.current_device;
  auto it = m.module_per_device.find(dev);
  if (it != m.module_per_device.end()) return it->second;
  uint64_t mid = s.rt->device(dev).load_module(m.ptx);
  m.module_per_device[dev] = mid;
  return mid;
}

template <class F>
cudaError_t guard(const char* api, F&& body) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  try {
    ensure_init(s);
    cudaError_t rc = body(s);
    if (rc != cudaSuccess) g_last_error = rc;
    return rc;
  } catch (const vgpu::Error& e) {
    return set_error(s, e, api);
  } catch (const std::exception& e) {
    if (!quiet()) std::fprintf(stderr, "[vgpu] %s: %s\n", api, e.what());
    g_last_error = cudaErrorUnknown;
    return cudaErrorUnknown;
  }
}

bool is_device_ptr(const void* p) {
  return reinterpret_cast<uint64_t>(p) >= vgpu::kDeviceVaBase;
}

// Pending chevron launch configuration, pushed by __cudaPushCallConfiguration
// and consumed by __cudaPopCallConfiguration inside the generated launch stub.
struct PendingConfig {
  dim3 grid, block;
  size_t shared = 0;
  void* stream = nullptr;
  bool valid = false;
};
thread_local PendingConfig g_pending_config;

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

/* ===================================================================== */
/* nvcc host boilerplate ABI (chevron launches lower onto these)         */
/* ===================================================================== */

VGPU_EXPORT void** __cudaRegisterFatBinary(void* fatCubin) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  try {
    ensure_init(s);
    auto rm = std::make_unique<RegisteredModule>();
    // fatCubin is a __fatBinC_Wrapper_t*; extract the best PTX image.
    auto ptxs = vgpu::cuda::extract_ptx(fatCubin);
    if (ptxs.empty()) {
      if (!quiet())
        std::fprintf(stderr,
                     "[vgpu] __cudaRegisterFatBinary: no PTX in fatbin (SASS-only build); rebuild "
                     "with an -arch that embeds PTX\n");
      // Return a handle anyway; the failure surfaces at launch with context.
    } else {
      size_t best = 0;
      for (size_t i = 1; i < ptxs.size(); ++i)
        if (ptxs[i].arch > ptxs[best].arch) best = i;
      rm->ptx = std::move(ptxs[best].text);
    }
    RegisteredModule* raw = rm.get();
    s.modules.push_back(std::move(rm));
    if (trace()) std::fprintf(stderr, "[vgpu][trace] __cudaRegisterFatBinary -> %p\n", (void*)raw);
    return reinterpret_cast<void**>(raw);
  } catch (const std::exception& e) {
    if (!quiet()) std::fprintf(stderr, "[vgpu] __cudaRegisterFatBinary: %s\n", e.what());
    return nullptr;
  }
}

VGPU_EXPORT void __cudaRegisterFatBinaryEnd(void**) {}
VGPU_EXPORT void __cudaUnregisterFatBinary(void**) {}
VGPU_EXPORT char __cudaInitModule(void**) { return 1; }

VGPU_EXPORT void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun, char* deviceFun,
                                        const char* deviceName, int thread_limit, void* tid,
                                        void* bid, void* bDim, void* gDim, int* wSize) {
  (void)deviceFun;
  (void)thread_limit;
  (void)tid;
  (void)bid;
  (void)bDim;
  (void)gDim;
  (void)wSize;
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  auto* mod = reinterpret_cast<RegisteredModule*>(fatCubinHandle);
  s.kernels[reinterpret_cast<const void*>(hostFun)] = {mod, deviceName};
  if (trace())
    std::fprintf(stderr, "[vgpu][trace] __cudaRegisterFunction: stub %p -> '%s'\n", (void*)hostFun,
                 deviceName);
}

VGPU_EXPORT void __cudaRegisterVar(void** /*handle*/, char* /*hostVar*/, char* /*deviceAddress*/,
                                   const char* /*deviceName*/, int /*ext*/, size_t /*size*/,
                                   int /*constant*/, int /*global*/) {
  // __device__/__constant__ variables: the workloads in scope do not read them
  // from the host, so recording is deferred. (No silent misbehavior: a kernel
  // that references an unregistered global still gets a clear symbol error.)
}

VGPU_EXPORT unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                                 void* stream) {
  g_pending_config = {gridDim, blockDim, sharedMem, stream, true};
  return 0;  // 0 == configuration accepted; the stub then runs the kernel body
}

VGPU_EXPORT cudaError_t __cudaPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem,
                                                   void* stream) {
  if (!g_pending_config.valid) return cudaErrorInvalidConfiguration;
  if (gridDim) *gridDim = g_pending_config.grid;
  if (blockDim) *blockDim = g_pending_config.block;
  if (sharedMem) *sharedMem = g_pending_config.shared;
  if (stream) *reinterpret_cast<void**>(stream) = g_pending_config.stream;
  g_pending_config.valid = false;
  return cudaSuccess;
}

/* ===================================================================== */
/* Launch                                                                */
/* ===================================================================== */

// CUDA 12.4+ chevron lowering: the stub calls __cudaGetKernel(&handle, hostFun)
// once, then __cudaLaunchKernel(handle, ...). We use the host stub pointer
// itself as the opaque kernel handle, so both map back to the registered entry.
typedef void* vgpu_cudaKernel_t;

VGPU_EXPORT cudaError_t __cudaGetKernel(vgpu_cudaKernel_t* kernel, const void* hostFun) {
  if (!kernel) return cudaErrorInvalidValue;
  *kernel = const_cast<void*>(hostFun);
  return cudaSuccess;
}

extern "C" cudaError_t cudaLaunchKernel(const void*, dim3, dim3, void**, size_t, cudaStream_t);

VGPU_EXPORT cudaError_t __cudaLaunchKernel(vgpu_cudaKernel_t kernel, dim3 gridDim, dim3 blockDim,
                                          void** args, size_t sharedMem, cudaStream_t stream) {
  return cudaLaunchKernel(kernel, gridDim, blockDim, args, sharedMem, stream);
}
VGPU_EXPORT cudaError_t __cudaLaunchKernel_ptsz(vgpu_cudaKernel_t kernel, dim3 gridDim, dim3 blockDim,
                                               void** args, size_t sharedMem, cudaStream_t stream) {
  return cudaLaunchKernel(kernel, gridDim, blockDim, args, sharedMem, stream);
}

// Defined with the CUDA Graph machinery below: records this launch instead of
// running it when its stream is capturing.
bool vgpu_record_launch_if_capturing(const void* func, dim3 grid, dim3 block, void** args,
                                     size_t sharedMem, cudaStream_t stream,
                                     const std::vector<uint32_t>& param_sizes);

VGPU_EXPORT cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args,
                                         size_t sharedMem, cudaStream_t stream) {
  return guard("cudaLaunchKernel", [&](State& s) -> cudaError_t {
    auto it = s.kernels.find(func);
    if (it == s.kernels.end()) {
      if (!quiet())
        std::fprintf(stderr, "[vgpu] cudaLaunchKernel: unregistered kernel stub %p\n", func);
      return cudaErrorInvalidDeviceFunction;
    }
    KernelInfo& ki = it->second;
    if (!ki.mod || ki.mod->ptx.empty())
      throw vgpu::Error::make(vgpu::Err::Unsupported,
                              "kernel '" + ki.entry_name +
                                  "' has no PTX (SASS-only fatbin); rebuild with embedded PTX");
    uint64_t mid = module_on_current(s, *ki.mod);
    vgpu::runtime::Device& dev = current(s);
    const vgpu::ptx::EntryFn* fn = dev.get_function(mid, ki.entry_name);

    // A kernel with parameters needs an argument array, and every slot in it
    // must be a real pointer: dereferencing what the caller passed is the one
    // place a bad argument turns into a crash instead of an error code.
    if (!fn->params.empty() && !args) {
      if (!quiet())
        std::fprintf(stderr, "[vgpu] cudaLaunchKernel: kernel '%s' takes %zu parameters but the "
                             "argument array is NULL\n", ki.entry_name.c_str(), fn->params.size());
      return cudaErrorInvalidValue;
    }
    for (size_t i = 0; i < fn->params.size(); ++i) {
      if (!args[i]) {
        if (!quiet())
          std::fprintf(stderr, "[vgpu] cudaLaunchKernel: kernel '%s' argument %zu is NULL\n",
                       ki.entry_name.c_str(), i);
        return cudaErrorInvalidValue;
      }
    }
    std::vector<uint32_t> param_sizes(fn->params.size());
    for (size_t i = 0; i < fn->params.size(); ++i) param_sizes[i] = fn->params[i].size;
    // Under stream capture the launch is recorded for later replay, not run.
    if (vgpu_record_launch_if_capturing(func, gridDim, blockDim, args, sharedMem, stream, param_sizes))
      return cudaSuccess;

    std::vector<std::vector<uint8_t>> kargs(fn->params.size());
    for (size_t i = 0; i < fn->params.size(); ++i) {
      kargs[i].resize(param_sizes[i]);
      std::memcpy(kargs[i].data(), args[i], param_sizes[i]);
    }
    vgpu::exec::LaunchConfig cfg;
    cfg.grid = {gridDim.x, gridDim.y, gridDim.z};
    cfg.block = {blockDim.x, blockDim.y, blockDim.z};
    cfg.shared_bytes = static_cast<uint32_t>(sharedMem);
    dev.launch(*fn, cfg, kargs, dev.symbols(mid));
    return cudaSuccess;
  });
}

/* ===================================================================== */
/* Device management                                                     */
/* ===================================================================== */

VGPU_EXPORT cudaError_t cudaGetDeviceCount(int* count) {
  return guard("cudaGetDeviceCount", [&](State& s) {
    if (!count) return cudaErrorInvalidValue;
    *count = s.rt->device_count();
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaSetDevice(int device) {
  return guard("cudaSetDevice", [&](State& s) {
    if (device < 0 || device >= s.rt->device_count()) return cudaErrorInvalidDevice;
    s.current_device = device;
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaGetDevice(int* device) {
  return guard("cudaGetDevice", [&](State& s) {
    if (!device) return cudaErrorInvalidValue;
    *device = s.current_device;
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaSetDeviceFlags(unsigned int) { return cudaSuccess; }
VGPU_EXPORT cudaError_t cudaGetDeviceFlags(unsigned int* flags) {
  if (flags) *flags = 0;
  return cudaSuccess;
}

// Work is synchronous here, so there is nothing to wait for -- but a failed
// launch must still be observable through synchronization, which is how most
// programs check for errors. Returning the sticky error (and clearing it, as
// CUDA does) keeps a failed kernel from looking like success.
VGPU_EXPORT cudaError_t cudaDeviceSynchronize(void) {
  cudaError_t e = g_last_error;
  g_last_error = cudaSuccess;
  return e;
}
VGPU_EXPORT cudaError_t cudaDeviceReset(void) {
  g_last_error = cudaSuccess;
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaThreadSynchronize(void) { return cudaDeviceSynchronize(); }

// CUDA 12 renamed this entry point when cudaDeviceProp changed shape, so a
// binary built against that toolkit calls cudaGetDeviceProperties_v2 while one
// built against CUDA 13 calls the plain name. The header's rename would also
// rewrite the definition below, leaving whichever name the build host happens
// to use -- so it is undone and both names are exported deliberately.
#ifdef cudaGetDeviceProperties
#undef cudaGetDeviceProperties
#endif
VGPU_EXPORT cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
  return guard("cudaGetDeviceProperties", [&](State& s) {
    if (!prop) return cudaErrorInvalidValue;
    if (device < 0 || device >= s.rt->device_count()) return cudaErrorInvalidDevice;
    const vgpu::DeviceProfile& p = s.rt->device(device).profile();
    std::memset(prop, 0, sizeof *prop);
    std::snprintf(prop->name, sizeof prop->name, "%s", p.model.c_str());
    prop->totalGlobalMem = p.vram_bytes;
    prop->warpSize = static_cast<int>(p.warp_size);
    prop->major = p.cc_major;
    prop->minor = p.cc_minor;
    prop->multiProcessorCount = static_cast<int>(p.limits.multiprocessors);
    prop->maxThreadsPerBlock = static_cast<int>(p.limits.max_threads_per_block);
    prop->maxThreadsPerMultiProcessor = static_cast<int>(p.limits.max_threads_per_sm);
    prop->regsPerMultiprocessor = static_cast<int>(p.limits.registers_per_sm);
    prop->maxBlocksPerMultiProcessor = static_cast<int>(p.limits.max_blocks_per_sm);
    prop->sharedMemPerBlock = p.limits.shared_mem_per_block;
    prop->sharedMemPerBlockOptin = p.limits.shared_mem_per_block_optin;
    prop->sharedMemPerMultiprocessor = p.limits.shared_mem_per_block_optin;
    prop->regsPerBlock = static_cast<int>(p.limits.registers_per_block);
    prop->maxThreadsDim[0] = static_cast<int>(p.limits.max_block_dim[0]);
    prop->maxThreadsDim[1] = static_cast<int>(p.limits.max_block_dim[1]);
    prop->maxThreadsDim[2] = static_cast<int>(p.limits.max_block_dim[2]);
    prop->maxGridSize[0] = static_cast<int>(p.limits.max_grid_dim[0]);
    prop->maxGridSize[1] = static_cast<int>(p.limits.max_grid_dim[1]);
    prop->maxGridSize[2] = static_cast<int>(p.limits.max_grid_dim[2]);
    prop->totalConstMem = 65536;
    // Performance-related fields (clocks, bus width) are placeholders — VirtualGPU
    // models no performance. CUDA 13 dropped clockRate/memoryClockRate/computeMode
    // from cudaDeviceProp entirely.
    prop->memoryBusWidth = 256;  // placeholder
    prop->l2CacheSize = 8 * 1024 * 1024;
    prop->concurrentKernels = 1;
    prop->unifiedAddressing = 1;
    prop->canMapHostMemory = 1;
    prop->pciBusID = device + 1;
    prop->pciDeviceID = 0;
    prop->integrated = 0;
    prop->ECCEnabled = 0;
    prop->maxThreadsPerBlock = static_cast<int>(p.limits.max_threads_per_block);
    // Deterministic fake UUID (matches the driver shim scheme).
    unsigned char b[16] = {'V', 'G', 'P', 'U'};
    uint32_t h = 2166136261u;
    for (char c : p.id) h = (h ^ static_cast<unsigned char>(c)) * 16777619u;
    std::memcpy(b + 4, &h, 4);
    b[8] = static_cast<unsigned char>(device);
    std::memcpy(&prop->uuid, b, 16);
    return cudaSuccess;
  });
}
VGPU_EXPORT cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp* prop, int device) {
  return cudaGetDeviceProperties(prop, device);
}

VGPU_EXPORT cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
  return guard("cudaDeviceGetAttribute", [&](State& s) {
    if (!value) return cudaErrorInvalidValue;
    if (device < 0 || device >= s.rt->device_count()) return cudaErrorInvalidDevice;
    const vgpu::DeviceProfile& p = s.rt->device(device).profile();
    // Named rather than numbered: these used to be the ordinals of
    // cudaDeviceAttr, which is a table that only has to be renumbered once.
    switch (attr) {
      case cudaDevAttrMaxThreadsPerBlock: *value = static_cast<int>(p.limits.max_threads_per_block); break;
      case cudaDevAttrMaxSharedMemoryPerBlock: *value = static_cast<int>(p.limits.shared_mem_per_block); break;
      case cudaDevAttrWarpSize: *value = static_cast<int>(p.warp_size); break;
      case cudaDevAttrMultiProcessorCount: *value = static_cast<int>(p.limits.multiprocessors); break;
      case 39: *value = static_cast<int>(p.limits.max_threads_per_sm); break;  // MaxThreads/SM
      case 82: *value = static_cast<int>(p.limits.registers_per_sm); break;    // MaxRegistersPerSM
      case 75: *value = p.cc_major; break;                                       // ComputeCapabilityMajor
      case 76: *value = p.cc_minor; break;                                       // ComputeCapabilityMinor
      default:
        if (trace()) std::fprintf(stderr, "[vgpu][trace] cudaDeviceGetAttribute(%d) -> 0\n", attr);
        *value = 0;
    }
    return cudaSuccess;
  });
}

// cudaFuncGetAttributes: report the kernel's actual register footprint and
// local frame, which is what occupancy tools and tuning scripts read.
//
// Uses the toolkit's own cudaFuncAttributes: the layout is version-specific,
// and writing a hand-rolled copy produced garbage (numRegs in the tens of
// thousands) until this was switched to the real type.
static_assert(sizeof(cudaFuncAttributes) > 0, "cudaFuncAttributes must come from the toolkit");

VGPU_EXPORT cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func) {
  return guard("cudaFuncGetAttributes", [&](State& s) -> cudaError_t {
    cudaFuncAttributes* a = attr;
    if (!a) return cudaErrorInvalidValue;
    auto it = s.kernels.find(func);
    if (it == s.kernels.end()) return cudaErrorInvalidDeviceFunction;
    KernelInfo& ki = it->second;
    if (!ki.mod || ki.mod->ptx.empty()) return cudaErrorInvalidDeviceFunction;
    uint64_t mid = module_on_current(s, *ki.mod);
    vgpu::runtime::Device& dev = current(s);
    const vgpu::ptx::EntryFn* fn = dev.get_function(mid, ki.entry_name);
    const vgpu::DeviceProfile& p = dev.profile();
    auto res = vgpu::exec::kernel_resources(*fn, p, p.limits.max_threads_per_block, 0);
    std::memset(a, 0, sizeof *a);
    a->numRegs = static_cast<int>(res.usage.regs_per_thread);
    a->localSizeBytes = res.usage.local_bytes;
    a->sharedSizeBytes = fn->static_shared_size;
    a->maxThreadsPerBlock = static_cast<int>(p.limits.max_threads_per_block);
    a->ptxVersion = 83;
    a->binaryVersion = p.cc_major * 10 + p.cc_minor;
    a->maxDynamicSharedSizeBytes = static_cast<int>(p.limits.shared_mem_per_block_optin);
    return cudaSuccess;
  });
}

// Real occupancy, from the same analysis the launch path uses.
VGPU_EXPORT cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                                      const void* func,
                                                                      int blockSize,
                                                                      size_t dynamicSMemSize) {
  return guard("cudaOccupancyMaxActiveBlocksPerMultiprocessor", [&](State& s) -> cudaError_t {
    if (!numBlocks || blockSize <= 0) return cudaErrorInvalidValue;
    auto it = s.kernels.find(func);
    if (it == s.kernels.end()) return cudaErrorInvalidDeviceFunction;
    KernelInfo& ki = it->second;
    if (!ki.mod || ki.mod->ptx.empty()) return cudaErrorInvalidDeviceFunction;
    uint64_t mid = module_on_current(s, *ki.mod);
    vgpu::runtime::Device& dev = current(s);
    const vgpu::ptx::EntryFn* fn = dev.get_function(mid, ki.entry_name);
    auto res = vgpu::exec::kernel_resources(*fn, dev.profile(),
                                            static_cast<uint32_t>(blockSize),
                                            static_cast<uint32_t>(dynamicSMemSize));
    *numBlocks = static_cast<int>(res.occupancy.blocks_per_sm);
    return cudaSuccess;
  });
}
VGPU_EXPORT cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* n, const void* f, int bs, size_t dyn, unsigned int) {
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(n, f, bs, dyn);
}

VGPU_EXPORT cudaError_t cudaDeviceCanAccessPeer(int* can, int device, int peerDevice) {
  return guard("cudaDeviceCanAccessPeer", [&](State& s) {
    if (!can) return cudaErrorInvalidValue;
    if (device < 0 || device >= s.rt->device_count() || peerDevice < 0 ||
        peerDevice >= s.rt->device_count())
      return cudaErrorInvalidDevice;
    // Distinct virtual devices can always reach each other; a device is not
    // its own peer (matching the runtime API contract).
    *can = (device != peerDevice) ? 1 : 0;
    return cudaSuccess;
  });
}
VGPU_EXPORT cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int) {
  return guard("cudaDeviceEnablePeerAccess", [&](State& s) {
    if (peerDevice < 0 || peerDevice >= s.rt->device_count()) return cudaErrorInvalidDevice;
    if (peerDevice == s.current_device) return cudaErrorInvalidDevice;
    return cudaSuccess;  // peer mappings are implicit in this engine
  });
}
VGPU_EXPORT cudaError_t cudaDeviceDisablePeerAccess(int) { return cudaSuccess; }

// Copies between two virtual devices' memories. Device pointers are only
// meaningful on their own device, so each side is resolved against its own
// MemoryManager.
VGPU_EXPORT cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice,
                                       size_t count) {
  return guard("cudaMemcpyPeer", [&](State& s) {
    if (dstDevice < 0 || dstDevice >= s.rt->device_count() || srcDevice < 0 ||
        srcDevice >= s.rt->device_count())
      return cudaErrorInvalidDevice;
    if (count == 0) return cudaSuccess;
    std::vector<uint8_t> tmp(count);
    s.rt->device(srcDevice).memory().read(reinterpret_cast<uint64_t>(src), tmp.data(), count);
    s.rt->device(dstDevice).memory().write(reinterpret_cast<uint64_t>(dst), tmp.data(), count);
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaMemGetInfo(size_t* free_b, size_t* total_b) {
  return guard("cudaMemGetInfo", [&](State& s) {
    if (!free_b || !total_b) return cudaErrorInvalidValue;
    vgpu::MemoryManager& mm = current(s).memory();
    *total_b = static_cast<size_t>(mm.capacity());
    *free_b = static_cast<size_t>(mm.capacity() - mm.used());
    return cudaSuccess;
  });
}

/* ===================================================================== */
/* Memory                                                                */
/* ===================================================================== */

VGPU_EXPORT cudaError_t cudaMalloc(void** ptr, size_t size) {
  return guard("cudaMalloc", [&](State& s) {
    if (!ptr) return cudaErrorInvalidValue;
    *ptr = reinterpret_cast<void*>(current(s).memory().alloc(size));
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaFree(void* ptr) {
  return guard("cudaFree", [&](State& s) {
    if (!ptr) return cudaSuccess;  // cudaFree(NULL) is a documented no-op
    owner_memory(s, ptr).free(reinterpret_cast<uint64_t>(ptr));
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
  return guard("cudaMemcpy", [&](State& s) {
    auto _t0 = std::chrono::steady_clock::now();
    bool dd = is_device_ptr(dst), sd = is_device_ptr(src);
    // Under unified addressing a device pointer names its own device, so each
    // side is resolved against the device that owns it rather than against
    // whichever device happens to be current. Without this a
    // cudaMemcpyDeviceToDevice between two devices reads the current device's
    // memory at the same numeric address -- silently, and with the wrong bytes.
    vgpu::MemoryManager& dmm = dd ? owner_memory(s, dst) : current(s).memory();
    vgpu::MemoryManager& smm = sd ? owner_memory(s, src) : current(s).memory();
    if (kind == cudaMemcpyDefault) kind = dd && sd ? cudaMemcpyDeviceToDevice
                                          : dd      ? cudaMemcpyHostToDevice
                                          : sd      ? cudaMemcpyDeviceToHost
                                                    : cudaMemcpyHostToHost;
    switch (kind) {
      case cudaMemcpyHostToDevice:
        dmm.write(reinterpret_cast<uint64_t>(dst), src, count);
        break;
      case cudaMemcpyDeviceToHost:
        smm.read(reinterpret_cast<uint64_t>(src), dst, count);
        break;
      case cudaMemcpyDeviceToDevice: {
        std::vector<uint8_t> tmp(count);
        smm.read(reinterpret_cast<uint64_t>(src), tmp.data(), count);
        dmm.write(reinterpret_cast<uint64_t>(dst), tmp.data(), count);
        break;
      }
      case cudaMemcpyHostToHost:
        std::memcpy(dst, src, count);
        break;
      default:
        return cudaErrorInvalidValue;
    }
    current(s).note_transfer(
        count, std::chrono::duration<double>(std::chrono::steady_clock::now() - _t0).count());
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                                        cudaMemcpyKind kind, cudaStream_t) {
  return cudaMemcpy(dst, src, count, kind);
}

// Pitched allocations. Image code -- NPP included -- allocates rows padded to
// an alignment and then copies rectangles rather than runs, so these are not
// optional for anything that handles pictures.
VGPU_EXPORT cudaError_t cudaMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height) {
  return guard("cudaMallocPitch", [&](State& s) {
    if (!ptr || !pitch) return cudaErrorInvalidValue;
    if (width == 0 || height == 0) {
      *ptr = nullptr;
      *pitch = 0;
      return cudaSuccess;
    }
    // 512 bytes is the alignment CUDA documents for pitched allocations and the
    // one NPP's own allocators assume.
    *pitch = (width + 511) / 512 * 512;
    *ptr = reinterpret_cast<void*>(current(s).memory().alloc(*pitch * height));
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                                     size_t width, size_t height, cudaMemcpyKind kind) {
  if (width == 0 || height == 0) return cudaSuccess;
  if (!dst || !src) return cudaErrorInvalidValue;
  // A rectangle is a run of rows; each row goes through the same path as a
  // linear copy, so the owning-device resolution applies to it too.
  for (size_t y = 0; y < height; ++y) {
    const cudaError_t e =
        cudaMemcpy(static_cast<char*>(dst) + y * dpitch,
                   static_cast<const char*>(src) + y * spitch, width, kind);
    if (e != cudaSuccess) return e;
  }
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                                          size_t width, size_t height, cudaMemcpyKind kind,
                                          cudaStream_t) {
  return cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
}


VGPU_EXPORT cudaError_t cudaMemset(void* dst, int value, size_t count) {
  return guard("cudaMemset", [&](State& s) {
    const uint8_t byte = static_cast<uint8_t>(value);
    owner_memory(s, dst).fill(reinterpret_cast<uint64_t>(dst), &byte, 1, count);
    return cudaSuccess;
  });
}
VGPU_EXPORT cudaError_t cudaMemsetAsync(void* dst, int value, size_t count, cudaStream_t) {
  return cudaMemset(dst, value, count);
}
VGPU_EXPORT cudaError_t cudaMemset2D(void* dst, size_t pitch, int value, size_t width,
                                     size_t height) {
  if (width == 0 || height == 0) return cudaSuccess;
  if (!dst) return cudaErrorInvalidValue;
  for (size_t y = 0; y < height; ++y) {
    const cudaError_t e = cudaMemset(static_cast<char*>(dst) + y * pitch, value, width);
    if (e != cudaSuccess) return e;
  }
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaMemset2DAsync(void* dst, size_t pitch, int value, size_t width,
                                          size_t height, cudaStream_t) {
  return cudaMemset2D(dst, pitch, value, width, height);
}


VGPU_EXPORT cudaError_t cudaMallocHost(void** ptr, size_t size) {
  return guard("cudaMallocHost", [&](State&) {
    if (!ptr) return cudaErrorInvalidValue;
    void* p = std::aligned_alloc(4096, (size + 4095) / 4096 * 4096);
    if (!p) return cudaErrorMemoryAllocation;
    st().host_allocs[p] = size;
    *ptr = p;
    return cudaSuccess;
  });
}
VGPU_EXPORT cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int) {
  return cudaMallocHost(ptr, size);
}
VGPU_EXPORT cudaError_t cudaFreeHost(void* ptr) {
  return guard("cudaFreeHost", [&](State& s) {
    if (!ptr) return cudaSuccess;
    s.host_allocs.erase(ptr);
    std::free(ptr);
    return cudaSuccess;
  });
}
VGPU_EXPORT cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src,
                                            int srcDevice, size_t count, cudaStream_t) {
  return cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
}

/* ===================================================================== */
/* Streams and events (synchronous / wall-clock)                         */
/* ===================================================================== */

VGPU_EXPORT cudaError_t cudaStreamCreate(cudaStream_t* s) {
  if (s) *s = reinterpret_cast<cudaStream_t>(0x1);
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned int) {
  return cudaStreamCreate(s);
}

/* ---- entry points PyTorch and other frameworks link against ----
 *
 * A framework resolves these at load time, so a missing one stops the import
 * before any kernel runs. Everything here is either a faithful implementation
 * or an honest error: reporting success for something not actually done would
 * make a framework believe it had memory or a capability it does not have.
 */

// Streams are executed inline, so every priority is equally honoured. CUDA
// reports the range as [greatest, least] with lower meaning higher priority.
VGPU_EXPORT cudaError_t cudaDeviceGetStreamPriorityRange(int* least, int* greatest) {
  if (least) *least = 0;
  if (greatest) *greatest = 0;
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaStreamCreateWithPriority(cudaStream_t* s, unsigned int, int) {
  return cudaStreamCreate(s);
}

// The async allocator maps onto the ordinary one: work is synchronous here, so
// a stream-ordered allocation is already complete when it returns.
VGPU_EXPORT cudaError_t cudaMallocAsync(void** ptr, size_t size, cudaStream_t) {
  return cudaMalloc(ptr, size);
}
VGPU_EXPORT cudaError_t cudaFreeAsync(void* ptr, cudaStream_t) { return cudaFree(ptr); }
VGPU_EXPORT cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* pool, int) {
  if (!pool) return cudaErrorInvalidValue;
  *pool = reinterpret_cast<cudaMemPool_t>(0x1);
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t, cudaMemPoolAttr, void*) {
  return cudaSuccess;  // pool trimming and thresholds have no effect here
}
VGPU_EXPORT cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t, cudaMemPoolAttr attr, void* value) {
  if (!value) return cudaErrorInvalidValue;
  // The numeric attributes are byte counts; nothing is pooled, so they are 0.
  switch (attr) {
    case cudaMemPoolReuseFollowEventDependencies:
    case cudaMemPoolReuseAllowOpportunistic:
    case cudaMemPoolReuseAllowInternalDependencies:
      *static_cast<int*>(value) = 0;
      return cudaSuccess;
    default:
      *static_cast<unsigned long long*>(value) = 0;
      return cudaSuccess;
  }
}
VGPU_EXPORT cudaError_t cudaMemPoolSetAccess(cudaMemPool_t, const cudaMemAccessDesc*, size_t) {
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaMemPoolTrimTo(cudaMemPool_t, size_t) { return cudaSuccess; }

// Page-locking host memory changes nothing when the "device" shares the host's
// address space, so registration succeeds and maps to the same pointer.
VGPU_EXPORT cudaError_t cudaHostRegister(void* p, size_t, unsigned int) {
  return p ? cudaSuccess : cudaErrorInvalidValue;
}
VGPU_EXPORT cudaError_t cudaHostUnregister(void* p) {
  return p ? cudaSuccess : cudaErrorInvalidValue;
}
VGPU_EXPORT cudaError_t cudaHostGetDevicePointer(void** dev, void* host, unsigned int) {
  if (!dev || !host) return cudaErrorInvalidValue;
  // Host allocations are not addressable by device code here: a kernel would
  // resolve the pointer against device memory and read the wrong bytes. Say so
  // rather than hand back something that appears to work.
  return cudaErrorInvalidValue;
}

// Where a pointer lives. Frameworks branch on this to pick a copy path, so
// getting it wrong sends a device buffer through a host memcpy.
VGPU_EXPORT cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attr, const void* p) {
  return guard("cudaPointerGetAttributes", [&](State& st) {
    if (!attr) return cudaErrorInvalidValue;
    std::memset(attr, 0, sizeof *attr);
    const uint64_t a = reinterpret_cast<uint64_t>(p);
    if (a >= vgpu::kDeviceVaBase) {
      attr->type = cudaMemoryTypeDevice;
      attr->device = static_cast<int>((a - vgpu::kDeviceVaBase) / vgpu::kDeviceVaStride);
      attr->devicePointer = const_cast<void*>(p);
    } else {
      attr->type = cudaMemoryTypeUnregistered;
      attr->device = st.current_device;
      attr->hostPointer = const_cast<void*>(p);
    }
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaDeviceGetPCIBusId(char* buf, int len, int device) {
  return guard("cudaDeviceGetPCIBusId", [&](State& st) {
    if (!buf || len <= 0) return cudaErrorInvalidValue;
    if (device < 0 || device >= static_cast<int>(st.rt->device_count()))
      return cudaErrorInvalidDevice;
    // Same identity NVML reports, so a framework that parses one and compares
    // against the other sees a consistent device.
    vgpu::telemetry::DeviceSample snap{};
    vgpu::telemetry::describe_device(st.rt->device(device).profile(), device, &snap);
    std::snprintf(buf, static_cast<size_t>(len), "%s", snap.bus_id);
    return cudaSuccess;
  });
}

// Callbacks run immediately: the stream they are attached to has no work left
// outstanding by the time this is reached.
VGPU_EXPORT cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t cb,
                                              void* user, unsigned int) {
  if (cb) cb(stream, cudaSuccess, user);
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaFuncSetAttribute(const void*, cudaFuncAttribute, int) {
  // Opting into a larger shared-memory carveout is a hardware tuning knob; the
  // interpreter honours whatever a launch asks for.
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) {
  if (!mode) return cudaErrorInvalidValue;
  *mode = cudaStreamCaptureModeGlobal;
  return cudaSuccess;
}

// IPC shares device memory between processes through the driver. Device memory
// here is this process's heap, so a handle would be meaningless in another
// process -- and silently producing one would corrupt data rather than fail.
VGPU_EXPORT cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t*, void*) {
  return cudaErrorNotSupported;
}
VGPU_EXPORT cudaError_t cudaIpcOpenMemHandle(void**, cudaIpcMemHandle_t, unsigned int) {
  return cudaErrorNotSupported;
}
VGPU_EXPORT cudaError_t cudaIpcCloseMemHandle(void*) { return cudaErrorNotSupported; }
VGPU_EXPORT cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t*, cudaEvent_t) {
  return cudaErrorNotSupported;
}
VGPU_EXPORT cudaError_t cudaIpcOpenEventHandle(cudaEvent_t*, cudaIpcEventHandle_t) {
  return cudaErrorNotSupported;
}

// Graphs are captured and replayed by running the work inline, so a captured
// graph has no node list to walk. Report an empty graph rather than a count a
// caller would then try to read nodes out of.
VGPU_EXPORT cudaError_t cudaGraphGetNodes(cudaGraph_t, cudaGraphNode_t*, size_t* numNodes) {
  if (numNodes) *numNodes = 0;
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaGraphDebugDotPrint(cudaGraph_t, const char* path, unsigned int) {
  if (!path) return cudaErrorInvalidValue;
  std::FILE* f = std::fopen(path, "w");
  if (!f) return cudaErrorOperatingSystem;
  std::fprintf(f, "digraph vgpu {\n  // graphs execute inline; no captured nodes\n}\n");
  std::fclose(f);
  return cudaSuccess;
}

// The extended launch form carries an attribute list (cluster dims, cooperative
// launch, programmatic dependencies). None changes what the interpreter does
// with the grid, so the launch itself is the ordinary path.
VGPU_EXPORT cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* cfg, const void* func,
                                            void** args) {
  if (!cfg) return cudaErrorInvalidValue;
  return cudaLaunchKernel(func, cfg->gridDim, cfg->blockDim, args, cfg->dynamicSmemBytes,
                          cfg->stream);
}

// Profiler control is a no-op: there is no external profiler attached, and a
// framework toggling it must not fail.
VGPU_EXPORT cudaError_t cudaProfilerStart(void) { return cudaSuccess; }
VGPU_EXPORT cudaError_t cudaProfilerStop(void) { return cudaSuccess; }

VGPU_EXPORT cudaError_t cudaStreamGetPriority(cudaStream_t, int* priority) {
  if (!priority) return cudaErrorInvalidValue;
  *priority = 0;  // the single priority this implementation offers
  return cudaSuccess;
}

// A host callback is enqueued behind the stream's work. Work is synchronous
// here, so everything before it has already finished and it runs now.
VGPU_EXPORT cudaError_t cudaLaunchHostFunc(cudaStream_t, cudaHostFn_t fn, void* user) {
  if (!fn) return cudaErrorInvalidValue;
  fn(user);
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream,
                                                    cudaStreamCaptureStatus* status,
                                                    unsigned long long* id, cudaGraph_t* graph,
                                                    const cudaGraphNode_t** deps,
                                                    size_t* numDeps) {
  const cudaError_t e = cudaStreamIsCapturing(stream, status);
  if (e != cudaSuccess) return e;
  if (id) *id = 0;
  if (graph) *graph = nullptr;
  if (deps) *deps = nullptr;
  if (numDeps) *numDeps = 0;
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
VGPU_EXPORT cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaDeviceSynchronize(); }
VGPU_EXPORT cudaError_t cudaStreamQuery(cudaStream_t) { return cudaSuccess; }
VGPU_EXPORT cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned int) {
  return cudaSuccess;
}
// cudaStreamIsCapturing is defined with the CUDA Graph machinery below.

VGPU_EXPORT cudaError_t cudaEventCreate(cudaEvent_t* e) {
  if (e) *e = reinterpret_cast<cudaEvent_t>(std::malloc(sizeof(long long)));
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned int) {
  return cudaEventCreate(e);
}
VGPU_EXPORT cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaSuccess; }
VGPU_EXPORT cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
VGPU_EXPORT cudaError_t cudaEventQuery(cudaEvent_t) { return cudaSuccess; }
VGPU_EXPORT cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) {
  if (ms) *ms = 0.0f;  // VirtualGPU does not model timing
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaEventDestroy(cudaEvent_t e) {
  std::free(e);
  return cudaSuccess;
}

/* ===================================================================== */
/* Errors and versions                                                   */
/* ===================================================================== */

VGPU_EXPORT cudaError_t cudaGetLastError(void) {
  cudaError_t e = g_last_error;
  g_last_error = cudaSuccess;
  return e;
}
VGPU_EXPORT cudaError_t cudaPeekAtLastError(void) { return g_last_error; }

VGPU_EXPORT const char* cudaGetErrorString(cudaError_t error) {
  switch (error) {
    case cudaSuccess: return "no error";
    case cudaErrorMemoryAllocation: return "out of memory";
    case cudaErrorInvalidValue: return "invalid argument";
    case cudaErrorInvalidDevice: return "invalid device ordinal";
    case cudaErrorInvalidDeviceFunction: return "invalid device function";
    case cudaErrorInvalidPtx: return "a PTX JIT compilation failed";
    case cudaErrorIllegalAddress: return "an illegal memory access was encountered";
    case cudaErrorLaunchTimeout: return "the launch timed out and was terminated";
    case cudaErrorNotSupported: return "operation not supported";
    default: return "unknown error";
  }
}
VGPU_EXPORT const char* cudaGetErrorName(cudaError_t error) {
  switch (error) {
    case cudaSuccess: return "cudaSuccess";
    case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
    case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
    case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
    case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction";
    case cudaErrorInvalidPtx: return "cudaErrorInvalidPtx";
    case cudaErrorIllegalAddress: return "cudaErrorIllegalAddress";
    case cudaErrorNotSupported: return "cudaErrorNotSupported";
    default: return "cudaErrorUnknown";
  }
}

VGPU_EXPORT cudaError_t cudaDriverGetVersion(int* v) {
  if (v) *v = kDriverVersion;
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaRuntimeGetVersion(int* v) {
  if (v) *v = kRuntimeVersion;
  return cudaSuccess;
}

/* ===================================================================== */
/* CUDA Graphs: real stream capture and replay                           */
/*                                                                       */
/* Capture must genuinely record instead of executing — workloads verify */
/* a replayed graph's output against a directly-launched reference, so a */
/* no-op stub would silently produce wrong results.                      */
/* ===================================================================== */

namespace {

struct RecordedLaunch {
  const void* func = nullptr;
  dim3 grid, block;
  size_t shared = 0;
  std::vector<std::vector<uint8_t>> arg_bytes;  // deep copy of parameter values
  std::vector<void*> arg_ptrs;                  // rebuilt to point at arg_bytes
};

struct GraphRec {
  std::vector<RecordedLaunch> launches;
};

std::mutex g_graph_mu;
std::unordered_map<void*, std::unique_ptr<GraphRec>> g_graphs;      // graph handles
std::unordered_map<void*, std::unique_ptr<GraphRec>> g_graph_execs; // instantiated graphs
// Streams currently capturing. VirtualGPU's streams are all the same
// synchronous engine, so capture state is keyed by the stream handle value.
std::unordered_map<void*, std::unique_ptr<GraphRec>> g_capturing;

// Returns the capture buffer for `stream`, or nullptr when not capturing.
GraphRec* capture_target(cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto it = g_capturing.find(reinterpret_cast<void*>(stream));
  return it == g_capturing.end() ? nullptr : it->second.get();
}

}  // namespace

// Records a launch during capture. Returns true if it was recorded (and must
// therefore NOT execute now).
bool vgpu_record_launch_if_capturing(const void* func, dim3 grid, dim3 block, void** args,
                                     size_t sharedMem, cudaStream_t stream,
                                     const std::vector<uint32_t>& param_sizes) {
  GraphRec* g = capture_target(stream);
  if (!g) return false;
  RecordedLaunch rl;
  rl.func = func;
  rl.grid = grid;
  rl.block = block;
  rl.shared = sharedMem;
  rl.arg_bytes.resize(param_sizes.size());
  for (size_t i = 0; i < param_sizes.size(); ++i) {
    rl.arg_bytes[i].resize(param_sizes[i]);
    std::memcpy(rl.arg_bytes[i].data(), args[i], param_sizes[i]);
  }
  std::lock_guard<std::mutex> lock(g_graph_mu);
  g->launches.push_back(std::move(rl));
  return true;
}

VGPU_EXPORT cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
  (void)mode;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  g_capturing[reinterpret_cast<void*>(stream)] = std::make_unique<GraphRec>();
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto it = g_capturing.find(reinterpret_cast<void*>(stream));
  if (it == g_capturing.end()) return cudaErrorStreamCaptureImplicit;
  auto graph = std::move(it->second);
  g_capturing.erase(it);
  void* handle = graph.get();
  g_graphs[handle] = std::move(graph);
  if (pGraph) *pGraph = static_cast<cudaGraph_t>(handle);
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pExec, cudaGraph_t graph,
                                             unsigned long long) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto it = g_graphs.find(static_cast<void*>(graph));
  if (it == g_graphs.end()) return cudaErrorInvalidValue;
  auto exec = std::make_unique<GraphRec>(*it->second);  // snapshot at instantiate time
  void* handle = exec.get();
  g_graph_execs[handle] = std::move(exec);
  if (pExec) *pExec = static_cast<cudaGraphExec_t>(handle);
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* pExec, cudaGraph_t graph,
                                                      unsigned long long flags) {
  return cudaGraphInstantiate(pExec, graph, flags);
}

VGPU_EXPORT cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
  std::vector<RecordedLaunch> replay;
  {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    auto it = g_graph_execs.find(static_cast<void*>(exec));
    if (it == g_graph_execs.end()) return cudaErrorInvalidValue;
    replay = it->second->launches;  // copy so we can run without holding the lock
  }
  for (auto& rl : replay) {
    std::vector<void*> ptrs(rl.arg_bytes.size());
    for (size_t i = 0; i < rl.arg_bytes.size(); ++i) ptrs[i] = rl.arg_bytes[i].data();
    cudaError_t rc = cudaLaunchKernel(rl.func, rl.grid, rl.block, ptrs.data(), rl.shared, stream);
    if (rc != cudaSuccess) return rc;
  }
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  g_graphs.erase(static_cast<void*>(graph));
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  g_graph_execs.erase(static_cast<void*>(exec));
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaStreamIsCapturing(cudaStream_t stream,
                                              cudaStreamCaptureStatus* status) {
  if (status)
    *status = capture_target(stream) ? cudaStreamCaptureStatusActive
                                     : cudaStreamCaptureStatusNone;
  return cudaSuccess;
}
