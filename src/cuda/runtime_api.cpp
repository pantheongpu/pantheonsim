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
#include <functional>
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
#include "vgpu/faults.hpp"
#include "vgpu/profiling.hpp"
#include "vgpu/telemetry.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"

namespace {

// The version this shim reports, taken from the toolkit it was built against
// rather than written down. The soname is already derived that way -- a build
// with a CUDA 12 nvcc produces libcudart.so.12 -- and a hardcoded number meant
// that build would call itself .so.12 and then report 13000, which is the kind
// of disagreement a caller has no way to make sense of.
constexpr int kRuntimeVersion = CUDART_VERSION;
// The driver is at least as new as the runtime it serves; reporting the same
// number is what a matched pair looks like.
constexpr int kDriverVersion = CUDART_VERSION;

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
  // Managed allocations, kept apart from host_allocs because freeing one has
  // to unmap it from the device side as well.
  std::map<void*, size_t> managed_allocs;
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
  // A program that only uses the runtime API never reaches cuInit, and a
  // profiler attached to it would otherwise never be invited in.
  vgpu::load_injection_library();
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
    // "trap" is what a failed device assert and an unreachable path compile to,
    // and hardware surfaces it as an illegal instruction.
    case Err::Trap: code = cudaErrorIllegalInstruction; break;
    case Err::DeviceAssert: code = cudaErrorAssert; break;
    // A data race is this simulator's own finding rather than a CUDA condition.
    // "unspecified launch failure" is the closest real code, and it is at least
    // true that the launch did not produce a result anyone should use.
    case Err::DataRace: code = cudaErrorLaunchFailure; break;
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
// Called by the operations that cannot be recorded. Marks any in-flight
// capture as unusable rather than letting it silently omit the work.
void vgpu_invalidate_capture(const char* what);
bool vgpu_record_memcpy_if_capturing(void* dst, const void* src, size_t bytes,
                                     cudaMemcpyKind kind, cudaStream_t stream);
bool vgpu_record_memset_if_capturing(void* dst, int value, size_t bytes, cudaStream_t stream);

bool vgpu_record_launch_if_capturing(const void* func, dim3 grid, dim3 block, void** args,
                                     size_t sharedMem, cudaStream_t stream,
                                     const std::vector<uint32_t>& param_sizes);
bool vgpu_record_host_op_if_capturing(cudaStream_t stream, std::function<void()> op);

// The body of both launch entry points. `cooperative` is the only difference,
// and it changes one thing: whether the blocks are resident together and may
// wait on each other. See the scheduler note in interpreter.cpp.
static cudaError_t launch_kernel_impl(const char* api, const void* func, dim3 gridDim,
                                      dim3 blockDim, void** args, size_t sharedMem,
                                      cudaStream_t stream, bool cooperative,
                                      std::array<uint32_t, 3> cluster = {0, 0, 0}) {
  return guard(api, [&](State& s) -> cudaError_t {
    auto it = s.kernels.find(func);
    if (it == s.kernels.end()) {
      if (!quiet())
        std::fprintf(stderr, "[vgpu] cudaLaunchKernel: unregistered kernel stub %p\n", func);
      return cudaErrorInvalidDeviceFunction;
    }
    KernelInfo& ki = it->second;
    // A profiler wants the launch even when nothing else does; recording is a
    // relaxed load away when nobody is listening.
    const bool profiling = vgpu::profiling::enabled();
    const uint64_t t0 = profiling ? vgpu::profiling::now_ns() : 0;
    if (trace())
      std::fprintf(stderr, "[vgpu][trace] launch %s grid %ux%ux%u block %ux%ux%u shared %zu\n",
                   ki.entry_name.c_str(), gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y,
                   blockDim.z, sharedMem);
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

    if (vgpu::faults::should_fail(vgpu::faults::Op::Launch)) {
      if (!quiet())
        std::fprintf(stderr, "[vgpu] %s: failing this launch, as VGPU_FAIL_LAUNCH asked. This is "
                             "injected, not a real failure\n", api);
      return cudaErrorLaunchFailure;
    }

    std::vector<std::vector<uint8_t>> kargs(fn->params.size());
    for (size_t i = 0; i < fn->params.size(); ++i) {
      kargs[i].resize(param_sizes[i]);
      std::memcpy(kargs[i].data(), args[i], param_sizes[i]);
    }
    vgpu::exec::LaunchConfig cfg;
    cfg.grid = {gridDim.x, gridDim.y, gridDim.z};
    cfg.block = {blockDim.x, blockDim.y, blockDim.z};
    cfg.shared_bytes = static_cast<uint32_t>(sharedMem);
    cfg.cooperative = cooperative;
    cfg.cluster = cluster;
    if (cooperative) {
      // A cooperative launch promises every block is resident, so the grid has
      // to fit. Hardware refuses a grid that does not, and so does this: a
      // kernel whose blocks wait on blocks that were never started does not
      // produce a wrong answer, it hangs, and the error is far more useful.
      const vgpu::DeviceProfile& p = dev.profile();
      const uint64_t resident =
          uint64_t{p.limits.max_blocks_per_sm} * p.limits.multiprocessors;
      const uint64_t want = uint64_t{gridDim.x} * gridDim.y * gridDim.z;
      if (resident && want > resident) {
        if (!quiet())
          std::fprintf(stderr,
                       "[vgpu] %s: grid of %llu blocks exceeds what %s can hold resident "
                       "(%u blocks/SM x %u SMs = %llu). A cooperative launch requires every "
                       "block to be resident, so this cannot run here or on the real part.\n",
                       api, static_cast<unsigned long long>(want), p.id.c_str(),
                       p.limits.max_blocks_per_sm, p.limits.multiprocessors,
                       static_cast<unsigned long long>(resident));
        return cudaErrorCooperativeLaunchTooLarge;
      }
    }
    dev.launch(*fn, cfg, kargs, dev.symbols(mid));
    if (profiling) {
      vgpu::profiling::Event ev;
      ev.kind = vgpu::profiling::EventKind::Kernel;
      ev.start_ns = t0;
      ev.end_ns = vgpu::profiling::now_ns();
      ev.device = static_cast<uint32_t>(s.current_device);
      ev.correlation = vgpu::profiling::next_correlation();
      ev.stream = reinterpret_cast<uint64_t>(stream);
      ev.name = ki.entry_name;
      ev.grid[0] = gridDim.x; ev.grid[1] = gridDim.y; ev.grid[2] = gridDim.z;
      ev.block[0] = blockDim.x; ev.block[1] = blockDim.y; ev.block[2] = blockDim.z;
      ev.shared_bytes = static_cast<uint32_t>(sharedMem);
      vgpu::profiling::record(std::move(ev));
    }
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args,
                                         size_t sharedMem, cudaStream_t stream) {
  return launch_kernel_impl("cudaLaunchKernel", func, gridDim, blockDim, args, sharedMem, stream,
                            /*cooperative=*/false);
}

// cg::this_grid().sync() is only defined for a kernel launched this way, and
// nothing in the PTX distinguishes the two -- the grid barrier is an atomic and
// a spin on ordinary global memory. So the promise has to come from the launch.
VGPU_EXPORT cudaError_t cudaLaunchCooperativeKernel(const void* func, dim3 gridDim, dim3 blockDim,
                                                    void** args, size_t sharedMem,
                                                    cudaStream_t stream) {
  return launch_kernel_impl("cudaLaunchCooperativeKernel", func, gridDim, blockDim, args, sharedMem,
                            stream, /*cooperative=*/true);
}

// The multi-device form needs peer grids on separate devices waiting on each
// other. Refused rather than run as if it were single-device, which would
// deadlock or silently compute the wrong thing.
//
// Deprecated in CUDA 12 and removed in 13, so it is only defined when the
// toolkit still declares it -- and its first parameter is cudaLaunchParams*,
// not void*. A mismatch here is a hard error rather than a subtle one, because
// the vendor header is included: C linkage makes two declarations of the same
// name with different types a conflict, which is exactly the check that caught
// this.
#if CUDART_VERSION < 13000
VGPU_EXPORT cudaError_t cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams*,
                                                               unsigned int, unsigned int) {
  return cudaErrorNotSupported;
}
#endif

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
  // Returns the recorded error but does NOT clear it. CUDA resets the recorded
  // error in exactly one place -- cudaGetLastError -- and clearing it here
  // broke the most common way anyone checks a kernel:
  //
  //     kernel<<<...>>>();
  //     cudaDeviceSynchronize();
  //     if (cudaGetLastError() != cudaSuccess) ...
  //
  // The sync consumed the error, the check found cudaSuccess, and a kernel
  // that had died on an illegal address reported success. Every launch here is
  // synchronous, so by the time this is called the error is already recorded;
  // there is nothing to wait for and nothing to consume.
  return g_last_error;
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
      case cudaDevAttrMaxBlockDimX: *value = static_cast<int>(p.limits.max_block_dim[0]); break;
      case cudaDevAttrMaxBlockDimY: *value = static_cast<int>(p.limits.max_block_dim[1]); break;
      case cudaDevAttrMaxBlockDimZ: *value = static_cast<int>(p.limits.max_block_dim[2]); break;
      // CUB clamps its tile count to the maximum grid extent. Reporting zero
      // for it -- which the silent default below used to do -- asked the device
      // to run no blocks at all, and every scan failed to launch.
      case cudaDevAttrMaxGridDimX: *value = static_cast<int>(p.limits.max_grid_dim[0]); break;
      case cudaDevAttrMaxGridDimY: *value = static_cast<int>(p.limits.max_grid_dim[1]); break;
      case cudaDevAttrMaxGridDimZ: *value = static_cast<int>(p.limits.max_grid_dim[2]); break;
      case cudaDevAttrMaxSharedMemoryPerBlock: *value = static_cast<int>(p.limits.shared_mem_per_block); break;
      case cudaDevAttrMaxSharedMemoryPerBlockOptin: *value = static_cast<int>(p.limits.shared_mem_per_block_optin); break;
      case cudaDevAttrMaxSharedMemoryPerMultiprocessor: *value = static_cast<int>(p.limits.shared_mem_per_sm); break;
      case cudaDevAttrMaxRegistersPerBlock: *value = static_cast<int>(p.limits.registers_per_block); break;
      case cudaDevAttrMaxRegistersPerMultiprocessor: *value = static_cast<int>(p.limits.registers_per_sm); break;
      case cudaDevAttrMaxThreadsPerMultiProcessor: *value = static_cast<int>(p.limits.max_threads_per_sm); break;
      case cudaDevAttrMaxBlocksPerMultiprocessor: *value = static_cast<int>(p.limits.max_blocks_per_sm); break;
      case cudaDevAttrWarpSize: *value = static_cast<int>(p.warp_size); break;
      case cudaDevAttrMultiProcessorCount: *value = static_cast<int>(p.limits.multiprocessors); break;
      case cudaDevAttrComputeCapabilityMajor: *value = p.cc_major; break;
      case cudaDevAttrComputeCapabilityMinor: *value = p.cc_minor; break;
      case cudaDevAttrTotalConstantMemory: *value = 64 * 1024; break;
      case cudaDevAttrClockRate: *value = static_cast<int>(p.telemetry.sm_clock_max_mhz) * 1000; break;
      case cudaDevAttrMemoryClockRate: *value = static_cast<int>(p.telemetry.mem_clock_max_mhz) * 1000; break;
      case cudaDevAttrPciBusId: *value = 0; break;
      case cudaDevAttrPciDeviceId: *value = device; break;
      case cudaDevAttrPciDomainId: *value = 0; break;
      // Capabilities, where zero is the answer rather than the absence of one.
      case cudaDevAttrUnifiedAddressing: *value = 1; break;
      case cudaDevAttrConcurrentKernels: *value = 1; break;
      case cudaDevAttrAsyncEngineCount: *value = 1; break;
      case cudaDevAttrIntegrated: *value = 0; break;
      case cudaDevAttrEccEnabled: *value = 0; break;
      case cudaDevAttrCanMapHostMemory: *value = 0; break;
      case cudaDevAttrManagedMemory: *value = 1; break;
      // Grid-wide sync works under cudaLaunchCooperativeKernel; the
      // multi-device form does not.
      case cudaDevAttrCooperativeLaunch: *value = 1; break;
      case cudaDevAttrComputeMode: *value = 0; break;         // cudaComputeModeDefault
      default:
        // A silent zero here is how a scan came to launch no blocks. An
        // attribute this does not model is reported, so the caller either
        // handles it or fails where the cause is visible -- rather than being
        // told the device has none of whatever it asked about.
        if (!quiet())
          std::fprintf(stderr,
                       "[vgpu] cudaDeviceGetAttribute: attribute %d is not modelled by this "
                       "profile; add it to runtime_api.cpp rather than assuming zero\n",
                       static_cast<int>(attr));
        return cudaErrorInvalidValue;
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
    // Returning the error silently made this very hard to place: CUB asks about
    // its own kernels before it launches any, so the failure surfaced as
    // "invalid device function" from a sort, with nothing said about which
    // kernel could not be described.
    auto it = s.kernels.find(func);
    if (it == s.kernels.end()) {
      if (!quiet())
        std::fprintf(stderr, "[vgpu] cudaFuncGetAttributes: unregistered kernel stub %p\n", func);
      return cudaErrorInvalidDeviceFunction;
    }
    KernelInfo& ki = it->second;
    if (!ki.mod || ki.mod->ptx.empty()) {
      if (!quiet())
        std::fprintf(stderr,
                     "[vgpu] cudaFuncGetAttributes: kernel '%s' has no PTX in its fatbin\n",
                     ki.entry_name.c_str());
      return cudaErrorInvalidDeviceFunction;
    }
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
    // ptxVersion is the *virtual architecture* the function was compiled for,
    // not the PTX ISA version -- CUB multiplies it by ten and dispatches on the
    // result, so reporting 83 for "PTX ISA 8.3" produced 830, an architecture
    // no kernel was ever built for, and every CUB algorithm refused to run with
    // cudaErrorInvalidDeviceFunction. It comes from the module's own .target.
    const int arch = dev.module_arch(mid);
    a->ptxVersion = arch ? arch : p.cc_major * 10 + p.cc_minor;
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
    if (it == s.kernels.end()) {
      if (!quiet())
        std::fprintf(stderr,
                     "[vgpu] cudaOccupancyMaxActiveBlocksPerMultiprocessor: unregistered kernel "
                     "stub %p\n",
                     func);
      return cudaErrorInvalidDeviceFunction;
    }
    KernelInfo& ki = it->second;
    if (!ki.mod || ki.mod->ptx.empty()) {
      if (!quiet())
        std::fprintf(stderr,
                     "[vgpu] cudaOccupancyMaxActiveBlocksPerMultiprocessor: kernel '%s' has no "
                     "PTX in its fatbin\n",
                     ki.entry_name.c_str());
      return cudaErrorInvalidDeviceFunction;
    }
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
    // Injected failure, if this occurrence was selected. Returns the documented
    // out-of-memory the API is allowed to return at any time, which is the
    // error most callers claim to handle and few ever execute.
    if (vgpu::faults::should_fail(vgpu::faults::Op::Alloc)) {
      if (!quiet())
        std::fprintf(stderr, "[vgpu] cudaMalloc: failing this allocation, as VGPU_FAIL_ALLOC "
                             "asked. This is injected, not a real exhaustion\n");
      return cudaErrorMemoryAllocation;
    }
    // A zero-byte allocation succeeds on hardware and yields a distinct pointer
    // that can be freed. ggml asks for one and treats a failure as fatal, so
    // rejecting it stopped whole operations that were doing nothing wrong.
    // Back it with a single byte: that gives an address no other allocation
    // shares, which is what makes the pointer usable as an identity and
    // free-able.
    *ptr = reinterpret_cast<void*>(current(s).memory().alloc(size ? size : 1));
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaFree(void* ptr) {
  return guard("cudaFree", [&](State& s) {
    if (!ptr) return cudaSuccess;  // cudaFree(NULL) is a documented no-op
    // Managed memory frees the same way as device memory from the caller's
    // side, but it is host memory underneath, so it has to leave the device's
    // map before it is released -- otherwise a later kernel could address a
    // pointer this process has given back to the allocator.
    auto mit = s.managed_allocs.find(ptr);
    if (mit != s.managed_allocs.end()) {
      current(s).memory().unmap_host(reinterpret_cast<uint64_t>(ptr));
      s.managed_allocs.erase(mit);
      std::free(ptr);
      return cudaSuccess;
    }
    owner_memory(s, ptr).free(reinterpret_cast<uint64_t>(ptr));
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
  return guard("cudaMemcpy", [&](State& s) {
    if (vgpu::faults::should_fail(vgpu::faults::Op::Memcpy)) {
      if (!quiet())
        std::fprintf(stderr, "[vgpu] cudaMemcpy: failing this copy, as VGPU_FAIL_MEMCPY asked. "
                             "This is injected, not a real failure\n");
      return cudaErrorInvalidValue;
    }
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
    const auto _t1 = std::chrono::steady_clock::now();
    current(s).note_transfer(count, std::chrono::duration<double>(_t1 - _t0).count());
    if (vgpu::profiling::enabled()) {
      vgpu::profiling::Event ev;
      ev.kind = vgpu::profiling::EventKind::Memcpy;
      // The transfer's own clock, so a copy and the kernel beside it line up on
      // one timeline rather than two.
      ev.end_ns = vgpu::profiling::now_ns();
      ev.start_ns = ev.end_ns - static_cast<uint64_t>(
                                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        _t1 - _t0).count());
      ev.device = static_cast<uint32_t>(s.current_device);
      ev.correlation = vgpu::profiling::next_correlation();
      ev.bytes = count;
      ev.copy_kind = static_cast<uint32_t>(kind);
      vgpu::profiling::record(std::move(ev));
    }
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                                        cudaMemcpyKind kind, cudaStream_t stream) {
  if (vgpu_record_memcpy_if_capturing(dst, src, count, kind, stream)) return cudaSuccess;
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
VGPU_EXPORT cudaError_t cudaMemsetAsync(void* dst, int value, size_t count, cudaStream_t stream) {
  if (vgpu_record_memset_if_capturing(dst, value, count, stream)) return cudaSuccess;
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
// Graphs execute inline, so an instantiated graph holds no captured topology
// to compare against. Report that the update did not apply rather than
// claiming success: a caller told the update succeeded will skip
// re-instantiating and then replay work that was never updated.
VGPU_EXPORT cudaError_t cudaGraphExecUpdate(cudaGraphExec_t, cudaGraph_t,
                                            cudaGraphExecUpdateResultInfo* info) {
  if (info) {
    std::memset(info, 0, sizeof *info);
    info->result = cudaGraphExecUpdateErrorTopologyChanged;
  }
  return cudaErrorGraphExecUpdateFailure;
}

/* ===================================================================== */
/* Texture and surface objects                                           */
/*                                                                       */
/* A texture object is a handle over memory plus a description of how to */
/* read it. The kernel receives only the handle, so everything else has  */
/* to be recorded here for the launch to consult -- see                  */
/* include/vgpu/exec/texture.hpp.                                        */
/*                                                                       */
/* No texture cache is modelled: a fetch reads the same bytes an         */
/* ordinary load would, because VirtualGPU has no memory hierarchy to    */
/* model it against. Addressing and format conversion are implemented,   */
/* because those change results rather than timing.                      */
/* ===================================================================== */

namespace {

// A cudaArray here is device memory with its shape recorded alongside. Hardware
// stores arrays in an opaque, swizzled layout that only the texture units can
// address; nothing here depends on that layout, so a dense row-major buffer
// serves, and cudaMemcpy2DToArray is an ordinary strided copy.
struct ArrayRec {
  uint64_t base = 0;
  uint64_t bytes = 0;
  uint32_t width = 0, height = 0, depth = 0;
  cudaChannelFormatDesc fmt{};
  uint32_t texel_bytes = 0;
  int device = 0;
};
std::unordered_map<uint64_t, ArrayRec> g_arrays;
uint64_t g_next_array = 1;

uint32_t texel_bytes_of(const cudaChannelFormatDesc& f) {
  return static_cast<uint32_t>((f.x + f.y + f.z + f.w + 7) / 8);
}
uint32_t channels_of(const cudaChannelFormatDesc& f) {
  return (f.x ? 1u : 0u) + (f.y ? 1u : 0u) + (f.z ? 1u : 0u) + (f.w ? 1u : 0u);
}

bool channel_kind_of(const cudaChannelFormatDesc& f, vgpu::exec::ChannelKind* out) {
  switch (f.f) {
    case cudaChannelFormatKindSigned: *out = vgpu::exec::ChannelKind::Signed; return true;
    case cudaChannelFormatKindUnsigned: *out = vgpu::exec::ChannelKind::Unsigned; return true;
    case cudaChannelFormatKindFloat: *out = vgpu::exec::ChannelKind::Float; return true;
    default: return false;  // NV12, block-compressed and the rest
  }
}

bool address_mode_of(cudaTextureAddressMode m, vgpu::exec::TexAddress* out) {
  switch (m) {
    case cudaAddressModeWrap: *out = vgpu::exec::TexAddress::Wrap; return true;
    case cudaAddressModeClamp: *out = vgpu::exec::TexAddress::Clamp; return true;
    case cudaAddressModeMirror: *out = vgpu::exec::TexAddress::Mirror; return true;
    case cudaAddressModeBorder: *out = vgpu::exec::TexAddress::Border; return true;
  }
  return false;
}

// Handles are ordinary integers on hardware too, and a kernel can only tell
// them apart by what the table says. Starting well above zero means a
// forgotten initialisation looks like the invalid handle it is.
uint64_t g_next_texobj = 0x1000;

// Fills in the parts of the descriptor that come from the resource, whichever
// kind it is. Returns an error code on the forms not implemented.
cudaError_t fill_from_resource(const cudaResourceDesc* res, vgpu::exec::TextureDesc* d) {
  switch (res->resType) {
    case cudaResourceTypeLinear: {
      d->base = reinterpret_cast<uint64_t>(res->res.linear.devPtr);
      d->texel_bytes = texel_bytes_of(res->res.linear.desc);
      if (d->texel_bytes == 0) return cudaErrorInvalidValue;
      d->width = static_cast<uint32_t>(res->res.linear.sizeInBytes / d->texel_bytes);
      d->height = 0;
      d->depth = 0;
      d->pitch_bytes = 0;
      d->channels = channels_of(res->res.linear.desc);
      d->channel_bits[0] = static_cast<uint32_t>(res->res.linear.desc.x);
      d->channel_bits[1] = static_cast<uint32_t>(res->res.linear.desc.y);
      d->channel_bits[2] = static_cast<uint32_t>(res->res.linear.desc.z);
      d->channel_bits[3] = static_cast<uint32_t>(res->res.linear.desc.w);
      if (!channel_kind_of(res->res.linear.desc, &d->kind)) return cudaErrorNotSupported;
      return cudaSuccess;
    }
    case cudaResourceTypePitch2D: {
      d->base = reinterpret_cast<uint64_t>(res->res.pitch2D.devPtr);
      d->texel_bytes = texel_bytes_of(res->res.pitch2D.desc);
      if (d->texel_bytes == 0) return cudaErrorInvalidValue;
      d->width = static_cast<uint32_t>(res->res.pitch2D.width);
      d->height = static_cast<uint32_t>(res->res.pitch2D.height);
      d->depth = 0;
      d->pitch_bytes = static_cast<uint32_t>(res->res.pitch2D.pitchInBytes);
      d->channels = channels_of(res->res.pitch2D.desc);
      d->channel_bits[0] = static_cast<uint32_t>(res->res.pitch2D.desc.x);
      d->channel_bits[1] = static_cast<uint32_t>(res->res.pitch2D.desc.y);
      d->channel_bits[2] = static_cast<uint32_t>(res->res.pitch2D.desc.z);
      d->channel_bits[3] = static_cast<uint32_t>(res->res.pitch2D.desc.w);
      if (!channel_kind_of(res->res.pitch2D.desc, &d->kind)) return cudaErrorNotSupported;
      return cudaSuccess;
    }
    case cudaResourceTypeArray: {
      auto it = g_arrays.find(reinterpret_cast<uint64_t>(res->res.array.array));
      if (it == g_arrays.end()) return cudaErrorInvalidValue;
      const ArrayRec& a = it->second;
      d->base = a.base;
      d->width = a.width;
      d->height = a.height;
      d->depth = a.depth;
      d->pitch_bytes = a.width * a.texel_bytes;
      d->texel_bytes = a.texel_bytes;
      d->channels = channels_of(a.fmt);
      d->channel_bits[0] = static_cast<uint32_t>(a.fmt.x);
      d->channel_bits[1] = static_cast<uint32_t>(a.fmt.y);
      d->channel_bits[2] = static_cast<uint32_t>(a.fmt.z);
      d->channel_bits[3] = static_cast<uint32_t>(a.fmt.w);
      d->from_array = true;
      if (!channel_kind_of(a.fmt, &d->kind)) return cudaErrorNotSupported;
      return cudaSuccess;
    }
    default:
      // Mipmapped arrays need a level-of-detail selection this does not have.
      return cudaErrorNotSupported;
  }
}

}  // namespace

// The templated cudaCreateChannelDesc<T>() in the toolkit header is an inline
// that calls this, so it has to exist even though it computes nothing that
// needs a device.
VGPU_EXPORT cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w,
                                                        cudaChannelFormatKind f) {
  cudaChannelFormatDesc d;
  d.x = x;
  d.y = y;
  d.z = z;
  d.w = w;
  d.f = f;
  return d;
}

VGPU_EXPORT cudaError_t cudaMallocArray(cudaArray_t* array, const cudaChannelFormatDesc* desc,
                                        size_t width, size_t height, unsigned int flags) {
  return guard("cudaMallocArray", [&](State& s) -> cudaError_t {
    if (!array || !desc) return cudaErrorInvalidValue;
    (void)flags;  // cudaArraySurfaceLoadStore changes nothing about the storage here
    ArrayRec rec;
    rec.fmt = *desc;
    rec.texel_bytes = texel_bytes_of(*desc);
    if (rec.texel_bytes == 0 || width == 0) return cudaErrorInvalidValue;
    rec.width = static_cast<uint32_t>(width);
    rec.height = static_cast<uint32_t>(height);
    rec.depth = 0;
    rec.bytes = uint64_t{rec.width} * (rec.height ? rec.height : 1) * rec.texel_bytes;
    rec.device = s.current_device;
    rec.base = current(s).memory().alloc(rec.bytes);
    const uint64_t handle = g_next_array++;
    g_arrays[handle] = rec;
    *array = reinterpret_cast<cudaArray_t>(handle);
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaFreeArray(cudaArray_t array) {
  return guard("cudaFreeArray", [&](State& s) -> cudaError_t {
    auto it = g_arrays.find(reinterpret_cast<uint64_t>(array));
    if (it == g_arrays.end()) return cudaErrorInvalidValue;
    s.rt->device(it->second.device).memory().free(it->second.base);
    g_arrays.erase(it);
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array) {
  return guard("cudaGetChannelDesc", [&](State&) -> cudaError_t {
    auto it = g_arrays.find(reinterpret_cast<uint64_t>(array));
    if (!desc || it == g_arrays.end()) return cudaErrorInvalidValue;
    *desc = it->second.fmt;
    return cudaSuccess;
  });
}

// Copies into and out of an array. The array is dense row-major here, so these
// are strided copies; on hardware they also convert between the linear source
// and the array's swizzled layout, which is exactly the detail nothing outside
// the texture unit is allowed to depend on.
VGPU_EXPORT cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset,
                                            const void* src, size_t spitch, size_t width,
                                            size_t height, cudaMemcpyKind kind) {
  return guard("cudaMemcpy2DToArray", [&](State& s) -> cudaError_t {
    auto it = g_arrays.find(reinterpret_cast<uint64_t>(dst));
    if (it == g_arrays.end() || !src) return cudaErrorInvalidValue;
    const ArrayRec& a = it->second;
    const uint64_t dpitch = uint64_t{a.width} * a.texel_bytes;
    if (wOffset + width > dpitch || hOffset + height > (a.height ? a.height : 1))
      return cudaErrorInvalidValue;
    vgpu::MemoryManager& mem = s.rt->device(a.device).memory();
    for (size_t row = 0; row < height; ++row) {
      const uint64_t d = a.base + (hOffset + row) * dpitch + wOffset;
      const uint8_t* srow = static_cast<const uint8_t*>(src) + row * spitch;
      if (kind == cudaMemcpyDeviceToDevice) {
        // Device to device: stage the row rather than assuming the two live in
        // one address space, which they need not.
        std::vector<uint8_t> stage(width);
        mem.read(reinterpret_cast<uint64_t>(srow), stage.data(), width);
        mem.write(d, stage.data(), width);
      } else {
        mem.write(d, srow, width);
      }
    }
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src,
                                              size_t wOffset, size_t hOffset, size_t width,
                                              size_t height, cudaMemcpyKind kind) {
  return guard("cudaMemcpy2DFromArray", [&](State& s) -> cudaError_t {
    auto it = g_arrays.find(reinterpret_cast<uint64_t>(src));
    if (it == g_arrays.end() || !dst) return cudaErrorInvalidValue;
    const ArrayRec& a = it->second;
    const uint64_t spitch = uint64_t{a.width} * a.texel_bytes;
    if (wOffset + width > spitch || hOffset + height > (a.height ? a.height : 1))
      return cudaErrorInvalidValue;
    vgpu::MemoryManager& mem = s.rt->device(a.device).memory();
    for (size_t row = 0; row < height; ++row) {
      const uint64_t srow = a.base + (hOffset + row) * spitch + wOffset;
      uint8_t* drow = static_cast<uint8_t*>(dst) + row * dpitch;
      if (kind == cudaMemcpyDeviceToDevice) {
        std::vector<uint8_t> stage(width);
        mem.read(srow, stage.data(), width);
        mem.write(reinterpret_cast<uint64_t>(drow), stage.data(), width);
      } else {
        mem.read(srow, drow, width);
      }
    }
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaCreateTextureObject(cudaTextureObject_t* out,
                                                const cudaResourceDesc* res,
                                                const cudaTextureDesc* tex,
                                                const cudaResourceViewDesc* view) {
  return guard("cudaCreateTextureObject", [&](State& s) -> cudaError_t {
    if (!out || !res) return cudaErrorInvalidValue;
    if (view) return cudaErrorNotSupported;  // resource views reinterpret the format
    vgpu::exec::TextureDesc d;
    d.object = vgpu::exec::TexKind::Texture;
    if (cudaError_t e = fill_from_resource(res, &d); e != cudaSuccess) return e;
    if (tex) {
      for (int i = 0; i < 3; ++i)
        if (!address_mode_of(tex->addressMode[i], &d.address[i])) return cudaErrorInvalidValue;
      // Linear filtering is refused rather than approximated -- see the note at
      // the fetch. sRGB and anisotropy change the result too.
      if (tex->filterMode == cudaFilterModeLinear) d.filter = vgpu::exec::TexFilter::Linear;
      if (tex->sRGB) return cudaErrorNotSupported;
      if (tex->maxAnisotropy > 1) return cudaErrorNotSupported;
      d.normalized_coords = tex->normalizedCoords != 0;
      d.read_as_normalized_float = tex->readMode == cudaReadModeNormalizedFloat;
    }
    const uint64_t handle = g_next_texobj++;
    current(s).textures()[handle] = d;
    *out = static_cast<cudaTextureObject_t>(handle);
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaDestroyTextureObject(cudaTextureObject_t obj) {
  return guard("cudaDestroyTextureObject", [&](State& s) -> cudaError_t {
    return current(s).textures().erase(static_cast<uint64_t>(obj)) ? cudaSuccess
                                                                   : cudaErrorInvalidValue;
  });
}

VGPU_EXPORT cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* out,
                                                const cudaResourceDesc* res) {
  return guard("cudaCreateSurfaceObject", [&](State& s) -> cudaError_t {
    if (!out || !res) return cudaErrorInvalidValue;
    vgpu::exec::TextureDesc d;
    d.object = vgpu::exec::TexKind::Surface;
    if (cudaError_t e = fill_from_resource(res, &d); e != cudaSuccess) return e;
    const uint64_t handle = g_next_texobj++;
    current(s).textures()[handle] = d;
    *out = static_cast<cudaSurfaceObject_t>(handle);
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t obj) {
  return guard("cudaDestroySurfaceObject", [&](State& s) -> cudaError_t {
    return current(s).textures().erase(static_cast<uint64_t>(obj)) ? cudaSuccess
                                                                   : cudaErrorInvalidValue;
  });
}

VGPU_EXPORT cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc*, cudaTextureObject_t) {
  return cudaErrorNotSupported;
}

// Managed memory is one allocation the CPU and GPU both address. Device memory
// here lives in a separate virtual window that host code cannot dereference, so
// handing back a device pointer would fault the moment the host touched it.
VGPU_EXPORT cudaError_t cudaMallocManaged(void** ptr, size_t size, unsigned int) {
  return guard("cudaMallocManaged", [&](State& s) -> cudaError_t {
    if (!ptr) return cudaErrorInvalidValue;
    if (vgpu::faults::should_fail(vgpu::faults::Op::Alloc)) return cudaErrorMemoryAllocation;
    const size_t n = size ? size : 1;
    void* p = std::aligned_alloc(4096, (n + 4095) / 4096 * 4096);
    if (!p) return cudaErrorMemoryAllocation;
    // Real host memory, addressable by kernels at its own address. This is the
    // one place being a simulator makes something *easier*: on hardware
    // managed memory needs page migration between two physical memories, and
    // here there is only one.
    current(s).memory().map_host(reinterpret_cast<uint64_t>(p), p, n);
    st().managed_allocs[p] = n;
    *ptr = p;
    return cudaSuccess;
  });
}

// Prefetching and advice describe where pages should live. There is one memory
// here, so both are honest no-ops rather than refusals: a caller that
// prefetches is asking for a performance hint, and not getting one is not a
// behavioural difference it can observe.
//
// CUDA 13 changed both to take a cudaMemLocation where 12 took an int device,
// so each needs the signature of the toolkit in use. The vendor header is
// included, and C linkage makes a mismatch a hard error rather than a subtle
// one -- which is how the last one of these was caught.
#if CUDART_VERSION >= 13000
VGPU_EXPORT cudaError_t cudaMemPrefetchAsync(const void*, size_t, struct cudaMemLocation,
                                             unsigned int, cudaStream_t) {
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaMemAdvise(const void*, size_t, cudaMemoryAdvise,
                                      struct cudaMemLocation) {
  return cudaSuccess;
}
#else
VGPU_EXPORT cudaError_t cudaMemPrefetchAsync(const void*, size_t, int, cudaStream_t) {
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaMemAdvise(const void*, size_t, cudaMemoryAdvise, int) {
  return cudaSuccess;
}
#endif

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
  // The attribute list is the whole point of the Ex form, and this used to
  // drop it. That was invisible until thread-block clusters existed: a kernel
  // launched with cudaLaunchAttributeClusterDimension ran with no cluster at
  // all, and every block read %cluster_ctarank as 0 and %cluster_nctarank as
  // 1. The launch succeeded and the answer was wrong, which is the failure
  // this engine is built to not have.
  //
  // The vendor struct is used rather than a hand-rolled one because its size
  // is not stable: sizeof(cudaLaunchAttribute) is 72 with this toolkit, and a
  // guess at the stride would walk the array wrong on any other version.
  unsigned cluster[3] = {0, 0, 0};
  for (unsigned i = 0; i < cfg->numAttrs; ++i) {
    const cudaLaunchAttribute& a = cfg->attrs[i];
    if (a.id == cudaLaunchAttributeClusterDimension) {
      cluster[0] = a.val.clusterDim.x;
      cluster[1] = a.val.clusterDim.y;
      cluster[2] = a.val.clusterDim.z;
    }
    // Every other attribute is inert here for a reason that is already true
    // elsewhere in this shim: priority and memory-sync domains need a stream
    // scheduler, access policy windows need a cache model, and programmatic
    // events need asynchrony. None of them changes what a kernel computes.
  }
  return launch_kernel_impl("cudaLaunchKernelEx", func, cfg->gridDim, cfg->blockDim, args,
                            cfg->dynamicSmemBytes, cfg->stream, /*cooperative=*/false,
                            {cluster[0], cluster[1], cluster[2]});
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

// Both of these must name every code this shim can return. They did not: the
// tables listed eight of the seventeen, so a program that printed
// cudaGetErrorName() of a real, correctly-returned error was told
// "cudaErrorUnknown" -- confidently wrong, and indistinguishable from the
// simulator having no idea what happened.
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
    case cudaErrorInvalidConfiguration: return "invalid configuration argument";
    case cudaErrorIllegalInstruction: return "an illegal instruction was encountered";
    case cudaErrorAssert: return "device-side assert triggered";
    case cudaErrorLaunchFailure: return "unspecified launch failure";
    case cudaErrorCooperativeLaunchTooLarge:
      return "too many blocks in cooperative launch";
    case cudaErrorStreamCaptureImplicit:
      return "operation would make the legacy stream depend on a capturing stream";
    case cudaErrorStreamCaptureInvalidated:
      return "operation failed due to a previous error during capture";
    case cudaErrorGraphExecUpdateFailure: return "the graph update was not performed";
    case cudaErrorOperatingSystem: return "OS call failed or operation not supported on this OS";
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
    case cudaErrorLaunchTimeout: return "cudaErrorLaunchTimeout";
    case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration";
    case cudaErrorIllegalInstruction: return "cudaErrorIllegalInstruction";
    case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
    case cudaErrorCooperativeLaunchTooLarge: return "cudaErrorCooperativeLaunchTooLarge";
    case cudaErrorStreamCaptureImplicit: return "cudaErrorStreamCaptureImplicit";
    case cudaErrorStreamCaptureInvalidated: return "cudaErrorStreamCaptureInvalidated";
    case cudaErrorGraphExecUpdateFailure: return "cudaErrorGraphExecUpdateFailure";
    case cudaErrorOperatingSystem: return "cudaErrorOperatingSystem";
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

// One operation inside a captured region. A graph is not only kernels: a
// framework captures copies and fills alongside them, and replaying with those
// missing produces confidently wrong results rather than an error.
struct RecordedLaunch {
  enum class Kind { Kernel, Memcpy, Memset, Host } kind = Kind::Kernel;
  // Kernel
  const void* func = nullptr;
  dim3 grid, block;
  size_t shared = 0;
  std::vector<std::vector<uint8_t>> arg_bytes;  // deep copy of parameter values
  std::vector<void*> arg_ptrs;                  // rebuilt to point at arg_bytes
  // Memcpy / Memset. Pointers are recorded, not the data behind them: a graph
  // reads whatever the buffers hold at replay time, which is what makes
  // replaying a captured step with new inputs meaningful.
  void* dst = nullptr;
  const void* src = nullptr;
  size_t bytes = 0;
  cudaMemcpyKind copy_kind = cudaMemcpyDefault;
  int fill_value = 0;
  // Host: a vendor-library call that computes on the CPU. It is replayed by
  // re-running the closure, which re-reads device memory then -- so it sees
  // what the graph's kernels produced, exactly as the real library would.
  std::function<void()> host_op;
};

struct GraphRec {
  std::vector<RecordedLaunch> launches;
  // Set when something happened during capture that this implementation cannot
  // record. Kernel launches, copies, fills and host-computed library calls are
  // all captured; anything else would run immediately and be missing from every
  // replay, so the capture is no longer a faithful record of the work and must
  // not be used.
  bool invalidated = false;
  const char* invalidated_by = nullptr;
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

// Records a copy or a fill during capture. Returns true when it was recorded,
// in which case the caller must not perform it now -- it belongs to the graph.
bool vgpu_record_memcpy_if_capturing(void* dst, const void* src, size_t bytes,
                                     cudaMemcpyKind kind, cudaStream_t stream) {
  GraphRec* g = capture_target(stream);
  if (!g) return false;
  RecordedLaunch r;
  r.kind = RecordedLaunch::Kind::Memcpy;
  r.dst = dst;
  r.src = src;
  r.bytes = bytes;
  r.copy_kind = kind;
  g->launches.push_back(std::move(r));
  return true;
}

bool vgpu_record_memset_if_capturing(void* dst, int value, size_t bytes, cudaStream_t stream) {
  GraphRec* g = capture_target(stream);
  if (!g) return false;
  RecordedLaunch r;
  r.kind = RecordedLaunch::Kind::Memset;
  r.dst = dst;
  r.fill_value = value;
  r.bytes = bytes;
  g->launches.push_back(std::move(r));
  return true;
}

// Records a host-computed library call during capture. Returns true when it was
// recorded, in which case the caller must not do the work now.
bool vgpu_record_host_op_if_capturing(cudaStream_t stream, std::function<void()> op) {
  GraphRec* g = capture_target(stream);
  if (!g) return false;
  RecordedLaunch r;
  r.kind = RecordedLaunch::Kind::Host;
  r.host_op = std::move(op);
  std::lock_guard<std::mutex> lock(g_graph_mu);
  g->launches.push_back(std::move(r));
  return true;
}

// Marks any in-flight capture as unusable. Called by the operations that this
// implementation cannot record, so a capture that would silently omit work
// fails at cudaStreamEndCapture instead of replaying an incomplete graph.
void vgpu_invalidate_capture(const char* what) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  for (auto& entry : g_capturing)
    if (entry.second && !entry.second->invalidated) {
      entry.second->invalidated = true;
      entry.second->invalidated_by = what;
    }
}

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
  // Kernel launches, copies and fills are all recorded, so a captured region
  // replays the work it actually contained. Anything still unrecordable marks
  // the capture invalid and cudaStreamEndCapture reports it, rather than
  // handing back a graph that silently omits work.
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
  if (graph->invalidated) {
    // CUDA reports a capture that saw an unsupported operation this way, and
    // callers fall back to running the work directly. Handing back a graph that
    // silently omits the copies would give wrong answers on every replay.
    if (!quiet())
      std::fprintf(stderr,
                   "[vgpu] stream capture invalidated: %s inside a captured region is not "
                   "recorded (only kernel launches are).\n"
                   "       Run the work directly instead of replaying a graph.\n",
                   graph->invalidated_by ? graph->invalidated_by : "an operation");
    if (pGraph) *pGraph = nullptr;
    return cudaErrorStreamCaptureInvalidated;
  }
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
    cudaError_t rc = cudaSuccess;
    if (rl.kind == RecordedLaunch::Kind::Memcpy) {
      rc = cudaMemcpy(rl.dst, rl.src, rl.bytes, rl.copy_kind);
    } else if (rl.kind == RecordedLaunch::Kind::Memset) {
      rc = cudaMemset(rl.dst, rl.fill_value, rl.bytes);
    } else if (rl.kind == RecordedLaunch::Kind::Host) {
      rl.host_op();
    } else {
      std::vector<void*> ptrs(rl.arg_bytes.size());
      for (size_t i = 0; i < rl.arg_bytes.size(); ++i) ptrs[i] = rl.arg_bytes[i].data();
      rc = cudaLaunchKernel(rl.func, rl.grid, rl.block, ptrs.data(), rl.shared, stream);
    }
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
