// libvgpucuda — VirtualGPU's implementation of a CUDA Driver API subset.
//
// Clean-room: implemented from NVIDIA's public documentation only (see
// nvidia/include/vgpu_cuda.h). This is the C ABI surface external programs link
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
#include <dlfcn.h>
#include <fstream>
#include <sstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <ctime>
#include <set>

#include "error_names.hpp"
#include "fatbin.hpp"
#include "vgpu/driver_version.hpp"
#include "vgpu/error.hpp"
#include "vgpu/profiling.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"

namespace {

// cuDriverGetVersion comes from vgpu::driver_version() -- the session's
// declared CUDA version, shared with the runtime shim and nvidia-smi so the
// three cannot disagree. See include/vgpu/driver_version.hpp. Still no toolkit
// dependency: it reads an environment variable, not a CUDA header.

// Handle tagging: low 3 bits encode the handle type so passing e.g. a module
// where a context belongs is caught instead of misbehaving.
constexpr uintptr_t kTagCtx = 1, kTagModule = 2, kTagFunc = 3, kTagStream = 4, kTagEvent = 5,
                    kTagLibrary = 6, kTagKernel = 7;

struct FuncRec {
  int device = 0;
  uintptr_t module_handle = 0;
  const vgpu::ptx::EntryFn* fn = nullptr;
  const vgpu::exec::SymbolTable* syms = nullptr;
};

// A context-independent code library (CUDA 12+ cuLibrary API): holds PTX
// text, instantiated per device on first use.
struct LibRec {
  std::string ptx;
  std::unordered_map<int, uint64_t> per_device_module;  // device -> runtime module id
};

struct KernelRec {
  uintptr_t library = 0;
  std::string name;
};

struct EventRec {
  bool recorded = false;
  struct timespec when {};
};

struct ShimState {
  std::recursive_mutex mu;
  bool initialized = false;
  std::unique_ptr<vgpu::runtime::Runtime> rt;
  uintptr_t next_id = 8;

  std::unordered_map<uintptr_t, int> contexts;         // ctx handle -> device ordinal
  std::unordered_map<int, uintptr_t> primary_ctx;      // device -> primary ctx handle
  // Retain count per device. A release with nothing retained is an error the
  // caller needs to see: it means some other component's retain is about to
  // be undone out from under it.
  std::unordered_map<int, int> primary_refs;
  std::unordered_map<uintptr_t, std::pair<int, uint64_t>> modules;  // handle -> (dev, module id)
  std::unordered_map<uintptr_t, FuncRec> functions;
  std::unordered_map<uintptr_t, LibRec> libraries;
  std::unordered_map<uintptr_t, KernelRec> kernels;
  std::unordered_map<uintptr_t, EventRec> events;
  std::set<uintptr_t> streams;      // explicitly created streams (all synchronous)
  std::set<void*> host_allocs;      // cuMemHostAlloc results
};

ShimState& state() {
  static ShimState s;
  return s;
}

// The current-context stack. CUDA documents it as belonging to the calling
// host thread, and it used to be one stack shared by the whole process: a new
// thread's cuCtxGetCurrent returned whatever the main thread had pushed, and a
// push in a worker changed which device the main thread's next allocation
// landed on. Contexts themselves stay process-wide (ShimState::contexts); only
// which one is current is per thread, and a new thread starts with none.
std::vector<uintptr_t>& ctx_stack() {
  thread_local std::vector<uintptr_t> stack;
  return stack;
}

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

// VGPU_TRACE=1: log every driver entry point resolution and unknown-attribute
// query — the tool for growing the shim against new applications.
// VGPU_TRACE=2: also log every driver call and its result. Louder, and the only
// way to see what a caller does between the calls it tells you about -- a
// statically linked runtime runs a self-test before it will report a device,
// and this is how you find out what the self-test asks for.
int trace_level() {
  const char* t = std::getenv("VGPU_TRACE");
  if (!t || !t[0]) return 0;
  return t[0] - '0';
}
bool trace() { return trace_level() >= 1; }
bool trace_calls() { return trace_level() >= 2; }

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
    // Both of these reached the switch's fallthrough before, and so were
    // reported as CUDA_ERROR_UNKNOWN -- the least informative code available,
    // for the two conditions this project most wants to be legible.
    case Err::Trap: return CUDA_ERROR_ILLEGAL_INSTRUCTION;
    case Err::DeviceAssert: return CUDA_ERROR_ASSERT;
    case Err::DataRace: return CUDA_ERROR_LAUNCH_FAILED;
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
    CUresult r = body(s);
    if (trace_calls()) std::fprintf(stderr, "[vgpu][call] %s -> %d\n", name, static_cast<int>(r));
    return r;
  } catch (const vgpu::Error& e) {
    report(name, e.what());
    const CUresult r = map_error(e, kernel_context);
    if (trace_calls()) std::fprintf(stderr, "[vgpu][call] %s -> %d (threw)\n", name, static_cast<int>(r));
    return r;
  } catch (const std::exception& e) {
    report(name, std::string("unexpected: ") + e.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

int current_device(ShimState& s) {
  if (ctx_stack().empty())
    throw vgpu::Error::make(vgpu::Err::InvalidValue,
                            "no current context (create one with cuCtxCreate or "
                            "cuDevicePrimaryCtxRetain + cuCtxSetCurrent)");
  // With a stack per thread, another thread can destroy the context this one
  // still has current. That is the caller's error, and it should be reported
  // as one rather than escaping as std::out_of_range and CUDA_ERROR_UNKNOWN.
  auto it = s.contexts.find(ctx_stack().back());
  if (it == s.contexts.end())
    throw vgpu::Error::make(vgpu::Err::InvalidValue,
                            "the current context has been destroyed (by cuCtxDestroy, possibly on "
                            "another thread)");
  return it->second;
}

vgpu::runtime::Device& current(ShimState& s) { return s.rt->device(current_device(s)); }

// Device VA windows are disjoint, so a device pointer names its own device.
// Resolving against the current context instead would make a copy between two
// devices read the wrong memory at the same numeric address.
vgpu::MemoryManager& owner_memory(ShimState& s, CUdeviceptr p) {
  for (int d = 0; d < s.rt->device_count(); ++d)
    if (s.rt->device(d).memory().owns(p)) return s.rt->device(d).memory();
  return current(s).memory();
}

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

// Picks the best PTX image from a fatbin: the highest-arch one (most specific
// for the emulated device generation).
std::string best_ptx(const void* image) {
  auto ptxs = vgpu::cuda::extract_ptx(image);
  if (ptxs.empty())
    throw vgpu::Error::make(vgpu::Err::Unsupported,
                            "fatbin contains no PTX image (SASS-only fatbin?); rebuild with an "
                            "-arch=sm_XX that embeds PTX, or add -gencode arch=compute_XX,"
                            "code=compute_XX");
  size_t best = 0;
  for (size_t i = 1; i < ptxs.size(); ++i)
    if (ptxs[i].arch > ptxs[best].arch) best = i;
  return std::move(ptxs[best].text);
}

// Attribute values beyond the profile-backed set.
//
// The numbers are CUdevice_attribute, and they are the whole difficulty: an
// answer filed under the wrong one is worse than no answer, because it is
// returned confidently. This table previously had ten entries numbered wrong,
// including MAX_BLOCKS_PER_MULTIPROCESSOR -- added after a CUB scan launched no
// blocks -- filed under 134, which is HOST_NUMA_ID. The scan bug was still
// there, and 134 was answering a NUMA query with a block count. Every id below
// is checked against the toolkit's cuda.h.
//
// Texture and surface limits are the documented per-compute-capability values
// from the CUDA C Programming Guide's technical-specification table. Nothing
// here models performance; the clock and bandwidth entries are placeholders and
// say so.
int extra_attribute(const vgpu::DeviceProfile& p, int attrib) {
  switch (attrib) {
    case 11: return 2147483647;                      // MAX_PITCH
    case 14: return 512;                             // TEXTURE_ALIGNMENT
    case 15: return 1;                               // GPU_OVERLAP
    case 17: return 0;                               // KERNEL_EXEC_TIMEOUT
    case 18: return 0;                               // INTEGRATED
    case 19: return 1;                               // CAN_MAP_HOST_MEMORY
    case 20: return 0;                               // COMPUTE_MODE (default)

    // ---- texture limits (documented, per compute capability) ----
    case 21: return 131072;                          // MAXIMUM_TEXTURE1D_WIDTH
    case 22: return 131072;                          // MAXIMUM_TEXTURE2D_WIDTH
    case 23: return 65536;                           // MAXIMUM_TEXTURE2D_HEIGHT
    case 24: return 16384;                           // MAXIMUM_TEXTURE3D_WIDTH
    case 25: return 16384;                           // MAXIMUM_TEXTURE3D_HEIGHT
    case 26: return 16384;                           // MAXIMUM_TEXTURE3D_DEPTH
    case 27: return 32768;                           // MAXIMUM_TEXTURE2D_LAYERED_WIDTH
    case 28: return 32768;                           // MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
    case 29: return 2048;                            // MAXIMUM_TEXTURE2D_LAYERED_LAYERS

    case 30: return 512;                             // SURFACE_ALIGNMENT
    case 31: return 1;                               // CONCURRENT_KERNELS
    case 32: return 0;                               // ECC_ENABLED
    case 33: return 1;                               // PCI_BUS_ID
    case 34: return 0;                               // PCI_DEVICE_ID
    case 35: return 0;                               // TCC_DRIVER (Linux is always 0)
    case 36: return 1000000;                         // MEMORY_CLOCK_RATE (placeholder)
    case 37: return 256;                             // GLOBAL_MEMORY_BUS_WIDTH (placeholder)
    case 38: return 8 * 1024 * 1024;                 // L2_CACHE_SIZE (placeholder)
    case 39:                                         // MAX_THREADS_PER_MULTIPROCESSOR
      return static_cast<int>(p.limits.max_threads_per_sm);
    case 40: return 2;                               // ASYNC_ENGINE_COUNT
    case 41: return 1;                               // UNIFIED_ADDRESSING (64-bit Linux is UVA)
    case 42: return 32768;                           // MAXIMUM_TEXTURE1D_LAYERED_WIDTH
    case 43: return 2048;                            // MAXIMUM_TEXTURE1D_LAYERED_LAYERS
    case 45: return 32768;                           // MAXIMUM_TEXTURE2D_GATHER_WIDTH
    case 46: return 32768;                           // MAXIMUM_TEXTURE2D_GATHER_HEIGHT
    case 47: return 16384;                           // MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE
    case 48: return 16384;                           // MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE
    case 49: return 16384;                           // MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE
    case 50: return 0;                               // PCI_DOMAIN_ID
    case 51: return 32;                              // TEXTURE_PITCH_ALIGNMENT
    case 52: return 32768;                           // MAXIMUM_TEXTURECUBEMAP_WIDTH
    case 53: return 32768;                           // MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH
    case 54: return 2046;                            // MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS

    // ---- surface limits ----
    case 55: return 32768;                           // MAXIMUM_SURFACE1D_WIDTH
    case 56: return 131072;                          // MAXIMUM_SURFACE2D_WIDTH
    case 57: return 65536;                           // MAXIMUM_SURFACE2D_HEIGHT
    case 58: return 16384;                           // MAXIMUM_SURFACE3D_WIDTH
    case 59: return 16384;                           // MAXIMUM_SURFACE3D_HEIGHT
    case 60: return 16384;                           // MAXIMUM_SURFACE3D_DEPTH
    case 61: return 32768;                           // MAXIMUM_SURFACE1D_LAYERED_WIDTH
    case 62: return 2048;                            // MAXIMUM_SURFACE1D_LAYERED_LAYERS
    case 63: return 32768;                           // MAXIMUM_SURFACE2D_LAYERED_WIDTH
    case 64: return 32768;                           // MAXIMUM_SURFACE2D_LAYERED_HEIGHT
    case 65: return 2048;                            // MAXIMUM_SURFACE2D_LAYERED_LAYERS
    case 66: return 32768;                           // MAXIMUM_SURFACECUBEMAP_WIDTH
    case 67: return 32768;                           // MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH
    case 68: return 2046;                            // MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS

    case 70: return 131072;                          // MAXIMUM_TEXTURE2D_LINEAR_WIDTH
    case 71: return 65000;                           // MAXIMUM_TEXTURE2D_LINEAR_HEIGHT
    case 72: return 2097120;                         // MAXIMUM_TEXTURE2D_LINEAR_PITCH
    case 73: return 32768;                           // MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH
    case 74: return 32768;                           // MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT
    case 77: return 32768;                           // MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH

    case 78: return 1;                               // STREAM_PRIORITIES_SUPPORTED
    case 79: return 1;                               // GLOBAL_L1_CACHE_SUPPORTED
    case 80: return 1;                               // LOCAL_L1_CACHE_SUPPORTED
    case 81: return static_cast<int>(p.limits.shared_mem_per_sm);
                                                     // MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
    case 82: return static_cast<int>(p.limits.registers_per_sm);
                                                     // MAX_REGISTERS_PER_MULTIPROCESSOR
    case 84: return 0;                               // MULTI_GPU_BOARD
    case 85: return 0;                               // MULTI_GPU_BOARD_GROUP_ID
    case 90: return 1;                               // COMPUTE_PREEMPTION_SUPPORTED
    // MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: the ceiling a kernel can raise its
    // dynamic shared memory to, above the 48 KiB default. Triton reads it to
    // decide how large a tile it may stage, so answering zero caps every kernel
    // at the smallest tile it knows.
    case 97: return static_cast<int>(p.limits.shared_mem_per_block_optin);
    case 98: return 0;                               // CAN_FLUSH_REMOTE_WRITES
    case 99: return 1;                               // HOST_REGISTER_SUPPORTED
    // A real quantity, and answering zero for it is what had a CUB scan launch
    // no blocks: it is a divisor in occupancy arithmetic. 106, not 134.
    case 106: return static_cast<int>(p.limits.max_blocks_per_sm);
    case 107: return 0;                              // GENERIC_COMPRESSION_SUPPORTED
    case 110: return 0;                              // GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED
    case 111: return 0;                              // RESERVED_SHARED_MEMORY_PER_BLOCK

    // ---- capabilities this deliberately does not implement ----
    // Zero is the true answer for each, and saying so explicitly keeps them out
    // of the "not modelled" report below: a caller that asks whether managed
    // memory works needs a truthful no, not a warning.
    case 83: return 0;                               // MANAGED_MEMORY
    case 86: return 0;                               // HOST_NATIVE_ATOMIC_SUPPORTED
    case 88: return 0;                               // PAGEABLE_MEMORY_ACCESS
    case 89: return 0;                               // CONCURRENT_MANAGED_ACCESS
    case 91: return 0;                               // CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM
    // COOPERATIVE_LAUNCH: cudaLaunchCooperativeKernel works, because the
    // scheduler can hold every block resident and interleave them. The
    // multi-device form (96) needs grids on separate devices waiting on each
    // other, which it cannot.
    case 95: return 1;
    case 96: return 0;                               // COOPERATIVE_MULTI_DEVICE_LAUNCH
    case 100: return 0;                              // PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
    case 101: return 0;                              // DIRECT_MANAGED_MEM_ACCESS_FROM_HOST
    case 102: return 0;                              // VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
    case 103: return 0;                              // HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED
    case 104: return 0;                              // HANDLE_TYPE_WIN32_HANDLE_SUPPORTED
    case 105: return 0;                              // HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED
    case 108: return 0;                              // MAX_PERSISTING_L2_CACHE_SIZE
    case 109: return 0;                              // MAX_ACCESS_POLICY_WINDOW_SIZE
    case 112: return 0;                              // SPARSE_CUDA_ARRAY_SUPPORTED
    case 113: return 0;                              // READ_ONLY_HOST_REGISTER_SUPPORTED
    case 114: return 0;                              // TIMELINE_SEMAPHORE_INTEROP_SUPPORTED
    case 115: return 0;                              // MEMORY_POOLS_SUPPORTED (no cuMemPool*)
    case 116: return 0;                              // GPU_DIRECT_RDMA_SUPPORTED
    case 117: return 0;                              // GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS
    case 118: return 0;                              // GPU_DIRECT_RDMA_WRITES_ORDERING
    case 119: return 0;                              // MEMPOOL_SUPPORTED_HANDLE_TYPES
    case 120: return 0;                              // CLUSTER_LAUNCH (no thread-block clusters)
    case 121: return 0;                              // DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED
    case 124: return 0;                              // DMA_BUF_SUPPORTED
    case 125: return 0;                              // IPC_EVENT_SUPPORTED (see cuIpc* above)
    case 128: return 0;                              // TENSOR_MAP_ACCESS_SUPPORTED
    case 129: return 0;                              // UNIFIED_FUNCTION_POINTERS
    case 130: return 0;                              // NUMA_CONFIG (not NUMA-attached)
    case 131: return 0;                              // NUMA_ID
    case 133: return 0;                              // MPS_ENABLED
    case 134: return -1;                             // HOST_NUMA_ID (-1: no NUMA affinity)
    case 135: return 0;                              // D3D12_CIG_SUPPORTED (Windows only)
    case 136: return 0;                              // MEM_DECOMPRESS_ALGORITHM_MASK
    case 137: return 0;                              // MEM_DECOMPRESS_MAXIMUM_LENGTH
    case 138: return 0;                              // VULKAN_CIG_SUPPORTED
    // GPU_PCI_DEVICE_ID: the 16-bit PCI device and vendor ids packed into one
    // word. The profile carries the pair that `vgpu smi --lspci` renders, so
    // this is the same identity the rest of the stack presents.
    case 139:
      return static_cast<int>((p.telemetry.pci_device_id << 16) | p.telemetry.pci_vendor_id);
    case 140: return 0;                              // GPU_PCI_SUBSYSTEM_ID
    case 141: return 0;                              // HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
    case 142: return 0;                              // HOST_NUMA_MEMORY_POOLS_SUPPORTED
    case 143: return 0;                              // HOST_NUMA_MULTINODE_IPC_SUPPORTED
    default:
      // Answering zero for a quantity nobody modelled is how a scan came to
      // launch no blocks, on the runtime side of this same question. The
      // driver API is reached by more varied software, and an error here is
      // more likely to be fatal than useful -- so this still answers zero, but
      // says so where it can be seen rather than only under VGPU_TRACE.
      if (!quiet())
        std::fprintf(stderr,
                     "[vgpu] cuDeviceGetAttribute: attribute %d is not modelled; answering 0. If "
                     "that is wrong for your program, add it to driver_api.cpp\n",
                     attrib);
      return 0;
  }
}

// Instantiates a context-independent library on `dev` (parsing its PTX into a
// runtime module on first use) and returns the runtime module id.
uint64_t library_module_on(ShimState& s, uintptr_t lib_handle, int dev) {
  auto it = s.libraries.find(lib_handle);
  if (it == s.libraries.end())
    throw vgpu::Error::make(vgpu::Err::InvalidValue, "invalid library handle");
  LibRec& lib = it->second;
  auto mit = lib.per_device_module.find(dev);
  if (mit != lib.per_device_module.end()) return mit->second;
  uint64_t mid = s.rt->device(dev).load_module(lib.ptx);
  lib.per_device_module[dev] = mid;
  return mid;
}

// Resolves a CUkernel handle to a launchable function handle on the current
// device, creating the per-device module instantiation as needed.
uintptr_t kernel_to_function(ShimState& s, uintptr_t kernel_handle) {
  check_handle(kernel_handle, kTagKernel, "kernel");
  auto it = s.kernels.find(kernel_handle);
  if (it == s.kernels.end())
    throw vgpu::Error::make(vgpu::Err::InvalidValue, "invalid kernel handle");
  int dev = current_device(s);
  uint64_t mid = library_module_on(s, it->second.library, dev);
  const vgpu::ptx::EntryFn* fn = s.rt->device(dev).get_function(mid, it->second.name);
  uintptr_t fh = make_handle(s, kTagFunc);
  s.functions[fh] = {dev, it->second.library, fn, s.rt->device(dev).symbols(mid)};
  return fh;
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
    vgpu::apply_vram_override(profile);
    s.rt = std::make_unique<vgpu::runtime::Runtime>(profile, count);
    s.initialized = true;
    if (!quiet())
      std::fprintf(stderr, "[vgpu] virtual GPU platform initialized: %d x %s (%s)\n", count,
                   profile.id.c_str(), profile.model.c_str());
    vgpu::load_injection_library();
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuDriverGetVersion(int* driverVersion) {
  if (!driverVersion) return CUDA_ERROR_INVALID_VALUE;
  *driverVersion = vgpu::driver_version();
  return CUDA_SUCCESS;
}

// Both look the code up in the table the runtime shim shares
// (error_names.hpp), which lists every CUresult cuda.h declares. This used to
// be a table of the thirteen codes this file returns, so a program asking for
// the name of CUDA_ERROR_NO_DEVICE or CUDA_ERROR_NOT_READY -- codes it can
// receive from any library -- was told the code itself was invalid. A code no
// header declares still gets CUDA_ERROR_INVALID_VALUE and NULL, as documented.
VGPU_EXPORT CUresult cuGetErrorName(CUresult error, const char** pStr) {
  if (!pStr) return CUDA_ERROR_INVALID_VALUE;
  const vgpu::cuda::ErrorInfo* e = vgpu::cuda::find_error(static_cast<int>(error));
  *pStr = e ? e->driver_name : nullptr;
  return *pStr ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

VGPU_EXPORT CUresult cuGetErrorString(CUresult error, const char** pStr) {
  if (!pStr) return CUDA_ERROR_INVALID_VALUE;
  const vgpu::cuda::ErrorInfo* e = vgpu::cuda::find_error(static_cast<int>(error));
  *pStr = e && e->driver_name ? e->text : nullptr;
  return *pStr ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
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
    // Read the attribute as the integer the ABI actually passes, not as the
    // enum. A caller built against a newer CUDA header legitimately passes
    // values this shim's headers do not enumerate -- CUDA 13 sends 134, and
    // the `default:` arm below exists precisely to answer them. But *loading*
    // an enum object holding a value outside its enumerators is undefined:
    // UBSan reports it, and a compiler is entitled to assume the value is in
    // range and delete the default arm, which would turn forward compatibility
    // into a wrong answer with no diagnostic. memcpy reads the bytes without
    // making that claim about them.
    static_assert(sizeof(attrib) == sizeof(int), "CUdevice_attribute is not int-sized");
    int attr_id;
    std::memcpy(&attr_id, &attrib, sizeof attr_id);
    switch (attr_id) {
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
        *pi = extra_attribute(p, attr_id);
        break;
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
    ctx_stack().push_back(h);  // cuCtxCreate makes the new context current
    *pctx = reinterpret_cast<CUcontext>(h);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
  return cuCtxCreate_v2(pctx, flags, dev);
}
// CUDA 13's header maps cuCtxCreate to cuCtxCreate_v4, which takes a parameter
// block for green contexts and execution affinity. Without this symbol a
// program built against that toolkit fails to load at all -- the plain name it
// never calls is no help. The parameter block is not supported, and says so
// rather than being ignored.
VGPU_EXPORT CUresult cuCtxCreate_v3(CUcontext* pctx, void* exec_affinity_params, int num_params,
                                    unsigned int flags, CUdevice dev) {
  if (exec_affinity_params && num_params > 0) return CUDA_ERROR_NOT_SUPPORTED;
  return cuCtxCreate_v2(pctx, flags, dev);
}
VGPU_EXPORT CUresult cuCtxCreate_v4(CUcontext* pctx, void* ctx_create_params, unsigned int flags,
                                    CUdevice dev) {
  if (ctx_create_params) return CUDA_ERROR_NOT_SUPPORTED;
  return cuCtxCreate_v2(pctx, flags, dev);
}

VGPU_EXPORT CUresult cuCtxDestroy_v2(CUcontext ctx) {
  return api("cuCtxDestroy", true, false, [&](ShimState& s) {
    uintptr_t h = check_handle(reinterpret_cast<uintptr_t>(ctx), kTagCtx, "context");
    if (!s.contexts.erase(h)) return CUDA_ERROR_INVALID_CONTEXT;
    std::erase(ctx_stack(), h);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuCtxDestroy(CUcontext ctx) { return cuCtxDestroy_v2(ctx); }

VGPU_EXPORT CUresult cuCtxSetCurrent(CUcontext ctx) {
  return api("cuCtxSetCurrent", true, false, [&](ShimState& s) {
    if (!ctx) {  // NULL pops/clears the current context binding
      if (!ctx_stack().empty()) ctx_stack().pop_back();
      return CUDA_SUCCESS;
    }
    uintptr_t h = check_handle(reinterpret_cast<uintptr_t>(ctx), kTagCtx, "context");
    if (!s.contexts.count(h)) return CUDA_ERROR_INVALID_CONTEXT;
    if (!ctx_stack().empty())
      ctx_stack().back() = h;
    else
      ctx_stack().push_back(h);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuCtxGetCurrent(CUcontext* pctx) {
  return api("cuCtxGetCurrent", true, false, [&](ShimState& s) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    *pctx = ctx_stack().empty() ? nullptr : reinterpret_cast<CUcontext>(ctx_stack().back());
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
    ++s.primary_refs[dev];
    *pctx = reinterpret_cast<CUcontext>(it->second);  // NOTE: does not make it current
    return CUDA_SUCCESS;
  });
}

// Every retain needs its release, and a release with nothing retained is an
// error. It used to succeed no matter how many times it was called, which hid
// the bug a library's own teardown most often has: releasing a primary context
// it never retained, and so dropping a reference some other component holds.
// The context handle survives a count of zero -- a later retain gets the same
// one back -- but cuDevicePrimaryCtxGetState reports it inactive.
VGPU_EXPORT CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
  return api("cuDevicePrimaryCtxRelease", true, false, [&](ShimState& s) {
    check_device(s, dev);
    auto it = s.primary_refs.find(dev);
    if (it == s.primary_refs.end() || it->second == 0) {
      report("cuDevicePrimaryCtxRelease",
             "primary context of device " + std::to_string(dev) +
                 " released more times than it was retained");
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    --it->second;
    return CUDA_SUCCESS;
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
    owner_memory(s, dptr).free(dptr);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemFree(CUdeviceptr dptr) { return cuMemFree_v2(dptr); }

VGPU_EXPORT CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
  return api("cuMemcpyHtoD", true, false, [&](ShimState& s) {
    if (!srcHost && ByteCount) return CUDA_ERROR_INVALID_VALUE;
    owner_memory(s, dstDevice).write(dstDevice, srcHost, ByteCount);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemcpyHtoD(CUdeviceptr d, const void* h, size_t n) {
  return cuMemcpyHtoD_v2(d, h, n);
}

VGPU_EXPORT CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  return api("cuMemcpyDtoH", true, false, [&](ShimState& s) {
    if (!dstHost && ByteCount) return CUDA_ERROR_INVALID_VALUE;
    owner_memory(s, srcDevice).read(srcDevice, dstHost, ByteCount);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemcpyDtoH(void* h, CUdeviceptr d, size_t n) {
  return cuMemcpyDtoH_v2(h, d, n);
}

VGPU_EXPORT CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
  return api("cuMemcpyDtoD", true, false, [&](ShimState& s) {
    std::vector<uint8_t> tmp(ByteCount);
    owner_memory(s, srcDevice).read(srcDevice, tmp.data(), ByteCount);
    owner_memory(s, dstDevice).write(dstDevice, tmp.data(), ByteCount);
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
    std::string extracted;
    uint32_t magic = 0;
    std::memcpy(&magic, image, 4);
    if (magic == 0x466243B1u || magic == 0xBA55ED50u) {
      // A fatbin (wrapper or container): pull out the PTX image.
      extracted = best_ptx(image);
      text = extracted.c_str();
    } else if (text[0] == 0x7f) {
      throw vgpu::Error::make(vgpu::Err::Unsupported,
                              "cuModuleLoadData received a bare cubin/ELF image; VirtualGPU loads "
                              "PTX (embedded PTX text or a fatbin containing PTX)");
    }
    int dev = current_device(s);
    uint64_t mid = s.rt->device(dev).load_module(text);
    uintptr_t h = make_handle(s, kTagModule);
    s.modules[h] = {dev, mid};
    *module = reinterpret_cast<CUmodule>(h);
    return CUDA_SUCCESS;
  });
}

// A fatbin handed straight to the driver takes the same path as one passed to
// cuModuleLoadData: pull the PTX out and load it.
VGPU_EXPORT CUresult cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) {
  return cuModuleLoadData(module, fatCubin);
}

// Releasing the primary context. Nothing is cached per context here, so this
// succeeds without tearing down the device.
VGPU_EXPORT CUresult cuDevicePrimaryCtxReset(CUdevice) { return CUDA_SUCCESS; }

/* ---- runtime JIT linking ----
 *
 * On hardware the link step turns PTX and relocatable cubins into a single
 * cubin that cuModuleLoadData then loads. Here the module loader consumes PTX
 * directly, so the "cubin" this produces is PTX text: inputs are collected,
 * merged, and handed back as the completed image. That keeps the contract the
 * caller depends on -- complete() yields something loadable -- without pretending
 * to emit machine code.
 *
 * Numba is the reason this exists: it resolves and calls the link API for every
 * kernel it compiles, so refusing here stopped it before it ever reached a
 * launch.
 */
struct LinkState {
  std::vector<std::string> inputs;  // PTX modules, in the order they were added
  std::string image;                // merged result, kept alive until destroy
};

std::mutex& link_mutex() {
  static std::mutex m;
  return m;
}
std::unordered_map<void*, std::unique_ptr<LinkState>>& link_states() {
  static std::unordered_map<void*, std::unique_ptr<LinkState>> m;
  return m;
}

// Looks a link handle up, throwing rather than dereferencing something that was
// never handed out (or was already destroyed).
LinkState& link_state(void* h) {
  std::lock_guard<std::mutex> g(link_mutex());
  auto it = link_states().find(h);
  if (it == link_states().end())
    throw vgpu::Error::make(vgpu::Err::InvalidValue,
                            "cuLink call on a handle that is not an open link state");
  return *it->second;
}

// Merges PTX modules textually. Only the first module keeps its .version /
// .target / .address_size directives; repeating them is a parse error, and the
// module-level directives of a second input carry no information the first
// does not already have.
std::string merge_ptx(const std::vector<std::string>& inputs) {
  std::string out;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (i == 0) {
      out = inputs[0];
      if (!out.empty() && out.back() != '\n') out.push_back('\n');
      continue;
    }
    std::istringstream in(inputs[i]);
    std::string line;
    while (std::getline(in, line)) {
      std::string trimmed = line;
      size_t b = trimmed.find_first_not_of(" \t");
      if (b != std::string::npos) trimmed = trimmed.substr(b);
      if (trimmed.rfind(".version", 0) == 0 || trimmed.rfind(".target", 0) == 0 ||
          trimmed.rfind(".address_size", 0) == 0)
        continue;
      out += line;
      out.push_back('\n');
    }
  }
  return out;
}

// Accepts one input into a link state. PTX is taken as-is, a fatbin has its PTX
// pulled out, and a bare cubin is refused with the same message the module
// loader gives -- there is no SASS decoder behind this.
void link_add(LinkState& st, int type, const void* data, size_t size, const char* name) {
  const char* what = name && *name ? name : "<anonymous>";
  if (!data || size == 0)
    throw vgpu::Error::make(vgpu::Err::InvalidValue, "cuLinkAddData: empty input '", what, "'");
  uint32_t magic = 0;
  if (size >= 4) std::memcpy(&magic, data, 4);
  if (magic == 0x466243B1u || magic == 0xBA55ED50u) {
    st.inputs.push_back(best_ptx(data));
    return;
  }
  const char* text = static_cast<const char*>(data);
  if (text[0] == 0x7f)
    throw vgpu::Error::make(vgpu::Err::Unsupported, "cuLinkAddData: input '", what,
                            "' is a cubin/ELF image; VirtualGPU links PTX (CU_JIT_INPUT_PTX or a "
                            "fatbin containing PTX)");
  if (type != 1 /* CU_JIT_INPUT_PTX */ && type != 0 && type != 2)
    throw vgpu::Error::make(vgpu::Err::Unsupported, "cuLinkAddData: input type ", type,
                            " is not supported; VirtualGPU links PTX");
  // PTX may or may not carry a terminating NUL inside the reported size.
  size_t len = size;
  while (len > 0 && text[len - 1] == '\0') --len;
  st.inputs.emplace_back(text, len);
}

VGPU_EXPORT CUresult cuLinkAddFile_v2(void* state, int type, const char* path, unsigned int,
                                      void*, void*) {
  return api("cuLinkAddFile_v2", true, false, [&](ShimState&) {
    if (!state || !path) return CUDA_ERROR_INVALID_VALUE;
    LinkState& st = link_state(state);
    std::ifstream f(path, std::ios::binary);
    if (!f)
      throw vgpu::Error::make(vgpu::Err::InvalidValue, "cuLinkAddFile: cannot open '", path, "'");
    std::string blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    link_add(st, type, blob.data(), blob.size(), path);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuLinkAddFile(void* state, int type, const char* path, unsigned int n,
                                   void* keys, void* vals) {
  return cuLinkAddFile_v2(state, type, path, n, keys, vals);
}

VGPU_EXPORT CUresult cuLinkDestroy(void* state) {
  return api("cuLinkDestroy", true, false, [&](ShimState&) {
    if (!state) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::mutex> g(link_mutex());
    if (link_states().erase(state) == 0) return CUDA_ERROR_INVALID_VALUE;
    return CUDA_SUCCESS;
  });
}

/* ---- virtual memory management ----
 * The VMM API reserves address space and maps physical handles into it, which
 * is how PyTorch's expandable_segments allocator works. The sparse chunk model
 * here has no separable physical handles to map, so these refuse: a framework
 * that asked for a mapping and got silent success would write into memory that
 * was never backed.
 */
VGPU_EXPORT CUresult cuMemAddressReserve(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUresult cuMemAddressFree(CUdeviceptr, size_t) { return CUDA_ERROR_NOT_SUPPORTED; }
VGPU_EXPORT CUresult cuMemCreate(void*, size_t, const void*, unsigned long long) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUresult cuMemRelease(unsigned long long) { return CUDA_ERROR_NOT_SUPPORTED; }
VGPU_EXPORT CUresult cuMemMap(CUdeviceptr, size_t, size_t, unsigned long long, unsigned long long) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUresult cuMemUnmap(CUdeviceptr, size_t) { return CUDA_ERROR_NOT_SUPPORTED; }
VGPU_EXPORT CUresult cuMemSetAccess(CUdeviceptr, size_t, const void*, size_t) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUresult cuMemGetAllocationGranularity(size_t* granularity, const void*, int) {
  // Answering this one is harmless and lets a caller size a request before
  // discovering the mapping calls are unavailable.
  if (!granularity) return CUDA_ERROR_INVALID_VALUE;
  *granularity = 64u * 1024u;  // the chunk size the sparse backing uses
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuMemExportToShareableHandle(void*, unsigned long long, int, unsigned long long) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUresult cuMemImportFromShareableHandle(unsigned long long*, void*, int) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

// Multicast objects span several devices' memory; there is no such fabric here.
VGPU_EXPORT CUresult cuMulticastCreate(unsigned long long*, const void*) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUresult cuMulticastAddDevice(unsigned long long, CUdevice) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUresult cuMulticastBindMem(unsigned long long, size_t, unsigned long long, size_t,
                                        size_t, unsigned long long) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

// Stream memory ops. Work is synchronous, so the write happens now.
VGPU_EXPORT CUresult cuStreamWriteValue32(CUstream, CUdeviceptr addr, unsigned int value,
                                          unsigned int) {
  return api("cuStreamWriteValue32", true, false, [&](ShimState& s) {
    owner_memory(s, addr).write(addr, &value, sizeof value);
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
    s.functions[fh] = {dev, h, fn, s.rt->device(dev).symbols(mid)};
    *hfunc = reinterpret_cast<CUfunction>(fh);
    return CUDA_SUCCESS;
  });
}

namespace {
CUresult launch_kernel_common(const char* api_name, CUfunction f, unsigned int gridDimX,
                              unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX,
                              unsigned int blockDimY, unsigned int blockDimZ,
                              unsigned int sharedMemBytes, CUstream hStream, void** kernelParams,
                              void** extra, bool cooperative) {
  return api(api_name, true, true, [&](ShimState& s) {
    uintptr_t fh = reinterpret_cast<uintptr_t>(f);
    if ((fh & 7) == kTagKernel) fh = kernel_to_function(s, fh);  // CUkernel is launchable directly
    check_handle(fh, kTagFunc, "function");
    auto it = s.functions.find(fh);
    if (it == s.functions.end()) return CUDA_ERROR_NOT_FOUND;
    const FuncRec& rec = it->second;
    // Every stream is the synchronous default stream in this engine: legacy
    // (0/1), per-thread (2), and created stream handles all execute in order.
    (void)hStream;
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
    cfg.cooperative = cooperative;
    if (cooperative) {
      // Every block of a cooperative launch waits for every other, so a grid
      // that cannot all be resident does not run slowly -- it hangs. Hardware
      // refuses it, and so does this.
      const vgpu::DeviceProfile& p = s.rt->device(rec.device).profile();
      const uint64_t resident = uint64_t{p.limits.max_blocks_per_sm} * p.limits.multiprocessors;
      const uint64_t want = uint64_t{gridDimX} * gridDimY * gridDimZ;
      if (resident && want > resident)
        throw vgpu::Error::make(vgpu::Err::LaunchConfig, "cooperative launch of ", want,
                                " blocks exceeds what ", p.id, " can hold resident (",
                                p.limits.max_blocks_per_sm, " blocks/SM x ",
                                p.limits.multiprocessors, " SMs = ", resident, ")");
    }
    s.rt->device(rec.device).launch(*rec.fn, cfg, args, rec.syms);
    return CUDA_SUCCESS;
  });
}
}  // namespace

VGPU_EXPORT CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                                    unsigned int gridDimZ, unsigned int blockDimX,
                                    unsigned int blockDimY, unsigned int blockDimZ,
                                    unsigned int sharedMemBytes, CUstream hStream,
                                    void** kernelParams, void** extra) {
  return launch_kernel_common("cuLaunchKernel", f, gridDimX, gridDimY, gridDimZ, blockDimX,
                              blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra,
                              /*cooperative=*/false);
}

/* ======================================================================== */
/* Extended surface for hosting NVIDIA's static cudart (unmodified binaries) */
/* ======================================================================== */

/* ---- context-independent libraries (CUDA 12+ cuLibrary API) ---- */

VGPU_EXPORT CUresult cuLibraryLoadData(void** library, const void* code, void* jitOptions,
                                       void** jitOptionValues, unsigned int numJitOptions,
                                       void* libraryOptions, void** libraryOptionValues,
                                       unsigned int numLibraryOptions) {
  (void)jitOptions;
  (void)jitOptionValues;
  (void)numJitOptions;
  (void)libraryOptions;
  (void)libraryOptionValues;
  (void)numLibraryOptions;
  return api("cuLibraryLoadData", true, false, [&](ShimState& s) {
    if (!library || !code) return CUDA_ERROR_INVALID_VALUE;
    uint32_t magic = 0;
    std::memcpy(&magic, code, 4);
    LibRec lib;
    if (magic == 0x466243B1u || magic == 0xBA55ED50u)
      lib.ptx = best_ptx(code);
    else
      lib.ptx.assign(static_cast<const char*>(code));  // NUL-terminated PTX text
    uintptr_t h = make_handle(s, kTagLibrary);
    s.libraries[h] = std::move(lib);
    *library = reinterpret_cast<void*>(h);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuLibraryUnload(void* library) {
  return api("cuLibraryUnload", true, false, [&](ShimState& s) {
    uintptr_t h = check_handle(reinterpret_cast<uintptr_t>(library), kTagLibrary, "library");
    auto it = s.libraries.find(h);
    if (it == s.libraries.end()) return CUDA_ERROR_INVALID_VALUE;
    for (auto& [dev, mid] : it->second.per_device_module) s.rt->device(dev).unload_module(mid);
    // Drop kernels and functions minted from this library.
    for (auto k = s.kernels.begin(); k != s.kernels.end();)
      k = k->second.library == h ? s.kernels.erase(k) : std::next(k);
    for (auto f = s.functions.begin(); f != s.functions.end();)
      f = f->second.module_handle == h ? s.functions.erase(f) : std::next(f);
    s.libraries.erase(it);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuLibraryGetKernel(void** pKernel, void* library, const char* name) {
  return api("cuLibraryGetKernel", true, false, [&](ShimState& s) {
    if (!pKernel || !name) return CUDA_ERROR_INVALID_VALUE;
    uintptr_t h = check_handle(reinterpret_cast<uintptr_t>(library), kTagLibrary, "library");
    auto it = s.libraries.find(h);
    if (it == s.libraries.end()) return CUDA_ERROR_INVALID_VALUE;
    // Validate the kernel exists (parse errors surface here, like a JIT would).
    int dev = ctx_stack().empty() ? 0 : current_device(s);
    uint64_t mid = library_module_on(s, h, dev);
    (void)s.rt->device(dev).get_function(mid, name);
    uintptr_t kh = make_handle(s, kTagKernel);
    s.kernels[kh] = {h, name};
    *pKernel = reinterpret_cast<void*>(kh);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuLibraryGetModule(CUmodule* pMod, void* library) {
  return api("cuLibraryGetModule", true, false, [&](ShimState& s) {
    if (!pMod) return CUDA_ERROR_INVALID_VALUE;
    uintptr_t h = check_handle(reinterpret_cast<uintptr_t>(library), kTagLibrary, "library");
    int dev = current_device(s);
    uint64_t mid = library_module_on(s, h, dev);
    uintptr_t mh = make_handle(s, kTagModule);
    s.modules[mh] = {dev, mid};
    *pMod = reinterpret_cast<CUmodule>(mh);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuKernelGetFunction(CUfunction* pFunc, void* kernel) {
  return api("cuKernelGetFunction", true, false, [&](ShimState& s) {
    if (!pFunc) return CUDA_ERROR_INVALID_VALUE;
    uintptr_t fh = kernel_to_function(s, reinterpret_cast<uintptr_t>(kernel));
    *pFunc = reinterpret_cast<CUfunction>(fh);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuKernelGetName(const char** name, void* kernel) {
  return api("cuKernelGetName", true, false, [&](ShimState& s) {
    if (!name) return CUDA_ERROR_INVALID_VALUE;
    uintptr_t kh = check_handle(reinterpret_cast<uintptr_t>(kernel), kTagKernel, "kernel");
    auto it = s.kernels.find(kh);
    if (it == s.kernels.end()) return CUDA_ERROR_INVALID_VALUE;
    *name = it->second.name.c_str();
    return CUDA_SUCCESS;
  });
}

/* ---- function/kernel attributes ---- */

namespace {
int func_attribute(const vgpu::ptx::EntryFn* fn, const vgpu::DeviceProfile& p, int attrib) {
  switch (attrib) {
    case 0: return static_cast<int>(p.limits.max_threads_per_block);  // MAX_THREADS_PER_BLOCK
    case 1: return static_cast<int>(fn ? fn->static_shared_size : 0); // SHARED_SIZE_BYTES
    case 2: return 0;                                                 // CONST_SIZE_BYTES
    case 3: return static_cast<int>(fn ? fn->local_frame_size : 0);   // LOCAL_SIZE_BYTES
    case 4:                                                           // NUM_REGS
      return fn ? static_cast<int>(
                      vgpu::exec::kernel_resources(*fn, p, p.limits.max_threads_per_block, 0)
                          .usage.regs_per_thread)
                : 0;
    case 5: return 90;                                                // PTX_VERSION
    case 6: return p.cc_major * 10 + p.cc_minor;                      // BINARY_VERSION
    case 8: return static_cast<int>(p.limits.shared_mem_per_block_optin);
    default: return 0;
  }
}
}  // namespace

VGPU_EXPORT CUresult cuFuncGetAttribute(int* pi, int attrib, CUfunction hfunc) {
  return api("cuFuncGetAttribute", true, false, [&](ShimState& s) {
    if (!pi) return CUDA_ERROR_INVALID_VALUE;
    auto it = s.functions.find(reinterpret_cast<uintptr_t>(hfunc));
    if (it == s.functions.end()) return CUDA_ERROR_INVALID_VALUE;
    *pi = func_attribute(it->second.fn, s.rt->device(it->second.device).profile(), attrib);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuKernelGetAttribute(int* pi, int attrib, void* kernel, CUdevice dev) {
  return api("cuKernelGetAttribute", true, false, [&](ShimState& s) {
    if (!pi) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    uintptr_t kh = check_handle(reinterpret_cast<uintptr_t>(kernel), kTagKernel, "kernel");
    if (!s.kernels.count(kh)) return CUDA_ERROR_INVALID_VALUE;
    *pi = func_attribute(nullptr, s.rt->device(dev).profile(), attrib);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuFuncSetAttribute(CUfunction, int, int) { return CUDA_SUCCESS; }
VGPU_EXPORT CUresult cuKernelSetAttribute(int, int, void*, CUdevice) { return CUDA_SUCCESS; }
VGPU_EXPORT CUresult cuFuncSetCacheConfig(CUfunction, int) { return CUDA_SUCCESS; }
VGPU_EXPORT CUresult cuFuncIsLoaded(int* state, CUfunction) {
  if (state) *state = 1;  // CU_FUNCTION_LOADING_STATE_LOADED
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuFuncLoad(CUfunction) { return CUDA_SUCCESS; }

VGPU_EXPORT CUresult cuLaunchKernelEx(const void* config, CUfunction f, void** kernelParams,
                                      void** extra) {
  // CUlaunchConfig: 6x u32 dims, u32 sharedMemBytes, CUstream, attrs*, numAttrs.
  struct LaunchCfgABI {
    unsigned gx, gy, gz, bx, by, bz;
    unsigned shared_bytes;
    CUstream stream;
    void* attrs;
    unsigned num_attrs;
  };
  const auto* c = static_cast<const LaunchCfgABI*>(config);
  if (!c) return CUDA_ERROR_INVALID_VALUE;
  // Attributes are refused rather than dropped.
  //
  // This shim forwarded the config and ignored `attrs` entirely, which was
  // harmless while nothing it could carry was implemented. Thread-block
  // clusters changed that: a launch carrying
  // CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION would have run with no cluster, and
  // every block would have read %cluster_ctarank as 0 and %cluster_nctarank
  // as 1. It would have succeeded and been wrong, silently, which is the one
  // outcome this engine is built to not produce.
  //
  // Walking the array is not an option here. Unlike the runtime shim, this
  // file deliberately depends on no vendor header -- see vgpu_cuda.h -- and
  // the entry stride is not a constant that can be hard-coded:
  // sizeof(CUlaunchAttribute) is 72 with CUDA 13 against the 40 an older
  // toolkit's union gives, so a fixed guess reads the wrong bytes on some
  // toolkit and reports a cluster shape nobody asked for.
  //
  // So: no attributes is the supported case, and anything else says so. The
  // runtime entry point (cudaLaunchKernelEx) does read attributes, because
  // that shim is compiled against the vendor headers, and it is the path CUDA
  // C++ actually takes.
  if (c->num_attrs != 0 && c->attrs != nullptr) {
    if (!quiet())
      std::fprintf(stderr,
                   "[vgpu] cuLaunchKernelEx: %u launch attribute(s) given; this entry point "
                   "cannot read them (the attribute struct size differs between CUDA "
                   "toolkits) and will not ignore them silently. Use cudaLaunchKernelEx, "
                   "which does read them.\n",
                   c->num_attrs);
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  return cuLaunchKernel(f, c->gx, c->gy, c->gz, c->bx, c->by, c->bz, c->shared_bytes, c->stream,
                        kernelParams, extra);
}

/* ---- memcpy/memset variants (everything is synchronous) ---- */

VGPU_EXPORT CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr d, const void* h, size_t n, CUstream) {
  return cuMemcpyHtoD_v2(d, h, n);
}
VGPU_EXPORT CUresult cuMemcpyHtoDAsync(CUdeviceptr d, const void* h, size_t n, CUstream) {
  return cuMemcpyHtoD_v2(d, h, n);
}
VGPU_EXPORT CUresult cuMemcpyDtoHAsync_v2(void* h, CUdeviceptr d, size_t n, CUstream) {
  return cuMemcpyDtoH_v2(h, d, n);
}
VGPU_EXPORT CUresult cuMemcpyDtoHAsync(void* h, CUdeviceptr d, size_t n, CUstream) {
  return cuMemcpyDtoH_v2(h, d, n);
}
VGPU_EXPORT CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr a, CUdeviceptr b, size_t n, CUstream) {
  return cuMemcpyDtoD_v2(a, b, n);
}

namespace {
bool is_device_ptr(uint64_t p) { return vgpu::is_device_va(p); }
}  // namespace

// Direction-inferring copies (UVA style): device-range vs host pointers.
VGPU_EXPORT CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t n) {
  bool dd = is_device_ptr(dst), sd = is_device_ptr(src);
  if (dd && sd) return cuMemcpyDtoD_v2(dst, src, n);
  if (dd) return cuMemcpyHtoD_v2(dst, reinterpret_cast<const void*>(src), n);
  if (sd) return cuMemcpyDtoH_v2(reinterpret_cast<void*>(dst), src, n);
  std::memcpy(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), n);
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream) {
  return cuMemcpy(dst, src, n);
}

namespace {
template <typename T>
CUresult memset_impl(const char* name, CUdeviceptr dptr, T value, size_t n) {
  return api(name, true, false, [&](ShimState& s) {
    owner_memory(s, dptr).fill(dptr, reinterpret_cast<const uint8_t*>(&value), sizeof(T),
                               n * sizeof(T));
    return CUDA_SUCCESS;
  });
}
}  // namespace

/* ---- entry points frameworks link against ----
 * Each of these reports a failure rather than pretending. A cooperative launch
 * whose kernel calls grid.sync() would deadlock or silently produce wrong
 * results if run as an ordinary launch, and a JIT-linked module that quietly
 * did nothing would surface much later as a wrong answer.
 */
VGPU_EXPORT CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX,
                                               unsigned int gridDimY, unsigned int gridDimZ,
                                               unsigned int blockDimX, unsigned int blockDimY,
                                               unsigned int blockDimZ, unsigned int sharedMemBytes,
                                               CUstream hStream, void** kernelParams) {
  return launch_kernel_common("cuLaunchCooperativeKernel", f, gridDimX, gridDimY, gridDimZ,
                              blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream,
                              kernelParams, nullptr, /*cooperative=*/true);
}
// The JIT-link types are not in the header subset this file compiles against;
// these take opaque parameters because they only need to exist and refuse.
VGPU_EXPORT CUresult cuLinkCreate_v2(unsigned int, void*, void*, void** stateOut) {
  return api("cuLinkCreate_v2", true, false, [&](ShimState&) {
    if (!stateOut) return CUDA_ERROR_INVALID_VALUE;
    // JIT options (register caps, optimisation level, log buffers) describe a
    // code generator this has no equivalent of, so they are accepted and left
    // unused rather than refused.
    auto st = std::make_unique<LinkState>();
    void* h = st.get();
    {
      std::lock_guard<std::mutex> g(link_mutex());
      link_states()[h] = std::move(st);
    }
    *stateOut = h;
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuLinkCreate(unsigned int n, void* keys, void* vals, void** stateOut) {
  return cuLinkCreate_v2(n, keys, vals, stateOut);
}

VGPU_EXPORT CUresult cuLinkAddData_v2(void* state, int type, void* data, size_t size,
                                      const char* name, unsigned int, void*, void*) {
  return api("cuLinkAddData_v2", true, false, [&](ShimState&) {
    if (!state) return CUDA_ERROR_INVALID_VALUE;
    link_add(link_state(state), type, data, size, name);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuLinkAddData(void* state, int type, void* data, size_t size, const char* name,
                                   unsigned int n, void* keys, void* vals) {
  return cuLinkAddData_v2(state, type, data, size, name, n, keys, vals);
}

VGPU_EXPORT CUresult cuLinkComplete(void* state, void** imageOut, size_t* sizeOut) {
  return api("cuLinkComplete", true, false, [&](ShimState&) {
    if (!state || !imageOut) return CUDA_ERROR_INVALID_VALUE;
    LinkState& st = link_state(state);
    if (st.inputs.empty())
      throw vgpu::Error::make(vgpu::Err::InvalidValue, "cuLinkComplete: no inputs were added");
    st.image = merge_ptx(st.inputs);
    st.image.push_back('\0');  // the loader reads the image as a C string
    *imageOut = st.image.data();
    // The reported size excludes the terminator, matching how a cubin size is
    // reported: the caller only ever passes it back to cuModuleLoadDataEx.
    if (sizeOut) *sizeOut = st.image.size() - 1;
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuTensorMapEncodeTiled(void*, unsigned int, unsigned int, void*,
                                            const unsigned long long*, const unsigned long long*,
                                            const unsigned int*, const unsigned int*, unsigned int,
                                            unsigned int, unsigned int, unsigned int) {
  return CUDA_ERROR_NOT_SUPPORTED;  // Hopper TMA descriptors
}

VGPU_EXPORT CUresult cuMemsetD8_v2(CUdeviceptr d, unsigned char v, size_t n) {
  return memset_impl("cuMemsetD8", d, v, n);
}
VGPU_EXPORT CUresult cuMemsetD16_v2(CUdeviceptr d, unsigned short v, size_t n) {
  return memset_impl("cuMemsetD16", d, v, n);
}
VGPU_EXPORT CUresult cuMemsetD32_v2(CUdeviceptr d, unsigned int v, size_t n) {
  return memset_impl("cuMemsetD32", d, v, n);
}
VGPU_EXPORT CUresult cuMemsetD8Async(CUdeviceptr d, unsigned char v, size_t n, CUstream) {
  return memset_impl("cuMemsetD8", d, v, n);
}
VGPU_EXPORT CUresult cuMemsetD16Async(CUdeviceptr d, unsigned short v, size_t n, CUstream) {
  return memset_impl("cuMemsetD16", d, v, n);
}
VGPU_EXPORT CUresult cuMemsetD32Async(CUdeviceptr d, unsigned int v, size_t n, CUstream) {
  return memset_impl("cuMemsetD32", d, v, n);
}

VGPU_EXPORT CUresult cuMemGetAddressRange_v2(CUdeviceptr* base, size_t* size, CUdeviceptr dptr) {
  return api("cuMemGetAddressRange", true, false, [&](ShimState& s) {
    uint64_t b = 0, sz = 0;
    if (!owner_memory(s, dptr).find_allocation(dptr, &b, &sz)) return CUDA_ERROR_INVALID_VALUE;
    if (base) *base = b;
    if (size) *size = sz;
    return CUDA_SUCCESS;
  });
}

/* ---- host (pinned) memory: plain aligned host allocations ---- */

VGPU_EXPORT CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int) {
  return api("cuMemHostAlloc", true, false, [&](ShimState& s) {
    if (!pp || bytesize == 0) return CUDA_ERROR_INVALID_VALUE;
    void* p = std::aligned_alloc(4096, (bytesize + 4095) / 4096 * 4096);
    if (!p) return CUDA_ERROR_OUT_OF_MEMORY;
    s.host_allocs.insert(p);
    *pp = p;
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemAllocHost_v2(void** pp, size_t bytesize) {
  return cuMemHostAlloc(pp, bytesize, 0);
}
VGPU_EXPORT CUresult cuMemFreeHost(void* p) {
  return api("cuMemFreeHost", true, false, [&](ShimState& s) {
    if (!s.host_allocs.erase(p)) return CUDA_ERROR_INVALID_VALUE;
    std::free(p);
    return CUDA_SUCCESS;
  });
}

/* ---- streams (all synchronous) and events (wall-clock timestamps) ---- */

VGPU_EXPORT CUresult cuStreamCreate(CUstream* s_out, unsigned int) {
  return api("cuStreamCreate", true, false, [&](ShimState& s) {
    if (!s_out) return CUDA_ERROR_INVALID_VALUE;
    uintptr_t h = make_handle(s, kTagStream);
    s.streams.insert(h);
    *s_out = reinterpret_cast<CUstream>(h);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuStreamCreateWithPriority(CUstream* s_out, unsigned int flags, int) {
  return cuStreamCreate(s_out, flags);
}
VGPU_EXPORT CUresult cuStreamDestroy_v2(CUstream stream) {
  return api("cuStreamDestroy", true, false, [&](ShimState& s) {
    s.streams.erase(reinterpret_cast<uintptr_t>(stream));
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuStreamDestroy(CUstream stream) { return cuStreamDestroy_v2(stream); }
VGPU_EXPORT CUresult cuStreamSynchronize(CUstream) { return cuCtxSynchronize(); }
VGPU_EXPORT CUresult cuStreamQuery(CUstream) { return CUDA_SUCCESS; }  // always idle
VGPU_EXPORT CUresult cuStreamWaitEvent(CUstream, void*, unsigned int) { return CUDA_SUCCESS; }
VGPU_EXPORT CUresult cuStreamGetPriority(CUstream, int* p) {
  if (p) *p = 0;
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuStreamGetFlags(CUstream, unsigned int* f) {
  if (f) *f = 0;
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuStreamGetCaptureInfo_v2(CUstream, int* status, unsigned long long* id,
                                               void*, const void**, size_t*) {
  if (status) *status = 0;  // CU_STREAM_CAPTURE_STATUS_NONE
  if (id) *id = 0;
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuStreamIsCapturing(CUstream, int* status) {
  if (status) *status = 0;
  return CUDA_SUCCESS;
}

VGPU_EXPORT CUresult cuEventCreate(void** ev, unsigned int) {
  return api("cuEventCreate", true, false, [&](ShimState& s) {
    if (!ev) return CUDA_ERROR_INVALID_VALUE;
    uintptr_t h = make_handle(s, kTagEvent);
    s.events[h] = {};
    *ev = reinterpret_cast<void*>(h);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuEventRecord(void* ev, CUstream) {
  return api("cuEventRecord", true, false, [&](ShimState& s) {
    auto it = s.events.find(reinterpret_cast<uintptr_t>(ev));
    if (it == s.events.end()) return CUDA_ERROR_INVALID_VALUE;
    it->second.recorded = true;
    clock_gettime(CLOCK_MONOTONIC, &it->second.when);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuEventQuery(void*) { return CUDA_SUCCESS; }
VGPU_EXPORT CUresult cuEventSynchronize(void*) { return CUDA_SUCCESS; }
VGPU_EXPORT CUresult cuEventDestroy_v2(void* ev) {
  return api("cuEventDestroy", true, false, [&](ShimState& s) {
    s.events.erase(reinterpret_cast<uintptr_t>(ev));
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuEventDestroy(void* ev) { return cuEventDestroy_v2(ev); }
VGPU_EXPORT CUresult cuEventElapsedTime(float* ms, void* start, void* end) {
  return api("cuEventElapsedTime", true, false, [&](ShimState& s) {
    auto a = s.events.find(reinterpret_cast<uintptr_t>(start));
    auto b = s.events.find(reinterpret_cast<uintptr_t>(end));
    if (!ms || a == s.events.end() || b == s.events.end() || !a->second.recorded ||
        !b->second.recorded)
      return CUDA_ERROR_INVALID_VALUE;
    double ns = (b->second.when.tv_sec - a->second.when.tv_sec) * 1e9 +
                (b->second.when.tv_nsec - a->second.when.tv_nsec);
    *ms = static_cast<float>(ns / 1e6);
    return CUDA_SUCCESS;
  });
}
// Same story as cuCtxCreate: CUDA 13 renames this one too, and a program built
// there resolves the versioned name, not the plain one.
VGPU_EXPORT CUresult cuEventElapsedTime_v2(float* ms, void* start, void* end) {
  return cuEventElapsedTime(ms, start, end);
}

/* ---- context odds and ends static cudart asks about ---- */

VGPU_EXPORT CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
  return api("cuCtxPushCurrent", true, false, [&](ShimState& s) {
    uintptr_t h = check_handle(reinterpret_cast<uintptr_t>(ctx), kTagCtx, "context");
    if (!s.contexts.count(h)) return CUDA_ERROR_INVALID_CONTEXT;
    ctx_stack().push_back(h);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuCtxPushCurrent(CUcontext ctx) { return cuCtxPushCurrent_v2(ctx); }
VGPU_EXPORT CUresult cuCtxPopCurrent_v2(CUcontext* pctx) {
  return api("cuCtxPopCurrent", true, false, [&](ShimState& s) {
    if (ctx_stack().empty()) return CUDA_ERROR_INVALID_CONTEXT;
    if (pctx) *pctx = reinterpret_cast<CUcontext>(ctx_stack().back());
    ctx_stack().pop_back();
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuCtxPopCurrent(CUcontext* pctx) { return cuCtxPopCurrent_v2(pctx); }

VGPU_EXPORT CUresult cuCtxGetLimit(size_t* v, int limit) {
  if (!v) return CUDA_ERROR_INVALID_VALUE;
  switch (limit) {
    case 0: *v = 1024; break;              // STACK_SIZE
    case 1: *v = 1024 * 1024; break;       // PRINTF_FIFO_SIZE
    case 2: *v = 8 * 1024 * 1024; break;   // MALLOC_HEAP_SIZE
    default: *v = 0; break;
  }
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuCtxSetLimit(int, size_t) { return CUDA_SUCCESS; }
VGPU_EXPORT CUresult cuCtxGetApiVersion(CUcontext, unsigned int* v) {
  if (v) *v = 3020;
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuCtxGetStreamPriorityRange(int* least, int* greatest) {
  if (least) *least = 0;
  if (greatest) *greatest = 0;
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuCtxGetFlags(unsigned int* f) {
  if (f) *f = 0;
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuCtxSetFlags(unsigned int) { return CUDA_SUCCESS; }
VGPU_EXPORT CUresult cuCtxGetId(CUcontext, unsigned long long* id) {
  if (id) *id = 1;
  return CUDA_SUCCESS;
}

VGPU_EXPORT CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice, unsigned int) { return CUDA_SUCCESS; }
VGPU_EXPORT CUresult cuDevicePrimaryCtxSetFlags(CUdevice, unsigned int) { return CUDA_SUCCESS; }
VGPU_EXPORT CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) {
  return api("cuDevicePrimaryCtxGetState", true, false, [&](ShimState& s) {
    check_device(s, dev);
    if (flags) *flags = 0;
    const auto refs = s.primary_refs.find(dev);
    if (active) *active = refs != s.primary_refs.end() && refs->second > 0 ? 1 : 0;
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuDevicePrimaryCtxReset_v2(CUdevice) { return CUDA_SUCCESS; }

VGPU_EXPORT CUresult cuModuleGetLoadingMode(int* mode) {
  if (!mode) return CUDA_ERROR_INVALID_VALUE;
  *mode = 1;  // CU_MODULE_EAGER_LOADING
  return CUDA_SUCCESS;
}

VGPU_EXPORT CUresult cuDeviceGetUuid(void* uuid, CUdevice dev) {
  return api("cuDeviceGetUuid", true, false, [&](ShimState& s) {
    if (!uuid) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    // Deterministic fake UUID: "VGPU" + profile hash + ordinal.
    unsigned char b[16] = {'V', 'G', 'P', 'U'};
    const std::string& id = s.rt->device(dev).profile().id;
    uint32_t h = 2166136261u;
    for (char c : id) h = (h ^ static_cast<unsigned char>(c)) * 16777619u;
    std::memcpy(b + 4, &h, 4);
    b[8] = static_cast<unsigned char>(dev);
    std::memcpy(uuid, b, 16);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuDeviceGetUuid_v2(void* uuid, CUdevice dev) { return cuDeviceGetUuid(uuid, dev); }

VGPU_EXPORT CUresult cuDeviceGetPCIBusId(char* id, int len, CUdevice dev) {
  if (!id || len <= 0) return CUDA_ERROR_INVALID_VALUE;
  std::snprintf(id, static_cast<size_t>(len), "0000:%02x:00.0", dev + 1);
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuDeviceCanAccessPeer(int* can, CUdevice, CUdevice) {
  if (can) *can = 0;
  return CUDA_SUCCESS;
}

/* ---- pointer queries (expected probes: fail quietly, no stderr) ---- */

VGPU_EXPORT CUresult cuPointerGetAttribute(void* data, int attribute, CUdeviceptr ptr) {
  if (!data) return CUDA_ERROR_INVALID_VALUE;
  bool dev = vgpu::is_device_va(ptr);
  switch (attribute) {
    case 1: {  // CU_POINTER_ATTRIBUTE_CONTEXT
      if (!dev) return CUDA_ERROR_INVALID_VALUE;
      *static_cast<void**>(data) = nullptr;
      return CUDA_SUCCESS;
    }
    case 2:  // CU_POINTER_ATTRIBUTE_MEMORY_TYPE
      if (!dev) return CUDA_ERROR_INVALID_VALUE;
      *static_cast<unsigned int*>(data) = 2;  // CU_MEMORYTYPE_DEVICE
      return CUDA_SUCCESS;
    case 3:  // DEVICE_POINTER
      if (!dev) return CUDA_ERROR_INVALID_VALUE;
      *static_cast<CUdeviceptr*>(data) = ptr;
      return CUDA_SUCCESS;
    default:
      return CUDA_ERROR_INVALID_VALUE;
  }
}

/* ---- occupancy (functional placeholder: enough blocks to look busy) ---- */

VGPU_EXPORT CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* num, CUfunction, int blockSize,
                                                                 size_t) {
  if (!num || blockSize <= 0) return CUDA_ERROR_INVALID_VALUE;
  *num = std::max(1, std::min(1536 / blockSize, 16));
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* num, CUfunction f,
                                                                          int blockSize, size_t dyn,
                                                                          unsigned int) {
  return cuOccupancyMaxActiveBlocksPerMultiprocessor(num, f, blockSize, dyn);
}

/* ---- internal export tables: refused, loudly under trace ---- */

/*
 * The "dark API": undocumented internal vtables that NVIDIA's static cudart
 * requires from the driver. Layouts follow the de-facto interface established
 * by prior clean-room reimplementations (gpgpu-sim, ZLUDA) and are verified
 * empirically against the local toolkit's cudart. Entries we understand get
 * real implementations; every other slot is a logging stub that returns
 * success, so a newly-exercised slot shows up under VGPU_TRACE instead of
 * crashing on a null jump.
 */
namespace {

// uuid 6bd5fb6c-5bf4-e74a-8987-d93912fd9df9 — "cudart interface"
constexpr unsigned char kUuidCudartInterface[16] = {0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a,
                                                    0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9};
// uuid a094798c-2e74-2e74-93f2-0800200c0a66 — tools runtime callback hooks
constexpr unsigned char kUuidToolsHooks[16] = {0xa0, 0x94, 0x79, 0x8c, 0x2e, 0x74, 0x2e, 0x74,
                                               0x93, 0xf2, 0x08, 0x00, 0x20, 0x0c, 0x0a, 0x66};
// uuid c693336e-1121-df11-a8c3-68f355d89593 — context-local storage
constexpr unsigned char kUuidCtxStorage[16] = {0xc6, 0x93, 0x33, 0x6e, 0x11, 0x21, 0xdf, 0x11,
                                               0xa8, 0xc3, 0x68, 0xf3, 0x55, 0xd8, 0x95, 0x93};
// Other internal tables (semantics unknown): cudart refuses to initialize
// unless every table it asks for exists, so unknown UUIDs are served generic
// logging-stub tables, assigned on demand and grown empirically (VGPU_TRACE).

// An experiment knob, not a feature: VGPU_DARK_FILL=1 makes every stub zero the
// eight bytes at each argument that looks like a writable pointer before
// returning. The dark API has no specification, so the only question that can
// be asked of it is "does the caller care?" -- and a caller that reads an
// out-parameter we never wrote is the most likely way an unspecified slot
// changes behaviour. Off by default: writing through a pointer whose meaning is
// unknown is a guess, and a wrong guess corrupts the caller's memory.
bool dark_fill() {
  const char* e = std::getenv("VGPU_DARK_FILL");
  return e && e[0] == '1';
}

// Plausibly a pointer into the caller's own mapping. Deliberately conservative:
// only addresses in the range a userspace heap or stack occupies.
bool looks_writable(uintptr_t v) {
  return v > 0x10000 && v < (uintptr_t{1} << 47) && (v & 7) == 0;
}

template <int Table, int Index>
uintptr_t dark_stub(uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4,
                    uintptr_t a5) {
  if (trace())
    std::fprintf(stderr,
                 "[vgpu][trace] dark-api table %d slot %d(%#lx, %#lx, %#lx, %#lx, %#lx, %#lx)"
                 " (stubbed -> 0)\n",
                 Table, Index, a0, a1, a2, a3, a4, a5);
  if (dark_fill())
    for (uintptr_t a : {a0, a1, a2, a3, a4, a5})
      if (looks_writable(a)) *reinterpret_cast<uint64_t*>(a) = 0;
  return 0;
}

// cudart-interface slot 1/6/8: materialize a module from a registered fatbin.
CUresult dark_get_module(CUmodule* mod, const void* wrapper) {
  return cuModuleLoadData(mod, wrapper);
}
CUresult dark_get_module_ext1(CUmodule* mod, const void* wrapper, void*, void*, uintptr_t) {
  return cuModuleLoadData(mod, wrapper);
}
CUresult dark_get_module_ext2(const void* wrapper, CUmodule* mod, void*, void*, unsigned int) {
  return cuModuleLoadData(mod, wrapper);
}

// Context-local storage: cudart stashes per-context state through these.
struct CtxStorageRec {
  void* state = nullptr;
};
std::mutex g_ctx_storage_mu;
std::unordered_map<const void*, std::unordered_map<uintptr_t, CtxStorageRec>> g_ctx_storage;

CUresult dark_ctx_storage_ctor(CUcontext ctx, void* mgr, void* state, void* dtor) {
  (void)dtor;
  std::lock_guard<std::mutex> lock(g_ctx_storage_mu);
  g_ctx_storage[mgr][reinterpret_cast<uintptr_t>(ctx)] = {state};
  return CUDA_SUCCESS;
}
CUresult dark_ctx_storage_dtor(void* mgr, void*) {
  std::lock_guard<std::mutex> lock(g_ctx_storage_mu);
  g_ctx_storage.erase(mgr);
  return CUDA_SUCCESS;
}
CUresult dark_ctx_storage_get(void** out, CUcontext ctx, void* mgr) {
  if (!out) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(g_ctx_storage_mu);
  auto m = g_ctx_storage.find(mgr);
  if (m != g_ctx_storage.end()) {
    auto it = m->second.find(reinterpret_cast<uintptr_t>(ctx));
    if (it != m->second.end()) {
      *out = it->second.state;
      return CUDA_SUCCESS;
    }
  }
  *out = nullptr;
  return CUDA_ERROR_INVALID_VALUE;
}

// Tools-hooks slots 2 and 6 hand cudart scratch buffers.
unsigned char g_tools_buf[4096];
CUresult dark_tools_buffer(void** buf, size_t* size) {
  if (buf) *buf = g_tools_buf;
  if (size) *size = sizeof g_tools_buf;
  return CUDA_SUCCESS;
}

template <int Table, size_t N, size_t... Is>
void fill_stubs(void* (&table)[N], std::index_sequence<Is...>) {
  ((table[Is] = reinterpret_cast<void*>(&dark_stub<Table, static_cast<int>(Is)>)), ...);
}

void* g_cudart_interface_table[16];
void* g_tools_hooks_table[16];
void* g_ctx_storage_table[4];
constexpr int kMaxGenericTables = 8;
void* g_generic_tables[kMaxGenericTables][16];
unsigned char g_generic_uuids[kMaxGenericTables][16];
int g_generic_count = 0;

// VGPU_DARK_DENY: refuse specific export tables, by UUID hex prefix, or "all".
// The dark API has no specification to implement against, so the only way to
// learn which table a caller actually depends on is to withhold one and see
// what changes. This is that experiment, kept because the question recurs
// whenever a new toolkit version bootstraps differently.
bool dark_denied(const unsigned char* uuid) {
  const char* deny = std::getenv("VGPU_DARK_DENY");
  if (!deny || !deny[0]) return false;
  char hex[33];
  for (int i = 0; i < 16; ++i) std::snprintf(hex + 2 * i, 3, "%02x", uuid[i]);
  std::string list(deny);
  if (list == "all") return true;
  size_t start = 0;
  while (start <= list.size()) {
    size_t comma = list.find(',', start);
    std::string item = list.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    if (!item.empty() && std::strncmp(hex, item.c_str(), item.size()) == 0) return true;
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return false;
}

const void* dark_table_for(const unsigned char* uuid) {
  if (dark_denied(uuid)) return nullptr;
  static bool init = false;
  if (!init) {
    init = true;
    fill_stubs<1>(g_cudart_interface_table, std::make_index_sequence<16>{});
    g_cudart_interface_table[0] = reinterpret_cast<void*>(sizeof g_cudart_interface_table);
    g_cudart_interface_table[1] = reinterpret_cast<void*>(&dark_get_module);
    g_cudart_interface_table[6] = reinterpret_cast<void*>(&dark_get_module_ext1);
    g_cudart_interface_table[8] = reinterpret_cast<void*>(&dark_get_module_ext2);
    fill_stubs<2>(g_tools_hooks_table, std::make_index_sequence<16>{});
    g_tools_hooks_table[0] = reinterpret_cast<void*>(sizeof g_tools_hooks_table);
    g_tools_hooks_table[2] = reinterpret_cast<void*>(&dark_tools_buffer);
    g_tools_hooks_table[6] = reinterpret_cast<void*>(&dark_tools_buffer);
    g_ctx_storage_table[0] = reinterpret_cast<void*>(&dark_ctx_storage_ctor);
    g_ctx_storage_table[1] = reinterpret_cast<void*>(&dark_ctx_storage_dtor);
    g_ctx_storage_table[2] = reinterpret_cast<void*>(&dark_ctx_storage_get);
    g_ctx_storage_table[3] = nullptr;
  }
  if (std::memcmp(uuid, kUuidCudartInterface, 16) == 0) return g_cudart_interface_table;
  if (std::memcmp(uuid, kUuidToolsHooks, 16) == 0) return g_tools_hooks_table;
  if (std::memcmp(uuid, kUuidCtxStorage, 16) == 0) return g_ctx_storage_table;
  for (int i = 0; i < g_generic_count; ++i)
    if (std::memcmp(uuid, g_generic_uuids[i], 16) == 0) return g_generic_tables[i];
  if (g_generic_count < kMaxGenericTables) {
    int i = g_generic_count++;
    std::memcpy(g_generic_uuids[i], uuid, 16);
    switch (i) {  // distinct stub families so the trace names the table
      case 0: fill_stubs<10>(g_generic_tables[i], std::make_index_sequence<16>{}); break;
      case 1: fill_stubs<11>(g_generic_tables[i], std::make_index_sequence<16>{}); break;
      case 2: fill_stubs<12>(g_generic_tables[i], std::make_index_sequence<16>{}); break;
      case 3: fill_stubs<13>(g_generic_tables[i], std::make_index_sequence<16>{}); break;
      case 4: fill_stubs<14>(g_generic_tables[i], std::make_index_sequence<16>{}); break;
      case 5: fill_stubs<15>(g_generic_tables[i], std::make_index_sequence<16>{}); break;
      case 6: fill_stubs<16>(g_generic_tables[i], std::make_index_sequence<16>{}); break;
      default: fill_stubs<17>(g_generic_tables[i], std::make_index_sequence<16>{}); break;
    }
    g_generic_tables[i][0] = reinterpret_cast<void*>(sizeof g_generic_tables[i]);
    return g_generic_tables[i];
  }
  return nullptr;
}

}  // namespace

/* ---- interprocess memory ----
   Device memory here is a per-process virtual address space with per-process
   backing, so a handle from one process names nothing in another. Saying so is
   the honest answer, and the runtime API's cudaIpc* family says the same.

   They have to exist even so: a caller that looks the symbols up at startup --
   Numba resolves cuIpcOpenMemHandle before it will report a device at all --
   fails on the lookup rather than on the call, which reads as "no CUDA here"
   instead of "no interprocess sharing here". */

VGPU_EXPORT CUresult cuIpcGetMemHandle(CUipcMemHandle*, CUdeviceptr) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUresult cuIpcOpenMemHandle(CUdeviceptr*, CUipcMemHandle, unsigned int) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUresult cuIpcOpenMemHandle_v2(CUdeviceptr*, CUipcMemHandle, unsigned int) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUresult cuIpcCloseMemHandle(CUdeviceptr) { return CUDA_ERROR_NOT_SUPPORTED; }
VGPU_EXPORT CUresult cuIpcGetEventHandle(CUipcEventHandle*, CUevent) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT CUresult cuIpcOpenEventHandle(CUevent*, CUipcEventHandle) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

VGPU_EXPORT CUresult cuGetExportTable(const void** table, const void* uuid) {
  if (!table || !uuid) return CUDA_ERROR_INVALID_VALUE;
  // Only a statically linked CUDA runtime asks for these: the shared runtime
  // never touches them, because VirtualGPU's libcudart answers instead. A
  // static runtime is NVIDIA's own, living inside the binary and reaching the
  // driver through this undocumented table -- it gets a little further and
  // then refuses with cudaErrorSoftwareValidityNotEstablished, which reaches
  // the program as "integrity checks failed" on its first CUDA call and
  // explains nothing. Say what is happening once, while there is still time
  // for it to be useful.
  static std::once_flag warned;
  std::call_once(warned, [] {
    if (std::getenv("VGPU_QUIET") && std::getenv("VGPU_QUIET")[0] == '1') return;
    // Two very different callers reach here, and telling someone to rebuild
    // their program when it was a profiler asking sends them somewhere useless.
    std::fprintf(stderr,
                 "[vgpu] something asked for a driver export table, which is NVIDIA's "
                 "undocumented internal interface.\n"
                 "       If this is your program: it links the CUDA runtime statically, which "
                 "cannot run on a simulated driver. Rebuild with\n"
                 "       'nvcc -cudart shared', or build inside 'vgpu shell', which supplies an "
                 "nvcc that adds it. Without that the next\n"
                 "       CUDA call fails with error 103, \"integrity checks failed\".\n"
                 "       If this is a profiler: Nsight Systems collects through its own bundled "
                 "CUPTI, which needs these tables.\n"
                 "       nvprof works instead -- see nvidia/docs/cupti.md.\n");
  });
  const unsigned char* u = static_cast<const unsigned char*>(uuid);
  const void* t = dark_table_for(u);
  if (trace()) {
    std::fprintf(stderr, "[vgpu][trace] cuGetExportTable uuid=");
    for (int i = 0; i < 16; ++i) std::fprintf(stderr, "%02x", u[i]);
    std::fprintf(stderr, " -> %s\n", t ? "table" : "NOT_SUPPORTED");
  }
  *table = t;
  return t ? CUDA_SUCCESS : CUDA_ERROR_NOT_SUPPORTED;
}

/* ---- cuGetProcAddress: how modern cudart resolves everything ---- */

namespace {

struct ProcEntry {
  const char* name;
  void* fn;
};

#define VGPU_PROC(name) {#name, reinterpret_cast<void*>(&name)}

const ProcEntry kProcTable[] = {
    VGPU_PROC(cuInit), VGPU_PROC(cuDriverGetVersion), VGPU_PROC(cuGetErrorName),
    VGPU_PROC(cuGetErrorString), VGPU_PROC(cuDeviceGetCount), VGPU_PROC(cuDeviceGet),
    VGPU_PROC(cuDeviceGetName), VGPU_PROC(cuDeviceTotalMem), VGPU_PROC(cuDeviceGetAttribute),
    VGPU_PROC(cuDeviceComputeCapability), VGPU_PROC(cuDeviceGetUuid), VGPU_PROC(cuDeviceGetUuid_v2),
    VGPU_PROC(cuDeviceGetPCIBusId), VGPU_PROC(cuDeviceCanAccessPeer),
    VGPU_PROC(cuCtxCreate), VGPU_PROC(cuCtxDestroy), VGPU_PROC(cuCtxSetCurrent),
    VGPU_PROC(cuCtxGetCurrent), VGPU_PROC(cuCtxGetDevice), VGPU_PROC(cuCtxSynchronize),
    VGPU_PROC(cuCtxPushCurrent), VGPU_PROC(cuCtxPopCurrent), VGPU_PROC(cuCtxGetLimit),
    VGPU_PROC(cuCtxSetLimit), VGPU_PROC(cuCtxGetApiVersion), VGPU_PROC(cuCtxGetStreamPriorityRange),
    VGPU_PROC(cuCtxGetFlags), VGPU_PROC(cuCtxSetFlags), VGPU_PROC(cuCtxGetId),
    VGPU_PROC(cuDevicePrimaryCtxRetain), VGPU_PROC(cuDevicePrimaryCtxRelease),
    VGPU_PROC(cuDevicePrimaryCtxSetFlags), VGPU_PROC(cuDevicePrimaryCtxGetState),
    VGPU_PROC(cuDevicePrimaryCtxReset_v2),
    VGPU_PROC(cuMemAlloc), VGPU_PROC(cuMemFree), VGPU_PROC(cuMemcpyHtoD), VGPU_PROC(cuMemcpyDtoH),
    VGPU_PROC(cuMemcpyDtoD), VGPU_PROC(cuMemcpy), VGPU_PROC(cuMemcpyAsync),
    VGPU_PROC(cuMemcpyHtoDAsync), VGPU_PROC(cuMemGetInfo), VGPU_PROC(cuMemGetAddressRange_v2),
    VGPU_PROC(cuMemHostAlloc), VGPU_PROC(cuMemAllocHost_v2), VGPU_PROC(cuMemFreeHost),
    VGPU_PROC(cuPointerGetAttribute),
    VGPU_PROC(cuModuleLoadData), VGPU_PROC(cuModuleLoadDataEx), VGPU_PROC(cuModuleUnload),
    VGPU_PROC(cuModuleGetFunction), VGPU_PROC(cuModuleGetLoadingMode),
    VGPU_PROC(cuLibraryLoadData), VGPU_PROC(cuLibraryUnload), VGPU_PROC(cuLibraryGetKernel),
    VGPU_PROC(cuLibraryGetModule), VGPU_PROC(cuKernelGetFunction), VGPU_PROC(cuKernelGetName),
    VGPU_PROC(cuKernelGetAttribute), VGPU_PROC(cuKernelSetAttribute),
    VGPU_PROC(cuFuncGetAttribute), VGPU_PROC(cuFuncSetAttribute), VGPU_PROC(cuFuncSetCacheConfig),
    VGPU_PROC(cuFuncIsLoaded), VGPU_PROC(cuFuncLoad),
    VGPU_PROC(cuLaunchKernel), VGPU_PROC(cuLaunchKernelEx),
    VGPU_PROC(cuMemsetD8Async), VGPU_PROC(cuMemsetD16Async), VGPU_PROC(cuMemsetD32Async),
    VGPU_PROC(cuStreamCreate), VGPU_PROC(cuStreamCreateWithPriority), VGPU_PROC(cuStreamDestroy),
    VGPU_PROC(cuStreamSynchronize), VGPU_PROC(cuStreamQuery), VGPU_PROC(cuStreamWaitEvent),
    VGPU_PROC(cuStreamGetPriority), VGPU_PROC(cuStreamGetFlags),
    VGPU_PROC(cuStreamGetCaptureInfo_v2), VGPU_PROC(cuStreamIsCapturing),
    VGPU_PROC(cuEventCreate), VGPU_PROC(cuEventRecord), VGPU_PROC(cuEventQuery),
    VGPU_PROC(cuEventSynchronize), VGPU_PROC(cuEventDestroy), VGPU_PROC(cuEventElapsedTime),
    VGPU_PROC(cuOccupancyMaxActiveBlocksPerMultiprocessor),
    VGPU_PROC(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags),
    VGPU_PROC(cuGetExportTable),
};

// Versioned/suffixed request names resolve to the same synchronous impls:
// strip _v2/_v3 and _ptsz/_ptds suffixes when looking up.
void* find_proc(const std::string& request) {
  std::string base = request;
  for (const char* suf : {"_ptsz", "_ptds", "_v3", "_v2"}) {
    size_t l = std::strlen(suf);
    if (base.size() > l && base.compare(base.size() - l, l, suf) == 0)
      base = base.substr(0, base.size() - l);
  }
  for (const auto& e : kProcTable) {
    if (base == e.name) return e.fn;
    // Table entries themselves may carry _v2 (e.g. cuMemGetAddressRange_v2).
    std::string en = e.name;
    for (const char* suf : {"_v3", "_v2"}) {
      size_t l = std::strlen(suf);
      if (en.size() > l && en.compare(en.size() - l, l, suf) == 0) en = en.substr(0, en.size() - l);
    }
    if (base == en) return e.fn;
  }
  return nullptr;
}

}  // namespace

VGPU_EXPORT CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion,
                                      unsigned long long flags);

VGPU_EXPORT CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion,
                                         unsigned long long flags, int* symbolStatus) {
  if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;
  (void)flags;
  void* fn = nullptr;
  // Bootstrap: cudart resolves cuGetProcAddress through itself, and the
  // requested cudaVersion selects which signature it expects.
  if (std::strcmp(symbol, "cuGetProcAddress") == 0)
    fn = cudaVersion >= 12000 ? reinterpret_cast<void*>(&cuGetProcAddress_v2)
                              : reinterpret_cast<void*>(&cuGetProcAddress);
  if (!fn) fn = find_proc(symbol);
  if (trace())
    std::fprintf(stderr, "[vgpu][trace] cuGetProcAddress(\"%s\", v%d) -> %s\n", symbol, cudaVersion,
                 fn ? "ok" : "MISSING");
  *pfn = fn;
  if (symbolStatus) *symbolStatus = fn ? 0 : 1;  // SUCCESS : SYMBOL_NOT_FOUND
  return fn ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
}

VGPU_EXPORT CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion,
                                      unsigned long long flags) {
  return cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, nullptr);
}
