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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iterator>
#include <sstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <ctime>
#include <map>
#include <set>

#include "error_names.hpp"
#include "fatbin.hpp"
#include "vgpu/sass/cubin.hpp"
#include "vgpu/sass/exec.hpp"
#include "vgpu/driver_version.hpp"
#include "vgpu/error.hpp"
#include "vgpu/exec/tensormap.hpp"
#include "vgpu/exec/texture.hpp"
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
  bool nonportable_cluster = false;   // CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED
  int max_dynamic_shared = -1;        // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, -1 until set
  // The launch the pre-CUDA 4 API builds up on the function itself
  // (cuFuncSetBlockShape, cuFuncSetSharedSize, cuParamSet*) and cuLaunchGrid
  // runs: a block of one thread, no shared memory and no parameters until
  // told otherwise, as on the card.
  unsigned block[3] = {1, 1, 1};
  unsigned shared_bytes = 0;
  std::vector<uint8_t> params;   // the parameter buffer, as written
  size_t param_size = 0;         // cuParamSetSize
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
  bool timing = true;   // false for CU_EVENT_DISABLE_TIMING
  struct timespec when {};
};

// A CUDA array: its descriptor and the device memory behind it, laid out row
// by row (Width elements of NumChannels, then Height rows, then Depth slices;
// a zero Height or Depth counts as one).
struct ArrayRec {
  CUDA_ARRAY3D_DESCRIPTOR desc{};
  size_t elem = 0;       // bytes an element takes, all channels
  size_t row = 0;        // bytes a row takes
  size_t rows = 1, slices = 1;
  int device = 0;
  CUdeviceptr mem = 0;
  bool mip_level = false;   // a level of a mipmapped array, which owns it
};

// A mipmapped array: one ordinary array per level, which is what
// cuMipmappedArrayGetLevel hands out (the same handle each time).
struct MipmapRec {
  std::vector<uintptr_t> levels;
};

// A texture reference made with cuTexRefCreate: the state the cuTexRefSet*
// calls record and the cuTexRefGet* calls read back. Nothing launches with
// it -- a module's texture references are what a kernel reads, and CUDA 12
// compilers no longer emit them (see cuModuleGetTexRef) -- but a program that
// sets one up is told what the card tells it. Defaults as an RTX 3060 reports
// them: clamp addressing, point filtering, one float channel.
struct TexRefRec {
  int address[3] = {1, 1, 1};   // CU_TR_ADDRESS_MODE_CLAMP
  int filter = 0;                // CU_TR_FILTER_MODE_POINT
  unsigned flags = 0;
  int format = 0x20;             // CU_AD_FORMAT_FLOAT
  int channels = 1;
  CUdeviceptr address_base = 0;  // bound linear memory, or 0
  CUarray array = nullptr;       // bound array, or null
};

struct ShimState {
  // One lock for both CUDA libraries, since they share one machine
  // (shared_runtime.cpp).
  std::recursive_mutex& mu = vgpu::runtime::shared_api_mutex();
  bool initialized = false;
  vgpu::runtime::Runtime* rt = nullptr;   // the process's machine, not owned (shared_runtime.cpp)
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
  std::map<uintptr_t, size_t> managed;  // cuMemAllocManaged results: base -> bytes
  // The memory a loaded module's __managed__ globals were moved to, by (device,
  // module id): entries in `managed` too, freed when the module is unloaded.
  std::map<std::pair<int, uint64_t>, std::vector<uintptr_t>> module_managed;
  std::unordered_map<uintptr_t, ArrayRec> arrays;  // cuArray3DCreate results
  std::unordered_map<uintptr_t, MipmapRec> mipmaps;  // cuMipmappedArrayCreate results
  // Mipmapped arrays destroyed, and the level arrays that went with them: the
  // card answers a handle from either with CUDA_ERROR_CONTEXT_IS_DESTROYED.
  std::set<uintptr_t> retired;
  std::unordered_map<uintptr_t, TexRefRec> texrefs;  // cuTexRefCreate results
  // Peer access enabled, as (context, peer context): cuCtxEnablePeerAccess
  // is about contexts, and enabling it one way says nothing of the other.
  std::set<std::pair<uintptr_t, uintptr_t>> peer_access;
  std::unordered_map<uintptr_t, int> attached;   // cuCtxAttach references, by context
  int cache_config = 0;   // cuCtxSetCacheConfig: CU_FUNC_CACHE_PREFER_NONE until set
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
      // Inside a kernel these are the moral equivalent of a device-side fault.
      return kernel_context ? CUDA_ERROR_ILLEGAL_ADDRESS : CUDA_ERROR_INVALID_VALUE;
    // A kernel's misaligned access has a code of its own on hardware.
    case Err::MisalignedAccess:
      return kernel_context ? CUDA_ERROR_MISALIGNED_ADDRESS : CUDA_ERROR_INVALID_VALUE;
    case Err::UninitializedRegister: return CUDA_ERROR_ILLEGAL_ADDRESS;
    // Both of these reached the switch's fallthrough before, and so were
    // reported as CUDA_ERROR_UNKNOWN -- the least informative code available,
    // for the two conditions this project most wants to be legible.
    // An RTX 3060 reports a kernel's "trap" (and "brkpt") as LAUNCH_FAILED.
    case Err::Trap: return CUDA_ERROR_LAUNCH_FAILED;
    case Err::DeviceAssert: return CUDA_ERROR_ASSERT;
    case Err::EccUncorrectable: return CUDA_ERROR_ECC_UNCORRECTABLE;
    // What programs report when their GPU falls off the bus.
    case Err::DeviceLost: return CUDA_ERROR_LAUNCH_FAILED;
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

// Whether an error leaves the context unusable: a kernel's illegal address,
// trap or failed assert, as the runtime decides it (runtime_api.cpp's
// poisons_context), and uncorrectable memory whatever read it.
bool poisons_context(vgpu::Err e, bool kernel_context) {
  using vgpu::Err;
  switch (e) {
    case Err::InvalidPointer:
    case Err::UseAfterFree:
    case Err::OutOfBounds:
    case Err::MisalignedAccess:
    case Err::UninitializedRegister:
    case Err::Trap:
    case Err::DeviceAssert:
    case Err::DeviceLost:
      return kernel_context;
    case Err::EccUncorrectable:
      return true;
    default:
      return false;
  }
}

// The calls an RTX 3060 still answers once a kernel has killed the context:
// what a device is, which context is current, and the primary context's own
// bookkeeping. Every other call that needs the context fails with the fault
// (Runtime::context_fault), the next cuCtxSynchronize first among them.
bool answers_dead_context(const char* name) {
  static const char* const kLive[] = {"cuInit", "cuDevice", "cuCtxGetCurrent", "cuCtxSetCurrent",
                                      "cuCtxPushCurrent", "cuCtxPopCurrent", "cuCtxGetDevice",
                                      "cuCtxCreate", "cuCtxDestroy", "cuCtxAttach", "cuTensorMap"};
  for (const char* p : kLive)
    if (std::strncmp(name, p, std::strlen(p)) == 0) return true;
  return false;
}

// The fault a dead context answers with, or CUDA_SUCCESS. For the entry points
// that do not go through api().
CUresult dead_context() {
  ShimState& s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  return s.initialized ? static_cast<CUresult>(s.rt->context_fault()) : CUDA_SUCCESS;
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
  if (s.initialized && !answers_dead_context(name))
    if (const int fault = s.rt->context_fault()) return static_cast<CUresult>(fault);
  try {
    CUresult r = body(s);
    if (trace_calls()) std::fprintf(stderr, "[vgpu][call] %s -> %d\n", name, static_cast<int>(r));
    return r;
  } catch (const vgpu::Error& e) {
    report(name, e.what());
    CUresult r = map_error(e, kernel_context);
    if (s.initialized && poisons_context(e.code(), kernel_context)) {
      s.rt->set_context_fault(static_cast<int>(r));
      // A kernel's fault is reported by the calls after its launch, which on
      // the card returned before the kernel ran: cuLaunchKernel succeeds, and
      // the next cuCtxSynchronize is CUDA_ERROR_ASSERT or ILLEGAL_ADDRESS.
      if (kernel_context) r = CUDA_SUCCESS;
    }
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

// The managed allocation holding [p, p + n), or nullptr.
const std::pair<const uintptr_t, size_t>* managed_range(ShimState& s, CUdeviceptr p, size_t n) {
  auto it = s.managed.upper_bound(p);
  if (it == s.managed.begin()) return nullptr;
  --it;
  if (p + n > it->first + it->second || p + n < p) return nullptr;
  return &*it;
}

// Reads, writes and fills of device memory. Managed memory is host memory the
// devices map, and copies and memsets reach it directly, as the runtime's do.
void dev_write(ShimState& s, CUdeviceptr dst, const void* src, size_t n) {
  if (n && managed_range(s, dst, n)) {
    std::memcpy(reinterpret_cast<void*>(dst), src, n);
    return;
  }
  owner_memory(s, dst).write(dst, src, n);
}
void dev_read(ShimState& s, void* dst, CUdeviceptr src, size_t n) {
  if (n && managed_range(s, src, n)) {
    std::memcpy(dst, reinterpret_cast<const void*>(src), n);
    return;
  }
  owner_memory(s, src).read(src, dst, n);
}
void dev_fill(ShimState& s, CUdeviceptr dst, const uint8_t* pattern, size_t width, size_t n) {
  if (n && managed_range(s, dst, n)) {
    auto* out = reinterpret_cast<uint8_t*>(dst);
    for (size_t i = 0; i < n; i += width) std::memcpy(out + i, pattern, std::min(width, n - i));
    return;
  }
  owner_memory(s, dst).fill(dst, pattern, width, n);
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

// Picks the PTX image the driver would JIT -- see pick_ptx for the rule.
std::string best_ptx(const void* image) {
  auto ptxs = vgpu::cuda::extract_ptx(image);
  if (ptxs.empty())
    throw vgpu::Error::make(vgpu::Err::Unsupported,
                            "fatbin contains no PTX image (SASS-only fatbin?); rebuild with an "
                            "-arch=sm_XX that embeds PTX, or add -gencode arch=compute_XX,"
                            "code=compute_XX");
  uint32_t cc = ~0u;   // every device of a simulated machine has one profile
  if (ShimState& s = state(); s.rt && s.rt->device_count() > 0) {
    const vgpu::DeviceProfile& p = s.rt->device(0).profile();
    cc = static_cast<uint32_t>(p.cc_major * 10 + p.cc_minor);
  }
  return std::move(ptxs[vgpu::cuda::pick_ptx(ptxs, cc)].text);
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
    // Managed memory is host memory every device maps (cuMemAllocManaged), so
    // the host and the devices may use it at once, as on a Linux machine with
    // a Pascal or newer GPU. The runtime answers the same.
    case 83: return 1;                               // MANAGED_MEMORY
    case 89: return 1;                               // CONCURRENT_MANAGED_ACCESS
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
    // VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED: cuMemAddressReserve, cuMemCreate,
    // cuMemMap and cuMemSetAccess work (nvidia/tests/e2e/vmm.cu), and callers
    // gate on this before using them -- CUDA's own vectorAddMMAP sample, and
    // allocators that grow a buffer in place. An RTX 3060 answers 1.
    case 102: return 1;
    // A real quantity, and answering zero for it is what had a CUB scan launch
    // no blocks: it is a divisor in occupancy arithmetic. 106, not 134.
    case 106: return static_cast<int>(p.limits.max_blocks_per_sm);
    case 107: return 0;                              // GENERIC_COMPRESSION_SUPPORTED
    case 110: return 0;                              // GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED
    case 111: return static_cast<int>(p.reserved_smem_per_block());  // RESERVED_SHARED_MEMORY_PER_BLOCK

    // ---- capabilities this deliberately does not implement ----
    // Zero is the true answer for each, and saying so explicitly keeps them out
    // of the "not modelled" report below: a caller that asks whether managed
    // memory works needs a truthful no, not a warning.
    case 86: return 0;                               // HOST_NATIVE_ATOMIC_SUPPORTED
    case 88: return 0;                               // PAGEABLE_MEMORY_ACCESS
    case 91: return 0;                               // CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM
    // COOPERATIVE_LAUNCH: cudaLaunchCooperativeKernel works, because the
    // scheduler can hold every block resident and interleave them. The
    // multi-device form (96) needs grids on separate devices waiting on each
    // other, which it cannot.
    case 95: return 1;
    case 96: return 0;                               // COOPERATIVE_MULTI_DEVICE_LAUNCH
    case 100: return 0;                              // PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
    case 101: return 0;                              // DIRECT_MANAGED_MEM_ACCESS_FROM_HOST
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
    case 125: return 1;                              // IPC_EVENT_SUPPORTED (cuIpc* above)
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
// A loaded module's __managed__ globals, moved onto managed memory: host memory
// every device maps, at one address for the host and every device, holding the
// variable's initial value -- as the runtime moves a registered __managed__
// variable (runtime_api.cpp's bind_managed_vars). It used to stay ordinary
// device memory, which the host could not touch. On an RTX 3060 a module loaded
// with cuModuleLoadData has its managed global at the address cuModuleGetGlobal
// returns, readable and writable from the host, reported as managed, and a
// kernel's writes to it visible there after cuCtxSynchronize.
void bind_managed_globals(ShimState& s, int dev, uint64_t mid) {
  vgpu::runtime::Device& d = s.rt->device(dev);
  std::vector<uintptr_t> storage;
  for (const std::string& name : d.managed_globals(mid)) {
    uint64_t at = 0, size = 0;
    if (!d.global(mid, name, &at, &size)) continue;
    const size_t n = std::max<size_t>(size, 1);
    void* p = std::aligned_alloc(4096, (n + 4095) / 4096 * 4096);
    if (!p) throw vgpu::Error::make(vgpu::Err::OutOfMemory, "__managed__ '", name, "'");
    d.memory().read(at, p, size);   // the initial value
    for (int o = 0; o < s.rt->device_count(); ++o)
      s.rt->device(o).memory().map_host(reinterpret_cast<uint64_t>(p), p, n);
    s.managed[reinterpret_cast<uintptr_t>(p)] = n;
    d.rebind_global(mid, name, reinterpret_cast<uint64_t>(p));
    storage.push_back(reinterpret_cast<uintptr_t>(p));
  }
  if (!storage.empty()) s.module_managed[{dev, mid}] = std::move(storage);
}

// Unloads a module, and frees what its managed globals were moved to.
void unload_module(ShimState& s, int dev, uint64_t mid) {
  s.rt->device(dev).unload_module(mid);
  const auto it = s.module_managed.find({dev, mid});
  if (it == s.module_managed.end()) return;
  for (uintptr_t p : it->second) {
    for (int o = 0; o < s.rt->device_count(); ++o) s.rt->device(o).memory().unmap_host(p);
    s.managed.erase(p);
    std::free(reinterpret_cast<void*>(p));
  }
  s.module_managed.erase(it);
}

uint64_t library_module_on(ShimState& s, uintptr_t lib_handle, int dev) {
  auto it = s.libraries.find(lib_handle);
  if (it == s.libraries.end())
    throw vgpu::Error::make(vgpu::Err::InvalidValue, "invalid library handle");
  LibRec& lib = it->second;
  auto mit = lib.per_device_module.find(dev);
  if (mit != lib.per_device_module.end()) return mit->second;
  uint64_t mid = s.rt->device(dev).load_module(lib.ptx);
  lib.per_device_module[dev] = mid;
  bind_managed_globals(s, dev, mid);
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
    // The machine libcudart may already have made (shared_runtime.cpp).
    s.rt = vgpu::runtime::shared_runtime();
    s.initialized = true;
    vgpu::load_injection_library();
    return CUDA_SUCCESS;
  });
}

// What libcudart does for the driver when a program uses both APIs, as the
// real runtime does through the real driver: the device's primary context,
// retained once on the runtime's behalf, made current for the calling thread,
// so the program's own driver calls find a context there. A context the
// program made current itself is left alone; a primary one is swapped for the
// runtime's current device's. Called by the runtime shim, found through the
// process's global scope (runtime_api.cpp).
VGPU_EXPORT int vgpu_driver_bind_primary_v1(int dev) {
  if (cuInit(0) != CUDA_SUCCESS) return CUDA_ERROR_NOT_INITIALIZED;
  return api("vgpu_driver_bind_primary", true, false, [&](ShimState& s) {
    check_device(s, dev);
    auto it = s.primary_ctx.find(dev);
    if (it == s.primary_ctx.end()) {
      const uintptr_t h = make_handle(s, kTagCtx);
      s.contexts[h] = dev;
      it = s.primary_ctx.emplace(dev, h).first;
    }
    static std::set<int> retained;   // the runtime's one reference per device
    if (retained.insert(dev).second) ++s.primary_refs[dev];
    auto& stack = ctx_stack();
    bool primary_current = false;
    if (!stack.empty())
      for (const auto& [d, h] : s.primary_ctx) primary_current |= h == stack.back();
    if (stack.empty()) stack.push_back(it->second);
    else if (primary_current) stack.back() = it->second;
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

// The deprecated CUdevprop: the same limits cuDeviceGetAttribute reports.
VGPU_EXPORT CUresult cuDeviceGetProperties(CUdevprop* prop, CUdevice dev) {
  return api("cuDeviceGetProperties", true, false, [&](ShimState& s) {
    if (!prop) return CUDA_ERROR_INVALID_VALUE;
    if (dev < 0 || dev >= s.rt->device_count()) return CUDA_ERROR_INVALID_DEVICE;
    const vgpu::DeviceProfile& p = s.rt->device(dev).profile();
    prop->maxThreadsPerBlock = static_cast<int>(p.limits.max_threads_per_block);
    for (int i = 0; i < 3; ++i) {
      prop->maxThreadsDim[i] = static_cast<int>(p.limits.max_block_dim[i]);
      prop->maxGridSize[i] = static_cast<int>(p.limits.max_grid_dim[i]);
    }
    prop->sharedMemPerBlock = static_cast<int>(p.limits.shared_mem_per_block);
    prop->totalConstantMemory = 65536;
    prop->SIMDWidth = static_cast<int>(p.warp_size);
    prop->memPitch = extra_attribute(p, 11);      // MAX_PITCH
    prop->regsPerBlock = static_cast<int>(p.limits.registers_per_block);
    prop->clockRate = 1000000;                    // the placeholder CLOCK_RATE reports
    prop->textureAlign = extra_attribute(p, 14);  // TEXTURE_ALIGNMENT
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
// never calls is no help. Affinity and CIG parameters are not supported, and
// say so rather than being ignored; a block that asks for neither is an
// ordinary context. That is what CUDA 13's own samples pass (a zeroed
// CUctxCreateParams), and the card takes it.
VGPU_EXPORT CUresult cuCtxCreate_v3(CUcontext* pctx, void* exec_affinity_params, int num_params,
                                    unsigned int flags, CUdevice dev) {
  if (exec_affinity_params && num_params > 0) return CUDA_ERROR_NOT_SUPPORTED;
  return cuCtxCreate_v2(pctx, flags, dev);
}
VGPU_EXPORT CUresult cuCtxCreate_v4(CUcontext* pctx, void* ctx_create_params, unsigned int flags,
                                    CUdevice dev) {
  // CUctxCreateParams: execAffinityParams, numExecAffinityParams, cigParams.
  struct CreateParamsABI {
    void* exec_affinity;
    int num_exec_affinity;
    void* cig;
  };
  if (const auto* p = static_cast<const CreateParamsABI*>(ctx_create_params);
      p && ((p->exec_affinity && p->num_exec_affinity > 0) || p->cig))
    return CUDA_ERROR_NOT_SUPPORTED;
  return cuCtxCreate_v2(pctx, flags, dev);
}

VGPU_EXPORT CUresult cuCtxDestroy_v2(CUcontext ctx) {
  return api("cuCtxDestroy", true, false, [&](ShimState& s) {
    uintptr_t h = check_handle(reinterpret_cast<uintptr_t>(ctx), kTagCtx, "context");
    if (!s.contexts.erase(h)) return CUDA_ERROR_INVALID_CONTEXT;
    std::erase(ctx_stack(), h);
    std::erase_if(s.peer_access, [h](const auto& p) { return p.first == h || p.second == h; });
    s.attached.erase(h);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuCtxDestroy(CUcontext ctx) { return cuCtxDestroy_v2(ctx); }

// The deprecated usage count: cuCtxAttach takes another reference to the
// current context and returns it (flags must be 0), cuCtxDetach drops one,
// and the context goes when its creator's reference does -- one detach more
// than there were attaches.
VGPU_EXPORT CUresult cuCtxAttach(CUcontext* pctx, unsigned int flags) {
  return api("cuCtxAttach", true, false, [&](ShimState& s) {
    if (!pctx || flags != 0) return CUDA_ERROR_INVALID_VALUE;
    if (ctx_stack().empty()) return CUDA_ERROR_INVALID_CONTEXT;
    (void)current_device(s);
    ++s.attached[ctx_stack().back()];
    *pctx = reinterpret_cast<CUcontext>(ctx_stack().back());
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuCtxDetach(CUcontext ctx) {
  {
    ShimState& s = state();
    std::lock_guard<std::recursive_mutex> lock(s.mu);
    auto it = s.attached.find(reinterpret_cast<uintptr_t>(ctx));
    if (it != s.attached.end() && it->second > 0) {
      if (--it->second == 0) s.attached.erase(it);
      return CUDA_SUCCESS;
    }
  }
  return cuCtxDestroy_v2(ctx);
}

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
    if (auto it = s.managed.find(dptr); it != s.managed.end()) {
      for (int d = 0; d < s.rt->device_count(); ++d) s.rt->device(d).memory().unmap_host(dptr);
      std::free(reinterpret_cast<void*>(dptr));
      s.managed.erase(it);
      return CUDA_SUCCESS;
    }
    // The device heap's blocks are not the host's to free (as cudaFree).
    if (owner_memory(s, dptr).heap_contains(dptr)) return CUDA_ERROR_INVALID_VALUE;
    owner_memory(s, dptr).free(dptr);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemFree(CUdeviceptr dptr) { return cuMemFree_v2(dptr); }

VGPU_EXPORT CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
  return api("cuMemcpyHtoD", true, false, [&](ShimState& s) {
    if (!srcHost && ByteCount) return CUDA_ERROR_INVALID_VALUE;
    dev_write(s, dstDevice, srcHost, ByteCount);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemcpyHtoD(CUdeviceptr d, const void* h, size_t n) {
  return cuMemcpyHtoD_v2(d, h, n);
}

VGPU_EXPORT CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  return api("cuMemcpyDtoH", true, false, [&](ShimState& s) {
    if (!dstHost && ByteCount) return CUDA_ERROR_INVALID_VALUE;
    dev_read(s, dstHost, srcDevice, ByteCount);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemcpyDtoH(void* h, CUdeviceptr d, size_t n) {
  return cuMemcpyDtoH_v2(h, d, n);
}

VGPU_EXPORT CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
  return api("cuMemcpyDtoD", true, false, [&](ShimState& s) {
    std::vector<uint8_t> tmp(ByteCount);
    dev_read(s, tmp.data(), srcDevice, ByteCount);
    dev_write(s, dstDevice, tmp.data(), ByteCount);
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
    int dev = current_device(s);
    uint64_t mid = 0;
    const vgpu::DeviceProfile& prof = s.rt->device(dev).profile();
    const uint32_t cc = static_cast<uint32_t>(prof.cc_major * 10 + prof.cc_minor);
    if (magic == 0x466243B1u || magic == 0xBA55ED50u) {
      // A fatbin (wrapper or container): its SASS for this GPU if it has
      // some, as the real driver runs; else its PTX.
      const std::string cubin = vgpu::cuda::pick_cubin(image, cc);
      if (!cubin.empty()) {
        mid = s.rt->device(dev).load_cubin(reinterpret_cast<const uint8_t*>(cubin.data()), cubin.size());
      } else {
        extracted = best_ptx(image);
        text = extracted.c_str();
        mid = s.rt->device(dev).load_module(text);
      }
    } else if (magic == 0x464c457fu) {
      // An ELF image: a cubin. The first eight bytes say whether it is a
      // 64-bit CUDA one before anything further is read from a pointer that
      // came with no length.
      const auto* b = static_cast<const uint8_t*>(image);
      // OS/ABI 0x41 is CUDA 13's (ELF ABI version 8), 0x33 CUDA 12's (7).
      if (b[4] != 2 || (b[7] != 0x41 && b[7] != 0x33)) return CUDA_ERROR_INVALID_IMAGE;
      // A bare cubin: its size is in its own headers (the section table ends it).
      uint64_t shoff;
      uint16_t shentsize, shnum;
      std::memcpy(&shoff, b + 0x28, 8);
      std::memcpy(&shentsize, b + 0x3a, 2);
      std::memcpy(&shnum, b + 0x3c, 2);
      const uint64_t size = shoff + static_cast<uint64_t>(shentsize) * shnum;
      uint32_t eflags;
      std::memcpy(&eflags, b + 0x30, 4);
      // The architecture is in e_flags' second byte from ABI version 8 on,
      // and its low byte before (as sass::parse_cubin reads it): a CUDA 12
      // cubin for sm_86 carries 0x560556.
      const int sm = static_cast<int>(b[7] == 0x33 ? eflags & 0xff : (eflags >> 8) & 0xff);
      if (!vgpu::sass::runs_on(sm, false, static_cast<int>(cc))) return CUDA_ERROR_NO_BINARY_FOR_GPU;
      try {
        mid = s.rt->device(dev).load_cubin(b, size);
      } catch (const vgpu::Error& e) {
        if (e.code() == vgpu::Err::InvalidValue) return CUDA_ERROR_INVALID_IMAGE;
        throw;
      }
    } else {
      mid = s.rt->device(dev).load_module(text);
    }
    bind_managed_globals(s, dev, mid);
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

// A module from a file: its bytes, loaded as cuModuleLoadData loads them.
// PTX text is terminated, as the driver requires of an image in memory.
VGPU_EXPORT CUresult cuModuleLoad(CUmodule* module, const char* fname) {
  if (!module || !fname) return CUDA_ERROR_INVALID_VALUE;
  std::ifstream f(fname, std::ios::binary);
  if (!f) return CUDA_ERROR_FILE_NOT_FOUND;
  std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (bytes.size() < 4) return CUDA_ERROR_INVALID_IMAGE;
  return cuModuleLoadData(module, bytes.c_str());
}

// Releasing the primary context. Nothing is cached per context here, so this
// succeeds without tearing down the device -- except a fault a kernel left the
// context with, which a reset is the one way out of, as cudaDeviceReset is.
VGPU_EXPORT CUresult cuDevicePrimaryCtxReset(CUdevice) {
  ShimState& s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (s.initialized) s.rt->set_context_fault(0);
  return CUDA_SUCCESS;
}

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
 * Address space, physical memory and the mapping between them, as CUDA
 * documents them: cuMemAddressReserve takes addresses with nothing behind
 * them, cuMemCreate takes memory with no address, cuMemMap joins the two and
 * cuMemSetAccess makes the result usable. This is how PyTorch's expandable
 * segments and NCCL's windows grow a buffer without moving it.
 *
 * Every refusal a device makes, the engine makes: touching reserved space with
 * nothing mapped, touching a mapping before access is granted, and writing
 * through a read-only mapping all fault with a diagnostic naming which it was
 * (vgpu/memory.hpp).
 */

// A handle id the engine gave out, as the API's opaque handle. The engine's
// ids start at 1, so 0 stays available as "no handle".
namespace {
constexpr size_t kVmmGranularity = 64u * 1024u;

// The property struct a caller passes. Only a device-local pinned allocation
// exists here; anything else is refused rather than quietly treated as one.
CUresult check_prop(const CUmemAllocationProp* prop, ShimState& s, int* device_out) {
  if (!prop) return CUDA_ERROR_INVALID_VALUE;
  if (prop->type != CU_MEM_ALLOCATION_TYPE_PINNED) return CUDA_ERROR_INVALID_VALUE;
  if (prop->location.type != CU_MEM_LOCATION_TYPE_DEVICE) return CUDA_ERROR_INVALID_VALUE;
  if (prop->location.id < 0 || prop->location.id >= s.rt->device_count())
    return CUDA_ERROR_INVALID_DEVICE;
  // An exportable handle would have to mean something to another process.
  if (prop->requestedHandleTypes != 0) return CUDA_ERROR_NOT_SUPPORTED;
  if (device_out) *device_out = prop->location.id;
  return CUDA_SUCCESS;
}
}  // namespace

VGPU_EXPORT CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment,
                                         CUdeviceptr addr, unsigned long long flags) {
  return api("cuMemAddressReserve", true, false, [&](ShimState& s) {
    if (!ptr || size == 0 || flags != 0) return CUDA_ERROR_INVALID_VALUE;
    // A fixed address is a request for one particular range; the engine hands
    // out address space monotonically and cannot honour it.
    if (addr != 0) return CUDA_ERROR_NOT_SUPPORTED;
    *ptr = current(s).memory().reserve(size, alignment);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
  return api("cuMemAddressFree", true, false, [&](ShimState& s) {
    owner_memory(s, ptr).address_free(ptr, size);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size,
                                 const CUmemAllocationProp* prop, unsigned long long flags) {
  return api("cuMemCreate", true, false, [&](ShimState& s) {
    if (!handle || size == 0 || flags != 0) return CUDA_ERROR_INVALID_VALUE;
    int device = 0;
    if (const CUresult rc = check_prop(prop, s, &device); rc != CUDA_SUCCESS) return rc;
    *handle = s.rt->device(device).memory().create_handle(size);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
  return api("cuMemRelease", true, false, [&](ShimState& s) {
    // A handle belongs to the device it was created on, and nothing in the
    // handle says which that is, so every device is asked.
    for (int d = 0; d < s.rt->device_count(); ++d) {
      try {
        s.rt->device(d).memory().handle_size(handle);
      } catch (const vgpu::Error&) {
        continue;
      }
      s.rt->device(d).memory().release_handle(handle);
      return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_VALUE;
  });
}

VGPU_EXPORT CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                              CUmemGenericAllocationHandle handle, unsigned long long flags) {
  return api("cuMemMap", true, false, [&](ShimState& s) {
    if (flags != 0) return CUDA_ERROR_INVALID_VALUE;
    owner_memory(s, ptr).map(ptr, size, offset, handle);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
  return api("cuMemUnmap", true, false, [&](ShimState& s) {
    owner_memory(s, ptr).unmap(ptr, size);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc,
                                    size_t count) {
  return api("cuMemSetAccess", true, false, [&](ShimState& s) {
    if (!desc || count == 0) return CUDA_ERROR_INVALID_VALUE;
    vgpu::MemoryManager& mem = owner_memory(s, ptr);
    // Each descriptor names a device and what it may do. The engine keeps one
    // set of flags per mapping, which is the right model while every device
    // here reaches the memory through the same address.
    bool readable = false, writable = false;
    for (size_t i = 0; i < count; ++i) {
      if (desc[i].location.type != CU_MEM_LOCATION_TYPE_DEVICE) return CUDA_ERROR_INVALID_VALUE;
      if (desc[i].location.id < 0 || desc[i].location.id >= s.rt->device_count())
        return CUDA_ERROR_INVALID_DEVICE;
      switch (desc[i].flags) {
        case CU_MEM_ACCESS_FLAGS_PROT_READWRITE: readable = writable = true; break;
        case CU_MEM_ACCESS_FLAGS_PROT_READ: readable = true; break;
        case CU_MEM_ACCESS_FLAGS_PROT_NONE: break;
        default: return CUDA_ERROR_INVALID_VALUE;
      }
    }
    mem.set_access(ptr, size, readable, writable);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location,
                                    CUdeviceptr ptr) {
  return api("cuMemGetAccess", true, false, [&](ShimState& s) {
    if (!flags || !location) return CUDA_ERROR_INVALID_VALUE;
    if (location->type != CU_MEM_LOCATION_TYPE_DEVICE) return CUDA_ERROR_INVALID_VALUE;
    if (location->id < 0 || location->id >= s.rt->device_count()) return CUDA_ERROR_INVALID_DEVICE;
    bool readable = false, writable = false;
    if (!owner_memory(s, ptr).access_at(ptr, &readable, &writable)) return CUDA_ERROR_INVALID_VALUE;
    *flags = writable ? CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                      : readable ? CU_MEM_ACCESS_FLAGS_PROT_READ : CU_MEM_ACCESS_FLAGS_PROT_NONE;
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuMemGetAllocationGranularity(size_t* granularity,
                                                   const CUmemAllocationProp* prop,
                                                   CUmemAllocationGranularity_flags option) {
  if (!granularity) return CUDA_ERROR_INVALID_VALUE;
  if (option != CU_MEM_ALLOC_GRANULARITY_MINIMUM && option != CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)
    return CUDA_ERROR_INVALID_VALUE;
  (void)prop;
  // The chunk the sparse backing materializes, which is the unit every
  // reservation, handle and mapping here is measured in.
  *granularity = kVmmGranularity;
  return CUDA_SUCCESS;
}

VGPU_EXPORT CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop,
                                                            CUmemGenericAllocationHandle handle) {
  return api("cuMemGetAllocationPropertiesFromHandle", true, false, [&](ShimState& s) {
    if (!prop) return CUDA_ERROR_INVALID_VALUE;
    for (int d = 0; d < s.rt->device_count(); ++d) {
      try {
        s.rt->device(d).memory().handle_size(handle);
      } catch (const vgpu::Error&) {
        continue;
      }
      *prop = CUmemAllocationProp{};
      prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop->location.id = d;
      return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_VALUE;
  });
}

VGPU_EXPORT CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle, void* addr) {
  return api("cuMemRetainAllocationHandle", true, false, [&](ShimState& s) {
    if (!handle || !addr) return CUDA_ERROR_INVALID_VALUE;
    const auto va = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(addr));
    const uint64_t h = owner_memory(s, va).retain_handle_at(va);
    if (!h) return CUDA_ERROR_INVALID_VALUE;
    *handle = h;
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuMemExportToShareableHandle(void*, unsigned long long, int, unsigned long long) {
  // Another process would have to be able to map the same memory; device
  // memory here lives in this process's own sparse backing.
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
    dev_write(s, addr, &value, sizeof value);
    return CUDA_SUCCESS;
  });
}

// JIT options are performance and verbosity hints, and ignored -- except the
// ones that hand back output. There is no JIT log here (the loader's
// diagnostics go to stderr), so the info and error log buffers get an empty
// string and their sizes the 0 bytes written, and the wall time is 0. Left
// alone, a program printing its info log -- CUDA's matrixMulDynlinkJIT
// sample -- printed whatever its buffer held.
VGPU_EXPORT CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions,
                                        void* options, void** optionValues) {
  const CUresult r = cuModuleLoadData(module, image);
  const int* opts = static_cast<const int*>(options);
  if (!opts || !optionValues) return r;
  size_t info_size = 0, error_size = 0;
  for (unsigned i = 0; i < numOptions; ++i) {
    if (opts[i] == 4) info_size = reinterpret_cast<size_t>(optionValues[i]);    // INFO_LOG_BUFFER_SIZE_BYTES
    if (opts[i] == 6) error_size = reinterpret_cast<size_t>(optionValues[i]);   // ERROR_LOG_BUFFER_SIZE_BYTES
  }
  for (unsigned i = 0; i < numOptions; ++i) {
    switch (opts[i]) {
      case 2: {   // CU_JIT_WALL_TIME: a float, in the slot itself
        const float ms = 0.0f;
        std::memcpy(&optionValues[i], &ms, sizeof ms);
        break;
      }
      case 3:     // CU_JIT_INFO_LOG_BUFFER
        if (optionValues[i] && info_size) static_cast<char*>(optionValues[i])[0] = '\0';
        break;
      case 5:     // CU_JIT_ERROR_LOG_BUFFER
        if (optionValues[i] && error_size) static_cast<char*>(optionValues[i])[0] = '\0';
        break;
      case 4:
      case 6: optionValues[i] = nullptr; break;   // bytes written: none
      default: break;
    }
  }
  return r;
}

VGPU_EXPORT CUresult cuModuleUnload(CUmodule hmod) {
  return api("cuModuleUnload", true, false, [&](ShimState& s) {
    uintptr_t h = check_handle(reinterpret_cast<uintptr_t>(hmod), kTagModule, "module");
    auto it = s.modules.find(h);
    if (it == s.modules.end()) return CUDA_ERROR_NOT_FOUND;
    auto [dev, mid] = it->second;
    unload_module(s, dev, mid);
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
// Whether a kernel calls cudaGraphLaunch from the device, in its own code or a
// device function's. Such a kernel runs only in a graph, and this driver has
// no graphs: an RTX 3060's runtime refused one launched on its own with
// cudaErrorNotSupported, without running it.
bool launches_graphs_uncached(const vgpu::ptx::EntryFn& fn) {
  if (fn.sass)
    for (const std::string& f : vgpu::sass::reachable(*fn.sass, fn.name))
      if (f == "cudaGraphLaunch") return true;
  std::set<const vgpu::ptx::EntryFn*> seen;
  std::vector<const vgpu::ptx::EntryFn*> todo{&fn};
  while (!todo.empty()) {
    const vgpu::ptx::EntryFn* f = todo.back();
    todo.pop_back();
    if (!seen.insert(f).second) continue;
    for (const auto& ins : f->body)
      if (const auto* c = std::get_if<vgpu::ptx::OpCall>(&ins.op)) {
        if (c->callee == "cudaGraphLaunch") return true;
        if (c->target) todo.push_back(c->target);
      }
  }
  return false;
}
bool launches_graphs(const vgpu::ptx::EntryFn& fn) {
  static std::mutex mu;
  static std::unordered_map<const vgpu::ptx::EntryFn*, bool> known;
  std::lock_guard<std::mutex> lock(mu);
  const auto it = known.find(&fn);
  if (it != known.end()) return it->second;
  return known[&fn] = launches_graphs_uncached(fn);
}

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
      // The whole parameter: a struct passed by value is a .b8 array in PTX,
      // whose element is one byte and whose size is the struct's.
      uint32_t size = params[i].size;
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
    cfg.nonportable_cluster = rec.nonportable_cluster;
    if (launches_graphs(*rec.fn)) return CUDA_ERROR_NOT_SUPPORTED;
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
    for (auto& [dev, mid] : it->second.per_device_module) unload_module(s, dev, mid);
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

VGPU_EXPORT CUresult cuFuncSetAttribute(CUfunction hfunc, int attrib, int value) {
  // The non-portable cluster size changes what may launch, and the dynamic
  // shared memory ceiling what occupancy counts; the others are tuning knobs
  // the interpreter has no use for.
  if (attrib != 14 && attrib != 8) return CUDA_SUCCESS;   // NON_PORTABLE_CLUSTER_SIZE_ALLOWED, MAX_DYNAMIC_SHARED_SIZE_BYTES
  return api("cuFuncSetAttribute", true, false, [&](ShimState& s) {
    auto it = s.functions.find(reinterpret_cast<uintptr_t>(hfunc));
    // A CUkernel is accepted here too; its ceiling is not kept, as before.
    if (it == s.functions.end()) return attrib == 8 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
    if (attrib == 14) it->second.nonportable_cluster = value != 0;
    else it->second.max_dynamic_shared = value;
    return CUDA_SUCCESS;
  });
}
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

/* ---- the pre-CUDA 4 launch: shape and parameters kept on the function ----
 * cuFuncSetBlockShape, cuFuncSetSharedSize and cuParamSet* build a launch up
 * on the function, and cuLaunchGrid runs it. The parameter buffer is laid out
 * as the kernel declares its parameters: each at the next offset its
 * alignment allows. What is checked is what an RTX 3060 checks: a block the
 * device can run, at most 48 KiB of shared memory, parameters inside 32764
 * bytes, and a parameter size no larger than the kernel's parameter block
 * (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES at launch). A size smaller than the
 * block launches, with whatever the buffer holds. */

namespace {
constexpr size_t kMaxParamBytes = 32764;

FuncRec* legacy_func(ShimState& s, CUfunction f) {
  auto it = s.functions.find(reinterpret_cast<uintptr_t>(f));
  return it == s.functions.end() ? nullptr : &it->second;
}

CUresult param_set(CUfunction f, int offset, const void* data, unsigned n) {
  return api("cuParamSet", true, false, [&](ShimState& s) {
    FuncRec* rec = legacy_func(s, f);
    if (!rec) return CUDA_ERROR_INVALID_HANDLE;
    if (!data || offset < 0 || static_cast<size_t>(offset) + n > kMaxParamBytes) return CUDA_ERROR_INVALID_VALUE;
    if (rec->params.size() < static_cast<size_t>(offset) + n) rec->params.resize(static_cast<size_t>(offset) + n);
    if (n) std::memcpy(rec->params.data() + offset, data, n);
    return CUDA_SUCCESS;
  });
}

CUresult launch_grid(const char* name, CUfunction f, int width, int height, CUstream stream) {
  std::vector<std::vector<uint8_t>> storage;
  std::vector<void*> args;
  FuncRec rec;
  {
    ShimState& s = state();
    std::lock_guard<std::recursive_mutex> lock(s.mu);
    if (!s.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    const FuncRec* r = legacy_func(s, f);
    if (!r) return CUDA_ERROR_INVALID_HANDLE;
    rec = *r;
    const vgpu::DeviceProfile& p = s.rt->device(rec.device).profile();
    if (width <= 0 || height <= 0 || static_cast<uint32_t>(width) > p.limits.max_grid_dim[0] ||
        static_cast<uint32_t>(height) > p.limits.max_grid_dim[1])
      return CUDA_ERROR_INVALID_VALUE;
  }
  // Each declared parameter from its offset in the buffer; bytes never
  // written read as zero.
  size_t end = 0;
  for (const auto& param : rec.fn->params) {
    const size_t align = param.align ? param.align : std::max<uint32_t>(1, param.ty.bytes());
    const size_t at = (end + align - 1) / align * align;
    std::vector<uint8_t> bytes(param.size, 0);
    if (at < rec.params.size())
      std::memcpy(bytes.data(), rec.params.data() + at, std::min<size_t>(param.size, rec.params.size() - at));
    storage.push_back(std::move(bytes));
    end = at + param.size;
  }
  if (rec.param_size > end) {
    report(name, "cuParamSetSize gave " + std::to_string(rec.param_size) + " bytes of parameters; kernel '" +
                     rec.fn->name + "' takes " + std::to_string(end));
    return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
  }
  for (auto& b : storage) args.push_back(b.data());
  return launch_kernel_common(name, f, static_cast<unsigned>(width), static_cast<unsigned>(height), 1,
                              rec.block[0], rec.block[1], rec.block[2], rec.shared_bytes, stream,
                              args.empty() ? nullptr : args.data(), nullptr, /*cooperative=*/false);
}
}  // namespace

VGPU_EXPORT CUresult cuFuncSetBlockShape(CUfunction f, int x, int y, int z) {
  return api("cuFuncSetBlockShape", true, false, [&](ShimState& s) {
    FuncRec* rec = legacy_func(s, f);
    if (!rec) return CUDA_ERROR_INVALID_HANDLE;
    const vgpu::DeviceProfile& p = s.rt->device(rec->device).profile();
    if (x <= 0 || y <= 0 || z <= 0 || static_cast<uint32_t>(x) > p.limits.max_block_dim[0] ||
        static_cast<uint32_t>(y) > p.limits.max_block_dim[1] || static_cast<uint32_t>(z) > p.limits.max_block_dim[2] ||
        uint64_t{static_cast<uint32_t>(x)} * y * z > p.limits.max_threads_per_block)
      return CUDA_ERROR_INVALID_VALUE;
    rec->block[0] = static_cast<unsigned>(x);
    rec->block[1] = static_cast<unsigned>(y);
    rec->block[2] = static_cast<unsigned>(z);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuFuncSetSharedSize(CUfunction f, unsigned int bytes) {
  return api("cuFuncSetSharedSize", true, false, [&](ShimState& s) {
    FuncRec* rec = legacy_func(s, f);
    if (!rec) return CUDA_ERROR_INVALID_HANDLE;
    if (bytes > s.rt->device(rec->device).profile().limits.shared_mem_per_block) return CUDA_ERROR_INVALID_VALUE;
    rec->shared_bytes = bytes;
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuParamSetSize(CUfunction f, unsigned int numbytes) {
  return api("cuParamSetSize", true, false, [&](ShimState& s) {
    FuncRec* rec = legacy_func(s, f);
    if (!rec) return CUDA_ERROR_INVALID_HANDLE;
    if (numbytes > kMaxParamBytes) return CUDA_ERROR_INVALID_VALUE;
    rec->param_size = numbytes;
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuParamSeti(CUfunction f, int offset, unsigned int value) {
  return param_set(f, offset, &value, sizeof value);
}
VGPU_EXPORT CUresult cuParamSetf(CUfunction f, int offset, float value) {
  return param_set(f, offset, &value, sizeof value);
}
VGPU_EXPORT CUresult cuParamSetv(CUfunction f, int offset, void* ptr, unsigned int numbytes) {
  return param_set(f, offset, ptr, numbytes);
}
// Texture references are bound to a module's kernels by name, not passed; the
// card takes this call and it has nothing to do.
VGPU_EXPORT CUresult cuParamSetTexRef(CUfunction f, int, CUtexref tex) {
  return api("cuParamSetTexRef", true, false, [&](ShimState& s) {
    if (!legacy_func(s, f) || !s.texrefs.count(reinterpret_cast<uintptr_t>(tex))) return CUDA_ERROR_INVALID_HANDLE;
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
  return launch_grid("cuLaunchGrid", f, grid_width, grid_height, nullptr);
}
VGPU_EXPORT CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream stream) {
  return launch_grid("cuLaunchGridAsync", f, grid_width, grid_height, stream);
}
VGPU_EXPORT CUresult cuLaunch(CUfunction f) { return launch_grid("cuLaunch", f, 1, 1, nullptr); }

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
VGPU_EXPORT CUresult cuMemcpyDtoDAsync(CUdeviceptr a, CUdeviceptr b, size_t n, CUstream) {
  return cuMemcpyDtoD_v2(a, b, n);
}

// No profiler collects anything to start or stop; the card succeeds.
VGPU_EXPORT CUresult cuProfilerStart(void) { return CUDA_SUCCESS; }
VGPU_EXPORT CUresult cuProfilerStop(void) { return CUDA_SUCCESS; }

/* ---- graphics interop ----
 * There is no OpenGL, Direct3D or Vulkan here to register a resource with, so
 * no CUgraphicsResource ever exists: each of these is answered as an RTX 3060
 * answers a null one (INVALID_HANDLE), and mapping or unmapping no resources
 * as it answers that (INVALID_VALUE). */
VGPU_EXPORT CUresult cuGraphicsUnregisterResource(CUgraphicsResource) { return CUDA_ERROR_INVALID_HANDLE; }
VGPU_EXPORT CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource* res, CUstream) {
  return count == 0 || !res ? CUDA_ERROR_INVALID_VALUE : CUDA_ERROR_INVALID_HANDLE;
}
VGPU_EXPORT CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource* res, CUstream) {
  return count == 0 || !res ? CUDA_ERROR_INVALID_VALUE : CUDA_ERROR_INVALID_HANDLE;
}
VGPU_EXPORT CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource, unsigned int) {
  return CUDA_ERROR_INVALID_HANDLE;
}
VGPU_EXPORT CUresult cuGraphicsResourceSetMapFlags(CUgraphicsResource r, unsigned int f) {
  return cuGraphicsResourceSetMapFlags_v2(r, f);
}
VGPU_EXPORT CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr*, size_t*, CUgraphicsResource) {
  return CUDA_ERROR_INVALID_HANDLE;
}
VGPU_EXPORT CUresult cuGraphicsResourceGetMappedPointer(CUdeviceptr* p, size_t* n, CUgraphicsResource r) {
  return cuGraphicsResourceGetMappedPointer_v2(p, n, r);
}
VGPU_EXPORT CUresult cuGraphicsSubResourceGetMappedArray(CUarray*, CUgraphicsResource, unsigned int, unsigned int) {
  return CUDA_ERROR_INVALID_HANDLE;
}
VGPU_EXPORT CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray*, CUgraphicsResource) {
  return CUDA_ERROR_INVALID_HANDLE;
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
// The range in `m` (host ranges by base) holding byte `p`, or nullptr. A
// range's size is taken exactly: the card answers for a registration's own
// bytes, not for the rest of the pages it touches.
const vgpu::runtime::HostRange* host_range_at(const std::map<void*, vgpu::runtime::HostRange>& m,
                                              const void* p, void** base = nullptr) {
  auto it = m.upper_bound(const_cast<void*>(p));
  if (it == m.begin()) return nullptr;
  --it;
  if (static_cast<const char*>(p) >= static_cast<const char*>(it->first) + it->second.size) return nullptr;
  if (base) *base = it->first;
  return &it->second;
}

// Host memory registered with cuMemHostRegister or cudaHostRegister. One
// registry for both libraries, kept in the machine they share.
std::map<void*, vgpu::runtime::HostRange>& registrations(ShimState& s) {
  return s.rt->host_registrations();
}

// Pinned host memory, with the flags it was allocated with: cuMemHostAlloc's
// and the runtime's cudaMallocHost and cudaHostAlloc results in one record, in
// the machine both libraries share. The driver used to keep its own, so
// cuMemHostGetFlags on memory the runtime pinned was INVALID_VALUE where an
// RTX 3060 reports its flags, and cuMemFreeHost refused it.
std::map<void*, vgpu::runtime::HostRange>& pinned(ShimState& s) {
  return s.rt->host_allocations();
}

// `rows` rows of `width` elements of T, `pitch` bytes apart -- a 1D fill is
// one row. What an RTX 3060 checks: nothing, for an empty region (a null
// pointer included); a pointer aligned to the element; and, only when there is
// more than one row, a pitch that holds a row and is a whole number of
// elements.
//
// Registered host memory is refused, as cudaMemset refuses it. The core's
// fill reaches host mappings, and a registration is one (cuMemHostRegister
// maps the range at its host address), so without this the fill would land.
// On the card the registered pointer itself is refused too: a fill reaches
// registered memory only through the separate device alias
// cuMemHostGetDevicePointer returns there. Here that alias is the host
// address, and the two cannot be told apart, so the fill is refused for both,
// the same answer cudaMemset gives. Pinned memory (cuMemHostAlloc) is filled,
// as the card fills it.
template <typename T>
CUresult memset_impl(const char* name, CUdeviceptr dptr, size_t pitch, T value, size_t width,
                     size_t rows) {
  return api(name, true, false, [&](ShimState& s) {
    if (width == 0 || rows == 0) return CUDA_SUCCESS;
    if (!dptr || dptr % sizeof(T)) return CUDA_ERROR_INVALID_VALUE;
    if (rows > 1 && (pitch < width * sizeof(T) || pitch % sizeof(T))) return CUDA_ERROR_INVALID_VALUE;
    if (!is_device_ptr(dptr) && host_range_at(registrations(s), reinterpret_cast<void*>(dptr)))
      return CUDA_ERROR_INVALID_VALUE;
    // The whole region inside its allocation before any of it is written, as
    // the card refuses a fill that runs past the end without starting it.
    const uint64_t end = dptr + (rows - 1) * pitch + width * sizeof(T);
    if (is_device_ptr(dptr)) {
      uint64_t base = 0, size = 0;
      if (owner_memory(s, dptr).find_allocation(dptr, &base, &size) && end > base + size)
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (size_t y = 0; y < rows; ++y)
      dev_fill(s, dptr + y * pitch, reinterpret_cast<const uint8_t*>(&value), sizeof(T), width * sizeof(T));
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
// Hopper TMA descriptors. The object is opaque (include/vgpu/exec/tensormap.hpp
// holds this simulator's layout of it); what is not opaque is the list of
// requirements cuda.h documents, and each one is checked, so a map that the
// real driver would refuse is refused here with the rule it broke. Encoding
// needs no context: it touches no device, and CUTLASS calls it before any.
VGPU_EXPORT CUresult cuTensorMapEncodeTiled(void* tensorMap, unsigned int dataType, unsigned int rank,
                                            void* globalAddress, const unsigned long long* globalDim,
                                            const unsigned long long* globalStrides,
                                            const unsigned int* boxDim,
                                            const unsigned int* elementStrides, unsigned int interleave,
                                            unsigned int swizzle, unsigned int l2Promotion,
                                            unsigned int oobFill) {
  return api("cuTensorMapEncodeTiled", false, false, [&](ShimState&) -> CUresult {
    std::string why;
    switch (vgpu::exec::encode_tiled(tensorMap, dataType, rank, globalAddress, globalDim, globalStrides,
                                     boxDim, elementStrides, interleave, swizzle, l2Promotion, oobFill,
                                     &why)) {
      case vgpu::exec::TmapResult::Ok: return CUDA_SUCCESS;
      case vgpu::exec::TmapResult::Unsupported:
        throw vgpu::Error::make(vgpu::Err::Unsupported, "cuTensorMapEncodeTiled: ", why);
      case vgpu::exec::TmapResult::Invalid: break;
    }
    throw vgpu::Error::make(vgpu::Err::InvalidValue, "cuTensorMapEncodeTiled: ", why);
  });
}

// Points an existing map at a new base address, keeping everything else.
VGPU_EXPORT CUresult cuTensorMapReplaceAddress(void* tensorMap, void* globalAddress) {
  return api("cuTensorMapReplaceAddress", false, false, [&](ShimState&) -> CUresult {
    std::string why;
    if (vgpu::exec::replace_address(tensorMap, globalAddress, &why) != vgpu::exec::TmapResult::Ok)
      throw vgpu::Error::make(vgpu::Err::InvalidValue, "cuTensorMapReplaceAddress: ", why);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuTensorMapEncodeIm2col(void* tensorMap, unsigned int dataType, unsigned int rank,
                                             void* globalAddress, const unsigned long long* globalDim,
                                             const unsigned long long* globalStrides,
                                             const int* lowerCorner, const int* upperCorner,
                                             unsigned int channelsPerPixel, unsigned int pixelsPerColumn,
                                             const unsigned int* elementStrides, unsigned int interleave,
                                             unsigned int swizzle, unsigned int l2Promotion,
                                             unsigned int oobFill) {
  return api("cuTensorMapEncodeIm2col", false, false, [&](ShimState&) -> CUresult {
    std::string why;
    switch (vgpu::exec::encode_im2col(tensorMap, dataType, rank, globalAddress, globalDim, globalStrides,
                                      lowerCorner, upperCorner, channelsPerPixel, pixelsPerColumn,
                                      elementStrides, interleave, swizzle, l2Promotion, oobFill, &why)) {
      case vgpu::exec::TmapResult::Ok: return CUDA_SUCCESS;
      case vgpu::exec::TmapResult::Unsupported:
        throw vgpu::Error::make(vgpu::Err::Unsupported, "cuTensorMapEncodeIm2col: ", why);
      case vgpu::exec::TmapResult::Invalid: break;
    }
    throw vgpu::Error::make(vgpu::Err::InvalidValue, "cuTensorMapEncodeIm2col: ", why);
  });
}

VGPU_EXPORT CUresult cuMemsetD8_v2(CUdeviceptr d, unsigned char v, size_t n) {
  return memset_impl("cuMemsetD8", d, 0, v, n, 1);
}
VGPU_EXPORT CUresult cuMemsetD16_v2(CUdeviceptr d, unsigned short v, size_t n) {
  return memset_impl("cuMemsetD16", d, 0, v, n, 1);
}
VGPU_EXPORT CUresult cuMemsetD32_v2(CUdeviceptr d, unsigned int v, size_t n) {
  return memset_impl("cuMemsetD32", d, 0, v, n, 1);
}
VGPU_EXPORT CUresult cuMemsetD8(CUdeviceptr d, unsigned char v, size_t n) { return cuMemsetD8_v2(d, v, n); }
VGPU_EXPORT CUresult cuMemsetD16(CUdeviceptr d, unsigned short v, size_t n) { return cuMemsetD16_v2(d, v, n); }
VGPU_EXPORT CUresult cuMemsetD32(CUdeviceptr d, unsigned int v, size_t n) { return cuMemsetD32_v2(d, v, n); }
VGPU_EXPORT CUresult cuMemsetD8Async(CUdeviceptr d, unsigned char v, size_t n, CUstream) {
  return memset_impl("cuMemsetD8", d, 0, v, n, 1);
}
VGPU_EXPORT CUresult cuMemsetD16Async(CUdeviceptr d, unsigned short v, size_t n, CUstream) {
  return memset_impl("cuMemsetD16", d, 0, v, n, 1);
}
VGPU_EXPORT CUresult cuMemsetD32Async(CUdeviceptr d, unsigned int v, size_t n, CUstream) {
  return memset_impl("cuMemsetD32", d, 0, v, n, 1);
}
// Width counts elements, the pitch bytes.
VGPU_EXPORT CUresult cuMemsetD2D8_v2(CUdeviceptr d, size_t pitch, unsigned char v, size_t w, size_t h) {
  return memset_impl("cuMemsetD2D8", d, pitch, v, w, h);
}
VGPU_EXPORT CUresult cuMemsetD2D16_v2(CUdeviceptr d, size_t pitch, unsigned short v, size_t w, size_t h) {
  return memset_impl("cuMemsetD2D16", d, pitch, v, w, h);
}
VGPU_EXPORT CUresult cuMemsetD2D32_v2(CUdeviceptr d, size_t pitch, unsigned int v, size_t w, size_t h) {
  return memset_impl("cuMemsetD2D32", d, pitch, v, w, h);
}
VGPU_EXPORT CUresult cuMemsetD2D8(CUdeviceptr d, size_t pitch, unsigned char v, size_t w, size_t h) {
  return cuMemsetD2D8_v2(d, pitch, v, w, h);
}
VGPU_EXPORT CUresult cuMemsetD2D16(CUdeviceptr d, size_t pitch, unsigned short v, size_t w, size_t h) {
  return cuMemsetD2D16_v2(d, pitch, v, w, h);
}
VGPU_EXPORT CUresult cuMemsetD2D32(CUdeviceptr d, size_t pitch, unsigned int v, size_t w, size_t h) {
  return cuMemsetD2D32_v2(d, pitch, v, w, h);
}
VGPU_EXPORT CUresult cuMemsetD2D8Async(CUdeviceptr d, size_t pitch, unsigned char v, size_t w, size_t h,
                                       CUstream) {
  return memset_impl("cuMemsetD2D8", d, pitch, v, w, h);
}
VGPU_EXPORT CUresult cuMemsetD2D16Async(CUdeviceptr d, size_t pitch, unsigned short v, size_t w, size_t h,
                                        CUstream) {
  return memset_impl("cuMemsetD2D16", d, pitch, v, w, h);
}
VGPU_EXPORT CUresult cuMemsetD2D32Async(CUdeviceptr d, size_t pitch, unsigned int v, size_t w, size_t h,
                                        CUstream) {
  return memset_impl("cuMemsetD2D32", d, pitch, v, w, h);
}

VGPU_EXPORT CUresult cuMemGetAddressRange_v2(CUdeviceptr* base, size_t* size, CUdeviceptr dptr) {
  return api("cuMemGetAddressRange", true, false, [&](ShimState& s) {
    uint64_t b = 0, sz = 0;
    // Managed memory is the allocation it was made as -- for a module's
    // managed global, the variable, as the card reports it.
    if (const auto* m = managed_range(s, dptr, 1)) {
      if (base) *base = m->first;
      if (size) *size = m->second;
      return CUDA_SUCCESS;
    }
    // A device-heap block is no allocation of the host's: the card answers
    // NOT_FOUND for it.
    if (owner_memory(s, dptr).heap_contains(dptr)) return CUDA_ERROR_NOT_FOUND;
    if (!owner_memory(s, dptr).find_allocation(dptr, &b, &sz)) return CUDA_ERROR_INVALID_VALUE;
    if (base) *base = b;
    if (size) *size = sz;
    return CUDA_SUCCESS;
  });
}

/* ---- host (pinned) memory: aligned host allocations every device maps ----
 * Under unified addressing pinned memory is device-addressable at its host
 * address, as the runtime's cudaMallocHost makes it: a kernel can read it and
 * cuMemsetD* fills it, as on the card. */

VGPU_EXPORT CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int flags) {
  return api("cuMemHostAlloc", true, false, [&](ShimState& s) {
    if (!pp || bytesize == 0) return CUDA_ERROR_INVALID_VALUE;
    void* p = std::aligned_alloc(4096, (bytesize + 4095) / 4096 * 4096);
    if (!p) return CUDA_ERROR_OUT_OF_MEMORY;
    for (int d = 0; d < s.rt->device_count(); ++d)
      s.rt->device(d).memory().map_host(reinterpret_cast<uint64_t>(p), p, bytesize);
    pinned(s)[p] = vgpu::runtime::HostRange{bytesize, 0, flags};
    *pp = p;
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemAllocHost_v2(void** pp, size_t bytesize) {
  return cuMemHostAlloc(pp, bytesize, 0);
}
VGPU_EXPORT CUresult cuMemAllocHost(void** pp, size_t bytesize) { return cuMemHostAlloc(pp, bytesize, 0); }
VGPU_EXPORT CUresult cuMemFreeHost(void* p) {
  return api("cuMemFreeHost", true, false, [&](ShimState& s) {
    auto it = pinned(s).find(p);
    if (it == pinned(s).end()) return CUDA_ERROR_INVALID_VALUE;
    for (int d = 0; d < s.rt->device_count(); ++d) s.rt->device(d).memory().unmap_host(reinterpret_cast<uint64_t>(p));
    pinned(s).erase(it);
    std::free(p);
    return CUDA_SUCCESS;
  });
}

/* ---- registered host memory ----
 * cuMemHostRegister and cudaHostRegister keep one registry, in the machine
 * both libraries share: on the card, memory one API registered is already
 * registered to the other (CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED), and
 * either API unregisters it. As cudaHostRegister does, the range is mapped
 * into every device at its host address, and that is the device pointer
 * cuMemHostGetDevicePointer reports. The answers below are an RTX 3060's. */

VGPU_EXPORT CUresult cuMemHostRegister_v2(void* p, size_t bytesize, unsigned int flags) {
  return api("cuMemHostRegister", true, false, [&](ShimState& s) {
    // PORTABLE, DEVICEMAP, IOMEMORY and READ_ONLY; nothing else. Read-only
    // registration is not supported -- nothing here would stop a kernel
    // writing -- which READ_ONLY_HOST_REGISTER_SUPPORTED (113) says, and
    // which the documented answer for a device without it is.
    if (!p || bytesize == 0 || (flags & ~0xFu)) return CUDA_ERROR_INVALID_VALUE;
    if (flags & CU_MEMHOSTREGISTER_READ_ONLY) return CUDA_ERROR_NOT_SUPPORTED;
    auto& regs = registrations(s);
    const char* lo = static_cast<const char*>(p);
    void* hi = static_cast<char*>(p) + bytesize;
    // The ranges are disjoint, so the one starting last below `hi` is the only
    // one that can reach past `lo`.
    auto it = regs.lower_bound(hi);
    if (it != regs.begin() && static_cast<const char*>(std::prev(it)->first) + std::prev(it)->second.size > lo)
      return CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
    // Memory CUDA allocated -- pinned or managed, by either library -- is not
    // registered over; the card answers that with INVALID_VALUE.
    vgpu::MemoryManager& mem = s.rt->device(0).memory();
    if (mem.is_host_mapped(reinterpret_cast<uint64_t>(p)) ||
        mem.is_host_mapped(reinterpret_cast<uint64_t>(p) + bytesize - 1))
      return CUDA_ERROR_INVALID_VALUE;
    const int dev = current_device(s);
    for (int d = 0; d < s.rt->device_count(); ++d)
      s.rt->device(d).memory().map_host(reinterpret_cast<uint64_t>(p), p, bytesize);
    regs[p] = vgpu::runtime::HostRange{bytesize, dev, flags};
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemHostRegister(void* p, size_t bytesize, unsigned int flags) {
  return cuMemHostRegister_v2(p, bytesize, flags);
}

// Only by the pointer that was registered: one inside a registration is
// INVALID_VALUE, and so is memory CUDA allocated (pinned or managed); one
// never registered is HOST_MEMORY_NOT_REGISTERED.
VGPU_EXPORT CUresult cuMemHostUnregister(void* p) {
  return api("cuMemHostUnregister", true, false, [&](ShimState& s) {
    if (!p) return CUDA_ERROR_INVALID_VALUE;
    auto& regs = registrations(s);
    auto it = regs.find(p);
    if (it == regs.end())
      return host_range_at(regs, p) || s.rt->device(0).memory().is_host_mapped(reinterpret_cast<uint64_t>(p))
                 ? CUDA_ERROR_INVALID_VALUE
                 : CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED;
    for (int d = 0; d < s.rt->device_count(); ++d) s.rt->device(d).memory().unmap_host(reinterpret_cast<uint64_t>(p));
    regs.erase(it);
    return CUDA_SUCCESS;
  });
}

// What the card reports: DEVICEMAP always, since every such range is mapped,
// with PORTABLE (and, for pinned allocations, WRITECOMBINED) as requested.
// IOMEMORY and READ_ONLY are not reported back. Managed memory answers
// DEVICEMAP too.
VGPU_EXPORT CUresult cuMemHostGetFlags(unsigned int* flags, void* p) {
  return api("cuMemHostGetFlags", true, false, [&](ShimState& s) {
    if (!flags || !p) return CUDA_ERROR_INVALID_VALUE;
    if (const auto* r = host_range_at(registrations(s), p)) {
      *flags = (r->flags & CU_MEMHOSTREGISTER_PORTABLE) | CU_MEMHOSTALLOC_DEVICEMAP;
      return CUDA_SUCCESS;
    }
    if (const auto* r = host_range_at(pinned(s), p)) {
      *flags = (r->flags & 0x7u) | CU_MEMHOSTALLOC_DEVICEMAP;
      return CUDA_SUCCESS;
    }
    if (managed_range(s, reinterpret_cast<CUdeviceptr>(p), 1)) {
      *flags = CU_MEMHOSTALLOC_DEVICEMAP;
      return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_VALUE;
  });
}

// The device address of pinned or registered memory, the byte asked about
// included: its host address, where every device maps it.
VGPU_EXPORT CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr* dptr, void* p, unsigned int flags) {
  return api("cuMemHostGetDevicePointer", true, false, [&](ShimState& s) {
    if (!dptr || !p || flags != 0) return CUDA_ERROR_INVALID_VALUE;
    if (!host_range_at(registrations(s), p) && !host_range_at(pinned(s), p))
      return CUDA_ERROR_INVALID_VALUE;
    *dptr = reinterpret_cast<CUdeviceptr>(p);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemHostGetDevicePointer(CUdeviceptr* dptr, void* p, unsigned int flags) {
  return cuMemHostGetDevicePointer_v2(dptr, p, flags);
}

/* ---- managed memory ----
 * One allocation the host and every device address at the same pointer. Here
 * that is host memory mapped into each device's address space, as the
 * runtime's cudaMallocManaged does: there is one memory, so nothing migrates,
 * and prefetches and advice have nothing to move. What they check is what an
 * RTX 3080 Ti's driver checks: the flags, that the range is managed, and that
 * a device named exists. */

namespace {
CUresult check_location(ShimState& s, const CUmemLocation& loc) {
  switch (loc.type) {
    case CU_MEM_LOCATION_TYPE_DEVICE:
      return loc.id >= 0 && loc.id < s.rt->device_count() ? CUDA_SUCCESS : CUDA_ERROR_INVALID_DEVICE;
    case CU_MEM_LOCATION_TYPE_HOST:
    case CU_MEM_LOCATION_TYPE_HOST_NUMA:
    case CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT:
      return CUDA_SUCCESS;
    default:
      return CUDA_ERROR_INVALID_VALUE;
  }
}
CUmemLocation location_of(CUdevice dev) {
  CUmemLocation loc{};
  loc.type = dev == -1 ? CU_MEM_LOCATION_TYPE_HOST : CU_MEM_LOCATION_TYPE_DEVICE;  // -1: CU_DEVICE_CPU
  loc.id = dev == -1 ? 0 : dev;
  return loc;
}
}  // namespace

VGPU_EXPORT CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) {
  return api("cuMemAllocManaged", true, false, [&](ShimState& s) {
    // CU_MEM_ATTACH_GLOBAL (1) or CU_MEM_ATTACH_HOST (2), exactly; a size of 0 is refused.
    if (!dptr || bytesize == 0 || (flags != 1 && flags != 2)) return CUDA_ERROR_INVALID_VALUE;
    const size_t n = (bytesize + 4095) / 4096 * 4096;
    void* p = std::aligned_alloc(4096, n);
    if (!p) return CUDA_ERROR_OUT_OF_MEMORY;
    for (int d = 0; d < s.rt->device_count(); ++d)
      s.rt->device(d).memory().map_host(reinterpret_cast<uint64_t>(p), p, bytesize);
    s.managed[reinterpret_cast<uintptr_t>(p)] = bytesize;
    *dptr = reinterpret_cast<CUdeviceptr>(p);
    return CUDA_SUCCESS;
  });
}

// Which stream may touch an allocation. The whole allocation, from its base
// (length 0 or its full size), with GLOBAL (1), HOST (2) or SINGLE (4).
VGPU_EXPORT CUresult cuStreamAttachMemAsync(CUstream, CUdeviceptr dptr, size_t length, unsigned int flags) {
  return api("cuStreamAttachMemAsync", true, false, [&](ShimState& s) {
    auto it = s.managed.find(dptr);
    if (it == s.managed.end() || (length != 0 && length != it->second) ||
        (flags != 1 && flags != 2 && flags != 4))
      return CUDA_ERROR_INVALID_VALUE;
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuMemPrefetchAsync_v2(CUdeviceptr dptr, size_t count, CUmemLocation location,
                                           unsigned int, CUstream) {
  return api("cuMemPrefetchAsync", true, false, [&](ShimState& s) {
    if (!managed_range(s, dptr, count)) return CUDA_ERROR_INVALID_VALUE;
    return check_location(s, location);
  });
}
VGPU_EXPORT CUresult cuMemPrefetchAsync(CUdeviceptr dptr, size_t count, CUdevice dstDevice, CUstream stream) {
  return cuMemPrefetchAsync_v2(dptr, count, location_of(dstDevice), 0, stream);
}
VGPU_EXPORT CUresult cuMemAdvise_v2(CUdeviceptr dptr, size_t count, int advice, CUmemLocation location) {
  return api("cuMemAdvise", true, false, [&](ShimState& s) {
    if (!managed_range(s, dptr, count) || advice < 1 || advice > 6) return CUDA_ERROR_INVALID_VALUE;
    // Read-mostly (1, 2) names no location; the others name one.
    if (advice > 2) return check_location(s, location);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemAdvise(CUdeviceptr dptr, size_t count, int advice, CUdevice device) {
  return cuMemAdvise_v2(dptr, count, advice, location_of(device));
}

/* ---- CUDA arrays and 2D/3D copies ----
 * An array is device memory with a shape. The rules for creating one are the
 * ones an RTX 3060's and an RTX 3080 Ti's drivers enforce (they agree):
 * Width is never 0; a Depth needs a Height unless the array is layered, and a
 * layered one needs at least one layer; a cubemap is square with a Depth of 6,
 * or a multiple of 6 when layered; 1, 2 or 4 channels; a known format and
 * known flag bits. The formats are the classic eight; the newer packed and
 * block-compressed ones are refused here. */

namespace {
constexpr uintptr_t kTagArray = 0;

// Taken by reference and read as the raw value: an application can pass a
// format this header does not name, and loading that as the enum is undefined
// (UBSan: "not a valid value for type 'CUarray_format_enum'").
size_t format_bytes(const CUarray_format& f) {
  unsigned raw;
  static_assert(sizeof raw == sizeof f);
  std::memcpy(&raw, &f, sizeof raw);
  switch (raw) {
    case CU_AD_FORMAT_UNSIGNED_INT8: case CU_AD_FORMAT_SIGNED_INT8: return 1;
    case CU_AD_FORMAT_UNSIGNED_INT16: case CU_AD_FORMAT_SIGNED_INT16: case CU_AD_FORMAT_HALF: return 2;
    case CU_AD_FORMAT_UNSIGNED_INT32: case CU_AD_FORMAT_SIGNED_INT32: case CU_AD_FORMAT_FLOAT: return 4;
  }
  return 0;
}

bool valid_array(const CUDA_ARRAY3D_DESCRIPTOR& d) {
  const unsigned known = CUDA_ARRAY3D_LAYERED | CUDA_ARRAY3D_SURFACE_LDST | CUDA_ARRAY3D_CUBEMAP |
                         CUDA_ARRAY3D_TEXTURE_GATHER;
  if (d.Flags & ~known) return false;
  if (!format_bytes(d.Format)) return false;
  if (d.NumChannels != 1 && d.NumChannels != 2 && d.NumChannels != 4) return false;
  if (d.Width == 0) return false;
  const bool layered = d.Flags & CUDA_ARRAY3D_LAYERED, cube = d.Flags & CUDA_ARRAY3D_CUBEMAP;
  if (cube) {
    if (d.Width != d.Height) return false;
    if (layered ? (d.Depth == 0 || d.Depth % 6 != 0) : d.Depth != 6) return false;
    return true;
  }
  if (layered) return d.Depth != 0;
  if (d.Height == 0 && d.Depth != 0) return false;
  return true;
}

ArrayRec* array_rec(ShimState& s, CUarray a) {
  auto it = s.arrays.find(reinterpret_cast<uintptr_t>(a));
  return it == s.arrays.end() ? nullptr : &it->second;
}

// One side of a 2D or 3D copy, resolved to where its bytes are.
struct Side {
  bool host = false;        // host memory, reached with memcpy
  const char* hsrc = nullptr;
  char* hdst = nullptr;
  CUdeviceptr dev = 0;      // device, managed or array memory
  size_t pitch = 0, height = 0;
};

CUresult resolve(ShimState& s, CUmemorytype t, const void* h_in, void* h_out, CUdeviceptr d, CUarray a,
                 size_t pitch, size_t height, size_t x, size_t y, size_t z, size_t w, size_t rows,
                 size_t depth, Side* out) {
  switch (t) {
    case CU_MEMORYTYPE_HOST:
      if (!h_in && !h_out) return CUDA_ERROR_INVALID_VALUE;
      out->host = true;
      out->hsrc = static_cast<const char*>(h_in);
      out->hdst = static_cast<char*>(h_out);
      out->pitch = pitch;
      out->height = height;
      break;
    case CU_MEMORYTYPE_DEVICE:
    case CU_MEMORYTYPE_UNIFIED:
      out->dev = d;
      out->pitch = pitch;
      out->height = height;
      break;
    case CU_MEMORYTYPE_ARRAY: {
      const ArrayRec* r = array_rec(s, a);
      if (!r) return CUDA_ERROR_INVALID_VALUE;
      // The region stays inside the array: its rows, rows and slices.
      if (x + w > r->row || y + rows > r->rows || z + depth > r->slices) return CUDA_ERROR_INVALID_VALUE;
      out->dev = r->mem;
      out->pitch = r->row;
      out->height = r->rows;
      break;
    }
    default:
      return CUDA_ERROR_INVALID_VALUE;
  }
  // A pitched host or device region must hold a row.
  if (t != CU_MEMORYTYPE_ARRAY && rows > 1 && out->pitch < w) return CUDA_ERROR_INVALID_VALUE;
  if (!out->host) out->dev += z * out->pitch * out->height + y * out->pitch + x;
  else if (out->hsrc) out->hsrc += z * out->pitch * out->height + y * out->pitch + x;
  else out->hdst += z * out->pitch * out->height + y * out->pitch + x;
  return CUDA_SUCCESS;
}

CUresult copy_rows(ShimState& s, const Side& src, const Side& dst, size_t w, size_t rows, size_t depth) {
  std::vector<uint8_t> tmp(w);
  for (size_t z = 0; z < depth; ++z)
    for (size_t y = 0; y < rows; ++y) {
      const size_t so = z * src.pitch * src.height + y * src.pitch;
      const size_t d_o = z * dst.pitch * dst.height + y * dst.pitch;
      if (src.host) std::memcpy(tmp.data(), src.hsrc + so, w);
      else dev_read(s, tmp.data(), src.dev + so, w);
      if (dst.host) std::memcpy(dst.hdst + d_o, tmp.data(), w);
      else dev_write(s, dst.dev + d_o, tmp.data(), w);
    }
  return CUDA_SUCCESS;
}

CUresult copy3d(ShimState& s, const CUDA_MEMCPY3D& p) {
  if (p.WidthInBytes == 0 || p.Height == 0 || p.Depth == 0) return CUDA_SUCCESS;
  if (p.srcLOD || p.dstLOD) return CUDA_ERROR_INVALID_VALUE;
  Side src, dst;
  if (CUresult r = resolve(s, p.srcMemoryType, p.srcHost, nullptr, p.srcDevice, p.srcArray, p.srcPitch,
                           p.srcHeight, p.srcXInBytes, p.srcY, p.srcZ, p.WidthInBytes, p.Height, p.Depth,
                           &src))
    return r;
  if (CUresult r = resolve(s, p.dstMemoryType, nullptr, p.dstHost, p.dstDevice, p.dstArray, p.dstPitch,
                           p.dstHeight, p.dstXInBytes, p.dstY, p.dstZ, p.WidthInBytes, p.Height, p.Depth,
                           &dst))
    return r;
  if (p.Depth > 1 && ((!src.host && src.height < p.Height && p.srcMemoryType != CU_MEMORYTYPE_ARRAY) ||
                      (src.host && src.height < p.Height) || (dst.host && dst.height < p.Height) ||
                      (!dst.host && dst.height < p.Height && p.dstMemoryType != CU_MEMORYTYPE_ARRAY)))
    return CUDA_ERROR_INVALID_VALUE;  // slices would overlap
  return copy_rows(s, src, dst, p.WidthInBytes, p.Height, p.Depth);
}

CUDA_MEMCPY3D as3d(const CUDA_MEMCPY2D& c) {
  CUDA_MEMCPY3D p{};
  p.srcXInBytes = c.srcXInBytes; p.srcY = c.srcY; p.srcMemoryType = c.srcMemoryType;
  p.srcHost = c.srcHost; p.srcDevice = c.srcDevice; p.srcArray = c.srcArray; p.srcPitch = c.srcPitch;
  p.dstXInBytes = c.dstXInBytes; p.dstY = c.dstY; p.dstMemoryType = c.dstMemoryType;
  p.dstHost = c.dstHost; p.dstDevice = c.dstDevice; p.dstArray = c.dstArray; p.dstPitch = c.dstPitch;
  p.WidthInBytes = c.WidthInBytes; p.Height = c.Height; p.Depth = 1;
  return p;
}

// Bytes [off, off + n) of an array's first row, for cuMemcpyHtoA and the
// rest: the card takes them for an array of any shape, within that row.
CUresult linear_part(ShimState& s, CUarray a, size_t off, size_t n, CUdeviceptr* at) {
  const ArrayRec* r = array_rec(s, a);
  if (!r) return CUDA_ERROR_INVALID_VALUE;
  if (off > r->row || n > r->row - off) return CUDA_ERROR_INVALID_VALUE;
  *at = r->mem + off;
  return CUDA_SUCCESS;
}
}  // namespace

namespace {
// A new array on the current device, for a descriptor valid_array accepts.
uintptr_t create_array(ShimState& s, const CUDA_ARRAY3D_DESCRIPTOR& desc) {
  ArrayRec r;
  r.desc = desc;
  r.elem = format_bytes(desc.Format) * desc.NumChannels;
  r.row = desc.Width * r.elem;
  r.rows = desc.Height ? desc.Height : 1;
  r.slices = desc.Depth ? desc.Depth : 1;
  r.device = current_device(s);
  r.mem = current(s).memory().alloc(r.row * r.rows * r.slices);
  const uintptr_t h = make_handle(s, kTagArray);
  s.arrays[h] = r;
  return h;
}
}  // namespace

VGPU_EXPORT CUresult cuArray3DCreate_v2(CUarray* out, const CUDA_ARRAY3D_DESCRIPTOR* desc) {
  return api("cuArray3DCreate", true, false, [&](ShimState& s) {
    if (!out || !desc || !valid_array(*desc)) return CUDA_ERROR_INVALID_VALUE;
    *out = reinterpret_cast<CUarray>(create_array(s, *desc));
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuArray3DCreate(CUarray* out, const CUDA_ARRAY3D_DESCRIPTOR* desc) {
  return cuArray3DCreate_v2(out, desc);
}
VGPU_EXPORT CUresult cuArrayCreate_v2(CUarray* out, const CUDA_ARRAY_DESCRIPTOR* desc) {
  if (!desc) return CUDA_ERROR_INVALID_VALUE;
  CUDA_ARRAY3D_DESCRIPTOR d{};
  d.Width = desc->Width;
  d.Height = desc->Height;
  d.Format = desc->Format;
  d.NumChannels = desc->NumChannels;
  return cuArray3DCreate_v2(out, &d);
}
VGPU_EXPORT CUresult cuArrayCreate(CUarray* out, const CUDA_ARRAY_DESCRIPTOR* desc) {
  return cuArrayCreate_v2(out, desc);
}
VGPU_EXPORT CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR* desc, CUarray a) {
  return api("cuArray3DGetDescriptor", true, false, [&](ShimState& s) {
    if (s.retired.count(reinterpret_cast<uintptr_t>(a))) return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    const ArrayRec* r = array_rec(s, a);
    if (!desc || !r) return CUDA_ERROR_INVALID_VALUE;
    *desc = r->desc;
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR* desc, CUarray a) {
  return cuArray3DGetDescriptor_v2(desc, a);
}
// The 2D descriptor of any array, a 3D one included (the card answers for both).
VGPU_EXPORT CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR* desc, CUarray a) {
  CUDA_ARRAY3D_DESCRIPTOR d{};
  if (!desc) return CUDA_ERROR_INVALID_VALUE;
  if (CUresult r = cuArray3DGetDescriptor_v2(&d, a)) return r;
  desc->Width = d.Width;
  desc->Height = d.Height;
  desc->Format = d.Format;
  desc->NumChannels = d.NumChannels;
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR* desc, CUarray a) {
  return cuArrayGetDescriptor_v2(desc, a);
}
VGPU_EXPORT CUresult cuArrayDestroy(CUarray a) {
  return api("cuArrayDestroy", true, false, [&](ShimState& s) {
    auto it = s.arrays.find(reinterpret_cast<uintptr_t>(a));
    if (it == s.arrays.end()) return CUDA_ERROR_INVALID_VALUE;
    // A mipmapped array's level belongs to it: the card reports success and
    // the level stays, as cuMipmappedArrayGetLevel still hands it out.
    if (it->second.mip_level) return CUDA_SUCCESS;
    s.rt->device(it->second.device).memory().free(it->second.mem);
    s.arrays.erase(it);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuMemcpyHtoA_v2(CUarray dst, size_t off, const void* src, size_t n) {
  return api("cuMemcpyHtoA", true, false, [&](ShimState& s) {
    CUdeviceptr at = 0;
    if (!src && n) return CUDA_ERROR_INVALID_VALUE;
    if (CUresult r = linear_part(s, dst, off, n, &at)) return r;
    dev_write(s, at, src, n);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemcpyHtoA(CUarray d, size_t off, const void* src, size_t n) {
  return cuMemcpyHtoA_v2(d, off, src, n);
}
VGPU_EXPORT CUresult cuMemcpyAtoH_v2(void* dst, CUarray src, size_t off, size_t n) {
  return api("cuMemcpyAtoH", true, false, [&](ShimState& s) {
    CUdeviceptr at = 0;
    if (!dst && n) return CUDA_ERROR_INVALID_VALUE;
    if (CUresult r = linear_part(s, src, off, n, &at)) return r;
    dev_read(s, dst, at, n);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemcpyAtoH(void* dst, CUarray src, size_t off, size_t n) {
  return cuMemcpyAtoH_v2(dst, src, off, n);
}
VGPU_EXPORT CUresult cuMemcpyDtoA_v2(CUarray dst, size_t off, CUdeviceptr src, size_t n) {
  return api("cuMemcpyDtoA", true, false, [&](ShimState& s) {
    CUdeviceptr at = 0;
    if (CUresult r = linear_part(s, dst, off, n, &at)) return r;
    std::vector<uint8_t> tmp(n);
    dev_read(s, tmp.data(), src, n);
    dev_write(s, at, tmp.data(), n);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemcpyDtoA(CUarray d, size_t off, CUdeviceptr src, size_t n) {
  return cuMemcpyDtoA_v2(d, off, src, n);
}
VGPU_EXPORT CUresult cuMemcpyAtoD_v2(CUdeviceptr dst, CUarray src, size_t off, size_t n) {
  return api("cuMemcpyAtoD", true, false, [&](ShimState& s) {
    CUdeviceptr at = 0;
    if (CUresult r = linear_part(s, src, off, n, &at)) return r;
    std::vector<uint8_t> tmp(n);
    dev_read(s, tmp.data(), at, n);
    dev_write(s, dst, tmp.data(), n);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemcpyAtoD(CUdeviceptr d, CUarray src, size_t off, size_t n) {
  return cuMemcpyAtoD_v2(d, src, off, n);
}

// Bytes between two arrays' first rows, as cuMemcpyHtoA and the rest take them.
VGPU_EXPORT CUresult cuMemcpyAtoA_v2(CUarray dst, size_t dst_off, CUarray src, size_t src_off, size_t n) {
  return api("cuMemcpyAtoA", true, false, [&](ShimState& s) {
    CUdeviceptr from = 0, to = 0;
    if (CUresult r = linear_part(s, src, src_off, n, &from)) return r;
    if (CUresult r = linear_part(s, dst, dst_off, n, &to)) return r;
    std::vector<uint8_t> tmp(n);
    dev_read(s, tmp.data(), from, n);
    dev_write(s, to, tmp.data(), n);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemcpyAtoA(CUarray d, size_t doff, CUarray src, size_t soff, size_t n) {
  return cuMemcpyAtoA_v2(d, doff, src, soff, n);
}
VGPU_EXPORT CUresult cuMemcpyHtoAAsync_v2(CUarray dst, size_t off, const void* src, size_t n, CUstream) {
  return cuMemcpyHtoA_v2(dst, off, src, n);
}
VGPU_EXPORT CUresult cuMemcpyHtoAAsync(CUarray dst, size_t off, const void* src, size_t n, CUstream st) {
  return cuMemcpyHtoAAsync_v2(dst, off, src, n, st);
}
VGPU_EXPORT CUresult cuMemcpyAtoHAsync_v2(void* dst, CUarray src, size_t off, size_t n, CUstream) {
  return cuMemcpyAtoH_v2(dst, src, off, n);
}
VGPU_EXPORT CUresult cuMemcpyAtoHAsync(void* dst, CUarray src, size_t off, size_t n, CUstream st) {
  return cuMemcpyAtoHAsync_v2(dst, src, off, n, st);
}

/* ---- mipmapped arrays ----
 * One ordinary array per level, each max(1, size >> level) along the
 * dimensions the descriptor has (a layered or cubemap array keeps its layer
 * count), with the level 0 descriptor's format, channels and flags. As an
 * RTX 3060 answers: a level count of 0 makes one level, one past the full
 * chain is cut to the chain (floor(log2(largest)) + 1 levels), a level past
 * the last is INVALID_VALUE, and a destroyed mipmapped array -- or one of its
 * levels -- is CUDA_ERROR_CONTEXT_IS_DESTROYED. */

VGPU_EXPORT CUresult cuMipmappedArrayCreate(CUmipmappedArray* out, const CUDA_ARRAY3D_DESCRIPTOR* desc,
                                            unsigned int numLevels) {
  return api("cuMipmappedArrayCreate", true, false, [&](ShimState& s) {
    if (!out || !desc || !valid_array(*desc)) return CUDA_ERROR_INVALID_VALUE;
    const bool layers = desc->Flags & (CUDA_ARRAY3D_LAYERED | CUDA_ARRAY3D_CUBEMAP);
    const size_t largest = std::max({desc->Width, desc->Height, layers ? size_t{0} : desc->Depth});
    unsigned full = 1;
    while ((size_t{1} << full) <= largest) ++full;
    const unsigned levels = std::clamp(numLevels, 1u, full);
    MipmapRec rec;
    for (unsigned l = 0; l < levels; ++l) {
      CUDA_ARRAY3D_DESCRIPTOR d = *desc;
      d.Width = std::max<size_t>(1, desc->Width >> l);
      d.Height = desc->Height ? std::max<size_t>(1, desc->Height >> l) : 0;
      if (!layers) d.Depth = desc->Depth ? std::max<size_t>(1, desc->Depth >> l) : 0;
      const uintptr_t h = create_array(s, d);
      s.arrays[h].mip_level = true;
      rec.levels.push_back(h);
    }
    const uintptr_t h = make_handle(s, kTagArray);
    s.mipmaps[h] = std::move(rec);
    *out = reinterpret_cast<CUmipmappedArray>(h);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuMipmappedArrayGetLevel(CUarray* level, CUmipmappedArray mm, unsigned int l) {
  return api("cuMipmappedArrayGetLevel", true, false, [&](ShimState& s) {
    if (!level) return CUDA_ERROR_INVALID_VALUE;
    const uintptr_t h = reinterpret_cast<uintptr_t>(mm);
    if (s.retired.count(h)) return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    auto it = s.mipmaps.find(h);
    if (it == s.mipmaps.end()) return CUDA_ERROR_INVALID_HANDLE;
    if (l >= it->second.levels.size()) return CUDA_ERROR_INVALID_VALUE;
    *level = reinterpret_cast<CUarray>(it->second.levels[l]);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuMipmappedArrayDestroy(CUmipmappedArray mm) {
  return api("cuMipmappedArrayDestroy", true, false, [&](ShimState& s) {
    const uintptr_t h = reinterpret_cast<uintptr_t>(mm);
    if (s.retired.count(h)) return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    auto it = s.mipmaps.find(h);
    if (it == s.mipmaps.end()) return CUDA_ERROR_INVALID_HANDLE;
    for (uintptr_t lh : it->second.levels) {
      auto a = s.arrays.find(lh);
      s.rt->device(a->second.device).memory().free(a->second.mem);
      s.arrays.erase(a);
      s.retired.insert(lh);
    }
    s.mipmaps.erase(it);
    s.retired.insert(h);
    return CUDA_SUCCESS;
  });
}

/* ---- texture and surface references (deprecated) ----
 * A kernel reads a texture reference its module declares, and CUDA 12's
 * compilers no longer emit them: there is none for cuModuleGetTexRef or
 * cuModuleGetSurfRef to find (CUDA_ERROR_NOT_FOUND, as the card answers a name
 * the module lacks), so nothing ever samples through one. What remains is the
 * reference cuTexRefCreate makes, whose settings are kept and read back with
 * the checks an RTX 3060 applies. */

namespace {
TexRefRec* texref(ShimState& s, CUtexref t) {
  auto it = s.texrefs.find(reinterpret_cast<uintptr_t>(t));
  return it == s.texrefs.end() ? nullptr : &it->second;
}
// Runs `body` on the texture reference, or answers INVALID_HANDLE.
template <class F>
CUresult with_texref(const char* name, CUtexref t, F&& body) {
  return api(name, true, false, [&](ShimState& s) -> CUresult {
    TexRefRec* r = texref(s, t);
    return r ? body(s, *r) : CUDA_ERROR_INVALID_HANDLE;
  });
}
}  // namespace

VGPU_EXPORT CUresult cuModuleGetTexRef(CUtexref* out, CUmodule hmod, const char*) {
  return api("cuModuleGetTexRef", true, false, [&](ShimState& s) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    if (!s.modules.count(reinterpret_cast<uintptr_t>(hmod))) return CUDA_ERROR_INVALID_HANDLE;
    return CUDA_ERROR_NOT_FOUND;
  });
}
VGPU_EXPORT CUresult cuModuleGetSurfRef(CUsurfref* out, CUmodule hmod, const char*) {
  return api("cuModuleGetSurfRef", true, false, [&](ShimState& s) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    if (!s.modules.count(reinterpret_cast<uintptr_t>(hmod))) return CUDA_ERROR_INVALID_HANDLE;
    return CUDA_ERROR_NOT_FOUND;
  });
}
// No surface reference exists to name.
VGPU_EXPORT CUresult cuSurfRefSetArray(CUsurfref, CUarray, unsigned int) { return CUDA_ERROR_INVALID_HANDLE; }
VGPU_EXPORT CUresult cuSurfRefGetArray(CUarray*, CUsurfref) { return CUDA_ERROR_INVALID_HANDLE; }

VGPU_EXPORT CUresult cuTexRefCreate(CUtexref* out) {
  return api("cuTexRefCreate", true, false, [&](ShimState& s) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    const uintptr_t h = make_handle(s, kTagArray);
    s.texrefs[h] = TexRefRec{};
    *out = reinterpret_cast<CUtexref>(h);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuTexRefDestroy(CUtexref t) {
  return api("cuTexRefDestroy", true, false, [&](ShimState& s) {
    return s.texrefs.erase(reinterpret_cast<uintptr_t>(t)) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
  });
}
// Dimension 0 to 2; wrap, clamp, mirror or border.
VGPU_EXPORT CUresult cuTexRefSetAddressMode(CUtexref t, int dim, int mode) {
  return with_texref("cuTexRefSetAddressMode", t, [&](ShimState&, TexRefRec& r) {
    if (dim < 0 || dim > 2 || mode < 0 || mode > 3) return CUDA_ERROR_INVALID_VALUE;
    r.address[dim] = mode;
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuTexRefGetAddressMode(int* mode, CUtexref t, int dim) {
  return with_texref("cuTexRefGetAddressMode", t, [&](ShimState&, TexRefRec& r) {
    if (!mode || dim < 0 || dim > 2) return CUDA_ERROR_INVALID_VALUE;
    *mode = r.address[dim];
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuTexRefSetFilterMode(CUtexref t, int mode) {
  return with_texref("cuTexRefSetFilterMode", t, [&](ShimState&, TexRefRec& r) {
    if (mode < 0 || mode > 1) return CUDA_ERROR_INVALID_VALUE;
    r.filter = mode;
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuTexRefGetFilterMode(int* mode, CUtexref t) {
  return with_texref("cuTexRefGetFilterMode", t, [&](ShimState&, TexRefRec& r) {
    if (!mode) return CUDA_ERROR_INVALID_VALUE;
    *mode = r.filter;
    return CUDA_SUCCESS;
  });
}
// The CU_TRSF_* bits and no others.
VGPU_EXPORT CUresult cuTexRefSetFlags(CUtexref t, unsigned int flags) {
  return with_texref("cuTexRefSetFlags", t, [&](ShimState&, TexRefRec& r) {
    const unsigned known = CU_TRSF_READ_AS_INTEGER | CU_TRSF_NORMALIZED_COORDINATES | CU_TRSF_SRGB |
                           CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION | CU_TRSF_SEAMLESS_CUBEMAP;
    if (flags & ~known) return CUDA_ERROR_INVALID_VALUE;
    r.flags = flags;
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuTexRefGetFlags(unsigned int* flags, CUtexref t) {
  return with_texref("cuTexRefGetFlags", t, [&](ShimState&, TexRefRec& r) {
    if (!flags) return CUDA_ERROR_INVALID_VALUE;
    *flags = r.flags;
    return CUDA_SUCCESS;
  });
}
// 1, 2 or 4 channels. The format is kept as given: the card takes one no
// header names.
VGPU_EXPORT CUresult cuTexRefSetFormat(CUtexref t, int format, int channels) {
  return with_texref("cuTexRefSetFormat", t, [&](ShimState&, TexRefRec& r) {
    if (channels != 1 && channels != 2 && channels != 4) return CUDA_ERROR_INVALID_VALUE;
    r.format = format;
    r.channels = channels;
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuTexRefGetFormat(int* format, int* channels, CUtexref t) {
  return with_texref("cuTexRefGetFormat", t, [&](ShimState&, TexRefRec& r) {
    if (!format && !channels) return CUDA_ERROR_INVALID_VALUE;
    if (format) *format = r.format;
    if (channels) *channels = r.channels;
    return CUDA_SUCCESS;
  });
}
// Linear memory, bound at the texture alignment (512 bytes) at or below the
// address; the distance above it is returned as the offset a kernel adds.
VGPU_EXPORT CUresult cuTexRefSetAddress_v2(size_t* offset, CUtexref t, CUdeviceptr dptr, size_t) {
  return with_texref("cuTexRefSetAddress", t, [&](ShimState& s, TexRefRec& r) {
    const CUdeviceptr align = static_cast<CUdeviceptr>(extra_attribute(current(s).profile(), 14));
    r.address_base = dptr / align * align;
    r.array = nullptr;
    if (offset) *offset = static_cast<size_t>(dptr - r.address_base);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuTexRefSetAddress(size_t* offset, CUtexref t, CUdeviceptr dptr, size_t bytes) {
  return cuTexRefSetAddress_v2(offset, t, dptr, bytes);
}
// Pitched linear memory: the address at the texture alignment (512 bytes),
// the pitch at the texture pitch alignment (32 bytes), and a pitch that holds
// a row.
VGPU_EXPORT CUresult cuTexRefSetAddress2D_v3(CUtexref t, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr,
                                             size_t pitch) {
  return with_texref("cuTexRefSetAddress2D", t, [&](ShimState& s, TexRefRec& r) {
    const vgpu::DeviceProfile& p = current(s).profile();
    const size_t align = static_cast<size_t>(extra_attribute(p, 14));        // TEXTURE_ALIGNMENT
    const size_t pitch_align = static_cast<size_t>(extra_attribute(p, 51));  // TEXTURE_PITCH_ALIGNMENT
    if (!desc || dptr % align || pitch % pitch_align ||
        pitch < desc->Width * format_bytes(desc->Format) * desc->NumChannels)
      return CUDA_ERROR_INVALID_VALUE;
    r.address_base = dptr;
    r.array = nullptr;
    std::memcpy(&r.format, &desc->Format, sizeof r.format);
    r.channels = static_cast<int>(desc->NumChannels);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuTexRefSetAddress2D_v2(CUtexref t, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr,
                                             size_t pitch) {
  return cuTexRefSetAddress2D_v3(t, desc, dptr, pitch);
}
VGPU_EXPORT CUresult cuTexRefSetAddress2D(CUtexref t, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr,
                                          size_t pitch) {
  return cuTexRefSetAddress2D_v3(t, desc, dptr, pitch);
}
// Bound to linear memory only; bound to an array, or to nothing, is INVALID_VALUE.
VGPU_EXPORT CUresult cuTexRefGetAddress_v2(CUdeviceptr* out, CUtexref t) {
  return with_texref("cuTexRefGetAddress", t, [&](ShimState&, TexRefRec& r) {
    if (!out || !r.address_base) return CUDA_ERROR_INVALID_VALUE;
    *out = r.address_base;
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuTexRefGetAddress(CUdeviceptr* out, CUtexref t) { return cuTexRefGetAddress_v2(out, t); }
// CU_TRSA_OVERRIDE_FORMAT, which takes the array's format, is the one flag.
VGPU_EXPORT CUresult cuTexRefSetArray(CUtexref t, CUarray a, unsigned int flags) {
  return with_texref("cuTexRefSetArray", t, [&](ShimState& s, TexRefRec& r) {
    const ArrayRec* arr = array_rec(s, a);
    if (!arr || (flags & ~unsigned{CU_TRSA_OVERRIDE_FORMAT})) return CUDA_ERROR_INVALID_VALUE;
    r.array = a;
    r.address_base = 0;
    if (flags & CU_TRSA_OVERRIDE_FORMAT) {
      std::memcpy(&r.format, &arr->desc.Format, sizeof r.format);
      r.channels = static_cast<int>(arr->desc.NumChannels);
    }
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuTexRefGetArray(CUarray* out, CUtexref t) {
  return with_texref("cuTexRefGetArray", t, [&](ShimState&, TexRefRec& r) {
    if (!out || !r.array) return CUDA_ERROR_INVALID_VALUE;
    *out = r.array;
    return CUDA_SUCCESS;
  });
}

/* ---- texture and surface objects ----
 * The same objects cudaCreateTextureObject makes, described through the
 * driver's structs and kept in the device's table, where a kernel's texture
 * and surface instructions find them. The handles come from a range of their
 * own, far above the runtime's, since the two libraries share each device's
 * table. What the runtime refuses is refused here too: resource views and
 * anisotropic filtering, and the packed and block-compressed formats. */

namespace {
// CUDA_RESOURCE_DESC and CUDA_TEXTURE_DESC, as cuda.h lays them out.
struct ResourceDescABI {
  int type;   // CU_RESOURCE_TYPE_ARRAY, _MIPMAPPED_ARRAY, _LINEAR, _PITCH2D
  union {
    struct { CUarray array; } array;
    struct { CUmipmappedArray mipmap; } mipmap;
    struct { CUdeviceptr ptr; CUarray_format format; unsigned channels; size_t bytes; } linear;
    struct {
      CUdeviceptr ptr;
      CUarray_format format;
      unsigned channels;
      size_t width, height, pitch;
    } pitch2d;
    int reserved[32];
  } res;
  unsigned flags;
};
struct TextureDescABI {
  int address[3];
  int filter;
  unsigned flags;
  unsigned max_anisotropy;
  int mip_filter;
  float mip_bias, mip_min, mip_max;
  float border[4];
  int reserved[12];
};

uint64_t next_texobj = uint64_t{1} << 40;

// A driver format's channel kind and width, or false for one this does not read.
bool format_kind(const CUarray_format& f, vgpu::exec::ChannelKind* kind, uint32_t* bits) {
  unsigned raw;
  std::memcpy(&raw, &f, sizeof raw);
  switch (raw) {
    case CU_AD_FORMAT_UNSIGNED_INT8: *kind = vgpu::exec::ChannelKind::Unsigned; *bits = 8; return true;
    case CU_AD_FORMAT_UNSIGNED_INT16: *kind = vgpu::exec::ChannelKind::Unsigned; *bits = 16; return true;
    case CU_AD_FORMAT_UNSIGNED_INT32: *kind = vgpu::exec::ChannelKind::Unsigned; *bits = 32; return true;
    case CU_AD_FORMAT_SIGNED_INT8: *kind = vgpu::exec::ChannelKind::Signed; *bits = 8; return true;
    case CU_AD_FORMAT_SIGNED_INT16: *kind = vgpu::exec::ChannelKind::Signed; *bits = 16; return true;
    case CU_AD_FORMAT_SIGNED_INT32: *kind = vgpu::exec::ChannelKind::Signed; *bits = 32; return true;
    case CU_AD_FORMAT_HALF: *kind = vgpu::exec::ChannelKind::Float; *bits = 16; return true;
    case CU_AD_FORMAT_FLOAT: *kind = vgpu::exec::ChannelKind::Float; *bits = 32; return true;
  }
  return false;
}

// The format part of a descriptor: channels of one kind and width.
CUresult set_format(vgpu::exec::TextureDesc* d, const CUarray_format& format, unsigned channels) {
  uint32_t bits = 0;
  if (!format_kind(format, &d->kind, &bits)) return CUDA_ERROR_NOT_SUPPORTED;
  if (channels != 1 && channels != 2 && channels != 4) return CUDA_ERROR_INVALID_VALUE;
  d->channels = channels;
  for (unsigned c = 0; c < 4; ++c) d->channel_bits[c] = c < channels ? bits : 0;
  d->texel_bytes = bits / 8 * channels;
  return CUDA_SUCCESS;
}

// An array's shape, as cudaCreateTextureObject reads a cudaArray's.
void set_array_shape(vgpu::exec::TextureDesc* d, const ArrayRec& a) {
  d->base = a.mem;
  d->width = static_cast<uint32_t>(a.desc.Width);
  d->height = static_cast<uint32_t>(a.desc.Height);
  d->depth = static_cast<uint32_t>(a.desc.Depth);
  d->cubemap = a.desc.Flags & CUDA_ARRAY3D_CUBEMAP;
  if (a.desc.Flags & CUDA_ARRAY3D_LAYERED) d->layers = static_cast<uint32_t>(d->cubemap ? a.desc.Depth / 6 : a.desc.Depth);
  if (d->cubemap || d->layers) d->depth = 0;
  d->pitch_bytes = static_cast<uint32_t>(a.row);
  d->from_array = true;
}

CUresult describe_resource(ShimState& s, const ResourceDescABI& r, vgpu::exec::TextureDesc* d) {
  switch (r.type) {
    case 0: {   // CU_RESOURCE_TYPE_ARRAY
      const ArrayRec* a = array_rec(s, r.res.array.array);
      if (!a) return CUDA_ERROR_INVALID_HANDLE;
      set_array_shape(d, *a);
      return set_format(d, a->desc.Format, a->desc.NumChannels);
    }
    case 1: {   // CU_RESOURCE_TYPE_MIPMAPPED_ARRAY
      auto it = s.mipmaps.find(reinterpret_cast<uintptr_t>(r.res.mipmap.mipmap));
      if (it == s.mipmaps.end()) return CUDA_ERROR_INVALID_HANDLE;
      const auto& lv = it->second.levels;
      if (lv.size() > 17) return CUDA_ERROR_NOT_SUPPORTED;
      const ArrayRec& a = s.arrays.at(lv[0]);
      set_array_shape(d, a);
      d->mip_levels = static_cast<uint32_t>(lv.size());
      for (size_t l = 0; l < lv.size(); ++l) d->level_base[l] = s.arrays.at(lv[l]).mem;
      return set_format(d, a.desc.Format, a.desc.NumChannels);
    }
    case 2: {   // CU_RESOURCE_TYPE_LINEAR
      if (!r.res.linear.ptr) return CUDA_ERROR_INVALID_VALUE;
      if (CUresult e = set_format(d, r.res.linear.format, r.res.linear.channels)) return e;
      d->base = r.res.linear.ptr;
      d->width = static_cast<uint32_t>(r.res.linear.bytes / d->texel_bytes);
      d->pitch_bytes = 0;
      return CUDA_SUCCESS;
    }
    case 3: {   // CU_RESOURCE_TYPE_PITCH2D
      if (!r.res.pitch2d.ptr) return CUDA_ERROR_INVALID_VALUE;
      if (CUresult e = set_format(d, r.res.pitch2d.format, r.res.pitch2d.channels)) return e;
      d->base = r.res.pitch2d.ptr;
      d->width = static_cast<uint32_t>(r.res.pitch2d.width);
      d->height = static_cast<uint32_t>(r.res.pitch2d.height);
      d->pitch_bytes = static_cast<uint32_t>(r.res.pitch2d.pitch);
      return CUDA_SUCCESS;
    }
    default:
      return CUDA_ERROR_INVALID_VALUE;
  }
}

CUresult make_object(const char* name, unsigned long long* out, const void* res, const void* tex,
                     const void* view, vgpu::exec::TexKind kind) {
  return api(name, true, false, [&](ShimState& s) -> CUresult {
    if (!out || !res) return CUDA_ERROR_INVALID_VALUE;
    if (view) return CUDA_ERROR_NOT_SUPPORTED;   // a resource view reinterprets the format
    vgpu::exec::TextureDesc d;
    d.object = kind;
    if (CUresult e = describe_resource(s, *static_cast<const ResourceDescABI*>(res), &d)) return e;
    if (const auto* t = static_cast<const TextureDescABI*>(tex)) {
      for (int i = 0; i < 3; ++i) {
        if (t->address[i] < 0 || t->address[i] > 3) return CUDA_ERROR_INVALID_VALUE;
        d.address[i] = static_cast<vgpu::exec::TexAddress>(t->address[i]);   // the same order
      }
      if (t->max_anisotropy > 1) return CUDA_ERROR_NOT_SUPPORTED;
      d.filter = t->filter == 1 ? vgpu::exec::TexFilter::Linear : vgpu::exec::TexFilter::Point;
      d.normalized_coords = t->flags & CU_TRSF_NORMALIZED_COORDINATES;
      d.srgb = t->flags & CU_TRSF_SRGB;
      // Integer texels come back as floats in [0, 1] or [-1, 1] unless
      // CU_TRSF_READ_AS_INTEGER keeps them integers -- the driver's default is
      // the opposite of the runtime's. Only 8- and 16-bit channels promote.
      d.read_as_normalized_float = !(t->flags & CU_TRSF_READ_AS_INTEGER) &&
                                   d.kind != vgpu::exec::ChannelKind::Float && d.channel_bits[0] < 32;
      static_assert(sizeof d.border_bits == sizeof t->border);
      std::memcpy(d.border_bits, t->border, sizeof d.border_bits);
      auto q = [](float v) { return static_cast<int32_t>(std::trunc(std::clamp(v, -1e6f, 1e6f) * 256)); };
      d.mip_filter = t->mip_filter == 1 ? vgpu::exec::TexFilter::Linear : vgpu::exec::TexFilter::Point;
      d.mip_bias = q(t->mip_bias);
      d.mip_min = q(t->mip_min);
      d.mip_max = q(t->mip_max);
    }
    const uint64_t handle = next_texobj++;
    current(s).textures()[handle] = d;
    *out = handle;
    return CUDA_SUCCESS;
  });
}

CUresult destroy_object(const char* name, unsigned long long obj) {
  return api(name, true, false, [&](ShimState& s) {
    for (int dev = 0; dev < s.rt->device_count(); ++dev)
      if (s.rt->device(dev).textures().erase(obj)) return CUDA_SUCCESS;
    return CUDA_ERROR_INVALID_VALUE;
  });
}
}  // namespace

VGPU_EXPORT CUresult cuTexObjectCreate(unsigned long long* out, const void* res, const void* tex,
                                       const void* view) {
  return make_object("cuTexObjectCreate", out, res, tex, view, vgpu::exec::TexKind::Texture);
}
VGPU_EXPORT CUresult cuTexObjectDestroy(unsigned long long obj) {
  return destroy_object("cuTexObjectDestroy", obj);
}
VGPU_EXPORT CUresult cuSurfObjectCreate(unsigned long long* out, const void* res) {
  return make_object("cuSurfObjectCreate", out, res, nullptr, nullptr, vgpu::exec::TexKind::Surface);
}
VGPU_EXPORT CUresult cuSurfObjectDestroy(unsigned long long obj) {
  return destroy_object("cuSurfObjectDestroy", obj);
}

VGPU_EXPORT CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D* p) {
  return api("cuMemcpy3D", true, false, [&](ShimState& s) {
    if (!p) return CUDA_ERROR_INVALID_VALUE;
    return copy3d(s, *p);
  });
}
VGPU_EXPORT CUresult cuMemcpy3D(const CUDA_MEMCPY3D* p) { return cuMemcpy3D_v2(p); }
VGPU_EXPORT CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D* p, CUstream) { return cuMemcpy3D_v2(p); }
VGPU_EXPORT CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D* p, CUstream st) { return cuMemcpy3DAsync_v2(p, st); }
VGPU_EXPORT CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D* c) {
  return api("cuMemcpy2D", true, false, [&](ShimState& s) {
    if (!c) return CUDA_ERROR_INVALID_VALUE;
    return copy3d(s, as3d(*c));
  });
}
VGPU_EXPORT CUresult cuMemcpy2D(const CUDA_MEMCPY2D* c) { return cuMemcpy2D_v2(c); }
VGPU_EXPORT CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D* c) { return cuMemcpy2D_v2(c); }
VGPU_EXPORT CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D* c) { return cuMemcpy2D_v2(c); }
VGPU_EXPORT CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D* c, CUstream) { return cuMemcpy2D_v2(c); }
VGPU_EXPORT CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D* c, CUstream st) { return cuMemcpy2DAsync_v2(c, st); }

// Stream-ordered allocation: the stream is synchronous, so the memory is
// ready at once. A request for 0 bytes succeeds, as the card's does.
VGPU_EXPORT CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream) {
  if (!dptr) return CUDA_ERROR_INVALID_VALUE;
  if (bytesize == 0) {
    *dptr = 0;
    return CUDA_SUCCESS;
  }
  return cuMemAlloc_v2(dptr, bytesize);
}
VGPU_EXPORT CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream) {
  if (dptr == 0) return CUDA_SUCCESS;
  return cuMemFree_v2(dptr);
}

/* ---- pitched memory, module globals, context configuration ----
 * Each answers as an RTX 3060's and an RTX 3080 Ti's drivers do (they agree). */

// Rows padded to 512 bytes; the element size must be 4, 8 or 16 and neither
// dimension 0.
VGPU_EXPORT CUresult cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pitch, size_t width, size_t height,
                                        unsigned int elem) {
  return api("cuMemAllocPitch", true, false, [&](ShimState& s) {
    if (!dptr || !pitch || width == 0 || height == 0 || (elem != 4 && elem != 8 && elem != 16))
      return CUDA_ERROR_INVALID_VALUE;
    *pitch = (width + 511) / 512 * 512;
    *dptr = current(s).memory().alloc(*pitch * height);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemAllocPitch(CUdeviceptr* dptr, size_t* pitch, size_t width, size_t height,
                                     unsigned int elem) {
  return cuMemAllocPitch_v2(dptr, pitch, width, height, elem);
}

// A module's __device__ variable: its address and size, either of which may be
// omitted but not both. A name the module does not declare is NOT_FOUND; a
// kernel's name is INVALID_VALUE, as on the card.
VGPU_EXPORT CUresult cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name) {
  return api("cuModuleGetGlobal", true, false, [&](ShimState& s) {
    if ((!dptr && !bytes) || !name) return CUDA_ERROR_INVALID_VALUE;
    auto it = s.modules.find(reinterpret_cast<uintptr_t>(hmod));
    if (it == s.modules.end()) return CUDA_ERROR_INVALID_HANDLE;
    const auto [dev, mid] = it->second;
    uint64_t addr = 0, size = 0;
    if (!s.rt->device(dev).global(mid, name, &addr, &size))
      return s.rt->device(dev).has_kernel(mid, name) ? CUDA_ERROR_INVALID_VALUE : CUDA_ERROR_NOT_FOUND;
    if (dptr) *dptr = addr;
    if (bytes) *bytes = static_cast<size_t>(size);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name) {
  return cuModuleGetGlobal_v2(dptr, bytes, hmod, name);
}

// The L1/shared split is a preference the simulator has no cache to apply to;
// it is kept and read back, as the card reads it back.
VGPU_EXPORT CUresult cuCtxSetCacheConfig(int config) {
  return api("cuCtxSetCacheConfig", true, false, [&](ShimState& s) {
    if (config < 0 || config > 3) return CUDA_ERROR_INVALID_VALUE;
    s.cache_config = config;
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuCtxGetCacheConfig(int* config) {
  return api("cuCtxGetCacheConfig", true, false, [&](ShimState& s) {
    if (!config) return CUDA_ERROR_INVALID_VALUE;
    *config = s.cache_config;
    return CUDA_SUCCESS;
  });
}
// Shared memory banks are four bytes wide on every GPU this simulates: a
// configuration is accepted and has no effect, and the answer stays
// CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE, as the card's does.
VGPU_EXPORT CUresult cuCtxSetSharedMemConfig(int config) {
  return config >= 0 && config <= 2 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
VGPU_EXPORT CUresult cuCtxGetSharedMemConfig(int* config) {
  if (!config) return CUDA_ERROR_INVALID_VALUE;
  *config = 1;
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuFuncSetSharedMemConfig(CUfunction, int config) {
  return config >= 0 && config <= 2 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

// A stream callback runs at once: the stream is synchronous, so the work
// queued before it is done. It is told the stream succeeded. Flags must be 0.
VGPU_EXPORT CUresult cuStreamAddCallback(CUstream stream, CUstreamCallback cb, void* user, unsigned int flags) {
  if (!cb || flags != 0) return CUDA_ERROR_INVALID_VALUE;
  cb(stream, CUDA_SUCCESS, user);
  return CUDA_SUCCESS;
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
// Streams are synchronous, so everything queued before the function has run
// by the time it is called, which is all a host function is promised.
VGPU_EXPORT CUresult cuLaunchHostFunc(CUstream, CUhostFn fn, void* user) {
  if (!fn) return CUDA_ERROR_INVALID_VALUE;
  fn(user);
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuStreamQuery(CUstream) { return dead_context(); }  // always idle
VGPU_EXPORT CUresult cuStreamWaitEvent(CUstream, void*, unsigned int) { return dead_context(); }
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

VGPU_EXPORT CUresult cuEventCreate(void** ev, unsigned int flags) {
  return api("cuEventCreate", true, false, [&](ShimState& s) {
    if (!ev) return CUDA_ERROR_INVALID_VALUE;
    uintptr_t h = make_handle(s, kTagEvent);
    s.events[h] = {};
    s.events[h].timing = !(flags & 0x2);   // CU_EVENT_DISABLE_TIMING
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
VGPU_EXPORT CUresult cuEventQuery(void*) { return dead_context(); }
VGPU_EXPORT CUresult cuEventSynchronize(void*) { return dead_context(); }
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
    // As the card answers: an event never recorded, or made with
    // CU_EVENT_DISABLE_TIMING, is an invalid handle here, not a bad value.
    if (!ms) return CUDA_ERROR_INVALID_VALUE;
    if (a == s.events.end() || b == s.events.end() || !a->second.recorded || !b->second.recorded ||
        !a->second.timing || !b->second.timing)
      return CUDA_ERROR_INVALID_HANDLE;
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
  if (const CUresult dead = dead_context()) return dead;
  if (!v) return CUDA_ERROR_INVALID_VALUE;
  switch (limit) {
    case 0: *v = 1024; break;              // STACK_SIZE
    case 1: *v = 1024 * 1024; break;       // PRINTF_FIFO_SIZE
    case 2: *v = 8 * 1024 * 1024; break;   // MALLOC_HEAP_SIZE
    default: *v = 0; break;
  }
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuCtxSetLimit(int, size_t) { return dead_context(); }
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
VGPU_EXPORT CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) { return cuDevicePrimaryCtxReset(dev); }

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

// A buffer too short for the id gets as much of it as fits, and
// INVALID_VALUE, as on the card.
VGPU_EXPORT CUresult cuDeviceGetPCIBusId(char* id, int len, CUdevice dev) {
  if (!id || len <= 0) return CUDA_ERROR_INVALID_VALUE;
  const int n = std::snprintf(id, static_cast<size_t>(len), "0000:%02x:00.0", dev + 1);
  return n < len ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

namespace {
// "[domain:]bus:device[.function]", each part hexadecimal (a "0x" prefix is
// taken), as an RTX 3060's driver reads it: "0000:01:00.0", "01:00.0",
// "1:0.0" and "0000:01:00" all name one device, and anything left over --
// trailing text, a trailing space -- makes the string invalid. False when it
// does not parse; a string that parses may still name no device.
bool parse_bus_id(const char* text, unsigned long out[4]) {
  std::vector<std::string> parts;
  std::string cur;
  for (const char* c = text; *c; ++c) {
    if (*c == ':') {
      parts.push_back(cur);
      cur.clear();
    } else {
      cur += *c;
    }
  }
  parts.push_back(cur);
  if (parts.size() != 2 && parts.size() != 3) return false;
  std::string fn = "0";
  if (const size_t dot = parts.back().find('.'); dot != std::string::npos) {
    fn = parts.back().substr(dot + 1);
    parts.back().resize(dot);
  }
  if (parts.size() == 2) parts.insert(parts.begin(), "0");
  parts.push_back(fn);
  for (int i = 0; i < 4; ++i) {
    if (parts[i].empty()) return false;
    char* end = nullptr;
    out[i] = std::strtoul(parts[i].c_str(), &end, 16);
    if (*end) return false;
  }
  return true;
}
}  // namespace

// The device a PCI bus id names, compared against the id cuDeviceGetPCIBusId
// gives each device. A well-formed id naming no device -- another bus, a
// function other than 0, a domain other than 0 -- is INVALID_DEVICE; a
// malformed one is INVALID_VALUE.
VGPU_EXPORT CUresult cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId) {
  return api("cuDeviceGetByPCIBusId", true, false, [&](ShimState& s) {
    unsigned long want[4];
    if (!dev || !pciBusId || !parse_bus_id(pciBusId, want)) return CUDA_ERROR_INVALID_VALUE;
    for (int d = 0; d < s.rt->device_count(); ++d) {
      char id[32];
      unsigned long have[4];
      cuDeviceGetPCIBusId(id, sizeof id, d);
      if (parse_bus_id(id, have) && std::equal(have, have + 4, want)) {
        *dev = d;
        return CUDA_SUCCESS;
      }
    }
    return CUDA_ERROR_INVALID_DEVICE;
  });
}

// Distinct simulated devices reach each other's memory, as the runtime's
// cudaDeviceCanAccessPeer answers; a device is not its own peer. (A pair of
// RTX 3060s under WSL answers 0 for each other: that host has no peer path.)
VGPU_EXPORT CUresult cuDeviceCanAccessPeer(int* can, CUdevice dev, CUdevice peer) {
  return api("cuDeviceCanAccessPeer", true, false, [&](ShimState& s) {
    if (!can) return CUDA_ERROR_INVALID_VALUE;
    const int n = s.rt->device_count();
    if (dev < 0 || dev >= n || peer < 0 || peer >= n) return CUDA_ERROR_INVALID_DEVICE;
    *can = dev != peer ? 1 : 0;
    return CUDA_SUCCESS;
  });
}

// Peer mappings are implicit here -- device address windows are disjoint and
// every kernel reaches all of them -- but whether one is enabled is state the
// documented answers depend on: enabling twice is PEER_ACCESS_ALREADY_ENABLED,
// disabling what was never enabled PEER_ACCESS_NOT_ENABLED, and a context on
// the same device as the current one is no peer (PEER_ACCESS_UNSUPPORTED, as
// the card answers). Flags must be 0.
VGPU_EXPORT CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int flags) {
  return api("cuCtxEnablePeerAccess", true, false, [&](ShimState& s) {
    if (flags != 0 || !peerContext) return CUDA_ERROR_INVALID_VALUE;
    const uintptr_t peer = check_handle(reinterpret_cast<uintptr_t>(peerContext), kTagCtx, "context");
    auto it = s.contexts.find(peer);
    if (it == s.contexts.end()) return CUDA_ERROR_INVALID_CONTEXT;
    if (it->second == current_device(s)) return CUDA_ERROR_PEER_ACCESS_UNSUPPORTED;
    if (!s.peer_access.emplace(ctx_stack().back(), peer).second) return CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
  return api("cuCtxDisablePeerAccess", true, false, [&](ShimState& s) {
    if (!peerContext) return CUDA_ERROR_INVALID_VALUE;
    const uintptr_t peer = check_handle(reinterpret_cast<uintptr_t>(peerContext), kTagCtx, "context");
    if (!s.contexts.count(peer)) return CUDA_ERROR_INVALID_CONTEXT;
    (void)current_device(s);
    if (!s.peer_access.erase({ctx_stack().back(), peer})) return CUDA_ERROR_PEER_ACCESS_NOT_ENABLED;
    return CUDA_SUCCESS;
  });
}

// A copy between two contexts' memory. Device address windows are disjoint,
// so each address names its own device and the contexts add nothing; the card
// takes a null one as well.
VGPU_EXPORT CUresult cuMemcpyPeer(CUdeviceptr dst, CUcontext, CUdeviceptr src, CUcontext, size_t n) {
  return cuMemcpyDtoD_v2(dst, src, n);
}
VGPU_EXPORT CUresult cuMemcpyPeerAsync(CUdeviceptr dst, CUcontext dctx, CUdeviceptr src, CUcontext sctx,
                                       size_t n, CUstream) {
  return cuMemcpyPeer(dst, dctx, src, sctx, n);
}

/* ---- pointer queries (expected probes: fail quietly, no stderr) ---- */

VGPU_EXPORT CUresult cuPointerGetAttribute(void* data, int attribute, CUdeviceptr ptr) {
  if (!data) return CUDA_ERROR_INVALID_VALUE;
  bool managed = false;
  {
    auto& s = state();
    std::lock_guard<std::recursive_mutex> g(s.mu);
    managed = s.initialized && managed_range(s, ptr, 1);
  }
  if (managed) {
    // What an RTX 3080 Ti's driver answers for a managed pointer: device
    // memory, managed, the same address on both sides. The context is
    // answered as it is for device memory below.
    switch (attribute) {
      case 1: *static_cast<void**>(data) = nullptr; return CUDA_SUCCESS;  // CONTEXT
      case 2: *static_cast<unsigned int*>(data) = 2; return CUDA_SUCCESS;  // MEMORY_TYPE: DEVICE
      case 3: *static_cast<CUdeviceptr*>(data) = ptr; return CUDA_SUCCESS;  // DEVICE_POINTER
      case 4: *static_cast<void**>(data) = reinterpret_cast<void*>(ptr); return CUDA_SUCCESS;  // HOST_POINTER
      case 8: *static_cast<int*>(data) = 1; return CUDA_SUCCESS;  // IS_MANAGED
      default: return CUDA_ERROR_INVALID_VALUE;
    }
  }
  bool dev = vgpu::is_device_va(ptr);
  switch (attribute) {
    case 8:  // CU_POINTER_ATTRIBUTE_IS_MANAGED
      if (!dev) return CUDA_ERROR_INVALID_VALUE;
      *static_cast<int*>(data) = 0;
      return CUDA_SUCCESS;
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

/* ---- occupancy ----
 * From the same register and shared-memory analysis the launch path uses, as
 * the runtime's cudaOccupancyMaxActiveBlocksPerMultiprocessor computes it. This
 * used to be a placeholder that answered at least one block whatever the
 * kernel, which is the one answer a block-size search cannot use. */

namespace {
// The function a CUfunction or a CUkernel names, or nullptr.
const FuncRec* occupancy_func(ShimState& s, CUfunction f) {
  uintptr_t fh = reinterpret_cast<uintptr_t>(f);
  if ((fh & 7) == kTagKernel) fh = kernel_to_function(s, fh);
  auto it = s.functions.find(fh);
  return it == s.functions.end() ? nullptr : &it->second;
}

// Blocks of `block` threads resident per multiprocessor. None, as on the
// card, when the block is larger than a block may be or asks for more shared
// memory than a block may have: 48 KiB with the static part, unless the
// kernel's CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES allows more.
int blocks_per_sm(ShimState& s, const FuncRec& f, int block, size_t dyn) {
  const vgpu::DeviceProfile& p = s.rt->device(f.device).profile();
  if (static_cast<uint32_t>(block) > p.limits.max_threads_per_block) return 0;
  const uint64_t static_bytes = f.fn->static_shared_size;
  const uint64_t limit = f.max_dynamic_shared >= 0
                             ? static_bytes + static_cast<uint64_t>(f.max_dynamic_shared)
                             : uint64_t{p.limits.shared_mem_per_block};
  if (static_bytes + dyn > limit) return 0;
  return static_cast<int>(vgpu::exec::kernel_resources(*f.fn, p, static_cast<uint32_t>(block),
                                                       static_cast<uint32_t>(dyn))
                              .occupancy.blocks_per_sm);
}
}  // namespace

VGPU_EXPORT CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* num, CUfunction f, int blockSize,
                                                                 size_t dyn) {
  return api("cuOccupancyMaxActiveBlocksPerMultiprocessor", true, false, [&](ShimState& s) {
    if (!num || blockSize <= 0) return CUDA_ERROR_INVALID_VALUE;
    const FuncRec* rec = occupancy_func(s, f);
    if (!rec) return CUDA_ERROR_INVALID_HANDLE;
    *num = blocks_per_sm(s, *rec, blockSize, dyn);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* num, CUfunction f,
                                                                          int blockSize, size_t dyn,
                                                                          unsigned int) {
  return cuOccupancyMaxActiveBlocksPerMultiprocessor(num, f, blockSize, dyn);
}

// The block size with the most threads resident per multiprocessor, and the
// grid that fills the device at it. The search is the one CUDA documents and
// cuda_runtime.h spells out for cudaOccupancyMaxPotentialBlockSize: every
// multiple of the warp size from the largest allowed block down (the limit
// itself first, when it is not a multiple), keeping the first size to reach
// the best occupancy and stopping at a full multiprocessor. Per-block dynamic
// shared memory comes from the callback when one is given, and is otherwise
// the constant. No block fits at all: 0 and 0, successfully, as on the card.
VGPU_EXPORT CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize,
                                                               CUfunction func,
                                                               CUoccupancyB2DSize blockToSmem,
                                                               size_t dynamicSMemSize, int blockSizeLimit,
                                                               unsigned int flags) {
  return api("cuOccupancyMaxPotentialBlockSize", true, false, [&](ShimState& s) {
    // CU_OCCUPANCY_DEFAULT or CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE, which
    // changes nothing without a cache.
    if (!minGridSize || !blockSize || blockSizeLimit < 0 || flags > 1) return CUDA_ERROR_INVALID_VALUE;
    const FuncRec* rec = func ? occupancy_func(s, func) : nullptr;
    if (!rec) return CUDA_ERROR_INVALID_HANDLE;
    const vgpu::DeviceProfile& p = s.rt->device(rec->device).profile();
    const int per_sm = static_cast<int>(p.limits.max_threads_per_sm);
    const int granularity = static_cast<int>(p.warp_size);
    const int dev_max = static_cast<int>(p.limits.max_threads_per_block);
    const int func_max = func_attribute(rec->fn, p, 0);   // MAX_THREADS_PER_BLOCK
    int limit = blockSizeLimit == 0 ? dev_max : blockSizeLimit;
    limit = std::min({limit, dev_max, func_max});
    const int aligned = (limit + granularity - 1) / granularity * granularity;
    int best_block = 0, best_blocks = 0, best = 0;
    for (int size = aligned; size > 0; size -= granularity) {
      const int try_size = std::min(size, limit);
      const size_t dyn = blockToSmem ? blockToSmem(try_size) : dynamicSMemSize;
      const int blocks = blocks_per_sm(s, *rec, try_size, dyn);
      if (blocks * try_size > best) {
        best_block = try_size;
        best_blocks = blocks;
        best = blocks * try_size;
      }
      if (best == per_sm) break;
    }
    *minGridSize = best_blocks * static_cast<int>(p.limits.multiprocessors);
    *blockSize = best_block;
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, CUfunction func,
                                                      CUoccupancyB2DSize blockToSmem, size_t dynamicSMemSize,
                                                      int blockSizeLimit) {
  return cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize, blockSize, func, blockToSmem,
                                                   dynamicSMemSize, blockSizeLimit, 0);
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
// Asked for by NVIDIA's NVRTC (13.0) when it finds a driver loaded. Refused,
// NVRTC compiles exactly as it does on a machine with no driver; given the
// generic stub table it returns success with an empty PTX. The NVRTC shim
// compiles through NVIDIA's NVRTC where it is installed, so this is the one
// table refused by name.
constexpr unsigned char kUuidNvrtcProbe[16] = {0xda, 0x91, 0x51, 0xd3, 0x3a, 0xe6, 0xcc, 0x41,
                                               0xa5, 0xc0, 0x4f, 0x26, 0xd5, 0x33, 0xe3, 0x28};

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
  if (std::memcmp(uuid, kUuidNvrtcProbe, 16) == 0) return nullptr;
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
   The same sharing the runtime API offers, through the same handles: an
   exported allocation is moved into a file in the machine directory, mapped
   MAP_SHARED and left at the same device address, and another process maps that
   file at an address of its own (vgpu/memory.hpp, nvidia/src/runtime_api.cpp).
   A handle from either API opens in either, as on a real driver.

   They also have to exist at all: a caller that looks the symbols up at startup
   -- Numba resolves cuIpcOpenMemHandle before it will report a device -- fails
   on the lookup rather than on the call, which reads as "no CUDA here". */

namespace {

constexpr uint32_t kIpcMagic = 0x43504956;   // "VIPC", as the runtime writes it
constexpr uint32_t kIpcVersion = 1;

struct IpcMemPayload {
  uint32_t magic;
  uint32_t version;
  uint64_t size;
  uint32_t device;
  uint32_t pid;
  char id[40];
};
static_assert(sizeof(IpcMemPayload) == 64, "an IPC handle is 64 bytes");

std::string ipc_path(const char* id) { return vgpu::telemetry::default_path() + "/ipc-" + id; }

std::mutex g_ipc_mu;
std::map<CUdeviceptr, int> g_ipc_open;   // imported pointer -> device

}  // namespace

VGPU_EXPORT CUresult cuIpcGetMemHandle(CUipcMemHandle* handle, CUdeviceptr ptr) {
  return api("cuIpcGetMemHandle", true, false, [&](ShimState& s) -> CUresult {
    if (!handle || !ptr) return CUDA_ERROR_INVALID_VALUE;
    int device = -1;
    for (int d = 0; d < s.rt->device_count(); ++d)
      if (s.rt->device(d).memory().owns(ptr)) device = d;
    if (device < 0) return CUDA_ERROR_INVALID_VALUE;
    static std::atomic<uint32_t> counter{0};
    char id[40] = {0};
    std::snprintf(id, sizeof id, "%x-d%x", static_cast<unsigned>(::getpid()),
                  counter.fetch_add(1) + 1);
    IpcMemPayload p{};
    p.magic = kIpcMagic;
    p.version = kIpcVersion;
    p.device = static_cast<uint32_t>(device);
    p.pid = static_cast<uint32_t>(::getpid());
    std::memcpy(p.id, id, sizeof p.id);
    p.size = s.rt->device(device).memory().share(ptr, ipc_path(id));
    std::memcpy(handle, &p, sizeof p);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuIpcOpenMemHandle_v2(CUdeviceptr* ptr, CUipcMemHandle handle,
                                           unsigned int flags) {
  return api("cuIpcOpenMemHandle", true, false, [&](ShimState& s) -> CUresult {
    if (!ptr) return CUDA_ERROR_INVALID_VALUE;
    // CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS is the only flag, and it is required.
    if (flags != 1) return CUDA_ERROR_INVALID_VALUE;
    IpcMemPayload p{};
    std::memcpy(&p, &handle, sizeof p);
    if (p.magic != kIpcMagic || p.version != kIpcVersion || !p.size)
      return CUDA_ERROR_INVALID_VALUE;
    if (p.pid == static_cast<uint32_t>(::getpid())) return CUDA_ERROR_INVALID_VALUE;
    char id[sizeof p.id + 1] = {0};
    std::memcpy(id, p.id, sizeof p.id);
    const int device = p.device < static_cast<uint32_t>(s.rt->device_count())
                           ? static_cast<int>(p.device) : 0;
    *ptr = s.rt->device(device).memory().adopt(ipc_path(id), p.size);
    std::lock_guard<std::mutex> lock(g_ipc_mu);
    g_ipc_open[*ptr] = device;
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuIpcOpenMemHandle(CUdeviceptr* ptr, CUipcMemHandle handle,
                                        unsigned int flags) {
  return cuIpcOpenMemHandle_v2(ptr, handle, flags);
}

VGPU_EXPORT CUresult cuIpcCloseMemHandle(CUdeviceptr ptr) {
  return api("cuIpcCloseMemHandle", true, false, [&](ShimState& s) -> CUresult {
    int device = -1;
    {
      std::lock_guard<std::mutex> lock(g_ipc_mu);
      auto it = g_ipc_open.find(ptr);
      if (it == g_ipc_open.end()) return CUDA_ERROR_INVALID_VALUE;
      device = it->second;
      g_ipc_open.erase(it);
    }
    s.rt->device(device).memory().abandon(ptr);
    return CUDA_SUCCESS;
  });
}

// An event handle carries its origin and nothing else: every operation here
// finishes before the call that started it returns, so an event whose handle
// another process can read is already complete.
VGPU_EXPORT CUresult cuIpcGetEventHandle(CUipcEventHandle* handle, CUevent ev) {
  return api("cuIpcGetEventHandle", true, false, [&](ShimState& s) -> CUresult {
    if (!handle) return CUDA_ERROR_INVALID_VALUE;
    if (!s.events.count(reinterpret_cast<uintptr_t>(ev))) return CUDA_ERROR_INVALID_VALUE;
    IpcMemPayload p{};
    p.magic = kIpcMagic;
    p.version = kIpcVersion;
    p.pid = static_cast<uint32_t>(::getpid());
    std::memcpy(handle, &p, sizeof p);
    return CUDA_SUCCESS;
  });
}

VGPU_EXPORT CUresult cuIpcOpenEventHandle(CUevent* ev, CUipcEventHandle handle) {
  return api("cuIpcOpenEventHandle", true, false, [&](ShimState& s) -> CUresult {
    if (!ev) return CUDA_ERROR_INVALID_VALUE;
    IpcMemPayload p{};
    std::memcpy(&p, &handle, sizeof p);
    if (p.magic != kIpcMagic || p.version != kIpcVersion) return CUDA_ERROR_INVALID_VALUE;
    if (p.pid == static_cast<uint32_t>(::getpid())) return CUDA_ERROR_INVALID_VALUE;
    const uintptr_t h = make_handle(s, kTagEvent);
    s.events[h] = {};
    *ev = reinterpret_cast<CUevent>(h);
    return CUDA_SUCCESS;
  });
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
  // NVRTC's probe is expected and refused; it is no sign of a static runtime.
  const bool nvrtc_probe = std::memcmp(uuid, kUuidNvrtcProbe, 16) == 0;
  static std::once_flag warned;
  if (!nvrtc_probe) std::call_once(warned, [] {
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
    VGPU_PROC(cuMemAllocManaged), VGPU_PROC(cuStreamAttachMemAsync), VGPU_PROC(cuMemPrefetchAsync),
    VGPU_PROC(cuMemPrefetchAsync_v2), VGPU_PROC(cuMemAdvise), VGPU_PROC(cuMemAdvise_v2),
    VGPU_PROC(cuLaunchHostFunc),
    VGPU_PROC(cuArray3DCreate), VGPU_PROC(cuArray3DCreate_v2), VGPU_PROC(cuArrayCreate),
    VGPU_PROC(cuArrayCreate_v2), VGPU_PROC(cuArray3DGetDescriptor), VGPU_PROC(cuArray3DGetDescriptor_v2),
    VGPU_PROC(cuArrayGetDescriptor), VGPU_PROC(cuArrayGetDescriptor_v2), VGPU_PROC(cuArrayDestroy),
    VGPU_PROC(cuMemcpyHtoA), VGPU_PROC(cuMemcpyHtoA_v2), VGPU_PROC(cuMemcpyAtoH), VGPU_PROC(cuMemcpyAtoH_v2),
    VGPU_PROC(cuMemcpyDtoA), VGPU_PROC(cuMemcpyDtoA_v2), VGPU_PROC(cuMemcpyAtoD), VGPU_PROC(cuMemcpyAtoD_v2),
    VGPU_PROC(cuMemcpy2D), VGPU_PROC(cuMemcpy2D_v2), VGPU_PROC(cuMemcpy2DUnaligned),
    VGPU_PROC(cuMemcpy2DUnaligned_v2), VGPU_PROC(cuMemcpy2DAsync), VGPU_PROC(cuMemcpy2DAsync_v2),
    VGPU_PROC(cuMemcpy3D), VGPU_PROC(cuMemcpy3D_v2), VGPU_PROC(cuMemcpy3DAsync), VGPU_PROC(cuMemcpy3DAsync_v2),
    VGPU_PROC(cuMemAllocAsync), VGPU_PROC(cuMemFreeAsync),
    VGPU_PROC(cuMemAllocPitch), VGPU_PROC(cuMemAllocPitch_v2), VGPU_PROC(cuModuleGetGlobal),
    VGPU_PROC(cuModuleGetGlobal_v2), VGPU_PROC(cuCtxSetCacheConfig), VGPU_PROC(cuCtxGetCacheConfig),
    VGPU_PROC(cuCtxSetSharedMemConfig), VGPU_PROC(cuCtxGetSharedMemConfig),
    VGPU_PROC(cuFuncSetSharedMemConfig), VGPU_PROC(cuStreamAddCallback),
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
    VGPU_PROC(cuTensorMapEncodeTiled), VGPU_PROC(cuTensorMapReplaceAddress),
    VGPU_PROC(cuTensorMapEncodeIm2col),
    VGPU_PROC(cuMemsetD8_v2), VGPU_PROC(cuMemsetD16_v2), VGPU_PROC(cuMemsetD32_v2),
    VGPU_PROC(cuMemsetD2D8_v2), VGPU_PROC(cuMemsetD2D16_v2), VGPU_PROC(cuMemsetD2D32_v2),
    VGPU_PROC(cuMemsetD2D8Async), VGPU_PROC(cuMemsetD2D16Async), VGPU_PROC(cuMemsetD2D32Async),
    VGPU_PROC(cuMemcpyDtoDAsync_v2), VGPU_PROC(cuMemcpyDtoHAsync_v2),
    VGPU_PROC(cuMemcpyAtoA_v2), VGPU_PROC(cuMemcpyHtoAAsync_v2), VGPU_PROC(cuMemcpyAtoHAsync_v2),
    VGPU_PROC(cuMemcpyPeer), VGPU_PROC(cuMemcpyPeerAsync),
    VGPU_PROC(cuMemHostRegister_v2), VGPU_PROC(cuMemHostUnregister), VGPU_PROC(cuMemHostGetFlags),
    VGPU_PROC(cuMemHostGetDevicePointer_v2),
    VGPU_PROC(cuDeviceGetByPCIBusId), VGPU_PROC(cuDeviceGetProperties),
    VGPU_PROC(cuCtxEnablePeerAccess), VGPU_PROC(cuCtxDisablePeerAccess),
    VGPU_PROC(cuCtxAttach), VGPU_PROC(cuCtxDetach),
    VGPU_PROC(cuOccupancyMaxPotentialBlockSize), VGPU_PROC(cuOccupancyMaxPotentialBlockSizeWithFlags),
    VGPU_PROC(cuMipmappedArrayCreate), VGPU_PROC(cuMipmappedArrayGetLevel), VGPU_PROC(cuMipmappedArrayDestroy),
    VGPU_PROC(cuFuncSetBlockShape), VGPU_PROC(cuFuncSetSharedSize), VGPU_PROC(cuParamSetSize),
    VGPU_PROC(cuParamSeti), VGPU_PROC(cuParamSetf), VGPU_PROC(cuParamSetv), VGPU_PROC(cuParamSetTexRef),
    VGPU_PROC(cuLaunch), VGPU_PROC(cuLaunchGrid), VGPU_PROC(cuLaunchGridAsync),
    VGPU_PROC(cuModuleGetTexRef), VGPU_PROC(cuModuleGetSurfRef), VGPU_PROC(cuTexRefCreate),
    VGPU_PROC(cuTexRefDestroy), VGPU_PROC(cuTexRefSetAddress_v2), VGPU_PROC(cuTexRefGetAddress_v2),
    VGPU_PROC(cuTexRefSetAddress2D_v3), VGPU_PROC(cuTexRefSetArray), VGPU_PROC(cuTexRefGetArray),
    VGPU_PROC(cuTexRefSetFormat), VGPU_PROC(cuTexRefGetFormat), VGPU_PROC(cuTexRefSetFlags),
    VGPU_PROC(cuTexRefGetFlags), VGPU_PROC(cuTexRefSetAddressMode), VGPU_PROC(cuTexRefGetAddressMode),
    VGPU_PROC(cuTexRefSetFilterMode), VGPU_PROC(cuTexRefGetFilterMode),
    VGPU_PROC(cuSurfRefSetArray), VGPU_PROC(cuSurfRefGetArray),
    VGPU_PROC(cuProfilerStart), VGPU_PROC(cuProfilerStop),
    VGPU_PROC(cuGraphicsUnregisterResource), VGPU_PROC(cuGraphicsMapResources),
    VGPU_PROC(cuGraphicsUnmapResources), VGPU_PROC(cuGraphicsResourceSetMapFlags_v2),
    VGPU_PROC(cuGraphicsResourceGetMappedPointer_v2), VGPU_PROC(cuGraphicsSubResourceGetMappedArray),
    VGPU_PROC(cuTexObjectCreate), VGPU_PROC(cuTexObjectDestroy), VGPU_PROC(cuSurfObjectCreate),
    VGPU_PROC(cuSurfObjectDestroy),
};

// A driver function this library exports but the table above does not list:
// cuMemGetAllocationGranularity and the rest of the virtual memory API, which
// NanoVDB fetches with cudaGetDriverEntryPoint and calls without checking.
// Only a name with a single ABI is resolved this way. One that also has a _v2
// export changed signature between releases, and which one a caller wants
// depends on the version it asked for, so that stays the table's decision.
void* exported_proc(const std::string& name) {
  if (name.rfind("cu", 0) != 0) return nullptr;
  static void* const self = [] {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&exported_proc), &info) || !info.dli_fname) return (void*)nullptr;
    return dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD);
  }();
  if (!self) return nullptr;
  void* fn = dlsym(self, name.c_str());
  if (!fn || dlsym(self, (name + "_v2").c_str())) return nullptr;
  // dlsym searches the library's dependencies too; only this library's own
  // definition is an answer.
  Dl_info where{}, mine{};
  if (!dladdr(fn, &where) || !dladdr(reinterpret_cast<void*>(&exported_proc), &mine) ||
      where.dli_fbase != mine.dli_fbase)
    return nullptr;
  return fn;
}

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
  return exported_proc(request);
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
