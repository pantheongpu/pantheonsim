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
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dlfcn.h>

#include <functional>
#include <fstream>
#include <unistd.h>
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
#include <stdexcept>

#include "error_names.hpp"
#include "vgpu/exec/devrt.hpp"
#include "fatbin.hpp"
#include "texture_formats.hpp"
#include "vgpu/sass/cubin.hpp"
#include "vgpu/sass/exec.hpp"
#include "vgpu/cuda_attributes.hpp"
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
  std::string cubin;   // a cubin (ELF) the library was loaded from, run as SASS

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
  vgpu::cuda::TexFormat tf;   // what the descriptor's format is to the texture unit
  size_t elem = 0;       // bytes an element takes, all channels (a block-compressed array's: a 4 x 4 block)
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

// A stream-ordered memory pool (CUmemoryPool). Every stream here is
// synchronous, so the ordering half of the API costs nothing: an allocation is
// usable when the call returns and a free is complete when it returns. What a
// pool is for is the caching, and that is modelled: memory freed to a pool is
// kept, up to the pool's release threshold, and handed out again.
//
// Which cached block a request reuses is this engine's choice, not the API's:
// the first block at least as large as the request and no more than twice its
// size, so a small request never quietly takes a large block out of circulation.
//
// The runtime's cudaMemPool_t is a different object: a pool made through one
// library is not a handle the other recognises, and each library's default pool
// is its own.
struct PoolRec {
  int device = 0;
  bool is_default = false;
  bool destroyed = false;
  unsigned long long threshold = 0;   // CU_MEMPOOL_ATTR_RELEASE_THRESHOLD
  // The reuse policies are on by default, and with synchronous streams they
  // change nothing observable; they are kept because a program reads them back.
  int reuse_follow_event_deps = 1, reuse_allow_opportunistic = 1, reuse_allow_internal_deps = 1;
  uint64_t used = 0, used_high = 0, reserved = 0, reserved_high = 0;
  std::map<uint64_t, uint64_t> live;                  // handed out: pointer -> size
  std::vector<std::pair<uint64_t, uint64_t>> cached;  // freed and kept, oldest first
  // Access granted to each device, as a CUmemAccess_flags value. A pool is
  // accessible from the device it lives on and from no other until told so.
  std::map<int, int> access;
  uint64_t cached_bytes() const {
    uint64_t n = 0;
    for (const auto& [p, sz] : cached) n += sz;
    return n;
  }
};

struct ShimState {
  // One lock for both CUDA libraries, since they share one machine
  // (shared_runtime.cpp).
  std::recursive_mutex& mu = vgpu::runtime::shared_api_mutex();
  bool initialized = false;
  vgpu::runtime::Runtime* rt = nullptr;   // the process's machine, not owned (shared_runtime.cpp)
  uintptr_t next_id = 8;

  std::unordered_map<uintptr_t, int> contexts;         // ctx handle -> device ordinal
  // What a context carries besides its device: the flags it reports (cuCtxGetFlags) and the id
  // cuCtxGetId gives, numbered from 1 in the order contexts are made, the primary ones included.
  struct CtxState {
    unsigned flags = 0;
    unsigned long long id = 0;
  };
  std::unordered_map<uintptr_t, CtxState> ctx_state;
  unsigned long long next_ctx_id = 1;
  // cuDevicePrimaryCtxSetFlags, per device, and whether the primary context is active: it is from
  // a retain until a reset or until its last reference is released.
  std::map<int, unsigned> primary_flags;
  std::set<int> primary_active;
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
  std::unordered_map<uintptr_t, int> stream_priority;   // those made with a priority, clamped
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
  // Memory pools. In a deque so a pool's address, which is its handle, never
  // moves; each device's default pool is made on first use.
  std::deque<PoolRec> pools;
  std::map<int, PoolRec*> default_pool;   // by device
  std::map<int, PoolRec*> current_pool;
  // The device each managed allocation (by base) was made against, which
  // CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL reports.
  std::map<uintptr_t, int> managed_device;
  // Allocations (by base) CU_POINTER_ATTRIBUTE_SYNC_MEMOPS has been set on.
  std::set<uint64_t> sync_memops;
  // cuCtxSetLimit, per device, with the defaults the runtime shim keeps
  // (runtime_api.cpp, State::DeviceLimits): a program with a static cudart
  // sets and reads its limits through here, and a kernel's device runtime
  // reads them.
  struct Limits {
    size_t stack = 1024;
    size_t printf_fifo = 1u << 20;
    size_t malloc_heap = 8u << 20;
    size_t sync_depth = 2;
    size_t pending_launches = 2048;
    size_t l2_fetch_granularity = 64;
    size_t persisting_l2 = 0;
  };
  std::map<int, Limits> limits;
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
    case Err::IllegalInstruction: return CUDA_ERROR_ILLEGAL_INSTRUCTION;
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
    case Err::IllegalInstruction:
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

// Thrown where a call needs the thread's current context and it has none (or it was destroyed
// under the thread): CUDA_ERROR_INVALID_CONTEXT, which is what the card answers, where a missing
// argument would be INVALID_VALUE.
struct NoContext : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// The entry points that fail with CUDA_ERROR_INVALID_CONTEXT when the thread has no current
// context. Measured on an RTX 3060: allocation (device, host, managed, pitched, pool, arrays),
// copies and fills, streams, events, modules, kernel launches and the context queries. Others
// (cuMemFree(NULL), cuMemFreeHost(NULL), the device queries, the primary-context calls) do not need
// one.
bool needs_current_context(const char* name) {
  static const char* const kNeeds[] = {
      "cuMemAlloc", "cuMemAllocPitch", "cuMemAllocManaged", "cuMemAllocAsync", "cuMemAllocFromPoolAsync",
      "cuMemHostAlloc", "cuMemAllocHost", "cuMemGetInfo", "cuMemGetAddressRange", "cuMemcpy", "cuMemset",
      "cuStreamCreate", "cuStreamSynchronize", "cuStreamQuery", "cuEventCreate", "cuModuleLoad",
      "cuArrayCreate", "cuArray3DCreate", "cuMipmappedArrayCreate", "cuLaunchKernel", "cuCtxSynchronize",
      "cuCtxGetLimit", "cuCtxSetLimit", "cuCtxGetCacheConfig", "cuCtxSetCacheConfig",
      "cuCtxGetSharedMemConfig", "cuCtxSetSharedMemConfig", "cuCtxGetStreamPriorityRange", "cuCtxGetFlags",
      "cuCtxSetFlags"};
  for (const char* n : kNeeds)
    if (std::strncmp(name, n, std::strlen(n)) == 0) return true;
  return false;
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
  if (s.initialized && ctx_stack().empty() && needs_current_context(name)) {
    report(name, "no current context (create one with cuCtxCreate or cuDevicePrimaryCtxRetain + cuCtxSetCurrent)");
    return CUDA_ERROR_INVALID_CONTEXT;
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
  } catch (const NoContext& e) {
    report(name, e.what());
    return CUDA_ERROR_INVALID_CONTEXT;
  } catch (const std::exception& e) {
    report(name, std::string("unexpected: ") + e.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

int current_device(ShimState& s) {
  if (ctx_stack().empty())
    throw NoContext("no current context (create one with cuCtxCreate or "
                    "cuDevicePrimaryCtxRetain + cuCtxSetCurrent)");
  // With a stack per thread, another thread can destroy the context this one
  // still has current. That is the caller's error, and it should be reported
  // as one rather than escaping as std::out_of_range and CUDA_ERROR_UNKNOWN.
  auto it = s.contexts.find(ctx_stack().back());
  if (it == s.contexts.end())
    throw NoContext("the current context has been destroyed (by cuCtxDestroy, possibly on "
                    "another thread)");
  return it->second;
}

vgpu::runtime::Device& current(ShimState& s) { return s.rt->device(current_device(s)); }

// Context flags (CUctx_flags): one scheduling mode (0 auto, 1 spin, 2 yield, 4 blocking sync), and
// any of MAP_HOST 8, LMEM_RESIZE_TO_MAX 0x10, COREDUMP_ENABLE 0x20, USER_COREDUMP_ENABLE 0x40 and
// SYNC_MEMOPS 0x80. Measured on an RTX 3060: anything above 0xff, and two scheduling bits at once,
// are CUDA_ERROR_INVALID_VALUE. (The card under WSL answers UNKNOWN for the two core-dump flags,
// which a Linux driver takes.)
constexpr unsigned kCtxSchedMask = 0x7;
constexpr unsigned kCtxMapHost = 0x8;
bool ctx_flags_valid(unsigned flags) {
  if (flags & ~0xFFu) return false;
  const unsigned sched = flags & kCtxSchedMask;
  return sched == 0 || sched == 1 || sched == 2 || sched == 4;
}
// What cuCtxSetFlags and cuDevicePrimaryCtxSetFlags keep: the scheduling mode, LMEM_RESIZE_TO_MAX
// and SYNC_MEMOPS; MAP_HOST and the core-dump flags are taken and not stored (measured).
constexpr unsigned kCtxSettableMask = kCtxSchedMask | 0x10u | 0x80u;

uintptr_t make_handle(ShimState& s, uintptr_t tag);
// Makes a context's record, with the next id. Caller holds the lock.
uintptr_t new_context(ShimState& s, int dev, unsigned flags) {
  const uintptr_t h = make_handle(s, kTagCtx);
  s.contexts[h] = dev;
  s.ctx_state[h] = ShimState::CtxState{flags, s.next_ctx_id++};
  return h;
}
// The primary context of `dev`, made on first use. Its flags are the primary flags set so far with
// MAP_HOST always on, as the card reports (0x8 for a fresh one).
uintptr_t ensure_primary(ShimState& s, int dev) {
  auto it = s.primary_ctx.find(dev);
  if (it == s.primary_ctx.end())
    it = s.primary_ctx.emplace(dev, new_context(s, dev, (s.primary_flags[dev] & kCtxSettableMask) | kCtxMapHost)).first;
  return it->second;
}

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
// The PTX the NVRTC shim noted for a cubin it handed out (empty for any other
// cubin, and when the toolkit's own NVRTC is the library loaded). Found by name
// in whichever libnvrtc is loaded, since a program may have opened it privately.
std::string nvrtc_ptx_for_cubin(const void* cubin, size_t size) {
  using Fn = char* (*)(const void*, size_t);
  Fn fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "vgpu_nvrtc_ptx_for_cubin"));
  for (const char* so : {"libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2", "libnvrtc.so"}) {
    if (fn) break;
    if (void* h = dlopen(so, RTLD_NOLOAD | RTLD_LAZY)) fn = reinterpret_cast<Fn>(dlsym(h, "vgpu_nvrtc_ptx_for_cubin"));
  }
  if (!fn) return {};
  char* copy = fn(cubin, size);
  if (!copy) return {};
  std::string ptx(copy);
  std::free(copy);
  return ptx;
}

// A bare cubin's size: where the image ends, the later of its section table and its program header
// table, which a cubin keeps after the sections (an RTX 3060 reports the cubin it loaded as that size).
// All the driver has to go by, handed a pointer.
size_t bare_cubin_size(const uint8_t* b) {
  uint64_t shoff, phoff;
  uint16_t shentsize, shnum, phentsize, phnum;
  std::memcpy(&shoff, b + 0x28, 8);
  std::memcpy(&shentsize, b + 0x3a, 2);
  std::memcpy(&shnum, b + 0x3c, 2);
  std::memcpy(&phoff, b + 0x20, 8);
  std::memcpy(&phentsize, b + 0x36, 2);
  std::memcpy(&phnum, b + 0x38, 2);
  return static_cast<size_t>(std::max(shoff + static_cast<uint64_t>(shentsize) * shnum,
                                      phoff + static_cast<uint64_t>(phentsize) * phnum));
}

// The PTX the NVRTC shim noted for this cubin, when the SASS engine cannot run all of the cubin yet --
// the fallback a fatbin's PTX gives (VGPU_SASS=1 insists on the SASS, VGPU_SASS=0 asks for none).
// Empty when the cubin is to run.
std::string ptx_for_unsupported_cubin(const uint8_t* b, size_t size) {
  const char* force = std::getenv("VGPU_SASS");
  if (force && (force[0] == '1' || force[0] == '0')) return {};
  std::string aside = nvrtc_ptx_for_cubin(b, size);
  if (aside.empty()) return {};
  std::string why;
  try {
    why = vgpu::sass::unsupported(b, size);
  } catch (const vgpu::Error& e) {
    why = e.message();
  }
  if (why.empty()) return {};
  if (const char* log = std::getenv("VGPU_SASS_LOG"); log && log[0] == '1')
    std::fprintf(stderr, "[vgpu] running PTX instead of SASS: %s\n", why.c_str());
  return aside;
}

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

// Attribute values by id, for the places in this file that need a limit
// (texture alignment, the pitch); cuDeviceGetAttribute answers from the same
// table (vgpu/cuda_attributes.hpp), which the runtime's cudaDeviceGetAttribute
// and cudaDeviceProp use too. An id that table does not know is zero here.
int extra_attribute(const vgpu::DeviceProfile& p, int attrib) {
  int v = 0;
  vgpu::cuda::device_attribute(p, 0, attrib, &v);
  return v;
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
    s.managed_device[reinterpret_cast<uintptr_t>(p)] = dev;
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
    s.managed_device.erase(p);
    s.sync_memops.erase(p);
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
  uint64_t mid = !lib.cubin.empty()
                     ? s.rt->device(dev).load_cubin(reinterpret_cast<const uint8_t*>(lib.cubin.data()), lib.cubin.size())
                     : s.rt->device(dev).load_module(lib.ptx);
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

/* ---- tracing ----
 * A profiler (CUPTI) is told of the driver calls a program makes: the entry
 * points a program calls itself are wrapped in traced(), which reports the call
 * once, with pointers to its arguments, and everything the shim does underneath
 * (cuStreamSynchronize through cuCtxSynchronize) is part of that one call. The
 * work a call issues -- a kernel, a copy, a fill, a wait -- is recorded where
 * it is done, with the correlation of the call that issued it. See
 * vgpu/profiling.hpp, and nvidia/docs/cupti.md for what is delivered.
 */

// Reports one driver call from the program's side. `name` is NVIDIA's own
// spelling of the function (cuMemAlloc_v2), which is what a subscriber is told.
template <class Body, class... A>
CUresult traced(const char* name, Body body, A... a) {
  if (vgpu::profiling::enabled() || vgpu::profiling::hooked()) {
    const void* argv[sizeof...(A) + 1] = {static_cast<const void*>(&a)..., nullptr};
    const uint16_t sizes[sizeof...(A) + 1] = {static_cast<uint16_t>(sizeof(A))..., 0};
    vgpu::profiling::note_args(argv, static_cast<int>(sizeof...(A)), sizes);
  }
  vgpu::profiling::ApiCall call(name, vgpu::profiling::Domain::Driver);
  const CUresult rc = body(a...);
  call.set_result(static_cast<int32_t>(rc));
  return rc;
}

// The device a record names: that of the calling thread's current context.
uint32_t profiled_device(ShimState& s) {
  if (ctx_stack().empty()) return 0;
  const auto it = s.contexts.find(ctx_stack().back());
  return it == s.contexts.end() ? 0u : static_cast<uint32_t>(it->second);
}
uint32_t profiled_device() {
  ShimState& s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  return profiled_device(s);
}

// What a profiler calls the memory a pointer names (defined with the host
// memory registries it reads).
vgpu::profiling::MemKind profiled_kind(ShimState& s, const void* p, bool device);

// A copy that was done. copy_kind: 1 host to device, 2 device to host, 3 device
// to device. A copy of nothing is not recorded.
void record_copy(ShimState& s, uint32_t copy_kind, const void* dst, const void* src, size_t bytes,
                 uint64_t t0, CUstream stream, bool async) {
  if (!vgpu::profiling::enabled() || bytes == 0) return;
  vgpu::profiling::Event ev;
  ev.kind = vgpu::profiling::EventKind::Memcpy;
  ev.start_ns = t0;
  ev.end_ns = vgpu::profiling::now_ns();
  ev.device = profiled_device(s);
  ev.correlation = vgpu::profiling::work_correlation();
  ev.bytes = bytes;
  ev.copy_kind = copy_kind;
  ev.stream = reinterpret_cast<uint64_t>(stream);
  ev.async = async;
  ev.src_kind = profiled_kind(s, src, copy_kind == 2 || copy_kind == 3);
  ev.dst_kind = profiled_kind(s, dst, copy_kind == 1 || copy_kind == 3);
  vgpu::profiling::record(std::move(ev));
}

// A fill that was done.
void record_memset(ShimState& s, CUdeviceptr dst, uint32_t value, size_t bytes, uint64_t t0, CUstream stream,
                   bool async) {
  if (!vgpu::profiling::enabled() || bytes == 0) return;
  vgpu::profiling::Event ev;
  ev.kind = vgpu::profiling::EventKind::Memset;
  ev.start_ns = t0;
  ev.end_ns = vgpu::profiling::now_ns();
  ev.device = profiled_device(s);
  ev.correlation = vgpu::profiling::work_correlation();
  ev.bytes = bytes;
  ev.value = value;
  ev.stream = reinterpret_cast<uint64_t>(stream);
  ev.async = async;
  ev.dst_kind = profiled_kind(s, reinterpret_cast<const void*>(dst), vgpu::is_device_va(dst));
  vgpu::profiling::record(std::move(ev));
}

// A context coming into being: a subscriber is told, and a context record made.
void announce_context_created(ShimState& s, uintptr_t handle) {
  vgpu::profiling::note_context_made();
  if (!vgpu::profiling::enabled() && !vgpu::profiling::hooked()) return;
  const uint32_t device = profiled_device(s);
  vgpu::profiling::notify_resource(vgpu::profiling::Resource::ContextCreated, handle, device);
  if (vgpu::profiling::enabled()) {
    vgpu::profiling::Event ev;
    ev.kind = vgpu::profiling::EventKind::Context;
    ev.device = device;
    ev.start_ns = ev.end_ns = vgpu::profiling::now_ns();
    ev.correlation = vgpu::profiling::work_correlation();
    ev.handle = handle;   // a context of the driver's own, not the device's primary one
    vgpu::profiling::record(std::move(ev));
  }
}

// A stream coming into being.
void announce_stream_created(ShimState& s, uintptr_t handle, unsigned flags, int priority) {
  if (!vgpu::profiling::enabled() && !vgpu::profiling::hooked()) return;
  const uint32_t device = profiled_device(s);
  if (vgpu::profiling::enabled()) {
    vgpu::profiling::Event ev;
    ev.kind = vgpu::profiling::EventKind::Stream;
    ev.device = device;
    ev.start_ns = ev.end_ns = vgpu::profiling::now_ns();
    ev.correlation = vgpu::profiling::work_correlation();
    ev.handle = handle;
    ev.flags = flags;
    ev.priority = priority;
    vgpu::profiling::record(std::move(ev));
  }
  vgpu::profiling::notify_resource(vgpu::profiling::Resource::StreamCreated, handle, device);
}

// The name a subscriber is told a launch is of: the kernel's entry name.
const char* launch_symbol(CUfunction f) {
  ShimState& s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  const uintptr_t h = reinterpret_cast<uintptr_t>(f);
  if ((h & 7) == kTagKernel) {
    const auto it = s.kernels.find(h);
    return it == s.kernels.end() ? nullptr : it->second.name.c_str();
  }
  const auto it = s.functions.find(h);
  return it == s.functions.end() || !it->second.fn ? nullptr : it->second.fn->name.c_str();
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

/* ---- init / version / errors ---- */

static CUresult cuInit_impl(unsigned int flags) {
  return api("cuInit", false, false, [&](ShimState& s) {
    if (flags != 0) return CUDA_ERROR_INVALID_VALUE;
    if (s.initialized) return CUDA_SUCCESS;
    // The machine libcudart may already have made (shared_runtime.cpp).
    s.rt = vgpu::runtime::shared_runtime();
    // No device shown (CUDA_VISIBLE_DEVICES), or a bad list: cuInit says so,
    // and nothing is initialized, as on the card (CUDA_ERROR_NO_DEVICE or
    // CUDA_ERROR_INVALID_DEVICE, and then cuDeviceGetCount answers
    // CUDA_ERROR_NOT_INITIALIZED).
    if (const int shown = s.rt->visibility_error()) return static_cast<CUresult>(shown);
    s.initialized = true;
    vgpu::load_injection_library();
    return CUDA_SUCCESS;
  });
}
// The call that brings the driver up is not one a profiler is told of, as on
// NVIDIA's (measured: the first cuInit raises no callback and leaves no record,
// a later one raises both); a subscriber is told initialisation finished
// instead, before the next call.
VGPU_EXPORT CUresult cuInit(unsigned int flags) {
  bool first;
  {
    ShimState& s = state();
    std::lock_guard<std::recursive_mutex> lock(s.mu);
    first = !s.initialized;
  }
  if (!first) return traced("cuInit", cuInit_impl, flags);
  const CUresult r = cuInit_impl(flags);
  if (r == CUDA_SUCCESS) vgpu::profiling::notify_init_finished();
  return r;
}

// What libcudart does for the driver when a program uses both APIs, as the
// real runtime does through the real driver: the device's primary context,
// retained once on the runtime's behalf, made current for the calling thread,
// so the program's own driver calls find a context there. A context the
// program made current itself is left alone; a primary one is swapped for the
// runtime's current device's. Called by the runtime shim, found through the
// process's global scope (runtime_api.cpp).
VGPU_EXPORT int vgpu_driver_bind_primary_v1(int dev) {
  if (cuInit_impl(0) != CUDA_SUCCESS) return CUDA_ERROR_NOT_INITIALIZED;
  return api("vgpu_driver_bind_primary", true, false, [&](ShimState& s) {
    check_device(s, dev);
    ensure_primary(s, dev);
    auto it = s.primary_ctx.find(dev);
    static std::set<int> retained;   // the runtime's one reference per device
    if (retained.insert(dev).second) ++s.primary_refs[dev];
    s.primary_active.insert(dev);
    auto& stack = ctx_stack();
    bool primary_current = false;
    if (!stack.empty())
      for (const auto& [d, h] : s.primary_ctx) primary_current |= h == stack.back();
    if (stack.empty()) stack.push_back(it->second);
    else if (primary_current) stack.back() = it->second;
    return CUDA_SUCCESS;
  });
}

static CUresult cuDriverGetVersion_impl(int* driverVersion);
VGPU_EXPORT CUresult cuDriverGetVersion(int* driverVersion) { return traced("cuDriverGetVersion", cuDriverGetVersion_impl, driverVersion); }
static CUresult cuDriverGetVersion_impl(int* driverVersion) {
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
static CUresult cuGetErrorName_impl(CUresult error, const char** pStr);
VGPU_EXPORT CUresult cuGetErrorName(CUresult error, const char** pStr) { return traced("cuGetErrorName", cuGetErrorName_impl, error, pStr); }
static CUresult cuGetErrorName_impl(CUresult error, const char** pStr) {
  if (!pStr) return CUDA_ERROR_INVALID_VALUE;
  const vgpu::cuda::ErrorInfo* e = vgpu::cuda::find_error(static_cast<int>(error));
  *pStr = e ? e->driver_name : nullptr;
  return *pStr ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

static CUresult cuGetErrorString_impl(CUresult error, const char** pStr);
VGPU_EXPORT CUresult cuGetErrorString(CUresult error, const char** pStr) { return traced("cuGetErrorString", cuGetErrorString_impl, error, pStr); }
static CUresult cuGetErrorString_impl(CUresult error, const char** pStr) {
  if (!pStr) return CUDA_ERROR_INVALID_VALUE;
  const vgpu::cuda::ErrorInfo* e = vgpu::cuda::find_error(static_cast<int>(error));
  *pStr = e && e->driver_name ? e->text : nullptr;
  return *pStr ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

/* ---- device discovery ---- */

static CUresult cuDeviceGetCount_impl(int* count) {
  return api("cuDeviceGetCount", true, false, [&](ShimState& s) {
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    *count = s.rt->device_count();
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuDeviceGetCount(int* count) {
  return traced("cuDeviceGetCount", cuDeviceGetCount_impl, count);
}

static CUresult cuDeviceGet_impl(CUdevice* device, int ordinal) {
  return api("cuDeviceGet", true, false, [&](ShimState& s) {
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, ordinal);
    *device = ordinal;
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuDeviceGet(CUdevice* device, int ordinal) {
  return traced("cuDeviceGet", cuDeviceGet_impl, device, ordinal);
}

static CUresult cuDeviceGetName_impl(char* name, int len, CUdevice dev) {
  return api("cuDeviceGetName", true, false, [&](ShimState& s) {
    if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    const std::string& model = s.rt->device(dev).profile().model;
    std::snprintf(name, static_cast<size_t>(len), "%s", model.c_str());
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
  return traced("cuDeviceGetName", cuDeviceGetName_impl, name, len, dev);
}

static CUresult cuDeviceTotalMem_v2_impl(size_t* bytes, CUdevice dev) {
  return api("cuDeviceTotalMem", true, false, [&](ShimState& s) {
    if (!bytes) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    *bytes = static_cast<size_t>(s.rt->device(dev).profile().vram_bytes);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev) {
  return traced("cuDeviceTotalMem_v2", cuDeviceTotalMem_v2_impl, bytes, dev);
}
static CUresult cuDeviceTotalMem_impl(size_t* bytes, CUdevice dev);
VGPU_EXPORT CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev) { return traced("cuDeviceTotalMem", cuDeviceTotalMem_impl, bytes, dev); }
static CUresult cuDeviceTotalMem_impl(size_t* bytes, CUdevice dev) {
  return cuDeviceTotalMem_v2_impl(bytes, dev);
}

// A device attribute by its number (cuDeviceGetAttribute's body; a kernel's
// cudaDeviceGetAttribute asks here too, vgpu/exec/devrt.hpp).
static CUresult device_attribute_by_id(ShimState& s, int* pi, int attr_id, CUdevice dev) {
  {
    if (!pi) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    const vgpu::DeviceProfile& p = s.rt->device(dev).profile();
    // An id that is not an attribute of any CUDA this knows is refused, as
    // the card refuses it (CUDA_ERROR_INVALID_VALUE), not answered with a zero
    // that reads as "the device has none of it".
    if (!vgpu::cuda::device_attribute(p, s.rt->device(dev).physical(), attr_id, pi)) {
      if (!quiet())
        std::fprintf(stderr, "[vgpu] cuDeviceGetAttribute: attribute %d is not a CUDA device attribute\n", attr_id);
      return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
  }
}

static CUresult cuDeviceGetAttribute_impl(int* pi, CUdevice_attribute attrib, CUdevice dev) {
  // Read the attribute as the integer the ABI actually passes, not as the
  // enum. A caller built against a newer CUDA header legitimately passes
  // values this shim's headers do not enumerate -- CUDA 13 sends 134, and
  // the `default:` arm of the switch exists precisely to answer them. But
  // *loading* an enum object holding a value outside its enumerators is
  // undefined: UBSan reports it, and a compiler is entitled to assume the
  // value is in range and delete the default arm, which would turn forward
  // compatibility into a wrong answer with no diagnostic. memcpy reads the
  // bytes without making that claim about them.
  static_assert(sizeof(attrib) == sizeof(int), "CUdevice_attribute is not int-sized");
  int attr_id;
  std::memcpy(&attr_id, &attrib, sizeof attr_id);
  return api("cuDeviceGetAttribute", true, false,
             [&](ShimState& s) { return device_attribute_by_id(s, pi, attr_id, dev); });
}
VGPU_EXPORT CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
  return traced("cuDeviceGetAttribute", cuDeviceGetAttribute_impl, pi, attrib, dev);
}

static CUresult cuDeviceComputeCapability_impl(int* major, int* minor, CUdevice dev) {
  return api("cuDeviceComputeCapability", true, false, [&](ShimState& s) {
    if (!major || !minor) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    *major = s.rt->device(dev).profile().cc_major;
    *minor = s.rt->device(dev).profile().cc_minor;
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev) {
  return traced("cuDeviceComputeCapability", cuDeviceComputeCapability_impl, major, minor, dev);
}

// The deprecated CUdevprop: the same limits cuDeviceGetAttribute reports.
static CUresult cuDeviceGetProperties_impl(CUdevprop* prop, CUdevice dev);
VGPU_EXPORT CUresult cuDeviceGetProperties(CUdevprop* prop, CUdevice dev) { return traced("cuDeviceGetProperties", cuDeviceGetProperties_impl, prop, dev); }
static CUresult cuDeviceGetProperties_impl(CUdevprop* prop, CUdevice dev) {
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
    prop->clockRate = extra_attribute(p, 13);     // CLOCK_RATE
    prop->textureAlign = extra_attribute(p, 14);  // TEXTURE_ALIGNMENT
    return CUDA_SUCCESS;
  });
}

/* ---- contexts ---- */

static CUresult cuCtxCreate_v2_impl(CUcontext* pctx, unsigned int flags, CUdevice dev) {
  return api("cuCtxCreate", true, false, [&](ShimState& s) {
    // The scheduling flags are performance hints, kept for cuCtxGetFlags and otherwise inert; an
    // invalid combination is refused as the card refuses it.
    if (!pctx || !ctx_flags_valid(flags)) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    uintptr_t h = new_context(s, dev, flags);
    ctx_stack().push_back(h);  // cuCtxCreate makes the new context current
    *pctx = reinterpret_cast<CUcontext>(h);
    announce_context_created(s, h);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev) {
  return traced("cuCtxCreate_v2", cuCtxCreate_v2_impl, pctx, flags, dev);
}
static CUresult cuCtxCreate_impl(CUcontext* pctx, unsigned int flags, CUdevice dev);
VGPU_EXPORT CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) { return traced("cuCtxCreate", cuCtxCreate_impl, pctx, flags, dev); }
static CUresult cuCtxCreate_impl(CUcontext* pctx, unsigned int flags, CUdevice dev) {
  return cuCtxCreate_v2_impl(pctx, flags, dev);
}
// CUDA 13's header maps cuCtxCreate to cuCtxCreate_v4, which takes a parameter
// block for green contexts and execution affinity. Without this symbol a
// program built against that toolkit fails to load at all -- the plain name it
// never calls is no help. A block that asks for neither is an ordinary context
// (CUDA 13's own samples pass a zeroed CUctxCreateParams, and the card takes it).
//
// Execution affinity (CU_EXEC_AFFINITY_TYPE_SM_COUNT: a context confined to some number of
// multiprocessors, as MPS gives) is not modelled, which cuDeviceGetExecAffinitySupport says (0). A
// request for it is CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY (224), as on an RTX 3060 whatever the
// type or count; a block that names parameters and a count of 0, or a count and no parameters, is
// CUDA_ERROR_INVALID_VALUE. A CIG block (a graphics-queue context) is CUDA_ERROR_NOT_SUPPORTED.
namespace {
constexpr CUresult kUnsupportedExecAffinity = static_cast<CUresult>(224);
struct ExecAffinityParamABI {   // CUexecAffinityParam: the type, then the SM count of a union
  int type;
  unsigned sm_count;
};
}  // namespace
static CUresult cuCtxCreate_v3_impl(CUcontext* pctx, void* exec_affinity_params, int num_params, unsigned int flags, CUdevice dev) {
  if (num_params < 0 || (exec_affinity_params == nullptr) != (num_params == 0)) return CUDA_ERROR_INVALID_VALUE;
  if (num_params > 0) return kUnsupportedExecAffinity;
  return cuCtxCreate_v2_impl(pctx, flags, dev);
}
VGPU_EXPORT CUresult cuCtxCreate_v3(CUcontext* pctx, void* exec_affinity_params, int num_params, unsigned int flags, CUdevice dev) {
  return traced("cuCtxCreate_v3", cuCtxCreate_v3_impl, pctx, exec_affinity_params, num_params, flags, dev);
}
static CUresult cuCtxCreate_v4_impl(CUcontext* pctx, void* ctx_create_params, unsigned int flags, CUdevice dev) {
  // CUctxCreateParams: execAffinityParams, numExecAffinityParams, cigParams.
  struct CreateParamsABI {
    void* exec_affinity;
    int num_exec_affinity;
    void* cig;
  };
  if (const auto* p = static_cast<const CreateParamsABI*>(ctx_create_params)) {
    if (p->num_exec_affinity < 0 || (p->exec_affinity == nullptr) != (p->num_exec_affinity == 0))
      return CUDA_ERROR_INVALID_VALUE;
    if (p->num_exec_affinity > 0) return kUnsupportedExecAffinity;
    if (p->cig) return CUDA_ERROR_NOT_SUPPORTED;
  }
  return cuCtxCreate_v2_impl(pctx, flags, dev);
}
VGPU_EXPORT CUresult cuCtxCreate_v4(CUcontext* pctx, void* ctx_create_params, unsigned int flags, CUdevice dev) {
  return traced("cuCtxCreate_v4", cuCtxCreate_v4_impl, pctx, ctx_create_params, flags, dev);
}
// Whether execution affinity of a type is supported on a device: never here, for any type
// (measured: a type CUDA has no such value for is answered too, with 0).
static CUresult cuDeviceGetExecAffinitySupport_impl(int* pi, int type, CUdevice dev);
VGPU_EXPORT CUresult cuDeviceGetExecAffinitySupport(int* pi, int type, CUdevice dev) { return traced("cuDeviceGetExecAffinitySupport", cuDeviceGetExecAffinitySupport_impl, pi, type, dev); }
static CUresult cuDeviceGetExecAffinitySupport_impl(int* pi, int type, CUdevice dev) {
  (void)type;
  return api("cuDeviceGetExecAffinitySupport", true, false, [&](ShimState& s) {
    if (!pi) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    *pi = 0;
    return CUDA_SUCCESS;
  });
}
// The affinity of the current context: all the device's multiprocessors, since none is confined.
// A type but SM_COUNT is CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY (measured).
static CUresult cuCtxGetExecAffinity_impl(void* pExecAffinity, int type);
VGPU_EXPORT CUresult cuCtxGetExecAffinity(void* pExecAffinity, int type) { return traced("cuCtxGetExecAffinity", cuCtxGetExecAffinity_impl, pExecAffinity, type); }
static CUresult cuCtxGetExecAffinity_impl(void* pExecAffinity, int type) {
  return api("cuCtxGetExecAffinity", true, false, [&](ShimState& s) {
    if (ctx_stack().empty()) return CUDA_ERROR_INVALID_CONTEXT;
    if (!pExecAffinity) return CUDA_ERROR_INVALID_VALUE;
    if (type != 0) return kUnsupportedExecAffinity;
    auto* out = static_cast<ExecAffinityParamABI*>(pExecAffinity);
    out->type = 0;
    out->sm_count = s.rt->device(current_device(s)).profile().limits.multiprocessors;
    return CUDA_SUCCESS;
  });
}

static CUresult cuCtxDestroy_v2_impl(CUcontext ctx) {
  return api("cuCtxDestroy", true, false, [&](ShimState& s) {
    uintptr_t h = check_handle(reinterpret_cast<uintptr_t>(ctx), kTagCtx, "context");
    if (!s.contexts.count(h)) return CUDA_ERROR_INVALID_CONTEXT;
    // A subscriber is told before the context goes, while it can still be asked about.
    vgpu::profiling::notify_resource(vgpu::profiling::Resource::ContextDestroyStarting, h, profiled_device(s));
    s.contexts.erase(h);
    s.ctx_state.erase(h);
    // Measured on an RTX 3060: destroying the thread's current context makes the one below it
    // current; destroying one further down the stack leaves it there, and a pop hands it back.
    if (!ctx_stack().empty() && ctx_stack().back() == h) ctx_stack().pop_back();
    std::erase_if(s.peer_access, [h](const auto& p) { return p.first == h || p.second == h; });
    s.attached.erase(h);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuCtxDestroy_v2(CUcontext ctx) {
  return traced("cuCtxDestroy_v2", cuCtxDestroy_v2_impl, ctx);
}
static CUresult cuCtxDestroy_impl(CUcontext ctx);
VGPU_EXPORT CUresult cuCtxDestroy(CUcontext ctx) { return traced("cuCtxDestroy", cuCtxDestroy_impl, ctx); }
static CUresult cuCtxDestroy_impl(CUcontext ctx) { return cuCtxDestroy_v2_impl(ctx); }

// The deprecated usage count: cuCtxAttach takes another reference to the
// current context and returns it (flags must be 0), cuCtxDetach drops one,
// and the context goes when its creator's reference does -- one detach more
// than there were attaches.
static CUresult cuCtxAttach_impl(CUcontext* pctx, unsigned int flags);
VGPU_EXPORT CUresult cuCtxAttach(CUcontext* pctx, unsigned int flags) { return traced("cuCtxAttach", cuCtxAttach_impl, pctx, flags); }
static CUresult cuCtxAttach_impl(CUcontext* pctx, unsigned int flags) {
  return api("cuCtxAttach", true, false, [&](ShimState& s) {
    if (!pctx || flags != 0) return CUDA_ERROR_INVALID_VALUE;
    if (ctx_stack().empty()) return CUDA_ERROR_INVALID_CONTEXT;
    (void)current_device(s);
    ++s.attached[ctx_stack().back()];
    *pctx = reinterpret_cast<CUcontext>(ctx_stack().back());
    return CUDA_SUCCESS;
  });
}
static CUresult cuCtxDetach_impl(CUcontext ctx);
VGPU_EXPORT CUresult cuCtxDetach(CUcontext ctx) { return traced("cuCtxDetach", cuCtxDetach_impl, ctx); }
static CUresult cuCtxDetach_impl(CUcontext ctx) {
  {
    ShimState& s = state();
    std::lock_guard<std::recursive_mutex> lock(s.mu);
    auto it = s.attached.find(reinterpret_cast<uintptr_t>(ctx));
    if (it != s.attached.end() && it->second > 0) {
      if (--it->second == 0) s.attached.erase(it);
      return CUDA_SUCCESS;
    }
  }
  return cuCtxDestroy_v2_impl(ctx);
}

static CUresult cuCtxSetCurrent_impl(CUcontext ctx) {
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
VGPU_EXPORT CUresult cuCtxSetCurrent(CUcontext ctx) {
  return traced("cuCtxSetCurrent", cuCtxSetCurrent_impl, ctx);
}

static CUresult cuCtxGetCurrent_impl(CUcontext* pctx) {
  return api("cuCtxGetCurrent", true, false, [&](ShimState& s) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    *pctx = ctx_stack().empty() ? nullptr : reinterpret_cast<CUcontext>(ctx_stack().back());
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuCtxGetCurrent(CUcontext* pctx) {
  return traced("cuCtxGetCurrent", cuCtxGetCurrent_impl, pctx);
}

// With no context current, both answer CUDA_ERROR_INVALID_CONTEXT, as the
// driver does on an RTX 3060 (CUDA 13.0); cuda.core asks cuCtxGetDevice first
// and takes that answer as "no device chosen yet".
static CUresult cuCtxGetDevice_impl(CUdevice* device);
VGPU_EXPORT CUresult cuCtxGetDevice(CUdevice* device) { return traced("cuCtxGetDevice", cuCtxGetDevice_impl, device); }
static CUresult cuCtxGetDevice_impl(CUdevice* device) {
  return api("cuCtxGetDevice", true, false, [&](ShimState& s) {
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    if (ctx_stack().empty()) return CUDA_ERROR_INVALID_CONTEXT;
    *device = current_device(s);
    return CUDA_SUCCESS;
  });
}

static CUresult cuCtxSynchronize_impl(void) {
  return api("cuCtxSynchronize", true, false, [&](ShimState& s) {
    if (ctx_stack().empty()) return CUDA_ERROR_INVALID_CONTEXT;
    (void)current_device(s);  // requires a current context
    vgpu::profiling::SyncScope waited(vgpu::profiling::SyncKind::Context, 0, 0, profiled_device(s));
    return CUDA_SUCCESS;      // everything is synchronous today
  });
}
VGPU_EXPORT CUresult cuCtxSynchronize(void) { return traced("cuCtxSynchronize", cuCtxSynchronize_impl); }

static CUresult cuDevicePrimaryCtxRetain_impl(CUcontext* pctx, CUdevice dev);
VGPU_EXPORT CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) { return traced("cuDevicePrimaryCtxRetain", cuDevicePrimaryCtxRetain_impl, pctx, dev); }
static CUresult cuDevicePrimaryCtxRetain_impl(CUcontext* pctx, CUdevice dev) {
  return api("cuDevicePrimaryCtxRetain", true, false, [&](ShimState& s) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    const uintptr_t h = ensure_primary(s, dev);
    ++s.primary_refs[dev];
    s.primary_active.insert(dev);
    *pctx = reinterpret_cast<CUcontext>(h);  // NOTE: does not make it current
    return CUDA_SUCCESS;
  });
}

// Every retain needs its release, and a release with nothing retained is an
// error. It used to succeed no matter how many times it was called, which hid
// the bug a library's own teardown most often has: releasing a primary context
// it never retained, and so dropping a reference some other component holds.
// The context handle survives a count of zero -- a later retain gets the same
// one back -- but cuDevicePrimaryCtxGetState reports it inactive.
static CUresult cuDevicePrimaryCtxRelease_v2_impl(CUdevice dev);
VGPU_EXPORT CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) { return traced("cuDevicePrimaryCtxRelease_v2", cuDevicePrimaryCtxRelease_v2_impl, dev); }
static CUresult cuDevicePrimaryCtxRelease_v2_impl(CUdevice dev) {
  return api("cuDevicePrimaryCtxRelease", true, false, [&](ShimState& s) {
    check_device(s, dev);
    auto it = s.primary_refs.find(dev);
    if (it == s.primary_refs.end() || it->second == 0) {
      report("cuDevicePrimaryCtxRelease",
             "primary context of device " + std::to_string(dev) +
                 " released more times than it was retained");
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (--it->second == 0) s.primary_active.erase(dev);
    return CUDA_SUCCESS;
  });
}
static CUresult cuDevicePrimaryCtxRelease_impl(CUdevice dev);
VGPU_EXPORT CUresult cuDevicePrimaryCtxRelease(CUdevice dev) { return traced("cuDevicePrimaryCtxRelease", cuDevicePrimaryCtxRelease_impl, dev); }
static CUresult cuDevicePrimaryCtxRelease_impl(CUdevice dev) {
  return cuDevicePrimaryCtxRelease_v2(dev);
}

/* ---- memory ---- */

// A profiler's record of memory coming or going (vgpu/profiling.hpp), as the
// runtime tells it.
static void profile_memory(uint8_t op, uint64_t address, uint64_t bytes, vgpu::profiling::MemKind kind, int device) {
  if (!vgpu::profiling::enabled()) return;
  vgpu::profiling::Event e;
  e.kind = vgpu::profiling::EventKind::Memory;
  e.op = op;
  e.start_ns = e.end_ns = vgpu::profiling::host_ns();
  e.device = static_cast<uint32_t>(device);
  e.correlation = vgpu::profiling::work_correlation();
  e.process_id = static_cast<uint32_t>(::getpid());
  e.address = address;
  e.bytes = bytes;
  e.src_kind = kind;
  vgpu::profiling::record(std::move(e));
}

static CUresult cuMemAlloc_v2_impl(CUdeviceptr* dptr, size_t bytesize) {
  return api("cuMemAlloc", true, false, [&](ShimState& s) {
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;
    *dptr = current(s).memory().alloc(bytesize);
    profile_memory(1, *dptr, bytesize, vgpu::profiling::MemKind::Device, current_device(s));
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
  return traced("cuMemAlloc_v2", cuMemAlloc_v2_impl, dptr, bytesize);
}
static CUresult cuMemAlloc_impl(CUdeviceptr* dptr, size_t bytesize);
VGPU_EXPORT CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) { return traced("cuMemAlloc", cuMemAlloc_impl, dptr, bytesize); }
static CUresult cuMemAlloc_impl(CUdeviceptr* dptr, size_t bytesize) {
  return cuMemAlloc_v2_impl(dptr, bytesize);
}

static CUresult cuMemFree_v2_impl(CUdeviceptr dptr) {
  return api("cuMemFree", true, false, [&](ShimState& s) {
    if (dptr == 0) return CUDA_SUCCESS;   // a documented no-op, with or without a context
    if (auto it = s.managed.find(dptr); it != s.managed.end()) {
      profile_memory(2, dptr, it->second, vgpu::profiling::MemKind::Managed, current_device(s));
      for (int d = 0; d < s.rt->device_count(); ++d) s.rt->device(d).memory().unmap_host(dptr);
      std::free(reinterpret_cast<void*>(dptr));
      s.managed_device.erase(dptr);
      s.sync_memops.erase(dptr);
      s.managed.erase(it);
      return CUDA_SUCCESS;
    }
    // The device heap's blocks are not the host's to free (as cudaFree).
    if (owner_memory(s, dptr).heap_contains(dptr)) return CUDA_ERROR_INVALID_VALUE;
    uint64_t base = 0, size = 0;
    const bool known = owner_memory(s, dptr).find_allocation(dptr, &base, &size);
    owner_memory(s, dptr).free(dptr);
    profile_memory(2, dptr, known ? size : 0, vgpu::profiling::MemKind::Device, current_device(s));
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemFree_v2(CUdeviceptr dptr) {
  return traced("cuMemFree_v2", cuMemFree_v2_impl, dptr);
}
static CUresult cuMemFree_impl(CUdeviceptr dptr);
VGPU_EXPORT CUresult cuMemFree(CUdeviceptr dptr) { return traced("cuMemFree", cuMemFree_impl, dptr); }
static CUresult cuMemFree_impl(CUdeviceptr dptr) { return cuMemFree_v2_impl(dptr); }

// The three directions, in their synchronous and stream-ordered spellings. Every
// stream is synchronous here, so the stream orders nothing; it is still the one
// the program named, which is what a profiler's timeline is drawn by.
static CUresult memcpy_htod(CUdeviceptr dst, const void* src, size_t n, CUstream stream, bool async) {
  return api("cuMemcpyHtoD", true, false, [&](ShimState& s) {
    if (!src && n) return CUDA_ERROR_INVALID_VALUE;
    const uint64_t t0 = vgpu::profiling::enabled() ? vgpu::profiling::now_ns() : 0;
    dev_write(s, dst, src, n);
    record_copy(s, 1, reinterpret_cast<const void*>(dst), src, n, t0, stream, async);
    return CUDA_SUCCESS;
  });
}
static CUresult memcpy_dtoh(void* dst, CUdeviceptr src, size_t n, CUstream stream, bool async) {
  return api("cuMemcpyDtoH", true, false, [&](ShimState& s) {
    if (!dst && n) return CUDA_ERROR_INVALID_VALUE;
    // Memory registered read-only is not a destination for a copy (an RTX 3060: INVALID_VALUE).
    if (n) {
      const auto& regs = s.rt->host_registrations();
      auto it = regs.lower_bound(static_cast<char*>(dst) + n);
      if (it != regs.begin()) {
        --it;
        if ((it->second.flags & CU_MEMHOSTREGISTER_READ_ONLY) &&
            static_cast<const char*>(it->first) + it->second.size > static_cast<const char*>(dst))
          return CUDA_ERROR_INVALID_VALUE;
      }
    }
    const uint64_t t0 = vgpu::profiling::enabled() ? vgpu::profiling::now_ns() : 0;
    dev_read(s, dst, src, n);
    record_copy(s, 2, dst, reinterpret_cast<const void*>(src), n, t0, stream, async);
    return CUDA_SUCCESS;
  });
}
static CUresult memcpy_dtod(CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream stream, bool async) {
  return api("cuMemcpyDtoD", true, false, [&](ShimState& s) {
    const uint64_t t0 = vgpu::profiling::enabled() ? vgpu::profiling::now_ns() : 0;
    std::vector<uint8_t> tmp(n);
    dev_read(s, tmp.data(), src, n);
    dev_write(s, dst, tmp.data(), n);
    record_copy(s, 3, reinterpret_cast<const void*>(dst), reinterpret_cast<const void*>(src), n, t0, stream, async);
    return CUDA_SUCCESS;
  });
}
static CUresult cuMemcpyHtoD_v2_impl(CUdeviceptr d, const void* h, size_t n) { return memcpy_htod(d, h, n, nullptr, false); }
static CUresult cuMemcpyDtoH_v2_impl(void* h, CUdeviceptr d, size_t n) { return memcpy_dtoh(h, d, n, nullptr, false); }
static CUresult cuMemcpyDtoD_v2_impl(CUdeviceptr d, CUdeviceptr s, size_t n) { return memcpy_dtod(d, s, n, nullptr, false); }
static CUresult cuMemcpyHtoDAsync_v2_impl(CUdeviceptr d, const void* h, size_t n, CUstream st) {
  return memcpy_htod(d, h, n, st, true);
}
static CUresult cuMemcpyDtoHAsync_v2_impl(void* h, CUdeviceptr d, size_t n, CUstream st) {
  return memcpy_dtoh(h, d, n, st, true);
}
static CUresult cuMemcpyDtoDAsync_v2_impl(CUdeviceptr d, CUdeviceptr s, size_t n, CUstream st) {
  return memcpy_dtod(d, s, n, st, true);
}
VGPU_EXPORT CUresult cuMemcpyHtoD_v2(CUdeviceptr d, const void* h, size_t n) {
  return traced("cuMemcpyHtoD_v2", cuMemcpyHtoD_v2_impl, d, h, n);
}
VGPU_EXPORT CUresult cuMemcpyDtoH_v2(void* h, CUdeviceptr d, size_t n) {
  return traced("cuMemcpyDtoH_v2", cuMemcpyDtoH_v2_impl, h, d, n);
}
VGPU_EXPORT CUresult cuMemcpyDtoD_v2(CUdeviceptr d, CUdeviceptr s, size_t n) {
  return traced("cuMemcpyDtoD_v2", cuMemcpyDtoD_v2_impl, d, s, n);
}
VGPU_EXPORT CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr d, const void* h, size_t n, CUstream st) {
  return traced("cuMemcpyHtoDAsync_v2", cuMemcpyHtoDAsync_v2_impl, d, h, n, st);
}
VGPU_EXPORT CUresult cuMemcpyDtoHAsync_v2(void* h, CUdeviceptr d, size_t n, CUstream st) {
  return traced("cuMemcpyDtoHAsync_v2", cuMemcpyDtoHAsync_v2_impl, h, d, n, st);
}
VGPU_EXPORT CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr d, CUdeviceptr s, size_t n, CUstream st) {
  return traced("cuMemcpyDtoDAsync_v2", cuMemcpyDtoDAsync_v2_impl, d, s, n, st);
}
// The pre-CUDA 3.2 spellings are not reported: their parameters are 32-bit and a
// profiler would be handed a structure that says so about 64-bit arguments.
static CUresult cuMemcpyHtoD_impl(CUdeviceptr d, const void* h, size_t n);
VGPU_EXPORT CUresult cuMemcpyHtoD(CUdeviceptr d, const void* h, size_t n) { return traced("cuMemcpyHtoD", cuMemcpyHtoD_impl, d, h, n); }
static CUresult cuMemcpyHtoD_impl(CUdeviceptr d, const void* h, size_t n) { return cuMemcpyHtoD_v2_impl(d, h, n); }
static CUresult cuMemcpyDtoH_impl(void* h, CUdeviceptr d, size_t n);
VGPU_EXPORT CUresult cuMemcpyDtoH(void* h, CUdeviceptr d, size_t n) { return traced("cuMemcpyDtoH", cuMemcpyDtoH_impl, h, d, n); }
static CUresult cuMemcpyDtoH_impl(void* h, CUdeviceptr d, size_t n) { return cuMemcpyDtoH_v2_impl(h, d, n); }
static CUresult cuMemcpyDtoD_impl(CUdeviceptr d, CUdeviceptr sptr, size_t n);
VGPU_EXPORT CUresult cuMemcpyDtoD(CUdeviceptr d, CUdeviceptr sptr, size_t n) { return traced("cuMemcpyDtoD", cuMemcpyDtoD_impl, d, sptr, n); }
static CUresult cuMemcpyDtoD_impl(CUdeviceptr d, CUdeviceptr sptr, size_t n) {
  return cuMemcpyDtoD_v2_impl(d, sptr, n);
}

static CUresult cuMemGetInfo_v2_impl(size_t* free_out, size_t* total);
VGPU_EXPORT CUresult cuMemGetInfo_v2(size_t* free_out, size_t* total) { return traced("cuMemGetInfo_v2", cuMemGetInfo_v2_impl, free_out, total); }
static CUresult cuMemGetInfo_v2_impl(size_t* free_out, size_t* total) {
  return api("cuMemGetInfo", true, false, [&](ShimState& s) {
    if (!free_out || !total) return CUDA_ERROR_INVALID_VALUE;
    vgpu::MemoryManager& mm = current(s).memory();
    *total = static_cast<size_t>(mm.capacity());
    *free_out = static_cast<size_t>(mm.capacity() - mm.used());
    return CUDA_SUCCESS;
  });
}
static CUresult cuMemGetInfo_impl(size_t* f, size_t* t);
VGPU_EXPORT CUresult cuMemGetInfo(size_t* f, size_t* t) { return traced("cuMemGetInfo", cuMemGetInfo_impl, f, t); }
static CUresult cuMemGetInfo_impl(size_t* f, size_t* t) { return cuMemGetInfo_v2(f, t); }

/* ---- modules / launch ---- */

// What a profiler is told of a module loaded through the driver API (the same
// callbacks as the runtime's, vgpu/profiling.hpp): the driver loads a module
// when asked, with its code when that is a cubin this engine holds (the module
// came as a cubin, or as a fatbin with SASS for the device) and no code when it
// is PTX the engine compiles itself.
struct ModuleProf {
  uint32_t id = 0;
  int device = 0;
  std::string cubin;
};
static std::mutex g_module_prof_mu;
static std::unordered_map<uintptr_t, ModuleProf> g_module_prof;

static void announce_module_event(vgpu::profiling::Resource what, const ModuleProf& m) {
  if (!vgpu::profiling::hooked() || vgpu::profiling::silenced()) return;
  vgpu::profiling::ResourceInfo info;
  info.what = what;
  info.device = static_cast<uint32_t>(m.device);
  info.module_id = m.id;
  info.cubin = m.cubin.empty() ? nullptr : m.cubin.data();
  info.cubin_size = m.cubin.size();
  vgpu::profiling::notify_resource(info);
}

static void announce_module_loaded(uintptr_t handle, int device, const uint8_t* cubin, size_t cubin_size,
                                   const std::vector<std::string>& functions) {
  ModuleProf m;
  m.id = vgpu::profiling::next_module_id(static_cast<uint32_t>(device));
  m.device = device;
  if (cubin && cubin_size) m.cubin.assign(reinterpret_cast<const char*>(cubin), cubin_size);
  {
    std::lock_guard<std::mutex> lock(g_module_prof_mu);
    g_module_prof[handle] = m;
  }
  announce_module_event(vgpu::profiling::Resource::ModuleLoaded, m);
  if (!vgpu::profiling::enabled()) return;
  uint32_t index = 0;
  for (const std::string& name : functions) {   // in the order the source declares them (see the runtime's)
    vgpu::profiling::Event e;
    e.kind = vgpu::profiling::EventKind::Function;
    e.start_ns = e.end_ns = vgpu::profiling::host_ns();
    e.device = static_cast<uint32_t>(device);
    e.correlation = vgpu::profiling::work_correlation();
    e.module_id = m.id;
    e.function_index = index++;
    e.name = name;
    e.handle = vgpu::profiling::next_function_id();
    vgpu::profiling::record(std::move(e));
  }
}

static CUresult cuModuleLoadData_impl(CUmodule* module, const void* image) {
  return api("cuModuleLoadData", true, false, [&](ShimState& s) {
    if (!module || !image) return CUDA_ERROR_INVALID_VALUE;
    const char* text = static_cast<const char*>(image);
    std::string extracted;
    uint32_t magic = 0;
    std::memcpy(&magic, image, 4);
    int dev = current_device(s);
    uint64_t mid = 0;
    std::vector<uint8_t> loaded_cubin;   // the module's code, when it is a cubin (for a profiler)
    const vgpu::DeviceProfile& prof = s.rt->device(dev).profile();
    const uint32_t cc = static_cast<uint32_t>(prof.cc_major * 10 + prof.cc_minor);
    if (magic == 0x466243B1u || magic == 0xBA55ED50u) {
      // A fatbin (wrapper or container): its SASS for this GPU if it has
      // some, as the real driver runs; else its PTX.
      const std::string cubin = vgpu::cuda::pick_cubin(image, cc);
      if (!cubin.empty()) {
        mid = s.rt->device(dev).load_cubin(reinterpret_cast<const uint8_t*>(cubin.data()), cubin.size());
        loaded_cubin.assign(cubin.begin(), cubin.end());
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
      // A bare cubin: its size is in its own headers (see bare_cubin_size).
      const uint64_t size = bare_cubin_size(b);
      uint32_t eflags;
      std::memcpy(&eflags, b + 0x30, 4);
      // The architecture is in e_flags' second byte from ABI version 8 on,
      // and its low byte before (as sass::parse_cubin reads it): a CUDA 12
      // cubin for sm_86 carries 0x560556.
      const int sm = static_cast<int>(b[7] == 0x33 ? eflags & 0xff : (eflags >> 8) & 0xff);
      // An sm_XYa cubin runs on XY alone, a plain one on its major from XY up.
      if (!vgpu::sass::runs_on(sm, vgpu::sass::cubin_arch_specific(b, size), static_cast<int>(cc)))
        return CUDA_ERROR_NO_BINARY_FOR_GPU;
      const std::string aside = ptx_for_unsupported_cubin(b, size);
      try {
        mid = !aside.empty() ? s.rt->device(dev).load_module(aside.c_str()) : s.rt->device(dev).load_cubin(b, size);
      } catch (const vgpu::Error& e) {
        if (e.code() == vgpu::Err::InvalidValue) return CUDA_ERROR_INVALID_IMAGE;
        throw;
      }
      loaded_cubin.assign(b, b + size);
    } else {
      mid = s.rt->device(dev).load_module(text);
    }
    bind_managed_globals(s, dev, mid);
    uintptr_t h = make_handle(s, kTagModule);
    s.modules[h] = {dev, mid};
    *module = reinterpret_cast<CUmodule>(h);
    announce_module_loaded(h, dev, loaded_cubin.data(), loaded_cubin.size(),
                           [&] {
                             std::vector<std::string> names;
                             if (vgpu::profiling::enabled()) names = s.rt->device(dev).kernel_names(mid);
                             if (!loaded_cubin.empty()) std::reverse(names.begin(), names.end());
                             return names;
                           }());
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuModuleLoadData(CUmodule* module, const void* image) {
  return traced("cuModuleLoadData", cuModuleLoadData_impl, module, image);
}

// A fatbin handed straight to the driver takes the same path as one passed to
// cuModuleLoadData: pull the PTX out and load it.
static CUresult cuModuleLoadFatBinary_impl(CUmodule* module, const void* fatCubin) {
  return cuModuleLoadData_impl(module, fatCubin);
}
VGPU_EXPORT CUresult cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) {
  return traced("cuModuleLoadFatBinary", cuModuleLoadFatBinary_impl, module, fatCubin);
}

// A module from a file: its bytes, loaded as cuModuleLoadData loads them.
// PTX text is terminated, as the driver requires of an image in memory.
static CUresult cuModuleLoad_impl(CUmodule* module, const char* fname) {
  if (!module || !fname) return CUDA_ERROR_INVALID_VALUE;
  std::ifstream f(fname, std::ios::binary);
  if (!f) return CUDA_ERROR_FILE_NOT_FOUND;
  std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (bytes.size() < 4) return CUDA_ERROR_INVALID_IMAGE;
  return cuModuleLoadData_impl(module, bytes.c_str());
}
VGPU_EXPORT CUresult cuModuleLoad(CUmodule* module, const char* fname) {
  return traced("cuModuleLoad", cuModuleLoad_impl, module, fname);
}

// Releasing the primary context. Nothing is cached per context here, so this
// succeeds without tearing down the device -- except a fault a kernel left the
// context with, which a reset is the one way out of, as cudaDeviceReset is.
//
// Measured on an RTX 3060: the reset leaves the retain count alone (the release that follows it
// succeeds, a second one is CUDA_ERROR_INVALID_CONTEXT), takes the context out of the active state,
// and sets the flags back to 0; a device that does not exist is CUDA_ERROR_INVALID_DEVICE.
static CUresult cuDevicePrimaryCtxReset_impl(CUdevice dev);
VGPU_EXPORT CUresult cuDevicePrimaryCtxReset(CUdevice dev) { return traced("cuDevicePrimaryCtxReset", cuDevicePrimaryCtxReset_impl, dev); }
static CUresult cuDevicePrimaryCtxReset_impl(CUdevice dev) {
  return api("cuDevicePrimaryCtxReset", true, false, [&](ShimState& s) {
    check_device(s, dev);
    s.rt->set_context_fault(0);
    s.primary_flags[dev] = 0;
    s.primary_active.erase(dev);
    if (const auto it = s.primary_ctx.find(dev); it != s.primary_ctx.end())
      s.ctx_state[it->second].flags = kCtxMapHost;
    return CUDA_SUCCESS;
  });
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

static CUresult cuLinkAddFile_v2_impl(void* state, int type, const char* path, unsigned int, void*, void*);
VGPU_EXPORT CUresult cuLinkAddFile_v2(void* state, int type, const char* path, unsigned int a3, void* a4, void* a5) { return traced("cuLinkAddFile_v2", cuLinkAddFile_v2_impl, state, type, path, a3, a4, a5); }
static CUresult cuLinkAddFile_v2_impl(void* state, int type, const char* path, unsigned int,
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
static CUresult cuLinkAddFile_impl(void* state, int type, const char* path, unsigned int n, void* keys, void* vals);
VGPU_EXPORT CUresult cuLinkAddFile(void* state, int type, const char* path, unsigned int n, void* keys, void* vals) { return traced("cuLinkAddFile", cuLinkAddFile_impl, state, type, path, n, keys, vals); }
static CUresult cuLinkAddFile_impl(void* state, int type, const char* path, unsigned int n,
                                   void* keys, void* vals) {
  return cuLinkAddFile_v2(state, type, path, n, keys, vals);
}

static CUresult cuLinkDestroy_impl(void* state);
VGPU_EXPORT CUresult cuLinkDestroy(void* state) { return traced("cuLinkDestroy", cuLinkDestroy_impl, state); }
static CUresult cuLinkDestroy_impl(void* state) {
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
  // Measured on an RTX 3060: the POSIX file descriptor type (1) makes the memory exportable
  // (cuMemExportToShareableHandle); the Win32 (2), Win32 KMT (4) and fabric (8) types, alone or
  // with another, are CUDA_ERROR_INVALID_VALUE; a bit beyond them is ignored.
  const int types = prop->requestedHandleTypes;
  if (types & (2 | 4 | 8)) return CUDA_ERROR_INVALID_VALUE;
  if (device_out) *device_out = prop->location.id;
  return CUDA_SUCCESS;
}
}  // namespace

static CUresult cuMemAddressReserve_impl(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags);
VGPU_EXPORT CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) { return traced("cuMemAddressReserve", cuMemAddressReserve_impl, ptr, size, alignment, addr, flags); }
static CUresult cuMemAddressReserve_impl(CUdeviceptr* ptr, size_t size, size_t alignment,
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

static CUresult cuMemAddressFree_impl(CUdeviceptr ptr, size_t size);
VGPU_EXPORT CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) { return traced("cuMemAddressFree", cuMemAddressFree_impl, ptr, size); }
static CUresult cuMemAddressFree_impl(CUdeviceptr ptr, size_t size) {
  return api("cuMemAddressFree", true, false, [&](ShimState& s) {
    owner_memory(s, ptr).address_free(ptr, size);
    return CUDA_SUCCESS;
  });
}

static CUresult cuMemCreate_impl(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags);
VGPU_EXPORT CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags) { return traced("cuMemCreate", cuMemCreate_impl, handle, size, prop, flags); }
static CUresult cuMemCreate_impl(CUmemGenericAllocationHandle* handle, size_t size,
                                 const CUmemAllocationProp* prop, unsigned long long flags) {
  return api("cuMemCreate", true, false, [&](ShimState& s) {
    if (!handle || size == 0 || flags != 0) return CUDA_ERROR_INVALID_VALUE;
    int device = 0;
    if (const CUresult rc = check_prop(prop, s, &device); rc != CUDA_SUCCESS) return rc;
    *handle = s.rt->device(device).memory().create_handle(size, prop->requestedHandleTypes & 1);
    return CUDA_SUCCESS;
  });
}

static CUresult cuMemRelease_impl(CUmemGenericAllocationHandle handle);
VGPU_EXPORT CUresult cuMemRelease(CUmemGenericAllocationHandle handle) { return traced("cuMemRelease", cuMemRelease_impl, handle); }
static CUresult cuMemRelease_impl(CUmemGenericAllocationHandle handle) {
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

static CUresult cuMemMap_impl(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags);
VGPU_EXPORT CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) { return traced("cuMemMap", cuMemMap_impl, ptr, size, offset, handle, flags); }
static CUresult cuMemMap_impl(CUdeviceptr ptr, size_t size, size_t offset,
                              CUmemGenericAllocationHandle handle, unsigned long long flags) {
  return api("cuMemMap", true, false, [&](ShimState& s) {
    if (flags != 0) return CUDA_ERROR_INVALID_VALUE;
    owner_memory(s, ptr).map(ptr, size, offset, handle);
    return CUDA_SUCCESS;
  });
}

static CUresult cuMemUnmap_impl(CUdeviceptr ptr, size_t size);
VGPU_EXPORT CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) { return traced("cuMemUnmap", cuMemUnmap_impl, ptr, size); }
static CUresult cuMemUnmap_impl(CUdeviceptr ptr, size_t size) {
  return api("cuMemUnmap", true, false, [&](ShimState& s) {
    owner_memory(s, ptr).unmap(ptr, size);
    return CUDA_SUCCESS;
  });
}

static CUresult cuMemSetAccess_impl(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count);
VGPU_EXPORT CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) { return traced("cuMemSetAccess", cuMemSetAccess_impl, ptr, size, desc, count); }
static CUresult cuMemSetAccess_impl(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc,
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

static CUresult cuMemGetAccess_impl(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr);
VGPU_EXPORT CUresult cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr) { return traced("cuMemGetAccess", cuMemGetAccess_impl, flags, location, ptr); }
static CUresult cuMemGetAccess_impl(unsigned long long* flags, const CUmemLocation* location,
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

static CUresult cuMemGetAllocationGranularity_impl(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option);
VGPU_EXPORT CUresult cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option) { return traced("cuMemGetAllocationGranularity", cuMemGetAllocationGranularity_impl, granularity, prop, option); }
static CUresult cuMemGetAllocationGranularity_impl(size_t* granularity,
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

static CUresult cuMemGetAllocationPropertiesFromHandle_impl(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle);
VGPU_EXPORT CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle) { return traced("cuMemGetAllocationPropertiesFromHandle", cuMemGetAllocationPropertiesFromHandle_impl, prop, handle); }
static CUresult cuMemGetAllocationPropertiesFromHandle_impl(CUmemAllocationProp* prop,
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
      prop->requestedHandleTypes = s.rt->device(d).memory().handle_types(handle);
      prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop->location.id = d;
      return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_VALUE;
  });
}

static CUresult cuMemRetainAllocationHandle_impl(CUmemGenericAllocationHandle* handle, void* addr);
VGPU_EXPORT CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle, void* addr) { return traced("cuMemRetainAllocationHandle", cuMemRetainAllocationHandle_impl, handle, addr); }
static CUresult cuMemRetainAllocationHandle_impl(CUmemGenericAllocationHandle* handle, void* addr) {
  return api("cuMemRetainAllocationHandle", true, false, [&](ShimState& s) {
    if (!handle || !addr) return CUDA_ERROR_INVALID_VALUE;
    const auto va = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(addr));
    const uint64_t h = owner_memory(s, va).retain_handle_at(va);
    if (!h) return CUDA_ERROR_INVALID_VALUE;
    *handle = h;
    return CUDA_SUCCESS;
  });
}

// A handle made with the POSIX file descriptor handle type exports as a descriptor for an anonymous
// file that holds its memory, which another process (or this one) imports with
// cuMemImportFromShareableHandle and maps. Measured on an RTX 3060: exporting a handle that was not
// made with the type, with another type, or with flags is CUDA_ERROR_INVALID_VALUE; importing a
// descriptor that names no such file is CUDA_ERROR_INVALID_DEVICE (so is a Win32 type, which the
// card cannot take), the fabric type is CUDA_ERROR_NOT_SUPPORTED, and a type that is no single
// type is CUDA_ERROR_INVALID_VALUE.
static CUresult cuMemExportToShareableHandle_impl(void* shareableHandle, unsigned long long handle, int handleType, unsigned long long flags);
VGPU_EXPORT CUresult cuMemExportToShareableHandle(void* shareableHandle, unsigned long long handle, int handleType, unsigned long long flags) { return traced("cuMemExportToShareableHandle", cuMemExportToShareableHandle_impl, shareableHandle, handle, handleType, flags); }
static CUresult cuMemExportToShareableHandle_impl(void* shareableHandle, unsigned long long handle,
                                                  int handleType, unsigned long long flags) {
  return api("cuMemExportToShareableHandle", true, false, [&](ShimState& s) {
    if (!shareableHandle || !handle || flags != 0 || handleType != 1) return CUDA_ERROR_INVALID_VALUE;
    for (int d = 0; d < s.rt->device_count(); ++d) {
      try {
        s.rt->device(d).memory().handle_size(handle);
      } catch (const vgpu::Error&) {
        continue;
      }
      try {
        *static_cast<int*>(shareableHandle) = s.rt->device(d).memory().export_handle(handle);
      } catch (const vgpu::Error& e) {
        if (e.code() != vgpu::Err::InvalidValue) throw;
        return CUDA_ERROR_INVALID_VALUE;
      }
      return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_VALUE;
  });
}
static CUresult cuMemImportFromShareableHandle_impl(unsigned long long* handle, void* osHandle, int shHandleType);
VGPU_EXPORT CUresult cuMemImportFromShareableHandle(unsigned long long* handle, void* osHandle, int shHandleType) { return traced("cuMemImportFromShareableHandle", cuMemImportFromShareableHandle_impl, handle, osHandle, shHandleType); }
static CUresult cuMemImportFromShareableHandle_impl(unsigned long long* handle, void* osHandle, int shHandleType) {
  return api("cuMemImportFromShareableHandle", true, false, [&](ShimState& s) {
    if (!handle) return CUDA_ERROR_INVALID_VALUE;
    switch (shHandleType) {
      case 1: break;
      case 2:
      case 4: return CUDA_ERROR_INVALID_DEVICE;
      case 8: return CUDA_ERROR_NOT_SUPPORTED;
      default: return CUDA_ERROR_INVALID_VALUE;
    }
    // The descriptor is passed as the pointer-sized value the API takes.
    const int fd = static_cast<int>(reinterpret_cast<intptr_t>(osHandle));
    try {
      *handle = current(s).memory().import_handle(fd);
    } catch (const vgpu::Error& e) {
      if (e.code() == vgpu::Err::NotFound) return CUDA_ERROR_INVALID_DEVICE;
      throw;
    }
    return CUDA_SUCCESS;
  });
}

// Multicast objects span several devices' memory; there is no such fabric here.
static CUresult cuMulticastCreate_impl(unsigned long long*, const void*);
VGPU_EXPORT CUresult cuMulticastCreate(unsigned long long* a0, const void* a1) { return traced("cuMulticastCreate", cuMulticastCreate_impl, a0, a1); }
static CUresult cuMulticastCreate_impl(unsigned long long*, const void*) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
static CUresult cuMulticastAddDevice_impl(unsigned long long, CUdevice);
VGPU_EXPORT CUresult cuMulticastAddDevice(unsigned long long a0, CUdevice a1) { return traced("cuMulticastAddDevice", cuMulticastAddDevice_impl, a0, a1); }
static CUresult cuMulticastAddDevice_impl(unsigned long long, CUdevice) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
static CUresult cuMulticastBindMem_impl(unsigned long long, size_t, unsigned long long, size_t, size_t, unsigned long long);
VGPU_EXPORT CUresult cuMulticastBindMem(unsigned long long a0, size_t a1, unsigned long long a2, size_t a3, size_t a4, unsigned long long a5) { return traced("cuMulticastBindMem", cuMulticastBindMem_impl, a0, a1, a2, a3, a4, a5); }
static CUresult cuMulticastBindMem_impl(unsigned long long, size_t, unsigned long long, size_t,
                                        size_t, unsigned long long) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
static CUresult cuMulticastBindAddr_impl(unsigned long long, size_t, unsigned long long, size_t, unsigned long long);
VGPU_EXPORT CUresult cuMulticastBindAddr(unsigned long long a0, size_t a1, unsigned long long a2, size_t a3, unsigned long long a4) { return traced("cuMulticastBindAddr", cuMulticastBindAddr_impl, a0, a1, a2, a3, a4); }
static CUresult cuMulticastBindAddr_impl(unsigned long long, size_t, unsigned long long, size_t,
                                         unsigned long long) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
static CUresult cuMulticastUnbind_impl(unsigned long long, CUdevice, size_t, size_t);
VGPU_EXPORT CUresult cuMulticastUnbind(unsigned long long a0, CUdevice a1, size_t a2, size_t a3) { return traced("cuMulticastUnbind", cuMulticastUnbind_impl, a0, a1, a2, a3); }
static CUresult cuMulticastUnbind_impl(unsigned long long, CUdevice, size_t, size_t) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
static CUresult cuMulticastGetGranularity_impl(size_t*, const void*, int);
VGPU_EXPORT CUresult cuMulticastGetGranularity(size_t* a0, const void* a1, int a2) { return traced("cuMulticastGetGranularity", cuMulticastGetGranularity_impl, a0, a1, a2); }
static CUresult cuMulticastGetGranularity_impl(size_t*, const void*, int) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

// Stream memory ops. Work is synchronous, so the write happens now.
static CUresult cuStreamWriteValue32_impl(CUstream, CUdeviceptr addr, unsigned int value, unsigned int);
VGPU_EXPORT CUresult cuStreamWriteValue32(CUstream a0, CUdeviceptr addr, unsigned int value, unsigned int a3) { return traced("cuStreamWriteValue32", cuStreamWriteValue32_impl, a0, addr, value, a3); }
static CUresult cuStreamWriteValue32_impl(CUstream, CUdeviceptr addr, unsigned int value,
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
static CUresult cuModuleLoadDataEx_impl(CUmodule* module, const void* image, unsigned int numOptions, void* options, void** optionValues) {
  const CUresult r = cuModuleLoadData_impl(module, image);
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
VGPU_EXPORT CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, void* options, void** optionValues) {
  return traced("cuModuleLoadDataEx", cuModuleLoadDataEx_impl, module, image, numOptions, options, optionValues);
}

static CUresult cuModuleUnload_impl(CUmodule hmod) {
  return api("cuModuleUnload", true, false, [&](ShimState& s) {
    uintptr_t h = check_handle(reinterpret_cast<uintptr_t>(hmod), kTagModule, "module");
    auto it = s.modules.find(h);
    if (it == s.modules.end()) return CUDA_ERROR_NOT_FOUND;
    auto [dev, mid] = it->second;
    {
      ModuleProf m;
      bool known = false;
      {
        std::lock_guard<std::mutex> lock(g_module_prof_mu);
        const auto p = g_module_prof.find(h);
        if (p != g_module_prof.end()) {
          m = std::move(p->second);
          g_module_prof.erase(p);
          known = true;
        }
      }
      if (known) announce_module_event(vgpu::profiling::Resource::ModuleUnloadStarting, m);
    }
    unload_module(s, dev, mid);
    s.modules.erase(it);
    for (auto fit = s.functions.begin(); fit != s.functions.end();) {
      // Function handles from this module are now dangling; drop them.
      fit = fit->second.module_handle == h ? s.functions.erase(fit) : std::next(fit);
    }
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuModuleUnload(CUmodule hmod) {
  return traced("cuModuleUnload", cuModuleUnload_impl, hmod);
}

static CUresult cuModuleGetFunction_impl(CUfunction* hfunc, CUmodule hmod, const char* name) {
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
VGPU_EXPORT CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
  return traced("cuModuleGetFunction", cuModuleGetFunction_impl, hfunc, hmod, name);
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

constexpr CUresult kUnsupportedLimit = static_cast<CUresult>(215);   // kUnsupportedLimit = cudaErrorUnsupportedLimit

// A limit by its number (CUlimit's and cudaLimit's agree) on a device with
// compute capability `cc_major`: what cuCtxGetLimit, and a kernel's
// cudaDeviceGetLimit, read. kUnsupportedLimit for a number that
// names none, and for the sync depth from compute capability 9.0, which has none.
CUresult limit_value(const ShimState::Limits& l, const vgpu::DeviceProfile& profile, int limit, size_t* v) {
  const int cc_major = profile.cc_major;
  switch (limit) {
    case 0: *v = l.stack; return CUDA_SUCCESS;
    case 1: *v = l.printf_fifo; return CUDA_SUCCESS;
    case 2: *v = l.malloc_heap; return CUDA_SUCCESS;
    case 3:
      if (cc_major >= 9) return kUnsupportedLimit;
      *v = l.sync_depth;
      return CUDA_SUCCESS;
    case 4: *v = l.pending_launches; return CUDA_SUCCESS;
    case 5: *v = l.l2_fetch_granularity; return CUDA_SUCCESS;
    case 6: *v = l.persisting_l2; return CUDA_SUCCESS;
    // The shared memory a multiprocessor offers less what the system keeps back (an RTX 3060: 100 KiB
    // less 1 KiB, 101376), and whether CUDA in Graphics is enabled (never).
    case 7: *v = profile.limits.shared_mem_per_sm - profile.reserved_smem_per_block(); return CUDA_SUCCESS;
    case 8: *v = 0; return CUDA_SUCCESS;
    case 9: return kUnsupportedLimit;
    default: return CUDA_ERROR_INVALID_VALUE;
  }
}

// What a kernel's calls into the device runtime (cudaDeviceGetAttribute,
// cudaDeviceGetLimit, error strings...) ask of this library: the answers the
// host's own calls give, so a kernel sees what its host sees. cuCtx*Limit,
// the cache configuration and the attributes are this shim's own.
struct DriverDevRt final : vgpu::exec::devrt::Services {
  int attribute(int device, int attr, int* value) const override {
    // The attribute numbers the device runtime answers (1 to 148) are
    // CUdevice_attribute's too.
    ShimState& s = state();
    std::lock_guard<std::recursive_mutex> lock(s.mu);
    if (!s.initialized) return 3;   // cudaErrorInitializationError
    try {
      return static_cast<int>(device_attribute_by_id(s, value, attr, device));
    } catch (const vgpu::Error&) {
      return 101;   // cudaErrorInvalidDevice: a device the machine has not
    }
  }
  int limit(int device, int id, uint64_t* value) const override {
    ShimState& s = state();
    std::lock_guard<std::recursive_mutex> lock(s.mu);
    if (!s.initialized || device < 0 || device >= s.rt->device_count()) return 101;   // cudaErrorInvalidDevice
    size_t v = 0;
    const CUresult r = limit_value(s.limits[device], s.rt->device(device).profile(), id, &v);
    if (r == CUDA_SUCCESS) *value = v;
    // kUnsupportedLimit is cudaErrorUnsupportedLimit, 215, in both.
    return static_cast<int>(r);
  }
  int cache_config(int, int* value) const override {
    ShimState& s = state();
    std::lock_guard<std::recursive_mutex> lock(s.mu);
    *value = s.cache_config;
    return 0;
  }
  int shared_mem_config(int, int* value) const override {
    *value = 1;   // CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE
    return 0;
  }
  const char* error_name(int code) const override {
    const vgpu::cuda::ErrorInfo* e = vgpu::cuda::find_error(code);
    return e && e->runtime_name ? e->runtime_name : "unrecognized error code";
  }
  const char* error_string(int code) const override {
    const vgpu::cuda::ErrorInfo* e = vgpu::cuda::find_error(code);
    return e && e->runtime_name ? e->text : "unrecognized error code";
  }
};
const vgpu::exec::devrt::Services& driver_devrt_services() {
  static const DriverDevRt services;
  return services;
}

CUresult launch_kernel_common(const char* api_name, CUfunction f, unsigned int gridDimX,
                              unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX,
                              unsigned int blockDimY, unsigned int blockDimZ,
                              unsigned int sharedMemBytes, CUstream hStream, void** kernelParams,
                              void** extra, bool cooperative,
                              std::array<uint32_t, 3> cluster = {0, 0, 0}) {
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
    cfg.cluster = cluster;
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
    // On a stream that is capturing (a CUDA graph PyTorch is recording; the
    // capture itself is the runtime's) the launch belongs to the graph: it
    // becomes a kernel node and runs, repeated with these arguments, at each
    // launch of the graph.
    using RecordFn = bool (*)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, size_t,
                              const std::function<void()>*);
    if (auto record = reinterpret_cast<RecordFn>(dlsym(RTLD_DEFAULT, "vgpu_record_driver_launch_v1"))) {
      const std::function<void()> again = [api_name, f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                                           blockDimZ, sharedMemBytes, args, cooperative, cluster] {
        std::vector<void*> ptrs(args.size());
        for (size_t i = 0; i < args.size(); ++i) ptrs[i] = const_cast<uint8_t*>(args[i].data());
        launch_kernel_common(api_name, f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                             sharedMemBytes, nullptr, ptrs.data(), nullptr, cooperative, cluster);
      };
      if (record(reinterpret_cast<void*>(hStream), gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                 sharedMemBytes, &again))
        return CUDA_SUCCESS;
    }
    // What the device is limited to, and what a kernel's device runtime asks of this library.
    const ShimState::Limits& lim = s.limits[rec.device];
    cfg.device_heap_bytes = lim.malloc_heap;
    cfg.stack_bytes = lim.stack;
    cfg.devrt = &driver_devrt_services();
    if (vgpu::profiling::hooked()) {
      ModuleProf m;
      bool known = false;
      {
        std::lock_guard<std::mutex> lock(g_module_prof_mu);
        const auto p = g_module_prof.find(rec.module_handle);
        if (p != g_module_prof.end()) {
          m = p->second;
          known = true;
        }
      }
      if (known) announce_module_event(vgpu::profiling::Resource::ModuleProfiled, m);
    }
    const bool profiling = vgpu::profiling::enabled();
    const uint64_t t0 = profiling ? vgpu::profiling::now_ns() : 0;
    s.rt->device(rec.device).launch(*rec.fn, cfg, args, rec.syms);
    if (profiling) {
      vgpu::profiling::Event ev;
      ev.kind = vgpu::profiling::EventKind::Kernel;
      ev.start_ns = t0;
      ev.end_ns = vgpu::profiling::now_ns();
      ev.device = static_cast<uint32_t>(rec.device);
      ev.correlation = vgpu::profiling::work_correlation();
      ev.stream = reinterpret_cast<uint64_t>(hStream);
      ev.name = rec.fn->name;
      ev.grid[0] = gridDimX; ev.grid[1] = gridDimY; ev.grid[2] = gridDimZ;
      ev.block[0] = blockDimX; ev.block[1] = blockDimY; ev.block[2] = blockDimZ;
      ev.shared_bytes = sharedMemBytes;
      ev.static_shared_bytes = rec.fn->static_shared_size;
      {
        const vgpu::DeviceProfile& p = s.rt->device(rec.device).profile();
        const auto res = vgpu::exec::kernel_resources(*rec.fn, p, p.limits.max_threads_per_block, 0);
        ev.registers_per_thread = res.usage.regs_per_thread;
        ev.local_bytes_per_thread = res.usage.local_bytes;
      }
      vgpu::profiling::record(std::move(ev));
    }
    return CUDA_SUCCESS;
  });
}
}  // namespace

static CUresult cuLaunchKernel_impl(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                                    unsigned int gridDimZ, unsigned int blockDimX,
                                    unsigned int blockDimY, unsigned int blockDimZ,
                                    unsigned int sharedMemBytes, CUstream hStream,
                                    void** kernelParams, void** extra) {
  return launch_kernel_common("cuLaunchKernel", f, gridDimX, gridDimY, gridDimZ, blockDimX,
                              blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra,
                              /*cooperative=*/false);
}
VGPU_EXPORT CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                                    unsigned int gridDimZ, unsigned int blockDimX,
                                    unsigned int blockDimY, unsigned int blockDimZ,
                                    unsigned int sharedMemBytes, CUstream hStream,
                                    void** kernelParams, void** extra) {
  if (vgpu::profiling::hooked()) vgpu::profiling::note_symbol(launch_symbol(f));
  return traced("cuLaunchKernel", cuLaunchKernel_impl, f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

/* ======================================================================== */
/* Extended surface for hosting NVIDIA's static cudart (unmodified binaries) */
/* ======================================================================== */

/* ---- context-independent libraries (CUDA 12+ cuLibrary API) ---- */

static CUresult cuLibraryLoadData_impl(void** library, const void* code, void* jitOptions, void** jitOptionValues, unsigned int numJitOptions, void* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions);
VGPU_EXPORT CUresult cuLibraryLoadData(void** library, const void* code, void* jitOptions, void** jitOptionValues, unsigned int numJitOptions, void* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) { return traced("cuLibraryLoadData", cuLibraryLoadData_impl, library, code, jitOptions, jitOptionValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions); }
static CUresult cuLibraryLoadData_impl(void** library, const void* code, void* jitOptions,
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
    if (magic == 0x466243B1u || magic == 0xBA55ED50u) {
      lib.ptx = best_ptx(code);
    } else if (magic == 0x464c457fu) {
      // A cubin (NVRTC's, for one): run as SASS, or as the PTX NVRTC made it from where the SASS engine
      // cannot yet. The same checks as cuModuleLoadData's.
      const auto* b = static_cast<const uint8_t*>(code);
      if (b[4] != 2 || (b[7] != 0x41 && b[7] != 0x33)) return CUDA_ERROR_INVALID_IMAGE;
      const size_t size = bare_cubin_size(b);
      std::string aside = ptx_for_unsupported_cubin(b, size);
      if (!aside.empty()) lib.ptx = std::move(aside);
      else lib.cubin.assign(reinterpret_cast<const char*>(b), size);
    } else {
      lib.ptx.assign(static_cast<const char*>(code));  // NUL-terminated PTX text
    }
    uintptr_t h = make_handle(s, kTagLibrary);
    s.libraries[h] = std::move(lib);
    *library = reinterpret_cast<void*>(h);
    return CUDA_SUCCESS;
  });
}

static CUresult cuLibraryUnload_impl(void* library);
VGPU_EXPORT CUresult cuLibraryUnload(void* library) { return traced("cuLibraryUnload", cuLibraryUnload_impl, library); }
static CUresult cuLibraryUnload_impl(void* library) {
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

static CUresult cuLibraryGetKernel_impl(void** pKernel, void* library, const char* name);
VGPU_EXPORT CUresult cuLibraryGetKernel(void** pKernel, void* library, const char* name) { return traced("cuLibraryGetKernel", cuLibraryGetKernel_impl, pKernel, library, name); }
static CUresult cuLibraryGetKernel_impl(void** pKernel, void* library, const char* name) {
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

static CUresult cuLibraryGetModule_impl(CUmodule* pMod, void* library);
VGPU_EXPORT CUresult cuLibraryGetModule(CUmodule* pMod, void* library) { return traced("cuLibraryGetModule", cuLibraryGetModule_impl, pMod, library); }
static CUresult cuLibraryGetModule_impl(CUmodule* pMod, void* library) {
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

static CUresult cuKernelGetFunction_impl(CUfunction* pFunc, void* kernel);
VGPU_EXPORT CUresult cuKernelGetFunction(CUfunction* pFunc, void* kernel) { return traced("cuKernelGetFunction", cuKernelGetFunction_impl, pFunc, kernel); }
static CUresult cuKernelGetFunction_impl(CUfunction* pFunc, void* kernel) {
  return api("cuKernelGetFunction", true, false, [&](ShimState& s) {
    if (!pFunc) return CUDA_ERROR_INVALID_VALUE;
    uintptr_t fh = kernel_to_function(s, reinterpret_cast<uintptr_t>(kernel));
    *pFunc = reinterpret_cast<CUfunction>(fh);
    return CUDA_SUCCESS;
  });
}

static CUresult cuKernelGetName_impl(const char** name, void* kernel);
VGPU_EXPORT CUresult cuKernelGetName(const char** name, void* kernel) { return traced("cuKernelGetName", cuKernelGetName_impl, name, kernel); }
static CUresult cuKernelGetName_impl(const char** name, void* kernel) {
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

static CUresult cuFuncGetAttribute_impl(int* pi, int attrib, CUfunction hfunc);
VGPU_EXPORT CUresult cuFuncGetAttribute(int* pi, int attrib, CUfunction hfunc) { return traced("cuFuncGetAttribute", cuFuncGetAttribute_impl, pi, attrib, hfunc); }
static CUresult cuFuncGetAttribute_impl(int* pi, int attrib, CUfunction hfunc) {
  return api("cuFuncGetAttribute", true, false, [&](ShimState& s) {
    if (!pi) return CUDA_ERROR_INVALID_VALUE;
    auto it = s.functions.find(reinterpret_cast<uintptr_t>(hfunc));
    if (it == s.functions.end()) return CUDA_ERROR_INVALID_VALUE;
    *pi = func_attribute(it->second.fn, s.rt->device(it->second.device).profile(), attrib);
    return CUDA_SUCCESS;
  });
}

static CUresult cuKernelGetAttribute_impl(int* pi, int attrib, void* kernel, CUdevice dev);
VGPU_EXPORT CUresult cuKernelGetAttribute(int* pi, int attrib, void* kernel, CUdevice dev) { return traced("cuKernelGetAttribute", cuKernelGetAttribute_impl, pi, attrib, kernel, dev); }
static CUresult cuKernelGetAttribute_impl(int* pi, int attrib, void* kernel, CUdevice dev) {
  return api("cuKernelGetAttribute", true, false, [&](ShimState& s) {
    if (!pi) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    uintptr_t kh = check_handle(reinterpret_cast<uintptr_t>(kernel), kTagKernel, "kernel");
    if (!s.kernels.count(kh)) return CUDA_ERROR_INVALID_VALUE;
    *pi = func_attribute(nullptr, s.rt->device(dev).profile(), attrib);
    return CUDA_SUCCESS;
  });
}

static CUresult cuFuncSetAttribute_impl(CUfunction hfunc, int attrib, int value);
VGPU_EXPORT CUresult cuFuncSetAttribute(CUfunction hfunc, int attrib, int value) { return traced("cuFuncSetAttribute", cuFuncSetAttribute_impl, hfunc, attrib, value); }
static CUresult cuFuncSetAttribute_impl(CUfunction hfunc, int attrib, int value) {
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
static CUresult cuKernelSetAttribute_impl(int, int, void*, CUdevice);
VGPU_EXPORT CUresult cuKernelSetAttribute(int a0, int a1, void* a2, CUdevice a3) { return traced("cuKernelSetAttribute", cuKernelSetAttribute_impl, a0, a1, a2, a3); }
static CUresult cuKernelSetAttribute_impl(int, int, void*, CUdevice) { return CUDA_SUCCESS; }
static CUresult cuFuncSetCacheConfig_impl(CUfunction, int);
VGPU_EXPORT CUresult cuFuncSetCacheConfig(CUfunction a0, int a1) { return traced("cuFuncSetCacheConfig", cuFuncSetCacheConfig_impl, a0, a1); }
static CUresult cuFuncSetCacheConfig_impl(CUfunction, int) { return CUDA_SUCCESS; }
static CUresult cuFuncIsLoaded_impl(int* state, CUfunction);
VGPU_EXPORT CUresult cuFuncIsLoaded(int* state, CUfunction a1) { return traced("cuFuncIsLoaded", cuFuncIsLoaded_impl, state, a1); }
static CUresult cuFuncIsLoaded_impl(int* state, CUfunction) {
  if (state) *state = 1;  // CU_FUNCTION_LOADING_STATE_LOADED
  return CUDA_SUCCESS;
}
static CUresult cuFuncLoad_impl(CUfunction);
VGPU_EXPORT CUresult cuFuncLoad(CUfunction a0) { return traced("cuFuncLoad", cuFuncLoad_impl, a0); }
static CUresult cuFuncLoad_impl(CUfunction) { return CUDA_SUCCESS; }

// CUlaunchAttribute, as the driver ABI fixes it: a 4-byte id, padded to 8, then
// a union padded to 64 bytes -- 72 bytes an entry. cuda.h of every CUDA 12
// toolkit (12.0 to 12.9) declares exactly that, and it has to stay: an array
// of these crosses the driver boundary, so a toolkit that moved the stride
// would stop its own binaries launching on an older driver. This file reads
// the entries through its own declaration rather than including a vendor
// header (see vgpu_cuda.h).
namespace {
struct LaunchAttrABI {
  uint32_t id;
  char pad[4];
  union {
    char pad64[64];
    int cooperative;                           // CU_LAUNCH_ATTRIBUTE_COOPERATIVE
    struct { uint32_t x, y, z; } cluster_dim;  // CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
  } value;
};
static_assert(sizeof(LaunchAttrABI) == 72, "CUlaunchAttribute is 72 bytes in every CUDA 12 cuda.h");
constexpr uint32_t kLaunchAttrCooperative = 2;       // CU_LAUNCH_ATTRIBUTE_COOPERATIVE
constexpr uint32_t kLaunchAttrClusterDimension = 4;  // CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
}  // namespace

static CUresult cuLaunchKernelEx_impl(const void* config, CUfunction f, void** kernelParams, void** extra);
VGPU_EXPORT CUresult cuLaunchKernelEx(const void* config, CUfunction f, void** kernelParams, void** extra) { return traced("cuLaunchKernelEx", cuLaunchKernelEx_impl, config, f, kernelParams, extra); }
static CUresult cuLaunchKernelEx_impl(const void* config, CUfunction f, void** kernelParams,
                                      void** extra) {
  // CUlaunchConfig: 6x u32 dims, u32 sharedMemBytes, CUstream, attrs*, numAttrs.
  struct LaunchCfgABI {
    unsigned gx, gy, gz, bx, by, bz;
    unsigned shared_bytes;
    CUstream stream;
    const LaunchAttrABI* attrs;
    unsigned num_attrs;
  };
  const auto* c = static_cast<const LaunchCfgABI*>(config);
  if (!c) return CUDA_ERROR_INVALID_VALUE;
  if (c->num_attrs != 0 && c->attrs == nullptr) return CUDA_ERROR_INVALID_VALUE;
  // The attribute list is the point of the Ex form. Two attributes change what
  // the grid does -- a cluster shape and a cooperative launch -- and are
  // honoured. Ignoring them would run a clustered kernel with no cluster, every
  // block reading %cluster_ctarank as 0 and %cluster_nctarank as 1: a launch
  // that succeeds and computes the wrong thing, which this engine is built not
  // to do. (This entry point used to refuse any attribute for fear of reading
  // the entries at the wrong stride; see LaunchAttrABI.)
  //
  // The rest are inert here for reasons already true of the runtime's
  // cudaLaunchKernelEx: priority and memory-sync domains need a stream
  // scheduler, access policy windows a cache model, programmatic events
  // asynchrony, and the preferred cluster shape and shared-memory carveout are
  // preferences a launch is free to ignore. None changes what a kernel computes.
  std::array<uint32_t, 3> cluster{0, 0, 0};
  bool cooperative = false;
  for (unsigned i = 0; i < c->num_attrs; ++i) {
    const LaunchAttrABI& a = c->attrs[i];
    if (a.id == kLaunchAttrClusterDimension)
      cluster = {a.value.cluster_dim.x, a.value.cluster_dim.y, a.value.cluster_dim.z};
    else if (a.id == kLaunchAttrCooperative)
      cooperative = a.value.cooperative != 0;
  }
  return launch_kernel_common("cuLaunchKernelEx", f, c->gx, c->gy, c->gz, c->bx, c->by, c->bz,
                              c->shared_bytes, c->stream, kernelParams, extra, cooperative, cluster);
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

static CUresult cuFuncSetBlockShape_impl(CUfunction f, int x, int y, int z);
VGPU_EXPORT CUresult cuFuncSetBlockShape(CUfunction f, int x, int y, int z) { return traced("cuFuncSetBlockShape", cuFuncSetBlockShape_impl, f, x, y, z); }
static CUresult cuFuncSetBlockShape_impl(CUfunction f, int x, int y, int z) {
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
static CUresult cuFuncSetSharedSize_impl(CUfunction f, unsigned int bytes);
VGPU_EXPORT CUresult cuFuncSetSharedSize(CUfunction f, unsigned int bytes) { return traced("cuFuncSetSharedSize", cuFuncSetSharedSize_impl, f, bytes); }
static CUresult cuFuncSetSharedSize_impl(CUfunction f, unsigned int bytes) {
  return api("cuFuncSetSharedSize", true, false, [&](ShimState& s) {
    FuncRec* rec = legacy_func(s, f);
    if (!rec) return CUDA_ERROR_INVALID_HANDLE;
    if (bytes > s.rt->device(rec->device).profile().limits.shared_mem_per_block) return CUDA_ERROR_INVALID_VALUE;
    rec->shared_bytes = bytes;
    return CUDA_SUCCESS;
  });
}
static CUresult cuParamSetSize_impl(CUfunction f, unsigned int numbytes);
VGPU_EXPORT CUresult cuParamSetSize(CUfunction f, unsigned int numbytes) { return traced("cuParamSetSize", cuParamSetSize_impl, f, numbytes); }
static CUresult cuParamSetSize_impl(CUfunction f, unsigned int numbytes) {
  return api("cuParamSetSize", true, false, [&](ShimState& s) {
    FuncRec* rec = legacy_func(s, f);
    if (!rec) return CUDA_ERROR_INVALID_HANDLE;
    if (numbytes > kMaxParamBytes) return CUDA_ERROR_INVALID_VALUE;
    rec->param_size = numbytes;
    return CUDA_SUCCESS;
  });
}
static CUresult cuParamSeti_impl(CUfunction f, int offset, unsigned int value);
VGPU_EXPORT CUresult cuParamSeti(CUfunction f, int offset, unsigned int value) { return traced("cuParamSeti", cuParamSeti_impl, f, offset, value); }
static CUresult cuParamSeti_impl(CUfunction f, int offset, unsigned int value) {
  return param_set(f, offset, &value, sizeof value);
}
static CUresult cuParamSetf_impl(CUfunction f, int offset, float value);
VGPU_EXPORT CUresult cuParamSetf(CUfunction f, int offset, float value) { return traced("cuParamSetf", cuParamSetf_impl, f, offset, value); }
static CUresult cuParamSetf_impl(CUfunction f, int offset, float value) {
  return param_set(f, offset, &value, sizeof value);
}
static CUresult cuParamSetv_impl(CUfunction f, int offset, void* ptr, unsigned int numbytes);
VGPU_EXPORT CUresult cuParamSetv(CUfunction f, int offset, void* ptr, unsigned int numbytes) { return traced("cuParamSetv", cuParamSetv_impl, f, offset, ptr, numbytes); }
static CUresult cuParamSetv_impl(CUfunction f, int offset, void* ptr, unsigned int numbytes) {
  return param_set(f, offset, ptr, numbytes);
}
// Texture references are bound to a module's kernels by name, not passed; the
// card takes this call and it has nothing to do.
static CUresult cuParamSetTexRef_impl(CUfunction f, int, CUtexref tex);
VGPU_EXPORT CUresult cuParamSetTexRef(CUfunction f, int a1, CUtexref tex) { return traced("cuParamSetTexRef", cuParamSetTexRef_impl, f, a1, tex); }
static CUresult cuParamSetTexRef_impl(CUfunction f, int, CUtexref tex) {
  return api("cuParamSetTexRef", true, false, [&](ShimState& s) {
    if (!legacy_func(s, f) || !s.texrefs.count(reinterpret_cast<uintptr_t>(tex))) return CUDA_ERROR_INVALID_HANDLE;
    return CUDA_SUCCESS;
  });
}
static CUresult cuLaunchGrid_impl(CUfunction f, int grid_width, int grid_height);
VGPU_EXPORT CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) { return traced("cuLaunchGrid", cuLaunchGrid_impl, f, grid_width, grid_height); }
static CUresult cuLaunchGrid_impl(CUfunction f, int grid_width, int grid_height) {
  return launch_grid("cuLaunchGrid", f, grid_width, grid_height, nullptr);
}
static CUresult cuLaunchGridAsync_impl(CUfunction f, int grid_width, int grid_height, CUstream stream);
VGPU_EXPORT CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream stream) { return traced("cuLaunchGridAsync", cuLaunchGridAsync_impl, f, grid_width, grid_height, stream); }
static CUresult cuLaunchGridAsync_impl(CUfunction f, int grid_width, int grid_height, CUstream stream) {
  return launch_grid("cuLaunchGridAsync", f, grid_width, grid_height, stream);
}
static CUresult cuLaunch_impl(CUfunction f);
VGPU_EXPORT CUresult cuLaunch(CUfunction f) { return traced("cuLaunch", cuLaunch_impl, f); }
static CUresult cuLaunch_impl(CUfunction f) { return launch_grid("cuLaunch", f, 1, 1, nullptr); }

/* ---- memcpy/memset variants (everything is synchronous) ---- */

static CUresult cuMemcpyHtoDAsync_impl(CUdeviceptr d, const void* h, size_t n, CUstream st);
VGPU_EXPORT CUresult cuMemcpyHtoDAsync(CUdeviceptr d, const void* h, size_t n, CUstream st) { return traced("cuMemcpyHtoDAsync", cuMemcpyHtoDAsync_impl, d, h, n, st); }
static CUresult cuMemcpyHtoDAsync_impl(CUdeviceptr d, const void* h, size_t n, CUstream st) {
  return cuMemcpyHtoDAsync_v2_impl(d, h, n, st);
}
static CUresult cuMemcpyDtoHAsync_impl(void* h, CUdeviceptr d, size_t n, CUstream st);
VGPU_EXPORT CUresult cuMemcpyDtoHAsync(void* h, CUdeviceptr d, size_t n, CUstream st) { return traced("cuMemcpyDtoHAsync", cuMemcpyDtoHAsync_impl, h, d, n, st); }
static CUresult cuMemcpyDtoHAsync_impl(void* h, CUdeviceptr d, size_t n, CUstream st) {
  return cuMemcpyDtoHAsync_v2_impl(h, d, n, st);
}
static CUresult cuMemcpyDtoDAsync_impl(CUdeviceptr a, CUdeviceptr b, size_t n, CUstream st);
VGPU_EXPORT CUresult cuMemcpyDtoDAsync(CUdeviceptr a, CUdeviceptr b, size_t n, CUstream st) { return traced("cuMemcpyDtoDAsync", cuMemcpyDtoDAsync_impl, a, b, n, st); }
static CUresult cuMemcpyDtoDAsync_impl(CUdeviceptr a, CUdeviceptr b, size_t n, CUstream st) {
  return cuMemcpyDtoDAsync_v2_impl(a, b, n, st);
}

// No profiler collects anything to start or stop; the card succeeds.
static CUresult cuProfilerStart_impl(void);
VGPU_EXPORT CUresult cuProfilerStart(void) { return traced("cuProfilerStart", cuProfilerStart_impl); }
static CUresult cuProfilerStart_impl(void) { return CUDA_SUCCESS; }
static CUresult cuProfilerStop_impl(void);
VGPU_EXPORT CUresult cuProfilerStop(void) { return traced("cuProfilerStop", cuProfilerStop_impl); }
static CUresult cuProfilerStop_impl(void) { return CUDA_SUCCESS; }

/* ---- graphics interop ----
 * There is no OpenGL, Direct3D or Vulkan here to register a resource with, so
 * no CUgraphicsResource ever exists: each of these is answered as an RTX 3060
 * answers a null one (INVALID_HANDLE), and mapping or unmapping no resources
 * as it answers that (INVALID_VALUE). */
static CUresult cuGraphicsUnregisterResource_impl(CUgraphicsResource);
VGPU_EXPORT CUresult cuGraphicsUnregisterResource(CUgraphicsResource a0) { return traced("cuGraphicsUnregisterResource", cuGraphicsUnregisterResource_impl, a0); }
static CUresult cuGraphicsUnregisterResource_impl(CUgraphicsResource) { return CUDA_ERROR_INVALID_HANDLE; }
static CUresult cuGraphicsMapResources_impl(unsigned int count, CUgraphicsResource* res, CUstream);
VGPU_EXPORT CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource* res, CUstream a2) { return traced("cuGraphicsMapResources", cuGraphicsMapResources_impl, count, res, a2); }
static CUresult cuGraphicsMapResources_impl(unsigned int count, CUgraphicsResource* res, CUstream) {
  return count == 0 || !res ? CUDA_ERROR_INVALID_VALUE : CUDA_ERROR_INVALID_HANDLE;
}
static CUresult cuGraphicsUnmapResources_impl(unsigned int count, CUgraphicsResource* res, CUstream);
VGPU_EXPORT CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource* res, CUstream a2) { return traced("cuGraphicsUnmapResources", cuGraphicsUnmapResources_impl, count, res, a2); }
static CUresult cuGraphicsUnmapResources_impl(unsigned int count, CUgraphicsResource* res, CUstream) {
  return count == 0 || !res ? CUDA_ERROR_INVALID_VALUE : CUDA_ERROR_INVALID_HANDLE;
}
static CUresult cuGraphicsResourceSetMapFlags_v2_impl(CUgraphicsResource, unsigned int);
VGPU_EXPORT CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource a0, unsigned int a1) { return traced("cuGraphicsResourceSetMapFlags_v2", cuGraphicsResourceSetMapFlags_v2_impl, a0, a1); }
static CUresult cuGraphicsResourceSetMapFlags_v2_impl(CUgraphicsResource, unsigned int) {
  return CUDA_ERROR_INVALID_HANDLE;
}
static CUresult cuGraphicsResourceSetMapFlags_impl(CUgraphicsResource r, unsigned int f);
VGPU_EXPORT CUresult cuGraphicsResourceSetMapFlags(CUgraphicsResource r, unsigned int f) { return traced("cuGraphicsResourceSetMapFlags", cuGraphicsResourceSetMapFlags_impl, r, f); }
static CUresult cuGraphicsResourceSetMapFlags_impl(CUgraphicsResource r, unsigned int f) {
  return cuGraphicsResourceSetMapFlags_v2(r, f);
}
static CUresult cuGraphicsResourceGetMappedPointer_v2_impl(CUdeviceptr*, size_t*, CUgraphicsResource);
VGPU_EXPORT CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr* a0, size_t* a1, CUgraphicsResource a2) { return traced("cuGraphicsResourceGetMappedPointer_v2", cuGraphicsResourceGetMappedPointer_v2_impl, a0, a1, a2); }
static CUresult cuGraphicsResourceGetMappedPointer_v2_impl(CUdeviceptr*, size_t*, CUgraphicsResource) {
  return CUDA_ERROR_INVALID_HANDLE;
}
static CUresult cuGraphicsResourceGetMappedPointer_impl(CUdeviceptr* p, size_t* n, CUgraphicsResource r);
VGPU_EXPORT CUresult cuGraphicsResourceGetMappedPointer(CUdeviceptr* p, size_t* n, CUgraphicsResource r) { return traced("cuGraphicsResourceGetMappedPointer", cuGraphicsResourceGetMappedPointer_impl, p, n, r); }
static CUresult cuGraphicsResourceGetMappedPointer_impl(CUdeviceptr* p, size_t* n, CUgraphicsResource r) {
  return cuGraphicsResourceGetMappedPointer_v2(p, n, r);
}
static CUresult cuGraphicsSubResourceGetMappedArray_impl(CUarray*, CUgraphicsResource, unsigned int, unsigned int);
VGPU_EXPORT CUresult cuGraphicsSubResourceGetMappedArray(CUarray* a0, CUgraphicsResource a1, unsigned int a2, unsigned int a3) { return traced("cuGraphicsSubResourceGetMappedArray", cuGraphicsSubResourceGetMappedArray_impl, a0, a1, a2, a3); }
static CUresult cuGraphicsSubResourceGetMappedArray_impl(CUarray*, CUgraphicsResource, unsigned int, unsigned int) {
  return CUDA_ERROR_INVALID_HANDLE;
}
static CUresult cuGraphicsResourceGetMappedMipmappedArray_impl(CUmipmappedArray*, CUgraphicsResource);
VGPU_EXPORT CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray* a0, CUgraphicsResource a1) { return traced("cuGraphicsResourceGetMappedMipmappedArray", cuGraphicsResourceGetMappedMipmappedArray_impl, a0, a1); }
static CUresult cuGraphicsResourceGetMappedMipmappedArray_impl(CUmipmappedArray*, CUgraphicsResource) {
  return CUDA_ERROR_INVALID_HANDLE;
}

namespace {
bool is_device_ptr(uint64_t p) { return vgpu::is_device_va(p); }
}  // namespace

// Direction-inferring copies (UVA style): device-range vs host pointers.
static CUresult cuMemcpy_impl(CUdeviceptr dst, CUdeviceptr src, size_t n);
VGPU_EXPORT CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t n) { return traced("cuMemcpy", cuMemcpy_impl, dst, src, n); }
static CUresult cuMemcpy_impl(CUdeviceptr dst, CUdeviceptr src, size_t n) {
  bool dd = is_device_ptr(dst), sd = is_device_ptr(src);
  if (dd && sd) return cuMemcpyDtoD_v2_impl(dst, src, n);
  if (dd) return cuMemcpyHtoD_v2_impl(dst, reinterpret_cast<const void*>(src), n);
  if (sd) return cuMemcpyDtoH_v2_impl(reinterpret_cast<void*>(dst), src, n);
  std::memcpy(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), n);
  return CUDA_SUCCESS;
}
static CUresult cuMemcpyAsync_impl(CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream);
VGPU_EXPORT CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream a3) { return traced("cuMemcpyAsync", cuMemcpyAsync_impl, dst, src, n, a3); }
static CUresult cuMemcpyAsync_impl(CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream) {
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

// What a profiler calls each end of a copy or fill: managed and pinned (or
// registered) host memory by the allocation that holds the pointer, device
// memory by where the pointer points or what the call says it is, and anything
// else pageable.
vgpu::profiling::MemKind profiled_kind(ShimState& s, const void* p, bool device) {
  using vgpu::profiling::MemKind;
  if (managed_range(s, reinterpret_cast<CUdeviceptr>(p), 1)) return MemKind::Managed;
  if (host_range_at(pinned(s), p) || host_range_at(registrations(s), p)) return MemKind::Pinned;
  if (device || vgpu::is_device_va(reinterpret_cast<uint64_t>(p))) return MemKind::Device;
  return MemKind::Pageable;
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
                     size_t rows, CUstream stream = nullptr, bool async = false) {
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
    const uint64_t t0 = vgpu::profiling::enabled() ? vgpu::profiling::now_ns() : 0;
    for (size_t y = 0; y < rows; ++y)
      dev_fill(s, dptr + y * pitch, reinterpret_cast<const uint8_t*>(&value), sizeof(T), width * sizeof(T));
    record_memset(s, dptr, static_cast<uint32_t>(value), width * sizeof(T) * rows, t0, stream, async);
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
static CUresult cuLaunchCooperativeKernel_impl(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams);
VGPU_EXPORT CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams) { return traced("cuLaunchCooperativeKernel", cuLaunchCooperativeKernel_impl, f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams); }
static CUresult cuLaunchCooperativeKernel_impl(CUfunction f, unsigned int gridDimX,
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
static CUresult cuLinkCreate_v2_impl(unsigned int, void*, void*, void** stateOut);
VGPU_EXPORT CUresult cuLinkCreate_v2(unsigned int a0, void* a1, void* a2, void** stateOut) { return traced("cuLinkCreate_v2", cuLinkCreate_v2_impl, a0, a1, a2, stateOut); }
static CUresult cuLinkCreate_v2_impl(unsigned int, void*, void*, void** stateOut) {
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
static CUresult cuLinkCreate_impl(unsigned int n, void* keys, void* vals, void** stateOut);
VGPU_EXPORT CUresult cuLinkCreate(unsigned int n, void* keys, void* vals, void** stateOut) { return traced("cuLinkCreate", cuLinkCreate_impl, n, keys, vals, stateOut); }
static CUresult cuLinkCreate_impl(unsigned int n, void* keys, void* vals, void** stateOut) {
  return cuLinkCreate_v2(n, keys, vals, stateOut);
}

static CUresult cuLinkAddData_v2_impl(void* state, int type, void* data, size_t size, const char* name, unsigned int, void*, void*);
VGPU_EXPORT CUresult cuLinkAddData_v2(void* state, int type, void* data, size_t size, const char* name, unsigned int a5, void* a6, void* a7) { return traced("cuLinkAddData_v2", cuLinkAddData_v2_impl, state, type, data, size, name, a5, a6, a7); }
static CUresult cuLinkAddData_v2_impl(void* state, int type, void* data, size_t size,
                                      const char* name, unsigned int, void*, void*) {
  return api("cuLinkAddData_v2", true, false, [&](ShimState&) {
    if (!state) return CUDA_ERROR_INVALID_VALUE;
    link_add(link_state(state), type, data, size, name);
    return CUDA_SUCCESS;
  });
}
static CUresult cuLinkAddData_impl(void* state, int type, void* data, size_t size, const char* name, unsigned int n, void* keys, void* vals);
VGPU_EXPORT CUresult cuLinkAddData(void* state, int type, void* data, size_t size, const char* name, unsigned int n, void* keys, void* vals) { return traced("cuLinkAddData", cuLinkAddData_impl, state, type, data, size, name, n, keys, vals); }
static CUresult cuLinkAddData_impl(void* state, int type, void* data, size_t size, const char* name,
                                   unsigned int n, void* keys, void* vals) {
  return cuLinkAddData_v2(state, type, data, size, name, n, keys, vals);
}

static CUresult cuLinkComplete_impl(void* state, void** imageOut, size_t* sizeOut);
VGPU_EXPORT CUresult cuLinkComplete(void* state, void** imageOut, size_t* sizeOut) { return traced("cuLinkComplete", cuLinkComplete_impl, state, imageOut, sizeOut); }
static CUresult cuLinkComplete_impl(void* state, void** imageOut, size_t* sizeOut) {
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
static CUresult cuTensorMapEncodeTiled_impl(void* tensorMap, unsigned int dataType, unsigned int rank, void* globalAddress, const unsigned long long* globalDim, const unsigned long long* globalStrides, const unsigned int* boxDim, const unsigned int* elementStrides, unsigned int interleave, unsigned int swizzle, unsigned int l2Promotion, unsigned int oobFill);
VGPU_EXPORT CUresult cuTensorMapEncodeTiled(void* tensorMap, unsigned int dataType, unsigned int rank, void* globalAddress, const unsigned long long* globalDim, const unsigned long long* globalStrides, const unsigned int* boxDim, const unsigned int* elementStrides, unsigned int interleave, unsigned int swizzle, unsigned int l2Promotion, unsigned int oobFill) { return traced("cuTensorMapEncodeTiled", cuTensorMapEncodeTiled_impl, tensorMap, dataType, rank, globalAddress, globalDim, globalStrides, boxDim, elementStrides, interleave, swizzle, l2Promotion, oobFill); }
static CUresult cuTensorMapEncodeTiled_impl(void* tensorMap, unsigned int dataType, unsigned int rank,
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
static CUresult cuTensorMapReplaceAddress_impl(void* tensorMap, void* globalAddress);
VGPU_EXPORT CUresult cuTensorMapReplaceAddress(void* tensorMap, void* globalAddress) { return traced("cuTensorMapReplaceAddress", cuTensorMapReplaceAddress_impl, tensorMap, globalAddress); }
static CUresult cuTensorMapReplaceAddress_impl(void* tensorMap, void* globalAddress) {
  return api("cuTensorMapReplaceAddress", false, false, [&](ShimState&) -> CUresult {
    std::string why;
    if (vgpu::exec::replace_address(tensorMap, globalAddress, &why) != vgpu::exec::TmapResult::Ok)
      throw vgpu::Error::make(vgpu::Err::InvalidValue, "cuTensorMapReplaceAddress: ", why);
    return CUDA_SUCCESS;
  });
}

static CUresult cuTensorMapEncodeIm2col_impl(void* tensorMap, unsigned int dataType, unsigned int rank, void* globalAddress, const unsigned long long* globalDim, const unsigned long long* globalStrides, const int* lowerCorner, const int* upperCorner, unsigned int channelsPerPixel, unsigned int pixelsPerColumn, const unsigned int* elementStrides, unsigned int interleave, unsigned int swizzle, unsigned int l2Promotion, unsigned int oobFill);
VGPU_EXPORT CUresult cuTensorMapEncodeIm2col(void* tensorMap, unsigned int dataType, unsigned int rank, void* globalAddress, const unsigned long long* globalDim, const unsigned long long* globalStrides, const int* lowerCorner, const int* upperCorner, unsigned int channelsPerPixel, unsigned int pixelsPerColumn, const unsigned int* elementStrides, unsigned int interleave, unsigned int swizzle, unsigned int l2Promotion, unsigned int oobFill) { return traced("cuTensorMapEncodeIm2col", cuTensorMapEncodeIm2col_impl, tensorMap, dataType, rank, globalAddress, globalDim, globalStrides, lowerCorner, upperCorner, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, swizzle, l2Promotion, oobFill); }
static CUresult cuTensorMapEncodeIm2col_impl(void* tensorMap, unsigned int dataType, unsigned int rank,
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

static CUresult cuMemsetD8_v2_impl(CUdeviceptr d, unsigned char v, size_t n) {
  return memset_impl("cuMemsetD8", d, 0, v, n, 1);
}
static CUresult cuMemsetD16_v2_impl(CUdeviceptr d, unsigned short v, size_t n) {
  return memset_impl("cuMemsetD16", d, 0, v, n, 1);
}
static CUresult cuMemsetD32_v2_impl(CUdeviceptr d, unsigned int v, size_t n) {
  return memset_impl("cuMemsetD32", d, 0, v, n, 1);
}
static CUresult cuMemsetD8Async_impl(CUdeviceptr d, unsigned char v, size_t n, CUstream st) {
  return memset_impl("cuMemsetD8", d, 0, v, n, 1, st, true);
}
static CUresult cuMemsetD16Async_impl(CUdeviceptr d, unsigned short v, size_t n, CUstream st) {
  return memset_impl("cuMemsetD16", d, 0, v, n, 1, st, true);
}
static CUresult cuMemsetD32Async_impl(CUdeviceptr d, unsigned int v, size_t n, CUstream st) {
  return memset_impl("cuMemsetD32", d, 0, v, n, 1, st, true);
}
VGPU_EXPORT CUresult cuMemsetD8_v2(CUdeviceptr d, unsigned char v, size_t n) {
  return traced("cuMemsetD8_v2", cuMemsetD8_v2_impl, d, v, n);
}
VGPU_EXPORT CUresult cuMemsetD16_v2(CUdeviceptr d, unsigned short v, size_t n) {
  return traced("cuMemsetD16_v2", cuMemsetD16_v2_impl, d, v, n);
}
VGPU_EXPORT CUresult cuMemsetD32_v2(CUdeviceptr d, unsigned int v, size_t n) {
  return traced("cuMemsetD32_v2", cuMemsetD32_v2_impl, d, v, n);
}
static CUresult cuMemsetD8_impl(CUdeviceptr d, unsigned char v, size_t n);
VGPU_EXPORT CUresult cuMemsetD8(CUdeviceptr d, unsigned char v, size_t n) { return traced("cuMemsetD8", cuMemsetD8_impl, d, v, n); }
static CUresult cuMemsetD8_impl(CUdeviceptr d, unsigned char v, size_t n) { return cuMemsetD8_v2_impl(d, v, n); }
static CUresult cuMemsetD16_impl(CUdeviceptr d, unsigned short v, size_t n);
VGPU_EXPORT CUresult cuMemsetD16(CUdeviceptr d, unsigned short v, size_t n) { return traced("cuMemsetD16", cuMemsetD16_impl, d, v, n); }
static CUresult cuMemsetD16_impl(CUdeviceptr d, unsigned short v, size_t n) { return cuMemsetD16_v2_impl(d, v, n); }
static CUresult cuMemsetD32_impl(CUdeviceptr d, unsigned int v, size_t n);
VGPU_EXPORT CUresult cuMemsetD32(CUdeviceptr d, unsigned int v, size_t n) { return traced("cuMemsetD32", cuMemsetD32_impl, d, v, n); }
static CUresult cuMemsetD32_impl(CUdeviceptr d, unsigned int v, size_t n) { return cuMemsetD32_v2_impl(d, v, n); }
VGPU_EXPORT CUresult cuMemsetD8Async(CUdeviceptr d, unsigned char v, size_t n, CUstream st) {
  return traced("cuMemsetD8Async", cuMemsetD8Async_impl, d, v, n, st);
}
VGPU_EXPORT CUresult cuMemsetD16Async(CUdeviceptr d, unsigned short v, size_t n, CUstream st) {
  return traced("cuMemsetD16Async", cuMemsetD16Async_impl, d, v, n, st);
}
VGPU_EXPORT CUresult cuMemsetD32Async(CUdeviceptr d, unsigned int v, size_t n, CUstream st) {
  return traced("cuMemsetD32Async", cuMemsetD32Async_impl, d, v, n, st);
}
// Width counts elements, the pitch bytes.
static CUresult cuMemsetD2D8_v2_impl(CUdeviceptr d, size_t pitch, unsigned char v, size_t w, size_t h);
VGPU_EXPORT CUresult cuMemsetD2D8_v2(CUdeviceptr d, size_t pitch, unsigned char v, size_t w, size_t h) { return traced("cuMemsetD2D8_v2", cuMemsetD2D8_v2_impl, d, pitch, v, w, h); }
static CUresult cuMemsetD2D8_v2_impl(CUdeviceptr d, size_t pitch, unsigned char v, size_t w, size_t h) {
  return memset_impl("cuMemsetD2D8", d, pitch, v, w, h);
}
static CUresult cuMemsetD2D16_v2_impl(CUdeviceptr d, size_t pitch, unsigned short v, size_t w, size_t h);
VGPU_EXPORT CUresult cuMemsetD2D16_v2(CUdeviceptr d, size_t pitch, unsigned short v, size_t w, size_t h) { return traced("cuMemsetD2D16_v2", cuMemsetD2D16_v2_impl, d, pitch, v, w, h); }
static CUresult cuMemsetD2D16_v2_impl(CUdeviceptr d, size_t pitch, unsigned short v, size_t w, size_t h) {
  return memset_impl("cuMemsetD2D16", d, pitch, v, w, h);
}
static CUresult cuMemsetD2D32_v2_impl(CUdeviceptr d, size_t pitch, unsigned int v, size_t w, size_t h);
VGPU_EXPORT CUresult cuMemsetD2D32_v2(CUdeviceptr d, size_t pitch, unsigned int v, size_t w, size_t h) { return traced("cuMemsetD2D32_v2", cuMemsetD2D32_v2_impl, d, pitch, v, w, h); }
static CUresult cuMemsetD2D32_v2_impl(CUdeviceptr d, size_t pitch, unsigned int v, size_t w, size_t h) {
  return memset_impl("cuMemsetD2D32", d, pitch, v, w, h);
}
static CUresult cuMemsetD2D8_impl(CUdeviceptr d, size_t pitch, unsigned char v, size_t w, size_t h);
VGPU_EXPORT CUresult cuMemsetD2D8(CUdeviceptr d, size_t pitch, unsigned char v, size_t w, size_t h) { return traced("cuMemsetD2D8", cuMemsetD2D8_impl, d, pitch, v, w, h); }
static CUresult cuMemsetD2D8_impl(CUdeviceptr d, size_t pitch, unsigned char v, size_t w, size_t h) {
  return cuMemsetD2D8_v2(d, pitch, v, w, h);
}
static CUresult cuMemsetD2D16_impl(CUdeviceptr d, size_t pitch, unsigned short v, size_t w, size_t h);
VGPU_EXPORT CUresult cuMemsetD2D16(CUdeviceptr d, size_t pitch, unsigned short v, size_t w, size_t h) { return traced("cuMemsetD2D16", cuMemsetD2D16_impl, d, pitch, v, w, h); }
static CUresult cuMemsetD2D16_impl(CUdeviceptr d, size_t pitch, unsigned short v, size_t w, size_t h) {
  return cuMemsetD2D16_v2(d, pitch, v, w, h);
}
static CUresult cuMemsetD2D32_impl(CUdeviceptr d, size_t pitch, unsigned int v, size_t w, size_t h);
VGPU_EXPORT CUresult cuMemsetD2D32(CUdeviceptr d, size_t pitch, unsigned int v, size_t w, size_t h) { return traced("cuMemsetD2D32", cuMemsetD2D32_impl, d, pitch, v, w, h); }
static CUresult cuMemsetD2D32_impl(CUdeviceptr d, size_t pitch, unsigned int v, size_t w, size_t h) {
  return cuMemsetD2D32_v2(d, pitch, v, w, h);
}
static CUresult cuMemsetD2D8Async_impl(CUdeviceptr d, size_t pitch, unsigned char v, size_t w, size_t h, CUstream);
VGPU_EXPORT CUresult cuMemsetD2D8Async(CUdeviceptr d, size_t pitch, unsigned char v, size_t w, size_t h, CUstream a5) { return traced("cuMemsetD2D8Async", cuMemsetD2D8Async_impl, d, pitch, v, w, h, a5); }
static CUresult cuMemsetD2D8Async_impl(CUdeviceptr d, size_t pitch, unsigned char v, size_t w, size_t h,
                                       CUstream) {
  return memset_impl("cuMemsetD2D8", d, pitch, v, w, h);
}
static CUresult cuMemsetD2D16Async_impl(CUdeviceptr d, size_t pitch, unsigned short v, size_t w, size_t h, CUstream);
VGPU_EXPORT CUresult cuMemsetD2D16Async(CUdeviceptr d, size_t pitch, unsigned short v, size_t w, size_t h, CUstream a5) { return traced("cuMemsetD2D16Async", cuMemsetD2D16Async_impl, d, pitch, v, w, h, a5); }
static CUresult cuMemsetD2D16Async_impl(CUdeviceptr d, size_t pitch, unsigned short v, size_t w, size_t h,
                                        CUstream) {
  return memset_impl("cuMemsetD2D16", d, pitch, v, w, h);
}
static CUresult cuMemsetD2D32Async_impl(CUdeviceptr d, size_t pitch, unsigned int v, size_t w, size_t h, CUstream);
VGPU_EXPORT CUresult cuMemsetD2D32Async(CUdeviceptr d, size_t pitch, unsigned int v, size_t w, size_t h, CUstream a5) { return traced("cuMemsetD2D32Async", cuMemsetD2D32Async_impl, d, pitch, v, w, h, a5); }
static CUresult cuMemsetD2D32Async_impl(CUdeviceptr d, size_t pitch, unsigned int v, size_t w, size_t h,
                                        CUstream) {
  return memset_impl("cuMemsetD2D32", d, pitch, v, w, h);
}

static CUresult cuMemGetAddressRange_v2_impl(CUdeviceptr* base, size_t* size, CUdeviceptr dptr);
VGPU_EXPORT CUresult cuMemGetAddressRange_v2(CUdeviceptr* base, size_t* size, CUdeviceptr dptr) { return traced("cuMemGetAddressRange_v2", cuMemGetAddressRange_v2_impl, base, size, dptr); }
static CUresult cuMemGetAddressRange_v2_impl(CUdeviceptr* base, size_t* size, CUdeviceptr dptr) {
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

static CUresult cuMemHostAlloc_impl(void** pp, size_t bytesize, unsigned int flags) {
  return api("cuMemHostAlloc", true, false, [&](ShimState& s) {
    if (!pp || bytesize == 0) return CUDA_ERROR_INVALID_VALUE;
    void* p = std::aligned_alloc(4096, (bytesize + 4095) / 4096 * 4096);
    if (!p) return CUDA_ERROR_OUT_OF_MEMORY;
    for (int d = 0; d < s.rt->device_count(); ++d)
      s.rt->device(d).memory().map_host(reinterpret_cast<uint64_t>(p), p, bytesize);
    pinned(s)[p] = vgpu::runtime::HostRange{bytesize, 0, flags};
    *pp = p;
    profile_memory(1, reinterpret_cast<uint64_t>(p), bytesize, vgpu::profiling::MemKind::Pinned, current_device(s));
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int flags) {
  return traced("cuMemHostAlloc", cuMemHostAlloc_impl, pp, bytesize, flags);
}
static CUresult cuMemAllocHost_v2_impl(void** pp, size_t bytesize) {
  return cuMemHostAlloc_impl(pp, bytesize, 0);
}
VGPU_EXPORT CUresult cuMemAllocHost_v2(void** pp, size_t bytesize) {
  return traced("cuMemAllocHost_v2", cuMemAllocHost_v2_impl, pp, bytesize);
}
static CUresult cuMemAllocHost_impl(void** pp, size_t bytesize);
VGPU_EXPORT CUresult cuMemAllocHost(void** pp, size_t bytesize) { return traced("cuMemAllocHost", cuMemAllocHost_impl, pp, bytesize); }
static CUresult cuMemAllocHost_impl(void** pp, size_t bytesize) { return cuMemHostAlloc_impl(pp, bytesize, 0); }
static CUresult cuMemFreeHost_impl(void* p) {
  return api("cuMemFreeHost", true, false, [&](ShimState& s) {
    if (!p) return CUDA_SUCCESS;   // as cuMemFree(0): a no-op, with or without a context
    auto it = pinned(s).find(p);
    if (it == pinned(s).end()) return CUDA_ERROR_INVALID_VALUE;
    profile_memory(2, reinterpret_cast<uint64_t>(p), it->second.size, vgpu::profiling::MemKind::Pinned, current_device(s));
    for (int d = 0; d < s.rt->device_count(); ++d) s.rt->device(d).memory().unmap_host(reinterpret_cast<uint64_t>(p));
    pinned(s).erase(it);
    std::free(p);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuMemFreeHost(void* p) {
  return traced("cuMemFreeHost", cuMemFreeHost_impl, p);
}

/* ---- registered host memory ----
 * cuMemHostRegister and cudaHostRegister keep one registry, in the machine
 * both libraries share: on the card, memory one API registered is already
 * registered to the other (CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED), and
 * either API unregisters it. As cudaHostRegister does, the range is mapped
 * into every device at its host address, and that is the device pointer
 * cuMemHostGetDevicePointer reports. The answers below are an RTX 3060's. */

static CUresult cuMemHostRegister_v2_impl(void* p, size_t bytesize, unsigned int flags);
VGPU_EXPORT CUresult cuMemHostRegister_v2(void* p, size_t bytesize, unsigned int flags) { return traced("cuMemHostRegister_v2", cuMemHostRegister_v2_impl, p, bytesize, flags); }
static CUresult cuMemHostRegister_v2_impl(void* p, size_t bytesize, unsigned int flags) {
  return api("cuMemHostRegister", true, false, [&](ShimState& s) {
    // PORTABLE, DEVICEMAP, IOMEMORY and READ_ONLY; nothing else. Read-only registration works
    // (READ_ONLY_HOST_REGISTER_SUPPORTED, 113, is 1): the device reads the range and a kernel's
    // store or atomic to it faults, as on an RTX 3060.
    if (!p || bytesize == 0 || (flags & ~0xFu)) return CUDA_ERROR_INVALID_VALUE;
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
      s.rt->device(d).memory().map_host(reinterpret_cast<uint64_t>(p), p, bytesize,
                                        (flags & CU_MEMHOSTREGISTER_READ_ONLY) != 0);
    regs[p] = vgpu::runtime::HostRange{bytesize, dev, flags};
    return CUDA_SUCCESS;
  });
}
static CUresult cuMemHostRegister_impl(void* p, size_t bytesize, unsigned int flags);
VGPU_EXPORT CUresult cuMemHostRegister(void* p, size_t bytesize, unsigned int flags) { return traced("cuMemHostRegister", cuMemHostRegister_impl, p, bytesize, flags); }
static CUresult cuMemHostRegister_impl(void* p, size_t bytesize, unsigned int flags) {
  return cuMemHostRegister_v2(p, bytesize, flags);
}

// Only by the pointer that was registered: one inside a registration is
// INVALID_VALUE, and so is memory CUDA allocated (pinned or managed); one
// never registered is HOST_MEMORY_NOT_REGISTERED.
static CUresult cuMemHostUnregister_impl(void* p);
VGPU_EXPORT CUresult cuMemHostUnregister(void* p) { return traced("cuMemHostUnregister", cuMemHostUnregister_impl, p); }
static CUresult cuMemHostUnregister_impl(void* p) {
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
static CUresult cuMemHostGetFlags_impl(unsigned int* flags, void* p);
VGPU_EXPORT CUresult cuMemHostGetFlags(unsigned int* flags, void* p) { return traced("cuMemHostGetFlags", cuMemHostGetFlags_impl, flags, p); }
static CUresult cuMemHostGetFlags_impl(unsigned int* flags, void* p) {
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
static CUresult cuMemHostGetDevicePointer_v2_impl(CUdeviceptr* dptr, void* p, unsigned int flags);
VGPU_EXPORT CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr* dptr, void* p, unsigned int flags) { return traced("cuMemHostGetDevicePointer_v2", cuMemHostGetDevicePointer_v2_impl, dptr, p, flags); }
static CUresult cuMemHostGetDevicePointer_v2_impl(CUdeviceptr* dptr, void* p, unsigned int flags) {
  return api("cuMemHostGetDevicePointer", true, false, [&](ShimState& s) {
    if (!dptr || !p || flags != 0) return CUDA_ERROR_INVALID_VALUE;
    if (!host_range_at(registrations(s), p) && !host_range_at(pinned(s), p))
      return CUDA_ERROR_INVALID_VALUE;
    *dptr = reinterpret_cast<CUdeviceptr>(p);
    return CUDA_SUCCESS;
  });
}
static CUresult cuMemHostGetDevicePointer_impl(CUdeviceptr* dptr, void* p, unsigned int flags);
VGPU_EXPORT CUresult cuMemHostGetDevicePointer(CUdeviceptr* dptr, void* p, unsigned int flags) { return traced("cuMemHostGetDevicePointer", cuMemHostGetDevicePointer_impl, dptr, p, flags); }
static CUresult cuMemHostGetDevicePointer_impl(CUdeviceptr* dptr, void* p, unsigned int flags) {
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

static CUresult cuMemAllocManaged_impl(CUdeviceptr* dptr, size_t bytesize, unsigned int flags);
VGPU_EXPORT CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) { return traced("cuMemAllocManaged", cuMemAllocManaged_impl, dptr, bytesize, flags); }
static CUresult cuMemAllocManaged_impl(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) {
  return api("cuMemAllocManaged", true, false, [&](ShimState& s) {
    // CU_MEM_ATTACH_GLOBAL (1) or CU_MEM_ATTACH_HOST (2), exactly; a size of 0 is refused.
    if (!dptr || bytesize == 0 || (flags != 1 && flags != 2)) return CUDA_ERROR_INVALID_VALUE;
    const size_t n = (bytesize + 4095) / 4096 * 4096;
    void* p = std::aligned_alloc(4096, n);
    if (!p) return CUDA_ERROR_OUT_OF_MEMORY;
    for (int d = 0; d < s.rt->device_count(); ++d)
      s.rt->device(d).memory().map_host(reinterpret_cast<uint64_t>(p), p, bytesize);
    s.managed[reinterpret_cast<uintptr_t>(p)] = bytesize;
    s.managed_device[reinterpret_cast<uintptr_t>(p)] = current_device(s);
    *dptr = reinterpret_cast<CUdeviceptr>(p);
    return CUDA_SUCCESS;
  });
}

// Which stream may touch an allocation. The whole allocation, from its base
// (length 0 or its full size), with GLOBAL (1), HOST (2) or SINGLE (4).
static CUresult cuStreamAttachMemAsync_impl(CUstream, CUdeviceptr dptr, size_t length, unsigned int flags);
VGPU_EXPORT CUresult cuStreamAttachMemAsync(CUstream a0, CUdeviceptr dptr, size_t length, unsigned int flags) { return traced("cuStreamAttachMemAsync", cuStreamAttachMemAsync_impl, a0, dptr, length, flags); }
static CUresult cuStreamAttachMemAsync_impl(CUstream, CUdeviceptr dptr, size_t length, unsigned int flags) {
  return api("cuStreamAttachMemAsync", true, false, [&](ShimState& s) {
    auto it = s.managed.find(dptr);
    if (it == s.managed.end() || (length != 0 && length != it->second) ||
        (flags != 1 && flags != 2 && flags != 4))
      return CUDA_ERROR_INVALID_VALUE;
    return CUDA_SUCCESS;
  });
}

static CUresult cuMemPrefetchAsync_v2_impl(CUdeviceptr dptr, size_t count, CUmemLocation location, unsigned int, CUstream);
VGPU_EXPORT CUresult cuMemPrefetchAsync_v2(CUdeviceptr dptr, size_t count, CUmemLocation location, unsigned int a3, CUstream a4) { return traced("cuMemPrefetchAsync_v2", cuMemPrefetchAsync_v2_impl, dptr, count, location, a3, a4); }
static CUresult cuMemPrefetchAsync_v2_impl(CUdeviceptr dptr, size_t count, CUmemLocation location,
                                           unsigned int, CUstream) {
  return api("cuMemPrefetchAsync", true, false, [&](ShimState& s) {
    if (!managed_range(s, dptr, count)) return CUDA_ERROR_INVALID_VALUE;
    return check_location(s, location);
  });
}
static CUresult cuMemPrefetchAsync_impl(CUdeviceptr dptr, size_t count, CUdevice dstDevice, CUstream stream);
VGPU_EXPORT CUresult cuMemPrefetchAsync(CUdeviceptr dptr, size_t count, CUdevice dstDevice, CUstream stream) { return traced("cuMemPrefetchAsync", cuMemPrefetchAsync_impl, dptr, count, dstDevice, stream); }
static CUresult cuMemPrefetchAsync_impl(CUdeviceptr dptr, size_t count, CUdevice dstDevice, CUstream stream) {
  return cuMemPrefetchAsync_v2(dptr, count, location_of(dstDevice), 0, stream);
}
static CUresult cuMemAdvise_v2_impl(CUdeviceptr dptr, size_t count, int advice, CUmemLocation location);
VGPU_EXPORT CUresult cuMemAdvise_v2(CUdeviceptr dptr, size_t count, int advice, CUmemLocation location) { return traced("cuMemAdvise_v2", cuMemAdvise_v2_impl, dptr, count, advice, location); }
static CUresult cuMemAdvise_v2_impl(CUdeviceptr dptr, size_t count, int advice, CUmemLocation location) {
  return api("cuMemAdvise", true, false, [&](ShimState& s) {
    if (!managed_range(s, dptr, count) || advice < 1 || advice > 6) return CUDA_ERROR_INVALID_VALUE;
    // Read-mostly (1, 2) names no location; the others name one.
    if (advice > 2) return check_location(s, location);
    return CUDA_SUCCESS;
  });
}
static CUresult cuMemAdvise_impl(CUdeviceptr dptr, size_t count, int advice, CUdevice device);
VGPU_EXPORT CUresult cuMemAdvise(CUdeviceptr dptr, size_t count, int advice, CUdevice device) { return traced("cuMemAdvise", cuMemAdvise_impl, dptr, count, advice, device); }
static CUresult cuMemAdvise_impl(CUdeviceptr dptr, size_t count, int advice, CUdevice device) {
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

// The descriptor's format as the texture unit sees it, with the channel count the format has, or false
// for one the card refuses (CUDA_ERROR_INVALID_VALUE): see texture_format_from_driver.
bool array_format(const CUDA_ARRAY3D_DESCRIPTOR& d, vgpu::cuda::TexFormat* tf) {
  unsigned raw;
  static_assert(sizeof raw == sizeof d.Format);
  std::memcpy(&raw, &d.Format, sizeof raw);
  return vgpu::cuda::texture_format_from_driver(raw, d.NumChannels, tf);
}

// Block-compressed data cannot be read or written as a surface: asking for the surface flag on such an array
// is CUDA_ERROR_NOT_SUPPORTED (measured on an RTX 3060, for plain and mipmapped arrays).
bool surface_on_blocks(const CUDA_ARRAY3D_DESCRIPTOR& d) {
  vgpu::cuda::TexFormat tf;
  return array_format(d, &tf) && tf.block != vgpu::exec::BlockFormat::None && (d.Flags & CUDA_ARRAY3D_SURFACE_LDST);
}

bool valid_array(const CUDA_ARRAY3D_DESCRIPTOR& d) {
  const unsigned known = CUDA_ARRAY3D_LAYERED | CUDA_ARRAY3D_SURFACE_LDST | CUDA_ARRAY3D_CUBEMAP |
                         CUDA_ARRAY3D_TEXTURE_GATHER;
  if (d.Flags & ~known) return false;
  vgpu::cuda::TexFormat tf;
  if (!array_format(d, &tf)) return false;
  if (d.Width == 0) return false;
  // A block-compressed array has at least two dimensions (measured: a zero height is invalid).
  if (tf.block != vgpu::exec::BlockFormat::None && d.Height == 0) return false;
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
  array_format(desc, &r.tf);
  r.elem = r.tf.texel_bytes;
  const bool blocky = r.tf.block != vgpu::exec::BlockFormat::None;
  // A block-compressed array holds 4 x 4 blocks: its rows are rows of blocks (which is what a copy addresses).
  r.row = (blocky ? (desc.Width + 3) / 4 : desc.Width) * r.elem;
  r.rows = blocky ? (desc.Height + 3) / 4 : (desc.Height ? desc.Height : 1);
  r.slices = desc.Depth ? desc.Depth : 1;
  r.device = current_device(s);
  r.mem = current(s).memory().alloc(r.row * r.rows * r.slices);
  const uintptr_t h = make_handle(s, kTagArray);
  s.arrays[h] = r;
  return h;
}
}  // namespace

static CUresult cuArray3DCreate_v2_impl(CUarray* out, const CUDA_ARRAY3D_DESCRIPTOR* desc);
VGPU_EXPORT CUresult cuArray3DCreate_v2(CUarray* out, const CUDA_ARRAY3D_DESCRIPTOR* desc) { return traced("cuArray3DCreate_v2", cuArray3DCreate_v2_impl, out, desc); }
static CUresult cuArray3DCreate_v2_impl(CUarray* out, const CUDA_ARRAY3D_DESCRIPTOR* desc) {
  return api("cuArray3DCreate", true, false, [&](ShimState& s) {
    if (!out || !desc || !valid_array(*desc)) return CUDA_ERROR_INVALID_VALUE;
    if (surface_on_blocks(*desc)) return CUDA_ERROR_NOT_SUPPORTED;
    *out = reinterpret_cast<CUarray>(create_array(s, *desc));
    return CUDA_SUCCESS;
  });
}
static CUresult cuArray3DCreate_impl(CUarray* out, const CUDA_ARRAY3D_DESCRIPTOR* desc);
VGPU_EXPORT CUresult cuArray3DCreate(CUarray* out, const CUDA_ARRAY3D_DESCRIPTOR* desc) { return traced("cuArray3DCreate", cuArray3DCreate_impl, out, desc); }
static CUresult cuArray3DCreate_impl(CUarray* out, const CUDA_ARRAY3D_DESCRIPTOR* desc) {
  return cuArray3DCreate_v2(out, desc);
}
static CUresult cuArrayCreate_v2_impl(CUarray* out, const CUDA_ARRAY_DESCRIPTOR* desc);
VGPU_EXPORT CUresult cuArrayCreate_v2(CUarray* out, const CUDA_ARRAY_DESCRIPTOR* desc) { return traced("cuArrayCreate_v2", cuArrayCreate_v2_impl, out, desc); }
static CUresult cuArrayCreate_v2_impl(CUarray* out, const CUDA_ARRAY_DESCRIPTOR* desc) {
  if (!desc) return CUDA_ERROR_INVALID_VALUE;
  CUDA_ARRAY3D_DESCRIPTOR d{};
  d.Width = desc->Width;
  d.Height = desc->Height;
  d.Format = desc->Format;
  d.NumChannels = desc->NumChannels;
  return cuArray3DCreate_v2(out, &d);
}
static CUresult cuArrayCreate_impl(CUarray* out, const CUDA_ARRAY_DESCRIPTOR* desc);
VGPU_EXPORT CUresult cuArrayCreate(CUarray* out, const CUDA_ARRAY_DESCRIPTOR* desc) { return traced("cuArrayCreate", cuArrayCreate_impl, out, desc); }
static CUresult cuArrayCreate_impl(CUarray* out, const CUDA_ARRAY_DESCRIPTOR* desc) {
  return cuArrayCreate_v2(out, desc);
}
static CUresult cuArray3DGetDescriptor_v2_impl(CUDA_ARRAY3D_DESCRIPTOR* desc, CUarray a);
VGPU_EXPORT CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR* desc, CUarray a) { return traced("cuArray3DGetDescriptor_v2", cuArray3DGetDescriptor_v2_impl, desc, a); }
static CUresult cuArray3DGetDescriptor_v2_impl(CUDA_ARRAY3D_DESCRIPTOR* desc, CUarray a) {
  return api("cuArray3DGetDescriptor", true, false, [&](ShimState& s) {
    if (s.retired.count(reinterpret_cast<uintptr_t>(a))) return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    const ArrayRec* r = array_rec(s, a);
    if (!desc || !r) return CUDA_ERROR_INVALID_VALUE;
    *desc = r->desc;
    return CUDA_SUCCESS;
  });
}
static CUresult cuArray3DGetDescriptor_impl(CUDA_ARRAY3D_DESCRIPTOR* desc, CUarray a);
VGPU_EXPORT CUresult cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR* desc, CUarray a) { return traced("cuArray3DGetDescriptor", cuArray3DGetDescriptor_impl, desc, a); }
static CUresult cuArray3DGetDescriptor_impl(CUDA_ARRAY3D_DESCRIPTOR* desc, CUarray a) {
  return cuArray3DGetDescriptor_v2(desc, a);
}
// The 2D descriptor of any array, a 3D one included (the card answers for both).
static CUresult cuArrayGetDescriptor_v2_impl(CUDA_ARRAY_DESCRIPTOR* desc, CUarray a);
VGPU_EXPORT CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR* desc, CUarray a) { return traced("cuArrayGetDescriptor_v2", cuArrayGetDescriptor_v2_impl, desc, a); }
static CUresult cuArrayGetDescriptor_v2_impl(CUDA_ARRAY_DESCRIPTOR* desc, CUarray a) {
  CUDA_ARRAY3D_DESCRIPTOR d{};
  if (!desc) return CUDA_ERROR_INVALID_VALUE;
  if (CUresult r = cuArray3DGetDescriptor_v2(&d, a)) return r;
  desc->Width = d.Width;
  desc->Height = d.Height;
  desc->Format = d.Format;
  desc->NumChannels = d.NumChannels;
  return CUDA_SUCCESS;
}
static CUresult cuArrayGetDescriptor_impl(CUDA_ARRAY_DESCRIPTOR* desc, CUarray a);
VGPU_EXPORT CUresult cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR* desc, CUarray a) { return traced("cuArrayGetDescriptor", cuArrayGetDescriptor_impl, desc, a); }
static CUresult cuArrayGetDescriptor_impl(CUDA_ARRAY_DESCRIPTOR* desc, CUarray a) {
  return cuArrayGetDescriptor_v2(desc, a);
}
static CUresult cuArrayDestroy_impl(CUarray a);
VGPU_EXPORT CUresult cuArrayDestroy(CUarray a) { return traced("cuArrayDestroy", cuArrayDestroy_impl, a); }
static CUresult cuArrayDestroy_impl(CUarray a) {
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

static CUresult cuMemcpyHtoA_v2_impl(CUarray dst, size_t off, const void* src, size_t n);
VGPU_EXPORT CUresult cuMemcpyHtoA_v2(CUarray dst, size_t off, const void* src, size_t n) { return traced("cuMemcpyHtoA_v2", cuMemcpyHtoA_v2_impl, dst, off, src, n); }
static CUresult cuMemcpyHtoA_v2_impl(CUarray dst, size_t off, const void* src, size_t n) {
  return api("cuMemcpyHtoA", true, false, [&](ShimState& s) {
    CUdeviceptr at = 0;
    if (!src && n) return CUDA_ERROR_INVALID_VALUE;
    if (CUresult r = linear_part(s, dst, off, n, &at)) return r;
    dev_write(s, at, src, n);
    return CUDA_SUCCESS;
  });
}
static CUresult cuMemcpyHtoA_impl(CUarray d, size_t off, const void* src, size_t n);
VGPU_EXPORT CUresult cuMemcpyHtoA(CUarray d, size_t off, const void* src, size_t n) { return traced("cuMemcpyHtoA", cuMemcpyHtoA_impl, d, off, src, n); }
static CUresult cuMemcpyHtoA_impl(CUarray d, size_t off, const void* src, size_t n) {
  return cuMemcpyHtoA_v2(d, off, src, n);
}
static CUresult cuMemcpyAtoH_v2_impl(void* dst, CUarray src, size_t off, size_t n);
VGPU_EXPORT CUresult cuMemcpyAtoH_v2(void* dst, CUarray src, size_t off, size_t n) { return traced("cuMemcpyAtoH_v2", cuMemcpyAtoH_v2_impl, dst, src, off, n); }
static CUresult cuMemcpyAtoH_v2_impl(void* dst, CUarray src, size_t off, size_t n) {
  return api("cuMemcpyAtoH", true, false, [&](ShimState& s) {
    CUdeviceptr at = 0;
    if (!dst && n) return CUDA_ERROR_INVALID_VALUE;
    if (CUresult r = linear_part(s, src, off, n, &at)) return r;
    dev_read(s, dst, at, n);
    return CUDA_SUCCESS;
  });
}
static CUresult cuMemcpyAtoH_impl(void* dst, CUarray src, size_t off, size_t n);
VGPU_EXPORT CUresult cuMemcpyAtoH(void* dst, CUarray src, size_t off, size_t n) { return traced("cuMemcpyAtoH", cuMemcpyAtoH_impl, dst, src, off, n); }
static CUresult cuMemcpyAtoH_impl(void* dst, CUarray src, size_t off, size_t n) {
  return cuMemcpyAtoH_v2(dst, src, off, n);
}
static CUresult cuMemcpyDtoA_v2_impl(CUarray dst, size_t off, CUdeviceptr src, size_t n);
VGPU_EXPORT CUresult cuMemcpyDtoA_v2(CUarray dst, size_t off, CUdeviceptr src, size_t n) { return traced("cuMemcpyDtoA_v2", cuMemcpyDtoA_v2_impl, dst, off, src, n); }
static CUresult cuMemcpyDtoA_v2_impl(CUarray dst, size_t off, CUdeviceptr src, size_t n) {
  return api("cuMemcpyDtoA", true, false, [&](ShimState& s) {
    CUdeviceptr at = 0;
    if (CUresult r = linear_part(s, dst, off, n, &at)) return r;
    std::vector<uint8_t> tmp(n);
    dev_read(s, tmp.data(), src, n);
    dev_write(s, at, tmp.data(), n);
    return CUDA_SUCCESS;
  });
}
static CUresult cuMemcpyDtoA_impl(CUarray d, size_t off, CUdeviceptr src, size_t n);
VGPU_EXPORT CUresult cuMemcpyDtoA(CUarray d, size_t off, CUdeviceptr src, size_t n) { return traced("cuMemcpyDtoA", cuMemcpyDtoA_impl, d, off, src, n); }
static CUresult cuMemcpyDtoA_impl(CUarray d, size_t off, CUdeviceptr src, size_t n) {
  return cuMemcpyDtoA_v2(d, off, src, n);
}
static CUresult cuMemcpyAtoD_v2_impl(CUdeviceptr dst, CUarray src, size_t off, size_t n);
VGPU_EXPORT CUresult cuMemcpyAtoD_v2(CUdeviceptr dst, CUarray src, size_t off, size_t n) { return traced("cuMemcpyAtoD_v2", cuMemcpyAtoD_v2_impl, dst, src, off, n); }
static CUresult cuMemcpyAtoD_v2_impl(CUdeviceptr dst, CUarray src, size_t off, size_t n) {
  return api("cuMemcpyAtoD", true, false, [&](ShimState& s) {
    CUdeviceptr at = 0;
    if (CUresult r = linear_part(s, src, off, n, &at)) return r;
    std::vector<uint8_t> tmp(n);
    dev_read(s, tmp.data(), at, n);
    dev_write(s, dst, tmp.data(), n);
    return CUDA_SUCCESS;
  });
}
static CUresult cuMemcpyAtoD_impl(CUdeviceptr d, CUarray src, size_t off, size_t n);
VGPU_EXPORT CUresult cuMemcpyAtoD(CUdeviceptr d, CUarray src, size_t off, size_t n) { return traced("cuMemcpyAtoD", cuMemcpyAtoD_impl, d, src, off, n); }
static CUresult cuMemcpyAtoD_impl(CUdeviceptr d, CUarray src, size_t off, size_t n) {
  return cuMemcpyAtoD_v2(d, src, off, n);
}

// Bytes between two arrays' first rows, as cuMemcpyHtoA and the rest take them.
static CUresult cuMemcpyAtoA_v2_impl(CUarray dst, size_t dst_off, CUarray src, size_t src_off, size_t n);
VGPU_EXPORT CUresult cuMemcpyAtoA_v2(CUarray dst, size_t dst_off, CUarray src, size_t src_off, size_t n) { return traced("cuMemcpyAtoA_v2", cuMemcpyAtoA_v2_impl, dst, dst_off, src, src_off, n); }
static CUresult cuMemcpyAtoA_v2_impl(CUarray dst, size_t dst_off, CUarray src, size_t src_off, size_t n) {
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
static CUresult cuMemcpyAtoA_impl(CUarray d, size_t doff, CUarray src, size_t soff, size_t n);
VGPU_EXPORT CUresult cuMemcpyAtoA(CUarray d, size_t doff, CUarray src, size_t soff, size_t n) { return traced("cuMemcpyAtoA", cuMemcpyAtoA_impl, d, doff, src, soff, n); }
static CUresult cuMemcpyAtoA_impl(CUarray d, size_t doff, CUarray src, size_t soff, size_t n) {
  return cuMemcpyAtoA_v2(d, doff, src, soff, n);
}
static CUresult cuMemcpyHtoAAsync_v2_impl(CUarray dst, size_t off, const void* src, size_t n, CUstream);
VGPU_EXPORT CUresult cuMemcpyHtoAAsync_v2(CUarray dst, size_t off, const void* src, size_t n, CUstream a4) { return traced("cuMemcpyHtoAAsync_v2", cuMemcpyHtoAAsync_v2_impl, dst, off, src, n, a4); }
static CUresult cuMemcpyHtoAAsync_v2_impl(CUarray dst, size_t off, const void* src, size_t n, CUstream) {
  return cuMemcpyHtoA_v2(dst, off, src, n);
}
static CUresult cuMemcpyHtoAAsync_impl(CUarray dst, size_t off, const void* src, size_t n, CUstream st);
VGPU_EXPORT CUresult cuMemcpyHtoAAsync(CUarray dst, size_t off, const void* src, size_t n, CUstream st) { return traced("cuMemcpyHtoAAsync", cuMemcpyHtoAAsync_impl, dst, off, src, n, st); }
static CUresult cuMemcpyHtoAAsync_impl(CUarray dst, size_t off, const void* src, size_t n, CUstream st) {
  return cuMemcpyHtoAAsync_v2(dst, off, src, n, st);
}
static CUresult cuMemcpyAtoHAsync_v2_impl(void* dst, CUarray src, size_t off, size_t n, CUstream);
VGPU_EXPORT CUresult cuMemcpyAtoHAsync_v2(void* dst, CUarray src, size_t off, size_t n, CUstream a4) { return traced("cuMemcpyAtoHAsync_v2", cuMemcpyAtoHAsync_v2_impl, dst, src, off, n, a4); }
static CUresult cuMemcpyAtoHAsync_v2_impl(void* dst, CUarray src, size_t off, size_t n, CUstream) {
  return cuMemcpyAtoH_v2(dst, src, off, n);
}
static CUresult cuMemcpyAtoHAsync_impl(void* dst, CUarray src, size_t off, size_t n, CUstream st);
VGPU_EXPORT CUresult cuMemcpyAtoHAsync(void* dst, CUarray src, size_t off, size_t n, CUstream st) { return traced("cuMemcpyAtoHAsync", cuMemcpyAtoHAsync_impl, dst, src, off, n, st); }
static CUresult cuMemcpyAtoHAsync_impl(void* dst, CUarray src, size_t off, size_t n, CUstream st) {
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

static CUresult cuMipmappedArrayCreate_impl(CUmipmappedArray* out, const CUDA_ARRAY3D_DESCRIPTOR* desc, unsigned int numLevels);
VGPU_EXPORT CUresult cuMipmappedArrayCreate(CUmipmappedArray* out, const CUDA_ARRAY3D_DESCRIPTOR* desc, unsigned int numLevels) { return traced("cuMipmappedArrayCreate", cuMipmappedArrayCreate_impl, out, desc, numLevels); }
static CUresult cuMipmappedArrayCreate_impl(CUmipmappedArray* out, const CUDA_ARRAY3D_DESCRIPTOR* desc,
                                            unsigned int numLevels) {
  return api("cuMipmappedArrayCreate", true, false, [&](ShimState& s) {
    if (!out || !desc || !valid_array(*desc)) return CUDA_ERROR_INVALID_VALUE;
    if (surface_on_blocks(*desc)) return CUDA_ERROR_NOT_SUPPORTED;
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

static CUresult cuMipmappedArrayGetLevel_impl(CUarray* level, CUmipmappedArray mm, unsigned int l);
VGPU_EXPORT CUresult cuMipmappedArrayGetLevel(CUarray* level, CUmipmappedArray mm, unsigned int l) { return traced("cuMipmappedArrayGetLevel", cuMipmappedArrayGetLevel_impl, level, mm, l); }
static CUresult cuMipmappedArrayGetLevel_impl(CUarray* level, CUmipmappedArray mm, unsigned int l) {
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

static CUresult cuMipmappedArrayDestroy_impl(CUmipmappedArray mm);
VGPU_EXPORT CUresult cuMipmappedArrayDestroy(CUmipmappedArray mm) { return traced("cuMipmappedArrayDestroy", cuMipmappedArrayDestroy_impl, mm); }
static CUresult cuMipmappedArrayDestroy_impl(CUmipmappedArray mm) {
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

static CUresult cuModuleGetTexRef_impl(CUtexref* out, CUmodule hmod, const char*);
VGPU_EXPORT CUresult cuModuleGetTexRef(CUtexref* out, CUmodule hmod, const char* a2) { return traced("cuModuleGetTexRef", cuModuleGetTexRef_impl, out, hmod, a2); }
static CUresult cuModuleGetTexRef_impl(CUtexref* out, CUmodule hmod, const char*) {
  return api("cuModuleGetTexRef", true, false, [&](ShimState& s) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    if (!s.modules.count(reinterpret_cast<uintptr_t>(hmod))) return CUDA_ERROR_INVALID_HANDLE;
    return CUDA_ERROR_NOT_FOUND;
  });
}
static CUresult cuModuleGetSurfRef_impl(CUsurfref* out, CUmodule hmod, const char*);
VGPU_EXPORT CUresult cuModuleGetSurfRef(CUsurfref* out, CUmodule hmod, const char* a2) { return traced("cuModuleGetSurfRef", cuModuleGetSurfRef_impl, out, hmod, a2); }
static CUresult cuModuleGetSurfRef_impl(CUsurfref* out, CUmodule hmod, const char*) {
  return api("cuModuleGetSurfRef", true, false, [&](ShimState& s) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    if (!s.modules.count(reinterpret_cast<uintptr_t>(hmod))) return CUDA_ERROR_INVALID_HANDLE;
    return CUDA_ERROR_NOT_FOUND;
  });
}
// No surface reference exists to name.
static CUresult cuSurfRefSetArray_impl(CUsurfref, CUarray, unsigned int);
VGPU_EXPORT CUresult cuSurfRefSetArray(CUsurfref a0, CUarray a1, unsigned int a2) { return traced("cuSurfRefSetArray", cuSurfRefSetArray_impl, a0, a1, a2); }
static CUresult cuSurfRefSetArray_impl(CUsurfref, CUarray, unsigned int) { return CUDA_ERROR_INVALID_HANDLE; }
static CUresult cuSurfRefGetArray_impl(CUarray*, CUsurfref);
VGPU_EXPORT CUresult cuSurfRefGetArray(CUarray* a0, CUsurfref a1) { return traced("cuSurfRefGetArray", cuSurfRefGetArray_impl, a0, a1); }
static CUresult cuSurfRefGetArray_impl(CUarray*, CUsurfref) { return CUDA_ERROR_INVALID_HANDLE; }

static CUresult cuTexRefCreate_impl(CUtexref* out);
VGPU_EXPORT CUresult cuTexRefCreate(CUtexref* out) { return traced("cuTexRefCreate", cuTexRefCreate_impl, out); }
static CUresult cuTexRefCreate_impl(CUtexref* out) {
  return api("cuTexRefCreate", true, false, [&](ShimState& s) {
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    const uintptr_t h = make_handle(s, kTagArray);
    s.texrefs[h] = TexRefRec{};
    *out = reinterpret_cast<CUtexref>(h);
    return CUDA_SUCCESS;
  });
}
static CUresult cuTexRefDestroy_impl(CUtexref t);
VGPU_EXPORT CUresult cuTexRefDestroy(CUtexref t) { return traced("cuTexRefDestroy", cuTexRefDestroy_impl, t); }
static CUresult cuTexRefDestroy_impl(CUtexref t) {
  return api("cuTexRefDestroy", true, false, [&](ShimState& s) {
    return s.texrefs.erase(reinterpret_cast<uintptr_t>(t)) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
  });
}
// Dimension 0 to 2; wrap, clamp, mirror or border.
static CUresult cuTexRefSetAddressMode_impl(CUtexref t, int dim, int mode);
VGPU_EXPORT CUresult cuTexRefSetAddressMode(CUtexref t, int dim, int mode) { return traced("cuTexRefSetAddressMode", cuTexRefSetAddressMode_impl, t, dim, mode); }
static CUresult cuTexRefSetAddressMode_impl(CUtexref t, int dim, int mode) {
  return with_texref("cuTexRefSetAddressMode", t, [&](ShimState&, TexRefRec& r) {
    if (dim < 0 || dim > 2 || mode < 0 || mode > 3) return CUDA_ERROR_INVALID_VALUE;
    r.address[dim] = mode;
    return CUDA_SUCCESS;
  });
}
static CUresult cuTexRefGetAddressMode_impl(int* mode, CUtexref t, int dim);
VGPU_EXPORT CUresult cuTexRefGetAddressMode(int* mode, CUtexref t, int dim) { return traced("cuTexRefGetAddressMode", cuTexRefGetAddressMode_impl, mode, t, dim); }
static CUresult cuTexRefGetAddressMode_impl(int* mode, CUtexref t, int dim) {
  return with_texref("cuTexRefGetAddressMode", t, [&](ShimState&, TexRefRec& r) {
    if (!mode || dim < 0 || dim > 2) return CUDA_ERROR_INVALID_VALUE;
    *mode = r.address[dim];
    return CUDA_SUCCESS;
  });
}
static CUresult cuTexRefSetFilterMode_impl(CUtexref t, int mode);
VGPU_EXPORT CUresult cuTexRefSetFilterMode(CUtexref t, int mode) { return traced("cuTexRefSetFilterMode", cuTexRefSetFilterMode_impl, t, mode); }
static CUresult cuTexRefSetFilterMode_impl(CUtexref t, int mode) {
  return with_texref("cuTexRefSetFilterMode", t, [&](ShimState&, TexRefRec& r) {
    if (mode < 0 || mode > 1) return CUDA_ERROR_INVALID_VALUE;
    r.filter = mode;
    return CUDA_SUCCESS;
  });
}
static CUresult cuTexRefGetFilterMode_impl(int* mode, CUtexref t);
VGPU_EXPORT CUresult cuTexRefGetFilterMode(int* mode, CUtexref t) { return traced("cuTexRefGetFilterMode", cuTexRefGetFilterMode_impl, mode, t); }
static CUresult cuTexRefGetFilterMode_impl(int* mode, CUtexref t) {
  return with_texref("cuTexRefGetFilterMode", t, [&](ShimState&, TexRefRec& r) {
    if (!mode) return CUDA_ERROR_INVALID_VALUE;
    *mode = r.filter;
    return CUDA_SUCCESS;
  });
}
// The CU_TRSF_* bits and no others.
static CUresult cuTexRefSetFlags_impl(CUtexref t, unsigned int flags);
VGPU_EXPORT CUresult cuTexRefSetFlags(CUtexref t, unsigned int flags) { return traced("cuTexRefSetFlags", cuTexRefSetFlags_impl, t, flags); }
static CUresult cuTexRefSetFlags_impl(CUtexref t, unsigned int flags) {
  return with_texref("cuTexRefSetFlags", t, [&](ShimState&, TexRefRec& r) {
    const unsigned known = CU_TRSF_READ_AS_INTEGER | CU_TRSF_NORMALIZED_COORDINATES | CU_TRSF_SRGB |
                           CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION | CU_TRSF_SEAMLESS_CUBEMAP;
    if (flags & ~known) return CUDA_ERROR_INVALID_VALUE;
    r.flags = flags;
    return CUDA_SUCCESS;
  });
}
static CUresult cuTexRefGetFlags_impl(unsigned int* flags, CUtexref t);
VGPU_EXPORT CUresult cuTexRefGetFlags(unsigned int* flags, CUtexref t) { return traced("cuTexRefGetFlags", cuTexRefGetFlags_impl, flags, t); }
static CUresult cuTexRefGetFlags_impl(unsigned int* flags, CUtexref t) {
  return with_texref("cuTexRefGetFlags", t, [&](ShimState&, TexRefRec& r) {
    if (!flags) return CUDA_ERROR_INVALID_VALUE;
    *flags = r.flags;
    return CUDA_SUCCESS;
  });
}
// 1, 2 or 4 channels. The format is kept as given: the card takes one no
// header names.
static CUresult cuTexRefSetFormat_impl(CUtexref t, int format, int channels);
VGPU_EXPORT CUresult cuTexRefSetFormat(CUtexref t, int format, int channels) { return traced("cuTexRefSetFormat", cuTexRefSetFormat_impl, t, format, channels); }
static CUresult cuTexRefSetFormat_impl(CUtexref t, int format, int channels) {
  return with_texref("cuTexRefSetFormat", t, [&](ShimState&, TexRefRec& r) {
    if (channels != 1 && channels != 2 && channels != 4) return CUDA_ERROR_INVALID_VALUE;
    r.format = format;
    r.channels = channels;
    return CUDA_SUCCESS;
  });
}
static CUresult cuTexRefGetFormat_impl(int* format, int* channels, CUtexref t);
VGPU_EXPORT CUresult cuTexRefGetFormat(int* format, int* channels, CUtexref t) { return traced("cuTexRefGetFormat", cuTexRefGetFormat_impl, format, channels, t); }
static CUresult cuTexRefGetFormat_impl(int* format, int* channels, CUtexref t) {
  return with_texref("cuTexRefGetFormat", t, [&](ShimState&, TexRefRec& r) {
    if (!format && !channels) return CUDA_ERROR_INVALID_VALUE;
    if (format) *format = r.format;
    if (channels) *channels = r.channels;
    return CUDA_SUCCESS;
  });
}
// Linear memory, bound at the texture alignment (512 bytes) at or below the
// address; the distance above it is returned as the offset a kernel adds.
static CUresult cuTexRefSetAddress_v2_impl(size_t* offset, CUtexref t, CUdeviceptr dptr, size_t);
VGPU_EXPORT CUresult cuTexRefSetAddress_v2(size_t* offset, CUtexref t, CUdeviceptr dptr, size_t a3) { return traced("cuTexRefSetAddress_v2", cuTexRefSetAddress_v2_impl, offset, t, dptr, a3); }
static CUresult cuTexRefSetAddress_v2_impl(size_t* offset, CUtexref t, CUdeviceptr dptr, size_t) {
  return with_texref("cuTexRefSetAddress", t, [&](ShimState& s, TexRefRec& r) {
    const CUdeviceptr align = static_cast<CUdeviceptr>(extra_attribute(current(s).profile(), 14));
    r.address_base = dptr / align * align;
    r.array = nullptr;
    if (offset) *offset = static_cast<size_t>(dptr - r.address_base);
    return CUDA_SUCCESS;
  });
}
static CUresult cuTexRefSetAddress_impl(size_t* offset, CUtexref t, CUdeviceptr dptr, size_t bytes);
VGPU_EXPORT CUresult cuTexRefSetAddress(size_t* offset, CUtexref t, CUdeviceptr dptr, size_t bytes) { return traced("cuTexRefSetAddress", cuTexRefSetAddress_impl, offset, t, dptr, bytes); }
static CUresult cuTexRefSetAddress_impl(size_t* offset, CUtexref t, CUdeviceptr dptr, size_t bytes) {
  return cuTexRefSetAddress_v2(offset, t, dptr, bytes);
}
// Pitched linear memory: the address at the texture alignment (512 bytes),
// the pitch at the texture pitch alignment (32 bytes), and a pitch that holds
// a row.
static CUresult cuTexRefSetAddress2D_v3_impl(CUtexref t, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t pitch);
VGPU_EXPORT CUresult cuTexRefSetAddress2D_v3(CUtexref t, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t pitch) { return traced("cuTexRefSetAddress2D_v3", cuTexRefSetAddress2D_v3_impl, t, desc, dptr, pitch); }
static CUresult cuTexRefSetAddress2D_v3_impl(CUtexref t, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr,
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
static CUresult cuTexRefSetAddress2D_v2_impl(CUtexref t, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t pitch);
VGPU_EXPORT CUresult cuTexRefSetAddress2D_v2(CUtexref t, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t pitch) { return traced("cuTexRefSetAddress2D_v2", cuTexRefSetAddress2D_v2_impl, t, desc, dptr, pitch); }
static CUresult cuTexRefSetAddress2D_v2_impl(CUtexref t, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr,
                                             size_t pitch) {
  return cuTexRefSetAddress2D_v3(t, desc, dptr, pitch);
}
static CUresult cuTexRefSetAddress2D_impl(CUtexref t, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t pitch);
VGPU_EXPORT CUresult cuTexRefSetAddress2D(CUtexref t, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t pitch) { return traced("cuTexRefSetAddress2D", cuTexRefSetAddress2D_impl, t, desc, dptr, pitch); }
static CUresult cuTexRefSetAddress2D_impl(CUtexref t, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr,
                                          size_t pitch) {
  return cuTexRefSetAddress2D_v3(t, desc, dptr, pitch);
}
// Bound to linear memory only; bound to an array, or to nothing, is INVALID_VALUE.
static CUresult cuTexRefGetAddress_v2_impl(CUdeviceptr* out, CUtexref t);
VGPU_EXPORT CUresult cuTexRefGetAddress_v2(CUdeviceptr* out, CUtexref t) { return traced("cuTexRefGetAddress_v2", cuTexRefGetAddress_v2_impl, out, t); }
static CUresult cuTexRefGetAddress_v2_impl(CUdeviceptr* out, CUtexref t) {
  return with_texref("cuTexRefGetAddress", t, [&](ShimState&, TexRefRec& r) {
    if (!out || !r.address_base) return CUDA_ERROR_INVALID_VALUE;
    *out = r.address_base;
    return CUDA_SUCCESS;
  });
}
static CUresult cuTexRefGetAddress_impl(CUdeviceptr* out, CUtexref t);
VGPU_EXPORT CUresult cuTexRefGetAddress(CUdeviceptr* out, CUtexref t) { return traced("cuTexRefGetAddress", cuTexRefGetAddress_impl, out, t); }
static CUresult cuTexRefGetAddress_impl(CUdeviceptr* out, CUtexref t) { return cuTexRefGetAddress_v2(out, t); }
// CU_TRSA_OVERRIDE_FORMAT, which takes the array's format, is the one flag.
static CUresult cuTexRefSetArray_impl(CUtexref t, CUarray a, unsigned int flags);
VGPU_EXPORT CUresult cuTexRefSetArray(CUtexref t, CUarray a, unsigned int flags) { return traced("cuTexRefSetArray", cuTexRefSetArray_impl, t, a, flags); }
static CUresult cuTexRefSetArray_impl(CUtexref t, CUarray a, unsigned int flags) {
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
static CUresult cuTexRefGetArray_impl(CUarray* out, CUtexref t);
VGPU_EXPORT CUresult cuTexRefGetArray(CUarray* out, CUtexref t) { return traced("cuTexRefGetArray", cuTexRefGetArray_impl, out, t); }
static CUresult cuTexRefGetArray_impl(CUarray* out, CUtexref t) {
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

// The format part of a descriptor, from a format and channel count the texture unit takes.
void apply_texture_format(vgpu::exec::TextureDesc* d, const vgpu::cuda::TexFormat& f) {
  d->kind = f.kind;
  d->channels = f.channels;
  for (int c = 0; c < 4; ++c) d->channel_bits[c] = f.bits[c];
  d->block = f.block;
  d->packed_1010102 = f.packed_1010102;
  d->texel_bytes = vgpu::cuda::sampled_texel_bytes(f);
}

// A linear or pitched resource takes the plain formats and 10:10:10:2 only (measured on an RTX 3060: the
// normalized and block-compressed ones are CUDA_ERROR_INVALID_VALUE there, as is a channel count the format
// does not have); the format's checks are texture_format_from_driver's.
CUresult linear_format(const CUarray_format& format, unsigned channels, vgpu::cuda::TexFormat* tf) {
  unsigned raw;
  std::memcpy(&raw, &format, sizeof raw);
  if (!vgpu::cuda::texture_format_from_driver(raw, channels, tf)) return CUDA_ERROR_INVALID_VALUE;
  if (tf->block != vgpu::exec::BlockFormat::None || tf->norm_only) return CUDA_ERROR_INVALID_VALUE;
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

CUresult describe_resource(ShimState& s, const ResourceDescABI& r, vgpu::exec::TextureDesc* d,
                           vgpu::cuda::TexFormat* tf) {
  switch (r.type) {
    case 0: {   // CU_RESOURCE_TYPE_ARRAY
      const ArrayRec* a = array_rec(s, r.res.array.array);
      if (!a) return CUDA_ERROR_INVALID_HANDLE;
      set_array_shape(d, *a);
      *tf = a->tf;
      apply_texture_format(d, *tf);
      return CUDA_SUCCESS;
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
      *tf = a.tf;
      apply_texture_format(d, *tf);
      return CUDA_SUCCESS;
    }
    case 2: {   // CU_RESOURCE_TYPE_LINEAR
      if (!r.res.linear.ptr) return CUDA_ERROR_INVALID_VALUE;
      if (CUresult e = linear_format(r.res.linear.format, r.res.linear.channels, tf)) return e;
      apply_texture_format(d, *tf);
      d->base = r.res.linear.ptr;
      d->width = static_cast<uint32_t>(r.res.linear.bytes / d->texel_bytes);
      d->pitch_bytes = 0;
      return CUDA_SUCCESS;
    }
    case 3: {   // CU_RESOURCE_TYPE_PITCH2D
      if (!r.res.pitch2d.ptr) return CUDA_ERROR_INVALID_VALUE;
      if (CUresult e = linear_format(r.res.pitch2d.format, r.res.pitch2d.channels, tf)) return e;
      apply_texture_format(d, *tf);
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

// CUDA_RESOURCE_VIEW_DESC.
struct ViewDescABI {
  int format;
  size_t width, height, depth;
  unsigned first_mip, last_mip, first_layer, last_layer;
  unsigned reserved[16];
};

CUresult make_object(const char* name, unsigned long long* out, const void* res, const void* tex,
                     const void* view, vgpu::exec::TexKind kind) {
  return api(name, true, false, [&](ShimState& s) -> CUresult {
    if (!out || !res) return CUDA_ERROR_INVALID_VALUE;
    vgpu::exec::TextureDesc d;
    d.object = kind;
    vgpu::cuda::TexFormat format;
    const auto& rdesc = *static_cast<const ResourceDescABI*>(res);
    if (CUresult e = describe_resource(s, rdesc, &d, &format)) return e;
    // A surface cannot be made over block-compressed data (measured on an RTX 3060: invalid value).
    if (kind == vgpu::exec::TexKind::Surface && format.block != vgpu::exec::BlockFormat::None)
      return CUDA_ERROR_INVALID_VALUE;
    bool srgb_flag = false;
    if (const auto* t = static_cast<const TextureDescABI*>(tex)) {
      for (int i = 0; i < 3; ++i) {
        if (t->address[i] < 0 || t->address[i] > 3) return CUDA_ERROR_INVALID_VALUE;
        d.address[i] = static_cast<vgpu::exec::TexAddress>(t->address[i]);   // the same order
      }
      // Any anisotropy is accepted, as the card accepts any (measured: 0, 1, 2, 8, 16, 17, 100 and
      // 2^32-1). It sharpens the blend between two levels of an explicit-level fetch (tex_mip_lod in
      // interpreter.cpp) and acts on nothing else a fetch with no derivatives does.
      d.max_anisotropy = t->max_anisotropy;
      d.filter = t->filter == 1 ? vgpu::exec::TexFilter::Linear : vgpu::exec::TexFilter::Point;
      d.normalized_coords = t->flags & CU_TRSF_NORMALIZED_COORDINATES;
      srgb_flag = t->flags & CU_TRSF_SRGB;
      d.srgb = srgb_flag;
      // Integer texels come back as floats in [0, 1] or [-1, 1] unless
      // CU_TRSF_READ_AS_INTEGER keeps them integers -- the driver's default is
      // the opposite of the runtime's. Only 8- and 16-bit channels promote; the normalized,
      // block-compressed and 10:10:10:2 formats always come back as floats.
      d.read_as_normalized_float = format.norm_only || format.packed_1010102 ||
                                   (!(t->flags & CU_TRSF_READ_AS_INTEGER) &&
                                    d.kind != vgpu::exec::ChannelKind::Float && d.channel_bits[0] < 32);
      static_assert(sizeof d.border_bits == sizeof t->border);
      std::memcpy(d.border_bits, t->border, sizeof d.border_bits);
      auto q = [](float v) { return static_cast<int32_t>(std::trunc(std::clamp(v, -1e6f, 1e6f) * 256)); };
      d.mip_filter = t->mip_filter == 1 ? vgpu::exec::TexFilter::Linear : vgpu::exec::TexFilter::Point;
      d.mip_bias = q(t->mip_bias);
      d.mip_bias_exact = std::isfinite(t->mip_bias) ? static_cast<double>(std::clamp(t->mip_bias, -1e6f, 1e6f)) * 256 : 0.0;
      d.mip_min = q(t->mip_min);
      d.mip_max = q(t->mip_max);
    } else {
      d.read_as_normalized_float = format.norm_only || format.packed_1010102;
    }
    // The sRGB block-compressed formats need the sRGB flag (measured: invalid value without it).
    if (format.srgb_only && !srgb_flag) return CUDA_ERROR_INVALID_VALUE;
    if (view) {
      const auto& v = *static_cast<const ViewDescABI*>(view);
      const bool array = rdesc.type == 0 || rdesc.type == 1;
      if (const int e = vgpu::cuda::apply_view_format(v.format, v.width, v.height, v.depth, v.first_mip,
                                                      v.last_mip, v.first_layer, v.last_layer, array, &d, &format))
        return static_cast<CUresult>(e);
      d.read_as_normalized_float = d.read_as_normalized_float || format.norm_only || format.packed_1010102;
    }
    // BC6H and BC7 arrays can be made and filled (their blocks are only bytes), but a texture of them cannot be
    // sampled here: their decoders are not written. Say so, by name, once (the real driver takes them).
    if (d.block == vgpu::exec::BlockFormat::BC6HU || d.block == vgpu::exec::BlockFormat::BC6HS ||
        d.block == vgpu::exec::BlockFormat::BC7) {
      static std::once_flag said;
      std::call_once(said, [&] {
        if (!quiet())
          std::fprintf(stderr, "[vgpu] %s: textures of BC6H and BC7 blocks are not implemented (BC1 to BC5 are); "
                               "returning CUDA_ERROR_NOT_SUPPORTED\n", name);
      });
      return CUDA_ERROR_NOT_SUPPORTED;
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

static CUresult cuTexObjectCreate_impl(unsigned long long* out, const void* res, const void* tex, const void* view);
VGPU_EXPORT CUresult cuTexObjectCreate(unsigned long long* out, const void* res, const void* tex, const void* view) { return traced("cuTexObjectCreate", cuTexObjectCreate_impl, out, res, tex, view); }
static CUresult cuTexObjectCreate_impl(unsigned long long* out, const void* res, const void* tex,
                                       const void* view) {
  return make_object("cuTexObjectCreate", out, res, tex, view, vgpu::exec::TexKind::Texture);
}
static CUresult cuTexObjectDestroy_impl(unsigned long long obj);
VGPU_EXPORT CUresult cuTexObjectDestroy(unsigned long long obj) { return traced("cuTexObjectDestroy", cuTexObjectDestroy_impl, obj); }
static CUresult cuTexObjectDestroy_impl(unsigned long long obj) {
  return destroy_object("cuTexObjectDestroy", obj);
}
static CUresult cuSurfObjectCreate_impl(unsigned long long* out, const void* res);
VGPU_EXPORT CUresult cuSurfObjectCreate(unsigned long long* out, const void* res) { return traced("cuSurfObjectCreate", cuSurfObjectCreate_impl, out, res); }
static CUresult cuSurfObjectCreate_impl(unsigned long long* out, const void* res) {
  return make_object("cuSurfObjectCreate", out, res, nullptr, nullptr, vgpu::exec::TexKind::Surface);
}
static CUresult cuSurfObjectDestroy_impl(unsigned long long obj);
VGPU_EXPORT CUresult cuSurfObjectDestroy(unsigned long long obj) { return traced("cuSurfObjectDestroy", cuSurfObjectDestroy_impl, obj); }
static CUresult cuSurfObjectDestroy_impl(unsigned long long obj) {
  return destroy_object("cuSurfObjectDestroy", obj);
}

static CUresult cuMemcpy3D_v2_impl(const CUDA_MEMCPY3D* p);
VGPU_EXPORT CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D* p) { return traced("cuMemcpy3D_v2", cuMemcpy3D_v2_impl, p); }
static CUresult cuMemcpy3D_v2_impl(const CUDA_MEMCPY3D* p) {
  return api("cuMemcpy3D", true, false, [&](ShimState& s) {
    if (!p) return CUDA_ERROR_INVALID_VALUE;
    return copy3d(s, *p);
  });
}
static CUresult cuMemcpy3D_impl(const CUDA_MEMCPY3D* p);
VGPU_EXPORT CUresult cuMemcpy3D(const CUDA_MEMCPY3D* p) { return traced("cuMemcpy3D", cuMemcpy3D_impl, p); }
static CUresult cuMemcpy3D_impl(const CUDA_MEMCPY3D* p) { return cuMemcpy3D_v2(p); }
static CUresult cuMemcpy3DAsync_v2_impl(const CUDA_MEMCPY3D* p, CUstream);
VGPU_EXPORT CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D* p, CUstream a1) { return traced("cuMemcpy3DAsync_v2", cuMemcpy3DAsync_v2_impl, p, a1); }
static CUresult cuMemcpy3DAsync_v2_impl(const CUDA_MEMCPY3D* p, CUstream) { return cuMemcpy3D_v2(p); }
static CUresult cuMemcpy3DAsync_impl(const CUDA_MEMCPY3D* p, CUstream st);
VGPU_EXPORT CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D* p, CUstream st) { return traced("cuMemcpy3DAsync", cuMemcpy3DAsync_impl, p, st); }
static CUresult cuMemcpy3DAsync_impl(const CUDA_MEMCPY3D* p, CUstream st) { return cuMemcpy3DAsync_v2(p, st); }
static CUresult cuMemcpy2D_v2_impl(const CUDA_MEMCPY2D* c);
VGPU_EXPORT CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D* c) { return traced("cuMemcpy2D_v2", cuMemcpy2D_v2_impl, c); }
static CUresult cuMemcpy2D_v2_impl(const CUDA_MEMCPY2D* c) {
  return api("cuMemcpy2D", true, false, [&](ShimState& s) {
    if (!c) return CUDA_ERROR_INVALID_VALUE;
    return copy3d(s, as3d(*c));
  });
}
static CUresult cuMemcpy2D_impl(const CUDA_MEMCPY2D* c);
VGPU_EXPORT CUresult cuMemcpy2D(const CUDA_MEMCPY2D* c) { return traced("cuMemcpy2D", cuMemcpy2D_impl, c); }
static CUresult cuMemcpy2D_impl(const CUDA_MEMCPY2D* c) { return cuMemcpy2D_v2(c); }
static CUresult cuMemcpy2DUnaligned_v2_impl(const CUDA_MEMCPY2D* c);
VGPU_EXPORT CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D* c) { return traced("cuMemcpy2DUnaligned_v2", cuMemcpy2DUnaligned_v2_impl, c); }
static CUresult cuMemcpy2DUnaligned_v2_impl(const CUDA_MEMCPY2D* c) { return cuMemcpy2D_v2(c); }
static CUresult cuMemcpy2DUnaligned_impl(const CUDA_MEMCPY2D* c);
VGPU_EXPORT CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D* c) { return traced("cuMemcpy2DUnaligned", cuMemcpy2DUnaligned_impl, c); }
static CUresult cuMemcpy2DUnaligned_impl(const CUDA_MEMCPY2D* c) { return cuMemcpy2D_v2(c); }
static CUresult cuMemcpy2DAsync_v2_impl(const CUDA_MEMCPY2D* c, CUstream);
VGPU_EXPORT CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D* c, CUstream a1) { return traced("cuMemcpy2DAsync_v2", cuMemcpy2DAsync_v2_impl, c, a1); }
static CUresult cuMemcpy2DAsync_v2_impl(const CUDA_MEMCPY2D* c, CUstream) { return cuMemcpy2D_v2(c); }
static CUresult cuMemcpy2DAsync_impl(const CUDA_MEMCPY2D* c, CUstream st);
VGPU_EXPORT CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D* c, CUstream st) { return traced("cuMemcpy2DAsync", cuMemcpy2DAsync_impl, c, st); }
static CUresult cuMemcpy2DAsync_impl(const CUDA_MEMCPY2D* c, CUstream st) { return cuMemcpy2DAsync_v2(c, st); }

// ---- stream-ordered memory pools ----
//
// See PoolRec. cuMemAllocAsync takes from the current context's device's
// current pool (its default pool until cuDeviceSetMemPool says otherwise), and
// cuMemFreeAsync gives the memory back to the pool it came from.

namespace {

PoolRec& default_pool_for(ShimState& s, int device) {
  auto it = s.default_pool.find(device);
  if (it != s.default_pool.end()) return *it->second;
  s.pools.emplace_back();
  PoolRec& p = s.pools.back();
  p.device = device;
  p.is_default = true;
  p.access[device] = 3;   // CU_MEM_ACCESS_FLAGS_PROT_READWRITE, on its own device
  s.default_pool[device] = &p;
  return p;
}

PoolRec& pool_for(ShimState& s, int device) {
  auto it = s.current_pool.find(device);
  return it != s.current_pool.end() ? *it->second : default_pool_for(s, device);
}

// A CUmemoryPool is the pool's address; this checks one before following it.
PoolRec* pool_from_handle(ShimState& s, const void* h) {
  for (PoolRec& p : s.pools)
    if (static_cast<const void*>(&p) == h && !p.destroyed) return &p;
  return nullptr;
}

// Gives cached blocks back to the device until no more than `keep` bytes are
// held, oldest first.
void release_cached(ShimState& s, PoolRec& p, uint64_t keep) {
  while (p.cached_bytes() > keep && !p.cached.empty()) {
    const auto [ptr, size] = p.cached.front();
    p.cached.erase(p.cached.begin());
    s.rt->device(p.device).memory().free(ptr);
    p.reserved -= size;
  }
}

CUdeviceptr pool_alloc(ShimState& s, PoolRec& p, size_t size) {
  // Reuse: the first cached block big enough, and no more than twice the size
  // asked for.
  for (size_t i = 0; i < p.cached.size(); ++i) {
    const auto [ptr, block] = p.cached[i];
    if (block >= size && block - size <= size) {
      p.cached.erase(p.cached.begin() + static_cast<long>(i));
      p.live[ptr] = block;
      p.used += block;
      p.used_high = std::max(p.used_high, p.used);
      return ptr;
    }
  }
  const uint64_t ptr = s.rt->device(p.device).memory().alloc(size);
  p.live[ptr] = size;
  p.used += size;
  p.reserved += size;
  p.used_high = std::max(p.used_high, p.used);
  p.reserved_high = std::max(p.reserved_high, p.reserved);
  return ptr;
}

// The pool a pointer was allocated from, or null.
PoolRec* pool_of(ShimState& s, uint64_t ptr) {
  for (PoolRec& p : s.pools)
    if (p.live.count(ptr)) return &p;
  return nullptr;
}

void pool_free(ShimState& s, PoolRec& p, uint64_t ptr) {
  const uint64_t size = p.live[ptr];
  p.live.erase(ptr);
  p.used -= size;
  p.cached.emplace_back(ptr, size);
  // The default threshold is 0, so by default this hands the memory straight
  // back to the device and the pool keeps nothing.
  release_cached(s, p, p.threshold);
}

// CUmemLocation and CUmemAccessDesc, as cuda.h lays them out.
struct MemLocationABI { int type; int id; };
struct MemAccessDescABI { MemLocationABI location; int flags; };
// CUmemPoolProps: the fields this reads, which sit at the same offsets in every
// CUDA 12 cuda.h (what follows them -- a maximum size, a usage mask, reserved
// bytes -- was added over the releases and is not looked at).
struct MemPoolPropsABI {
  int alloc_type;
  int handle_types;
  MemLocationABI location;
  void* win32_security_attributes;
};
constexpr int kMemAllocationTypePinned = 1;      // CU_MEM_ALLOCATION_TYPE_PINNED
constexpr int kMemLocationTypeDevice = 1;        // CU_MEM_LOCATION_TYPE_DEVICE
constexpr int kMemAccessNone = 0, kMemAccessRead = 1, kMemAccessReadWrite = 3;   // CU_MEM_ACCESS_FLAGS_PROT_*

}  // namespace

// Zero bytes succeed, as the card's do, with a null pointer.
static CUresult cuMemAllocAsync_impl(CUdeviceptr* dptr, size_t bytesize, CUstream);
VGPU_EXPORT CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream a2) { return traced("cuMemAllocAsync", cuMemAllocAsync_impl, dptr, bytesize, a2); }
static CUresult cuMemAllocAsync_impl(CUdeviceptr* dptr, size_t bytesize, CUstream) {
  return api("cuMemAllocAsync", true, false, [&](ShimState& s) {
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;
    if (bytesize == 0) {
      *dptr = 0;
      return CUDA_SUCCESS;
    }
    *dptr = pool_alloc(s, pool_for(s, current_device(s)), bytesize);
    return CUDA_SUCCESS;
  });
}
static CUresult cuMemAllocFromPoolAsync_impl(CUdeviceptr* dptr, size_t bytesize, void* pool, CUstream);
VGPU_EXPORT CUresult cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, void* pool, CUstream a3) { return traced("cuMemAllocFromPoolAsync", cuMemAllocFromPoolAsync_impl, dptr, bytesize, pool, a3); }
static CUresult cuMemAllocFromPoolAsync_impl(CUdeviceptr* dptr, size_t bytesize, void* pool, CUstream) {
  return api("cuMemAllocFromPoolAsync", true, false, [&](ShimState& s) {
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;
    PoolRec* p = pool_from_handle(s, pool);
    if (!p) return CUDA_ERROR_INVALID_VALUE;
    if (bytesize == 0) {
      *dptr = 0;
      return CUDA_SUCCESS;
    }
    *dptr = pool_alloc(s, *p, bytesize);
    return CUDA_SUCCESS;
  });
}
// A pointer from a pool goes back to it; any other device allocation is freed
// as cuMemFree frees it.
static CUresult cuMemFreeAsync_impl(CUdeviceptr dptr, CUstream);
VGPU_EXPORT CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream a1) { return traced("cuMemFreeAsync", cuMemFreeAsync_impl, dptr, a1); }
static CUresult cuMemFreeAsync_impl(CUdeviceptr dptr, CUstream) {
  if (dptr == 0) return CUDA_SUCCESS;
  bool pooled = false;
  const CUresult r = api("cuMemFreeAsync", true, false, [&](ShimState& s) {
    if (PoolRec* p = pool_of(s, dptr)) {
      pool_free(s, *p, dptr);
      pooled = true;
    }
    return CUDA_SUCCESS;
  });
  return pooled || r != CUDA_SUCCESS ? r : cuMemFree_v2_impl(dptr);
}

static CUresult cuDeviceGetDefaultMemPool_impl(void** pool, CUdevice dev);
VGPU_EXPORT CUresult cuDeviceGetDefaultMemPool(void** pool, CUdevice dev) { return traced("cuDeviceGetDefaultMemPool", cuDeviceGetDefaultMemPool_impl, pool, dev); }
static CUresult cuDeviceGetDefaultMemPool_impl(void** pool, CUdevice dev) {
  return api("cuDeviceGetDefaultMemPool", true, false, [&](ShimState& s) {
    if (!pool) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    *pool = &default_pool_for(s, dev);
    return CUDA_SUCCESS;
  });
}
static CUresult cuDeviceGetMemPool_impl(void** pool, CUdevice dev);
VGPU_EXPORT CUresult cuDeviceGetMemPool(void** pool, CUdevice dev) { return traced("cuDeviceGetMemPool", cuDeviceGetMemPool_impl, pool, dev); }
static CUresult cuDeviceGetMemPool_impl(void** pool, CUdevice dev) {
  return api("cuDeviceGetMemPool", true, false, [&](ShimState& s) {
    if (!pool) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    *pool = &pool_for(s, dev);
    return CUDA_SUCCESS;
  });
}
static CUresult cuDeviceSetMemPool_impl(CUdevice dev, void* pool);
VGPU_EXPORT CUresult cuDeviceSetMemPool(CUdevice dev, void* pool) { return traced("cuDeviceSetMemPool", cuDeviceSetMemPool_impl, dev, pool); }
static CUresult cuDeviceSetMemPool_impl(CUdevice dev, void* pool) {
  return api("cuDeviceSetMemPool", true, false, [&](ShimState& s) {
    check_device(s, dev);
    PoolRec* p = pool_from_handle(s, pool);
    // A pool belongs to the device it was made for.
    if (!p || p->device != dev) return CUDA_ERROR_INVALID_VALUE;
    s.current_pool[dev] = p;
    return CUDA_SUCCESS;
  });
}

static CUresult cuMemPoolCreate_impl(void** pool, const void* poolProps);
VGPU_EXPORT CUresult cuMemPoolCreate(void** pool, const void* poolProps) { return traced("cuMemPoolCreate", cuMemPoolCreate_impl, pool, poolProps); }
static CUresult cuMemPoolCreate_impl(void** pool, const void* poolProps) {
  return api("cuMemPoolCreate", true, false, [&](ShimState& s) {
    if (!pool || !poolProps) return CUDA_ERROR_INVALID_VALUE;
    const auto* props = static_cast<const MemPoolPropsABI*>(poolProps);
    if (props->alloc_type != kMemAllocationTypePinned) return CUDA_ERROR_INVALID_VALUE;
    // Measured on an RTX 3060 (driver 13.2): a location that is none, a device that does not exist and
    // a handle type (the device supports none: cuDeviceGetAttribute
    // CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES is 0) are all CUDA_ERROR_INVALID_VALUE. A pool on
    // the host, or on a NUMA node, would hold host memory the device reaches: the card makes one, and
    // only device pools are made here.
    if (props->location.type == 0) return CUDA_ERROR_INVALID_VALUE;
    if (props->location.type != kMemLocationTypeDevice) {
      report("cuMemPoolCreate", "only a pool located on a device is implemented (a host or NUMA "
                                "location is not)");
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (props->location.id < 0 || props->location.id >= s.rt->device_count()) return CUDA_ERROR_INVALID_VALUE;
    if (props->handle_types != 0) return CUDA_ERROR_INVALID_VALUE;
    s.pools.emplace_back();
    PoolRec& p = s.pools.back();
    p.device = props->location.id;
    p.access[p.device] = kMemAccessReadWrite;
    *pool = &p;
    return CUDA_SUCCESS;
  });
}
static CUresult cuMemPoolDestroy_impl(void* pool);
VGPU_EXPORT CUresult cuMemPoolDestroy(void* pool) { return traced("cuMemPoolDestroy", cuMemPoolDestroy_impl, pool); }
static CUresult cuMemPoolDestroy_impl(void* pool) {
  return api("cuMemPoolDestroy", true, false, [&](ShimState& s) {
    PoolRec* p = pool_from_handle(s, pool);
    if (!p || p->is_default) return CUDA_ERROR_INVALID_VALUE;   // the default pool is the device's
    // Outstanding allocations would have to outlive the pool, and freeing them
    // afterwards needs it. Refused instead.
    if (!p->live.empty()) {
      report("cuMemPoolDestroy", std::to_string(p->live.size()) +
                                     " allocation(s) are still outstanding; free them with cuMemFreeAsync first");
      return CUDA_ERROR_INVALID_VALUE;
    }
    release_cached(s, *p, 0);
    p->destroyed = true;
    for (auto it = s.current_pool.begin(); it != s.current_pool.end();)
      it = it->second == p ? s.current_pool.erase(it) : std::next(it);
    return CUDA_SUCCESS;
  });
}
static CUresult cuMemPoolTrimTo_impl(void* pool, size_t minBytesToKeep);
VGPU_EXPORT CUresult cuMemPoolTrimTo(void* pool, size_t minBytesToKeep) { return traced("cuMemPoolTrimTo", cuMemPoolTrimTo_impl, pool, minBytesToKeep); }
static CUresult cuMemPoolTrimTo_impl(void* pool, size_t minBytesToKeep) {
  return api("cuMemPoolTrimTo", true, false, [&](ShimState& s) {
    PoolRec* p = pool_from_handle(s, pool);
    if (!p) return CUDA_ERROR_INVALID_VALUE;
    release_cached(s, *p, minBytesToKeep);
    return CUDA_SUCCESS;
  });
}

// CUmemPool_attribute: the three reuse policies are ints, the rest 64-bit.
namespace {
constexpr int kPoolReuseFollowEventDeps = 1, kPoolReuseAllowOpportunistic = 2, kPoolReuseAllowInternalDeps = 3,
              kPoolReleaseThreshold = 4, kPoolReservedCurrent = 5, kPoolReservedHigh = 6, kPoolUsedCurrent = 7,
              kPoolUsedHigh = 8;
}  // namespace
static CUresult cuMemPoolSetAttribute_impl(void* pool, int attr, void* value);
VGPU_EXPORT CUresult cuMemPoolSetAttribute(void* pool, int attr, void* value) { return traced("cuMemPoolSetAttribute", cuMemPoolSetAttribute_impl, pool, attr, value); }
static CUresult cuMemPoolSetAttribute_impl(void* pool, int attr, void* value) {
  return api("cuMemPoolSetAttribute", true, false, [&](ShimState& s) {
    PoolRec* p = pool_from_handle(s, pool);
    if (!p || !value) return CUDA_ERROR_INVALID_VALUE;
    switch (attr) {
      case kPoolReleaseThreshold:
        p->threshold = *static_cast<unsigned long long*>(value);
        // Lowering it takes effect now; every operation here is already synchronous.
        release_cached(s, *p, p->threshold);
        return CUDA_SUCCESS;
      case kPoolReuseFollowEventDeps: p->reuse_follow_event_deps = *static_cast<int*>(value); return CUDA_SUCCESS;
      case kPoolReuseAllowOpportunistic: p->reuse_allow_opportunistic = *static_cast<int*>(value); return CUDA_SUCCESS;
      case kPoolReuseAllowInternalDeps: p->reuse_allow_internal_deps = *static_cast<int*>(value); return CUDA_SUCCESS;
      // The documented way to reset a high-water mark is to write 0 to it.
      case kPoolReservedHigh:
        if (*static_cast<unsigned long long*>(value) != 0) return CUDA_ERROR_INVALID_VALUE;
        p->reserved_high = p->reserved;
        return CUDA_SUCCESS;
      case kPoolUsedHigh:
        if (*static_cast<unsigned long long*>(value) != 0) return CUDA_ERROR_INVALID_VALUE;
        p->used_high = p->used;
        return CUDA_SUCCESS;
      default:
        return CUDA_ERROR_INVALID_VALUE;   // the current totals are read-only
    }
  });
}
static CUresult cuMemPoolGetAttribute_impl(void* pool, int attr, void* value);
VGPU_EXPORT CUresult cuMemPoolGetAttribute(void* pool, int attr, void* value) { return traced("cuMemPoolGetAttribute", cuMemPoolGetAttribute_impl, pool, attr, value); }
static CUresult cuMemPoolGetAttribute_impl(void* pool, int attr, void* value) {
  return api("cuMemPoolGetAttribute", true, false, [&](ShimState& s) {
    PoolRec* p = pool_from_handle(s, pool);
    if (!p || !value) return CUDA_ERROR_INVALID_VALUE;
    auto u64 = [&](uint64_t v) {
      *static_cast<unsigned long long*>(value) = v;
      return CUDA_SUCCESS;
    };
    switch (attr) {
      case kPoolReuseFollowEventDeps: *static_cast<int*>(value) = p->reuse_follow_event_deps; return CUDA_SUCCESS;
      case kPoolReuseAllowOpportunistic: *static_cast<int*>(value) = p->reuse_allow_opportunistic; return CUDA_SUCCESS;
      case kPoolReuseAllowInternalDeps: *static_cast<int*>(value) = p->reuse_allow_internal_deps; return CUDA_SUCCESS;
      case kPoolReleaseThreshold: return u64(p->threshold);
      case kPoolReservedCurrent: return u64(p->reserved);
      case kPoolReservedHigh: return u64(p->reserved_high);
      case kPoolUsedCurrent: return u64(p->used);
      case kPoolUsedHigh: return u64(p->used_high);
      default: return CUDA_ERROR_INVALID_VALUE;
    }
  });
}

// What a pool's memory may be reached from. It starts as the device it lives
// on, read and write, and no other; a request can widen that to a peer or take
// it away. Nothing enforces it -- every kernel here reaches every device's
// memory -- so it is state a program reads back, not a fence.
static CUresult cuMemPoolSetAccess_impl(void* pool, const void* map, size_t count);
VGPU_EXPORT CUresult cuMemPoolSetAccess(void* pool, const void* map, size_t count) { return traced("cuMemPoolSetAccess", cuMemPoolSetAccess_impl, pool, map, count); }
static CUresult cuMemPoolSetAccess_impl(void* pool, const void* map, size_t count) {
  return api("cuMemPoolSetAccess", true, false, [&](ShimState& s) {
    PoolRec* p = pool_from_handle(s, pool);
    if (!p || (!map && count)) return CUDA_ERROR_INVALID_VALUE;
    const auto* descs = static_cast<const MemAccessDescABI*>(map);
    for (size_t i = 0; i < count; ++i) {
      if (descs[i].location.type != kMemLocationTypeDevice || descs[i].location.id < 0 ||
          descs[i].location.id >= s.rt->device_count())
        return CUDA_ERROR_INVALID_VALUE;
      if (descs[i].flags != kMemAccessNone && descs[i].flags != kMemAccessRead &&
          descs[i].flags != kMemAccessReadWrite)
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (size_t i = 0; i < count; ++i) p->access[descs[i].location.id] = descs[i].flags;
    return CUDA_SUCCESS;
  });
}
static CUresult cuMemPoolGetAccess_impl(int* flags, void* pool, void* location);
VGPU_EXPORT CUresult cuMemPoolGetAccess(int* flags, void* pool, void* location) { return traced("cuMemPoolGetAccess", cuMemPoolGetAccess_impl, flags, pool, location); }
static CUresult cuMemPoolGetAccess_impl(int* flags, void* pool, void* location) {
  return api("cuMemPoolGetAccess", true, false, [&](ShimState& s) {
    PoolRec* p = pool_from_handle(s, pool);
    if (!p || !flags || !location) return CUDA_ERROR_INVALID_VALUE;
    const auto* loc = static_cast<const MemLocationABI*>(location);
    if (loc->type != kMemLocationTypeDevice || loc->id < 0 || loc->id >= s.rt->device_count())
      return CUDA_ERROR_INVALID_VALUE;
    const auto it = p->access.find(loc->id);
    *flags = it == p->access.end() ? kMemAccessNone : it->second;
    return CUDA_SUCCESS;
  });
}

// Sharing a pool, or one of its allocations, with another process needs memory
// that process can map; device memory here is this process's own sparse backing.
// Sharing a pool, or a pointer from one, with another process: the device supports no handle type for pools
// (CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES is 0), so no pool can be exported. Measured on an
// RTX 3060, as the runtime's calls of the same names are: the exports are CUDA_ERROR_INVALID_VALUE
// whatever is passed; an import checks its pointers, flags (0 only) and type first -- the file descriptor, Win32 and
// fabric types the device does not support are CUDA_ERROR_NOT_SUPPORTED, any other type
// CUDA_ERROR_INVALID_VALUE; the pointer calls are CUDA_ERROR_INVALID_VALUE.
static CUresult cuMemPoolExportToShareableHandle_impl(void*, void*, int, unsigned long long);
VGPU_EXPORT CUresult cuMemPoolExportToShareableHandle(void* a0, void* a1, int a2, unsigned long long a3) { return traced("cuMemPoolExportToShareableHandle", cuMemPoolExportToShareableHandle_impl, a0, a1, a2, a3); }
static CUresult cuMemPoolExportToShareableHandle_impl(void*, void*, int, unsigned long long) {
  return api("cuMemPoolExportToShareableHandle", true, false, [&](ShimState&) { return CUDA_ERROR_INVALID_VALUE; });
}
static CUresult cuMemPoolImportFromShareableHandle_impl(void** pool, void* handle, int type, unsigned long long flags);
VGPU_EXPORT CUresult cuMemPoolImportFromShareableHandle(void** pool, void* handle, int type, unsigned long long flags) { return traced("cuMemPoolImportFromShareableHandle", cuMemPoolImportFromShareableHandle_impl, pool, handle, type, flags); }
static CUresult cuMemPoolImportFromShareableHandle_impl(void** pool, void* handle, int type, unsigned long long flags) {
  return api("cuMemPoolImportFromShareableHandle", true, false, [&](ShimState&) {
    if (!pool || !handle || flags != 0) return CUDA_ERROR_INVALID_VALUE;
    return type == 1 || type == 2 || type == 8 ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_VALUE;
  });
}
static CUresult cuMemPoolExportPointer_impl(void*, CUdeviceptr);
VGPU_EXPORT CUresult cuMemPoolExportPointer(void* a0, CUdeviceptr a1) { return traced("cuMemPoolExportPointer", cuMemPoolExportPointer_impl, a0, a1); }
static CUresult cuMemPoolExportPointer_impl(void*, CUdeviceptr) {
  return api("cuMemPoolExportPointer", true, false, [&](ShimState&) { return CUDA_ERROR_INVALID_VALUE; });
}
static CUresult cuMemPoolImportPointer_impl(CUdeviceptr*, void*, void*);
VGPU_EXPORT CUresult cuMemPoolImportPointer(CUdeviceptr* a0, void* a1, void* a2) { return traced("cuMemPoolImportPointer", cuMemPoolImportPointer_impl, a0, a1, a2); }
static CUresult cuMemPoolImportPointer_impl(CUdeviceptr*, void*, void*) {
  return api("cuMemPoolImportPointer", true, false, [&](ShimState&) { return CUDA_ERROR_INVALID_VALUE; });
}

/* ---- pitched memory, module globals, context configuration ----
 * Each answers as an RTX 3060's and an RTX 3080 Ti's drivers do (they agree). */

// Rows padded to 512 bytes; the element size must be 4, 8 or 16 and neither
// dimension 0.
static CUresult cuMemAllocPitch_v2_impl(CUdeviceptr* dptr, size_t* pitch, size_t width, size_t height, unsigned int elem);
VGPU_EXPORT CUresult cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pitch, size_t width, size_t height, unsigned int elem) { return traced("cuMemAllocPitch_v2", cuMemAllocPitch_v2_impl, dptr, pitch, width, height, elem); }
static CUresult cuMemAllocPitch_v2_impl(CUdeviceptr* dptr, size_t* pitch, size_t width, size_t height,
                                        unsigned int elem) {
  return api("cuMemAllocPitch", true, false, [&](ShimState& s) {
    if (!dptr || !pitch || width == 0 || height == 0 || (elem != 4 && elem != 8 && elem != 16))
      return CUDA_ERROR_INVALID_VALUE;
    *pitch = (width + 511) / 512 * 512;
    *dptr = current(s).memory().alloc(*pitch * height);
    return CUDA_SUCCESS;
  });
}
static CUresult cuMemAllocPitch_impl(CUdeviceptr* dptr, size_t* pitch, size_t width, size_t height, unsigned int elem);
VGPU_EXPORT CUresult cuMemAllocPitch(CUdeviceptr* dptr, size_t* pitch, size_t width, size_t height, unsigned int elem) { return traced("cuMemAllocPitch", cuMemAllocPitch_impl, dptr, pitch, width, height, elem); }
static CUresult cuMemAllocPitch_impl(CUdeviceptr* dptr, size_t* pitch, size_t width, size_t height,
                                     unsigned int elem) {
  return cuMemAllocPitch_v2(dptr, pitch, width, height, elem);
}

// A module's __device__ variable: its address and size, either of which may be
// omitted but not both. A name the module does not declare is NOT_FOUND; a
// kernel's name is INVALID_VALUE, as on the card.
static CUresult cuModuleGetGlobal_v2_impl(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name);
VGPU_EXPORT CUresult cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name) { return traced("cuModuleGetGlobal_v2", cuModuleGetGlobal_v2_impl, dptr, bytes, hmod, name); }
static CUresult cuModuleGetGlobal_v2_impl(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name) {
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
static CUresult cuModuleGetGlobal_impl(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name);
VGPU_EXPORT CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name) { return traced("cuModuleGetGlobal", cuModuleGetGlobal_impl, dptr, bytes, hmod, name); }
static CUresult cuModuleGetGlobal_impl(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name) {
  return cuModuleGetGlobal_v2(dptr, bytes, hmod, name);
}

// The L1/shared split is a preference the simulator has no cache to apply to;
// it is kept and read back, as the card reads it back.
static CUresult cuCtxSetCacheConfig_impl(int config);
VGPU_EXPORT CUresult cuCtxSetCacheConfig(int config) { return traced("cuCtxSetCacheConfig", cuCtxSetCacheConfig_impl, config); }
static CUresult cuCtxSetCacheConfig_impl(int config) {
  return api("cuCtxSetCacheConfig", true, false, [&](ShimState& s) {
    if (config < 0 || config > 3) return CUDA_ERROR_INVALID_VALUE;
    s.cache_config = config;
    return CUDA_SUCCESS;
  });
}
static CUresult cuCtxGetCacheConfig_impl(int* config);
VGPU_EXPORT CUresult cuCtxGetCacheConfig(int* config) { return traced("cuCtxGetCacheConfig", cuCtxGetCacheConfig_impl, config); }
static CUresult cuCtxGetCacheConfig_impl(int* config) {
  return api("cuCtxGetCacheConfig", true, false, [&](ShimState& s) {
    if (!config) return CUDA_ERROR_INVALID_VALUE;
    *config = s.cache_config;
    return CUDA_SUCCESS;
  });
}
// Shared memory banks are four bytes wide on every GPU this simulates: a
// configuration is accepted and has no effect, and the answer stays
// CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE, as the card's does.
static CUresult cuCtxSetSharedMemConfig_impl(int config);
VGPU_EXPORT CUresult cuCtxSetSharedMemConfig(int config) { return traced("cuCtxSetSharedMemConfig", cuCtxSetSharedMemConfig_impl, config); }
static CUresult cuCtxSetSharedMemConfig_impl(int config) {
  return api("cuCtxSetSharedMemConfig", true, false, [&](ShimState&) {
    return config >= 0 && config <= 2 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
  });
}
static CUresult cuCtxGetSharedMemConfig_impl(int* config);
VGPU_EXPORT CUresult cuCtxGetSharedMemConfig(int* config) { return traced("cuCtxGetSharedMemConfig", cuCtxGetSharedMemConfig_impl, config); }
static CUresult cuCtxGetSharedMemConfig_impl(int* config) {
  return api("cuCtxGetSharedMemConfig", true, false, [&](ShimState&) {
    if (!config) return CUDA_ERROR_INVALID_VALUE;
    *config = 1;
    return CUDA_SUCCESS;
  });
}
static CUresult cuFuncSetSharedMemConfig_impl(CUfunction, int config);
VGPU_EXPORT CUresult cuFuncSetSharedMemConfig(CUfunction a0, int config) { return traced("cuFuncSetSharedMemConfig", cuFuncSetSharedMemConfig_impl, a0, config); }
static CUresult cuFuncSetSharedMemConfig_impl(CUfunction, int config) {
  return config >= 0 && config <= 2 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

// A stream callback runs at once: the stream is synchronous, so the work
// queued before it is done. It is told the stream succeeded. Flags must be 0.
static CUresult cuStreamAddCallback_impl(CUstream stream, CUstreamCallback cb, void* user, unsigned int flags);
VGPU_EXPORT CUresult cuStreamAddCallback(CUstream stream, CUstreamCallback cb, void* user, unsigned int flags) { return traced("cuStreamAddCallback", cuStreamAddCallback_impl, stream, cb, user, flags); }
static CUresult cuStreamAddCallback_impl(CUstream stream, CUstreamCallback cb, void* user, unsigned int flags) {
  if (!cb || flags != 0) return CUDA_ERROR_INVALID_VALUE;
  cb(stream, CUDA_SUCCESS, user);
  return CUDA_SUCCESS;
}

/* ---- streams (all synchronous) and events (wall-clock timestamps) ---- */

// The priority is a record for a profiler; every stream is the same here.
static CUresult stream_create(CUstream* s_out, unsigned int flags, int priority) {
  return api("cuStreamCreate", true, false, [&](ShimState& s) {
    if (!s_out) return CUDA_ERROR_INVALID_VALUE;
    uintptr_t h = make_handle(s, kTagStream);
    s.streams.insert(h);
    *s_out = reinterpret_cast<CUstream>(h);
    // A priority outside the range is clamped to it, as the card does, and read back by
    // cuStreamGetPriority; streams run in order, so it changes nothing else.
    s.stream_priority[h] = std::min(vgpu::cuda::kLeastStreamPriority, std::max(vgpu::cuda::kGreatestStreamPriority, priority));
    announce_stream_created(s, h, flags, priority);
    return CUDA_SUCCESS;
  });
}
static CUresult cuStreamCreate_impl(CUstream* s_out, unsigned int flags) {
  return stream_create(s_out, flags, 0);
}
VGPU_EXPORT CUresult cuStreamCreate(CUstream* s_out, unsigned int flags) {
  return traced("cuStreamCreate", cuStreamCreate_impl, s_out, flags);
}
static CUresult cuStreamCreateWithPriority_impl(CUstream* s_out, unsigned int flags, int priority) {
  return stream_create(s_out, flags, priority);
}
VGPU_EXPORT CUresult cuStreamCreateWithPriority(CUstream* s_out, unsigned int flags, int priority) {
  return traced("cuStreamCreateWithPriority", cuStreamCreateWithPriority_impl, s_out, flags, priority);
}
static CUresult cuStreamDestroy_v2_impl(CUstream stream) {
  return api("cuStreamDestroy", true, false, [&](ShimState& s) {
    const uintptr_t h = reinterpret_cast<uintptr_t>(stream);
    if (s.streams.count(h))
      vgpu::profiling::notify_resource(vgpu::profiling::Resource::StreamDestroyStarting, h, profiled_device(s));
    s.streams.erase(h);
    s.stream_priority.erase(h);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuStreamDestroy_v2(CUstream stream) {
  return traced("cuStreamDestroy_v2", cuStreamDestroy_v2_impl, stream);
}
static CUresult cuStreamDestroy_impl(CUstream stream);
VGPU_EXPORT CUresult cuStreamDestroy(CUstream stream) { return traced("cuStreamDestroy", cuStreamDestroy_impl, stream); }
static CUresult cuStreamDestroy_impl(CUstream stream) { return cuStreamDestroy_v2_impl(stream); }
// A wait that returned is recorded for a profiler, and the stream and context
// waits are told to a subscriber (vgpu/profiling.hpp).
static CUresult cuStreamSynchronize_impl(CUstream stream) {
  return api("cuStreamSynchronize", true, false, [&](ShimState& s) {
    if (ctx_stack().empty()) return CUDA_ERROR_INVALID_CONTEXT;
    (void)current_device(s);  // requires a current context
    vgpu::profiling::SyncScope waited(vgpu::profiling::SyncKind::Stream, reinterpret_cast<uint64_t>(stream), 0,
                                      profiled_device(s));
    return CUDA_SUCCESS;      // everything is synchronous today
  });
}
VGPU_EXPORT CUresult cuStreamSynchronize(CUstream stream) {
  return traced("cuStreamSynchronize", cuStreamSynchronize_impl, stream);
}
// Streams are synchronous, so everything queued before the function has run
// by the time it is called, which is all a host function is promised.
static CUresult cuLaunchHostFunc_impl(CUstream, CUhostFn fn, void* user);
VGPU_EXPORT CUresult cuLaunchHostFunc(CUstream a0, CUhostFn fn, void* user) { return traced("cuLaunchHostFunc", cuLaunchHostFunc_impl, a0, fn, user); }
static CUresult cuLaunchHostFunc_impl(CUstream, CUhostFn fn, void* user) {
  if (!fn) return CUDA_ERROR_INVALID_VALUE;
  fn(user);
  return CUDA_SUCCESS;
}
// Always idle. The card records a query that finds the stream idle as a wait
// on it, and tells a subscriber the stream was synchronized.
static CUresult cuStreamQuery_impl(CUstream stream) {
  {   // with no current context the card answers CUDA_ERROR_INVALID_CONTEXT
    ShimState& st = state();
    std::lock_guard<std::recursive_mutex> lock(st.mu);
    if (st.initialized && ctx_stack().empty()) return CUDA_ERROR_INVALID_CONTEXT;
  }
  const CUresult r = dead_context();
  if (r == CUDA_SUCCESS)
    vgpu::profiling::SyncScope waited(vgpu::profiling::SyncKind::Stream, reinterpret_cast<uint64_t>(stream), 0,
                                      profiled_device());
  return r;
}
VGPU_EXPORT CUresult cuStreamQuery(CUstream stream) { return traced("cuStreamQuery", cuStreamQuery_impl, stream); }
static CUresult cuStreamWaitEvent_impl(CUstream stream, void* event, unsigned int) {
  const CUresult r = dead_context();
  if (r == CUDA_SUCCESS)
    vgpu::profiling::SyncScope waited(vgpu::profiling::SyncKind::StreamWaitEvent,
                                      reinterpret_cast<uint64_t>(stream), reinterpret_cast<uint64_t>(event),
                                      profiled_device());
  return r;
}
VGPU_EXPORT CUresult cuStreamWaitEvent(CUstream stream, void* event, unsigned int flags) {
  return traced("cuStreamWaitEvent", cuStreamWaitEvent_impl, stream, event, flags);
}
static CUresult cuStreamGetPriority_impl(CUstream stream, int* p);
VGPU_EXPORT CUresult cuStreamGetPriority(CUstream stream, int* p) { return traced("cuStreamGetPriority", cuStreamGetPriority_impl, stream, p); }
static CUresult cuStreamGetPriority_impl(CUstream stream, int* p) {
  if (!p) return CUDA_SUCCESS;
  ShimState& s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  const auto it = s.stream_priority.find(reinterpret_cast<uintptr_t>(stream));
  *p = it == s.stream_priority.end() ? 0 : it->second;
  return CUDA_SUCCESS;
}
static CUresult cuStreamGetFlags_impl(CUstream, unsigned int* f);
VGPU_EXPORT CUresult cuStreamGetFlags(CUstream a0, unsigned int* f) { return traced("cuStreamGetFlags", cuStreamGetFlags_impl, a0, f); }
static CUresult cuStreamGetFlags_impl(CUstream, unsigned int* f) {
  if (f) *f = 0;
  return CUDA_SUCCESS;
}
// A stream belongs to the context that was current when it was created, and the
// simulator keeps one context per thread's stack, so that is the answer here;
// the legacy default stream (NULL) answers the current context as well. With
// no context current the driver answers CUDA_ERROR_INVALID_CONTEXT and leaves
// the output alone. Kokkos' CUDA backend asks for it when it wraps a stream.
static CUresult cuStreamGetCtx_impl(CUstream, CUcontext* pctx);
VGPU_EXPORT CUresult cuStreamGetCtx(CUstream a0, CUcontext* pctx) { return traced("cuStreamGetCtx", cuStreamGetCtx_impl, a0, pctx); }
static CUresult cuStreamGetCtx_impl(CUstream, CUcontext* pctx) {
  return api("cuStreamGetCtx", true, false, [&](ShimState&) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    if (ctx_stack().empty()) return CUDA_ERROR_INVALID_CONTEXT;
    *pctx = reinterpret_cast<CUcontext>(ctx_stack().back());
    return CUDA_SUCCESS;
  });
}
// The CUDA 12.5 form also reports a green context when the stream belongs to
// one. The simulator has none, so for every stream the answer is NULL, as an
// RTX 3080 Ti's driver answers for an ordinary stream.
static CUresult cuStreamGetCtx_v2_impl(CUstream stream, CUcontext* pctx, CUgreenCtx* pgreen);
VGPU_EXPORT CUresult cuStreamGetCtx_v2(CUstream stream, CUcontext* pctx, CUgreenCtx* pgreen) { return traced("cuStreamGetCtx_v2", cuStreamGetCtx_v2_impl, stream, pctx, pgreen); }
static CUresult cuStreamGetCtx_v2_impl(CUstream stream, CUcontext* pctx, CUgreenCtx* pgreen) {
  const CUresult r = cuStreamGetCtx(stream, pctx);
  if (r == CUDA_SUCCESS && pgreen) *pgreen = nullptr;
  return r;
}
// Capture is the runtime's (cudaStreamBeginCapture); the driver asks it by name,
// so a program that never loaded libcudart has nothing capturing.
static CUresult cuStreamGetCaptureInfo_v2_impl(CUstream stream, int* status, unsigned long long* id, void* graph, const void** deps, size_t* ndeps);
VGPU_EXPORT CUresult cuStreamGetCaptureInfo_v2(CUstream stream, int* status, unsigned long long* id, void* graph, const void** deps, size_t* ndeps) { return traced("cuStreamGetCaptureInfo_v2", cuStreamGetCaptureInfo_v2_impl, stream, status, id, graph, deps, ndeps); }
static CUresult cuStreamGetCaptureInfo_v2_impl(CUstream stream, int* status, unsigned long long* id,
                                               void* graph, const void** deps, size_t* ndeps) {
  using Fn = int (*)(void*, int*, unsigned long long*, void*, const void**, size_t*);
  auto fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "cudaStreamGetCaptureInfo_v2"));
  if (!fn) fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "cudaStreamGetCaptureInfo"));
  if (fn) {
    int st = 0;
    unsigned long long cid = 0;
    if (fn(stream, &st, &cid, graph, deps, ndeps) == 0) {
      if (status) *status = st;   // the runtime's cudaStreamCaptureStatus has the driver's values
      if (id) *id = cid;
      return CUDA_SUCCESS;
    }
  }
  if (status) *status = 0;  // CU_STREAM_CAPTURE_STATUS_NONE
  if (id) *id = 0;
  return CUDA_SUCCESS;
}
static CUresult cuStreamIsCapturing_impl(CUstream stream, int* status);
VGPU_EXPORT CUresult cuStreamIsCapturing(CUstream stream, int* status) { return traced("cuStreamIsCapturing", cuStreamIsCapturing_impl, stream, status); }
static CUresult cuStreamIsCapturing_impl(CUstream stream, int* status) {
  using Fn = int (*)(void*, int*);
  auto fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "cudaStreamIsCapturing"));
  int st = 0;
  if (fn && fn(stream, &st) != 0) st = 0;
  if (status) *status = st;
  return CUDA_SUCCESS;
}

static CUresult cuEventCreate_impl(void** ev, unsigned int flags) {
  return api("cuEventCreate", true, false, [&](ShimState& s) {
    if (!ev) return CUDA_ERROR_INVALID_VALUE;
    uintptr_t h = make_handle(s, kTagEvent);
    s.events[h] = {};
    s.events[h].timing = !(flags & 0x2);   // CU_EVENT_DISABLE_TIMING
    *ev = reinterpret_cast<void*>(h);
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuEventCreate(void** ev, unsigned int flags) {
  return traced("cuEventCreate", cuEventCreate_impl, ev, flags);
}
static CUresult cuEventRecord_impl(void* ev, CUstream stream) {
  return api("cuEventRecord", true, false, [&](ShimState& s) {
    auto it = s.events.find(reinterpret_cast<uintptr_t>(ev));
    if (it == s.events.end()) return CUDA_ERROR_INVALID_VALUE;
    it->second.recorded = true;
    clock_gettime(CLOCK_MONOTONIC, &it->second.when);
    if (vgpu::profiling::enabled()) {
      vgpu::profiling::Event e;
      e.kind = vgpu::profiling::EventKind::CudaEvent;
      e.start_ns = e.end_ns = vgpu::profiling::now_ns();
      e.device = static_cast<uint32_t>(current_device(s));
      e.correlation = vgpu::profiling::work_correlation();
      e.stream = reinterpret_cast<uint64_t>(stream);
      e.handle = reinterpret_cast<uint64_t>(ev);
      vgpu::profiling::record(std::move(e));
    }
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuEventRecord(void* ev, CUstream a1) {
  return traced("cuEventRecord", cuEventRecord_impl, ev, a1);
}
// Like a stream query, a query that finds the event done is recorded as a wait on it.
//
// A NULL event is CUDA_ERROR_INVALID_HANDLE, as on the card. (Events the runtime made are
// handles here too, so a handle this library never saw is not necessarily wrong.)
static CUresult event_exists(void* event) { return event ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE; }
static CUresult cuEventQuery_impl(void* event) {
  CUresult r = dead_context();
  if (r == CUDA_SUCCESS) r = event_exists(event);
  if (r == CUDA_SUCCESS)
    vgpu::profiling::SyncScope waited(vgpu::profiling::SyncKind::Event, 0, reinterpret_cast<uint64_t>(event),
                                      profiled_device());
  return r;
}
VGPU_EXPORT CUresult cuEventQuery(void* event) { return traced("cuEventQuery", cuEventQuery_impl, event); }
static CUresult cuEventSynchronize_impl(void* event) {
  CUresult r = dead_context();
  if (r == CUDA_SUCCESS) r = event_exists(event);
  if (r == CUDA_SUCCESS)
    vgpu::profiling::SyncScope waited(vgpu::profiling::SyncKind::Event, 0, reinterpret_cast<uint64_t>(event),
                                      profiled_device());
  return r;
}
VGPU_EXPORT CUresult cuEventSynchronize(void* event) {
  return traced("cuEventSynchronize", cuEventSynchronize_impl, event);
}
static CUresult cuEventDestroy_v2_impl(void* ev) {
  return api("cuEventDestroy", true, false, [&](ShimState& s) {
    s.events.erase(reinterpret_cast<uintptr_t>(ev));
    return CUDA_SUCCESS;
  });
}
VGPU_EXPORT CUresult cuEventDestroy_v2(void* ev) {
  return traced("cuEventDestroy_v2", cuEventDestroy_v2_impl, ev);
}
static CUresult cuEventDestroy_impl(void* ev);
VGPU_EXPORT CUresult cuEventDestroy(void* ev) { return traced("cuEventDestroy", cuEventDestroy_impl, ev); }
static CUresult cuEventDestroy_impl(void* ev) { return cuEventDestroy_v2_impl(ev); }
static CUresult cuEventElapsedTime_impl(float* ms, void* start, void* end) {
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
VGPU_EXPORT CUresult cuEventElapsedTime(float* ms, void* start, void* end) {
  return traced("cuEventElapsedTime", cuEventElapsedTime_impl, ms, start, end);
}
// Same story as cuCtxCreate: CUDA 13 renames this one too, and a program built
// there resolves the versioned name, not the plain one.
static CUresult cuEventElapsedTime_v2_impl(float* ms, void* start, void* end) {
  return cuEventElapsedTime_impl(ms, start, end);
}
VGPU_EXPORT CUresult cuEventElapsedTime_v2(float* ms, void* start, void* end) {
  return traced("cuEventElapsedTime_v2", cuEventElapsedTime_v2_impl, ms, start, end);
}

/* ---- context odds and ends static cudart asks about ---- */

static CUresult cuCtxPushCurrent_v2_impl(CUcontext ctx);
VGPU_EXPORT CUresult cuCtxPushCurrent_v2(CUcontext ctx) { return traced("cuCtxPushCurrent_v2", cuCtxPushCurrent_v2_impl, ctx); }
static CUresult cuCtxPushCurrent_v2_impl(CUcontext ctx) {
  return api("cuCtxPushCurrent", true, false, [&](ShimState& s) {
    uintptr_t h = check_handle(reinterpret_cast<uintptr_t>(ctx), kTagCtx, "context");
    if (!s.contexts.count(h)) return CUDA_ERROR_INVALID_CONTEXT;
    ctx_stack().push_back(h);
    return CUDA_SUCCESS;
  });
}
static CUresult cuCtxPushCurrent_impl(CUcontext ctx);
VGPU_EXPORT CUresult cuCtxPushCurrent(CUcontext ctx) { return traced("cuCtxPushCurrent", cuCtxPushCurrent_impl, ctx); }
static CUresult cuCtxPushCurrent_impl(CUcontext ctx) { return cuCtxPushCurrent_v2(ctx); }
static CUresult cuCtxPopCurrent_v2_impl(CUcontext* pctx);
VGPU_EXPORT CUresult cuCtxPopCurrent_v2(CUcontext* pctx) { return traced("cuCtxPopCurrent_v2", cuCtxPopCurrent_v2_impl, pctx); }
static CUresult cuCtxPopCurrent_v2_impl(CUcontext* pctx) {
  return api("cuCtxPopCurrent", true, false, [&](ShimState&) {
    if (ctx_stack().empty()) return CUDA_ERROR_INVALID_CONTEXT;
    if (pctx) *pctx = reinterpret_cast<CUcontext>(ctx_stack().back());
    ctx_stack().pop_back();
    return CUDA_SUCCESS;
  });
}
static CUresult cuCtxPopCurrent_impl(CUcontext* pctx);
VGPU_EXPORT CUresult cuCtxPopCurrent(CUcontext* pctx) { return traced("cuCtxPopCurrent", cuCtxPopCurrent_impl, pctx); }
static CUresult cuCtxPopCurrent_impl(CUcontext* pctx) { return cuCtxPopCurrent_v2(pctx); }

static CUresult cuCtxGetLimit_impl(size_t* v, int limit);
VGPU_EXPORT CUresult cuCtxGetLimit(size_t* v, int limit) { return traced("cuCtxGetLimit", cuCtxGetLimit_impl, v, limit); }
static CUresult cuCtxGetLimit_impl(size_t* v, int limit) {
  if (const CUresult dead = dead_context()) return dead;
  return api("cuCtxGetLimit", false, false, [&](ShimState& s) {
    // With no context current -- or no machine yet -- the card answers
    // CUDA_ERROR_INVALID_CONTEXT, as it does for every call that needs one.
    if (!s.initialized || ctx_stack().empty()) return CUDA_ERROR_INVALID_CONTEXT;
    if (!v) return CUDA_ERROR_INVALID_VALUE;
    const int device = current_device(s);
    return limit_value(s.limits[device], s.rt->device(device).profile(), limit, v);
  });
}
static CUresult cuCtxSetLimit_impl(int limit, size_t value);
VGPU_EXPORT CUresult cuCtxSetLimit(int limit, size_t value) { return traced("cuCtxSetLimit", cuCtxSetLimit_impl, limit, value); }
static CUresult cuCtxSetLimit_impl(int limit, size_t value) {
  if (const CUresult dead = dead_context()) return dead;
  return api("cuCtxSetLimit", false, false, [&](ShimState& s) {
    if (!s.initialized || ctx_stack().empty()) return CUDA_ERROR_INVALID_CONTEXT;
    const int device = current_device(s);
    ShimState::Limits& l = s.limits[device];
    // Measured on an RTX 3060 (driver 13.2): a limit above its largest is CUDA_ERROR_INVALID_VALUE
    // where the stack (more than the 512 KiB of local memory a thread has), the device runtime's
    // synchronization depth (24) and the L2 fetch granularity (128) are concerned; the printf FIFO
    // and the malloc heap are raised to their smallest instead; and the shared memory size and
    // CIG limits (7 and 8) can be read and not set (CUDA_ERROR_NOT_PERMITTED).
    switch (limit) {
      case 0:
        if (value > (512u << 10)) return CUDA_ERROR_INVALID_VALUE;
        l.stack = (value + 15) / 16 * 16;   // a whole element
        return CUDA_SUCCESS;
      case 1: l.printf_fifo = std::max<size_t>(value, 393216); return CUDA_SUCCESS;
      case 2: l.malloc_heap = std::max<size_t>(value, 4u << 20); return CUDA_SUCCESS;
      case 3:
        if (s.rt->device(device).profile().cc_major >= 9) return kUnsupportedLimit;
        if (value > 24) return CUDA_ERROR_INVALID_VALUE;
        l.sync_depth = value;
        return CUDA_SUCCESS;
      case 4: l.pending_launches = std::max<size_t>(value, 32); return CUDA_SUCCESS;
      case 5:
        if (value > 128) return CUDA_ERROR_INVALID_VALUE;
        l.l2_fetch_granularity = value;
        return CUDA_SUCCESS;
      case 6: l.persisting_l2 = 0; return CUDA_SUCCESS;   // nothing to set aside
      case 7:
      case 8: return static_cast<CUresult>(800);   // CUDA_ERROR_NOT_PERMITTED
      case 9: return kUnsupportedLimit;
      default: return CUDA_ERROR_INVALID_VALUE;
    }
  });
}
// The context an entry point means: the one named, or the thread's current one for NULL.
// CUDA_ERROR_INVALID_CONTEXT for none, or for one that does not exist.
static CUresult named_or_current_context(ShimState& s, CUcontext ctx, uintptr_t* out) {
  if (ctx) {
    const uintptr_t h = reinterpret_cast<uintptr_t>(ctx);
    if (!s.contexts.count(h)) return CUDA_ERROR_INVALID_CONTEXT;
    *out = h;
    return CUDA_SUCCESS;
  }
  if (ctx_stack().empty() || !s.contexts.count(ctx_stack().back())) return CUDA_ERROR_INVALID_CONTEXT;
  *out = ctx_stack().back();
  return CUDA_SUCCESS;
}
// 3020 is what an RTX 3060's driver 13.2 answers for any context.
static CUresult cuCtxGetApiVersion_impl(CUcontext ctx, unsigned int* v);
VGPU_EXPORT CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int* v) { return traced("cuCtxGetApiVersion", cuCtxGetApiVersion_impl, ctx, v); }
static CUresult cuCtxGetApiVersion_impl(CUcontext ctx, unsigned int* v) {
  return api("cuCtxGetApiVersion", true, false, [&](ShimState& s) {
    uintptr_t h = 0;
    if (const CUresult r = named_or_current_context(s, ctx, &h); r != CUDA_SUCCESS) return r;
    if (!v) return CUDA_ERROR_INVALID_VALUE;
    *v = 3020;
    return CUDA_SUCCESS;
  });
}
static CUresult cuCtxGetStreamPriorityRange_impl(int* least, int* greatest);
VGPU_EXPORT CUresult cuCtxGetStreamPriorityRange(int* least, int* greatest) { return traced("cuCtxGetStreamPriorityRange", cuCtxGetStreamPriorityRange_impl, least, greatest); }
static CUresult cuCtxGetStreamPriorityRange_impl(int* least, int* greatest) {
  return api("cuCtxGetStreamPriorityRange", true, false, [&](ShimState&) {
    if (least) *least = vgpu::cuda::kLeastStreamPriority;
    if (greatest) *greatest = vgpu::cuda::kGreatestStreamPriority;
    return CUDA_SUCCESS;
  });
}
// The flags the current context reports: as it was created (cuCtxCreate), or, for a primary one,
// the flags set for the device with MAP_HOST always on; cuCtxSetFlags changes the scheduling
// mode, LMEM_RESIZE_TO_MAX and SYNC_MEMOPS (measured).
static CUresult cuCtxGetFlags_impl(unsigned int* f);
VGPU_EXPORT CUresult cuCtxGetFlags(unsigned int* f) { return traced("cuCtxGetFlags", cuCtxGetFlags_impl, f); }
static CUresult cuCtxGetFlags_impl(unsigned int* f) {
  return api("cuCtxGetFlags", true, false, [&](ShimState& s) {
    uintptr_t h = 0;
    if (const CUresult r = named_or_current_context(s, nullptr, &h); r != CUDA_SUCCESS) return r;
    if (!f) return CUDA_ERROR_INVALID_VALUE;
    *f = s.ctx_state[h].flags;
    return CUDA_SUCCESS;
  });
}
static CUresult cuCtxSetFlags_impl(unsigned int flags);
VGPU_EXPORT CUresult cuCtxSetFlags(unsigned int flags) { return traced("cuCtxSetFlags", cuCtxSetFlags_impl, flags); }
static CUresult cuCtxSetFlags_impl(unsigned int flags) {
  return api("cuCtxSetFlags", true, false, [&](ShimState& s) {
    uintptr_t h = 0;
    if (const CUresult r = named_or_current_context(s, nullptr, &h); r != CUDA_SUCCESS) return r;
    if (!ctx_flags_valid(flags)) return CUDA_ERROR_INVALID_VALUE;
    ShimState::CtxState& c = s.ctx_state[h];
    c.flags = (flags & kCtxSettableMask) | (c.flags & kCtxMapHost);
    return CUDA_SUCCESS;
  });
}
// Contexts are numbered from 1 in the order they are made, the primary ones among them.
static CUresult cuCtxGetId_impl(CUcontext ctx, unsigned long long* id);
VGPU_EXPORT CUresult cuCtxGetId(CUcontext ctx, unsigned long long* id) { return traced("cuCtxGetId", cuCtxGetId_impl, ctx, id); }
static CUresult cuCtxGetId_impl(CUcontext ctx, unsigned long long* id) {
  return api("cuCtxGetId", true, false, [&](ShimState& s) {
    uintptr_t h = 0;
    if (const CUresult r = named_or_current_context(s, ctx, &h); r != CUDA_SUCCESS) return r;
    if (!id) return CUDA_ERROR_INVALID_VALUE;
    *id = s.ctx_state[h].id;
    return CUDA_SUCCESS;
  });
}

// The flags a device's primary context is made with. Measured on an RTX 3060: a scheduling mode
// (one of 0, 1, 2, 4) with any of LMEM_RESIZE_TO_MAX, the two core-dump flags and SYNC_MEMOPS; MAP_HOST
// is refused because a primary context always has it, and so is anything above 0xff. Setting them
// on a context that is active is accepted and changes what it reports.
static CUresult primary_set_flags(CUdevice dev, unsigned int flags) {
  return api("cuDevicePrimaryCtxSetFlags", true, false, [&](ShimState& s) {
    check_device(s, dev);
    if (!ctx_flags_valid(flags) || (flags & kCtxMapHost)) return CUDA_ERROR_INVALID_VALUE;
    s.primary_flags[dev] = flags;
    if (const auto it = s.primary_ctx.find(dev); it != s.primary_ctx.end())
      s.ctx_state[it->second].flags = (flags & kCtxSettableMask) | kCtxMapHost;
    return CUDA_SUCCESS;
  });
}
static CUresult cuDevicePrimaryCtxSetFlags_v2_impl(CUdevice dev, unsigned int flags);
VGPU_EXPORT CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) { return traced("cuDevicePrimaryCtxSetFlags_v2", cuDevicePrimaryCtxSetFlags_v2_impl, dev, flags); }
static CUresult cuDevicePrimaryCtxSetFlags_v2_impl(CUdevice dev, unsigned int flags) { return primary_set_flags(dev, flags); }
static CUresult cuDevicePrimaryCtxSetFlags_impl(CUdevice dev, unsigned int flags);
VGPU_EXPORT CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) { return traced("cuDevicePrimaryCtxSetFlags", cuDevicePrimaryCtxSetFlags_impl, dev, flags); }
static CUresult cuDevicePrimaryCtxSetFlags_impl(CUdevice dev, unsigned int flags) { return primary_set_flags(dev, flags); }
static CUresult cuDevicePrimaryCtxGetState_impl(CUdevice dev, unsigned int* flags, int* active);
VGPU_EXPORT CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) { return traced("cuDevicePrimaryCtxGetState", cuDevicePrimaryCtxGetState_impl, dev, flags, active); }
static CUresult cuDevicePrimaryCtxGetState_impl(CUdevice dev, unsigned int* flags, int* active) {
  return api("cuDevicePrimaryCtxGetState", true, false, [&](ShimState& s) {
    check_device(s, dev);
    if (!flags || !active) return CUDA_ERROR_INVALID_VALUE;
    *flags = s.primary_flags[dev];
    *active = s.primary_active.count(dev) ? 1 : 0;
    return CUDA_SUCCESS;
  });
}
static CUresult cuDevicePrimaryCtxReset_v2_impl(CUdevice dev);
VGPU_EXPORT CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) { return traced("cuDevicePrimaryCtxReset_v2", cuDevicePrimaryCtxReset_v2_impl, dev); }
static CUresult cuDevicePrimaryCtxReset_v2_impl(CUdevice dev) { return cuDevicePrimaryCtxReset(dev); }

static CUresult cuModuleGetLoadingMode_impl(int* mode);
VGPU_EXPORT CUresult cuModuleGetLoadingMode(int* mode) { return traced("cuModuleGetLoadingMode", cuModuleGetLoadingMode_impl, mode); }
static CUresult cuModuleGetLoadingMode_impl(int* mode) {
  if (!mode) return CUDA_ERROR_INVALID_VALUE;
  *mode = 1;  // CU_MODULE_EAGER_LOADING
  return CUDA_SUCCESS;
}

static CUresult cuDeviceGetUuid_impl(void* uuid, CUdevice dev);
VGPU_EXPORT CUresult cuDeviceGetUuid(void* uuid, CUdevice dev) { return traced("cuDeviceGetUuid", cuDeviceGetUuid_impl, uuid, dev); }
static CUresult cuDeviceGetUuid_impl(void* uuid, CUdevice dev) {
  return api("cuDeviceGetUuid", true, false, [&](ShimState& s) {
    if (!uuid) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    // The bytes of the "GPU-" UUID NVML and nvidia-smi print for the device,
    // as on a card: CUDA_VISIBLE_DEVICES and every tool that correlates the
    // two names a device by it.
    const vgpu::DeviceProfile& p = s.rt->device(dev).profile();
    std::memcpy(uuid, vgpu::cuda::identity(p, s.rt->device(dev).physical()).uuid, 16);
    return CUDA_SUCCESS;
  });
}
static CUresult cuDeviceGetUuid_v2_impl(void* uuid, CUdevice dev);
VGPU_EXPORT CUresult cuDeviceGetUuid_v2(void* uuid, CUdevice dev) { return traced("cuDeviceGetUuid_v2", cuDeviceGetUuid_v2_impl, uuid, dev); }
static CUresult cuDeviceGetUuid_v2_impl(void* uuid, CUdevice dev) { return cuDeviceGetUuid(uuid, dev); }

// A buffer too short for the id gets as much of it as fits, and
// INVALID_VALUE, as on the card.
static CUresult cuDeviceGetPCIBusId_impl(char* id, int len, CUdevice dev);
VGPU_EXPORT CUresult cuDeviceGetPCIBusId(char* id, int len, CUdevice dev) { return traced("cuDeviceGetPCIBusId", cuDeviceGetPCIBusId_impl, id, len, dev); }
static CUresult cuDeviceGetPCIBusId_impl(char* id, int len, CUdevice dev) {
  return api("cuDeviceGetPCIBusId", true, false, [&](ShimState& s) {
    if (!id || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    check_device(s, dev);
    // The address NVML reports for the device (its domain without the zeros
    // NVML pads it with), so a tool correlating the two finds one device.
    char text[32];
    vgpu::cuda::pci_bus_id(vgpu::cuda::identity(s.rt->device(dev).profile(), s.rt->device(dev).physical()), text,
                           sizeof text);
    const int n = std::snprintf(id, static_cast<size_t>(len), "%s", text);
    return n < len ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
  });
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
static CUresult cuDeviceGetByPCIBusId_impl(CUdevice* dev, const char* pciBusId);
VGPU_EXPORT CUresult cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId) { return traced("cuDeviceGetByPCIBusId", cuDeviceGetByPCIBusId_impl, dev, pciBusId); }
static CUresult cuDeviceGetByPCIBusId_impl(CUdevice* dev, const char* pciBusId) {
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
static CUresult cuDeviceCanAccessPeer_impl(int* can, CUdevice dev, CUdevice peer);
VGPU_EXPORT CUresult cuDeviceCanAccessPeer(int* can, CUdevice dev, CUdevice peer) { return traced("cuDeviceCanAccessPeer", cuDeviceCanAccessPeer_impl, can, dev, peer); }
static CUresult cuDeviceCanAccessPeer_impl(int* can, CUdevice dev, CUdevice peer) {
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
static CUresult cuCtxEnablePeerAccess_impl(CUcontext peerContext, unsigned int flags);
VGPU_EXPORT CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int flags) { return traced("cuCtxEnablePeerAccess", cuCtxEnablePeerAccess_impl, peerContext, flags); }
static CUresult cuCtxEnablePeerAccess_impl(CUcontext peerContext, unsigned int flags) {
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
static CUresult cuCtxDisablePeerAccess_impl(CUcontext peerContext);
VGPU_EXPORT CUresult cuCtxDisablePeerAccess(CUcontext peerContext) { return traced("cuCtxDisablePeerAccess", cuCtxDisablePeerAccess_impl, peerContext); }
static CUresult cuCtxDisablePeerAccess_impl(CUcontext peerContext) {
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
static CUresult cuMemcpyPeer_impl(CUdeviceptr dst, CUcontext, CUdeviceptr src, CUcontext, size_t n);
VGPU_EXPORT CUresult cuMemcpyPeer(CUdeviceptr dst, CUcontext a1, CUdeviceptr src, CUcontext a3, size_t n) { return traced("cuMemcpyPeer", cuMemcpyPeer_impl, dst, a1, src, a3, n); }
static CUresult cuMemcpyPeer_impl(CUdeviceptr dst, CUcontext, CUdeviceptr src, CUcontext, size_t n) {
  return cuMemcpyDtoD_v2_impl(dst, src, n);
}
static CUresult cuMemcpyPeerAsync_impl(CUdeviceptr dst, CUcontext dctx, CUdeviceptr src, CUcontext sctx, size_t n, CUstream);
VGPU_EXPORT CUresult cuMemcpyPeerAsync(CUdeviceptr dst, CUcontext dctx, CUdeviceptr src, CUcontext sctx, size_t n, CUstream a5) { return traced("cuMemcpyPeerAsync", cuMemcpyPeerAsync_impl, dst, dctx, src, sctx, n, a5); }
static CUresult cuMemcpyPeerAsync_impl(CUdeviceptr dst, CUcontext dctx, CUdeviceptr src, CUcontext sctx,
                                       size_t n, CUstream) {
  return cuMemcpyPeer(dst, dctx, src, sctx, n);
}

/* ---- pointer queries (expected probes: fail quietly, no stderr) ---- */

/* ---- what a pointer is ----
 * cuPointerGetAttribute, cuPointerGetAttributes and cuPointerSetAttribute. A
 * pointer here is device memory (a live allocation in one device's window),
 * managed memory, pinned host memory, or registered host memory; anything else
 * is not a CUDA pointer, which these answer as the documentation says. */

namespace {

enum class PtrKind { None, Device, Managed, Pinned, Registered };
struct PtrInfo {
  PtrKind kind = PtrKind::None;
  int device = 0;
  uint64_t base = 0, size = 0;   // the allocation holding the pointer
};

PtrInfo describe_pointer(ShimState& s, CUdeviceptr ptr) {
  PtrInfo pi;
  if (const auto* m = managed_range(s, ptr, 1)) {
    pi.kind = PtrKind::Managed;
    pi.base = m->first;
    pi.size = m->second;
    const auto d = s.managed_device.find(m->first);
    pi.device = d == s.managed_device.end() ? 0 : d->second;
    return pi;
  }
  void* base = nullptr;
  if (const auto* r = host_range_at(pinned(s), reinterpret_cast<void*>(ptr), &base)) {
    pi = {PtrKind::Pinned, r->device, reinterpret_cast<uint64_t>(base), r->size};
    return pi;
  }
  if (const auto* r = host_range_at(registrations(s), reinterpret_cast<void*>(ptr), &base)) {
    pi = {PtrKind::Registered, r->device, reinterpret_cast<uint64_t>(base), r->size};
    return pi;
  }
  // An address in a device window is a device pointer only while something is
  // allocated there; a freed one, or one nothing ever was, is not a pointer.
  if (vgpu::is_device_va(ptr))
    for (int d = 0; d < s.rt->device_count(); ++d) {
      vgpu::MemoryManager& mem = s.rt->device(d).memory();
      uint64_t b = 0, sz = 0;
      if (mem.owns(ptr) && !mem.heap_contains(ptr) && mem.find_allocation(ptr, &b, &sz)) {
        pi = {PtrKind::Device, d, b, sz};
        return pi;
      }
    }
  return pi;
}

// The context a pointer on `device` was allocated in. The allocation does not
// record one, so this is the current context when it is on that device and the
// device's primary context otherwise -- which is the one, in a program that
// uses a single context per device, it was made in. 0 when there is neither.
uintptr_t pointer_context(ShimState& s, int device) {
  if (!ctx_stack().empty()) {
    const auto it = s.contexts.find(ctx_stack().back());
    if (it != s.contexts.end() && it->second == device) return ctx_stack().back();
  }
  const auto p = s.primary_ctx.find(device);
  return p == s.primary_ctx.end() ? 0 : p->second;
}

// Bytes a CUpointer_attribute's value occupies, 0 for one that is unknown.
size_t pointer_attribute_size(int attribute) {
  switch (attribute) {
    case 1: return sizeof(void*);              // CONTEXT
    case 2: return sizeof(unsigned int);       // MEMORY_TYPE
    case 3: return sizeof(CUdeviceptr);        // DEVICE_POINTER
    case 4: return sizeof(void*);              // HOST_POINTER
    case 6: return sizeof(int);                // SYNC_MEMOPS
    case 7: return sizeof(unsigned long long); // BUFFER_ID
    case 8: return sizeof(int);                // IS_MANAGED
    case 9: return sizeof(int);                // DEVICE_ORDINAL
    case 10: return sizeof(int);               // IS_LEGACY_CUDA_IPC_CAPABLE
    case 11: return sizeof(void*);             // RANGE_START_ADDR
    case 12: return sizeof(size_t);            // RANGE_SIZE
    case 13: return sizeof(int);               // MAPPED
    case 14: return sizeof(int);               // ALLOWED_HANDLE_TYPES
    case 17: return sizeof(void*);             // MEMPOOL_HANDLE
    default: return 0;
  }
}

// One attribute of a pointer that is a CUDA pointer. The answers for managed
// memory that an RTX 3080 Ti's driver gives -- device memory, the same address
// on both sides -- are kept as they were.
CUresult pointer_attribute(ShimState& s, int attribute, CUdeviceptr ptr, const PtrInfo& pi, void* data) {
  const bool managed = pi.kind == PtrKind::Managed;
  const bool host = pi.kind == PtrKind::Pinned || pi.kind == PtrKind::Registered;
  switch (attribute) {
    case 1:   // CONTEXT: the context the memory was allocated or registered in
      *static_cast<void**>(data) = reinterpret_cast<void*>(pointer_context(s, pi.device));
      return CUDA_SUCCESS;
    case 2:   // MEMORY_TYPE: CU_MEMORYTYPE_HOST (1) for host memory, DEVICE (2) otherwise
      *static_cast<unsigned int*>(data) = host ? 1 : 2;
      return CUDA_SUCCESS;
    case 3:   // DEVICE_POINTER: the address kernels reach it by, which under UVA is its own
      *static_cast<CUdeviceptr*>(data) = ptr;
      return CUDA_SUCCESS;
    case 4:   // HOST_POINTER: none for device memory, which the host cannot address
      if (pi.kind == PtrKind::Device) return CUDA_ERROR_INVALID_VALUE;
      *static_cast<void**>(data) = reinterpret_cast<void*>(ptr);
      return CUDA_SUCCESS;
    case 6:   // SYNC_MEMOPS
      *static_cast<int*>(data) = s.sync_memops.count(pi.base) ? 1 : 0;
      return CUDA_SUCCESS;
    case 7:   // BUFFER_ID: unique over the process and never reused. A device allocation's
              // base is both, as the address space of a device is handed out once.
      if (pi.kind != PtrKind::Device) {
        report("cuPointerGetAttribute", "CU_POINTER_ATTRIBUTE_BUFFER_ID is implemented for device "
                                        "memory only");
        return CUDA_ERROR_INVALID_VALUE;
      }
      *static_cast<unsigned long long*>(data) = pi.base;
      return CUDA_SUCCESS;
    case 8:   // IS_MANAGED
      *static_cast<int*>(data) = managed ? 1 : 0;
      return CUDA_SUCCESS;
    case 9:   // DEVICE_ORDINAL: the device the memory was allocated or registered against
      *static_cast<int*>(data) = pi.device;
      return CUDA_SUCCESS;
    case 11:  // RANGE_START_ADDR
      *static_cast<void**>(data) = reinterpret_cast<void*>(pi.base);
      return CUDA_SUCCESS;
    case 12:  // RANGE_SIZE
      *static_cast<size_t*>(data) = pi.size;
      return CUDA_SUCCESS;
    case 13:  // MAPPED: in a valid address range with something behind it
      *static_cast<int*>(data) = 1;
      return CUDA_SUCCESS;
    default:
      report("cuPointerGetAttribute", "attribute " + std::to_string(attribute) +
                                          " is not implemented (P2P tokens, legacy IPC capability, "
                                          "allowed handle types and pool handle are not modelled)");
      return CUDA_ERROR_INVALID_VALUE;
  }
}

}  // namespace

static CUresult cuPointerGetAttribute_impl(void* data, int attribute, CUdeviceptr ptr);
VGPU_EXPORT CUresult cuPointerGetAttribute(void* data, int attribute, CUdeviceptr ptr) { return traced("cuPointerGetAttribute", cuPointerGetAttribute_impl, data, attribute, ptr); }
static CUresult cuPointerGetAttribute_impl(void* data, int attribute, CUdeviceptr ptr) {
  if (!data) return CUDA_ERROR_INVALID_VALUE;
  return api("cuPointerGetAttribute", true, false, [&](ShimState& s) {
    const PtrInfo pi = describe_pointer(s, ptr);
    // Not a pointer CUDA knows: INVALID_VALUE, as documented.
    if (pi.kind == PtrKind::None) return CUDA_ERROR_INVALID_VALUE;
    return pointer_attribute(s, attribute, ptr, pi, data);
  });
}

// Several attributes at once. Unlike the single query this is not an error for
// a pointer CUDA does not know: each value is set to its NULL default (zero
// bytes) and the call succeeds, which is how a caller asks "is this one of
// yours?" of an arbitrary pointer.
static CUresult cuPointerGetAttributes_impl(unsigned int numAttributes, int* attributes, void** data, CUdeviceptr ptr);
VGPU_EXPORT CUresult cuPointerGetAttributes(unsigned int numAttributes, int* attributes, void** data, CUdeviceptr ptr) { return traced("cuPointerGetAttributes", cuPointerGetAttributes_impl, numAttributes, attributes, data, ptr); }
static CUresult cuPointerGetAttributes_impl(unsigned int numAttributes, int* attributes, void** data,
                                            CUdeviceptr ptr) {
  if (numAttributes == 0) return CUDA_SUCCESS;
  if (!attributes || !data) return CUDA_ERROR_INVALID_VALUE;
  return api("cuPointerGetAttributes", true, false, [&](ShimState& s) {
    const PtrInfo pi = describe_pointer(s, ptr);
    for (unsigned i = 0; i < numAttributes; ++i) {
      const size_t size = pointer_attribute_size(attributes[i]);
      if (!data[i] || size == 0) return CUDA_ERROR_INVALID_VALUE;
      if (pi.kind == PtrKind::None) {
        std::memset(data[i], 0, size);
        continue;
      }
      if (const CUresult r = pointer_attribute(s, attributes[i], ptr, pi, data[i]); r != CUDA_SUCCESS) return r;
    }
    return CUDA_SUCCESS;
  });
}

// The one attribute that can be set is SYNC_MEMOPS, a boolean. Every operation
// here is already synchronous, so setting it changes nothing a program can
// see but what a query of it reads back.
static CUresult cuPointerSetAttribute_impl(const void* value, int attribute, CUdeviceptr ptr);
VGPU_EXPORT CUresult cuPointerSetAttribute(const void* value, int attribute, CUdeviceptr ptr) { return traced("cuPointerSetAttribute", cuPointerSetAttribute_impl, value, attribute, ptr); }
static CUresult cuPointerSetAttribute_impl(const void* value, int attribute, CUdeviceptr ptr) {
  if (!value) return CUDA_ERROR_INVALID_VALUE;
  return api("cuPointerSetAttribute", true, false, [&](ShimState& s) {
    if (attribute != 6) return CUDA_ERROR_INVALID_VALUE;   // CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
    const PtrInfo pi = describe_pointer(s, ptr);
    if (pi.kind == PtrKind::None) return CUDA_ERROR_INVALID_VALUE;
    const unsigned int v = *static_cast<const unsigned int*>(value);
    if (v > 1) return CUDA_ERROR_INVALID_VALUE;   // a boolean: set (1) or unset (0)
    if (v) s.sync_memops.insert(pi.base);
    else s.sync_memops.erase(pi.base);
    return CUDA_SUCCESS;
  });
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

static CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor_impl(int* num, CUfunction f, int blockSize, size_t dyn);
VGPU_EXPORT CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* num, CUfunction f, int blockSize, size_t dyn) { return traced("cuOccupancyMaxActiveBlocksPerMultiprocessor", cuOccupancyMaxActiveBlocksPerMultiprocessor_impl, num, f, blockSize, dyn); }
static CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor_impl(int* num, CUfunction f, int blockSize,
                                                                 size_t dyn) {
  return api("cuOccupancyMaxActiveBlocksPerMultiprocessor", true, false, [&](ShimState& s) {
    if (!num || blockSize <= 0) return CUDA_ERROR_INVALID_VALUE;
    const FuncRec* rec = occupancy_func(s, f);
    if (!rec) return CUDA_ERROR_INVALID_HANDLE;
    *num = blocks_per_sm(s, *rec, blockSize, dyn);
    return CUDA_SUCCESS;
  });
}
static CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_impl(int* num, CUfunction f, int blockSize, size_t dyn, unsigned int);
VGPU_EXPORT CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* num, CUfunction f, int blockSize, size_t dyn, unsigned int a4) { return traced("cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_impl, num, f, blockSize, dyn, a4); }
static CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_impl(int* num, CUfunction f,
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
static CUresult cuOccupancyMaxPotentialBlockSizeWithFlags_impl(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockToSmem, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags);
VGPU_EXPORT CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockToSmem, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags) { return traced("cuOccupancyMaxPotentialBlockSizeWithFlags", cuOccupancyMaxPotentialBlockSizeWithFlags_impl, minGridSize, blockSize, func, blockToSmem, dynamicSMemSize, blockSizeLimit, flags); }
static CUresult cuOccupancyMaxPotentialBlockSizeWithFlags_impl(int* minGridSize, int* blockSize,
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
static CUresult cuOccupancyMaxPotentialBlockSize_impl(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockToSmem, size_t dynamicSMemSize, int blockSizeLimit);
VGPU_EXPORT CUresult cuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockToSmem, size_t dynamicSMemSize, int blockSizeLimit) { return traced("cuOccupancyMaxPotentialBlockSize", cuOccupancyMaxPotentialBlockSize_impl, minGridSize, blockSize, func, blockToSmem, dynamicSMemSize, blockSizeLimit); }
static CUresult cuOccupancyMaxPotentialBlockSize_impl(int* minGridSize, int* blockSize, CUfunction func,
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
  return cuModuleLoadData_impl(mod, wrapper);
}
CUresult dark_get_module_ext1(CUmodule* mod, const void* wrapper, void*, void*, uintptr_t) {
  return cuModuleLoadData_impl(mod, wrapper);
}
CUresult dark_get_module_ext2(const void* wrapper, CUmodule* mod, void*, void*, unsigned int) {
  return cuModuleLoadData_impl(mod, wrapper);
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

static CUresult cuIpcGetMemHandle_impl(CUipcMemHandle* handle, CUdeviceptr ptr);
VGPU_EXPORT CUresult cuIpcGetMemHandle(CUipcMemHandle* handle, CUdeviceptr ptr) { return traced("cuIpcGetMemHandle", cuIpcGetMemHandle_impl, handle, ptr); }
static CUresult cuIpcGetMemHandle_impl(CUipcMemHandle* handle, CUdeviceptr ptr) {
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

static CUresult cuIpcOpenMemHandle_v2_impl(CUdeviceptr* ptr, CUipcMemHandle handle, unsigned int flags);
VGPU_EXPORT CUresult cuIpcOpenMemHandle_v2(CUdeviceptr* ptr, CUipcMemHandle handle, unsigned int flags) { return traced("cuIpcOpenMemHandle_v2", cuIpcOpenMemHandle_v2_impl, ptr, handle, flags); }
static CUresult cuIpcOpenMemHandle_v2_impl(CUdeviceptr* ptr, CUipcMemHandle handle,
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

static CUresult cuIpcOpenMemHandle_impl(CUdeviceptr* ptr, CUipcMemHandle handle, unsigned int flags);
VGPU_EXPORT CUresult cuIpcOpenMemHandle(CUdeviceptr* ptr, CUipcMemHandle handle, unsigned int flags) { return traced("cuIpcOpenMemHandle", cuIpcOpenMemHandle_impl, ptr, handle, flags); }
static CUresult cuIpcOpenMemHandle_impl(CUdeviceptr* ptr, CUipcMemHandle handle,
                                        unsigned int flags) {
  return cuIpcOpenMemHandle_v2(ptr, handle, flags);
}

static CUresult cuIpcCloseMemHandle_impl(CUdeviceptr ptr);
VGPU_EXPORT CUresult cuIpcCloseMemHandle(CUdeviceptr ptr) { return traced("cuIpcCloseMemHandle", cuIpcCloseMemHandle_impl, ptr); }
static CUresult cuIpcCloseMemHandle_impl(CUdeviceptr ptr) {
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
static CUresult cuIpcGetEventHandle_impl(CUipcEventHandle* handle, CUevent ev);
VGPU_EXPORT CUresult cuIpcGetEventHandle(CUipcEventHandle* handle, CUevent ev) { return traced("cuIpcGetEventHandle", cuIpcGetEventHandle_impl, handle, ev); }
static CUresult cuIpcGetEventHandle_impl(CUipcEventHandle* handle, CUevent ev) {
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

static CUresult cuIpcOpenEventHandle_impl(CUevent* ev, CUipcEventHandle handle);
VGPU_EXPORT CUresult cuIpcOpenEventHandle(CUevent* ev, CUipcEventHandle handle) { return traced("cuIpcOpenEventHandle", cuIpcOpenEventHandle_impl, ev, handle); }
static CUresult cuIpcOpenEventHandle_impl(CUevent* ev, CUipcEventHandle handle) {
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

static CUresult cuGetExportTable_impl(const void** table, const void* uuid);
VGPU_EXPORT CUresult cuGetExportTable(const void** table, const void* uuid) { return traced("cuGetExportTable", cuGetExportTable_impl, table, uuid); }
static CUresult cuGetExportTable_impl(const void** table, const void* uuid) {
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
    VGPU_PROC(cuDeviceGetExecAffinitySupport), VGPU_PROC(cuCtxGetExecAffinity),
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
    VGPU_PROC(cuMemAllocAsync), VGPU_PROC(cuMemFreeAsync), VGPU_PROC(cuMemAllocFromPoolAsync),
    VGPU_PROC(cuDeviceGetDefaultMemPool), VGPU_PROC(cuDeviceGetMemPool), VGPU_PROC(cuDeviceSetMemPool),
    VGPU_PROC(cuMemPoolCreate), VGPU_PROC(cuMemPoolDestroy), VGPU_PROC(cuMemPoolTrimTo),
    VGPU_PROC(cuMemPoolSetAttribute), VGPU_PROC(cuMemPoolGetAttribute), VGPU_PROC(cuMemPoolSetAccess),
    VGPU_PROC(cuMemPoolGetAccess), VGPU_PROC(cuMemPoolExportToShareableHandle),
    VGPU_PROC(cuMemPoolImportFromShareableHandle), VGPU_PROC(cuMemPoolExportPointer),
    VGPU_PROC(cuMemPoolImportPointer),
    VGPU_PROC(cuMemAllocPitch), VGPU_PROC(cuMemAllocPitch_v2), VGPU_PROC(cuModuleGetGlobal),
    VGPU_PROC(cuModuleGetGlobal_v2), VGPU_PROC(cuCtxSetCacheConfig), VGPU_PROC(cuCtxGetCacheConfig),
    VGPU_PROC(cuCtxSetSharedMemConfig), VGPU_PROC(cuCtxGetSharedMemConfig),
    VGPU_PROC(cuFuncSetSharedMemConfig), VGPU_PROC(cuStreamAddCallback),
    VGPU_PROC(cuPointerGetAttribute), VGPU_PROC(cuPointerGetAttributes), VGPU_PROC(cuPointerSetAttribute),
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
    VGPU_PROC(cuStreamGetPriority), VGPU_PROC(cuStreamGetFlags), VGPU_PROC(cuStreamGetCtx),
    VGPU_PROC(cuStreamGetCtx_v2),
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

static CUresult cuGetProcAddress_v2_impl(const char* symbol, void** pfn, int cudaVersion, unsigned long long flags, int* symbolStatus);
VGPU_EXPORT CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, unsigned long long flags, int* symbolStatus) { return traced("cuGetProcAddress_v2", cuGetProcAddress_v2_impl, symbol, pfn, cudaVersion, flags, symbolStatus); }
static CUresult cuGetProcAddress_v2_impl(const char* symbol, void** pfn, int cudaVersion,
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

static CUresult cuGetProcAddress_impl(const char* symbol, void** pfn, int cudaVersion, unsigned long long flags);
VGPU_EXPORT CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, unsigned long long flags) { return traced("cuGetProcAddress", cuGetProcAddress_impl, symbol, pfn, cudaVersion, flags); }
static CUresult cuGetProcAddress_impl(const char* symbol, void** pfn, int cudaVersion,
                                      unsigned long long flags) {
  return cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, nullptr);
}
