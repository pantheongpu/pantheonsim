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
#include <algorithm>
#include <cstring>
#include <functional>
#include <dlfcn.h>
#include <atomic>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "error_names.hpp"
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
#include "vgpu/driver_version.hpp"
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
// cudaDriverGetVersion reports the driver's version, not the runtime's: see
// vgpu::driver_version(). It used to return CUDART_VERSION, which is the
// toolkit this shim was built with and says nothing about the driver.

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

// A __device__ or __constant__ variable. The host handle nvcc passes to
// cudaMemcpyToSymbol is the address of a *host* shadow object, never a device
// pointer, so the only way to reach the device copy is to remember what that
// handle was registered as.
struct VarInfo {
  RegisteredModule* mod = nullptr;
  std::string device_name;
  size_t size = 0;
  bool is_constant = false;
};

// A host buffer the runtime allocated or was handed, and the device that was
// current when that happened -- which is the device a cudaDeviceReset of that
// device releases it with.
struct HostRange {
  size_t size = 0;
  int device = 0;
};

// What cudaMemAdvise was told about which bytes, so cudaMemRangeGetAttribute
// can answer. A value covers a range; a query over a range is answered only
// when every byte of it agrees, which is what CUDA reports.
//
// CUDA applies advice by page, so a hint about part of a page reaches the whole
// page. This records exactly the bytes it was given, which differs only for a
// range that is not page-aligned -- and differs by being more precise about
// what was asked for.
class RangeValues {
 public:
  void set(uint64_t begin, uint64_t end, int value) {
    clear(begin, end);
    if (begin < end) spans_[begin] = {end, value};
    merge();
  }
  void clear(uint64_t begin, uint64_t end) {
    if (begin >= end) return;
    // Split or trim every span that overlaps [begin, end).
    std::map<uint64_t, std::pair<uint64_t, int>> out;
    for (const auto& [b, ev] : spans_) {
      const auto [e, v] = ev;
      if (e <= begin || b >= end) { out[b] = {e, v}; continue; }
      if (b < begin) out[b] = {begin, v};
      if (e > end) out[end] = {e, v};
    }
    spans_.swap(out);
  }
  // The one value every byte of [begin, end) has, or false when the range is
  // not covered by a single value.
  bool uniform(uint64_t begin, uint64_t end, int* value) const {
    if (begin >= end) return false;
    uint64_t at = begin;
    int seen = 0;
    bool first = true;
    while (at < end) {
      auto it = spans_.upper_bound(at);
      if (it == spans_.begin()) return false;
      --it;
      const auto [span_end, v] = it->second;
      if (at < it->first || at >= span_end) return false;   // a gap
      if (first) { seen = v; first = false; }
      else if (v != seen) return false;
      at = span_end;
    }
    if (value) *value = seen;
    return true;
  }
  bool empty() const { return spans_.empty(); }

 private:
  // Joins neighbours that carry the same value, so a range advised in pieces
  // answers as one.
  void merge() {
    for (auto it = spans_.begin(); it != spans_.end();) {
      auto next = std::next(it);
      if (next != spans_.end() && it->second.first == next->first &&
          it->second.second == next->second.second) {
        it->second.first = next->second.first;
        spans_.erase(next);
      } else {
        ++it;
      }
    }
  }
  std::map<uint64_t, std::pair<uint64_t, int>> spans_;   // begin -> (end, value)
};

// The hints a managed allocation carries. Advice is a performance hint and
// changes nothing about what a program computes -- there is one physical copy
// of managed memory here -- but it is state a program sets and reads back.
struct ManagedAdvice {
  RangeValues read_mostly;        // 1 where set
  RangeValues preferred;          // the device id, or cudaCpuDeviceId
  RangeValues last_prefetch;      // the device id of the last prefetch
  std::map<int, RangeValues> accessed_by;   // by device id
};

// One stream-ordered memory pool.
//
// CUDA documents what is observable: what a pool has handed out
// (UsedMemCurrent), what it holds from the device (ReservedMemCurrent), the
// high-water marks of both, and a release threshold -- the bytes of freed-but-
// cached memory it keeps rather than giving back. The default threshold is 0,
// so by default a free goes straight back to the device and the pool caches
// nothing, exactly as the documentation says.
//
// Which cached block a request reuses is this engine's choice, not the API's:
// the first block at least as large as the request and no more than twice its
// size, so reuse never quietly wastes a multiple of what was asked for.
struct MemPool {
  int device = 0;
  bool is_default = false;
  bool destroyed = false;
  unsigned long long threshold = 0;   // cudaMemPoolAttrReleaseThreshold
  // The reuse policies. All three are on by default on a device, and here
  // every stream is already synchronous, so they change nothing that can be
  // observed; they are kept and reported because a program reads them back.
  int reuse_follow_event_deps = 1, reuse_allow_opportunistic = 1, reuse_allow_internal_deps = 1;
  uint64_t used = 0, used_high = 0, reserved = 0, reserved_high = 0;
  std::map<uint64_t, uint64_t> live;                 // handed out: pointer -> size
  std::vector<std::pair<uint64_t, uint64_t>> cached; // freed and kept, oldest first
  uint64_t cached_bytes() const {
    uint64_t n = 0;
    for (const auto& [p, sz] : cached) n += sz;
    return n;
  }
};

struct State {
  std::recursive_mutex mu;
  std::unique_ptr<vgpu::runtime::Runtime> rt;
  std::vector<std::unique_ptr<RegisteredModule>> modules;
  std::unordered_map<const void*, KernelInfo> kernels;  // host stub ptr -> kernel
  std::unordered_map<const void*, VarInfo> vars;        // host shadow ptr -> device symbol
  // cudaMallocHost / cudaHostAlloc results. cudaFreeHost consults this before
  // it frees anything: it used to hand whatever it was given to free(), so a
  // malloc'd pointer, a device pointer or a second free took the process down
  // inside the allocator instead of returning cudaErrorInvalidValue.
  std::map<void*, HostRange> host_allocs;
  // Managed allocations, kept apart from host_allocs because freeing one has
  // to unmap it from the device side as well.
  std::map<void*, HostRange> managed_allocs;
  // What cudaMemAdvise and cudaMemPrefetchAsync were told about them.
  std::map<void*, ManagedAdvice> managed_advice;
  // cudaDeviceSetLimit, per device, with the defaults CUDA documents or current
  // devices report. Two of them may not change once a kernel that uses them has
  // launched on the device, which is what the two flags record.
  struct DeviceLimits {
    size_t stack = 1024;
    size_t printf_fifo = 1u << 20;
    size_t malloc_heap = 8u << 20;
    size_t sync_depth = 2;
    size_t pending_launches = 2048;
    size_t l2_fetch_granularity = 64;
    size_t persisting_l2 = 0;
    bool heap_used = false;     // a kernel that calls malloc() or free() has launched
    bool printf_used = false;   // a kernel that calls printf() has launched
  };
  std::map<int, DeviceLimits> limits;
  // Which device system calls each kernel makes, found once from its code.
  struct KernelCalls {
    bool heap = false;
    bool printf = false;
  };
  std::unordered_map<const vgpu::ptx::EntryFn*, KernelCalls> kernel_calls;
  // cudaHostRegister'd ranges. The memory is the caller's; only the record
  // and the device mapping are ours.
  std::map<void*, HostRange> registered;
  // Enabled peer mappings as (accessing device, peer device). Direction
  // matters: enabling 0 -> 1 says nothing about 1 -> 0.
  std::set<std::pair<int, int>> peer_access;
  // Stream-ordered memory pools (cudaMallocAsync). A pool caches what is freed
  // to it and hands it out again, which is the whole reason the API exists: an
  // allocator that asks the driver once and reuses after that.
  //
  // Held in a deque so a pool's address is its handle and never moves. Each
  // device's default pool is made on first use; a program may also create its
  // own and make one of them the device's current pool.
  std::deque<MemPool> pools;
  std::map<int, MemPool*> default_pool;   // by device
  std::map<int, MemPool*> current_pool;
  bool initialized = false;
};

State& st() {
  static State s;
  return s;
}

// The device cudaSetDevice selected, for the calling host thread only. CUDA
// documents the current device as per-thread state, with every new thread
// starting on device 0. This was a field of the process-wide State, so a worker
// thread that called cudaSetDevice(1) silently moved the main thread's next
// allocation and launch onto device 1 -- the usual one-thread-per-GPU pattern
// then raced on a single variable.
thread_local int t_current_device = 0;

// The first mapped host range in `m` containing `p`, or m.end().
std::map<void*, HostRange>::iterator find_range(std::map<void*, HostRange>& m, const void* p) {
  auto it = m.upper_bound(const_cast<void*>(p));
  if (it == m.begin()) return m.end();
  --it;
  const bool inside = static_cast<const char*>(p) <
                      static_cast<const char*>(it->first) + std::max<size_t>(it->second.size, 1);
  return inside ? it : m.end();
}

// Pinned, managed and registered host memory are addressable by kernels on
// every device, at their host address -- that is what unified addressing
// promises, and what cudaPointerGetAttributes reports. Mapping only on the
// device that happened to be current made a managed buffer allocated with
// device 0 current an illegal address on device 1, and unmapping only on the
// current device left a freed buffer mapped on the others, where a kernel
// could still write into memory the allocator had taken back.
void map_host_everywhere(State& s, void* p, size_t n) {
  for (int d = 0; d < s.rt->device_count(); ++d)
    s.rt->device(d).memory().map_host(reinterpret_cast<uint64_t>(p), p, std::max<size_t>(n, 1));
}
void unmap_host_everywhere(State& s, void* p) {
  for (int d = 0; d < s.rt->device_count(); ++d)
    s.rt->device(d).memory().unmap_host(reinterpret_cast<uint64_t>(p));
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
  vgpu::apply_vram_override(profile);
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

// The failure of a kernel launch, as opposed to any other call that was
// refused. On hardware a kernel runs after its launch has returned, so this is
// what cudaDeviceSynchronize exists to report -- and a launch that could not
// run at all (a sampling mode refused, a kernel with no PTX) must not look
// like work that finished. Here it is recorded when the launch returns, and
// kept until cudaGetLastError collects it, like the last error.
thread_local cudaError_t g_async_error = cudaSuccess;

// A context a kernel has corrupted. CUDA documents an illegal address, an
// illegal instruction and a device-side assert as leaving the context unusable:
// every later call fails the same way until the device is reset. This used to
// be forgotten on the next call -- cudaMalloc succeeded right after a kernel had
// written past its allocation -- so a program that did not check the launch
// carried on computing with whatever the dead kernel had left behind, which on
// hardware it could not have done. Per process, like the context it models.
//
// Atomic because it is read without the state lock -- cudaGetLastError,
// cudaPeekAtLastError and cudaDeviceSynchronize take none -- while a launch on
// another thread may be setting it. As a plain global that was a data race.
std::atomic<cudaError_t> g_sticky_error{cudaSuccess};

// Whether an error arose inside a running kernel rather than in the host call.
bool in_kernel(const char* api) {
  return std::strncmp(api, "cudaLaunch", 10) == 0 ||
         std::strncmp(api, "cudaGraphLaunch", 15) == 0;
}

// Only faults inside a kernel poison the context. The same codes from a host
// call -- a cudaMemcpy past the end of a buffer -- are argument errors, and a
// data race is this simulator's own finding, not a condition hardware has.
bool poisons_context(const char* api, vgpu::Err e) {
  using vgpu::Err;
  const bool kernel = in_kernel(api);
  switch (e) {
    case Err::InvalidPointer:
    case Err::UseAfterFree:
    case Err::OutOfBounds:
    case Err::MisalignedAccess:
    case Err::UninitializedRegister:
    case Err::Trap:
    case Err::DeviceAssert:
    case Err::DeviceLost:
      return kernel;
    // Uncorrectable memory is fatal to the context whichever operation read
    // it, a kernel or a copy.
    case Err::EccUncorrectable:
      return true;
    default:
      return false;
  }
}

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
    // A bad pointer is an illegal address only when a kernel dereferenced it.
    // Handed to a host call -- a stack address to cudaFree, a host buffer to a
    // copy whose kind says device, a peer copy with its devices swapped -- it
    // is an argument the call refuses, and the documented answer is
    // cudaErrorInvalidValue, which the driver shim already gave. Reporting 700
    // for those sent people hunting a kernel fault in programs whose kernels
    // were fine.
    case Err::InvalidPointer:
    case Err::UseAfterFree:
    case Err::OutOfBounds:
      code = in_kernel(api) ? cudaErrorIllegalAddress : cudaErrorInvalidValue;
      break;
    // A kernel's misaligned access has a code of its own on hardware (716).
    case Err::MisalignedAccess:
      code = in_kernel(api) ? cudaErrorMisalignedAddress : cudaErrorInvalidValue;
      break;
    case Err::UninitializedRegister: code = cudaErrorIllegalAddress; break;
    // "trap" is what a failed device assert and an unreachable path compile to,
    // and hardware surfaces it as an illegal instruction.
    case Err::Trap: code = cudaErrorIllegalInstruction; break;
    case Err::DeviceAssert: code = cudaErrorAssert; break;
    case Err::EccUncorrectable: code = cudaErrorECCUncorrectable; break;
    // "unspecified launch failure": what programs report when their GPU falls
    // off the bus.
    case Err::DeviceLost: code = cudaErrorLaunchFailure; break;
    // A data race is this simulator's own finding rather than a CUDA condition.
    // "unspecified launch failure" is the closest real code, and it is at least
    // true that the launch did not produce a result anyone should use.
    case Err::DataRace: code = cudaErrorLaunchFailure; break;
    // Too many threads per block, a block or grid dimension past the device's
    // limit, or a zero-sized one. CUDA's name for that is "invalid configuration
    // argument"; it used to fall through to cudaErrorInvalidValue, which is not
    // what anyone searching for that message would find. (The driver API's
    // cuLaunchKernel reports CUDA_ERROR_INVALID_VALUE for the same launch, and
    // keeps doing so in driver_api.cpp.)
    case Err::LaunchConfig: code = cudaErrorInvalidConfiguration; break;
    default: code = cudaErrorInvalidValue; break;
  }
  (void)s;
  if (poisons_context(api, e.code())) g_sticky_error.store(code);
  if (!quiet()) std::fprintf(stderr, "[vgpu] %s: %s\n", api, e.what());
  g_last_error = code;
  return code;
}

vgpu::runtime::Device& current(State& s) { return s.rt->device(t_current_device); }

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
  int dev = t_current_device;
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
  // A profiler's API trace, and the correlation the work this call issues
  // carries (vgpu/profiling.hpp).
  vgpu::profiling::ApiCall call(api);
  const cudaError_t rc = [&]() -> cudaError_t {
    try {
      ensure_init(s);
      if (const cudaError_t sticky = g_sticky_error.load(); sticky != cudaSuccess) {
        g_last_error = sticky;
        return sticky;
      }
      const cudaError_t r = body(s);
      if (r != cudaSuccess) g_last_error = r;
      return r;
    } catch (const vgpu::Error& e) {
      return set_error(s, e, api);
    } catch (const std::exception& e) {
      if (!quiet()) std::fprintf(stderr, "[vgpu] %s: %s\n", api, e.what());
      g_last_error = cudaErrorUnknown;
      return cudaErrorUnknown;
    }
  }();
  call.set_result(rc);
  return rc;
}

bool is_device_ptr(const void* p) {
  return vgpu::is_device_va(reinterpret_cast<uint64_t>(p));
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
    // Drops the leading .version/.target/.address_size directives from a
    // linked-in piece, so the first piece's header describes the whole module.
    auto strip_ptx_header = [](const std::string& text) {
      size_t pos = 0;
      while (pos < text.size()) {
        const size_t eol = text.find('\n', pos);
        const size_t len = (eol == std::string::npos ? text.size() : eol) - pos;
        std::string line = text.substr(pos, len);
        size_t a = line.find_first_not_of(" \t");
        const bool blank = a == std::string::npos;
        const bool header = !blank && (line.compare(a, 8, ".version") == 0 ||
                                       line.compare(a, 7, ".target") == 0 ||
                                       line.compare(a, 13, ".address_size") == 0 ||
                                       line.compare(a, 2, "//") == 0);
        if (!blank && !header) break;
        if (eol == std::string::npos) return std::string{};
        pos = eol + 1;
      }
      return text.substr(pos);
    };
    // The PTX the driver would JIT for this device -- see pick_ptx for the
    // rule. (Every device of a simulated machine has one profile.)
    const vgpu::DeviceProfile& dev = s.rt->device(0).profile();
    const uint32_t cc = static_cast<uint32_t>(dev.cc_major * 10 + dev.cc_minor);
    auto pick_best = [cc](std::vector<vgpu::cuda::FatbinPtx>& v) -> std::string {
      if (v.empty()) return {};
      return std::move(v[vgpu::cuda::pick_ptx(v, cc)].text);
    };
    auto ptxs = vgpu::cuda::extract_ptx(fatCubin);
    rm->ptx = pick_best(ptxs);
    if (rm->ptx.empty()) {
      // A separately compiled build (-rdc=true) leaves the primary fatbin
      // empty -- 16 bytes, just a header -- and puts the real device code in a
      // list of *relocatable* fatbins hanging off the wrapper's fourth field.
      // Device linking would normally consume them; with a PTX-only -code
      // there is nothing for nvlink to link, so the pieces arrive here still
      // separate and the runtime is expected to put them together.
      //
      // The wrapper is
      //   { int magic; int version; const void* data; void* filename_or_fatbins; }
      // and that last field is a filename in version 1 and a NULL-terminated
      // array of fatbin pointers in version 2 -- so the version has to be
      // checked before it is walked, or a char* gets dereferenced as an array.
      int version = 0;
      const void* const* relocatable = nullptr;
      const uint8_t* wp = static_cast<const uint8_t*>(fatCubin);
      if (wp) {
        std::memcpy(&version, wp + 4, 4);
        if (version >= 2) std::memcpy(&relocatable, wp + 16, 8);
      }
      std::string linked;
      size_t pieces = 0;
      for (size_t i = 0; relocatable && relocatable[i] && i < 64; ++i) {
        try {
          auto part = vgpu::cuda::extract_ptx(relocatable[i]);
          std::string text = pick_best(part);
          if (text.empty()) continue;
          // Concatenated rather than merged: cross-piece references resolve
          // by name, exactly as they would after a link.
          //
          // Every piece carries its own .version/.target/.address_size header,
          // and only the first one's counts. The first relocatable fatbin is
          // the translation unit being registered; the rest are libraries
          // linked into it, compiled for whatever the toolkit's default
          // architecture happened to be. Letting the last header win made a
          // module built for sm_80 claim to target sm_121 and be refused on an
          // A100 -- with an error about the *device* being too old, which is
          // the opposite of what had happened.
          if (pieces > 0) text = strip_ptx_header(text);
          linked += text;
          linked += "\n";
          ++pieces;
        } catch (const std::exception&) {
          // A piece that will not parse is skipped rather than failing the
          // whole registration: the others may still hold the kernel.
        }
      }
      rm->ptx = std::move(linked);
      if (trace() && pieces)
        std::fprintf(stderr, "[vgpu][trace] linked %zu relocatable PTX pieces (-rdc build)\n",
                     pieces);
      if (rm->ptx.empty() && !quiet())
        std::fprintf(stderr,
                     "[vgpu] __cudaRegisterFatBinary: no PTX in fatbin (SASS-only build); rebuild "
                     "with an -arch that embeds PTX\n");
      // Return a handle anyway; the failure surfaces at launch with context.
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

VGPU_EXPORT void __cudaRegisterVar(void** fatCubinHandle, char* hostVar, char* /*deviceAddress*/,
                                   const char* deviceName, int /*ext*/, size_t size,
                                   int constant, int /*global*/) {
  // Recorded now, because cudaMemcpyToSymbol has nothing else to go on: the
  // handle it receives is the address of a host shadow object, and only this
  // registration ties it to a name in the module.
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  auto* mod = reinterpret_cast<RegisteredModule*>(fatCubinHandle);
  s.vars[reinterpret_cast<const void*>(hostVar)] =
      VarInfo{mod, deviceName ? deviceName : "", size, constant != 0};
  if (trace())
    std::fprintf(stderr, "[vgpu][trace] __cudaRegisterVar: %p -> '%s' (%zu bytes%s)\n",
                 (void*)hostVar, deviceName ? deviceName : "?", size,
                 constant ? ", constant" : "");
}

namespace {
// Device address and size of a registered symbol, or an error.
cudaError_t symbol_address(State& s, const void* symbol, uint64_t* addr, size_t* size) {
  auto it = s.vars.find(symbol);
  if (it == s.vars.end()) {
    if (!quiet())
      std::fprintf(stderr,
                   "[vgpu] symbol %p is not a registered __device__ or __constant__ variable\n",
                   symbol);
    return cudaErrorInvalidSymbol;
  }
  VarInfo& v = it->second;
  if (!v.mod || v.mod->ptx.empty()) return cudaErrorInvalidSymbol;
  const uint64_t mid = module_on_current(s, *v.mod);
  const vgpu::exec::SymbolTable* syms = current(s).symbols(mid);
  if (!syms) return cudaErrorInvalidSymbol;
  auto sym = syms->find(v.device_name);
  if (sym == syms->end()) {
    if (!quiet())
      std::fprintf(stderr, "[vgpu] '%s' is registered but the module defines no such global\n",
                   v.device_name.c_str());
    return cudaErrorInvalidSymbol;
  }
  *addr = sym->second;
  *size = v.size;
  return cudaSuccess;
}
}  // namespace

VGPU_EXPORT cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count,
                                           size_t offset, cudaMemcpyKind /*kind*/) {
  return guard("cudaMemcpyToSymbol", [&](State& s) -> cudaError_t {
    uint64_t addr = 0;
    size_t size = 0;
    const cudaError_t e = symbol_address(s, symbol, &addr, &size);
    if (e != cudaSuccess) return e;
    if (offset + count > size) return cudaErrorInvalidValue;
    current(s).memory().write(addr + offset, src, count);
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaMemcpyFromSymbol(void* dst, const void* symbol, size_t count,
                                             size_t offset, cudaMemcpyKind /*kind*/) {
  return guard("cudaMemcpyFromSymbol", [&](State& s) -> cudaError_t {
    uint64_t addr = 0;
    size_t size = 0;
    const cudaError_t e = symbol_address(s, symbol, &addr, &size);
    if (e != cudaSuccess) return e;
    if (offset + count > size) return cudaErrorInvalidValue;
    current(s).memory().read(addr + offset, dst, count);
    return cudaSuccess;
  });
}

// The capture hooks, defined with the graph machinery; declared here because
// the first async copies that use them come before it.
bool capture_active(cudaStream_t stream);
bool vgpu_record_copy_if_capturing(const cudaMemcpy3DParms& p, cudaStream_t stream,
                                   cudaError_t* rc);
bool vgpu_record_fill_if_capturing(const cudaMemsetParams& p, cudaStream_t stream,
                                   cudaError_t* rc);
namespace {
cudaMemcpy3DParms linear_copy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
}  // namespace

// On a capturing stream a symbol copy is a copy node to or from the symbol's
// address, as CUDA records it. Run now instead, it would be missing from every
// replay of the graph -- which is what these did.
namespace {
cudaError_t symbol_span(const void* symbol, size_t count, size_t offset, char** at) {
  return guard("cudaMemcpyToSymbolAsync", [&](State& s) -> cudaError_t {
    uint64_t addr = 0;
    size_t size = 0;
    const cudaError_t e = symbol_address(s, symbol, &addr, &size);
    if (e != cudaSuccess) return e;
    if (offset > size || count > size - offset) return cudaErrorInvalidValue;
    *at = reinterpret_cast<char*>(addr) + offset;
    return cudaSuccess;
  });
}
}  // namespace

VGPU_EXPORT cudaError_t cudaMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count,
                                                size_t offset, cudaMemcpyKind kind,
                                                cudaStream_t stream) {
  if (capture_active(stream)) {
    char* at = nullptr;
    if (const cudaError_t e = symbol_span(symbol, count, offset, &at); e != cudaSuccess) return e;
    cudaError_t rc = cudaSuccess;
    if (count == 0) return cudaSuccess;
    vgpu_record_copy_if_capturing(linear_copy(at, src, count, kind), stream, &rc);
    return rc;
  }
  return cudaMemcpyToSymbol(symbol, src, count, offset, kind);
}
VGPU_EXPORT cudaError_t cudaMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t count,
                                                  size_t offset, cudaMemcpyKind kind,
                                                  cudaStream_t stream) {
  if (capture_active(stream)) {
    char* at = nullptr;
    if (const cudaError_t e = symbol_span(symbol, count, offset, &at); e != cudaSuccess) return e;
    cudaError_t rc = cudaSuccess;
    if (count == 0) return cudaSuccess;
    vgpu_record_copy_if_capturing(linear_copy(dst, at, count, kind), stream, &rc);
    return rc;
  }
  return cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
}

VGPU_EXPORT cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol) {
  return guard("cudaGetSymbolAddress", [&](State& s) -> cudaError_t {
    uint64_t addr = 0;
    size_t size = 0;
    const cudaError_t e = symbol_address(s, symbol, &addr, &size);
    if (e != cudaSuccess) return e;
    if (devPtr) *devPtr = reinterpret_cast<void*>(addr);
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaGetSymbolSize(size_t* out, const void* symbol) {
  return guard("cudaGetSymbolSize", [&](State& s) -> cudaError_t {
    uint64_t addr = 0;
    size_t size = 0;
    const cudaError_t e = symbol_address(s, symbol, &addr, &size);
    if (e != cudaSuccess) return e;
    if (out) *out = size;
    return cudaSuccess;
  });
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
// Whether a stream is capturing right now.
bool capture_active(cudaStream_t stream);
// Stream-ordered allocation, host functions and refused operations on a
// capturing stream; each returns false when the stream is not capturing.
bool capture_malloc(State& s, cudaStream_t stream, size_t size, int device, void** ptr,
                    cudaError_t* rc);
bool capture_free(cudaStream_t stream, void* ptr, cudaError_t* rc);
bool capture_host_fn(cudaStream_t stream, cudaHostFn_t fn, void* user);
bool capture_refuse(cudaStream_t stream, const char* what);
// The same for copies and fills with a shape (2D, 3D, pitched). When one returns
// true the operation belongs to the capture, and `*rc` is the call's result.
bool vgpu_record_copy_if_capturing(const cudaMemcpy3DParms& p, cudaStream_t stream,
                                   cudaError_t* rc);
bool vgpu_record_fill_if_capturing(const cudaMemsetParams& p, cudaStream_t stream,
                                   cudaError_t* rc);

bool vgpu_record_launch_if_capturing(const void* func, dim3 grid, dim3 block, void** args,
                                     size_t sharedMem, cudaStream_t stream,
                                     const std::vector<uint32_t>& param_sizes,
                                     bool cooperative = false,
                                     std::array<uint32_t, 3> cluster = {0, 0, 0});
bool vgpu_record_host_op_if_capturing(cudaStream_t stream, std::function<void()> op);
// Discards a capture in progress on a stream that is being destroyed.
void vgpu_drop_capture(cudaStream_t stream);

// Defined with the graph machinery. Frees an allocation that a graph's allocation
// node made, when a program frees it from outside the graph: `*handled` says
// whether the pointer was one of those at all, and the result is what cudaFree
// should return when it was.
cudaError_t free_graph_alloc(State& s, void* ptr, bool* handled);

// The body of both launch entry points. `cooperative` is the only difference,
// and it changes one thing: whether the blocks are resident together and may
// wait on each other. See the scheduler note in interpreter.cpp.
static cudaError_t launch_kernel_impl(const char* api, const void* func, dim3 gridDim,
                                      dim3 blockDim, void** args, size_t sharedMem,
                                      cudaStream_t stream, bool cooperative,
                                      std::array<uint32_t, 3> cluster = {0, 0, 0}) {
  const cudaError_t rc = guard(api, [&](State& s) -> cudaError_t {
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
    if (vgpu_record_launch_if_capturing(func, gridDim, blockDim, args, sharedMem, stream, param_sizes,
                                        cooperative, cluster))
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
    {
      // The heap and printf limits stop being settable once a kernel that uses
      // them has launched on this device, which is decided by what the kernel's
      // code calls rather than by what one run of it happened to reach.
      auto& calls = s.kernel_calls;
      auto kc = calls.find(fn);
      if (kc == calls.end()) {
        State::KernelCalls found;
        for (const auto& ins : fn->body)
          if (const auto* c = std::get_if<vgpu::ptx::OpCall>(&ins.op)) {
            if (c->callee == "malloc" || c->callee == "free") found.heap = true;
            if (c->callee == "vprintf") found.printf = true;
          }
        kc = calls.emplace(fn, found).first;
      }
      State::DeviceLimits& lim = s.limits[t_current_device];
      lim.heap_used = lim.heap_used || kc->second.heap;
      lim.printf_used = lim.printf_used || kc->second.printf;
      cfg.device_heap_bytes = lim.malloc_heap;
    }
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
      ev.device = static_cast<uint32_t>(t_current_device);
      ev.correlation = vgpu::profiling::work_correlation();
      ev.stream = reinterpret_cast<uint64_t>(stream);
      ev.name = ki.entry_name;
      ev.grid[0] = gridDim.x; ev.grid[1] = gridDim.y; ev.grid[2] = gridDim.z;
      ev.block[0] = blockDim.x; ev.block[1] = blockDim.y; ev.block[2] = blockDim.z;
      ev.shared_bytes = static_cast<uint32_t>(sharedMem);
      vgpu::profiling::record(std::move(ev));
    }
    return cudaSuccess;
  });
  if (rc != cudaSuccess) g_async_error = rc;  // see g_async_error
  return rc;
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
    t_current_device = device;
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaGetDevice(int* device) {
  return guard("cudaGetDevice", [&](State& s) {
    if (!device) return cudaErrorInvalidValue;
    *device = t_current_device;
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
  //
  // What it reports is the failure of launched work -- a sticky fault, or a
  // launch that failed without poisoning the context -- and nothing else. It
  // used to return the last error of any kind, so after a cudaMalloc refused
  // with out-of-memory, a synchronize with no kernel in sight reported
  // cudaErrorMemoryAllocation. A refused call is reported by that call and by
  // cudaGetLastError, not by synchronization.
  const cudaError_t sticky = g_sticky_error.load();
  return sticky != cudaSuccess ? sticky : g_async_error;
}

static void forget_arrays_on(int device);

// The one way out of a corrupted context, as on hardware -- and, as documented,
// the end of everything the current device holds: cudaMalloc, pitched and array
// allocations, loaded modules and texture objects, the pinned, managed and
// registered host memory set up with it current, and its peer mappings. This
// used to clear the error and release nothing, so a test harness that reset
// between cases grew without bound, cudaMemGetInfo never recovered, and a
// pointer from before the reset still freed successfully afterwards.
//
// Not through guard(): a reset is exactly the call a sticky error must not
// refuse.
VGPU_EXPORT cudaError_t cudaDeviceReset(void) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  g_last_error = cudaSuccess;
  g_async_error = cudaSuccess;
  g_sticky_error.store(cudaSuccess);
  if (!s.initialized) return cudaSuccess;  // nothing was ever set up
  const int dev = t_current_device;
  auto release = [&](std::map<void*, HostRange>& m, bool ours) {
    for (auto it = m.begin(); it != m.end();) {
      if (it->second.device != dev) {
        ++it;
        continue;
      }
      unmap_host_everywhere(s, it->first);
      if (ours) std::free(it->first);  // registered memory belongs to the caller
      it = m.erase(it);
    }
  };
  release(s.host_allocs, true);
  release(s.managed_allocs, true);
  release(s.registered, false);
  std::erase_if(s.peer_access,
                [dev](const std::pair<int, int>& p) { return p.first == dev || p.second == dev; });
  // Registered fatbins load again on the next launch, as they would into a
  // freshly created context.
  for (auto& m : s.modules) m->module_per_device.erase(dev);
  forget_arrays_on(dev);
  s.limits.erase(dev);   // a fresh context starts from the default limits
  try {
    s.rt->device(dev).reset();
  } catch (const std::exception& e) {
    if (!quiet()) std::fprintf(stderr, "[vgpu] cudaDeviceReset: %s\n", e.what());
  }
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
      // Pinned and registered host memory is mapped into every device at its
      // host address; see cudaHostGetDevicePointer.
      case cudaDevAttrCanMapHostMemory: *value = 1; break;
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
// Peer mappings are implicit in this engine -- every device window is reachable
// from every kernel -- but whether one has been enabled is state a program can
// observe, and callers act on the documented answers: enabling twice is
// cudaErrorPeerAccessAlreadyEnabled, which frameworks deliberately tolerate,
// and disabling what was never enabled is cudaErrorPeerAccessNotEnabled. Both
// used to succeed unconditionally, whatever the flags, and so did disabling a
// device that does not exist.
VGPU_EXPORT cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
  return guard("cudaDeviceEnablePeerAccess", [&](State& s) -> cudaError_t {
    if (peerDevice < 0 || peerDevice >= s.rt->device_count()) return cudaErrorInvalidDevice;
    // A device is not its own peer; cudaDeviceCanAccessPeer says so.
    if (peerDevice == t_current_device) return cudaErrorInvalidDevice;
    if (flags != 0) return cudaErrorInvalidValue;  // reserved, must be 0
    if (!s.peer_access.emplace(t_current_device, peerDevice).second)
      return cudaErrorPeerAccessAlreadyEnabled;
    return cudaSuccess;
  });
}
VGPU_EXPORT cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) {
  return guard("cudaDeviceDisablePeerAccess", [&](State& s) -> cudaError_t {
    if (peerDevice < 0 || peerDevice >= s.rt->device_count()) return cudaErrorInvalidDevice;
    if (!s.peer_access.erase({t_current_device, peerDevice})) return cudaErrorPeerAccessNotEnabled;
    return cudaSuccess;
  });
}

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
      unmap_host_everywhere(s, ptr);
      s.managed_allocs.erase(mit);
      s.managed_advice.erase(ptr);   // its hints go with it
      std::free(ptr);
      return cudaSuccess;
    }
    // An allocation a graph made and did not free itself can be freed from
    // outside, which is what the API documents; one that a free node in its own
    // graph ends belongs to that graph, and freeing it here is refused.
    bool was_graph_alloc = false;
    if (const cudaError_t rc = free_graph_alloc(s, ptr, &was_graph_alloc); was_graph_alloc)
      return rc;
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
      ev.device = static_cast<uint32_t>(t_current_device);
      ev.correlation = vgpu::profiling::work_correlation();
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
                                          cudaStream_t stream) {
  // On a capturing stream: one copy node with the rectangle's shape, as CUDA
  // records it, rather than a copy done now and absent from every replay.
  if (width != 0 && height != 0 && capture_active(stream)) {
    if (!dst || !src) return cudaErrorInvalidValue;
    if (width > dpitch || width > spitch) return cudaErrorInvalidPitchValue;
    cudaMemcpy3DParms p{};
    p.srcPtr = cudaPitchedPtr{const_cast<void*>(src), spitch, width, height};
    p.dstPtr = cudaPitchedPtr{dst, dpitch, width, height};
    p.extent = cudaExtent{width, height, 1};
    p.kind = kind;
    cudaError_t rc = cudaSuccess;
    vgpu_record_copy_if_capturing(p, stream, &rc);
    return rc;
  }
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
                                          size_t height, cudaStream_t stream) {
  if (width != 0 && height != 0 && capture_active(stream)) {
    if (!dst) return cudaErrorInvalidValue;
    cudaMemsetParams p{};
    p.dst = dst;
    p.pitch = pitch;
    p.value = static_cast<unsigned>(value);
    p.elementSize = 1;
    p.width = width;
    p.height = height;
    cudaError_t rc = cudaSuccess;
    vgpu_record_fill_if_capturing(p, stream, &rc);
    return rc;
  }
  return cudaMemset2D(dst, pitch, value, width, height);
}


VGPU_EXPORT cudaError_t cudaMallocHost(void** ptr, size_t size) {
  return guard("cudaMallocHost", [&](State& s) -> cudaError_t {
    if (!ptr) return cudaErrorInvalidValue;
    const size_t n = size ? size : 1;
    void* p = std::aligned_alloc(4096, (n + 4095) / 4096 * 4096);
    if (!p) return cudaErrorMemoryAllocation;
    s.host_allocs[p] = HostRange{n, t_current_device};
    // Pinned memory is device-addressable under unified addressing, at its
    // host address; see cudaHostGetDevicePointer.
    map_host_everywhere(s, p, n);
    *ptr = p;
    return cudaSuccess;
  });
}
VGPU_EXPORT cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int) {
  return cudaMallocHost(ptr, size);
}
VGPU_EXPORT cudaError_t cudaFreeHost(void* ptr) {
  return guard("cudaFreeHost", [&](State& s) -> cudaError_t {
    if (!ptr) return cudaSuccess;
    // Only a base pointer this runtime handed out, and only once. Anything
    // else used to go straight to free(): a malloc'd pointer aborted inside
    // glibc, a device pointer segfaulted, and a second free corrupted the heap.
    auto it = s.host_allocs.find(ptr);
    if (it == s.host_allocs.end()) {
      if (!quiet())
        std::fprintf(stderr,
                     "[vgpu] cudaFreeHost: %p was not returned by cudaMallocHost or cudaHostAlloc, "
                     "or has already been freed\n",
                     ptr);
      return cudaErrorInvalidValue;
    }
    unmap_host_everywhere(s, ptr);
    s.host_allocs.erase(it);
    std::free(ptr);
    return cudaSuccess;
  });
}
VGPU_EXPORT cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src,
                                            int srcDevice, size_t count, cudaStream_t stream) {
  // Captured, a peer copy is a copy node between the two devices' addresses:
  // each side's device is found from its address when the graph runs.
  if (count != 0 && capture_active(stream)) {
    if (!dst || !src) return cudaErrorInvalidValue;
    cudaError_t rc = cudaSuccess;
    vgpu_record_copy_if_capturing(linear_copy(dst, src, count, cudaMemcpyDefault), stream, &rc);
    return rc;
  }
  return cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
}

// A 3D copy between devices' linear, pitched memory: each row of the extent
// (width bytes) from its place in the source to its place in the destination,
// through the peer copy. CUDA arrays are not modelled for 3D copies.
VGPU_EXPORT cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms* p) {
  if (!p) return cudaErrorInvalidValue;
  if (p->srcArray || p->dstArray) {
    if (!quiet()) std::fprintf(stderr, "[vgpu] cudaMemcpy3DPeer: CUDA arrays are not supported\n");
    return cudaErrorNotSupported;
  }
  const cudaPitchedPtr& sp = p->srcPtr;
  const cudaPitchedPtr& dp = p->dstPtr;
  if (!sp.ptr || !dp.ptr || p->extent.width > sp.pitch || p->extent.width > dp.pitch)
    return cudaErrorInvalidValue;
  const size_t src_slice = sp.pitch * sp.ysize, dst_slice = dp.pitch * dp.ysize;
  for (size_t z = 0; z < p->extent.depth; ++z)
    for (size_t y = 0; y < p->extent.height; ++y) {
      const char* s = static_cast<const char*>(sp.ptr) + (p->srcPos.z + z) * src_slice +
                      (p->srcPos.y + y) * sp.pitch + p->srcPos.x;
      char* d = static_cast<char*>(dp.ptr) + (p->dstPos.z + z) * dst_slice + (p->dstPos.y + y) * dp.pitch +
                p->dstPos.x;
      const cudaError_t e = cudaMemcpyPeer(d, p->dstDevice, s, p->srcDevice, p->extent.width);
      if (e != cudaSuccess) return e;
    }
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms* p,
                                              cudaStream_t stream) {
  if (p && capture_active(stream)) {
    cudaMemcpy3DParms c{};
    c.srcArray = p->srcArray;
    c.srcPos = p->srcPos;
    c.srcPtr = p->srcPtr;
    c.dstArray = p->dstArray;
    c.dstPos = p->dstPos;
    c.dstPtr = p->dstPtr;
    c.extent = p->extent;
    c.kind = cudaMemcpyDefault;
    cudaError_t rc = cudaSuccess;
    vgpu_record_copy_if_capturing(c, stream, &rc);
    return rc;
  }
  return cudaMemcpy3DPeer(p);
}

/* ===================================================================== */
/* Streams and events (synchronous / wall-clock)                         */
/* ===================================================================== */

// ---- streams ----------------------------------------------------------------
//
// Every stream runs on the same synchronous engine, so what a stream is for here
// is identity: which one a capture is recording, and what a program is told
// when it asks about one. Every cudaStreamCreate used to return the same handle
// -- 0x1, which is cudaStreamLegacy -- so two streams compared equal, a capture
// begun on one captured the work sent to every other, and a kernel launched on
// a second stream during a capture never ran: it was recorded into the first
// stream's graph instead. A stream is a record now, and its address is its
// handle.
//
// The default streams have no record: 0 and cudaStreamLegacy are the legacy
// stream, and cudaStreamPerThread is each thread's own. A handle this runtime
// did not create may be a stream the driver API made, which this library cannot
// see, so it is answered the way every stream here behaves -- the defaults --
// rather than refused.
struct RtStream {
  unsigned flags = cudaStreamDefault;
  int priority = 0;
  int device = 0;
  unsigned long long id = 0;
  // Attributes a program set on the stream. Access-policy windows and memory
  // synchronisation domains are performance policy for hardware this does not
  // model, and a set attribute reads back as what was set, which is the
  // contract; the priority attribute is the stream's priority.
  std::map<int, cudaStreamAttrValue> attrs;
};

std::mutex g_stream_mu;
std::unordered_map<void*, std::unique_ptr<RtStream>> g_streams;
// Ids are never reused, and start well above any small integer so an id cannot
// be mistaken for the handle of a stream the driver API made.
std::atomic<unsigned long long> g_next_stream_id{1ull << 32};
constexpr unsigned long long kLegacyStreamId = 1;

bool is_default_stream(cudaStream_t s) {
  return s == nullptr || s == cudaStreamLegacy || s == cudaStreamPerThread;
}

// The record for a stream this runtime created, or nullptr. Caller holds
// g_stream_mu.
RtStream* stream_record(cudaStream_t s) {
  const auto it = g_streams.find(static_cast<void*>(s));
  return it == g_streams.end() ? nullptr : it->second.get();
}

// Each thread's per-thread default stream is its own stream, so it has its own
// id, taken the first time the thread asks.
unsigned long long per_thread_stream_id() {
  thread_local unsigned long long id = g_next_stream_id.fetch_add(1);
  return id;
}

// Priorities are clamped to the device's range, as CUDA documents. The range
// here is a single level, so every stream gets it.
int clamp_priority(int p) {
  int least = 0, greatest = 0;
  cudaDeviceGetStreamPriorityRange(&least, &greatest);
  return std::min(std::max(p, greatest), least);
}

cudaError_t create_stream(cudaStream_t* out, unsigned flags, int priority) {
  if (!out) return cudaErrorInvalidValue;
  if (flags & ~static_cast<unsigned>(cudaStreamNonBlocking)) return cudaErrorInvalidValue;
  auto rec = std::make_unique<RtStream>();
  rec->flags = flags;
  rec->priority = clamp_priority(priority);
  rec->device = t_current_device;
  rec->id = g_next_stream_id.fetch_add(1);
  void* handle = rec.get();
  {
    std::lock_guard<std::mutex> lock(g_stream_mu);
    g_streams[handle] = std::move(rec);
  }
  *out = static_cast<cudaStream_t>(handle);
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaStreamCreate(cudaStream_t* s) {
  return create_stream(s, cudaStreamDefault, 0);
}
VGPU_EXPORT cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned int flags) {
  return create_stream(s, flags, 0);
}

VGPU_EXPORT cudaError_t cudaStreamGetFlags(cudaStream_t stream, unsigned int* flags) {
  if (!flags) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_stream_mu);
  const RtStream* r = stream_record(stream);
  *flags = r ? r->flags : cudaStreamDefault;
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaStreamGetDevice(cudaStream_t stream, int* device) {
  if (!device) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_stream_mu);
  const RtStream* r = stream_record(stream);
  // A stream belongs to the device that was current when it was made; the
  // default streams, and streams this runtime did not make, to the current one.
  *device = r ? r->device : t_current_device;
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaStreamGetId(cudaStream_t stream, unsigned long long* id) {
  if (!id) return cudaErrorInvalidValue;
  if (stream == nullptr || stream == cudaStreamLegacy) {
    *id = kLegacyStreamId;
    return cudaSuccess;
  }
  if (stream == cudaStreamPerThread) {
    *id = per_thread_stream_id();
    return cudaSuccess;
  }
  std::lock_guard<std::mutex> lock(g_stream_mu);
  const RtStream* r = stream_record(stream);
  // A stream the driver API made is known here only by its handle, which is
  // unique among live streams and below every id this runtime hands out.
  *id = r ? r->id : reinterpret_cast<uintptr_t>(stream);
  return cudaSuccess;
}

namespace {
bool is_stream_attribute(cudaStreamAttrID a) {
  return a == cudaStreamAttributeAccessPolicyWindow ||
         a == cudaStreamAttributeSynchronizationPolicy ||
         a == cudaStreamAttributeMemSyncDomainMap || a == cudaStreamAttributeMemSyncDomain ||
         a == cudaStreamAttributePriority;
}
// Default streams keep their attributes in records of their own, made the first
// time something is set on them: one for the legacy stream, one per thread for
// the per-thread stream.
RtStream& attribute_record(cudaStream_t s) {  // caller holds g_stream_mu
  if (RtStream* r = stream_record(s)) return *r;
  if (s == cudaStreamPerThread) {
    thread_local RtStream per_thread;
    return per_thread;
  }
  static RtStream legacy;   // the legacy stream, and any stream made elsewhere
  return legacy;
}
}  // namespace

VGPU_EXPORT cudaError_t cudaStreamGetAttribute(cudaStream_t stream, cudaStreamAttrID attr,
                                               cudaStreamAttrValue* value) {
  if (!value || !is_stream_attribute(attr)) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_stream_mu);
  RtStream& r = attribute_record(stream);
  std::memset(value, 0, sizeof *value);
  if (attr == cudaStreamAttributePriority) {
    value->priority = r.priority;
    return cudaSuccess;
  }
  if (const auto it = r.attrs.find(static_cast<int>(attr)); it != r.attrs.end()) *value = it->second;
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaStreamSetAttribute(cudaStream_t stream, cudaStreamAttrID attr,
                                               const cudaStreamAttrValue* value) {
  if (!value || !is_stream_attribute(attr)) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_stream_mu);
  RtStream& r = attribute_record(stream);
  if (attr == cudaStreamAttributePriority) {
    r.priority = clamp_priority(value->priority);
    return cudaSuccess;
  }
  r.attrs[static_cast<int>(attr)] = *value;
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src) {
  std::lock_guard<std::mutex> lock(g_stream_mu);
  RtStream& from = attribute_record(src);
  RtStream& to = attribute_record(dst);
  if (&from == &to) return cudaSuccess;
  to.attrs = from.attrs;
  to.priority = from.priority;
  return cudaSuccess;
}

/* ---- entry points PyTorch and other frameworks link against ----
 *
 * A framework resolves these at load time, so a missing one stops the import
 * before any kernel runs. Everything here is either a faithful implementation
 * or an honest error: reporting success for something not actually done would
 * make a framework believe it had memory or a capability it does not have.
 */

// Resource limits. Each is kept per device and reads back as what was set,
// after the clamping CUDA documents the driver may do. The heap limit is real:
// it is the size of the heap a kernel's malloc() draws from. The stack and
// printf-buffer sizes are recorded and reported but bound nothing here -- a
// kernel's stack grows as far as it needs, and printf output is written as it
// happens rather than through a buffer that can fill -- so a program that fits
// the hardware's limits runs here too. The device-runtime limits describe
// dynamic parallelism, which is not implemented, and the L2 ones are
// performance hints CUDA itself says may be ignored.
namespace {
// The limit a caller passed, as the integer it is. A program built against a
// newer toolkit can pass a limit this one does not name, and switching on an
// enum holding a value outside its enumerators is undefined behaviour -- which
// UBSan reports -- so the bytes are read as an int and the unknown case is a
// clean cudaErrorUnsupportedLimit.
int limit_id(cudaLimit limit) {
  int id = 0;
  std::memcpy(&id, &limit, sizeof id < sizeof limit ? sizeof id : sizeof limit);
  return id;
}
}  // namespace

VGPU_EXPORT cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value) {
  const int id = limit_id(limit);
  return guard("cudaDeviceSetLimit", [&](State& s) -> cudaError_t {
    State::DeviceLimits& lim = s.limits[t_current_device];
    switch (id) {
      case cudaLimitStackSize:
        lim.stack = (value + 15) / 16 * 16;   // rounded up to a whole element
        return cudaSuccess;
      case cudaLimitPrintfFifoSize:
        // Not after a kernel that calls printf() has launched, as documented.
        if (lim.printf_used) return cudaErrorInvalidValue;
        lim.printf_fifo = value;
        return cudaSuccess;
      case cudaLimitMallocHeapSize:
        // Not after a kernel that calls malloc() or free() has launched: the
        // heap it drew from is already laid out.
        if (lim.heap_used) return cudaErrorInvalidValue;
        lim.malloc_heap = value;
        return cudaSuccess;
      case cudaLimitDevRuntimeSyncDepth: {
        // Only below compute capability 9.0, and no deeper than 24 levels.
        const vgpu::DeviceProfile& p = s.rt->device(t_current_device).profile();
        if (p.cc_major >= 9) return cudaErrorUnsupportedLimit;
        lim.sync_depth = std::min<size_t>(value, 24);
        return cudaSuccess;
      }
      case cudaLimitDevRuntimePendingLaunchCount:
        lim.pending_launches = value;
        return cudaSuccess;
      case cudaLimitMaxL2FetchGranularity:
        lim.l2_fetch_granularity = std::min<size_t>(value, 128);   // 0 to 128 bytes
        return cudaSuccess;
      case cudaLimitPersistingL2CacheSize:
        // Clamped to what the device reports it can set aside, which is none.
        lim.persisting_l2 = 0;
        return cudaSuccess;
      default:
        return cudaErrorUnsupportedLimit;
    }
  });
}

VGPU_EXPORT cudaError_t cudaDeviceGetLimit(size_t* value, cudaLimit limit) {
  if (!value) return cudaErrorInvalidValue;
  const int id = limit_id(limit);
  return guard("cudaDeviceGetLimit", [&](State& s) -> cudaError_t {
    const State::DeviceLimits& lim = s.limits[t_current_device];
    switch (id) {
      case cudaLimitStackSize: *value = lim.stack; return cudaSuccess;
      case cudaLimitPrintfFifoSize: *value = lim.printf_fifo; return cudaSuccess;
      case cudaLimitMallocHeapSize: *value = lim.malloc_heap; return cudaSuccess;
      case cudaLimitDevRuntimeSyncDepth: {
        const vgpu::DeviceProfile& p = s.rt->device(t_current_device).profile();
        if (p.cc_major >= 9) return cudaErrorUnsupportedLimit;
        *value = lim.sync_depth;
        return cudaSuccess;
      }
      case cudaLimitDevRuntimePendingLaunchCount: *value = lim.pending_launches; return cudaSuccess;
      case cudaLimitMaxL2FetchGranularity: *value = lim.l2_fetch_granularity; return cudaSuccess;
      case cudaLimitPersistingL2CacheSize: *value = lim.persisting_l2; return cudaSuccess;
      default: return cudaErrorUnsupportedLimit;
    }
  });
}

// Streams are executed inline, so every priority is equally honoured. CUDA
// reports the range as [greatest, least] with lower meaning higher priority.
VGPU_EXPORT cudaError_t cudaDeviceGetStreamPriorityRange(int* least, int* greatest) {
  if (least) *least = 0;
  if (greatest) *greatest = 0;
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaStreamCreateWithPriority(cudaStream_t* s, unsigned int flags,
                                                     int priority) {
  return create_stream(s, flags, priority);
}

// ---- stream-ordered memory pools -------------------------------------------------
//
// Every stream here is synchronous, so the *ordering* half of the API needs
// nothing: an allocation is usable when it returns and a free is complete when
// it returns. What the API is really for is the caching, and that is modelled
// for real -- a pool holds what was freed to it, up to its release threshold,
// and hands it out again.

namespace {

MemPool& default_pool_for(State& s, int device) {
  auto it = s.default_pool.find(device);
  if (it != s.default_pool.end()) return *it->second;
  s.pools.push_back(MemPool{});
  MemPool& p = s.pools.back();
  p.device = device;
  p.is_default = true;
  s.default_pool[device] = &p;
  return p;
}

MemPool& pool_for(State& s, int device) {
  auto it = s.current_pool.find(device);
  return it != s.current_pool.end() ? *it->second : default_pool_for(s, device);
}

// A handle is the pool's address; this checks one before following it.
MemPool* from_handle(State& s, cudaMemPool_t h) {
  for (MemPool& p : s.pools)
    if (reinterpret_cast<cudaMemPool_t>(&p) == h && !p.destroyed) return &p;
  return nullptr;
}

// Gives cached blocks back to the device until no more than `keep` bytes are
// held, oldest first.
void release_cached(State& s, MemPool& p, uint64_t keep) {
  while (p.cached_bytes() > keep && !p.cached.empty()) {
    const auto [ptr, size] = p.cached.front();
    p.cached.erase(p.cached.begin());
    s.rt->device(p.device).memory().free(ptr);
    p.reserved -= size;
  }
}

cudaError_t pool_alloc(State& s, MemPool& p, size_t size, void** out) {
  // Reuse: the first cached block big enough, and no more than twice the size
  // asked for, so a small request cannot take a huge block out of circulation.
  for (size_t i = 0; i < p.cached.size(); ++i) {
    const auto [ptr, block] = p.cached[i];
    if (block >= size && block <= 2 * size) {
      p.cached.erase(p.cached.begin() + static_cast<long>(i));
      p.live[ptr] = block;
      p.used += block;
      p.used_high = std::max(p.used_high, p.used);
      *out = reinterpret_cast<void*>(ptr);
      return cudaSuccess;
    }
  }
  const uint64_t ptr = s.rt->device(p.device).memory().alloc(size ? size : 1);
  p.live[ptr] = size ? size : 1;
  p.used += p.live[ptr];
  p.reserved += p.live[ptr];
  p.used_high = std::max(p.used_high, p.used);
  p.reserved_high = std::max(p.reserved_high, p.reserved);
  *out = reinterpret_cast<void*>(ptr);
  return cudaSuccess;
}

// The pool a pointer came from, or null: what tells cudaFreeAsync whether it is
// freeing pool memory or an ordinary allocation.
MemPool* pool_of(State& s, uint64_t ptr) {
  for (MemPool& p : s.pools)
    if (p.live.count(ptr)) return &p;
  return nullptr;
}

void pool_free(State& s, MemPool& p, uint64_t ptr) {
  const uint64_t size = p.live[ptr];
  p.live.erase(ptr);
  p.used -= size;
  p.cached.emplace_back(ptr, size);
  // The default threshold is 0, so by default this hands the memory straight
  // back to the device and the pool keeps nothing.
  release_cached(s, p, p.threshold);
}

}  // namespace

VGPU_EXPORT cudaError_t cudaMallocAsync(void** ptr, size_t size, cudaStream_t stream) {
  return guard("cudaMallocAsync", [&](State& s) -> cudaError_t {
    if (!ptr) return cudaErrorInvalidValue;
    if (vgpu::faults::should_fail(vgpu::faults::Op::Alloc)) return cudaErrorMemoryAllocation;
    cudaError_t rc = cudaSuccess;
    if (capture_malloc(s, stream, size, t_current_device, ptr, &rc)) return rc;
    return pool_alloc(s, pool_for(s, t_current_device), size, ptr);
  });
}

VGPU_EXPORT cudaError_t cudaMallocFromPoolAsync(void** ptr, size_t size, cudaMemPool_t pool,
                                                cudaStream_t stream) {
  return guard("cudaMallocFromPoolAsync", [&](State& s) -> cudaError_t {
    if (!ptr) return cudaErrorInvalidValue;
    MemPool* p = from_handle(s, pool);
    if (!p) return cudaErrorInvalidValue;
    // Captured, the pool's properties -- its device -- set the node's; the
    // allocation belongs to the graph, not to the pool.
    cudaError_t rc = cudaSuccess;
    if (capture_malloc(s, stream, size, p->device, ptr, &rc)) return rc;
    return pool_alloc(s, *p, size, ptr);
  });
}

VGPU_EXPORT cudaError_t cudaFreeAsync(void* ptr, cudaStream_t stream) {
  return guard("cudaFreeAsync", [&](State& s) -> cudaError_t {
    if (!ptr) return cudaSuccess;   // as cudaFree(nullptr) is
    cudaError_t rc = cudaSuccess;
    if (capture_free(stream, ptr, &rc)) return rc;
    if (MemPool* p = pool_of(s, reinterpret_cast<uint64_t>(ptr))) {
      pool_free(s, *p, reinterpret_cast<uint64_t>(ptr));
      return cudaSuccess;
    }
    // Not pool memory: an ordinary allocation, which cudaFreeAsync also takes.
    return cudaFree(ptr);
  });
}

VGPU_EXPORT cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* pool, int device) {
  return guard("cudaDeviceGetDefaultMemPool", [&](State& s) -> cudaError_t {
    if (!pool || device < 0 || device >= s.rt->device_count()) return cudaErrorInvalidValue;
    *pool = reinterpret_cast<cudaMemPool_t>(&default_pool_for(s, device));
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaDeviceGetMemPool(cudaMemPool_t* pool, int device) {
  return guard("cudaDeviceGetMemPool", [&](State& s) -> cudaError_t {
    if (!pool || device < 0 || device >= s.rt->device_count()) return cudaErrorInvalidValue;
    *pool = reinterpret_cast<cudaMemPool_t>(&pool_for(s, device));
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t pool) {
  return guard("cudaDeviceSetMemPool", [&](State& s) -> cudaError_t {
    if (device < 0 || device >= s.rt->device_count()) return cudaErrorInvalidValue;
    MemPool* p = from_handle(s, pool);
    if (!p) return cudaErrorInvalidValue;
    // A pool belongs to the device it was made for; pointing another device at
    // it would hand out addresses that device does not own.
    if (p->device != device) return cudaErrorInvalidValue;
    s.current_pool[device] = p;
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaMemPoolCreate(cudaMemPool_t* pool, const cudaMemPoolProps* props) {
  return guard("cudaMemPoolCreate", [&](State& s) -> cudaError_t {
    if (!pool || !props) return cudaErrorInvalidValue;
    if (props->allocType != cudaMemAllocationTypePinned) return cudaErrorInvalidValue;
    if (props->location.type != cudaMemLocationTypeDevice) return cudaErrorInvalidValue;
    if (props->location.id < 0 || props->location.id >= s.rt->device_count())
      return cudaErrorInvalidDevice;
    // A pool another process could allocate from would have to share this
    // process's own memory.
    if (props->handleTypes != cudaMemHandleTypeNone) return cudaErrorNotSupported;
    s.pools.push_back(MemPool{});
    MemPool& p = s.pools.back();
    p.device = props->location.id;
    *pool = reinterpret_cast<cudaMemPool_t>(&p);
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaMemPoolDestroy(cudaMemPool_t pool) {
  return guard("cudaMemPoolDestroy", [&](State& s) -> cudaError_t {
    MemPool* p = from_handle(s, pool);
    if (!p || p->is_default) return cudaErrorInvalidValue;   // the default pool is the device's
    // Outstanding allocations outlive the pool on hardware only until they are
    // freed, and freeing them afterwards needs the pool. Refuse instead.
    if (!p->live.empty()) {
      if (!quiet())
        std::fprintf(stderr, "[vgpu] cudaMemPoolDestroy: the pool still has %zu allocation(s); "
                             "free them with cudaFreeAsync first\n", p->live.size());
      return cudaErrorInvalidValue;
    }
    release_cached(s, *p, 0);
    p->destroyed = true;
    for (auto it = s.current_pool.begin(); it != s.current_pool.end();)
      it = it->second == p ? s.current_pool.erase(it) : std::next(it);
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t pool, cudaMemPoolAttr attr,
                                                void* value) {
  return guard("cudaMemPoolSetAttribute", [&](State& s) -> cudaError_t {
    MemPool* p = from_handle(s, pool);
    if (!p || !value) return cudaErrorInvalidValue;
    switch (attr) {
      case cudaMemPoolAttrReleaseThreshold:
        p->threshold = *static_cast<unsigned long long*>(value);
        // Lowering it takes effect now, as it does on a device at the next
        // synchronization -- and here every operation is already synchronous.
        release_cached(s, *p, p->threshold);
        return cudaSuccess;
      case cudaMemPoolReuseFollowEventDependencies:
        p->reuse_follow_event_deps = *static_cast<int*>(value);
        return cudaSuccess;
      case cudaMemPoolReuseAllowOpportunistic:
        p->reuse_allow_opportunistic = *static_cast<int*>(value);
        return cudaSuccess;
      case cudaMemPoolReuseAllowInternalDependencies:
        p->reuse_allow_internal_deps = *static_cast<int*>(value);
        return cudaSuccess;
      // The documented way to reset a high-water mark is to write 0 to it.
      case cudaMemPoolAttrReservedMemHigh:
        if (*static_cast<unsigned long long*>(value) != 0) return cudaErrorInvalidValue;
        p->reserved_high = p->reserved;
        return cudaSuccess;
      case cudaMemPoolAttrUsedMemHigh:
        if (*static_cast<unsigned long long*>(value) != 0) return cudaErrorInvalidValue;
        p->used_high = p->used;
        return cudaSuccess;
      default:
        return cudaErrorInvalidValue;   // the current totals are read-only
    }
  });
}

VGPU_EXPORT cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t pool, cudaMemPoolAttr attr,
                                                void* value) {
  return guard("cudaMemPoolGetAttribute", [&](State& s) -> cudaError_t {
    MemPool* p = from_handle(s, pool);
    if (!p || !value) return cudaErrorInvalidValue;
    switch (attr) {
      case cudaMemPoolReuseFollowEventDependencies:
        *static_cast<int*>(value) = p->reuse_follow_event_deps;
        return cudaSuccess;
      case cudaMemPoolReuseAllowOpportunistic:
        *static_cast<int*>(value) = p->reuse_allow_opportunistic;
        return cudaSuccess;
      case cudaMemPoolReuseAllowInternalDependencies:
        *static_cast<int*>(value) = p->reuse_allow_internal_deps;
        return cudaSuccess;
      case cudaMemPoolAttrReleaseThreshold:
        *static_cast<unsigned long long*>(value) = p->threshold;
        return cudaSuccess;
      case cudaMemPoolAttrReservedMemCurrent:
        *static_cast<unsigned long long*>(value) = p->reserved;
        return cudaSuccess;
      case cudaMemPoolAttrReservedMemHigh:
        *static_cast<unsigned long long*>(value) = p->reserved_high;
        return cudaSuccess;
      case cudaMemPoolAttrUsedMemCurrent:
        *static_cast<unsigned long long*>(value) = p->used;
        return cudaSuccess;
      case cudaMemPoolAttrUsedMemHigh:
        *static_cast<unsigned long long*>(value) = p->used_high;
        return cudaSuccess;
      default:
        return cudaErrorInvalidValue;
    }
  });
}

VGPU_EXPORT cudaError_t cudaMemPoolSetAccess(cudaMemPool_t pool, const cudaMemAccessDesc* desc,
                                             size_t count) {
  return guard("cudaMemPoolSetAccess", [&](State& s) -> cudaError_t {
    MemPool* p = from_handle(s, pool);
    if (!p || (!desc && count)) return cudaErrorInvalidValue;
    // Every device here reaches a pool's memory through the same address, and
    // the owning device always may, so the only request that means anything is
    // one for a device that exists.
    for (size_t i = 0; i < count; ++i)
      if (desc[i].location.type != cudaMemLocationTypeDevice ||
          desc[i].location.id < 0 || desc[i].location.id >= s.rt->device_count())
        return cudaErrorInvalidValue;
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaMemPoolGetAccess(enum cudaMemAccessFlags* flags, cudaMemPool_t pool,
                                             struct cudaMemLocation* location) {
  return guard("cudaMemPoolGetAccess", [&](State& s) -> cudaError_t {
    MemPool* p = from_handle(s, pool);
    if (!p || !flags || !location) return cudaErrorInvalidValue;
    if (location->type != cudaMemLocationTypeDevice) return cudaErrorInvalidValue;
    if (location->id < 0 || location->id >= s.rt->device_count()) return cudaErrorInvalidValue;
    *flags = cudaMemAccessFlagsProtReadWrite;
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaMemPoolTrimTo(cudaMemPool_t pool, size_t keep) {
  return guard("cudaMemPoolTrimTo", [&](State& s) -> cudaError_t {
    MemPool* p = from_handle(s, pool);
    if (!p) return cudaErrorInvalidValue;
    release_cached(s, *p, keep);
    return cudaSuccess;
  });
}

// Page-locking changes nothing when the "device" shares the host's memory, but
// a registration is still state: it makes the range device-addressable, it is
// what cudaPointerGetAttributes reports, and the documented errors depend on
// it. These used to succeed for anything non-NULL, so registering twice,
// unregistering what was never registered, and asking what registered memory
// was all got answers no driver gives.
VGPU_EXPORT cudaError_t cudaHostRegister(void* p, size_t size, unsigned int) {
  return guard("cudaHostRegister", [&](State& s) -> cudaError_t {
    if (!p || size == 0) return cudaErrorInvalidValue;
    const char* lo = static_cast<const char*>(p);
    void* hi = static_cast<char*>(p) + size;
    // Any overlap with a registered range, or with memory CUDA already pins,
    // is "already registered". Ranges in each map are disjoint, so the one
    // starting last below `hi` is the only one that can reach past `lo`.
    auto overlaps = [&](const std::map<void*, HostRange>& m) {
      auto it = m.lower_bound(hi);
      if (it == m.begin()) return false;
      --it;
      return static_cast<const char*>(it->first) + std::max<size_t>(it->second.size, 1) > lo;
    };
    if (overlaps(s.registered) || overlaps(s.host_allocs) || overlaps(s.managed_allocs))
      return cudaErrorHostMemoryAlreadyRegistered;
    s.registered[p] = HostRange{size, t_current_device};
    map_host_everywhere(s, p, size);
    return cudaSuccess;
  });
}
VGPU_EXPORT cudaError_t cudaHostUnregister(void* p) {
  return guard("cudaHostUnregister", [&](State& s) -> cudaError_t {
    if (!p) return cudaErrorInvalidValue;
    auto it = s.registered.find(p);
    if (it == s.registered.end()) return cudaErrorHostMemoryNotRegistered;
    unmap_host_everywhere(s, p);
    s.registered.erase(it);
    return cudaSuccess;
  });
}
// Under unified addressing, memory CUDA pins or registers is reachable from
// every device at its host address, and that is the device pointer -- the same
// one cudaPointerGetAttributes reports. This used to refuse even
// cudaHostAllocMapped memory, which exists to be asked this, while the
// attributes call named the host address as the device pointer; a kernel given
// that pointer then failed with an illegal address, because nothing had
// mapped it.
VGPU_EXPORT cudaError_t cudaHostGetDevicePointer(void** dev, void* host, unsigned int flags) {
  return guard("cudaHostGetDevicePointer", [&](State& s) -> cudaError_t {
    if (!dev || !host || flags != 0) return cudaErrorInvalidValue;
    if (find_range(s.host_allocs, host) == s.host_allocs.end() &&
        find_range(s.registered, host) == s.registered.end())
      return cudaErrorInvalidValue;
    *dev = host;
    return cudaSuccess;
  });
}

// Where a pointer lives. Frameworks branch on this to pick a copy path, so
// getting it wrong sends a device buffer through a host memcpy.
VGPU_EXPORT cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attr, const void* p) {
  return guard("cudaPointerGetAttributes", [&](State& st) {
    if (!attr) return cudaErrorInvalidValue;
    std::memset(attr, 0, sizeof *attr);
    const uint64_t a = reinterpret_cast<uint64_t>(p);
    // Pinned, registered and managed allocations are host addresses, and
    // callers branch on the difference: a framework picks a zero-copy path for
    // a pinned buffer and a migration path for a managed one. Each is
    // addressable by kernels at the same address, as with UVA.
    auto host_kind = [&](std::map<void*, HostRange>& m, cudaMemoryType type) {
      auto it = find_range(m, p);
      if (it == m.end()) return false;
      attr->type = type;
      attr->device = it->second.device;
      attr->hostPointer = const_cast<void*>(p);
      attr->devicePointer = const_cast<void*>(p);
      return true;
    };
    if (host_kind(st.managed_allocs, cudaMemoryTypeManaged) ||
        host_kind(st.host_allocs, cudaMemoryTypeHost) ||
        host_kind(st.registered, cudaMemoryTypeHost))
      return cudaSuccess;
    if (vgpu::is_device_va(a)) {
      // An address in a device window is a device pointer only while something
      // is allocated there. This used to answer "device memory" for any address
      // past the base -- a freed pointer, a random one -- with a device ordinal
      // computed from it that no rack has, so an is-this-mine check accepted
      // memory that was gone.
      const uint64_t ord = (a - vgpu::kDeviceVaBase) / vgpu::kDeviceVaStride;
      if (ord >= static_cast<uint64_t>(st.rt->device_count()) ||
          !st.rt->device(static_cast<int>(ord)).memory().find_allocation(a, nullptr, nullptr)) {
        if (!quiet())
          std::fprintf(stderr,
                       "[vgpu] cudaPointerGetAttributes: %p is in the device address range but "
                       "not inside any live allocation (freed, or never allocated)\n",
                       p);
        return cudaErrorInvalidValue;
      }
      attr->type = cudaMemoryTypeDevice;
      attr->device = static_cast<int>(ord);
      attr->devicePointer = const_cast<void*>(p);
      return cudaSuccess;
    }
    // Plain host memory CUDA knows nothing about.
    attr->type = cudaMemoryTypeUnregistered;
    attr->device = t_current_device;
    attr->hostPointer = const_cast<void*>(p);
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
  // Not supported under stream capture, as documented -- cudaLaunchHostFunc is
  // the one that is -- so a capturing stream refuses it and the capture fails.
  if (capture_refuse(stream, "cudaStreamAddCallback")) return cudaErrorStreamCaptureUnsupported;
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

// ---- sharing memory between processes (IPC) ---------------------------------------
//
// A device pointer means nothing in another process, which is why these used to
// refuse. What can cross a process boundary is a file, and the memory backing
// already maps its slices MAP_SHARED -- so exporting an allocation moves it into
// a file of its own in the machine directory, mapped shared, still at the same
// device address (vgpu/memory.hpp). The importing process maps the same file at
// an address of its own. Both then read and write the same bytes, which is what
// the API promises, and a kernel in either process reaches them the way it
// reaches managed memory.
//
// The handle is 64 bytes, as CUDA's is, and carries only what the other process
// needs: the file's name inside the machine directory, its size, the device it
// belongs to, and the process that exported it.
namespace {

constexpr uint32_t kIpcMagic = 0x43504956;   // "VIPC"
constexpr uint32_t kIpcVersion = 1;

struct IpcMemPayload {
  uint32_t magic;
  uint32_t version;
  uint64_t size;
  uint32_t device;
  uint32_t pid;
  char id[40];       // file name, NUL-padded
};
static_assert(sizeof(IpcMemPayload) == CUDA_IPC_HANDLE_SIZE,
              "an IPC handle payload has to fit the handle CUDA defines");

struct IpcEventPayload {
  uint32_t magic;
  uint32_t version;
  uint32_t pid;
  uint32_t reserved;
  char unused[CUDA_IPC_HANDLE_SIZE - 16];
};
static_assert(sizeof(IpcEventPayload) == CUDA_IPC_HANDLE_SIZE, "same for an event handle");

std::string ipc_path(const char* id) { return vgpu::telemetry::default_path() + "/ipc-" + id; }

// Imported mappings, so close can undo exactly what open did.
std::mutex g_ipc_mu;
std::map<void*, int> g_ipc_open;   // pointer -> device it was mapped on

}  // namespace

VGPU_EXPORT cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* ptr) {
  return guard("cudaIpcGetMemHandle", [&](State& s) -> cudaError_t {
    if (!handle || !ptr) return cudaErrorInvalidValue;
    const auto addr = reinterpret_cast<uint64_t>(ptr);
    // Which device owns it, since the handle names it and the importer checks.
    int device = -1;
    for (int d = 0; d < s.rt->device_count(); ++d)
      if (s.rt->device(d).memory().owns(addr)) device = d;
    // cudaIpcGetMemHandle's documented errors do not include an
    // invalid-device-pointer, so a pointer that is not a device allocation is an
    // invalid value -- which is in that set.
    if (device < 0) return cudaErrorInvalidValue;
    static std::atomic<uint32_t> counter{0};
    char id[40] = {0};
    std::snprintf(id, sizeof id, "%x-%x", static_cast<unsigned>(::getpid()),
                  counter.fetch_add(1) + 1);
    IpcMemPayload p{};
    p.magic = kIpcMagic;
    p.version = kIpcVersion;
    p.device = static_cast<uint32_t>(device);
    p.pid = static_cast<uint32_t>(::getpid());
    std::memcpy(p.id, id, sizeof p.id);
    // Moves the allocation into the file and keeps this process's pointer
    // working; throws if `ptr` is not the base of a live allocation, which is
    // what CUDA refuses too.
    p.size = s.rt->device(device).memory().share(addr, ipc_path(id));
    std::memcpy(handle, &p, sizeof p);
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaIpcOpenMemHandle(void** ptr, cudaIpcMemHandle_t handle,
                                             unsigned int flags) {
  return guard("cudaIpcOpenMemHandle", [&](State& s) -> cudaError_t {
    if (!ptr) return cudaErrorInvalidValue;
    // The only documented flag, and it is required.
    if (flags != cudaIpcMemLazyEnablePeerAccess) return cudaErrorInvalidValue;
    IpcMemPayload p{};
    std::memcpy(&p, &handle, sizeof p);
    if (p.magic != kIpcMagic || p.version != kIpcVersion || !p.size) return cudaErrorInvalidValue;
    // CUDA does not support opening a handle in the process that exported it,
    // and neither does this: the exporter already has the memory mapped.
    if (p.pid == static_cast<uint32_t>(::getpid())) {
      if (!quiet())
        std::fprintf(stderr, "[vgpu] cudaIpcOpenMemHandle: this handle was exported by this "
                             "process; a process cannot import its own memory\n");
      return cudaErrorInvalidValue;
    }
    char id[sizeof p.id + 1] = {0};
    std::memcpy(id, p.id, sizeof p.id);
    const int device = p.device < static_cast<uint32_t>(s.rt->device_count())
                           ? static_cast<int>(p.device) : t_current_device;
    const uint64_t va = s.rt->device(device).memory().adopt(ipc_path(id), p.size);
    *ptr = reinterpret_cast<void*>(va);
    std::lock_guard<std::mutex> lock(g_ipc_mu);
    g_ipc_open[*ptr] = device;
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaIpcCloseMemHandle(void* ptr) {
  return guard("cudaIpcCloseMemHandle", [&](State& s) -> cudaError_t {
    if (!ptr) return cudaErrorInvalidValue;
    int device = -1;
    {
      std::lock_guard<std::mutex> lock(g_ipc_mu);
      auto it = g_ipc_open.find(ptr);
      if (it == g_ipc_open.end()) return cudaErrorInvalidValue;
      device = it->second;
      g_ipc_open.erase(it);
    }
    s.rt->device(device).memory().abandon(reinterpret_cast<uint64_t>(ptr));
    return cudaSuccess;
  });
}



// Graphs are captured and replayed by running the work inline, so a captured
// graph has no node list to walk. Report an empty graph rather than a count a
// caller would then try to read nodes out of.

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

// The array records of one device, dropped by cudaDeviceReset along with the
// memory behind them.
static void forget_arrays_on(int device) {
  std::erase_if(g_arrays, [device](const auto& kv) { return kv.second.device == device; });
}

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
    rec.device = t_current_device;
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
    map_host_everywhere(s, p, n);
    s.managed_allocs[p] = HostRange{n, t_current_device};
    *ptr = p;
    return cudaSuccess;
  });
}

// Prefetching and advice describe where pages should live. There is one memory
// here, so neither moves anything -- but both are state a program sets and
// reads back through cudaMemRangeGetAttribute, and both refuse what a driver
// refuses: memory that is not managed, and a device that does not exist. They
// used to return success for any pointer at all, which told a caller its hint
// had been accepted for memory the API does not accept hints about.
//
// CUDA 13 changed both to take a cudaMemLocation where 12 took an int device,
// so each needs the signature of the toolkit in use. The vendor header is
// included, and C linkage makes a mismatch a hard error rather than a subtle
// one -- which is how the last one of these was caught.
namespace {

// The managed allocation holding [p, p+n), or null. Advice is only for managed
// memory, and only within one allocation.
std::pair<void* const, HostRange>* managed_range(State& s, const void* p, size_t n) {
  const auto addr = reinterpret_cast<uint64_t>(p);
  for (auto& entry : s.managed_allocs) {
    const auto base = reinterpret_cast<uint64_t>(entry.first);
    if (addr >= base && addr + n <= base + entry.second.size && n) return &entry;
  }
  return nullptr;
}

cudaError_t advise(State& s, const void* p, size_t n, cudaMemoryAdvise kind, int device) {
  auto* range = managed_range(s, p, n);
  if (!range) return cudaErrorInvalidValue;
  const bool needs_device = kind == cudaMemAdviseSetAccessedBy || kind == cudaMemAdviseUnsetAccessedBy ||
                            kind == cudaMemAdviseSetPreferredLocation ||
                            kind == cudaMemAdviseUnsetPreferredLocation;
  if (needs_device && device != cudaCpuDeviceId &&
      (device < 0 || device >= s.rt->device_count()))
    return cudaErrorInvalidDevice;
  ManagedAdvice& a = s.managed_advice[range->first];
  const auto begin = reinterpret_cast<uint64_t>(p);
  const uint64_t end = begin + n;
  switch (kind) {
    case cudaMemAdviseSetReadMostly: a.read_mostly.set(begin, end, 1); return cudaSuccess;
    case cudaMemAdviseUnsetReadMostly: a.read_mostly.clear(begin, end); return cudaSuccess;
    case cudaMemAdviseSetPreferredLocation: a.preferred.set(begin, end, device); return cudaSuccess;
    case cudaMemAdviseUnsetPreferredLocation: a.preferred.clear(begin, end); return cudaSuccess;
    case cudaMemAdviseSetAccessedBy: a.accessed_by[device].set(begin, end, 1); return cudaSuccess;
    case cudaMemAdviseUnsetAccessedBy: a.accessed_by[device].clear(begin, end); return cudaSuccess;
    default: return cudaErrorInvalidValue;
  }
}

cudaError_t prefetch(State& s, const void* p, size_t n, int device) {
  auto* range = managed_range(s, p, n);
  if (!range) return cudaErrorInvalidValue;
  if (device != cudaCpuDeviceId && (device < 0 || device >= s.rt->device_count()))
    return cudaErrorInvalidDevice;
  const auto begin = reinterpret_cast<uint64_t>(p);
  s.managed_advice[range->first].last_prefetch.set(begin, begin + n, device);
  return cudaSuccess;
}

// One attribute of one range, as cudaMemRangeGetAttribute reports it: a value
// only when every byte of the range agrees, and the documented "no answer"
// otherwise -- 0 for read-mostly, an invalid device id for the locations.
cudaError_t range_attribute(State& s, void* data, size_t data_size, cudaMemRangeAttribute attr,
                            const void* p, size_t n) {
  if (!data || !data_size) return cudaErrorInvalidValue;
  auto* range = managed_range(s, p, n);
  if (!range) return cudaErrorInvalidValue;
  const ManagedAdvice& a = s.managed_advice[range->first];
  const auto begin = reinterpret_cast<uint64_t>(p);
  const uint64_t end = begin + n;
  switch (attr) {
    case cudaMemRangeAttributeReadMostly: {
      if (data_size != sizeof(int)) return cudaErrorInvalidValue;
      int v = 0;
      *static_cast<int*>(data) = a.read_mostly.uniform(begin, end, &v) ? 1 : 0;
      return cudaSuccess;
    }
    case cudaMemRangeAttributePreferredLocation: {
      if (data_size != sizeof(int)) return cudaErrorInvalidValue;
      int v = cudaInvalidDeviceId;
      if (!a.preferred.uniform(begin, end, &v)) v = cudaInvalidDeviceId;
      *static_cast<int*>(data) = v;
      return cudaSuccess;
    }
    case cudaMemRangeAttributeLastPrefetchLocation: {
      if (data_size != sizeof(int)) return cudaErrorInvalidValue;
      int v = cudaInvalidDeviceId;
      if (!a.last_prefetch.uniform(begin, end, &v)) v = cudaInvalidDeviceId;
      *static_cast<int*>(data) = v;
      return cudaSuccess;
    }
    case cudaMemRangeAttributeAccessedBy: {
      // An array of device ids, as many as fit; the rest are filled with the
      // invalid id, which is how a caller knows where the list ends.
      if (data_size % sizeof(int)) return cudaErrorInvalidValue;
      int* out = static_cast<int*>(data);
      const size_t slots = data_size / sizeof(int);
      size_t at = 0;
      for (const auto& [device, where] : a.accessed_by) {
        if (at == slots) break;
        int v = 0;
        if (where.uniform(begin, end, &v)) out[at++] = device;
      }
      for (; at < slots; ++at) out[at] = cudaInvalidDeviceId;
      return cudaSuccess;
    }
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace

#if CUDART_VERSION >= 13000
VGPU_EXPORT cudaError_t cudaMemPrefetchAsync(const void* p, size_t n, struct cudaMemLocation loc,
                                             unsigned int, cudaStream_t) {
  return guard("cudaMemPrefetchAsync", [&](State& s) -> cudaError_t {
    if (loc.type != cudaMemLocationTypeDevice && loc.type != cudaMemLocationTypeHost)
      return cudaErrorInvalidValue;
    return prefetch(s, p, n, loc.type == cudaMemLocationTypeHost ? cudaCpuDeviceId : loc.id);
  });
}
VGPU_EXPORT cudaError_t cudaMemAdvise(const void* p, size_t n, cudaMemoryAdvise kind,
                                      struct cudaMemLocation loc) {
  return guard("cudaMemAdvise", [&](State& s) -> cudaError_t {
    if (loc.type != cudaMemLocationTypeDevice && loc.type != cudaMemLocationTypeHost)
      return cudaErrorInvalidValue;
    return advise(s, p, n, kind, loc.type == cudaMemLocationTypeHost ? cudaCpuDeviceId : loc.id);
  });
}
#else
VGPU_EXPORT cudaError_t cudaMemPrefetchAsync(const void* p, size_t n, int device, cudaStream_t) {
  return guard("cudaMemPrefetchAsync", [&](State& s) { return prefetch(s, p, n, device); });
}
VGPU_EXPORT cudaError_t cudaMemAdvise(const void* p, size_t n, cudaMemoryAdvise kind, int device) {
  return guard("cudaMemAdvise", [&](State& s) { return advise(s, p, n, kind, device); });
}
#endif

VGPU_EXPORT cudaError_t cudaMemRangeGetAttribute(void* data, size_t data_size,
                                                 cudaMemRangeAttribute attr, const void* p,
                                                 size_t n) {
  return guard("cudaMemRangeGetAttribute",
               [&](State& s) { return range_attribute(s, data, data_size, attr, p, n); });
}

VGPU_EXPORT cudaError_t cudaMemRangeGetAttributes(void** data, size_t* data_sizes,
                                                  cudaMemRangeAttribute* attrs, size_t count,
                                                  const void* p, size_t n) {
  return guard("cudaMemRangeGetAttributes", [&](State& s) -> cudaError_t {
    if (!data || !data_sizes || !attrs || !count) return cudaErrorInvalidValue;
    for (size_t i = 0; i < count; ++i) {
      const cudaError_t rc = range_attribute(s, data[i], data_sizes[i], attrs[i], p, n);
      if (rc != cudaSuccess) return rc;
    }
    return cudaSuccess;
  });
}


// The extended launch form carries an attribute list. Two of them change what
// the grid does -- a cluster shape and a cooperative launch -- and are honoured;
// the rest are inert here, for reasons given below.
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
  if (cfg->numAttrs && !cfg->attrs) return cudaErrorInvalidValue;
  unsigned cluster[3] = {0, 0, 0};
  bool cooperative = false;
  for (unsigned i = 0; i < cfg->numAttrs; ++i) {
    const cudaLaunchAttribute& a = cfg->attrs[i];
    if (a.id == cudaLaunchAttributeClusterDimension) {
      cluster[0] = a.val.clusterDim.x;
      cluster[1] = a.val.clusterDim.y;
      cluster[2] = a.val.clusterDim.z;
    }
    // A cooperative launch is what cudaLaunchCooperativeKernel does, asked for
    // through the attribute list instead, and it is not inert: its blocks are
    // resident together and the kernel is handed the grid-barrier workspace
    // that grid.sync() needs. Dropping it -- which this did -- left
    // cooperative_groups without a workspace, so it trapped, and a correct
    // program failed.
    if (a.id == cudaLaunchAttributeCooperative) cooperative = a.val.cooperative != 0;
    // Every other attribute is inert here for a reason that is already true
    // elsewhere in this shim: priority and memory-sync domains need a stream
    // scheduler, access policy windows need a cache model, and programmatic
    // events need asynchrony. None of them changes what a kernel computes.
  }
  return launch_kernel_impl("cudaLaunchKernelEx", func, cfg->gridDim, cfg->blockDim, args,
                            cfg->dynamicSmemBytes, cfg->stream, cooperative,
                            {cluster[0], cluster[1], cluster[2]});
}

// Profiler control is a no-op: there is no external profiler attached, and a
// framework toggling it must not fail.
VGPU_EXPORT cudaError_t cudaProfilerStart(void) { return cudaSuccess; }
VGPU_EXPORT cudaError_t cudaProfilerStop(void) { return cudaSuccess; }

VGPU_EXPORT cudaError_t cudaStreamGetPriority(cudaStream_t stream, int* priority) {
  if (!priority) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_stream_mu);
  *priority = attribute_record(stream).priority;
  return cudaSuccess;
}

// A host callback is enqueued behind the stream's work. Work is synchronous
// here, so everything before it has already finished and it runs now.
VGPU_EXPORT cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* user) {
  if (!fn) return cudaErrorInvalidValue;
  if (capture_host_fn(stream, fn, user)) return cudaSuccess;   // a host node, run on launch
  fn(user);
  return cudaSuccess;
}

// cudaStreamGetCaptureInfo_v2 is defined with the graph machinery below: what
// it reports is the capture's own graph and dependency set.
VGPU_EXPORT cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  // The default streams are not a program's to destroy.
  if (is_default_stream(stream)) return cudaErrorInvalidResourceHandle;
  {
    std::lock_guard<std::mutex> lock(g_stream_mu);
    // A handle this runtime did not make may be a driver-API stream, which the
    // driver destroys; there is nothing here to free for it.
    g_streams.erase(static_cast<void*>(stream));
  }
  // A capture in progress on it ends with it. Left behind, it would be
  // inherited by the next stream that happened to get the same address.
  vgpu_drop_capture(stream);
  return cudaSuccess;
}
// Neither is permitted on a capturing stream, as documented: its work has not
// run and will not until the graph is launched. The capture is invalidated, as
// CUDA invalidates it, and the call says why.
VGPU_EXPORT cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  if (capture_refuse(stream, "cudaStreamSynchronize")) return cudaErrorStreamCaptureUnsupported;
  return cudaDeviceSynchronize();
}
VGPU_EXPORT cudaError_t cudaStreamQuery(cudaStream_t stream) {
  if (capture_refuse(stream, "cudaStreamQuery")) return cudaErrorStreamCaptureUnsupported;
  return cudaSuccess;
}

// cudaStreamWaitEvent is defined with the events it waits on.
// cudaStreamIsCapturing is defined with the CUDA Graph machinery below.

// Events keep the host clock at cudaEventRecord. Work is synchronous, so an
// event is complete the moment it is recorded, and the interval between two is
// the host time spent between them -- what a program timing a region with
// events measures on hardware too. This used to report 0 ms for every pair,
// recorded or not, so a benchmark divided by zero and a timing check could
// never fail. (The driver shim's cuEvent* already kept timestamps; this is the
// same record.) Handles are looked up, so a destroyed or made-up event is
// cudaErrorInvalidResourceHandle rather than a free() of something nobody
// allocated.
namespace {
struct RtEvent {
  bool timing = true;
  bool recorded = false;
  std::chrono::steady_clock::time_point when{};
  // Last recorded on a capturing stream: which capture, and the position that
  // stream had reached. A cudaStreamWaitEvent on it depends on that position --
  // joining the capture, or bringing the waiting stream into it -- and it cannot
  // be queried, synchronized or timed (cudaErrorCapturedEvent, as documented)
  // until it is recorded again outside a capture.
  bool captured = false;
  unsigned long long capture_id = 0;
  std::vector<void*> capture_deps;   // the capture's nodes, as opaque pointers here
};
std::mutex g_event_mu;
std::unordered_map<cudaEvent_t, std::unique_ptr<RtEvent>> g_events;
}  // namespace
// The capture side of events, defined with the graph machinery at file scope --
// declared outside this anonymous namespace so the declarations and the
// definitions are the same functions. Each returns false, or leaves a wait as a
// plain wait, when the stream is not capturing.
bool capture_position(cudaStream_t stream, unsigned long long* id, std::vector<void*>* deps);
bool capture_event_node(cudaStream_t stream, bool record, cudaEvent_t e, unsigned long long* id,
                        std::vector<void*>* deps);
cudaError_t capture_wait(cudaStream_t stream, bool captured, unsigned long long id,
                         const std::vector<void*>& deps);
bool capture_refuse(cudaStream_t stream, const char* what);
namespace {

RtEvent* find_event(cudaEvent_t e) {  // caller holds g_event_mu
  auto it = g_events.find(e);
  return it == g_events.end() ? nullptr : it->second.get();
}
}  // namespace

VGPU_EXPORT cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned int flags) {
  if (!e) return cudaErrorInvalidValue;
  auto rec = std::make_unique<RtEvent>();
  rec->timing = (flags & cudaEventDisableTiming) == 0;
  std::lock_guard<std::mutex> lock(g_event_mu);
  *e = reinterpret_cast<cudaEvent_t>(rec.get());
  g_events[*e] = std::move(rec);
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaEventCreate(cudaEvent_t* e) {
  return cudaEventCreateWithFlags(e, cudaEventDefault);
}
VGPU_EXPORT cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(g_event_mu);
  RtEvent* r = find_event(e);
  if (!r) return cudaErrorInvalidResourceHandle;
  // On a capturing stream an event records where the capture has got to, for a
  // later wait to depend on; it is not a node, and nothing has run. (The event
  // lock is taken before the graph lock, the order kept everywhere.)
  unsigned long long id = 0;
  std::vector<void*> deps;
  if (capture_position(stream, &id, &deps)) {
    r->captured = true;
    r->capture_id = id;
    r->capture_deps = std::move(deps);
    return cudaSuccess;
  }
  r->captured = false;
  r->capture_deps.clear();
  r->recorded = true;
  r->when = std::chrono::steady_clock::now();
  return cudaSuccess;
}

// cudaEventRecordExternal makes the record a node of its own when the stream
// is capturing -- the event is recorded each time the graph runs, which is how
// a graph is timed from outside. Without it, this is cudaEventRecord.
VGPU_EXPORT cudaError_t cudaEventRecordWithFlags(cudaEvent_t e, cudaStream_t stream,
                                                 unsigned int flags) {
  if (flags & ~static_cast<unsigned>(cudaEventRecordExternal)) return cudaErrorInvalidValue;
  if (!(flags & cudaEventRecordExternal)) return cudaEventRecord(e, stream);
  std::lock_guard<std::mutex> lock(g_event_mu);
  RtEvent* r = find_event(e);
  if (!r) return cudaErrorInvalidResourceHandle;
  unsigned long long id = 0;
  std::vector<void*> deps;
  if (capture_event_node(stream, /*record=*/true, e, &id, &deps)) {
    r->captured = true;   // a wait later in the capture depends on the node
    r->capture_id = id;
    r->capture_deps = std::move(deps);
    return cudaSuccess;
  }
  r->captured = false;
  r->capture_deps.clear();
  r->recorded = true;
  r->when = std::chrono::steady_clock::now();
  return cudaSuccess;
}

// What an event-record node does when its graph runs, and what a wait node
// does: record the event now; and check it exists, since everything before a
// wait has already finished in a synchronous engine. Neither consults capture
// state -- they are the graph running, not a stream being captured.
cudaError_t record_event_now(cudaEvent_t e) {
  std::lock_guard<std::mutex> lock(g_event_mu);
  RtEvent* r = find_event(e);
  if (!r) return cudaErrorInvalidResourceHandle;
  r->captured = false;
  r->capture_deps.clear();
  r->recorded = true;
  r->when = std::chrono::steady_clock::now();
  return cudaSuccess;
}
cudaError_t wait_event_now(cudaEvent_t e) {
  std::lock_guard<std::mutex> lock(g_event_mu);
  return find_event(e) ? cudaSuccess : cudaErrorInvalidResourceHandle;
}

// Outside a capture every operation has finished before its call returns, so a
// wait has nothing to wait for. Inside one it is how streams are joined: the
// waiting stream's next operation depends on where the event's stream had got
// to, and a stream not yet capturing joins the capture -- a fork. The rules on
// what may be waited on are CUDA's, each with the error it documents.
VGPU_EXPORT cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t e, unsigned int flags) {
  if (flags & ~static_cast<unsigned>(cudaEventWaitExternal)) return cudaErrorInvalidValue;
  bool captured = false;
  unsigned long long id = 0;
  std::vector<void*> deps;
  {
    std::lock_guard<std::mutex> lock(g_event_mu);
    const RtEvent* r = find_event(e);
    if (!r) return cudaErrorInvalidResourceHandle;
    captured = r->captured;
    id = r->capture_id;
    deps = r->capture_deps;
  }
  // cudaEventWaitExternal: a wait node of its own in the capture, on an event
  // from outside it.
  if (flags & cudaEventWaitExternal) {
    capture_event_node(stream, /*record=*/false, e, nullptr, nullptr);
    return cudaSuccess;
  }
  return capture_wait(stream, captured, id, deps);
}

// An event handle carries no state beyond its origin. Every operation here
// finishes before the call that started it returns, so an event another process
// records is complete by the time its handle can be read -- which makes a
// cross-process wait on it satisfied, rather than skipped.
VGPU_EXPORT cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) {
  if (!handle) return cudaErrorInvalidValue;
  {
    std::lock_guard<std::mutex> lock(g_event_mu);
    if (!find_event(event)) return cudaErrorInvalidResourceHandle;
  }
  IpcEventPayload p{};
  p.magic = kIpcMagic;
  p.version = kIpcVersion;
  p.pid = static_cast<uint32_t>(::getpid());
  std::memcpy(handle, &p, sizeof p);
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle) {
  if (!event) return cudaErrorInvalidValue;
  IpcEventPayload p{};
  std::memcpy(&p, &handle, sizeof p);
  if (p.magic != kIpcMagic || p.version != kIpcVersion) return cudaErrorInvalidValue;
  if (p.pid == static_cast<uint32_t>(::getpid())) return cudaErrorInvalidValue;
  // Already recorded: the work it stands for was complete before the exporting
  // process could hand the handle over.
  const cudaError_t rc = cudaEventCreateWithFlags(event, cudaEventDisableTiming);
  if (rc != cudaSuccess) return rc;
  std::lock_guard<std::mutex> lock(g_event_mu);
  if (RtEvent* r = find_event(*event)) {
    r->recorded = true;
    r->when = std::chrono::steady_clock::now();
  }
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaEventSynchronize(cudaEvent_t e) {
  std::lock_guard<std::mutex> lock(g_event_mu);
  const RtEvent* r = find_event(e);
  if (!r) return cudaErrorInvalidResourceHandle;
  return r->captured ? cudaErrorCapturedEvent : cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaEventQuery(cudaEvent_t e) {
  std::lock_guard<std::mutex> lock(g_event_mu);
  const RtEvent* r = find_event(e);
  if (!r) return cudaErrorInvalidResourceHandle;
  return r->captured ? cudaErrorCapturedEvent : cudaSuccess;   // it stands for work not yet run
}
VGPU_EXPORT cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
  if (!ms) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_event_mu);
  const RtEvent* a = find_event(start);
  const RtEvent* b = find_event(end);
  // Documented: an event never recorded, or created with
  // cudaEventDisableTiming, is cudaErrorInvalidResourceHandle.
  // cudaErrorNotReady is for recorded work that has not finished, which a
  // synchronous engine never leaves behind.
  if (a && b && (a->captured || b->captured)) return cudaErrorCapturedEvent;
  if (!a || !b || !a->recorded || !b->recorded || !a->timing || !b->timing)
    return cudaErrorInvalidResourceHandle;
  *ms = std::chrono::duration<float, std::milli>(b->when - a->when).count();
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaEventDestroy(cudaEvent_t e) {
  std::lock_guard<std::mutex> lock(g_event_mu);
  return g_events.erase(e) ? cudaSuccess : cudaErrorInvalidResourceHandle;
}

/* ===================================================================== */
/* Errors and versions                                                   */
/* ===================================================================== */

// A sticky error is not cleared by reading it: the context is still unusable.
VGPU_EXPORT cudaError_t cudaGetLastError(void) {
  cudaError_t e = g_last_error;
  g_last_error = cudaSuccess;
  g_async_error = cudaSuccess;
  const cudaError_t sticky = g_sticky_error.load();
  return sticky != cudaSuccess ? sticky : e;
}
VGPU_EXPORT cudaError_t cudaPeekAtLastError(void) {
  const cudaError_t sticky = g_sticky_error.load();
  return sticky != cudaSuccess ? sticky : g_last_error;
}

// Both name every code the runtime API declares, from the table shared with
// the driver shim (error_names.hpp). They used to name only the codes this
// shim returns, so a program printing the name of a real code it got from a
// library -- cudaErrorNoDevice, cudaErrorNotReady, cudaErrorPeerAccessAlreadyEnabled
// -- was told "cudaErrorUnknown", confidently wrong. A code no header declares
// gets the documented "unrecognized error code".
VGPU_EXPORT const char* cudaGetErrorString(cudaError_t error) {
  const vgpu::cuda::ErrorInfo* e = vgpu::cuda::find_error(static_cast<int>(error));
  return e && e->runtime_name ? e->text : "unrecognized error code";
}
VGPU_EXPORT const char* cudaGetErrorName(cudaError_t error) {
  const vgpu::cuda::ErrorInfo* e = vgpu::cuda::find_error(static_cast<int>(error));
  return e && e->runtime_name ? e->runtime_name : "unrecognized error code";
}

VGPU_EXPORT cudaError_t cudaDriverGetVersion(int* v) {
  if (v) *v = vgpu::driver_version();
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaRuntimeGetVersion(int* v) {
  if (v) *v = kRuntimeVersion;
  return cudaSuccess;
}

// Driver entry points: how a runtime program reaches a driver function without
// linking libcuda -- CUTLASS gets cuTensorMapEncodeTiled this way. The answer
// has to come from this simulator's libcuda, not from whatever libcuda.so.1 the
// loader would find by name: on a machine with an NVIDIA driver installed that
// is the real one, and a tensor map encoded by it means nothing here. So the
// library is the one sitting next to this libcudart.
namespace {
using GetProcAddressFn = int (*)(const char*, void**, int, unsigned long long, int*);
GetProcAddressFn simulator_get_proc_address() {
  static GetProcAddressFn fn = []() -> GetProcAddressFn {
    Dl_info self{};
    if (!dladdr(reinterpret_cast<void*>(&simulator_get_proc_address), &self) || !self.dli_fname)
      return nullptr;
    std::string dir = self.dli_fname;
    dir = dir.substr(0, dir.find_last_of('/') + 1);
    void* h = dlopen((dir + "libcuda.so.1").c_str(), RTLD_NOW | RTLD_GLOBAL);
    return h ? reinterpret_cast<GetProcAddressFn>(dlsym(h, "cuGetProcAddress_v2")) : nullptr;
  }();
  return fn;
}

cudaError_t driver_entry_point(const char* symbol, void** funcPtr, int version,
                               unsigned long long flags,
                               cudaDriverEntryPointQueryResult* driverStatus) {
  if (!symbol || !funcPtr) return cudaErrorInvalidValue;
  if (version > vgpu::driver_version()) return cudaErrorInvalidValue;
  *funcPtr = nullptr;
  GetProcAddressFn get = simulator_get_proc_address();
  if (!get) {
    std::fprintf(stderr, "[vgpu] cudaGetDriverEntryPoint: this simulator's libcuda.so.1 was not "
                         "found beside its libcudart\n");
    return cudaErrorNotSupported;
  }
  int status = 0;
  // The CUresult is not the question; the status says whether it was found.
  (void)get(symbol, funcPtr, version, flags, &status);
  if (!*funcPtr && status == 0) status = 1;   // CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND
  if (driverStatus) *driverStatus = static_cast<cudaDriverEntryPointQueryResult>(status);
  return cudaSuccess;
}
}  // namespace

VGPU_EXPORT cudaError_t cudaGetDriverEntryPoint(const char* symbol, void** funcPtr,
                                                unsigned long long flags,
                                                cudaDriverEntryPointQueryResult* driverStatus) {
  return driver_entry_point(symbol, funcPtr, CUDART_VERSION, flags, driverStatus);
}
#if CUDART_VERSION >= 12050
VGPU_EXPORT cudaError_t cudaGetDriverEntryPointByVersion(const char* symbol, void** funcPtr,
                                                         unsigned int cudaVersion,
                                                         unsigned long long flags,
                                                         cudaDriverEntryPointQueryResult* driverStatus) {
  return driver_entry_point(symbol, funcPtr, static_cast<int>(cudaVersion), flags, driverStatus);
}
#endif

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
  // How it was launched, which the replay has to repeat: a cooperative launch's
  // blocks are resident together and get the grid-barrier workspace grid.sync()
  // needs, and a cluster shape is what %cluster_* report. Dropping either would
  // run the kernel as something else.
  bool cooperative = false;
  std::array<uint32_t, 3> cluster{0, 0, 0};
  // Memcpy / Memset. Pointers are recorded, not the data behind them: a graph
  // reads whatever the buffers hold at replay time, which is what makes
  // replaying a captured step with new inputs meaningful.
  void* dst = nullptr;
  const void* src = nullptr;
  size_t bytes = 0;
  cudaMemcpyKind copy_kind = cudaMemcpyDefault;
  int fill_value = 0;
  // A copy or fill with a shape: `bytes` is one row, and there are `height`
  // rows `*_pitch` bytes apart in `depth` slices `*_slice` bytes apart. A
  // linear one is one row. A fill writes elements of `elem_size` bytes, each
  // the low `elem_size` bytes of `fill_value` -- a 4-byte fill of 2 writes 2
  // into every int, not the byte 2 into every byte.
  size_t height = 1, depth = 1;
  size_t dst_pitch = 0, src_pitch = 0, dst_slice = 0, src_slice = 0;
  unsigned elem_size = 1;
  // What the node was built or set with, which Get*Params hands back as given:
  // the shape above is how it runs, these are how it was asked for.
  cudaMemcpy3DParms copy_params{};
  cudaMemsetParams fill_params{};
  // Host: either the function a program handed cudaGraphAddHostNode, which
  // HostNodeGetParams has to hand back, or a vendor-library call that computes
  // on the CPU and left a closure behind. A closure is replayed by re-running
  // it, which re-reads device memory then -- so it sees what the graph's
  // kernels produced, exactly as the real library would.
  cudaHostFn_t host_fn = nullptr;
  void* host_user = nullptr;
  std::function<void()> host_op;
  // Event record and event wait nodes: which event.
  cudaEvent_t event = nullptr;
};

// A graph is a directed acyclic graph of nodes, which is what CUDA's own
// structure is and what the explicit API builds: a program adds nodes, says
// which depend on which, and the launch runs them in an order that respects
// that. Capture builds the same structure -- each recorded operation depends on
// the one before it, which is the order the stream would have run them in.
//
// A node's address is its handle, so nodes are held behind pointers and never
// move.
struct GraphNodeRec {
  cudaGraphNodeType type = cudaGraphNodeTypeKernel;
  RecordedLaunch work;                    // empty nodes carry none
  std::vector<GraphNodeRec*> deps;        // the nodes that must run first
  struct GraphRec* child = nullptr;       // a child-graph node's graph
  // Whether a launch runs this node. A disabled node keeps its place in the
  // dependency structure and does nothing, which is what CUDA documents: it is
  // an empty node in all but type, and its parameters are left alone. The flag
  // belongs to the instantiated graph, not to the graph it was built from.
  bool enabled = true;
};

struct GraphRec {
  std::vector<std::unique_ptr<GraphNodeRec>> nodes;
  // Child graphs a node points at, cloned when the node was added so a later
  // change to the original graph cannot change this one -- which is what CUDA
  // documents.
  std::vector<std::unique_ptr<GraphRec>> children;
  // Set when something happened during capture that this implementation cannot
  // record. Kernel launches, copies, fills and host-computed library calls are
  // all captured; anything else would run immediately and be missing from every
  // replay, so the capture is no longer a faithful record of the work and must
  // not be used.
  bool invalidated = false;
  const char* invalidated_by = nullptr;

  // Adds a node with the given predecessors. Returns its handle.
  GraphNodeRec* add(cudaGraphNodeType type, RecordedLaunch work,
                    const std::vector<GraphNodeRec*>& deps) {
    nodes.push_back(std::make_unique<GraphNodeRec>());
    GraphNodeRec* n = nodes.back().get();
    n->type = type;
    n->work = std::move(work);
    n->deps = deps;
    return n;
  }
  bool holds(const GraphNodeRec* n) const {
    for (const auto& up : nodes)
      if (up.get() == n) return true;
    return false;
  }
};

// Deep copy, with dependencies remapped onto the new nodes. `mapping` receives
// old -> new for every node, which is what cudaGraphNodeFindInClone answers
// from.
std::unique_ptr<GraphRec> clone_graph(const GraphRec& src,
                                      std::map<const GraphNodeRec*, GraphNodeRec*>* mapping) {
  auto out = std::make_unique<GraphRec>();
  out->invalidated = src.invalidated;
  out->invalidated_by = src.invalidated_by;
  std::map<const GraphNodeRec*, GraphNodeRec*> local;
  for (const auto& up : src.nodes) {
    out->nodes.push_back(std::make_unique<GraphNodeRec>());
    GraphNodeRec* n = out->nodes.back().get();
    n->type = up->type;
    n->work = up->work;
    n->enabled = up->enabled;
    local[up.get()] = n;
  }
  for (const auto& up : src.nodes) {
    GraphNodeRec* n = local[up.get()];
    for (const GraphNodeRec* d : up->deps) n->deps.push_back(local[d]);
    if (up->child) {
      out->children.push_back(clone_graph(*up->child, nullptr));
      n->child = out->children.back().get();
    }
  }
  if (mapping) *mapping = local;
  return out;
}

/* ---- memory a graph owns ------------------------------------------------------
 *
 * An allocation node allocates when the graph reaches it and a free node frees
 * it, so a graph that needs scratch space carries it instead of the program
 * holding it for the graph's whole life. CUDA documents one thing about the
 * address that shapes the whole implementation: it is fixed, across every
 * instantiation and every launch. So the address space is reserved when the node
 * is built and kept until the allocation is gone for good, and only the physical
 * memory behind it comes and goes -- which is exactly what the mapping API this
 * engine already has is for.
 *
 * That gives the accounting its meaning, and the two attribute pairs their
 * difference: `used` is what a graph is holding right now, between an allocation
 * node running and the free that ends it, and `reserved` is what the device has
 * actually handed over, which outlives a free because the memory is kept for the
 * next launch. cudaDeviceGraphMemTrim is what gives that back, and the address
 * survives it: the next launch maps new memory at the same place.
 */
struct GraphAlloc {
  uint64_t va = 0;              // the fixed address, reserved for its whole life
  uint64_t reserved_size = 0;   // rounded up to the mapping granularity
  size_t size = 0;              // what the program asked for
  int device = 0;
  void* owner = nullptr;        // handle of the graph whose node created it
  bool freed_in_owner = false;  // a free node in the owning graph ends it
  bool freed_elsewhere = false; // a free node in another graph ends it
  bool mapped = false;          // physical memory is behind the address
  bool in_use = false;          // allocated and not yet freed
  // The owning graph has been destroyed, so no launch can allocate here again
  // and the address itself can go -- which a trim does, or the free that ends
  // the allocation if the program is still holding it.
  bool owner_gone = false;
};

struct GraphMemStats {
  uint64_t used = 0, used_high = 0, reserved = 0, reserved_high = 0;
};

std::mutex g_graph_mem_mu;      // a leaf: nothing else is taken while it is held
std::map<uint64_t, GraphAlloc> g_graph_allocs;
std::map<int, GraphMemStats> g_graph_mem;
// Which graph an instantiated graph came from, and whether it was asked to free
// what a previous launch left allocated. Both are needed only by graphs that own
// memory: one graph that owns memory may have one instantiation at a time, and
// the auto-free flag is about its allocations.
std::map<void*, void*> g_exec_source;        // exec handle -> graph handle
std::set<void*> g_exec_auto_free;            // execs instantiated with the flag

uint64_t round_up_to(uint64_t v, uint64_t to) { return (v + to - 1) / to * to; }

// Whether a graph holds an allocation or free node, which brings the
// restrictions CUDA documents with it: such a graph cannot be cloned, cannot be
// a child of another graph, and cannot have nodes or edges taken out of it.
// Every one of those would leave an allocation whose address nothing owns.
bool holds_graph_memory(const GraphRec& g) {
  for (const auto& up : g.nodes) {
    if (up->type == cudaGraphNodeTypeMemAlloc || up->type == cudaGraphNodeTypeMemFree) return true;
    if (up->child && holds_graph_memory(*up->child)) return true;
  }
  return false;
}

// The nodes in an order that runs every node after everything it depends on.
// Empty when the dependencies contain a cycle, which is how adding one is
// refused.
std::vector<GraphNodeRec*> topological_order(const GraphRec& g) {
  std::map<const GraphNodeRec*, size_t> remaining;
  for (const auto& up : g.nodes) remaining[up.get()] = up->deps.size();
  std::vector<GraphNodeRec*> out, ready;
  for (const auto& up : g.nodes)
    if (up->deps.empty()) ready.push_back(up.get());
  while (!ready.empty()) {
    GraphNodeRec* n = ready.back();
    ready.pop_back();
    out.push_back(n);
    for (const auto& up : g.nodes) {
      if (std::find(up->deps.begin(), up->deps.end(), n) == up->deps.end()) continue;
      if (--remaining[up.get()] == 0) ready.push_back(up.get());
    }
  }
  if (out.size() != g.nodes.size()) return {};   // a cycle
  return out;
}

std::mutex g_graph_mu;
std::unordered_map<void*, std::unique_ptr<GraphRec>> g_graphs;      // graph handles
// A child graph a caller asked to look at: the node owns it, so this holds a
// handle for it without owning it. Destroying one through cudaGraphDestroy is
// refused, as CUDA refuses it -- the node's graph is the node's.
std::unordered_map<void*, GraphRec*> g_borrowed_graphs;
std::unordered_map<void*, std::unique_ptr<GraphRec>> g_graph_execs; // instantiated graphs
// Capture sequences, and the streams taking part in them. A sequence owns the
// graph it builds until cudaStreamEndCapture hands it over. Every stream taking
// part -- the one that began it, and any that joined by waiting on an event
// recorded in it -- has its own position in it: the nodes its next operation
// depends on. That is what lets two streams build two branches of one graph and
// join them again, which is how a multi-stream program is captured.
struct Capture {
  std::unique_ptr<GraphRec> graph;
  unsigned long long id = 0;
  void* origin = nullptr;   // the stream that began it; only it can end it
};
struct StreamCapture {
  Capture* cap = nullptr;
  // Normally the one operation this stream captured last. A library splicing
  // its own nodes in sets it with cudaStreamUpdateCaptureDependencies, and a
  // wait on an event adds what that event captured.
  std::vector<GraphNodeRec*> deps;
  std::vector<cudaGraphNode_t> deps_out;   // what cudaStreamGetCaptureInfo hands out
};
std::unordered_map<void*, std::unique_ptr<Capture>> g_captures;   // by the stream that began each
std::unordered_map<void*, StreamCapture> g_stream_capture;        // every stream taking part

// The capture state of `stream`, or nullptr. Caller holds g_graph_mu.
StreamCapture* stream_capture(cudaStream_t stream) {
  const auto it = g_stream_capture.find(reinterpret_cast<void*>(stream));
  return it == g_stream_capture.end() ? nullptr : &it->second;
}

// The graph `stream` is capturing into, or nullptr when it is not capturing.
GraphRec* capture_target(cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  const StreamCapture* sc = stream_capture(stream);
  return sc ? sc->cap->graph.get() : nullptr;
}

// Adds a captured operation: it depends on the stream's position, and becomes
// it. Caller holds g_graph_mu.
GraphNodeRec* capture_add(StreamCapture& sc, cudaGraphNodeType type, RecordedLaunch work) {
  GraphNodeRec* n = sc.cap->graph->add(type, std::move(work), sc.deps);
  sc.deps.assign(1, n);
  return n;
}

// Records `work` on `stream` if it is capturing; says whether it was.
bool capture_record(cudaStream_t stream, cudaGraphNodeType type, RecordedLaunch work) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  StreamCapture* sc = stream_capture(stream);
  if (!sc) return false;
  capture_add(*sc, type, std::move(work));
  return true;
}

// Every stream taking part in `cap` leaves it. Caller holds g_graph_mu.
void leave_capture(const Capture* cap) {
  std::erase_if(g_stream_capture, [cap](const auto& kv) { return kv.second.cap == cap; });
}

// Every capture sequence gets its own id, as CUDA documents -- a library uses
// it to tell one capture from the next on the same stream. Zero is never one.
std::atomic<unsigned long long> g_next_capture_id{1};

}  // namespace

namespace {
// Defined with the graph-building calls; the capture hooks below use them too,
// so a captured copy and a built one are the same node.
cudaError_t fill_memcpy_work(RecordedLaunch* w, const cudaMemcpy3DParms* p);
cudaError_t fill_memset_work(RecordedLaunch* w, const cudaMemsetParams* p);
cudaMemcpy3DParms linear_copy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
}  // namespace

// Records a copy or a fill during capture. Returns true when it was recorded,
// in which case the caller must not perform it now -- it belongs to the graph.
// A copy of any shape: one node, as CUDA makes it. `*rc` is what the caller
// should return when the copy was recorded -- a shape a node cannot carry fails
// the call rather than being run now and missing from every replay.
bool vgpu_record_copy_if_capturing(const cudaMemcpy3DParms& p, cudaStream_t stream,
                                   cudaError_t* rc) {
  if (!capture_active(stream)) return false;
  RecordedLaunch r;
  *rc = fill_memcpy_work(&r, &p);
  if (*rc == cudaSuccess) capture_record(stream, cudaGraphNodeTypeMemcpy, std::move(r));
  return true;
}

bool vgpu_record_fill_if_capturing(const cudaMemsetParams& p, cudaStream_t stream,
                                   cudaError_t* rc) {
  if (!capture_active(stream)) return false;
  RecordedLaunch r;
  *rc = fill_memset_work(&r, &p);
  if (*rc == cudaSuccess) capture_record(stream, cudaGraphNodeTypeMemset, std::move(r));
  return true;
}

bool capture_active(cudaStream_t stream) { return capture_target(stream) != nullptr; }

bool vgpu_record_memcpy_if_capturing(void* dst, const void* src, size_t bytes,
                                     cudaMemcpyKind kind, cudaStream_t stream) {
  if (bytes == 0) return capture_target(stream) != nullptr;   // nothing to record or do
  cudaError_t rc = cudaSuccess;
  return vgpu_record_copy_if_capturing(linear_copy(dst, src, bytes, kind), stream, &rc);
}

bool vgpu_record_memset_if_capturing(void* dst, int value, size_t bytes, cudaStream_t stream) {
  if (bytes == 0) return capture_target(stream) != nullptr;
  cudaMemsetParams p{};
  p.dst = dst;
  p.value = static_cast<unsigned>(value);
  p.elementSize = 1;
  p.width = bytes;
  p.height = 1;
  cudaError_t rc = cudaSuccess;
  return vgpu_record_fill_if_capturing(p, stream, &rc);
}

// Records a host-computed library call during capture. Returns true when it was
// recorded, in which case the caller must not do the work now.
bool vgpu_record_host_op_if_capturing(cudaStream_t stream, std::function<void()> op) {
  if (!capture_active(stream)) return false;
  RecordedLaunch r;
  r.kind = RecordedLaunch::Kind::Host;
  r.host_op = std::move(op);
  return capture_record(stream, cudaGraphNodeTypeHost, std::move(r));
}

void vgpu_drop_capture(cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  StreamCapture* sc = stream_capture(stream);
  if (!sc) return;
  Capture* cap = sc->cap;
  if (cap->origin != reinterpret_cast<void*>(stream)) {   // a stream that joined just leaves
    g_stream_capture.erase(reinterpret_cast<void*>(stream));
    return;
  }
  g_borrowed_graphs.erase(cap->graph.get());   // the stream that began it takes it along
  leave_capture(cap);
  g_captures.erase(reinterpret_cast<void*>(stream));
}

// Marks any in-flight capture as unusable. Called by the operations that this
// implementation cannot record, so a capture that would silently omit work
// fails at cudaStreamEndCapture instead of replaying an incomplete graph.
void vgpu_invalidate_capture(const char* what) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  for (auto& entry : g_captures)
    if (entry.second && !entry.second->graph->invalidated) {
      entry.second->graph->invalidated = true;
      entry.second->graph->invalidated_by = what;
    }
}

// Records a launch during capture. Returns true if it was recorded (and must
// therefore NOT execute now).
bool vgpu_record_launch_if_capturing(const void* func, dim3 grid, dim3 block, void** args,
                                     size_t sharedMem, cudaStream_t stream,
                                     const std::vector<uint32_t>& param_sizes, bool cooperative,
                                     std::array<uint32_t, 3> cluster) {
  if (!capture_active(stream)) return false;
  RecordedLaunch rl;
  rl.func = func;
  rl.grid = grid;
  rl.block = block;
  rl.shared = sharedMem;
  rl.cooperative = cooperative;
  rl.cluster = cluster;
  rl.arg_bytes.resize(param_sizes.size());
  for (size_t i = 0; i < param_sizes.size(); ++i) {
    rl.arg_bytes[i].resize(param_sizes[i]);
    std::memcpy(rl.arg_bytes[i].data(), args[i], param_sizes[i]);
  }
  return capture_record(stream, cudaGraphNodeTypeKernel, std::move(rl));
}

VGPU_EXPORT cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
  (void)mode;
  // Kernel launches, copies and fills are all recorded, so a captured region
  // replays the work it actually contained. Anything still unrecordable marks
  // the capture invalid and cudaStreamEndCapture reports it, rather than
  // handing back a graph that silently omits work.
  // Not on the legacy stream, as documented: it synchronizes with every other
  // stream, so everything would become part of the capture.
  if (stream == nullptr || stream == cudaStreamLegacy) return cudaErrorStreamCaptureUnsupported;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  if (stream_capture(stream)) return cudaErrorIllegalState;   // already capturing
  auto cap = std::make_unique<Capture>();
  cap->graph = std::make_unique<GraphRec>();
  cap->id = g_next_capture_id.fetch_add(1);
  cap->origin = reinterpret_cast<void*>(stream);
  // The graph being captured into is a handle a program can use before the
  // capture ends: cudaStreamGetCaptureInfo hands it out, and a library adds its
  // own nodes to it. The capture owns it, so it is lent rather than given --
  // cudaGraphDestroy refuses it -- and it keeps its address when the capture
  // ends, so the handle is the same graph cudaStreamEndCapture returns.
  g_borrowed_graphs[cap->graph.get()] = cap->graph.get();
  g_stream_capture[reinterpret_cast<void*>(stream)] = StreamCapture{cap.get(), {}, {}};
  g_captures[reinterpret_cast<void*>(stream)] = std::move(cap);
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  StreamCapture* sc = stream_capture(stream);
  if (!sc) return cudaErrorIllegalState;   // not capturing
  // Only the stream that began a capture ends it; one that joined it cannot.
  if (sc->cap->origin != reinterpret_cast<void*>(stream)) return cudaErrorStreamCaptureUnmatched;
  Capture* cap = sc->cap;
  // Every stream that joined must have been joined back: whatever it did last
  // has to come before where the stream that began the capture is now. A fork
  // left dangling is work that would run after the graph was declared done.
  std::set<const GraphNodeRec*> before_end;
  {
    std::vector<const GraphNodeRec*> todo(sc->deps.begin(), sc->deps.end());
    while (!todo.empty()) {
      const GraphNodeRec* n = todo.back();
      todo.pop_back();
      if (!before_end.insert(n).second) continue;
      for (const GraphNodeRec* d : n->deps) todo.push_back(d);
    }
  }
  bool unjoined = false;
  for (const auto& [other, osc] : g_stream_capture) {
    if (osc.cap != cap || other == reinterpret_cast<void*>(stream)) continue;
    for (const GraphNodeRec* n : osc.deps)
      if (!before_end.count(n)) unjoined = true;
  }
  std::unique_ptr<GraphRec> graph = std::move(cap->graph);
  leave_capture(cap);
  g_captures.erase(reinterpret_cast<void*>(stream));
  g_borrowed_graphs.erase(graph.get());   // no longer lent: returned, or dropped below
  if (unjoined) {
    if (pGraph) *pGraph = nullptr;
    return cudaErrorStreamCaptureUnjoined;
  }
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
                                             unsigned long long flags) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto it = g_graphs.find(static_cast<void*>(graph));
  if (it == g_graphs.end()) return cudaErrorInvalidValue;
  if (topological_order(*it->second).empty() && !it->second->nodes.empty())
    return cudaErrorInvalidValue;                      // a cycle cannot be instantiated
  if (holds_graph_memory(*it->second)) {
    // One instantiation at a time, as documented: two would each believe they
    // own the allocation, and the second to free it would free it twice.
    for (const auto& [exec_handle, source] : g_exec_source)
      if (source == static_cast<void*>(graph)) return cudaErrorInvalidValue;
  }
  auto exec = clone_graph(*it->second, nullptr);        // a snapshot, as CUDA takes
  void* handle = exec.get();
  g_graph_execs[handle] = std::move(exec);
  g_exec_source[handle] = static_cast<void*>(graph);
  if (flags & cudaGraphInstantiateFlagAutoFreeOnLaunch) g_exec_auto_free.insert(handle);
  if (pExec) *pExec = static_cast<cudaGraphExec_t>(handle);
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* pExec, cudaGraph_t graph,
                                                      unsigned long long flags) {
  return cudaGraphInstantiate(pExec, graph, flags);
}

// Runs one node's work. Nodes with no work of their own (empty ones) do
// nothing but order the nodes around them.
static cudaError_t run_graph_node(GraphNodeRec* n, cudaStream_t stream);

namespace {

// Puts physical memory behind a graph allocation's address, which is what its
// node does when a launch reaches it. The address was reserved when the node was
// built and does not change; on the second and later launches the memory is
// usually still mapped, because a free keeps it for the next launch.
cudaError_t graph_mem_alloc_run(uint64_t va) {
  return guard("cudaGraphLaunch", [&](State& s) -> cudaError_t {
    std::lock_guard<std::mutex> lock(g_graph_mem_mu);
    const auto it = g_graph_allocs.find(va);
    if (it == g_graph_allocs.end()) return cudaErrorInvalidValue;
    GraphAlloc& a = it->second;
    if (a.in_use) return cudaSuccess;   // a launch of a graph already holding it
    vgpu::MemoryManager& mm = s.rt->device(a.device).memory();
    if (!a.mapped) {
      const uint64_t handle = mm.create_handle(a.reserved_size);
      mm.map(a.va, a.reserved_size, 0, handle);
      mm.set_access(a.va, a.reserved_size, true, true);
      mm.release_handle(handle);   // the mapping holds it now
      a.mapped = true;
      GraphMemStats& st_ = g_graph_mem[a.device];
      st_.reserved += a.reserved_size;
      st_.reserved_high = std::max(st_.reserved_high, st_.reserved);
    }
    a.in_use = true;
    GraphMemStats& st_ = g_graph_mem[a.device];
    st_.used += a.size;
    st_.used_high = std::max(st_.used_high, st_.used);
    return cudaSuccess;
  });
}

// Ends a graph allocation's life as an allocation, which is what its free node
// does. The memory stays mapped, because the next launch of the same graph will
// allocate at the same address and CUDA documents that address as fixed; what
// gives it back is cudaDeviceGraphMemTrim.
cudaError_t graph_mem_free_run(uint64_t va) {
  std::lock_guard<std::mutex> lock(g_graph_mem_mu);
  const auto it = g_graph_allocs.find(va);
  if (it == g_graph_allocs.end()) return cudaErrorInvalidValue;
  GraphAlloc& a = it->second;
  if (!a.in_use) return cudaSuccess;
  a.in_use = false;
  GraphMemStats& st_ = g_graph_mem[a.device];
  st_.used -= std::min<uint64_t>(st_.used, a.size);
  return cudaSuccess;
}

}  // namespace

// Every node of a graph, in an order that respects its dependencies. With one
// synchronous engine any such order is a correct execution of the graph, and a
// program that depended on more than the order it asked for would be depending
// on something CUDA does not promise either.
static cudaError_t run_graph(GraphRec& g, cudaStream_t stream) {
  const std::vector<GraphNodeRec*> order = topological_order(g);
  if (order.size() != g.nodes.size()) return cudaErrorInvalidValue;
  for (GraphNodeRec* n : order) {
    const cudaError_t rc = run_graph_node(n, stream);
    if (rc != cudaSuccess) return rc;
  }
  return cudaSuccess;
}

// A copy node: every row of every slice, each through the same path a linear
// copy takes, so the device each side belongs to is resolved per row.
static cudaError_t replay_copy(const RecordedLaunch& rl) {
  for (size_t z = 0; z < rl.depth; ++z)
    for (size_t y = 0; y < rl.height; ++y) {
      const cudaError_t e = cudaMemcpy(
          static_cast<char*>(rl.dst) + z * rl.dst_slice + y * rl.dst_pitch,
          static_cast<const char*>(rl.src) + z * rl.src_slice + y * rl.src_pitch, rl.bytes,
          rl.copy_kind);
      if (e != cudaSuccess) return e;
    }
  return cudaSuccess;
}

// A fill node: each row set to repeated elements of the node's width.
static cudaError_t replay_fill(const RecordedLaunch& rl) {
  return guard("cudaGraphLaunch", [&](State& s) -> cudaError_t {
    const uint32_t value = static_cast<uint32_t>(rl.fill_value);
    uint8_t pattern[4];
    for (unsigned i = 0; i < 4; ++i) pattern[i] = static_cast<uint8_t>(value >> (8 * i));
    for (size_t y = 0; y < rl.height; ++y) {
      void* row = static_cast<char*>(rl.dst) + y * rl.dst_pitch;
      owner_memory(s, row).fill(reinterpret_cast<uint64_t>(row), pattern, rl.elem_size, rl.bytes);
    }
    return cudaSuccess;
  });
}

static cudaError_t run_graph_node(GraphNodeRec* n, cudaStream_t stream) {
  RecordedLaunch& rl = n->work;
  if (!n->enabled) return cudaSuccess;   // switched off: it still orders its neighbours
  switch (n->type) {
    case cudaGraphNodeTypeEmpty:
      return cudaSuccess;
    case cudaGraphNodeTypeMemcpy:
      return replay_copy(rl);
    case cudaGraphNodeTypeMemset:
      return replay_fill(rl);
    case cudaGraphNodeTypeHost:
      // The function a program gave the node, or a captured library's closure.
      if (rl.host_fn) rl.host_fn(rl.host_user);
      else if (rl.host_op) rl.host_op();
      return cudaSuccess;
    case cudaGraphNodeTypeMemAlloc:
      return graph_mem_alloc_run(reinterpret_cast<uint64_t>(rl.dst));
    case cudaGraphNodeTypeMemFree:
      return graph_mem_free_run(reinterpret_cast<uint64_t>(rl.dst));
    case cudaGraphNodeTypeEventRecord:
      return record_event_now(rl.event);
    case cudaGraphNodeTypeWaitEvent:
      // Everything before this node has already finished -- one synchronous
      // engine, and a launch runs the nodes in dependency order -- so the wait
      // is satisfied rather than skipped. An event nothing has recorded is a
      // no-op here for the same reason CUDA documents it as one.
      return wait_event_now(rl.event);
    case cudaGraphNodeTypeGraph:
      return n->child ? run_graph(*n->child, stream) : cudaSuccess;
    default: {
      std::vector<void*> ptrs(rl.arg_bytes.size());
      for (size_t i = 0; i < rl.arg_bytes.size(); ++i) ptrs[i] = rl.arg_bytes[i].data();
      return launch_kernel_impl("cudaGraphLaunch", rl.func, rl.grid, rl.block, ptrs.data(),
                                rl.shared, stream, rl.cooperative, rl.cluster);
    }
  }
}

VGPU_EXPORT cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
  std::unique_ptr<GraphRec> replay;
  bool auto_free = false;
  void* source = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    auto it = g_graph_execs.find(static_cast<void*>(exec));
    if (it == g_graph_execs.end()) return cudaErrorInvalidValue;
    // Launched on a capturing stream, a graph is captured as a child-graph node:
    // it runs when the graph being captured runs, not now. A graph that owns
    // memory cannot be a child, as documented, so that one refuses.
    if (StreamCapture* sc = stream_capture(stream)) {
      if (holds_graph_memory(*it->second)) {
        GraphRec& g = *sc->cap->graph;
        if (!g.invalidated) {
          g.invalidated = true;
          g.invalidated_by = "cudaGraphLaunch of a graph that owns memory";
        }
        return cudaErrorNotSupported;
      }
      GraphRec& g = *sc->cap->graph;
      g.children.push_back(clone_graph(*it->second, nullptr));
      GraphNodeRec* n = capture_add(*sc, cudaGraphNodeTypeGraph, RecordedLaunch{});
      n->child = g.children.back().get();
      return cudaSuccess;
    }
    replay = clone_graph(*it->second, nullptr);   // run without holding the lock
    auto_free = g_exec_auto_free.count(static_cast<void*>(exec)) != 0;
    if (const auto src = g_exec_source.find(static_cast<void*>(exec)); src != g_exec_source.end())
      source = src->second;
  }
  // Instantiated with cudaGraphInstantiateFlagAutoFreeOnLaunch: what a previous
  // launch allocated and did not free is freed before this one runs, which is
  // what lets a graph with allocation nodes and no free nodes be relaunched.
  if (auto_free && source) {
    std::lock_guard<std::mutex> lock(g_graph_mem_mu);
    for (auto& [va, a] : g_graph_allocs) {
      if (a.owner != source || !a.in_use) continue;
      a.in_use = false;
      GraphMemStats& st_ = g_graph_mem[a.device];
      st_.used -= std::min<uint64_t>(st_.used, a.size);
    }
  }
  return run_graph(*replay, stream);
}

/* ---- building a graph node by node -------------------------------------------
 *
 * The other way to make a graph: instead of capturing a stream, a program
 * creates an empty graph, adds nodes and says which depend on which. This is
 * how a framework that knows its own dependency structure builds one --
 * cuDNN's and TensorRT's graphs, and anything that reuses a graph with new
 * parameters through cudaGraphExecKernelNodeSetParams.
 *
 * A node's handle is its address. Nodes are owned by their graph and outlive
 * nothing: destroying the graph destroys them.
 */
namespace {

// The parameter sizes of a registered kernel stub, which a graph node needs to
// copy argument values the way a launch does. Returns false for a stub this
// process never registered, or one whose module has no PTX.
//
// This takes the runtime's own lock, so a caller must not already hold
// g_graph_mu: a launch inside a capture holds the runtime lock and then takes
// the graph lock, and the reverse order in the other direction is a deadlock
// between a thread building a graph and a thread launching a kernel. The rule
// for the whole file is the runtime's lock first, the graph lock second.
bool kernel_param_sizes(const void* func, std::vector<uint32_t>* out) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (!s.rt) return false;
  auto it = s.kernels.find(func);
  if (it == s.kernels.end() || !it->second.mod || it->second.mod->ptx.empty()) return false;
  const uint64_t mid = module_on_current(s, *it->second.mod);
  const vgpu::ptx::EntryFn* fn = current(s).get_function(mid, it->second.entry_name);
  if (!fn) return false;
  out->resize(fn->params.size());
  for (size_t i = 0; i < fn->params.size(); ++i) (*out)[i] = fn->params[i].size;
  return true;
}

// The graph a node belongs to: a graph a program owns, or one it has been lent
// -- a child graph's own graph, or the graph a stream is capturing into. Nodes
// of a lent graph are handles like any other, and a query that only looked at
// owned graphs refused them. Caller holds g_graph_mu.
GraphRec* owner_of(const GraphNodeRec* n) {
  if (!n) return nullptr;
  for (const auto& [handle, g] : g_graphs)
    if (g->holds(n)) return g.get();
  for (const auto& [handle, g] : g_borrowed_graphs)
    if (g->holds(n)) return g;
  return nullptr;
}

GraphRec* graph_from(cudaGraph_t h) {   // caller holds g_graph_mu
  if (const auto it = g_graphs.find(static_cast<void*>(h)); it != g_graphs.end())
    return it->second.get();
  const auto borrowed = g_borrowed_graphs.find(static_cast<void*>(h));
  return borrowed == g_borrowed_graphs.end() ? nullptr : borrowed->second;
}

// Checks the dependency list a caller passed: every node has to belong to this
// graph, which is what stops a dependency on another graph's node.
bool deps_ok(const GraphRec& g, const cudaGraphNode_t* deps, size_t count,
             std::vector<GraphNodeRec*>* out) {
  for (size_t i = 0; i < count; ++i) {
    auto* n = reinterpret_cast<GraphNodeRec*>(deps[i]);
    if (!n || !g.holds(n)) return false;
    out->push_back(n);
  }
  return true;
}

// Copies a kernel node's parameters out of the API's struct.
cudaError_t fill_kernel_work(RecordedLaunch* w, const cudaKernelNodeParams* p) {
  if (!p || !p->func) return cudaErrorInvalidValue;
  w->kind = RecordedLaunch::Kind::Kernel;
  w->func = p->func;
  w->grid = p->gridDim;
  w->block = p->blockDim;
  w->shared = p->sharedMemBytes;
  // The parameter values are copied, not the pointers: a graph launched later
  // has to use what the arguments were when the node was set up, which is the
  // same rule capture follows.
  std::vector<uint32_t> sizes;
  if (!kernel_param_sizes(p->func, &sizes)) return cudaErrorInvalidDeviceFunction;
  w->arg_bytes.assign(sizes.size(), {});
  for (size_t i = 0; i < sizes.size(); ++i) {
    w->arg_bytes[i].resize(sizes[i]);
    if (!p->kernelParams || !p->kernelParams[i]) return cudaErrorInvalidValue;
    std::memcpy(w->arg_bytes[i].data(), p->kernelParams[i], sizes[i]);
  }
  return cudaSuccess;
}

// A copy node's work from a 3D copy's parameters: the rows and slices of a
// pitched region, between two pointers. Positions are in bytes for pointers,
// as CUDA defines them, and fold into the start address. CUDA arrays are not
// carried by a node here and are refused by name.
cudaError_t fill_memcpy_work(RecordedLaunch* w, const cudaMemcpy3DParms* p) {
  if (!p) return cudaErrorInvalidValue;
  if (p->srcArray || p->dstArray) return cudaErrorNotSupported;   // a CUDA array in a node
  if (p->srcPtr.ptr == nullptr || p->dstPtr.ptr == nullptr) return cudaErrorInvalidValue;
  const cudaExtent& e = p->extent;
  if (e.width == 0 || e.height == 0 || e.depth == 0) return cudaErrorInvalidValue;
  const bool shaped = e.height > 1 || e.depth > 1;
  // A row wider than the pitch would overlap the next one.
  if (shaped && (e.width > p->srcPtr.pitch || e.width > p->dstPtr.pitch))
    return cudaErrorInvalidValue;
  // Slices are ysize rows apart, so more than one needs to know how many.
  if (e.depth > 1 && (p->srcPtr.ysize == 0 || p->dstPtr.ysize == 0)) return cudaErrorInvalidValue;
  const size_t src_slice = p->srcPtr.pitch * p->srcPtr.ysize;
  const size_t dst_slice = p->dstPtr.pitch * p->dstPtr.ysize;
  w->kind = RecordedLaunch::Kind::Memcpy;
  w->src = static_cast<const char*>(p->srcPtr.ptr) + p->srcPos.z * src_slice +
           p->srcPos.y * p->srcPtr.pitch + p->srcPos.x;
  w->dst = static_cast<char*>(p->dstPtr.ptr) + p->dstPos.z * dst_slice +
           p->dstPos.y * p->dstPtr.pitch + p->dstPos.x;
  w->bytes = e.width;
  w->height = e.height;
  w->depth = e.depth;
  w->src_pitch = p->srcPtr.pitch;
  w->dst_pitch = p->dstPtr.pitch;
  w->src_slice = src_slice;
  w->dst_slice = dst_slice;
  w->copy_kind = p->kind;
  w->copy_params = *p;
  return cudaSuccess;
}

// The 3D-copy description of a linear copy, which is how CUDA reports one.
cudaMemcpy3DParms linear_copy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
  cudaMemcpy3DParms p{};
  p.srcPtr = cudaPitchedPtr{const_cast<void*>(src), count, count, 1};
  p.dstPtr = cudaPitchedPtr{dst, count, count, 1};
  p.extent = cudaExtent{count, 1, 1};
  p.kind = kind;
  return p;
}

// A fill node's work: `height` rows of `width` elements, each `elementSize`
// bytes set to the value's low bytes -- which is what cuMemsetD16/D32 do, and
// what a 2- or 4-byte cudaMemsetParams asks for.
cudaError_t fill_memset_work(RecordedLaunch* w, const cudaMemsetParams* p) {
  if (!p || !p->dst) return cudaErrorInvalidValue;
  if (p->elementSize != 1 && p->elementSize != 2 && p->elementSize != 4)
    return cudaErrorInvalidValue;
  if (p->width == 0 || p->height == 0) return cudaErrorInvalidValue;
  const size_t row = static_cast<size_t>(p->width) * p->elementSize;
  if (p->height > 1 && p->pitch < row) return cudaErrorInvalidValue;   // rows would overlap
  w->kind = RecordedLaunch::Kind::Memset;
  w->dst = p->dst;
  w->fill_value = static_cast<int>(p->value);
  w->elem_size = p->elementSize;
  w->bytes = row;
  w->height = p->height;
  w->depth = 1;
  w->dst_pitch = p->pitch;
  w->fill_params = *p;
  return cudaSuccess;
}


// The node in an instantiated graph that answers for `original` in the graph it
// was instantiated from. Every cudaGraphExec*NodeSetParams call names the node
// in the original graph, because the executable graph's own nodes are not
// handles a program ever sees; the copy sits at the same position, which is what
// makes the correspondence well defined.
GraphNodeRec* exec_twin(GraphRec& exec, const GraphNodeRec* original) {  // holds g_graph_mu
  for (const auto& [handle, g] : g_graphs) {
    if (!g->holds(original)) continue;
    for (size_t i = 0; i < g->nodes.size(); ++i) {
      if (g->nodes[i].get() != original) continue;
      return i < exec.nodes.size() ? exec.nodes[i].get() : nullptr;
    }
  }
  return nullptr;
}

// Whether an event handle is one this process created. Checked before the graph
// lock is taken, never under it: the event lock comes first everywhere, because
// a launch records an event while running a node.
bool event_exists(cudaEvent_t e) {
  std::lock_guard<std::mutex> lock(g_event_mu);
  return find_event(e) != nullptr;
}

// Whether two graphs have the same shape: the same nodes, of the same types, in
// the same order, depending on the same positions. cudaGraphExecUpdate and
// cudaGraphExecChildGraphNodeSetParams both need it, and both document it as
// the condition for taking new parameters.
bool same_topology(const GraphRec& a, const GraphRec& b) {
  if (a.nodes.size() != b.nodes.size()) return false;
  std::map<const GraphNodeRec*, size_t> ia, ib;
  for (size_t i = 0; i < a.nodes.size(); ++i) ia[a.nodes[i].get()] = i;
  for (size_t i = 0; i < b.nodes.size(); ++i) ib[b.nodes[i].get()] = i;
  for (size_t i = 0; i < a.nodes.size(); ++i) {
    if (a.nodes[i]->type != b.nodes[i]->type) return false;
    if (a.nodes[i]->deps.size() != b.nodes[i]->deps.size()) return false;
    std::vector<size_t> da, db;
    for (const GraphNodeRec* d : a.nodes[i]->deps) da.push_back(ia[d]);
    for (const GraphNodeRec* d : b.nodes[i]->deps) db.push_back(ib[d]);
    std::sort(da.begin(), da.end());
    std::sort(db.begin(), db.end());
    if (da != db) return false;
    const bool child_a = a.nodes[i]->child != nullptr, child_b = b.nodes[i]->child != nullptr;
    if (child_a != child_b) return false;
    if (child_a && !same_topology(*a.nodes[i]->child, *b.nodes[i]->child)) return false;
  }
  return true;
}

// Takes new parameters for every node of a graph that already has the right
// shape, which is exactly what an update is: the work each node does changes,
// the structure does not. Child graphs are updated the same way, recursively,
// as cudaGraphExecChildGraphNodeSetParams documents.
void copy_params(GraphRec& dst, const GraphRec& src) {
  for (size_t i = 0; i < dst.nodes.size() && i < src.nodes.size(); ++i) {
    RecordedLaunch& d = dst.nodes[i]->work;
    const RecordedLaunch& s = src.nodes[i]->work;
    d.grid = s.grid;
    d.block = s.block;
    d.shared = s.shared;
    d.arg_bytes = s.arg_bytes;
    d.cooperative = s.cooperative;
    d.cluster = s.cluster;
    d.dst = s.dst;
    d.src = s.src;
    d.bytes = s.bytes;
    d.copy_kind = s.copy_kind;
    d.fill_value = s.fill_value;
    d.height = s.height;
    d.depth = s.depth;
    d.dst_pitch = s.dst_pitch;
    d.src_pitch = s.src_pitch;
    d.dst_slice = s.dst_slice;
    d.src_slice = s.src_slice;
    d.elem_size = s.elem_size;
    d.copy_params = s.copy_params;
    d.fill_params = s.fill_params;
    d.host_fn = s.host_fn;
    d.host_user = s.host_user;
    d.host_op = s.host_op;
    d.event = s.event;
    if (dst.nodes[i]->child && src.nodes[i]->child)
      copy_params(*dst.nodes[i]->child, *src.nodes[i]->child);
  }
}

// Whether every kernel node still runs the same kernel. Which kernel a node
// runs is part of a graph's shape, not one of its parameters, so an update that
// changes one is refused rather than applied.
const GraphNodeRec* changed_function(const GraphRec& have, const GraphRec& want) {
  for (size_t i = 0; i < have.nodes.size() && i < want.nodes.size(); ++i) {
    if (have.nodes[i]->type == cudaGraphNodeTypeKernel &&
        have.nodes[i]->work.func != want.nodes[i]->work.func)
      return want.nodes[i].get();
    if (have.nodes[i]->child && want.nodes[i]->child)
      if (const GraphNodeRec* n = changed_function(*have.nodes[i]->child, *want.nodes[i]->child))
        return n;
  }
  return nullptr;
}

// Whether `from` can already be reached from `to`: adding to -> from would then
// close a cycle, which CUDA refuses.
bool reaches(const GraphNodeRec* from, const GraphNodeRec* to) {
  if (from == to) return true;
  for (const GraphNodeRec* d : from->deps)
    if (reaches(d, to)) return true;
  return false;
}

}  // namespace

VGPU_EXPORT cudaError_t cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags) {
  if (!pGraph || flags != 0) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto g = std::make_unique<GraphRec>();
  void* handle = g.get();
  g_graphs[handle] = std::move(g);
  *pGraph = static_cast<cudaGraph_t>(handle);
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* pNode, cudaGraph_t graph,
                                              const cudaGraphNode_t* deps, size_t numDeps,
                                              const cudaKernelNodeParams* params) {
  // Before the graph lock, because copying the parameters needs the runtime's
  // lock and that one comes first everywhere (see kernel_param_sizes).
  RecordedLaunch w;
  if (const cudaError_t rc = fill_kernel_work(&w, params); rc != cudaSuccess) return rc;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* g = graph_from(graph);
  if (!g || !pNode) return cudaErrorInvalidValue;
  std::vector<GraphNodeRec*> pred;
  if (!deps_ok(*g, deps, numDeps, &pred)) return cudaErrorInvalidValue;
  *pNode = reinterpret_cast<cudaGraphNode_t>(g->add(cudaGraphNodeTypeKernel, std::move(w), pred));
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t* pNode, cudaGraph_t graph,
                                               const cudaGraphNode_t* deps, size_t numDeps,
                                               const cudaMemcpy3DParms* params) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* g = graph_from(graph);
  if (!g || !pNode) return cudaErrorInvalidValue;
  std::vector<GraphNodeRec*> pred;
  if (!deps_ok(*g, deps, numDeps, &pred)) return cudaErrorInvalidValue;
  RecordedLaunch w;
  if (const cudaError_t rc = fill_memcpy_work(&w, params); rc != cudaSuccess) return rc;
  *pNode = reinterpret_cast<cudaGraphNode_t>(g->add(cudaGraphNodeTypeMemcpy, std::move(w), pred));
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pNode, cudaGraph_t graph,
                                                 const cudaGraphNode_t* deps, size_t numDeps,
                                                 void* dst, const void* src, size_t count,
                                                 cudaMemcpyKind kind) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* g = graph_from(graph);
  if (!g || !pNode || !dst || !src) return cudaErrorInvalidValue;
  std::vector<GraphNodeRec*> pred;
  if (!deps_ok(*g, deps, numDeps, &pred)) return cudaErrorInvalidValue;
  RecordedLaunch w;
  const cudaMemcpy3DParms p = linear_copy(dst, src, count, kind);
  if (const cudaError_t rc = fill_memcpy_work(&w, &p); rc != cudaSuccess) return rc;
  *pNode = reinterpret_cast<cudaGraphNode_t>(g->add(cudaGraphNodeTypeMemcpy, std::move(w), pred));
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t* pNode, cudaGraph_t graph,
                                               const cudaGraphNode_t* deps, size_t numDeps,
                                               const cudaMemsetParams* params) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* g = graph_from(graph);
  if (!g || !pNode) return cudaErrorInvalidValue;
  std::vector<GraphNodeRec*> pred;
  if (!deps_ok(*g, deps, numDeps, &pred)) return cudaErrorInvalidValue;
  RecordedLaunch w;
  if (const cudaError_t rc = fill_memset_work(&w, params); rc != cudaSuccess) return rc;
  *pNode = reinterpret_cast<cudaGraphNode_t>(g->add(cudaGraphNodeTypeMemset, std::move(w), pred));
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t* pNode, cudaGraph_t graph,
                                              const cudaGraphNode_t* deps, size_t numDeps) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* g = graph_from(graph);
  if (!g || !pNode) return cudaErrorInvalidValue;
  std::vector<GraphNodeRec*> pred;
  if (!deps_ok(*g, deps, numDeps, &pred)) return cudaErrorInvalidValue;
  *pNode = reinterpret_cast<cudaGraphNode_t>(g->add(cudaGraphNodeTypeEmpty, RecordedLaunch{}, pred));
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t* pNode, cudaGraph_t graph,
                                                   const cudaGraphNode_t* deps, size_t numDeps,
                                                   cudaGraph_t childGraph) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* g = graph_from(graph);
  GraphRec* child = graph_from(childGraph);
  if (!g || !child || !pNode || g == child) return cudaErrorInvalidValue;
  std::vector<GraphNodeRec*> pred;
  if (!deps_ok(*g, deps, numDeps, &pred)) return cudaErrorInvalidValue;
  // A graph that owns memory cannot be a child of another graph: the child is
  // taken by copy, and a copy of an allocation node would name an address the
  // original owns. CUDA documents the same restriction.
  if (holds_graph_memory(*child)) return cudaErrorInvalidValue;
  // Cloned now: CUDA takes the child's structure as it is at this moment.
  g->children.push_back(clone_graph(*child, nullptr));
  GraphNodeRec* n = g->add(cudaGraphNodeTypeGraph, RecordedLaunch{}, pred);
  n->child = g->children.back().get();
  *pNode = reinterpret_cast<cudaGraphNode_t>(n);
  return cudaSuccess;
}

static cudaError_t graph_add_deps(cudaGraph_t graph, const cudaGraphNode_t* from,
                                 const cudaGraphNode_t* to, size_t numDeps) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* g = graph_from(graph);
  if (!g || (numDeps && (!from || !to))) return cudaErrorInvalidValue;
  for (size_t i = 0; i < numDeps; ++i) {
    auto* f = reinterpret_cast<GraphNodeRec*>(from[i]);
    auto* t = reinterpret_cast<GraphNodeRec*>(to[i]);
    if (!f || !t || !g->holds(f) || !g->holds(t)) return cudaErrorInvalidValue;
    // `to` runs after `from`. A dependency that closes a cycle is refused,
    // because a graph with one could never be launched.
    if (reaches(f, t)) return cudaErrorInvalidValue;
    if (std::find(t->deps.begin(), t->deps.end(), f) == t->deps.end()) t->deps.push_back(f);
  }
  return cudaSuccess;
}

static cudaError_t graph_remove_deps(cudaGraph_t graph, const cudaGraphNode_t* from,
                                    const cudaGraphNode_t* to, size_t numDeps) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* g = graph_from(graph);
  if (!g || (numDeps && (!from || !to))) return cudaErrorInvalidValue;
  // Edges cannot be taken out of a graph that owns memory: the order an
  // allocation, its uses and its free run in is the whole reason it is safe.
  if (holds_graph_memory(*g)) return cudaErrorInvalidValue;
  for (size_t i = 0; i < numDeps; ++i) {
    auto* f = reinterpret_cast<GraphNodeRec*>(from[i]);
    auto* t = reinterpret_cast<GraphNodeRec*>(to[i]);
    if (!f || !t || !g->holds(f) || !g->holds(t)) return cudaErrorInvalidValue;
    const auto at = std::find(t->deps.begin(), t->deps.end(), f);
    if (at == t->deps.end()) return cudaErrorInvalidValue;   // not a dependency
    t->deps.erase(at);
  }
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto* n = reinterpret_cast<GraphNodeRec*>(node);
  for (auto& [handle, g] : g_graphs) {
    if (!g->holds(n)) continue;
    // Nodes cannot be taken out of a graph that owns memory: removing the
    // allocation node or the free node would leave the other one alone.
    if (holds_graph_memory(*g)) return cudaErrorInvalidValue;
    for (auto& up : g->nodes)
      up->deps.erase(std::remove(up->deps.begin(), up->deps.end(), n), up->deps.end());
    for (auto& [s, sc] : g_stream_capture)
      if (sc.cap->graph.get() == g.get()) std::erase(sc.deps, n);
    std::erase_if(g->nodes, [&](const std::unique_ptr<GraphNodeRec>& up) { return up.get() == n; });
    return cudaSuccess;
  }
  return cudaErrorInvalidValue;
}

// ---- what a graph is, read back ----

VGPU_EXPORT cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes,
                                          size_t* numNodes) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* g = graph_from(graph);
  if (!g || !numNodes) return cudaErrorInvalidValue;
  // With no array, the count is the answer; with one, it says how many fit.
  const size_t have = g->nodes.size();
  if (!nodes) {
    *numNodes = have;
    return cudaSuccess;
  }
  const size_t take = std::min(*numNodes, have);
  for (size_t i = 0; i < take; ++i)
    nodes[i] = reinterpret_cast<cudaGraphNode_t>(g->nodes[i].get());
  *numNodes = take;
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* nodes,
                                              size_t* numRootNodes) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* g = graph_from(graph);
  if (!g || !numRootNodes) return cudaErrorInvalidValue;
  std::vector<GraphNodeRec*> roots;
  for (const auto& up : g->nodes)
    if (up->deps.empty()) roots.push_back(up.get());
  if (!nodes) {
    *numRootNodes = roots.size();
    return cudaSuccess;
  }
  const size_t take = std::min(*numRootNodes, roots.size());
  for (size_t i = 0; i < take; ++i) nodes[i] = reinterpret_cast<cudaGraphNode_t>(roots[i]);
  *numRootNodes = take;
  return cudaSuccess;
}

static cudaError_t graph_get_edges(cudaGraph_t graph, cudaGraphNode_t* from,
                                  cudaGraphNode_t* to, size_t* numEdges) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* g = graph_from(graph);
  if (!g || !numEdges) return cudaErrorInvalidValue;
  std::vector<std::pair<GraphNodeRec*, GraphNodeRec*>> edges;
  for (const auto& up : g->nodes)
    for (GraphNodeRec* d : up->deps) edges.emplace_back(d, up.get());
  if (!from || !to) {
    *numEdges = edges.size();
    return cudaSuccess;
  }
  const size_t take = std::min(*numEdges, edges.size());
  for (size_t i = 0; i < take; ++i) {
    from[i] = reinterpret_cast<cudaGraphNode_t>(edges[i].first);
    to[i] = reinterpret_cast<cudaGraphNode_t>(edges[i].second);
  }
  *numEdges = take;
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* type) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto* n = reinterpret_cast<GraphNodeRec*>(node);
  if (!n || !type || !owner_of(n)) return cudaErrorInvalidValue;
  *type = n->type;
  return cudaSuccess;
}

static cudaError_t node_get_deps(cudaGraphNode_t node, cudaGraphNode_t* deps, size_t* numDeps) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto* n = reinterpret_cast<GraphNodeRec*>(node);
  if (!n || !numDeps) return cudaErrorInvalidValue;
  if (!deps) {
    *numDeps = n->deps.size();
    return cudaSuccess;
  }
  const size_t take = std::min(*numDeps, n->deps.size());
  for (size_t i = 0; i < take; ++i) deps[i] = reinterpret_cast<cudaGraphNode_t>(n->deps[i]);
  *numDeps = take;
  return cudaSuccess;
}

static cudaError_t node_get_dependents(cudaGraphNode_t node, cudaGraphNode_t* dependent,
                                      size_t* numDependentNodes) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto* n = reinterpret_cast<GraphNodeRec*>(node);
  if (!n || !numDependentNodes) return cudaErrorInvalidValue;
  std::vector<GraphNodeRec*> after;
  if (GraphRec* g = owner_of(n))
    for (const auto& up : g->nodes)
      if (std::find(up->deps.begin(), up->deps.end(), n) != up->deps.end()) after.push_back(up.get());
  if (!dependent) {
    *numDependentNodes = after.size();
    return cudaSuccess;
  }
  const size_t take = std::min(*numDependentNodes, after.size());
  for (size_t i = 0; i < take; ++i) dependent[i] = reinterpret_cast<cudaGraphNode_t>(after[i]);
  *numDependentNodes = take;
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto* n = reinterpret_cast<GraphNodeRec*>(node);
  if (!n || !pGraph || n->type != cudaGraphNodeTypeGraph || !n->child) return cudaErrorInvalidValue;
  // The node owns its child graph. What is handed out is a handle that can be
  // read like any other graph and not destroyed: the parent's node frees it.
  g_borrowed_graphs[n->child] = n->child;
  *pGraph = reinterpret_cast<cudaGraph_t>(n->child);
  return cudaSuccess;
}

// ---- parameters, read and changed ----

VGPU_EXPORT cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node,
                                                     cudaKernelNodeParams* params) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto* n = reinterpret_cast<GraphNodeRec*>(node);
  if (!n || !params || n->type != cudaGraphNodeTypeKernel) return cudaErrorInvalidValue;
  params->func = const_cast<void*>(n->work.func);
  params->gridDim = n->work.grid;
  params->blockDim = n->work.block;
  params->sharedMemBytes = static_cast<unsigned>(n->work.shared);
  // The argument *values* live in the node; a caller reading them back gets
  // pointers into the node's own copies, which is what the driver does too.
  n->work.arg_ptrs.resize(n->work.arg_bytes.size());
  for (size_t i = 0; i < n->work.arg_bytes.size(); ++i)
    n->work.arg_ptrs[i] = n->work.arg_bytes[i].data();
  params->kernelParams = n->work.arg_ptrs.data();
  params->extra = nullptr;
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node,
                                                     const cudaKernelNodeParams* params) {
  RecordedLaunch w;   // the runtime's lock first, then the graph lock
  if (const cudaError_t rc = fill_kernel_work(&w, params); rc != cudaSuccess) return rc;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto* n = reinterpret_cast<GraphNodeRec*>(node);
  if (!n || n->type != cudaGraphNodeTypeKernel) return cudaErrorInvalidValue;
  // The kernel a node runs cannot change, only its arguments and shape.
  if (w.func != n->work.func) return cudaErrorInvalidValue;
  n->work = std::move(w);
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node,
                                                     cudaMemsetParams* params) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto* n = reinterpret_cast<GraphNodeRec*>(node);
  if (!n || !params || n->type != cudaGraphNodeTypeMemset) return cudaErrorInvalidValue;
  *params = n->work.fill_params;   // as it was given: element size, width, pitch, rows
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node,
                                                     const cudaMemsetParams* params) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto* n = reinterpret_cast<GraphNodeRec*>(node);
  if (!n || n->type != cudaGraphNodeTypeMemset) return cudaErrorInvalidValue;
  return fill_memset_work(&n->work, params);
}

VGPU_EXPORT cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node,
                                                     cudaMemcpy3DParms* params) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto* n = reinterpret_cast<GraphNodeRec*>(node);
  if (!n || !params || n->type != cudaGraphNodeTypeMemcpy) return cudaErrorInvalidValue;
  *params = n->work.copy_params;   // as it was given, positions and pitches included
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node,
                                                     const cudaMemcpy3DParms* params) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto* n = reinterpret_cast<GraphNodeRec*>(node);
  if (!n || n->type != cudaGraphNodeTypeMemcpy) return cudaErrorInvalidValue;
  return fill_memcpy_work(&n->work, params);
}

VGPU_EXPORT cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst,
                                                       const void* src, size_t count,
                                                       cudaMemcpyKind kind) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto* n = reinterpret_cast<GraphNodeRec*>(node);
  if (!n || !dst || !src || n->type != cudaGraphNodeTypeMemcpy) return cudaErrorInvalidValue;
  const cudaMemcpy3DParms p = linear_copy(dst, src, count, kind);
  return fill_memcpy_work(&n->work, &p);
}

// ---- cloning, and updating an instantiated graph ----

VGPU_EXPORT cudaError_t cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* src = graph_from(originalGraph);
  if (!src || !pGraphClone) return cudaErrorInvalidValue;
  // A graph that owns memory cannot be cloned: the clone's allocation nodes
  // would name the original's addresses, and one of the two graphs would free
  // memory the other still uses. CUDA refuses it for the same reason.
  if (holds_graph_memory(*src)) return cudaErrorInvalidValue;
  auto copy = clone_graph(*src, nullptr);
  void* handle = copy.get();
  g_graphs[handle] = std::move(copy);
  *pGraphClone = static_cast<cudaGraph_t>(handle);
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t* pNode,
                                                 cudaGraphNode_t originalNode,
                                                 cudaGraph_t clonedGraph) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* clone = graph_from(clonedGraph);
  auto* original = reinterpret_cast<GraphNodeRec*>(originalNode);
  if (!clone || !original || !pNode) return cudaErrorInvalidValue;
  // A clone keeps its nodes in the order the original had them, so the node at
  // the same position is the same node.
  for (auto& [handle, g] : g_graphs) {
    if (!g->holds(original)) continue;
    for (size_t i = 0; i < g->nodes.size(); ++i)
      if (g->nodes[i].get() == original) {
        if (i >= clone->nodes.size()) return cudaErrorInvalidValue;
        *pNode = reinterpret_cast<cudaGraphNode_t>(clone->nodes[i].get());
        return cudaSuccess;
      }
  }
  return cudaErrorInvalidValue;
}

// An instantiated graph can take new parameters without being built again, as
// long as the shape it was built from has not changed. That is the whole point
// of the call: a framework re-uses one executable graph with new pointers.
VGPU_EXPORT cudaError_t cudaGraphExecUpdate(cudaGraphExec_t exec, cudaGraph_t graph,
                                            cudaGraphExecUpdateResultInfo* info) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  if (info) std::memset(info, 0, sizeof *info);
  auto it = g_graph_execs.find(static_cast<void*>(exec));
  GraphRec* want = graph_from(graph);
  if (it == g_graph_execs.end() || !want) return cudaErrorInvalidValue;
  GraphRec& have = *it->second;
  // Same shape means the same nodes in the same order depending on the same
  // ones. Counting nodes and edges was not enough: a graph rewired between two
  // nodes has the same counts, and taking its parameters would have run the old
  // wiring with the new work and reported success.
  if (!same_topology(have, *want)) {
    if (info) info->result = cudaGraphExecUpdateErrorTopologyChanged;
    return cudaErrorGraphExecUpdateFailure;
  }
  if (const GraphNodeRec* n = changed_function(have, *want)) {
    if (info) {
      info->result = cudaGraphExecUpdateErrorFunctionChanged;
      info->errorNode = reinterpret_cast<cudaGraphNode_t>(const_cast<GraphNodeRec*>(n));
    }
    return cudaErrorGraphExecUpdateFailure;
  }
  copy_params(have, *want);
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t exec,
                                                          cudaGraphNode_t node,
                                                          const cudaKernelNodeParams* params) {
  RecordedLaunch w;   // the runtime's lock first, then the graph lock
  if (const cudaError_t rc = fill_kernel_work(&w, params); rc != cudaSuccess) return rc;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto it = g_graph_execs.find(static_cast<void*>(exec));
  if (it == g_graph_execs.end()) return cudaErrorInvalidValue;
  auto* original = reinterpret_cast<GraphNodeRec*>(node);
  if (!original) return cudaErrorInvalidValue;
  // The node named is the one in the graph that was instantiated; the
  // executable graph holds its own copy at the same position.
  for (auto& [handle, g] : g_graphs) {
    if (!g->holds(original)) continue;
    for (size_t i = 0; i < g->nodes.size(); ++i) {
      if (g->nodes[i].get() != original) continue;
      if (i >= it->second->nodes.size()) return cudaErrorInvalidValue;
      GraphNodeRec* target = it->second->nodes[i].get();
      if (target->type != cudaGraphNodeTypeKernel) return cudaErrorInvalidValue;
      if (w.func != target->work.func) return cudaErrorInvalidValue;
      target->work = std::move(w);
      return cudaSuccess;
    }
  }
  return cudaErrorInvalidValue;
}


/* ---- host functions, events, and a node switched off -------------------------
 *
 * The node types a launch can run that the explicit API could not build. A host
 * node runs a function on the CPU when the graph reaches it, which is how a
 * program folds its own bookkeeping into a graph instead of breaking the graph
 * in half around it. An event record node records an event and an event wait
 * node waits for one, so a graph can be timed and can be joined with work
 * outside it.
 *
 * Switching a node off is the other half of reusing an instantiated graph: a
 * program that wants one step skipped this launch disables it rather than
 * rebuilding and re-instantiating the graph around it.
 */

VGPU_EXPORT cudaError_t cudaGraphAddHostNode(cudaGraphNode_t* pNode, cudaGraph_t graph,
                                             const cudaGraphNode_t* deps, size_t numDeps,
                                             const cudaHostNodeParams* params) {
  if (!pNode || !params || !params->fn) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* g = graph_from(graph);
  if (!g) return cudaErrorInvalidValue;
  std::vector<GraphNodeRec*> pred;
  if (!deps_ok(*g, deps, numDeps, &pred)) return cudaErrorInvalidValue;
  RecordedLaunch w;
  w.kind = RecordedLaunch::Kind::Host;
  w.host_fn = params->fn;
  w.host_user = params->userData;
  *pNode = reinterpret_cast<cudaGraphNode_t>(g->add(cudaGraphNodeTypeHost, std::move(w), pred));
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node,
                                                   cudaHostNodeParams* params) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto* n = reinterpret_cast<GraphNodeRec*>(node);
  if (!n || !params || n->type != cudaGraphNodeTypeHost) return cudaErrorInvalidValue;
  params->fn = n->work.host_fn;
  params->userData = n->work.host_user;
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node,
                                                   const cudaHostNodeParams* params) {
  if (!params || !params->fn) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto* n = reinterpret_cast<GraphNodeRec*>(node);
  if (!n || n->type != cudaGraphNodeTypeHost) return cudaErrorInvalidValue;
  n->work.host_fn = params->fn;
  n->work.host_user = params->userData;
  n->work.host_op = nullptr;   // a captured library's closure is replaced, not kept beside it
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t* pNode, cudaGraph_t graph,
                                                    const cudaGraphNode_t* deps, size_t numDeps,
                                                    cudaEvent_t event) {
  if (!pNode || !event) return cudaErrorInvalidValue;
  if (!event_exists(event)) return cudaErrorInvalidResourceHandle;   // before the graph lock
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* g = graph_from(graph);
  if (!g) return cudaErrorInvalidValue;
  std::vector<GraphNodeRec*> pred;
  if (!deps_ok(*g, deps, numDeps, &pred)) return cudaErrorInvalidValue;
  RecordedLaunch w;
  w.event = event;
  *pNode = reinterpret_cast<cudaGraphNode_t>(
      g->add(cudaGraphNodeTypeEventRecord, std::move(w), pred));
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t* pNode, cudaGraph_t graph,
                                                   const cudaGraphNode_t* deps, size_t numDeps,
                                                   cudaEvent_t event) {
  if (!pNode || !event) return cudaErrorInvalidValue;
  if (!event_exists(event)) return cudaErrorInvalidResourceHandle;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* g = graph_from(graph);
  if (!g) return cudaErrorInvalidValue;
  std::vector<GraphNodeRec*> pred;
  if (!deps_ok(*g, deps, numDeps, &pred)) return cudaErrorInvalidValue;
  RecordedLaunch w;
  w.event = event;
  *pNode = reinterpret_cast<cudaGraphNode_t>(
      g->add(cudaGraphNodeTypeWaitEvent, std::move(w), pred));
  return cudaSuccess;
}

namespace {
// Both event node types answer the same two calls, one per type.
cudaError_t event_node_get(cudaGraphNode_t node, cudaEvent_t* out, cudaGraphNodeType want) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto* n = reinterpret_cast<GraphNodeRec*>(node);
  if (!n || !out || n->type != want) return cudaErrorInvalidValue;
  *out = n->work.event;
  return cudaSuccess;
}
cudaError_t event_node_set(cudaGraphNode_t node, cudaEvent_t event, cudaGraphNodeType want) {
  if (!event) return cudaErrorInvalidValue;
  if (!event_exists(event)) return cudaErrorInvalidResourceHandle;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto* n = reinterpret_cast<GraphNodeRec*>(node);
  if (!n || n->type != want) return cudaErrorInvalidValue;
  n->work.event = event;
  return cudaSuccess;
}
}  // namespace

VGPU_EXPORT cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node,
                                                         cudaEvent_t* out) {
  return event_node_get(node, out, cudaGraphNodeTypeEventRecord);
}
VGPU_EXPORT cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node,
                                                         cudaEvent_t event) {
  return event_node_set(node, event, cudaGraphNodeTypeEventRecord);
}
VGPU_EXPORT cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* out) {
  return event_node_get(node, out, cudaGraphNodeTypeWaitEvent);
}
VGPU_EXPORT cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) {
  return event_node_set(node, event, cudaGraphNodeTypeWaitEvent);
}

VGPU_EXPORT cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t exec, cudaGraphNode_t node,
                                                unsigned int isEnabled) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto it = g_graph_execs.find(static_cast<void*>(exec));
  auto* original = reinterpret_cast<GraphNodeRec*>(node);
  if (it == g_graph_execs.end() || !original) return cudaErrorInvalidValue;
  // Kernel, memset and memcpy nodes only, which is what CUDA documents: those
  // are the ones an empty node can stand in for without changing what the rest
  // of the graph sees.
  if (original->type != cudaGraphNodeTypeKernel && original->type != cudaGraphNodeTypeMemset &&
      original->type != cudaGraphNodeTypeMemcpy)
    return cudaErrorInvalidValue;
  GraphNodeRec* target = exec_twin(*it->second, original);
  if (!target) return cudaErrorInvalidValue;
  target->enabled = isEnabled != 0;
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphNodeGetEnabled(cudaGraphExec_t exec, cudaGraphNode_t node,
                                                unsigned int* isEnabled) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto it = g_graph_execs.find(static_cast<void*>(exec));
  auto* original = reinterpret_cast<GraphNodeRec*>(node);
  if (it == g_graph_execs.end() || !original || !isEnabled) return cudaErrorInvalidValue;
  GraphNodeRec* target = exec_twin(*it->second, original);
  if (!target) return cudaErrorInvalidValue;
  *isEnabled = target->enabled ? 1u : 0u;
  return cudaSuccess;
}

// New parameters for one node of an instantiated graph, without going back to
// the graph it came from. Each call names the node in the original graph and
// leaves that node alone, as documented: only the executable copy changes.
VGPU_EXPORT cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t exec,
                                                         cudaGraphNode_t node,
                                                         const cudaMemsetParams* params) {
  if (!params) return cudaErrorInvalidValue;
  // A 2D fill, before and after, is refused here as invalid rather than
  // unsupported: this call documents that both operands must be 1D.
  if (params->height > 1 || params->width == 0) return cudaErrorInvalidValue;
  RecordedLaunch w;
  if (fill_memset_work(&w, params) != cudaSuccess) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto it = g_graph_execs.find(static_cast<void*>(exec));
  auto* original = reinterpret_cast<GraphNodeRec*>(node);
  if (it == g_graph_execs.end() || !original) return cudaErrorInvalidValue;
  GraphNodeRec* target = exec_twin(*it->second, original);
  if (!target || target->type != cudaGraphNodeTypeMemset) return cudaErrorInvalidValue;
  if (target->work.height > 1) return cudaErrorInvalidValue;   // the original was 2D
  target->work = std::move(w);
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t exec,
                                                         cudaGraphNode_t node,
                                                         const cudaMemcpy3DParms* params) {
  RecordedLaunch w;
  if (fill_memcpy_work(&w, params) != cudaSuccess) return cudaErrorInvalidValue;
  // Both the operands it was instantiated with and the new ones have to be 1D,
  // as this call documents; a shape changes on the graph, not here.
  if (w.height > 1 || w.depth > 1) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto it = g_graph_execs.find(static_cast<void*>(exec));
  auto* original = reinterpret_cast<GraphNodeRec*>(node);
  if (it == g_graph_execs.end() || !original) return cudaErrorInvalidValue;
  GraphNodeRec* target = exec_twin(*it->second, original);
  if (!target || target->type != cudaGraphNodeTypeMemcpy) return cudaErrorInvalidValue;
  if (target->work.height > 1 || target->work.depth > 1) return cudaErrorInvalidValue;
  target->work = std::move(w);
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t exec,
                                                           cudaGraphNode_t node, void* dst,
                                                           const void* src, size_t count,
                                                           cudaMemcpyKind kind) {
  if (!dst || !src || count == 0) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto it = g_graph_execs.find(static_cast<void*>(exec));
  auto* original = reinterpret_cast<GraphNodeRec*>(node);
  if (it == g_graph_execs.end() || !original) return cudaErrorInvalidValue;
  GraphNodeRec* target = exec_twin(*it->second, original);
  if (!target || target->type != cudaGraphNodeTypeMemcpy) return cudaErrorInvalidValue;
  if (target->work.height > 1 || target->work.depth > 1) return cudaErrorInvalidValue;
  const cudaMemcpy3DParms p = linear_copy(dst, src, count, kind);
  return fill_memcpy_work(&target->work, &p);
}

VGPU_EXPORT cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t exec, cudaGraphNode_t node,
                                                       const cudaHostNodeParams* params) {
  if (!params || !params->fn) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto it = g_graph_execs.find(static_cast<void*>(exec));
  auto* original = reinterpret_cast<GraphNodeRec*>(node);
  if (it == g_graph_execs.end() || !original) return cudaErrorInvalidValue;
  GraphNodeRec* target = exec_twin(*it->second, original);
  if (!target || target->type != cudaGraphNodeTypeHost) return cudaErrorInvalidValue;
  target->work.host_fn = params->fn;
  target->work.host_user = params->userData;
  target->work.host_op = nullptr;
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t exec,
                                                             cudaGraphNode_t node,
                                                             cudaGraph_t childGraph) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto it = g_graph_execs.find(static_cast<void*>(exec));
  auto* original = reinterpret_cast<GraphNodeRec*>(node);
  GraphRec* want = graph_from(childGraph);
  if (it == g_graph_execs.end() || !original || !want) return cudaErrorInvalidValue;
  GraphNodeRec* target = exec_twin(*it->second, original);
  if (!target || target->type != cudaGraphNodeTypeGraph || !target->child)
    return cudaErrorInvalidValue;
  // The new child has to have the shape the instantiated one has, in the same
  // insertion order: this call supplies parameters, not a different graph.
  if (!same_topology(*target->child, *want)) return cudaErrorInvalidValue;
  if (changed_function(*target->child, *want)) return cudaErrorInvalidValue;
  copy_params(*target->child, *want);
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t exec,
                                                             cudaGraphNode_t node,
                                                             cudaEvent_t event) {
  if (!event) return cudaErrorInvalidValue;
  if (!event_exists(event)) return cudaErrorInvalidResourceHandle;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto it = g_graph_execs.find(static_cast<void*>(exec));
  auto* original = reinterpret_cast<GraphNodeRec*>(node);
  if (it == g_graph_execs.end() || !original) return cudaErrorInvalidValue;
  GraphNodeRec* target = exec_twin(*it->second, original);
  if (!target || target->type != cudaGraphNodeTypeEventRecord) return cudaErrorInvalidValue;
  target->work.event = event;
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t exec,
                                                           cudaGraphNode_t node,
                                                           cudaEvent_t event) {
  if (!event) return cudaErrorInvalidValue;
  if (!event_exists(event)) return cudaErrorInvalidResourceHandle;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto it = g_graph_execs.find(static_cast<void*>(exec));
  auto* original = reinterpret_cast<GraphNodeRec*>(node);
  if (it == g_graph_execs.end() || !original) return cudaErrorInvalidValue;
  GraphNodeRec* target = exec_twin(*it->second, original);
  if (!target || target->type != cudaGraphNodeTypeWaitEvent) return cudaErrorInvalidValue;
  target->work.event = event;
  return cudaSuccess;
}

// Uploading an instantiated graph to the device ahead of its first launch is
// what this asks for, and there is no device to upload to: the nodes are
// already in this process's memory. It validates the handle and succeeds, so a
// program that uploads before launching behaves as it would on a card, and
// nothing about the launch is faster or slower for it.
VGPU_EXPORT cudaError_t cudaGraphUpload(cudaGraphExec_t exec, cudaStream_t) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  return g_graph_execs.count(static_cast<void*>(exec)) ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t free_graph_alloc(State& s, void* ptr, bool* handled) {
  std::lock_guard<std::mutex> lock(g_graph_mem_mu);
  *handled = false;
  const auto it = g_graph_allocs.find(reinterpret_cast<uint64_t>(ptr));
  if (it == g_graph_allocs.end()) return cudaSuccess;
  *handled = true;
  GraphAlloc& a = it->second;
  // Freed by a node in the graph that owns it: that graph frees it, and a second
  // free from outside would be freeing an address the graph still allocates at.
  if (a.freed_in_owner) return cudaErrorInvalidValue;
  vgpu::MemoryManager& mm = s.rt->device(a.device).memory();
  GraphMemStats& st_ = g_graph_mem[a.device];
  if (a.in_use) st_.used -= std::min<uint64_t>(st_.used, a.size);
  a.in_use = false;
  if (a.mapped) {
    mm.unmap(a.va, a.reserved_size);
    a.mapped = false;
    st_.reserved -= std::min<uint64_t>(st_.reserved, a.reserved_size);
  }
  // The address stays while the node that allocates at it still exists: a later
  // launch of that graph allocates here again, which is the fixed address CUDA
  // documents. Once the graph is gone, nothing can, and the address goes too.
  if (a.owner_gone) {
    mm.address_free(a.va, a.reserved_size);
    g_graph_allocs.erase(it);
  }
  return cudaSuccess;
}

namespace {
// A graph allocation's fixed address, reserved on `device` (the current one
// when negative). The caller holds the runtime's lock and no graph lock -- the
// one order this file keeps.
cudaError_t reserve_graph_alloc(State& s, int device, size_t bytes, int* use_device,
                                uint64_t* va, uint64_t* reserved) {
  if (device >= 0) {
    if (device >= static_cast<int>(s.rt->device_count())) return cudaErrorInvalidDevice;
    *use_device = device;
  } else {
    *use_device = t_current_device;
  }
  vgpu::MemoryManager& mm = s.rt->device(*use_device).memory();
  *reserved = round_up_to(bytes, vgpu::MemoryManager::kVmmGranularity);
  *va = mm.reserve(*reserved, 0);
  return cudaSuccess;
}

// The allocation node, registered as owned by `owner` (a graph handle). The
// caller holds g_graph_mu.
GraphNodeRec* add_alloc_node(GraphRec& g, void* owner, const std::vector<GraphNodeRec*>& pred,
                             uint64_t va, uint64_t reserved, size_t bytes, int device) {
  RecordedLaunch w;
  w.dst = reinterpret_cast<void*>(va);
  w.bytes = bytes;
  GraphNodeRec* n = g.add(cudaGraphNodeTypeMemAlloc, std::move(w), pred);
  std::lock_guard<std::mutex> mem_lock(g_graph_mem_mu);
  GraphAlloc a;
  a.va = va;
  a.reserved_size = reserved;
  a.size = bytes;
  a.device = device;
  a.owner = owner;
  g_graph_allocs[va] = a;
  return n;
}

// The free node for a graph allocation, in the graph whose handle is `graph`.
// Refused -- nullptr, with *rc set -- for anything that is not a graph
// allocation, or one already freed by a node: an allocation is freed once, and
// in one graph, either the one that owns it or another, never both. Both rules
// are CUDA's, and both exist because a second free would be freeing an address
// nothing holds. The caller holds g_graph_mu.
GraphNodeRec* add_free_node(GraphRec& g, void* graph, const std::vector<GraphNodeRec*>& pred,
                            void* dptr, cudaError_t* rc) {
  {
    std::lock_guard<std::mutex> mem_lock(g_graph_mem_mu);
    const auto it = g_graph_allocs.find(reinterpret_cast<uint64_t>(dptr));
    if (it == g_graph_allocs.end() || it->second.freed_in_owner || it->second.freed_elsewhere) {
      *rc = cudaErrorInvalidValue;
      return nullptr;
    }
    if (it->second.owner == graph) it->second.freed_in_owner = true;
    else it->second.freed_elsewhere = true;
  }
  RecordedLaunch w;
  w.dst = dptr;
  *rc = cudaSuccess;
  return g.add(cudaGraphNodeTypeMemFree, std::move(w), pred);
}
}  // namespace

VGPU_EXPORT cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t* pNode, cudaGraph_t graph,
                                                 const cudaGraphNode_t* deps, size_t numDeps,
                                                 cudaMemAllocNodeParams* params) {
  if (!pNode || !params || params->bytesize == 0) return cudaErrorInvalidValue;
  // Sharing a graph allocation with another process would need a handle type
  // this does not offer, and the API documents IPC as unsupported here too.
  if (params->poolProps.handleTypes != cudaMemHandleTypeNone) return cudaErrorNotSupported;
  const int device = params->poolProps.location.type == cudaMemLocationTypeDevice
                         ? params->poolProps.location.id
                         : -1;
  uint64_t va = 0, reserved_size = 0;
  int use_device = 0;
  // The address is reserved now, under the runtime's lock, before the graph lock
  // is taken: that is the one lock order this file keeps.
  const cudaError_t rc = guard("cudaGraphAddMemAllocNode", [&](State& s) -> cudaError_t {
    return reserve_graph_alloc(s, device, params->bytesize, &use_device, &va, &reserved_size);
  });
  if (rc != cudaSuccess) return rc;

  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* g = graph_from(graph);
  std::vector<GraphNodeRec*> pred;
  if (!g || !deps_ok(*g, deps, numDeps, &pred)) return cudaErrorInvalidValue;
  GraphNodeRec* n = add_alloc_node(*g, static_cast<void*>(graph), pred, va, reserved_size,
                                   params->bytesize, use_device);
  params->dptr = reinterpret_cast<void*>(va);   // the address, now and for good
  *pNode = reinterpret_cast<cudaGraphNode_t>(n);
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node,
                                                      cudaMemAllocNodeParams* params) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto* n = reinterpret_cast<GraphNodeRec*>(node);
  if (!n || !params || n->type != cudaGraphNodeTypeMemAlloc) return cudaErrorInvalidValue;
  std::memset(params, 0, sizeof *params);
  params->bytesize = n->work.bytes;
  params->dptr = n->work.dst;
  params->poolProps.allocType = cudaMemAllocationTypePinned;
  params->poolProps.location.type = cudaMemLocationTypeDevice;
  {
    std::lock_guard<std::mutex> mem_lock(g_graph_mem_mu);
    const auto it = g_graph_allocs.find(reinterpret_cast<uint64_t>(n->work.dst));
    if (it != g_graph_allocs.end()) params->poolProps.location.id = it->second.device;
  }
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t* pNode, cudaGraph_t graph,
                                                const cudaGraphNode_t* deps, size_t numDeps,
                                                void* dptr) {
  if (!pNode || !dptr) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* g = graph_from(graph);
  std::vector<GraphNodeRec*> pred;
  if (!g || !deps_ok(*g, deps, numDeps, &pred)) return cudaErrorInvalidValue;
  cudaError_t rc = cudaSuccess;
  GraphNodeRec* n = add_free_node(*g, static_cast<void*>(graph), pred, dptr, &rc);
  if (!n) return rc;
  *pNode = reinterpret_cast<cudaGraphNode_t>(n);
  return cudaSuccess;
}

// ---- stream-ordered allocation and host functions on a capturing stream ----
//
// What CUDA documents each becomes in the capture's graph: cudaMallocAsync an
// allocation node the graph owns, cudaFreeAsync a free node (of a graph
// allocation, and nothing else), cudaLaunchHostFunc a host node. Each used to
// happen the moment it was called instead, so a captured malloc/free pair freed
// the memory before the graph ever ran and every launch used freed memory, and
// a host function ran once, at capture, and never again.
bool capture_malloc(State& s, cudaStream_t stream, size_t size, int device, void** ptr,
                    cudaError_t* rc) {
  if (!capture_active(stream)) return false;
  if (size == 0) {
    *ptr = nullptr;   // nothing to own; no node
    *rc = cudaSuccess;
    return true;
  }
  int use_device = 0;
  uint64_t va = 0, reserved = 0;
  *rc = reserve_graph_alloc(s, device, size, &use_device, &va, &reserved);
  if (*rc != cudaSuccess) return true;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  StreamCapture* sc = stream_capture(stream);
  if (!sc) {   // the capture ended while the address was reserved
    s.rt->device(use_device).memory().address_free(va, reserved);
    *rc = cudaErrorStreamCaptureInvalidated;
    return true;
  }
  GraphRec& g = *sc->cap->graph;
  GraphNodeRec* n = add_alloc_node(g, static_cast<void*>(&g), sc->deps, va, reserved, size,
                                   use_device);
  sc->deps.assign(1, n);
  *ptr = reinterpret_cast<void*>(va);
  *rc = cudaSuccess;
  return true;
}

bool capture_free(cudaStream_t stream, void* ptr, cudaError_t* rc) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  StreamCapture* sc = stream_capture(stream);
  if (!sc) return false;
  GraphRec& g = *sc->cap->graph;
  // Only a graph allocation can be freed by a node, as documented.
  GraphNodeRec* n = add_free_node(g, static_cast<void*>(&g), sc->deps, ptr, rc);
  if (n) sc->deps.assign(1, n);
  return true;
}

bool capture_host_fn(cudaStream_t stream, cudaHostFn_t fn, void* user) {
  RecordedLaunch r;
  r.kind = RecordedLaunch::Kind::Host;
  r.host_fn = fn;
  r.host_user = user;
  return capture_record(stream, cudaGraphNodeTypeHost, std::move(r));
}

// Where `stream`'s capture has got to, for an event recorded on it.
bool capture_position(cudaStream_t stream, unsigned long long* id, std::vector<void*>* deps) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  StreamCapture* sc = stream_capture(stream);
  if (!sc) return false;
  *id = sc->cap->id;
  deps->assign(sc->deps.begin(), sc->deps.end());
  return true;
}

// An event-record or event-wait node on a capturing stream, and where the
// stream is after it.
bool capture_event_node(cudaStream_t stream, bool record, cudaEvent_t e, unsigned long long* id,
                        std::vector<void*>* deps) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  StreamCapture* sc = stream_capture(stream);
  if (!sc) return false;
  RecordedLaunch w;
  w.event = e;
  capture_add(*sc, record ? cudaGraphNodeTypeEventRecord : cudaGraphNodeTypeWaitEvent,
              std::move(w));
  if (id) *id = sc->cap->id;
  if (deps) deps->assign(sc->deps.begin(), sc->deps.end());
  return true;
}

cudaError_t capture_wait(cudaStream_t stream, bool captured, unsigned long long id,
                         const std::vector<void*>& deps) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  StreamCapture* sc = stream_capture(stream);
  const auto invalidate = [&](const char* why) {
    GraphRec& g = *sc->cap->graph;
    if (!g.invalidated) {
      g.invalidated = true;
      g.invalidated_by = why;
    }
  };
  if (!captured) {
    if (!sc) return cudaSuccess;   // everything the event stood for has finished
    // A capture may depend only on work inside it; an event recorded outside is
    // a dependency across its boundary.
    invalidate("a wait on an event recorded outside the capture");
    return cudaErrorStreamCaptureIsolation;
  }
  Capture* cap = nullptr;
  for (auto& [origin, c] : g_captures)
    if (c->id == id) cap = c.get();
  if (!cap) return cudaErrorCapturedEvent;   // the capture it was recorded in has ended
  std::vector<GraphNodeRec*> pos;
  for (void* d : deps)
    if (auto* n = static_cast<GraphNodeRec*>(d); cap->graph->holds(n)) pos.push_back(n);
  if (sc) {
    if (sc->cap != cap) {
      invalidate("a wait that would merge two captures");
      return cudaErrorStreamCaptureMerge;
    }
    for (GraphNodeRec* n : pos)   // a join
      if (std::find(sc->deps.begin(), sc->deps.end(), n) == sc->deps.end()) sc->deps.push_back(n);
    return cudaSuccess;
  }
  // A fork: the stream joins the capture, starting where the event's stream was.
  // Not the legacy stream, which every other stream synchronizes with.
  if (stream == nullptr || stream == cudaStreamLegacy) return cudaErrorStreamCaptureImplicit;
  g_stream_capture[reinterpret_cast<void*>(stream)] = StreamCapture{cap, std::move(pos), {}};
  return cudaSuccess;
}

// An operation CUDA does not allow on a capturing stream. The capture it was
// called on is invalidated, so cudaStreamEndCapture says so and names it, and
// the call itself reports it -- rather than doing it now, outside the graph.
bool capture_refuse(cudaStream_t stream, const char* what) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  StreamCapture* sc = stream_capture(stream);
  if (!sc) return false;
  GraphRec& g = *sc->cap->graph;
  if (!g.invalidated) {
    g.invalidated = true;
    g.invalidated_by = what;
  }
  return true;
}

VGPU_EXPORT cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  auto* n = reinterpret_cast<GraphNodeRec*>(node);
  if (!n || !dptr_out || n->type != cudaGraphNodeTypeMemFree) return cudaErrorInvalidValue;
  *static_cast<void**>(dptr_out) = n->work.dst;
  return cudaSuccess;
}

// Gives back the memory behind graph allocations nothing is holding: those a
// free node has ended, and those whose node has never been reached. The
// addresses stay reserved, so a later launch allocates at the same place, which
// is the guarantee that makes a graph allocation usable at all.
VGPU_EXPORT cudaError_t cudaDeviceGraphMemTrim(int device) {
  return guard("cudaDeviceGraphMemTrim", [&](State& s) -> cudaError_t {
    if (device < 0 || device >= static_cast<int>(s.rt->device_count())) return cudaErrorInvalidDevice;
    std::lock_guard<std::mutex> lock(g_graph_mem_mu);
    vgpu::MemoryManager& mm = s.rt->device(device).memory();
    for (auto& [va, a] : g_graph_allocs) {
      if (a.device != device || a.in_use) continue;
      if (a.mapped) {
        mm.unmap(a.va, a.reserved_size);
        a.mapped = false;
        GraphMemStats& st_ = g_graph_mem[device];
        st_.reserved -= std::min<uint64_t>(st_.reserved, a.reserved_size);
      }
      // Nothing can allocate here again, so the address space goes back as well.
      if (a.owner_gone) mm.address_free(a.va, a.reserved_size);
    }
    std::erase_if(g_graph_allocs, [&](const auto& kv) {
      return kv.second.device == device && kv.second.owner_gone && !kv.second.in_use;
    });
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaDeviceGetGraphMemAttribute(int device,
                                                       cudaGraphMemAttributeType attr,
                                                       void* value) {
  if (!value) return cudaErrorInvalidValue;
  return guard("cudaDeviceGetGraphMemAttribute", [&](State& s) -> cudaError_t {
    if (device < 0 || device >= static_cast<int>(s.rt->device_count())) return cudaErrorInvalidDevice;
    std::lock_guard<std::mutex> lock(g_graph_mem_mu);
    const GraphMemStats& st_ = g_graph_mem[device];
    switch (attr) {
      case cudaGraphMemAttrUsedMemCurrent: *static_cast<uint64_t*>(value) = st_.used; return cudaSuccess;
      case cudaGraphMemAttrUsedMemHigh: *static_cast<uint64_t*>(value) = st_.used_high; return cudaSuccess;
      case cudaGraphMemAttrReservedMemCurrent:
        *static_cast<uint64_t*>(value) = st_.reserved;
        return cudaSuccess;
      case cudaGraphMemAttrReservedMemHigh:
        *static_cast<uint64_t*>(value) = st_.reserved_high;
        return cudaSuccess;
      default: return cudaErrorInvalidValue;
    }
  });
}

// A high-water mark is reset by writing zero to it, and nothing else can be
// written: the current totals are what they are.
VGPU_EXPORT cudaError_t cudaDeviceSetGraphMemAttribute(int device,
                                                       cudaGraphMemAttributeType attr,
                                                       void* value) {
  if (!value) return cudaErrorInvalidValue;
  return guard("cudaDeviceSetGraphMemAttribute", [&](State& s) -> cudaError_t {
    if (device < 0 || device >= static_cast<int>(s.rt->device_count())) return cudaErrorInvalidDevice;
    if (*static_cast<uint64_t*>(value) != 0) return cudaErrorInvalidValue;
    std::lock_guard<std::mutex> lock(g_graph_mem_mu);
    GraphMemStats& st_ = g_graph_mem[device];
    switch (attr) {
      case cudaGraphMemAttrUsedMemHigh: st_.used_high = st_.used; return cudaSuccess;
      case cudaGraphMemAttrReservedMemHigh: st_.reserved_high = st_.reserved; return cudaSuccess;
      default: return cudaErrorInvalidValue;   // the current totals are not settable
    }
  });
}

// CUDA 13 added an edge-data argument to each of these: ordering information
// for an edge, whose only value this engine can honour is the default. The
// shim is built against one toolkit, so each is exported with that toolkit's
// own signature -- C linkage makes a mismatch a hard error rather than a
// crash in somebody's program.
namespace {
// An edge-data array is accepted when it asks for the default edge; anything
// else describes ordering this engine does not model.
#if CUDART_VERSION >= 13000
bool default_edges(const cudaGraphEdgeData* data, size_t count) {
  if (!data) return true;
  for (size_t i = 0; i < count; ++i)
    if (data[i].from_port || data[i].to_port || data[i].type) return false;
  return true;
}
#endif
}  // namespace

static cudaError_t capture_info(cudaStream_t stream, cudaStreamCaptureStatus* status,
                                unsigned long long* id, cudaGraph_t* graph,
                                const cudaGraphNode_t** deps, size_t* numDeps);
static cudaError_t update_capture_deps(cudaStream_t stream, cudaGraphNode_t* dependencies,
                                       size_t numDependencies, unsigned int flags);

// The capture queries. A binary built against CUDA 12 headers calls
// cudaStreamGetCaptureInfo_v2 (the header maps the plain name to it), so that
// one is always exported. CUDA 13 made the plain name the edge-data form and
// gave cudaStreamUpdateCaptureDependencies an edge-data argument, so those are
// exported with whichever signature the toolkit this is built against declares.
// Without the CUDA 13 cudaStreamGetCaptureInfo, a program built with it could
// not resolve the symbol at all.
VGPU_EXPORT cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream,
                                                    cudaStreamCaptureStatus* status,
                                                    unsigned long long* id, cudaGraph_t* graph,
                                                    const cudaGraphNode_t** deps,
                                                    size_t* numDeps) {
  return capture_info(stream, status, id, graph, deps, numDeps);
}
#if CUDART_VERSION >= 13000
VGPU_EXPORT cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream,
                                                 cudaStreamCaptureStatus* status,
                                                 unsigned long long* id, cudaGraph_t* graph,
                                                 const cudaGraphNode_t** deps,
                                                 const cudaGraphEdgeData** edgeData,
                                                 size_t* numDeps) {
  size_t n = 0;
  const cudaError_t rc = capture_info(stream, status, id, graph, deps, &n);
  if (numDeps) *numDeps = n;
  if (edgeData) {
    // Every edge a capture makes is a plain dependency, so the edge data is the
    // default for each -- zeroed, as the API defines the default.
    static thread_local std::vector<cudaGraphEdgeData> edges;
    edges.assign(n, cudaGraphEdgeData{});
    *edgeData = n ? edges.data() : nullptr;
  }
  return rc;
}
VGPU_EXPORT cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream,
                                                            cudaGraphNode_t* dependencies,
                                                            const cudaGraphEdgeData* dependencyData,
                                                            size_t numDependencies,
                                                            unsigned int flags) {
  if (!default_edges(dependencyData, numDependencies)) return cudaErrorNotSupported;
  return update_capture_deps(stream, dependencies, numDependencies, flags);
}
#else
VGPU_EXPORT cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream,
                                                            cudaGraphNode_t* dependencies,
                                                            size_t numDependencies,
                                                            unsigned int flags) {
  return update_capture_deps(stream, dependencies, numDependencies, flags);
}
#endif

#if CUDART_VERSION >= 13000
VGPU_EXPORT cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t* from,
                                                const cudaGraphNode_t* to,
                                                const cudaGraphEdgeData* edgeData,
                                                size_t numDeps) {
  if (!default_edges(edgeData, numDeps)) return cudaErrorNotSupported;
  return graph_add_deps(graph, from, to, numDeps);
}
VGPU_EXPORT cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t* from,
                                                   const cudaGraphNode_t* to,
                                                   const cudaGraphEdgeData* edgeData,
                                                   size_t numDeps) {
  if (!default_edges(edgeData, numDeps)) return cudaErrorNotSupported;
  return graph_remove_deps(graph, from, to, numDeps);
}
VGPU_EXPORT cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from,
                                         cudaGraphNode_t* to, cudaGraphEdgeData* edgeData,
                                         size_t* numEdges) {
  const cudaError_t rc = graph_get_edges(graph, from, to, numEdges);
  // Every edge this engine has is a plain dependency.
  if (rc == cudaSuccess && edgeData && numEdges)
    std::memset(edgeData, 0, *numEdges * sizeof(cudaGraphEdgeData));
  return rc;
}
VGPU_EXPORT cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* deps,
                                                    cudaGraphEdgeData* edgeData, size_t* numDeps) {
  const cudaError_t rc = node_get_deps(node, deps, numDeps);
  if (rc == cudaSuccess && edgeData && numDeps)
    std::memset(edgeData, 0, *numDeps * sizeof(cudaGraphEdgeData));
  return rc;
}
VGPU_EXPORT cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node,
                                                      cudaGraphNode_t* dependent,
                                                      cudaGraphEdgeData* edgeData,
                                                      size_t* numDependentNodes) {
  const cudaError_t rc = node_get_dependents(node, dependent, numDependentNodes);
  if (rc == cudaSuccess && edgeData && numDependentNodes)
    std::memset(edgeData, 0, *numDependentNodes * sizeof(cudaGraphEdgeData));
  return rc;
}
#else
VGPU_EXPORT cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t* from,
                                                const cudaGraphNode_t* to, size_t numDeps) {
  return graph_add_deps(graph, from, to, numDeps);
}
VGPU_EXPORT cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t* from,
                                                   const cudaGraphNode_t* to, size_t numDeps) {
  return graph_remove_deps(graph, from, to, numDeps);
}
VGPU_EXPORT cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from,
                                         cudaGraphNode_t* to, size_t* numEdges) {
  return graph_get_edges(graph, from, to, numEdges);
}
VGPU_EXPORT cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* deps,
                                                    size_t* numDeps) {
  return node_get_deps(node, deps, numDeps);
}
VGPU_EXPORT cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node,
                                                      cudaGraphNode_t* dependent,
                                                      size_t* numDependentNodes) {
  return node_get_dependents(node, dependent, numDependentNodes);
}
#endif

// The graph as a DOT drawing: one node per node, one edge per dependency. It
// used to print an empty graph, which is a picture of nothing.
VGPU_EXPORT cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int) {
  if (!path) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* g = graph_from(graph);
  if (!g) return cudaErrorInvalidValue;
  std::FILE* f = std::fopen(path, "w");
  if (!f) return cudaErrorOperatingSystem;
  const auto label = [](cudaGraphNodeType t) {
    switch (t) {
      case cudaGraphNodeTypeKernel: return "KERNEL";
      case cudaGraphNodeTypeMemcpy: return "MEMCPY";
      case cudaGraphNodeTypeMemset: return "MEMSET";
      case cudaGraphNodeTypeHost: return "HOST";
      case cudaGraphNodeTypeGraph: return "GRAPH";
      case cudaGraphNodeTypeEmpty: return "EMPTY";
      case cudaGraphNodeTypeEventRecord: return "EVENT_RECORD";
      case cudaGraphNodeTypeWaitEvent: return "WAIT_EVENT";
      default: return "NODE";
    }
  };
  std::map<const GraphNodeRec*, size_t> index;
  for (size_t i = 0; i < g->nodes.size(); ++i) index[g->nodes[i].get()] = i;
  std::fprintf(f, "digraph dot {\n");
  for (size_t i = 0; i < g->nodes.size(); ++i)
    std::fprintf(f, "  \"graph_%zu\" [label=\"%zu\\n%s\"]\n", i, i, label(g->nodes[i]->type));
  for (size_t i = 0; i < g->nodes.size(); ++i)
    for (const GraphNodeRec* d : g->nodes[i]->deps)
      std::fprintf(f, "  \"graph_%zu\" -> \"graph_%zu\"\n", index[d], i);
  std::fprintf(f, "}\n");
  std::fclose(f);
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  // A child graph belongs to the node that holds it; destroying it here would
  // free memory the node still points at.
  if (g_borrowed_graphs.count(static_cast<void*>(graph)) &&
      !g_graphs.count(static_cast<void*>(graph)))
    return cudaErrorInvalidValue;
  if (GraphRec* g = graph_from(graph))
    for (const auto& child : g->children) std::erase_if(
        g_borrowed_graphs, [&](const auto& kv) { return kv.second == child.get(); });
  // Allocations this graph made outlive it, as CUDA documents: a program still
  // holding one frees it itself. What ends with the graph is the chance of ever
  // allocating at that address again, so the address can be given back -- by the
  // free that ends the allocation, or by the next trim.
  {
    std::lock_guard<std::mutex> mem_lock(g_graph_mem_mu);
    for (auto& [va, a] : g_graph_allocs)
      if (a.owner == static_cast<void*>(graph)) a.owner_gone = true;
  }
  g_graphs.erase(static_cast<void*>(graph));
  return cudaSuccess;
}
VGPU_EXPORT cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  g_graph_execs.erase(static_cast<void*>(exec));
  g_exec_source.erase(static_cast<void*>(exec));
  g_exec_auto_free.erase(static_cast<void*>(exec));
  return cudaSuccess;
}
// What a library asks before it adds work to a stream that might be capturing:
// whether it is, which capture, the graph being built, and the nodes the next
// operation will depend on. With the graph and the dependencies a library can
// add its own nodes to the capture and then say they come next, which is what
// cudaStreamUpdateCaptureDependencies is for. This used to report an active
// capture with no graph and no dependencies, so a library that did that was
// handed a null graph.
static cudaError_t capture_info(cudaStream_t stream, cudaStreamCaptureStatus* status,
                                unsigned long long* id, cudaGraph_t* graph,
                                const cudaGraphNode_t** deps, size_t* numDeps) {
  if (!status) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  StreamCapture* sc = stream_capture(stream);
  if (!sc) {
    *status = cudaStreamCaptureStatusNone;
    return cudaSuccess;   // the other outputs mean something only while capturing
  }
  GraphRec& g = *sc->cap->graph;
  *status = g.invalidated ? cudaStreamCaptureStatusInvalidated : cudaStreamCaptureStatusActive;
  if (id) *id = sc->cap->id;   // the same for every stream in the capture
  if (graph) *graph = static_cast<cudaGraph_t>(static_cast<void*>(&g));
  // This stream's own position. The array stays valid until the next call that
  // names this stream, which is what the API promises and why it is kept here.
  sc->deps_out.assign(sc->deps.size(), nullptr);
  for (size_t i = 0; i < sc->deps.size(); ++i)
    sc->deps_out[i] = reinterpret_cast<cudaGraphNode_t>(sc->deps[i]);
  if (deps) *deps = sc->deps_out.empty() ? nullptr : sc->deps_out.data();
  if (numDeps) *numDeps = sc->deps_out.size();
  return cudaSuccess;
}

// Replaces or extends the set of nodes the next captured operation depends on.
// The nodes have to be in the graph being captured: a dependency on anything
// else would join two graphs, which is not something a capture can do.
static cudaError_t update_capture_deps(cudaStream_t stream, cudaGraphNode_t* dependencies,
                                       size_t numDependencies, unsigned int flags) {
  if (flags != cudaStreamAddCaptureDependencies && flags != cudaStreamSetCaptureDependencies)
    return cudaErrorInvalidValue;
  if (numDependencies && !dependencies) return cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  StreamCapture* sc = stream_capture(stream);
  if (!sc) return cudaErrorIllegalState;   // as documented
  const GraphRec& g = *sc->cap->graph;
  std::vector<GraphNodeRec*> named;
  for (size_t i = 0; i < numDependencies; ++i) {
    auto* n = reinterpret_cast<GraphNodeRec*>(dependencies[i]);
    if (!n || !g.holds(n)) return cudaErrorInvalidValue;
    named.push_back(n);
  }
  if (flags == cudaStreamSetCaptureDependencies) sc->deps.clear();
  for (GraphNodeRec* n : named)
    if (std::find(sc->deps.begin(), sc->deps.end(), n) == sc->deps.end()) sc->deps.push_back(n);
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaStreamIsCapturing(cudaStream_t stream,
                                              cudaStreamCaptureStatus* status) {
  if (status)
    *status = capture_target(stream) ? cudaStreamCaptureStatusActive
                                     : cudaStreamCaptureStatusNone;
  return cudaSuccess;
}
