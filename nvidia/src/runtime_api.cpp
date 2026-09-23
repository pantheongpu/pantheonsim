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
    // The PTX the driver would JIT for this device: the newest whose target
    // it can run -- no newer than its compute capability -- as a fatbin built
    // for many architectures (sm_75 ... sm_121) carries one per target. Only
    // when none qualifies, the newest of all, which is then refused at load
    // with the reason. (Every device of a simulated machine has one profile.)
    const vgpu::DeviceProfile& dev = s.rt->device(0).profile();
    const uint32_t cc = static_cast<uint32_t>(dev.cc_major * 10 + dev.cc_minor);
    auto pick_best = [cc](std::vector<vgpu::cuda::FatbinPtx>& v) -> std::string {
      if (v.empty()) return {};
      size_t best = v.size();
      for (size_t i = 0; i < v.size(); ++i)
        if (v[i].arch <= cc && (best == v.size() || v[i].arch > v[best].arch)) best = i;
      if (best == v.size()) {
        best = 0;
        for (size_t i = 1; i < v.size(); ++i)
          if (v[i].arch > v[best].arch) best = i;
      }
      return std::move(v[best].text);
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

VGPU_EXPORT cudaError_t cudaMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count,
                                                size_t offset, cudaMemcpyKind kind, cudaStream_t) {
  return cudaMemcpyToSymbol(symbol, src, count, offset, kind);
}
VGPU_EXPORT cudaError_t cudaMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t count,
                                                  size_t offset, cudaMemcpyKind kind,
                                                  cudaStream_t) {
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
                                            int srcDevice, size_t count, cudaStream_t) {
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
VGPU_EXPORT cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms* p, cudaStream_t) {
  return cudaMemcpy3DPeer(p);
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

VGPU_EXPORT cudaError_t cudaMallocAsync(void** ptr, size_t size, cudaStream_t) {
  return guard("cudaMallocAsync", [&](State& s) -> cudaError_t {
    if (!ptr) return cudaErrorInvalidValue;
    if (vgpu::faults::should_fail(vgpu::faults::Op::Alloc)) return cudaErrorMemoryAllocation;
    return pool_alloc(s, pool_for(s, t_current_device), size, ptr);
  });
}

VGPU_EXPORT cudaError_t cudaMallocFromPoolAsync(void** ptr, size_t size, cudaMemPool_t pool,
                                                cudaStream_t) {
  return guard("cudaMallocFromPoolAsync", [&](State& s) -> cudaError_t {
    if (!ptr) return cudaErrorInvalidValue;
    MemPool* p = from_handle(s, pool);
    if (!p) return cudaErrorInvalidValue;
    return pool_alloc(s, *p, size, ptr);
  });
}

VGPU_EXPORT cudaError_t cudaFreeAsync(void* ptr, cudaStream_t) {
  return guard("cudaFreeAsync", [&](State& s) -> cudaError_t {
    if (!ptr) return cudaSuccess;   // as cudaFree(nullptr) is
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
};
std::mutex g_event_mu;
std::unordered_map<cudaEvent_t, std::unique_ptr<RtEvent>> g_events;
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
VGPU_EXPORT cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t) {
  std::lock_guard<std::mutex> lock(g_event_mu);
  RtEvent* r = find_event(e);
  if (!r) return cudaErrorInvalidResourceHandle;
  r->recorded = true;
  r->when = std::chrono::steady_clock::now();
  return cudaSuccess;
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
  return find_event(e) ? cudaSuccess : cudaErrorInvalidResourceHandle;
}
VGPU_EXPORT cudaError_t cudaEventQuery(cudaEvent_t e) {
  std::lock_guard<std::mutex> lock(g_event_mu);
  return find_event(e) ? cudaSuccess : cudaErrorInvalidResourceHandle;
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
