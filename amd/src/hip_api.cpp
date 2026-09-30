// libamdhip64: the HIP runtime API, over VirtualGPU's devices.
//
// A HIP program asks for devices and memory, loads a code object, and
// launches the kernels in it. That is what this implements: the devices come
// from a profile (VGPU_GPU, an AMD one), the memory from the device's manager,
// and a launch decodes and runs the kernel's own CDNA instructions
// (vgpu/amd_exec.hpp).
//
// The interface is the documented HIP one (amd/include/vgpu_hip.h, written
// from AMD's public documentation); nothing here is AMD's code. What is not
// implemented is refused by name rather than ignored, because a HIP program
// that believes a launch happened will compare wrong answers.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <mutex>
#include <string>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "vgpu/amd_bundle.hpp"
#include "vgpu/amd_chip.hpp"
#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_decode_cache.hpp"
#include "vgpu/amd_exec.hpp"
#include "vgpu/amd_hostcall.hpp"
#include "vgpu/amd_kfd.hpp"
#include "vgpu/error.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"
#include "vgpu/telemetry.hpp"
#include "vgpu/hip_abi.hpp"
#include "vgpu/hip_profiler.hpp"
#include "hip_queue.hpp"
#include "hip_shared.hpp"
#include "vgpu_hip.h"

namespace {

using vgpu::amd::CodeObject;
using vgpu::amd::Kernel;

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

// Says what went wrong and why, once, where a program would otherwise only
// see an error code.
hipError_t fail(hipError_t code, const std::string& what) {
  if (!quiet()) std::fprintf(stderr, "VirtualGPU HIP: %s\n", what.c_str());
  return code;
}

struct Module {
  CodeObject object;
  int device = 0;         // the device it was loaded for, which its functions launch on
  uint64_t globals = 0;   // where the module's own variables were placed
  // Where a linked module's image is (CodeObject::image): its code runs from
  // there, and its variables are in it, so `globals` is the same address.
  uint64_t code_base = 0;
  uint64_t code_size = 0;   // how much of the device the image takes
  // What a profiler knows it and its kernels by (vgpu/hip_profiler.hpp).
  uint64_t code_object_id = 0;
  std::vector<uint64_t> kernel_ids;   // one per kernel, in the object's order
  // Its code as its launches have decoded it, made once it is placed.
  std::unique_ptr<vgpu::amd::DecodeCache> decoded;
};

// ---- What a profiler is told -----------------------------------------------
//
// A profiler that attaches (vgpu/hip_profiler.hpp) hears of every HIP call,
// every code object placed on a device, every launch with what it counted,
// every copy and every allocation. Nothing is attached unless one is.
std::atomic<const vgpu::amd::hipprof::Hooks*> g_profiler{nullptr};
std::atomic<uint64_t> g_next_code_object{1}, g_next_kernel{1}, g_next_dispatch{1};

const vgpu::amd::hipprof::Hooks* profiler() { return g_profiler.load(std::memory_order_acquire); }

// One HIP call, from the outermost: a call one HIP function makes through
// another is part of the first, as it is in ROCm's runtime.
class ApiCall {
 public:
  explicit ApiCall(const char* name) {
    if (depth()++ != 0) return;
    // VGPU_TRACE_API=1 names each HIP call a program makes, one line to
    // stderr: what a library does through HIP, when it does not do what it
    // should, is otherwise invisible.
    static const bool trace = [] {
      const char* t = std::getenv("VGPU_TRACE_API");
      return t && t[0] == '1';
    }();
    if (trace) std::fprintf(stderr, "VirtualGPU HIP: %s\n", name);
    if (const auto* p = profiler(); p && p->api_enter) {
      hooks_ = p;
      token_ = p->api_enter(name);
    }
  }
  ~ApiCall() {
    --depth();
    if (hooks_ && hooks_->api_exit) hooks_->api_exit(token_);
  }
  ApiCall(const ApiCall&) = delete;
  ApiCall& operator=(const ApiCall&) = delete;

 private:
  static int& depth() {
    thread_local int d = 0;
    return d;
  }
  const vgpu::amd::hipprof::Hooks* hooks_ = nullptr;
  uint64_t token_ = 0;
};

// A module placed on a device gets its ids here, and a profiler is told of
// it and its kernels.
void report_loaded(Module& m, int device, const void* image, size_t image_size) {
  m.code_object_id = g_next_code_object.fetch_add(1);
  m.kernel_ids.clear();
  for (size_t i = 0; i < m.object.kernels.size(); ++i) m.kernel_ids.push_back(g_next_kernel.fetch_add(1));
  const auto* p = profiler();
  if (!p || !p->code_object_loaded) return;
  // Where ROCm's loader says a code object came from when it was handed one
  // in memory rather than a file.
  char uri[128];
  std::snprintf(uri, sizeof uri, "memory://%d#offset=0x%llx&size=%zu", static_cast<int>(::getpid()),
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(image)), image_size);
  vgpu::amd::hipprof::CodeObject o;
  o.id = m.code_object_id;
  o.device = device;
  o.uri = uri;
  o.image = image;
  o.image_size = image_size;
  o.load_base = m.object.linked ? m.code_base : m.object.text_addr;
  o.load_size = m.object.linked ? m.code_size : m.object.text.size();
  std::vector<vgpu::amd::hipprof::KernelSymbol> symbols;
  for (size_t i = 0; i < m.object.kernels.size(); ++i) {
    const Kernel& k = m.object.kernels[i];
    vgpu::amd::hipprof::KernelSymbol sym;
    sym.kernel_id = m.kernel_ids[i];
    sym.name = k.name.c_str();
    sym.entry = m.code_base + k.entry;
    sym.kernarg_size = k.kernarg_size;
    sym.kernarg_alignment = k.kernarg_align;
    sym.group_segment = k.group_segment;
    sym.private_segment = k.private_segment;
    sym.sgprs = k.sgpr_count;
    sym.vgprs = k.vgpr_count;
    sym.agprs = k.agpr_count;
    symbols.push_back(sym);
  }
  p->code_object_loaded(o, symbols.data(), symbols.size());
}

uint64_t kernel_id(const Module& m, const Kernel& k) {
  const size_t i = static_cast<size_t>(&k - m.object.kernels.data());
  return i < m.kernel_ids.size() ? m.kernel_ids[i] : 0;
}

// A kernel a program has looked up: the module it came from, and which kernel.
struct Function {
  Module* module = nullptr;
  const Kernel* kernel = nullptr;
};

// ---- Programs built by hipcc ---------------------------------------------
//
// A program built by hipcc never loads a module. Its device code is inside the
// executable, and before main it hands that to the runtime and says which of
// its host-side functions stands for which kernel; a chevron launch then hands
// the runtime the host-side function. The device code is a clang offload
// bundle -- one entry per target -- and a device takes the entry for its own
// gfx target, loaded and placed the first time a kernel runs on it, since each
// device has memory of its own for the module's variables. The bundle is only
// read then: a library registers every one it carries as it loads, dozens,
// some tens of megabytes once inflated, and a program uses few of them.
struct FatBinary {
  const uint8_t* image = nullptr;                   // the bundle, in the program's own memory
  std::map<int, std::unique_ptr<Module>> on_device;
};

struct HostFunction {
  FatBinary* binary = nullptr;
  std::string kernel;
};

// A __device__ or __constant__ variable of a hipcc-built program: the host
// has a variable of its own that stands for it (its address is what
// hipMemcpyToSymbol is given), and the device's lives in the binary's module.
struct HostVar {
  FatBinary* binary = nullptr;
  std::string name;
  size_t size = 0;
};

// What a stream was made with, which a program can ask for back, and the
// queue its work runs on.
using Queue = vgpu::amd::WorkQueue<hipError_t>;

struct Stream {
  int device = 0;
  unsigned flags = 0;
  int priority = 0;
  std::vector<uint32_t> cu_mask;   // empty: every compute unit
  std::shared_ptr<Queue> queue;    // its work, in order (hip_queue.hpp)
  // What hipStreamGetId says: a number no other stream in the process has
  // had or will have, handles being reused as they are.
  unsigned long long id = next_stream_id();
  // hipStreamSetAttribute's values, by attribute, kept as set: none changes
  // how the stream runs here. The synchronization policy starts at Auto.
  std::map<int, vgpu::amd::abi::LaunchAttributeValue> attrs;
  // How far into each other queue this stream has waited (hipStreamWaitEvent):
  // memory freed there before that point is the stream's to reuse.
  std::map<const Queue*, uint64_t> waited;

  static unsigned long long next_stream_id() {
    static std::atomic<unsigned long long> next{1};
    return next++;
  }
};

// A stream-ordered memory pool. What is freed to it is kept for reuse until
// a synchronization finds more kept than the release threshold, or a trim
// asks for less: a block freed in a stream is the next allocation's in that
// stream at once (the stream's order keeps the two apart), and another
// stream's where it waited past the free, or once the free has run, as the
// reuse attributes allow. What it holds in use and kept is what it has
// reserved; its attributes are what was set.
struct Pool {
  int device = 0;
  bool destroyed = false;
  std::map<int, int> access;   // other devices' hipMemAccessFlags, as granted
  uint64_t release_threshold = 0;
  uint64_t max_size = 0;       // 0: as much as the device has
  int reuse[4] = {0, 1, 1, 1};   // by hipMemPoolAttr 1 to 3
  int handle_types = 0;          // hipMemAllocationHandleType it may be exported as
  // A pool exported to other processes, or imported from one, reads and
  // writes its grants through this file (PoolShare, below); -1 if neither.
  int share_fd = -1;
  bool imported = false;
  uint64_t used = 0, used_high = 0, reserved_high = 0;
  struct Kept {
    uint64_t va = 0, size = 0;
    std::shared_ptr<Queue> queue;   // where it was freed, and at which item
    uint64_t seq = 0;
  };
  std::vector<Kept> kept;
  uint64_t reserved() const {
    uint64_t r = used;
    for (const Kept& k : kept) r += k.size;
    return r;
  }
};

// A shared pool's file: who made the pool, on which device, and what each
// device has been granted, which both processes read as it changes.
struct PoolShare {
  uint32_t magic, version, pid;
  int32_t device;
  int32_t access[64];   // hipMemAccessFlags by device; the pool's own is read-write
};
constexpr uint32_t kPoolShareMagic = 0x48505348;   // "HPSH"
void publish_pool_access(const Pool& p) {
  if (p.share_fd < 0) return;
  int32_t access[64] = {};
  for (const auto& [dev, flags] : p.access)
    if (dev >= 0 && dev < 64) access[dev] = flags;
  (void)!::pwrite(p.share_fd, access, sizeof access, offsetof(PoolShare, access));
}
void refresh_pool_access(Pool& p) {
  if (p.share_fd < 0 || !p.imported) return;
  int32_t access[64];
  if (::pread(p.share_fd, access, sizeof access, offsetof(PoolShare, access)) != sizeof access) return;
  p.access.clear();
  for (int dev = 0; dev < 64; ++dev)
    if (access[dev] && dev != p.device) p.access[dev] = access[dev];
}
std::map<void*, int> g_pool_imports;   // imported pool pointer -> the device it was mapped on

struct State {
  std::mutex mutex;
  std::unique_ptr<vgpu::runtime::Runtime> rt;
  std::vector<std::unique_ptr<Module>> modules;
  std::vector<std::unique_ptr<Function>> functions;
  std::vector<std::unique_ptr<FatBinary>> fat_binaries;
  // A program's bundles, read once for each target a device of it runs:
  // only that target's code is kept (vgpu/amd_bundle.hpp).
  std::map<std::pair<const uint8_t*, std::string>, std::unique_ptr<vgpu::amd::Bundle>> bundles;
  std::map<const void*, HostFunction> host_functions;
  std::map<const void*, HostVar> host_vars;
  // A program's __managed__ variables: each is managed memory, which the
  // host reaches through a pointer of its own and a device's code through a
  // pointer variable of the same name in the binary's module.
  struct ManagedVar {
    FatBinary* binary;
    std::string name;
    void** pointer;
  };
  std::vector<ManagedVar> managed_vars;
  std::set<std::pair<int, int>> peers;           // (device, peer) pairs with access enabled
  // Devices whose memory holds another process's opened IPC allocations:
  // every other device reaches them, without hipDeviceEnablePeerAccess.
  std::set<int> ipc_mapped;
  // Each device's hostcall buffer, made when a kernel is first launched on it:
  // what device-side printf writes through (vgpu/amd_hostcall.hpp).
  std::map<int, std::unique_ptr<vgpu::amd::Hostcall>> hostcalls;
  std::map<hipStream_t, Stream> streams;
  std::deque<Pool> pools;                       // a pool's address is its handle
  std::map<int, Pool*> default_pools;           // by device
  std::map<uint64_t, std::pair<Pool*, uint64_t>> pool_allocations;   // address -> pool, bytes
  // The device hipSetDevice chose, for the calling host thread alone, as HIP
  // (and CUDA) documents it: each new thread starts on device 0. RCCL gives
  // each GPU a thread of its own that sets its device; with one current
  // device for the process, every thread ended up on the last one set.
  static thread_local int current;
  // Each device's null stream: the legacy default stream, which work on the
  // blocking streams waits behind and which waits behind theirs.
  std::map<int, std::shared_ptr<Queue>> null_queues;
  // What the calling thread's last HIP call returned, as HIP keeps it: per
  // thread, so a stream's worker never overwrites a program thread's.
  static thread_local hipError_t last;
  std::string profile_id;
  // hipSetDeviceFlags' schedule bits (or hipCtxCreate's flags, whole), by
  // device, for any thread to read back.
  std::map<int, unsigned> device_flags;
  // Each device's scratch limit, as last set (shared::scratch_limit).
  std::map<int, size_t> scratch_limit;
  // The pool hipMallocAsync takes from, by device, where a program set one.
  std::map<int, Pool*> current_pools;
  // Virtual memory another device was granted: by mapped range, each
  // device's hipMemAccessFlags.
  std::map<std::pair<uint64_t, uint64_t>, std::map<int, int>> vmm_access;
  // Device allocations, by where they start, whose copies are always
  // synchronous (HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS, set by
  // hipPointerSetAttribute).
  std::set<uint64_t> sync_memops;
  // Whether the calling thread chose its device with hipSetDevice, which
  // hipSetValidDevices then leaves alone.
  static thread_local bool device_chosen;
};

thread_local int State::current = 0;
thread_local hipError_t State::last = hipSuccess;
thread_local bool State::device_chosen = false;

// Never destroyed: a library's exit handlers (a fat binary unregistered by
// the program's module destructor, ROCm's HIP tearing down its programs)
// still reach the runtime after static destructors would have run. The
// memory it holds goes with the process; files it spills to are unlinked
// already.
State& state() {
  static State& s = *new State;
  return s;
}

// The rack the environment describes. An NVIDIA profile is refused here: a
// HIP program on an NVIDIA GPU is a mistake worth naming, not a silent
// translation.
hipError_t ensure_runtime(State& s) {
  if (s.rt) return hipSuccess;
  const char* gpu = std::getenv("VGPU_GPU");
  const std::string id = gpu && *gpu ? gpu : "amd/mi300x";
  int count = 1;
  if (const char* c = std::getenv("VGPU_DEVICE_COUNT"); c && *c) count = std::atoi(c);
  if (count < 1) count = 1;
  vgpu::DeviceProfile p;
  try {
    p = vgpu::load_gpu(id);
  } catch (const std::exception& e) {
    return fail(hipErrorInvalidDevice, std::string("no usable GPU profile: ") + e.what());
  }
  // The devices a program is shown, as ROCm shows them: ROCR_VISIBLE_DEVICES
  // picks from the machine's, then HIP_VISIBLE_DEVICES (or
  // CUDA_VISIBLE_DEVICES) from those -- each a list read up to the first
  // entry that names no device. An entry is an index, or a UUID: "GPU-" and
  // the hex of either the HSA agent's UUID or KFD's unique_id, which is what
  // Ollama passes. None shown is no device.
  const int machine = count;
  const auto uuid_of = [&](const std::string& item) -> int {
    if (item.rfind("GPU-", 0) != 0 || item.size() == 4) return -1;
    char* end = nullptr;
    const unsigned long long want = std::strtoull(item.c_str() + 4, &end, 16);
    if (*end) return -1;
    for (int i = 0; i < machine; ++i) {
      vgpu::telemetry::DeviceSample d;
      vgpu::telemetry::describe_device(p, i, &d);
      if (want == 0x5647505500000000ull + static_cast<unsigned>(i) || want == vgpu::amd::kfd_unique_id(d.uuid))
        return i;
    }
    return -1;
  };
  std::vector<int> visible(static_cast<size_t>(machine));
  for (int i = 0; i < machine; ++i) visible[static_cast<size_t>(i)] = i;
  const auto shown = [&](const char* var) {
    const char* v = std::getenv(var);
    if (!v) return;
    std::vector<int> picked;
    std::stringstream list(v);
    for (std::string item; std::getline(list, item, ',');) {
      int at = -1;   // a position in `visible`
      if (!item.empty() && item.find_first_not_of("0123456789") == std::string::npos) {
        at = std::atoi(item.c_str());
        if (at >= static_cast<int>(visible.size())) at = -1;
      } else if (const int dev = uuid_of(item); dev >= 0) {
        const auto it = std::find(visible.begin(), visible.end(), dev);
        if (it != visible.end()) at = static_cast<int>(it - visible.begin());
      }
      if (at < 0) break;
      const int dev = visible[static_cast<size_t>(at)];
      if (std::find(picked.begin(), picked.end(), dev) != picked.end()) break;
      picked.push_back(dev);
    }
    visible = picked;
  };
  shown("ROCR_VISIBLE_DEVICES");
  shown(std::getenv("HIP_VISIBLE_DEVICES") ? "HIP_VISIBLE_DEVICES" : "CUDA_VISIBLE_DEVICES");
  count = static_cast<int>(visible.size());
  if (count < 1) return fail(hipErrorNoDevice, "no device is visible (ROCR_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES)");
  try {
    if (p.vendor != "amd")
      return fail(hipErrorInvalidDevice, "VGPU_GPU=" + id +
                                             " is not an AMD GPU, and HIP runs on AMD GPUs. Set VGPU_GPU to an "
                                             "amd/ profile (for example VGPU_GPU=amd/mi300x).");
    vgpu::apply_vram_override(p);
    s.rt = std::make_unique<vgpu::runtime::Runtime>(p, count);
    s.profile_id = p.id;
  } catch (const std::exception& e) {
    return fail(hipErrorInvalidDevice, std::string("no usable GPU profile: ") + e.what());
  }
  return hipSuccess;
}

vgpu::runtime::Device* device(State& s) {
  if (ensure_runtime(s) != hipSuccess) return nullptr;
  if (s.current < 0 || s.current >= s.rt->device_count()) return nullptr;
  return &s.rt->device(s.current);
}

// What the calling thread's next hipGetLastError says: the last call that
// failed, as ROCm's HIP keeps it -- a call that succeeds leaves an earlier
// failure where it is, and hipErrorNotReady (a query's answer) is no failure.
hipError_t record(State& s, hipError_t e) {
  if (e != hipSuccess && e != hipErrorNotReady) s.last = e;
  return e;
}

// Stream capture (hip_graph.inc). The caller holds s.mutex for each, and
// passes a stream handle already resolved.
struct Event;
bool stream_capturing(State& s, hipStream_t resolved);
// Whether a capture takes in any stream of the device.
bool any_capture_on(State& s, int device);
// Work on the legacy null stream while a blocking stream of `device`
// captures: hipErrorStreamCaptureImplicit, and those captures are given up.
hipError_t legacy_conflict(State& s, int device);
// A call HIP calls potentially unsafe during a capture (allocating host
// memory, synchronizing): hipErrorStreamCaptureUnsupported where the calling
// thread's capture mode says so, and the captures it would upset are given up.
hipError_t unsafe_during_capture(State& s);
hipError_t capture_kernel(State& s, hipStream_t resolved, const void* func, const Kernel& k,
                          vgpu::amd::abi::Dim3 grid, vgpu::amd::abi::Dim3 block, uint32_t shared,
                          std::vector<uint8_t> args, void** extra, bool cooperative);
hipError_t capture_host(State& s, hipStream_t resolved, void (*fn)(void*), void* data);
hipError_t capture_alloc(State& s, hipStream_t resolved, int device, size_t bytes, void** ptr);
hipError_t capture_free(State& s, hipStream_t resolved, void* ptr);
hipError_t capture_record(State& s, hipStream_t resolved, Event* e);
bool capture_wait(State& s, hipStream_t resolved, const Event* e, hipError_t* result);
bool captured_event(const Event* e);
void forget_stream(State& s, hipStream_t h);
// Graph memory (an allocation node's) freed by hipFree, and how big it is.
bool free_graph_memory(State& s, void* ptr, hipError_t* result);
bool graph_memory_size(uint64_t va, uint64_t* base, uint64_t* size);
// The same for any address an allocation node set aside, mapped or not yet:
// a node may name it before the graph that maps it runs.
bool graph_memory_range(uint64_t va, uint64_t* base, uint64_t* size);

// ---- Stream order ------------------------------------------------------------
//
// Work runs in stream order on each stream's queue (hip_queue.hpp), and
// streams run at once. The null stream is the legacy default stream: its work
// waits for what the blocking streams (those made without
// hipStreamNonBlocking) already have, and theirs for what it already has.
// VGPU_SYNC_LAUNCHES=1 makes every call wait for its own work, as this
// runtime did before streams ran at once -- a way to rule them out.
//
// No thread waits on a queue while it holds s.mutex, and work takes it only
// briefly: a queue's work may be what another thread's wait is for.
bool synchronous_launches() {
  static const bool v = [] {
    const char* e = std::getenv("VGPU_SYNC_LAUNCHES");
    return e && e[0] == '1';
  }();
  return v;
}

// HIP's two reserved handles besides the null stream: hipStreamLegacy (1),
// the legacy default stream the null handle also names, and
// hipStreamPerThread (2), the calling thread's own default stream on the
// current device. A per-thread stream is a blocking stream like one a program
// makes -- it waits for the legacy stream's work and the legacy stream for
// its -- made the first time the thread names it and gone when the thread
// is: its handle is in s.streams, so everything that finds a stream finds it.
// The _spt forms of HIP's calls, which a program built with
// -fgpu-default-stream=per-thread calls, take the null handle as this one.
constexpr intptr_t kStreamLegacy = 1;
constexpr intptr_t kStreamPerThread = 2;

// This thread's per-thread streams, one per device, removed when it ends:
// what they still have to do runs to the end first.
struct PerThreadStreams {
  std::map<int, hipStream_t> by_device;
  ~PerThreadStreams();
};
thread_local PerThreadStreams t_per_thread;

hipStream_t per_thread_stream(State& s, int device) {   // the caller holds s.mutex
  hipStream_t& h = t_per_thread.by_device[device];
  if (!h) {
    static intptr_t next = 0x10000000;   // apart from hipStreamCreate's handles
    h = reinterpret_cast<hipStream_t>(next++);
    s.streams[h] = Stream{device, 0, 0, {}, nullptr};
  }
  return h;
}

// The stream a handle names: the null stream for either legacy handle, the
// thread's own for hipStreamPerThread, and any other as it is. The caller
// holds s.mutex.
hipStream_t resolve(State& s, hipStream_t stream) {
  const intptr_t v = reinterpret_cast<intptr_t>(stream);
  if (v == kStreamLegacy) return nullptr;
  if (v == kStreamPerThread) return per_thread_stream(s, s.current);
  return stream;
}
// What an _spt call is given: its null handle is the thread's own stream.
hipStream_t spt(hipStream_t stream) { return stream ? stream : reinterpret_cast<hipStream_t>(kStreamPerThread); }

std::shared_ptr<Queue> null_queue(State& s, int device) {   // the caller holds s.mutex
  std::shared_ptr<Queue>& q = s.null_queues[device];
  if (!q) q = std::make_shared<Queue>(hipSuccess, hipErrorLaunchFailure);
  return q;
}

struct Order {
  std::shared_ptr<Queue> queue;
  std::vector<Queue::Marker> after;
  int device = 0;
};

// Where work on `stream` goes and what it must follow. The caller holds
// s.mutex.
hipError_t order_for(State& s, hipStream_t stream, Order* o) {
  stream = resolve(s, stream);
  if (!stream) {
    o->device = s.current;
    o->queue = null_queue(s, o->device);
    for (auto& [handle, st] : s.streams)
      if (st.device == o->device && !(st.flags & hipStreamNonBlocking) && st.queue && !st.queue->idle())
        o->after.push_back({st.queue, st.queue->tail()});
    return hipSuccess;
  }
  const auto it = s.streams.find(stream);
  if (it == s.streams.end()) return hipErrorInvalidHandle;
  Stream& st = it->second;
  if (!st.queue) st.queue = std::make_shared<Queue>(hipSuccess, hipErrorLaunchFailure);
  o->device = st.device;
  o->queue = st.queue;
  if (!(st.flags & hipStreamNonBlocking)) {
    const std::shared_ptr<Queue> legacy = null_queue(s, st.device);
    if (!legacy->idle()) o->after.push_back({legacy, legacy->tail()});
  }
  return hipSuccess;
}

// Runs `work` in `stream`'s order: queued, and returned from at once -- or,
// for a synchronous call (wait) and under VGPU_SYNC_LAUNCHES, waited for, and
// its own result returned. Called without s.mutex held.
hipError_t in_order(hipStream_t stream, Queue::Work work, bool wait = false) {
  State& s = state();
  Order o;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    if (const hipError_t e = order_for(s, stream, &o); e != hipSuccess) return e;
  }
  if (!wait && !synchronous_launches()) {
    o.queue->submit(std::move(work), std::move(o.after));
    return hipSuccess;
  }
  auto result = std::make_shared<hipError_t>(hipSuccess);
  const uint64_t seq = o.queue->submit(
      [result, work = std::move(work)] {
        *result = work();
        return hipSuccess;   // told to the caller, not kept for a later synchronization
      },
      std::move(o.after));
  o.queue->wait(seq);
  // A blocking call also returns a failure the stream's earlier asynchronous
  // work left (HIP's "may also return error codes from previous,
  // asynchronous launches"): a copy back after a kernel that failed says so,
  // rather than handing over whatever the buffer held.
  if (*result == hipSuccess)
    if (const hipError_t e = o.queue->take_error(); e != hipSuccess) return e;
  return *result;
}

// Every queue of a device -- its null stream's and each of its streams' --
// or only those the legacy null stream orders against. The caller holds s.mutex.
std::vector<std::shared_ptr<Queue>> queues_of(State& s, int device, bool blocking_only = false) {
  std::vector<std::shared_ptr<Queue>> out;
  if (const auto it = s.null_queues.find(device); it != s.null_queues.end() && it->second) out.push_back(it->second);
  for (auto& [handle, st] : s.streams)
    if (st.device == device && st.queue && !(blocking_only && (st.flags & hipStreamNonBlocking)))
      out.push_back(st.queue);
  return out;
}

// Waits for the queues and returns the first failure any of them had since
// it was last asked. Called without s.mutex held.
hipError_t drain(const std::vector<std::shared_ptr<Queue>>& queues) {
  hipError_t first = hipSuccess;
  for (const auto& q : queues) {
    q->drain();
    if (const hipError_t e = q->take_error(); e != hipSuccess && first == hipSuccess) first = e;
  }
  return first;
}
PerThreadStreams::~PerThreadStreams() {
  State& s = state();
  std::vector<std::shared_ptr<Queue>> qs;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    for (const auto& [device, h] : by_device) {
      const auto it = s.streams.find(h);
      if (it == s.streams.end()) continue;
      if (it->second.queue) qs.push_back(it->second.queue);
      forget_stream(s, h);
      s.streams.erase(it);
    }
  }
  for (const auto& q : qs) q->drain();
}

// At a synchronization, the stream-ordered pools give back what they keep
// beyond their release thresholds (see Pool).
void release_to_thresholds(State& s);

hipError_t drain_device(int device, bool blocking_only = false) {
  State& s = state();
  std::vector<std::shared_ptr<Queue>> qs;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    qs = queues_of(s, device, blocking_only);
  }
  return drain(qs);
}

// The kernel's arguments, as HIP passes them: either one packed buffer
// (`extra`), or one pointer per argument (`kernelParams`), which are packed
// here at the offsets the code object's metadata gives.
hipError_t build_kernargs(const Kernel& k, void** params, void** extra, std::vector<uint8_t>* out) {
  out->assign(k.kernarg_size, 0);
  if (extra) {
    const void* buffer = nullptr;
    size_t size = 0;
    for (size_t i = 0; extra[i] != HIP_LAUNCH_PARAM_END; ++i) {
      if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_POINTER) buffer = extra[++i];
      else if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_SIZE) size = *static_cast<size_t*>(extra[++i]);
      else return fail(hipErrorInvalidValue, "extra[] holds something that is not a launch parameter");
    }
    if (!buffer) return fail(hipErrorInvalidValue, "extra[] has no argument buffer");
    if (size > out->size()) out->resize(size);
    std::memcpy(out->data(), buffer, size ? size : out->size());
    return hipSuccess;
  }
  if (!params) {
    // A kernel that takes nothing of the program's needs neither: what the
    // compiler adds (hidden arguments) the runtime fills in.
    const bool takes = std::any_of(k.args.begin(), k.args.end(), [](const vgpu::amd::KernelArg& a) { return !a.hidden(); });
    return takes ? fail(hipErrorInvalidValue, "the kernel takes arguments, and the launch passed none") : hipSuccess;
  }
  for (size_t i = 0; i < k.args.size(); ++i) {
    const vgpu::amd::KernelArg& a = k.args[i];
    if (a.hidden()) continue;   // what the compiler adds is not the program's to pass
    if (!params[i]) return fail(hipErrorInvalidValue, "argument " + std::to_string(i) + " is a null pointer");
    if (a.offset + a.size > out->size())
      return fail(hipErrorInvalidValue, "argument " + std::to_string(i) + " is past the kernarg segment");
    std::memcpy(out->data() + a.offset, params[i], a.size);
  }
  return hipSuccess;
}

// A launch made ready while s.mutex is held -- its device, its code, its
// arguments, the hostcall buffer device printf writes through, the peers it
// may reach -- to run on its stream's thread without it (run_launch).
struct LaunchJob {
  int ordinal = 0;
  vgpu::runtime::Device* device = nullptr;
  const Module* module = nullptr;
  const Kernel* kernel = nullptr;
  vgpu::amd::abi::Dim3 grid{1, 1, 1}, block{1, 1, 1};
  uint32_t shared = 0;
  std::vector<uint8_t> args;
  hipStream_t stream = nullptr;
  bool cooperative = false;
  vgpu::amd::Hostcall* hostcall = nullptr;
  std::vector<vgpu::MemoryManager*> peers;
  std::string profile_id;
  // Arguments the caller has already placed on the device (an HSA dispatch
  // packet's kernarg_address), used where they are rather than copied.
  uint64_t kernarg_at = 0;
  // A grid in work-items that is not whole work-groups, and whether the
  // kernel's own work-group limit holds (vgpu/amd_exec.hpp): an HSA packet's.
  uint32_t grid_items[3] = {0, 0, 0};
  bool kernel_limits = true;
};

const Stream* find_stream(State& s, hipStream_t stream, Stream* null_stream);

// What ROCm's HIP refuses before a launch it is asked for runs: a grid or
// block with an empty or oversized dimension, or a block of more work-items
// than the device or the kernel (its launch bounds) takes, is the wrong
// configuration; more LDS than a block may have, or a stream that is not
// there, is the wrong value. A packet an HSA program puts on a queue goes to
// the hardware as it is, and none of this is asked of it. The caller holds
// s.mutex.
hipError_t check_launch(State& s, int ordinal, const Kernel& kernel, vgpu::amd::abi::Dim3 grid,
                        vgpu::amd::abi::Dim3 block, size_t shared, hipStream_t stream) {
  if (!block.x || !block.y || !block.z || !grid.x || !grid.y || !grid.z) return hipErrorInvalidConfiguration;
  const vgpu::Limits& lim = s.rt->device(ordinal).profile().limits;
  const uint32_t bdim[3] = {block.x, block.y, block.z}, gdim[3] = {grid.x, grid.y, grid.z};
  for (int i = 0; i < 3; ++i) {
    if (lim.max_block_dim[i] && bdim[i] > lim.max_block_dim[i]) return hipErrorInvalidConfiguration;
    if (lim.max_grid_dim[i] && gdim[i] > lim.max_grid_dim[i]) return hipErrorInvalidConfiguration;
  }
  const uint64_t threads = uint64_t{block.x} * block.y * block.z;
  if (lim.max_threads_per_block && threads > lim.max_threads_per_block) return hipErrorInvalidConfiguration;
  if (kernel.max_flat_workgroup_size && threads > kernel.max_flat_workgroup_size)
    return fail(hipErrorInvalidConfiguration, "the block has more work-items than the kernel's launch bounds allow");
  if (lim.shared_mem_per_block && uint64_t{shared} + kernel.group_segment > lim.shared_mem_per_block)
    return fail(hipErrorInvalidValue, "the launch asks for more LDS than a block may have");
  if (stream) {
    Stream null_stream;
    if (!find_stream(s, stream, &null_stream)) return hipErrorInvalidValue;
  }
  return hipSuccess;
}

hipError_t prepare_launch(State& s, int ordinal, const Module& module, const Kernel& kernel,
                          vgpu::amd::abi::Dim3 grid, vgpu::amd::abi::Dim3 block, uint32_t shared,
                          std::vector<uint8_t> args, hipStream_t stream, bool cooperative, LaunchJob* job) {
  if (!block.x || !block.y || !block.z || !grid.x || !grid.y || !grid.z) return hipErrorInvalidConfiguration;
  vgpu::runtime::Device& d = s.rt->device(ordinal);
  auto& hostcall = s.hostcalls[ordinal];
  if (!hostcall) {
    try {
      // What a kernel prints goes where the program's own printf goes, a
      // whole printf at a time.
      hostcall = std::make_unique<vgpu::amd::Hostcall>(d.memory(), [](int stream, const std::string& text) {
        static std::mutex mu;
        std::lock_guard<std::mutex> lock(mu);
        std::FILE* f = stream == 1 ? stderr : stdout;
        std::fwrite(text.data(), 1, text.size(), f);
        std::fflush(f);
      });
    } catch (const std::exception& e) {
      return fail(hipErrorOutOfMemory, std::string("no room for the hostcall buffer: ") + e.what());
    }
  }
  job->ordinal = ordinal;
  job->device = &d;
  job->module = &module;
  job->kernel = &kernel;
  job->grid = grid;
  job->block = block;
  job->shared = shared;
  job->args = std::move(args);
  job->stream = stream;
  job->cooperative = cooperative;
  job->hostcall = hostcall.get();
  const auto reach = [&](int to) {
    if (job->peers.size() <= static_cast<size_t>(to)) job->peers.resize(static_cast<size_t>(to) + 1);
    job->peers[static_cast<size_t>(to)] = &s.rt->device(to).memory();
  };
  for (const auto& [from, to] : s.peers)
    if (from == ordinal) reach(to);
  for (const int to : s.ipc_mapped)
    if (to != ordinal) reach(to);
  job->profile_id = s.profile_id;
  return hipSuccess;
}

// Runs one kernel on one device: the arguments go into device memory as its
// kernarg segment, and the dispatch runs to completion before this returns.
// On a stream's thread, without s.mutex.
hipError_t run_launch(const LaunchJob& job) {
  vgpu::runtime::Device& d = *job.device;
  const Module& module = *job.module;
  const Kernel& kernel = *job.kernel;
  const CodeObject& object = module.object;
  const auto grid = job.grid, block = job.block;
  const uint32_t shared = job.shared;
  const std::vector<uint8_t>& args = job.args;
  const bool cooperative = job.cooperative;
  const int ordinal = job.ordinal;
  const hipStream_t stream = job.stream;
  // VGPU_TRACE_LAUNCHES=1 says what each launch runs, one line to stderr:
  // what a program that calls a library cannot otherwise see.
  static const bool trace = [] {
    const char* t = std::getenv("VGPU_TRACE_LAUNCHES");
    return t && t[0] == '1';
  }();
  if (trace)
    std::fprintf(stderr, "VirtualGPU HIP: launch %s on device %d, grid %ux%ux%u of %ux%ux%u, %u bytes of LDS\n",
                 kernel.name.c_str(), ordinal, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared);
  vgpu::MemoryManager& mem = d.memory();
  uint64_t kernarg = 0;
  const vgpu::amd::hipprof::Hooks* prof = profiler();
  vgpu::amd::hipprof::Launch launch;
  void* token = nullptr;
  uint64_t start = 0;
  uint64_t grid_sync = 0;
  try {
    // The segment with room past its end, zeroed: ROCm hands kernels
    // kernarg memory padded well beyond what they declare, and the compiler
    // counts on it, widening a scalar load of the last arguments past the
    // segment's size (rocBLAS's rotmg reads 32 bytes at 0x60 of 124).
    if (job.kernarg_at) {
      kernarg = job.kernarg_at;
    } else {
      const size_t padded = (args.size() + 63) / 64 * 64 + 64;
      kernarg = mem.alloc(padded);
      std::vector<uint8_t> segment(padded, 0);
      std::copy(args.begin(), args.end(), segment.begin());
      mem.write(kernarg, segment.data(), segment.size());
    }
    if (cooperative && !job.kernarg_at) {
      // What the device library's grid barrier counts on (ockl's mg_info): a
      // grid of one, its work-groups, its work-items, and a counter for a
      // multi-grid barrier over that one grid, which passes straight through.
      const uint64_t groups = uint64_t{grid.x} * grid.y * grid.z;
      const uint64_t items = groups * block.x * block.y * block.z;
      grid_sync = mem.alloc(64);
      std::vector<uint8_t> info(64, 0);
      const uint64_t mgs = grid_sync + 48;
      std::memcpy(&info[0], &mgs, 8);         // mgs
      const uint32_t one = 1;
      std::memcpy(&info[12], &one, 4);        // num_grids; grid_id is 0
      std::memcpy(&info[24], &items, 8);      // all_sum; prev_sum is 0
      const uint32_t nwg = static_cast<uint32_t>(groups);
      std::memcpy(&info[40], &nwg, 4);        // num_wg; the barrier's count at 32 starts at 0
      mem.write(grid_sync, info.data(), info.size());
    }
    vgpu::amd::Dispatch dispatch;
    dispatch.object = &object;
    dispatch.kernel = &kernel;
    dispatch.kernarg = kernarg;
    dispatch.groups[0] = grid.x;
    dispatch.groups[1] = grid.y;
    dispatch.groups[2] = grid.z;
    dispatch.group_size[0] = block.x;
    dispatch.group_size[1] = block.y;
    dispatch.group_size[2] = block.z;
    for (int i = 0; i < 3; ++i) dispatch.grid_items[i] = job.grid_items[i];
    dispatch.kernel_limits = job.kernel_limits;
    dispatch.fill_hidden = !job.kernarg_at;   // a packet's segment is the caller's, hidden arguments and all
    dispatch.wave_size = static_cast<uint32_t>(d.profile().warp_size);
    dispatch.dynamic_lds = shared;   // what the launch adds to the kernel's own LDS
    dispatch.hostcall = job.hostcall;
    dispatch.code_base = module.code_base;
    dispatch.decoded = module.decoded.get();
    dispatch.cooperative = cooperative;
    dispatch.grid_sync = grid_sync;
    dispatch.peers = job.peers;
    if (prof) {
      launch.device = ordinal;
      launch.kernel_id = kernel_id(module, kernel);
      launch.dispatch_id = g_next_dispatch.fetch_add(1);
      launch.stream = reinterpret_cast<uint64_t>(stream);
      for (int i = 0; i < 3; ++i) {
        launch.groups[i] = dispatch.groups[i];
        launch.group_size[i] = dispatch.group_size[i];
      }
      launch.group_segment = kernel.group_segment + shared;
      launch.private_segment = kernel.private_segment;
      if (prof->launching) token = prof->launching(launch);
      start = vgpu::amd::hipprof::now_ns();
    }
    const vgpu::amd::DispatchStats stats = vgpu::amd::execute(dispatch, mem);
    if (prof && prof->launched) {
      prof->launched(launch, &stats, start, vgpu::amd::hipprof::now_ns(), token);
      prof = nullptr;   // told
    }
    if (!job.kernarg_at) mem.free(kernarg);
    if (grid_sync) mem.free(grid_sync);
    // What the device spent, as telemetry reports a kernel: the instructions
    // a wave retires, at the profile's clock.
    const uint32_t mhz = d.profile().telemetry.sm_clock_max_mhz;
    const double clock = mhz ? mhz * 1e6 : 1e9;
    d.note_busy(static_cast<double>(stats.instructions) / clock);
  } catch (const std::exception& e) {
    // A launch that failed counted nothing, and the profiler is told so.
    if (prof && prof->launched && start) prof->launched(launch, nullptr, start, vgpu::amd::hipprof::now_ns(), token);
    for (uint64_t a : {job.kernarg_at ? 0 : kernarg, grid_sync})
      if (a) {
        try {
          mem.free(a);
        } catch (const std::exception&) {
        }
      }
    return fail(hipErrorLaunchFailure, job.profile_id + ": " + e.what());
  }
  return hipSuccess;
}

// A prepared launch, run in its stream's order.
hipError_t launch_in_order(LaunchJob job) {
  const hipStream_t stream = job.stream;
  return in_order(stream, [job = std::move(job)] { return run_launch(job); });
}

// Puts a loaded module on a device: a linked one's whole image, from which its
// code runs and in which its variables sit; an unlinked one's variables, with
// its code told where they went.
void place(Module& m, vgpu::MemoryManager& mem) {
  m.decoded = std::make_unique<vgpu::amd::DecodeCache>(m.object.text.size());
  if (m.object.linked) {
    m.code_size = m.object.image.size();
    m.code_base = mem.alloc(m.object.image.empty() ? 1 : m.object.image.size());
    vgpu::amd::relocate_image(m.object, m.code_base);   // its global offset table, now it has an address
    if (!m.object.image.empty()) mem.write(m.code_base, m.object.image.data(), m.object.image.size());
    m.globals = m.code_base;
    // The device has the image now, and nothing reads the host's copy again:
    // for a library's hundreds of megabytes of kernels, that copy was a third
    // of what loading it cost.
    std::vector<uint8_t>().swap(m.object.image);
    return;
  }
  if (!m.object.data.empty()) {
    m.globals = mem.alloc(m.object.data.size());
    mem.write(m.globals, m.object.data.data(), m.object.data.size());
  }
  vgpu::amd::place_globals(m.object, m.globals);
}

// The module a registered binary becomes on one device: the code object for
// that device's gfx target, loaded, with its variables placed in the device's
// own memory. A program built for other targets only is told so by name.
hipError_t module_on(State& s, FatBinary& fb, int ordinal, Module** out) {
  if (auto it = fb.on_device.find(ordinal); it != fb.on_device.end()) {
    *out = it->second.get();
    return hipSuccess;
  }
  vgpu::runtime::Device& d = s.rt->device(ordinal);
  const std::string& gfx = d.profile().gcn_arch;
  if (!fb.image) return fail(hipErrorInvalidImage, "the program's device code is not a clang offload bundle");
  auto& bundle = s.bundles[{fb.image, d.profile().gcn_arch_full}];
  try {
    if (!bundle) bundle = vgpu::amd::read_bundle(fb.image, d.profile().gcn_arch_full);
  } catch (const std::exception& e) {
    return fail(hipErrorInvalidImage, e.what());
  }
  if (!bundle) return fail(hipErrorInvalidImage, "the program's device code is not a clang offload bundle");
  const std::string_view* code = vgpu::amd::code_for(*bundle, d.profile().gcn_arch_full);
  if (!code)
    return fail(hipErrorNoBinaryForGpu, "the program carries device code for " + vgpu::amd::target_list(*bundle) +
                                            ", and this device is " + d.profile().gcn_arch_full +
                                            ". Build it with --offload-arch=" + gfx + ".");
  try {
    auto m = std::make_unique<Module>();
    m->object = vgpu::amd::load_code_object(std::string(*code), "the program's " + gfx + " code");
    m->device = ordinal;
    place(*m, d.memory());
    // Its __managed__ variables' pointers, to the memory each already has.
    for (const State::ManagedVar& v : s.managed_vars)
      if (v.binary == &fb)
        if (const vgpu::amd::GlobalVar* g = vgpu::amd::find_global(m->object, v.name)) {
          const uint64_t at = reinterpret_cast<uint64_t>(*v.pointer);
          d.memory().write(m->globals + g->offset, &at, sizeof at);
        }
    report_loaded(*m, ordinal, code->data(), code->size());
    *out = m.get();
    fb.on_device.emplace(ordinal, std::move(m));
  } catch (const std::exception& e) {
    return fail(hipErrorInvalidImage, e.what());
  }
  return hipSuccess;
}

// Each thread's pending chevron launch: hipcc pushes the configuration, then
// calls the kernel's host-side function, which pops it and launches.
struct CallConfiguration {
  vgpu::amd::abi::Dim3 grid, block;
  size_t shared;
  hipStream_t stream;
};
// Kept through a pointer that outlives the thread's thread_local
// destructors, which exit() runs before the static destructors that may
// still launch (see the profiler's frames(), rocprofiler_sdk.cpp).
std::vector<CallConfiguration>& call_configurations() {
  thread_local std::vector<CallConfiguration>* v = nullptr;
  thread_local struct Reaper {
    std::vector<CallConfiguration>** p;
    ~Reaper() {
      delete *p;
      *p = nullptr;
    }
  } reaper{&v};
  (void)reaper;
  if (!v) v = new std::vector<CallConfiguration>;
  return *v;
}

// Each device's primary context, which a hipCtx_t names: the handle stands
// for the device (see "Devices and contexts").
char g_contexts[64];

// Host memory kernels reach: see hipHostMalloc.
std::mutex g_host_mutex;
std::map<uint64_t, size_t> g_host_allocations;   // hipHostMalloc
// hipHostRegister: each registration as made, by address, which may overlap
// (ROCm's HIP refuses none), and the ranges they cover together -- what the
// devices map, and what lookups by address go by.
std::map<uint64_t, size_t> g_host_registrations;
std::map<uint64_t, size_t> g_host_registered;
std::map<uint64_t, size_t> g_managed;            // hipMallocManaged
// The flags each pinned allocation was made with, which hipHostGetFlags
// gives back as they were given.
std::map<uint64_t, unsigned> g_host_flags;
int owner_of(State& s, uint64_t va);

// The advice managed memory was given, page by page (see hipMemAdvise).
struct PageAdvice {
  bool read_mostly = false;
  int preferred = -2;          // a device, hipCpuDeviceId (-1), or hipInvalidDeviceId (-2): none
  int last_prefetch = -2;
  bool coarse = false;         // hipMemAdviseSetCoarseGrain
  std::set<int> accessed_by;   // devices, and hipCpuDeviceId
};
std::map<uint64_t, std::vector<PageAdvice>> g_advice;   // by managed allocation, under g_host_mutex

void map_host_everywhere(State& s, void* p, size_t n) {
  for (int d = 0; d < s.rt->device_count(); ++d)
    s.rt->device(d).memory().map_host(reinterpret_cast<uint64_t>(p), p, std::max<size_t>(n, 1));
}
void unmap_host_everywhere(State& s, void* p) {
  for (int d = 0; d < s.rt->device_count(); ++d) s.rt->device(d).memory().unmap_host(reinterpret_cast<uint64_t>(p));
}
// Brings the devices' mappings of registered memory, and g_host_registered,
// to the union of the registrations: ranges no longer covered are unmapped
// and new ones mapped. Under g_host_mutex.
void remap_registered(State& s) {
  std::map<uint64_t, size_t> merged;
  for (const auto& [lo, n] : g_host_registrations) {
    const uint64_t hi = lo + n;
    if (!merged.empty()) {
      auto& [last_lo, last_n] = *std::prev(merged.end());
      if (lo <= last_lo + last_n) {
        last_n = std::max<uint64_t>(last_lo + last_n, hi) - last_lo;
        continue;
      }
    }
    merged[lo] = n;
  }
  for (const auto& [lo, n] : g_host_registered)
    if (const auto it = merged.find(lo); it == merged.end() || it->second != n)
      unmap_host_everywhere(s, reinterpret_cast<void*>(lo));
  for (const auto& [lo, n] : merged)
    if (const auto it = g_host_registered.find(lo); it == g_host_registered.end() || it->second != n)
      map_host_everywhere(s, reinterpret_cast<void*>(lo), n);
  g_host_registered = std::move(merged);
}
// The range of m holding va, or m.end().
std::map<uint64_t, size_t>::const_iterator find_range(const std::map<uint64_t, size_t>& m, uint64_t va) {
  auto it = m.upper_bound(va);
  if (it == m.begin()) return m.end();
  --it;
  return va < it->first + std::max<size_t>(it->second, 1) ? it : m.end();
}
// Host memory of `bytes`, page-aligned, mapped for every device and kept in m
// with the flags it was made with. None at all (0 bytes) is no allocation:
// the pointer is null, as ROCm's HIP gives it.
hipError_t host_alloc(void** ptr, size_t size, std::map<uint64_t, size_t>& m, unsigned flags = 0) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (const hipError_t e = unsafe_during_capture(s); e != hipSuccess) return record(s, e);
  if (!ptr) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (!size && &m != &g_managed) {
    *ptr = nullptr;
    return record(s, hipSuccess);
  }
  const size_t n = size ? size : 1;
  if (n > (size_t{1} << 47)) return record(s, hipErrorOutOfMemory);   // more than any host has
  void* p = std::aligned_alloc(4096, (n + 4095) / 4096 * 4096);
  if (!p) return record(s, hipErrorOutOfMemory);
  // Devices reach it in whole pages, as a card maps it: a kernel's 8-byte
  // atomic on a 4-byte allocation stays inside the page.
  map_host_everywhere(s, p, (n + 4095) / 4096 * 4096);
  std::lock_guard<std::mutex> host_lock(g_host_mutex);
  m[reinterpret_cast<uint64_t>(p)] = n;
  g_host_flags[reinterpret_cast<uint64_t>(p)] = flags;
  *ptr = p;
  return record(s, hipSuccess);
}
// Gives back what host_alloc gave, if m holds it.
bool host_free(State& s, void* ptr, std::map<uint64_t, size_t>& m) {
  std::lock_guard<std::mutex> host_lock(g_host_mutex);
  const auto it = m.find(reinterpret_cast<uint64_t>(ptr));
  if (it == m.end()) return false;
  if (s.rt) unmap_host_everywhere(s, ptr);
  m.erase(it);
  g_host_flags.erase(reinterpret_cast<uint64_t>(ptr));
  g_advice.erase(reinterpret_cast<uint64_t>(ptr));
  std::free(ptr);
  return true;
}


// ---- Copies and fills of every shape ------------------------------------------
//
// A region is `depth` slices of `rows` rows of `width` bytes, the rows `pitch`
// bytes apart and the slices `slice` bytes apart: one row for a plain copy, a
// plane for a 2D one, a box for a 3D one. Every copy and fill HIP has is one
// region (a copy has one on each side, of the same shape), queued as one piece
// of its stream's work.
struct Region {
  uint64_t width = 0, rows = 1, depth = 1, pitch = 0, slice = 0;
  bool empty() const { return !width || !rows || !depth; }
  // How far the region reaches from its start, or false where that does not
  // fit in 64 bits (and so reaches past any allocation).
  bool reach(uint64_t* bytes) const {
    uint64_t a = 0, b = 0;
    if (__builtin_mul_overflow(depth - 1, slice, &a) || __builtin_mul_overflow(rows - 1, pitch, &b) ||
        __builtin_add_overflow(a, b, &a) || __builtin_add_overflow(a, width, &a))
      return false;
    *bytes = a;
    return true;
  }
  uint64_t bytes() const { return width * rows * depth; }
};

// Where an address is: in a device's memory (`mem` set, `device` true), host
// memory the runtime keeps mapped for every device (pinned, registered or
// managed: `mem` set, `device` false), or the program's own pageable memory
// (`mem` null). Device memory has to be inside what was allocated, or
// mapped, for all of the region -- checked now, so the call that asked is
// the one refused. The caller holds s.mutex.
struct Side {
  vgpu::MemoryManager* mem = nullptr;
  bool device = false;
  int ordinal = -1;   // the device whose memory it is, where it is a device's
};
bool vmm_range(uint64_t va, uint64_t* base, uint64_t* size);
hipError_t locate(State& s, const void* p, const Region& r, Side* side) {
  const uint64_t va = reinterpret_cast<uint64_t>(p);
  for (int i = 0; i < s.rt->device_count(); ++i) {
    vgpu::MemoryManager& m = s.rt->device(i).memory();
    if (!m.owns(va)) continue;
    side->mem = &m;
    side->ordinal = i;
    uint64_t base = 0, size = 0, reach = 0;
    // Virtual memory, mapped: device memory, however it is kept.
    if (uint64_t vbase = 0, vsize = 0; vmm_range(va, &vbase, &vsize)) {
      side->device = true;
      if (!r.reach(&reach) || reach > vsize - (va - vbase))
        return fail(hipErrorInvalidValue, "the region runs past the end of the mapping it starts in");
      return hipSuccess;
    }
    if (m.is_host_mapped(va)) {   // pinned, registered or managed host memory: inside what was allocated
      std::lock_guard<std::mutex> host_lock(g_host_mutex);
      for (const auto* hm : {&g_host_allocations, &g_host_registered, &g_managed})
        if (const auto it = find_range(*hm, va); it != hm->end()) {
          if (!r.reach(&reach) || reach > it->second - (va - it->first))
            return fail(hipErrorInvalidValue, "the region runs past the end of the host memory it starts in");
          return hipSuccess;
        }
      return hipSuccess;
    }
    side->device = true;
    bool readable = false, writable = false;
    if (m.find_allocation(va, &base, &size)) {
      if (!r.reach(&reach) || reach > size - (va - base))
        return fail(hipErrorInvalidValue, "the region runs past the end of the allocation it starts in");
      return hipSuccess;
    }
    if (graph_memory_range(va, &base, &size)) {
      if (!r.reach(&reach) || reach > size - (va - base))
        return fail(hipErrorInvalidValue, "the region runs past the end of the graph allocation it starts in");
      return hipSuccess;
    }
    if (m.access_at(va, &readable, &writable)) return hipSuccess;   // virtual memory, mapped
    return fail(hipErrorInvalidValue, "that address is not in any allocation");
  }
  return hipSuccess;   // the program's own host memory
}

// Where a copy or fill goes other than onto its stream: a graph node's
// parameters are checked by the call that would make it (kRecord, which hands
// back the regions and queues nothing), and a graph's launch runs its copies
// and fills there on its own thread (kRun). Per thread; none for a call.
struct RegionSink {
  enum Mode { kRecord, kRun } mode = kRecord;
  bool got = false;
  void* dst = nullptr;
  const void* src = nullptr;
  Region dr, sr;
  hipMemcpyKind kind = hipMemcpyDefault;
};
thread_local RegionSink* t_sink = nullptr;

// Stream capture's copies and fills (hip_graph.inc; see "Stream capture"
// below). The caller holds s.mutex.
hipError_t capture_copy(State& s, hipStream_t resolved, void* dst, const Region& dr, const void* src, const Region& sr,
                        hipMemcpyKind kind);
hipError_t capture_fill(State& s, hipStream_t resolved, void* dst, const Region& r, const uint8_t* pattern,
                        uint32_t len);

// A copy between two regions of one shape, in `stream`'s order. Addresses
// are unified: where each side is comes from its address, whatever the kind
// says, as ROCm's HIP finds each pointer's allocation -- a "host to host"
// copy into device memory is a copy into device memory -- and a copy may run
// between two devices. The kind only has to be one HIP knows (1024, device
// to device without compute units, is one).
//
// When the caller waits is ROCm's HIP's: a synchronous copy until it is done
// -- except one from device memory to device memory, which is queued and
// returned from at once (still in the stream's order, so what reads the
// result after it sees it) unless either allocation asked for synchronous
// copies (hipPointerSetAttribute); an asynchronous one not at all, except
// into the program's own pageable memory, which it must have when the call
// returns, and from host memory to host memory, which is a plain memcpy. An
// asynchronous copy from pageable memory takes its bytes when it is called.
hipError_t copy_region(void* dst, const Region& dr, const void* src, const Region& sr, hipMemcpyKind kind,
                       hipStream_t stream, bool async) {
  State& s = state();
  Side to, from;
  vgpu::runtime::Device* d = nullptr;
  int ordinal = 0;
  const uint64_t dst_va = reinterpret_cast<uint64_t>(dst), src_va = reinterpret_cast<uint64_t>(src);
  RegionSink* const sink = t_sink;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    d = device(s);
    if (!d) return hipErrorInvalidDevice;
    // A call on a stream that is capturing becomes a node of its graph; one
    // on the legacy null stream while a blocking stream captures is refused.
    hipStream_t captured = nullptr;
    if (!sink) {
      const hipStream_t r = resolve(s, stream);
      if (!r) {
        if (const hipError_t e = legacy_conflict(s, s.current); e != hipSuccess) return e;
      } else if (stream_capturing(s, r)) {
        captured = r;
      }
    }
    if (kind != hipMemcpyDefault && kind != hipMemcpyHostToHost && kind != hipMemcpyHostToDevice &&
        kind != hipMemcpyDeviceToHost && kind != hipMemcpyDeviceToDevice && kind != 1024 /* NoCU */)
      return hipErrorInvalidMemcpyDirection;
    if (dr.empty()) return hipSuccess;
    if (!dst || !src) return hipErrorInvalidValue;
    ordinal = s.current;
    if (const hipError_t e = locate(s, dst, dr, &to); e != hipSuccess) return e;
    if (const hipError_t e = locate(s, src, sr, &from); e != hipSuccess) return e;
    if (sink && sink->mode == RegionSink::kRecord) {
      *sink = RegionSink{RegionSink::kRecord, true, dst, src, dr, sr, kind};
      return hipSuccess;
    }
    if (captured) return capture_copy(s, captured, dst, dr, src, sr, kind);
    if (!async && to.device && from.device) {
      uint64_t dbase = 0, sbase = 0, size = 0;
      const bool dsync = to.mem->find_allocation(dst_va, &dbase, &size) && s.sync_memops.count(dbase);
      const bool ssync = from.mem->find_allocation(src_va, &sbase, &size) && s.sync_memops.count(sbase);
      if (!dsync && !ssync) async = true;
    }
  }
  // Each side is reached through a device's memory where the runtime has it
  // (device memory, and host memory it maps), and directly otherwise.
  const bool to_device = to.mem != nullptr, from_device = from.mem != nullptr;
  // A synchronous copy of another device's memory also comes after what that
  // device's blocking streams were given: it is ordered on the device the
  // memory is on, as ROCm's HIP orders it, as well as the current one.
  // A graph's launch runs its copy right here, on its stream's thread.
  const bool run_here = sink != nullptr;
  if (!async && !run_here)
    for (const Side* side : {&to, &from})
      if (side->device && side->ordinal != ordinal) drain_device(side->ordinal, true);
  bool wait = !async || run_here;
  if (async && (!to.mem || (!to.device && !from.device))) wait = true;
  const uint64_t w = dr.width, rows = dr.rows, depth = dr.depth;
  // An asynchronous copy's unpinned source is taken now, packed.
  std::shared_ptr<std::vector<uint8_t>> staged;
  if (!wait && !from.mem) {
    staged = std::make_shared<std::vector<uint8_t>>(dr.bytes());
    for (uint64_t z = 0; z < depth; ++z)
      for (uint64_t y = 0; y < rows; ++y)
        std::memcpy(staged->data() + (z * rows + y) * w, static_cast<const uint8_t*>(src) + z * sr.slice + y * sr.pitch,
                    w);
  }
  vgpu::MemoryManager *tm = to.mem, *fm = from.mem;
  auto work = [=] {
    const uint64_t start = vgpu::amd::hipprof::now_ns();
    try {
      std::vector<uint8_t> buf(to_device && from_device ? w : 0);
      for (uint64_t z = 0; z < depth; ++z)
        for (uint64_t y = 0; y < rows; ++y) {
          const uint64_t dva = dst_va + z * dr.slice + y * dr.pitch, sva = src_va + z * sr.slice + y * sr.pitch;
          const void* source = staged ? static_cast<const void*>(staged->data() + (z * rows + y) * w)
                                      : reinterpret_cast<const void*>(sva);
          if (to_device && from_device) {
            fm->read(sva, buf.data(), w);
            tm->write(dva, buf.data(), w);
          } else if (to_device) {
            tm->write(dva, source, w);
          } else if (from_device) {
            fm->read(sva, reinterpret_cast<void*>(dva), w);
          } else {
            std::memmove(reinterpret_cast<void*>(dva), source, w);
          }
        }
    } catch (const std::exception& e) {
      return fail(hipErrorInvalidValue, e.what());
    }
    d->note_transfer(dr.bytes(), 0.0);
    if (const auto* p = profiler(); p && p->copied) {
      // Pinned, registered and managed memory is the host's, as a profiler
      // is told.
      using vgpu::amd::hipprof::Copy;
      const Copy k = to.device && from.device ? Copy::DeviceToDevice
                     : to.device             ? Copy::HostToDevice
                     : from.device           ? Copy::DeviceToHost
                                             : Copy::HostToHost;
      p->copied(k, ordinal, ordinal, dr.bytes(), dst_va, src_va, start, vgpu::amd::hipprof::now_ns());
    }
    return hipSuccess;
  };
  return run_here ? work() : in_order(stream, work, wait);
}

// A copy of `bytes`, or of `rows` rows of them `dpitch` and `spitch` apart.
// A 2D copy's rows have to fit its pitches.
hipError_t copy_in_order(void* dst, const void* src, size_t bytes, hipMemcpyKind kind, hipStream_t stream,
                         bool async, size_t rows = 1, size_t dpitch = 0, size_t spitch = 0) {
  const bool two_d = rows > 1 || dpitch || spitch;
  if (two_d && bytes && rows && (bytes > dpitch || bytes > spitch)) return hipErrorInvalidPitchValue;
  if (!two_d) dpitch = spitch = bytes;
  return copy_region(dst, Region{bytes, rows, 1, dpitch, 0}, src, Region{bytes, rows, 1, spitch, 0}, kind, stream,
                     async);
}

// A region filled with a pattern of 1, 2 or 4 bytes, repeated from each row's
// start, in `stream`'s order. The memory has to be the runtime's: device
// memory, or host memory it pinned, registered or manages. A synchronous
// fill of device memory is queued and returned from at once, as ROCm's HIP's
// is; one of host memory is waited for.
hipError_t fill_region(void* dst, const Region& r, const uint8_t* pattern, uint32_t pattern_len, hipStream_t stream,
                       bool async) {
  State& s = state();
  Side side;
  RegionSink* const sink = t_sink;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return e;
    if (!device(s)) return hipErrorInvalidDevice;
    hipStream_t captured = nullptr;
    if (!sink) {
      const hipStream_t rs = resolve(s, stream);
      if (!rs) {
        if (const hipError_t e = legacy_conflict(s, s.current); e != hipSuccess) return e;
      } else if (stream_capturing(s, rs)) {
        captured = rs;
      }
    }
    if (r.empty()) return hipSuccess;
    if (!dst) return hipErrorInvalidValue;
    if (const hipError_t e = locate(s, dst, r, &side); e != hipSuccess) return e;
    if (!side.mem) return fail(hipErrorInvalidValue, "that is host memory the runtime did not allocate or register");
    if (sink && sink->mode == RegionSink::kRecord) {
      sink->got = true;
      sink->dst = dst;
      sink->dr = r;
      return hipSuccess;
    }
    if (captured) return capture_fill(s, captured, dst, r, pattern, pattern_len);
  }
  const std::vector<uint8_t> p(pattern, pattern + pattern_len);
  vgpu::MemoryManager* mem = side.mem;
  const bool on_device = side.device;
  const uint64_t base = reinterpret_cast<uint64_t>(dst);
  const auto work = [=] {
        try {
          for (uint64_t z = 0; z < r.depth; ++z)
            for (uint64_t y = 0; y < r.rows; ++y) {
              const uint64_t at = base + z * r.slice + y * r.pitch;
              if (on_device) {
                mem->fill(at, p.data(), static_cast<uint32_t>(p.size()), r.width);
              } else {
                // Host memory the runtime maps is filled where it is.
                auto* row = reinterpret_cast<uint8_t*>(at);
                for (uint64_t i = 0; i < r.width; ++i) row[i] = p[i % p.size()];
              }
            }
        } catch (const std::exception& e) {
          return fail(hipErrorInvalidValue, e.what());
        }
        return hipSuccess;
      };
  // A graph's launch runs its fill right here, on its stream's thread.
  if (sink) return work();
  return in_order(stream, work, !async && !side.device);
}
// `bytes` bytes of a pattern.
hipError_t fill_in_order(void* dst, const uint8_t* pattern, uint32_t pattern_len, uint64_t bytes, hipStream_t stream,
                         bool async) {
  return fill_region(dst, Region{bytes, 1, 1, bytes, 0}, pattern, pattern_len, stream, async);
}
}  // namespace

extern "C" {

hipError_t hipInit(unsigned int flags) {
  const ApiCall api("hipInit");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (flags) return record(s, hipErrorInvalidValue);   // HIP defines none
  return record(s, ensure_runtime(s));
}

hipError_t hipGetDeviceCount(int* count) {
  const ApiCall api("hipGetDeviceCount");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!count) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) {
    if (e == hipErrorNoDevice) *count = 0;
    return record(s, e);
  }
  *count = s.rt->device_count();
  return record(s, hipSuccess);
}

hipError_t hipSetDevice(int d) {
  const ApiCall api("hipSetDevice");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (d < 0 || d >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  s.current = d;
  s.device_chosen = true;
  return record(s, hipSuccess);
}

hipError_t hipGetDevice(int* d) {
  const ApiCall api("hipGetDevice");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!d) return record(s, hipErrorInvalidValue);
  *d = s.current;
  return record(s, hipSuccess);
}


hipError_t hipDeviceSynchronize(void) {
  const ApiCall api("hipDeviceSynchronize");
  State& s = state();
  int ordinal = 0;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
    ordinal = s.current;
    // Nothing a capture has is there to wait for.
    if (any_capture_on(s, ordinal)) return record(s, hipErrorStreamCaptureUnsupported);
  }
  // Everything queued on the device, on every stream, and the first failure
  // any of it had: where an asynchronous error comes out. The pools then
  // keep no more than their release thresholds.
  const hipError_t e = drain_device(ordinal);
  std::lock_guard<std::mutex> lock(s.mutex);
  release_to_thresholds(s);
  return record(s, e);
}

hipError_t hipDeviceReset(void) {
  const ApiCall api("hipDeviceReset");
  State& s = state();
  if (s.rt) {
    int ordinal = 0;
    {
      std::lock_guard<std::mutex> lock(s.mutex);
      ordinal = s.current;
    }
    drain_device(ordinal);
  }
  std::lock_guard<std::mutex> lock(s.mutex);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  s.functions.clear();
  s.modules.clear();
  s.hostcalls.clear();   // their memory is about to go with everything else
  for (int i = 0; i < s.rt->device_count(); ++i) s.rt->device(i).reset();
  return record(s, hipSuccess);
}

hipError_t hipMalloc(void** ptr, size_t size) {
  const ApiCall api("hipMalloc");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (const hipError_t e = unsafe_during_capture(s); e != hipSuccess) return record(s, e);
  if (!ptr) return record(s, hipErrorInvalidValue);
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
  if (!size) {
    *ptr = nullptr;
    return record(s, hipSuccess);
  }
  const uint64_t start = vgpu::amd::hipprof::now_ns();
  try {
    *ptr = reinterpret_cast<void*>(d->memory().alloc(size));
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorOutOfMemory, e.what()));
  }
  if (const auto* p = profiler(); p && p->allocated)
    p->allocated(s.current, reinterpret_cast<uint64_t>(*ptr), size, false, start, vgpu::amd::hipprof::now_ns());
  return record(s, hipSuccess);
}

hipError_t free_now(void* ptr) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr) return hipSuccess;
  // Managed memory, and pinned host memory, which ROCm's HIP frees here too.
  if (host_free(s, ptr, g_managed) || host_free(s, ptr, g_host_allocations)) return hipSuccess;
  // A graph's allocation: its memory goes, and its address stays the graph's.
  if (hipError_t e = hipSuccess; free_graph_memory(s, ptr, &e)) return e;
  // Another process's pool memory, imported: this process's mapping goes.
  if (const auto im = g_pool_imports.find(ptr); im != g_pool_imports.end()) {
    try {
      s.rt->device(im->second).memory().abandon(reinterpret_cast<uint64_t>(ptr));
    } catch (const std::exception& e) {
      return fail(hipErrorInvalidValue, e.what());
    }
    g_pool_imports.erase(im);
    return hipSuccess;
  }
  vgpu::runtime::Device* d = device(s);
  if (!d) return (hipErrorInvalidDevice);
  if (const auto pa = s.pool_allocations.find(reinterpret_cast<uint64_t>(ptr)); pa != s.pool_allocations.end()) {
    pa->second.first->used -= pa->second.second;
    s.pool_allocations.erase(pa);
  }
  // Freed on the device whose memory it is, whichever is current, as HIP's
  // unified addresses allow; the current device answers for any other.
  int owner = s.current;
  for (int i = 0; i < s.rt->device_count(); ++i)
    if (s.rt->device(i).memory().owns(reinterpret_cast<uint64_t>(ptr))) owner = i;
  const uint64_t start = vgpu::amd::hipprof::now_ns();
  // What was never allocated, or was freed already, is the wrong value, as
  // ROCm's HIP has it.
  try {
    s.rt->device(owner).memory().free(reinterpret_cast<uint64_t>(ptr));
  } catch (const std::exception& e) {
    return (fail(hipErrorInvalidValue, e.what()));
  }
  if (const auto* p = profiler(); p && p->allocated)
    p->allocated(owner, reinterpret_cast<uint64_t>(ptr), 0, true, start, vgpu::amd::hipprof::now_ns());
  return hipSuccess;
}

// Freed once the device's work is done with it: HIP's hipFree waits for the
// device, as the work queued before it may still be using the memory.
hipError_t hipFree(void* ptr) {
  const ApiCall api("hipFree");
  State& s = state();
  if (!ptr) return record(s, hipSuccess);
  int ordinal = 0;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
    ordinal = s.current;
    for (int i = 0; i < s.rt->device_count(); ++i)
      if (s.rt->device(i).memory().owns(reinterpret_cast<uint64_t>(ptr))) ordinal = i;
  }
  drain_device(ordinal);
  return record(s, free_now(ptr));
}

hipError_t hipMemcpy(void* dst, const void* src, size_t bytes, hipMemcpyKind kind) {
  const ApiCall api("hipMemcpy");
  return record(state(), copy_in_order(dst, src, bytes, kind, nullptr, false));
}

hipError_t hipMemset(void* dst, int value, size_t bytes) {
  const ApiCall api("hipMemset");
  const uint8_t byte = static_cast<uint8_t>(value);
  return record(state(), fill_in_order(dst, &byte, 1, bytes, nullptr, false));
}

// The asynchronous forms: queued on the stream, and returned from at once
// (but see copy_in_order on host memory the runtime did not pin).
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t bytes, hipMemcpyKind kind, hipStream_t stream);
// Copies in a batch, in order on the stream, each as hipMemcpyAsync with the
// direction worked out from the pointers. The attributes are hints about
// the source's access order and where the copy may run; they change nothing
// here. The first copy refused is reported through failIdx.
hipError_t hipMemcpyBatchAsync(void** dsts, void** srcs, size_t* sizes, size_t count, void* attrs,
                               size_t* attrs_idxs, size_t num_attrs, size_t* fail_idx, hipStream_t stream) {
  (void)attrs;
  (void)attrs_idxs;
  (void)num_attrs;
  if (fail_idx) *fail_idx = SIZE_MAX;
  if (!count || !dsts || !srcs || !sizes) return hipErrorInvalidValue;
  for (size_t i = 0; i < count; ++i)
    if (const hipError_t e = hipMemcpyAsync(dsts[i], srcs[i], sizes[i], hipMemcpyDefault, stream); e != hipSuccess) {
      if (fail_idx) *fail_idx = i;
      return e;
    }
  return hipSuccess;
}
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t bytes, hipMemcpyKind kind, hipStream_t stream) {
  const ApiCall api("hipMemcpyAsync");
  return record(state(), copy_in_order(dst, src, bytes, kind, stream, true));
}

hipError_t hipMemsetAsync(void* dst, int value, size_t bytes, hipStream_t stream) {
  const ApiCall api("hipMemsetAsync");
  const uint8_t byte = static_cast<uint8_t>(value);
  return record(state(), fill_in_order(dst, &byte, 1, bytes, stream, true));
}

hipError_t hipMemGetInfo(size_t* free, size_t* total) {
  const ApiCall api("hipMemGetInfo");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
  const uint64_t capacity = d->memory().capacity(), used = d->memory().used();
  if (total) *total = static_cast<size_t>(capacity);
  if (free) *free = static_cast<size_t>(capacity > used ? capacity - used : 0);
  return record(s, hipSuccess);
}

hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t ordinal) {
  const ApiCall api("hipDeviceTotalMem");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!bytes) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (ordinal < 0 || ordinal >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  *bytes = static_cast<size_t>(s.rt->device(ordinal).profile().vram_bytes);
  return record(s, hipSuccess);
}

hipError_t hipDeviceGetName(char* name, int len, hipDevice_t ordinal) {
  const ApiCall api("hipDeviceGetName");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!name || len <= 0) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (ordinal < 0 || ordinal >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  std::snprintf(name, static_cast<size_t>(len), "%s", s.rt->device(ordinal).profile().model.c_str());
  return record(s, hipSuccess);
}

hipError_t hipDeviceGet(hipDevice_t* device_out, int ordinal) {
  const ApiCall api("hipDeviceGet");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!device_out) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (ordinal < 0 || ordinal >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  *device_out = ordinal;   // a device is its ordinal here
  return record(s, hipSuccess);
}

hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
  const ApiCall api("hipModuleLoadData");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!module || !image) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  const uint8_t* bytes = static_cast<const uint8_t*>(image);
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
  // An offload bundle -- what hipcc --genco writes -- carries a code object
  // per target, and the device's is the one loaded.
  std::unique_ptr<vgpu::amd::Bundle> bundle;
  if (vgpu::amd::is_bundle(bytes)) {
    try {
      bundle = vgpu::amd::read_bundle(bytes, d->profile().gcn_arch_full);
    } catch (const std::exception& e) {
      return record(s, fail(hipErrorInvalidImage, e.what()));
    }
    const std::string_view* code = bundle ? vgpu::amd::code_for(*bundle, d->profile().gcn_arch_full) : nullptr;
    if (!code)
      return record(s, fail(hipErrorNoBinaryForGpu, "the bundle carries code for " +
                                                        (bundle ? vgpu::amd::target_list(*bundle) : "no GPU") +
                                                        ", and this device is " + d->profile().gcn_arch_full));
    bytes = reinterpret_cast<const uint8_t*>(code->data());
  }
  // A code object's length is in its own header; the ELF says where its
  // sections end, and the last of them is where the image stops.
  if (std::memcmp(bytes, "\x7F" "ELF", 4) != 0)
    return record(s, fail(hipErrorInvalidImage, "the image is not an ELF code object or an offload bundle"));
  uint64_t shoff = 0;
  std::memcpy(&shoff, bytes + 0x28, 8);
  uint16_t shentsize = 0, shnum = 0;
  std::memcpy(&shentsize, bytes + 0x3A, 2);
  std::memcpy(&shnum, bytes + 0x3C, 2);
  const uint64_t size = shoff + uint64_t{shentsize} * shnum;
  try {
    auto m = std::make_unique<Module>();
    m->object = vgpu::amd::load_code_object(std::string(reinterpret_cast<const char*>(bytes), size), "the image");
    // VGPU_DUMP_CODE_OBJECTS=<dir> keeps a copy of every code object a
    // program loads, named for its first kernel: what a library builds at
    // run time (MIOpen's convolutions, rocFFT's plans) is otherwise nowhere
    // to disassemble.
    if (const char* dir = std::getenv("VGPU_DUMP_CODE_OBJECTS"); dir && *dir) {
      static int count = 0;
      std::string name = m->object.kernels.empty() ? "module" : m->object.kernels.front().name;
      if (name.size() > 120) name.resize(120);
      std::ofstream(std::string(dir) + "/" + std::to_string(count++) + "-" + name + ".co", std::ios::binary)
          .write(reinterpret_cast<const char*>(bytes), static_cast<std::streamsize>(size));
    }
    // The module goes on the device, and the code is told where its
    // variables are: until that is done, a kernel reaching one reads nothing.
    m->device = s.current;
    place(*m, d->memory());
    report_loaded(*m, s.current, image, size);
    *module = reinterpret_cast<hipModule_t>(m.get());
    s.modules.push_back(std::move(m));
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidImage, e.what()));
  }
  return record(s, hipSuccess);
}

hipError_t hipModuleLoad(hipModule_t* module, const char* path) {
  const ApiCall api("hipModuleLoad");
  if (!module || !path || !*path) return record(state(), hipErrorInvalidValue);
  std::ifstream in(path, std::ios::binary);
  if (!in) return fail(hipErrorFileNotFound, std::string("no code object at ") + path);
  const std::string bytes((std::istreambuf_iterator<char>(in)), {});
  if (bytes.empty()) return fail(hipErrorInvalidImage, std::string(path) + " is empty");
  return hipModuleLoadData(module, bytes.data());
}

hipError_t hipModuleUnload(hipModule_t module) {
  const ApiCall api("hipModuleUnload");
  State& s = state();
  if (!module) return record(s, hipErrorInvalidResourceHandle);
  // Its kernels may still be queued or running: unloading waits for the device.
  if (s.rt) {
    int ordinal = 0;
    {
      std::lock_guard<std::mutex> lock(s.mutex);
      ordinal = s.current;
    }
    drain_device(ordinal);
  }
  std::lock_guard<std::mutex> lock(s.mutex);
  for (size_t i = 0; i < s.modules.size(); ++i)
    if (reinterpret_cast<hipModule_t>(s.modules[i].get()) == module) {
      for (size_t f = s.functions.size(); f-- > 0;)
        if (s.functions[f]->module == s.modules[i].get()) s.functions.erase(s.functions.begin() + f);
      if (s.modules[i]->globals) {
        if (vgpu::runtime::Device* d = device(s)) {
          try {
            d->memory().free(s.modules[i]->globals);
          } catch (const std::exception&) {
          }
        }
      }
      s.modules.erase(s.modules.begin() + i);
      return record(s, hipSuccess);
    }
  return record(s, hipErrorNotFound);
}

hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* name) {
  const ApiCall api("hipModuleGetFunction");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!function || !module || !name) return record(s, hipErrorInvalidValue);
  Module* m = reinterpret_cast<Module*>(module);
  const Kernel* k = vgpu::amd::find_kernel(m->object, name);
  if (!k) return record(s, fail(hipErrorNotFound, std::string("the code object has no kernel named ") + name));
  auto f = std::make_unique<Function>();
  f->module = m;
  f->kernel = k;
  *function = reinterpret_cast<hipFunction_t>(f.get());
  s.functions.push_back(std::move(f));
  return record(s, hipSuccess);
}

hipError_t hipModuleGetGlobal(void** dptr, size_t* bytes, hipModule_t module, const char* name) {
  const ApiCall api("hipModuleGetGlobal");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!module) return record(s, hipErrorInvalidResourceHandle);
  if (!name || !*name || (!dptr && !bytes)) return record(s, hipErrorInvalidValue);
  Module* m = reinterpret_cast<Module*>(module);
  const vgpu::amd::GlobalVar* g = vgpu::amd::find_global(m->object, name);
  if (!g) return record(s, fail(hipErrorNotFound, std::string("the module has no variable named ") + name));
  if (dptr) *dptr = reinterpret_cast<void*>(m->globals + g->offset);
  if (bytes) *bytes = static_cast<size_t>(g->size);
  return record(s, hipSuccess);
}

namespace {
// A module launch: `grid_items`, where it is set, is the grid in work-items
// when it is not a whole number of work-groups (hipExtModuleLaunchKernel).
hipError_t module_launch(hipFunction_t f, unsigned int gx, unsigned int gy, unsigned int gz, unsigned int bx,
                         unsigned int by, unsigned int bz, unsigned int shared, hipStream_t stream, void** params,
                         void** extra, const uint32_t* grid_items);
}  // namespace

hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gx, unsigned int gy, unsigned int gz,
                                 unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared,
                                 hipStream_t stream, void** params, void** extra) {
  const ApiCall api("hipModuleLaunchKernel");
  return module_launch(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra, nullptr);
}

namespace {
hipError_t module_launch(hipFunction_t f, unsigned int gx, unsigned int gy, unsigned int gz, unsigned int bx,
                         unsigned int by, unsigned int bz, unsigned int shared, hipStream_t stream, void** params,
                         void** extra, const uint32_t* grid_items) {
  State& s = state();
  std::unique_lock<std::mutex> lock(s.mutex);
  if (!f) return record(s, hipErrorInvalidValue);
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
  Function* fn = reinterpret_cast<Function*>(f);
  if (const hipError_t e = check_launch(s, s.current, *fn->kernel, {gx, gy, gz}, {bx, by, bz}, shared, stream);
      e != hipSuccess)
    return record(s, e);
  std::vector<uint8_t> args;
  if (const hipError_t e = build_kernargs(*fn->kernel, params, extra, &args); e != hipSuccess)
    return record(s, e);
  if (const hipStream_t r = resolve(s, stream); !r) {
    if (const hipError_t e = legacy_conflict(s, s.current); e != hipSuccess) return record(s, e);
  } else if (stream_capturing(s, r)) {
    return record(s, capture_kernel(s, r, f, *fn->kernel, {gx, gy, gz}, {bx, by, bz}, shared, std::move(args), extra,
                                    false));
  }

  LaunchJob job;
  if (const hipError_t e = prepare_launch(s, s.current, *fn->module, *fn->kernel, {gx, gy, gz}, {bx, by, bz}, shared,
                                          std::move(args), stream, false, &job);
      e != hipSuccess)
    return record(s, e);
  if (grid_items)
    for (int i = 0; i < 3; ++i) job.grid_items[i] = grid_items[i];
  lock.unlock();
  return record(s, launch_in_order(std::move(job)));
}
}  // namespace

}  // extern "C"
namespace {
// hip_errors.inc's names and texts, by value; nullptr for a value HIP does not
// define.
struct ErrorText {
  const char* name;
  const char* text;
};
const ErrorText* error_text(int e) {
  static const std::map<int, ErrorText> table = [] {
    std::map<int, ErrorText> t;
#define E(value, name, text) t.emplace(value, ErrorText{name, text});
#include "hip_errors.inc"
#undef E
    return t;
  }();
  // ROCm 7.1's own HIP knows neither texture code its header names
  // (hipErrorInvalidChannelDescriptor, hipErrorInvalidTexture): to it they
  // are unknown errors.
  if (e == 911 || e == 912) return nullptr;
  const auto it = table.find(e);
  return it == table.end() ? nullptr : &it->second;
}
}  // namespace
extern "C" {

const char* hipGetErrorName(hipError_t e) {
  const ApiCall api("hipGetErrorName");
  const ErrorText* t = error_text(e);
  return t ? t->name : "hipErrorUnknown";
}

const char* hipGetErrorString(hipError_t e) {
  const ApiCall api("hipGetErrorString");
  const ErrorText* t = error_text(e);
  return t ? t->text : "unknown error";
}

hipError_t hipGetLastError(void) {
  const ApiCall api("hipGetLastError");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  const hipError_t e = s.last;
  s.last = hipSuccess;
  return e;
}

hipError_t hipPeekAtLastError(void) {
  const ApiCall api("hipPeekAtLastError");
  return state().last;
}

}  // extern "C"

namespace {

// A stream on the current device, kept with what it was made with. Handles
// count up from 1 and are never reused; the null stream is not in the table.
hipError_t create_stream(hipStream_t* stream, unsigned flags, int priority, std::vector<uint32_t> cu_mask = {}) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!stream) return record(s, hipErrorInvalidValue);
  // Handles from 0x100: HIP reserves the small ones (hipStreamPerThread is 2).
  static intptr_t next = 0x100;
  *stream = reinterpret_cast<hipStream_t>(next++);
  s.streams[*stream] = Stream{s.current, flags, priority, std::move(cu_mask), nullptr};
  return record(s, hipSuccess);
}

// The stream's own record, or the null stream's: the current device, no
// flags, the normal priority. A handle that was never made, or was
// destroyed, is nullptr.
const Stream* find_stream(State& s, hipStream_t stream, Stream* null_stream) {
  stream = resolve(s, stream);
  if (!stream) {
    *null_stream = Stream{s.current, 0, 0, {}, nullptr};
    return null_stream;
  }
  const auto it = s.streams.find(stream);
  return it == s.streams.end() ? nullptr : &it->second;
}

}  // namespace

extern "C" {

int hipGetStreamDeviceId(hipStream_t stream) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  Stream null_stream;
  const Stream* st = find_stream(s, stream, &null_stream);
  return st ? st->device : -1;
}

// Streams: a handle, what it was made with, and a queue its work runs on,
// made with its first work (see "Stream order" above). The null stream is
// what a program gets by default.
hipError_t hipStreamCreate(hipStream_t* stream) {
  const ApiCall api("hipStreamCreate");
  return create_stream(stream, 0, 0);
}
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
  const ApiCall api("hipStreamCreateWithFlags");
  // Neither flag changes anything here: there is no other work for a stream
  // to be blocking on, and nothing to synchronise against.
  if (flags & ~static_cast<unsigned>(hipStreamNonBlocking)) return hipErrorInvalidValue;
  return create_stream(stream, flags, 0);
}
hipError_t hipStreamDestroy(hipStream_t stream) {
  const ApiCall api("hipStreamDestroy");
  State& s = state();
  std::shared_ptr<Queue> q;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    // The default streams, legacy and per-thread, are the runtime's, not the
    // program's to destroy: ROCm's HIP refuses them as it refuses a handle it
    // never made.
    const intptr_t v = reinterpret_cast<intptr_t>(stream);
    if (!stream || v == kStreamLegacy || v == kStreamPerThread) return record(s, hipErrorInvalidHandle);
    const auto it = s.streams.find(stream);
    if (it == s.streams.end()) return record(s, hipErrorInvalidHandle);
    q = it->second.queue;
    forget_stream(s, stream);
    s.streams.erase(it);
  }
  // Its work still runs to the end, as HIP lets a stream destroyed with work
  // in it finish: the queue lives until its last item has.
  if (q) q->drain();
  return record(s, hipSuccess);
}
// The null stream's synchronization waits for the blocking streams too: the
// legacy default stream orders against them, and a program that synchronizes
// it means everything it launched there.
hipError_t hipStreamSynchronize(hipStream_t stream) {
  const ApiCall api("hipStreamSynchronize");
  State& s = state();
  std::vector<std::shared_ptr<Queue>> qs;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
    stream = resolve(s, stream);
    // A capturing stream has nothing to wait for yet; waiting on another
    // is refused where its capture mode says.
    if (stream && stream_capturing(s, stream)) return record(s, hipErrorStreamCaptureUnsupported);
    if (const hipError_t e = unsafe_during_capture(s); e != hipSuccess) return record(s, e);
    if (!stream) {
      qs = queues_of(s, s.current, true);
    } else {
      const auto it = s.streams.find(stream);
      if (it == s.streams.end()) return record(s, hipErrorInvalidHandle);
      if (it->second.queue) qs.push_back(it->second.queue);
    }
  }
  const hipError_t e = drain(qs);
  std::lock_guard<std::mutex> lock(s.mutex);
  release_to_thresholds(s);
  return record(s, e);
}

// Events: a program records one behind the work on a stream and asks when
// that work finished, or waits for it, or makes another stream wait for it.
// Recording puts an item on the stream's queue that takes the time when it
// runs, so the time between two events is the time the simulator took for
// the work between them -- not what a card would have taken, which this does
// not claim to know.
namespace {

struct Stamp {
  std::atomic<bool> taken{false};
  std::chrono::steady_clock::time_point when{};
};
// An interprocess event's state, which every process that opened it shares
// (a small file, mapped): how many records have been made of it, anywhere,
// and how many of those have happened. Another process's event is waited on
// by these, having no queue of this process's to wait on.
struct IpcEventState {
  std::atomic<uint64_t>* counters = nullptr;   // [0] records made, [1] records done
  std::string path;
  bool owner = false;
  static constexpr size_t kBytes = 4096;
  ~IpcEventState() {
    if (counters) ::munmap(counters, kBytes);
    if (owner) ::unlink(path.c_str());
  }
  void done(uint64_t n) {
    uint64_t now = counters[1].load();
    while (now < n && !counters[1].compare_exchange_weak(now, n)) {
    }
  }
  bool finished(uint64_t upto) const { return counters[1].load() >= upto; }
  void wait(uint64_t upto) const {
    while (!finished(upto)) std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
};
std::shared_ptr<IpcEventState> map_ipc_event(const std::string& path, bool create) {
  const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC | (create ? O_CREAT | O_EXCL : 0), 0600);
  if (fd < 0) return nullptr;
  if (create && ::ftruncate(fd, IpcEventState::kBytes) != 0) {
    ::close(fd);
    ::unlink(path.c_str());
    return nullptr;
  }
  void* p = ::mmap(nullptr, IpcEventState::kBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) return nullptr;
  auto st = std::make_shared<IpcEventState>();
  st->counters = static_cast<std::atomic<uint64_t>*>(p);
  st->path = path;
  st->owner = create;
  return st;
}
struct Event {
  int device = 0;   // the device current when it was made: it records only there
  bool timing = true;
  bool recorded = false;
  std::shared_ptr<Queue> queue;   // what the last record was queued on
  uint64_t seq = 0;               // and which item of it
  std::shared_ptr<Stamp> stamp;
  // Where it was last recorded in a capture (hip_graph.inc): which capture,
  // which stream, after how many of its operations, and the graph nodes the
  // stream's next operation would have followed then.
  uint64_t capture = 0;
  hipStream_t capture_stream = nullptr;
  uint64_t capture_ops = 0;
  std::vector<void*> capture_deps;
  // An interprocess event's shared state; `remote` where another process
  // made it and this one opened it.
  std::shared_ptr<IpcEventState> ipc;
  bool remote = false;
  char ipc_id[40] = {};
};

// Lock order: s.mutex, then this.
std::mutex g_event_mutex;
std::map<hipEvent_t, std::unique_ptr<Event>> g_events;

Event* find_event(hipEvent_t e) {   // the caller holds g_event_mutex
  const auto it = g_events.find(e);
  return it == g_events.end() ? nullptr : it->second.get();
}

}  // namespace

// The flags ROCm's HIP takes: blocking synchronization, no timing, one
// another process may open (which must not time), and the fences AMD adds
// (none, device scope, system scope). None changes how an event works here.
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int flags) {
  const ApiCall api("hipEventCreateWithFlags");
  constexpr unsigned kInterprocess = 0x4, kDisableSystemFence = 0x20000000, kReleaseToDevice = 0x40000000,
                     kReleaseToSystem = 0x80000000;
  constexpr unsigned kKnown = hipEventBlockingSync | hipEventDisableTiming | kInterprocess | kDisableSystemFence |
                              kReleaseToDevice | kReleaseToSystem;
  State& s = state();
  if (!event || (flags & ~kKnown)) return record(s, hipErrorInvalidValue);
  if ((flags & kInterprocess) && !(flags & hipEventDisableTiming)) return record(s, hipErrorInvalidValue);
  if ((flags & kReleaseToDevice) && (flags & kReleaseToSystem)) return record(s, hipErrorInvalidValue);
  int device = 0;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    device = s.current;
  }
  std::lock_guard<std::mutex> lock(g_event_mutex);
  static intptr_t next = 1;
  const hipEvent_t handle = reinterpret_cast<hipEvent_t>(next++);
  auto e = std::make_unique<Event>();
  e->device = device;
  e->timing = (flags & hipEventDisableTiming) == 0;
  // An interprocess event's state is a file from the start, so records made
  // before its handle is handed out count too.
  if (flags & kInterprocess) {
    static std::atomic<uint32_t> counter{0};
    std::snprintf(e->ipc_id, sizeof e->ipc_id, "%x-e%x", static_cast<unsigned>(::getpid()), counter.fetch_add(1) + 1);
    e->ipc = map_ipc_event(vgpu::telemetry::default_path() + "/ipc-hip-" + e->ipc_id, true);
    if (!e->ipc) return record(s, fail(hipErrorOutOfMemory, "no file for the interprocess event's state"));
  }
  g_events.emplace(handle, std::move(e));
  *event = handle;
  return hipSuccess;
}

hipError_t hipEventCreate(hipEvent_t* event) {
  const ApiCall api("hipEventCreate");
  return hipEventCreateWithFlags(event, hipEventDefault);
}

hipError_t hipEventDestroy(hipEvent_t event) {
  const ApiCall api("hipEventDestroy");
  std::lock_guard<std::mutex> lock(g_event_mutex);
  return g_events.erase(event) ? hipSuccess : record(state(), hipErrorInvalidHandle);
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream);
// hipEventRecordExternal (1) matters only inside a graph being captured,
// where an event records as a node of its own; outside one both record alike.
hipError_t hipEventRecordWithFlags(hipEvent_t event, hipStream_t stream, unsigned int flags) {
  if (flags > 1) {
    const ApiCall api("hipEventRecordWithFlags");
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return record(s, hipErrorInvalidValue);
  }
  return hipEventRecord(event, stream);
}
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  const ApiCall api("hipEventRecord");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  Order o;
  if (const hipError_t e = order_for(s, stream, &o); e != hipSuccess) return record(s, e);
  std::lock_guard<std::mutex> event_lock(g_event_mutex);
  Event* e = find_event(event);
  if (!e) return record(s, hipErrorInvalidHandle);
  // An event records only in a stream of its own device.
  if (e->device != o.device) return record(s, hipErrorInvalidHandle);
  // In a capture it stands for where the stream's capture has got to.
  if (const hipStream_t r = resolve(s, stream); r && stream_capturing(s, r)) return record(s, capture_record(s, r, e));
  auto stamp = std::make_shared<Stamp>();
  e->recorded = true;
  e->stamp = stamp;
  e->queue = o.queue;
  // An interprocess event's record counts for every process that has it.
  std::shared_ptr<IpcEventState> ipc = e->ipc;
  const uint64_t made = ipc ? ipc->counters[0].fetch_add(1) + 1 : 0;
  e->seq = o.queue->submit(
      [stamp, ipc, made] {
        stamp->when = std::chrono::steady_clock::now();
        stamp->taken.store(true, std::memory_order_release);
        if (ipc) ipc->done(made);
        return hipSuccess;
      },
      std::move(o.after));
  return record(s, hipSuccess);
}

// What an event marks: the queue and the item, or nothing where it was never
// recorded.
bool event_marker(hipEvent_t event, Queue::Marker* m, std::shared_ptr<Stamp>* stamp, bool* exists) {
  std::lock_guard<std::mutex> lock(g_event_mutex);
  Event* e = find_event(event);
  *exists = e != nullptr;
  if (!e || !e->recorded) return false;
  *m = {e->queue, e->seq};
  if (stamp) *stamp = e->stamp;
  return true;
}

// Whether an event was last recorded in a capture still going on, which
// has nothing to wait for or ask about yet.
static bool in_capture(hipEvent_t event) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  std::lock_guard<std::mutex> event_lock(g_event_mutex);
  const Event* e = find_event(event);
  return e && captured_event(e);
}

// Another process's event: its shared state, and how many records of it
// have been made by now -- what a wait on it waits for.
static std::shared_ptr<IpcEventState> remote_event(hipEvent_t event, uint64_t* made) {
  std::lock_guard<std::mutex> lock(g_event_mutex);
  const Event* e = find_event(event);
  if (!e || !e->remote || !e->ipc) return nullptr;
  *made = e->ipc->counters[0].load();
  return e->ipc;
}

hipError_t hipEventSynchronize(hipEvent_t event) {
  const ApiCall api("hipEventSynchronize");
  if (in_capture(event)) return record(state(), hipErrorCapturedEvent);
  if (uint64_t made = 0; const auto ipc = remote_event(event, &made)) {
    ipc->wait(made);
    return hipSuccess;
  }
  Queue::Marker m;
  bool exists = false;
  if (event_marker(event, &m, nullptr, &exists)) m.queue->wait(m.seq);
  if (!exists) return record(state(), hipErrorInvalidHandle);
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  release_to_thresholds(s);
  return hipSuccess;
}

hipError_t hipEventQuery(hipEvent_t event) {
  const ApiCall api("hipEventQuery");
  if (in_capture(event)) return record(state(), hipErrorCapturedEvent);
  if (uint64_t made = 0; const auto ipc = remote_event(event, &made))
    return ipc->finished(made) ? hipSuccess : hipErrorNotReady;
  Queue::Marker m;
  bool exists = false;
  if (event_marker(event, &m, nullptr, &exists) && !m.queue->done(m.seq)) return hipErrorNotReady;
  return exists ? hipSuccess : record(state(), hipErrorInvalidHandle);
}

hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t end) {
  const ApiCall api("hipEventElapsedTime");
  if (!ms) return record(state(), hipErrorInvalidValue);
  std::lock_guard<std::mutex> lock(g_event_mutex);
  const Event* a = find_event(start);
  const Event* b = find_event(end);
  // An event that was never recorded, or one created without timing, has no
  // time to give, nor two on different devices; one whose work has not
  // finished has none yet.
  if (!a || !b || !a->recorded || !b->recorded || !a->timing || !b->timing || a->device != b->device)
    return record(state(), hipErrorInvalidHandle);
  if (!a->stamp->taken.load(std::memory_order_acquire) || !b->stamp->taken.load(std::memory_order_acquire))
    return hipErrorNotReady;
  *ms = std::chrono::duration<float, std::milli>(b->stamp->when - a->stamp->when).count();
  return hipSuccess;
}

// The versions a program checks before it trusts a feature. HIP reports
// these as major * 10000000 + minor * 100000 + patch.
// ---- The launch ABI hipcc compiles a program against ------------------------

// Before main: the program's device code. The handle it is given back is what
// it names the binary by when it registers functions and when it exits.
void** __hipRegisterFatBinary(const void* data) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  auto fb = std::make_unique<FatBinary>();
  const auto* wrapper = static_cast<const vgpu::amd::abi::FatbinWrapper*>(data);
  if (!wrapper || wrapper->magic != vgpu::amd::abi::kFatbinMagic || !wrapper->binary ||
      !vgpu::amd::is_bundle(static_cast<const uint8_t*>(wrapper->binary)))
    fail(hipErrorInvalidImage, "the program's device code is not a clang offload bundle this can read");
  else
    fb->image = static_cast<const uint8_t*>(wrapper->binary);
  void** handle = reinterpret_cast<void**>(fb.get());
  s.fat_binaries.push_back(std::move(fb));
  return handle;
}

// Before main, once per kernel: which host-side function stands for which
// kernel in which binary.
void __hipRegisterFunction(void** modules, const void* host_function, char*, const char* device_name, unsigned int,
                           void*, void*, void*, void*, int*) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!modules || !host_function || !device_name) return;
  s.host_functions[host_function] = HostFunction{reinterpret_cast<FatBinary*>(modules), device_name};
}

// Before main, once per __device__ or __constant__ variable: which host
// variable stands for which of the binary's.
void __hipRegisterVar(void** modules, void* host_var, char*, const char* device_name, int, size_t size, int, int) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!modules || !host_var || !device_name) return;
  s.host_vars[host_var] = HostVar{reinterpret_cast<FatBinary*>(modules), device_name, size};
}

// Before main, once per __managed__ variable: its memory, managed and
// holding its initial value, which the host's pointer is set to; the
// modules loaded from the binary point their variable of the same name at it.
void __hipRegisterManagedVar(void* modules, void** pointer, void* init_value, const char* name, size_t size,
                             unsigned) {
  if (!modules || !pointer || !name || !size) return;
  void* p = nullptr;
  if (host_alloc(&p, size, g_managed) != hipSuccess) return;
  if (init_value) std::memcpy(p, init_value, size);
  *pointer = p;
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  auto* fb = reinterpret_cast<FatBinary*>(modules);
  s.managed_vars.push_back({fb, name, pointer});
  for (auto& [ordinal, m] : fb->on_device)
    if (const vgpu::amd::GlobalVar* g = vgpu::amd::find_global(m->object, name)) {
      const uint64_t at = reinterpret_cast<uint64_t>(p);
      s.rt->device(ordinal).memory().write(m->globals + g->offset, &at, sizeof at);
    }
}

// At exit: the binary's modules go, and their variables with them.
void __hipUnregisterFatBinary(void** modules) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  FatBinary* fb = reinterpret_cast<FatBinary*>(modules);
  for (auto it = s.host_functions.begin(); it != s.host_functions.end();)
    it = it->second.binary == fb ? s.host_functions.erase(it) : std::next(it);
  for (auto it = s.host_vars.begin(); it != s.host_vars.end();)
    it = it->second.binary == fb ? s.host_vars.erase(it) : std::next(it);
  s.managed_vars.erase(std::remove_if(s.managed_vars.begin(), s.managed_vars.end(),
                                      [fb](const State::ManagedVar& v) { return v.binary == fb; }),
                       s.managed_vars.end());
  for (size_t i = 0; i < s.fat_binaries.size(); ++i)
    if (s.fat_binaries[i].get() == fb) {
      for (auto& [ordinal, m] : fb->on_device)
        if (m->globals && s.rt && ordinal < s.rt->device_count()) {
          try {
            s.rt->device(ordinal).memory().free(m->globals);
          } catch (const std::exception&) {
          }
        }
      s.fat_binaries.erase(s.fat_binaries.begin() + static_cast<std::ptrdiff_t>(i));
      break;
    }
}

hipError_t __hipPushCallConfiguration(vgpu::amd::abi::Dim3 grid, vgpu::amd::abi::Dim3 block, size_t shared,
                                      hipStream_t stream) {
  const ApiCall api("__hipPushCallConfiguration");
  call_configurations().push_back({grid, block, shared, stream});
  return hipSuccess;
}

hipError_t __hipPopCallConfiguration(vgpu::amd::abi::Dim3* grid, vgpu::amd::abi::Dim3* block, size_t* shared,
                                     hipStream_t* stream) {
  const ApiCall api("__hipPopCallConfiguration");
  if (call_configurations().empty()) return hipErrorInvalidConfiguration;
  const CallConfiguration c = call_configurations().back();
  call_configurations().pop_back();
  if (grid) *grid = c.grid;
  if (block) *block = c.block;
  if (shared) *shared = c.shared;
  if (stream) *stream = c.stream;
  return hipSuccess;
}

}  // extern "C"
namespace {
hipError_t launch_host_function(const void* host_function, vgpu::amd::abi::Dim3 grid, vgpu::amd::abi::Dim3 block,
                                void** args, void** extra, size_t shared, hipStream_t stream, bool cooperative);
}  // namespace
extern "C" {

hipError_t hipLaunchKernel(const void* host_function, vgpu::amd::abi::Dim3 grid, vgpu::amd::abi::Dim3 block,
                           void** args, size_t shared, hipStream_t stream) {
  const ApiCall api("hipLaunchKernel");
  return record(state(), launch_host_function(host_function, grid, block, args, nullptr, shared, stream, false));
}

// ---- What the device is, in the layout the HIP headers give it ---------------

}  // extern "C"

namespace {

// What a device is, in the layout the HIP headers give it: what
// hipGetDeviceProperties returns and hipDeviceGetAttribute answers from.
// HIP's version of an AMD device is its gfx version's first two parts:
// gfx942 is 9.4, gfx90a 9.0 ("gfx" + major + minor digit + stepping).
void hip_version(const vgpu::DeviceProfile& p, int* major, int* minor) {
  *major = p.cc_major;
  *minor = p.cc_minor;
  if (p.gcn_arch.size() >= 6 && p.gcn_arch.rfind("gfx", 0) == 0) {
    const std::string digits = p.gcn_arch.substr(3);
    *major = std::atoi(digits.substr(0, digits.size() - 2).c_str());
    *minor = static_cast<int>(std::strtol(digits.substr(digits.size() - 2, 1).c_str(), nullptr, 16));
  }
}

void fill_properties(const vgpu::DeviceProfile& p, int ordinal, vgpu::amd::abi::DevicePropR0600* props) {
  std::memset(props, 0, sizeof *props);
  std::snprintf(props->name, sizeof props->name, "%s", p.model.c_str());
  std::snprintf(props->gcnArchName, sizeof props->gcnArchName, "%s", p.gcn_arch_full.c_str());
  props->totalGlobalMem = static_cast<size_t>(p.vram_bytes);
  props->sharedMemPerBlock = static_cast<size_t>(p.limits.shared_mem_per_block);
  props->sharedMemPerBlockOptin = static_cast<size_t>(p.limits.shared_mem_per_block_optin);
  props->sharedMemPerMultiprocessor = static_cast<size_t>(p.limits.shared_mem_per_sm);
  props->maxSharedMemoryPerMultiProcessor = static_cast<size_t>(p.limits.shared_mem_per_sm);
  props->regsPerBlock = static_cast<int>(p.limits.registers_per_block);
  props->regsPerMultiprocessor = static_cast<int>(p.limits.registers_per_sm);
  props->warpSize = static_cast<int>(p.warp_size);
  props->maxThreadsPerBlock = static_cast<int>(p.limits.max_threads_per_block);
  for (int i = 0; i < 3; ++i) {
    props->maxThreadsDim[i] = static_cast<int>(p.limits.max_block_dim[i]);
    // HIP's grid sizes are ints, and AMD's runtime caps them there. A
    // profile read from a card's HSA agent (MI325X's) has the agent's
    // 4294967295, which as an int is -1: every grid PyTorch sized against
    // it was too big ("M should be less than maximum CUDA grid size").
    props->maxGridSize[i] = static_cast<int>(std::min<uint64_t>(p.limits.max_grid_dim[i], INT32_MAX));
  }
  props->clockRate = static_cast<int>(p.telemetry.sm_clock_max_mhz) * 1000;
  props->memoryClockRate = static_cast<int>(p.telemetry.mem_clock_max_mhz) * 1000;
  props->multiProcessorCount = static_cast<int>(p.limits.multiprocessors);
  props->maxThreadsPerMultiProcessor = static_cast<int>(p.limits.max_threads_per_sm);
  props->maxBlocksPerMultiProcessor = static_cast<int>(p.limits.max_blocks_per_sm);
  props->l2CacheSize = static_cast<int>(p.limits.l2_cache_bytes);
  hip_version(p, &props->major, &props->minor);
  // The PCI address rocm-smi and sysfs give the device (telemetry's
  // describe_device): a bus of its own, ordinal + 1, device 0. Two devices on
  // one bus looked to RCCL like one GPU twice.
  props->pciDomainID = 0;
  props->pciBusID = ordinal + 1;
  props->pciDeviceID = 0;
  props->concurrentKernels = 1;
  props->cooperativeLaunch = 1;
  // What its code can do, as the compiler's __HIP_ARCH_HAS_*__ macros say
  // for every AMD target: atomics of each width in global and shared memory,
  // doubles, the warp functions, a system fence, 3D grids -- and no funnel
  // shift, extended syncthreads, surface functions or dynamic parallelism.
  auto& a = props->arch;
  a.hasGlobalInt32Atomics = a.hasGlobalFloatAtomicExch = a.hasSharedInt32Atomics = 1;
  a.hasSharedFloatAtomicExch = a.hasFloatAtomicAdd = a.hasGlobalInt64Atomics = a.hasSharedInt64Atomics = 1;
  a.hasDoubles = a.hasWarpVote = a.hasWarpBallot = a.hasWarpShuffle = a.hasThreadFenceSystem = a.has3dGrid = 1;
  props->unifiedAddressing = 1;
  // Pinned, registered and managed host memory are mapped for every
  // device's kernels (see hipHostMalloc), and allocations come from pools.
  props->canMapHostMemory = 1;
  props->managedMemory = 1;
  // Managed memory is host memory every device reaches at once: the host and
  // the devices may use it at the same time.
  props->concurrentManagedAccess = 1;
  // But these are discrete GPUs: the host reaching device memory directly is
  // an APU's property (ROCm's HIP sets it from the agent's direct host
  // access), and RCCL, seeing it, allocates its host-visible flags as
  // uncached device memory and zeroes them from the CPU.
  props->directManagedMemAccessFromHost = 0;
  props->hostRegisterSupported = 1;
  props->memoryPoolsSupported = 1;
  props->ECCEnabled = p.telemetry.ecc ? 1 : 0;
  // The widest row a 2D or 3D copy takes (the attribute, an int, caps it
  // there). The texture alignments stay 0, as ROCm's HIP gives them for a
  // device without image support.
  props->memPitch = static_cast<size_t>(std::min<uint64_t>(p.vram_bytes, INT32_MAX));
  // What ROCm's HIP gives as a device's UUID: the sixteen characters after
  // "GPU-" in its HSA agent's UUID (hsa_api.cpp), which rocminfo prints.
  char uuid[17];
  std::snprintf(uuid, sizeof uuid, "%016llx", 0x5647505500000000ull + static_cast<unsigned>(ordinal));
  std::memcpy(props->uuid.bytes, uuid, 16);
}

// The same properties in the older layout: what hipGetDeviceProperties (and
// hipGetDevicePropertiesR0000) fills, for a program built to that ABI. Each field is the R0600 one of the same name, so the two cannot
// disagree; R0000's gcnArch, the number a gfx target was before it had a
// name, is the target's digits (942).
void fill_properties_r0000(const vgpu::DeviceProfile& p, int ordinal, vgpu::amd::abi::DevicePropR0000* out) {
  vgpu::amd::abi::DevicePropR0600 in;
  fill_properties(p, ordinal, &in);
  *out = {};
  std::memcpy(out->name, in.name, sizeof out->name);
  std::memcpy(out->gcnArchName, in.gcnArchName, sizeof out->gcnArchName);
  for (int i = 0; i < 3; ++i) {
    out->maxThreadsDim[i] = in.maxThreadsDim[i];
    out->maxGridSize[i] = in.maxGridSize[i];
    out->maxTexture3D[i] = in.maxTexture3D[i];
  }
  for (int i = 0; i < 2; ++i) out->maxTexture2D[i] = in.maxTexture2D[i];
  out->totalGlobalMem = in.totalGlobalMem;
  out->sharedMemPerBlock = in.sharedMemPerBlock;
  out->regsPerBlock = in.regsPerBlock;
  out->warpSize = in.warpSize;
  out->maxThreadsPerBlock = in.maxThreadsPerBlock;
  out->clockRate = in.clockRate;
  out->memoryClockRate = in.memoryClockRate;
  out->memoryBusWidth = in.memoryBusWidth;
  out->totalConstMem = in.totalConstMem;
  out->major = in.major;
  out->minor = in.minor;
  out->multiProcessorCount = in.multiProcessorCount;
  out->l2CacheSize = in.l2CacheSize;
  out->maxThreadsPerMultiProcessor = in.maxThreadsPerMultiProcessor;
  out->computeMode = in.computeMode;
  out->clockInstructionRate = in.clockInstructionRate;
  out->arch = in.arch;
  out->concurrentKernels = in.concurrentKernels;
  out->pciDomainID = in.pciDomainID;
  out->pciBusID = in.pciBusID;
  out->pciDeviceID = in.pciDeviceID;
  out->maxSharedMemoryPerMultiProcessor = in.maxSharedMemoryPerMultiProcessor;
  out->isMultiGpuBoard = in.isMultiGpuBoard;
  out->canMapHostMemory = in.canMapHostMemory;
  out->integrated = in.integrated;
  out->cooperativeLaunch = in.cooperativeLaunch;
  out->cooperativeMultiDeviceLaunch = in.cooperativeMultiDeviceLaunch;
  out->maxTexture1DLinear = in.maxTexture1DLinear;
  out->maxTexture1D = in.maxTexture1D;
  out->hdpMemFlushCntl = in.hdpMemFlushCntl;
  out->hdpRegFlushCntl = in.hdpRegFlushCntl;
  out->memPitch = in.memPitch;
  out->textureAlignment = in.textureAlignment;
  out->texturePitchAlignment = in.texturePitchAlignment;
  out->kernelExecTimeoutEnabled = in.kernelExecTimeoutEnabled;
  out->ECCEnabled = in.ECCEnabled;
  out->tccDriver = in.tccDriver;
  out->cooperativeMultiDeviceUnmatchedFunc = in.cooperativeMultiDeviceUnmatchedFunc;
  out->cooperativeMultiDeviceUnmatchedGridDim = in.cooperativeMultiDeviceUnmatchedGridDim;
  out->cooperativeMultiDeviceUnmatchedBlockDim = in.cooperativeMultiDeviceUnmatchedBlockDim;
  out->cooperativeMultiDeviceUnmatchedSharedMem = in.cooperativeMultiDeviceUnmatchedSharedMem;
  out->isLargeBar = in.isLargeBar;
  out->asicRevision = in.asicRevision;
  out->managedMemory = in.managedMemory;
  out->directManagedMemAccessFromHost = in.directManagedMemAccessFromHost;
  out->concurrentManagedAccess = in.concurrentManagedAccess;
  out->pageableMemoryAccess = in.pageableMemoryAccess;
  out->pageableMemoryAccessUsesHostPageTables = in.pageableMemoryAccessUsesHostPageTables;
  out->gcnArch = std::atoi(p.gcn_arch.c_str() + (p.gcn_arch.rfind("gfx", 0) == 0 ? 3 : 0));
}

}  // namespace

extern "C" {

// The older layout, under both names a program may ask for it by.
static_assert(sizeof(hipDeviceProp_t) == sizeof(vgpu::amd::abi::DevicePropR0000),
              "vgpu_hip.h's hipDeviceProp_t is the R0000 layout");
hipError_t hipGetDevicePropertiesR0000(vgpu::amd::abi::DevicePropR0000* props, int ordinal) {
  const ApiCall api("hipGetDevicePropertiesR0000");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!props) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (ordinal < 0 || ordinal >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  fill_properties_r0000(s.rt->device(ordinal).profile(), ordinal, props);
  return record(s, hipSuccess);
}
hipError_t hipGetDeviceProperties(hipDeviceProp_t* props, int ordinal) {
  const ApiCall api("hipGetDeviceProperties");
  return hipGetDevicePropertiesR0000(reinterpret_cast<vgpu::amd::abi::DevicePropR0000*>(props), ordinal);
}

hipError_t hipGetDevicePropertiesR0600(vgpu::amd::abi::DevicePropR0600* props, int ordinal) {
  const ApiCall api("hipGetDevicePropertiesR0600");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!props) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (ordinal < 0 || ordinal >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  fill_properties(s.rt->device(ordinal).profile(), ordinal, props);
  return record(s, hipSuccess);
}

// One property of a device, as a number: the same answer
// hipGetDeviceProperties gives, so the two cannot disagree. Sizes past what an
// int holds are clamped to the largest int. An attribute that is not a number
// -- a name, a pointer -- or one not answered here is refused.
hipError_t hipDeviceGetAttribute(int* value, int attribute, int ordinal) {
  const ApiCall api("hipDeviceGetAttribute");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!value) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (ordinal < 0 || ordinal >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  vgpu::amd::abi::DevicePropR0600 p;
  fill_properties(s.rt->device(ordinal).profile(), ordinal, &p);
  const auto clamp = [](size_t v) { return v > 0x7fffffff ? 0x7fffffff : static_cast<int>(v); };
  using A = vgpu::amd::abi::DeviceAttribute;
  switch (static_cast<A>(attribute)) {
    case A::kEccEnabled: *value = p.ECCEnabled; break;
    case A::kAsyncEngineCount: *value = p.asyncEngineCount; break;
    case A::kCanMapHostMemory: *value = p.canMapHostMemory; break;
    case A::kCanUseHostPointerForRegisteredMem: *value = p.canUseHostPointerForRegisteredMem; break;
    case A::kClockRate: *value = p.clockRate; break;
    case A::kComputeMode: *value = p.computeMode; break;
    case A::kComputePreemptionSupported: *value = p.computePreemptionSupported; break;
    case A::kConcurrentKernels: *value = p.concurrentKernels; break;
    case A::kConcurrentManagedAccess: *value = p.concurrentManagedAccess; break;
    case A::kCooperativeLaunch: *value = p.cooperativeLaunch; break;
    case A::kCooperativeMultiDeviceLaunch: *value = p.cooperativeMultiDeviceLaunch; break;
    case A::kDeviceOverlap: *value = p.deviceOverlap; break;
    case A::kDirectManagedMemAccessFromHost: *value = p.directManagedMemAccessFromHost; break;
    case A::kGlobalL1CacheSupported: *value = p.globalL1CacheSupported; break;
    case A::kHostNativeAtomicSupported: *value = p.hostNativeAtomicSupported; break;
    case A::kIntegrated: *value = p.integrated; break;
    case A::kIsMultiGpuBoard: *value = p.isMultiGpuBoard; break;
    case A::kKernelExecTimeout: *value = p.kernelExecTimeoutEnabled; break;
    case A::kL2CacheSize: *value = p.l2CacheSize; break;
    case A::kLocalL1CacheSupported: *value = p.localL1CacheSupported; break;
    case A::kComputeCapabilityMajor: *value = p.major; break;
    case A::kManagedMemory: *value = p.managedMemory; break;
    case A::kMaxBlocksPerMultiProcessor: *value = p.maxBlocksPerMultiProcessor; break;
    case A::kMaxBlockDimX: *value = p.maxThreadsDim[0]; break;
    case A::kMaxBlockDimY: *value = p.maxThreadsDim[1]; break;
    case A::kMaxBlockDimZ: *value = p.maxThreadsDim[2]; break;
    case A::kMaxGridDimX: *value = p.maxGridSize[0]; break;
    case A::kMaxGridDimY: *value = p.maxGridSize[1]; break;
    case A::kMaxGridDimZ: *value = p.maxGridSize[2]; break;
    case A::kMaxThreadsPerBlock: *value = p.maxThreadsPerBlock; break;
    case A::kMaxThreadsPerMultiProcessor: *value = p.maxThreadsPerMultiProcessor; break;
    case A::kMaxPitch: *value = clamp(p.memPitch); break;
    case A::kMemoryBusWidth: *value = p.memoryBusWidth; break;
    case A::kMemoryClockRate: *value = p.memoryClockRate; break;
    case A::kComputeCapabilityMinor: *value = p.minor; break;
    case A::kMultiGpuBoardGroupID: *value = p.multiGpuBoardGroupID; break;
    case A::kMultiprocessorCount: *value = p.multiProcessorCount; break;
    case A::kPageableMemoryAccess: *value = p.pageableMemoryAccess; break;
    case A::kPageableMemoryAccessUsesHostPageTables: *value = p.pageableMemoryAccessUsesHostPageTables; break;
    case A::kPciBusId: *value = p.pciBusID; break;
    case A::kPciDeviceId: *value = p.pciDeviceID; break;
    case A::kPciDomainID: *value = p.pciDomainID; break;
    case A::kPersistingL2CacheMaxSize: *value = p.persistingL2CacheMaxSize; break;
    case A::kMaxRegistersPerBlock: *value = p.regsPerBlock; break;
    case A::kMaxRegistersPerMultiprocessor: *value = p.regsPerMultiprocessor; break;
    case A::kReservedSharedMemPerBlock: *value = clamp(p.reservedSharedMemPerBlock); break;
    case A::kMaxSharedMemoryPerBlock: *value = clamp(p.sharedMemPerBlock); break;
    case A::kSharedMemPerBlockOptin: *value = clamp(p.sharedMemPerBlockOptin); break;
    case A::kSharedMemPerMultiprocessor: *value = clamp(p.sharedMemPerMultiprocessor); break;
    case A::kSingleToDoublePrecisionPerfRatio: *value = p.singleToDoublePrecisionPerfRatio; break;
    case A::kStreamPrioritiesSupported: *value = p.streamPrioritiesSupported; break;
    case A::kSurfaceAlignment: *value = clamp(p.surfaceAlignment); break;
    case A::kTccDriver: *value = p.tccDriver; break;
    case A::kTextureAlignment: *value = clamp(p.textureAlignment); break;
    case A::kTexturePitchAlignment: *value = clamp(p.texturePitchAlignment); break;
    case A::kTotalConstantMemory: *value = clamp(p.totalConstMem); break;
    case A::kTotalGlobalMem: *value = clamp(p.totalGlobalMem); break;
    case A::kUnifiedAddressing: *value = p.unifiedAddressing; break;
    case A::kWarpSize: *value = p.warpSize; break;
    case A::kMemoryPoolsSupported: *value = p.memoryPoolsSupported; break;
    case A::kHostRegisterSupported: *value = p.hostRegisterSupported; break;
    case A::kClockInstructionRate: *value = p.clockInstructionRate; break;
    case A::kMaxSharedMemoryPerMultiprocessor: *value = clamp(p.maxSharedMemoryPerMultiProcessor); break;
    case A::kCooperativeMultiDeviceUnmatchedFunc: *value = p.cooperativeMultiDeviceUnmatchedFunc; break;
    case A::kCooperativeMultiDeviceUnmatchedGridDim: *value = p.cooperativeMultiDeviceUnmatchedGridDim; break;
    case A::kCooperativeMultiDeviceUnmatchedBlockDim: *value = p.cooperativeMultiDeviceUnmatchedBlockDim; break;
    case A::kCooperativeMultiDeviceUnmatchedSharedMem: *value = p.cooperativeMultiDeviceUnmatchedSharedMem; break;
    case A::kIsLargeBar: *value = p.isLargeBar; break;
    case A::kAsicRevision: *value = p.asicRevision; break;
    case A::kPhysicalMultiProcessorCount: *value = p.multiProcessorCount; break;
    case A::kAccessPolicyMaxWindowSize: *value = p.accessPolicyMaxWindowSize; break;
    case A::kLuid: std::memcpy(value, p.luid, sizeof(int)); break;   // the first bytes, as ROCm's HIP gives them
    case A::kLuidDeviceNodeMask: *value = static_cast<int>(p.luidDeviceNodeMask); break;
    // The size limits for images, by their first dimension where the
    // properties give several, as ROCm's HIP answers them.
    case A::kMaxSurface1D: *value = p.maxSurface1D; break;
    case A::kMaxSurface1DLayered: *value = p.maxSurface1DLayered[0]; break;
    case A::kMaxSurface2D: *value = p.maxSurface2D[0]; break;
    case A::kMaxSurface2DLayered: *value = p.maxSurface2DLayered[0]; break;
    case A::kMaxSurface3D: *value = p.maxSurface3D[0]; break;
    case A::kMaxSurfaceCubemap: *value = p.maxSurfaceCubemap; break;
    case A::kMaxSurfaceCubemapLayered: *value = p.maxSurfaceCubemapLayered[0]; break;
    case A::kMaxTexture1DWidth: *value = p.maxTexture1D; break;
    case A::kMaxTexture1DLayered: *value = p.maxTexture1DLayered[0]; break;
    case A::kMaxTexture1DLinear: *value = p.maxTexture1DLinear; break;
    case A::kMaxTexture1DMipmap: *value = p.maxTexture1DMipmap; break;
    case A::kMaxTexture2DWidth: *value = p.maxTexture2D[0]; break;
    case A::kMaxTexture2DHeight: *value = p.maxTexture2D[1]; break;
    case A::kMaxTexture2DGather: *value = p.maxTexture2DGather[0]; break;
    case A::kMaxTexture2DLayered: *value = p.maxTexture2DLayered[0]; break;
    case A::kMaxTexture2DLinear: *value = p.maxTexture2DLinear[0]; break;
    case A::kMaxTexture2DMipmap: *value = p.maxTexture2DMipmap[0]; break;
    case A::kMaxTexture3DWidth: *value = p.maxTexture3D[0]; break;
    case A::kMaxTexture3DHeight: *value = p.maxTexture3D[1]; break;
    case A::kMaxTexture3DDepth: *value = p.maxTexture3D[2]; break;
    case A::kMaxTexture3DAlt: *value = p.maxTexture3DAlt[0]; break;
    case A::kMaxTextureCubemap: *value = p.maxTextureCubemap; break;
    case A::kMaxTextureCubemapLayered: *value = p.maxTextureCubemapLayered[0]; break;
    case A::kVirtualMemoryManagementSupported: *value = 1; break;   // hipMemCreate and hipMemMap work
    case A::kMemoryPoolSupportedHandleTypes: *value = 1; break;     // hipMemHandleTypePosixFileDescriptor
    // ROCm's HIP writes the register's address, a pointer, where the int
    // is: a program asking passes a pointer's room.
    case A::kHdpMemFlushCntl: std::memcpy(value, &p.hdpMemFlushCntl, sizeof(p.hdpMemFlushCntl)); break;
    case A::kHdpRegFlushCntl: std::memcpy(value, &p.hdpRegFlushCntl, sizeof(p.hdpRegFlushCntl)); break;
    // The real-time clock s_memrealtime and wall_clock64() read, in kHz.
    case A::kWallClockRate: *value = 100000; break;
    case A::kNumberOfXccs:
      *value = static_cast<int>(vgpu::amd::chip(s.rt->device(ordinal).profile().architecture.c_str()).xccs);
      break;
    // A work-item's VGPRs: gfx90a and later add as many accumulation
    // registers again, which a kernel may use as either.
    case A::kMaxAvailableVgprsPerThread: {
      const std::string arch = p.gcnArchName;
      *value = arch.rfind("gfx90a", 0) == 0 || arch.rfind("gfx94", 0) == 0 || arch.rfind("gfx95", 0) == 0 ? 512 : 256;
      break;
    }
    case A::kPciChipId: *value = static_cast<int>(s.rt->device(ordinal).profile().telemetry.pci_device_id); break;
    // Not here: images and textures -- the MI300 family has no texture units,
    // and hipcc refuses the texture API for gfx94x and gfx950 -- and
    // fine-grained host memory.
    case A::kImageSupport:
    case A::kFineGrainSupport: *value = 0; break;
    case A::kCanUseStreamWaitValue: *value = 1; break;   // hipStreamWaitValue32/64
    default:
      return record(s, fail(hipErrorInvalidValue, "device attribute " + std::to_string(attribute) + " is not answered here"));
  }
  return record(s, hipSuccess);
}

// Nothing here waits on anything, so how a program would like to wait changes
// nothing; the flags it may ask for are accepted and any other is refused.
// The flags are kept for hipGetDeviceFlags: the schedule bits, which is all
// ROCm's HIP keeps (host memory is always mapped, and LDS never resized).
hipError_t hipSetDeviceFlags(unsigned int flags) {
  const ApiCall api("hipSetDeviceFlags");
  const unsigned int known = 0x7 /* schedule */ | 0x8 /* map host */ | 0x10 /* lmem resize */;
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (flags & ~known) return record(s, hipErrorInvalidValue);
  s.device_flags[s.current] = flags & 0x7;
  return record(s, hipSuccess);
}

// ---- Host memory -------------------------------------------------------------
//
// Pinned (hipHostMalloc), registered (hipHostRegister) and managed
// (hipMallocManaged) memory is the host's own, and a kernel on any device
// reaches it at its host address, as HIP's unified addressing promises: each
// range is mapped into every device's memory. Managed memory needs no
// migration here -- there is one memory, not two. What was handed out is kept
// by address, so hipPointerGetAttributes can say which kind an address is.
// Lock order: State's mutex, then this one.
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags) {
  const ApiCall api("hipHostMalloc");
  // Coherent and non-coherent at once is a contradiction HIP refuses.
  if ((flags & 0x40000000u) && (flags & 0x80000000u)) return record(state(), hipErrorInvalidValue);
  return host_alloc(ptr, size, g_host_allocations, flags);
}

hipError_t hipHostFree(void* ptr) {
  const ApiCall api("hipHostFree");
  State& s = state();
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    if (const hipError_t e = unsafe_during_capture(s); e != hipSuccess) return record(s, e);
  }
  // Queued copies may still read or write it: HIP's hipHostFree waits for
  // the device first.
  if (ptr) {
    int ordinal = 0;
    {
      std::lock_guard<std::mutex> lock(s.mutex);
      ordinal = s.current;
    }
    if (s.rt) drain_device(ordinal);
  }
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr) return record(s, hipSuccess);
  return record(s, host_free(s, ptr, g_host_allocations) ? hipSuccess : hipErrorInvalidValue);
}

// The caller's own memory, made reachable from kernels where it is. Any
// overlap with memory pinned or managed is refused; registered memory may be
// registered again, as ROCm's HIP allows.
// The flags there are: portable, mapped, I/O memory, read-only, and AMD's
// uncached. Memory that cannot be there -- more than the host has -- is the
// wrong value too.
hipError_t hipHostRegister(void* ptr, size_t size, unsigned int flags) {
  const ApiCall api("hipHostRegister");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (const hipError_t e = unsafe_during_capture(s); e != hipSuccess) return record(s, e);
  if (!ptr || !size || (flags & ~(0xFu | 0x80000000u))) return record(s, hipErrorInvalidValue);
  // Every page of it has to be the process's: msync says which are not.
  const uintptr_t page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
  const uintptr_t first = reinterpret_cast<uintptr_t>(ptr) & ~(page - 1);
  const uintptr_t end = reinterpret_cast<uintptr_t>(ptr) + size;
  if (end < first || msync(reinterpret_cast<void*>(first), (end - first + page - 1) & ~(page - 1), MS_ASYNC) != 0)
    return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  const uint64_t lo = reinterpret_cast<uint64_t>(ptr), hi = lo + size;
  std::lock_guard<std::mutex> host_lock(g_host_mutex);
  for (const auto* m : {&g_host_allocations, &g_managed}) {
    auto it = m->lower_bound(hi);
    if (it != m->begin() && std::prev(it)->first + std::prev(it)->second > lo)
      return record(s, hipErrorHostMemoryAlreadyRegistered);
  }
  // Registered memory registered again, in part or whole, is accepted as
  // ROCm's HIP accepts it; the same address again takes the new size.
  g_host_registrations[lo] = size;
  remap_registered(s);
  return record(s, hipSuccess);
}

hipError_t hipHostUnregister(void* ptr) {
  const ApiCall api("hipHostUnregister");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (const hipError_t e = unsafe_during_capture(s); e != hipSuccess) return record(s, e);
  if (!ptr) return record(s, hipErrorInvalidValue);
  std::lock_guard<std::mutex> host_lock(g_host_mutex);
  const auto it = g_host_registrations.find(reinterpret_cast<uint64_t>(ptr));
  if (it == g_host_registrations.end()) return record(s, hipErrorHostMemoryNotRegistered);
  g_host_registrations.erase(it);
  if (s.rt) remap_registered(s);
  return record(s, hipSuccess);
}

// Pinned or registered memory is reached by kernels at its host address, so
// that is its device pointer, as hipPointerGetAttributes also says.
hipError_t hipHostGetDevicePointer(void** device_ptr, void* host_ptr, unsigned int flags) {
  const ApiCall api("hipHostGetDevicePointer");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!device_ptr || !host_ptr || flags) return record(s, hipErrorInvalidValue);
  const uint64_t va = reinterpret_cast<uint64_t>(host_ptr);
  std::lock_guard<std::mutex> host_lock(g_host_mutex);
  if (find_range(g_host_allocations, va) == g_host_allocations.end() &&
      find_range(g_host_registered, va) == g_host_registered.end())
    return record(s, hipErrorInvalidValue);
  *device_ptr = host_ptr;
  return record(s, hipSuccess);
}

// One allocation the host and every device address alike: host memory,
// mapped for kernels, freed with hipFree.
hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags) {
  const ApiCall api("hipMallocManaged");
  if (flags != 1 /* hipMemAttachGlobal */ && flags != 2 /* hipMemAttachHost */)
    return record(state(), hipErrorInvalidValue);
  if (!size) return record(state(), hipErrorInvalidValue);
  return host_alloc(ptr, size, g_managed);
}

// Advice about where managed memory should live and who reads it, and where
// it was last prefetched to. There is one memory here, not a host copy and a
// device copy, so none of it moves anything; it is kept, page by page (4 KiB,
// the range a call names rounded out to whole pages), and read back by
// hipMemRangeGetAttribute(s) as ROCm's HIP reads it. What HIP refuses is
// refused the same way: no address, no bytes, advice it does not know, a
// device that is not there, memory that is not managed, or a range running
// past the allocation it starts in. llama.cpp's HIP backend (Ollama's) links
// against hipMemAdvise, and a library without it is not loaded at all.
}  // extern "C"
namespace {
constexpr uint64_t kAdvicePage = 4096;

// The managed allocation `[ptr, ptr + count)` lies in, and the pages it
// covers. The caller holds g_host_mutex.
hipError_t managed_pages(const void* ptr, size_t count, std::vector<PageAdvice>** pages, size_t* first,
                         size_t* last) {
  const uint64_t va = reinterpret_cast<uint64_t>(ptr);
  const auto it = find_range(g_managed, va);
  if (it == g_managed.end()) return hipErrorInvalidValue;
  if (count > it->second - (va - it->first)) return hipErrorInvalidValue;
  auto& v = g_advice[it->first];
  if (v.empty()) v.resize((it->second + kAdvicePage - 1) / kAdvicePage);
  *pages = &v;
  *first = (va - it->first) / kAdvicePage;
  *last = (va - it->first + count + kAdvicePage - 1) / kAdvicePage;
  return hipSuccess;
}

hipError_t advise(State& s, const void* ptr, size_t count, int advice, int device) {   // holds s.mutex
  if (!ptr || !count) return hipErrorInvalidValue;
  // hipMemAdviseSet/UnsetReadMostly, PreferredLocation, AccessedBy are 1-6;
  // AMD's Set/UnsetCoarseGrain are 100 and 101.
  if (!((advice >= 1 && advice <= 6) || advice == 100 || advice == 101)) return hipErrorInvalidValue;
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return e;
  const bool names_device = advice >= 3 && advice <= 6;
  if (names_device && device != -1 /* hipCpuDeviceId */ && (device < 0 || device >= s.rt->device_count()))
    return hipErrorInvalidDevice;
  std::lock_guard<std::mutex> host_lock(g_host_mutex);
  std::vector<PageAdvice>* pages = nullptr;
  size_t first = 0, last = 0;
  if (const hipError_t e = managed_pages(ptr, count, &pages, &first, &last); e != hipSuccess) {
    // Pinned and registered memory takes advice too, and keeps none of it.
    const uint64_t va = reinterpret_cast<uint64_t>(ptr);
    for (const auto* m : {&g_host_allocations, &g_host_registered})
      if (const auto it = find_range(*m, va); it != m->end())
        return count > it->second - (va - it->first) ? hipErrorInvalidValue : hipSuccess;
    return e;
  }
  for (size_t p = first; p < last; ++p) {
    PageAdvice& a = (*pages)[p];
    switch (advice) {
      case 1: a.read_mostly = true; break;
      case 2: a.read_mostly = false; break;
      case 3: a.preferred = device; break;
      case 4: a.preferred = -2; break;
      case 5: a.accessed_by.insert(device); break;
      case 6: a.accessed_by.erase(device); break;
      case 100: a.coarse = true; break;
      case 101: a.coarse = false; break;
    }
  }
  return hipSuccess;
}
// A hipMemLocation, as the device a v1 call would name: a device by its id,
// the host (any node) as hipCpuDeviceId. Its type has to be one there is.
hipError_t location_device(State& s, const vgpu::amd::abi::MemLocation& loc, int* device) {   // holds s.mutex
  switch (loc.type) {
    case 1:   // hipMemLocationTypeDevice
      if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return e;
      if (loc.id < 0 || loc.id >= s.rt->device_count()) return hipErrorInvalidDevice;
      *device = loc.id;
      return hipSuccess;
    case 2:   // Host
    case 3:   // HostNuma
    case 4:   // HostNumaCurrent
      *device = -1;
      return hipSuccess;
    default: return hipErrorInvalidValue;
  }
}
hipError_t prefetch(hipStream_t stream, const void* ptr, size_t count, int device) {
  {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!ptr || !count) return hipErrorInvalidValue;
    if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return e;
    std::lock_guard<std::mutex> host_lock(g_host_mutex);
    std::vector<PageAdvice>* pages = nullptr;
    size_t first = 0, last = 0;
    if (const hipError_t e = managed_pages(ptr, count, &pages, &first, &last); e != hipSuccess) return e;
    if (device != -1 && (device < 0 || device >= s.rt->device_count())) return hipErrorInvalidDevice;
    // Where it was last sent is known as soon as it is asked for.
    for (size_t p = first; p < last; ++p) (*pages)[p].last_prefetch = device;
  }
  // Nothing moves; the prefetch still takes its place in the stream.
  return in_order(stream, [] { return hipSuccess; });
}
// One attribute over the pages of a range: the same value on every page, or
// none. `size` is the bytes the caller gave for it.
hipError_t range_attribute(void* data, size_t size, int attr, const void* ptr, size_t count) {
  if (!data || !size) return hipErrorInvalidValue;
  if (attr == 3 /* AccessedBy */ ? size % 4 != 0 : size != 4) return hipErrorInvalidValue;
  if (attr != 1 && attr != 2 && attr != 3 && attr != 4 && attr != 100) return hipErrorInvalidValue;
  if (!ptr || !count) return hipErrorInvalidValue;
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return e;
  std::lock_guard<std::mutex> host_lock(g_host_mutex);
  std::vector<PageAdvice>* pages = nullptr;
  size_t first = 0, last = 0;
  if (const hipError_t e = managed_pages(ptr, count, &pages, &first, &last); e != hipSuccess) {
    // Device memory is coarse-grained, and says so; it has no other advice.
    if (attr == 100 && owner_of(s, reinterpret_cast<uint64_t>(ptr)) >= 0) {
      *static_cast<int*>(data) = 1;   // hipMemRangeCoherencyModeCoarseGrain
      return hipSuccess;
    }
    return e;
  }
  const auto same = [&](auto get, int none) {
    const int v = get((*pages)[first]);
    for (size_t p = first + 1; p < last; ++p)
      if (get((*pages)[p]) != v) return none;
    return v;
  };
  auto* out = static_cast<int*>(data);
  switch (attr) {
    case 1: *out = same([](const PageAdvice& a) { return a.read_mostly ? 1 : 0; }, 0); break;
    case 2: *out = same([](const PageAdvice& a) { return a.preferred; }, -2); break;
    case 4: *out = same([](const PageAdvice& a) { return a.last_prefetch; }, -2); break;
    case 100: *out = same([](const PageAdvice& a) { return a.coarse ? 1 : 0; }, 2); break;
    case 3: {
      // The devices every page is accessed by: GPUs in order, then the host,
      // then hipInvalidDeviceId in the slots left.
      std::set<int> all = (*pages)[first].accessed_by;
      for (size_t p = first + 1; p < last; ++p) {
        std::set<int> both;
        for (int d : all)
          if ((*pages)[p].accessed_by.count(d)) both.insert(d);
        all.swap(both);
      }
      std::vector<int> order;
      for (int d : all)
        if (d >= 0) order.push_back(d);
      if (all.count(-1)) order.push_back(-1);
      const size_t slots = size / 4;
      for (size_t i = 0; i < slots; ++i) out[i] = i < order.size() ? order[i] : -2;
      break;
    }
  }
  return hipSuccess;
}
}  // namespace
extern "C" {

hipError_t hipMemAdvise(const void* ptr, size_t count, int advice, int device) {
  const ApiCall api("hipMemAdvise");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return record(s, advise(s, ptr, count, advice, device));
}
hipError_t hipMemAdvise_v2(const void* ptr, size_t count, int advice, vgpu::amd::abi::MemLocation location) {
  const ApiCall api("hipMemAdvise_v2");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr || !count) return record(s, hipErrorInvalidValue);
  int device = 0;
  if (const hipError_t e = location_device(s, location, &device); e != hipSuccess) return record(s, e);
  return record(s, advise(s, ptr, count, advice, device));
}
hipError_t hipMemPrefetchAsync(const void* ptr, size_t count, int device, hipStream_t stream) {
  const ApiCall api("hipMemPrefetchAsync");
  return record(state(), prefetch(stream, ptr, count, device));
}
hipError_t hipMemPrefetchAsync_v2(const void* ptr, size_t count, vgpu::amd::abi::MemLocation location,
                                  unsigned int flags, hipStream_t stream) {
  const ApiCall api("hipMemPrefetchAsync_v2");
  int device = 0;
  {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!ptr || !count || flags) return record(s, hipErrorInvalidValue);
    {
      std::lock_guard<std::mutex> host_lock(g_host_mutex);
      std::vector<PageAdvice>* pages = nullptr;
      size_t first = 0, last = 0;
      if (const hipError_t e = managed_pages(ptr, count, &pages, &first, &last); e != hipSuccess) return record(s, e);
    }
    if (const hipError_t e = location_device(s, location, &device); e != hipSuccess) return record(s, e);
  }
  return record(state(), prefetch(stream, ptr, count, device));
}
hipError_t hipMemRangeGetAttribute(void* data, size_t size, int attr, const void* ptr, size_t count) {
  const ApiCall api("hipMemRangeGetAttribute");
  return record(state(), range_attribute(data, size, attr, ptr, count));
}
hipError_t hipMemRangeGetAttributes(void** data, size_t* sizes, int* attrs, size_t n, const void* ptr, size_t count) {
  const ApiCall api("hipMemRangeGetAttributes");
  if (!data || !sizes || !attrs || !n) return record(state(), hipErrorInvalidValue);
  for (size_t i = 0; i < n; ++i)
    if (!data[i]) return record(state(), hipErrorInvalidValue);
  for (size_t i = 0; i < n; ++i)
    if (const hipError_t e = range_attribute(data[i], sizes[i], attrs[i], ptr, count); e != hipSuccess)
      return record(state(), e);
  return record(state(), hipSuccess);
}
// Which streams may reach managed memory: every device reaches all of it at
// once here (concurrent managed access), so the attachment is kept to its
// stream's order and changes nothing. The whole allocation (length 0, or
// all of it) is attached, with one of the three flags, and a single stream's
// attachment needs a stream.
hipError_t hipStreamAttachMemAsync(hipStream_t stream, void* ptr, size_t length, unsigned int flags) {
  const ApiCall api("hipStreamAttachMemAsync");
  {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!ptr || (flags != 1 && flags != 2 && flags != 4)) return record(s, hipErrorInvalidValue);
    if (flags == 4 /* hipMemAttachSingle */ && !stream) return record(s, hipErrorInvalidValue);
    std::lock_guard<std::mutex> host_lock(g_host_mutex);
    const uint64_t va = reinterpret_cast<uint64_t>(ptr);
    const auto it = find_range(g_managed, va);
    if (it == g_managed.end() || it->first != va) return record(s, hipErrorInvalidValue);
    if (length && length != it->second) return record(s, hipErrorInvalidValue);
  }
  return record(state(), in_order(stream, [] { return hipSuccess; }));
}

// ---- One device reaching another --------------------------------------------

hipError_t hipDeviceCanAccessPeer(int* can, int ordinal, int peer) {
  const ApiCall api("hipDeviceCanAccessPeer");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!can) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  const int n = s.rt->device_count();
  if (ordinal < 0 || ordinal >= n || peer < 0 || peer >= n) return record(s, hipErrorInvalidDevice);
  *can = ordinal != peer;   // every device here can reach every other; none reaches itself as a peer
  return record(s, hipSuccess);
}

hipError_t hipDeviceEnablePeerAccess(int peer, unsigned int flags) {
  const ApiCall api("hipDeviceEnablePeerAccess");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (flags) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (peer < 0 || peer >= s.rt->device_count() || peer == s.current) return record(s, hipErrorInvalidDevice);
  if (!s.peers.insert({s.current, peer}).second) return record(s, hipErrorPeerAccessAlreadyEnabled);
  return record(s, hipSuccess);
}

hipError_t hipDeviceDisablePeerAccess(int peer) {
  const ApiCall api("hipDeviceDisablePeerAccess");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (peer < 0 || peer >= s.rt->device_count() || peer == s.current) return record(s, hipErrorInvalidDevice);
  if (!s.peers.erase({s.current, peer})) return record(s, hipErrorPeerAccessNotEnabled);
  return record(s, hipSuccess);
}

// A copy from one device's memory to another's. The address says which device
// owns it, and the device numbers the program gave have to agree.
}  // extern "C"
namespace {
hipError_t peer_copy(void* dst, int dst_device, const void* src, int src_device, size_t bytes, hipStream_t stream,
                     bool async);
}  // namespace
extern "C" {
hipError_t hipMemcpyPeerAsync(void* dst, int dst_device, const void* src, int src_device, size_t bytes,
                              hipStream_t stream) {
  const ApiCall api("hipMemcpyPeerAsync");
  return record(state(), peer_copy(dst, dst_device, src, src_device, bytes, stream, true));
}

// ---- Occupancy: how many work-groups of a kernel a compute unit holds -------

}  // extern "C"

namespace {

// ROCm's runtime works this out from the kernel's registers and LDS and the
// compute unit's limits (hip_platform.cpp), and this is that arithmetic, with
// CDNA's limits (LLVM's gfx9 register tables): four SIMDs to a compute unit,
// at most eight waves on each, 512 vector registers each allocated eight at a
// time, 800 scalar registers sixteen at a time, and the device's LDS.
struct Occupancy {
  int blocks_per_cu = 0;   // of the block size asked about
  int grid_blocks = 0;     // blocks that fill the device at the best block size
  int best_block = 0;
};

hipError_t occupancy(const vgpu::DeviceProfile& p, const Kernel& k, int block, size_t dynamic_lds, bool potential,
                     Occupancy* out) {
  // RDNA3 and RDNA4 alike (gfx12's register file and wave limits are gfx11's).
  const bool rdna3 = p.gcn_arch.rfind("gfx11", 0) == 0 || p.gcn_arch.rfind("gfx12", 0) == 0;
  const bool rdna2 = p.gcn_arch.rfind("gfx103", 0) == 0;
  if (p.gcn_arch.rfind("gfx9", 0) != 0 && !rdna3 && !rdna2)
    return fail(hipErrorNotSupported, "occupancy is worked out for CDNA (gfx9), RDNA2 (gfx103x) and RDNA3 (gfx11) here, and this device is " + p.gcn_arch);
  const int max_group = static_cast<int>(p.limits.max_threads_per_block);
  if (!potential) {
    if (block <= 0) return hipErrorInvalidValue;
    if (block > max_group) {
      *out = {};
      return hipSuccess;
    }
  } else if (block <= 0 || block > max_group) {
    block = max_group;   // no limit asked for, or past what the hardware allows
  }
  const int wave = static_cast<int>(k.wavefront_size ? k.wavefront_size : 64);
  const auto align_up = [](size_t v, size_t to) { return (v + to - 1) / to * to; };
  // RDNA3 (LLVM's gfx11 tables, for the parts with the full register file:
  // gfx1100, 1101, 1151): 16 waves a SIMD, 1536 vector registers a SIMD in
  // wave32 allocated 24 at a time, scalar registers never the limit, and a
  // "compute unit" as HIP counts it is a workgroup processor of four SIMDs.
  // RDNA2 (gfx10.3): the same but for 1024 vector registers, 8 at a time.
  // Each has half as many, half as many at a time, in wave64.
  const bool w64 = k.wavefront_size == 64;
  const bool rdna = rdna3 || rdna2;
  const size_t kMaxWavesPerSimd = rdna ? 16 : 8,
               kVgprsPerSimd = rdna3 ? (w64 ? 768 : 1536) : rdna2 ? (w64 ? 512 : 1024) : 512,
               kVgprGranule = rdna3 ? (w64 ? 12 : 24) : rdna2 ? (w64 ? 4 : 8) : 8, kSgprsPerSimd = 800;
  constexpr size_t kSimdsPerCu = 4;
  size_t gpr_waves = kMaxWavesPerSimd;
  if (k.vgpr_count) gpr_waves = std::min(gpr_waves, kVgprsPerSimd / align_up(k.vgpr_count, kVgprGranule));
  if (gpr_waves == 0) return fail(hipErrorUnknown, "the kernel uses more vector registers than a SIMD has");
  if (k.sgpr_count && !rdna) gpr_waves = std::min(gpr_waves, kSgprsPerSimd / align_up(k.sgpr_count, 16));
  const int alu_threads = static_cast<int>(kSimdsPerCu * std::min(kMaxWavesPerSimd, gpr_waves)) * wave;
  int lds_groups = INT_MAX;
  if (const size_t lds = k.group_segment + dynamic_lds; lds)
    lds_groups = static_cast<int>(p.limits.shared_mem_per_sm / lds);
  out->blocks_per_cu = std::min(alu_threads / static_cast<int>(align_up(block, wave)), lds_groups);
  out->best_block = std::min(alu_threads, static_cast<int>(align_up(block, wave)));
  out->grid_blocks = static_cast<int>(p.limits.multiprocessors) * std::min(alu_threads / out->best_block, lds_groups);
  return hipSuccess;
}

// The kernel a hipcc-built program's host function stands for, on the current
// device.
hipError_t kernel_for(State& s, const void* host_function, const Kernel** k) {
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return e;
  const auto hf = s.host_functions.find(host_function);
  if (hf == s.host_functions.end())
    return fail(hipErrorInvalidDeviceFunction, "no kernel was registered for that function");
  Module* m = nullptr;
  if (const hipError_t e = module_on(s, *hf->second.binary, s.current, &m); e != hipSuccess) return e;
  *k = vgpu::amd::find_kernel(m->object, hf->second.kernel);
  return *k ? hipSuccess : fail(hipErrorInvalidDeviceFunction, "no kernel named " + hf->second.kernel);
}

hipError_t blocks_per_cu(const Kernel* k, int* blocks, int block, size_t lds) {
  State& s = state();
  if (!blocks || !k) return hipErrorInvalidValue;
  Occupancy o;
  if (const hipError_t e = occupancy(s.rt->device(s.current).profile(), *k, block, lds, false, &o); e != hipSuccess)
    return e;
  *blocks = o.blocks_per_cu;
  return hipSuccess;
}

hipError_t best_block(const Kernel* k, int* grid, int* block, size_t lds, int limit) {
  State& s = state();
  if (!grid || !block || !k) return hipErrorInvalidValue;
  Occupancy o;
  if (const hipError_t e = occupancy(s.rt->device(s.current).profile(), *k, limit, lds, true, &o); e != hipSuccess)
    return e;
  *grid = o.grid_blocks;
  *block = o.best_block;
  return hipSuccess;
}

}  // namespace

extern "C" {

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* blocks, const void* f, int block, size_t lds) {
  const ApiCall api("hipOccupancyMaxActiveBlocksPerMultiprocessor");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  const Kernel* k = nullptr;
  if (const hipError_t e = kernel_for(s, f, &k); e != hipSuccess) return record(s, e);
  return record(s, blocks_per_cu(k, blocks, block, lds));
}

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* blocks, const void* f, int block, size_t lds,
                                                                 unsigned int flags) {
  const ApiCall api("hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (flags > 1) return record(s, hipErrorInvalidValue);   // hipOccupancyDefault, or DisableCachingOverride
  const Kernel* k = nullptr;
  if (const hipError_t e = kernel_for(s, f, &k); e != hipSuccess) return record(s, e);
  return record(s, blocks_per_cu(k, blocks, block, lds));
}

hipError_t hipOccupancyMaxPotentialBlockSize(int* grid, int* block, const void* f, size_t lds, int limit) {
  const ApiCall api("hipOccupancyMaxPotentialBlockSize");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  const Kernel* k = nullptr;
  if (const hipError_t e = kernel_for(s, f, &k); e != hipSuccess) return record(s, e);
  return record(s, best_block(k, grid, block, lds, limit));
}

hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int* blocks, hipFunction_t f, int block, size_t lds) {
  const ApiCall api("hipModuleOccupancyMaxActiveBlocksPerMultiprocessor");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!f) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  return record(s, blocks_per_cu(reinterpret_cast<Function*>(f)->kernel, blocks, block, lds));
}

hipError_t hipModuleOccupancyMaxPotentialBlockSize(int* grid, int* block, hipFunction_t f, size_t lds, int limit) {
  const ApiCall api("hipModuleOccupancyMaxPotentialBlockSize");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!f) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  return record(s, best_block(reinterpret_cast<Function*>(f)->kernel, grid, block, lds, limit));
}

// A launch whose work-groups may wait on one another (a grid barrier,
// cooperative_groups::this_grid().sync()). They all have to be resident at
// once, so a grid larger than the device holds of this kernel at this block
// size -- what the occupancy calls say -- is refused, as HIP refuses it.
}  // extern "C"
namespace {

// A kernel named by its host-side function, launched on the current device:
// what a chevron launch, hipLaunchKernel and every other form that names a
// kernel that way come to. Its arguments are one pointer to each value
// (`args`), or already packed (`extra`, as hipModuleLaunchKernel takes them).
// A cooperative launch's grid must fit on the device at once. On a stream
// that is capturing, the launch is recorded rather than run. The caller holds
// no lock and records the result.
hipError_t launch_host_function(const void* host_function, vgpu::amd::abi::Dim3 grid, vgpu::amd::abi::Dim3 block,
                                void** args, void** extra, size_t shared, hipStream_t stream, bool cooperative) {
  State& s = state();
  std::unique_lock<std::mutex> lock(s.mutex);
  vgpu::runtime::Device* d = device(s);
  if (!d) return hipErrorInvalidDevice;
  const auto hf = s.host_functions.find(host_function);
  if (hf == s.host_functions.end())
    return fail(hipErrorInvalidDeviceFunction, "no kernel was registered for that function");
  Module* m = nullptr;
  if (const hipError_t e = module_on(s, *hf->second.binary, s.current, &m); e != hipSuccess) return e;
  const Kernel* k = vgpu::amd::find_kernel(m->object, hf->second.kernel);
  if (!k) return fail(hipErrorInvalidDeviceFunction, "the program's device code has no kernel named " + hf->second.kernel);
  if (cooperative) {
    if (!block.x || !block.y || !block.z || !grid.x || !grid.y || !grid.z) return hipErrorInvalidConfiguration;
    Occupancy o;
    const int threads = static_cast<int>(block.x * block.y * block.z);
    const vgpu::DeviceProfile& p = d->profile();
    if (const hipError_t e = occupancy(p, *k, threads, shared, false, &o); e != hipSuccess) return e;
    const uint64_t resident = uint64_t(o.blocks_per_cu) * p.limits.multiprocessors;
    if (uint64_t{grid.x} * grid.y * grid.z > resident)
      return fail(hipErrorCooperativeLaunchTooLarge,
                  "a cooperative grid of " + std::to_string(uint64_t{grid.x} * grid.y * grid.z) +
                      " work-groups is more than the " + std::to_string(resident) +
                      " this device holds at once at this block size");
  }
  if (const hipError_t e = check_launch(s, s.current, *k, grid, block, shared, stream); e != hipSuccess) return e;
  std::vector<uint8_t> packed;
  if (const hipError_t e = build_kernargs(*k, args, extra, &packed); e != hipSuccess) return e;
  stream = resolve(s, stream);
  if (!stream) {
    if (const hipError_t e = legacy_conflict(s, s.current); e != hipSuccess) return e;
  } else if (stream_capturing(s, stream)) {
    return capture_kernel(s, stream, host_function, *k, grid, block, static_cast<uint32_t>(shared), std::move(packed),
                          extra, cooperative);
  }
  LaunchJob job;
  if (const hipError_t e = prepare_launch(s, s.current, *m, *k, grid, block, static_cast<uint32_t>(shared),
                                          std::move(packed), stream, cooperative, &job);
      e != hipSuccess)
    return e;
  lock.unlock();
  return launch_in_order(std::move(job));
}

}  // namespace
extern "C" {

hipError_t hipLaunchCooperativeKernel(const void* host_function, vgpu::amd::abi::Dim3 grid,
                                      vgpu::amd::abi::Dim3 block, void** args, unsigned int shared,
                                      hipStream_t stream) {
  const ApiCall api("hipLaunchCooperativeKernel");
  return record(state(), launch_host_function(host_function, grid, block, args, nullptr, shared, stream, true));
}

// ---- A program's own device variables, by their host-side stand-ins ---------

}  // extern "C"

namespace {

// Where the variable a host symbol stands for is, on the current device: in
// the binary's module there, loaded if no kernel has been launched from it yet.
hipError_t symbol_on(State& s, const void* symbol, uint64_t* address, size_t* size) {
  const auto it = s.host_vars.find(symbol);
  if (it == s.host_vars.end())
    return fail(hipErrorInvalidSymbol, "that is not a __device__ or __constant__ variable the program registered");
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return e;
  Module* m = nullptr;
  if (const hipError_t e = module_on(s, *it->second.binary, s.current, &m); e != hipSuccess) return e;
  const vgpu::amd::GlobalVar* g = vgpu::amd::find_global(m->object, it->second.name);
  if (!g)
    return fail(hipErrorInvalidSymbol, "the program's device code has no variable named " + it->second.name);
  *address = m->globals + g->offset;
  *size = static_cast<size_t>(g->size ? g->size : it->second.size);
  return hipSuccess;
}

hipError_t copy_symbol(bool to_symbol, const void* symbol, void* host, size_t bytes, size_t offset,
                       hipMemcpyKind kind, hipStream_t stream = nullptr, bool async = false) {
  State& s = state();
  uint64_t address = 0;
  size_t size = 0;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    if (const hipError_t e = symbol_on(s, symbol, &address, &size); e != hipSuccess) return record(s, e);
    if (offset > size || bytes > size - offset)
      return record(s, fail(hipErrorInvalidValue, "the copy runs past the end of the variable"));
  }
  if (kind == hipMemcpyDefault) kind = to_symbol ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost;
  // A copy to a variable comes from the host or a device, and one from a
  // variable goes to either: the other directions are the wrong way round.
  if (kind == hipMemcpyHostToHost || kind == (to_symbol ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice))
    return record(s, hipErrorInvalidMemcpyDirection);
  void* device = reinterpret_cast<void*>(address + offset);
  return record(s, to_symbol ? copy_in_order(device, host, bytes, kind, stream, async)
                             : copy_in_order(host, device, bytes, kind, stream, async));
}

}  // namespace

extern "C" {

hipError_t hipGetSymbolAddress(void** ptr, const void* symbol) {
  const ApiCall api("hipGetSymbolAddress");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr) return record(s, hipErrorInvalidValue);
  uint64_t address = 0;
  size_t size = 0;
  if (const hipError_t e = symbol_on(s, symbol, &address, &size); e != hipSuccess) return record(s, e);
  *ptr = reinterpret_cast<void*>(address);
  return record(s, hipSuccess);
}

hipError_t hipGetSymbolSize(size_t* bytes, const void* symbol) {
  const ApiCall api("hipGetSymbolSize");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!bytes) return record(s, hipErrorInvalidValue);
  uint64_t address = 0;
  if (const hipError_t e = symbol_on(s, symbol, &address, bytes); e != hipSuccess) return record(s, e);
  return record(s, hipSuccess);
}

hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t bytes, size_t offset, hipMemcpyKind kind) {
  const ApiCall api("hipMemcpyToSymbol");
  return copy_symbol(true, symbol, const_cast<void*>(src), bytes, offset, kind);
}

hipError_t hipMemcpyFromSymbol(void* dst, const void* symbol, size_t bytes, size_t offset, hipMemcpyKind kind) {
  const ApiCall api("hipMemcpyFromSymbol");
  return copy_symbol(false, symbol, dst, bytes, offset, kind);
}

// Every copy here has finished when it returns, as every launch has.
hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src, size_t bytes, size_t offset,
                                  hipMemcpyKind kind, hipStream_t stream) {
  const ApiCall api("hipMemcpyToSymbolAsync");
  return copy_symbol(true, symbol, const_cast<void*>(src), bytes, offset, kind, stream, true);
}

hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t bytes, size_t offset, hipMemcpyKind kind,
                                    hipStream_t stream) {
  const ApiCall api("hipMemcpyFromSymbolAsync");
  return copy_symbol(false, symbol, dst, bytes, offset, kind, stream, true);
}

hipError_t hipRuntimeGetVersion(int* version) {
  const ApiCall api("hipRuntimeGetVersion");
  if (!version) return hipErrorInvalidValue;
  *version = 60443483;   // 6.4.43483, a ROCm 6.4 runtime
  return hipSuccess;
}
hipError_t hipDriverGetVersion(int* version) {
  const ApiCall api("hipDriverGetVersion");
  return hipRuntimeGetVersion(version);
}

}  // extern "C"

// ---- The profiler's side (vgpu/hip_profiler.hpp) ------------------------------

extern "C" {

int vgpu_hip_profiler_attach(const vgpu::amd::hipprof::Hooks* hooks) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  g_profiler.store(hooks, std::memory_order_release);
  if (ensure_runtime(s) != hipSuccess) return -1;
  return s.rt->device_count();
}

int vgpu_hip_profiler_device(int ordinal, vgpu::amd::hipprof::Device* out) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!out || ensure_runtime(s) != hipSuccess || ordinal < 0 || ordinal >= s.rt->device_count()) return -1;
  const vgpu::DeviceProfile& p = s.rt->device(ordinal).profile();
  vgpu::amd::hipprof::Device d;
  d.ordinal = ordinal;
  d.gfx = p.gcn_arch.c_str();
  d.model = p.model.c_str();
  d.vendor_id = static_cast<uint16_t>(p.telemetry.pci_vendor_id);
  d.device_id = static_cast<uint16_t>(p.telemetry.pci_device_id);
  d.compute_units = p.limits.multiprocessors;
  d.wave_size = p.warp_size;
  d.lds_bytes = static_cast<uint32_t>(p.limits.shared_mem_per_block);
  d.max_threads_per_cu = p.limits.max_threads_per_sm;
  d.max_workgroup = p.limits.max_threads_per_block;
  for (int i = 0; i < 3; ++i) {
    d.max_block[i] = p.limits.max_block_dim[i];
    d.max_grid[i] = p.limits.max_grid_dim[i];
  }
  d.clock_mhz = p.telemetry.sm_clock_max_mhz;
  d.vram_bytes = p.vram_bytes;
  // The bus and UUID the device reports everywhere else: rocm-smi, sysfs.
  vgpu::telemetry::DeviceSample sample{};
  vgpu::telemetry::describe_device(p, ordinal, &sample);
  unsigned bus = 0;
  if (std::sscanf(sample.bus_id, "%*x:%x:", &bus) == 1) d.pci_bus = bus;
  // "GPU-xxxxxxxx-xxxx-...": its hex digits, sixteen bytes of them.
  size_t n = 0;
  for (const char* c = sample.uuid; *c && n < 32; ++c) {
    int v = *c >= '0' && *c <= '9' ? *c - '0' : *c >= 'a' && *c <= 'f' ? *c - 'a' + 10 : *c >= 'A' && *c <= 'F' ? *c - 'A' + 10 : -1;
    if (v < 0) continue;
    d.uuid[n / 2] = static_cast<uint8_t>(d.uuid[n / 2] << 4 | v);
    ++n;
  }
  *out = d;
  return 0;
}

}  // extern "C"

// ---- What ROCm's libraries call ---------------------------------------------
//
// rocBLAS, hipBLASLt and their kind call these; every one of them here is
// what HIP's documentation says of it.
extern "C" {

// ---- Stream-ordered allocation ------------------------------------------------
//
// The stream has nothing queued ahead of it, so the memory is there, and gone,
// at once. Each allocation is counted against the pool it came from -- a
// device's default pool, or one the program made -- so a pool's usage reads
// back as HIP reports it. Nothing freed is held back, so what a pool has
// reserved is what is in use.
}  // extern "C"

namespace {

// The device's default pool, made the first time it is asked for.
Pool* default_pool(State& s, int ordinal) {
  Pool*& p = s.default_pools[ordinal];
  if (!p) {
    s.pools.push_back(Pool{});
    p = &s.pools.back();
    p->device = ordinal;
  }
  return p;
}
Pool* find_pool(State& s, void* handle) {
  for (Pool& p : s.pools)
    if (&p == handle && !p.destroyed) return &p;
  return nullptr;
}
Pool* current_pool(State& s, int ordinal) {
  const auto it = s.current_pools.find(ordinal);
  return it != s.current_pools.end() ? it->second : default_pool(s, ordinal);
}
// An allocation from `pool`, on the pool's device. The caller holds the lock.
// Gives back to the device what the pool keeps and no stream still has
// ahead of it, until it keeps no more than `keep` bytes reserved. The caller
// holds s.mutex.
void release_kept(State& s, Pool* pool, uint64_t keep) {
  for (size_t i = 0; i < pool->kept.size() && pool->reserved() > keep;) {
    const Pool::Kept& k = pool->kept[i];
    if (!k.queue->done(k.seq)) {
      ++i;
      continue;
    }
    try {
      s.rt->device(pool->device).memory().free(k.va);
    } catch (const std::exception&) {
    }
    pool->kept.erase(pool->kept.begin() + static_cast<std::ptrdiff_t>(i));
  }
}
// At a synchronization, every pool keeps no more than its release threshold.
void release_to_thresholds(State& s) {   // the caller holds s.mutex
  for (Pool& p : s.pools)
    if (!p.destroyed) release_kept(s, &p, p.release_threshold);
}
// An allocation from `pool`, on the pool's device, in `stream`'s order: a
// block the pool keeps where the reuse rules allow, or new memory as far as
// the pool's maximum. The caller holds the lock.
hipError_t pool_alloc(State& s, Pool* pool, void** ptr, size_t size, hipStream_t stream) {
  if (!ptr) return hipErrorInvalidValue;
  if (!size) {
    *ptr = nullptr;
    return hipSuccess;
  }
  Order o;
  if (const hipError_t e = order_for(s, stream, &o); e != hipSuccess) return e;
  const hipStream_t r = resolve(s, stream);
  const Stream* st = r ? &s.streams.at(r) : nullptr;
  for (size_t i = 0; i < pool->kept.size(); ++i) {
    const Pool::Kept& k = pool->kept[i];
    if (k.size != size) continue;
    bool reusable = k.queue == o.queue;   // freed earlier in this stream
    if (!reusable && pool->reuse[1] && st)
      if (const auto w = st->waited.find(k.queue.get()); w != st->waited.end() && w->second >= k.seq) reusable = true;
    if (!reusable && pool->reuse[2] && k.queue->done(k.seq)) reusable = true;
    if (!reusable) continue;
    *ptr = reinterpret_cast<void*>(k.va);
    pool->kept.erase(pool->kept.begin() + static_cast<std::ptrdiff_t>(i));
    s.pool_allocations[reinterpret_cast<uint64_t>(*ptr)] = {pool, size};
    pool->used += size;
    pool->used_high = std::max(pool->used_high, pool->used);
    return hipSuccess;
  }
  // ROCm's HIP holds a pool's maximum against the most it has reserved,
  // which trimming does not bring down (hip-tests' hipGetProcAddress pool
  // checks): new memory past it is refused.
  if (pool->max_size && std::max(pool->reserved_high, pool->reserved()) + size > pool->max_size)
    return fail(hipErrorOutOfMemory, "the pool is at its maximum size");
  try {
    *ptr = reinterpret_cast<void*>(s.rt->device(pool->device).memory().alloc(size));
  } catch (const std::exception& e) {
    return fail(hipErrorOutOfMemory, e.what());
  }
  s.pool_allocations[reinterpret_cast<uint64_t>(*ptr)] = {pool, size};
  pool->used += size;
  pool->used_high = std::max(pool->used_high, pool->used);
  pool->reserved_high = std::max(pool->reserved_high, pool->reserved());
  return hipSuccess;
}

}  // namespace

extern "C" {

hipError_t hipMallocFromPoolAsync(void** ptr, size_t size, void* pool, hipStream_t stream) {
  const ApiCall api("hipMallocFromPoolAsync");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  Pool* p = find_pool(s, pool);
  if (!p) return record(s, hipErrorInvalidValue);
  // In a capture, an allocation node: its address is handed back now.
  if (const hipStream_t r = resolve(s, stream); r && stream_capturing(s, r))
    return record(s, capture_alloc(s, r, p->device, size, ptr));
  return record(s, pool_alloc(s, p, ptr, size, stream));
}
hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream) {
  const ApiCall api("hipMallocAsync");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!device(s)) return record(s, hipErrorInvalidDevice);
  if (const hipStream_t r = resolve(s, stream); r && stream_capturing(s, r))
    return record(s, capture_alloc(s, r, s.current, size, ptr));
  return record(s, pool_alloc(s, current_pool(s, s.current), ptr, size, stream));
}
// Freed in the stream's order, once the work before it on the stream is done.
// Memory from a pool goes back to the pool now, marked with where in the
// stream its free falls (see Pool).
hipError_t hipFreeAsync(void* ptr, hipStream_t stream) {
  const ApiCall api("hipFreeAsync");
  if (!ptr) return record(state(), hipErrorInvalidValue);
  {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    // In a capture, a node that frees it when the graph runs.
    if (const hipStream_t r = resolve(s, stream); r && stream_capturing(s, r)) return record(s, capture_free(s, r, ptr));
    // Memory a pool keeps was freed already.
    for (const Pool& p : s.pools)
      for (const Pool::Kept& k : p.kept)
        if (k.va == reinterpret_cast<uint64_t>(ptr)) return record(s, hipErrorInvalidValue);
    const auto pa = s.pool_allocations.find(reinterpret_cast<uint64_t>(ptr));
    if (pa != s.pool_allocations.end() && !pa->second.first->destroyed) {
      Order o;
      if (const hipError_t e = order_for(s, stream, &o); e != hipSuccess) return record(s, e);
      Pool* p = pa->second.first;
      const uint64_t size = pa->second.second;
      const uint64_t seq = o.queue->submit([] { return hipSuccess; }, std::move(o.after));
      p->kept.push_back({pa->first, size, o.queue, seq});
      p->used -= size;
      s.pool_allocations.erase(pa);
      return record(s, hipSuccess);
    }
    // Anything else is the start of an allocation that is still there.
    const uint64_t va = reinterpret_cast<uint64_t>(ptr);
    bool live = false;
    {
      std::lock_guard<std::mutex> host_lock(g_host_mutex);
      live = g_host_allocations.count(va) || g_managed.count(va);
    }
    uint64_t b = 0, n = 0;
    // A graph's allocation may be freed in stream order before the launch that maps it has run.
    if (!live) live = graph_memory_range(va, &b, &n) && b == va;
    if (!live && s.rt)
      for (int d = 0; d < s.rt->device_count() && !live; ++d)
        live = s.rt->device(d).memory().find_allocation(va, &b, &n) && b == va;
    if (!live) return record(s, hipErrorInvalidValue);
  }
  return record(state(), in_order(stream, [ptr] { return free_now(ptr); }));
}

hipError_t hipDeviceGetDefaultMemPool(void** pool, int ordinal) {
  const ApiCall api("hipDeviceGetDefaultMemPool");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!pool) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (ordinal < 0 || ordinal >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  *pool = default_pool(s, ordinal);
  return record(s, hipSuccess);
}
// What a pool keeps goes back to the device until it keeps `keep` bytes or
// fewer -- all but what a stream still has ahead of it.
hipError_t hipMemPoolTrimTo(void* pool, size_t keep) {
  const ApiCall api("hipMemPoolTrimTo");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  Pool* p = find_pool(s, pool);
  if (!p) return record(s, hipErrorInvalidValue);
  release_kept(s, p, keep);
  return record(s, hipSuccess);
}
hipError_t hipMemPoolCreate(void** pool, const vgpu::amd::abi::MemPoolProps* props) {
  const ApiCall api("hipMemPoolCreate");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!pool || !props) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  // Pinned device memory, on a device there is.
  if (props->allocType != 1 || props->location.type != 1 || props->location.id < 0 ||
      props->location.id >= s.rt->device_count())
    return record(s, hipErrorInvalidValue);
  s.pools.push_back(Pool{});
  s.pools.back().device = props->location.id;
  s.pools.back().max_size = props->maxSize;
  s.pools.back().handle_types = props->handleTypes;
  *pool = &s.pools.back();
  return record(s, hipSuccess);
}
// A device's default pool is not the program's to destroy.
hipError_t hipMemPoolDestroy(void* pool) {
  const ApiCall api("hipMemPoolDestroy");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  Pool* p = find_pool(s, pool);
  if (!p) return record(s, hipErrorInvalidValue);
  for (const auto& d : s.default_pools)
    if (d.second == p) return record(s, hipErrorInvalidValue);
  release_kept(s, p, 0);
  p->destroyed = true;
  if (p->share_fd >= 0) ::close(p->share_fd);
  p->share_fd = -1;
  // A device whose current pool it was goes back to its default.
  for (auto it = s.current_pools.begin(); it != s.current_pools.end();)
    it = it->second == p ? s.current_pools.erase(it) : std::next(it);
  return record(s, hipSuccess);
}
// The reuse flags are ints, the rest 64-bit counts. Of the counts, only the
// release threshold and the two high-water marks may be set, and the marks
// only back to zero, where they start again from what is in use.
hipError_t hipMemPoolGetAttribute(void* pool, int attr, void* value) {
  const ApiCall api("hipMemPoolGetAttribute");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  Pool* p = find_pool(s, pool);
  if (!p || !value) return record(s, hipErrorInvalidValue);
  using A = vgpu::amd::abi::MemPoolAttr;
  switch (attr) {
    case A::kPoolReuseFollowEventDependencies:
    case A::kPoolReuseAllowOpportunistic:
    case A::kPoolReuseAllowInternalDependencies: *static_cast<int*>(value) = p->reuse[attr]; break;
    case A::kPoolReleaseThreshold: *static_cast<uint64_t*>(value) = p->release_threshold; break;
    case A::kPoolReservedMemCurrent: *static_cast<uint64_t*>(value) = p->reserved(); break;
    case A::kPoolUsedMemCurrent: *static_cast<uint64_t*>(value) = p->used; break;
    case A::kPoolReservedMemHigh: *static_cast<uint64_t*>(value) = p->reserved_high; break;
    case A::kPoolUsedMemHigh: *static_cast<uint64_t*>(value) = p->used_high; break;
    default: return record(s, hipErrorInvalidValue);
  }
  return record(s, hipSuccess);
}
hipError_t hipMemPoolSetAttribute(void* pool, int attr, void* value) {
  const ApiCall api("hipMemPoolSetAttribute");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  Pool* p = find_pool(s, pool);
  if (!p || !value) return record(s, hipErrorInvalidValue);
  using A = vgpu::amd::abi::MemPoolAttr;
  switch (attr) {
    case A::kPoolReuseFollowEventDependencies:
    case A::kPoolReuseAllowOpportunistic:
    case A::kPoolReuseAllowInternalDependencies: p->reuse[attr] = *static_cast<int*>(value) ? 1 : 0; break;
    case A::kPoolReleaseThreshold: p->release_threshold = *static_cast<uint64_t*>(value); break;
    // Set back to zero, not to what is in use now, as ROCm's HIP does.
    case A::kPoolReservedMemHigh:
      if (*static_cast<uint64_t*>(value)) return record(s, hipErrorInvalidValue);
      p->reserved_high = 0;
      break;
    case A::kPoolUsedMemHigh:
      if (*static_cast<uint64_t*>(value)) return record(s, hipErrorInvalidValue);
      p->used_high = 0;
      break;
    default: return record(s, hipErrorInvalidValue);
  }
  return record(s, hipSuccess);
}
// Every device reaches every pool's memory already; what each was granted is
// kept for hipMemPoolGetAccess. A pool's own device cannot lose its access.
hipError_t hipMemPoolSetAccess(void* pool, const vgpu::amd::abi::MemAccessDesc* desc, size_t count) {
  const ApiCall api("hipMemPoolSetAccess");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  Pool* p = find_pool(s, pool);
  if (!p || (count && !desc)) return record(s, hipErrorInvalidValue);
  if (count > static_cast<size_t>(s.rt->device_count())) return record(s, hipErrorInvalidDevice);
  for (size_t i = 0; i < count; ++i) {
    const auto& d = desc[i];
    if (d.location.type != 1) return record(s, hipErrorInvalidValue);
    if (d.location.id < 0 || d.location.id >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
    if (d.flags != 0 && d.flags != 1 && d.flags != 3) return record(s, hipErrorInvalidValue);
    if (d.location.id == p->device && d.flags != 3) return record(s, hipErrorInvalidDevice);
  }
  // A device granted access has its kernels reach the pool's device.
  for (size_t i = 0; i < count; ++i)
    if (desc[i].location.id != p->device) {
      p->access[desc[i].location.id] = desc[i].flags;
      if (desc[i].flags) s.peers.insert({desc[i].location.id, p->device});
    }
  publish_pool_access(*p);
  return record(s, hipSuccess);
}

// A copy of height rows, each width bytes, from one pitched buffer to another.
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                       hipMemcpyKind kind) {
  const ApiCall api("hipMemcpy2D");
  return record(state(), copy_in_order(dst, src, width, kind, nullptr, false, height, dpitch, spitch));
}
hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                            hipMemcpyKind kind, hipStream_t stream) {
  const ApiCall api("hipMemcpy2DAsync");
  return record(state(), copy_in_order(dst, src, width, kind, stream, true, height, dpitch, spitch));
}

// What an address is: host memory the runtime pinned, registered or manages,
// a device's memory, or host memory the runtime knows nothing of -- which,
// as in HIP since 6.0 and CUDA since 11, is an answer rather than an error.
// The host kinds are asked first: they are mapped into every device too.
hipError_t hipPointerGetAttributes(vgpu::amd::abi::PointerAttribute* out, const void* ptr) {
  const ApiCall api("hipPointerGetAttributes");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!out || !ptr) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  *out = {};
  out->device = -1;
  const uint64_t va = reinterpret_cast<uint64_t>(ptr);
  {
    std::lock_guard<std::mutex> host_lock(g_host_mutex);
    const bool managed = find_range(g_managed, va) != g_managed.end();
    if (managed || find_range(g_host_allocations, va) != g_host_allocations.end() ||
        find_range(g_host_registered, va) != g_host_registered.end()) {
      // Reached by kernels at the host's own address.
      out->type = managed ? vgpu::amd::abi::kMemoryManaged : vgpu::amd::abi::kMemoryHost;
      out->isManaged = managed ? 1 : 0;
      out->device = s.current;
      out->devicePointer = out->hostPointer = const_cast<void*>(ptr);
      return record(s, hipSuccess);
    }
  }
  // A device's memory while it is allocated or mapped; an address in its
  // range that was freed, or never given out, is no one's.
  for (int i = 0; i < s.rt->device_count(); ++i) {
    const vgpu::MemoryManager& m = s.rt->device(i).memory();
    bool readable = false, writable = false;
    if (m.owns(va) && (m.find_allocation(va, nullptr, nullptr) || m.access_at(va, &readable, &writable))) {
      out->type = vgpu::amd::abi::kMemoryDevice;
      out->device = i;
      out->devicePointer = const_cast<void*>(ptr);
      return record(s, hipSuccess);
    }
  }
  out->type = vgpu::amd::abi::kMemoryUnregistered;
  return record(s, hipSuccess);
}

// Nothing is ever left for a stream to do.
// Whether the stream has work still to finish: hipErrorNotReady while it has.
hipError_t hipStreamQuery(hipStream_t stream) {
  const ApiCall api("hipStreamQuery");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  std::vector<std::shared_ptr<Queue>> qs;
  stream = resolve(s, stream);
  if (stream && stream_capturing(s, stream)) return record(s, hipErrorStreamCaptureUnsupported);
  if (const hipError_t e = unsafe_during_capture(s); e != hipSuccess) return record(s, e);
  if (!stream) {
    qs = queues_of(s, s.current, true);
  } else {
    const auto it = s.streams.find(stream);
    if (it == s.streams.end()) return record(s, hipErrorInvalidHandle);
    if (it->second.queue) qs.push_back(it->second.queue);
  }
  for (const auto& q : qs)
    if (!q->idle()) return record(s, hipErrorNotReady);
  return record(s, hipSuccess);
}

// A HIP function by name, as a program that binds HIP at run time asks for
// it (Triton's HIP driver does, for every call it makes): the library's own
// export of that name, found the way the dynamic loader would find it.
namespace {
void* own_function(const char* symbol) {
  static void* self = [] {
    Dl_info info{};
    dladdr(reinterpret_cast<void*>(&hipGetLastError), &info);
    return info.dli_fname ? dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD) : nullptr;
  }();
  return self && symbol ? dlsym(self, symbol) : nullptr;
}
}  // namespace

// A caller built for HIP 6 or later (the version it passes, in either of
// HIP's forms: 600, or the driver's 60000000) means the functions HIP 6
// renamed to take its struct layouts -- hipGetDeviceProperties is
// hipGetDevicePropertiesR0600 to it, as HIP's headers make it.
hipError_t hipGetProcAddress(const char* symbol, void** pfn, int hip_version, uint64_t, int* status) {
  const ApiCall api("hipGetProcAddress");
  if (!symbol || !pfn) return record(state(), hipErrorInvalidValue);
  *pfn = nullptr;
  if (hip_version >= 600) *pfn = own_function((std::string(symbol) + "R0600").c_str());
  if (!*pfn) *pfn = own_function(symbol);
  if (status) *status = *pfn ? 0 /* SUCCESS */ : 1 /* SYMBOL_NOT_FOUND */;
  return record(state(), *pfn ? hipSuccess : hipErrorNotFound);
}

hipError_t hipGetDriverEntryPoint(const char* symbol, void** pfn, unsigned long long flags, int* status) {
  const ApiCall api("hipGetDriverEntryPoint");
  // hipEnableDefault, hipEnableLegacyStream and hipEnablePerThreadDefaultStream
  // are the flags there are.
  if (!symbol || !*symbol || !pfn || flags > 2) return record(state(), hipErrorInvalidValue);
  *pfn = own_function(symbol);
  if (status) *status = *pfn ? 0 /* SUCCESS */ : 1 /* SYMBOL_NOT_FOUND */;
  return record(state(), *pfn ? hipSuccess : hipErrorNotFound);
}

// One attribute of a module's kernel (hipFunction_attribute), which Triton
// reads for every kernel it compiles: its register and scratch use, and the
// work-group size its metadata allows.
hipError_t hipFuncGetAttribute(int* value, int attribute, hipFunction_t f) {
  const ApiCall api("hipFuncGetAttribute");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!value) return record(s, hipErrorInvalidValue);
  if (!f) return record(s, hipErrorInvalidResourceHandle);
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
  const Kernel& k = *reinterpret_cast<Function*>(f)->kernel;
  const vgpu::DeviceProfile& p = d->profile();
  int major = 0, minor = 0;
  hip_version(p, &major, &minor);
  switch (attribute) {
    case 0: *value = static_cast<int>(k.max_flat_workgroup_size ? k.max_flat_workgroup_size : 1024); break;
    case 1: *value = static_cast<int>(k.group_segment); break;                          // SHARED_SIZE_BYTES
    case 2: *value = 0; break;                                                          // CONST_SIZE_BYTES
    case 3: *value = static_cast<int>(k.private_segment); break;                        // LOCAL_SIZE_BYTES
    case 4: *value = static_cast<int>(k.vgpr_count); break;                             // NUM_REGS
    case 5:                                                                             // PTX_VERSION
    case 6: *value = major * 10 + minor; break;                                         // BINARY_VERSION
    case 7: *value = 0; break;                                                          // CACHE_MODE_CA
    case 8: *value = static_cast<int>(p.limits.shared_mem_per_block - k.group_segment); break;
    case 9: *value = -1; break;                                                         // CARVEOUT: none preferred
    default: return record(s, hipErrorInvalidValue);
  }
  return record(s, hipSuccess);
}

hipError_t hipModuleLaunchCooperativeKernel(hipFunction_t f, unsigned int gx, unsigned int gy, unsigned int gz,
                                            unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared,
                                            hipStream_t stream, void** params);

// A module launch described by a configuration and a list of attributes
// (HIP_LAUNCH_CONFIG): of those, a cooperative launch is what changes how it
// runs.
hipError_t hipDrvLaunchKernelEx(const void* config, hipFunction_t f, void** params, void** extra) {
  const ApiCall api("hipDrvLaunchKernelEx");
  if (!config || !f) return record(state(), hipErrorInvalidValue);
  const auto* c = static_cast<const unsigned char*>(config);
  uint32_t dims[7];
  std::memcpy(dims, c, sizeof dims);   // grid x, y, z; block x, y, z; dynamic LDS
  hipStream_t stream;
  std::memcpy(&stream, c + 32, sizeof stream);
  const unsigned char* attrs;
  std::memcpy(&attrs, c + 40, sizeof attrs);
  uint32_t count;
  std::memcpy(&count, c + 48, 4);
  bool cooperative = false;
  for (uint32_t i = 0; attrs && i < count; ++i) {   // hipLaunchAttribute: an id, then its value at 8; 72 bytes
    uint32_t id;
    int v;
    std::memcpy(&id, attrs + 72 * i, 4);
    std::memcpy(&v, attrs + 72 * i + 8, 4);
    if (id == 2 /* hipLaunchAttributeCooperative */) cooperative = v != 0;
  }
  if (cooperative)
    return hipModuleLaunchCooperativeKernel(f, dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], stream,
                                            params);
  return hipModuleLaunchKernel(f, dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], stream, params,
                               extra);
}

hipError_t hipExtGetLastError(void) {
  const ApiCall api("hipExtGetLastError");
  return hipGetLastError();
}

// A module launch sized in work-items rather than work-groups, as HSA sizes a
// dispatch, with events recorded either side of it. A grid that is not a
// whole number of work-groups runs its last ones short (vgpu/amd_exec.hpp),
// as MIOpen's kernels on gfx950 ask for.
hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t lx, uint32_t ly,
                                    uint32_t lz, size_t shared, hipStream_t stream, void** params, void** extra,
                                    hipEvent_t start, hipEvent_t stop, uint32_t) {
  const ApiCall api("hipExtModuleLaunchKernel");
  if (!lx || !ly || !lz || !gx || !gy || !gz) return record(state(), hipErrorInvalidConfiguration);
  if (start)
    if (const hipError_t e = hipEventRecord(start, stream); e != hipSuccess) return e;
  const uint32_t items[3] = {gx % lx ? gx : 0, gy % ly ? gy : 0, gz % lz ? gz : 0};
  if (const hipError_t e = module_launch(f, (gx + lx - 1) / lx, (gy + ly - 1) / ly, (gz + lz - 1) / lz, lx, ly, lz,
                                         static_cast<unsigned>(shared), stream, params, extra, items);
      e != hipSuccess)
    return e;
  return stop ? hipEventRecord(stop, stream) : hipSuccess;
}
// The same, under the name HCC gave it.
hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t lx, uint32_t ly,
                                    uint32_t lz, size_t shared, hipStream_t stream, void** params, void** extra,
                                    hipEvent_t start, hipEvent_t stop) {
  const ApiCall api("hipHccModuleLaunchKernel");
  return hipExtModuleLaunchKernel(f, gx, gy, gz, lx, ly, lz, shared, stream, params, extra, start, stop, 0);
}

}  // extern "C"

// ---- What PyTorch's ROCm build calls --------------------------------------
//
// PyTorch links every ROCm library it ships (MIOpen, RCCL, hipBLASLt,
// rocSOLVER, ...) and each asks for its own part of HIP, so these are the
// calls its libraries bind to when they load. Most are what HIP's
// documentation says of them over a runtime where each call has finished by
// the time it returns; the few that need something this does not model are
// refused by name.
namespace {

// A call this does not implement, and why, said once.
hipError_t refused(const char* name, const char* why) {
  return record(state(), fail(hipErrorNotSupported, std::string(name) + " is not supported: " + why));
}

// The device whose memory holds `va`, or -1.
int owner_of(State& s, uint64_t va) {
  for (int i = 0; i < s.rt->device_count(); ++i)
    if (s.rt->device(i).memory().owns(va)) return i;
  return -1;
}

// Physical memory a program made with hipMemCreate (or imported): host
// memory the simulator keeps -- a file, where it may be shared with another
// process -- which a mapping puts in place at an address a reservation set
// aside, on any device. Its address is the handle HIP hands out.
struct VmmAllocation {
  int device;            // where it was made: a device, or -1 for the host
  uint64_t handle;       // the device's count of what it holds (MemoryManager::create_handle), or 0
  size_t size;
  int type;              // hipMemAllocationTypePinned or Uncached, as made
  int refs;              // hipMemCreate's, and one for each hipMemRetainAllocationHandle
  int fd = -1;           // its file, where it is one
  uint8_t* host = nullptr;
  uint64_t bytes = 0;    // how much is mapped here: size, in whole granules
  int maps = 0;          // its mappings still there
  int handle_types = 0;  // hipMemAllocationHandleType it was made to be exported as
};
std::deque<VmmAllocation> g_vmm;   // under State's mutex; entries are never moved
std::set<VmmAllocation*> g_vmm_live;
// Where each allocation is mapped: start -> (size, allocation), and the
// offset into it each mapping starts at.
std::map<uint64_t, std::pair<size_t, VmmAllocation*>> g_vmm_maps;
std::map<uint64_t, uint64_t> g_vmm_offsets;
// The mapping an address is in, if any.
std::map<uint64_t, std::pair<size_t, VmmAllocation*>>::iterator vmm_mapping(uint64_t va) {
  auto it = g_vmm_maps.upper_bound(va);
  if (it == g_vmm_maps.begin()) return g_vmm_maps.end();
  --it;
  return va < it->first + it->second.first ? it : g_vmm_maps.end();
}
// The mapped run an address is in: its mapping, and those placed end to end
// after it.
bool vmm_range(uint64_t va, uint64_t* base, uint64_t* size) {
  auto it = vmm_mapping(va);
  if (it == g_vmm_maps.end()) return false;
  *base = it->first;
  uint64_t end = it->first + it->second.first;
  for (++it; it != g_vmm_maps.end() && it->first == end; ++it) end += it->second.first;
  *size = end - *base;
  return true;
}
// Host-located memory the CPU was given access to: mapped at its device
// address in this process too, by address, with its length.
std::map<uint64_t, uint64_t> g_vmm_cpu_maps;
// Device allocations handed out as files (hipMemGetHandleForAddressRange):
// each one's file, by where it starts.
std::map<uint64_t, std::pair<std::string, uint64_t>> g_dmabuf_paths;   // start -> (file, size)

// An IPC memory handle's payload: which process shared which of its
// allocations, and the file the bytes now live in.
struct IpcPayload {
  uint32_t magic, version, device, pid;
  uint64_t size;
  char id[40];
};
static_assert(sizeof(IpcPayload) <= sizeof(vgpu::amd::abi::IpcMemHandle), "an IPC payload fits HIP's handle");
constexpr uint32_t kIpcMagic = 0x48495043;   // "HIPC"
std::string ipc_path(const char* id) { return vgpu::telemetry::default_path() + "/ipc-hip-" + id; }
std::map<void*, int> g_ipc_open;   // pointer -> the device it was mapped on
static_assert(sizeof(IpcPayload) <= 64, "an IPC payload fits hipMemPoolPtrExportData");
std::map<uint64_t, IpcPayload> g_pool_exports;   // pool allocation -> how it was exported


}  // namespace

extern "C" {

// ---- Devices and contexts ----------------------------------------------------

// A context is a device's, and each device has one, its primary context:
// the handle stands for the device (g_contexts). Each thread has a stack of
// them, as ROCm's HIP keeps one: creating a context pushes it and gives its
// device the context's flags, without making the device current; pushing one
// makes its device current; popping takes it off and leaves the current
// device as it was. The current context is always the current device's.
// What ROCm's HIP does not do with contexts -- cache and shared-memory
// settings of their own, flags, an API version, synchronizing one -- it
// answers as not supported, and so does this.
}  // extern "C"
namespace {
thread_local std::vector<int> t_contexts;
// The device a context handle is for, or -1. The caller holds s.mutex.
int context_device(State& s, const void* ctx) {
  const auto* c = static_cast<const char*>(ctx);
  if (c < g_contexts || c >= g_contexts + 64 || ensure_runtime(s) != hipSuccess) return -1;
  const int d = static_cast<int>(c - g_contexts);
  return d < s.rt->device_count() ? d : -1;
}
bool valid_device(State& s, int d) { return ensure_runtime(s) == hipSuccess && d >= 0 && d < s.rt->device_count(); }
}  // namespace
extern "C" {

hipError_t hipCtxGetCurrent(void** ctx) {
  const ApiCall api("hipCtxGetCurrent");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ctx) return record(s, hipErrorInvalidValue);
  *ctx = s.current < 64 ? &g_contexts[s.current] : nullptr;
  return record(s, hipSuccess);
}
// Replaces the context on top (a null one pops it), and makes its device
// current, as hipSetDevice does.
hipError_t hipCtxSetCurrent(void* ctx) {
  const ApiCall api("hipCtxSetCurrent");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ctx) {
    if (!t_contexts.empty()) t_contexts.pop_back();
    return record(s, hipSuccess);
  }
  const int d = context_device(s, ctx);
  if (d < 0) return record(s, hipErrorInvalidContext);
  if (t_contexts.empty()) t_contexts.push_back(d);
  else t_contexts.back() = d;
  s.current = d;
  return record(s, hipSuccess);
}
// A new context is the device's primary one, pushed; its flags become the
// device's, all of them (hipGetDeviceFlags gives them back there).
hipError_t hipCtxCreate(void** ctx, unsigned int flags, int ordinal) {
  const ApiCall api("hipCtxCreate");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ctx) return record(s, hipErrorInvalidValue);
  if (!valid_device(s, ordinal) || ordinal >= 64) return record(s, hipErrorInvalidValue);
  s.device_flags[ordinal] = flags;
  t_contexts.push_back(ordinal);
  *ctx = &g_contexts[ordinal];
  return record(s, hipSuccess);
}
// The primary context lives on; destroying one takes it off this thread's
// stack if it is on top.
hipError_t hipCtxDestroy(void* ctx) {
  const ApiCall api("hipCtxDestroy");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ctx) return record(s, hipErrorInvalidValue);
  const int d = context_device(s, ctx);
  if (d < 0) return record(s, hipErrorInvalidContext);
  if (!t_contexts.empty() && t_contexts.back() == d) t_contexts.pop_back();
  return record(s, hipSuccess);
}
hipError_t hipCtxPushCurrent(void* ctx) {
  const ApiCall api("hipCtxPushCurrent");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  const int d = context_device(s, ctx);
  if (d < 0) return record(s, hipErrorInvalidContext);
  t_contexts.push_back(d);
  s.current = d;
  return record(s, hipSuccess);
}
hipError_t hipCtxPopCurrent(void** ctx) {
  const ApiCall api("hipCtxPopCurrent");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (t_contexts.empty()) return record(s, hipErrorInvalidContext);
  const int d = t_contexts.back();
  t_contexts.pop_back();
  if (ctx) *ctx = &g_contexts[d];
  return record(s, hipSuccess);
}
hipError_t hipCtxGetDevice(int* ordinal) {
  const ApiCall api("hipCtxGetDevice");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ordinal) return record(s, hipErrorInvalidValue);
  *ordinal = s.current;
  return record(s, hipSuccess);
}
hipError_t hipCtxGetApiVersion(void*, unsigned int*) {
  const ApiCall api("hipCtxGetApiVersion");
  return record(state(), hipErrorNotSupported);
}
hipError_t hipCtxGetCacheConfig(int*) {
  const ApiCall api("hipCtxGetCacheConfig");
  return record(state(), hipErrorNotSupported);
}
hipError_t hipCtxSetCacheConfig(int config) {
  const ApiCall api("hipCtxSetCacheConfig");
  return record(state(), config < 0 || config > 3 ? hipErrorInvalidValue : hipErrorNotSupported);
}
// LDS banks are four bytes wide, and that is all there is to say or set.
hipError_t hipCtxGetSharedMemConfig(int* config) {
  const ApiCall api("hipCtxGetSharedMemConfig");
  if (!config) return record(state(), hipErrorInvalidValue);
  *config = 1;   // hipSharedMemBankSizeFourByte
  return record(state(), hipSuccess);
}
hipError_t hipCtxSetSharedMemConfig(int) {
  const ApiCall api("hipCtxSetSharedMemConfig");
  return record(state(), hipErrorNotSupported);
}
hipError_t hipCtxSynchronize(void) {
  const ApiCall api("hipCtxSynchronize");
  return record(state(), hipErrorNotSupported);
}
hipError_t hipCtxGetFlags(unsigned int*) {
  const ApiCall api("hipCtxGetFlags");
  return record(state(), hipErrorNotSupported);
}
// Every device already reaches every other here (hipDeviceEnablePeerAccess
// is what a program uses); these change nothing and succeed.
hipError_t hipCtxEnablePeerAccess(void*, unsigned int) {
  const ApiCall api("hipCtxEnablePeerAccess");
  return record(state(), hipSuccess);
}
hipError_t hipCtxDisablePeerAccess(void*) {
  const ApiCall api("hipCtxDisablePeerAccess");
  return record(state(), hipSuccess);
}
// The primary context is always active: releasing and resetting it change
// nothing, and its flags cannot be changed while it is in use.
hipError_t hipDevicePrimaryCtxRelease(int ordinal) {
  const ApiCall api("hipDevicePrimaryCtxRelease");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return record(s, valid_device(s, ordinal) ? hipSuccess : hipErrorInvalidDevice);
}
hipError_t hipDevicePrimaryCtxReset(int ordinal) {
  const ApiCall api("hipDevicePrimaryCtxReset");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return record(s, valid_device(s, ordinal) ? hipSuccess : hipErrorInvalidDevice);
}
hipError_t hipDevicePrimaryCtxSetFlags(int ordinal, unsigned int) {
  const ApiCall api("hipDevicePrimaryCtxSetFlags");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return record(s, valid_device(s, ordinal) ? hipErrorContextAlreadyInUse : hipErrorInvalidDevice);
}
hipError_t hipDevicePrimaryCtxRetain(void** ctx, int ordinal) {
  const ApiCall api("hipDevicePrimaryCtxRetain");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ctx) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (ordinal < 0 || ordinal >= s.rt->device_count() || ordinal >= 64) return record(s, hipErrorInvalidDevice);
  *ctx = &g_contexts[ordinal];
  return record(s, hipSuccess);
}
hipError_t hipDevicePrimaryCtxGetState(int ordinal, unsigned int* flags, int* active) {
  const ApiCall api("hipDevicePrimaryCtxGetState");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!flags || !active) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (ordinal < 0 || ordinal >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  *flags = 0;
  *active = 1;
  return record(s, hipSuccess);
}

// "domain:bus:device.function", as the properties give them.
hipError_t hipDeviceGetPCIBusId(char* bus_id, int len, int ordinal) {
  const ApiCall api("hipDeviceGetPCIBusId");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!bus_id || len <= 0) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (ordinal < 0 || ordinal >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  vgpu::amd::abi::DevicePropR0600 p;
  fill_properties(s.rt->device(ordinal).profile(), ordinal, &p);
  // As snprintf fills it: a buffer too short for the whole string gets what
  // fits, and the call fails.
  const int n = std::snprintf(bus_id, static_cast<size_t>(len), "%04x:%02x:%02x.0", p.pciDomainID, p.pciBusID,
                              p.pciDeviceID);
  return record(s, n >= len ? hipErrorInvalidValue : hipSuccess);
}
hipError_t hipDeviceGetByPCIBusId(int* ordinal, const char* bus_id) {
  const ApiCall api("hipDeviceGetByPCIBusId");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ordinal || !bus_id) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  unsigned domain = 0, bus = 0, dev = 0;
  if (std::sscanf(bus_id, "%x:%x:%x", &domain, &bus, &dev) != 3) return record(s, hipErrorInvalidValue);
  for (int i = 0; i < s.rt->device_count(); ++i) {
    vgpu::amd::abi::DevicePropR0600 p;
    fill_properties(s.rt->device(i).profile(), i, &p);
    if (unsigned(p.pciDomainID) == domain && unsigned(p.pciBusID) == bus && unsigned(p.pciDeviceID) == dev) {
      *ordinal = i;
      return record(s, hipSuccess);
    }
  }
  return record(s, hipErrorInvalidDevice);
}

// HIP's three stream priorities: high (-1), normal (0) and low (1), a lower
// number the higher priority.
hipError_t hipDeviceGetStreamPriorityRange(int* least, int* greatest) {
  const ApiCall api("hipDeviceGetStreamPriorityRange");
  if (least) *least = 1;
  if (greatest) *greatest = -1;
  return record(state(), hipSuccess);
}

// There is no cache to configure; the setting is accepted, as HIP accepts it
// on a device without one.
hipError_t hipDeviceSetCacheConfig(int) {
  const ApiCall api("hipDeviceSetCacheConfig");
  return record(state(), hipSuccess);
}
// The same, for one kernel: a preference between L1 and LDS the simulator's
// kernels do not have to choose between.
hipError_t hipFuncSetCacheConfig(const void* function, int config) {
  const ApiCall api("hipFuncSetCacheConfig");
  if (!function || config < 0 || config > 3) return record(state(), hipErrorInvalidValue);
  return record(state(), hipSuccess);
}

// The per-thread stack and the device heap: kept as set, and read back.
// Nothing here runs short of either. And on gfx94x and gfx950, AMD's scratch
// limits: the private memory a device's queues may have, in bytes, from 0 up
// to what HSA says there is, the current one the HSA runtime's too
// (shared::scratch_limit). ROCm's HIP takes a limit as an int: from
// hipLimitRange up it is no limit at all (hipErrorInvalidValue); below that,
// one it does not handle is unsupported.
namespace {
size_t g_limits[3] = {1024, 64 << 20, 8 << 20};   // stack, printf FIFO, heap
constexpr int kLimitScratchMin = 0x1000, kLimitScratchMax = 0x1001, kLimitScratchCurrent = 0x1002,
              kLimitRange = 0x1003;
constexpr size_t kScratchMax = vgpu::amd::shared::kScratchLimitMax;
size_t scratch_limit_locked(State& s, int ordinal) {   // the caller holds s.mutex
  const auto it = s.scratch_limit.find(ordinal);
  return it == s.scratch_limit.end() ? kScratchMax : it->second;
}
bool set_scratch_limit_locked(State& s, int ordinal, size_t bytes) {
  if (bytes > kScratchMax) return false;
  s.scratch_limit[ordinal] = bytes;
  return true;
}
bool has_scratch_limits(State& s) {   // the caller holds s.mutex
  const vgpu::runtime::Device* d = device(s);
  if (!d) return false;
  const std::string& arch = d->profile().gcn_arch;
  return arch.rfind("gfx94", 0) == 0 || arch.rfind("gfx95", 0) == 0;
}
}  // namespace
hipError_t hipDeviceSetLimit(int limit, size_t value) {
  const ApiCall api("hipDeviceSetLimit");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (limit >= kLimitRange) return record(s, hipErrorInvalidValue);
  if (limit == 0 || limit == 2) {   // hipLimitStackSize, hipLimitMallocHeapSize
    g_limits[limit] = value;
    return record(s, hipSuccess);
  }
  if (limit == kLimitScratchCurrent && has_scratch_limits(s))
    return record(s, set_scratch_limit_locked(s, s.current, value) ? hipSuccess : hipErrorInvalidValue);
  return record(s, hipErrorUnsupportedLimit);
}
hipError_t hipDeviceGetLimit(size_t* value, int limit) {
  const ApiCall api("hipDeviceGetLimit");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!value || limit >= kLimitRange) return record(s, hipErrorInvalidValue);
  if (limit >= 0 && limit <= 2) {
    *value = g_limits[limit];
    return record(s, hipSuccess);
  }
  if (limit >= kLimitScratchMin && limit <= kLimitScratchCurrent && has_scratch_limits(s)) {
    *value = limit == kLimitScratchMin   ? 0
             : limit == kLimitScratchMax ? kScratchMax
                                         : scratch_limit_locked(s, s.current);
    return record(s, hipSuccess);
  }
  return record(s, hipErrorUnsupportedLimit);
}

// The driver API's forms: a value HIP does not define is refused rather than
// named "unknown".
hipError_t hipDrvGetErrorString(hipError_t error, const char** text) {
  const ApiCall api("hipDrvGetErrorString");
  // hipErrorTbd has a name and, for the driver, no text.
  if (!text || !error_text(error) || static_cast<int>(error) == 1054) return record(state(), hipErrorInvalidValue);
  *text = error_text(error)->text;
  return hipSuccess;
}
hipError_t hipDrvGetErrorName(hipError_t error, const char** name) {
  const ApiCall api("hipDrvGetErrorName");
  if (!name || !error_text(error)) return record(state(), hipErrorInvalidValue);
  *name = error_text(error)->name;
  return hipSuccess;
}

// ---- Kernels -------------------------------------------------------------------

// What the code object says of a kernel: its LDS, its scratch, its registers,
// and the largest work-group it was built for.
hipError_t hipFuncGetAttributes(vgpu::amd::abi::FuncAttributes* attr, const void* host_function) {
  const ApiCall api("hipFuncGetAttributes");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!attr) return record(s, hipErrorInvalidValue);
  if (!device(s)) return record(s, hipErrorInvalidDevice);
  const Kernel* k = nullptr;
  if (const hipError_t e = kernel_for(s, host_function, &k); e != hipSuccess) return record(s, e);
  const vgpu::DeviceProfile& p = s.rt->device(s.current).profile();
  *attr = {};
  attr->binaryVersion = attr->ptxVersion = p.cc_major * 10 + p.cc_minor;
  attr->localSizeBytes = k->private_segment;
  attr->sharedSizeBytes = k->group_segment;
  attr->maxDynamicSharedSizeBytes = static_cast<int>(p.limits.shared_mem_per_block - k->group_segment);
  attr->maxThreadsPerBlock = static_cast<int>(k->max_flat_workgroup_size ? k->max_flat_workgroup_size : 1024);
  attr->numRegs = static_cast<int>(k->vgpr_count);
  return record(s, hipSuccess);
}
// The dynamic LDS a kernel may ask for (8) and the LDS carve-out it prefers
// (9): accepted within what the device has, since every launch here gets
// the LDS it asks for.
hipError_t hipFuncSetAttribute(const void* host_function, int attr, int value) {
  const ApiCall api("hipFuncSetAttribute");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!device(s)) return record(s, hipErrorInvalidDevice);
  const Kernel* k = nullptr;
  if (const hipError_t e = kernel_for(s, host_function, &k); e != hipSuccess) return record(s, e);
  const vgpu::DeviceProfile& p = s.rt->device(s.current).profile();
  if (attr == 8 && (value < 0 || uint64_t(value) + k->group_segment > p.limits.shared_mem_per_block))
    return record(s, hipErrorInvalidValue);
  if (attr == 9 && (value < -1 || value > 100)) return record(s, hipErrorInvalidValue);
  // hipFuncAttributeMax (10) is taken too, as ROCm's HIP takes it.
  if (attr != 8 && attr != 9 && attr != 10) return record(s, hipErrorInvalidValue);
  return record(s, hipSuccess);
}

const char* hipKernelNameRef(const hipFunction_t f) {
  return f ? reinterpret_cast<const Function*>(f)->kernel->name.c_str() : nullptr;
}
const char* hipKernelNameRefByPtr(const void* host_function, hipStream_t) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  const auto it = s.host_functions.find(host_function);
  return it == s.host_functions.end() ? nullptr : it->second.kernel.c_str();
}

// hipLaunchKernel with an event recorded either side of it.
hipError_t hipExtLaunchKernel(const void* host_function, vgpu::amd::abi::Dim3 grid, vgpu::amd::abi::Dim3 block,
                              void** args, size_t shared, hipStream_t stream, hipEvent_t start, hipEvent_t stop, int) {
  const ApiCall api("hipExtLaunchKernel");
  // Events that are not there are the wrong value here, before anything runs.
  {
    std::lock_guard<std::mutex> lock(g_event_mutex);
    if ((start && !find_event(start)) || (stop && !find_event(stop))) return record(state(), hipErrorInvalidValue);
  }
  if (start)
    if (const hipError_t e = hipEventRecord(start, stream); e != hipSuccess) return e;
  if (const hipError_t e = hipLaunchKernel(host_function, grid, block, args, shared, stream); e != hipSuccess) return e;
  return stop ? hipEventRecord(stop, stream) : hipSuccess;
}

// A module's kernel launched so its work-groups may wait on one another: the
// grid has to fit on the device at once, as with hipLaunchCooperativeKernel.
hipError_t hipModuleLaunchCooperativeKernel(hipFunction_t f, unsigned int gx, unsigned int gy, unsigned int gz,
                                            unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared,
                                            hipStream_t stream, void** params) {
  const ApiCall api("hipModuleLaunchCooperativeKernel");
  State& s = state();
  std::unique_lock<std::mutex> lock(s.mutex);
  if (!f) return record(s, hipErrorInvalidValue);
  if (!device(s)) return record(s, hipErrorInvalidDevice);
  Function* fn = reinterpret_cast<Function*>(f);
  if (const hipError_t e = check_launch(s, s.current, *fn->kernel, {gx, gy, gz}, {bx, by, bz}, shared, stream);
      e != hipSuccess)
    return record(s, e);
  Occupancy o;
  const vgpu::DeviceProfile& p = s.rt->device(s.current).profile();
  if (const hipError_t e = occupancy(p, *fn->kernel, static_cast<int>(bx * by * bz), shared, false, &o);
      e != hipSuccess)
    return record(s, e);
  const uint64_t resident = uint64_t(o.blocks_per_cu) * p.limits.multiprocessors;
  if (uint64_t{gx} * gy * gz > resident) return record(s, hipErrorCooperativeLaunchTooLarge);
  std::vector<uint8_t> args;
  if (const hipError_t e = build_kernargs(*fn->kernel, params, nullptr, &args); e != hipSuccess) return record(s, e);
  if (const hipStream_t r = resolve(s, stream); r && stream_capturing(s, r))
    return record(s, capture_kernel(s, r, f, *fn->kernel, {gx, gy, gz}, {bx, by, bz}, shared, std::move(args), nullptr,
                                    true));
  LaunchJob job;
  if (const hipError_t e = prepare_launch(s, s.current, *fn->module, *fn->kernel, {gx, gy, gz}, {bx, by, bz}, shared,
                                          std::move(args), stream, true, &job);
      e != hipSuccess)
    return record(s, e);
  lock.unlock();
  return record(s, launch_in_order(std::move(job)));
}

// The options a module is loaded with tune a JIT; there is none here.
hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image, unsigned int, void*, void**) {
  const ApiCall api("hipModuleLoadDataEx");
  return hipModuleLoadData(module, image);
}

// A host function run in stream order: nothing is queued ahead of it, so it
// runs now. In a capture it would become a graph node, which this refuses.
// A host function run in stream order, on the stream's thread, once the work
// before it is done. In a capture it would become a graph node, which this
// refuses.
hipError_t hipLaunchHostFunc(hipStream_t stream, void (*fn)(void*), void* data) {
  const ApiCall api("hipLaunchHostFunc");
  {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!fn) return record(s, hipErrorInvalidValue);
    stream = resolve(s, stream);
    // In a capture, a host node.
    if (stream && stream_capturing(s, stream)) return record(s, capture_host(s, stream, fn, data));
  }
  return record(state(), in_order(stream, [fn, data] {
    fn(data);
    return hipSuccess;
  }));
}

// A host function in the stream's order that is told which stream it ran in
// and how the stream's work had gone: the older form of hipLaunchHostFunc.
// Its status is the failure the stream's earlier work left, if any, which
// the callback sees without taking it from the next synchronization.
typedef void (*hipStreamCallback_t)(hipStream_t stream, hipError_t status, void* data);
hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* data, unsigned int flags) {
  const ApiCall api("hipStreamAddCallback");
  std::shared_ptr<Queue> queue;
  {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!callback || flags) return record(s, hipErrorInvalidValue);
    Stream null_stream;
    if (!find_stream(s, stream, &null_stream)) return record(s, hipErrorInvalidHandle);
    if (const hipStream_t r = resolve(s, stream); r && stream_capturing(s, r))
      return record(s, fail(hipErrorStreamCaptureUnsupported, "a stream callback cannot be captured"));
  }
  // It is told the stream by the handle it was given -- but the thread's own
  // stream by its own handle, as ROCm's HIP passes it, not the reserved one.
  hipStream_t told = stream;
  if (reinterpret_cast<intptr_t>(stream) == kStreamPerThread) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    told = resolve(s, stream);
  }
  return record(state(), in_order(stream, [told, callback, data] {
    callback(told, hipSuccess, data);
    return hipSuccess;
  }));
}

// ---- Streams -------------------------------------------------------------------

hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority) {
  const ApiCall api("hipStreamCreateWithPriority");
  if (flags & ~static_cast<unsigned>(hipStreamNonBlocking)) return hipErrorInvalidValue;
  return create_stream(stream, flags, std::clamp(priority, -1, 1));
}
// A stream kept to some of the device's compute units. Every launch runs to
// completion either way; the mask is kept and read back.
hipError_t hipExtStreamCreateWithCUMask(hipStream_t* stream, uint32_t words, const uint32_t* mask) {
  const ApiCall api("hipExtStreamCreateWithCUMask");
  if (!words || !mask) return hipErrorInvalidValue;
  // A mask with no compute unit in it is no mask: the stream gets them all,
  // as ROCm gives it the default.
  std::vector<uint32_t> cus(mask, mask + words);
  if (std::all_of(cus.begin(), cus.end(), [](uint32_t w) { return w == 0; })) cus.clear();
  return create_stream(stream, 0, 0, std::move(cus));
}
hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t words, uint32_t* mask) {
  const ApiCall api("hipExtStreamGetCUMask");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!mask || !words) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  Stream null_stream;
  const Stream* st = find_stream(s, stream, &null_stream);
  if (!st) return record(s, hipErrorInvalidHandle);
  const uint32_t cus = s.rt->device(st->device).profile().limits.multiprocessors;
  for (uint32_t w = 0; w < words; ++w) {
    if (!st->cu_mask.empty()) {
      mask[w] = w < st->cu_mask.size() ? st->cu_mask[w] : 0;
    } else {
      const uint32_t first = 32 * w;
      mask[w] = first >= cus ? 0 : cus - first >= 32 ? ~0u : (1u << (cus - first)) - 1;
    }
  }
  return record(s, hipSuccess);
}
hipError_t hipStreamGetPriority(hipStream_t stream, int* priority) {
  const ApiCall api("hipStreamGetPriority");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  Stream null_stream;
  const Stream* st = find_stream(s, stream, &null_stream);
  if (!priority) return record(s, hipErrorInvalidValue);
  if (!st) return record(s, hipErrorInvalidHandle);
  *priority = st->priority;
  return record(s, hipSuccess);
}
// ROCm's HIP refuses the null handle here (a per-thread program's _spt form
// takes it as its own stream, and answers).
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags) {
  const ApiCall api("hipStreamGetFlags");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!stream) return record(s, hipErrorInvalidValue);
  Stream null_stream;
  const Stream* st = find_stream(s, stream, &null_stream);
  if (!flags) return record(s, hipErrorInvalidValue);
  if (!st) return record(s, hipErrorInvalidHandle);
  *flags = st->flags;
  return record(s, hipSuccess);
}
hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t* ordinal) {
  const ApiCall api("hipStreamGetDevice");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  Stream null_stream;
  const Stream* st = find_stream(s, stream, &null_stream);
  if (!ordinal) return record(s, hipErrorInvalidValue);
  if (!st) return record(s, hipErrorInvalidHandle);
  *ordinal = st->device;
  return record(s, hipSuccess);
}
// Whatever the event marks has already happened.
// The stream's later work waits for what the event marks. An event never
// recorded marks nothing, and the stream does not wait.
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  const ApiCall api("hipStreamWaitEvent");
  // hipEventWaitDefault (0) and hipEventWaitExternal (1) are the flags there are.
  if (flags > 1) return record(state(), hipErrorInvalidValue);
  {
    // An event from a capture, or a stream that is capturing: the capture's
    // to handle (a stream joins a capture this way).
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
    std::lock_guard<std::mutex> event_lock(g_event_mutex);
    const Event* e = find_event(event);
    if (!e) return record(s, hipErrorInvalidHandle);
    hipError_t result = hipSuccess;
    if (capture_wait(s, resolve(s, stream), e, &result)) return record(s, result);
    // Another process's event: the stream waits until its records so far
    // have happened there.
    if (e->remote && e->ipc) {
      const std::shared_ptr<IpcEventState> ipc = e->ipc;
      const uint64_t made = ipc->counters[0].load();
      Order o;
      if (const hipError_t err = order_for(s, stream, &o); err != hipSuccess) return record(s, err);
      o.queue->submit(
          [ipc, made] {
            ipc->wait(made);
            return hipSuccess;
          },
          std::move(o.after));
      return record(s, hipSuccess);
    }
  }
  Queue::Marker m;
  bool exists = false;
  const bool recorded = event_marker(event, &m, nullptr, &exists);
  if (!exists) return record(state(), hipErrorInvalidHandle);
  if (!recorded) return record(state(), hipSuccess);
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  Order o;
  if (const hipError_t e = order_for(s, stream, &o); e != hipSuccess) return record(s, e);
  if (const hipStream_t r = resolve(s, stream); r) {
    uint64_t& w = s.streams.at(r).waited[m.queue.get()];
    w = std::max(w, m.seq);
  }
  o.after.push_back(m);
  o.queue->submit([] { return hipSuccess; }, std::move(o.after));
  return record(s, hipSuccess);
}
}  // extern "C"
namespace {
hipError_t write_value(hipStream_t stream, void* ptr, uint64_t value, unsigned bytes, unsigned flags);
}  // namespace
extern "C" {

// A 32-bit value written to memory in stream order (see "Waiting on and
// writing memory in stream order").
hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags) {
  const ApiCall api("hipStreamWriteValue32");
  return record(state(), write_value(stream, ptr, value, 4, flags));
}

// ---- Memory ----------------------------------------------------------------------

// Device memory by kind. Coarse-grained (0), fine-grained (1), uncached (3)
// and contiguous (4) are all device memory, as ROCm's HIP reports them, and
// what RCCL fills and shares between processes: every write is visible at
// once here, so the grain changes nothing. Signal memory (2) is host memory
// every device reaches, mapped as hipHostMallocMapped memory is, since the
// host reads the signal directly; it is freed with hipFree.
hipError_t hipExtMallocWithFlags(void** ptr, size_t size, unsigned int flags) {
  const ApiCall api("hipExtMallocWithFlags");
  if (!ptr || flags > 4) return record(state(), hipErrorInvalidValue);
  // Signal memory is one 64-bit signal, and nothing else.
  if (flags == 2 && size != 8) return record(state(), hipErrorInvalidValue);
  if (!size) {
    *ptr = nullptr;
    return record(state(), hipSuccess);
  }
  if (flags == 2) return host_alloc(ptr, size, g_host_allocations, 2);   // hipHostMallocMapped
  return hipMalloc(ptr, size);
}
// A copy in the stream's order that the caller waits for.
hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t bytes, hipMemcpyKind kind, hipStream_t stream) {
  const ApiCall api("hipMemcpyWithStream");
  return record(state(), copy_in_order(dst, src, bytes, kind, stream, false));
}
// count 32-bit words of value.
hipError_t hipMemsetD32Async(void* dst, int value, size_t count, hipStream_t stream) {
  const ApiCall api("hipMemsetD32Async");
  if (dst && reinterpret_cast<uint64_t>(dst) % 4) return record(state(), hipErrorInvalidValue);
  uint8_t pattern[4];
  std::memcpy(pattern, &value, 4);
  return record(state(), fill_in_order(dst, pattern, 4, uint64_t{count} * 4, stream, true));
}

// The allocation an address falls in: its start and its size.
hipError_t hipMemGetAddressRange(void** base, size_t* size, void* ptr) {
  const ApiCall api("hipMemGetAddressRange");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr) return record(s, hipErrorNotFound);   // no allocation holds nothing, as ROCm's HIP has it
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  const uint64_t va = reinterpret_cast<uint64_t>(ptr);
  uint64_t b = 0, n = 0;
  // Host memory the devices map is an allocation of its own too.
  {
    std::lock_guard<std::mutex> host_lock(g_host_mutex);
    for (const auto* m : {&g_host_allocations, &g_host_registered, &g_managed})
      if (const auto it = find_range(*m, va); it != m->end()) {
        b = it->first;
        n = it->second;
      }
  }
  if (!n) graph_memory_size(va, &b, &n);
  const int d = n ? -1 : owner_of(s, va);
  if (!n && (d < 0 || !s.rt->device(d).memory().find_allocation(va, &b, &n))) return record(s, hipErrorNotFound);
  if (base) *base = reinterpret_cast<void*>(b);
  if (size) *size = static_cast<size_t>(n);
  return record(s, hipSuccess);
}
hipError_t hipMemPtrGetInfo(void* ptr, size_t* size) {
  const ApiCall api("hipMemPtrGetInfo");
  if (!size) return record(state(), hipErrorInvalidValue);
  // What a zero-byte hipMalloc gives back holds nothing.
  if (!ptr) {
    *size = 0;
    return record(state(), hipSuccess);
  }
  const hipError_t e = hipMemGetAddressRange(nullptr, size, ptr);
  return e == hipErrorNotFound ? record(state(), hipErrorInvalidValue) : e;
}

// One fact about an address, of those hipPointerGetAttributes gives and the
// allocation it falls in.
// The three yes-or-no attributes are one byte, and the buffer id and range
// size 32 bits, as ROCm's HIP writes them.
hipError_t hipPointerGetAttribute(void* data, int attribute, void* ptr) {
  const ApiCall api("hipPointerGetAttribute");
  if (!data || !ptr) return record(state(), hipErrorInvalidValue);
  vgpu::amd::abi::PointerAttribute a{};
  if (const hipError_t e = hipPointerGetAttributes(&a, ptr); e != hipSuccess) return e;
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  uint64_t base = reinterpret_cast<uint64_t>(ptr), size = 0;
  if (a.type == vgpu::amd::abi::kMemoryDevice) s.rt->device(a.device).memory().find_allocation(base, &base, &size);
  using K = vgpu::amd::abi::PointerAttributeKind;
  switch (attribute) {
    case K::kPointerContext: *static_cast<void**>(data) = a.device >= 0 ? &g_contexts[a.device % 64] : nullptr; break;
    case K::kPointerMemoryType: *static_cast<unsigned*>(data) = static_cast<unsigned>(a.type); break;
    case K::kPointerDevicePointer: *static_cast<void**>(data) = a.devicePointer; break;
    case K::kPointerHostPointer:
      if (!a.hostPointer) return record(s, hipErrorInvalidValue);   // device memory has none
      *static_cast<void**>(data) = a.hostPointer;
      break;
    case K::kPointerSyncMemops: *static_cast<bool*>(data) = s.sync_memops.count(base) != 0; break;
    case K::kPointerBufferId: *static_cast<uint32_t*>(data) = static_cast<uint32_t>((base >> 8) ^ (base >> 40)); break;
    case K::kPointerIsManaged: *static_cast<bool*>(data) = a.isManaged != 0; break;
    case K::kPointerDeviceOrdinal: *static_cast<int*>(data) = a.device; break;
    case K::kPointerRangeStartAddr: *static_cast<void**>(data) = reinterpret_cast<void*>(base); break;
    case K::kPointerRangeSize: *static_cast<uint32_t*>(data) = static_cast<uint32_t>(size); break;
    case K::kPointerMapped: *static_cast<bool*>(data) = a.type != vgpu::amd::abi::kMemoryUnregistered; break;
    // P2P tokens, legacy IPC, handle types, GPUDirect RDMA and the pool:
    // attributes HIP names but does not answer.
    case 16: *static_cast<int*>(data) = 3; break;   // ACCESS_FLAGS: the device reads and writes it
    case 5: case 10: case 14: case 15: case 17: return record(s, hipErrorNotSupported);
    default: return record(s, hipErrorInvalidValue);
  }
  if (a.type == vgpu::amd::abi::kMemoryUnregistered && attribute != K::kPointerMemoryType &&
      attribute != K::kPointerMapped)
    return record(s, hipErrorInvalidValue);
  return record(s, hipSuccess);
}

// ---- Virtual memory: address space, memory, and the mapping between ------------
//
// The device's memory manager already models it the way HIP (and CUDA)
// documents it: reserved address space with nothing behind it, memory with
// no address, a mapping of one into the other, and access granted before it
// can be used (vgpu/memory.hpp).
//
// HIP's minimum granularity is the 4 KiB page, and its recommended one is
// larger; the manager works in 64 KiB chunks, so a size HIP accepts is
// rounded up to whole chunks underneath it.

constexpr size_t kVmmPage = 4096;
constexpr size_t kVmmRecommended = static_cast<size_t>(vgpu::MemoryManager::kVmmGranularity);
uint64_t vmm_chunks(size_t size) {
  return (size + kVmmRecommended - 1) / kVmmRecommended * kVmmRecommended;
}
// Pinned or uncached memory, on a device or the host.
bool vmm_prop_ok(const vgpu::amd::abi::MemAllocationProp* prop) {
  return prop && (prop->type == 1 || prop->type == 0x40000000) && (prop->location.type == 1 || prop->location.type == 2);
}

hipError_t hipMemGetAllocationGranularity(size_t* granularity, const vgpu::amd::abi::MemAllocationProp* prop,
                                          int option) {
  const ApiCall api("hipMemGetAllocationGranularity");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!granularity || !vmm_prop_ok(prop) || (option != 0 && option != 1)) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (prop->location.type == 1 && (prop->location.id < 0 || prop->location.id >= s.rt->device_count()))
    return record(s, hipErrorInvalidValue);
  *granularity = option == 0 ? kVmmPage : kVmmRecommended;
  return record(s, hipSuccess);
}
hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void*, unsigned long long) {
  const ApiCall api("hipMemAddressReserve");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr || !size || size % kVmmPage || (alignment & (alignment - 1))) return record(s, hipErrorInvalidValue);
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
  try {
    *ptr = reinterpret_cast<void*>(
        d->memory().reserve(vmm_chunks(size), alignment > kVmmRecommended ? alignment : 0));
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  return record(s, hipSuccess);
}
hipError_t hipMemAddressFree(void* ptr, size_t size) {
  const ApiCall api("hipMemAddressFree");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr || !size || size % kVmmPage || !s.rt) return record(s, hipErrorInvalidValue);
  const int d = owner_of(s, reinterpret_cast<uint64_t>(ptr));
  if (d < 0) return record(s, hipErrorInvalidValue);
  // Address space still mapped is not free to go.
  const uint64_t lo = reinterpret_cast<uint64_t>(ptr);
  if (const auto m = g_vmm_maps.lower_bound(lo); m != g_vmm_maps.end() && m->first < lo + size)
    return record(s, fail(hipErrorInvalidValue, "freeing address space that is still mapped (hipMemUnmap first)"));
  try {
    s.rt->device(d).memory().address_free(reinterpret_cast<uint64_t>(ptr), vmm_chunks(size));
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  return record(s, hipSuccess);
}
}  // extern "C"
namespace {
// Host memory for a VMM allocation of `bytes`: a file (memfd), where it may
// be shared, or plain shared anonymous memory.
hipError_t vmm_memory(uint64_t bytes, bool as_file, VmmAllocation* v) {
  int fd = -1;
  void* host = MAP_FAILED;
  if (as_file) {
    fd = ::memfd_create("hip-vmm", MFD_CLOEXEC);
    if (fd >= 0 && ::ftruncate(fd, static_cast<off_t>(bytes)) == 0)
      host = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  } else {
    host = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  }
  if (host == MAP_FAILED) {
    if (fd >= 0) ::close(fd);
    return fail(hipErrorOutOfMemory, std::string("virtual memory's backing: ") + std::strerror(errno));
  }
  v->fd = fd;
  v->host = static_cast<uint8_t*>(host);
  v->bytes = bytes;
  return hipSuccess;
}
// An allocation goes once nothing holds it: no reference and no mapping.
void drop_vmm(State& s, VmmAllocation* v) {
  if (v->refs > 0 || v->maps > 0 || !v->host) return;
  ::munmap(v->host, v->bytes);
  if (v->fd >= 0) ::close(v->fd);
  if (v->handle && v->device >= 0) {
    try {
      s.rt->device(v->device).memory().release_handle(v->handle);
    } catch (const std::exception&) {
    }
  }
  v->fd = -1;
  v->host = nullptr;
}
// Where memory that was not made to be shared is asked for as a file: its
// bytes move into one, mapped at the same host address, so every mapping of
// it goes on reaching them.
hipError_t vmm_as_file(VmmAllocation* v) {
  if (v->fd >= 0) return hipSuccess;
  const int fd = ::memfd_create("hip-vmm", MFD_CLOEXEC);
  if (fd < 0 || ::ftruncate(fd, static_cast<off_t>(v->bytes)) != 0 ||
      ::pwrite(fd, v->host, v->bytes, 0) != static_cast<ssize_t>(v->bytes) ||
      ::mmap(v->host, v->bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0) == MAP_FAILED) {
    if (fd >= 0) ::close(fd);
    return fail(hipErrorOutOfMemory, std::string("moving virtual memory into a file: ") + std::strerror(errno));
  }
  v->fd = fd;
  return hipSuccess;
}
// Memory from a file descriptor: the file, mapped here, as an allocation of
// the current device's. A file shorter than whole pages is lengthened to
// them, so a mapping never runs past its end.
hipError_t vmm_from_fd(State& s, int fd, void** handle) {
  struct stat st{};
  if (fd < 0 || ::fstat(fd, &st) != 0 || st.st_size <= 0) return hipErrorInvalidValue;
  const uint64_t bytes = vmm_chunks(static_cast<size_t>(st.st_size));
  if (static_cast<uint64_t>(st.st_size) < bytes) (void)!::ftruncate(fd, static_cast<off_t>(bytes));
  const int mine = ::fcntl(fd, F_DUPFD_CLOEXEC, 0);
  void* host = mine < 0 ? MAP_FAILED : ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, mine, 0);
  if (host == MAP_FAILED) {
    if (mine >= 0) ::close(mine);
    return fail(hipErrorInvalidValue, std::string("mapping shared memory: ") + std::strerror(errno));
  }
  g_vmm.push_back(VmmAllocation{s.current, 0, static_cast<size_t>(st.st_size), 1, 1});
  VmmAllocation& v = g_vmm.back();
  v.fd = mine;
  v.host = static_cast<uint8_t*>(host);
  v.bytes = bytes;
  v.handle_types = 1;
  g_vmm_live.insert(&v);
  *handle = &v;
  return hipSuccess;
}
}  // namespace
extern "C" {

// Memory on a device, or pinned host memory (hipMemLocationTypeHost), in
// whole pages. Memory made to be exported as a file descriptor is a file
// from the start. A device's memory is counted against it.
hipError_t hipMemCreate(void** handle, size_t size, const vgpu::amd::abi::MemAllocationProp* prop,
                        unsigned long long flags) {
  const ApiCall api("hipMemCreate");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  constexpr int kLocationDevice = 1, kLocationHost = 2;
  if (!handle || !prop || (prop->type != 1 && prop->type != 0x40000000) || !size || size % kVmmPage || flags)
    return record(s, hipErrorInvalidValue);
  if (prop->location.type != kLocationDevice && prop->location.type != kLocationHost)
    return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  const bool on_host = prop->location.type == kLocationHost;
  const int d = on_host ? -1 : prop->location.id;
  if (!on_host && (d < 0 || d >= s.rt->device_count())) return record(s, hipErrorInvalidDevice);
  const uint64_t bytes = vmm_chunks(size);
  uint64_t counted = 0;
  if (!on_host) {
    try {
      counted = s.rt->device(d).memory().create_handle(bytes);
    } catch (const std::exception& e) {
      return record(s, fail(hipErrorOutOfMemory, e.what()));
    }
  }
  VmmAllocation v{d, counted, size, prop->type, 1};
  v.handle_types = prop->requestedHandleType;
  // Host memory is a file too, which the CPU may be given at its address.
  if (const hipError_t e = vmm_memory(bytes, (prop->requestedHandleType & 1) || on_host, &v); e != hipSuccess) {
    if (counted) s.rt->device(d).memory().release_handle(counted);
    return record(s, e);
  }
  g_vmm.push_back(v);
  g_vmm_live.insert(&g_vmm.back());
  *handle = &g_vmm.back();
  return record(s, hipSuccess);
}
// The memory goes when its last reference and its last mapping have, in
// either order.
hipError_t hipMemRelease(void* handle) {
  const ApiCall api("hipMemRelease");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  auto* v = static_cast<VmmAllocation*>(handle);
  if (!g_vmm_live.count(v)) return record(s, hipErrorInvalidValue);
  if (--v->refs > 0) return record(s, hipSuccess);
  g_vmm_live.erase(v);
  drop_vmm(s, v);
  return record(s, hipSuccess);
}
// Into address space some reservation set aside, on any device, where
// nothing is mapped yet: `size` bytes of the memory from `offset`.
hipError_t hipMemMap(void* ptr, size_t size, size_t offset, void* handle, unsigned long long flags) {
  const ApiCall api("hipMemMap");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  auto* v = static_cast<VmmAllocation*>(handle);
  if (!ptr || !size || size % kVmmPage || offset % kVmmPage || flags || !g_vmm_live.count(v))
    return record(s, hipErrorInvalidValue);
  if (offset > v->bytes || size > v->bytes - offset) return record(s, hipErrorInvalidValue);
  const uint64_t lo = reinterpret_cast<uint64_t>(ptr), hi = lo + size;
  const int owner = owner_of(s, lo);
  if (owner < 0 || owner_of(s, hi - 1) != owner) return record(s, hipErrorInvalidValue);
  if (const auto next = g_vmm_maps.lower_bound(lo); next != g_vmm_maps.end() && next->first < hi)
    return record(s, fail(hipErrorInvalidValue, "that address space is already mapped"));
  if (vmm_mapping(lo) != g_vmm_maps.end()) return record(s, fail(hipErrorInvalidValue, "that address space is already mapped"));
  s.rt->device(owner).memory().map_host(lo, v->host + offset, size);
  ++v->maps;
  g_vmm_maps[lo] = {size, v};
  g_vmm_offsets[lo] = offset;
  return record(s, hipSuccess);
}
// Every mapping inside the range goes; the range has to hold one.
hipError_t hipMemUnmap(void* ptr, size_t size) {
  const ApiCall api("hipMemUnmap");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr || !size || size % kVmmPage || !s.rt) return record(s, hipErrorInvalidValue);
  const uint64_t lo = reinterpret_cast<uint64_t>(ptr), hi = lo + size;
  const auto first = g_vmm_maps.find(lo);
  if (first == g_vmm_maps.end() || first->second.first > size) return record(s, hipErrorInvalidValue);
  const int d = owner_of(s, lo);
  for (auto it = g_vmm_maps.lower_bound(lo); it != g_vmm_maps.end() && it->first < hi;) {
    if (it->first + it->second.first > hi) return record(s, hipErrorInvalidValue);
    ++it;
  }
  for (auto it = g_vmm_maps.lower_bound(lo); it != g_vmm_maps.end() && it->first < hi;) {
    VmmAllocation* v = it->second.second;
    s.rt->device(d).memory().unmap_host(it->first);
    if (const auto cpu = g_vmm_cpu_maps.find(it->first); cpu != g_vmm_cpu_maps.end()) {
      ::munmap(reinterpret_cast<void*>(cpu->first), cpu->second);
      g_vmm_cpu_maps.erase(cpu);
    }
    --v->maps;
    if (!g_vmm_live.count(v)) drop_vmm(s, v);
    for (auto a = s.vmm_access.begin(); a != s.vmm_access.end();)
      a = a->first.first >= it->first && a->first.first < it->first + it->second.first ? s.vmm_access.erase(a)
                                                                                        : std::next(a);
    g_vmm_offsets.erase(it->first);
    it = g_vmm_maps.erase(it);
  }
  return record(s, hipSuccess);
}
// The allocation mapped at an address, with a reference of its own that
// hipMemRelease gives back, as RCCL takes it for memory it is handed.
hipError_t hipMemRetainAllocationHandle(void** handle, void* addr) {
  const ApiCall api("hipMemRetainAllocationHandle");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!handle || !addr) return record(s, hipErrorInvalidValue);
  const auto it = vmm_mapping(reinterpret_cast<uint64_t>(addr));
  if (it == g_vmm_maps.end() || !g_vmm_live.count(it->second.second)) return record(s, hipErrorInvalidValue);
  ++it->second.second->refs;
  *handle = it->second.second;
  return record(s, hipSuccess);
}
// What an allocation was made as.
hipError_t hipMemGetAllocationPropertiesFromHandle(vgpu::amd::abi::MemAllocationProp* prop, void* handle) {
  const ApiCall api("hipMemGetAllocationPropertiesFromHandle");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  auto* v = static_cast<VmmAllocation*>(handle);
  if (!prop || !g_vmm_live.count(v)) return record(s, hipErrorInvalidValue);
  std::memset(prop, 0, sizeof *prop);
  prop->type = v->type;
  prop->location.type = v->device < 0 ? 2 : 1;   // hipMemLocationTypeHost or Device
  prop->location.id = v->device < 0 ? 0 : v->device;
  prop->requestedHandleType = v->handle_types;
  return record(s, hipSuccess);
}
// Access for devices to mapped memory: all of the range mapped, each grant
// for a device. Every device reaches the memory here; one granted access to
// another's also has its kernels reach that device's address space.
hipError_t hipMemSetAccess(void* ptr, size_t size, const vgpu::amd::abi::MemAccessDesc* desc, size_t count) {
  const ApiCall api("hipMemSetAccess");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr || !size || size % kVmmPage || !desc || !count || !s.rt) return record(s, hipErrorInvalidValue);
  const uint64_t lo = reinterpret_cast<uint64_t>(ptr), hi = lo + size;
  // Mapped all the way, one mapping after another.
  for (uint64_t at = lo; at < hi;) {
    const auto it = g_vmm_maps.find(at);
    if (it == g_vmm_maps.end() || it->first + it->second.first > hi) return record(s, hipErrorInvalidValue);
    at += it->second.first;
  }
  // The host may be given host memory only.
  bool all_host = true;
  for (auto it = g_vmm_maps.find(lo); it != g_vmm_maps.end() && it->first < hi; ++it)
    all_host = all_host && it->second.second->device < 0;
  for (size_t i = 0; i < count; ++i) {
    if (desc[i].location.type == 2) {
      if (!all_host) return record(s, hipErrorInvalidValue);
    } else if (desc[i].location.type != 1 || desc[i].location.id < 0 ||
               desc[i].location.id >= s.rt->device_count()) {
      return record(s, hipErrorInvalidValue);
    }
    if (desc[i].flags != 0 && desc[i].flags != 1 && desc[i].flags != 3) return record(s, hipErrorInvalidValue);
  }
  const int owner = owner_of(s, lo);
  for (size_t i = 0; i < count; ++i) {
    if (desc[i].location.type == 2) {
      // The CPU reaches it at its device address: the file mapped there.
      for (auto it = g_vmm_maps.find(lo); it != g_vmm_maps.end() && it->first < hi; ++it) {
        if (g_vmm_cpu_maps.count(it->first) || !desc[i].flags) continue;
        void* at = ::mmap(reinterpret_cast<void*>(it->first), it->second.first, PROT_READ | PROT_WRITE,
                          MAP_SHARED | MAP_FIXED_NOREPLACE, it->second.second->fd,
                          static_cast<off_t>(g_vmm_offsets[it->first]));
        if (at == MAP_FAILED || at != reinterpret_cast<void*>(it->first)) {
          if (at != MAP_FAILED) ::munmap(at, it->second.first);
          return record(s, fail(hipErrorOutOfMemory, "the host's own memory is at that address"));
        }
        g_vmm_cpu_maps[it->first] = it->second.first;
      }
      continue;
    }
    s.vmm_access[{lo, size}][desc[i].location.id] = desc[i].flags;
    if (desc[i].flags && desc[i].location.id != owner) s.peers.insert({desc[i].location.id, owner});
  }
  return record(s, hipSuccess);
}

// Memory made to be shared (hipMemHandleTypePosixFileDescriptor) is handed
// out as a file descriptor of its file, which another process -- or this
// one -- imports as memory of its own that maps the same bytes.
hipError_t hipMemExportToShareableHandle(void* out, void* handle, int type, unsigned long long flags) {
  const ApiCall api("hipMemExportToShareableHandle");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  auto* v = static_cast<VmmAllocation*>(handle);
  if (!out || !g_vmm_live.count(v) || type != 1 || flags) return record(s, hipErrorInvalidValue);
  if (v->fd < 0 || !(v->handle_types & 1))
    return record(s, fail(hipErrorInvalidValue, "the memory was not made to be exported as a file descriptor"));
  const int fd = ::fcntl(v->fd, F_DUPFD_CLOEXEC, 0);
  if (fd < 0) return record(s, fail(hipErrorOutOfMemory, std::string("dup: ") + std::strerror(errno)));
  *static_cast<int*>(out) = fd;
  return record(s, hipSuccess);
}
hipError_t hipMemImportFromShareableHandle(void** handle, void* os_handle, int type) {
  const ApiCall api("hipMemImportFromShareableHandle");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!handle || !os_handle || type != 1) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  return record(s, vmm_from_fd(s, static_cast<int>(reinterpret_cast<uintptr_t>(os_handle)), handle));
}

// ---- Memory another process maps (IPC) ---------------------------------------------
//
// An allocation that is shared moves into a file of its own, at the same
// address, and the other process maps that file: both then reach the same
// bytes (vgpu/memory.hpp's share and adopt).

}  // extern "C"
namespace {
// Sharing moves an allocation's bytes into a file at the same address. Work
// still queued on its device has to finish first -- a hipMemset of device
// memory returns before it runs, as ROCm's does -- or it would write memory
// that has moved. Called without the state lock, which the queues take.
void settle_before_sharing(const void* ptr) {
  State& s = state();
  int d = -1;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.rt || !ptr) return;
    d = owner_of(s, reinterpret_cast<uint64_t>(ptr));
  }
  if (d >= 0) (void)drain_device(d);
}
}  // namespace
extern "C" {

hipError_t hipIpcGetMemHandle(vgpu::amd::abi::IpcMemHandle* handle, void* ptr) {
  const ApiCall api("hipIpcGetMemHandle");
  settle_before_sharing(ptr);
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!handle || !ptr) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  const int d = owner_of(s, reinterpret_cast<uint64_t>(ptr));
  if (d < 0) return record(s, hipErrorInvalidDevicePointer);
  static std::atomic<uint32_t> counter{0};
  IpcPayload p{};
  p.magic = kIpcMagic;
  p.version = 1;
  p.device = static_cast<uint32_t>(d);
  p.pid = static_cast<uint32_t>(::getpid());
  std::snprintf(p.id, sizeof p.id, "%x-%x", p.pid, counter.fetch_add(1) + 1);
  try {
    p.size = s.rt->device(d).memory().share(reinterpret_cast<uint64_t>(ptr), ipc_path(p.id));
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  std::memset(handle, 0, sizeof *handle);
  std::memcpy(handle, &p, sizeof p);
  return record(s, hipSuccess);
}
hipError_t hipIpcOpenMemHandle(void** ptr, vgpu::amd::abi::IpcMemHandle handle, unsigned int flags) {
  const ApiCall api("hipIpcOpenMemHandle");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr || flags > 1) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  IpcPayload p{};
  std::memcpy(&p, &handle, sizeof p);
  if (p.magic != kIpcMagic || p.version != 1 || !p.size) return record(s, hipErrorInvalidValue);
  // The exporting process has the memory already, and HIP does not let it
  // open its own handle either.
  if (p.pid == static_cast<uint32_t>(::getpid()))
    return record(s, fail(hipErrorInvalidContext, "a process cannot open a memory handle it exported"));
  const int d = p.device < static_cast<uint32_t>(s.rt->device_count()) ? static_cast<int>(p.device) : s.current;
  try {
    *ptr = reinterpret_cast<void*>(s.rt->device(d).memory().adopt(ipc_path(p.id), p.size));
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  g_ipc_open[*ptr] = d;
  // hipIpcMemLazyEnablePeerAccess, which RCCL passes, and which is what an
  // opened handle behaves as on a card anyway: the memory is mapped where
  // the exporter's device is numbered here, and this process's other devices
  // reach it without asking. RCCL's kernels on device 0 write straight into
  // the other process's buffer on device 1.
  s.ipc_mapped.insert(d);
  return record(s, hipSuccess);
}
hipError_t hipIpcCloseMemHandle(void* ptr) {
  const ApiCall api("hipIpcCloseMemHandle");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  const auto it = g_ipc_open.find(ptr);
  if (it == g_ipc_open.end()) return record(s, hipErrorInvalidValue);
  s.rt->device(it->second).memory().abandon(reinterpret_cast<uint64_t>(ptr));
  g_ipc_open.erase(it);
  return record(s, hipSuccess);
}

// ---- Pools another process uses ----------------------------------------------------
//
// A pool made for export is a file descriptor, sent the way the program
// sends descriptors (a Unix socket), carrying who made the pool and its
// grants. Each allocation in it is exported on its own, as an IPC memory
// handle is: the bytes move into a file at the same address, and the
// importing process maps that file.

hipError_t hipMemPoolExportToShareableHandle(void* handle, void* pool, int type, unsigned long long flags) {
  const ApiCall api("hipMemPoolExportToShareableHandle");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  Pool* p = find_pool(s, pool);
  if (!handle || !p || flags || type != 1 || !(p->handle_types & 1)) return record(s, hipErrorInvalidValue);
  if (p->share_fd < 0) {
    const int fd = ::memfd_create("hip-mempool", MFD_CLOEXEC);
    if (fd < 0) return record(s, fail(hipErrorOutOfMemory, std::string("memfd_create: ") + std::strerror(errno)));
    PoolShare share{};
    share.magic = kPoolShareMagic;
    share.version = 1;
    share.pid = static_cast<uint32_t>(::getpid());
    share.device = p->device;
    if (::pwrite(fd, &share, sizeof share, 0) != static_cast<ssize_t>(sizeof share)) {
      ::close(fd);
      return record(s, fail(hipErrorOutOfMemory, "writing the pool's shared description"));
    }
    p->share_fd = fd;
    publish_pool_access(*p);
  }
  // The caller's descriptor is its own to send and close.
  const int out = ::fcntl(p->share_fd, F_DUPFD_CLOEXEC, 0);
  if (out < 0) return record(s, fail(hipErrorOutOfMemory, std::string("dup: ") + std::strerror(errno)));
  *static_cast<void**>(handle) = reinterpret_cast<void*>(static_cast<intptr_t>(out));
  return record(s, hipSuccess);
}
hipError_t hipMemPoolImportFromShareableHandle(void** pool, void* handle, int type, unsigned long long flags) {
  const ApiCall api("hipMemPoolImportFromShareableHandle");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!pool || !handle || type != 1 || flags) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  const int fd = static_cast<int>(reinterpret_cast<intptr_t>(handle));
  PoolShare share{};
  if (::pread(fd, &share, sizeof share, 0) != static_cast<ssize_t>(sizeof share) || share.magic != kPoolShareMagic ||
      share.version != 1)
    return record(s, fail(hipErrorInvalidValue, "the handle is not an exported memory pool"));
  const int mine = ::fcntl(fd, F_DUPFD_CLOEXEC, 0);
  if (mine < 0) return record(s, fail(hipErrorOutOfMemory, std::string("dup: ") + std::strerror(errno)));
  s.pools.push_back(Pool{});
  Pool& p = s.pools.back();
  p.device = share.device >= 0 && share.device < s.rt->device_count() ? share.device : s.current;
  p.handle_types = 1;
  p.share_fd = mine;
  p.imported = true;
  refresh_pool_access(p);
  *pool = &p;
  return record(s, hipSuccess);
}
hipError_t hipMemPoolExportPointer(void* data, void* ptr) {
  const ApiCall api("hipMemPoolExportPointer");
  settle_before_sharing(ptr);
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!data || !ptr) return record(s, hipErrorInvalidValue);
  const auto pa = s.pool_allocations.find(reinterpret_cast<uint64_t>(ptr));
  if (pa == s.pool_allocations.end() || !(pa->second.first->handle_types & 1))
    return record(s, fail(hipErrorInvalidValue, "not the start of an allocation from a pool made for export"));
  const int d = pa->second.first->device;
  IpcPayload p{};
  p.magic = kIpcMagic;
  p.version = 1;
  p.device = static_cast<uint32_t>(d);
  p.pid = static_cast<uint32_t>(::getpid());
  static std::atomic<uint32_t> counter{0};
  std::snprintf(p.id, sizeof p.id, "%x-p%x", p.pid, counter.fetch_add(1) + 1);
  vgpu::MemoryManager& m = s.rt->device(d).memory();
  try {
    // Exported once; each later export names the same file.
    if (const auto it = g_pool_exports.find(reinterpret_cast<uint64_t>(ptr));
        it != g_pool_exports.end() && m.is_shared(reinterpret_cast<uint64_t>(ptr)))
      p = it->second;
    else
      p.size = m.share(reinterpret_cast<uint64_t>(ptr), ipc_path(p.id));
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  g_pool_exports[reinterpret_cast<uint64_t>(ptr)] = p;
  std::memset(data, 0, 64);
  std::memcpy(data, &p, sizeof p);
  return record(s, hipSuccess);
}
hipError_t hipMemPoolImportPointer(void** ptr, void* pool, void* data) {
  const ApiCall api("hipMemPoolImportPointer");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  Pool* pl = find_pool(s, pool);
  if (!ptr || !pl || !data) return record(s, hipErrorInvalidValue);
  IpcPayload p{};
  std::memcpy(&p, data, sizeof p);
  if (p.magic != kIpcMagic || p.version != 1 || !p.size) return record(s, hipErrorInvalidValue);
  try {
    *ptr = reinterpret_cast<void*>(s.rt->device(pl->device).memory().adopt(ipc_path(p.id), p.size));
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  g_pool_imports[*ptr] = pl->device;
  // Its kernels reach the exporting process's memory as that process's do.
  s.ipc_mapped.insert(pl->device);
  return record(s, hipSuccess);
}

// An interprocess event (hipEventInterprocess) is handed to another process
// as the name of its shared state, which that process opens as an event of
// its own: its waits wait for the records made anywhere, and its records
// count everywhere. A process cannot open its own event's handle.
struct IpcEventPayload {
  uint32_t magic, version, pid, reserved;
  char id[40];
};
static_assert(sizeof(IpcEventPayload) <= sizeof(vgpu::amd::abi::IpcMemHandle), "an event's handle fits HIP's");
constexpr uint32_t kIpcEventMagic = 0x48455643;   // "HEVC"
hipError_t hipIpcGetEventHandle(void* handle, hipEvent_t event) {
  const ApiCall api("hipIpcGetEventHandle");
  std::lock_guard<std::mutex> lock(g_event_mutex);
  const Event* e = find_event(event);
  if (!handle || !e || !e->ipc) return record(state(), hipErrorInvalidValue);
  IpcEventPayload p{};
  p.magic = kIpcEventMagic;
  p.version = 1;
  p.pid = e->remote ? 0 : static_cast<uint32_t>(::getpid());
  std::memcpy(p.id, e->ipc_id, sizeof p.id);
  std::memset(handle, 0, sizeof(vgpu::amd::abi::IpcMemHandle));
  std::memcpy(handle, &p, sizeof p);
  return hipSuccess;
}
hipError_t hipIpcOpenEventHandle(hipEvent_t* event, vgpu::amd::abi::IpcMemHandle handle) {
  const ApiCall api("hipIpcOpenEventHandle");
  State& s = state();
  if (!event) return record(s, hipErrorInvalidValue);
  IpcEventPayload p{};
  std::memcpy(&p, &handle, sizeof p);
  if (p.magic != kIpcEventMagic || p.version != 1 || !p.id[0]) return record(s, hipErrorInvalidValue);
  if (p.pid == static_cast<uint32_t>(::getpid()))
    return record(s, fail(hipErrorInvalidContext, "a process cannot open an event handle it made"));
  int device = 0;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
    device = s.current;
  }
  auto e = std::make_unique<Event>();
  e->device = device;
  e->timing = false;
  e->remote = true;
  std::memcpy(e->ipc_id, p.id, sizeof e->ipc_id);
  e->ipc_id[sizeof e->ipc_id - 1] = '\0';
  e->ipc = map_ipc_event(vgpu::telemetry::default_path() + "/ipc-hip-" + e->ipc_id, false);
  if (!e->ipc) return record(s, fail(hipErrorInvalidValue, "the event's shared state is not there"));
  std::lock_guard<std::mutex> lock(g_event_mutex);
  static intptr_t next = 1 << 30;   // apart from hipEventCreate's handles
  const hipEvent_t h = reinterpret_cast<hipEvent_t>(next++);
  g_events.emplace(h, std::move(e));
  *event = h;
  return hipSuccess;
}

// ---- Arrays, textures and surfaces -------------------------------------------
//
// The GPUs modelled here (the MI300 family, gfx942 and gfx950) have no
// texture units: hipcc refuses the texture API in their device code
// (__HIP_NO_IMAGE_SUPPORT), and ROCm's HIP on them says image support is 0
// and answers every call that would make an array, a texture or a surface
// with hipErrorNotSupported. These are its answers, found by asking it
// (ROCm's libamdhip64 on this HSA runtime, amd/tests/hipcc/textures.cpp), so
// a program or library that calls them is told what it would be told on the
// card, rather than failing to load for want of the symbol.
namespace {
hipError_t no_images(const char* name) {
  const ApiCall api(name);
  return record(state(), hipErrorNotSupported);
}
hipError_t no_such_array(const char* name) {   // a handle to what cannot exist
  const ApiCall api(name);
  return record(state(), hipErrorInvalidHandle);
}
hipError_t freeing_nothing(const char* name) {
  const ApiCall api(name);
  return record(state(), hipErrorInvalidValue);
}
hipError_t destroying(const char* name, uint64_t object) {   // none is ever made, so only 0 is fine
  const ApiCall api(name);
  return record(state(), object ? hipErrorInvalidValue : hipSuccess);
}
}  // namespace

// hipChannelFormatDesc, which hipCreateChannelDesc returns by value.
struct ChannelFormatDesc {
  int x, y, z, w;
  int f;
};
ChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, int f) { return {x, y, z, w, f}; }

hipError_t hipDeviceGetTexture1DLinearMaxWidth(size_t* width, const void*, int device) {
  const ApiCall api("hipDeviceGetTexture1DLinearMaxWidth");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!width) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (device < 0 || device >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  *width = 0;
  return record(s, hipSuccess);
}

// What a hipcc-built program registers before main for a texture or surface
// reference it declares: nothing to keep, since none can be bound.
void __hipRegisterTexture(void*, void*, char*, const char*, int, int, int) {}
void __hipRegisterSurface(void*, void*, char*, const char*, int, int) {}

// A mipmapped array: its arguments checked, then refused like every array.
hipError_t hipMallocMipmappedArray(void** array, const void* desc, vgpu::amd::abi::Extent, unsigned int, unsigned int) {
  if (!array || !desc) {
    const ApiCall api("hipMallocMipmappedArray");
    return record(state(), hipErrorInvalidValue);
  }
  return no_images("hipMallocMipmappedArray");
}
#define VGPU_NO_IMAGES(name) \
  hipError_t name() { return no_images(#name); }
#define VGPU_NO_SUCH_ARRAY(name) \
  hipError_t name() { return no_such_array(#name); }
#define VGPU_FREEING_NOTHING(name) \
  hipError_t name() { return freeing_nothing(#name); }
// Making one, and the texture-reference API, which needs a texture to bind.
VGPU_NO_IMAGES(hipMallocArray)
VGPU_NO_IMAGES(hipMalloc3DArray)
VGPU_NO_IMAGES(hipArrayCreate)
VGPU_NO_IMAGES(hipArray3DCreate)
VGPU_NO_IMAGES(hipMipmappedArrayCreate)
VGPU_NO_IMAGES(hipCreateTextureObject)
VGPU_NO_IMAGES(hipTexObjectCreate)
VGPU_NO_IMAGES(hipCreateSurfaceObject)
VGPU_NO_IMAGES(hipGetTextureReference)
VGPU_NO_IMAGES(hipModuleGetTexRef)
VGPU_NO_IMAGES(hipBindTexture)
VGPU_NO_IMAGES(hipBindTexture2D)
VGPU_NO_IMAGES(hipBindTextureToArray)
VGPU_NO_IMAGES(hipBindTextureToMipmappedArray)
VGPU_NO_IMAGES(hipUnbindTexture)
VGPU_NO_IMAGES(hipGetTextureAlignmentOffset)
VGPU_NO_IMAGES(hipGetMipmappedArrayLevel)
VGPU_NO_IMAGES(hipMipmappedArrayGetLevel)
VGPU_NO_IMAGES(hipMemMapArrayAsync)
VGPU_NO_IMAGES(hipTexRefGetAddress)
VGPU_NO_IMAGES(hipTexRefGetAddressMode)
VGPU_NO_IMAGES(hipTexRefGetArray)
VGPU_NO_IMAGES(hipTexRefGetBorderColor)
VGPU_NO_IMAGES(hipTexRefGetFilterMode)
VGPU_NO_IMAGES(hipTexRefGetFlags)
VGPU_NO_IMAGES(hipTexRefGetFormat)
VGPU_NO_IMAGES(hipTexRefGetMaxAnisotropy)
VGPU_NO_IMAGES(hipTexRefGetMipMappedArray)
VGPU_NO_IMAGES(hipTexRefGetMipmapFilterMode)
VGPU_NO_IMAGES(hipTexRefGetMipmapLevelBias)
VGPU_NO_IMAGES(hipTexRefGetMipmapLevelClamp)
VGPU_NO_IMAGES(hipTexRefSetAddress)
VGPU_NO_IMAGES(hipTexRefSetAddress2D)
VGPU_NO_IMAGES(hipTexRefSetAddressMode)
VGPU_NO_IMAGES(hipTexRefSetArray)
VGPU_NO_IMAGES(hipTexRefSetBorderColor)
VGPU_NO_IMAGES(hipTexRefSetFilterMode)
VGPU_NO_IMAGES(hipTexRefSetFlags)
VGPU_NO_IMAGES(hipTexRefSetFormat)
VGPU_NO_IMAGES(hipTexRefSetMaxAnisotropy)
VGPU_NO_IMAGES(hipTexRefSetMipmapFilterMode)
VGPU_NO_IMAGES(hipTexRefSetMipmapLevelBias)
VGPU_NO_IMAGES(hipTexRefSetMipmapLevelClamp)
VGPU_NO_IMAGES(hipTexRefSetMipmappedArray)
// Graphics and external-memory interop, which has no graphics API to share with.
VGPU_NO_IMAGES(hipGraphicsSubResourceGetMappedArray)
// Asking about, or copying to or from, an array: there is none to name.
VGPU_NO_SUCH_ARRAY(hipGetChannelDesc)
VGPU_NO_SUCH_ARRAY(hipArrayGetDescriptor)
VGPU_NO_SUCH_ARRAY(hipArray3DGetDescriptor)
VGPU_NO_SUCH_ARRAY(hipArrayGetInfo)
VGPU_NO_SUCH_ARRAY(hipMemcpyToArray)
VGPU_NO_SUCH_ARRAY(hipMemcpyFromArray)
VGPU_NO_SUCH_ARRAY(hipMemcpyFromArray_spt)
VGPU_NO_SUCH_ARRAY(hipMemcpy2DToArray)
VGPU_NO_SUCH_ARRAY(hipMemcpy2DToArray_spt)
VGPU_NO_SUCH_ARRAY(hipMemcpy2DToArrayAsync)
VGPU_NO_SUCH_ARRAY(hipMemcpy2DToArrayAsync_spt)
VGPU_NO_SUCH_ARRAY(hipMemcpy2DFromArray)
VGPU_NO_SUCH_ARRAY(hipMemcpy2DFromArray_spt)
VGPU_NO_SUCH_ARRAY(hipMemcpy2DFromArrayAsync)
VGPU_NO_SUCH_ARRAY(hipMemcpy2DFromArrayAsync_spt)
VGPU_NO_SUCH_ARRAY(hipMemcpy2DArrayToArray)
VGPU_NO_SUCH_ARRAY(hipGetTextureObjectResourceDesc)
VGPU_NO_SUCH_ARRAY(hipGetTextureObjectResourceViewDesc)
VGPU_NO_SUCH_ARRAY(hipGetTextureObjectTextureDesc)
VGPU_NO_SUCH_ARRAY(hipTexObjectGetResourceDesc)
VGPU_NO_SUCH_ARRAY(hipTexObjectGetResourceViewDesc)
VGPU_NO_SUCH_ARRAY(hipTexObjectGetTextureDesc)
// Freeing one.
VGPU_FREEING_NOTHING(hipFreeArray)
VGPU_FREEING_NOTHING(hipArrayDestroy)
VGPU_FREEING_NOTHING(hipFreeMipmappedArray)
VGPU_FREEING_NOTHING(hipMipmappedArrayDestroy)
hipError_t hipDestroyTextureObject(uint64_t object) { return destroying("hipDestroyTextureObject", object); }
hipError_t hipTexObjectDestroy(uint64_t object) { return destroying("hipTexObjectDestroy", object); }
hipError_t hipDestroySurfaceObject(uint64_t object) { return destroying("hipDestroySurfaceObject", object); }

// ---- The rest of HIP's device, stream and launch API --------------------------
//
// Each call below answers as ROCm's HIP answers on an MI300-class device,
// down to the error for each wrong argument: AMD's own tests (hip-tests) are
// the reference for what that is.

// The flags hipSetDeviceFlags or hipCtxCreate left for the current device.
hipError_t hipGetDeviceFlags(unsigned int* flags) {
  const ApiCall api("hipGetDeviceFlags");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!flags) return record(s, hipErrorInvalidValue);
  const auto it = s.device_flags.find(s.current);
  *flags = it == s.device_flags.end() ? 1 /* hipDeviceScheduleSpin, ROCm's HIP's default */ : it->second;
  return record(s, hipSuccess);
}

// The properties' UUID: the sixteen characters of the HSA agent's.
hipError_t hipDeviceGetUuid(vgpu::amd::abi::Uuid* uuid, int ordinal) {
  const ApiCall api("hipDeviceGetUuid");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!uuid) return record(s, hipErrorInvalidValue);
  if (!valid_device(s, ordinal)) return record(s, hipErrorInvalidDevice);
  vgpu::amd::abi::DevicePropR0600 p;
  fill_properties(s.rt->device(ordinal).profile(), ordinal, &p);
  *uuid = p.uuid;
  return record(s, hipSuccess);
}

// The device that best matches the properties: the devices of a rack here
// are identical, so the first.
hipError_t hipChooseDeviceR0600(int* ordinal, const vgpu::amd::abi::DevicePropR0600* props) {
  const ApiCall api("hipChooseDeviceR0600");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ordinal || !props) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  *ordinal = 0;
  return record(s, hipSuccess);
}
hipError_t hipChooseDeviceR0000(int* ordinal, const vgpu::amd::abi::DevicePropR0000* props) {
  const ApiCall api("hipChooseDeviceR0000");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ordinal || !props) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  *ordinal = 0;
  return record(s, hipSuccess);
}
hipError_t hipChooseDevice(int* ordinal, const vgpu::amd::abi::DevicePropR0600* props) {
  return hipChooseDeviceR0600(ordinal, props);
}

// The device's gfx version, as the properties' major and minor give it.
hipError_t hipDeviceComputeCapability(int* major, int* minor, int ordinal) {
  const ApiCall api("hipDeviceComputeCapability");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!major || !minor) return record(s, hipErrorInvalidValue);
  if (!valid_device(s, ordinal)) return record(s, hipErrorInvalidDevice);
  vgpu::amd::abi::DevicePropR0600 p;
  fill_properties(s.rt->device(ordinal).profile(), ordinal, &p);
  *major = p.major;
  *minor = p.minor;
  return record(s, hipSuccess);
}

// Caches and LDS banks as a program may ask about them: no preference, and
// four-byte banks, whatever it set (hipDeviceSetCacheConfig accepts any).
hipError_t hipDeviceGetCacheConfig(int* config) {
  const ApiCall api("hipDeviceGetCacheConfig");
  if (!config) return record(state(), hipErrorInvalidValue);
  *config = 0;   // hipFuncCachePreferNone
  return record(state(), hipSuccess);
}
hipError_t hipDeviceGetSharedMemConfig(int* config) {
  const ApiCall api("hipDeviceGetSharedMemConfig");
  if (!config) return record(state(), hipErrorInvalidValue);
  *config = 1;   // hipSharedMemBankSizeFourByte
  return record(state(), hipSuccess);
}
hipError_t hipDeviceSetSharedMemConfig(int config) {
  const ApiCall api("hipDeviceSetSharedMemConfig");
  return record(state(), config < 0 || config > 2 ? hipErrorInvalidValue : hipSuccess);
}
hipError_t hipFuncSetSharedMemConfig(const void* function, int config) {
  const ApiCall api("hipFuncSetSharedMemConfig");
  if (!function) return record(state(), hipErrorInvalidDeviceFunction);
  return record(state(), config < 0 || config > 2 ? hipErrorInvalidValue : hipSuccess);
}

// How two devices are joined, as HSA gives it and ROCm SMI's
// rsmi_topo_get_link_type does (rocm_smi.cpp): Instinct GPUs one Infinity
// Fabric (XGMI, 4) hop apart, Radeon GPUs two PCIe (2) hops, through the host.
hipError_t hipExtGetLinkTypeAndHopCount(int a, int b, uint32_t* link_type, uint32_t* hops) {
  const ApiCall api("hipExtGetLinkTypeAndHopCount");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!link_type || !hops || a == b || a < 0 || b < 0) return record(s, hipErrorInvalidValue);
  if (!valid_device(s, a) || !valid_device(s, b)) return record(s, hipErrorInvalidDevice);
  const auto instinct = [&](int d) { return s.rt->device(d).profile().gcn_arch.rfind("gfx9", 0) == 0; };
  const bool xgmi = instinct(a) && instinct(b);
  *link_type = xgmi ? 4 : 2;
  *hops = xgmi ? 1 : 2;
  return record(s, hipSuccess);
}

// What one device may do with another's memory: reach it and use atomics on
// it, at the one rank there is; arrays there are none of on MI300.
hipError_t hipDeviceGetP2PAttribute(int* value, int attr, int src, int dst) {
  const ApiCall api("hipDeviceGetP2PAttribute");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!value || attr < 0 || attr > 3) return record(s, hipErrorInvalidValue);
  if (!valid_device(s, src) || !valid_device(s, dst) || src == dst) return record(s, hipErrorInvalidDevice);
  vgpu::amd::abi::DevicePropR0600 p;
  fill_properties(s.rt->device(src).profile(), src, &p);
  const int image = s.rt->device(src).profile().gcn_arch.rfind("gfx9", 0) == 0 ? 0 : 1;
  *value = attr == 0 ? 0 : attr == 3 ? image : 1;
  return record(s, hipSuccess);
}

// A stream's number: its own (see Stream::id), and 0 for the legacy default
// stream the null handle and hipStreamLegacy both name.
hipError_t hipStreamGetId(hipStream_t stream, unsigned long long* id) {
  const ApiCall api("hipStreamGetId");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!id) return record(s, hipErrorInvalidValue);
  const hipStream_t r = resolve(s, stream);
  if (!r) {
    *id = 0;
    return record(s, hipSuccess);
  }
  const auto it = s.streams.find(r);
  if (it == s.streams.end()) return record(s, hipErrorInvalidHandle);
  *id = it->second.id;
  return record(s, hipSuccess);
}

// A stream's attributes, kept as set (none changes how it runs here); its
// priority is the one it was made with. The synchronization policy is Auto
// (1) until set, and one of Auto, Spin, Yield or BlockingSync.
namespace {
hipError_t stream_attribute(hipStream_t stream, int attr, vgpu::amd::abi::LaunchAttributeValue* get,
                            const vgpu::amd::abi::LaunchAttributeValue* set) {
  using namespace vgpu::amd::abi;
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  const hipStream_t r = resolve(s, stream);
  Stream* st = nullptr;
  if (r) {
    const auto it = s.streams.find(r);
    if (it == s.streams.end()) return hipErrorInvalidHandle;
    st = &it->second;
  }
  if (attr != kLaunchAttributeAccessPolicyWindow && attr != kLaunchAttributeSynchronizationPolicy &&
      attr != kLaunchAttributePriority && attr != kLaunchAttributeMemSyncDomainMap &&
      attr != kLaunchAttributeMemSyncDomain)
    return hipErrorInvalidValue;
  if (!get && !set) return hipErrorInvalidValue;
  static std::map<int, LaunchAttributeValue> null_stream_attrs;
  auto& attrs = st ? st->attrs : null_stream_attrs;
  if (set) {
    if (attr == kLaunchAttributeSynchronizationPolicy && (set->sync_policy < 1 || set->sync_policy > 4))
      return hipErrorInvalidValue;
    attrs[attr] = *set;
    if (attr == kLaunchAttributePriority && st) st->priority = std::clamp(set->priority, -1, 1);
    return hipSuccess;
  }
  std::memset(get, 0, sizeof *get);
  if (const auto it = attrs.find(attr); it != attrs.end()) *get = it->second;
  else if (attr == kLaunchAttributeSynchronizationPolicy) get->sync_policy = 1;
  if (attr == kLaunchAttributePriority) get->priority = st ? st->priority : 0;
  return hipSuccess;
}
}  // namespace
hipError_t hipStreamGetAttribute(hipStream_t stream, int attr, vgpu::amd::abi::LaunchAttributeValue* value) {
  const ApiCall api("hipStreamGetAttribute");
  return record(state(), value ? stream_attribute(stream, attr, value, nullptr) : hipErrorInvalidValue);
}
hipError_t hipStreamSetAttribute(hipStream_t stream, int attr, const vgpu::amd::abi::LaunchAttributeValue* value) {
  const ApiCall api("hipStreamSetAttribute");
  if (!value) {
    // The stream is still checked first, as ROCm's HIP checks it.
    vgpu::amd::abi::LaunchAttributeValue v{};
    const hipError_t e = stream_attribute(stream, attr, &v, nullptr);
    return record(state(), e == hipErrorInvalidHandle ? e : hipErrorInvalidValue);
  }
  return record(state(), stream_attribute(stream, attr, nullptr, value));
}

// Pinned host memory by its older names. hipHostAlloc takes the flags
// hipHostMalloc does, less the coherence and NUMA ones, which it refuses.
hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags) {
  const ApiCall api("hipHostAlloc");
  constexpr unsigned kAllowed = 0x1 /* Portable */ | 0x2 /* Mapped */ | 0x4 /* WriteCombined */ |
                                0x10000000 /* Uncached */;
  if (!ptr || (flags & ~kAllowed)) return record(state(), hipErrorInvalidValue);
  return host_alloc(ptr, size, g_host_allocations, flags);
}
hipError_t hipMallocHost(void** ptr, size_t size) {
  const ApiCall api("hipMallocHost");
  return host_alloc(ptr, size, g_host_allocations);
}
hipError_t hipMemAllocHost(void** ptr, size_t size) {
  const ApiCall api("hipMemAllocHost");
  return host_alloc(ptr, size, g_host_allocations);
}
hipError_t hipFreeHost(void* ptr) { return hipHostFree(ptr); }
// The flags pinned memory was made with, as they were given: found by any
// address in it (on AMD a device pointer to it is the same address).
hipError_t hipHostGetFlags(unsigned int* flags, void* ptr) {
  const ApiCall api("hipHostGetFlags");
  if (!flags || !ptr) return record(state(), hipErrorInvalidValue);
  std::lock_guard<std::mutex> host_lock(g_host_mutex);
  const auto it = find_range(g_host_allocations, reinterpret_cast<uint64_t>(ptr));
  if (it == g_host_allocations.end()) return record(state(), hipErrorInvalidValue);
  *flags = g_host_flags[it->first];
  return record(state(), hipSuccess);
}

// The devices this thread may use, the first of them made current -- unless
// the thread already chose one with hipSetDevice. Only the first `len`
// entries are read.
hipError_t hipSetValidDevices(int* devices, int len) {
  const ApiCall api("hipSetValidDevices");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (len < 0 || len > s.rt->device_count() || (len && !devices)) return record(s, hipErrorInvalidValue);
  for (int i = 0; i < len; ++i)
    if (!valid_device(s, devices[i])) return record(s, hipErrorInvalidDevice);
  if (len && !s.device_chosen) s.current = devices[0];
  return record(s, hipSuccess);
}

// The profiler switches, which ROCm documents as not supported.
hipError_t hipProfilerStart(void) {
  const ApiCall api("hipProfilerStart");
  return record(state(), hipErrorNotSupported);
}
hipError_t hipProfilerStop(void) {
  const ApiCall api("hipProfilerStop");
  return record(state(), hipErrorNotSupported);
}

// A function's name by its trace id (hip_api_names.inc), "unknown" for one
// there is none of.
const char* hipApiName(uint32_t id) {
  static const std::map<uint32_t, const char*> names = [] {
    std::map<uint32_t, const char*> m;
#define N(id, name) m.emplace(id, name);
#include "hip_api_names.inc"
#undef N
    return m;
  }();
  const auto it = names.find(id);
  return it == names.end() ? "unknown" : it->second;
}
// What kind of command an activity record was: none are recorded here.
const char* hipGetCmdName(unsigned int) { return "unknown"; }

// ---- Waiting on and writing memory in stream order ----------------------------
//
// A stream can write a 32- or 64-bit value to memory, and wait until a value
// there passes a test, holding back everything after it on the stream -- and
// nothing on any other. The memory may be device memory, or host memory
// registered or pinned (which every device reaches at its host address).
// A handle that is no stream is a destroyed context to ROCm's HIP.
}  // extern "C"
namespace {
hipError_t memory_at(State& s, const void* ptr, unsigned align, vgpu::MemoryManager** mem) {   // holds s.mutex
  if (!ptr || reinterpret_cast<uint64_t>(ptr) % align) return hipErrorInvalidValue;
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return e;
  const int d = owner_of(s, reinterpret_cast<uint64_t>(ptr));
  if (d < 0) return hipErrorInvalidValue;
  *mem = &s.rt->device(d).memory();
  return hipSuccess;
}
bool known_stream(State& s, hipStream_t stream) {   // holds s.mutex
  Stream null_stream;
  return find_stream(s, stream, &null_stream) != nullptr;
}
hipError_t write_value(hipStream_t stream, void* ptr, uint64_t value, unsigned bytes, unsigned flags) {
  State& s = state();
  vgpu::MemoryManager* mem = nullptr;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!known_stream(s, stream)) return hipErrorContextIsDestroyed;
    if (flags) return hipErrorInvalidValue;
    if (const hipError_t e = memory_at(s, ptr, bytes, &mem); e != hipSuccess) return e;
  }
  return in_order(stream, [mem, ptr, value, bytes] {
    try {
      mem->write(reinterpret_cast<uint64_t>(ptr), &value, bytes);
    } catch (const std::exception& e) {
      return fail(hipErrorInvalidValue, e.what());
    }
    return hipSuccess;
  });
}
// Whether `v` passes: hipStreamWaitValueGte, Eq, And or Nor, under `mask`,
// at the value's own width.
bool passes(uint64_t v, uint64_t value, uint64_t mask, unsigned flags, unsigned bytes) {
  const uint64_t width = bytes == 8 ? ~0ull : 0xFFFFFFFFull;
  v &= mask & width;
  switch (flags) {
    case 0: return v >= value;
    case 1: return v == value;
    case 2: return (v & value) != 0;
    default: return (~(v | value) & width) != 0;
  }
}
hipError_t wait_value(hipStream_t stream, void* ptr, uint64_t value, uint64_t mask, unsigned bytes, unsigned flags) {
  State& s = state();
  vgpu::MemoryManager* mem = nullptr;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!known_stream(s, stream)) return hipErrorContextIsDestroyed;
    if (flags > 3) return hipErrorInvalidValue;
    if (const hipError_t e = memory_at(s, ptr, bytes, &mem); e != hipSuccess) return e;
  }
  // The stream's own thread waits: its later work stays behind, and every
  // other stream goes on.
  return in_order(stream, [mem, ptr, value, mask, bytes, flags] {
    for (;;) {
      uint64_t v = 0;
      try {
        mem->read(reinterpret_cast<uint64_t>(ptr), &v, bytes);
      } catch (const std::exception& e) {
        return fail(hipErrorInvalidValue, e.what());
      }
      if (passes(v, value, mask, flags, bytes)) return hipSuccess;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  });
}
}  // namespace
extern "C" {

hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags) {
  const ApiCall api("hipStreamWriteValue64");
  return record(state(), write_value(stream, ptr, value, 8, flags));
}
hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags, uint32_t mask) {
  const ApiCall api("hipStreamWaitValue32");
  return record(state(), wait_value(stream, ptr, value, mask, 4, flags));
}
hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags, uint64_t mask) {
  const ApiCall api("hipStreamWaitValue64");
  return record(state(), wait_value(stream, ptr, value, mask, 8, flags));
}

// Up to 256 of those in one call, in order: waits and writes of either
// width. A barrier or a flush of remote writes is not supported on AMD. The
// null stream is refused.
struct StreamMemOp {
  int operation;
  void* address;
  union {
    uint32_t value;
    uint64_t value64;
  };
  unsigned int flags;
  void* alias;
};
static_assert(sizeof(StreamMemOp) <= 48, "hipStreamBatchMemOpParams is six words");
hipError_t hipStreamBatchMemOp(hipStream_t stream, unsigned int count, void* params, unsigned int flags) {
  const ApiCall api("hipStreamBatchMemOp");
  {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!stream) return record(s, hipErrorInvalidValue);
    if (!known_stream(s, stream)) return record(s, hipErrorContextIsDestroyed);
    if (!params || !count || count > 256 || flags) return record(s, hipErrorInvalidValue);
  }
  for (unsigned i = 0; i < count; ++i) {
    const auto* op = reinterpret_cast<const StreamMemOp*>(static_cast<const uint8_t*>(params) + 48 * size_t{i});
    hipError_t e = hipSuccess;
    switch (op->operation) {
      case 1: e = wait_value(stream, op->address, op->value, 0xFFFFFFFFu, 4, op->flags); break;
      case 2: e = write_value(stream, op->address, op->value, 4, 0); break;
      case 4: e = wait_value(stream, op->address, op->value64, ~0ull, 8, op->flags); break;
      case 5: e = write_value(stream, op->address, op->value64, 8, 0); break;
      case 3:
      case 6: e = hipErrorNotSupported; break;
      default: e = hipErrorInvalidValue;
    }
    if (e != hipSuccess) return record(state(), e);
  }
  return record(state(), hipSuccess);
}

// ---- Kernels named by their host-side functions -------------------------------

// A module function for a kernel of the program's own, on the current
// device: what hipModuleLaunchKernel takes, one per kernel and device.
hipError_t hipGetFuncBySymbol(hipFunction_t* function, const void* symbol) {
  const ApiCall api("hipGetFuncBySymbol");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!function || !symbol) return record(s, hipErrorInvalidValue);
  if (!device(s)) return record(s, hipErrorInvalidDevice);
  const auto hf = s.host_functions.find(symbol);
  if (hf == s.host_functions.end()) return record(s, hipErrorInvalidDeviceFunction);
  Module* m = nullptr;
  if (const hipError_t e = module_on(s, *hf->second.binary, s.current, &m); e != hipSuccess) return record(s, e);
  const Kernel* k = vgpu::amd::find_kernel(m->object, hf->second.kernel);
  if (!k) return record(s, hipErrorInvalidDeviceFunction);
  for (const auto& f : s.functions)
    if (f->module == m && f->kernel == k) {
      *function = reinterpret_cast<hipFunction_t>(f.get());
      return record(s, hipSuccess);
    }
  auto f = std::make_unique<Function>();
  f->module = m;
  f->kernel = k;
  *function = reinterpret_cast<hipFunction_t>(f.get());
  s.functions.push_back(std::move(f));
  return record(s, hipSuccess);
}
// The kernels a module has for its device.
hipError_t hipModuleGetFunctionCount(unsigned int* count, hipModule_t module) {
  const ApiCall api("hipModuleGetFunctionCount");
  if (!module) return record(state(), hipErrorInvalidHandle);
  if (!count) return record(state(), hipErrorInvalidValue);
  *count = static_cast<unsigned>(reinterpret_cast<const Module*>(module)->object.kernels.size());
  return record(state(), hipSuccess);
}

// A launch whose configuration comes as a structure with attributes: the
// cooperative one makes it a cooperative launch; the others (access policy,
// synchronization, priority) change nothing here.
hipError_t hipLaunchKernelExC(const vgpu::amd::abi::LaunchConfig* config, const void* function, void** args) {
  const ApiCall api("hipLaunchKernelExC");
  if (!function) return record(state(), hipErrorInvalidDeviceFunction);
  if (!config || (config->num_attrs && !config->attrs)) return record(state(), hipErrorInvalidValue);
  bool cooperative = false;
  for (unsigned i = 0; i < config->num_attrs; ++i)
    if (config->attrs[i].id == vgpu::amd::abi::kLaunchAttributeCooperative)
      cooperative = config->attrs[i].value.cooperative != 0;
  return record(state(), launch_host_function(function, config->grid, config->block, args, nullptr,
                                              config->dynamic_shared, static_cast<hipStream_t>(config->stream),
                                              cooperative));
}

// The oldest launch API: a configuration, then each argument copied into
// place, then the launch. The configuration goes on the stack chevron
// launches use; the arguments into a buffer of this thread's.
namespace {
thread_local std::vector<uint8_t> t_setup_args;
}
hipError_t hipConfigureCall(vgpu::amd::abi::Dim3 grid, vgpu::amd::abi::Dim3 block, size_t shared, hipStream_t stream) {
  const ApiCall api("hipConfigureCall");
  call_configurations().push_back({grid, block, shared, stream});
  t_setup_args.clear();
  return hipSuccess;
}
hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset) {
  const ApiCall api("hipSetupArgument");
  if (!arg && size) return record(state(), hipErrorInvalidValue);
  if (t_setup_args.size() < offset + size) t_setup_args.resize(offset + size);
  if (size) std::memcpy(t_setup_args.data() + offset, arg, size);
  return hipSuccess;
}
hipError_t hipLaunchByPtr(const void* function) {
  const ApiCall api("hipLaunchByPtr");
  if (call_configurations().empty()) return record(state(), hipErrorInvalidConfiguration);
  const CallConfiguration c = call_configurations().back();
  call_configurations().pop_back();
  std::vector<uint8_t> args = std::move(t_setup_args);
  t_setup_args.clear();
  if (!function) return record(state(), hipErrorInvalidDeviceFunction);
  size_t size = args.size();
  void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                   HIP_LAUNCH_PARAM_END};
  return record(state(), launch_host_function(function, c.grid, c.block, nullptr, args.empty() ? nullptr : extra,
                                              c.shared, c.stream, false));
}

// One kernel launched on each of several devices, each entry on its own
// device's stream. Two entries on one device are refused, as is a list longer
// than the devices there are, or a flag other than NoPreSync and NoPostSync.
namespace {
hipError_t launch_on_devices(const vgpu::amd::abi::LaunchParams* list, int n, unsigned flags, bool cooperative) {
  State& s = state();
  std::vector<int> devices;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!list || n <= 0) return hipErrorInvalidValue;
    if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return e;
    for (int i = 0; i < n; ++i) {
      Stream null_stream;
      const Stream* st = find_stream(s, static_cast<hipStream_t>(list[i].stream), &null_stream);
      if (!st) return hipErrorInvalidHandle;
      if (std::find(devices.begin(), devices.end(), st->device) != devices.end()) return hipErrorInvalidDevice;
      devices.push_back(st->device);
    }
    if (n > s.rt->device_count() || (flags & ~3u)) return hipErrorInvalidValue;
  }
  const int was = s.current;
  for (int i = 0; i < n; ++i) {
    s.current = devices[static_cast<size_t>(i)];
    const hipError_t e = launch_host_function(list[i].func, list[i].grid, list[i].block, list[i].args, nullptr,
                                              list[i].shared, static_cast<hipStream_t>(list[i].stream), cooperative);
    if (e != hipSuccess) {
      s.current = was;
      return e;
    }
  }
  s.current = was;
  return hipSuccess;
}
}  // namespace
hipError_t hipExtLaunchMultiKernelMultiDevice(vgpu::amd::abi::LaunchParams* list, int n, unsigned int flags) {
  const ApiCall api("hipExtLaunchMultiKernelMultiDevice");
  return record(state(), launch_on_devices(list, n, flags, false));
}
hipError_t hipLaunchCooperativeKernelMultiDevice(vgpu::amd::abi::LaunchParams* list, int n, unsigned int flags) {
  const ApiCall api("hipLaunchCooperativeKernelMultiDevice");
  return record(state(), launch_on_devices(list, n, flags, true));
}
// The same with module functions, which must all be the same kernel with the
// same configuration, each launched on the device its module was loaded for,
// on a stream of that device.
hipError_t hipModuleLaunchCooperativeKernelMultiDevice(vgpu::amd::abi::FunctionLaunchParams* list, unsigned int n,
                                                       unsigned int flags) {
  const ApiCall api("hipModuleLaunchCooperativeKernelMultiDevice");
  State& s = state();
  std::vector<int> devices;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
    if (!list || !n || n > static_cast<unsigned>(s.rt->device_count()) || (flags & ~3u))
      return record(s, hipErrorInvalidValue);
    for (unsigned i = 0; i < n; ++i) {
      const auto& p = list[i];
      if (!p.function) return record(s, hipErrorInvalidResourceHandle);
      if (i && (p.grid_x != list[0].grid_x || p.grid_y != list[0].grid_y || p.grid_z != list[0].grid_z ||
                p.block_x != list[0].block_x || p.block_y != list[0].block_y || p.block_z != list[0].block_z ||
                p.shared != list[0].shared))
        return record(s, hipErrorInvalidValue);
      Stream null_stream;
      const Stream* st = find_stream(s, static_cast<hipStream_t>(p.stream), &null_stream);
      if (!st) return record(s, hipErrorInvalidHandle);
      const int d = reinterpret_cast<const Function*>(p.function)->module->device;
      if (d != st->device || std::find(devices.begin(), devices.end(), d) != devices.end())
        return record(s, hipErrorInvalidDevice);
      devices.push_back(d);
    }
  }
  const int was = s.current;
  for (unsigned i = 0; i < n; ++i) {
    const auto& p = list[i];
    s.current = devices[i];
    const hipError_t e = hipModuleLaunchCooperativeKernel(
        static_cast<hipFunction_t>(p.function), p.grid_x, p.grid_y, p.grid_z, p.block_x, p.block_y, p.block_z,
        p.shared, static_cast<hipStream_t>(p.stream), p.params);
    if (e != hipSuccess) {
      s.current = was;
      return e;
    }
  }
  s.current = was;
  return record(s, hipSuccess);
}

// Occupancy of a module function with flags: the default, or the one to
// disable a caching override, which changes nothing here.
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* blocks, hipFunction_t f, int block,
                                                                       size_t lds, unsigned int flags) {
  if (flags > 1) return record(state(), hipErrorInvalidValue);
  return hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(blocks, f, block, lds);
}
hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int* grid, int* block, hipFunction_t f, size_t lds,
                                                            int limit, unsigned int flags) {
  if (flags > 1) return record(state(), hipErrorInvalidValue);
  return hipModuleOccupancyMaxPotentialBlockSize(grid, block, f, lds, limit);
}

// ---- Pointer attributes, set and got in bulk ---------------------------------

// Only HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS can be set: it makes every copy
// between device allocations with it synchronous (see copy_in_order).
hipError_t hipPointerSetAttribute(const void* value, int attribute, void* ptr) {
  const ApiCall api("hipPointerSetAttribute");
  using K = vgpu::amd::abi::PointerAttributeKind;
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  // HIP_POINTER_ATTRIBUTE_CONTEXT (1) to _MEMPOOL_HANDLE (17) are the ones there are.
  if (!value || !ptr || attribute < 1 || attribute > 17) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  uint64_t base = 0, size = 0;
  bool found = false;
  for (int d = 0; d < s.rt->device_count() && !found; ++d)
    found = s.rt->device(d).memory().find_allocation(reinterpret_cast<uint64_t>(ptr), &base, &size);
  if (!found) return record(s, hipErrorInvalidDevicePointer);
  if (attribute != K::kPointerSyncMemops) return record(s, hipErrorNotSupported);
  if (*static_cast<const int*>(value)) s.sync_memops.insert(base);
  else s.sync_memops.erase(base);
  return record(s, hipSuccess);
}
hipError_t hipDrvPointerGetAttributes(unsigned int count, int* attributes, void** data, void* ptr) {
  const ApiCall api("hipDrvPointerGetAttributes");
  if (!count || !attributes || !data || !ptr) return record(state(), hipErrorInvalidValue);
  for (unsigned i = 0; i < count; ++i) {
    const hipError_t e = hipPointerGetAttribute(data[i], attributes[i], ptr);
    // Device memory's host pointer is null here, where on its own it is an error.
    if (e == hipErrorInvalidValue && attributes[i] == vgpu::amd::abi::PointerAttributeKind::kPointerHostPointer) {
      *static_cast<void**>(data[i]) = nullptr;
      (void)hipGetLastError();
      continue;
    }
    if (e != hipSuccess) return e;
  }
  return record(state(), hipSuccess);
}

// ---- Copies, fills and allocations of every shape -------------------------------
//
// The driver API's forms, the 2D and 3D ones, pitched allocations and the
// memsets of each width: all of them regions (see "Copies and fills of every
// shape"), answered as ROCm's HIP answers them -- AMD's hip-tests are the
// reference for each error. This device has no image units, so a copy to or
// from an array is refused as not supported, once its arguments are checked.

hipError_t hipMemcpyHtoD(void* dst, const void* src, size_t bytes) {
  const ApiCall api("hipMemcpyHtoD");
  return record(state(), copy_in_order(dst, src, bytes, hipMemcpyHostToDevice, nullptr, false));
}
hipError_t hipMemcpyDtoH(void* dst, const void* src, size_t bytes) {
  const ApiCall api("hipMemcpyDtoH");
  return record(state(), copy_in_order(dst, src, bytes, hipMemcpyDeviceToHost, nullptr, false));
}
hipError_t hipMemcpyDtoD(void* dst, const void* src, size_t bytes) {
  const ApiCall api("hipMemcpyDtoD");
  return record(state(), copy_in_order(dst, src, bytes, hipMemcpyDeviceToDevice, nullptr, false));
}
hipError_t hipMemcpyHtoDAsync(void* dst, const void* src, size_t bytes, hipStream_t stream) {
  const ApiCall api("hipMemcpyHtoDAsync");
  return record(state(), copy_in_order(dst, src, bytes, hipMemcpyHostToDevice, stream, true));
}
hipError_t hipMemcpyDtoHAsync(void* dst, const void* src, size_t bytes, hipStream_t stream) {
  const ApiCall api("hipMemcpyDtoHAsync");
  return record(state(), copy_in_order(dst, src, bytes, hipMemcpyDeviceToHost, stream, true));
}
hipError_t hipMemcpyDtoDAsync(void* dst, const void* src, size_t bytes, hipStream_t stream) {
  const ApiCall api("hipMemcpyDtoDAsync");
  return record(state(), copy_in_order(dst, src, bytes, hipMemcpyDeviceToDevice, stream, true));
}

// A copy between devices. The devices have to be devices; where the bytes
// are is what the addresses say, as ROCm's HIP copies them (a program may
// name a device the memory is not on). No peer access is needed: every
// device reaches every other's memory here.
}  // extern "C"
namespace {
hipError_t peer_copy(void* dst, int dst_device, const void* src, int src_device, size_t bytes, hipStream_t stream,
                     bool async) {
  {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return e;
    const int n = s.rt->device_count();
    if (dst_device < 0 || dst_device >= n || src_device < 0 || src_device >= n) return hipErrorInvalidDevice;
  }
  return copy_in_order(dst, src, bytes, hipMemcpyDeviceToDevice, stream, async);
}

// The widest row a 2D or 3D copy takes: hipDeviceAttributeMaxPitch.
constexpr size_t kMaxPitch = INT32_MAX;

// The pointer a driver-style copy's side names: its memory type says which
// field. An array is refused -- there are no image units -- unless it is
// null, which is the wrong value.
hipError_t side_pointer(int type, const void* host, void* dev, void* array, const void** out) {
  using namespace vgpu::amd::abi;
  switch (type) {
    case kMemTypeHost: *out = host; break;
    case kMemTypeDevice:
    case kMemTypeManaged:
    case kMemTypeUnified: *out = dev; break;
    case kMemTypeArray: return array ? hipErrorNotSupported : hipErrorInvalidValue;
    default: return hipErrorInvalidValue;
  }
  return *out ? hipSuccess : hipErrorInvalidValue;
}
// A row's pitch: none is as wide as the row; narrower than the row, or wider
// than any row may be, is the wrong value.
hipError_t row_pitch(size_t pitch, size_t width, size_t* out) {
  if (!pitch) pitch = width;
  if (pitch < width || pitch >= kMaxPitch) return hipErrorInvalidValue;
  *out = pitch;
  return hipSuccess;
}

// hipMemcpyParam2D and its kin.
hipError_t param_2d(const vgpu::amd::abi::Memcpy2D* p, hipStream_t stream, bool async) {
  if (!p) return hipErrorInvalidValue;
  const void *src = nullptr, *dst = nullptr;
  if (const hipError_t e = side_pointer(p->srcMemoryType, p->srcHost, p->srcDevice, p->srcArray, &src); e != hipSuccess)
    return e;
  if (const hipError_t e = side_pointer(p->dstMemoryType, p->dstHost, p->dstDevice, p->dstArray, &dst); e != hipSuccess)
    return e;
  size_t spitch = 0, dpitch = 0;
  if (const hipError_t e = row_pitch(p->srcPitch, p->WidthInBytes, &spitch); e != hipSuccess) return e;
  if (const hipError_t e = row_pitch(p->dstPitch, p->WidthInBytes, &dpitch); e != hipSuccess) return e;
  if (!p->WidthInBytes || !p->Height) return hipSuccess;
  const auto* s0 = static_cast<const uint8_t*>(src) + p->srcY * spitch + p->srcXInBytes;
  auto* d0 = const_cast<uint8_t*>(static_cast<const uint8_t*>(dst)) + p->dstY * dpitch + p->dstXInBytes;
  return copy_region(d0, Region{p->WidthInBytes, p->Height, 1, dpitch, 0}, s0,
                     Region{p->WidthInBytes, p->Height, 1, spitch, 0}, hipMemcpyDefault, stream, async);
}

// hipMemcpy3D and its kin: each side an array or a pitched pointer, one of
// the two. A slice is the pointer's ysize rows (the extent's height where it
// says none).
hipError_t copy_3d(const vgpu::amd::abi::Memcpy3DParms* p, hipStream_t stream, bool async) {
  if (!p) return hipErrorInvalidValue;
  const bool sarr = p->srcArray != nullptr, darr = p->dstArray != nullptr;
  const bool sptr = p->srcPtr.ptr != nullptr, dptr = p->dstPtr.ptr != nullptr;
  if (sarr == sptr || darr == dptr) return hipErrorInvalidValue;
  const int kind = p->kind;
  if (kind != hipMemcpyDefault && kind != hipMemcpyHostToHost && kind != hipMemcpyHostToDevice &&
      kind != hipMemcpyDeviceToHost && kind != hipMemcpyDeviceToDevice)
    return hipErrorInvalidMemcpyDirection;
  if (sarr || darr) return hipErrorNotSupported;
  const auto& e = p->extent;
  size_t spitch = 0, dpitch = 0;
  if (p->srcPtr.pitch >= kMaxPitch || p->dstPtr.pitch >= kMaxPitch) return hipErrorInvalidValue;
  spitch = p->srcPtr.pitch ? p->srcPtr.pitch : e.width;
  dpitch = p->dstPtr.pitch ? p->dstPtr.pitch : e.width;
  if (!e.width || !e.height || !e.depth) return hipSuccess;
  const uint64_t sslice = spitch * (p->srcPtr.ysize ? p->srcPtr.ysize : e.height);
  const uint64_t dslice = dpitch * (p->dstPtr.ysize ? p->dstPtr.ysize : e.height);
  const auto* s0 = static_cast<const uint8_t*>(p->srcPtr.ptr) + p->srcPos.z * sslice + p->srcPos.y * spitch + p->srcPos.x;
  auto* d0 = static_cast<uint8_t*>(p->dstPtr.ptr) + p->dstPos.z * dslice + p->dstPos.y * dpitch + p->dstPos.x;
  return copy_region(d0, Region{e.width, e.height, e.depth, dpitch, dslice}, s0,
                     Region{e.width, e.height, e.depth, spitch, sslice}, static_cast<hipMemcpyKind>(kind), stream,
                     async);
}

// hipDrvMemcpy3D and its kin: the driver API's 3D copy, whose sides are
// named by memory type, and whose slice is Height rows unless a side says.
hipError_t drv_3d(const vgpu::amd::abi::Memcpy3D* p, hipStream_t stream, bool async) {
  if (!p) return hipErrorInvalidValue;
  const void *src = nullptr, *dst = nullptr;
  if (const hipError_t e = side_pointer(p->srcMemoryType, p->srcHost, p->srcDevice, p->srcArray, &src); e != hipSuccess)
    return e;
  if (const hipError_t e = side_pointer(p->dstMemoryType, p->dstHost, p->dstDevice, p->dstArray, &dst); e != hipSuccess)
    return e;
  size_t spitch = 0, dpitch = 0;
  if (const hipError_t e = row_pitch(p->srcPitch, p->WidthInBytes, &spitch); e != hipSuccess) return e;
  if (const hipError_t e = row_pitch(p->dstPitch, p->WidthInBytes, &dpitch); e != hipSuccess) return e;
  if (!p->WidthInBytes || !p->Height || !p->Depth) return hipSuccess;
  const uint64_t sslice = spitch * (p->srcHeight ? p->srcHeight : p->Height);
  const uint64_t dslice = dpitch * (p->dstHeight ? p->dstHeight : p->Height);
  const auto* s0 = static_cast<const uint8_t*>(src) + p->srcZ * sslice + p->srcY * spitch + p->srcXInBytes;
  auto* d0 = const_cast<uint8_t*>(static_cast<const uint8_t*>(dst)) + p->dstZ * dslice + p->dstY * dpitch +
             p->dstXInBytes;
  return copy_region(d0, Region{p->WidthInBytes, p->Height, p->Depth, dpitch, dslice}, s0,
                     Region{p->WidthInBytes, p->Height, p->Depth, spitch, sslice}, hipMemcpyDefault, stream, async);
}

// hipMemcpy3DPeer: the same, with a device for each side.
hipError_t peer_3d(const vgpu::amd::abi::Memcpy3DPeerParms* p, hipStream_t stream, bool async) {
  if (!p) return hipErrorInvalidValue;
  {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return e;
    const int n = s.rt->device_count();
    if (p->srcDevice < 0 || p->srcDevice >= n || p->dstDevice < 0 || p->dstDevice >= n) return hipErrorInvalidDevice;
  }
  vgpu::amd::abi::Memcpy3DParms q{};
  q.srcArray = p->srcArray;
  q.srcPos = p->srcPos;
  q.srcPtr = p->srcPtr;
  q.dstArray = p->dstArray;
  q.dstPos = p->dstPos;
  q.dstPtr = p->dstPtr;
  q.extent = p->extent;
  q.kind = hipMemcpyDefault;
  return copy_3d(&q, stream, async);
}

// Memory whose rows are `pitch` bytes apart: each row rounded up to 256
// bytes, as ROCm's HIP rounds it, and the allocation exactly pitch * height
// * depth. None at all (a zero dimension) is no allocation.
hipError_t pitched_alloc(void** ptr, size_t* pitch, size_t width, size_t height, size_t depth) {
  if (!ptr || !pitch) return hipErrorInvalidValue;
  if (!width || !height || !depth) {
    *ptr = nullptr;
    *pitch = 0;
    return hipSuccess;
  }
  if (width > SIZE_MAX - 255) return hipErrorOutOfMemory;
  const size_t p = (width + 255) / 256 * 256;
  size_t bytes = 0;
  if (__builtin_mul_overflow(p, height, &bytes) || __builtin_mul_overflow(bytes, depth, &bytes))
    return hipErrorOutOfMemory;
  if (const hipError_t e = hipMalloc(ptr, bytes); e != hipSuccess) return e;
  *pitch = p;
  return hipSuccess;
}

// A memset of `rows` rows (and `depth` slices) of `width` bytes, with a
// pattern of 1, 2 or 4 bytes.
hipError_t set_2d(void* dst, size_t pitch, const void* pattern, uint32_t pattern_len, size_t width, size_t height,
                  hipStream_t stream, bool async) {
  if (!width || !height) return hipSuccess;
  if (!dst || pitch < width) return hipErrorInvalidValue;
  return fill_region(dst, Region{width, height, 1, pitch, 0}, static_cast<const uint8_t*>(pattern), pattern_len,
                     stream, async);
}
hipError_t set_3d(const vgpu::amd::abi::PitchedPtr& p, int value, const vgpu::amd::abi::Extent& e, hipStream_t stream,
                  bool async) {
  if (!e.width || !e.height || !e.depth) return hipSuccess;
  if (!p.ptr || p.pitch < e.width) return hipErrorInvalidValue;
  uint64_t slice = 0;
  if (__builtin_mul_overflow(p.pitch, p.ysize ? p.ysize : e.height, &slice)) return hipErrorInvalidValue;
  const uint8_t byte = static_cast<uint8_t>(value);
  return fill_region(p.ptr, Region{e.width, e.height, e.depth, p.pitch, slice}, &byte, 1, stream, async);
}
// The driver API's 2D memsets: `width` is elements of the pattern's size,
// and the pitch has to be at least that many. A row is written up to its
// pitch at most -- what a card writes past it lands in the next row, with
// the same pattern -- so the last row never runs past the allocation.
hipError_t set_d2d(void* dst, size_t pitch, const void* pattern, uint32_t size, size_t width, size_t height,
                   hipStream_t stream, bool async) {
  if (!dst) return hipErrorInvalidValue;
  if (!width || !height) return hipSuccess;
  if (pitch < width) return hipErrorInvalidValue;
  const size_t bytes = width > pitch / size ? pitch / size * size : width * size;
  return fill_region(dst, Region{bytes, height, 1, pitch, 0}, static_cast<const uint8_t*>(pattern), size, stream,
                     async);
}
}  // namespace
extern "C" {

hipError_t hipMemcpyPeer(void* dst, int dst_device, const void* src, int src_device, size_t bytes) {
  const ApiCall api("hipMemcpyPeer");
  return record(state(), peer_copy(dst, dst_device, src, src_device, bytes, nullptr, false));
}
hipError_t hipMemcpyParam2D(const vgpu::amd::abi::Memcpy2D* p) {
  const ApiCall api("hipMemcpyParam2D");
  return record(state(), param_2d(p, nullptr, false));
}
hipError_t hipMemcpyParam2DAsync(const vgpu::amd::abi::Memcpy2D* p, hipStream_t stream) {
  const ApiCall api("hipMemcpyParam2DAsync");
  return record(state(), param_2d(p, stream, true));
}
hipError_t hipDrvMemcpy2DUnaligned(const vgpu::amd::abi::Memcpy2D* p) {
  const ApiCall api("hipDrvMemcpy2DUnaligned");
  return record(state(), param_2d(p, nullptr, false));
}
hipError_t hipMemcpy3D(const vgpu::amd::abi::Memcpy3DParms* p) {
  const ApiCall api("hipMemcpy3D");
  return record(state(), copy_3d(p, nullptr, false));
}
hipError_t hipMemcpy3DAsync(const vgpu::amd::abi::Memcpy3DParms* p, hipStream_t stream) {
  const ApiCall api("hipMemcpy3DAsync");
  return record(state(), copy_3d(p, stream, true));
}
hipError_t hipDrvMemcpy3D(const vgpu::amd::abi::Memcpy3D* p) {
  const ApiCall api("hipDrvMemcpy3D");
  return record(state(), drv_3d(p, nullptr, false));
}
hipError_t hipDrvMemcpy3DAsync(const vgpu::amd::abi::Memcpy3D* p, hipStream_t stream) {
  const ApiCall api("hipDrvMemcpy3DAsync");
  return record(state(), drv_3d(p, stream, true));
}
hipError_t hipMemcpy3DPeer(vgpu::amd::abi::Memcpy3DPeerParms* p) {
  const ApiCall api("hipMemcpy3DPeer");
  return record(state(), peer_3d(p, nullptr, false));
}
hipError_t hipMemcpy3DPeerAsync(vgpu::amd::abi::Memcpy3DPeerParms* p, hipStream_t stream) {
  const ApiCall api("hipMemcpy3DPeerAsync");
  return record(state(), peer_3d(p, stream, true));
}
// Several 3D copies, in order on the stream, each operand a pointer with its
// row length (bytes) and slice height. The flags must be zero.
hipError_t hipMemcpy3DBatchAsync(size_t count, vgpu::amd::abi::Memcpy3DBatchOp* ops, size_t* fail_idx,
                                 unsigned long long flags, hipStream_t stream) {
  const ApiCall api("hipMemcpy3DBatchAsync");
  if (!count || !ops || flags) return record(state(), hipErrorInvalidValue);
  for (size_t i = 0; i < count; ++i) {
    const auto& op = ops[i];
    hipError_t e = hipSuccess;
    if (op.src.type != 1 || op.dst.type != 1) {
      e = op.src.type == 2 || op.dst.type == 2 ? hipErrorNotSupported : hipErrorInvalidValue;
    } else {
      const auto& x = op.extent;
      const uint64_t sp = op.src.op.ptr.rowLength ? op.src.op.ptr.rowLength : x.width;
      const uint64_t dp = op.dst.op.ptr.rowLength ? op.dst.op.ptr.rowLength : x.width;
      const uint64_t ss = sp * (op.src.op.ptr.layerHeight ? op.src.op.ptr.layerHeight : x.height);
      const uint64_t ds = dp * (op.dst.op.ptr.layerHeight ? op.dst.op.ptr.layerHeight : x.height);
      e = copy_region(op.dst.op.ptr.ptr, Region{x.width, x.height, x.depth, dp, ds}, op.src.op.ptr.ptr,
                      Region{x.width, x.height, x.depth, sp, ss}, hipMemcpyDefault, stream, true);
    }
    if (e != hipSuccess) {
      if (fail_idx) *fail_idx = i;
      return record(state(), e);
    }
  }
  return record(state(), hipSuccess);
}

hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height) {
  const ApiCall api("hipMallocPitch");
  return record(state(), pitched_alloc(ptr, pitch, width, height, 1));
}
hipError_t hipMemAllocPitch(void** ptr, size_t* pitch, size_t width, size_t height, unsigned int) {
  const ApiCall api("hipMemAllocPitch");
  return record(state(), pitched_alloc(ptr, pitch, width, height, 1));
}
hipError_t hipMalloc3D(vgpu::amd::abi::PitchedPtr* p, vgpu::amd::abi::Extent e) {
  const ApiCall api("hipMalloc3D");
  if (!p) return record(state(), hipErrorInvalidValue);
  void* ptr = nullptr;
  size_t pitch = 0;
  if (const hipError_t err = pitched_alloc(&ptr, &pitch, e.width, e.height, e.depth); err != hipSuccess)
    return record(state(), err);
  *p = {ptr, pitch, e.width, e.height};
  return record(state(), hipSuccess);
}

// The memsets by element width: bytes, 16-bit and 32-bit words.
hipError_t hipMemsetD8(void* dst, unsigned char value, size_t count) {
  const ApiCall api("hipMemsetD8");
  return record(state(), fill_in_order(dst, &value, 1, count, nullptr, false));
}
hipError_t hipMemsetD8Async(void* dst, unsigned char value, size_t count, hipStream_t stream) {
  const ApiCall api("hipMemsetD8Async");
  return record(state(), fill_in_order(dst, &value, 1, count, stream, true));
}
hipError_t hipMemsetD16(void* dst, unsigned short value, size_t count) {
  const ApiCall api("hipMemsetD16");
  return record(state(), fill_in_order(dst, reinterpret_cast<const uint8_t*>(&value), 2, count * 2, nullptr, false));
}
hipError_t hipMemsetD16Async(void* dst, unsigned short value, size_t count, hipStream_t stream) {
  const ApiCall api("hipMemsetD16Async");
  return record(state(), fill_in_order(dst, reinterpret_cast<const uint8_t*>(&value), 2, count * 2, stream, true));
}
hipError_t hipMemsetD32(void* dst, int value, size_t count) {
  const ApiCall api("hipMemsetD32");
  return record(state(), fill_in_order(dst, reinterpret_cast<const uint8_t*>(&value), 4, count * 4, nullptr, false));
}
hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height) {
  const ApiCall api("hipMemset2D");
  const uint8_t byte = static_cast<uint8_t>(value);
  return record(state(), set_2d(dst, pitch, &byte, 1, width, height, nullptr, false));
}
hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value, size_t width, size_t height, hipStream_t stream) {
  const ApiCall api("hipMemset2DAsync");
  const uint8_t byte = static_cast<uint8_t>(value);
  return record(state(), set_2d(dst, pitch, &byte, 1, width, height, stream, true));
}
hipError_t hipMemset3D(vgpu::amd::abi::PitchedPtr p, int value, vgpu::amd::abi::Extent e) {
  const ApiCall api("hipMemset3D");
  return record(state(), set_3d(p, value, e, nullptr, false));
}
hipError_t hipMemset3DAsync(vgpu::amd::abi::PitchedPtr p, int value, vgpu::amd::abi::Extent e, hipStream_t stream) {
  const ApiCall api("hipMemset3DAsync");
  return record(state(), set_3d(p, value, e, stream, true));
}
hipError_t hipMemsetD2D8(void* dst, size_t pitch, unsigned char value, size_t width, size_t height) {
  const ApiCall api("hipMemsetD2D8");
  return record(state(), set_d2d(dst, pitch, &value, 1, width, height, nullptr, false));
}
hipError_t hipMemsetD2D8Async(void* dst, size_t pitch, unsigned char value, size_t width, size_t height,
                              hipStream_t stream) {
  const ApiCall api("hipMemsetD2D8Async");
  return record(state(), set_d2d(dst, pitch, &value, 1, width, height, stream, true));
}
hipError_t hipMemsetD2D16(void* dst, size_t pitch, unsigned short value, size_t width, size_t height) {
  const ApiCall api("hipMemsetD2D16");
  return record(state(), set_d2d(dst, pitch, &value, 2, width, height, nullptr, false));
}
hipError_t hipMemsetD2D16Async(void* dst, size_t pitch, unsigned short value, size_t width, size_t height,
                               hipStream_t stream) {
  const ApiCall api("hipMemsetD2D16Async");
  return record(state(), set_d2d(dst, pitch, &value, 2, width, height, stream, true));
}
hipError_t hipMemsetD2D32(void* dst, size_t pitch, unsigned int value, size_t width, size_t height) {
  const ApiCall api("hipMemsetD2D32");
  return record(state(), set_d2d(dst, pitch, &value, 4, width, height, nullptr, false));
}
hipError_t hipMemsetD2D32Async(void* dst, size_t pitch, unsigned int value, size_t width, size_t height,
                               hipStream_t stream) {
  const ApiCall api("hipMemsetD2D32Async");
  return record(state(), set_d2d(dst, pitch, &value, 4, width, height, stream, true));
}

// Copies to and from arrays: there are none on this device.
hipError_t hipMemcpyAtoA(void* dst, size_t, void* src, size_t, size_t) {
  const ApiCall api("hipMemcpyAtoA");
  return record(state(), dst && src ? hipErrorNotSupported : hipErrorInvalidValue);
}
hipError_t hipMemcpyAtoD(void* dst, void* src, size_t, size_t) {
  const ApiCall api("hipMemcpyAtoD");
  return record(state(), dst && src ? hipErrorNotSupported : hipErrorInvalidValue);
}
hipError_t hipMemcpyAtoH(void* dst, void* src, size_t, size_t) {
  const ApiCall api("hipMemcpyAtoH");
  return record(state(), dst && src ? hipErrorNotSupported : hipErrorInvalidValue);
}
hipError_t hipMemcpyAtoHAsync(void* dst, void* src, size_t, size_t, hipStream_t) {
  const ApiCall api("hipMemcpyAtoHAsync");
  return record(state(), dst && src ? hipErrorNotSupported : hipErrorInvalidValue);
}
hipError_t hipMemcpyDtoA(void* dst, size_t, void* src, size_t) {
  const ApiCall api("hipMemcpyDtoA");
  return record(state(), dst && src ? hipErrorNotSupported : hipErrorInvalidValue);
}
hipError_t hipMemcpyHtoA(void* dst, size_t, const void* src, size_t) {
  const ApiCall api("hipMemcpyHtoA");
  return record(state(), dst && src ? hipErrorNotSupported : hipErrorInvalidValue);
}
hipError_t hipMemcpyHtoAAsync(void* dst, size_t, const void* src, size_t, hipStream_t) {
  const ApiCall api("hipMemcpyHtoAAsync");
  return record(state(), dst && src ? hipErrorNotSupported : hipErrorInvalidValue);
}

// The per-thread default stream's forms of the 3D copy and the 2D and 3D
// memsets (see "The per-thread default stream's forms").
hipError_t hipMemcpy3D_spt(const vgpu::amd::abi::Memcpy3DParms* p) {
  const ApiCall api("hipMemcpy3D_spt");
  return record(state(), copy_3d(p, spt(nullptr), false));
}
hipError_t hipMemcpy3DAsync_spt(const vgpu::amd::abi::Memcpy3DParms* p, hipStream_t stream) {
  return hipMemcpy3DAsync(p, spt(stream));
}
hipError_t hipMemset2D_spt(void* dst, size_t pitch, int value, size_t width, size_t height) {
  const ApiCall api("hipMemset2D_spt");
  const uint8_t byte = static_cast<uint8_t>(value);
  return record(state(), set_2d(dst, pitch, &byte, 1, width, height, spt(nullptr), false));
}
hipError_t hipMemset2DAsync_spt(void* dst, size_t pitch, int value, size_t width, size_t height, hipStream_t stream) {
  return hipMemset2DAsync(dst, pitch, value, width, height, spt(stream));
}
hipError_t hipMemset3D_spt(vgpu::amd::abi::PitchedPtr p, int value, vgpu::amd::abi::Extent e) {
  const ApiCall api("hipMemset3D_spt");
  return record(state(), set_3d(p, value, e, spt(nullptr), false));
}
hipError_t hipMemset3DAsync_spt(vgpu::amd::abi::PitchedPtr p, int value, vgpu::amd::abi::Extent e,
                                hipStream_t stream) {
  return hipMemset3DAsync(p, value, e, spt(stream));
}

// ---- Pools a device allocates from, and who reaches them ---------------------

// The pool hipMallocAsync takes from, for each device: its default pool until
// a program sets another, which every thread then sees, and the default
// again once the one set is destroyed. Wrong arguments are the wrong value
// here, the device included.
hipError_t hipDeviceGetMemPool(void** pool, int ordinal) {
  const ApiCall api("hipDeviceGetMemPool");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!pool || !valid_device(s, ordinal)) return record(s, hipErrorInvalidValue);
  *pool = current_pool(s, ordinal);
  return record(s, hipSuccess);
}
hipError_t hipDeviceSetMemPool(int ordinal, void* pool) {
  const ApiCall api("hipDeviceSetMemPool");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!pool || !valid_device(s, ordinal)) return record(s, hipErrorInvalidValue);
  Pool* p = find_pool(s, pool);
  if (!p) return record(s, hipErrorInvalidValue);
  s.current_pools[ordinal] = p;
  return record(s, hipSuccess);
}
// A pool's own device reads and writes it always; another device as it was
// last granted (none until it is).
hipError_t hipMemPoolGetAccess(int* flags, void* pool, vgpu::amd::abi::MemLocation* location) {
  const ApiCall api("hipMemPoolGetAccess");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  Pool* p = find_pool(s, pool);
  if (!p || !flags || !location || location->type != 1 || !valid_device(s, location->id))
    return record(s, hipErrorInvalidValue);
  refresh_pool_access(*p);
  if (location->id == p->device) *flags = 3;   // hipMemAccessFlagsProtReadWrite
  else if (const auto it = p->access.find(location->id); it != p->access.end()) *flags = it->second;
  else *flags = 0;
  return record(s, hipSuccess);
}

// Virtual memory's access, by device, as it was granted: none until then.
hipError_t hipMemGetAccess(unsigned long long* flags, const vgpu::amd::abi::MemLocation* location, void* ptr) {
  const ApiCall api("hipMemGetAccess");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!flags || !location || !ptr || location->type != 1 || !valid_device(s, location->id))
    return record(s, hipErrorInvalidValue);
  const uint64_t va = reinterpret_cast<uint64_t>(ptr);
  if (vmm_mapping(va) == g_vmm_maps.end()) return record(s, hipErrorInvalidValue);
  *flags = 0;
  for (const auto& [range, grants] : s.vmm_access)
    if (va >= range.first && va < range.first + range.second)
      if (const auto it = grants.find(location->id); it != grants.end()) *flags = static_cast<unsigned>(it->second);
  return record(s, hipSuccess);
}

// ---- Libraries: modules by another name ------------------------------------------
//
// hipLibrary_t is CUDA 12's library, which is a module here, loaded on the
// current device; a hipKernel_t from it is the module's function. JIT and
// library options change nothing.
hipError_t hipLibraryLoadData(void** library, const void* code, void*, void*, unsigned int, void*, void*,
                              unsigned int) {
  const ApiCall api("hipLibraryLoadData");
  if (!library || !code) return record(state(), hipErrorInvalidValue);
  return hipModuleLoadData(reinterpret_cast<hipModule_t*>(library), code);
}
hipError_t hipLibraryLoadFromFile(void** library, const char* path, void*, void*, unsigned int, void*, void*,
                                  unsigned int) {
  const ApiCall api("hipLibraryLoadFromFile");
  if (!library || !path) return record(state(), hipErrorInvalidValue);
  const hipError_t e = hipModuleLoad(reinterpret_cast<hipModule_t*>(library), path);
  return e == hipErrorFileNotFound ? record(state(), hipErrorInvalidValue) : e;
}
hipError_t hipLibraryUnload(void* library) {
  const ApiCall api("hipLibraryUnload");
  if (!library) return record(state(), hipErrorInvalidValue);
  return hipModuleUnload(static_cast<hipModule_t>(library));
}
hipError_t hipLibraryGetKernel(void** kernel, void* library, const char* name) {
  const ApiCall api("hipLibraryGetKernel");
  if (!kernel || !library || !name) return record(state(), hipErrorInvalidValue);
  return hipModuleGetFunction(reinterpret_cast<hipFunction_t*>(kernel), static_cast<hipModule_t>(library), name);
}
hipError_t hipLibraryGetKernelCount(unsigned int* count, void* library) {
  const ApiCall api("hipLibraryGetKernelCount");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!count || !library) return record(s, hipErrorInvalidValue);
  for (const auto& m : s.modules)
    if (m.get() == library) {
      *count = static_cast<unsigned int>(m->object.kernels.size());
      return record(s, hipSuccess);
    }
  return record(s, hipErrorInvalidHandle);
}

// ---- Linking at run time -------------------------------------------------------------
//
// ROCm's HIP links LLVM bitcode with AMD's compiler library (comgr), which
// is not here. A code object handed in as it is -- an AMDGPU ELF, or a
// bundle holding one -- is what the link gives back.
}  // extern "C"
namespace {
struct LinkState {
  std::vector<std::vector<uint8_t>> inputs;
  std::vector<uint8_t> output;
  bool bitcode = false;
};
std::mutex g_link_mutex;
std::set<LinkState*> g_links;
hipError_t link_add(LinkState* st, int type, const void* data, size_t size) {
  std::lock_guard<std::mutex> lock(g_link_mutex);
  if (!g_links.count(st)) return hipErrorInvalidHandle;
  if (!data || !size) return hipErrorInvalidImage;
  constexpr int kLLVMBitcode = 100, kLLVMBundledBitcode = 101, kLLVMArchive = 102;
  if (type == kLLVMBitcode || type == kLLVMBundledBitcode || type == kLLVMArchive) st->bitcode = true;
  const auto* p = static_cast<const uint8_t*>(data);
  st->inputs.emplace_back(p, p + size);
  return hipSuccess;
}
}  // namespace
extern "C" {
hipError_t hipLinkCreate(unsigned int, void*, void**, void** state_out) {
  const ApiCall api("hipLinkCreate");
  if (!state_out) return record(state(), hipErrorInvalidValue);
  auto* st = new LinkState;
  std::lock_guard<std::mutex> lock(g_link_mutex);
  g_links.insert(st);
  *state_out = st;
  return hipSuccess;
}
hipError_t hipLinkAddData(void* link, int type, void* data, size_t size, const char*, unsigned int, void*, void**) {
  const ApiCall api("hipLinkAddData");
  return record(state(), link_add(static_cast<LinkState*>(link), type, data, size));
}
hipError_t hipLinkAddFile(void* link, int type, const char* path, unsigned int, void*, void**) {
  const ApiCall api("hipLinkAddFile");
  if (!path) return record(state(), hipErrorInvalidValue);
  std::ifstream f(path, std::ios::binary);
  if (!f) return record(state(), hipErrorInvalidValue);
  const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return record(state(), link_add(static_cast<LinkState*>(link), type, bytes.data(), bytes.size()));
}
hipError_t hipLinkComplete(void* link, void** binary, size_t* size) {
  const ApiCall api("hipLinkComplete");
  auto* st = static_cast<LinkState*>(link);
  std::lock_guard<std::mutex> lock(g_link_mutex);
  if (!g_links.count(st) || !binary || !size) return record(state(), hipErrorInvalidValue);
  if (st->bitcode)
    return record(state(), fail(hipErrorNotSupported, "linking LLVM bitcode needs AMD's compiler library (comgr)"));
  if (st->inputs.size() > 1)
    return record(state(), fail(hipErrorNotSupported, "linking more than one code object into one is not modelled"));
  st->output = st->inputs.empty() ? std::vector<uint8_t>{} : st->inputs.front();
  *binary = st->output.empty() ? nullptr : st->output.data();
  *size = st->output.size();
  return hipSuccess;
}
hipError_t hipLinkDestroy(void* link) {
  const ApiCall api("hipLinkDestroy");
  auto* st = static_cast<LinkState*>(link);
  std::lock_guard<std::mutex> lock(g_link_mutex);
  if (!g_links.erase(st)) return record(state(), hipErrorInvalidValue);
  delete st;
  return hipSuccess;
}

// A fat binary is loaded as any module is: the bundle's code for the device.
hipError_t hipModuleLoadFatBinary(hipModule_t* module, const void* fatbin) {
  const ApiCall api("hipModuleLoadFatBinary");
  if (!module || !fatbin) return record(state(), hipErrorInvalidValue);
  return hipModuleLoadData(module, fatbin);
}

// ---- OpenGL interop ------------------------------------------------------------------
//
// There is no OpenGL context for a device to share with: no device is an
// OpenGL one, and what would register or map a buffer or an image is
// refused as ROCm's HIP refuses it there.
hipError_t hipGLGetDevices(unsigned int* count, int*, unsigned int, int) {
  const ApiCall api("hipGLGetDevices");
  if (!count) return record(state(), hipErrorInvalidValue);
  *count = 0;
  return record(state(), hipErrorNoDevice);
}
hipError_t hipGraphicsGLRegisterBuffer(void**, unsigned int, unsigned int) {
  const ApiCall api("hipGraphicsGLRegisterBuffer");
  return record(state(), hipErrorInvalidValue);
}
hipError_t hipGraphicsGLRegisterImage(void**, unsigned int, unsigned int, unsigned int) {
  const ApiCall api("hipGraphicsGLRegisterImage");
  return record(state(), hipErrorInvalidValue);
}
hipError_t hipGraphicsMapResources(int, void**, hipStream_t) {
  const ApiCall api("hipGraphicsMapResources");
  return record(state(), hipErrorUnknown);
}
hipError_t hipGraphicsResourceGetMappedPointer(void**, size_t*, void*) {
  const ApiCall api("hipGraphicsResourceGetMappedPointer");
  return record(state(), hipErrorUnknown);
}
hipError_t hipGraphicsUnmapResources(int, void**, hipStream_t) {
  const ApiCall api("hipGraphicsUnmapResources");
  return record(state(), hipErrorInvalidHandle);
}
hipError_t hipGraphicsUnregisterResource(void*) {
  const ApiCall api("hipGraphicsUnregisterResource");
  return record(state(), hipErrorInvalidValue);
}

// ---- A file descriptor for a range of device memory ------------------------------------------
//
// What ROCm hands out as a dma-buf: here, the file the memory's bytes are
// in. An allocation of the device's moves into a file of its own for it (as
// an IPC handle's does, keeping its address); shared virtual memory is one
// already. Either imports as memory of its own (hipMemImportFromShareableHandle).
hipError_t hipMemGetHandleForAddressRange(void* handle, void* ptr, size_t size, int type, unsigned long long flags) {
  const ApiCall api("hipMemGetHandleForAddressRange");
  settle_before_sharing(ptr);   // a device allocation may move into a file
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  constexpr int kDmaBufFd = 1;
  if (!handle || !ptr || !size || type != kDmaBufFd || flags) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  const uint64_t va = reinterpret_cast<uint64_t>(ptr);
  const int d = owner_of(s, va);
  if (d < 0) return record(s, hipErrorInvalidValue);
  int fd = -1;
  if (const auto m = vmm_mapping(va); m != g_vmm_maps.end()) {
    VmmAllocation* v = m->second.second;
    if (size > m->second.first - (va - m->first)) return record(s, hipErrorInvalidValue);
    if (const hipError_t e = vmm_as_file(v); e != hipSuccess) return record(s, e);
    fd = ::fcntl(v->fd, F_DUPFD_CLOEXEC, 0);
  } else {
    uint64_t base = 0, bytes = 0;
    vgpu::MemoryManager& mem = s.rt->device(d).memory();
    // Moved into a file once; each later handle opens that file again.
    auto path = g_dmabuf_paths.upper_bound(va);
    if (path != g_dmabuf_paths.begin() && va < std::prev(path)->first + std::prev(path)->second.second &&
        mem.is_shared(std::prev(path)->first)) {
      --path;
      base = path->first;
      bytes = path->second.second;
    } else {
      path = g_dmabuf_paths.end();
      if (!mem.find_allocation(va, &base, &bytes)) return record(s, hipErrorInvalidValue);
    }
    if (size > bytes - (va - base)) return record(s, hipErrorInvalidValue);
    if (path == g_dmabuf_paths.end()) {
      static std::atomic<uint32_t> counter{0};
      char id[40];
      std::snprintf(id, sizeof id, "%x-d%x", static_cast<unsigned>(::getpid()), counter.fetch_add(1) + 1);
      if (mem.is_shared(base)) return record(s, fail(hipErrorInvalidValue, "that memory is already shared another way"));
      try {
        mem.share(base, ipc_path(id));
      } catch (const std::exception& e) {
        return record(s, fail(hipErrorInvalidValue, e.what()));
      }
      path = g_dmabuf_paths.insert_or_assign(base, std::make_pair(ipc_path(id), bytes)).first;
    }
    fd = ::open(path->second.first.c_str(), O_RDWR | O_CLOEXEC);
  }
  if (fd < 0) return record(s, fail(hipErrorInvalidValue, std::string("opening the memory's file: ") + std::strerror(errno)));
  *static_cast<int*>(handle) = fd;
  return record(s, hipSuccess);
}

// ---- What ROCm's own tools reach into -------------------------------------------------------

// hiprtc's precompiled HIP headers: there are none; hiprtc parses the
// headers themselves.
void __hipGetPCH(const char** pch, unsigned int* size) {
  if (pch) *pch = nullptr;
  if (size) *size = 0;
}
// What a debugger asks the runtime it attaches to.
const char* amd_dbgapi_get_build_name() { return ""; }
const char* amd_dbgapi_get_git_hash() { return ""; }
size_t amd_dbgapi_get_build_id() { return 0; }
// roctracer's hook into the runtime. A profiler attaches here through
// vgpu/hip_profiler.hpp instead, so there is nothing to register.
hipError_t hipRegisterTracerCallback(const void*) { return hipSuccess; }

// compiler-rt's half-precision conversions, which ROCm's library carries for
// programs built without them: IEEE binary16, rounded to nearest even.
uint16_t __gnu_f2h_ieee(float f) {
  uint32_t x = 0;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u, exp = (x >> 23) & 0xFFu, man = x & 0x7FFFFFu;
  if (exp == 0xFF) return static_cast<uint16_t>(sign | 0x7C00u | (man ? 0x200u | (man >> 13) : 0));
  const int e = static_cast<int>(exp) - 127 + 15;
  if (e >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
  if (e <= 0) {
    if (e < -10) return static_cast<uint16_t>(sign);
    const uint32_t m = man | 0x800000u;
    const int shift = 14 - e;
    uint32_t h = m >> shift;
    const uint32_t rest = m & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rest > half || (rest == half && (h & 1))) ++h;
    return static_cast<uint16_t>(sign | h);
  }
  uint32_t h = (static_cast<uint32_t>(e) << 10) | (man >> 13);
  const uint32_t rest = man & 0x1FFFu;
  if (rest > 0x1000u || (rest == 0x1000u && (h & 1))) ++h;   // may carry into the exponent, up to infinity
  return static_cast<uint16_t>(sign | h);
}
float __gnu_h2f_ieee(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, x = 0;
  if (exp == 0x1F) {
    x = sign | 0x7F800000u | (man << 13);
  } else if (exp) {
    x = sign | ((exp - 15 + 127) << 23) | (man << 13);
  } else if (man) {
    int e = -1;
    do {
      ++e;
      man <<= 1;
    } while (!(man & 0x400u));
    x = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | ((man & 0x3FFu) << 13);
  } else {
    x = sign;
  }
  float f = 0;
  std::memcpy(&f, &x, 4);
  return f;
}

}  // extern "C"

// ---- Graphs and stream capture -------------------------------------------------
#include "hip_graph.inc"

extern "C" {

// ---- Memory and semaphores another API made (Vulkan interop) -------------------
//
// Nothing here can import another API's memory or semaphores; what a call
// gets wrong about its arguments is still refused as ROCm's HIP refuses it,
// before that.
hipError_t hipImportExternalMemory(void** out, const void* desc) {
  const ApiCall api("hipImportExternalMemory");
  if (!out || !desc) return record(state(), hipErrorInvalidValue);
  const int type = *static_cast<const int*>(desc);
  const auto size = *reinterpret_cast<const unsigned long long*>(static_cast<const uint8_t*>(desc) + 24);
  const auto flags = *reinterpret_cast<const unsigned*>(static_cast<const uint8_t*>(desc) + 32);
  if (type < 1 || type > 8 || !size || flags > 1) return record(state(), hipErrorInvalidValue);
  return record(state(), hipErrorNotSupported);
}
hipError_t hipExternalMemoryGetMappedBuffer(void** ptr, void* mem, const void* desc) {
  const ApiCall api("hipExternalMemoryGetMappedBuffer");
  (void)ptr, (void)mem, (void)desc;
  return record(state(), hipErrorInvalidValue);   // there is no external memory to map
}
hipError_t hipDestroyExternalMemory(void*) {
  const ApiCall api("hipDestroyExternalMemory");
  return record(state(), hipErrorInvalidValue);
}
hipError_t hipImportExternalSemaphore(void** out, const void* desc) {
  const ApiCall api("hipImportExternalSemaphore");
  if (!out || !desc) return record(state(), hipErrorInvalidValue);
  const int type = *static_cast<const int*>(desc);
  if (type < 1 || type > 10) return record(state(), hipErrorInvalidValue);
  return record(state(), hipErrorNotSupported);
}
hipError_t hipSignalExternalSemaphoresAsync(const void* sems, const void* params, unsigned int, hipStream_t) {
  const ApiCall api("hipSignalExternalSemaphoresAsync");
  (void)sems, (void)params;   // no semaphore can have been imported: any array is the wrong value
  return record(state(), hipErrorInvalidValue);
}
hipError_t hipWaitExternalSemaphoresAsync(const void* sems, const void* params, unsigned int, hipStream_t) {
  const ApiCall api("hipWaitExternalSemaphoresAsync");
  (void)sems, (void)params;   // no semaphore can have been imported: any array is the wrong value
  return record(state(), hipErrorInvalidValue);
}
hipError_t hipDestroyExternalSemaphore(void*) {
  const ApiCall api("hipDestroyExternalSemaphore");
  return record(state(), hipErrorInvalidValue);
}

// ---- The per-thread default stream's forms (_spt) -----------------------------
//
// What a program built with -fgpu-default-stream=per-thread calls in place of
// each call that works in a stream: the same call, with the null handle
// naming the calling thread's own default stream (see "Stream order") rather
// than the legacy one. A synchronous call is synchronous on that stream.
hipError_t hipMemcpy_spt(void* dst, const void* src, size_t bytes, hipMemcpyKind kind) {
  const ApiCall api("hipMemcpy_spt");
  return record(state(), copy_in_order(dst, src, bytes, kind, spt(nullptr), false));
}
hipError_t hipMemcpyAsync_spt(void* dst, const void* src, size_t bytes, hipMemcpyKind kind, hipStream_t stream) {
  return hipMemcpyAsync(dst, src, bytes, kind, spt(stream));
}
hipError_t hipMemcpy2D_spt(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                           hipMemcpyKind kind) {
  const ApiCall api("hipMemcpy2D_spt");
  return record(state(), copy_in_order(dst, src, width, kind, spt(nullptr), false, height, dpitch, spitch));
}
hipError_t hipMemcpy2DAsync_spt(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                size_t height, hipMemcpyKind kind, hipStream_t stream) {
  return hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, spt(stream));
}
hipError_t hipMemcpyToSymbol_spt(const void* symbol, const void* src, size_t bytes, size_t offset,
                                 hipMemcpyKind kind) {
  const ApiCall api("hipMemcpyToSymbol_spt");
  return copy_symbol(true, symbol, const_cast<void*>(src), bytes, offset, kind, spt(nullptr), false);
}
hipError_t hipMemcpyFromSymbol_spt(void* dst, const void* symbol, size_t bytes, size_t offset, hipMemcpyKind kind) {
  const ApiCall api("hipMemcpyFromSymbol_spt");
  return copy_symbol(false, symbol, dst, bytes, offset, kind, spt(nullptr), false);
}
hipError_t hipMemcpyToSymbolAsync_spt(const void* symbol, const void* src, size_t bytes, size_t offset,
                                      hipMemcpyKind kind, hipStream_t stream) {
  return hipMemcpyToSymbolAsync(symbol, src, bytes, offset, kind, spt(stream));
}
hipError_t hipMemcpyFromSymbolAsync_spt(void* dst, const void* symbol, size_t bytes, size_t offset,
                                        hipMemcpyKind kind, hipStream_t stream) {
  return hipMemcpyFromSymbolAsync(dst, symbol, bytes, offset, kind, spt(stream));
}
hipError_t hipMemset_spt(void* dst, int value, size_t bytes) {
  const ApiCall api("hipMemset_spt");
  const uint8_t byte = static_cast<uint8_t>(value);
  return record(state(), fill_in_order(dst, &byte, 1, bytes, spt(nullptr), false));
}
hipError_t hipMemsetAsync_spt(void* dst, int value, size_t bytes, hipStream_t stream) {
  return hipMemsetAsync(dst, value, bytes, spt(stream));
}
hipError_t hipStreamQuery_spt(hipStream_t stream) { return hipStreamQuery(spt(stream)); }
hipError_t hipStreamSynchronize_spt(hipStream_t stream) { return hipStreamSynchronize(spt(stream)); }
hipError_t hipStreamGetPriority_spt(hipStream_t stream, int* priority) {
  return hipStreamGetPriority(spt(stream), priority);
}
hipError_t hipStreamGetFlags_spt(hipStream_t stream, unsigned int* flags) {
  return hipStreamGetFlags(spt(stream), flags);
}
hipError_t hipStreamWaitEvent_spt(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  return hipStreamWaitEvent(spt(stream), event, flags);
}
hipError_t hipStreamAddCallback_spt(hipStream_t stream, hipStreamCallback_t callback, void* data,
                                    unsigned int flags) {
  return hipStreamAddCallback(spt(stream), callback, data, flags);
}
hipError_t hipEventRecord_spt(hipEvent_t event, hipStream_t stream) { return hipEventRecord(event, spt(stream)); }
hipError_t hipLaunchKernel_spt(const void* host_function, vgpu::amd::abi::Dim3 grid, vgpu::amd::abi::Dim3 block,
                               void** args, size_t shared, hipStream_t stream) {
  return hipLaunchKernel(host_function, grid, block, args, shared, spt(stream));
}
hipError_t hipLaunchCooperativeKernel_spt(const void* host_function, vgpu::amd::abi::Dim3 grid,
                                          vgpu::amd::abi::Dim3 block, void** args, uint32_t shared,
                                          hipStream_t stream) {
  return hipLaunchCooperativeKernel(host_function, grid, block, args, shared, spt(stream));
}
hipError_t hipLaunchHostFunc_spt(hipStream_t stream, void (*fn)(void*), void* data) {
  return hipLaunchHostFunc(spt(stream), fn, data);
}
hipError_t hipGraphLaunch_spt(void* exec, hipStream_t stream) { return hipGraphLaunch(exec, spt(stream)); }
hipError_t hipStreamBeginCapture_spt(hipStream_t stream, int mode) {
  return hipStreamBeginCapture(spt(stream), mode);
}
hipError_t hipStreamEndCapture_spt(hipStream_t stream, void** graph) {
  return hipStreamEndCapture(spt(stream), graph);
}
hipError_t hipStreamIsCapturing_spt(hipStream_t stream, int* status) {
  return hipStreamIsCapturing(spt(stream), status);
}
hipError_t hipStreamGetCaptureInfo_spt(hipStream_t stream, int* status, unsigned long long* id) {
  return hipStreamGetCaptureInfo(spt(stream), status, id);
}
hipError_t hipStreamGetCaptureInfo_v2_spt(hipStream_t stream, int* status, unsigned long long* id, void** graph,
                                          const void*** deps, size_t* count) {
  return hipStreamGetCaptureInfo_v2(spt(stream), status, id, graph, deps, count);
}
hipError_t hipGetDriverEntryPoint_spt(const char* symbol, void** pfn, unsigned long long flags, int* status) {
  return hipGetDriverEntryPoint(symbol, pfn, flags, status);
}

}  // extern "C"

// ---- What the HSA runtime takes from this one (hip_shared.hpp) -------------

namespace vgpu::amd::shared {

struct Loaded : Module {
  std::vector<uint8_t> host_image;
};
namespace {
std::vector<std::unique_ptr<Loaded>> g_loaded;   // under State's mutex
}  // namespace

bool start(std::string* why) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (ensure_runtime(s) == hipSuccess) return true;
  if (why) *why = "no AMD GPU to run on (VGPU_GPU names an amd/ profile)";
  return false;
}
int device_count() { return state().rt ? state().rt->device_count() : 0; }
const DeviceProfile& profile(int ordinal) { return state().rt->device(ordinal).profile(); }
MemoryManager& memory(int ordinal) { return state().rt->device(ordinal).memory(); }

const Loaded* load(int ordinal, const void* bytes, size_t size, std::string* why) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (ensure_runtime(s) != hipSuccess || ordinal < 0 || ordinal >= s.rt->device_count()) {
    if (why) *why = "no such device";
    return nullptr;
  }
  try {
    auto m = std::make_unique<Loaded>();
    m->object = load_code_object(std::string(static_cast<const char*>(bytes), size), "the code object");
    m->host_image = m->object.image;   // place() gives up the object's own copy
    vgpu::runtime::Device& d = s.rt->device(ordinal);
    place(*m, d.memory());
    report_loaded(*m, ordinal, bytes, size);
    g_loaded.push_back(std::move(m));
    return g_loaded.back().get();
  } catch (const std::exception& e) {
    if (why) *why = e.what();
    return nullptr;
  }
}

void unload(const Loaded* m) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  for (size_t i = 0; i < g_loaded.size(); ++i)
    if (g_loaded[i].get() == m) {
      for (int d = 0; d < s.rt->device_count(); ++d)
        if (m->globals && s.rt->device(d).memory().owns(m->globals)) {
          try {
            s.rt->device(d).memory().free(m->globals);
          } catch (const std::exception&) {
          }
        }
      g_loaded.erase(g_loaded.begin() + static_cast<long>(i));
      return;
    }
}

const CodeObject& object(const Loaded* m) { return m->object; }
const std::vector<uint8_t>& host_image(const Loaded* m) { return m->host_image; }
uint64_t code_base(const Loaded* m) { return m->code_base; }

bool run(int ordinal, const Loaded* m, const Kernel& k, const uint32_t grid[3], const uint32_t group_size[3],
         uint32_t dynamic_lds, uint64_t kernarg, bool cooperative, std::string* why) {
  State& s = state();
  uint32_t groups[3];
  for (int i = 0; i < 3; ++i) groups[i] = group_size[i] ? (grid[i] + group_size[i] - 1) / group_size[i] : 0;
  LaunchJob job;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    const hipError_t e = prepare_launch(s, ordinal, *m, k, {groups[0], groups[1], groups[2]},
                                        {group_size[0], group_size[1], group_size[2]}, dynamic_lds, {}, nullptr,
                                        cooperative, &job);
    if (e != hipSuccess) {
      if (why) *why = hipGetErrorString(e);
      return false;
    }
  }
  job.kernarg_at = kernarg;
  for (int i = 0; i < 3; ++i) job.grid_items[i] = grid[i] % group_size[i] ? grid[i] : 0;
  job.kernel_limits = false;   // a packet goes to the hardware as it is
  const hipError_t e = run_launch(job);
  if (e != hipSuccess && why) *why = "the kernel " + k.name + " failed";
  return e == hipSuccess;
}

void allow_peer(int from, int to) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (from != to) s.peers.insert({from, to});
}
size_t scratch_limit(int ordinal) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return scratch_limit_locked(s, ordinal);
}
bool set_scratch_limit(int ordinal, size_t bytes) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return set_scratch_limit_locked(s, ordinal, bytes);
}
int owner(uint64_t address) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return s.rt ? owner_of(s, address) : -1;
}

void map_host(void* p, size_t n) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (ensure_runtime(s) == hipSuccess) map_host_everywhere(s, p, n);
}
void unmap_host(void* p) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.rt) unmap_host_everywhere(s, p);
}

bool copy(void* dst, const void* src, size_t n, std::string* why) {
  State& s = state();
  vgpu::MemoryManager *to = nullptr, *from = nullptr;
  const uint64_t dst_va = reinterpret_cast<uint64_t>(dst), src_va = reinterpret_cast<uint64_t>(src);
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    if (ensure_runtime(s) != hipSuccess) return false;
    for (int i = 0; i < s.rt->device_count(); ++i) {
      if (s.rt->device(i).memory().owns(dst_va)) to = &s.rt->device(i).memory();
      if (s.rt->device(i).memory().owns(src_va)) from = &s.rt->device(i).memory();
    }
  }
  try {
    if (to && from) {
      std::vector<uint8_t> buf(n);
      from->read(src_va, buf.data(), n);
      to->write(dst_va, buf.data(), n);
    } else if (to) {
      to->write(dst_va, src, n);
    } else if (from) {
      from->read(src_va, dst, n);
    } else {
      std::memmove(dst, src, n);
    }
  } catch (const std::exception& e) {
    if (why) *why = e.what();
    return false;
  }
  return true;
}

}  // namespace vgpu::amd::shared
