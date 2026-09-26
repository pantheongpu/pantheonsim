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
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <mutex>
#include <string>
#include <dlfcn.h>
#include <unistd.h>
#include <vector>

#include "vgpu/amd_bundle.hpp"
#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vgpu/amd_hostcall.hpp"
#include "vgpu/error.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"
#include "vgpu/telemetry.hpp"
#include "vgpu/hip_abi.hpp"
#include "vgpu/hip_profiler.hpp"
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
  uint64_t globals = 0;   // where the module's own variables were placed
  // Where a linked module's image is (CodeObject::image): its code runs from
  // there, and its variables are in it, so `globals` is the same address.
  uint64_t code_base = 0;
  // What a profiler knows it and its kernels by (vgpu/hip_profiler.hpp).
  uint64_t code_object_id = 0;
  std::vector<uint64_t> kernel_ids;   // one per kernel, in the object's order
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
  o.load_size = m.object.linked ? m.object.image.size() : m.object.text.size();
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

// A kernel launch a graph holds: what to run and with what, copied when it was
// captured, since a graph replays what the stream was asked to do then.
struct Node {
  int device = 0;
  const void* host_function = nullptr;
  vgpu::amd::abi::Dim3 grid{1, 1, 1}, block{1, 1, 1};
  uint32_t shared = 0;
  std::vector<uint8_t> args;
};
struct Graph {
  std::vector<Node> nodes;
  unsigned long long capture_id = 0;   // which capture recorded it
};

// What a stream was made with. Every launch finishes before it returns, so a
// stream holds no work; it is these, which a program can ask for back.
struct Stream {
  int device = 0;
  unsigned flags = 0;
  int priority = 0;
  std::vector<uint32_t> cu_mask;   // empty: every compute unit
};

// A stream-ordered memory pool. Nothing freed to it is held back, so what it
// has reserved is what is in use, and its attributes are what was set.
struct Pool {
  int device = 0;
  bool destroyed = false;
  uint64_t release_threshold = 0;
  int reuse[4] = {0, 1, 1, 1};   // by hipMemPoolAttr 1 to 3
  uint64_t used = 0, used_high = 0;
};

struct State {
  std::mutex mutex;
  std::unique_ptr<vgpu::runtime::Runtime> rt;
  std::vector<std::unique_ptr<Module>> modules;
  std::vector<std::unique_ptr<Function>> functions;
  std::vector<std::unique_ptr<FatBinary>> fat_binaries;
  // Each bundle read, by where it is: a library's binaries can share one.
  std::map<const uint8_t*, std::unique_ptr<vgpu::amd::Bundle>> bundles;
  std::map<const void*, HostFunction> host_functions;
  std::map<const void*, HostVar> host_vars;
  std::set<std::pair<int, int>> peers;           // (device, peer) pairs with access enabled
  std::map<hipStream_t, Graph> capturing;        // streams recording rather than running
  // Each device's hostcall buffer, made when a kernel is first launched on it:
  // what device-side printf writes through (vgpu/amd_hostcall.hpp).
  std::map<int, std::unique_ptr<vgpu::amd::Hostcall>> hostcalls;
  std::vector<std::unique_ptr<Graph>> graphs, graph_execs;
  std::map<hipStream_t, Stream> streams;
  std::deque<Pool> pools;                       // a pool's address is its handle
  std::map<int, Pool*> default_pools;           // by device
  std::map<uint64_t, std::pair<Pool*, uint64_t>> pool_allocations;   // address -> pool, bytes
  int current = 0;
  hipError_t last = hipSuccess;
  std::string profile_id;
};

State& state() {
  static State s;
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
  try {
    vgpu::DeviceProfile p = vgpu::load_gpu(id);
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

hipError_t record(State& s, hipError_t e) {
  s.last = e;
  return e;
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
    // A kernel that takes nothing needs neither.
    return k.kernarg_size && !k.args.empty()
               ? fail(hipErrorInvalidValue, "the kernel takes arguments, and the launch passed none")
               : hipSuccess;
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

// Runs one kernel on one device: the arguments go into device memory as its
// kernarg segment, and the dispatch runs to completion before this returns.
hipError_t dispatch_kernel(State& s, int ordinal, const Module& module, const Kernel& kernel,
                           vgpu::amd::abi::Dim3 grid, vgpu::amd::abi::Dim3 block, uint32_t shared,
                           const std::vector<uint8_t>& args, hipStream_t stream, bool cooperative = false) {
  vgpu::runtime::Device& d = s.rt->device(ordinal);
  const CodeObject& object = module.object;
  // VGPU_TRACE_LAUNCHES=1 says what each launch runs, one line to stderr:
  // what a program that calls a library cannot otherwise see.
  static const bool trace = [] {
    const char* t = std::getenv("VGPU_TRACE_LAUNCHES");
    return t && t[0] == '1';
  }();
  if (trace)
    std::fprintf(stderr, "VirtualGPU HIP: launch %s on device %d, grid %ux%ux%u of %ux%ux%u, %u bytes of LDS\n",
                 kernel.name.c_str(), ordinal, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared);
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
  if (!block.x || !block.y || !block.z || !grid.x || !grid.y || !grid.z) return hipErrorInvalidConfiguration;
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
    const size_t padded = (args.size() + 63) / 64 * 64 + 64;
    kernarg = mem.alloc(padded);
    std::vector<uint8_t> segment(padded, 0);
    std::copy(args.begin(), args.end(), segment.begin());
    mem.write(kernarg, segment.data(), segment.size());
    if (cooperative) {
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
    dispatch.wave_size = static_cast<uint32_t>(d.profile().warp_size);
    dispatch.dynamic_lds = shared;   // what the launch adds to the kernel's own LDS
    dispatch.hostcall = hostcall.get();
    dispatch.code_base = module.code_base;
    dispatch.cooperative = cooperative;
    dispatch.grid_sync = grid_sync;
    for (const auto& [from, to] : s.peers)
      if (from == ordinal) {
        if (dispatch.peers.size() <= static_cast<size_t>(to)) dispatch.peers.resize(static_cast<size_t>(to) + 1);
        dispatch.peers[static_cast<size_t>(to)] = &s.rt->device(to).memory();
      }
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
    mem.free(kernarg);
    if (grid_sync) mem.free(grid_sync);
    // What the device spent, as telemetry reports a kernel: the instructions
    // a wave retires, at the profile's clock.
    const uint32_t mhz = d.profile().telemetry.sm_clock_max_mhz;
    const double clock = mhz ? mhz * 1e6 : 1e9;
    d.note_busy(static_cast<double>(stats.instructions) / clock);
  } catch (const std::exception& e) {
    // A launch that failed counted nothing, and the profiler is told so.
    if (prof && prof->launched && start) prof->launched(launch, nullptr, start, vgpu::amd::hipprof::now_ns(), token);
    for (uint64_t a : {kernarg, grid_sync})
      if (a) {
        try {
          mem.free(a);
        } catch (const std::exception&) {
        }
      }
    return fail(hipErrorLaunchFailure, s.profile_id + ": " + e.what());
  }
  return hipSuccess;
}

// Puts a loaded module on a device: a linked one's whole image, from which its
// code runs and in which its variables sit; an unlinked one's variables, with
// its code told where they went.
void place(Module& m, vgpu::MemoryManager& mem) {
  if (m.object.linked) {
    m.code_base = mem.alloc(m.object.image.empty() ? 1 : m.object.image.size());
    if (!m.object.image.empty()) mem.write(m.code_base, m.object.image.data(), m.object.image.size());
    m.globals = m.code_base;
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
  auto& bundle = s.bundles[fb.image];
  try {
    if (!bundle) bundle = vgpu::amd::read_bundle(fb.image);
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
    place(*m, d.memory());
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
thread_local std::vector<CallConfiguration> g_call_configurations;

// Host memory kernels reach: see hipHostMalloc.
std::mutex g_host_mutex;
std::map<uint64_t, size_t> g_host_allocations;   // hipHostMalloc
std::map<uint64_t, size_t> g_host_registered;    // hipHostRegister
std::map<uint64_t, size_t> g_managed;            // hipMallocManaged

void map_host_everywhere(State& s, void* p, size_t n) {
  for (int d = 0; d < s.rt->device_count(); ++d)
    s.rt->device(d).memory().map_host(reinterpret_cast<uint64_t>(p), p, std::max<size_t>(n, 1));
}
void unmap_host_everywhere(State& s, void* p) {
  for (int d = 0; d < s.rt->device_count(); ++d) s.rt->device(d).memory().unmap_host(reinterpret_cast<uint64_t>(p));
}
// The range of m holding va, or m.end().
std::map<uint64_t, size_t>::const_iterator find_range(const std::map<uint64_t, size_t>& m, uint64_t va) {
  auto it = m.upper_bound(va);
  if (it == m.begin()) return m.end();
  --it;
  return va < it->first + std::max<size_t>(it->second, 1) ? it : m.end();
}
// Host memory of `bytes`, page-aligned, mapped for every device and kept in m.
hipError_t host_alloc(void** ptr, size_t size, std::map<uint64_t, size_t>& m) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  const size_t n = size ? size : 1;
  void* p = std::aligned_alloc(4096, (n + 4095) / 4096 * 4096);
  if (!p) return record(s, hipErrorOutOfMemory);
  map_host_everywhere(s, p, n);
  std::lock_guard<std::mutex> host_lock(g_host_mutex);
  m[reinterpret_cast<uint64_t>(p)] = n;
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
  std::free(ptr);
  return true;
}

}  // namespace

extern "C" {

hipError_t hipInit(unsigned int) {
  const ApiCall api("hipInit");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return record(s, ensure_runtime(s));
}

hipError_t hipGetDeviceCount(int* count) {
  const ApiCall api("hipGetDeviceCount");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!count) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
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
  std::lock_guard<std::mutex> lock(s.mutex);
  return record(s, ensure_runtime(s));   // every launch here has already finished
}

hipError_t hipDeviceReset(void) {
  const ApiCall api("hipDeviceReset");
  State& s = state();
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

hipError_t hipFree(void* ptr) {
  const ApiCall api("hipFree");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr) return record(s, hipSuccess);
  if (host_free(s, ptr, g_managed)) return record(s, hipSuccess);
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
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
  try {
    s.rt->device(owner).memory().free(reinterpret_cast<uint64_t>(ptr));
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidDevicePointer, e.what()));
  }
  if (const auto* p = profiler(); p && p->allocated)
    p->allocated(owner, reinterpret_cast<uint64_t>(ptr), 0, true, start, vgpu::amd::hipprof::now_ns());
  return record(s, hipSuccess);
}

hipError_t hipMemcpy(void* dst, const void* src, size_t bytes, hipMemcpyKind kind) {
  const ApiCall api("hipMemcpy");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
  if (!bytes) return record(s, hipSuccess);
  if (!dst || !src) return record(s, hipErrorInvalidValue);
  const uint64_t dst_va = reinterpret_cast<uint64_t>(dst), src_va = reinterpret_cast<uint64_t>(src);
  // Addresses are unified: a device pointer says which device's memory it
  // is, so a device-to-device copy may run between two devices, as HIP's
  // does. The current device's memory answers for an address no device owns,
  // and says what is wrong with it.
  auto owner = [&](uint64_t va) -> vgpu::MemoryManager& {
    for (int i = 0; i < s.rt->device_count(); ++i)
      if (s.rt->device(i).memory().owns(va)) return s.rt->device(i).memory();
    return d->memory();
  };
  vgpu::MemoryManager& to = owner(dst_va);
  vgpu::MemoryManager& from = owner(src_va);
  // hipMemcpyDefault asks the runtime to tell device memory from host memory,
  // which it does by whether a device owns the address.
  bool to_device = kind == hipMemcpyHostToDevice, from_device = kind == hipMemcpyDeviceToHost;
  if (kind == hipMemcpyDeviceToDevice) to_device = from_device = true;
  if (kind == hipMemcpyDefault) {
    to_device = to.owns(dst_va);
    from_device = from.owns(src_va);
  } else if (kind != hipMemcpyHostToHost && kind != hipMemcpyHostToDevice && kind != hipMemcpyDeviceToHost &&
             kind != hipMemcpyDeviceToDevice) {
    return record(s, hipErrorInvalidMemcpyDirection);
  }
  const uint64_t start = vgpu::amd::hipprof::now_ns();
  try {
    if (to_device && from_device) {
      std::vector<uint8_t> buf(bytes);
      from.read(src_va, buf.data(), bytes);
      to.write(dst_va, buf.data(), bytes);
    } else if (to_device) {
      to.write(dst_va, src, bytes);
    } else if (from_device) {
      from.read(src_va, dst, bytes);
    } else {
      std::memcpy(dst, src, bytes);
    }
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  d->note_transfer(bytes, 0.0);
  if (const auto* p = profiler(); p && p->copied) {
    using vgpu::amd::hipprof::Copy;
    const Copy k = to_device && from_device ? Copy::DeviceToDevice
                   : to_device             ? Copy::HostToDevice
                   : from_device           ? Copy::DeviceToHost
                                           : Copy::HostToHost;
    p->copied(k, s.current, s.current, bytes, dst_va, src_va, start, vgpu::amd::hipprof::now_ns());
  }
  return record(s, hipSuccess);
}

hipError_t hipMemset(void* dst, int value, size_t bytes) {
  const ApiCall api("hipMemset");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
  if (!bytes) return record(s, hipSuccess);
  if (!dst) return record(s, hipErrorInvalidValue);
  const uint8_t byte = static_cast<uint8_t>(value);
  try {
    d->memory().fill(reinterpret_cast<uint64_t>(dst), &byte, 1, bytes);
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  return record(s, hipSuccess);
}

// Every launch and copy here finishes before it returns, so the asynchronous
// forms are the synchronous ones: a stream is a handle, and there is nothing
// for it to be waiting on.
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t bytes, hipMemcpyKind kind, hipStream_t) {
  const ApiCall api("hipMemcpyAsync");
  return hipMemcpy(dst, src, bytes, kind);
}

hipError_t hipMemsetAsync(void* dst, int value, size_t bytes, hipStream_t) {
  const ApiCall api("hipMemsetAsync");
  return hipMemset(dst, value, bytes);
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
      bundle = vgpu::amd::read_bundle(bytes);
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
  if (!path) return hipErrorInvalidValue;
  std::ifstream in(path, std::ios::binary);
  if (!in) return fail(hipErrorFileNotFound, std::string("no code object at ") + path);
  const std::string bytes((std::istreambuf_iterator<char>(in)), {});
  if (bytes.empty()) return fail(hipErrorInvalidImage, std::string(path) + " is empty");
  return hipModuleLoadData(module, bytes.data());
}

hipError_t hipModuleUnload(hipModule_t module) {
  const ApiCall api("hipModuleUnload");
  State& s = state();
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
  if (!module || !name) return record(s, hipErrorInvalidValue);
  Module* m = reinterpret_cast<Module*>(module);
  const vgpu::amd::GlobalVar* g = vgpu::amd::find_global(m->object, name);
  if (!g) return record(s, fail(hipErrorNotFound, std::string("the module has no variable named ") + name));
  if (dptr) *dptr = reinterpret_cast<void*>(m->globals + g->offset);
  if (bytes) *bytes = static_cast<size_t>(g->size);
  return record(s, hipSuccess);
}

hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gx, unsigned int gy, unsigned int gz,
                                 unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared,
                                 hipStream_t stream, void** params, void** extra) {
  const ApiCall api("hipModuleLaunchKernel");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!f) return record(s, hipErrorInvalidValue);
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
  Function* fn = reinterpret_cast<Function*>(f);
  if (!bx || !by || !bz || !gx || !gy || !gz) return record(s, hipErrorInvalidConfiguration);
  std::vector<uint8_t> args;
  if (const hipError_t e = build_kernargs(*fn->kernel, params, extra, &args); e != hipSuccess)
    return record(s, e);

  return record(s, dispatch_kernel(s, s.current, *fn->module, *fn->kernel, {gx, gy, gz}, {bx, by, bz}, shared, args,
                                   stream));
}

const char* hipGetErrorName(hipError_t e) {
  const ApiCall api("hipGetErrorName");
  switch (e) {
    case hipSuccess: return "hipSuccess";
    case hipErrorInvalidValue: return "hipErrorInvalidValue";
    case hipErrorOutOfMemory: return "hipErrorOutOfMemory";
    case hipErrorNotInitialized: return "hipErrorNotInitialized";
    case hipErrorDeinitialized: return "hipErrorDeinitialized";
    case hipErrorInvalidConfiguration: return "hipErrorInvalidConfiguration";
    case hipErrorInvalidSymbol: return "hipErrorInvalidSymbol";
    case hipErrorInvalidPitchValue: return "hipErrorInvalidPitchValue";
    case hipErrorInvalidDevicePointer: return "hipErrorInvalidDevicePointer";
    case hipErrorInvalidMemcpyDirection: return "hipErrorInvalidMemcpyDirection";
    case hipErrorInvalidDevice: return "hipErrorInvalidDevice";
    case hipErrorInvalidImage: return "hipErrorInvalidImage";
    case hipErrorInvalidContext: return "hipErrorInvalidContext";
    case hipErrorFileNotFound: return "hipErrorFileNotFound";
    case hipErrorNotFound: return "hipErrorNotFound";
    case hipErrorNotSupported: return "hipErrorNotSupported";
    case hipErrorInvalidDeviceFunction: return "hipErrorInvalidDeviceFunction";
    case hipErrorNoBinaryForGpu: return "hipErrorNoBinaryForGpu";
    case hipErrorInvalidHandle: return "hipErrorInvalidHandle";
    case hipErrorIllegalState: return "hipErrorIllegalState";
    case hipErrorNotReady: return "hipErrorNotReady";
    case hipErrorPeerAccessAlreadyEnabled: return "hipErrorPeerAccessAlreadyEnabled";
    case hipErrorPeerAccessNotEnabled: return "hipErrorPeerAccessNotEnabled";
    case hipErrorLaunchFailure: return "hipErrorLaunchFailure";
    case hipErrorCooperativeLaunchTooLarge: return "hipErrorCooperativeLaunchTooLarge";
    case hipErrorStreamCaptureUnsupported: return "hipErrorStreamCaptureUnsupported";
    case hipErrorStreamCaptureUnmatched: return "hipErrorStreamCaptureUnmatched";
    case hipErrorUnsupportedLimit: return "hipErrorUnsupportedLimit";
    case hipErrorHostMemoryAlreadyRegistered: return "hipErrorHostMemoryAlreadyRegistered";
    case hipErrorHostMemoryNotRegistered: return "hipErrorHostMemoryNotRegistered";
    case hipErrorUnknown: break;
  }
  return "hipErrorUnknown";
}

const char* hipGetErrorString(hipError_t e) {
  const ApiCall api("hipGetErrorString");
  switch (e) {
    case hipSuccess: return "no error";
    case hipErrorInvalidValue: return "invalid argument";
    case hipErrorOutOfMemory: return "out of memory";
    case hipErrorNotInitialized: return "invalid device ordinal";
    case hipErrorDeinitialized: return "driver shutting down";
    case hipErrorInvalidConfiguration: return "invalid configuration argument";
    case hipErrorInvalidSymbol: return "invalid device symbol";
    case hipErrorInvalidPitchValue: return "invalid pitch argument";
    case hipErrorInvalidDevicePointer: return "invalid device pointer";
    case hipErrorInvalidMemcpyDirection: return "invalid copy direction for memcpy";
    case hipErrorInvalidDevice: return "invalid device ordinal";
    case hipErrorInvalidImage: return "invalid device function image";
    case hipErrorInvalidContext: return "invalid device context";
    case hipErrorFileNotFound: return "file not found";
    case hipErrorNotFound: return "named symbol not found";
    case hipErrorNotSupported: return "operation not supported";
    case hipErrorInvalidDeviceFunction: return "invalid device function";
    case hipErrorNoBinaryForGpu: return "no kernel image is available for execution on the device";
    case hipErrorInvalidHandle: return "invalid resource handle";
    case hipErrorIllegalState: return "the operation cannot be performed in the present state";
    case hipErrorNotReady: return "device not ready";
    case hipErrorPeerAccessAlreadyEnabled: return "peer access is already enabled";
    case hipErrorPeerAccessNotEnabled: return "peer access has not been enabled";
    case hipErrorLaunchFailure: return "unspecified launch failure";
    case hipErrorCooperativeLaunchTooLarge: return "too many blocks in cooperative launch";
    case hipErrorStreamCaptureUnsupported: return "operation not permitted when stream is capturing";
    case hipErrorStreamCaptureUnmatched: return "the capture was not initiated in this stream";
    case hipErrorUnsupportedLimit: return "limit is not supported on this architecture";
    case hipErrorHostMemoryAlreadyRegistered: return "part or all of the requested memory range is already mapped";
    case hipErrorHostMemoryNotRegistered: return "pointer does not correspond to a registered memory region";
    case hipErrorUnknown: break;
  }
  return "unknown error";
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
  static intptr_t next = 1;
  *stream = reinterpret_cast<hipStream_t>(next++);
  s.streams[*stream] = Stream{s.current, flags, priority, std::move(cu_mask)};
  return record(s, hipSuccess);
}

// The stream's own record, or the null stream's: the current device, no
// flags, the normal priority. A handle that was never made, or was
// destroyed, is nullptr.
const Stream* find_stream(State& s, hipStream_t stream, Stream* null_stream) {
  if (!stream) {
    *null_stream = Stream{s.current, 0, 0, {}};
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

// Streams: every launch here finishes before it returns, so a stream is a
// handle and what it was made with. The null stream is what a program gets
// by default.
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
  std::lock_guard<std::mutex> lock(s.mutex);
  if (stream) s.streams.erase(stream);
  return record(s, hipSuccess);
}
hipError_t hipStreamSynchronize(hipStream_t) {
  const ApiCall api("hipStreamSynchronize");
  return hipSuccess;
}

// Events: a program records one before its work and one after, and asks how
// long there was between them. Every launch here has finished by the time it
// returns, so an event is recorded the moment the call is made and the time
// between two of them is the time the simulator took -- not what a card would
// have taken, which this does not claim to know.
namespace {

struct Event {
  bool timing = true;
  bool recorded = false;
  std::chrono::steady_clock::time_point when{};
};

std::mutex g_event_mutex;
std::map<hipEvent_t, std::unique_ptr<Event>> g_events;

Event* find_event(hipEvent_t e) {   // the caller holds g_event_mutex
  const auto it = g_events.find(e);
  return it == g_events.end() ? nullptr : it->second.get();
}

}  // namespace

hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int flags) {
  const ApiCall api("hipEventCreateWithFlags");
  if (!event) return hipErrorInvalidValue;
  if (flags & ~static_cast<unsigned>(hipEventBlockingSync | hipEventDisableTiming)) return hipErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_event_mutex);
  static intptr_t next = 1;
  const hipEvent_t handle = reinterpret_cast<hipEvent_t>(next++);
  auto e = std::make_unique<Event>();
  e->timing = (flags & hipEventDisableTiming) == 0;
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
  return g_events.erase(event) ? hipSuccess : hipErrorInvalidHandle;
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t) {
  const ApiCall api("hipEventRecord");
  std::lock_guard<std::mutex> lock(g_event_mutex);
  Event* e = find_event(event);
  if (!e) return hipErrorInvalidHandle;
  e->recorded = true;
  e->when = std::chrono::steady_clock::now();
  return hipSuccess;
}

// There is never work left behind an event, so both of these answer at once.
hipError_t hipEventSynchronize(hipEvent_t event) {
  const ApiCall api("hipEventSynchronize");
  std::lock_guard<std::mutex> lock(g_event_mutex);
  return find_event(event) ? hipSuccess : hipErrorInvalidHandle;
}

hipError_t hipEventQuery(hipEvent_t event) {
  const ApiCall api("hipEventQuery");
  return hipEventSynchronize(event);
}

hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t end) {
  const ApiCall api("hipEventElapsedTime");
  if (!ms) return hipErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_event_mutex);
  const Event* a = find_event(start);
  const Event* b = find_event(end);
  // An event that was never recorded, or one created without timing, has no
  // time to give.
  if (!a || !b || !a->recorded || !b->recorded || !a->timing || !b->timing) return hipErrorInvalidHandle;
  *ms = std::chrono::duration<float, std::milli>(b->when - a->when).count();
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

// At exit: the binary's modules go, and their variables with them.
void __hipUnregisterFatBinary(void** modules) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  FatBinary* fb = reinterpret_cast<FatBinary*>(modules);
  for (auto it = s.host_functions.begin(); it != s.host_functions.end();)
    it = it->second.binary == fb ? s.host_functions.erase(it) : std::next(it);
  for (auto it = s.host_vars.begin(); it != s.host_vars.end();)
    it = it->second.binary == fb ? s.host_vars.erase(it) : std::next(it);
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
  g_call_configurations.push_back({grid, block, shared, stream});
  return hipSuccess;
}

hipError_t __hipPopCallConfiguration(vgpu::amd::abi::Dim3* grid, vgpu::amd::abi::Dim3* block, size_t* shared,
                                     hipStream_t* stream) {
  if (g_call_configurations.empty()) return hipErrorInvalidConfiguration;
  const CallConfiguration c = g_call_configurations.back();
  g_call_configurations.pop_back();
  if (grid) *grid = c.grid;
  if (block) *block = c.block;
  if (shared) *shared = c.shared;
  if (stream) *stream = c.stream;
  return hipSuccess;
}

// A chevron launch, and hipLaunchKernel called by hand: the kernel is named by
// its host-side function, and each argument by a pointer to its value. On a
// stream that is capturing, the launch is recorded rather than run.
hipError_t hipLaunchKernel(const void* host_function, vgpu::amd::abi::Dim3 grid, vgpu::amd::abi::Dim3 block,
                           void** args, size_t shared, hipStream_t stream) {
  const ApiCall api("hipLaunchKernel");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
  const auto hf = s.host_functions.find(host_function);
  if (hf == s.host_functions.end())
    return record(s, fail(hipErrorInvalidDeviceFunction, "no kernel was registered for that function"));
  Module* m = nullptr;
  if (const hipError_t e = module_on(s, *hf->second.binary, s.current, &m); e != hipSuccess) return record(s, e);
  const Kernel* k = vgpu::amd::find_kernel(m->object, hf->second.kernel);
  if (!k)
    return record(s, fail(hipErrorInvalidDeviceFunction, "the program's device code has no kernel named " +
                                                             hf->second.kernel));
  std::vector<uint8_t> packed;
  if (const hipError_t e = build_kernargs(*k, args, nullptr, &packed); e != hipSuccess) return record(s, e);
  if (auto cap = s.capturing.find(stream); stream && cap != s.capturing.end()) {
    cap->second.nodes.push_back(Node{s.current, host_function, grid, block, static_cast<uint32_t>(shared), packed});
    return record(s, hipSuccess);
  }
  return record(s, dispatch_kernel(s, s.current, *m, *k, grid, block, static_cast<uint32_t>(shared), packed, stream));
}

// ---- What the device is, in the layout the HIP headers give it ---------------

}  // extern "C"

namespace {

// What a device is, in the layout the HIP headers give it: what
// hipGetDeviceProperties returns and hipDeviceGetAttribute answers from.
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
    props->maxGridSize[i] = static_cast<int>(p.limits.max_grid_dim[i]);
  }
  props->clockRate = static_cast<int>(p.telemetry.sm_clock_max_mhz) * 1000;
  props->memoryClockRate = static_cast<int>(p.telemetry.mem_clock_max_mhz) * 1000;
  props->multiProcessorCount = static_cast<int>(p.limits.multiprocessors);
  props->maxThreadsPerMultiProcessor = static_cast<int>(p.limits.max_threads_per_sm);
  props->maxBlocksPerMultiProcessor = static_cast<int>(p.limits.max_blocks_per_sm);
  props->l2CacheSize = static_cast<int>(p.limits.l2_cache_bytes);
  // HIP's version of an AMD device is its gfx version's first two parts:
  // gfx942 is 9.4, gfx90a 9.0 ("gfx" + major + minor digit + stepping).
  props->major = p.cc_major;
  props->minor = p.cc_minor;
  if (p.gcn_arch.size() >= 6 && p.gcn_arch.rfind("gfx", 0) == 0) {
    const std::string digits = p.gcn_arch.substr(3);
    props->major = std::atoi(digits.substr(0, digits.size() - 2).c_str());
    props->minor = static_cast<int>(std::strtol(digits.substr(digits.size() - 2, 1).c_str(), nullptr, 16));
  }
  props->pciDeviceID = ordinal;
  props->concurrentKernels = 1;
  props->cooperativeLaunch = 1;
  props->unifiedAddressing = 1;
  // Pinned, registered and managed host memory are mapped for every
  // device's kernels (see hipHostMalloc), and allocations come from pools.
  props->canMapHostMemory = 1;
  props->managedMemory = 1;
  props->hostRegisterSupported = 1;
  props->memoryPoolsSupported = 1;
  props->ECCEnabled = p.telemetry.ecc ? 1 : 0;
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
    // Not here: images and textures (no image instructions yet), a stream
    // waiting on a value in memory, fine-grained host memory.
    case A::kImageSupport:
    case A::kCanUseStreamWaitValue:
    case A::kFineGrainSupport: *value = 0; break;
    default:
      return record(s, fail(hipErrorInvalidValue, "device attribute " + std::to_string(attribute) + " is not answered here"));
  }
  return record(s, hipSuccess);
}

// Nothing here waits on anything, so how a program would like to wait changes
// nothing; the flags it may ask for are accepted and any other is refused.
hipError_t hipSetDeviceFlags(unsigned int flags) {
  const ApiCall api("hipSetDeviceFlags");
  const unsigned int known = 0x7 /* schedule */ | 0x8 /* map host */ | 0x10 /* lmem resize */;
  return record(state(), (flags & ~known) ? hipErrorInvalidValue : hipSuccess);
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
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int) {
  const ApiCall api("hipHostMalloc");
  return host_alloc(ptr, size, g_host_allocations);
}

hipError_t hipHostFree(void* ptr) {
  const ApiCall api("hipHostFree");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr) return record(s, hipSuccess);
  return record(s, host_free(s, ptr, g_host_allocations) ? hipSuccess : hipErrorInvalidValue);
}

// The caller's own memory, made reachable from kernels where it is. Any
// overlap with memory already registered, pinned or managed is refused, as
// HIP refuses it.
hipError_t hipHostRegister(void* ptr, size_t size, unsigned int) {
  const ApiCall api("hipHostRegister");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr || !size) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  const uint64_t lo = reinterpret_cast<uint64_t>(ptr), hi = lo + size;
  std::lock_guard<std::mutex> host_lock(g_host_mutex);
  for (const auto* m : {&g_host_registered, &g_host_allocations, &g_managed}) {
    auto it = m->lower_bound(hi);
    if (it != m->begin() && std::prev(it)->first + std::prev(it)->second > lo)
      return record(s, hipErrorHostMemoryAlreadyRegistered);
  }
  map_host_everywhere(s, ptr, size);
  g_host_registered[lo] = size;
  return record(s, hipSuccess);
}

hipError_t hipHostUnregister(void* ptr) {
  const ApiCall api("hipHostUnregister");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr) return record(s, hipErrorInvalidValue);
  std::lock_guard<std::mutex> host_lock(g_host_mutex);
  const auto it = g_host_registered.find(reinterpret_cast<uint64_t>(ptr));
  if (it == g_host_registered.end()) return record(s, hipErrorHostMemoryNotRegistered);
  if (s.rt) unmap_host_everywhere(s, ptr);
  g_host_registered.erase(it);
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
  if (flags && flags != 1 /* hipMemAttachGlobal */ && flags != 2 /* hipMemAttachHost */)
    return record(state(), hipErrorInvalidValue);
  if (!size) return record(state(), hipErrorInvalidValue);
  return host_alloc(ptr, size, g_managed);
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
hipError_t hipMemcpyPeerAsync(void* dst, int dst_device, const void* src, int src_device, size_t bytes,
                              hipStream_t) {
  const ApiCall api("hipMemcpyPeerAsync");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  const int n = s.rt->device_count();
  if (dst_device < 0 || dst_device >= n || src_device < 0 || src_device >= n) return record(s, hipErrorInvalidDevice);
  if (!bytes) return record(s, hipSuccess);
  if (!dst || !src) return record(s, hipErrorInvalidValue);
  vgpu::MemoryManager& to = s.rt->device(dst_device).memory();
  vgpu::MemoryManager& from = s.rt->device(src_device).memory();
  const uint64_t dst_va = reinterpret_cast<uint64_t>(dst), src_va = reinterpret_cast<uint64_t>(src);
  if (!to.owns(dst_va) || !from.owns(src_va))
    return record(s, fail(hipErrorInvalidValue, "a peer copy's addresses are not on the devices it names"));
  const uint64_t start = vgpu::amd::hipprof::now_ns();
  try {
    std::vector<uint8_t> buf(bytes);
    from.read(src_va, buf.data(), bytes);
    to.write(dst_va, buf.data(), bytes);
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  s.rt->device(src_device).note_transfer(bytes, 0.0);
  if (const auto* p = profiler(); p && p->copied)
    p->copied(vgpu::amd::hipprof::Copy::DeviceToDevice, src_device, dst_device, bytes, dst_va, src_va, start,
              vgpu::amd::hipprof::now_ns());
  return record(s, hipSuccess);
}

// ---- Graphs a stream is recorded into ----------------------------------------
//
// Between beginning and ending a capture, kernels launched on the stream are
// recorded rather than run. The graph that comes out can be instantiated and
// launched, which runs what was recorded, in order, with the arguments it was
// recorded with.
hipError_t hipStreamBeginCapture(hipStream_t stream, int) {
  const ApiCall api("hipStreamBeginCapture");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!stream) return record(s, fail(hipErrorStreamCaptureUnsupported, "the null stream cannot be captured"));
  static unsigned long long next_capture = 1;
  if (!s.capturing.emplace(stream, Graph{{}, next_capture}).second) return record(s, hipErrorIllegalState);
  ++next_capture;
  return record(s, hipSuccess);
}

hipError_t hipStreamEndCapture(hipStream_t stream, void** graph) {
  const ApiCall api("hipStreamEndCapture");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  const auto it = s.capturing.find(stream);
  if (it == s.capturing.end()) return record(s, hipErrorStreamCaptureUnmatched);
  if (!graph) return record(s, hipErrorInvalidValue);
  s.graphs.push_back(std::make_unique<Graph>(std::move(it->second)));
  s.capturing.erase(it);
  *graph = s.graphs.back().get();
  return record(s, hipSuccess);
}

hipError_t hipGraphInstantiate(void** exec, void* graph, void*, char*, size_t) {
  const ApiCall api("hipGraphInstantiate");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!exec || !graph) return record(s, hipErrorInvalidValue);
  s.graph_execs.push_back(std::make_unique<Graph>(*static_cast<Graph*>(graph)));
  *exec = s.graph_execs.back().get();
  return record(s, hipSuccess);
}

hipError_t hipGraphLaunch(void* exec, hipStream_t stream) {
  const ApiCall api("hipGraphLaunch");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!exec) return record(s, hipErrorInvalidValue);
  for (const Node& node : static_cast<Graph*>(exec)->nodes) {
    const auto hf = s.host_functions.find(node.host_function);
    if (hf == s.host_functions.end()) return record(s, hipErrorInvalidDeviceFunction);
    Module* m = nullptr;
    if (const hipError_t e = module_on(s, *hf->second.binary, node.device, &m); e != hipSuccess) return record(s, e);
    const Kernel* k = vgpu::amd::find_kernel(m->object, hf->second.kernel);
    if (!k) return record(s, hipErrorInvalidDeviceFunction);
    if (const hipError_t e = dispatch_kernel(s, node.device, *m, *k, node.grid, node.block, node.shared, node.args,
                                             stream);
        e != hipSuccess)
      return record(s, e);
  }
  return record(s, hipSuccess);
}

hipError_t hipGraphDestroy(void* graph) {
  const ApiCall api("hipGraphDestroy");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  for (size_t i = 0; i < s.graphs.size(); ++i)
    if (s.graphs[i].get() == graph) {
      s.graphs.erase(s.graphs.begin() + static_cast<std::ptrdiff_t>(i));
      return record(s, hipSuccess);
    }
  return record(s, hipErrorInvalidValue);
}

hipError_t hipGraphExecDestroy(void* exec) {
  const ApiCall api("hipGraphExecDestroy");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  for (size_t i = 0; i < s.graph_execs.size(); ++i)
    if (s.graph_execs[i].get() == exec) {
      s.graph_execs.erase(s.graph_execs.begin() + static_cast<std::ptrdiff_t>(i));
      return record(s, hipSuccess);
    }
  return record(s, hipErrorInvalidValue);
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
  if (p.gcn_arch.rfind("gfx9", 0) != 0)
    return fail(hipErrorNotSupported, "occupancy is worked out for CDNA (gfx9) here, and this device is " + p.gcn_arch);
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
  constexpr size_t kMaxWavesPerSimd = 8, kVgprsPerSimd = 512, kVgprGranule = 8, kSgprsPerSimd = 800;
  constexpr size_t kSimdsPerCu = 4;
  size_t gpr_waves = kMaxWavesPerSimd;
  if (k.vgpr_count) gpr_waves = kVgprsPerSimd / align_up(k.vgpr_count, kVgprGranule);
  if (gpr_waves == 0) return fail(hipErrorUnknown, "the kernel uses more vector registers than a SIMD has");
  if (k.sgpr_count) gpr_waves = std::min(gpr_waves, kSgprsPerSimd / align_up(k.sgpr_count, 16));
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
hipError_t hipLaunchCooperativeKernel(const void* host_function, vgpu::amd::abi::Dim3 grid,
                                      vgpu::amd::abi::Dim3 block, void** args, unsigned int shared,
                                      hipStream_t stream) {
  const ApiCall api("hipLaunchCooperativeKernel");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!device(s)) return record(s, hipErrorInvalidDevice);
  const Kernel* k = nullptr;
  if (const hipError_t e = kernel_for(s, host_function, &k); e != hipSuccess) return record(s, e);
  Module* m = nullptr;
  const auto hf = s.host_functions.find(host_function);
  if (const hipError_t e = module_on(s, *hf->second.binary, s.current, &m); e != hipSuccess) return record(s, e);
  Occupancy o;
  const int threads = static_cast<int>(block.x * block.y * block.z);
  const vgpu::DeviceProfile& p = s.rt->device(s.current).profile();
  if (const hipError_t e = occupancy(p, *k, threads, shared, false, &o); e != hipSuccess) return record(s, e);
  const uint64_t resident = uint64_t(o.blocks_per_cu) * p.limits.multiprocessors;
  if (uint64_t{grid.x} * grid.y * grid.z > resident)
    return record(s, fail(hipErrorCooperativeLaunchTooLarge,
                          "a cooperative grid of " + std::to_string(uint64_t{grid.x} * grid.y * grid.z) +
                              " work-groups is more than the " + std::to_string(resident) +
                              " this device holds at once at this block size"));
  std::vector<uint8_t> packed;
  if (const hipError_t e = build_kernargs(*k, args, nullptr, &packed); e != hipSuccess) return record(s, e);
  return record(s, dispatch_kernel(s, s.current, *m, *k, grid, block, shared, packed, stream, true));
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
                       hipMemcpyKind kind) {
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
  void* device = reinterpret_cast<void*>(address + offset);
  return to_symbol ? hipMemcpy(device, host, bytes, kind) : hipMemcpy(host, device, bytes, kind);
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
                                  hipMemcpyKind kind, hipStream_t) {
  const ApiCall api("hipMemcpyToSymbolAsync");
  return copy_symbol(true, symbol, const_cast<void*>(src), bytes, offset, kind);
}

hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t bytes, size_t offset, hipMemcpyKind kind,
                                    hipStream_t) {
  const ApiCall api("hipMemcpyFromSymbolAsync");
  return copy_symbol(false, symbol, dst, bytes, offset, kind);
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
// what HIP's documentation says of it, over a runtime where each call has
// finished by the time it returns.
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
// An allocation from `pool`, on the pool's device. The caller holds the lock.
hipError_t pool_alloc(State& s, Pool* pool, void** ptr, size_t size) {
  if (!ptr) return hipErrorInvalidValue;
  if (!size) {
    *ptr = nullptr;
    return hipSuccess;
  }
  try {
    *ptr = reinterpret_cast<void*>(s.rt->device(pool->device).memory().alloc(size));
  } catch (const std::exception& e) {
    return fail(hipErrorOutOfMemory, e.what());
  }
  s.pool_allocations[reinterpret_cast<uint64_t>(*ptr)] = {pool, size};
  pool->used += size;
  pool->used_high = std::max(pool->used_high, pool->used);
  return hipSuccess;
}

}  // namespace

extern "C" {

hipError_t hipMallocFromPoolAsync(void** ptr, size_t size, void* pool, hipStream_t) {
  const ApiCall api("hipMallocFromPoolAsync");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  Pool* p = find_pool(s, pool);
  if (!p) return record(s, hipErrorInvalidValue);
  return record(s, pool_alloc(s, p, ptr, size));
}
hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t) {
  const ApiCall api("hipMallocAsync");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!device(s)) return record(s, hipErrorInvalidDevice);
  return record(s, pool_alloc(s, default_pool(s, s.current), ptr, size));
}
hipError_t hipFreeAsync(void* ptr, hipStream_t) {
  const ApiCall api("hipFreeAsync");
  return hipFree(ptr);
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
// A pool holds nothing back, so trimming it has nothing to give back.
hipError_t hipMemPoolTrimTo(void* pool, size_t) {
  const ApiCall api("hipMemPoolTrimTo");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return record(s, find_pool(s, pool) ? hipSuccess : hipErrorInvalidValue);
}
hipError_t hipMemPoolCreate(void** pool, const vgpu::amd::abi::MemPoolProps* props) {
  const ApiCall api("hipMemPoolCreate");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!pool || !props) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (props->location.id < 0 || props->location.id >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  s.pools.push_back(Pool{});
  s.pools.back().device = props->location.id;
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
  p->destroyed = true;
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
    case A::kPoolReservedMemCurrent:
    case A::kPoolUsedMemCurrent: *static_cast<uint64_t*>(value) = p->used; break;
    case A::kPoolReservedMemHigh:
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
    case A::kPoolReservedMemHigh:
    case A::kPoolUsedMemHigh:
      if (*static_cast<uint64_t*>(value)) return record(s, hipErrorInvalidValue);
      p->used_high = p->used;
      break;
    default: return record(s, hipErrorInvalidValue);
  }
  return record(s, hipSuccess);
}
// Every device reaches every pool's memory already.
hipError_t hipMemPoolSetAccess(void* pool, const vgpu::amd::abi::MemAccessDesc* desc, size_t count) {
  const ApiCall api("hipMemPoolSetAccess");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!find_pool(s, pool) || (count && !desc)) return record(s, hipErrorInvalidValue);
  for (size_t i = 0; i < count; ++i)
    if (desc[i].location.id < 0 || desc[i].location.id >= s.rt->device_count())
      return record(s, hipErrorInvalidDevice);
  return record(s, hipSuccess);
}

// A copy of height rows, each width bytes, from one pitched buffer to another.
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                       hipMemcpyKind kind) {
  const ApiCall api("hipMemcpy2D");
  if (!width || !height) return record(state(), hipSuccess);
  if (width > dpitch || width > spitch) return record(state(), hipErrorInvalidPitchValue);
  for (size_t row = 0; row < height; ++row)
    if (const hipError_t e = hipMemcpy(static_cast<uint8_t*>(dst) + row * dpitch,
                                       static_cast<const uint8_t*>(src) + row * spitch, width, kind);
        e != hipSuccess)
      return e;
  return hipSuccess;
}
hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                            hipMemcpyKind kind, hipStream_t) {
  const ApiCall api("hipMemcpy2DAsync");
  return hipMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
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
  for (int i = 0; i < s.rt->device_count(); ++i)
    if (s.rt->device(i).memory().owns(va)) {
      out->type = vgpu::amd::abi::kMemoryDevice;
      out->device = i;
      out->devicePointer = const_cast<void*>(ptr);
      return record(s, hipSuccess);
    }
  out->type = vgpu::amd::abi::kMemoryUnregistered;
  return record(s, hipSuccess);
}

// hipStreamCaptureStatus: 0 not capturing, 1 capturing.
hipError_t hipStreamIsCapturing(hipStream_t stream, int* status) {
  const ApiCall api("hipStreamIsCapturing");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!status) return record(s, hipErrorInvalidValue);
  *status = stream && s.capturing.count(stream) ? 1 : 0;
  return record(s, hipSuccess);
}

// Nothing is ever left for a stream to do.
hipError_t hipStreamQuery(hipStream_t) {
  const ApiCall api("hipStreamQuery");
  return hipSuccess;
}

hipError_t hipExtGetLastError(void) {
  const ApiCall api("hipExtGetLastError");
  return hipGetLastError();
}

// A module launch sized in work-items rather than work-groups, as HSA sizes a
// dispatch, with events recorded either side of it. A grid that is not a
// whole number of work-groups leaves its last ones partial, which this does
// not model, and refuses.
hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t lx, uint32_t ly,
                                    uint32_t lz, size_t shared, hipStream_t stream, void** params, void** extra,
                                    hipEvent_t start, hipEvent_t stop, uint32_t) {
  const ApiCall api("hipExtModuleLaunchKernel");
  if (!lx || !ly || !lz) return record(state(), hipErrorInvalidConfiguration);
  if (gx % lx || gy % ly || gz % lz)
    return record(state(), fail(hipErrorNotSupported, "a grid of " + std::to_string(gx) + "x" + std::to_string(gy) +
                                                          "x" + std::to_string(gz) +
                                                          " work-items is not a whole number of work-groups of " +
                                                          std::to_string(lx) + "x" + std::to_string(ly) + "x" +
                                                          std::to_string(lz) + ", which this does not model"));
  if (start)
    if (const hipError_t e = hipEventRecord(start, stream); e != hipSuccess) return e;
  if (const hipError_t e = hipModuleLaunchKernel(f, gx / lx, gy / ly, gz / lz, lx, ly, lz,
                                                 static_cast<unsigned>(shared), stream, params, extra);
      e != hipSuccess)
    return e;
  return stop ? hipEventRecord(stop, stream) : hipSuccess;
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

// Physical memory a program made with hipMemCreate: which device's, and the
// manager's own handle for it. Its address is the handle HIP hands out.
struct VmmAllocation {
  int device;
  uint64_t handle;
  size_t size;
};
std::deque<VmmAllocation> g_vmm;   // under State's mutex; entries are never moved
std::set<VmmAllocation*> g_vmm_live;

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

thread_local int t_capture_mode = 0;   // hipStreamCaptureMode, per thread as HIP keeps it

}  // namespace

extern "C" {

// ---- Devices and contexts ----------------------------------------------------

// A context is a device's, and each device has one: its handle is enough.
hipError_t hipCtxGetCurrent(void** ctx) {
  const ApiCall api("hipCtxGetCurrent");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ctx) return record(s, hipErrorInvalidValue);
  static char contexts[64];
  *ctx = s.current < 64 ? &contexts[s.current] : nullptr;
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
  std::snprintf(bus_id, static_cast<size_t>(len), "%04x:%02x:%02x.0", p.pciDomainID, p.pciBusID, p.pciDeviceID);
  return record(s, hipSuccess);
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

// The per-thread stack and the device heap: kept as set, and read back.
// Nothing here runs short of either.
namespace {
size_t g_limits[3] = {1024, 64 << 20, 8 << 20};   // stack, printf FIFO, heap
}
hipError_t hipDeviceSetLimit(int limit, size_t value) {
  const ApiCall api("hipDeviceSetLimit");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (limit != 0 && limit != 2) return record(s, hipErrorUnsupportedLimit);   // hipLimitStackSize, MallocHeapSize
  g_limits[limit] = value;
  return record(s, hipSuccess);
}
hipError_t hipDeviceGetLimit(size_t* value, int limit) {
  const ApiCall api("hipDeviceGetLimit");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!value) return record(s, hipErrorInvalidValue);
  if (limit < 0 || limit > 2) return record(s, hipErrorUnsupportedLimit);
  *value = g_limits[limit];
  return record(s, hipSuccess);
}

hipError_t hipDrvGetErrorString(hipError_t error, const char** text) {
  if (!text) return hipErrorInvalidValue;
  *text = hipGetErrorString(error);
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
  if (attr != 8 && attr != 9) return record(s, hipErrorInvalidValue);
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
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!f) return record(s, hipErrorInvalidValue);
  if (!device(s)) return record(s, hipErrorInvalidDevice);
  if (!bx || !by || !bz || !gx || !gy || !gz) return record(s, hipErrorInvalidConfiguration);
  Function* fn = reinterpret_cast<Function*>(f);
  Occupancy o;
  const vgpu::DeviceProfile& p = s.rt->device(s.current).profile();
  if (const hipError_t e = occupancy(p, *fn->kernel, static_cast<int>(bx * by * bz), shared, false, &o);
      e != hipSuccess)
    return record(s, e);
  const uint64_t resident = uint64_t(o.blocks_per_cu) * p.limits.multiprocessors;
  if (uint64_t{gx} * gy * gz > resident) return record(s, hipErrorCooperativeLaunchTooLarge);
  std::vector<uint8_t> args;
  if (const hipError_t e = build_kernargs(*fn->kernel, params, nullptr, &args); e != hipSuccess) return record(s, e);
  return record(s, dispatch_kernel(s, s.current, *fn->module, *fn->kernel, {gx, gy, gz}, {bx, by, bz}, shared, args,
                                   stream, true));
}

// The options a module is loaded with tune a JIT; there is none here.
hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image, unsigned int, void*, void**) {
  const ApiCall api("hipModuleLoadDataEx");
  return hipModuleLoadData(module, image);
}

// A host function run in stream order: nothing is queued ahead of it, so it
// runs now. In a capture it would become a graph node, which this refuses.
hipError_t hipLaunchHostFunc(hipStream_t stream, void (*fn)(void*), void* data) {
  const ApiCall api("hipLaunchHostFunc");
  {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!fn) return record(s, hipErrorInvalidValue);
    if (stream && s.capturing.count(stream))
      return record(s, fail(hipErrorStreamCaptureUnsupported, "a host function in a captured stream is not modelled"));
  }
  fn(data);
  return hipSuccess;
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
  return create_stream(stream, 0, 0, std::vector<uint32_t>(mask, mask + words));
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
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags) {
  const ApiCall api("hipStreamGetFlags");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
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
hipError_t hipStreamWaitEvent(hipStream_t, hipEvent_t event, unsigned int) {
  const ApiCall api("hipStreamWaitEvent");
  return record(state(), event ? hipSuccess : hipErrorInvalidHandle);
}
// A 32-bit value written to device memory in stream order: now.
hipError_t hipStreamWriteValue32(hipStream_t, void* ptr, uint32_t value, unsigned int) {
  const ApiCall api("hipStreamWriteValue32");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr || reinterpret_cast<uint64_t>(ptr) % 4) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  const int d = owner_of(s, reinterpret_cast<uint64_t>(ptr));
  if (d < 0) return record(s, hipErrorInvalidValue);
  try {
    s.rt->device(d).memory().write(reinterpret_cast<uint64_t>(ptr), &value, 4);
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  return record(s, hipSuccess);
}

// ---- Stream capture --------------------------------------------------------------

// hipStreamCaptureStatus: 0 none, 1 active; and the capture's id.
hipError_t hipStreamGetCaptureInfo(hipStream_t stream, int* status, unsigned long long* id) {
  const ApiCall api("hipStreamGetCaptureInfo");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!status) return record(s, hipErrorInvalidValue);
  const auto it = stream ? s.capturing.find(stream) : s.capturing.end();
  *status = it == s.capturing.end() ? 0 : 1;
  if (id && it != s.capturing.end()) *id = it->second.capture_id;
  return record(s, hipSuccess);
}
// The same, with the graph being recorded. A captured graph here is a
// sequence, so there is never a set of nodes to depend on beyond the last.
hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, int* status, unsigned long long* id, void** graph,
                                      const void*** deps, size_t* count) {
  const ApiCall api("hipStreamGetCaptureInfo_v2");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!status) return record(s, hipErrorInvalidValue);
  const auto it = stream ? s.capturing.find(stream) : s.capturing.end();
  *status = it == s.capturing.end() ? 0 : 1;
  if (it != s.capturing.end()) {
    if (id) *id = it->second.capture_id;
    if (graph) *graph = &it->second;
  }
  if (deps) *deps = nullptr;
  if (count) *count = 0;
  return record(s, hipSuccess);
}
hipError_t hipThreadExchangeStreamCaptureMode(int* mode) {
  const ApiCall api("hipThreadExchangeStreamCaptureMode");
  if (!mode || *mode < 0 || *mode > 2) return record(state(), hipErrorInvalidValue);
  std::swap(*mode, t_capture_mode);
  return record(state(), hipSuccess);
}
hipError_t hipGraphInstantiateWithFlags(void** exec, void* graph, unsigned long long) {
  const ApiCall api("hipGraphInstantiateWithFlags");
  return hipGraphInstantiate(exec, graph, nullptr, nullptr, 0);
}
// A graph's nodes are its launches, in order; each one's handle is its place.
hipError_t hipGraphGetNodes(void* graph, void** nodes, size_t* count) {
  const ApiCall api("hipGraphGetNodes");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!graph || !count) return record(s, hipErrorInvalidValue);
  auto& all = static_cast<Graph*>(graph)->nodes;
  if (nodes)
    for (size_t i = 0; i < std::min(*count, all.size()); ++i) nodes[i] = &all[i];
  *count = all.size();
  return record(s, hipSuccess);
}
// A graph here is the sequence of launches a capture recorded, so building
// one node by node, or with nodes other than launches, is not modelled.
hipError_t hipGraphAddDependencies(void*, const void*, const void*, size_t) {
  return refused("hipGraphAddDependencies", "a graph here is a recorded sequence of launches");
}
hipError_t hipGraphAddEventRecordNode(void**, void*, const void*, size_t, hipEvent_t) {
  return refused("hipGraphAddEventRecordNode", "a graph here holds only kernel launches");
}
hipError_t hipGraphAddHostNode(void**, void*, const void*, size_t, const void*) {
  return refused("hipGraphAddHostNode", "a graph here holds only kernel launches");
}
hipError_t hipGraphNodeGetDependencies(void*, void**, size_t*) {
  return refused("hipGraphNodeGetDependencies", "a graph here is a recorded sequence of launches");
}
hipError_t hipStreamUpdateCaptureDependencies(hipStream_t, void**, size_t, unsigned int) {
  return refused("hipStreamUpdateCaptureDependencies", "a capture here records one sequence");
}
hipError_t hipGraphDebugDotPrint(void*, const char*, unsigned int) {
  return refused("hipGraphDebugDotPrint", "graphs are not drawn");
}
hipError_t hipUserObjectCreate(void**, void*, void (*)(void*), unsigned int, unsigned int) {
  return refused("hipUserObjectCreate", "a graph here does not own objects");
}
hipError_t hipGraphRetainUserObject(void*, void*, unsigned int, unsigned int) {
  return refused("hipGraphRetainUserObject", "a graph here does not own objects");
}

// ---- Memory ----------------------------------------------------------------------

hipError_t hipExtMallocWithFlags(void** ptr, size_t size, unsigned int) {
  const ApiCall api("hipExtMallocWithFlags");
  // Fine- or coarse-grained, the memory is the same here: every write is
  // visible at once.
  return hipMalloc(ptr, size);
}
hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t bytes, hipMemcpyKind kind, hipStream_t) {
  const ApiCall api("hipMemcpyWithStream");
  return hipMemcpy(dst, src, bytes, kind);
}
// count 32-bit words of value.
hipError_t hipMemsetD32Async(void* dst, int value, size_t count, hipStream_t) {
  const ApiCall api("hipMemsetD32Async");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!count) return record(s, hipSuccess);
  if (!dst || reinterpret_cast<uint64_t>(dst) % 4) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  const int d = owner_of(s, reinterpret_cast<uint64_t>(dst));
  if (d < 0) return record(s, hipErrorInvalidDevicePointer);
  uint8_t pattern[4];
  std::memcpy(pattern, &value, 4);
  try {
    s.rt->device(d).memory().fill(reinterpret_cast<uint64_t>(dst), pattern, 4, uint64_t{count} * 4);
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  return record(s, hipSuccess);
}

// The allocation an address falls in: its start and its size.
hipError_t hipMemGetAddressRange(void** base, size_t* size, void* ptr) {
  const ApiCall api("hipMemGetAddressRange");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  const uint64_t va = reinterpret_cast<uint64_t>(ptr);
  uint64_t b = 0, n = 0;
  const int d = owner_of(s, va);
  if (d < 0 || !s.rt->device(d).memory().find_allocation(va, &b, &n)) return record(s, hipErrorNotFound);
  if (base) *base = reinterpret_cast<void*>(b);
  if (size) *size = static_cast<size_t>(n);
  return record(s, hipSuccess);
}
hipError_t hipMemPtrGetInfo(void* ptr, size_t* size) {
  const ApiCall api("hipMemPtrGetInfo");
  if (!ptr || !size) return record(state(), hipErrorInvalidValue);
  return hipMemGetAddressRange(nullptr, size, ptr);
}

// One fact about an address, of those hipPointerGetAttributes gives and the
// allocation it falls in.
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
  static char contexts[64];
  switch (attribute) {
    case K::kPointerContext: *static_cast<void**>(data) = a.device >= 0 ? &contexts[a.device % 64] : nullptr; break;
    case K::kPointerMemoryType: *static_cast<unsigned*>(data) = static_cast<unsigned>(a.type); break;
    case K::kPointerDevicePointer: *static_cast<void**>(data) = a.devicePointer; break;
    case K::kPointerHostPointer: *static_cast<void**>(data) = a.hostPointer; break;
    case K::kPointerSyncMemops: *static_cast<int*>(data) = 1; break;
    case K::kPointerBufferId: *static_cast<uint64_t*>(data) = base; break;
    case K::kPointerIsManaged: *static_cast<int*>(data) = a.isManaged; break;
    case K::kPointerDeviceOrdinal: *static_cast<int*>(data) = a.device; break;
    case K::kPointerRangeStartAddr: *static_cast<void**>(data) = reinterpret_cast<void*>(base); break;
    case K::kPointerRangeSize: *static_cast<size_t*>(data) = static_cast<size_t>(size); break;
    case K::kPointerMapped: *static_cast<int*>(data) = a.type != vgpu::amd::abi::kMemoryUnregistered; break;
    default: return record(s, fail(hipErrorNotSupported, "pointer attribute " + std::to_string(attribute) +
                                                            " is not modelled"));
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

hipError_t hipMemGetAllocationGranularity(size_t* granularity, const vgpu::amd::abi::MemAllocationProp* prop, int) {
  const ApiCall api("hipMemGetAllocationGranularity");
  if (!granularity || !prop) return record(state(), hipErrorInvalidValue);
  *granularity = static_cast<size_t>(vgpu::MemoryManager::kVmmGranularity);
  return record(state(), hipSuccess);
}
hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void*, unsigned long long) {
  const ApiCall api("hipMemAddressReserve");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr || !size) return record(s, hipErrorInvalidValue);
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
  try {
    *ptr = reinterpret_cast<void*>(d->memory().reserve(size, alignment));
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  return record(s, hipSuccess);
}
hipError_t hipMemAddressFree(void* ptr, size_t size) {
  const ApiCall api("hipMemAddressFree");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr || !size || !s.rt) return record(s, hipErrorInvalidValue);
  const int d = owner_of(s, reinterpret_cast<uint64_t>(ptr));
  if (d < 0) return record(s, hipErrorInvalidValue);
  try {
    s.rt->device(d).memory().address_free(reinterpret_cast<uint64_t>(ptr), size);
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  return record(s, hipSuccess);
}
hipError_t hipMemCreate(void** handle, size_t size, const vgpu::amd::abi::MemAllocationProp* prop,
                        unsigned long long) {
  const ApiCall api("hipMemCreate");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!handle || !prop || !size) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  const int d = prop->location.id;
  if (d < 0 || d >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  try {
    g_vmm.push_back(VmmAllocation{d, s.rt->device(d).memory().create_handle(size), size});
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorOutOfMemory, e.what()));
  }
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
  if (!g_vmm_live.erase(v)) return record(s, hipErrorInvalidValue);
  try {
    s.rt->device(v->device).memory().release_handle(v->handle);
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  return record(s, hipSuccess);
}
// Memory made on one device goes into that device's address space: a
// reservation on another is refused, since each manager maps its own.
hipError_t hipMemMap(void* ptr, size_t size, size_t offset, void* handle, unsigned long long) {
  const ApiCall api("hipMemMap");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  auto* v = static_cast<VmmAllocation*>(handle);
  if (!ptr || !size || !g_vmm_live.count(v)) return record(s, hipErrorInvalidValue);
  if (owner_of(s, reinterpret_cast<uint64_t>(ptr)) != v->device)
    return record(s, fail(hipErrorInvalidValue, "memory made on one device mapped into another's reservation"));
  try {
    s.rt->device(v->device).memory().map(reinterpret_cast<uint64_t>(ptr), size, offset, v->handle);
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  return record(s, hipSuccess);
}
hipError_t hipMemUnmap(void* ptr, size_t size) {
  const ApiCall api("hipMemUnmap");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr || !size || !s.rt) return record(s, hipErrorInvalidValue);
  const int d = owner_of(s, reinterpret_cast<uint64_t>(ptr));
  if (d < 0) return record(s, hipErrorInvalidValue);
  try {
    s.rt->device(d).memory().unmap(reinterpret_cast<uint64_t>(ptr), size);
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  return record(s, hipSuccess);
}
// Access for the device that holds the mapping is what its kernels are
// checked against. Every device reaches every other's memory here, so a
// grant to another device changes nothing.
hipError_t hipMemSetAccess(void* ptr, size_t size, const vgpu::amd::abi::MemAccessDesc* desc, size_t count) {
  const ApiCall api("hipMemSetAccess");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr || !size || !desc || !count || !s.rt) return record(s, hipErrorInvalidValue);
  const int d = owner_of(s, reinterpret_cast<uint64_t>(ptr));
  if (d < 0) return record(s, hipErrorInvalidValue);
  try {
    for (size_t i = 0; i < count; ++i) {
      if (desc[i].location.id < 0 || desc[i].location.id >= s.rt->device_count())
        return record(s, hipErrorInvalidDevice);
      if (desc[i].location.id == d)
        s.rt->device(d).memory().set_access(reinterpret_cast<uint64_t>(ptr), size, desc[i].flags & 1,
                                            (desc[i].flags & 2) != 0);
    }
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  return record(s, hipSuccess);
}
hipError_t hipMemExportToShareableHandle(void*, void*, int, unsigned long long) {
  return refused("hipMemExportToShareableHandle", "memory made with hipMemCreate is not shared between processes");
}
hipError_t hipMemImportFromShareableHandle(void**, void*, int) {
  return refused("hipMemImportFromShareableHandle", "memory made with hipMemCreate is not shared between processes");
}

// ---- Memory another process maps (IPC) ---------------------------------------------
//
// An allocation that is shared moves into a file of its own, at the same
// address, and the other process maps that file: both then reach the same
// bytes (vgpu/memory.hpp's share and adopt).

hipError_t hipIpcGetMemHandle(vgpu::amd::abi::IpcMemHandle* handle, void* ptr) {
  const ApiCall api("hipIpcGetMemHandle");
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
hipError_t hipIpcGetEventHandle(void*, hipEvent_t) {
  return refused("hipIpcGetEventHandle", "events are not shared between processes");
}
hipError_t hipIpcOpenEventHandle(hipEvent_t*, vgpu::amd::abi::IpcMemHandle) {
  return refused("hipIpcOpenEventHandle", "events are not shared between processes");
}

}  // extern "C"
