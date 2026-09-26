// VirtualGPU's HSA runtime: the interface ROCm's libhsa-runtime64 gives a
// program, over the simulated GPUs its HIP runtime runs (hip_shared.hpp).
//
// A program finds the agents -- one CPU, and a GPU for each simulated device
// -- allocates memory from their pools, loads a code object into an
// executable, and dispatches kernels by writing AQL packets into a queue and
// ringing its doorbell. Each queue has a thread of its own, the packet
// processor: it takes packets in order, runs a kernel dispatch on the device,
// waits out a barrier packet's signals, and decrements each packet's
// completion signal once it is done, which is what the program waits on.
//
// The interface is the HSA Foundation's specification and AMD's documented
// extensions, declared in vgpu/hsa_abi.h. Memory the host allocates from the
// CPU's pools (fine-grained, and kernarg) is reachable from every device's
// kernels at its own address, as system memory is on a real system; memory
// from a GPU's pool is that device's, which the host reaches by copying.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "hip_queue.hpp"
#include "hip_shared.hpp"
#include "vgpu/amd_bundle.hpp"
#include "vgpu/hsa_abi.h"

namespace {

namespace shared = vgpu::amd::shared;

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}
// Says what went wrong, once, where the program would otherwise only see a status.
hsa_status_t fail(hsa_status_t status, const std::string& what) {
  if (!quiet()) std::fprintf(stderr, "VirtualGPU HSA: %s\n", what.c_str());
  return status;
}

// ---- Agents -----------------------------------------------------------------
//
// Handles: the CPU is agent 1; GPU i is 0x100 + i. A pool or region is its
// agent's handle times 16 plus its index.

constexpr uint64_t kCpuAgent = 1, kGpuAgentBase = 0x100;

std::mutex g_mutex;   // the runtime's own tables below
int g_refs = 0;       // hsa_init less hsa_shut_down

bool started() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_refs > 0;
}
int gpu_of(hsa_agent_t a) {
  if (a.handle < kGpuAgentBase) return -1;
  const uint64_t i = a.handle - kGpuAgentBase;
  return i < static_cast<uint64_t>(shared::device_count()) ? static_cast<int>(i) : -1;
}
bool valid_agent(hsa_agent_t a) { return a.handle == kCpuAgent || gpu_of(a) >= 0; }
hsa_agent_t gpu_agent(int i) { return {kGpuAgentBase + static_cast<uint64_t>(i)}; }

// The pools each agent has. The CPU's: system memory, fine-grained and the
// one kernel arguments come from; and system memory, coarse-grained. A GPU's:
// its own memory, coarse-grained.
enum class PoolKind { SystemFine, SystemCoarse, Device };
struct PoolId {
  hsa_agent_t agent;
  PoolKind kind;
};
bool pool_of(uint64_t handle, PoolId* out) {
  const hsa_agent_t agent{handle / 16};
  const uint64_t index = handle % 16;
  if (agent.handle == kCpuAgent && index < 2) {
    *out = {agent, index == 0 ? PoolKind::SystemFine : PoolKind::SystemCoarse};
    return true;
  }
  if (gpu_of(agent) >= 0 && index == 0) {
    *out = {agent, PoolKind::Device};
    return true;
  }
  return false;
}
std::vector<uint64_t> pools_of(hsa_agent_t a) {
  if (a.handle == kCpuAgent) return {a.handle * 16, a.handle * 16 + 1};
  return {a.handle * 16};
}

// Host memory the runtime allocated, which it maps into every device.
std::map<uintptr_t, size_t> g_system;   // under g_mutex
// Device memory the runtime allocated, by device.
std::map<uint64_t, int> g_device;       // under g_mutex

uint64_t now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// ---- Signals -----------------------------------------------------------------

struct Signal {
  std::atomic<int64_t> value{0};
  std::mutex mu;
  std::condition_variable cv;
  // Changes the value and wakes whoever waits on it -- under the lock, since
  // a waiter that sees the value may destroy the signal as soon as it has the
  // lock back, and must not do that while this is still waking it.
  template <typename F>
  int64_t update(F f) {
    std::lock_guard<std::mutex> lock(mu);
    const int64_t old = value.load();
    value.store(f(old));
    cv.notify_all();
    return old;
  }
};
Signal* signal_of(hsa_signal_t s) { return reinterpret_cast<Signal*>(s.handle); }

bool satisfied(int64_t v, hsa_signal_condition_t c, int64_t compare) {
  switch (c) {
    case HSA_SIGNAL_CONDITION_EQ: return v == compare;
    case HSA_SIGNAL_CONDITION_NE: return v != compare;
    case HSA_SIGNAL_CONDITION_LT: return v < compare;
    case HSA_SIGNAL_CONDITION_GTE: return v >= compare;
  }
  return false;
}
// Waits until the condition holds or `timeout` nanoseconds pass (the
// timestamp frequency is 1 GHz), and returns the value it saw last.
int64_t wait(Signal* s, hsa_signal_condition_t c, int64_t compare, uint64_t timeout) {
  std::unique_lock<std::mutex> lock(s->mu);
  const auto pred = [&] { return satisfied(s->value.load(), c, compare); };
  if (timeout == UINT64_MAX || timeout > (uint64_t{1} << 62)) s->cv.wait(lock, pred);
  else s->cv.wait_for(lock, std::chrono::nanoseconds(timeout), pred);
  return s->value.load();
}

// ---- Code objects and executables ------------------------------------------------

struct Reader {
  std::string bytes;
};
struct Symbol {
  hsa_symbol_kind_t kind;
  std::string name;   // as HSA names it: a kernel's is "<name>.kd"
  int device;
  const shared::Loaded* loaded;
  const vgpu::amd::Kernel* kernel;   // a kernel's
  uint64_t address = 0, size = 0;    // a variable's
};
struct Executable {
  bool frozen = false;
  std::vector<const shared::Loaded*> loaded;
  std::deque<Symbol> symbols;   // a symbol's address is its handle
};
std::set<Executable*> g_executables;   // under g_mutex

// A kernel object -- the address of a kernel's descriptor on its device --
// and what it names.
struct KernelRef {
  int device;
  const shared::Loaded* loaded;
  const vgpu::amd::Kernel* kernel;
};
std::map<uint64_t, KernelRef> g_kernels;   // under g_mutex

Executable* executable_of(hsa_executable_t e) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto* p = reinterpret_cast<Executable*>(e.handle);
  return g_executables.count(p) ? p : nullptr;
}

// ---- Queues ------------------------------------------------------------------

struct Queue;
void process(Queue* q);

struct Queue {
  hsa_queue_t q{};   // first: the program's hsa_queue_t* is this
  int device = 0;
  std::atomic<uint64_t> write_index{0}, read_index{0};
  std::unique_ptr<Signal> doorbell = std::make_unique<Signal>();
  void (*callback)(hsa_status_t, hsa_queue_t*, void*) = nullptr;
  void* data = nullptr;
  std::vector<uint8_t> ring_storage;
  bool stop = false;   // under doorbell->mu
  std::thread worker;
};

// The packet processor's view of the header: its type, read as the hardware
// reads it, atomically, since the program writes it last to publish the packet.
uint16_t header_of(const void* packet) {
  return __atomic_load_n(static_cast<const uint16_t*>(packet), __ATOMIC_ACQUIRE);
}

void complete(hsa_signal_t s) {
  if (s.handle) signal_of(s)->update([](int64_t v) { return v - 1; });
}

// Runs one kernel dispatch packet. The grid is given in work-items; a grid
// that is not a whole number of work-groups is refused, which this does not
// model yet.
hsa_status_t dispatch(Queue* q, const hsa_kernel_dispatch_packet_t& p, std::string* why) {
  KernelRef ref;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_kernels.find(p.kernel_object);
    if (it == g_kernels.end()) {
      *why = "the packet's kernel_object is no kernel an executable loaded";
      return HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
    }
    ref = it->second;
  }
  if (ref.device != q->device) {
    *why = "the kernel was loaded for another agent than this queue's";
    return HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
  }
  const unsigned dims = p.setup & 3;
  const uint32_t wg[3] = {p.workgroup_size_x, dims > 1 ? p.workgroup_size_y : 1u, dims > 2 ? p.workgroup_size_z : 1u};
  const uint32_t grid[3] = {p.grid_size_x, dims > 1 ? p.grid_size_y : 1u, dims > 2 ? p.grid_size_z : 1u};
  uint32_t groups[3];
  for (int i = 0; i < 3; ++i) {
    if (!wg[i] || !grid[i]) {
      *why = "a dispatch with a zero grid or work-group size";
      return HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
    }
    if (grid[i] % wg[i]) {
      *why = "a grid of " + std::to_string(grid[i]) + " work-items is not a whole number of work-groups of " +
             std::to_string(wg[i]) + ", which this does not run yet";
      return HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
    }
    groups[i] = grid[i] / wg[i];
  }
  // What the packet asks for beyond the kernel's own LDS is the launch's.
  const uint32_t lds = p.group_segment_size > ref.kernel->group_segment ? p.group_segment_size - ref.kernel->group_segment : 0;
  if (!shared::run(q->device, ref.loaded, *ref.kernel, groups, wg, lds, reinterpret_cast<uint64_t>(p.kernarg_address),
                   why))
    return HSA_STATUS_ERROR_EXCEPTION;
  return HSA_STATUS_SUCCESS;
}

// The packet processor: takes each packet once the program has published it
// (its header's type is no longer INVALID), in order, until the queue is
// destroyed.
void process(Queue* q) {
  const uint32_t size = q->q.size;
  for (;;) {
    const uint64_t at = q->read_index.load();
    uint8_t* packet = static_cast<uint8_t*>(q->q.base_address) + (at % size) * 64;
    {
      std::unique_lock<std::mutex> lock(q->doorbell->mu);
      // Woken by the doorbell; a program that publishes a packet without
      // ringing is still seen, a moment later.
      q->doorbell->cv.wait_for(lock, std::chrono::milliseconds(20), [&] {
        return q->stop || (at < q->write_index.load() && (header_of(packet) & 0xFF) != HSA_PACKET_TYPE_INVALID);
      });
      if (q->stop) return;
      if (at >= q->write_index.load() || (header_of(packet) & 0xFF) == HSA_PACKET_TYPE_INVALID) continue;
    }
    const uint16_t header = header_of(packet);
    const unsigned type = header & 0xFF;
    hsa_status_t status = HSA_STATUS_SUCCESS;
    std::string why;
    hsa_signal_t completion{0};
    if (type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
      hsa_kernel_dispatch_packet_t p;
      std::memcpy(&p, packet, sizeof p);
      completion = p.completion_signal;
      status = dispatch(q, p, &why);
    } else if (type == HSA_PACKET_TYPE_BARRIER_AND || type == HSA_PACKET_TYPE_BARRIER_OR) {
      hsa_barrier_and_packet_t p;
      std::memcpy(&p, packet, sizeof p);
      completion = p.completion_signal;
      std::vector<Signal*> deps;
      for (const hsa_signal_t& d : p.dep_signal)
        if (d.handle) deps.push_back(signal_of(d));
      if (type == HSA_PACKET_TYPE_BARRIER_AND) {
        for (Signal* d : deps) wait(d, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX);
      } else if (!deps.empty()) {
        // Any one of them reaching zero.
        for (bool any = false; !any;) {
          for (Signal* d : deps) any = any || d->value.load() == 0;
          if (!any) wait(deps.front(), HSA_SIGNAL_CONDITION_EQ, 0, 1000000);
        }
      }
    } else {
      status = HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
      why = "a packet of type " + std::to_string(type) + ", which this does not process";
    }
    if (status != HSA_STATUS_SUCCESS) {
      // As ROCm's runtime does: the queue's callback hears of it; with none,
      // the error ends the program, since whatever waits on the packet would
      // wait forever.
      fail(status, why);
      if (q->callback) q->callback(status, &q->q, q->data);
      else std::abort();
    }
    // The packet is the program's again, and what waits on it may go on.
    __atomic_store_n(reinterpret_cast<uint16_t*>(packet), static_cast<uint16_t>(HSA_PACKET_TYPE_INVALID),
                     __ATOMIC_RELEASE);
    q->read_index.store(at + 1);
    complete(completion);
  }
}

Queue* queue_of(const hsa_queue_t* q) { return reinterpret_cast<Queue*>(const_cast<hsa_queue_t*>(q)); }

// Asynchronous copies, one after another on a thread of their own: the
// system's copy engine.
vgpu::amd::WorkQueue<int>& copy_engine() {
  static vgpu::amd::WorkQueue<int> q(0, 1);
  return q;
}

template <typename T>
void put(void* value, T v) {
  std::memcpy(value, &v, sizeof v);
}
void put_string(void* value, const std::string& s, size_t room) {
  std::memset(value, 0, room);
  std::memcpy(value, s.data(), std::min(s.size(), room - 1));
}

std::string isa_name(int device) { return "amdgcn-amd-amdhsa--" + shared::profile(device).gcn_arch_full; }

}  // namespace

extern "C" {

// ---- The system ---------------------------------------------------------------

hsa_status_t hsa_init(void) {
  std::string why;
  if (!shared::start(&why)) return fail(HSA_STATUS_ERROR_OUT_OF_RESOURCES, why);
  std::lock_guard<std::mutex> lock(g_mutex);
  ++g_refs;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_shut_down(void) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_refs) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  --g_refs;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_status_string(hsa_status_t status, const char** out) {
  if (!out) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  static const std::map<int, const char*> text = {
      {HSA_STATUS_SUCCESS, "HSA_STATUS_SUCCESS: The function has been executed successfully."},
      {HSA_STATUS_INFO_BREAK, "HSA_STATUS_INFO_BREAK: A traversal over a list of elements has been interrupted."},
      {HSA_STATUS_ERROR, "HSA_STATUS_ERROR: A generic error has occurred."},
      {HSA_STATUS_ERROR_INVALID_ARGUMENT, "HSA_STATUS_ERROR_INVALID_ARGUMENT: One of the actual arguments does not "
                                          "meet a precondition stated in the documentation of the corresponding formal "
                                          "argument."},
      {HSA_STATUS_ERROR_INVALID_QUEUE_CREATION, "HSA_STATUS_ERROR_INVALID_QUEUE_CREATION: The requested queue "
                                                "creation is not valid."},
      {HSA_STATUS_ERROR_INVALID_ALLOCATION, "HSA_STATUS_ERROR_INVALID_ALLOCATION: The requested allocation is not "
                                            "valid."},
      {HSA_STATUS_ERROR_INVALID_AGENT, "HSA_STATUS_ERROR_INVALID_AGENT: The agent is invalid."},
      {HSA_STATUS_ERROR_INVALID_REGION, "HSA_STATUS_ERROR_INVALID_REGION: The memory region is invalid."},
      {HSA_STATUS_ERROR_INVALID_SIGNAL, "HSA_STATUS_ERROR_INVALID_SIGNAL: The signal is invalid."},
      {HSA_STATUS_ERROR_INVALID_QUEUE, "HSA_STATUS_ERROR_INVALID_QUEUE: The queue is invalid."},
      {HSA_STATUS_ERROR_OUT_OF_RESOURCES, "HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate the "
                                          "necessary resources."},
      {HSA_STATUS_ERROR_INVALID_PACKET_FORMAT, "HSA_STATUS_ERROR_INVALID_PACKET_FORMAT: The AQL packet is "
                                               "malformed."},
      {HSA_STATUS_ERROR_NOT_INITIALIZED, "HSA_STATUS_ERROR_NOT_INITIALIZED: An API other than hsa_init has been "
                                         "invoked while the reference count of the HSA runtime is zero."},
      {HSA_STATUS_ERROR_INVALID_CODE_OBJECT, "HSA_STATUS_ERROR_INVALID_CODE_OBJECT: The code object is invalid."},
      {HSA_STATUS_ERROR_INVALID_EXECUTABLE, "HSA_STATUS_ERROR_INVALID_EXECUTABLE: The executable is invalid."},
      {HSA_STATUS_ERROR_FROZEN_EXECUTABLE, "HSA_STATUS_ERROR_FROZEN_EXECUTABLE: The executable is frozen."},
      {HSA_STATUS_ERROR_INVALID_SYMBOL_NAME, "HSA_STATUS_ERROR_INVALID_SYMBOL_NAME: There is no symbol with the "
                                             "given name."},
      {HSA_STATUS_ERROR_EXCEPTION, "HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware "
                                   "exception."},
      {HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL, "HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL: The executable "
                                                   "symbol is invalid."},
      {HSA_STATUS_ERROR_INVALID_FILE, "HSA_STATUS_ERROR_INVALID_FILE: The file descriptor is invalid."},
      {HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER, "HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER: The code object "
                                                    "reader is invalid."},
      {HSA_STATUS_ERROR_INVALID_MEMORY_POOL, "HSA_STATUS_ERROR_INVALID_MEMORY_POOL: The memory pool is invalid."},
      {HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION, "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent "
                                                   "attempted to access memory beyond the largest legal address."},
      {HSA_STATUS_ERROR_NOT_SUPPORTED, "HSA_STATUS_ERROR_NOT_SUPPORTED: The requested feature is not supported."},
  };
  const auto it = text.find(status);
  if (it == text.end()) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *out = it->second;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_system_get_info(hsa_system_info_t attribute, void* value) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!value) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  switch (attribute) {
    case HSA_SYSTEM_INFO_VERSION_MAJOR: put<uint16_t>(value, 1); break;
    case HSA_SYSTEM_INFO_VERSION_MINOR: put<uint16_t>(value, 1); break;
    case HSA_SYSTEM_INFO_TIMESTAMP: put<uint64_t>(value, now_ns()); break;
    case HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY: put<uint64_t>(value, 1000000000); break;
    case HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT: put<uint64_t>(value, UINT64_MAX); break;
    case HSA_SYSTEM_INFO_ENDIANNESS: put<uint32_t>(value, HSA_ENDIANNESS_LITTLE); break;
    case HSA_SYSTEM_INFO_MACHINE_MODEL: put<uint32_t>(value, HSA_MACHINE_MODEL_LARGE); break;
    case HSA_SYSTEM_INFO_EXTENSIONS: std::memset(value, 0, 128); break;
    default: return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return HSA_STATUS_SUCCESS;
}

// ---- Agents -------------------------------------------------------------------

hsa_status_t hsa_iterate_agents(hsa_status_t (*callback)(hsa_agent_t, void*), void* data) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!callback) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (const hsa_status_t s = callback({kCpuAgent}, data); s != HSA_STATUS_SUCCESS) return s;
  for (int i = 0; i < shared::device_count(); ++i)
    if (const hsa_status_t s = callback(gpu_agent(i), data); s != HSA_STATUS_SUCCESS) return s;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute, void* value) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  if (!value) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const int gpu = gpu_of(agent);
  const bool is_gpu = gpu >= 0;
  const unsigned cpus = std::max(1u, std::thread::hardware_concurrency());
  const int attr = attribute;
  switch (attr) {
    case HSA_AGENT_INFO_NAME: put_string(value, is_gpu ? shared::profile(gpu).gcn_arch : "VirtualGPU host CPU", 64); break;
    case HSA_AGENT_INFO_VENDOR_NAME: put_string(value, is_gpu ? "AMD" : "CPU", 64); break;
    case HSA_AGENT_INFO_FEATURE: put<uint32_t>(value, is_gpu ? HSA_AGENT_FEATURE_KERNEL_DISPATCH : 0); break;
    case HSA_AGENT_INFO_MACHINE_MODEL: put<uint32_t>(value, HSA_MACHINE_MODEL_LARGE); break;
    case HSA_AGENT_INFO_PROFILE: put<uint32_t>(value, HSA_PROFILE_BASE); break;
    case HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE: put<uint32_t>(value, HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR); break;
    case HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES:
      put<uint32_t>(value, HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR);
      break;
    case HSA_AGENT_INFO_FAST_F16_OPERATION: put<bool>(value, is_gpu); break;
    case HSA_AGENT_INFO_WAVEFRONT_SIZE: put<uint32_t>(value, is_gpu ? shared::profile(gpu).warp_size : 0); break;
    case HSA_AGENT_INFO_WORKGROUP_MAX_DIM: {
      const uint16_t n = is_gpu ? 1024 : 0;
      const uint16_t dims[3] = {n, n, n};
      std::memcpy(value, dims, sizeof dims);
      break;
    }
    case HSA_AGENT_INFO_WORKGROUP_MAX_SIZE: put<uint32_t>(value, is_gpu ? 1024 : 0); break;
    case HSA_AGENT_INFO_GRID_MAX_DIM: {
      const hsa_dim3_t d = is_gpu ? hsa_dim3_t{UINT32_MAX, UINT32_MAX, UINT32_MAX} : hsa_dim3_t{0, 0, 0};
      put(value, d);
      break;
    }
    case HSA_AGENT_INFO_GRID_MAX_SIZE: put<uint32_t>(value, is_gpu ? UINT32_MAX : 0); break;
    case HSA_AGENT_INFO_FBARRIER_MAX_SIZE: put<uint32_t>(value, is_gpu ? 32 : 0); break;
    case HSA_AGENT_INFO_QUEUES_MAX: put<uint32_t>(value, is_gpu ? 128 : 0); break;
    case HSA_AGENT_INFO_QUEUE_MIN_SIZE: put<uint32_t>(value, is_gpu ? 64 : 0); break;
    case HSA_AGENT_INFO_QUEUE_MAX_SIZE: put<uint32_t>(value, is_gpu ? 131072 : 0); break;
    case HSA_AGENT_INFO_QUEUE_TYPE: put<uint32_t>(value, HSA_QUEUE_TYPE_MULTI); break;
    case HSA_AGENT_INFO_NODE: put<uint32_t>(value, is_gpu ? static_cast<uint32_t>(gpu + 1) : 0); break;
    case HSA_AGENT_INFO_DEVICE: put<uint32_t>(value, is_gpu ? HSA_DEVICE_TYPE_GPU : HSA_DEVICE_TYPE_CPU); break;
    case HSA_AGENT_INFO_CACHE_SIZE: {
      uint32_t sizes[4] = {0, 0, 0, 0};
      if (is_gpu) sizes[0] = 16 * 1024, sizes[1] = static_cast<uint32_t>(shared::profile(gpu).limits.l2_cache_bytes);
      std::memcpy(value, sizes, sizeof sizes);
      break;
    }
    case HSA_AGENT_INFO_ISA: put<hsa_isa_t>(value, {is_gpu ? agent.handle : 0}); break;
    case HSA_AGENT_INFO_EXTENSIONS: std::memset(value, 0, 128); break;
    case HSA_AGENT_INFO_VERSION_MAJOR: put<uint16_t>(value, 1); break;
    case HSA_AGENT_INFO_VERSION_MINOR: put<uint16_t>(value, 1); break;
    case HSA_AMD_AGENT_INFO_CHIP_ID: put<uint32_t>(value, is_gpu ? 0x74a1 : 0); break;
    case HSA_AMD_AGENT_INFO_CACHELINE_SIZE: put<uint32_t>(value, is_gpu ? 128 : 64); break;
    case HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT:
      put<uint32_t>(value, is_gpu ? static_cast<uint32_t>(shared::profile(gpu).limits.multiprocessors) : cpus);
      break;
    case HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY:
      put<uint32_t>(value, is_gpu ? static_cast<uint32_t>(shared::profile(gpu).telemetry.sm_clock_max_mhz) : 0);
      break;
    case HSA_AMD_AGENT_INFO_DRIVER_NODE_ID: put<uint32_t>(value, is_gpu ? static_cast<uint32_t>(gpu + 1) : 0); break;
    // The PCI location: bus ordinal + 1, device 0, function 0, as HIP reports it.
    case HSA_AMD_AGENT_INFO_BDFID: put<uint32_t>(value, is_gpu ? static_cast<uint32_t>(gpu + 1) << 8 : 0); break;
    case HSA_AMD_AGENT_INFO_DOMAIN: put<uint32_t>(value, 0); break;
    case HSA_AMD_AGENT_INFO_PRODUCT_NAME:
      put_string(value, is_gpu ? shared::profile(gpu).model : "VirtualGPU host CPU", 64);
      break;
    case HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU: put<uint32_t>(value, is_gpu ? 32 : 0); break;
    case HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU: put<uint32_t>(value, is_gpu ? 4 : 0); break;
    case HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES: put<bool>(value, false); break;
    case HSA_AMD_AGENT_INFO_UUID: {
      char uuid[21];
      if (is_gpu) std::snprintf(uuid, sizeof uuid, "GPU-%016llx", 0x5647505500000000ull + static_cast<unsigned>(gpu));
      else std::snprintf(uuid, sizeof uuid, "CPU-XX");
      put_string(value, uuid, 21);
      break;
    }
    case HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS: put<bool>(value, false); break;
    case HSA_AMD_AGENT_INFO_MEMORY_AVAIL:
      put<uint64_t>(value, is_gpu ? shared::profile(gpu).vram_bytes : 0);
      break;
    case HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY: put<uint64_t>(value, 1000000000); break;
    default: return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return HSA_STATUS_SUCCESS;
}

// ---- Instruction sets -------------------------------------------------------------
//
// A GPU agent's ISA handle is its agent's handle.

hsa_status_t hsa_agent_iterate_isas(hsa_agent_t agent, hsa_status_t (*callback)(hsa_isa_t, void*), void* data) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  if (!callback) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (gpu_of(agent) < 0) return HSA_STATUS_SUCCESS;
  return callback({agent.handle}, data);
}

hsa_status_t hsa_isa_get_info_alt(hsa_isa_t isa, hsa_isa_info_t attribute, void* value) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  const int gpu = gpu_of({isa.handle});
  if (gpu < 0) return HSA_STATUS_ERROR_INVALID_ISA;
  if (!value) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const std::string name = isa_name(gpu);
  switch (attribute) {
    case HSA_ISA_INFO_NAME_LENGTH: put<uint32_t>(value, static_cast<uint32_t>(name.size())); break;
    case HSA_ISA_INFO_NAME: std::memcpy(value, name.data(), name.size()); break;
    case HSA_ISA_INFO_MACHINE_MODELS: {
      const bool models[2] = {false, true};
      std::memcpy(value, models, sizeof models);
      break;
    }
    case HSA_ISA_INFO_PROFILES: {
      const bool profiles[2] = {true, false};
      std::memcpy(value, profiles, sizeof profiles);
      break;
    }
    case HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES:
    case HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES: {
      const bool modes[3] = {false, false, true};
      std::memcpy(value, modes, sizeof modes);
      break;
    }
    case HSA_ISA_INFO_FAST_F16_OPERATION: put<bool>(value, true); break;
    case HSA_ISA_INFO_WORKGROUP_MAX_DIM: {
      const uint16_t dims[3] = {1024, 1024, 1024};
      std::memcpy(value, dims, sizeof dims);
      break;
    }
    case HSA_ISA_INFO_WORKGROUP_MAX_SIZE: put<uint32_t>(value, 1024); break;
    case HSA_ISA_INFO_GRID_MAX_DIM: put(value, hsa_dim3_t{UINT32_MAX, UINT32_MAX, UINT32_MAX}); break;
    case HSA_ISA_INFO_GRID_MAX_SIZE: put<uint64_t>(value, UINT64_MAX); break;
    case HSA_ISA_INFO_FBARRIER_MAX_SIZE: put<uint32_t>(value, 32); break;
    default: return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_isa_from_name(const char* name, hsa_isa_t* isa) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!name || !isa) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (int i = 0; i < shared::device_count(); ++i)
    if (isa_name(i) == name) {
      isa->handle = gpu_agent(i).handle;
      return HSA_STATUS_SUCCESS;
    }
  return HSA_STATUS_ERROR_INVALID_ISA_NAME;
}

// ---- Signals ------------------------------------------------------------------

hsa_status_t hsa_signal_create(hsa_signal_value_t initial, uint32_t, const hsa_agent_t*, hsa_signal_t* signal) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!signal) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  auto* s = new Signal;
  s->value = initial;
  signal->handle = reinterpret_cast<uint64_t>(s);
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_amd_signal_create(hsa_signal_value_t initial, uint32_t n, const hsa_agent_t* consumers, uint64_t,
                                   hsa_signal_t* signal) {
  return hsa_signal_create(initial, n, consumers, signal);
}
hsa_status_t hsa_signal_destroy(hsa_signal_t signal) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!signal.handle) return HSA_STATUS_ERROR_INVALID_SIGNAL;
  delete signal_of(signal);
  return HSA_STATUS_SUCCESS;
}

// Every memory order the specification names does the same here: each
// operation is sequentially consistent, and wakes every waiter.
#define VGPU_SIGNAL_LOAD(order) \
  hsa_signal_value_t hsa_signal_load_##order(hsa_signal_t s) { return signal_of(s)->value.load(); }
VGPU_SIGNAL_LOAD(relaxed)
VGPU_SIGNAL_LOAD(scacquire)
VGPU_SIGNAL_LOAD(acquire)
#define VGPU_SIGNAL_STORE(name)                                                  \
  void hsa_signal_##name(hsa_signal_t s, hsa_signal_value_t v) {                 \
    signal_of(s)->update([v](int64_t) { return v; });                            \
  }
VGPU_SIGNAL_STORE(store_relaxed)
VGPU_SIGNAL_STORE(store_screlease)
VGPU_SIGNAL_STORE(store_release)
VGPU_SIGNAL_STORE(silent_store_relaxed)
VGPU_SIGNAL_STORE(silent_store_screlease)
#define VGPU_SIGNAL_RMW(op, order, expr)                                         \
  void hsa_signal_##op##_##order(hsa_signal_t s, hsa_signal_value_t v) {         \
    signal_of(s)->update([v](int64_t old) { return expr; });                     \
  }
#define VGPU_SIGNAL_RMW_ALL(op, expr)            \
  VGPU_SIGNAL_RMW(op, relaxed, expr)             \
  VGPU_SIGNAL_RMW(op, scacquire, expr)           \
  VGPU_SIGNAL_RMW(op, screlease, expr)           \
  VGPU_SIGNAL_RMW(op, scacq_screl, expr)         \
  VGPU_SIGNAL_RMW(op, acquire, expr)             \
  VGPU_SIGNAL_RMW(op, release, expr)             \
  VGPU_SIGNAL_RMW(op, acq_rel, expr)
VGPU_SIGNAL_RMW_ALL(add, old + v)
VGPU_SIGNAL_RMW_ALL(subtract, old - v)
VGPU_SIGNAL_RMW_ALL(and, old & v)
VGPU_SIGNAL_RMW_ALL(or, old | v)
VGPU_SIGNAL_RMW_ALL(xor, old ^ v)
#define VGPU_SIGNAL_EXCHANGE(order)                                                   \
  hsa_signal_value_t hsa_signal_exchange_##order(hsa_signal_t s, hsa_signal_value_t v) { \
    return signal_of(s)->update([v](int64_t) { return v; });                            \
  }
VGPU_SIGNAL_EXCHANGE(relaxed)
VGPU_SIGNAL_EXCHANGE(scacquire)
VGPU_SIGNAL_EXCHANGE(screlease)
VGPU_SIGNAL_EXCHANGE(scacq_screl)
VGPU_SIGNAL_EXCHANGE(acquire)
VGPU_SIGNAL_EXCHANGE(release)
VGPU_SIGNAL_EXCHANGE(acq_rel)
#define VGPU_SIGNAL_CAS(order)                                                                             \
  hsa_signal_value_t hsa_signal_cas_##order(hsa_signal_t s, hsa_signal_value_t expected, hsa_signal_value_t v) { \
    return signal_of(s)->update([=](int64_t old) { return old == expected ? v : old; });                         \
  }
VGPU_SIGNAL_CAS(relaxed)
VGPU_SIGNAL_CAS(scacquire)
VGPU_SIGNAL_CAS(screlease)
VGPU_SIGNAL_CAS(scacq_screl)
VGPU_SIGNAL_CAS(acquire)
VGPU_SIGNAL_CAS(release)
VGPU_SIGNAL_CAS(acq_rel)
#define VGPU_SIGNAL_WAIT(order)                                                                        \
  hsa_signal_value_t hsa_signal_wait_##order(hsa_signal_t s, hsa_signal_condition_t c, hsa_signal_value_t v, \
                                             uint64_t timeout, hsa_wait_state_t) {                     \
    return wait(signal_of(s), c, v, timeout);                                                          \
  }
VGPU_SIGNAL_WAIT(relaxed)
VGPU_SIGNAL_WAIT(scacquire)
VGPU_SIGNAL_WAIT(acquire)

// ---- Queues -------------------------------------------------------------------

hsa_status_t hsa_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                              void (*callback)(hsa_status_t, hsa_queue_t*, void*), void* data, uint32_t, uint32_t,
                              hsa_queue_t** queue) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  const int gpu = gpu_of(agent);
  if (gpu < 0) return fail(HSA_STATUS_ERROR_INVALID_QUEUE_CREATION, "the CPU agent takes no AQL packets");
  if (!queue || size < 64 || size > 131072 || (size & (size - 1)) || type > HSA_QUEUE_TYPE_SINGLE)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  auto* q = new Queue;
  q->device = gpu;
  q->callback = callback;
  q->data = data;
  // The ring, 64-byte aligned, every packet INVALID until the program writes it.
  q->ring_storage.assign(size_t{size} * 64 + 64, 0);
  auto* ring = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(q->ring_storage.data()) + 63) & ~uintptr_t{63});
  for (uint32_t i = 0; i < size; ++i) {
    const uint16_t invalid = HSA_PACKET_TYPE_INVALID;
    std::memcpy(ring + size_t{i} * 64, &invalid, 2);
  }
  static std::atomic<uint64_t> next_id{0};
  q->q.type = type;
  q->q.features = HSA_QUEUE_FEATURE_KERNEL_DISPATCH;
  q->q.base_address = ring;
  q->q.doorbell_signal.handle = reinterpret_cast<uint64_t>(q->doorbell.get());
  q->q.size = size;
  q->q.id = next_id++;
  q->worker = std::thread(process, q);
  *queue = &q->q;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_queue_destroy(hsa_queue_t* queue) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!queue) return HSA_STATUS_ERROR_INVALID_QUEUE;
  Queue* q = queue_of(queue);
  {
    std::lock_guard<std::mutex> lock(q->doorbell->mu);
    q->stop = true;
  }
  q->doorbell->cv.notify_all();
  q->worker.join();
  delete q;
  return HSA_STATUS_SUCCESS;
}

#define VGPU_QUEUE_LOAD(which, order) \
  uint64_t hsa_queue_load_##which##_index_##order(const hsa_queue_t* q) { return queue_of(q)->which##_index.load(); }
VGPU_QUEUE_LOAD(read, relaxed)
VGPU_QUEUE_LOAD(read, scacquire)
VGPU_QUEUE_LOAD(read, acquire)
VGPU_QUEUE_LOAD(write, relaxed)
VGPU_QUEUE_LOAD(write, scacquire)
VGPU_QUEUE_LOAD(write, acquire)
#define VGPU_QUEUE_STORE(which, order)                                                   \
  void hsa_queue_store_##which##_index_##order(const hsa_queue_t* q, uint64_t v) {      \
    queue_of(q)->which##_index.store(v);                                                 \
    queue_of(q)->doorbell->cv.notify_all();                                              \
  }
VGPU_QUEUE_STORE(write, relaxed)
VGPU_QUEUE_STORE(write, screlease)
VGPU_QUEUE_STORE(write, release)
VGPU_QUEUE_STORE(read, relaxed)
VGPU_QUEUE_STORE(read, screlease)
VGPU_QUEUE_STORE(read, release)
#define VGPU_QUEUE_ADD(order)                                                            \
  uint64_t hsa_queue_add_write_index_##order(const hsa_queue_t* q, uint64_t v) {        \
    return queue_of(q)->write_index.fetch_add(v);                                        \
  }
VGPU_QUEUE_ADD(relaxed)
VGPU_QUEUE_ADD(scacquire)
VGPU_QUEUE_ADD(screlease)
VGPU_QUEUE_ADD(scacq_screl)
VGPU_QUEUE_ADD(acquire)
VGPU_QUEUE_ADD(release)
VGPU_QUEUE_ADD(acq_rel)
#define VGPU_QUEUE_CAS(order)                                                                   \
  uint64_t hsa_queue_cas_write_index_##order(const hsa_queue_t* q, uint64_t expected, uint64_t v) { \
    queue_of(q)->write_index.compare_exchange_strong(expected, v);                              \
    return expected;                                                                            \
  }
VGPU_QUEUE_CAS(relaxed)
VGPU_QUEUE_CAS(scacquire)
VGPU_QUEUE_CAS(screlease)
VGPU_QUEUE_CAS(scacq_screl)
VGPU_QUEUE_CAS(acquire)
VGPU_QUEUE_CAS(release)
VGPU_QUEUE_CAS(acq_rel)

// ---- Memory ------------------------------------------------------------------

hsa_status_t hsa_amd_agent_iterate_memory_pools(hsa_agent_t agent,
                                                hsa_status_t (*callback)(hsa_amd_memory_pool_t, void*), void* data) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  if (!callback) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (uint64_t p : pools_of(agent))
    if (const hsa_status_t s = callback({p}, data); s != HSA_STATUS_SUCCESS) return s;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_memory_pool_get_info(hsa_amd_memory_pool_t pool, hsa_amd_memory_pool_info_t attribute,
                                          void* value) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  PoolId id;
  if (!pool_of(pool.handle, &id)) return HSA_STATUS_ERROR_INVALID_MEMORY_POOL;
  if (!value) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const bool device = id.kind == PoolKind::Device;
  const uint64_t size = device ? shared::profile(gpu_of(id.agent)).vram_bytes
                               : static_cast<uint64_t>(sysconf(_SC_PHYS_PAGES)) * sysconf(_SC_PAGESIZE);
  switch (attribute) {
    case HSA_AMD_MEMORY_POOL_INFO_SEGMENT: put<uint32_t>(value, HSA_AMD_SEGMENT_GLOBAL); break;
    case HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS:
      put<uint32_t>(value, id.kind == PoolKind::SystemFine
                               ? HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED | HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT
                               : HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED);
      break;
    case HSA_AMD_MEMORY_POOL_INFO_SIZE: put<size_t>(value, size); break;
    case HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED: put<bool>(value, true); break;
    case HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE:
    case HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT:
    case HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE: put<size_t>(value, 4096); break;
    case HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL: put<bool>(value, !device); break;
    case HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE: put<size_t>(value, size); break;
    case HSA_AMD_MEMORY_POOL_INFO_LOCATION:
      put<uint32_t>(value, device ? HSA_AMD_MEMORY_POOL_LOCATION_GPU : HSA_AMD_MEMORY_POOL_LOCATION_CPU);
      break;
    default: return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_agent_memory_pool_get_info(hsa_agent_t agent, hsa_amd_memory_pool_t pool,
                                                hsa_amd_agent_memory_pool_info_t attribute, void* value) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  PoolId id;
  if (!pool_of(pool.handle, &id)) return HSA_STATUS_ERROR_INVALID_MEMORY_POOL;
  if (!value) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  // Everyone reaches system memory; a GPU's own memory is its own, and
  // another agent reaches it once hsa_amd_agents_allow_access says so.
  const bool own = id.kind != PoolKind::Device || id.agent.handle == agent.handle;
  switch (attribute) {
    case HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS:
      put<uint32_t>(value, own ? HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT
                               : HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT);
      break;
    case HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS: put<uint32_t>(value, own ? 0 : 1); break;
    default: return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t pool, size_t size, uint32_t, void** ptr) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  PoolId id;
  if (!pool_of(pool.handle, &id)) return HSA_STATUS_ERROR_INVALID_MEMORY_POOL;
  if (!ptr || !size) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (id.kind == PoolKind::Device) {
    const int gpu = gpu_of(id.agent);
    try {
      const uint64_t va = shared::memory(gpu).alloc(size);
      std::lock_guard<std::mutex> lock(g_mutex);
      g_device[va] = gpu;
      *ptr = reinterpret_cast<void*>(va);
    } catch (const std::exception& e) {
      return fail(HSA_STATUS_ERROR_OUT_OF_RESOURCES, e.what());
    }
    return HSA_STATUS_SUCCESS;
  }
  const size_t rounded = (size + 4095) / 4096 * 4096;
  void* p = std::aligned_alloc(4096, rounded);
  if (!p) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  std::memset(p, 0, rounded);
  shared::map_host(p, rounded);
  std::lock_guard<std::mutex> lock(g_mutex);
  g_system[reinterpret_cast<uintptr_t>(p)] = rounded;
  *ptr = p;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_memory_pool_free(void* ptr) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!ptr) return HSA_STATUS_SUCCESS;
  std::unique_lock<std::mutex> lock(g_mutex);
  if (const auto it = g_system.find(reinterpret_cast<uintptr_t>(ptr)); it != g_system.end()) {
    g_system.erase(it);
    lock.unlock();
    shared::unmap_host(ptr);
    std::free(ptr);
    return HSA_STATUS_SUCCESS;
  }
  if (const auto it = g_device.find(reinterpret_cast<uint64_t>(ptr)); it != g_device.end()) {
    const int gpu = it->second;
    g_device.erase(it);
    lock.unlock();
    try {
      shared::memory(gpu).free(reinterpret_cast<uint64_t>(ptr));
    } catch (const std::exception& e) {
      return fail(HSA_STATUS_ERROR_INVALID_ARGUMENT, e.what());
    }
    return HSA_STATUS_SUCCESS;
  }
  return fail(HSA_STATUS_ERROR_INVALID_ARGUMENT, "freeing memory no pool allocated");
}

hsa_status_t hsa_amd_agents_allow_access(uint32_t num_agents, const hsa_agent_t* agents, const uint32_t*,
                                         const void* ptr) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!num_agents || !agents || !ptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (uint32_t i = 0; i < num_agents; ++i)
    if (!valid_agent(agents[i])) return HSA_STATUS_ERROR_INVALID_AGENT;
  // The host reaches every device's memory by copying, which needs no leave,
  // and a device reaches system memory already.
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_memory_copy(void* dst, const void* src, size_t size) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!size) return HSA_STATUS_SUCCESS;
  if (!dst || !src) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::string why;
  if (!shared::copy(dst, src, size, &why)) return fail(HSA_STATUS_ERROR_INVALID_ARGUMENT, why);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_memory_async_copy(void* dst, hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent,
                                       size_t size, uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
                                       hsa_signal_t completion_signal) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!valid_agent(dst_agent) || !valid_agent(src_agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  if (!dst || !src || (num_dep_signals && !dep_signals)) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::vector<Signal*> deps;
  for (uint32_t i = 0; i < num_dep_signals; ++i) deps.push_back(signal_of(dep_signals[i]));
  copy_engine().submit([=] {
    for (Signal* d : deps) wait(d, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX);
    std::string why;
    const bool ok = !size || shared::copy(dst, src, size, &why);
    if (!ok) fail(HSA_STATUS_ERROR_INVALID_ARGUMENT, "an asynchronous copy failed: " + why);
    complete(completion_signal);
    return ok ? 0 : 1;
  });
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_memory_fill(void* ptr, uint32_t value, size_t count) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!ptr || (reinterpret_cast<uintptr_t>(ptr) & 3)) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::vector<uint32_t> words(count, value);
  std::string why;
  if (count && !shared::copy(ptr, words.data(), count * 4, &why)) return fail(HSA_STATUS_ERROR_INVALID_ARGUMENT, why);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_memory_lock(void* host_ptr, size_t size, hsa_agent_t*, int, void** agent_ptr) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!host_ptr || !size || !agent_ptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  shared::map_host(host_ptr, size);
  *agent_ptr = host_ptr;   // devices reach it where it is
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_amd_memory_unlock(void* host_ptr) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  shared::unmap_host(host_ptr);
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_memory_register(void* ptr, size_t size) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!ptr || !size) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  shared::map_host(ptr, size);
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_memory_deregister(void* ptr, size_t) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  shared::unmap_host(ptr);
  return HSA_STATUS_SUCCESS;
}

// The HSA 1.0 regions, which older programs look for: the same memory as the
// pools. The CPU's first is where kernel arguments go.
hsa_status_t hsa_agent_iterate_regions(hsa_agent_t agent, hsa_status_t (*callback)(hsa_region_t, void*), void* data) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  if (!callback) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (uint64_t p : pools_of(agent))
    if (const hsa_status_t s = callback({p}, data); s != HSA_STATUS_SUCCESS) return s;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_region_get_info(hsa_region_t region, hsa_region_info_t attribute, void* value) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  PoolId id;
  if (!pool_of(region.handle, &id)) return HSA_STATUS_ERROR_INVALID_REGION;
  if (!value) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  switch (attribute) {
    case HSA_REGION_INFO_SEGMENT: put<uint32_t>(value, HSA_REGION_SEGMENT_GLOBAL); return HSA_STATUS_SUCCESS;
    case HSA_REGION_INFO_GLOBAL_FLAGS:
      put<uint32_t>(value, id.kind == PoolKind::SystemFine
                               ? HSA_REGION_GLOBAL_FLAG_KERNARG | HSA_REGION_GLOBAL_FLAG_FINE_GRAINED
                               : HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED);
      return HSA_STATUS_SUCCESS;
    case HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE: put<uint32_t>(value, 0); return HSA_STATUS_SUCCESS;
    case HSA_REGION_INFO_SIZE:
      return hsa_amd_memory_pool_get_info({region.handle}, HSA_AMD_MEMORY_POOL_INFO_SIZE, value);
    case HSA_REGION_INFO_ALLOC_MAX_SIZE:
      return hsa_amd_memory_pool_get_info({region.handle}, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE, value);
    case HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED:
      return hsa_amd_memory_pool_get_info({region.handle}, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, value);
    case HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE:
    case HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT: put<size_t>(value, 4096); return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t hsa_memory_allocate(hsa_region_t region, size_t size, void** ptr) {
  return hsa_amd_memory_pool_allocate({region.handle}, size, 0, ptr);
}
hsa_status_t hsa_memory_free(void* ptr) { return hsa_amd_memory_pool_free(ptr); }

// ---- Code objects and executables -------------------------------------------------

hsa_status_t hsa_code_object_reader_create_from_memory(const void* code_object, size_t size,
                                                       hsa_code_object_reader_t* reader) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!code_object || !size || !reader) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  auto* r = new Reader{std::string(static_cast<const char*>(code_object), size)};
  reader->handle = reinterpret_cast<uint64_t>(r);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_code_object_reader_create_from_file(hsa_file_t file, hsa_code_object_reader_t* reader) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!reader) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::string bytes;
  char buf[65536];
  for (ssize_t n; (n = read(file, buf, sizeof buf)) != 0;) {
    if (n < 0) return HSA_STATUS_ERROR_INVALID_FILE;
    bytes.append(buf, static_cast<size_t>(n));
  }
  if (bytes.empty()) return HSA_STATUS_ERROR_INVALID_FILE;
  reader->handle = reinterpret_cast<uint64_t>(new Reader{std::move(bytes)});
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_code_object_reader_destroy(hsa_code_object_reader_t reader) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!reader.handle) return HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER;
  delete reinterpret_cast<Reader*>(reader.handle);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_executable_create_alt(hsa_profile_t, hsa_default_float_rounding_mode_t, const char*,
                                       hsa_executable_t* executable) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!executable) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  auto* e = new Executable;
  std::lock_guard<std::mutex> lock(g_mutex);
  g_executables.insert(e);
  executable->handle = reinterpret_cast<uint64_t>(e);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_executable_load_agent_code_object(hsa_executable_t executable, hsa_agent_t agent,
                                                   hsa_code_object_reader_t reader, const char*,
                                                   hsa_loaded_code_object_t* loaded_code_object) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  Executable* e = executable_of(executable);
  if (!e) return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  if (e->frozen) return HSA_STATUS_ERROR_FROZEN_EXECUTABLE;
  const int gpu = gpu_of(agent);
  if (gpu < 0) return HSA_STATUS_ERROR_INVALID_AGENT;
  if (!reader.handle) return HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER;
  const std::string& bytes = reinterpret_cast<Reader*>(reader.handle)->bytes;
  // A bundle -- what hipcc --genco writes -- carries the code for each
  // target; the device's is the one loaded.
  std::string_view code(bytes);
  std::unique_ptr<vgpu::amd::Bundle> bundle;
  const std::string target = shared::profile(gpu).gcn_arch_full;
  if (bytes.size() >= 24 && vgpu::amd::is_bundle(reinterpret_cast<const uint8_t*>(bytes.data()))) {
    try {
      bundle = vgpu::amd::read_bundle(reinterpret_cast<const uint8_t*>(bytes.data()), target);
    } catch (const std::exception& ex) {
      return fail(HSA_STATUS_ERROR_INVALID_CODE_OBJECT, ex.what());
    }
    const std::string_view* c = bundle ? vgpu::amd::code_for(*bundle, target) : nullptr;
    if (!c) return fail(HSA_STATUS_ERROR_INVALID_ISA, "the bundle carries no code for " + target);
    code = *c;
  }
  std::string why;
  const shared::Loaded* m = shared::load(gpu, code.data(), code.size(), &why);
  if (!m) return fail(HSA_STATUS_ERROR_INVALID_CODE_OBJECT, why);
  const vgpu::amd::CodeObject& o = shared::object(m);
  std::lock_guard<std::mutex> lock(g_mutex);
  e->loaded.push_back(m);
  for (const vgpu::amd::Kernel& k : o.kernels) {
    // A linked object's kernel object is its descriptor on the device, as
    // ROCm's loader gives it; an unlinked one's is any address no other
    // kernel has, since only this runtime reads it.
    const uint64_t object = o.linked ? shared::code_base(m) + k.descriptor
                                     : (uint64_t{1} << 62) + g_kernels.size() * 64;
    g_kernels[object] = {gpu, m, &k};
    Symbol s{HSA_SYMBOL_KIND_KERNEL, k.name + ".kd", gpu, m, &k};
    s.address = object;
    e->symbols.push_back(std::move(s));
  }
  for (const vgpu::amd::GlobalVar& g : o.globals) {
    if (g.name.size() > 3 && g.name.compare(g.name.size() - 3, 3, ".kd") == 0) continue;   // a kernel's, above
    Symbol s{HSA_SYMBOL_KIND_VARIABLE, g.name, gpu, m, nullptr};
    s.address = (o.linked ? shared::code_base(m) : 0) + g.offset;
    s.size = g.size;
    e->symbols.push_back(std::move(s));
  }
  if (loaded_code_object) loaded_code_object->handle = reinterpret_cast<uint64_t>(m);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_executable_freeze(hsa_executable_t executable, const char*) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  Executable* e = executable_of(executable);
  if (!e) return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  if (e->frozen) return HSA_STATUS_ERROR_FROZEN_EXECUTABLE;
  e->frozen = true;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_executable_destroy(hsa_executable_t executable) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  Executable* e = executable_of(executable);
  if (!e) return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_executables.erase(e);
    for (auto it = g_kernels.begin(); it != g_kernels.end();)
      it = std::find(e->loaded.begin(), e->loaded.end(), it->second.loaded) != e->loaded.end() ? g_kernels.erase(it)
                                                                                              : std::next(it);
  }
  for (const shared::Loaded* m : e->loaded) shared::unload(m);
  delete e;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_executable_get_symbol_by_name(hsa_executable_t executable, const char* name,
                                               const hsa_agent_t* agent, hsa_executable_symbol_t* symbol) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  Executable* e = executable_of(executable);
  if (!e) return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  if (!name || !symbol) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const int gpu = agent ? gpu_of(*agent) : -1;
  if (agent && gpu < 0) return HSA_STATUS_ERROR_INVALID_AGENT;
  for (Symbol& s : e->symbols)
    if (s.name == name && (!agent || s.device == gpu)) {
      symbol->handle = reinterpret_cast<uint64_t>(&s);
      return HSA_STATUS_SUCCESS;
    }
  return HSA_STATUS_ERROR_INVALID_SYMBOL_NAME;
}

hsa_status_t hsa_executable_iterate_agent_symbols(
    hsa_executable_t executable, hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t, void*), void* data) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  Executable* e = executable_of(executable);
  if (!e) return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  const int gpu = gpu_of(agent);
  if (gpu < 0) return HSA_STATUS_ERROR_INVALID_AGENT;
  if (!callback) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (Symbol& s : e->symbols)
    if (s.device == gpu)
      if (const hsa_status_t st = callback(executable, agent, {reinterpret_cast<uint64_t>(&s)}, data);
          st != HSA_STATUS_SUCCESS)
        return st;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_executable_symbol_get_info(hsa_executable_symbol_t symbol, hsa_executable_symbol_info_t attribute,
                                            void* value) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!symbol.handle) return HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL;
  if (!value) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const Symbol& s = *reinterpret_cast<const Symbol*>(symbol.handle);
  const bool kernel = s.kind == HSA_SYMBOL_KIND_KERNEL;
  switch (attribute) {
    case HSA_EXECUTABLE_SYMBOL_INFO_TYPE: put<uint32_t>(value, s.kind); break;
    case HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH: put<uint32_t>(value, static_cast<uint32_t>(s.name.size())); break;
    case HSA_EXECUTABLE_SYMBOL_INFO_NAME: std::memcpy(value, s.name.data(), s.name.size()); break;
    case HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE: put<uint32_t>(value, HSA_SYMBOL_LINKAGE_PROGRAM); break;
    case HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION: put<bool>(value, true); break;
    case HSA_EXECUTABLE_SYMBOL_INFO_AGENT: put<hsa_agent_t>(value, gpu_agent(s.device)); break;
    case HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS:
      if (kernel) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      put<uint64_t>(value, s.address);
      break;
    case HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE:
      if (kernel) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      put<uint32_t>(value, static_cast<uint32_t>(s.size));
      break;
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT:
      if (!kernel) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      put<uint64_t>(value, s.address);
      break;
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE:
      if (!kernel) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      put<uint32_t>(value, s.kernel->kernarg_size);
      break;
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT:
      if (!kernel) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      put<uint32_t>(value, std::max<uint32_t>(s.kernel->kernarg_align, 16));
      break;
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE:
      if (!kernel) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      put<uint32_t>(value, s.kernel->group_segment);
      break;
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE:
      if (!kernel) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      put<uint32_t>(value, s.kernel->private_segment);
      break;
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK:
      if (!kernel) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      put<bool>(value, false);
      break;
    default: return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return HSA_STATUS_SUCCESS;
}

}  // extern "C"
