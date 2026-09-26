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
#include <cstddef>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
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

// VGPU_TRACE_HSA=1 says what memory the program allocates, locks and
// registers, and what it loads, one line each to stderr: what a runtime built
// on this one (ROCm's HIP) does with it is otherwise invisible.
bool tracing() {
  static const bool on = [] {
    const char* t = std::getenv("VGPU_TRACE_HSA");
    return t && t[0] == '1';
  }();
  return on;
}
#define VGPU_HSA_TRACE(...)                              \
  do {                                                   \
    if (tracing()) {                                     \
      std::fprintf(stderr, "VirtualGPU HSA: " __VA_ARGS__); \
      std::fputc('\n', stderr);                          \
    }                                                    \
  } while (0)

// An attribute this does not answer, named, so a program's failed query says
// which it was.
hsa_status_t unknown(const char* call, int attribute) {
  char hex[16];
  std::snprintf(hex, sizeof hex, "0x%x", attribute);
  return fail(HSA_STATUS_ERROR_INVALID_ARGUMENT, std::string(call) + ": attribute " + hex + " is not one this answers");
}

// ---- Agents -----------------------------------------------------------------
//
// Handles: the CPU is agent 1; GPU i is 0x100 + i. A pool or region is its
// agent's handle times 16 plus its index.

constexpr uint64_t kCpuAgent = 1, kGpuAgentBase = 0x100;

// The runtime's tables are never destroyed, so a caller's exit handlers
// (ROCm's HIP destroys its executables and queues at exit) still find them.
std::mutex& g_mutex = *new std::mutex;   // the runtime's own tables below
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
// its own memory, coarse-grained; and its LDS, the group segment, which
// nothing allocates from but which says how much a work-group has.
enum class PoolKind { SystemFine, SystemCoarse, Device, Group };
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
  if (gpu_of(agent) >= 0 && index < 2) {
    *out = {agent, index == 0 ? PoolKind::Device : PoolKind::Group};
    return true;
  }
  return false;
}
std::vector<uint64_t> pools_of(hsa_agent_t a) { return {a.handle * 16, a.handle * 16 + 1}; }
constexpr uint32_t kLdsBytes = 64 * 1024;   // a CDNA work-group's

// Host memory the runtime allocated, which it maps into every device.
std::map<uintptr_t, size_t>& g_system = *new std::map<uintptr_t, size_t>;   // under g_mutex
// Device memory the runtime allocated: its device and size.
std::map<uint64_t, std::pair<int, size_t>>& g_device = *new std::map<uint64_t, std::pair<int, size_t>>;   // under g_mutex
// Host memory a program locked (hsa_amd_memory_lock), and its size.
std::map<uintptr_t, size_t>& g_locked = *new std::map<uintptr_t, size_t>;   // under g_mutex
// What a program attached to an allocation (hsa_amd_pointer_info_set_userdata).
std::map<uintptr_t, void*>& g_userdata = *new std::map<uintptr_t, void*>;   // under g_mutex

uint64_t now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// ---- Signals -----------------------------------------------------------------

// Asynchronous handlers (hsa_amd_signal_async_handler) are checked by a
// thread of their own whenever any signal changes.
std::mutex& g_handlers_mu = *new std::mutex;
std::condition_variable& g_handlers_cv = *new std::condition_variable;
uint64_t g_signal_changes = 0;   // under g_handlers_mu
void signal_changed() {
  {
    std::lock_guard<std::mutex> lock(g_handlers_mu);
    ++g_signal_changes;
  }
  g_handlers_cv.notify_all();
}

// A signal as ROCm lays one out (amd_hsa_signal.h's amd_signal_t), since a
// kernel reaches it through its handle: the device library adds to `value`
// to ring a doorbell (hostcall does), and would raise the event in the
// mailbox where one is set. Every device maps it at its own address.
struct AmdSignal {
  int64_t kind = 1;   // AMD_SIGNAL_KIND_USER
  std::atomic<int64_t> value{0};
  uint64_t event_mailbox_ptr = 0;
  uint32_t event_id = 0, reserved1 = 0;
  // When the dispatch or copy that completes it ran, in the system's
  // timestamps (nanoseconds): what hsa_amd_profiling reads back.
  std::atomic<uint64_t> start{0}, end{0};
  uint64_t queue_ptr = 0;
  uint32_t reserved3[2] = {0, 0};
};
static_assert(sizeof(AmdSignal) == 64 && offsetof(AmdSignal, value) == 8 && offsetof(AmdSignal, start) == 32,
              "amd_signal_t's layout");

struct Signal {
  alignas(64) AmdSignal amd;   // first: the handle is its address
  std::mutex mu;
  std::condition_variable cv;
  // Changes the value and wakes whoever waits on it -- under the lock, since
  // a waiter that sees the value may destroy the signal as soon as it has the
  // lock back, and must not do that while this is still waking it.
  template <typename F>
  int64_t update(F f) {
    int64_t old;
    {
      std::lock_guard<std::mutex> lock(mu);
      old = amd.value.load();
      amd.value.store(f(old));
      cv.notify_all();
    }
    signal_changed();
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
  // A kernel changes a signal without waking anyone (it adds to the value in
  // memory, as a doorbell), so the wait looks again every little while too.
  const bool forever = timeout == UINT64_MAX || timeout > (uint64_t{1} << 62);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(forever ? 0 : timeout);
  std::unique_lock<std::mutex> lock(s->mu);
  for (auto nap = std::chrono::microseconds(20);; nap = std::min(nap * 2, std::chrono::microseconds(1000))) {
    const int64_t v = s->amd.value.load();
    if (satisfied(v, c, compare)) return v;
    const auto now = std::chrono::steady_clock::now();
    if (!forever && now >= deadline) return v;
    s->cv.wait_for(lock, forever ? nap : std::min<std::chrono::nanoseconds>(nap, deadline - now));
  }
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
struct Executable;
// A code object an executable loaded for an agent, and where it came from.
struct LoadedCode {
  Executable* executable;
  int device;
  const shared::Loaded* loaded;
  std::string storage;   // the bytes it was read from
};
struct Executable {
  bool frozen = false;
  std::vector<const shared::Loaded*> loaded;
  std::deque<LoadedCode> code;   // an entry's address is its hsa_loaded_code_object_t
  std::deque<Symbol> symbols;    // a symbol's address is its handle
};
std::set<Executable*>& g_executables = *new std::set<Executable*>;   // under g_mutex

// A kernel object -- the address of a kernel's descriptor on its device --
// and what it names.
struct KernelRef {
  int device;
  const shared::Loaded* loaded;
  const vgpu::amd::Kernel* kernel;
};
std::map<uint64_t, KernelRef>& g_kernels = *new std::map<uint64_t, KernelRef>;   // under g_mutex

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

// Runs one kernel dispatch packet. The grid is given in work-items, and need
// not be a whole number of work-groups: the last one in a dimension then has
// what is left over.
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
  if (dims < 1 || dims > 3) {
    *why = "a dispatch of " + std::to_string(dims) + " dimensions";
    return HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
  }
  for (int i = 0; i < 3; ++i)
    if (!wg[i] || !grid[i]) {
      *why = "a dispatch with a zero grid or work-group size";
      return HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
    }
  // What the packet asks for beyond the kernel's own LDS is the launch's.
  const uint32_t lds = p.group_segment_size > ref.kernel->group_segment ? p.group_segment_size - ref.kernel->group_segment : 0;
  if (!shared::run(q->device, ref.loaded, *ref.kernel, grid, wg, lds, reinterpret_cast<uint64_t>(p.kernarg_address),
                   q->q.type == HSA_QUEUE_TYPE_COOPERATIVE, why))
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
      const uint64_t start = now_ns();
      status = dispatch(q, p, &why);
      if (completion.handle) {
        signal_of(completion)->amd.start = start;
        signal_of(completion)->amd.end = now_ns();
      }
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
          for (Signal* d : deps) any = any || d->amd.value.load() == 0;
          if (!any) wait(deps.front(), HSA_SIGNAL_CONDITION_EQ, 0, 1000000);
        }
      }
    } else if (type == HSA_PACKET_TYPE_VENDOR_SPECIFIC && packet[2] == 2 /* HSA_AMD_PACKET_TYPE_BARRIER_VALUE */) {
      // AMD's barrier-value packet: holds the queue until (signal & mask)
      // meets the condition against the value.
      int64_t value, mask;
      uint32_t cond;
      hsa_signal_t on;
      std::memcpy(&on, packet + 8, 8);
      std::memcpy(&value, packet + 16, 8);
      std::memcpy(&mask, packet + 24, 8);
      std::memcpy(&cond, packet + 32, 4);
      std::memcpy(&completion, packet + 56, 8);
      if (on.handle) {
        Signal* sig = signal_of(on);
        std::unique_lock<std::mutex> lock(sig->mu);
        while (!satisfied(sig->amd.value.load() & mask, static_cast<hsa_signal_condition_t>(cond), value) && !q->stop)
          sig->cv.wait_for(lock, std::chrono::milliseconds(1));
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
  static vgpu::amd::WorkQueue<int>& q = *new vgpu::amd::WorkQueue<int>(0, 1);
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
    case HSA_AMD_SYSTEM_INFO_BUILD_VERSION: put<const char*>(value, "VirtualGPU"); break;
    case HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED:
    case HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT:
    case HSA_AMD_SYSTEM_INFO_MWAITX_ENABLED:
    case HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED:
    case HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED:
    case HSA_AMD_SYSTEM_INFO_XNACK_ENABLED: put<bool>(value, false); break;
    // The version of AMD's extensions this follows: ROCm 7.0's hsa_ext_amd.h.
    case HSA_AMD_SYSTEM_INFO_EXT_VERSION_MAJOR: put<uint16_t>(value, 1); break;
    case HSA_AMD_SYSTEM_INFO_EXT_VERSION_MINOR: put<uint16_t>(value, 11); break;
    default: return unknown("hsa_system_get_info", attribute);
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
    case HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES: put<bool>(value, is_gpu); break;
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
    // The rest of what AMD's runtime reports, as an MI300X-class part has it
    // where the profile does not say: eight compute dies (XCCs) of four
    // shader engines, HBM3 8192 bits wide at 1300 MHz.
    case HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS: put<uint32_t>(value, is_gpu ? 4 : 0); break;
    case HSA_AMD_AGENT_INFO_MEMORY_WIDTH: put<uint32_t>(value, is_gpu ? 8192 : 0); break;
    case HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY: put<uint32_t>(value, is_gpu ? 1300 : 0); break;
    case HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES: put<uint32_t>(value, is_gpu ? 32 : 0); break;
    case HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE: put<uint32_t>(value, is_gpu ? 1 : 0); break;
    case HSA_AMD_AGENT_INFO_HDP_FLUSH: std::memset(value, 0, 2 * sizeof(void*)); break;
    case HSA_AMD_AGENT_INFO_ASIC_REVISION: put<uint32_t>(value, is_gpu ? 1 : 0); break;
    case HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT:
      put<uint32_t>(value, is_gpu ? static_cast<uint32_t>(shared::profile(gpu).limits.multiprocessors) : 0);
      break;
    case HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID:
    case HSA_AMD_AGENT_INFO_UCODE_VERSION:
    case HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION:
    case HSA_AMD_AGENT_INFO_DRIVER_UID: put<uint32_t>(value, 0); break;
    case HSA_AMD_AGENT_INFO_NUM_SDMA_ENG: put<uint32_t>(value, is_gpu ? 2 : 0); break;
    case HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG: put<uint32_t>(value, 0); break;
    case HSA_AMD_AGENT_INFO_IOMMU_SUPPORT: put<uint32_t>(value, 0); break;   // HSA_IOMMU_SUPPORT_NONE
    case HSA_AMD_AGENT_INFO_NUM_XCC: put<uint32_t>(value, is_gpu ? 8 : 0); break;
    case HSA_AMD_AGENT_INFO_NEAREST_CPU: put<hsa_agent_t>(value, {kCpuAgent}); break;
    case HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES:
    case HSA_AMD_AGENT_INFO_AQL_EXTENSIONS: std::memset(value, 0, 8); break;
    case HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_MAX:
    case HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT: put<uint64_t>(value, is_gpu ? uint64_t{1} << 32 : 0); break;
    case HSA_AMD_AGENT_INFO_CLOCK_COUNTERS: {
      const uint64_t t = now_ns();
      put(value, hsa_amd_clock_counters_t{t, t, t, 1000000000});
      break;
    }
    default: return unknown("hsa_agent_get_info", attribute);
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
    default: return unknown("hsa_isa_get_info_alt", attribute);
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
  s->amd.value = initial;
  shared::map_host(&s->amd, sizeof s->amd);   // a kernel reaches it through its handle
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
  shared::unmap_host(&signal_of(signal)->amd);
  delete signal_of(signal);
  return HSA_STATUS_SUCCESS;
}

// Every memory order the specification names does the same here: each
// operation is sequentially consistent, and wakes every waiter.
#define VGPU_SIGNAL_LOAD(order) \
  hsa_signal_value_t hsa_signal_load_##order(hsa_signal_t s) { return signal_of(s)->amd.value.load(); }
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
  // A cooperative queue's dispatches have every work-group resident at once,
  // so they may wait on one another (a grid barrier).
  if (!queue || size < 64 || size > 131072 || (size & (size - 1)) || type > HSA_QUEUE_TYPE_COOPERATIVE)
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
  const bool device = id.kind == PoolKind::Device, group = id.kind == PoolKind::Group;
  const uint64_t size = group    ? kLdsBytes
                        : device ? shared::profile(gpu_of(id.agent)).vram_bytes
                                 : static_cast<uint64_t>(sysconf(_SC_PHYS_PAGES)) * sysconf(_SC_PAGESIZE);
  switch (attribute) {
    case HSA_AMD_MEMORY_POOL_INFO_SEGMENT:
      put<uint32_t>(value, group ? HSA_AMD_SEGMENT_GROUP : HSA_AMD_SEGMENT_GLOBAL);
      break;
    case HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS:
      put<uint32_t>(value, group ? 0
                           : id.kind == PoolKind::SystemFine
                               ? HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED | HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT
                               : HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED);
      break;
    case HSA_AMD_MEMORY_POOL_INFO_SIZE: put<size_t>(value, size); break;
    case HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED: put<bool>(value, !group); break;
    case HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE:
    case HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT:
    case HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE: put<size_t>(value, group ? 0 : 4096); break;
    case HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL: put<bool>(value, !device && !group); break;
    case HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE: put<size_t>(value, group ? 0 : size); break;
    case HSA_AMD_MEMORY_POOL_INFO_LOCATION:
      put<uint32_t>(value, device || group ? HSA_AMD_MEMORY_POOL_LOCATION_GPU : HSA_AMD_MEMORY_POOL_LOCATION_CPU);
      break;
    default: return unknown("hsa_amd_memory_pool_get_info", attribute);
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
  // Everyone reaches system memory. A GPU's own memory and LDS are its own;
  // another GPU reaches its memory once hsa_amd_agents_allow_access says so,
  // and the host never does directly -- a device address here is not a host
  // address -- so a runtime above this copies rather than writes through it.
  const bool own = (id.kind != PoolKind::Device && id.kind != PoolKind::Group) || id.agent.handle == agent.handle;
  const bool never = id.kind == PoolKind::Group || agent.handle == kCpuAgent;
  switch (attribute) {
    case HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS:
      put<uint32_t>(value, own     ? HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT
                           : never ? HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED
                                   : HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT);
      break;
    case HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS: put<uint32_t>(value, own ? 0 : 1); break;
    default: return unknown("hsa_amd_agent_memory_pool_get_info", attribute);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t pool, size_t size, uint32_t, void** ptr) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  PoolId id;
  if (!pool_of(pool.handle, &id)) return HSA_STATUS_ERROR_INVALID_MEMORY_POOL;
  if (!ptr || !size) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (id.kind == PoolKind::Group) return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  if (id.kind == PoolKind::Device) {
    const int gpu = gpu_of(id.agent);
    try {
      const uint64_t va = shared::memory(gpu).alloc(size);
      std::lock_guard<std::mutex> lock(g_mutex);
      g_device[va] = {gpu, size};
      *ptr = reinterpret_cast<void*>(va);
      VGPU_HSA_TRACE("allocated %zu bytes of device %d's memory at %p", size, gpu, *ptr);
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
  VGPU_HSA_TRACE("allocated %zu bytes of system memory at %p", rounded, p);
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
    const int gpu = it->second.first;
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
  // Another GPU given a device's memory reaches it from its kernels, as a
  // peer. The host reaches every device's memory by copying, which needs no
  // leave, and a device reaches system memory already.
  const int owner = shared::owner(reinterpret_cast<uint64_t>(ptr));
  if (owner >= 0)
    for (uint32_t i = 0; i < num_agents; ++i)
      if (const int gpu = gpu_of(agents[i]); gpu >= 0 && gpu != owner) shared::allow_peer(gpu, owner);
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
    const uint64_t start = now_ns();
    const bool ok = !size || shared::copy(dst, src, size, &why);
    if (completion_signal.handle) {
      signal_of(completion_signal)->amd.start = start;
      signal_of(completion_signal)->amd.end = now_ns();
    }
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
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_locked[reinterpret_cast<uintptr_t>(host_ptr)] = size;
  }
  VGPU_HSA_TRACE("locked %zu bytes of host memory at %p", size, host_ptr);
  *agent_ptr = host_ptr;   // devices reach it where it is
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_amd_memory_unlock(void* host_ptr) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_locked.erase(reinterpret_cast<uintptr_t>(host_ptr));
  }
  shared::unmap_host(host_ptr);
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_amd_memory_lock_to_pool(void* host_ptr, size_t size, hsa_agent_t* agents, int num_agent,
                                         hsa_amd_memory_pool_t, uint32_t, void** agent_ptr) {
  return hsa_amd_memory_lock(host_ptr, size, agents, num_agent, agent_ptr);
}
hsa_status_t hsa_memory_register(void* ptr, size_t size) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!ptr || !size) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  shared::map_host(ptr, size);
  VGPU_HSA_TRACE("registered %zu bytes of host memory at %p", size, ptr);
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
    case HSA_REGION_INFO_SEGMENT:
      put<uint32_t>(value, id.kind == PoolKind::Group ? HSA_REGION_SEGMENT_GROUP : HSA_REGION_SEGMENT_GLOBAL);
      return HSA_STATUS_SUCCESS;
    case HSA_REGION_INFO_GLOBAL_FLAGS:
      put<uint32_t>(value, id.kind == PoolKind::Group        ? 0
                           : id.kind == PoolKind::SystemFine
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
  e->code.push_back({e, gpu, m, std::string(code)});
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
  if (loaded_code_object) loaded_code_object->handle = reinterpret_cast<uint64_t>(&e->code.back());
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
    case 3: put<uint32_t>(value, 0); break;   // MODULE_NAME_LENGTH: a program symbol has none
    case 4: break;                            // MODULE_NAME
    case 6:                                   // VARIABLE_ALLOCATION: the agent's
    case 7:                                   // VARIABLE_SEGMENT: global
      if (kernel) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      put<uint32_t>(value, 0);
      break;
    case 8:   // VARIABLE_ALIGNMENT
      if (kernel) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      put<uint32_t>(value, 16);
      break;
    case 10:   // VARIABLE_IS_CONST
      if (kernel) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      put<bool>(value, false);
      break;
    case 18:   // KERNEL_CALL_CONVENTION
      if (!kernel) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      put<uint32_t>(value, 0);
      break;
    default: return unknown("hsa_executable_symbol_get_info", attribute);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_executable_iterate_symbols(hsa_executable_t executable,
                                            hsa_status_t (*callback)(hsa_executable_t, hsa_executable_symbol_t, void*),
                                            void* data) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  Executable* e = executable_of(executable);
  if (!e) return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  if (!callback) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (Symbol& s : e->symbols)
    if (const hsa_status_t st = callback(executable, {reinterpret_cast<uint64_t>(&s)}, data); st != HSA_STATUS_SUCCESS)
      return st;
  return HSA_STATUS_SUCCESS;
}

// ---- AMD's loader extension (hsa_ven_amd_loader.h) ----------------------------------

namespace {
const LoadedCode* code_at(uint64_t device_address) {
  for (Executable* e : g_executables)
    for (const LoadedCode& c : e->code) {
      const uint64_t base = shared::code_base(c.loaded), size = shared::host_image(c.loaded).size();
      if (device_address >= base && device_address - base < size) return &c;
    }
  return nullptr;
}
}  // namespace

hsa_status_t hsa_ven_amd_loader_query_host_address(const void* device_address, const void** host_address) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!device_address || !host_address) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> lock(g_mutex);
  const uint64_t at = reinterpret_cast<uint64_t>(device_address);
  const LoadedCode* c = code_at(at);
  if (!c) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *host_address = shared::host_image(c->loaded).data() + (at - shared::code_base(c->loaded));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_ven_amd_loader_query_segment_descriptors(void* segment_descriptors, size_t* num_segment_descriptors) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!num_segment_descriptors) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  (void)segment_descriptors;
  *num_segment_descriptors = 0;   // what a debugger asks for, which this does not describe
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_ven_amd_loader_query_executable(const void* device_address, hsa_executable_t* executable) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!device_address || !executable) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> lock(g_mutex);
  const LoadedCode* c = code_at(reinterpret_cast<uint64_t>(device_address));
  if (!c) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  executable->handle = reinterpret_cast<uint64_t>(c->executable);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
    hsa_executable_t executable, hsa_status_t (*callback)(hsa_executable_t, hsa_loaded_code_object_t, void*),
    void* data) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  Executable* e = executable_of(executable);
  if (!e) return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  if (!callback) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (LoadedCode& c : e->code)
    if (const hsa_status_t st = callback(executable, {reinterpret_cast<uint64_t>(&c)}, data); st != HSA_STATUS_SUCCESS)
      return st;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_ven_amd_loader_loaded_code_object_get_info(hsa_loaded_code_object_t loaded_code_object,
                                                            int attribute, void* value) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!loaded_code_object.handle || !value) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const LoadedCode& c = *reinterpret_cast<const LoadedCode*>(loaded_code_object.handle);
  const uint64_t base = shared::code_base(c.loaded);
  // Where the bytes came from, as ROCm's loader names memory: the process,
  // then where and how much.
  char uri[128];
  std::snprintf(uri, sizeof uri, "memory://%d#offset=0x%llx&size=%zu", static_cast<int>(getpid()),
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(c.storage.data())), c.storage.size());
  switch (attribute) {
    case 1: put<hsa_executable_t>(value, {reinterpret_cast<uint64_t>(c.executable)}); break;   // EXECUTABLE
    case 2: put<uint32_t>(value, 2); break;                                                     // KIND: AGENT
    case 3: put<hsa_agent_t>(value, gpu_agent(c.device)); break;                                // AGENT
    case 4: put<uint32_t>(value, 2); break;                                                     // STORAGE: MEMORY
    case 5: put<uint64_t>(value, reinterpret_cast<uintptr_t>(c.storage.data())); break;         // MEMORY_BASE
    case 6: put<uint64_t>(value, c.storage.size()); break;                                      // MEMORY_SIZE
    case 7: put<int>(value, -1); break;                                                         // FILE: none
    case 8: put<int64_t>(value, static_cast<int64_t>(base)); break;                             // LOAD_DELTA
    case 9: put<uint64_t>(value, base); break;                                                  // LOAD_BASE
    case 10: put<uint64_t>(value, shared::host_image(c.loaded).size()); break;                  // LOAD_SIZE
    case 11: put<uint32_t>(value, static_cast<uint32_t>(std::strlen(uri))); break;              // URI_LENGTH
    case 12: std::memcpy(value, uri, std::strlen(uri)); break;                                  // URI
    default: return unknown("hsa_ven_amd_loader_loaded_code_object_get_info", attribute);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size(
    hsa_file_t file, size_t offset, size_t size, hsa_code_object_reader_t* reader) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!reader || !size) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::string bytes(size, '\0');
  size_t got = 0;
  while (got < size) {
    const ssize_t n = pread(file, bytes.data() + got, size - got, static_cast<off_t>(offset + got));
    if (n <= 0) return HSA_STATUS_ERROR_INVALID_FILE;
    got += static_cast<size_t>(n);
  }
  reader->handle = reinterpret_cast<uint64_t>(new Reader{std::move(bytes)});
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_ven_amd_loader_iterate_executables(hsa_status_t (*callback)(hsa_executable_t, void*), void* data) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!callback) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::vector<Executable*> all;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    all.assign(g_executables.begin(), g_executables.end());
  }
  for (Executable* e : all)
    if (const hsa_status_t st = callback({reinterpret_cast<uint64_t>(e)}, data); st != HSA_STATUS_SUCCESS) return st;
  return HSA_STATUS_SUCCESS;
}

// Which extensions there are: AMD's loader, version 1.3.
namespace {
constexpr uint16_t kExtensionAmdLoader = 0x201;
}
hsa_status_t hsa_system_extension_supported(uint16_t extension, uint16_t major, uint16_t minor, bool* result) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!result) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *result = extension == kExtensionAmdLoader && major == 1 && minor <= 3;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_system_major_extension_supported(uint16_t extension, uint16_t major, uint16_t* minor, bool* result) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!result || !minor) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *result = extension == kExtensionAmdLoader && major == 1;
  if (*result) *minor = 3;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_agent_extension_supported(uint16_t extension, hsa_agent_t agent, uint16_t major, uint16_t minor,
                                           bool* result) {
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  return hsa_system_extension_supported(extension, major, minor, result);
}
hsa_status_t hsa_agent_major_extension_supported(uint16_t extension, hsa_agent_t agent, uint16_t major,
                                                 uint16_t* minor, bool* result) {
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  return hsa_system_major_extension_supported(extension, major, minor, result);
}
hsa_status_t hsa_system_get_major_extension_table(uint16_t extension, uint16_t major, size_t table_length,
                                                  void* table) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!table) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (extension != kExtensionAmdLoader || major != 1)
    return fail(HSA_STATUS_ERROR_NOT_SUPPORTED, "extension " + std::to_string(extension) + " is not one this has");
  // hsa_ven_amd_loader_1_03_pfn_t, in its order; a caller asking for an
  // older version's table takes the front of it.
  void* const functions[] = {
      reinterpret_cast<void*>(hsa_ven_amd_loader_query_host_address),
      reinterpret_cast<void*>(hsa_ven_amd_loader_query_segment_descriptors),
      reinterpret_cast<void*>(hsa_ven_amd_loader_query_executable),
      reinterpret_cast<void*>(hsa_ven_amd_loader_executable_iterate_loaded_code_objects),
      reinterpret_cast<void*>(hsa_ven_amd_loader_loaded_code_object_get_info),
      reinterpret_cast<void*>(hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size),
      reinterpret_cast<void*>(hsa_ven_amd_loader_iterate_executables),
  };
  std::memcpy(table, functions, std::min(table_length, sizeof functions));
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_system_get_extension_table(uint16_t extension, uint16_t major, uint16_t, void* table) {
  return hsa_system_get_major_extension_table(extension, major, sizeof(void*) * 7, table);
}

// ---- What an allocation is ---------------------------------------------------------

namespace {
using PointerInfo = hsa_amd_pointer_info_t;
extern "C++" template <typename Map>
typename Map::const_iterator containing(const Map& m, uintptr_t at) {
  auto it = m.upper_bound(at);
  if (it == m.begin()) return m.end();
  --it;
  const size_t size = [&] {
    if constexpr (std::is_same_v<typename Map::mapped_type, size_t>) return it->second;
    else return it->second.second;
  }();
  return at - it->first < std::max<size_t>(size, 1) ? it : m.end();
}
}  // namespace

hsa_status_t hsa_amd_pointer_info(const void* ptr, hsa_amd_pointer_info_t* info_out, void* (*alloc)(size_t),
                                  uint32_t* num_agents_accessible, hsa_agent_t** accessible) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!info_out) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  PointerInfo info{};
  std::memcpy(&info.size, info_out, 4);
  const uint32_t room = std::min<uint32_t>(info.size, sizeof(PointerInfo));
  info.size = sizeof(PointerInfo);
  std::vector<hsa_agent_t> agents;
  const uintptr_t at = reinterpret_cast<uintptr_t>(ptr);
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (auto it = containing(g_system, at); it != g_system.end()) {
      info.type = HSA_EXT_POINTER_TYPE_HSA;
      info.agentBaseAddress = info.hostBaseAddress = reinterpret_cast<void*>(it->first);
      info.sizeInBytes = it->second;
      info.agentOwner = {kCpuAgent};
      info.global_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED;
    } else if (auto it = containing(g_device, at); it != g_device.end()) {
      info.type = HSA_EXT_POINTER_TYPE_HSA;
      info.agentBaseAddress = reinterpret_cast<void*>(it->first);
      info.sizeInBytes = it->second.second;
      info.agentOwner = gpu_agent(it->second.first);
      info.global_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
    } else if (auto it = containing(g_locked, at); it != g_locked.end()) {
      info.type = HSA_EXT_POINTER_TYPE_LOCKED;
      info.agentBaseAddress = info.hostBaseAddress = reinterpret_cast<void*>(it->first);
      info.sizeInBytes = it->second;
      info.agentOwner = {kCpuAgent};
      info.global_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED;
    }
    if (info.type) {
      if (auto u = g_userdata.find(reinterpret_cast<uintptr_t>(info.agentBaseAddress)); u != g_userdata.end())
        info.userData = u->second;
      // System memory every agent reaches; a device's memory, its device.
      if (info.agentOwner.handle == kCpuAgent) {
        agents.push_back({kCpuAgent});
        for (int i = 0; i < shared::device_count(); ++i) agents.push_back(gpu_agent(i));
      } else {
        agents.push_back(info.agentOwner);
      }
    }
  }
  std::memcpy(info_out, &info, room);
  if (num_agents_accessible) *num_agents_accessible = static_cast<uint32_t>(agents.size());
  if (accessible) {
    if (!alloc) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    *accessible = static_cast<hsa_agent_t*>(alloc(std::max<size_t>(agents.size(), 1) * sizeof(hsa_agent_t)));
    if (!*accessible) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    std::copy(agents.begin(), agents.end(), *accessible);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_pointer_info_set_userdata(const void* ptr, void* userdata) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  std::lock_guard<std::mutex> lock(g_mutex);
  const uintptr_t at = reinterpret_cast<uintptr_t>(ptr);
  if (!g_system.count(at) && !g_device.count(at) && !g_locked.count(at)) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  g_userdata[at] = userdata;
  return HSA_STATUS_SUCCESS;
}

// ---- Asynchronous signal handlers ----------------------------------------------------

namespace {
struct Handler {
  Signal* signal;
  hsa_signal_condition_t condition;
  hsa_signal_value_t value;
  bool (*handler)(hsa_signal_value_t, void*);
  void* arg;
};
std::vector<Handler>& g_handlers = *new std::vector<Handler>;   // under g_handlers_mu

// Calls each handler whose condition holds, once, and keeps it only where it
// asks to be called again.
void run_handlers() {
  std::unique_lock<std::mutex> lock(g_handlers_mu);
  for (uint64_t seen = ~uint64_t{0};;) {
    std::vector<std::pair<Handler, hsa_signal_value_t>> ready;
    for (size_t i = 0; i < g_handlers.size();) {
      const hsa_signal_value_t v = g_handlers[i].signal->amd.value.load();
      if (satisfied(v, g_handlers[i].condition, g_handlers[i].value)) {
        ready.emplace_back(g_handlers[i], v);
        g_handlers.erase(g_handlers.begin() + static_cast<long>(i));
      } else {
        ++i;
      }
    }
    if (!ready.empty()) {
      lock.unlock();
      std::vector<Handler> again;
      for (auto& [h, v] : ready)
        if (h.handler(v, h.arg)) again.push_back(h);
      lock.lock();
      g_handlers.insert(g_handlers.end(), again.begin(), again.end());
      continue;
    }
    seen = g_signal_changes;
    g_handlers_cv.wait_for(lock, std::chrono::milliseconds(2), [&] { return g_signal_changes != seen; });
  }
}
}  // namespace

hsa_status_t hsa_amd_signal_async_handler(hsa_signal_t signal, hsa_signal_condition_t condition,
                                          hsa_signal_value_t value, bool (*handler)(hsa_signal_value_t, void*),
                                          void* arg) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!signal.handle) return HSA_STATUS_ERROR_INVALID_SIGNAL;
  if (!handler) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  static std::once_flag started_thread;
  std::call_once(started_thread, [] { std::thread(run_handlers).detach(); });
  {
    std::lock_guard<std::mutex> lock(g_handlers_mu);
    g_handlers.push_back({signal_of(signal), condition, value, handler, arg});
    ++g_signal_changes;
  }
  g_handlers_cv.notify_all();
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_signal_value_pointer(hsa_signal_t signal, volatile hsa_signal_value_t** value_ptr) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!signal.handle || !value_ptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *value_ptr = reinterpret_cast<volatile hsa_signal_value_t*>(&signal_of(signal)->amd.value);
  return HSA_STATUS_SUCCESS;
}

// ---- Profiling ----------------------------------------------------------------------

hsa_status_t hsa_amd_profiling_set_profiler_enabled(hsa_queue_t* queue, int) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  return queue ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_INVALID_QUEUE;
}
hsa_status_t hsa_amd_profiling_async_copy_enable(bool) {
  return started() ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_NOT_INITIALIZED;
}
hsa_status_t hsa_amd_profiling_get_dispatch_time(hsa_agent_t agent, hsa_signal_t signal,
                                                 hsa_amd_profiling_dispatch_time_t* time) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (gpu_of(agent) < 0) return HSA_STATUS_ERROR_INVALID_AGENT;
  if (!signal.handle || !time) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const uint64_t t[2] = {signal_of(signal)->amd.start.load(), signal_of(signal)->amd.end.load()};
  std::memcpy(time, t, sizeof t);
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_amd_profiling_get_async_copy_time(hsa_signal_t signal, void* time) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!signal.handle || !time) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const uint64_t t[2] = {signal_of(signal)->amd.start.load(), signal_of(signal)->amd.end.load()};
  std::memcpy(time, t, sizeof t);
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_amd_profiling_convert_tick_to_system_domain(hsa_agent_t agent, uint64_t agent_tick,
                                                             uint64_t* system_tick) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  if (!system_tick) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *system_tick = agent_tick;   // one clock for all
  return HSA_STATUS_SUCCESS;
}

// ---- Queues and copies, as AMD's runtime extends them ----------------------------------

hsa_status_t hsa_amd_queue_cu_set_mask(const hsa_queue_t* queue, uint32_t, const uint32_t*) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  return queue ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_INVALID_QUEUE;   // every CU runs every queue here
}
hsa_status_t hsa_amd_queue_cu_get_mask(const hsa_queue_t* queue, uint32_t count, uint32_t* mask) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!queue) return HSA_STATUS_ERROR_INVALID_QUEUE;
  if (!mask) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (uint32_t i = 0; i < (count + 31) / 32; ++i) mask[i] = ~0u;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_amd_queue_set_priority(hsa_queue_t* queue, int) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  return queue ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_INVALID_QUEUE;
}

hsa_status_t hsa_amd_memory_async_copy_on_engine(void* dst, hsa_agent_t dst_agent, const void* src,
                                                 hsa_agent_t src_agent, size_t size, uint32_t num_dep_signals,
                                                 const hsa_signal_t* dep_signals, hsa_signal_t completion_signal,
                                                 int, bool) {
  return hsa_amd_memory_async_copy(dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals,
                                   completion_signal);
}
hsa_status_t hsa_amd_memory_copy_engine_status(hsa_agent_t, hsa_agent_t, uint32_t* mask) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!mask) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *mask = 1;   // one engine, always free
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_amd_memory_get_preferred_copy_engine(hsa_agent_t a, hsa_agent_t b, uint32_t* mask) {
  return hsa_amd_memory_copy_engine_status(a, b, mask);
}

// A box of `range` (bytes wide, rows high, slices deep) between two pitched
// allocations, row by row, once the dependencies have reached zero.
hsa_status_t hsa_amd_memory_async_copy_rect(const void* dst_ptr, const hsa_dim3_t* dst_offset, const void* src_ptr,
                                            const hsa_dim3_t* src_offset, const hsa_dim3_t* range, hsa_agent_t,
                                            int, uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
                                            hsa_signal_t completion_signal) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  struct Pitched {
    char* base;
    size_t pitch, slice;
  };
  if (!dst_ptr || !src_ptr || !dst_offset || !src_offset || !range || (num_dep_signals && !dep_signals))
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const Pitched dst = *static_cast<const Pitched*>(dst_ptr), src = *static_cast<const Pitched*>(src_ptr);
  const hsa_dim3_t d = *dst_offset, s = *src_offset, r = *range;
  std::vector<Signal*> deps;
  for (uint32_t i = 0; i < num_dep_signals; ++i) deps.push_back(signal_of(dep_signals[i]));
  copy_engine().submit([=] {
    for (Signal* dep : deps) wait(dep, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX);
    const uint64_t start = now_ns();
    bool ok = true;
    std::string why;
    for (uint32_t z = 0; z < r.z && ok; ++z)
      for (uint32_t y = 0; y < r.y && ok; ++y)
        ok = shared::copy(dst.base + (d.z + z) * dst.slice + (d.y + y) * dst.pitch + d.x,
                          src.base + (s.z + z) * src.slice + (s.y + y) * src.pitch + s.x, r.x, &why);
    if (!ok) fail(HSA_STATUS_ERROR_INVALID_ARGUMENT, "a rectangular copy failed: " + why);
    if (completion_signal.handle) {
      signal_of(completion_signal)->amd.start = start;
      signal_of(completion_signal)->amd.end = now_ns();
    }
    complete(completion_signal);
    return ok ? 0 : 1;
  });
  return HSA_STATUS_SUCCESS;
}

// What AMD's runtime lets a program tune, which has nothing to tune here.
hsa_status_t hsa_amd_agent_set_async_scratch_limit(hsa_agent_t agent, size_t) {
  return valid_agent(agent) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_INVALID_AGENT;
}
hsa_status_t hsa_amd_coherency_set_type(hsa_agent_t agent, int) {
  return valid_agent(agent) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_INVALID_AGENT;
}
hsa_status_t hsa_amd_coherency_get_type(hsa_agent_t agent, int* type) {
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  if (!type) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *type = 0;   // HSA_AMD_COHERENCY_TYPE_COHERENT
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_amd_enable_logging(uint8_t*, void*) { return HSA_STATUS_SUCCESS; }
hsa_status_t hsa_amd_register_system_event_handler(void*, void*) { return HSA_STATUS_SUCCESS; }

// What this does not model yet, refused by name: images and samplers (the
// texture path), virtual memory, sharing memory between processes, SVM,
// graphics interop and DMA-buf.
#define VGPU_HSA_REFUSED(name, what) \
  hsa_status_t name() { return fail(HSA_STATUS_ERROR_NOT_SUPPORTED, #name " is not supported: " what); }
VGPU_HSA_REFUSED(hsa_amd_image_create, "images are not modelled yet")
VGPU_HSA_REFUSED(hsa_ext_image_create, "images are not modelled yet")
VGPU_HSA_REFUSED(hsa_ext_image_create_with_layout, "images are not modelled yet")
VGPU_HSA_REFUSED(hsa_ext_image_data_get_info, "images are not modelled yet")
VGPU_HSA_REFUSED(hsa_ext_image_destroy, "images are not modelled yet")
VGPU_HSA_REFUSED(hsa_ext_image_export, "images are not modelled yet")
VGPU_HSA_REFUSED(hsa_ext_image_import, "images are not modelled yet")
VGPU_HSA_REFUSED(hsa_ext_sampler_create_v2, "samplers are not modelled yet")
VGPU_HSA_REFUSED(hsa_ext_sampler_destroy, "samplers are not modelled yet")
VGPU_HSA_REFUSED(hsa_amd_vmem_address_reserve, "virtual memory is not modelled yet")
VGPU_HSA_REFUSED(hsa_amd_vmem_address_free, "virtual memory is not modelled yet")
VGPU_HSA_REFUSED(hsa_amd_vmem_handle_create, "virtual memory is not modelled yet")
VGPU_HSA_REFUSED(hsa_amd_vmem_handle_release, "virtual memory is not modelled yet")
VGPU_HSA_REFUSED(hsa_amd_vmem_map, "virtual memory is not modelled yet")
VGPU_HSA_REFUSED(hsa_amd_vmem_unmap, "virtual memory is not modelled yet")
VGPU_HSA_REFUSED(hsa_amd_vmem_set_access, "virtual memory is not modelled yet")
VGPU_HSA_REFUSED(hsa_amd_vmem_get_access, "virtual memory is not modelled yet")
VGPU_HSA_REFUSED(hsa_amd_vmem_export_shareable_handle, "virtual memory is not modelled yet")
VGPU_HSA_REFUSED(hsa_amd_vmem_import_shareable_handle, "virtual memory is not modelled yet")
VGPU_HSA_REFUSED(hsa_amd_vmem_retain_alloc_handle, "virtual memory is not modelled yet")
VGPU_HSA_REFUSED(hsa_amd_ipc_memory_create, "memory is not shared between processes yet")
VGPU_HSA_REFUSED(hsa_amd_ipc_memory_attach, "memory is not shared between processes yet")
VGPU_HSA_REFUSED(hsa_amd_ipc_memory_detach, "memory is not shared between processes yet")
VGPU_HSA_REFUSED(hsa_amd_svm_attributes_get, "SVM is not modelled")
VGPU_HSA_REFUSED(hsa_amd_svm_attributes_set, "SVM is not modelled")
VGPU_HSA_REFUSED(hsa_amd_svm_prefetch_async, "SVM is not modelled")
VGPU_HSA_REFUSED(hsa_amd_interop_map_buffer, "there is no graphics driver to share with")
VGPU_HSA_REFUSED(hsa_amd_interop_unmap_buffer, "there is no graphics driver to share with")
VGPU_HSA_REFUSED(hsa_amd_portable_export_dmabuf, "there is no DMA-buf to export")
VGPU_HSA_REFUSED(hsa_executable_agent_global_variable_define, "external variables are not defined yet")

}  // extern "C"
