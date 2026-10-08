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
#include "vgpu/amd_chip.hpp"
#include "vgpu/amd_kfd.hpp"
#include "vgpu/amd_image.hpp"
#include "vgpu/hip_abi.hpp"
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

// The chip's facts beyond the profile (vgpu/amd_chip.hpp).
vgpu::amd::Chip chip(const vgpu::DeviceProfile& p) { return vgpu::amd::chip(p.architecture.c_str()); }
uint32_t chip_id(const vgpu::DeviceProfile& p) { return p.telemetry.pci_device_id ? p.telemetry.pci_device_id : 0x74a1; }
uint32_t compute_units(const vgpu::DeviceProfile& p) {
  return static_cast<uint32_t>(p.limits.multiprocessors) * chip(p).cus_per_mp;
}

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

// Signals created to be shared (HSA_AMD_SIGNAL_IPC): each one's serial, which
// its handle carries so a stale address is not taken for it, and how many
// creates and attaches have yet to be destroyed.
std::map<uintptr_t, std::pair<uint32_t, int>>& g_ipc_signals = *new std::map<uintptr_t, std::pair<uint32_t, int>>;   // under g_mutex
uint32_t g_ipc_serial = 0;   // under g_mutex
// Who asked to hear of an address's release (hsa_amd_register_deallocation_callback).
struct DeallocWatch {
  uintptr_t ptr;
  void (*callback)(void*, void*);
  void* user_data;
};
std::vector<DeallocWatch>& g_dealloc_watches = *new std::vector<DeallocWatch>;   // under g_mutex
// System memory a program moved to the other CPU pool (hsa_amd_memory_migrate): true for coarse-grained.
std::map<uintptr_t, bool>& g_system_grain = *new std::map<uintptr_t, bool>;   // under g_mutex

// A GPU's clock: 100 MHz, a tick every 10 ns.
constexpr uint64_t kGpuClockHz = 100000000, kGpuTickNs = 10;

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

// Whether a GPU has texture units: the Radeon targets (hsa_images.inc).
bool agent_has_images(int gpu) {
  const std::string& a = shared::profile(gpu).gcn_arch;
  return a.rfind("gfx10", 0) == 0 || a.rfind("gfx11", 0) == 0 || a.rfind("gfx12", 0) == 0;
}
// The largest image of each HSA geometry (1D, 2D, 3D, 1DA, 2DA, 1DB,
// 2DDEPTH, 2DADEPTH): width, height, depth, layers -- ROCm's image
// runtime's for every Radeon target (image_lut_kv.cpp).
constexpr uint32_t kImageMaxDims[8][4] = {
    {16384, 1, 1, 1},        {16384, 16384, 1, 1},  {16384, 16384, 8192, 1}, {16384, 1, 1, 8192},
    {16384, 16384, 1, 8192}, {0xFFFFFFFFu, 1, 1, 1}, {16384, 16384, 1, 1},    {16384, 16384, 1, 8192},
};
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
    // SVM's interfaces, as on a machine with HMM; ranges are given to an
    // agent explicitly (not accessible by default, as without XNACK).
    case HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED: put<bool>(value, true); break;
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
    case HSA_AGENT_INFO_NODE: put<uint32_t>(value, is_gpu ? static_cast<uint32_t>(shared::physical(gpu) + 1) : 0); break;
    case HSA_AGENT_INFO_DEVICE: put<uint32_t>(value, is_gpu ? HSA_DEVICE_TYPE_GPU : HSA_DEVICE_TYPE_CPU); break;
    case HSA_AGENT_INFO_CACHE_SIZE: {
      uint32_t sizes[4] = {0, 0, 0, 0};
      if (is_gpu) {
        const auto& p = shared::profile(gpu);
        sizes[0] = chip(p).l1_kb * 1024;
        sizes[1] = static_cast<uint32_t>(p.limits.l2_cache_bytes);
        sizes[2] = chip(p).l3_mb * 1024 * 1024;
      }
      std::memcpy(value, sizes, sizeof sizes);
      break;
    }
    case HSA_AGENT_INFO_ISA: put<hsa_isa_t>(value, {is_gpu ? agent.handle : 0}); break;
    // The images extension on every GPU agent, as ROCm's runtime has it; an
    // agent without texture units says so in its image limits (all 0).
    case HSA_AGENT_INFO_EXTENSIONS:
      std::memset(value, 0, 128);
      if (is_gpu) static_cast<uint8_t*>(value)[0] |= 1 << HSA_EXTENSION_IMAGES;
      break;
    case HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS: {
      // The geometry each asks about, and how many of its sizes it gives.
      static const int kGeometry[] = {0, 3, 5, 1, 4, 6, 7, 2};
      static const int kSizes[] = {1, 1, 1, 2, 2, 2, 2, 3};
      const int k = attr - HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS;
      const bool images = is_gpu && agent_has_images(gpu);
      for (int i = 0; i < kSizes[k]; ++i)
        static_cast<uint32_t*>(value)[i] = images ? kImageMaxDims[kGeometry[k]][i] : 0;
      break;
    }
    case HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS:
      put<uint32_t>(value, is_gpu && agent_has_images(gpu) ? kImageMaxDims[4][3] : 0);
      break;
    case HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES: put<uint32_t>(value, is_gpu && agent_has_images(gpu) ? 128 : 0); break;
    case HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES: put<uint32_t>(value, is_gpu && agent_has_images(gpu) ? 64 : 0); break;
    case HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS: put<uint32_t>(value, is_gpu && agent_has_images(gpu) ? 16 : 0); break;
    case HSA_EXT_AGENT_INFO_IMAGE_LINEAR_ROW_PITCH_ALIGNMENT:
      put<uint32_t>(value, is_gpu && agent_has_images(gpu) ? 256 : 0);
      break;
    case HSA_AGENT_INFO_VERSION_MAJOR: put<uint16_t>(value, 1); break;
    case HSA_AGENT_INFO_VERSION_MINOR: put<uint16_t>(value, 1); break;
    case HSA_AMD_AGENT_INFO_CHIP_ID: put<uint32_t>(value, is_gpu ? chip_id(shared::profile(gpu)) : 0); break;
    case HSA_AMD_AGENT_INFO_CACHELINE_SIZE: put<uint32_t>(value, is_gpu ? 128 : 64); break;
    case HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT:
      put<uint32_t>(value, is_gpu ? compute_units(shared::profile(gpu)) : cpus);
      break;
    case HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY:
      put<uint32_t>(value, is_gpu ? static_cast<uint32_t>(shared::profile(gpu).telemetry.sm_clock_max_mhz) : 0);
      break;
    case HSA_AMD_AGENT_INFO_DRIVER_NODE_ID:
      put<uint32_t>(value, is_gpu ? static_cast<uint32_t>(shared::physical(gpu) + 1) : 0);
      break;
    // The PCI location: bus ordinal + 1, device 0, function 0, as HIP reports it.
    case HSA_AMD_AGENT_INFO_BDFID:
      put<uint32_t>(value, is_gpu ? static_cast<uint32_t>(shared::physical(gpu) + 1) << 8 : 0);
      break;
    case HSA_AMD_AGENT_INFO_DOMAIN: put<uint32_t>(value, 0); break;
    case HSA_AMD_AGENT_INFO_PRODUCT_NAME:
      put_string(value, is_gpu ? shared::profile(gpu).model : "VirtualGPU host CPU", 64);
      break;
    case HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU: put<uint32_t>(value, is_gpu ? 32 : 0); break;
    case HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU: put<uint32_t>(value, is_gpu ? chip(shared::profile(gpu)).simds : 0); break;
    case HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES: put<bool>(value, is_gpu); break;
    case HSA_AMD_AGENT_INFO_UUID: {
      char uuid[21];
      if (is_gpu) std::snprintf(uuid, sizeof uuid, "GPU-%016llx", 0x5647505500000000ull + static_cast<unsigned>(shared::physical(gpu)));
      else std::snprintf(uuid, sizeof uuid, "CPU-XX");
      put_string(value, uuid, 21);
      break;
    }
    case HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS: put<bool>(value, false); break;
    case HSA_AMD_AGENT_INFO_MEMORY_AVAIL:
      put<uint64_t>(value, is_gpu ? shared::profile(gpu).vram_bytes : 0);
      break;
    // A GPU's own clock runs at 100 MHz -- the one s_memrealtime reads, and
    // what ROCm's HIP reports as the wall clock rate; the system's (and the
    // host's) is in nanoseconds.
    case HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY: put<uint64_t>(value, is_gpu ? kGpuClockHz : 1000000000); break;
    // The rest of what AMD's runtime reports, from the chip (chip() above).
    case HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS: put<uint32_t>(value, is_gpu ? 4 : 0); break;
    case HSA_AMD_AGENT_INFO_MEMORY_WIDTH: put<uint32_t>(value, is_gpu ? chip(shared::profile(gpu)).mem_bits : 0); break;
    case HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY:
      put<uint32_t>(value, is_gpu ? chip(shared::profile(gpu)).mem_mhz : 0);
      break;
    case HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES: put<uint32_t>(value, is_gpu ? chip(shared::profile(gpu)).engines : 0); break;
    case HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE:
      put<uint32_t>(value, is_gpu ? chip(shared::profile(gpu)).arrays : 0);
      break;
    case HSA_AMD_AGENT_INFO_HDP_FLUSH: std::memset(value, 0, 2 * sizeof(void*)); break;
    case HSA_AMD_AGENT_INFO_ASIC_REVISION: put<uint32_t>(value, is_gpu ? 1 : 0); break;
    case HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT:
      put<uint32_t>(value, is_gpu ? compute_units(shared::profile(gpu)) : 0);
      break;
    case HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID:
    case HSA_AMD_AGENT_INFO_UCODE_VERSION:
    case HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION:
    case HSA_AMD_AGENT_INFO_DRIVER_UID: put<uint32_t>(value, 0); break;
    case HSA_AMD_AGENT_INFO_NUM_SDMA_ENG: put<uint32_t>(value, is_gpu ? 2 : 0); break;
    case HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG: put<uint32_t>(value, 0); break;
    case HSA_AMD_AGENT_INFO_IOMMU_SUPPORT: put<uint32_t>(value, 0); break;   // HSA_IOMMU_SUPPORT_NONE
    case HSA_AMD_AGENT_INFO_NUM_XCC: put<uint32_t>(value, is_gpu ? chip(shared::profile(gpu)).xccs : 0); break;
    case HSA_AMD_AGENT_INFO_NEAREST_CPU: put<hsa_agent_t>(value, {kCpuAgent}); break;
    case HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES:
    case HSA_AMD_AGENT_INFO_AQL_EXTENSIONS: std::memset(value, 0, 8); break;
    case HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_MAX: put<uint64_t>(value, is_gpu ? shared::kScratchLimitMax : 0); break;
    case HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT: put<uint64_t>(value, is_gpu ? shared::scratch_limit(gpu) : 0); break;
    case HSA_AMD_AGENT_INFO_CLOCK_COUNTERS: {
      // The GPU's counter in its own ticks, the CPU's and the system's in
      // nanoseconds, all read at once.
      const uint64_t t = now_ns();
      put(value, hsa_amd_clock_counters_t{is_gpu ? t / kGpuTickNs : t, t, t, 1000000000});
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

// ---- Caches, wavefronts and ISA compatibility ---------------------------------------
//
// A GPU agent's caches are the levels hsa_agent_get_info's deprecated
// HSA_AGENT_INFO_CACHE_SIZE gives sizes for (L1, L2 and, where the chip has
// one, L3), the same numbers: a handle is the agent's times 16 plus the
// level. The CPU agent reports none, as that attribute does. The names are
// the levels' ("L1"): the specification asks only for a description.
// A wavefront's handle is its ISA's, and a GPU has one wavefront size, the
// profile's.

namespace {
uint32_t cache_size(int gpu, uint32_t level) {
  const auto& p = shared::profile(gpu);
  switch (level) {
    case 1: return chip(p).l1_kb * 1024;
    case 2: return static_cast<uint32_t>(p.limits.l2_cache_bytes);
    case 3: return chip(p).l3_mb * 1024 * 1024;
  }
  return 0;
}
}  // namespace

hsa_status_t hsa_agent_iterate_caches(hsa_agent_t agent, hsa_status_t (*callback)(hsa_cache_t, void*), void* data) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  if (!callback) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const int gpu = gpu_of(agent);
  if (gpu < 0) return HSA_STATUS_SUCCESS;
  for (uint32_t level = 1; level <= 3; ++level) {
    if (!cache_size(gpu, level)) continue;
    const hsa_status_t s = callback({agent.handle * 16 + level}, data);
    if (s != HSA_STATUS_SUCCESS) return s;   // the traversal stops, and says why
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_cache_get_info(hsa_cache_t cache, hsa_cache_info_t attribute, void* value) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  const uint32_t level = static_cast<uint32_t>(cache.handle % 16);
  const int gpu = gpu_of({cache.handle / 16});
  if (gpu < 0 || level < 1 || level > 3 || !cache_size(gpu, level)) return HSA_STATUS_ERROR_INVALID_CACHE;
  if (!value) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const std::string name = "L" + std::to_string(level);
  switch (attribute) {
    case HSA_CACHE_INFO_NAME_LENGTH: put<uint32_t>(value, static_cast<uint32_t>(name.size())); break;
    case HSA_CACHE_INFO_NAME: std::memcpy(value, name.c_str(), name.size() + 1); break;
    case HSA_CACHE_INFO_LEVEL: put<uint8_t>(value, static_cast<uint8_t>(level)); break;
    case HSA_CACHE_INFO_SIZE: put<uint32_t>(value, cache_size(gpu, level)); break;
    default: return unknown("hsa_cache_get_info", attribute);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_isa_iterate_wavefronts(hsa_isa_t isa, hsa_status_t (*callback)(hsa_wavefront_t, void*), void* data) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (gpu_of({isa.handle}) < 0) return HSA_STATUS_ERROR_INVALID_ISA;
  if (!callback) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  return callback({isa.handle}, data);
}

hsa_status_t hsa_wavefront_get_info(hsa_wavefront_t wavefront, hsa_wavefront_info_t attribute, void* value) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  const int gpu = gpu_of({wavefront.handle});
  if (gpu < 0) return HSA_STATUS_ERROR_INVALID_WAVEFRONT;
  if (!value) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (attribute != HSA_WAVEFRONT_INFO_SIZE) return unknown("hsa_wavefront_get_info", attribute);
  put<uint32_t>(value, shared::profile(gpu).warp_size);
  return HSA_STATUS_SUCCESS;
}

// Code for one ISA runs on an agent of the same ISA: the full target names
// (with their sramecc and xnack settings) are equal. Another target is
// reported incompatible, though a generic target's code can run on its
// family -- the machines here are never given one.
hsa_status_t hsa_isa_compatible(hsa_isa_t code_object_isa, hsa_isa_t agent_isa, bool* result) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  const int code = gpu_of({code_object_isa.handle}), agent = gpu_of({agent_isa.handle});
  if (code < 0 || agent < 0) return HSA_STATUS_ERROR_INVALID_ISA;
  if (!result) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *result = isa_name(code) == isa_name(agent);
  return HSA_STATUS_SUCCESS;
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
hsa_status_t hsa_amd_signal_create(hsa_signal_value_t initial, uint32_t n, const hsa_agent_t* consumers,
                                   uint64_t attributes, hsa_signal_t* signal) {
  if (const hsa_status_t st = hsa_signal_create(initial, n, consumers, signal); st != HSA_STATUS_SUCCESS) return st;
  if (attributes & HSA_AMD_SIGNAL_IPC) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_ipc_signals[static_cast<uintptr_t>(signal->handle)] = {++g_ipc_serial, 1};
  }
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_signal_destroy(hsa_signal_t signal) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!signal.handle) return HSA_STATUS_ERROR_INVALID_SIGNAL;
  {
    // A shared signal lives until every create and attach has been destroyed.
    std::lock_guard<std::mutex> lock(g_mutex);
    if (const auto it = g_ipc_signals.find(static_cast<uintptr_t>(signal.handle)); it != g_ipc_signals.end()) {
      if (--it->second.second > 0) return HSA_STATUS_SUCCESS;
      g_ipc_signals.erase(it);
    }
  }
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

// ---- Signal groups ------------------------------------------------------------------
//
// A group is a list of signals waited on together. The wait looks at each
// signal in turn, with the memory order the name says (every order is
// sequentially consistent here), and sleeps a little between rounds, as a
// single signal's wait does where a kernel changes a value without waking
// anyone. The handle is the group's address; a destroyed group is no longer
// one.

namespace {
struct SignalGroup {
  std::vector<hsa_signal_t> signals;
};
std::mutex& g_groups_mutex = *new std::mutex;
std::set<SignalGroup*>& live_groups() {
  static std::set<SignalGroup*>& groups = *new std::set<SignalGroup*>;
  return groups;
}
hsa_status_t wait_any_in_group(hsa_signal_group_t group, const hsa_signal_condition_t* conditions,
                               const hsa_signal_value_t* values, hsa_signal_t* signal, hsa_signal_value_t* value) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  std::vector<hsa_signal_t> signals;
  {
    std::lock_guard<std::mutex> lock(g_groups_mutex);
    auto* g = reinterpret_cast<SignalGroup*>(group.handle);
    if (!live_groups().count(g)) return HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP;
    signals = g->signals;   // the group may be destroyed while this waits; the signals are the caller's
  }
  if (!conditions || !values || !signal || !value) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (auto nap = std::chrono::microseconds(20);; nap = std::min(nap * 2, std::chrono::microseconds(1000))) {
    for (size_t i = 0; i < signals.size(); ++i) {
      const int64_t v = signal_of(signals[i])->amd.value.load();
      if (satisfied(v, conditions[i], values[i])) {
        *signal = signals[i];
        *value = v;
        return HSA_STATUS_SUCCESS;
      }
    }
    std::this_thread::sleep_for(nap);
  }
}
}  // namespace

hsa_status_t hsa_signal_group_create(uint32_t num_signals, const hsa_signal_t* signals, uint32_t num_consumers,
                                     const hsa_agent_t* consumers, hsa_signal_group_t* group) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!num_signals || !signals || !num_consumers || !consumers || !group) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (uint32_t i = 0; i < num_signals; ++i)
    if (!signals[i].handle) return HSA_STATUS_ERROR_INVALID_SIGNAL;
  auto* g = new SignalGroup;
  g->signals.assign(signals, signals + num_signals);
  {
    std::lock_guard<std::mutex> lock(g_groups_mutex);
    live_groups().insert(g);
  }
  group->handle = reinterpret_cast<uint64_t>(g);
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_signal_group_destroy(hsa_signal_group_t group) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  auto* g = reinterpret_cast<SignalGroup*>(group.handle);
  {
    std::lock_guard<std::mutex> lock(g_groups_mutex);
    if (!live_groups().erase(g)) return HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP;
  }
  delete g;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_signal_group_wait_any_scacquire(hsa_signal_group_t group, const hsa_signal_condition_t* conditions,
                                                 const hsa_signal_value_t* values, hsa_wait_state_t,
                                                 hsa_signal_t* signal, hsa_signal_value_t* value) {
  return wait_any_in_group(group, conditions, values, signal, value);
}
hsa_status_t hsa_signal_group_wait_any_relaxed(hsa_signal_group_t group, const hsa_signal_condition_t* conditions,
                                               const hsa_signal_value_t* values, hsa_wait_state_t,
                                               hsa_signal_t* signal, hsa_signal_value_t* value) {
  return wait_any_in_group(group, conditions, values, signal, value);
}

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
  const uint64_t lds = shared::profile(gpu_of(id.agent) >= 0 ? gpu_of(id.agent) : 0).limits.shared_mem_per_block;
  const uint64_t size = group    ? (lds ? lds : kLdsBytes)
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
    // The links from the agent to the pool, as KFD's topology lists them
    // (amd/src/kfd.cpp): zero where the pool is the agent's own or never
    // reachable; a GPU reaches system memory over its PCI Express link, an
    // Instinct GPU another's memory over one XGMI hop, a Radeon another's
    // through the host: two PCI Express hops.
    case HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS:
    case HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO: {
      const int gpu = gpu_of(agent);
      const bool to_gpu = id.kind == PoolKind::Device;
      const int peer = to_gpu ? gpu_of(id.agent) : -1;
      const bool xgmi = to_gpu && id.agent.handle != agent.handle && !never && gpu >= 0 && peer >= 0 &&
                        shared::profile(gpu).gcn_arch.rfind("gfx9", 0) == 0 &&
                        shared::profile(peer).gcn_arch.rfind("gfx9", 0) == 0;
      const bool belongs = to_gpu || id.kind == PoolKind::Group ? id.agent.handle == agent.handle
                                                                : agent.handle == kCpuAgent;
      const uint32_t hops = belongs || never ? 0 : xgmi ? 1 : to_gpu ? 2 : 1;
      if (attribute == HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS) {
        put<uint32_t>(value, hops);
        break;
      }
      const auto& prof = shared::profile(gpu >= 0 ? gpu : 0);
      const uint32_t pcie = vgpu::amd::kfd_pcie_mb_per_s(static_cast<uint32_t>(prof.telemetry.pcie_gen),
                                                         static_cast<uint32_t>(prof.telemetry.pcie_width));
      auto* out = static_cast<hsa_amd_memory_pool_link_info_t*>(value);
      for (uint32_t h = 0; h < hops; ++h) {
        out[h] = {};
        out[h].link_type = xgmi ? HSA_AMD_LINK_INFO_TYPE_XGMI : HSA_AMD_LINK_INFO_TYPE_PCIE;
        out[h].min_bandwidth = out[h].max_bandwidth = xgmi ? 64000 : pcie;
        out[h].numa_distance = xgmi ? 15 : 20;
        out[h].atomic_support_32bit = out[h].atomic_support_64bit = out[h].coherent_support = true;
      }
      break;
    }
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
  if (id.kind == PoolKind::SystemCoarse) g_system_grain[reinterpret_cast<uintptr_t>(p)] = true;
  *ptr = p;
  VGPU_HSA_TRACE("allocated %zu bytes of system memory at %p", rounded, p);
  return HSA_STATUS_SUCCESS;
}

namespace {
// Tells each watcher of an address in [base, end) that it is released, once:
// the notice is dropped as it is given (hsa_ext_amd.h).
void notify_deallocation(uintptr_t base, uintptr_t end) {
  std::vector<DeallocWatch> due;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (size_t i = 0; i < g_dealloc_watches.size();) {
      if (g_dealloc_watches[i].ptr >= base && g_dealloc_watches[i].ptr < end) {
        due.push_back(g_dealloc_watches[i]);
        g_dealloc_watches.erase(g_dealloc_watches.begin() + static_cast<long>(i));
      } else {
        ++i;
      }
    }
  }
  for (const DeallocWatch& w : due) w.callback(reinterpret_cast<void*>(w.ptr), w.user_data);
}
}  // namespace

hsa_status_t hsa_amd_memory_pool_free(void* ptr) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!ptr) return HSA_STATUS_SUCCESS;
  std::unique_lock<std::mutex> lock(g_mutex);
  if (const auto it = g_system.find(reinterpret_cast<uintptr_t>(ptr)); it != g_system.end()) {
    const uintptr_t base = it->first, end = it->first + it->second;
    g_system.erase(it);
    g_system_grain.erase(base);
    lock.unlock();
    notify_deallocation(base, end);
    shared::unmap_host(ptr);
    std::free(ptr);
    return HSA_STATUS_SUCCESS;
  }
  if (const auto it = g_device.find(reinterpret_cast<uint64_t>(ptr)); it != g_device.end()) {
    const int gpu = it->second.first;
    const uintptr_t base = it->first, end = it->first + it->second.second;
    g_device.erase(it);
    lock.unlock();
    notify_deallocation(base, end);
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
  *result = (extension == kExtensionAmdLoader && major == 1 && minor <= 3) ||
            (extension == HSA_EXTENSION_IMAGES && major == 1 && minor == 0);
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_system_major_extension_supported(uint16_t extension, uint16_t major, uint16_t* minor, bool* result) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!result || !minor) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *result = (extension == kExtensionAmdLoader || extension == HSA_EXTENSION_IMAGES) && major == 1;
  if (*result) *minor = extension == kExtensionAmdLoader ? 3 : 0;
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
  if (extension == HSA_EXTENSION_IMAGES && major == 1) {
    // hsa_ext_images_1_pfn_t, in its order.
    void* const images[] = {
        reinterpret_cast<void*>(hsa_ext_image_get_capability), reinterpret_cast<void*>(hsa_ext_image_data_get_info),
        reinterpret_cast<void*>(hsa_ext_image_create),         reinterpret_cast<void*>(hsa_ext_image_destroy),
        reinterpret_cast<void*>(hsa_ext_image_copy),           reinterpret_cast<void*>(hsa_ext_image_import),
        reinterpret_cast<void*>(hsa_ext_image_export),         reinterpret_cast<void*>(hsa_ext_image_clear),
        reinterpret_cast<void*>(hsa_ext_sampler_create),       reinterpret_cast<void*>(hsa_ext_sampler_destroy),
        reinterpret_cast<void*>(hsa_ext_image_get_capability_with_layout),
        reinterpret_cast<void*>(hsa_ext_image_data_get_info_with_layout),
        reinterpret_cast<void*>(hsa_ext_image_create_with_layout),
    };
    std::memcpy(table, images, std::min(table_length, sizeof images));
    return HSA_STATUS_SUCCESS;
  }
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
  // The loader's version 1.01 table, or the images extension's 1.00 one.
  return hsa_system_get_major_extension_table(extension, major,
                                              sizeof(void*) * (extension == HSA_EXTENSION_IMAGES ? 10 : 7), table);
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
      info.global_flags = g_system_grain.count(it->first) ? HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED
                                                          : HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED;
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
  // A GPU's 100 MHz ticks, and the system's nanoseconds, count from the same
  // moment.
  *system_tick = gpu_of(agent) >= 0 ? agent_tick * kGpuTickNs : agent_tick;
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

// The scratch limit is kept as set (shared::scratch_limit): nothing here runs
// short of scratch, but a program reads back what it asked for.
hsa_status_t hsa_amd_agent_set_async_scratch_limit(hsa_agent_t agent, size_t bytes) {
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  const int gpu = gpu_of(agent);
  if (gpu < 0) return HSA_STATUS_ERROR_INVALID_AGENT;
  return shared::set_scratch_limit(gpu, bytes) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_INVALID_ARGUMENT;
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

// ---- Images and samplers -----------------------------------------------------------
#include "hsa_images.inc"

// ---- Virtual memory and memory shared between processes --------------------------
#include "hsa_vmem.inc"

// ---- Shared virtual memory ---------------------------------------------------------
#include "hsa_svm.inc"


// ---- Waiting on several signals, shared signals, migration, release notices -------------

// The index of the first signal whose condition holds, and its value in
// *satisfying_value. hsa_ext_amd.h does not say what a wait that times out
// (or is given a bad argument) returns; this answers UINT32_MAX, which is no
// index into any list a caller can have passed. `timeout_hint` is in the
// system's 1 GHz ticks, as hsa_signal_wait's is, and UINT64_MAX waits for
// ever. Like every wait here it is relaxed and sleeps a little between looks.
uint32_t hsa_amd_signal_wait_any(uint32_t signal_count, hsa_signal_t* signals, hsa_signal_condition_t* conds,
                                 hsa_signal_value_t* values, uint64_t timeout_hint, hsa_wait_state_t,
                                 hsa_signal_value_t* satisfying_value) {
  constexpr uint32_t kNone = UINT32_MAX;
  if (!started()) return kNone;
  if (!signal_count || !signals || !conds || !values) {
    fail(HSA_STATUS_ERROR_INVALID_ARGUMENT, "hsa_amd_signal_wait_any: a list of signals, conditions and values is needed");
    return kNone;
  }
  for (uint32_t i = 0; i < signal_count; ++i)
    if (!signals[i].handle) {
      fail(HSA_STATUS_ERROR_INVALID_SIGNAL, "hsa_amd_signal_wait_any: signal " + std::to_string(i) + " is null");
      return kNone;
    }
  const bool forever = timeout_hint == UINT64_MAX || timeout_hint > (uint64_t{1} << 62);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(forever ? 0 : timeout_hint);
  for (auto nap = std::chrono::microseconds(20);; nap = std::min(nap * 2, std::chrono::microseconds(1000))) {
    for (uint32_t i = 0; i < signal_count; ++i) {
      const int64_t v = signal_of(signals[i])->amd.value.load();
      if (satisfied(v, conds[i], values[i])) {
        if (satisfying_value) *satisfying_value = v;
        return i;
      }
    }
    if (!forever && std::chrono::steady_clock::now() >= deadline) return kNone;
    std::this_thread::sleep_for(nap);
  }
}

// Runs the function on a thread of its own, as the runtime runs it on its
// asynchronous one.
hsa_status_t hsa_amd_async_function(void (*callback)(void*), void* arg) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!callback) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::thread([callback, arg] { callback(arg); }).detach();
  return HSA_STATUS_SUCCESS;
}

// A shared signal's handle: the process that made it, its serial and its address. Within one
// process it can be attached again; a handle from another process is refused, since a signal
// is an object of the process's own heap (its lock and its waiters), not a file.
namespace {
constexpr uint32_t kHsaIpcSignalMagic = 0x48534153;   // "HSAS"
}  // namespace
hsa_status_t hsa_amd_ipc_signal_create(hsa_signal_t signal, hsa_amd_ipc_signal_t* handle) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!signal.handle || !handle) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto it = g_ipc_signals.find(static_cast<uintptr_t>(signal.handle));
  if (it == g_ipc_signals.end())
    return fail(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                "hsa_amd_ipc_signal_create: the signal was not created with HSA_AMD_SIGNAL_IPC");
  std::memset(handle, 0, sizeof *handle);
  handle->handle[0] = kHsaIpcSignalMagic;
  handle->handle[1] = static_cast<uint32_t>(::getpid());
  handle->handle[2] = it->second.first;
  handle->handle[3] = static_cast<uint32_t>(signal.handle);
  handle->handle[4] = static_cast<uint32_t>(signal.handle >> 32);
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_amd_ipc_signal_attach(const hsa_amd_ipc_signal_t* handle, hsa_signal_t* signal) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!handle || !signal || handle->handle[0] != kHsaIpcSignalMagic) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (handle->handle[1] != static_cast<uint32_t>(::getpid()))
    return fail(HSA_STATUS_ERROR_NOT_SUPPORTED,
                "hsa_amd_ipc_signal_attach: a signal shared by another process is not supported: only an "
                "attach within the process that created the handle is");
  const uintptr_t at = static_cast<uintptr_t>(handle->handle[3]) | static_cast<uintptr_t>(handle->handle[4]) << 32;
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto it = g_ipc_signals.find(at);
  if (it == g_ipc_signals.end() || it->second.first != handle->handle[2])
    return fail(HSA_STATUS_ERROR_INVALID_ARGUMENT, "hsa_amd_ipc_signal_attach: the signal behind the handle is gone");
  if (it->second.second == INT32_MAX) return HSA_STATUS_ERROR_REFCOUNT_OVERFLOW;
  ++it->second.second;
  signal->handle = at;
  return HSA_STATUS_SUCCESS;
}

// Whether the contents of one pool can be moved to another. A buffer keeps its address when it
// moves, which holds here between the CPU's two pools (the difference is the grain the buffer
// is reported with) and from a pool to itself; moving between a GPU's memory and system memory,
// or between two GPUs, would change what the address means, and is not modelled.
hsa_status_t hsa_amd_memory_pool_can_migrate(hsa_amd_memory_pool_t src, hsa_amd_memory_pool_t dst, bool* result) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  PoolId a, b;
  if (!pool_of(src.handle, &a) || !pool_of(dst.handle, &b)) return HSA_STATUS_ERROR_INVALID_MEMORY_POOL;
  if (!result) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const bool cpu = a.agent.handle == kCpuAgent && b.agent.handle == kCpuAgent;
  *result = a.kind != PoolKind::Group && b.kind != PoolKind::Group &&
            ((cpu) || (a.agent.handle == b.agent.handle && a.kind == b.kind));
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_amd_memory_migrate(const void* ptr, hsa_amd_memory_pool_t pool, uint32_t flags) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  PoolId id;
  if (!pool_of(pool.handle, &id) || id.kind == PoolKind::Group) return HSA_STATUS_ERROR_INVALID_MEMORY_POOL;
  if (flags != 0 || !ptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> lock(g_mutex);
  const uintptr_t at = reinterpret_cast<uintptr_t>(ptr);
  if (g_system.count(at)) {
    if (id.kind == PoolKind::SystemFine) g_system_grain.erase(at);
    else if (id.kind == PoolKind::SystemCoarse) g_system_grain[at] = true;
    else
      return fail(HSA_STATUS_ERROR_NOT_SUPPORTED,
                  "hsa_amd_memory_migrate: moving system memory into a GPU's memory is not supported (the "
                  "buffer would need an address in that device's memory)");
    return HSA_STATUS_SUCCESS;
  }
  if (const auto it = g_device.find(at); it != g_device.end()) {
    if (id.kind == PoolKind::Device && gpu_of(id.agent) == it->second.first) return HSA_STATUS_SUCCESS;
    return fail(HSA_STATUS_ERROR_NOT_SUPPORTED,
                "hsa_amd_memory_migrate: moving a GPU's memory to system memory or to another GPU is not supported "
                "(the buffer would need another address)");
  }
  return fail(HSA_STATUS_ERROR_INVALID_ARGUMENT, "hsa_amd_memory_migrate: the address is the start of no allocation");
}

hsa_status_t hsa_amd_register_deallocation_callback(void* ptr, hsa_amd_deallocation_callback_t callback,
                                                    void* user_data) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!ptr || !callback) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> lock(g_mutex);
  const uintptr_t at = reinterpret_cast<uintptr_t>(ptr);
  if (containing(g_system, at) == g_system.end() && containing(g_device, at) == g_device.end())
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  g_dealloc_watches.push_back({at, callback, user_data});
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_amd_deregister_deallocation_callback(void* ptr, hsa_amd_deallocation_callback_t callback) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  std::lock_guard<std::mutex> lock(g_mutex);
  for (auto it = g_dealloc_watches.begin(); it != g_dealloc_watches.end(); ++it)
    if (it->ptr == reinterpret_cast<uintptr_t>(ptr) && it->callback == callback) {
      g_dealloc_watches.erase(it);
      return HSA_STATUS_SUCCESS;
    }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

// ---- The deprecated code object calls (hsa.h) -------------------------------------------
//
// Superseded by the reader and by hsa_executable_load_agent_code_object, and
// answered through them: a "code object" is the bytes, as the reader keeps them.
namespace {
std::set<Reader*>& g_old_code_objects = *new std::set<Reader*>;   // under g_mutex
}  // namespace
hsa_status_t hsa_code_object_deserialize(void* serialized, size_t size, const char*, hsa_code_object_t* code_object) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!serialized || !size || !code_object) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  auto* r = new Reader{std::string(static_cast<const char*>(serialized), size)};
  std::lock_guard<std::mutex> lock(g_mutex);
  g_old_code_objects.insert(r);
  code_object->handle = reinterpret_cast<uint64_t>(r);
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_code_object_destroy(hsa_code_object_t code_object) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  auto* r = reinterpret_cast<Reader*>(code_object.handle);
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_old_code_objects.erase(r)) return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
  delete r;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_executable_create(hsa_profile_t profile, hsa_executable_state_t state, const char* options,
                                   hsa_executable_t* executable) {
  if (const hsa_status_t st = hsa_executable_create_alt(profile, static_cast<hsa_default_float_rounding_mode_t>(0), options, executable);
      st != HSA_STATUS_SUCCESS)
    return st;
  if (state == HSA_EXECUTABLE_STATE_FROZEN) reinterpret_cast<Executable*>(executable->handle)->frozen = true;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_executable_load_code_object(hsa_executable_t executable, hsa_agent_t agent,
                                             hsa_code_object_t code_object, const char* options) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_old_code_objects.count(reinterpret_cast<Reader*>(code_object.handle)))
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
  return hsa_executable_load_agent_code_object(executable, agent, {code_object.handle}, options, nullptr);
}
// The symbol of the program, which is every symbol here: there are no modules.
hsa_status_t hsa_executable_get_symbol(hsa_executable_t executable, const char* module_name, const char* symbol_name,
                                       hsa_agent_t agent, int32_t, hsa_executable_symbol_t* symbol) {
  if (module_name && *module_name)
    return fail(HSA_STATUS_ERROR_INVALID_SYMBOL_NAME,
                "hsa_executable_get_symbol: a symbol of a named module is not supported (an ELF code object has no modules)");
  return hsa_executable_get_symbol_by_name(executable, symbol_name, &agent, symbol);
}
// An executable here has nothing left undefined, so it is valid as it stands.
hsa_status_t hsa_executable_validate(hsa_executable_t executable, uint32_t* result) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!executable_of(executable)) return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  if (!result) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *result = 0;
  return HSA_STATUS_SUCCESS;
}

// What this does not model, refused by name: images laid out by graphics
// interop (hsa_amd_image_create), and graphics interop.
#define VGPU_HSA_REFUSED(name, what) \
  hsa_status_t name() { return fail(HSA_STATUS_ERROR_NOT_SUPPORTED, #name " is not supported: " what); }
VGPU_HSA_REFUSED(hsa_amd_image_create, "images over another API's layout are not modelled")
VGPU_HSA_REFUSED(hsa_amd_interop_map_buffer, "there is no graphics driver to share with")
VGPU_HSA_REFUSED(hsa_amd_interop_unmap_buffer, "there is no graphics driver to share with")
VGPU_HSA_REFUSED(hsa_executable_agent_global_variable_define,
                 "a variable defined from outside would have to be bound to the code object's undefined symbol, and "
                 "the loader does not resolve undefined symbols")
VGPU_HSA_REFUSED(hsa_executable_global_variable_define, "a variable defined from outside is not bound to a code object's undefined symbol")
VGPU_HSA_REFUSED(hsa_executable_readonly_variable_define, "a variable defined from outside is not bound to a code object's undefined symbol")

// Stream Performance Monitor: KFD's hardware counter stream, which no simulated counter backs.
VGPU_HSA_REFUSED(hsa_amd_spm_acquire, "there is no stream of performance counters to take")
VGPU_HSA_REFUSED(hsa_amd_spm_release, "there is no stream of performance counters to give back")
VGPU_HSA_REFUSED(hsa_amd_spm_set_dest_buffer, "there is no stream of performance counters to write")

// The finalizer (hsa_ext_finalize.h) turns HSAIL into machine code. AMD's runtime has no finalizer
// since ROCm 2 (no extension answers HSA_EXTENSION_FINALIZER), and none exists here.
VGPU_HSA_REFUSED(hsa_ext_program_create, "there is no finalizer, HSAIL is not compiled")
VGPU_HSA_REFUSED(hsa_ext_program_destroy, "there is no finalizer, HSAIL is not compiled")
VGPU_HSA_REFUSED(hsa_ext_program_add_module, "there is no finalizer, HSAIL is not compiled")
VGPU_HSA_REFUSED(hsa_ext_program_iterate_modules, "there is no finalizer, HSAIL is not compiled")
VGPU_HSA_REFUSED(hsa_ext_program_get_info, "there is no finalizer, HSAIL is not compiled")
VGPU_HSA_REFUSED(hsa_ext_program_finalize, "there is no finalizer, HSAIL is not compiled")

// The rest of the deprecated code object interface: what it reads is a finalizer's code object
// (with modules and per-symbol records) rather than an ELF the loader reads.
VGPU_HSA_REFUSED(hsa_code_object_serialize, "a code object here is the ELF it was loaded from")
VGPU_HSA_REFUSED(hsa_code_object_get_info, "a code object's header fields are not kept apart from its ELF")
VGPU_HSA_REFUSED(hsa_code_object_get_symbol, "a code object's symbols are an executable's here")
VGPU_HSA_REFUSED(hsa_code_object_get_symbol_from_name, "a code object's symbols are an executable's here")
VGPU_HSA_REFUSED(hsa_code_object_iterate_symbols, "a code object's symbols are an executable's here")
VGPU_HSA_REFUSED(hsa_code_symbol_get_info, "a code object's symbols are an executable's here")
VGPU_HSA_REFUSED(hsa_executable_load_program_code_object, "a code object is loaded for an agent")
VGPU_HSA_REFUSED(hsa_executable_iterate_program_symbols, "symbols are iterated for an agent")

// Queue interception and the AQL profile library: interception hands the packets a program writes
// to a tool before the packet processor sees them, and the profile library turns counter requests
// into PM4 packets for that hardware; neither is modelled (rocprofiler's counters here come from
// the executor, not from packets).
VGPU_HSA_REFUSED(hsa_amd_queue_intercept_create, "packets are not diverted to a tool before the packet processor")
VGPU_HSA_REFUSED(hsa_amd_queue_intercept_register, "packets are not diverted to a tool before the packet processor")
VGPU_HSA_REFUSED(hsa_ven_amd_aqlprofile_validate_event, "there are no PM4 counter packets")
VGPU_HSA_REFUSED(hsa_ven_amd_aqlprofile_start, "there are no PM4 counter packets")
VGPU_HSA_REFUSED(hsa_ven_amd_aqlprofile_stop, "there are no PM4 counter packets")
VGPU_HSA_REFUSED(hsa_ven_amd_aqlprofile_read, "there are no PM4 counter packets")
VGPU_HSA_REFUSED(hsa_ven_amd_aqlprofile_legacy_get_pm4, "there are no PM4 counter packets")
VGPU_HSA_REFUSED(hsa_ven_amd_aqlprofile_get_info, "there are no PM4 counter packets")
VGPU_HSA_REFUSED(hsa_ven_amd_aqlprofile_iterate_data, "there are no PM4 counter packets")

}  // extern "C"
