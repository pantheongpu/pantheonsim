// librocprofiler-sdk: rocprofiler-sdk's interface, answered from VirtualGPU's
// HIP runtime, so AMD's own profiler (rocprofv3) runs on a simulated GPU as it
// is.
//
// A profiling tool asks this library for the machine's agents and the
// counters each supports, subscribes to what the runtime does, and is handed
// records. The records here come from libamdhip64 (vgpu/hip_profiler.hpp):
// every HIP call, every code object placed on a device and its kernels, every
// copy and allocation, and every kernel launch with the instructions its waves
// issued. The declarations are rocprofiler-sdk's (vgpu/rocprofiler_abi.hpp);
// nothing here is AMD's code.
//
// What is counted is what the interpreter can count exactly: waves, and the
// instructions they issue by the unit that takes them (vgpu/amd_exec.hpp).
// Counters of cycles, stalls, caches and memory traffic need a model of the
// hardware's timing that there is none of here, so they are not offered at
// all: a tool asking for one is told the device does not have it, which is
// how rocprofv3 reports a counter a real GPU lacks, rather than given a
// number that was made up.
//
// What is not here: HSA (there is no HSA runtime -- the HIP runtime reaches the
// simulated devices directly), PC sampling and thread trace (which need the
// hardware's sampling and trace units), and the kernels ROCm's runtime runs
// on its own behalf, such as the ones that do a device-to-device copy or a
// memset: here those happen without a kernel, so there is none to report.
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "vgpu/hip_profiler.hpp"
#include "vgpu/rocprofiler_abi.hpp"

using namespace vgpu::amd::rocprof;
namespace hp = vgpu::amd::hipprof;
using vgpu::amd::DispatchStats;

#define ROCPROFILER_API extern "C" __attribute__((visibility("default")))

namespace {

// ---- The counters -------------------------------------------------------------
//
// The descriptions are AMD's (rocprofiler-sdk's counter_defs.yaml, MIT licence;
// the notice is in vgpu/rocprofiler_abi.hpp).

struct Counter {
  const char* name;
  const char* block;          // "" for a derived counter
  const char* expression;     // "" for a hardware counter
  uint64_t (*value)(const DispatchStats&);
  const char* description;
};

const Counter kCounters[] = {
    {"SQ_WAVES", "SQ", "", [](const DispatchStats& s) { return s.waves; },
     "Count number of waves sent to distributed sequencers (SQs). This value represents the number of waves that are "
     "sent to each SQ. This only counts new waves sent since the start of collection (for dispatch profiling this is "
     "the timeframe of kernel execution, for agent profiling it is the timeframe between start_context and read "
     "counter data). A sum of all SQ_WAVES values will give the total number of waves started by the application "
     "during the collection timeframe. Returns one value per-SE (aggregates of SIMD values)."},
    {"SQ_WAVES_EQ_64", "SQ", "", [](const DispatchStats& s) { return s.waves_eq64; },
     "Count number of waves with exactly 64 active threads sent to SQs. This value represents the number of waves "
     "that an each individual SIMD has enqueued during the collection timeframe (for dispatch profiling this is the "
     "timeframe of kernel execution, for agent profiling it is the timeframe between start_context and read counter "
     "data) with exactly 64 threads. A sum of all SQ_WAVES_EQ_64 values will give the total number of waves with 64 "
     "threads enqueued during the collection timeframe by the application. Returns one value per-SE (aggregates of "
     "SIMD values). Useful for checking for wavefront occupancy."},
    {"SQ_WAVES_LT_64", "SQ", "", [](const DispatchStats& s) { return s.waves_lt64; },
     "Count number of waves with <64 active threads sent to SQs. This value represents the number of waves that an "
     "each individual SIMD has enqueued during the collection timeframe (for dispatch profiling this is the timeframe "
     "of kernel execution, for agent profiling it is the timeframe between start_context and read counter data) with "
     "less than 64 threads. A sum of all SQ_WAVES_LT_64 values will give the total number of waves with 64 threads "
     "enqueued during the collection timeframe by the application. Returns one value per-SE (aggregates of SIMD "
     "values). Useful for checking for wavefront occupancy."},
    {"SQ_WAVES_LT_48", "SQ", "", [](const DispatchStats& s) { return s.waves_lt48; },
     "Count number of waves with <48 active threads sent to SQs. This value represents the number of waves that an "
     "each individual SIMD has enqueued during the collection timeframe (for dispatch profiling this is the timeframe "
     "of kernel execution, for agent profiling it is the timeframe between start_context and read counter data) with "
     "less than 48 threads. A sum of all SQ_WAVES_LT_48 values will give the total number of waves with 48 threads "
     "enqueued during the collection timeframe by the application. Returns one value per-SE (aggregates of SIMD "
     "values). Useful for checking for wavefront occupancy."},
    {"SQ_WAVES_LT_32", "SQ", "", [](const DispatchStats& s) { return s.waves_lt32; },
     "Count number of waves sent <32 active threads sent to SQs. This value represents the number of waves that an "
     "each individual SIMD has enqueued during the collection timeframe (for dispatch profiling this is the timeframe "
     "of kernel execution, for agent profiling it is the timeframe between start_context and read counter data) with "
     "less than 32 threads. A sum of all SQ_WAVES_LT_32 values will give the total number of waves with 32 threads "
     "enqueued during the collection timeframe by the application. Returns one value per-SE (aggregates of SIMD "
     "values). Useful for checking for wavefront occupancy."},
    {"SQ_WAVES_LT_16", "SQ", "", [](const DispatchStats& s) { return s.waves_lt16; },
     "Count number of waves sent <16 active threads sent to SQs. (per-simd, emulated, global). This value represents "
     "the number of waves that an each individual SIMD has enqueued during the collection timeframe (for dispatch "
     "profiling this is the timeframe of kernel execution, for agent profiling it is the timeframe between "
     "start_context and read counter data) with less than 16 threads. A sum of all SQ_WAVES_LT_16 values will give "
     "the total number of waves with 16 threads enqueued during the collection timeframe by the application. Returns "
     "one value per-SE (aggregates of SIMD values). Useful for checking for wavefront occupancy."},
    {"SQ_INSTS_VALU", "SQ", "", [](const DispatchStats& s) { return s.counts.valu; },
     "The number of VALU (Vector ALU) instructions issued. The value is returned per-SE (aggregate of values in SIMDs "
     "in the SE). See AMD ISAs for more information on VALU instructions."},
    {"SQ_INSTS_MFMA", "SQ", "", [](const DispatchStats& s) { return s.counts.mfma; },
     "Total number of MFMA (Matrix-Fused-Multiply-Add) instructions issued. This value is returned per-SE (aggregate "
     "of values in SIMDs in the SE). See AMD ISAs for more information on MFMA instructions."},
    {"SQ_INSTS_SALU", "SQ", "", [](const DispatchStats& s) { return s.counts.salu; },
     "Total Number of SALU (Scalar ALU) instructions issued. This value is returned per-SE (aggregate of values in "
     "SIMDs in the SE). See AMD ISAs for more information on SALU instructions."},
    {"SQ_INSTS_SMEM", "SQ", "", [](const DispatchStats& s) { return s.counts.smem; },
     "Total number of SMEM (Scalar Memory Read) instructions issued. This value is returned per-SE (aggregate of "
     "values in SIMDs in the SE). See AMD ISAs for more information on SMEM instructions."},
    {"SQ_INSTS_VMEM", "SQ", "", [](const DispatchStats& s) { return s.counts.vmem; },
     "The number of VMEM (GPU Memory) instructions issued. The value is returned per-SE (aggregate of values in SIMDs "
     "in the SE)."},
    {"SQ_INSTS_FLAT", "SQ", "", [](const DispatchStats& s) { return s.counts.flat; },
     "Total number of FLAT instructions issued. When used in combination with SQ_ACTIVE_INST_FLAT (cycle count for "
     "executing instructions) the average latency of FLAT instruction execution can be calculated "
     "(SQ_ACTIVE_INST_FLAT / SQ_INSTS). This value is returned per-SE (aggregate of values in SIMDs in the SE)."},
    {"SQ_INSTS_LDS", "SQ", "", [](const DispatchStats& s) { return s.counts.lds; },
     "Total number of LDS instructions issued (including FLAT). This value is returned per-SE (aggregate of values in "
     "SIMDs in the SE). See AMD ISAs for more information on LDS instructions."},
    {"SQ_INSTS_BRANCH", "SQ", "", [](const DispatchStats& s) { return s.counts.branch; },
     "Total number of BRANCH instructions issued. This value is returned per-SE (aggregate of values in SIMDs in the "
     "SE). This value SHOULD NOT be used in combination with SQ_ACTIVE_INST_MISC to calculate latency. "
     "SQ_ACTIVE_INST_MISC includes both BRANCH and SENDMSG instructions while this is only BRANCH."},
    {"SQ_INSTS_SENDMSG", "SQ", "", [](const DispatchStats& s) { return s.counts.sendmsg; },
     "Total number of Sendmsg (typically an interrupt to the CPU host) instructions issued. This value is returned "
     "per-SE (aggregate of values in SIMDs in the SE). See AMD ISAs for more information on Sendmsg instructions."},
    {"SQ_INSTS_GDS", "SQ", "", [](const DispatchStats& s) { return s.counts.gds; },
     "Total number of GDS (global data sync) instructions issued. This value is returned per-SE (aggregate of values "
     "in SIMDs in the SE). See AMD ISAs for more information on GDS (global data sync) instructions."},
    // A compute kernel exports nothing, and GDS is refused rather than run, so
    // every instruction this counts is a GDS one.
    {"SQ_INSTS_EXP_GDS", "SQ", "", [](const DispatchStats& s) { return s.counts.gds; },
     "Total number of EXPORT or GDS (global wave state) instructions issued. When used in combination with "
     "SQ_ACTIVE_INST_EXP_GDS (cycle count for executing instructions) the average latency of EXPORT/GDS instruction "
     "execution can be calculated (SQ_ACTIVE_INST_EXP_GDS / SQ_INSTS_EXP_GDS). This value is returned per-SE "
     "(aggregate of values in SIMDs in the SE)."},
    {"TA_FLAT_WAVEFRONTS", "TA", "", [](const DispatchStats& s) { return s.counts.flat; },
     "Number of flat opcode wavfronts processed by the TA."},
    {"TA_FLAT_READ_WAVEFRONTS", "TA", "", [](const DispatchStats& s) { return s.counts.flat_read; },
     "Number of flat opcode reads processed by the TA."},
    {"TA_FLAT_WRITE_WAVEFRONTS", "TA", "", [](const DispatchStats& s) { return s.counts.flat_write; },
     "Number of flat opcode writes processed by the TA."},
    {"TA_FLAT_ATOMIC_WAVEFRONTS", "TA", "", [](const DispatchStats& s) { return s.counts.flat_atomic; },
     "Number of flat opcode atomics processed by the TA."},
    {"SQ_WAVES_sum", "", "reduce(SQ_WAVES,sum)", [](const DispatchStats& s) { return s.waves; },
     "Gives the total number of waves currently enqueued by the application during the collection timeframe (for "
     "dispatch profiling this is the timeframe of kernel execution, for agent profiling it is the timeframe between "
     "start_context and read counter data). See SQ_WAVES for more details."},
    {"TA_FLAT_WAVEFRONTS_sum", "", "reduce(TA_FLAT_WAVEFRONTS,sum)", [](const DispatchStats& s) { return s.counts.flat; },
     "Number of flat opcode wavfronts processed by the TA. Sum over TA instances."},
    {"TA_FLAT_READ_WAVEFRONTS_sum", "", "reduce(TA_FLAT_READ_WAVEFRONTS,sum)",
     [](const DispatchStats& s) { return s.counts.flat_read; },
     "Number of flat opcode reads processed by the TA. Sum over TA instances."},
    {"TA_FLAT_WRITE_WAVEFRONTS_sum", "", "reduce(TA_FLAT_WRITE_WAVEFRONTS,sum)",
     [](const DispatchStats& s) { return s.counts.flat_write; },
     "Number of flat opcode writes processed by the TA. Sum over TA instances."},
    {"TA_FLAT_ATOMIC_WAVEFRONTS_sum", "", "reduce(TA_FLAT_ATOMIC_WAVEFRONTS,sum)",
     [](const DispatchStats& s) { return s.counts.flat_atomic; },
     "Number of flat opcode atomics processed by the TA. Sum over TA instances."},
};
constexpr size_t kCounterCount = sizeof kCounters / sizeof kCounters[0];

// A counter's id is its place in the table, from 1. Each has one instance:
// the device's total, since the interpreter does not say which shader engine
// or texture unit a wave used -- the value a tool gets by summing a real GPU's
// instances. An instance id is the counter's id above its position.
const Counter* counter(uint64_t id) { return id >= 1 && id <= kCounterCount ? &kCounters[id - 1] : nullptr; }
uint64_t instance_id(uint64_t counter_id) { return counter_id << 32; }

const rocprofiler_counter_record_dimension_info_t kDimension{"DIMENSION_INSTANCE", 1, 0};
const rocprofiler_counter_record_dimension_info_t* const kDimensions[] = {&kDimension};
const rocprofiler_counter_dimension_info_t kInstanceDimension{sizeof(rocprofiler_counter_dimension_info_t),
                                                              "DIMENSION_INSTANCE", 0};
const rocprofiler_counter_dimension_info_t* const kInstanceDimensions[] = {&kInstanceDimension};

// ---- Names --------------------------------------------------------------------

const char* const kBufferKinds[] = {
    "NONE", "HSA_CORE_API", "HSA_AMD_EXT_API", "HSA_IMAGE_EXT_API", "HSA_FINALIZE_EXT_API", "HIP_RUNTIME_API",
    "HIP_COMPILER_API", "MARKER_CORE_API", "MARKER_CONTROL_API", "MARKER_NAME_API", "MEMORY_COPY",
    "KERNEL_DISPATCH", "SCRATCH_MEMORY", "CORRELATION_ID_RETIREMENT", "RCCL_API", "OMPT", "MEMORY_ALLOCATION",
    "RUNTIME_INITIALIZATION", "ROCDECODE_API", "ROCJPEG_API", "HIP_STREAM", "HIP_RUNTIME_API_EXT",
    "HIP_COMPILER_API_EXT", "ROCDECODE_API_EXT", "KFD_EVENT_PAGE_MIGRATE", "KFD_EVENT_PAGE_FAULT",
    "KFD_EVENT_QUEUE", "KFD_EVENT_UNMAP_FROM_GPU", "KFD_EVENT_DROPPED_EVENTS", "KFD_PAGE_MIGRATE",
    "KFD_PAGE_FAULT", "KFD_QUEUE", "MARKER_CORE_RANGE_API"};
static_assert(sizeof kBufferKinds / sizeof kBufferKinds[0] == ROCPROFILER_BUFFER_TRACING_LAST);

const char* const kCallbackKinds[] = {
    "NONE", "HSA_CORE_API", "HSA_AMD_EXT_API", "HSA_IMAGE_EXT_API", "HSA_FINALIZE_EXT_API", "HIP_RUNTIME_API",
    "HIP_COMPILER_API", "MARKER_CORE_API", "MARKER_CONTROL_API", "MARKER_NAME_API", "CODE_OBJECT",
    "SCRATCH_MEMORY", "KERNEL_DISPATCH", "MEMORY_COPY", "RCCL_API", "OMPT", "MEMORY_ALLOCATION",
    "RUNTIME_INITIALIZATION", "ROCDECODE_API", "ROCJPEG_API", "HIP_STREAM", "MARKER_CORE_RANGE_API"};
static_assert(sizeof kCallbackKinds / sizeof kCallbackKinds[0] == ROCPROFILER_CALLBACK_TRACING_LAST);

struct HipApi {
  const char* name;
  int id;
};
const HipApi kHipApis[] = {
#include "rocprofiler_hip_ids.inc"
};

int hip_api_id(const char* name) {
  static const std::map<std::string, int> ids = [] {
    std::map<std::string, int> m;
    for (const HipApi& a : kHipApis) m.emplace(a.name, a.id);
    return m;
  }();
  const auto it = ids.find(name);
  return it == ids.end() ? -1 : it->second;
}

// The operations of the kinds this names: the names are the enumerations'
// own, less their prefix.
const std::vector<const char*>* operation_names(const char* kind) {
  static const std::map<std::string, std::vector<const char*>> ops = {
      {"MEMORY_COPY", {"MEMORY_COPY_NONE", "MEMORY_COPY_HOST_TO_HOST", "MEMORY_COPY_HOST_TO_DEVICE",
                       "MEMORY_COPY_DEVICE_TO_HOST", "MEMORY_COPY_DEVICE_TO_DEVICE"}},
      {"KERNEL_DISPATCH", {"KERNEL_DISPATCH_NONE", "KERNEL_DISPATCH_ENQUEUE", "KERNEL_DISPATCH_COMPLETE"}},
      {"MEMORY_ALLOCATION", {"MEMORY_ALLOCATION_NONE", "MEMORY_ALLOCATION_ALLOCATE", "MEMORY_ALLOCATION_VMEM_ALLOCATE",
                             "MEMORY_ALLOCATION_FREE", "MEMORY_ALLOCATION_VMEM_FREE"}},
      {"SCRATCH_MEMORY", {"SCRATCH_MEMORY_NONE", "SCRATCH_MEMORY_ALLOC", "SCRATCH_MEMORY_FREE",
                          "SCRATCH_MEMORY_ASYNC_RECLAIM"}},
      {"CODE_OBJECT", {"CODE_OBJECT_NONE", "CODE_OBJECT_LOAD", "CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER",
                       "CODE_OBJECT_HOST_KERNEL_SYMBOL_REGISTER"}},
      {"RUNTIME_INITIALIZATION", {"RUNTIME_INITIALIZATION_NONE", "RUNTIME_INITIALIZATION_HSA",
                                  "RUNTIME_INITIALIZATION_HIP", "RUNTIME_INITIALIZATION_MARKER",
                                  "RUNTIME_INITIALIZATION_RCCL", "RUNTIME_INITIALIZATION_ROCDECODE",
                                  "RUNTIME_INITIALIZATION_ROCJPEG"}},
      {"HIP_STREAM", {"HIP_STREAM_NONE", "HIP_STREAM_CREATE", "HIP_STREAM_DESTROY", "HIP_STREAM_SET"}},
  };
  const auto it = ops.find(kind);
  return it == ops.end() ? nullptr : &it->second;
}

bool is_hip_api(const char* kind) {
  return std::strcmp(kind, "HIP_RUNTIME_API") == 0 || std::strcmp(kind, "HIP_RUNTIME_API_EXT") == 0;
}

// An operation's name, or nullptr. NONE (0) has no name a tool is shown.
const char* operation_name(const char* kind, int op) {
  if (is_hip_api(kind)) {
    for (const HipApi& a : kHipApis)
      if (a.id == op) return a.name;
    return nullptr;
  }
  const auto* names = operation_names(kind);
  if (!names || op <= 0 || static_cast<size_t>(op) >= names->size()) return nullptr;
  return (*names)[static_cast<size_t>(op)];
}

template <class Cb>
rocprofiler_status_t iterate_operations(const char* kind, Cb&& each) {
  if (is_hip_api(kind)) {
    for (const HipApi& a : kHipApis)
      if (each(a.id) != 0) break;
    return ROCPROFILER_STATUS_SUCCESS;
  }
  if (const auto* names = operation_names(kind))
    for (size_t op = 1; op < names->size(); ++op)
      if (each(static_cast<int>(op)) != 0) break;
  return ROCPROFILER_STATUS_SUCCESS;
}

// ---- State --------------------------------------------------------------------

struct CallbackService {
  rocprofiler_callback_tracing_cb_t callback = nullptr;
  void* data = nullptr;
  std::set<int> operations;   // empty: every one
};
struct BufferService {
  uint64_t buffer = 0;
  std::set<int> operations;
};

struct Context {
  uint64_t id = 0;
  bool active = false;
  std::map<int, CallbackService> callbacks;   // by callback tracing kind
  std::map<int, BufferService> buffered;      // by buffer tracing kind
  rocprofiler_dispatch_counting_service_cb_t dispatch = nullptr;
  void* dispatch_data = nullptr;
  rocprofiler_dispatch_counting_record_cb_t record = nullptr;
  void* record_data = nullptr;
  rocprofiler_external_correlation_id_request_cb_t external = nullptr;
  void* external_data = nullptr;
  std::set<int> external_kinds;
};

struct Buffer {
  uint64_t id = 0;
  uint64_t context = 0;
  size_t watermark = 0;
  rocprofiler_buffer_tracing_cb_t callback = nullptr;
  void* data = nullptr;
  std::vector<rocprofiler_record_header_t> headers;
  std::vector<std::unique_ptr<uint8_t[]>> payloads;
  size_t held = 0;   // bytes of records not yet delivered
};

struct Config {
  uint64_t agent = 0;
  std::vector<uint64_t> counters;
};

struct Client {
  rocprofiler_client_id_t* id = nullptr;
  rocprofiler_tool_configure_result_t* result = nullptr;
  bool finalized = false;
};

struct Agents {
  std::vector<rocprofiler_agent_v0_t> list;
  std::deque<std::string> strings;   // what the agents' names point into
  std::vector<int> ordinal;          // each agent's device, or -1 for the CPU
};

struct Sdk {
  std::recursive_mutex mutex;
  bool configured = false, initialized = false, finalized = false;
  std::vector<Client> clients;
  std::vector<Context> contexts;
  std::vector<Buffer> buffers;
  std::map<uint64_t, Config> configs;
  uint64_t next_config = 1, next_callback_thread = 1;
  Agents agents;
  bool agents_ready = false;
  std::atomic<uint64_t> next_correlation{1};
};

// Never destroyed: a tool finalizes from an atexit handler, which may run
// after this library's static destructors would have.
Sdk& sdk() {
  static Sdk* s = new Sdk;
  return *s;
}

uint64_t thread_id() { return static_cast<uint64_t>(::syscall(SYS_gettid)); }

Context* context(Sdk& s, rocprofiler_context_id_t id) {
  for (Context& c : s.contexts)
    if (c.id == id.handle) return &c;
  return nullptr;
}
Buffer* buffer(Sdk& s, uint64_t id) {
  for (Buffer& b : s.buffers)
    if (b.id == id) return &b;
  return nullptr;
}

// ---- Agents -------------------------------------------------------------------

std::string cpuinfo(const char* key) {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line))
    if (line.rfind(key, 0) == 0) {
      const size_t colon = line.find(':');
      if (colon == std::string::npos) continue;
      std::string v = line.substr(colon + 1);
      v.erase(0, v.find_first_not_of(" \t"));
      return v;
    }
  return "";
}

// "gfx942" is 9.4.2: the major version, then the minor and the stepping as a
// hex digit each, which is how KFD's gfx_target_version writes them
// (90402).
uint32_t gfx_target_version(const std::string& gfx) {
  if (gfx.rfind("gfx", 0) != 0 || gfx.size() < 6) return 0;
  const std::string digits = gfx.substr(3);
  const uint32_t major = static_cast<uint32_t>(std::strtoul(digits.substr(0, digits.size() - 2).c_str(), nullptr, 10));
  const uint32_t minor = static_cast<uint32_t>(std::strtoul(digits.substr(digits.size() - 2, 1).c_str(), nullptr, 16));
  const uint32_t step = static_cast<uint32_t>(std::strtoul(digits.substr(digits.size() - 1).c_str(), nullptr, 16));
  return major * 10000 + minor * 100 + step;
}

// The machine's agents as KFD numbers them: the CPU first, then each GPU.
// What a GPU's profile does not say (its shader engines and arrays, its
// XCDs, its engines' firmware) is left zero rather than guessed.
void load_agents(Sdk& s) {
  if (s.agents_ready) return;
  s.agents_ready = true;
  Agents& a = s.agents;
  auto keep = [&](std::string v) { return a.strings.emplace_back(std::move(v)).c_str(); };

  rocprofiler_agent_v0_t cpu{};
  cpu.size = sizeof cpu;
  cpu.type = ROCPROFILER_AGENT_TYPE_CPU;
  cpu.cpu_cores_count = std::thread::hardware_concurrency();
  const std::string model = cpuinfo("model name");
  cpu.name = keep(model);
  cpu.product_name = keep(model);
  cpu.vendor_name = keep(cpuinfo("vendor_id"));
  cpu.model_name = keep("");
  cpu.runtime_visibility.hsa = 1;
  a.list.push_back(cpu);
  a.ordinal.push_back(-1);

  for (int i = 0;; ++i) {
    hp::Device d;
    if (vgpu_hip_profiler_device(i, &d) != 0) break;
    rocprofiler_agent_v0_t g{};
    g.size = sizeof g;
    g.type = ROCPROFILER_AGENT_TYPE_GPU;
    g.cu_count = d.compute_units;
    // Four SIMDs to a compute unit is CDNA's (the CDNA3 ISA guide).
    g.simd_per_cu = 4;
    g.simd_count = d.compute_units * g.simd_per_cu;
    g.wave_front_size = d.wave_size;
    g.max_waves_per_cu = d.wave_size ? d.max_threads_per_cu / d.wave_size : 0;
    g.max_waves_per_simd = g.max_waves_per_cu / g.simd_per_cu;
    g.lds_size_in_kb = d.lds_bytes / 1024;
    g.gfx_target_version = gfx_target_version(d.gfx);
    g.vendor_id = d.vendor_id;
    g.device_id = d.device_id;
    g.location_id = d.pci_bus << 8;
    g.drm_render_minor = 128 + static_cast<uint32_t>(i);
    g.max_engine_clk_fcompute = d.clock_mhz;
    g.workgroup_max_size = d.max_workgroup;
    g.grid_max_size = d.max_grid[0];
    g.local_mem_size = d.vram_bytes;
    // KFD's gpu_id is a 16-bit hash that tells one GPU from another; the
    // device's UUID folded to 16 bits does the same.
    for (int b = 0; b < 16; b += 2) g.gpu_id ^= static_cast<uint64_t>(d.uuid[b]) << 8 | d.uuid[b + 1];
    g.workgroup_max_dim = {d.max_block[0], d.max_block[1], d.max_block[2]};
    g.grid_max_dim = {d.max_grid[0], d.max_grid[1], d.max_grid[2]};
    g.name = keep(d.gfx);
    g.vendor_name = keep("AMD");
    g.product_name = keep(d.model);
    g.model_name = keep("");
    g.runtime_visibility.hsa = 1;
    g.runtime_visibility.hip = 1;
    std::memcpy(g.uuid.bytes, d.uuid, sizeof g.uuid.bytes);
    a.list.push_back(g);
    a.ordinal.push_back(i);
  }
  for (size_t i = 0; i < a.list.size(); ++i) {
    a.list[i].node_id = static_cast<uint32_t>(i);
    a.list[i].logical_node_id = static_cast<int32_t>(i);
    a.list[i].logical_node_type_id = a.ordinal[i] < 0 ? 0 : a.ordinal[i];
    // An agent's id is its node, from 1: zero is no agent.
    a.list[i].id.handle = i + 1;
  }
}

rocprofiler_agent_id_t gpu_agent(Sdk& s, int ordinal) {
  for (size_t i = 0; i < s.agents.ordinal.size(); ++i)
    if (s.agents.ordinal[i] == ordinal) return s.agents.list[i].id;
  return {0};
}
rocprofiler_agent_id_t cpu_agent(Sdk& s) { return gpu_agent(s, -1); }
const rocprofiler_agent_v0_t* agent(Sdk& s, uint64_t id) {
  for (const auto& a : s.agents.list)
    if (a.id.handle == id) return &a;
  return nullptr;
}

// ---- Delivering records ---------------------------------------------------------

void flush(Sdk& s, Buffer& b) {
  if (b.headers.empty()) return;
  std::vector<rocprofiler_record_header_t> headers;
  std::vector<std::unique_ptr<uint8_t[]>> payloads;
  headers.swap(b.headers);
  payloads.swap(b.payloads);
  b.held = 0;
  std::vector<rocprofiler_record_header_t*> list;
  for (auto& h : headers) list.push_back(&h);
  if (b.callback) b.callback({b.context}, {b.id}, list.data(), list.size(), b.data, 0);
  (void)s;
}

template <class Record>
void push(Sdk& s, uint64_t buffer_id, uint32_t category, uint32_t kind, const Record& r) {
  Buffer* b = buffer(s, buffer_id);
  if (!b) return;
  auto payload = std::make_unique<uint8_t[]>(sizeof(Record));
  std::memcpy(payload.get(), &r, sizeof(Record));
  rocprofiler_record_header_t h{};
  h.category = category;
  h.kind = kind;
  h.payload = payload.get();
  b->headers.push_back(h);
  b->payloads.push_back(std::move(payload));
  b->held += sizeof(Record);
  if (b->watermark && b->held >= b->watermark) flush(s, *b);
}

bool wants(const std::set<int>& operations, int op) { return operations.empty() || operations.count(op); }

// The buffers of every active context that asked for records of this kind
// and operation.
template <class Each>
void for_buffered(Sdk& s, int kind, int op, Each&& each) {
  for (Context& c : s.contexts) {
    if (!c.active) continue;
    const auto it = c.buffered.find(kind);
    if (it != c.buffered.end() && wants(it->second.operations, op)) each(c, it->second.buffer);
  }
}

template <class Each>
void for_callbacks(Sdk& s, int kind, int op, Each&& each) {
  for (Context& c : s.contexts) {
    if (!c.active) continue;
    const auto it = c.callbacks.find(kind);
    if (it != c.callbacks.end() && it->second.callback && wants(it->second.operations, op)) each(c, it->second);
  }
}

// What a context's tool attaches to a correlation id of this kind, if it
// asked to.
rocprofiler_user_data_t external_id(Context& c, rocprofiler_external_correlation_id_request_kind_t kind, int op,
                                    uint64_t internal) {
  rocprofiler_user_data_t v{};
  if (c.external && c.external_kinds.count(kind))
    c.external(thread_id(), {c.id}, kind, op, internal, &v, c.external_data);
  return v;
}

// ---- The HIP call a thread is in ------------------------------------------------

struct ApiFrame {
  uint64_t correlation = 0;
  int op = -1;
  uint64_t start = 0;
};
std::vector<ApiFrame>& frames() {
  thread_local std::vector<ApiFrame> f;
  return f;
}
uint64_t current_correlation(Sdk& s) {
  return frames().empty() ? s.next_correlation.fetch_add(1) : frames().back().correlation;
}

// ---- Registration -------------------------------------------------------------------

void finalize_client(Client& c) {
  if (c.finalized) return;
  c.finalized = true;
  if (c.result && c.result->finalize) c.result->finalize(c.result->tool_data);
}

void finalize_all() {
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  if (s.finalized) return;
  for (Client& c : s.clients) finalize_client(c);
  s.finalized = true;
  for (Context& c : s.contexts) c.active = false;
}

void client_finalize(rocprofiler_client_id_t id) {
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  for (Client& c : s.clients)
    if (c.id && c.id->handle == id.handle) finalize_client(c);
}

// Finds every tool -- the one handed in, the ones ROCP_TOOL_LIBRARIES names,
// and any rocprofiler_configure already loaded -- configures each, and then
// initializes them, as rocprofiler-sdk does.
void configure(rocprofiler_configure_func_t forced) {
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  if (s.configured) return;
  s.configured = true;
  std::vector<rocprofiler_configure_func_t> found;
  auto add = [&](void* f) {
    auto fn = reinterpret_cast<rocprofiler_configure_func_t>(f);
    if (fn && std::find(found.begin(), found.end(), fn) == found.end()) found.push_back(fn);
  };
  add(reinterpret_cast<void*>(forced));
  if (const char* libs = std::getenv("ROCP_TOOL_LIBRARIES")) {
    std::string list = libs;
    size_t at = 0;
    while (at <= list.size()) {
      const size_t colon = list.find(':', at);
      const std::string path = list.substr(at, colon == std::string::npos ? std::string::npos : colon - at);
      if (!path.empty()) {
        void* h = ::dlopen(path.c_str(), RTLD_NOLOAD | RTLD_LAZY);
        if (!h) h = ::dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (h) add(::dlsym(h, "rocprofiler_configure"));
      }
      if (colon == std::string::npos) break;
      at = colon + 1;
    }
  }
  add(::dlsym(RTLD_DEFAULT, "rocprofiler_configure"));

  load_agents(s);
  char version[64];
  std::snprintf(version, sizeof version, "%u.%u.%u (VirtualGPU)", kVersionMajor, kVersionMinor, kVersionPatch);
  for (size_t i = 0; i < found.size(); ++i) {
    auto* id = new rocprofiler_client_id_t{sizeof(rocprofiler_client_id_t), nullptr, static_cast<uint32_t>(i + 1)};
    rocprofiler_tool_configure_result_t* r = found[i](kVersion, version, static_cast<uint32_t>(i), id);
    if (r) s.clients.push_back(Client{id, r, false});
  }
  s.initialized = true;
  for (Client& c : s.clients)
    if (c.result->initialize && c.result->initialize(&client_finalize, c.result->tool_data) != 0)
      c.finalized = true;   // a tool that failed to start is not finalized
  std::atexit(finalize_all);
}

// rocprofv3 asks for tools to be found as the library loads; a HIP program
// has made its first call by the time any tool could be told of it.
bool configures_itself() {
  const char* ctor = std::getenv("ROCPROFILER_LIBRARY_CTOR");
  const char* libs = std::getenv("ROCP_TOOL_LIBRARIES");
  return (ctor && *ctor && std::strcmp(ctor, "0") != 0) || (libs && *libs);
}

void ensure_configured() {
  Sdk& s = sdk();
  if (!s.configured && configures_itself()) configure(nullptr);
}

// ---- What libamdhip64 reports -------------------------------------------------------

uint64_t on_api_enter(const char* name) {
  ensure_configured();
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  if (s.finalized) return 0;
  ApiFrame f;
  f.correlation = s.next_correlation.fetch_add(1);
  f.op = hip_api_id(name);
  f.start = hp::now_ns();
  frames().push_back(f);
  return frames().size();
}

void on_api_exit(uint64_t token) {
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  if (!token || frames().size() != token) return;
  const ApiFrame f = frames().back();
  frames().pop_back();
  if (s.finalized || f.op < 0) return;
  const uint64_t end = hp::now_ns();
  for (int kind : {ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT}) {
    for_buffered(s, kind, f.op, [&](Context& c, uint64_t buf) {
      rocprofiler_correlation_id_t corr{};
      corr.internal = f.correlation;
      corr.external = external_id(c, ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_RUNTIME_API, f.op, f.correlation);
      if (kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT) {
        rocprofiler_buffer_tracing_hip_api_ext_record_t r{};
        r.size = sizeof r;
        r.kind = static_cast<rocprofiler_buffer_tracing_kind_t>(kind);
        r.operation = f.op;
        r.correlation_id = corr;
        r.start_timestamp = f.start;
        r.end_timestamp = end;
        r.thread_id = thread_id();
        push(s, buf, ROCPROFILER_BUFFER_CATEGORY_TRACING, static_cast<uint32_t>(kind), r);
      } else {
        rocprofiler_buffer_tracing_hip_api_record_t r{};
        r.size = sizeof r;
        r.kind = static_cast<rocprofiler_buffer_tracing_kind_t>(kind);
        r.operation = f.op;
        r.correlation_id = corr;
        r.start_timestamp = f.start;
        r.end_timestamp = end;
        r.thread_id = thread_id();
        push(s, buf, ROCPROFILER_BUFFER_CATEGORY_TRACING, static_cast<uint32_t>(kind), r);
      }
    });
  }
}

void on_code_object(const hp::CodeObject& o, const hp::KernelSymbol* kernels, size_t count) {
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  if (s.finalized) return;
  rocprofiler_callback_tracing_code_object_load_data_t load{};
  load.size = sizeof load;
  load.code_object_id = o.id;
  load.agent_id = gpu_agent(s, o.device);
  load.uri = o.uri;
  load.load_base = o.load_base;
  load.load_size = o.load_size;
  load.storage_type = ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY;
  load.memory_base = reinterpret_cast<uint64_t>(o.image);
  load.memory_size = o.image_size;
  for_callbacks(s, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, ROCPROFILER_CODE_OBJECT_LOAD,
                [&](Context& c, CallbackService& svc) {
                  rocprofiler_callback_tracing_record_t r{};
                  r.context_id = {c.id};
                  r.thread_id = thread_id();
                  r.kind = ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT;
                  r.operation = ROCPROFILER_CODE_OBJECT_LOAD;
                  r.phase = ROCPROFILER_CALLBACK_PHASE_LOAD;
                  r.payload = &load;
                  rocprofiler_user_data_t user{};
                  svc.callback(r, &user, svc.data);
                });
  for (size_t i = 0; i < count; ++i) {
    const hp::KernelSymbol& k = kernels[i];
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t sym{};
    sym.size = sizeof sym;
    sym.kernel_id = k.kernel_id;
    sym.code_object_id = o.id;
    sym.kernel_name = k.name;
    sym.kernel_object = k.entry;
    sym.kernarg_segment_size = k.kernarg_size;
    sym.kernarg_segment_alignment = k.kernarg_alignment;
    sym.group_segment_size = k.group_segment;
    sym.private_segment_size = k.private_segment;
    sym.sgpr_count = k.sgprs;
    sym.arch_vgpr_count = k.vgprs;
    sym.accum_vgpr_count = k.agprs;
    sym.kernel_address.value = k.entry;
    for_callbacks(s, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER,
                  [&](Context& c, CallbackService& svc) {
                    rocprofiler_callback_tracing_record_t r{};
                    r.context_id = {c.id};
                    r.thread_id = thread_id();
                    r.kind = ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT;
                    r.operation = ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER;
                    r.phase = ROCPROFILER_CALLBACK_PHASE_LOAD;
                    r.payload = &sym;
                    rocprofiler_user_data_t user{};
                    svc.callback(r, &user, svc.data);
                  });
  }
}

// A launch in flight: the dispatch as a tool is shown it, and which contexts
// asked to count it, with what.
struct Counting {
  uint64_t context = 0;
  rocprofiler_counter_config_id_t config{};
  rocprofiler_user_data_t user{};
};
struct InFlight {
  uint64_t correlation = 0;
  rocprofiler_kernel_dispatch_info_t info{};
  std::vector<Counting> counting;
  // What each context's tool attached to the launch's correlation id, asked
  // once: a tool may allocate what it attaches.
  std::map<uint64_t, rocprofiler_user_data_t> external;
};

void* on_launching(const hp::Launch& l) {
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  if (s.finalized) return nullptr;
  auto* f = new InFlight;
  f->correlation = current_correlation(s);
  rocprofiler_kernel_dispatch_info_t& i = f->info;
  i.size = offsetof(rocprofiler_kernel_dispatch_info_t, reserved_padding);
  i.agent_id = gpu_agent(s, l.device);
  i.queue_id = {l.stream};
  i.kernel_id = l.kernel_id;
  i.dispatch_id = l.dispatch_id;
  i.private_segment_size = l.private_segment;
  i.group_segment_size = l.group_segment;
  i.workgroup_size = {l.group_size[0], l.group_size[1], l.group_size[2]};
  // The grid in work-items, as an HSA dispatch packet gives it.
  i.grid_size = {l.groups[0] * l.group_size[0], l.groups[1] * l.group_size[1], l.groups[2] * l.group_size[2]};
  for (Context& c : s.contexts)
    if (c.active)
      f->external[c.id] = external_id(c, ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH, 0, f->correlation);
  for (Context& c : s.contexts) {
    if (!c.active || !c.dispatch) continue;
    Counting k;
    k.context = c.id;
    rocprofiler_dispatch_counting_service_data_t data{};
    data.size = sizeof data;
    data.correlation_id = {f->correlation, f->external[c.id]};
    data.dispatch_info = f->info;
    c.dispatch(data, &k.config, &k.user, c.dispatch_data);
    if (k.config.handle && s.configs.count(k.config.handle)) f->counting.push_back(k);
  }
  return f;
}

void on_launched(const hp::Launch&, const DispatchStats* stats, uint64_t start, uint64_t end, void* token) {
  Sdk& s = sdk();
  std::unique_ptr<InFlight> f(static_cast<InFlight*>(token));
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  if (!f || s.finalized || !stats) return;   // a launch that failed counted nothing
  for (const Counting& k : f->counting) {
    Context* c = context(s, {k.context});
    if (!c || !c->record) continue;
    const Config& cfg = s.configs.at(k.config.handle);
    std::vector<rocprofiler_counter_record_t> records;
    for (uint64_t id : cfg.counters) {
      rocprofiler_counter_record_t r{};
      r.id = instance_id(id);
      r.counter_value = static_cast<double>(counter(id)->value(*stats));
      r.dispatch_id = f->info.dispatch_id;
      r.user_data = k.user;
      r.agent_id = f->info.agent_id;
      records.push_back(r);
    }
    rocprofiler_dispatch_counting_service_data_t data{};
    data.size = sizeof data;
    data.correlation_id = {f->correlation, f->external[k.context]};
    data.start_timestamp = start;
    data.end_timestamp = end;
    data.dispatch_info = f->info;
    c->record(data, records.data(), records.size(), k.user, c->record_data);
  }
  for_buffered(s, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
               [&](Context& c, uint64_t buf) {
                 rocprofiler_buffer_tracing_kernel_dispatch_record_t r{};
                 r.size = sizeof r;
                 r.kind = ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH;
                 r.operation = ROCPROFILER_KERNEL_DISPATCH_COMPLETE;
                 r.correlation_id = {f->correlation, f->external[c.id]};
                 r.thread_id = thread_id();
                 r.start_timestamp = start;
                 r.end_timestamp = end;
                 r.dispatch_info = f->info;
                 push(s, buf, ROCPROFILER_BUFFER_CATEGORY_TRACING, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, r);
               });
}

void on_copied(hp::Copy kind, int src_device, int dst_device, uint64_t bytes, uint64_t dst, uint64_t src,
               uint64_t start, uint64_t end) {
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  if (s.finalized) return;
  const int op = static_cast<int>(kind);
  const bool from_device = kind == hp::Copy::DeviceToHost || kind == hp::Copy::DeviceToDevice;
  const bool to_device = kind == hp::Copy::HostToDevice || kind == hp::Copy::DeviceToDevice;
  const uint64_t corr = current_correlation(s);
  for_buffered(s, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, op, [&](Context& c, uint64_t buf) {
    rocprofiler_buffer_tracing_memory_copy_record_t r{};
    r.size = sizeof r;
    r.kind = ROCPROFILER_BUFFER_TRACING_MEMORY_COPY;
    r.operation = static_cast<rocprofiler_memory_copy_operation_t>(op);
    r.correlation_id = {corr, external_id(c, ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY, op, corr)};
    r.thread_id = thread_id();
    r.start_timestamp = start;
    r.end_timestamp = end;
    r.dst_agent_id = to_device ? gpu_agent(s, dst_device) : cpu_agent(s);
    r.src_agent_id = from_device ? gpu_agent(s, src_device) : cpu_agent(s);
    r.bytes = bytes;
    r.dst_address.value = dst;
    r.src_address.value = src;
    push(s, buf, ROCPROFILER_BUFFER_CATEGORY_TRACING, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, r);
  });
}

void on_allocated(int device, uint64_t address, uint64_t bytes, bool freed, uint64_t start, uint64_t end) {
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  if (s.finalized) return;
  const int op = freed ? ROCPROFILER_MEMORY_ALLOCATION_FREE : ROCPROFILER_MEMORY_ALLOCATION_ALLOCATE;
  const uint64_t corr = current_correlation(s);
  for_buffered(s, ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION, op, [&](Context& c, uint64_t buf) {
    rocprofiler_buffer_tracing_memory_allocation_record_t r{};
    r.size = sizeof r;
    r.kind = ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION;
    r.operation = static_cast<rocprofiler_memory_allocation_operation_t>(op);
    r.correlation_id.internal = corr;
    r.correlation_id.external =
        external_id(c, ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_ALLOCATION, op, corr);
    r.thread_id = thread_id();
    r.start_timestamp = start;
    r.end_timestamp = end;
    r.agent_id = gpu_agent(s, device);
    r.address.value = address;
    r.allocation_size = bytes;
    push(s, buf, ROCPROFILER_BUFFER_CATEGORY_TRACING, ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION, r);
  });
}

const hp::Hooks kHooks = [] {
  hp::Hooks h;
  h.api_enter = on_api_enter;
  h.api_exit = on_api_exit;
  h.code_object_loaded = on_code_object;
  h.launching = on_launching;
  h.launched = on_launched;
  h.copied = on_copied;
  h.allocated = on_allocated;
  return h;
}();

// Attached as the library loads, so the HIP runtime reports from its first
// call: that is when tools are found and configured (ensure_configured).
__attribute__((constructor)) void attach() { vgpu_hip_profiler_attach(&kHooks); }

}  // namespace

// ---- The interface ------------------------------------------------------------------

ROCPROFILER_API rocprofiler_status_t rocprofiler_is_initialized(int* status) {
  if (!status) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  *status = sdk().initialized ? 1 : 0;
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_is_finalized(int* status) {
  if (!status) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  *status = sdk().finalized ? 1 : 0;
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_force_configure(rocprofiler_configure_func_t configure_func) {
  if (sdk().configured) return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;
  configure(configure_func);
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_get_timestamp(rocprofiler_timestamp_t* ts) {
  if (!ts) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  *ts = hp::now_ns();
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API const char* rocprofiler_get_status_name(rocprofiler_status_t status) {
  static const char* const names[] = {
      "ROCPROFILER_STATUS_SUCCESS", "ROCPROFILER_STATUS_ERROR", "ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND",
      "ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND", "ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND",
      "ROCPROFILER_STATUS_ERROR_OPERATION_NOT_FOUND", "ROCPROFILER_STATUS_ERROR_THREAD_NOT_FOUND",
      "ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND", "ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND",
      "ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR", "ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID",
      "ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED", "ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT",
      "ROCPROFILER_STATUS_ERROR_CONTEXT_ID_NOT_ZERO", "ROCPROFILER_STATUS_ERROR_BUFFER_BUSY",
      "ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED", "ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED",
      "ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED", "ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI",
      "ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT", "ROCPROFILER_STATUS_ERROR_METRIC_NOT_VALID_FOR_AGENT",
      "ROCPROFILER_STATUS_ERROR_FINALIZED", "ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED",
      "ROCPROFILER_STATUS_ERROR_DIM_NOT_FOUND", "ROCPROFILER_STATUS_ERROR_PROFILE_COUNTER_NOT_FOUND",
      "ROCPROFILER_STATUS_ERROR_AST_GENERATION_FAILED", "ROCPROFILER_STATUS_ERROR_AST_NOT_FOUND",
      "ROCPROFILER_STATUS_ERROR_AQL_NO_EVENT_COORD", "ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL",
      "ROCPROFILER_STATUS_ERROR_OUT_OF_RESOURCES", "ROCPROFILER_STATUS_ERROR_PROFILE_NOT_FOUND",
      "ROCPROFILER_STATUS_ERROR_AGENT_DISPATCH_CONFLICT", "ROCPROFILER_STATUS_INTERNAL_NO_AGENT_CONTEXT",
      "ROCPROFILER_STATUS_ERROR_SAMPLE_RATE_EXCEEDED", "ROCPROFILER_STATUS_ERROR_NO_PROFILE_QUEUE",
      "ROCPROFILER_STATUS_ERROR_NO_HARDWARE_COUNTERS", "ROCPROFILER_STATUS_ERROR_AGENT_MISMATCH",
      "ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE", "ROCPROFILER_STATUS_ERROR_EXCEEDS_HW_LIMIT",
      "ROCPROFILER_STATUS_ERROR_AGENT_ARCH_NOT_SUPPORTED", "ROCPROFILER_STATUS_ERROR_PERMISSION_DENIED"};
  static_assert(sizeof names / sizeof names[0] == ROCPROFILER_STATUS_LAST);
  return status >= 0 && status < ROCPROFILER_STATUS_LAST ? names[status] : nullptr;
}

ROCPROFILER_API const char* rocprofiler_get_status_string(rocprofiler_status_t status) {
  switch (status) {
    case ROCPROFILER_STATUS_SUCCESS: return "Success";
    case ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE:
      return "The service is not available on this simulated GPU (VirtualGPU has no model of it)";
    case ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND: return "Counter identifier does not exist";
    case ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND: return "Agent identifier not found";
    case ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND: return "No valid context for given context id";
    case ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND: return "No valid buffer for given buffer id";
    case ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND: return "Kind identifier is invalid";
    case ROCPROFILER_STATUS_ERROR_OPERATION_NOT_FOUND: return "Operation identifier is invalid for domain";
    case ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT: return "Function invoked with one or more invalid arguments";
    case ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED: return "Configuration is locked";
    default: return rocprofiler_get_status_name(status);
  }
}

// -- agents and counters --

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_available_agents(rocprofiler_agent_version_t version,
                                                                        rocprofiler_query_available_agents_cb_t callback,
                                                                        size_t agent_size, void* user_data) {
  if (version != ROCPROFILER_AGENT_INFO_VERSION_0 || !callback) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  if (agent_size > sizeof(rocprofiler_agent_v0_t)) return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;
  Sdk& s = sdk();
  std::vector<const void*> list;
  {
    std::lock_guard<std::recursive_mutex> lock(s.mutex);
    load_agents(s);
    for (const auto& a : s.agents.list) list.push_back(&a);
  }
  return callback(version, list.data(), list.size(), user_data);
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_iterate_agent_supported_counters(
    rocprofiler_agent_id_t agent_id, rocprofiler_available_counters_cb_t cb, void* user_data) {
  if (!cb) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  Sdk& s = sdk();
  const rocprofiler_agent_v0_t* a = nullptr;
  {
    std::lock_guard<std::recursive_mutex> lock(s.mutex);
    load_agents(s);
    a = agent(s, agent_id.handle);
  }
  if (!a) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;
  std::vector<rocprofiler_counter_id_t> ids;
  if (a->type == ROCPROFILER_AGENT_TYPE_GPU)
    for (uint64_t i = 1; i <= kCounterCount; ++i) ids.push_back({i});
  return cb(agent_id, ids.data(), ids.size(), user_data);
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_counter_info(rocprofiler_counter_id_t counter_id,
                                                                    rocprofiler_counter_info_version_id_t version,
                                                                    void* info) {
  const Counter* c = counter(counter_id.handle);
  if (!c) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
  if (!info) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  const bool derived = *c->expression != '\0';
  if (version == ROCPROFILER_COUNTER_INFO_VERSION_0) {
    auto* v = static_cast<rocprofiler_counter_info_v0_t*>(info);
    *v = {};
    v->id = counter_id;
    v->name = c->name;
    v->description = c->description;
    v->block = c->block;
    v->expression = c->expression;
    v->is_derived = derived;
    return ROCPROFILER_STATUS_SUCCESS;
  }
  if (version == ROCPROFILER_COUNTER_INFO_VERSION_1) {
    // Each counter's single instance, kept for as long as the library is.
    static std::vector<rocprofiler_counter_record_dimension_instance_info_t> instances = [] {
      std::vector<rocprofiler_counter_record_dimension_instance_info_t> v(kCounterCount);
      for (uint64_t i = 1; i <= kCounterCount; ++i) {
        auto& in = v[i - 1];
        in.size = sizeof in;
        in.instance_id = instance_id(i);
        in.counter_id = i;
        in.dimensions_count = 1;
        in.dimensions = const_cast<const rocprofiler_counter_dimension_info_t**>(kInstanceDimensions);
      }
      return v;
    }();
    static std::vector<const rocprofiler_counter_record_dimension_instance_info_t*> instance_ptrs = [] {
      std::vector<const rocprofiler_counter_record_dimension_instance_info_t*> v;
      for (auto& in : instances) v.push_back(&in);
      return v;
    }();
    auto* v = static_cast<rocprofiler_counter_info_v1_t*>(info);
    *v = {};
    v->size = sizeof *v;
    v->id = counter_id;
    v->name = c->name;
    v->description = c->description;
    v->block = c->block;
    v->expression = c->expression;
    v->is_derived = derived;
    v->dimensions_count = 1;
    v->dimensions = const_cast<const rocprofiler_counter_record_dimension_info_t**>(kDimensions);
    v->dimensions_instances_count = 1;
    v->dimensions_instances = &instance_ptrs[counter_id.handle - 1];
    return ROCPROFILER_STATUS_SUCCESS;
  }
  return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_record_counter_id(rocprofiler_counter_instance_id_t id,
                                                                         rocprofiler_counter_id_t* counter_id) {
  if (!counter_id) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  if (!counter(id >> 32)) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
  counter_id->handle = id >> 32;
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_record_dimension_position(rocprofiler_counter_instance_id_t id,
                                                                                 rocprofiler_counter_dimension_id_t dim,
                                                                                 size_t* pos) {
  if (!pos) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  if (!counter(id >> 32)) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
  if (dim != kDimension.id) return ROCPROFILER_STATUS_ERROR_DIM_NOT_FOUND;
  *pos = id & 0xFFFFFFFFu;
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_create_counter_config(rocprofiler_agent_id_t agent_id,
                                                                       rocprofiler_counter_id_t* counters_list,
                                                                       size_t counters_count,
                                                                       rocprofiler_counter_config_id_t* config_id) {
  if (!config_id || (counters_count && !counters_list)) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  load_agents(s);
  const rocprofiler_agent_v0_t* a = agent(s, agent_id.handle);
  if (!a || a->type != ROCPROFILER_AGENT_TYPE_GPU) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;
  if (!counters_count) return ROCPROFILER_STATUS_ERROR_NO_HARDWARE_COUNTERS;
  Config cfg;
  cfg.agent = agent_id.handle;
  for (size_t i = 0; i < counters_count; ++i) {
    if (!counter(counters_list[i].handle)) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
    cfg.counters.push_back(counters_list[i].handle);
  }
  config_id->handle = s.next_config++;
  s.configs.emplace(config_id->handle, std::move(cfg));
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_destroy_counter_config(rocprofiler_counter_config_id_t config_id) {
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  return s.configs.erase(config_id.handle) ? ROCPROFILER_STATUS_SUCCESS : ROCPROFILER_STATUS_ERROR_PROFILE_NOT_FOUND;
}

// Counters are the ones the interpreter counts; a definition added by a tool
// would name hardware events that are not simulated.
ROCPROFILER_API rocprofiler_status_t rocprofiler_load_counter_definition(const char*, size_t, int) {
  return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
}

// -- contexts and services --

ROCPROFILER_API rocprofiler_status_t rocprofiler_create_context(rocprofiler_context_id_t* context_id) {
  if (!context_id) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  if (s.finalized) return ROCPROFILER_STATUS_ERROR_FINALIZED;
  Context c;
  c.id = s.contexts.size() + 1;
  s.contexts.push_back(c);
  context_id->handle = c.id;
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_start_context(rocprofiler_context_id_t context_id) {
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  Context* c = context(s, context_id);
  if (!c) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
  c->active = !s.finalized;
  return s.finalized ? ROCPROFILER_STATUS_ERROR_FINALIZED : ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_stop_context(rocprofiler_context_id_t context_id) {
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  Context* c = context(s, context_id);
  if (!c) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
  c->active = false;
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_context_is_active(rocprofiler_context_id_t context_id, int* status) {
  if (!status) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  Context* c = context(s, context_id);
  if (!c) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
  *status = c->active ? 1 : 0;
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_context_is_valid(rocprofiler_context_id_t context_id, int* status) {
  if (!status) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  *status = context(s, context_id) ? 1 : 0;
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_configure_callback_tracing_service(
    rocprofiler_context_id_t context_id, rocprofiler_callback_tracing_kind_t kind,
    const rocprofiler_tracing_operation_t* operations, size_t operations_count, rocprofiler_callback_tracing_cb_t callback,
    void* callback_args) {
  if (kind <= ROCPROFILER_CALLBACK_TRACING_NONE || kind >= ROCPROFILER_CALLBACK_TRACING_LAST)
    return ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND;
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  Context* c = context(s, context_id);
  if (!c) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
  if (c->callbacks.count(kind)) return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED;
  CallbackService svc;
  svc.callback = callback;
  svc.data = callback_args;
  for (size_t i = 0; i < operations_count; ++i) svc.operations.insert(operations[i]);
  c->callbacks.emplace(kind, svc);
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_configure_buffer_tracing_service(
    rocprofiler_context_id_t context_id, rocprofiler_buffer_tracing_kind_t kind,
    const rocprofiler_tracing_operation_t* operations, size_t operations_count, rocprofiler_buffer_id_t buffer_id) {
  if (kind <= ROCPROFILER_BUFFER_TRACING_NONE || kind >= ROCPROFILER_BUFFER_TRACING_LAST)
    return ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND;
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  Context* c = context(s, context_id);
  if (!c) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
  if (!buffer(s, buffer_id.handle)) return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;
  if (c->buffered.count(kind)) return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED;
  BufferService svc;
  svc.buffer = buffer_id.handle;
  for (size_t i = 0; i < operations_count; ++i) svc.operations.insert(operations[i]);
  c->buffered.emplace(kind, svc);
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_configure_callback_dispatch_counting_service(
    rocprofiler_context_id_t context_id, rocprofiler_dispatch_counting_service_cb_t dispatch_callback,
    void* dispatch_callback_args, rocprofiler_dispatch_counting_record_cb_t record_callback,
    void* record_callback_args) {
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  Context* c = context(s, context_id);
  if (!c) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
  if (c->dispatch) return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED;
  c->dispatch = dispatch_callback;
  c->dispatch_data = dispatch_callback_args;
  c->record = record_callback;
  c->record_data = record_callback_args;
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_configure_external_correlation_id_request_service(
    rocprofiler_context_id_t context_id, const rocprofiler_external_correlation_id_request_kind_t* kinds,
    size_t kinds_count, rocprofiler_external_correlation_id_request_cb_t callback, void* callback_args) {
  if (!callback) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  Context* c = context(s, context_id);
  if (!c) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
  if (c->external) return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED;
  c->external = callback;
  c->external_data = callback_args;
  if (kinds_count == 0)
    for (int k = 1; k < ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_LAST; ++k) c->external_kinds.insert(k);
  for (size_t i = 0; i < kinds_count; ++i) c->external_kinds.insert(kinds[i]);
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_external_correlation_id_request_kind_name(
    rocprofiler_external_correlation_id_request_kind_t kind, const char** name, uint64_t* name_len) {
  static const char* const names[] = {
      "NONE", "HSA_CORE_API", "HSA_AMD_EXT_API", "HSA_IMAGE_EXT_API", "HSA_FINALIZE_EXT_API", "HIP_RUNTIME_API",
      "HIP_COMPILER_API", "MARKER_CORE_API", "MARKER_CONTROL_API", "MARKER_NAME_API", "MEMORY_COPY",
      "KERNEL_DISPATCH", "SCRATCH_MEMORY", "RCCL_API", "OMPT", "MEMORY_ALLOCATION", "ROCDECODE_API", "ROCJPEG_API",
      "MARKER_CORE_RANGE_API"};
  static_assert(sizeof names / sizeof names[0] == ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_LAST);
  if (kind < 0 || kind >= ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_LAST) return ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND;
  if (name) *name = names[kind];
  if (name_len) *name_len = std::strlen(names[kind]);
  return ROCPROFILER_STATUS_SUCCESS;
}

// -- buffers and the threads that deliver them --

ROCPROFILER_API rocprofiler_status_t rocprofiler_create_buffer(rocprofiler_context_id_t context_id, size_t size,
                                                               size_t watermark, rocprofiler_buffer_policy_t,
                                                               rocprofiler_buffer_tracing_cb_t callback,
                                                               void* callback_data, rocprofiler_buffer_id_t* buffer_id) {
  if (!callback || !buffer_id) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  if (!context(s, context_id)) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
  Buffer b;
  b.id = s.buffers.size() + 1;
  b.context = context_id.handle;
  b.watermark = watermark ? watermark : size;
  b.callback = callback;
  b.data = callback_data;
  s.buffers.push_back(std::move(b));
  buffer_id->handle = s.buffers.back().id;
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_flush_buffer(rocprofiler_buffer_id_t buffer_id) {
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  Buffer* b = buffer(s, buffer_id.handle);
  if (!b) return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;
  flush(s, *b);
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_destroy_buffer(rocprofiler_buffer_id_t buffer_id) {
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  Buffer* b = buffer(s, buffer_id.handle);
  if (!b) return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;
  flush(s, *b);
  b->callback = nullptr;
  return ROCPROFILER_STATUS_SUCCESS;
}

// Records are delivered on the thread that produced them, or that flushed
// them; a tool that asks for a thread of its own to receive them gets one to
// name, which that is.
ROCPROFILER_API rocprofiler_status_t rocprofiler_create_callback_thread(rocprofiler_callback_thread_t* cb_thread_id) {
  if (!cb_thread_id) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  cb_thread_id->handle = s.next_callback_thread++;
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_assign_callback_thread(rocprofiler_buffer_id_t buffer_id,
                                                                        rocprofiler_callback_thread_t) {
  Sdk& s = sdk();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  return buffer(s, buffer_id.handle) ? ROCPROFILER_STATUS_SUCCESS : ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_at_internal_thread_create(void (*)(rocprofiler_runtime_library_t, void*),
                                                                           void (*)(rocprofiler_runtime_library_t, void*),
                                                                           int, void*) {
  return ROCPROFILER_STATUS_SUCCESS;   // this library starts no threads of its own
}

// -- names --

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_buffer_tracing_kind_name(rocprofiler_buffer_tracing_kind_t kind,
                                                                                const char** name, uint64_t* name_len) {
  if (kind < 0 || kind >= ROCPROFILER_BUFFER_TRACING_LAST) return ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND;
  if (name) *name = kBufferKinds[kind];
  if (name_len) *name_len = std::strlen(kBufferKinds[kind]);
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_buffer_tracing_kind_operation_name(
    rocprofiler_buffer_tracing_kind_t kind, rocprofiler_tracing_operation_t operation, const char** name,
    uint64_t* name_len) {
  if (kind < 0 || kind >= ROCPROFILER_BUFFER_TRACING_LAST) return ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND;
  const char* n = operation_name(kBufferKinds[kind], operation);
  if (!n) return ROCPROFILER_STATUS_ERROR_OPERATION_NOT_FOUND;
  if (name) *name = n;
  if (name_len) *name_len = std::strlen(n);
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_iterate_buffer_tracing_kinds(rocprofiler_buffer_tracing_kind_cb_t callback,
                                                                              void* data) {
  if (!callback) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  for (int k = 1; k < ROCPROFILER_BUFFER_TRACING_LAST; ++k)
    if (callback(static_cast<rocprofiler_buffer_tracing_kind_t>(k), data) != 0) break;
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_iterate_buffer_tracing_kind_operations(
    rocprofiler_buffer_tracing_kind_t kind, rocprofiler_buffer_tracing_kind_operation_cb_t callback, void* data) {
  if (!callback) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  if (kind <= 0 || kind >= ROCPROFILER_BUFFER_TRACING_LAST) return ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND;
  return iterate_operations(kBufferKinds[kind], [&](int op) { return callback(kind, op, data); });
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_callback_tracing_kind_name(
    rocprofiler_callback_tracing_kind_t kind, const char** name, uint64_t* name_len) {
  if (kind < 0 || kind >= ROCPROFILER_CALLBACK_TRACING_LAST) return ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND;
  if (name) *name = kCallbackKinds[kind];
  if (name_len) *name_len = std::strlen(kCallbackKinds[kind]);
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_callback_tracing_kind_operation_name(
    rocprofiler_callback_tracing_kind_t kind, rocprofiler_tracing_operation_t operation, const char** name,
    uint64_t* name_len) {
  if (kind < 0 || kind >= ROCPROFILER_CALLBACK_TRACING_LAST) return ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND;
  const char* n = operation_name(kCallbackKinds[kind], operation);
  if (!n) return ROCPROFILER_STATUS_ERROR_OPERATION_NOT_FOUND;
  if (name) *name = n;
  if (name_len) *name_len = std::strlen(n);
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_iterate_callback_tracing_kinds(
    rocprofiler_callback_tracing_kind_cb_t callback, void* data) {
  if (!callback) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  for (int k = 1; k < ROCPROFILER_CALLBACK_TRACING_LAST; ++k)
    if (callback(static_cast<rocprofiler_callback_tracing_kind_t>(k), data) != 0) break;
  return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_iterate_callback_tracing_kind_operations(
    rocprofiler_callback_tracing_kind_t kind, rocprofiler_callback_tracing_kind_operation_cb_t callback, void* data) {
  if (!callback) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
  if (kind <= 0 || kind >= ROCPROFILER_CALLBACK_TRACING_LAST) return ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND;
  return iterate_operations(kCallbackKinds[kind], [&](int op) { return callback(kind, op, data); });
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_intercept_table_name(rocprofiler_intercept_table_t kind,
                                                                            const char** name, uint64_t* name_len) {
  const char* n = kind == ROCPROFILER_HSA_TABLE              ? "HSA"
                  : kind == ROCPROFILER_HIP_RUNTIME_TABLE    ? "HIP_RUNTIME"
                  : kind == ROCPROFILER_HIP_COMPILER_TABLE   ? "HIP_COMPILER"
                  : kind == ROCPROFILER_MARKER_CORE_TABLE    ? "MARKER_CORE"
                  : kind == ROCPROFILER_MARKER_CONTROL_TABLE ? "MARKER_CONTROL"
                  : kind == ROCPROFILER_MARKER_NAME_TABLE    ? "MARKER_NAME"
                  : kind == ROCPROFILER_RCCL_TABLE           ? "RCCL"
                  : kind == ROCPROFILER_ROCDECODE_TABLE      ? "ROCDECODE"
                  : kind == ROCPROFILER_ROCJPEG_TABLE        ? "ROCJPEG"
                                                             : nullptr;
  if (!n) return ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND;
  if (name) *name = n;
  if (name_len) *name_len = std::strlen(n);
  return ROCPROFILER_STATUS_SUCCESS;
}

// A tool asks to be handed a runtime's dispatch table as it loads, to wrap its
// functions. The HIP runtime here has none to hand over -- it reports each
// call itself -- and there is no HSA runtime, so the request is taken and
// nothing follows from it.
ROCPROFILER_API rocprofiler_status_t rocprofiler_at_intercept_table_registration(rocprofiler_intercept_library_cb_t,
                                                                                 int, void*) {
  return ROCPROFILER_STATUS_SUCCESS;
}

// The arguments of a traced call, which are not recorded here.
ROCPROFILER_API rocprofiler_status_t rocprofiler_iterate_buffer_tracing_record_args(rocprofiler_record_header_t, void*,
                                                                                    void*) {
  return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
}

// -- what needs hardware there is no model of --

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_pc_sampling_agent_configurations(rocprofiler_agent_id_t, void*,
                                                                                        void*) {
  return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
}
ROCPROFILER_API rocprofiler_status_t rocprofiler_configure_pc_sampling_service(rocprofiler_context_id_t, rocprofiler_agent_id_t,
                                                                               int, int, uint64_t, rocprofiler_buffer_id_t,
                                                                               int) {
  return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
}
ROCPROFILER_API const char* rocprofiler_get_pc_sampling_instruction_not_issued_reason_name(int) { return nullptr; }
ROCPROFILER_API const char* rocprofiler_get_pc_sampling_instruction_type_name(int) { return nullptr; }
ROCPROFILER_API rocprofiler_status_t rocprofiler_configure_device_thread_trace_service(void*, ...) {
  return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
}
ROCPROFILER_API rocprofiler_status_t rocprofiler_configure_dispatch_thread_trace_service(void*, ...) {
  return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
}
ROCPROFILER_API rocprofiler_status_t rocprofiler_thread_trace_decoder_create(void*, ...) {
  return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
}
ROCPROFILER_API void rocprofiler_thread_trace_decoder_destroy(uint64_t) {}
ROCPROFILER_API rocprofiler_status_t rocprofiler_thread_trace_decoder_codeobj_load(uint64_t, ...) {
  return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
}
ROCPROFILER_API rocprofiler_status_t rocprofiler_thread_trace_decoder_codeobj_unload(uint64_t, ...) {
  return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
}
ROCPROFILER_API const char* rocprofiler_thread_trace_decoder_info_string(uint64_t, ...) { return nullptr; }
ROCPROFILER_API rocprofiler_status_t rocprofiler_trace_decode(uint64_t, ...) {
  return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
}
