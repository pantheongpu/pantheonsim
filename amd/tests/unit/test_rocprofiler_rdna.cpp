// VirtualGPU's librocprofiler-sdk on a Radeon card, from a profiling tool's side:
// what rocprofv3 recorded for a program on a real RX 6800 and a real RX 6700 XT
// (ROCm 7.2), checked without ROCm installed. amd/tests/e2e/run_rocprof_case.sh
// compares whole traces with the card's wherever rocprofv3 is; this is the part
// of it that hosted CI can run.
//
// VGPU_GPU names the card (amd/rx6800, or amd/rx6700xt with HSA_OVERRIDE_GFX_VERSION=10.3.0, which makes
// the HIP runtime see gfx1030 while the profiler's agent is still the gfx1031 card).
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "vgpu/rocprofiler_abi.hpp"
#include "vgpu_hip.h"
#include "vtest.hpp"

using namespace vgpu::amd::rocprof;

extern "C" {
// Pinned host memory (vgpu_hip.h declares no more of HIP than the other tests use).
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags);
hipError_t hipHostFree(void* ptr);
hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream);
hipError_t hipFreeAsync(void* ptr, hipStream_t stream);
rocprofiler_status_t rocprofiler_force_configure(rocprofiler_configure_func_t);
rocprofiler_status_t rocprofiler_query_available_agents(rocprofiler_agent_version_t,
                                                        rocprofiler_query_available_agents_cb_t, size_t, void*);
rocprofiler_status_t rocprofiler_iterate_agent_supported_counters(rocprofiler_agent_id_t, rocprofiler_available_counters_cb_t,
                                                                  void*);
rocprofiler_status_t rocprofiler_query_counter_info(rocprofiler_counter_id_t, rocprofiler_counter_info_version_id_t, void*);
rocprofiler_status_t rocprofiler_create_context(rocprofiler_context_id_t*);
rocprofiler_status_t rocprofiler_start_context(rocprofiler_context_id_t);
rocprofiler_status_t rocprofiler_configure_callback_tracing_service(rocprofiler_context_id_t,
                                                                    rocprofiler_callback_tracing_kind_t,
                                                                    const rocprofiler_tracing_operation_t*, size_t,
                                                                    rocprofiler_callback_tracing_cb_t, void*);
rocprofiler_status_t rocprofiler_create_buffer(rocprofiler_context_id_t, size_t, size_t, rocprofiler_buffer_policy_t,
                                               rocprofiler_buffer_tracing_cb_t, void*, rocprofiler_buffer_id_t*);
rocprofiler_status_t rocprofiler_configure_buffer_tracing_service(rocprofiler_context_id_t,
                                                                  rocprofiler_buffer_tracing_kind_t,
                                                                  const rocprofiler_tracing_operation_t*, size_t,
                                                                  rocprofiler_buffer_id_t);
rocprofiler_status_t rocprofiler_flush_buffer(rocprofiler_buffer_id_t);
int rocprofiler_set_api_table(const char*, uint64_t, uint64_t, void**, uint64_t);
rocprofiler_status_t rocprofiler_query_buffer_tracing_kind_operation_name(rocprofiler_buffer_tracing_kind_t,
                                                                          rocprofiler_tracing_operation_t,
                                                                          const char**, uint64_t*);
}

namespace {

const std::string kData = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/";

struct Call {
  std::string name;
  uint64_t id;
  uint64_t ancestor = 0;
};
// A ROCTx callback-tracing record as the tool saw it.
struct MarkerCallback {
  int kind, operation, phase;
  uint64_t id, ancestor;
  std::string message;
  const void* user;
};
struct Seen {
  std::vector<rocprofiler_agent_v0_t> agents;
  struct Kernel {
    std::string name;
    uint32_t sgprs, vgprs;
  };
  std::vector<Kernel> kernels;
  std::map<uint64_t, std::string> kernel_names;   // by kernel id
  struct Dispatch {
    std::string kernel;
    uint64_t grid, workgroup;
  };
  std::vector<Dispatch> dispatches;
  std::vector<std::string> uris;
  std::vector<rocprofiler_buffer_tracing_memory_copy_record_t> copies;
  std::vector<rocprofiler_buffer_tracing_memory_allocation_record_t> allocations;
  std::vector<Call> calls;
  std::vector<MarkerCallback> marker_callbacks;
  std::vector<rocprofiler_buffer_tracing_marker_api_record_t> ranges;
  rocprofiler_buffer_id_t buffer{};
};
Seen& seen() {
  static Seen s;
  return s;
}

void on_code_object(rocprofiler_callback_tracing_record_t r, rocprofiler_user_data_t*, void*) {
  if (r.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
  if (r.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    seen().uris.push_back(static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(r.payload)->uri);
  if (r.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) {
    auto* k = static_cast<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(r.payload);
    seen().kernels.push_back({k->kernel_name, k->sgpr_count, k->arch_vgpr_count});
    seen().kernel_names[k->kernel_id] = k->kernel_name;
  }
}

void on_records(rocprofiler_context_id_t, rocprofiler_buffer_id_t, rocprofiler_record_header_t** headers, size_t n,
                void*, uint64_t) {
  for (size_t i = 0; i < n; ++i) {
    const rocprofiler_record_header_t* h = headers[i];
    if (h->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) continue;
    if (h->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
      seen().copies.push_back(*static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(h->payload));
    if (h->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {
      auto* r = static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(h->payload);
      seen().dispatches.push_back({seen().kernel_names[r->dispatch_info.kernel_id], r->dispatch_info.grid_size.x,
                                   r->dispatch_info.workgroup_size.x});
    }
    if (h->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION)
      seen().allocations.push_back(*static_cast<rocprofiler_buffer_tracing_memory_allocation_record_t*>(h->payload));
    if (h->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT) {
      auto* r = static_cast<rocprofiler_buffer_tracing_hip_api_ext_record_t*>(h->payload);
      const char* name = nullptr;
      rocprofiler_query_buffer_tracing_kind_operation_name(ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT, r->operation,
                                                           &name, nullptr);
      seen().calls.push_back({name ? name : "?", r->correlation_id.internal, r->correlation_id.ancestor});
    }
    if (h->kind == ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API)
      seen().ranges.push_back(*static_cast<rocprofiler_buffer_tracing_marker_api_record_t*>(h->payload));
  }
}

// What rocprofv3 does with ROCTx: the messages of marks and ranges come from the range callbacks, the pause and
// resume calls from the control ones.
void on_marker(rocprofiler_callback_tracing_record_t r, rocprofiler_user_data_t* user, void*) {
  std::string message;
  if (r.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_RANGE_API) {
    auto* d = static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(r.payload);
    const char* m = r.operation == 0   ? d->args.roctxMarkA.message
                    : r.operation == 1 ? d->args.roctxThreadRangeA.message
                                       : d->args.roctxProcessRangeA.message;
    message = m ? m : "";
  }
  seen().marker_callbacks.push_back({static_cast<int>(r.kind), r.operation, static_cast<int>(r.phase),
                                     r.correlation_id.internal, r.correlation_id.ancestor, message, user});
}

int tool_init(rocprofiler_client_finalize_t, void*) {
  Seen& s = seen();
  rocprofiler_query_available_agents(
      ROCPROFILER_AGENT_INFO_VERSION_0,
      [](rocprofiler_agent_version_t, const void** agents, size_t n, void*) {
        for (size_t i = 0; i < n; ++i) seen().agents.push_back(*static_cast<const rocprofiler_agent_v0_t*>(agents[i]));
        return ROCPROFILER_STATUS_SUCCESS;
      },
      sizeof(rocprofiler_agent_v0_t), nullptr);
  rocprofiler_context_id_t objects{}, tracing{};
  rocprofiler_create_context(&objects);
  rocprofiler_configure_callback_tracing_service(objects, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
                                                 on_code_object, nullptr);
  rocprofiler_create_context(&tracing);
  rocprofiler_create_buffer(tracing, 1 << 16, 1 << 15, ROCPROFILER_BUFFER_POLICY_LOSSLESS, on_records, nullptr,
                            &s.buffer);
  for (auto kind : {ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION,
                    ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                    ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT})
    rocprofiler_configure_buffer_tracing_service(tracing, kind, nullptr, 0, s.buffer);
  rocprofiler_context_id_t markers{};
  rocprofiler_create_context(&markers);
  for (auto kind : {ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_RANGE_API, ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
                    ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API})
    rocprofiler_configure_callback_tracing_service(markers, kind, nullptr, 0, on_marker, nullptr);
  rocprofiler_configure_buffer_tracing_service(tracing, ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API, nullptr, 0,
                                               s.buffer);
  for (auto c : {objects, tracing, markers}) rocprofiler_start_context(c);
  return 0;
}

rocprofiler_tool_configure_result_t* configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id) {
  id->name = "test_amd_rocprofiler_rdna";
  static rocprofiler_tool_configure_result_t result{sizeof(rocprofiler_tool_configure_result_t), tool_init, nullptr,
                                                    nullptr};
  return &result;
}

// ROCTx's side of it: the library's own functions, which a profiler wraps in the tables it is handed (the library
// passes them to rocprofiler_set_api_table as it starts). These keep a nesting depth and number the ranges they
// start from 100, to see what the wrappers pass through.
int g_depth = 0;
uint64_t g_range = 100;
std::vector<std::string> g_roctx;
roctxCoreApiTable_t g_core{sizeof(roctxCoreApiTable_t),
                           [](const char*) { g_roctx.push_back("mark"); },
                           [](const char*) { g_roctx.push_back("push"); return g_depth++; },
                           []() { g_roctx.push_back("pop"); return --g_depth; },
                           [](const char*) { g_roctx.push_back("start"); return ++g_range; },
                           [](uint64_t) { g_roctx.push_back("stop"); },
                           [](uint64_t* tid) { *tid = 7; return 0; }};
roctxControlApiTable_t g_control{sizeof(roctxControlApiTable_t), [](uint64_t) { return 0; }, [](uint64_t) { return 0; }};
roctxNameApiTable_t g_name{sizeof(roctxNameApiTable_t), [](const char*) { return 0; },
                           [](const char*, const void*) { return 0; }, [](const char*, int) { return 0; },
                           [](const char*, const void*) { return 0; }};
int g_returned_push[2];
uint64_t g_returned_start;

// The sequence of rocprof/markers.cpp, with a HIP call here and there.
void run_markers() {
  void* tables[3] = {&g_core, &g_control, &g_name};
  VCHECK_EQ(rocprofiler_set_api_table("roctx", 0, 0, tables, 3), 0);
  g_name.roctxNameOsThread_fn("main");
  g_core.roctxMarkA_fn("start");
  g_returned_push[0] = g_core.roctxRangePushA_fn("setup");
  hipDeviceSynchronize();
  g_core.roctxRangePop_fn();
  g_returned_start = g_core.roctxRangeStartA_fn("compute");
  g_returned_push[1] = g_core.roctxRangePushA_fn("launch");
  int device = 0;
  hipGetDevice(&device);
  g_core.roctxRangePop_fn();
  hipDeviceSynchronize();
  g_core.roctxRangeStop_fn(g_returned_start);
  hipGetDevice(&device);
  g_core.roctxMarkA_fn("end");
  g_control.roctxProfilerPause_fn(0);
  g_control.roctxProfilerResume_fn(0);
}

// A card's memory traffic, once for every test to read: a module, then copies of 4 KiB, 16 KiB, 16 KiB and 4
// bytes and 64 KiB each way, a copy within the device, and pinned memory.
void run_once() {
  static bool done = false;
  if (done) return;
  done = true;
  VCHECK_EQ(rocprofiler_force_configure(&configure), ROCPROFILER_STATUS_SUCCESS);
  hipModule_t module = nullptr;
  VCHECK_EQ(hipModuleLoad(&module, (kData + "bytes.gfx1030.o").c_str()), hipSuccess);
  void *a = nullptr, *b = nullptr, *pinned = nullptr;
  VCHECK_EQ(hipMalloc(&a, 1 << 20), hipSuccess);
  VCHECK_EQ(hipMalloc(&b, 1 << 20), hipSuccess);
  std::vector<char> host(1 << 20);
  for (size_t bytes : {size_t{4096}, size_t{16384}, size_t{16388}, size_t{65536}})
    VCHECK_EQ(hipMemcpy(a, host.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
  VCHECK_EQ(hipMemcpy(host.data(), a, 65536, hipMemcpyDeviceToHost), hipSuccess);
  VCHECK_EQ(hipMemcpy(b, a, 65536, hipMemcpyDeviceToDevice), hipSuccess);
  VCHECK_EQ(hipHostMalloc(&pinned, 1 << 20, 0), hipSuccess);
  VCHECK_EQ(hipMemcpy(a, pinned, 1 << 20, hipMemcpyHostToDevice), hipSuccess);
  VCHECK_EQ(hipHostFree(pinned), hipSuccess);
  // Fills: 100 bytes are 96 and a tail of 4; 4096 bytes from 1 past a 16-byte boundary a head of 15, a body
  // of 4080 and a tail of 1.
  VCHECK_EQ(hipMemset(a, 0, 100), hipSuccess);
  VCHECK_EQ(hipMemset(static_cast<char*>(a) + 1, 0, 4096), hipSuccess);
  VCHECK_EQ(hipDeviceSynchronize(), hipSuccess);
  // A stream, and the memory pool: 1 KiB and 40 MiB need two 32 MiB ranges; once both are freed, the
  // synchronization gives them back.
  hipStream_t stream = nullptr;
  VCHECK_EQ(hipStreamCreate(&stream), hipSuccess);
  void *p1 = nullptr, *p2 = nullptr;
  VCHECK_EQ(hipMallocAsync(&p1, 1024, stream), hipSuccess);
  VCHECK_EQ(hipMallocAsync(&p2, size_t{40} << 20, stream), hipSuccess);
  VCHECK_EQ(hipFreeAsync(p1, stream), hipSuccess);
  VCHECK_EQ(hipStreamSynchronize(stream), hipSuccess);
  VCHECK_EQ(hipFreeAsync(p2, stream), hipSuccess);
  VCHECK_EQ(hipStreamSynchronize(stream), hipSuccess);
  VCHECK_EQ(hipStreamDestroy(stream), hipSuccess);
  VCHECK_EQ(hipFree(a), hipSuccess);
  VCHECK_EQ(hipFree(b), hipSuccess);
  run_markers();
  rocprofiler_flush_buffer(seen().buffer);
}

bool is_rx6700xt() {
  const char* g = std::getenv("VGPU_GPU");
  return g && std::strstr(g, "rx6700xt");
}

}  // namespace

VTEST(the_agent_is_the_card_as_kfd_counts_it) {
  run_once();
  const Seen& s = seen();
  VCHECK(s.agents.size() >= 2);
  const rocprofiler_agent_v0_t& g = s.agents[1];
  VCHECK_EQ(g.type, ROCPROFILER_AGENT_TYPE_GPU);
  // An RX 6800 is 60 compute units, two SIMDs each, 32 waves to a compute unit, and an RX 6700 XT 40. HIP counts
  // workgroup processors (30 and 20). With HSA_OVERRIDE_GFX_VERSION the runtime says gfx1030 for both; KFD, and
  // so the profiler, says what the card is.
  if (is_rx6700xt()) {
    VCHECK_EQ(std::string(g.name), std::string("gfx1031"));
    VCHECK_EQ(std::string(g.product_name), std::string("AMD Radeon RX 6700 XT"));
    VCHECK_EQ(g.gfx_target_version, 100301u);
    VCHECK_EQ(g.cu_count, 40u);
  } else {
    VCHECK_EQ(std::string(g.name), std::string("gfx1030"));
    VCHECK_EQ(std::string(g.product_name), std::string("AMD Radeon RX 6800"));
    VCHECK_EQ(g.gfx_target_version, 100300u);
    VCHECK_EQ(g.cu_count, 60u);
  }
  VCHECK_EQ(g.simd_per_cu, 2u);
  VCHECK_EQ(g.simd_count, g.cu_count * 2);
  VCHECK_EQ(g.wave_front_size, 32u);
  VCHECK_EQ(g.max_waves_per_cu, 32u);
  VCHECK_EQ(g.max_waves_per_simd, 16u);
}

// What rocprofv3 printed for the agent on the cards (amd/tests/data/rocprof_agents*.expected): the fields the
// profile does not carry.
VTEST(the_agent_has_the_fields_the_card_measured) {
  run_once();
  const rocprofiler_agent_v0_t& g = seen().agents[1];
  VCHECK_EQ(g.num_xcc, 1u);
  VCHECK_EQ(g.num_gws, 64u);
  VCHECK_EQ(g.simd_arrays_per_engine, 2u);
  VCHECK_EQ(g.cu_per_simd_array, 10u);
  VCHECK_EQ(g.max_slots_scratch_cu, 32u);
  VCHECK_EQ(g.num_sdma_queues_per_engine, 8u);
  VCHECK_EQ(g.num_cp_queues, 8u);
  VCHECK_EQ(g.max_engine_clk_ccompute, 5756u);
  VCHECK_EQ(g.fw_version.Value, 116u);
  VCHECK_EQ(g.family_id, 143u);
  VCHECK_EQ(g.local_mem_size, uint64_t{0});   // KFD keeps a discrete card's memory in its bank
  VCHECK_EQ(g.grid_max_size, 4294967295u);
  VCHECK_EQ(g.grid_max_dim.y, 65535u);
  VCHECK_EQ(g.mem_banks_count, 1u);
  VCHECK_EQ(g.io_links_count, 1u);
  VCHECK_EQ(g.io_links[0].max_bandwidth, 32000u);
  VCHECK_EQ(g.io_links[0].node_to, 0u);
  VCHECK_EQ(g.io_links[0].node_from, g.node_id);
  if (is_rx6700xt()) {
    VCHECK_EQ(std::string(g.model_name), std::string("navy_flounder"));
    VCHECK_EQ(g.array_count, 4u);
    VCHECK_EQ(g.num_shader_banks, 2u);
    VCHECK_EQ(g.num_sdma_engines, 2u);
    VCHECK_EQ(g.caches_count, 86u);
    VCHECK_EQ(g.sdma_fw_version.Value, 80u);
    VCHECK_EQ(g.max_engine_clk_fcompute, 2855u);
    VCHECK_EQ(g.mem_banks[0].width, 192u);
    VCHECK_EQ(g.mem_banks[0].size_in_bytes, uint64_t{12868124672});
  } else {
    VCHECK_EQ(std::string(g.model_name), std::string("sienna_cichlid"));
    VCHECK_EQ(g.array_count, 8u);
    VCHECK_EQ(g.num_shader_banks, 4u);
    VCHECK_EQ(g.num_sdma_engines, 4u);
    VCHECK_EQ(g.caches_count, 128u);
    VCHECK_EQ(g.sdma_fw_version.Value, 83u);
    VCHECK_EQ(g.max_engine_clk_fcompute, 2475u);
    VCHECK_EQ(g.mem_banks[0].width, 256u);
    VCHECK_EQ(g.mem_banks[0].size_in_bytes, uint64_t{17163091968});
  }
  VCHECK(g.caches[g.caches_count - 1].level == 3);
}

// The counters the card offers that are counted exactly: waves, and the agent's own properties. The cycle, stall and
// cache counters are not offered, nor are the card's wave32-instruction counters and averages.
VTEST(only_the_counters_counted_exactly_are_offered) {
  run_once();
  const rocprofiler_agent_v0_t& g = seen().agents[1];
  static std::vector<std::string> names;
  names.clear();
  VCHECK_EQ(rocprofiler_iterate_agent_supported_counters(
                g.id,
                [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* ids, size_t n, void*) {
                  for (size_t i = 0; i < n; ++i) {
                    rocprofiler_counter_info_v0_t info{};
                    rocprofiler_query_counter_info(ids[i], ROCPROFILER_COUNTER_INFO_VERSION_0, &info);
                    names.push_back(info.name);
                  }
                  return ROCPROFILER_STATUS_SUCCESS;
                },
                nullptr),
            ROCPROFILER_STATUS_SUCCESS);
  auto has = [&](const char* n) { return std::find(names.begin(), names.end(), n) != names.end(); };
  VCHECK(has("SQ_WAVES"));
  VCHECK(has("SQ_WAVES_sum"));
  VCHECK(has("Wavefronts"));
  VCHECK(has("SE_NUM"));
  VCHECK(has("num_sdma_engines"));
  VCHECK(!has("SQ_WAVES_EQ_64"));   // CDNA's: a wave32 card has no counter of 64-lane waves
  VCHECK(!has("SQ_INSTS_MFMA"));
  VCHECK(!has("GRBM_COUNT"));
  for (const char* n : {"SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_SMEM", "SQ_INSTS_FLAT", "SQ_INSTS_LDS", "SQ_INSTS_GDS",
                        "TA_FLAT_LOAD_WAVEFRONTS", "TA_FLAT_STORE_WAVEFRONTS_sum", "SQ_WAVE32_INSTS", "SQ_WAVE64_INSTS"})
    VCHECK(has(n));
  VCHECK(!has("SQ_INSTS_WAVE32"));   // a wave64 instruction counts "1 or 2" by what gets issued
  VCHECK(!has("VALUInsts"));         // an average is a fraction
}

VTEST(a_code_object_is_named_by_a_uri_and_its_kernels_by_their_descriptors) {
  run_once();
  const Seen& s = seen();
  // The runtime's own code object (copy and fill kernels) first, then the program's.
  VCHECK_EQ(s.uris.size(), size_t{2});
  // The string must outlive the callback (rocprofv3 keeps the pointer); it is read here, after the program ran.
  for (const std::string& u : s.uris) VCHECK(u.rfind("memory://", 0) == 0 || u.rfind("file://", 0) == 0);
  VCHECK(!s.kernels.empty());
  for (const Seen::Kernel& k : s.kernels) {
    // "<kernel>.kd", the descriptor's symbol; from gfx10 on 128 scalar registers are reported whatever the kernel
    // uses, and the vector registers in granules of 8 (wave32).
    VCHECK(k.name.size() > 3 && k.name.compare(k.name.size() - 3, 3, ".kd") == 0);
    VCHECK_EQ(k.sgprs, 128u);
    VCHECK(k.vgprs > 0 && k.vgprs % 8 == 0);
  }
}

VTEST(only_what_the_runtime_gives_a_dma_engine_is_a_memory_copy) {
  run_once();
  const Seen& s = seen();
  // 4 KiB and 16 KiB are copied by a kernel the runtime runs itself, as is everything within a device; 16388
  // bytes and up go to a DMA engine, pinned memory too. (Measured with rocprofv3 on both cards.)
  std::vector<std::pair<int, uint64_t>> got;
  for (const auto& c : s.copies) got.push_back({static_cast<int>(c.operation), c.bytes});
  const std::vector<std::pair<int, uint64_t>> want = {{ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE, 16388},
                                                      {ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE, 65536},
                                                      {ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST, 65536},
                                                      {ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE, 1 << 20}};
  VCHECK(got == want);
}

VTEST(an_allocation_record_has_its_own_id_and_names_the_call_that_made_it) {
  run_once();
  const Seen& s = seen();
  std::map<uint64_t, std::string> call_of;
  for (const Call& c : s.calls) call_of[c.id] = c.name;
  int allocated = 0, freed = 0, host_allocated = 0, host_freed = 0;
  for (const auto& r : s.allocations) {
    VCHECK(r.correlation_id.ancestor != 0);
    VCHECK(r.correlation_id.internal != r.correlation_id.ancestor);
    VCHECK(call_of.count(r.correlation_id.ancestor) == 1);
    const std::string& by = call_of[r.correlation_id.ancestor];
    if (r.operation == ROCPROFILER_MEMORY_ALLOCATION_ALLOCATE) {
      // (The runtime's own blocks under a call are not the program's: sizes of 4 MiB, 16 MiB and 56 bytes.)
      if (by == "hipMalloc" && r.allocation_size == uint64_t{1} << 20) ++allocated;
      if (by == "hipHostMalloc") ++host_allocated;
    } else if (r.operation == ROCPROFILER_MEMORY_ALLOCATION_FREE) {
      if (by == "hipFree") ++freed;
      if (by == "hipHostFree") ++host_freed;
    }
  }
  VCHECK_EQ(allocated, 2);
  VCHECK_EQ(freed, 2);
  VCHECK_EQ(host_allocated, 1);
  VCHECK_EQ(host_freed, 1);
}

VTEST(the_runtimes_own_kernels_do_the_small_and_device_to_device_copies_and_the_fills) {
  run_once();
  const Seen& s = seen();
  // Sixteen bytes to a thread in workgroups of 512 for aligned copies (4 KiB: one workgroup; 16 KiB: 1024
  // threads; the 64 KiB copy within the device: 4096), then the two fills' kernels of 256 (a head, a body and a
  // tail each). Measured with rocprofv3 on both cards.
  const std::string copy = "__amd_rocclr_copyBuffer.kd", fill = "__amd_rocclr_fillBufferAligned.kd";
  std::vector<Seen::Dispatch> want = {{copy, 512, 512},  {copy, 1024, 512}, {copy, 4096, 512},
                                      {fill, 256, 256},  {fill, 256, 256}, {fill, 256, 256},
                                      {fill, 256, 256},  {fill, 256, 256}};
  VCHECK_EQ(s.dispatches.size(), want.size());
  for (size_t i = 0; i < want.size() && i < s.dispatches.size(); ++i) {
    VCHECK_EQ(s.dispatches[i].kernel, want[i].kernel);
    VCHECK_EQ(s.dispatches[i].grid, want[i].grid);
    VCHECK_EQ(s.dispatches[i].workgroup, want[i].workgroup);
  }
}

VTEST(the_runtimes_own_allocations_are_staging_buffers_and_a_free_names_no_agent) {
  run_once();
  const Seen& s = seen();
  std::map<uint64_t, std::string> call_of;
  for (const Call& c : s.calls) call_of[c.id] = c.name;
  // The first runtime call (hipModuleLoad here) starts the runtime: a 4 MiB staging block and one of 56 bytes on the
  // host; the first work on the default stream (the first hipMemcpy) makes its queue: 4 MiB and 16 MiB.
  std::vector<std::pair<std::string, uint64_t>> got;
  for (const auto& r : s.allocations) {
    if (r.operation != ROCPROFILER_MEMORY_ALLOCATION_ALLOCATE || r.agent_id.handle != s.agents[0].id.handle) continue;
    got.push_back({call_of[r.correlation_id.ancestor], r.allocation_size});
  }
  const std::vector<std::pair<std::string, uint64_t>> want = {{"hipModuleLoad", uint64_t{4} << 20},
                                                              {"hipModuleLoad", 56},
                                                              {"hipMemcpy", uint64_t{4} << 20},
                                                              {"hipMemcpy", uint64_t{16} << 20},
                                                              {"hipHostMalloc", uint64_t{1} << 20},
                                                              {"hipStreamCreate", uint64_t{4} << 20},
                                                              {"hipStreamCreate", uint64_t{16} << 20}};
  VCHECK(got == want);
  // The pool's ranges and the frees, under the calls that made them; a stream's destruction frees three blocks
  // (the first has no address), and no free names an agent.
  std::vector<std::string> vmem, frees;
  for (const auto& r : s.allocations) {
    const std::string by = call_of[r.correlation_id.ancestor];
    if (r.operation == ROCPROFILER_MEMORY_ALLOCATION_VMEM_ALLOCATE) {
      VCHECK_EQ(r.allocation_size, uint64_t{32} << 20);
      vmem.push_back("alloc " + by);
    }
    if (r.operation == ROCPROFILER_MEMORY_ALLOCATION_VMEM_FREE) vmem.push_back("free " + by);
    if (r.operation == ROCPROFILER_MEMORY_ALLOCATION_FREE || r.operation == ROCPROFILER_MEMORY_ALLOCATION_VMEM_FREE) {
      VCHECK_EQ(r.agent_id.handle, uint64_t{0});
      if (by == "hipStreamDestroy") frees.push_back(r.address.value ? "block" : "null");
    }
  }
  const std::vector<std::string> want_vmem = {"alloc hipMallocAsync", "alloc hipMallocAsync", "free hipStreamSynchronize",
                                              "free hipStreamSynchronize"};
  VCHECK(vmem == want_vmem);
  VCHECK((frees == std::vector<std::string>{"null", "block", "block"}));
}

VTEST(the_compiler_api_has_its_operations) {
  const char* name = nullptr;
  VCHECK_EQ(rocprofiler_query_buffer_tracing_kind_operation_name(ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API, 2, &name,
                                                                 nullptr),
            ROCPROFILER_STATUS_SUCCESS);
  VCHECK_EQ(std::string(name), std::string("__hipRegisterFatBinary"));
  VCHECK_EQ(rocprofiler_query_buffer_tracing_kind_operation_name(ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API, 0, &name,
                                                                 nullptr),
            ROCPROFILER_STATUS_SUCCESS);
  VCHECK_EQ(std::string(name), std::string("__hipPopCallConfiguration"));
}

VTEST(roctx_marks_and_ranges_are_records_with_the_range_they_were_made_in) {
  run_once();
  const Seen& s = seen();
  // The wrappers pass through to ROCTx: its return values come back, and each function ran once.
  VCHECK_EQ(g_returned_push[0], 0);
  VCHECK_EQ(g_returned_push[1], 0);
  VCHECK_EQ(g_returned_start, uint64_t{101});
  VCHECK_EQ(g_roctx.size(), size_t{8});

  // The messages come from the callbacks, by correlation id, as rocprofv3 reads them.
  std::map<uint64_t, std::string> message;
  for (const MarkerCallback& c : s.marker_callbacks)
    if (c.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_RANGE_API && !c.message.empty()) message[c.id] = c.message;
  VCHECK_EQ(message.size(), size_t{5});

  // The buffer records: marks at once, a range when it ends, each with the operation of its kind.
  std::vector<std::string> got;
  std::map<std::string, uint64_t> id_of;
  for (const auto& r : s.ranges) {
    const char* name = nullptr;
    VCHECK_EQ(rocprofiler_query_buffer_tracing_kind_operation_name(ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API,
                                                                   r.operation, &name, nullptr),
              ROCPROFILER_STATUS_SUCCESS);
    const std::string m = message[r.correlation_id.internal];
    id_of[m] = r.correlation_id.internal;
    got.push_back(std::string(name) + " " + m + " in " +
                  (r.correlation_id.ancestor ? message[r.correlation_id.ancestor] : std::string("-")));
  }
  const std::vector<std::string> want = {"roctxMarkA start in -", "roctxThreadRangeA setup in -",
                                         "roctxThreadRangeA launch in compute", "roctxProcessRangeA compute in -",
                                         "roctxMarkA end in -"};
  VCHECK(got == want);

  // A range is entered where it begins and exited where it ends, with the same user data; a mark does both at once.
  std::map<uint64_t, std::vector<const MarkerCallback*>> by_id;
  for (const MarkerCallback& c : s.marker_callbacks)
    if (c.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_RANGE_API) by_id[c.id].push_back(&c);
  VCHECK_EQ(by_id.size(), size_t{5});
  for (const auto& entry : by_id) {
    const uint64_t id = entry.first;
    const auto& calls = entry.second;
    VCHECK_EQ(calls.size(), size_t{2});
    VCHECK_EQ(calls[0]->phase, static_cast<int>(ROCPROFILER_CALLBACK_PHASE_ENTER));
    VCHECK_EQ(calls[1]->phase, static_cast<int>(ROCPROFILER_CALLBACK_PHASE_EXIT));
    VCHECK(calls[0]->user == calls[1]->user);
    VCHECK_EQ(calls[1]->message, message[id]);
  }

  // The names and the profiler control are traced as calls, enter and exit.
  int names = 0, controls = 0;
  for (const MarkerCallback& c : s.marker_callbacks) {
    names += c.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API;
    controls += c.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API;
  }
  VCHECK_EQ(names, 2);
  VCHECK_EQ(controls, 4);

  // The HIP calls made inside a range have it as their ancestor, the innermost open one, a range started by id
  // included until it is stopped.
  std::vector<std::pair<std::string, uint64_t>> hip;
  for (const Call& c : s.calls)
    if (c.name == "hipDeviceSynchronize" || c.name == "hipGetDevice") hip.push_back({c.name, c.ancestor});
  VCHECK_EQ(hip.size(), size_t{4});
  VCHECK_EQ(hip[0].second, id_of["setup"]);
  VCHECK_EQ(hip[1].second, id_of["launch"]);
  VCHECK_EQ(hip[2].second, id_of["compute"]);
  VCHECK_EQ(hip[3].second, uint64_t{0});
}

VTEST(roctx_operations_have_their_names) {
  const char* name = nullptr;
  const struct {
    rocprofiler_buffer_tracing_kind_t kind;
    int op;
    const char* want;
  } names[] = {{ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API, 0, "roctxMarkA"},
               {ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API, 4, "roctxRangeStop"},
               {ROCPROFILER_BUFFER_TRACING_MARKER_CONTROL_API, 1, "roctxProfilerResume"},
               {ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API, 3, "roctxNameHipStream"},
               {ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API, 2, "roctxProcessRangeA"}};
  for (const auto& n : names) {
    VCHECK_EQ(rocprofiler_query_buffer_tracing_kind_operation_name(n.kind, n.op, &name, nullptr),
              ROCPROFILER_STATUS_SUCCESS);
    VCHECK_EQ(std::string(name), std::string(n.want));
  }
}

VTEST_MAIN
