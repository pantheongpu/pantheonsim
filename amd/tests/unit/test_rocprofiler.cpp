// VirtualGPU's librocprofiler-sdk, from a profiling tool's side: this program
// is the tool. It configures itself through rocprofiler-sdk's interface,
// subscribes to code objects, kernel dispatches, copies and HIP calls, asks
// for counters on every dispatch, and then runs a HIP program on the simulated
// MI300X through libamdhip64 -- the path AMD's rocprofv3 takes, without
// needing ROCm installed.
//
// The kernel is counted_mix (amd/tests/data/counters.c), which has no
// branches, so what each counter should read is its listing's count times the
// waves (counters_oracle.hpp). amd/tests/e2e/run_rocprofv3.sh runs AMD's own
// rocprofv3 the same way wherever ROCm's rocprofiler-sdk is installed.
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "counters_oracle.hpp"
#include "vgpu/rocprofiler_abi.hpp"
#include "vgpu_hip.h"
#include "vtest.hpp"

using namespace vgpu::amd::rocprof;

extern "C" {
rocprofiler_status_t rocprofiler_force_configure(rocprofiler_configure_func_t);
rocprofiler_status_t rocprofiler_is_initialized(int*);
rocprofiler_status_t rocprofiler_query_available_agents(rocprofiler_agent_version_t,
                                                        rocprofiler_query_available_agents_cb_t, size_t, void*);
rocprofiler_status_t rocprofiler_iterate_agent_supported_counters(rocprofiler_agent_id_t,
                                                                  rocprofiler_available_counters_cb_t, void*);
rocprofiler_status_t rocprofiler_query_counter_info(rocprofiler_counter_id_t, rocprofiler_counter_info_version_id_t,
                                                    void*);
rocprofiler_status_t rocprofiler_query_record_counter_id(rocprofiler_counter_instance_id_t, rocprofiler_counter_id_t*);
rocprofiler_status_t rocprofiler_create_counter_config(rocprofiler_agent_id_t, rocprofiler_counter_id_t*, size_t,
                                                       rocprofiler_counter_config_id_t*);
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
rocprofiler_status_t rocprofiler_configure_callback_dispatch_counting_service(
    rocprofiler_context_id_t, rocprofiler_dispatch_counting_service_cb_t, void*,
    rocprofiler_dispatch_counting_record_cb_t, void*);
rocprofiler_status_t rocprofiler_flush_buffer(rocprofiler_buffer_id_t);
rocprofiler_status_t rocprofiler_query_buffer_tracing_kind_operation_name(rocprofiler_buffer_tracing_kind_t,
                                                                          rocprofiler_tracing_operation_t,
                                                                          const char**, uint64_t*);
}

namespace {

const std::string kData = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/";

// What the tool was told.
struct Seen {
  std::vector<rocprofiler_agent_v0_t> agents;
  std::map<std::string, uint64_t> counter_ids;           // every counter a GPU offers, by name
  std::map<uint64_t, std::string> kernels;               // kernel id -> name
  int code_objects = 0;
  std::map<std::string, double> counts;                  // the one dispatch's counters, by name
  std::vector<rocprofiler_buffer_tracing_kernel_dispatch_record_t> dispatches;
  std::vector<rocprofiler_buffer_tracing_memory_copy_record_t> copies;
  std::vector<std::string> calls;                        // HIP functions, in order
  rocprofiler_buffer_id_t buffer{};
  rocprofiler_counter_config_id_t config{};
  bool initialized = false;
};
Seen& seen() {
  static Seen s;
  return s;
}

void on_code_object(rocprofiler_callback_tracing_record_t r, rocprofiler_user_data_t*, void*) {
  if (r.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
  if (r.operation == ROCPROFILER_CODE_OBJECT_LOAD) ++seen().code_objects;
  if (r.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) {
    auto* k = static_cast<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(r.payload);
    seen().kernels[k->kernel_id] = k->kernel_name;
  }
}

void on_records(rocprofiler_context_id_t, rocprofiler_buffer_id_t, rocprofiler_record_header_t** headers, size_t n,
                void*, uint64_t) {
  for (size_t i = 0; i < n; ++i) {
    const rocprofiler_record_header_t* h = headers[i];
    if (h->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) continue;
    if (h->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
      seen().dispatches.push_back(*static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(h->payload));
    if (h->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
      seen().copies.push_back(*static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(h->payload));
    if (h->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT) {
      auto* r = static_cast<rocprofiler_buffer_tracing_hip_api_ext_record_t*>(h->payload);
      const char* name = nullptr;
      rocprofiler_query_buffer_tracing_kind_operation_name(ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT, r->operation,
                                                           &name, nullptr);
      seen().calls.push_back(name ? name : "?");
    }
  }
}

void on_dispatch(rocprofiler_dispatch_counting_service_data_t, rocprofiler_counter_config_id_t* config,
                 rocprofiler_user_data_t*, void*) {
  *config = seen().config;
}

void on_counts(rocprofiler_dispatch_counting_service_data_t, rocprofiler_counter_record_t* records, size_t n,
               rocprofiler_user_data_t, void*) {
  for (size_t i = 0; i < n; ++i) {
    rocprofiler_counter_id_t id{};
    rocprofiler_query_record_counter_id(records[i].id, &id);
    for (const auto& [name, cid] : seen().counter_ids)
      if (cid == id.handle) seen().counts[name] += records[i].counter_value;
  }
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
  rocprofiler_agent_id_t gpu{};
  for (const auto& a : s.agents)
    if (a.type == ROCPROFILER_AGENT_TYPE_GPU && !gpu.handle) gpu = a.id;
  rocprofiler_iterate_agent_supported_counters(
      gpu,
      [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* ids, size_t n, void*) {
        for (size_t i = 0; i < n; ++i) {
          rocprofiler_counter_info_v1_t info{};
          rocprofiler_query_counter_info(ids[i], ROCPROFILER_COUNTER_INFO_VERSION_1, &info);
          seen().counter_ids[info.name] = ids[i].handle;
        }
        return ROCPROFILER_STATUS_SUCCESS;
      },
      nullptr);
  std::vector<rocprofiler_counter_id_t> all;
  for (const auto& [name, id] : s.counter_ids) all.push_back({id});
  rocprofiler_create_counter_config(gpu, all.data(), all.size(), &s.config);

  rocprofiler_context_id_t objects{}, tracing{}, counting{};
  rocprofiler_create_context(&objects);
  rocprofiler_configure_callback_tracing_service(objects, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
                                                 on_code_object, nullptr);
  rocprofiler_create_context(&tracing);
  rocprofiler_create_buffer(tracing, 1 << 16, 1 << 15, ROCPROFILER_BUFFER_POLICY_LOSSLESS, on_records, nullptr,
                            &s.buffer);
  for (auto kind : {ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                    ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT})
    rocprofiler_configure_buffer_tracing_service(tracing, kind, nullptr, 0, s.buffer);
  rocprofiler_create_context(&counting);
  rocprofiler_configure_callback_dispatch_counting_service(counting, on_dispatch, nullptr, on_counts, nullptr);
  for (auto c : {objects, tracing, counting}) rocprofiler_start_context(c);
  s.initialized = true;
  return 0;
}

rocprofiler_tool_configure_result_t* configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id) {
  id->name = "test_amd_rocprofiler";
  static rocprofiler_tool_configure_result_t result{sizeof(rocprofiler_tool_configure_result_t), tool_init, nullptr,
                                                    nullptr};
  return &result;
}

// Configures the tool, then runs counted_mix on three groups of 64 and copies
// its inputs in and its results out -- once, for every test to read.
void run_once() {
  static bool done = false;
  if (done) return;
  done = true;
  VCHECK_EQ(rocprofiler_force_configure(&configure), ROCPROFILER_STATUS_SUCCESS);
  hipModule_t module = nullptr;
  VCHECK_EQ(hipModuleLoad(&module, (kData + "counters.gfx942.o").c_str()), hipSuccess);
  hipFunction_t f = nullptr;
  VCHECK_EQ(hipModuleGetFunction(&f, module, "counted_mix"), hipSuccess);
  const size_t n = 3 * 128;
  std::vector<float> in(n, 1.5f), out(n);
  void *din = nullptr, *dout = nullptr, *dhits = nullptr, *dg = nullptr;
  for (void** p : {&din, &dout, &dhits, &dg}) VCHECK_EQ(hipMalloc(p, n * 4), hipSuccess);
  VCHECK_EQ(hipMemcpy(din, in.data(), n * 4, hipMemcpyHostToDevice), hipSuccess);
  int use_lds = 0;
  void* args[] = {&din, &dout, &dhits, &dg, &use_lds};
  VCHECK_EQ(hipModuleLaunchKernel(f, 3, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr), hipSuccess);
  VCHECK_EQ(hipMemcpy(out.data(), dout, n * 4, hipMemcpyDeviceToHost), hipSuccess);
  rocprofiler_flush_buffer(seen().buffer);
}

}  // namespace

VTEST(the_tool_is_configured_and_sees_the_machine) {
  run_once();
  const Seen& s = seen();
  VCHECK(s.initialized);
  int status = 0;
  rocprofiler_is_initialized(&status);
  VCHECK_EQ(status, 1);
  // The CPU, then the GPUs, numbered as KFD numbers them.
  VCHECK(s.agents.size() >= 2);
  VCHECK_EQ(s.agents[0].type, ROCPROFILER_AGENT_TYPE_CPU);
  const rocprofiler_agent_v0_t& g = s.agents[1];
  VCHECK_EQ(g.type, ROCPROFILER_AGENT_TYPE_GPU);
  VCHECK_EQ(std::string(g.name), std::string("gfx942"));
  VCHECK_EQ(std::string(g.product_name), std::string("AMD Instinct MI300X"));
  VCHECK_EQ(g.gfx_target_version, 90402u);
  VCHECK_EQ(g.cu_count, 304u);
  VCHECK_EQ(g.wave_front_size, 64u);
  VCHECK_EQ(g.node_id, 1u);
}

VTEST(a_gpu_offers_only_what_is_counted_exactly) {
  run_once();
  const auto& ids = seen().counter_ids;
  for (const char* c : {"SQ_WAVES", "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_SMEM", "SQ_INSTS_LDS",
                        "SQ_INSTS_MFMA", "TA_FLAT_READ_WAVEFRONTS", "TA_FLAT_WRITE_WAVEFRONTS"})
    VCHECK(ids.count(c) == 1);
  // What needs a model of time, caches or memory traffic is not offered.
  for (const char* c : {"GRBM_COUNT", "GRBM_GUI_ACTIVE", "SQ_BUSY_CYCLES", "TCC_HIT", "TCC_MISS", "TCP_TCC_READ_REQ"})
    VCHECK(ids.count(c) == 0);
}

VTEST(the_code_object_and_its_kernels_are_announced) {
  run_once();
  VCHECK_EQ(seen().code_objects, 1);
  bool found = false;
  for (const auto& [id, name] : seen().kernels) found = found || name == "counted_mix";
  VCHECK(found);
}

VTEST(the_dispatch_is_counted_as_the_listing_says) {
  run_once();
  const Expected e = from_listing(kData);
  const auto& c = seen().counts;
  const double w = 3;   // three groups of one wave
  VCHECK_EQ(c.at("SQ_WAVES"), w);
  VCHECK_EQ(c.at("SQ_WAVES_EQ_64"), w);
  VCHECK_EQ(c.at("SQ_INSTS_VALU"), double(e.valu) * w);
  VCHECK_EQ(c.at("SQ_INSTS_MFMA"), double(e.mfma) * w);
  VCHECK_EQ(c.at("SQ_INSTS_SALU"), double(e.salu) * w);
  VCHECK_EQ(c.at("SQ_INSTS_SMEM"), double(e.smem) * w);
  VCHECK_EQ(c.at("SQ_INSTS_VMEM"), double(e.vmem) * w);
  VCHECK_EQ(c.at("SQ_INSTS_FLAT"), double(e.flat) * w);
  VCHECK_EQ(c.at("SQ_INSTS_LDS"), double(e.lds) * w);
  VCHECK_EQ(c.at("SQ_INSTS_BRANCH"), 0.0);
  VCHECK_EQ(c.at("TA_FLAT_READ_WAVEFRONTS"), double(e.reads) * w);
  VCHECK_EQ(c.at("TA_FLAT_WRITE_WAVEFRONTS"), double(e.writes) * w);
  VCHECK_EQ(c.at("TA_FLAT_ATOMIC_WAVEFRONTS"), double(e.atomics) * w);
  VCHECK_EQ(c.at("TA_FLAT_READ_WAVEFRONTS_sum"), c.at("TA_FLAT_READ_WAVEFRONTS"));
}

VTEST(the_dispatch_the_copies_and_the_calls_are_traced) {
  run_once();
  const Seen& s = seen();
  VCHECK_EQ(s.dispatches.size(), size_t{1});
  const auto& d = s.dispatches.at(0);
  VCHECK_EQ(s.kernels.at(d.dispatch_info.kernel_id), std::string("counted_mix"));
  VCHECK_EQ(d.dispatch_info.grid_size.x, 192u);   // in work-items
  VCHECK_EQ(d.dispatch_info.workgroup_size.x, 64u);
  VCHECK(d.end_timestamp >= d.start_timestamp && d.start_timestamp > 0);
  VCHECK_EQ(s.copies.size(), size_t{2});
  VCHECK_EQ(s.copies.at(0).operation, ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE);
  VCHECK_EQ(s.copies.at(1).operation, ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST);
  VCHECK_EQ(s.copies.at(0).bytes, uint64_t{3 * 128 * 4});
  // Each HIP call once, by the name rocprofiler-sdk gives its number; the
  // launch is the call the dispatch belongs to.
  bool launched = false;
  for (const std::string& c : s.calls) launched = launched || c == "hipModuleLaunchKernel";
  VCHECK(launched);
  VCHECK_EQ(s.calls.front(), std::string("hipModuleLoad"));
}

VTEST_MAIN
