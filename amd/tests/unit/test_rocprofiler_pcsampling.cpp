// PC sampling through VirtualGPU's librocprofiler-sdk, from a tool's side:
// this program is the tool. It asks the agents which PC sampling
// configurations they offer, configures the host-trap method on the simulated
// MI300X, runs counted_mix (amd/tests/data/counters.c) on three work-groups of
// one wave each, and reads the samples out of the buffer.
//
// The simulator has no clock, so the sampling interval is in instructions: a
// wave is sampled when it is about to issue its 1st, (1 + interval)th, ...
// instruction. counted_mix has no branches, so every wave issues the
// instructions of its listing once, in order, and the sampled program counters
// are the listing's addresses (counters.gfx942.pcs, from llvm-objdump), every
// interval-th one. The interval comes from VGPU_TEST_PCS_INTERVAL, so ctest
// runs this twice (1 and 5).
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "vgpu/rocprofiler_abi.hpp"
#include "vgpu_hip.h"
#include "vtest.hpp"

using namespace vgpu::amd::rocprof;

extern "C" {
rocprofiler_status_t rocprofiler_force_configure(rocprofiler_configure_func_t);
rocprofiler_status_t rocprofiler_query_available_agents(rocprofiler_agent_version_t,
                                                        rocprofiler_query_available_agents_cb_t, size_t, void*);
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
rocprofiler_status_t rocprofiler_query_pc_sampling_agent_configurations(
    rocprofiler_agent_id_t, rocprofiler_available_pc_sampling_configurations_cb_t, void*);
rocprofiler_status_t rocprofiler_configure_pc_sampling_service(rocprofiler_context_id_t, rocprofiler_agent_id_t,
                                                               rocprofiler_pc_sampling_method_t,
                                                               rocprofiler_pc_sampling_unit_t, uint64_t,
                                                               rocprofiler_buffer_id_t, int);
}

namespace {

const std::string kData = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/";

struct Seen {
  std::vector<rocprofiler_agent_v0_t> agents;
  rocprofiler_agent_id_t gpu{}, cpu{};
  uint64_t interval = 1;
  // The configurations a GPU agent offered before and after the service was
  // configured, and a CPU agent's.
  std::vector<rocprofiler_pc_sampling_configuration_t> before, after, cpu_configs;
  rocprofiler_status_t unknown_agent = ROCPROFILER_STATUS_SUCCESS;
  // What configuring in ways that must be refused answered.
  std::map<std::string, rocprofiler_status_t> refused;
  rocprofiler_status_t configured = ROCPROFILER_STATUS_ERROR;
  // The code object and the kernel's address, from the code object callback.
  uint64_t code_object_id = 0, load_base = 0, load_size = 0, kernel_address = 0;
  rocprofiler_buffer_id_t pc_buffer{}, trace_buffer{};
  std::vector<rocprofiler_pc_sampling_record_host_trap_v0_t> samples;
  int other_kinds = 0;   // PC sampling records that are not host-trap ones
  std::vector<rocprofiler_buffer_tracing_kernel_dispatch_record_t> dispatches;
};
Seen& seen() {
  static Seen s;
  return s;
}

std::vector<rocprofiler_pc_sampling_configuration_t> query(rocprofiler_agent_id_t agent, rocprofiler_status_t* status) {
  std::vector<rocprofiler_pc_sampling_configuration_t> out;
  *status = rocprofiler_query_pc_sampling_agent_configurations(
      agent,
      [](const rocprofiler_pc_sampling_configuration_t* c, size_t n, void* user) {
        for (size_t i = 0; i < n; ++i)
          static_cast<std::vector<rocprofiler_pc_sampling_configuration_t>*>(user)->push_back(c[i]);
        return ROCPROFILER_STATUS_SUCCESS;
      },
      &out);
  return out;
}

void on_code_object(rocprofiler_callback_tracing_record_t r, rocprofiler_user_data_t*, void*) {
  if (r.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
  if (r.operation == ROCPROFILER_CODE_OBJECT_LOAD) {
    auto* l = static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(r.payload);
    seen().code_object_id = l->code_object_id;
    seen().load_base = l->load_base;
    seen().load_size = l->load_size;
  }
  if (r.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) {
    auto* k = static_cast<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(r.payload);
    if (std::string(k->kernel_name) == "counted_mix") seen().kernel_address = k->kernel_address.value;
  }
}

void on_records(rocprofiler_context_id_t, rocprofiler_buffer_id_t, rocprofiler_record_header_t** headers, size_t n,
                void*, uint64_t) {
  for (size_t i = 0; i < n; ++i) {
    const rocprofiler_record_header_t* h = headers[i];
    if (h->category == ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING) {
      if (h->kind == ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE)
        seen().samples.push_back(*static_cast<rocprofiler_pc_sampling_record_host_trap_v0_t*>(h->payload));
      else
        ++seen().other_kinds;
    } else if (h->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
               h->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {
      seen().dispatches.push_back(*static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(h->payload));
    }
  }
}

int tool_init(rocprofiler_client_finalize_t, void*) {
  Seen& s = seen();
  if (const char* v = std::getenv("VGPU_TEST_PCS_INTERVAL")) s.interval = std::strtoull(v, nullptr, 10);
  rocprofiler_query_available_agents(
      ROCPROFILER_AGENT_INFO_VERSION_0,
      [](rocprofiler_agent_version_t, const void** agents, size_t n, void*) {
        for (size_t i = 0; i < n; ++i) seen().agents.push_back(*static_cast<const rocprofiler_agent_v0_t*>(agents[i]));
        return ROCPROFILER_STATUS_SUCCESS;
      },
      sizeof(rocprofiler_agent_v0_t), nullptr);
  for (const auto& a : s.agents) {
    if (a.type == ROCPROFILER_AGENT_TYPE_GPU && !s.gpu.handle) s.gpu = a.id;
    if (a.type == ROCPROFILER_AGENT_TYPE_CPU && !s.cpu.handle) s.cpu = a.id;
  }
  rocprofiler_status_t st;
  s.before = query(s.gpu, &st);
  s.cpu_configs = query(s.cpu, &st);
  query({9999}, &s.unknown_agent);

  rocprofiler_context_id_t objects{}, tracing{}, sampling{}, other{};
  rocprofiler_create_context(&objects);
  rocprofiler_configure_callback_tracing_service(objects, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
                                                 on_code_object, nullptr);
  rocprofiler_create_context(&tracing);
  rocprofiler_create_buffer(tracing, 1 << 16, 1 << 15, ROCPROFILER_BUFFER_POLICY_LOSSLESS, on_records, nullptr,
                            &s.trace_buffer);
  rocprofiler_configure_buffer_tracing_service(tracing, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0,
                                               s.trace_buffer);

  rocprofiler_create_context(&sampling);
  rocprofiler_create_buffer(sampling, 1 << 20, 1 << 19, ROCPROFILER_BUFFER_POLICY_LOSSLESS, on_records, nullptr,
                            &s.pc_buffer);
  using M = rocprofiler_pc_sampling_method_t;
  using U = rocprofiler_pc_sampling_unit_t;
  auto host = ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP;
  auto insts = ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS;
  // Refused: the stochastic method (no arbiter state to read), the time and
  // cycle units (no clock), an interval of zero, junk, an unknown agent, a CPU.
  s.refused["stochastic"] = rocprofiler_configure_pc_sampling_service(
      sampling, s.gpu, ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC, ROCPROFILER_PC_SAMPLING_UNIT_CYCLES, 1 << 20,
      s.pc_buffer, 0);
  s.refused["time"] = rocprofiler_configure_pc_sampling_service(sampling, s.gpu, host, ROCPROFILER_PC_SAMPLING_UNIT_TIME,
                                                                10000, s.pc_buffer, 0);
  s.refused["cycles"] = rocprofiler_configure_pc_sampling_service(
      sampling, s.gpu, host, ROCPROFILER_PC_SAMPLING_UNIT_CYCLES, 10000, s.pc_buffer, 0);
  s.refused["zero"] = rocprofiler_configure_pc_sampling_service(sampling, s.gpu, host, insts, 0, s.pc_buffer, 0);
  s.refused["method"] = rocprofiler_configure_pc_sampling_service(sampling, s.gpu, static_cast<M>(0), insts, 1,
                                                                  s.pc_buffer, 0);
  s.refused["unit"] = rocprofiler_configure_pc_sampling_service(sampling, s.gpu, host, static_cast<U>(77), 1,
                                                                s.pc_buffer, 0);
  s.refused["flags"] = rocprofiler_configure_pc_sampling_service(sampling, s.gpu, host, insts, 1, s.pc_buffer, 1);
  s.refused["agent"] = rocprofiler_configure_pc_sampling_service(sampling, {9999}, host, insts, 1, s.pc_buffer, 0);
  s.refused["cpu"] = rocprofiler_configure_pc_sampling_service(sampling, s.cpu, host, insts, 1, s.pc_buffer, 0);
  s.refused["buffer"] = rocprofiler_configure_pc_sampling_service(sampling, s.gpu, host, insts, 1, {9999}, 0);
  s.refused["context"] = rocprofiler_configure_pc_sampling_service({9999}, s.gpu, host, insts, 1, s.pc_buffer, 0);
  s.configured = rocprofiler_configure_pc_sampling_service(sampling, s.gpu, host, insts, s.interval, s.pc_buffer, 0);
  s.refused["again"] = rocprofiler_configure_pc_sampling_service(sampling, s.gpu, host, insts, s.interval, s.pc_buffer, 0);
  // One context in the process has PC sampling; another asking is told it is
  // not available. And PC sampling and counter collection do not share a context.
  rocprofiler_create_context(&other);
  s.refused["second context"] = rocprofiler_configure_pc_sampling_service(other, s.gpu, host, insts, s.interval,
                                                                          s.pc_buffer, 0);
  s.refused["counting with sampling"] = rocprofiler_configure_callback_dispatch_counting_service(
      sampling, [](rocprofiler_dispatch_counting_service_data_t, rocprofiler_counter_config_id_t*,
                   rocprofiler_user_data_t*, void*) {},
      nullptr,
      [](rocprofiler_dispatch_counting_service_data_t, rocprofiler_counter_record_t*, size_t,
         rocprofiler_user_data_t, void*) {},
      nullptr);
  s.after = query(s.gpu, &st);
  for (auto c : {objects, tracing, sampling}) rocprofiler_start_context(c);
  return 0;
}

rocprofiler_tool_configure_result_t* configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id) {
  id->name = "test_amd_rocprofiler_pcsampling";
  static rocprofiler_tool_configure_result_t result{sizeof(rocprofiler_tool_configure_result_t), tool_init, nullptr,
                                                    nullptr};
  return &result;
}

// The addresses of counted_mix's instructions, from the listing.
std::vector<uint64_t> listing() {
  std::ifstream in(kData + "counters.gfx942.pcs");
  if (!in) throw vtest::Failure("no listing at " + kData + "counters.gfx942.pcs");
  std::vector<uint64_t> a;
  std::string line;
  while (std::getline(in, line))
    if (!line.empty()) a.push_back(std::stoull(line, nullptr, 16));
  return a;
}

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
  // The kernel still did its work with the sampling on.
  for (size_t i = 0; i < 3; ++i) VCHECK(out[i] != 0.0f);
  rocprofiler_flush_buffer(seen().pc_buffer);
  rocprofiler_flush_buffer(seen().trace_buffer);
}

}  // namespace

VTEST(a_gpu_offers_host_trap_by_instructions_and_a_cpu_nothing) {
  run_once();
  const Seen& s = seen();
  VCHECK(s.gpu.handle != 0 && s.cpu.handle != 0);
  VCHECK_EQ(s.before.size(), size_t{1});
  const auto& c = s.before.at(0);
  VCHECK_EQ(c.size, uint64_t{sizeof c});
  VCHECK_EQ(int(c.method), int(ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP));
  VCHECK_EQ(int(c.unit), int(ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS));
  VCHECK_EQ(c.min_interval, size_t{1});
  VCHECK(c.max_interval > 1000000);
  VCHECK_EQ(c.flags, uint64_t{0});
  VCHECK(s.cpu_configs.empty());
  VCHECK_EQ(int(s.unknown_agent), int(ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND));
}

VTEST(the_configuration_in_use_is_the_one_offered_afterwards) {
  run_once();
  const Seen& s = seen();
  VCHECK_EQ(int(s.configured), int(ROCPROFILER_STATUS_SUCCESS));
  VCHECK_EQ(s.after.size(), size_t{1});
  VCHECK_EQ(s.after.at(0).min_interval, size_t{s.interval});
  VCHECK_EQ(s.after.at(0).max_interval, size_t{s.interval});
}

VTEST(what_cannot_be_sampled_is_refused_not_faked) {
  run_once();
  const auto& r = seen().refused;
  const auto is = [&](const char* what, rocprofiler_status_t want) {
    VCHECK_EQ(int(r.at(what)), int(want));
  };
  is("stochastic", ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE);   // "GPU device does not support requested PC sampling method"
  is("time", ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);      // no clock to count microseconds on
  is("cycles", ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
  is("zero", ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
  is("method", ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
  is("unit", ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
  is("flags", ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
  is("agent", ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND);
  is("cpu", ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND);
  is("buffer", ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND);
  is("context", ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND);
  is("again", ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED);
  is("second context", ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE);   // one context with PC sampling in a process
  is("counting with sampling", ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT);   // not with counter collection
}

VTEST(every_interval_th_instruction_of_every_wave_is_sampled_where_the_listing_puts_it) {
  run_once();
  const Seen& s = seen();
  const std::vector<uint64_t> pcs = listing();
  VCHECK_EQ(pcs.size(), size_t{49});
  VCHECK_EQ(s.other_kinds, 0);
  // A wave is sampled before its 1st, (1 + interval)th, ... instruction.
  std::vector<uint64_t> want;
  for (size_t i = 0; i < pcs.size(); i += s.interval) want.push_back(pcs[i]);
  VCHECK_EQ(s.samples.size(), 3 * want.size());
  // Three work-groups of one wave: each got exactly those, once, in order of
  // sampling (the timestamps are not decreasing).
  std::map<uint32_t, std::vector<uint64_t>> by_group;
  for (const auto& r : s.samples) {
    VCHECK_EQ(r.size, uint64_t{sizeof r});
    VCHECK_EQ(r.workgroup_id.y, 0u);
    VCHECK_EQ(r.workgroup_id.z, 0u);
    VCHECK_EQ(uint32_t(r.wave_in_group), 0u);
    VCHECK_EQ(r.pc.code_object_id, s.code_object_id);
    by_group[r.workgroup_id.x].push_back(r.pc.code_object_offset);
  }
  VCHECK_EQ(by_group.size(), size_t{3});
  for (uint32_t g = 0; g < 3; ++g) {
    auto offsets = by_group.at(g);
    std::sort(offsets.begin(), offsets.end());
    VCHECK(offsets == want);
  }
  for (size_t i = 1; i < s.samples.size(); ++i) VCHECK(s.samples[i].timestamp >= s.samples[i - 1].timestamp);
}

VTEST(a_samples_offset_added_to_the_load_base_is_the_instructions_address) {
  run_once();
  const Seen& s = seen();
  // The first instruction of the kernel is the kernel's address (the symbol
  // callback), and every sampled address is in the object loaded there.
  bool first = false;
  for (const auto& r : s.samples) {
    VCHECK(r.pc.code_object_offset < s.load_size);
    if (s.load_base + r.pc.code_object_offset == s.kernel_address) first = true;
  }
  VCHECK(first);
}

VTEST(a_sample_says_which_wave_and_what_it_could_run) {
  run_once();
  const Seen& s = seen();
  VCHECK(!s.dispatches.empty());
  const auto& d = s.dispatches.at(0);
  // The first instruction a wave issues runs with every lane on.
  const auto first = [&](uint32_t group) {
    for (const auto& r : s.samples)
      if (r.workgroup_id.x == group && r.pc.code_object_offset == 0) return r;
    throw vtest::Failure("no sample of the kernel's first instruction");
  };
  for (uint32_t g = 0; g < 3; ++g) {
    const auto r = first(g);
    VCHECK_EQ(r.exec_mask, ~uint64_t{0});
    // Work-groups go to the compute dies in turn (Dispatch::Layout), and the
    // MI300X has eight.
    VCHECK_EQ(uint32_t(r.hw_id.chiplet), g);
    VCHECK_EQ(uint32_t(r.hw_id.wave_id), 0u);
  }
  // The sample belongs to the dispatch the tool was told of, and to the HIP
  // call that made it, and was taken while it ran.
  for (const auto& r : s.samples) {
    VCHECK_EQ(r.dispatch_id, d.dispatch_info.dispatch_id);
    VCHECK_EQ(r.correlation_id.internal, d.correlation_id.internal);
    VCHECK(r.timestamp >= d.start_timestamp && r.timestamp <= d.end_timestamp);
  }
}

VTEST_MAIN
