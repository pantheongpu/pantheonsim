// VirtualGPU's librocprofiler-sdk on a Radeon card, from a profiling tool's side:
// what rocprofv3 recorded for a program on a real RX 6800 and a real RX 6700 XT
// (ROCm 7.2), checked without ROCm installed. amd/tests/e2e/run_rocprof_case.sh
// compares whole traces with the card's wherever rocprofv3 is; this is the part
// of it that hosted CI can run.
//
// VGPU_GPU names the card (amd/rx6800, or amd/rx6700xt with HSA_OVERRIDE_GFX_VERSION=10.3.0, which makes
// the HIP runtime see gfx1030 while the profiler's agent is still the gfx1031 card).
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
rocprofiler_status_t rocprofiler_flush_buffer(rocprofiler_buffer_id_t);
rocprofiler_status_t rocprofiler_query_buffer_tracing_kind_operation_name(rocprofiler_buffer_tracing_kind_t,
                                                                          rocprofiler_tracing_operation_t,
                                                                          const char**, uint64_t*);
}

namespace {

const std::string kData = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/";

struct Call {
  std::string name;
  uint64_t id;
};
struct Seen {
  std::vector<rocprofiler_agent_v0_t> agents;
  struct Kernel {
    std::string name;
    uint32_t sgprs, vgprs;
  };
  std::vector<Kernel> kernels;
  std::vector<std::string> uris;
  std::vector<rocprofiler_buffer_tracing_memory_copy_record_t> copies;
  std::vector<rocprofiler_buffer_tracing_memory_allocation_record_t> allocations;
  std::vector<Call> calls;
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
  }
}

void on_records(rocprofiler_context_id_t, rocprofiler_buffer_id_t, rocprofiler_record_header_t** headers, size_t n,
                void*, uint64_t) {
  for (size_t i = 0; i < n; ++i) {
    const rocprofiler_record_header_t* h = headers[i];
    if (h->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) continue;
    if (h->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
      seen().copies.push_back(*static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(h->payload));
    if (h->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION)
      seen().allocations.push_back(*static_cast<rocprofiler_buffer_tracing_memory_allocation_record_t*>(h->payload));
    if (h->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT) {
      auto* r = static_cast<rocprofiler_buffer_tracing_hip_api_ext_record_t*>(h->payload);
      const char* name = nullptr;
      rocprofiler_query_buffer_tracing_kind_operation_name(ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT, r->operation,
                                                           &name, nullptr);
      seen().calls.push_back({name ? name : "?", r->correlation_id.internal});
    }
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
  rocprofiler_context_id_t objects{}, tracing{};
  rocprofiler_create_context(&objects);
  rocprofiler_configure_callback_tracing_service(objects, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
                                                 on_code_object, nullptr);
  rocprofiler_create_context(&tracing);
  rocprofiler_create_buffer(tracing, 1 << 16, 1 << 15, ROCPROFILER_BUFFER_POLICY_LOSSLESS, on_records, nullptr,
                            &s.buffer);
  for (auto kind : {ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION,
                    ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT})
    rocprofiler_configure_buffer_tracing_service(tracing, kind, nullptr, 0, s.buffer);
  for (auto c : {objects, tracing}) rocprofiler_start_context(c);
  return 0;
}

rocprofiler_tool_configure_result_t* configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id) {
  id->name = "test_amd_rocprofiler_rdna";
  static rocprofiler_tool_configure_result_t result{sizeof(rocprofiler_tool_configure_result_t), tool_init, nullptr,
                                                    nullptr};
  return &result;
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
  VCHECK_EQ(hipFree(a), hipSuccess);
  VCHECK_EQ(hipFree(b), hipSuccess);
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

VTEST(a_code_object_is_named_by_a_uri_and_its_kernels_by_their_descriptors) {
  run_once();
  const Seen& s = seen();
  VCHECK_EQ(s.uris.size(), size_t{1});
  // The string must outlive the callback (rocprofv3 keeps the pointer); it is read here, after the program ran.
  VCHECK(s.uris[0].rfind("memory://", 0) == 0 || s.uris[0].rfind("file://", 0) == 0);
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
      if (by == "hipMalloc" || by == "hipHostMalloc") VCHECK_EQ(r.allocation_size, uint64_t{1} << 20);
      if (by == "hipMalloc") ++allocated;
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

VTEST_MAIN
