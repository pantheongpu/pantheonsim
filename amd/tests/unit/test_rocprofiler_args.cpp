// The arguments of HIP calls, from a profiling tool's side: what rocprofv3 prints for a call
// (amd/tests/data/rocprof_*.expected holds the card's), asked for through
// rocprofiler_iterate_buffer_tracing_record_args and, in callbacks,
// rocprofiler_iterate_callback_tracing_kind_operation_args, plus the table behind them
// (amd/src/rocprofiler_hip_args.hpp). Needs no ROCm.
#include <cstring>
#include <string>
#include <vector>

#include "rocprofiler_hip_args.hpp"
#include "vgpu/rocprofiler_abi.hpp"
#include "vgpu_hip.h"
#include "vtest.hpp"

using namespace vgpu::amd::rocprof;
namespace ha = vgpu::amd::hipargs;
namespace hp = vgpu::amd::hipprof;

extern "C" {
rocprofiler_status_t rocprofiler_force_configure(rocprofiler_configure_func_t);
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
rocprofiler_status_t rocprofiler_iterate_buffer_tracing_record_args(rocprofiler_record_header_t,
                                                                    rocprofiler_buffer_tracing_operation_args_cb_t,
                                                                    void*);
rocprofiler_status_t rocprofiler_iterate_callback_tracing_kind_operation_args(
    rocprofiler_callback_tracing_record_t, rocprofiler_callback_tracing_operation_args_cb_t, int32_t, void*);
}

namespace {

// A pointer prints as <ptr> in the card's expected files: the address is the machine's.
std::string normal(const std::string& v) {
  return v.size() > 2 && v[0] == '0' && v[1] == 'x' ? "<ptr>" : v;
}

struct Seen {
  std::vector<std::string> buffered;                  // name(arg=value, ...) per call, from the buffer records
  std::vector<std::string> entered, exited;           // the same from callbacks, at each phase
  std::vector<std::string> types;                     // typeid name of each argument of the first hipMalloc
  rocprofiler_buffer_id_t buffer{};
  bool initialized = false;
  int failures = 0;   // callbacks run inside the library: they count, the tests check
};
Seen& seen() {
  static Seen s;
  return s;
}

struct Line {
  std::string text;
  bool first = true;
};

std::string buffered_line(const rocprofiler_record_header_t& h) {
  const auto* r = static_cast<const rocprofiler_buffer_tracing_hip_api_ext_record_t*>(h.payload);
  const char* name = nullptr;
  rocprofiler_query_buffer_tracing_kind_operation_name(ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT, r->operation,
                                                       &name, nullptr);
  Line line{std::string(name ? name : "?") + "("};
  if (rocprofiler_iterate_buffer_tracing_record_args(
                h,
                [](rocprofiler_buffer_tracing_kind_t, rocprofiler_tracing_operation_t, uint32_t, const void* const, int32_t,
                   const char*, const char* arg_name, const char* value, void* data) {
                  auto* l = static_cast<Line*>(data);
                  l->text += (l->first ? "" : ", ") + std::string(arg_name) + "=" + normal(value);
                  l->first = false;
                  return 0;
                },
                &line) != ROCPROFILER_STATUS_SUCCESS)
    ++seen().failures;
  return line.text + ")";
}

std::string callback_line(rocprofiler_callback_tracing_record_t r) {
  const char* name = nullptr;
  // The operation names are the buffer kind's.
  rocprofiler_query_buffer_tracing_kind_operation_name(ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, r.operation, &name,
                                                       nullptr);
  Line line{std::string(name ? name : "?") + "("};
  if (rocprofiler_iterate_callback_tracing_kind_operation_args(
                r,
                [](rocprofiler_callback_tracing_kind_t, rocprofiler_tracing_operation_t, uint32_t, const void* const, int32_t,
                   const char*, const char* arg_name, const char* value, int32_t, void* data) {
                  auto* l = static_cast<Line*>(data);
                  l->text += (l->first ? "" : ", ") + std::string(arg_name) + "=" + normal(value);
                  l->first = false;
                  return 0;
                },
                1, &line) != ROCPROFILER_STATUS_SUCCESS)
    ++seen().failures;
  return line.text + ")";
}

void on_records(rocprofiler_context_id_t, rocprofiler_buffer_id_t, rocprofiler_record_header_t** headers, size_t n,
                void*, uint64_t) {
  for (size_t i = 0; i < n; ++i) {
    const rocprofiler_record_header_t* h = headers[i];
    if (h->category != ROCPROFILER_BUFFER_CATEGORY_TRACING || h->kind != ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT)
      continue;
    seen().buffered.push_back(buffered_line(*h));
    if (seen().types.empty()) {
      const auto* r = static_cast<const rocprofiler_buffer_tracing_hip_api_ext_record_t*>(h->payload);
      const char* name = nullptr;
      rocprofiler_query_buffer_tracing_kind_operation_name(ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT,
                                                           r->operation, &name, nullptr);
      if (std::string(name) == "hipMalloc")
        rocprofiler_iterate_buffer_tracing_record_args(
            *h,
            [](rocprofiler_buffer_tracing_kind_t, rocprofiler_tracing_operation_t, uint32_t, const void* const, int32_t,
               const char* type, const char*, const char*, void*) {
              seen().types.push_back(type);
              return 0;
            },
            nullptr);
    }
  }
}

void on_call(rocprofiler_callback_tracing_record_t r, rocprofiler_user_data_t* user, void*) {
  if (r.kind != ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API) return;
  if (r.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
    user->value = 0x5eed;   // comes back at exit
    seen().entered.push_back(callback_line(r));
  } else {
    if (user->value != 0x5eed) ++seen().failures;
    seen().exited.push_back(callback_line(r));
  }
}

int tool_init(rocprofiler_client_finalize_t, void*) {
  Seen& s = seen();
  rocprofiler_context_id_t tracing{}, calls{};
  rocprofiler_create_context(&tracing);
  rocprofiler_create_buffer(tracing, 1 << 16, 1 << 15, ROCPROFILER_BUFFER_POLICY_LOSSLESS, on_records, nullptr, &s.buffer);
  rocprofiler_configure_buffer_tracing_service(tracing, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT, nullptr, 0,
                                               s.buffer);
  rocprofiler_create_context(&calls);
  rocprofiler_configure_callback_tracing_service(calls, ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API, nullptr, 0,
                                                 on_call, nullptr);
  for (auto c : {tracing, calls}) rocprofiler_start_context(c);
  s.initialized = true;
  return 0;
}

rocprofiler_tool_configure_result_t* configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id) {
  id->name = "test_amd_rocprofiler_args";
  static rocprofiler_tool_configure_result_t result{sizeof(rocprofiler_tool_configure_result_t), tool_init, nullptr,
                                                    nullptr};
  return &result;
}

// The calls of amd/tests/rocprof/memset_d2d.cpp and vecadd.cpp, which the card's expected files show.
void run_once() {
  static bool done = false;
  if (done) return;
  done = true;
  VCHECK_EQ(rocprofiler_force_configure(&configure), ROCPROFILER_STATUS_SUCCESS);
  unsigned char host[1024];
  void* p = nullptr;
  VCHECK_EQ(hipMalloc(&p, 1024), hipSuccess);
  VCHECK_EQ(hipMemset(p, 90, 1024), hipSuccess);
  VCHECK_EQ(hipMemcpy(host, p, 1024, hipMemcpyDeviceToHost), hipSuccess);
  hipStream_t s = nullptr;
  VCHECK_EQ(hipStreamCreate(&s), hipSuccess);
  VCHECK_EQ(hipMemsetAsync(p, 17, 512, s), hipSuccess);
  VCHECK_EQ(hipMemcpyAsync(p, host, 512, hipMemcpyHostToDevice, s), hipSuccess);
  VCHECK_EQ(hipStreamSynchronize(s), hipSuccess);
  VCHECK_EQ(hipStreamDestroy(s), hipSuccess);
  VCHECK_EQ(hipDeviceSynchronize(), hipSuccess);
  VCHECK_EQ(hipFree(p), hipSuccess);
  rocprofiler_flush_buffer(seen().buffer);
}

const std::vector<std::string> kExpected = {
    "hipMalloc(ptr=<ptr>, size=1024)",
    "hipMemset(dst=<ptr>, value=90, sizeBytes=1024)",
    "hipMemcpy(dst=<ptr>, src=<ptr>, sizeBytes=1024, kind=DeviceToHost)",
    "hipStreamCreate(stream=<ptr>)",
    "hipMemsetAsync(dst=<ptr>, value=17, sizeBytes=512, stream=<ptr>)",
    "hipMemcpyAsync(dst=<ptr>, src=<ptr>, sizeBytes=512, kind=HostToDevice, stream=<ptr>)",
    "hipStreamSynchronize(stream=<ptr>)",
    "hipStreamDestroy(stream=<ptr>)",
    "hipDeviceSynchronize()",
    "hipFree(ptr=<ptr>)",
};

}  // namespace

VTEST(a_buffer_record_iterates_the_arguments_of_its_call) {
  run_once();
  VCHECK(seen().initialized);
  VCHECK_EQ(seen().failures, 0);
  VCHECK_EQ(seen().buffered.size(), kExpected.size());
  for (size_t i = 0; i < kExpected.size() && i < seen().buffered.size(); ++i)
    VCHECK_EQ(seen().buffered[i], kExpected[i]);
}

VTEST(the_types_are_the_typeid_names_the_card_prints) {
  run_once();
  VCHECK_EQ(seen().types.size(), size_t{2});
  if (seen().types.size() == 2) {
    VCHECK_EQ(seen().types[0], std::string("PPv"));   // void**
    VCHECK_EQ(seen().types[1], std::string("m"));     // size_t
  }
}

VTEST(callbacks_carry_the_arguments_at_entry_and_exit_with_the_users_data) {
  run_once();
  VCHECK_EQ(seen().entered.size(), kExpected.size());
  VCHECK_EQ(seen().exited.size(), kExpected.size());
  for (size_t i = 0; i < kExpected.size() && i < seen().entered.size(); ++i) {
    VCHECK_EQ(seen().entered[i], kExpected[i]);
    VCHECK_EQ(seen().exited[i], kExpected[i]);
  }
}

VTEST(a_record_whose_call_handed_over_no_arguments_has_none_to_show) {
  rocprofiler_buffer_tracing_hip_api_ext_record_t r{};
  r.kind = ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT;
  r.operation = 215;   // hipMalloc
  r.retval[sizeof r.retval - 1] = 1;
  rocprofiler_record_header_t h{};
  h.category = ROCPROFILER_BUFFER_CATEGORY_TRACING;
  h.kind = ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT;
  h.payload = &r;
  int calls = 0;
  VCHECK_EQ(rocprofiler_iterate_buffer_tracing_record_args(
                h, [](auto, auto, auto, auto, auto, auto, auto, auto, void* n) { ++*static_cast<int*>(n); return 0; }, &calls),
            ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE);
  VCHECK_EQ(calls, 0);
}

VTEST(arguments_are_stored_where_rocprofiler_sdks_union_has_them_and_printed_as_rocprofv3_prints) {
  const ha::CallSpec* c = ha::find("hipMemcpy");
  VCHECK(c != nullptr);
  void* dst = nullptr;
  const void* src = reinterpret_cast<const void*>(0x1000);
  size_t bytes = 4096;
  int kind = hipMemcpyHostToDevice;   // an entry point may declare an enumeration as int
  const hp::ArgRef refs[] = {{&dst, 8}, {&src, 8}, {&bytes, 8}, {&kind, 4}};
  uint8_t out[ha::kArgsBytes] = {};
  VCHECK(ha::store(*c, 4, refs, out));
  VCHECK_EQ(c->count, size_t{4});
  VCHECK_EQ(std::string(c->args[0].name), std::string("dst"));
  VCHECK_EQ(c->args[2].offset, 16u);   // dst 0, src 8, sizeBytes 16, kind 24
  VCHECK_EQ(c->args[3].offset, 24u);
  VCHECK_EQ(ha::format(c->args[0], out), std::string("0"));          // a null void* is "0"
  VCHECK_EQ(ha::format(c->args[1], out), std::string("0x1000"));
  VCHECK_EQ(ha::format(c->args[2], out), std::string("4096"));
  VCHECK_EQ(ha::format(c->args[3], out), std::string("HostToDevice"));
  // the wrong number of arguments, or one of the wrong size, stores nothing
  uint8_t none[ha::kArgsBytes] = {};
  VCHECK(!ha::store(*c, 3, refs, none));
  const hp::ArgRef narrow[] = {{&dst, 8}, {&src, 8}, {&kind, 4}, {&kind, 4}};
  VCHECK(ha::store(*c, 4, narrow, none));   // an int for a size_t widens
  const hp::ArgRef wrong[] = {{&dst, 4}, {&src, 8}, {&bytes, 8}, {&kind, 4}};
  VCHECK(!ha::store(*c, 4, wrong, none));   // a 4-byte pointer is not one
  // a dim3 by value prints z, y, x
  const ha::CallSpec* push = ha::find("__hipPushCallConfiguration");
  VCHECK(push != nullptr);
  uint32_t grid[3] = {4, 1, 1}, block[3] = {256, 2, 3};
  size_t shared = 0;
  hipStream_t stream = nullptr;
  const hp::ArgRef prefs[] = {{grid, 12}, {block, 12}, {&shared, 8}, {&stream, 8}};
  uint8_t pout[ha::kArgsBytes] = {};
  VCHECK(ha::store(*push, 4, prefs, pout));
  VCHECK_EQ(ha::format(push->args[0], pout), std::string("{z=1, y=1, x=4}"));
  VCHECK_EQ(ha::format(push->args[1], pout), std::string("{z=3, y=2, x=256}"));
  VCHECK_EQ(ha::format(push->args[3], pout), std::string("0"));   // a null stream is "0"
  // what the card printed for the rest of the kinds (amd/tests/data/rocprof_args.expected)
  const ha::CallSpec* advise = ha::find("hipMemAdvise_v2");
  VCHECK(advise != nullptr);
  const void* ptr = reinterpret_cast<const void*>(0x1000);
  size_t count = 4096;
  int advice = 1;
  int loc[2] = {1, 0};
  const hp::ArgRef arefs[] = {{&ptr, 8}, {&count, 8}, {&advice, 4}, {loc, 8}};
  uint8_t aout[ha::kArgsBytes] = {};
  VCHECK(ha::store(*advise, 4, arefs, aout));
  VCHECK_EQ(ha::format(advise->args[2], aout), std::string("1"));   // an enumeration but hipMemcpyKind: its number
  VCHECK_EQ(ha::format(advise->args[3], aout), std::string("{type=Device, id=0}"));
  loc[0] = 9;
  VCHECK(ha::store(*advise, 4, arefs, aout));
  VCHECK_EQ(ha::format(advise->args[3], aout), std::string("{type=hipMemLocationType_UNKNOWN=9, id=0}"));
  const ha::CallSpec* set3d = ha::find("hipMemset3D");
  VCHECK(set3d != nullptr);
  uint64_t pitched[4] = {0x2000, 256, 64, 2}, extent[3] = {64, 2, 1};
  int value = 5;
  const hp::ArgRef srefs[] = {{pitched, 32}, {&value, 4}, {extent, 24}};
  uint8_t sout[ha::kArgsBytes] = {};
  VCHECK(ha::store(*set3d, 3, srefs, sout));
  VCHECK_EQ(ha::format(set3d->args[0], sout), std::string("{ysize=2, xsize=64, pitch=256}"));   // not the pointer
  VCHECK_EQ(ha::format(set3d->args[2], sout), std::string("{depth=1, height=2, width=64}"));
  const ha::CallSpec* host = ha::find("hipLaunchHostFunc");
  VCHECK(host != nullptr);
  void (*fn)(void*) = +[](void*) {};
  void* user = nullptr;
  const hp::ArgRef hrefs[] = {{&stream, 8}, {&fn, 8}, {&user, 8}};
  uint8_t hout[ha::kArgsBytes] = {};
  VCHECK(ha::store(*host, 3, hrefs, hout));
  VCHECK_EQ(ha::format(host->args[1], hout), std::string("1"));   // a function pointer prints as the bool it converts to
  VCHECK_EQ(ha::format(host->args[2], hout), std::string("0"));
  // a strings: its text, or (null)
  const ha::CallSpec* fn_by_name = ha::find("hipModuleGetFunction");
  VCHECK(fn_by_name != nullptr);
  void** function = nullptr;
  void* module = nullptr;
  const char* kname = "kernel_name";
  const hp::ArgRef frefs[] = {{&function, 8}, {&module, 8}, {&kname, 8}};
  uint8_t fout[ha::kArgsBytes] = {};
  VCHECK(ha::store(*fn_by_name, 3, frefs, fout));
  VCHECK_EQ(ha::format(fn_by_name->args[0], fout), std::string("(null)"));   // a null pointer to a pointer
  VCHECK_EQ(ha::format(fn_by_name->args[1], fout), std::string("0"));        // a null handle
  VCHECK_EQ(ha::format(fn_by_name->args[2], fout), std::string("kernel_name"));
  kname = nullptr;
  VCHECK(ha::store(*fn_by_name, 3, frefs, fout));
  VCHECK_EQ(ha::format(fn_by_name->args[2], fout), std::string("(null)"));
  // hipMemcpyKind: by name, Default and NoCU too, an unknown value as the card prints it
  int unknown_kind = 77;
  const hp::ArgRef krefs[] = {{&dst, 8}, {&src, 8}, {&bytes, 8}, {&unknown_kind, 4}};
  VCHECK(ha::store(*c, 4, krefs, out));
  VCHECK_EQ(ha::format(c->args[3], out), std::string("hipMemcpy_UNKNOWN=77"));
  unknown_kind = 1024;
  VCHECK(ha::store(*c, 4, krefs, out));
  VCHECK_EQ(ha::format(c->args[3], out), std::string("DeviceToDeviceNoCU"));
  unknown_kind = 4;
  VCHECK(ha::store(*c, 4, krefs, out));
  VCHECK_EQ(ha::format(c->args[3], out), std::string("Default"));
  // kinds not measured on a card are refused, not guessed: a struct by value that is not described here
  const ha::CallSpec* ipc = ha::find("hipIpcOpenMemHandle");
  VCHECK(ipc != nullptr);
  uint8_t handle[64] = {};
  void* devptr = nullptr;
  unsigned int ipc_flags = 0;
  const hp::ArgRef irefs[] = {{&devptr, 8}, {handle, 64}, {&ipc_flags, 4}};
  uint8_t iout[ha::kArgsBytes] = {};
  VCHECK(!ha::store(*ipc, 3, irefs, iout));
  VCHECK(ha::find("hipDeviceSynchronize") != nullptr && ha::find("hipDeviceSynchronize")->count == 0);
  VCHECK(ha::find("hipNoSuchCall") == nullptr);
}

VTEST_MAIN
