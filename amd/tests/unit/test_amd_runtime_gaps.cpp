// The runtime calls the AMD gap register listed as refused, and what became of each:
//
//   HSA: hsa_amd_signal_wait_any, hsa_amd_ipc_signal_*, hsa_amd_memory_migrate and
//   hsa_amd_memory_pool_can_migrate, the deallocation callbacks, hsa_amd_async_function, the
//   deprecated code object calls (hsa_code_object_deserialize, hsa_executable_create,
//   hsa_executable_load_code_object, hsa_executable_get_symbol, hsa_executable_validate), and
//   the refusals that stay (SPM, the finalizer, queue interception, the AQL profile library, a
//   variable defined from outside), each answering NOT_SUPPORTED by name;
//   HIP: several code objects linked into one module (hipLink*), device-side malloc and free
//   (the hostcall's device-memory service, with code hipcc built), the refusals of a stream
//   capture, and sRGB over linear memory.
//
// One process holds one GPU (VGPU_GPU, VGPU_DEVICE_COUNT); ctest runs this on an MI300X and,
// for the texture check, on a Radeon.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "vgpu/hsa_abi.h"
#include "vgpu_hip.h"
#include "vtest.hpp"

extern "C" {
hipError_t hipLinkCreate(unsigned int, void*, void**, void** state_out);
hipError_t hipLinkAddData(void* link, int type, void* data, size_t size, const char* name, unsigned int, void*, void**);
hipError_t hipLinkAddFile(void* link, int type, const char* path, unsigned int, void*, void**);
hipError_t hipLinkComplete(void* link, void** binary, size_t* size);
hipError_t hipLinkDestroy(void* link);
hipError_t hipModuleGetFunctionCount(unsigned int* count, hipModule_t module);
hipError_t hipStreamBeginCapture(hipStream_t stream, int mode);
hipError_t hipStreamEndCapture(hipStream_t stream, void** graph);
hipError_t hipGraphDestroy(void* graph);
// hipResourceDesc, hipTextureDesc and hipChannelFormatDesc, as HIP's headers lay them out.
struct TestChannelDesc {
  int x, y, z, w, f;
};
struct TestResourceDesc {
  int resType;
  union {
    struct {
      void* devPtr;
      TestChannelDesc desc;
      size_t sizeInBytes;
    } linear;
    struct {
      void* devPtr;
      TestChannelDesc desc;
      size_t width, height, pitchInBytes;
    } pitch2D;
  } res;
};
struct TestTextureDesc {
  int addressMode[3];
  int filterMode;
  int readMode;
  int sRGB;
  float borderColor[4];
  int normalizedCoords;
  unsigned maxAnisotropy;
  int mipmapFilterMode;
  float mipmapLevelBias, minMipmapLevelClamp, maxMipmapLevelClamp;
};
hipError_t hipCreateTextureObject(void** object, const TestResourceDesc* res, const TestTextureDesc* tex, const void* view);
hipError_t hipDestroyTextureObject(void* object);
}

namespace {

std::string data_file(const std::string& name) {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/" + name;
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no fixture at " + path);
  return std::string((std::istreambuf_iterator<char>(in)), {});
}

bool radeon() {
  const char* gpu = std::getenv("VGPU_GPU");
  return gpu && std::string(gpu).find("amd/rx") == 0;
}

// The runtime is started once for the process.
struct Hsa {
  Hsa() { started = hsa_init() == HSA_STATUS_SUCCESS; }
  ~Hsa() {
    if (started) hsa_shut_down();
  }
  bool started = false;
};
Hsa& hsa() {
  static Hsa h;
  return h;
}

hsa_amd_memory_pool_t pool_where(hsa_agent_t agent, bool want_fine, bool gpu) {
  struct Search {
    bool want_fine, gpu;
    hsa_amd_memory_pool_t found{0};
  } search{want_fine, gpu};
  hsa_amd_agent_iterate_memory_pools(
      agent,
      [](hsa_amd_memory_pool_t pool, void* data) {
        auto* s = static_cast<Search*>(data);
        uint32_t flags = 0, segment = 0;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
        if (segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
        const bool fine = (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) != 0;
        if (s->gpu || fine == s->want_fine) s->found = pool;
        return HSA_STATUS_SUCCESS;
      },
      &search);
  return search.found;
}

hsa_agent_t first_gpu() {
  hsa_agent_t gpu{0};
  hsa_iterate_agents(
      [](hsa_agent_t a, void* data) {
        hsa_device_type_t t;
        hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
        if (t == HSA_DEVICE_TYPE_GPU && !static_cast<hsa_agent_t*>(data)->handle) *static_cast<hsa_agent_t*>(data) = a;
        return HSA_STATUS_SUCCESS;
      },
      &gpu);
  return gpu;
}
hsa_agent_t cpu_agent() {
  hsa_agent_t cpu{0};
  hsa_iterate_agents(
      [](hsa_agent_t a, void* data) {
        hsa_device_type_t t;
        hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
        if (t == HSA_DEVICE_TYPE_CPU && !static_cast<hsa_agent_t*>(data)->handle) *static_cast<hsa_agent_t*>(data) = a;
        return HSA_STATUS_SUCCESS;
      },
      &cpu);
  return cpu;
}

}  // namespace

// ---- HSA ---------------------------------------------------------------------------

VTEST(wait_any_returns_the_index_of_the_first_signal_whose_condition_holds) {
  VCHECK(hsa().started);
  hsa_signal_t s[3];
  for (auto& x : s) VCHECK_EQ(hsa_signal_create(5, 0, nullptr, &x), HSA_STATUS_SUCCESS);
  hsa_signal_condition_t conds[3] = {HSA_SIGNAL_CONDITION_EQ, HSA_SIGNAL_CONDITION_EQ, HSA_SIGNAL_CONDITION_LT};
  hsa_signal_value_t values[3] = {0, 0, 3};
  // Nothing holds yet, so a short wait gives up. The header does not say what that returns; this
  // runtime answers UINT32_MAX.
  hsa_signal_value_t got = 77;
  VCHECK_EQ(hsa_amd_signal_wait_any(3, s, conds, values, 2000000, HSA_WAIT_STATE_BLOCKED, &got), UINT32_MAX);
  VCHECK_EQ(got, hsa_signal_value_t{77});   // untouched
  // Signal 1 reaches its value while the wait is on.
  std::thread t([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    hsa_signal_store_screlease(s[1], 0);
  });
  const uint32_t index = hsa_amd_signal_wait_any(3, s, conds, values, UINT64_MAX, HSA_WAIT_STATE_BLOCKED, &got);
  t.join();
  VCHECK_EQ(index, 1u);
  VCHECK_EQ(got, hsa_signal_value_t{0});
  // Several hold: the first in the list is the answer, and the value pointer may be null.
  hsa_signal_store_screlease(s[0], 0);
  VCHECK_EQ(hsa_amd_signal_wait_any(3, s, conds, values, 0, HSA_WAIT_STATE_ACTIVE, nullptr), 0u);
  // A null signal, or no list, is the caller's error and answers the same non-index.
  hsa_signal_t bad[1] = {{0}};
  VCHECK_EQ(hsa_amd_signal_wait_any(1, bad, conds, values, 0, HSA_WAIT_STATE_ACTIVE, nullptr), UINT32_MAX);
  VCHECK_EQ(hsa_amd_signal_wait_any(0, s, conds, values, 0, HSA_WAIT_STATE_ACTIVE, nullptr), UINT32_MAX);
  for (auto& x : s) hsa_signal_destroy(x);
}

VTEST(a_shared_signal_attaches_within_the_process_and_lives_until_every_handle_is_destroyed) {
  VCHECK(hsa().started);
  hsa_signal_t plain, shared;
  VCHECK_EQ(hsa_signal_create(1, 0, nullptr, &plain), HSA_STATUS_SUCCESS);
  VCHECK_EQ(hsa_amd_signal_create(7, 0, nullptr, HSA_AMD_SIGNAL_IPC, &shared), HSA_STATUS_SUCCESS);
  hsa_amd_ipc_signal_t handle;
  // Only a signal made to be shared has a handle.
  VCHECK_EQ(hsa_amd_ipc_signal_create(plain, &handle), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  VCHECK_EQ(hsa_amd_ipc_signal_create(shared, nullptr), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  VCHECK_EQ(hsa_amd_ipc_signal_create(shared, &handle), HSA_STATUS_SUCCESS);
  hsa_signal_t attached;
  VCHECK_EQ(hsa_amd_ipc_signal_attach(&handle, &attached), HSA_STATUS_SUCCESS);
  // It is the same signal: what one handle stores, the other loads.
  hsa_signal_store_screlease(shared, 42);
  VCHECK_EQ(hsa_signal_load_scacquire(attached), hsa_signal_value_t{42});
  // Each attach is one more destroy.
  hsa_signal_t again;
  VCHECK_EQ(hsa_amd_ipc_signal_attach(&handle, &again), HSA_STATUS_SUCCESS);
  VCHECK_EQ(hsa_signal_destroy(shared), HSA_STATUS_SUCCESS);
  VCHECK_EQ(hsa_signal_destroy(attached), HSA_STATUS_SUCCESS);
  hsa_signal_store_screlease(again, 9);   // still alive: one handle is left
  VCHECK_EQ(hsa_signal_load_scacquire(again), hsa_signal_value_t{9});
  VCHECK_EQ(hsa_signal_destroy(again), HSA_STATUS_SUCCESS);
  // Now nothing is behind the handle.
  VCHECK_EQ(hsa_amd_ipc_signal_attach(&handle, &attached), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  hsa_amd_ipc_signal_t junk;
  std::memset(&junk, 0, sizeof junk);
  VCHECK_EQ(hsa_amd_ipc_signal_attach(&junk, &attached), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  hsa_signal_destroy(plain);
}

VTEST(a_shared_signal_from_another_process_is_refused_by_name) {
  VCHECK(hsa().started);
  hsa_signal_t shared;
  VCHECK_EQ(hsa_amd_signal_create(0, 0, nullptr, HSA_AMD_SIGNAL_IPC, &shared), HSA_STATUS_SUCCESS);
  hsa_amd_ipc_signal_t handle;
  VCHECK_EQ(hsa_amd_ipc_signal_create(shared, &handle), HSA_STATUS_SUCCESS);
  handle.handle[1] ^= 0x5a5a;   // the exporting process is another one
  hsa_signal_t attached{0};
  VCHECK_EQ(hsa_amd_ipc_signal_attach(&handle, &attached), HSA_STATUS_ERROR_NOT_SUPPORTED);
  VCHECK_EQ(attached.handle, uint64_t{0});
  hsa_signal_destroy(shared);
}

namespace {
struct Notice {
  void* ptr = nullptr;
  void* user = nullptr;
  int calls = 0;
};
void note(void* ptr, void* user) {
  auto* n = static_cast<Notice*>(user);
  n->ptr = ptr;
  n->user = user;
  ++n->calls;
}
}  // namespace

VTEST(a_deallocation_callback_fires_once_when_the_pool_frees_the_address) {
  VCHECK(hsa().started);
  const hsa_agent_t gpu = first_gpu(), cpu = cpu_agent();
  const hsa_amd_memory_pool_t system = pool_where(cpu, true, false), device = pool_where(gpu, false, true);
  void *sys = nullptr, *dev = nullptr;
  VCHECK_EQ(hsa_amd_memory_pool_allocate(system, 8192, 0, &sys), HSA_STATUS_SUCCESS);
  VCHECK_EQ(hsa_amd_memory_pool_allocate(device, 8192, 0, &dev), HSA_STATUS_SUCCESS);
  Notice a, b, removed;
  VCHECK_EQ(hsa_amd_register_deallocation_callback(sys, note, &a), HSA_STATUS_SUCCESS);
  // An address inside the allocation counts, and is the address the callback is given.
  void* inside = static_cast<char*>(dev) + 100;
  VCHECK_EQ(hsa_amd_register_deallocation_callback(inside, note, &b), HSA_STATUS_SUCCESS);
  VCHECK_EQ(hsa_amd_register_deallocation_callback(dev, note, &removed), HSA_STATUS_SUCCESS);
  VCHECK_EQ(hsa_amd_deregister_deallocation_callback(dev, note), HSA_STATUS_SUCCESS);
  // Not registered (any more): the arguments must be identical to the registration's.
  VCHECK_EQ(hsa_amd_deregister_deallocation_callback(dev, note), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  // No allocation, no callback, no pointer.
  int stack = 0;
  VCHECK_EQ(hsa_amd_register_deallocation_callback(&stack, note, &a), HSA_STATUS_ERROR_INVALID_ALLOCATION);
  VCHECK_EQ(hsa_amd_register_deallocation_callback(sys, nullptr, &a), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  VCHECK_EQ(hsa_amd_register_deallocation_callback(nullptr, note, &a), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  VCHECK_EQ(a.calls, 0);
  VCHECK_EQ(hsa_amd_memory_pool_free(sys), HSA_STATUS_SUCCESS);
  VCHECK_EQ(a.calls, 1);
  VCHECK(a.ptr == sys);
  VCHECK_EQ(b.calls, 0);
  VCHECK_EQ(hsa_amd_memory_pool_free(dev), HSA_STATUS_SUCCESS);
  VCHECK_EQ(b.calls, 1);
  VCHECK(b.ptr == inside);
  VCHECK_EQ(removed.calls, 0);
  // It was dropped as it fired: a new allocation, even at the same address, is not watched.
  void* again = nullptr;
  VCHECK_EQ(hsa_amd_memory_pool_allocate(system, 8192, 0, &again), HSA_STATUS_SUCCESS);
  hsa_amd_memory_pool_free(again);
  VCHECK_EQ(a.calls, 1);
}

VTEST(system_memory_migrates_between_the_cpu_pools_and_the_other_moves_are_refused_by_name) {
  VCHECK(hsa().started);
  const hsa_agent_t gpu = first_gpu(), cpu = cpu_agent();
  const hsa_amd_memory_pool_t fine = pool_where(cpu, true, false), coarse = pool_where(cpu, false, false),
                              device = pool_where(gpu, false, true);
  VCHECK(fine.handle != 0 && coarse.handle != 0 && device.handle != 0 && fine.handle != coarse.handle);
  bool can = false;
  VCHECK_EQ(hsa_amd_memory_pool_can_migrate(fine, coarse, &can), HSA_STATUS_SUCCESS);
  VCHECK(can);
  VCHECK_EQ(hsa_amd_memory_pool_can_migrate(device, device, &can), HSA_STATUS_SUCCESS);
  VCHECK(can);
  VCHECK_EQ(hsa_amd_memory_pool_can_migrate(device, fine, &can), HSA_STATUS_SUCCESS);
  VCHECK(!can);
  VCHECK_EQ(hsa_amd_memory_pool_can_migrate(fine, device, &can), HSA_STATUS_SUCCESS);
  VCHECK(!can);
  VCHECK_EQ(hsa_amd_memory_pool_can_migrate({0xdead0}, fine, &can), HSA_STATUS_ERROR_INVALID_MEMORY_POOL);
  VCHECK_EQ(hsa_amd_memory_pool_can_migrate(fine, coarse, nullptr), HSA_STATUS_ERROR_INVALID_ARGUMENT);

  void* p = nullptr;
  VCHECK_EQ(hsa_amd_memory_pool_allocate(fine, 4096, 0, &p), HSA_STATUS_SUCCESS);
  auto grain = [&](void* ptr) {
    hsa_amd_pointer_info_t info;
    info.size = sizeof info;
    VCHECK_EQ(hsa_amd_pointer_info(ptr, &info, nullptr, nullptr, nullptr), HSA_STATUS_SUCCESS);
    return info.global_flags;
  };
  VCHECK_EQ(grain(p), uint32_t{HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED});
  VCHECK_EQ(hsa_amd_memory_migrate(p, coarse, 1), HSA_STATUS_ERROR_INVALID_ARGUMENT);   // flags must be zero
  VCHECK_EQ(hsa_amd_memory_migrate(p, coarse, 0), HSA_STATUS_SUCCESS);
  VCHECK_EQ(grain(p), uint32_t{HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED});
  VCHECK_EQ(hsa_amd_memory_migrate(p, fine, 0), HSA_STATUS_SUCCESS);
  VCHECK_EQ(grain(p), uint32_t{HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED});
  VCHECK_EQ(hsa_amd_memory_migrate(p, device, 0), HSA_STATUS_ERROR_NOT_SUPPORTED);
  int stack = 0;
  VCHECK_EQ(hsa_amd_memory_migrate(&stack, coarse, 0), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  VCHECK_EQ(hsa_amd_memory_migrate(p, {0xdead0}, 0), HSA_STATUS_ERROR_INVALID_MEMORY_POOL);
  // Memory allocated from the coarse pool starts coarse.
  void* c = nullptr;
  VCHECK_EQ(hsa_amd_memory_pool_allocate(coarse, 4096, 0, &c), HSA_STATUS_SUCCESS);
  VCHECK_EQ(grain(c), uint32_t{HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED});
  hsa_amd_memory_pool_free(c);
  hsa_amd_memory_pool_free(p);

  void* d = nullptr;
  VCHECK_EQ(hsa_amd_memory_pool_allocate(device, 4096, 0, &d), HSA_STATUS_SUCCESS);
  VCHECK_EQ(hsa_amd_memory_migrate(d, device, 0), HSA_STATUS_SUCCESS);   // already there
  VCHECK_EQ(hsa_amd_memory_migrate(d, fine, 0), HSA_STATUS_ERROR_NOT_SUPPORTED);
  hsa_amd_memory_pool_free(d);
}

namespace {
std::atomic<int> g_async_ran{0};
void async_body(void* arg) { g_async_ran.store(*static_cast<int*>(arg)); }
}  // namespace

VTEST(an_async_function_runs_on_a_thread_of_the_runtimes) {
  VCHECK(hsa().started);
  int value = 31;
  VCHECK_EQ(hsa_amd_async_function(async_body, &value), HSA_STATUS_SUCCESS);
  for (int i = 0; i < 500 && g_async_ran.load() == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(2));
  VCHECK_EQ(g_async_ran.load(), 31);
  VCHECK_EQ(hsa_amd_async_function(nullptr, nullptr), HSA_STATUS_ERROR_INVALID_ARGUMENT);
}

VTEST(the_deprecated_code_object_calls_load_a_kernel_through_the_reader_path) {
  if (radeon()) return;   // the fixtures are gfx942 code objects
  VCHECK(hsa().started);
  const hsa_agent_t gpu = first_gpu();
  std::string image = data_file("vector_add.gfx942.hsaco");
  hsa_code_object_t co;
  VCHECK_EQ(hsa_code_object_deserialize(image.data(), image.size(), nullptr, &co), HSA_STATUS_SUCCESS);
  VCHECK_EQ(hsa_code_object_deserialize(nullptr, 0, nullptr, &co), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  hsa_code_object_t second;
  VCHECK_EQ(hsa_code_object_deserialize(image.data(), image.size(), nullptr, &second), HSA_STATUS_SUCCESS);
  hsa_executable_t exec;
  VCHECK_EQ(hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN, nullptr, &exec), HSA_STATUS_SUCCESS);
  VCHECK_EQ(hsa_executable_load_code_object(exec, gpu, co, nullptr), HSA_STATUS_SUCCESS);
  VCHECK_EQ(hsa_executable_load_code_object(exec, gpu, {0x1234}, nullptr), HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  VCHECK_EQ(hsa_executable_freeze(exec, nullptr), HSA_STATUS_SUCCESS);
  uint32_t result = 99;
  VCHECK_EQ(hsa_executable_validate(exec, &result), HSA_STATUS_SUCCESS);
  VCHECK_EQ(result, 0u);
  hsa_executable_symbol_t symbol;
  VCHECK_EQ(hsa_executable_get_symbol(exec, nullptr, "vector_add.kd", gpu, -1, &symbol), HSA_STATUS_SUCCESS);
  uint32_t kind = 99;
  VCHECK_EQ(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind), HSA_STATUS_SUCCESS);
  VCHECK_EQ(kind, uint32_t{HSA_SYMBOL_KIND_KERNEL});
  VCHECK_EQ(hsa_executable_get_symbol(exec, nullptr, "nope.kd", gpu, -1, &symbol), HSA_STATUS_ERROR_INVALID_SYMBOL_NAME);
  // A symbol of a named module: an ELF code object has none.
  VCHECK_EQ(hsa_executable_get_symbol(exec, "module", "vector_add.kd", gpu, -1, &symbol),
            HSA_STATUS_ERROR_INVALID_SYMBOL_NAME);
  VCHECK_EQ(hsa_executable_destroy(exec), HSA_STATUS_SUCCESS);
  // An executable created frozen takes no code.
  VCHECK_EQ(hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_FROZEN, nullptr, &exec), HSA_STATUS_SUCCESS);
  VCHECK_EQ(hsa_executable_load_code_object(exec, gpu, second, nullptr), HSA_STATUS_ERROR_FROZEN_EXECUTABLE);
  hsa_executable_destroy(exec);
  VCHECK_EQ(hsa_code_object_destroy(co), HSA_STATUS_SUCCESS);
  VCHECK_EQ(hsa_code_object_destroy(co), HSA_STATUS_ERROR_INVALID_CODE_OBJECT);   // gone
  VCHECK_EQ(hsa_code_object_destroy(second), HSA_STATUS_SUCCESS);
}

// The calls that stay refused answer NOT_SUPPORTED, and say which they are on stderr.
extern "C" {
hsa_status_t hsa_amd_spm_acquire(hsa_agent_t);
hsa_status_t hsa_amd_spm_release(hsa_agent_t);
hsa_status_t hsa_amd_spm_set_dest_buffer(hsa_agent_t, size_t, uint32_t*, uint32_t*, void*, bool*);
hsa_status_t hsa_system_extension_supported(uint16_t extension, uint16_t major, uint16_t minor, bool* result);
hsa_status_t hsa_ext_program_create();
hsa_status_t hsa_ext_program_finalize();
hsa_status_t hsa_code_object_serialize();
hsa_status_t hsa_amd_queue_intercept_create();
hsa_status_t hsa_amd_queue_intercept_register();
hsa_status_t hsa_ven_amd_aqlprofile_start();
hsa_status_t hsa_ven_amd_aqlprofile_read();
hsa_status_t hsa_amd_image_create();
hsa_status_t hsa_amd_interop_map_buffer();
hsa_status_t hsa_amd_interop_unmap_buffer();
hsa_status_t hsa_executable_agent_global_variable_define();
}

VTEST(what_stays_refused_is_refused_by_name_with_not_supported) {
  VCHECK(hsa().started);
  const hsa_agent_t gpu = first_gpu();
  VCHECK_EQ(hsa_amd_spm_acquire(gpu), HSA_STATUS_ERROR_NOT_SUPPORTED);
  VCHECK_EQ(hsa_amd_spm_release(gpu), HSA_STATUS_ERROR_NOT_SUPPORTED);
  VCHECK_EQ(hsa_amd_spm_set_dest_buffer(gpu, 0, nullptr, nullptr, nullptr, nullptr), HSA_STATUS_ERROR_NOT_SUPPORTED);
  VCHECK_EQ(hsa_ext_program_create(), HSA_STATUS_ERROR_NOT_SUPPORTED);
  VCHECK_EQ(hsa_ext_program_finalize(), HSA_STATUS_ERROR_NOT_SUPPORTED);
  VCHECK_EQ(hsa_code_object_serialize(), HSA_STATUS_ERROR_NOT_SUPPORTED);
  VCHECK_EQ(hsa_amd_queue_intercept_create(), HSA_STATUS_ERROR_NOT_SUPPORTED);
  VCHECK_EQ(hsa_amd_queue_intercept_register(), HSA_STATUS_ERROR_NOT_SUPPORTED);
  VCHECK_EQ(hsa_ven_amd_aqlprofile_start(), HSA_STATUS_ERROR_NOT_SUPPORTED);
  VCHECK_EQ(hsa_ven_amd_aqlprofile_read(), HSA_STATUS_ERROR_NOT_SUPPORTED);
  VCHECK_EQ(hsa_amd_image_create(), HSA_STATUS_ERROR_NOT_SUPPORTED);
  VCHECK_EQ(hsa_amd_interop_map_buffer(), HSA_STATUS_ERROR_NOT_SUPPORTED);
  VCHECK_EQ(hsa_amd_interop_unmap_buffer(), HSA_STATUS_ERROR_NOT_SUPPORTED);
  VCHECK_EQ(hsa_executable_agent_global_variable_define(), HSA_STATUS_ERROR_NOT_SUPPORTED);
  // No finalizer extension is on offer to ask for.
  bool finalizer = true;
  VCHECK_EQ(hsa_system_extension_supported(0 /* HSA_EXTENSION_FINALIZER */, 1, 0, &finalizer), HSA_STATUS_SUCCESS);
  VCHECK(!finalizer);
}

// ---- HIP ---------------------------------------------------------------------------

namespace {
// Links the named fixtures into one binary, as a program would.
std::string link_inputs(const std::vector<std::string>& files, hipError_t* status) {
  void* link = nullptr;
  *status = hipLinkCreate(0, nullptr, nullptr, &link);
  if (*status != hipSuccess) return {};
  std::vector<std::string> keep;
  for (const std::string& f : files) {
    keep.push_back(data_file(f));
    *status = hipLinkAddData(link, 3 /* an object */, keep.back().data(), keep.back().size(), f.c_str(), 0, nullptr, nullptr);
    if (*status != hipSuccess) {
      hipLinkDestroy(link);
      return {};
    }
  }
  void* binary = nullptr;
  size_t size = 0;
  *status = hipLinkComplete(link, &binary, &size);
  std::string out = *status == hipSuccess ? std::string(static_cast<const char*>(binary), size) : std::string();
  hipLinkDestroy(link);
  return out;
}
}  // namespace

VTEST(several_code_objects_link_into_one_module_whose_kernels_launch) {
  if (radeon()) return;   // the fixtures are gfx942 code objects
  hipError_t status = hipSuccess;
  const std::string linked = link_inputs({"vector_add.gfx942.hsaco", "linked.gfx942.hsaco"}, &status);
  VCHECK_EQ(status, hipSuccess);
  VCHECK(!linked.empty());
  hipModule_t module = nullptr;
  VCHECK_EQ(hipModuleLoadData(&module, linked.data()), hipSuccess);
  unsigned count = 0;
  VCHECK_EQ(hipModuleGetFunctionCount(&count, module), hipSuccess);
  VCHECK_EQ(count, 5u);   // vector_add, reduce_sum; scalar, vector, group_z
  hipFunction_t add = nullptr, other = nullptr;
  VCHECK_EQ(hipModuleGetFunction(&add, module, "vector_add"), hipSuccess);
  VCHECK_EQ(hipModuleGetFunction(&other, module, "group_z"), hipSuccess);
  VCHECK(add != other);
  VCHECK_EQ(hipModuleGetFunction(&other, module, "no_such_kernel"), hipErrorNotFound);
  // The first object's kernel runs: out = a + b.
  const int n = 300;
  std::vector<float> a(n), b(n), out(n, -1.0f);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = 2.0f * static_cast<float>(i);
  }
  void *da, *db, *dout;
  VCHECK_EQ(hipMalloc(&da, n * 4), hipSuccess);
  VCHECK_EQ(hipMalloc(&db, n * 4), hipSuccess);
  VCHECK_EQ(hipMalloc(&dout, n * 4), hipSuccess);
  hipMemcpy(da, a.data(), n * 4, hipMemcpyHostToDevice);
  hipMemcpy(db, b.data(), n * 4, hipMemcpyHostToDevice);
  int len = n;
  void* args[] = {&da, &db, &dout, &len};
  VCHECK_EQ(hipModuleLaunchKernel(add, 2, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr), hipSuccess);
  VCHECK_EQ(hipDeviceSynchronize(), hipSuccess);
  hipMemcpy(out.data(), dout, n * 4, hipMemcpyDeviceToHost);
  int wrong = 0;
  for (int i = 0; i < n; ++i) wrong += out[i] != 3.0f * static_cast<float>(i);
  VCHECK_EQ(wrong, 0);
  hipFree(da);
  hipFree(db);
  hipFree(dout);
  VCHECK_EQ(hipModuleUnload(module), hipSuccess);
  VCHECK_EQ(hipModuleUnload(module), hipErrorNotFound);
}

VTEST(a_link_that_would_define_a_kernel_twice_or_takes_a_non_code_object_is_refused_by_name) {
  if (radeon()) return;   // the fixtures are gfx942 code objects
  hipError_t status = hipSuccess;
  link_inputs({"vector_add.gfx942.hsaco", "vector_add.gfx942.hsaco"}, &status);
  VCHECK_EQ(status, hipErrorInvalidValue);   // multiple definition of kernel vector_add
  // PTX or anything else that is not an ELF code object or a bundle.
  void* link = nullptr;
  VCHECK_EQ(hipLinkCreate(0, nullptr, nullptr, &link), hipSuccess);
  char text[64] = ".version 7.0 .target sm_80 // this is not an AMDGPU code object";
  std::string ok = data_file("vector_add.gfx942.hsaco");
  VCHECK_EQ(hipLinkAddData(link, 1 /* PTX */, text, sizeof text, "a.ptx", 0, nullptr, nullptr), hipSuccess);
  VCHECK_EQ(hipLinkAddData(link, 3, ok.data(), ok.size(), "ok.o", 0, nullptr, nullptr), hipSuccess);
  void* binary = nullptr;
  size_t size = 0;
  VCHECK_EQ(hipLinkComplete(link, &binary, &size), hipErrorInvalidImage);
  VCHECK_EQ(hipLinkDestroy(link), hipSuccess);
  // LLVM bitcode needs AMD's compiler library: still refused.
  VCHECK_EQ(hipLinkCreate(0, nullptr, nullptr, &link), hipSuccess);
  VCHECK_EQ(hipLinkAddData(link, 100 /* LLVM bitcode */, text, sizeof text, "a.bc", 0, nullptr, nullptr), hipSuccess);
  VCHECK_EQ(hipLinkComplete(link, &binary, &size), hipErrorNotSupported);
  VCHECK_EQ(hipLinkDestroy(link), hipSuccess);
  // One input comes back as it is.
  const std::string one = link_inputs({"vector_add.gfx942.hsaco"}, &status);
  VCHECK_EQ(status, hipSuccess);
  VCHECK(one == ok);
}

VTEST(device_side_malloc_and_free_run_on_the_hostcalls_device_memory_service) {
  if (radeon()) return;   // the fixtures are gfx942 code objects
  // devmalloc.cpp, built by hipcc: each thread mallocs 64 bytes (from the allocator's slab, which
  // it asks the host for), fills, sums and frees; a block of 3 MiB comes straight from the host.
  const std::string image = data_file("devmalloc.gfx942.hsaco");
  hipModule_t module = nullptr;
  VCHECK_EQ(hipModuleLoadData(&module, image.data()), hipSuccess);
  hipFunction_t small = nullptr, large = nullptr;
  VCHECK_EQ(hipModuleGetFunction(&small, module, "small_blocks"), hipSuccess);
  VCHECK_EQ(hipModuleGetFunction(&large, module, "large_block"), hipSuccess);
  void *dsmall, *dlarge;
  VCHECK_EQ(hipMalloc(&dsmall, 256 * 4), hipSuccess);
  VCHECK_EQ(hipMalloc(&dlarge, 4 * 8), hipSuccess);
  void* a1[] = {&dsmall};
  VCHECK_EQ(hipModuleLaunchKernel(small, 2, 1, 1, 128, 1, 1, 0, nullptr, a1, nullptr), hipSuccess);
  VCHECK_EQ(hipDeviceSynchronize(), hipSuccess);
  int h[256];
  hipMemcpy(h, dsmall, sizeof h, hipMemcpyDeviceToHost);
  int wrong = 0;
  for (int i = 0; i < 256; ++i) wrong += h[i] != 16 * (i % 128) + 120;   // sum of (t + k), k < 16
  VCHECK_EQ(wrong, 0);
  // Launched again: the allocator's state is on the device and carries over.
  VCHECK_EQ(hipModuleLaunchKernel(small, 2, 1, 1, 128, 1, 1, 0, nullptr, a1, nullptr), hipSuccess);
  VCHECK_EQ(hipDeviceSynchronize(), hipSuccess);
  void* a2[] = {&dlarge};
  VCHECK_EQ(hipModuleLaunchKernel(large, 4, 1, 1, 64, 1, 1, 0, nullptr, a2, nullptr), hipSuccess);
  VCHECK_EQ(hipDeviceSynchronize(), hipSuccess);
  long long l[4];
  hipMemcpy(l, dlarge, sizeof l, hipMemcpyDeviceToHost);
  for (int b = 0; b < 4; ++b) VCHECK_EQ(l[b], 36ll + 3 * b);   // three rounds of (round + 1) + (block + 10)
  hipFree(dsmall);
  hipFree(dlarge);
  VCHECK_EQ(hipModuleUnload(module), hipSuccess);
}

VTEST(a_stream_capture_refuses_the_null_stream_and_the_calls_unsafe_during_capture) {
  // The legacy stream cannot be captured.
  VCHECK_EQ(hipStreamBeginCapture(nullptr, 0), hipErrorStreamCaptureUnsupported);
  hipStream_t stream = nullptr;
  VCHECK_EQ(hipStreamCreate(&stream), hipSuccess);
  VCHECK_EQ(hipStreamBeginCapture(stream, 0 /* global */), hipSuccess);
  // A call that synchronizes the device is unsafe while a global capture is on.
  void* p = nullptr;
  VCHECK_EQ(hipMalloc(&p, 64), hipErrorStreamCaptureUnsupported);
  void* graph = nullptr;
  const hipError_t end = hipStreamEndCapture(stream, &graph);
  VCHECK(end != hipSuccess);   // the capture was invalidated
  if (graph) hipGraphDestroy(graph);
  VCHECK_EQ(hipStreamDestroy(stream), hipSuccess);
  // The runtime is usable again.
  VCHECK_EQ(hipMalloc(&p, 64), hipSuccess);
  hipFree(p);
}

VTEST(srgb_over_linear_memory_is_refused_by_name_where_there_are_texture_units) {
  if (!radeon()) return;   // an Instinct has no texture units: every texture call is refused alike
  void* memory = nullptr;
  VCHECK_EQ(hipMalloc(&memory, 4096), hipSuccess);
  TestResourceDesc res{};
  res.resType = 2;
  res.res.linear.devPtr = memory;
  res.res.linear.desc = {8, 8, 8, 8, 1 /* unsigned */};
  res.res.linear.sizeInBytes = 4096;
  TestTextureDesc tex{};
  tex.readMode = 1;   // normalized floats
  tex.sRGB = 1;
  void* object = nullptr;
  VCHECK_EQ(hipCreateTextureObject(&object, &res, &tex, nullptr), hipErrorNotSupported);
  // The same texels without sRGB are fine, and so is sRGB over a 2D pitched allocation.
  tex.sRGB = 0;
  VCHECK_EQ(hipCreateTextureObject(&object, &res, &tex, nullptr), hipSuccess);
  VCHECK_EQ(hipDestroyTextureObject(object), hipSuccess);
  tex.sRGB = 1;
  TestResourceDesc pitched{};
  pitched.resType = 3;
  pitched.res.pitch2D.devPtr = memory;
  pitched.res.pitch2D.desc = {8, 8, 8, 8, 1};
  pitched.res.pitch2D.width = 16;
  pitched.res.pitch2D.height = 16;
  pitched.res.pitch2D.pitchInBytes = 256;
  VCHECK_EQ(hipCreateTextureObject(&object, &pitched, &tex, nullptr), hipSuccess);
  VCHECK_EQ(hipDestroyTextureObject(object), hipSuccess);
  hipFree(memory);
}

VTEST_MAIN
