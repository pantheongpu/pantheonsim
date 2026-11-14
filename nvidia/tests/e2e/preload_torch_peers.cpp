// The shims other than the runtime that call the driver by name, under `vgpu run --preload`, where the driver is loaded privately
// (VGPU_PRELOAD_PRIVATE, #597) and so is not in the global scope. They looked it up with dlsym(RTLD_DEFAULT), which cannot see it:
//   - CUPTI's callbacks named a placeholder for the current context (current_context in cupti_api.cpp), not the context the
//     program's driver calls find current: a profiler that keys its records by context (PyTorch's Kineto does) saw two;
//   - nvshmemx_cumodule_init and nvshmemx_culibrary_init found no cuModuleGetGlobal / cuLibraryGetGlobal and failed, so a
//     module loaded with the driver API never got its nvshmemi_device_state_d.
// They now ask the runtime's lookup, which knows the private driver (driver_symbol.hpp).
// Run by run_preload_torch_like.sh: <libcudart soname> <libcupti soname or -> <libnvshmem_host soname or ->.
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
int g_fails = 0;
void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++g_fails;
}

struct CallbackData {   // CUpti_CallbackData
  int callbackSite;     // CUPTI_API_ENTER 0, CUPTI_API_EXIT 1
  const char* functionName;
  const void* functionParams;
  void* functionReturnValue;
  const char* symbolName;
  void* context;
  uint32_t contextUid;
  uint64_t* correlationData;
  uint32_t correlationId;
};
constexpr int kRuntimeDomain = 2;   // CUPTI_CB_DOMAIN_RUNTIME_API
void* g_seen_context = nullptr;
int g_seen = 0;
void on_api(void*, int domain, uint32_t, const void* data) {
  const auto* d = static_cast<const CallbackData*>(data);
  if (domain != kRuntimeDomain || !d || d->callbackSite != 0 || std::strcmp(d->functionName, "cudaFree") != 0) return;
  g_seen_context = d->context;
  ++g_seen;
}

// A module that carries the struct NVSHMEM's device code reads (848 bytes: nvshmem_abi.hpp).
const char* kPtx = R"(
.version 8.0
.target sm_75
.address_size 64
.visible .global .align 8 .b8 nvshmemi_device_state_d[848];
.visible .entry nop()
{
  ret;
}
)";
constexpr int kStateVersion = (1 << 16) + 848;
}  // namespace

template <class F>
F sym(void* h, const char* name) {
  return reinterpret_cast<F>(dlsym(h, name));
}

int main(int argc, char** argv) {
  if (argc < 4) return 2;
  void* rt = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!rt) {
    std::printf("cannot load %s: %s\n", argv[1], dlerror());
    return 2;
  }
  const auto set_device = sym<int (*)(int)>(rt, "cudaSetDevice");
  const auto cuda_malloc = sym<int (*)(void**, size_t)>(rt, "cudaMalloc");
  const auto cuda_free = sym<int (*)(void*)>(rt, "cudaFree");
  if (!set_device || !cuda_malloc || !cuda_free) return 2;
  void* p = nullptr;
  check(set_device(0) == 0 && cuda_malloc(&p, 4096) == 0, "the runtime starts and allocates");

  void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
  check(driver != nullptr, "the driver is loaded");
  if (!driver) return 1;
  check(dlsym(RTLD_DEFAULT, "cuCtxGetCurrent") == nullptr, "the driver is private (out of the global scope), as --preload loads it");
  void* ctx = nullptr;
  check(sym<int (*)(void**)>(driver, "cuCtxGetCurrent")(&ctx) == 0 && ctx != nullptr, "the driver has the runtime's context current");

  if (std::strcmp(argv[2], "-") != 0) {
    void* cupti = dlopen(argv[2], RTLD_NOW);
    check(cupti != nullptr, "CUPTI loads");
    if (cupti) {
      using Callback = void (*)(void*, int, uint32_t, const void*);
      const auto subscribe = sym<int (*)(void**, Callback, void*)>(cupti, "cuptiSubscribe");
      const auto enable_domain = sym<int (*)(uint32_t, void*, int)>(cupti, "cuptiEnableDomain");
      const auto unsubscribe = sym<int (*)(void*)>(cupti, "cuptiUnsubscribe");
      void* sub = nullptr;
      check(subscribe && enable_domain && unsubscribe && subscribe(&sub, on_api, nullptr) == 0 &&
                enable_domain(1, sub, kRuntimeDomain) == 0,
            "CUPTI subscribes to the runtime's calls");
      check(cuda_free(p) == 0, "cudaFree");
      p = nullptr;
      if (unsubscribe && sub) unsubscribe(sub);
      check(g_seen == 1, "CUPTI calls back once for cudaFree");
      check(g_seen_context == ctx, "CUPTI's callback names the context the driver has current, not a placeholder");
      if (g_seen_context != ctx) std::printf("     callback %p, driver %p\n", g_seen_context, ctx);
    }
  }
  if (p) cuda_free(p);

  if (std::strcmp(argv[3], "-") != 0) {
    void* shmem = dlopen(argv[3], RTLD_NOW);
    check(shmem != nullptr, "NVSHMEM loads");
    if (shmem) {
      const auto init = sym<int (*)(unsigned, void*)>(shmem, "nvshmemx_hostlib_init_attr");
      const auto fini = sym<void (*)()>(shmem, "nvshmemx_hostlib_finalize");
      const auto module_init = sym<int (*)(void*)>(shmem, "nvshmemx_cumodule_init");
      const auto library_init = sym<int (*)(void*)>(shmem, "nvshmemx_culibrary_init");
      const auto load = sym<int (*)(void**, const void*)>(driver, "cuModuleLoadData");
      const auto unload = sym<int (*)(void*)>(driver, "cuModuleUnload");
      const auto module_global = sym<int (*)(uint64_t*, size_t*, void*, const char*)>(driver, "cuModuleGetGlobal_v2");
      const auto lib_load = sym<int (*)(void**, const void*, void*, void**, unsigned, void*, void**, unsigned)>(driver, "cuLibraryLoadData");
      const auto lib_unload = sym<int (*)(void*)>(driver, "cuLibraryUnload");
      const auto lib_global = sym<int (*)(uint64_t*, size_t*, void*, const char*)>(driver, "cuLibraryGetGlobal");
      const auto dtoh = sym<int (*)(void*, uint64_t, size_t)>(driver, "cuMemcpyDtoH_v2");
      if (!init || !fini || !module_init || !library_init || !load || !unload || !module_global || !lib_load || !lib_unload ||
          !lib_global || !dtoh)
        return 2;
      check(init(0, nullptr) == 0, "NVSHMEM starts (one PE)");
      const auto version_at = [&](uint64_t at) {
        int v = 0;
        return dtoh(&v, at, sizeof v) == 0 ? v : -1;
      };
      void* mod = nullptr;
      uint64_t at = 0;
      size_t n = 0;
      check(load(&mod, kPtx) == 0, "the driver loads a module");
      check(module_init(mod) == 0, "nvshmemx_cumodule_init: success");
      check(module_global(&at, &n, mod, "nvshmemi_device_state_d") == 0 && version_at(at) == kStateVersion,
            "and the module's nvshmemi_device_state_d is filled in");
      if (mod) unload(mod);
      void* lib = nullptr;
      check(lib_load(&lib, kPtx, nullptr, nullptr, 0, nullptr, nullptr, 0) == 0, "the driver loads a library");
      check(library_init(lib) == 0, "nvshmemx_culibrary_init: success");
      check(lib_global(&at, &n, lib, "nvshmemi_device_state_d") == 0 && version_at(at) == kStateVersion,
            "and the library's nvshmemi_device_state_d is filled in");
      if (lib) lib_unload(lib);
      fini();
    }
  }
  if (g_fails) return 1;
  std::printf("ok\n");
  return 0;
}
