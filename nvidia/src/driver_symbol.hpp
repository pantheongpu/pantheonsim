// A function of the driver shim (libcuda.so.1), for a library other than the runtime that calls the driver by name.
//
// `vgpu run --preload` loads the driver privately (RTLD_LOCAL, VGPU_PRELOAD_PRIVATE: #597), so dlsym(RTLD_DEFAULT) does
// not find its functions. The runtime keeps the private driver's handle and looks the driver up through it first
// (driver_dlsym, runtime_api.cpp); it exports that lookup as vgpu_driver_dlsym_v1, and the other shims ask it here
// rather than keeping a second copy of the rule. Without a runtime in the global scope, or when no driver is private,
// this is dlsym(RTLD_DEFAULT) as before.
//
// Before this, CUPTI's callbacks named a placeholder for the current context, and nvshmemx_cumodule_init and
// nvshmemx_culibrary_init failed, under --preload (tests: e2e_preload_torch_like).
#pragma once

#include <dlfcn.h>

#include <atomic>

namespace {   // one copy per shim: a function the shims export would be bound to the first shim's in the global scope

inline void* vgpu_driver_symbol(const char* name) {
  using Lookup = void* (*)(const char*);
  // Kept once found. Not before: a library can be loaded ahead of the runtime and asked before the runtime is there.
  static std::atomic<Lookup> through_runtime{nullptr};
  Lookup lookup = through_runtime.load(std::memory_order_acquire);
  if (!lookup) {
    lookup = reinterpret_cast<Lookup>(dlsym(RTLD_DEFAULT, "vgpu_driver_dlsym_v1"));
    if (lookup) through_runtime.store(lookup, std::memory_order_release);
  }
  return lookup ? lookup(name) : dlsym(RTLD_DEFAULT, name);
}

}  // namespace
