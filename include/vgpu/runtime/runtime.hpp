// The VirtualGPU runtime: devices, memory, modules, launches.
//
// This is the internal, C++-typed layer that both the CLI and the C driver-API
// shim (libvgpucuda) sit on. It models CUDA's *primary context* world: one
// context per device, owned by the device. Explicit context stacks can be
// layered on later without changing this core.
//
// Everything is synchronous and deterministic today: launches and copies
// complete before returning. Streams (M8) will introduce async on top of this
// without changing these semantics for the default stream.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <list>
#include <map>
#include <unordered_map>
#include <vector>

#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/profile.hpp"
#include "vgpu/ptx/ast.hpp"
#include "vgpu/telemetry.hpp"

namespace vgpu::runtime {

class Device {
 public:
  // Defined with the fault hook it installs, which is only complete there.
  // `physical` is which of the machine's devices this one is, where a
  // visible-devices list shows the program only some of them in another
  // order (-1: the same as its ordinal). It names the device everywhere
  // outside the program -- its identity, its telemetry, the faults armed for
  // it, and its address window, so processes shown it differently agree on
  // where its memory is.
  Device(DeviceProfile profile, int ordinal, telemetry::Publisher* telemetry, int physical = -1);

  ~Device();

  const DeviceProfile& profile() const { return profile_; }
  int ordinal() const { return ordinal_; }
  int physical() const { return physical_; }
  // How many devices the machine has, for device-side cudaGetDeviceCount.
  void set_device_count(int n) { device_count_ = n; }
  MemoryManager& memory() { return mem_; }

  // Reports device-busy time to telemetry without running a kernel. Used by
  // `vgpu serve` to present a rack under a chosen synthetic load.
  void note_busy(double seconds) {
    if (telemetry_) telemetry_->note_kernel(static_cast<uint32_t>(physical_), seconds);
  }

  // Reports host<->device traffic to telemetry (bytes and the time it took).
  void note_transfer(uint64_t bytes, double seconds) {
    if (telemetry_) telemetry_->note_transfer(static_cast<uint32_t>(physical_), bytes, seconds);
  }

  // Destroys everything this device holds -- loaded modules and their globals,
  // texture objects, every allocation -- as cudaDeviceReset documents. Handles
  // issued before the reset are invalid afterwards; the device itself stays
  // usable, and modules load again on next use.
  void reset();

  // Loads a PTX module; returns a module handle valid for this device.
  // PTX errors are augmented with the device's profile id.
  uint64_t load_module(const std::string& ptx_src);
  // Loads a cubin (the SASS a fatbin carries for this architecture). Its
  // kernels are looked up and launched like a PTX module's.
  uint64_t load_cubin(const uint8_t* image, size_t size);
  void unload_module(uint64_t module_id);

  // Looks up a kernel. The returned pointer lives as long as the module.
  const ptx::EntryFn* get_function(uint64_t module_id, const std::string& name) const;

  // The virtual architecture the module was compiled for, from its ".target"
  // directive: 86 for ".target sm_86". Zero when the module does not say.
  // Reported through cudaFuncGetAttributes::ptxVersion, which CUB reads to pick
  // a kernel policy, so it has to be an architecture and not an ISA version.
  int module_arch(uint64_t module_id) const;

  // The module's global-variable addresses (valid while the module is loaded).
  const exec::SymbolTable* symbols(uint64_t module_id) const;

  // Whether this device remembers what it works out for a launch (the table of every loaded kernel, and, in the
  // API shims, which function a kernel handle names) instead of working it out again at every launch. On unless
  // VGPU_LAUNCH_CACHE=0; the answer must not depend on it, which is what turning it off is for (A/B timing, and
  // tests/unit/test_launch_cache.cpp, which runs one sequence both ways and requires the same results).
  bool launch_cache() const { return launch_cache_; }
  void set_launch_cache(bool on) {
    launch_cache_ = on;
    forget_kernel_table();
  }

  // Counts the modules this device has unloaded. A caller that keeps what
  // get_function returned for a module id may keep it for as long as this has
  // not changed: module ids are never reused, so a function of a module that is
  // still loaded stays where it is, and an unload anywhere on the device is
  // the one thing that can take one away.
  uint64_t unload_epoch() const { return unload_epoch_.load(std::memory_order_acquire); }
  // A module's global variable by name: its address and its declared size.
  // False when the module declares no global of that name.
  bool global(uint64_t module_id, const std::string& name, uint64_t* addr, uint64_t* size) const;
  // Points a module's global at `addr` instead of the device memory it was
  // given at load: a __managed__ variable lives in host memory every device
  // maps, at one address for the host and all devices, and the runtime moves
  // each device's copy of the module onto it. Kernels launched afterwards use
  // the new address, as do globals initialised with this one's address.
  void rebind_global(uint64_t module_id, const std::string& name, uint64_t addr);
  // The names of a module's __managed__ globals (".attribute(.managed)"), which
  // whoever loaded the module moves onto host-shared memory with rebind_global.
  std::vector<std::string> managed_globals(uint64_t module_id) const;
  // Whether the module defines a kernel of that name.
  bool has_kernel(uint64_t module_id, const std::string& name) const;
  // The names of the kernels the module defines, in the order it declares them.
  // Empty for a module that is not loaded.
  std::vector<std::string> kernel_names(uint64_t module_id) const;

  void launch(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg,
              const std::vector<std::vector<uint8_t>>& args,
              const exec::SymbolTable* syms = nullptr);

  // Texture and surface objects live for as long as the device, not the
  // module: the host creates them, kernels receive them as parameters, and
  // nothing ties them to the code that reads them.
  exec::TextureTable& textures() { return textures_; }
  const exec::TextureTable& textures() const { return textures_; }

 private:
  // Delivers faults armed with `vgpu fault arm` to this device: to its kernel
  // accesses, and a hang to its next launch. It also logs the Xid a kernel's
  // own fault raises.
  void install_fault_hook();
  void run_kernel(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg,
                  const std::vector<std::vector<uint8_t>>& args, const exec::SymbolTable* syms);
  struct LoadedModule;
  const ptx::EntryFn* lazy_function(LoadedModule& lm, const std::string& name) const;
  std::unique_ptr<class FaultHook> fault_;
  DeviceProfile profile_;
  int ordinal_;
  int physical_;
  int device_count_ = 1;
  MemoryManager mem_;
  telemetry::Publisher* telemetry_ = nullptr;
  uint64_t next_module_id_ = 1;
  uint64_t next_kernel_va_ = kKernelVaBase;   // every kernel ever loaded has its own
  exec::TextureTable textures_;
  // A large module is not parsed whole (see load_module): its kernels are
  // parsed one at a time, the first time each is looked up.
  struct LazyModule;
  struct LoadedModule {
    uint64_t id = 0;
    std::shared_ptr<ptx::Module> mod;   // the whole module, or only its declarations when lazy
    exec::SymbolTable symbols;          // .global variables -> device VAs
    std::vector<uint64_t> global_vas;   // to free on unload
    // address -> kernel; a lazy module's kernels are null until parsed
    std::vector<std::pair<uint64_t, const ptx::EntryFn*>> kernels;
    std::shared_ptr<LazyModule> lazy;
    std::shared_ptr<sass::Module> sass;   // a cubin's: its code, banks and variables
    // A cubin's kernels by name, to the first entry of that name (ptx::Module::find_entry's answer).
    // A cubin can carry thousands, and looking one up by name was a search of all of them.
    std::unordered_map<std::string, size_t> entry_index;
  };
  // A list, not a vector: a function handle keeps a pointer to its module's
  // symbol table, and a vector moved every module whenever another loaded, so
  // a kernel launched after that read a freed table (PyTorch's jiterator
  // loads a module per kernel and launches earlier ones again).
  // mutable: a lazy module parses kernels on lookup.
  mutable std::list<LoadedModule> modules_;
  // modules_ by module id (list elements do not move), so the per-launch lookups are not a walk of every
  // module the program has loaded. Kept with modules_: load_module, load_cubin and unload_module.
  std::unordered_map<uint64_t, std::list<LoadedModule>::iterator> module_index_;
  LoadedModule* find_module(uint64_t module_id) const {
    const auto it = module_index_.find(module_id);
    return it == module_index_.end() ? nullptr : &*it->second;
  }
  // Every kernel loaded on the device, by address, which a launch hands the
  // kernel for the child grids it launches (LaunchConfig::kernels). It was
  // rebuilt, a node per kernel, at every launch, so a launch cost as much as
  // the number of kernels the program had loaded. Now it is built when
  // something has changed what it holds -- a module loaded or unloaded, or a
  // lazy module's kernel parsed -- and a launch takes a reference to the
  // current one (a snapshot: a launch running while another thread loads a
  // module keeps the table it started with).
  std::shared_ptr<const exec::KernelTable> kernel_table() const;
  void forget_kernel_table() const;
  std::atomic<uint64_t> unload_epoch_{0};
  bool launch_cache_ = true;
  mutable std::mutex kernel_table_mu_;
  mutable std::shared_ptr<const exec::KernelTable> kernel_table_;
};

// A range of host memory CUDA knows about, and the device that was current
// when it was set up (the one whose cudaDeviceReset releases it). `flags` are
// the ones the caller passed, which cuMemHostGetFlags and cudaHostGetFlags
// report back.
struct HostRange {
  size_t size = 0;
  int device = 0;
  unsigned flags = 0;
};

class Runtime;
// The process's simulated machine, made from the environment (VGPU_GPU,
// VGPU_DEVICE_COUNT) the first time either CUDA library asks, and the same
// one for both (shared_runtime.cpp).
Runtime* shared_runtime();
// The lock both CUDA libraries take around every API call, so that calls into
// the one machine from either never overlap (shared_runtime.cpp).
std::recursive_mutex& shared_api_mutex();

class Runtime {
 public:
  // Creates `device_count` identical virtual devices of the given profile.
  // (Heterogeneous multi-GPU topologies are a planned profile extension.)
  // `physical`, where given, says which machine device each one is (see
  // Device), and `machine` how many the machine has.
  explicit Runtime(const DeviceProfile& profile, int device_count = 1, std::vector<int> physical = {},
                   int machine = 0);

  int device_count() const { return static_cast<int>(devices_.size()); }
  Device& device(int ordinal);

  // Host memory registered with cudaHostRegister or cuMemHostRegister, by its
  // base. One record for both libraries, as the card keeps one: memory either
  // API registered is already registered to the other, and either may
  // unregister it. Callers hold shared_api_mutex().
  std::map<void*, HostRange>& host_registrations() { return host_registrations_; }
  // Pinned host memory either library allocated (cudaMallocHost, cudaHostAlloc,
  // cuMemHostAlloc), by its base: one record, as the card keeps one, so the
  // driver reports the flags of memory the runtime pinned and either library
  // frees what the other allocated. Callers hold shared_api_mutex().
  std::map<void*, HostRange>& host_allocations() { return host_allocations_; }

  // The fault a kernel left the context with, as the CUDA error code it is
  // reported as (the runtime's and the driver's codes for these are the same
  // numbers), or 0. A kernel that hits an illegal address, a trap or a failed
  // assert leaves the context unusable, and every later call that needs it
  // fails with that code until a reset -- whichever library launched the
  // kernel and whichever is asked, since they share one context. Atomic: it is
  // read without the API lock.
  // The error every CUDA call answers when CUDA_VISIBLE_DEVICES shows the
  // program no device (cudaErrorNoDevice) or a bad list (cudaErrorInvalidDevice):
  // 0 otherwise (runtime/visible_devices.hpp). A machine still exists then,
  // one device of it, so the calls that report the error have something to ask.
  int visibility_error() const { return visibility_error_; }
  void set_visibility_error(int code) { visibility_error_ = code; }
  int context_fault() const { return context_fault_.load(); }
  void set_context_fault(int code) { context_fault_.store(code); }

 private:
  void publish_identity(const DeviceProfile& profile, int ordinal);

  telemetry::Publisher telemetry_;
  std::vector<std::unique_ptr<Device>> devices_;
  std::map<void*, HostRange> host_registrations_;
  std::map<void*, HostRange> host_allocations_;
  std::atomic<int> context_fault_{0};
  int visibility_error_ = 0;
};

}  // namespace vgpu::runtime
