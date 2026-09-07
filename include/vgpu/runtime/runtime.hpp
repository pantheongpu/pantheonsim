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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/profile.hpp"
#include "vgpu/ptx/ast.hpp"
#include "vgpu/telemetry.hpp"

namespace vgpu::runtime {

class Device {
 public:
  Device(DeviceProfile profile, int ordinal, telemetry::Publisher* telemetry)
      : profile_(std::move(profile)), ordinal_(ordinal), mem_(profile_.vram_bytes, static_cast<uint32_t>(ordinal)),
        telemetry_(telemetry) {
    if (telemetry_) {
      int ord = ordinal_;
      telemetry::Publisher* pub = telemetry_;
      mem_.set_usage_observer(
          [pub, ord](uint64_t used) { pub->note_memory(static_cast<uint32_t>(ord), used); });
    }
  }

  const DeviceProfile& profile() const { return profile_; }
  int ordinal() const { return ordinal_; }
  MemoryManager& memory() { return mem_; }

  // Reports device-busy time to telemetry without running a kernel. Used by
  // `vgpu serve` to present a rack under a chosen synthetic load.
  void note_busy(double seconds) {
    if (telemetry_) telemetry_->note_kernel(static_cast<uint32_t>(ordinal_), seconds);
  }

  // Reports host<->device traffic to telemetry (bytes and the time it took).
  void note_transfer(uint64_t bytes, double seconds) {
    if (telemetry_) telemetry_->note_transfer(static_cast<uint32_t>(ordinal_), bytes, seconds);
  }

  // Loads a PTX module; returns a module handle valid for this device.
  // PTX errors are augmented with the device's profile id.
  uint64_t load_module(const std::string& ptx_src);
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

  void launch(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg,
              const std::vector<std::vector<uint8_t>>& args,
              const exec::SymbolTable* syms = nullptr);

  // Texture and surface objects live for as long as the device, not the
  // module: the host creates them, kernels receive them as parameters, and
  // nothing ties them to the code that reads them.
  exec::TextureTable& textures() { return textures_; }
  const exec::TextureTable& textures() const { return textures_; }

 private:
  DeviceProfile profile_;
  int ordinal_;
  MemoryManager mem_;
  telemetry::Publisher* telemetry_ = nullptr;
  uint64_t next_module_id_ = 1;
  exec::TextureTable textures_;
  struct LoadedModule {
    uint64_t id = 0;
    std::shared_ptr<ptx::Module> mod;
    exec::SymbolTable symbols;          // .global variables -> device VAs
    std::vector<uint64_t> global_vas;   // to free on unload
  };
  std::vector<LoadedModule> modules_;
};

class Runtime {
 public:
  // Creates `device_count` identical virtual devices of the given profile.
  // (Heterogeneous multi-GPU topologies are a planned profile extension.)
  explicit Runtime(const DeviceProfile& profile, int device_count = 1);

  int device_count() const { return static_cast<int>(devices_.size()); }
  Device& device(int ordinal);

 private:
  void publish_identity(const DeviceProfile& profile, int ordinal);

  telemetry::Publisher telemetry_;
  std::vector<std::unique_ptr<Device>> devices_;
};

}  // namespace vgpu::runtime
