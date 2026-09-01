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

namespace vgpu::runtime {

class Device {
 public:
  explicit Device(DeviceProfile profile, int ordinal)
      : profile_(std::move(profile)), ordinal_(ordinal), mem_(profile_.vram_bytes) {}

  const DeviceProfile& profile() const { return profile_; }
  int ordinal() const { return ordinal_; }
  MemoryManager& memory() { return mem_; }

  // Loads a PTX module; returns a module handle valid for this device.
  // PTX errors are augmented with the device's profile id.
  uint64_t load_module(const std::string& ptx_src);
  void unload_module(uint64_t module_id);

  // Looks up a kernel. The returned pointer lives as long as the module.
  const ptx::EntryFn* get_function(uint64_t module_id, const std::string& name) const;

  void launch(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg,
              const std::vector<std::vector<uint8_t>>& args);

 private:
  DeviceProfile profile_;
  int ordinal_;
  MemoryManager mem_;
  uint64_t next_module_id_ = 1;
  // module id -> parsed module (shared_ptr so EntryFn pointers stay valid
  // while a caller holds the module).
  std::vector<std::pair<uint64_t, std::shared_ptr<ptx::Module>>> modules_;
};

class Runtime {
 public:
  // Creates `device_count` identical virtual devices of the given profile.
  // (Heterogeneous multi-GPU topologies are a planned profile extension.)
  explicit Runtime(const DeviceProfile& profile, int device_count = 1);

  int device_count() const { return static_cast<int>(devices_.size()); }
  Device& device(int ordinal);

 private:
  std::vector<std::unique_ptr<Device>> devices_;
};

}  // namespace vgpu::runtime
