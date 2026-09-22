// What differs between vendors in the register model (vgpu/regs.hpp). Internal:
// for the engine and the vendors' register code, not for tools.
//
// src/core/regs.cpp is the part every PCI device shares: the register
// databases and their semantics, configuration space and its capability
// chains, BAR sizing, AER status from the reliability counts, the shared
// state and its log, and each GPU model's register file. What only one
// vendor's cards have lives in that vendor's directory and reaches the engine
// through a Vendor: amd/src/regs.cpp, nvidia/src/regs.cpp.
#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "vgpu/embedded_config_images.hpp"
#include "vgpu/regs.hpp"
#include "vgpu/telemetry.hpp"

namespace vgpu::regs::vendor {

// One base address register as firmware left it.
struct BarLayout {
  uint64_t size = 0;   // 0: not implemented
  bool io = false, is64 = false, pref = false;
  bool high = false;   // the upper half of the 64-bit BAR before it
  uint64_t base = 0;   // where firmware put it
};

// Where firmware put a device's BARs: a 32-bit and a 64-bit window per slot,
// the slot being the device's bus number less one.
struct Windows {
  uint64_t slot, mmio32, mmio64;
};
Windows windows(const telemetry::DeviceSample& d);
uint64_t pow2_at_least(uint64_t v);

// The words of the shared register state a vendor keeps its own device logic
// in (the SMU mailbox, on AMD), shared by every process on the machine.
inline constexpr uint32_t kStateWords = 4;

// What a vendor's backing is evaluated against.
struct Context {
  const telemetry::DeviceSample& d;
  uint32_t base;       // the register's reset value on this device
  uint64_t* words;     // kStateWords of shared state
};

struct Vendor {
  const char* name;             // telemetry::DeviceSample::vendor
  Space mmio;                   // the vendor's MMIO space
  const char* mmio_database;    // its embedded register database
  const char* mmio_origin;      // the database's path, for errors
  uint32_t mmio_size;           // bytes
  uint32_t mmio_undeclared;     // what a dword no register declares reads as
  bool (*has_mmio)(const telemetry::DeviceSample& d);
  std::array<BarLayout, 6> (*bars)(const telemetry::DeviceSample& d);
  // The captured configuration space the device replays, or null.
  const embedded::ConfigImage* (*captured_config)(const telemetry::DeviceSample& d);
  // The generation Link Status reports now, from the link as trained.
  uint32_t (*link_gen)(const telemetry::DeviceSample& d);
  // A backing the vendor owns -- its profile.* values and its live device
  // state: true, with the value in *out, when `key` is the vendor's.
  bool (*backed)(const std::string& key, const Context& c, uint32_t* out);
  // A write to a register the vendor's device logic backs (a read-write
  // register with a live backing): true when `key` is the vendor's.
  bool (*write)(const std::string& key, const Context& c, uint32_t value);
  // The vendor's driver files in a device's sysfs directory, beside the PCI
  // files every device has.
  void (*sysfs)(const telemetry::DeviceSample& d, const std::string& dir);
};

const Vendor& amd();
const Vendor& nvidia();
// The device's vendor; NVIDIA's for anything that is not AMD's.
const Vendor& of(const telemetry::DeviceSample& d);
// The vendor whose MMIO space `s` is; null for configuration space.
const Vendor* owning(Space s);

}  // namespace vgpu::regs::vendor
