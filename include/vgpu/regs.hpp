// Register spaces of a VirtualGPU device, backed by a register database.
//
// Every register is declared in registers/<space>.yaml (embedded at build
// time): its offset, width, access, reset value, and what backs it -- a
// constant, the profile, the link, the reliability counts. A read evaluates
// the declaration against the device's state; a write applies the register's
// semantics (read-write bits kept, write-1-to-clear status, base address sizing
// probes). What was written lives in a state file shared by every process on
// the machine, beside the reliability state, so one tool's write is the next
// tool's read. Every access is logged with the process that made it.
//
// Two spaces so far. PCI configuration space, the same layout on every
// vendor's card: the type-0 header, power management, MSI, PCI Express, and
// AER in the extended space. And an AMD GPU's MMIO registers behind BAR5:
// engine status and the SMU mailbox, from the Linux amdgpu headers.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vgpu/telemetry.hpp"

namespace vgpu::regs {

enum class Access { Ro, Rw, Rw1c, Bar };
const char* access_name(Access a);

// "mmio" names a GPU's MMIO space, whichever vendor's: an AMD GPU's registers
// behind BAR5 (from the amdgpu headers), or an NVIDIA GPU's behind BAR0 (as
// measured on real cards -- so far only the measured card's own profile).
enum class Space { Config, AmdMmio, NvidiaMmio };
const char* space_name(Space s);                 // "config", "mmio"
bool parse_space(const std::string& name, Space* out);
uint32_t space_size(Space s);                    // bytes
// The space `s` names on this device: "mmio" is the device's own MMIO space.
Space resolve_space(const telemetry::DeviceSample& d, Space s);

struct Register {
  std::string name;
  uint32_t offset = 0;
  uint32_t width = 0;          // bits: 8, 16, 24 or 32
  Access access = Access::Ro;
  uint32_t reset = 0;
  uint32_t write_mask = 0;     // rw: the bits a write changes
  std::string backing;         // "" when the reset value is the whole story
  std::vector<std::string> fields;
  std::vector<std::string> surfaces;
  std::string status;          // "done" or "model"
  std::string source;          // where the offset comes from, when it is not a standard's
  std::string measured;        // what a real card read, where one has been checked
  std::string capability;      // pm, msi, pcie or aer: found through the capability chain
};

// A space's registers, ordered by offset. Throws on a malformed database,
// naming the entry.
const std::vector<Register>& registers(Space s);
// The register at `offset` exactly, or by name; null when there is none.
const Register* find(Space s, const std::string& name_or_offset);
inline const std::vector<Register>& config_registers() { return registers(Space::Config); }
inline const Register* find_config(const std::string& key) { return find(Space::Config, key); }
// Whether a device has the space: AMD MMIO on an AMD GPU, NVIDIA MMIO where a
// card of its model was measured.
bool has_space(const telemetry::DeviceSample& d, Space s);

inline constexpr uint32_t kConfigSize = 4096;

// One access, as the log keeps it.
struct LogEntry {
  uint64_t seq = 0;
  uint64_t time_ns = 0;   // wall clock
  uint32_t pid = 0;
  std::string process;
  bool write = false;
  uint32_t offset = 0;
  uint32_t size = 0;      // bytes; a whole-space dump is one entry
  uint32_t value = 0;     // for an access of 4 bytes or fewer
};
inline constexpr uint32_t kLogEntries = 256;

// One register space of one device. `d` is a reading of the device with the
// injected faults applied (cli::read_machine, NVML's refresh), which is what
// its link, error and engine state come from. Throws std::invalid_argument for
// a space the device does not have.
class RegisterSpace {
 public:
  RegisterSpace(Space s, const telemetry::DeviceSample& d);
  ~RegisterSpace();
  RegisterSpace(const RegisterSpace&) = delete;
  RegisterSpace& operator=(const RegisterSpace&) = delete;
  Space space() const;

  // An access inside the space -- in configuration space 1, 2 or 4 bytes,
  // naturally aligned; in MMIO 4 bytes, aligned -- or std::invalid_argument.
  uint32_t read(uint32_t offset, uint32_t size);
  void write(uint32_t offset, uint32_t size, uint32_t value);
  // The first `len` bytes of the space (256 for the standard header and
  // capabilities, 4096 with the extended space), logged as one access.
  std::vector<uint8_t> image(uint32_t len);

  // The value a register reads as now, without logging.
  uint32_t value(const Register& r);
  // Where a register is on this device: its offset in the generic layout, or,
  // for a capability's register, wherever the device's capability chain puts
  // that capability. kAbsent when the device does not have it.
  static constexpr uint32_t kAbsent = 0xFFFFFFFFu;
  uint32_t offset_of(const Register& r) const;
  // The register at an offset on this device, or null.
  const Register* at(uint32_t offset) const;
  // "generic", or the captured card whose configuration space this device
  // replays (registers/measurements/).
  const char* layout() const;
  // The first `len` bytes of the space, without logging: for files a session
  // keeps up to date, which no tool has read yet.
  std::vector<uint8_t> image_unlogged(uint32_t len);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// A device's configuration space.
class ConfigSpace : public RegisterSpace {
 public:
  explicit ConfigSpace(const telemetry::DeviceSample& d) : RegisterSpace(Space::Config, d) {}
};

// The SMU messages the mailbox answers, from smu_v13_0_6_ppsmc.h, and what it
// answers with. Any other message is refused as an unknown command.
inline constexpr uint32_t kSmuTestMessage = 0x1, kSmuGetSmuVersion = 0x2, kSmuGetDriverIfVersion = 0x4,
                          kSmuGetMetricsVersion = 0x8;
inline constexpr uint32_t kSmuResultOk = 0x1, kSmuResultUnknownCmd = 0xFE;

// The PCIe link as its registers report it -- Link Capabilities for the
// maximum, Link Status for the link as trained -- read (and logged) the way a
// monitoring tool reads them. nvidia-smi and NVML report the link from here.
struct Link {
  uint32_t gen = 0, width = 0;          // as trained now
  uint32_t max_gen = 0, max_width = 0;  // what the card and slot support
};
Link link(RegisterSpace& cs);

// The files the kernel keeps for a PCI device in sysfs -- config, resource,
// vendor, device, class, subsystem_vendor, subsystem_device, revision, and
// current_ and max_link_speed and _width -- written from the register model
// into `dir`, for a device `d` read with its faults applied.
void write_sysfs_files(const telemetry::DeviceSample& d, const std::string& dir);
// The file names write_sysfs_files writes, for a session to link to.
extern const char* const kSysfsFiles[12];

// The newest accesses to one of a device's register spaces, oldest first.
std::vector<LogEntry> access_log(const std::string& uuid, Space s = Space::Config);

// A base address register's region: what lspci -v and sysfs `resource` show.
struct Bar {
  uint64_t base = 0;
  uint64_t size = 0;         // 0 when the BAR is not implemented
  bool io = false;
  bool is64 = false;
  bool prefetchable = false;
};
// BAR n (0-5) of the device as programmed now. The high half of a 64-bit BAR
// reads as not implemented.
Bar bar(RegisterSpace& cs, const telemetry::DeviceSample& d, int n);

}  // namespace vgpu::regs
