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
// The first space is PCI configuration space, the same layout on every
// vendor's card: the type-0 header, power management, MSI, PCI Express, and
// AER in the extended space.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vgpu/telemetry.hpp"

namespace vgpu::regs {

enum class Access { Ro, Rw, Rw1c, Bar };
const char* access_name(Access a);

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
};

// The configuration-space registers, ordered by offset. Throws on a malformed
// database, naming the entry.
const std::vector<Register>& config_registers();
// The register at `offset` exactly, or by name; null when there is none.
const Register* find_config(const std::string& name_or_offset);

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

// One device's configuration space. `d` is a reading of the device with the
// injected faults applied (cli::read_machine, NVML's refresh), which is what
// its link and error state come from.
class ConfigSpace {
 public:
  explicit ConfigSpace(const telemetry::DeviceSample& d);
  ~ConfigSpace();
  ConfigSpace(const ConfigSpace&) = delete;
  ConfigSpace& operator=(const ConfigSpace&) = delete;

  // A 1-, 2- or 4-byte access, naturally aligned, inside the space; throws
  // std::invalid_argument otherwise.
  uint32_t read(uint32_t offset, uint32_t size);
  void write(uint32_t offset, uint32_t size, uint32_t value);
  // The first `len` bytes of the space (256 for the standard header and
  // capabilities, 4096 with the extended space), logged as one access.
  std::vector<uint8_t> image(uint32_t len);

  // The value a register reads as now, without logging.
  uint32_t value(const Register& r);
  // The first `len` bytes of the space, without logging: for files a session
  // keeps up to date, which no tool has read yet.
  std::vector<uint8_t> image_unlogged(uint32_t len);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The PCIe link as its registers report it -- Link Capabilities for the
// maximum, Link Status for the link as trained -- read (and logged) the way a
// monitoring tool reads them. nvidia-smi and NVML report the link from here.
struct Link {
  uint32_t gen = 0, width = 0;          // as trained now
  uint32_t max_gen = 0, max_width = 0;  // what the card and slot support
};
Link link(ConfigSpace& cs);

// The files the kernel keeps for a PCI device in sysfs -- config, resource,
// vendor, device, class, subsystem_vendor, subsystem_device, revision, and
// current_ and max_link_speed and _width -- written from the register model
// into `dir`, for a device `d` read with its faults applied.
void write_sysfs_files(const telemetry::DeviceSample& d, const std::string& dir);
// The file names write_sysfs_files writes, for a session to link to.
extern const char* const kSysfsFiles[12];

// The newest accesses to a device's registers, oldest first.
std::vector<LogEntry> access_log(const std::string& uuid);

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
Bar bar(ConfigSpace& cs, const telemetry::DeviceSample& d, int n);

}  // namespace vgpu::regs
