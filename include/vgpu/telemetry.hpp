// Live device telemetry, shared with other processes.
//
// Monitoring tools (nvidia-smi, rocm-smi, `vgpu smi`) run in a *different*
// process from the workload, so telemetry lives in a small file-backed shared
// segment that the running VirtualGPU maps read-write and observers map
// read-only.
//
// WHAT IS REAL AND WHAT IS MODELLED -- this matters, because VirtualGPU does
// not do performance modelling:
//   * memory used ......... REAL. Taken straight from the MemoryManager.
//   * memory total ........ REAL. The framebuffer measured on the real card,
//                            as nvidia-smi and NVML report it -- larger than
//                            the totalGlobalMem CUDA reports by the driver's
//                            reserve (see TelemetryClass).
//   * utilization .......... REAL. The fraction of wall-clock time the device
//                            spent inside kernel launches, plus the share of
//                            that time doing memory traffic.
//   * kernels/bytes ........ REAL counters.
//   * power, temperature,
//     voltage, clocks, fan .. SYNTHETIC. A deterministic first-order model
//                            driven by the real utilization above. They move
//                            the way a real card's would (rise under load, lag
//                            behind it, decay when idle) so that tooling and
//                            dashboards behave, but they are NOT predictions
//                            of any physical device's power or thermals.
// `vgpu smi` labels the synthetic columns; see docs/telemetry.md.
#pragma once

#include <cstdint>
#include <string>

#include "vgpu/profile.hpp"

namespace vgpu::telemetry {

inline constexpr uint32_t kMagic = 0x56475054;  // "VGPT"
inline constexpr uint32_t kVersion = 1;
inline constexpr int kMaxDevices = 16;
inline constexpr int kMaxProcs = 8;

struct ProcSample {
  uint32_t pid;
  uint64_t used_bytes;
  char name[64];
};

// POD, mapped across processes: no pointers, fixed-size fields only.
struct DeviceSample {
  char name[96];
  char uuid[48];
  char bus_id[24];        // "00000000:01:00.0"
  char architecture[24];
  char vendor[16];        // "nvidia" | "amd"
  uint32_t pci_device_id; // PCI device id (vendor id in the high half)
  uint32_t pci_subsystem_id;
  int32_t cc_major, cc_minor;
  uint32_t multiprocessors;

  uint64_t vram_total_bytes;   // REAL
  uint64_t vram_used_bytes;    // REAL
  uint32_t utilization_gpu;    // REAL, percent
  uint32_t utilization_mem;    // REAL, percent
  uint64_t kernels_launched;   // REAL
  uint64_t bytes_moved;        // REAL

  uint32_t temperature_c;      // synthetic
  uint32_t temperature_max_c;  // synthetic (slowdown threshold)
  uint32_t power_mw;           // synthetic
  uint32_t power_limit_mw;     // synthetic (from the profile's class)
  uint32_t voltage_mv;         // synthetic
  uint32_t sm_clock_mhz;       // synthetic
  uint32_t mem_clock_mhz;      // synthetic
  uint32_t sm_clock_max_mhz;   // synthetic
  uint32_t mem_clock_max_mhz;  // synthetic
  uint32_t fan_percent;        // synthetic
  uint32_t perf_state;         // synthetic, 0 = P0 (max) .. 8 = P8 (idle)

  uint32_t proc_count;
  ProcSample procs[kMaxProcs];
};

struct Shared {
  uint32_t magic;
  uint32_t version;
  uint32_t device_count;
  uint32_t writer_pid;
  uint64_t update_seq;   // bumped around every update; readers retry on odd
  int64_t update_ns;
  char driver_version[32];
  char cuda_version[16];
  DeviceSample devices[kMaxDevices];
};

// Directory holding one segment per publishing process. Honors
// VGPU_TELEMETRY_PATH, else $XDG_RUNTIME_DIR/vgpu-telemetry.d, else
// /tmp/vgpu-telemetry-<uid>.d
//
// One file per process matters: a session (`vgpu shell`) and the workloads run
// inside it are separate processes sharing the same virtual machine, and a
// workload exiting must not erase the machine.
std::string default_path();

// Maps the segment read-write and publishes device identity. Called by the
// runtime when it creates devices. Safe to call when telemetry is disabled
// (VGPU_TELEMETRY=0), in which case it becomes a no-op.
class Publisher {
 public:
  Publisher();
  ~Publisher();
  Publisher(const Publisher&) = delete;
  Publisher& operator=(const Publisher&) = delete;

  bool active() const { return shared_ != nullptr; }
  void set_device_count(uint32_t n);
  // Identity fields, written once at device creation.
  DeviceSample* device(uint32_t ordinal);
  void set_driver_version(const std::string& driver, const std::string& cuda);

  // Called by the runtime as work happens.
  void note_memory(uint32_t ordinal, uint64_t used_bytes);
  void note_kernel(uint32_t ordinal, double seconds_busy);
  void note_transfer(uint32_t ordinal, uint64_t bytes, double seconds_busy);
  // Recomputes the derived (synthetic) columns from real activity. Cheap;
  // called on every update and by observers' liveness checks.
  void refresh(uint32_t ordinal);

 private:
  // Per-device integration state. Owned by the Publisher rather than kept in a
  // file-scope global, so two Runtimes in one process (the driver-API and
  // runtime-API shims can both initialize) do not inherit each other's
  // half-integrated power/thermal state.
  struct Accum {
    int64_t window_start_ns = 0;
    double busy_seconds = 0;      // time inside kernels this window
    double mem_busy_seconds = 0;  // time attributable to memory traffic
    double power_w = 0;           // lagged state
    double temp_c = 0;            // lagged state
    bool primed = false;
  };

  void begin_update();
  void end_update();
  Shared* shared_ = nullptr;
  int fd_ = -1;
  size_t size_ = 0;
  std::string path_;
  Accum accum_[kMaxDevices];
};

// Merges every live publisher in `dir` into one view of the machine, the way
// several processes share one physical GPU: device identity comes from the
// publisher that owns the most devices, memory and counters are summed, and
// each contributing process appears in the per-device process list. Returns
// false when no live VirtualGPU process has published anything.
bool read_snapshot(Shared* out, const std::string& dir = default_path());

// Fills in a device's identity and static limits from its profile. Used both
// by a live publisher and by the monitoring tools when nothing is running:
// a real nvidia-smi answers on an idle machine, so this one does too.
void describe_device(const DeviceProfile& p, int ordinal, DeviceSample* out);

// A snapshot of an idle rack built straight from a profile, for when no
// process is publishing. Everything dynamic reads as idle, which is what an
// idle device reports.
Shared idle_snapshot(const DeviceProfile& p, int device_count);

}  // namespace vgpu::telemetry
