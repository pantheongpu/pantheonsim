// Reliability state of a simulated GPU: the ECC counts, retired pages, remapped
// rows and PCIe transport errors that nvidia-smi, NVML and rocm-smi report and
// that health tools act on.
//
// It outlives the processes that change it -- a real card's error counts do not
// reset when the program that caused them exits -- so it lives in files, not in
// a process's telemetry segment. Two lifetimes, as on a real card:
//
//   volatile   since the driver loaded. Kept in the machine's runtime directory
//              (telemetry::default_path), so a new session starts at zero.
//   aggregate  the life of the card. Kept in the state directory, keyed by the
//              device UUID, which is stable per profile and slot:
//              VGPU_STATE_DIR, else $XDG_STATE_HOME/vgpu, else
//              ~/.local/state/vgpu.
//
// Counts come only from injection (`vgpu fault`, or the API below). Nothing in
// the simulator faults on its own, so a machine nobody has injected into reads
// zero everywhere, which is the truth about it.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vgpu/telemetry.hpp"

namespace vgpu::ras {

enum class Severity : uint32_t { Corrected, Uncorrected };
inline constexpr uint32_t kSeverities = 2;

// nvidia-smi's ECC locations, in the order its query fields list them. "dram"
// is another name for device memory and "total" is the sum, so neither is
// stored.
enum class Location : uint32_t {
  DeviceMemory, RegisterFile, L1Cache, L2Cache, TextureMemory, Cbu, Sram
};
inline constexpr uint32_t kLocations = 7;

// NVML's PCIe transport counters.
enum class Pcie : uint32_t {
  Replay, ReplayRollover, L0ToRecovery, Correctable, NaksReceived, BadTlp,
  NaksSent, BadDllp, NonFatal, Fatal, Lcrc, Lane
};
inline constexpr uint32_t kPcieCounters = 12;

// How a card takes failing memory out of service. GDDR cards with ECC retire
// pages; HBM cards remap rows. Mirrors telemetry::DeviceSample::memory_retirement.
enum class Retirement : uint32_t { None = 0, Pages = 1, Rows = 2 };

// Every member is a uint64_t: the file is read and written a word at a time.
struct Counters {
  uint64_t ecc[kSeverities][kLocations];
  uint64_t retired_sbe, retired_dbe, retired_pending;
  uint64_t rows_correctable, rows_uncorrectable, rows_pending, rows_failure;
  uint64_t pcie[kPcieCounters];
  // Faults armed for delivery to a running kernel (volatile file only), by
  // where they are taken (Target) and then corrected, uncorrected, bit flip;
  // how many are armed at each target; and the silent bit flips delivered.
  uint64_t armed[5][3];
  uint64_t armed_pending[5];   // plus kRatePending while a rate is set at the target
  // Faults taken at a rate instead of a count: the chance each access takes
  // one, in units of 2^-32, by target and kind; and the generator's seed.
  uint64_t rate[5][3];
  uint64_t rate_seed;
  uint64_t bitflips_delivered;
  // A kernel hang armed for the next launch (not counted in armed_total, which
  // loads read), and how long it lasts: 0 is until the process is stopped.
  uint64_t armed_hang, hang_seconds;
  // Injected clock-event reasons: the current window, and the time each reason
  // was active in windows already closed, in microseconds.
  uint64_t throttle_mask, throttle_since_ns, throttle_until_ns;
  uint64_t throttle_us[8];
  // NVML events, the newest kEvents of them: the sequence of the last one
  // written, and a ring of {seq, type, data, time_ns}.
  uint64_t event_seq;
  uint64_t events[32][4];
  // Whether the GPU has fallen off the bus, and stuck cells: how many, and up
  // to 16 of {byte offset, kStuckValid | value << 8 | bit}. Volatile file only,
  // and kept across a driver reload.
  uint64_t lost;
  uint64_t link_gen, link_width;   // a degraded link (`vgpu fault link`); 0 is as trained
  uint64_t stuck_count;
  uint64_t stuck[16][2];

  uint64_t ecc_total(Severity s) const;
  uint64_t armed_total() const;
};

struct State {
  Counters since_load{};   // volatile
  Counters lifetime{};     // aggregate; also holds retired pages and remapped rows
};

// A device's state. One nobody has injected into reads zero.
State read(const std::string& uuid);

// Adds n ECC errors at a location, to both lifetimes. An uncorrectable error in
// device memory also takes the memory out of service the way the card does:
// a GDDR card retires a page, an HBM card remaps a row, and either is pending
// until the next driver load.
void inject_ecc(const std::string& uuid, Severity s, Location l, uint64_t n, Retirement scheme);

// Adds n to a PCIe transport counter. These count since the driver loaded.
void inject_pcie(const std::string& uuid, Pcie c, uint64_t n);

// Zeroes the volatile counts. A driver reload also completes pending page
// retirements and row remaps; `nvidia-smi -p 0` only zeroes the counts.
void reset_volatile(const std::string& uuid, bool driver_reload = true);

// `nvidia-smi -p 1`: zeroes the aggregate ECC counts. Retired pages and remapped
// rows stay: they are memory taken out of service, not counters.
void reset_aggregate(const std::string& uuid);

// Names as nvidia-smi and `vgpu fault` spell them.
const char* location_name(Location l);
bool parse_location(const std::string& s, Location* out);   // also accepts "dram"
const char* pcie_name(Pcie c);
bool parse_pcie(const std::string& s, Pcie* out);

// Where aggregate state is kept (see above).
std::string state_dir();

// ---- The session's sysfs ------------------------------------------------------
//
// Inside `vgpu shell`, the files the kernel and amdgpu keep for a device --
// the PCIe AER stats (aer_dev_correctable, aer_dev_nonfatal, aer_dev_fatal)
// and amdgpu's per-block ras/*_err_count -- rewritten from this state whenever
// a count changes, under <session>/ras/<uuid>/. The session's
// /sys/class/drm/cardN/device links to them. Outside a session (no
// VGPU_SESSION, or `session` empty) this does nothing.
void publish_session(const std::string& uuid, const std::string& session = "");
// amdgpu's RAS blocks with an err_count file, in its names.
inline constexpr const char* kAmdgpuRasBlocks[] = {"umc", "sdma", "gfx", "mmhub", "pcie_bif", "hdp",
                                                    "xgmi_wafl"};
inline constexpr const char* kAerFiles[] = {"aer_dev_correctable", "aer_dev_nonfatal", "aer_dev_fatal"};

// ---- Faults delivered to a running kernel -----------------------------------
//
// `vgpu fault arm` loads faults that a running kernel's next accesses take: a
// corrected error is counted and changes nothing; an uncorrectable one is
// counted, logged, and fails the kernel with cudaErrorECCUncorrectable; a bit
// flip silently corrupts the value, the way a fault that ECC does not cover
// would. Most severe first.
enum class Armed : uint32_t { None, Corrected, Uncorrected, Bitflip, Hang };
// Where a fault is taken: device-memory loads, device-memory stores (bit flips
// only -- ECC is checked when memory is read, so a store's flip is what the
// next read finds), shared-memory loads, whose ECC errors count as the L1
// cache's since shared memory and L1 are one SRAM, or floating-point and
// matrix results (bit flips only: no ECC covers an ALU, so nothing counts or
// reports the error -- silent data corruption, which result checks catch), or
// copies out of device memory: a corrected error is counted, an uncorrectable
// one fails the copy, and a bit flip corrupts what that copy delivers.
enum class Target : uint32_t { Load, Store, Shared, Alu, Copy };
inline constexpr uint32_t kTargets = 5;
static_assert(sizeof(Counters::armed_pending) / sizeof(uint64_t) == kTargets);
const char* target_name(Target t);
bool parse_target(const std::string& s, Target* out);
// Throws std::invalid_argument for an ECC error armed on stores or results.
void arm(const std::string& uuid, Armed kind, uint64_t n, Target at = Target::Load);
// A fault taken at a rate: every access at `at` takes one with probability
// `per_access` (0 stops it), drawn from a generator seeded with `seed`, until
// stopped or a driver reload. Counted faults armed at the same target are
// taken first. Errors then grow with memory traffic, as a failing part's do.
inline constexpr uint64_t kRatePending = uint64_t{1} << 40;
void arm_rate(const std::string& uuid, Armed kind, double per_access, Target at, uint64_t seed);
double rate_of(const Counters& c, Target at, Armed kind);
// A hang for the next kernel launch: it stalls for `seconds` and then fails
// with cudaErrorLaunchTimeout, or with seconds 0 never returns.
void arm_hang(const std::string& uuid, uint64_t seconds);

// One device's armed faults, mapped for the life of the process. Checked on
// every access of a target, so what is read there is a single word.
class ArmedFaults {
 public:
  explicit ArmedFaults(const std::string& uuid);   // creates the state file
  ~ArmedFaults();
  ArmedFaults(const ArmedFaults&) = delete;
  ArmedFaults& operator=(const ArmedFaults&) = delete;

  // Non-zero while anything is armed at `at`.
  const uint64_t* pending(Target at = Target::Load) const;
  // Takes one fault, most severe first. None when another thread took the last.
  Armed take(Target at = Target::Load);
  void note_bitflip();
  // Takes an armed hang, if there is one, with how long it lasts.
  bool take_hang(uint64_t* seconds);
  // Non-zero while any cell is stuck.
  const uint64_t* stuck_pending() const;
  // Whether the GPU has fallen off the bus.
  bool lost() const;
  // Forces the stuck bits that fall in `len` bytes read from `offset`.
  void apply_stuck(uint64_t offset, uint8_t* bytes, uint64_t len) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// ---- Stuck cells ------------------------------------------------------------
//
// `vgpu fault stuck`: a bit of device memory that reads as 0 or 1 whatever was
// written to it, which is what an address-aware memory test looks for. Every
// read sees it -- a kernel's loads and atomics, and copies back to the host.
// The offset counts bytes from the start of the device's memory, which is
// where its first allocation starts; allocations follow in order and are never
// moved or reused. A cell stays stuck across a driver reload, as a real one
// does, until cleared.
inline constexpr uint32_t kStuckCells = 16;
struct StuckCell {
  uint64_t offset;
  uint32_t bit;     // 0-7, within the byte
  uint32_t value;   // what it reads as
};
// Throws std::length_error when kStuckCells are already stuck.
void stick(const std::string& uuid, uint64_t offset, uint32_t bit, uint32_t value);
void unstick_all(const std::string& uuid);
std::vector<StuckCell> stuck_cells(const std::string& uuid);

// ---- A GPU that has fallen off the bus ------------------------------------------
//
// `vgpu fault lose`: the GPU stops answering, as one does after Xid 79. NVML
// answers NVML_ERROR_GPU_IS_LOST for it, nvidia-smi reports it lost and exits
// 15, and a program's next launch on it fails. A driver reload does not bring
// it back; `vgpu fault lose --clear` does, as a reset or reboot would.
void lose(const std::string& uuid, const std::string& bus_id);
void recover(const std::string& uuid);
bool is_lost(const std::string& uuid);

// ---- The kernel log ---------------------------------------------------------
//
// Appends a line to the session's dmesg ($VGPU_SESSION/dmesg.log), stamped
// after the last line already there. Outside a session there is no simulated
// kernel log, and this does nothing.
void log_kernel(const std::string& message);

// A line in the NVIDIA driver's Xid form, for the device at bus_id
// ("00000000:01:00.0"). `process` is "pid=123, name=python3", or empty for an
// error no process caused, which the driver prints as pid='<unknown>'.
std::string xid_line(const std::string& bus_id, int xid, const std::string& process,
                     const std::string& detail);

// The amdgpu driver's line for ECC errors it counted in a RAS block: device
// memory is the UMC block, the on-chip memories GFX.
std::string amdgpu_ras_line(const std::string& bus_id, Severity s, Location l, uint64_t n);

// The kernel's AER line for a PCIe error on bus_id, or "" for a counter AER
// does not report (NAKs, lane errors, recovery entries).
std::string aer_line(const std::string& bus_id, Pcie c);

// ---- Clock-event reasons (`vgpu fault throttle`) ----------------------------
//
// NVML's bits. GPU idle is never injected: it is reported whenever a device is
// idle, as a real one reports it.
inline constexpr uint64_t kGpuIdle = 0x1, kSwPowerCap = 0x4, kHwSlowdown = 0x8,
                          kSwThermalSlowdown = 0x20, kHwThermalSlowdown = 0x40,
                          kHwPowerBrakeSlowdown = 0x80;
// nvidia-smi's names for the injectable ones; 0 for any other name.
uint64_t reason_bit(const std::string& name);

// Starts a window with these reasons active, for `seconds` or (0) until cleared.
// Starting one closes the last, so its time is kept in the counters.
void throttle(const std::string& uuid, uint64_t reasons, uint64_t seconds);
void clear_throttle(const std::string& uuid);

// The reasons active now, and a reading adjusted to agree with them: a thermal
// slowdown puts the temperature at the slowdown threshold, a power cap holds
// power at the limit, and a slowdown pulls the SM clock down (to half for the
// hardware reasons, to three quarters for the software ones). Sets
// d.clock_event_reasons; returns it.
uint64_t apply_throttle(telemetry::DeviceSample& d);

// ---- A degraded PCIe link (`vgpu fault link`) ---------------------------------
//
// The link trained below what the card and slot support -- a lower generation,
// fewer lanes, or both -- as a bad riser, a dirty contact or a marginal slot
// leaves it. Reported as the link's current state, beside its unchanged
// maximum. A driver reload does not retrain it; clearing does, as a reset would.
// gen or width 0 leaves that half as trained.
void degrade_link(const std::string& uuid, uint32_t gen, uint32_t width);
void restore_link(const std::string& uuid);
// Applies the degraded link to a reading, as apply_throttle does its reasons.
void apply_link(telemetry::DeviceSample& d);

// How long a reason has been active, over every window, in microseconds.
uint64_t throttle_time_us(const std::string& uuid, uint64_t reason);

// ---- NVML events --------------------------------------------------------------
//
// What an NVML event set waits for, recorded where every process on the machine
// can see it: `vgpu fault` and faults delivered to running kernels record them,
// and a health daemon in another process receives them.
inline constexpr uint64_t kEventSingleBitEcc = 0x1, kEventDoubleBitEcc = 0x2, kEventXid = 0x8;
inline constexpr uint32_t kEvents = 32;
struct Event {
  uint64_t seq, type, data, time_ns;
};
// Records one event: data is the Xid for kEventXid, and 0 otherwise.
void record_event(const std::string& uuid, uint64_t type, uint64_t data);
// The sequence of the newest event, 0 when there is none. A new watch starts
// here, so it sees only what happens after it began.
uint64_t event_head(const std::string& uuid);
// The first event after *after whose type is in `mask`, if there is one yet.
// Moves *after past every event it looked at, past any the ring has already
// overwritten, and back to the start if the ring was reset.
bool next_event(const std::string& uuid, uint64_t mask, uint64_t* after, Event* out);
// An Xid as the driver reports it: the line in dmesg and the NVML event.
void report_xid(const std::string& uuid, const std::string& bus_id, int xid,
                const std::string& process, const std::string& detail);

}  // namespace vgpu::ras
