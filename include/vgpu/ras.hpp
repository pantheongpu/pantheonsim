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
#include <string>

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

  uint64_t ecc_total(Severity s) const;
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

}  // namespace vgpu::ras
