// CPER records for an AMD GPU's RAS errors, as amdgpu writes them.
//
// A CPER is the UEFI Common Platform Error Record (UEFI specification,
// appendix N): a header, one descriptor per section, then the sections.
// amdgpu writes one to its CPER ring for every RAS error it handles
// (amdgpu_cper.c), with the layout of amd_cper.h; amd-smi reads the ring from
// debugfs and dumps each record as a .cper file (`amd-smi ras --cper
// --folder`). A runtime error -- an ECC error in device memory here -- is one
// AMD non-standard error section carrying the ACA bank registers that
// reported it, which amd-smi decodes to an AFID (AMD Field ID).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vgpu/telemetry.hpp"

namespace vgpu::amd {

// enum cper_error_severity.
enum class CperSeverity : uint32_t { NonFatalUncorrected = 0, Fatal = 1, NonFatalCorrected = 2 };

struct CperRecord {
  uint64_t time_ns = 0;
  CperSeverity severity = CperSeverity::NonFatalCorrected;
  std::string record_id;   // "<socket>:<n>", n counting the GPU's records from 1
  std::string bytes;       // the record, 472 bytes for one runtime section
  // The AFID amd-smi decodes from the record's ACA registers (its
  // amdsmi_get_afids_from_cper, AMD's ras-decode tables).
  int afid = 0;
};
inline constexpr uint32_t kCperRuntimeRecordLength = 472;

// A runtime record for one ECC error in device memory: the UMC bank's ACA
// registers, corrected or not. An uncorrectable error in device memory is
// non-fatal -- the page is poisoned and retired, and the GPU goes on -- so it
// is recorded as NonFatalUncorrected with the latent-error flag, as amdgpu
// records poison. `socket` is the GPU's OAM socket, `n` its record count.
CperRecord umc_record(const telemetry::DeviceSample& d, uint32_t socket, bool uncorrected, uint64_t time_ns,
                      uint32_t n);

// The records for a GPU's ECC errors since the machine started, oldest first
// (ras::next_event's single- and double-bit ECC events).
std::vector<CperRecord> cper_records(const telemetry::DeviceSample& d, uint32_t socket);

// A record's header as amd-smi reports it (`amd-smi ras --cper` without a
// folder, and the .json beside each dumped .cper): error_severity,
// notify_type, timestamp and the header's fields, in amd-smi's order.
std::string cper_header_json(const CperRecord& r, int indent);

// "non_fatal_corrected", "non_fatal_uncorrected", "fatal": amd-smi's names.
const char* cper_severity_name(CperSeverity s);

}  // namespace vgpu::amd
