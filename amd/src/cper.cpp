#include "vgpu/amd_cper.hpp"

#include <cstdio>
#include <cstring>
#include <ctime>

#include "vgpu/ras.hpp"

namespace vgpu::amd {
namespace {

// Byte offsets of amd_cper.h's packed structures.
constexpr uint32_t kHdrLen = 128;       // struct cper_hdr
constexpr uint32_t kSecDescLen = 72;    // struct cper_sec_desc
constexpr uint32_t kNonstdSecLen = 272; // struct cper_sec_nonstd_err

// A GUID as Linux stores guid_t (GUID_INIT): the first three fields little
// endian, the last eight bytes as written.
struct Guid {
  uint32_t a;
  uint16_t b, c;
  uint8_t d[8];
};
// CPER_NOTIFY_CMC, CPER_NOTIFY_MCE and AMD_GPU_NONSTANDARD_ERROR (amd_cper.h).
constexpr Guid kNotifyCmc = {0x2DCE8BB1, 0xBDD7, 0x450e, {0xB9, 0xAD, 0x9C, 0xF4, 0xEB, 0xD4, 0xF8, 0x90}};
constexpr Guid kNotifyMce = {0xE8F56FFE, 0x919C, 0x4cc5, {0xBA, 0x88, 0x65, 0xAB, 0xE1, 0x49, 0x13, 0xBB}};
constexpr Guid kRuntime = {0x32AC0C78, 0x2623, 0x48F6, {0x81, 0xA2, 0xAC, 0x69, 0x17, 0x80, 0x55, 0x1D}};

void put8(std::string& b, uint32_t at, uint8_t v) { b[at] = static_cast<char>(v); }
void put16(std::string& b, uint32_t at, uint16_t v) {
  for (int i = 0; i < 2; ++i) put8(b, at + i, static_cast<uint8_t>(v >> (8 * i)));
}
void put32(std::string& b, uint32_t at, uint32_t v) {
  for (int i = 0; i < 4; ++i) put8(b, at + i, static_cast<uint8_t>(v >> (8 * i)));
}
void put64(std::string& b, uint32_t at, uint64_t v) {
  put32(b, at, static_cast<uint32_t>(v));
  put32(b, at + 4, static_cast<uint32_t>(v >> 32));
}
void put_guid(std::string& b, uint32_t at, const Guid& g) {
  put32(b, at, g.a);
  put16(b, at + 4, g.b);
  put16(b, at + 6, g.c);
  for (int i = 0; i < 8; ++i) put8(b, at + 8 + i, g.d[i]);
}
// A fixed-size char field, as snprintf(field, size, ...) leaves it.
void put_str(std::string& b, uint32_t at, uint32_t size, const std::string& s) {
  for (uint32_t i = 0; i + 1 < size && i < s.size(); ++i) put8(b, at + i, static_cast<uint8_t>(s[i]));
}

// The UMC bank's ACA registers for an error in HBM, as 64-bit CTL, STATUS,
// ADDR, MISC0, CONFIG, IPID, SYND and what follows them: AMD's own example of
// an HBM corrected error (amdsmi's ras-decode, "HBM CORRECTED ERROR"), which
// AMD's decoder reads as bank umc, On-die ECC, AFID 24. An uncorrected one
// differs in MCA_STATUS alone: Deferred (bit 44), which is how a poisoned
// page is reported, and which the same decoder reads as Uncorrected,
// Non-fatal, AFID 22. The values are that example's, not measured on a card.
constexpr uint64_t kUmcAca[16] = {0xffff, 0xdc2040000000011bull, 0x0, 0xd008000801000000ull, 0x25000001ffull,
                                  0x209600191f00ull, 0xa000000, 0, 0, 0, 0xd008000801000000ull, 0, 0, 0, 0, 0};
constexpr uint64_t kMcaStatusDeferred = 1ull << 44;
constexpr int kAfidUmcCorrected = 24, kAfidUmcUncorrected = 22;

std::string two(int v) {
  char b[8];
  std::snprintf(b, sizeof b, "%02d", v);
  return b;
}

}  // namespace

const char* cper_severity_name(CperSeverity s) {
  switch (s) {
    case CperSeverity::NonFatalUncorrected: return "non_fatal_uncorrected";
    case CperSeverity::Fatal: return "fatal";
    case CperSeverity::NonFatalCorrected: return "non_fatal_corrected";
  }
  return "unknown";
}

CperRecord umc_record(const telemetry::DeviceSample& d, uint32_t socket, bool uncorrected, uint64_t time_ns,
                      uint32_t n) {
  CperRecord r;
  r.time_ns = time_ns;
  r.severity = uncorrected ? CperSeverity::NonFatalUncorrected : CperSeverity::NonFatalCorrected;
  char id[16];
  std::snprintf(id, 9, "%u:%X", socket, n);   // snprintf(record_id, 9, "%d:%X", ...)
  r.record_id = id;
  r.afid = uncorrected ? kAfidUmcUncorrected : kAfidUmcCorrected;

  std::string& b = r.bytes;
  b.assign(kCperRuntimeRecordLength, '\0');
  // The header (amdgpu_cper_entry_fill_hdr).
  b.replace(0, 4, "CPER");
  put16(b, 4, 0x100);                          // CPER_HDR_REV_1
  put32(b, 6, 0xFFFFFFFFu);                    // signature_end
  put16(b, 10, 1);                             // sec_cnt
  put32(b, 12, static_cast<uint32_t>(r.severity));
  put32(b, 16, 0x3);                           // valid: platform_id, timestamp
  put32(b, 20, kCperRuntimeRecordLength);
  // The timestamp in UTC, the century apart (amdgpu_cper_get_timestamp).
  const std::time_t t = static_cast<std::time_t>(time_ns / 1000000000ull);
  std::tm tm{};
  gmtime_r(&t, &tm);
  const int year = 1900 + tm.tm_year;
  const uint8_t stamp[8] = {static_cast<uint8_t>(tm.tm_sec), static_cast<uint8_t>(tm.tm_min),
                            static_cast<uint8_t>(tm.tm_hour), 0, static_cast<uint8_t>(tm.tm_mday),
                            static_cast<uint8_t>(tm.tm_mon + 1), static_cast<uint8_t>(year % 100),
                            static_cast<uint8_t>(year / 100)};
  for (int i = 0; i < 8; ++i) put8(b, 24 + i, stamp[i]);
  char platform[20];
  std::snprintf(platform, sizeof platform, "0x%04X:0x%04X", d.pci_device_id & 0xFFFF, d.pci_device_id >> 16);
  put_str(b, 32, 16, platform);
  put_str(b, 64, 16, "amdgpu");                // CPER_CREATOR_ID_AMDGPU
  put_guid(b, 80, uncorrected ? kNotifyMce : kNotifyCmc);
  put_str(b, 96, 8 + 1, r.record_id);          // record_id[8], copied without its NUL

  // The section descriptor (amdgpu_cper_entry_fill_section_desc).
  const uint32_t desc = kHdrLen, sec = kHdrLen + kSecDescLen;
  put32(b, desc + 0, sec);
  put32(b, desc + 4, kNonstdSecLen);
  put8(b, desc + 8, 0x01);                     // CPER_SEC_MINOR_REV_1
  put8(b, desc + 9, 0x22);                     // CPER_SEC_MAJOR_REV_22
  put8(b, desc + 10, 0x2);                     // valid: fru_text
  put32(b, desc + 12, 0x1 | (uncorrected ? 0x10u : 0u));   // primary; latent_err for poison
  put_guid(b, desc + 16, kRuntime);
  put32(b, desc + 48, static_cast<uint32_t>(r.severity));
  put_str(b, desc + 52, 20, "OAM" + std::to_string(socket));

  // The non-standard error section (amdgpu_cper_entry_fill_runtime_section).
  put64(b, sec + 0, (1ull << 2) | (1ull << 8));  // err_info_cnt 1, err_context_cnt 1
  put_guid(b, sec + 64, kRuntime);               // info.error_type
  put64(b, sec + 88, 0x1);                       // ms_chk: err_type_valid
  put16(b, sec + 128, 1);                        // CPER_CTX_TYPE_CRASH
  put16(b, sec + 130, 128);                      // sizeof reg_dump
  for (int i = 0; i < 16; ++i)
    put64(b, sec + 144 + 8 * i, i == 1 && uncorrected ? kUmcAca[1] | kMcaStatusDeferred : kUmcAca[i]);
  return r;
}

std::vector<CperRecord> cper_records(const telemetry::DeviceSample& d, uint32_t socket) {
  std::vector<CperRecord> out;
  uint64_t after = 0;
  ras::Event e{};
  while (ras::next_event(d.uuid, ras::kEventSingleBitEcc | ras::kEventDoubleBitEcc, &after, &e))
    out.push_back(umc_record(d, socket, e.type == ras::kEventDoubleBitEcc, e.time_ns,
                             static_cast<uint32_t>(out.size() + 1)));
  return out;
}

std::string cper_header_json(const CperRecord& r, int indent) {
  const std::string& b = r.bytes;
  const auto u8 = [&](uint32_t at) { return static_cast<unsigned>(static_cast<uint8_t>(b[at])); };
  const auto u32 = [&](uint32_t at) { return u8(at) | u8(at + 1) << 8 | u8(at + 2) << 16 | u8(at + 3) << 24; };
  const auto str = [&](uint32_t at, uint32_t size) {
    std::string s;
    for (uint32_t i = 0; i < size && b[at + i]; ++i) s += b[at + i];
    return s;
  };
  const std::string pad(indent, ' '), in(indent + 2, ' ');
  const std::string when = std::to_string(u8(31) * 100 + u8(30)) + "/" + two(u8(29)) + "/" + two(u8(28)) + " " +
                           two(u8(26)) + ":" + two(u8(25)) + ":" + two(u8(24));
  std::string j = "{\n";
  j += in + "\"error_severity\": \"" + cper_severity_name(r.severity) + "\",\n";
  j += in + "\"notify_type\": \"" + (r.severity == CperSeverity::NonFatalCorrected ? "CMC" : "MCE") + "\",\n";
  j += in + "\"timestamp\": \"" + when + "\",\n";
  j += in + "\"signature\": \"CPER\",\n";
  j += in + "\"revision\": " + std::to_string(u8(4) | u8(5) << 8) + ",\n";
  j += in + "\"signature_end\": \"0xffffffff\",\n";
  j += in + "\"sec_cnt\": " + std::to_string(u8(10) | u8(11) << 8) + ",\n";
  j += in + "\"record_length\": " + std::to_string(u32(20)) + ",\n";
  j += in + "\"platform_id\": \"" + str(32, 16) + "\",\n";
  j += in + "\"creator_id\": \"" + str(64, 16) + "\",\n";
  j += in + "\"record_id\": \"" + str(96, 8) + "\",\n";
  j += in + "\"flags\": " + std::to_string(u32(104)) + ",\n";
  j += in + "\"persistence_info\": 0\n";
  return j + pad + "}";
}

}  // namespace vgpu::amd
