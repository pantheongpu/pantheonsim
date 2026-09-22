// `vgpu smi` — nvidia-smi-style view of the running virtual GPUs.
//
// Reads the shared telemetry segment published by a live VirtualGPU process,
// so it works from another terminal while a workload runs. Columns sourced
// from real simulator state (memory, utilization) are exact; the power /
// temperature / clock / voltage columns are a synthetic model driven by real
// utilization and are marked as such by --explain.
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cinttypes>
#include <csignal>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <string>
#include <vector>

#include "args.hpp"
#include "machine.hpp"
#include "vgpu/ras.hpp"
#include "vgpu/regs.hpp"
#include "vgpu/driver_version.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/telemetry.hpp"

namespace {

const char* perf_state_name(uint32_t p) {
  static char buf[8];
  std::snprintf(buf, sizeof buf, "P%u", p);
  return buf;
}

int mib(uint64_t bytes) { return static_cast<int>(bytes / (1024 * 1024)); }

// Which devices a command talks about: all of them, or the -i/--id selection.
// nvidia-smi accepts an index, a UUID or a PCI bus id, comma-separated, and the
// selection used to be read and thrown away -- `nvidia-smi -i 1` printed every
// GPU, and `-i 7` on a two-GPU machine printed both and exited 0, so a script
// asking about one card got the whole rack and no error.
std::string lower(std::string v) {
  for (char& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return v;
}
bool select_devices(const vgpu::telemetry::Shared& s, const std::string& spec,
                    std::vector<uint32_t>* out) {
  out->clear();
  if (spec.empty()) {
    for (uint32_t i = 0; i < s.device_count; ++i) out->push_back(i);
    return true;
  }
  for (size_t at = 0; at <= spec.size();) {
    const size_t comma = spec.find(',', at);
    std::string tok = spec.substr(at, (comma == std::string::npos ? spec.size() : comma) - at);
    while (!tok.empty() && tok.front() == ' ') tok.erase(tok.begin());
    while (!tok.empty() && tok.back() == ' ') tok.pop_back();
    bool found = false;
    for (uint32_t i = 0; i < s.device_count && !found; ++i) {
      const auto& d = s.devices[i];
      const bool digits = !tok.empty() &&
                          tok.find_first_not_of("0123456789") == std::string::npos;
      if (digits) {
        // strtoull with a range check, not stoul: an index past any integer
        // threw out of here as "vgpu: stoul" instead of naming no device.
        errno = 0;
        const unsigned long long v = std::strtoull(tok.c_str(), nullptr, 10);
        found = errno != ERANGE && v == i;
      } else if (lower(tok) == lower(d.uuid)) {
        found = true;
      } else if (tok.find(':') != std::string::npos) {
        // "00000000:01:00.0", "0000:01:00.0" and "01:00.0" all name the same
        // device, so compare from the right.
        const std::string id = lower(d.bus_id), want = lower(tok);
        found = id.size() >= want.size() && id.compare(id.size() - want.size(), want.size(), want) == 0;
      }
      if (found) out->push_back(i);
    }
    if (!found) return false;
    if (comma == std::string::npos) break;
    at = comma + 1;
  }
  return true;
}

// What -q prints for brand and architecture. The profile carries lower-case
// ids ("turing", "nvidia"); the tool prints names.
// The PCIe link, read from the device's registers as a driver reads it, so the
// read shows in the register access log; without register state, the
// reading's own link.
vgpu::regs::Link link_of(const vgpu::telemetry::DeviceSample& d) {
  vgpu::regs::Link l{d.pcie_gen, d.pcie_width, d.pcie_gen_max, d.pcie_width_max};
  try {
    vgpu::regs::ConfigSpace cs(d);
    l = vgpu::regs::link(cs);
  } catch (const std::exception&) {
  }
  return l;
}

const char* brand_name(const vgpu::telemetry::DeviceSample& d) {
  if (std::strcmp(d.vendor, "amd") == 0) return "AMD";
  return std::strstr(d.name, "GeForce") || std::strstr(d.name, "RTX 30") ? "GeForce" : "NVIDIA";
}
std::string architecture_name(const char* arch) {
  const std::string a = lower(arch);
  if (a == "ada" || a == "ada_lovelace" || a == "ada lovelace") return "Ada Lovelace";
  std::string out = arch;
  if (!out.empty()) out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
  return out;
}

// The frame nvidia-smi draws: 91 columns, three fields of 41, 24 and 22
// between the bars. Every width below was measured from a real driver's output
// rather than guessed, because this table is a parsed interface -- people
// scrape these columns with awk and cut, and a session that renders its own
// idea of the layout is a session those scripts break on. Nothing here is
// copied from NVIDIA: the format is read off the observable output, the same
// way the --query-gpu shape below already is.
const char* const kFrame =
    "+-----------------------------------------------------------------------------------------+\n";
const char* const kRowRule =
    "+-----------------------------------------+------------------------+----------------------+\n";

void print_gpu_table(const vgpu::telemetry::Shared& s, const std::vector<uint32_t>& sel) {
  // Real nvidia-smi opens with the wall clock, padded out to the frame width.
  std::time_t now = std::time(nullptr);
  std::tm tm{};
  ::localtime_r(&now, &tm);
  char when[64];
  std::strftime(when, sizeof when, "%a %b %e %H:%M:%S %Y", &tm);
  std::printf("%-31s\n", when);

  // The left-hand version is nvidia-smi's own, which tracks the driver on a
  // real machine; there is one binary here, so it reports the driver too.
  const char* driver = s.driver_version[0] ? s.driver_version : vgpu::kDefaultDriverRelease;
  const char* cuda = s.cuda_version[0] ? s.cuda_version : "13.0";

  std::printf("%s", kFrame);
  std::printf("| NVIDIA-SMI %-23.23sDriver Version: %-15.15sCUDA Version: %-9.9s|\n",
              driver, driver, cuda);
  std::printf("%s", kRowRule);
  std::printf("| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n");
  std::printf("| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n");
  std::printf("|                                         |                        |               MIG M. |\n");
  std::printf("|=========================================+========================+======================|\n");

  for (uint32_t i : sel) {
    const auto& d = s.devices[i];
    // "58W /  700W": the cap is padded to four digits, which is what puts the
    // two numbers under one another across a multi-GPU listing.
    char power[32];
    std::snprintf(power, sizeof power, "%uW / %4uW", d.power_mw / 1000, d.power_limit_mw / 1000);

    // The volatile uncorrected-ECC count on a card that ships with ECC, and N/A
    // on one without it -- the same answers the query fields give. MIG, on the
    // line below, is N/A because no profile carries MIG state.
    const std::string uncorrected =
        d.ecc_enabled ? std::to_string(vgpu::ras::read(d.uuid).since_load.ecc_total(
                            vgpu::ras::Severity::Uncorrected))
                      : "N/A";
    std::printf("|%4u  %-30.30s%3s  |   %-16.16s %3s |%21s |\n", i, d.name, "On", d.bus_id, "Off",
                uncorrected.c_str());
    std::printf("|%3u%%%5uC%6s%24s |%8uMiB /%7uMiB |%7u%%%13s |\n", d.fan_percent, d.temperature_c,
                perf_state_name(d.perf_state), power, mib(d.vram_used_bytes),
                mib(d.vram_total_bytes), d.utilization_gpu, "Default");
    std::printf("|                                         |                        |%21s |\n", "N/A");
    std::printf("%s", kRowRule);
  }

  // The process table is not optional decoration -- a missing block is itself
  // a difference from the real tool, and "no processes" is a normal answer.
  std::printf("\n");
  std::printf("%s", kFrame);
  std::printf("| Processes:                                                                              |\n");
  std::printf("|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n");
  std::printf("|        ID   ID                                                               Usage      |\n");
  std::printf("|=========================================================================================|\n");
  uint32_t shown = 0;
  for (uint32_t i : sel) {
    const auto& d = s.devices[i];
    for (uint32_t j = 0; j < d.proc_count && j < vgpu::telemetry::kMaxProcs; ++j) {
      const auto& pr = d.procs[j];
      char used[16];
      std::snprintf(used, sizeof used, "%dMiB", mib(pr.used_bytes));
      // GI and CI are MIG instance ids; without MIG a real driver prints N/A.
      // Type C is "compute", which is the only kind of context here.
      std::printf("|%5u   %3s  %3s%16u%7s   %-38.38s%-9s|\n", i, "N/A", "N/A", pr.pid, "C",
                  pr.name, used);
      ++shown;
    }
  }
  if (shown == 0)
    std::printf("|  %-87s|\n", "No running processes found");
  std::printf("%s", kFrame);
}

// What the frame above cannot say: which of those columns came from the
// simulator and which came from a model. Off by default because the default
// has to match the real tool byte for byte, and available because a number
// whose origin you cannot check is worse than no number.
void print_virtual_details(const vgpu::telemetry::Shared& s) {
  std::printf("\n");
  std::printf("%s", kFrame);
  std::printf("| Virtual device details                                                                  |\n");
  std::printf("|=========================================================================================|\n");
  for (uint32_t i = 0; i < s.device_count; ++i) {
    const auto& d = s.devices[i];
    char line[128];
    std::snprintf(line, sizeof line, "%u: %s  (vendor=%s arch=%s)", i, d.name, d.vendor,
                  d.architecture);
    std::printf("| %-87.87s |\n", line);
    std::snprintf(line, sizeof line, "   UUID        : %s", d.uuid);
    std::printf("| %-87.87s |\n", line);
    std::snprintf(line, sizeof line, "   SM clock    : %u / %u MHz      Mem clock : %u / %u MHz",
                  d.sm_clock_mhz, d.sm_clock_max_mhz, d.mem_clock_mhz, d.mem_clock_max_mhz);
    std::printf("| %-87.87s |\n", line);
    std::snprintf(line, sizeof line, "   Voltage     : %u mV            Mem util  : %u%%",
                  d.voltage_mv, d.utilization_mem);
    std::printf("| %-87.87s |\n", line);
    std::snprintf(line, sizeof line, "   Kernels     : %" PRIu64 "             Bytes moved: %" PRIu64,
                  d.kernels_launched, d.bytes_moved);
    std::printf("| %-87.87s |\n", line);
    std::printf("%s", kFrame);
  }
}

// The --query-gpu fields this answers, by the names `nvidia-smi --help-query-gpu`
// documents, with the unit a CSV header carries for each ("memory.total [MiB]").
// A name that is not here is refused before anything is printed, as real
// nvidia-smi refuses it. It used to come back "[N/A]" with exit 0, so a
// misspelt field passed a script's parse and read as a card with no value.
struct QueryField {
  const char* name;
  const char* unit;
};
const QueryField kGpuFields[] = {
    {"timestamp", nullptr},         {"driver_version", nullptr},
    {"count", nullptr},             {"index", nullptr},
    {"name", nullptr},              {"gpu_name", nullptr},
    {"serial", nullptr},            {"gpu_serial", nullptr},
    {"uuid", nullptr},              {"gpu_uuid", nullptr},
    {"pci.bus_id", nullptr},        {"gpu_bus_id", nullptr},
    {"pci.domain", nullptr},        {"pci.bus", nullptr},
    {"pci.device", nullptr},        {"pci.device_id", nullptr},
    {"pci.sub_device_id", nullptr}, {"compute_cap", nullptr},
    {"compute_mode", nullptr},      {"persistence_mode", nullptr},
    {"ecc.mode.current", nullptr},  {"mig.mode.current", nullptr},
    {"pstate", nullptr},            {"fan.speed", "%"},
    {"memory.total", "MiB"},        {"memory.used", "MiB"},
    {"memory.free", "MiB"},         {"utilization.gpu", "%"},
    {"utilization.memory", "%"},    {"temperature.gpu", nullptr},
    {"power.draw", "W"},            {"power.limit", "W"},
    {"enforced.power.limit", "W"},  {"power.max_limit", "W"},
    {"clocks.current.graphics", "MHz"}, {"clocks.gr", "MHz"},
    {"clocks.current.memory", "MHz"},   {"clocks.mem", "MHz"},
    {"clocks.max.graphics", "MHz"},     {"clocks.max.gr", "MHz"},
    {"clocks.max.memory", "MHz"},       {"clocks.max.mem", "MHz"},
    // VirtualGPU's own earlier spellings of the graphics clock, kept so the
    // callers that already use them keep working.
    {"clocks.sm", "MHz"}, {"clocks.current.sm", "MHz"}, {"clocks.max.sm", "MHz"},
    {"clocks_throttle_reasons.active", nullptr}, {"clocks_event_reasons.active", nullptr},
    // Reliability, link and clock-event fields, as nvidia-smi 595 lists them in
    // --help-query-gpu, with the retired_pages.sbe/.dbe aliases it also accepts.
    // Pantheon's RAS snapshot asks for most of them in one query, and one name
    // missing here failed the whole query.
    {"ecc.mode.pending", nullptr},
    {"ecc.errors.corrected.volatile.device_memory", nullptr},
    {"ecc.errors.corrected.volatile.dram", nullptr},
    {"ecc.errors.corrected.volatile.register_file", nullptr},
    {"ecc.errors.corrected.volatile.l1_cache", nullptr},
    {"ecc.errors.corrected.volatile.l2_cache", nullptr},
    {"ecc.errors.corrected.volatile.texture_memory", nullptr},
    {"ecc.errors.corrected.volatile.cbu", nullptr},
    {"ecc.errors.corrected.volatile.sram", nullptr},
    {"ecc.errors.corrected.volatile.total", nullptr},
    {"ecc.errors.corrected.aggregate.device_memory", nullptr},
    {"ecc.errors.corrected.aggregate.dram", nullptr},
    {"ecc.errors.corrected.aggregate.register_file", nullptr},
    {"ecc.errors.corrected.aggregate.l1_cache", nullptr},
    {"ecc.errors.corrected.aggregate.l2_cache", nullptr},
    {"ecc.errors.corrected.aggregate.texture_memory", nullptr},
    {"ecc.errors.corrected.aggregate.cbu", nullptr},
    {"ecc.errors.corrected.aggregate.sram", nullptr},
    {"ecc.errors.corrected.aggregate.total", nullptr},
    {"ecc.errors.uncorrected.volatile.device_memory", nullptr},
    {"ecc.errors.uncorrected.volatile.dram", nullptr},
    {"ecc.errors.uncorrected.volatile.register_file", nullptr},
    {"ecc.errors.uncorrected.volatile.l1_cache", nullptr},
    {"ecc.errors.uncorrected.volatile.l2_cache", nullptr},
    {"ecc.errors.uncorrected.volatile.texture_memory", nullptr},
    {"ecc.errors.uncorrected.volatile.cbu", nullptr},
    {"ecc.errors.uncorrected.volatile.sram", nullptr},
    {"ecc.errors.uncorrected.volatile.total", nullptr},
    {"ecc.errors.uncorrected.aggregate.device_memory", nullptr},
    {"ecc.errors.uncorrected.aggregate.dram", nullptr},
    {"ecc.errors.uncorrected.aggregate.register_file", nullptr},
    {"ecc.errors.uncorrected.aggregate.l1_cache", nullptr},
    {"ecc.errors.uncorrected.aggregate.l2_cache", nullptr},
    {"ecc.errors.uncorrected.aggregate.texture_memory", nullptr},
    {"ecc.errors.uncorrected.aggregate.cbu", nullptr},
    {"ecc.errors.uncorrected.aggregate.sram", nullptr},
    {"ecc.errors.uncorrected.aggregate.total", nullptr},
    {"ecc.errors.uncorrected.volatile.sram.parity", nullptr},
    {"ecc.errors.uncorrected.volatile.sram.secded", nullptr},
    {"ecc.errors.uncorrected.aggregate.sram.parity", nullptr},
    {"ecc.errors.uncorrected.aggregate.sram.secded", nullptr},
    {"ecc.errors.uncorrected.aggregate.sram.thresholdExceeded", nullptr},
    {"ecc.errors.uncorrected.aggregate.sram.l2", nullptr},
    {"ecc.errors.uncorrected.aggregate.sram.sm", nullptr},
    {"ecc.errors.uncorrected.aggregate.sram.mcu", nullptr},
    {"ecc.errors.uncorrected.aggregate.sram.pcie", nullptr},
    {"ecc.errors.uncorrected.aggregate.sram.other", nullptr},
    {"retired_pages.single_bit_ecc.count", nullptr},
    {"retired_pages.sbe", nullptr},
    {"retired_pages.double_bit.count", nullptr},
    {"retired_pages.dbe", nullptr},
    {"retired_pages.pending", nullptr},
    {"remapped_rows.correctable", nullptr},
    {"remapped_rows.correctable_inactive", nullptr},
    {"remapped_rows.uncorrectable", nullptr},
    {"remapped_rows.uncorrectable_inactive", nullptr},
    {"remapped_rows.pending", nullptr},
    {"remapped_rows.failure", nullptr},
    {"remapped_rows.histogram.max", nullptr},
    {"remapped_rows.histogram.high", nullptr},
    {"remapped_rows.histogram.partial", nullptr},
    {"remapped_rows.histogram.low", nullptr},
    {"remapped_rows.histogram.none", nullptr},
    {"temperature.gpu.tlimit", nullptr},
    {"temperature.memory", nullptr},
    {"pcie.link.gen.current", nullptr},
    {"pcie.link.gen.gpucurrent", nullptr},
    {"pcie.link.gen.max", nullptr},
    {"pcie.link.gen.gpumax", nullptr},
    {"pcie.link.gen.hostmax", nullptr},
    {"pcie.link.width.current", nullptr},
    {"pcie.link.width.max", nullptr},
    {"clocks_event_reasons.supported", nullptr},
    {"clocks_event_reasons.gpu_idle", nullptr},
    {"clocks_event_reasons.applications_clocks_setting", nullptr},
    {"clocks_event_reasons.sw_power_cap", nullptr},
    {"clocks_event_reasons.hw_slowdown", nullptr},
    {"clocks_event_reasons.hw_thermal_slowdown", nullptr},
    {"clocks_event_reasons.hw_power_brake_slowdown", nullptr},
    {"clocks_event_reasons.sw_thermal_slowdown", nullptr},
    {"clocks_event_reasons.sync_boost", nullptr},
    {"clocks_throttle_reasons.supported", nullptr},
    {"clocks_throttle_reasons.gpu_idle", nullptr},
    {"clocks_throttle_reasons.applications_clocks_setting", nullptr},
    {"clocks_throttle_reasons.sw_power_cap", nullptr},
    {"clocks_throttle_reasons.hw_slowdown", nullptr},
    {"clocks_throttle_reasons.hw_thermal_slowdown", nullptr},
    {"clocks_throttle_reasons.hw_power_brake_slowdown", nullptr},
    {"clocks_throttle_reasons.sw_thermal_slowdown", nullptr},
    {"clocks_throttle_reasons.sync_boost", nullptr},
    {"clocks_event_reasons_counters.sw_power_cap", "us"},
    {"clocks_event_reasons_counters.sync_boost", "us"},
    {"clocks_event_reasons_counters.sw_thermal_slowdown", "us"},
    {"clocks_event_reasons_counters.hw_thermal_slowdown", "us"},
    {"clocks_event_reasons_counters.hw_power_brake_slowdown", "us"},
};
// The same for --query-compute-apps (`nvidia-smi --help-query-compute-apps`).
const QueryField kAppFields[] = {
    {"timestamp", nullptr},  {"gpu_name", nullptr},     {"gpu_bus_id", nullptr},
    {"gpu_serial", nullptr}, {"gpu_uuid", nullptr},     {"pid", nullptr},
    {"process_name", nullptr}, {"name", nullptr},
    {"used_gpu_memory", "MiB"}, {"used_memory", "MiB"},
};
template <size_t N>
const QueryField* find_field(const QueryField (&table)[N], const std::string& name) {
  for (const QueryField& f : table)
    if (name == f.name) return &f;
  return nullptr;
}

// "2026/09/14 10:22:31.487", the form the timestamp query field has. Taken once
// per report, so every row of one sample carries the same time.
std::string query_timestamp() {
  timespec ts{};
  ::clock_gettime(CLOCK_REALTIME, &ts);
  std::tm tm{};
  ::localtime_r(&ts.tv_sec, &tm);
  char date[32], out[48];
  std::strftime(date, sizeof date, "%Y/%m/%d %H:%M:%S", &tm);
  std::snprintf(out, sizeof out, "%s.%03ld", date, ts.tv_nsec / 1000000L);
  return out;
}

// nvidia-smi's --query-gpu output: the caller names the fields, and the format
// modifiers decide whether a header and units come with them. Tools parse this,
// so the shape matters as much as the numbers -- the separator is a comma
// followed by a space, and a field with no value is "[N/A]", both of which real
// nvidia-smi does and both of which parsers depend on.
// A device's reliability counts, read once per report: every field of one
// report sees the same counts, and a loop sees them change between reports.
const vgpu::ras::State& ras_for(const vgpu::telemetry::DeviceSample& d, const std::string& when) {
  static std::string cached_when;
  static std::map<std::string, vgpu::ras::State> cache;
  if (when != cached_when) {
    cache.clear();
    cached_when = when;
  }
  auto it = cache.find(d.uuid);
  if (it == cache.end()) it = cache.emplace(d.uuid, vgpu::ras::read(d.uuid)).first;
  return it->second;
}

std::string query_field(const vgpu::telemetry::DeviceSample& d, uint32_t index,
                        const vgpu::telemetry::Shared& s, const std::string& field, bool units,
                        const std::string& when) {
  auto with = [&](const std::string& v, const char* unit) {
    return units && unit && *unit ? v + " " + unit : v;
  };
  auto num = [&](long long v, const char* unit) { return with(std::to_string(v), unit); };
  char buf[64];
  if (field == "timestamp") return when;
  if (field == "index") return std::to_string(index);
  if (field == "pci.domain" || field == "pci.bus" || field == "pci.device") {
    unsigned domain = 0, bus = 0, dev = 0, fn = 0;
    std::sscanf(d.bus_id, "%x:%x:%x.%x", &domain, &bus, &dev, &fn);
    if (field == "pci.domain")
      std::snprintf(buf, sizeof buf, "0x%04X", domain);
    else
      std::snprintf(buf, sizeof buf, "0x%02X", field == "pci.bus" ? bus : dev);
    return buf;
  }
  if (field == "pci.device_id" || field == "pci.sub_device_id") {
    std::snprintf(buf, sizeof buf, "0x%08X",
                  field == "pci.device_id" ? d.pci_device_id : d.pci_subsystem_id);
    return buf;
  }
  // The table's "On" and "Default" columns, in the words the query form uses;
  // MIG is N/A there too, since no profile carries it.
  if (field == "persistence_mode") return "Enabled";
  if (field == "compute_mode") return "Default";
  if (field == "mig.mode.current") return "[N/A]";
  // ECC. A card that ships with it reports it on, with the counts injected into
  // it (`vgpu fault`) -- zero until then, since nothing faults on its own. One
  // without it answers [N/A], as a GeForce card does. The SRAM breakdown and
  // its threshold flag answer on every card -- a real RTX 3060 reports 0 and
  // "No" -- and they are what makes Pantheon's RAS check find a supported
  // source on a card without ECC.
  if (field == "ecc.mode.current" || field == "ecc.mode.pending")
    return d.ecc_enabled ? "Enabled" : "[N/A]";
  if (field.rfind("ecc.errors.", 0) == 0) {
    if (field.find(".sram.") != std::string::npos)
      return field.size() > 17 && field.compare(field.size() - 17, 17, "thresholdExceeded") == 0
                 ? "No"
                 : "0";
    if (!d.ecc_enabled) return "[N/A]";
    // ecc.errors.<corrected|uncorrected>.<volatile|aggregate>.<location>
    std::vector<std::string> part;
    for (size_t at = 0; at <= field.size();) {
      const size_t dot = field.find('.', at);
      part.push_back(field.substr(at, (dot == std::string::npos ? field.size() : dot) - at));
      at = dot == std::string::npos ? field.size() + 1 : dot + 1;
    }
    if (part.size() != 5) return "[N/A]";
    const vgpu::ras::State& st = ras_for(d, when);
    const vgpu::ras::Counters& c = part[3] == "aggregate" ? st.lifetime : st.since_load;
    const auto sev = part[2] == "corrected" ? vgpu::ras::Severity::Corrected
                                            : vgpu::ras::Severity::Uncorrected;
    if (part[4] == "total") return std::to_string(c.ecc_total(sev));
    vgpu::ras::Location loc;
    if (!vgpu::ras::parse_location(part[4], &loc)) return "[N/A]";
    return std::to_string(c.ecc[static_cast<uint32_t>(sev)][static_cast<uint32_t>(loc)]);
  }
  // GDDR cards with ECC retire pages; HBM cards remap rows. The bank-availability
  // histogram needs a per-card bank count that no profile records yet.
  if (field.rfind("retired_pages.", 0) == 0) {
    if (d.memory_retirement != 1) return "[N/A]";
    const vgpu::ras::Counters& life = ras_for(d, when).lifetime;
    if (field == "retired_pages.pending") return life.retired_pending ? "Yes" : "No";
    const bool single = field == "retired_pages.sbe" || field == "retired_pages.single_bit_ecc.count";
    return std::to_string(single ? life.retired_sbe : life.retired_dbe);
  }
  if (field.rfind("remapped_rows.", 0) == 0) {
    if (d.memory_retirement != 2 || field.rfind("remapped_rows.histogram.", 0) == 0) return "[N/A]";
    const vgpu::ras::Counters& life = ras_for(d, when).lifetime;
    if (field == "remapped_rows.pending") return life.rows_pending ? "Yes" : "No";
    if (field == "remapped_rows.failure") return life.rows_failure ? "Yes" : "No";
    if (field == "remapped_rows.correctable") return std::to_string(life.rows_correctable);
    if (field == "remapped_rows.uncorrectable") return std::to_string(life.rows_uncorrectable);
    return "0";   // the *_inactive counts: rows remapped in a bank since swapped out
  }
  // A memory sensor only where real cards of the model report one. The real
  // driver prints this field's absence as N/A, without the brackets.
  if (field == "temperature.memory")
    return d.has_memory_temperature
               ? std::to_string(d.temperature_mem_c ? d.temperature_mem_c : d.temperature_c)
               : "N/A";
  // current and gpucurrent are the link as trained now; max, gpumax and
  // hostmax what it can train to.
  if (field.rfind("pcie.link.gen.", 0) == 0 || field.rfind("pcie.link.width.", 0) == 0) {
    const vgpu::regs::Link l = link_of(d);
    const bool current = field.find("current") != std::string::npos;
    const uint32_t v = field.rfind("pcie.link.gen.", 0) == 0 ? (current ? l.gen : l.max_gen)
                                                            : (current ? l.width : l.max_width);
    return v ? std::to_string(v) : "[N/A]";
  }
  if (field == "count") return std::to_string(s.device_count);
  if (field == "name" || field == "gpu_name") return d.name;
  if (field == "uuid" || field == "gpu_uuid") return d.uuid;
  if (field == "serial" || field == "gpu_serial") return "[N/A]";
  if (field == "pci.bus_id" || field == "gpu_bus_id") return d.bus_id;
  if (field == "driver_version")
    // The snapshot's, as the table header prints: an environment lookup here
    // could disagree with the header two lines above it.
    return s.driver_version[0] ? s.driver_version : vgpu::kDefaultDriverRelease;
  if (field == "compute_cap") {
    std::snprintf(buf, sizeof buf, "%u.%u", d.cc_major, d.cc_minor);
    return buf;
  }
  if (field == "memory.total") return num(mib(d.vram_total_bytes), "MiB");
  if (field == "memory.used") return num(mib(d.vram_used_bytes), "MiB");
  if (field == "memory.free")
    return num(mib(d.vram_total_bytes) - mib(d.vram_used_bytes), "MiB");
  if (field == "utilization.gpu") return num(d.utilization_gpu, "%");
  if (field == "utilization.memory") return num(d.utilization_mem, "%");
  if (field == "temperature.gpu") return std::to_string(d.temperature_c);
  if (field == "power.draw") {
    std::snprintf(buf, sizeof buf, "%.2f", d.power_mw / 1000.0);
    return with(buf, "W");
  }
  if (field == "power.limit" || field == "power.max_limit" ||
      field == "enforced.power.limit") {
    std::snprintf(buf, sizeof buf, "%.2f", d.power_limit_mw / 1000.0);
    return with(buf, "W");
  }
  // clocks.gr is the documented short form of the graphics (SM) clock, and the
  // one Pantheon queries; it used to come back [N/A].
  if (field == "clocks.sm" || field == "clocks.current.sm" || field == "clocks.gr" ||
      field == "clocks.current.graphics")
    return num(d.sm_clock_mhz, "MHz");
  if (field == "clocks.mem" || field == "clocks.current.memory")
    return num(d.mem_clock_mhz, "MHz");
  if (field == "clocks.max.sm" || field == "clocks.max.gr" || field == "clocks.max.graphics")
    return num(d.sm_clock_max_mhz, "MHz");
  if (field == "clocks.max.memory" || field == "clocks.max.mem")
    return num(d.mem_clock_max_mhz, "MHz");
  if (field == "fan.speed") return num(d.fan_percent, "%");
  // Clock-event reasons, in the forms a real driver prints: GPU idle whenever
  // the device is, as a real idle card reports it, and whatever was injected
  // with `vgpu fault throttle` (read_machine applied it). The supported mask
  // is an RTX 3060's.
  const uint64_t active =
      (d.utilization_gpu == 0 ? vgpu::ras::kGpuIdle : 0) | d.clock_event_reasons;
  if (field == "clocks_throttle_reasons.active" || field == "clocks_event_reasons.active") {
    std::snprintf(buf, sizeof buf, "0x%016llX", static_cast<unsigned long long>(active));
    return buf;
  }
  if (field == "clocks_throttle_reasons.supported" || field == "clocks_event_reasons.supported")
    return "0x00000000000001FF";
  if (field.rfind("clocks_event_reasons_counters.", 0) == 0) {
    const uint64_t bit = vgpu::ras::reason_bit(field.substr(field.find_last_of('.') + 1));
    return num(bit ? static_cast<long long>(vgpu::ras::throttle_time_us(d.uuid, bit)) : 0, "us");
  }
  if (field.rfind("clocks_event_reasons.", 0) == 0 || field.rfind("clocks_throttle_reasons.", 0) == 0) {
    const std::string name = field.substr(field.find('.') + 1);
    const uint64_t bit = name == "gpu_idle" ? vgpu::ras::kGpuIdle : vgpu::ras::reason_bit(name);
    return bit && (active & bit) ? "Active" : "Not Active";
  }
  if (field == "pstate") { std::snprintf(buf, sizeof buf, "P%u", d.perf_state); return buf; }
  return "[N/A]";
}

// The unit nvidia-smi puts in a CSV header, "memory.total [MiB]". It is there
// with or without nounits -- nounits strips the values, not the header -- and
// scripts that read the CSV by column name look for exactly that string.
const char* field_unit(const std::string& f) {
  const QueryField* q = find_field(kGpuFields, f);
  return q ? q->unit : nullptr;
}

std::vector<std::string> split_fields(const std::string& fields) {
  std::vector<std::string> names;
  for (size_t at = 0; at <= fields.size();) {
    const size_t comma = fields.find(',', at);
    const size_t end = comma == std::string::npos ? fields.size() : comma;
    std::string f = fields.substr(at, end - at);
    while (!f.empty() && (f.front() == ' ' || f.front() == '\t')) f.erase(f.begin());
    while (!f.empty() && (f.back() == ' ' || f.back() == '\t')) f.pop_back();
    if (!f.empty()) names.push_back(f);
    if (comma == std::string::npos) break;
    at = comma + 1;
  }
  return names;
}

void print_header(const std::vector<std::string>& names,
                  const char* (*unit_of)(const std::string&),
                  std::string (*label_of)(const std::string&)) {
  for (size_t i = 0; i < names.size(); ++i) {
    const char* unit = unit_of(names[i]);
    std::printf("%s%s%s%s%s", i ? ", " : "", label_of(names[i]).c_str(), unit ? " [" : "",
                unit ? unit : "", unit ? "]" : "");
  }
  std::printf("\n");
}
std::string same_label(const std::string& f) { return f; }

void print_query_gpu(const vgpu::telemetry::Shared& s, const std::vector<uint32_t>& sel,
                     const std::string& fields, bool header, bool units) {
  const std::vector<std::string> names = split_fields(fields);
  if (names.empty()) return;
  if (header) print_header(names, field_unit, same_label);
  const std::string when = query_timestamp();
  for (uint32_t g : sel)
    for (size_t i = 0; i < names.size(); ++i)
      std::printf("%s%s%s", i ? ", " : "",
                  query_field(s.devices[g], g, s, names[i], units, when).c_str(),
                  i + 1 == names.size() ? "\n" : "");
}

// --query-compute-apps: one row per process holding a context. With nothing
// running a real driver prints the header and no rows, which is the normal
// answer on an idle machine -- scripts that check for stray processes before a
// job read exactly that.
const char* app_unit(const std::string& f) {
  const QueryField* q = find_field(kAppFields, f);
  return q ? q->unit : nullptr;
}
std::string app_label(const std::string& f) {
  if (f == "used_memory") return "used_gpu_memory";
  if (f == "name") return "process_name";
  return f;
}
void print_query_apps(const vgpu::telemetry::Shared& s, const std::vector<uint32_t>& sel,
                      const std::string& fields, bool header, bool units) {
  const std::vector<std::string> names = split_fields(fields);
  if (names.empty()) return;
  if (header) print_header(names, app_unit, app_label);
  const std::string when = query_timestamp();
  for (uint32_t g : sel) {
    const auto& d = s.devices[g];
    for (uint32_t j = 0; j < d.proc_count && j < vgpu::telemetry::kMaxProcs; ++j) {
      const auto& pr = d.procs[j];
      for (size_t i = 0; i < names.size(); ++i) {
        const std::string& f = names[i];
        std::string v;
        if (f == "timestamp") v = when;
        else if (f == "gpu_serial") v = "[N/A]";
        else if (f == "pid") v = std::to_string(pr.pid);
        else if (f == "process_name" || f == "name") v = pr.name;
        else if (f == "used_memory" || f == "used_gpu_memory")
          v = std::to_string(mib(pr.used_bytes)) + (units ? " MiB" : "");
        else if (f == "gpu_uuid") v = d.uuid;
        else if (f == "gpu_bus_id") v = d.bus_id;
        else if (f == "gpu_name") v = d.name;
        else v = "[N/A]";
        std::printf("%s%s%s", i ? ", " : "", v.c_str(), i + 1 == names.size() ? "\n" : "");
      }
    }
  }
}

// The sections -d selects from the -q report, by nvidia-smi's names for them.
enum : unsigned {
  kSecMemory = 1u << 0,
  kSecUtilization = 1u << 1,
  kSecEcc = 1u << 2,
  kSecTemperature = 1u << 3,
  kSecPower = 1u << 4,
  kSecClock = 1u << 5,
  kSecCompute = 1u << 6,
  kSecPids = 1u << 7,
  kSecPerformance = 1u << 8,
  kSecAll = ~0u,
};

// The verbose "-q" report. Tools scrape it for identity and limits, so the
// indentation and the "key : value" alignment are part of the interface.
// `sections` is the -d filter. With one, each GPU's bus id is followed by only
// those blocks and the identity lines are left out, as nvidia-smi does; -d used
// to be skipped over, so `-q -d MEMORY` printed the whole report.
void print_verbose(const vgpu::telemetry::Shared& s, const std::vector<uint32_t>& sel,
                   unsigned sections) {
  const auto want = [&](unsigned sec) { return (sections & sec) != 0; };
  char when[64];
  std::time_t now = std::time(nullptr);
  std::tm tm{};
  ::localtime_r(&now, &tm);
  std::strftime(when, sizeof when, "%a %b %e %H:%M:%S %Y", &tm);
  // The same versions the table header prints. This used to say CUDA 13.0
  // whatever the session was, beside a header two commands away that said 12.4.
  std::printf("\n==============NVSMI LOG==============\n\n");
  std::printf("%-55s: %s\n", "Timestamp", when);
  std::printf("%-55s: %s\n", "Driver Version",
              s.driver_version[0] ? s.driver_version : vgpu::kDefaultDriverRelease);
  std::printf("%-55s: %s\n", "CUDA Version", s.cuda_version[0] ? s.cuda_version : "13.0");
  std::printf("\n%-55s: %u\n", "Attached GPUs", s.device_count);
  for (uint32_t i : sel) {
    const auto& d = s.devices[i];
    std::printf("GPU %s\n", d.bus_id);
    if (sections == kSecAll) {
      std::printf("    %-51s: %s\n", "Product Name", d.name);
      std::printf("    %-51s: %s\n", "Product Brand", brand_name(d));
      std::printf("    %-51s: %s\n", "Product Architecture", architecture_name(d.architecture).c_str());
      std::printf("    %-51s: %s\n", "Virtualization Mode", "Pass-Through");
      std::printf("    %-51s: %s\n", "Serial Number", "N/A");
      std::printf("    %-51s: %s\n", "GPU UUID", d.uuid);
      std::printf("    %-51s: %u.%u\n", "Compute Capability", d.cc_major, d.cc_minor);
      std::printf("    PCI\n");
      std::printf("        %-47s: %s\n", "Bus Id", d.bus_id);
      std::printf("        %-47s: 0x%08X\n", "Device Id", d.pci_device_id);
      if (d.pcie_gen_max) {
        // Current is the link as trained now, Max what it can train to.
        const vgpu::regs::Link l = link_of(d);
        std::printf("        GPU Link Info\n");
        std::printf("            PCIe Generation\n");
        std::printf("                %-39s: %u\n", "Max", l.max_gen);
        std::printf("                %-39s: %u\n", "Current", l.gen);
        std::printf("                %-39s: %u\n", "Device Current", l.gen);
        std::printf("                %-39s: %u\n", "Device Max", l.max_gen);
        std::printf("                %-39s: %u\n", "Host Max", l.max_gen);
        std::printf("            Link Width\n");
        std::printf("                %-39s: %ux\n", "Max", l.max_width);
        std::printf("                %-39s: %ux\n", "Current", l.width);
      }
      std::printf("    %-51s: %u %%\n", "Fan Speed", d.fan_percent);
    }
    if (want(kSecPerformance)) std::printf("    %-51s: P%u\n", "Performance State", d.perf_state);
    if (want(kSecMemory)) {
      std::printf("    FB Memory Usage\n");
      std::printf("        %-47s: %d MiB\n", "Total", mib(d.vram_total_bytes));
      std::printf("        %-47s: %d MiB\n", "Used", mib(d.vram_used_bytes));
      std::printf("        %-47s: %d MiB\n", "Free",
                  mib(d.vram_total_bytes) - mib(d.vram_used_bytes));
    }
    if (want(kSecCompute)) std::printf("    %-51s: %s\n", "Compute Mode", "Default");
    if (want(kSecUtilization)) {
      std::printf("    Utilization\n");
      std::printf("        %-47s: %u %%\n", "Gpu", d.utilization_gpu);
      std::printf("        %-47s: %u %%\n", "Memory", d.utilization_mem);
    }
    if (want(kSecEcc)) {
      // The profile says whether the card ships with ECC on; the table's ECC
      // column and the ecc.mode query fields give the same answer.
      const char* mode = d.ecc_enabled ? "Enabled" : "N/A";
      std::printf("    ECC Mode\n");
      std::printf("        %-47s: %s\n", "Current", mode);
      std::printf("        %-47s: %s\n", "Pending", mode);
    }
    if (want(kSecTemperature)) {
      std::printf("    Temperature\n");
      std::printf("        %-47s: %u C\n", "GPU Current Temp", d.temperature_c);
      if (d.has_memory_temperature)
        std::printf("        %-47s: %u C\n", "Memory Current Temp",
                    d.temperature_mem_c ? d.temperature_mem_c : d.temperature_c);
      else
        std::printf("        %-47s: %s\n", "Memory Current Temp", "N/A");
    }
    if (want(kSecPower)) {
      std::printf("    Power Readings\n");
      std::printf("        %-47s: %.2f W\n", "Power Draw", d.power_mw / 1000.0);
      std::printf("        %-47s: %.2f W\n", "Current Power Limit", d.power_limit_mw / 1000.0);
    }
    if (want(kSecClock)) {
      std::printf("    Clocks\n");
      std::printf("        %-47s: %u MHz\n", "SM", d.sm_clock_mhz);
      std::printf("        %-47s: %u MHz\n", "Memory", d.mem_clock_mhz);
    }
    if (want(kSecPids)) {
      if (d.proc_count == 0) {
        std::printf("    %-51s: %s\n", "Processes", "None");
      } else {
        std::printf("    Processes\n");
        for (uint32_t j = 0; j < d.proc_count && j < vgpu::telemetry::kMaxProcs; ++j) {
          const auto& pr = d.procs[j];
          std::printf("        %-47s: %s\n", "GPU instance ID", "N/A");
          std::printf("        %-47s: %s\n", "Compute instance ID", "N/A");
          std::printf("        %-47s: %u\n", "Process ID", pr.pid);
          std::printf("            %-43s: %s\n", "Type", "C");
          std::printf("            %-43s: %s\n", "Name", pr.name);
          std::printf("            %-43s: %d MiB\n", "Used GPU Memory", mib(pr.used_bytes));
        }
      }
    }
    std::printf("\n");
  }
}

// -d/--display: nvidia-smi's documented section names, comma-separated. The
// second list is documented too, but is state no profile has; those are
// refused by name rather than printed as an empty block that looks like an
// answer.
int parse_display(const std::string& spec, unsigned* out) {
  static const struct {
    const char* name;
    unsigned bit;
  } kKnown[] = {{"MEMORY", kSecMemory},           {"UTILIZATION", kSecUtilization},
                {"ECC", kSecEcc},                 {"TEMPERATURE", kSecTemperature},
                {"POWER", kSecPower},             {"CLOCK", kSecClock},
                {"COMPUTE", kSecCompute},         {"PIDS", kSecPids},
                {"PERFORMANCE", kSecPerformance}};
  static const char* const kNoData[] = {
      "SUPPORTED_CLOCKS", "PAGE_RETIREMENT", "ACCOUNTING",   "ENCODER_STATS",
      "SUPPORTED_GPU_TARGET_TEMP", "VOLTAGE", "FBC_STATS",  "ROW_REMAPPER",
      "RESET_STATUS", "GSP_FIRMWARE_VERSION"};
  unsigned mask = 0;
  for (const std::string& raw : split_fields(spec)) {
    std::string name = raw;
    for (char& ch : name) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    bool found = false;
    for (const auto& k : kKnown)
      if (name == k.name) { mask |= k.bit; found = true; }
    if (found) continue;
    for (const char* n : kNoData)
      if (name == n) {
        std::fprintf(stderr, "vgpu smi: -d %s is an nvidia-smi section VirtualGPU has no data for\n",
                     raw.c_str());
        return 2;
      }
    std::fprintf(stderr,
                 "vgpu smi: '%s' is not a -d section. Use MEMORY, UTILIZATION, ECC, TEMPERATURE,\n"
                 "          POWER, CLOCK, COMPUTE, PIDS or PERFORMANCE, comma-separated.\n",
                 raw.c_str());
    return 2;
  }
  if (mask == 0) {
    std::fprintf(stderr, "vgpu smi: -d needs at least one section, for example -d MEMORY\n");
    return 2;
  }
  *out = mask;
  return 0;
}

// The CPUs this process may run on, as nvidia-smi prints affinity: "0-15" or
// "0-7,16-23". Real, from the host -- the one part of a topology the
// simulator does not have to model.
std::string cpu_affinity() {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof set, &set) != 0) return "N/A";
  std::string out;
  for (int c = 0; c < CPU_SETSIZE;) {
    if (!CPU_ISSET(c, &set)) { ++c; continue; }
    int end = c;
    while (end + 1 < CPU_SETSIZE && CPU_ISSET(end + 1, &set)) ++end;
    if (!out.empty()) out += ",";
    out += end > c ? std::to_string(c) + "-" + std::to_string(end) : std::to_string(c);
    c = end + 1;
  }
  return out.empty() ? "N/A" : out;
}

int numa_nodes() {
  int n = 0;
  std::error_code ec;
  for (const auto& e : std::filesystem::directory_iterator("/sys/devices/system/node", ec)) {
    const std::string name = e.path().filename().string();
    if (name.rfind("node", 0) == 0 && name.size() > 4 &&
        name.find_first_not_of("0123456789", 4) == std::string::npos)
      ++n;
  }
  return n;
}

// `nvidia-smi topo -m`: how each pair of GPUs is connected, and which CPUs are
// near each one. Tab-separated, like the real matrix, because that is what
// scripts split it on.
//
// Every simulated device reaches every other through the host -- peer copies
// and NCCL's transport both go through host memory -- and no profile carries
// NVLink data measured from a card. So the honest link is PHB, "through a host
// bridge". An NV# here would be a number made up for a machine that has no
// links, and a topology-aware program would then act on it.
void print_topology(const vgpu::telemetry::Shared& s) {
  const std::string affinity = cpu_affinity();
  const std::string numa = numa_nodes() == 1 ? "0" : "N/A";
  std::printf("\t");
  for (uint32_t j = 0; j < s.device_count; ++j) std::printf("GPU%u\t", j);
  std::printf("CPU Affinity\tNUMA Affinity\tGPU NUMA ID\n");
  for (uint32_t i = 0; i < s.device_count; ++i) {
    std::printf("GPU%u\t", i);
    for (uint32_t j = 0; j < s.device_count; ++j) std::printf("%s\t", i == j ? " X " : "PHB");
    std::printf("%s\t%s\t\tN/A\n", affinity.c_str(), numa.c_str());
  }
  std::printf("\nLegend:\n\n"
              "  X    = Self\n"
              "  SYS  = Across PCIe and the interconnect between NUMA nodes\n"
              "  NODE = Across PCIe and the host bridges within one NUMA node\n"
              "  PHB  = Across PCIe and a PCIe host bridge (usually the CPU)\n"
              "  PXB  = Across more than one PCIe bridge, without the host bridge\n"
              "  PIX  = Across at most one PCIe bridge\n"
              "  NV#  = Across a bonded set of # NVLinks\n");
}

std::string xml_escape(const char* v) {
  std::string out;
  for (const char* p = v; *p; ++p) {
    switch (*p) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      case '"': out += "&quot;"; break;
      default: out += *p;
    }
  }
  return out;
}

// `nvidia-smi -q -x`: the verbose report as XML, under the element names the
// real tool uses (nvsmi_device's schema), for the tools that parse XML rather
// than scrape text. Same values as -q, from the same snapshot.
void print_verbose_xml(const vgpu::telemetry::Shared& s, const std::vector<uint32_t>& sel) {
  char when[64];
  std::time_t now = std::time(nullptr);
  std::tm tm{};
  ::localtime_r(&now, &tm);
  std::strftime(when, sizeof when, "%a %b %e %H:%M:%S %Y", &tm);
  std::printf("<?xml version=\"1.0\" ?>\n");
  std::printf("<!DOCTYPE nvidia_smi_log SYSTEM \"nvsmi_device_v12.dtd\">\n");
  std::printf("<nvidia_smi_log>\n");
  std::printf("\t<timestamp>%s</timestamp>\n", when);
  std::printf("\t<driver_version>%s</driver_version>\n",
              xml_escape(s.driver_version[0] ? s.driver_version : vgpu::kDefaultDriverRelease).c_str());
  std::printf("\t<cuda_version>%s</cuda_version>\n",
              xml_escape(s.cuda_version[0] ? s.cuda_version : "13.0").c_str());
  std::printf("\t<attached_gpus>%u</attached_gpus>\n", s.device_count);
  for (uint32_t i : sel) {
    const auto& d = s.devices[i];
    unsigned domain = 0, bus = 0, dev = 0, fn = 0;
    std::sscanf(d.bus_id, "%x:%x:%x.%x", &domain, &bus, &dev, &fn);
    std::printf("\t<gpu id=\"%s\">\n", xml_escape(d.bus_id).c_str());
    std::printf("\t\t<product_name>%s</product_name>\n", xml_escape(d.name).c_str());
    std::printf("\t\t<product_brand>%s</product_brand>\n", brand_name(d));
    std::printf("\t\t<product_architecture>%s</product_architecture>\n",
                xml_escape(architecture_name(d.architecture).c_str()).c_str());
    std::printf("\t\t<persistence_mode>Enabled</persistence_mode>\n");
    std::printf("\t\t<serial>N/A</serial>\n");
    std::printf("\t\t<uuid>%s</uuid>\n", xml_escape(d.uuid).c_str());
    std::printf("\t\t<minor_number>%u</minor_number>\n", i);
    std::printf("\t\t<pci>\n");
    std::printf("\t\t\t<pci_bus>%02X</pci_bus>\n", bus);
    std::printf("\t\t\t<pci_device>%02X</pci_device>\n", dev);
    std::printf("\t\t\t<pci_domain>%04X</pci_domain>\n", domain);
    std::printf("\t\t\t<pci_device_id>%08X</pci_device_id>\n", d.pci_device_id);
    std::printf("\t\t\t<pci_bus_id>%s</pci_bus_id>\n", xml_escape(d.bus_id).c_str());
    std::printf("\t\t\t<pci_sub_system_id>%08X</pci_sub_system_id>\n", d.pci_subsystem_id);
    std::printf("\t\t</pci>\n");
    std::printf("\t\t<fan_speed>%u %%</fan_speed>\n", d.fan_percent);
    std::printf("\t\t<performance_state>P%u</performance_state>\n", d.perf_state);
    std::printf("\t\t<fb_memory_usage>\n");
    std::printf("\t\t\t<total>%d MiB</total>\n", mib(d.vram_total_bytes));
    std::printf("\t\t\t<used>%d MiB</used>\n", mib(d.vram_used_bytes));
    std::printf("\t\t\t<free>%d MiB</free>\n", mib(d.vram_total_bytes) - mib(d.vram_used_bytes));
    std::printf("\t\t</fb_memory_usage>\n");
    std::printf("\t\t<compute_mode>Default</compute_mode>\n");
    std::printf("\t\t<utilization>\n");
    std::printf("\t\t\t<gpu_util>%u %%</gpu_util>\n", d.utilization_gpu);
    std::printf("\t\t\t<memory_util>%u %%</memory_util>\n", d.utilization_mem);
    std::printf("\t\t</utilization>\n");
    std::printf("\t\t<temperature>\n");
    std::printf("\t\t\t<gpu_temp>%u C</gpu_temp>\n", d.temperature_c);
    std::printf("\t\t</temperature>\n");
    std::printf("\t\t<gpu_power_readings>\n");
    std::printf("\t\t\t<power_state>P%u</power_state>\n", d.perf_state);
    std::printf("\t\t\t<power_draw>%.2f W</power_draw>\n", d.power_mw / 1000.0);
    std::printf("\t\t\t<current_power_limit>%.2f W</current_power_limit>\n", d.power_limit_mw / 1000.0);
    std::printf("\t\t</gpu_power_readings>\n");
    std::printf("\t\t<clocks>\n");
    std::printf("\t\t\t<graphics_clock>%u MHz</graphics_clock>\n", d.sm_clock_mhz);
    std::printf("\t\t\t<mem_clock>%u MHz</mem_clock>\n", d.mem_clock_mhz);
    std::printf("\t\t</clocks>\n");
    std::printf("\t\t<processes>\n");
    for (uint32_t j = 0; j < d.proc_count && j < vgpu::telemetry::kMaxProcs; ++j) {
      const auto& pr = d.procs[j];
      std::printf("\t\t\t<process_info>\n");
      std::printf("\t\t\t\t<pid>%u</pid>\n", pr.pid);
      std::printf("\t\t\t\t<type>C</type>\n");
      std::printf("\t\t\t\t<process_name>%s</process_name>\n", xml_escape(pr.name).c_str());
      std::printf("\t\t\t\t<used_memory>%d MiB</used_memory>\n", mib(pr.used_bytes));
      std::printf("\t\t\t</process_info>\n");
    }
    std::printf("\t\t</processes>\n");
    std::printf("\t</gpu>\n");
  }
  std::printf("</nvidia_smi_log>\n");
}

// The -i selection applies here and to the rocm-smi table as it does to the
// nvidia-smi table: both used to print every device whatever was selected.
void print_csv(const vgpu::telemetry::Shared& s, const std::vector<uint32_t>& sel) {
  std::printf("index,name,uuid,bus_id,memory.total,memory.used,utilization.gpu,temperature.gpu,"
              "power.draw,power.limit,clocks.sm,clocks.mem\n");
  for (uint32_t i : sel) {
    const auto& d = s.devices[i];
    std::printf("%u,%s,%s,%s,%d MiB,%d MiB,%u %%,%u,%.2f W,%.2f W,%u MHz,%u MHz\n", i, d.name,
                d.uuid, d.bus_id, mib(d.vram_total_bytes), mib(d.vram_used_bytes),
                d.utilization_gpu, d.temperature_c, d.power_mw / 1000.0, d.power_limit_mw / 1000.0,
                d.sm_clock_mhz, d.mem_clock_mhz);
  }
}

// ROCm's rocm-smi table, for virtual AMD devices (and NVIDIA ones, which it
// shows too so a mixed rack is visible from one tool).
void print_rocm(const vgpu::telemetry::Shared& s, const std::vector<uint32_t>& sel) {
  std::printf("\n");
  std::printf("========================= ROCm System Management Interface =========================\n");
  std::printf("=================================== Concise Info ===================================\n");
  std::printf("GPU  Temp   AvgPwr  SCLK     MCLK     Fan   Perf  PwrCap  VRAM%%  GPU%%\n");
  for (uint32_t i : sel) {
    const auto& d = s.devices[i];
    uint32_t vram_pct = d.vram_total_bytes
                            ? static_cast<uint32_t>(d.vram_used_bytes * 100 / d.vram_total_bytes)
                            : 0;
    std::printf("%-4u %-6s %-7s %-8s %-8s %-5s %-5s %-7s %-6u %-4u\n", i,
                (std::to_string(d.temperature_c) + "c").c_str(),
                (std::to_string(d.power_mw / 1000) + "W").c_str(),
                (std::to_string(d.sm_clock_mhz) + "Mhz").c_str(),
                (std::to_string(d.mem_clock_mhz) + "Mhz").c_str(),
                (std::to_string(d.fan_percent) + "%").c_str(),
                (std::string("auto")).c_str(),
                (std::to_string(d.power_limit_mw / 1000) + "W").c_str(), vram_pct,
                d.utilization_gpu);
  }
  std::printf("====================================================================================\n");
  std::printf("=============================== End of ROCm SMI Log ================================\n");
}

// CDNA3 -> gfx942, CDNA4 -> gfx950 (the targets these parts report).
const char* gfx_target(const vgpu::telemetry::DeviceSample& d) {
  return std::strcmp(d.architecture, "cdna4") == 0 ? "gfx950" : "gfx942";
}

// rocm_agent_enumerator output: one ISA target per line, CPU agent first.
// Returns how many GPU agents were listed.
int print_agents(const vgpu::telemetry::Shared& s) {
  std::printf("gfx000\n");  // the host CPU agent, as ROCm reports
  int listed = 0;
  for (uint32_t i = 0; i < s.device_count; ++i) {
    const auto& d = s.devices[i];
    if (std::strcmp(d.vendor, "amd") != 0) continue;
    std::printf("%s\n", gfx_target(d));
    ++listed;
  }
  return listed;
}

// An lspci view of the virtual devices.
//
// `--lspci` prints the familiar one-line-per-device listing. `--lspci-dump`
// emits each device's configuration space -- all 4096 bytes, from the register
// model (vgpu/regs.hpp) -- in `lspci -xxxx` format, which the REAL lspci renders
// with `lspci -F <file>`, capabilities and all: `lspci -F <file> -vvv` shows the
// link's speed and width and the AER status the fault model has set.
void print_lspci(const vgpu::telemetry::Shared& s, bool dump) {
  for (uint32_t i = 0; i < s.device_count; ++i) {
    const auto& d = s.devices[i];
    // NVML-style bus ids carry an 8-digit domain; lspci uses 4.
    const char* bdf = std::strlen(d.bus_id) > 4 ? d.bus_id + 4 : d.bus_id;
    const char* vendor_name = std::strcmp(d.vendor, "amd") == 0
                                  ? "Advanced Micro Devices, Inc. [AMD/ATI]"
                                  : "NVIDIA Corporation";
    vgpu::regs::ConfigSpace cs(d);
    const std::vector<uint8_t> cfg = cs.image(dump ? vgpu::regs::kConfigSize : 16);
    const uint32_t cls = static_cast<uint32_t>(cfg[0x0b]) << 8 | cfg[0x0a];
    const char* class_name = cls == 0x0300 ? "VGA compatible controller"
                             : cls == 0x1200 ? "Processing accelerators"
                                             : "3D controller";
    std::printf("%s %s: %s %s (rev %02x)\n", bdf, class_name, vendor_name, d.name, cfg[0x08]);
    if (!dump) continue;
    for (uint32_t row = 0; row < vgpu::regs::kConfigSize / 16; ++row) {
      std::printf("%02x:", row * 16);
      for (uint32_t col = 0; col < 16; ++col) std::printf(" %02x", cfg[row * 16 + col]);
      std::printf("\n");
    }
    std::printf("\n");
  }
}

void print_explain() {
  std::printf(
      "vgpu smi reports two kinds of value:\n"
      "\n"
      "  REAL (measured from the simulator)\n"
      "    memory.used / memory.total   exact allocator state\n"
      "    utilization.gpu              fraction of wall time inside kernel launches\n"
      "    kernels, bytes moved         exact counters\n"
      "\n"
      "  SYNTHETIC (a deterministic model driven by the real utilization above)\n"
      "    power, temperature, voltage, clocks, fan, perf state\n"
      "\n"
      "The synthetic columns move the way a real card's would -- they rise under\n"
      "load, lag behind it, and decay when idle -- so monitoring tools and\n"
      "dashboards behave. They are NOT predictions of any physical device's power\n"
      "or thermal behavior. VirtualGPU does not model performance.\n");
}

}  // namespace

namespace {

using vgpu::cli::read_machine;

// `nvidia-smi --version`: the four lines the real tool prints, every one a
// number. NVML's version is the CUDA major followed by the driver release
// ("12.550.54.15"), the same default the NVML shim reports. Both drop-in
// scripts and the session's wrapper reach this one function; they used to
// print three different texts, one of them "VirtualGPU (simulated)".
void print_version() {
  const char* drv = std::getenv("VGPU_DRIVER_VERSION");
  const std::string driver = drv && *drv ? drv : vgpu::kDefaultDriverRelease;
  const char* nvml_env = std::getenv("VGPU_NVML_VERSION");
  const std::string nvml = nvml_env && *nvml_env
                               ? nvml_env
                               : std::to_string(vgpu::driver_version() / 1000) + "." + driver;
  std::printf("NVIDIA-SMI version  : %s\n", driver.c_str());
  std::printf("NVML version        : %s\n", nvml.c_str());
  std::printf("DRIVER version      : %s\n", driver.c_str());
  std::printf("CUDA Version        : %s\n", vgpu::driver_version_string().c_str());
}

void print_smi_usage(FILE* to) {
  std::fprintf(
      to,
      "Usage: nvidia-smi [options]   (VirtualGPU's drop-in; also `vgpu smi`)\n"
      "\n"
      "  -h, --help                  This help\n"
      "      --version               Driver, NVML and CUDA versions\n"
      "  -L, --list-gpus             One line per GPU\n"
      "  -i, --id=ID                 Only these GPUs: index, UUID or PCI bus id, comma-separated\n"
      "  -q, --query                 The verbose report\n"
      "  -d, --display=SECTIONS      With -q, only these sections, comma-separated: MEMORY,\n"
      "                              UTILIZATION, ECC, TEMPERATURE, POWER, CLOCK, COMPUTE,\n"
      "                              PIDS, PERFORMANCE\n"
      "  -x, --xml-format            The verbose report as XML\n"
      "      --query-gpu=FIELDS      Chosen fields per GPU (see --help-query-gpu); needs --format\n"
      "      --query-compute-apps=FIELDS\n"
      "                              Chosen fields per compute process; needs --format\n"
      "      --format=csv[,noheader][,nounits]\n"
      "  -l, --loop[=SEC]            Repeat every SEC seconds (default 5) until interrupted\n"
      "  -lms, --loop-ms=MS          Repeat every MS milliseconds until interrupted\n"
      "  topo -m                     How the GPUs are connected\n"
      "\n"
      "VirtualGPU additions:\n"
      "      --details               Append the virtual device details to the table\n"
      "      --explain               Which values are measured and which are modelled\n"
      "      --csv                   A fixed CSV of the main columns\n"
      "      --rocm [ARGS]           rocm-smi, with rocm-smi's own arguments (--rocm --help)\n"
      "      --agents                rocm_agent_enumerator's output\n"
      "      --lspci, --lspci-dump   A PCI listing, or config space for `lspci -F`\n"
      "\n"
      "Exit status: 0 success, 2 invalid argument, 6 no such device.\n");
}

void print_topo_usage(FILE* to) {
  std::fprintf(to,
               "Usage: nvidia-smi topo -m\n"
               "\n"
               "  -m, --matrix   GPU-to-GPU connections and CPU affinity\n"
               "\n"
               "The pairwise forms (-p2p, -i) need link data no device profile carries.\n");
}

volatile std::sig_atomic_t g_loop_stop = 0;
void on_loop_signal(int) { g_loop_stop = 1; }

// ---------------------------------------------------------------------------
// rocm-smi
// ---------------------------------------------------------------------------

void print_rocm_usage(FILE* to) {
  std::fprintf(
      to,
      "usage: rocm-smi [-h] [--version] [-d DEVICE [DEVICE ...]] [-a] [-i]\n"
      "                [--showproductname] [--showmeminfo TYPE [TYPE ...]] [-t] [-P] [-u]\n"
      "                [-v] [-c] [-f] [--showmaxpower] [--showserial] [--showuniqueid]\n"
      "                [--showmemvendor] [--showrasinfo [BLOCK ...]] [--json] [--csv]\n"
      "\n"
      "VirtualGPU's drop-in rocm-smi. With no option it prints the concise table.\n"
      "\n"
      "  -d, --device DEVICE ...   only these GPUs, by index\n"
      "  -a, --showallinfo         every section below\n"
      "  -i, --showid              device name and PCI device id\n"
      "      --showproductname     card series, model, vendor and gfx target\n"
      "      --showmeminfo TYPE    vram, vis_vram, gtt or all (only vram is modelled)\n"
      "  -t, --showtemp            temperature\n"
      "  -P, --showpower           power draw\n"
      "  -u, --showuse             GPU use\n"
      "  -v, --showvbios           VBIOS version (none is modelled: N/A)\n"
      "  -c, --showclocks          current sclk and mclk\n"
      "  -f, --showfan             fan speed (N/A: Instinct cards are passively cooled)\n"
      "      --showmaxpower        power cap\n"
      "      --showserial          serial number (N/A)\n"
      "      --showuniqueid        unique ID, stable per device\n"
      "      --showmemvendor       memory vendor (unknown: no profile records it)\n"
      "      --showrasinfo [BLOCK] ECC state and error counts per RAS block\n"
      "      --json                the selected values as JSON\n"
      "      --csv                 the selected values as CSV\n");
}

// "=== Title ===" centred in rocm-smi's 84-column rule, like print_rocm's banners.
void rocm_rule(const std::string& title) {
  constexpr int kWidth = 84;
  const std::string t = title.empty() ? "" : " " + title + " ";
  const int left = std::max(0, (kWidth - static_cast<int>(t.size())) / 2);
  const int right = std::max(0, kWidth - left - static_cast<int>(t.size()));
  std::printf("%s%s%s\n", std::string(left, '=').c_str(), t.c_str(),
              std::string(right, '=').c_str());
}

std::string json_string(const std::string& v) {
  std::string out = "\"";
  for (char ch : v) {
    if (ch == '"' || ch == '\\') out += '\\';
    if (static_cast<unsigned char>(ch) < 0x20) {
      char esc[8];
      std::snprintf(esc, sizeof esc, "\\u%04x", ch);
      out += esc;
      continue;
    }
    out += ch;
  }
  return out + "\"";
}

std::string csv_cell(const std::string& v) {
  if (v.find_first_of(",\"\n") == std::string::npos) return v;
  std::string out = "\"";
  for (char ch : v) out += ch == '"' ? std::string("\"\"") : std::string(1, ch);
  return out + "\"";
}

// rocm-smi, parsed the way rocm-smi parses: its documented option names, and
// -d for the device selection (rocm-smi's -i is --showid). Both the drop-in
// script and the session's wrapper come here. The session used to hand these
// to the nvidia-smi parser, which rejected every one, and the drop-in script
// ignored them, printing the nvidia-smi table for -a.
int cmd_rocm_smi(const std::vector<std::string>& args) {
  bool json = false, csv = false, all = false, show_id = false, show_product = false,
       show_temp = false, show_power = false, show_use = false, show_vbios = false,
       show_clocks = false, show_fan = false, show_maxpower = false, show_serial = false,
       show_uniqueid = false, show_memvendor = false, show_ras = false;
  // rocm-smi's RAS block names, in the order its table lists them.
  static const char* const kRasBlocks[] = {"umc", "sdma", "gfx", "mmhub", "athub", "pcie_bif",
                                           "hdp", "xgmi_wafl", "df", "smn", "sem", "mp0", "mp1",
                                           "fuse"};
  std::vector<std::string> ras_blocks;
  std::vector<std::string> mem_types;
  std::vector<long long> want;
  auto error = [](const std::string& msg) {
    std::fprintf(stderr, "rocm-smi: error: %s\n", msg.c_str());
    return 2;
  };
  // argparse's nargs='+': every following word that is not an option.
  auto words = [&](size_t& i) {
    std::vector<std::string> out;
    while (i + 1 < args.size() && !args[i + 1].empty() && args[i + 1][0] != '-')
      out.push_back(args[++i]);
    return out;
  };
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a == "-h" || a == "--help") {
      print_rocm_usage(stdout);
      return 0;
    }
    if (a == "--version") {
      std::printf("ROCm SMI version: VirtualGPU 0.1.0\n");
      return 0;
    }
    if (a == "-a" || a == "--showallinfo") all = true;
    else if (a == "-i" || a == "--showid") show_id = true;
    else if (a == "--showproductname") show_product = true;
    else if (a == "-t" || a == "--showtemp") show_temp = true;
    else if (a == "-P" || a == "--showpower") show_power = true;
    else if (a == "-u" || a == "--showuse") show_use = true;
    // The forms Pantheon's AMD telemetry poll and inventory use. Each used to be
    // refused, which left it with no AMD telemetry and an empty GPU list.
    else if (a == "-v" || a == "--showvbios") show_vbios = true;
    else if (a == "-c" || a == "--showclocks") show_clocks = true;
    else if (a == "-f" || a == "--showfan") show_fan = true;
    else if (a == "--showmaxpower") show_maxpower = true;
    else if (a == "--showserial") show_serial = true;
    else if (a == "--showuniqueid") show_uniqueid = true;
    else if (a == "--showmemvendor") show_memvendor = true;
    else if (a == "--showrasinfo") {
      // An optional list of blocks; none means every block.
      for (const std::string& b : words(i)) {
        if (std::find_if(std::begin(kRasBlocks), std::end(kRasBlocks),
                         [&](const char* k) { return b == k; }) == std::end(kRasBlocks))
          return error("argument --showrasinfo: invalid choice: '" + b + "'");
        ras_blocks.push_back(b);
      }
      show_ras = true;
    }
    else if (a == "--json") json = true;
    else if (a == "--csv") csv = true;
    else if (a == "--showmeminfo") {
      const std::vector<std::string> types = words(i);
      if (types.empty()) return error("argument --showmeminfo: expected at least one argument");
      for (const std::string& t : types) {
        if (t == "all") {
          mem_types.insert(mem_types.end(), {"vram", "vis_vram", "gtt"});
        } else if (t == "vram" || t == "vis_vram" || t == "gtt") {
          mem_types.push_back(t);
        } else {
          return error("argument --showmeminfo: invalid choice: '" + t +
                       "' (choose from 'vram', 'vis_vram', 'gtt', 'all')");
        }
      }
    } else if (a == "-d" || a == "--device") {
      const std::vector<std::string> ids = words(i);
      if (ids.empty()) return error("argument -d/--device: expected at least one argument");
      for (const std::string& id : ids) {
        long long n = 0;
        if (!vgpu::cli::parse_int(id, 0, 1 << 20, &n))
          return error("argument -d/--device: invalid int value: '" + id + "'");
        want.push_back(n);
      }
    } else {
      return error("unrecognized arguments: " + a);
    }
  }

  vgpu::telemetry::Shared snap{};
  if (!read_machine(&snap)) return 1;
  std::vector<uint32_t> sel;
  if (want.empty()) {
    for (uint32_t i = 0; i < snap.device_count; ++i) sel.push_back(i);
  } else {
    for (long long n : want) {
      if (n >= static_cast<long long>(snap.device_count)) {
        std::fprintf(stderr, "rocm-smi: error: there is no GPU[%lld]; this machine has %u\n", n,
                     snap.device_count);
        return 2;
      }
      if (std::find(sel.begin(), sel.end(), static_cast<uint32_t>(n)) == sel.end())
        sel.push_back(static_cast<uint32_t>(n));
    }
  }

  const bool any_show = all || show_id || show_product || show_temp || show_power || show_use ||
                        show_vbios || show_clocks || show_fan || show_maxpower || show_serial ||
                        show_uniqueid || show_memvendor || show_ras || !mem_types.empty();
  if (!any_show && !json && !csv) {
    print_rocm(snap, sel);
    return 0;
  }
  // JSON or CSV with nothing selected carries what the concise table shows.
  if (!any_show) {
    show_temp = show_power = show_use = true;
    mem_types = {"vram"};
  }
  if (all) {
    show_id = show_product = show_temp = show_power = show_use = true;
    show_vbios = show_clocks = show_fan = show_maxpower = show_serial = show_uniqueid =
        show_memvendor = show_ras = true;
    mem_types = {"vram", "vis_vram", "gtt"};
  }

  using Values = std::vector<std::pair<std::string, std::string>>;
  struct Block {
    std::string title;
    std::vector<Values> rows;  // one per entry of sel
  };
  std::vector<Block> blocks;
  auto add = [&](const char* title, auto fill) {
    Block b{title, {}};
    for (uint32_t i : sel) {
      b.rows.emplace_back();
      fill(snap.devices[i], b.rows.back());
    }
    blocks.push_back(std::move(b));
  };
  auto hex4 = [](unsigned v) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "0x%04x", v);
    return std::string(buf);
  };
  using Sample = vgpu::telemetry::DeviceSample;
  const auto is_amd = [](const Sample& d) { return std::strcmp(d.vendor, "amd") == 0; };
  if (show_id)
    add("ID", [&](const Sample& d, Values& v) {
      v.emplace_back("Device Name", d.name);
      v.emplace_back("Device ID", hex4((d.pci_device_id >> 16) & 0xFFFF));
    });
  if (show_product)
    add("Product Info", [&](const Sample& d, Values& v) {
      v.emplace_back("Card Series", d.name);
      v.emplace_back("Card Model", hex4((d.pci_device_id >> 16) & 0xFFFF));
      v.emplace_back("Card Vendor", is_amd(d) ? "Advanced Micro Devices, Inc. [AMD/ATI]"
                                              : "NVIDIA Corporation");
      if (is_amd(d)) v.emplace_back("GFX Version", gfx_target(d));
    });
  if (!mem_types.empty())
    add("Memory Usage (Bytes)", [&](const Sample& d, Values& v) {
      for (const char* t : {"vram", "vis_vram", "gtt"}) {
        if (std::find(mem_types.begin(), mem_types.end(), t) == mem_types.end()) continue;
        if (std::strcmp(t, "vram") == 0) {
          v.emplace_back("VRAM Total Memory (B)", std::to_string(d.vram_total_bytes));
          v.emplace_back("VRAM Total Used Memory (B)", std::to_string(d.vram_used_bytes));
        } else {
          // Neither the CPU-visible aperture nor GTT is in any profile, so they
          // are N/A rather than a copy of the VRAM figures.
          const std::string up = std::strcmp(t, "gtt") == 0 ? "GTT" : "VIS_VRAM";
          v.emplace_back(up + " Total Memory (B)", "N/A");
          v.emplace_back(up + " Total Used Memory (B)", "N/A");
        }
      }
    });
  if (show_temp)
    add("Temperature", [&](const Sample& d, Values& v) {
      // One synthetic die temperature serves as both edge and junction. Memory
      // is reported only where the profile says real cards have the sensor.
      v.emplace_back("Temperature (Sensor edge) (C)", std::to_string(d.temperature_c) + ".0");
      v.emplace_back("Temperature (Sensor junction) (C)", std::to_string(d.temperature_c) + ".0");
      v.emplace_back("Temperature (Sensor memory) (C)",
                     d.has_memory_temperature
                         ? std::to_string(d.temperature_mem_c ? d.temperature_mem_c
                                                              : d.temperature_c) + ".0"
                         : "N/A");
    });
  if (show_power)
    add("Power Consumption", [&](const Sample& d, Values& v) {
      char buf[32];
      std::snprintf(buf, sizeof buf, "%.1f", d.power_mw / 1000.0);
      v.emplace_back("Current Socket Graphics Package Power (W)", buf);
    });
  if (show_use)
    add("% time GPU is busy", [&](const Sample& d, Values& v) {
      v.emplace_back("GPU use (%)", std::to_string(d.utilization_gpu));
    });
  if (show_vbios)
    add("VBIOS", [&](const Sample&, Values& v) {
      // No profile records a VBIOS, so N/A -- the answer NVML gives as well.
      v.emplace_back("VBIOS version", "N/A");
    });
  if (show_clocks)
    add("Current clock frequencies", [&](const Sample& d, Values& v) {
      v.emplace_back("mclk clock speed:", "(" + std::to_string(d.mem_clock_mhz) + "Mhz)");
      v.emplace_back("sclk clock speed:", "(" + std::to_string(d.sm_clock_mhz) + "Mhz)");
    });
  if (show_fan)
    add("Fan speed", [&](const Sample& d, Values& v) {
      // Instinct accelerators are passively cooled, so there is no fan to read.
      const std::string pct = is_amd(d) ? "N/A" : std::to_string(d.fan_percent);
      v.emplace_back("Fan speed (level)", "N/A");
      v.emplace_back("Fan speed (%)", pct);
      v.emplace_back("Fan RPM", "N/A");
    });
  if (show_maxpower)
    add("Power Cap", [&](const Sample& d, Values& v) {
      char buf[32];
      std::snprintf(buf, sizeof buf, "%.1f", d.power_limit_mw / 1000.0);
      v.emplace_back("Max Graphics Package Power (W)", buf);
    });
  if (show_serial)
    add("Serial Number", [&](const Sample&, Values& v) { v.emplace_back("Serial Number", "N/A"); });
  if (show_uniqueid)
    add("Unique ID", [&](const Sample& d, Values& v) {
      // Stable per profile and device: derived from the UUID NVML reports.
      uint64_t h = 1469598103934665603ull;
      for (const char* c = d.uuid; *c; ++c) h = (h ^ static_cast<unsigned char>(*c)) * 1099511628211ull;
      char buf[24];
      std::snprintf(buf, sizeof buf, "0x%016llx", static_cast<unsigned long long>(h));
      v.emplace_back("Unique ID", buf);
    });
  if (show_memvendor)
    add("GPU memory vendor", [&](const Sample&, Values& v) {
      // No profile records whose memory a card carries.
      v.emplace_back("GPU memory vendor", "unknown");
    });
  if (show_ras)
    add("RAS Info", [&](const Sample& d, Values& v) {
      // Per-block ECC state and the counts since the driver loaded, from the
      // same state nvidia-smi reads: device memory is the UMC (memory
      // controller) block, and the on-chip memories are GFX. The key names
      // follow the columns of rocm-smi's RAS table and are not yet checked
      // against a real MI-series card.
      const vgpu::ras::Counters c = vgpu::ras::read(d.uuid).since_load;
      auto count = [&](const std::string& blk, vgpu::ras::Severity s) -> uint64_t {
        const auto si = static_cast<uint32_t>(s);
        if (blk == "umc") return c.ecc[si][static_cast<uint32_t>(vgpu::ras::Location::DeviceMemory)];
        if (blk == "gfx") return c.ecc_total(s) - c.ecc[si][static_cast<uint32_t>(vgpu::ras::Location::DeviceMemory)];
        return 0;
      };
      for (const char* block : kRasBlocks) {
        if (!ras_blocks.empty() &&
            std::find(ras_blocks.begin(), ras_blocks.end(), block) == ras_blocks.end())
          continue;
        std::string up = block;
        for (char& ch : up) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        v.emplace_back(up + " RAS status", d.ecc_enabled ? "ENABLED" : "DISABLED");
        v.emplace_back(up + " correctable errors",
                       d.ecc_enabled ? std::to_string(count(block, vgpu::ras::Severity::Corrected)) : "N/A");
        v.emplace_back(up + " uncorrectable errors",
                       d.ecc_enabled ? std::to_string(count(block, vgpu::ras::Severity::Uncorrected)) : "N/A");
      }
    });

  if (json) {
    std::printf("{");
    for (size_t k = 0; k < sel.size(); ++k) {
      std::printf("%s\"card%u\": {", k ? ", " : "", sel[k]);
      bool first = true;
      for (const Block& b : blocks)
        for (const auto& [key, val] : b.rows[k]) {
          std::printf("%s%s: %s", first ? "" : ", ", json_string(key).c_str(),
                      json_string(val).c_str());
          first = false;
        }
      std::printf("}");
    }
    std::printf("}\n");
    return 0;
  }
  if (csv) {
    std::printf("device");
    for (const Block& b : blocks)
      for (const auto& kv : b.rows.front()) std::printf(",%s", csv_cell(kv.first).c_str());
    std::printf("\n");
    for (size_t k = 0; k < sel.size(); ++k) {
      std::printf("card%u", sel[k]);
      for (const Block& b : blocks)
        for (const auto& kv : b.rows[k]) std::printf(",%s", csv_cell(kv.second).c_str());
      std::printf("\n");
    }
    return 0;
  }
  std::printf("\n");
  rocm_rule("ROCm System Management Interface");
  for (const Block& b : blocks) {
    rocm_rule(b.title);
    for (size_t k = 0; k < sel.size(); ++k)
      for (const auto& [key, val] : b.rows[k])
        std::printf("GPU[%u]\t\t: %s: %s\n", sel[k], key.c_str(), val.c_str());
    rocm_rule("");
  }
  rocm_rule("End of ROCm SMI Log");
  return 0;
}

}  // namespace

// amd-smi (amdsmi.cpp).
int cmd_amd_smi(const std::vector<std::string>& args);

int cmd_smi(const std::vector<std::string>& args) {
  // rocm-smi has arguments of its own, some of which mean something else to
  // nvidia-smi (-i, -d), so it gets its own parser.
  for (size_t k = 0; k < args.size(); ++k) {
    if (args[k] == "--rocm" || args[k] == "--amd") {
      std::vector<std::string> rest(args.begin(), args.end());
      rest.erase(rest.begin() + static_cast<std::ptrdiff_t>(k));
      return args[k] == "--rocm" ? cmd_rocm_smi(rest) : cmd_amd_smi(rest);
    }
  }

  bool csv = false, explain = false, agents = false, lspci = false, lspci_dump = false,
       verbose = false, details = false, help = false, version = false;
  std::string query_fields, app_fields, id_spec, format, display;
  bool have_query_gpu = false, have_query_apps = false, have_format = false, have_display = false;
  bool header = true, units = true, list = false, xml = false, topo = false;
  int reset_ecc = -1;   // -p: 0 volatile, 1 aggregate
  long long loop_ms = 0;
  auto fail = [](const std::string& msg) {
    std::fprintf(stderr, "vgpu smi: %s\n", msg.c_str());
    return 2;
  };
  // Seconds or milliseconds, whole and positive; a day is a generous ceiling.
  auto interval = [&](const std::string& v, long long unit_ms, const char* flag) {
    long long n = 0;
    if (!vgpu::cli::parse_int(v, 1, 86400000LL / unit_ms, &n)) {
      fail(std::string(flag) + " needs a positive whole number, got '" + v + "'");
      return false;
    }
    loop_ms = n * unit_ms;
    return true;
  };
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a.rfind("--query-gpu=", 0) == 0) {
      have_query_gpu = true;
      query_fields = a.substr(std::string("--query-gpu=").size());
    } else if (a.rfind("--query-compute-apps=", 0) == 0) {
      have_query_apps = true;
      app_fields = a.substr(std::string("--query-compute-apps=").size());
    } else if (a == "--query-gpu" || a == "--query-compute-apps") {
      return fail(a + " takes its fields after '=', for example " + a + "=index,name");
    } else if (a == "-i" || a == "--id") {
      if (i + 1 >= args.size()) {
        std::fprintf(stderr, "vgpu smi: %s needs a device\n", a.c_str());
        return 2;
      }
      id_spec = args[++i];
    } else if (a.rfind("--id=", 0) == 0 || a.rfind("-i=", 0) == 0) {
      id_spec = a.substr(a.find('=') + 1);
    } else if (a.rfind("--format=", 0) == 0) {
      have_format = true;
      format = a.substr(std::string("--format=").size());
    } else if (a == "-q" || a == "--query") {
      verbose = true;
    } else if (a == "-x" || a == "--xml-format") {
      xml = true;
    } else if (a == "-d" || a == "--display") {
      if (i + 1 >= args.size()) return fail(a + " needs a section, for example -d MEMORY");
      have_display = true;
      display = args[++i];
    } else if (a.rfind("--display=", 0) == 0) {
      have_display = true;
      display = a.substr(std::string("--display=").size());
    } else if (a == "-l" || a == "--loop") {
      // The interval is optional, and five seconds when it is left out. A
      // following word that starts with a digit is the interval, and has to
      // be a valid one.
      if (i + 1 < args.size() && !args[i + 1].empty() &&
          std::isdigit(static_cast<unsigned char>(args[i + 1][0]))) {
        if (!interval(args[++i], 1000, "-l")) return 2;
      } else {
        loop_ms = 5000;
      }
    } else if (a.rfind("--loop=", 0) == 0) {
      if (!interval(a.substr(std::string("--loop=").size()), 1000, "--loop")) return 2;
    } else if (a == "-lms" || a == "--loop-ms") {
      if (i + 1 >= args.size()) return fail(a + " needs a number of milliseconds");
      if (!interval(args[++i], 1, a.c_str())) return 2;
    } else if (a.rfind("--loop-ms=", 0) == 0) {
      if (!interval(a.substr(std::string("--loop-ms=").size()), 1, "--loop-ms")) return 2;
    } else if (a == "-h" || a == "--help") {
      help = true;
    } else if (a == "--help-query-gpu") {
      for (const QueryField& f : kGpuFields) std::printf("%s\n", f.name);
      return 0;
    } else if (a == "--help-query-compute-apps") {
      for (const QueryField& f : kAppFields) std::printf("%s\n", f.name);
      return 0;
    } else if (a == "--version") {
      version = true;
    } else if (a == "topo") {
      // Only the matrix. The real subcommand also answers pairwise questions
      // (-p2p, -i) that need link data no profile has. Bare `topo` prints its
      // usage and succeeds, as nvidia-smi's does; it used to be exit 2.
      if (i + 1 >= args.size() || args[i + 1] == "-h" || args[i + 1] == "--help") {
        print_topo_usage(stdout);
        return 0;
      }
      if (args[i + 1] != "-m" && args[i + 1] != "--matrix") {
        std::fprintf(stderr, "vgpu smi: topo supports -m (--matrix)\n\n");
        print_topo_usage(stderr);
        return 2;
      }
      topo = true;
      ++i;
    } else if (a == "-L" || a == "--list-gpus" || a == "--list") {
      // One line per GPU, "GPU <n>: <name> (UUID: <uuid>)". Recognized here,
      // wherever it appears: the session wrapper only translated it as the
      // first argument, so `nvidia-smi -i 1 -L` was an unknown argument.
      list = true;
    } else if (a == "--csv") {
      csv = true;
    } else if (a == "--explain") {
      explain = true;
    } else if (a == "--details") {
      details = true;
    } else if (a == "-p" || a == "--reset-ecc-errors" || a.rfind("--reset-ecc-errors=", 0) == 0) {
      std::string v;
      if (a.rfind("--reset-ecc-errors=", 0) == 0) v = a.substr(a.find('=') + 1);
      else if (i + 1 < args.size()) v = args[++i];
      if (v != "0" && v != "1") return fail(a + " needs 0 (volatile) or 1 (aggregate)");
      reset_ecc = v == "1" ? 1 : 0;
    } else if (a == "--agents") {
      agents = true;
    } else if (a == "--lspci") {
      lspci = true;
    } else if (a == "--lspci-dump") {
      lspci_dump = true;
    } else {
      std::fprintf(stderr, "vgpu smi: unknown argument '%s'\n", a.c_str());
      return 2;
    }
  }
  if (help) {
    print_smi_usage(stdout);
    return 0;
  }
  if (explain) {
    print_explain();
    return 0;
  }
  if (version) {
    print_version();
    return 0;
  }
  // -p resets the ECC counts of the selected GPUs, all of them by default.
  if (reset_ecc >= 0) {
    vgpu::telemetry::Shared snap{};
    if (!read_machine(&snap)) return 1;
    std::vector<uint32_t> sel;
    if (!select_devices(snap, id_spec, &sel)) {
      std::printf("No devices were found\n");
      return 6;
    }
    int done = 0;
    for (uint32_t i : sel) {
      const auto& d = snap.devices[i];
      if (!d.ecc_enabled) {
        std::printf("Resetting ECC errors is not supported for GPU %s.\n", d.bus_id);
        continue;
      }
      if (reset_ecc == 0) vgpu::ras::reset_volatile(d.uuid, /*driver_reload=*/false);
      else vgpu::ras::reset_aggregate(d.uuid);
      std::printf("Reset %s ECC errors to zero for GPU %s.\n", reset_ecc ? "aggregate" : "volatile",
                  d.bus_id);
      ++done;
    }
    std::printf("All done.\n");
    return done ? 0 : 3;
  }

  // The query forms are checked the way nvidia-smi checks them, before anything
  // is printed. `--query-gpu=` with no fields and `--format=csv` with no query
  // used to fall through to the table and exit 0, which a script reads as an
  // answer.
  if (have_query_gpu && have_query_apps)
    return fail("--query-gpu and --query-compute-apps cannot be combined");
  if (have_query_gpu || have_query_apps) {
    const char* flag = have_query_gpu ? "--query-gpu" : "--query-compute-apps";
    const std::vector<std::string> names = split_fields(have_query_gpu ? query_fields : app_fields);
    if (names.empty())
      return fail(std::string(flag) + "= needs at least one field; --help" + (flag + 1) +
                  " lists them");
    for (const std::string& f : names) {
      if (have_query_gpu ? find_field(kGpuFields, f) != nullptr : find_field(kAppFields, f) != nullptr)
        continue;
      std::fprintf(stderr, "Field \"%s\" is not a valid field to query.\n", f.c_str());
      return 2;
    }
    if (!have_format)
      return fail(std::string("--format is required with ") + flag + ", for example --format=csv");
  } else if (have_format) {
    return fail("--format applies to --query-gpu or --query-compute-apps, and neither was given");
  }
  if (have_format) {
    bool is_csv = false;
    for (const std::string& t : split_fields(format)) {
      if (t == "csv") is_csv = true;
      else if (t == "noheader") header = false;
      else if (t == "nounits") units = false;
      else return fail("--format: unknown option '" + t + "' (csv, noheader, nounits)");
    }
    if (!is_csv) return fail("--format must include csv, for example --format=csv,noheader");
  }
  // -x is the verbose report as XML, with or without -q. Alone it used to be
  // ignored, and printed the table.
  if (xml) verbose = true;
  unsigned sections = kSecAll;
  if (have_display) {
    if (!verbose) return fail("-d selects sections of the -q report; add -q");
    if (xml) return fail("-d cannot be combined with -x, as with nvidia-smi");
    if (parse_display(display, &sections) != 0) return 2;
  }
  if (loop_ms > 0 && (topo || list || agents || lspci || lspci_dump))
    return fail("-l/-lms repeats the table, -q, --csv and the query forms; nothing else");
  const bool query_mode = have_query_gpu || have_query_apps;

  // One report, from a fresh snapshot, so a loop shows the machine as it is now.
  auto render = [&](bool first) {
    vgpu::telemetry::Shared snap{};
    if (!read_machine(&snap)) return 1;
    std::vector<uint32_t> sel;
    if (!select_devices(snap, id_spec, &sel)) {
      // What the real tool says, and its exit code for "the object asked for
      // was not found".
      std::printf("No devices were found\n");
      return 6;
    }
    // A GPU that has fallen off the bus (`vgpu fault lose`) has no handle to
    // report through: the tool says so, reports the rest, and exits 15, its
    // code for a GPU that has become inaccessible.
    bool any_lost = false;
    if (!agents && !lspci && !lspci_dump) {
      std::vector<uint32_t> answering;
      for (uint32_t i : sel) {
        bool lost = false;
        try {
          lost = vgpu::ras::is_lost(snap.devices[i].uuid);
        } catch (const std::exception&) {
        }
        if (!lost) {
          answering.push_back(i);
          continue;
        }
        std::fprintf(stderr,
                     "Unable to determine the device handle for GPU%s: GPU is lost.  Reboot the "
                     "system to recover this GPU\n",
                     snap.devices[i].bus_id);
        any_lost = true;
      }
      sel = answering;
      if (sel.empty()) return 15;
    }
    // A looping query prints its header once, so the output is one CSV that
    // can be appended to a file and read back as one.
    const bool with_header = header && first;
    if (topo) {
      print_topology(snap);
    } else if (list) {
      for (uint32_t i : sel)
        std::printf("GPU %u: %s (UUID: %s)\n", i, snap.devices[i].name, snap.devices[i].uuid);
    } else if (have_query_apps) {
      print_query_apps(snap, sel, app_fields, with_header, units);
    } else if (have_query_gpu) {
      print_query_gpu(snap, sel, query_fields, with_header, units);
    } else if (verbose && xml) {
      print_verbose_xml(snap, sel);
    } else if (verbose) {
      print_verbose(snap, sel, sections);
    } else if (csv) {
      print_csv(snap, sel);
    } else if (agents) {
      // Only gfx000 is what ROCm prints on a machine without an AMD GPU, and it
      // stays the output. But outside a session that is almost always a
      // simulator nobody told which GPU to be, so say so where scripts do not
      // read it.
      const char* q = std::getenv("VGPU_QUIET");
      if (print_agents(snap) == 0 && !(q && q[0] == '1')) {
        const char* gpu = std::getenv("VGPU_GPU");
        if (gpu && *gpu)
          std::fprintf(stderr,
                       "rocm_agent_enumerator: VGPU_GPU=%s is not an AMD GPU, so only the CPU "
                       "agent is listed\n",
                       gpu);
        else
          std::fprintf(stderr,
                       "rocm_agent_enumerator: no simulated AMD GPU is configured, so only the "
                       "CPU agent is listed.\n"
                       "  Set VGPU_GPU (for example VGPU_GPU=amd/mi300x) or run inside "
                       "`vgpu shell --gpu amd/mi300x`.\n");
      }
    } else if (lspci || lspci_dump) {
      print_lspci(snap, lspci_dump);
    } else {
      print_gpu_table(snap, sel);
      if (details) print_virtual_details(snap);
    }
    return any_lost ? 15 : 0;
  };
  if (loop_ms == 0) return render(true);

  // -l / -lms: report, sleep, repeat, until SIGINT or SIGTERM -- which end the
  // loop cleanly with status 0, since being stopped is how a loop is meant to
  // finish. The table and -q reports are separated by a blank line; the query
  // forms are not, so their output stays one CSV. No SA_RESTART, so a signal
  // wakes the sleep instead of waiting it out.
  struct sigaction sa {};
  sa.sa_handler = on_loop_signal;
  sigemptyset(&sa.sa_mask);
  ::sigaction(SIGINT, &sa, nullptr);
  ::sigaction(SIGTERM, &sa, nullptr);
  for (bool first = true; !g_loop_stop; first = false) {
    if (!first && !query_mode) std::printf("\n");
    const int rc = render(first);
    std::fflush(stdout);
    if (rc != 0) return rc;
    for (long long slept = 0; slept < loop_ms && !g_loop_stop; slept += 50) {
      const long long step = std::min<long long>(50, loop_ms - slept);
      timespec ts{static_cast<time_t>(step / 1000), static_cast<long>(step % 1000) * 1000000L};
      ::nanosleep(&ts, nullptr);
    }
  }
  return 0;
}
