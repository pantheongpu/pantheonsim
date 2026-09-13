// `vgpu smi` — nvidia-smi-style view of the running virtual GPUs.
//
// Reads the shared telemetry segment published by a live VirtualGPU process,
// so it works from another terminal while a workload runs. Columns sourced
// from real simulator state (memory, utilization) are exact; the power /
// temperature / clock / voltage columns are a synthetic model driven by real
// utilization and are marked as such by --explain.
#include <sched.h>
#include <unistd.h>

#include <cctype>
#include <cinttypes>
#include <filesystem>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

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
        found = std::to_string(i) == std::to_string(std::stoul(tok));
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
  const char* driver = s.driver_version[0] ? s.driver_version : "580.00.00";
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

    // ECC and MIG are "N/A" because the device profiles carry no ECC or MIG
    // state to report. A number here would be invented, and this table is
    // exactly where an invented number would be believed.
    std::printf("|%4u  %-30.30s%3s  |   %-16.16s %3s |%21s |\n", i, d.name, "On", d.bus_id, "Off",
                "N/A");
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

// nvidia-smi's --query-gpu output: the caller names the fields, and the format
// modifiers decide whether a header and units come with them. Tools parse this,
// so the shape matters as much as the numbers -- the separator is a comma
// followed by a space, and an unsupported field is "[N/A]", both of which real
// nvidia-smi does and both of which parsers depend on.
std::string query_field(const vgpu::telemetry::DeviceSample& d, uint32_t index,
                        const vgpu::telemetry::Shared& s, const std::string& field, bool units) {
  auto with = [&](const std::string& v, const char* unit) {
    return units && unit && *unit ? v + " " + unit : v;
  };
  auto num = [&](long long v, const char* unit) { return with(std::to_string(v), unit); };
  char buf[64];
  if (field == "index") return std::to_string(index);
  if (field == "count") return std::to_string(s.device_count);
  if (field == "name" || field == "gpu_name") return d.name;
  if (field == "uuid" || field == "gpu_uuid") return d.uuid;
  if (field == "serial" || field == "gpu_serial") return "[N/A]";
  if (field == "pci.bus_id" || field == "gpu_bus_id") return d.bus_id;
  if (field == "driver_version")
    // The snapshot's, as the table header prints: an environment lookup here
    // could disagree with the header two lines above it.
    return s.driver_version[0] ? s.driver_version : "580.00.00";
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
  if (field == "clocks.max.memory") return num(d.mem_clock_max_mhz, "MHz");
  if (field == "fan.speed") return num(d.fan_percent, "%");
  // Nothing throttles a simulated clock, so the honest bitmask is empty.
  if (field == "clocks_throttle_reasons.active" || field == "clocks_event_reasons.active")
    return "0x0000000000000000";
  if (field == "pstate") { std::snprintf(buf, sizeof buf, "P%u", d.perf_state); return buf; }
  return "[N/A]";
}

// The unit nvidia-smi puts in a CSV header, "memory.total [MiB]". It is there
// with or without nounits -- nounits strips the values, not the header -- and
// scripts that read the CSV by column name look for exactly that string.
const char* field_unit(const std::string& f) {
  if (f.rfind("memory.", 0) == 0) return "MiB";
  if (f.rfind("utilization.", 0) == 0 || f == "fan.speed") return "%";
  if (f.rfind("power.", 0) == 0 || f == "enforced.power.limit") return "W";
  if (f.rfind("clocks.", 0) == 0) return "MHz";
  return nullptr;
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
  for (uint32_t g : sel)
    for (size_t i = 0; i < names.size(); ++i)
      std::printf("%s%s%s", i ? ", " : "",
                  query_field(s.devices[g], g, s, names[i], units).c_str(),
                  i + 1 == names.size() ? "\n" : "");
}

// --query-compute-apps: one row per process holding a context. With nothing
// running a real driver prints the header and no rows, which is the normal
// answer on an idle machine -- scripts that check for stray processes before a
// job read exactly that.
const char* app_unit(const std::string& f) {
  return f == "used_memory" || f == "used_gpu_memory" ? "MiB" : nullptr;
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
  for (uint32_t g : sel) {
    const auto& d = s.devices[g];
    for (uint32_t j = 0; j < d.proc_count && j < vgpu::telemetry::kMaxProcs; ++j) {
      const auto& pr = d.procs[j];
      for (size_t i = 0; i < names.size(); ++i) {
        const std::string& f = names[i];
        std::string v;
        if (f == "pid") v = std::to_string(pr.pid);
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

// The verbose "-q" report. Tools scrape it for identity and limits, so the
// indentation and the "key : value" alignment are part of the interface.
void print_verbose(const vgpu::telemetry::Shared& s, const std::vector<uint32_t>& sel) {
  // The same versions the table header prints. This used to say CUDA 13.0
  // whatever the session was, beside a header two commands away that said 12.4.
  std::printf("\n==============NVSMI LOG==============\n\n");
  std::printf("%-55s: %s\n", "Driver Version", s.driver_version[0] ? s.driver_version : "580.00.00");
  std::printf("%-55s: %s\n", "CUDA Version", s.cuda_version[0] ? s.cuda_version : "13.0");
  std::printf("\n%-55s: %u\n", "Attached GPUs", s.device_count);
  for (uint32_t i : sel) {
    const auto& d = s.devices[i];
    std::printf("GPU %s\n", d.bus_id);
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
    std::printf("    %-51s: %u %%\n", "Fan Speed", d.fan_percent);
    std::printf("    %-51s: P%u\n", "Performance State", d.perf_state);
    std::printf("    FB Memory Usage\n");
    std::printf("        %-47s: %d MiB\n", "Total", mib(d.vram_total_bytes));
    std::printf("        %-47s: %d MiB\n", "Used", mib(d.vram_used_bytes));
    std::printf("        %-47s: %d MiB\n", "Free",
                mib(d.vram_total_bytes) - mib(d.vram_used_bytes));
    std::printf("    Utilization\n");
    std::printf("        %-47s: %u %%\n", "Gpu", d.utilization_gpu);
    std::printf("        %-47s: %u %%\n", "Memory", d.utilization_mem);
    std::printf("    Temperature\n");
    std::printf("        %-47s: %u C\n", "GPU Current Temp", d.temperature_c);
    std::printf("    Power Readings\n");
    std::printf("        %-47s: %.2f W\n", "Power Draw", d.power_mw / 1000.0);
    std::printf("        %-47s: %.2f W\n", "Current Power Limit", d.power_limit_mw / 1000.0);
    std::printf("    Clocks\n");
    std::printf("        %-47s: %u MHz\n", "SM", d.sm_clock_mhz);
    std::printf("        %-47s: %u MHz\n", "Memory", d.mem_clock_mhz);
    std::printf("\n");
  }
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
              xml_escape(s.driver_version[0] ? s.driver_version : "580.00.00").c_str());
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

void print_csv(const vgpu::telemetry::Shared& s) {
  std::printf("index,name,uuid,bus_id,memory.total,memory.used,utilization.gpu,temperature.gpu,"
              "power.draw,power.limit,clocks.sm,clocks.mem\n");
  for (uint32_t i = 0; i < s.device_count; ++i) {
    const auto& d = s.devices[i];
    std::printf("%u,%s,%s,%s,%d MiB,%d MiB,%u %%,%u,%.2f W,%.2f W,%u MHz,%u MHz\n", i, d.name,
                d.uuid, d.bus_id, mib(d.vram_total_bytes), mib(d.vram_used_bytes),
                d.utilization_gpu, d.temperature_c, d.power_mw / 1000.0, d.power_limit_mw / 1000.0,
                d.sm_clock_mhz, d.mem_clock_mhz);
  }
}

// ROCm's rocm-smi table, for virtual AMD devices (and NVIDIA ones, which it
// shows too so a mixed rack is visible from one tool).
void print_rocm(const vgpu::telemetry::Shared& s) {
  std::printf("\n");
  std::printf("========================= ROCm System Management Interface =========================\n");
  std::printf("=================================== Concise Info ===================================\n");
  std::printf("GPU  Temp   AvgPwr  SCLK     MCLK     Fan   Perf  PwrCap  VRAM%%  GPU%%\n");
  for (uint32_t i = 0; i < s.device_count; ++i) {
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

// rocm_agent_enumerator output: one ISA target per line, CPU agent first.
void print_agents(const vgpu::telemetry::Shared& s) {
  std::printf("gfx000\n");  // the host CPU agent, as ROCm reports
  for (uint32_t i = 0; i < s.device_count; ++i) {
    const auto& d = s.devices[i];
    if (std::strcmp(d.vendor, "amd") != 0) continue;
    // CDNA3 -> gfx942, CDNA4 -> gfx950 (the targets these parts report).
    const char* isa = std::strcmp(d.architecture, "cdna4") == 0 ? "gfx950" : "gfx942";
    std::printf("%s\n", isa);
  }
}

// An lspci view of the virtual devices.
//
// `--lspci` prints the familiar one-line-per-device listing. `--lspci-dump`
// emits a synthesized PCI configuration space in `lspci -x` format, which the
// REAL lspci can render with `lspci -F <file>` -- so the system tool shows the
// virtual GPUs, looking their names up from the host's pci.ids as usual.
void print_lspci(const vgpu::telemetry::Shared& s, bool dump) {
  for (uint32_t i = 0; i < s.device_count; ++i) {
    const auto& d = s.devices[i];
    uint32_t vendor = d.pci_device_id & 0xFFFF;
    uint32_t device = (d.pci_device_id >> 16) & 0xFFFF;
    // NVML-style bus ids carry an 8-digit domain; lspci uses 4.
    const char* bdf = std::strlen(d.bus_id) > 4 ? d.bus_id + 4 : d.bus_id;
    const char* vendor_name = std::strcmp(d.vendor, "amd") == 0
                                  ? "Advanced Micro Devices, Inc. [AMD/ATI]"
                                  : "NVIDIA Corporation";
    if (!dump) {
      std::printf("%s 3D controller: %s %s (rev a1)\n", bdf, vendor_name, d.name);
      continue;
    }
    // Type-0 configuration header. Class 0x030200 = 3D controller, which is
    // what compute GPUs report.
    unsigned char cfg[64];
    std::memset(cfg, 0, sizeof cfg);
    cfg[0x00] = vendor & 0xFF;         cfg[0x01] = (vendor >> 8) & 0xFF;
    cfg[0x02] = device & 0xFF;         cfg[0x03] = (device >> 8) & 0xFF;
    cfg[0x04] = 0x07;                  cfg[0x05] = 0x00;   // command: mem+bus master
    cfg[0x06] = 0x10;                  cfg[0x07] = 0x00;   // status: caps list
    cfg[0x08] = 0xa1;                                      // revision
    cfg[0x09] = 0x00;                  // prog-if
    cfg[0x0a] = 0x02;                  // subclass: 3D controller
    cfg[0x0b] = 0x03;                  // class: display controller
    cfg[0x0e] = 0x00;                  // header type 0
    cfg[0x2c] = vendor & 0xFF;         cfg[0x2d] = (vendor >> 8) & 0xFF;
    cfg[0x2e] = device & 0xFF;         cfg[0x2f] = (device >> 8) & 0xFF;
    cfg[0x34] = 0x60;                  // capabilities pointer

    std::printf("%s 3D controller: %s %s (rev a1)\n", bdf, vendor_name, d.name);
    for (int row = 0; row < 4; ++row) {
      std::printf("%02x:", row * 16);
      for (int col = 0; col < 16; ++col) std::printf(" %02x", cfg[row * 16 + col]);
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

int cmd_smi(const std::vector<std::string>& args) {
  bool csv = false, explain = false, rocm = false, agents = false, lspci = false,
       lspci_dump = false, verbose = false, details = false;
  std::string query_fields, app_fields, id_spec;
  bool header = true, units = true, list = false, xml = false, topo = false;
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i].rfind("--query-gpu=", 0) == 0) {
      query_fields = args[i].substr(std::string("--query-gpu=").size());
    } else if (args[i].rfind("--query-compute-apps=", 0) == 0) {
      app_fields = args[i].substr(std::string("--query-compute-apps=").size());
    } else if (args[i] == "-i" || args[i] == "--id") {
      if (i + 1 >= args.size()) {
        std::fprintf(stderr, "vgpu smi: %s needs a device\n", args[i].c_str());
        return 2;
      }
      id_spec = args[++i];
    } else if (args[i].rfind("--id=", 0) == 0 || args[i].rfind("-i=", 0) == 0) {
      id_spec = args[i].substr(args[i].find('=') + 1);
    } else if (args[i].rfind("--format=", 0) == 0) {
      const std::string fmt = args[i].substr(std::string("--format=").size());
      header = fmt.find("noheader") == std::string::npos;
      units = fmt.find("nounits") == std::string::npos;
    } else if (args[i] == "-q" || args[i] == "--query") {
      verbose = true;
    } else if (args[i] == "-x" || args[i] == "--xml-format") {
      xml = true;   // with -q: the same report as XML
    } else if (args[i] == "topo") {
      // Only the matrix. The real subcommand also answers pairwise questions
      // (-p2p, -i) that need link data no profile has.
      if (i + 1 >= args.size() || (args[i + 1] != "-m" && args[i + 1] != "--matrix")) {
        std::fprintf(stderr, "vgpu smi: topo supports -m (--matrix)\n");
        return 2;
      }
      topo = true;
      ++i;
    } else if (args[i] == "-d" || args[i] == "--display") {
      ++i;   // a section filter for -q; the whole report is printed
    } else if (args[i] == "-L" || args[i] == "--list-gpus" || args[i] == "--list") {
      // One line per GPU, "GPU <n>: <name> (UUID: <uuid>)". Recognized here,
      // wherever it appears: the session wrapper only translated it as the
      // first argument, so `nvidia-smi -i 1 -L` was an unknown argument.
      list = true;
    } else if (args[i] == "--csv")
      csv = true;
    else if (args[i] == "--explain")
      explain = true;
    else if (args[i] == "--details")
      details = true;
    else if (args[i] == "--rocm")
      rocm = true;
    else if (args[i] == "--agents")
      agents = true;
    else if (args[i] == "--lspci")
      lspci = true;
    else if (args[i] == "--lspci-dump")
      lspci_dump = true;
    else {
      std::fprintf(stderr, "vgpu smi: unknown argument '%s'\n", args[i].c_str());
      return 2;
    }
  }
  if (explain) {
    print_explain();
    return 0;
  }
  vgpu::telemetry::Shared snap{};
  if (!vgpu::telemetry::read_snapshot(&snap)) {
    // Nothing is publishing, which on a real machine is the ordinary case:
    // nvidia-smi answers about an idle GPU rather than failing. Monitoring
    // tools poll before and after a workload and treat a non-zero exit as "no
    // GPU", so describe the configured rack as idle instead.
    const char* gpu = std::getenv("VGPU_GPU");
    int count = 1;
    if (const char* c = std::getenv("VGPU_DEVICE_COUNT"); c && c[0]) count = std::atoi(c);
    try {
      vgpu::DeviceProfile p = vgpu::load_gpu(gpu && *gpu ? gpu : "nvidia/h100");
      vgpu::apply_vram_override(p);   // the card the session's programs see
      snap = vgpu::telemetry::idle_snapshot(p, count);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "vgpu smi: no running VirtualGPU and no usable profile (%s)\n",
                   e.what());
      return 1;
    }
  }
  std::vector<uint32_t> sel;
  if (!select_devices(snap, id_spec, &sel)) {
    // What the real tool says, and its exit code for "the object asked for
    // was not found".
    std::printf("No devices were found\n");
    return 6;
  }
  if (topo) {
    print_topology(snap);
  } else if (list) {
    for (uint32_t i : sel)
      std::printf("GPU %u: %s (UUID: %s)\n", i, snap.devices[i].name, snap.devices[i].uuid);
  } else if (!app_fields.empty())
    print_query_apps(snap, sel, app_fields, header, units);
  else if (!query_fields.empty())
    print_query_gpu(snap, sel, query_fields, header, units);
  else if (verbose && xml)
    print_verbose_xml(snap, sel);
  else if (verbose)
    print_verbose(snap, sel);
  else if (csv)
    print_csv(snap);
  else if (rocm)
    print_rocm(snap);
  else if (agents)
    print_agents(snap);
  else if (lspci || lspci_dump)
    print_lspci(snap, lspci_dump);
  else {
    print_gpu_table(snap, sel);
    if (details) print_virtual_details(snap);
  }
  return 0;
}
