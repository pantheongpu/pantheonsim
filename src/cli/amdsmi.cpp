// VirtualGPU's drop-in amd-smi: the commands health tools run against AMD
// GPUs -- list, metric --ecc/--ecc-blocks and ras --cper -- answering from the
// same machine state rocm-smi and nvidia-smi read. The output follows amd-smi's
// documented shapes: JSON is a list with one object per GPU keyed by "gpu", and
// the human-readable form is its upper-cased keys, indented by four.
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <map>
#include <set>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "args.hpp"
#include "machine.hpp"
#include "vgpu/amd_chip.hpp"
#include "vgpu/amd_cper.hpp"
#include "vgpu/ras.hpp"
#include "vgpu/telemetry.hpp"

namespace {

// A value in amd-smi's output: a count, a string, or named children.
struct Node {
  std::string key;
  std::string text;
  bool number = false;
  std::vector<Node> kids;
  bool object = false;
  std::string unit;   // a measurement: "750 W" to read, {"value": 750, "unit": "W"} in JSON
};
Node num(const std::string& key, uint64_t v) { return {key, std::to_string(v), true, {}, false, ""}; }
Node str(const std::string& key, const std::string& v) { return {key, v, false, {}, false, ""}; }
Node obj(const std::string& key, std::vector<Node> kids) { return {key, "", false, std::move(kids), true, ""}; }
Node measured(const std::string& key, uint64_t v, const std::string& unit) {
  return {key, std::to_string(v), true, {}, false, unit};
}

std::string quoted(const std::string& s) {
  std::string out = "\"";
  for (char ch : s) {
    if (ch == '"' || ch == '\\') out += '\\';
    out += ch;
  }
  return out + "\"";
}

void print_json(const Node& n, int depth, bool last) {
  const std::string pad(static_cast<size_t>(depth) * 4, ' ');
  std::printf("%s%s: ", pad.c_str(), quoted(n.key).c_str());
  if (!n.unit.empty()) {
    std::printf("{\n%s    \"value\": %s,\n%s    \"unit\": %s\n%s}%s\n", pad.c_str(), n.text.c_str(), pad.c_str(),
                quoted(n.unit).c_str(), pad.c_str(), last ? "" : ",");
    return;
  }
  if (!n.object) {
    std::printf("%s%s\n", n.number ? n.text.c_str() : quoted(n.text).c_str(), last ? "" : ",");
    return;
  }
  std::printf("{\n");
  for (size_t i = 0; i < n.kids.size(); ++i) print_json(n.kids[i], depth + 1, i + 1 == n.kids.size());
  std::printf("%s}%s\n", pad.c_str(), last ? "" : ",");
}

// One list entry per GPU, each an object whose first key is "gpu".
void print_gpus_json(const std::vector<std::pair<uint32_t, std::vector<Node>>>& gpus) {
  std::printf("[\n");
  for (size_t g = 0; g < gpus.size(); ++g) {
    std::printf("    {\n        \"gpu\": %u%s\n", gpus[g].first, gpus[g].second.empty() ? "" : ",");
    const auto& kids = gpus[g].second;
    for (size_t i = 0; i < kids.size(); ++i) print_json(kids[i], 2, i + 1 == kids.size());
    std::printf("    }%s\n", g + 1 == gpus.size() ? "" : ",");
  }
  std::printf("]\n");
}

std::string upper(std::string s) {
  for (char& ch : s) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  return s;
}

void print_human(const Node& n, int depth) {
  const std::string pad(static_cast<size_t>(depth) * 4, ' ');
  if (!n.object) {
    std::printf("%s%s: %s%s\n", pad.c_str(), upper(n.key).c_str(), n.text.c_str(),
                n.unit.empty() ? "" : (" " + n.unit).c_str());
    return;
  }
  std::printf("%s%s:\n", pad.c_str(), upper(n.key).c_str());
  for (const Node& k : n.kids) print_human(k, depth + 1);
}

void print_gpus_human(const std::vector<std::pair<uint32_t, std::vector<Node>>>& gpus) {
  for (const auto& [index, kids] : gpus) {
    std::printf("GPU: %u\n", index);
    for (const Node& k : kids) print_human(k, 1);
    std::printf("\n");
  }
}

int usage(FILE* to) {
  std::fprintf(to,
               "usage: amd-smi list [-g GPU ...] [--json]\n"
               "       amd-smi static [-a] [-b] [-V] [-l] [-d] [-B] [-r] [-p] [-u] [-v] [-c] [-C] [-g GPU ...] [--json]\n"
               "       amd-smi metric [-u] [-p] [-c] [-t] [-P] [-e] [-k] [-f] [-l] [-x] [-E] [-m] [-g GPU ...] [--json]\n"
               "       amd-smi process [-g GPU ...] [--json]\n"
               "       amd-smi topology [-g GPU ...] [--json]\n"
               "       amd-smi monitor [-w SECONDS] [-i ITERATIONS] [-g GPU ...]\n"
               "       amd-smi firmware | partition | bad-pages | xgmi [-g GPU ...] [--json]\n"
               "       amd-smi ras --cper [--severity SEVERITY ...] [--folder DIR [--file-limit N]]\n"
               "                   [-g GPU ...]\n"
               "       amd-smi version\n"
               "\n"
               "VirtualGPU's drop-in amd-smi, answering from the same machine state as rocm-smi.\n"
               "What a simulated GPU has no value for (VBIOS, serials, firmware) is N/A.\n"
               "ras --cper lists the CPER records amdgpu wrote for each GPU's ECC errors;\n"
               "--folder writes each as a .cper file with its header beside it as .json, as\n"
               "amd-smi dumps them, and --file-limit keeps only the newest N.\n");
  return to == stdout ? 0 : 2;
}

bool is_amd(const vgpu::telemetry::DeviceSample& d) { return std::strcmp(d.vendor, "amd") == 0; }

// "00000000:03:00.0" as amdgpu writes it: a four-digit domain.
std::string bdf(const vgpu::telemetry::DeviceSample& d) {
  const std::string b = d.bus_id;
  return b.size() > 4 && b.compare(0, 4, "0000") == 0 ? b.substr(4) : b;
}

// The RAS blocks amd-smi counts from sysfs, in the order it lists them. Device
// memory is the UMC (memory controller); the on-chip memories are GFX, as in
// rocm-smi's --showrasinfo.
constexpr const char* kBlocks[] = {"UMC", "SDMA", "GFX", "MMHUB", "PCIE_BIF", "HDP", "XGMI_WAFL"};

std::vector<Node> ecc_totals(const vgpu::telemetry::DeviceSample& d, const vgpu::ras::Counters& c) {
  using vgpu::ras::Severity;
  if (!d.ecc_enabled)
    // What amd-smi prints when the driver has no ECC count to give.
    return {str("total_correctable_count", "N/A"), str("total_uncorrectable_count", "N/A"),
            str("cache_correctable_count", "N/A"), str("cache_uncorrectable_count", "N/A")};
  const auto umc = [&](Severity s) {
    return c.ecc[static_cast<uint32_t>(s)][static_cast<uint32_t>(vgpu::ras::Location::DeviceMemory)];
  };
  return {num("total_correctable_count", c.ecc_total(Severity::Corrected)),
          num("total_uncorrectable_count", c.ecc_total(Severity::Uncorrected)),
          num("total_deferred_count", 0),
          num("cache_correctable_count", c.ecc_total(Severity::Corrected) - umc(Severity::Corrected)),
          num("cache_uncorrectable_count", c.ecc_total(Severity::Uncorrected) - umc(Severity::Uncorrected))};
}

Node ecc_blocks(const vgpu::telemetry::DeviceSample& d, const vgpu::ras::Counters& c) {
  using vgpu::ras::Severity;
  if (!d.ecc_enabled) return str("ecc_blocks", "N/A");
  std::vector<Node> blocks;
  for (const char* b : kBlocks) {
    const auto count = [&](Severity s) -> uint64_t {
      const uint64_t memory = c.ecc[static_cast<uint32_t>(s)][static_cast<uint32_t>(vgpu::ras::Location::DeviceMemory)];
      if (std::strcmp(b, "UMC") == 0) return memory;
      if (std::strcmp(b, "GFX") == 0) return c.ecc_total(s) - memory;
      return 0;
    };
    blocks.push_back(obj(b, {num("correctable_count", count(Severity::Corrected)),
                             num("uncorrectable_count", count(Severity::Uncorrected)),
                             num("deferred_count", 0)}));
  }
  return obj("ecc_blocks", std::move(blocks));
}

// CPER records for a GPU's recent ECC errors, oldest first: an uncorrectable
// error in device memory is recorded as non-fatal (the page is retired and the
// device goes on), a corrected one as corrected.
// Writes a file whole; false when it cannot be written.
bool write_file(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(out);
}

std::string cper_time(uint64_t ns) {
  const std::time_t t = static_cast<std::time_t>(ns / 1000000000ull);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y/%m/%d %H:%M:%S", &tm);
  return buf;
}

// ---- static, metric, process: what amd-smi reports of each GPU ----

bool instinct(const vgpu::telemetry::DeviceSample& d) { return d.architecture[0] == 'c'; }
bool partitionable(const vgpu::telemetry::DeviceSample& d) {
  return std::strcmp(d.architecture, "cdna3") == 0 || std::strcmp(d.architecture, "cdna4") == 0;
}
std::string hex(uint32_t v, int digits) {
  char b[16];
  std::snprintf(b, sizeof b, "0x%0*x", digits, v);
  return b;
}
// A file amdgpu keeps for the device, as the session publishes it ("" outside one).
std::string driver_file(const vgpu::telemetry::DeviceSample& d, const char* name) {
  std::string v;
  if (const char* sess = std::getenv("VGPU_SESSION"); sess && *sess) {
    std::ifstream in(std::string(sess) + "/sysfs/" + d.uuid + "/" + name);
    std::getline(in, v);
  }
  return v;
}
uint32_t compute_units(const vgpu::telemetry::DeviceSample& d) {
  return d.multiprocessors * vgpu::amd::chip(d.architecture).cus_per_mp;
}
// amdgpu's version, or the kernel's for an in-tree driver, as amd-smi and
// rocm-smi report it.
std::string driver_version() {
  std::string v;
  if (std::ifstream f("/sys/module/amdgpu/version"); f) std::getline(f, v);
  if (!v.empty()) return v;
  if (FILE* p = ::popen("uname -r 2>/dev/null", "r")) {
    char buf[128] = {};
    if (std::fgets(buf, sizeof buf, p)) v = buf;
    ::pclose(p);
  }
  while (!v.empty() && (v.back() == '\n' || v.back() == ' ')) v.pop_back();
  return v.empty() ? "N/A" : v;
}
uint64_t gt_per_s(uint32_t gen) {
  static const uint64_t kGt[] = {0, 2, 5, 8, 16, 32, 64};
  return gen < 7 ? kGt[gen] : 0;
}

std::vector<Node> static_nodes(const vgpu::telemetry::DeviceSample& d, uint32_t index, const std::set<std::string>& want) {
  const bool all = want.empty();
  const auto on = [&](const char* k) { return all || want.count(k); };
  const vgpu::amd::Chip chip = vgpu::amd::chip(d.architecture);
  const uint32_t device = d.pci_device_id >> 16, vendor = d.pci_device_id & 0xFFFF;
  std::vector<Node> out;
  if (on("asic"))
    out.push_back(obj("asic", {str("market_name", d.name), str("vendor_id", hex(vendor, 4)),
                               str("vendor_name", "Advanced Micro Devices Inc. [AMD/ATI]"),
                               str("subvendor_id", hex(vendor, 4)), str("device_id", hex(device, 4)),
                               str("subsystem_id", hex(device, 4)),
                               str("rev_id", hex(instinct(d) ? 0x00 : device == 0x744c ? 0xc8 : 0xc0, 2)),
                               str("asic_serial", "N/A"), str("oam_id", instinct(d) ? std::to_string(index) : "N/A"),
                               num("num_compute_units", compute_units(d)),
                               str("target_graphics_version", chip.gfx)}));
  if (on("bus"))
    out.push_back(obj("bus", {str("bdf", bdf(d)), num("max_pcie_width", d.pcie_width_max),
                              measured("max_pcie_speed", gt_per_s(d.pcie_gen_max), "GT/s"),
                              str("pcie_interface_version", "Gen " + std::to_string(d.pcie_gen_max)),
                              str("slot_type", instinct(d) ? "OAM" : "PCIE")}));
  if (on("vbios"))
    out.push_back(obj("vbios", {str("name", "N/A"), str("build_date", "N/A"), str("part_number", "N/A"),
                                str("version", "N/A")}));
  if (on("limit"))
    out.push_back(obj("limit", {measured("max_power", d.power_limit_mw / 1000, "W"), measured("min_power", 0, "W"),
                                measured("socket_power", d.power_limit_mw / 1000, "W"),
                                str("slowdown_edge_temperature", "N/A"),
                                measured("slowdown_hotspot_temperature", d.temperature_max_c, "\xC2\xB0""C"),
                                str("slowdown_vram_temperature", "N/A"), str("shutdown_edge_temperature", "N/A"),
                                str("shutdown_hotspot_temperature", "N/A"), str("shutdown_vram_temperature", "N/A")}));
  if (on("driver")) out.push_back(obj("driver", {str("name", "amdgpu"), str("version", driver_version())}));
  if (on("board"))
    out.push_back(obj("board", {str("model_number", "N/A"), str("product_serial", "N/A"), str("fru_id", "N/A"),
                                str("product_name", d.name),
                                str("manufacturer_name", "Advanced Micro Devices, Inc. [AMD/ATI]")}));
  if (on("ras")) {
    std::vector<Node> blocks;
    for (const char* b : kBlocks) blocks.push_back(str(b, d.ecc_enabled ? "ENABLED" : "DISABLED"));
    out.push_back(obj("ras", {str("eeprom_version", "N/A"), str("parity_schema", "DISABLED"),
                              str("single_bit_schema", "DISABLED"), str("double_bit_schema", "DISABLED"),
                              str("poison_schema", d.ecc_enabled ? "ENABLED" : "DISABLED"),
                              obj("ecc_block_state", std::move(blocks))}));
  }
  if (on("partition")) {
    const std::string cp = driver_file(d, "current_compute_partition"), mp = driver_file(d, "current_memory_partition");
    out.push_back(obj("partition", {str("compute_partition", partitionable(d) && !cp.empty() ? cp : "N/A"),
                                    str("memory_partition", partitionable(d) && !mp.empty() ? mp : "N/A"),
                                    num("partition_id", 0)}));
  }
  if (on("soc_pstate")) out.push_back(str("soc_pstate", "N/A"));
  if (on("xgmi_plpd")) out.push_back(str("xgmi_plpd", "N/A"));
  if (on("process_isolation")) out.push_back(str("process_isolation", "Disabled"));
  if (on("numa")) out.push_back(obj("numa", {num("node", 0), num("affinity", 0)}));
  if (on("vram")) {
    std::string type = chip.memory;
    if (std::strstr(d.name, "MI325X")) type = "HBM3E";
    out.push_back(obj("vram", {str("type", type), str("vendor", "N/A"),
                               measured("size", d.vram_total_bytes >> 20, "MB"), num("bit_width", chip.mem_bits)}));
  }
  if (on("cache")) {
    const uint32_t cus = compute_units(d);
    std::vector<Node> caches;
    auto cache = [&](uint32_t kb, uint32_t level, const char* props, uint32_t shared, uint32_t instances) {
      caches.push_back(obj("cache_" + std::to_string(caches.size()),
                           {str("cache_properties", props), measured("cache_size", kb, "KB"), num("cache_level", level),
                            num("max_num_cu_shared", shared), num("num_cache_instance", instances)}));
    };
    // The vector L1 of each compute unit, the L2 of each compute die, and
    // the Infinity Cache the whole chip shares.
    cache(chip.l1_kb, 1, "DATA_CACHE, SIMD_CACHE", 1, cus);
    cache(chip.l2_kb, 2, "DATA_CACHE, SIMD_CACHE", cus / chip.xccs, chip.xccs);
    if (chip.l3_mb) cache(chip.l3_mb * 1024, 3, "DATA_CACHE, SIMD_CACHE", cus, 1);
    out.push_back(obj("cache_info", std::move(caches)));
  }
  if (on("clock")) {
    auto levels = [](uint32_t lo, uint32_t hi) {
      return obj("frequency_levels", {str("level_0", std::to_string(lo) + " MHz"), str("level_1", std::to_string(hi) + " MHz")});
    };
    // Two levels, idle and the most, as the clock model runs them (the
    // same range rocm-smi -s and the concise table show).
    const uint32_t mclk = d.mem_clock_max_mhz;
    // (The lower level is the clock it runs at below the most, as rocm-smi
    // and ROCm SMI's library give it.)
    const uint32_t sclk = d.sm_clock_max_mhz;
    out.push_back(obj("clock", {obj("sys", {num("current_level", d.sm_clock_mhz >= sclk ? 1 : 0),
                                            levels(d.sm_clock_mhz < sclk ? d.sm_clock_mhz : sclk / 6, sclk)}),
                                obj("mem", {num("current_level", d.mem_clock_mhz >= mclk ? 1 : 0),
                                            levels(d.mem_clock_mhz < mclk ? d.mem_clock_mhz : mclk / 5, mclk)})}));
  }
  return out;
}

std::vector<Node> metric_nodes(const vgpu::telemetry::DeviceSample& d, const vgpu::ras::Counters& c,
                               const std::set<std::string>& want) {
  const bool all = want.empty();
  const auto on = [&](const char* k) { return all || want.count(k); };
  const vgpu::amd::Chip chip = vgpu::amd::chip(d.architecture);
  std::vector<Node> out;
  if (on("usage"))
    out.push_back(obj("usage", {measured("gfx_activity", d.utilization_gpu, "%"),
                                measured("umc_activity", d.utilization_mem, "%"), str("mm_activity", "N/A")}));
  if (on("power"))
    out.push_back(obj("power", {measured("socket_power", d.power_mw / 1000, "W"),
                                measured("gfx_voltage", d.voltage_mv, "mV"), str("soc_voltage", "N/A"),
                                str("mem_voltage", "N/A"),
                                str("throttle_status", d.clock_event_reasons ? "THROTTLED" : "UNTHROTTLED"),
                                str("power_management", "ENABLED")}));
  if (on("clock")) {
    std::vector<Node> clocks;
    for (uint32_t x = 0; x < chip.xccs; ++x)
      clocks.push_back(obj("gfx_" + std::to_string(x),
                           {measured("clk", d.sm_clock_mhz, "MHz"), measured("min_clk", d.sm_clock_max_mhz / 6, "MHz"),
                            measured("max_clk", d.sm_clock_max_mhz, "MHz"), str("clk_locked", "DISABLED"),
                            str("deep_sleep", d.utilization_gpu ? "DISABLED" : "ENABLED")}));
    clocks.push_back(obj("mem_0", {measured("clk", d.mem_clock_mhz, "MHz"),
                                   measured("min_clk", d.mem_clock_max_mhz / 5, "MHz"),
                                   measured("max_clk", d.mem_clock_max_mhz, "MHz"), str("clk_locked", "N/A"),
                                   str("deep_sleep", "DISABLED")}));
    out.push_back(obj("clock", std::move(clocks)));
  }
  if (on("temperature")) {
    std::vector<Node> t;
    // Instinct cards have no edge sensor; Radeon cards report one.
    if (instinct(d)) t.push_back(str("edge", "N/A"));
    else t.push_back(measured("edge", d.temperature_c, "\xC2\xB0""C"));
    t.push_back(measured("hotspot", d.temperature_c, "\xC2\xB0""C"));
    if (d.has_memory_temperature) t.push_back(measured("mem", d.temperature_mem_c ? d.temperature_mem_c : d.temperature_c, "\xC2\xB0""C"));
    else t.push_back(str("mem", "N/A"));
    out.push_back(obj("temperature", std::move(t)));
  }
  if (on("pcie"))
    out.push_back(obj("pcie", {num("width", d.pcie_width), measured("speed", gt_per_s(d.pcie_gen), "GT/s"),
                               str("bandwidth", "N/A"), num("replay_count", c.pcie[0]),
                               num("l0_to_recovery_count", 0), num("replay_roll_over_count", 0),
                               num("nak_sent_count", 0), num("nak_received_count", 0),
                               str("current_bandwidth_sent", "N/A"), str("current_bandwidth_received", "N/A"),
                               str("max_packet_size", "N/A")}));
  if (on("ecc")) out.push_back(obj("ecc", ecc_totals(d, c)));
  if (on("ecc_blocks")) out.push_back(ecc_blocks(d, c));
  if (on("fan")) {
    if (instinct(d)) out.push_back(obj("fan", {str("speed", "N/A"), str("max", "N/A"), str("rpm", "N/A"), str("usage", "N/A")}));
    else out.push_back(obj("fan", {num("speed", d.fan_percent * 255 / 100), num("max", 255), str("rpm", "N/A"),
                                   measured("usage", d.fan_percent, "%")}));
  }
  if (on("perf_level")) out.push_back(str("perf_level", "AMDSMI_DEV_PERF_LEVEL_AUTO"));
  if (on("xgmi_err")) out.push_back(str("xgmi_err", instinct(d) ? "0" : "N/A"));
  if (on("energy")) out.push_back(obj("energy", {str("total_energy_consumption", "N/A")}));
  if (on("mem_usage")) {
    const uint64_t total = d.vram_total_bytes >> 20, used = d.vram_used_bytes >> 20;
    out.push_back(obj("mem_usage", {measured("total_vram", total, "MB"), measured("used_vram", used, "MB"),
                                    measured("free_vram", total - used, "MB"), measured("total_visible_vram", total, "MB"),
                                    measured("used_visible_vram", used, "MB"),
                                    measured("free_visible_vram", total - used, "MB"), str("total_gtt", "N/A"),
                                    str("used_gtt", "N/A"), str("free_gtt", "N/A")}));
  }
  return out;
}

std::vector<Node> process_nodes(const vgpu::telemetry::DeviceSample& d) {
  if (d.proc_count == 0) return {str("process_info", "No running processes detected")};
  std::vector<Node> out;
  for (uint32_t k = 0; k < d.proc_count && k < vgpu::telemetry::kMaxProcs; ++k) {
    const auto& p = d.procs[k];
    out.push_back(obj("process_info", {str("name", p.name), num("pid", p.pid),
                                       obj("memory_usage", {measured("gtt_mem", 0, "B"), measured("cpu_mem", 0, "B"),
                                                            measured("vram_mem", p.used_bytes, "B")}),
                                       measured("mem_usage", p.used_bytes, "B"),
                                       obj("usage", {measured("gfx", 0, "ns"), measured("enc", 0, "ns")})}));
  }
  return out;
}

// Between every pair of GPUs: whether one reaches the other, the link's
// weight and hops, its type, and the NUMA bandwidth. Instinct cards are one
// XGMI hop apart; Radeon cards reach each other over PCIe.
void print_topology(const vgpu::telemetry::Shared& snap, const std::vector<uint32_t>& amd,
                    const std::vector<uint32_t>& sel, bool json) {
  const auto dev = [&](uint32_t g) -> const vgpu::telemetry::DeviceSample& { return snap.devices[amd[g]]; };
  const auto xgmi = [&](uint32_t a, uint32_t b) { return instinct(dev(a)) && instinct(dev(b)); };
  if (json) {
    std::printf("[\n");
    for (size_t i = 0; i < sel.size(); ++i) {
      std::printf("    {\n        \"gpu\": %u,\n        \"bdf\": %s,\n        \"links\": [\n", sel[i], quoted(bdf(dev(sel[i]))).c_str());
      for (size_t j = 0; j < sel.size(); ++j) {
        const bool self = i == j, x = xgmi(sel[i], sel[j]);
        std::printf("            {\n                \"gpu\": %u,\n                \"bdf\": %s,\n"
                    "                \"weight\": %d,\n                \"link_status\": \"ENABLED\",\n"
                    "                \"link_type\": %s,\n                \"num_hops\": %d,\n"
                    "                \"bandwidth\": %s\n            }%s\n",
                    sel[j], quoted(bdf(dev(sel[j]))).c_str(), self ? 0 : x ? 15 : 40,
                    self ? "\"SELF\"" : x ? "\"XGMI\"" : "\"PCIE\"", self ? 0 : x ? 1 : 2,
                    self || !x ? "\"N/A\"" : "\"50000-50000\"", j + 1 == sel.size() ? "" : ",");
      }
      std::printf("        ]\n    }%s\n", i + 1 == sel.size() ? "" : ",");
    }
    std::printf("]\n");
    return;
  }
  const auto table = [&](const char* title, auto cell) {
    std::printf("%s:\n%-13s", title, "");
    for (uint32_t j : sel) std::printf("%-13s", bdf(dev(j)).c_str());
    std::printf("\n");
    for (uint32_t i : sel) {
      std::printf("%-13s", bdf(dev(i)).c_str());
      for (uint32_t j : sel) std::printf("%-13s", cell(i, j).c_str());
      std::printf("\n");
    }
    std::printf("\n");
  };
  table("ACCESS TABLE", [&](uint32_t, uint32_t) { return std::string("ENABLED"); });
  table("WEIGHT TABLE", [&](uint32_t i, uint32_t j) { return std::to_string(i == j ? 0 : xgmi(i, j) ? 15 : 40); });
  table("HOPS TABLE", [&](uint32_t i, uint32_t j) { return std::to_string(i == j ? 0 : xgmi(i, j) ? 1 : 2); });
  table("LINK TYPE TABLE", [&](uint32_t i, uint32_t j) { return std::string(i == j ? "SELF" : xgmi(i, j) ? "XGMI" : "PCIE"); });
  table("NUMA BW TABLE", [&](uint32_t i, uint32_t j) { return std::string(i == j || !xgmi(i, j) ? "N/A" : "50000-50000"); });
}

}  // namespace

int cmd_amd_smi(const std::vector<std::string>& args) {
  if (args.empty()) return usage(stderr);
  const std::string& cmd = args[0];
  if (cmd == "-h" || cmd == "--help" || cmd == "help") return usage(stdout);
  static const std::set<std::string> kCommands = {"list", "static", "metric", "process", "topology", "monitor",
                                                  "firmware", "fw", "partition", "bad-pages", "xgmi", "ras", "version"};
  if (!kCommands.count(cmd)) {
    std::fprintf(stderr,
                 "amd-smi: '%s' is not modelled by VirtualGPU (list, static, metric, process, topology, monitor, "
                 "firmware, partition, bad-pages, xgmi, ras, version)\n",
                 cmd.c_str());
    return 2;
  }
  // Each command's options that pick what it reports: amd-smi's short and
  // long names, and the section each selects.
  static const std::map<std::string, std::map<std::string, std::string>> kSections = {
      {"static", {{"-a", "asic"}, {"--asic", "asic"}, {"-b", "bus"}, {"--bus", "bus"}, {"-V", "vbios"},
                  {"--vbios", "vbios"}, {"-l", "limit"}, {"--limit", "limit"}, {"-d", "driver"},
                  {"--driver", "driver"}, {"-B", "board"}, {"--board", "board"}, {"-r", "ras"}, {"--ras", "ras"},
                  {"-p", "partition"}, {"--partition", "partition"}, {"-s", "soc_pstate"},
                  {"--soc-pstate", "soc_pstate"}, {"-x", "xgmi_plpd"}, {"--xgmi-plpd", "xgmi_plpd"},
                  {"-R", "process_isolation"}, {"--process-isolation", "process_isolation"}, {"-u", "numa"},
                  {"--numa", "numa"}, {"-v", "vram"}, {"--vram", "vram"}, {"-c", "cache"}, {"--cache", "cache"},
                  {"-C", "clock"}, {"--clock", "clock"}}},
      {"metric", {{"-u", "usage"}, {"--usage", "usage"}, {"-p", "power"}, {"--power", "power"}, {"-c", "clock"},
                  {"--clock", "clock"}, {"-t", "temperature"}, {"--temperature", "temperature"}, {"-P", "pcie"},
                  {"--pcie", "pcie"}, {"-e", "ecc"}, {"--ecc", "ecc"}, {"-k", "ecc_blocks"},
                  {"--ecc-blocks", "ecc_blocks"}, {"-f", "fan"}, {"--fan", "fan"}, {"-l", "perf_level"},
                  {"--perf-level", "perf_level"}, {"-x", "xgmi_err"}, {"--xgmi-err", "xgmi_err"}, {"-E", "energy"},
                  {"--energy", "energy"}, {"-m", "mem_usage"}, {"--mem-usage", "mem_usage"}}}};
  std::set<std::string> sections;
  long long watch_seconds = 0, iterations = 1;

  bool json = false, csv = false, ecc = false, blocks = false, cper = false;
  std::vector<std::string> gpu_args, severities, folder, file_limit;
  for (size_t i = 1; i < args.size(); ++i) {
    std::string a = args[i];
    std::string inline_value;
    if (const size_t eq = a.find('='); a.rfind("--", 0) == 0 && eq != std::string::npos) {
      inline_value = a.substr(eq + 1);
      a = a.substr(0, eq);
    }
    // Options that take one or more values run to the next option.
    auto values = [&](std::vector<std::string>* into) {
      if (!inline_value.empty()) {
        into->push_back(inline_value);
        return;
      }
      while (i + 1 < args.size() && args[i + 1].rfind("-", 0) != 0) into->push_back(args[++i]);
    };
    if (a == "--json") json = true;
    else if (a == "--csv") csv = true;
    else if (a == "-g" || a == "--gpu") values(&gpu_args);
    else if (kSections.count(cmd) && kSections.at(cmd).count(a)) sections.insert(kSections.at(cmd).at(a));
    else if (cmd == "monitor" && (a == "-w" || a == "--watch" || a == "-i" || a == "--iterations")) {
      std::vector<std::string> v;
      values(&v);
      long long n = 0;
      if (v.size() != 1 || !vgpu::cli::parse_int(v[0], 1, 86400, &n)) {
        std::fprintf(stderr, "amd-smi monitor: %s takes one positive number\n", a.c_str());
        return 2;
      }
      if (a == "-w" || a == "--watch") watch_seconds = n;
      else iterations = n;
    }
    else if (cmd == "ras" && a == "--cper") cper = true;
    else if (cmd == "ras" && a == "--severity") values(&severities);
    else if (cmd == "ras" && a == "--folder") values(&folder);
    else if (cmd == "ras" && a == "--file-limit") values(&file_limit);
    else if (a == "-h" || a == "--help") return usage(stdout);
    else if (cmd == "metric" && a.rfind("-", 0) == 0) {
      // Voltage curves, overdrive and the like: nothing a simulated card has.
      std::fprintf(stderr, "amd-smi metric: %s is not modelled by VirtualGPU\n", a.c_str());
      return 2;
    } else if (cmd == "ras" && (a == "--follow" || a == "--afid" || a == "--cper-file")) {
      std::fprintf(stderr, "amd-smi ras: %s is not modelled by VirtualGPU\n", a.c_str());
      return 2;
    } else {
      std::fprintf(stderr, "amd-smi %s: unrecognized argument '%s'\n", cmd.c_str(), a.c_str());
      return 2;
    }
  }
  if (csv) {
    std::fprintf(stderr, "amd-smi: --csv is not modelled by VirtualGPU; use --json\n");
    return 2;
  }

  (void)ecc;
  (void)blocks;
  if (cmd == "version") {
    const char* rocm = std::getenv("VGPU_ROCM_VERSION");
    std::printf("AMDSMI Tool: VirtualGPU | AMDSMI Library version: VirtualGPU | ROCm version: %s | amdgpu version: %s\n",
                rocm && *rocm ? rocm : "N/A", driver_version().c_str());
    return 0;
  }

  vgpu::telemetry::Shared snap{};
  if (!vgpu::cli::read_machine(&snap)) return 1;
  vgpu::drop_lost_amd(&snap);
  std::vector<uint32_t> amd;
  for (uint32_t i = 0; i < snap.device_count; ++i)
    if (is_amd(snap.devices[i])) amd.push_back(i);
  if (amd.empty()) {
    std::fprintf(stderr, "amd-smi: no AMD GPU in this machine (it has %s)\n",
                 snap.device_count ? snap.devices[0].name : "none");
    return 1;
  }
  std::vector<uint32_t> sel;
  bool all = gpu_args.empty();
  for (const std::string& g : gpu_args) {
    if (g == "all") {
      all = true;
      continue;
    }
    long long n = -1;
    if (!vgpu::cli::parse_int(g, 0, static_cast<long long>(amd.size()) - 1, &n)) {
      std::fprintf(stderr, "amd-smi: GPU '%s' is not one of 0-%zu\n", g.c_str(), amd.size() - 1);
      return 2;
    }
    sel.push_back(static_cast<uint32_t>(n));
  }
  if (all) {
    sel.clear();
    for (uint32_t i = 0; i < amd.size(); ++i) sel.push_back(i);
  }

  if (cmd == "ras") {
    if (!cper) {
      std::fprintf(stderr, "amd-smi ras: name --cper\n");
      return 2;
    }
    bool want_fatal = severities.empty(), want_uncorrected = severities.empty(),
         want_corrected = severities.empty();
    for (const std::string& s : severities) {
      if (s == "all") want_fatal = want_uncorrected = want_corrected = true;
      else if (s == "fatal") want_fatal = true;
      else if (s == "nonfatal" || s == "nonfatal-uncorrected") want_uncorrected = true;
      else if (s == "nonfatal-corrected" || s == "corrected") want_corrected = true;
      else {
        std::fprintf(stderr, "amd-smi ras: unknown --severity '%s'\n", s.c_str());
        return 2;
      }
    }
    (void)want_fatal;   // nothing injected is fatal
    if (folder.size() > 1 || file_limit.size() > 1 || (!file_limit.empty() && folder.empty())) {
      std::fprintf(stderr, "amd-smi ras: --folder takes one directory, and --file-limit one count with it\n");
      return 2;
    }
    long long limit = 0;
    if (!file_limit.empty() && !vgpu::cli::parse_int(file_limit[0], 1, 1 << 30, &limit)) {
      std::fprintf(stderr, "amd-smi ras: --file-limit is a positive count, got '%s'\n", file_limit[0].c_str());
      return 2;
    }
    const std::filesystem::path dir = folder.empty() ? "" : folder[0];
    std::error_code ec;
    if (!folder.empty() && (std::filesystem::create_directories(dir, ec), ec)) {
      std::fprintf(stderr, "amd-smi ras: cannot create %s: %s\n", dir.c_str(), ec.message().c_str());
      return 1;
    }
    // amd-smi prints this table whatever format was asked for; with a folder
    // it names each file it wrote and the AFIDs decoded from it.
    if (folder.empty()) {
      std::printf("WARNING: No CPER files will be dumped unless --folder=<folder_name> is specified "
                  "and cper entries exist.\n");
      std::printf("%-20s %-7s %-20s\n", "timestamp", "gpu_id", "severity");
    } else {
      std::printf("%-20s %-7s %-20s %-17s %s\n", "timestamp", "gpu_id", "severity", "file_name", "list of afids");
    }
    int count = 0;   // numbers the files across GPUs, as amd-smi does
    std::vector<std::string> rows;
    for (uint32_t g : sel) {
      std::vector<vgpu::amd::CperRecord> records;
      try {
        records = vgpu::amd::cper_records(snap.devices[amd[g]], g);
      } catch (const std::exception&) {
      }
      for (const auto& r : records) {
        const bool uncorrected = r.severity == vgpu::amd::CperSeverity::NonFatalUncorrected;
        if (uncorrected ? !want_uncorrected : !want_corrected) continue;
        const char* severity = uncorrected ? "NONFATAL-UNCORRECTED" : "NONFATAL-CORRECTED";
        char row[160];
        if (folder.empty()) {
          std::snprintf(row, sizeof row, "%-20s %-7u %-20s", cper_time(r.time_ns).c_str(), g, severity);
        } else {
          const std::string name = std::string(uncorrected ? "uncorrected" : "corrected") + "-" + std::to_string(++count);
          if (!write_file(dir / (name + ".cper"), r.bytes) ||
              !write_file(dir / (name + ".json"), vgpu::amd::cper_header_json(r, 0) + "\n")) {
            std::fprintf(stderr, "amd-smi ras: cannot write %s in %s\n", name.c_str(), dir.c_str());
            return 1;
          }
          std::snprintf(row, sizeof row, "%-20s %-7u %-20s %-17s %d", cper_time(r.time_ns).c_str(), g, severity,
                        (name + ".cper").c_str(), r.afid);
        }
        rows.push_back(row);
      }
    }
    // Over the limit, the oldest files go, each with its .json, once all are
    // written -- oldest by modification time, as amd-smi sorts them.
    if (limit) {
      std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> files;
      for (const auto& f : std::filesystem::directory_iterator(dir, ec))
        if (f.path().extension() == ".cper") files.emplace_back(f.last_write_time(ec), f.path());
      std::sort(files.begin(), files.end());
      for (size_t k = 0; k + static_cast<size_t>(limit) < files.size(); ++k) {
        std::filesystem::remove(files[k].second, ec);
        std::filesystem::remove(std::filesystem::path(files[k].second).replace_extension(".json"), ec);
      }
    }
    for (const auto& row : rows) std::printf("%s\n", row.c_str());
    return 0;
  }

  if (cmd == "topology") {
    print_topology(snap, amd, sel, json);
    return 0;
  }
  if (cmd == "monitor") {
    // One row per GPU, every -w seconds, -i times.
    for (long long it = 0; it < iterations; ++it) {
      if (it) {
        std::this_thread::sleep_for(std::chrono::seconds(watch_seconds ? watch_seconds : 1));
        if (!vgpu::cli::read_machine(&snap)) return 1;
        vgpu::drop_lost_amd(&snap);
      }
      std::printf("GPU  POWER  GPU_TEMP  MEM_TEMP  GFX_UTIL  GFX_CLOCK  MEM_UTIL  MEM_CLOCK  VRAM_USED  VRAM_TOTAL\n");
      for (uint32_t g : sel) {
        const auto& d = snap.devices[amd[g]];
        const std::string mem_t = d.has_memory_temperature
                                      ? std::to_string(d.temperature_mem_c ? d.temperature_mem_c : d.temperature_c) + " \xC2\xB0""C"
                                      : "N/A";
        std::printf("%3u %4u W  %5u \xC2\xB0""C  %8s  %6u %%  %5u MHz  %6u %%  %5u MHz  %6llu MB  %7llu MB\n", g,
                    d.power_mw / 1000, d.temperature_c, mem_t.c_str(), d.utilization_gpu, d.sm_clock_mhz,
                    d.utilization_mem, d.mem_clock_mhz,
                    static_cast<unsigned long long>(d.vram_used_bytes >> 20),
                    static_cast<unsigned long long>(d.vram_total_bytes >> 20));
      }
      std::fflush(stdout);
    }
    return 0;
  }
  if (cmd == "partition") {
    std::printf("CURRENT_PARTITION:\n%-8s%-8s%-18s%-27s%s\n", "GPU_ID", "MEMORY", "ACCELERATOR_TYPE",
                "ACCELERATOR_PROFILE_INDEX", "PARTITION_ID");
    for (uint32_t g : sel) {
      const auto& d = snap.devices[amd[g]];
      const std::string cp = driver_file(d, "current_compute_partition"), mp = driver_file(d, "current_memory_partition");
      std::printf("%-8u%-8s%-18s%-27s%u\n", g, partitionable(d) && !mp.empty() ? mp.c_str() : "N/A",
                  partitionable(d) && !cp.empty() ? cp.c_str() : "N/A", partitionable(d) ? "0" : "N/A", 0u);
    }
    return 0;
  }
  if (cmd == "xgmi") {
    // The XGMI links between Instinct GPUs, and the traffic each carried
    // (none is counted: 0).
    std::vector<uint32_t> linked;
    for (uint32_t g : sel)
      if (instinct(snap.devices[amd[g]])) linked.push_back(g);
    if (linked.empty()) {
      std::printf("No XGMI links: these GPUs reach each other over PCIe\n");
      return 0;
    }
    std::printf("LINK METRIC TABLE:\n%-7s%-14s%-10s%-15s%-11s", "", "bdf", "bit_rate", "max_bandwidth", "link_type");
    for (uint32_t g : linked) std::printf("%-14s", bdf(snap.devices[amd[g]]).c_str());
    std::printf("\n");
    for (uint32_t g : linked) {
      std::printf("GPU%-4u%-14s%-10s%-15s%-11s", g, bdf(snap.devices[amd[g]]).c_str(), "32 Gb/s", "512 Gb/s", "XGMI");
      for (uint32_t h : linked) std::printf("%-14s", g == h ? "N/A" : "0 KB");
      std::printf("\n");
    }
    return 0;
  }

  std::vector<std::pair<uint32_t, std::vector<Node>>> gpus;
  for (uint32_t g : sel) {
    const auto& d = snap.devices[amd[g]];
    std::vector<Node> kids;
    if (cmd == "static") {
      kids = static_nodes(d, g, sections);
    } else if (cmd == "process") {
      kids = process_nodes(d);
    } else if (cmd == "firmware" || cmd == "fw") {
      // No firmware is loaded into a simulated GPU.
      kids = {str("fw_list", "N/A")};
    } else if (cmd == "bad-pages") {
      vgpu::ras::Counters c{};
      try {
        c = vgpu::ras::read(d.uuid).since_load;
      } catch (const std::exception&) {
      }
      const uint64_t retired = c.retired_sbe + c.retired_dbe;
      kids = {retired ? num("retired", retired) : str("retired", "No bad pages found."),
              c.retired_pending ? num("pending", c.retired_pending) : str("pending", "No bad pages found."),
              str("un_res", "No bad pages found.")};
    } else if (cmd == "list") {
      std::string uuid = d.uuid;
      if (uuid.rfind("GPU-", 0) == 0) uuid = uuid.substr(4);
      // KFD's id for the GPU is a hash; this one is stable per device. Node 0
      // is the CPU, so the GPUs start at 1.
      uint64_t kfd = 1469598103934665603ull;
      for (char ch : uuid) kfd = (kfd ^ static_cast<unsigned char>(ch)) * 1099511628211ull;
      kids = {str("bdf", bdf(d)), str("uuid", uuid), num("kfd_id", kfd % 65536), num("node_id", g + 1),
              num("partition_id", 0)};
    } else {
      vgpu::ras::Counters c{};
      try {
        c = vgpu::ras::read(d.uuid).since_load;
      } catch (const std::exception&) {
      }
      kids = metric_nodes(d, c, sections);   // metric alone: every one
    }
    gpus.emplace_back(g, std::move(kids));
  }
  if (json) print_gpus_json(gpus);
  else print_gpus_human(gpus);
  return 0;
}
