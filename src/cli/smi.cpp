// `vgpu smi` — nvidia-smi-style view of the running virtual GPUs.
//
// Reads the shared telemetry segment published by a live VirtualGPU process,
// so it works from another terminal while a workload runs. Columns sourced
// from real simulator state (memory, utilization) are exact; the power /
// temperature / clock / voltage columns are a synthetic model driven by real
// utilization and are marked as such by --explain.
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/telemetry.hpp"

namespace {

const char* perf_state_name(uint32_t p) {
  static char buf[8];
  std::snprintf(buf, sizeof buf, "P%u", p);
  return buf;
}

int mib(uint64_t bytes) { return static_cast<int>(bytes / (1024 * 1024)); }

void print_table(const vgpu::telemetry::Shared& s) {
  std::printf(
      "+-----------------------------------------------------------------------------------------+\n");
  std::printf("| VGPU-SMI 0.1.0                  Driver Version: %-13s CUDA Version: %-7s |\n",
              s.driver_version, s.cuda_version);
  std::printf(
      "|-----------------------------------------+------------------------+----------------------|\n");
  std::printf(
      "| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n");
  std::printf(
      "| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n");
  std::printf(
      "|                                         |                        |               MIG M. |\n");
  std::printf(
      "|=========================================+========================+======================|\n");
  for (uint32_t i = 0; i < s.device_count; ++i) {
    const auto& d = s.devices[i];
    char power[32];
    std::snprintf(power, sizeof power, "%uW / %uW", d.power_mw / 1000, d.power_limit_mw / 1000);
    char memory[32];
    std::snprintf(memory, sizeof memory, "%dMiB / %dMiB", mib(d.vram_used_bytes),
                  mib(d.vram_total_bytes));
    std::printf("| %3u  %-28.28s On  | %-16.16s Off |                  N/A |\n", i, d.name,
                d.bus_id);
    std::printf("| %2u%%  %3uC    %-3s   %15.15s | %22.22s |    %3u%%      Default |\n",
                d.fan_percent, d.temperature_c, perf_state_name(d.perf_state), power, memory,
                d.utilization_gpu);
    std::printf(
        "|                                         |                        |                  N/A |\n");
    std::printf(
        "+-----------------------------------------+------------------------+----------------------+\n");
  }
  std::printf("\n");
  std::printf(
      "+-----------------------------------------------------------------------------------------+\n");
  std::printf("| Virtual device details                                                                  |\n");
  std::printf(
      "|=========================================================================================|\n");
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
    std::printf(
        "+-----------------------------------------------------------------------------------------+\n");
  }
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
       lspci_dump = false;
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--csv")
      csv = true;
    else if (args[i] == "--explain")
      explain = true;
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
    if (agents) {
      // With nothing running there are no agents beyond the host CPU one.
      std::printf("gfx000\n");
      return 0;
    }
    std::fprintf(stderr,
                 "vgpu smi: no running VirtualGPU found.\n"
                 "  Telemetry is published by a live process; start a workload first, e.g.\n"
                 "    LD_LIBRARY_PATH=build/shim ./my_cuda_app\n"
                 "  (looked in %s; override with VGPU_TELEMETRY_PATH)\n",
                 vgpu::telemetry::default_path().c_str());
    return 1;
  }
  if (csv)
    print_csv(snap);
  else if (rocm)
    print_rocm(snap);
  else if (agents)
    print_agents(snap);
  else if (lspci || lspci_dump)
    print_lspci(snap, lspci_dump);
  else
    print_table(snap);
  return 0;
}
