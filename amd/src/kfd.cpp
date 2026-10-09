// amdkfd's topology for simulated GPUs (vgpu/amd_kfd.hpp).
//
// Every value is one another interface already reports for the device, so a
// tool that reads this agrees with HIP and HSA (rocminfo), rocm-smi and
// amd-smi: compute units, SIMDs, engines and arrays from the chip
// (vgpu/amd_chip.hpp), the PCI location HIP gives (bus = device + 1), the ids
// and clocks the telemetry publishes, and the links ROCm SMI's library
// reports (XGMI between Instinct GPUs, PCI Express through the host between
// Radeon ones).
#include "vgpu/amd_kfd.hpp"

#include <cctype>
#include <cstdio>
#include <cstring>

#include "vgpu/amd_chip.hpp"
#include "vgpu/telemetry.hpp"

namespace vgpu::amd {
namespace {

// The kernel's io-link types (CRAT, kfd_crat.h) and the flag it sets on a
// link that is up.
constexpr uint32_t kLinkPcie = 2, kLinkXgmi = 11, kLinkEnabled = 1;
// Link weights, as rocm-smi --showtopo prints them: a PCI Express link to
// the host is 20, an XGMI link 15, and a GPU reaches a Radeon peer through
// the host, 20 each way.
constexpr uint32_t kWeightPcie = 20, kWeightXgmi = 15;

uint64_t fnv(const char* s) {
  uint64_t h = 1469598103934665603ull;
  for (; *s; ++s) h = (h ^ static_cast<unsigned char>(*s)) * 1099511628211ull;
  return h;
}

struct Props {
  std::string text;
  Props& add(const char* name, uint64_t value) {
    text += name;
    text += ' ';
    text += std::to_string(value);
    text += '\n';
    return *this;
  }
};

// An io link's properties, in the kernel's order. Bandwidth is MB/s.
std::string link(uint32_t type, uint32_t from, uint32_t to, uint32_t weight, uint32_t mb_per_s) {
  Props p;
  p.add("type", type).add("version_major", 0).add("version_minor", 0).add("node_from", from).add("node_to", to);
  p.add("weight", weight).add("min_latency", 0).add("max_latency", 0);
  p.add("min_bandwidth", mb_per_s).add("max_bandwidth", mb_per_s);
  p.add("recommended_transfer_size", 0).add("recommended_sdma_engine_id_mask", 0).add("flags", kLinkEnabled);
  return p.text;
}

// A PCI Express link's bandwidth, in MB/s: about 1, 2 and 4 GB/s a lane at
// generations 3, 4 and 5.
uint32_t pcie_mb_per_s(uint32_t gen, uint32_t width) {
  const uint32_t lane = gen >= 5 ? 4000 : gen == 4 ? 2000 : gen == 3 ? 1000 : 500;
  return lane * (width ? width : 16);
}

// The name amdgpu gives the chip. Parts it finds by IP discovery, rather
// than from a table of known chips, are all "ip discovery".
const char* asic_name(const std::string& gfx) {
  if (gfx == "gfx90a") return "aldebaran";
  if (gfx == "gfx1030") return "sienna_cichlid";
  if (gfx == "gfx1031") return "navy_flounder";
  return "ip discovery";
}

}  // namespace

uint32_t kfd_gfx_target_version(const std::string& gfx) {
  if (gfx.size() < 6 || gfx.compare(0, 3, "gfx") != 0) return 0;
  const std::string digits = gfx.substr(3);
  const auto hex = [](char c) -> int {
    if (std::isdigit(static_cast<unsigned char>(c))) return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
  };
  const int minor = hex(digits[digits.size() - 2]), stepping = hex(digits[digits.size() - 1]);
  const std::string major = digits.substr(0, digits.size() - 2);
  if (minor < 0 || stepping < 0 || major.empty()) return 0;
  for (char c : major)
    if (!std::isdigit(static_cast<unsigned char>(c))) return 0;
  return static_cast<uint32_t>(std::stoul(major) * 10000 + minor * 100 + stepping);
}

uint32_t kfd_pcie_mb_per_s(uint32_t gen, uint32_t width) { return pcie_mb_per_s(gen, width); }

uint64_t kfd_unique_id(const char* uuid) { return fnv(uuid); }

uint32_t kfd_gpu_id(const char* uuid) {
  if (std::strncmp(uuid, "GPU-", 4) == 0) uuid += 4;
  return static_cast<uint32_t>(fnv(uuid) % 65536);
}

std::vector<KfdFile> kfd_topology(const DeviceProfile& p, int count, uint32_t cpu_cores, uint64_t system_memory) {
  std::vector<KfdFile> out;
  const Chip c = chip(p.architecture.c_str(), p.telemetry.pci_device_id);
  const bool instinct = !p.architecture.empty() && p.architecture[0] == 'c';
  const uint32_t cus = p.limits.multiprocessors * c.cus_per_mp;
  const uint32_t arrays = c.engines * c.arrays;
  const auto node = [](int n) { return "nodes/" + std::to_string(n) + "/"; };

  std::vector<telemetry::DeviceSample> gpus(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) telemetry::describe_device(p, i, &gpus[static_cast<size_t>(i)]);
  const uint32_t link_mb = count ? pcie_mb_per_s(gpus[0].pcie_gen, gpus[0].pcie_width) : 0;
  // One XGMI hive holds every Instinct GPU in the machine.
  const uint64_t hive = instinct && count ? fnv(gpus[0].uuid) : 0;

  out.push_back({"generation_id", "1\n"});
  out.push_back({"system_properties", "platform_oem 0\nplatform_id 0\nplatform_rev 0\n"});

  // Node 0: the host CPU, with system memory and a link to each GPU.
  {
    Props q;
    q.add("cpu_cores_count", cpu_cores).add("simd_count", 0).add("mem_banks_count", 1).add("caches_count", 0);
    q.add("io_links_count", static_cast<uint64_t>(count)).add("p2p_links_count", 0);
    q.add("cpu_core_id_base", 0).add("simd_id_base", 0).add("max_waves_per_simd", 0);
    q.add("lds_size_in_kb", 0).add("gds_size_in_kb", 0).add("num_gws", 0).add("wave_front_size", 0);
    q.add("array_count", 0).add("simd_arrays_per_engine", 0).add("cu_per_simd_array", 0).add("simd_per_cu", 0);
    q.add("max_slots_scratch_cu", 0).add("gfx_target_version", 0).add("vendor_id", 0).add("device_id", 0);
    q.add("location_id", 0).add("domain", 0).add("drm_render_minor", 0).add("hive_id", 0);
    q.add("num_sdma_engines", 0).add("num_sdma_xgmi_engines", 0).add("num_sdma_queues_per_engine", 0);
    q.add("num_cp_queues", 0).add("max_engine_clk_ccompute", 0);
    out.push_back({node(0) + "properties", q.text});
    out.push_back({node(0) + "gpu_id", "0\n"});
    out.push_back({node(0) + "name", "\n"});
    Props m;
    m.add("heap_type", 0).add("size_in_bytes", system_memory).add("flags", 0).add("width", 0).add("mem_clk_max", 0);
    out.push_back({node(0) + "mem_banks/0/properties", m.text});
    for (int i = 0; i < count; ++i)
      out.push_back({node(0) + "io_links/" + std::to_string(i) + "/properties",
                     link(kLinkPcie, 0, static_cast<uint32_t>(i + 1), kWeightPcie, link_mb)});
  }

  // Nodes 1..count: the GPUs.
  for (int i = 0; i < count; ++i) {
    const telemetry::DeviceSample& d = gpus[static_cast<size_t>(i)];
    const uint32_t n = static_cast<uint32_t>(i + 1);
    const uint32_t peers = static_cast<uint32_t>(count - 1);
    Props q;
    q.add("cpu_cores_count", 0).add("simd_count", cus * c.simds).add("mem_banks_count", 1).add("caches_count", 0);
    // An Instinct GPU's XGMI links to its peers are direct io links; a Radeon
    // GPU reaches its peers through the host, which KFD lists as p2p links.
    q.add("io_links_count", instinct ? 1 + peers : 1).add("p2p_links_count", instinct ? 0 : peers);
    q.add("cpu_core_id_base", 0).add("simd_id_base", 0x80001000ull + i * 0x1000ull);
    // HSA reports 32 waves to a compute unit; KFD counts them per SIMD.
    q.add("max_waves_per_simd", 32 / c.simds);
    q.add("lds_size_in_kb", (p.limits.shared_mem_per_sm ? p.limits.shared_mem_per_sm : 65536) / 1024);
    q.add("gds_size_in_kb", 0).add("num_gws", instinct ? 64 : 0).add("wave_front_size", p.warp_size);
    q.add("array_count", arrays).add("simd_arrays_per_engine", c.arrays);
    q.add("cu_per_simd_array", arrays ? cus / arrays : 0).add("simd_per_cu", c.simds);
    q.add("max_slots_scratch_cu", 32).add("gfx_target_version", kfd_gfx_target_version(p.gcn_arch));
    q.add("vendor_id", 0x1002).add("device_id", p.telemetry.pci_device_id);
    q.add("location_id", n << 8).add("domain", 0).add("drm_render_minor", 128 + n - 1).add("hive_id", hive);
    // As HSA reports them (rocminfo): two SDMA engines, none for XGMI alone,
    // and its queue limit.
    q.add("num_sdma_engines", 2).add("num_sdma_xgmi_engines", 0).add("num_sdma_queues_per_engine", 8);
    q.add("num_cp_queues", 128).add("max_engine_clk_fcompute", d.sm_clock_max_mhz).add("local_mem_size", 0);
    q.add("fw_version", 0).add("capability", 0).add("debug_prop", 0).add("sdma_fw_version", 0);
    q.add("unique_id", kfd_unique_id(d.uuid)).add("num_xcc", c.xccs).add("max_engine_clk_ccompute", 0);
    out.push_back({node(i + 1) + "properties", q.text});
    out.push_back({node(i + 1) + "gpu_id", std::to_string(kfd_gpu_id(d.uuid)) + "\n"});
    out.push_back({node(i + 1) + "name", std::string(asic_name(p.gcn_arch)) + "\n"});
    Props m;
    m.add("heap_type", 1).add("size_in_bytes", d.vram_total_bytes).add("flags", 0);
    m.add("width", c.mem_bits).add("mem_clk_max", d.mem_clock_max_mhz);
    out.push_back({node(i + 1) + "mem_banks/0/properties", m.text});
    out.push_back({node(i + 1) + "io_links/0/properties", link(kLinkPcie, n, 0, kWeightPcie, link_mb)});
    uint32_t k = instinct ? 1 : 0;
    for (int j = 0; j < count; ++j) {
      if (j == i) continue;
      const uint32_t to = static_cast<uint32_t>(j + 1);
      if (instinct)
        out.push_back({node(i + 1) + "io_links/" + std::to_string(k++) + "/properties",
                       link(kLinkXgmi, n, to, kWeightXgmi, 64000)});
      else
        out.push_back({node(i + 1) + "p2p_links/" + std::to_string(k++) + "/properties",
                       link(kLinkPcie, n, to, 2 * kWeightPcie, link_mb)});
    }
  }
  return out;
}

}  // namespace vgpu::amd
