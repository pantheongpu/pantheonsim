// amdkfd's topology for simulated GPUs (vgpu/amd_kfd.hpp): the files a
// session puts under /sys/class/kfd/kfd/topology. Tools find AMD GPUs by
// reading them -- rocm_agent_enumerator takes each node's gfx_target_version,
// Ollama and RCCL match nodes to PCI devices by location_id -- so each value
// must be the one HIP, rocm-smi and amd-smi report for the same device.
#include <cstdio>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "vgpu/amd_chip.hpp"
#include "vgpu/amd_kfd.hpp"
#include "vgpu/profile.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/telemetry.hpp"
#include "vtest.hpp"

namespace {

using Files = std::map<std::string, std::string>;
using Props = std::map<std::string, unsigned long long>;

Files topology(const vgpu::DeviceProfile& p, int count) {
  Files f;
  for (const auto& e : vgpu::amd::kfd_topology(p, count, 16, 64ull << 30)) f[e.path] = e.contents;
  return f;
}

// "name value" lines, as the kernel writes them; the names in order too.
Props parse(const std::string& text, std::vector<std::string>* order = nullptr) {
  Props out;
  std::istringstream in(text);
  std::string name;
  unsigned long long value;
  while (in >> name >> value) {
    out[name] = value;
    if (order) order->push_back(name);
  }
  return out;
}

// rocm_agent_enumerator's reading of gfx_target_version.
std::string enumerator_name(unsigned long long v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "gfx%llu%llx%llx", v / 10000, (v / 100) % 100, v % 100);
  return buf;
}

}  // namespace

VTEST(gfx_target_version_is_the_kernels_encoding) {
  VCHECK_EQ(vgpu::amd::kfd_gfx_target_version("gfx942"), 90402u);
  VCHECK_EQ(vgpu::amd::kfd_gfx_target_version("gfx90a"), 90010u);
  VCHECK_EQ(vgpu::amd::kfd_gfx_target_version("gfx950"), 90500u);
  VCHECK_EQ(vgpu::amd::kfd_gfx_target_version("gfx1030"), 100300u);
  VCHECK_EQ(vgpu::amd::kfd_gfx_target_version("gfx1100"), 110000u);
  VCHECK_EQ(vgpu::amd::kfd_gfx_target_version("gfx1201"), 120001u);
  VCHECK_EQ(vgpu::amd::kfd_gfx_target_version("sm_90"), 0u);
  VCHECK_EQ(vgpu::amd::kfd_gfx_target_version(""), 0u);
}

VTEST(every_amd_gpu_is_a_node_as_other_tools_describe_it) {
  int amd = 0;
  for (const auto& id : vgpu::available_gpus()) {
    const vgpu::DeviceProfile p = vgpu::load_gpu(id);
    if (p.vendor != "amd") continue;
    ++amd;
    const int count = 3;
    const Files f = topology(p, count);
    const vgpu::amd::Chip c = vgpu::amd::chip(p.architecture.c_str());
    const bool instinct = p.architecture[0] == 'c';

    // The CPU is node 0, with a link to each GPU and no SIMDs.
    const Props cpu = parse(f.at("nodes/0/properties"));
    VCHECK_EQ(cpu.at("cpu_cores_count"), 16ull);
    VCHECK_EQ(cpu.at("simd_count"), 0ull);
    VCHECK_EQ(cpu.at("io_links_count"), 3ull);
    VCHECK_EQ(parse(f.at("nodes/0/mem_banks/0/properties")).at("size_in_bytes"), 64ull << 30);

    std::set<std::string> gpu_ids;
    for (int i = 0; i < count; ++i) {
      const std::string node = "nodes/" + std::to_string(i + 1) + "/";
      vgpu::telemetry::DeviceSample d{};
      vgpu::telemetry::describe_device(p, i, &d);
      const Props g = parse(f.at(node + "properties"));
      // rocm_agent_enumerator lists the profile's own target.
      VCHECK_EQ(enumerator_name(g.at("gfx_target_version")), p.gcn_arch);
      // The chip as rocminfo counts it.
      VCHECK_EQ(g.at("simd_count"), 1ull * p.limits.multiprocessors * c.cus_per_mp * c.simds);
      VCHECK_EQ(g.at("simd_per_cu"), 1ull * c.simds);
      VCHECK_EQ(g.at("array_count"), 1ull * c.engines * c.arrays);
      VCHECK_EQ(g.at("wave_front_size"), 1ull * p.warp_size);
      VCHECK_EQ(g.at("num_xcc"), 1ull * c.xccs);
      // The PCI device HIP reports: vendor, device, bus = ordinal + 1.
      VCHECK_EQ(g.at("vendor_id"), 0x1002ull);
      VCHECK_EQ(g.at("device_id"), 1ull * p.telemetry.pci_device_id);
      VCHECK_EQ(g.at("location_id"), 1ull * (i + 1) << 8);
      VCHECK_EQ(g.at("drm_render_minor"), 128ull + i);
      VCHECK_EQ(g.at("max_engine_clk_fcompute"), 1ull * d.sm_clock_max_mhz);
      // Only Instinct GPUs share an XGMI hive.
      VCHECK_EQ(g.at("hive_id") != 0, instinct);
      // The id rocm-smi shows as GUID and amd-smi as kfd_id; one per device.
      const std::string gpu_id = f.at(node + "gpu_id");
      VCHECK_EQ(std::stoul(gpu_id), 1ul * vgpu::amd::kfd_gpu_id(d.uuid));
      gpu_ids.insert(gpu_id);
      // Its memory is the device's.
      VCHECK_EQ(parse(f.at(node + "mem_banks/0/properties")).at("size_in_bytes"), 1ull * d.vram_total_bytes);

      // Its first link goes to the CPU over PCI Express.
      const Props up = parse(f.at(node + "io_links/0/properties"));
      VCHECK_EQ(up.at("type"), 2ull);
      VCHECK_EQ(up.at("node_from"), 1ull * (i + 1));
      VCHECK_EQ(up.at("node_to"), 0ull);
      // Its peers: over XGMI, one hop, on Instinct; through the host on Radeon.
      const char* kind = instinct ? "io_links/" : "p2p_links/";
      VCHECK_EQ(g.at("io_links_count"), instinct ? 3ull : 1ull);
      VCHECK_EQ(g.at("p2p_links_count"), instinct ? 0ull : 2ull);
      for (int k = 0; k < 2; ++k) {
        const std::string path = node + kind + std::to_string(instinct ? k + 1 : k) + "/properties";
        VCHECK(f.count(path) == 1);
        if (!f.count(path)) continue;
        const Props peer = parse(f.at(path));
        VCHECK_EQ(peer.at("type"), instinct ? 11ull : 2ull);
        VCHECK_EQ(peer.at("weight"), instinct ? 15ull : 40ull);
        VCHECK_EQ(peer.at("node_from"), 1ull * (i + 1));
        VCHECK(peer.at("node_to") != 0 && peer.at("node_to") != 1ull * (i + 1));
      }
    }
    VCHECK_EQ(gpu_ids.size(), size_t{3});
  }
  VCHECK(amd >= 7);
}

// The kernel's property names, in its order: tools read them by name, and
// the thunk expects each one.
VTEST(a_gpu_nodes_properties_are_the_kernels) {
  const Files f = topology(vgpu::load_gpu("amd/mi300x"), 1);
  std::vector<std::string> names;
  parse(f.at("nodes/1/properties"), &names);
  const std::vector<std::string> kernel = {
      "cpu_cores_count", "simd_count", "mem_banks_count", "caches_count", "io_links_count", "p2p_links_count",
      "cpu_core_id_base", "simd_id_base", "max_waves_per_simd", "lds_size_in_kb", "gds_size_in_kb", "num_gws",
      "wave_front_size", "array_count", "simd_arrays_per_engine", "cu_per_simd_array", "simd_per_cu",
      "max_slots_scratch_cu", "gfx_target_version", "vendor_id", "device_id", "location_id", "domain",
      "drm_render_minor", "hive_id", "num_sdma_engines", "num_sdma_xgmi_engines", "num_sdma_queues_per_engine",
      "num_cp_queues", "max_engine_clk_fcompute", "local_mem_size", "fw_version", "capability", "debug_prop",
      "sdma_fw_version", "unique_id", "num_xcc", "max_engine_clk_ccompute"};
  VCHECK(names == kernel);
  VCHECK_EQ(f.at("nodes/1/name"), std::string("ip discovery\n"));
  VCHECK_EQ(f.at("generation_id"), std::string("1\n"));
}

VTEST_MAIN
