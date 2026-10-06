// VirtualGPU's librocm_smi64, as RCCL uses it: how many GPUs, where each is
// on the PCI bus, and what links each pair. The answers have to agree with
// what the HIP runtime says -- RCCL matches the two by PCI address -- and
// name the link the machine has: Infinity Fabric (XGMI) between Instinct
// GPUs, PCI Express between Radeon ones. And as ROCm's rocm-smi uses it:
// each GPU's identity, sensors, clocks, memory, ECC and partitions, from the
// machine `vgpu smi` reads, with what is not modelled refused as a card
// without it refuses it.
#include <sys/stat.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "vtest.hpp"

extern "C" {
int rsmi_init(uint64_t);
int rsmi_num_monitor_devices(uint32_t*);
int rsmi_dev_pci_id_get(uint32_t, uint64_t*);
int rsmi_topo_get_link_type(uint32_t, uint32_t, uint64_t*, int*);
int rsmi_topo_get_link_weight(uint32_t, uint32_t, uint64_t*);
int rsmi_minmax_bandwidth_get(uint32_t, uint32_t, uint64_t*, uint64_t*);
int hipDeviceGetPCIBusId(char*, int, int);
int rsmi_dev_id_get(uint32_t, uint16_t*);
int rsmi_dev_vendor_id_get(uint32_t, uint16_t*);
int rsmi_dev_revision_get(uint32_t, uint16_t*);
int rsmi_dev_market_name_get(uint32_t, char*, uint32_t);
int rsmi_dev_target_graphics_version_get(uint32_t, uint64_t*);
int rsmi_dev_temp_metric_get(uint32_t, uint32_t, int, int64_t*);
int rsmi_dev_power_get(uint32_t, uint64_t*, int*);
int rsmi_dev_power_cap_get(uint32_t, uint32_t, uint64_t*);
int rsmi_dev_memory_total_get(uint32_t, int, uint64_t*);
int rsmi_dev_memory_usage_get(uint32_t, int, uint64_t*);
int rsmi_dev_busy_percent_get(uint32_t, uint32_t*);
int rsmi_dev_ecc_enabled_get(uint32_t, uint64_t*);
int rsmi_dev_compute_partition_get(uint32_t, char*, uint32_t);
int rsmi_dev_memory_partition_get(uint32_t, char*, uint32_t);
int rsmi_dev_fan_speed_get(uint32_t, uint32_t, int64_t*);
int rsmi_dev_vbios_version_get(uint32_t, char*, uint32_t);
int rsmi_dev_memory_reserved_pages_get(uint32_t, uint32_t*, void*);
int rsmi_dev_perf_level_set(uint32_t, int);
int rsmi_dev_gpu_metrics_info_get(uint32_t, void*);
struct Frequencies {
  bool has_deep_sleep;
  uint32_t num_supported, current;
  uint64_t frequency[33];
};
int rsmi_dev_gpu_clk_freq_get(uint32_t, int, Frequencies*);
}

namespace {
constexpr int kSuccess = 0, kNotSupported = 2, kOutOfBounds = 7;
constexpr int kPcie = 1, kXgmi = 2;

bool radeon() {
  const char* gpu = std::getenv("VGPU_GPU");
  return gpu && std::string(gpu).rfind("amd/rx", 0) == 0;
}

// The address RCCL reads from HIP, in ROCm SMI's BDFID form.
uint64_t hip_bdfid(int device) {
  char id[32] = {};
  if (hipDeviceGetPCIBusId(id, sizeof id, device) != 0) return ~0ull;
  unsigned domain = 0, bus = 0, dev = 0, fn = 0;
  std::sscanf(id, "%x:%x:%x.%x", &domain, &bus, &dev, &fn);
  return uint64_t{domain} << 32 | uint64_t{bus} << 8 | uint64_t{dev} << 3 | fn;
}
}  // namespace

VTEST(it_counts_the_devices_hip_counts) {
  VCHECK_EQ(rsmi_init(0), kSuccess);
  uint32_t n = 0;
  VCHECK_EQ(rsmi_num_monitor_devices(&n), kSuccess);
  VCHECK_EQ(n, 2u);
}

VTEST(each_device_is_where_hip_says) {
  for (uint32_t d = 0; d < 2; ++d) {
    uint64_t bdfid = 0;
    VCHECK_EQ(rsmi_dev_pci_id_get(d, &bdfid), kSuccess);
    VCHECK_EQ(bdfid, hip_bdfid(static_cast<int>(d)));
  }
  uint64_t bdfid = 0;
  VCHECK_EQ(rsmi_dev_pci_id_get(2, &bdfid), kOutOfBounds);
}

VTEST(instinct_gpus_are_linked_by_xgmi) {
  if (radeon()) return;
  uint64_t hops = 9, weight = 0, lo = 0, hi = 0;
  int type = 0;
  VCHECK_EQ(rsmi_topo_get_link_type(0, 1, &hops, &type), kSuccess);
  VCHECK_EQ(hops, 1u);
  VCHECK_EQ(type, kXgmi);
  VCHECK_EQ(rsmi_topo_get_link_weight(0, 1, &weight), kSuccess);
  VCHECK_EQ(weight, 15u);
  VCHECK_EQ(rsmi_minmax_bandwidth_get(0, 1, &lo, &hi), kSuccess);
  VCHECK(hi > 0 && lo <= hi);
  VCHECK_EQ(rsmi_topo_get_link_type(0, 2, &hops, &type), kOutOfBounds);
}

VTEST(rccl_is_told_to_ask_this_library_where_there_is_no_kfd) {
  // Where there is no AMD kernel driver, the library, once loaded, has asked
  // RCCL to use it; where there is one, RCCL's own choice stands.
  struct stat st{};
  if (stat("/sys/class/kfd/kfd/topology/nodes", &st) == 0) return;
  const char* v = std::getenv("RCCL_USE_ROCM_SMI_LIB");
  VCHECK(v != nullptr);
  VCHECK_EQ(std::string(v), std::string("1"));
}

VTEST(radeon_gpus_meet_over_pci_express) {
  // Run again, by ctest, with VGPU_GPU=amd/rx7900xtx; on an Instinct part the
  // link is XGMI (above), and this one has nothing to say.
  if (!radeon()) return;
  uint64_t hops = 0, lo = 0, hi = 0;
  int type = 0;
  VCHECK_EQ(rsmi_topo_get_link_type(0, 1, &hops, &type), kSuccess);
  VCHECK_EQ(type, kPcie);
  VCHECK_EQ(rsmi_minmax_bandwidth_get(0, 1, &lo, &hi), kNotSupported);
}

VTEST(each_gpu_says_what_it_is) {
  const bool rx = radeon();
  for (uint32_t d = 0; d < 2; ++d) {
    uint16_t id = 0, vendor = 0, rev = 0;
    VCHECK_EQ(rsmi_dev_id_get(d, &id), kSuccess);
    VCHECK_EQ(id, rx ? 0x744c : 0x74a1);
    VCHECK_EQ(rsmi_dev_vendor_id_get(d, &vendor), kSuccess);
    VCHECK_EQ(vendor, 0x1002);
    VCHECK_EQ(rsmi_dev_revision_get(d, &rev), kSuccess);
    VCHECK_EQ(rev, rx ? 0xc8 : 0x00);
    char name[64] = {};
    VCHECK_EQ(rsmi_dev_market_name_get(d, name, sizeof name), kSuccess);
    VCHECK_EQ(std::string(name), std::string(rx ? "AMD Radeon RX 7900 XTX" : "AMD Instinct MI300X"));
    // The target, its digits as hex, which rocm-smi prints back as gfx942.
    uint64_t gfx = 0;
    VCHECK_EQ(rsmi_dev_target_graphics_version_get(d, &gfx), kSuccess);
    VCHECK_EQ(gfx, rx ? 0x1100u : 0x942u);
  }
  uint16_t id = 0;
  VCHECK_EQ(rsmi_dev_id_get(2, &id), kOutOfBounds);
  char small[4] = {};
  VCHECK_EQ(rsmi_dev_market_name_get(0, small, sizeof small), 11);   // INSUFFICIENT_SIZE
}

VTEST(sensors_clocks_and_memory_come_from_the_machine) {
  const bool rx = radeon();
  int64_t t = 0;
  VCHECK_EQ(rsmi_dev_temp_metric_get(0, 1, 0, &t), kSuccess);   // junction, current
  VCHECK(t >= 20000 && t <= 110000);
  // An Instinct card has no edge sensor; a Radeon does.
  VCHECK_EQ(rsmi_dev_temp_metric_get(0, 0, 0, &t), rx ? kSuccess : kNotSupported);
  uint64_t power = 0, cap = 0;
  int type = -1;
  VCHECK_EQ(rsmi_dev_power_get(0, &power, &type), kSuccess);
  VCHECK_EQ(rsmi_dev_power_cap_get(0, 0, &cap), kSuccess);
  VCHECK(power > 0 && power <= cap && cap == (rx ? 355000000u : 750000000u));
  uint64_t total = 0, used = 0;
  VCHECK_EQ(rsmi_dev_memory_total_get(0, 0, &total), kSuccess);
  VCHECK_EQ(rsmi_dev_memory_usage_get(0, 0, &used), kSuccess);
  VCHECK(total > (uint64_t{16} << 30) && used <= total);
  VCHECK_EQ(rsmi_dev_memory_total_get(0, 2, &total), kNotSupported);   // GTT
  uint32_t busy = 101;
  VCHECK_EQ(rsmi_dev_busy_percent_get(0, &busy), kSuccess);
  VCHECK(busy <= 100);
  // Two levels, the current one's clock the one the machine runs at.
  Frequencies f{};
  VCHECK_EQ(rsmi_dev_gpu_clk_freq_get(0, 0, &f), kSuccess);
  VCHECK(f.num_supported == 2 && f.current < 2 && f.frequency[0] < f.frequency[1]);
  VCHECK_EQ(rsmi_dev_gpu_clk_freq_get(0, 1, &f), kNotSupported);   // the data fabric's
  int64_t fan = 0;
  VCHECK_EQ(rsmi_dev_fan_speed_get(0, 0, &fan), rx ? kSuccess : kNotSupported);   // Instinct: no fan
}

VTEST(ecc_and_partitions_as_the_card_has_them) {
  const bool rx = radeon();
  uint64_t mask = 0;
  VCHECK_EQ(rsmi_dev_ecc_enabled_get(0, &mask), kSuccess);
  VCHECK(rx ? mask == 0 : (mask & 1) != 0);   // the UMC block's bit
  char part[16] = {};
  // Retired pages are recorded only where there is ECC (the Radeon profiles have none).
  uint32_t pages = 0;
  VCHECK_EQ(rsmi_dev_memory_reserved_pages_get(0, &pages, nullptr), rx ? kNotSupported : kSuccess);
  VCHECK_EQ(rsmi_dev_compute_partition_get(0, part, sizeof part), rx ? kNotSupported : kSuccess);
  if (!rx) VCHECK_EQ(std::string(part), std::string("SPX"));
  VCHECK_EQ(rsmi_dev_memory_partition_get(0, part, sizeof part), rx ? kNotSupported : kSuccess);
  if (!rx) VCHECK_EQ(std::string(part), std::string("NPS1"));
}

VTEST(what_is_not_modelled_is_refused_as_a_card_refuses_it) {
  char buf[64];
  VCHECK_EQ(rsmi_dev_vbios_version_get(0, buf, sizeof buf), kNotSupported);
  VCHECK_EQ(rsmi_dev_gpu_metrics_info_get(0, buf), kNotSupported);
  VCHECK_EQ(rsmi_dev_perf_level_set(0, 2), kNotSupported);
  VCHECK_EQ(rsmi_dev_perf_level_set(5, 2), kOutOfBounds);
}

VTEST_MAIN
