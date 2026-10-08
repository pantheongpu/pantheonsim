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
#include <cstring>
#include <unistd.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "vtest.hpp"

extern "C" {
int rsmi_init(uint64_t);
int rsmi_dev_firmware_version_get(uint32_t, int, uint64_t*);
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
int rsmi_dev_metrics_header_info_get(uint32_t, void*);
int rsmi_dev_metrics_xcd_counter_get(uint32_t, uint16_t*);
int rsmi_dev_drm_render_minor_get(uint32_t, uint32_t*);
int rsmi_dev_xgmi_hive_id_get(uint32_t, uint64_t*);
int rsmi_dev_pcie_slot_type_get(uint32_t, int*);
int rsmi_dev_sku_get(uint32_t, uint16_t*);
int rsmi_dev_xgmi_physical_id_get(uint32_t, uint16_t*);
int rsmi_driver_status(void*);
int amdsmi_get_gpu_metrics_info(void*, void*);
struct Frequencies {
  bool has_deep_sleep;
  uint32_t num_supported, current;
  uint64_t frequency[33];
};
int rsmi_dev_gpu_clk_freq_get(uint32_t, int, Frequencies*);
int rsmi_dev_energy_count_get(uint32_t, uint64_t*, float*, uint64_t*);
int amdsmi_init(uint64_t);
int amdsmi_get_socket_handles(uint32_t*, void**);
int amdsmi_get_processor_handles(void*, uint32_t*, void**);
int amdsmi_get_energy_count(void*, uint64_t*, float*, uint64_t*);
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

// The energy accumulator: Instinct only, in ticks of 15.3 uJ, one accumulator
// behind this library and AMD SMI's, so the second reading of either follows
// the first of the other.
VTEST(energy_is_instinct_only_and_one_counter_serves_both_libraries) {
  rsmi_init(0);
  amdsmi_init(2);
  uint32_t ns = 8;
  void* sockets[8];
  VCHECK_EQ(amdsmi_get_socket_handles(&ns, sockets), kSuccess);
  uint64_t e0 = 0, ts0 = 0, e1 = 0, ts1 = 0, e2 = 0, ts2 = 0;
  float res = 0;
  if (radeon()) {
    VCHECK_EQ(rsmi_dev_energy_count_get(0, &e0, &res, &ts0), kNotSupported);
    return;
  }
  VCHECK_EQ(rsmi_dev_energy_count_get(0, &e0, &res, &ts0), kSuccess);
  VCHECK(res > 15.0f && res < 15.5f);
  VCHECK(e0 > 0 && ts0 > 0);
  uint32_t np = 1;
  void* gpu = nullptr;
  VCHECK_EQ(amdsmi_get_processor_handles(sockets[0], &np, &gpu), kSuccess);
  usleep(20000);
  VCHECK_EQ(amdsmi_get_energy_count(gpu, &e1, &res, &ts1), kSuccess);
  VCHECK(e1 > e0 && ts1 > ts0);
  usleep(20000);
  VCHECK_EQ(rsmi_dev_energy_count_get(0, &e2, &res, &ts2), kSuccess);
  VCHECK(e2 > e1 && ts2 > ts1);
  // 20 ms of at most a few kW is a few tens of joules: the ticks that follow
  // are a few million, not a counter restarted from its offset.
  VCHECK(e2 - e0 < 100 * 1000000 / 15);
  VCHECK_EQ(rsmi_dev_energy_count_get(0, nullptr, &res, &ts0), 1);   // invalid arguments
  VCHECK_EQ(rsmi_dev_energy_count_get(99, &e0, &res, &ts0), kOutOfBounds);
}

VTEST(what_is_not_modelled_is_refused_as_a_card_refuses_it) {
  char buf[64];
  VCHECK_EQ(rsmi_dev_vbios_version_get(0, buf, sizeof buf), kNotSupported);
  VCHECK_EQ(rsmi_dev_perf_level_set(0, 2), kNotSupported);
  VCHECK_EQ(rsmi_dev_perf_level_set(5, 2), kOutOfBounds);
}

// rsmi_gpu_metrics_t (4544 bytes in ROCm 7.2.0), read at the offsets compiled
// from that release's rocm_smi.h: the same table, copied across in the same
// way, as amdsmi_gpu_metrics_t.
VTEST(gpu_metrics_are_the_v1_5_table_in_rocm_smis_structure) {
  VCHECK_EQ(rsmi_init(0), kSuccess);
  std::vector<unsigned char> m(4544, 0);
  const auto u16 = [&](size_t o) { uint16_t v; std::memcpy(&v, &m[o], 2); return v; };
  const auto u32 = [&](size_t o) { uint32_t v; std::memcpy(&v, &m[o], 4); return v; };
  const auto u64 = [&](size_t o) { uint64_t v; std::memcpy(&v, &m[o], 8); return v; };
  for (uint32_t d = 0; d < 2; ++d) {
    VCHECK_EQ(rsmi_dev_gpu_metrics_info_get(d, m.data()), kSuccess);
    VCHECK_EQ(u16(0), 360u);                      // the table's size
    VCHECK_EQ(m[2], 1);                           // format revision
    VCHECK_EQ(m[3], 5);                           // content revision
    VCHECK_EQ(u16(4), 0xFFFFu);                   // temperature_edge: not in the table
    VCHECK(u16(6) > 0 && u16(6) < 150);           // temperature_hotspot
    VCHECK(u16(120) > 0);                         // current_socket_power
    VCHECK_EQ(u16(22), 0xFFFFu);                  // average_socket_power: not in v1.5
    VCHECK_EQ(u16(54), u16(312));                 // current_gfxclk is current_gfxclks[0]
    VCHECK(u16(312) > 0 && u16(58) > 0);          // the graphics and memory clocks
    VCHECK_EQ(u32(132), 0u);                      // gfxclk_lock_status
    VCHECK(u64(32) > 0);                          // system_clock_counter
    VCHECK_EQ(u64(4520), ~uint64_t{0});           // vram_max_bandwidth: v1.7, not in the table
    VCHECK_EQ(u16(4528), 0xFFFFu);                // xgmi_link_status[0]
    VCHECK_EQ(u64(184), 0u);                      // xgmi_read_data_acc[0]
    // The accumulator: Instinct only, the one rsmi_dev_energy_count_get counts.
    if (radeon()) {
      VCHECK_EQ(u64(24), ~uint64_t{0});
    } else {
      uint64_t e = 0, ts = 0;
      float res = 0;
      VCHECK(u64(24) > 0 && u64(24) != ~uint64_t{0});
      VCHECK_EQ(rsmi_dev_energy_count_get(d, &e, &res, &ts), kSuccess);
      VCHECK(e >= u64(24) && e - u64(24) < 1000000);
    }
    unsigned char header[4] = {};
    VCHECK_EQ(rsmi_dev_metrics_header_info_get(d, header), kSuccess);
    VCHECK(header[0] == 104 && header[1] == 1 && header[2] == 1 && header[3] == 5);   // 360 = 0x168
  }
  VCHECK_EQ(rsmi_dev_gpu_metrics_info_get(0, nullptr), 1);          // invalid arguments
  VCHECK_EQ(rsmi_dev_gpu_metrics_info_get(9, m.data()), kOutOfBounds);
  VCHECK_EQ(rsmi_dev_metrics_header_info_get(0, nullptr), 1);
  VCHECK_EQ(rsmi_dev_metrics_header_info_get(9, m.data()), kOutOfBounds);
  // The XCD counter has no documented source here: refused.
  uint16_t xcd = 0;
  VCHECK_EQ(rsmi_dev_metrics_xcd_counter_get(0, &xcd), kNotSupported);
}

// Every field both libraries copy is the same in the two structures: the
// libraries answer one question.
VTEST(rocm_smi_and_amd_smi_agree_on_the_metrics_table) {
  VCHECK_EQ(amdsmi_init(2), kSuccess);
  uint32_t sockets = 0;
  VCHECK_EQ(amdsmi_get_socket_handles(&sockets, nullptr), kSuccess);
  std::vector<void*> sock(sockets);
  VCHECK_EQ(amdsmi_get_socket_handles(&sockets, sock.data()), kSuccess);
  uint32_t n = 1;
  void* gpu = nullptr;
  VCHECK_EQ(amdsmi_get_processor_handles(sock[0], &n, &gpu), kSuccess);
  std::vector<unsigned char> a(4544, 0), r(4544, 0);
  VCHECK_EQ(amdsmi_get_gpu_metrics_info(gpu, a.data()), kSuccess);
  VCHECK_EQ(rsmi_dev_gpu_metrics_info_get(0, r.data()), kSuccess);
  // Byte for byte but for what moves between two calls: the clocks of the
  // system and firmware, and the energy counter (offsets 24, 32 and 96).
  for (size_t i = 0; i < a.size(); ++i) {
    if ((i >= 24 && i < 40) || (i >= 96 && i < 104)) continue;
    if (a[i] != r[i]) {
      std::fprintf(stderr, "metrics differ at byte %zu: %u and %u\n", i, a[i], r[i]);
      VCHECK(false);
      break;
    }
  }
}

VTEST(the_identity_queries_the_header_has_answer_from_the_machine) {
  for (uint32_t d = 0; d < 2; ++d) {
    uint32_t minor = 0;
    VCHECK_EQ(rsmi_dev_drm_render_minor_get(d, &minor), kSuccess);
    VCHECK_EQ(minor, 128u + d);
    int slot = -1;
    VCHECK_EQ(rsmi_dev_pcie_slot_type_get(d, &slot), kSuccess);
    VCHECK_EQ(slot, radeon() ? 0 : 2);          // RSMI_PCIE_SLOT_PCIE, _OAM
  }
  uint32_t minor = 0;
  VCHECK_EQ(rsmi_dev_drm_render_minor_get(2, &minor), kOutOfBounds);
  VCHECK_EQ(rsmi_dev_drm_render_minor_get(0, nullptr), 1);
  uint64_t hive0 = 0, hive1 = 1;
  if (radeon()) {
    VCHECK_EQ(rsmi_dev_xgmi_hive_id_get(0, &hive0), kNotSupported);
  } else {
    VCHECK_EQ(rsmi_dev_xgmi_hive_id_get(0, &hive0), kSuccess);
    VCHECK_EQ(rsmi_dev_xgmi_hive_id_get(1, &hive1), kSuccess);
    VCHECK(hive0 != 0 && hive0 == hive1);       // one hive, KFD's
  }
}

VTEST(every_function_of_the_header_is_there_and_refused_where_there_is_no_answer) {
  uint16_t v = 0;
  VCHECK_EQ(rsmi_dev_sku_get(0, &v), kNotSupported);
  VCHECK_EQ(rsmi_dev_xgmi_physical_id_get(0, &v), kNotSupported);
  VCHECK_EQ(rsmi_dev_sku_get(7, &v), kOutOfBounds);
  VCHECK_EQ(rsmi_driver_status(nullptr), kNotSupported);
}

// RCCL 2.27 reads the compute microengine's firmware (block 5) to decide whether it needs
// HSA_NO_SCRATCH_RECLAIM, and does not start on a card that gives none.
VTEST(the_compute_microengine_firmware_is_what_rccl_accepts) {
  uint64_t fw = 0;
  VCHECK_EQ(rsmi_init(0), 0);
  const char* gpu = std::getenv("VGPU_GPU");
  const std::string g = gpu ? gpu : "amd/mi300x";
  const bool mi300 = g == "amd/mi300x" || g == "amd/mi325x", mi350 = g == "amd/mi350x";
  VCHECK_EQ(rsmi_dev_firmware_version_get(0, 5, &fw), mi300 || mi350 ? 0 : 2);   // 2: not supported
  if (mi300) VCHECK_EQ(fw, uint64_t{177});
  if (mi350) VCHECK_EQ(fw, uint64_t{24});
  VCHECK_EQ(rsmi_dev_firmware_version_get(0, 1, &fw), 2);   // the other blocks are not modelled
}

VTEST_MAIN
