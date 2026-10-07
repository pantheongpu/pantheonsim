// VirtualGPU's libamd_smi, as AMD's Python package (amdsmi) and vLLM use it:
// the GPUs, one to a socket; each one's target, IDs, compute units, UUID, PCI
// address, KFD node and memory; the links between them. Each answer agrees
// with what the rest of the machine says of the same GPU: KFD's topology,
// ROCm SMI and amd-smi. And the package binds every function of the library
// when it is imported, so every one must be exported, the ones not modelled
// answering NOT_SUPPORTED.
#include <dlfcn.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "vgpu/amd_kfd.hpp"
#include "vgpu/ras.hpp"
#include "vgpu/telemetry.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

extern "C" {
int amdsmi_init(uint64_t);
int amdsmi_shut_down();
int amdsmi_get_socket_handles(uint32_t*, void**);
int amdsmi_get_socket_info(void*, size_t, char*);
int amdsmi_get_processor_handles(void*, uint32_t*, void**);
int amdsmi_get_processor_type(void*, int*);
int amdsmi_get_processor_handle_from_bdf(uint64_t, void**);
int amdsmi_get_gpu_device_bdf(void*, uint64_t*);
int amdsmi_get_gpu_device_uuid(void*, unsigned*, char*);
int amdsmi_get_gpu_kfd_info(void*, void*);
int amdsmi_get_gpu_memory_total(void*, int, uint64_t*);
int amdsmi_topo_get_link_type(void*, void*, uint64_t*, int*);
int amdsmi_topo_get_numa_node_number(void*, uint32_t*);
int amdsmi_get_gpu_vbios_info(void*, void*);
int amdsmi_set_gpu_perf_level(void*, int);
int amdsmi_get_gpu_total_ecc_count(void*, void*);
int amdsmi_get_gpu_ecc_enabled(void*, uint64_t*);
int amdsmi_get_gpu_ecc_count(void*, uint64_t, void*);
int amdsmi_get_gpu_ecc_status(void*, uint64_t, int*);
int amdsmi_get_gpu_bad_page_info(void*, uint32_t*, void*);
int amdsmi_get_xgmi_info(void*, void*);
int amdsmi_gpu_xgmi_error_status(void*, int*);
int amdsmi_get_gpu_xgmi_link_status(void*, void*);
int amdsmi_get_pcie_info(void*, void*);
int amdsmi_get_gpu_pci_replay_counter(void*, uint64_t*);
int amdsmi_get_gpu_pci_bandwidth(void*, void*);
int amdsmi_get_temp_metric(void*, int, int, int64_t*);
int amdsmi_get_power_info(void*, void*);
int amdsmi_get_power_cap_info(void*, uint32_t, void*);
int amdsmi_get_gpu_fan_speed(void*, uint32_t, int64_t*);
int amdsmi_get_clock_info(void*, int, void*);
int amdsmi_get_gpu_volt_metric(void*, int, int, int64_t*);
int amdsmi_get_gpu_compute_partition(void*, char*, uint32_t);
int amdsmi_get_gpu_memory_partition(void*, char*, uint32_t);
int amdsmi_get_gpu_process_list(void*, uint32_t*, void*);
int amdsmi_get_gpu_metrics_info(void*, void*);
int amdsmi_get_violation_status(void*, void*);
int amdsmi_get_clk_freq(void*, int, void*);
int amdsmi_set_clk_freq(void*, int, uint64_t);
int amdsmi_get_energy_count(void*, uint64_t*, float*, uint64_t*);
}

namespace {
constexpr int kSuccess = 0, kNotSupported = 2, kNotFound = 31, kNotInit = 32;
constexpr int kPcie = 1, kXgmi = 2;

// amdsmi_asic_info_t and amdsmi_kfd_info_t, as the package lays them out.
struct AsicInfo {
  char market_name[256];
  uint32_t vendor_id;
  char vendor_name[256];
  uint32_t subvendor_id;
  uint64_t device_id;
  uint32_t rev_id;
  char asic_serial[256];
  uint32_t oam_id, num_of_compute_units;
  uint64_t target_graphics_version;
  uint32_t subsystem_id, reserved[21];
};
struct KfdInfo {
  uint64_t kfd_id;
  uint32_t node_id, current_partition_id, reserved[12];
};
extern "C" int amdsmi_get_gpu_asic_info(void*, AsicInfo*);

// amdsmi_frequencies_t, which RVS reads, as the header lays it out (the power,
// ECC and clock structures are below).
struct Frequencies {
  bool has_deep_sleep;
  uint32_t num_supported, current;
  uint64_t frequency[33];
};
constexpr int kOutOfBounds = 17;
constexpr int kClkSys = 0, kClkMem = 4;

bool radeon() {
  const char* gpu = std::getenv("VGPU_GPU");
  return gpu && std::string(gpu).rfind("amd/rx", 0) == 0;
}

std::vector<void*> gpus() {
  uint32_t n = 0;
  amdsmi_get_socket_handles(&n, nullptr);
  std::vector<void*> sockets(n);
  amdsmi_get_socket_handles(&n, sockets.data());
  std::vector<void*> out;
  for (void* s : sockets) {
    uint32_t k = 0;
    amdsmi_get_processor_handles(s, &k, nullptr);
    std::vector<void*> h(k);
    amdsmi_get_processor_handles(s, &k, h.data());
    out.insert(out.end(), h.begin(), h.begin() + k);
  }
  return out;
}
}  // namespace

VTEST(nothing_answers_before_init) {
  uint32_t n = 0;
  VCHECK_EQ(amdsmi_get_socket_handles(&n, nullptr), kNotInit);
}

VTEST(each_gpu_is_a_socket_with_one_processor) {
  VCHECK_EQ(amdsmi_init(2), kSuccess);   // AMDSMI_INIT_AMD_GPUS
  uint32_t n = 0;
  VCHECK_EQ(amdsmi_get_socket_handles(&n, nullptr), kSuccess);
  VCHECK_EQ(n, 2u);
  const std::vector<void*> g = gpus();
  VCHECK_EQ(g.size(), size_t{2});
  for (void* h : g) {
    int type = 0;
    VCHECK_EQ(amdsmi_get_processor_type(h, &type), kSuccess);
    VCHECK_EQ(type, 1);   // AMDSMI_PROCESSOR_TYPE_AMD_GPU
  }
  // A socket is named for its GPU's PCI address.
  void* sockets[2] = {};
  n = 2;
  amdsmi_get_socket_handles(&n, sockets);
  char name[64] = {};
  VCHECK_EQ(amdsmi_get_socket_info(sockets[1], sizeof name, name), kSuccess);
  VCHECK_EQ(std::string(name), std::string("0000:02:00"));
  int type = 0;
  VCHECK_EQ(amdsmi_get_processor_type(reinterpret_cast<void*>(0x1234), &type), kNotFound);
}

VTEST(each_gpu_says_what_it_is_as_the_machine_does) {
  amdsmi_init(2);
  const bool rx = radeon();
  const vgpu::DeviceProfile p = vgpu::load_gpu(rx ? "amd/rx7900xtx" : "amd/mi300x");
  const std::vector<void*> g = gpus();
  for (size_t i = 0; i < g.size(); ++i) {
    vgpu::telemetry::DeviceSample d{};
    vgpu::telemetry::describe_device(p, static_cast<int>(i), &d);
    AsicInfo a{};
    VCHECK_EQ(amdsmi_get_gpu_asic_info(g[i], &a), kSuccess);
    VCHECK_EQ(std::string(a.market_name), std::string(rx ? "AMD Radeon RX 7900 XTX" : "AMD Instinct MI300X"));
    VCHECK_EQ(a.vendor_id, 0x1002u);
    VCHECK_EQ(a.device_id, uint64_t{rx ? 0x744cu : 0x74a1u});
    VCHECK_EQ(a.rev_id, rx ? 0xc8u : 0x00u);
    // The package prints "gfx" + hex(value): the target's digits as hex.
    VCHECK_EQ(a.target_graphics_version, uint64_t{rx ? 0x1100u : 0x942u});
    VCHECK_EQ(a.num_of_compute_units, rx ? 96u : 304u);
    // An OAM module id only on Instinct boards; the package's N/A otherwise.
    VCHECK_EQ(a.oam_id, rx ? 0xFFFFFFFFu : static_cast<uint32_t>(i));
    // The UUID amd-smi list prints, and the KFD node and id of the topology.
    char uuid[38] = {};
    unsigned len = sizeof uuid;
    VCHECK_EQ(amdsmi_get_gpu_device_uuid(g[i], &len, uuid), kSuccess);
    VCHECK_EQ(std::string("GPU-") + uuid, std::string(d.uuid));
    KfdInfo k{};
    VCHECK_EQ(amdsmi_get_gpu_kfd_info(g[i], &k), kSuccess);
    VCHECK_EQ(k.node_id, static_cast<uint32_t>(i + 1));
    VCHECK_EQ(k.kfd_id, uint64_t{vgpu::amd::kfd_gpu_id(d.uuid)});
    // Its memory, and where it is: bus i + 1, found again by its address.
    uint64_t total = 0, bdf = 0;
    VCHECK_EQ(amdsmi_get_gpu_memory_total(g[i], 0, &total), kSuccess);
    VCHECK_EQ(total, d.vram_total_bytes);
    VCHECK_EQ(amdsmi_get_gpu_device_bdf(g[i], &bdf), kSuccess);
    VCHECK_EQ((bdf >> 8) & 0xff, uint64_t{i + 1});
    void* found = nullptr;
    VCHECK_EQ(amdsmi_get_processor_handle_from_bdf(bdf, &found), kSuccess);
    VCHECK(found == g[i]);
    uint32_t numa = 9;
    VCHECK_EQ(amdsmi_topo_get_numa_node_number(g[i], &numa), kSuccess);
    VCHECK_EQ(numa, 0u);
  }
}

// vLLM's test for a fully connected machine: one XGMI hop between every pair.
VTEST(gpus_are_linked_as_rocm_smi_says) {
  amdsmi_init(2);
  const std::vector<void*> g = gpus();
  uint64_t hops = 9;
  int type = -1;
  VCHECK_EQ(amdsmi_topo_get_link_type(g[0], g[1], &hops, &type), kSuccess);
  VCHECK_EQ(hops, radeon() ? 2u : 1u);
  VCHECK_EQ(type, radeon() ? kPcie : kXgmi);
  VCHECK_EQ(amdsmi_topo_get_link_type(g[0], g[0], &hops, &type), kSuccess);
  VCHECK_EQ(hops, 0u);
}

namespace {
// The layouts of amdsmi.h's health structures (the library checks their sizes).
struct ErrorCount {
  uint64_t correctable, uncorrectable, deferred, reserved[5];
};
struct RetiredPage {
  uint64_t address, size;
  int status;
};
struct XgmiInfo {
  uint8_t lanes;
  uint64_t hive_id, node_id;
  uint32_t index, reserved[9];
};
struct XgmiLinks {
  uint32_t total;
  int status[8];
  uint64_t reserved[7];
};
struct PcieInfo {
  struct {
    uint16_t max_width;
    uint32_t max_speed, version;
    int slot_type;
    uint32_t max_version;
    uint64_t reserved[9];
  } stat;
  struct {
    uint16_t width;
    uint32_t speed, bandwidth;
    uint64_t replay, l0_to_recovery, roll_over, nak_sent, nak_received;
    uint32_t other_end;
    uint64_t reserved[12];
  } metric;
  uint64_t reserved[32];
};
struct PowerInfo {
  uint64_t socket;
  uint32_t current, average;
  uint64_t gfx_mv, soc_mv, mem_mv;
  uint32_t limit;
  uint64_t reserved[18];
};
struct PowerCap {
  uint64_t cap, def, dpm, min, max, reserved[3];
};
struct ClkInfo {
  uint32_t clk, min_clk, max_clk;
  uint8_t locked, deep_sleep;
  uint32_t reserved[4];
};
constexpr int kInval = 1, kInsufficientSize = 41;
constexpr uint64_t kBlockUmc = 1, kBlockGfx = 4, kBlockXgmiWafl = 0x80;

std::string uuid_of(size_t i) {
  vgpu::telemetry::DeviceSample d{};
  vgpu::telemetry::describe_device(vgpu::load_gpu(radeon() ? "amd/rx7900xtx" : "amd/mi300x"), static_cast<int>(i), &d);
  return d.uuid;
}

// A machine of its own for the tests that inject errors, so none sees another's
// (or an earlier run's: the volatile counts live in the telemetry directory).
struct TempState {
  std::string root = "/tmp/vgpu-amdsmi-test-" + std::to_string(getpid());
  void clean() {
    for (size_t i = 0; i < 2; ++i) {
      vgpu::ras::reset_volatile(uuid_of(i));
      vgpu::ras::reset_aggregate(uuid_of(i));
    }
  }
  TempState() {
    std::filesystem::remove_all(root);
    setenv("VGPU_STATE_DIR", root.c_str(), 1);
    clean();
  }
  ~TempState() {
    clean();
    unsetenv("VGPU_STATE_DIR");
    std::filesystem::remove_all(root);
  }
};
}  // namespace

// Instinct profiles have ECC (HBM); Radeon profiles have none, and no RAS
// files: the counts are refused and the blocks read disabled.
VTEST(ecc_counts_are_the_machines_where_the_profile_has_ecc) {
  amdsmi_init(2);
  TempState state;
  const std::vector<void*> g = gpus();
  const bool rx = radeon();
  uint64_t mask = 0;
  VCHECK_EQ(amdsmi_get_gpu_ecc_enabled(g[0], &mask), kSuccess);
  VCHECK_EQ(mask != 0, !rx);
  int ras = 0;
  VCHECK_EQ(amdsmi_get_gpu_ecc_status(g[0], kBlockUmc, &ras), kSuccess);
  VCHECK_EQ(ras, rx ? 1 : 6);   // AMDSMI_RAS_ERR_STATE_DISABLED / _ENABLED
  VCHECK_EQ(amdsmi_get_gpu_ecc_status(g[0], kBlockUmc | kBlockGfx, &ras), kInval);   // one block at a time
  ErrorCount total{};
  if (rx) {
    VCHECK_EQ(amdsmi_get_gpu_total_ecc_count(g[0], &total), kNotSupported);
    VCHECK_EQ(amdsmi_get_gpu_ecc_count(g[0], kBlockUmc, &total), kNotSupported);
    return;
  }
  VCHECK_EQ(amdsmi_get_gpu_total_ecc_count(g[0], &total), kSuccess);
  VCHECK_EQ(total.correctable + total.uncorrectable + total.deferred, uint64_t{0});
  // Errors injected into device memory are UMC's; into a cache, GFX's.
  using vgpu::ras::Location;
  using vgpu::ras::Severity;
  const std::string u = uuid_of(0);
  vgpu::ras::inject_ecc(u, Severity::Corrected, Location::DeviceMemory, 3, vgpu::ras::Retirement::Rows);
  vgpu::ras::inject_ecc(u, Severity::Corrected, Location::L2Cache, 2, vgpu::ras::Retirement::Rows);
  vgpu::ras::inject_ecc(u, Severity::Uncorrected, Location::DeviceMemory, 1, vgpu::ras::Retirement::Rows);
  ErrorCount umc{}, gfx{}, xgmi{};
  VCHECK_EQ(amdsmi_get_gpu_total_ecc_count(g[0], &total), kSuccess);
  VCHECK_EQ(total.correctable, uint64_t{5});
  VCHECK_EQ(total.uncorrectable, uint64_t{1});
  VCHECK_EQ(amdsmi_get_gpu_ecc_count(g[0], kBlockUmc, &umc), kSuccess);
  VCHECK_EQ(umc.correctable, uint64_t{3});
  VCHECK_EQ(umc.uncorrectable, uint64_t{1});
  VCHECK_EQ(amdsmi_get_gpu_ecc_count(g[0], kBlockGfx, &gfx), kSuccess);
  VCHECK_EQ(gfx.correctable, uint64_t{2});
  VCHECK_EQ(gfx.uncorrectable, uint64_t{0});
  VCHECK_EQ(amdsmi_get_gpu_ecc_count(g[0], kBlockXgmiWafl, &xgmi), kSuccess);   // enabled, nothing modelled
  VCHECK_EQ(xgmi.correctable, uint64_t{0});
  VCHECK_EQ(amdsmi_get_gpu_ecc_count(g[0], uint64_t{1} << 17, &xgmi), kNotSupported);   // IH: not a block here
  VCHECK_EQ(amdsmi_get_gpu_ecc_count(g[0], kBlockUmc | kBlockGfx, &xgmi), kInval);
  // The other GPU is untouched.
  VCHECK_EQ(amdsmi_get_gpu_total_ecc_count(g[1], &total), kSuccess);
  VCHECK_EQ(total.correctable, uint64_t{0});
}

// An uncorrectable error takes a row out of service; each retired page is
// listed, and a short array gets as many as fit and INSUFFICIENT_SIZE.
VTEST(bad_pages_are_listed_where_the_card_has_ecc) {
  amdsmi_init(2);
  TempState state;
  const std::vector<void*> g = gpus();
  uint32_t n = 0;
  if (radeon()) {
    VCHECK_EQ(amdsmi_get_gpu_bad_page_info(g[0], &n, nullptr), kNotSupported);
    return;
  }
  VCHECK_EQ(amdsmi_get_gpu_bad_page_info(g[0], &n, nullptr), kSuccess);
  VCHECK_EQ(n, 0u);
  using vgpu::ras::Location;
  using vgpu::ras::Severity;
  vgpu::ras::inject_ecc(uuid_of(0), Severity::Uncorrected, Location::DeviceMemory, 2, vgpu::ras::Retirement::Rows);
  VCHECK_EQ(amdsmi_get_gpu_bad_page_info(g[0], &n, nullptr), kSuccess);
  VCHECK(n >= 2);
  const uint32_t all = n;
  std::vector<RetiredPage> pages(all);
  VCHECK_EQ(amdsmi_get_gpu_bad_page_info(g[0], &n, pages.data()), kSuccess);
  VCHECK_EQ(n, all);
  for (const RetiredPage& p : pages) VCHECK(p.size == 4096 && p.status >= 0 && p.status <= 1);
  VCHECK_EQ(pages.front().status, 0);   // reserved
  VCHECK_EQ(pages.back().status, 1);    // the newest is pending until a driver reload
  uint32_t one = 1;
  VCHECK_EQ(amdsmi_get_gpu_bad_page_info(g[0], &one, pages.data()), kInsufficientSize);
  VCHECK_EQ(one, 1u);
}

// XGMI: Instinct GPUs share one hive (KFD's id), have a link to each peer, and
// no errors; Radeon GPUs have no XGMI at all.
VTEST(xgmi_is_there_only_between_instinct_gpus) {
  amdsmi_init(2);
  const std::vector<void*> g = gpus();
  XgmiInfo a{}, b{};
  XgmiLinks links{};
  int status = -1;
  if (radeon()) {
    VCHECK_EQ(amdsmi_get_xgmi_info(g[0], &a), kNotSupported);
    VCHECK_EQ(amdsmi_gpu_xgmi_error_status(g[0], &status), kNotSupported);
    VCHECK_EQ(amdsmi_get_gpu_xgmi_link_status(g[0], &links), kNotSupported);
    return;
  }
  VCHECK_EQ(amdsmi_get_xgmi_info(g[0], &a), kSuccess);
  VCHECK_EQ(amdsmi_get_xgmi_info(g[1], &b), kSuccess);
  VCHECK(a.hive_id != 0 && a.hive_id == b.hive_id);
  VCHECK(a.node_id != b.node_id);
  VCHECK_EQ(a.lanes, 16);
  VCHECK_EQ(a.index, 0u);
  VCHECK_EQ(b.index, 1u);
  VCHECK_EQ(amdsmi_gpu_xgmi_error_status(g[0], &status), kSuccess);
  VCHECK_EQ(status, 0);   // AMDSMI_XGMI_STATUS_NO_ERRORS
  VCHECK_EQ(amdsmi_get_gpu_xgmi_link_status(g[0], &links), kSuccess);
  VCHECK_EQ(links.total, 1u);   // the one peer of a two-GPU machine
  VCHECK_EQ(links.status[0], 1);   // up
}

VTEST(pcie_info_replays_and_link_as_the_machine_has_them) {
  amdsmi_init(2);
  TempState state;
  const std::vector<void*> g = gpus();
  const bool rx = radeon();
  vgpu::ras::inject_pcie(uuid_of(0), vgpu::ras::Pcie::Replay, 4);
  vgpu::ras::inject_pcie(uuid_of(0), vgpu::ras::Pcie::NaksSent, 2);
  PcieInfo p{};
  VCHECK_EQ(amdsmi_get_pcie_info(g[0], &p), kSuccess);
  // MI300X: Gen5 x16; RX 7900 XTX: Gen4 x16 (the profiles').
  VCHECK_EQ(p.stat.max_width, 16);
  VCHECK_EQ(p.stat.max_speed, rx ? 16u : 32u);
  VCHECK_EQ(p.stat.max_version, rx ? 4u : 5u);
  VCHECK_EQ(p.stat.slot_type, rx ? 0 : 1);   // PCIe card, OAM module
  VCHECK_EQ(p.metric.width, 16);
  VCHECK_EQ(p.metric.speed, rx ? 16000u : 32000u);   // MT/s
  VCHECK_EQ(p.metric.replay, uint64_t{4});
  VCHECK_EQ(p.metric.nak_sent, uint64_t{2});
  VCHECK_EQ(p.metric.nak_received, uint64_t{0});
  uint64_t replay = 0;
  VCHECK_EQ(amdsmi_get_gpu_pci_replay_counter(g[0], &replay), kSuccess);
  VCHECK_EQ(replay, uint64_t{4});
  VCHECK_EQ(amdsmi_get_gpu_pci_replay_counter(g[1], &replay), kSuccess);
  VCHECK_EQ(replay, uint64_t{0});
  // The generations up to the card's, the trained one marked.
  struct Bw {
    bool deep;
    uint32_t n, current;
    uint64_t freq[33];
    uint32_t lanes[33];
  } bw{};
  VCHECK_EQ(amdsmi_get_gpu_pci_bandwidth(g[0], &bw), kSuccess);
  VCHECK_EQ(bw.n, rx ? 4u : 5u);
  VCHECK_EQ(bw.current, bw.n - 1);
  VCHECK_EQ(bw.freq[bw.current], rx ? uint64_t{16000000000} : uint64_t{32000000000});
  VCHECK_EQ(bw.lanes[0], 16u);
}

VTEST(sensors_are_the_profiles_and_missing_ones_are_refused) {
  amdsmi_init(2);
  const bool rx = radeon();
  vgpu::telemetry::DeviceSample d{};
  vgpu::telemetry::describe_device(vgpu::load_gpu(rx ? "amd/rx7900xtx" : "amd/mi300x"), 0, &d);
  const std::vector<void*> g = gpus();
  int64_t t = 0;
  // Temperatures in degrees: no edge sensor on an Instinct card.
  VCHECK_EQ(amdsmi_get_temp_metric(g[0], 0, 0, &t), rx ? kSuccess : kNotSupported);
  VCHECK_EQ(amdsmi_get_temp_metric(g[0], 1, 0, &t), kSuccess);
  VCHECK(t > 0 && t < 150);
  VCHECK_EQ(amdsmi_get_temp_metric(g[0], 1, 1, &t), kSuccess);
  VCHECK_EQ(t, int64_t{d.temperature_max_c});
  VCHECK_EQ(amdsmi_get_temp_metric(g[0], 2, 0, &t), kSuccess);   // both profiles have a memory sensor
  VCHECK_EQ(amdsmi_get_temp_metric(g[0], 3, 0, &t), rx ? kNotSupported : kSuccess);   // HBM stack 0
  VCHECK_EQ(amdsmi_get_temp_metric(g[0], 100, 0, &t), kNotSupported);   // a board node
  // Power, in watts, and its cap in microwatts.
  PowerInfo pw{};
  VCHECK_EQ(amdsmi_get_power_info(g[0], &pw), kSuccess);
  VCHECK_EQ(pw.limit, rx ? 355u : 750u);
  VCHECK_EQ(rx ? pw.current : pw.average, 0xFFFFFFFFu);   // CDNA3 gives the current power, Radeon the average
  VCHECK_EQ(pw.socket, uint64_t{rx ? pw.average : pw.current});
  VCHECK(pw.gfx_mv > 0);
  VCHECK_EQ(amdsmi_get_gpu_volt_metric(g[0], 0, 0, &t), kSuccess);   // the one voltage the model keeps
  VCHECK_EQ(int64_t{static_cast<int64_t>(pw.gfx_mv)}, t);
  PowerCap cap{};
  VCHECK_EQ(amdsmi_get_power_cap_info(g[0], 0, &cap), kSuccess);
  VCHECK_EQ(cap.cap, uint64_t{rx ? 355000000u : 750000000u});
  VCHECK_EQ(cap.max, cap.cap);
  VCHECK_EQ(amdsmi_get_power_cap_info(g[0], 1, &cap), kNotSupported);
  // Clocks: graphics and memory; the data fabric's is not modelled.
  ClkInfo clk{};
  VCHECK_EQ(amdsmi_get_clock_info(g[0], 0, &clk), kSuccess);
  VCHECK_EQ(clk.max_clk, d.sm_clock_max_mhz);
  VCHECK(clk.min_clk < clk.max_clk && clk.clk <= clk.max_clk);
  VCHECK_EQ(amdsmi_get_clock_info(g[0], 4, &clk), kSuccess);
  VCHECK_EQ(clk.max_clk, d.mem_clock_max_mhz);
  VCHECK_EQ(amdsmi_get_clock_info(g[0], 1, &clk), kNotSupported);
  VCHECK_EQ(amdsmi_get_gpu_volt_metric(g[0], 0, 0, &t), kSuccess);
  VCHECK_EQ(amdsmi_get_gpu_volt_metric(g[0], 1, 0, &t), kNotSupported);
  // Fans on Radeon cards only.
  VCHECK_EQ(amdsmi_get_gpu_fan_speed(g[0], 0, &t), rx ? kSuccess : kNotSupported);
}

VTEST(partitions_exist_only_on_cdna3_and_cdna4) {
  amdsmi_init(2);
  const std::vector<void*> g = gpus();
  char part[16] = {};
  VCHECK_EQ(amdsmi_get_gpu_compute_partition(g[0], part, sizeof part), radeon() ? kNotSupported : kSuccess);
  if (!radeon()) VCHECK_EQ(std::string(part), std::string("SPX"));
  VCHECK_EQ(amdsmi_get_gpu_memory_partition(g[0], part, sizeof part), radeon() ? kNotSupported : kSuccess);
  if (!radeon()) VCHECK_EQ(std::string(part), std::string("NPS1"));
}

VTEST(the_process_list_counts_then_fills) {
  amdsmi_init(2);
  const std::vector<void*> g = gpus();
  uint32_t n = 7;
  VCHECK_EQ(amdsmi_get_gpu_process_list(g[0], &n, nullptr), kSuccess);
  VCHECK_EQ(n, 0u);   // nothing is running on the test machine
  VCHECK_EQ(amdsmi_get_gpu_process_list(nullptr, &n, nullptr), kNotFound);
}

// What RVS's power, thermal and stress modules ask each GPU, answered from the
// profile: watts, degrees, clock levels it can pin, an energy counter that only
// Instinct has.
VTEST(sensors_answer_what_rvs_asks_by_the_profile) {
  amdsmi_init(2);
  const bool rx = radeon();
  const char* env = std::getenv("VGPU_GPU");
  const vgpu::DeviceProfile p = vgpu::load_gpu(env ? env : "amd/mi300x");
  const std::vector<void*> g = gpus();
  for (size_t i = 0; i < g.size(); ++i) {
    vgpu::telemetry::DeviceSample d{};
    vgpu::telemetry::describe_device(p, static_cast<int>(i), &d);
    PowerInfo pw{};
    VCHECK_EQ(amdsmi_get_power_info(g[i], &pw), kSuccess);
    VCHECK(pw.socket > 0 && pw.socket < 2000);
    VCHECK_EQ(pw.limit, (d.power_limit_mw + 500) / 1000);
    // One of the two readings, the other UINT32_MAX: MI300 and newer read
    // current, the earlier Instinct and Radeon average.
    const bool current = std::string(d.architecture) == "cdna3" || std::string(d.architecture) == "cdna4";
    VCHECK_EQ(current ? pw.average : pw.current, 0xFFFFFFFFu);
    VCHECK_EQ(current ? pw.current : pw.average, static_cast<uint32_t>(pw.socket));
    PowerCap cap{};
    VCHECK_EQ(amdsmi_get_power_cap_info(g[i], 0, &cap), kSuccess);
    VCHECK_EQ(cap.max, uint64_t{d.power_limit_mw} * 1000);
    VCHECK_EQ(amdsmi_get_power_cap_info(g[i], 3, &cap), kNotSupported);

    int64_t t = -1;
    VCHECK_EQ(amdsmi_get_temp_metric(g[i], 1, 0, &t), kSuccess);   // junction, degrees
    VCHECK(t > 0 && t < 150);
    // The slowdown limit, where the profile has one (mi325x's is unknown, 0).
    VCHECK_EQ(amdsmi_get_temp_metric(g[i], 1, 1, &t), d.temperature_max_c ? kSuccess : kNotSupported);
    if (d.temperature_max_c) VCHECK_EQ(t, int64_t{d.temperature_max_c});
    // No edge sensor on an Instinct; no HBM on a Radeon.
    VCHECK_EQ(amdsmi_get_temp_metric(g[i], 0, 0, &t), rx ? kSuccess : kNotSupported);
    VCHECK_EQ(amdsmi_get_temp_metric(g[i], 3, 0, &t), rx || !d.has_memory_temperature ? kNotSupported : kSuccess);
    VCHECK_EQ(amdsmi_get_temp_metric(g[i], 2, 0, &t), d.has_memory_temperature ? kSuccess : kNotSupported);   // VRAM
    VCHECK_EQ(amdsmi_get_temp_metric(g[i], 300, 0, &t), kInval);

    Frequencies f{};
    VCHECK_EQ(amdsmi_get_clk_freq(g[i], kClkSys, &f), kSuccess);
    VCHECK_EQ(f.num_supported, 2u);
    VCHECK_EQ(f.frequency[1], uint64_t{d.sm_clock_max_mhz} * 1000000);
    VCHECK(f.frequency[0] < f.frequency[1]);
    VCHECK_EQ(amdsmi_get_clk_freq(g[i], kClkMem, &f), kSuccess);
    VCHECK_EQ(f.frequency[1], uint64_t{d.mem_clock_max_mhz} * 1000000);
    VCHECK_EQ(amdsmi_get_clk_freq(g[i], 1, &f), kNotSupported);   // data fabric: not modelled
    // RVS's PULSE probe: each level alone must be settable, the one past the last not.
    VCHECK_EQ(amdsmi_set_clk_freq(g[i], kClkSys, 1), kSuccess);
    VCHECK_EQ(amdsmi_get_clk_freq(g[i], kClkSys, &f), kSuccess);
    VCHECK_EQ(f.current, 0u);
    VCHECK_EQ(amdsmi_set_clk_freq(g[i], kClkSys, 2), kSuccess);
    amdsmi_get_clk_freq(g[i], kClkSys, &f);
    VCHECK_EQ(f.current, 1u);
    VCHECK_EQ(amdsmi_set_clk_freq(g[i], kClkSys, 4), kOutOfBounds);
    VCHECK_EQ(amdsmi_set_clk_freq(g[i], kClkSys, 3), kSuccess);   // restored
    VCHECK_EQ(amdsmi_set_clk_freq(g[i], kClkMem, 0), kOutOfBounds);

    // Energy: Instinct's accumulator counts up; a Radeon has none.
    uint64_t e0 = 0, e1 = 0, ts = 0;
    float res = 0;
    const int r = amdsmi_get_energy_count(g[i], &e0, &res, &ts);
    VCHECK_EQ(r, rx ? kNotSupported : kSuccess);
    if (!rx) {
      VCHECK(e0 > 0 && res > 15.0f && res < 15.5f);
      usleep(20000);
      VCHECK_EQ(amdsmi_get_energy_count(g[i], &e1, &res, &ts), kSuccess);
      VCHECK(e1 > e0);
    }
  }
  void* bad = reinterpret_cast<void*>(0x1234);
  PowerInfo pw{};
  VCHECK_EQ(amdsmi_get_power_info(bad, &pw), kNotFound);
  VCHECK_EQ(amdsmi_get_power_info(g[0], nullptr), kInval);
}

// ECC and RAS: Instinct ships with them, a Radeon without; counts follow the RAS model.
VTEST(ecc_for_rvs_is_there_only_where_the_profile_has_it) {
  amdsmi_init(2);
  const bool rx = radeon();
  const std::vector<void*> g = gpus();
  uint64_t blocks = 99;
  VCHECK_EQ(amdsmi_get_gpu_ecc_enabled(g[0], &blocks), kSuccess);
  VCHECK_EQ(blocks != 0, !rx);
  int state = 0;
  VCHECK_EQ(amdsmi_get_gpu_ecc_status(g[0], 1, &state), kSuccess);   // UMC
  VCHECK_EQ(state, rx ? 1 : 6);                                     // DISABLED : ENABLED
  VCHECK_EQ(amdsmi_get_gpu_ecc_status(g[0], 3, &state), kInval);    // two blocks at once
  VCHECK_EQ(amdsmi_get_gpu_ecc_status(g[0], 0x80, &state), kSuccess);                       // XGMI: as the others
  ErrorCount ec{};
  VCHECK_EQ(amdsmi_get_gpu_total_ecc_count(g[0], &ec), rx ? kNotSupported : kSuccess);
  if (!rx) VCHECK_EQ(ec.correctable + ec.uncorrectable, uint64_t{0});
}

// amdsmi_gpu_metrics_t (4544 bytes in ROCm 7.2.0), read at the offsets compiled
// from that release's header: the table's v1.5 fields copied across, the rest
// all ones.
VTEST(gpu_metrics_are_the_v1_5_table_in_amd_smis_structure) {
  amdsmi_init(2);
  const std::vector<void*> g = gpus();
  std::vector<unsigned char> m(4544, 0);
  const auto u16 = [&](size_t o) { uint16_t v; std::memcpy(&v, &m[o], 2); return v; };
  const auto u32 = [&](size_t o) { uint32_t v; std::memcpy(&v, &m[o], 4); return v; };
  const auto u64 = [&](size_t o) { uint64_t v; std::memcpy(&v, &m[o], 8); return v; };
  const bool rx = radeon();
  const char* env = std::getenv("VGPU_GPU");
  const vgpu::DeviceProfile p = vgpu::load_gpu(env ? env : "amd/mi300x");
  for (size_t i = 0; i < g.size(); ++i) {
    vgpu::telemetry::DeviceSample d{};
    vgpu::telemetry::describe_device(p, static_cast<int>(i), &d);
    VCHECK_EQ(amdsmi_get_gpu_metrics_info(g[i], m.data()), kSuccess);
    VCHECK_EQ(u16(0), 360u);                      // the table's size, as the header says
    VCHECK_EQ(m[2], 1);                           // format revision
    VCHECK_EQ(m[3], 5);                           // content revision
    VCHECK_EQ(u16(4), 0xFFFFu);                   // temperature_edge: not in the table
    VCHECK(u16(6) > 0 && u16(6) < 150);           // temperature_hotspot
    VCHECK(u16(120) > 0 && u16(120) <= d.power_limit_mw / 1000);   // current_socket_power
    VCHECK_EQ(u16(22), 0xFFFFu);                  // average_socket_power: not in v1.5
    VCHECK_EQ(u16(74), d.pcie_width);             // pcie_link_width
    VCHECK_EQ(u16(54), u16(312));                 // current_gfxclk is current_gfxclks[0]
    VCHECK(u16(312) > 0 && u16(312) <= d.sm_clock_max_mhz);
    VCHECK(u16(58) > 0 && u16(58) <= d.mem_clock_max_mhz);   // current_uclk
    VCHECK_EQ(u32(132), 0u);                      // gfxclk_lock_status
    VCHECK(u64(32) > 0);                          // system_clock_counter
    VCHECK_EQ(u64(4520), ~uint64_t{0});           // vram_max_bandwidth: v1.7, not in the table
    VCHECK_EQ(u16(4528), 0xFFFFu);                // xgmi_link_status[0]
    VCHECK_EQ(u64(184), 0u);                      // xgmi_read_data_acc[0]
    // The accumulator: Instinct only, the one amdsmi_get_energy_count counts.
    if (rx) {
      VCHECK_EQ(u64(24), ~uint64_t{0});
    } else {
      uint64_t e = 0, ts = 0;
      float res = 0;
      VCHECK(u64(24) > 0 && u64(24) != ~uint64_t{0});
      VCHECK_EQ(amdsmi_get_energy_count(g[i], &e, &res, &ts), kSuccess);
      VCHECK(e >= u64(24) && e - u64(24) < 1000000);
    }
  }
  VCHECK_EQ(amdsmi_get_gpu_metrics_info(g[0], nullptr), kInval);
  VCHECK_EQ(amdsmi_get_gpu_metrics_info(reinterpret_cast<void*>(0x1234), m.data()), kNotFound);
}

VTEST(what_is_not_modelled_is_refused_and_every_function_is_there) {
  amdsmi_init(2);
  const std::vector<void*> g = gpus();
  char buf[4096] = {};
  VCHECK_EQ(amdsmi_get_gpu_vbios_info(g[0], buf), kNotSupported);
  VCHECK_EQ(amdsmi_set_gpu_perf_level(g[0], 0), kNotSupported);
  VCHECK_EQ(amdsmi_get_violation_status(g[0], buf), kNotSupported);
  int missing = 0, count = 0;
#define AMDSMI_SYMBOL(name)                              \
  ++count;                                               \
  if (!dlsym(RTLD_DEFAULT, #name)) {                     \
    std::fprintf(stderr, "not exported: %s\n", #name);   \
    ++missing;                                           \
  }
#include "../../src/amd_smi_symbols.inc"
#undef AMDSMI_SYMBOL
  VCHECK_EQ(missing, 0);
  VCHECK(count >= 190);
  VCHECK_EQ(amdsmi_shut_down(), kSuccess);
}

VTEST_MAIN
