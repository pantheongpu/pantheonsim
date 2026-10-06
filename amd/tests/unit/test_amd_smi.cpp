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
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/amd_kfd.hpp"
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
int amdsmi_get_clk_freq(void*, int, void*);
int amdsmi_set_clk_freq(void*, int, uint64_t);
int amdsmi_get_temp_metric(void*, int, int, int64_t*);
int amdsmi_get_power_info(void*, void*);
int amdsmi_get_power_cap_info(void*, uint32_t, void*);
int amdsmi_get_energy_count(void*, uint64_t*, float*, uint64_t*);
int amdsmi_get_gpu_ecc_enabled(void*, uint64_t*);
int amdsmi_get_gpu_ecc_status(void*, uint64_t, int*);
int amdsmi_get_gpu_total_ecc_count(void*, void*);
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

// The structs ROCm Validation Suite (RVS) reads: amdsmi_frequencies_t,
// amdsmi_power_info_t and amdsmi_power_cap_info_t, as the header lays them out.
struct Frequencies {
  bool has_deep_sleep;
  uint32_t num_supported, current;
  uint64_t frequency[33];
};
struct PowerInfo {
  uint64_t socket_power;
  uint32_t current_socket_power, average_socket_power;
  uint64_t gfx_voltage, soc_voltage, mem_voltage;
  uint32_t power_limit;
  uint64_t reserved[18];
};
struct PowerCapInfo {
  uint64_t power_cap, default_power_cap, dpm_cap, min_power_cap, max_power_cap, reserved[3];
};
struct ErrorCount {
  uint64_t correctable, uncorrectable, deferred, reserved[5];
};
constexpr int kInval = 1, kOutOfBounds = 17;
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
    VCHECK(pw.socket_power > 0 && pw.socket_power < 2000);
    VCHECK_EQ(pw.power_limit, (d.power_limit_mw + 500) / 1000);
    // One of the two readings, the other UINT32_MAX: MI300 and newer read
    // current, the earlier Instinct and Radeon average.
    const bool current = std::string(d.architecture) == "cdna3" || std::string(d.architecture) == "cdna4";
    VCHECK_EQ(current ? pw.average_socket_power : pw.current_socket_power, 0xFFFFFFFFu);
    VCHECK_EQ(current ? pw.current_socket_power : pw.average_socket_power, static_cast<uint32_t>(pw.socket_power));
    PowerCapInfo cap{};
    VCHECK_EQ(amdsmi_get_power_cap_info(g[i], 0, &cap), kSuccess);
    VCHECK_EQ(cap.max_power_cap, uint64_t{d.power_limit_mw} * 1000);
    VCHECK_EQ(amdsmi_get_power_cap_info(g[i], 3, &cap), kNotSupported);

    int64_t t = -1;
    VCHECK_EQ(amdsmi_get_temp_metric(g[i], 1, 0, &t), kSuccess);   // junction, degrees
    VCHECK(t > 0 && t < 150);
    VCHECK_EQ(amdsmi_get_temp_metric(g[i], 1, 1, &t), kSuccess);
    VCHECK_EQ(t, int64_t{d.temperature_max_c});
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
  VCHECK_EQ(amdsmi_get_gpu_ecc_status(g[0], 0x80, &state), rx ? kNotSupported : kSuccess);   // XGMI
  ErrorCount ec{};
  VCHECK_EQ(amdsmi_get_gpu_total_ecc_count(g[0], &ec), rx ? kNotSupported : kSuccess);
  if (!rx) VCHECK_EQ(ec.correctable + ec.uncorrectable, uint64_t{0});
}

VTEST(what_is_not_modelled_is_refused_and_every_function_is_there) {
  amdsmi_init(2);
  const std::vector<void*> g = gpus();
  char buf[4096] = {};
  VCHECK_EQ(amdsmi_get_gpu_vbios_info(g[0], buf), kNotSupported);
  VCHECK_EQ(amdsmi_set_gpu_perf_level(g[0], 0), kNotSupported);
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
