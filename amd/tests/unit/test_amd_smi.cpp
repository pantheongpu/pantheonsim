// VirtualGPU's libamd_smi, as AMD's Python package (amdsmi) and vLLM use it:
// the GPUs, one to a socket; each one's target, IDs, compute units, UUID, PCI
// address, KFD node and memory; the links between them. Each answer agrees
// with what the rest of the machine says of the same GPU: KFD's topology,
// ROCm SMI and amd-smi. And the package binds every function of the library
// when it is imported, so every one must be exported, the ones not modelled
// answering NOT_SUPPORTED.
#include <dlfcn.h>

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
