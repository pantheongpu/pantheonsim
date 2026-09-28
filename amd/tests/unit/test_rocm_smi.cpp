// VirtualGPU's librocm_smi64, as RCCL uses it: how many GPUs, where each is
// on the PCI bus, and what links each pair. The answers have to agree with
// what the HIP runtime says -- RCCL matches the two by PCI address -- and
// name the link the machine has: Infinity Fabric (XGMI) between Instinct
// GPUs, PCI Express between Radeon ones.
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

VTEST_MAIN
