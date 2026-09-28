// librocm_smi64: the part of ROCm SMI's library that RCCL asks about the GPUs
// it connects -- how many there are, where each sits on the PCI bus, and what
// links them -- answered from VirtualGPU's HIP runtime, so RCCL (and PyTorch's
// multi-GPU and distributed training over it) starts on simulated GPUs.
//
// RCCL learns the topology one of two ways: by reading the AMD kernel
// driver's topology under /sys/class/kfd (its default), or through this
// library (RCCL_USE_ROCM_SMI_LIB=1). Under WSL it does neither. On any other
// Linux machine there is no /sys/class/kfd without an AMD GPU, and RCCL's
// initialization fails ("internal error"), so this library, once loaded,
// asks RCCL for the second way unless the environment already says which.
// RCCL reads that setting when it first initializes, after the libraries it
// links -- this one among them -- are loaded.
//
// The answers are the ones the HIP runtime gives, so the two agree: the
// devices HIP counts, each on the bus HIP reports it on, and the link a
// machine of that kind has between its GPUs -- AMD's Infinity Fabric (XGMI)
// between Instinct GPUs, PCI Express between Radeon ones. The declarations
// follow ROCm SMI's documented interface (rocm_smi/rocm_smi.h, MIT); nothing
// here is AMD's code. Everything else ROCm SMI offers -- clocks, power,
// temperatures -- is not here; `vgpu smi` answers those.
#include <sys/stat.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "vgpu/hip_abi.hpp"

namespace {

// rsmi_status_t, in rocm_smi.h's order.
enum Status : int {
  kSuccess = 0,
  kInvalidArgs = 1,
  kNotSupported = 2,
  kInputOutOfBounds = 7,
};

// RSMI_IO_LINK_TYPE.
enum LinkType : int {
  kLinkUndefined = 0,
  kLinkPcie = 1,
  kLinkXgmi = 2,
};

struct Version {  // rsmi_version_t
  uint32_t major, minor, patch;
  const char* build;
};

using vgpu::amd::abi::DevicePropR0600;
using Attr = vgpu::amd::abi::DeviceAttribute;

}  // namespace

// The HIP runtime's own entry points, from libamdhip64 beside this library.
extern "C" int hipGetDeviceCount(int* count);
extern "C" int hipDeviceGetAttribute(int* value, int attribute, int device);
extern "C" int hipGetDevicePropertiesR0600(DevicePropR0600* props, int device);

#define RSMI_API extern "C" __attribute__((visibility("default")))

namespace {

int count() {
  int n = 0;
  return hipGetDeviceCount(&n) == 0 ? n : 0;
}

bool valid(uint32_t d) { return d < static_cast<uint32_t>(count()); }

// Instinct GPUs (gfx9: CDNA) are joined by Infinity Fabric in the machines
// they come in; Radeon ones (gfx10 and later: RDNA) by PCI Express alone.
bool instinct(uint32_t d) {
  DevicePropR0600 p{};
  if (hipGetDevicePropertiesR0600(&p, static_cast<int>(d)) != 0) return false;
  return std::strncmp(p.gcnArchName, "gfx9", 4) == 0;
}

// Tells RCCL to ask this library rather than read /sys/class/kfd, where
// there is none to read. An environment that already chose is left alone.
__attribute__((constructor)) void prefer_this_library() {
  struct stat st{};
  if (stat("/sys/class/kfd/kfd/topology/nodes", &st) != 0) setenv("RCCL_USE_ROCM_SMI_LIB", "1", 0);
}

}  // namespace

RSMI_API int rsmi_init(uint64_t) { return kSuccess; }
RSMI_API int rsmi_shut_down() { return kSuccess; }

RSMI_API int rsmi_num_monitor_devices(uint32_t* n) {
  if (!n) return kInvalidArgs;
  *n = static_cast<uint32_t>(count());
  return kSuccess;
}

// BDFID: domain in bits 32-63, bus 8-15, device 3-7, function 0-2 -- the
// address hipDeviceGetPCIBusId prints.
RSMI_API int rsmi_dev_pci_id_get(uint32_t d, uint64_t* bdfid) {
  if (!bdfid) return kInvalidArgs;
  if (!valid(d)) return kInputOutOfBounds;
  int domain = 0, bus = 0, device = 0;
  const int dev = static_cast<int>(d);
  hipDeviceGetAttribute(&domain, static_cast<int>(Attr::kPciDomainID), dev);
  hipDeviceGetAttribute(&bus, static_cast<int>(Attr::kPciBusId), dev);
  hipDeviceGetAttribute(&device, static_cast<int>(Attr::kPciDeviceId), dev);
  *bdfid = (static_cast<uint64_t>(static_cast<uint32_t>(domain)) << 32) |
           (static_cast<uint64_t>(bus & 0xff) << 8) | (static_cast<uint64_t>(device & 0x1f) << 3);
  return kSuccess;
}

// One hop between any two: every Instinct GPU in a platform has a link to
// each of the others, and Radeon GPUs meet at the host's PCIe root.
RSMI_API int rsmi_topo_get_link_type(uint32_t src, uint32_t dst, uint64_t* hops, int* type) {
  if (!hops || !type) return kInvalidArgs;
  if (!valid(src) || !valid(dst)) return kInputOutOfBounds;
  *hops = src == dst ? 0 : 1;
  *type = instinct(src) && instinct(dst) ? kLinkXgmi : kLinkPcie;
  return kSuccess;
}

// The weights the AMD kernel driver gives these links: 15 for XGMI, 20 for
// PCI Express; lower is closer.
RSMI_API int rsmi_topo_get_link_weight(uint32_t src, uint32_t dst, uint64_t* weight) {
  if (!weight) return kInvalidArgs;
  if (!valid(src) || !valid(dst)) return kInputOutOfBounds;
  *weight = src == dst ? 0 : instinct(src) && instinct(dst) ? 15 : 20;
  return kSuccess;
}

// An XGMI link's bandwidth, in MB/s: MI300X's datasheet gives each of its
// Infinity Fabric links 64 GB/s in each direction. Placeholder, like the
// profiles' numbers; the simulator's copies take no modelled time. ROCm SMI
// answers this only for XGMI.
RSMI_API int rsmi_minmax_bandwidth_get(uint32_t src, uint32_t dst, uint64_t* min_bw, uint64_t* max_bw) {
  if (!min_bw || !max_bw) return kInvalidArgs;
  if (!valid(src) || !valid(dst) || src == dst) return kInputOutOfBounds;
  if (!instinct(src) || !instinct(dst)) return kNotSupported;
  *min_bw = *max_bw = 64000;
  return kSuccess;
}

RSMI_API int rsmi_version_get(Version* v) {
  if (!v) return kInvalidArgs;
  *v = {7, 8, 0, "VirtualGPU"};
  return kSuccess;
}

RSMI_API int rsmi_status_string(int status, const char** out) {
  if (!out) return kInvalidArgs;
  switch (status) {
    case kSuccess: *out = "RSMI_STATUS_SUCCESS: The function has been executed successfully."; break;
    case kInvalidArgs: *out = "RSMI_STATUS_INVALID_ARGS: The provided arguments do not meet the preconditions required for the input."; break;
    case kNotSupported: *out = "RSMI_STATUS_NOT_SUPPORTED: The requested information or action is not available for the given input, on the given system"; break;
    case kInputOutOfBounds: *out = "RSMI_STATUS_INPUT_OUT_OF_BOUNDS: The provided input is out of allowable or safe range"; break;
    default: *out = "An unknown error occurred"; break;
  }
  return kSuccess;
}
