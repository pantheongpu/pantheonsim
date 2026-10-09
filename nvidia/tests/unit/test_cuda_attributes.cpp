// What CUDA tells a program about a device (vgpu/cuda_attributes.hpp), and which
// devices it is shown (vgpu/runtime/visible_devices.hpp), with no simulator
// running.
//
// The numbers in rtx3060_matches_the_card were read from an RTX 3060 (driver
// 13.0) with nvidia/tools/cuda-attributes.cu; those of the visible-devices
// cases from two of them, running a program under each value of
// CUDA_VISIBLE_DEVICES. They are the card's answers, not ours.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "vgpu/cuda_attributes.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/visible_devices.hpp"
#include "vgpu/telemetry.hpp"
#include "vtest.hpp"

namespace {

using vgpu::cuda::device_attribute;
namespace A = vgpu::cuda;

int attr(const vgpu::DeviceProfile& p, int id, int physical = 0) {
  int v = -987654;
  VCHECK(device_attribute(p, physical, id, &v));
  return v;
}

std::vector<std::string> nvidia_gpus() {
  std::vector<std::string> out;
  for (const std::string& id : vgpu::available_gpus())
    if (id.rfind("nvidia/", 0) == 0) out.push_back(id);
  return out;
}

}  // namespace

// Every value below is one the card gave, for an attribute that is a fact of
// the card or of its compute capability. The ones that depend on where a card
// sits (its PCI address, the display watchdog, the host's NUMA node) are left
// out; so are the capabilities the simulator does not implement, which the
// card says yes to and this says no (sparse and deferred-mapped arrays,
// compressible memory, host memory pools, 64-bit stream memory operations).
VTEST(rtx3060_matches_the_card) {
  const vgpu::DeviceProfile p = vgpu::load_gpu("nvidia/rtx3060");
  const struct { int id; int want; } card[] = {
      {A::kClockRate, 1837000},
      {A::kMemoryClockRate, 7501000},
      {A::kGlobalMemoryBusWidth, 192},
      {A::kL2CacheSize, 2359296},
      {A::kMaxPersistingL2CacheSize, 1622016},
      {A::kMaxAccessPolicyWindowSize, 134213632},
      {A::kAsyncEngineCount, 1},
      {A::kSingleToDoublePrecisionPerfRatio, 64},
      {A::kMaximumTexture1dWidth, 131072},
      {A::kMaximumTexture1dLinearWidth, 268435456},
      {A::kMaximumTexture3dWidthAlternate, 8192},
      {A::kMaximumTexture3dHeightAlternate, 8192},
      {A::kMaximumTexture3dDepthAlternate, 32768},
      {A::kCanTex2dGather, 1},
      {A::kMaximumSurface2dWidth, 131072},
      {A::kMaximumSurfacecubemapLayeredLayers, 2046},
      {A::kMemSyncDomainCount, 1},
      {A::kClusterLaunch, 0},
      {A::kTensorMapAccessSupported, 0},
      {A::kMemoryPoolsSupported, 1},
      {A::kComputePreemptionSupported, 1},
      {A::kMaxSharedMemoryPerBlockOptin, 101376},
      {A::kMaxBlocksPerMultiprocessor, 16},
      {A::kReservedSharedMemoryPerBlock, 1024},
      {A::kMultiprocessorCount, 28},
      {A::kGpuPciDeviceId, 621023454},
  };
  for (const auto& c : card) {
    const int got = attr(p, c.id);
    if (got != c.want) std::fprintf(stderr, "attribute %d: card %d, here %d\n", c.id, c.want, got);
    VCHECK_EQ(got, c.want);
  }
}

// The single- to double-precision ratio follows NVIDIA's "Throughput of Native
// Arithmetic Instructions" table (CUDA C++ Best Practices Guide 13.4), fp32 over
// fp64 results per clock per multiprocessor: 7.5 is 64/2, 8.0 is 64/32, 8.6 and
// 8.9 are 128/2, 9.0 and 10.0 are 128/64, and 10.3 and 12.x are 128/2. The 10.3
// column shares its fp64 cell with 12.x, so a B300 is not a B200 here. 10.7 (Rubin)
// is derived from the 130 and 33 TFLOPS in NVIDIA's Rubin blog, and 11.0 (Thor)
// has no published rate, so it carries the 12.x stand-in (see device_attributes.cpp).
VTEST(single_to_double_ratio_follows_the_arithmetic_table) {
  const std::pair<int, int> want[] = {{75, 32}, {80, 2}, {86, 64}, {89, 64}, {90, 2}, {100, 2},
                                      {103, 64}, {107, 4}, {110, 64}, {120, 64}, {121, 64}};
  int seen = 0;
  for (const std::string& id : nvidia_gpus()) {
    const vgpu::DeviceProfile p = vgpu::load_gpu(id);
    const int cc = p.cc_major * 10 + p.cc_minor;
    bool found = false;
    for (const auto& w : want) {
      if (w.first != cc) continue;
      found = true;
      if (attr(p, A::kSingleToDoublePrecisionPerfRatio) != w.second)
        std::fprintf(stderr, "%s (cc %d): ratio %d, table says %d\n", id.c_str(), cc,
                     attr(p, A::kSingleToDoublePrecisionPerfRatio), w.second);
      VCHECK_EQ(attr(p, A::kSingleToDoublePrecisionPerfRatio), w.second);
    }
    VCHECK(found);   // a profile of a new compute capability must be added to the table above
    ++seen;
  }
  VCHECK(seen >= 12);
}

// Whatever a profile says, the table answers every attribute CUDA 13.2 has and
// refuses every number that is not one, and the answers do not contradict one
// another or the hardware's rules.
VTEST(every_nvidia_profile_answers_every_attribute_and_stays_consistent) {
  const std::vector<std::string> gpus = nvidia_gpus();
  VCHECK(gpus.size() >= 12);
  for (const std::string& id : gpus) {
    const vgpu::DeviceProfile p = vgpu::load_gpu(id);
    int v = 0;
    for (int a = 1; a <= A::kAtomicReductionSupported; ++a)
      if (!device_attribute(p, 0, a, &v)) std::fprintf(stderr, "%s: attribute %d refused\n", id.c_str(), a);
    for (int a : {0, -1, A::kAtomicReductionSupported + 1, 1000, 123456}) {
      v = 77;
      VCHECK(!device_attribute(p, 0, a, &v));
      VCHECK_EQ(v, 77);   // untouched
    }
    // The L2 and the memory interface are facts of the card, not zeros -- except
    // where NVIDIA has not published the L2 (the profile leaves it unset).
    const bool l2_unpublished = id == "nvidia/b200" || id == "nvidia/b300" || id == "nvidia/gb200" ||
                                id == "nvidia/vr200" || id == "nvidia/thor" || id == "nvidia/gb10" ||
                                id == "nvidia/a30";
    if (!l2_unpublished) VCHECK(attr(p, A::kL2CacheSize) > 0);
    VCHECK(attr(p, A::kGlobalMemoryBusWidth) >= 128);
    VCHECK(attr(p, A::kClockRate) >= 1000000);
    VCHECK(attr(p, A::kMemoryClockRate) >= 1000000);
    VCHECK(attr(p, A::kAsyncEngineCount) >= 1);
    VCHECK(attr(p, A::kMaxPersistingL2CacheSize) <= attr(p, A::kL2CacheSize));
    const int cc = p.cc_major * 10 + p.cc_minor;
    // The access policy window and its L2 set-aside come with 8.0.
    VCHECK_EQ(attr(p, A::kMaxAccessPolicyWindowSize) > 0, cc >= 80);
    VCHECK_EQ(attr(p, A::kMaxPersistingL2CacheSize) > 0, cc >= 80 && !l2_unpublished);
    // Clusters and the tensor map are Hopper's.
    VCHECK_EQ(attr(p, A::kClusterLaunch), cc >= 90 ? 1 : 0);
    VCHECK_EQ(attr(p, A::kTensorMapAccessSupported), cc >= 90 ? 1 : 0);
    // What the profile's own limits say, said the same way.
    VCHECK_EQ(attr(p, A::kMultiprocessorCount), static_cast<int>(p.limits.multiprocessors));
    VCHECK_EQ(attr(p, A::kMaxSharedMemoryPerBlockOptin), static_cast<int>(p.limits.shared_mem_per_block_optin));
    VCHECK_EQ(attr(p, A::kEccEnabled), p.telemetry.ecc ? 1 : 0);
    // The CUDA clock is the SM clock NVML knows, or a boost clock below it.
    VCHECK(attr(p, A::kClockRate) <= static_cast<int>(p.telemetry.sm_clock_max_mhz) * 1000);
    VCHECK_EQ(attr(p, A::kMemoryClockRate), static_cast<int>(p.telemetry.mem_clock_max_mhz) * 1000);
  }
}

// CUDA and NVML name one device the same way: the UUID bytes are the digits of
// the "GPU-" string nvidia-smi prints, and the PCI address is the one its bus id
// says (with CUDA's four-digit domain).
VTEST(a_device_has_one_identity_for_cuda_and_nvml) {
  const vgpu::DeviceProfile p = vgpu::load_gpu("nvidia/a100");
  std::set<std::string> seen;
  for (int ordinal = 0; ordinal < 3; ++ordinal) {
    vgpu::telemetry::DeviceSample d{};
    vgpu::telemetry::describe_device(p, ordinal, &d);
    const A::Identity id = A::identity(p, ordinal);
    std::string hex;
    for (int i = 0; i < 16; ++i) {
      char b[3];
      std::snprintf(b, sizeof b, "%02x", id.uuid[i]);
      hex += b;
    }
    std::string nvml = d.uuid + 4;   // after "GPU-"
    nvml.erase(std::remove(nvml.begin(), nvml.end(), '-'), nvml.end());
    VCHECK_EQ(hex, nvml);
    seen.insert(hex);
    char text[32];
    A::pci_bus_id(id, text, sizeof text);
    char want[32];
    std::snprintf(want, sizeof want, "0000:%02X:00.0", ordinal + 1);
    VCHECK_EQ(std::string(text), std::string(want));
    VCHECK_EQ(std::string("0000") + (d.bus_id + 8), std::string(want));
    VCHECK_EQ(attr(p, A::kPciBusId, ordinal), ordinal + 1);
    VCHECK_EQ(attr(p, A::kPciDomainId, ordinal), 0);
    VCHECK_EQ(attr(p, A::kPciDeviceId, ordinal), 0);
  }
  VCHECK_EQ(seen.size(), size_t{3});   // three devices, three UUIDs
}

namespace {

// What CUDA_VISIBLE_DEVICES shows of a machine of `machine` devices, as the
// devices' numbers and the error, in one string: "1 0" or "error 100".
std::string shown(const vgpu::DeviceProfile& p, int machine, const char* visible, const char* order = nullptr) {
  const vgpu::runtime::VisibleDevices v = vgpu::runtime::cuda_visible_devices(p, machine, visible, order);
  if (v.error) return "error " + std::to_string(v.error);
  std::string s;
  for (int d : v.physical) s += (s.empty() ? "" : " ") + std::to_string(d);
  return s;
}

std::string uuid_of(const vgpu::DeviceProfile& p, int ordinal) {
  vgpu::telemetry::DeviceSample d{};
  vgpu::telemetry::describe_device(p, ordinal, &d);
  return d.uuid;
}

}  // namespace

// Every row is what the pair of RTX 3060s did under that value.
VTEST(cuda_visible_devices_reads_as_the_driver_reads_it) {
  const vgpu::DeviceProfile p = vgpu::load_gpu("nvidia/rtx3060");
  const std::string u0 = uuid_of(p, 0), u1 = uuid_of(p, 1);
  const struct { const char* visible; const char* want; } rows[] = {
      {"0", "0"},
      {"1", "1"},
      {"0,1", "0 1"},
      {"1,0", "1 0"},
      {" 1", "1"},
      {"1 ", "1"},
      {"0, 1", "0 1"},
      {"01", "1"},          // read as a number
      {"0x1", "0"},         // as far as a number goes
      {"1.0", "1"},
      {"0,5", "0"},         // past the last device ends the list
      {"0,-1,1", "0"},      // so does a negative one
      {"0,abc,1", "0"},     // and one that is not a number
      {"", "error 100"},    // nothing shown
      {"5", "error 100"},
      {"5,0", "error 100"},
      {"-1", "error 100"},
      {"abc", "error 100"},
      {"NoDevFiles", "error 100"},
      {"GPU-", "error 100"},          // a UUID prefix of both names none
      {"GPU-00000000", "error 100"},  // and one of neither
      {"0,0", "error 101"},           // a device named twice by index
      {"1,1,0", "error 101"},
  };
  for (const auto& r : rows) {
    const std::string got = shown(p, 2, r.visible);
    if (got != r.want) std::fprintf(stderr, "CUDA_VISIBLE_DEVICES=\"%s\": card %s, here %s\n", r.visible, r.want, got.c_str());
    VCHECK_EQ(got, std::string(r.want));
  }
  VCHECK_EQ(shown(p, 2, nullptr), std::string("0 1"));   // unset
  // By UUID: the whole of it, a prefix that is one device's, either order, mixed with indices.
  VCHECK_EQ(shown(p, 2, u1.c_str()), std::string("1"));
  VCHECK_EQ(shown(p, 2, u1.substr(0, 12).c_str()), std::string("1"));
  VCHECK_EQ(shown(p, 2, (u0 + "," + u1).c_str()), std::string("0 1"));
  VCHECK_EQ(shown(p, 2, (u1 + "," + u0).c_str()), std::string("1 0"));
  // A list is all indices or all UUIDs: the first kind ends at an element of the other.
  VCHECK_EQ(shown(p, 2, (u1 + ",0").c_str()), std::string("1"));
  VCHECK_EQ(shown(p, 2, ("0," + u1).c_str()), std::string("0"));
  // A device named again, by the same kind of name, is an error.
  VCHECK_EQ(shown(p, 2, ("0," + u0).c_str()), std::string("0"));
  VCHECK_EQ(shown(p, 2, (u0 + ",0").c_str()), std::string("0"));
  VCHECK_EQ(shown(p, 2, ("1," + u1 + ",0").c_str()), std::string("1"));
  VCHECK_EQ(shown(p, 2, (u0 + "," + u0).c_str()), std::string("error 101"));
  VCHECK_EQ(shown(p, 2, (u0.substr(0, 12) + "," + u0).c_str()), std::string("error 101"));   // a prefix is a UUID too
  // CUDA_DEVICE_ORDER: the two orders it knows agree on identical devices; another is an error.
  VCHECK_EQ(shown(p, 2, "1,0", "FASTEST_FIRST"), std::string("1 0"));
  VCHECK_EQ(shown(p, 2, "1,0", "PCI_BUS_ID"), std::string("1 0"));
  VCHECK_EQ(shown(p, 2, nullptr, "bogus"), std::string("error 101"));
  VCHECK_EQ(shown(p, 2, "0", "bogus"), std::string("error 101"));
  // More devices than the pair the card runs had: the same rules.
  VCHECK_EQ(shown(p, 4, "3,1"), std::string("3 1"));
  VCHECK_EQ(shown(p, 4, "2,4,1"), std::string("2"));
}

// A machine of AMD's GPUs has no NVIDIA driver, so CUDA finds no device on it,
// whatever CUDA_VISIBLE_DEVICES says (offload-arch asked CUDA and was told of
// two sm_00 devices on a simulated MI300X).
VTEST(an_amd_machine_has_no_cuda_device) {
  for (const char* id : {"amd/mi300x", "amd/rx7900xtx"}) {
    const vgpu::DeviceProfile p = vgpu::load_gpu(id);
    VCHECK_EQ(shown(p, 2, nullptr), std::string("error 100"));
    VCHECK_EQ(shown(p, 2, "0"), std::string("error 100"));
  }
}

VTEST_MAIN
