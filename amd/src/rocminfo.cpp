// rocminfo for a machine without ROCm: the HSA system and its agents, as
// ROCm's rocminfo prints them, asked of the runtime through the public HSA
// interface (vgpu/hsa_abi.h). Linked against VirtualGPU's HSA runtime, it
// reports what ROCm's own rocminfo reports on the same simulated machine, and
// a session runs ROCm's when the host has one (amd/tests/e2e/run_rocminfo.sh
// compares the two).
//
// The layout is rocminfo's: a label padded to 25 columns, then its value
// padded to 35, indented two columns a level.
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "vgpu/hsa_abi.h"

namespace {

void label(int indent, const char* l) { std::printf("%*s%-25s\n", indent, "", l); }
void value(int indent, const char* l, const std::string& v) { std::printf("%*s%-25s%-35s\n", indent, "", l, v.c_str()); }
void raw(int indent, const char* l, const std::string& v) { std::printf("%*s%-25s%s\n", indent, "", l, v.c_str()); }
std::string num(uint64_t n) {
  char b[48];
  std::snprintf(b, sizeof b, "%" PRIu64 "(0x%" PRIx64 ")", n, n);
  return b;
}
std::string dec(uint64_t n) { return std::to_string(n); }

template <typename T>
T agent(hsa_agent_t a, int attr) {
  T v{};
  hsa_agent_get_info(a, static_cast<hsa_agent_info_t>(attr), &v);
  return v;
}
std::string agent_string(hsa_agent_t a, int attr) {
  char s[128] = {};
  hsa_agent_get_info(a, static_cast<hsa_agent_info_t>(attr), s);
  return s;
}
template <typename T>
T pool(hsa_amd_memory_pool_t p, hsa_amd_memory_pool_info_t attr) {
  T v{};
  hsa_amd_memory_pool_get_info(p, attr, &v);
  return v;
}
template <typename T>
T isa(hsa_isa_t i, hsa_isa_info_t attr) {
  T v{};
  hsa_isa_get_info_alt(i, attr, &v);
  return v;
}
const char* rounding(uint32_t m) {
  return m == HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR ? "NEAR" : m == HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO ? "ZERO" : "DEFAULT";
}
const char* yes(bool b) { return b ? "TRUE" : "FALSE"; }

void dims(int indent, const char* heading, const uint32_t* d) {
  label(indent, heading);
  const char* axes[3] = {"x", "y", "z"};
  for (int k = 0; k < 3; ++k) value(indent + 2, axes[k], num(d[k]));
}

void print_pool(hsa_amd_memory_pool_t p, int n) {
  label(4, ("Pool " + std::to_string(n)).c_str());
  const uint32_t seg = pool<uint32_t>(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT);
  std::string segment = seg == HSA_AMD_SEGMENT_GLOBAL ? "GLOBAL; FLAGS:" : seg == HSA_AMD_SEGMENT_GROUP ? "GROUP"
                        : seg == HSA_AMD_SEGMENT_PRIVATE                  ? "PRIVATE"
                                                                          : "READONLY";
  if (seg == HSA_AMD_SEGMENT_GLOBAL) {
    const uint32_t f = pool<uint32_t>(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS);
    std::vector<std::string> flags;
    if (f & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) flags.push_back("KERNARG");
    if (f & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) flags.push_back("FINE GRAINED");
    if (f & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) flags.push_back("COARSE GRAINED");
    for (size_t k = 0; k < flags.size(); ++k) segment += (k ? ", " : " ") + flags[k];
  }
  value(6, "Segment:", segment);
  const uint64_t kb = pool<size_t>(p, HSA_AMD_MEMORY_POOL_INFO_SIZE) / 1024;
  value(6, "Size:", num(kb) + " KB");
  value(6, "Allocatable:", yes(pool<bool>(p, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED)));
  value(6, "Alloc Granule:", dec(pool<size_t>(p, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE) / 1024) + "KB");
  value(6, "Alloc Recommended Granule:",
        dec(pool<size_t>(p, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE) / 1024) + "KB");
  value(6, "Alloc Alignment:", dec(pool<size_t>(p, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT) / 1024) + "KB");
  value(6, "Accessible by all:", yes(pool<bool>(p, HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL)));
}

void print_isa(hsa_isa_t i, int n) {
  label(4, ("ISA " + std::to_string(n)).c_str());
  const uint32_t len = isa<uint32_t>(i, HSA_ISA_INFO_NAME_LENGTH);
  std::string name(len, '\0');
  hsa_isa_get_info_alt(i, HSA_ISA_INFO_NAME, name.data());
  name.resize(std::strlen(name.c_str()));
  value(6, "Name:", name);
  bool models[2] = {};
  hsa_isa_get_info_alt(i, HSA_ISA_INFO_MACHINE_MODELS, models);
  std::string m;
  if (models[0]) m += "HSA_MACHINE_MODEL_SMALL ";
  if (models[1]) m += "HSA_MACHINE_MODEL_LARGE";
  value(6, "Machine Models:", m);
  bool profiles[2] = {};
  hsa_isa_get_info_alt(i, HSA_ISA_INFO_PROFILES, profiles);
  std::string pr;
  if (profiles[0]) pr += "HSA_PROFILE_BASE";
  if (profiles[1]) pr += std::string(pr.empty() ? "" : " ") + "HSA_PROFILE_FULL";
  value(6, "Profiles:", pr);
  // Each a flag per mode (default, zero, near): the ones the ISA has.
  for (const hsa_isa_info_t attr : {HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES, HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES}) {
    bool modes[3] = {};
    hsa_isa_get_info_alt(i, attr, modes);
    std::string m;
    for (uint32_t k = 0; k < 3; ++k)
      if (modes[k]) m += std::string(m.empty() ? "" : " ") + rounding(k);
    value(6, "Default Rounding Mode:", m);
  }
  value(6, "Fast f16:", yes(isa<bool>(i, HSA_ISA_INFO_FAST_F16_OPERATION)));
  value(6, "Workgroup Max Size:", num(isa<uint32_t>(i, HSA_ISA_INFO_WORKGROUP_MAX_SIZE)));
  uint16_t wg[3] = {};
  hsa_isa_get_info_alt(i, HSA_ISA_INFO_WORKGROUP_MAX_DIM, wg);
  const uint32_t wg32[3] = {wg[0], wg[1], wg[2]};
  dims(6, "Workgroup Max Size per Dimension:", wg32);
  value(6, "Grid Max Size:", num(isa<uint32_t>(i, HSA_ISA_INFO_GRID_MAX_SIZE)));
  const hsa_dim3_t g = isa<hsa_dim3_t>(i, HSA_ISA_INFO_GRID_MAX_DIM);
  const uint32_t g32[3] = {g.x, g.y, g.z};
  dims(6, "Grid Max Size per Dimension:", g32);
  value(6, "FBarrier Max Size:", dec(isa<uint32_t>(i, HSA_ISA_INFO_FBARRIER_MAX_SIZE)));
}

void print_agent(hsa_agent_t a, int n) {
  label(0, "*******");
  label(0, ("Agent " + std::to_string(n)).c_str());
  label(0, "*******");
  const bool gpu = agent<uint32_t>(a, HSA_AGENT_INFO_DEVICE) == HSA_DEVICE_TYPE_GPU;
  value(2, "Name:", agent_string(a, HSA_AGENT_INFO_NAME));
  value(2, "Uuid:", agent_string(a, HSA_AMD_AGENT_INFO_UUID));
  value(2, "Marketing Name:", agent_string(a, HSA_AMD_AGENT_INFO_PRODUCT_NAME));
  value(2, "Vendor Name:", agent_string(a, HSA_AGENT_INFO_VENDOR_NAME));
  const uint32_t feature = agent<uint32_t>(a, HSA_AGENT_INFO_FEATURE);
  std::string features;
  if (feature & HSA_AGENT_FEATURE_KERNEL_DISPATCH) features += "KERNEL_DISPATCH ";
  if (feature & HSA_AGENT_FEATURE_AGENT_DISPATCH) features += "AGENT_DISPATCH ";
  value(2, "Feature:", features.empty() ? "None specified" : features.substr(0, features.size() - 1));
  value(2, "Profile:", agent<uint32_t>(a, HSA_AGENT_INFO_PROFILE) == HSA_PROFILE_FULL ? "FULL_PROFILE" : "BASE_PROFILE");
  value(2, "Float Round Mode:", rounding(agent<uint32_t>(a, HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE)));
  value(2, "Max Queue Number:", num(agent<uint32_t>(a, HSA_AGENT_INFO_QUEUES_MAX)));
  value(2, "Queue Min Size:", num(agent<uint32_t>(a, HSA_AGENT_INFO_QUEUE_MIN_SIZE)));
  value(2, "Queue Max Size:", num(agent<uint32_t>(a, HSA_AGENT_INFO_QUEUE_MAX_SIZE)));
  value(2, "Queue Type:", agent<uint32_t>(a, HSA_AGENT_INFO_QUEUE_TYPE) == HSA_QUEUE_TYPE_MULTI ? "MULTI" : "SINGLE");
  value(2, "Node:", dec(agent<uint32_t>(a, HSA_AGENT_INFO_NODE)));
  value(2, "Device Type:", gpu ? "GPU" : "CPU");
  label(2, "Cache Info:");
  uint32_t cache[4] = {};
  hsa_agent_get_info(a, HSA_AGENT_INFO_CACHE_SIZE, cache);
  for (int k = 0; k < 4; ++k)
    if (cache[k]) value(4, ("L" + std::to_string(k + 1) + ":").c_str(), num(cache[k] / 1024) + " KB");
  value(2, "Chip ID:", num(agent<uint32_t>(a, HSA_AMD_AGENT_INFO_CHIP_ID)));
  value(2, "Cacheline Size:", num(agent<uint32_t>(a, HSA_AMD_AGENT_INFO_CACHELINE_SIZE)));
  if (const uint32_t mhz = agent<uint32_t>(a, HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY)) value(2, "Max Clock Freq. (MHz):", dec(mhz));
  value(2, "BDFID:", dec(agent<uint32_t>(a, HSA_AMD_AGENT_INFO_BDFID)));
  value(2, "Internal Node ID:", dec(agent<uint32_t>(a, HSA_AMD_AGENT_INFO_DRIVER_NODE_ID)));
  value(2, "Compute Unit:", dec(agent<uint32_t>(a, HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT)));
  value(2, "SIMDs per CU:", dec(agent<uint32_t>(a, HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU)));
  value(2, "Shader Engines:", dec(agent<uint32_t>(a, HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES)));
  value(2, "Shader Arrs. per Eng.:", dec(agent<uint32_t>(a, HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE)));
  if (gpu) value(2, "Coherent Host Access:", yes(agent<bool>(a, HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS)));
  label(2, "Memory Properties:");
  raw(2, "Features:", features.empty() ? "None" : features);
  if (gpu) {
    value(2, "Fast F16 Operation:", yes(agent<bool>(a, HSA_AGENT_INFO_FAST_F16_OPERATION)));
    const uint32_t wave = agent<uint32_t>(a, HSA_AGENT_INFO_WAVEFRONT_SIZE);
    value(2, "Wavefront Size:", num(wave));
    value(2, "Workgroup Max Size:", num(agent<uint32_t>(a, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE)));
    uint16_t wg[3] = {};
    hsa_agent_get_info(a, HSA_AGENT_INFO_WORKGROUP_MAX_DIM, wg);
    const uint32_t wg32[3] = {wg[0], wg[1], wg[2]};
    dims(2, "Workgroup Max Size per Dimension:", wg32);
    const uint32_t waves = agent<uint32_t>(a, HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU);
    value(2, "Max Waves Per CU:", num(waves));
    value(2, "Max Work-item Per CU:", num(uint64_t{waves} * wave));
    value(2, "Grid Max Size:", num(agent<uint32_t>(a, HSA_AGENT_INFO_GRID_MAX_SIZE)));
    const hsa_dim3_t g = agent<hsa_dim3_t>(a, HSA_AGENT_INFO_GRID_MAX_DIM);
    const uint32_t g32[3] = {g.x, g.y, g.z};
    dims(2, "Grid Max Size per Dimension:", g32);
    value(2, "Max fbarriers/Workgrp:", dec(agent<uint32_t>(a, HSA_AGENT_INFO_FBARRIER_MAX_SIZE)));
    value(2, "Packet Processor uCode::", dec(agent<uint32_t>(a, HSA_AMD_AGENT_INFO_UCODE_VERSION)));
    value(2, "SDMA engine uCode::", dec(agent<uint32_t>(a, HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION)));
    value(2, "IOMMU Support::", agent<uint32_t>(a, HSA_AMD_AGENT_INFO_IOMMU_SUPPORT) ? "IOMMU_V2" : "None");
  }
  label(2, "Pool Info:");
  std::vector<hsa_amd_memory_pool_t> pools;
  hsa_amd_agent_iterate_memory_pools(
      a,
      [](hsa_amd_memory_pool_t p, void* d) {
        static_cast<std::vector<hsa_amd_memory_pool_t>*>(d)->push_back(p);
        return HSA_STATUS_SUCCESS;
      },
      &pools);
  for (size_t k = 0; k < pools.size(); ++k) print_pool(pools[k], static_cast<int>(k + 1));
  label(2, "ISA Info:");
  std::vector<hsa_isa_t> isas;
  hsa_agent_iterate_isas(
      a,
      [](hsa_isa_t i, void* d) {
        static_cast<std::vector<hsa_isa_t>*>(d)->push_back(i);
        return HSA_STATUS_SUCCESS;
      },
      &isas);
  for (size_t k = 0; k < isas.size(); ++k) print_isa(isas[k], static_cast<int>(k + 1));
}

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      std::printf("Usage: rocminfo\n  Prints the HSA system attributes and every agent (VirtualGPU's rocminfo).\n");
      return 0;
    }
  }
  // What rocminfo says of the kernel driver first. The simulated machine has
  // no amdgpu module version to read, and rocminfo says so this way when a
  // module does not give one.
  std::string version;
  if (std::ifstream v("/sys/module/amdgpu/version"); v) std::getline(v, version);
  if (!version.empty()) std::printf("ROCk module version %s is loaded\n", version.c_str());
  else std::printf("ROCk module is loaded\n");
  if (hsa_init() != HSA_STATUS_SUCCESS) {
    std::printf("hsa api call failure at: rocminfo\nCall returned HSA_STATUS_ERROR_OUT_OF_RESOURCES\n");
    return 1;
  }
  label(0, "=====================");
  label(0, "HSA System Attributes");
  label(0, "=====================");
  uint16_t major = 0, minor = 0, ext_major = 0, ext_minor = 0;
  hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &major);
  hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MINOR, &minor);
  hsa_system_get_info(HSA_AMD_SYSTEM_INFO_EXT_VERSION_MAJOR, &ext_major);
  hsa_system_get_info(HSA_AMD_SYSTEM_INFO_EXT_VERSION_MINOR, &ext_minor);
  raw(0, "Runtime Version:", std::to_string(major) + "." + std::to_string(minor));
  raw(0, "Runtime Ext Version:", std::to_string(ext_major) + "." + std::to_string(ext_minor));
  uint64_t freq = 0, max_wait = 0;
  hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq);
  hsa_system_get_info(HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT, &max_wait);
  char b[128];
  std::snprintf(b, sizeof b, "%fMHz", static_cast<double>(freq) / 1e6);
  raw(0, "System Timestamp Freq.:", b);
  std::snprintf(b, sizeof b, "%" PRIu64 " (0x%" PRIX64 ") (timestamp count)", max_wait, max_wait);
  raw(0, "Sig. Max Wait Duration:", b);
  uint32_t model = 0, endian = 0;
  hsa_system_get_info(HSA_SYSTEM_INFO_MACHINE_MODEL, &model);
  hsa_system_get_info(HSA_SYSTEM_INFO_ENDIANNESS, &endian);
  value(0, "Machine Model:", model == HSA_MACHINE_MODEL_LARGE ? "LARGE" : "SMALL");
  value(0, "System Endianness:", endian == HSA_ENDIANNESS_LITTLE ? "LITTLE" : "BIG");
  const auto flag = [](hsa_system_info_t attr) {
    bool v = false;
    hsa_system_get_info(attr, &v);
    return v;
  };
  raw(0, "Mwaitx:", flag(HSA_AMD_SYSTEM_INFO_MWAITX_ENABLED) ? "ENABLED" : "DISABLED");
  raw(0, "XNACK enabled:", flag(HSA_AMD_SYSTEM_INFO_XNACK_ENABLED) ? "YES" : "NO");
  raw(0, "DMAbuf Support:", flag(HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED) ? "YES" : "NO");
  raw(0, "VMM Support:", flag(HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED) ? "YES" : "NO");
  std::printf("\n");
  label(0, "==========");
  label(0, "HSA Agents");
  label(0, "==========");
  std::vector<hsa_agent_t> agents;
  hsa_iterate_agents(
      [](hsa_agent_t a, void* d) {
        static_cast<std::vector<hsa_agent_t>*>(d)->push_back(a);
        return HSA_STATUS_SUCCESS;
      },
      &agents);
  for (size_t k = 0; k < agents.size(); ++k) print_agent(agents[k], static_cast<int>(k + 1));
  label(0, "*** Done ***");
  hsa_shut_down();
  return 0;
}
