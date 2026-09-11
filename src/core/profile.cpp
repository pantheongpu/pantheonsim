#include "vgpu/profile.hpp"

#include "vgpu/error.hpp"
#include "vgpu/yamlish.hpp"

namespace vgpu {
namespace {

using yamlish::Value;

[[noreturn]] void fail(const std::string& origin, const std::string& msg) {
  throw Error::make(Err::ProfileParse, origin, ": ", msg);
}

const Value& require(const Value& map, const char* key, const std::string& origin) {
  auto it = map.map.find(key);
  if (it == map.map.end()) fail(origin, std::string("missing required key: ") + key);
  return it->second;
}

std::string get_str(const Value& map, const char* key, const std::string& origin) {
  const Value& v = require(map, key, origin);
  if (v.kind != Value::Kind::Str) fail(origin, "key '" + std::string(key) + "' must be a string");
  return v.str;
}

// A string key that need not be present. Returns "" when it is absent.
std::string get_str_opt(const Value& map, const char* key) {
  auto it = map.map.find(key);
  if (it == map.map.end() || it->second.kind != Value::Kind::Str) return "";
  return it->second.str;
}

// Telemetry values are presentation-only, and a driver that does not report one
// is a fact about the device rather than a broken profile: a real GH200 reports
// no thermal threshold, and requiring the key made every profile characterized
// from one unloadable. Absent means zero, which callers render as "unknown".
int64_t opt_int(const Value& map, const char* key, const std::string& origin, int64_t dflt) {
  auto it = map.map.find(key);
  if (it == map.map.end()) return dflt;
  if (it->second.kind != Value::Kind::Int)
    fail(origin, "key '" + std::string(key) + "' must be an integer");
  return it->second.i;
}

int64_t get_int(const Value& map, const char* key, const std::string& origin) {
  const Value& v = require(map, key, origin);
  if (v.kind != Value::Kind::Int) fail(origin, "key '" + std::string(key) + "' must be an integer");
  return v.i;
}

bool get_bool(const Value& map, const char* key, const std::string& origin) {
  const Value& v = require(map, key, origin);
  if (v.kind != Value::Kind::Bool) fail(origin, "key '" + std::string(key) + "' must be true/false");
  return v.b;
}

std::array<uint32_t, 3> get_dim3(const Value& map, const char* key, const std::string& origin) {
  const Value& v = require(map, key, origin);
  if (v.kind != Value::Kind::List || v.list.size() != 3) fail(origin, "key '" + std::string(key) + "' must be [x, y, z]");
  std::array<uint32_t, 3> out{};
  for (int i = 0; i < 3; ++i) {
    if (v.list[i].kind != Value::Kind::Int || v.list[i].i < 0)
      fail(origin, "key '" + std::string(key) + "' must contain non-negative integers");
    out[i] = static_cast<uint32_t>(v.list[i].i);
  }
  return out;
}

}  // namespace

DeviceProfile DeviceProfile::from_yaml(const std::string& src, const std::string& origin) {
  Value doc = yamlish::parse(src, origin);
  DeviceProfile p;
  p.id = get_str(doc, "id", origin);
  p.vendor = get_str(doc, "vendor", origin);
  p.model = get_str(doc, "model", origin);
  p.architecture = get_str(doc, "architecture", origin);
  p.verified = get_bool(doc, "verified", origin);

  // A compute capability is an NVIDIA concept. AMD parts do not have one, and
  // an AMD profile that carried a plausible-looking number -- "9.4" derived
  // from gfx942 -- would be exactly the invented value this project keeps
  // finding and removing. What identifies an AMD device is its gfx target, so
  // that is what is required instead, and each vendor is asked only for the
  // thing it actually has.
  if (p.vendor == "amd") {
    p.gcn_arch = get_str_opt(doc, "gcn_arch");
    if (p.gcn_arch.empty())
      fail(origin, "an AMD profile needs gcn_arch (the gfx target, e.g. \"gfx942\")");
  } else {
    const std::string cc = get_str(doc, "compute_capability", origin);
    const size_t dot = cc.find('.');
    if (dot == std::string::npos) fail(origin, "compute_capability must look like \"9.0\"");
    try {
      p.cc_major = std::stoi(cc.substr(0, dot));
      p.cc_minor = std::stoi(cc.substr(dot + 1));
    } catch (const std::exception&) {
      fail(origin, "compute_capability must look like \"9.0\", got: " + cc);
    }
  }

  int64_t ws = get_int(doc, "warp_size", origin);
  if (ws <= 0) fail(origin, "warp_size must be positive");
  p.warp_size = static_cast<uint32_t>(ws);

  int64_t vram = get_int(doc, "vram_bytes", origin);
  if (vram <= 0) fail(origin, "vram_bytes must be positive");
  p.vram_bytes = static_cast<uint64_t>(vram);

  const Value& lim = require(doc, "limits", origin);
  if (lim.kind != Value::Kind::Map) fail(origin, "'limits' must be a map");
  p.limits.max_threads_per_block = static_cast<uint32_t>(get_int(lim, "max_threads_per_block", origin));
  p.limits.max_block_dim = get_dim3(lim, "max_block_dim", origin);
  p.limits.max_grid_dim = get_dim3(lim, "max_grid_dim", origin);
  p.limits.shared_mem_per_block = static_cast<uint32_t>(get_int(lim, "shared_mem_per_block_bytes", origin));
  p.limits.shared_mem_per_block_optin =
      static_cast<uint32_t>(get_int(lim, "shared_mem_per_block_optin_bytes", origin));
  // rocminfo does not report a register-file size, and inventing one would put
  // a fabricated number straight into the occupancy model -- where it would
  // look measured. Zero means "the device did not say", and an AMD profile
  // therefore cannot drive occupancy yet, which is true and visible rather
  // than plausible and wrong.
  p.limits.registers_per_block =
      p.vendor == "amd"
          ? static_cast<uint32_t>(opt_int(lim, "registers_per_block", origin, 0))
          : static_cast<uint32_t>(get_int(lim, "registers_per_block", origin));
  p.limits.multiprocessors = static_cast<uint32_t>(get_int(lim, "multiprocessors", origin));
  // Optional residency ceilings; older profiles without them fall back to
  // values derived from the fields that are present.
  auto opt = [&](const char* key, uint32_t fallback) -> uint32_t {
    auto it = lim.map.find(key);
    if (it == lim.map.end() || it->second.kind != Value::Kind::Int) return fallback;
    return static_cast<uint32_t>(it->second.i);
  };
  p.limits.registers_per_sm = opt("registers_per_sm", p.limits.registers_per_block);
  p.limits.max_threads_per_sm = opt("max_threads_per_sm", 2048);
  p.limits.max_blocks_per_sm = opt("max_blocks_per_sm", 16);
  p.limits.max_registers_per_thread = opt("max_registers_per_thread", 255);
  p.limits.shared_mem_per_sm = opt("shared_mem_per_sm", p.limits.shared_mem_per_block_optin);

  // Optional: presentation-only values for monitoring tools.
  if (auto it = doc.map.find("telemetry"); it != doc.map.end()) {
    if (it->second.kind != Value::Kind::Map) fail(origin, "'telemetry' must be a map");
    const Value& t = it->second;
    p.telemetry.power_limit_w = static_cast<uint32_t>(opt_int(t, "power_limit_w", origin, 0));
    p.telemetry.sm_clock_max_mhz = static_cast<uint32_t>(opt_int(t, "sm_clock_max_mhz", origin, 0));
    p.telemetry.mem_clock_max_mhz = static_cast<uint32_t>(opt_int(t, "mem_clock_max_mhz", origin, 0));
    p.telemetry.temperature_max_c = static_cast<uint32_t>(opt_int(t, "temperature_max_c", origin, 0));
    p.telemetry.pci_vendor_id = static_cast<uint32_t>(opt_int(t, "pci_vendor_id", origin, 0));
    p.telemetry.pci_device_id = static_cast<uint32_t>(opt_int(t, "pci_device_id", origin, 0));
    // framebuffer_mb is recorded as measured -- the "Memory-Usage" total from
    // nvidia-smi on the real card -- and turned into the driver's reserve here.
    // It can be absent (not yet measured) but it cannot be smaller than the
    // memory CUDA reports: that is a profile describing an impossible card.
    if (const int64_t fb = opt_int(t, "framebuffer_mb", origin, 0); fb > 0) {
      const uint64_t fb_bytes = static_cast<uint64_t>(fb) * 1024ull * 1024ull;
      if (fb_bytes < p.vram_bytes)
        fail(origin, "telemetry.framebuffer_mb (" + std::to_string(fb) +
                         " MiB) is smaller than vram_bytes; the framebuffer includes it");
      p.telemetry.framebuffer_reserve_bytes = fb_bytes - p.vram_bytes;
    }
  }

  if (auto it = doc.map.find("features"); it != doc.map.end()) {
    if (it->second.kind != Value::Kind::Map) fail(origin, "'features' must be a map");
    for (auto& [k, v] : it->second.map) {
      if (v.kind != Value::Kind::Bool) fail(origin, "feature '" + k + "' must be true/false");
      p.features[k] = v.b;
    }
  }
  return p;
}

}  // namespace vgpu
