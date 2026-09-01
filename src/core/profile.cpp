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

  std::string cc = get_str(doc, "compute_capability", origin);
  size_t dot = cc.find('.');
  if (dot == std::string::npos) fail(origin, "compute_capability must look like \"9.0\"");
  try {
    p.cc_major = std::stoi(cc.substr(0, dot));
    p.cc_minor = std::stoi(cc.substr(dot + 1));
  } catch (const std::exception&) {
    fail(origin, "compute_capability must look like \"9.0\", got: " + cc);
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
  p.limits.registers_per_block = static_cast<uint32_t>(get_int(lim, "registers_per_block", origin));
  p.limits.multiprocessors = static_cast<uint32_t>(get_int(lim, "multiprocessors", origin));

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
