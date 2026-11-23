#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
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

// Whether the document has the key a provenance list or an evidence entry names: "section.key" for an entry of a section,
// a bare name for a top-level scalar.
bool has_key(const Value& doc, const std::string& key) {
  const size_t dot = key.find('.');
  if (dot == std::string::npos) {
    auto it = doc.map.find(key);
    return it != doc.map.end() && it->second.kind != Value::Kind::Map;
  }
  const std::string section = key.substr(0, dot);
  if (section == "provenance" || section == "evidence") return false;
  auto it = doc.map.find(section);
  return it != doc.map.end() && it->second.kind == Value::Kind::Map && it->second.map.count(key.substr(dot + 1)) != 0;
}

bool parse_tensor_sum(const std::string& s, TensorSum& out) {
  for (TensorSum m : {TensorSum::Sequential, TensorSum::Ampere, TensorSum::Hopper})
    if (s == tensor_sum_name(m)) {
      out = m;
      return true;
    }
  return false;
}

// The `numerics:` section (Numerics in profile.hpp).
void parse_numerics(const Value& doc, DeviceProfile& p, const std::string& origin) {
  auto it = doc.map.find("numerics");
  if (it == doc.map.end()) return;
  const Value& n = it->second;
  if (n.kind != Value::Kind::Map) fail(origin, "'numerics' must be a map");
  for (const auto& [key, value] : n.map) {
    (void)value;
    if (key != "tensor_sum" && key != "tensor_sum_chunk" && key != "tensor_sum_chunk_tf32" && key != "tensor_sum_grid" &&
        key != "tensor_sum_published" && key != "tensor_sum_fp8" && key != "tensor_sum_fp8_chunk" && key != "tensor_sum_fp8_grid" &&
        key != "tensor_sum_fp8_out_bits" && key != "tensor_sum_wgmma_fp8" && key != "tensor_sum_wgmma_fp8_chunk" &&
        key != "tensor_sum_wgmma_fp8_grid" && key != "tensor_sum_wgmma_fp8_out_bits" && key != "tensor_sum_wgmma" &&
        key != "tensor_sum_tcgen05" && key != "tensor_sum_m8n8k4")
      fail(origin, "numerics: unknown key '" + key + "'");
  }
  // The tensor-core sum is read by the NVIDIA engines only. On another vendor's profile it would be a property nothing consults.
  if (p.vendor != "nvidia")
    fail(origin, "'numerics' holds NVIDIA tensor-core properties only; an " + p.vendor + " profile cannot carry it");
  p.numerics.present = true;
  if (!parse_tensor_sum(get_str(n, "tensor_sum", origin), p.numerics.tensor_sum))
    fail(origin, "numerics.tensor_sum must be sequential, ampere or hopper, got: " + get_str(n, "tensor_sum", origin));
  const TensorSumRule* rule = tensor_sum_rule(p.numerics.tensor_sum);
  if (!rule) {
    for (const char* key : {"tensor_sum_chunk", "tensor_sum_chunk_tf32", "tensor_sum_grid"})
      if (n.map.count(key)) fail(origin, std::string("numerics.") + key + " has no meaning with tensor_sum: sequential");
  } else {
    const int64_t chunk = get_int(n, "tensor_sum_chunk", origin), chunk_tf32 = get_int(n, "tensor_sum_chunk_tf32", origin),
                  grid = get_int(n, "tensor_sum_grid", origin);
    // The engines implement the two rules of profile.hpp, not a sum of any chunk and grid: numbers that are not the named
    // model's would be printed and hashed and would not be what runs.
    if (chunk != rule->chunk || chunk_tf32 != rule->chunk_tf32 || grid != rule->grid)
      fail(origin, std::string("numerics: tensor_sum ") + tensor_sum_name(p.numerics.tensor_sum) + " is chunks of " +
                       std::to_string(rule->chunk) + " (tf32 " + std::to_string(rule->chunk_tf32) + "), grid " +
                       std::to_string(rule->grid) + "; the profile says " + std::to_string(chunk) + ", " +
                       std::to_string(chunk_tf32) + ", " + std::to_string(grid) + ", and the engines have no such sum");
    p.numerics.tensor_sum_chunk = static_cast<uint32_t>(chunk);
    p.numerics.tensor_sum_chunk_tf32 = static_cast<uint32_t>(chunk_tf32);
    p.numerics.tensor_sum_grid = static_cast<int32_t>(grid);
  }
  if (n.map.count("tensor_sum_published")) {
    const std::string pub = get_str(n, "tensor_sum_published", origin);
    if (!parse_tensor_sum(pub, p.numerics.tensor_sum_published) || p.numerics.tensor_sum_published == TensorSum::Sequential)
      fail(origin, "numerics.tensor_sum_published must be ampere or hopper, got: " + pub);
    if (rule) fail(origin, "numerics.tensor_sum_published is for a device whose own sum is not known: it needs tensor_sum: sequential");
  }
  // The FP8 sum: optional, and when named, written out with its numbers as the f16 sum is (via_f16 has none: it is the f16 rule's).
  const bool fp8_numbers = n.map.count("tensor_sum_fp8_chunk") || n.map.count("tensor_sum_fp8_grid") || n.map.count("tensor_sum_fp8_out_bits");
  if (n.map.count("tensor_sum_fp8")) {
    const std::string name = get_str(n, "tensor_sum_fp8", origin);
    bool known = false;
    for (Fp8Sum m : {Fp8Sum::Sequential, Fp8Sum::Ada, Fp8Sum::ViaF16})
      if (name == fp8_sum_name(m)) {
        p.numerics.tensor_sum_fp8 = m;
        known = true;
      }
    if (!known) fail(origin, "numerics.tensor_sum_fp8 must be sequential, ada or via_f16, got: " + name);
    const Fp8SumRule* r8 = fp8_sum_rule(p.numerics.tensor_sum_fp8);
    if (!r8) {
      if (fp8_numbers) fail(origin, "numerics.tensor_sum_fp8_* have no meaning with tensor_sum_fp8: " + name);
      if (p.numerics.tensor_sum_fp8 == Fp8Sum::ViaF16 && !tensor_sum_rule(p.numerics.tensor_sum))
        fail(origin, "numerics.tensor_sum_fp8: via_f16 sums as the f16 rule does, so it needs a tensor_sum that is not sequential");
    } else {
      const int64_t chunk = get_int(n, "tensor_sum_fp8_chunk", origin), grid = get_int(n, "tensor_sum_fp8_grid", origin),
                    out_bits = get_int(n, "tensor_sum_fp8_out_bits", origin);
      if (chunk != r8->chunk || grid != r8->grid || out_bits != r8->out_bits)
        fail(origin, "numerics: tensor_sum_fp8 " + name + " is chunks of " + std::to_string(r8->chunk) + ", grid " + std::to_string(r8->grid) +
                         ", " + std::to_string(r8->out_bits) + " result bits; the profile says " + std::to_string(chunk) + ", " +
                         std::to_string(grid) + ", " + std::to_string(out_bits) + ", and the engines have no such sum");
      p.numerics.tensor_sum_fp8_chunk = static_cast<uint32_t>(chunk);
      p.numerics.tensor_sum_fp8_grid = static_cast<int32_t>(grid);
      p.numerics.tensor_sum_fp8_out_bits = static_cast<int32_t>(out_bits);
    }
  } else if (fp8_numbers) {
    fail(origin, "numerics.tensor_sum_fp8_* need numerics.tensor_sum_fp8");
  }
  // The FP8 sum of wgmma: optional; named, it is written out with its numbers.
  const bool wg_numbers = n.map.count("tensor_sum_wgmma_fp8_chunk") || n.map.count("tensor_sum_wgmma_fp8_grid") ||
                          n.map.count("tensor_sum_wgmma_fp8_out_bits");
  if (n.map.count("tensor_sum_wgmma_fp8")) {
    const std::string name = get_str(n, "tensor_sum_wgmma_fp8", origin);
    bool known = false;
    for (WgmmaFp8Sum m : {WgmmaFp8Sum::Sequential, WgmmaFp8Sum::Hopper})
      if (name == wgmma_fp8_sum_name(m)) {
        p.numerics.tensor_sum_wgmma_fp8 = m;
        known = true;
      }
    if (!known) fail(origin, "numerics.tensor_sum_wgmma_fp8 must be sequential or hopper, got: " + name);
    const Fp8SumRule* rw = wgmma_fp8_sum_rule(p.numerics.tensor_sum_wgmma_fp8);
    if (!rw) {
      if (wg_numbers) fail(origin, "numerics.tensor_sum_wgmma_fp8_* have no meaning with tensor_sum_wgmma_fp8: " + name);
    } else {
      const int64_t chunk = get_int(n, "tensor_sum_wgmma_fp8_chunk", origin), grid = get_int(n, "tensor_sum_wgmma_fp8_grid", origin),
                    out_bits = get_int(n, "tensor_sum_wgmma_fp8_out_bits", origin);
      if (chunk != rw->chunk || grid != rw->grid || out_bits != rw->out_bits)
        fail(origin, "numerics: tensor_sum_wgmma_fp8 " + name + " is chunks of " + std::to_string(rw->chunk) + ", grid " + std::to_string(rw->grid) +
                         ", " + std::to_string(rw->out_bits) + " result bits; the profile says " + std::to_string(chunk) + ", " +
                         std::to_string(grid) + ", " + std::to_string(out_bits) + ", and the engines have no such sum");
      p.numerics.tensor_sum_wgmma_fp8_chunk = static_cast<uint32_t>(chunk);
      p.numerics.tensor_sum_wgmma_fp8_grid = static_cast<int32_t>(grid);
      p.numerics.tensor_sum_wgmma_fp8_out_bits = static_cast<int32_t>(out_bits);
    }
  } else if (wg_numbers) {
    fail(origin, "numerics.tensor_sum_wgmma_fp8_* need numerics.tensor_sum_wgmma_fp8");
  }
  // The sums of wgmma, tcgen05.mma and m8n8k4 on f16, bf16 and tf32 operands: optional; sequential, or the one rule the form has.
  const auto form_sum = [&](const char* key, FormSum named, FormSum& out) {
    if (!n.map.count(key)) return;
    const std::string name = get_str(n, key, origin);
    if (name == form_sum_name(FormSum::Sequential)) out = FormSum::Sequential;
    else if (name == form_sum_name(named)) out = named;
    else fail(origin, std::string("numerics.") + key + " must be sequential or " + form_sum_name(named) + ", got: " + name);
  };
  form_sum("tensor_sum_wgmma", FormSum::Hopper, p.numerics.tensor_sum_wgmma);
  form_sum("tensor_sum_tcgen05", FormSum::Hopper, p.numerics.tensor_sum_tcgen05);
  form_sum("tensor_sum_m8n8k4", FormSum::Volta, p.numerics.tensor_sum_m8n8k4);
}

// The `provenance:` and `evidence:` sections (Provenance in profile.hpp), and the rule that every numerics key is tagged.
void parse_provenance(const Value& doc, DeviceProfile& p, const std::string& origin) {
  if (auto it = doc.map.find("provenance"); it != doc.map.end()) {
    if (it->second.kind != Value::Kind::Map) fail(origin, "'provenance' must be a map of class: [keys]");
    for (const auto& [tag, keys] : it->second.map) {
      Provenance cls = Provenance::Unknown;
      if (tag == "measured") cls = Provenance::Measured;
      else if (tag == "documented") cls = Provenance::Documented;
      else if (tag == "inferred") cls = Provenance::Inferred;
      else if (tag == "unknown") cls = Provenance::Unknown;
      else fail(origin, "provenance: unknown class '" + tag + "' (the classes are measured, documented, inferred, unknown)");
      if (keys.kind != Value::Kind::List) fail(origin, "provenance." + tag + " must be a list of keys");
      for (const Value& e : keys.list) {
        if (e.kind != Value::Kind::Str) fail(origin, "provenance." + tag + " must be a list of keys");
        if (!has_key(doc, e.str)) fail(origin, "provenance." + tag + " names '" + e.str + "', a key this profile does not have");
        if (auto had = p.provenance.find(e.str); had != p.provenance.end())
          fail(origin, "provenance: '" + e.str + "' is tagged twice (" + provenance_name(had->second) + " and " + tag +
                           "); a key carries one class");
        if (cls != Provenance::Unknown && p.is_placeholder(e.str))
          fail(origin, "provenance: '" + e.str + "' is in placeholders, so it can only be unknown, not " + tag);
        p.provenance.emplace(e.str, cls);
      }
    }
  }
  if (auto it = doc.map.find("evidence"); it != doc.map.end()) {
    if (it->second.kind != Value::Kind::Map) fail(origin, "'evidence' must be a map of key: \"text\"");
    for (const auto& [key, text] : it->second.map) {
      if (text.kind != Value::Kind::Str || text.str.empty()) fail(origin, "evidence." + key + " must be a text");
      // An entry is for a tagged key, or for a section ("numerics") at least one key of which is tagged.
      bool used = p.provenance.count(key) != 0;
      if (!used && key.find('.') == std::string::npos)
        for (const auto& [tagged, cls] : p.provenance) {
          (void)cls;
          if (tagged.compare(0, key.size() + 1, key + ".") == 0) used = true;
        }
      if (!used) fail(origin, "evidence: '" + key + "' is neither a key that provenance tags nor a section with one");
      p.evidence.emplace(key, text.str);
    }
  }
  // A class other than unknown is a claim, and a claim says what it rests on.
  for (const auto& [key, cls] : p.provenance)
    if (cls != Provenance::Unknown && p.evidence_of(key).empty())
      fail(origin, "provenance: '" + key + "' is " + provenance_name(cls) + " and has no evidence entry (its own or its section's)");
  // Numerics decide result bits, so none goes without a class.
  if (auto it = doc.map.find("numerics"); it != doc.map.end() && it->second.kind == Value::Kind::Map)
    for (const auto& [key, value] : it->second.map) {
      (void)value;
      if (!p.provenance.count("numerics." + key))
        fail(origin, "numerics." + key + " has no provenance class: every numerics key must be listed under provenance " +
                         "(measured, documented, inferred or unknown)");
    }
}

}  // namespace

std::optional<Provenance> DeviceProfile::provenance_of(const std::string& key) const {
  if (auto it = provenance.find(key); it != provenance.end()) return it->second;
  if (is_placeholder(key)) return Provenance::Unknown;
  return std::nullopt;
}

const std::string& DeviceProfile::evidence_of(const std::string& key) const {
  static const std::string none;
  if (auto it = evidence.find(key); it != evidence.end()) return it->second;
  if (const auto dot = key.find('.'); dot != std::string::npos)
    if (auto it = evidence.find(key.substr(0, dot)); it != evidence.end()) return it->second;
  return none;
}

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
    p.gcn_arch_full = get_str_opt(doc, "gcn_arch_full");
    if (p.gcn_arch_full.empty()) p.gcn_arch_full = p.gcn_arch;
    if (p.gcn_arch_full.compare(0, p.gcn_arch.size(), p.gcn_arch) != 0 ||
        (p.gcn_arch_full.size() > p.gcn_arch.size() && p.gcn_arch_full[p.gcn_arch.size()] != ':'))
      fail(origin, "gcn_arch_full must be gcn_arch with its features after it, e.g. \"gfx942:sramecc+:xnack-\"");
  } else if (p.vendor == "intel") {
    // An Xe part has neither a compute capability nor a gfx target; what identifies it is its architecture
    // name (required above) and the Xe-core layout below. A cc_major of 0 keeps every NVIDIA-only rule off.
    if (p.architecture.empty()) fail(origin, "an Intel profile needs architecture (e.g. \"xe2-hpg\")");
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

  // Intel: the Xe core's make-up, from which the residency ceilings below are derived.
  if (p.vendor == "intel") {
    const Value& xe = require(doc, "xe", origin);
    if (xe.kind != Value::Kind::Map) fail(origin, "'xe' must be a map");
    p.xe.eus_per_xe_core = static_cast<uint32_t>(get_int(xe, "eus_per_xe_core", origin));
    p.xe.threads_per_eu = static_cast<uint32_t>(get_int(xe, "threads_per_eu", origin));
    p.xe.grf_per_thread = static_cast<uint32_t>(get_int(xe, "grf_per_thread", origin));
    p.xe.grf_bytes = static_cast<uint32_t>(get_int(xe, "grf_bytes", origin));
    const Value& sg = require(xe, "sub_group_sizes", origin);
    if (sg.kind != Value::Kind::List || sg.list.empty())
      fail(origin, "xe.sub_group_sizes must be a non-empty list such as [16, 32]");
    for (const Value& e : sg.list) {
      if (e.kind != Value::Kind::Int || (e.i != 8 && e.i != 16 && e.i != 32))
        fail(origin, "xe.sub_group_sizes may contain only 8, 16 and 32");
      p.xe.sub_group_sizes.push_back(static_cast<uint32_t>(e.i));
    }
    if (!p.xe.known()) fail(origin, "xe: every count must be positive");
    if (!p.xe.has_sub_group_size(static_cast<uint32_t>(ws)))
      fail(origin, "warp_size " + std::to_string(ws) + " is not one of xe.sub_group_sizes");
  }
  if (auto it = doc.map.find("placeholders"); it != doc.map.end()) {
    if (it->second.kind != Value::Kind::List) fail(origin, "'placeholders' must be a list of keys");
    for (const Value& e : it->second.list) {
      if (e.kind != Value::Kind::Str) fail(origin, "'placeholders' must be a list of keys");
      p.placeholders.push_back(e.str);
    }
  }

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
      p.vendor == "amd" || p.vendor == "intel"
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
  // An Xe core's ceilings follow from what it is made of unless the profile says otherwise: a work-group lives
  // in one Xe core, so its register budget is the core's register file; a hardware thread holds one sub-group;
  // and a thread's registers are its GRFs shared out over the lanes of the profile's default sub-group.
  uint32_t d_threads = 2048, d_blocks = 16, d_regs = 255;
  if (p.vendor == "intel") {
    const uint64_t words = p.xe.register_words_per_xe_core();
    if (!p.limits.registers_per_block) p.limits.registers_per_block = static_cast<uint32_t>(words);
    d_threads = p.xe.threads_per_xe_core() * p.warp_size;                       // one sub-group a hardware thread, at the default width
    d_blocks = p.xe.threads_per_xe_core();                                  // a work-group takes at least one thread
    d_regs = p.xe.grf_per_thread * p.xe.grf_bytes / 4 / p.warp_size;
  }
  p.limits.registers_per_sm = opt("registers_per_sm", p.limits.registers_per_block);
  p.limits.max_threads_per_sm = opt("max_threads_per_sm", d_threads);
  p.limits.max_blocks_per_sm = opt("max_blocks_per_sm", d_blocks);
  p.limits.max_registers_per_thread = opt("max_registers_per_thread", d_regs);
  p.limits.shared_mem_per_sm = opt("shared_mem_per_sm", p.limits.shared_mem_per_block_optin);
  if (auto it = lim.map.find("l2_cache_bytes"); it != lim.map.end() && it->second.kind == Value::Kind::Int)
    p.limits.l2_cache_bytes = static_cast<uint64_t>(it->second.i);

  // Optional: the facts of the card CUDA reports beyond its limits.
  if (auto it = doc.map.find("cuda"); it != doc.map.end()) {
    if (it->second.kind != Value::Kind::Map) fail(origin, "'cuda' must be a map");
    const Value& c = it->second;
    p.cuda.memory_bus_width_bits = static_cast<uint32_t>(opt_int(c, "memory_bus_width_bits", origin, 0));
    p.cuda.clock_khz = static_cast<uint32_t>(opt_int(c, "clock_khz", origin, 0));
    p.cuda.async_engine_count = static_cast<uint32_t>(opt_int(c, "async_engine_count", origin, 0));
    p.cuda.persisting_l2_bytes = static_cast<uint64_t>(opt_int(c, "persisting_l2_bytes", origin, 0));
    for (const auto& [key, field] : {std::pair<const char*, bool*>{"peer_access", &p.cuda.peer_access},
                                     {"concurrent_managed_access", &p.cuda.concurrent_managed_access}}) {
      if (auto b = c.map.find(key); b != c.map.end()) {
        if (b->second.kind != Value::Kind::Bool) fail(origin, std::string("cuda.") + key + " must be true/false");
        *field = b->second.b;
      }
    }
  }

  // Optional: how the SMs are grouped (GpuLayout). The counts must add up to the SM count.
  if (auto it = doc.map.find("layout"); it != doc.map.end()) {
    if (it->second.kind != Value::Kind::Map) fail(origin, "'layout' must be a map");
    const Value& l = it->second;
    p.layout.gpcs = static_cast<uint32_t>(opt_int(l, "gpcs", origin, 0));
    p.layout.tpcs = static_cast<uint32_t>(opt_int(l, "tpcs", origin, 0));
    p.layout.sms_per_tpc = static_cast<uint32_t>(opt_int(l, "sms_per_tpc", origin, 0));
    if (auto d = l.map.find("counts_derived"); d != l.map.end()) {
      if (d->second.kind != Value::Kind::Bool) fail(origin, "layout.counts_derived must be true/false");
      p.layout.counts_derived = d->second.b;
    }
    if (p.layout.tpcs && p.layout.sms_per_tpc &&
        uint64_t{p.layout.tpcs} * p.layout.sms_per_tpc != p.limits.multiprocessors)
      fail(origin, "layout: " + std::to_string(p.layout.tpcs) + " TPCs of " + std::to_string(p.layout.sms_per_tpc) +
                       " SMs are not the " + std::to_string(p.limits.multiprocessors) + " multiprocessors");
    if (p.layout.gpcs > p.layout.tpcs) fail(origin, "layout: more GPCs than TPCs");
  }

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
    // Reliability and link; TelemetryClass says what each means and where the
    // values come from. All optional: a profile without them reports no ECC,
    // no memory sensor and no link, which is [N/A] rather than a guess.
    auto flag = [&](const char* key) {
      auto f = t.map.find(key);
      if (f == t.map.end()) return false;
      if (f->second.kind != Value::Kind::Bool)
        fail(origin, "telemetry." + std::string(key) + " must be true/false");
      return f->second.b;
    };
    p.telemetry.ecc = flag("ecc");
    p.telemetry.memory_temperature = flag("memory_temperature");
    const std::string memory = get_str_opt(t, "memory");
    if (!memory.empty() && memory != "hbm" && memory != "gddr")
      fail(origin, "telemetry.memory must be hbm or gddr, got: " + memory);
    p.telemetry.hbm = memory == "hbm";
    if (const std::string link = get_str_opt(t, "pcie_link"); !link.empty()) {
      unsigned gen = 0, width = 0;
      char extra = 0;
      if (std::sscanf(link.c_str(), "Gen%u x%u%c", &gen, &width, &extra) != 2 || gen < 1 ||
          gen > 6 || (width != 1 && width != 2 && width != 4 && width != 8 && width != 16))
        fail(origin, "telemetry.pcie_link must look like \"Gen4 x16\", got: " + link);
      p.telemetry.pcie_gen = gen;
      p.telemetry.pcie_width = width;
    }
    if (const std::string nv = get_str_opt(t, "nvlink"); !nv.empty()) {
      unsigned version = 0, links = 0;
      char extra = 0;
      if (std::sscanf(nv.c_str(), "NVLink%u x%u%c", &version, &links, &extra) != 2 || version < 1 ||
          version > 6 || links < 1 || links > 18)
        fail(origin, "telemetry.nvlink must look like \"NVLink4 x18\", got: " + nv);
      p.telemetry.nvlink_version = version;
      p.telemetry.nvlink_count = links;
    }
  }

  if (auto it = doc.map.find("features"); it != doc.map.end()) {
    if (it->second.kind != Value::Kind::Map) fail(origin, "'features' must be a map");
    for (auto& [k, v] : it->second.map) {
      if (v.kind != Value::Kind::Bool) fail(origin, "feature '" + k + "' must be true/false");
      p.features[k] = v.b;
    }
  }

  // Optional: the numerics, and what each tagged value rests on.
  parse_numerics(doc, p, origin);
  parse_provenance(doc, p, origin);
  return p;
}

}  // namespace vgpu
