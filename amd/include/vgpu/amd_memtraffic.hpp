// How many bytes a lane of a vector memory instruction moves, worked out from the instruction's name.
//
// The AMD counters (InstructionCounts) count instructions: a 16-byte load counts like a 4-byte one. These give the
// bytes each active lane asks for, which the executor adds up at the dispatch (InstructionCounts::vmem_bytes_*,
// lds_bytes_*), as the NVIDIA engine does for global and shared memory.
//
// What is counted is what the active lanes ask for: lanes x bytes a lane. A lane whose address falls outside a buffer
// is still counted (the instruction asked for it), and so is a flat access whose lanes reach LDS (it is a vector memory
// instruction). An atomic counts as written bytes, as it counts as a write in vmem_wr, whether or not it returns a value.
//
// What is NOT counted, and reads as zero: image and tensor instructions, formatted buffer accesses (their size is the
// format's), loads and async copies to LDS, scalar memory, and the LDS instructions that move no data to or from LDS
// memory (swizzle, permute, bpermute, the global wave sync and the counters of ds_append and ds_consume).
#pragma once

#include <cstdint>
#include <string_view>

namespace vgpu::amd::gcn {

struct LaneTraffic {
  uint32_t read = 0;      // bytes a lane reads
  uint32_t written = 0;   // bytes a lane writes (an atomic's operand width)
};

namespace detail {

// The widest "b32", "u16", "i8", "f64" in the name, in bits: `ds_read_b128`, `ds_load_u8_d16_hi`, `ds_load_b128_tr_b6` (128:
// the widest is the access), `ds_add_rtn_f32`. "d16" and "tr16" are not types, so they do not count.
inline uint32_t widest_type_bits(std::string_view s) {
  uint32_t best = 0;
  for (size_t i = 0; i + 1 < s.size(); ++i) {
    const char c = s[i];
    const bool lead = i == 0 || s[i - 1] == '_';
    if (!lead || (c != 'b' && c != 'u' && c != 'i' && c != 'f')) continue;
    size_t j = i + 1;
    uint32_t v = 0;
    while (j < s.size() && s[j] >= '0' && s[j] <= '9') v = v * 10 + static_cast<uint32_t>(s[j++] - '0');
    if (j > i + 1 && (j == s.size() || s[j] == '_')) best = best > v ? best : v;
  }
  return best;
}

inline bool has(std::string_view s, std::string_view part) { return s.find(part) != std::string_view::npos; }
inline bool begins(std::string_view s, std::string_view part) { return s.substr(0, part.size()) == part; }

}  // namespace detail

// One LDS instruction ("ds_read_b64", name with or without the "ds_"), per active lane.
inline LaneTraffic ds_lane_traffic(std::string_view name) {
  if (detail::begins(name, "ds_")) name.remove_prefix(3);
  using detail::begins;
  using detail::has;
  // Move no data to or from LDS memory: the cross-lane operations, the wave sync, the counters, the BVH stack helpers, and the
  // instructions whose operand is a register of the sequencer, not LDS.
  if (name == "nop" || has(name, "swizzle") || has(name, "permute") || begins(name, "gws") || name == "append" || name == "consume" ||
      name == "ordered_count" || name == "param_load" || name == "direct_load" || begins(name, "bvh_") || has(name, "barrier") ||
      has(name, "_gs_reg") || has(name, "condxchg32"))
    return {};
  const bool reads = begins(name, "read") || begins(name, "load");
  const bool writes = (begins(name, "write") || begins(name, "store")) && !begins(name, "storexchg");
  // Two addresses an instruction: ds_read2_b32, ds_read2st64_b64, ds_load_2addr_b32, ds_write2_b64, ds_wrxchg2_rtn_b32, ds_storexchg_2addr_rtn_b32.
  const bool pair = has(name, "2addr") || has(name, "read2") || has(name, "write2") || has(name, "wrxchg2");
  uint32_t bytes;
  if (has(name, "addtid")) bytes = 4;                             // ds_load_addtid_b32, ds_write_addtid_b32: a dword a lane
  else if (has(name, "pk_add")) bytes = 4;                        // two packed halves
  else bytes = detail::widest_type_bits(name) / 8;
  if (bytes == 0) bytes = 4;                                      // sub_clamp_rtn_u32 and the like carry a type; this is the dword they all share
  if (pair) bytes *= 2;
  if (reads) return {bytes, 0};
  if (writes) return {0, bytes};
  return {0, bytes};                                              // an atomic: counted as written, whether or not it returns
}

}  // namespace vgpu::amd::gcn
