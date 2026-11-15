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

// ---- LDS bank conflicts ----------------------------------------------------------------------------------------------
//
// LDS has 32 banks of 4 bytes (bank = dword address mod 32). A wave's access takes one pass through the banks when its lanes
// land on different banks, or on the same dword (a read is served to every lane that asks for it); lanes on one bank with
// different dwords take a pass each. SQC_LDS_BANK_CONFLICT counts the passes beyond the first.
//
// This is MEASURED, not read from a document: a wave of one to 64 lanes issuing one LDS instruction 1000 times at strides from
// 0 to 4096 bytes, read on an RX 6800 and an RX 6700 XT (gfx1030, gfx1031; rocprofv2, SQC_LDS_BANK_CONFLICT), 906 combinations of
// instruction, stride and lanes, all of which this reproduces. What the measurements show:
//  * A wave64 is two wave32 halves, counted one after the other.
//  * A lane that moves more than one dword is served in groups of lanes, 32 banks at a time: 16 lanes for a 64-bit access,
//    8 for 96 and 128 bits. Conflicts are counted per group, and groups do not share a pass (so two lanes on one dword
//    conflict if they are in different groups).
//  * 8-bit and 16-bit accesses are dword accesses (the dword they touch).
//  * An atomic is not served from one fetch: lanes on the same dword take a pass each.
//  * ds_read2 / ds_write2 (and the st64 forms) are two accesses, one per offset, counted separately.
//  * An access whose address is not a multiple of its width (64 bits and wider) is served one lane at a time and counts no
//    conflicts. (96 bits is taken to need 16: the card showed 8-aligned 96-bit accesses behave so; 12 was not measured.)
//  * A partly filled wave of 96- or 128-bit READS pays at least min(4, ceil(lanes/4)) passes, so a few lanes at one dword
//    count fewer conflicts than the same writes do (the writes count the groups' own excess).
// Verified on RDNA2 only. The same rule is applied to the other targets (CDNA's LDS also has 32 banks); there it is modelled,
// not checked against a card.
struct LdsBankAccess {
  bool counted = false;   // false: no model for this instruction (it counts no conflicts)
  uint32_t words = 1;     // dwords a lane moves in one access
  bool atomic = false;
  bool write = false;     // a store, or an atomic
  bool pair = false;      // two accesses, at offset0 and offset1 (in units of `unit` bytes)
  uint32_t unit = 4;      // bytes in one unit of those offsets
};

inline LdsBankAccess ds_bank_access(std::string_view name) {
  using detail::begins;
  using detail::has;
  LdsBankAccess a;
  if (detail::begins(name, "ds_")) name.remove_prefix(3);
  const LaneTraffic t = ds_lane_traffic(name);
  if (!t.read && !t.written) return a;                    // moves nothing through the banks
  if (has(name, "addtid") || has(name, "_tr") || has(name, "tr4") || has(name, "tr6") || has(name, "tr8")) return a;   // not measured
  const bool reads = begins(name, "read") || begins(name, "load");
  const bool stores = (begins(name, "write") || begins(name, "store")) && !begins(name, "storexchg");
  const bool pair = has(name, "2addr") || has(name, "read2") || has(name, "write2");
  if (!reads && !stores && (has(name, "2addr") || has(name, "wrxchg2"))) return a;   // an atomic exchange of two addresses: not measured
  const uint32_t bits = detail::widest_type_bits(name);
  a.words = bits >= 128 ? 4 : bits >= 96 ? 3 : bits >= 64 ? 2 : 1;
  a.atomic = !reads && !stores;
  a.write = !reads;
  a.pair = pair;
  a.unit = a.words == 2 ? 8 : 4;
  if (has(name, "st64") || has(name, "stride64")) a.unit *= 64;
  a.counted = true;
  return a;
}

// The conflicts of ONE access by a wave, from each lane's byte address (`addr[lane]`, 64 lanes) and the lanes that are active.
inline uint32_t lds_bank_conflicts(const uint32_t* addr, uint64_t exec, const LdsBankAccess& a) {
  if (!exec) return 0;
  const uint32_t align = a.words == 2 ? 8 : a.words >= 3 ? 16 : 1;
  for (uint64_t live = exec; live; live &= live - 1)
    if (align > 1 && addr[__builtin_ctzll(live)] % align) return 0;   // served a lane at a time
  const uint32_t group = a.words == 1 ? 32 : a.words == 2 ? 16 : 8;     // lanes served together
  const bool wide_read = a.words >= 3 && !a.write;
  uint32_t total = 0;
  for (uint32_t half = 0; half < 64; half += 32) {
    const uint32_t mask = static_cast<uint32_t>(exec >> half);
    if (!mask) continue;
    const uint32_t active = static_cast<uint32_t>(__builtin_popcount(mask));
    uint32_t passes = 0, excess = 0;
    for (uint32_t g = 0; g < 32; g += group) {
      const uint32_t gm = group == 32 ? mask : (mask >> g) & ((1u << group) - 1);
      if (!gm) continue;
      uint32_t seen[32];                 // the dwords asked for in this group (at most 32)
      uint8_t per_bank[32] = {};
      uint32_t n = 0, worst = 0;
      for (uint32_t m = gm; m; m &= m - 1) {
        const uint32_t base = addr[half + g + static_cast<uint32_t>(__builtin_ctz(m))] / 4;
        for (uint32_t k = 0; k < a.words; ++k) {
          const uint32_t dword = base + k;
          bool dup = false;
          if (!a.atomic)
            for (uint32_t i = 0; i < n; ++i)
              if (seen[i] == dword) { dup = true; break; }
          if (dup) continue;
          seen[n++] = dword;
          const uint32_t c = ++per_bank[dword % 32];
          worst = c > worst ? c : worst;
        }
      }
      passes += worst;
      excess += worst - 1;
    }
    if (wide_read) {
      const uint32_t least = active >= 16 ? 4 : (active + 3) / 4;   // min(4, ceil(active / 4))
      total += passes > least ? passes - least : 0;
    } else {
      total += excess;
    }
  }
  return total;
}

}  // namespace vgpu::amd::gcn
