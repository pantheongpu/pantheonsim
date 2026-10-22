// A global load or store whose lanes run over consecutive addresses goes to memory as one call, and atomics across
// host threads are a compare-and-swap on the word: each held against the way it was done before -- lane by lane, and
// under a lock -- bit for bit.
//
// The kernels are amd/tests/data/memruns.c: each lane copies one 4, 8 or 16 byte value from `src + soff[t]` to
// `dst + doff[t]` if its `on[t]` is set. The test hands them address patterns of every kind (consecutive,
// strided, descending, repeated, overlapping, misaligned, partly or wholly out of range, across chunk and
// allocation edges, random) and masks of lanes, runs the dispatch with the new path on and then off against twin
// memories, and compares what was left in memory, how much of it was made real, the error (its text) or the
// counts. The threaded checks are in test_gcn_atomics.cpp.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vgpu/error.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

amd::CodeObject object() {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/memruns." + vtest::amd_target() + ".o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

struct Rng {
  uint64_t s;
  uint64_t next() {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    return s;
  }
  uint64_t below(uint64_t n) { return next() % n; }
};

constexpr uint32_t kLanes = 64;
constexpr uint64_t kRegion = 3 * kChunkSize + 4096;   // each of src and dst: chunks, the last short

struct Image {
  MemoryManager mem{1ull << 30};
  uint64_t src = 0, dst = 0, soff = 0, doff = 0, on = 0, args = 0;
};

// The same memory every time for one seed: src random bytes, dst random in places, untouched or one repeated byte
// in others, so a store may land in a chunk that is real, uniform or not yet made.
void fill(Image& m, uint64_t seed, const amd::Kernel& k, bool wide_offsets, const uint64_t* so, const uint64_t* dof,
          const uint32_t* on) {
  Rng rng{seed * 2654435761ull + 1};
  m.src = m.mem.alloc(kRegion);
  m.dst = m.mem.alloc(kRegion);
  // A neighbour right after dst, so an access running off its end is in another allocation or in none.
  m.mem.alloc(8192);
  std::vector<uint8_t> junk(kRegion);
  for (auto& b : junk) b = static_cast<uint8_t>(rng.next());
  m.mem.write(m.src, junk.data(), junk.size());
  for (uint64_t off = 0; off + kChunkSize <= kRegion; off += kChunkSize) {
    switch (rng.below(3)) {
      case 0: {
        const uint8_t pat[1] = {static_cast<uint8_t>(rng.below(2) ? 0xFF : 0x00)};
        m.mem.fill(m.dst + off, pat, 1, kChunkSize);
        break;
      }
      case 1: m.mem.write(m.dst + off, junk.data() + rng.below(1000), 512); break;
      default: break;   // never written
    }
  }
  const size_t osz = wide_offsets ? 8 : 4;
  std::vector<uint8_t> bs(kLanes * osz), bd(kLanes * osz);
  for (uint32_t t = 0; t < kLanes; ++t) {
    if (wide_offsets) {
      std::memcpy(&bs[t * 8], &so[t], 8), std::memcpy(&bd[t * 8], &dof[t], 8);
    } else {
      const uint32_t a = static_cast<uint32_t>(so[t]), b = static_cast<uint32_t>(dof[t]);
      std::memcpy(&bs[t * 4], &a, 4), std::memcpy(&bd[t * 4], &b, 4);
    }
  }
  m.soff = m.mem.alloc(bs.size()), m.doff = m.mem.alloc(bd.size()), m.on = m.mem.alloc(kLanes * 4);
  m.mem.write(m.soff, bs.data(), bs.size());
  m.mem.write(m.doff, bd.data(), bd.size());
  m.mem.write(m.on, on, kLanes * 4);
  std::vector<uint8_t> buf(k.kernarg_size, 0);
  const uint64_t values[5] = {m.src, m.dst, m.soff, m.doff, m.on};
  for (size_t i = 0; i < 5 && i < k.args.size(); ++i)
    for (uint32_t b = 0; b < k.args[i].size && b < 8; ++b) buf[k.args[i].offset + b] = static_cast<uint8_t>(values[i] >> (8 * b));
  m.args = m.mem.alloc(buf.size());
  m.mem.write(m.args, buf.data(), buf.size());
}

struct Outcome {
  std::string error;
  std::vector<uint8_t> src, dst;
  uint64_t resident = 0, waves = 0, instructions = 0, vmem_rd = 0, vmem_wr = 0, flat = 0;
  bool operator==(const Outcome& o) const {
    return error == o.error && src == o.src && dst == o.dst && resident == o.resident && waves == o.waves &&
           instructions == o.instructions && vmem_rd == o.vmem_rd && vmem_wr == o.vmem_wr && flat == o.flat;
  }
};

Outcome run(const amd::CodeObject& o, const char* name, bool wide_offsets, uint64_t seed, const uint64_t* so,
            const uint64_t* dof, const uint32_t* on, bool runs, uint64_t* took = nullptr) {
  (void)took;
  const amd::Kernel* k = amd::find_kernel(o, name);
  if (!k) throw vtest::Failure(std::string("no kernel named ") + name);
  Image m;
  fill(m, seed, *k, wide_offsets, so, dof, on);
  amd::set_fast_paths(true, runs);
  Outcome out;
  try {
    amd::Dispatch d;
    d.object = &o;
    d.kernel = k;
    d.kernarg = m.args;
    d.groups[0] = 1;
    d.group_size[0] = kLanes;
    const amd::DispatchStats st = amd::execute(d, m.mem);
    out.waves = st.waves, out.instructions = st.instructions, out.vmem_rd = st.counts.vmem_rd,
    out.vmem_wr = st.counts.vmem_wr, out.flat = st.counts.flat;
  } catch (const std::exception& e) {
    out.error = e.what();
  }
  amd::set_fast_paths(true, true);
  out.src.resize(kRegion), out.dst.resize(kRegion);
  m.mem.read(m.src, out.src.data(), kRegion);
  m.mem.read(m.dst, out.dst.data(), kRegion);
  out.resident = m.mem.resident_bytes();
  return out;
}

struct Case {
  const char* kernel;
  bool wide_offsets;
  uint32_t bytes;
};
const Case kCases[] = {{"copy32", true, 4},  {"copy64", true, 8},  {"copy64v", true, 8},  {"copy128", true, 16},
                       {"copy32s", false, 4}, {"copy64s", false, 8}, {"copy128s", false, 16}};

// One address pattern for `bytes`-byte values: where lane t reads (soff) and writes (doff).
void pattern(Rng& rng, uint32_t kind, uint32_t bytes, uint64_t* so, uint64_t* dof) {
  const uint64_t base_s = (rng.below(4) == 0 ? rng.below(4096) : rng.below(8) * 16 * (1 + rng.below(8))) +
                          (rng.below(3) == 0 ? rng.below(4) : 0) * (rng.below(2) ? 1 : 0);
  const uint64_t base_d = rng.below(4) == 0 ? rng.below(4096) : rng.below(8) * 16 * (1 + rng.below(8));
  const uint64_t near_chunk = kChunkSize - bytes * (1 + rng.below(40));
  for (uint32_t t = 0; t < kLanes; ++t) {
    switch (kind) {
      case 0: so[t] = base_s + uint64_t{t} * bytes, dof[t] = base_d + uint64_t{t} * bytes; break;   // consecutive
      case 1: so[t] = base_s + uint64_t{t} * bytes, dof[t] = base_d + 4096 * 2 + uint64_t{kLanes - 1 - t} * bytes; break;
      case 2: so[t] = base_s + uint64_t{t} * 2 * bytes, dof[t] = base_d + uint64_t{t} * bytes; break;   // strided
      case 3: so[t] = base_s + uint64_t{kLanes - 1 - t} * bytes, dof[t] = base_d + uint64_t{t} * bytes; break;
      case 4: so[t] = base_s, dof[t] = base_d + uint64_t{t} * bytes; break;                         // all one address
      case 5: so[t] = base_s + uint64_t{t} * bytes, dof[t] = base_d + uint64_t{t / 2} * bytes; break;   // overlapping stores
      case 6: so[t] = base_s + uint64_t{t / 3} * bytes, dof[t] = base_d + uint64_t{t} * bytes; break;   // repeats
      case 7: so[t] = near_chunk + uint64_t{t} * bytes, dof[t] = near_chunk + 3 * kChunkSize / 2 + uint64_t{t} * bytes; break;
      case 8: so[t] = kRegion - bytes * 20 + uint64_t{t} * bytes, dof[t] = base_d + uint64_t{t} * bytes; break;   // runs off the end
      case 9: so[t] = base_s + uint64_t{t} * bytes, dof[t] = kRegion - bytes * 30 + uint64_t{t} * bytes; break;
      case 10: so[t] = base_s + uint64_t{t} * bytes + 1 + rng.below(3) * 0, dof[t] = base_d + uint64_t{t} * bytes + 2; break;   // misaligned
      case 11: so[t] = base_s + uint64_t{t} * bytes + 4, dof[t] = base_d + uint64_t{t} * bytes + 4; break;   // 4-aligned only
      case 12: so[t] = base_s + uint64_t{t} * bytes + (bytes >= 8 ? 8 : 0), dof[t] = base_d + uint64_t{t} * bytes + (bytes >= 16 ? 8 : 0); break;
      case 13: so[t] = rng.below(kRegion / bytes) * bytes, dof[t] = rng.below(kRegion / bytes) * bytes; break;   // random, aligned
      case 14: so[t] = rng.below(kRegion), dof[t] = rng.below(kRegion); break;                       // random, any byte
      case 15: so[t] = (t < 20 ? base_s + uint64_t{t} * bytes : rng.below(kRegion / bytes) * bytes), dof[t] = base_d + uint64_t{t} * bytes; break;
      case 16: so[t] = ~uint64_t{0} - bytes * 4 + uint64_t{t} * bytes, dof[t] = base_d + uint64_t{t} * bytes; break;   // wild addresses
      default:   // consecutive in pieces
        so[t] = base_s + (uint64_t{t} / 8) * 64 * bytes + uint64_t{t % 8} * bytes, dof[t] = base_d + uint64_t{t} * bytes;
        break;
    }
  }
}
constexpr uint32_t kKinds = 18;

void mask(Rng& rng, uint32_t kind, uint32_t* on) {
  for (uint32_t t = 0; t < kLanes; ++t) {
    switch (kind) {
      case 0: on[t] = 1; break;
      case 1: on[t] = t % 2; break;
      case 2: on[t] = t < 1 + rng.below(63); break;
      case 3: on[t] = t != 13 && t != 40; break;
      case 4: on[t] = t >= 5 && t < 29; break;
      default: on[t] = rng.below(5) != 0; break;
    }
  }
}

}  // namespace

VTEST(runs_of_lanes_move_the_bits_lane_by_lane_would) {
  const amd::CodeObject o = object();
  uint64_t checked = 0, failed_runs = 0;
  for (const Case& c : kCases) {
    for (uint32_t kind = 0; kind < kKinds; ++kind) {
      for (uint32_t m = 0; m < 7; ++m) {
        for (uint32_t rep = 0; rep < 3; ++rep) {
          Rng rng{(uint64_t{kind} << 20 | m << 12 | rep << 6 | c.bytes) * 0x9E3779B97F4A7C15ull + 7};
          uint64_t so[kLanes], dof[kLanes];
          uint32_t on[kLanes];
          pattern(rng, kind, c.bytes, so, dof);
          mask(rng, m, on);
          const uint64_t seed = rng.next();
          const Outcome fast = run(o, c.kernel, c.wide_offsets, seed, so, dof, on, true);
          const Outcome slow = run(o, c.kernel, c.wide_offsets, seed, so, dof, on, false);
          if (!(fast == slow))
            throw vtest::Failure(std::string(c.kernel) + ": pattern " + std::to_string(kind) + ", mask " +
                                 std::to_string(m) + ", rep " + std::to_string(rep) +
                                 " differ; error with runs {" + fast.error + "}, without {" + slow.error + "}");
          ++checked;
          failed_runs += !slow.error.empty();
        }
      }
    }
  }
  VCHECK(checked > 2000);
  VCHECK(failed_runs > 50);            // the error paths were met
  VCHECK(failed_runs < checked / 2);   // and most patterns ran to the end
}

VTEST(a_run_of_lanes_keeps_overlapping_stores_in_lane_order) {
  // Eight lanes storing to the same word, then pairs of lanes sharing a word: the last lane's value stays.
  const amd::CodeObject o = object();
  uint64_t so[kLanes], dof[kLanes];
  uint32_t on[kLanes];
  for (uint32_t t = 0; t < kLanes; ++t) so[t] = 4 * t, dof[t] = 4 * (t / 8), on[t] = 1;
  const Outcome a = run(o, "copy32", true, 99, so, dof, on, true);
  const Outcome b = run(o, "copy32", true, 99, so, dof, on, false);
  VCHECK(a == b);
  VCHECK(a.error.empty());
  Image m;
  const amd::Kernel* k = amd::find_kernel(o, "copy32");
  fill(m, 99, *k, true, so, dof, on);
  std::vector<uint8_t> src(kRegion);
  m.mem.read(m.src, src.data(), kRegion);
  for (uint32_t w = 0; w < 8; ++w)
    VCHECK_EQ(std::memcmp(a.dst.data() + 4 * w, src.data() + 4 * (8 * w + 7), 4), 0);
}

VTEST_MAIN
