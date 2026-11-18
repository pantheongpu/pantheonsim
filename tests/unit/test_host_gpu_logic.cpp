// The host-GPU mode's decisions that need no card: which kernels may run on it with the simulator's bits, the chunk residency
// protocol, and the size rule (include/vgpu/host_gpu.hpp). The part that drives the real GPU is run by
// nvidia/tests/e2e/run_host_gpu.sh, on a machine with a card.
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <initializer_list>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "vgpu/exec/launch.hpp"
#include "vgpu/host_gpu.hpp"
#include "vgpu/host_gpu_image.hpp"
#include "vgpu/host_gpu_source.hpp"
#include "vgpu/sass/exec.hpp"
#include "vtest.hpp"

using namespace vgpu::hostgpu;

VTEST(mode_is_exact_turbo_or_off) {
  VCHECK(mode_from_text("exact") == Mode::Exact);
  VCHECK(mode_from_text("turbo") == Mode::Turbo);
  VCHECK(mode_from_text(nullptr) == Mode::Off);
  VCHECK(mode_from_text("") == Mode::Off);
  VCHECK(mode_from_text("approx") == Mode::Off);   // not a mode: a typo must not turn one on
  VCHECK(mode_from_text("EXACT") == Mode::Off);
  VCHECK(mode_from_text("TURBO") == Mode::Off);
  VCHECK(mode_from_text("turbo ") == Mode::Off);
  VCHECK(mode_from_text("1") == Mode::Off);
}

VTEST(exact_set_takes_integer_and_ieee_code) {
  for (const char* m : {"IMAD", "IADD3", "LOP3", "PLOP3", "LEA", "SHF", "ISETP", "MOV", "SEL", "PRMT", "FADD", "FMUL", "FFMA", "FSETP", "DFMA", "DADD", "DMUL",
                        "I2FP", "F2I", "LDG", "STG", "LDS", "STS", "S2R", "BAR", "BRA", "EXIT", "SHFL", "VOTE", "FCHK", "FRND"})
    VCHECK(exact_opcode_allowed(m, {}));
  VCHECK(exact_opcode_allowed("FADD", {"FTZ"}));
  VCHECK(exact_opcode_allowed("FFMA", {"RM"}));
  VCHECK(exact_opcode_allowed("RED", {"E", "ADD", "STRONG", "GPU"}));   // an integer reduction: the order of the adds is not visible
}

VTEST(exact_set_refuses_what_differed_in_the_spike) {
  // Every kernel of the sampled launches that differed from the card held one of these (09 doc, M2).
  for (const char* m : {"MUFU", "HMMA", "IMMA", "DMMA", "TEX", "TLD", "ATOMG", "ATOMS", "ATOM", "CALL", "RET", "HFMA2", "BMOV.UNKNOWN", "VPRINTF"})
    VCHECK(!exact_opcode_allowed(m, {}));
  VCHECK(!exact_opcode_allowed("RED", {"E", "ADD", "F32", "FTZ", "RN", "STRONG", "GPU"}));   // float reduction: the order changes the bits
  VCHECK(!exact_opcode_allowed("RED", {"E", "ADD", "F64"}));
  VCHECK(!exact_opcode_allowed("RED", {"E", "CAS"}));
  VCHECK(!exact_opcode_allowed("RED", {"E", "EXCH"}));
  // ... in the decoder's own spelling, where the type and its rounding are one modifier (src/sass/decode.cpp kAtomType, dec_atom_f)
  VCHECK(!exact_opcode_allowed("RED", {"E", "ADD", "F32.FTZ.RN", "STRONG", "GPU"}));
  VCHECK(!exact_opcode_allowed("RED", {"E", "ADD", "F64.RN", "STRONG", "GPU"}));
  VCHECK(!exact_opcode_allowed("RED", {"E", "ADD", "F16x2.RN"}));
  VCHECK(!exact_opcode_allowed("RED", {"ADD", "F32x4.FTZ.RN"}) && !exact_opcode_allowed("RED", {"MAX", "BF16x2.RN"}));
  VCHECK(!exact_opcode_allowed("RED", {"E.ADD.FTZ"}) && !exact_opcode_allowed("RED", {"S32.CAS"}) && exact_opcode_allowed("RED", {"E.ADD.S32"}));   // every part of a joined modifier is read
  VCHECK(exact_opcode_allowed("RED", {"E", "ADD", "S32", "STRONG", "GPU"}) && exact_opcode_allowed("RED", {"E", "MIN", "64"}) && exact_opcode_allowed("RED", {"E", "OR", "S64", "WEAK", "SYS"}));
  VCHECK(!exact_opcode_allowed("", {}));
  VCHECK(!exact_opcode_allowed("imad", {}));   // names are SASS's, upper case; an unknown spelling is unknown
}

// The code of a few real kernels, as the opcode sets the trace of the spike showed: the verdict has to be the one the card gave.
static bool kernel_ok(const std::vector<std::pair<std::string, std::vector<std::string>>>& code) {
  for (const auto& [m, mods] : code)
    if (!exact_opcode_allowed(m, mods)) return false;
  return true;
}
VTEST(exact_set_matches_the_measured_kernels) {
  // PolyBench gemm_kernel: integer address arithmetic, FFMA, loads, stores: agreed.
  VCHECK(kernel_ok({{"S2R", {}}, {"IMAD", {}}, {"IADD3", {}}, {"LDG", {"E"}}, {"FFMA", {}}, {"FMUL", {}}, {"STG", {"E"}}, {"ISETP", {"GE", "AND"}}, {"BRA", {}}, {"EXIT", {}}}));
  // Rodinia lud_internal (shared memory, barrier): agreed.
  VCHECK(kernel_ok({{"LDS", {}}, {"STS", {}}, {"BAR", {"SYNC"}}, {"FFMA", {}}, {"LDG", {"E"}}, {"STG", {"E"}}}));
  // HeCBench silu forward (expf: MUFU.EX2 and MUFU.RCP): differed by one ulp.
  VCHECK(!kernel_ok({{"FFMA", {}}, {"MUFU", {"EX2"}}, {"MUFU", {"RCP"}}, {"FCHK", {}}, {"FADD", {"FTZ"}}}));
  // HeCBench aop compute_partial_beta (double atomicAdd as a compare-and-swap loop): card-nondeterministic.
  VCHECK(!kernel_ok({{"DADD", {}}, {"ATOMS", {"CAST", "SPIN", "64"}}, {"DFMA", {}}}));
  // HeCBench bh TreeBuildingKernel (ATOMG.E.CAS): card-nondeterministic.
  VCHECK(!kernel_ok({{"FSETP", {"GEU", "AND"}}, {"ATOMG", {"E", "CAS", "STRONG", "GPU"}}}));
}

// ---- residency: a reader on either side always sees the last write ------------------------------------------------------------

namespace {
struct World {
  uint64_t n;
  Residency res;
  std::vector<int> truth, host, card;
  int next = 1;
  explicit World(uint64_t chunks) : n(chunks), res(chunks), truth(chunks, 0), host(chunks, 0), card(chunks, -1) {}

  void sim_write(uint64_t c, bool whole) {
    for (uint64_t p : res.sim_write(c, c, whole)) host[p] = card[p];   // a partial write needs the chunk's bytes first
    if (!whole) VCHECK_EQ(host[c], truth[c]);                          // ... and the chunk is current before it is merged into
    host[c] = truth[c] = next++;
  }
  void sim_read(uint64_t c) {
    for (uint64_t p : res.sim_read(c, c)) host[p] = card[p];
    VCHECK_EQ(host[c], truth[c]);
  }
  void sim_launch(std::mt19937& rng) {   // the simulator runs a launch: Card chunks come back first, then it may write any chunk
    for (uint64_t p : res.sim_launch()) host[p] = card[p];
    for (uint64_t c = 0; c < n; ++c) VCHECK_EQ(host[c], truth[c]);
    for (uint64_t c = 0; c < n; ++c)
      if (rng() % 3 == 0) host[c] = truth[c] = next++;
    VCHECK_EQ(res.card_valid(), uint64_t{0});   // nothing of the card's copy is valid any more
  }
  void card_launch(std::mt19937& rng) {
    const std::vector<uint64_t> to_push = res.host_chunks();
    for (uint64_t p : res.card_prepare()) card[p] = host[p];
    VCHECK(res.host_chunks().empty());
    (void)to_push;
    for (uint64_t c = 0; c < n; ++c) VCHECK_EQ(card[c], truth[c]);     // the card sees every chunk as the simulator last left it
    for (uint64_t c = 0; c < n; ++c)
      if (rng() % 3 == 0) card[c] = truth[c] = next++;                 // the kernel writes some chunks
    res.card_launched();
  }
};
}  // namespace

VTEST(residency_keeps_both_sides_coherent) {
  for (uint32_t seed = 1; seed <= 200; ++seed) {
    std::mt19937 rng(seed);
    World w(1 + seed % 13);
    for (int step = 0; step < 400; ++step) {
      const uint64_t c = rng() % w.n;
      switch (rng() % 7) {
        case 0: w.sim_write(c, true); break;
        case 1: w.sim_write(c, false); break;
        case 2:
        case 3: w.sim_read(c); break;
        case 4: w.sim_launch(rng); break;
        default: w.card_launch(rng); break;
      }
    }
  }
}

// The counts are kept by every change of a chunk, not recounted; the gate's fast path trusts `access_changes_state` to skip only what is a no-op.
namespace {
void check_counts(const Residency& r) {
  uint64_t n[3] = {0, 0, 0};
  for (uint64_t c = 0; c < r.chunks(); ++c) ++n[static_cast<unsigned>(r.at(c))];
  VCHECK_EQ(r.count(Where::Host), n[0]);
  VCHECK_EQ(r.count(Where::Card), n[1]);
  VCHECK_EQ(r.count(Where::Both), n[2]);
  VCHECK_EQ(r.card_valid(), n[1] + n[2]);
  VCHECK_EQ(r.host_chunks().size(), n[0]);
}
bool same_states(const Residency& a, const Residency& b) {
  for (uint64_t c = 0; c < a.chunks(); ++c)
    if (a.at(c) != b.at(c)) return false;
  return true;
}
}  // namespace

VTEST(residency_counts_match_a_recount_after_every_operation) {
  for (uint32_t seed = 1; seed <= 300; ++seed) {
    std::mt19937 rng(seed);
    const uint64_t n = 1 + seed % 17;
    Residency r(n);
    check_counts(r);
    for (int step = 0; step < 300; ++step) {
      const uint64_t a = rng() % n, b = a + rng() % (n - a);
      switch (rng() % 9) {
        case 0: r.sim_write(a, b, rng() % 2); break;
        case 1: r.sim_read(a, b); break;
        case 2: r.card_prepare(); break;
        case 3: r.card_launched(); break;
        case 4: r.card_launched_writing(a, b); break;
        case 5: r.sim_launch(); break;
        case 6: r.set(a, static_cast<Where>(rng() % 3)); break;
        case 7: r.sim_read(a, b + 3); break;   // past the end: clamped
        default: r.sim_write(a, b + 3, true); break;
      }
      check_counts(r);
    }
  }
}

VTEST(access_changes_state_is_exactly_whether_the_gate_would_do_anything) {   // a skipped access must be one that the full path would leave alone
  for (uint32_t seed = 1; seed <= 300; ++seed) {
    std::mt19937 rng(seed);
    const uint64_t n = 1 + seed % 11;
    Residency r(n);
    for (int step = 0; step < 200; ++step) {
      switch (rng() % 4) {
        case 0: r.card_launched(); break;
        case 1: r.set(rng() % n, static_cast<Where>(rng() % 3)); break;
        case 2: r.card_prepare(); break;
        default: break;
      }
      const uint64_t a = rng() % n, b = a + rng() % (n - a);
      for (bool write : {false, true}) {
        Residency full = r;
        const std::vector<uint64_t> pulled = write ? full.sim_write(a, b, false) : full.sim_read(a, b);
        const bool changes = !pulled.empty() || !same_states(full, r);
        VCHECK_EQ(r.access_changes_state(a, b, write), changes);
      }
    }
  }
}

VTEST(residency_queries_do_not_grow_with_the_allocation) {   // the gate asked for the counts after every access: 2 x 1 M chunks scanned each time
  Residency r(1u << 20);   // 64 GiB of chunks
  r.card_launched();
  r.sim_read(0, 3);
  const auto t0 = std::chrono::steady_clock::now();
  uint64_t sink = 0;
  for (uint64_t i = 0; i < 200000; ++i) {
    sink += r.card_valid() + r.count(Where::Card) + (r.access_changes_state(i, i, false) ? 1 : 0);
    r.sim_read(i % 8, i % 8);
  }
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  VCHECK(sink > 0);
  VCHECK(s < 2.0);   // microseconds in all; a scan of the chunks took more than 20 s for this many queries
}

VTEST(chunk_runs_group_adjacent_chunks) {
  VCHECK(chunk_runs({}).empty());
  const auto r = chunk_runs({0, 1, 2, 5, 7, 8});
  VCHECK_EQ(r.size(), size_t{3});
  VCHECK(r[0] == std::make_pair(uint64_t{0}, uint64_t{3}));
  VCHECK(r[1] == std::make_pair(uint64_t{5}, uint64_t{1}));
  VCHECK(r[2] == std::make_pair(uint64_t{7}, uint64_t{2}));
}

VTEST(chunk_span_knows_what_a_write_covers) {
  const uint64_t K = 64 * 1024, base = 0x2000'0000'0000ull;
  ChunkSpan s = chunk_span(base, 4 * K, base, 4 * K, K);   // the whole allocation
  VCHECK(s.any && s.first == 0 && s.last == 3 && s.first_whole && s.last_whole);
  s = chunk_span(base, 4 * K, base + 10, 100, K);          // a few bytes inside chunk 0
  VCHECK(s.any && s.first == 0 && s.last == 0 && !s.first_whole && !s.last_whole);
  s = chunk_span(base, 4 * K, base + K, K, K);             // exactly chunk 1
  VCHECK(s.first == 1 && s.last == 1 && s.first_whole && s.last_whole);
  s = chunk_span(base, 4 * K, base + K + 1, 2 * K, K);     // runs from inside chunk 1 into chunk 3
  VCHECK(s.first == 1 && s.last == 3 && !s.first_whole && !s.last_whole);
  s = chunk_span(base, 3 * K + 100, base + 3 * K, 100, K); // the short last chunk, covered in full
  VCHECK(s.first == 3 && s.last == 3 && s.first_whole && s.last_whole);
  s = chunk_span(base, 4 * K, base + 5 * K, 10, K);        // outside the allocation
  VCHECK(!s.any);
  s = chunk_span(base, 4 * K, base - 100, 200, K);         // starts below it: clamped
  VCHECK(s.any && s.first == 0 && s.last == 0 && !s.first_whole);
  s = chunk_span(base, 4 * K, base, 0, K);                 // empty
  VCHECK(!s.any);
}

VTEST(residency_moves_only_what_it_must) {
  Residency r(8);
  VCHECK_EQ(r.count(Where::Host), uint64_t{8});
  VCHECK_EQ(r.card_prepare().size(), size_t{8});   // everything is new: all pushed
  VCHECK_EQ(r.count(Where::Both), uint64_t{8});
  VCHECK_EQ(r.card_prepare().size(), size_t{0});   // a second launch with no simulator writes in between pushes nothing
  r.card_launched();
  VCHECK_EQ(r.count(Where::Card), uint64_t{8});
  VCHECK_EQ(r.sim_read(2, 3).size(), size_t{2});   // the simulator reads two chunks: two pulled
  VCHECK_EQ(r.sim_read(2, 3).size(), size_t{0});   // and they are Both: no second pull
  VCHECK_EQ(r.sim_write(2, 2, true).size(), size_t{0});   // a whole-chunk overwrite needs nothing from the card
  VCHECK(r.at(2) == Where::Host);
  VCHECK_EQ(r.sim_write(5, 5, false).size(), size_t{1});  // a partial write of a Card chunk pulls it first
  VCHECK_EQ(r.card_prepare().size(), size_t{2});          // chunks 2 and 5 are the simulator's again: pushed; 3 is Both, the others Card
  VCHECK(r.at(4) == Where::Card);                         // a launch that follows still treats a Card chunk as the card's
}

// ---- the size rule ------------------------------------------------------------------------------------------------------------

VTEST(size_rule_refuses_small_launches) {
  Costs c;
  Estimate e;
  e.sim_s = 50e-6;   // the simulator needs 50 us: nothing the card does can beat that by the margin
  VCHECK(!should_offload(e, c));
  e.sim_s = 150e-6;  // below the floor, whatever the card path costs
  VCHECK(!should_offload(e, c));
}

VTEST(size_rule_weighs_the_copies) {
  Costs c;
  Estimate e;
  e.sim_s = 0.9;                // a 900 ms simulated gemm (PolyBench) with 3 MiB of memory
  e.push_bytes = e.pull_bytes = 3u << 20;
  VCHECK(should_offload(e, c));
  e.sim_s = 0.01;               // 10 ms of simulation with a GiB to move each way: the copies cost more than the margin allows
  e.push_bytes = e.pull_bytes = 1ull << 30;
  VCHECK(!should_offload(e, c));
  e.sim_s = 0.5;                // the same data, a long kernel: 1 GiB is ~0.19 s both ways, 2x margin = 0.38 s < 0.5 s
  VCHECK(should_offload(e, c));
  e.sim_s = 0.3;
  VCHECK(!should_offload(e, c));
}

VTEST(size_rule_margin_is_applied_to_the_whole_path) {
  Costs c;
  c.margin = 2.0;
  Estimate e;
  e.push_bytes = 0;
  e.pull_bytes = 0;
  e.card_kernel_s = 1e-3;
  const double path = card_path_seconds(e, c);
  VCHECK(path > 1e-3 && path < 1.1e-3);
  e.sim_s = 2.0 * path * 0.99;
  VCHECK(!should_offload(e, c));
  e.sim_s = 2.0 * path * 1.01;
  VCHECK(should_offload(e, c));
}

VTEST(sim_time_table_needs_two_samples_and_keeps_the_smallest) {
  SimTimeTable t;
  VCHECK(!t.predict("k", 10).has_value());   // never seen: the first launch is the simulator's
  t.record("k", 10, 0.2);                    // a cold first launch (decode, set-up): 20 ms per block, under kOneSampleMinSeconds
  VCHECK(!t.predict("k", 10).has_value());   // one sample is not enough: it is the cold one
  t.record("k", 10, 0.05);                   // the warm launch: 5 ms per block
  VCHECK(t.predict("k", 10).has_value());
  VCHECK(*t.predict("k", 10) > 0.049 && *t.predict("k", 10) < 0.051);     // the smaller of the two
  VCHECK(*t.predict("k", 20) > 0.099 && *t.predict("k", 20) < 0.101);     // twice the blocks, twice the time
  VCHECK(!t.predict("other", 10).has_value());
  t.record("k", 10, 9.0);                    // a slower launch never raises the estimate: it errs towards the simulator
  VCHECK(*t.predict("k", 10) < 0.051);
  t.record("k", 10, 0.01);                   // a faster one lowers it
  VCHECK(*t.predict("k", 10) < 0.011);
}

VTEST(a_launch_that_lost_to_the_estimate_ends_offloading) {
  VCHECK(offload_paid_off(0.010, 0.050));
  VCHECK(offload_paid_off(0.050, 0.050));
  VCHECK(!offload_paid_off(0.100, 0.003));   // 3DCONV: 100 ms of copies for a 3 ms simulation
}

constexpr uint64_t kMiB = 1ull << 20;

VTEST(traffic_ban_constants_are_the_replayed_ones) {   // chosen on the 64 saved traces (docs/perf-design/rounds/08-rounds-sp11-sp15.md, Round sp14); changing one needs a new replay
  VCHECK(kTrafficLaunches == 2 && kTrafficSharePercent == 25 && kTrafficMinBytes == 12 * kMiB && kTrafficSlowShare == 0.5);
}

VTEST(traffic_ban_fires_on_the_second_heavy_launch_in_a_row) {   // LU's shape: the whole 16 MiB buffer pushed again every launch
  TrafficTable t;
  VCHECK(!t.record("lu", 16 * kMiB, 16 * kMiB, 0.004, 0.006));   // one heavy launch: not yet
  VCHECK(t.record("lu", 16 * kMiB, 16 * kMiB, 0.004, 0.006));    // two in a row: banned
  VCHECK(t.record("lu", 16 * kMiB, 16 * kMiB, 0.004, 0.006));    // and it stays so while it keeps pushing
}

VTEST(traffic_ban_needs_the_launches_in_a_row) {
  TrafficTable t;
  VCHECK(!t.record("k", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
  VCHECK(!t.record("k", 0, 16 * kMiB, 0.0001, 0.006));            // a launch that pushed nothing resets the count
  VCHECK(!t.record("k", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
  VCHECK(t.record("k", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
}

VTEST(traffic_ban_ignores_a_chain_without_pushes_and_the_first_upload) {
  TrafficTable t;
  VCHECK(!t.record("fdtd", 48 * kMiB, 48 * kMiB, 0.1, 0.1));      // the initial upload: heavy once (FDTD-2D)
  for (int i = 0; i < 100; ++i) VCHECK(!t.record("fdtd", 0, 48 * kMiB, 0.0002, 0.1));   // then nothing moves
  TrafficTable u;
  for (int i = 0; i < 100; ++i) VCHECK(!u.record("chain", 0, 48 * kMiB, 0.0002, 0.1));
}

VTEST(traffic_ban_spares_a_big_push_that_is_a_small_share) {   // base64e: 8 MiB of hundreds, and aligned-types: big buffers pushed in pieces
  TrafficTable t;
  for (int i = 0; i < 10; ++i) VCHECK(!t.record("b64", 100 * kMiB, 1000 * kMiB, 0.5, 0.6));   // 10% of live: under the share
  for (int i = 0; i < 10; ++i) VCHECK(!t.record("b64b", 249 * kMiB, 1000 * kMiB, 0.5, 0.6));  // 24.9%
  TrafficTable v;
  VCHECK(!v.record("exact", 250 * kMiB, 1000 * kMiB, 0.5, 0.6));   // exactly the share: not above it
  VCHECK(!v.record("exact", 250 * kMiB, 1000 * kMiB, 0.5, 0.6));
  VCHECK(!v.record("over", 251 * kMiB, 1000 * kMiB, 0.5, 0.6));
  VCHECK(v.record("over", 251 * kMiB, 1000 * kMiB, 0.5, 0.6));     // one MiB over: the same pair bans
}

VTEST(traffic_ban_spares_a_small_push) {   // a share of a small heap is not traffic worth a ban
  TrafficTable t;
  for (int i = 0; i < 10; ++i) VCHECK(!t.record("small", 4 * kMiB, 4 * kMiB, 0.004, 0.006));
  TrafficTable e;
  VCHECK(!e.record("edge", 12 * kMiB, 12 * kMiB, 0.004, 0.006));   // exactly the floor: not above it
  VCHECK(!e.record("edge", 12 * kMiB, 12 * kMiB, 0.004, 0.006));
  VCHECK(!e.record("above", 12 * kMiB + 1, 12 * kMiB + 1, 0.004, 0.006));
  VCHECK(e.record("above", 12 * kMiB + 1, 12 * kMiB + 1, 0.004, 0.006));
}

VTEST(traffic_ban_spares_a_launch_the_card_path_wins_by_far) {   // GRAMSCHM's shape: huge pushes, but the simulator would take 3x longer
  TrafficTable t;
  for (int i = 0; i < 10; ++i) VCHECK(!t.record("gs", 48 * kMiB, 48 * kMiB, 0.040, 0.120));   // 0.33 of the estimate
  TrafficTable u;
  VCHECK(!u.record("edge", 48 * kMiB, 48 * kMiB, 0.05, 0.10));   // exactly half the estimate: not past it
  VCHECK(!u.record("edge", 48 * kMiB, 48 * kMiB, 0.05, 0.10));
  VCHECK(!u.record("past", 48 * kMiB, 48 * kMiB, 0.0501, 0.10));
  VCHECK(u.record("past", 48 * kMiB, 48 * kMiB, 0.0501, 0.10));
}

VTEST(traffic_ban_is_per_kernel) {
  TrafficTable t;
  VCHECK(!t.record("a", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
  VCHECK(!t.record("b", 16 * kMiB, 16 * kMiB, 0.004, 0.006));   // another kernel's first heavy launch is not a's second
  VCHECK(!t.record("c", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
  VCHECK(t.record("a", 16 * kMiB, 16 * kMiB, 0.004, 0.006));    // a's second
  VCHECK(t.record("b", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
  TrafficTable u;
  VCHECK(!u.record("a", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
  VCHECK(!u.record("b", 0, 16 * kMiB, 0.0, 0.006));             // b's quiet launch does not reset a
  VCHECK(u.record("a", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
}

VTEST(traffic_ban_first_card_launch_counts_traffic_only) {   // its path includes the module load: no time verdict from it
  TrafficTable t;
  VCHECK(!t.record("k", 16 * kMiB, 16 * kMiB, 0.0001, 0.006, false));   // timing not valid: traffic alone makes it heavy
  VCHECK(t.record("k", 16 * kMiB, 16 * kMiB, 0.004, 0.006, true));
  TrafficTable u;
  VCHECK(!u.record("k", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
  VCHECK(!u.record("k", 16 * kMiB, 16 * kMiB, 0.0001, 0.006, true));    // a valid, fast launch resets
  VCHECK(!u.record("k", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
}

VTEST(traffic_ban_with_no_live_bytes_never_fires) {
  TrafficTable t;
  for (int i = 0; i < 5; ++i) VCHECK(!t.record("k", 16 * kMiB, 0, 0.004, 0.006));
}

VTEST(static_estimate_scales_with_the_grid) {   // an estimate, not a bound: see host_gpu.hpp
  Costs c;
  // PolyBench gemm_kernel: 1,024 blocks of 256 threads, about 60 static instructions: 5 ms of simulation by this count (it took 900 ms).
  const double a = static_sim_seconds(1024, 256, 60, c);
  VCHECK(a > 4.9e-3 && a < 5.0e-3);
  VCHECK(static_sim_seconds(2048, 256, 60, c) > 1.99 * a);
  VCHECK(static_sim_seconds(1024, 33, 60, c) > 1.99 * static_sim_seconds(1024, 32, 60, c));   // 33 threads is two warps
  Estimate e;
  e.sim_s = a;
  e.push_bytes = e.pull_bytes = 3u << 20;
  VCHECK(should_offload(e, c));       // a 5 ms lower bound already pays for 3 MiB each way
  e.sim_s = static_sim_seconds(2, 32, 60, c);   // a two-block launch: far below the floor
  VCHECK(!should_offload(e, c));
}

VTEST(back_edge_scan) {   // (pc, target) of every branch of a kernel
  VCHECK(!has_back_edge({}));
  VCHECK(!has_back_edge({{0x20, 0x80}, {0x40, 0x60}}));   // forward branches only: a bounds check
  VCHECK(has_back_edge({{0x20, 0x80}, {0x90, 0x30}}));    // a loop's closing branch
  VCHECK(has_back_edge({{0x40, 0x40}}));                  // a branch to itself: a spin
  VCHECK(has_back_edge({{0x40, 0x0}}));
}

// A PolyBench gemm_kernel launch on an A10G (80 multiprocessors): a loop, 1,024 blocks of 256 threads, about 60 static instructions, three 4 MiB matrices.
static FirstLaunch gemm_shape() {
  FirstLaunch f;
  f.blocks = 1024;
  f.threads_per_block = 256;
  f.static_instrs = 60;
  f.back_edge = true;
  f.multiprocessors = 80;
  f.live_bytes = 12ull << 20;
  return f;
}

VTEST(first_launch_takes_a_looping_kernel_on_a_full_grid) {
  Costs c;
  VCHECK(first_launch_offload(gemm_shape(), c));
}

VTEST(first_launch_refuses_a_kernel_without_a_loop) {   // 3DCONV: a stencil, threads leave at a bounds check; the static count was 38x off
  Costs c;
  FirstLaunch f = gemm_shape();
  f.back_edge = false;
  VCHECK(!first_launch_offload(f, c));
}

VTEST(first_launch_refuses_a_grid_that_does_not_fill_the_machine) {   // MVT: 16 blocks
  Costs c;
  FirstLaunch f = gemm_shape();
  f.static_instrs = 4000;   // so large that the margin is never what refuses: only the grid is under test
  VCHECK(first_launch_offload(f, c));
  f.blocks = 16;
  VCHECK(!first_launch_offload(f, c));
  f.blocks = 8 * f.multiprocessors - 1;
  VCHECK(!first_launch_offload(f, c));
  f.blocks = 8 * f.multiprocessors;   // exactly 8 per multiprocessor is enough
  VCHECK(first_launch_offload(f, c));
  f.multiprocessors = 0;              // a profile that does not say
  VCHECK(!first_launch_offload(f, c));
}

VTEST(first_launch_caps_what_a_mistaken_launch_may_move) {   // 3DCONV pushed 128 MiB per launch; the cap is 256 MiB of live memory
  Costs c;
  FirstLaunch f = gemm_shape();
  f.static_instrs = 100000;               // so large that the margin is no obstacle
  f.live_bytes = 256ull << 20;
  VCHECK(first_launch_offload(f, c));
  f.live_bytes = (256ull << 20) + 1;
  VCHECK(!first_launch_offload(f, c));
}

VTEST(first_launch_needs_the_estimate_to_beat_the_card_path) {
  Costs c;
  FirstLaunch f = gemm_shape();
  f.static_instrs = 2;                    // a few instructions per thread: the copies cost more than the simulator would
  VCHECK(!first_launch_offload(f, c));
  f.static_instrs = 60;
  f.live_bytes = 200ull << 20;            // the same kernel on 200 MiB: 33 ms of copies against 5 ms of simulation
  VCHECK(!first_launch_offload(f, c));
}

VTEST(sim_time_table_counts_samples) {
  SimTimeTable t;
  VCHECK(t.samples("k") == 0);
  t.record("k", 4, 0.1);
  VCHECK(t.samples("k") == 1 && !t.predict("k", 4));
  t.record("k", 4, 0.1);
  VCHECK(t.samples("k") == 2 && t.predict("k", 4));
  VCHECK(t.samples("other") == 0);
}

// ---- the one-sample rule: a first launch that took 0.25 s or more is estimate enough, for two offloaded launches ----
VTEST(one_sample_is_enough_for_a_long_first_launch) {
  SimTimeTable t;
  t.record("k", 100, kOneSampleMinSeconds);                 // exactly at the threshold: counts
  VCHECK(t.predict("k", 100).has_value());
  VCHECK(*t.predict("k", 100) > 0.249 && *t.predict("k", 100) < 0.251);
  VCHECK(*t.predict("k", 200) > 0.499 && *t.predict("k", 200) < 0.501);   // scaled by the grid
  SimTimeTable s;
  s.record("short", 100, kOneSampleMinSeconds * 0.99);      // just under: not a long launch
  VCHECK(!s.predict("short", 100).has_value());
}

VTEST(one_sample_allowance_ends_after_two_offloaded_launches) {
  SimTimeTable t;
  t.record("k", 10, 1.0);
  for (unsigned i = 0; i < kOneSampleOffloads; ++i) {
    VCHECK(t.predict("k", 10).has_value());                 // still within the allowance
    t.note_offload("k");
  }
  VCHECK(!t.predict("k", 10).has_value());                  // spent: the next launch is the simulator's, for the second sample
  t.record("k", 10, 0.4);                                   // the warm sample
  VCHECK(t.predict("k", 10).has_value());                   // two samples now: the usual rule, no allowance needed
  VCHECK(*t.predict("k", 10) > 0.39 && *t.predict("k", 10) < 0.41);
  t.note_offload("k");                                      // (offloads on two samples do not count against anything)
  VCHECK(t.predict("k", 10).has_value());
}

VTEST(one_sample_allowance_is_per_kernel) {
  SimTimeTable t;
  t.record("a", 10, 1.0);
  t.record("b", 10, 1.0);
  for (unsigned i = 0; i < kOneSampleOffloads; ++i) t.note_offload("a");
  VCHECK(!t.predict("a", 10).has_value());
  VCHECK(t.predict("b", 10).has_value());
  t.note_offload("never_seen");                             // no entry is made by an offload alone
  VCHECK(!t.predict("never_seen", 10).has_value());
  VCHECK(t.samples("never_seen") == 0);
}

// ---- the driver start is charged by the offload decision (sp43; docs/performance.md "Round sp43") ----------------------------------------------
// An estimate that passes the plain rule by a wide margin: 50 ms of simulation, 1 MiB each way.
static Estimate fifty_ms() {
  Estimate e;
  e.sim_s = 0.05;
  e.push_bytes = e.pull_bytes = 1u << 20;
  return e;
}

VTEST(the_start_cost_is_the_measured_one) {   // 0.23 s on an A10G (Round sp42), rounded up; a change needs a new measurement
  Costs c;
  VCHECK(c.start_cost_s == 0.25);
}

VTEST(a_launch_that_cannot_repay_the_start_does_not_start_the_driver) {   // Rodinia: 5 to 50 ms of simulation in the whole program, 0.04x to 0.29x with the start
  Costs c;
  VCHECK(should_offload(fifty_ms(), c));                                   // worth a card with the driver up ...
  VCHECK(!should_offload_with_start(fifty_ms(), c, false, 0.0));           // ... not before it is paid for
  VCHECK(should_offload_with_start(fifty_ms(), c, true, 0.0));             // and plainly so once it is up
}

VTEST(one_large_launch_repays_the_start_at_once) {   // PolyBench SYRK: 4 s of simulation in one launch
  Costs c;
  Estimate e;
  e.sim_s = 4.0;
  e.push_bytes = e.pull_bytes = 8u << 20;
  VCHECK(should_offload_with_start(e, c, false, 0.0));
  e.sim_s = 0.4;   // (the margin: 0.4 s is less than twice the 0.25 s start plus the copies)
  VCHECK(!should_offload_with_start(e, c, false, 0.0));
  e.sim_s = 0.6;
  VCHECK(should_offload_with_start(e, c, false, 0.0));
}

VTEST(the_start_is_bought_when_as_much_was_spent_renting) {   // ski rental: a thousand launches of 50 ms, the driver starts after 0.25 s of them
  Costs c;
  VCHECK(!should_offload_with_start(fifty_ms(), c, false, 0.0));
  VCHECK(!should_offload_with_start(fifty_ms(), c, false, c.start_cost_s * 0.99));
  VCHECK(should_offload_with_start(fifty_ms(), c, false, c.start_cost_s));
  VCHECK(should_offload_with_start(fifty_ms(), c, false, 10.0));
}

VTEST(spent_time_never_sends_a_launch_the_plain_rule_refuses) {   // a tiny launch of a program that spent minutes elsewhere stays in the simulator
  Costs c;
  Estimate e;
  e.sim_s = 150e-6;   // under the floor
  VCHECK(!should_offload_with_start(e, c, false, 1000.0));
  VCHECK(!should_offload_with_start(e, c, true, 1000.0));
  e.sim_s = 0.01;     // 10 ms against a GiB to move each way
  e.push_bytes = e.pull_bytes = 1ull << 30;
  VCHECK(!should_offload_with_start(e, c, false, 1000.0));
}

VTEST(a_start_cost_of_zero_is_the_rule_before_sp43) {   // VGPU_HOST_GPU_START_COST=0
  Costs c;
  c.start_cost_s = 0;
  for (double s : {50e-6, 0.001, 0.05, 4.0}) {
    Estimate e = fifty_ms();
    e.sim_s = s;
    VCHECK_EQ(should_offload_with_start(e, c, false, 0.0), should_offload(e, c));
  }
}

VTEST(with_the_driver_up_the_decision_is_the_plain_rule) {   // for any account of spent time
  Costs c;
  for (double s : {50e-6, 0.001, 0.05, 4.0})
    for (double spent : {0.0, 0.1, 5.0}) {
      Estimate e = fifty_ms();
      e.sim_s = s;
      VCHECK_EQ(should_offload_with_start(e, c, true, spent), should_offload(e, c));
    }
}

VTEST(the_start_is_part_of_the_card_path) {
  Costs c;
  Estimate e = fifty_ms();
  const double warm = card_path_seconds(e, c);
  e.start_s = 0.25;
  VCHECK(card_path_seconds(e, c) > warm + 0.2499 && card_path_seconds(e, c) < warm + 0.2501);
}

VTEST(the_first_launch_rule_pays_the_start_too) {   // the static estimate of the gemm shape is 5 ms
  Costs c;
  VCHECK(first_launch_offload(gemm_shape(), c));                    // (default: the driver is up)
  VCHECK(!first_launch_offload(gemm_shape(), c, false, 0.0));       // a first launch of 5 ms estimated cannot start a driver
  FirstLaunch f = gemm_shape();
  f.static_instrs = 60000;                                          // 1000x the code: 5 s estimated
  VCHECK(first_launch_offload(f, c, false, 0.0));                   // that launch repays it
  // or the program has spent as much in the simulator already, and the launch saves its share of the start (12.5 ms for a static estimate: the gemm
  // shape's 2.5 ms does not)
  VCHECK(!first_launch_offload(gemm_shape(), c, false, c.start_cost_s));
  f.static_instrs = 240;                                            // 4x the code: 19.7 ms estimated, 17.3 ms saved
  VCHECK(first_launch_offload(f, c, false, c.start_cost_s));
  VCHECK(!first_launch_offload(f, c, false, c.start_cost_s * 0.99));
  Costs before = c;
  before.start_launches = 0;                                        // (VGPU_HOST_GPU_START_LAUNCHES=0: the account alone, as before)
  VCHECK(first_launch_offload(gemm_shape(), before, false, c.start_cost_s));
}

// ---- the share of the start a launch must save when only the account would send it (integration of sp64-sp68) ----
// A launch of `sim_s` with no copies: its card path with the driver up is launch_s + kernel_s_default (107 us).
static Estimate bare(double sim_s) {
  Estimate e;
  e.sim_s = sim_s;
  return e;
}

VTEST(the_start_share_is_five_ms_by_default) {
  const Costs c;
  VCHECK(c.start_launches == 50);
  VCHECK(c.start_cost_s / c.start_launches == 0.005);
}

VTEST(a_launch_the_account_sends_saves_its_share_of_the_start) {
  const Costs c;
  const double card = c.launch_s + c.kernel_s_default;
  // exactly at the share it goes (the share is "at least"), just under it stays
  VCHECK(should_offload_with_start(bare(0.005 + card + 1e-9), c, false, c.start_cost_s));
  VCHECK(!should_offload_with_start(bare(0.005 + card - 1e-9), c, false, c.start_cost_s));
  VCHECK(start_share_saved(bare(0.005 + card + 1e-9), c) && !start_share_saved(bare(0.005 + card - 1e-9), c));
  // exactly at the share, with every number exact in binary: 0.375 s simulated against a card path of 0.125 s saves 0.25 s, the share of 0.5 s in 2
  Costs exact = c;
  exact.launch_s = 0;
  exact.kernel_s_default = 0.125;
  exact.start_cost_s = 0.5;
  exact.start_launches = 2;
  VCHECK(bare(0.375).sim_s - card_path_seconds(bare(0.375), exact) == exact.start_cost_s / exact.start_launches);
  VCHECK(start_share_saved(bare(0.375), exact) && should_offload_with_start(bare(0.375), exact, false, 0.5));
  // the account must still be full: the share alone buys nothing
  VCHECK(!should_offload_with_start(bare(0.1), c, false, c.start_cost_s * 0.99));
  VCHECK(should_offload_with_start(bare(0.1), c, false, c.start_cost_s));
  // and a launch that repays the start by itself needs neither
  VCHECK(should_offload_with_start(bare(1.0), c, false, 0.0));
  // what the launch moves counts against what it saves: 7.5 ms with 8 MiB each way saves 5.9 ms, with 16 MiB each way 4.3 ms
  Estimate e = bare(0.0075);
  e.push_bytes = e.pull_bytes = 8ull << 20;
  VCHECK(should_offload(e, c) && should_offload_with_start(e, c, false, c.start_cost_s));
  e.push_bytes = e.pull_bytes = 16ull << 20;
  VCHECK(should_offload(e, c) && !should_offload_with_start(e, c, false, c.start_cost_s));
  VCHECK(should_offload_with_start(e, c, true, 0.0));   // with the driver up, the plain rule
  // the share is the start's: twice the start, twice the share
  Costs dear = c;
  dear.start_cost_s = 0.5;
  VCHECK(should_offload_with_start(bare(0.008), c, false, 10.0) && !should_offload_with_start(bare(0.008), dear, false, 10.0));
  Costs few = c;
  few.start_launches = 10;   // repaid within 10 launches: 25 ms a launch
  VCHECK(!should_offload_with_start(bare(0.02), few, false, 10.0) && should_offload_with_start(bare(0.026), few, false, 10.0));
}

VTEST(the_four_samples_that_lost_the_start_in_sp66_no_longer_buy_it) {   // A10G, Round sp66: what the launch that bought the start was
  const Costs c;
  VCHECK(!should_offload_with_start(bare(0.0005), c, false, 0.45));   // LargeKernelParameter: under 0.5 ms a launch
  VCHECK(!should_offload_with_start(bare(0.0035), c, false, 0.26));   // threadFenceReduction: about 3.5 ms
  VCHECK(!should_offload_with_start(bare(0.0038), c, false, 0.26));   // dct8x8: 3.8 ms
  FirstLaunch pi;   // a first launch whose static estimate is MC_EstimatePiInlineP's last kernel's (3.0 ms, logged on server3), after a 3.3 s launch
  pi.blocks = 640;
  pi.threads_per_block = 256;
  pi.static_instrs = 59;
  pi.back_edge = true;
  pi.multiprocessors = 72;
  pi.live_bytes = 1u << 20;
  VCHECK(static_sim_seconds(pi.blocks, pi.threads_per_block, pi.static_instrs, c) > 0.003 && first_launch_offload(pi, c, true));
  VCHECK(!first_launch_offload(pi, c, false, 3.3));
  Costs before = c;
  before.start_launches = 0;   // all four bought it under the account alone
  for (double s : {0.0005, 0.0035, 0.0038}) VCHECK(should_offload_with_start(bare(s), before, false, 0.45));
  VCHECK(first_launch_offload(pi, before, false, 3.3));
}

VTEST(the_programs_whose_start_the_account_bought_and_repaid_still_buy_it) {   // the launch that bought it (server3 and the rig, see host_gpu.hpp)
  const Costs c;
  VCHECK(should_offload_with_start(bare(0.028), c, false, 0.27));   // FDTD-2D, 28 ms on server3
  VCHECK(should_offload_with_start(bare(0.023), c, false, 0.25));   // LU
  VCHECK(should_offload_with_start(bare(0.0186), c, false, 0.26));  // GRAMSCHM
  VCHECK(should_offload_with_start(bare(0.020), c, false, 0.25));   // gaussian on the rig (5.4 ms on server3, whose simulator is about 4x as fast)
  VCHECK(should_offload_with_start(bare(0.008), c, false, 0.26));   // asyncAllocation (turbo) on the rig: 800 launches in 6.6 s
  VCHECK(!should_offload_with_start(bare(0.0019), c, false, 0.26)); // ... and on server3, 1.9 ms: a faster simulator buys fewer starts
  FirstLaunch mm3 = gemm_shape();                                    // 3MM's last kernel: 19 ms estimated, 0.32 s spent
  mm3.static_instrs = 232;
  VCHECK(first_launch_offload(mm3, c, false, 0.32));
}

VTEST(a_first_launch_saves_a_larger_share_than_a_launch_with_a_history) {   // its estimate is static: 12.5 ms against 5 ms
  const Costs c;
  VCHECK(c.start_launches_static == 20 && c.start_cost_s / c.start_launches_static == 0.0125);
  FirstLaunch f = gemm_shape();
  f.live_bytes = 0;   // (the card path: launch_s + kernel_s_default, as bare())
  const double card = c.launch_s + c.kernel_s_default;
  const auto with_instrs_for = [&](double sim_s) {   // the static instruction count whose estimate is closest to sim_s from above
    FirstLaunch g = f;
    const double per_instr = static_sim_seconds(g.blocks, g.threads_per_block, 1, c);
    g.static_instrs = static_cast<uint64_t>(std::ceil(sim_s / per_instr));
    return g;
  };
  // dct8x8's quantization kernel in turbo mode: 8.5 ms static, 8.4 ms saved; a launch with a history of 8.5 ms would buy the start, this one does not
  VCHECK(!first_launch_offload(with_instrs_for(0.0085), c, false, 0.3));
  VCHECK(should_offload_with_start(bare(0.0085), c, false, 0.3));
  // HeCBench atomicPerf (15.5 ms) and 3MM (19 ms) still buy it
  VCHECK(first_launch_offload(with_instrs_for(0.0155), c, false, 0.26));
  VCHECK(first_launch_offload(with_instrs_for(0.019), c, false, 0.32));
  // the bar: 12.5 ms saved (the largest count whose estimate is under it, and one more)
  const double per_instr = static_sim_seconds(f.blocks, f.threads_per_block, 1, c);
  FirstLaunch below = f, above = f;
  below.static_instrs = static_cast<uint64_t>(std::floor((0.0125 + card) / per_instr));
  above.static_instrs = below.static_instrs + 1;
  VCHECK(static_sim_seconds(below.blocks, below.threads_per_block, below.static_instrs, c) - card < 0.0125);
  VCHECK(static_sim_seconds(above.blocks, above.threads_per_block, above.static_instrs, c) - card >= 0.0125);
  VCHECK(first_launch_offload(above, c, false, 1.0) && !first_launch_offload(below, c, false, 1.0));
  VCHECK(first_launch_offload(below, c, true, 0.0));   // with the driver up, the plain rule
  // the static share is its own: a launch with a history needs only 5 ms
  Costs other = c;
  other.start_launches_static = 50;
  VCHECK(first_launch_offload(with_instrs_for(0.0085), other, false, 0.3));
  other = c;
  other.start_launches = 10;   // (25 ms for a launch with a history; the first launch's share stays 12.5 ms)
  VCHECK(first_launch_offload(with_instrs_for(0.019), other, false, 0.32) && !should_offload_with_start(bare(0.019), other, false, 0.32));
  // and VGPU_HOST_GPU_START_LAUNCHES=0 asks neither
  other = c;
  other.start_launches = 0;
  VCHECK(first_launch_offload(with_instrs_for(0.0085), other, false, 0.3));
}

VTEST(a_start_share_of_zero_is_the_account_alone) {   // VGPU_HOST_GPU_START_LAUNCHES=0: the rule from sp43 to sp68
  Costs c;
  c.start_launches = 0;
  for (double s : {50e-6, 0.0005, 0.001, 0.0035, 0.006, 0.01, 0.05, 0.4, 4.0})
    for (double spent : {0.0, 0.1, 0.25, 5.0})
      for (bool up : {false, true}) {
        Estimate e = bare(s);
        Estimate cold = e;
        cold.start_s = c.start_cost_s;
        const bool old_rule = should_offload(e, c) && (up || should_offload(cold, c) || spent >= c.start_cost_s);
        VCHECK_EQ(should_offload_with_start(e, c, up, spent), old_rule);
      }
  c.start_launches = -1;   // (a negative count is no requirement either)
  VCHECK(should_offload_with_start(bare(0.001), c, false, 1.0));
}

// ---- a kernel launched many times (Round sp69; host_gpu.hpp start_share_launches): the start repaid within the launches of it seen so far ----
VTEST(the_start_is_repaid_within_fifty_launches_until_a_kernel_has_run_two_hundred_times) {
  const Costs c;
  VCHECK(c.start_repeats == 200 && c.start_repeats_max == 500 && c.min_sim_s == 200e-6 && c.kernel_prior_ratio == 0 && c.ban_strikes == 1);   // (the rule before #660)
  for (uint64_t kept : {0ull, 1ull, 50ull, 51ull, 199ull}) VCHECK(start_share_launches(c, kept) == 50);
  VCHECK(start_share_launches(c, 200) == 200);
  VCHECK(start_share_launches(c, 201) == 201);
  VCHECK(start_share_launches(c, 499) == 499 && start_share_launches(c, 500) == 500);
  VCHECK(start_share_launches(c, 501) == 500 && start_share_launches(c, 1000000) == 500);   // at most 500 count: 0.5 ms saved at the least
  VCHECK(c.start_cost_s / start_share_launches(c, 1000000) == 0.0005);
  Costs more = c;
  more.start_launches = 300;   // never fewer launches than start_launches
  VCHECK(start_share_launches(more, 199) == 300 && start_share_launches(more, 250) == 300 && start_share_launches(more, 301) == 301);
  more.start_launches = 600;   // ... even above the most that count
  VCHECK(start_share_launches(more, 100000) == 600);
  Costs longer = c;
  longer.start_repeats_max = 2000;
  VCHECK(start_share_launches(longer, 1500) == 1500 && start_share_launches(longer, 2500) == 2000);
  Costs sooner = c;
  sooner.start_repeats = 60;
  VCHECK(start_share_launches(sooner, 59) == 50 && start_share_launches(sooner, 60) == 60);
  Costs off = c;
  off.start_repeats = 0;       // VGPU_HOST_GPU_START_REPEATS=0
  for (uint64_t kept : {0ull, 199ull, 200ull, 100000ull}) VCHECK(start_share_launches(off, kept) == 50);
  off.start_repeats = -1;
  VCHECK(start_share_launches(off, 100000) == 50);
}

VTEST(many_short_launches_of_one_kernel_buy_the_start_when_as_many_again_repay_it) {
  const Costs c;
  // Rodinia hotspot on the A10G rig: 1,000 launches of 1.04 ms, each saving 0.933 ms (card path 107 us): 268 of them repay 0.25 s
  const Estimate hs = bare(0.00104);
  VCHECK(!should_offload_with_start(hs, c, false, 0.30));         // (the default: no launch of the kernel seen)
  VCHECK(!should_offload_with_start(hs, c, false, 0.30, 199));
  VCHECK(!should_offload_with_start(hs, c, false, 0.30, 200));    // 200 are enough to count on, but would repay 0.187 s
  VCHECK(!should_offload_with_start(hs, c, false, 0.30, 267));
  VCHECK(should_offload_with_start(hs, c, false, 0.30, 268));
  VCHECK(should_offload_with_start(hs, c, false, 0.30, 999));
  VCHECK(start_share_saved(hs, c, 268) && !start_share_saved(hs, c, 267) && !start_share_saved(hs, c));
  // the account is still asked
  VCHECK(!should_offload_with_start(hs, c, false, c.start_cost_s * 0.99, 999));
  VCHECK(should_offload_with_start(hs, c, false, c.start_cost_s, 999));
  // the bar is exactly the start over the launches seen (binary-exact numbers: a start of 0.25 s, 256 launches, 1/1024 s saved each)
  Costs exact = c;
  exact.launch_s = 0;
  exact.kernel_s_default = 1.0 / 1024;
  VCHECK(bare(2.0 / 1024).sim_s - card_path_seconds(bare(2.0 / 1024), exact) == exact.start_cost_s / 256);
  exact.margin = 1.5;
  VCHECK(should_offload_with_start(bare(2.0 / 1024), exact, false, 1.0, 256));
  VCHECK(!should_offload_with_start(bare(2.0 / 1024), exact, false, 1.0, 255));
  // HeCBench asyncAllocation on server3 (1.9 ms a launch, 800 launches): 200 launches seen repay 0.36 s
  VCHECK(!should_offload_with_start(bare(0.0019), c, false, 0.26, 199) && should_offload_with_start(bare(0.0019), c, false, 0.38, 200));
  // a longer start needs more launches seen, in proportion (0.4 s / 0.933 ms: 429), and one that 500 launches do not repay is not bought
  Costs dear = c;
  dear.start_cost_s = 0.4;
  VCHECK(!should_offload_with_start(hs, dear, false, 1.0, 428) && should_offload_with_start(hs, dear, false, 1.0, 429));
  dear.start_cost_s = 0.5;
  VCHECK(!should_offload_with_start(hs, dear, false, 1.0, 536) && !should_offload_with_start(hs, dear, false, 1.0, 100000));
  // at most 500 launches count: a launch that saves under 0.5 ms does not buy the start however often its kernel ran
  VCHECK(!should_offload_with_start(bare(0.0006), c, false, 1.0, 100000));   // 0.493 ms saved
  VCHECK(should_offload_with_start(bare(0.00061), c, false, 1.0, 500) && !should_offload_with_start(bare(0.00061), c, false, 1.0, 496));   // 0.503 ms: 498 repay it
  // a launch not worth a card with the driver up is not sent however often its kernel ran (PolyBench JACOBI1D: 20,000 launches of 0.12 ms)
  VCHECK(!should_offload(bare(0.00012), c) && !should_offload_with_start(bare(0.00012), c, false, 2.0, 19000));
  // with the driver up the count is not asked
  VCHECK(should_offload_with_start(hs, c, true, 0.0, 0));
}

VTEST(the_four_samples_still_do_not_buy_the_start_with_their_launch_counts) {   // A10G: launches of each kernel, what each takes
  const Costs c;
  for (uint64_t kept = 0; kept <= 101; ++kept) VCHECK(!should_offload_with_start(bare(0.0035), c, false, 0.36, kept));    // threadFenceReduction: 101 launches
  for (uint64_t kept = 0; kept <= 118; ++kept) VCHECK(!should_offload_with_start(bare(0.0038), c, false, 0.97, kept));    // dct8x8: 118 launches in all
  // LargeKernelParameter: two kernels, 1,001 launches each, 0.19 and 0.28 ms a launch (0.47 s in 2,002): 0.17 ms saved at the most
  for (uint64_t kept = 0; kept <= 1001; ++kept) VCHECK(!should_offload_with_start(bare(0.00028), c, false, 0.47, kept));
  VCHECK(!should_offload_with_start(bare(0.00056), c, false, 0.47, 1001));   // ... nor on a host twice as slow (0.453 ms saved)
  // (MC_EstimatePiInlineP's launch is a first launch: first_launch_offload takes no count)
}

VTEST(start_repeats_of_zero_is_the_share_of_fifty_launches_whatever_the_history) {   // VGPU_HOST_GPU_START_REPEATS=0: the rule of sp64-sp68
  Costs off;
  off.start_repeats = 0;
  const Costs c;
  for (double s : {50e-6, 0.0005, 0.001, 0.0035, 0.0052, 0.006, 0.05, 0.4})
    for (double spent : {0.0, 0.1, 0.25, 5.0})
      for (bool up : {false, true})
        for (uint64_t kept : {0ull, 199ull, 200ull, 5000ull, 1000000ull}) {
          VCHECK_EQ(should_offload_with_start(bare(s), off, up, spent, kept), should_offload_with_start(bare(s), c, up, spent));
          // and the count only ever adds launches: what the share of fifty sends is sent whatever the count
          if (should_offload_with_start(bare(s), c, up, spent)) { VCHECK(should_offload_with_start(bare(s), c, up, spent, kept)); }
        }
  VCHECK(!should_offload_with_start(bare(0.00104), off, false, 0.30, 999));
  Costs none = c;   // VGPU_HOST_GPU_START_LAUNCHES=0 is still the account alone
  none.start_launches = 0;
  VCHECK(should_offload_with_start(bare(0.0005), none, false, 0.25, 0) && should_offload_with_start(bare(0.0005), none, false, 0.25, 5000));
}

// ---- the division idiom (host_gpu.hpp division_idiom_only), on the shape ptxas 13.0 gave `a / b` for sm_86 (cuobjdump of an own kernel) ----
using K = CodeInsn::Kind;
static CodeInsn ci(K kind, uint64_t pc, int64_t target = -1, int src0 = -1, int src1 = -1) {
  CodeInsn c;
  c.kind = kind;
  c.pc = pc;
  c.target = target;
  c.src0 = src0;
  c.src1 = src1;
  return c;
}
// main line: ... MUFU.RCP R8, R3 ; FCHK P0, R0, R3 ; FFMA x5 ; @!P0 BRA ; MOV ; CALL -> slow ; ... EXIT ; slow: ALU..., MUFU.RCP R8, R3, ALU..., RET
static std::vector<CodeInsn> division_kernel() {
  std::vector<CodeInsn> c;
  c.reserve(64);   // (GCC 13 at -O3 warns of a write past the end in the growth path of push_back here, a false positive under -Werror)
  uint64_t pc = 0;
  const auto add = [&](K k, int64_t target = -1, int s0 = -1, int s1 = -1) { c.push_back(ci(k, pc, target, s0, s1)); pc += 16; };
  add(K::Other);                    // LDG is Memory only inside a subroutine: here it is the kernel's own load
  add(K::Other);
  add(K::MufuRcp, -1, 3);           // MUFU.RCP R8, R3
  add(K::Fchk, -1, 0, 3);           // FCHK P0, R0, R3
  for (int i = 0; i < 5; ++i) add(K::Other);   // FFMAs
  add(K::Other);                    // @!P0 BRA
  add(K::Other);                    // MOV R4, return
  add(K::Call, 0);                  // target patched below
  add(K::Other);
  add(K::Other);                    // STG
  add(K::Other);                    // EXIT
  const size_t slow_at = c.size();
  const uint64_t slow_pc = pc;
  add(K::Other);
  add(K::Other);
  add(K::MufuRcp, -1, 3);           // the slow path's own MUFU.RCP, no FCHK
  add(K::Other);
  add(K::Ret);
  for (CodeInsn& x : c)
    if (x.kind == K::Call) x.target = static_cast<int64_t>(slow_pc);
  (void)slow_at;
  return c;
}

VTEST(division_idiom_is_recognized) {
  VCHECK(division_idiom_only(division_kernel()));
}

VTEST(division_idiom_refuses_a_bare_reciprocal) {
  // rcp.approx.ftz.f32: one MUFU.RCP and nothing that makes the result independent of its last bit
  VCHECK(!division_idiom_only({ci(K::Other, 0), ci(K::MufuRcp, 16, -1, 3), ci(K::Other, 32), ci(K::Other, 48)}));
  // an FCHK of another register is not the divisor's
  auto c = division_kernel();
  for (CodeInsn& x : c)
    if (x.kind == K::Fchk) x.src1 = 4;
  VCHECK(!division_idiom_only(c));
  // an FCHK too far after the MUFU
  c = division_kernel();
  for (size_t i = 0; i < c.size(); ++i)
    if (c[i].kind == K::MufuRcp) {
      std::vector<CodeInsn> d(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(i + 1));
      for (unsigned k = 0; k < kIdiomMufuToFchk; ++k) d.push_back(ci(K::Other, 1000 + k));
      d.push_back(ci(K::Fchk, 2000, -1, 0, 3));
      VCHECK(!division_idiom_only(d));
      break;
    }
}

VTEST(division_idiom_refuses_other_sfu_and_calls) {
  auto c = division_kernel();
  c[0].kind = K::MufuOther;                        // an EX2 anywhere
  VCHECK(!division_idiom_only(c));
  c = division_kernel();
  for (CodeInsn& x : c)
    if (x.kind == K::Fchk) x.kind = K::Other;       // no FCHK: the CALL is not a division's slow path
  VCHECK(!division_idiom_only(c));
  c = division_kernel();
  c.back() = ci(K::Other, c.back().pc);             // a subroutine that never returns
  VCHECK(!division_idiom_only(c));
  c = division_kernel();
  c[c.size() - 2].kind = K::Memory;                 // a subroutine that touches memory is not the slow path
  VCHECK(!division_idiom_only(c));
  c = division_kernel();
  c.insert(c.begin() + 1, ci(K::Ret, 8));           // a RET outside any subroutine
  VCHECK(!division_idiom_only(c));
  c = division_kernel();
  for (CodeInsn& x : c)
    if (x.kind == K::Call) x.target = 9999;         // a call to an address that is not an instruction
  VCHECK(!division_idiom_only(c));
}

VTEST(division_idiom_refuses_a_device_function_call) {
  // a __noinline__ function: a CALL to a subroutine and a RET, with no division and no FCHK anywhere -- not the slow path, so not the idiom
  VCHECK(!division_idiom_only({ci(K::Other, 0), ci(K::Call, 16, 64), ci(K::Other, 32), ci(K::Other, 48), ci(K::Other, 64), ci(K::Ret, 80)}));
  // with an FCHK far before the call (outside kIdiomFchkToCall instructions) it is still not
  std::vector<CodeInsn> c = {ci(K::Fchk, 0, -1, 0, 1)};
  for (unsigned k = 1; k <= kIdiomFchkToCall + 1; ++k) c.push_back(ci(K::Other, 16 * k));
  const uint64_t call_pc = 16 * (kIdiomFchkToCall + 2);
  c.push_back(ci(K::Call, call_pc, static_cast<int64_t>(call_pc + 32)));
  c.push_back(ci(K::Other, call_pc + 16));
  c.push_back(ci(K::Other, call_pc + 32));
  c.push_back(ci(K::Ret, call_pc + 48));
  VCHECK(!division_idiom_only(c));
}

VTEST(division_idiom_takes_the_nan_generator_only_in_the_slow_path) {
  auto c = division_kernel();
  c[c.size() - 2].kind = K::NanGen;                 // inside the subroutine: the slow path's 0/0 case
  VCHECK(division_idiom_only(c));
  c = division_kernel();
  c[0].kind = K::NanGen;                            // in the kernel's own code: a bare NaN constant through an SFU instruction
  VCHECK(!division_idiom_only(c));
}

VTEST(division_idiom_takes_two_divisions_sharing_a_slow_path) {
  auto one = division_kernel();
  // a second MUFU.RCP/FCHK/CALL group before the slow path, calling the same subroutine
  std::vector<CodeInsn> c;
  uint64_t slow_pc = 0;
  for (const CodeInsn& x : one)
    if (x.kind == K::Call) slow_pc = static_cast<uint64_t>(x.target);
  for (const CodeInsn& x : one) {
    if (x.pc == slow_pc) {
      c.push_back(ci(K::MufuRcp, slow_pc - 1000, -1, 6));
      c.push_back(ci(K::Fchk, slow_pc - 999, -1, 5, 6));
      c.push_back(ci(K::Call, slow_pc - 998, static_cast<int64_t>(slow_pc)));
    }
    c.push_back(x);
  }
  VCHECK(division_idiom_only(c));
}

VTEST(idiom_classification_by_mnemonic) {
  VCHECK(classify_for_idiom("MUFU", {"RCP"}) == K::MufuRcp);
  VCHECK(classify_for_idiom("MUFU", {"RCP64H"}) == K::MufuOther);
  VCHECK(classify_for_idiom("MUFU", {"EX2"}) == K::MufuOther);
  VCHECK(classify_for_idiom("MUFU", {"RCP", "F16"}) == K::MufuOther);
  VCHECK(classify_for_idiom("MUFU", {"RSQ"}) == K::MufuOther);                  // an RSQ of a register is an SFU result
  VCHECK(classify_for_idiom("MUFU", {"RSQ"}, true) == K::NanGen);                // of an immediate NaN it is the division's NaN generator
  VCHECK(classify_for_idiom("MUFU", {"EX2"}, true) == K::MufuOther);
  VCHECK(classify_for_idiom("MUFU", {"RCP"}, true) == K::MufuRcp);
  VCHECK(classify_for_idiom("FCHK", {}) == K::Fchk);
  VCHECK(classify_for_idiom("CALL", {"REL", "NOINC"}) == K::Call);
  VCHECK(classify_for_idiom("RET", {"REL", "NODEC"}) == K::Ret);
  VCHECK(classify_for_idiom("LDG", {"E"}) == K::Memory);
  VCHECK(classify_for_idiom("SHFL", {}) == K::Memory);
  VCHECK(classify_for_idiom("HMMA", {}) == K::Memory);        // not in the exact set: not arithmetic of the slow path
  VCHECK(classify_for_idiom("FFMA", {"RM"}) == K::Other);
  VCHECK(classify_for_idiom("SHF", {"R", "U32", "HI"}) == K::Other);
  // the default exact set itself is unchanged by all of this
  VCHECK(!exact_opcode_allowed("MUFU", {"RCP"}));
  VCHECK(!exact_opcode_allowed("CALL", {"REL", "NOINC"}));
  VCHECK(!exact_opcode_allowed("RET", {"REL", "NODEC"}));
}

// The start-up refusals (08 section 3): a mode value that is not a mode is said, and every diagnostic that only the simulator gives switches the mode off.
VTEST(mode_text_known_and_diagnostic_switches) {
  VCHECK(mode_text_is_known(nullptr) && mode_text_is_known("") && mode_text_is_known("off") && mode_text_is_known("exact") && mode_text_is_known("0"));
  VCHECK(mode_text_is_known("turbo"));
  VCHECK(!mode_text_is_known("approx") && !mode_text_is_known("EXACT") && !mode_text_is_known("on") && !mode_text_is_known("1") && !mode_text_is_known("Turbo"));
  for (const char* v : kIncompatibleVars) {
    const std::string name(v);
    VCHECK(!diagnostic_switched_on(name, nullptr));
    VCHECK(!diagnostic_switched_on(name, ""));
    VCHECK(!diagnostic_switched_on(name, "0"));
    if (name != "VGPU_SCHEDULER") VCHECK(diagnostic_switched_on(name, "1"));
  }
  // the scheduler's default name is what a config file writes for `scheduler: deterministic`: not a diagnostic; the others are
  VCHECK(!diagnostic_switched_on("VGPU_SCHEDULER", "deterministic"));
  VCHECK(diagnostic_switched_on("VGPU_SCHEDULER", "random") && diagnostic_switched_on("VGPU_SCHEDULER", "adversarial"));
  // every way the simulator can be told to fail or look inside a launch is on the list
  const char* must[] = {"VGPU_COUNTERS", "VGPU_SASS_TRACE", "VGPU_KERNEL_DIGEST", "VGPU_TRACE", "VGPU_MAX_STEPS", "VGPU_FAIL_ALLOC", "VGPU_FAIL_LAUNCH",
                        "VGPU_FAIL_MEMCPY", "VGPU_RACE", "VGPU_STRICT", "VGPU_SCHEDULER"};
  for (const char* m : must) {
    bool found = false;
    for (const char* v : kIncompatibleVars) found = found || std::string(v) == m;
    if (!found) std::fprintf(stderr, "%s is not in kIncompatibleVars\n", m);
    VCHECK(found);
  }
}

// ---- turbo mode (docs/perf-design/12-host-gpu-turbo.md) -------------------------------------------------------------------------------------

// Exact mode is what it was: the mnemonics it takes are this list and no other name the decoder can print. The list is a copy made when turbo mode
// was added; a change to exact_mnemonic_allowed has to change it here too, on purpose.
VTEST(exact_set_is_the_list_it_was_before_turbo_mode) {
  const std::set<std::string> frozen = {
      "IMAD", "IADD3", "IADD", "LOP3", "LOP", "LEA", "SHF", "SHL", "SHR", "ISETP", "IMNMX", "IABS", "POPC", "FLO", "BREV", "PRMT", "SEL", "MOV", "R2P", "P2R",
      "PLOP3", "UPLOP3", "PSETP", "ISET", "FSET", "DSET", "I2I", "I2IP", "SGXT", "BMSK", "VIADD", "VIMNMX", "FADD", "FMUL", "FFMA", "FSETP", "FSEL", "FMNMX",
      "DADD", "DMUL", "DFMA", "DSETP", "DMNMX", "I2F", "I2FP", "F2I", "F2F", "F2FP", "FRND", "FCHK", "LDG", "STG", "LDS", "STS", "LDL", "STL", "LD", "ST",
      "LDC", "ULDC", "S2R", "S2UR", "CS2R", "UMOV", "UIADD3", "UIMAD", "ULOP3", "USHF", "ULEA", "UISETP", "USEL", "UPRMT", "UFLO", "UPOPC", "UP2UR", "R2UR",
      "VOTEU", "BRA", "BRX", "BSSY", "BSYNC", "BREAK", "BMOV", "EXIT", "NOP", "YIELD", "WARPSYNC", "BAR", "MEMBAR", "ERRBAR", "DEPBAR", "CCTL", "SHFL", "VOTE",
      "MATCH", "RED"};
  VCHECK_EQ(frozen.size(), size_t{96});
  for (const std::string& m : frozen) VCHECK(exact_opcode_allowed(m, {}));
  // every instruction the decoder knows, and a few names it does not
  std::vector<std::string> names = {"REDG", "OMMA", "VPRINTF", "", "imad", "MUFU.RCP"};
#define VGPU_SASS_OP(name) names.push_back(#name);
#include "vgpu/sass/ops.inc"
#undef VGPU_SASS_OP
  VCHECK(names.size() > 150);
  unsigned taken = 0;
  for (const std::string& m : names) {
    const bool want = frozen.count(m) != 0;
    if (exact_opcode_allowed(m, {}) != want) std::fprintf(stderr, "%s: exact mode %s it\n", m.c_str(), want ? "no longer takes" : "now takes");
    VCHECK_EQ(exact_opcode_allowed(m, {}), want);
    taken += want;
  }
  VCHECK_EQ(taken, 89u);   // (the other 7 are spellings that are no instruction of the decoder: LOP, SHL, SHR, ISET, DSET, DMNMX, PSETP)
  // RED's modifiers, as before
  for (const char* bad : {"F16", "F32", "F64", "BF16", "FTZ", "CAS", "EXCH", "INC", "DEC", "SPIN", "CAST", "POPC"}) VCHECK(!exact_opcode_allowed("RED", {"E", bad}));
  for (const char* good : {"E", "ADD", "MIN", "MAX", "AND", "OR", "XOR", "S32", "64", "S64", "STRONG", "GPU"}) VCHECK(exact_opcode_allowed("RED", {good}));
}

VTEST(turbo_classes_every_kind_of_instruction) {
  using C = InsnClass;
  // exact: whatever the exact set takes
  for (const char* m : {"IMAD", "FFMA", "DFMA", "LDG", "STG", "BAR", "BRA", "EXIT", "SHFL", "FCHK", "S2R"}) VCHECK(turbo_class(m, {}) == C::Exact);
  VCHECK(turbo_class("RED", {"E", "ADD", "STRONG", "GPU"}) == C::Exact);
  // approximate
  for (const char* f : {"COS", "SIN", "EX2", "LG2", "RCP", "RSQ", "RCP64H", "RSQ64H", "SQRT", "TANH"}) VCHECK(turbo_class("MUFU", {f}) == C::Mufu);
  VCHECK(turbo_class("MUFU", {"(10)"}) == C::Unlisted);   // a function the decoder cannot name: the card runs it all the same (turbo fit)
  for (const char* m : {"HADD2", "HMUL2", "HFMA2", "HSETP2", "HSET2", "HMNMX2"}) VCHECK(turbo_class(m, {}) == C::Half);
  VCHECK(turbo_class("HFMA2", {"MMA"}) == C::Half);
  VCHECK(turbo_class("CALL", {"REL", "NOINC"}) == C::Call);
  VCHECK(turbo_class("RET", {"REL", "NODEC"}) == C::Call);
  // the decoder's own spellings of a float reduction: the type is ONE modifier, "F32.FTZ.RN"
  VCHECK(turbo_class("RED", {"E", "ADD", "F32.FTZ.RN", "STRONG", "GPU"}) == C::FloatRed);
  VCHECK(turbo_class("RED", {"E", "ADD", "F64.RN", "STRONG", "GPU"}) == C::FloatRed);
  VCHECK(turbo_class("RED", {"E", "ADD", "F16x2.RN"}) == C::FloatRed);
  VCHECK(turbo_class("REDG", {"E", "ADD", "F32x2.FTZ.RN"}) == C::FloatRed);
  VCHECK(turbo_class("REDG", {"E", "MAX", "BF16x4.RN"}) == C::FloatRed);
  VCHECK(turbo_class("RED", {"E", "ADD", "F32", "FTZ", "RN"}) == C::FloatRed);   // and split
  // the two modes agree on what a float reduction is: exact refuses exactly the reductions turbo does not class as exact
  for (const std::vector<std::string>& mods : std::vector<std::vector<std::string>>{{"E", "ADD", "F32.FTZ.RN"}, {"E", "ADD", "F64.RN"}, {"ADD", "F16x2.RN"}, {"E", "INC"}, {"E", "EXCH"},
                                                                                     {"E", "ADD"}, {"E", "MAX", "S32"}, {"E", "AND", "64"}})
    VCHECK_EQ(exact_opcode_allowed("RED", mods), turbo_class("RED", mods) == C::Exact);
  VCHECK(turbo_class("REDG", {"E", "ADD", "STRONG", "GPU"}) == C::Extra);
  VCHECK(turbo_class("RED", {"E", "INC"}) == C::AtomInt);
  VCHECK(turbo_class("RED", {"E", "EXCH"}) == C::AtomSwap);
  // tensor
  for (const char* m : {"HMMA", "IMMA", "DMMA"}) VCHECK(turbo_class(m, {"16816", "F32"}) == C::Tensor);
  // BMMA (#661): integer popcount arithmetic, measured equal on an A10G; the exact set does not hold it
  for (const char* op : {"XOR", "AND"}) {
    VCHECK(turbo_class("BMMA", {"88128", op, "POPC"}) == C::Tensor);
    VCHECK(!exact_opcode_allowed("BMMA", {"88128", op, "POPC"}));
  }
  VCHECK(!exact_opcode_allowed("BMMA", {}));
  VCHECK(turbo_class("BMMA", {}) == C::Tensor);
  VCHECK(turbo_class("BMMA", {"88128", "XOR", "(9)"}) == C::Unlisted);   // an undecoded field is unlisted, not tensor
  VCHECK(turbo_class("QMMA", {}) == C::Unlisted);                        // its neighbour is unnamed (turbo fit: the card runs it where it has it)
  // atomics
  for (const char* m : {"ATOM", "ATOMG", "ATOMS"}) {
    for (const char* op : {"ADD", "MIN", "MAX", "INC", "DEC", "AND", "OR", "XOR"}) {
      VCHECK(turbo_class(m, {"E", op}) == C::AtomInt);
      VCHECK(turbo_class(m, {"E", op, "S32"}) == C::AtomInt);
      VCHECK(turbo_class(m, {op, "64", "STRONG", "GPU"}) == C::AtomInt);
    }
    VCHECK(turbo_class(m, {"E", "ADD", "F32.FTZ.RN"}) == C::AtomFloat);
    VCHECK(turbo_class(m, {"E", "ADD", "F64.RN"}) == C::AtomFloat);
    VCHECK(turbo_class(m, {"E", "MIN", "F16x2.RN"}) == C::AtomFloat);
    VCHECK(turbo_class(m, {"E", "CAS", "STRONG", "GPU"}) == C::AtomSwap);
    VCHECK(turbo_class(m, {"CAST", "SPIN", "64"}) == C::AtomSwap);
    VCHECK(turbo_class(m, {"E", "EXCH"}) == C::AtomSwap);
    VCHECK(turbo_class(m, {"E", "EXCH", "F32.FTZ.RN"}) == C::AtomSwap);   // an exchange is an exchange whatever its type
    VCHECK(turbo_class(m, {"E", "(9)"}) == C::Unlisted);
  }
  VCHECK(turbo_class("ATOMS", {"POPC", "INC", "32"}) == C::AtomInt);
  VCHECK(turbo_class("ATOMS", {"ARRIVE", "64"}) == C::Unlisted);
  // extra: named one by one
  for (const char* m : {"IDP", "VABSDIFF", "VABSDIFF4", "VIADDMNMX", "VIMNMX3", "FMNMX3", "F2IP", "FSWZADD", "REDUX", "LDGSTS", "LDGDEPBAR", "LDSM", "MOVM",
                        "FFMA2", "FMUL2", "FADD2", "B2R", "NANOSLEEP"})
    VCHECK(turbo_class(m, {}) == C::Extra);
  VCHECK(turbo_class("LEPC", {}) == C::Call && !exact_opcode_allowed("LEPC", {}));   // (Round sp66: the return address of vprintf's call)
  VCHECK(turbo_class("B2R", {"RESULT"}) == C::Extra && !exact_opcode_allowed("B2R", {"RESULT"}));   // (Round sp66; exact's set is unchanged)
  // never: what needs a table only the simulator holds (textures, surfaces) or ends the card's context (BPT), in any spelling of its modifiers
  for (const char* m : {"TEX", "TLD", "TLD4", "TXD", "SULD", "SUST", "BPT"}) {
    VCHECK(turbo_class(m, {}) == C::Never);
    VCHECK(turbo_class(m, {"(9)"}) == C::Never);   // (an undecoded field does not make it merely unlisted)
  }
  // unlisted (turbo fit): what nobody named, what is not an instruction, the sm_90 and later forms; the card runs the cubin's own words
  for (const char* m : {"QMMA", "OMMA", "VPRINTF", "", "mufu", "Unknown"}) VCHECK(turbo_class(m, {}) == C::Unlisted);
  // never: the sm_90 and later forms that touch state the simulator keeps in its own format (the tensor-map descriptor, "GPUTMAP1") or that no card run has shown safe
  for (const char* m : {"UTMALDG", "UTMASTG", "UTMAREDG", "UTMAPF", "UTMACMDFLUSH", "UBLKCP", "UBLKRED", "UCGABAR", "HGMMA", "IGMMA", "BGMMA", "QGMMA", "WARPGROUP", "SYNCS", "ELECT",
                        "USETMAXREG", "LDGMC", "ENDCOLLECTIVE", "STSM", "STAS", "REDAS", "ARRIVES", "UGETNEXTWORKID", "RPCMOV", "LDTM", "STTM", "UTCSHIFT", "UTCMMA", "UTCBAR", "UTCCP",
                        "UTCATOMSWS", "JMP"}) {
    VCHECK(unmeasured_private_mnemonic(m));
    VCHECK(turbo_class(m, {}) == C::Never);
    VCHECK(turbo_class(m, {"(9)"}) == C::Never);
    VCHECK(!exact_opcode_allowed(m, {}));
  }
  VCHECK(!unmeasured_private_mnemonic("QMMA") && !unmeasured_private_mnemonic("IMAD") && !unmeasured_private_mnemonic("UTMAL"));
}

// An unlisted CALL (a modifier the decoder cannot name) must still go through the call-target scan; a call is a call whatever class its modifiers made of it.
VTEST(a_call_is_scanned_whatever_its_class) {
  using C = InsnClass;
  VCHECK(turbo_class("CALL", {"REL", "(7)"}) == C::Unlisted);
  VCHECK(call_needs_scan(true, turbo_class("CALL", {"REL", "(7)"})));
  VCHECK(call_needs_scan(true, turbo_class("CALL", {"ABS", "NOINC"})));
  VCHECK(!call_needs_scan(false, C::Unlisted) && !call_needs_scan(false, C::Call));
  VCHECK(!call_needs_scan(true, C::Never) && !call_needs_scan(true, C::Exact));
}

// Turbo mode takes an unlisted instruction and exact mode never did: the exact whitelist is not touched by the new class.
VTEST(unlisted_is_turbo_only) {
  using C = InsnClass;
  for (const char* m : {"QMMA", "OMMA", "Unknown", ""}) {
    VCHECK(turbo_class(m, {}) == C::Unlisted);
    VCHECK(!exact_opcode_allowed(m, {}));
  }
  for (const char* m : {"JMP", "STSM", "UTMALDG", "UBLKCP", "HGMMA", "SYNCS"}) VCHECK(turbo_class(m, {}) == C::Never);   // (TMA reads the simulator's own descriptor)
  VCHECK(!exact_opcode_allowed("ATOMS", {"ARRIVE", "64"}) && turbo_class("ATOMS", {"ARRIVE", "64"}) == C::Unlisted);
  const TurboSwitches def;
  VCHECK(turbo_refusal(class_bit(C::Unlisted), def) == TurboRefusal::None);
  VCHECK(turbo_refusal(class_bit(C::Exact) | class_bit(C::Unlisted) | class_bit(C::Mufu), def) == TurboRefusal::None);
  VCHECK(turbo_refusal(class_bit(C::Unlisted) | class_bit(C::Never), def) == TurboRefusal::Never);   // an unlisted instruction does not admit a texture
  TurboSwitches off;
  VCHECK(parse_turbo_off("unlisted", &off.off));
  VCHECK(turbo_refusal(class_bit(C::Unlisted), off) == TurboRefusal::Off);
  VCHECK(turbo_refusal(class_bit(C::Exact) | class_bit(C::Mufu), off) == TurboRefusal::None);
  VCHECK(std::string(insn_class_name(C::Unlisted)) == "unlisted");
  VCHECK(kInsnClasses == 12);
}

// Every instruction the decoder knows has a class, and the classes of the ones turbo mode adds are the ones the doc lists: a new instruction in ops.inc is
// Never until someone names it here and there.
VTEST(turbo_whitelist_covers_the_decoder_and_nothing_unnamed) {
  std::vector<std::string> names;
#define VGPU_SASS_OP(name) names.push_back(#name);
#include "vgpu/sass/ops.inc"
#undef VGPU_SASS_OP
  std::map<InsnClass, unsigned> n;
  for (const std::string& m : names) ++n[turbo_class(m, {})];
  VCHECK_EQ(n[InsnClass::Mufu], 1u);
  VCHECK_EQ(n[InsnClass::Half], 6u);
  VCHECK_EQ(n[InsnClass::Call], 2u);
  VCHECK_EQ(n[InsnClass::Tensor], 4u);   // HMMA, IMMA, DMMA, BMMA
  VCHECK_EQ(n[InsnClass::AtomInt], 3u);   // ATOM, ATOMG, ATOMS with no modifier
  VCHECK_EQ(n[InsnClass::Extra], 17u);   // (B2R is a mnemonic of the BAR op, not an op of its own; NANOSLEEP is one)
  VCHECK_EQ(n[InsnClass::FloatRed] + n[InsnClass::AtomFloat] + n[InsnClass::AtomSwap], 0u);
  VCHECK_EQ(n[InsnClass::Exact] + n[InsnClass::Never] + n[InsnClass::Unlisted] + 33u, static_cast<unsigned>(names.size()));
  // Never is only what the card cannot run for the simulator: the texture and surface instructions the decoder names, and BPT. Every other mnemonic the
  // decoder knows and no class names is Unlisted (turbo fit: admitted).
  unsigned never_names = 0;
  for (const std::string& m : names) never_names += texture_mnemonic(m) || m == "BPT" || unmeasured_private_mnemonic(m);
  VCHECK_EQ(n[InsnClass::Never], never_names);
  VCHECK(n[InsnClass::Unlisted] >= 1);   // (QMMA, OMMA ...: what is left once the sm_90+ private forms are Never)
  for (unsigned c = 0; c < kInsnClasses; ++c) VCHECK(std::string(insn_class_name(static_cast<InsnClass>(c))).size() > 3);
  VCHECK(std::string(insn_class_name(InsnClass::Never)) == "never" && std::string(insn_class_name(InsnClass::Exact)) == "exact");
}

VTEST(modifier_parts_are_read_inside_joined_modifiers) {
  VCHECK(mods_have({"E", "ADD", "F32.FTZ.RN"}, "FTZ"));
  VCHECK(mods_have({"F32.FTZ.RN"}, "RN") && mods_have({"F32.FTZ.RN"}, "F32"));
  VCHECK(!mods_have({"F32.FTZ.RN"}, "F3") && !mods_have({"F32.FTZ.RN"}, "FTZ.RN") && !mods_have({}, "E") && !mods_have({"", "."}, ""));
  VCHECK(!mods_have({"CAST"}, "CAS") && !mods_have({"EXCHANGE"}, "EXCH"));
  VCHECK(mods_float_typed({"F16x2.RN"}) && mods_float_typed({"E", "BF16x8.RN"}) && mods_float_typed({"F64"}) && mods_float_typed({"X", "F32x4.FTZ.RN"}));
  VCHECK(!mods_float_typed({"E", "ADD", "S32", "64", "FTZ", "RN"}) && !mods_float_typed({}));
  VCHECK(mods_undecoded({"E", "(9)"}) && mods_undecoded({"X.(3)"}) && !mods_undecoded({"E", "ADD"}) && !mods_undecoded({}));
}

VTEST(a_call_must_stay_inside_the_kernel) {
  VCHECK(call_stays_inside(true, true, 0x100, 0x0, 0x400));
  VCHECK(call_stays_inside(true, true, 0x0, 0x0, 0x400) && call_stays_inside(true, true, 0x400, 0x0, 0x400));   // the first and the last instruction
  VCHECK(!call_stays_inside(true, true, 0x410, 0x0, 0x400));    // past the end
  VCHECK(!call_stays_inside(true, true, 0x70, 0x80, 0x400));    // before the start
  VCHECK(!call_stays_inside(true, true, -16, 0x0, 0x400));
  VCHECK(!call_stays_inside(false, true, 0x100, 0x0, 0x400));   // CALL.ABS
  VCHECK(!call_stays_inside(true, false, 0x100, 0x0, 0x400));   // through a register
  // an absolute call to another section's first instruction is followed (Round sp66), and nothing else is
  VCHECK(call_to_entry(false, true, 0x6ffb00000100, 0x6ffb00000100, 12));
  VCHECK(!call_to_entry(false, true, 0x6ffb00000110, 0x6ffb00000100, 12));   // into the middle of a function
  VCHECK(!call_to_entry(true, true, 0x6ffb00000100, 0x6ffb00000100, 12));    // a relative call is the section's own
  VCHECK(!call_to_entry(false, false, 0x6ffb00000100, 0x6ffb00000100, 12));  // through a register
  VCHECK(!call_to_entry(false, true, 0x6ffb00000100, 0x6ffb00000100, 0));    // a section with no code
  VCHECK(!call_to_entry(false, true, -1, 0xffffffffffffffffull, 1));
}

VTEST(turbo_admits_by_class_and_switch) {
  using C = InsnClass;
  const TurboSwitches def;
  const auto mask = [](std::initializer_list<C> cs) {
    ClassMask m = 0;
    for (C c : cs) m |= class_bit(c);
    return m;
  };
  // the defaults (Round sp66): everything but what has no class; float atomics with a value and swaps included
  VCHECK(def.atomics && def.off == 0 && !def.printf_off);
  VCHECK(turbo_refusal(mask({C::Exact}), def) == TurboRefusal::None);
  VCHECK(turbo_refusal(0, def) == TurboRefusal::None);
  for (C c : {C::Mufu, C::Half, C::FloatRed, C::Call, C::Extra, C::Tensor, C::AtomInt, C::AtomFloat, C::AtomSwap})
    VCHECK(turbo_refusal(mask({C::Exact, c}), def) == TurboRefusal::None);
  VCHECK(turbo_refusal(mask({C::Exact, C::Mufu, C::Half, C::FloatRed, C::Call, C::Extra, C::Tensor, C::AtomInt, C::AtomFloat, C::AtomSwap}), def) == TurboRefusal::None);
  VCHECK(turbo_refusal(mask({C::Never}), def) == TurboRefusal::Never);
  VCHECK(turbo_refusal(mask({C::AtomSwap, C::Never}), def) == TurboRefusal::Never);   // nothing admits Never
  VCHECK(turbo_refusal(mask({C::Exact, C::Unlisted}), def) == TurboRefusal::None);     // (turbo fit: an instruction no class names is admitted)
  // VGPU_HOST_GPU_ATOMICS=0: float atomics with a value and swaps stay; integer atomics are the off list's
  TurboSwitches no;
  no.atomics = false;
  VCHECK(turbo_refusal(mask({C::Exact, C::AtomFloat}), no) == TurboRefusal::Atomics);
  VCHECK(turbo_refusal(mask({C::Exact, C::AtomSwap}), no) == TurboRefusal::Atomics);
  VCHECK(turbo_refusal(mask({C::Exact, C::AtomInt, C::Mufu}), no) == TurboRefusal::None);
  // the off list: each name switches its class off, and only its class
  const struct { const char* name; std::vector<C> off; } rows[] = {
      {"mufu", {C::Mufu}}, {"tensor", {C::Tensor}}, {"half", {C::Half}}, {"call", {C::Call}}, {"red", {C::FloatRed}}, {"extra", {C::Extra}},
      {"atomics", {C::AtomInt, C::AtomFloat, C::AtomSwap}}, {"unlisted", {C::Unlisted}}};
  for (const auto& row : rows) {
    TurboSwitches sw;
    sw.atomics = true;
    VCHECK(parse_turbo_off(row.name, &sw.off));
    for (unsigned c = 0; c < kInsnClasses; ++c) {
      const C cls = static_cast<C>(c);
      const bool is_off = std::find(row.off.begin(), row.off.end(), cls) != row.off.end();
      const TurboRefusal want = cls == C::Never ? TurboRefusal::Never : is_off ? TurboRefusal::Off : TurboRefusal::None;
      VCHECK(turbo_refusal(class_bit(cls), sw) == want);
    }
  }
  // the order: no class beats a switched-off class beats VGPU_HOST_GPU_ATOMICS=0
  TurboSwitches sw;
  sw.atomics = false;
  VCHECK(parse_turbo_off("mufu", &sw.off));
  VCHECK(turbo_refusal(mask({C::Mufu, C::AtomSwap}), sw) == TurboRefusal::Off);
  VCHECK(turbo_refusal(mask({C::Mufu, C::AtomSwap, C::Never}), sw) == TurboRefusal::Never);
  // the exact class cannot be switched off
  VCHECK(parse_turbo_off("mufu,tensor,half,call,atomics,red,extra,unlisted", &sw.off));
  VCHECK(turbo_refusal(mask({C::Exact}), sw) == TurboRefusal::None);
  VCHECK((sw.off & (class_bit(C::Exact) | class_bit(C::Never))) == 0);
}

VTEST(turbo_off_list_parses_names_and_refuses_the_rest) {
  ClassMask off = 99;
  std::string bad;
  VCHECK(parse_turbo_off(nullptr, &off) && off == 0);
  VCHECK(parse_turbo_off("", &off) && off == 0);
  VCHECK(parse_turbo_off("mufu,call", &off) && off == (class_bit(InsnClass::Mufu) | class_bit(InsnClass::Call)));
  VCHECK(parse_turbo_off(",mufu,,", &off) && off == class_bit(InsnClass::Mufu));
  VCHECK(!parse_turbo_off("mufu,sfu", &off, &bad) && bad == "sfu");
  VCHECK(!parse_turbo_off("MUFU", &off, &bad) && bad == "MUFU");
  VCHECK(!parse_turbo_off("mufu call", &off, &bad) && bad == "mufu call");
  VCHECK(!parse_turbo_off("exact", &off, &bad) && !parse_turbo_off("never", &off) && !parse_turbo_off("all", &off));
  // printf (Round sp66) is not a class: it sets its own switch, and is a bad name where nobody asks for it
  bool pf = true;
  VCHECK(parse_turbo_off("mufu", &off, nullptr, &pf) && !pf && off == class_bit(InsnClass::Mufu));
  VCHECK(parse_turbo_off("printf", &off, nullptr, &pf) && pf && off == 0);
  pf = false;
  VCHECK(parse_turbo_off("call,printf,red", &off, nullptr, &pf) && pf && off == (class_bit(InsnClass::Call) | class_bit(InsnClass::FloatRed)));
  VCHECK(parse_turbo_off(nullptr, &off, nullptr, &pf) && !pf);
  VCHECK(!parse_turbo_off("printf", &off, &bad) && bad == "printf");
  VCHECK(!parse_turbo_off("print", &off, &bad, &pf) && bad == "print");
}

// The kernels of the spike again, as turbo mode sees them (the classes of exact_set_matches_the_measured_kernels above).
VTEST(turbo_takes_the_kernels_exact_refused_for_numerics) {
  const auto classes = [](const std::vector<std::pair<std::string, std::vector<std::string>>>& code) {
    ClassMask m = 0;
    for (const auto& [mn, mods] : code) m |= class_bit(turbo_class(mn, mods));
    return m;
  };
  const TurboSwitches def;   // (atomics admitted, Round sp66)
  TurboSwitches at;
  at.atomics = false;           // VGPU_HOST_GPU_ATOMICS=0
  // HeCBench silu forward (expf): refused in exact, admitted in turbo
  const ClassMask silu = classes({{"FFMA", {}}, {"MUFU", {"EX2"}}, {"MUFU", {"RCP"}}, {"FCHK", {}}, {"FADD", {"FTZ"}}});
  VCHECK(turbo_refusal(silu, def) == TurboRefusal::None);
  // HeCBench aop compute_partial_beta (double atomicAdd as a compare-and-swap loop): only with the opt-in
  const ClassMask aop = classes({{"DADD", {}}, {"ATOMS", {"CAST", "SPIN", "64"}}, {"DFMA", {}}});
  VCHECK(turbo_refusal(aop, def) == TurboRefusal::None && turbo_refusal(aop, at) == TurboRefusal::Atomics);
  // HeCBench bh TreeBuildingKernel (ATOMG.E.CAS)
  const ClassMask bh = classes({{"FSETP", {"GEU", "AND"}}, {"ATOMG", {"E", "CAS", "STRONG", "GPU"}}});
  VCHECK(turbo_refusal(bh, def) == TurboRefusal::None && turbo_refusal(bh, at) == TurboRefusal::Atomics);
  // a histogram (integer atomicAdd whose old value is used) and a float reduction: admitted
  VCHECK(turbo_refusal(classes({{"LDG", {"E"}}, {"ATOMG", {"E", "ADD", "STRONG", "GPU"}}, {"RED", {"E", "ADD", "F32.FTZ.RN", "STRONG", "GPU"}}}), def) == TurboRefusal::None);
  // a texture fetch: never, whatever is switched on
  VCHECK(turbo_refusal(classes({{"TEX", {"SCR", "LL"}}, {"FMUL", {}}}), def) == TurboRefusal::Never);
}

// ---- the first blocks of a launch, timed (Round sp60b) ----

VTEST(sample_constants_are_the_ones_the_round_measured) {
  // (the text of Round sp60b and docs/perf-design/12-host-gpu-turbo.md quote these: a change here is a change there)
  VCHECK(kSampleMinBlocks == 256 && kSampleMaxBlocks == 32);
  VCHECK(kSampleBudgetSeconds == 0.002 && kSampleTimeoutSeconds == 0.050 && kSampleShareOfAge == 0.02 && kSampleCostGuessSeconds == 0.003);
  VCHECK(kSampleStaticFloorSeconds == 0.030 && kSampleSecondsPerLiveByte == 6e-10 && kSampleCardSpeedup == 8.0);
  VCHECK(kSampleTimeoutSecondsPerLiveByte == 2e-10);
}

VTEST(sample_runs_a_few_blocks_and_none_of_a_small_grid) {
  VCHECK(sample_blocks(0) == 0 && sample_blocks(1) == 0 && sample_blocks(kSampleMinBlocks - 1) == 0);
  VCHECK(sample_blocks(kSampleMinBlocks) == kSampleMaxBlocks);
  VCHECK(sample_blocks(4096) == kSampleMaxBlocks);                    // PolyBench SYRK's grid
  VCHECK(sample_blocks(uint64_t{1} << 40) == kSampleMaxBlocks);       // never more
  for (uint64_t b : {kSampleMinBlocks, uint64_t{300}, uint64_t{4096}, uint64_t{1} << 30})
    VCHECK(sample_blocks(b) >= 2 && sample_blocks(b) * 2 <= b);       // half the grid at most
}

VTEST(a_copy_is_given_longer_to_answer_with_more_live_memory) {
  VCHECK(sample_timeout_seconds(0) == kSampleTimeoutSeconds);
  VCHECK(sample_timeout_seconds(8ull << 20) > 0.0516 && sample_timeout_seconds(8ull << 20) < 0.0517);   // SYRK: 8 MiB
  VCHECK(sample_timeout_seconds(2000ull << 20) > 0.469 && sample_timeout_seconds(2000ull << 20) < 0.470);   // black-scholes: 1.9 GiB
  // never more than a ninth of the static estimate the floor asks for (the wait is lost when the copy does not answer)
  for (uint64_t live : {uint64_t{0}, uint64_t{8} << 20, uint64_t{80} << 20, uint64_t{1} << 30, uint64_t{16} << 30})
    VCHECK(sample_timeout_seconds(live) - kSampleTimeoutSeconds <= sample_static_floor(live) / 9 + 1e-12);
}

VTEST(sampled_blocks_are_spread_over_the_grid) {
  VCHECK(sample_stride(0) == 1 && sample_stride(kSampleMinBlocks - 1) == 1);   // nothing sampled: any stride
  VCHECK(sample_stride(kSampleMinBlocks) == kSampleMinBlocks / kSampleMaxBlocks);
  VCHECK(sample_stride(4096) == 128 && sample_stride(16384) == 512);
  for (uint64_t b : {kSampleMinBlocks, uint64_t{300}, uint64_t{463}, uint64_t{4096}, uint64_t{16384}, uint64_t{65536}, (uint64_t{1} << 30) + 7}) {
    const uint64_t n = sample_blocks(b), st = sample_stride(b);
    VCHECK(st >= 2);                                   // never the first blocks alone
    VCHECK((n - 1) * st < b);                          // the last sampled block is a block of the grid
    VCHECK(n * st + st > b - b / n);                   // ... and in its last part: the samples reach across the grid
  }
}

VTEST(samples_may_cost_a_share_of_the_time_the_process_has_run) {
  // a young process is not delayed by a sample: 3 ms is 2% of 150 ms
  VCHECK(!sample_affordable(0.060, 0.0, kSampleCostGuessSeconds));    // Rodinia srad_v2 is over in 60 ms
  VCHECK(!sample_affordable(0.149, 0.0, 0.003));
  VCHECK(sample_affordable(0.151, 0.0, 0.003));
  // what was spent counts: 2 s old, 38 ms spent, 3 ms more would pass 40 ms
  VCHECK(sample_affordable(2.0, 0.036, 0.003));
  VCHECK(!sample_affordable(2.0, 0.038, 0.003));
  // a dearer expected sample needs an older process
  VCHECK(!sample_affordable(1.0, 0.0, 0.050));
  VCHECK(sample_affordable(2.6, 0.0, 0.050));
  // a negative expectation is none, not a credit
  VCHECK(!sample_affordable(1.0, 0.021, -1.0));
  VCHECK(sample_affordable(1.0, 0.019, -1.0));
}

// `timed` blocks, whose second runs took `seconds`.
// ---- code_has_loop: a cycle that the entry reaches ----

static FlowInsn flow(int64_t target, bool falls_through = true, bool indirect = false) {
  FlowInsn f;
  f.target = target;
  f.falls_through = falls_through;
  f.indirect = indirect;
  return f;
}

VTEST(a_loop_is_a_cycle_the_entry_reaches) {
  VCHECK(!code_has_loop({}));
  VCHECK(!code_has_loop({flow(-1), flow(-1), flow(-1, false)}));                       // straight code, EXIT
  // a bounds check that skips to the exit, then the compiler's branch to itself after the EXIT: a back edge, no loop (2DCONV, GEMVER)
  VCHECK(!code_has_loop({flow(3), flow(-1), flow(-1), flow(-1, false), flow(4, false)}));
  // a shared exit placed first, reached by a branch back: still no loop (the exit does not go on)
  VCHECK(!code_has_loop({flow(2, false), flow(-1, false), flow(-1), flow(1, false)}));
  // a counted loop: the conditional branch at 3 goes back to 1 (SYRK, SYR2K)
  VCHECK(code_has_loop({flow(-1), flow(-1), flow(-1), flow(1), flow(-1, false)}));
  // a loop closed by an unconditional branch, left by a conditional one
  VCHECK(code_has_loop({flow(-1), flow(4), flow(-1), flow(1, false), flow(-1, false)}));
  // a branch to itself that IS reached (a spin)
  VCHECK(code_has_loop({flow(-1), flow(1)}));
  // a cycle the entry cannot reach is not one
  VCHECK(!code_has_loop({flow(-1), flow(-1, false), flow(-1), flow(2, false)}));
  // a diamond (two paths that meet) is not a cycle
  VCHECK(!code_has_loop({flow(2), flow(3, false), flow(-1), flow(-1, false)}));
  // anything indirect (a jump through a register, a call) is taken to be one
  VCHECK(code_has_loop({flow(-1), flow(-1, true, true), flow(-1, false)}));
  // a target outside the code is no edge; the last instruction falling off the end is none either
  VCHECK(!code_has_loop({flow(99), flow(-1)}));
  // a long straight run does not overflow anything (the walk is not recursive)
  std::vector<FlowInsn> big(200000, flow(-1));
  VCHECK(!code_has_loop(big));
  big.back() = flow(0);
  VCHECK(code_has_loop(big));
}

static Sample sample_of(double seconds, uint64_t timed, uint64_t total, unsigned threads = 4) {
  Sample s;
  s.seconds = seconds;
  s.timed = timed;
  s.total = total;
  s.threads = threads;
  return s;
}

VTEST(sample_extrapolates_the_launch_from_the_timed_blocks) {
  // 15 blocks of 4096 in 15 ms, for what would be 4 threads: 1 ms a block, 4096 blocks, shared among the threads
  const auto whole = sampled_launch_seconds(sample_of(0.015, 15, 4096, 4));
  VCHECK(whole.has_value());
  const double want = 0.001 * 4096 / 4;
  VCHECK(*whole > want * 0.999 && *whole < want * 1.001);
  // linear in the time and the grid, inverse in the threads; a thread count of 0 is read as 1
  VCHECK(*sampled_launch_seconds(sample_of(0.030, 15, 4096, 4)) > 1.999 * *whole && *sampled_launch_seconds(sample_of(0.030, 15, 4096, 4)) < 2.001 * *whole);
  VCHECK(*sampled_launch_seconds(sample_of(0.015, 15, 8192, 4)) > 1.999 * *whole && *sampled_launch_seconds(sample_of(0.015, 15, 8192, 4)) < 2.001 * *whole);
  VCHECK(*sampled_launch_seconds(sample_of(0.015, 15, 4096, 8)) > 0.499 * *whole && *sampled_launch_seconds(sample_of(0.015, 15, 4096, 8)) < 0.501 * *whole);
  VCHECK(*sampled_launch_seconds(sample_of(0.015, 15, 4096, 0)) == *sampled_launch_seconds(sample_of(0.015, 15, 4096, 1)));
  // per timed block: the same time over fewer blocks is more per block
  VCHECK(*sampled_launch_seconds(sample_of(0.015, 5, 4096, 4)) > 2.99 * *whole && *sampled_launch_seconds(sample_of(0.015, 5, 4096, 4)) < 3.01 * *whole);
  // nothing to say when no block was timed (the copy did not answer, or the launch's blocks take turns), or there is no time or grid
  VCHECK(!sampled_launch_seconds(sample_of(0.015, 0, 4096)));
  VCHECK(!sampled_launch_seconds(sample_of(0.0, 15, 4096)));
  VCHECK(!sampled_launch_seconds(sample_of(-1.0, 15, 4096)));
  VCHECK(!sampled_launch_seconds(sample_of(0.015, 15, 0)));
}

// A sample whose estimate of the launch is `launch_s` (4 threads, 15 blocks of 4096).
static Sample sample_saying(double launch_s) { return sample_of(launch_s * 15 * 4 / 4096, 15, 4096, 4); }

VTEST(sample_sends_only_what_repays_the_driver_start_and_the_card_path) {
  const Costs c;
  const uint64_t live = 8ull << 20;   // SYRK's 8 MiB
  VCHECK(sample_offload(sample_saying(4.0), live, live, c, false));      // PolyBench SYRK on the A10G rig: 4.2 s in the simulator
  VCHECK(sample_offload(sample_saying(7.0), live, live, c, false));      // SYR2K
  VCHECK(!sample_offload(sample_saying(0.3), live, live, c, false));     // 0.3 s: the start alone (0.25 s, margin 2) is more
  VCHECK(sample_offload(sample_saying(0.3), live, live, c, true));       // ... worth it once the driver is up
  VCHECK(!sample_offload(sample_saying(0.0001), live, live, c, true));   // under the floor of any offload
  VCHECK(!sample_offload(sample_of(0.0, 0, 4096), 0, 0, c, true));       // a copy that did not answer: no estimate, nothing sent
  VCHECK(!sample_offload(sample_of(0.015, 0, 4096), 0, 0, c, true));
  // the bar, by bisection: the launch must take more than margin x (start + copies each way + launch + the card's assumed kernel time)
  double lo = 0.3, hi = 4.0;
  for (int i = 0; i < 60; ++i) {
    const double mid = (lo + hi) / 2;
    (sample_offload(sample_saying(mid), live, live, c, false) ? hi : lo) = mid;
  }
  const double path = c.start_cost_s + c.launch_s + hi / kSampleCardSpeedup + double(live) / c.push_bytes_per_s + double(live) / c.pull_bytes_per_s;
  VCHECK(hi > c.margin * path * 0.999 && hi < c.margin * path * 1.001);
  VCHECK(hi > 2 * c.start_cost_s);                                       // never under twice the start
}

VTEST(sample_does_not_use_the_account_of_time_already_spent) {
  // should_offload_with_start buys the start once the program has spent as much in the simulator: that is how a short program paid for a start it
  // could not use. The sampled decision has no such argument: the same sample gives the same answer whatever was spent before.
  const Costs c;
  const Estimate e = sample_estimate(sample_saying(0.3), 8ull << 20, 8ull << 20, c, true);
  VCHECK(should_offload_with_start(e, c, false, 10.0));                  // the account would send it ...
  VCHECK(!sample_offload(sample_saying(0.3), 8ull << 20, 8ull << 20, c, false));   // ... the sample does not
}

VTEST(sample_charges_the_memory_and_an_unknown_card_time) {
  const Costs c;
  // 1 s with the driver up: goes with 8 MiB live, not with 5 GiB to push (0.45 s) nor with 4 GiB charged for coming back (0.43 s) and 2 GiB to push
  VCHECK(sample_offload(sample_saying(1.0), 8ull << 20, 8ull << 20, c, true));
  VCHECK(!sample_offload(sample_saying(1.0), 5ull << 30, 8ull << 20, c, true));
  VCHECK(sample_offload(sample_saying(1.0), 8ull << 20, 2ull << 30, c, true));
  VCHECK(!sample_offload(sample_saying(1.0), 2ull << 30, 4ull << 30, c, true));
  const Estimate e = sample_estimate(sample_saying(1.0), 3ull << 20, 8ull << 20, c, true);
  VCHECK(e.push_bytes == (3ull << 20) && e.pull_bytes == (8ull << 20) && e.start_s == 0);
  VCHECK(e.sim_s > 0.999 && e.sim_s < 1.001);
  VCHECK(e.card_kernel_s.has_value() && *e.card_kernel_s > 1.0 / kSampleCardSpeedup * 0.999 && *e.card_kernel_s < 1.0 / kSampleCardSpeedup * 1.001);
  VCHECK(sample_estimate(sample_saying(1.0), 0, 0, c, false).start_s == c.start_cost_s);
  // never under the default kernel time for a short launch
  VCHECK(*sample_estimate(sample_saying(0.0001), 0, 0, c, true).card_kernel_s == c.kernel_s_default);
  // a kernel the card has run before: its measured time is used
  VCHECK(*sample_estimate(sample_saying(1.0), 0, 0, c, true, 0.4).card_kernel_s == 0.4);
  VCHECK(sample_offload(sample_saying(1.0), 0, 0, c, true, 0.4));        // 1 s against 2 x 0.4 s
  VCHECK(!sample_offload(sample_saying(1.0), 0, 0, c, true, 0.6));       // not against 2 x 0.6 s
  // a start cost of 0 (VGPU_HOST_GPU_START_COST=0) charges nothing; a negative one is no credit
  Costs free_start;
  free_start.start_cost_s = 0;
  VCHECK(sample_offload(sample_saying(0.3), 8ull << 20, 8ull << 20, free_start, false));
  free_start.start_cost_s = -5;
  VCHECK(sample_estimate(sample_saying(0.3), 0, 0, free_start, false).start_s == 0);
}

static SampleCandidate old_enough(uint64_t blocks) {
  SampleCandidate w;
  w.blocks = blocks;
  w.loop = true;
  w.age_s = 10.0;
  return w;
}

VTEST(sampling_is_for_grids_with_blocks_to_sample) {
  const Costs c;
  VCHECK(worth_sampling(old_enough(4096), c, 0.26));
  VCHECK(!worth_sampling(old_enough(kSampleMinBlocks - 1), c, 0.26));
  VCHECK(worth_sampling(old_enough(kSampleMinBlocks), c, 0.26));
  VCHECK(!worth_sampling(old_enough(8), c, 0.26));   // PolyBench COVAR's long kernel runs on 8 blocks: nothing to sample
  SampleCandidate big = old_enough(8);
  big.static_s = 100.0;                              // ... whatever its static estimate
  VCHECK(!worth_sampling(big, c, 0.26));
}

VTEST(sampling_is_not_repeated_for_a_kernel_whose_whole_launch_was_short) {
  const Costs c;
  SampleCandidate w = old_enough(4096);
  w.whole_s = 0.003;                               // PolyBench 3DCONV: 254 launches of 3 ms
  VCHECK(!worth_sampling(w, c, 0.26));
  w.whole_s = c.margin * 0.26;                     // at the bar: not above it
  VCHECK(!worth_sampling(w, c, 0.26));
  w.whole_s = c.margin * 0.26 * 1.01;
  VCHECK(worth_sampling(w, c, 0.26));
}

VTEST(sampling_needs_a_long_static_estimate_or_a_process_that_can_afford_it) {
  const Costs c;
  SampleCandidate w = old_enough(4096);
  w.age_s = 0.015;                                 // PolyBench SYR2K launches 15 ms into the process ...
  w.static_s = 0.0446;                             // ... with a static estimate of 45 ms: sampled
  VCHECK(worth_sampling(w, c, 0.26));
  w.static_s = 0.0016;                             // HeCBench backprop, a program of 14 ms: 1.6 ms, not sampled
  VCHECK(!worth_sampling(w, c, 0.26));
  w.static_s = kSampleStaticFloorSeconds;          // the floor itself is enough
  VCHECK(worth_sampling(w, c, 0.26));
  w.static_s = kSampleStaticFloorSeconds * 0.999;
  VCHECK(!worth_sampling(w, c, 0.26));
  // the floor rises with the live memory (the faults after the fork): three times 0.6 ms a MiB, once that is over 30 ms
  VCHECK(sample_static_floor(0) == kSampleStaticFloorSeconds && sample_static_floor(12ull << 20) == kSampleStaticFloorSeconds);   // SYR2K: 12 MiB, 45 ms
  VCHECK(sample_static_floor(80ull << 20) > 0.150 && sample_static_floor(80ull << 20) < 0.152);    // 2MM's first kernel: 80 MiB, 335 ms
  VCHECK(sample_static_floor(1ull << 30) > 1.93 && sample_static_floor(1ull << 30) < 1.94);
  w.live_bytes = 1ull << 30;
  w.static_s = 1.9;
  VCHECK(!worth_sampling(w, c, 0.26));
  w.static_s = 2.0;
  VCHECK(worth_sampling(w, c, 0.26));
  w.live_bytes = 0;
  // a kernel with no loop is not sampled, whatever its estimate and however old the process (Rodinia backprop: 105 ms, 36 MiB, no loop)
  w.static_s = 0.105;
  w.age_s = 100.0;
  w.loop = false;
  VCHECK(!worth_sampling(w, c, 0.26));
  w.loop = true;
  VCHECK(worth_sampling(w, c, 0.26));
  w.age_s = 0.015;
  w.static_s = 1.0;                                // the floor does not lift the other conditions
  w.whole_s = 0.003;
  VCHECK(!worth_sampling(w, c, 0.26));
  w.whole_s.reset();
  w.static_s = 0;
  w.age_s = 0.060;                                 // under the floor: by the process's age. Too young for the first sample's guess
  VCHECK(!worth_sampling(w, c, 0.26));
  w.age_s = 0.2;
  VCHECK(worth_sampling(w, c, 0.26));
  w.spent_s = 0.002;                               // 2 ms spent and 3 ms expected are more than 2% of 0.2 s
  VCHECK(!worth_sampling(w, c, 0.26));
  w.age_s = 1.0;
  VCHECK(worth_sampling(w, c, 0.26));
  w.expected_s = 0.050;                            // the process's dearest sample so far was a copy that never answered
  VCHECK(!worth_sampling(w, c, 0.26));
}

VTEST(sim_time_table_peeks_at_one_sample_and_takes_an_extrapolation_as_history) {
  SimTimeTable t;
  VCHECK(!t.peek("k", 100).has_value());
  t.record("k", 100, 0.01);                        // one short sample: no prediction, but a peek
  VCHECK(!t.predict("k", 100).has_value());
  VCHECK(t.peek("k", 100).has_value() && *t.peek("k", 100) > 0.0099 && *t.peek("k", 100) < 0.0101);
  VCHECK(*t.peek("k", 1000) > 0.099 && *t.peek("k", 1000) < 0.101);
  VCHECK(!t.peek("other", 100).has_value());
  // a launch sent on its sample: what the simulator was on course for is a full history at once
  SimTimeTable m;
  m.record_extrapolated("long", 65536, 7.7);
  VCHECK(m.samples("long") >= SimTimeTable::kSamplesNeeded);
  VCHECK(m.predict("long", 65536).has_value() && *m.predict("long", 65536) > 7.69 && *m.predict("long", 65536) < 7.71);
  // ... and not limited to the two offloaded launches of the one-sample allowance
  for (int i = 0; i < 5; ++i) m.note_offload("long");
  VCHECK(m.predict("long", 65536).has_value());
  // a later whole launch in the simulator still lowers it (the estimate is the smallest per-block time seen)
  m.record("long", 65536, 3.0);
  VCHECK(*m.predict("long", 65536) < 3.01);
  // an extrapolation under kOneSampleMinSeconds is a history too (record() alone would not be)
  SimTimeTable q;
  q.record_extrapolated("quick", 100, 0.1);
  VCHECK(q.predict("quick", 100).has_value());
}

// ---- progress inside a block (vgpu/exec/block_progress.hpp, find_probe_loops; Round sp61) ----

static vgpu::exec::ProbeLoop counted(int64_t step, int64_t offset = 0, bool is_unsigned = false) {
  vgpu::exec::ProbeLoop l;
  l.step = step;
  l.iv_offset = offset;
  l.is_unsigned = is_unsigned;
  return l;
}

VTEST(turns_left_are_the_distance_to_the_bound_in_steps) {
  using vgpu::exec::lane_turns_left;
  VCHECK(lane_turns_left(1, 2049, counted(1)) == 2048);          // COVAR: j2 from 1 to M + 1 = 2049
  VCHECK(lane_turns_left(2048, 2049, counted(1)) == 1);
  VCHECK(lane_turns_left(2049, 2049, counted(1)) == 0);          // at the bound: none
  VCHECK(lane_turns_left(3000, 2049, counted(1)) == 0);          // past it: none
  VCHECK(lane_turns_left(0, 10, counted(4)) == 3);               // unrolled by 4: a part turn is a turn
  VCHECK(lane_turns_left(0, 12, counted(4)) == 3);
  VCHECK(lane_turns_left(0, 13, counted(4)) == 4);
  VCHECK(lane_turns_left(100, 0, counted(-1)) == 100);           // counting down to 0
  VCHECK(lane_turns_left(100, 0, counted(-3)) == 34);
  VCHECK(lane_turns_left(0, 100, counted(-1)) == 0);             // moving away from the bound: none
  VCHECK(lane_turns_left(5, 10, counted(0)) == 0);               // no step: none
  VCHECK(lane_turns_left(10, 5, counted(0)) == 0);               // ... on either side of the bound
  VCHECK(lane_turns_left(4, 10, counted(1, 1)) == 5);            // the compare reads iv + 1
  VCHECK(lane_turns_left(4, 10, counted(1, -2)) == 8);
  // signed by default: -1 is below any positive bound; unsigned, 0xffffffff is above it
  VCHECK(lane_turns_left(0xffffffffu, 3, counted(1)) == 4);
  VCHECK(lane_turns_left(0xffffffffu, 3, counted(1, 0, true)) == 0);
  VCHECK(lane_turns_left(0x80000000u, 0x80000010u, counted(1, 0, true)) == 16);
  VCHECK(lane_turns_left(0x7ffffff0u, 0x7fffffffu, counted(1)) == 15);
  VCHECK(lane_turns_left(0xfffffff0u, 0xfffffff8u, counted(1)) == 8);     // -16 to -8, signed
  VCHECK(lane_turns_left(0xfffffff8u, 0xfffffff0u, counted(-2)) == 4);
}

VTEST(turns_between_two_pauses_are_what_the_register_moved) {
  using vgpu::exec::turns_between;
  VCHECK(turns_between(10, 13, counted(1)) == 3);
  VCHECK(turns_between(10, 10, counted(1)) == 0);
  VCHECK(turns_between(13, 10, counted(1)) == 0);                // backwards against the step: none
  VCHECK(turns_between(0, 9, counted(4)) == 2);                  // whole turns only
  VCHECK(turns_between(20, 8, counted(-4)) == 3);
  VCHECK(turns_between(8, 20, counted(-4)) == 0);
  VCHECK(turns_between(1, 5, counted(0)) == 0);
  VCHECK(turns_between(5, 1, counted(0)) == 0);
  VCHECK(turns_between(0xfffffffeu, 1, counted(1)) == 3);        // signed: -2 to 1
  VCHECK(turns_between(0xfffffffeu, 1, counted(1, 0, true)) == 0);
}

VTEST(a_measured_turn_gives_steps_and_seconds) {
  using vgpu::exec::TurnMeasure;
  TurnMeasure m;
  VCHECK(vgpu::exec::steps_per_turn(m) == 0 && vgpu::exec::seconds_per_step(m) == 0);   // nothing measured
  m.steps = 40000;
  m.turns = 4;
  m.seconds = 0.004;
  VCHECK(vgpu::exec::steps_per_turn(m) == 10000);
  VCHECK(vgpu::exec::seconds_per_step(m) > 0.99e-7 && vgpu::exec::seconds_per_step(m) < 1.01e-7);
  m.turns = 0;                                       // no turn completed: the run is the least a turn takes
  VCHECK(vgpu::exec::steps_per_turn(m) == 40000);
  m.turns = 0.5;
  VCHECK(vgpu::exec::steps_per_turn(m) == 40000);
  m.turns = 1;
  VCHECK(vgpu::exec::steps_per_turn(m) == 40000);
  m.seconds = 0;
  VCHECK(vgpu::exec::seconds_per_step(m) == 0);
  m.seconds = -0.001;                                // (a clock that went back: no measure)
  VCHECK(vgpu::exec::seconds_per_step(m) == 0);
  TurnMeasure none;                                  // a time and no step: no measure, and no division by zero
  none.seconds = 0.004;
  VCHECK(vgpu::exec::seconds_per_step(none) == 0 && vgpu::exec::steps_per_turn(none) == 0);
  vgpu::exec::BlockReading b;
  b.steps = 1000;
  b.turns = 50;
  VCHECK(vgpu::exec::block_steps(b, 200) == 11000);
  VCHECK(vgpu::exec::block_steps(b, -5) == 1000);  // (no negative cost)
}

VTEST(blocks_read_inside_a_small_grid) {
  VCHECK(inblock_blocks(8) == 8 && inblock_stride(8) == 1);       // COVAR: every block
  VCHECK(inblock_blocks(1) == 1 && inblock_stride(1) == 1);
  VCHECK(inblock_blocks(0) == 0);
  VCHECK(inblock_blocks(kSampleMaxBlocks) == kSampleMaxBlocks);
  VCHECK(inblock_blocks(kSampleMaxBlocks + 1) == kSampleMaxBlocks);
  VCHECK(inblock_blocks(255) == kSampleMaxBlocks && inblock_stride(255) == 255 / kSampleMaxBlocks);
  VCHECK(inblock_blocks(kSampleMinBlocks) == 0);                 // the whole-block sample's grids are not read inside
  VCHECK(inblock_blocks(4096) == 0 && inblock_stride(4096) == 1);
  VCHECK(kInBlockEnterSeconds == 0.001 && kInBlockMeasureSeconds == 0.004 && kInBlockBudgetSeconds == 0.012);   // what section 8 quotes
}

static vgpu::exec::BlockReading reading(double steps, double turns) {
  vgpu::exec::BlockReading b;
  b.steps = steps;
  b.turns = turns;
  return b;
}

VTEST(a_launch_from_the_blocks_read) {
  vgpu::exec::TurnMeasure m;
  m.steps = 30000;
  m.turns = 3;
  m.seconds = 0.003;                                 // 10,000 steps a turn, 0.1 us a step
  // no block, no measure, no grid: no estimate
  VCHECK(!inblock_launch_seconds({}, m, 8, 4).has_value());
  VCHECK(!inblock_launch_seconds({reading(0, 10)}, vgpu::exec::TurnMeasure{}, 8, 4).has_value());
  VCHECK(!inblock_launch_seconds({reading(0, 10)}, m, 0, 4).has_value());
  // 8 blocks read on 4 threads, 2048 turns of 10,000 steps left in each of 8 warps: 8 x 8 x 2048 x 1e4 steps / 4 x 1e-7 s
  std::vector<vgpu::exec::BlockReading> eight(8, reading(0, 8 * 2048));
  const auto e = inblock_launch_seconds(eight, m, 8, 4);
  VCHECK(e.has_value() && *e > 32.7 && *e < 32.8);
  // the steps taken count
  const auto e2 = inblock_launch_seconds({reading(1e6, 0)}, m, 1, 1);
  VCHECK(e2.has_value() && *e2 > 0.0999 && *e2 < 0.1001);
  // a grid of fewer blocks than threads: the longest block, not a share of it
  const auto one = inblock_launch_seconds({reading(0, 100), reading(0, 300)}, m, 2, 4);
  VCHECK(one.has_value() && *one > 0.2999 && *one < 0.3001);
  // the blocks read stand for the grid: two read of 16, on 4 threads
  const auto many = inblock_launch_seconds({reading(0, 100), reading(0, 300)}, m, 16, 4);
  VCHECK(many.has_value() && *many > 0.7999 && *many < 0.8001);
  // threads 0 is taken as 1
  const auto t0 = inblock_launch_seconds({reading(0, 100)}, m, 2, 0);
  VCHECK(t0.has_value() && *t0 > 0.1999 && *t0 < 0.2001);
  // no turn completed in the measure: a turn takes at least the run's 30,000 steps
  vgpu::exec::TurnMeasure slow = m;
  slow.turns = 0;
  const auto low = inblock_launch_seconds(eight, slow, 8, 4);
  VCHECK(low.has_value() && *low > 98.2 && *low < 98.4);
}

VTEST(a_small_grid_is_sampled_inside_only_with_a_followed_loop) {
  const Costs c;
  SampleCandidate w = old_enough(8);
  VCHECK(!sample_inside(w) && !worth_sampling(w, c, 0.26));
  w.followed = true;
  VCHECK(sample_inside(w) && worth_sampling(w, c, 0.26));
  w.loop = false;                                    // the loop test still holds
  VCHECK(!worth_sampling(w, c, 0.26));
  w.loop = true;
  w.whole_s = 0.003;                                 // so does a short whole launch
  VCHECK(!worth_sampling(w, c, 0.26));
  w.whole_s.reset();
  w.age_s = 0.015;                                   // and the floor or the process's age
  w.static_s = 0.001;
  VCHECK(!worth_sampling(w, c, 0.26));
  w.static_s = kSampleStaticFloorSeconds;
  VCHECK(worth_sampling(w, c, 0.26));
  SampleCandidate g = old_enough(4096);              // a grid the whole-block sample takes is not read inside
  g.followed = true;
  VCHECK(!sample_inside(g) && worth_sampling(g, c, 0.26));
  g.blocks = 0;
  VCHECK(!sample_inside(g) && !worth_sampling(g, c, 0.26));
}

VTEST(a_launch_sampled_inside_goes_by_the_rule_of_sampled_launches) {
  const Costs c;
  VCHECK(!sampled_offload(0, 0, 0, c, true));
  VCHECK(sampled_offload(1.0, 0, 0, c, true, 0.4));
  VCHECK(!sampled_offload(1.0, 0, 0, c, true, 0.6));
  // COVAR: 23 s in the simulator, 64 MiB live, driver not started: the card kernel is guessed at 1/8 of it
  VCHECK(sampled_offload(23.0, 64ull << 20, 64ull << 20, c, false));
  const Estimate e = sampled_estimate(23.0, 64ull << 20, 64ull << 20, c, false);
  VCHECK(e.card_kernel_s && *e.card_kernel_s > 2.87 && *e.card_kernel_s < 2.88 && e.start_s == c.start_cost_s);
  // a launch of 50 ms does not repay the start
  VCHECK(!sampled_offload(0.05, 64ull << 20, 64ull << 20, c, false));
  // the old entry point is the same rule
  Sample s;
  s.seconds = 1.0;
  s.timed = 1;
  s.total = 1;
  VCHECK(sample_offload(s, 0, 0, c, true, 0.4) == sampled_offload(1.0, 0, 0, c, true, 0.4));
}

// The shapes of nvcc's counted loops, as find_probe_loops is given them: registers 0..254, uniform ones 256 + n; predicates 0..6, uniform 8 + n.
static LoopCodeInsn op_other(std::vector<int> writes = {}) {
  LoopCodeInsn c;
  c.flow = flow(-1);
  c.writes = std::move(writes);
  return c;
}
static LoopCodeInsn op_add(int dst, int src, int64_t imm) {
  LoopCodeInsn c = op_other({dst});
  c.kind = LoopCodeInsn::Kind::AddImm;
  c.add_dst = dst;
  c.add_src = src;
  c.add_imm = imm;
  return c;
}
static vgpu::exec::ProbeOperand opnd_reg(int n) { return probe_reg_operand(n); }
static vgpu::exec::ProbeOperand opnd_imm(int64_t v) {
  vgpu::exec::ProbeOperand o;
  o.kind = vgpu::exec::ProbeOperand::Imm;
  o.imm = v;
  return o;
}
static vgpu::exec::ProbeOperand opnd_param(int64_t off) {
  vgpu::exec::ProbeOperand o;
  o.kind = vgpu::exec::ProbeOperand::CBank;
  o.reg = 0;
  o.imm = off;
  return o;
}
static LoopCodeInsn op_cmp(int pred, vgpu::exec::ProbeOperand a, vgpu::exec::ProbeOperand b, bool is_unsigned = false) {
  LoopCodeInsn c = op_other();
  c.kind = LoopCodeInsn::Kind::Compare;
  c.cmp_pred = pred;
  c.cmp_a = a;
  c.cmp_b = b;
  c.cmp_unsigned = is_unsigned;
  c.pred_writes = {pred};
  return c;
}
static LoopCodeInsn op_bra(int64_t target, int pred = -1) {
  LoopCodeInsn c = op_other();
  c.flow = flow(target, pred >= 0);
  c.branch_pred = pred;
  return c;
}
static LoopCodeInsn op_exit(bool conditional = false) {
  LoopCodeInsn c = op_other();
  c.flow = flow(-1, conditional);
  return c;
}

// COVAR's covar_kernel, in the shape nvcc gives it: a guard, the outer loop over j2 (R5, bound the parameter M + 1 at c[0x0][0x160]) and inside it
// the loop over i (R7).
static std::vector<LoopCodeInsn> covar_shape() {
  return {
      op_other({0}),                                   // 0: S2R / IMAD: j1
      op_cmp(0, opnd_reg(0), opnd_param(0x160)),       // 1
      op_exit(true),                                   // 2: @P0 EXIT
      op_other({5}),                                   // 3: MOV R5, R0
      op_other({7}),                                   // 4: the outer head: MOV R7, 0x1
      op_other({8}),                                   // 5: the inner head: DFMA
      op_add(7, 7, 1),                                 // 6
      op_cmp(1, opnd_reg(7), opnd_param(0x164)),       // 7
      op_bra(5, 1),                                    // 8: @P1 BRA 5
      op_add(5, 5, 1),                                 // 9
      op_cmp(0, opnd_reg(5), opnd_param(0x160)),       // 10
      op_bra(4, 0),                                    // 11: @!P0 BRA 4
      op_exit(),                                       // 12
      op_bra(13),                                      // 13: BRA 13, never reached
  };
}

VTEST(the_top_level_counted_loop_is_found) {
  const auto f = find_probe_loops(covar_shape());
  VCHECK(f.size() == 1);
  VCHECK(f[0].lo == 4 && f[0].hi == 11);
  VCHECK(f[0].loop.iv.kind == vgpu::exec::ProbeOperand::Reg && f[0].loop.iv.reg == 5 && f[0].loop.step == 1 && f[0].loop.iv_offset == 0);
  VCHECK(f[0].loop.bound.kind == vgpu::exec::ProbeOperand::CBank && f[0].loop.bound.reg == 0 && f[0].loop.bound.imm == 0x160);
  VCHECK(!f[0].loop.is_unsigned);
  // an unsigned compare says so
  auto u = covar_shape();
  u[10] = op_cmp(0, opnd_reg(5), opnd_param(0x160), true);
  VCHECK(find_probe_loops(u).size() == 1 && find_probe_loops(u)[0].loop.is_unsigned);
  // the bound on the left
  auto sw = covar_shape();
  sw[10] = op_cmp(0, opnd_param(0x160), opnd_reg(5));
  VCHECK(find_probe_loops(sw).size() == 1 && find_probe_loops(sw)[0].loop.iv.reg == 5);
  // an immediate bound, or a register the loop does not write
  auto im = covar_shape();
  im[10] = op_cmp(0, opnd_reg(5), opnd_imm(2049));
  VCHECK(find_probe_loops(im).size() == 1 && find_probe_loops(im)[0].loop.bound.kind == vgpu::exec::ProbeOperand::Imm);
  auto rb = covar_shape();
  rb[10] = op_cmp(0, opnd_reg(5), opnd_reg(9));
  VCHECK(find_probe_loops(rb).size() == 1 && find_probe_loops(rb)[0].loop.bound.reg == 9);
  // a branch on the uniform predicate of a uniform counter
  auto ur = covar_shape();
  ur[9] = op_add(256 + 4, 256 + 4, 2);
  ur[10] = op_cmp(8, opnd_reg(256 + 4), opnd_param(0x160));
  ur[11] = op_bra(4, 8);
  const auto uf = find_probe_loops(ur);
  VCHECK(uf.size() == 1 && uf[0].loop.iv.kind == vgpu::exec::ProbeOperand::UReg && uf[0].loop.iv.reg == 4 && uf[0].loop.step == 2);
}

VTEST(a_loop_whose_counter_or_bound_is_not_plain_is_not_followed) {
  auto bw = covar_shape();   // the bound register is written in the loop
  bw[10] = op_cmp(0, opnd_reg(5), opnd_reg(8));
  VCHECK(find_probe_loops(bw).empty());
  auto twice = covar_shape();   // the counter is written twice
  twice[5] = op_other({5});
  VCHECK(find_probe_loops(twice).empty());
  auto other = covar_shape();   // the counter's one write is not an add of a constant to itself
  other[9] = op_other({5});
  VCHECK(find_probe_loops(other).empty());
  auto fromb = covar_shape();   // an add from another register that is not itself a counter
  fromb[9] = op_add(5, 8, 1);
  VCHECK(find_probe_loops(fromb).empty());
  auto zero = covar_shape();    // no step
  zero[9] = op_add(5, 5, 0);
  VCHECK(find_probe_loops(zero).empty());
  auto between = covar_shape();   // another instruction writes the predicate after the compare
  between[11].flow = flow(4, true);
  between.insert(between.begin() + 11, op_other());
  between[11].pred_writes = {0};
  for (auto& c : between)
    if (c.flow.target > 11) ++c.flow.target;
  VCHECK(find_probe_loops(between).empty());
  auto before = covar_shape();   // the predicate is set before the loop only
  before[10] = op_other();
  VCHECK(find_probe_loops(before).empty());
  auto pre = covar_shape();      // ... even when that compare reads the counter
  pre[1] = op_cmp(0, opnd_reg(5), opnd_param(0x160));
  pre[10] = op_other();
  VCHECK(find_probe_loops(pre).empty());
  auto notcmp = covar_shape();   // the last write of the predicate is not a compare
  notcmp[10] = op_other();
  notcmp[10].pred_writes = {0};
  VCHECK(find_probe_loops(notcmp).empty());
  auto wrongp = covar_shape();   // the compare writes another predicate than the branch reads
  wrongp[10].cmp_pred = 3;
  VCHECK(find_probe_loops(wrongp).empty());
  auto none = covar_shape();   // no register side at all
  none[10] = op_cmp(0, opnd_imm(1), opnd_param(0x160));
  VCHECK(find_probe_loops(none).empty());
  auto nob = covar_shape();   // no bound operand
  nob[10] = op_cmp(0, opnd_reg(5), vgpu::exec::ProbeOperand{});
  VCHECK(find_probe_loops(nob).empty());
  auto ind = covar_shape();   // a jump through a register: nothing is followed
  ind[3].flow.indirect = true;
  VCHECK(find_probe_loops(ind).empty());
  auto uncond = covar_shape();   // an unconditional back edge and no exit branch: nothing decides
  uncond[11] = op_bra(4);
  uncond[10] = op_other();
  VCHECK(find_probe_loops(uncond).empty());
}

VTEST(the_counter_may_be_read_through_a_register_the_loop_adds_to_it) {
  auto t = covar_shape();
  t[9] = op_add(5, 5, 2);       // R5 += 2
  t[10] = op_cmp(0, opnd_reg(6), opnd_param(0x160));
  t.insert(t.begin() + 10, op_add(6, 5, 1));   // R6 = R5 + 1, compared
  for (auto& c : t)
    if (c.flow.target > 10) ++c.flow.target;
  const auto f = find_probe_loops(t);
  VCHECK(f.size() == 1 && f[0].loop.iv.reg == 5 && f[0].loop.iv_offset == 1 && f[0].loop.step == 2 && f[0].hi == 12);
  auto t2 = t;   // ... but R5 must be a counter
  t2[9] = op_other({5});
  VCHECK(find_probe_loops(t2).empty());
}

VTEST(a_loop_left_by_a_branch_out_of_it_is_followed) {
  // while (true) { if (i >= n) break; ...; i += 1 }
  std::vector<LoopCodeInsn> c = {
      op_other({5}),                                   // 0: MOV R5, RZ
      op_cmp(2, opnd_reg(5), opnd_param(0x168)),       // 1: the head
      op_bra(6, 2),                                    // 2: @P2 BRA 6 (out)
      op_other({8}),                                   // 3
      op_add(5, 5, 1),                                 // 4
      op_bra(1),                                       // 5: BRA 1
      op_exit(),                                       // 6
  };
  const auto f = find_probe_loops(c);
  VCHECK(f.size() == 1 && f[0].lo == 1 && f[0].hi == 5 && f[0].loop.bound.imm == 0x168);
}

VTEST(loops_one_after_another_and_inside_one_another) {
  // two top-level loops in a row: both; the inner loop of the first is not top-level
  std::vector<LoopCodeInsn> c = {
      op_other({5, 6}),                                // 0
      op_add(5, 5, 1),                                 // 1: loop A head
      op_cmp(0, opnd_reg(5), opnd_imm(100)),           // 2
      op_bra(1, 0),                                    // 3
      op_add(6, 6, -1),                                // 4: loop B head
      op_cmp(1, opnd_reg(6), opnd_imm(0)),             // 5
      op_bra(4, 1),                                    // 6
      op_exit(),                                       // 7
  };
  const auto f = find_probe_loops(c);
  VCHECK(f.size() == 2 && f[0].lo == 1 && f[0].hi == 3 && f[1].lo == 4 && f[1].hi == 6 && f[1].loop.step == -1);
  // ranges that cross without one holding the other: neither
  std::vector<LoopCodeInsn> x = {
      op_other({5, 6}),                                // 0
      op_add(5, 5, 1),                                 // 1: head of A
      op_add(6, 6, 1),                                 // 2: head of B
      op_cmp(0, opnd_reg(5), opnd_imm(100)),           // 3
      op_bra(1, 0),                                    // 4: A = [1, 4]
      op_cmp(1, opnd_reg(6), opnd_imm(100)),           // 5
      op_bra(2, 1),                                    // 6: B = [2, 6]
      op_exit(),                                       // 7
  };
  VCHECK(find_probe_loops(x).empty());
  // two back edges to one head make one loop, to the last of them
  std::vector<LoopCodeInsn> two = {
      op_other({5}),                                   // 0
      op_add(5, 5, 1),                                 // 1: head
      op_cmp(0, opnd_reg(5), opnd_imm(100)),           // 2
      op_bra(1, 0),                                    // 3: continue
      op_other(),                                      // 4
      op_cmp(0, opnd_reg(5), opnd_imm(100)),           // 5
      op_bra(1, 0),                                    // 6: continue again
      op_exit(),                                       // 7
  };
  const auto t = find_probe_loops(two);
  VCHECK(t.size() == 1 && t[0].lo == 1 && t[0].hi == 6);
  // no more than kProbeLoops (each loop's counter is written once in its own range, which is all a loop is asked about)
  std::vector<LoopCodeInsn> many = {op_other({5})};
  for (int k = 0; k < 6; ++k) {
    const int64_t h = static_cast<int64_t>(many.size());
    many.push_back(op_add(5, 5, 1));
    many.push_back(op_cmp(0, opnd_reg(5), opnd_imm(100)));
    many.push_back(op_bra(h, 0));
  }
  many.push_back(op_exit());
  const auto m = find_probe_loops(many);
  VCHECK(m.size() == vgpu::exec::kProbeLoops && m[0].lo == 1 && m[3].lo == 10);
  // an unreachable back edge is no loop (nvcc's BRA to itself after the EXIT), nor is a branch to a later instruction
  VCHECK(find_probe_loops({op_exit(), op_bra(1)}).empty());
  // ... even a whole counted loop after the EXIT
  VCHECK(find_probe_loops({op_exit(), op_add(5, 5, 1), op_cmp(0, opnd_reg(5), opnd_imm(100)), op_bra(1, 0), op_exit()}).empty());
  VCHECK(find_probe_loops({op_other({5}), op_add(5, 5, 1), op_cmp(0, opnd_reg(5), opnd_imm(100)), op_bra(1, 0), op_exit()}).size() == 1);   // (reached: found)
  VCHECK(find_probe_loops({op_cmp(0, opnd_reg(5), opnd_imm(1)), op_bra(3, 0), op_other(), op_exit()}).empty());
  VCHECK(find_probe_loops({}).empty());
}

VTEST(a_small_grid_s_floor_counts_the_followed_loop_s_bound) {
  VCHECK(bound_turns(2048, 1) == 2048 && bound_turns(2048, 4) == 512 && bound_turns(2049, 4) == 512);
  VCHECK(bound_turns(-100, -1) == 100 && bound_turns(100, -3) == 33 && bound_turns(5, 0) == 0 && bound_turns(0, 1) == 0);
  const Costs c;
  // COVAR's covar_kernel, as server3's log has it: 8 blocks, 87 us counted, 32 MiB live, 39 ms into the process, its loop bounded by m = 2048
  SampleCandidate w;
  w.blocks = 8;
  w.loop = true;
  w.followed = true;
  w.static_s = 0.000087;
  w.live_bytes = 33562624;
  w.age_s = 0.039;
  VCHECK(!worth_sampling(w, c, 0.26));               // the count alone: under the floor, and too young
  w.bound_turns = 2048;
  VCHECK(sample_static_estimate(w) > 0.178 && sample_static_estimate(w) < 0.179);
  VCHECK(worth_sampling(w, c, 0.26));
  w.bound_turns = 100;                                 // a short bound: 9 ms, under the 60 ms that 32 MiB ask
  VCHECK(!worth_sampling(w, c, 0.26));
  w.bound_turns = 0;                                   // (no bound known: the count alone)
  VCHECK(sample_static_estimate(w) == w.static_s);
  // a grid the whole-block sample takes is not scaled by the bound
  SampleCandidate g = w;
  g.blocks = 4096;
  g.bound_turns = 2048;
  VCHECK(sample_static_estimate(g) == g.static_s && !worth_sampling(g, c, 0.26));
}

// ---- Round sp65: a launch while the card holds the live memory ----

VTEST(a_streak_launch_needs_the_driver_up_data_on_the_card_and_an_estimate) {
  const Costs c;
  StreakLaunch s;
  s.sim_s = 0.00015;
  s.card_only_bytes = 48ull << 20;   // GRAMSCHM: three 2048 x 2048 float matrices
  s.card_kernel_s = 20e-6;
  VCHECK(streak_offload(s, c, true));
  VCHECK(!streak_offload(s, c, false));   // never what starts the driver
  StreakLaunch none = s;
  none.card_only_bytes = 0;               // the simulator holds everything: nothing to pull, today's rule decides
  VCHECK(!streak_offload(none, c, true));
  none.card_only_bytes = 1;
  VCHECK(streak_offload(none, c, true) == (streak_keep_seconds(none, c) >= c.min_sim_s && streak_keep_seconds(none, c) > c.margin * streak_card_seconds(none, c)));
  // a long launch with nothing on the card is the size rule's to decide (should_offload says yes to it), not this rule's
  StreakLaunch long_none;
  long_none.sim_s = 0.001;
  long_none.card_kernel_s = 20e-6;
  Estimate le;
  le.sim_s = long_none.sim_s;
  le.card_kernel_s = long_none.card_kernel_s;
  VCHECK(should_offload(le, c) && !streak_offload(long_none, c, true));
  StreakLaunch no_estimate = s;
  no_estimate.sim_s = 0;
  VCHECK(!streak_offload(no_estimate, c, true));
  no_estimate.sim_s = -1;
  VCHECK(!streak_offload(no_estimate, c, true));
  no_estimate.sim_s = -1e-6;   // (with the pull added this would pass both bars)
  VCHECK(!streak_offload(no_estimate, c, true));
  no_estimate.sim_s = std::numeric_limits<double>::quiet_NaN();
  VCHECK(!streak_offload(no_estimate, c, true));
}

VTEST(a_streak_launch_is_priced_with_the_pull_it_saves_and_the_push_it_needs) {
  const Costs c;
  StreakLaunch s;
  s.sim_s = 0.00015;
  s.card_only_bytes = 48ull << 20;
  s.card_kernel_s = 20e-6;
  // the two sides, by the formula
  VCHECK(streak_keep_seconds(s, c) == s.sim_s + static_cast<double>(s.card_only_bytes) / c.pull_bytes_per_s);
  VCHECK(streak_card_seconds(s, c) == c.launch_s + 20e-6);
  s.push_bytes = 1 << 20;
  VCHECK(streak_card_seconds(s, c) == c.launch_s + 20e-6 + static_cast<double>(1 << 20) / c.push_bytes_per_s);
  // the same launch judged by should_offload alone stays: 150 us is under the floor, and the pull is not seen
  Estimate e;
  e.sim_s = s.sim_s;
  e.push_bytes = s.push_bytes;
  e.card_kernel_s = s.card_kernel_s;
  VCHECK(!should_offload(e, c));
  VCHECK(streak_offload(s, c, true));
  // before the kernel has run on the card its time there is kernel_s_default
  StreakLaunch fresh = s;
  fresh.card_kernel_s.reset();
  VCHECK(streak_card_seconds(fresh, c) == c.launch_s + c.kernel_s_default + static_cast<double>(1 << 20) / c.push_bytes_per_s);
  // what the card needs pushed counts against it: much to push and little to pull back stays
  StreakLaunch heavy = s;
  heavy.card_only_bytes = 1 << 20;
  heavy.push_bytes = 64ull << 20;
  VCHECK(!streak_offload(heavy, c, true));
  heavy.push_bytes = 0;
  VCHECK(streak_offload(heavy, c, true));
  // the rates are the Costs'
  Costs free_pull = c;
  free_pull.pull_bytes_per_s = 1e15;   // only the simulator's 150 us is left: under the floor
  VCHECK(!streak_offload(s, free_pull, true));
  Costs dear_push = c;
  dear_push.push_bytes_per_s = 1e7;    // 1 MiB to push costs 0.1 s
  VCHECK(!streak_offload(s, dear_push, true));
  Costs slow_launch = c;
  slow_launch.launch_s = 1.0;
  VCHECK(!streak_offload(s, slow_launch, true));
  Costs slow_default = c;
  slow_default.kernel_s_default = 1.0;
  VCHECK(streak_offload(s, slow_default, true) && !streak_offload(fresh, slow_default, true));
}

VTEST(a_streak_launch_keeps_the_margin_and_the_floor) {
  const Costs c;
  // a small live memory: 64 KiB to pull (6.6 us) and a kernel of 5 us; 50 us of simulator time beats 2 x 12 us, but the floor of 200 us asks more
  StreakLaunch s;
  s.card_only_bytes = 64 * 1024;
  s.card_kernel_s = 5e-6;
  s.sim_s = 50e-6;
  VCHECK(streak_keep_seconds(s, c) > c.margin * streak_card_seconds(s, c));
  VCHECK(!streak_offload(s, c, true));
  // the floor is on the simulator's whole side, time plus pull
  const double pull = static_cast<double>(s.card_only_bytes) / c.pull_bytes_per_s;
  s.sim_s = c.min_sim_s - pull + 1e-9;
  VCHECK(streak_offload(s, c, true));
  s.sim_s = c.min_sim_s - pull - 1e-9;
  VCHECK(!streak_offload(s, c, true));
  // exactly at the floor: goes (the floor is "at least"); 1e6 bytes at 10 GB/s and 100 us add up to 200 us exactly in binary
  StreakLaunch at;
  at.card_only_bytes = 1000000;
  at.card_kernel_s = 5e-6;
  at.sim_s = 1e-4;
  VCHECK(streak_keep_seconds(at, c) == c.min_sim_s && streak_offload(at, c, true));
  // exactly at the margin: stays (the margin is "more than"); every number here is exact in binary
  Costs exact = c;
  exact.launch_s = 0.25;
  exact.pull_bytes_per_s = 2;
  StreakLaunch edge;
  edge.card_only_bytes = 1;     // 0.5 s to pull
  edge.card_kernel_s = 0.25;
  edge.sim_s = 0.5;
  VCHECK(streak_keep_seconds(edge, exact) == 1.0 && exact.margin * streak_card_seconds(edge, exact) == 1.0 && !streak_offload(edge, exact, true));
  edge.sim_s = 0.5 + 1e-9;
  VCHECK(streak_offload(edge, exact, true));
  // the margin, found by bisection on the simulator's time against the formula: the launch goes just above margin x the card side, not at it
  StreakLaunch m;
  m.card_only_bytes = 1ull << 20;
  m.push_bytes = 8ull << 20;
  m.card_kernel_s = 300e-6;
  const double card = streak_card_seconds(m, c);
  const double mpull = static_cast<double>(m.card_only_bytes) / c.pull_bytes_per_s;
  double lo = 0, hi = 1;
  for (int i = 0; i < 200; ++i) {
    m.sim_s = (lo + hi) / 2;
    (streak_offload(m, c, true) ? hi : lo) = m.sim_s;
  }
  VCHECK(std::abs(hi - (c.margin * card - mpull)) < 1e-12);
  VCHECK(c.margin * card - mpull > c.min_sim_s);   // (the bar found is the margin's, not the floor's)
  Costs one = c;
  one.margin = 1.0;
  m.sim_s = card - mpull + 1e-9;
  VCHECK(streak_offload(m, one, true) && !streak_offload(m, c, true));
}

VTEST(the_bans_weigh_a_launch_against_the_pull_it_saves_while_the_card_holds_memory) {
  const Costs c;
  // GRAMSCHM's kernel1 on the rig: 0.29 ms estimated, a card path that once took 0.5 ms, 48 MiB only on the card
  StreakLaunch s;
  s.sim_s = 0.00029;
  s.card_only_bytes = 48ull << 20;
  const double card_total = 0.0005;
  VCHECK(ban_reference_seconds(s, c, true) == streak_keep_seconds(s, c));
  VCHECK(!offload_paid_off(card_total, s.sim_s));                          // judged by the estimate alone: banned
  VCHECK(offload_paid_off(card_total, ban_reference_seconds(s, c, true)));  // with the pull it saved: not
  // the traffic ban's "slow" side reads the same reference: two launches that push a quarter of the memory, slow against the estimate alone
  TrafficTable old_t, new_t;
  const uint64_t live = 64ull << 20, push = 20ull << 20;
  const double total = 0.002;   // over half of 0.29 ms, under half of 0.29 ms + 5 ms
  VCHECK(!old_t.record("k", push, live, total, s.sim_s) && old_t.record("k", push, live, total, s.sim_s));
  VCHECK(!new_t.record("k", push, live, total, ban_reference_seconds(s, c, true)) && !new_t.record("k", push, live, total, ban_reference_seconds(s, c, true)));
  // and the reverse: with nothing only on the card, or with the rule switched off, the estimate alone, and the same slow launch is banned
  StreakLaunch none = s;
  none.card_only_bytes = 0;
  VCHECK(ban_reference_seconds(none, c, true) == none.sim_s && !offload_paid_off(card_total, ban_reference_seconds(none, c, true)));
  VCHECK(ban_reference_seconds(s, c, false) == s.sim_s && !offload_paid_off(card_total, ban_reference_seconds(s, c, false)));
  // a launch slower than even the pull it saves is still banned
  VCHECK(!offload_paid_off(0.1, ban_reference_seconds(s, c, true)));
  // what is pushed does not enter the reference (it is the card's side), nor the card's kernel time
  StreakLaunch pushed = s;
  pushed.push_bytes = 1ull << 30;
  pushed.card_kernel_s = 1.0;
  VCHECK(ban_reference_seconds(pushed, c, true) == ban_reference_seconds(s, c, true));
}

VTEST(a_launch_with_a_zero_grid_or_block_dimension_is_the_simulator_s) {
  const uint32_t one[3] = {1, 1, 1};
  VCHECK(!launch_shape_empty(one, one));
  const uint32_t big[3] = {65535, 65535, 64};
  VCHECK(!launch_shape_empty(big, big));
  for (int i = 0; i < 3; ++i) {
    uint32_t g[3] = {4, 4, 4}, b[3] = {32, 8, 1};
    g[i] = 0;
    VCHECK(launch_shape_empty(g, b));   // PolyBench LU's last launches: grid 0x1x1 and 0x0x1
    uint32_t g2[3] = {4, 4, 4}, b2[3] = {32, 8, 1};
    b2[i] = 0;
    VCHECK(launch_shape_empty(g2, b2));
  }
}

// ---- what a kernel reaches besides its code (Round sp66) ----

VTEST(externs_by_kind_and_what_each_mode_admits) {
  using K = ExternKind;
  VCHECK(extern_kind("vprintf", false) == K::Printf);
  VCHECK(extern_kind("__assertfail", false) == K::Assert && extern_kind("__assert_fail", false) == K::Assert);
  VCHECK(extern_kind("malloc", false) == K::Heap && extern_kind("free", false) == K::Heap);
  VCHECK(extern_kind("cudaLaunchDeviceV2", true) == K::DeviceLaunch);
  VCHECK(extern_kind("vprintf", true) == K::DeviceLaunch);   // (the device runtime's own name wins)
  VCHECK(extern_kind("printf", false) == K::Other && extern_kind("vprintf_", false) == K::Other && extern_kind("", false) == K::Other);
  VCHECK(extern_kind("Malloc", false) == K::Other && extern_kind("__nv_malloc", false) == K::Other);
  // only vprintf, only in turbo mode, only when not switched off
  VCHECK(extern_admitted(K::Printf, Mode::Turbo, false));
  VCHECK(!extern_admitted(K::Printf, Mode::Turbo, true));
  VCHECK(!extern_admitted(K::Printf, Mode::Exact, false));
  VCHECK(!extern_admitted(K::Printf, Mode::Off, false));
  for (K k : {K::Assert, K::Heap, K::DeviceLaunch, K::Other})
    for (Mode m : {Mode::Exact, Mode::Turbo}) VCHECK(!extern_admitted(k, m, false));
  const std::set<std::string> names = {extern_kind_name(K::Printf), extern_kind_name(K::Assert), extern_kind_name(K::Heap), extern_kind_name(K::DeviceLaunch),
                                       extern_kind_name(K::Other)};
  VCHECK(names.size() == 5 && names.count("vprintf") && names.count("__assertfail") && names.count("malloc/free"));
}

VTEST(texture_and_surface_instructions_are_named_and_never_admitted) {
  for (const char* m : {"TEX", "TEXS", "TLD", "TLDS", "TLD4", "TLD4S", "TXD", "TXQ", "TMML", "SULD", "SUST", "SUATOM", "SURED", "SUQUERY"}) {
    VCHECK(texture_mnemonic(m));
    VCHECK(turbo_class(m, {}) == InsnClass::Never);   // what lets the device-wide refusal go: no class admits one
    VCHECK(!exact_opcode_allowed(m, {}));
  }
  for (const char* m : {"LDG", "LDS", "LDC", "TEXDEPBAR", "ST", "SUB", "TLD5", "", "tex", "TE"}) VCHECK(!texture_mnemonic(m));
}

namespace {
// A cubin-shaped ELF64 for the card image: sections by name, symbols, relocation tables (REL or RELA) against a section.
struct ElfBuilder {
  struct Sec { std::string name; uint32_t type; std::vector<uint8_t> bytes; uint64_t nobits = 0; uint32_t link = 0, info = 0; uint64_t entsize = 0; };
  struct Sym { std::string name; uint16_t shndx; uint64_t value, size; uint8_t info; };
  struct Rel { uint64_t offset; uint32_t sym, type; int64_t addend; };
  std::vector<Sec> secs{{"", 0, {}}};
  std::vector<Sym> syms{{"", 0, 0, 0, 0}};
  uint16_t add(const std::string& name, uint32_t type, std::vector<uint8_t> bytes, uint64_t nobits = 0) {
    secs.push_back({name, type, std::move(bytes), nobits});
    return static_cast<uint16_t>(secs.size() - 1);
  }
  uint32_t sym(const std::string& name, uint16_t shndx, uint64_t value, uint64_t size = 4, uint8_t info = 0x11) {
    syms.push_back({name, shndx, value, size, info});
    return static_cast<uint32_t>(syms.size() - 1);
  }
  // Adds the relocation tables, the symbol and string tables and the section-name table, and lays the image out.
  std::vector<uint8_t> build(const std::vector<std::tuple<uint16_t, bool, std::vector<Rel>>>& tables, std::vector<size_t>* table_index = nullptr) {
    std::vector<uint8_t> strtab{0};
    std::vector<uint8_t> symtab;
    const auto put = [](std::vector<uint8_t>& v, uint64_t x, int n) { for (int k = 0; k < n; ++k) v.push_back(static_cast<uint8_t>(x >> (8 * k))); };
    for (const Sym& y : syms) {
      const uint32_t name_at = static_cast<uint32_t>(y.name.empty() ? 0 : strtab.size());
      if (!y.name.empty()) strtab.insert(strtab.end(), y.name.begin(), y.name.end()), strtab.push_back(0);
      put(symtab, name_at, 4);
      put(symtab, y.info, 1);
      put(symtab, 0, 1);
      put(symtab, y.shndx, 2);
      put(symtab, y.value, 8);
      put(symtab, y.size, 8);
    }
    const uint16_t st = add(".symtab", 2, symtab);
    const uint16_t ss = add(".strtab", 3, strtab);
    secs[st].link = ss;
    secs[st].entsize = 24;
    for (const auto& [dst, rela, rels] : tables) {
      std::vector<uint8_t> t;
      for (const Rel& r : rels) {
        put(t, r.offset, 8);
        put(t, (uint64_t{r.sym} << 32) | r.type, 8);
        if (rela) put(t, static_cast<uint64_t>(r.addend), 8);
      }
      const uint16_t i = add((rela ? ".rela" : ".rel") + secs[dst].name, rela ? 4 : 9, t);
      secs[i].link = st;
      secs[i].info = dst;
      secs[i].entsize = rela ? 24 : 16;
      if (table_index) table_index->push_back(i);
    }
    std::vector<uint8_t> shstr{0};
    std::vector<uint32_t> name_at(secs.size() + 1, 0);
    for (size_t i = 1; i < secs.size(); ++i) {
      name_at[i] = static_cast<uint32_t>(shstr.size());
      shstr.insert(shstr.end(), secs[i].name.begin(), secs[i].name.end());
      shstr.push_back(0);
    }
    name_at[secs.size()] = static_cast<uint32_t>(shstr.size());
    const std::string own = ".shstrtab";
    shstr.insert(shstr.end(), own.begin(), own.end());
    shstr.push_back(0);
    secs.push_back({own, 3, shstr});
    std::vector<uint8_t> img(64, 0);
    std::vector<uint64_t> off(secs.size(), 0);
    for (size_t i = 1; i < secs.size(); ++i) {
      while (img.size() % 8) img.push_back(0);
      off[i] = img.size();
      img.insert(img.end(), secs[i].bytes.begin(), secs[i].bytes.end());
    }
    while (img.size() % 8) img.push_back(0);
    const uint64_t shoff = img.size();
    for (size_t i = 0; i < secs.size(); ++i) {
      put(img, name_at[i], 4);
      put(img, secs[i].type, 4);
      put(img, 0, 8);
      put(img, 0, 8);
      put(img, off[i], 8);
      put(img, secs[i].type == 8 ? secs[i].nobits : secs[i].bytes.size(), 8);
      put(img, secs[i].link, 4);
      put(img, secs[i].info, 4);
      put(img, 8, 8);
      put(img, secs[i].entsize, 8);
    }
    const uint8_t ident[16] = {0x7f, 'E', 'L', 'F', 2, 1, 1, 0x33, 7};
    std::memcpy(img.data(), ident, 16);
    const auto at = [&](size_t o, uint64_t x, int n) { for (int k = 0; k < n; ++k) img[o + k] = static_cast<uint8_t>(x >> (8 * k)); };
    at(0x10, 2, 2);        // ET_EXEC
    at(0x12, 190, 2);      // EM_CUDA
    at(0x28, shoff, 8);
    at(0x34, 64, 2);
    at(0x3a, 64, 2);
    at(0x3c, secs.size(), 2);
    at(0x3e, secs.size() - 1, 2);
    return img;
  }
};
uint64_t rd64(const std::vector<uint8_t>& v, size_t o) { uint64_t x; std::memcpy(&x, &v[o], 8); return x; }
// A section's header field in an image ElfBuilder made (sections in the order added).
uint64_t sh_field(const std::vector<uint8_t>& img, size_t i, size_t field_off) { return rd64(img, rd64(img, 0x28) + 64 * i + field_off); }
}  // namespace

VTEST(the_card_image_has_the_simulator_s_addresses_of_the_module_s_variables) {
  ElfBuilder b;
  std::vector<uint8_t> code(64);
  for (size_t k = 0; k < code.size(); ++k) code[k] = static_cast<uint8_t>(0x40 + k);
  const uint16_t text = b.add(".text.k", 1, code);
  const uint16_t c4 = b.add(".nv.constant4", 1, std::vector<uint8_t>(24, 0xee));
  const uint16_t glob = b.add(".nv.global", 8, {}, 256);
  const uint16_t c3 = b.add(".nv.constant3", 1, std::vector<uint8_t>(32, 0));
  const uint16_t init = b.add(".nv.global.init", 1, std::vector<uint8_t>(16, 0));
  const uint32_t g = b.sym("g", glob, 8, 64);
  const uint32_t c = b.sym("c", c3, 4, 16);
  const uint32_t str = b.sym("$str", init, 0, 16, 0x01);
  const uint32_t vp = b.sym("vprintf", 0, 0, 0, 0x12);
  const uint32_t kf = b.sym("k", text, 0, 64, 0x12);
  std::vector<size_t> tab;
  const std::vector<uint8_t> img = b.build({{text, true, {{0, g, 56, 4}, {16, g, 57, 4}, {32, vp, 58, 0}, {48, str, 56, 0}}},
                                            {c4, false, {{0, c, 2, 0}, {8, kf, 35, 0}, {16, g, 4, 0}}}}, &tab);
  const uint64_t gva = 0x7f0000100000ull, cva = 0x7f0000200000ull, iva = 0x7f0000300000ull;
  const auto vars = [&](const std::string& sec) -> std::optional<uint64_t> {
    if (sec == ".nv.global") return gva;
    if (sec == ".nv.constant3") return cva;
    if (sec == ".nv.global.init") return iva;
    return std::nullopt;
  };
  const CardImage ci = card_image_at_sim_addresses(img.data(), img.size(), vars);
  VCHECK(ci.ok && ci.why.empty());
  VCHECK(ci.applied == 5 && ci.bytes.size() == img.size());
  const uint64_t text_off = sh_field(img, text, 24), c4_off = sh_field(img, c4, 24);
  // the code: the low half of g + 8 + 4 at bits 32-63 of the first instruction, the high half in the second, the string's low half in the fourth
  uint64_t lo0 = rd64(ci.bytes, text_off), lo1 = rd64(ci.bytes, text_off + 16), lo3 = rd64(ci.bytes, text_off + 48);
  VCHECK((lo0 >> 32) == ((gva + 12) & 0xffffffffu) && (lo0 & 0xffffffffu) == (rd64(img, text_off) & 0xffffffffu));
  VCHECK((lo1 >> 32) == ((gva + 12) >> 32));
  VCHECK((lo3 >> 32) == (iva & 0xffffffffu));
  VCHECK(rd64(ci.bytes, text_off + 8) == rd64(img, text_off + 8));   // (the high word of a MOV is not the relocation's)
  VCHECK(rd64(ci.bytes, text_off + 32) == rd64(img, text_off + 32) && rd64(ci.bytes, text_off + 40) == rd64(img, text_off + 40));   // vprintf's call: the driver's
  // the bank: c's address (cva + 4) and g's (G64); the function descriptor left alone
  VCHECK(rd64(ci.bytes, c4_off) == cva + 4 && rd64(ci.bytes, c4_off + 16) == gva + 8 && rd64(ci.bytes, c4_off + 8) == 0xeeeeeeeeeeeeeeeeull);
  // the tables: only what was left, in order, and their sizes say so
  VCHECK(sh_field(ci.bytes, tab[0], 32) == 24 && sh_field(img, tab[0], 32) == 96);
  const uint64_t t0 = sh_field(ci.bytes, tab[0], 24);
  VCHECK(rd64(ci.bytes, t0) == 32 && rd64(ci.bytes, t0 + 8) == ((uint64_t{vp} << 32) | 58));
  VCHECK(sh_field(ci.bytes, tab[1], 32) == 16);
  const uint64_t t1 = sh_field(ci.bytes, tab[1], 24);
  VCHECK(rd64(ci.bytes, t1) == 8 && rd64(ci.bytes, t1 + 8) == ((uint64_t{kf} << 32) | 35));
  // nothing else of the image moved
  size_t changed = 0;
  for (size_t k = 0; k < img.size(); ++k) changed += img[k] != ci.bytes[k];
  VCHECK(changed > 0 && changed <= 4 * 4 + 2 * 8 + 2 * 8 + 2 * 24 + 2);
  // no variable section named: the image as it was
  const CardImage same = card_image_at_sim_addresses(img.data(), img.size(), [](const std::string&) -> std::optional<uint64_t> { return std::nullopt; });
  VCHECK(same.ok && same.applied == 0 && same.bytes == img);
  // only the constant bank's variables: the code's references to g are the driver's
  const CardImage only_c = card_image_at_sim_addresses(img.data(), img.size(), [&](const std::string& sec) -> std::optional<uint64_t> {
    return sec == ".nv.constant3" ? std::optional<uint64_t>(cva) : std::nullopt;
  });
  VCHECK(only_c.ok && only_c.applied == 1 && rd64(only_c.bytes, c4_off) == cva + 4 && rd64(only_c.bytes, text_off) == rd64(img, text_off));
  VCHECK(sh_field(only_c.bytes, tab[0], 32) == 96 && sh_field(only_c.bytes, tab[1], 32) == 32);
}

VTEST(the_card_image_refuses_what_it_cannot_apply) {
  const auto vars = [](const std::string& sec) -> std::optional<uint64_t> {
    return sec == ".nv.global" ? std::optional<uint64_t>(0x7f0000100000ull) : std::nullopt;
  };
  const auto one = [&](uint32_t type, bool into_code, bool into_nobits, uint64_t offset) {
    ElfBuilder b;
    const uint16_t text = b.add(".text.k", 1, std::vector<uint8_t>(32, 0));
    const uint16_t c4 = b.add(".nv.constant4", 1, std::vector<uint8_t>(16, 0));
    const uint16_t glob = b.add(".nv.global", 8, {}, 64);
    const uint32_t g = b.sym("g", glob, 0);
    const std::vector<uint8_t> img = b.build({{into_nobits ? glob : into_code ? text : c4, true, {{offset, g, type, 0}}}});
    return card_image_at_sim_addresses(img.data(), img.size(), vars);
  };
  VCHECK(one(56, true, false, 0).ok && one(57, true, false, 16).ok && one(58, true, false, 0).ok && one(75, true, false, 16).ok);
  VCHECK(one(62, true, false, 0).ok && one(63, true, false, 0).ok);
  VCHECK(one(2, false, false, 0).ok && one(4, false, false, 8).ok);
  VCHECK(!one(35, false, false, 0).ok);                      // a function descriptor of a variable
  VCHECK(!one(2, true, false, 0).ok && !one(56, false, false, 0).ok);   // a data type in code, a code type in data
  VCHECK(!one(1, true, false, 0).ok);                        // a type nobody named
  VCHECK(!one(56, true, false, 17).ok && !one(56, true, false, 32).ok);   // past the end of the code
  VCHECK(!one(2, false, false, 9).ok && !one(2, false, false, 16).ok);    // past the end of the bank
  VCHECK(one(2, false, false, 8).ok);                        // the last 8 bytes
  VCHECK(!one(2, false, true, 0).ok);                        // into a section with no bytes
  const CardImage bad = one(35, false, false, 0);
  VCHECK(bad.bytes.empty() && bad.why.find("35") != std::string::npos);
  // not an image
  const uint8_t junk[80] = {0x7f, 'E', 'L', 'F', 1};
  VCHECK(!card_image_at_sim_addresses(junk, sizeof junk, vars).ok);
  VCHECK(!card_image_at_sim_addresses(junk, 10, vars).ok);
  std::vector<uint8_t> trunc;
  {
    ElfBuilder b;
    b.add(".text.k", 1, std::vector<uint8_t>(32, 0));
    trunc = b.build({});
  }
  VCHECK(card_image_at_sim_addresses(trunc.data(), trunc.size(), vars).ok);
  VCHECK(!card_image_at_sim_addresses(trunc.data(), trunc.size() - 1, vars).ok);   // the section headers run past the end
  std::vector<uint8_t> elf32 = trunc;
  elf32[4] = 1;   // ELFCLASS32: the same bytes are not a 64-bit image
  VCHECK(!card_image_at_sim_addresses(elf32.data(), elf32.size(), vars).ok);
  std::vector<uint8_t> nomagic = trunc;
  nomagic[1] = 'e';
  VCHECK(!card_image_at_sim_addresses(nomagic.data(), nomagic.size(), vars).ok);
}

VTEST(code_relocations_write_the_bits_the_loader_writes) {
  uint8_t w[16];
  const auto fill = [&] { for (int k = 0; k < 16; ++k) w[k] = static_cast<uint8_t>(0xa0 + k); };
  const auto lo = [&] { uint64_t x; std::memcpy(&x, w, 8); return x; };
  const auto hi = [&] { uint64_t x; std::memcpy(&x, w + 8, 8); return x; };
  fill();
  const uint64_t lo0 = lo(), hi0 = hi();
  const uint64_t v = 0x0001234589abcdefull;
  vgpu::sass::patch_code_word(w, 56, v);
  VCHECK(lo() == ((lo0 & 0xffffffffu) | (uint64_t{0x89abcdef} << 32)) && hi() == hi0);
  fill();
  vgpu::sass::patch_code_word(w, 62, v);
  VCHECK(lo() == ((lo0 & 0xffffffffu) | (uint64_t{0x89abcdef} << 32)) && hi() == hi0);
  fill();
  vgpu::sass::patch_code_word(w, 57, v);
  VCHECK(lo() == ((lo0 & 0xffffffffu) | (uint64_t{0x00012345} << 32)) && hi() == hi0);
  fill();
  vgpu::sass::patch_code_word(w, 63, v);
  VCHECK(lo() == ((lo0 & 0xffffffffu) | (uint64_t{0x00012345} << 32)) && hi() == hi0);
  fill();
  vgpu::sass::patch_code_word(w, 58, v);   // 49 bits at 32
  VCHECK(lo() == ((lo0 & 0xffffffffu) | (v << 32)) && hi() == ((hi0 & ~uint64_t{0x1ffff}) | ((v >> 32) & 0x1ffff)));
  fill();
  const uint64_t a = 0x0000123456789abcull & ~uint64_t{3};
  vgpu::sass::patch_code_word(w, 75, a);   // words: 8 bits at 16, the rest at 34
  const uint64_t words = a >> 2;
  VCHECK(((lo() >> 16) & 0xff) == (words & 0xff) && (lo() >> 34) == ((words >> 8) & 0x3fffffff) && (hi() & 0x1ffff) == ((words >> 38) & 0x1ffff));
  VCHECK((lo() & 0xffff) == (lo0 & 0xffff) && ((lo() >> 24) & 0x3ff) == ((lo0 >> 24) & 0x3ff) && (hi() >> 17) == (hi0 >> 17));
  for (uint32_t t : {56u, 57u, 58u, 62u, 63u, 75u}) VCHECK(vgpu::sass::code_reloc_type(t));
  for (uint32_t t : {0u, 1u, 2u, 4u, 35u, 55u, 59u, 64u, 74u, 76u}) VCHECK(!vgpu::sass::code_reloc_type(t));
}

// ---- bounded blocks (Round sp75) ------------------------------------------------------------------------------------------------

VTEST(a_bounded_block_of_a_sampled_grid_has_a_fixed_limit_that_fits_the_wait) {
  vgpu::exec::LaunchPart p;
  VCHECK(vgpu::exec::part_block_limit(p, 400, 3) == 0);          // off: whole blocks, as before the round
  p.block_limit_s = kSampleBlockSeconds;
  p.block_cap_s = kSampleBlockSeconds;
  for (uint64_t units : {uint64_t{256}, uint64_t{400}, uint64_t{1} << 20})
    for (unsigned threads : {1u, 3u, 4u, 64u}) VCHECK(vgpu::exec::part_block_limit(p, units, threads) == kSampleBlockSeconds);
  // both runs of a block (the first, and the timed one) fit the wait the process already gave a copy
  VCHECK(2 * kSampleBlockSeconds + 0.020 <= kSampleTimeoutSeconds);   // ... with 20 ms to spare
  VCHECK(kSampleBlockSeconds > 5 * kSampleBudgetSeconds);        // ... and a block the budget would have let run whole still does
}

VTEST(a_block_stopped_at_the_limit_counts_for_the_time_it_ran) {
  // HeCBench aidw on server3: 400 blocks of 0.58 s each, 3 host threads, 2.7 MiB live. The copy did not answer in 50 ms and the launch (78 s) stayed.
  const Costs c;
  const uint64_t live = 2867200;
  Sample none;                                                    // the copy that did not answer: no block timed
  none.total = 400;
  none.threads = 3;
  VCHECK(!sampled_launch_seconds(none).has_value());
  VCHECK(!sample_offload(none, live, live, c, false));
  Sample stopped;                                                 // the first block, stopped at the limit: its timed run of the same rounds
  stopped.seconds = 0.0138;
  stopped.timed = 1;
  stopped.total = 400;
  stopped.threads = 3;
  const double est = *sampled_launch_seconds(stopped);
  VCHECK(est > 1.83 && est < 1.85);                               // under the 78 s the launch took, and over what the card path needs
  VCHECK(sample_offload(stopped, live, live, c, false));
  // the same block on a host with 64 threads says 0.09 s: not enough, and the launch stays (a lost chance, not a loss)
  stopped.threads = 64;
  VCHECK(!sample_offload(stopped, live, live, c, false));
  // a block that ran the limit of a grid of 256 blocks on 4 threads: 0.96 s, which repays the start; on 8 threads it does not
  Sample least;
  least.seconds = kSampleBlockSeconds;
  least.timed = 1;
  least.total = kSampleMinBlocks;
  least.threads = 4;
  VCHECK(sample_offload(least, 0, 0, c, false));
  least.threads = 8;
  VCHECK(!sample_offload(least, 0, 0, c, false));
}

VTEST(sample_need_is_the_least_time_the_sample_rule_sends) {
  const Costs c;
  for (bool started : {false, true})
    for (uint64_t live : {uint64_t{0}, uint64_t{2867200}, uint64_t{201326592}, uint64_t{805306368}, uint64_t{4} << 30}) {
      const double need = sample_need_seconds(live, live, c, started);
      VCHECK(need > 0);
      VCHECK(!sampled_offload(need * 0.999, live, live, c, started));
      VCHECK(sampled_offload(need * 1.001, live, live, c, started));
    }
  // the CUDA Samples' tensor-core GEMMs: 192 MiB live asks 0.765 s of the launch, 768 MiB 1.06 s
  const double tc = sample_need_seconds(201326592, 201326592, c, false), tf = sample_need_seconds(805306368, 805306368, c, false);
  VCHECK(tc > 0.76 && tc < 0.77);
  VCHECK(tf > 1.06 && tf < 1.07);
  // with the driver up and nothing to move, the floor of the size rule
  VCHECK(sample_need_seconds(0, 0, c, true) >= c.min_sim_s && sample_need_seconds(0, 0, c, true) < 0.001);
  Costs floor = c;
  floor.min_sim_s = 0.010;                                        // the size rule's floor is the least, whatever the card path
  VCHECK(sample_need_seconds(0, 0, floor, true) == 0.010);
  VCHECK(!sampled_offload(0.0099, 0, 0, floor, true) && sampled_offload(0.0101, 0, 0, floor, true));
  Costs slow = c;
  slow.margin = 8.0;                                              // a margin no card time of 1/8 passes
  VCHECK(sample_need_seconds(1 << 20, 1 << 20, slow, false) > 1e20);
}

VTEST(a_small_grid_is_sampled_by_bounded_blocks_when_a_stopped_block_can_show_enough) {
  const Costs c;
  vgpu::exec::LaunchPart p;
  p.block_limit_s = kSampleBlockSeconds;
  p.block_cap_s = kSampleBlockCapSeconds;
  // cudaTensorCoreGemm on the A10G rig: 80 blocks, 4 host threads, 192 MiB live: a block must last 53.5 ms
  p.need_s = sample_need_seconds(201326592, 201326592, c, false);
  const double tc = vgpu::exec::part_block_limit(p, 80, 4);
  VCHECK(tc > 0.053 && tc < 0.054 && tc <= kSampleBlockCapSeconds);
  // ... and a block stopped there, whose timed run was 0.95 of the limit, sends the launch
  Sample s;
  s.seconds = 0.95 * tc;
  s.timed = 1;
  s.total = 80;
  s.threads = 4;
  VCHECK(sample_offload(s, 201326592, 201326592, c, false));
  s.seconds = 0.70 * tc;                                          // (the slack is 1.4: under 0.71 of the limit it would not)
  VCHECK(!sample_offload(s, 201326592, 201326592, c, false));
  // tf32TensorCoreGemm: 160 blocks, 768 MiB
  p.need_s = sample_need_seconds(805306368, 805306368, c, false);
  const double tf = vgpu::exec::part_block_limit(p, 160, 4);
  VCHECK(tf > 0.037 && tf < 0.038);
  // on 16 host threads a block of the 80 would have to be watched for 0.19 s: over the cap, so no block is run
  p.need_s = sample_need_seconds(201326592, 201326592, c, false);
  VCHECK(vgpu::exec::part_block_limit(p, 80, 16) == 0);
  VCHECK(vgpu::exec::part_block_limit(p, 80, 4) > 0 && vgpu::exec::part_block_limit(p, 80, 5) == 0);
  // a grid whose blocks need little: the limit is never under the one of a sampled grid
  p.need_s = 0.5;
  VCHECK(vgpu::exec::part_block_limit(p, 255, 1) == kSampleBlockSeconds);
  // the blocks taken and the wait
  VCHECK(bounded_blocks(1) == 0 && bounded_blocks(2) == 2 && bounded_blocks(80) == 32 && bounded_blocks(255) == 32 && bounded_blocks(256) == 0);
  VCHECK(bounded_stride(80) == 2 && bounded_stride(160) == 5 && bounded_stride(2) == 1 && bounded_stride(256) == 1);
  VCHECK(bounded_timeout_seconds(0) == kSampleTimeoutSeconds + 2 * kSampleBlockCapSeconds);
  VCHECK(bounded_timeout_seconds(1ull << 30) == sample_timeout_seconds(1ull << 30) + 2 * kSampleBlockCapSeconds);
}

VTEST(a_small_grid_without_a_followed_loop_is_sampled_only_with_bounded_blocks_and_a_counted_30_ms) {
  const Costs c;
  // cudaTensorCoreGemm on server3: 80 blocks of 256 threads, 7,248 instructions (46 ms counted), 192 MiB live, the process 0.44 s old
  SampleCandidate w;
  w.blocks = 80;
  w.loop = true;
  w.live_bytes = 201326592;
  w.static_s = 0.046387;
  w.age_s = 0.444;
  w.expected_s = kSampleCostGuessSeconds + kSampleSecondsPerLiveByte * 201326592.0;
  VCHECK(!w.bounded);                                             // the rule before the round (VGPU_HOST_GPU_SAMPLE_BLOCK=0): not sampled
  VCHECK(!worth_sampling(w, c, 0.29) && sample_refusal(w, c, 0.29) == NoSample::SmallGrid && !sample_bounded_small(w));
  w.bounded = true;
  VCHECK(sample_bounded_small(w) && !sample_inside(w));
  VCHECK(worth_sampling(w, c, 0.29) && sample_refusal(w, c, 0.29) == NoSample::None);
  VCHECK(sample_floor(w) == kSampleStaticFloorSeconds);          // not the 0.36 s the live memory asks of a sampled grid
  VCHECK(sample_static_floor(w.live_bytes) > 0.36);
  // tf32 (160 blocks, 9,088 instructions: 116 ms; 768 MiB) and bf16 (3,592: 46 ms)
  w.blocks = 160;
  w.live_bytes = 805306368;
  w.static_s = 0.116326;
  VCHECK(worth_sampling(w, c, 0.39));
  w.static_s = 0.045978;
  VCHECK(worth_sampling(w, c, 0.39));
  // under 30 ms counted the grid is not sampled, in a young process: Rodinia lavaMD (216 blocks, 3.4 ms), HeCBench s3d (108 blocks, 1.4 to 9.7 ms)
  w.blocks = 216;
  w.live_bytes = 1696984;
  w.static_s = 0.003387;
  w.age_s = 0.004;
  w.expected_s = kSampleCostGuessSeconds;
  VCHECK(!worth_sampling(w, c, 0.26) && sample_refusal(w, c, 0.26) == NoSample::Floor);
  w.static_s = kSampleStaticFloorSeconds * 0.999;
  VCHECK(!worth_sampling(w, c, 0.26));
  w.static_s = kSampleStaticFloorSeconds;
  VCHECK(worth_sampling(w, c, 0.26));
  // under the floor a small grid is sampled only by a process old enough to afford two runs of a block at the cap: HeCBench bilateral's third
  // kernel (128 blocks, 6 ms counted) comes 0.2 s into the process, and was sampled for 13 ms of a program of 0.45 s before that was counted
  w.blocks = 128;
  w.static_s = 0.00614;
  w.age_s = 0.2;
  VCHECK(sample_expected_seconds(w) == kSampleCostGuessSeconds + 2 * kSampleBlockCapSeconds);
  VCHECK(!worth_sampling(w, c, 0.26) && sample_refusal(w, c, 0.26) == NoSample::Floor);
  w.age_s = 6.0;                                                  // 2 % of 6 s is under the 0.123 s
  VCHECK(!worth_sampling(w, c, 0.26));
  w.age_s = 6.2;
  VCHECK(worth_sampling(w, c, 0.26));
  w.blocks = 4096;                                                // a sampled grid is afforded as before: 3 ms, by a process 0.15 s old
  w.age_s = 0.2;
  VCHECK(sample_expected_seconds(w) == kSampleCostGuessSeconds && worth_sampling(w, c, 0.26));
  w.blocks = 216;
  w.static_s = kSampleStaticFloorSeconds;
  w.age_s = 0.004;
  // the other conditions stand: a loop, no short whole launch before, at least two blocks
  w.loop = false;
  VCHECK(sample_refusal(w, c, 0.26) == NoSample::NoLoop);
  w.loop = true;
  w.whole_s = 0.003;
  VCHECK(sample_refusal(w, c, 0.26) == NoSample::ShortBefore);
  w.whole_s.reset();
  w.blocks = 1;
  VCHECK(sample_refusal(w, c, 0.26) == NoSample::SmallGrid);
  // a small grid with a followed loop is still sampled inside its blocks (Round sp61), by that rule's floor
  w.blocks = 8;
  w.followed = true;
  VCHECK(sample_inside(w) && !sample_bounded_small(w));
  w.live_bytes = 1ull << 30;
  VCHECK(sample_floor(w) == sample_static_floor(1ull << 30));
  // a grid the sample of whole blocks takes keeps its floor, bounded blocks or not (warpAggregatedAtomicsCG: 0.325 s counted, 280 MiB live)
  SampleCandidate big;
  big.blocks = 19531;
  big.loop = true;
  big.bounded = true;
  big.live_bytes = 280000034;
  big.static_s = 0.324996;
  big.age_s = 0.34;
  big.expected_s = kSampleCostGuessSeconds + kSampleSecondsPerLiveByte * 280000034.0;
  VCHECK(!sample_bounded_small(big) && sample_floor(big) > 0.5 && sample_refusal(big, c, 0.3) == NoSample::Floor);
}

VTEST(the_reasons_a_launch_is_not_sampled_are_the_rule_itself) {
  const Costs c;
  std::set<std::string> texts;
  for (NoSample r : {NoSample::None, NoSample::NoLoop, NoSample::SmallGrid, NoSample::ShortBefore, NoSample::Floor}) texts.insert(no_sample_text(r));
  VCHECK(texts.size() == 5);
  for (uint64_t blocks : {uint64_t{1}, uint64_t{8}, uint64_t{80}, uint64_t{255}, uint64_t{256}, uint64_t{4096}})
    for (bool loop : {false, true})
      for (bool followed : {false, true})
        for (bool bounded : {false, true})
          for (double static_s : {0.001, 0.03, 1.0})
            for (double age : {0.004, 10.0})
              for (int whole = 0; whole < 3; ++whole) {
                SampleCandidate w;
                w.blocks = blocks;
                w.loop = loop;
                w.followed = followed;
                w.bounded = bounded;
                w.static_s = static_s;
                w.age_s = age;
                if (whole) w.whole_s = whole == 1 ? 0.003 : 5.0;
                VCHECK(worth_sampling(w, c, 0.26) == (sample_refusal(w, c, 0.26) == NoSample::None));
                if (!bounded) {   // without bounded blocks: the rule of Round sp61, written out
                  const bool takes = sample_blocks(blocks) != 0 || (followed && inblock_blocks(blocks) != 0);
                  const bool was = loop && takes && !(w.whole_s && *w.whole_s <= c.margin * 0.26) &&
                                   (sample_static_estimate(w) >= sample_static_floor(0) || sample_affordable(age, 0, w.expected_s));
                  VCHECK(worth_sampling(w, c, 0.26) == was);
                }
              }
}

// ---- Round sp76: a sample once the driver is up --------------------------------------------------------------------------------------

VTEST(a_launch_is_sampled_with_the_driver_up_and_never_under_the_rule_before) {
  // (start begun, driver ready, start failed, the rule on)
  VCHECK(sample_allowed(false, false, false, true) && sample_allowed(false, false, false, false));   // before the start: as ever, whatever the switch
  VCHECK(sample_allowed(true, true, false, true));                                                   // the driver is up: sampled (HeCBench aidw's second kernel)
  VCHECK(!sample_allowed(true, true, false, false));                                                 // VGPU_HOST_GPU_SAMPLE_STARTED=0: the rule before
  VCHECK(!sample_allowed(true, false, false, true) && !sample_allowed(true, false, false, false));   // a start that has begun and is not ready: never
  for (bool begun : {false, true})
    for (bool ready : {false, true})
      for (bool rule : {false, true}) VCHECK(!sample_allowed(begun, ready, true, rule));             // a driver that could not be started: the mode is off
}

VTEST(with_the_driver_up_a_launch_that_may_be_sampled_has_no_start_to_repay) {
  const Costs c;
  const uint64_t live = 2867200;   // aidw: 2.73 MiB
  const double up = sample_card_path_seconds(live, c, true), cold = sample_card_path_seconds(live, c, false);
  VCHECK(std::fabs(up - (7e-6 + 100e-6 + 2867200.0 / 12e9 + 2867200.0 / 10e9)) < 1e-12);
  VCHECK(std::fabs(cold - up - 0.25) < 1e-12);
  Costs free_start = c;
  free_start.start_cost_s = 0;
  VCHECK(sample_card_path_seconds(live, free_start, false) == sample_card_path_seconds(live, free_start, true));
  free_start.start_cost_s = -1;                                   // (a cost no one can set: never a credit)
  VCHECK(sample_card_path_seconds(live, free_start, false) == sample_card_path_seconds(live, free_start, true));
  VCHECK(std::fabs(sample_card_path_seconds(0, c, true) - 107e-6) < 1e-15);
  VCHECK(sample_card_path_seconds(1ull << 30, c, true) > 0.19 && sample_card_path_seconds(1ull << 30, c, true) < 0.20);   // the memory each way is still charged
  // aidw's AIDW_Kernel_Tiled on the A10G rig: 400 blocks of 256 threads, 2,624 instructions (84 ms counted), the process 44 s old, the driver up
  SampleCandidate w;
  w.blocks = 400;
  w.loop = true;
  w.bounded = true;
  w.live_bytes = live;
  w.static_s = static_sim_seconds(400, 256, 2624, c);
  w.age_s = 44.0;
  w.driver_up = true;
  VCHECK(w.static_s > 0.083 && w.static_s < 0.085);
  VCHECK(worth_sampling(w, c, up));
  // its first block, stopped at 15 ms and timed at 12.7 ms on 4 host threads, says 1.27 s at the least: sent, with or without a start to repay
  Sample s;
  s.seconds = 0.012685;
  s.timed = 1;
  s.total = 400;
  s.threads = 4;
  VCHECK(sample_offload(s, live, live, c, true) && sample_offload(s, live, live, c, false));
  // what the start was: a launch of 0.4 s repays the card path with the driver up and not the start
  VCHECK(sampled_offload(0.4, live, live, c, true) && !sampled_offload(0.4, live, live, c, false));
  // ... and a launch must still repay the card path: the memory each way (0.53 ms here), the launch, the kernel
  VCHECK(!sampled_offload(0.0012, live, live, c, true) && sampled_offload(0.002, live, live, c, true));
  VCHECK(!sampled_offload(0.00019, 0, 0, c, true));               // (the floor of the size rule: 200 us)
  // the least a sampled launch must take: 1.4 ms with the driver up against 0.67 s before the start
  const double need_up = sample_need_seconds(live, live, c, true), need_cold = sample_need_seconds(live, live, c, false);
  VCHECK(need_up > 0.0014 && need_up < 0.0015 && need_cold > 0.66 && need_cold < 0.68);
  VCHECK(!sampled_offload(need_up * 0.99, live, live, c, true) && sampled_offload(need_up * 1.01, live, live, c, true));
}

VTEST(with_the_driver_up_a_kernel_is_sampled_again_only_if_its_whole_launch_reached_the_floor) {
  const Costs c;
  SampleCandidate w;
  w.blocks = 4096;
  w.loop = true;
  w.bounded = true;
  w.static_s = 1.0;
  w.age_s = 10.0;
  const double up = sample_card_path_seconds(0, c, true);         // 107 us
  // not marked: the bar is the margin over the card path, as before the start (0.214 ms here)
  VCHECK(sample_short_before_seconds(w, c, up) == c.margin * up);
  w.whole_s = 0.0002;
  VCHECK(sample_refusal(w, c, up) == NoSample::ShortBefore);
  w.whole_s = 0.001;
  VCHECK(sample_refusal(w, c, up) == NoSample::None);
  // the driver up: a whole launch of a millisecond is not worth a sample of three; the floor (30 ms) is asked of it
  w.driver_up = true;
  VCHECK(sample_short_before_seconds(w, c, up) == kSampleStaticFloorSeconds);
  VCHECK(sample_refusal(w, c, up) == NoSample::ShortBefore);
  w.whole_s = kSampleStaticFloorSeconds;
  VCHECK(sample_refusal(w, c, up) == NoSample::ShortBefore);      // (at the floor is not over it)
  w.whole_s = kSampleStaticFloorSeconds * 1.01;
  VCHECK(sample_refusal(w, c, up) == NoSample::None);
  // ... and with live memory the floor is the one a sample's page faults ask (100 MiB: 0.19 s)
  w.live_bytes = 100ull << 20;
  const double up100 = sample_card_path_seconds(w.live_bytes, c, true);
  VCHECK(sample_floor(w) > 0.18 && sample_floor(w) < 0.19 && c.margin * up100 < 0.04);
  w.whole_s = 0.1;
  VCHECK(sample_refusal(w, c, up100) == NoSample::ShortBefore);
  w.whole_s = 0.2;
  VCHECK(sample_refusal(w, c, up100) == NoSample::None);
  // ... except where the card path asks more than the floor (1 GiB each way: 0.39 s with the margin, against a floor of 30 ms for a small grid)
  SampleCandidate g;
  g.blocks = 80;
  g.loop = true;
  g.bounded = true;
  g.driver_up = true;
  g.static_s = 1.0;
  g.age_s = 10.0;
  g.live_bytes = 1ull << 30;
  const double up1g = sample_card_path_seconds(g.live_bytes, c, true);
  VCHECK(sample_floor(g) == kSampleStaticFloorSeconds && sample_short_before_seconds(g, c, up1g) == c.margin * up1g);
  g.whole_s = 0.3;
  VCHECK(sample_refusal(g, c, up1g) == NoSample::ShortBefore);
  g.whole_s = 0.5;
  VCHECK(sample_refusal(g, c, up1g) == NoSample::None);
  // before the start nothing changes: the bar is the card path with the start (0.52 s with the margin)
  SampleCandidate b;
  b.blocks = 4096;
  b.loop = true;
  b.static_s = 1.0;
  b.age_s = 10.0;
  b.live_bytes = 100ull << 20;
  const double cold = sample_card_path_seconds(0, c, false);
  b.whole_s = 0.4;
  VCHECK(sample_refusal(b, c, cold) == NoSample::ShortBefore);
  b.whole_s = 0.6;
  VCHECK(sample_refusal(b, c, cold) == NoSample::None);
  // the other reasons stand with the driver up: no loop, a grid of one block, the floor in a young process
  SampleCandidate n;
  n.blocks = 4096;
  n.driver_up = true;
  n.static_s = 1.0;
  n.age_s = 10.0;
  VCHECK(sample_refusal(n, c, up) == NoSample::NoLoop);
  n.loop = true;
  n.blocks = 1;
  VCHECK(sample_refusal(n, c, up) == NoSample::SmallGrid);
  n.blocks = 4096;
  n.static_s = 0.001;
  n.age_s = 0.01;
  VCHECK(sample_refusal(n, c, up) == NoSample::Floor);
  n.age_s = 0.16;                                                 // 2 % of 0.16 s is the 3 ms a sample is taken to cost
  VCHECK(sample_refusal(n, c, up) == NoSample::None);
}

VTEST(with_the_driver_up_a_whole_launch_over_the_floor_stands_in_for_the_static_count) {
  const Costs c;
  // the CUDA Samples' histogram256Kernel on server3: 240 blocks (a small grid sampled by bounded blocks), 2.8 ms counted, 64 MiB live, the
  // process 4.06 s old; its first launch after the driver start took 0.188 s in the simulator
  SampleCandidate w;
  w.blocks = 240;
  w.loop = true;
  w.bounded = true;
  w.live_bytes = 67355648;
  w.static_s = 0.002765;
  w.age_s = 4.056599;
  w.expected_s = kSampleCostGuessSeconds + kSampleSecondsPerLiveByte * 67355648.0;
  w.driver_up = true;
  const double up = sample_card_path_seconds(w.live_bytes, c, true);
  VCHECK(sample_floor(w) == kSampleStaticFloorSeconds && !sample_affordable(w.age_s, 0, sample_expected_seconds(w)));   // two runs at the cap: 0.16 s against 2 % of 4 s
  VCHECK(sample_floor_estimate(w) == w.static_s);                 // its first launch: nothing measured, the count is asked, and refuses
  VCHECK(sample_refusal(w, c, up) == NoSample::Floor);
  w.whole_s = 0.187685;                                           // its second: the launch the simulator ran is asked instead
  VCHECK(sample_floor_estimate(w) == 0.187685);
  VCHECK(sample_refusal(w, c, up) == NoSample::None);
  w.whole_s = 0.031;                                              // just over the floor (and over twice the card path, 25 ms): sampled
  VCHECK(sample_short_before_seconds(w, c, up) == kSampleStaticFloorSeconds && sample_refusal(w, c, up) == NoSample::None);
  w.whole_s = 0.029;                                              // under it: short, as before
  VCHECK(sample_refusal(w, c, up) == NoSample::ShortBefore);
  // a whole launch under the count does not lower what the floor is asked of
  w.static_s = 0.5;
  w.whole_s = 0.04;
  VCHECK(sample_floor_estimate(w) == 0.5);
  // before the start the count alone is asked, whatever a whole launch took (0.2 s there is under the bar of the start anyway)
  SampleCandidate b = w;
  b.driver_up = false;
  b.static_s = 0.002765;
  b.whole_s = 0.6;
  const double cold = sample_card_path_seconds(b.live_bytes, c, false);
  VCHECK(sample_floor_estimate(b) == b.static_s);
  VCHECK(c.margin * cold < 0.6 && sample_refusal(b, c, cold) == NoSample::Floor);
  // a grid sampled by whole blocks, 100 MiB live (the floor 0.19 s): a whole launch of 0.2 s is over it, one of 0.18 s is short
  SampleCandidate g;
  g.blocks = 4096;
  g.loop = true;
  g.bounded = true;
  g.driver_up = true;
  g.live_bytes = 100ull << 20;
  g.static_s = 0.01;
  g.age_s = 1.0;
  g.expected_s = kSampleCostGuessSeconds + kSampleSecondsPerLiveByte * static_cast<double>(g.live_bytes);
  const double up100 = sample_card_path_seconds(g.live_bytes, c, true);
  VCHECK(!sample_affordable(g.age_s, 0, sample_expected_seconds(g)) && sample_refusal(g, c, up100) == NoSample::Floor);
  g.whole_s = 0.2;
  VCHECK(sample_refusal(g, c, up100) == NoSample::None);
  g.whole_s = 0.18;
  VCHECK(sample_refusal(g, c, up100) == NoSample::ShortBefore);
}

// ---- turbo fit: the card's own image, whether a launch fits, the PTX scan (include/vgpu/host_gpu_source.hpp) ----

namespace {
CardCandidate cubin_of(int arch, char suffix = 0) {
  CardCandidate c;
  c.kind = CardCandidate::Cubin;
  c.arch = arch;
  c.suffix = suffix;
  c.bytes = "cubin" + std::to_string(arch) + suffix;
  return c;
}
CardCandidate ptx_of(int arch, char suffix = 0) {
  CardCandidate c;
  c.kind = CardCandidate::Ptx;
  c.arch = arch;
  c.suffix = suffix;
  c.bytes = "ptx" + std::to_string(arch) + suffix;
  return c;
}
}  // namespace

VTEST(card_image_is_the_best_one_the_card_runs) {
  // An sm_86 card (an RTX 3060) given the fatbin of a program built for a simulated H100 (sm_90): the cubin of its own architecture, else the same
  // major with a lower minor, else PTX (the driver's JIT), else nothing: the kernels stay in the simulator.
  VCHECK_EQ(pick_card_image({cubin_of(86), cubin_of(90)}, 86), 0);
  VCHECK_EQ(pick_card_image({cubin_of(90), cubin_of(86)}, 86), 1);   // (the image the simulated profile runs is not the card's)
  VCHECK_EQ(pick_card_image({cubin_of(80), cubin_of(90)}, 86), 0);   // lower minor of the same major
  VCHECK_EQ(pick_card_image({cubin_of(80), cubin_of(86), cubin_of(90)}, 86), 1);   // exact over lower
  VCHECK_EQ(pick_card_image({cubin_of(75)}, 86), -1);     // another major
  VCHECK_EQ(pick_card_image({cubin_of(90)}, 86), -1);     // newer
  VCHECK_EQ(pick_card_image({cubin_of(89)}, 86), -1);     // a newer minor
  VCHECK_EQ(pick_card_image({cubin_of(90, 'a')}, 86), -1);
  VCHECK_EQ(pick_card_image({cubin_of(90, 'a')}, 90), 0);   // an `a` image runs on exactly its architecture ...
  VCHECK_EQ(pick_card_image({cubin_of(80, 'a')}, 86), -1);  // ... and not on a newer minor
  VCHECK_EQ(pick_card_image({cubin_of(90), cubin_of(90, 'a')}, 90), 1);   // of two for one architecture, the `a` one
  VCHECK_EQ(pick_card_image({cubin_of(90, 'a'), cubin_of(90)}, 90), 0);
  // PTX: only when no cubin runs; the newest target the card can run
  VCHECK_EQ(pick_card_image({cubin_of(90), ptx_of(80), ptx_of(75)}, 86), 1);
  VCHECK_EQ(pick_card_image({cubin_of(90), ptx_of(90)}, 86), -1);   // PTX for a newer architecture does not run: a fatbin built only for sm_90
  VCHECK_EQ(pick_card_image({ptx_of(75), ptx_of(86), ptx_of(80)}, 86), 1);
  VCHECK_EQ(pick_card_image({cubin_of(86), ptx_of(86)}, 86), 0);   // a cubin beats PTX of the same architecture
  VCHECK_EQ(pick_card_image({cubin_of(75), ptx_of(75)}, 86), 1);   // the sm_75-only build: the card JITs its PTX
  VCHECK_EQ(pick_card_image({ptx_of(90, 'a')}, 90), 0);
  VCHECK_EQ(pick_card_image({ptx_of(90, 'a')}, 100), -1);   // `a`: exactly
  VCHECK_EQ(pick_card_image({ptx_of(100, 'f')}, 103), 0);   // `f`: within the major
  VCHECK_EQ(pick_card_image({ptx_of(100, 'f')}, 90), -1);
  VCHECK_EQ(pick_card_image({ptx_of(80), ptx_of(80, 'a')}, 80), 1);
  VCHECK_EQ(pick_card_image({}, 86), -1);
  // the two rules are the simulator's own
  for (int cubin = 50; cubin <= 130; ++cubin)
    for (int host = 50; host <= 130; ++host)
      for (bool a : {false, true}) VCHECK_EQ(cubin_runs_on(cubin, a, host), vgpu::sass::runs_on(cubin, a, host));
  VCHECK(describe_candidates({cubin_of(75), cubin_of(90, 'a'), ptx_of(90)}) == "SASS sm_75, sm_90a, PTX sm_90");
  VCHECK(describe_candidates({ptx_of(75)}) == "PTX sm_75" && describe_candidates({}) == "no image");
}

VTEST(ptx_target_is_read_from_its_line) {
  int arch = -1;
  char suffix = 'x';
  VCHECK(ptx_target_of(".version 8.5\n.target sm_90a\n.address_size 64\n", &arch, &suffix) && arch == 90 && suffix == 'a');
  VCHECK(ptx_target_of(".version 8.5\n.target sm_86, debug\n", &arch, &suffix) && arch == 86 && suffix == 0);
  VCHECK(ptx_target_of("// c\n.target sm_100f\n", &arch, &suffix) && arch == 100 && suffix == 'f');
  VCHECK(!ptx_target_of(".version 8.5\n", &arch, &suffix) && arch == 0 && suffix == 0);
  VCHECK(!ptx_target_of(".target texmode_independent\nsm_90\n", &arch, &suffix));   // (the architecture on a later line is not the target's)
  VCHECK(ptx_target_runs_on(75, 0, 86) && ptx_target_runs_on(86, 0, 86) && !ptx_target_runs_on(90, 0, 86));
  VCHECK(ptx_target_runs_on(90, 'a', 90) && !ptx_target_runs_on(90, 'a', 100) && !ptx_target_runs_on(100, 'a', 90));
  VCHECK(ptx_target_runs_on(100, 'f', 103) && !ptx_target_runs_on(100, 'f', 120) && !ptx_target_runs_on(103, 'f', 100));
}

VTEST(a_launch_fits_the_card_by_its_limits) {
  // An RTX 3060 (sm_86): 1024 threads a block, dims 1024 x 1024 x 64, grid 2^31-1 x 65535 x 65535, 99 KiB shared opt-in, 65536 registers a block.
  CardLimits c;
  c.max_threads_per_block = 1024;
  c.max_block[0] = 1024; c.max_block[1] = 1024; c.max_block[2] = 64;
  c.max_grid[0] = 2147483647ull; c.max_grid[1] = 65535; c.max_grid[2] = 65535;
  c.shared_optin = 101376;
  c.regs_per_block = 65536;
  c.memory_total = 12ull << 30;
  FuncAttrs f;
  f.known = true;
  f.max_threads = 1024;
  f.static_shared = 0;
  f.regs = 32;
  const auto fits = [&](std::initializer_list<uint32_t> grid, std::initializer_list<uint32_t> block, uint64_t dyn, uint64_t window = 1ull << 20) {
    uint32_t g[3], b[3];
    std::copy(grid.begin(), grid.end(), g);
    std::copy(block.begin(), block.end(), b);
    return fits_card(c, f, g, b, dyn, window);
  };
  VCHECK(c.known() && f.known);
  VCHECK(fits({1, 1, 1}, {256, 1, 1}, 0) == FitRefusal::None);
  VCHECK(fits({1, 1, 1}, {1024, 1, 1}, 0) == FitRefusal::None);   // 1024 x 32 registers = 32768 <= 65536
  VCHECK(fits({1, 1, 1}, {1025, 1, 1}, 0) == FitRefusal::BlockDim);
  VCHECK(fits({1, 1, 1}, {1, 1, 65}, 0) == FitRefusal::BlockDim);
  VCHECK(fits({1, 1, 1}, {32, 32, 2}, 0) == FitRefusal::Threads);   // 2048 threads, each dimension under its own limit
  VCHECK(fits({1, 1, 1}, {32, 32, 1}, 0) == FitRefusal::None);
  f.max_threads = 512;   // a kernel built with __launch_bounds__(512), or whose registers allow no more
  VCHECK(fits({1, 1, 1}, {512, 1, 1}, 0) == FitRefusal::None);
  VCHECK(fits({1, 1, 1}, {513, 1, 1}, 0) == FitRefusal::KernelThreads);
  f.max_threads = 1024;
  f.regs = 64;
  VCHECK(fits({1, 1, 1}, {1024, 1, 1}, 0) == FitRefusal::None);   // 65536: exactly the file
  f.regs = 65;
  VCHECK(fits({1, 1, 1}, {1024, 1, 1}, 0) == FitRefusal::Registers);
  VCHECK(fits({1, 1, 1}, {1008, 1, 1}, 0) == FitRefusal::None);   // 1008 x 65 = 65520
  f.regs = 32;
  f.static_shared = 48 * 1024;
  VCHECK(fits({1, 1, 1}, {64, 1, 1}, 101376 - 48 * 1024) == FitRefusal::None);      // static + dynamic = the opt-in limit exactly
  VCHECK(fits({1, 1, 1}, {64, 1, 1}, 101376 - 48 * 1024 + 1) == FitRefusal::Shared);   // one byte more
  f.static_shared = 0;
  VCHECK(fits({1, 1, 1}, {64, 1, 1}, 227 * 1024) == FitRefusal::Shared);   // what an H100 allows a block, and the card does not
  VCHECK(fits({2147483647u, 65535, 65535}, {64, 1, 1}, 0) == FitRefusal::None);
  VCHECK(fits({2147483648u, 1, 1}, {64, 1, 1}, 0) == FitRefusal::GridDim);
  VCHECK(fits({1, 65536, 1}, {64, 1, 1}, 0) == FitRefusal::GridDim);
  VCHECK(fits({1, 1, 65536}, {64, 1, 1}, 0) == FitRefusal::GridDim);
  VCHECK(fits({1, 1, 1}, {64, 1, 1}, 0, 12ull << 30) == FitRefusal::None);          // the window is the card's whole memory ...
  VCHECK(fits({1, 1, 1}, {64, 1, 1}, 0, (12ull << 30) + 1) == FitRefusal::Memory);   // ... one byte more
  // what is not known is not guessed
  f.known = false;
  VCHECK(fits({1, 1, 1}, {64, 1, 1}, 0) == FitRefusal::Unknown);
  f.known = true;
  c.shared_optin = 0;
  VCHECK(!c.known() && fits({1, 1, 1}, {64, 1, 1}, 0) == FitRefusal::Unknown);
  for (unsigned r = 0; r <= static_cast<unsigned>(FitRefusal::SharedAttribute); ++r) VCHECK(r ? std::string(fit_refusal_text(static_cast<FitRefusal>(r))).size() > 10u : std::string(fit_refusal_text(FitRefusal::None)).empty());
}

VTEST(the_window_needs_the_free_memory_and_a_reserve) {
  VCHECK(window_fits(1ull << 30, (1ull << 30) + kWindowReserve));
  VCHECK(!window_fits(1ull << 30, (1ull << 30) + kWindowReserve - 1));
  VCHECK(window_fits(0, kWindowReserve) && !window_fits(0, kWindowReserve - 1));
  VCHECK_EQ(kWindowReserve, 256ull << 20);
}

VTEST(the_fit_switch_and_the_notice) {
  bool fit = false;
  VCHECK(parse_fit(nullptr, &fit) && !fit);   // (unset: the default, the caller's own value stays)
  fit = true;
  VCHECK(parse_fit("", &fit) && fit);
  VCHECK(parse_fit("0", &fit) && !fit);
  VCHECK(parse_fit("1", &fit) && fit);
  for (const char* bad : {"2", "on", "off", "true", "fit", " 1", "01", "-1"}) {
    fit = true;
    VCHECK(!parse_fit(bad, &fit));
  }
  const std::string n = turbo_notice();
  VCHECK(n.find("turbo") != std::string::npos && n.find("may differ from the simulator") != std::string::npos && n.find("exact") != std::string::npos);
}

VTEST(ptx_the_card_may_be_given) {
  const std::string head = ".version 8.5\n.target sm_75\n.address_size 64\n";
  const std::string plain = head + ".visible .entry k(.param .u64 k_param_0)\n{\n.reg .b64 %rd<3>;\n.reg .b32 %r<3>;\nld.param.u64 %rd1, [k_param_0];\nld.global.u32 %r1, [%rd1];\nst.global.u32 [%rd1], %r1;\nret;\n}\n";
  PtxScan r = scan_ptx_for_card(plain, false);
  VCHECK(r.refuse.empty() && !r.never && !r.printf);   // (ld.global and st.global are instructions, not declarations)
  // a module variable: the card's module would have a second copy
  for (const char* decl : {".global .align 4 .u32 g;", ".visible .global .align 4 .b8 g[16];", ".const .align 4 .u32 c[4];", ".extern .global .u32 g;", ".weak .const .u32 c;"}) {
    r = scan_ptx_for_card(head + decl + "\n" + plain.substr(head.size()), false, false, /*unreferenced_ok=*/false);   // (the rule before section 20: any variable)
    VCHECK(!r.refuse.empty() && !r.never);
  }
  VCHECK(scan_ptx_for_card(head + ".shared .align 4 .b8 sh[64];\n.local .align 4 .b8 lc[16];\n" + plain.substr(head.size()), false).refuse.empty());
  // textures and surfaces: class never
  for (const char* t : {".global .texref t;", ".global .surfref s;", ".tex .u32 t;"}) {
    r = scan_ptx_for_card(head + t + "\n", false);
    VCHECK(!r.refuse.empty() && r.never);
  }
  for (const char* ins : {"tex.1d.v4.f32.f32 {%f1,%f2,%f3,%f4}, [t, {%f0}];", "tld4.r.2d.v4.f32.f32 {%f1,%f2,%f3,%f4}, [t, {%f0,%f0}];", "txq.width.b32 %r1, [t];",
                          "suld.b.1d.b32.trap %r1, [s, {%r0}];", "sust.b.1d.b32.trap [s, {%r0}], {%r1};", "sured.b.add.1d.u32.trap [s, {%r0}], %r1;", "suq.width.b32 %r1, [s];",
                          "trap;", "brkpt;"}) {
    r = scan_ptx_for_card(head + ".visible .entry k()\n{\n" + ins + "\nret;\n}\n", false);
    VCHECK(!r.refuse.empty() && r.never);
  }
  // calls: vprintf is admitted (turbo mode; not when switched off), the rest of the external functions are not, nor is a call through a pointer
  const std::string pf = head + ".extern .func  (.param .b32 func_retval0) vprintf\n(\n.param .b64 vprintf_param_0,\n.param .b64 vprintf_param_1\n)\n;\n.visible .entry k()\n{\n"
                               "{\n.param .b32 retval0;\ncall.uni (retval0), vprintf, (a, b);\n}\nret;\n}\n";
  r = scan_ptx_for_card(pf, false);
  VCHECK(r.refuse.empty() && r.printf);
  r = scan_ptx_for_card(pf, true);
  VCHECK(!r.refuse.empty() && !r.printf && r.refuse.find("printf") != std::string::npos);
  const std::string malloc_ = head + ".extern .func  (.param .b64 func_retval0) malloc\n(\n.param .b64 malloc_param_0\n)\n;\n.visible .entry k()\n{\ncall.uni (retval0), malloc, (n);\nret;\n}\n";
  r = scan_ptx_for_card(malloc_, false);
  VCHECK(!r.refuse.empty() && !r.never && r.refuse.find("malloc") != std::string::npos);
  r = scan_ptx_for_card(head + ".visible .entry k()\n{\ncall (retval0), %rd5, (p), proto;\nret;\n}\n", false);
  VCHECK(!r.refuse.empty() && r.never);
  // a call to a function of the module itself is the module's own, and comments hold nothing
  r = scan_ptx_for_card(head + ".func f()\n{\nret;\n}\n.visible .entry k()\n{\ncall.uni f, ();\n// tex.1d .global .u32 g; trap;\n/* suld.b trap; */ ret;\n}\n", false);
  VCHECK(r.refuse.empty());
}

VTEST(wsl_is_read_from_the_kernel_version) {
  VCHECK(host_is_wsl("Linux version 6.18.58.1-microsoft-standard-WSL2 (root@x) (gcc)", false));
  VCHECK(host_is_wsl("Linux version 5.15.0-Microsoft", false));
  VCHECK(host_is_wsl("Linux version 6.8.0-1021-aws (buildd@lcy02) (gcc 13.3.0)", true));   // /dev/dxg
  VCHECK(!host_is_wsl("Linux version 6.8.0-1021-aws (buildd@lcy02) (gcc 13.3.0)", false));
  VCHECK(!host_is_wsl("", false));
}

VTEST(ptx_classes_follow_the_switches) {
  const std::string head = ".version 8.5\n.target sm_75\n.address_size 64\n";
  const auto scan = [&](const char* body) { return scan_ptx_for_card(head + ".visible .entry k()\n{\n" + body + "\nret;\n}\n", false); };
  using C = InsnClass;
  VCHECK(scan("atom.global.add.u32 %r1, [%rd1], 1;").classes == class_bit(C::AtomInt));
  VCHECK(scan("atom.global.cas.b32 %r1, [%rd1], 1, 2;").classes == class_bit(C::AtomSwap));
  VCHECK(scan("atom.global.exch.b32 %r1, [%rd1], 2;").classes == class_bit(C::AtomSwap));
  VCHECK(scan("atom.global.add.f32 %f1, [%rd1], %f2;").classes == class_bit(C::AtomFloat));
  VCHECK(scan("red.global.add.f32 [%rd1], %f2;").classes == class_bit(C::FloatRed));
  VCHECK(scan("ex2.approx.ftz.f32 %f1, %f2;").classes == class_bit(C::Mufu));
  VCHECK(scan("fma.rn.f16x2 %r1, %r2, %r3, %r4;").classes == class_bit(C::Half));
  VCHECK(scan("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4};").classes == class_bit(C::Tensor));
  VCHECK(scan("add.s32 %r1, %r2, %r3;").classes == 0);
  // the turbo switches decide on those classes: ATOMICS=0 keeps swaps and float atomics out, TURBO_OFF=atomics all atomics, =mufu the approximations
  TurboSwitches no;
  no.atomics = false;
  VCHECK(turbo_refusal(scan("atom.global.cas.b32 %r1, [%rd1], 1, 2;").classes, no) == TurboRefusal::Atomics);
  VCHECK(turbo_refusal(scan("atom.global.add.f32 %f1, [%rd1], %f2;").classes, no) == TurboRefusal::Atomics);
  VCHECK(turbo_refusal(scan("atom.global.add.u32 %r1, [%rd1], 1;").classes, no) == TurboRefusal::None);
  TurboSwitches off;
  VCHECK(parse_turbo_off("atomics", &off.off));
  VCHECK(turbo_refusal(scan("atom.global.add.u32 %r1, [%rd1], 1;").classes, off) == TurboRefusal::Off);
  VCHECK(parse_turbo_off("mufu", &off.off));
  VCHECK(turbo_refusal(scan("ex2.approx.ftz.f32 %f1, %f2;").classes, off) == TurboRefusal::Off);
  // sm_90 and later forms (the tensor-map descriptor is the simulator's own): never
  for (const char* ins : {"cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%r1], [%rd1, {%r2, %r3}], [%r4];", "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%f1}, %rd1, %rd2, 1, 1, 1, 0, 0;",
                          "setmaxnreg.inc.sync.aligned.u32 240;", "elect.sync %r1|%p1, 0xffffffff;", "barrier.cluster.arrive;", "mapa.shared::cluster.u64 %rd1, %rd2, %r1;",
                          "prefetch.tensormap [%rd1];", "mov.u32 %r1, %clusterid.x;", "stmatrix.sync.aligned.m8n8.x1.shared.b16 [%r1], {%r2};"}) {
    const PtxScan r = scan(ins);
    VCHECK(!r.refuse.empty() && r.never);
  }
}

namespace {
vgpu::sass::Cubin cubin_with(std::initializer_list<std::tuple<const char*, const char*, uint64_t, uint64_t>> vars, uint64_t global_size = 64, uint64_t const3_size = 32) {
  vgpu::sass::Cubin c;
  for (const char* name : {".nv.global", ".nv.constant3"}) {
    vgpu::sass::CubinSection s;
    s.name = name;
    s.size = std::string(name) == ".nv.global" ? global_size : const3_size;
    c.section_index[name] = c.sections.size();
    c.sections.push_back(s);
  }
  for (const auto& [name, section, value, size] : vars) {
    vgpu::sass::CubinSymbol y;
    y.name = name;
    y.section = section;
    y.value = value;
    y.size = size;
    c.symbols.push_back(y);
  }
  return c;
}
}  // namespace

VTEST(the_card_image_must_lay_variables_out_as_the_simulator_s_does) {
  const vgpu::sass::Cubin sim = cubin_with({{"a", ".nv.global", 0, 16}, {"b", ".nv.global", 16, 8}, {"c", ".nv.constant3", 0, 4}});
  VCHECK(variable_layout_mismatch(sim, sim).empty());
  VCHECK(variable_layout_mismatch(cubin_with({{"c", ".nv.constant3", 0, 4}, {"b", ".nv.global", 16, 8}, {"a", ".nv.global", 0, 16}}), sim).empty());   // (the order is not compared)
  VCHECK(!variable_layout_mismatch(cubin_with({{"a", ".nv.global", 0, 16}, {"b", ".nv.global", 24, 8}, {"c", ".nv.constant3", 0, 4}}), sim).empty());   // an offset
  VCHECK(!variable_layout_mismatch(cubin_with({{"a", ".nv.global", 0, 16}, {"b", ".nv.global", 16, 12}, {"c", ".nv.constant3", 0, 4}}), sim).empty());  // a size
  VCHECK(!variable_layout_mismatch(cubin_with({{"a", ".nv.global", 0, 16}, {"b", ".nv.constant3", 16, 8}, {"c", ".nv.constant3", 0, 4}}), sim).empty());  // a section
  VCHECK(!variable_layout_mismatch(cubin_with({{"a", ".nv.global", 0, 16}, {"b", ".nv.global", 16, 8}, {"c", ".nv.constant3", 0, 4}}, 128), sim).empty());  // a section's size
  VCHECK(!variable_layout_mismatch(cubin_with({{"a", ".nv.global", 0, 16}, {"c", ".nv.constant3", 0, 4}}), sim).empty());   // a variable the card's image lacks
  VCHECK(!variable_layout_mismatch(cubin_with({{"a", ".nv.global", 0, 16}, {"b", ".nv.global", 16, 8}, {"c", ".nv.constant3", 0, 4}, {"d", ".nv.global", 24, 4}}), sim).empty());   // one the simulator lacks
  // not variables: functions, names that start with '.', symbols of other sections (static shared memory) are not compared
  vgpu::sass::Cubin card = sim;
  vgpu::sass::CubinSymbol sh;
  sh.name = "tile";
  sh.section = ".nv.shared.k";
  sh.size = 4096;
  card.symbols.push_back(sh);
  VCHECK(variable_layout_mismatch(card, sim).empty());
  vgpu::sass::CubinSymbol fn;
  fn.name = "f";
  fn.function = true;
  fn.section = ".text.f";
  card.symbols.push_back(fn);
  VCHECK(variable_layout_mismatch(card, sim).empty());
}

VTEST(the_card_image_must_take_the_simulator_s_arguments) {
  vgpu::sass::CubinKernel k;
  k.params = {{0, 0, 8}, {1, 8, 4}};
  VCHECK(parameter_sizes_agree(k, {8, 4}));
  VCHECK(!parameter_sizes_agree(k, {8, 8}));   // a size
  VCHECK(!parameter_sizes_agree(k, {8}));      // a count
  VCHECK(!parameter_sizes_agree(k, {8, 4, 4}));
  VCHECK(parameter_sizes_agree(vgpu::sass::CubinKernel{}, {}));
}

VTEST(plain_sm_90_ptx_is_retargeted_for_an_older_card) {
  const auto ptx = [](int arch, char suffix) {
    CardCandidate c;
    c.kind = CardCandidate::Ptx;
    c.arch = arch;
    c.suffix = suffix;
    c.bytes = ".version 9.0\n.target sm_" + std::to_string(arch) + (suffix ? std::string(1, suffix) : std::string()) + "\n.address_size 64\n";
    return c;
  };
  for (int host : {80, 86, 87}) VCHECK_EQ(pick_retarget_image({ptx(90, 0)}, host), 0);
  for (int host : {70, 75, 89, 90, 100, 120}) VCHECK_EQ(pick_retarget_image({ptx(90, 0)}, host), -1);   // only the sm_80 family has been measured
  VCHECK_EQ(pick_retarget_image({ptx(90, 'a')}, 86), -1);       // sm_90a PTX has wgmma and setmaxnreg: no plain sibling, no retarget
  VCHECK_EQ(pick_retarget_image({ptx(100, 0)}, 86), -1);        // another architecture's PTX
  VCHECK_EQ(pick_retarget_image({ptx(90, 'a'), ptx(90, 0)}, 86), 1);   // the plain sm_90 PTX of an sm_90a build
  VCHECK_EQ(pick_retarget_image({}, 86), -1);
  VCHECK(pick_card_image({ptx(90, 0)}, 86) < 0);   // (it is the fallback: the ordinary choice finds nothing)
  // the rewrite: the target line only, the version untouched
  VCHECK(retarget_ptx_text(".version 9.0\n.target sm_90\n.address_size 64\n", 86) == ".version 9.0\n.target sm_86\n.address_size 64\n");
  VCHECK(retarget_ptx_text(".version 9.0\n.target sm_90, debug\n", 80) == ".version 9.0\n.target sm_80, debug\n");
  VCHECK(retarget_ptx_text(".target sm_90\n.visible .entry k(){ // sm_90 here\n}", 87) == ".target sm_87\n.visible .entry k(){ // sm_90 here\n}");
  VCHECK(retarget_ptx_text(".target sm_90a\n", 86).empty() && retarget_ptx_text(".target sm_900\n", 86).empty() && retarget_ptx_text(".target sm_86\n", 80).empty() && retarget_ptx_text("", 86).empty());
}

// The forms ptxas 13.0 rejects for sm_86 in a module retargeted from sm_90 ("requires .target sm_90 or higher", micro-tests of the experiment), and the ones it takes.
VTEST(retargeted_ptx_is_refused_for_the_forms_the_older_card_lacks) {
  const std::string head = ".version 9.0\n.target sm_90\n.address_size 64\n";
  const auto scan = [&](const char* body, bool retarget = true) { return scan_ptx_for_card(head + ".visible .entry k()\n{\n" + body + "\nret;\n}\n", false, retarget); };
  for (const char* ins : {"mbarrier.try_wait.shared::cta.b64 %p1, [%rd1], %rd2;", "mbarrier.try_wait.parity.shared::cta.b64 %p1, [%rd1], %r2;", "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%rd1], %r2;",
                          "mbarrier.arrive.shared::cta.b64 %rd2, [%rd1], %r3;", "atom.global.v2.f32.add {%f1, %f2}, [%rd1], {%f3, %f4};", "red.global.add.noftz.v4.f16x2 [%rd1], {%r1, %r2, %r3, %r4};",
                          "atom.global.exch.b128 %rq1, [%rd1], %rq2;", "fence.proxy.async.shared::cta;", "mov.u32 %r1, %aggr_smem_size;", "max.s32.relu %r1, %r2, %r3;",
                          "add.s16x2 %r1, %r2, %r3;", "min.u16x2 %r1, %r2, %r3;", "cvt.rn.satfinite.e4m3x2.f32 %rs1, %f1, %f2;", "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4};"}) {
    const PtxScan r = scan(ins);
    VCHECK(!r.refuse.empty() && r.never);
  }
  for (const char* d : {".reqnctapercluster 2, 1, 1;", ".maxclusterrank 4;", ".explicitcluster;"}) {
    const PtxScan r = scan_ptx_for_card(head + ".visible .entry k()\n" + d + "\n{\nret;\n}\n", false, true);
    VCHECK(!r.refuse.empty() && r.never);
  }
  // what assembles for sm_86 in the experiment (ptxas 13.0): basic mbarrier, ldmatrix, fences, mma.sync, bf16 and cp.async
  for (const char* ins : {"mbarrier.init.shared::cta.b64 [%rd1], 1;", "mbarrier.arrive.shared::cta.b64 %rd2, [%rd1];", "mbarrier.test_wait.shared::cta.b64 %p1, [%rd1], %rd2;", "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%r1}, [%r2];",
                          "fence.sc.gpu;", "fence.acq_rel.gpu;", "membar.gl;", "cp.async.ca.shared.global [%r1], [%rd1], 16;", "cp.async.commit_group;", "cvt.rn.bf16x2.f32 %r1, %f1, %f2;",
                          "redux.sync.add.s32 %r1, %r2, 0xffffffff;", "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4};",
                          "shfl.sync.bfly.b32 %r1, %r2, 1, 31, 0xffffffff;", "nanosleep.u32 100;", "add.f16x2 %r1, %r2, %r3;"})
    VCHECK(scan(ins).refuse.empty());
  VCHECK(scan("mbarrier.try_wait.shared::cta.b64 %p1, [%rd1], %rd2;", false).refuse.empty());   // (without the flag only the ordinary rules apply)
  // a variable still keeps a retargeted module out (rule 1: the card's JIT module has its own copy)
  const PtxScan v = scan_ptx_for_card(head + ".global .align 4 .u32 g;\n", false, true, false);
  VCHECK(!v.refuse.empty() && !v.never);
}

// The scan's shortcuts (pantheonsim #662) must give what the plain code gave.
VTEST(exact_mnemonic_lookup_is_the_name_list) {
  static const char* const kNames[] = {"IMAD", "IADD3", "IADD", "LOP3", "LOP", "LEA", "SHF", "SHL", "SHR", "ISETP", "IMNMX", "IABS", "POPC", "FLO", "BREV", "PRMT",
      "SEL", "MOV", "R2P", "P2R", "PLOP3", "UPLOP3", "PSETP", "ISET", "FSET", "DSET", "I2I", "I2IP", "SGXT", "BMSK", "VIADD", "VIMNMX", "FADD", "FMUL", "FFMA",
      "FSETP", "FSEL", "FMNMX", "DADD", "DMUL", "DFMA", "DSETP", "DMNMX", "I2F", "I2FP", "F2I", "F2F", "F2FP", "FRND", "FCHK", "LDG", "STG", "LDS", "STS", "LDL",
      "STL", "LD", "ST", "LDC", "ULDC", "S2R", "S2UR", "CS2R", "UMOV", "UIADD3", "UIMAD", "ULOP3", "USHF", "ULEA", "UISETP", "USEL", "UPRMT", "UFLO", "UPOPC",
      "UP2UR", "R2UR", "VOTEU", "BRA", "BRX", "BSSY", "BSYNC", "BREAK", "BMOV", "EXIT", "NOP", "YIELD", "WARPSYNC", "BAR", "MEMBAR", "ERRBAR", "DEPBAR", "CCTL",
      "SHFL", "VOTE", "MATCH", "RED"};
  std::set<std::string> names(std::begin(kNames), std::end(kNames));
  for (const std::string& n : names) {
    VCHECK(exact_mnemonic_allowed(n));
    // what is not a name: a prefix, a longer name, another case, the name with a letter more or fewer
    VCHECK(names.count(n + "X") || !exact_mnemonic_allowed(n + "X"));
    VCHECK(names.count(n.substr(0, n.size() - 1)) || !exact_mnemonic_allowed(n.substr(0, n.size() - 1)));
    VCHECK(!exact_mnemonic_allowed(n + "ABCDEFGHI"));
    std::string low = n;
    for (char& c : low) c = static_cast<char>(c | 0x20);
    VCHECK(!exact_mnemonic_allowed(low));
  }
  VCHECK(!exact_mnemonic_allowed(""));
  VCHECK(!exact_mnemonic_allowed("MUFU") && !exact_mnemonic_allowed("ATOM") && !exact_mnemonic_allowed("HMMA") && !exact_mnemonic_allowed("TEX"));
}

VTEST(undecoded_modifier_by_its_parts) {
  VCHECK(!mods_undecoded({}));
  VCHECK(!mods_undecoded({"", "F32.FTZ.RN", "U32"}));
  VCHECK(mods_undecoded({"(9)"}));
  VCHECK(mods_undecoded({"U32", "F32.(3)"}));
  VCHECK(mods_undecoded({"E.(0).X"}));
  VCHECK(!mods_undecoded({"E(0)"}));   // (not the start of a part)
}

VTEST(reachable_back_edge_is_what_a_loop_needs) {
  const auto flow = [](std::initializer_list<std::pair<int64_t, bool>> l) {   // (target, falls through)
    std::vector<FlowInsn> v;
    for (const auto& [t, f] : l) {
      FlowInsn x;
      x.target = t;
      x.falls_through = f;
      v.push_back(x);
    }
    return v;
  };
  // the compilers' kernel: work, EXIT, and a branch to itself nothing reaches
  auto tail = flow({{-1, true}, {-1, true}, {-1, false}, {3, false}});
  VCHECK(!has_reachable_back_edge(tail) && !code_has_loop(tail));
  // a counted loop: body, back branch (conditional), EXIT
  auto loop = flow({{-1, true}, {-1, true}, {1, true}, {-1, false}});
  VCHECK(has_reachable_back_edge(loop) && code_has_loop(loop));
  // a branch to itself the entry reaches
  auto spin = flow({{-1, true}, {1, false}});
  VCHECK(has_reachable_back_edge(spin) && code_has_loop(spin));
  // random flows: with no reachable back edge there is no cycle, and no loop to follow
  std::mt19937 rng(7);
  for (int round = 0; round < 3000; ++round) {
    const size_t n = 1 + rng() % 10;
    std::vector<LoopCodeInsn> lc(n);
    std::vector<FlowInsn> f(n);
    for (size_t i = 0; i < n; ++i) {
      f[i].target = rng() % 3 ? static_cast<int64_t>(rng() % n) : -1;
      f[i].falls_through = rng() % 2;
      lc[i].flow = f[i];
    }
    if (!has_reachable_back_edge(f)) {
      VCHECK(!code_has_loop(f));
      VCHECK(find_probe_loops(lc).empty());
    }
    if (code_has_loop(f)) VCHECK(has_reachable_back_edge(f));
  }
}

// ---- very short launches (#660; A10G: an empty launch + sync 8 us, so the card prior follows the simulator's estimate) ----
VTEST(the_card_prior_of_an_unseen_kernel_is_a_twentieth_of_the_simulator_time_up_to_100_us) {
  const Costs c = Costs::short_launches(), before;
  VCHECK(kernel_prior_seconds(0.0004, c) == 0.0004 / 20 && kernel_prior_seconds(0.001, c) == 0.00005);
  VCHECK(kernel_prior_seconds(0.002, c) == c.kernel_s_default && kernel_prior_seconds(5.0, c) == c.kernel_s_default);
  VCHECK(kernel_prior_seconds(0.0004, before) == before.kernel_s_default);   // the rule before: always 100 us
  Estimate e = bare(0.0004);
  e.card_kernel_s = 30e-6;                                                    // a time the card measured wins over the prior
  VCHECK(card_path_seconds(e, c) == c.launch_s + 30e-6);
  const StreakLaunch s{0.0004, 1 << 20, 0, std::nullopt};
  VCHECK(streak_card_seconds(s, c) == c.launch_s + 0.00002 && streak_card_seconds(s, before) == c.launch_s + before.kernel_s_default);
}

VTEST(a_launch_of_a_hundred_microseconds_goes_to_the_card_that_runs_it_in_a_few) {
  const Costs c = Costs::short_launches(), before;
  VCHECK(c.min_sim_s == 40e-6 && c.start_repeats_max == 5000 && before.min_sim_s == 200e-6 && before.start_repeats_max == 500);
  VCHECK(!should_offload(bare(39e-6), c) && should_offload(bare(41e-6), c));   // the floor: 5 latencies
  VCHECK(!should_offload(bare(0.000111), before) && should_offload(bare(0.000111), c));   // JACOBI1D kernel1: 111 us, 20,000 launches
  VCHECK(should_offload(bare(0.000091), c));
}

VTEST(a_kernel_launched_thousands_of_times_buys_the_start_at_fifty_microseconds_a_launch) {
  const Costs c = Costs::short_launches(), before;
  const Estimate jacobi = bare(0.000111);   // saves 98 us a launch: 0.25 s over 2551 launches
  VCHECK(!should_offload_with_start(jacobi, c, false, 0.30, 2000) && should_offload_with_start(jacobi, c, false, 0.30, 2600));
  VCHECK(!should_offload_with_start(jacobi, before, false, 0.30, 20000));
  VCHECK(!should_offload_with_start(jacobi, c, false, 0.20, 5000));         // the account is still asked
  Estimate sc = bare(0.00054);                // sc_gpu: 0.4 MiB each way, 1,430 launches, 0.15 ms card path by the rule
  sc.push_bytes = sc.pull_bytes = 400 * 1024;
  VCHECK(should_offload_with_start(sc, c, false, 0.30, 1430) && !should_offload_with_start(sc, before, false, 0.30, 1430));
  VCHECK(!should_offload_with_start(bare(0.00022), c, false, 0.30, 1000));   // LargeKernelParameter: saves 0.2 ms, the start needs 0.25: stays
  VCHECK(should_offload_with_start(bare(0.00022), c, true, 0.0, 1000));      // (with the driver up it would go)
}

VTEST(a_card_time_is_kept_at_its_least_and_the_first_launch_counts_for_no_more_than_the_prior) {
  const Costs c = Costs::short_launches(), before;
  // JACOBI1D kernel1: the first card launch took 65 us with the module load, the simulator takes 70 us for it; the prior is 3.5 us
  VCHECK(card_time_to_keep(65e-6, true, std::nullopt, 70e-6, c) == 70e-6 / 20);
  VCHECK(card_time_to_keep(65e-6, true, std::nullopt, 70e-6, before) == 65e-6);                  // the rule before keeps what was measured
  VCHECK(card_time_to_keep(22e-6, false, std::nullopt, 70e-6, c) == 22e-6);                      // the first later launch replaces the prior
  VCHECK(card_time_to_keep(41e-6, false, 20e-6, 70e-6, c) == 20e-6);                            // a slow one among 20 us ones is forgotten
  VCHECK(card_time_to_keep(15e-6, false, 20e-6, 70e-6, c) == 15e-6);
  VCHECK(card_time_to_keep(41e-6, false, 20e-6, 70e-6, before) == 41e-6);                       // before: the last time
}

VTEST(a_kernel_is_banned_after_three_slow_launches_in_a_row_with_the_short_rule) {
  const Costs c = Costs::short_launches(), before;
  VCHECK(c.ban_strikes == 3 && before.ban_strikes == 1);
  unsigned s = 0;
  VCHECK(!offload_ends(s, false, c) && !offload_ends(s, false, c) && offload_ends(s, false, c));
  s = 0;
  VCHECK(!offload_ends(s, false, c) && !offload_ends(s, false, c) && !offload_ends(s, true, c) && !offload_ends(s, false, c) && !offload_ends(s, false, c));
  unsigned t = 0;
  VCHECK(offload_ends(t, false, before) && !offload_ends(t = 0, true, before));                  // one is enough before
}

// ---- section 20: a PTX module's variables that no instruction names ---------------------------------------------------------------------------
VTEST(ptx_variables_that_no_instruction_names_do_not_keep_a_module_off_the_card) {
  const std::string head = ".version 8.5\n.target sm_90\n.address_size 64\n";
  const std::string kern = ".visible .entry k(.param .u64 p)\n{\n.reg .b64 %rd<4>;\n.reg .b32 %r<3>;\nld.param.u64 %rd1, [p];\nBODY\nret;\n}\n";
  const auto with = [&](const std::string& decls, const std::string& body) {
    std::string k = kern;
    k.replace(k.find("BODY"), 4, body);
    return head + decls + k;
  };
  const std::string marker = ".global .align 1 .b8 _ZN4cuda3std3__45__cpo5beginE[1];\n.global .align 1 .b8 _ZN4cuda3std3__45__cpo3endE[1];\n";
  // the libcu++ markers: declared, never named: admitted, and counted
  PtxScan r = scan_ptx_for_card(with(marker, "st.global.u32 [%rd1], 1;"), false, true);
  VCHECK(r.refuse.empty() && r.unreferenced_variables == 2);
  // the rule before (`unreferenced_ok = false`) refuses the same text
  r = scan_ptx_for_card(with(marker, "st.global.u32 [%rd1], 1;"), false, true, false);
  VCHECK(!r.refuse.empty() && !r.never);
  // every way an instruction can name a variable: referenced, refused (not class never: a Safety refusal, the module's)
  for (const char* body : {"mov.u64 %rd2, g;", "cvta.global.u64 %rd2, g;", "ld.global.u32 %r1, [g];", "ld.global.u32 %r1, [g+4];", "st.global.u32 [g], %r1;", "atom.global.add.u32 %r1, [g], 1;",
                           "ld.const.u32 %r1, [g];", "mov.u64 %rd2, generic(g);", "ld.global.u32 %r1, [%rd2+g];", "call.uni foo, (g);", "mov.u64 %rd2, g+8;"}) {
    r = scan_ptx_for_card(with(".global .align 4 .b8 g[16];\n", body), false, true);
    VCHECK(!r.refuse.empty() && !r.never && r.refuse.find("module variable") != std::string::npos);
  }
  // another function of the module names it; so does another variable's initialiser
  r = scan_ptx_for_card(head + ".global .align 4 .b8 g[16];\n.func f()\n{\n.reg .b64 %rd<2>;\nmov.u64 %rd1, g;\nret;\n}\n" + with("", "st.global.u32 [%rd1], 1;").substr(head.size()), false, true);
  VCHECK(!r.refuse.empty());
  r = scan_ptx_for_card(with(".global .align 4 .b8 g[16];\n.global .u64 ptr = g;\n", "st.global.u32 [%rd1], 1;"), false, true);
  VCHECK(!r.refuse.empty());
  // linkage that another module or the host can see, __managed__, a declaration list and a name the scan cannot read are all treated as referenced
  for (const char* decl : {".visible .global .align 4 .b8 g[16];\n", ".extern .global .align 4 .b8 g[16];\n", ".weak .global .align 4 .b8 g[16];\n", ".common .global .align 4 .b8 g[16];\n",
                           ".global .attribute(.managed) .align 4 .b8 g[16];\n", ".global .u32 a, b;\n", ".global .align 4 .b8 [16];\n"}) {
    r = scan_ptx_for_card(with(decl, "st.global.u32 [%rd1], 1;"), false, true);
    VCHECK(!r.refuse.empty() && !r.never);
  }
  // a name that is only part of another identifier is not a reference; the name as a whole identifier in a string or a directive is (over-counting is safe)
  r = scan_ptx_for_card(with(".global .align 4 .b8 g[16];\n", "ld.global.u32 %r1, [%rd1+4];\nld.global.u32 %r2, [gg];\nmov.u32 %r1, g2;"), false, true);
  VCHECK(r.refuse.empty());
  r = scan_ptx_for_card(with(".global .align 4 .b8 g[16];\n.pragma \"g\";\n", "st.global.u32 [%rd1], 1;"), false, true);
  VCHECK(!r.refuse.empty());
  // a name with a dot (LLVM names such as `tab.1`): the scan reads identifiers apart at the dot, so a piece of the name counts
  r = scan_ptx_for_card(with(".global .align 4 .b8 tab.1[16];\n", "ld.global.u32 %r1, [tab.1+4];"), false, true);
  VCHECK(!r.refuse.empty());
  r = scan_ptx_for_card(with(".global .align 4 .b8 tab.1[16];\n", "ld.global.u32 %r1, [%rd1+4];"), false, true);
  VCHECK(r.refuse.empty());
  // initialised data of a variable is not a reference to the variable (the data statements hold numbers); a string literal's bytes are not names
  r = scan_ptx_for_card(with(".global .align 1 .b8 s[6] = {104, 101, 108, 108, 111, 0};\n", "st.global.u32 [%rd1], 1;"), false, true);
  VCHECK(r.refuse.empty() && r.unreferenced_variables == 1);
  // textures stay class never whatever else the text says
  r = scan_ptx_for_card(with(".global .texref t;\n", "st.global.u32 [%rd1], 1;"), false, true);
  VCHECK(r.never);
  // the declaration helpers
  VCHECK(ptx_decl_name(".global .align 4 .b8 name[16] =") == "name");
  VCHECK(ptx_decl_name(".global .u32 x") == "x" && ptx_decl_name(".const .align 1 .b8 $str$1[8]") == "$str$1");
  VCHECK(ptx_decl_name(".global .u32 a, b").empty() && ptx_decl_name(".global .align 4").empty());
}

// ---- section 20: assert, trap and the call through a register ----------------------------------------------------------------------------------
VTEST(assert_and_trap_are_admitted_by_turbo_only_and_not_when_switched_off) {
  using K = ExternKind;
  VCHECK(extern_admitted(K::Assert, Mode::Turbo, false, false));
  VCHECK(!extern_admitted(K::Assert, Mode::Turbo, false, true));    // VGPU_HOST_GPU_TURBO_OFF=assert
  VCHECK(!extern_admitted(K::Assert, Mode::Turbo, false));          // (the default of the argument is the rule before)
  VCHECK(!extern_admitted(K::Assert, Mode::Exact, false, false));   // exact mode's promise is the simulator's bits
  for (K k : {K::Heap, K::DeviceLaunch, K::Other}) VCHECK(!extern_admitted(k, Mode::Turbo, false, false));
  VCHECK(extern_admitted(K::Printf, Mode::Turbo, false, false) && !extern_admitted(K::Printf, Mode::Turbo, true, false));
  // BPT: only the TRAP form, only when the caller says so; the debugger forms and an undecoded one stay class never
  const std::vector<std::string> trap{"TRAP"};
  VCHECK(turbo_class("BPT", trap) == InsnClass::Never);
  VCHECK(turbo_class("BPT", trap, true) == InsnClass::Unlisted);
  for (const char* m : {"DRAIN", "CAL", "PAUSE", "INT", "(0)", "(6)", "(7)"}) {
    VCHECK(turbo_class("BPT", {m}, true) == InsnClass::Never);
    VCHECK(!bpt_is_trap("BPT", {m}));
  }
  VCHECK(turbo_class("BPT", {"TRAP", "X"}, true) == InsnClass::Never && turbo_class("BPT", {}, true) == InsnClass::Never);
  VCHECK(turbo_class("BPT", trap, true) != InsnClass::Exact);
  VCHECK(bpt_is_trap("BPT", trap) && !bpt_is_trap("EXIT", trap));
  // the switch
  ClassMask off = 0;
  bool printf_off = true, assert_off = false;
  VCHECK(parse_turbo_off("assert", &off, nullptr, &printf_off, &assert_off) && assert_off && !printf_off && off == 0);
  VCHECK(parse_turbo_off("mufu,assert,printf", &off, nullptr, &printf_off, &assert_off) && assert_off && printf_off && off == class_bit(InsnClass::Mufu));
  std::string bad;
  VCHECK(!parse_turbo_off("assert", &off, &bad, &printf_off) && bad == "assert");   // (a caller that does not know the switch gets a bad name, as for printf)
  VCHECK(parse_turbo_off("", &off, nullptr, &printf_off, &assert_off) && !assert_off);
  // PTX: trap, brkpt and a call to __assertfail are admitted only when the caller says so
  const std::string head = ".version 8.5\n.target sm_75\n.address_size 64\n";
  const std::string af = head + ".extern .func __assertfail (.param .b64 a)\n;\n.visible .entry k()\n{\n{\ncall.uni __assertfail, (a);\n}\nret;\n}\n";
  for (const char* ins : {"trap;", "brkpt;"}) {
    const std::string t = head + ".visible .entry k()\n{\n" + ins + "\nret;\n}\n";
    VCHECK(!scan_ptx_for_card(t, false).refuse.empty());
    const PtxScan r = scan_ptx_for_card(t, false, false, true, false);
    VCHECK(r.refuse.empty() && r.trap);
    VCHECK(!scan_ptx_for_card(t, false, false, true, true).refuse.empty());
  }
  VCHECK(!scan_ptx_for_card(af, false).refuse.empty());
  const PtxScan a = scan_ptx_for_card(af, false, false, true, false);
  VCHECK(a.refuse.empty() && a.trap);
}

VTEST(a_call_through_a_register_is_followed_only_when_its_target_slot_is_known) {
  using K = RcInsn::Kind;
  const auto mov = [](uint32_t reg, int64_t imm) { RcInsn r; r.kind = K::MovImm; r.reg = reg; r.imm = imm; r.dst_reg = reg; r.dst_n = 1; return r; };
  const auto ldc = [](uint32_t dst, uint32_t idx, int64_t off, uint32_t bank = 4, bool wide = true) {
    RcInsn r; r.kind = K::LdcBank; r.reg = idx; r.imm = off; r.bank = bank; r.wide = wide; r.dst_reg = dst; r.dst_n = 2; return r;
  };
  const auto other = [](uint32_t dst, uint32_t n = 1) { RcInsn r; r.dst_reg = dst; r.dst_n = n; return r; };
  const auto call = [](uint32_t reg) { RcInsn r; r.kind = K::CallReg; r.reg = reg; return r; };
  // the sequence nvcc 13 emits for an assert (measured, HeCBench assert-cuda sm_86): MOV R2, 0x0 ; ... ; LDC.64 R2, c[0x4][R2] ; ... ; CALL.ABS.NOINC R2
  std::vector<RcInsn> v{mov(2, 0), other(4), other(5), ldc(2, 2, 0), other(7), other(14), other(20), call(2)};
  VCHECK(register_call_slot(v, 7) == 0);
  v[0] = mov(2, 8);
  VCHECK(register_call_slot(v, 7) == 8);
  v[3] = ldc(2, 2, 0x10);   // (the offset of the load is added)
  VCHECK(register_call_slot(v, 7) == 0x18);
  v[3] = ldc(2, kRcRZ, 0x20);   // (no index register)
  VCHECK(register_call_slot(v, 7) == 0x20);
  v[3] = ldc(2, 2, 0);
  v[0] = mov(2, 0);
  // refused: the index comes from elsewhere, the load is not bank 4 or not 64 bits, another instruction writes the target (either half), a guarded load, a join point
  std::vector<RcInsn> bad = v;
  bad[0] = other(2);
  VCHECK(register_call_slot(bad, 7) == -1);   // (the index register is not set by a MOV immediate)
  bad = v; bad[3] = ldc(2, 2, 0, 3);
  VCHECK(register_call_slot(bad, 7) == -1);
  bad = v; bad[3] = ldc(2, 2, 0, 4, false); bad[3].dst_n = 1;
  VCHECK(register_call_slot(bad, 7) == -1);
  bad = v; bad[5] = other(2);
  VCHECK(register_call_slot(bad, 7) == -1);   // (the target is overwritten after the load)
  bad = v; bad[5] = other(3);
  VCHECK(register_call_slot(bad, 7) == -1);   // (so is its high half)
  bad = v; bad[5] = other(1, 2);
  VCHECK(register_call_slot(bad, 7) == -1);   // (a two-register write that covers R2)
  bad = v; bad[3].guarded = true;
  VCHECK(register_call_slot(bad, 7) == -1);
  bad = v; bad[0].guarded = true;
  VCHECK(register_call_slot(bad, 7) == -1);
  bad = v; bad[5].entry = true;
  VCHECK(register_call_slot(bad, 7) == -1);   // (a join point between the load and the call)
  bad = v; bad[7].entry = true;
  VCHECK(register_call_slot(bad, 7) == -1);   // (the call itself can be entered from elsewhere)
  bad = v; bad[3].entry = true;
  VCHECK(register_call_slot(bad, 7) == -1);   // (entered at the load: the index register is whatever the other path left)
  bad[3] = ldc(2, kRcRZ, 0x20); bad[3].entry = true;
  VCHECK(register_call_slot(bad, 7) == 0x20); // (with no index register, entering AT the load is fine: it is the first thing that runs)
  bad = v; bad.insert(bad.begin(), RcInsn{}); bad[0].kind = K::Control;
  VCHECK(register_call_slot(bad, 8) == 0);    // (a branch BEFORE the index is set does not matter: the walk stops at the index's MOV)
  bad = v; bad[1].kind = K::Control;
  VCHECK(register_call_slot(bad, 7) == -1);   // (one between the index and the load does)
  bad = v; bad[5].kind = K::Control;
  VCHECK(register_call_slot(bad, 7) == -1);   // (a branch or call between the load and the call)
  bad = v; bad[2].entry = true;
  VCHECK(register_call_slot(bad, 7) == -1);   // (a join point between the index and the load: not through MOV ... LDC straight-line)
  volatile size_t far_index = 99;   // (read through a volatile: with the constant, GCC 13 inlines the guarded access and warns array-bounds, which -Werror makes an error)
  VCHECK(register_call_slot(v, 3) == -1 && register_call_slot(v, far_index) == -1);   // (not a call, and past the end)
  bad = v; bad[7].reg = kRcRZ;
  VCHECK(register_call_slot(bad, 7) == -1);
  bad = v; bad[0] = mov(2, -8);
  VCHECK(register_call_slot(bad, 7) == -1);   // (a negative slot)
  // no load at all in the section before the call
  std::vector<RcInsn> none{other(4), call(2)};
  VCHECK(register_call_slot(none, 1) == -1);
}


VTEST_MAIN
