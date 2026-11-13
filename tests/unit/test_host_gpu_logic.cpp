// The host-GPU mode's decisions that need no card: which kernels may run on it with the simulator's bits, the chunk residency
// protocol, and the size rule (include/vgpu/host_gpu.hpp). The part that drives the real GPU is run by
// nvidia/tests/e2e/run_host_gpu.sh, on a machine with a card.
#include <chrono>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <initializer_list>
#include <map>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "vgpu/host_gpu.hpp"
#include "vgpu/host_gpu_image.hpp"
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
  VCHECK(first_launch_offload(gemm_shape(), c, false, c.start_cost_s));   // or the program has spent as much in the simulator already
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
  VCHECK(turbo_class("MUFU", {"(10)"}) == C::Never);   // a function the decoder cannot name
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
    VCHECK(turbo_class(m, {"E", "(9)"}) == C::Never);
  }
  VCHECK(turbo_class("ATOMS", {"POPC", "INC", "32"}) == C::AtomInt);
  VCHECK(turbo_class("ATOMS", {"ARRIVE", "64"}) == C::Never);
  // extra: named one by one
  for (const char* m : {"IDP", "VABSDIFF", "VABSDIFF4", "VIADDMNMX", "VIMNMX3", "FMNMX3", "F2IP", "FSWZADD", "REDUX", "LDGSTS", "LDGDEPBAR", "LDSM", "MOVM",
                        "FFMA2", "FMUL2", "FADD2", "B2R", "NANOSLEEP"})
    VCHECK(turbo_class(m, {}) == C::Extra);
  VCHECK(turbo_class("LEPC", {}) == C::Call && !exact_opcode_allowed("LEPC", {}));   // (Round sp66: the return address of vprintf's call)
  VCHECK(turbo_class("B2R", {"RESULT"}) == C::Extra && !exact_opcode_allowed("B2R", {"RESULT"}));   // (Round sp66; exact's set is unchanged)
  // never: textures, surfaces, what nobody named, what is not an instruction
  for (const char* m : {"TEX", "TLD", "TLD4", "TXD", "SULD", "SUST", "QMMA", "BMMA", "OMMA", "HGMMA", "JMP", "BPT", "SYNCS", "STSM", "UTCMMA",
                        "LDTM", "STTM", "UTMALDG", "VPRINTF", "", "mufu", "Unknown"})
    VCHECK(turbo_class(m, {}) == C::Never);
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
  VCHECK_EQ(n[InsnClass::Tensor], 3u);
  VCHECK_EQ(n[InsnClass::AtomInt], 3u);   // ATOM, ATOMG, ATOMS with no modifier
  VCHECK_EQ(n[InsnClass::Extra], 17u);   // (B2R is a mnemonic of the BAR op, not an op of its own; NANOSLEEP is one)
  VCHECK_EQ(n[InsnClass::FloatRed] + n[InsnClass::AtomFloat] + n[InsnClass::AtomSwap], 0u);
  VCHECK_EQ(n[InsnClass::Exact] + n[InsnClass::Never] + 32u, static_cast<unsigned>(names.size()));
  VCHECK(n[InsnClass::Never] >= 30);
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
  // VGPU_HOST_GPU_ATOMICS=0: float atomics with a value and swaps stay; integer atomics are the off list's
  TurboSwitches no;
  no.atomics = false;
  VCHECK(turbo_refusal(mask({C::Exact, C::AtomFloat}), no) == TurboRefusal::Atomics);
  VCHECK(turbo_refusal(mask({C::Exact, C::AtomSwap}), no) == TurboRefusal::Atomics);
  VCHECK(turbo_refusal(mask({C::Exact, C::AtomInt, C::Mufu}), no) == TurboRefusal::None);
  // the off list: each name switches its class off, and only its class
  const struct { const char* name; std::vector<C> off; } rows[] = {
      {"mufu", {C::Mufu}}, {"tensor", {C::Tensor}}, {"half", {C::Half}}, {"call", {C::Call}}, {"red", {C::FloatRed}}, {"extra", {C::Extra}},
      {"atomics", {C::AtomInt, C::AtomFloat, C::AtomSwap}}};
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
  VCHECK(parse_turbo_off("mufu,tensor,half,call,atomics,red,extra", &sw.off));
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

VTEST_MAIN
