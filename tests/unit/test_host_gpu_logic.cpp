// The host-GPU mode's decisions that need no card: which kernels may run on it with the simulator's bits, the chunk residency
// protocol, and the size rule (include/vgpu/host_gpu.hpp). The part that drives the real GPU is run by
// nvidia/tests/e2e/run_host_gpu.sh, on a machine with a card.
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "vgpu/host_gpu.hpp"
#include "vtest.hpp"

using namespace vgpu::hostgpu;

VTEST(mode_is_exact_or_off) {
  VCHECK(mode_from_text("exact") == Mode::Exact);
  VCHECK(mode_from_text(nullptr) == Mode::Off);
  VCHECK(mode_from_text("") == Mode::Off);
  VCHECK(mode_from_text("approx") == Mode::Off);   // not built: a typo must not turn a mode on
  VCHECK(mode_from_text("EXACT") == Mode::Off);
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
  void card_launch(std::mt19937& rng) {
    for (uint64_t p : res.card_prepare()) card[p] = host[p];
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
      switch (rng() % 5) {
        case 0: w.sim_write(c, true); break;
        case 1: w.sim_write(c, false); break;
        case 2:
        case 3: w.sim_read(c); break;
        default: w.card_launch(rng); break;
      }
    }
  }
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
  t.record("k", 10, 5.0);                    // a cold first launch (decode, set-up): 0.5 s per block
  VCHECK(!t.predict("k", 10).has_value());   // one sample is not enough: it is the cold one
  t.record("k", 10, 0.5);                    // the warm launch: 50 ms per block
  VCHECK(t.predict("k", 10).has_value());
  VCHECK(*t.predict("k", 10) > 0.49 && *t.predict("k", 10) < 0.51);       // the smaller of the two
  VCHECK(*t.predict("k", 20) > 0.99 && *t.predict("k", 20) < 1.01);       // twice the blocks, twice the time
  VCHECK(!t.predict("other", 10).has_value());
  t.record("k", 10, 9.0);                    // a slower launch never raises the estimate: it errs towards the simulator
  VCHECK(*t.predict("k", 10) < 0.51);
  t.record("k", 10, 0.1);                    // a faster one lowers it
  VCHECK(*t.predict("k", 10) < 0.11);
}

VTEST(a_launch_that_lost_to_the_estimate_ends_offloading) {
  VCHECK(offload_paid_off(0.010, 0.050));
  VCHECK(offload_paid_off(0.050, 0.050));
  VCHECK(!offload_paid_off(0.100, 0.003));   // 3DCONV: 100 ms of copies for a 3 ms simulation
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

VTEST_MAIN
