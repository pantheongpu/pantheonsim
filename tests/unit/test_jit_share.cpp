// Tier 2, round 14: the share rule (src/sass/jit.hpp, ShareRule). A section is sampled once, after it has run `sample_after` warp instructions in the hook loop;
// if the generated code would have computed less than `min_share` of them it is declined, goes back to the plain run loop and asks neither the cache nor the
// compiler. These tests drive the decision without an executor (Unit::note and Unit::sample, as the run loop calls them) with the rule on: the settings of
// test_jit.cpp turn it off (HOT_PERMILLE=0 compiles every region), these keep the defaults of everything but the sample.
//
// The executor side (a result, a counter or a fault is the same whatever the rule decides) is nvidia/tests/e2e/run_sass_jit.sh.
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "sass/jit.hpp"
#include "sass/tier.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

namespace fs = std::filesystem;
using namespace vgpu::sass;

namespace {

struct EnvInit {
  fs::path dir;
  EnvInit() {
    const char* t = std::getenv("TMPDIR");
    dir = fs::path(t && *t ? t : "/tmp") / ("vgpu-test-jit-share." + std::to_string(::getpid()));
    fs::create_directories(dir);
    ::setenv("VGPU_JIT", "1", 1);
    ::setenv("VGPU_JIT_SYNC", "1", 1);
    ::setenv("VGPU_JIT_THRESHOLD", "1000000", 1);
    ::setenv("VGPU_JIT_SAMPLE", "2000", 1);
    ::setenv("VGPU_JIT_MIN_SHARE", "60", 1);
    ::unsetenv("VGPU_JIT_HOT_PERMILLE");
    ::unsetenv("VGPU_JIT_MIN_REGION");
    ::unsetenv("VGPU_JIT_LOOKUP_AFTER");
    ::setenv("VGPU_JIT_CACHE", (dir / "cache").c_str(), 1);
    ::setenv("VGPU_QUIET", "1", 1);
    ::unsetenv("VGPU_TIER");
  }
  ~EnvInit() { std::error_code ec; fs::remove_all(dir, ec); }
} g_env;

void dummy_handler(void*, const tier::MOp&, uint32_t) {}

// A section of `kinds`: 'g' an op the generated code computes (a MOV of a constant), 'a' an op it calls the tier's handler for (an escape), 'x' an op with no
// handler (it ends a region).
struct FakeProgram {
  std::vector<Instr> instrs;
  tier::Program prog;
  explicit FakeProgram(const std::string& kinds) {
    instrs.resize(kinds.size());
    prog.ops.resize(kinds.size());
    prog.pool.resize(1);
    for (size_t i = 0; i < kinds.size(); ++i) {
      Instr& in = instrs[i];
      in.w.lo = 0x1111000000000000ull + i;
      in.w.hi = 0x2222;
      in.op = Op::IADD3;
      tier::MOp& op = prog.ops[i];
      op.ins = &instrs[i];
      op.straight = true;
      op.fn = kinds[i] == 'x' ? nullptr : &dummy_handler;
      op.form = kinds[i] == 'g' ? tier::Form::Mov : tier::Form::None;
      op.unguarded = true;
      op.maxreg = static_cast<uint16_t>(4 + i);
    }
  }
};

}  // namespace

VTEST(jit_share_rule_arithmetic) {
  jit::ShareRule r;
  r.on = true;
  r.min_share_permille = 600;
  r.sample_after = 1000;
  VCHECK(jit::share_ok(600, 1000, r));    // exactly the bar: kept
  VCHECK(!jit::share_ok(599, 1000, r));
  VCHECK(jit::share_ok(1000, 1000, r));
  VCHECK(!jit::share_ok(0, 1000, r));
  VCHECK(jit::share_ok(0, 0, r));          // nothing sampled: nothing to decide on
  r.on = false;
  VCHECK(jit::share_ok(0, 1000, r));       // the rule off keeps everything
  // (no overflow for the largest runs the simulator makes: 10^13 warp instructions)
  r.on = true;
  VCHECK(jit::share_ok(6'000'000'000'000ull, 10'000'000'000'000ull, r));
  VCHECK(!jit::share_ok(5'999'999'999'999ull, 10'000'000'000'000ull, r));
}

VTEST(jit_share_rule_reads_its_settings) {
  if (!jit::requested()) return;
  const jit::ShareRule& r = jit::share_rule();
  VCHECK(r.on);
  VCHECK_EQ(r.min_share_permille, 600u);
  VCHECK_EQ(r.sample_after, uint64_t{2000});
}

VTEST(jit_a_section_with_no_region_that_is_half_generated_never_gets_the_hook) {
  if (!jit::requested()) return;
  VCHECK(jit::make_unit(FakeProgram("aaaaaaaaaaaaaaaa").prog, 80) == nullptr);          // a region of escapes only
  VCHECK(jit::make_unit(FakeProgram("gaaaaaaxgaaaaaaxaaaaaaa").prog, 80) == nullptr);   // 1 of 7 in each
  VCHECK(jit::make_unit(FakeProgram("ggxggxggxggxgggg").prog, 80) == nullptr);          // regions of 2: shorter than VGPU_JIT_MIN_REGION (4)
  const auto u = jit::make_unit(FakeProgram("ggggggggaaaaaaaaaaaaaaaa").prog, 80);       // one region of 24, 8 generated: a third, not half
  VCHECK(u == nullptr);
  const auto v = jit::make_unit(FakeProgram("ggggggggggaaaaaaaaaaaaaaaa").prog, 80);     // 10 of 26: not half
  VCHECK(v == nullptr);
  const auto ok = jit::make_unit(FakeProgram("ggggggaaaaax").prog, 80);                  // 6 of 11: half or more
  VCHECK(ok != nullptr);
  VCHECK(ok->sampling.load());                                                           // (and it starts sampled)
  VCHECK(ok->hook());
}

VTEST(jit_a_section_whose_regions_are_a_small_share_of_its_work_is_declined_and_asks_nobody) {
  if (!jit::requested()) return;
  const uint64_t declined0 = jit::stats().declined.load(), accepted0 = jit::stats().accepted.load(), hits0 = jit::stats().cache_hits.load(),
                 compiles0 = jit::stats().compiles.load(), fail0 = jit::stats().failures.load();
  FakeProgram fp("ggggggggggx");   // one region of 10, all generated
  const auto u = jit::make_unit(fp.prog, 80);
  VCHECK(u != nullptr);
  VCHECK_EQ(u->regions.size(), size_t{1});
  VCHECK_EQ(u->regions[0].gen, 10u);
  // The run loop ran 1999 warp instructions of the section, 590 of them in the region (59 entries): not yet a sample.
  u->note(0, 590);
  u->sample(1999);
  VCHECK(u->sampling.load());
  VCHECK(u->hook());
  // One more instruction completes it: 590 of 2000 is 295 per thousand, below the bar of 600.
  u->sample(1);
  VCHECK(!u->sampling.load());
  VCHECK(u->state.load() == static_cast<int>(jit::State::Declined));
  VCHECK(!u->hook());
  VCHECK_EQ(jit::stats().declined.load(), declined0 + 1);
  VCHECK_EQ(jit::stats().accepted.load(), accepted0);
  // Declined for good: even the region running a million instructions asks neither the cache nor the compiler, and nothing is counted as a failure.
  u->note(0, 2'000'000);
  u->sample(5'000'000);
  VCHECK(u->state.load() == static_cast<int>(jit::State::Declined));
  VCHECK_EQ(jit::stats().cache_hits.load(), hits0);
  VCHECK_EQ(jit::stats().compiles.load(), compiles0);
  VCHECK_EQ(jit::stats().failures.load(), fail0);
}

VTEST(jit_a_section_whose_regions_are_most_of_its_work_is_kept_and_asks_after_the_sample) {
  if (!jit::requested()) return;
  const uint64_t declined0 = jit::stats().declined.load(), accepted0 = jit::stats().accepted.load();
  FakeProgram fp("ggggggggggx");
  const auto u = jit::make_unit(fp.prog, 80);
  VCHECK(u != nullptr);
  u->note(0, 1200);   // 120 entries of the region of 10
  u->sample(2000);    // 1200 of 2000: exactly the bar (600 per thousand)
  VCHECK(!u->sampling.load());
  VCHECK(u->state.load() != static_cast<int>(jit::State::Declined));
  VCHECK(u->hook());
  VCHECK_EQ(jit::stats().accepted.load(), accepted0 + 1);
  VCHECK_EQ(jit::stats().declined.load(), declined0);
  u->sample(1'000'000);   // (decided: a later sample changes nothing)
  VCHECK(u->state.load() != static_cast<int>(jit::State::Declined));
}

VTEST(jit_the_cache_is_not_asked_before_the_share_rule_has_decided) {
  if (!jit::requested()) return;
  const uint64_t hits0 = jit::stats().cache_hits.load(), failures0 = jit::stats().failures.load();
  FakeProgram fp("ggggggggggx");
  const auto u = jit::make_unit(fp.prog, 80);
  VCHECK(u != nullptr);
  u->note(0, 1'500'000);   // past the lookup point (the smaller of 1 M and the threshold) with no sample in yet: the lookup waits
  VCHECK(u->state.load() == static_cast<int>(jit::State::Idle));
  VCHECK_EQ(jit::stats().cache_hits.load(), hits0);
  VCHECK_EQ(jit::stats().failures.load(), failures0);
  u->sample(2000);   // 1.5 M of 2000: more than all of it. (A share over 1000 is no reason to decline.)
  VCHECK(u->state.load() != static_cast<int>(jit::State::Declined));
  u->note(0, 4096);  // now it may ask (a note that passes a multiple of 4096 is where the section's total is worked out)
  VCHECK(u->state.load() != static_cast<int>(jit::State::Idle));
}

VTEST_MAIN
