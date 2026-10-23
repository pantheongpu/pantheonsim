// Register residence of the JIT's generated code (src/sass/jit.hpp, RowCtx): random straight-line regions of the generated forms over a
// small set of registers (so rows are reused, written and read again, a destination is often a source, 64-bit pairs overlap), with
// guards (@P, @!P, @UP, @!PT), carries and predicate outputs, and escapes interleaved. Each region is compiled twice, with residence (rows
// kept in locals) and without (every access in the register file, the milestone-2 code), and run on the same random register file,
// predicates, constant bank and group (a whole warp, a part, one lane). The final register file, predicates, counters and everything an
// escape sees (the whole register file and predicates, hashed, at every escape) must be identical; an escape also changes registers and
// predicates, as the tier's handlers do, so a row that is not stored before it, or not reloaded after it, shows. A second run throws from
// an escape in the middle of the region: the register file and counters must then be identical too.
//
// VGPU_JIT_RANDOM_UNITS=<n> (default 4; the rig runs hundreds) units of VGPU_JIT_RANDOM_REGIONS (12) regions each.
// The forms themselves are held to the tier by nvidia/tests/e2e/run_sass_jit.sh; this holds residence to the code without it.
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>

#include "sass/jit.hpp"
#include "sass/tier.hpp"
#include "vtest.hpp"

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define VJ_SAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define VJ_SAN 1
#endif
#endif
#ifndef VJ_SAN
#define VJ_SAN 0
#endif

namespace fs = std::filesystem;
using namespace vgpu::sass;
using tier::Form;

namespace {

constexpr unsigned kRegs = 16;   // rows of the register file
constexpr uint16_t kNo = 0xffff;

struct Gen {
  std::mt19937_64 rng;
  explicit Gen(uint64_t seed) : rng(seed) {}
  uint64_t u(uint64_t n) { return rng() % n; }
  bool coin(unsigned pct) { return u(100) < pct; }

  uint32_t special() {
    static const uint32_t v[] = {0u, 1u, 0xffffffffu, 0x80000000u, 0x7fffffffu, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0x00000001u, 0x80000001u,
                                 0x3f800000u, 0xbf800000u, 0x00400000u, 31u, 32u, 63u, 64u};
    return coin(50) ? v[u(sizeof v / sizeof v[0])] : static_cast<uint32_t>(rng());
  }
  // A 32-bit source.
  jit::SrcFacts src(bool flt) {
    jit::SrcFacts s;
    const unsigned k = static_cast<unsigned>(u(100));
    if (k < 62) {
      s.kind = 0;
      s.off = static_cast<uint32_t>(u(kRegs)) * 32u;
    } else if (k < 72) {
      s.kind = 1;
      s.val = special();
    } else if (k < 86) {
      s.kind = 2;
      s.dyn = 0;
      s.off = static_cast<uint32_t>(u(kRegs));
      s.mod = static_cast<uint8_t>(flt ? (u(4) | (coin(30) ? 4 : 0)) : u(4));
    } else if (k < 93) {
      s.kind = 2;
      s.dyn = 1;
      s.off = static_cast<uint32_t>(u(8)) * 4u;
      s.mod = static_cast<uint8_t>(flt ? u(4) : u(4));
    } else {
      s.kind = 2;
      s.dyn = 3;
      s.off = static_cast<uint32_t>(u(32));
      s.mod = static_cast<uint8_t>(u(4));
    }
    return s;
  }
  // A 64-bit source: a register pair, a constant pair, bank 0 or uniform registers as a pair.
  jit::SrcFacts src64() {
    jit::SrcFacts s;
    const unsigned k = static_cast<unsigned>(u(100));
    if (k < 60) {
      s.kind = 0;
      s.off = static_cast<uint32_t>(u(kRegs - 1)) * 32u;
    } else if (k < 75) {
      s.kind = 1;
      s.val = special();
      s.val2 = special();
    } else if (k < 88) {
      s.kind = 2;
      s.dyn = 1;
      s.off = static_cast<uint32_t>(u(7)) * 4u;
    } else {
      s.kind = 2;
      s.dyn = 3;
      s.off = static_cast<uint32_t>(u(30));
    }
    s.mod = static_cast<uint8_t>(u(4));
    return s;
  }
  jit::SrcFacts pred() {
    jit::SrcFacts s;
    s.kind = 2;
    s.dyn = coin(80) ? 0 : 3;
    s.off = static_cast<uint32_t>(u(8));
    s.mod = coin(30) ? 1 : 0;
    return s;
  }
  uint16_t dreg() { return coin(5) ? kNo : static_cast<uint16_t>(u(kRegs - 1)); }
  uint16_t dpred(bool allow_no) { return allow_no && coin(30) ? kNo : static_cast<uint16_t>(u(8)); }

  jit::OpFacts op() {
    jit::OpFacts o;
    o.lo = rng();
    o.hi = rng();
    o.cls = static_cast<uint8_t>(u(12));
    if (coin(55)) {
      o.unguarded = true;
    } else {
      o.unguarded = false;
      o.guard = static_cast<uint8_t>(u(8));
      o.guard_not = coin(40);
      o.guard_uniform = coin(25);
    }
    if (coin(14)) return o;   // an escape
    static const Form forms[] = {Form::Mov,   Form::Iadd3,    Form::Iadd3x, Form::Imad,   Form::Imad32, Form::ImadWide, Form::ImadHi, Form::Lop3,
                                 Form::Lea,   Form::Shf,      Form::Isetp,  Form::Fsetp,  Form::Fsel,   Form::Ffma,     Form::Fadd,   Form::Fmul,
                                 Form::Prmt,  Form::Imnmx,    Form::I2fp};
    const Form f = forms[u(sizeof forms / sizeof forms[0])];
    o.form = static_cast<uint8_t>(f);
    o.dst = dreg();
    switch (f) {
      case Form::Mov: o.s[0] = src(false); break;
      case Form::Iadd3: for (auto& x : o.s) x = src(false); o.a = static_cast<uint32_t>(u(4)); break;
      case Form::Iadd3x:
        for (auto& x : o.s) x = src(false);
        o.a = static_cast<uint32_t>(u(4));
        o.aux = coin(50) ? 1 : 0;
        o.p = pred();
        o.p2 = pred();
        o.dst2 = dpred(true);
        o.c = coin(40) ? kNo : static_cast<uint32_t>(u(8));
        break;
      case Form::Imad: for (auto& x : o.s) x = src(false); break;
      case Form::Imad32:
        for (auto& x : o.s) x = src(false);
        o.aux = coin(50) ? 1 : 0;
        o.p = pred();
        o.dst2 = dpred(true);
        break;
      case Form::ImadWide:
        o.s[0] = src(false);
        o.s[1] = src(false);
        o.s[2] = src64();
        o.aux = static_cast<uint8_t>(u(4));
        o.p = pred();
        o.dst2 = dpred(true);
        break;
      case Form::ImadHi: {
        o.s[0] = src(false);
        o.s[1] = src(false);
        o.aux = static_cast<uint8_t>(u(8));
        o.s[2] = (o.aux & 4) ? src64() : src(false);
        o.p = pred();
        o.dst2 = dpred(true);
        break;
      }
      case Form::Lop3:
        for (auto& x : o.s) x = src(false);
        o.a = static_cast<uint32_t>(u(256));
        o.aux = coin(60) ? 0 : static_cast<uint8_t>(1 | (coin(50) ? 2 : 0));
        o.p = pred();
        o.dst2 = dpred(false);
        break;
      case Form::Lea:
        o.s[0] = src(false);
        o.s[1] = src(false);
        o.s[2] = src(false);
        o.a = static_cast<uint32_t>(u(34));
        o.aux = static_cast<uint8_t>(u(16));
        o.p = pred();
        o.dst2 = dpred(true);
        break;
      case Form::Shf:
        for (auto& x : o.s) x = src(false);
        o.a = static_cast<uint32_t>(u(4));
        o.aux = static_cast<uint8_t>(u(8));
        break;
      case Form::Isetp:
        o.s[0] = src(false);
        o.s[1] = src(false);
        o.a = static_cast<uint32_t>(1 + u(6));
        o.b = static_cast<uint32_t>(u(3));
        o.c = static_cast<uint32_t>(u(2));
        o.p = pred();
        o.dst = static_cast<uint16_t>(u(8));
        o.dst2 = static_cast<uint16_t>(u(8));
        break;
      case Form::Fsetp:
        o.s[0] = src(false);
        o.s[1] = src(false);
        o.a = static_cast<uint32_t>(u(16));
        o.b = static_cast<uint32_t>(u(3));
        o.c = static_cast<uint32_t>(u(2));
        o.p = pred();
        o.dst = static_cast<uint16_t>(u(8));
        o.dst2 = static_cast<uint16_t>(u(8));
        break;
      case Form::Fsel:
        o.s[0] = src(true);
        o.s[1] = src(true);
        o.p = pred();
        break;
      case Form::Ffma: for (auto& x : o.s) x = src(true); o.a = static_cast<uint32_t>(u(4)); break;
      case Form::Fadd: case Form::Fmul: o.s[0] = src(true); o.s[1] = src(true); o.a = static_cast<uint32_t>(u(4)); break;
      case Form::Prmt: for (auto& x : o.s) x = src(false); break;
      case Form::Imnmx: o.s[0] = src(false); o.s[1] = src(false); o.c = static_cast<uint32_t>(u(2)); o.p = pred(); break;
      case Form::I2fp: o.s[0] = src(false); o.c = static_cast<uint32_t>(u(2)); break;
      default: break;
    }
    if (!jit::generates(o)) {   // (an escape, if the generator will not take it)
      jit::OpFacts e;
      e.lo = o.lo; e.hi = o.hi; e.cls = o.cls; e.guard = o.guard; e.guard_not = o.guard_not; e.guard_uniform = o.guard_uniform; e.unguarded = o.unguarded;
      return e;
    }
    return o;
  }
};

// The machine a region runs on, and what an escape does to it.
struct Sim {
  alignas(32) uint32_t R[kRegs * 32];
  uint32_t p[8];
  bool up[8];
  uint32_t ur[256];
  uint8_t b0[64];
  uint64_t steps = 0, insts = 0, tinsts = 0, by_class[16] = {};
  uint64_t seed = 0;
  std::vector<uint64_t> log;
  long throw_at = -1;
  long calls = 0;
};

uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  h *= 0xff51afd7ed558ccdull;
  return h ^ (h >> 33);
}

void sim_escape(jit::Ctx* c, uint32_t op, uint32_t ex) {
  Sim& s = *static_cast<Sim*>(c->host);
  uint64_t h = mix(s.seed, op * 977u + ex);
  for (unsigned i = 0; i < kRegs * 32; ++i) h = mix(h, s.R[i]);
  for (unsigned i = 0; i < 8; ++i) h = mix(h, s.p[i]);
  s.log.push_back(h);
  if (s.throw_at == s.calls++) throw std::runtime_error("escape fault");
  // The handler's effect: some rows (any of them: it is a handler the generator knows nothing of), for the executing lanes, and a predicate.
  std::mt19937_64 r(h);
  const unsigned nrows = static_cast<unsigned>(r() % 4);
  for (unsigned k = 0; k < nrows; ++k) {
    const unsigned row = static_cast<unsigned>(r() % kRegs);
    const uint32_t v = static_cast<uint32_t>(r());
    for (unsigned l = 0; l < 32; ++l)
      if ((ex >> l) & 1u) s.R[row * 32 + l] = v ^ (l * 0x9e3779b1u);
  }
  if (r() % 3 == 0) {
    const unsigned pr = static_cast<unsigned>(r() % 7);
    s.p[pr] = (s.p[pr] & ~ex) | (static_cast<uint32_t>(r()) & ex);
  }
}
void sim_note(jit::Ctx*, uint32_t) {}

jit::Ctx make_ctx(Sim& s, bool stats) {
  jit::Ctx c{};
  c.abi = jit::kAbiVersion;
  c.flags = stats ? 1u : 0u;
  c.up = s.up;
  c.p = s.p;
  c.ur = s.ur;
  c.steps = &s.steps;
  c.instructions = &s.insts;
  c.thread_instructions = &s.tinsts;
  c.by_class = s.by_class;
  c.escape = &sim_escape;
  c.note = &sim_note;
  c.host = &s;
  c.R = s.R;
  c.b0 = s.b0;
  c.b0n = sizeof s.b0;
  return c;
}

void randomize(Sim& s, std::mt19937_64& rng) {
  for (auto& v : s.R) v = static_cast<uint32_t>(rng());
  for (auto& v : s.p) v = static_cast<uint32_t>(rng());
  s.p[7] = 0xffffffffu;
  for (auto& v : s.up) v = rng() & 1;
  s.up[7] = true;
  for (auto& v : s.ur) v = static_cast<uint32_t>(rng());
  for (auto& v : s.b0) v = static_cast<uint8_t>(rng());
  s.steps = s.insts = s.tinsts = 0;
  std::memset(s.by_class, 0, sizeof s.by_class);
  s.log.clear();
  s.calls = 0;
}

bool same(const Sim& a, const Sim& b) {
  return std::memcmp(a.R, b.R, sizeof a.R) == 0 && std::memcmp(a.p, b.p, sizeof a.p) == 0 && a.steps == b.steps && a.insts == b.insts &&
         a.tinsts == b.tinsts && std::memcmp(a.by_class, b.by_class, sizeof a.by_class) == 0 && a.log == b.log;
}

}  // namespace

VTEST(jit_residence_random_regions_equal_the_register_file_code) {
  if (VJ_SAN) return;
  {
    jit::Config probe;
    jit::Engine e(probe);
    if (e.compiler_id().empty()) {
      std::printf("  (no C++ compiler on PATH: skipped)\n");
      return;
    }
  }
  const unsigned units = static_cast<unsigned>(std::strtoul(std::getenv("VGPU_JIT_RANDOM_UNITS") ? std::getenv("VGPU_JIT_RANDOM_UNITS") : "4", nullptr, 10));
  const unsigned nreg = static_cast<unsigned>(std::strtoul(std::getenv("VGPU_JIT_RANDOM_REGIONS") ? std::getenv("VGPU_JIT_RANDOM_REGIONS") : "12", nullptr, 10));
  const uint64_t seed0 = std::getenv("VGPU_JIT_RANDOM_SEED") ? std::strtoull(std::getenv("VGPU_JIT_RANDOM_SEED"), nullptr, 10) : 1;
  const char* t = std::getenv("TMPDIR");
  std::string tmpl = std::string(t && *t ? t : "/tmp") + "/vgpu-test-jit-resid.XXXXXX";
  std::vector<char> b(tmpl.begin(), tmpl.end());
  b.push_back(0);
  if (!::mkdtemp(b.data())) throw vtest::Failure("mkdtemp failed");
  const std::string dir = b.data();
  std::atomic<unsigned> next{0}, failures{0}, regions_checked{0}, runs{0}, escapes_seen{0}, gen_ops{0};
  std::mutex mu;
  std::string first_failure;
  const auto worker = [&] {
    for (;;) {
      const unsigned unit = next.fetch_add(1);
      if (unit >= units) return;
      Gen g(seed0 * 1000003u + unit);
      jit::UnitSpec spec;
      spec.sm = 80;
      for (unsigned r = 0; r < nreg; ++r) {
        jit::RegionSpec rs;
        const unsigned n = 4 + static_cast<unsigned>(g.u(36));
        for (unsigned i = 0; i < n; ++i) {
          rs.ops.push_back(g.op());
          if (rs.ops.back().form) ++gen_ops;
        }
        spec.regions.push_back(std::move(rs));
      }
      jit::Config on, off;
      on.cache_dir = dir + "/on";
      off.cache_dir = dir + "/off";
      off.residence = false;
      jit::Engine eon(on), eoff(off);
      const jit::Result ron = eon.obtain(spec), roff = eoff.obtain(spec);
      if (!ron.found || !roff.found) {
        std::lock_guard<std::mutex> lk(mu);
        ++failures;
        if (first_failure.empty()) first_failure = "unit " + std::to_string(unit) + ": compile failed: " + (ron.found ? roff.message : ron.message);
        continue;
      }
      std::mt19937_64 rng(seed0 * 7919u + unit);
      for (unsigned r = 0; r < nreg; ++r) {
        ++regions_checked;
        const uint32_t glanes_of[] = {32, 32, 32};
        (void)glanes_of;
        for (unsigned trial = 0; trial < 6; ++trial) {
          Sim a, bsim;
          randomize(a, rng);
          uint32_t group = trial < 2 ? 0xffffffffu : trial == 2 ? 0x0000ffffu : trial == 3 ? 0x00000001u : static_cast<uint32_t>(rng()) | 1u;
          if (trial == 5) group = 0xffffu << 16;   // the second half of a warp (a 48-thread block's last warp is the first half)
          bsim = a;
          a.seed = bsim.seed = rng();
          const bool stats = trial & 1;
          const bool throw_case = trial == 4 && r % 2 == 0;
          if (throw_case) a.throw_at = bsim.throw_at = static_cast<long>(rng() % 3);
          jit::Ctx ca = make_ctx(a, stats), cb = make_ctx(bsim, stats);
          bool ta = false, tb = false;
          const uint32_t gl = static_cast<uint32_t>(__builtin_popcount(group));
          try { ron.lib->fns[r](&ca, group, gl); } catch (const std::runtime_error&) { ta = true; }
          try { roff.lib->fns[r](&cb, group, gl); } catch (const std::runtime_error&) { tb = true; }
          ++runs;
          escapes_seen += static_cast<unsigned>(a.log.size());
          if (ta != tb || !same(a, bsim)) {
            std::lock_guard<std::mutex> lk(mu);
            ++failures;
            if (first_failure.empty()) {
              first_failure = "unit " + std::to_string(unit) + " region " + std::to_string(r) + " trial " + std::to_string(trial) + ": differs (R " +
                              (std::memcmp(a.R, bsim.R, sizeof a.R) ? "differs" : "same") + ", p " + (std::memcmp(a.p, bsim.p, sizeof a.p) ? "differs" : "same") +
                              ", counters " + (a.insts == bsim.insts && a.tinsts == bsim.tinsts ? "same" : "differ") + ", escape logs " + (a.log == bsim.log ? "same" : "differ") +
                              ", threw " + std::to_string(ta) + "/" + std::to_string(tb) + ")";
            }
          }
        }
      }
    }
  };
  const unsigned nthreads = std::min<unsigned>(units, std::max(1u, std::thread::hardware_concurrency()));
  std::vector<std::thread> pool;
  for (unsigned i = 0; i < nthreads; ++i) pool.emplace_back(worker);
  for (auto& th : pool) th.join();
  std::printf("  residence: %u units, %u regions, %u runs, %u generated ops, %u escapes executed, %u failures\n", units, regions_checked.load(), runs.load(),
              gen_ops.load(), escapes_seen.load(), failures.load());
  std::error_code ec;
  fs::remove_all(dir, ec);
  if (failures) throw vtest::Failure(first_failure);
}

VTEST_MAIN
