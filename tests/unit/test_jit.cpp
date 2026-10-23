// Tier 2, milestone 1 (src/sass/jit.hpp): the generator, the key, the cache and its verification, the compile and its limits, the
// failure modes, exceptions through generated frames, region discovery, and the promotion state machine. Needs a C++ compiler on
// PATH (the tests that compile skip with a note without one) and does nothing in a sanitizer build, where the JIT is off.
//
// The executor side (counters and faults inside Runner::tier_run, against the tier) is nvidia/tests/e2e/run_sass_jit.sh.
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <thread>

#include "sass/jit.hpp"
#include "sass/tier.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define VJ_SANITIZED 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define VJ_SANITIZED 1
#endif
#endif
#ifndef VJ_SANITIZED
#define VJ_SANITIZED 0
#endif

namespace fs = std::filesystem;
using namespace vgpu::sass;

namespace {

// The manager (promotion, the executor's units) reads its settings once, at its first use: set them before main, deterministic and
// private to this test.
struct EnvInit {
  fs::path dir;
  EnvInit() {
    const char* t = std::getenv("TMPDIR");
    dir = fs::path(t && *t ? t : "/tmp") / ("vgpu-test-jit-mgr." + std::to_string(::getpid()));
    fs::create_directories(dir);
    ::setenv("VGPU_JIT", "1", 1);
    ::setenv("VGPU_JIT_SYNC", "1", 1);
    ::setenv("VGPU_JIT_THRESHOLD", "100", 1);
    ::setenv("VGPU_JIT_MIN_REGION", "2", 1);
    ::setenv("VGPU_JIT_STATS", "1", 1);
    ::setenv("VGPU_JIT_HOT_PERMILLE", "0", 1);   // (every region is compiled: the model test below runs them all)
    ::setenv("VGPU_JIT_CACHE", (dir / "cache").c_str(), 1);
    ::setenv("VGPU_QUIET", "1", 1);
    ::unsetenv("VGPU_TIER");
  }
  ~EnvInit() { std::error_code ec; fs::remove_all(dir, ec); }
} g_env;

struct TmpDir {
  fs::path p;
  TmpDir() {
    const char* t = std::getenv("TMPDIR");
    std::string tmpl = std::string(t && *t ? t : "/tmp") + "/vgpu-test-jit.XXXXXX";
    std::vector<char> b(tmpl.begin(), tmpl.end());
    b.push_back(0);
    if (!::mkdtemp(b.data())) throw vtest::Failure("mkdtemp failed");
    p = b.data();
  }
  ~TmpDir() { std::error_code ec; fs::remove_all(p, ec); }
  std::string str(const char* sub = "") const { return (p / sub).string(); }
};

bool have_compiler() {
  jit::Config c;
  jit::Engine e(c);
  return !e.compiler_id().empty();
}
#define NEED_COMPILER()                                                          \
  do {                                                                           \
    if (VJ_SANITIZED) { std::printf("  (sanitizer build: the JIT is off)\n"); return; } \
    if (!have_compiler()) { std::printf("  (no C++ compiler on PATH: skipped)\n"); return; } \
  } while (0)

jit::UnitSpec make_spec(size_t nregions, size_t ops, uint64_t seed, int sm = 80) {
  std::mt19937_64 rng(seed);
  jit::UnitSpec u;
  u.sm = sm;
  for (size_t r = 0; r < nregions; ++r) {
    jit::RegionSpec rs;
    for (size_t i = 0; i < ops; ++i) {
      jit::OpFacts f;
      f.lo = rng();
      f.hi = rng();
      f.cls = static_cast<uint8_t>(rng() % 12);
      const unsigned kind = static_cast<unsigned>(rng() % 6);
      if (kind < 2) {
        f.unguarded = true;
      } else {
        f.unguarded = false;
        f.guard = static_cast<uint8_t>(rng() % 8);   // 7 is PT
        f.guard_not = rng() & 1;
        f.guard_uniform = kind == 5;
      }
      rs.ops.push_back(f);
    }
    u.regions.push_back(std::move(rs));
  }
  return u;
}

// What the generated region does, written out again from the run loop's rules (tier_ops.inc: tier_guard and the counters).
struct Counters {
  uint64_t steps = 0, instructions = 0, thread_instructions = 0, by_class[16] = {};
  bool operator==(const Counters& o) const {
    return steps == o.steps && instructions == o.instructions && thread_instructions == o.thread_instructions &&
           std::memcmp(by_class, o.by_class, sizeof by_class) == 0;
  }
};
using Calls = std::vector<std::pair<uint32_t, uint32_t>>;

struct Rec {
  Calls calls;
  long throw_at = -1;
};
void rec_escape(jit::Ctx* c, uint32_t op, uint32_t ex) {
  Rec* r = static_cast<Rec*>(c->host);
  r->calls.emplace_back(op, ex);
  if (static_cast<long>(r->calls.size()) - 1 == r->throw_at) throw std::runtime_error("boom at " + std::to_string(op));
}

uint32_t model_ex(const jit::OpFacts& o, uint32_t group, const uint32_t* p, const bool* up) {
  if (o.unguarded) return group;
  if (o.guard == 7 && !o.guard_not) return group;
  uint32_t g;
  if (o.guard_uniform) g = (o.guard == 7 || up[o.guard]) ? 0xffffffffu : 0;
  else g = o.guard == 7 ? 0xffffffffu : p[o.guard];
  if (o.guard_not) g = ~g;
  return group & g;
}

void model_run(const jit::RegionSpec& r, uint32_t group, const uint32_t* p, const bool* up, bool stats, Counters& c, Calls& calls) {
  for (size_t i = 0; i < r.ops.size(); ++i) {
    const uint32_t ex = model_ex(r.ops[i], group, p, up);
    const uint32_t lanes = static_cast<uint32_t>(__builtin_popcount(ex));
    ++c.steps;
    ++c.instructions;
    c.thread_instructions += lanes;
    c.by_class[r.ops[i].cls] += lanes;
    if (ex || stats) calls.emplace_back(static_cast<uint32_t>(i), ex);
  }
}

struct Run {
  Counters counters;
  Calls calls;
  bool threw = false;
  std::string what;
};
Run run_region(jit::RegionFn fn, uint32_t group, const uint32_t* p, const bool* up, bool stats, long throw_at = -1) {
  Run out;
  Rec rec;
  rec.throw_at = throw_at;
  jit::Ctx c{};
  c.abi = jit::kAbiVersion;
  c.flags = stats ? 1u : 0u;
  c.up = up;
  c.p = const_cast<uint32_t*>(p);
  c.steps = &out.counters.steps;
  c.instructions = &out.counters.instructions;
  c.thread_instructions = &out.counters.thread_instructions;
  c.by_class = out.counters.by_class;
  c.escape = &rec_escape;
  c.host = &rec;
  try {
    fn(&c, group, static_cast<uint32_t>(__builtin_popcount(group)));
  } catch (const std::runtime_error& e) {
    out.threw = true;
    out.what = e.what();
  }
  out.calls = rec.calls;
  return out;
}

// Every region of a loaded unit against the model, on random groups and predicates.
void check_against_model(const jit::UnitSpec& spec, const jit::Loaded& lib, uint64_t seed, int trials) {
  VCHECK_EQ(lib.fns.size(), spec.regions.size());
  std::mt19937_64 rng(seed);
  for (int t = 0; t < trials; ++t) {
    uint32_t p[8];
    bool up[8];
    for (int i = 0; i < 8; ++i) {
      p[i] = static_cast<uint32_t>(rng());
      if (rng() % 5 == 0) p[i] = rng() & 1 ? 0u : 0xffffffffu;
      up[i] = rng() & 1;
    }
    uint32_t group = static_cast<uint32_t>(rng());
    if (t % 3 == 0) group = 0xffffffffu;
    if (group == 0) group = 1;
    const bool stats = t % 2;
    for (size_t r = 0; r < spec.regions.size(); ++r) {
      Counters mc;
      Calls mcalls;
      model_run(spec.regions[r], group, p, up, stats, mc, mcalls);
      const Run got = run_region(lib.fns[r], group, p, up, stats);
      VCHECK(!got.threw);
      VCHECK(got.counters == mc);
      VCHECK(got.calls == mcalls);
    }
  }
}

jit::Result must_obtain(jit::Engine& e, const jit::UnitSpec& spec) {
  jit::Result r = e.obtain(spec);
  if (!r.found) throw vtest::Failure(std::string("obtain failed: ") + jit::status_name(r.status) + ": " + r.message);
  return r;
}

std::string write_script(const fs::path& p, const std::string& body) {
  std::ofstream(p) << "#!/bin/sh\n" << body;
  ::chmod(p.c_str(), 0755);
  return p.string();
}

// A compiler that reports another identity and otherwise is the system's.
std::string fake_compiler(const TmpDir& d, const char* name, const char* id) {
  return write_script(d.p / name, std::string("if [ \"$1\" = --version ]; then echo '") + id + "'; exit 0; fi\n" +
                                      "if [ \"$1\" = -dumpmachine ]; then echo fake-machine; exit 0; fi\nexec c++ \"$@\"\n");
}

size_t count_files(const fs::path& dir, const char* ext) {
  size_t n = 0;
  for (const auto& e : fs::directory_iterator(dir))
    if (e.path().extension() == ext) ++n;
  return n;
}

}  // namespace

// ---- the generator and the key -------------------------------------------------------------------------------------------------------------

VTEST(jit_source_is_a_function_of_the_spec_and_names_every_layout_offset) {
  const jit::UnitSpec a = make_spec(3, 7, 1), b = make_spec(3, 7, 1), c = make_spec(3, 7, 2);
  VCHECK(jit::Engine::source(a) == jit::Engine::source(b));
  VCHECK(jit::Engine::source(a) != jit::Engine::source(c));
  const std::string src = jit::Engine::source(a);
  VCHECK_CONTAINS(src, "static_assert(sizeof(Ctx) == " + std::to_string(sizeof(jit::Ctx)));
  VCHECK_CONTAINS(src, "static_assert(offsetof(Ctx, escape) == " + std::to_string(offsetof(jit::Ctx, escape)));
  VCHECK_CONTAINS(src, "vgpu_jit_r2(");
  // Nothing in the text but numbers and fixed templates: no byte of it comes from a string in the input.
  for (char ch : src) VCHECK(ch == '\n' || (ch >= 32 && ch < 127));
}

VTEST(jit_key_depends_on_everything_it_must) {
  NEED_COMPILER();
  TmpDir d;
  const jit::UnitSpec base = make_spec(2, 5, 3);
  jit::Config cfg;
  cfg.cache_dir = d.str("c");
  jit::Engine e(cfg);
  const std::string k0 = e.key(base);
  VCHECK_EQ(k0.size(), size_t{32});
  VCHECK_EQ(e.key(base), k0);
  jit::UnitSpec w = base;
  w.regions[1].ops[3].lo ^= 1;   // one bit of one instruction word
  VCHECK(e.key(w) != k0);
  jit::UnitSpec g = base;
  g.regions[0].ops[0].guard_not = !g.regions[0].ops[0].guard_not;   // a fact the counters use
  VCHECK(e.key(g) != k0);
  jit::UnitSpec a = base;
  a.sm = 90;
  VCHECK(e.key(a) != k0);   // the architecture
  jit::UnitSpec r = base;
  r.regions.pop_back();
  VCHECK(e.key(r) != k0);
  jit::Config f = cfg;
  f.extra_flags = "-fno-exceptions";
  VCHECK(jit::Engine(f).key(base) != k0);   // the flag set
  jit::Config i = cfg;
  i.isa = "somewhere-else";
  VCHECK(jit::Engine(i).key(base) != k0);   // the ISA level
  const std::string cc1 = fake_compiler(d, "cc1", "fakecc 1.0"), cc2 = fake_compiler(d, "cc2", "fakecc 2.0");
  jit::Config c1 = cfg, c2 = cfg;
  c1.cxx = cc1;
  c2.cxx = cc2;
  VCHECK(jit::Engine(c1).key(base) != jit::Engine(c2).key(base));   // the compiler's identity (the same wrapper, another version string)
  VCHECK(jit::Engine(c1).key(base) != k0);
  // (the cache directory and the process are not in the key: a cache can be shared)
  jit::Config other = cfg;
  other.cache_dir = d.str("elsewhere");
  VCHECK_EQ(jit::Engine(other).key(base), k0);
}

// ---- generated code against the model --------------------------------------------------------------------------------------------------------------

VTEST(jit_generated_regions_count_and_call_the_escape_as_the_run_loop_does) {
  NEED_COMPILER();
  TmpDir d;
  jit::Config cfg;
  cfg.cache_dir = d.str("c");
  jit::Engine e(cfg);
  const jit::UnitSpec spec = make_spec(4, 24, 7);
  const jit::Result r = must_obtain(e, spec);
  VCHECK(r.compiled);
  VCHECK(!r.hit);
  check_against_model(spec, *r.lib, 11, 300);
}

VTEST(jit_exceptions_unwind_through_the_generated_frames_with_and_without_fexceptions) {
  NEED_COMPILER();
  for (const char* flags : {"", "-fno-exceptions"}) {
    TmpDir d;
    jit::Config cfg;
    cfg.cache_dir = d.str("c");
    cfg.extra_flags = flags;
    jit::Engine e(cfg);
    const jit::UnitSpec spec = make_spec(1, 20, 5);
    const jit::Result r = must_obtain(e, spec);
    const uint32_t p[8] = {0xaaaaaaaau, 0x55555555u, 0xffffffffu, 0, 1, 2, 3, 0xffffffffu};
    const bool up[8] = {true, false, true, false, true, false, true, true};
    // Throw at the 6th escape: the exception reaches the caller, the ops before it and the one that threw are counted, none after.
    const Run full = run_region(r.lib->fns[0], 0xffffffffu, p, up, true);
    VCHECK(!full.threw);
    VCHECK_EQ(full.calls.size(), size_t{20});   // (stats on: every op reaches the escape)
    const Run cut = run_region(r.lib->fns[0], 0xffffffffu, p, up, true, 5);
    VCHECK(cut.threw);
    VCHECK_EQ(cut.what, std::string("boom at 5"));
    VCHECK_EQ(cut.calls.size(), size_t{6});
    VCHECK_EQ(cut.counters.steps, uint64_t{6});
    VCHECK_EQ(cut.counters.instructions, uint64_t{6});
    // The same counters the model has for the first six ops.
    jit::RegionSpec head;
    head.ops.assign(spec.regions[0].ops.begin(), spec.regions[0].ops.begin() + 6);
    Counters mc;
    Calls mcalls;
    model_run(head, 0xffffffffu, p, up, true, mc, mcalls);
    VCHECK(cut.counters == mc);
  }
}

// ---- the cache ----------------------------------------------------------------------------------------------------------------------------------

VTEST(jit_second_engine_hits_the_cache_without_compiling) {
  NEED_COMPILER();
  TmpDir d;
  jit::Config cfg;
  cfg.cache_dir = d.str("c");
  const jit::UnitSpec spec = make_spec(3, 10, 9);
  {
    jit::Engine e(cfg);
    const jit::Result r = must_obtain(e, spec);
    VCHECK(r.compiled);
    VCHECK_EQ(e.compiles(), uint64_t{1});
  }
  jit::Engine e2(cfg);
  const jit::Result l = e2.lookup(spec);   // the cache alone finds it
  VCHECK(l.found && l.hit && !l.compiled);
  const jit::Result r = must_obtain(e2, spec);
  VCHECK(r.hit && !r.compiled);
  VCHECK_EQ(e2.compiles(), uint64_t{0});
  check_against_model(spec, *r.lib, 13, 50);
  // A spec that differs finds nothing and does not trip over the other's object.
  VCHECK(!e2.lookup(make_spec(3, 10, 10)).found);
}

VTEST(jit_damaged_objects_are_replaced_never_loaded) {
  NEED_COMPILER();
  const jit::UnitSpec spec = make_spec(2, 12, 21);
  const jit::UnitSpec other = make_spec(2, 12, 22);
  enum Damage { Flip, Truncate, Empty, GarbageSum, MissingSum, OtherObject, Zeroed };
  for (Damage dmg : {Flip, Truncate, Empty, GarbageSum, MissingSum, OtherObject, Zeroed}) {
    TmpDir d;
    jit::Config cfg;
    cfg.cache_dir = d.str("c");
    std::string so_other, sum_other;
    // (The objects are built in a child process: dlopen returns the library already mapped under a path without looking at the file again, so
    // an object loaded here would hide the damage done to it.)
    if (const pid_t pid = ::fork(); pid == 0) {
      jit::Engine e(cfg);
      bool ok = e.obtain(spec).found;
      if (dmg == OtherObject) ok = e.obtain(other).found && ok;
      ::_exit(ok ? 0 : 1);
    } else {
      int st = 0;
      ::waitpid(pid, &st, 0);
      VCHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0);
    }
    const std::string key = jit::Engine(cfg).key(spec);
    if (dmg == OtherObject) {   // a perfectly good object, with a perfectly good checksum, made for other words
      const std::string ok = jit::Engine(cfg).key(other);
      std::ifstream a(d.str("c") + "/" + ok + ".so", std::ios::binary), b(d.str("c") + "/" + ok + ".sum");
      so_other.assign(std::istreambuf_iterator<char>(a), {});
      sum_other.assign(std::istreambuf_iterator<char>(b), {});
    }
    const std::string so = d.str("c") + "/" + key + ".so", sum = d.str("c") + "/" + key + ".sum";
    std::string bytes;
    {
      std::ifstream a(so, std::ios::binary);
      bytes.assign(std::istreambuf_iterator<char>(a), {});
    }
    const auto put = [](const std::string& path, const std::string& data) { std::ofstream(path, std::ios::binary | std::ios::trunc) << data; };
    switch (dmg) {
      case Flip: bytes[bytes.size() / 2] ^= 0x40; put(so, bytes); break;
      case Truncate: put(so, bytes.substr(0, bytes.size() / 2)); break;
      case Empty: put(so, ""); break;
      case GarbageSum: put(sum, "this is not a checksum\n"); break;
      case MissingSum: fs::remove(sum); break;
      case OtherObject: put(so, so_other); put(sum, sum_other); break;
      case Zeroed: put(so, std::string(bytes.size(), '\0')); break;
    }
    jit::Engine e(cfg);
    const jit::Result l = e.lookup(spec);
    VCHECK(!l.found);   // never loaded
    const jit::Result r = must_obtain(e, spec);
    VCHECK(r.compiled);   // rebuilt
    VCHECK(!r.hit);
    if (dmg != MissingSum) VCHECK(r.replaced_corrupt);
    check_against_model(spec, *r.lib, 17, 20);
    // And now the cache is good again.
    jit::Engine e3(cfg);
    VCHECK(e3.lookup(spec).found);
  }
}

VTEST(jit_an_object_of_another_abi_or_generator_is_rejected) {
  NEED_COMPILER();
  const jit::UnitSpec spec = make_spec(1, 6, 31);
  struct Case { const char* what; std::string from, to; };
  const std::string abi_word = [&] {
    char b[32];
    std::snprintf(b, sizeof b, "0x%08xu,", jit::kAbiVersion);
    return std::string(b);
  }();
  for (int variant = 0; variant < 3; ++variant) {
    TmpDir d;
    jit::Config cfg;
    cfg.cache_dir = d.str("c");
    jit::Engine e(cfg);
    const std::string key = e.key(spec);
    std::string src = jit::Engine::source(spec);
    // The metadata array begins: magic, ABI, generator, sm, regions. Change the second or third word.
    const size_t at = src.find("vgpu_jit_meta[] = {");
    VCHECK(at != std::string::npos);
    size_t w = src.find("0x", at);
    w = src.find("0x", w + 2);   // the ABI word
    if (variant == 1) w = src.find("0x", w + 2);   // the generator word
    if (variant == 2) { w = src.find("0x", w + 2); w = src.find("0x", w + 2); }   // the architecture
    src.replace(w, 10, "0x00000063");
    fs::create_directories(d.p / "c");
    std::ofstream(d.str("w.cpp")) << src;
    const std::string cmd = "c++ -std=c++20 -O1 -fPIC -shared -w -o " + d.str("w.so") + " " + d.str("w.cpp");
    VCHECK_EQ(std::system(cmd.c_str()), 0);
    std::string bytes;
    {
      std::ifstream a(d.str("w.so"), std::ios::binary);
      bytes.assign(std::istreambuf_iterator<char>(a), {});
    }
    fs::create_directories(d.p / "c");
    ::chmod(d.str("c").c_str(), 0700);
    std::ofstream(d.str("c") + "/" + key + ".so", std::ios::binary) << bytes;
    std::ofstream(d.str("c") + "/" + key + ".sum") << jit::checksum(bytes) << " " << bytes.size() << "\n";
    const jit::Result l = e.lookup(spec);
    VCHECK(!l.found);
    const jit::Result r = must_obtain(e, spec);
    VCHECK(r.compiled && r.replaced_corrupt);
    check_against_model(spec, *r.lib, 19, 10);
  }
  // A layout that differs fails the compile: the generated static_asserts name the host's offsets.
  TmpDir d;
  std::string src = jit::Engine::source(spec);
  const std::string want = "sizeof(Ctx) == " + std::to_string(sizeof(jit::Ctx));
  const size_t at = src.find(want);
  VCHECK(at != std::string::npos);
  src.replace(at, want.size(), "sizeof(Ctx) == " + std::to_string(sizeof(jit::Ctx) + 8));
  std::ofstream(d.str("w.cpp")) << src;
  VCHECK(std::system(("c++ -std=c++20 -fPIC -shared -w -o " + d.str("w.so") + " " + d.str("w.cpp") + " 2>/dev/null").c_str()) != 0);
}

VTEST(jit_a_selection_of_regions_is_compiled_and_found_again_by_the_section) {
  NEED_COMPILER();
  TmpDir d;
  jit::Config cfg;
  cfg.cache_dir = d.str("c");
  const jit::UnitSpec full = make_spec(5, 10, 81);
  const std::vector<uint32_t> sel{1, 3, 4};
  {
    jit::Engine e(cfg);
    const jit::Result r = e.obtain_selected(full, sel);
    VCHECK(r.found && r.compiled);
    VCHECK_EQ(r.lib->fns.size(), size_t{3});
    check_against_model(jit::Engine::subspec(full, sel), *r.lib, 83, 20);
  }
  jit::Engine e2(cfg);
  std::vector<uint32_t> got;
  const jit::Result l = e2.lookup_selected(full, &got);
  VCHECK(l.found && l.hit && !l.compiled);
  VCHECK(got == sel);
  // Another section (one word changed) has no selection recorded.
  jit::UnitSpec other = full;
  other.regions[2].ops[0].lo ^= 1;
  VCHECK(!e2.lookup_selected(other, nullptr).found);
  // A damaged selection file is a miss, not a crash or a wrong load.
  const std::string skey = e2.section_key(full);
  for (const char* junk : {"", "VJSEL1 0\n", "VJSEL1 2 4 1\n", "VJSEL1 9 0 1 2 3 4 5 6 7 8\n", "garbage", "VJSEL1 2 1 99\n"}) {
    std::ofstream(d.str("c") + "/" + skey + ".sel", std::ios::trunc) << junk;
    VCHECK(!e2.lookup_selected(full, nullptr).found);
  }
}

// ---- failure modes: every one is a Status, none a crash ---------------------------------------------------------------------------------------

VTEST(jit_no_compiler) {
  if (VJ_SANITIZED) return;
  TmpDir d;
  const jit::UnitSpec spec = make_spec(1, 4, 41);
  {
    jit::Config cfg;
    cfg.cache_dir = d.str("c");
    cfg.cxx = "/nonexistent/c++";
    jit::Engine e(cfg);
    const jit::Result r = e.obtain(spec);
    VCHECK(!r.found);
    VCHECK(r.status == jit::Status::NoCompiler);
    VCHECK_CONTAINS(r.message, "VGPU_JIT_CXX");
    VCHECK(e.key(spec).empty());
  }
  {
    jit::Config cfg;
    cfg.cache_dir = d.str("c2");
    cfg.path = d.str("empty");   // none of c++, g++, clang++ there
    fs::create_directories(d.p / "empty");
    jit::Engine e(cfg);
    const jit::Result r = e.lookup(spec);
    VCHECK(r.status == jit::Status::NoCompiler);
    VCHECK_CONTAINS(r.message, "no C++ compiler");
  }
  {   // a "compiler" that cannot run
    jit::Config cfg;
    cfg.cache_dir = d.str("c3");
    cfg.cxx = write_script(d.p / "broken", "exit 3\n");
    jit::Engine e(cfg);
    VCHECK(e.obtain(spec).status == jit::Status::NoCompiler);
  }
}

VTEST(jit_a_failing_compile_leaves_nothing_in_the_cache) {
  if (VJ_SANITIZED) return;
  TmpDir d;
  jit::Config cfg;
  cfg.cache_dir = d.str("c");
  cfg.cxx = write_script(d.p / "cc", "if [ \"$1\" = --version ]; then echo failcc; exit 0; fi\nif [ \"$1\" = -dumpmachine ]; then echo m; exit 0; fi\n"
                                     "echo 'error: this compiler hates you' >&2\nexit 1\n");
  jit::Engine e(cfg);
  const jit::Result r = e.obtain(make_spec(1, 4, 43));
  VCHECK(!r.found);
  VCHECK(r.status == jit::Status::CompileFailed);
  VCHECK_CONTAINS(r.message, "hates you");
  VCHECK_EQ(count_files(d.p / "c", ".so"), size_t{0});
  VCHECK_EQ(count_files(d.p / "c", ".cpp"), size_t{0});
  VCHECK_EQ(count_files(d.p / "c", ".sum"), size_t{0});
}

VTEST(jit_a_compile_that_takes_too_long_is_killed) {
  if (VJ_SANITIZED) return;
  TmpDir d;
  jit::Config cfg;
  cfg.cache_dir = d.str("c");
  cfg.timeout_s = 1;
  cfg.cxx = write_script(d.p / "cc", "if [ \"$1\" = --version ]; then echo slowcc; exit 0; fi\nif [ \"$1\" = -dumpmachine ]; then echo m; exit 0; fi\n"
                                     "sleep 60\n");
  jit::Engine e(cfg);
  const auto t0 = std::chrono::steady_clock::now();
  const jit::Result r = e.obtain(make_spec(1, 4, 45));
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  VCHECK(r.status == jit::Status::CompileTimeout);
  VCHECK(s < 10.0);
  VCHECK_EQ(count_files(d.p / "c", ".so"), size_t{0});
}

VTEST(jit_the_compiler_runs_under_an_address_space_limit) {
  if (VJ_SANITIZED) return;
  TmpDir d;
  jit::Config cfg;
  cfg.cache_dir = d.str("c");
  cfg.mem_mb = 777;
  cfg.cxx = write_script(d.p / "cc", "if [ \"$1\" = --version ]; then echo limcc; exit 0; fi\nif [ \"$1\" = -dumpmachine ]; then echo m; exit 0; fi\n"
                                     "echo \"limit=$(ulimit -v)\" >&2\nexit 1\n");
  jit::Engine e(cfg);
  const jit::Result r = e.obtain(make_spec(1, 4, 47));
  VCHECK_CONTAINS(r.message, "limit=" + std::to_string(777 * 1024));
}

VTEST(jit_an_unusable_cache_directory_is_reported) {
  NEED_COMPILER();
  TmpDir d;
  const jit::UnitSpec spec = make_spec(1, 4, 49);
  {   // below a file
    std::ofstream(d.str("file")) << "x";
    jit::Config cfg;
    cfg.cache_dir = d.str("file/sub");
    jit::Engine e(cfg);
    const jit::Result r = e.obtain(spec);
    VCHECK(r.status == jit::Status::CacheUnusable);
  }
  {   // writable by everyone: someone else could plant an object
    fs::create_directories(d.p / "open");
    ::chmod(d.str("open").c_str(), 0777);
    jit::Config cfg;
    cfg.cache_dir = d.str("open");
    jit::Engine e(cfg);
    const jit::Result r = e.obtain(spec);
    VCHECK(r.status == jit::Status::CacheUnusable);
    VCHECK_CONTAINS(r.message, "writable by others");
  }
  {   // created by us, 0700
    jit::Config cfg;
    cfg.cache_dir = d.str("new/deep/dir");
    jit::Engine e(cfg);
    must_obtain(e, spec);
    struct stat st;
    VCHECK_EQ(::stat(d.str("new/deep/dir").c_str(), &st), 0);
    VCHECK_EQ(static_cast<unsigned>(st.st_mode & 0777), 0700u);
  }
}

VTEST(jit_an_object_that_cannot_be_mapped_is_a_cache_problem_not_a_crash) {
  NEED_COMPILER();
  TmpDir d;
  jit::Config cfg;
  cfg.cache_dir = d.str("c");
  // (What a noexec mount does: dlopen fails to map the segment. The container check of this is in docs/perf-design/round11-jit.md.)
  cfg.dlopen_fn = [](const char*, std::string* err) -> void* {
    *err = "failed to map segment from shared object";
    return nullptr;
  };
  jit::Engine e(cfg);
  const jit::Result r = e.obtain(make_spec(1, 4, 51));
  VCHECK(!r.found);
  VCHECK(r.status == jit::Status::CacheUnusable);
  VCHECK_CONTAINS(r.message, "noexec");
  VCHECK_EQ(count_files(d.p / "c", ".so"), size_t{0});
}

// ---- concurrency --------------------------------------------------------------------------------------------------------------------------------

VTEST(jit_threads_compiling_one_key_compile_it_once) {
  NEED_COMPILER();
  TmpDir d;
  jit::Config cfg;
  cfg.cache_dir = d.str("c");
  const jit::UnitSpec spec = make_spec(2, 8, 61);
  {   // one engine, eight threads
    jit::Engine e(cfg);
    std::vector<std::thread> ts;
    std::atomic<int> ok{0};
    for (int i = 0; i < 8; ++i)
      ts.emplace_back([&] {
        const jit::Result r = e.obtain(spec);
        if (r.found && r.lib->fns.size() == 2) ++ok;
      });
    for (auto& t : ts) t.join();
    VCHECK_EQ(ok.load(), 8);
    VCHECK_EQ(e.compiles(), uint64_t{1});
  }
  {   // eight engines (as eight processes would be), one directory
    TmpDir d2;
    jit::Config c2;
    c2.cache_dir = d2.str("c");
    std::vector<std::unique_ptr<jit::Engine>> es;
    for (int i = 0; i < 8; ++i) es.push_back(std::make_unique<jit::Engine>(c2));
    std::vector<std::thread> ts;
    std::atomic<int> ok{0};
    for (int i = 0; i < 8; ++i)
      ts.emplace_back([&, i] {
        const jit::Result r = es[static_cast<size_t>(i)]->obtain(spec);
        if (r.found && r.lib->fns.size() == 2) ++ok;
      });
    for (auto& t : ts) t.join();
    VCHECK_EQ(ok.load(), 8);
    uint64_t total = 0;
    for (auto& e : es) total += e->compiles();
    VCHECK_EQ(total, uint64_t{1});
  }
}

VTEST(jit_processes_compiling_one_key_compile_it_once) {
  NEED_COMPILER();
  TmpDir d;
  jit::Config cfg;
  cfg.cache_dir = d.str("c");
  const jit::UnitSpec spec = make_spec(2, 8, 63);
  std::vector<pid_t> kids;
  for (int i = 0; i < 6; ++i) {
    const pid_t pid = ::fork();
    if (pid == 0) {
      jit::Engine e(cfg);
      const jit::Result r = e.obtain(spec);
      ::_exit(!r.found ? 2 : r.compiled ? 1 : 0);   // 0: from the cache, 1: this one compiled
    }
    kids.push_back(pid);
  }
  int compiled = 0, hit = 0, bad = 0;
  for (pid_t k : kids) {
    int st = 0;
    ::waitpid(k, &st, 0);
    const int code = WIFEXITED(st) ? WEXITSTATUS(st) : 99;
    (code == 1 ? compiled : code == 0 ? hit : bad)++;
  }
  VCHECK_EQ(bad, 0);
  VCHECK_EQ(compiled, 1);
  VCHECK_EQ(hit, 5);
}

// ---- admission: what the run loop checks per op, checked once for a region ---------------------------------------------------------------------

VTEST(jit_a_region_runs_in_one_go_only_where_every_op_would_have_run_in_the_loop) {
  // n: ops already run in this pass; the loop runs an op while n < 256, so a region of k ops fits when n + k <= 256.
  VCHECK(jit::kRunLimit == 256u);
  VCHECK(jit::admissible(0, 256, 0, 1000, 10, 10));
  VCHECK(!jit::admissible(0, 257, 0, 1000, 10, 10));
  VCHECK(jit::admissible(160, 96, 0, 1000, 10, 10));
  VCHECK(!jit::admissible(161, 96, 0, 1000, 10, 10));
  VCHECK(jit::admissible(255, 1, 0, 1000, 10, 10));
  VCHECK(!jit::admissible(255, 2, 0, 1000, 10, 10));
  VCHECK(!jit::admissible(256, 2, 0, 1000, 10, 10));
  // steps: each op takes one, one past the budget is the error, so steps + k <= max_steps.
  VCHECK(jit::admissible(0, 50, 950, 1000, 10, 10));
  VCHECK(!jit::admissible(0, 50, 951, 1000, 10, 10));
  VCHECK(!jit::admissible(0, 2, 1000, 1000, 10, 10));
  // registers: the file holds what the region touches.
  VCHECK(jit::admissible(0, 4, 0, 1000, 32, 32));
  VCHECK(!jit::admissible(0, 4, 0, 1000, 33, 32));
  VCHECK(jit::admissible(0, 4, 0, 1000, 0, 0));
}

// ---- discovery and promotion (the executor's side of the machinery, without an executor) -------------------------------------------------------------

namespace {
void dummy_handler(void*, const tier::MOp&, uint32_t) {}

struct FakeProgram {
  std::vector<Instr> instrs;
  tier::Program prog;
  // kinds: 'a' a handler, straight; 's' a store (STG); 'n' a @!PT op with no handler; 'x' no handler, not @!PT; 'b' a BRA to `target`
  FakeProgram(const std::string& kinds, std::vector<std::pair<size_t, size_t>> branches = {}) {
    instrs.resize(kinds.size());
    prog.ops.resize(kinds.size());
    for (size_t i = 0; i < kinds.size(); ++i) {
      Instr& in = instrs[i];
      in.w.lo = 0x1111000000000000ull + i;
      in.w.hi = 0x2222;
      in.op = kinds[i] == 's' ? Op::STG : kinds[i] == 'b' ? Op::BRA : Op::IADD3;
      tier::MOp& op = prog.ops[i];
      op.ins = &instrs[i];
      op.straight = kinds[i] != 'b';
      op.fn = (kinds[i] == 'a' || kinds[i] == 's') ? &dummy_handler : nullptr;
      op.form = kinds[i] == 'b' ? tier::Form::Bra : kinds[i] == 's' ? tier::Form::Stg : tier::Form::None;   // (None: the generated code calls the handler, which these tests give the fake context for)
      op.never = kinds[i] == 'n';
      op.unguarded = kinds[i] != 'n';
      op.maxreg = static_cast<uint16_t>(4 + i);
    }
    for (auto [at, target] : branches) prog.ops[at].imm = static_cast<int64_t>(target * 16);
  }
};

std::vector<std::pair<uint32_t, uint32_t>> shape(const jit::Unit& u) {
  std::vector<std::pair<uint32_t, uint32_t>> v;
  for (const jit::Region& r : u.regions) v.emplace_back(r.first, r.k);
  return v;
}
}  // namespace

VTEST(jit_regions_are_straight_runs_of_ops_with_a_handler_and_no_store) {
  NEED_COMPILER();
  if (!jit::requested()) return;
  //                      0123456789...
  FakeProgram fp("aasaaaaannxaabaa", {{13, 5}});
  // 0,1 a a -> region; 2 is a store; 3,4 -> region; 5..7 a,a,a start a region (5 is a BRA target) and run on through the @!PT op 8; 9 (n, second)
  // is a @!PT op too, so the region is 5..9; 10 has no handler; 11,12 -> region; 13 is the BRA; 14 is the second to last; 15 is the last op of
  // the section: never in a region.
  const auto u = jit::make_unit(fp.prog, 80);
  VCHECK(u != nullptr);
  const std::vector<std::pair<uint32_t, uint32_t>> want{{0, 2}, {3, 2}, {5, 5}, {11, 2}};
  VCHECK(shape(*u) == want);
  VCHECK_EQ(u->region_at[0], 1u);
  VCHECK_EQ(u->region_at[1], 0u);   // (only the first op of a region has an entry)
  VCHECK_EQ(u->region_at[5], 3u);
  VCHECK_EQ(u->regions[2].maxreg, 4u + 7u);   // the @!PT ops have no handler and so no register demand: op 7 is the highest that has one
  VCHECK_EQ(u->spec.regions[2].ops.size(), size_t{5});
  VCHECK_EQ(u->spec.regions[0].ops[0].lo, uint64_t{0x1111000000000000ull});
  VCHECK_EQ(u->spec.regions[0].ops[1].lo, uint64_t{0x1111000000000001ull});
}

VTEST(jit_long_runs_are_split_and_the_last_op_of_a_section_is_never_in_a_region) {
  NEED_COMPILER();
  if (!jit::requested()) return;
  FakeProgram fp(std::string(300, 'a'));
  const auto u = jit::make_unit(fp.prog, 80);
  VCHECK(u != nullptr);
  const std::vector<std::pair<uint32_t, uint32_t>> want{{0, 96}, {96, 96}, {192, 96}, {288, 11}};
  VCHECK(shape(*u) == want);
  VCHECK(jit::make_unit(FakeProgram("ab").prog, 80) == nullptr);   // nothing of two ops or more: no unit
  VCHECK(jit::make_unit(FakeProgram("aaa").prog, 80) != nullptr);  // two ops and a last one
}

VTEST(jit_a_unit_is_promoted_after_the_threshold_and_a_second_one_with_the_same_words_hits_the_cache) {
  NEED_COMPILER();
  if (!jit::requested()) return;
  const uint64_t compiles0 = jit::stats().compiles.load(), hits0 = jit::stats().cache_hits.load();
  FakeProgram a("aaaaxaaaaaxa");
  auto u = jit::make_unit(a.prog, 80);
  VCHECK(u != nullptr);
  VCHECK_EQ(u->regions.size(), size_t{2});
  VCHECK(u->state.load() == static_cast<int>(jit::State::Idle));   // (nothing asked of the cache before the regions have run)
  VCHECK(!u->ready());
  u->note(0, 40);
  VCHECK(!u->ready());   // 40 of 100
  u->note(1, 40);
  VCHECK(!u->ready());   // 80
  u->note(0, 40);
  VCHECK(u->ready());    // 120: the cache asked (a miss, synchronous mode), then compiled and loaded
  VCHECK_EQ(jit::stats().compiles.load(), compiles0 + 1);
  check_against_model(u->spec, *u->lib, 23, 30);
  u->note(0, 1000);   // (counting is over: nothing happens)
  VCHECK_EQ(jit::stats().compiles.load(), compiles0 + 1);
  // The same words in another program: ready at once, from the cache.
  FakeProgram b("aaaaxaaaaaxa");
  const auto u2 = jit::make_unit(b.prog, 80);
  VCHECK(u2 != nullptr);
  VCHECK(!u2->ready());
  u2->note(0, 100);   // the lookup threshold (the smaller of 1 M and the compile threshold) reached: the cache has it
  VCHECK(u2->ready());
  VCHECK_EQ(jit::stats().compiles.load(), compiles0 + 1);
  VCHECK_EQ(jit::stats().cache_hits.load(), hits0 + 1);
  // Other words (here: another architecture) are another unit.
  FakeProgram c("aaaaxaaaaaxa");
  const auto u3 = jit::make_unit(c.prog, 90);
  VCHECK(u3 != nullptr && !u3->ready());
  u3->note(0, 100);
  VCHECK(u3->state.load() == static_cast<int>(jit::State::Counting) || u3->ready());   // (a miss: counting on; or, the threshold being reached by the same note, compiled already)
}

// ---- cost (VGPU_JIT_REPORT=1 prints it) ---------------------------------------------------------------------------------------------------------

VTEST(jit_cost_of_compiling_and_of_a_cache_hit) {
  NEED_COMPILER();
  if (!std::getenv("VGPU_JIT_REPORT")) return;
  for (size_t regions : {size_t{1}, size_t{10}, size_t{50}}) {
    TmpDir d;
    jit::Config cfg;
    cfg.cache_dir = d.str("c");
    jit::Engine e(cfg);
    const jit::UnitSpec spec = make_spec(regions, 60, 71);
    const jit::Result r = must_obtain(e, spec);
    double hit = 0;
    for (int i = 0; i < 20; ++i) {
      jit::Engine e2(cfg);
      const jit::Result h = e2.lookup(spec);
      hit += h.lookup_ms;
    }
    // (a lookup in an Engine that has not asked the compiler its identity yet includes that: the version query, twice)
    jit::Result warm;
    double warm_ms = 0;
    for (int i = 0; i < 20; ++i) {
      warm = e.lookup(spec);
      warm_ms += warm.lookup_ms;
    }
    std::printf("  jit cost: %zu regions of 60 ops: compile %.1f ms (%.1f ms per region), lookup in a new engine %.2f ms, lookup in a warm engine %.3f ms\n",
                regions, r.compile_ms, r.compile_ms / double(regions), hit / 20, warm_ms / 20);
  }
}

VTEST_MAIN
