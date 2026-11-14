// Tier 2, round sp64: the compile of a section and the run around it (src/sass/jit.hpp, jit.cpp).
//  * A section whose compile is running (State::Queued) runs on the plain loop: it counts nothing and has no functions yet, and the hook loop cost
//    it a few per cent for as long as the compile took.
//  * The compile log of VGPU_JIT_STATS: one `jit-compile:` line a compile, after the summary (why, regions, ops, source bytes, when it was asked for,
//    started and done, the outcome). Run in a child process of the test itself (the statistics are printed when the process ends).
//  * The precompiled runtime header (GCC): made after the first compile of a process that found none, read by every compile after it, in that process
//    and the next; a damaged one does not fail a compile; VGPU_JIT_PCH=0 makes and reads none.
//
// Mutants: mutated copies of jit.hpp / jit.cpp compiled outside the tree (docs/perf-design/data/sp64/scripts/mutants.sh); each makes a test here fail.
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "sass/jit.hpp"
#include "sass/tier.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

namespace fs = std::filesystem;
using namespace vgpu::sass;

namespace {

std::string slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), {});
}

struct EnvInit {
  fs::path dir;
  EnvInit() {
    const char* t = std::getenv("TMPDIR");
    dir = fs::path(t && *t ? t : "/tmp") / ("vgpu-test-jit-threads." + std::to_string(::getpid()));
    fs::create_directories(dir);
    ::setenv("VGPU_JIT", "1", 1);
    ::setenv("VGPU_JIT_SYNC", "1", 1);
    ::setenv("VGPU_JIT_HOT_PERMILLE", "0", 1);   // (no share rule: the lookup is the first thing)
    ::setenv("VGPU_JIT_LOOKUP_AFTER", "128", 1);
    ::setenv("VGPU_JIT_PROFILE", "0", 1);
    ::setenv("VGPU_JIT_THRESHOLD", "1024", 1);
    ::setenv("VGPU_JIT_MIN_REGION", "2", 1);
    ::setenv("VGPU_JIT_CACHE", (dir / "cache").c_str(), 1);
    ::setenv("VGPU_QUIET", "1", 1);
    ::unsetenv("VGPU_TIER");
  }
  ~EnvInit() {
    if (std::getenv("VJT_CHILD")) return;   // (the parent removes it)
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
} g_env;

void dummy_handler(void*, const tier::MOp&, uint32_t) {}

// A section of `kinds`: 'g' an op the generated code computes (a MOV), 'x' an op with no handler (it ends a region); `salt` makes its words its own.
struct FakeProgram {
  std::vector<Instr> instrs;
  tier::Program prog;
  FakeProgram(const std::string& kinds, uint64_t salt) {
    instrs.resize(kinds.size());
    prog.ops.resize(kinds.size());
    prog.pool.resize(1);
    for (size_t i = 0; i < kinds.size(); ++i) {
      Instr& in = instrs[i];
      in.w.lo = 0x7777000000000000ull + (salt << 16) + i;
      in.w.hi = 0x8888;
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

bool have_compiler() {
  jit::Engine e(jit::Config::from_env());
  return !e.compiler_id().empty();
}

std::string self_exe() {
  char buf[4096];
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof buf - 1);
  return n > 0 ? std::string(buf, static_cast<size_t>(n)) : std::string();
}

// ---- the child process ------------------------------------------------------------------------------------------------------------------------
// VJT_CHILD=log: two sections, one compiled at its threshold, then the same section again from the cache (a second unit of the same code), with
// VGPU_JIT_STATS on and the statistics appended to VJT_STATS. VJT_CHILD=queued: a section's compile (a fake compiler that sleeps, VJT_CXX) runs
// on the worker; the child prints the section's state and whether it is hooked while it does ("queued=<state> hook=<0|1>").
void child(const char* what) {
  ::setenv("VGPU_JIT_CACHE", std::getenv("VJT_CACHE"), 1);
  if (std::strcmp(what, "pch") == 0) {   // VJT_CHILD=pch: VJT_N sections (salts 20, 21, ...) compiled at their thresholds, one after the other
    ::setenv("VGPU_JIT_STATS", "1", 1);
    ::setenv("VGPU_JIT_STATS_FILE", std::getenv("VJT_STATS"), 1);
    const int n = std::atoi(std::getenv("VJT_N"));
    static std::vector<std::unique_ptr<FakeProgram>> keep;
    for (int i = 0; i < n; ++i) {
      keep.push_back(std::make_unique<FakeProgram>("ggggggggx", 20 + i + 10 * std::atoi(std::getenv("VJT_ROUND"))));
      const auto u = jit::make_unit(keep.back()->prog, 80);
      if (!u) std::exit(3);
      u->note(0, 128);
      u->note(0, 1024);
      if (!u->ready()) std::exit(4);
    }
    std::exit(0);
  }
  if (std::strcmp(what, "log") == 0) {
    ::setenv("VGPU_JIT_STATS", "1", 1);
    ::setenv("VGPU_JIT_STATS_FILE", std::getenv("VJT_STATS"), 1);
    static FakeProgram fp("ggggggggx", 7);
    const auto u = jit::make_unit(fp.prog, 80);
    if (!u) std::exit(3);
    u->note(0, 128);    // the lookup: a miss
    u->note(0, 1024);   // the threshold: compiled (synchronously)
    if (!u->ready()) std::exit(4);
    const auto v = jit::make_unit(fp.prog, 80);   // the same words again: the lookup finds the object
    v->note(0, 128);
    if (!v->ready()) std::exit(5);
    std::exit(0);   // (the statistics are printed by the manager's destructor)
  }
  ::setenv("VGPU_JIT_CXX", std::getenv("VJT_CXX"), 1);
  ::setenv("VGPU_JIT_SYNC", "0", 1);
  static FakeProgram fp("ggggggggx", 8);
  const auto u = jit::make_unit(fp.prog, 80);
  if (!u) std::exit(3);
  u->note(0, 128);
  jit::drain();
  u->note(0, 1024);   // queued; the worker starts the compiler, which sleeps
  for (int i = 0; i < 200 && !fs::exists(std::string(std::getenv("VJT_CACHE")) + "/started"); ++i) ::usleep(10000);
  std::printf("queued=%d hook=%d\n", u->state.load(), u->hook() ? 1 : 0);
  std::fflush(stdout);
  std::exit(0);   // (the manager's destructor cancels the compile)
}
struct ChildInit {
  ChildInit() {
    if (const char* w = std::getenv("VJT_CHILD"); w && *w) child(w);
  }
} g_child;

struct ChildOut {
  int rc = -1;
  std::string out;
};
ChildOut run_child(const std::string& env) {
  const fs::path o = g_env.dir / "child.out";
  const std::string cmd = env + " '" + self_exe() + "' > '" + o.string() + "' 2>&1";
  ChildOut r;
  const int st = std::system(cmd.c_str());
  r.rc = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
  r.out = slurp(o.string());
  return r;
}

}  // namespace

VTEST(jit_a_section_whose_compile_runs_is_on_the_plain_loop) {
  if (!jit::requested()) return;   // (a sanitizer build: the JIT is off)
  FakeProgram fp("ggggggggx", 1);
  const auto u = jit::make_unit(fp.prog, 80);
  VCHECK(u != nullptr);
  const auto at = [&](jit::State s) {
    u->state.store(static_cast<int>(s));
    return u->hook();
  };
  VCHECK(at(jit::State::Idle));
  VCHECK(at(jit::State::Looking));
  VCHECK(at(jit::State::Counting));
  VCHECK(!at(jit::State::Queued));   // round sp64: the compile is running, nothing is counted
  VCHECK(at(jit::State::Ready));
  VCHECK(!at(jit::State::Failed));
  VCHECK(!at(jit::State::Declined));
  VCHECK(!at(jit::State::Waiting));
}

VTEST(jit_a_section_whose_compile_runs_in_the_background_is_not_hooked) {
  if (!jit::requested()) return;
  if (self_exe().empty()) return;
  const fs::path cache = g_env.dir / "q";
  fs::create_directories(cache);
  fs::permissions(cache, fs::perms::owner_all);   // (a cache directory others can write is refused)
  const fs::path cxx = g_env.dir / "sleep-c++";
  {
    std::ofstream f(cxx);
    f << "#!/bin/sh\ncase \"$1\" in --version) echo 'fake c++ 1.0';; -dumpmachine) echo x86_64-linux-gnu;; *) touch '" << cache.string()
      << "/started'; exec sleep 60;; esac\n";
  }
  fs::permissions(cxx, fs::perms::owner_all);
  const ChildOut r = run_child("VJT_CHILD=queued VJT_CACHE='" + cache.string() + "' VJT_CXX='" + cxx.string() + "'");
  if (r.out.find("hook=0") == std::string::npos) std::printf("  child said: %s\n", r.out.c_str());
  VCHECK_EQ(r.rc, 0);
  VCHECK(r.out.find("queued=" + std::to_string(static_cast<int>(jit::State::Queued)) + " hook=0") != std::string::npos);
}

VTEST(jit_the_statistics_have_a_line_for_each_compile) {
  if (!jit::requested()) return;
  if (self_exe().empty()) return;
  if (!have_compiler()) {
    std::printf("  (no C++ compiler on PATH: skipped)\n");
    return;
  }
  const fs::path cache = g_env.dir / "log", stats = g_env.dir / "log.stats";
  const ChildOut r = run_child("VJT_CHILD=log VJT_CACHE='" + cache.string() + "' VJT_STATS='" + stats.string() + "'");
  VCHECK_EQ(r.rc, 0);
  const std::string s = slurp(stats.string());
  size_t n = 0;
  for (size_t at = s.find("jit-compile: "); at != std::string::npos; at = s.find("jit-compile: ", at + 1)) ++n;
  VCHECK_EQ(n, size_t{1});   // one compile (the second section was a cache hit at its lookup, which is no compile)
  const size_t at = s.find("jit-compile: ");
  const std::string line = at == std::string::npos ? std::string() : s.substr(at, s.find('\n', at) - at);
  VCHECK(line.find("jit-compile: threshold, 1 regions, 8 ops, ") == 0);
  VCHECK(line.find(" bytes of source; asked at ") != std::string::npos);
  VCHECK(line.find("; ok") != std::string::npos);
  unsigned long long bytes = 0;
  double asked = -1, started = -1, done = -1, took = -1;
  VCHECK(std::sscanf(line.c_str(), "jit-compile: threshold, 1 regions, 8 ops, %llu bytes of source; asked at %lf ms, started %lf, done %lf (%lf ms)", &bytes, &asked, &started,
                     &done, &took) == 5);
  VCHECK(bytes > 1000);
  VCHECK(asked >= 0 && started >= asked && done >= started && took > 0);
  VCHECK(s.find("jit-compile: ") > s.rfind("jit: "));   // after the summary (what reads the file takes its last "jit: " line for it)
  if (line.empty()) std::printf("  statistics: %s\n", s.c_str());
}

namespace {
size_t count_of(const std::string& s, const std::string& what) {
  size_t n = 0;
  for (size_t at = s.find(what); at != std::string::npos; at = s.find(what, at + 1)) ++n;
  return n;
}
std::vector<fs::path> gch_files(const fs::path& cache) {
  std::vector<fs::path> out;
  if (!fs::exists(cache)) return out;
  for (const auto& e : fs::recursive_directory_iterator(cache))
    if (e.path().extension() == ".gch") out.push_back(e.path());
  return out;
}
bool gcc() {
  jit::Engine e(jit::Config::from_env());
  const std::string id = e.compiler_id();
  return !id.empty() && id.find("clang") == std::string::npos;
}
}  // namespace

VTEST(jit_the_runtime_header_is_precompiled_after_the_first_compile_and_read_by_the_compiles_after_it) {
  if (!jit::requested()) return;
  if (self_exe().empty()) return;
  if (!gcc()) {
    std::printf("  (no GCC on PATH: skipped)\n");
    return;
  }
  const fs::path cache = g_env.dir / "pch", st1 = g_env.dir / "pch1.stats", st2 = g_env.dir / "pch2.stats", st3 = g_env.dir / "pch3.stats", st4 = g_env.dir / "pch4.stats";
  const std::string common = "VJT_CHILD=pch VJT_CACHE='" + cache.string() + "' ";
  // A process with two sections: the first compile finds no header and is made without it; the header is made after it; the second compile reads it.
  ChildOut a = run_child(common + "VJT_N=2 VJT_ROUND=0 VJT_STATS='" + st1.string() + "'");
  VCHECK_EQ(a.rc, 0);
  const std::string s1 = slurp(st1.string());
  VCHECK(s1.find("jit-pch: 1 compiles read the precompiled runtime header, 1 made it") != std::string::npos);
  VCHECK_EQ(count_of(s1, "jit-compile: "), size_t{2});
  VCHECK_EQ(count_of(s1, "; ok, precompiled header"), size_t{1});
  VCHECK_EQ(gch_files(cache).size(), size_t{1});
  // The next process reads it from its first compile on and makes none.
  ChildOut b = run_child(common + "VJT_N=1 VJT_ROUND=1 VJT_STATS='" + st2.string() + "'");
  VCHECK_EQ(b.rc, 0);
  const std::string s2 = slurp(st2.string());
  VCHECK(s2.find("jit-pch: 1 compiles read the precompiled runtime header, 0 made it") != std::string::npos);
  // A damaged header does not fail a compile (GCC ignores one it cannot read, or the compile is run again without it).
  const std::vector<fs::path> g = gch_files(cache);
  if (!g.empty()) std::ofstream(g[0], std::ios::binary | std::ios::trunc) << "not a precompiled header";
  ChildOut c = run_child(common + "VJT_N=1 VJT_ROUND=2 VJT_STATS='" + st3.string() + "'");
  VCHECK_EQ(c.rc, 0);
  VCHECK_EQ(count_of(slurp(st3.string()), "jit-compile: "), size_t{1});
  // VGPU_JIT_PCH=0: none read, none made.
  const fs::path cache2 = g_env.dir / "nopch";
  ChildOut d = run_child("VGPU_JIT_PCH=0 VJT_CHILD=pch VJT_CACHE='" + cache2.string() + "' VJT_N=2 VJT_ROUND=3 VJT_STATS='" + st4.string() + "'");
  VCHECK_EQ(d.rc, 0);
  const std::string s4 = slurp(st4.string());
  VCHECK_EQ(count_of(s4, "jit-compile: "), size_t{2});
  VCHECK_EQ(count_of(s4, "jit-pch:"), size_t{0});
  VCHECK(gch_files(cache2).empty());
}

VTEST_MAIN
