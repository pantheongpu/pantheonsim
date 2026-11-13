// Tier 2, round sp63: what a cold run and a default-on JIT need (src/sass/jit.hpp, jit.cpp).
//  * The wait before the lookup: a section the share rule keeps waits for its lookup on the plain loop (State::Waiting, Unit::wait_then Idle) and is
//    counted in the hook loop again when all of its instructions make up the rest of VGPU_JIT_LOOKUP_AFTER; the wait counts towards the threshold too.
//  * History: the selection of a section is recorded before its compile, so a compile that does not finish (cancelled at the process's exit) leaves it
//    behind, and the next run's lookup compiles it at once (Result::pending, VGPU_JIT_HISTORY).
//  * The cache: the default directory is named by the GNU build id of the code; a size cap and an age rule (Engine::trim); a hit renews an object's time.
//  * The notices: one a day per cache directory and kind; none with VGPU_JIT=auto for "no compiler" and "cache unusable"; none for a compile
//    cancelled at exit. These run the test's own executable as a child process (VJC_CHILD), as a user's program would run.
//
// Mutants (-DVGPU_JIT_MUTANTS=ON, VGPU_JIT_MUTANT=<name>): prewait_never, prewait_stuck, prewait_to_counting, wait_no_credit, history_never,
// history_after, trim_no_age, trim_no_cap, trim_no_touch, cachedir_date, notice_every_run, auto_loud, cancel_notice: each makes a test here fail.
#include <fcntl.h>
#include <link.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "sass/jit.hpp"
#include "sass/tier.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

namespace fs = std::filesystem;
using namespace vgpu::sass;

namespace {

constexpr uint64_t kThreshold = 10000000, kLookup = 100000, kProfile = 100000, kSample = 1000;

std::string slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), {});
}

struct EnvInit {
  fs::path dir;
  EnvInit() {
    const char* t = std::getenv("TMPDIR");
    dir = fs::path(t && *t ? t : "/tmp") / ("vgpu-test-jit-cold." + std::to_string(::getpid()));
    fs::create_directories(dir);
    ::setenv("VGPU_JIT", "1", 1);
    ::setenv("VGPU_JIT_SYNC", "1", 1);
    ::setenv("VGPU_JIT_THRESHOLD", std::to_string(kThreshold).c_str(), 1);
    ::setenv("VGPU_JIT_LOOKUP_AFTER", std::to_string(kLookup).c_str(), 1);
    ::setenv("VGPU_JIT_PROFILE", std::to_string(kProfile).c_str(), 1);
    ::setenv("VGPU_JIT_SAMPLE", std::to_string(kSample).c_str(), 1);   // (the share rule is on, with its default bar of 40 %)
    ::setenv("VGPU_JIT_MIN_REGION", "2", 1);
    ::setenv("VGPU_JIT_CACHE", (dir / "cache").c_str(), 1);
    ::setenv("VGPU_QUIET", "1", 1);
    ::unsetenv("VGPU_TIER");
    ::unsetenv("VGPU_JIT_HISTORY");
  }
  ~EnvInit() {
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
      in.w.lo = 0x5555000000000000ull + (salt << 16) + i;
      in.w.hi = 0x6666;
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

int state_of(const jit::Unit& u) { return u.state.load(); }
constexpr int kIdle = static_cast<int>(jit::State::Idle), kCounting = static_cast<int>(jit::State::Counting), kWaiting = static_cast<int>(jit::State::Waiting),
              kDeclined = static_cast<int>(jit::State::Declined);

bool have_compiler() {
  jit::Engine e(jit::Config::from_env());
  return !e.compiler_id().empty();
}

// ---- the child process (the notices) ---------------------------------------------------------------------------------------------------------
// VJC_CHILD=nocompiler: a section asks the cache with no compiler; VJC_CHILD=cancel: a section's compile (a fake compiler that never finishes) is still
// running when the process ends. VJC_MODE is VGPU_JIT's value, VJC_CACHE the cache directory. The child prints "requested=<0|1>" on stdout.
void child(const char* what) {
  ::setenv("VGPU_JIT", std::getenv("VJC_MODE"), 1);
  ::setenv("VGPU_JIT_CACHE", std::getenv("VJC_CACHE"), 1);
  ::unsetenv("VGPU_QUIET");
  ::setenv("VGPU_JIT_HOT_PERMILLE", "0", 1);   // (no share rule: the lookup is the first thing)
  ::setenv("VGPU_JIT_LOOKUP_AFTER", "128", 1);
  ::setenv("VGPU_JIT_PROFILE", "0", 1);
  ::setenv("VGPU_JIT_THRESHOLD", "1024", 1);
  const bool cancel = std::strcmp(what, "cancel") == 0;
  if (cancel) {
    ::setenv("VGPU_JIT_CXX", std::getenv("VJC_CXX"), 1);
    ::setenv("VGPU_JIT_SYNC", "0", 1);
  } else {
    ::unsetenv("VGPU_JIT_CXX");
    ::setenv("PATH", "/nonexistent-vjc", 1);   // (no c++, g++ or clang++ on it)
  }
  std::printf("requested=%d\n", jit::requested() ? 1 : 0);
  std::fflush(stdout);
  static FakeProgram fp("ggggggggx", 99);
  const auto u = jit::make_unit(fp.prog, 80);
  if (!u) std::exit(3);
  u->note(0, 128);   // the lookup (inline, or on the worker)
  jit::drain();
  if (cancel) {
    u->note(0, 1024);   // the threshold: the compile is queued, and runs on the worker
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
  }
  std::exit(0);   // (the manager's destructor cancels a compile that is still running)
}
struct ChildInit {
  ChildInit() {
    if (const char* w = std::getenv("VJC_CHILD"); w && *w) child(w);
  }
} g_child;

std::string self_exe() {
  char buf[4096];
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof buf - 1);
  return n > 0 ? std::string(buf, static_cast<size_t>(n)) : std::string();
}

struct ChildOut {
  int rc = -1;
  std::string out, err;
};
ChildOut run_child(const std::string& what, const std::string& mode, const std::string& cache, const std::string& cxx = "") {
  const fs::path o = g_env.dir / "child.out", e = g_env.dir / "child.err";
  std::string cmd = "VJC_CHILD=" + what + " VJC_MODE=" + mode + " VJC_CACHE='" + cache + "' VJC_CXX='" + cxx + "' VGPU_JIT_MUTANT='" +
                    (std::getenv("VGPU_JIT_MUTANT") ? std::getenv("VGPU_JIT_MUTANT") : "") + "' '" + self_exe() + "' > '" + o.string() + "' 2> '" + e.string() + "'";
  ChildOut r;
  const int st = std::system(cmd.c_str());
  r.rc = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
  r.out = slurp(o.string());
  r.err = slurp(e.string());
  return r;
}
size_t count_of(const std::string& s, const std::string& what) {
  size_t n = 0;
  for (size_t at = s.find(what); at != std::string::npos; at = s.find(what, at + 1)) ++n;
  return n;
}

}  // namespace

// ---- the wait before the lookup ------------------------------------------------------------------------------------------------------------------

VTEST(jit_a_kept_section_waits_for_its_lookup_on_the_plain_loop_and_that_wait_counts_towards_the_threshold) {
  if (!jit::requested()) return;   // (a sanitizer build: the JIT is off)
  const uint64_t pre0 = jit::stats().prewaits.load(), fails0 = jit::stats().failures.load();
  FakeProgram fp("ggggggggx", 1);
  const auto u = jit::make_unit(fp.prog, 80);
  VCHECK(u != nullptr);
  VCHECK(u->sampling.load());
  u->note(0, 8192);                  // 8,192 region instructions (one check of the section's total: far from the lookup)
  VCHECK_EQ(state_of(*u), kIdle);
  u->sample(kSample);                // the sample is in: every one of its instructions is generated code, the section is kept ...
  VCHECK(!u->sampling.load());
  VCHECK_EQ(state_of(*u), kWaiting);  // ... and waits for the lookup on the plain loop
  VCHECK(!u->hook());
  VCHECK(u->waiting());
  VCHECK_EQ(jit::stats().prewaits.load(), pre0 + 1);
  VCHECK_EQ(u->wait_for.load(), kLookup - 8192);
  VCHECK_EQ(u->wait_then.load(), kIdle);
  VCHECK_EQ(u->wait_batch.load(), (kLookup - 8192) >> 6);   // (91,808 instructions: a thread brings 1,434 at a time)
  u->note(0, 5000000);               // (a thread still in the hook loop: nothing)
  VCHECK_EQ(state_of(*u), kWaiting);
  u->wait_note(kLookup - 8192 - 1);
  VCHECK_EQ(state_of(*u), kWaiting);  // one short
  u->wait_note(u->wait_batch.load());
  VCHECK_EQ(state_of(*u), kIdle);     // the hook loop again, counting up to the lookup
  VCHECK(u->hook());
  VCHECK_EQ(u->credit.load(), kLookup - 8192);
  // The next check of the total (a region's count passing a multiple of 4,096) finds the lookup's 100,000 reached with what the wait ran: the cache
  // is asked, misses, and the section waits again, this time for its threshold, with both the counted and the waited instructions done.
  u->note(0, 4096);
  if (jit::stats().failures.load() != fails0) {
    std::printf("  (no C++ compiler on PATH: the rest skipped)\n");
    return;
  }
  VCHECK_EQ(state_of(*u), kWaiting);
  VCHECK_EQ(u->wait_then.load(), kCounting);
  VCHECK_EQ(u->wait_for.load(), kThreshold - (8192 + 4096) - (kLookup - 8192) - kProfile);
}

VTEST(jit_a_section_near_its_lookup_or_declined_by_the_share_rule_does_not_wait_for_the_lookup) {
  if (!jit::requested()) return;
  const uint64_t pre0 = jit::stats().prewaits.load();
  FakeProgram fp("ggggggggx", 2);
  const auto u = jit::make_unit(fp.prog, 80);
  VCHECK(u != nullptr);
  u->note(0, kLookup - 4096 + 1);   // fewer than 4,096 instructions to the lookup: not worth a wait
  u->sample(kSample);
  VCHECK_EQ(state_of(*u), kIdle);
  VCHECK_EQ(jit::stats().prewaits.load(), pre0);
  // A section whose regions are a small share of its sample is declined, as before: the plain loop for good, no wait.
  FakeProgram fq("ggggggggx", 3);
  const auto v = jit::make_unit(fq.prog, 80);
  VCHECK(v != nullptr);
  v->note(0, 8);
  v->sample(kSample * 100);
  VCHECK_EQ(state_of(*v), kDeclined);
  VCHECK_EQ(jit::stats().prewaits.load(), pre0);
}

// ---- history: a compile that did not finish ---------------------------------------------------------------------------------------------------------

VTEST(jit_a_selection_whose_compile_did_not_finish_is_compiled_at_the_next_lookup) {
  if (!jit::requested()) return;
  if (!have_compiler()) {
    std::printf("  (no C++ compiler on PATH: skipped)\n");
    return;
  }
  FakeProgram fp("ggggggggx", 4);
  const auto u = jit::make_unit(fp.prog, 80);
  VCHECK(u != nullptr);
  {
    // An earlier run: its compile of region 0 was killed before it finished (here: a time limit of nothing; at a process's exit: the cancel).
    jit::Config cfg = jit::Config::from_env();
    cfg.timeout_s = 0;
    jit::Engine e(cfg);
    const jit::Result r = e.obtain_selected(u->spec, {0});
    VCHECK(!r.found);
    VCHECK(r.compiled);
    VCHECK(r.status == jit::Status::CompileTimeout);
  }
  {
    jit::Engine e(jit::Config::from_env());
    std::vector<uint32_t> sel;
    const jit::Result l = e.lookup_selected(u->spec, &sel);
    VCHECK(!l.found);
    VCHECK(l.status == jit::Status::Ok);
    VCHECK(l.pending);                 // the selection is on record, its object is not
    VCHECK(sel == std::vector<uint32_t>{0});
    FakeProgram fo("ggggggggx", 5);    // a section with nothing on record: a plain miss
    const auto o = jit::make_unit(fo.prog, 80);
    const jit::Result m = e.lookup_selected(o->spec, nullptr);
    VCHECK(!m.found && !m.pending);
  }
  // This run: the section is kept, waits for its lookup, and the lookup compiles the recorded selection at once (synchronous here) instead of waiting
  // ten million instructions for the threshold.
  const uint64_t hist0 = jit::stats().history.load(), compiles0 = jit::stats().compiles.load();
  u->note(0, 8192);
  u->sample(kSample);
  VCHECK_EQ(state_of(*u), kWaiting);
  u->wait_note(kLookup);
  VCHECK_EQ(state_of(*u), kIdle);
  u->note(0, 4096);
  VCHECK(u->ready());
  VCHECK_EQ(jit::stats().history.load(), hist0 + 1);
  VCHECK_EQ(jit::stats().compiles.load(), compiles0 + 1);
  VCHECK(u->fn(0) != nullptr);
}

// ---- the cache: its directory, the cap, the age rule ----------------------------------------------------------------------------------------------

namespace {
int own_build_id(struct dl_phdr_info* info, size_t, void* data) {
  auto* out = static_cast<std::string*>(data);
  const uintptr_t here = reinterpret_cast<uintptr_t>(&own_build_id);
  bool inside = false;
  for (int i = 0; i < info->dlpi_phnum; ++i) {
    const ElfW(Phdr)& p = info->dlpi_phdr[i];
    if (p.p_type == PT_LOAD && here >= info->dlpi_addr + p.p_vaddr && here < info->dlpi_addr + p.p_vaddr + p.p_memsz) inside = true;
  }
  if (!inside) return 0;
  for (int i = 0; i < info->dlpi_phnum; ++i) {
    const ElfW(Phdr)& p = info->dlpi_phdr[i];
    if (p.p_type != PT_NOTE) continue;
    const unsigned char* at = reinterpret_cast<const unsigned char*>(info->dlpi_addr + p.p_vaddr);
    const unsigned char* end = at + p.p_memsz;
    while (static_cast<size_t>(end - at) >= sizeof(ElfW(Nhdr))) {
      ElfW(Nhdr) n;
      std::memcpy(&n, at, sizeof n);
      const size_t nm = (n.n_namesz + 3u) & ~3u, ds = (n.n_descsz + 3u) & ~3u;
      if (static_cast<size_t>(end - at) < sizeof n + nm + ds) break;
      if (n.n_type == NT_GNU_BUILD_ID && n.n_namesz == 4 && std::memcmp(at + sizeof n, "GNU", 4) == 0) {
        char hx[3];
        for (uint32_t k = 0; k < n.n_descsz && k < 32; ++k) {
          std::snprintf(hx, sizeof hx, "%02x", at[sizeof n + nm + k]);
          *out += hx;
        }
        return 1;
      }
      at += sizeof n + nm + ds;
    }
  }
  return 1;
}
}  // namespace

VTEST(jit_the_default_cache_directory_is_named_by_the_build_id) {
  std::string id;
  dl_iterate_phdr(own_build_id, &id);   // (the test is linked with the simulator's archive: its executable is the object the JIT's code is in)
  if (id.empty()) {
    std::printf("  (no GNU build id in this executable: skipped)\n");
    return;
  }
  const std::string saved = std::getenv("VGPU_JIT_CACHE");
  const char* xdg = std::getenv("XDG_CACHE_HOME");
  const std::string saved_xdg = xdg ? xdg : "";
  ::unsetenv("VGPU_JIT_CACHE");
  ::setenv("XDG_CACHE_HOME", (g_env.dir / "xdg").c_str(), 1);
  jit::Engine e(jit::Config{});
  const std::string path = e.cache_path();
  ::setenv("VGPU_JIT_CACHE", saved.c_str(), 1);
  if (xdg) ::setenv("XDG_CACHE_HOME", saved_xdg.c_str(), 1);
  else ::unsetenv("XDG_CACHE_HOME");
  const std::string want = (g_env.dir / "xdg").string() + "/pantheonsim/jit/" + jit::checksum("build-id " + id + " abi" + std::to_string(jit::kAbiVersion)).substr(0, 12);
  VCHECK_EQ(path, want);
}

namespace {
// A file of `bytes` bytes changed `age_s` seconds ago; its directory is given the same time (making the file renewed it).
void put(const fs::path& p, size_t bytes, long age_s) {
  fs::create_directories(p.parent_path());
  std::ofstream(p, std::ios::binary) << std::string(bytes, 'x');
  const struct timespec ts[2] = {{std::time(nullptr) - age_s, 0}, {std::time(nullptr) - age_s, 0}};
  ::utimensat(AT_FDCWD, p.c_str(), ts, 0);
  ::utimensat(AT_FDCWD, p.parent_path().c_str(), ts, 0);
}
std::string key(char c) { return std::string(32, c); }
}  // namespace

VTEST(jit_the_cache_keeps_under_its_cap_least_recently_used_first_and_removes_old_builds_directories) {
  const fs::path root = g_env.dir / "trim" / "jit";
  const fs::path cur = root / "aaaaaaaaaaaa", stale = root / "bbbbbbbbbbbb", emptied = root / "cccccccccccc", fresh = root / "dddddddddddd", other = root / "notours";
  const long day = 86400;
  // The directory in use: four objects of 40 KB (k0 the oldest, and the one just made), the runtime headers.
  put(cur / (key('0') + ".so"), 40000, 9000);
  put(cur / (key('0') + ".sum"), 50, 9000);
  put(cur / (key('1') + ".so"), 40000, 5000);
  put(cur / (key('1') + ".sum"), 50, 5000);
  put(cur / (key('2') + ".so"), 40000, 4000);
  put(cur / (key('2') + ".sum"), 50, 4000);
  put(cur / (key('3') + ".so"), 40000, 10);
  put(cur / (key('3') + ".sum"), 50, 10);
  put(cur / "rt-0123456789abcdef" / "vgpu_jit_rt.h", 10000, 9000);
  // Another build's directory untouched for 20 days, one with an old object, one written a minute ago (a process of another build at work), and a
  // directory that is not the cache's.
  put(stale / (key('4') + ".so"), 1000, 20 * day);
  put(stale / "rt-0123456789abcdef" / "vgpu_jit_rt.h", 1000, 20 * day);
  put(emptied / (key('5') + ".so"), 40000, 8000);
  put(emptied / (key('5') + ".sum"), 50, 8000);
  put(emptied / "rt-0123456789abcdef" / "vgpu_jit_rt.h", 10000, 8000);
  put(fresh / "cid-x.txt", 30, 60);
  put(other / "file", 100, 30 * day);
  jit::Config cfg;
  cfg.cache_dir = cur.string();
  cfg.trim_siblings = true;
  cfg.max_bytes = 150000;   // 220 KB in scope (the stale directory aside): down to 135 KB
  const uint64_t ev0 = jit::stats().evicted.load();
  jit::Engine e(cfg);
  e.trim(key('0'));
  VCHECK(!fs::exists(stale));                              // the age rule
  VCHECK(!fs::exists(emptied / (key('5') + ".so")));       // the least recently used first: k5, k1, k2
  VCHECK(!fs::exists(emptied));                            // ... which left that directory with no object: removed with its headers
  VCHECK(!fs::exists(cur / (key('1') + ".so")));
  VCHECK(!fs::exists(cur / (key('1') + ".sum")));
  VCHECK(!fs::exists(cur / (key('2') + ".so")));
  VCHECK(fs::exists(cur / (key('3') + ".so")));            // 100 KB left: under 90 % of the cap, the newest stays
  VCHECK(fs::exists(cur / (key('3') + ".sum")));
  VCHECK(fs::exists(cur / (key('0') + ".so")));            // the object just made, though the oldest
  VCHECK(fs::exists(cur / "rt-0123456789abcdef" / "vgpu_jit_rt.h"));
  VCHECK(fs::exists(fresh / "cid-x.txt"));                 // a directory in use by another build
  VCHECK(fs::exists(other / "file"));                      // not the cache's
  VCHECK_EQ(jit::stats().evicted.load(), ev0 + 5);         // stale, k5, k1, k2, emptied
  // An explicit cache directory (VGPU_JIT_CACHE, Config::cache_dir) is trimmed by itself: its siblings are not the cache's.
  put(stale / (key('6') + ".so"), 1000, 20 * day);
  jit::Config c2;
  c2.cache_dir = cur.string();
  jit::Engine e2(c2);
  e2.trim(key('0'));
  VCHECK(fs::exists(stale / (key('6') + ".so")));
  // No cap: nothing goes but old builds' directories.
  jit::Config c3 = cfg;
  c3.max_bytes = 0;
  put(cur / (key('7') + ".so"), 900000, 99999);
  jit::Engine e3(c3);
  e3.trim(key('0'));
  VCHECK(fs::exists(cur / (key('7') + ".so")));
  VCHECK(!fs::exists(stale));
}

VTEST(jit_a_cache_hit_renews_the_objects_time) {
  if (!have_compiler()) {
    std::printf("  (no C++ compiler on PATH: skipped)\n");
    return;
  }
  FakeProgram fp("ggggggggx", 6);
  const auto u = jit::make_unit(fp.prog, 80);
  if (!u) return;   // (a sanitizer build)
  jit::Config cfg;
  cfg.cache_dir = (g_env.dir / "touch").string();
  jit::Engine e(cfg);
  const jit::Result r = e.obtain(u->spec);
  VCHECK(r.found && r.compiled);
  const std::string so = e.object_path(r.key);
  const struct timespec old[2] = {{std::time(nullptr) - 10000, 0}, {std::time(nullptr) - 10000, 0}};
  ::utimensat(AT_FDCWD, so.c_str(), old, 0);
  jit::Engine e2(cfg);
  const jit::Result l = e2.lookup(u->spec);
  VCHECK(l.found && l.hit);
  struct stat st;
  VCHECK(::stat(so.c_str(), &st) == 0);
  VCHECK(st.st_mtime > std::time(nullptr) - 1000);
}

// ---- the notices ------------------------------------------------------------------------------------------------------------------------------------

VTEST(jit_a_notice_is_given_once_a_day_and_not_at_all_with_auto) {
  if (!jit::requested()) return;
  if (self_exe().empty()) return;
  const std::string c1 = (g_env.dir / "n1").string(), c2 = (g_env.dir / "n2").string();
  const std::string line = "pantheonsim: JIT: no C++ compiler";
  ChildOut a = run_child("nocompiler", "1", c1);
  VCHECK_EQ(a.rc, 0);
  VCHECK_EQ(count_of(a.err, line), size_t{1});
  ChildOut b = run_child("nocompiler", "1", c1);   // the same day, the same cache directory: not again
  VCHECK_EQ(b.rc, 0);
  VCHECK_EQ(count_of(b.err, line), size_t{0});
  const std::string stamp = c1 + "/notice-no-compiler";
  VCHECK(fs::exists(stamp));
  const struct timespec old[2] = {{std::time(nullptr) - 2 * 86400, 0}, {std::time(nullptr) - 2 * 86400, 0}};
  ::utimensat(AT_FDCWD, stamp.c_str(), old, 0);
  ChildOut c = run_child("nocompiler", "1", c1);   // two days later: again
  VCHECK_EQ(count_of(c.err, line), size_t{1});
  ChildOut d = run_child("nocompiler", "auto", c2);   // auto: on, and silent about where it cannot run
  VCHECK_EQ(d.rc, 0);
  VCHECK(d.out.find("requested=1") != std::string::npos);
  VCHECK_EQ(count_of(d.err, "pantheonsim: JIT"), size_t{0});
}

VTEST(jit_a_compile_cancelled_at_exit_says_nothing_and_leaves_its_selection_for_the_next_run) {
  if (!jit::requested()) return;
  if (self_exe().empty()) return;
  const fs::path cxx = g_env.dir / "fake-c++";
  {
    std::ofstream f(cxx);
    f << "#!/bin/sh\ncase \"$1\" in --version) echo 'fake c++ 1.0';; -dumpmachine) echo x86_64-linux-gnu;; *) exec sleep 60;; esac\n";
  }
  ::chmod(cxx.c_str(), 0755);
  const std::string c3 = (g_env.dir / "n3").string();
  const auto t0 = std::chrono::steady_clock::now();
  ChildOut a = run_child("cancel", "1", c3, cxx.string());
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  VCHECK_EQ(a.rc, 0);
  VCHECK_EQ(count_of(a.err, "pantheonsim: JIT"), size_t{0});
  VCHECK(s < 30);   // (the compiler was killed, not waited for)
  size_t sels = 0, objs = 0;
  for (const auto& ent : fs::directory_iterator(c3)) {
    sels += ent.path().extension() == ".sel";
    objs += ent.path().extension() == ".so";
  }
  VCHECK_EQ(sels, size_t{1});
  VCHECK_EQ(objs, size_t{0});
}

VTEST_MAIN
