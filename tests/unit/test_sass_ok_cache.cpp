// sass::unsupported leaves a file of the cubins it found to run in full (src/sass/ok_cache.cpp), so the next process
// that loads the same fatbins does not decode every word of them again. The file must change nothing a caller can see:
// a refusal is never recorded and keeps its text, VGPU_SASS_REFUSE and VGPU_SASS_CACHE=0 leave it alone (unset is on: the default), a different
// image is a miss (every bit of the image is in the key), and a file that is not this user's alone, or has a torn
// tail, is not trusted or not appended to. Each test below fails if the cache answered for an image it was not given,
// recorded a refusal, read a file anyone could have written, or appended behind a torn key.
//
// The cubins are built here (one .text.k section) from words found in the corpora of nvidia/tests/data/sass; the
// cache directory is a fresh one under TMPDIR per scenario (XDG_CACHE_HOME), so nothing of the user's is touched.
#include <dirent.h>
#include <elf.h>
#include <fcntl.h>
#include <ftw.h>
#include <link.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../../src/sass/ok_cache.hpp"
#include "vgpu/error.hpp"
#include "vgpu/sass/exec.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

using namespace vgpu;
namespace okcache = vgpu::sass::okcache;

namespace vgpu::sass {
bool runs_instr(const Instr& ins);   // src/sass/exec.cpp
}

// While g_foreign_owner is set, this process claims to be another user than the one that owns its files, so the
// ownership checks can be tested without root. (The executable's definition wins over libc's, also in the library code.)
static bool g_foreign_owner = false;
static uid_t real_euid() { return static_cast<uid_t>(syscall(SYS_geteuid)); }
extern "C" uid_t geteuid(void) noexcept { return g_foreign_owner ? real_euid() + 1 : real_euid(); }

namespace {

template <class T>
void put(std::string& s, T v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof v);
}

// A cubin for sm_<sm> whose only code is .text.k holding `words`, in order.
std::string make_cubin(int sm, const std::vector<sass::Word>& words) {
  const std::string names = std::string("\0.shstrtab\0.text.k\0", 19);
  std::string out(64, '\0');
  const uint64_t names_off = out.size();
  out += names;
  while (out.size() % 16) out.push_back('\0');
  const uint64_t text_off = out.size();
  for (const sass::Word& w : words) {
    put(out, w.lo);
    put(out, w.hi);
  }
  const uint64_t shoff = out.size();
  const auto shdr = [&](uint32_t name, uint32_t type, uint64_t off, uint64_t size) {
    put<uint32_t>(out, name);
    put<uint32_t>(out, type);
    put<uint64_t>(out, 0);
    put<uint64_t>(out, 0);
    put<uint64_t>(out, off);
    put<uint64_t>(out, size);
    put<uint32_t>(out, 0);
    put<uint32_t>(out, 0);
    put<uint64_t>(out, 16);
    put<uint64_t>(out, 0);
  };
  shdr(0, 0, 0, 0);
  shdr(1, 3, names_off, names.size());
  shdr(11, 1, text_off, 16 * words.size());
  out[0] = 0x7f;
  out[1] = 'E';
  out[2] = 'L';
  out[3] = 'F';
  out[4] = 2;
  out[5] = 1;
  out[6] = 1;
  out[7] = 0x41;
  out[8] = 8;
  const uint16_t type = 2, machine = 190, shentsize = 64, shnum = 3, shstrndx = 1;
  const uint32_t flags = 0x06000002u | (static_cast<uint32_t>(sm) << 8);
  std::memcpy(&out[16], &type, 2);
  std::memcpy(&out[18], &machine, 2);
  std::memcpy(&out[0x28], &shoff, 8);
  std::memcpy(&out[0x30], &flags, 4);
  std::memcpy(&out[0x3a], &shentsize, 2);
  std::memcpy(&out[0x3c], &shnum, 2);
  std::memcpy(&out[0x3e], &shstrndx, 2);
  return out;
}

std::string run(const std::string& cubin) {
  return sass::unsupported(reinterpret_cast<const uint8_t*>(cubin.data()), cubin.size());
}

bool runs(int sm, const sass::Word& w) {
  try {
    return sass::runs_instr(sass::decode(w, 0, sm));
  } catch (const Error&) {
    return false;
  }
}

std::vector<sass::Word> corpus(const char* name) {
  std::ifstream in(std::string(VGPU_SOURCE_DIR) + "/nvidia/tests/data/sass/" + name + ".txt");
  if (!in) throw vtest::Failure(std::string("no corpus ") + name);
  std::vector<sass::Word> out;
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    sass::Word w;
    w.hi = std::stoull(line.substr(0, 16), nullptr, 16);
    w.lo = std::stoull(line.substr(16, 16), nullptr, 16);
    out.push_back(w);
  }
  return out;
}

// `n` words that run on sm_86, starting at the `from`-th of the ones found (a different `from` is a different cubin).
std::vector<sass::Word> good_words(size_t n, size_t from) {
  std::vector<sass::Word> all = corpus("sm_86"), out;
  size_t seen = 0;
  for (size_t i = 0; i < all.size() && out.size() < n; i += 53)
    if (runs(86, all[i]) && seen++ >= from) out.push_back(all[i]);
  return out;
}

std::string unknown_cubin() {
  std::vector<sass::Word> w = good_words(10, 0);
  sass::Word bad;
  bad.lo = 0xfff;   // an opcode no decoder has
  w.insert(w.begin() + 4, bad);
  return make_cubin(86, w);
}

// The directories fresh_root made, removed when the test exits.
struct Roots {
  std::vector<std::string> made;
  ~Roots() {
    for (const std::string& r : made)
      nftw(r.c_str(), [](const char* p, const struct stat*, int, struct FTW*) { return std::remove(p); }, 8,
           FTW_DEPTH | FTW_PHYS);
  }
} g_roots;

// A fresh directory standing for $XDG_CACHE_HOME.
std::string fresh_root() {
  const char* t = std::getenv("TMPDIR");
  std::string tmpl = std::string(t && *t ? t : "/tmp") + "/vgpu-okcache-XXXXXX";
  if (!mkdtemp(tmpl.data())) throw vtest::Failure("mkdtemp");
  g_roots.made.push_back(tmpl);
  setenv("XDG_CACHE_HOME", tmpl.c_str(), 1);
  unsetenv("VGPU_SASS_REFUSE");
  unsetenv("VGPU_SASS_CACHE");   // the default is on
  return tmpl;
}

// The cache file under `root`, "" when there is none.
std::string cache_file(const std::string& root) {
  const std::string dir = root + "/pantheonsim";
  std::string found;
  if (DIR* d = opendir(dir.c_str())) {
    while (const dirent* e = readdir(d))
      if (std::strncmp(e->d_name, "sass-ok-v1-", 11) == 0) found = dir + "/" + e->d_name;
    closedir(d);
  }
  return found;
}

long size_of(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 ? static_cast<long>(st.st_size) : -1;
}

// The directory and file name a build with an id would use, found by making the cache write one.
// (Asked with VGPU_SASS_CACHE=1, so that a default of "off" is not mistaken for a build with no id: the tests that need
// a cache must not skip themselves when the cache is off.)
bool has_cache(const std::string& root) {
  const std::string probe = make_cubin(86, good_words(6, 200));
  const char* was = std::getenv("VGPU_SASS_CACHE");
  const std::string saved = was ? was : "";
  setenv("VGPU_SASS_CACHE", "1", 1);
  const bool ran = run(probe).empty();
  if (was) setenv("VGPU_SASS_CACHE", saved.c_str(), 1);
  else unsetenv("VGPU_SASS_CACHE");
  VCHECK(ran);
  return !cache_file(root).empty();
}

// The name of the cache file this build uses ("" when it has none), found once by making the cache write one.
const std::string& real_name() {
  static const std::string name = [] {
    const std::string r2 = fresh_root();
    has_cache(r2);
    const std::string f = cache_file(r2);
    return f.empty() ? f : f.substr(f.rfind('/') + 1);
  }();
  return name;
}

// A file named `name` under `root`/pantheonsim holding `bytes`, with the modes given.
std::string plant_named(const std::string& root, const std::string& name, const std::string& bytes, mode_t dir_mode,
                        mode_t file_mode) {
  setenv("XDG_CACHE_HOME", root.c_str(), 1);
  const std::string dir = root + "/pantheonsim";
  mkdir(dir.c_str(), 0700);
  chmod(dir.c_str(), dir_mode);
  const std::string path = dir + "/" + name;
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  chmod(path.c_str(), file_mode);
  return path;
}

std::string key_bytes(const std::string& cubin) {
  const okcache::Key k = okcache::key_of(reinterpret_cast<const uint8_t*>(cubin.data()), cubin.size());
  return std::string(reinterpret_cast<const char*>(&k), sizeof k);
}

// A file under `root` holding the keys of `cubins` and `tail` extra bytes, with the modes given.
std::string plant(const std::string& root, const std::vector<std::string>& cubins, size_t tail, mode_t dir_mode,
                  mode_t file_mode) {
  const std::string real = real_name();
  if (real.empty()) return {};
  std::string bytes;
  for (const std::string& c : cubins) bytes += key_bytes(c);
  bytes += std::string(tail, 'x');
  return plant_named(root, real, bytes, dir_mode, file_mode);
}

}  // namespace

VTEST(an_image_that_ran_is_answered_from_the_file_and_nothing_else_is) {
  const std::string root = fresh_root();
  if (!has_cache(root)) return;   // a build without a GNU build id has no cache
  const std::string root2 = fresh_root();
  const std::string a = make_cubin(86, good_words(40, 0)), b = make_cubin(86, good_words(40, 1));
  const uint64_t h0 = sass::unsupported_cache_hits();
  VCHECK(run(a).empty());
  VCHECK_EQ(sass::unsupported_cache_hits(), h0);   // decoded, then recorded
  const std::string file = cache_file(root2);
  VCHECK(!file.empty());
  VCHECK_EQ(size_of(file), 16L);
  VCHECK(run(a).empty());
  VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 1);
  VCHECK_EQ(size_of(file), 16L);   // known: not appended again
  VCHECK(run(b).empty());          // another image is a miss, however alike
  VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 1);
  VCHECK_EQ(size_of(file), 32L);
  // The same words for another architecture are another image (the architecture is in the ELF header).
  std::string c = a;
  c[0x30 + 1] = static_cast<char>(75);
  run(c);
  VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 1);
  VCHECK(run(a).empty());
  VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 2);
}

VTEST(a_refusal_is_never_recorded_and_keeps_its_text) {
  const std::string root = fresh_root();
  if (!has_cache(root)) return;
  const std::string root2 = fresh_root();
  const std::string bad = unknown_cubin();
  const uint64_t h0 = sass::unsupported_cache_hits();
  const std::string first = run(bad);
  VCHECK(first.find("unknown opcode") != std::string::npos);
  VCHECK(cache_file(root2).empty());
  VCHECK_EQ(run(bad), first);
  VCHECK_EQ(run(bad), first);
  VCHECK(cache_file(root2).empty());
  VCHECK_EQ(sass::unsupported_cache_hits(), h0);
}

VTEST(refuse_and_cache_off_bypass_the_file) {
  const std::string root = fresh_root();
  if (!has_cache(root)) return;
  const std::string root2 = fresh_root();
  const std::string a = make_cubin(86, good_words(40, 0));
  VCHECK(run(a).empty());   // recorded
  const std::string file = cache_file(root2);
  VCHECK_EQ(size_of(file), 16L);
  // The first instruction's name, refused: the recorded image no longer runs in full.
  const sass::Instr first = sass::decode(good_words(1, 0)[0], 0, 86);
  setenv("VGPU_SASS_REFUSE", first.mnemonic.c_str(), 1);
  const uint64_t h0 = sass::unsupported_cache_hits();
  VCHECK(!run(a).empty());
  VCHECK_EQ(sass::unsupported_cache_hits(), h0);
  unsetenv("VGPU_SASS_REFUSE");
  VCHECK(run(a).empty());
  VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 1);
  // VGPU_SASS_CACHE=0 (and "0..."): neither read nor written.
  const std::string b = make_cubin(86, good_words(40, 2));
  const char* const offs[] = {"0", "0x"};
  for (const char* off : offs) {
    setenv("VGPU_SASS_CACHE", off, 1);
    VCHECK(run(a).empty());
    VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 1);
    VCHECK(run(b).empty());
    VCHECK_EQ(size_of(file), 16L);
  }
  // Anything else, or unset, is on: the cache is the default.
  const char* const ons[] = {nullptr, "", "1", "yes"};
  uint64_t expect = h0 + 1;
  for (const char* on : ons) {
    if (on) setenv("VGPU_SASS_CACHE", on, 1);
    else unsetenv("VGPU_SASS_CACHE");
    VCHECK(run(a).empty());
    VCHECK_EQ(sass::unsupported_cache_hits(), ++expect);   // each answered from the file
  }
  unsetenv("VGPU_SASS_CACHE");
}

VTEST(the_cache_is_on_by_default_and_off_with_zero) {
  const std::string root = fresh_root();   // VGPU_SASS_CACHE unset
  if (!has_cache(root)) return;
  const std::string root2 = fresh_root();
  const std::string a = make_cubin(86, good_words(40, 0));
  const uint64_t h0 = sass::unsupported_cache_hits();
  VCHECK(run(a).empty());
  VCHECK_EQ(size_of(cache_file(root2)), 16L);   // written with nothing set
  VCHECK(run(a).empty());
  VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 1);
  // Off: no file is made, none is read.
  const std::string root3 = fresh_root();
  setenv("VGPU_SASS_CACHE", "0", 1);
  VCHECK(run(a).empty());
  VCHECK(run(a).empty());
  VCHECK(cache_file(root3).empty());
  VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 1);
  unsetenv("VGPU_SASS_CACHE");
}

VTEST(a_torn_tail_is_ignored_and_never_appended_to) {
  const std::string probe_root = fresh_root();
  if (!has_cache(probe_root)) return;
  const std::string a = make_cubin(86, good_words(40, 0)), b = make_cubin(86, good_words(40, 1));
  const std::string root = fresh_root();
  const std::string file = plant(root, {a}, 5, 0700, 0600);
  VCHECK(!file.empty());
  const uint64_t h0 = sass::unsupported_cache_hits();
  VCHECK(run(a).empty());
  VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 1);   // the whole key before the tail is read
  VCHECK(run(b).empty());
  VCHECK_EQ(size_of(file), 21L);   // b was not appended behind the torn bytes
}

VTEST(a_file_or_directory_others_can_write_is_not_trusted) {
  const std::string probe_root = fresh_root();
  if (!has_cache(probe_root)) return;
  const std::string a = make_cubin(86, good_words(40, 0));
  const uint64_t h0 = sass::unsupported_cache_hits();
  {
    const std::string root = fresh_root();
    const std::string file = plant(root, {a}, 0, 0700, 0666);   // anyone could have written the file
    VCHECK(!file.empty());
    VCHECK(run(a).empty());
    VCHECK_EQ(sass::unsupported_cache_hits(), h0);
    VCHECK_EQ(size_of(file), 16L);   // and it is not appended to
  }
  {
    const std::string root = fresh_root();
    const std::string file = plant(root, {a}, 0, 0777, 0600);   // or the directory
    VCHECK(!file.empty());
    VCHECK(run(a).empty());
    VCHECK_EQ(sass::unsupported_cache_hits(), h0);
    VCHECK_EQ(size_of(file), 16L);
  }
  {
    const std::string root = fresh_root();
    const std::string file = plant(root, {a}, 0, 0700, 0600);   // the same, as it should be: trusted
    VCHECK(!file.empty());
    VCHECK(run(a).empty());
    VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 1);
  }
}

VTEST(the_key_covers_every_bit_and_the_length) {
  std::vector<uint8_t> buf(100);
  for (size_t i = 0; i < buf.size(); ++i) buf[i] = static_cast<uint8_t>(i * 7 + 3);
  std::vector<okcache::Key> keys;
  for (size_t n = 0; n <= 70; ++n) keys.push_back(okcache::key_of(buf.data(), n));
  for (size_t i = 0; i < buf.size(); ++i)
    for (int bit = 0; bit < 8; ++bit) {
      buf[i] ^= static_cast<uint8_t>(1u << bit);
      keys.push_back(okcache::key_of(buf.data(), buf.size()));
      buf[i] ^= static_cast<uint8_t>(1u << bit);
    }
  keys.push_back(okcache::key_of(buf.data(), buf.size()));
  const std::vector<uint8_t> zeros(64, 0);
  for (size_t n = 1; n <= 64; ++n) keys.push_back(okcache::key_of(zeros.data(), n));   // lengths of nothing but zeros
  for (size_t i = 0; i < keys.size(); ++i)
    for (size_t j = i + 1; j < keys.size(); ++j) VCHECK(keys[i].a != keys[j].a || keys[i].b != keys[j].b);
}

// ---------------------------------------------------------------------------------------------------------------------
// The cache is meant to be on for everybody, so it must be harmless wherever it cannot work: no home, a read-only home,
// a directory or file that is not ours, a full disk, a cut file, a crowd of processes starting together, too many keys.
// In every case below the verdict of sass::unsupported is the same as without the cache (checked), nothing crashes or
// hangs, and the cache says "ran in full" only for an image it was told ran in full.
// ---------------------------------------------------------------------------------------------------------------------

namespace {

// Environment and working directory, put back when the scope ends.
struct EnvGuard {
  std::string home, xdg, cwd;
  bool has_home = false, has_xdg = false;
  EnvGuard() {
    if (const char* h = std::getenv("HOME")) { has_home = true; home = h; }
    if (const char* x = std::getenv("XDG_CACHE_HOME")) { has_xdg = true; xdg = x; }
    char buf[4096];
    if (getcwd(buf, sizeof buf)) cwd = buf;
  }
  ~EnvGuard() {
    if (has_home) setenv("HOME", home.c_str(), 1); else unsetenv("HOME");
    if (has_xdg) setenv("XDG_CACHE_HOME", xdg.c_str(), 1); else unsetenv("XDG_CACHE_HOME");
    if (!cwd.empty() && chdir(cwd.c_str()) != 0) std::perror("chdir");
    g_foreign_owner = false;
  }
};

// A directory's mode, put back when the scope ends (so the roots can be removed).
struct ModeGuard {
  std::string path;
  mode_t mode;
  ModeGuard(std::string p, mode_t m) : path(std::move(p)), mode(m) {}
  ~ModeGuard() { chmod(path.c_str(), mode); }
};

std::vector<std::string> entries_of(const std::string& dir) {
  std::vector<std::string> out;
  if (DIR* d = opendir(dir.c_str())) {
    while (const dirent* e = readdir(d))
      if (std::strcmp(e->d_name, ".") != 0 && std::strcmp(e->d_name, "..") != 0) out.push_back(e->d_name);
    closedir(d);
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::string slurp(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// A fresh empty directory, made the working directory.
std::string fresh_cwd() {
  const std::string d = fresh_root();
  if (chdir(d.c_str()) != 0) throw vtest::Failure("chdir");
  return d;
}

bool is_root() { return real_euid() == 0; }

// An image that ran must never be answered "refused", nor a refusal "ran": the verdict, with the cache on.
void expect_runs(const std::string& cubin) { VCHECK(run(cubin).empty()); }

}  // namespace

VTEST(no_home_and_no_cache_dir_means_no_cache) {
  const std::string probe = fresh_root();
  if (!has_cache(probe)) return;
  EnvGuard env;
  const std::string a = make_cubin(86, good_words(40, 0));
  struct Case { const char* home; const char* xdg; };
  // Nothing set; empty; relative (the XDG rule: ignored, or the cache would land in whatever directory the process is in).
  const Case cases[] = {{nullptr, nullptr}, {"", ""}, {"rel-home", "rel-xdg"}, {nullptr, "rel-xdg"}, {"rel-home", nullptr}};
  for (const Case& c : cases) {
    const std::string cwd = fresh_cwd();
    if (c.home) setenv("HOME", c.home, 1); else unsetenv("HOME");
    if (c.xdg) setenv("XDG_CACHE_HOME", c.xdg, 1); else unsetenv("XDG_CACHE_HOME");
    setenv("VGPU_SASS_CACHE", "1", 1);
    const uint64_t h0 = sass::unsupported_cache_hits();
    expect_runs(a);
    expect_runs(a);
    VCHECK_EQ(sass::unsupported_cache_hits(), h0);   // no file to answer from
    VCHECK(entries_of(cwd).empty());                 // and nothing written next to the process
  }
  // A relative XDG_CACHE_HOME is skipped, not the whole cache: HOME's .cache is used.
  {
    const std::string cwd = fresh_cwd(), home = fresh_root();
    setenv("HOME", home.c_str(), 1);
    setenv("XDG_CACHE_HOME", "rel-xdg", 1);
    setenv("VGPU_SASS_CACHE", "1", 1);
    expect_runs(a);
    VCHECK(entries_of(cwd).empty());
    VCHECK_EQ(size_of(home + "/.cache/pantheonsim/" + real_name()), 16L);
  }
  // HOME alone works too (~/.cache is made when it is missing).
  {
    const std::string home = fresh_root();
    unsetenv("XDG_CACHE_HOME");
    setenv("HOME", home.c_str(), 1);
    setenv("VGPU_SASS_CACHE", "1", 1);
    const uint64_t h0 = sass::unsupported_cache_hits();
    expect_runs(a);
    expect_runs(a);
    VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 1);
    VCHECK_EQ(size_of(home + "/.cache/pantheonsim/" + real_name()), 16L);
  }
}

VTEST(a_cache_dir_that_cannot_be_written_changes_no_verdict) {
  const std::string probe = fresh_root();
  if (!has_cache(probe)) return;
  const std::string a = make_cubin(86, good_words(40, 0)), b = make_cubin(86, good_words(40, 1));
  const uint64_t h0 = sass::unsupported_cache_hits();
  // The cache root is a file; its parent is missing; it is read-only.
  {
    const std::string root = fresh_root();
    const std::string file_root = root + "/a-file";
    { std::ofstream(file_root) << "x"; }
    setenv("XDG_CACHE_HOME", file_root.c_str(), 1);
    expect_runs(a);
    expect_runs(a);
    setenv("XDG_CACHE_HOME", (root + "/no/such/parent").c_str(), 1);
    expect_runs(a);
    expect_runs(a);
    VCHECK_EQ(entries_of(root).size(), size_t{1});   // only a-file
    VCHECK_EQ(sass::unsupported_cache_hits(), h0);
  }
  if (is_root()) return;   // permissions do not stop root; the rest needs a user
  {
    const std::string root = fresh_root();
    ModeGuard g(root, 0700);
    chmod(root.c_str(), 0500);   // no pantheonsim directory, and none can be made
    expect_runs(a);
    expect_runs(a);
    VCHECK(entries_of(root).empty());
    VCHECK_EQ(sass::unsupported_cache_hits(), h0);
  }
  {
    const std::string root = fresh_root();
    const std::string dir = root + "/pantheonsim";
    mkdir(dir.c_str(), 0700);
    ModeGuard g(dir, 0700);
    chmod(dir.c_str(), 0500);   // the directory is there but nothing can be made in it
    expect_runs(a);
    expect_runs(a);
    VCHECK(entries_of(dir).empty());
    VCHECK_EQ(sass::unsupported_cache_hits(), h0);
  }
  {
    // A warm file in a read-only place still answers, and is not appended to (nor does the failure to do so show).
    const std::string root = fresh_root();
    const std::string file = plant(root, {a}, 0, 0700, 0400);
    VCHECK(!file.empty());
    ModeGuard g(root + "/pantheonsim", 0700);
    chmod((root + "/pantheonsim").c_str(), 0500);
    expect_runs(a);
    VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 1);
    expect_runs(b);
    expect_runs(b);
    VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 1);
    VCHECK_EQ(size_of(file), 16L);
  }
}

VTEST(a_symlink_or_fifo_in_place_of_the_cache_is_not_used) {
  const std::string probe = fresh_root();
  if (!has_cache(probe)) return;
  const std::string a = make_cubin(86, good_words(40, 0)), b = make_cubin(86, good_words(40, 1));
  const uint64_t h0 = sass::unsupported_cache_hits();
  const std::string name = real_name();
  {
    // pantheonsim is a symlink to a good directory with a good, warm file in it.
    const std::string elsewhere = fresh_root(), root = fresh_root();
    const std::string target = plant(elsewhere, {a}, 0, 0700, 0600);
    VCHECK(!target.empty());
    setenv("XDG_CACHE_HOME", root.c_str(), 1);
    VCHECK(symlink((elsewhere + "/pantheonsim").c_str(), (root + "/pantheonsim").c_str()) == 0);
    expect_runs(a);
    expect_runs(b);
    VCHECK_EQ(sass::unsupported_cache_hits(), h0);
    VCHECK_EQ(size_of(target), 16L);
    VCHECK_EQ(entries_of(elsewhere + "/pantheonsim").size(), size_t{1});
  }
  {
    // The file is a symlink to a good, warm file.
    const std::string elsewhere = fresh_root(), root = fresh_root();
    const std::string target = plant(elsewhere, {a}, 0, 0700, 0600);
    plant_named(root, "unused", "", 0700, 0600);
    unlink((root + "/pantheonsim/unused").c_str());
    VCHECK(symlink(target.c_str(), (root + "/pantheonsim/" + name).c_str()) == 0);
    expect_runs(a);
    expect_runs(b);
    VCHECK_EQ(sass::unsupported_cache_hits(), h0);
    VCHECK_EQ(size_of(target), 16L);
  }
  {
    // The file is a dangling symlink: nothing is created behind it.
    const std::string elsewhere = fresh_root(), root = fresh_root();
    plant_named(root, "unused", "", 0700, 0600);
    unlink((root + "/pantheonsim/unused").c_str());
    const std::string target = elsewhere + "/planted";
    VCHECK(symlink(target.c_str(), (root + "/pantheonsim/" + name).c_str()) == 0);
    expect_runs(a);
    expect_runs(a);
    VCHECK_EQ(size_of(target), -1L);
    VCHECK_EQ(sass::unsupported_cache_hits(), h0);
  }
  {
    // The cache root itself may be a symlink (~/.cache often is): that works as usual.
    const std::string real = fresh_root(), holder = fresh_root();
    VCHECK(symlink(real.c_str(), (holder + "/link").c_str()) == 0);
    setenv("XDG_CACHE_HOME", (holder + "/link").c_str(), 1);
    expect_runs(a);
    expect_runs(a);
    VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 1);
    VCHECK_EQ(size_of(real + "/pantheonsim/" + name), 16L);
  }
  {
    // A FIFO where the file goes: opening it must neither block nor be trusted.
    const std::string root = fresh_root();
    plant_named(root, "unused", "", 0700, 0600);
    unlink((root + "/pantheonsim/unused").c_str());
    VCHECK(mkfifo((root + "/pantheonsim/" + name).c_str(), 0600) == 0);
    const uint64_t base = sass::unsupported_cache_hits();
    alarm(30);   // the default action ends the test run: a hang fails it
    expect_runs(a);
    expect_runs(b);
    alarm(0);
    VCHECK_EQ(sass::unsupported_cache_hits(), base);
  }
}

VTEST(a_file_or_directory_of_another_user_is_not_used) {
  const std::string probe = fresh_root();
  if (!has_cache(probe)) return;
  EnvGuard env;
  const std::string a = make_cubin(86, good_words(40, 0)), b = make_cubin(86, good_words(40, 1));
  const uint64_t h0 = sass::unsupported_cache_hits();
  const std::string root = fresh_root();
  const std::string file = plant(root, {a}, 0, 0700, 0600);   // mode-wise a perfect cache ...
  VCHECK(!file.empty());
  g_foreign_owner = true;                                       // ... but it belongs to somebody else
  expect_runs(a);
  expect_runs(b);
  g_foreign_owner = false;
  VCHECK_EQ(sass::unsupported_cache_hits(), h0);
  VCHECK_EQ(size_of(file), 16L);   // not appended to
  // And it is only the owner: as itself, the very same file answers.
  const std::string root2 = fresh_root();
  VCHECK(!plant(root2, {a}, 0, 0700, 0600).empty());
  expect_runs(a);
  VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 1);
  // A directory that is not ours is not made use of either, even when it holds nothing yet.
  const std::string root3 = fresh_root();
  g_foreign_owner = true;
  expect_runs(a);
  g_foreign_owner = false;
  VCHECK(cache_file(root3).empty());
}

VTEST(the_file_belongs_to_this_build_only) {
  const std::string probe = fresh_root();
  if (!has_cache(probe)) return;
  const std::string a = make_cubin(86, good_words(40, 0));
  const uint64_t h0 = sass::unsupported_cache_hits();
  const std::string name = real_name();
  // The name is "sass-ok-v1-" and the GNU build id of the object this code is linked into (the first one: this executable).
  std::string id;
  dl_iterate_phdr(
      [](struct dl_phdr_info* info, size_t, void* data) {
        std::string& out = *static_cast<std::string*>(data);
        for (int i = 0; i < info->dlpi_phnum; ++i) {
          const ElfW(Phdr)& p = info->dlpi_phdr[i];
          if (p.p_type != PT_NOTE) continue;
          const unsigned char* at = reinterpret_cast<const unsigned char*>(info->dlpi_addr + p.p_vaddr);
          const unsigned char* end = at + p.p_memsz;
          while (end - at >= static_cast<long>(sizeof(ElfW(Nhdr)))) {
            ElfW(Nhdr) n;
            std::memcpy(&n, at, sizeof n);
            const size_t nm = (n.n_namesz + 3u) & ~3u, ds = (n.n_descsz + 3u) & ~3u;
            if (n.n_type == NT_GNU_BUILD_ID) {
              for (uint32_t k = 0; k < n.n_descsz; ++k) {
                char two[3];
                std::snprintf(two, sizeof two, "%02x", at[sizeof n + nm + k]);
                out += two;
              }
              return 1;
            }
            at += sizeof n + nm + ds;
          }
        }
        return 1;   // the first object is the executable
      },
      &id);
  VCHECK(!id.empty());
  VCHECK_EQ(name, "sass-ok-v1-" + id);
  // A file of another build (another id, and another format version) in the same directory is left alone and not read.
  const std::string root = fresh_root();
  const std::string other_id = "sass-ok-v1-" + std::string(id.size(), '0');
  const std::string other_ver = "sass-ok-v2-" + id;
  const std::string f1 = plant_named(root, other_id, key_bytes(a), 0700, 0600);
  const std::string f2 = plant_named(root, other_ver, key_bytes(a), 0700, 0600);
  expect_runs(a);
  VCHECK_EQ(sass::unsupported_cache_hits(), h0);   // the other builds' answers are not this build's
  expect_runs(a);
  VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 1);
  VCHECK_EQ(size_of(f1), 16L);
  VCHECK_EQ(size_of(f2), 16L);
  VCHECK_EQ(size_of(root + "/pantheonsim/" + name), 16L);
  VCHECK_EQ(entries_of(root + "/pantheonsim").size(), size_t{3});
}

VTEST(a_file_cut_anywhere_is_read_as_far_as_it_is_whole) {
  const std::string probe = fresh_root();
  if (!has_cache(probe)) return;
  const std::string a = make_cubin(86, good_words(40, 0)), b = make_cubin(86, good_words(40, 1)),
                    c = make_cubin(86, good_words(40, 2));
  const uint64_t h0 = sass::unsupported_cache_hits();
  // Cut after 0, 1, 7, 15, 16, 17, 31, 32, 33 bytes of a file of the keys of a, b, c.
  const size_t cuts[] = {0, 1, 7, 15, 16, 17, 31, 32, 33};
  const std::string all = key_bytes(a) + key_bytes(b) + key_bytes(c);
  for (const size_t cut : cuts) {
    const std::string root = fresh_root();
    const std::string file = plant_named(root, real_name(), all.substr(0, cut), 0700, 0600);
    const uint64_t base = sass::unsupported_cache_hits();
    const uint64_t whole = cut / 16;
    expect_runs(a);
    expect_runs(b);
    expect_runs(c);
    // The keys before the cut are answered, the others are decoded.
    VCHECK_EQ(sass::unsupported_cache_hits() - base, std::min<uint64_t>(whole, 3));
    // A whole file is appended to (c at most once more); one with a torn end is left as it is.
    if (cut % 16 != 0) VCHECK_EQ(size_of(file), static_cast<long>(cut));
    else VCHECK_EQ(size_of(file), 48L);
    // A second pass, in the same process, is never wrong either.
    expect_runs(a);
    expect_runs(c);
  }
  VCHECK(sass::unsupported_cache_hits() >= h0);
}

VTEST(a_full_disk_leaves_the_file_whole_and_the_verdicts_right) {
  const std::string probe = fresh_root();
  if (!has_cache(probe)) return;
  signal(SIGXFSZ, SIG_IGN);   // a write past the size limit: a short count, then EFBIG, as on a full disk
  struct rlimit saved;
  VCHECK(getrlimit(RLIMIT_FSIZE, &saved) == 0);
  const auto limit = [&](rlim_t n) {
    struct rlimit r = saved;
    r.rlim_cur = n;
    VCHECK(setrlimit(RLIMIT_FSIZE, &r) == 0);
  };
  const std::string a = make_cubin(86, good_words(40, 0)), b = make_cubin(86, good_words(40, 1)),
                    c = make_cubin(86, good_words(40, 2)), d = make_cubin(86, good_words(40, 3));
  const std::string root = fresh_root();
  expect_runs(a);
  expect_runs(b);
  const std::string file = cache_file(root);
  VCHECK_EQ(size_of(file), 32L);
  const uint64_t h0 = sass::unsupported_cache_hits();
  limit(40);   // room for half of the next key
  expect_runs(c);
  VCHECK_EQ(size_of(file), 32L);   // the half key is taken back: the file stays whole
  limit(0);    // no room at all (EFBIG, nothing written)
  expect_runs(d);
  expect_runs(c);
  VCHECK_EQ(size_of(file), 32L);
  VCHECK_EQ(sass::unsupported_cache_hits(), h0);   // c and d were not recorded: they run again, fully
  expect_runs(a);                                  // and what is recorded is still answered
  VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 1);
  limit(saved.rlim_cur);   // space again: the next process (here: the next image) records
  expect_runs(c);
  VCHECK_EQ(size_of(file), 48L);
  expect_runs(c);
  VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 2);
  // The same from a cold start: the first key does not fit, so no file is made whole or torn.
  const std::string root2 = fresh_root();
  limit(8);
  expect_runs(a);
  expect_runs(b);
  limit(saved.rlim_cur);
  const std::string f2 = cache_file(root2);
  VCHECK(f2.empty() || size_of(f2) % 16 == 0);
  expect_runs(a);
  expect_runs(a);
  VCHECK_EQ(size_of(cache_file(root2)) % 16, 0L);
  setrlimit(RLIMIT_FSIZE, &saved);
  VCHECK_EQ(size_of(cache_file(root2)), 16L);
}

VTEST(processes_that_start_together_leave_one_good_file) {
  const std::string probe = fresh_root();
  if (!has_cache(probe)) return;
  constexpr int kProcs = 12, kRounds = 8, kShared = 5;
  std::vector<std::string> shared;
  for (int i = 0; i < kShared; ++i) shared.push_back(make_cubin(86, good_words(18 + 2 * i, 0)));
  std::vector<std::string> own;   // one of its own for each process
  for (int i = 0; i < kProcs; ++i) own.push_back(make_cubin(86, good_words(10 + 1 * i, 1)));
  const std::string bad = unknown_cubin();
  {
    std::vector<std::string> keys;
    for (const std::string& s : shared) keys.push_back(key_bytes(s));
    for (const std::string& s : own) keys.push_back(key_bytes(s));
    std::sort(keys.begin(), keys.end());
    VCHECK(std::adjacent_find(keys.begin(), keys.end()) == keys.end());   // all different images
  }
  for (int round = 0; round < kRounds; ++round) {
    const std::string root = fresh_root();   // a first run: no directory, no file
    int gate[2];
    VCHECK(pipe(gate) == 0);
    std::vector<pid_t> kids;
    for (int p = 0; p < kProcs; ++p) {
      const pid_t pid = fork();
      VCHECK(pid >= 0);
      if (pid == 0) {
        close(gate[1]);
        char ch;
        if (read(gate[0], &ch, 1) < 0) _exit(90);   // all start when the parent closes its end
        int code = 0;
        std::vector<const std::string*> order;
        for (const std::string& s : shared) order.push_back(&s);
        order.push_back(&own[p]);
        order.push_back(&bad);
        std::rotate(order.begin(), order.begin() + (p % order.size()), order.end());
        try {
          for (int pass = 0; pass < 2; ++pass)
            for (const std::string* img : order) {
              const std::string v = run(*img);
              if (img == &bad ? v.find("unknown opcode") == std::string::npos : !v.empty()) code = 1 + (img == &bad);
            }
        } catch (...) {
          code = 99;
        }
        _exit(code);
      }
      kids.push_back(pid);
    }
    close(gate[0]);
    close(gate[1]);
    for (const pid_t pid : kids) {
      int st = 0;
      VCHECK(waitpid(pid, &st, 0) == pid);
      VCHECK(WIFEXITED(st));
      VCHECK_EQ(WEXITSTATUS(st), 0);   // no crash, no wrong verdict
    }
    const std::string file = cache_file(root);
    VCHECK(!file.empty());
    const std::string bytes = slurp(file);
    VCHECK_EQ(bytes.size() % 16, size_t{0});   // no torn record
    VCHECK(bytes.size() <= static_cast<size_t>(kProcs) * (kShared + 1) * 16);
    std::vector<std::string> good;
    for (const std::string& s : shared) good.push_back(key_bytes(s));
    for (const std::string& s : own) good.push_back(key_bytes(s));
    std::vector<bool> seen(good.size(), false);
    for (size_t at = 0; at < bytes.size(); at += 16) {
      const std::string rec = bytes.substr(at, 16);
      const auto it = std::find(good.begin(), good.end(), rec);
      VCHECK(it != good.end());   // every record is the key of an image that ran
      if (it != good.end()) seen[static_cast<size_t>(it - good.begin())] = true;
      VCHECK(rec != key_bytes(bad));
    }
    for (size_t i = 0; i < seen.size(); ++i) VCHECK(seen[i]);   // and none of them was lost
    const uint64_t h0 = sass::unsupported_cache_hits();
    for (const std::string& s : shared) expect_runs(s);
    for (const std::string& s : own) expect_runs(s);
    VCHECK_EQ(sass::unsupported_cache_hits() - h0, static_cast<uint64_t>(shared.size() + own.size()));
  }
}

VTEST(the_file_stops_growing_at_65536_keys) {
  const std::string probe = fresh_root();
  if (!has_cache(probe)) return;
  constexpr size_t kCap = size_t{1} << 16;
  const std::string a = make_cubin(86, good_words(40, 0)), b = make_cubin(86, good_words(40, 1)),
                    c = make_cubin(86, good_words(40, 2));
  const auto fillers = [](size_t n) {
    std::string out;
    for (size_t i = 0; i < n; ++i) {
      const okcache::Key k{0x1234567800000000ull + i, ~static_cast<uint64_t>(i)};
      out.append(reinterpret_cast<const char*>(&k), sizeof k);
    }
    return out;
  };
  const uint64_t h0 = sass::unsupported_cache_hits();
  {
    // Exactly full, the last key being a's: read in full, never appended to.
    const std::string root = fresh_root();
    const std::string file = plant_named(root, real_name(), fillers(kCap - 1) + key_bytes(a), 0700, 0600);
    expect_runs(a);
    VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 1);
    expect_runs(b);
    expect_runs(b);
    VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 1);
    VCHECK_EQ(size_of(file), static_cast<long>(kCap * 16));
  }
  {
    // One short of full: one more key fits, and then it is full.
    const std::string root = fresh_root();
    const std::string file = plant_named(root, real_name(), fillers(kCap - 1), 0700, 0600);
    expect_runs(a);
    VCHECK_EQ(size_of(file), static_cast<long>(kCap * 16));
    expect_runs(b);
    expect_runs(c);
    VCHECK_EQ(size_of(file), static_cast<long>(kCap * 16));
    expect_runs(a);
    VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 2);
  }
  {
    // Longer than the cap (an older or foreign writer): the first 65,536 are read, no more, and it is not appended to.
    const std::string root = fresh_root();
    const std::string file = plant_named(root, real_name(), fillers(kCap - 1) + key_bytes(a) + key_bytes(b), 0700, 0600);
    const uint64_t base = sass::unsupported_cache_hits();
    expect_runs(a);
    expect_runs(b);
    expect_runs(c);
    VCHECK_EQ(sass::unsupported_cache_hits() - base, uint64_t{1});
    VCHECK_EQ(size_of(file), static_cast<long>((kCap + 1) * 16));
  }
}

VTEST_MAIN
