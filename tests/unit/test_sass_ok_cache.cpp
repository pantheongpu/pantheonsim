// sass::unsupported leaves a file of the cubins it found to run in full (src/sass/ok_cache.cpp), so the next process
// that loads the same fatbins does not decode every word of them again. The file must change nothing a caller can see:
// a refusal is never recorded and keeps its text, VGPU_SASS_REFUSE and VGPU_SASS_CACHE=0 or unset (the default) leave it alone, a different
// image is a miss (every bit of the image is in the key), and a file that is not this user's alone, or has a torn
// tail, is not trusted or not appended to. Each test below fails if the cache answered for an image it was not given,
// recorded a refusal, read a file anyone could have written, or appended behind a torn key.
//
// The cubins are built here (one .text.k section) from words found in the corpora of nvidia/tests/data/sass; the
// cache directory is a fresh one under TMPDIR per scenario (XDG_CACHE_HOME), so nothing of the user's is touched.
#include <dirent.h>
#include <ftw.h>
#include <sys/stat.h>
#include <unistd.h>

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
  setenv("VGPU_SASS_CACHE", "1", 1);
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
bool has_cache(const std::string& root) {
  const std::string probe = make_cubin(86, good_words(6, 200));
  VCHECK(run(probe).empty());
  return !cache_file(root).empty();
}

// A file under `root` holding the keys of `cubins` and `tail` extra bytes, with the modes given.
std::string plant(const std::string& root, const std::vector<std::string>& cubins, size_t tail, mode_t dir_mode,
                  mode_t file_mode) {
  const std::string real = [&] {
    const std::string r2 = fresh_root();
    has_cache(r2);
    const std::string f = cache_file(r2);
    return f.empty() ? f : f.substr(f.rfind('/') + 1);
  }();
  if (real.empty()) return {};
  setenv("XDG_CACHE_HOME", root.c_str(), 1);
  const std::string dir = root + "/pantheonsim";
  mkdir(dir.c_str(), 0700);
  chmod(dir.c_str(), dir_mode);
  const std::string path = dir + "/" + real;
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    for (const std::string& c : cubins) {
      const okcache::Key k = okcache::key_of(reinterpret_cast<const uint8_t*>(c.data()), c.size());
      out.write(reinterpret_cast<const char*>(&k), sizeof k);
    }
    out << std::string(tail, 'x');
  }
  chmod(path.c_str(), file_mode);
  return path;
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
  // VGPU_SASS_CACHE=0, or unset (the default): neither read nor written.
  const std::string b = make_cubin(86, good_words(40, 2));
  for (const char* off : {"0", nullptr}) {
    if (off) setenv("VGPU_SASS_CACHE", off, 1);
    else unsetenv("VGPU_SASS_CACHE");
    VCHECK(run(a).empty());
    VCHECK_EQ(sass::unsupported_cache_hits(), h0 + 1);
    VCHECK(run(b).empty());
    VCHECK_EQ(size_of(file), 16L);
  }
  setenv("VGPU_SASS_CACHE", "1", 1);
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

VTEST_MAIN
