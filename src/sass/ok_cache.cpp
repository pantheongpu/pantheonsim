// What sass::unsupported decides is a function of the cubin's bytes and of the decoder and executor that read them, so a
// process that finds a cubin runs in full can leave that fact for the next one: a PyTorch sweep starts a dozen
// processes that load the same fatbins, and each decoded every word of them again. The file holds 16-byte keys (a
// 128-bit hash of the image) of images that ran, one file per build of the simulator (named by the GNU build id of the
// library this code is linked into, so a rebuilt decoder never reads what an older one wrote, and a build without an id
// has no cache). Only passes are recorded: a refusal is found by the ordinary path, text and all.
//
// The file is trusted only if it is a regular file of this user's that nobody else can write, in a directory of the
// same kind; a short tail (a torn append) is ignored and never appended to. It is opt-in: `VGPU_SASS_CACHE=1` turns it on.
#include "ok_cache.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#if defined(__linux__) && defined(__GLIBC__)
#include <elf.h>
#include <link.h>
#define VGPU_OKCACHE_BUILD_ID 1
#endif

namespace vgpu::sass::okcache {

namespace {

constexpr uint64_t kP1 = 0x9e3779b185ebca87ull, kP2 = 0xc2b2ae3d27d4eb4full, kP3 = 0x165667b19e3779f9ull;
constexpr size_t kMaxKeys = size_t{1} << 16;   // 1 MB of keys; a full file is read, not appended to

inline uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
inline uint64_t round1(uint64_t v, uint64_t w) { return rotl(v + w * kP2, 31) * kP1; }
inline uint64_t fmix(uint64_t h) {
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdull;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ull;
  h ^= h >> 33;
  return h;
}

struct KeyHash {
  size_t operator()(const Key& k) const { return static_cast<size_t>(k.a); }
};
struct KeyEq {
  bool operator()(const Key& x, const Key& y) const { return x.a == y.a && x.b == y.b; }
};

struct State {
  std::mutex mu;
  std::string path;   // the file the fields below describe ("" = none yet)
  bool trusted = false;
  bool appendable = false;   // the file is whole (a multiple of 16 bytes) and has room
  std::unordered_set<Key, KeyHash, KeyEq> keys;
};
State& state() {
  static State s;
  return s;
}
std::atomic<uint64_t> g_hits{0};

#ifdef VGPU_OKCACHE_BUILD_ID
struct IdSearch {
  uintptr_t where;
  std::string id;
};
int find_id(struct dl_phdr_info* info, size_t, void* data) {
  IdSearch& s = *static_cast<IdSearch*>(data);
  bool inside = false;
  for (int i = 0; i < info->dlpi_phnum; ++i) {
    const ElfW(Phdr)& p = info->dlpi_phdr[i];
    const uintptr_t lo = info->dlpi_addr + p.p_vaddr;
    if (p.p_type == PT_LOAD && s.where >= lo && s.where < lo + p.p_memsz) inside = true;
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
      const size_t name = (n.n_namesz + 3u) & ~3u, desc = (n.n_descsz + 3u) & ~3u;
      if (static_cast<size_t>(end - at) < sizeof n + name + desc) break;
      if (n.n_type == NT_GNU_BUILD_ID && n.n_namesz == 4 && std::memcmp(at + sizeof n, "GNU", 4) == 0 && n.n_descsz > 0) {
        static const char hex[] = "0123456789abcdef";
        const unsigned char* d = at + sizeof n + name;
        for (uint32_t k = 0; k < n.n_descsz && k < 32; ++k) {
          s.id += hex[d[k] >> 4];
          s.id += hex[d[k] & 15];
        }
        return 1;
      }
      at += sizeof n + name + desc;
    }
  }
  return 1;   // the object that holds this code, with no id
}
#endif

// The GNU build id of the object this code is in ("" when it has none).
const std::string& build_id() {
  static const std::string id = [] {
#ifdef VGPU_OKCACHE_BUILD_ID
    IdSearch s{reinterpret_cast<uintptr_t>(&hits), {}};
    dl_iterate_phdr(find_id, &s);
    return s.id;
#else
    return std::string();
#endif
  }();
  return id;
}

std::string cache_root() {
  if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x) return x;
  if (const char* h = std::getenv("HOME"); h && *h) return std::string(h) + "/.cache";
  return {};
}

// "" when the cache is off: it is opt-in, `VGPU_SASS_CACHE=1`.
std::string cache_path() {
  if (const char* v = std::getenv("VGPU_SASS_CACHE"); !v || v[0] != '1') return {};
  const std::string& id = build_id();
  const std::string root = cache_root();
  if (id.empty() || root.empty()) return {};
  return root + "/pantheonsim/sass-ok-v1-" + id;
}

bool ours_alone(const struct stat& st, mode_t kind) {
  return (st.st_mode & S_IFMT) == kind && st.st_uid == geteuid() && (st.st_mode & 022) == 0;
}

// The directory of `path` exists and is this user's and nobody else's to write; made (0700, with its parent) when `make`.
bool dir_ok(const std::string& path, bool make) {
  const std::string dir = path.substr(0, path.rfind('/'));
  struct stat st;
  if (lstat(dir.c_str(), &st) != 0) {
    if (!make) return false;
    const std::string parent = dir.substr(0, dir.rfind('/'));
    if (mkdir(parent.c_str(), 0700) != 0 && lstat(parent.c_str(), &st) != 0) return false;
    if (mkdir(dir.c_str(), 0700) != 0 && lstat(dir.c_str(), &st) != 0) return false;
    if (lstat(dir.c_str(), &st) != 0) return false;
  }
  return ours_alone(st, S_IFDIR);
}

void load(State& s, const std::string& path) {
  s.path = path;
  s.trusted = false;
  s.appendable = false;
  s.keys.clear();
  if (path.empty() || !dir_ok(path, false)) return;
  const int fd = open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) return;
  struct stat st;
  if (fstat(fd, &st) == 0 && ours_alone(st, S_IFREG)) {
    s.trusted = true;
    const size_t whole = static_cast<size_t>(st.st_size) / 16;
    const size_t n = whole < kMaxKeys ? whole : kMaxKeys;
    std::vector<Key> buf(n);
    size_t got = 0;
    while (got < n * sizeof(Key)) {
      const ssize_t r = read(fd, reinterpret_cast<char*>(buf.data()) + got, n * sizeof(Key) - got);
      if (r <= 0) break;
      got += static_cast<size_t>(r);
    }
    for (size_t i = 0; i < got / sizeof(Key); ++i) s.keys.insert(buf[i]);
    s.appendable = st.st_size % 16 == 0 && whole < kMaxKeys;
  }
  close(fd);
}

void refresh(State& s, const std::string& path) {
  if (path != s.path) load(s, path);
}

}  // namespace

Key key_of(const uint8_t* p, size_t n) {
  uint64_t v[4] = {kP1 + kP2, kP2, 0, 0 - kP1};
  const uint8_t* const end = p + n;
  uint64_t w[4];
  while (static_cast<size_t>(end - p) >= 32) {
    std::memcpy(w, p, 32);
    for (int i = 0; i < 4; ++i) v[i] = round1(v[i], w[i]);
    p += 32;
  }
  uint64_t a = rotl(v[0], 1) + rotl(v[1], 7) + rotl(v[2], 12) + rotl(v[3], 18);
  uint64_t b = (v[0] ^ rotl(v[1], 13)) + (v[2] ^ rotl(v[3], 29)) * kP3;
  while (static_cast<size_t>(end - p) >= 8) {
    uint64_t x;
    std::memcpy(&x, p, 8);
    a = rotl(a ^ round1(0, x), 27) * kP1 + kP3;
    b = rotl(b + x * kP3, 23) * kP2 + kP1;
    p += 8;
  }
  for (; p < end; ++p) {
    a = rotl(a ^ (*p * kP3), 11) * kP1;
    b = rotl(b + (*p * kP1), 17) * kP2;
  }
  Key k;
  k.a = fmix(a ^ static_cast<uint64_t>(n));
  k.b = fmix(b + static_cast<uint64_t>(n) * kP2 + k.a);
  return k;
}

bool known(const Key& k) {
  const std::string path = cache_path();
  if (path.empty()) return false;
  State& s = state();
  std::lock_guard<std::mutex> g(s.mu);
  refresh(s, path);
  const bool yes = s.trusted && s.keys.count(k) != 0;
  if (yes) g_hits.fetch_add(1, std::memory_order_relaxed);
  return yes;
}

void remember(const Key& k) {
  const std::string path = cache_path();
  if (path.empty()) return;
  State& s = state();
  std::lock_guard<std::mutex> g(s.mu);
  refresh(s, path);
  if (s.keys.count(k) != 0) return;
  // Not a file that exists but is not ours alone; one that is whole and has room; or none yet.
  if (s.trusted ? !s.appendable : false) return;
  if (!s.trusted) {
    struct stat st;
    if (lstat(path.c_str(), &st) == 0) return;   // there, and load() did not trust it
    if (!dir_ok(path, true)) return;
  }
  const int fd = open(path.c_str(), O_WRONLY | O_APPEND | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0600);
  if (fd < 0) return;
  struct stat st;
  const bool fine = fstat(fd, &st) == 0 && ours_alone(st, S_IFREG) && st.st_size % 16 == 0 &&
                    static_cast<size_t>(st.st_size) / 16 < kMaxKeys;
  if (fine && write(fd, &k, sizeof k) == static_cast<ssize_t>(sizeof k)) {
    s.keys.insert(k);
    s.trusted = true;
    s.appendable = true;
  }
  close(fd);
}

uint64_t hits() { return g_hits.load(std::memory_order_relaxed); }

}  // namespace vgpu::sass::okcache
