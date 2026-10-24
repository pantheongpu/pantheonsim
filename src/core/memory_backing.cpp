#include "vgpu/memory_backing.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/memory.hpp"

namespace vgpu::backing {
namespace {

// The spill file grows a slice at a time: one fallocate and one mmap per
// 256 MiB rather than per chunk.
constexpr uint64_t kSliceChunks = 4096;
constexpr uint64_t kSliceBytes = kSliceChunks * kChunkSize;

struct State {
  std::mutex mu;
  bool configured = false;
  Config config;
  uint64_t ram = 0;
  uint64_t spill = 0;
  int fd = -1;
  pid_t owner = 0;                      // the process that created the spill file
  uint64_t file_size = 0;
  std::map<uint8_t*, uint64_t> slices;  // mapping base -> offset in the file
  std::vector<uint8_t*> free_file;      // file chunks ready to hand out, zero
  // Slices a forked child inherited from its parent: still mapped, never
  // released or handed out again here.
  std::map<uint8_t*, uint64_t> inherited;
  // Heap chunks given back nearly untouched, all zero, ready to hand out. They count as released in
  // `ram`; acquire() counts them again.
  std::vector<uint8_t*> pool;

  // The host-memory guard. since_check counts heap chunks since the last look
  // at the host; zero means the next heap chunk looks. low is what the last
  // look found, and last says why, for the message.
  uint64_t since_check = 0;
  uint64_t samples = 0;
  bool low = false;
  bool spilling_for_host = false;   // this chunk is in the file because the host is short
  struct Look {
    bool have_available = false, have_cgroup = false;
    uint64_t available = 0;        // MemAvailable
    uint64_t cgroup_headroom = 0;  // what the tightest cgroup has left
    uint64_t headroom = 0;         // the smaller of the two
    uint64_t reserve = 0;
  } last;
};

// Never destroyed: device memory can be released during static destruction,
// after a State with a destructor would already be gone.
State& state() {
  static State* s = new State;
  return *s;
}

// ---- The host-memory guard ----------------------------------------------------
//
// Reads a small file into a caller's buffer: no heap, so a guard that runs
// because memory is short never asks for any. Returns false when it cannot be
// read.
bool read_small(const std::string& path, char* buf, size_t cap) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  size_t n = 0;
  while (n + 1 < cap) {
    const ssize_t r = ::read(fd, buf + n, cap - 1 - n);
    if (r <= 0) break;
    n += static_cast<size_t>(r);
  }
  ::close(fd);
  buf[n] = '\0';
  return true;
}

// The number after `key` on the line that starts with it ("MemAvailable:   12 kB",
// "inactive_file 12"). A kB suffix is scaled to bytes.
bool find_value(const char* text, const char* key, uint64_t* out) {
  const size_t klen = std::strlen(key);
  for (const char* line = text; line && *line;) {
    if (std::strncmp(line, key, klen) == 0 && (line[klen] == ':' || line[klen] == ' ')) {
      const char* p = line + klen + 1;
      while (*p == ' ' || *p == '\t') ++p;
      char* end = nullptr;
      const unsigned long long v = std::strtoull(p, &end, 10);
      if (end == p) return false;
      while (*end == ' ') ++end;
      *out = static_cast<uint64_t>(v) * (std::strncmp(end, "kB", 2) == 0 ? 1024 : 1);
      return true;
    }
    line = std::strchr(line, '\n');
    if (line) ++line;
  }
  return false;
}

constexpr uint64_t kUnlimitedV1 = 1ull << 60;  // v1 says "no limit" as a number near 2^63

// One cgroup directory's say in the matter: its limit and what is used, where
// file cache the kernel can drop (it does, before it kills anything) is not
// counted as used -- a container that has read a lot of files has its
// memory.current at the limit without being short of anything.
void cgroup_dir(const std::string& dir, bool v2, State::Look& look, uint64_t* limit_seen) {
  char buf[8192];
  uint64_t limit = 0, used = 0;
  if (!read_small(dir + (v2 ? "/memory.max" : "/memory.limit_in_bytes"), buf, sizeof buf)) return;
  if (std::strncmp(buf, "max", 3) == 0) return;
  char* end = nullptr;
  const unsigned long long l = std::strtoull(buf, &end, 10);
  if (end == buf || l >= kUnlimitedV1) return;
  limit = l;
  if (!read_small(dir + (v2 ? "/memory.current" : "/memory.usage_in_bytes"), buf, sizeof buf)) return;
  const unsigned long long u = std::strtoull(buf, &end, 10);
  if (end == buf) return;
  used = u;
  if (read_small(dir + "/memory.stat", buf, sizeof buf)) {
    uint64_t inactive = 0, active = 0;
    find_value(buf, v2 ? "inactive_file" : "total_inactive_file", &inactive);
    find_value(buf, v2 ? "active_file" : "total_active_file", &active);
    const uint64_t cache = inactive + active;
    used = used > cache ? used - cache : 0;
  }
  const uint64_t headroom = limit > used ? limit - used : 0;
  look.cgroup_headroom = look.have_cgroup ? std::min(look.cgroup_headroom, headroom) : headroom;
  look.have_cgroup = true;
  *limit_seen = std::min(*limit_seen, limit);
}

// This process's cgroup and every ancestor, as far up as the root the
// configuration names.
void cgroup_walk(const std::string& root, std::string path, bool v2, State::Look& look, uint64_t* limit_seen) {
  if (const size_t sp = path.find(" (deleted)"); sp != std::string::npos) path.erase(sp);
  if (path.empty() || path[0] != '/') path = "/" + path;
  for (;;) {
    cgroup_dir(root + path, v2, look, limit_seen);
    if (path == "/") break;
    const size_t slash = path.rfind('/');
    path = slash == 0 ? "/" : path.substr(0, slash);
  }
}

// Looks at the host: what it can still give this process, the reserve to leave
// it, and the memory limit in play that the default reserve is a share of.
void look_at_host(State& s) {
  const Config& c = s.config;
  State::Look look;
  ++s.samples;
  char buf[8192];
  uint64_t total = std::numeric_limits<uint64_t>::max();
  if (read_small(c.meminfo_path, buf, sizeof buf)) {
    if (find_value(buf, "MemAvailable", &look.available)) look.have_available = true;
    uint64_t t = 0;
    if (find_value(buf, "MemTotal", &t) && t) total = t;
  }
  uint64_t limit_seen = std::numeric_limits<uint64_t>::max();
  // The cgroup the process is in: v2 has one line, "0::/path"; v1 has a line
  // per controller, "N:memory:/path" for this one.
  std::string v2_path = "/", v1_path;
  bool have_v2 = false, have_v1 = false;
  if (read_small(c.cgroup_self, buf, sizeof buf)) {
    for (const char* line = buf; line && *line;) {
      const char* c1 = std::strchr(line, ':');
      const char* c2 = c1 ? std::strchr(c1 + 1, ':') : nullptr;
      const char* eol = std::strchr(line, '\n');
      if (c2 && (!eol || c2 < eol)) {
        const std::string controllers(c1 + 1, static_cast<size_t>(c2 - c1 - 1));
        const std::string path = eol ? std::string(c2 + 1, static_cast<size_t>(eol - c2 - 1)) : std::string(c2 + 1);
        if (controllers.empty()) {
          v2_path = path;
          have_v2 = true;
        } else if (("," + controllers + ",").find(",memory,") != std::string::npos) {
          v1_path = path;
          have_v1 = true;
        }
      }
      line = eol ? eol + 1 : nullptr;
    }
  }
  cgroup_walk(c.cgroup_root, v2_path, true, look, &limit_seen);
  if (!look.have_cgroup && (have_v1 || !have_v2)) cgroup_walk(c.cgroup_root + "/memory", v1_path, false, look, &limit_seen);

  if (look.have_available) look.headroom = look.available;
  if (look.have_cgroup && (!look.have_available || look.cgroup_headroom < look.headroom))
    look.headroom = look.cgroup_headroom;

  if (c.host_reserve_bytes != kAutoReserve) {
    look.reserve = c.host_reserve_bytes;
  } else {
    const uint64_t in_play = std::min(total, limit_seen);
    constexpr uint64_t kGiB = 1ull << 30;
    // 6% of what is in play, at least 1 GiB and at most 4 GiB; a small
    // container is not asked to give up more than a quarter of itself.
    uint64_t r = std::clamp<uint64_t>(in_play == std::numeric_limits<uint64_t>::max() ? kGiB : in_play / 100 * 6,
                                      kGiB, 4 * kGiB);
    if (in_play != std::numeric_limits<uint64_t>::max()) r = std::min(r, in_play / 4);
    look.reserve = r;
  }
  s.last = look;
  // Nothing readable is no evidence of a shortage.
  s.low = (look.have_available || look.have_cgroup) && look.headroom < look.reserve;
}

// Whether the host is short of memory, looked at once per host_check_every_chunks
// heap chunks (the first chunk always looks). A shortage is looked at again on
// the very next chunk when there is nowhere to spill, so that the program that
// catches the error and frees memory is not refused for 16 MiB more than it
// has to be.
bool host_is_short(State& s) {
  if (s.since_check == 0) look_at_host(s);
  const uint64_t every = std::max<uint64_t>(1, s.config.host_check_every_chunks);
  if (++s.since_check >= every || (s.low && s.config.spill_dir.empty())) s.since_check = 0;
  return s.low;
}

[[noreturn]] void host_out_of_memory(const State& s) {
  const State::Look& l = s.last;
  constexpr uint64_t kMB = 1024 * 1024;
  std::string have;
  if (l.have_available) have += "MemAvailable " + std::to_string(l.available / kMB) + " MB";
  if (l.have_cgroup) have += (have.empty() ? "" : ", ") + std::string("cgroup headroom ") + std::to_string(l.cgroup_headroom / kMB) + " MB";
  throw Error::make(Err::OutOfMemory, "device out of memory: the host has ", l.headroom / kMB,
                    " MB of memory left (", have, "), below the reserve of ", l.reserve / kMB,
                    " MB that the simulator keeps for the rest of the machine. Set VGPU_MEMORY_DIR (with "
                    "VGPU_MEMORY_RAM_MB) to let device memory spill to disk, VGPU_MEMORY_HOST_RESERVE_MB to "
                    "change the reserve, or VGPU_MEMORY_HOST_GUARD=0 to turn this check off.");
}

}  // namespace

Config config_from_env() {
  Config c;
  if (const char* dir = std::getenv("VGPU_MEMORY_DIR"); dir && *dir) c.spill_dir = dir;
  const auto mb = [](const char* name, uint64_t* out) {
    const char* v = std::getenv(name);
    if (!v || !*v) return false;
    char* end = nullptr;
    const unsigned long long n = std::strtoull(v, &end, 10);
    if (end == v || *end != '\0') return false;
    *out = static_cast<uint64_t>(n) * 1024 * 1024;
    return true;
  };
  mb("VGPU_MEMORY_RAM_MB", &c.ram_limit_bytes);
  mb("VGPU_MEMORY_HOST_RESERVE_MB", &c.host_reserve_bytes);
  if (const char* g = std::getenv("VGPU_MEMORY_HOST_GUARD"); g && g[0] == '0' && g[1] == '\0') c.host_guard = false;
  return c;
}

namespace {

void configure_from_env(State& s) {
  if (s.configured) return;
  s.configured = true;
  s.config = config_from_env();
}

// The file offset of a chunk, if the chunk is in one of these slices.
bool in_slices(const std::map<uint8_t*, uint64_t>& slices, uint8_t* chunk, uint64_t* off) {
  auto it = slices.upper_bound(chunk);
  if (it == slices.begin()) return false;
  --it;
  if (chunk >= it->first + kSliceBytes) return false;
  *off = it->second + static_cast<uint64_t>(chunk - it->first);
  return true;
}
bool file_offset(const State& s, uint8_t* chunk, uint64_t* off) { return in_slices(s.slices, chunk, off); }

// A forked child shares the spill file's pages with its parent -- a shared
// mapping is the point of the file. Freeing a chunk there punched a hole in the
// parent's device memory, which read back as zeros, and a free chunk handed
// out in both processes would be the same bytes twice. So the first time a
// child touches the backing, everything it inherited is set aside: those
// chunks are left alone (the parent owns them), and the child spills into a
// file of its own. CUDA is not usable in a forked child anyway; this is about
// a child that frees memory or simply exits and runs its destructors.
void adopt_after_fork(State& s) {
  if (s.fd < 0 || s.owner == ::getpid()) return;
  s.inherited.insert(s.slices.begin(), s.slices.end());
  s.slices.clear();
  s.free_file.clear();
  ::close(s.fd);
  s.fd = -1;
  s.file_size = 0;
  s.spill = 0;
}

[[noreturn]] void cannot_spill(const State& s, const char* what) {
  const int err = errno;
  if (s.spilling_for_host) {
    constexpr uint64_t kMB = 1024 * 1024;
    throw Error::make(Err::OutOfMemory, "device out of memory: the host has ", s.last.headroom / kMB,
                      " MB of memory left, below the reserve of ", s.last.reserve / kMB, " MB, and ", what,
                      " in ", s.config.spill_dir, ": ", std::strerror(err));
  }
  throw Error::make(Err::OutOfMemory, "device memory needs more than VGPU_MEMORY_RAM_MB (",
                    s.config.ram_limit_bytes / (1024 * 1024), " MB) of host RAM and ", what, " in ",
                    s.config.spill_dir, ": ", std::strerror(err));
}

void grow(State& s) {
  if (s.fd < 0) {
    // Unnamed, so nothing is left on disk if the process dies. A filesystem
    // without O_TMPFILE gets a named file that is unlinked at once instead.
    s.fd = ::open(s.config.spill_dir.c_str(), O_TMPFILE | O_RDWR | O_CLOEXEC, 0600);
    if (s.fd < 0) {
      std::string name = s.config.spill_dir + "/vgpu-memory-XXXXXX";
      s.fd = ::mkostemp(name.data(), O_CLOEXEC);
      if (s.fd >= 0) ::unlink(name.c_str());
    }
    if (s.fd < 0) cannot_spill(s, "the spill file could not be created");
    s.owner = ::getpid();
  }
  // Reserved, not merely sized: a write into an unreserved page of a shared
  // mapping on a full disk is a SIGBUS, which would take the program down
  // with no diagnosis.
  if (::fallocate(s.fd, 0, static_cast<off_t>(s.file_size), static_cast<off_t>(kSliceBytes)) != 0)
    cannot_spill(s, "there is no room for it");
  void* m = ::mmap(nullptr, kSliceBytes, PROT_READ | PROT_WRITE, MAP_SHARED, s.fd,
                   static_cast<off_t>(s.file_size));
  if (m == MAP_FAILED) cannot_spill(s, "the spill file could not be mapped");
  const auto base = static_cast<uint8_t*>(m);
  s.slices[base] = s.file_size;
  s.file_size += kSliceBytes;
  s.free_file.reserve(s.free_file.size() + kSliceChunks);
  for (uint64_t i = kSliceChunks; i-- > 0;) s.free_file.push_back(base + i * kChunkSize);
}

}  // namespace

void configure(const Config& config) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mu);
  s.configured = true;
  s.config = config;
  s.since_check = 0;   // the next heap chunk looks at the host afresh
  s.low = false;
  // A spill file with nothing in it belongs to the old configuration: close it,
  // so the next spill uses the directory just given rather than the last one.
  adopt_after_fork(s);
  if (s.fd >= 0 && s.spill == 0) {
    for (const auto& [base, off] : s.slices) ::munmap(base, kSliceBytes);
    s.slices.clear();
    s.free_file.clear();
    ::close(s.fd);
    s.fd = -1;
    s.file_size = 0;
  }
}

uint8_t* acquire() {
  State& s = state();
  std::unique_lock<std::mutex> lock(s.mu);
  configure_from_env(s);
  adopt_after_fork(s);
  bool spill = !s.config.spill_dir.empty() && s.ram + kChunkSize > s.config.ram_limit_bytes;
  s.spilling_for_host = false;
  // A heap chunk is real memory the host may not have. Past the RAM limit the
  // chunk is a file's, and the guard has nothing to say.
  if (!spill && s.config.host_guard && host_is_short(s)) {
    if (s.config.spill_dir.empty()) host_out_of_memory(s);
    spill = s.spilling_for_host = true;
  }
  if (!spill) {
    s.ram += kChunkSize;
    if (!s.pool.empty()) {
      uint8_t* chunk = s.pool.back();
      s.pool.pop_back();
      return chunk;
    }
    lock.unlock();
    // Value-initialized: the chunk arrives zeroed, as untouched memory reads.
    return new uint8_t[kChunkSize]();
  }
  if (s.free_file.empty()) grow(s);
  uint8_t* chunk = s.free_file.back();
  uint64_t off = 0;
  file_offset(s, chunk, &off);
  // A chunk that was given back had its blocks released; take them again
  // before anyone writes to it, for the same SIGBUS reason as in grow().
  if (::fallocate(s.fd, 0, static_cast<off_t>(off), static_cast<off_t>(kChunkSize)) != 0)
    cannot_spill(s, "there is no room for it");
  s.free_file.pop_back();
  s.spill += kChunkSize;
  return chunk;
}

// How many chunks are kept. (What may be cleared under the lock to keep one is kPoolMaxDirty, in the header.)
constexpr size_t kPoolMax = 64;

void release(uint8_t* chunk, uint64_t dirty_bytes) {
  if (!chunk) return;
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mu);
  adopt_after_fork(s);
  uint64_t off = 0;
  if (in_slices(s.inherited, chunk, &off)) return;  // the parent's; see adopt_after_fork
  if (s.fd >= 0 && file_offset(s, chunk, &off)) {
    // The hole returns the disk space and makes the chunk read as zero when it
    // is handed out again. Where holes are not supported, zero it by hand.
    if (::fallocate(s.fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, static_cast<off_t>(off),
                    static_cast<off_t>(kChunkSize)) != 0)
      std::memset(chunk, 0, kChunkSize);
    s.free_file.push_back(chunk);
    s.spill -= kChunkSize;
    return;
  }
  s.ram -= kChunkSize;
  if (dirty_bytes <= kPoolMaxDirty && s.pool.size() < kPoolMax) {
    std::memset(chunk, 0, static_cast<size_t>(dirty_bytes));
    s.pool.push_back(chunk);
    return;
  }
  delete[] chunk;
}

uint64_t pooled_chunks() {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mu);
  return s.pool.size();
}

uint64_t ram_bytes() {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mu);
  return s.ram;
}

uint64_t spill_bytes() {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mu);
  return s.spill;
}

uint64_t host_guard_samples() {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mu);
  return s.samples;
}

}  // namespace vgpu::backing
