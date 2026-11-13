#include "vgpu/memory_backing.hpp"

#include <fcntl.h>
#include <malloc.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/memory.hpp"

// Under AddressSanitizer a chunk is a heap allocation, as it always was: the sanitizer puts red zones round it, which is how a write
// past a chunk's end is caught. Otherwise chunks are carved from arenas (see arena_chunk).
#if defined(__SANITIZE_ADDRESS__)
#define VGPU_HEAP_CHUNKS 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define VGPU_HEAP_CHUNKS 1
#endif
#endif

namespace vgpu::backing {
namespace {
std::atomic<uint64_t> g_madvise_dontneed{0};   // see kernel_calls()
std::atomic<uint64_t> g_arena_maps{0};

// Heap chunks come from arenas: mappings of kArenaBytes, aligned to the huge page, that the kernel is asked to back with huge pages
// (a program that fills gigabytes of device memory takes one page fault for each 2 MiB, not 512). A fresh mapping reads as zero.
constexpr uint64_t kHugePage = 2ull << 20;
constexpr uint64_t kArenaBytes = 64ull << 20;

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
  // Heap chunks whose pages went back to the kernel (madvise DONTNEED: they read as zero on the next touch and cost no memory until
  // then), ready to hand out; and where the chunks of the arena being carved go on from.
  std::vector<uint8_t*> cold;
  uint8_t* arena_next = nullptr;
  uint8_t* arena_end = nullptr;
  // Chunks release() is clearing outside the lock, on their way into the pool: they count against its room.
  uint64_t pool_pending = 0;

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

// Gives a heap chunk's memory back to the system, and the chunk goes on the cold list: it reads as zero when it is next touched.
// (Where the pages cannot be dropped, the chunk is cleared by hand.)
void make_cold(State& s, uint8_t* chunk) {
#ifdef VGPU_HEAP_CHUNKS
  (void)s;
  delete[] chunk;
#else
  g_madvise_dontneed.fetch_add(1, std::memory_order_relaxed);
  if (::madvise(chunk, kChunkSize, MADV_DONTNEED) != 0) std::memset(chunk, 0, kChunkSize);
  s.cold.push_back(chunk);
#endif
}

// Gives the pool's chunks back to the system; how many.
uint64_t drain_pool(State& s) {
  const uint64_t n = s.pool.size();
  for (uint8_t* c : s.pool) make_cold(s, c);
  s.pool.clear();
  s.pool.shrink_to_fit();
  return n;
}

// Looks at the host and, if it is short while the pool holds chunks, gives those back and looks again: the pool is
// memory kept for reuse, never a reason to refuse or spill an allocation the host could have served had the chunks been
// returned when they were released.
void look_at_host_with_pool(State& s) {
  look_at_host(s);
  if (s.low && !s.pool.empty()) {
    drain_pool(s);
    look_at_host(s);
  }
}

// Whether the host is short of memory, looked at once per host_check_every_chunks
// heap chunks (the first chunk always looks). A shortage is looked at again on
// the very next chunk when there is nowhere to spill, so that the program that
// catches the error and frees memory is not refused for 16 MiB more than it
// has to be.
bool host_is_short(State& s) {
  if (s.since_check == 0) look_at_host_with_pool(s);
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

#ifndef VGPU_HEAP_CHUNKS
// A zero chunk of the heap, the lock held: one that went back to the kernel, or the next of the arena, which is a fresh mapping.
uint8_t* arena_chunk(State& s) {
  if (!s.cold.empty()) {
    uint8_t* chunk = s.cold.back();
    s.cold.pop_back();
    return chunk;
  }
  if (s.arena_next == s.arena_end) {
    // (the head and tail the alignment leaves stay mapped and untouched: they cost no memory)
    g_arena_maps.fetch_add(1, std::memory_order_relaxed);
    void* m = ::mmap(nullptr, kArenaBytes + kHugePage, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (m == MAP_FAILED) {
      s.ram -= kChunkSize;
      throw std::bad_alloc();
    }
    const uintptr_t base = (reinterpret_cast<uintptr_t>(m) + kHugePage - 1) & ~(kHugePage - 1);
    ::madvise(reinterpret_cast<void*>(base), kArenaBytes, MADV_HUGEPAGE);   // (a kernel without huge pages says no, and nothing is lost)
    s.arena_next = reinterpret_cast<uint8_t*>(base);
    s.arena_end = s.arena_next + kArenaBytes;
  }
  uint8_t* chunk = s.arena_next;
  s.arena_next += kChunkSize;
  return chunk;
}
#endif

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
#ifdef VGPU_HEAP_CHUNKS
    lock.unlock();
    // Value-initialized: the chunk arrives zeroed, as untouched memory reads.
    return new uint8_t[kChunkSize]();
#else
    return arena_chunk(s);
#endif
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

// Whether the pool has room for one more chunk: it holds at most Config::pool_max_bytes, and where the RAM limit is set
// (it is what decides when chunks come from the spill file) no more than the limit leaves beside the chunks in use.
bool pool_has_room(const State& s) {
  uint64_t cap = s.config.pool_max_bytes;
  if (s.config.ram_limit_bytes != std::numeric_limits<uint64_t>::max())
    cap = std::min(cap, s.config.ram_limit_bytes > s.ram ? s.config.ram_limit_bytes - s.ram : uint64_t{0});
  return (s.pool.size() + s.pool_pending + 1) * kChunkSize <= cap;
}

// A heap chunk the pool has no room for: its pages go back to the kernel outside the lock (which that takes its time over), and it
// is handed out again, zero, from the cold list.
void give_back(State& s, uint8_t* chunk) {
#ifdef VGPU_HEAP_CHUNKS
  (void)s;
  delete[] chunk;
#else
  g_madvise_dontneed.fetch_add(1, std::memory_order_relaxed);
  if (::madvise(chunk, kChunkSize, MADV_DONTNEED) != 0) std::memset(chunk, 0, kChunkSize);
  std::lock_guard<std::mutex> lock(s.mu);
  s.cold.push_back(chunk);
#endif
}

void release(uint8_t* chunk, uint64_t dirty_bytes) {
  if (!chunk) return;
  State& s = state();
  bool keep = false;
  {
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
    keep = pool_has_room(s);
    if (keep) ++s.pool_pending;   // a place is held while the chunk is cleared, which is done without the lock
  }
  // (a chunk past the pool's room goes back to the heap outside the lock, which the allocator may take its time over)
  if (!keep) {
    give_back(s, chunk);
    return;
  }
  // Cleared to the byte promised: everything from there on is zero already.
  std::memset(chunk, 0, static_cast<size_t>(std::min<uint64_t>(dirty_bytes, kChunkSize)));
  {
    std::lock_guard<std::mutex> lock(s.mu);
    --s.pool_pending;
    // (the limits may have moved while the chunk was being cleared)
    if (pool_has_room(s)) {
      s.pool.push_back(chunk);
      return;
    }
  }
  give_back(s, chunk);
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

KernelCalls kernel_calls() {
  return {g_madvise_dontneed.load(std::memory_order_relaxed), g_arena_maps.load(std::memory_order_relaxed)};
}

namespace {
std::atomic<bool> g_heap_tuned{false};
}

void tune_host_heap() {
  static std::once_flag once;
  std::call_once(once, [] {
#if defined(__GLIBC__) && !defined(VGPU_HEAP_CHUNKS) && !defined(__SANITIZE_THREAD__)
    if (const char* e = std::getenv("VGPU_HOST_HEAP"); e && e[0] == '0' && e[1] == 0) return;
    for (const char* v : {"MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_", "MALLOC_TOP_PAD_"})
      if (std::getenv(v)) return;
    if (const char* t = std::getenv("GLIBC_TUNABLES"))
      for (const char* k : {"glibc.malloc.mmap_threshold", "glibc.malloc.trim_threshold", "glibc.malloc.top_pad"})
        if (std::strstr(t, k)) return;
    const bool ok = ::mallopt(M_MMAP_THRESHOLD, 32 << 20) == 1 && ::mallopt(M_TRIM_THRESHOLD, 64 << 20) == 1;
    g_heap_tuned.store(ok, std::memory_order_relaxed);
#endif
  });
}

bool host_heap_tuned() { return g_heap_tuned.load(std::memory_order_relaxed); }

}  // namespace vgpu::backing
