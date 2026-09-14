#include "vgpu/memory_backing.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
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
  uint64_t file_size = 0;
  std::map<uint8_t*, uint64_t> slices;  // mapping base -> offset in the file
  std::vector<uint8_t*> free_file;      // file chunks ready to hand out, zero
};

// Never destroyed: device memory can be released during static destruction,
// after a State with a destructor would already be gone.
State& state() {
  static State* s = new State;
  return *s;
}

void configure_from_env(State& s) {
  if (s.configured) return;
  s.configured = true;
  if (const char* dir = std::getenv("VGPU_MEMORY_DIR"); dir && *dir) s.config.spill_dir = dir;
  if (const char* mb = std::getenv("VGPU_MEMORY_RAM_MB"); mb && *mb) {
    char* end = nullptr;
    const unsigned long long v = std::strtoull(mb, &end, 10);
    if (end != mb && *end == '\0') s.config.ram_limit_bytes = static_cast<uint64_t>(v) * 1024 * 1024;
  }
}

// The file offset of a chunk, if the chunk is in the spill file.
bool file_offset(const State& s, uint8_t* chunk, uint64_t* off) {
  auto it = s.slices.upper_bound(chunk);
  if (it == s.slices.begin()) return false;
  --it;
  if (chunk >= it->first + kSliceBytes) return false;
  *off = it->second + static_cast<uint64_t>(chunk - it->first);
  return true;
}

[[noreturn]] void cannot_spill(const State& s, const char* what) {
  const int err = errno;
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
  // A spill file with nothing in it belongs to the old configuration: close it,
  // so the next spill uses the directory just given rather than the last one.
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
  const bool spill = !s.config.spill_dir.empty() && s.ram + kChunkSize > s.config.ram_limit_bytes;
  if (!spill) {
    s.ram += kChunkSize;
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

void release(uint8_t* chunk) {
  if (!chunk) return;
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mu);
  uint64_t off = 0;
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
  delete[] chunk;
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

}  // namespace vgpu::backing
