#include "vgpu/shared_state.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <filesystem>

#include "vgpu/error.hpp"

namespace vgpu {
namespace {
struct Header {
  uint32_t magic;
  uint32_t version;
  uint64_t reserved;
};
}  // namespace

SharedState::SharedState(const std::string& path, bool create, uint32_t magic, uint32_t version,
                         size_t payload_bytes, const char* what) {
  if (create) {
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
  }
  fd_ = ::open(path.c_str(), create ? (O_RDWR | O_CREAT) : O_RDWR, 0600);
  if (fd_ < 0) {
    if (!create && errno == ENOENT) return;
    throw Error::make(Err::Internal, "cannot open ", what, " ", path, ": ", std::strerror(errno));
  }
  map_bytes_ = sizeof(Header) + payload_bytes;
  struct stat st {};
  if (::fstat(fd_, &st) == 0 && st.st_size < static_cast<off_t>(map_bytes_) &&
      ::ftruncate(fd_, static_cast<off_t>(map_bytes_)) != 0)
    throw Error::make(Err::Internal, "cannot size ", what, " ", path, ": ", std::strerror(errno));
  void* p = ::mmap(nullptr, map_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (p == MAP_FAILED)
    throw Error::make(Err::Internal, "cannot map ", what, " ", path, ": ", std::strerror(errno));
  map_ = p;
  auto* h = static_cast<Header*>(p);
  payload_ = h + 1;
  words_ = payload_bytes / sizeof(uint64_t);
  // A new file is all zeros, and the first process to see it claims it. Two
  // processes creating it at once must not both clear it -- the second would
  // erase the first one's change -- hence the compare-and-swap.
  uint32_t expected = 0;
  if (__atomic_compare_exchange_n(&h->magic, &expected, magic, false, __ATOMIC_ACQ_REL,
                                  __ATOMIC_ACQUIRE)) {
    __atomic_store_n(&h->version, version, __ATOMIC_RELEASE);
  } else if (expected != magic || __atomic_load_n(&h->version, __ATOMIC_ACQUIRE) != version) {
    for (size_t i = 0; i < words_; ++i) __atomic_store_n(&words()[i], uint64_t{0}, __ATOMIC_RELAXED);
    __atomic_store_n(&h->version, version, __ATOMIC_RELEASE);
    __atomic_store_n(&h->magic, magic, __ATOMIC_RELEASE);
  }
}

SharedState::~SharedState() {
  if (map_) ::munmap(map_, map_bytes_);
  if (fd_ >= 0) ::close(fd_);
}

}  // namespace vgpu
