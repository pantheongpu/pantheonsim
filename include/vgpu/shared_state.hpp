// A state file mapped shared by every process on the machine, so all of them
// see and change one copy: the reliability counts (ras.hpp), the register
// state (regs.hpp). Values in it are read and written a word at a time with
// atomics, since other processes change them concurrently.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace vgpu {

class SharedState {
 public:
  // Maps `payload_bytes` (a multiple of 8) of state kept at `path`, after a
  // small header. With `create` the file and its directory are made as needed;
  // without, a file that does not exist maps to nothing and payload() is null.
  // A file from another build -- another magic or version -- is zeroed and
  // started again rather than misread. `what` names the state in errors.
  SharedState(const std::string& path, bool create, uint32_t magic, uint32_t version,
              size_t payload_bytes, const char* what);
  ~SharedState();
  SharedState(const SharedState&) = delete;
  SharedState& operator=(const SharedState&) = delete;

  void* payload() { return payload_; }
  uint64_t* words() { return static_cast<uint64_t*>(payload_); }
  size_t word_count() const { return words_; }

 private:
  int fd_ = -1;
  void* map_ = nullptr;
  size_t map_bytes_ = 0;
  void* payload_ = nullptr;
  size_t words_ = 0;
};

}  // namespace vgpu
