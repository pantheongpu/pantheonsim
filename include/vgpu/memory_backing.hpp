// Where device memory's bytes live.
//
// Device memory is sparse (see memory.hpp): 64 KiB chunks appear on first
// write. By default each chunk is ordinary heap memory, which is the fastest
// thing there is and fine while the memory a program touches fits in RAM.
//
// A memory stress test does not fit. It fills 99% of the card, and a 12 GB card
// filled in a 2 GB container is killed by the kernel's OOM killer: heap memory
// in a container without swap has nowhere else to go.
//
// So chunks can come from a file instead. Past VGPU_MEMORY_RAM_MB of heap
// chunks, new chunks are pages of a file in VGPU_MEMORY_DIR, mapped shared.
// The kernel may write those pages out and drop them from RAM whenever memory
// is tight, which it may never do with heap memory -- so the program keeps
// running, at the speed of the disk rather than of RAM.
//
// Both unset, nothing changes. VGPU_MEMORY_DIR without VGPU_MEMORY_RAM_MB never
// spills; the limit is what decides when to.
#pragma once
#include <cstdint>
#include <limits>
#include <string>

namespace vgpu::backing {

struct Config {
  // Heap chunks allowed before new chunks come from the spill file.
  uint64_t ram_limit_bytes = std::numeric_limits<uint64_t>::max();
  // Directory for the spill file. Empty: never spill.
  std::string spill_dir;
};

// Replaces the configuration the environment would give. For tests and for
// programs embedding the engine; call it before the first chunk is needed.
void configure(const Config& config);

// A zeroed chunk of kChunkSize bytes. Throws Error(Err::OutOfMemory) when the
// heap limit is reached and the spill file cannot grow: no directory, or no
// room left on its disk. Disk space is reserved before a chunk is handed out,
// so a full disk is this error and never a SIGBUS in the middle of a write.
uint8_t* acquire();

// Gives a chunk back. A file chunk has its disk blocks released at once.
void release(uint8_t* chunk);

// Bytes of chunks currently on the heap and in the spill file.
uint64_t ram_bytes();
uint64_t spill_bytes();

}  // namespace vgpu::backing
