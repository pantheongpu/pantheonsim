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
// Both unset, nothing changes there. VGPU_MEMORY_DIR without VGPU_MEMORY_RAM_MB
// never spills on the RAM limit; the limit is what decides when to.
//
// Independently of that limit there is a host-memory guard, on by default. A
// program that touches more memory than the host has would otherwise grow RAM
// until the kernel's OOM killer fired, and the killer picks its victims by size
// and badness, not by fault: on a shared machine it takes other services down.
// So before a new heap chunk is handed out the engine asks how much memory the
// host can still give this process -- the smaller of its cgroup's headroom
// (memory.max less memory.current, every ancestor considered, file cache that
// the kernel can drop not counted as used) and the system's MemAvailable. Below
// a reserve, the chunk is treated as past the RAM limit: it spills to
// VGPU_MEMORY_DIR if that is set and can grow, and otherwise the allocation
// fails with Error(Err::OutOfMemory), which the CUDA and HIP shims report as
// an ordinary device out-of-memory (cudaErrorMemoryAllocation,
// hipErrorOutOfMemory) as a full card would. The check reads two or three small
// files, so it runs once per 256 chunks (16 MiB), not for every chunk. Chunks
// that cost no memory, such as a card filled with one byte, never reach it.
//
//   VGPU_MEMORY_HOST_RESERVE_MB  memory to leave to the host, in MB. Default:
//                                6% of the memory limit in play (the cgroup's
//                                limit or the machine's RAM, whichever is
//                                smaller), but at least 1 GiB, at most 4 GiB,
//                                and never more than a quarter of that limit.
//   VGPU_MEMORY_HOST_GUARD=0     switch the guard off: the old behaviour, where
//                                only the kernel decides when memory is gone.
#pragma once
#include <cstdint>
#include <limits>
#include <string>

namespace vgpu::backing {

// host_reserve_bytes value meaning "work it out from the memory limit".
inline constexpr uint64_t kAutoReserve = std::numeric_limits<uint64_t>::max();

struct Config {
  // Heap chunks allowed before new chunks come from the spill file.
  uint64_t ram_limit_bytes = std::numeric_limits<uint64_t>::max();
  // Directory for the spill file. Empty: never spill.
  std::string spill_dir;

  // The host-memory guard (see the top of this file).
  bool host_guard = true;                         // VGPU_MEMORY_HOST_GUARD=0 clears it
  uint64_t host_reserve_bytes = kAutoReserve;     // VGPU_MEMORY_HOST_RESERVE_MB
  // Heap chunks handed out between two looks at the host's memory. The first
  // chunk always looks.
  uint64_t host_check_every_chunks = 256;
  // Where the guard reads from; tests point these at a directory of synthetic
  // files. cgroup_self lists the cgroup this process is in (/proc/self/cgroup
  // format) and cgroup_root is where those paths are rooted.
  std::string meminfo_path = "/proc/meminfo";
  std::string cgroup_root = "/sys/fs/cgroup";
  std::string cgroup_self = "/proc/self/cgroup";
};

// The configuration the environment gives: VGPU_MEMORY_RAM_MB, VGPU_MEMORY_DIR,
// VGPU_MEMORY_HOST_RESERVE_MB and VGPU_MEMORY_HOST_GUARD. It is what the engine
// uses until configure() replaces it.
Config config_from_env();

// Replaces the configuration the environment would give. For tests and for
// programs embedding the engine; call it before the first chunk is needed.
void configure(const Config& config);

// A zeroed chunk of kChunkSize bytes. Throws Error(Err::OutOfMemory) when the
// heap limit is reached, or the host is short of memory, and the spill file
// cannot grow: no directory, or no room left on its disk. (Host memory low with
// no spill directory is that error too.) Disk space is reserved before a chunk is handed out,
// so a full disk is this error and never a SIGBUS in the middle of a write.
uint8_t* acquire();

// Gives a chunk back. A file chunk has its disk blocks released at once.
//
// `dirty_bytes` is a promise from the caller: every byte of the chunk from that offset on is still
// zero. A heap chunk with only a few bytes written (a kernel's argument block, a flag) then goes to
// a short list of chunks to hand out again, cleared below the offset, and not back to the heap to be
// zeroed whole by the next acquire(). The default promises nothing.
inline constexpr uint64_t kAllBytesDirty = ~uint64_t{0};
// The most that is cleared (under the lock) to keep a chunk: a buffer of a few KiB, allocated, written and freed
// over and over, then costs the clearing of what it wrote and not a heap allocation and the zeroing of all 64 KiB
// on every allocation. Past it the chunk goes back to the heap.
inline constexpr uint64_t kPoolMaxDirty = 32 * 1024;
void release(uint8_t* chunk, uint64_t dirty_bytes = kAllBytesDirty);
// Chunks kept for hand-out (cleared, not counted in ram_bytes()).
uint64_t pooled_chunks();

// Bytes of chunks currently on the heap and in the spill file.
uint64_t ram_bytes();
uint64_t spill_bytes();

// How many times the host-memory guard has read the host's memory figures.
uint64_t host_guard_samples();

}  // namespace vgpu::backing
