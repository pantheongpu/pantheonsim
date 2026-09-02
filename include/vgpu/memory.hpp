// Virtual device memory.
//
// Design:
//  - Device pointers live in a fake VA range starting at kDeviceVaBase so they
//    can never be confused with host pointers.
//  - Backing is sparse: 64 KiB chunks materialize on first write. A virtual
//    H200 can advertise 141 GB of VRAM on a 16 GB laptop; only touched pages
//    cost host RAM. Untouched device memory reads as zero (documented
//    VirtualGPU behavior; real GPUs leave it undefined).
//  - Virtual addresses are handed out monotonically and never reused, so a
//    freed pointer can never alias a later allocation.
//  - Freed allocations are remembered in a bounded quarantine (the most recent
//    kQuarantineEntries frees) so use-after-free and double-free stay
//    detectable without growing without limit: workloads that cycle millions
//    of allocations would otherwise leak one record per free. Beyond the
//    quarantine a stale pointer is reported as an invalid pointer rather than
//    a use-after-free -- still an error, with a hint that it may be stale.
//  - Every access is bounds-checked and produces a rich diagnostic on failure
//    (these diagnostics are a product feature for CI, not just debug aids).
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace vgpu {

inline constexpr uint64_t kDeviceVaBase = 0x7fff'0000'0000ull;
// Each device owns a disjoint 1 TiB window above that base. CUDA guarantees
// unified virtual addressing -- a device pointer is unique process-wide and
// identifies the device that owns it -- and without separate windows two
// devices hand out the same numeric address for different memory, so a
// cross-device copy silently reads the wrong buffer instead of failing.
inline constexpr uint64_t kDeviceVaStride = 0x100'0000'0000ull;  // 1 TiB
inline constexpr uint64_t kAllocAlign = 256;  // matches CUDA's documented minimum alignment
inline constexpr uint64_t kChunkSize = 64 * 1024;
// How many freed allocations stay individually diagnosable. Bounded so that
// long-running alloc/free loops do not grow memory forever.
inline constexpr size_t kQuarantineEntries = 4096;

class MemoryManager {
 public:
  explicit MemoryManager(uint64_t capacity_bytes, uint32_t device_ordinal = 0)
      : capacity_(capacity_bytes),
        va_base_(kDeviceVaBase + static_cast<uint64_t>(device_ordinal) * kDeviceVaStride),
        next_va_(va_base_),
        high_water_va_(va_base_) {}

  // The device's window in the process-wide address space, and whether an
  // address falls inside it.
  uint64_t va_base() const { return va_base_; }
  bool owns(uint64_t addr) const { return addr >= va_base_ && addr < va_base_ + kDeviceVaStride; }

  // Allocates `size` bytes of virtual device memory. size == 0 is invalid.
  uint64_t alloc(uint64_t size);

  // Frees an allocation. `ptr` must be the exact base returned by alloc().
  void free(uint64_t ptr);

  // Bulk copies (the H2D/D2H/D2D building blocks).
  void write(uint64_t dst, const void* src, uint64_t len);
  void read(uint64_t src, void* dst, uint64_t len) const;

  // Fills a range by repeating a 1/2/4-byte pattern, without building a host
  // copy of the range first. Staging a buffer the size of the fill is what
  // made a memset of N bytes cost 2N of host RAM: the staging buffer is live
  // at the same time as the chunks it is being copied into. A zero pattern
  // skips untouched chunks entirely, since those already read as zero.
  void fill(uint64_t dst, const uint8_t* pattern, uint32_t pattern_len, uint64_t len);

  // Scalar access for the interpreter. size must be 1/2/4/8 and the address
  // naturally aligned. Values are zero-extended into the returned u64.
  uint64_t load_scalar(uint64_t addr, uint32_t size) const;
  void store_scalar(uint64_t addr, uint32_t size, uint64_t value);

  // Notified with the new total whenever allocated bytes change. Used to feed
  // live memory telemetry; optional and unset by default.
  void set_usage_observer(std::function<void(uint64_t)> obs) { usage_observer_ = std::move(obs); }

  // Locates the live allocation containing `addr`. Returns false if none.
  bool find_allocation(uint64_t addr, uint64_t* base, uint64_t* size) const;

  uint64_t used() const { return used_; }
  uint64_t capacity() const { return capacity_; }
  size_t live_allocations() const { return live_.size(); }

 private:
  // Chunks are a flat array of owning pointers rather than a map: a lookup is
  // then an index instead of a red-black tree walk, which is the difference
  // between one instruction and a cache-missing traversal on every scalar
  // access. Making them atomic also lets blocks run on several threads without
  // a lock on the fast path -- a chunk is created once and never moves, so a
  // reader either sees null (reads as zero) or the final pointer.
  struct Allocation {
    uint64_t size = 0;
    size_t chunk_count = 0;
    std::unique_ptr<std::atomic<uint8_t*>[]> chunks;

    Allocation() = default;
    // The move has to clear the source's count as well as its pointer: a
    // defaulted move leaves chunk_count behind, and the moved-from destructor
    // then walks a null array.
    Allocation(Allocation&& o) noexcept
        : size(o.size), chunk_count(o.chunk_count), chunks(std::move(o.chunks)) {
      o.size = 0;
      o.chunk_count = 0;
    }
    Allocation& operator=(Allocation&& o) noexcept {
      if (this != &o) {
        release();
        size = o.size;
        chunk_count = o.chunk_count;
        chunks = std::move(o.chunks);
        o.size = 0;
        o.chunk_count = 0;
      }
      return *this;
    }
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
    ~Allocation() { release(); }

   private:
    void release() {
      if (!chunks) return;
      for (size_t i = 0; i < chunk_count; ++i) delete[] chunks[i].load(std::memory_order_relaxed);
    }

   public:
  };
  struct FreedRecord {
    uint64_t size = 0;
    uint64_t seq = 0;  // eviction order
  };

  // Creates chunk `chunk_idx` if absent and returns the chunk that won the
  // race; safe to call from several block threads at once.
  static uint8_t* materialize(Allocation& a, uint64_t chunk_idx);

  // Maps addr to (allocation base, allocation); throws with diagnostics.
  const Allocation& resolve(uint64_t addr, uint64_t len, const char* op, uint64_t* base_out) const;
  Allocation& resolve_mut(uint64_t addr, uint64_t len, const char* op, uint64_t* base_out);

  uint64_t capacity_;
  uint64_t va_base_;
  uint64_t used_ = 0;
  uint64_t next_va_;
  void notify_usage() const {
    if (usage_observer_) usage_observer_(used_);
  }

  std::function<void(uint64_t)> usage_observer_;
  std::map<uint64_t, Allocation> live_;        // base -> allocation
  // base -> record, bounded to kQuarantineEntries (oldest evicted first).
  std::map<uint64_t, FreedRecord> freed_;
  std::map<uint64_t, uint64_t> freed_order_;  // seq -> base, for eviction
  uint64_t freed_seq_ = 0;
  // Highest VA ever handed out, so a pointer inside the retired range can be
  // called out as stale even after it leaves the quarantine.
  uint64_t high_water_va_;
};

}  // namespace vgpu
