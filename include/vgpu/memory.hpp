// Virtual device memory.
//
// Design:
//  - Device pointers live in a fake VA range starting at kDeviceVaBase so they
//    can never be confused with host pointers.
//  - Backing is sparse: 64 KiB chunks materialize on first write. A virtual
//    H200 can advertise 141 GB of VRAM on a 16 GB laptop; only touched pages
//    cost host RAM -- or disk, past VGPU_MEMORY_RAM_MB (memory_backing.hpp).
//  - A chunk that holds one byte value throughout costs nothing at all: it is
//    stored as that value. A memory stress test fills the whole card with one
//    pattern and then works a small part of it, so a 12 GB fill becomes a table
//    of tags, and only the chunks written with other data take RAM or disk. Untouched device memory reads as zero (documented
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
#include <mutex>

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "vgpu/memory_backing.hpp"

namespace vgpu {

// Device windows are 0x2000'0000'0000 to 0x3000'0000'0000. They used to start
// at 0x7fff'0000'0000, which is where Linux puts the stack: a stack buffer at
// 0x7fff'4d6e'2190 passed to cudaMemcpyDefault was taken for device memory, and
// a copy failed one run in five. Every range this can collide with is
// elsewhere: the stack and shared libraries near 0x7fff'…/0x7f…, a PIE program
// and its heap near 0x55…–0x56…, a non-PIE one near 0x40'0000, and the engine's
// own private windows at 0x6ffb'…–0x6fff'…. Every check is bounded above too.
inline constexpr uint64_t kDeviceVaBase = 0x2000'0000'0000ull;
// Device functions get addresses in their own window so a function pointer is
// a real value that can be stored, loaded and compared -- and so an indirect
// call can find the function again. The index is the address: nothing is ever
// loaded *from* this range, and giving it a window of its own means a stray
// dereference of a function pointer is diagnosable rather than a wild read.
inline constexpr uint64_t kFuncVaBase = 0x6ffb'0000'0000ull;
inline constexpr uint64_t kFuncVaStride = 8;
inline constexpr uint64_t kFuncVaSize = 1ull << 20;
// Each device owns a disjoint 1 TiB window above that base. CUDA guarantees
// unified virtual addressing -- a device pointer is unique process-wide and
// identifies the device that owns it -- and without separate windows two
// devices hand out the same numeric address for different memory, so a
// cross-device copy silently reads the wrong buffer instead of failing.
inline constexpr uint64_t kDeviceVaStride = 0x100'0000'0000ull;  // 1 TiB
// One window per device, as many as a rack may hold (telemetry::kMaxDevices).
inline constexpr uint64_t kDeviceVaWindows = 16;
inline constexpr uint64_t kDeviceVaEnd = kDeviceVaBase + kDeviceVaWindows * kDeviceVaStride;
// Whether an address is in some device's window. "At or above the base" alone
// claimed every higher address -- the stack included -- for the devices.
inline constexpr bool is_device_va(uint64_t addr) { return addr >= kDeviceVaBase && addr < kDeviceVaEnd; }
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
  bool owns(uint64_t addr) const {
    if (addr >= va_base_ && addr < va_base_ + kDeviceVaStride) return true;
    // Managed buffers live at their real host address, outside every device
    // window, and are still device-addressable.
    return host_maps_ && is_host_mapped(addr);
  }

  // Allocates `size` bytes of virtual device memory. size == 0 is invalid.
  uint64_t alloc(uint64_t size);

  // Frees an allocation. `ptr` must be the exact base returned by alloc().
  void free(uint64_t ptr);

  // Frees every live allocation, as a device reset does. Each goes through
  // free(), so the pointers stay in the quarantine and a program that uses one
  // afterwards is told it was freed.
  void free_all();

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

  // Faults taken on a kernel's accesses (`vgpu fault arm`): its device-memory
  // loads and stores, and its shared-memory loads. While an access's pending
  // word reads zero -- nearly always -- the access costs that one read. Each
  // returns the value to use in place of `value`, or throws.
  struct AccessFault {
    virtual ~AccessFault() = default;
    virtual uint64_t on_load(uint64_t addr, uint32_t size, uint64_t value) = 0;
    virtual uint64_t on_store(uint64_t addr, uint32_t size, uint64_t value) = 0;
    virtual uint64_t on_shared_load(uint64_t offset, uint32_t size, uint64_t value) = 0;
    // Stuck cells: forces the bits stuck in `len` bytes read from `offset`,
    // counted from the start of this device's memory. Every read of device
    // memory goes through it, copies included.
    virtual void on_read(uint64_t offset, uint8_t* bytes, uint64_t len) = 0;
    // An arithmetic result `bits` wide, as a kernel computed it.
    virtual uint64_t on_alu(uint64_t value, uint32_t bits) = 0;
    // `len` bytes a copy read from device memory at `addr`, about to be
    // delivered; may change them, or throw.
    virtual void on_copy(uint64_t addr, uint8_t* bytes, uint64_t len) = 0;
    const uint64_t* copy_pending = nullptr;
    const uint64_t* stuck_pending = nullptr;
    const uint64_t* alu_pending = nullptr;
    const uint64_t* load_pending = nullptr;
    const uint64_t* store_pending = nullptr;
    const uint64_t* shared_pending = nullptr;
  };
  void set_access_fault(AccessFault* f) { access_fault_ = f; }

  // A kernel's shared-memory load, at `offset` in its block's shared memory.
  // Shared memory is the SM's, not this manager's, but its faults are armed
  // with the device's.
  // Faults on arithmetic results reach kernels through here too, since this
  // is the device's one link to its armed faults. True while any is armed.
  bool alu_fault_armed() const {
    return access_fault_ && access_fault_->alu_pending &&
           __atomic_load_n(access_fault_->alu_pending, __ATOMIC_RELAXED);
  }
  uint64_t alu_result(uint64_t value, uint32_t bits) const { return access_fault_->on_alu(value, bits); }

  uint64_t shared_loaded(uint64_t offset, uint32_t size, uint64_t value) const {
    if (access_fault_ && access_fault_->shared_pending &&
        __atomic_load_n(access_fault_->shared_pending, __ATOMIC_RELAXED))
      return access_fault_->on_shared_load(offset, size, value);
    return value;
  }

  // Locates the live allocation containing `addr`. Returns false if none.
  bool find_allocation(uint64_t addr, uint64_t* base, uint64_t* size) const;

  // ---- managed memory ----
  //
  // One buffer both the host and a kernel can address, which is what
  // cudaMallocManaged promises. Everywhere else in this engine a device
  // pointer is a virtual address with no host meaning, so host code cannot
  // dereference it -- that is exactly why cudaMallocManaged used to refuse.
  //
  // The way out is available only to a simulator: allocate real host memory
  // and tell the device side to address it at its own real address. The host
  // dereferences it because it is an ordinary pointer, and a kernel reaches it
  // because these two calls put it on the map that read/write consult.
  //
  // Cost when unused is one relaxed atomic load, so a program with no managed
  // memory pays nothing.
  void map_host(uint64_t addr, void* host, uint64_t len);
  void unmap_host(uint64_t addr);
  // True when `addr` falls in a mapped host buffer rather than device VA.
  bool is_host_mapped(uint64_t addr) const;

  // Host memory actually backing this device's allocations: chunks that have
  // been materialized, times the chunk size. This is the quantity the sparse
  // backing exists to keep small, and unlike the process's resident size it is
  // exact and unaffected by the allocator, the page cache, or a sanitizer's
  // shadow memory -- which makes it something a test can assert on.
  uint64_t resident_bytes() const;

  uint64_t used() const { return used_; }
  uint64_t capacity() const { return capacity_; }
  size_t live_allocations() const { return live_.size(); }

 private:
  AccessFault* access_fault_ = nullptr;
  bool any_stuck() const {
    return access_fault_ && access_fault_->stuck_pending &&
           __atomic_load_n(access_fault_->stuck_pending, __ATOMIC_RELAXED);
  }
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
      for (size_t i = 0; i < chunk_count; ++i) {
        uint8_t* c = chunks[i].load(std::memory_order_relaxed);
        if (!is_uniform(c)) backing::release(c);
      }
    }

   public:
  };
  struct FreedRecord {
    uint64_t size = 0;
    uint64_t seq = 0;  // eviction order
  };

  // A uniform chunk is a tagged value in the chunk table: the byte, shifted up,
  // with the low bit set. Real chunks come from new[] or mmap and are always
  // aligned, so their low bit is clear and the two can never be confused.
  static bool is_uniform(const uint8_t* c) { return (reinterpret_cast<uintptr_t>(c) & 1) != 0; }
  static uint8_t uniform_byte(const uint8_t* c) {
    return static_cast<uint8_t>(reinterpret_cast<uintptr_t>(c) >> 1);
  }
  static uint8_t* uniform_chunk(uint8_t b) {
    return reinterpret_cast<uint8_t*>((static_cast<uintptr_t>(b) << 1) | 1);
  }

  // Makes chunk `chunk_idx` real -- from nothing, or from a uniform chunk's byte --
  // and returns the chunk that won the race; safe to call from several block
  // threads at once.
  static uint8_t* materialize(Allocation& a, uint64_t chunk_idx);

  // Maps addr to (allocation base, allocation); throws with diagnostics.
  const Allocation& resolve(uint64_t addr, uint64_t len, const char* op, uint64_t* base_out) const;
  // Copies `len` bytes at offset `off` in allocation `a` out to `d`.
  void read_chunks(const Allocation& a, uint64_t off, uint8_t* d, uint64_t len) const;
  Allocation& resolve_mut(uint64_t addr, uint64_t len, const char* op, uint64_t* base_out);

  uint64_t capacity_;
  uint64_t va_base_;
  uint64_t used_ = 0;
  uint64_t next_va_;
  void notify_usage() const {
    if (usage_observer_) usage_observer_(used_);
  }

  std::function<void(uint64_t)> usage_observer_;

  // Managed buffers: real host memory the device can also address. Rare and
  // written only at allocation, so a flag keeps the read path free for the
  // programs that never use it.
  struct HostMap {
    uint64_t base = 0;
    uint64_t len = 0;
    uint8_t* host = nullptr;
  };
  // Behind a pointer, and not for indirection's sake: a mutex and an atomic as
  // direct members make MemoryManager non-movable, and it is moved (a test
  // returns one by value). The pointer is also the fast-path check -- null
  // means no managed memory, which is almost every program.
  struct HostMaps {
    std::vector<HostMap> maps;
    // Recently unmapped buffers, so a kernel that touches one after it was
    // freed is told it is a use-after-free rather than "not a device pointer".
    // Bounded like the device-memory quarantine; a new mapping over the same
    // range retires the record, because the allocator reuses host addresses.
    std::vector<HostMap> retired;
    std::mutex mu;
    // The lowest and highest address ever mapped. Pinned buffers make host
    // maps common rather than rare, and without these every scalar access a
    // kernel makes to ordinary device memory would take the lock above to
    // learn it is not in a host map. Device windows sit far above the host
    // heap, so a relaxed range check sends those straight to the chunk table.
    // Only ever widened, so a stale read is conservative, never wrong.
    std::atomic<uint64_t> lo{UINT64_MAX};
    std::atomic<uint64_t> hi{0};
    bool may_contain(uint64_t addr) const {
      return addr >= lo.load(std::memory_order_relaxed) &&
             addr < hi.load(std::memory_order_relaxed);
    }
  };
  static constexpr size_t kRetiredHostMaps = 256;
  std::unique_ptr<HostMaps> host_maps_;
  const HostMap* find_host_map_locked(uint64_t addr, uint64_t len) const;
  // Where a kernel's scalar access lands: bytes in a chunk, memory nothing has
  // written (a load reads zero and must not copy), a uniform chunk (a load reads
  // its byte; a store of that same byte changes nothing), or a managed host
  // mapping. `store` is the value being stored, or null for a load.
  enum class ScalarAt { Chunk, Untouched, Uniform, Unchanged, HostMap };
  ScalarAt scalar_location(uint64_t addr, uint32_t size, const uint64_t* store,
                           const uint8_t** where) const;
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
