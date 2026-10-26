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

#include <type_traits>
#include <sys/mman.h>
#include <unistd.h>
#include <mutex>
#include "vgpu/brlock.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <functional>
#include <deque>
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
// Kernels get addresses in a window of their own for the same reason: a
// device-side launch (dynamic parallelism) names its child grid by the
// kernel's address, which the runtime's kernel table turns back into the
// kernel. Nothing is loaded from here either.
inline constexpr uint64_t kKernelVaBase = 0x6ffc'0000'0000ull;
inline constexpr uint64_t kKernelVaStride = 8;
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
  // `align` (a power of two) starts the allocation on a multiple of it if that
  // is more than the allocator's own alignment.
  uint64_t alloc(uint64_t size, uint64_t align = 0);

  // AMD's HIP hands device memory out a 4 KB page at a time, so a kernel that
  // reads a little past the end of a buffer reads the rest of its page, and
  // libraries count on it: hipSPARSELt's sparse GEMMs read a byte past their
  // compressed matrix. With a page size set, allocations start on a page and
  // a kernel's read of the rest of the last one is served; a copy or a write
  // past the end is still refused. 0 -- the default, and NVIDIA's -- keeps
  // every access to the bytes allocated.
  void set_page_size(uint64_t bytes) { page_ = bytes; }

  // The alignment cudaMalloc-style allocations start on, kAllocAlign unless set. NVIDIA's runtime promises a
  // pointer from cudaMalloc binds to a texture with no offset (cudaDeviceProp::textureAlignment, 512 bytes), and
  // programs rely on it: convolutionFFT2D makes a texture over each buffer it allocates. `bytes` is a power of two.
  void set_alloc_alignment(uint64_t bytes) { alloc_align_ = bytes; }

  // Frees an allocation. `ptr` must be the exact base returned by alloc().
  void free(uint64_t ptr);

  // Frees every live allocation, as a device reset does. Each goes through
  // free(), so the pointers stay in the quarantine and a program that uses one
  // afterwards is told it was freed. The device heap starts over empty.
  void free_all();

  // ---- the device heap ----
  //
  // malloc() and free() in a kernel, for both engines (the PTX interpreter's
  // call and the SASS executor's): one heap per device, kept here with the
  // memory it hands out, so a block one kernel allocates another may free,
  // whichever code each ran, and a reset takes the heap's blocks and its
  // budget with everything else. Blocks are ordinary allocations, so a kernel
  // that overruns one or uses it after free() is told so.
  //
  // `limit` is cudaLimitMallocHeapSize as the launch saw it. Returns the
  // block, or 0 -- not an error, as on the device -- for 0 bytes, for a
  // request past what is left of the limit, and for one the device's memory
  // cannot hold.
  uint64_t heap_alloc(uint64_t size, uint64_t limit);
  // Gives a block back to the heap and frees it. False, with nothing changed,
  // when `ptr` is not a live heap block (the caller reports it with its own
  // context); a null `ptr` is not passed here.
  bool heap_free(uint64_t ptr);
  // Bytes the heap has handed out and not had back.
  uint64_t heap_used() const;
  // Whether `addr` lies in a live heap block. The host API refuses those: an
  // RTX 3060's cudaFree and cuMemFree of a device-malloc'd block return
  // InvalidValue and leave it live (the device's free() still works), and
  // cuMemGetAddressRange answers NOT_FOUND -- they are the device heap's,
  // not allocations the host made.
  bool heap_contains(uint64_t addr) const;
  // A NUL-terminated copy of `text` in device memory, made once and kept until
  // the device resets: what the device runtime's cudaGetErrorString and
  // cudaGetErrorName point a kernel at.
  uint64_t intern_string(const std::string& text);

  // Bulk copies (the H2D/D2H/D2D building blocks).
  void write(uint64_t dst, const void* src, uint64_t len);
  void read(uint64_t src, void* dst, uint64_t len) const;
  // A kernel's run of `unit`-byte loads or stores (TMA's box rows), done at
  // once: exactly what load_scalar / store_scalar would do element by
  // element, each element still one relaxed atomic access, for a run wholly
  // inside one ordinary device allocation. Returns false, having done
  // nothing, whenever the two could differ -- fault injection attached, a host
  // mapping possibly in range, a run not inside one allocation, or one not
  // aligned to `unit` -- and the caller then goes element by element, errors
  // and all.
  bool load_run(uint64_t addr, void* dst, uint64_t len, uint32_t unit) const;
  bool store_run(uint64_t addr, const void* src, uint64_t len, uint32_t unit);

  // The accesses of one warp-wide load or store, made together, for the ordinary
  // case: `count` accesses of `size` bytes, in order. They return true having
  // done every one exactly as load_scalar / store_scalar would, in that order,
  // or false having done nothing -- whenever any access could differ (a fault
  // armed, a host mapping in range, a misaligned access or one outside the
  // allocation the first is in, an address that is no allocation's), in which
  // case the caller makes them one at a time and meets any error where it
  // would have. One lookup serves the lot.
  bool load_batch(const uint64_t* addrs, size_t count, uint32_t size, uint64_t* out) const;
  // A store that cannot get host memory for a chunk throws what store_scalar
  // would, with `*failed_at` the index of the access that did.
  bool store_batch(const uint64_t* addrs, size_t count, uint32_t size, const uint64_t* vals,
                   size_t* failed_at = nullptr);

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
  // An atomic's memory side: the load_scalar of `size` bytes at `addr`, `compute(old)`, and the store_scalar of its
  // result, with the address resolved once. Returns true having done all three (`*old_out` the value loaded, which
  // `compute` also got); a `compute` that throws leaves memory as it was and the exception passes through. Returns
  // false having done nothing -- and not called `compute` -- whenever the pair of calls could differ from this:
  // a fault or stuck cell armed, a host mapping possibly in range, a size that is not 1/2/4/8, a misaligned address,
  // or an access not wholly inside one ordinary allocation (freed, out of range, mapped, no lookup cache). The
  // caller then makes the two calls as before and meets any error where it would have. A chunk that cannot get host
  // memory throws what store_scalar would (the store is the only thing here that can).
  using RmwFn = uint64_t (*)(void* ctx, uint64_t old);
  bool rmw_scalar_fn(uint64_t addr, uint32_t size, RmwFn compute, void* ctx, uint64_t* old_out);
  template <class F>
  bool rmw_scalar(uint64_t addr, uint32_t size, F&& compute, uint64_t* old_out) {
    return rmw_scalar_fn(
        addr, size, [](void* c, uint64_t old) -> uint64_t { return (*static_cast<std::remove_reference_t<F>*>(c))(old); },
        const_cast<std::remove_const_t<std::remove_reference_t<F>>*>(&compute), old_out);
  }

  // The load, `compute(old)` and a store of the result where it differs (the executors' atomics on one thread), for a word in a
  // chunk that is real in the last allocation this thread found: one lookup for the two. Returns false, having done nothing, for
  // any other word (an untouched or uniform chunk, another allocation, a fault armed, a host mapping in reach, a misaligned
  // address) and the caller then makes the two calls.
  bool rmw_if_changed_fn(uint64_t addr, uint32_t size, RmwFn compute, void* ctx, uint64_t* old_out);
  template <class F>
  bool rmw_if_changed(uint64_t addr, uint32_t size, F&& compute, uint64_t* old_out) {
    return rmw_if_changed_fn(
        addr, size, [](void* c, uint64_t old) -> uint64_t { return (*static_cast<std::remove_reference_t<F>*>(c))(old); },
        const_cast<std::remove_const_t<std::remove_reference_t<F>>*>(&compute), old_out);
  }
  // Whether `addr` may be in a host mapping (false is a sure no, and the usual answer: nothing is mapped near a device window).
  bool host_map_possible(uint64_t addr) const { return host_maps_ && host_maps_->may_contain(addr); }

  // A lock-free atomic read-modify-write on a 4- or 8-byte naturally aligned word: `compute(old)` is applied with a
  // compare-and-swap loop on the word itself (the value it is called with is the one the swap replaces, so it is
  // called again when another thread got in between, and must be pure). Returns true with `*old_out` the value the
  // successful swap replaced; the result is stored only where load_scalar / store_scalar under a lock would have
  // stored it: when it differs from the old value, or (`always_store`) always -- a store that changes nothing still
  // makes a never-written chunk real. A chunk that is not real yet is made so first (a uniform chunk is left alone
  // by a store of its own byte). Returns false, having done nothing and not called `compute`, whenever the plain
  // pair of calls could differ: a fault armed, a host mapping possibly in range, another size, a misaligned
  // address, or an access not wholly inside one ordinary allocation; the caller then takes its lock and goes as
  // before. Every thread that reaches the same address takes the same road, so the two never meet on one word.
  bool atomic_rmw_fn(uint64_t addr, uint32_t size, RmwFn compute, void* ctx, uint64_t* old_out, bool always_store);
  template <class F>
  bool atomic_rmw(uint64_t addr, uint32_t size, F&& compute, uint64_t* old_out, bool always_store = false) {
    return atomic_rmw_fn(
        addr, size, [](void* c, uint64_t old) -> uint64_t { return (*static_cast<std::remove_reference_t<F>*>(c))(old); },
        const_cast<std::remove_const_t<std::remove_reference_t<F>>*>(&compute), old_out, always_store);
  }

  // A wave's read-modify-writes at once: `count` accesses of `size` bytes in order, access i loading the word, `compute(i, old)` and
  // storing the result where it differs (`lock_free`: as atomic_rmw does, a compare-and-swap on the word, for 4 and 8 bytes; else
  // the plain load and store of rmw_if_changed), with one lookup for the lot. `olds[i]` is what access i loaded; `*done` is how many
  // were made (all of them, or those before a store that could not get host memory and threw). Returns false, having done nothing,
  // wherever any access could differ from the calls these stand for: a fault armed, a host mapping in reach, an address misaligned or
  // outside the one allocation the first is in, a size that is not 1/2/4/8 (4/8 when lock_free).
  using RmwLaneFn = uint64_t (*)(void* ctx, size_t i, uint64_t old);
  bool rmw_batch_fn(const uint64_t* addrs, size_t count, uint32_t size, RmwLaneFn compute, void* ctx, uint64_t* olds, size_t* done,
                    bool lock_free);
  template <class F>
  bool rmw_batch(const uint64_t* addrs, size_t count, uint32_t size, F&& compute, uint64_t* olds, size_t* done, bool lock_free) {
    return rmw_batch_fn(
        addrs, count, size,
        [](void* c, size_t i, uint64_t old) -> uint64_t { return (*static_cast<std::remove_reference_t<F>*>(c))(i, old); },
        const_cast<std::remove_const_t<std::remove_reference_t<F>>*>(&compute), olds, done, lock_free);
  }

  // The fields of one record (a texel's channels), `n` of them end to end from `addr`, field i `sizes[i]` bytes
  // (0, 1, 2, 4 or 8; a 0 reads nothing and gives 0), each one load_scalar of its width. Returns true with
  // `out[i]` set, or false having done nothing -- a fault or stuck cell armed, a host mapping possibly in range,
  // a field not naturally aligned or not wholly inside the one ordinary allocation the record lies in -- and the
  // caller then reads the fields one at a time and meets any error where it would have. One lookup serves the lot.
  bool load_fields(uint64_t addr, const uint8_t* sizes, unsigned n, uint64_t* out) const;

  // Sixteen naturally aligned bytes as one access, as a GPU moves a wide load
  // or store (global_load_dwordx4): a kernel on another host thread sees them
  // all old or all new, never half of each. rocPRIM's decoupled look-back
  // packs a tile's flag and 64-bit prefix into sixteen bytes and trusts the
  // prefix once it sees the flag. One instruction where the host has one
  // that is atomic (x86-64 with AVX, whose aligned 16-byte SSE accesses Intel
  // and AMD guarantee are), a lock striped by address elsewhere. With a fault
  // armed (not merely attached, as the runtime attaches its faults to every
  // device) or a host mapping possibly in range, two 8-byte accesses, as
  // load_scalar and store_scalar make them.
  void load_quad(uint64_t addr, uint64_t out[2]) const;
  void store_quad(uint64_t addr, const uint64_t in[2]);

  // `count` consecutive naturally aligned sixteen-byte values from `addr`, each made exactly as load_quad / store_quad
  // make it (one access each, in order), for a run wholly inside one ordinary allocation, with one lookup for the lot.
  // Returns false, having done nothing, whenever the calls could differ -- a fault armed or cells stuck, a host mapping
  // possibly in range, a misaligned address, or a run not wholly inside one allocation (load_quad lets a read run on
  // into the last page of an allocation; the caller then goes one at a time and gets that).
  bool load_quad_run(uint64_t addr, size_t count, uint64_t* out) const;
  bool store_quad_run(uint64_t addr, size_t count, const uint64_t* in);

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
  // The word alu_fault_armed reads (null: no fault hook), for a caller that asks per instruction and keeps it
  // rather than following the hook and its pointer each time. It does not change once the hook is set.
  const uint64_t* alu_pending_flag() const { return access_fault_ ? access_fault_->alu_pending : nullptr; }

  // Whether a shared-memory load could come back changed; while none is,
  // shared_loaded returns its value as it is.
  bool shared_fault_armed() const {
    return access_fault_ && access_fault_->shared_pending &&
           __atomic_load_n(access_fault_->shared_pending, __ATOMIC_RELAXED);
  }
  uint64_t shared_loaded(uint64_t offset, uint32_t size, uint64_t value) const {
    if (access_fault_ && access_fault_->shared_pending &&
        __atomic_load_n(access_fault_->shared_pending, __ATOMIC_RELAXED))
      return access_fault_->on_shared_load(offset, size, value);
    return value;
  }

  // Whether a kernel's scalar accesses use resolve_scalar's per-thread memory of
  // recent lookups (on unless VGPU_LOOKUP_CACHE=0 when the manager is made).
  // Nothing observable depends on it -- values, errors and their text are
  // the same either way, which tests/unit/test_memfast.cpp checks -- so it is
  // there to be switched off and compared against, never to be tuned.
  void set_lookup_cache(bool on) { lookup_cache_ = on; }
  bool lookup_cache() const { return lookup_cache_; }

  // Locates the live allocation containing `addr`. Returns false if none.
  bool find_allocation(uint64_t addr, uint64_t* base, uint64_t* size) const;

  // ---- virtual memory management ----
  //
  // What the cuMem* mapping API needs, and the shape CUDA documents: address
  // space is *reserved* without memory behind it, memory is *created* as a
  // handle that has no address, a handle is *mapped* into a reservation, and a
  // mapping is unusable until access is granted for a device. That is how
  // PyTorch's expandable segments and NCCL's window allocations grow a buffer
  // without moving it, and it needs physical allocations separable from
  // addresses -- which is why the sparse chunk model alone could not answer it.
  //
  // Every state a real device faults on faults here, with a diagnostic saying
  // which one it was: reserved with nothing mapped, mapped with no access
  // granted, written through a read-only mapping, or touched after unmap.
  //
  // Sizes and addresses are multiples of kVmmGranularity, which is what
  // cuMemGetAllocationGranularity reports: 2 MiB, the minimum and the recommended one alike (an RTX 3060 under
  // NVIDIA's driver answers 2 MiB for every property it was asked about, and cuMemCreate of a smaller or
  // unaligned size is CUDA_ERROR_INVALID_VALUE). The sparse chunks are smaller than that and fit in it evenly.
  static constexpr uint64_t kVmmGranularity = 2 * 1024 * 1024;
  static_assert(kVmmGranularity % kChunkSize == 0, "a granule is a whole number of chunks");

  // Address space with nothing behind it. `alignment` 0 means the granularity, and so does a smaller one (the
  // card aligns every reservation to at least 2 MiB); it must be a power of two.
  uint64_t reserve(uint64_t size, uint64_t alignment);
  // Whether any part of [va, va + size) has a handle mapped into it.
  bool range_mapped(uint64_t va, uint64_t size) const;
  // Gives a reservation back. It must have nothing mapped in it.
  void address_free(uint64_t va, uint64_t size);

  // Physical memory with no address, counted against the device's capacity.
  // Returns a handle; handles are never 0 and are never reused.
  // `shareable_types` is the handle types the memory was asked to be exportable as (CUmemAllocationHandleType
  // bits); only the POSIX file descriptor (1) can be exported.
  // `compressible` is the request for generic memory compression (cuMemCreate's compressionType). The bytes
  // read and write the same either way; the handle only remembers that it was granted, which a program reads
  // back from cuMemGetAllocationPropertiesFromHandle.
  uint64_t create_handle(uint64_t size, int shareable_types = 0, bool compressible = false);
  uint64_t handle_size(uint64_t handle) const;
  int handle_types(uint64_t handle) const;
  bool handle_compressible(uint64_t handle) const;
  // A file descriptor another process can map to reach a handle's memory (cuMemExportToShareableHandle):
  // the memory moves into an anonymous file the first time, and every descriptor made from it
  // (a dup) names that file. Throws InvalidValue for a handle not made shareable.
  int export_handle(uint64_t handle);
  // A new handle over the memory behind a descriptor export_handle made, in this process or another.
  uint64_t import_handle(int fd);
  // The handle mapped at `va` (retaining a reference), or 0 when nothing is.
  uint64_t retain_handle_at(uint64_t va);
  void retain_handle(uint64_t handle);
  // One reference. The memory goes when the last reference and the last
  // mapping are gone, in either order -- CUDA lets a handle be released while
  // it is still mapped, and the mapping keeps working.
  void release_handle(uint64_t handle);

  // Puts `handle`'s memory at `va`. `offset` into the handle must be 0, as
  // cuMemMap documents.
  void map(uint64_t va, uint64_t size, uint64_t offset, uint64_t handle);
  // Takes it away again. The range must be exactly one mapping.
  void unmap(uint64_t va, uint64_t size);
  // Grants (or withdraws) a device's access to mapped address space.
  void set_access(uint64_t va, uint64_t size, bool readable, bool writable);
  // The access flags at `va`; false when nothing is mapped there.
  bool access_at(uint64_t va, bool* readable, bool* writable) const;
  // Reservations, mappings and handles that exist now, for tests and `vgpu`.
  size_t reservations() const { return reserved_.size(); }
  size_t mappings() const { return maps_.size(); }
  size_t handles() const { return handles_.size(); }

  // ---- memory another process can map (IPC) ----
  //
  // Device memory here is this process's own sparse chunks, which is why a
  // handle to it used to be refused: a pointer means nothing in another
  // process. What can be shared is a file, and the backing store already maps
  // its spill slices MAP_SHARED -- so an allocation that is exported is moved
  // into a file of its own, mapped shared, and put back at the same device
  // address. Its bytes then live somewhere another process can map them, and
  // every read, write and kernel access reaches them through the same path
  // managed memory uses.
  //
  // `share` moves the allocation at `ptr` into `path` and keeps the address
  // working. Returns the number of bytes now shared. Called once per
  // allocation; a second call returns the same size and changes nothing.
  uint64_t share(uint64_t ptr, const std::string& path);
  // Whether `ptr` is the base of an allocation this process has exported.
  bool is_shared(uint64_t ptr) const;
  // Maps `path` (`size` bytes, already shared by another process) at a device
  // address of this device's own, and returns it. The bytes are not copied:
  // both processes see the same memory.
  uint64_t adopt(const std::string& path, uint64_t size);
  // Gives up a mapping `adopt` returned. The file stays; the process that
  // exported it owns it.
  void abandon(uint64_t va);
  // Removes the file behind a region this process shared, keeping the
  // mapping: once every importer has opened it, nothing needs the name, and a
  // process that dies before freeing the region then leaves no file behind.
  // False if `ptr` is not the base of a region this process shared.
  bool unlink_shared(uint64_t ptr);

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
  //
  // `read_only` is memory registered with cudaHostRegisterReadOnly: the device may read it, and a
  // store or atomic to it faults (an RTX 3060: cudaErrorLaunchFailure at the next synchronize).
  void map_host(uint64_t addr, void* host, uint64_t len, bool read_only = false);
  void unmap_host(uint64_t addr);
  // True when `addr` falls in a mapped host buffer rather than device VA.
  bool is_host_mapped(uint64_t addr) const;
  // Where [addr, addr + len) is in host memory, when one mapped host buffer
  // holds all of it; null otherwise. An atomic on host memory the host also
  // updates has to be made with the CPU's own atomics, there.
  uint8_t* host_address(uint64_t addr, uint64_t len) const;

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
    size_t chunk_capacity = 0;   // entries chunks has room for: a recycled allocation keeps its array
    std::unique_ptr<std::atomic<uint8_t*>[]> chunks;
    uint8_t* external = nullptr;
    size_t external_len = 0;

    Allocation() = default;
    Allocation(Allocation&& o) noexcept
        : size(o.size), chunk_count(o.chunk_count), chunk_capacity(o.chunk_capacity), chunks(std::move(o.chunks)),
          external(o.external), external_len(o.external_len) {
      o.size = 0;
      o.chunk_count = 0;
      o.chunk_capacity = 0;
      o.external = nullptr;
    }
    Allocation& operator=(Allocation&& o) noexcept {
      if (this != &o) {
        release();
        size = o.size;
        chunk_count = o.chunk_count;
        chunk_capacity = o.chunk_capacity;
        chunks = std::move(o.chunks);
        external = o.external;
        external_len = o.external_len;
        o.size = 0;
        o.chunk_count = 0;
        o.chunk_capacity = 0;
        o.external = nullptr;
      }
      return *this;
    }
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
    ~Allocation() { release(); }

    // Gives the chunks back and leaves an empty allocation that still owns its (small) array, ready
    // for set_up().
    void recycle() {
      release();
      if (chunks && chunk_capacity <= kRecycledChunkEntries && !external) {
        for (size_t i = 0; i < chunk_count; ++i) chunks[i].store(nullptr, std::memory_order_relaxed);
      } else {
        chunks.reset();
        chunk_capacity = 0;
      }
      size = 0;
      chunk_count = 0;
      external = nullptr;
      external_len = 0;
    }
    // A fresh allocation of `bytes`: every chunk untouched.
    void set_up(uint64_t bytes) {
      size = bytes;
      chunk_count = static_cast<size_t>((bytes + kChunkSize - 1) / kChunkSize);
      if (!chunks || chunk_capacity < chunk_count) {
        chunks = std::make_unique<std::atomic<uint8_t*>[]>(chunk_count);
        chunk_capacity = chunk_count;
      }
    }

   private:
    static constexpr size_t kRecycledChunkEntries = 16;
    void release() {
      if (!chunks) return;
      if (external) {
        ::munmap(external, external_len);
        external = nullptr;
        return;
      }
      for (size_t i = 0; i < chunk_count; ++i) {
        uint8_t* c = chunks[i].load(std::memory_order_relaxed);
        if (is_uniform(c)) continue;
        // What may be non-zero in this chunk: all of it, or for the last only the part below the
        // allocation's end, because every write is bounds-checked against the size, and the one
        // write that goes past it, a chunk made from a uniform chunk of one byte, is of a whole
        // chunk, which only a whole chunk of the allocation can be (fill makes uniform chunks of
        // whole chunks only).
        const uint64_t dirty = i + 1 < chunk_count ? kChunkSize : size - i * kChunkSize;
        backing::release(c, dirty);
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
  const Allocation* run_allocation(uint64_t addr, uint64_t len, uint32_t unit, uint64_t* base) const;
  // rmw_scalar's parts. The slot holds where the access is; a chunk that is real stays the same pointer.
  struct RmwSlot {
    Allocation* a;
    uint64_t chunk_idx;
    uint64_t coff;      // offset in the chunk
    uint8_t* chunk;     // as read by rmw_locate: null, a uniform tag, or the bytes
  };
  bool rmw_locate(uint64_t addr, uint32_t size, RmwSlot* s) const;
  // atomic_rmw's swap loop on a located slot.
  void atomic_rmw_slot(RmwSlot& s, uint32_t size, RmwFn compute, void* ctx, uint64_t* old_out, bool always_store);
  uint64_t rmw_load(const RmwSlot& s, uint32_t size) const;
  void rmw_store(RmwSlot& s, uint32_t size, uint64_t value);
  // Whether a fault is armed that load_scalar or store_scalar would apply.
  bool access_faults_armed() const {
    const auto armed = [](const uint64_t* pending) { return pending && __atomic_load_n(pending, __ATOMIC_RELAXED); };
    return access_fault_ && (armed(access_fault_->stuck_pending) || armed(access_fault_->load_pending) ||
                             armed(access_fault_->store_pending));
  }

  // Maps addr to (allocation base, allocation); throws with diagnostics.
  // `writing` picks the diagnostic for a read-only mapping.
  // `page_slack` lets a read run on into the rest of the allocation's last
  // page (set_page_size).
  const Allocation& resolve(uint64_t addr, uint64_t len, const char* op, uint64_t* base_out,
                            bool writing = false, bool page_slack = false) const;
  // resolve() for a kernel's scalar access, where the lookup is nearly all the
  // access costs. A thread remembers the last few ordinary allocations it
  // resolved, with the table lock's epoch (vgpu/brlock.hpp) it found them
  // under; while no allocation, free or mapping has changed the table, an
  // access wholly inside one of them needs no lock and no map walk. Everything
  // else -- a miss, a range that runs off the end, a mapped or freed address --
  // takes resolve(), so every diagnostic is the one resolve() makes.
  // The ordinary allocation holding [addr, addr + len), from the memory of
  // recent lookups or the table; null if there is none or the memory is off.
  const Allocation* find_ordinary(uint64_t addr, uint64_t len, uint64_t* base_out) const;
  uint8_t* hot_location(uint64_t addr, uint32_t size) const;
  const Allocation* find_ordinary_miss(uint64_t addr, uint64_t len, uint64_t* base_out) const;
  bool batch_ok(const uint64_t* addrs, size_t count, uint32_t size, bool store, const Allocation** alloc,
                uint64_t* base) const;
  const Allocation& resolve_scalar(uint64_t addr, uint64_t len, const char* op, uint64_t* base_out,
                                   bool writing, bool page_slack) const;
  // The mapped-memory half of resolve: a hit returns the handle's allocation,
  // a miss returns null so the caller can carry on with its own diagnostics.
  const Allocation* resolve_mapped(uint64_t addr, uint64_t len, const char* op, uint64_t* base_out,
                                   bool writing) const;
  // Copies `len` bytes at offset `off` in allocation `a` out to `d`.
  void read_chunks(const Allocation& a, uint64_t off, uint8_t* d, uint64_t len) const;
  Allocation& resolve_mut(uint64_t addr, uint64_t len, const char* op, uint64_t* base_out);

  static bool lookup_cache_default();
  bool lookup_cache_ = lookup_cache_default();
  uint64_t capacity_;
  uint64_t va_base_;
  uint64_t used_ = 0;
  uint64_t next_va_;
  uint64_t page_ = 0;   // set_page_size
  uint64_t alloc_align_ = kAllocAlign;   // set_alloc_alignment
  void notify_usage() const {
    if (usage_observer_) usage_observer_(used_);
  }

  std::function<void(uint64_t)> usage_observer_;

  // ---- virtual memory management state ----
  // A reservation is address space and nothing else. A handle is memory and no
  // address: its chunks are an Allocation like any other, which is what lets a
  // mapped range use the same read, write and fault paths. A mapping joins the
  // two, and carries the access a device was granted.
  struct Reservation {
    uint64_t size = 0;
  };
  struct Handle {
    uint64_t size = 0;
    uint32_t refs = 1;      // cuMemCreate's own reference, plus every retain
    uint32_t mapped = 0;    // mappings pointing at it
    int types = 0;          // the handle types it can be exported as
    bool compressible = false;   // created with generic compression
    int fd = -1;            // the anonymous file holding it once exported or imported (owned)
    Allocation mem;
    Handle() = default;
    Handle(Handle&& o) noexcept
        : size(o.size), refs(o.refs), mapped(o.mapped), types(o.types), compressible(o.compressible), fd(o.fd),
          mem(std::move(o.mem)) {
      o.fd = -1;
    }
    ~Handle() {
      if (fd >= 0) ::close(fd);
    }
  };
  struct Mapping {
    uint64_t size = 0;
    uint64_t offset = 0;    // into the handle
    uint64_t handle = 0;
    bool readable = false;  // until cuMemSetAccess, a mapping is unusable
    bool writable = false;
  };
  std::map<uint64_t, Reservation> reserved_;   // by address
  std::map<uint64_t, Mapping> maps_;           // by address
  std::map<uint64_t, Handle> handles_;         // by handle id
  uint64_t next_handle_ = 1;
  // Drops a handle when its last reference and last mapping are gone.
  void collect_handle(uint64_t handle);

  // An allocation moved into a file for another process to map, or a mapping of
  // one this process did not export. Both are host mappings as far as reads and
  // writes are concerned; this is what free and close need to undo them.
  struct SharedRegion {
    uint64_t va = 0;
    uint64_t size = 0;
    void* host = nullptr;    // the MAP_SHARED mapping
    std::string path;
    bool owner = false;      // this process exported it, so it unlinks the file
  };
  std::map<uint64_t, SharedRegion> shared_;
  // Unmaps a shared region and forgets it; unlinks the file if we own it.
  void drop_shared(std::map<uint64_t, SharedRegion>::iterator it);

  // Managed buffers: real host memory the device can also address. Rare and
  // written only at allocation, so a flag keeps the read path free for the
  // programs that never use it.
  struct HostMap {
    uint64_t base = 0;
    uint64_t len = 0;
    uint8_t* host = nullptr;
    bool read_only = false;
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
    // Which version of `maps` this is. A kernel's scalar access to mapped host memory used to take `mu` and scan
    // `maps` every time (a 1M-element vector add over three mapped buffers ran 1.7x slower than over engine memory
    // on one thread and 3.9x slower on three; PantheonZE's docs/exec-address-model.md has the table); lookup_host() now
    // keeps each thread's last few hits and trusts them while this has not moved. Every change to `maps` takes a
    // value no HostMaps has had before (process-wide counter), so an entry cached for a destroyed manager whose
    // HostMaps lived at the same address can never be taken for a current one.
    std::atomic<uint64_t> gen;
    HostMaps();
    // The range alone is not enough: a PIE program's heap (0x55...) and the
    // mmap area (0x7f...) sit either side of the device windows, so a pinned
    // buffer in each -- PyTorch and CUTLASS's tests both make them -- spans
    // every device address, and each scalar access a kernel made took the
    // lock (a third of a CUTLASS blockwise GEMM's time, most of it waiting).
    // A host buffer inside a device window would need ~95 TiB of mappings
    // below the mmap base first; if one is ever mapped there, this is set and
    // device addresses go through the range check again.
    std::atomic<bool> in_device_window{false};
    // Whether any address from `first` to `last` (inclusive, first <= last) may_contain would not rule out: a range wholly in
    // the device windows (one interval) is ruled out as one address is.
    bool may_overlap(uint64_t first, uint64_t last) const {
      if (is_device_va(first) && is_device_va(last) && !in_device_window.load(std::memory_order_relaxed)) return false;
      return first < hi.load(std::memory_order_relaxed) && last >= lo.load(std::memory_order_relaxed);
    }
    bool may_contain(uint64_t addr) const {
      if (is_device_va(addr) && !in_device_window.load(std::memory_order_relaxed)) return false;
      return addr >= lo.load(std::memory_order_relaxed) &&
             addr < hi.load(std::memory_order_relaxed);
    }
  };
  static constexpr size_t kRetiredHostMaps = 256;
  // Made with the manager rather than on the first mapping: a kernel's worker
  // thread reads this pointer while the program maps a buffer (hipHostMalloc
  // while a launch runs), so it must never change after construction.
  std::unique_ptr<HostMaps> host_maps_ = std::make_unique<HostMaps>();
  const HostMap* find_host_map_locked(uint64_t addr, uint64_t len) const;
  // The mapping that holds [addr, addr + len), without taking the lock when this thread has just used it (a
  // small per-thread cache validated by HostMaps::gen). False where nothing is mapped over the whole range.
  struct HostHit {
    uint8_t* host;
    uint64_t base, len;
    bool read_only;
  };
  bool lookup_host(uint64_t addr, uint64_t len, HostHit* out) const;
  static uint64_t fresh_generation();
 public:
  // The cache is on unless VGPU_HOSTMAP_CACHE=0. This switches it for the whole process (a benchmark compares the
  // two; nothing else needs to).
  static void set_host_lookup_cache(bool on);
 private:
  // Host memory mapped over the whole of [addr, addr + len), perhaps as
  // several mappings placed end to end: the pieces, in order, as (host
  // address, offset into the range, length). False where any byte is not.
  struct HostPiece {
    uint8_t* host;
    uint64_t offset, len;
  };
  bool host_pieces(uint64_t addr, uint64_t len, std::vector<HostPiece>* pieces, bool for_write = false) const;
  // Where a kernel's scalar access lands: bytes in a chunk, memory nothing has
  // written (a load reads zero and must not copy), a uniform chunk (a load reads
  // its byte; a store of that same byte changes nothing), or a managed host
  // mapping. `store` is the value being stored, or null for a load.
  enum class ScalarAt { Chunk, Untouched, Uniform, Unchanged, HostMap };
  ScalarAt scalar_location(uint64_t addr, uint32_t size, const uint64_t* store,
                           const uint8_t** where) const;
  // Guards the tables below (and the mapping tables above) against a host
  // thread allocating, freeing or mapping while kernels on other threads
  // look addresses up in them (vgpu/brlock.hpp). Behind a pointer so the
  // manager stays movable.
  std::unique_ptr<BigReaderLock> table_lock_ = std::make_unique<BigReaderLock>();
  std::map<uint64_t, Allocation> live_;        // base -> allocation
  // base -> record, bounded to kQuarantineEntries (oldest evicted first).
  std::map<uint64_t, FreedRecord> freed_;
  // The frees, oldest first, as their entries in freed_ (a node of a map keeps its place): the
  // quarantine's eviction order. The entry at the front has sequence number freed_front_seq_, the
  // next one that of the entry after it, and so on.
  std::deque<std::map<uint64_t, FreedRecord>::iterator> freed_order_;
  uint64_t freed_front_seq_ = 0;
  // A node of live_ and one of freed_ kept from the last free and eviction, for the next allocation
  // and free to put back without asking the heap (used under the table's writer lock only).
  std::map<uint64_t, Allocation>::node_type spare_live_;
  std::map<uint64_t, FreedRecord>::node_type spare_freed_;
  void remember_freed(uint64_t ptr, uint64_t size);
  // Highest VA ever handed out, so a pointer inside the retired range can be
  // called out as stale even after it leaves the quarantine.
  uint64_t high_water_va_;

  // The device heap (heap_alloc): the bytes handed out and each block's size.
  // Its own lock, taken before the table lock and never inside it; behind a
  // pointer so the manager stays movable.
  struct DeviceHeap {
    std::mutex mu;
    uint64_t used = 0;
    std::map<uint64_t, uint64_t> blocks;   // base -> bytes asked for
    std::map<std::string, uint64_t> strings;   // intern_string's
  };
  std::unique_ptr<DeviceHeap> heap_ = std::make_unique<DeviceHeap>();
};

}  // namespace vgpu
