#include "vgpu/memory.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <mutex>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "vgpu/error.hpp"

namespace vgpu {
namespace {

// A kernel's scalar access, as an atomic on the backing bytes. The chunk is an
// unsigned-char array, which C++20 lets hold objects of other trivial types.
template <typename T>
uint64_t relaxed_load(const uint8_t* p) {
  return std::atomic_ref<T>(*reinterpret_cast<T*>(const_cast<uint8_t*>(p)))
      .load(std::memory_order_relaxed);
}
template <typename T>
void relaxed_store(uint8_t* p, uint64_t v) {
  std::atomic_ref<T>(*reinterpret_cast<T*>(p)).store(static_cast<T>(v), std::memory_order_relaxed);
}

// A scalar in place, as an atomic of its own width -- or byte by byte when the
// chunk's storage is not aligned for that width, so no path copies the bytes a
// concurrent store is writing.
uint64_t load_at(const uint8_t* p, uint32_t size) {
  // `size` is 1, 2, 4 or 8 in every caller: a mask, not a division.
  if ((reinterpret_cast<uintptr_t>(p) & (size - 1)) == 0) {
    switch (size) {
      case 1: return relaxed_load<uint8_t>(p);
      case 2: return relaxed_load<uint16_t>(p);
      case 4: return relaxed_load<uint32_t>(p);
      default: return relaxed_load<uint64_t>(p);
    }
  }
  uint64_t v = 0;
  for (uint32_t i = 0; i < size; ++i) v |= relaxed_load<uint8_t>(p + i) << (8 * i);
  return v;
}
void store_at(uint8_t* p, uint32_t size, uint64_t v) {
  if ((reinterpret_cast<uintptr_t>(p) & (size - 1)) == 0) {
    switch (size) {
      case 1: relaxed_store<uint8_t>(p, v); return;
      case 2: relaxed_store<uint16_t>(p, v); return;
      case 4: relaxed_store<uint32_t>(p, v); return;
      default: relaxed_store<uint64_t>(p, v); return;
    }
  }
  for (uint32_t i = 0; i < size; ++i) relaxed_store<uint8_t>(p + i, (v >> (8 * i)) & 0xff);
}

// Sixteen aligned bytes in place, as one access (MemoryManager::load_quad).
// x86-64 with AVX moves them in one SSE instruction, which Intel and AMD
// guarantee is atomic when aligned; without it, or elsewhere, each side takes
// one of a set of locks striped by address.
bool quad_is_one_instruction() {
#if defined(__x86_64__)
  static const bool avx = __builtin_cpu_supports("avx");
  return avx;
#else
  return false;
#endif
}
std::mutex& quad_lock(const uint8_t* p) {
  static std::array<std::mutex, 64> locks;
  return locks[(reinterpret_cast<uintptr_t>(p) >> 4) % locks.size()];
}
void load16(const uint8_t* p, uint64_t out[2]) {
#if defined(__x86_64__)
  if (quad_is_one_instruction() && reinterpret_cast<uintptr_t>(p) % 16 == 0) {
    __m128i v;
    asm volatile("movdqa %1, %0" : "=x"(v) : "m"(*reinterpret_cast<const __m128i*>(p)));
    std::memcpy(out, &v, 16);
    return;
  }
#endif
  std::lock_guard<std::mutex> lock(quad_lock(p));
  out[0] = load_at(p, 8), out[1] = load_at(p + 8, 8);
}
void store16(uint8_t* p, const uint64_t in[2]) {
#if defined(__x86_64__)
  if (quad_is_one_instruction() && reinterpret_cast<uintptr_t>(p) % 16 == 0) {
    __m128i v;
    std::memcpy(&v, in, 16);
    asm volatile("movdqa %1, %0" : "=m"(*reinterpret_cast<__m128i*>(p)) : "x"(v));
    return;
  }
#endif
  std::lock_guard<std::mutex> lock(quad_lock(p));
  store_at(p, 8, in[0]), store_at(p + 8, 8, in[1]);
}

// A `size`-byte scalar whose every byte is `b`, and whether `v` is one.
uint64_t repeated(uint8_t b, uint32_t size) {
  const uint64_t all = 0x0101010101010101ull * b;
  return size == 8 ? all : all & ((uint64_t{1} << (8 * size)) - 1);
}
bool all_bytes(const uint8_t* p, uint64_t n, uint8_t b) {
  for (uint64_t i = 0; i < n; ++i)
    if (p[i] != b) return false;
  return true;
}

// Formats an address as 0x-prefixed hex for diagnostics.
struct Hex {
  uint64_t v;
};
std::ostream& operator<<(std::ostream& os, Hex h) {
  auto f = os.flags();
  os << "0x" << std::hex << h.v;
  os.flags(f);
  return os;
}

}  // namespace

uint64_t MemoryManager::alloc(uint64_t size, uint64_t request) {
  // A new entry in the table makes no answer a thread remembered wrong, so readers' memories stay.
  ExclusiveGuard table_guard(table_lock_.get(), /*quiet=*/true);
  if (size == 0) throw Error::make(Err::InvalidValue, "cudaMalloc-style allocation of 0 bytes is invalid");
  if (used_ + size > capacity_ || used_ + size < used_)
    throw Error::make(Err::OutOfMemory, "device out of memory: requested ", size, " bytes, ", used_,
                      " of ", capacity_, " bytes already in use");
  const uint64_t align = std::max({alloc_align_, page_, request});
  uint64_t base = (next_va_ + align - 1) / align * align;
  uint64_t padded = (size + align - 1) / align * align;
  next_va_ = base + padded;
  high_water_va_ = next_va_;
  used_ += size;
  // The node of the last freed allocation, with its array of chunk slots, when there is one.
  auto node = std::move(spare_live_);
  if (node.empty()) {
    Allocation a;
    a.set_up(size);
    live_.emplace_hint(live_.end(), base, std::move(a));
  } else {
    node.key() = base;
    node.mapped().set_up(size);
    live_.insert(live_.end(), std::move(node));   // the newest address is the highest
  }
  notify_usage();
  return base;
}

void MemoryManager::remember_freed(uint64_t ptr, uint64_t size) {
  const FreedRecord record{size, freed_front_seq_ + freed_order_.size()};
  // Frees mostly come in the order of allocation, which is the order of address: the end is the place.
  auto at = freed_.empty() || ptr > freed_.rbegin()->first ? freed_.end() : freed_.lower_bound(ptr);
  std::map<uint64_t, FreedRecord>::iterator entry;
  if (at != freed_.end() && at->first == ptr) {
    at->second = record;
    entry = at;
  } else if (!spare_freed_.empty()) {
    spare_freed_.key() = ptr;
    spare_freed_.mapped() = record;
    entry = freed_.insert(at, std::move(spare_freed_));
  } else {
    entry = freed_.emplace_hint(at, ptr, record);
  }
  freed_order_.push_back(entry);
  // Evict the oldest entries once the bound is reached; the node of the last goes to the next free.
  while (freed_order_.size() > kQuarantineEntries) {
    const auto stale = freed_order_.front();
    // Only drop it if this record is the one that entry refers to.
    if (stale->second.seq == freed_front_seq_) spare_freed_ = freed_.extract(stale);
    freed_order_.pop_front();
    ++freed_front_seq_;
  }
}

void MemoryManager::free(uint64_t ptr) {
  // Quiet: a free says which allocation it takes (note_free) and leaves every other answer a thread
  // remembered as it was.
  ExclusiveGuard table_guard(table_lock_.get(), /*quiet=*/true);
  // An allocation that was exported for another process lives in a file now,
  // not in chunks. Freeing it takes the mapping and the file with it, which is
  // what the exporting process owns.
  if (const auto sh = shared_.find(ptr); sh != shared_.end() && sh->second.owner) {
    used_ -= sh->second.size;
    remember_freed(ptr, sh->second.size);
    table_lock_->note_free(ptr);
    drop_shared(sh);
    notify_usage();
    return;
  }
  auto it = live_.find(ptr);
  if (it != live_.end()) {
    used_ -= it->second.size;
    // Quarantine the freed range so use-after-free stays diagnosable, evicting
    // the oldest entry once the bound is reached.
    remember_freed(ptr, it->second.size);
    table_lock_->note_free(ptr);
    auto node = live_.extract(it);
    node.mapped().recycle();
    spare_live_ = std::move(node);
    notify_usage();
    return;
  }
  if (freed_.count(ptr))
    throw Error::make(Err::DoubleFree, "double free of device pointer ", Hex{ptr}, " (allocation of ",
                      freed_.at(ptr).size, " bytes was already freed)");
  // Interior pointer into a live allocation?
  auto up = live_.upper_bound(ptr);
  if (up != live_.begin()) {
    auto prev = std::prev(up);
    if (ptr < prev->first + prev->second.size)
      throw Error::make(Err::InvalidFree, "free of interior device pointer ", Hex{ptr},
                        ": allocation starts at ", Hex{prev->first}, " (", prev->second.size,
                        " bytes); pass the base pointer");
  }
  if (ptr >= va_base_ && ptr < high_water_va_)
    throw Error::make(Err::InvalidPointer, "free of device pointer ", Hex{ptr},
                      " that is not a live allocation; it is inside the retired address range, so "
                      "it was most likely freed earlier (beyond the ", kQuarantineEntries,
                      "-entry double-free quarantine)");
  throw Error::make(Err::InvalidPointer, "free of unknown device pointer ", Hex{ptr},
                    " (never returned by an allocation)");
}

uint64_t MemoryManager::heap_alloc(uint64_t size, uint64_t limit) {
  std::lock_guard<std::mutex> guard(heap_->mu);
  // Out of heap is a null pointer, not a failure, as on the device: each
  // thread's malloc stands alone, and the program is expected to check it.
  // Compared without the sum, which a huge request would wrap.
  if (size == 0 || heap_->used > limit || size > limit - heap_->used) return 0;
  uint64_t p = 0;
  try {
    p = alloc(size);
  } catch (const Error&) {
    return 0;   // the device's memory is full before the heap is
  }
  heap_->used += size;
  heap_->blocks.emplace(p, size);
  return p;
}

bool MemoryManager::heap_free(uint64_t ptr) {
  std::lock_guard<std::mutex> guard(heap_->mu);
  const auto it = heap_->blocks.find(ptr);
  if (it == heap_->blocks.end()) return false;
  heap_->used -= it->second;
  heap_->blocks.erase(it);
  free(ptr);
  return true;
}

uint64_t MemoryManager::heap_used() const {
  std::lock_guard<std::mutex> guard(heap_->mu);
  return heap_->used;
}

uint64_t MemoryManager::intern_string(const std::string& text) {
  std::lock_guard<std::mutex> guard(heap_->mu);
  const auto it = heap_->strings.find(text);
  if (it != heap_->strings.end()) return it->second;
  const uint64_t va = alloc(text.size() + 1);
  write(va, text.c_str(), text.size() + 1);
  heap_->strings[text] = va;
  return va;
}

bool MemoryManager::heap_contains(uint64_t addr) const {
  std::lock_guard<std::mutex> guard(heap_->mu);
  auto it = heap_->blocks.upper_bound(addr);
  if (it == heap_->blocks.begin()) return false;
  --it;
  return addr - it->first < it->second;
}

const MemoryManager::Allocation* MemoryManager::resolve_mapped(uint64_t addr, uint64_t len,
                                                               const char* op, uint64_t* base_out,
                                                               bool writing) const {
  if (maps_.empty() && reserved_.empty()) return nullptr;
  auto mu = maps_.upper_bound(addr);
  if (mu != maps_.begin()) {
    auto prev = std::prev(mu);
    const uint64_t va = prev->first;
    const Mapping& m = prev->second;
    if (addr - va < m.size) {
      const uint64_t remaining = va + m.size - addr;
      if (len > remaining)
        throw Error::make(Err::OutOfBounds, op, " of ", len, " bytes at ", Hex{addr},
                          " runs past the end of the ", m.size, "-byte mapping at ", Hex{va},
                          " (mapped address space ends at ", Hex{va + m.size - 1}, ")");
      // Mapped is not yet usable: CUDA grants access separately, and a device
      // faults on memory it was never given access to.
      if (!m.readable && !m.writable)
        throw Error::make(Err::InvalidPointer, op, " at ", Hex{addr},
                          " is mapped but no device has been given access to it "
                          "(cuMemSetAccess)");
      if (writing && !m.writable)
        throw Error::make(Err::InvalidPointer, op, " at ", Hex{addr},
                          " is through a read-only mapping (cuMemSetAccess granted read access "
                          "only)");
      const auto h = handles_.find(m.handle);
      if (h == handles_.end())
        throw Error::make(Err::Internal, op, " at ", Hex{addr}, ": mapping has no handle");
      // The handle's own offset 0 sits this far below the mapping's address, so
      // the caller's `addr - base` lands at the right offset in its chunks.
      if (base_out) *base_out = va - m.offset;
      return &h->second.mem;
    }
  }
  auto ru = reserved_.upper_bound(addr);
  if (ru != reserved_.begin()) {
    auto prev = std::prev(ru);
    if (addr - prev->first < prev->second.size)
      throw Error::make(Err::InvalidPointer, op, " at ", Hex{addr},
                        " is inside reserved address space with nothing mapped there (cuMemMap)");
  }
  return nullptr;
}

const MemoryManager::Allocation& MemoryManager::resolve(uint64_t addr, uint64_t len, const char* op,
                                                        uint64_t* base_out, bool writing, bool page_slack) const {
  SharedGuard table_guard(table_lock_.get());
  auto up = live_.upper_bound(addr);
  if (up != live_.begin()) {
    auto prev = std::prev(up);
    uint64_t base = prev->first;
    const Allocation& a = prev->second;
    // A kernel's read may run on to the end of the last page (set_page_size),
    // which never passes the allocation's last chunk: a chunk is a whole
    // number of pages.
    const uint64_t extent = page_slack && page_ ? (a.size + page_ - 1) / page_ * page_ : a.size;
    if (addr < base + extent) {
      // Computed as a remaining-length rather than an end-address: `addr + len`
      // wraps for a length near UINT64_MAX, and a wrapped sum compares *below*
      // the end, so the check passed exactly the accesses it exists to stop.
      const uint64_t remaining = base + extent - addr;
      if (len > remaining)
        throw Error::make(Err::OutOfBounds, op, " of ", len, " bytes at ", Hex{addr},
                          " runs past the end of the ", a.size, "-byte allocation at ", Hex{base},
                          " (last valid byte: ", Hex{base + a.size - 1}, ")");
      if (base_out) *base_out = base;
      return a;
    }
    // The VA range up to the alignment-padded end belongs to this allocation:
    // an access there is an overrun, which deserves a better diagnosis than
    // "unknown pointer".
    const uint64_t align = std::max(alloc_align_, page_);
    uint64_t padded = (a.size + align - 1) / align * align;
    if (addr < base + padded)
      throw Error::make(Err::OutOfBounds, op, " at ", Hex{addr}, " is ", addr - (base + a.size),
                        " bytes past the end of the ", a.size, "-byte allocation at ", Hex{base});
  }
  if (const Allocation* mapped = resolve_mapped(addr, len, op, base_out, writing)) return *mapped;
  // Freed allocation?
  auto fup = freed_.upper_bound(addr);
  if (fup != freed_.begin()) {
    auto prev = std::prev(fup);
    if (addr < prev->first + prev->second.size)
      throw Error::make(Err::UseAfterFree, op, " at ", Hex{addr}, " touches freed allocation ",
                        Hex{prev->first}, " (", prev->second.size, " bytes); device memory was freed");
  }
  if (addr >= va_base_ && addr < high_water_va_)
    throw Error::make(Err::UseAfterFree, op, " at ", Hex{addr},
                      " is inside the retired address range: the allocation it belonged to was "
                      "freed (older than the ", kQuarantineEntries,
                      "-entry quarantine, so its size is no longer recorded)");
  // A managed or pinned buffer that has since been freed lives at a host
  // address, outside every device window; without this a kernel touching it
  // was told it had a stray host pointer rather than a freed one.
  if (host_maps_ && !is_device_va(addr)) {
    std::lock_guard<std::mutex> lock(host_maps_->mu);
    for (const HostMap& r : host_maps_->retired)
      if (addr >= r.base && addr - r.base < r.len)
        throw Error::make(Err::UseAfterFree, op, " at ", Hex{addr}, " touches host buffer ",
                          Hex{r.base}, " (", r.len, " bytes), which has been freed or unregistered");
  }
  throw Error::make(Err::InvalidPointer, op, " at ", Hex{addr},
                    ": address is not inside any device allocation",
                    !is_device_va(addr)  ? " (looks like a host pointer, not a device pointer)"
                    : !owns(addr)      ? " (belongs to a different device)"
                                       : "");
}

namespace {
// What one thread remembers of its lookups in resolve_scalar. Zero-initialised,
// so it needs no construction; an epoch of zero is one no lock has.
struct LookupMemory {
  static constexpr unsigned kWays = 8;
  struct Entry {
    uint64_t epoch = 0;
    uint64_t base = 0;
    uint64_t size = 0;
    const void* allocation = nullptr;
  };
  Entry way[kWays];
  unsigned last = 0;     // the way that answered last: the next lookup tries it first
  uint32_t victim = 1;   // which way a new entry replaces: pseudo-random (xorshift), so a
                         // kernel cycling through more allocations than there are ways
                         // still hits some of the time instead of never
};
// A thread's memory is made on its first lookup and given back when it ends. It is found by the
// thread pointer in a table of slots, with no thread-local on the way: this code is linked into every
// shim library, a thread-local that is not general-dynamic would take static TLS from each (glibc has
// about 1.6 KB for all libraries loaded late) and a general-dynamic one is a call on every access.
// A slot belongs to the thread whose pointer it holds: only that thread writes its memory or takes
// the slot away, so the others' compare-and-swaps into other slots are all it can see. A thread that
// finds no room (all probes taken), or that is ending, reads an empty memory that is never written
// (`g_ended_lookups`), and so finds nothing and takes the table: slower, not wrong.
LookupMemory g_ended_lookups;
constexpr size_t kLookupSlots = 1024, kLookupProbes = 8;
struct LookupSlot {
  std::atomic<uintptr_t> thread{0};
  LookupMemory* memory = nullptr;
};
LookupSlot g_lookup_slots[kLookupSlots];
inline uintptr_t this_thread() { return reinterpret_cast<uintptr_t>(__builtin_thread_pointer()); }
inline size_t first_slot(uintptr_t t) { return static_cast<size_t>((t >> 6) ^ (t >> 16)) % 1024; }   // control blocks are 64-byte aligned
struct LookupOwner {   // gives the slot and the memory back when the thread ends
  uintptr_t thread = 0;
  LookupMemory* memory = nullptr;
  size_t slot = 0;
  bool ended = false;
  ~LookupOwner() {
    ended = true;
    if (!memory) return;
    g_lookup_slots[slot].thread.store(0, std::memory_order_release);
    delete memory;
  }
};
[[gnu::noinline]] LookupMemory* make_lookups() {
  thread_local LookupOwner owner;
  if (owner.ended || owner.memory) return &g_ended_lookups;
  const uintptr_t t = this_thread();
  for (size_t p = 0; p < kLookupProbes; ++p) {
    const size_t i = (first_slot(t) + p) % kLookupSlots;
    uintptr_t free_slot = 0;
    if (g_lookup_slots[i].thread.compare_exchange_strong(free_slot, t, std::memory_order_acq_rel)) {
      owner.memory = g_lookup_slots[i].memory = new LookupMemory;
      owner.thread = t;
      owner.slot = i;
      return owner.memory;
    }
  }
  return &g_ended_lookups;
}
inline LookupMemory& lookups() {
  const uintptr_t t = this_thread();
  const size_t h = first_slot(t);
  for (size_t p = 0; p < kLookupProbes; ++p) {
    const LookupSlot& s = g_lookup_slots[(h + p) % kLookupSlots];
    if (s.thread.load(std::memory_order_relaxed) == t) return *s.memory;
  }
  return *make_lookups();
}
}  // namespace

bool MemoryManager::lookup_cache_default() {
  const char* v = std::getenv("VGPU_LOOKUP_CACHE");
  return !(v && v[0] == '0' && v[1] == '\0');
}

// The table lookup behind a miss in the memory of recent lookups.
[[gnu::noinline]] const MemoryManager::Allocation* MemoryManager::find_ordinary_miss(uint64_t addr, uint64_t len,
                                                                                       uint64_t* base_out) const {
  BigReaderLock* lock = table_lock_.get();
  LookupMemory& mem = lookups();
  const uint64_t epoch = lock->epoch();
  for (unsigned i = 0; i < LookupMemory::kWays; ++i) {
    LookupMemory::Entry& e = mem.way[i];
    const uint64_t off = addr - e.base;
    if (off >= e.size || len > e.size - off) continue;
    if (e.epoch != epoch) {
      // Stale, but perhaps only by the frees of other allocations (a launch frees its argument block
      // every time): then it is still right, and is remembered under the new epoch.
      uint64_t now = 0;
      if (&mem == &g_ended_lookups || e.epoch == 0 || !lock->still_valid(e.epoch, e.base, &now)) continue;
      e.epoch = now;
    }
    mem.last = i;
    if (base_out) *base_out = e.base;
    return static_cast<const Allocation*>(e.allocation);
  }
  if (lock->written_by_me()) return nullptr;
  SharedGuard table_guard(lock);
  auto up = live_.upper_bound(addr);
  if (up == live_.begin()) return nullptr;
  auto prev = std::prev(up);
  const Allocation& a = prev->second;
  const uint64_t off = addr - prev->first;
  if (!(off < a.size && len <= a.size - off)) return nullptr;
  if (&mem == &g_ended_lookups) {
    if (base_out) *base_out = prev->first;
    return &a;
  }
  // The epoch read under the shared lock cannot move while it is held.
  uint32_t v = mem.victim;
  v ^= v << 13, v ^= v >> 17, v ^= v << 5;
  mem.victim = v;
  mem.last = v % LookupMemory::kWays;
  LookupMemory::Entry& e = mem.way[mem.last];
  e.epoch = lock->epoch();
  e.base = prev->first;
  e.size = a.size;
  e.allocation = &a;
  if (base_out) *base_out = prev->first;
  return &a;
}

[[gnu::always_inline]] inline const MemoryManager::Allocation* MemoryManager::find_ordinary(
    uint64_t addr, uint64_t len, uint64_t* base_out) const {
  BigReaderLock* lock = table_lock_.get();
  if (!lock || !lookup_cache_) return nullptr;
  LookupMemory& mem = lookups();
  const uint64_t epoch = lock->epoch();
  const LookupMemory::Entry& e = mem.way[mem.last];
  const uint64_t off = addr - e.base;
  // Written as a remaining length, as resolve() does: addr + len wraps.
  if (e.epoch == epoch && off < e.size && len <= e.size - off) {
    if (base_out) *base_out = e.base;
    return static_cast<const Allocation*>(e.allocation);
  }
  return find_ordinary_miss(addr, len, base_out);
}

// The bytes of a naturally aligned scalar in a chunk that is real, when the last allocation this
// thread found holds it and nothing else needs a say (no host mapping in reach): the answer the
// general path gives for such an access, without its calls. Null for anything else -- another
// allocation, an untouched or uniform chunk, a miss -- and the caller takes the general path.
[[gnu::always_inline]] inline uint8_t* MemoryManager::hot_location(uint64_t addr, uint32_t size) const {
  BigReaderLock* lock = table_lock_.get();
  if (!lock || !lookup_cache_ || (host_maps_ && host_maps_->may_contain(addr))) return nullptr;
  const LookupMemory& mem = lookups();
  const LookupMemory::Entry& e = mem.way[mem.last];
  const uint64_t off = addr - e.base;
  if (e.epoch != lock->epoch() || off >= e.size || size > e.size - off) return nullptr;
  const Allocation* a = static_cast<const Allocation*>(e.allocation);
  uint8_t* chunk = a->chunks[off / kChunkSize].load(std::memory_order_acquire);
  if (!chunk || is_uniform(chunk)) return nullptr;
  return chunk + off % kChunkSize;
}

const MemoryManager::Allocation& MemoryManager::resolve_scalar(uint64_t addr, uint64_t len, const char* op,
                                                               uint64_t* base_out, bool writing,
                                                               bool page_slack) const {
  if (const Allocation* a = find_ordinary(addr, len, base_out)) return *a;
  return resolve(addr, len, op, base_out, writing, page_slack);
}

// A warp's accesses at once -- the lanes of one load or store instruction -- for
// the case where nothing can differ from doing them one by one: no fault armed,
// no host mapping in reach, every access aligned and inside the one ordinary
// allocation the first lies in. Anything else returns false having changed
// nothing, and the caller makes the accesses one at a time, errors and all.
bool MemoryManager::batch_ok(const uint64_t* addrs, size_t count, uint32_t size, bool store,
                             const Allocation** alloc, uint64_t* base) const {
  if (count == 0 || (size != 1 && size != 2 && size != 4 && size != 8)) return false;
  if (store) {
    if (access_fault_ && access_fault_->store_pending &&
        __atomic_load_n(access_fault_->store_pending, __ATOMIC_RELAXED))
      return false;
  } else if (access_fault_ &&
             (any_stuck() || (access_fault_->load_pending &&
                              __atomic_load_n(access_fault_->load_pending, __ATOMIC_RELAXED)))) {
    return false;
  }
  const Allocation* a = find_ordinary(addrs[0], size, base);
  if (!a) return false;
  // One pass for the three tests every address must pass: aligned, inside the allocation (an address below its base wraps to
  // an offset far above any size), and not in reach of a host mapping -- which the range of the lot answers unless it overlaps
  // the maps' range, and then each address is asked.
  uint64_t bits = 0, max_off = 0, lo = addrs[0], hi = addrs[0];
  for (size_t i = 0; i < count; ++i) {
    const uint64_t off = addrs[i] - *base;
    bits |= addrs[i];
    max_off = off > max_off ? off : max_off;
    lo = addrs[i] < lo ? addrs[i] : lo;
    hi = addrs[i] > hi ? addrs[i] : hi;
  }
  if ((bits & (size - 1)) || max_off >= a->size || size > a->size - max_off) return false;
  if (host_maps_ && host_maps_->may_overlap(lo, hi))
    for (size_t i = 0; i < count; ++i)
      if (host_maps_->may_contain(addrs[i])) return false;
  *alloc = a;
  return true;
}

bool MemoryManager::load_batch(const uint64_t* addrs, size_t count, uint32_t size, uint64_t* out) const {
  const Allocation* a = nullptr;
  uint64_t base = 0;
  if (!batch_ok(addrs, count, size, /*store=*/false, &a, &base)) return false;
  // (The width a constant, so each load is the one instruction.)
  const auto each = [&](auto width) {
    constexpr uint32_t kSize = decltype(width)::value;
    for (size_t i = 0; i < count; ++i) {
      const uint64_t off = addrs[i] - base;
      const uint8_t* chunk = a->chunks[off / kChunkSize].load(std::memory_order_acquire);
      if (!chunk) out[i] = 0;
      else if (is_uniform(chunk)) out[i] = repeated(uniform_byte(chunk), kSize);
      else out[i] = load_at(chunk + off % kChunkSize, kSize);
    }
  };
  switch (size) {
    case 1: each(std::integral_constant<uint32_t, 1>()); break;
    case 2: each(std::integral_constant<uint32_t, 2>()); break;
    case 4: each(std::integral_constant<uint32_t, 4>()); break;
    default: each(std::integral_constant<uint32_t, 8>()); break;
  }
  return true;
}

bool MemoryManager::store_batch(const uint64_t* addrs, size_t count, uint32_t size, const uint64_t* vals,
                                size_t* failed_at) {
  const Allocation* ca = nullptr;
  uint64_t base = 0;
  if (!batch_ok(addrs, count, size, /*store=*/true, &ca, &base)) return false;
  Allocation& a = const_cast<Allocation&>(*ca);
  for (size_t i = 0; i < count; ++i) {
    const uint64_t off = addrs[i] - base;
    const uint64_t idx = off / kChunkSize;
    uint8_t* chunk = a.chunks[idx].load(std::memory_order_acquire);
    // As store_scalar: a uniform chunk is left alone by a store of its own byte.
    if (is_uniform(chunk) && vals[i] == repeated(uniform_byte(chunk), size)) continue;
    if (!chunk || is_uniform(chunk)) {
      try {
        chunk = materialize(a, idx);
      } catch (...) {
        // Out of host memory for the chunk: as at this access in a one-by-one
        // store, with the earlier ones done. The caller adds its context to the
        // error for the access that failed.
        if (failed_at) *failed_at = i;
        throw;
      }
    }
    store_at(chunk + off % kChunkSize, size, vals[i]);
  }
  return true;
}

MemoryManager::Allocation& MemoryManager::resolve_mut(uint64_t addr, uint64_t len, const char* op,
                                                      uint64_t* base_out) {
  return const_cast<Allocation&>(resolve(addr, len, op, base_out, /*writing=*/true));
}

uint8_t* MemoryManager::materialize(Allocation& a, uint64_t chunk_idx) {
  // A chunk becomes real on the first write that needs its bytes: from nothing,
  // which reads as zero, or from a uniform chunk, which reads as its byte. Two
  // threads can race here; the loser gives its copy back and uses the winner's,
  // so the pointer a reader sees is always the one that stays -- and a reader
  // that saw the tag read the same byte the real chunk starts with. The backing
  // hands chunks out zeroed, from the heap or, past its RAM limit, from disk.
  uint8_t* expected = a.chunks[chunk_idx].load(std::memory_order_acquire);
  while (!expected || is_uniform(expected)) {
    uint8_t* fresh = backing::acquire();
    const bool wide = expected && uniform_byte(expected);
    if (wide) std::memset(fresh, uniform_byte(expected), kChunkSize);
    if (a.chunks[chunk_idx].compare_exchange_strong(expected, fresh, std::memory_order_acq_rel))
      return fresh;
    backing::release(fresh, wide ? backing::kAllBytesDirty : 0);
  }
  return expected;
}

// A device store, atomic or copy into memory registered read-only. The card faults the kernel and
// the context ends (cudaErrorLaunchFailure), so this is a trap, not a bad pointer.
[[noreturn]] static void throw_read_only_host_write(uint64_t addr, uint64_t len) {
  throw Error::make(Err::Trap, "write of ", len, " bytes at ", Hex{addr},
                    " to host memory registered with cudaHostRegisterReadOnly");
}

void MemoryManager::fill(uint64_t dst, const uint8_t* pattern, uint32_t pattern_len, uint64_t len) {
  if (len == 0) return;
  // Host memory mapped here is filled where it is, the pattern's phase
  // running on across pieces.
  if (std::vector<HostPiece> pieces; host_pieces(dst, len, &pieces, true)) {
    for (const HostPiece& p : pieces)
      for (uint64_t i = 0; i < p.len; ++i) p.host[i] = pattern[(p.offset + i) % pattern_len];
    return;
  }
  uint64_t base = 0;
  Allocation& a = resolve_mut(dst, len, "device memory fill", &base);
  bool all_zero = true;
  for (uint32_t i = 0; i < pattern_len; ++i)
    if (pattern[i] != 0) all_zero = false;
  uint64_t off = dst - base;
  uint64_t done = 0;  // bytes filled so far; fixes the pattern's phase
  while (len > 0) {
    uint64_t chunk_idx = off / kChunkSize;
    uint64_t chunk_off = off % kChunkSize;
    uint64_t n = std::min(len, kChunkSize - chunk_off);
    uint8_t* chunk = a.chunks[chunk_idx].load(std::memory_order_acquire);
    if (pattern_len == 1 && chunk_off == 0 && n == kChunkSize && !a.external) {
      // A whole chunk of one byte is that byte: no RAM, no disk. Whatever it
      // held goes back to the backing, and zero is simply untouched memory.
      // Host calls do not run beside kernels -- launches are synchronous -- so
      // nothing is reading the chunk being replaced.
      uint8_t* old = a.chunks[chunk_idx].exchange(pattern[0] ? uniform_chunk(pattern[0]) : nullptr,
                                                  std::memory_order_acq_rel);
      if (old && !is_uniform(old)) backing::release(old);
    } else if (pattern_len == 1 && is_uniform(chunk) && uniform_byte(chunk) == pattern[0]) {
      // Already that byte throughout.
    } else if (chunk || !all_zero) {
      // Zeroing a chunk that was never touched is a no-op that would otherwise
      // cost 64 KiB of host RAM to express.
      if (!chunk || is_uniform(chunk)) chunk = materialize(a, chunk_idx);
      if (pattern_len == 1) {
        std::memset(chunk + chunk_off, pattern[0], static_cast<size_t>(n));
      } else {
        // The pattern repeats with its own length, so write one period at the right phase and
        // double it with memcpy rather than taking a remainder per byte.
        uint8_t* p = chunk + chunk_off;
        const uint64_t first = std::min<uint64_t>(n, pattern_len);
        for (uint64_t i = 0; i < first; ++i) p[i] = pattern[(done + i) % pattern_len];
        for (uint64_t filled = first; filled < n;) {
          const uint64_t c = std::min(filled, n - filled);
          std::memcpy(p + filled, p, static_cast<size_t>(c));
          filled += c;
        }
      }
    }
    off += n;
    done += n;
    len -= n;
  }
}

uint64_t MemoryManager::resident_bytes() const {
  SharedGuard table_guard(table_lock_.get());
  uint64_t chunks = 0;
  for (const auto& [base, a] : live_) {
    (void)base;
    if (!a.chunks) continue;
    for (size_t i = 0; i < a.chunk_count; ++i) {
      const uint8_t* c = a.chunks[i].load(std::memory_order_relaxed);
      if (c && !is_uniform(c)) ++chunks;
    }
  }
  return chunks * kChunkSize;
}

void MemoryManager::write(uint64_t dst, const void* src, uint64_t len) {
  if (len == 0) return;
  if (host_maps_ && host_maps_->may_contain(dst)) {
    std::lock_guard<std::mutex> lock(host_maps_->mu);
    if (const HostMap* m = find_host_map_locked(dst, len)) {
      if (m->read_only) throw_read_only_host_write(dst, len);
      std::memcpy(m->host + (dst - m->base), src, len);
      return;
    }
  }
  if (std::vector<HostPiece> pieces; src && host_pieces(dst, len, &pieces, true)) {
    for (const HostPiece& p : pieces) std::memcpy(p.host, static_cast<const uint8_t*>(src) + p.offset, p.len);
    return;
  }
  // A null host buffer would be dereferenced by the memcpy below and take the
  // process down with a signal, losing the diagnosis. Saying which argument was
  // null is the whole point of running on a simulator.
  if (!src)
    throw Error::make(Err::InvalidPointer,
                      "device memory write from a NULL host pointer (", len, " bytes to ",
                      Hex{dst}, ")");
  uint64_t base = 0;
  Allocation& a = resolve_mut(dst, len, "device memory write", &base);
  const uint8_t* s = static_cast<const uint8_t*>(src);
  uint64_t off = dst - base;
  while (len > 0) {
    uint64_t chunk_idx = off / kChunkSize;
    uint64_t chunk_off = off % kChunkSize;
    uint64_t n = std::min(len, kChunkSize - chunk_off);
    uint8_t* chunk = a.chunks[chunk_idx].load(std::memory_order_acquire);
    // Writing a uniform chunk's own byte changes nothing, and must not turn
    // 64 KiB of nothing into a chunk: a stress test rewrites its fill pattern.
    if (!(is_uniform(chunk) && all_bytes(s, n, uniform_byte(chunk)))) {
      if (!chunk || is_uniform(chunk)) chunk = materialize(a, chunk_idx);
      std::memcpy(chunk + chunk_off, s, n);
    }
    s += n;
    off += n;
    len -= n;
  }
}

bool MemoryManager::host_pieces(uint64_t addr, uint64_t len, std::vector<HostPiece>* pieces,
                                bool for_write) const {
  if (!host_maps_ || !host_maps_->may_contain(addr)) return false;
  std::lock_guard<std::mutex> lock(host_maps_->mu);
  pieces->clear();
  for (uint64_t at = addr, left = len; left;) {
    const HostMap* m = find_host_map_locked(at, 1);
    if (!m) return false;
    if (for_write && m->read_only) throw_read_only_host_write(at, left);
    const uint64_t n = std::min(left, m->base + m->len - at);
    pieces->push_back({m->host + (at - m->base), at - addr, n});
    at += n;
    left -= n;
  }
  return true;
}

const MemoryManager::HostMap* MemoryManager::find_host_map_locked(uint64_t addr,
                                                                   uint64_t len) const {
  if (!host_maps_) return nullptr;
  for (const HostMap& m : host_maps_->maps)
    if (addr >= m.base && addr + len <= m.base + m.len) return &m;
  return nullptr;
}

void MemoryManager::map_host(uint64_t addr, void* host, uint64_t len, bool read_only) {
  std::lock_guard<std::mutex> lock(host_maps_->mu);
  host_maps_->maps.push_back(HostMap{addr, len, static_cast<uint8_t*>(host), read_only});
  // The allocator hands freed host addresses out again, so an old record of a
  // free at this address now describes someone else's live buffer.
  std::erase_if(host_maps_->retired, [&](const HostMap& r) {
    return r.base < addr + len && addr < r.base + r.len;
  });
  if (addr < host_maps_->lo.load(std::memory_order_relaxed))
    host_maps_->lo.store(addr, std::memory_order_relaxed);
  if (addr + len > host_maps_->hi.load(std::memory_order_relaxed))
    host_maps_->hi.store(addr + len, std::memory_order_relaxed);
  if (addr < kDeviceVaEnd && addr + len > kDeviceVaBase)
    host_maps_->in_device_window.store(true, std::memory_order_relaxed);
}

void MemoryManager::unmap_host(uint64_t addr) {
  if (!host_maps_) return;
  std::lock_guard<std::mutex> lock(host_maps_->mu);
  auto& v = host_maps_->maps;
  for (size_t i = 0; i < v.size(); ++i)
    if (v[i].base == addr) {
      auto& retired = host_maps_->retired;
      if (retired.size() >= kRetiredHostMaps) retired.erase(retired.begin());
      retired.push_back(HostMap{v[i].base, v[i].len, nullptr});
      v.erase(v.begin() + static_cast<long>(i));
      break;
    }
}

void MemoryManager::free_all() {
  // The heap's blocks go in the sweep below with every other allocation; its
  // record of them, and the budget they held, go here.
  std::lock_guard<std::mutex> heap_guard(heap_->mu);
  heap_->used = 0;
  heap_->blocks.clear();
  heap_->strings.clear();   // the allocations go in the sweep below
  ExclusiveGuard table_guard(table_lock_.get());
  // A reset takes mapped memory, reservations and handles with it, as it takes
  // allocations: nothing survives it on a real device either.
  maps_.clear();
  reserved_.clear();
  for (auto& [id, h] : handles_) used_ -= h.size;
  handles_.clear();

  // Exported allocations first: they are not in live_, and a reset takes them.
  while (true) {
    auto it = std::find_if(shared_.begin(), shared_.end(),
                           [](const auto& kv) { return kv.second.owner; });
    if (it == shared_.end()) break;
    used_ -= it->second.size;
    drop_shared(it);
  }
  std::vector<uint64_t> bases;
  bases.reserve(live_.size());
  for (const auto& [base, a] : live_) bases.push_back(base);
  for (uint64_t base : bases) free(base);
}

// ---- virtual memory management ------------------------------------------------
//
// Validation is CUDA's: sizes and addresses are multiples of the granularity,
// a mapping goes inside a reservation, and nothing overlaps.

namespace {
uint64_t round_up(uint64_t v, uint64_t to) { return (v + to - 1) / to * to; }
}  // namespace

uint64_t MemoryManager::reserve(uint64_t size, uint64_t alignment) {
  ExclusiveGuard table_guard(table_lock_.get());
  if (size == 0 || size % kVmmGranularity)
    throw Error::make(Err::InvalidValue, "reserving ", size,
                      " bytes of address space: the size must be a non-zero multiple of the ",
                      kVmmGranularity, "-byte granularity");
  // Measured on an RTX 3060: every reservation lands on a 2 MiB boundary at least, whatever smaller
  // alignment was asked for; one that is not a power of two is refused, a larger one honoured.
  if (alignment & (alignment - 1))
    throw Error::make(Err::InvalidValue, "address space alignment ", alignment, " is not a power of two");
  if (alignment < kVmmGranularity) alignment = kVmmGranularity;
  const uint64_t base = round_up(next_va_, alignment);
  // Address space is this device's 1 TiB window; a reservation past it would
  // hand out addresses another device owns.
  if (base + size > va_base_ + kDeviceVaStride || base + size < base)
    throw Error::make(Err::OutOfMemory, "reserving ", size,
                      " bytes of address space would run past this device's window");
  next_va_ = base + size;
  high_water_va_ = next_va_;
  reserved_.emplace(base, Reservation{size});
  return base;
}

bool MemoryManager::range_mapped(uint64_t va, uint64_t size) const {
  SharedGuard table_guard(table_lock_.get());
  auto after = maps_.lower_bound(va);
  if (after != maps_.end() && after->first < va + size) return true;
  if (after != maps_.begin()) {
    auto before = std::prev(after);
    if (va - before->first < before->second.size) return true;
  }
  return false;
}

void MemoryManager::address_free(uint64_t va, uint64_t size) {
  ExclusiveGuard table_guard(table_lock_.get());
  auto it = reserved_.find(va);
  if (it == reserved_.end() || it->second.size != size)
    throw Error::make(Err::InvalidValue, "freeing address space at ", Hex{va}, " of ", size,
                      " bytes: no reservation of exactly that address and size");
  auto m = maps_.lower_bound(va);
  if (m != maps_.end() && m->first < va + size)
    throw Error::make(Err::InvalidValue, "freeing address space at ", Hex{va},
                      " while ", Hex{m->first}, " is still mapped (cuMemUnmap first)");
  reserved_.erase(it);
}

uint64_t MemoryManager::create_handle(uint64_t size, int shareable_types, bool compressible) {
  ExclusiveGuard table_guard(table_lock_.get());
  if (size == 0 || size % kVmmGranularity)
    throw Error::make(Err::InvalidValue, "creating ", size,
                      " bytes of device memory: the size must be a non-zero multiple of the ",
                      kVmmGranularity, "-byte granularity");
  if (used_ + size > capacity_ || used_ + size < used_)
    throw Error::make(Err::OutOfMemory, "device out of memory: requested ", size, " bytes, ", used_,
                      " of ", capacity_, " bytes already in use");
  used_ += size;
  Handle h;
  h.size = size;
  h.types = shareable_types;
  h.compressible = compressible;
  h.mem.size = size;
  h.mem.chunk_count = static_cast<size_t>((size + kChunkSize - 1) / kChunkSize);
  h.mem.chunks = std::make_unique<std::atomic<uint8_t*>[]>(h.mem.chunk_count);
  const uint64_t id = next_handle_++;
  handles_.emplace(id, std::move(h));
  notify_usage();
  return id;
}

int MemoryManager::handle_types(uint64_t handle) const {
  SharedGuard table_guard(table_lock_.get());
  auto it = handles_.find(handle);
  if (it == handles_.end())
    throw Error::make(Err::InvalidValue, "no such memory handle: ", handle);
  return it->second.types;
}

bool MemoryManager::handle_compressible(uint64_t handle) const {
  SharedGuard table_guard(table_lock_.get());
  auto it = handles_.find(handle);
  if (it == handles_.end())
    throw Error::make(Err::InvalidValue, "no such memory handle: ", handle);
  return it->second.compressible;
}

// Moves a handle's memory into an anonymous file mapped shared, once. The chunks become pieces of
// that one mapping (Allocation::external), so reads, writes and kernels reach it as they reach any
// handle's memory, and a process that maps the file reaches the same bytes.
int MemoryManager::export_handle(uint64_t handle) {
  ExclusiveGuard table_guard(table_lock_.get());
  auto it = handles_.find(handle);
  if (it == handles_.end())
    throw Error::make(Err::InvalidValue, "no such memory handle: ", handle);
  Handle& h = it->second;
  if (!(h.types & 1))
    throw Error::make(Err::InvalidValue, "exporting memory handle ", handle,
                      ", which was not created with a handle type to export it as");
  if (h.fd < 0) {
    const int fd = ::memfd_create("vgpu-vmm", MFD_CLOEXEC);
    if (fd < 0)
      throw Error::make(Err::Internal, "exporting a memory handle: memfd_create: ", std::strerror(errno));
    void* host = nullptr;
    if (::ftruncate(fd, static_cast<off_t>(h.size)) == 0) {
      host = ::mmap(nullptr, h.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      if (host == MAP_FAILED) host = nullptr;
    }
    if (!host) {
      const int err = errno;
      ::close(fd);
      throw Error::make(Err::Internal, "exporting a memory handle: could not map the file: ",
                        std::strerror(err));
    }
    // What is there now goes into the file; a file reads as zeros, so only what differs is written.
    Allocation& a = h.mem;
    for (size_t i = 0; i < a.chunk_count; ++i) {
      uint8_t* old = a.chunks[i].load(std::memory_order_acquire);
      uint8_t* dst = static_cast<uint8_t*>(host) + i * kChunkSize;
      const uint64_t n = std::min<uint64_t>(kChunkSize, h.size - i * kChunkSize);
      if (old && is_uniform(old)) {
        if (uniform_byte(old)) std::memset(dst, uniform_byte(old), n);
      } else if (old) {
        std::memcpy(dst, old, n);
        backing::release(old);
      }
      a.chunks[i].store(dst, std::memory_order_release);
    }
    a.external = static_cast<uint8_t*>(host);
    a.external_len = h.size;
    h.fd = fd;
  }
  const int out = ::dup(h.fd);
  if (out < 0)
    throw Error::make(Err::Internal, "exporting a memory handle: dup: ", std::strerror(errno));
  return out;
}

uint64_t MemoryManager::import_handle(int fd) {
  struct stat st;
  if (fd < 0 || ::fstat(fd, &st) != 0 || st.st_size <= 0)
    throw Error::make(Err::NotFound, "importing a memory handle: ", fd, " is not an exported handle");
  const uint64_t size = static_cast<uint64_t>(st.st_size);
  ExclusiveGuard table_guard(table_lock_.get());
  if (size % kVmmGranularity)
    throw Error::make(Err::NotFound, "importing a memory handle: the file is ", size,
                      " bytes, not a whole number of the ", kVmmGranularity, "-byte granularity");
  if (used_ + size > capacity_ || used_ + size < used_)
    throw Error::make(Err::OutOfMemory, "device out of memory: requested ", size, " bytes, ", used_,
                      " of ", capacity_, " bytes already in use");
  void* host = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (host == MAP_FAILED)
    throw Error::make(Err::NotFound, "importing a memory handle: could not map ", fd, ": ", std::strerror(errno));
  const int keep = ::fcntl(fd, F_DUPFD_CLOEXEC, 0);   // the importer's own reference to the file
  used_ += size;
  Handle h;
  h.size = size;
  h.types = 1;
  h.fd = keep;
  h.mem.size = size;
  h.mem.chunk_count = static_cast<size_t>((size + kChunkSize - 1) / kChunkSize);
  h.mem.chunks = std::make_unique<std::atomic<uint8_t*>[]>(h.mem.chunk_count);
  for (size_t i = 0; i < h.mem.chunk_count; ++i)
    h.mem.chunks[i].store(static_cast<uint8_t*>(host) + i * kChunkSize, std::memory_order_relaxed);
  h.mem.external = static_cast<uint8_t*>(host);
  h.mem.external_len = size;
  const uint64_t id = next_handle_++;
  handles_.emplace(id, std::move(h));
  notify_usage();
  return id;
}

uint64_t MemoryManager::handle_size(uint64_t handle) const {
  SharedGuard table_guard(table_lock_.get());
  auto it = handles_.find(handle);
  if (it == handles_.end())
    throw Error::make(Err::InvalidValue, "no such memory handle: ", handle);
  return it->second.size;
}

void MemoryManager::retain_handle(uint64_t handle) {
  ExclusiveGuard table_guard(table_lock_.get());
  auto it = handles_.find(handle);
  if (it == handles_.end())
    throw Error::make(Err::InvalidValue, "no such memory handle: ", handle);
  ++it->second.refs;
}

uint64_t MemoryManager::retain_handle_at(uint64_t va) {
  ExclusiveGuard table_guard(table_lock_.get());
  auto mu = maps_.upper_bound(va);
  if (mu == maps_.begin()) return 0;
  auto prev = std::prev(mu);
  if (va - prev->first >= prev->second.size) return 0;
  const uint64_t handle = prev->second.handle;
  retain_handle(handle);
  return handle;
}

void MemoryManager::release_handle(uint64_t handle) {
  ExclusiveGuard table_guard(table_lock_.get());
  auto it = handles_.find(handle);
  if (it == handles_.end())
    throw Error::make(Err::InvalidValue, "releasing memory handle ", handle,
                      ", which does not exist (already released?)");
  if (it->second.refs == 0)
    throw Error::make(Err::InvalidValue, "releasing memory handle ", handle,
                      ", which holds no references");
  --it->second.refs;
  collect_handle(handle);
}

void MemoryManager::collect_handle(uint64_t handle) {
  ExclusiveGuard table_guard(table_lock_.get());
  auto it = handles_.find(handle);
  if (it == handles_.end() || it->second.refs || it->second.mapped) return;
  used_ -= it->second.size;
  handles_.erase(it);
  notify_usage();
}

void MemoryManager::map(uint64_t va, uint64_t size, uint64_t offset, uint64_t handle) {
  ExclusiveGuard table_guard(table_lock_.get());
  if (size == 0 || size % kVmmGranularity || va % kVmmGranularity)
    throw Error::make(Err::InvalidValue, "mapping ", size, " bytes at ", Hex{va},
                      ": the address and size must be multiples of the ", kVmmGranularity,
                      "-byte granularity");
  // cuMemMap documents that the offset into the handle must be zero.
  if (offset != 0)
    throw Error::make(Err::InvalidValue, "mapping at an offset of ", offset,
                      " bytes into a handle: cuMemMap takes an offset of 0");
  auto h = handles_.find(handle);
  if (h == handles_.end())
    throw Error::make(Err::InvalidValue, "mapping memory handle ", handle, ", which does not exist");
  if (size > h->second.size - offset)
    throw Error::make(Err::InvalidValue, "mapping ", size, " bytes of memory handle ", handle,
                      ", which holds ", h->second.size, " bytes");
  // Inside one reservation, and it must be.
  auto ru = reserved_.upper_bound(va);
  bool inside = false;
  if (ru != reserved_.begin()) {
    auto prev = std::prev(ru);
    inside = va - prev->first < prev->second.size && va + size <= prev->first + prev->second.size;
  }
  if (!inside)
    throw Error::make(Err::InvalidValue, "mapping ", size, " bytes at ", Hex{va},
                      ": the range is not inside one reservation (cuMemAddressReserve)");
  // Nothing else may be mapped there.
  auto after = maps_.lower_bound(va);
  if (after != maps_.end() && after->first < va + size)
    throw Error::make(Err::InvalidValue, "mapping ", size, " bytes at ", Hex{va},
                      ": ", Hex{after->first}, " is already mapped");
  if (after != maps_.begin()) {
    auto before = std::prev(after);
    if (va - before->first < before->second.size)
      throw Error::make(Err::InvalidValue, "mapping ", size, " bytes at ", Hex{va},
                        ": it overlaps the mapping at ", Hex{before->first});
  }
  maps_.emplace(va, Mapping{size, offset, handle, false, false});
  ++h->second.mapped;
}

void MemoryManager::unmap(uint64_t va, uint64_t size) {
  ExclusiveGuard table_guard(table_lock_.get());
  // The range is whole mappings, one after another: a program that maps a
  // pool a chunk at a time (ggml's VMM pool) unmaps it all in one call.
  std::vector<uint64_t> handles;
  uint64_t at = va;
  while (size && at < va + size) {
    auto it = maps_.find(at);
    if (it == maps_.end() || at + it->second.size > va + size)
      throw Error::make(Err::InvalidValue, "unmapping ", size, " bytes at ", Hex{va},
                        ": no mapping of exactly that address and size (a range is whole mappings, and no mapping ",
                        it == maps_.end() ? "starts at " : "ends where the range does at ", Hex{at}, ")");
    handles.push_back(it->second.handle);
    at += it->second.size;
  }
  if (handles.empty())
    throw Error::make(Err::InvalidValue, "unmapping ", size, " bytes at ", Hex{va}, ": no mapping of exactly that address and size");
  for (uint64_t a = va; a < va + size;) {
    auto it = maps_.find(a);
    a += it->second.size;
    maps_.erase(it);
  }
  for (uint64_t handle : handles) {
    auto h = handles_.find(handle);
    if (h != handles_.end() && h->second.mapped) --h->second.mapped;
    collect_handle(handle);
  }
}

void MemoryManager::set_access(uint64_t va, uint64_t size, bool readable, bool writable) {
  ExclusiveGuard table_guard(table_lock_.get());
  if (size == 0 || size % kVmmGranularity || va % kVmmGranularity)
    throw Error::make(Err::InvalidValue, "granting access to ", size, " bytes at ", Hex{va},
                      ": the address and size must be multiples of the ", kVmmGranularity,
                      "-byte granularity");
  // Every byte of the range has to be mapped: granting access to address space
  // with nothing behind it would leave a pointer that faults on first use.
  uint64_t at = va;
  while (at < va + size) {
    auto it = maps_.find(at);
    if (it == maps_.end() || it->second.size > va + size - at)
      throw Error::make(Err::InvalidValue, "granting access to ", size, " bytes at ", Hex{va},
                        ": ", Hex{at}, " is not the start of a mapping inside that range");
    it->second.readable = readable;
    it->second.writable = writable;
    at += it->second.size;
  }
}

bool MemoryManager::access_at(uint64_t va, bool* readable, bool* writable) const {
  SharedGuard table_guard(table_lock_.get());
  auto mu = maps_.upper_bound(va);
  if (mu == maps_.begin()) return false;
  auto prev = std::prev(mu);
  if (va - prev->first >= prev->second.size) return false;
  if (readable) *readable = prev->second.readable;
  if (writable) *writable = prev->second.writable;
  return true;
}

// ---- memory another process can map -------------------------------------------

uint64_t MemoryManager::share(uint64_t ptr, const std::string& path) {
  ExclusiveGuard table_guard(table_lock_.get());
  if (const auto it = shared_.find(ptr); it != shared_.end()) return it->second.size;
  const auto live = live_.find(ptr);
  if (live == live_.end())
    throw Error::make(Err::InvalidPointer, "sharing device pointer ", Hex{ptr},
                      " that is not the base of a live allocation");
  const uint64_t size = live->second.size;
  const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
  if (fd < 0)
    throw Error::make(Err::Internal, "sharing device memory: could not create ", path, ": ",
                      std::strerror(errno));
  // Sparse: the file is as large as the allocation, and costs only the pages
  // that are written -- the same property the sparse chunks have.
  void* host = nullptr;
  if (::ftruncate(fd, static_cast<off_t>(size)) == 0) {
    host = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (host == MAP_FAILED) host = nullptr;
  }
  const int err = errno;
  ::close(fd);
  if (!host) {
    ::unlink(path.c_str());
    throw Error::make(Err::Internal, "sharing device memory: could not map ", path, ": ",
                      std::strerror(err));
  }
  // Move what is there now, then let the allocation's chunks go: from here on
  // the file is the memory, at the same device address. The new file reads
  // as zeros, so only chunks that hold something else are copied: writing the
  // zeros of chunks never touched would make every page of a sparse file
  // real (a 1 GiB NVSHMEM heap filled a 2.4 GiB tmpfs that way).
  for (uint64_t off = 0; off < size; off += kChunkSize) {
    const uint8_t* chunk = live->second.chunks[off / kChunkSize].load(std::memory_order_acquire);
    if (!chunk || (is_uniform(chunk) && uniform_byte(chunk) == 0)) continue;
    read_chunks(live->second, off, static_cast<uint8_t*>(host) + off, std::min(kChunkSize, size - off));
  }
  live_.erase(live);
  map_host(ptr, host, size);
  shared_[ptr] = SharedRegion{ptr, size, host, path, /*owner=*/true};
  return size;
}

bool MemoryManager::is_shared(uint64_t ptr) const {
  SharedGuard table_guard(table_lock_.get());
  return shared_.count(ptr) != 0;
}

uint64_t MemoryManager::adopt(const std::string& path, uint64_t size) {
  ExclusiveGuard table_guard(table_lock_.get());
  const int fd = ::open(path.c_str(), O_RDWR);
  if (fd < 0)
    throw Error::make(Err::InvalidValue, "opening shared device memory ", path, ": ",
                      std::strerror(errno));
  void* host = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  const int err = errno;
  ::close(fd);
  if (host == MAP_FAILED)
    throw Error::make(Err::Internal, "mapping shared device memory ", path, ": ",
                      std::strerror(err));
  // An address of this device's own, never reused, so the imported buffer can
  // never be confused with a local allocation -- present or freed.
  const uint64_t va = next_va_;
  next_va_ += (size + kAllocAlign - 1) / kAllocAlign * kAllocAlign;
  high_water_va_ = next_va_;
  map_host(va, host, size);
  shared_[va] = SharedRegion{va, size, host, path, /*owner=*/false};
  return va;
}

bool MemoryManager::unlink_shared(uint64_t ptr) {
  ExclusiveGuard table_guard(table_lock_.get());
  const auto it = shared_.find(ptr);
  if (it == shared_.end() || !it->second.owner) return false;
  ::unlink(it->second.path.c_str());
  return true;
}

void MemoryManager::abandon(uint64_t va) {
  ExclusiveGuard table_guard(table_lock_.get());
  const auto it = shared_.find(va);
  if (it == shared_.end() || it->second.owner)
    throw Error::make(Err::InvalidPointer, "closing shared device memory at ", Hex{va},
                      ": this process did not map it from another");
  drop_shared(it);
}

void MemoryManager::drop_shared(std::map<uint64_t, SharedRegion>::iterator it) {
  unmap_host(it->second.va);
  ::munmap(it->second.host, it->second.size);
  if (it->second.owner) ::unlink(it->second.path.c_str());
  shared_.erase(it);
}

uint8_t* MemoryManager::host_address(uint64_t addr, uint64_t len) const {
  if (!host_maps_->may_contain(addr)) return nullptr;
  std::lock_guard<std::mutex> lock(host_maps_->mu);
  const HostMap* m = find_host_map_locked(addr, len);
  // The one caller that gets a pointer to write through is an atomic.
  if (m && m->read_only) throw_read_only_host_write(addr, len);
  return m ? m->host + (addr - m->base) : nullptr;
}

bool MemoryManager::is_host_mapped(uint64_t addr) const {
  // Nothing mapped near it: no lock, as before any buffer was mapped at all.
  if (!host_maps_->may_contain(addr)) return false;
  std::lock_guard<std::mutex> lock(host_maps_->mu);
  return find_host_map_locked(addr, 1) != nullptr;
}

void MemoryManager::read(uint64_t src, void* dst, uint64_t len) const {
  if (len == 0) return;
  // Managed memory is the caller's own buffer; there is no chunk table to walk.
  if (host_maps_ && host_maps_->may_contain(src)) {
    std::lock_guard<std::mutex> lock(host_maps_->mu);
    if (const HostMap* m = find_host_map_locked(src, len)) {
      std::memcpy(dst, m->host + (src - m->base), len);
      return;
    }
  }
  if (std::vector<HostPiece> pieces; dst && host_pieces(src, len, &pieces)) {
    for (const HostPiece& p : pieces) std::memcpy(static_cast<uint8_t*>(dst) + p.offset, p.host, p.len);
    return;
  }
  if (!dst)
    throw Error::make(Err::InvalidPointer,
                      "device memory read into a NULL host pointer (", len, " bytes from ",
                      Hex{src}, ")");
  uint64_t base = 0;
  const Allocation& a = resolve(src, len, "device memory read", &base);
  uint8_t* d = static_cast<uint8_t*>(dst);
  read_chunks(a, src - base, d, len);
  if (any_stuck()) access_fault_->on_read(src - va_base_, d, len);
  // Only copies read device memory in bulk -- kernels read it a scalar at a
  // time -- so this is where a fault armed on copies is taken.
  if (access_fault_ && access_fault_->copy_pending &&
      __atomic_load_n(access_fault_->copy_pending, __ATOMIC_RELAXED))
    access_fault_->on_copy(src, d, len);
}

namespace {
// A run's bytes between chunk memory and a buffer, as relaxed atomics: an
// element (`unit` bytes, aligned) at a time up to an 8-byte boundary, whole
// words -- which never split an aligned element -- and elements again.
void run_out(const uint8_t* c, uint8_t* d, uint64_t n, uint32_t unit) {
  uint64_t i = 0;
  for (; i < n && reinterpret_cast<uintptr_t>(c + i) % 8; i += unit) {
    const uint64_t v = load_at(c + i, unit);
    std::memcpy(d + i, &v, unit);
  }
  for (; i + 8 <= n; i += 8) {
    const uint64_t v = relaxed_load<uint64_t>(c + i);
    std::memcpy(d + i, &v, 8);
  }
  for (; i < n; i += unit) {
    const uint64_t v = load_at(c + i, unit);
    std::memcpy(d + i, &v, unit);
  }
}
void run_in(uint8_t* c, const uint8_t* s, uint64_t n, uint32_t unit) {
  auto one = [&](uint64_t i, uint32_t size) {
    uint64_t v = 0;
    std::memcpy(&v, s + i, size);
    store_at(c + i, size, v);
  };
  uint64_t i = 0;
  for (; i < n && reinterpret_cast<uintptr_t>(c + i) % 8; i += unit) one(i, unit);
  for (; i + 8 <= n; i += 8) one(i, 8);
  for (; i < n; i += unit) one(i, unit);
}
}  // namespace

// The ordinary allocation holding [addr, addr + len), or null. Mapped
// (cuMemMap) ranges are not in live_, so they are never found here.
const MemoryManager::Allocation* MemoryManager::run_allocation(uint64_t addr, uint64_t len, uint32_t unit,
                                                               uint64_t* base) const {
  if (unit != 1 && unit != 2 && unit != 4 && unit != 8) return nullptr;
  // A fault armed for loads or stores, or stuck cells, change what an access returns or stores, and
  // load_scalar / store_scalar apply them; the runtime attaches its faults to every device, so it is
  // whether one is armed that counts, not whether any is attached (as load_quad decides).
  const auto armed = [](const uint64_t* pending) { return pending && __atomic_load_n(pending, __ATOMIC_RELAXED); };
  if (access_fault_ && (armed(access_fault_->stuck_pending) || armed(access_fault_->load_pending) ||
                        armed(access_fault_->store_pending)))
    return nullptr;
  if (addr % unit || len % unit || (host_maps_ && host_maps_->may_contain(addr))) return nullptr;
  if (const Allocation* a = find_ordinary(addr, len, base)) return a;
  SharedGuard table_guard(table_lock_.get());
  auto up = live_.upper_bound(addr);
  if (up == live_.begin()) return nullptr;
  auto prev = std::prev(up);
  *base = prev->first;
  if (addr - *base >= prev->second.size || len > prev->second.size - (addr - *base)) return nullptr;
  return &prev->second;
}

bool MemoryManager::load_run(uint64_t addr, void* dst, uint64_t len, uint32_t unit) const {
  if (len == 0) return true;
  uint64_t base = 0;
  const Allocation* a = run_allocation(addr, len, unit, &base);
  if (!a) return false;
  uint8_t* d = static_cast<uint8_t*>(dst);
  for (uint64_t off = addr - base; len > 0;) {
    const uint64_t n = std::min(len, kChunkSize - off % kChunkSize);
    const uint8_t* chunk = a->chunks[off / kChunkSize].load(std::memory_order_acquire);
    if (!chunk) std::memset(d, 0, n);   // untouched: reads zero
    else if (is_uniform(chunk)) std::memset(d, uniform_byte(chunk), n);
    else run_out(chunk + off % kChunkSize, d, n, unit);
    d += n;
    off += n;
    len -= n;
  }
  return true;
}

bool MemoryManager::store_run(uint64_t addr, const void* src, uint64_t len, uint32_t unit) {
  if (len == 0) return true;
  uint64_t base = 0;
  auto* a = const_cast<Allocation*>(run_allocation(addr, len, unit, &base));
  if (!a) return false;
  const uint8_t* s = static_cast<const uint8_t*>(src);
  for (uint64_t off = addr - base; len > 0;) {
    const uint64_t n = std::min(len, kChunkSize - off % kChunkSize);
    const uint64_t idx = off / kChunkSize;
    uint8_t* chunk = a->chunks[idx].load(std::memory_order_acquire);
    // As store_scalar: storing a uniform chunk's own byte changes nothing;
    // anything else makes the chunk real first.
    if (!(is_uniform(chunk) && all_bytes(s, n, uniform_byte(chunk)))) {
      if (!chunk || is_uniform(chunk)) chunk = materialize(*a, idx);
      run_in(chunk + off % kChunkSize, s, n, unit);
    }
    s += n;
    off += n;
    len -= n;
  }
  return true;
}

void MemoryManager::read_chunks(const Allocation& a, uint64_t off, uint8_t* d, uint64_t len) const {
  while (len > 0) {
    uint64_t chunk_idx = off / kChunkSize;
    uint64_t chunk_off = off % kChunkSize;
    uint64_t n = std::min(len, kChunkSize - chunk_off);
    const uint8_t* chunk = a.chunks[chunk_idx].load(std::memory_order_acquire);
    if (!chunk)
      std::memset(d, 0, n);  // untouched device memory reads as zero (documented)
    else if (is_uniform(chunk))
      std::memset(d, uniform_byte(chunk), n);
    else
      std::memcpy(d, chunk + chunk_off, n);
    d += n;
    off += n;
    len -= n;
  }
}

bool MemoryManager::find_allocation(uint64_t addr, uint64_t* base, uint64_t* size) const {
  SharedGuard table_guard(table_lock_.get());
  auto up = live_.upper_bound(addr);
  if (up != live_.begin()) {
    auto prev = std::prev(up);
    if (addr < prev->first + prev->second.size) {
      if (base) *base = prev->first;
      if (size) *size = prev->second.size;
      return true;
    }
  }
  // Memory exported for another process to map (cudaIpcGetMemHandle) leaves the chunk table for
  // a file of its own, and an imported one lives only there; both are still allocations.
  auto sh = shared_.upper_bound(addr);
  if (sh == shared_.begin()) return false;
  --sh;
  if (addr >= sh->second.va + sh->second.size) return false;
  if (base) *base = sh->second.va;
  if (size) *size = sh->second.size;
  return true;
}

uint64_t MemoryManager::load_scalar(uint64_t addr, uint32_t size) const {
  if (size != 1 && size != 2 && size != 4 && size != 8)
    throw Error::make(Err::Internal, "load_scalar: bad size ", size);
  if ((addr & (size - 1)) != 0)   // size is 1, 2, 4 or 8, checked above
    throw Error::make(Err::MisalignedAccess, "load of ", size, " bytes at ", Hex{addr},
                      " is not naturally aligned (real GPUs fault on this)");
  const uint8_t* p = nullptr;
  uint64_t v = 0;
  if ((p = hot_location(addr, size)) != nullptr) {
    v = load_at(p, size);
  } else switch (scalar_location(addr, size, /*store=*/nullptr, &p)) {
    case ScalarAt::Chunk:
      v = load_at(p, size);
      break;
    case ScalarAt::Uniform:
      v = repeated(uniform_byte(p), size);
      break;
    case ScalarAt::Untouched:
      // Zero, answered here. Falling back to read() instead was a race: a
      // first store on another thread could materialize the chunk between
      // this check and read()'s copy, which then memcpy'd bytes an atomic
      // store was writing. ThreadSanitizer caught it on main.
      v = 0;
      break;
    case ScalarAt::Unchanged:  // only answered for a store
    case ScalarAt::HostMap:
      // Host memory, mapped: no device ECC covers it, so no fault is taken.
      // Read as an atomic of its width, as a chunk's scalar is: the host, or
      // another process's kernels on shared memory, may be updating it with
      // the CPU's atomics -- which is how the engines make a device atomic on
      // such memory (vgpu/exec/host_atomic.hpp) -- and a memcpy racing those
      // is a data race (ThreadSanitizer, e2e_host_atomics).
      {
        std::lock_guard<std::mutex> lock(host_maps_->mu);
        if (const HostMap* m = find_host_map_locked(addr, size)) return load_at(m->host + (addr - m->base), size);
      }
      read(addr, &v, size);  // little-endian host assumption, documented in ARCHITECTURE.md
      return v;
  }
  // What the cells hold, stuck bits included (little-endian, as above), and
  // then whatever fault is armed for the load.
  if (any_stuck()) access_fault_->on_read(addr - va_base_, reinterpret_cast<uint8_t*>(&v), size);
  if (access_fault_ && access_fault_->load_pending &&
      __atomic_load_n(access_fault_->load_pending, __ATOMIC_RELAXED))
    return access_fault_->on_load(addr, size, v);
  return v;
}

void MemoryManager::store_scalar(uint64_t addr, uint32_t size, uint64_t value) {
  if (size != 1 && size != 2 && size != 4 && size != 8)
    throw Error::make(Err::Internal, "store_scalar: bad size ", size);
  if ((addr & (size - 1)) != 0)   // size is 1, 2, 4 or 8, checked above
    throw Error::make(Err::MisalignedAccess, "store of ", size, " bytes at ", Hex{addr},
                      " is not naturally aligned (real GPUs fault on this)");
  // A fault armed on stores changes what reaches device memory, so every later
  // read finds it. Mapped host memory has no device ECC and takes none.
  if (access_fault_ && access_fault_->store_pending &&
      __atomic_load_n(access_fault_->store_pending, __ATOMIC_RELAXED) && !is_host_mapped(addr))
    value = access_fault_->on_store(addr, size, value);
  if (uint8_t* hot = hot_location(addr, size)) {
    store_at(hot, size, value);
    return;
  }
  const uint8_t* p = nullptr;
  switch (scalar_location(addr, size, &value, &p)) {
    case ScalarAt::Chunk:
      store_at(const_cast<uint8_t*>(p), size, value);
      return;
    case ScalarAt::Unchanged:
      return;
    case ScalarAt::HostMap: {
      // An atomic store of its width, as load_scalar's load is.
      std::lock_guard<std::mutex> lock(host_maps_->mu);
      if (const HostMap* m = find_host_map_locked(addr, size)) {
        if (m->read_only) throw_read_only_host_write(addr, size);
        store_at(m->host + (addr - m->base), size, value);
        return;
      }
      break;
    }
    default:
      break;
  }
  write(addr, &value, size);  // a managed host mapping, behind its own lock
}

void MemoryManager::load_quad(uint64_t addr, uint64_t out[2]) const {
  // A fault armed for loads, or stuck cells, change what a load returns, and
  // load_scalar applies them; the runtime attaches its faults to every device,
  // so it is whether one is armed that counts, not whether any is attached.
  const bool faulted = access_fault_ && (any_stuck() || (access_fault_->load_pending &&
                                                         __atomic_load_n(access_fault_->load_pending, __ATOMIC_RELAXED)));
  if (addr % 16 || faulted || (host_maps_ && host_maps_->may_contain(addr))) {
    out[0] = load_scalar(addr, 8), out[1] = load_scalar(addr + 8, 8);
    return;
  }
  // Sixteen aligned bytes never straddle a chunk (kChunkSize is a multiple
  // of 16).
  uint64_t base = 0;
  const Allocation& a = resolve_scalar(addr, 16, "device memory read", &base, /*writing=*/false, /*page_slack=*/true);
  const uint64_t off = addr - base;
  const uint8_t* chunk = a.chunks[off / kChunkSize].load(std::memory_order_acquire);
  if (!chunk) {
    out[0] = out[1] = 0;
  } else if (is_uniform(chunk)) {
    out[0] = out[1] = repeated(uniform_byte(chunk), 8);
  } else {
    load16(chunk + off % kChunkSize, out);
  }
}

void MemoryManager::store_quad(uint64_t addr, const uint64_t in[2]) {
  const bool faulted = access_fault_ && access_fault_->store_pending &&
                       __atomic_load_n(access_fault_->store_pending, __ATOMIC_RELAXED);
  if (addr % 16 || faulted || (host_maps_ && host_maps_->may_contain(addr))) {
    store_scalar(addr, 8, in[0]), store_scalar(addr + 8, 8, in[1]);
    return;
  }
  uint64_t base = 0;
  auto& a = const_cast<Allocation&>(
      resolve_scalar(addr, 16, "device memory write", &base, /*writing=*/false, /*page_slack=*/false));
  const uint64_t off = addr - base;
  const uint64_t chunk_idx = off / kChunkSize;
  uint8_t* chunk = a.chunks[chunk_idx].load(std::memory_order_acquire);
  if (is_uniform(chunk) && in[0] == in[1] && in[0] == repeated(uniform_byte(chunk), 8)) return;
  if (!chunk || is_uniform(chunk)) chunk = materialize(a, chunk_idx);
  store16(chunk + off % kChunkSize, in);
}

bool MemoryManager::load_quad_run(uint64_t addr, size_t count, uint64_t* out) const {
  if (!count || addr % 16 || (host_maps_ && host_maps_->may_contain(addr + 16 * (count - 1)))) return false;
  uint64_t base = 0;
  const Allocation* a = run_allocation(addr, 16 * count, 8, &base);   // (also declines while a fault is armed)
  if (!a || (host_maps_ && host_maps_->may_contain(addr))) return false;
  for (size_t i = 0; i < count; ++i) {
    const uint64_t off = addr + 16 * i - base;
    const uint8_t* chunk = a->chunks[off / kChunkSize].load(std::memory_order_acquire);
    if (!chunk) {
      out[2 * i] = out[2 * i + 1] = 0;
    } else if (is_uniform(chunk)) {
      out[2 * i] = out[2 * i + 1] = repeated(uniform_byte(chunk), 8);
    } else {
      load16(chunk + off % kChunkSize, out + 2 * i);
    }
  }
  return true;
}

bool MemoryManager::store_quad_run(uint64_t addr, size_t count, const uint64_t* in) {
  if (!count || addr % 16 || (host_maps_ && host_maps_->may_contain(addr + 16 * (count - 1)))) return false;
  uint64_t base = 0;
  Allocation* a = const_cast<Allocation*>(run_allocation(addr, 16 * count, 8, &base));
  if (!a || (host_maps_ && host_maps_->may_contain(addr))) return false;
  for (size_t i = 0; i < count; ++i) {
    const uint64_t off = addr + 16 * i - base;
    const uint64_t chunk_idx = off / kChunkSize;
    uint8_t* chunk = a->chunks[chunk_idx].load(std::memory_order_acquire);
    if (is_uniform(chunk) && in[2 * i] == in[2 * i + 1] && in[2 * i] == repeated(uniform_byte(chunk), 8)) continue;
    if (!chunk || is_uniform(chunk)) chunk = materialize(*a, chunk_idx);
    store16(chunk + off % kChunkSize, in + 2 * i);
  }
  return true;
}

// Where a kernel's scalar load or store lives: bytes in a chunk (materialized
// first when this is a store), memory nothing has written (only ever answered
// for a load), or managed memory -- the caller's own buffer, which takes the
// general path behind its lock.
//
// Why this exists. Blocks run on several host threads, and a kernel is allowed
// to share memory between blocks -- CUB's decoupled look-back scan, under
// Thrust's sort and CUB's radix sort, publishes each block's prefix for later
// blocks to spin on. These accesses used to be a plain memcpy, which made that
// a C++ data race: undefined behaviour, reported by ThreadSanitizer the first
// time CI ran it. It worked in release builds because an aligned 8-byte copy
// is one instruction on x86, which nothing promises -- a torn read of a tile
// status is a wrong prefix or a spin that never ends. A GPU's accesses are
// atomic at this granularity; so are these now, relaxed, which on x86 is the
// same single instruction and costs nothing. Ordering is the fences' job (see
// the interpreter), not the load's.
//
// A naturally aligned scalar cannot straddle a chunk -- kChunkSize is a
// multiple of 8 -- so it is always one location in one chunk.
MemoryManager::ScalarAt MemoryManager::scalar_location(uint64_t addr, uint32_t size,
                                                       const uint64_t* store,
                                                       const uint8_t** where) const {
  const bool create = store != nullptr;
  if (host_maps_ && host_maps_->may_contain(addr)) {
    std::lock_guard<std::mutex> lock(host_maps_->mu);
    if (find_host_map_locked(addr, size)) return ScalarAt::HostMap;
  }
  uint64_t base = 0;
  auto& a = const_cast<Allocation&>(resolve_scalar(addr, size, create ? "device memory write" : "device memory read",
                                                   &base, /*writing=*/false, /*page_slack=*/!create));
  const uint64_t off = addr - base;
  const uint64_t chunk_idx = off / kChunkSize;
  uint8_t* chunk = a.chunks[chunk_idx].load(std::memory_order_acquire);
  if (is_uniform(chunk)) {
    if (!create) { *where = chunk; return ScalarAt::Uniform; }
    if (*store == repeated(uniform_byte(chunk), size)) return ScalarAt::Unchanged;
  }
  if (!chunk || is_uniform(chunk)) {
    if (!chunk && !create) return ScalarAt::Untouched;
    chunk = const_cast<MemoryManager*>(this)->materialize(a, chunk_idx);
  }
  *where = chunk + off % kChunkSize;
  return ScalarAt::Chunk;
}

// ---- read-modify-write and multi-field reads: load_scalar / store_scalar with one lookup ----

bool MemoryManager::rmw_locate(uint64_t addr, uint32_t size, RmwSlot* s) const {
  if ((size != 1 && size != 2 && size != 4 && size != 8) || (addr & (size - 1))) return false;
  if (access_faults_armed()) return false;
  if (host_maps_ && host_maps_->may_contain(addr)) return false;
  uint64_t base = 0;
  const Allocation* a = find_ordinary(addr, size, &base);
  if (!a) return false;
  const uint64_t off = addr - base;
  s->a = const_cast<Allocation*>(a);
  s->chunk_idx = off / kChunkSize;
  s->coff = off % kChunkSize;
  s->chunk = a->chunks[s->chunk_idx].load(std::memory_order_acquire);
  return true;
}

bool MemoryManager::rmw_scalar_fn(uint64_t addr, uint32_t size, RmwFn compute, void* ctx, uint64_t* old_out) {
  RmwSlot s;
  if (!rmw_locate(addr, size, &s)) return false;
  const uint64_t old = rmw_load(s, size);
  *old_out = old;
  rmw_store(s, size, compute(ctx, old));
  return true;
}

bool MemoryManager::atomic_rmw_fn(uint64_t addr, uint32_t size, RmwFn compute, void* ctx, uint64_t* old_out,
                                  bool always_store) {
  if (size != 4 && size != 8) return false;
  RmwSlot s;
  if (!rmw_locate(addr, size, &s)) return false;
  for (;;) {
    if (s.chunk && !is_uniform(s.chunk)) {
      uint8_t* p = s.chunk + s.coff;
      if (size == 4) {
        uint32_t* w = reinterpret_cast<uint32_t*>(p);
        uint32_t old = __atomic_load_n(w, __ATOMIC_RELAXED);
        for (;;) {
          const uint32_t now = static_cast<uint32_t>(compute(ctx, old));
          if (now == old) break;   // nothing to change: the load was the whole operation
          if (__atomic_compare_exchange_n(w, &old, now, true, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED)) break;
        }
        *old_out = old;
      } else {
        uint64_t* w = reinterpret_cast<uint64_t*>(p);
        uint64_t old = __atomic_load_n(w, __ATOMIC_RELAXED);
        for (;;) {
          const uint64_t now = compute(ctx, old);
          if (now == old) break;
          if (__atomic_compare_exchange_n(w, &old, now, true, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED)) break;
        }
        *old_out = old;
      }
      return true;
    }
    // Nothing real to swap on: what the chunk reads as, and whether the result would change it.
    const uint64_t old = rmw_load(s, size);
    const uint64_t now = compute(ctx, old);
    const bool uniform = s.chunk != nullptr;
    if (uniform ? now == old : (now == old && !always_store)) {
      *old_out = old;
      return true;
    }
    // The store makes the chunk real; the swap then runs on it, from whatever it holds by then.
    s.chunk = materialize(*s.a, s.chunk_idx);
  }
}

uint64_t MemoryManager::rmw_load(const RmwSlot& s, uint32_t size) const {
  if (!s.chunk) return 0;
  if (is_uniform(s.chunk)) return repeated(uniform_byte(s.chunk), size);
  return load_at(s.chunk + s.coff, size);
}

void MemoryManager::rmw_store(RmwSlot& s, uint32_t size, uint64_t value) {
  uint8_t* chunk = s.chunk;
  // A chunk that was real stays so; one that was not may have been made real since.
  if (!chunk || is_uniform(chunk)) chunk = s.a->chunks[s.chunk_idx].load(std::memory_order_acquire);
  if (is_uniform(chunk)) {
    // As store_scalar: a uniform chunk is left alone by a store of its own byte.
    if (value == repeated(uniform_byte(chunk), size)) return;
  }
  if (!chunk || is_uniform(chunk)) chunk = materialize(*s.a, s.chunk_idx);
  store_at(chunk + s.coff, size, value);
}

bool MemoryManager::load_fields(uint64_t addr, const uint8_t* sizes, unsigned n, uint64_t* out) const {
  uint64_t total = 0;
  for (unsigned i = 0; i < n; ++i) {
    const uint32_t sz = sizes[i];
    if (sz == 0) continue;
    if ((sz != 1 && sz != 2 && sz != 4 && sz != 8) || ((addr + total) & (sz - 1))) return false;
    if (host_maps_ && host_maps_->may_contain(addr + total)) return false;
    total += sz;
  }
  if (total == 0) {
    for (unsigned i = 0; i < n; ++i) out[i] = 0;
    return true;
  }
  if (access_faults_armed()) return false;
  uint64_t base = 0;
  const Allocation* a = find_ordinary(addr, total, &base);
  if (!a) return false;
  uint64_t off = addr - base;
  for (unsigned i = 0; i < n; ++i) {
    const uint32_t sz = sizes[i];
    if (sz == 0) {
      out[i] = 0;
      continue;
    }
    const uint8_t* chunk = a->chunks[off / kChunkSize].load(std::memory_order_acquire);
    if (!chunk) out[i] = 0;
    else if (is_uniform(chunk)) out[i] = repeated(uniform_byte(chunk), sz);
    else out[i] = load_at(chunk + off % kChunkSize, sz);
    off += sz;
  }
  return true;
}

}  // namespace vgpu
