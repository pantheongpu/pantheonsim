// Exact parallel launches: the shadow of device memory a trial of one launch is run under, and its undo log
// (vgpu/memory.hpp, "exact parallel launches"; docs/exact-threads.md has the argument for why a trial that commits
// left the bits of the serial run).
//
// The shadow has two levels. A page state stands for 4 KiB of the device's address range [va_base_, high_water_va_):
//     0                 no block has touched the page
//     b + 1             block b has read in it; no other block has touched it; nothing is noted by word
//     kMany             more than one block has read in it, and none has written in it; nothing is noted by word
//     kWords            the page is noted by word (below), and every word says what was done to its bytes
//     kWordsCoarse      the same, after one block read in the page by page state and then wrote in it: its reads were
//                       not noted by word, so every word started as "read by that block" (more than it read)
// A page moves 0 -> one reader -> many readers by compare-and-swap, or 0 -> kWords (the first block that writes in it,
// or, where the trial's plan says by word, the first that touches it), or (its only reader, about to write) its own
// reader mark -> kWordsCoarse; kMany, kWords and kWordsCoarse are final. A write is only made in a page noted by word.
// So a page some block read by page state is written by no other block: that block would have to bring the page to a
// word state, which only the page's sole reader can do, and it fills the words with its own reader mark first.
//
// In a page noted by word, one 32-bit word stands for four bytes:
//     0                 no block has touched the bytes
//     b + 1             block b has read them, and no other block has touched them
//     kMany             more than one block has read them, and none has written them
//     kWrite | (b + 1)  block b has written them (and may have read them); no other block has touched them
// A word only moves up that list (0 -> one reader -> many readers; 0 or its own reader mark -> its writer), by
// compare-and-swap, and a block makes its access only after the page and the word allow it. So for any two blocks and
// any byte:
//   * both read it: nothing to tell apart;
//   * one writes it and the other reads or writes it: whichever came second to the page or to the word found a state
//     it may not enter, and stopped the trial BEFORE touching the byte.
// A stopped trial is undone from the bytes each block kept when it first wrote a word (it owned the word then, so
// what it kept is what the launch started with), and the launch is run again in order.
//
// A stop at a page state, or at a word of a kWordsCoarse page that says "read", may be the note's fault and not the
// kernel's (the page says more was read than was). Such a stop is marked coarse and names the allocation; the caller
// may run the trial again with that allocation noted by word from the start (SpecPlan), which is the exact note.
#include "vgpu/memory.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include <sys/mman.h>

namespace vgpu {

class MemoryManager::Spec {
 public:
  // A mark of the shadow: 32 bits, read and moved only by the atomic builtins. (Not std::atomic: two marks side by side are also
  // read and taken as one 64-bit value, `Pair` below, which the builtins allow on a type that may alias.)
  struct Word {
    uint32_t v;
    uint32_t load(std::memory_order) const { return __atomic_load_n(&v, __ATOMIC_ACQUIRE); }
    void store(uint32_t x, std::memory_order) { __atomic_store_n(&v, x, __ATOMIC_RELEASE); }
    bool compare_exchange_weak(uint32_t& expected, uint32_t desired, std::memory_order) {
      return __atomic_compare_exchange_n(&v, &expected, desired, true, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    }
    bool compare_exchange_strong(uint32_t& expected, uint32_t desired, std::memory_order) {
      return __atomic_compare_exchange_n(&v, &expected, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    }
  };
  // Two neighbouring words of a page (the first at an even index, so the pair is aligned), the lower index in the low half.
  typedef uint64_t __attribute__((may_alias)) Pair;
  static Pair* pair_at(Word* first) { return reinterpret_cast<Pair*>(first); }
  static constexpr uint32_t kMany = 0xFFFFFFFFu, kWrite = 0x80000000u;
  // Page states only (no block's mark: run_exact tries fewer than 0x7FFFFFF0 blocks).
  static constexpr uint32_t kWords = 0xFFFFFFFEu, kWordsCoarse = 0xFFFFFFFDu;
  static bool by_words(uint32_t state) { return state == kWords || state == kWordsCoarse; }
  static constexpr uint64_t kGranule = 4;          // bytes a word stands for
  static constexpr uint64_t kPageWords = 1024;     // words of a shadow page: 4 KiB of device memory, 4 KiB of shadow
  static constexpr uint64_t kPageBytes = kGranule * kPageWords;
  static constexpr size_t kPoolPages = 65536;      // shadow pages kept between trials (256 MiB of host memory at most)
  static constexpr uint64_t kKeepLimit = uint64_t{1} << 30;   // bytes one worker may keep for undoing before the trial stops
  static constexpr size_t kSeen = 256;             // pages a worker remembers it may read without a look at the shadow
  static constexpr size_t kRanges = 4;             // runs of words a worker remembers its block has noted
  static constexpr uint64_t kRunWorth = 16;        // ... of at least so many words: a warp's lanes one after another, not one lane's
                                                   // few words (32 lanes' runs would push one another out, and asking costs each of them)
  static constexpr size_t kWordPages = 64;         // pages of words a worker remembers where to find
  static constexpr size_t kCoarseKept = 64;        // addresses of coarse stops kept for the caller

  // A page the worker's block may read with no further note: the page's state was the block's own reader mark (`tag` is the block's)
  // or kMany (`tag` is kMany: any block's, for the rest of the trial) when it was looked at.
  struct Seen {
    uint64_t slot = ~uint64_t{0};
    uint32_t tag = 0;
  };

  // Where the worker keeps its memory of a page: a warp's lanes are often a fixed number of pages apart (rows of a matrix), which
  // the low bits of the page number would send to a few places.
  static size_t seen_at(uint64_t slot) { return static_cast<size_t>((slot * 0x9E3779B97F4A7C15ull) >> 56); }

  // A run of words [g0, g1] (granules of the range) every one of which block `tag` has noted: as its writer (the worker's `wrote`:
  // each word holds the block's writer mark, which nothing changes), or for reading (`read`: each held the block's reader or
  // writer mark or the several-readers mark when it was noted, and from those a word only moves to another of them). The block's
  // next access inside the run has nothing to note.
  struct Run {
    uint64_t g0 = 1, g1 = 0;
    uint32_t tag = 0;   // (no block's: marks start at 1)
  };
  static bool known(const Run* runs, uint32_t tag, uint64_t g0, uint64_t g1) {
    for (size_t i = 0; i < kRanges; ++i)
      if (runs[i].tag == tag && g0 >= runs[i].g0 && g1 <= runs[i].g1) return true;
    return false;
  }
  // Two runs of one block that touch or overlap are one run (every word of either is noted).
  static void learn(Run* runs, uint8_t& next, uint32_t tag, uint64_t g0, uint64_t g1) {
    for (size_t i = 0; i < kRanges; ++i)
      if (runs[i].tag == tag && g0 <= runs[i].g1 + 1 && g1 + 1 >= runs[i].g0) {
        runs[i].g0 = std::min(runs[i].g0, g0);
        runs[i].g1 = std::max(runs[i].g1, g1);
        return;
      }
    runs[next] = {g0, g1, tag};
    next = static_cast<uint8_t>((next + 1) % kRanges);
  }

  struct Worker {
    uint32_t tag = 0;                    // block + 1 of the block it runs now
    std::vector<Word*> fresh;            // zeroed pages ready to install
    std::vector<uint64_t> installed;     // the top-table slots whose pages this worker installed
    std::vector<uint64_t> noted;         // the slots whose page state this worker moved from 0
    std::vector<std::pair<uint64_t, uint64_t>> kept_at;   // (address, length) of each kept piece, in the order kept
    std::vector<uint8_t> kept;                            // their bytes, one after another
    // The last page noted by word that was asked for (a block's accesses are mostly near one another).
    uint64_t last_slot = ~uint64_t{0};
    Word* last_page = nullptr;
    // The allocation the last kept piece was in.
    uint64_t keep_base = 1, keep_end = 0;
    const void* keep_alloc = nullptr;
    SpecStats st;
    Run wrote[kRanges], read[kRanges];
    uint8_t wrote_next = 0, read_next = 0;
    // Pages noted by word the worker has read in, and their words (a page's words stay where they are for the trial).
    struct WordPage {
      uint64_t slot = ~uint64_t{0};
      Word* page = nullptr;
    } pages[kWordPages];
    Seen seen[kSeen];

    // Ready for a trial. The undo log keeps its memory from the last trial (a launch that writes much writes as much the next time).
    void reset() {
      tag = 0;
      installed.clear();
      noted.clear();
      kept_at.clear();
      kept.clear();
      last_slot = ~uint64_t{0};
      last_page = nullptr;
      keep_base = 1, keep_end = 0;
      keep_alloc = nullptr;
      st = SpecStats{};
      for (size_t i = 0; i < kRanges; ++i) wrote[i] = read[i] = Run{};
      wrote_next = read_next = 0;
      for (Seen& e : seen) e = Seen{};
      for (WordPage& e : pages) e = WordPage{};
    }
  };

  ~Spec() {
    for (Word* p : pool) std::free(p);
    if (top) ::munmap(top, top_slots * sizeof(std::atomic<Word*>));
    if (pstate) ::munmap(pstate, top_slots * sizeof(Word));
  }

  // The top tables: a slot per shadow page of the range (its state, and its page of words once it is noted by word), zero pages of
  // the kernel's until a slot is written.
  void size_for(uint64_t base_va, uint64_t bytes) {
    base = base_va;
    span = bytes;
    const uint64_t slots = (bytes / kGranule + kPageWords - 1) / kPageWords + 1;
    if (slots <= top_slots) return;
    if (top) ::munmap(top, top_slots * sizeof(std::atomic<Word*>));
    if (pstate) ::munmap(pstate, top_slots * sizeof(Word));
    top = nullptr;
    pstate = nullptr;
    const uint64_t want = std::max<uint64_t>(slots * 2, 4096);
    top_slots = 0;
    void* m = ::mmap(nullptr, want * sizeof(std::atomic<Word*>), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (m == MAP_FAILED) throw std::bad_alloc();
    void* m2 = ::mmap(nullptr, want * sizeof(Word), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (m2 == MAP_FAILED) {
      ::munmap(m, want * sizeof(std::atomic<Word*>));
      throw std::bad_alloc();
    }
    top = static_cast<std::atomic<Word*>*>(m);   // (zero bytes: null pointers)
    pstate = static_cast<Word*>(m2);             // (zero bytes: untouched pages)
    top_slots = want;
  }

  Word* fresh_page(Worker& w) {
    if (w.fresh.empty()) {
      std::lock_guard<std::mutex> g(pool_mu);
      for (int i = 0; i < 16 && !pool.empty(); ++i) {
        w.fresh.push_back(pool.back());
        pool.pop_back();
      }
    }
    if (!w.fresh.empty()) {
      Word* p = w.fresh.back();
      w.fresh.pop_back();
      return p;
    }
    void* m = std::calloc(kPageWords, sizeof(Word));
    if (!m) throw std::bad_alloc();
    return static_cast<Word*>(m);
  }

  // The words of a page whose state is kWords or kWordsCoarse (installed zeroed by whoever comes first), remembered as the worker's last.
  Word* word_page(Worker& w, uint64_t slot) {
    Word* page = top[slot].load(std::memory_order_acquire);
    if (!page) {
      Word* mine = fresh_page(w);
      if (top[slot].compare_exchange_strong(page, mine, std::memory_order_acq_rel)) {
        page = mine;
        w.installed.push_back(slot);
      } else {
        w.fresh.push_back(mine);   // another worker's page is there
      }
    }
    w.last_slot = slot;
    w.last_page = page;
    return page;
  }

  // Whether the page is noted by word from its first touch (the plan of the trial: SpecPlan).
  bool by_word(uint64_t slot) const {
    if (!plan_pages) return true;
    const uint64_t lo = base + slot * kPageBytes, hi = lo + kPageBytes;
    for (const auto& [from, to] : plan_words)
      if (lo < to && from < hi) return true;
    return false;
  }

  [[noreturn, gnu::cold, gnu::noinline]] void refuse(const char* reason) {
    const char* none = nullptr;
    why.compare_exchange_strong(none, reason, std::memory_order_relaxed);
    stopped.store(true, std::memory_order_release);
    throw SpecStop{};
  }
  // A stop the coarseness of a page's note may have made (`addr` is in the page).
  [[noreturn, gnu::cold, gnu::noinline]] void refuse_coarse(uint64_t addr, const char* reason) {
    {
      std::lock_guard<std::mutex> g(pool_mu);
      if (coarse_at.size() < kCoarseKept) coarse_at.push_back(addr);
    }
    coarse.store(true, std::memory_order_release);
    refuse(reason);
  }

  // A block reads bytes [lo, hi] (offsets in the range, both in page `slot`) and the worker has no memory of the page: the note,
  // by the page's state when that is enough and by word when the page is noted by word.
  [[gnu::noinline]] void read_page(Worker& w, uint64_t slot, uint64_t lo, uint64_t hi) {
    const uint32_t tag = w.tag;
    const uint64_t first = lo / kGranule, last = hi / kGranule;
    Word* page;
    if (slot == w.last_slot) {
      page = w.last_page;
    } else {
      Word& ps = pstate[slot];
      uint32_t v = ps.load(std::memory_order_acquire);
      for (;;) {
        if (v == kMany || v == tag) {
          w.seen[seen_at(slot)] = {slot, v};
          return;
        }
        if (by_words(v)) break;
        if (v == 0) {
          const uint32_t to = by_word(slot) ? kWords : tag;
          if (!ps.compare_exchange_weak(v, to, std::memory_order_acq_rel)) continue;
          w.noted.push_back(slot);
          ++(to == tag ? w.st.pages_read : w.st.pages_by_word);
          v = to;
          continue;
        }
        // Another block's reader mark: several readers now.
        if (ps.compare_exchange_weak(v, kMany, std::memory_order_acq_rel)) {
          ++w.st.pages_shared;
          v = kMany;
        }
      }
      page = word_page(w, slot);
    }
    w.pages[seen_at(slot) % kWordPages] = {slot, page};
    const uint64_t g0 = first % kPageWords, g1 = last % kPageWords;
    w.st.words_read += g1 - g0 + 1;
    const auto mine = [tag](uint32_t v) { return v == kMany || v == tag || v == (tag | kWrite); };
    for (uint64_t g = g0; g <= g1; ++g) {
      Word& word = page[g];
      if (!(g & 1) && g < g1) {
        // Two words at once: both readable already, or both untouched and taken by one compare-and-swap.
        Pair* two = pair_at(&word);
        uint64_t v2 = __atomic_load_n(two, __ATOMIC_ACQUIRE);
        if (mine(static_cast<uint32_t>(v2)) && mine(static_cast<uint32_t>(v2 >> 32))) {
          ++g;
          continue;
        }
        if (v2 == 0 && __atomic_compare_exchange_n(two, &v2, uint64_t{tag} | uint64_t{tag} << 32, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
          w.st.words_read_first += 2;
          ++g;
          continue;
        }
        // (anything else: one word at a time, below)
      }
      uint32_t v = word.load(std::memory_order_acquire);
      if (mine(v)) [[likely]] continue;
      for (;;) {
        if (v == 0) {
          if (word.compare_exchange_weak(v, tag, std::memory_order_acq_rel)) {
            ++w.st.words_read_first;
            break;
          }
        } else if (v == kMany || v == tag || v == (tag | kWrite)) {
          break;
        } else if (v & kWrite) {
          refuse("a block read what another block wrote");   // (a writer's mark is always exact: no coarse stop here)
        } else if (word.compare_exchange_weak(v, kMany, std::memory_order_acq_rel)) {
          ++w.st.words_read_shared;
          break;
        }
      }
    }
    if (last - first + 1 >= kRunWorth) learn(w.read, w.read_next, tag, first, last);
  }

  // A block is about to write in page `slot` (`addr` is the store's, for a coarse stop): the page comes to be noted by word, or the
  // trial stops. Leaves the page as the worker's last.
  [[gnu::noinline]] void write_page(Worker& w, uint64_t slot, uint64_t addr) {
    const uint32_t tag = w.tag;
    Word& ps = pstate[slot];
    uint32_t v = ps.load(std::memory_order_acquire);
    for (;;) {
      if (by_words(v)) break;
      if (v == 0) {
        if (!ps.compare_exchange_weak(v, kWords, std::memory_order_acq_rel)) continue;
        w.noted.push_back(slot);
        ++w.st.pages_by_word;
        break;
      }
      if (v == tag) {
        // The page's only reader so far. What it read was not noted by word, so every word starts as read by it; the page of
        // words is put in place before the state says there is one (no one looks for it earlier).
        Word* mine = fresh_page(w);
        for (uint64_t i = 0; i < kPageWords; ++i) mine[i].store(tag, std::memory_order_relaxed);
        Word* none = nullptr;
        if (!top[slot].compare_exchange_strong(none, mine, std::memory_order_acq_rel)) {
          for (uint64_t i = 0; i < kPageWords; ++i) mine[i].store(0, std::memory_order_relaxed);
          w.fresh.push_back(mine);
          refuse_coarse(addr, "a page read by page was found noted by word");   // (cannot happen: said, not assumed)
        }
        w.installed.push_back(slot);
        if (ps.compare_exchange_strong(v, kWordsCoarse, std::memory_order_acq_rel)) {
          ++w.st.pages_coarse;
          break;
        }
        // (another block read in the page meanwhile: v is kMany, and the page of words is never looked at)
      }
      refuse_coarse(addr, "a block wrote in a page other blocks read");
    }
    word_page(w, slot);
  }

  std::atomic<Word*>* top = nullptr;
  Word* pstate = nullptr;
  uint64_t top_slots = 0;
  uint64_t base = 0, span = 0;
  bool host_maps = false;
  bool plan_pages = true;
  std::vector<std::pair<uint64_t, uint64_t>> plan_words;
  std::atomic<bool> stopped{false};
  std::atomic<bool> coarse{false};
  std::atomic<const char*> why{nullptr};
  std::vector<Worker> workers;
  std::mutex pool_mu;
  std::vector<Word*> pool;   // zeroed pages
  std::vector<uint64_t> coarse_at;   // under pool_mu
};

namespace {
thread_local MemoryManager::Spec::Worker* t_spec_worker = nullptr;
}  // namespace

bool MemoryManager::spec_possible() const {
  if (memo_sink_ || memo_untrackable() || memo_faults_armed() || memo_trace_active()) return false;
  SharedGuard table_guard(table_lock_.get());
  for (const auto& [base, a] : live_)
    if (a.external) return false;   // bytes that live in a file another process may map
  return high_water_va_ > va_base_;
}

void MemoryManager::spec_begin(unsigned workers, const SpecPlan* plan) {
  if (!spec_kept_) spec_kept_ = std::make_shared<Spec>();
  Spec& s = *spec_kept_;
  s.size_for(va_base_, high_water_va_ - va_base_);
  s.host_maps = memo_host_maps();
  s.plan_pages = plan ? plan->pages : true;
  s.plan_words.clear();
  if (plan) s.plan_words = plan->by_word;
  s.stopped.store(false, std::memory_order_relaxed);
  s.coarse.store(false, std::memory_order_relaxed);
  s.why.store(nullptr, std::memory_order_relaxed);
  s.coarse_at.clear();
  if (s.workers.size() != workers) {
    s.workers.clear();
    s.workers.resize(workers);
  }
  for (Spec::Worker& w : s.workers) w.reset();
  spec_ = &s;
}

void MemoryManager::spec_enter(unsigned worker) { t_spec_worker = &spec_->workers[worker]; }

void MemoryManager::spec_block(uint64_t block) { t_spec_worker->tag = static_cast<uint32_t>(block + 1); }

void MemoryManager::spec_leave() { t_spec_worker = nullptr; }

bool MemoryManager::spec_stopped() const { return spec_ && spec_->stopped.load(std::memory_order_relaxed); }

void MemoryManager::spec_stop(const char* why) {
  if (!spec_) return;
  const char* none = nullptr;
  spec_->why.compare_exchange_strong(none, why, std::memory_order_relaxed);
  spec_->stopped.store(true, std::memory_order_release);
}

void MemoryManager::spec_refuse(const char* why) const { spec_->refuse(why); }

void MemoryManager::SpecStats::add(const SpecStats& o) {
  hooks += o.hooks;
  pages_read += o.pages_read;
  pages_shared += o.pages_shared;
  pages_by_word += o.pages_by_word;
  pages_coarse += o.pages_coarse;
  page_looks += o.page_looks;
  words_read += o.words_read;
  words_read_first += o.words_read_first;
  words_read_shared += o.words_read_shared;
  words_written += o.words_written;
  words_written_first += o.words_written_first;
  keeps += o.keeps;
  runs_known += o.runs_known;
}

MemoryManager::SpecOutcome MemoryManager::spec_end() {
  Spec& s = *spec_;
  spec_ = nullptr;   // (the workers have left: what follows is this thread's alone, and write() below is no trial's)
  SpecOutcome out;
  out.committed = !s.stopped.load(std::memory_order_acquire);
  if (const char* why = s.why.load(std::memory_order_relaxed)) out.why = why;
  for (Spec::Worker& w : s.workers) {
    out.kept_bytes += w.kept.size();
    out.shadow_pages += w.noted.size();
    out.stats.add(w.st);
    out.stats.word_pages += w.installed.size();
    if (!out.committed) {
      // Each piece is what the launch started with at bytes only this worker's blocks wrote, so the order of putting back is free.
      uint64_t at = 0;
      for (const auto& [addr, len] : w.kept_at) {
        write(addr, w.kept.data() + at, len);
        at += len;
      }
    }
  }
  if (!out.committed && s.coarse.load(std::memory_order_acquire)) {
    // The allocations the coarse stops were in: what a second trial notes by word from the start.
    out.coarse = true;
    SharedGuard table_guard(table_lock_.get());
    for (const uint64_t addr : s.coarse_at) {
      auto up = live_.upper_bound(addr);
      if (up == live_.begin()) continue;
      const auto it = std::prev(up);
      const std::pair<uint64_t, uint64_t> range{it->first, it->first + it->second.size};
      if (addr < range.second && std::find(out.by_word.begin(), out.by_word.end(), range) == out.by_word.end()) out.by_word.push_back(range);
    }
  }
  // The shadow goes back to nothing: every page state is cleared, every installed page is zeroed and pooled, its slot cleared.
  std::lock_guard<std::mutex> g(s.pool_mu);
  for (Spec::Worker& w : s.workers) {
    for (const uint64_t slot : w.noted) s.pstate[slot].store(0, std::memory_order_relaxed);
    for (const uint64_t slot : w.installed) {
      Spec::Word* page = s.top[slot].exchange(nullptr, std::memory_order_acq_rel);
      if (s.pool.size() < Spec::kPoolPages) {
        std::memset(static_cast<void*>(page), 0, Spec::kPageWords * sizeof(Spec::Word));
        s.pool.push_back(page);
      } else {
        std::free(page);
      }
    }
    for (Spec::Word* page : w.fresh) {   // zeroed and never installed
      if (s.pool.size() < Spec::kPoolPages) s.pool.push_back(page);
      else std::free(page);
    }
    w.fresh.clear();
    w.installed.clear();
    w.noted.clear();
    w.kept_at.clear();
    w.kept.clear();
    // (the log's memory stays for the next trial, unless it grew large)
    if (w.kept.capacity() > (size_t{64} << 20)) std::vector<uint8_t>().swap(w.kept);
    if (w.kept_at.capacity() > (size_t{1} << 20)) std::vector<std::pair<uint64_t, uint64_t>>().swap(w.kept_at);
  }
  s.coarse_at.clear();
  return out;
}

void MemoryManager::spec_keep(uint64_t addr, uint64_t len) const {
  Spec::Worker& w = *t_spec_worker;
  ++w.st.keeps;
  if (w.kept.size() + len > Spec::kKeepLimit) spec_refuse("the undo log passed its limit");
  // Wholly inside the allocation the last piece was in (the table cannot change during a launch: an allocation or a free stops the trial).
  if (addr >= w.keep_base && addr + len <= w.keep_end) {
    const size_t at = w.kept.size();
    w.kept.resize(at + len);
    read_chunks(*static_cast<const Allocation*>(w.keep_alloc), addr - w.keep_base, w.kept.data() + at, len);
    w.kept_at.emplace_back(addr, len);
    return;
  }
  SharedGuard table_guard(table_lock_.get());
  const uint64_t end = addr + len;
  while (addr < end) {
    auto up = live_.upper_bound(addr);
    if (up == live_.begin() || addr >= std::prev(up)->first + std::prev(up)->second.size) {
      // No allocation holds this byte (padding after one, a freed range): nothing a store can change. On to the next allocation.
      if (up == live_.end() || up->first >= end) return;
      addr = up->first;
      continue;
    }
    const auto it = std::prev(up);
    w.keep_base = it->first;
    w.keep_end = it->first + it->second.size;
    w.keep_alloc = &it->second;
    const uint64_t n = std::min(end, w.keep_end) - addr;
    const size_t at = w.kept.size();
    w.kept.resize(at + n);
    read_chunks(it->second, addr - it->first, w.kept.data() + at, n);
    w.kept_at.emplace_back(addr, n);
    addr += n;
  }
}

// A block reads [addr, addr + len) (len > 0; the trial is on and this thread is `worker`).
[[gnu::always_inline]] inline void MemoryManager::spec_read_in(Spec& s, void* worker, uint64_t addr, uint64_t len) const {
  const uint64_t off = addr - s.base;
  // Outside the device's range: host memory (pinned, managed), which no block of a trial can write (spec_write_in), or
  // nothing at all, which the access itself reports.
  if (off >= s.span) return;
  Spec::Worker& w = *static_cast<Spec::Worker*>(worker);
  const uint64_t last = std::min(off + len, s.span) - 1;
  const uint64_t slot1 = last / Spec::kPageBytes;
  for (uint64_t slot = off / Spec::kPageBytes; slot <= slot1; ++slot) {
    // A page this block read by page state before, or that several blocks have: nothing more to note (no block can write in it).
    const Spec::Seen& seen = w.seen[Spec::seen_at(slot)];
    if (seen.slot == slot && (seen.tag == w.tag || seen.tag == Spec::kMany)) [[likely]] continue;
    const uint64_t lo = std::max(off, slot * Spec::kPageBytes), hi = std::min(last, slot * Spec::kPageBytes + Spec::kPageBytes - 1);
    const uint64_t g0 = lo / Spec::kGranule, g1 = hi / Spec::kGranule;
    // Words the block is remembered to have noted (a load in a loop).
    if (g1 - g0 + 1 >= Spec::kRunWorth && (Spec::known(w.wrote, w.tag, g0, g1) || Spec::known(w.read, w.tag, g0, g1))) {
      ++w.st.runs_known;
      continue;
    }
    // A page noted by word that the worker has read in before: words that already allow the block's read need nothing more.
    const Spec::Worker::WordPage& wp = w.pages[Spec::seen_at(slot) % Spec::kWordPages];
    if (wp.slot == slot) {
      const uint32_t tag = w.tag;
      uint64_t g = g0 % Spec::kPageWords;
      const uint64_t end = g1 % Spec::kPageWords;
      for (; g <= end; ++g) {
        const uint32_t v = wp.page[g].load(std::memory_order_acquire);
        if (!(v == Spec::kMany || v == tag || v == (tag | Spec::kWrite))) break;
      }
      if (g > end) {
        w.st.words_read += g1 - g0 + 1;
        continue;
      }
      // One untouched word (a lane of a scattered load): taken here as its first reader. (Anything else, and a word another
      // block took meanwhile, goes the long way.)
      if (g0 == g1) {
        uint32_t none = 0;
        if (wp.page[g].compare_exchange_strong(none, tag, std::memory_order_acq_rel)) {
          w.st.words_read += 1;
          w.st.words_read_first += 1;
          continue;
        }
      }
    }
    ++w.st.page_looks;
    s.read_page(w, slot, lo, hi);
  }
}

// A block is about to write [addr, addr + len): its pages come to be noted by word, it comes to own every word of it, and the
// bytes of the words it did not own yet are kept first.
[[gnu::always_inline]] inline void MemoryManager::spec_write_in(Spec& s, void* worker, uint64_t addr, uint64_t len) const {
  const uint64_t off = addr - s.base;
  // A store the shadow has no word for (host memory another thread or process may share; an address that is no
  // allocation's) cannot be undone or checked: the launch runs in order.
  if (off >= s.span || len > s.span - off) spec_refuse("a block wrote outside the device's own memory");
  if (s.host_maps && host_maps_ && host_maps_->may_overlap(addr, addr + len - 1)) spec_refuse("a block wrote host-mapped memory");
  Spec::Worker& w = *static_cast<Spec::Worker*>(worker);
  const uint32_t mine = w.tag | Spec::kWrite;
  const uint64_t g0 = off / Spec::kGranule, g1 = (off + len - 1) / Spec::kGranule;
  // Words the block holds as their writer already (a store in a loop): nothing to note.
  if (g1 - g0 + 1 >= Spec::kRunWorth && Spec::known(w.wrote, w.tag, g0, g1)) {
    ++w.st.runs_known;
    return;
  }
  const uint64_t mine2 = uint64_t{mine} | uint64_t{mine} << 32;
  uint64_t run = ~uint64_t{0};   // the first granule of the run of words taken just now, not kept yet
  const auto keep = [&](uint64_t upto) {
    if (run == ~uint64_t{0}) return;
    spec_keep(s.base + run * Spec::kGranule, (upto - run) * Spec::kGranule);
    run = ~uint64_t{0};
  };
  w.st.words_written += g1 - g0 + 1;
  for (uint64_t g = g0; g <= g1; ++g) {
    const uint64_t slot = g / Spec::kPageWords;
    // (A stop in write_page leaves the words taken in this call unkept: they are still unwritten, so undoing leaves them as they are.)
    if (slot != w.last_slot) [[unlikely]] s.write_page(w, slot, s.base + g * Spec::kGranule);
    Spec::Word& word = w.last_page[g % Spec::kPageWords];
    if (!(g & 1) && g < g1) {
      // Two words at once: both the block's already, or both free for it (untouched, or read by it alone) and taken by one
      // compare-and-swap. (An even index: the pair is in one page.)
      Spec::Pair* two = Spec::pair_at(&word);
      uint64_t v2 = __atomic_load_n(two, __ATOMIC_ACQUIRE);
      if (v2 == mine2) {
        keep(g);
        ++g;
        continue;
      }
      const uint32_t a = static_cast<uint32_t>(v2), b = static_cast<uint32_t>(v2 >> 32);
      if ((a == 0 || a == w.tag) && (b == 0 || b == w.tag) &&
          __atomic_compare_exchange_n(two, &v2, mine2, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        w.st.words_written_first += 2;
        if (run == ~uint64_t{0}) run = g;
        ++g;
        continue;
      }
      // (anything else: one word at a time, below)
    }
    uint32_t v = word.load(std::memory_order_acquire);
    for (;;) {
      if (v == mine) {
        keep(g);
        break;
      }
      if (v == 0 || v == w.tag) {
        if (!word.compare_exchange_weak(v, mine, std::memory_order_acq_rel)) continue;
        ++w.st.words_written_first;
        if (run == ~uint64_t{0}) run = g;
        break;
      }
      // Another block's word, or one several blocks read. The words taken in this call are still unwritten: kept or not, undoing
      // leaves them as they are.
      if (v != Spec::kMany && (v & Spec::kWrite)) spec_refuse("two blocks wrote the same bytes");
      // A reader's mark, which in a page that was first read by page state may say more than was read.
      if (s.pstate[slot].load(std::memory_order_acquire) == Spec::kWordsCoarse)
        s.refuse_coarse(s.base + g * Spec::kGranule, "a block wrote what another block read");
      spec_refuse("a block wrote what another block read");
    }
  }
  keep(g1 + 1);
  if (g1 - g0 + 1 >= Spec::kRunWorth) Spec::learn(w.wrote, w.wrote_next, w.tag, g0, g1);
}

// The trial's worker of the calling thread, after the two tests every hook makes first.
[[gnu::always_inline]] inline void* MemoryManager::spec_worker() const {
  if (spec_->stopped.load(std::memory_order_relaxed)) throw SpecStop{};
  Spec::Worker* w = t_spec_worker;
  if (!w) spec_refuse("a thread outside the launch used device memory");
  ++w->st.hooks;
  return w;
}

void MemoryManager::spec_read(uint64_t addr, uint64_t len) const {
  void* w = spec_worker();
  if (len) spec_read_in(*spec_, w, addr, len);
}

void MemoryManager::spec_write(uint64_t addr, uint64_t len) const {
  void* w = spec_worker();
  if (len) spec_write_in(*spec_, w, addr, len);
}

void MemoryManager::spec_read_each(const uint64_t* addrs, size_t count, uint32_t size) const {
  void* w = spec_worker();
  Spec& s = *spec_;
  if (size)
    for (size_t i = 0; i < count; ++i) spec_read_in(s, w, addrs[i], size);
}

void MemoryManager::spec_write_each(const uint64_t* addrs, size_t count, uint32_t size) const {
  void* w = spec_worker();
  Spec& s = *spec_;
  if (size)
    for (size_t i = 0; i < count; ++i) spec_write_in(s, w, addrs[i], size);
}

void MemoryManager::spec_write_strided(uint64_t a0, int64_t stride, size_t count, uint32_t size) const {
  void* w = spec_worker();
  Spec& s = *spec_;
  if (size)
    for (size_t i = 0; i < count; ++i) spec_write_in(s, w, a0 + static_cast<uint64_t>(stride) * i, size);
}

}  // namespace vgpu
