// A code object's instructions, each decoded once, the first time any wave of
// any dispatch reaches it, and shared by every thread that runs the object.
//
// A cache of one slot per instruction word for each thread of each dispatch
// cost a pointer per four bytes of .text per thread: for a hipBLASLt library
// of 300 MB of code, 600 MB a thread, and many gigabytes a launch -- enough to
// take a shared machine down (2026-09-25). This one holds a page of slots only
// where code runs, one set for the whole process, filled with a compare-and-
// swap so threads decoding the same instruction at once agree on one copy.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "vgpu/amd_gcn.hpp"

namespace vgpu::amd {

class DecodeCache {
 public:
  explicit DecodeCache(uint64_t text_bytes) : pages_(text_bytes / 4 / kPage + 1) {}
  ~DecodeCache() {
    for (auto& p : pages_)
      if (Page* page = p.load(std::memory_order_relaxed)) {
        for (auto& slot : page->slots) delete slot.load(std::memory_order_relaxed);
        delete page;
      }
  }
  DecodeCache(const DecodeCache&) = delete;
  DecodeCache& operator=(const DecodeCache&) = delete;

  // The instruction at word `word` of .text, decoded by `decode` the first
  // time. Null when the word is past the end.
  template <typename Decode>
  const gcn::Inst* get(uint64_t word, Decode&& decode) {
    const uint64_t pi = word / kPage;
    if (pi >= pages_.size()) return nullptr;
    Page* page = pages_[pi].load(std::memory_order_acquire);
    if (!page) {
      auto fresh = std::make_unique<Page>();
      if (pages_[pi].compare_exchange_strong(page, fresh.get(), std::memory_order_acq_rel)) page = fresh.release();
    }
    std::atomic<const gcn::Inst*>& slot = page->slots[word % kPage];
    const gcn::Inst* in = slot.load(std::memory_order_acquire);
    if (!in) {
      auto fresh = std::make_unique<const gcn::Inst>(decode());
      if (slot.compare_exchange_strong(in, fresh.get(), std::memory_order_acq_rel)) in = fresh.release();
    }
    return in;
  }

 private:
  static constexpr uint64_t kPage = 4096;   // instruction words: 16 KB of code
  struct Page {
    std::atomic<const gcn::Inst*> slots[kPage] = {};
  };
  std::vector<std::atomic<Page*>> pages_;
};

}  // namespace vgpu::amd
