// Blocks of the AMD pre-decoded tier (vgpu/amd_gcn_tier.hpp; the runner is Machine::run_block in gcn_exec.cpp): the switch, and the table
// VGPU_AMD_BLOCKS_STATS prints. In a unit of its own, so that the tier's (gcn_tier.cpp, built outside link-time optimization so that its
// layout depends on its own source only) holds nothing that is not a handler.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "vgpu/amd_gcn_tier.hpp"

namespace vgpu::amd::gcn::tier {

namespace {
// -1: not looked at yet. Off unless VGPU_AMD_BLOCKS is set to something but 0.
std::atomic<int> g_blocks{-1};
}  // namespace
bool blocks() {
  int s = g_blocks.load(std::memory_order_relaxed);
  if (s < 0) {
    const char* e = std::getenv("VGPU_AMD_BLOCKS");
    s = e && *e && *e != '0' ? 1 : 0;
    g_blocks.store(s, std::memory_order_relaxed);
  }
  return s != 0;
}
void set_blocks(bool on) { g_blocks.store(on ? 1 : 0, std::memory_order_relaxed); }
namespace {
struct BlockStats {
  struct Row {
    std::string text;
    uint32_t length = 0;
    uint64_t runs = 0, instructions = 0;
  };
  std::mutex m;
  std::map<std::pair<std::string, uint64_t>, Row> rows;   // by kernel and the offset of the block's first instruction in its code object
  uint64_t instructions = 0, in_blocks = 0;
  ~BlockStats() {
    if (!instructions) return;
    std::vector<const std::pair<const std::pair<std::string, uint64_t>, Row>*> v;
    uint64_t runs = 0, whole = 0;
    for (const auto& e : rows) v.push_back(&e), runs += e.second.runs, whole += e.second.runs * e.second.length;
    std::sort(v.begin(), v.end(), [](const auto* a, const auto* b) { return a->second.instructions > b->second.instructions; });
    std::fprintf(stderr, "AMD blocks: %llu of %llu instructions ran in blocks (%.1f%%), %llu block runs, %.1f instructions a run (%.1f a whole block), %zu blocks\n",
                 static_cast<unsigned long long>(in_blocks), static_cast<unsigned long long>(instructions),
                 100.0 * static_cast<double>(in_blocks) / static_cast<double>(instructions), static_cast<unsigned long long>(runs),
                 runs ? static_cast<double>(in_blocks) / static_cast<double>(runs) : 0.0, runs ? static_cast<double>(whole) / static_cast<double>(runs) : 0.0,
                 rows.size());
    const char* top = std::getenv("VGPU_AMD_BLOCKS_STATS");
    const size_t shown = top && std::atoi(top) > 1 ? static_cast<size_t>(std::atoi(top)) : 12;
    for (size_t i = 0; i < v.size() && i < shown; ++i) {
      const Row& r = v[i]->second;
      std::fprintf(stderr, "  %5.1f%%  %llu runs of %u instructions, %s +0x%llx%s\n", 100.0 * static_cast<double>(r.instructions) / static_cast<double>(instructions),
                   static_cast<unsigned long long>(r.runs), r.length, v[i]->first.first.c_str(), static_cast<unsigned long long>(v[i]->first.second),
                   r.text.c_str());
    }
  }
};
BlockStats& block_stats() {
  static BlockStats s;
  return s;
}
}  // namespace
bool blocks_stats() {
  static const bool on = [] {
    const char* e = std::getenv("VGPU_AMD_BLOCKS_STATS");
    return e && *e && *e != '0';
  }();
  return on;
}
void note_block(const char* kernel, uint64_t offset, uint32_t length, uint64_t runs, uint64_t instructions, std::string (*listing)(const void*),
                const void* block) {
  BlockStats& s = block_stats();
  std::lock_guard<std::mutex> lock(s.m);
  auto [it, fresh] = s.rows.try_emplace({kernel, offset});
  if (fresh) it->second.text = listing(block), it->second.length = length;
  it->second.runs += runs, it->second.instructions += instructions;
}
void note_block_totals(uint64_t instructions, uint64_t in_blocks) {
  BlockStats& s = block_stats();
  std::lock_guard<std::mutex> lock(s.m);
  s.instructions += instructions, s.in_blocks += in_blocks;
}

}  // namespace vgpu::amd::gcn::tier
