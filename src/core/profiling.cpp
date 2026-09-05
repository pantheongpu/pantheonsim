#include "vgpu/profiling.hpp"

#include <atomic>
#include <chrono>
#include <mutex>

namespace vgpu::profiling {
namespace {

std::atomic<bool> g_on{false};
std::mutex g_mu;
std::vector<Event> g_events;
std::atomic<uint32_t> g_correlation{1};

// A profiler that never drains must not grow the process without bound. The
// oldest events go first, which is the right end to lose: a timeline is read
// from the recent end, and dropping silently would be worse than dropping
// visibly, so the count is kept.
constexpr size_t kMaxBuffered = 1u << 20;

}  // namespace

bool enabled() { return g_on.load(std::memory_order_relaxed); }
void set_enabled(bool on) { g_on.store(on, std::memory_order_relaxed); }

void record(Event&& e) {
  if (!enabled()) return;
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_events.size() >= kMaxBuffered) g_events.erase(g_events.begin());
  g_events.push_back(std::move(e));
}

std::vector<Event> drain() {
  std::lock_guard<std::mutex> lock(g_mu);
  std::vector<Event> out;
  out.swap(g_events);
  return out;
}

uint64_t now_ns() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count());
}

uint32_t next_correlation() { return g_correlation.fetch_add(1, std::memory_order_relaxed); }

}  // namespace vgpu::profiling
