#include "vgpu/faults.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>

#include "vgpu/error.hpp"

namespace vgpu::faults {
namespace {

struct Plan {
  bool all = false;
  std::set<uint64_t> at;   // 1-based occurrences
  std::atomic<uint64_t> seen{0};
  bool active() const { return all || !at.empty(); }
};

Plan g_plans[static_cast<size_t>(Op::Count)];
std::atomic<bool> g_any{false};
std::mutex g_mu;

const char* env_name(Op op) {
  switch (op) {
    case Op::Alloc: return "VGPU_FAIL_ALLOC";
    case Op::Launch: return "VGPU_FAIL_LAUNCH";
    case Op::Memcpy: return "VGPU_FAIL_MEMCPY";
    default: return "";
  }
}

// "all", or a comma-separated list of 1-based occurrence numbers. A value that
// parses to nothing is an error rather than a silent no-op: someone who typed
// VGPU_FAIL_ALLOC=first and saw a clean run would conclude the error path is
// handled, which is the opposite of what happened.
void parse_into(Op op, const char* value) {
  Plan& p = g_plans[static_cast<size_t>(op)];
  p.all = false;
  p.at.clear();
  if (!value || !value[0]) return;
  const std::string v(value);
  if (v == "all") {
    p.all = true;
    return;
  }
  size_t start = 0;
  bool any = false;
  while (start <= v.size()) {
    const size_t comma = v.find(',', start);
    const std::string item =
        v.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    if (!item.empty()) {
      char* end = nullptr;
      const unsigned long long n = std::strtoull(item.c_str(), &end, 10);
      if (end && *end == '\0' && n > 0) {
        p.at.insert(n);
        any = true;
      } else {
        throw Error::make(Err::InvalidValue, env_name(op), "='", v,
                          "' is not a list of occurrence numbers. Use 'all', or 1-based "
                          "positions like '3' or '2,5,9'");
      }
    }
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  if (!any && !p.all)
    throw Error::make(Err::InvalidValue, env_name(op), "='", v, "' selected no occurrences");
}

}  // namespace

void refresh() {
  std::lock_guard<std::mutex> lock(g_mu);
  bool any = false;
  for (size_t i = 0; i < static_cast<size_t>(Op::Count); ++i) {
    const Op op = static_cast<Op>(i);
    parse_into(op, std::getenv(env_name(op)));
    if (g_plans[i].active()) any = true;
  }
  g_any.store(any, std::memory_order_relaxed);
}

bool enabled() {
  // Read the environment on first use, not only when a launch refreshes the
  // modes. Allocation failures are the main thing anyone injects, and every
  // allocation happens *before* the first kernel launch -- so tying setup to
  // launches meant VGPU_FAIL_ALLOC did nothing at all for a program that
  // allocated and then failed to get that far.
  static std::once_flag once;
  std::call_once(once, [] { refresh(); });
  return g_any.load(std::memory_order_relaxed);
}

bool should_fail(Op op) {
  if (!enabled()) return false;
  Plan& p = g_plans[static_cast<size_t>(op)];
  if (!p.active()) return false;
  // Counted whether or not it fails, so the numbering a caller reasons about
  // does not shift depending on which occurrences were selected.
  const uint64_t n = p.seen.fetch_add(1, std::memory_order_relaxed) + 1;
  if (p.all) return true;
  return p.at.count(n) > 0;
}

const char* op_name(Op op) {
  switch (op) {
    case Op::Alloc: return "allocation";
    case Op::Launch: return "kernel launch";
    case Op::Memcpy: return "memory copy";
    default: return "operation";
  }
}

}  // namespace vgpu::faults
