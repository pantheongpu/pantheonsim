#include "vgpu/exec/scheduler.hpp"

#include "vgpu/error.hpp"

namespace vgpu::exec {
namespace {

// Fixed round-robin: cycles through runnable warps in index order. With the
// rest of the engine being sequential, this yields bit-for-bit reproducible
// executions.
class RoundRobin final : public Scheduler {
 public:
  size_t pick(const std::vector<size_t>& runnable) override {
    // Choose the smallest runnable index strictly greater than the last pick,
    // wrapping around.
    for (size_t idx : runnable)
      if (idx > last_) return last_ = idx;
    return last_ = runnable.front();
  }

 private:
  size_t last_ = static_cast<size_t>(-1);
};

}  // namespace

std::unique_ptr<Scheduler> make_scheduler(SchedulerKind kind) {
  switch (kind) {
    case SchedulerKind::Deterministic: return std::make_unique<RoundRobin>();
  }
  throw Error::make(Err::Internal, "unknown scheduler kind");
}

}  // namespace vgpu::exec
