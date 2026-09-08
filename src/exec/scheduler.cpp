#include "vgpu/exec/scheduler.hpp"

#include <cstdlib>
#include <cstring>
#include <string>

#include "vgpu/error.hpp"

namespace vgpu::exec {
namespace {

// A small, fast, fully specified PRNG. std::mt19937 would do, but its stream is
// only reproducible across implementations by convention; this one is written
// down, so a seed replays the same execution on any machine and any compiler.
// That is the property the whole mode depends on -- a race you cannot re-run is
// a race you cannot fix.
class SplitMix64 {
 public:
  explicit SplitMix64(uint64_t seed) : state_(seed) {}
  uint64_t next() {
    uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  // Unbiased below `n` -- rejection rather than a modulo, because the bias a
  // modulo introduces lands on exactly the low indices a warp list is made of.
  uint64_t below(uint64_t n) {
    if (n <= 1) return 0;
    const uint64_t limit = UINT64_MAX - (UINT64_MAX % n) - 1;
    uint64_t v;
    do { v = next(); } while (v > limit);
    return v % n;
  }

 private:
  uint64_t state_;
};

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

// Seeded uniform choice, with a short randomised slice so the switch points
// vary too. Picking a warp uniformly but always running it to its barrier
// would explore far less than it looks: the interleaving, not the order of
// first issue, is what decides a racy program's answer.
class RandomOrder final : public Scheduler {
 public:
  explicit RandomOrder(uint64_t seed) : rng_(seed ? seed : 1) {}
  size_t pick(const std::vector<size_t>& runnable) override {
    return runnable[static_cast<size_t>(rng_.below(runnable.size()))];
  }
  uint64_t slice() override { return 1 + rng_.below(kMaxSlice); }

 private:
  static constexpr uint64_t kMaxSlice = 16;
  SplitMix64 rng_;
};

// Maximum interleaving: switch warp at every opportunity, and prefer whichever
// has waited longest.
//
// This is adversarial in the sense that matters -- it is the legal order most
// likely to make a racy program answer differently -- but it is not guided by
// what the warps are about to touch. The scheduler is not told about memory,
// so it cannot aim at a specific pair of conflicting accesses; it maximises
// switching and lets that do the work. Aiming would mean feeding the race
// detector's shadow state back into scheduling, which is a larger change and
// is noted in TODO.md rather than half-done here.
class Adversarial final : public Scheduler {
 public:
  explicit Adversarial(uint64_t seed) : rng_(seed ? seed : 1) {}

  size_t pick(const std::vector<size_t>& runnable) override {
    ++tick_;
    size_t best = runnable.front();
    uint64_t oldest = UINT64_MAX;
    for (size_t idx : runnable) {
      if (idx == last_ && runnable.size() > 1) continue;  // never twice in a row
      const uint64_t seen = idx < last_run_.size() ? last_run_[idx] : 0;
      // The seed breaks ties, so two warps that have waited equally long do not
      // always resolve the same way. Without it every run of a given kernel
      // explores one order and calling it a search would be a fiction.
      if (seen < oldest || (seen == oldest && rng_.below(2))) {
        oldest = seen;
        best = idx;
      }
    }
    if (best >= last_run_.size()) last_run_.resize(best + 1, 0);
    last_run_[best] = tick_;
    last_ = best;
    return best;
  }

  // One instruction per turn. Any larger and the interleaving this exists to
  // produce is coarser than the races it is looking for.
  uint64_t slice() override { return 1; }

 private:
  SplitMix64 rng_;
  std::vector<uint64_t> last_run_;
  uint64_t tick_ = 0;
  size_t last_ = static_cast<size_t>(-1);
};

}  // namespace

std::unique_ptr<Scheduler> make_scheduler(SchedulerKind kind, uint64_t seed) {
  switch (kind) {
    case SchedulerKind::Deterministic: return std::make_unique<RoundRobin>();
    case SchedulerKind::Random: return std::make_unique<RandomOrder>(seed);
    case SchedulerKind::Adversarial: return std::make_unique<Adversarial>(seed);
  }
  throw Error::make(Err::Internal, "unknown scheduler kind");
}

const char* scheduler_name(SchedulerKind kind) {
  switch (kind) {
    case SchedulerKind::Deterministic: return "deterministic";
    case SchedulerKind::Random: return "random";
    case SchedulerKind::Adversarial: return "adversarial";
  }
  return "?";
}

bool scheduler_from_env(SchedulerKind* kind, uint64_t* seed) {
  const char* mode = std::getenv("VGPU_SCHEDULER");
  if (!mode || !mode[0]) return false;
  const std::string m(mode);
  if (m == "deterministic") *kind = SchedulerKind::Deterministic;
  else if (m == "random") *kind = SchedulerKind::Random;
  else if (m == "adversarial") *kind = SchedulerKind::Adversarial;
  else
    throw Error::make(Err::InvalidValue, "VGPU_SCHEDULER='", m,
                      "' is not a scheduler; expected deterministic, random or adversarial. "
                      "Running the default instead would mean a result nobody could attribute "
                      "to an execution order");
  if (const char* s = std::getenv("VGPU_SCHEDULER_SEED"); s && s[0])
    *seed = std::strtoull(s, nullptr, 10);
  return true;
}

}  // namespace vgpu::exec
