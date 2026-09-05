// The event stream a profiler reads.
//
// Kept provider-neutral on purpose: the runtime records what happened, and a
// front end (CUPTI today) turns those into whatever shape its consumer wants.
// The alternative -- having the runtime build CUPTI structures directly --
// would put a toolkit header dependency in the core and tie the engine to one
// profiling interface.
//
// Timestamps are wall-clock nanoseconds spent simulating, not a prediction of
// how long a device would take. There is no timing model here. A profiler
// showing these will draw a truthful *ordering* and truthful durations of the
// simulation, and nothing about the performance of real hardware.
#ifndef VGPU_PROFILING_HPP
#define VGPU_PROFILING_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace vgpu::profiling {

enum class EventKind : uint8_t { Kernel, Memcpy, Memset };

struct Event {
  EventKind kind = EventKind::Kernel;
  uint64_t start_ns = 0;
  uint64_t end_ns = 0;
  uint32_t device = 0;
  uint32_t correlation = 0;
  uint64_t stream = 0;
  std::string name;              // kernel name; empty for copies
  uint64_t bytes = 0;            // copies and fills
  uint32_t copy_kind = 0;        // cudaMemcpyKind, as the caller gave it
  uint32_t grid[3] = {0, 0, 0};
  uint32_t block[3] = {0, 0, 0};
  uint32_t shared_bytes = 0;
  uint32_t registers_per_thread = 0;
};

// Off until a front end asks for it, so a program nobody is profiling pays
// nothing beyond one relaxed load per launch.
bool enabled();
void set_enabled(bool on);

void record(Event&& e);
std::vector<Event> drain();

// Monotonic, and the same clock the events carry.
uint64_t now_ns();

// CUPTI correlates an API call with the work it produced by this id.
uint32_t next_correlation();

}  // namespace vgpu::profiling

#endif  // VGPU_PROFILING_HPP
