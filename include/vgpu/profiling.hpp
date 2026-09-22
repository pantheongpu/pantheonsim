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

// Api: a runtime API call itself, on the host thread that made it; the work it
// issued carries the same correlation.
enum class EventKind : uint8_t { Kernel, Memcpy, Memset, Api };

struct Event {
  EventKind kind = EventKind::Kernel;
  uint64_t start_ns = 0;
  uint64_t end_ns = 0;
  uint32_t device = 0;
  uint32_t correlation = 0;
  uint64_t stream = 0;
  std::string name;              // kernel name, or the API function; empty for copies
  uint64_t bytes = 0;            // copies and fills
  uint32_t copy_kind = 0;        // cudaMemcpyKind, as the caller gave it
  uint32_t grid[3] = {0, 0, 0};
  uint32_t block[3] = {0, 0, 0};
  uint32_t shared_bytes = 0;
  uint32_t registers_per_thread = 0;
  uint32_t process_id = 0;       // Api: the caller
  uint32_t thread_id = 0;
  int32_t result = 0;            // Api: what the call returned
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
// The correlation the work being issued now carries: that of the API call this
// thread is inside, or a fresh one outside any.
uint32_t work_correlation();

// One API call, recorded as it returns when anyone is profiling. Calls nest (a
// call made through another public call); each keeps its own correlation.
class ApiCall {
 public:
  explicit ApiCall(const char* name);
  ~ApiCall();
  ApiCall(const ApiCall&) = delete;
  ApiCall& operator=(const ApiCall&) = delete;
  void set_result(int32_t r) { result_ = r; }

 private:
  const char* name_ = nullptr;   // null when nobody was profiling at the call
  uint32_t correlation_ = 0, outer_ = 0;
  uint64_t start_ = 0;
  int32_t result_ = 0;
};

}  // namespace vgpu::profiling

namespace vgpu {

// Opens the library named by CUDA_INJECTION64_PATH and calls its documented
// InitializeInjection entry point, once.
//
// A profiler does not ask the driver to profile; it asks the loader. nvprof,
// Nsight and anything else built on NVIDIA's injection library set this
// variable and rely on CUDA initialization to open the library and let the
// tool install its hooks. Without it the tool loads, the program runs
// correctly, and the tool reports "no profile data collected" -- which reads
// as a broken profiler rather than a driver that never invited it in.
void load_injection_library();

}  // namespace vgpu

#endif  // VGPU_PROFILING_HPP
