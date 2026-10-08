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
//
// Sync: a stream, context or event wait that returned; Stream and Context: a
// resource coming into being, which a profiler lists once.
enum class EventKind : uint8_t { Kernel, Memcpy, Memset, Api, Sync, Stream, Context, Device };

// What a copy's two ends are, as a profiler classifies them.
enum class MemKind : uint8_t { Unknown, Pageable, Pinned, Device, Array, Managed };

// Which wait a Sync event is.
enum class SyncKind : uint8_t { Event = 1, StreamWaitEvent = 2, Stream = 3, Context = 4 };

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
  MemKind src_kind = MemKind::Unknown, dst_kind = MemKind::Unknown;   // Memcpy; Memset: dst_kind
  bool async = false;            // Memcpy and Memset: the stream-ordered spelling
  uint32_t value = 0;            // Memset: the byte value
  uint32_t static_shared_bytes = 0;
  uint32_t local_bytes_per_thread = 0;
  uint8_t sync_kind = 0;         // Sync: a SyncKind
  uint64_t handle = 0;           // Sync: the event waited on; Stream: the stream made
  uint32_t flags = 0;            // Stream: cudaStream* flags
  int32_t priority = 0;          // Stream
  bool domain_driver = false;    // Api: a driver-API call rather than a runtime one
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

// ---- the callback side ----
//
// A tool that subscribes (CUPTI's Callback API) is told of each API call as it
// is entered and as it returns. The runtime reports the call without knowing
// who listens or what shape they want it in: a name, the correlation, and
// pointers to the arguments the caller passed, in declaration order. The
// front end turns those into the parameter structures its consumer reads.
enum class Domain : uint8_t { Runtime, Driver };

struct ApiInfo {
  Domain domain = Domain::Runtime;
  bool enter = true;
  const char* name = "";
  uint32_t correlation = 0;
  const void* const* args = nullptr;   // pointers to the caller's arguments; null if not captured
  int nargs = 0;
  const char* symbol = nullptr;        // a kernel launch: the kernel's name
  int32_t result = 0;                  // on exit
};

enum class Resource : uint8_t { CuInitFinished, ContextCreated, ContextDestroyStarting, StreamCreated, StreamDestroyStarting };

struct Hooks {
  void (*api)(const ApiInfo&) = nullptr;
  void (*resource)(Resource what, uint64_t handle, uint32_t device) = nullptr;
  void (*sync)(SyncKind what, uint64_t stream) = nullptr;
  // Asked on a kernel launch, when a subscriber wants the kernel's name.
  bool wants_symbols = false;
};
void set_hooks(const Hooks& h);          // all-null removes
bool hooked();
void notify_resource(Resource what, uint64_t handle, uint32_t device);
void notify_sync(SyncKind what, uint64_t stream);

// The arguments of the call about to be made: pointers to its parameters, in
// declaration order. Taken by the next ApiCall on this thread. Costs nothing
// when nobody listens.
void note_args(const void* const* args, int n);
void note_symbol(const char* name);

// A stretch of the shim's own code that makes public calls on the program's
// behalf (a front end asking the runtime what a device is): they are not the
// program's calls and are not reported as its.
class Silence {
 public:
  Silence();
  ~Silence();
  Silence(const Silence&) = delete;
  Silence& operator=(const Silence&) = delete;
};
// Whether this thread is inside a Silence: the shim is answering a profiler's
// question, not serving the program, and must not announce resources or bind
// contexts on the program's behalf as a side effect.
bool silenced();

// Driver initialisation finished: told to a subscriber once per process, by
// whichever library (driver or runtime) gets there first.
void notify_init_finished();

// A wait that returned, for a profiler's synchronization record and the
// callback API's synchronize domain. One public wait that goes through another
// (a stream wait through the context-wide one) is one wait.
class SyncScope {
 public:
  SyncScope(SyncKind kind, uint64_t stream, uint64_t event, uint32_t device);
  ~SyncScope();
  SyncScope(const SyncScope&) = delete;
  SyncScope& operator=(const SyncScope&) = delete;

 private:
  SyncKind kind_;
  uint64_t stream_, event_;
  uint32_t device_;
  uint64_t t0_ = 0;
  bool outermost_;
};

// One API call, recorded as it returns when anyone is profiling. A call made
// through another public call (cudaMemcpyAsync going through cudaMemcpy) is
// not a call of its own: a real runtime does not route its API through itself,
// so a profiler there lists the one the program made, and so does this. The
// work it issues carries that call's correlation.
class ApiCall {
 public:
  explicit ApiCall(const char* name, Domain domain = Domain::Runtime);
  ~ApiCall();
  ApiCall(const ApiCall&) = delete;
  ApiCall& operator=(const ApiCall&) = delete;
  void set_result(int32_t r) { result_ = r; }

 private:
  const char* name_ = nullptr;   // null when nobody was profiling at the call
  Domain domain_ = Domain::Runtime;
  bool outermost_ = false;       // a call made through another public call is not a call of its own
  bool hooked_ = false;
  const void* saved_[16] = {};
  const void* const* args_ = nullptr;
  int nargs_ = 0;
  const char* symbol_ = nullptr;
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
