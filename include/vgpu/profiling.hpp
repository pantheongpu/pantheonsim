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
//
// Memory: an allocation or a release of device, pinned or managed memory
// (`op`), with the stream-ordered pool it came from when it did. MemoryPool: a
// pool created, destroyed or trimmed. GraphTrace: one launch of a graph, as a
// single span (a launched graph's nodes are not reported one by one while
// this is wanted, as on NVIDIA's). Memcpy2: a copy straight between two
// devices' memories. CudaEvent: an event recorded. Module and Function: code
// brought onto a device (see Event::handle and ::name).
enum class EventKind : uint8_t { Kernel, Memcpy, Memset, Api, Sync, Stream, Context, Device, Marker, MarkerData, Name, ExternalCorrelation,
                                 Memory, MemoryPool, GraphTrace, Memcpy2, CudaEvent, Module, Function,
                                 // Made by a front end from the events above, never by the runtime:
                                 // one record per allocation for CUPTI's older memory kind, and the
                                 // profiler's own overhead.
                                 MemoryV1, Overhead };

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
  // How many times `stream` (or, for a stream record, `handle`) had been destroyed when the event was made:
  // a stream handle freed and made again is a new stream to a profiler (stream_generation).
  uint32_t stream_gen = 0;
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
  // NVTX: a Marker is an instant, the start or the end of a range (flags); its
  // MarkerData carries the attributes the program gave it; a Name names a
  // thread. `handle` is the marker's id; `name` its message.
  std::string domain;
  uint32_t color = 0;
  uint32_t category = 0;
  int32_t payload_kind = 0;
  uint64_t payload = 0;
  // Memory: op 1 = allocation, 2 = release; MemoryPool: 1 = created, 2 =
  // destroyed, 3 = trimmed. `address` is the allocation (Memory) or the pool
  // (MemoryPool); the pool_* fields describe the stream-ordered pool at that
  // moment (pool_handle is 0 for an allocation that is not from one).
  // MemoryPool: pool_released says a trim gave memory back.
  uint8_t op = 0;
  uint64_t address = 0;
  uint64_t pool_handle = 0, pool_size = 0, pool_threshold = 0, pool_utilized = 0;
  bool pool_released = false;
  // GraphTrace: the launched executable graph's id. Memcpy2: the two ends.
  // CudaEvent: `handle` is the event (cudaEvent_t), `sync_id` which of its
  // recordings this is. Module/Function: module_id; Function also name.
  uint32_t graph_id = 0;           // GraphTrace, and the work of a graph launch: the executable graph
  uint64_t graph_node_id = 0;      // that work: (graph id << 32) | the node's index in the executable graph
  uint32_t src_device = 0, dst_device = 0;
  uint64_t sync_id = 0;
  uint32_t module_id = 0;
  uint32_t function_index = 0;
  // Set by the front end on events it makes itself or completes (the ids it
  // gave a context and the streams the driver makes inside one); the runtime
  // leaves them zero.
  uint32_t context_id = 0;
  uint32_t stream_id = 0;
  // Memcpy: on one of the driver's own streams in the device's context rather
  // than a stream of the program's -- the nth of the eight made with the
  // context (-1: the stream above).
  int32_t internal_stream = -1;
};

// Off until a front end asks for it, so a program nobody is profiling pays
// nothing beyond one relaxed load per launch.
bool enabled();
void set_enabled(bool on);

void record(Event&& e);
std::vector<Event> drain();

// Told of each event as it is recorded, on the recording thread, with no lock
// of the engine's held: a front end that hands out buffers as the first record
// for one is made (CUPTI does) learns of it here. One hook; null to remove.
void set_record_hook(void (*fn)(const Event&));

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
  // sizeof each argument as the shim declares it, where the caller said (null
  // otherwise). A front end that copies an argument into a field of a size the
  // toolkit names checks the two agree before it reads the bytes.
  const uint16_t* arg_sizes = nullptr;
  const char* symbol = nullptr;        // a kernel launch: the kernel's name
  int32_t result = 0;                  // on exit
  // On exit, the call's return value where it is not a status code (a string, a
  // structure): a pointer to it, valid until the call returns.
  const void* return_value = nullptr;
};

enum class Resource : uint8_t {
  CuInitFinished, ContextCreated, ContextDestroyStarting, StreamCreated, StreamDestroyStarting,
  ModuleLoaded, ModuleUnloadStarting, ModuleProfiled,
  GraphCreated, GraphDestroyStarting, GraphCloned,
  GraphNodeCreateStarting, GraphNodeCreated, GraphNodeDestroyStarting,
  GraphNodeDependencyCreated, GraphNodeDependencyDestroyStarting,
  GraphExecCreateStarting, GraphExecCreated, GraphExecDestroyStarting,
  GraphNodeCloned, StreamAttributeChanged, GraphNodeUpdated, GraphNodeSetParams
};

// Everything a resource callback can carry. `handle` is the stream for stream
// resources. Graph events name the graph, the node and the other object the
// event relates it to (a clone's original, a dependency's other end, the
// executable graph) by their handles, 0 where the event has none; `node_type`
// is a CUgraphNodeType and set only where CUPTI sets it.
struct ResourceInfo {
  Resource what = Resource::CuInitFinished;
  uint64_t handle = 0;
  uint32_t device = 0;
  bool context = true;                  // false: the callback names no context, as with most graph events
  // modules
  uint32_t module_id = 0;
  const void* cubin = nullptr;          // null: the module's code is not a cubin this engine holds
  size_t cubin_size = 0;
  // graphs
  uint64_t graph = 0, original_graph = 0, node = 0, original_node = 0, dependency = 0, graph_exec = 0;
  int node_type = 0;
  // stream attributes
  int attribute = 0;
  const void* attribute_value = nullptr;
};

struct Hooks {
  void (*api)(const ApiInfo&) = nullptr;
  void (*resource)(const ResourceInfo&) = nullptr;
  void (*sync)(SyncKind what, uint64_t stream) = nullptr;
  // Asked on a kernel launch, when a subscriber wants the kernel's name.
  bool wants_symbols = false;
};
void set_hooks(const Hooks& h);          // all-null removes
bool hooked();
void notify_resource(Resource what, uint64_t handle, uint32_t device);
void notify_resource(const ResourceInfo& info);
void notify_sync(SyncKind what, uint64_t stream);
// A stream handle that is destroyed and made again names a different stream: a real driver numbers the new one
// afresh, while the shims reuse the pointer. Every StreamDestroyStarting counts one against the handle, and an
// event is stamped with the count when it is recorded, so the profiler can tell the two streams apart.
uint32_t stream_generation(uint64_t handle);

// The clock the profiler's own timestamps for host-side events (API calls,
// waits, markers) come from. A tool may supply one (CUPTI's timestamp
// callback); work on the device is stamped by now_ns() and put on the tool's
// timeline when the records are produced.
void set_host_clock(uint64_t (*fn)());
uint64_t host_ns();

// The arguments of the call about to be made: pointers to its parameters, in
// declaration order. Taken by the next ApiCall on this thread. Costs nothing
// when nobody listens.
void note_args(const void* const* args, int n, const uint16_t* sizes = nullptr);
void note_symbol(const char* name);

// External correlation ids: a framework tags the work it is about to issue
// ("this is op 100") with a push, and every API call made while the tag is on
// the thread's stack for its kind is reported with it. One stack per kind per
// thread, as NVIDIA's.
constexpr int kExternalKinds = 8;
bool push_external(int kind, uint64_t id);    // false: no such kind
bool pop_external(int kind, uint64_t* last);  // false: the stack is empty

// Graph ids, as a profiler numbers them: every graph made (a graph, a capture,
// a clone, and each executable graph with the graph it is made of) takes the
// next number, and a node is the graph's number in the high half and its place
// in the graph in the low half. Kept for as long as the object lives, whether
// or not anyone is profiling, because a tool may attach after it was made.
uint32_t next_graph_id();
// The numbers a profiler gives modules and the functions in them, across the
// runtime and the driver. A device's modules are numbered from 22 (the driver's
// own take the ones before), starting again when its context is made again
// after a reset (measured on an RTX 3060: the same program's module has the
// same number in the new context); functions are numbered from 1.
uint32_t next_module_id(uint32_t device);
void reset_module_ids(uint32_t device);
uint32_t next_function_id();
void register_graph(uint64_t handle, uint32_t id);
void register_graph_node(uint64_t handle, uint64_t id);
void forget_graph_object(uint64_t handle);
bool graph_id_of(uint64_t handle, uint32_t* id);
bool graph_node_id_of(uint64_t handle, uint64_t* id);

// The work a graph launch runs, node by node: while one of these is alive on a
// thread, a kernel, copy or fill recorded by it is tagged with the executable
// graph (`graph_id`) and the node (`graph_node_id`), which is how a profiler
// connects it to the graph it came from.
class GraphWork {
 public:
  GraphWork(uint32_t graph_id, uint32_t node_index);
  ~GraphWork();
  GraphWork(const GraphWork&) = delete;
  GraphWork& operator=(const GraphWork&) = delete;

 private:
  uint32_t saved_graph_, saved_node_;
};

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

// The first context of the process being made, once: what a tool had switched on
// by then is what it gets for the API calls it asks to be recorded (see
// nvidia/docs/cupti.md). `note_context_made` is called by whichever of the
// runtime and the driver makes a context first; the hook, if there is one, is
// called then.
void note_context_made();
bool context_made();
void set_context_hook(void (*fn)());

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
  // For a call that returns something other than a status code.
  void set_return_value(const void* p) { return_value_ = p; }

 private:
  const char* name_ = nullptr;   // null when nobody was profiling at the call
  Domain domain_ = Domain::Runtime;
  bool outermost_ = false;       // a call made through another public call is not a call of its own
  bool hooked_ = false;
  const void* saved_[16] = {};
  uint16_t saved_sizes_[16] = {};
  const void* const* args_ = nullptr;
  const uint16_t* sizes_ = nullptr;
  int nargs_ = 0;
  const char* symbol_ = nullptr;
  uint32_t correlation_ = 0, outer_ = 0;
  uint64_t start_ = 0;
  int32_t result_ = 0;
  const void* return_value_ = nullptr;
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
