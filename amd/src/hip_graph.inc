// HIP graphs and stream capture, as ROCm's HIP answers them. Included by
// hip_api.cpp, whose runtime internals (the streams' queues, copies and fills
// of every shape, launches, events, pools) it builds on.
//
// A graph is nodes in the order they were added and the edges between them.
// An executable graph is a copy of one, made at instantiation, which later
// edits of the graph do not reach. A launch is one piece of its stream's
// work: the nodes run one after another in an order their edges allow, each
// the way the call it stands for runs -- a kernel as a launch, a copy or a
// fill as the region copy or fill, a host node on the stream's thread. A
// capture records what a stream is asked to do into a graph instead of
// doing it. AMD's own tests (hip-tests) are the reference throughout,
// including for every error.
//
// Every handle -- graph, node, executable graph, user object -- is a number
// looked up in a table, never an address: tests hand in handles that were
// destroyed, or never were, and those are the wrong value, not a crash.
// Nothing here is reused, so a stale handle never names something new.
// Everything is under s.mutex, which a launch's work takes only briefly.

namespace {

using vgpu::amd::abi::Dim3;
using vgpu::amd::abi::GraphNodeType;

static_assert(sizeof(vgpu::amd::abi::KernelNodeParams) == 64, "hipKernelNodeParams");
static_assert(sizeof(vgpu::amd::abi::MemsetParams) == 48, "hipMemsetParams");
static_assert(sizeof(vgpu::amd::abi::MemAllocNodeParams) == 120, "hipMemAllocNodeParams");
static_assert(sizeof(vgpu::amd::abi::MemcpyNodeParams) == 176, "hipMemcpyNodeParams");
static_assert(sizeof(vgpu::amd::abi::GraphNodeParams) == 256, "hipGraphNodeParams");
static_assert(sizeof(vgpu::amd::abi::GraphInstantiateParams) == 32, "hipGraphInstantiateParams");
static_assert(sizeof(vgpu::amd::abi::BatchMemOpNodeParams) == 32, "hipBatchMemOpNodeParams");

// The largest access policy window a kernel node takes: this device reports
// none (hipDeviceAttributeAccessPolicyMaxWindowSize is 0).
constexpr size_t kAccessPolicyMaxWindow = 0;

struct GraphObj;

// A kernel node: the function as it was given (a host-side function or a
// hipFunction_t), its launch shape, and its arguments, copied when the node
// was made -- a program passes pointers to values that are gone by launch.
struct KernelData {
  const void* func = nullptr;
  bool module_fn = false;   // func is a hipFunction_t
  int device = 0;           // the device current when it was made, where it runs
  Dim3 grid{1, 1, 1}, block{1, 1, 1};
  uint32_t shared = 0;
  void** extra = nullptr;   // as given, which GetParams hands back
  bool from_extra = false;
  std::vector<uint8_t> args;                          // packed, as the kernarg segment takes them
  std::vector<std::pair<uint32_t, uint32_t>> slots;   // each argument's offset and size (size 0: hidden)
  std::vector<void*> params;                          // what GetParams hands out: pointers into args
  std::map<int, vgpu::amd::abi::LaunchAttributeValue> attrs;
  bool cooperative = false;
};

// A copy node: the parameters as given, which GetParams hands back, and the
// two regions it copies between, worked out when it was made.
enum CopyForm { kCopy3D, kCopy1D, kCopyFromSymbol, kCopyToSymbol, kCopyDrv, kCopyCaptured };
struct CopyData {
  CopyForm form = kCopy3D;
  vgpu::amd::abi::Memcpy3DParms p3{};
  vgpu::amd::abi::Memcpy3D drv{};
  void* dst = nullptr;
  const void* src = nullptr;
  size_t count = 0, offset = 0;
  int kind = 0;
  const void* symbol = nullptr;
  bool empty = true;   // nothing to copy
  void* rdst = nullptr;
  const void* rsrc = nullptr;
  Region dr, sr;
  int rkind = 0;
  // A captured copy whose source was on the capturing thread's stack: the bytes
  // as they were then (from the start of the first row), for a launch made after
  // that frame has gone. `snap_thread` is the thread whose stack it was.
  std::shared_ptr<const std::vector<uint8_t>> snap;
  std::thread::id snap_thread;
};

// A fill node: hipMemsetParams as given, and a depth for a captured 3D fill,
// which the struct has no room for.
struct FillData {
  vgpu::amd::abi::MemsetParams p{};
  size_t depth = 1, slice = 0;
};

struct GNode {
  void* handle = nullptr;
  GraphObj* graph = nullptr;
  int type = vgpu::amd::abi::kNodeEmpty;
  std::vector<GNode*> in, out;   // in the order the edges were made
  KernelData kernel;
  CopyData copy;
  FillData fill;
  vgpu::amd::abi::HostNodeParams host{};
  std::unique_ptr<GraphObj> child;
  hipEvent_t event = nullptr;
  vgpu::amd::abi::MemAllocNodeParams alloc{};
  void* free_ptr = nullptr;
  vgpu::amd::abi::BatchMemOpNodeParams batch{};
  std::vector<uint8_t> batch_ops;   // 48 bytes each
  void* source = nullptr;           // the node this one was cloned or instantiated from
  bool enabled = true;              // in an executable graph
  ~GNode();
};

// User objects a graph holds, by id: how many references.
using UserRefs = std::map<uint64_t, uint64_t>;

struct GraphObj {
  void* handle = nullptr;
  std::vector<std::unique_ptr<GNode>> nodes;      // in the order they were added
  std::vector<std::pair<GNode*, GNode*>> edges;   // in the order they were made
  GNode* owner = nullptr;                          // the child-graph node it is embedded in
  UserRefs user_objects;
  int live_execs = 0;   // executable graphs made from it that are still there
};

GNode::~GNode() = default;

struct ExecObj {
  void* handle = nullptr;
  std::unique_ptr<GraphObj> copy;
  std::map<void*, GNode*> by_source;   // the graph's node -> its copy here
  void* graph = nullptr;               // the graph it was made from
  unsigned long long flags = 0;
  UserRefs user_objects;
  std::shared_ptr<Queue> last_queue;   // its last launch, which the next follows
  uint64_t last_seq = 0;
};

struct UserObj {
  void* ptr = nullptr;
  void (*destroy)(void*) = nullptr;
  uint64_t refs = 0;
};

// A graph allocation: address space set aside when its node was made, with
// memory mapped there while the allocation is live.
struct GraphAlloc {
  int device = 0;
  size_t bytes = 0;
  uint64_t reserved = 0;   // the address space's size, whole VMM granules
  uint64_t handle = 0;     // the memory mapped there, while committed
  bool committed = false;
  void* free_node = nullptr;   // the node that frees it, in any graph
};

struct GraphTables {
  uint64_t next = 0x6700000;
  std::map<void*, GraphObj*> graphs;                  // every graph there is, a program's own or embedded
  std::map<void*, std::unique_ptr<GraphObj>> owned;   // those a program made (and destroys)
  std::map<void*, GNode*> nodes;
  std::map<void*, std::unique_ptr<ExecObj>> execs;
  std::map<uint64_t, UserObj> user_objects;           // live ones only
  std::map<uint64_t, GraphAlloc> allocs;              // by address
};
GraphTables& gt() {
  static GraphTables& t = *new GraphTables;
  return t;
}
void* new_handle() { return reinterpret_cast<void*>(gt().next += 0x10); }

GraphObj* find_graph(void* h) {
  const auto it = gt().graphs.find(h);
  return it == gt().graphs.end() ? nullptr : it->second;
}
GNode* find_node(void* h) {
  const auto it = gt().nodes.find(h);
  return it == gt().nodes.end() ? nullptr : it->second;
}
ExecObj* find_exec(void* h) {
  const auto it = gt().execs.find(h);
  return it == gt().execs.end() ? nullptr : it->second.get();
}

// Handles for a graph's nodes and embedded graphs, and back out of the tables.
void register_graph(GraphObj& g) {
  g.handle = new_handle();
  gt().graphs[g.handle] = &g;
}
void register_node(GNode& n) {
  n.handle = new_handle();
  gt().nodes[n.handle] = &n;
  if (n.child) {
    register_graph(*n.child);
    for (auto& c : n.child->nodes) register_node(*c);
  }
}
void unregister_graph(GraphObj& g);
void unregister_node(GNode& n) {
  gt().nodes.erase(n.handle);
  if (n.child) unregister_graph(*n.child);
}
void unregister_graph(GraphObj& g) {
  gt().graphs.erase(g.handle);
  for (auto& n : g.nodes) unregister_node(*n);
}

// ---- User objects -----------------------------------------------------------------
//
// A destructor runs once, when the last reference goes, on some host thread
// with no lock of the runtime's held: it may call HIP. Callers collect them
// here and run them after unlocking.
using Destructors = std::vector<std::pair<void (*)(void*), void*>>;
void run_destructors(const Destructors& d) {
  for (const auto& [fn, p] : d) fn(p);
}
void release_user_object(uint64_t id, uint64_t count, Destructors* out) {
  const auto it = gt().user_objects.find(id);
  if (it == gt().user_objects.end() || count > it->second.refs) return;
  it->second.refs -= count;
  if (it->second.refs) return;
  out->emplace_back(it->second.destroy, it->second.ptr);
  gt().user_objects.erase(it);
}
void release_refs(UserRefs& refs, Destructors* out) {
  for (const auto& [id, n] : refs) release_user_object(id, n, out);
  refs.clear();
}
// Takes references of its own for each one `from` holds.
void take_refs(const UserRefs& from, UserRefs* to) {
  for (const auto& [id, n] : from) {
    const auto it = gt().user_objects.find(id);
    if (it == gt().user_objects.end()) continue;
    it->second.refs += n;
    (*to)[id] += n;
  }
}

// ---- Graph memory -------------------------------------------------------------------
//
// What allocation nodes take is counted per device: in use, and reserved --
// in use, and what was freed and is kept for the next allocation until a
// trim. Each has a high-water mark that only a program sets back, to zero.
struct GraphMemoryCounts {
  uint64_t used = 0, used_high = 0, cached = 0, reserved_high = 0;
  uint64_t reserved() const { return used + cached; }
};
std::map<int, GraphMemoryCounts>& graph_counts() {
  static auto& m = *new std::map<int, GraphMemoryCounts>;
  return m;
}
uint64_t granules(uint64_t bytes) {
  const uint64_t g = vgpu::MemoryManager::kVmmGranularity;
  return (std::max<uint64_t>(bytes, 1) + g - 1) / g * g;
}
// Maps memory at a graph allocation's address. The caller holds s.mutex.
hipError_t commit_graph_alloc(State& s, uint64_t va) {
  GraphAlloc& a = gt().allocs.at(va);
  if (a.committed) return hipSuccess;
  vgpu::MemoryManager& m = s.rt->device(a.device).memory();
  try {
    a.handle = m.create_handle(a.reserved);
    m.map(va, a.reserved, 0, a.handle);
    m.set_access(va, a.reserved, true, true);
  } catch (const std::exception& e) {
    return fail(hipErrorOutOfMemory, std::string("a graph allocation: ") + e.what());
  }
  a.committed = true;
  GraphMemoryCounts& c = graph_counts()[a.device];
  c.cached = c.cached > a.bytes ? c.cached - a.bytes : 0;
  c.used += a.bytes;
  c.used_high = std::max(c.used_high, c.used);
  c.reserved_high = std::max(c.reserved_high, c.reserved());
  return hipSuccess;
}
hipError_t free_graph_alloc(State& s, uint64_t va) {
  const auto it = gt().allocs.find(va);
  if (it == gt().allocs.end() || !it->second.committed) return hipErrorInvalidValue;
  GraphAlloc& a = it->second;
  vgpu::MemoryManager& m = s.rt->device(a.device).memory();
  try {
    m.unmap(va, a.reserved);
    m.release_handle(a.handle);
  } catch (const std::exception& e) {
    return fail(hipErrorInvalidValue, e.what());
  }
  a.committed = false;
  GraphMemoryCounts& c = graph_counts()[a.device];
  c.used -= std::min<uint64_t>(c.used, a.bytes);
  c.cached += a.bytes;
  return hipSuccess;
}

// ---- Graphs: nodes, edges, copies -----------------------------------------------------

// Adds `n` to `g`, after each of `deps`. The dependencies were checked.
GNode* add_node(GraphObj& g, std::unique_ptr<GNode> n, const std::vector<GNode*>& deps) {
  GNode* p = n.get();
  p->graph = &g;
  register_node(*p);
  for (GNode* d : deps) {
    d->out.push_back(p);
    p->in.push_back(d);
    g.edges.emplace_back(d, p);
  }
  g.nodes.push_back(std::move(n));
  return p;
}

// The graph and dependencies an Add call names, checked: the graph there,
// the dependency array there if it has entries, and each entry a node of the
// graph. Only the first `n` are read.
hipError_t add_target(void* node_out, void* graph, const void* deps, size_t n, GraphObj** g,
                      std::vector<GNode*>* out) {
  if (!node_out) return hipErrorInvalidValue;
  *g = find_graph(graph);
  if (!*g) return hipErrorInvalidValue;
  if (n && !deps) return hipErrorInvalidValue;
  for (size_t i = 0; i < n; ++i) {
    GNode* d = find_node(static_cast<void* const*>(deps)[i]);
    if (!d || d->graph != *g) return hipErrorInvalidValue;
    if (std::find(out->begin(), out->end(), d) == out->end()) out->push_back(d);
  }
  return hipSuccess;
}

// A node's own parameters, copied into another node (not its edges).
void copy_node_data(const GNode& from, GNode* to);
std::unique_ptr<GraphObj> copy_graph(const GraphObj& from, bool registered, std::map<const GNode*, GNode*>* map);

void copy_node_data(const GNode& from, GNode* to) {
  to->type = from.type;
  to->kernel = from.kernel;
  to->kernel.params.clear();
  to->copy = from.copy;
  to->fill = from.fill;
  to->host = from.host;
  to->event = from.event;
  to->alloc = from.alloc;
  to->free_ptr = from.free_ptr;
  to->batch = from.batch;
  to->batch_ops = from.batch_ops;
  to->enabled = from.enabled;
  if (from.child) {
    std::map<const GNode*, GNode*> m;
    to->child = copy_graph(*from.child, false, &m);
    to->child->owner = to;
  }
}

// A deep copy: every node, in the same order, with every edge; each node
// remembers the one it was copied from. `registered` gives it handles.
std::unique_ptr<GraphObj> copy_graph(const GraphObj& from, bool registered, std::map<const GNode*, GNode*>* map) {
  auto g = std::make_unique<GraphObj>();
  if (registered) register_graph(*g);
  for (const auto& n : from.nodes) {
    auto c = std::make_unique<GNode>();
    copy_node_data(*n, c.get());
    c->source = n->handle;
    c->graph = g.get();
    (*map)[n.get()] = c.get();
    if (registered) register_node(*c);
    g->nodes.push_back(std::move(c));
  }
  for (const auto& [a, b] : from.edges) {
    GNode* x = (*map)[a];
    GNode* y = (*map)[b];
    x->out.push_back(y);
    y->in.push_back(x);
    g->edges.emplace_back(x, y);
  }
  return g;
}

// The order a launch runs a graph's nodes in: each after all it depends on,
// and otherwise in the order they were added. False where the edges go round
// in a cycle.
bool topo_order(const GraphObj& g, std::vector<const GNode*>* out) {
  std::map<const GNode*, size_t> waiting;
  for (const auto& n : g.nodes) waiting[n.get()] = n->in.size();
  std::vector<bool> done(g.nodes.size(), false);
  std::map<const GNode*, size_t> index;
  for (size_t i = 0; i < g.nodes.size(); ++i) index[g.nodes[i].get()] = i;
  for (size_t emitted = 0; emitted < g.nodes.size(); ++emitted) {
    const GNode* next = nullptr;
    for (size_t i = 0; i < g.nodes.size(); ++i)
      if (!done[i] && waiting[g.nodes[i].get()] == 0) {
        next = g.nodes[i].get();
        done[i] = true;
        break;
      }
    if (!next) return false;
    out->push_back(next);
    for (const GNode* o : next->out) --waiting[o];
  }
  return true;
}
bool acyclic(const GraphObj& g) {
  std::vector<const GNode*> order;
  if (!topo_order(g, &order)) return false;
  for (const auto& n : g.nodes)
    if (n->child && !acyclic(*n->child)) return false;
  return true;
}
bool has_memory_nodes(const GraphObj& g) {
  for (const auto& n : g.nodes)
    if (n->type == vgpu::amd::abi::kNodeMemAlloc || n->type == vgpu::amd::abi::kNodeMemFree ||
        (n->child && has_memory_nodes(*n->child)))
      return true;
  return false;
}

// Destroys a graph a program made, or one a capture made and gave up.
void destroy_graph(GraphObj* g, Destructors* d) {
  release_refs(g->user_objects, d);
  unregister_graph(*g);
  for (auto& n : g->nodes)
    if (n->type == vgpu::amd::abi::kNodeMemFree)
      for (auto& [va, a] : gt().allocs)
        if (a.free_node == n->handle) a.free_node = nullptr;
  gt().owned.erase(g->handle);
}

// ---- Kernel nodes -------------------------------------------------------------------

// The kernel a node's function names on `device`: a hipFunction_t is its own
// module's; a host-side function is the program's, loaded there. The caller
// holds s.mutex.
hipError_t kernel_of(State& s, const void* func, int device, bool* module_fn, const Module** m, const Kernel** k) {
  if (!func) return hipErrorInvalidValue;
  for (const auto& f : s.functions)
    if (f.get() == func) {
      *module_fn = true;
      *m = f->module;
      *k = f->kernel;
      return hipSuccess;
    }
  const auto hf = s.host_functions.find(func);
  if (hf == s.host_functions.end()) return fail(hipErrorInvalidDeviceFunction, "no kernel was registered for that function");
  Module* mod = nullptr;
  if (const hipError_t e = module_on(s, *hf->second.binary, device, &mod); e != hipSuccess) return e;
  const Kernel* kernel = vgpu::amd::find_kernel(mod->object, hf->second.kernel);
  if (!kernel) return hipErrorInvalidDeviceFunction;
  *module_fn = false;
  *m = mod;
  *k = kernel;
  return hipSuccess;
}
void fill_slots(const Kernel& k, KernelData* d) {
  d->slots.clear();
  for (const vgpu::amd::KernelArg& a : k.args)
    d->slots.emplace_back(static_cast<uint32_t>(a.offset), a.hidden() ? 0u : static_cast<uint32_t>(a.size));
}
// A kernel node's data from hipKernelNodeParams, its arguments copied now.
// The caller holds s.mutex.
hipError_t make_kernel(State& s, const vgpu::amd::abi::KernelNodeParams* p, KernelData* out) {
  if (!p || !p->func) return hipErrorInvalidValue;
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return e;
  KernelData d;
  d.device = s.current;
  const Module* m = nullptr;
  const Kernel* k = nullptr;
  if (const hipError_t e = kernel_of(s, p->func, d.device, &d.module_fn, &m, &k); e != hipSuccess) return e;
  if (const hipError_t e = check_launch(s, d.device, *k, p->gridDim, p->blockDim, p->sharedMemBytes, nullptr);
      e != hipSuccess)
    return e;
  if (const hipError_t e = build_kernargs(*k, p->kernelParams, p->extra, &d.args); e != hipSuccess) return e;
  d.func = p->func;
  d.grid = p->gridDim;
  d.block = p->blockDim;
  d.shared = p->sharedMemBytes;
  d.extra = p->extra;
  d.from_extra = p->extra && !p->kernelParams;
  fill_slots(*k, &d);
  d.attrs = std::move(out->attrs);   // a node's attributes stay with it
  d.cooperative = out->cooperative;
  *out = std::move(d);
  return hipSuccess;
}
void kernel_params_out(KernelData& d, vgpu::amd::abi::KernelNodeParams* p) {
  d.params.assign(d.slots.size(), nullptr);
  for (size_t i = 0; i < d.slots.size(); ++i)
    if (d.slots[i].second && d.slots[i].first + d.slots[i].second <= d.args.size()) d.params[i] = d.args.data() + d.slots[i].first;
  p->func = const_cast<void*>(d.func);
  p->gridDim = d.grid;
  p->blockDim = d.block;
  p->sharedMemBytes = d.shared;
  p->extra = d.extra;
  p->kernelParams = d.from_extra ? nullptr : d.params.data();
}

// ---- Copy and fill nodes: their regions, from the calls that check them ------------------
//
// A copy or fill node's parameters go through the same checks as the call
// would (copy_region, fill_region), which hand back the regions instead of
// queuing anything; a launch runs the region copy or fill right there.
struct SinkScope {
  explicit SinkScope(RegionSink* sink) : prev(t_sink) { t_sink = sink; }
  ~SinkScope() { t_sink = prev; }
  RegionSink* prev;
};
hipError_t record_copy(CopyData* d, const std::function<hipError_t()>& call) {
  RegionSink sink;
  sink.mode = RegionSink::kRecord;
  hipError_t e;
  {
    SinkScope scope(&sink);
    e = call();
  }
  if (e != hipSuccess) return e;
  d->empty = !sink.got;
  if (sink.got) {
    d->rdst = sink.dst;
    d->rsrc = sink.src;
    d->dr = sink.dr;
    d->sr = sink.sr;
    d->rkind = sink.kind;
  }
  return hipSuccess;
}
// Whether an address is a device's memory (not host memory, pinned or not).
bool device_memory(State& s, const void* p) {
  Side side;
  return locate(s, p, Region{1, 1, 1, 1, 0}, &side) == hipSuccess && side.device;
}
constexpr int kMemcpyH2H = hipMemcpyHostToHost, kMemcpyH2D = hipMemcpyHostToDevice,
              kMemcpyD2H = hipMemcpyDeviceToHost, kMemcpyD2D = hipMemcpyDeviceToDevice,
              kMemcpyDefault = hipMemcpyDefault;

// hipMemcpy3DParms: the checks hipMemcpy3D makes, and a row's pitch no
// narrower than the row (hipErrorInvalidPitchValue, before any other), nor
// the row running past it.
hipError_t make_copy_3d(const vgpu::amd::abi::Memcpy3DParms* p, CopyData* d) {
  if (!p) return hipErrorInvalidValue;
  const auto& e = p->extent;
  for (const vgpu::amd::abi::PitchedPtr* side : {&p->srcPtr, &p->dstPtr})
    if (side->ptr && side->pitch && side->pitch < e.width) return hipErrorInvalidPitchValue;
  if ((p->srcPtr.ptr && p->srcPtr.pitch && p->srcPos.x + e.width > p->srcPtr.pitch) ||
      (p->dstPtr.ptr && p->dstPtr.pitch && p->dstPos.x + e.width > p->dstPtr.pitch))
    return hipErrorInvalidValue;
  CopyData c;
  c.form = kCopy3D;
  c.p3 = *p;
  if (const hipError_t err = record_copy(&c, [p] { return copy_3d(p, nullptr, true); }); err != hipSuccess) return err;
  *d = c;
  return hipSuccess;
}
hipError_t make_copy_1d(void* dst, const void* src, size_t count, int kind, CopyData* d) {
  if (!dst || !src) return hipErrorInvalidValue;
  CopyData c;
  c.form = kCopy1D;
  c.dst = dst;
  c.src = src;
  c.count = count;
  c.kind = kind;
  if (const hipError_t e = record_copy(&c, [=] {
        return copy_in_order(dst, src, count, static_cast<hipMemcpyKind>(kind), nullptr, true);
      });
      e != hipSuccess)
    return e;
  *d = c;
  return hipSuccess;
}
// A copy from or to a __device__ or __constant__ variable of the program's,
// named by its host-side variable: `count` bytes at `offset` into it. Host
// memory to or from it needs a kind that says so (or the default).
hipError_t make_copy_symbol(bool to_symbol, void* ptr, const void* symbol, size_t count, size_t offset, int kind,
                            CopyData* d) {
  State& s = state();
  uint64_t address = 0;
  size_t size = 0;
  bool host_side = false;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!symbol) return hipErrorInvalidSymbol;
    if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return e;
    if (symbol_on(s, symbol, &address, &size) != hipSuccess) return hipErrorInvalidSymbol;
    if (!ptr) return hipErrorInvalidValue;
    if (!count || count > size || offset > size - count) return hipErrorInvalidValue;
    host_side = !device_memory(s, ptr);
  }
  if (kind < kMemcpyH2H || (kind > kMemcpyDefault && kind != 1024)) return hipErrorInvalidMemcpyDirection;
  if (host_side && kind != kMemcpyDefault && kind != (to_symbol ? kMemcpyH2D : kMemcpyD2H))
    return hipErrorInvalidValue;
  CopyData c;
  c.form = to_symbol ? kCopyToSymbol : kCopyFromSymbol;
  c.symbol = symbol;
  c.count = count;
  c.offset = offset;
  c.kind = kind;
  void* dev = reinterpret_cast<void*>(address + offset);
  if (to_symbol) {
    c.src = ptr;
    c.dst = dev;
  } else {
    c.dst = ptr;
    c.src = dev;
  }
  void* dst = c.dst;
  const void* src = c.src;
  if (const hipError_t e = record_copy(&c, [=] {
        return copy_in_order(dst, src, count, hipMemcpyDefault, nullptr, true);
      });
      e != hipSuccess)
    return e;
  // What GetParams gives back is the caller's pointer on its side.
  if (to_symbol) c.src = ptr;
  else c.dst = ptr;
  *d = c;
  return hipSuccess;
}
// HIP_MEMCPY3D: the driver's form, whose every geometry fault is the wrong
// value.
hipError_t make_copy_drv(const vgpu::amd::abi::Memcpy3D* p, CopyData* d) {
  if (!p) return hipErrorInvalidValue;
  if ((p->srcPitch && p->srcXInBytes + p->WidthInBytes > p->srcPitch) ||
      (p->dstPitch && p->dstXInBytes + p->WidthInBytes > p->dstPitch))
    return hipErrorInvalidValue;
  CopyData c;
  c.form = kCopyDrv;
  c.drv = *p;
  const hipError_t e = record_copy(&c, [p] { return drv_3d(p, nullptr, true); });
  if (e != hipSuccess) return e == hipErrorInvalidPitchValue ? hipErrorInvalidValue : e;
  *d = c;
  return hipSuccess;
}
// What a copy node's GetParams gives for a 1D, symbol or captured copy: one
// pitched pointer on each side, shaped like the copy.
vgpu::amd::abi::Memcpy3DParms params_3d(const CopyData& c) {
  if (c.form == kCopy3D) return c.p3;
  vgpu::amd::abi::Memcpy3DParms p{};
  if (c.form == kCopyCaptured || c.form == kCopyDrv) {
    const size_t rows_s = c.sr.pitch ? (c.sr.slice ? c.sr.slice / c.sr.pitch : c.sr.rows) : 1;
    const size_t rows_d = c.dr.pitch ? (c.dr.slice ? c.dr.slice / c.dr.pitch : c.dr.rows) : 1;
    p.srcPtr = {const_cast<void*>(c.rsrc), c.sr.pitch, c.sr.width, rows_s};
    p.dstPtr = {c.rdst, c.dr.pitch, c.dr.width, rows_d};
    p.extent = {c.dr.width, c.dr.rows, c.dr.depth};
    p.kind = c.rkind;
    return p;
  }
  p.srcPtr = {const_cast<void*>(c.src), c.count, c.count, 1};
  p.dstPtr = {c.dst, c.count, c.count, 1};
  p.extent = {c.count, 1, 1};
  p.kind = c.kind;
  return p;
}
int copy_kind(const CopyData& c) { return c.form == kCopy3D ? c.p3.kind : c.form == kCopyDrv ? kMemcpyDefault : c.kind; }

// hipMemsetParams: 1, 2 or 4-byte elements, at least a row, inside the
// allocation.
hipError_t make_fill(const vgpu::amd::abi::MemsetParams* p, FillData* d, size_t depth = 1, size_t slice = 0) {
  if (!p || !p->dst) return hipErrorInvalidValue;
  if (p->elementSize != 1 && p->elementSize != 2 && p->elementSize != 4) return hipErrorInvalidValue;
  if (!p->height || !p->width) return hipErrorInvalidValue;
  const uint64_t row = uint64_t{p->width} * p->elementSize;
  const uint64_t pitch = p->height > 1 ? p->pitch : row;
  if (p->height > 1 && pitch < row) return hipErrorInvalidValue;
  RegionSink sink;
  sink.mode = RegionSink::kRecord;
  uint8_t pattern[4];
  std::memcpy(pattern, &p->value, 4);
  hipError_t e;
  {
    SinkScope scope(&sink);
    e = fill_region(p->dst, Region{row, p->height, depth, pitch, slice}, pattern, p->elementSize, nullptr, true);
  }
  if (e != hipSuccess) return e;
  d->p = *p;
  d->depth = depth;
  d->slice = slice;
  return hipSuccess;
}

// ---- Launching: a graph as one piece of its stream's work ----------------------------------

// One node, ready to run: what it needs copied out of the graph when the
// launch was made, so the graph may change or go while it runs.
struct Step {
  int type = vgpu::amd::abi::kNodeEmpty;
  KernelData kernel;
  void* dst = nullptr;
  const void* src = nullptr;
  std::shared_ptr<const std::vector<uint8_t>> snap;   // keeps `src` valid where it points into one
  Region dr, sr;
  int kind = 0;
  uint8_t pattern[4] = {};
  uint32_t pattern_len = 0;
  vgpu::amd::abi::HostNodeParams host{};
  std::shared_ptr<Stamp> stamp;   // a record node's
  bool waits = false;             // a wait node's, bound at launch
  Queue::Marker wait;
  uint64_t va = 0;                // an allocation's address, or what a free frees
  bool auto_free = false;
  std::vector<uint8_t> ops;       // a batch of memory operations
};
struct Plan {
  // The launching call's caller's stack pointer, about: what lies at or above it
  // is still a live frame of the thread that launched.
  const char* live_from = nullptr;
  std::vector<Step> steps;
  std::vector<std::pair<hipEvent_t, std::shared_ptr<Stamp>>> records;
};

// The steps of a graph, in order, a child graph's in place of its node.
void plan_graph(State& s, const GraphObj& g, unsigned long long flags, Plan* plan) {
  std::vector<const GNode*> order;
  topo_order(g, &order);
  for (const GNode* n : order) {
    Step st;
    st.type = n->enabled ? n->type : vgpu::amd::abi::kNodeEmpty;
    switch (st.type) {
      case vgpu::amd::abi::kNodeKernel: st.kernel = n->kernel; break;
      case vgpu::amd::abi::kNodeMemcpy:
        if (n->copy.empty) st.type = vgpu::amd::abi::kNodeEmpty;
        st.dst = n->copy.rdst;
        st.src = n->copy.rsrc;
        // A source that was a stack variable when captured is taken from the capture
        // once its frame is gone (a library's local, copied from after it returned,
        // as rocPRIM's histogram does); a frame still there is read as it is now,
        // as a host buffer refilled before each launch is.
        if (n->copy.snap && !(n->copy.snap_thread == std::this_thread::get_id() && plan->live_from &&
                              static_cast<const char*>(n->copy.rsrc) >= plan->live_from)) {
          st.snap = n->copy.snap;
          st.src = st.snap->data();
        }
        st.dr = n->copy.dr;
        st.sr = n->copy.sr;
        st.kind = n->copy.rkind;
        break;
      case vgpu::amd::abi::kNodeMemset: {
        const auto& p = n->fill.p;
        st.dst = p.dst;
        const uint64_t row = uint64_t{p.width} * p.elementSize;
        st.dr = Region{row, p.height, n->fill.depth, p.height > 1 ? p.pitch : row, n->fill.slice};
        std::memcpy(st.pattern, &p.value, 4);
        st.pattern_len = p.elementSize;
        break;
      }
      case vgpu::amd::abi::kNodeHost: st.host = n->host; break;
      case vgpu::amd::abi::kNodeGraph:
        plan_graph(s, *n->child, flags, plan);
        continue;
      case vgpu::amd::abi::kNodeEventRecord:
        st.stamp = std::make_shared<Stamp>();
        plan->records.emplace_back(n->event, st.stamp);
        break;
      case vgpu::amd::abi::kNodeWaitEvent: {
        // Bound now to the event's last record before this launch, as a
        // stream's wait is when it is queued.
        bool exists = false;
        st.waits = event_marker(n->event, &st.wait, nullptr, &exists);
        break;
      }
      case vgpu::amd::abi::kNodeMemAlloc:
        st.va = reinterpret_cast<uint64_t>(n->alloc.dptr);
        st.auto_free = flags & 1;
        break;
      case vgpu::amd::abi::kNodeMemFree: st.va = reinterpret_cast<uint64_t>(n->free_ptr); break;
      case vgpu::amd::abi::kNodeBatchMemOp: st.ops = n->batch_ops; break;
      default: break;
    }
    plan->steps.push_back(std::move(st));
  }
}

// Record nodes' times go forward by at least a microsecond each, so the time
// between two is never zero however little ran between them.
std::chrono::steady_clock::time_point stamp_now() {
  static std::mutex mu;
  static std::chrono::steady_clock::time_point last{};
  std::lock_guard<std::mutex> lock(mu);
  auto now = std::chrono::steady_clock::now();
  if (now <= last) now = last + std::chrono::microseconds(1);
  last = now;
  return now;
}

hipError_t run_step(const Step& st, const Queue* self) {
  State& s = state();
  switch (st.type) {
    case vgpu::amd::abi::kNodeKernel: {
      LaunchJob job;
      {
        std::lock_guard<std::mutex> lock(s.mutex);
        bool module_fn = false;
        const Module* m = nullptr;
        const Kernel* k = nullptr;
        if (const hipError_t e = kernel_of(s, st.kernel.func, st.kernel.device, &module_fn, &m, &k); e != hipSuccess)
          return e;
        std::vector<uint8_t> args = st.kernel.args;
        if (args.size() < k->kernarg_size) args.resize(k->kernarg_size, 0);
        if (const hipError_t e = prepare_launch(s, module_fn ? m->device : st.kernel.device, *m, *k, st.kernel.grid,
                                                st.kernel.block, st.kernel.shared, std::move(args), nullptr,
                                                st.kernel.cooperative, &job);
            e != hipSuccess)
          return e;
      }
      return run_launch(job);
    }
    case vgpu::amd::abi::kNodeMemcpy: {
      RegionSink sink;
      sink.mode = RegionSink::kRun;
      SinkScope scope(&sink);
      return copy_region(const_cast<void*>(st.dst), st.dr, st.src, st.sr, static_cast<hipMemcpyKind>(st.kind), nullptr,
                         true);
    }
    case vgpu::amd::abi::kNodeMemset: {
      RegionSink sink;
      sink.mode = RegionSink::kRun;
      SinkScope scope(&sink);
      return fill_region(const_cast<void*>(st.dst), st.dr, st.pattern, st.pattern_len, nullptr, true);
    }
    case vgpu::amd::abi::kNodeHost: st.host.fn(st.host.userData); return hipSuccess;
    case vgpu::amd::abi::kNodeEventRecord:
      st.stamp->when = stamp_now();
      st.stamp->taken.store(true, std::memory_order_release);
      return hipSuccess;
    case vgpu::amd::abi::kNodeWaitEvent:
      if (st.waits && st.wait.queue.get() != self) st.wait.queue->wait(st.wait.seq);
      return hipSuccess;
    case vgpu::amd::abi::kNodeMemAlloc: {
      std::lock_guard<std::mutex> lock(s.mutex);
      const auto it = gt().allocs.find(st.va);
      if (it == gt().allocs.end()) return hipErrorInvalidValue;
      // Launched again with its allocation still live: freed first, where
      // the graph was instantiated to (hipGraphInstantiateFlagAutoFreeOnLaunch).
      if (it->second.committed && st.auto_free) free_graph_alloc(s, st.va);
      return commit_graph_alloc(s, st.va);
    }
    case vgpu::amd::abi::kNodeMemFree: {
      {
        std::lock_guard<std::mutex> lock(s.mutex);
        if (gt().allocs.count(st.va)) return free_graph_alloc(s, st.va);
      }
      return free_now(reinterpret_cast<void*>(st.va));
    }
    case vgpu::amd::abi::kNodeBatchMemOp:
      for (size_t i = 0; i + 48 <= st.ops.size(); i += 48) {
        const auto* op = reinterpret_cast<const StreamMemOp*>(st.ops.data() + i);
        const unsigned bytes = op->operation == 4 || op->operation == 5 ? 8 : 4;
        vgpu::MemoryManager* mem = nullptr;
        {
          std::lock_guard<std::mutex> lock(s.mutex);
          if (const hipError_t e = memory_at(s, op->address, bytes, &mem); e != hipSuccess) return e;
        }
        const uint64_t at = reinterpret_cast<uint64_t>(op->address);
        try {
          if (op->operation == 2 || op->operation == 5) {
            const uint64_t v = bytes == 8 ? op->value64 : op->value;
            mem->write(at, &v, bytes);
          } else if (op->operation == 1 || op->operation == 4) {
            const uint64_t want = bytes == 8 ? op->value64 : op->value;
            for (;;) {
              uint64_t v = 0;
              mem->read(at, &v, bytes);
              if (passes(v, want, ~0ull, op->flags, bytes)) break;
              std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
          }
        } catch (const std::exception& e) {
          return fail(hipErrorInvalidValue, e.what());
        }
      }
      return hipSuccess;
    default: return hipSuccess;
  }
}

// ---- Capture ------------------------------------------------------------------------------
//
// A capture records what its streams are asked to do into a graph: each
// operation becomes a node after the stream's current dependencies, which
// are then that node. An event recorded in a capture stands for the stream's
// dependencies at that point; another stream that waits on it joins the
// capture there. Nothing runs until the graph is launched.
struct CapStream {
  std::vector<GNode*> deps;
  uint64_t ops = 0;          // operations captured on it
  uint64_t joined_ops = 0;   // how many of them another stream of the capture has waited for
  std::vector<void*> deps_out;
};
struct Capture {
  uint64_t id = 0;
  int mode = 0;   // hipStreamCaptureMode: global, thread-local, relaxed
  hipStream_t origin = nullptr;
  std::thread::id thread;
  GraphObj* graph = nullptr;
  bool own_graph = true;   // a graph of its own, not one given to hipStreamBeginCaptureToGraph
  bool invalidated = false;
  std::map<hipStream_t, CapStream> streams;
};
struct CaptureTables {
  uint64_t next_id = 1;
  std::map<uint64_t, std::unique_ptr<Capture>> active;
  std::map<hipStream_t, Capture*> of_stream;
};
CaptureTables& ct() {
  static CaptureTables& t = *new CaptureTables;
  return t;
}
thread_local int t_capture_mode = 0;   // hipStreamCaptureMode, per thread as HIP keeps it

Capture* capture_of(hipStream_t resolved) {
  if (!resolved) return nullptr;
  const auto it = ct().of_stream.find(resolved);
  return it == ct().of_stream.end() ? nullptr : it->second;
}
bool blocking_stream(State& s, hipStream_t h, int* device) {
  const auto it = s.streams.find(h);
  if (it == s.streams.end()) return false;
  *device = it->second.device;
  return !(it->second.flags & hipStreamNonBlocking);
}
// The captures a blocking stream of `device` takes part in.
std::vector<Capture*> blocking_captures(State& s, int device) {
  std::vector<Capture*> out;
  for (auto& [id, c] : ct().active)
    for (auto& [h, cs] : c->streams) {
      int d = -1;
      if (blocking_stream(s, h, &d) && d == device) {
        out.push_back(c.get());
        break;
      }
    }
  return out;
}
void end_capture(Capture* c) {
  for (auto& [h, cs] : c->streams) ct().of_stream.erase(h);
  ct().active.erase(c->id);
}

}  // namespace

// The hooks hip_api.cpp calls (declared there).
namespace {

bool stream_capturing(State&, hipStream_t resolved) { return capture_of(resolved) != nullptr; }

bool any_capture_on(State& s, int device) {
  for (auto& [id, c] : ct().active)
    for (auto& [h, cs] : c->streams) {
      const auto it = s.streams.find(h);
      if (it != s.streams.end() && it->second.device == device) return true;
    }
  return false;
}

hipError_t legacy_conflict(State& s, int device) {
  if (ct().active.empty()) return hipSuccess;
  const std::vector<Capture*> cs = blocking_captures(s, device);
  if (cs.empty()) return hipSuccess;
  for (Capture* c : cs) c->invalidated = true;
  return hipErrorStreamCaptureImplicit;
}

hipError_t unsafe_during_capture(State&) {
  if (ct().active.empty() || t_capture_mode == 2) return hipSuccess;
  std::vector<Capture*> culprits;
  const std::thread::id me = std::this_thread::get_id();
  for (auto& [id, c] : ct().active) {
    if (c->mode == 2) continue;
    if (c->thread == me || (t_capture_mode == 0 && c->mode == 0)) culprits.push_back(c.get());
  }
  if (culprits.empty()) return hipSuccess;
  for (Capture* c : culprits) c->invalidated = true;
  // As HIP documents: a call that synchronizes or allocates is not allowed while a global (or this
  // thread's own) capture is on, and invalidates the capture.
  return fail(hipErrorStreamCaptureUnsupported, "a call that is unsafe during a stream capture was made while a capture "
                                                "in global or thread-local mode is on; the capture is invalidated");
}

GNode* capture_append(Capture* c, hipStream_t resolved, std::unique_ptr<GNode> n) {
  CapStream& cs = c->streams[resolved];
  GNode* p = add_node(*c->graph, std::move(n), cs.deps);
  cs.deps = {p};
  ++cs.ops;
  return p;
}

// Whether `p` is on the calling thread's stack.
bool on_this_stack(const void* p) {
  pthread_attr_t a;
  if (pthread_getattr_np(pthread_self(), &a) != 0) return false;
  void* lo = nullptr;
  size_t size = 0;
  const bool ok = pthread_attr_getstack(&a, &lo, &size) == 0;
  pthread_attr_destroy(&a);
  const char* q = static_cast<const char*>(p);
  return ok && q >= static_cast<const char*>(lo) && q < static_cast<const char*>(lo) + size;
}

hipError_t capture_copy(State&, hipStream_t resolved, void* dst, const Region& dr, const void* src, const Region& sr,
                        hipMemcpyKind kind) {
  Capture* c = capture_of(resolved);
  auto n = std::make_unique<GNode>();
  n->type = vgpu::amd::abi::kNodeMemcpy;
  n->copy.form = kCopyCaptured;
  n->copy.empty = dr.empty();
  n->copy.rdst = dst;
  n->copy.rsrc = src;
  n->copy.dr = dr;
  n->copy.sr = sr;
  n->copy.rkind = kind;
  n->copy.kind = kind;
  if (!dr.empty() && sr.width && on_this_stack(src)) {
    const uint64_t last = (sr.depth ? sr.depth - 1 : 0) * sr.slice + (sr.rows ? sr.rows - 1 : 0) * sr.pitch + sr.width;
    n->copy.snap = std::make_shared<std::vector<uint8_t>>(static_cast<const uint8_t*>(src),
                                                         static_cast<const uint8_t*>(src) + last);
    n->copy.snap_thread = std::this_thread::get_id();
  }
  capture_append(c, resolved, std::move(n));
  return hipSuccess;
}

hipError_t capture_fill(State&, hipStream_t resolved, void* dst, const Region& r, const uint8_t* pattern,
                        uint32_t len) {
  Capture* c = capture_of(resolved);
  auto n = std::make_unique<GNode>();
  n->type = vgpu::amd::abi::kNodeMemset;
  auto& p = n->fill.p;
  p.dst = dst;
  p.elementSize = len;
  p.width = r.width / len;
  p.height = r.rows;
  p.pitch = r.pitch;
  std::memcpy(&p.value, pattern, len);
  n->fill.depth = r.depth;
  n->fill.slice = r.slice;
  capture_append(c, resolved, std::move(n));
  return hipSuccess;
}

hipError_t capture_kernel(State& s, hipStream_t resolved, const void* func, const Kernel& k, Dim3 grid, Dim3 block,
                          uint32_t shared, std::vector<uint8_t> args, void** extra, bool cooperative) {
  Capture* c = capture_of(resolved);
  auto n = std::make_unique<GNode>();
  n->type = vgpu::amd::abi::kNodeKernel;
  KernelData& d = n->kernel;
  d.func = func;
  d.module_fn = false;
  for (const auto& f : s.functions)
    if (f.get() == func) d.module_fn = true;
  d.device = s.current;
  d.grid = grid;
  d.block = block;
  d.shared = shared;
  d.args = std::move(args);
  d.extra = extra;
  d.from_extra = extra != nullptr;
  d.cooperative = cooperative;
  fill_slots(k, &d);
  capture_append(c, resolved, std::move(n));
  return hipSuccess;
}

hipError_t capture_host(State&, hipStream_t resolved, void (*fn)(void*), void* data) {
  Capture* c = capture_of(resolved);
  auto n = std::make_unique<GNode>();
  n->type = vgpu::amd::abi::kNodeHost;
  n->host = {fn, data};
  capture_append(c, resolved, std::move(n));
  return hipSuccess;
}

// A graph allocation: address space set aside now, on the device, which the
// node's every launch maps memory at.
hipError_t reserve_graph_alloc(State& s, int device, size_t bytes, void** out) {
  if (!bytes) return hipErrorInvalidValue;
  const uint64_t reserved = granules(bytes);
  uint64_t va = 0;
  try {
    va = s.rt->device(device).memory().reserve(reserved, 0);
  } catch (const std::exception& e) {
    return fail(hipErrorOutOfMemory, e.what());
  }
  GraphAlloc a;
  a.device = device;
  a.bytes = bytes;
  a.reserved = reserved;
  gt().allocs[va] = a;
  *out = reinterpret_cast<void*>(va);
  return hipSuccess;
}

hipError_t capture_alloc(State& s, hipStream_t resolved, int device, size_t bytes, void** ptr) {
  Capture* c = capture_of(resolved);
  if (!ptr) return hipErrorInvalidValue;
  auto n = std::make_unique<GNode>();
  n->type = vgpu::amd::abi::kNodeMemAlloc;
  auto& p = n->alloc;
  p.poolProps.allocType = 1;
  p.poolProps.location.type = 1;
  p.poolProps.location.id = device;
  p.bytesize = bytes;
  if (const hipError_t e = reserve_graph_alloc(s, device, bytes, &p.dptr); e != hipSuccess) return e;
  *ptr = p.dptr;
  capture_append(c, resolved, std::move(n));
  return hipSuccess;
}

hipError_t capture_free(State&, hipStream_t resolved, void* ptr) {
  Capture* c = capture_of(resolved);
  if (!ptr) return hipErrorInvalidValue;
  auto n = std::make_unique<GNode>();
  n->type = vgpu::amd::abi::kNodeMemFree;
  n->free_ptr = ptr;
  GNode* p = capture_append(c, resolved, std::move(n));
  if (const auto it = gt().allocs.find(reinterpret_cast<uint64_t>(ptr)); it != gt().allocs.end())
    it->second.free_node = p->handle;
  return hipSuccess;
}

// An event recorded in a capture stands for the stream's dependencies there.
hipError_t capture_record(State&, hipStream_t resolved, Event* e) {
  Capture* c = capture_of(resolved);
  CapStream& cs = c->streams[resolved];
  e->capture = c->id;
  e->capture_stream = resolved;
  e->capture_ops = cs.ops;
  e->capture_deps.clear();
  for (GNode* d : cs.deps) e->capture_deps.push_back(d->handle);
  return hipSuccess;
}

// A stream waiting on an event recorded in a capture takes part in it from
// there; one already in it takes the event's dependencies on as well. True
// where the wait was the capture's to handle.
bool capture_wait(State&, hipStream_t resolved, const Event* e, hipError_t* result) {
  *result = hipSuccess;
  const auto it = e->capture ? ct().active.find(e->capture) : ct().active.end();
  Capture* mine = capture_of(resolved);
  if (it == ct().active.end()) return mine != nullptr;   // an event from outside: nothing to wait for in a graph
  Capture* c = it->second.get();
  if (mine && mine != c) {
    *result = hipErrorStreamCaptureMerge;
    return true;
  }
  if (!resolved) {
    *result = hipErrorStreamCaptureImplicit;
    return true;
  }
  CapStream& cs = c->streams[resolved];
  if (!mine) ct().of_stream[resolved] = c;
  for (void* h : e->capture_deps)
    if (GNode* d = find_node(h); d && std::find(cs.deps.begin(), cs.deps.end(), d) == cs.deps.end())
      cs.deps.push_back(d);
  if (const auto from = c->streams.find(e->capture_stream);
      from != c->streams.end() && from->first != resolved && e->capture_ops == from->second.ops)
    from->second.joined_ops = from->second.ops;
  return true;
}

// Whether an event was recorded in a capture that is still going on.
bool captured_event(const Event* e) { return e->capture && ct().active.count(e->capture); }

// A stream destroyed while it takes part in a capture leaves it; the stream
// a capture began on takes the capture with it (and a graph of the
// capture's own).
void forget_stream(State&, hipStream_t h) {
  const auto it = ct().of_stream.find(h);
  if (it == ct().of_stream.end()) return;
  Capture* c = it->second;
  if (c->origin != h) {
    c->streams.erase(h);
    ct().of_stream.erase(it);
    return;
  }
  Destructors ignored;   // a capture's graph holds no user objects of the program's
  if (c->own_graph) destroy_graph(c->graph, &ignored);
  end_capture(c);
}

// hipFree of graph memory: the allocation is no longer live.
bool free_graph_memory(State& s, void* ptr, hipError_t* result) {
  const auto it = gt().allocs.find(reinterpret_cast<uint64_t>(ptr));
  if (it == gt().allocs.end()) return false;
  *result = free_graph_alloc(s, it->first);
  return true;
}
bool graph_memory_size(uint64_t va, uint64_t* base, uint64_t* size) {
  auto it = gt().allocs.upper_bound(va);
  if (it == gt().allocs.begin()) return false;
  --it;
  if (va >= it->first + it->second.bytes || !it->second.committed) return false;
  *base = it->first;
  *size = it->second.bytes;
  return true;
}

bool graph_memory_range(uint64_t va, uint64_t* base, uint64_t* size) {
  auto it = gt().allocs.upper_bound(va);
  if (it == gt().allocs.begin()) return false;
  --it;
  if (va >= it->first + it->second.bytes) return false;
  *base = it->first;
  *size = it->second.bytes;
  return true;
}

// Every node of a capture's graph a hipStreamGetCaptureInfo_v2 names.
void deps_out(CapStream& cs, const void*** deps, size_t* n) {
  cs.deps_out.clear();
  for (GNode* d : cs.deps) cs.deps_out.push_back(d->handle);
  if (deps) *deps = const_cast<const void**>(cs.deps_out.data());
  if (n) *n = cs.deps_out.size();
}

// Answers for the null stream while a blocking stream captures: it is
// implicated, without the capture being given up.
bool legacy_query(State& s, hipStream_t given) {
  if (resolve(s, given)) return false;
  return !blocking_captures(s, s.current).empty();
}

}  // namespace

extern "C" {

// ---- Graphs and their topology --------------------------------------------------------------

hipError_t hipGraphCreate(void** graph, unsigned int flags) {
  const ApiCall api("hipGraphCreate");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!graph || flags) return record(s, hipErrorInvalidValue);
  auto g = std::make_unique<GraphObj>();
  register_graph(*g);
  *graph = g->handle;
  gt().owned[g->handle] = std::move(g);
  return record(s, hipSuccess);
}

// A graph embedded in a child-graph node is the node's, not the program's.
hipError_t hipGraphDestroy(void* graph) {
  const ApiCall api("hipGraphDestroy");
  State& s = state();
  Destructors d;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    GraphObj* g = find_graph(graph);
    if (!g || g->owner || !gt().owned.count(graph)) return record(s, hipErrorInvalidValue);
    for (auto& [id, c] : ct().active)
      if (c->graph == g) return record(s, hipErrorIllegalState);
    destroy_graph(g, &d);
  }
  run_destructors(d);
  return hipSuccess;
}

hipError_t hipGraphAddEmptyNode(void** node, void* graph, const void* deps, size_t n) {
  const ApiCall api("hipGraphAddEmptyNode");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = nullptr;
  std::vector<GNode*> d;
  if (const hipError_t e = add_target(node, graph, deps, n, &g, &d); e != hipSuccess) return record(s, e);
  *node = add_node(*g, std::make_unique<GNode>(), d)->handle;
  return record(s, hipSuccess);
}

// Edges: each a pair of distinct nodes of the graph, not already joined. A
// cycle is allowed here, and refused when the graph is instantiated.
hipError_t hipGraphAddDependencies(void* graph, const void* from, const void* to, size_t n) {
  const ApiCall api("hipGraphAddDependencies");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = find_graph(graph);
  if (!g) return record(s, hipErrorInvalidValue);
  if (!n) return record(s, hipSuccess);
  if (!from || !to) return record(s, hipErrorInvalidValue);
  std::vector<std::pair<GNode*, GNode*>> add;
  for (size_t i = 0; i < n; ++i) {
    GNode* a = find_node(static_cast<void* const*>(from)[i]);
    GNode* b = find_node(static_cast<void* const*>(to)[i]);
    if (!a || !b || a->graph != g || b->graph != g || a == b) return record(s, hipErrorInvalidValue);
    if (std::find(a->out.begin(), a->out.end(), b) != a->out.end()) return record(s, hipErrorInvalidValue);
    if (std::find(add.begin(), add.end(), std::make_pair(a, b)) != add.end()) return record(s, hipErrorInvalidValue);
    add.emplace_back(a, b);
  }
  for (auto [a, b] : add) {
    a->out.push_back(b);
    b->in.push_back(a);
    g->edges.emplace_back(a, b);
  }
  return record(s, hipSuccess);
}

hipError_t hipGraphRemoveDependencies(void* graph, const void* from, const void* to, size_t n) {
  const ApiCall api("hipGraphRemoveDependencies");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = find_graph(graph);
  if (!g) return record(s, hipErrorInvalidValue);
  if (!n) return record(s, hipSuccess);
  if (!from || !to) return record(s, hipErrorInvalidValue);
  std::vector<std::pair<GNode*, GNode*>> drop;
  for (size_t i = 0; i < n; ++i) {
    GNode* a = find_node(static_cast<void* const*>(from)[i]);
    GNode* b = find_node(static_cast<void* const*>(to)[i]);
    if (!a || !b || a->graph != g || b->graph != g) return record(s, hipErrorInvalidValue);
    if (std::find(a->out.begin(), a->out.end(), b) == a->out.end()) return record(s, hipErrorInvalidValue);
    if (std::find(drop.begin(), drop.end(), std::make_pair(a, b)) != drop.end()) return record(s, hipErrorInvalidValue);
    drop.emplace_back(a, b);
  }
  for (auto [a, b] : drop) {
    a->out.erase(std::find(a->out.begin(), a->out.end(), b));
    b->in.erase(std::find(b->in.begin(), b->in.end(), a));
    g->edges.erase(std::find(g->edges.begin(), g->edges.end(), std::make_pair(a, b)));
  }
  return record(s, hipSuccess);
}

}  // extern "C"
namespace {
// A list query's answer: with no array, the count; with one, up to *count
// entries -- and where it asked for more than there are, the rest null and
// *count what there are.
hipError_t list_out(const std::vector<void*>& all, void** out, size_t* count) {
  if (!count) return hipErrorInvalidValue;
  if (!out) {
    *count = all.size();
    return hipSuccess;
  }
  const size_t n = std::min(*count, all.size());
  for (size_t i = 0; i < n; ++i) out[i] = all[i];
  for (size_t i = n; i < *count; ++i) out[i] = nullptr;
  if (*count > all.size()) *count = all.size();
  return hipSuccess;
}
std::vector<void*> handles(const std::vector<GNode*>& nodes) {
  std::vector<void*> out;
  for (GNode* n : nodes) out.push_back(n->handle);
  return out;
}
}  // namespace
extern "C" {

hipError_t hipGraphGetNodes(void* graph, void** nodes, size_t* count) {
  const ApiCall api("hipGraphGetNodes");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = find_graph(graph);
  if (!g) return record(s, hipErrorInvalidValue);
  std::vector<void*> all;
  for (const auto& n : g->nodes) all.push_back(n->handle);
  return record(s, list_out(all, nodes, count));
}
hipError_t hipGraphGetRootNodes(void* graph, void** nodes, size_t* count) {
  const ApiCall api("hipGraphGetRootNodes");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = find_graph(graph);
  if (!g) return record(s, hipErrorInvalidValue);
  std::vector<void*> all;
  for (const auto& n : g->nodes)
    if (n->in.empty()) all.push_back(n->handle);
  return record(s, list_out(all, nodes, count));
}
hipError_t hipGraphGetEdges(void* graph, void** from, void** to, size_t* count) {
  const ApiCall api("hipGraphGetEdges");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = find_graph(graph);
  if (!g || !count || (!from) != (!to)) return record(s, hipErrorInvalidValue);
  if (!from) {
    *count = g->edges.size();
    return record(s, hipSuccess);
  }
  const size_t n = std::min(*count, g->edges.size());
  for (size_t i = 0; i < n; ++i) {
    from[i] = g->edges[i].first->handle;
    to[i] = g->edges[i].second->handle;
  }
  for (size_t i = n; i < *count; ++i) from[i] = to[i] = nullptr;
  if (*count > g->edges.size()) *count = g->edges.size();
  return record(s, hipSuccess);
}
hipError_t hipGraphNodeGetDependencies(void* node, void** deps, size_t* count) {
  const ApiCall api("hipGraphNodeGetDependencies");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = find_node(node);
  if (!n) return record(s, hipErrorInvalidValue);
  return record(s, list_out(handles(n->in), deps, count));
}
hipError_t hipGraphNodeGetDependentNodes(void* node, void** deps, size_t* count) {
  const ApiCall api("hipGraphNodeGetDependentNodes");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = find_node(node);
  if (!n) return record(s, hipErrorInvalidValue);
  return record(s, list_out(handles(n->out), deps, count));
}
hipError_t hipGraphNodeGetType(void* node, int* type) {
  const ApiCall api("hipGraphNodeGetType");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = find_node(node);
  if (!n || !type) return record(s, hipErrorInvalidValue);
  *type = n->type;
  return record(s, hipSuccess);
}

// A node goes with its edges; the nodes on either side are not joined up.
// An allocation node stays: the memory it stands for is already out there.
hipError_t hipGraphDestroyNode(void* node) {
  const ApiCall api("hipGraphDestroyNode");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = find_node(node);
  if (!n) return record(s, hipErrorInvalidValue);
  if (n->type == vgpu::amd::abi::kNodeMemAlloc) return record(s, hipErrorNotSupported);
  GraphObj* g = n->graph;
  for (GNode* a : n->in) a->out.erase(std::find(a->out.begin(), a->out.end(), n));
  for (GNode* b : n->out) b->in.erase(std::find(b->in.begin(), b->in.end(), n));
  g->edges.erase(std::remove_if(g->edges.begin(), g->edges.end(),
                                [n](const auto& e) { return e.first == n || e.second == n; }),
                 g->edges.end());
  for (auto& [id, c] : ct().active)
    for (auto& [h, cs] : c->streams) cs.deps.erase(std::remove(cs.deps.begin(), cs.deps.end(), n), cs.deps.end());
  if (n->type == vgpu::amd::abi::kNodeMemFree)
    for (auto& [va, a] : gt().allocs)
      if (a.free_node == n->handle) a.free_node = nullptr;
  unregister_node(*n);
  g->nodes.erase(std::find_if(g->nodes.begin(), g->nodes.end(), [n](const auto& p) { return p.get() == n; }));
  return record(s, hipSuccess);
}

// A copy that owes the original nothing, with references of its own to the
// user objects it holds. A graph with memory nodes cannot be copied.
hipError_t hipGraphClone(void** out, void* graph) {
  const ApiCall api("hipGraphClone");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = find_graph(graph);
  if (!out || !g) return record(s, hipErrorInvalidValue);
  if (has_memory_nodes(*g)) return record(s, hipErrorNotSupported);
  std::map<const GNode*, GNode*> m;
  auto c = copy_graph(*g, true, &m);
  take_refs(g->user_objects, &c->user_objects);
  *out = c->handle;
  gt().owned[c->handle] = std::move(c);
  return record(s, hipSuccess);
}
// The node of a copy that a node of the graph it was copied from became.
hipError_t hipGraphNodeFindInClone(void** out, void* original, void* clone) {
  const ApiCall api("hipGraphNodeFindInClone");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = find_graph(clone);
  if (!out || !original || !g) return record(s, hipErrorInvalidValue);
  for (const auto& n : g->nodes)
    if (n->source == original) {
      *out = n->handle;
      return record(s, hipSuccess);
    }
  return record(s, hipErrorInvalidValue);
}

// A child-graph node embeds a copy of the graph as it is now: the graph
// itself stays the program's (it may even be the graph the node joins).
hipError_t hipGraphAddChildGraphNode(void** node, void* graph, const void* deps, size_t n, void* child) {
  const ApiCall api("hipGraphAddChildGraphNode");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = nullptr;
  std::vector<GNode*> d;
  if (const hipError_t e = add_target(node, graph, deps, n, &g, &d); e != hipSuccess) return record(s, e);
  GraphObj* c = find_graph(child);
  if (!c) return record(s, hipErrorInvalidValue);
  auto nd = std::make_unique<GNode>();
  nd->type = vgpu::amd::abi::kNodeGraph;
  std::map<const GNode*, GNode*> m;
  nd->child = copy_graph(*c, false, &m);
  nd->child->owner = nd.get();
  *node = add_node(*g, std::move(nd), d)->handle;
  return record(s, hipSuccess);
}
// The node's own graph, which edits reach the node through.
hipError_t hipGraphChildGraphNodeGetGraph(void* node, void** graph) {
  const ApiCall api("hipGraphChildGraphNodeGetGraph");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = find_node(node);
  if (!n || !graph || n->type != vgpu::amd::abi::kNodeGraph) return record(s, hipErrorInvalidValue);
  *graph = n->child->handle;
  return record(s, hipSuccess);
}

// ---- Kernel nodes ----------------------------------------------------------------------------

hipError_t hipGraphAddKernelNode(void** node, void* graph, const void* deps, size_t n,
                                 const vgpu::amd::abi::KernelNodeParams* params) {
  const ApiCall api("hipGraphAddKernelNode");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = nullptr;
  std::vector<GNode*> d;
  if (const hipError_t e = add_target(node, graph, deps, n, &g, &d); e != hipSuccess) return record(s, e);
  auto nd = std::make_unique<GNode>();
  nd->type = vgpu::amd::abi::kNodeKernel;
  if (const hipError_t e = make_kernel(s, params, &nd->kernel); e != hipSuccess) return record(s, e);
  *node = add_node(*g, std::move(nd), d)->handle;
  return record(s, hipSuccess);
}
hipError_t hipGraphKernelNodeGetParams(void* node, vgpu::amd::abi::KernelNodeParams* params) {
  const ApiCall api("hipGraphKernelNodeGetParams");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = find_node(node);
  if (!n || !params || n->type != vgpu::amd::abi::kNodeKernel) return record(s, hipErrorInvalidValue);
  kernel_params_out(n->kernel, params);
  return record(s, hipSuccess);
}
hipError_t hipGraphKernelNodeSetParams(void* node, const vgpu::amd::abi::KernelNodeParams* params) {
  const ApiCall api("hipGraphKernelNodeSetParams");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = find_node(node);
  if (!n || !params || n->type != vgpu::amd::abi::kNodeKernel) return record(s, hipErrorInvalidValue);
  return record(s, make_kernel(s, params, &n->kernel));
}

// Kernel node attributes: kept as set. An access policy window's hit ratio is
// a fraction, its size at most the device's largest, and what misses cannot
// persist.
hipError_t hipGraphKernelNodeSetAttribute(void* node, int attr, const vgpu::amd::abi::LaunchAttributeValue* value) {
  const ApiCall api("hipGraphKernelNodeSetAttribute");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = find_node(node);
  if (!n || !value || n->type != vgpu::amd::abi::kNodeKernel || attr < 1 || attr > 10) return record(s, hipErrorInvalidValue);
  if (attr == 1) {
    const auto& w = value->access_policy_window;
    if (!(w.hit_ratio >= 0.0f && w.hit_ratio <= 1.0f) || w.miss_prop == 2 || w.num_bytes > kAccessPolicyMaxWindow)
      return record(s, hipErrorInvalidValue);
  }
  n->kernel.attrs[attr] = *value;
  if (attr == 2) n->kernel.cooperative = value->cooperative != 0;
  return record(s, hipSuccess);
}
hipError_t hipGraphKernelNodeGetAttribute(void* node, int attr, vgpu::amd::abi::LaunchAttributeValue* value) {
  const ApiCall api("hipGraphKernelNodeGetAttribute");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = find_node(node);
  if (!n || !value || n->type != vgpu::amd::abi::kNodeKernel || attr < 1 || attr > 10) return record(s, hipErrorInvalidValue);
  const auto it = n->kernel.attrs.find(attr);
  if (it != n->kernel.attrs.end()) *value = it->second;
  else std::memset(value, 0, sizeof *value);
  return record(s, hipSuccess);
}
hipError_t hipGraphKernelNodeCopyAttributes(void* src, void* dst) {
  const ApiCall api("hipGraphKernelNodeCopyAttributes");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* a = find_node(src);
  GNode* b = find_node(dst);
  if (!a || !b || a->type != vgpu::amd::abi::kNodeKernel || b->type != vgpu::amd::abi::kNodeKernel)
    return record(s, hipErrorInvalidValue);
  b->kernel.attrs = a->kernel.attrs;
  b->kernel.cooperative = a->kernel.cooperative;
  return record(s, hipSuccess);
}

// ---- Copy nodes -----------------------------------------------------------------------------

}  // extern "C"
namespace {
// An Add call for a node made from parameters checked without the lock (they
// go through the copy and fill checks, which take it).
hipError_t add_made(void** node, void* graph, const void* deps, size_t n, std::unique_ptr<GNode> nd) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = nullptr;
  std::vector<GNode*> d;
  if (const hipError_t e = add_target(node, graph, deps, n, &g, &d); e != hipSuccess) return e;
  *node = add_node(*g, std::move(nd), d)->handle;
  return hipSuccess;
}
// The common checks of an Add call, made before its parameters are.
hipError_t add_precheck(void** node, void* graph, const void* deps, size_t n) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = nullptr;
  std::vector<GNode*> d;
  return add_target(node, graph, deps, n, &g, &d);
}
// A node of the type asked for, for a Set or Get call, or null.
GNode* typed_node(void* node, int type) {
  GNode* n = find_node(node);
  return n && n->type == type ? n : nullptr;
}
bool copy_node_live(void* node) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return typed_node(node, vgpu::amd::abi::kNodeMemcpy) != nullptr;
}
hipError_t set_copy(void* node, const CopyData& c) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = typed_node(node, vgpu::amd::abi::kNodeMemcpy);
  if (!n) return hipErrorInvalidValue;
  n->copy = c;
  return hipSuccess;
}
}  // namespace
extern "C" {

hipError_t hipGraphAddMemcpyNode(void** node, void* graph, const void* deps, size_t n,
                                 const vgpu::amd::abi::Memcpy3DParms* params) {
  const ApiCall api("hipGraphAddMemcpyNode");
  if (const hipError_t e = add_precheck(node, graph, deps, n); e != hipSuccess) return record(state(), e);
  auto nd = std::make_unique<GNode>();
  nd->type = vgpu::amd::abi::kNodeMemcpy;
  if (const hipError_t e = make_copy_3d(params, &nd->copy); e != hipSuccess) return record(state(), e);
  return record(state(), add_made(node, graph, deps, n, std::move(nd)));
}
hipError_t hipGraphAddMemcpyNode1D(void** node, void* graph, const void* deps, size_t n, void* dst, const void* src,
                                   size_t count, hipMemcpyKind kind) {
  const ApiCall api("hipGraphAddMemcpyNode1D");
  if (const hipError_t e = add_precheck(node, graph, deps, n); e != hipSuccess) return record(state(), e);
  auto nd = std::make_unique<GNode>();
  nd->type = vgpu::amd::abi::kNodeMemcpy;
  if (const hipError_t e = make_copy_1d(dst, src, count, kind, &nd->copy); e != hipSuccess) return record(state(), e);
  return record(state(), add_made(node, graph, deps, n, std::move(nd)));
}
hipError_t hipGraphAddMemcpyNodeFromSymbol(void** node, void* graph, const void* deps, size_t n, void* dst,
                                           const void* symbol, size_t count, size_t offset, hipMemcpyKind kind) {
  const ApiCall api("hipGraphAddMemcpyNodeFromSymbol");
  if (const hipError_t e = add_precheck(node, graph, deps, n); e != hipSuccess) return record(state(), e);
  auto nd = std::make_unique<GNode>();
  nd->type = vgpu::amd::abi::kNodeMemcpy;
  if (const hipError_t e = make_copy_symbol(false, dst, symbol, count, offset, kind, &nd->copy); e != hipSuccess)
    return record(state(), e);
  return record(state(), add_made(node, graph, deps, n, std::move(nd)));
}
hipError_t hipGraphAddMemcpyNodeToSymbol(void** node, void* graph, const void* deps, size_t n, const void* symbol,
                                         const void* src, size_t count, size_t offset, hipMemcpyKind kind) {
  const ApiCall api("hipGraphAddMemcpyNodeToSymbol");
  if (const hipError_t e = add_precheck(node, graph, deps, n); e != hipSuccess) return record(state(), e);
  auto nd = std::make_unique<GNode>();
  nd->type = vgpu::amd::abi::kNodeMemcpy;
  if (const hipError_t e = make_copy_symbol(true, const_cast<void*>(src), symbol, count, offset, kind, &nd->copy);
      e != hipSuccess)
    return record(state(), e);
  return record(state(), add_made(node, graph, deps, n, std::move(nd)));
}
hipError_t hipGraphMemcpyNodeGetParams(void* node, vgpu::amd::abi::Memcpy3DParms* params) {
  const ApiCall api("hipGraphMemcpyNodeGetParams");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = typed_node(node, vgpu::amd::abi::kNodeMemcpy);
  if (!n || !params) return record(s, hipErrorInvalidValue);
  *params = params_3d(n->copy);
  return record(s, hipSuccess);
}
hipError_t hipGraphMemcpyNodeSetParams(void* node, const vgpu::amd::abi::Memcpy3DParms* params) {
  const ApiCall api("hipGraphMemcpyNodeSetParams");
  if (!copy_node_live(node)) return record(state(), hipErrorInvalidValue);
  CopyData c;
  if (const hipError_t e = make_copy_3d(params, &c); e != hipSuccess) return record(state(), e);
  return record(state(), set_copy(node, c));
}
// A 1D copy node's new copy: of something, not onto itself.
hipError_t hipGraphMemcpyNodeSetParams1D(void* node, void* dst, const void* src, size_t count, hipMemcpyKind kind) {
  const ApiCall api("hipGraphMemcpyNodeSetParams1D");
  if (!copy_node_live(node) || !count || dst == src) return record(state(), hipErrorInvalidValue);
  CopyData c;
  if (const hipError_t e = make_copy_1d(dst, src, count, kind, &c); e != hipSuccess) return record(state(), e);
  return record(state(), set_copy(node, c));
}
hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(void* node, void* dst, const void* symbol, size_t count,
                                                 size_t offset, hipMemcpyKind kind) {
  const ApiCall api("hipGraphMemcpyNodeSetParamsFromSymbol");
  if (!copy_node_live(node)) return record(state(), hipErrorInvalidValue);
  CopyData c;
  if (const hipError_t e = make_copy_symbol(false, dst, symbol, count, offset, kind, &c); e != hipSuccess)
    return record(state(), e);
  return record(state(), set_copy(node, c));
}
hipError_t hipGraphMemcpyNodeSetParamsToSymbol(void* node, const void* symbol, const void* src, size_t count,
                                               size_t offset, hipMemcpyKind kind) {
  const ApiCall api("hipGraphMemcpyNodeSetParamsToSymbol");
  if (!copy_node_live(node)) return record(state(), hipErrorInvalidValue);
  CopyData c;
  if (const hipError_t e = make_copy_symbol(true, const_cast<void*>(src), symbol, count, offset, kind, &c);
      e != hipSuccess)
    return record(state(), e);
  return record(state(), set_copy(node, c));
}

// The driver API's copy node, in HIP_MEMCPY3D's terms. The context is any
// the program has; addresses are unified, so it is not needed to find them.
hipError_t hipDrvGraphAddMemcpyNode(void** node, void* graph, const void* deps, size_t n,
                                    const vgpu::amd::abi::Memcpy3D* params, void*) {
  const ApiCall api("hipDrvGraphAddMemcpyNode");
  if (const hipError_t e = add_precheck(node, graph, deps, n); e != hipSuccess) return record(state(), e);
  auto nd = std::make_unique<GNode>();
  nd->type = vgpu::amd::abi::kNodeMemcpy;
  if (const hipError_t e = make_copy_drv(params, &nd->copy); e != hipSuccess) return record(state(), e);
  return record(state(), add_made(node, graph, deps, n, std::move(nd)));
}
hipError_t hipDrvGraphMemcpyNodeGetParams(void* node, vgpu::amd::abi::Memcpy3D* params) {
  const ApiCall api("hipDrvGraphMemcpyNodeGetParams");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = typed_node(node, vgpu::amd::abi::kNodeMemcpy);
  if (!n || !params) return record(s, hipErrorInvalidValue);
  if (n->copy.form == kCopyDrv) {
    *params = n->copy.drv;
  } else {
    std::memset(params, 0, sizeof *params);
    const auto p = params_3d(n->copy);
    params->srcMemoryType = vgpu::amd::abi::kMemTypeUnified;
    params->srcDevice = p.srcPtr.ptr;
    params->srcPitch = p.srcPtr.pitch;
    params->srcXInBytes = p.srcPos.x;
    params->srcY = p.srcPos.y;
    params->srcZ = p.srcPos.z;
    params->dstMemoryType = vgpu::amd::abi::kMemTypeUnified;
    params->dstDevice = p.dstPtr.ptr;
    params->dstPitch = p.dstPtr.pitch;
    params->dstXInBytes = p.dstPos.x;
    params->dstY = p.dstPos.y;
    params->dstZ = p.dstPos.z;
    params->WidthInBytes = p.extent.width;
    params->Height = p.extent.height;
    params->Depth = p.extent.depth;
  }
  return record(s, hipSuccess);
}
hipError_t hipDrvGraphMemcpyNodeSetParams(void* node, const vgpu::amd::abi::Memcpy3D* params) {
  const ApiCall api("hipDrvGraphMemcpyNodeSetParams");
  if (!copy_node_live(node)) return record(state(), hipErrorInvalidValue);
  CopyData c;
  if (const hipError_t e = make_copy_drv(params, &c); e != hipSuccess) return record(state(), e);
  return record(state(), set_copy(node, c));
}

// ---- Fill nodes --------------------------------------------------------------------------------

hipError_t hipGraphAddMemsetNode(void** node, void* graph, const void* deps, size_t n,
                                 const vgpu::amd::abi::MemsetParams* params) {
  const ApiCall api("hipGraphAddMemsetNode");
  if (const hipError_t e = add_precheck(node, graph, deps, n); e != hipSuccess) return record(state(), e);
  auto nd = std::make_unique<GNode>();
  nd->type = vgpu::amd::abi::kNodeMemset;
  if (const hipError_t e = make_fill(params, &nd->fill); e != hipSuccess) return record(state(), e);
  return record(state(), add_made(node, graph, deps, n, std::move(nd)));
}
hipError_t hipDrvGraphAddMemsetNode(void** node, void* graph, const void* deps, size_t n,
                                    const vgpu::amd::abi::MemsetParams* params, void*) {
  const ApiCall api("hipDrvGraphAddMemsetNode");
  return hipGraphAddMemsetNode(node, graph, deps, n, params);
}
hipError_t hipGraphMemsetNodeGetParams(void* node, vgpu::amd::abi::MemsetParams* params) {
  const ApiCall api("hipGraphMemsetNodeGetParams");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = typed_node(node, vgpu::amd::abi::kNodeMemset);
  if (!n || !params) return record(s, hipErrorInvalidValue);
  *params = n->fill.p;
  return record(s, hipSuccess);
}
hipError_t hipGraphMemsetNodeSetParams(void* node, const vgpu::amd::abi::MemsetParams* params) {
  const ApiCall api("hipGraphMemsetNodeSetParams");
  {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!typed_node(node, vgpu::amd::abi::kNodeMemset)) return record(s, hipErrorInvalidValue);
  }
  FillData f;
  if (const hipError_t e = make_fill(params, &f); e != hipSuccess) return record(state(), e);
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = typed_node(node, vgpu::amd::abi::kNodeMemset);
  if (!n) return record(s, hipErrorInvalidValue);
  n->fill = f;
  return record(s, hipSuccess);
}

// ---- Host, event and empty nodes ------------------------------------------------------------------

hipError_t hipGraphAddHostNode(void** node, void* graph, const void* deps, size_t n,
                               const vgpu::amd::abi::HostNodeParams* params) {
  const ApiCall api("hipGraphAddHostNode");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = nullptr;
  std::vector<GNode*> d;
  if (const hipError_t e = add_target(node, graph, deps, n, &g, &d); e != hipSuccess) return record(s, e);
  if (!params || !params->fn) return record(s, hipErrorInvalidValue);
  auto nd = std::make_unique<GNode>();
  nd->type = vgpu::amd::abi::kNodeHost;
  nd->host = *params;
  *node = add_node(*g, std::move(nd), d)->handle;
  return record(s, hipSuccess);
}
hipError_t hipGraphHostNodeGetParams(void* node, vgpu::amd::abi::HostNodeParams* params) {
  const ApiCall api("hipGraphHostNodeGetParams");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = typed_node(node, vgpu::amd::abi::kNodeHost);
  if (!n || !params) return record(s, hipErrorInvalidValue);
  *params = n->host;
  return record(s, hipSuccess);
}
hipError_t hipGraphHostNodeSetParams(void* node, const vgpu::amd::abi::HostNodeParams* params) {
  const ApiCall api("hipGraphHostNodeSetParams");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = typed_node(node, vgpu::amd::abi::kNodeHost);
  if (!n || !params || !params->fn) return record(s, hipErrorInvalidValue);
  n->host = *params;
  return record(s, hipSuccess);
}

}  // extern "C"
namespace {
bool live_event(hipEvent_t e) {
  std::lock_guard<std::mutex> lock(g_event_mutex);
  return find_event(e) != nullptr;
}
hipError_t add_event_node(void** node, void* graph, const void* deps, size_t n, hipEvent_t event, int type) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = nullptr;
  std::vector<GNode*> d;
  if (const hipError_t e = add_target(node, graph, deps, n, &g, &d); e != hipSuccess) return e;
  if (!event || !live_event(event)) return hipErrorInvalidValue;
  auto nd = std::make_unique<GNode>();
  nd->type = type;
  nd->event = event;
  *node = add_node(*g, std::move(nd), d)->handle;
  return hipSuccess;
}
hipError_t event_get(void* node, hipEvent_t* out, int type) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = typed_node(node, type);
  if (!n || !out) return hipErrorInvalidValue;
  *out = n->event;
  return hipSuccess;
}
hipError_t event_set(void* node, hipEvent_t event, int type) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = typed_node(node, type);
  if (!n || !event || !live_event(event)) return hipErrorInvalidValue;
  n->event = event;
  return hipSuccess;
}
}  // namespace
extern "C" {

hipError_t hipGraphAddEventRecordNode(void** node, void* graph, const void* deps, size_t n, hipEvent_t event) {
  const ApiCall api("hipGraphAddEventRecordNode");
  return record(state(), add_event_node(node, graph, deps, n, event, vgpu::amd::abi::kNodeEventRecord));
}
hipError_t hipGraphAddEventWaitNode(void** node, void* graph, const void* deps, size_t n, hipEvent_t event) {
  const ApiCall api("hipGraphAddEventWaitNode");
  return record(state(), add_event_node(node, graph, deps, n, event, vgpu::amd::abi::kNodeWaitEvent));
}
hipError_t hipGraphEventRecordNodeGetEvent(void* node, hipEvent_t* out) {
  const ApiCall api("hipGraphEventRecordNodeGetEvent");
  return record(state(), event_get(node, out, vgpu::amd::abi::kNodeEventRecord));
}
hipError_t hipGraphEventRecordNodeSetEvent(void* node, hipEvent_t event) {
  const ApiCall api("hipGraphEventRecordNodeSetEvent");
  return record(state(), event_set(node, event, vgpu::amd::abi::kNodeEventRecord));
}
hipError_t hipGraphEventWaitNodeGetEvent(void* node, hipEvent_t* out) {
  const ApiCall api("hipGraphEventWaitNodeGetEvent");
  return record(state(), event_get(node, out, vgpu::amd::abi::kNodeWaitEvent));
}
hipError_t hipGraphEventWaitNodeSetEvent(void* node, hipEvent_t event) {
  const ApiCall api("hipGraphEventWaitNodeSetEvent");
  return record(state(), event_set(node, event, vgpu::amd::abi::kNodeWaitEvent));
}

// ---- Memory nodes -----------------------------------------------------------------------------
//
// An allocation node sets its address aside when it is made, and hands it
// back in the parameters: every launch maps memory there, which stays until
// something frees it -- a free node, hipFree or hipFreeAsync -- however long
// the graph lasts.
hipError_t hipGraphAddMemAllocNode(void** node, void* graph, const void* deps, size_t n,
                                   vgpu::amd::abi::MemAllocNodeParams* params) {
  const ApiCall api("hipGraphAddMemAllocNode");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = nullptr;
  std::vector<GNode*> d;
  if (const hipError_t e = add_target(node, graph, deps, n, &g, &d); e != hipSuccess) return record(s, e);
  if (!params || params->poolProps.allocType != 1 || params->poolProps.location.type != 1 ||
      !valid_device(s, params->poolProps.location.id) || !params->bytesize ||
      params->accessDescCount > static_cast<size_t>(s.rt->device_count()))
    return record(s, hipErrorInvalidValue);
  void* va = nullptr;
  if (const hipError_t e = reserve_graph_alloc(s, params->poolProps.location.id, params->bytesize, &va); e != hipSuccess)
    return record(s, e);
  params->dptr = va;
  auto nd = std::make_unique<GNode>();
  nd->type = vgpu::amd::abi::kNodeMemAlloc;
  nd->alloc = *params;
  *node = add_node(*g, std::move(nd), d)->handle;
  return record(s, hipSuccess);
}
hipError_t hipGraphMemAllocNodeGetParams(void* node, vgpu::amd::abi::MemAllocNodeParams* params) {
  const ApiCall api("hipGraphMemAllocNodeGetParams");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = typed_node(node, vgpu::amd::abi::kNodeMemAlloc);
  if (!n || !params) return record(s, hipErrorInvalidValue);
  *params = n->alloc;
  return record(s, hipSuccess);
}
// A free node for device memory -- a graph's, or any allocation's start --
// that nothing else in any graph frees already.
hipError_t hipGraphAddMemFreeNode(void** node, void* graph, const void* deps, size_t n, void* dptr) {
  const ApiCall api("hipGraphAddMemFreeNode");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = nullptr;
  std::vector<GNode*> d;
  if (const hipError_t e = add_target(node, graph, deps, n, &g, &d); e != hipSuccess) return record(s, e);
  if (!dptr) return record(s, hipErrorInvalidValue);
  const uint64_t va = reinterpret_cast<uint64_t>(dptr);
  const auto a = gt().allocs.find(va);
  if (a != gt().allocs.end()) {
    if (a->second.free_node && find_node(a->second.free_node)) return record(s, hipErrorInvalidValue);
  } else {
    uint64_t base = 0, size = 0;
    const int owner = owner_of(s, va);
    if (owner < 0 || !s.rt->device(owner).memory().find_allocation(va, &base, &size) || base != va)
      return record(s, hipErrorInvalidValue);
  }
  auto nd = std::make_unique<GNode>();
  nd->type = vgpu::amd::abi::kNodeMemFree;
  nd->free_ptr = dptr;
  GNode* p = add_node(*g, std::move(nd), d);
  if (a != gt().allocs.end()) a->second.free_node = p->handle;
  *node = p->handle;
  return record(s, hipSuccess);
}
hipError_t hipDrvGraphAddMemFreeNode(void** node, void* graph, const void* deps, size_t n, void* dptr) {
  const ApiCall api("hipDrvGraphAddMemFreeNode");
  return hipGraphAddMemFreeNode(node, graph, deps, n, dptr);
}
hipError_t hipGraphMemFreeNodeGetParams(void* node, void* out) {
  const ApiCall api("hipGraphMemFreeNodeGetParams");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = typed_node(node, vgpu::amd::abi::kNodeMemFree);
  if (!n || !out) return record(s, hipErrorInvalidValue);
  *static_cast<void**>(out) = n->free_ptr;
  return record(s, hipSuccess);
}

// What graphs have allocated on a device: in use and reserved, now and at
// most. Only the two high-water marks may be set, and only to zero.
hipError_t hipDeviceGetGraphMemAttribute(int ordinal, int attr, void* value) {
  const ApiCall api("hipDeviceGetGraphMemAttribute");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!valid_device(s, ordinal)) return record(s, hipErrorInvalidDevice);
  if (attr < 0 || attr > 3 || !value) return record(s, hipErrorInvalidValue);
  const GraphMemoryCounts& c = graph_counts()[ordinal];
  const uint64_t v[4] = {c.used, c.used_high, c.reserved(), c.reserved_high};
  *static_cast<uint64_t*>(value) = v[attr];
  return record(s, hipSuccess);
}
hipError_t hipDeviceSetGraphMemAttribute(int ordinal, int attr, void* value) {
  const ApiCall api("hipDeviceSetGraphMemAttribute");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!valid_device(s, ordinal)) return record(s, hipErrorInvalidDevice);
  if ((attr != 1 && attr != 3) || !value || *static_cast<const uint64_t*>(value))
    return record(s, hipErrorInvalidValue);
  GraphMemoryCounts& c = graph_counts()[ordinal];
  if (attr == 1) c.used_high = 0;
  else c.reserved_high = 0;
  return record(s, hipSuccess);
}
// What freed graph memory the device keeps goes back.
hipError_t hipDeviceGraphMemTrim(int ordinal) {
  const ApiCall api("hipDeviceGraphMemTrim");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!valid_device(s, ordinal)) return record(s, hipErrorInvalidDevice);
  graph_counts()[ordinal].cached = 0;
  return record(s, hipSuccess);
}

// ---- Batch memory operation nodes -------------------------------------------------------------

hipError_t hipGraphAddBatchMemOpNode(void** node, void* graph, const void* deps, size_t n,
                                     const vgpu::amd::abi::BatchMemOpNodeParams* params) {
  const ApiCall api("hipGraphAddBatchMemOpNode");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = nullptr;
  std::vector<GNode*> d;
  if (const hipError_t e = add_target(node, graph, deps, n, &g, &d); e != hipSuccess) return record(s, e);
  if (!params || (params->count && !params->paramArray)) return record(s, hipErrorInvalidValue);
  auto nd = std::make_unique<GNode>();
  nd->type = vgpu::amd::abi::kNodeBatchMemOp;
  nd->batch = *params;
  const auto* p = static_cast<const uint8_t*>(params->paramArray);
  nd->batch_ops.assign(p, p + 48 * size_t{params->count});
  nd->batch.paramArray = nd->batch_ops.data();
  *node = add_node(*g, std::move(nd), d)->handle;
  return record(s, hipSuccess);
}
hipError_t hipGraphBatchMemOpNodeGetParams(void* node, vgpu::amd::abi::BatchMemOpNodeParams* params) {
  const ApiCall api("hipGraphBatchMemOpNodeGetParams");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = typed_node(node, vgpu::amd::abi::kNodeBatchMemOp);
  if (!n || !params) return record(s, hipErrorInvalidValue);
  *params = n->batch;
  params->paramArray = n->batch_ops.data();
  return record(s, hipSuccess);
}
hipError_t hipGraphBatchMemOpNodeSetParams(void* node, const vgpu::amd::abi::BatchMemOpNodeParams* params) {
  const ApiCall api("hipGraphBatchMemOpNodeSetParams");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = typed_node(node, vgpu::amd::abi::kNodeBatchMemOp);
  if (!n || !params || (params->count && !params->paramArray)) return record(s, hipErrorInvalidValue);
  std::vector<uint8_t> ops(static_cast<const uint8_t*>(params->paramArray),
                           static_cast<const uint8_t*>(params->paramArray) + 48 * size_t{params->count});
  n->batch = *params;
  n->batch_ops = std::move(ops);
  n->batch.paramArray = n->batch_ops.data();
  return record(s, hipSuccess);
}

// ---- External semaphore nodes --------------------------------------------------------------------
//
// Nothing here imports another API's semaphores (hipImportExternalSemaphore
// refuses them), so there are none for a node to signal or wait on.
hipError_t hipGraphAddExternalSemaphoresSignalNode(void** node, void* graph, const void* deps, size_t n,
                                                   const void* params) {
  const ApiCall api("hipGraphAddExternalSemaphoresSignalNode");
  if (const hipError_t e = add_precheck(node, graph, deps, n); e != hipSuccess) return record(state(), e);
  return record(state(), params ? hipErrorNotSupported : hipErrorInvalidValue);
}
hipError_t hipGraphAddExternalSemaphoresWaitNode(void** node, void* graph, const void* deps, size_t n,
                                                 const void* params) {
  const ApiCall api("hipGraphAddExternalSemaphoresWaitNode");
  if (const hipError_t e = add_precheck(node, graph, deps, n); e != hipSuccess) return record(state(), e);
  return record(state(), params ? hipErrorNotSupported : hipErrorInvalidValue);
}
hipError_t hipGraphExternalSemaphoresSignalNodeGetParams(void* node, void* params) {
  const ApiCall api("hipGraphExternalSemaphoresSignalNodeGetParams");
  return record(state(), node && params ? hipErrorNotSupported : hipErrorInvalidValue);
}
hipError_t hipGraphExternalSemaphoresSignalNodeSetParams(void* node, const void* params) {
  const ApiCall api("hipGraphExternalSemaphoresSignalNodeSetParams");
  return record(state(), node && params ? hipErrorNotSupported : hipErrorInvalidValue);
}
hipError_t hipGraphExternalSemaphoresWaitNodeGetParams(void* node, void* params) {
  const ApiCall api("hipGraphExternalSemaphoresWaitNodeGetParams");
  return record(state(), node && params ? hipErrorNotSupported : hipErrorInvalidValue);
}
hipError_t hipGraphExternalSemaphoresWaitNodeSetParams(void* node, const void* params) {
  const ApiCall api("hipGraphExternalSemaphoresWaitNodeSetParams");
  return record(state(), node && params ? hipErrorNotSupported : hipErrorInvalidValue);
}
hipError_t hipGraphExecExternalSemaphoresSignalNodeSetParams(void* exec, void* node, const void* params) {
  const ApiCall api("hipGraphExecExternalSemaphoresSignalNodeSetParams");
  return record(state(), exec && node && params ? hipErrorNotSupported : hipErrorInvalidValue);
}
hipError_t hipGraphExecExternalSemaphoresWaitNodeSetParams(void* exec, void* node, const void* params) {
  const ApiCall api("hipGraphExecExternalSemaphoresWaitNodeSetParams");
  return record(state(), exec && node && params ? hipErrorNotSupported : hipErrorInvalidValue);
}

// ---- Nodes by generic parameters --------------------------------------------------------------------

hipError_t hipGraphAddNode(void** node, void* graph, const void* deps, size_t n,
                           vgpu::amd::abi::GraphNodeParams* params) {
  const ApiCall api("hipGraphAddNode");
  if (const hipError_t e = add_precheck(node, graph, deps, n); e != hipSuccess) return record(state(), e);
  if (!params) return record(state(), hipErrorInvalidValue);
  using namespace vgpu::amd::abi;
  switch (params->type) {
    case kNodeKernel: return hipGraphAddKernelNode(node, graph, deps, n, &params->kernel);
    case kNodeMemcpy: return hipGraphAddMemcpyNode(node, graph, deps, n, &params->memcpy.copyParams);
    case kNodeMemset: return hipGraphAddMemsetNode(node, graph, deps, n, &params->memset);
    case kNodeHost: return hipGraphAddHostNode(node, graph, deps, n, &params->host);
    case kNodeGraph: return hipGraphAddChildGraphNode(node, graph, deps, n, params->graph);
    case kNodeEmpty: return hipGraphAddEmptyNode(node, graph, deps, n);
    case kNodeWaitEvent:
      return hipGraphAddEventWaitNode(node, graph, deps, n, static_cast<hipEvent_t>(params->event));
    case kNodeEventRecord:
      return hipGraphAddEventRecordNode(node, graph, deps, n, static_cast<hipEvent_t>(params->event));
    case kNodeMemAlloc: return hipGraphAddMemAllocNode(node, graph, deps, n, &params->alloc);
    case kNodeMemFree: return hipGraphAddMemFreeNode(node, graph, deps, n, params->free);
    default: return record(state(), hipErrorInvalidValue);
  }
}
hipError_t hipGraphNodeSetParams(void* node, vgpu::amd::abi::GraphNodeParams* params) {
  const ApiCall api("hipGraphNodeSetParams");
  int type = -1;
  {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    GNode* n = find_node(node);
    if (!n || !params) return record(s, hipErrorInvalidValue);
    type = n->type;
  }
  if (params->type != type) return record(state(), hipErrorInvalidValue);
  using namespace vgpu::amd::abi;
  switch (type) {
    case kNodeKernel: return hipGraphKernelNodeSetParams(node, &params->kernel);
    case kNodeMemcpy: return hipGraphMemcpyNodeSetParams(node, &params->memcpy.copyParams);
    case kNodeMemset: return hipGraphMemsetNodeSetParams(node, &params->memset);
    case kNodeHost: return hipGraphHostNodeSetParams(node, &params->host);
    case kNodeWaitEvent: return hipGraphEventWaitNodeSetEvent(node, static_cast<hipEvent_t>(params->event));
    case kNodeEventRecord: return hipGraphEventRecordNodeSetEvent(node, static_cast<hipEvent_t>(params->event));
    default: return record(state(), hipErrorInvalidValue);
  }
}

// ---- User objects -----------------------------------------------------------------------------------
//
// An object a graph keeps alive, which is destroyed -- its destructor run
// once -- when the last reference to it goes. Asking anything of one that is
// gone, or giving back more than there are, changes nothing.
hipError_t hipUserObjectCreate(void** object, void* ptr, void (*destroy)(void*), unsigned int refs,
                               unsigned int flags) {
  const ApiCall api("hipUserObjectCreate");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!object || !destroy || !refs || flags > 1) return record(s, hipErrorInvalidValue);
  const uint64_t id = reinterpret_cast<uint64_t>(new_handle());
  gt().user_objects[id] = UserObj{ptr, destroy, refs};
  *object = reinterpret_cast<void*>(id);
  return record(s, hipSuccess);
}
hipError_t hipUserObjectRetain(void* object, unsigned int count) {
  const ApiCall api("hipUserObjectRetain");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!object || !count) return record(s, hipErrorInvalidValue);
  if (const auto it = gt().user_objects.find(reinterpret_cast<uint64_t>(object)); it != gt().user_objects.end())
    it->second.refs += count;
  return record(s, hipSuccess);
}
hipError_t hipUserObjectRelease(void* object, unsigned int count) {
  const ApiCall api("hipUserObjectRelease");
  State& s = state();
  Destructors d;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!object || !count) return record(s, hipErrorInvalidValue);
    release_user_object(reinterpret_cast<uint64_t>(object), count, &d);
  }
  run_destructors(d);
  return hipSuccess;
}
// A graph's references: new ones, or (hipGraphUserObjectMove) the caller's.
hipError_t hipGraphRetainUserObject(void* graph, void* object, unsigned int count, unsigned int flags) {
  const ApiCall api("hipGraphRetainUserObject");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = find_graph(graph);
  if (!g || !object || !count || flags > 1) return record(s, hipErrorInvalidValue);
  const uint64_t id = reinterpret_cast<uint64_t>(object);
  const auto it = gt().user_objects.find(id);
  if (it == gt().user_objects.end()) return record(s, hipSuccess);
  if (!flags) it->second.refs += count;
  g->user_objects[id] += count;
  return record(s, hipSuccess);
}
hipError_t hipGraphReleaseUserObject(void* graph, void* object, unsigned int count) {
  const ApiCall api("hipGraphReleaseUserObject");
  State& s = state();
  Destructors d;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    GraphObj* g = find_graph(graph);
    if (!g || !object || !count) return record(s, hipErrorInvalidValue);
    const uint64_t id = reinterpret_cast<uint64_t>(object);
    const auto it = g->user_objects.find(id);
    if (it == g->user_objects.end() || count > it->second) return record(s, hipSuccess);
    it->second -= count;
    if (!it->second) g->user_objects.erase(it);
    release_user_object(id, count, &d);
  }
  run_destructors(d);
  return hipSuccess;
}

// ---- Executable graphs ------------------------------------------------------------------------------

}  // extern "C"
namespace {
// An executable graph: a copy of the graph, which checks it has no cycle,
// and -- where it allocates or frees -- that no other executable graph of it
// is there, since both would map or unmap memory at the same addresses (ROCm's
// HIP refuses a second one for a free node too).
hipError_t instantiate(State& s, void** out, void* graph, unsigned long long flags) {
  GraphObj* g = find_graph(graph);
  if (!out || !g) return hipErrorInvalidValue;
  if (!acyclic(*g)) return fail(hipErrorInvalidValue, "the graph's edges go round in a cycle");
  if (has_memory_nodes(*g) && g->live_execs) return hipErrorNotSupported;
  auto x = std::make_unique<ExecObj>();
  x->handle = new_handle();
  std::map<const GNode*, GNode*> m;
  x->copy = copy_graph(*g, false, &m);
  for (const auto& [from, to] : m) x->by_source[from->handle] = to;
  x->graph = graph;
  x->flags = flags;
  take_refs(g->user_objects, &x->user_objects);
  ++g->live_execs;
  *out = x->handle;
  gt().execs[x->handle] = std::move(x);
  return hipSuccess;
}
}  // namespace
extern "C" {

hipError_t hipGraphInstantiate(void** exec, void* graph, void** error_node, char* log, size_t log_size) {
  const ApiCall api("hipGraphInstantiate");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  const hipError_t e = instantiate(s, exec, graph, 0);
  if (e == hipSuccess && error_node) *error_node = nullptr;
  if (log && log_size) log[0] = '\0';
  return record(s, e);
}
// AutoFreeOnLaunch (1) and UseNodePriority (8) are taken; an upload on
// instantiation (2) and a launch from the device (4) are not.
hipError_t hipGraphInstantiateWithFlags(void** exec, void* graph, unsigned long long flags) {
  const ApiCall api("hipGraphInstantiateWithFlags");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (flags & ~9ull) return record(s, hipErrorInvalidValue);
  return record(s, instantiate(s, exec, graph, flags));
}
hipError_t hipGraphInstantiateWithParams(void** exec, void* graph, vgpu::amd::abi::GraphInstantiateParams* params) {
  const ApiCall api("hipGraphInstantiateWithParams");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!exec || !graph || !params) return record(s, hipErrorInvalidValue);
  if (params->flags & ~9ull) {
    params->result_out = 1;   // hipGraphInstantiateError
    return record(s, hipErrorInvalidValue);
  }
  const hipError_t e = instantiate(s, exec, graph, params->flags);
  params->result_out = e == hipSuccess ? 0 : 1;
  params->errNode_out = nullptr;
  return record(s, e);
}
hipError_t hipGraphExecGetFlags(void* exec, unsigned long long* flags) {
  const ApiCall api("hipGraphExecGetFlags");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  ExecObj* x = find_exec(exec);
  if (!x || !flags) return record(s, hipErrorInvalidValue);
  *flags = x->flags;
  return record(s, hipSuccess);
}
// Destroyed now; a launch still running finishes, and the user objects it
// holds go once it has.
hipError_t hipGraphExecDestroy(void* exec) {
  const ApiCall api("hipGraphExecDestroy");
  State& s = state();
  Destructors d;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto it = gt().execs.find(exec);
    if (it == gt().execs.end()) return record(s, hipErrorInvalidValue);
    std::unique_ptr<ExecObj> x = std::move(it->second);
    gt().execs.erase(it);
    if (GraphObj* g = find_graph(x->graph)) --g->live_execs;
    if (x->last_queue && !x->last_queue->done(x->last_seq) && !x->user_objects.empty()) {
      auto refs = std::make_shared<UserRefs>(std::move(x->user_objects));
      x->last_queue->submit([refs] {
        State& st = state();
        Destructors later;
        {
          std::lock_guard<std::mutex> lock(st.mutex);
          release_refs(*refs, &later);
        }
        run_destructors(later);
        return hipSuccess;
      });
    } else {
      release_refs(x->user_objects, &d);
    }
  }
  run_destructors(d);
  return hipSuccess;
}

// A launch is one piece of the stream's work, after what the stream has and
// after the graph's own last launch. One on the legacy null stream is waited
// for. On a stream that is capturing, the graph goes into the capture.
hipError_t hipGraphLaunch(void* exec, hipStream_t stream) {
  const ApiCall api("hipGraphLaunch");
  State& s = state();
  std::unique_lock<std::mutex> lock(s.mutex);
  ExecObj* x = find_exec(exec);
  if (!x) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  const hipStream_t r = resolve(s, stream);
  if (Capture* c = capture_of(r)) {
    auto n = std::make_unique<GNode>();
    n->type = vgpu::amd::abi::kNodeGraph;
    std::map<const GNode*, GNode*> m;
    n->child = copy_graph(*x->copy, false, &m);
    n->child->owner = n.get();
    capture_append(c, r, std::move(n));
    return record(s, hipSuccess);
  }
  Order o;
  if (const hipError_t e = order_for(s, stream, &o); e != hipSuccess) return record(s, e);
  if (x->last_queue && x->last_queue != o.queue) o.after.push_back({x->last_queue, x->last_seq});
  auto plan = std::make_shared<Plan>();
  plan->live_from = static_cast<const char*>(__builtin_frame_address(0)) + 2 * sizeof(void*);
  plan_graph(s, *x->copy, x->flags, plan.get());
  Queue* self = o.queue.get();
  const uint64_t seq = o.queue->submit(
      [plan, self] {
        for (const Step& st : plan->steps)
          if (const hipError_t e = run_step(st, self); e != hipSuccess) return e;
        return hipSuccess;
      },
      std::move(o.after));
  x->last_queue = o.queue;
  x->last_seq = seq;
  {
    std::lock_guard<std::mutex> event_lock(g_event_mutex);
    for (const auto& [event, stamp] : plan->records)
      if (Event* e = find_event(event)) {
        e->recorded = true;
        e->queue = o.queue;
        e->seq = seq;
        e->stamp = stamp;
      }
  }
  if (r || synchronous_launches()) return record(s, hipSuccess);
  std::shared_ptr<Queue> q = o.queue;
  lock.unlock();
  q->wait(seq);
  return record(s, q->take_error());
}
// Nothing to put on the device ahead of a launch: it is all in host memory.
hipError_t hipGraphUpload(void* exec, hipStream_t stream) {
  const ApiCall api("hipGraphUpload");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!find_exec(exec)) return record(s, hipErrorInvalidValue);
  Stream null_stream;
  if (stream && !find_stream(s, stream, &null_stream)) return record(s, hipErrorInvalidHandle);
  return record(s, hipSuccess);
}

// ---- Changing an executable graph ----------------------------------------------------------------------
//
// Each setter names the graph's node; its copy in the executable graph
// changes, for launches from then on, and the graph's node does not.

}  // extern "C"
namespace {
GNode* exec_node(void* exec, void* node, int type) {
  ExecObj* x = find_exec(exec);
  if (!x || !find_node(node)) return nullptr;
  const auto it = x->by_source.find(node);
  if (it == x->by_source.end() || (type >= 0 && it->second->type != type)) return nullptr;
  return it->second;
}
bool exec_node_live(void* exec, void* node, int type) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return exec_node(exec, node, type) != nullptr;
}
// A new copy for an executable graph's copy node: of the same kind.
hipError_t exec_set_copy(void* exec, void* node, const CopyData& c) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = exec_node(exec, node, vgpu::amd::abi::kNodeMemcpy);
  if (!n) return hipErrorInvalidValue;
  if (copy_kind(n->copy) != copy_kind(c)) return hipErrorInvalidValue;
  n->copy = c;
  return hipSuccess;
}
}  // namespace
extern "C" {

hipError_t hipGraphExecKernelNodeSetParams(void* exec, void* node, const vgpu::amd::abi::KernelNodeParams* params) {
  const ApiCall api("hipGraphExecKernelNodeSetParams");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = exec_node(exec, node, vgpu::amd::abi::kNodeKernel);
  if (!n || !params) return record(s, hipErrorInvalidValue);
  return record(s, make_kernel(s, params, &n->kernel));
}
hipError_t hipGraphExecHostNodeSetParams(void* exec, void* node, const vgpu::amd::abi::HostNodeParams* params) {
  const ApiCall api("hipGraphExecHostNodeSetParams");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = exec_node(exec, node, vgpu::amd::abi::kNodeHost);
  if (!n || !params || !params->fn) return record(s, hipErrorInvalidValue);
  n->host = *params;
  return record(s, hipSuccess);
}
hipError_t hipGraphExecMemcpyNodeSetParams(void* exec, void* node, const vgpu::amd::abi::Memcpy3DParms* params) {
  const ApiCall api("hipGraphExecMemcpyNodeSetParams");
  if (!exec_node_live(exec, node, vgpu::amd::abi::kNodeMemcpy) || !params) return record(state(), hipErrorInvalidValue);
  CopyData c;
  // Any fault in the new copy is the wrong value here, a pitch too narrow too.
  if (const hipError_t e = make_copy_3d(params, &c); e != hipSuccess)
    return record(state(), e == hipErrorInvalidPitchValue ? hipErrorInvalidValue : e);
  return record(state(), exec_set_copy(exec, node, c));
}
hipError_t hipGraphExecMemcpyNodeSetParams1D(void* exec, void* node, void* dst, const void* src, size_t count,
                                             hipMemcpyKind kind) {
  const ApiCall api("hipGraphExecMemcpyNodeSetParams1D");
  if (!exec_node_live(exec, node, vgpu::amd::abi::kNodeMemcpy) || !count || dst == src)
    return record(state(), hipErrorInvalidValue);
  CopyData c;
  if (const hipError_t e = make_copy_1d(dst, src, count, kind, &c); e != hipSuccess) return record(state(), e);
  return record(state(), exec_set_copy(exec, node, c));
}
hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(void* exec, void* node, void* dst, const void* symbol,
                                                     size_t count, size_t offset, hipMemcpyKind kind) {
  const ApiCall api("hipGraphExecMemcpyNodeSetParamsFromSymbol");
  if (!exec_node_live(exec, node, vgpu::amd::abi::kNodeMemcpy)) return record(state(), hipErrorInvalidValue);
  CopyData c;
  if (const hipError_t e = make_copy_symbol(false, dst, symbol, count, offset, kind, &c); e != hipSuccess)
    return record(state(), e);
  return record(state(), exec_set_copy(exec, node, c));
}
// The new side has to be on the device the old one was on.
hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(void* exec, void* node, const void* symbol, const void* src,
                                                   size_t count, size_t offset, hipMemcpyKind kind) {
  const ApiCall api("hipGraphExecMemcpyNodeSetParamsToSymbol");
  if (!exec_node_live(exec, node, vgpu::amd::abi::kNodeMemcpy)) return record(state(), hipErrorInvalidValue);
  CopyData c;
  if (const hipError_t e = make_copy_symbol(true, const_cast<void*>(src), symbol, count, offset, kind, &c);
      e != hipSuccess)
    return record(state(), e);
  {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    GNode* n = exec_node(exec, node, vgpu::amd::abi::kNodeMemcpy);
    if (!n) return record(s, hipErrorInvalidValue);
    const void* old = n->copy.form == kCopyToSymbol ? n->copy.src : n->copy.rsrc;
    const bool was_device = old && device_memory(s, old), is_device = device_memory(s, src);
    if (was_device && is_device &&
        owner_of(s, reinterpret_cast<uint64_t>(old)) != owner_of(s, reinterpret_cast<uint64_t>(src)))
      return record(s, hipErrorInvalidValue);
  }
  return record(state(), exec_set_copy(exec, node, c));
}
hipError_t hipDrvGraphExecMemcpyNodeSetParams(void* exec, void* node, const vgpu::amd::abi::Memcpy3D* params, void*) {
  const ApiCall api("hipDrvGraphExecMemcpyNodeSetParams");
  if (!exec_node_live(exec, node, vgpu::amd::abi::kNodeMemcpy)) return record(state(), hipErrorInvalidValue);
  CopyData c;
  if (const hipError_t e = make_copy_drv(params, &c); e != hipSuccess) return record(state(), e);
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = exec_node(exec, node, vgpu::amd::abi::kNodeMemcpy);
  if (!n) return record(s, hipErrorInvalidValue);
  n->copy = c;
  return record(s, hipSuccess);
}
// A fill of one row, on the device the node's was: its value, where it
// fills and how much.
hipError_t hipGraphExecMemsetNodeSetParams(void* exec, void* node, const vgpu::amd::abi::MemsetParams* params) {
  const ApiCall api("hipGraphExecMemsetNodeSetParams");
  int old_device = -1;
  {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    GNode* n = exec_node(exec, node, vgpu::amd::abi::kNodeMemset);
    if (!n || !params) return record(s, hipErrorInvalidValue);
    // A 2D memset keeps its shape, as ROCm's HIP holds it to: the same bytes
    // a row and the same rows (a 1D one only has to fit its allocation).
    const auto& was = n->fill.p;
    if (params->height > 1 &&
        (uint64_t{was.width} * was.elementSize != uint64_t{params->width} * params->elementSize ||
         was.height != params->height))
      return record(s, hipErrorInvalidValue);
    old_device = owner_of(s, reinterpret_cast<uint64_t>(n->fill.p.dst));
  }
  FillData f;
  if (const hipError_t e = make_fill(params, &f); e != hipSuccess) return record(state(), e);
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (owner_of(s, reinterpret_cast<uint64_t>(params->dst)) != old_device) return record(s, hipErrorInvalidValue);
  GNode* n = exec_node(exec, node, vgpu::amd::abi::kNodeMemset);
  if (!n) return record(s, hipErrorInvalidValue);
  n->fill = f;
  return record(s, hipSuccess);
}
hipError_t hipDrvGraphExecMemsetNodeSetParams(void* exec, void* node, const vgpu::amd::abi::MemsetParams* params,
                                              void*) {
  const ApiCall api("hipDrvGraphExecMemsetNodeSetParams");
  return hipGraphExecMemsetNodeSetParams(exec, node, params);
}
hipError_t hipGraphExecEventRecordNodeSetEvent(void* exec, void* node, hipEvent_t event) {
  const ApiCall api("hipGraphExecEventRecordNodeSetEvent");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = exec_node(exec, node, vgpu::amd::abi::kNodeEventRecord);
  if (!n || !event || !live_event(event)) return record(s, hipErrorInvalidValue);
  n->event = event;
  return record(s, hipSuccess);
}
hipError_t hipGraphExecEventWaitNodeSetEvent(void* exec, void* node, hipEvent_t event) {
  const ApiCall api("hipGraphExecEventWaitNodeSetEvent");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = exec_node(exec, node, vgpu::amd::abi::kNodeWaitEvent);
  if (!n || !event || !live_event(event)) return record(s, hipErrorInvalidValue);
  n->event = event;
  return record(s, hipSuccess);
}
hipError_t hipGraphExecBatchMemOpNodeSetParams(void* exec, void* node,
                                               const vgpu::amd::abi::BatchMemOpNodeParams* params) {
  const ApiCall api("hipGraphExecBatchMemOpNodeSetParams");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = exec_node(exec, node, vgpu::amd::abi::kNodeBatchMemOp);
  if (!n || !params || (params->count && !params->paramArray)) return record(s, hipErrorInvalidValue);
  std::vector<uint8_t> ops(static_cast<const uint8_t*>(params->paramArray),
                           static_cast<const uint8_t*>(params->paramArray) + 48 * size_t{params->count});
  n->batch = *params;
  n->batch_ops = std::move(ops);
  n->batch.paramArray = n->batch_ops.data();
  return record(s, hipSuccess);
}
hipError_t hipGraphExecChildGraphNodeSetParams(void* exec, void* node, void* child);
hipError_t hipGraphExecNodeSetParams(void* exec, void* node, vgpu::amd::abi::GraphNodeParams* params) {
  const ApiCall api("hipGraphExecNodeSetParams");
  if (!exec_node_live(exec, node, -1) || !params) return record(state(), hipErrorInvalidValue);
  using namespace vgpu::amd::abi;
  switch (params->type) {
    case kNodeKernel: return hipGraphExecKernelNodeSetParams(exec, node, &params->kernel);
    case kNodeMemcpy: return hipGraphExecMemcpyNodeSetParams(exec, node, &params->memcpy.copyParams);
    case kNodeMemset: return hipGraphExecMemsetNodeSetParams(exec, node, &params->memset);
    case kNodeHost: return hipGraphExecHostNodeSetParams(exec, node, &params->host);
    case kNodeGraph: return hipGraphExecChildGraphNodeSetParams(exec, node, params->graph);
    case kNodeWaitEvent: return hipGraphExecEventWaitNodeSetEvent(exec, node, static_cast<hipEvent_t>(params->event));
    case kNodeEventRecord:
      return hipGraphExecEventRecordNodeSetEvent(exec, node, static_cast<hipEvent_t>(params->event));
    default: return record(state(), hipErrorInvalidValue);
  }
}

}  // extern "C"
namespace {
// Whether two graphs have the same shape: as many nodes, of the same types
// in the same order, with the same edges between them.
bool same_shape(const GraphObj& a, const GraphObj& b) {
  if (a.nodes.size() != b.nodes.size()) return false;
  std::map<const GNode*, size_t> ia, ib;
  for (size_t i = 0; i < a.nodes.size(); ++i) {
    if (a.nodes[i]->type != b.nodes[i]->type) return false;
    ia[a.nodes[i].get()] = i;
    ib[b.nodes[i].get()] = i;
  }
  std::set<std::pair<size_t, size_t>> ea, eb;
  for (const auto& [x, y] : a.edges) ea.emplace(ia[x], ia[y]);
  for (const auto& [x, y] : b.edges) eb.emplace(ib[x], ib[y]);
  if (ea != eb) return false;
  for (size_t i = 0; i < a.nodes.size(); ++i)
    if (a.nodes[i]->child && !same_shape(*a.nodes[i]->child, *b.nodes[i]->child)) return false;
  return true;
}
// Takes on another graph's parameters, node for node (the shapes match).
void take_params(GraphObj& to, const GraphObj& from) {
  for (size_t i = 0; i < to.nodes.size(); ++i) {
    GNode& t = *to.nodes[i];
    const GNode& f = *from.nodes[i];
    const bool enabled = t.enabled;
    if (t.child) {
      take_params(*t.child, *f.child);
      continue;
    }
    copy_node_data(f, &t);
    t.enabled = enabled;
  }
}
}  // namespace
extern "C" {

// A child-graph node's contents replaced by another graph of the same shape.
hipError_t hipGraphExecChildGraphNodeSetParams(void* exec, void* node, void* child) {
  const ApiCall api("hipGraphExecChildGraphNodeSetParams");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = exec_node(exec, node, vgpu::amd::abi::kNodeGraph);
  GraphObj* c = find_graph(child);
  if (!n || !c || !same_shape(*n->child, *c)) return record(s, hipErrorInvalidValue);
  take_params(*n->child, *c);
  return record(s, hipSuccess);
}

// An executable graph brought up to a graph of the same shape: its nodes
// paired with the graph's in the order they were added, each pair of one
// type, the edges the same, and no copy changing direction nor kernel its
// device. What does not fit is told in the result.
hipError_t hipGraphExecUpdate(void* exec, void* graph, void** error_node, int* result) {
  const ApiCall api("hipGraphExecUpdate");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  ExecObj* x = find_exec(exec);
  GraphObj* g = find_graph(graph);
  if (!x || !g || !error_node || !result) return record(s, hipErrorInvalidValue);
  constexpr int kOk = 0, kTopology = 2, kNodeType = 3, kParameters = 5, kUnsupportedFunction = 7;
  const auto refuse = [&](int why, void* node) {
    *result = why;
    *error_node = node;
    return record(s, hipErrorGraphExecUpdateFailure);
  };
  GraphObj& to = *x->copy;
  if (to.nodes.size() != g->nodes.size()) return refuse(kTopology, nullptr);
  for (size_t i = 0; i < g->nodes.size(); ++i)
    if (to.nodes[i]->type != g->nodes[i]->type) return refuse(kNodeType, g->nodes[i]->handle);
  if (!same_shape(to, *g)) return refuse(kTopology, g->nodes.front()->handle);
  for (size_t i = 0; i < g->nodes.size(); ++i) {
    const GNode& f = *g->nodes[i];
    const GNode& t = *to.nodes[i];
    if (f.type == vgpu::amd::abi::kNodeMemcpy && copy_kind(f.copy) != copy_kind(t.copy))
      return refuse(kParameters, f.handle);
    if (f.type == vgpu::amd::abi::kNodeKernel && f.kernel.device != t.kernel.device)
      return refuse(kUnsupportedFunction, f.handle);
  }
  take_params(to, *g);
  // The executable graph's nodes now answer to the graph's.
  for (size_t i = 0; i < g->nodes.size(); ++i) {
    to.nodes[i]->source = g->nodes[i]->handle;
    x->by_source[g->nodes[i]->handle] = to.nodes[i].get();
  }
  *result = kOk;
  *error_node = nullptr;
  return record(s, hipSuccess);
}

// Whether a kernel, copy or fill node of an executable graph runs: one that
// does not is an empty node, its edges kept.
hipError_t hipGraphNodeSetEnabled(void* exec, void* node, unsigned int enabled) {
  const ApiCall api("hipGraphNodeSetEnabled");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = exec_node(exec, node, -1);
  if (!n || (n->type != vgpu::amd::abi::kNodeKernel && n->type != vgpu::amd::abi::kNodeMemcpy &&
             n->type != vgpu::amd::abi::kNodeMemset))
    return record(s, hipErrorInvalidValue);
  n->enabled = enabled != 0;
  return record(s, hipSuccess);
}
hipError_t hipGraphNodeGetEnabled(void* exec, void* node, unsigned int* enabled) {
  const ApiCall api("hipGraphNodeGetEnabled");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GNode* n = exec_node(exec, node, -1);
  if (!n || !enabled || (n->type != vgpu::amd::abi::kNodeKernel && n->type != vgpu::amd::abi::kNodeMemcpy &&
                         n->type != vgpu::amd::abi::kNodeMemset))
    return record(s, hipErrorInvalidValue);
  *enabled = n->enabled ? 1 : 0;
  return record(s, hipSuccess);
}

}  // extern "C"

// ---- Capture ----------------------------------------------------------------------------------------

namespace {
// A capture begins on a stream of the program's (or the thread's own): the
// null stream cannot be captured, nor a stream already in a capture.
hipError_t begin_capture(State& s, hipStream_t stream, int mode, GraphObj* into, const std::vector<GNode*>& deps) {
  if (mode < 0 || mode > 2) return hipErrorInvalidValue;
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return e;
  const hipStream_t r = resolve(s, stream);
  if (!r) return fail(hipErrorStreamCaptureUnsupported, "the null stream cannot be captured");
  if (!s.streams.count(r)) return hipErrorInvalidHandle;
  if (capture_of(r)) return hipErrorIllegalState;
  auto c = std::make_unique<Capture>();
  c->id = ct().next_id++;
  c->mode = mode;
  c->origin = r;
  c->thread = std::this_thread::get_id();
  if (into) {
    c->graph = into;
    c->own_graph = false;
  } else {
    auto g = std::make_unique<GraphObj>();
    register_graph(*g);
    c->graph = g.get();
    gt().owned[g->handle] = std::move(g);
  }
  c->streams[r].deps = deps;
  ct().of_stream[r] = c.get();
  ct().active[c->id] = std::move(c);
  return hipSuccess;
}
}  // namespace
extern "C" {

hipError_t hipStreamBeginCapture(hipStream_t stream, int mode) {
  const ApiCall api("hipStreamBeginCapture");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return record(s, begin_capture(s, stream, mode, nullptr, {}));
}
// A capture that adds to a graph the program has, after the nodes it names.
hipError_t hipStreamBeginCaptureToGraph(hipStream_t stream, void* graph, const void* deps, const void* edge_data,
                                        size_t n, int mode) {
  const ApiCall api("hipStreamBeginCaptureToGraph");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = find_graph(graph);
  if (!g || (n && !deps) || edge_data) return record(s, hipErrorInvalidValue);
  std::vector<GNode*> d;
  for (size_t i = 0; i < n; ++i) {
    GNode* p = find_node(static_cast<void* const*>(deps)[i]);
    if (!p || p->graph != g) return record(s, hipErrorInvalidValue);
    d.push_back(p);
  }
  return record(s, begin_capture(s, stream, mode, g, d));
}

// The capture's graph, from the stream it began on: in global and
// thread-local mode, on the thread that began it. Every stream the capture
// took in has to have been waited on again by the end; a capture something
// gave up on hands back no graph.
hipError_t hipStreamEndCapture(hipStream_t stream, void** graph) {
  const ApiCall api("hipStreamEndCapture");
  State& s = state();
  Destructors d;
  hipError_t result = hipSuccess;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    const hipStream_t r = resolve(s, stream);
    Capture* c = capture_of(r);
    if (!c) return record(s, hipErrorIllegalState);
    if (!graph) return record(s, hipErrorInvalidValue);
    if (c->origin != r) return record(s, hipErrorStreamCaptureUnmatched);
    if (c->mode != 2 && c->thread != std::this_thread::get_id()) return record(s, hipErrorStreamCaptureWrongThread);
    bool unjoined = false;
    for (const auto& [h, cs] : c->streams)
      if (h != c->origin && cs.ops && cs.joined_ops != cs.ops) unjoined = true;
    if (c->invalidated || unjoined) {
      result = c->invalidated ? hipErrorStreamCaptureInvalidated : hipErrorStreamCaptureUnjoined;
      if (c->own_graph) destroy_graph(c->graph, &d);
      *graph = nullptr;
    } else {
      *graph = c->graph->handle;
    }
    end_capture(c);
  }
  run_destructors(d);
  return record(s, result);
}

// 0 none, 1 active, 2 invalidated. The null stream is implicated while a
// blocking stream captures.
hipError_t hipStreamIsCapturing(hipStream_t stream, int* status) {
  const ApiCall api("hipStreamIsCapturing");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!status) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (legacy_query(s, stream)) return record(s, hipErrorStreamCaptureImplicit);
  const Capture* c = capture_of(resolve(s, stream));
  *status = !c ? 0 : c->invalidated ? 2 : 1;
  return record(s, hipSuccess);
}
hipError_t hipStreamGetCaptureInfo(hipStream_t stream, int* status, unsigned long long* id) {
  const ApiCall api("hipStreamGetCaptureInfo");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!status) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (legacy_query(s, stream)) return record(s, hipErrorStreamCaptureImplicit);
  const Capture* c = capture_of(resolve(s, stream));
  *status = !c ? 0 : c->invalidated ? 2 : 1;
  if (id) *id = c ? c->id : 0;
  return record(s, hipSuccess);
}
// The same, and the capture's graph as it is so far -- which the program may
// add to -- and what the stream's next operation will follow.
hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, int* status, unsigned long long* id, void** graph,
                                      const void*** deps, size_t* n) {
  const ApiCall api("hipStreamGetCaptureInfo_v2");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!status) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (legacy_query(s, stream)) return record(s, hipErrorStreamCaptureImplicit);
  const hipStream_t r = resolve(s, stream);
  Capture* c = capture_of(r);
  *status = !c ? 0 : c->invalidated ? 2 : 1;
  if (id) *id = c ? c->id : 0;
  if (graph) *graph = c ? c->graph->handle : nullptr;
  if (c) {
    deps_out(c->streams[r], deps, n);
  } else {
    if (deps) *deps = nullptr;
    if (n) *n = 0;
  }
  return record(s, hipSuccess);
}
// What a capturing stream's next operation follows: these nodes as well
// (hipStreamAddCaptureDependencies, 0) or instead (Set, 1).
hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, void** nodes, size_t n, unsigned int flags) {
  const ApiCall api("hipStreamUpdateCaptureDependencies");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if ((n && !nodes) || flags > 1) return record(s, hipErrorInvalidValue);
  const hipStream_t r = resolve(s, stream);
  Capture* c = capture_of(r);
  if (!c) return record(s, hipErrorIllegalState);
  std::vector<GNode*> add;
  for (size_t i = 0; i < n; ++i) {
    GNode* p = find_node(nodes[i]);
    if (!p || p->graph != c->graph) return record(s, hipErrorInvalidValue);
    add.push_back(p);
  }
  CapStream& cs = c->streams[r];
  if (flags == 1) cs.deps.clear();
  for (GNode* p : add)
    if (std::find(cs.deps.begin(), cs.deps.end(), p) == cs.deps.end()) cs.deps.push_back(p);
  return record(s, hipSuccess);
}
hipError_t hipThreadExchangeStreamCaptureMode(int* mode) {
  const ApiCall api("hipThreadExchangeStreamCaptureMode");
  if (!mode || *mode < 0 || *mode > 2) return record(state(), hipErrorInvalidValue);
  std::swap(*mode, t_capture_mode);
  return hipSuccess;
}

// ---- A graph drawn ---------------------------------------------------------------------------------
//
// Graphviz's dot: a cluster for the graph and one for each graph embedded in
// it, each node labelled with its type (and a copy with its direction, from
// where its two sides really are), and an edge for each edge. The flags ask
// for more or less detail; this draws the same for all of them.

}  // extern "C"
namespace {
std::string node_label(State& s, const GNode& n) {
  using namespace vgpu::amd::abi;
  switch (n.type) {
    case kNodeKernel: {
      std::string name = "KERNEL";
      bool module_fn = false;
      const Module* m = nullptr;
      const Kernel* k = nullptr;
      if (kernel_of(s, n.kernel.func, n.kernel.device, &module_fn, &m, &k) == hipSuccess) name += "\\n" + k->name;
      return name;
    }
    case kNodeMemcpy: {
      const auto side = [&](const void* p, bool array) {
        if (array) return 'A';
        return device_memory(s, p) ? 'D' : 'H';
      };
      char from = 'H', to = 'H';
      if (n.copy.form == kCopy3D) {
        from = side(n.copy.p3.srcPtr.ptr, n.copy.p3.srcArray != nullptr);
        to = side(n.copy.p3.dstPtr.ptr, n.copy.p3.dstArray != nullptr);
      } else if (n.copy.form == kCopyDrv) {
        from = n.copy.drv.srcMemoryType == kMemTypeArray ? 'A' : side(n.copy.rsrc, false);
        to = n.copy.drv.dstMemoryType == kMemTypeArray ? 'A' : side(n.copy.rdst, false);
      } else {
        from = side(n.copy.form == kCopyCaptured ? n.copy.rsrc : n.copy.src, false);
        to = side(n.copy.form == kCopyCaptured ? n.copy.rdst : n.copy.dst, false);
        if (n.copy.form == kCopyFromSymbol) from = 'D';
        if (n.copy.form == kCopyToSymbol) to = 'D';
      }
      return std::string("MEMCPY\\n(") + from + "to" + to + ")";
    }
    case kNodeMemset: return "MEMSET";
    case kNodeHost: return "HOST";
    case kNodeGraph: return "CHILD_GRAPH";
    case kNodeEmpty: return "EMPTY";
    case kNodeWaitEvent: return "EVENT_WAIT";
    case kNodeEventRecord: return "EVENT_RECORD";
    case kNodeMemAlloc: return "MEM_ALLOC";
    case kNodeMemFree: return "MEM_FREE";
    case kNodeBatchMemOp: return "BATCH_MEM_OP";
    default: return "NODE";
  }
}
void draw(State& s, const GraphObj& g, std::string& out, int& cluster) {
  out += "subgraph cluster_" + std::to_string(cluster++) + " {\n";
  for (const auto& n : g.nodes) {
    char id[32];
    std::snprintf(id, sizeof id, "n%llx", static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(n->handle)));
    out += std::string("\"") + id + "\" [label=\"" + node_label(s, *n) + "\"];\n";
  }
  for (const auto& n : g.nodes)
    if (n->child) draw(s, *n->child, out, cluster);
  for (const auto& [a, b] : g.edges) {
    char x[32], y[32];
    std::snprintf(x, sizeof x, "n%llx", static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(a->handle)));
    std::snprintf(y, sizeof y, "n%llx", static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(b->handle)));
    out += std::string("\"") + x + "\" -> \"" + y + "\";\n";
  }
  out += "}\n";
}
}  // namespace
extern "C" {

hipError_t hipGraphDebugDotPrint(void* graph, const char* path, unsigned int) {
  const ApiCall api("hipGraphDebugDotPrint");
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  GraphObj* g = find_graph(graph);
  if (!g || !path) return record(s, hipErrorInvalidValue);
  std::string out = "digraph dot {\n";
  int cluster = 0;
  draw(s, *g, out, cluster);
  out += "}\n";
  std::FILE* f = std::fopen(path, "w");
  if (!f) return record(s, hipErrorOperatingSystem);
  const bool ok = std::fwrite(out.data(), 1, out.size(), f) == out.size();
  std::fclose(f);
  return record(s, ok ? hipSuccess : hipErrorOperatingSystem);
}

}  // extern "C"
