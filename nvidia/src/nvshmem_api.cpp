// libvgpunvshmem_host -- VirtualGPU's NVSHMEM host library, presented as
// libnvshmem_host.so.3.
//
// NVSHMEM gives every GPU of a job a symmetric heap, and lets any of them read
// and write the others'. On hardware the heaps are mapped peer to peer over
// NVLink or PCIe (or reached through a NIC). Here each PE is a process, as
// NVSHMEM requires, with a simulated GPU of its own; its heap is device
// memory the simulator backs with a shared file (the CUDA IPC mechanism), and
// every PE maps every other PE's heap into its own device address space. So a
// kernel's store to a peer's heap is a store to shared memory, which the peer
// sees, and the job is the single-node, all-peer-to-peer case of NVSHMEM.
//
// The PEs find each other through a rendezvous file named by the job's unique
// ID (nvshmemx_get_uniqueid, NVSHMEMX_INIT_WITH_UNIQUEID: the documented
// bootstrap that needs no MPI), or, for scripts, through VGPU_NVSHMEM_RANK,
// VGPU_NVSHMEM_NPES and VGPU_NVSHMEM_ID. With neither, nvshmem_init() makes a
// job of one PE, as NVIDIA's library does when no launcher set it up.
//
// Device code is NVIDIA's: the nvshmem_* calls a kernel makes are inline
// functions in NVIDIA's public headers, and they read one struct,
// nvshmemi_device_state_d, that NVIDIA's device library (libnvshmem_device.a)
// defines. Its init code hands this library the struct's address
// (nvshmemid_hostlib_init_attr's callback) and this library fills it in --
// heap bases, peers, teams, the collectives' synchronization arrays -- as the
// public headers lay it out (nvshmem_abi.hpp). With every peer reachable by
// load and store, the inline code never leaves the kernel: puts and gets are
// stores and loads, quiet and fence are fences, barriers and signals are
// stores and polling.
//
// MPI, OpenSHMEM and PMIx also give the job (the application's own MPI or
// OpenSHMEM library, or libpmix, see "bootstraps through a launcher" below).
//
// Not here: the PMI-1 and PMI-2 bootstraps through libpmi, PEs on more than one node, NVLink SHARP multicast (nvshmemx_mc_ptr is
// NULL, as on hardware without it), and host-side reductions. Device-side
// atomics between PEs are atomic within a process but not across the PE
// processes; nvidia/docs/libraries.md says so.
#include "../include/vgpu_nvshmem.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__has_include)
#if __has_include(<pmix.h>)
#include <pmix.h>
#define VGPU_NVSHMEM_HAVE_PMIX 1
#endif
#endif
#include <strings.h>

#include "nvshmem_abi.hpp"
#include "vgpu/runtime/capture.hpp"
#include "vgpu/runtime/shim_memory.hpp"

namespace abi = vgpu::nvshmem_abi;

namespace {

constexpr int kMaxPes = 64;
constexpr int kMaxTeamsCap = 256;
constexpr uint32_t kSegMagic = 0x4d485356u;  // "VSHM"
constexpr uint32_t kSegVersion = 1;
constexpr uint32_t kUidMagic = 0x44495556u;  // "VUID"

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

void say(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void say(const char* fmt, ...) {
  if (quiet()) return;
  va_list ap;
  va_start(ap, fmt);
  std::fprintf(stderr, "[vgpu] nvshmem: ");
  std::vfprintf(stderr, fmt, ap);
  std::fprintf(stderr, "\n");
  va_end(ap);
}

double timeout_seconds() {
  if (const char* t = std::getenv("VGPU_NVSHMEM_TIMEOUT")) {
    const double v = std::atof(t);
    if (v > 0) return v;
  }
  return 300.0;
}

size_t env_size(const char* name, size_t dflt) {
  const char* v = std::getenv(name);
  if (!v || !*v) return dflt;
  char* end = nullptr;
  double x = std::strtod(v, &end);
  if (end && *end) {
    switch (*end) {
      case 'k': case 'K': x *= 1024.0; break;
      case 'm': case 'M': x *= 1024.0 * 1024; break;
      case 'g': case 'G': x *= 1024.0 * 1024 * 1024; break;
      case 't': case 'T': x *= 1024.0 * 1024 * 1024 * 1024; break;
      default: break;
    }
  }
  return x > 0 ? size_t(x) : dflt;
}

// ---- the rendezvous file -------------------------------------------------
//
// Mapped MAP_SHARED by every PE of the job: what each PE publishes about its
// heap, and the counters the host-side barriers and team splits use. All
// fields a peer reads are lock-free atomics, since the PEs are processes.
struct Segment {
  std::atomic<uint32_t> magic;
  uint32_t version;
  int32_t npes;
  int32_t pad0;
  std::atomic<uint32_t> attached;
  std::atomic<uint32_t> finished;
  std::atomic<uint32_t> lock;  // host-side atomic updates of symmetric memory
  uint32_t pad1;
  // A barrier per team index: arrivals, and the generation that releases them.
  std::atomic<uint32_t> arrived[kMaxTeamsCap];
  std::atomic<uint64_t> generation[kMaxTeamsCap];
  // Team indices in use, job-wide.
  std::atomic<uint64_t> team_bits[kMaxTeamsCap / 64];
  // The result of the latest split of each parent team.
  std::atomic<uint64_t> split_seq[kMaxTeamsCap];
  std::atomic<int32_t> split_index[kMaxTeamsCap];
  // nvshmemx_team_init: the teams being made from a unique ID, by that ID.
  struct UidSlot {
    std::atomic<uint64_t> uid;       // 0: free
    std::atomic<int32_t> index;      // the team index plus one, once the first member took it
    std::atomic<uint32_t> arrived;   // members that have called
    std::atomic<uint32_t> departed;  // members that have left the call
    std::atomic<int32_t> member[kMaxPes];  // world PE plus one, by index in the team
  } uid_slot[16];
  struct Pe {
    std::atomic<uint32_t> ready;
    int32_t pid;
    int32_t device;
    uint32_t pad;
    uint64_t heap_size;
    unsigned char handle[64];  // cudaIpcMemHandle_t
  } pe[kMaxPes];
};
static_assert(std::atomic<uint64_t>::is_always_lock_free && std::atomic<uint32_t>::is_always_lock_free,
              "the rendezvous needs address-free atomics");

std::string shm_dir() {
  if (const char* d = std::getenv("VGPU_NVSHMEM_DIR")) return d;
  struct stat sb {};
  if (::stat("/dev/shm", &sb) == 0 && S_ISDIR(sb.st_mode) && ::access("/dev/shm", W_OK) == 0) return "/dev/shm";
  const char* t = std::getenv("TMPDIR");
  return t && *t ? t : "/tmp";
}

// ---- library state ---------------------------------------------------------

struct HostTeam {
  bool valid = false;
  int start = 0, stride = 1, size = 0;  // in world PEs
  int my_pe = -1;
  std::vector<int> members;  // world PE of each team PE
  uint64_t splits = 0;       // how many splits of this team so far
  nvshmem_team_config_t config{};
  abi::Team* dev = nullptr;
  int* dev_mapping = nullptr;
};

struct State {
  std::recursive_mutex mu;
  int status = NVSHMEM_STATUS_NOT_INITIALIZED;
  int mype = 0, npes = 1, device = 0;
  int provided = NVSHMEM_THREAD_SERIALIZED;
  std::string seg_path;
  Segment* seg = nullptr;
  // The symmetric heap: [psync pool][user heap]. heap_base is what the device
  // code calls heap_base; every offset into it is the same on every PE.
  char* heap = nullptr;
  size_t heap_size = 0;
  size_t user_offset = 0;
  std::vector<char*> peer;  // each PE's heap, as this process addresses it
  // Free blocks of the user heap, by offset: first fit, coalescing.
  std::map<size_t, size_t> free_blocks;
  std::map<size_t, size_t> used_blocks;
  int max_teams = 32;
  std::vector<HostTeam> teams;
  abi::CollParams coll{};
  size_t psync_len = 0;  // longs per team
  std::vector<bool> psync_used;  // whether a team at this index has used its arrays
  // Device-side state.
  abi::DeviceState ds{};
  void** dev_peers = nullptr;
  void** dev_remote = nullptr;
  abi::Team** dev_team_pool = nullptr;
  long* dev_sync_counter = nullptr;
  abi::Timeout* dev_timeout = nullptr;
  unsigned long long* dev_wait_any = nullptr;
  int* dev_exit = nullptr;  // request state, then code
  std::vector<void*> state_targets;  // device-library copies of nvshmemi_device_state_d
  bool bootstrapped_by_launcher = false;  // MPI, OpenSHMEM or PMI(x): the status after nvshmem_finalize
  void (*close_bootstrap)() = nullptr;    // what finalize undoes of it (PMIx_Finalize)
};

State& st() {
  static State* s = new State;
  return *s;
}

bool initialized(State& s) { return s.status >= NVSHMEM_STATUS_IS_INITIALIZED; }


void cuda_ok(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return;
  std::fprintf(stderr, "[vgpu] nvshmem: %s failed: %s\n", what, cudaGetErrorString(e));
  cudaGetLastError();
}

// Waits for `done`, polling, giving up after the job's timeout.
void wait_for(const std::function<bool()>& done, const char* what) {
  const auto start = std::chrono::steady_clock::now();
  const double limit = timeout_seconds();
  for (unsigned spin = 0; !done(); ++spin) {
    if (spin < 64) {
      std::this_thread::yield();
      continue;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(spin < 4096 ? 20 : 200));
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > limit) {
      std::fprintf(stderr,
                   "[vgpu] nvshmem: PE %d waited %.0f s for %s; a PE of the job is missing or stuck "
                   "(VGPU_NVSHMEM_TIMEOUT sets the limit)\n",
                   st().mype, limit, what);
      std::abort();
    }
  }
}

// The host-side barrier of one team: arrivals counted, the last releases.
void team_barrier(State& s, int idx) {
  const HostTeam& t = s.teams[idx];
  if (t.size <= 1) return;
  std::atomic<uint32_t>& arrived = s.seg->arrived[idx];
  std::atomic<uint64_t>& gen = s.seg->generation[idx];
  const uint64_t g = gen.load(std::memory_order_acquire);
  if (arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == uint32_t(t.size)) {
    arrived.store(0, std::memory_order_relaxed);
    gen.fetch_add(1, std::memory_order_acq_rel);
    return;
  }
  wait_for([&] { return gen.load(std::memory_order_acquire) != g; }, "a barrier");
}

void seg_lock(State& s) {
  wait_for([&] {
    uint32_t z = 0;
    return s.seg->lock.compare_exchange_weak(z, 1, std::memory_order_acquire);
  }, "the job's atomic lock");
}
void seg_unlock(State& s) { s.seg->lock.store(0, std::memory_order_release); }

// ---- symmetric addresses -------------------------------------------------------

bool symmetric(State& s, const void* p, size_t n = 1) {
  const char* c = static_cast<const char*>(p);
  return s.heap && c >= s.heap && c < s.heap + s.heap_size && n <= size_t(s.heap + s.heap_size - c);
}

// The address of `p`'s copy on PE `pe`, as this process addresses it.
char* remote(State& s, const void* p, int pe) {
  return s.peer[pe] + (static_cast<const char*>(p) - s.heap);
}

bool check_pe(State& s, int pe, const char* api) {
  if (pe >= 0 && pe < s.npes) return true;
  say("%s: PE %d is not in the job (0..%d)", api, pe, s.npes - 1);
  return false;
}

bool copy(void* dst, const void* src, size_t n, const char* api) {
  if (n == 0) return true;
  const cudaError_t e = cudaMemcpy(dst, src, n, cudaMemcpyDefault);
  if (e != cudaSuccess) {
    cudaGetLastError();
    say("%s: the copy failed (%s)", api, cudaGetErrorString(e));
    return false;
  }
  return true;
}

void put(const void* dest, const void* source, size_t bytes, int pe, const char* api) {
  State& s = st();
  if (!initialized(s)) return say("%s before nvshmem_init", api);
  if (!check_pe(s, pe, api)) return;
  if (!symmetric(s, dest, bytes)) return say("%s: the destination is not symmetric memory", api);
  copy(remote(s, dest, pe), source, bytes, api);
}

void get(void* dest, const void* source, size_t bytes, int pe, const char* api) {
  State& s = st();
  if (!initialized(s)) return say("%s before nvshmem_init", api);
  if (!check_pe(s, pe, api)) return;
  if (!symmetric(s, source, bytes)) return say("%s: the source is not symmetric memory", api);
  copy(dest, remote(s, source, pe), bytes, api);
}

// Strided: element i goes from source + i*sst to dest + i*dst (in elements).
void iput(void* dest, const void* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, size_t es, int pe,
          bool is_put) {
  for (size_t i = 0; i < nelems; ++i) {
    char* d = static_cast<char*>(dest) + ptrdiff_t(i) * dst * ptrdiff_t(es);
    const char* sp = static_cast<const char*>(source) + ptrdiff_t(i) * sst * ptrdiff_t(es);
    if (is_put) put(d, sp, es, pe, "nvshmem_iput");
    else get(d, sp, es, pe, "nvshmem_iget");
  }
}

// Runs `work` in stream order: as a host node while the stream is captured,
// otherwise once the work queued before it has finished.
void in_stream_order(cudaStream_t stream, std::function<void()> work) {
  if (vgpu_record_host_op_if_capturing(stream, work)) return;
  cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  work();
}

void signal_op(uint64_t* sig, uint64_t value, int op, int pe, const char* api) {
  State& s = st();
  if (!check_pe(s, pe, api)) return;
  if (!symmetric(s, sig, sizeof(uint64_t))) return say("%s: the signal is not symmetric memory", api);
  uint64_t* where = reinterpret_cast<uint64_t*>(remote(s, sig, pe));
  if (op == NVSHMEM_SIGNAL_SET) {
    copy(where, &value, sizeof value, api);
  } else if (op == NVSHMEM_SIGNAL_ADD) {
    // Atomic against the other PEs' host-side signal operations.
    seg_lock(s);
    uint64_t v = 0;
    if (copy(&v, where, sizeof v, api)) {
      v += value;
      copy(where, &v, sizeof v, api);
    }
    seg_unlock(s);
  } else {
    say("%s: signal operation %d is neither NVSHMEM_SIGNAL_SET nor NVSHMEM_SIGNAL_ADD", api, op);
  }
}

bool compare(uint64_t a, int cmp, uint64_t b) {
  switch (cmp) {
    case NVSHMEM_CMP_EQ: return a == b;
    case NVSHMEM_CMP_NE: return a != b;
    case NVSHMEM_CMP_GT: return a > b;
    case NVSHMEM_CMP_LE: return a <= b;
    case NVSHMEM_CMP_LT: return a < b;
    case NVSHMEM_CMP_GE: return a >= b;
  }
  return false;
}

// ---- the symmetric allocator -------------------------------------------------
//
// Every PE makes the same calls in the same order (the allocation routines are
// collective), so first fit gives every PE the same offsets.

constexpr size_t kMinAlign = 512;

void* heap_alloc(State& s, size_t alignment, size_t size) {
  if (size == 0) return nullptr;
  alignment = std::max(alignment, kMinAlign);
  size = (size + kMinAlign - 1) / kMinAlign * kMinAlign;
  for (auto it = s.free_blocks.begin(); it != s.free_blocks.end(); ++it) {
    const size_t off = it->first, len = it->second;
    const size_t aligned = (off + alignment - 1) / alignment * alignment;
    if (aligned + size > off + len) continue;
    s.free_blocks.erase(it);
    if (aligned > off) s.free_blocks[off] = aligned - off;
    if (aligned + size < off + len) s.free_blocks[aligned + size] = off + len - aligned - size;
    s.used_blocks[aligned] = size;
    return s.heap + aligned;
  }
  return nullptr;
}

void heap_free(State& s, void* p) {
  const size_t off = size_t(static_cast<char*>(p) - s.heap);
  auto it = s.used_blocks.find(off);
  if (it == s.used_blocks.end()) return say("nvshmem_free: %p was not returned by nvshmem_malloc", p);
  size_t start = off, len = it->second;
  s.used_blocks.erase(it);
  auto next = s.free_blocks.lower_bound(start);
  if (next != s.free_blocks.end() && next->first == start + len) {
    len += next->second;
    next = s.free_blocks.erase(next);
  }
  if (next != s.free_blocks.begin()) {
    auto prev = std::prev(next);
    if (prev->first + prev->second == start) {
      start = prev->first;
      len += prev->second;
      s.free_blocks.erase(prev);
    }
  }
  s.free_blocks[start] = len;
}

// ---- teams --------------------------------------------------------------------

long* team_psync(State& s, int idx) {
  return reinterpret_cast<long*>(s.heap) + size_t(idx) * s.psync_len;
}

// Builds team `idx` on this PE: its host record, and the struct the device
// code reads through team_pool[idx], with its PE mapping.
void make_team(State& s, int idx, const std::vector<int>& members, int start, int stride) {
  HostTeam& t = s.teams[idx];
  t = HostTeam{};
  t.valid = true;
  t.members = members;
  t.size = int(members.size());
  t.start = start;
  t.stride = stride;
  t.my_pe = -1;
  for (int i = 0; i < t.size; ++i)
    if (members[i] == s.mype) t.my_pe = i;
  t.config = nvshmem_team_config_t{(2 << 16) + int(sizeof(nvshmem_team_config_t)), 1, ~0ull, {0}};
  // Fresh synchronization: the team's arrays zeroed, its counter at 1 (the
  // device barrier writes the counter into a peer's slot and waits for its
  // own to reach it). A new heap reads as zeros already, and clearing it
  // anyway would make every page of its sparse backing real.
  if (s.psync_used[size_t(idx)])
    cuda_ok(cudaMemset(team_psync(s, idx), 0, s.psync_len * sizeof(long)), "cudaMemset");
  s.psync_used[size_t(idx)] = true;
  long one[2] = {1, 1};
  cuda_ok(cudaMemcpy(s.dev_sync_counter + 2 * idx, one, sizeof one, cudaMemcpyHostToDevice), "cudaMemcpy");
  // The mapping: team PE -> world PE, then world PE -> team PE (-1 outside).
  std::vector<int> mapping(size_t(t.size) + size_t(s.npes), -1);
  for (int i = 0; i < t.size; ++i) {
    mapping[i] = members[i];
    mapping[size_t(t.size) + size_t(members[i])] = i;
  }
  cuda_ok(cudaMalloc(reinterpret_cast<void**>(&t.dev_mapping), mapping.size() * sizeof(int)), "cudaMalloc");
  cuda_ok(cudaMemcpy(t.dev_mapping, mapping.data(), mapping.size() * sizeof(int), cudaMemcpyHostToDevice),
          "cudaMemcpy");
  abi::Team d{};
  d.version = (4 << 16) + int(sizeof(abi::Team));
  d.my_pe = t.my_pe;
  d.start = start;
  d.stride = stride;
  d.size = t.size;
  d.team_idx = idx;
  d.config.version = t.config.version;
  d.config.num_contexts = t.config.num_contexts;
  d.config.uniqueid = t.config.uniqueid;
  d.config_mask = 0;
  d.reduce_recexch.version = (1 << 16) + int(sizeof(abi::ReduceRecexch));
  d.reduce_recexch.step1_sendto = -1;
  d.reduce_recexch.step1_nrecvs = -1;
  d.reduce_recexch.step2_nphases = -1;
  d.ll_flag = 1;
  d.fcollect_ll_flag = 1;
  d.are_gpus_p2p_connected = true;
  d.is_team_node = t.size == s.npes;  // one node: a team of every PE is the node team
  d.team_node = d.is_team_node ? NVSHMEMX_TEAM_NODE : NVSHMEM_TEAM_INVALID;
  d.is_team_same_mype_node = false;
  d.team_same_mype_node = NVSHMEM_TEAM_INVALID;
  for (int& x : d.team_dups) x = NVSHMEM_TEAM_INVALID;
  d.pe_mapping = t.dev_mapping;
  cuda_ok(cudaMalloc(reinterpret_cast<void**>(&t.dev), sizeof d), "cudaMalloc");
  cuda_ok(cudaMemcpy(t.dev, &d, sizeof d, cudaMemcpyHostToDevice), "cudaMemcpy");
  cuda_ok(cudaMemcpy(s.dev_team_pool + idx, &t.dev, sizeof t.dev, cudaMemcpyHostToDevice), "cudaMemcpy");
}

void drop_team(State& s, int idx) {
  HostTeam& t = s.teams[idx];
  if (!t.valid) return;
  abi::Team* none = nullptr;
  cuda_ok(cudaMemcpy(s.dev_team_pool + idx, &none, sizeof none, cudaMemcpyHostToDevice), "cudaMemcpy");
  cudaFree(t.dev);
  cudaFree(t.dev_mapping);
  t = HostTeam{};
}

bool take_team_index(State& s, int* out) {
  for (int i = NVSHMEM_TEAMS_MIN; i < s.max_teams; ++i) {
    std::atomic<uint64_t>& w = s.seg->team_bits[i / 64];
    const uint64_t bit = 1ull << (i % 64);
    uint64_t old = w.load();
    while (!(old & bit)) {
      if (w.compare_exchange_weak(old, old | bit)) {
        *out = i;
        return true;
      }
    }
  }
  return false;
}

void release_team_index(State& s, int idx) { s.seg->team_bits[idx / 64].fetch_and(~(1ull << (idx % 64))); }

// ---- the device-side state ------------------------------------------------------

void build_device_state(State& s) {
  abi::DeviceState& d = s.ds;
  d = abi::DeviceState{};
  d.version = (1 << 16) + int(sizeof(abi::DeviceState));
  d.mype = s.mype;
  d.npes = s.npes;
  d.node_mype = s.mype;
  d.node_npes = s.npes;
  d.pe_dist = 1;  // NVSHMEMI_PE_DIST_BLOCK: one node holds them all
  d.proxy = 0;    // NVSHMEMI_PROXY_NONE
  d.atomics_sync = 0;
  // Every PE reaches every other by load and store, atomics included.
  d.job_connectivity = abi::kJobGpuLdstAtomics;
  d.proxy_ops_are_ordered = true;
  d.atomics_complete_on_quiet = true;
  d.heap_base = s.heap;
  d.heap_size = s.heap_size;
  d.peer_heap_base_p2p = s.dev_peers;
  d.peer_heap_base_remote = s.dev_remote;
  d.symmetric_heap_kind = false;  // device memory
  d.timeout = s.dev_timeout;
  d.test_wait_any_start_idx_ptr = s.dev_wait_any;
  d.team_pool = s.dev_team_pool;
  d.psync_pool = reinterpret_cast<long*>(s.heap);
  d.sync_counter = s.dev_sync_counter;
  d.gpu_coll_env_params_var = s.coll;
  d.global_exit_request_state = s.dev_exit;
  d.global_exit_code = s.dev_exit + 1;
  d.nvshmemi_is_nvshmem_initialized = true;
  d.nvshmemi_is_nvshmem_bootstrapped = true;
  d.selected_device_transport = 0;  // NVSHMEMI_DEVICE_TRANSPORT_TYPE_PROXY; never used, all peers are P2P
}

void write_device_state(State& s, void* target) {
  if (!target) return;
  cuda_ok(cudaMemcpy(target, &s.ds, sizeof s.ds, cudaMemcpyHostToDevice), "writing nvshmemi_device_state_d");
}

// ---- initialization ---------------------------------------------------------------

struct Bootstrap {
  int rank = 0, nranks = 1;
  std::string id;
  bool launcher = false;  // MPI, OpenSHMEM or PMI(x): not a unique ID
  void (*close)() = nullptr;
};

std::string random_token() {
  uint64_t r[2] = {0, 0};
  if (::getrandom(r, sizeof r, 0) != sizeof r) {
    r[0] = uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    r[1] = uint64_t(::getpid());
  }
  char buf[64];
  std::snprintf(buf, sizeof buf, "%016llx%08x", (unsigned long long)(r[0] ^ r[1]), unsigned(::getpid()));
  return buf;
}

struct UidPayload {
  uint32_t magic;
  char token[48];
};

// ---- bootstraps through a launcher's library ---------------------------------
//
// NVSHMEM's MPI, OpenSHMEM and PMIx bootstraps are plugins that call those
// libraries. The PEs here need only their rank, the size of the job and one
// token to name the rendezvous file, so the calls are made through the
// application's own MPI or OpenSHMEM (found by name in the process, as the
// plugins link against it), or the system's libpmix. Measured against NVIDIA's
// NVSHMEM 3.8 on two RTX 3060s: NVSHMEMX_INIT_WITH_MPI_COMM and
// NVSHMEM_BOOTSTRAP=MPI take the job from a communicator (MPI_COMM_WORLD for the
// variable), likewise NVSHMEMX_INIT_WITH_SHMEM and =SHMEM from the OpenSHMEM job,
// NVSHMEM_BOOTSTRAP=PMI with NVSHMEM_BOOTSTRAP_PMI=PMIX from PMIx; PMI and PMI-2
// are libpmi.so and libpmi2.so, which a machine without them answers with a job of
// one PE; the variable also names a plugin library (=plugin with
// NVSHMEM_BOOTSTRAP_PLUGIN) and the init flags say which bootstrap a call uses.

constexpr size_t kTokenBytes = 64;

// MPI: Open MPI's handles are the addresses of its predefined objects; the
// MPICH family's are integer constants.
struct MpiApi {
  int (*initialized)(int*) = nullptr;
  int (*rank)(uintptr_t, int*) = nullptr;
  int (*size)(uintptr_t, int*) = nullptr;
  int (*bcast)(void*, int, uintptr_t, int, uintptr_t) = nullptr;
  uintptr_t byte = 0, world = 0;
  bool wide = false;  // 8-byte handles (Open MPI)
};

bool load_mpi(MpiApi& m) {
  m.initialized = reinterpret_cast<int (*)(int*)>(dlsym(RTLD_DEFAULT, "MPI_Initialized"));
  m.rank = reinterpret_cast<int (*)(uintptr_t, int*)>(dlsym(RTLD_DEFAULT, "MPI_Comm_rank"));
  m.size = reinterpret_cast<int (*)(uintptr_t, int*)>(dlsym(RTLD_DEFAULT, "MPI_Comm_size"));
  m.bcast = reinterpret_cast<int (*)(void*, int, uintptr_t, int, uintptr_t)>(dlsym(RTLD_DEFAULT, "MPI_Bcast"));
  if (!m.initialized || !m.rank || !m.size || !m.bcast) return false;
  if (void* w = dlsym(RTLD_DEFAULT, "ompi_mpi_comm_world")) {
    m.world = reinterpret_cast<uintptr_t>(w);
    m.byte = reinterpret_cast<uintptr_t>(dlsym(RTLD_DEFAULT, "ompi_mpi_byte"));
    m.wide = true;
    return m.byte != 0;
  }
  m.world = 0x44000000u;  // MPI_COMM_WORLD of MPICH and its relatives
  m.byte = 0x4c00010du;   // MPI_BYTE
  return true;
}

// A communicator, however it is in memory: `comm` points to an MPI_Comm.
int mpi_bootstrap(const void* comm_ptr, bool world, Bootstrap* b, const char* why) {
  MpiApi m;
  if (!load_mpi(m)) {
    say("%s needs the MPI library the program is linked with, and none is loaded in this process", why);
    return 1;
  }
  int up = 0;
  m.initialized(&up);
  if (!up) {
    say("%s: MPI is not initialized; call MPI_Init first", why);
    return 1;
  }
  uintptr_t comm = m.world;  // no communicator in the attributes (measured): MPI_COMM_WORLD
  if (!world && comm_ptr) {
    if (m.wide) {
      std::memcpy(&comm, comm_ptr, sizeof comm);
    } else {
      int c = 0;
      std::memcpy(&c, comm_ptr, sizeof c);
      comm = uintptr_t(unsigned(c));
    }
  }
  int rank = 0, size = 1;
  if (m.rank(comm, &rank) || m.size(comm, &size) || size < 1 || size > kMaxPes) {
    say("the MPI communicator is not usable (at most %d PEs)", kMaxPes);
    return 1;
  }
  char token[kTokenBytes] = {0};
  if (rank == 0) std::snprintf(token, sizeof token, "%s", random_token().c_str());
  if (m.bcast(token, int(sizeof token), m.byte, 0, comm)) {
    say("MPI_Bcast failed in the MPI bootstrap");
    return 1;
  }
  b->rank = rank;
  b->nranks = size;
  b->id = token;
  b->launcher = true;
  return 0;
}

// OpenSHMEM: the token travels in a symmetric buffer.
int shmem_bootstrap(Bootstrap* b) {
  auto my_pe = reinterpret_cast<int (*)()>(dlsym(RTLD_DEFAULT, "shmem_my_pe"));
  auto n_pes = reinterpret_cast<int (*)()>(dlsym(RTLD_DEFAULT, "shmem_n_pes"));
  auto malloc_ = reinterpret_cast<void* (*)(size_t)>(dlsym(RTLD_DEFAULT, "shmem_malloc"));
  auto free_ = reinterpret_cast<void (*)(void*)>(dlsym(RTLD_DEFAULT, "shmem_free"));
  auto getmem = reinterpret_cast<void (*)(void*, const void*, size_t, int)>(dlsym(RTLD_DEFAULT, "shmem_getmem"));
  auto barrier = reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "shmem_barrier_all"));
  if (!my_pe || !n_pes || !malloc_ || !free_ || !getmem || !barrier) {
    say("the OpenSHMEM bootstrap needs the OpenSHMEM library the program is linked with, and none is loaded in this process");
    return 1;
  }
  const int rank = my_pe(), size = n_pes();
  if (size < 1 || size > kMaxPes) {
    say("the OpenSHMEM job has %d PEs (at most %d)", size, kMaxPes);
    return 1;
  }
  char* buf = static_cast<char*>(malloc_(kTokenBytes));
  if (!buf) {
    say("shmem_malloc failed in the OpenSHMEM bootstrap");
    return 1;
  }
  std::memset(buf, 0, kTokenBytes);
  if (rank == 0) std::snprintf(buf, kTokenBytes, "%s", random_token().c_str());
  barrier();
  if (rank != 0) getmem(buf, buf, kTokenBytes, 0);
  barrier();
  b->id.assign(buf, strnlen(buf, kTokenBytes));
  free_(buf);
  b->rank = rank;
  b->nranks = size;
  b->launcher = true;
  return 0;
}

#ifdef VGPU_NVSHMEM_HAVE_PMIX
struct PmixApi {
  pmix_status_t (*init)(pmix_proc_t*, pmix_info_t[], size_t) = nullptr;
  pmix_status_t (*fin)(const pmix_info_t[], size_t) = nullptr;
  pmix_status_t (*put)(pmix_scope_t, const pmix_key_t, pmix_value_t*) = nullptr;
  pmix_status_t (*commit)() = nullptr;
  pmix_status_t (*fence)(const pmix_proc_t[], size_t, const pmix_info_t[], size_t) = nullptr;
  pmix_status_t (*get)(const pmix_proc_t*, const pmix_key_t, const pmix_info_t[], size_t, pmix_value_t**) = nullptr;
  void (*value_free)(pmix_value_t*, size_t) = nullptr;
};
PmixApi g_pmix;
bool g_pmix_up = false;

void pmix_close() {
  if (g_pmix_up && g_pmix.fin) g_pmix.fin(nullptr, 0);
  g_pmix_up = false;
}

int pmix_bootstrap(Bootstrap* b) {
  static void* lib = nullptr;
  if (!lib) {
    for (const char* name : {"libpmix.so.2", "libpmix.so"})
      if ((lib = dlopen(name, RTLD_NOW | RTLD_GLOBAL))) break;
  }
  if (!lib) {
    say("the PMIx bootstrap needs libpmix.so.2");
    return 1;
  }
  auto sym = [&](const char* n) { return dlsym(lib, n); };
  g_pmix.init = reinterpret_cast<decltype(g_pmix.init)>(sym("PMIx_Init"));
  g_pmix.fin = reinterpret_cast<decltype(g_pmix.fin)>(sym("PMIx_Finalize"));
  g_pmix.put = reinterpret_cast<decltype(g_pmix.put)>(sym("PMIx_Put"));
  g_pmix.commit = reinterpret_cast<decltype(g_pmix.commit)>(sym("PMIx_Commit"));
  g_pmix.fence = reinterpret_cast<decltype(g_pmix.fence)>(sym("PMIx_Fence"));
  g_pmix.get = reinterpret_cast<decltype(g_pmix.get)>(sym("PMIx_Get"));
  g_pmix.value_free = reinterpret_cast<decltype(g_pmix.value_free)>(sym("PMIx_Value_free"));
  if (!g_pmix.init || !g_pmix.fin || !g_pmix.put || !g_pmix.commit || !g_pmix.fence || !g_pmix.get) {
    say("libpmix.so.2 lacks the PMIx client calls");
    return 1;
  }
  pmix_proc_t me;
  std::memset(&me, 0, sizeof me);
  if (g_pmix.init(&me, nullptr, 0) != PMIX_SUCCESS) {
    say("PMIx_Init failed: this process was not started by a PMIx launcher (mpirun, srun, prterun)");
    return 1;
  }
  g_pmix_up = true;
  pmix_proc_t wild, root;
  std::memset(&wild, 0, sizeof wild);
  std::memcpy(wild.nspace, me.nspace, sizeof wild.nspace);
  wild.rank = PMIX_RANK_WILDCARD;
  root = wild;
  root.rank = 0;
  pmix_value_t* v = nullptr;
  uint32_t size = 0;
  if (g_pmix.get(&wild, PMIX_JOB_SIZE, nullptr, 0, &v) != PMIX_SUCCESS || !v) {
    say("PMIx cannot tell the size of the job");
    pmix_close();
    return 1;
  }
  size = v->data.uint32;
  if (g_pmix.value_free) g_pmix.value_free(v, 1);
  else std::free(v);
  if (size < 1 || size > unsigned(kMaxPes)) {
    say("the PMIx job has %u PEs (at most %d)", size, kMaxPes);
    pmix_close();
    return 1;
  }
  const char* key = "vgpu.nvshmem.token";
  std::string token = random_token();
  if (me.rank == 0) {
    pmix_value_t val;
    std::memset(&val, 0, sizeof val);
    val.type = PMIX_STRING;
    val.data.string = const_cast<char*>(token.c_str());
    if (g_pmix.put(PMIX_GLOBAL, key, &val) != PMIX_SUCCESS || g_pmix.commit() != PMIX_SUCCESS) {
      say("PMIx_Put failed in the PMIx bootstrap");
      pmix_close();
      return 1;
    }
  }
  pmix_info_t collect;
  std::memset(&collect, 0, sizeof collect);
  std::snprintf(collect.key, sizeof collect.key, "%s", PMIX_COLLECT_DATA);
  collect.value.type = PMIX_BOOL;
  collect.value.data.flag = true;
  if (g_pmix.fence(&wild, 1, &collect, 1) != PMIX_SUCCESS) {
    say("PMIx_Fence failed in the PMIx bootstrap");
    pmix_close();
    return 1;
  }
  if (me.rank != 0) {
    pmix_value_t* t = nullptr;
    if (g_pmix.get(&root, key, nullptr, 0, &t) != PMIX_SUCCESS || !t || t->type != PMIX_STRING || !t->data.string) {
      say("PMIx_Get of the job's token failed in the PMIx bootstrap");
      pmix_close();
      return 1;
    }
    token = t->data.string;
    if (g_pmix.value_free) g_pmix.value_free(t, 1);
  }
  b->rank = int(me.rank);
  b->nranks = int(size);
  b->id = token;
  b->launcher = true;
  b->close = pmix_close;
  return 0;
}
#else
int pmix_bootstrap(Bootstrap*) {
  say("this build of VirtualGPU's NVSHMEM has no PMIx support (pmix.h was not found at build time)");
  return 1;
}
#endif

// NVSHMEM_BOOTSTRAP and what it names (the documented values), for a call without init flags.
int env_bootstrap(Bootstrap* b) {
  const char* mode = std::getenv("NVSHMEM_BOOTSTRAP");
  std::string name = mode && *mode ? mode : "PMI";
  if (!strcasecmp(name.c_str(), "plugin")) {
    const char* plugin = std::getenv("NVSHMEM_BOOTSTRAP_PLUGIN");
    if (!plugin || !*plugin) {
      say("Plugin bootstrap requires NVSHMEM_BOOTSTRAP_PLUGIN to be set");
      return 1;
    }
    std::string file = plugin;
    const size_t slash = file.rfind('/');
    if (slash != std::string::npos) file = file.substr(slash + 1);
    // NVSHMEM's own plugins by their file names; the plugin interface itself is NVIDIA's and not implemented.
    if (file.rfind("nvshmem_bootstrap_mpi.so", 0) == 0) name = "MPI";
    else if (file.rfind("nvshmem_bootstrap_shmem.so", 0) == 0) name = "SHMEM";
    else if (file.rfind("nvshmem_bootstrap_pmix.so", 0) == 0) { name = "PMI"; setenv("NVSHMEM_BOOTSTRAP_PMI", "PMIX", 1); }
    else if (file.rfind("nvshmem_bootstrap_pmi2.so", 0) == 0) { name = "PMI"; setenv("NVSHMEM_BOOTSTRAP_PMI", "PMI-2", 1); }
    else if (file.rfind("nvshmem_bootstrap_pmi.so", 0) == 0) { name = "PMI"; setenv("NVSHMEM_BOOTSTRAP_PMI", "PMI", 1); }
    else if (::access(plugin, R_OK) != 0 && file.find('/') == std::string::npos && file.rfind("nvshmem_bootstrap_", 0) != 0) {
      say("Bootstrap library dlopen failed for %s", plugin);
      return 1;
    } else {
      say("the bootstrap plugin %s is not one of NVSHMEM's (mpi, shmem, pmi, pmi2, pmix); other plugins are not supported", plugin);
      return 1;
    }
  }
  if (!strcasecmp(name.c_str(), "MPI")) return mpi_bootstrap(nullptr, true, b, "NVSHMEM_BOOTSTRAP=MPI");
  if (!strcasecmp(name.c_str(), "SHMEM")) return shmem_bootstrap(b);
  if (!strcasecmp(name.c_str(), "PMI")) {
    const char* pmi = std::getenv("NVSHMEM_BOOTSTRAP_PMI");
    std::string kind = pmi && *pmi ? pmi : "PMI";
    if (!strcasecmp(kind.c_str(), "PMIX")) return pmix_bootstrap(b);
    if (strcasecmp(kind.c_str(), "PMI") && strcasecmp(kind.c_str(), "PMI-2")) {
      say("bootstrap_pmi_init invalid PMI bootstrap '%s'", kind.c_str());
      return 1;
    }
    // libpmi.so and libpmi2.so (of Slurm, MPICH, ...) are not used here: without them NVIDIA's library makes a job of one PE.
    b->rank = 0;
    b->nranks = 1;
    b->id = random_token();
    b->launcher = true;
    return 0;
  }
  say("Missing init flags for bootstrap %s. Retry with nvshmemx_init_attr and non-zero flags", name.c_str());
  return 1;
}

int choose_bootstrap(unsigned flags, nvshmemx_init_attr_t* attr, Bootstrap* b) {
  if (flags & NVSHMEMX_INIT_WITH_MPI_COMM) return mpi_bootstrap(attr ? attr->mpi_comm : nullptr, false, b, "NVSHMEMX_INIT_WITH_MPI_COMM");
  if (flags & NVSHMEMX_INIT_WITH_SHMEM) return shmem_bootstrap(b);
  if (flags & NVSHMEMX_INIT_WITH_UNIQUEID) {
    if (!attr || !attr->args.uid_args.id) {
      say("NVSHMEMX_INIT_WITH_UNIQUEID without a unique ID in the attributes");
      return 1;
    }
    UidPayload p{};
    std::memcpy(&p, attr->args.uid_args.id->internal, sizeof p);
    const int rank = attr->args.uid_args.myrank, n = attr->args.uid_args.nranks;
    if (p.magic != kUidMagic || n < 1 || n > kMaxPes || rank < 0 || rank >= n) {
      say("the unique ID or the rank (%d of %d) is not valid (at most %d PEs)", rank, n, kMaxPes);
      return 1;
    }
    b->rank = rank;
    b->nranks = n;
    b->id.assign(p.token, strnlen(p.token, sizeof p.token));
    return 0;
  }
  const char* r = std::getenv("VGPU_NVSHMEM_RANK");
  const char* n = std::getenv("VGPU_NVSHMEM_NPES");
  if (r && n) {
    b->rank = std::atoi(r);
    b->nranks = std::atoi(n);
    const char* id = std::getenv("VGPU_NVSHMEM_ID");
    b->id = id && *id ? id : "env";
    if (b->nranks < 1 || b->nranks > kMaxPes || b->rank < 0 || b->rank >= b->nranks) {
      say("VGPU_NVSHMEM_RANK=%s VGPU_NVSHMEM_NPES=%s is not a valid PE of a job (at most %d PEs)", r, n, kMaxPes);
      return 1;
    }
    return 0;
  }
  return env_bootstrap(b);
}

bool attach_segment(State& s, const Bootstrap& b) {
  s.seg_path = shm_dir() + "/vgpu-nvshmem-" + b.id;
  const int fd = ::open(s.seg_path.c_str(), O_RDWR | O_CREAT, 0600);
  if (fd < 0) {
    say("cannot open the rendezvous file %s: %s", s.seg_path.c_str(), std::strerror(errno));
    return false;
  }
  struct stat sb {};
  if (::fstat(fd, &sb) != 0 || (size_t(sb.st_size) < sizeof(Segment) && ::ftruncate(fd, sizeof(Segment)) != 0)) {
    ::close(fd);
    say("cannot size the rendezvous file %s", s.seg_path.c_str());
    return false;
  }
  void* m = ::mmap(nullptr, sizeof(Segment), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ::close(fd);
  if (m == MAP_FAILED) {
    say("cannot map the rendezvous file %s", s.seg_path.c_str());
    return false;
  }
  s.seg = static_cast<Segment*>(m);
  uint32_t zero = 0;
  if (s.seg->magic.compare_exchange_strong(zero, kSegMagic + 1)) {
    // First to arrive: describe the job, then publish the magic.
    s.seg->version = kSegVersion;
    s.seg->npes = b.nranks;
    s.seg->magic.store(kSegMagic, std::memory_order_release);
  }
  wait_for([&] { return s.seg->magic.load(std::memory_order_acquire) == kSegMagic; }, "the rendezvous file");
  if (s.seg->npes != b.nranks || s.seg->version != kSegVersion) {
    say("PE %d says the job has %d PEs, the rendezvous file %s says %d", b.rank, b.nranks, s.seg_path.c_str(),
        s.seg->npes);
    return false;
  }
  s.seg->attached.fetch_add(1);
  return true;
}

int init(unsigned flags, nvshmemx_init_attr_t* attr) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (initialized(s)) return 0;
  Bootstrap b;
  if (int r = choose_bootstrap(flags, attr, &b)) return r;
  if (attr && (attr->version >> 16) >= 2 && attr->args.cuda_device_id >= 0)
    cuda_ok(cudaSetDevice(attr->args.cuda_device_id), "cudaSetDevice");
  if (cudaGetDevice(&s.device) != cudaSuccess) {
    cudaGetLastError();
    say("no CUDA device for this PE");
    return 1;
  }
  s.mype = b.rank;
  s.npes = b.nranks;
  s.bootstrapped_by_launcher = b.launcher;
  s.close_bootstrap = b.close;
  if (!attach_segment(s, b)) return 1;
  s.status = NVSHMEM_STATUS_IS_BOOTSTRAPPED;

  // Collective parameters: NVSHMEM's documented defaults.
  abi::CollParams& c = s.coll;
  c = abi::CollParams{};
  c.version = (2 << 16) + int(sizeof(abi::CollParams));
  c.barrier_dissem_kval = int(env_size("NVSHMEM_BARRIER_DISSEM_KVAL", 2));
  c.barrier_tg_dissem_kval = int(env_size("NVSHMEM_BARRIER_TG_DISSEM_KVAL", 2));
  c.reduce_recexch_kval = 2;
  c.bcast_tree_kval = 2;
  c.bcast_algo = 0;
  c.reduce_algo = 0;
  c.fcollect_ll_threshold = env_size("NVSHMEM_FCOLLECT_LL_THRESHOLD", 2048);
  c.fcollect_nvls_threshold = 0;
  c.reduce_scratch_size = env_size("NVSHMEM_REDUCE_SCRATCH_SIZE", 524288);
  c.fcollect_algo = 0;
  c.reducescatter_nvls_threshold = 0;
  c.reducescatter_algo = 0;
  c.reduce_maxloc_algo = 1;
  c.fcollect_ll128_threshold = 0;
  c.reduce_nvls_threshold = 0;
  c.fcollect_ll_supported = true;
  s.psync_len = abi::psync_len_per_team(c, s.npes);
  // NVIDIA's default is 256 teams; each costs this many bytes of symmetric
  // heap, so the simulator keeps 32 unless NVSHMEM_MAX_TEAMS asks for more.
  s.max_teams = int(std::clamp<size_t>(env_size("NVSHMEM_MAX_TEAMS", 32), NVSHMEM_TEAMS_MIN + 1, kMaxTeamsCap));

  // The symmetric heap: the teams' synchronization arrays, then the user heap.
  const size_t user = (env_size("NVSHMEM_SYMMETRIC_SIZE", size_t(1) << 30) + kMinAlign - 1) / kMinAlign * kMinAlign;
  s.user_offset = (size_t(s.max_teams) * s.psync_len * sizeof(long) + 65535) / 65536 * 65536;
  s.heap_size = s.user_offset + user;
  if (cudaMalloc(reinterpret_cast<void**>(&s.heap), s.heap_size) != cudaSuccess) {
    cudaGetLastError();
    say("cannot allocate a %zu-byte symmetric heap (NVSHMEM_SYMMETRIC_SIZE sets the user part)", s.heap_size);
    return 1;
  }
  s.free_blocks.clear();
  s.used_blocks.clear();
  s.free_blocks[s.user_offset] = user;
  s.peer.assign(size_t(s.npes), nullptr);
  s.peer[s.mype] = s.heap;
  Segment::Pe& me = s.seg->pe[s.mype];
  if (s.npes > 1) {
    cudaIpcMemHandle_t h{};
    if (cudaIpcGetMemHandle(&h, s.heap) != cudaSuccess) {
      cudaGetLastError();
      say("cannot share the symmetric heap with the other PEs");
      return 1;
    }
    std::memcpy(me.handle, &h, std::min(sizeof h, sizeof me.handle));
  }
  me.pid = int(::getpid());
  me.device = s.device;
  me.heap_size = s.heap_size;
  me.ready.store(1, std::memory_order_release);
  for (int pe = 0; pe < s.npes; ++pe) {
    if (pe == s.mype) continue;
    wait_for([&] { return s.seg->pe[pe].ready.load(std::memory_order_acquire) == 1; }, "the other PEs to join");
    if (s.seg->pe[pe].heap_size != s.heap_size) {
      say("PE %d's symmetric heap is %zu bytes, this PE's %zu: every PE must set the same NVSHMEM_SYMMETRIC_SIZE "
          "and NVSHMEM_MAX_TEAMS", pe, size_t(s.seg->pe[pe].heap_size), s.heap_size);
      return 1;
    }
    cudaIpcMemHandle_t h{};
    std::memcpy(&h, s.seg->pe[pe].handle, std::min(sizeof h, sizeof s.seg->pe[pe].handle));
    void* p = nullptr;
    if (cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess) != cudaSuccess) {
      cudaGetLastError();
      say("cannot map PE %d's symmetric heap", pe);
      return 1;
    }
    s.peer[pe] = static_cast<char*>(p);
    // A peer's heap lives in the address range of the peer's device; this
    // PE's kernels reach it once peer access to that device is enabled.
    const int pd = s.seg->pe[pe].device;
    if (pd != s.device) {
      // NVIDIA's NVSHMEM starts no job whose PEs sit on two GPUs that cannot reach each other: "Peer GPU 1 is not
      // accessible", NVSHMEMX_ERROR_NOT_SUPPORTED (3) (measured on two RTX 3060s, which have no peer path; the
      // launcher's program then ends with status 255).
      int can = 0;
      if (cudaDeviceCanAccessPeer(&can, s.device, pd) != cudaSuccess || !can) {
        cudaGetLastError();
        say("Peer GPU %d is not accessible", pd);
        return 3;
      }
      const cudaError_t e = cudaDeviceEnablePeerAccess(pd, 0);
      if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
        say("cannot enable peer access from device %d to device %d", s.device, pd);
        return 1;
      }
      cudaGetLastError();
    }
  }

  // Device-side state: peers, teams, counters.
  cuda_ok(cudaMalloc(reinterpret_cast<void**>(&s.dev_peers), sizeof(void*) * size_t(s.npes)), "cudaMalloc");
  cuda_ok(cudaMemcpy(s.dev_peers, s.peer.data(), sizeof(void*) * size_t(s.npes), cudaMemcpyHostToDevice), "cudaMemcpy");
  cuda_ok(cudaMalloc(reinterpret_cast<void**>(&s.dev_remote), sizeof(void*) * size_t(s.npes)), "cudaMalloc");
  cuda_ok(cudaMemset(s.dev_remote, 0, sizeof(void*) * size_t(s.npes)), "cudaMemset");
  cuda_ok(cudaMalloc(reinterpret_cast<void**>(&s.dev_team_pool), sizeof(void*) * size_t(s.max_teams)), "cudaMalloc");
  cuda_ok(cudaMemset(s.dev_team_pool, 0, sizeof(void*) * size_t(s.max_teams)), "cudaMemset");
  cuda_ok(cudaMalloc(reinterpret_cast<void**>(&s.dev_sync_counter), sizeof(long) * 2 * size_t(s.max_teams)),
          "cudaMalloc");
  cuda_ok(cudaMalloc(reinterpret_cast<void**>(&s.dev_timeout), sizeof(abi::Timeout)), "cudaMalloc");
  abi::Timeout to{};
  to.version = (1 << 16) + int(sizeof to);
  cuda_ok(cudaMemcpy(s.dev_timeout, &to, sizeof to, cudaMemcpyHostToDevice), "cudaMemcpy");
  cuda_ok(cudaMalloc(reinterpret_cast<void**>(&s.dev_wait_any), sizeof(unsigned long long)), "cudaMalloc");
  cuda_ok(cudaMemset(s.dev_wait_any, 0, sizeof(unsigned long long)), "cudaMemset");
  cuda_ok(cudaMalloc(reinterpret_cast<void**>(&s.dev_exit), 2 * sizeof(int)), "cudaMalloc");
  cuda_ok(cudaMemset(s.dev_exit, 0, 2 * sizeof(int)), "cudaMemset");
  s.teams.assign(size_t(s.max_teams), HostTeam{});
  s.psync_used.assign(size_t(s.max_teams), false);
  std::vector<int> world(size_t(s.npes));
  for (int i = 0; i < s.npes; ++i) world[i] = i;
  make_team(s, NVSHMEM_TEAM_WORLD, world, 0, 1);
  make_team(s, NVSHMEM_TEAM_SHARED, world, 0, 1);  // every PE shares memory with every other
  make_team(s, NVSHMEMX_TEAM_NODE, world, 0, 1);   // one node
  make_team(s, NVSHMEMX_TEAM_SAME_MYPE_NODE, {s.mype}, s.mype, 1);
  make_team(s, NVSHMEMI_TEAM_SAME_GPU, {s.mype}, s.mype, 1);  // a GPU each
  make_team(s, NVSHMEMI_TEAM_GPU_LEADERS, world, 0, 1);       // each PE leads its GPU
  build_device_state(s);
  // PEs on one GPU (the same device index; here every PE has a GPU of its own, so this is the index): the multiple
  // processes per GPU mode of NVSHMEM, as measured on the card (status 3, nvshmem_malloc and nvshmem_ptr still work).
  bool shares_gpu = false;
  for (int pe = 0; pe < s.npes; ++pe)
    if (pe != s.mype && s.seg->pe[pe].device == s.device) shares_gpu = true;
  s.status = shares_gpu ? NVSHMEM_STATUS_LIMITED_MPG : NVSHMEM_STATUS_IS_INITIALIZED;
  team_barrier(s, NVSHMEM_TEAM_WORLD);
  // Every PE has mapped every heap: the files' names are no longer needed,
  // and without them a PE that dies leaves nothing behind.
  if (s.npes > 1) vgpu_ipc_unlink(s.heap);
  return 0;
}

void finalize() {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (!initialized(s)) return;
  cudaDeviceSynchronize();
  cudaGetLastError();
  team_barrier(s, NVSHMEM_TEAM_WORLD);
  for (int i = 0; i < s.max_teams; ++i) drop_team(s, i);
  for (int pe = 0; pe < s.npes; ++pe)
    if (pe != s.mype && s.peer[pe]) cudaIpcCloseMemHandle(s.peer[pe]);
  // No PE may free its heap while another still has it mapped.
  team_barrier(s, NVSHMEM_TEAM_WORLD);
  cudaFree(s.heap);
  cudaFree(s.dev_peers);
  cudaFree(s.dev_remote);
  cudaFree(s.dev_team_pool);
  cudaFree(s.dev_sync_counter);
  cudaFree(s.dev_timeout);
  cudaFree(s.dev_wait_any);
  cudaFree(s.dev_exit);
  cudaGetLastError();
  const uint32_t done = s.seg->finished.fetch_add(1) + 1;
  if (done == uint32_t(s.npes)) ::unlink(s.seg_path.c_str());
  ::munmap(s.seg, sizeof(Segment));
  s.seg = nullptr;
  s.heap = nullptr;
  s.peer.clear();
  s.state_targets.clear();
  if (s.close_bootstrap) s.close_bootstrap();
  s.close_bootstrap = nullptr;
  // MPI, OpenSHMEM and PMI bootstraps stay bootstrapped after the finalize (measured); a unique ID's does not.
  s.status = s.bootstrapped_by_launcher ? NVSHMEM_STATUS_IS_BOOTSTRAPPED : NVSHMEM_STATUS_NOT_INITIALIZED;
}

// Collectives on the host: synchronized, then plain puts.
int team_rank_ok(State& s, nvshmem_team_t team, const char* api) {
  if (!initialized(s)) {
    say("%s before nvshmem_init", api);
    return -1;
  }
  if (team < 0 || team >= s.max_teams || !s.teams[team].valid) {
    say("%s: team %d is not a team of this PE", api, team);
    return -1;
  }
  return s.teams[team].my_pe;
}

int broadcast(nvshmem_team_t team, void* dest, const void* src, size_t nelems, int root) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  const int me = team_rank_ok(s, team, "nvshmem_broadcastmem");
  if (me < 0) return 1;
  const HostTeam& t = s.teams[team];
  if (root < 0 || root >= t.size) return say("nvshmem_broadcastmem: root %d is not in the team", root), 1;
  team_barrier(s, team);
  if (me == root)
    for (int i = 0; i < t.size; ++i) put(dest, src, nelems, t.members[i], "nvshmem_broadcastmem");
  team_barrier(s, team);
  return 0;
}

int fcollect(nvshmem_team_t team, void* dest, const void* src, size_t nelems) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  const int me = team_rank_ok(s, team, "nvshmem_fcollectmem");
  if (me < 0) return 1;
  const HostTeam& t = s.teams[team];
  team_barrier(s, team);
  for (int i = 0; i < t.size; ++i)
    put(static_cast<char*>(dest) + size_t(me) * nelems, src, nelems, t.members[i], "nvshmem_fcollectmem");
  team_barrier(s, team);
  return 0;
}

int alltoall(nvshmem_team_t team, void* dest, const void* src, size_t nelems) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  const int me = team_rank_ok(s, team, "nvshmem_alltoallmem");
  if (me < 0) return 1;
  const HostTeam& t = s.teams[team];
  team_barrier(s, team);
  for (int i = 0; i < t.size; ++i)
    put(static_cast<char*>(dest) + size_t(me) * nelems, static_cast<const char*>(src) + size_t(i) * nelems, nelems,
        t.members[i], "nvshmem_alltoallmem");
  team_barrier(s, team);
  return 0;
}

int split_strided(nvshmem_team_t parent, int start, int stride, int size, const nvshmem_team_config_t* config,
                  long config_mask, nvshmem_team_t* new_team) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (!new_team) return 1;
  *new_team = NVSHMEM_TEAM_INVALID;
  const int me = team_rank_ok(s, parent, "nvshmem_team_split_strided");
  if (me < 0) return 1;
  HostTeam& p = s.teams[parent];
  // The new team's PEs, in the parent's numbering.
  if (size <= 0 || start < 0 || start >= p.size || (size > 1 && stride == 0)) {
    team_barrier(s, parent);
    return size <= 0 ? 0 : 1;
  }
  std::vector<int> members;
  for (int i = 0; i < size; ++i) {
    const long long q = (long long)start + (long long)i * stride;
    if (q < 0 || q >= p.size) {
      say("nvshmem_team_split_strided: start %d, stride %d, size %d leaves the parent team", start, stride, size);
      return 1;
    }
    members.push_back(p.members[size_t(q)]);
  }
  // One index for the whole new team, job-wide: the parent's PE 0 takes it,
  // the others read it, keyed by how many splits the parent has made.
  const uint64_t seq = ++p.splits;
  if (me == 0) {
    int idx = -1;
    if (!take_team_index(s, &idx)) idx = -1;
    s.seg->split_index[parent].store(idx, std::memory_order_relaxed);
    s.seg->split_seq[parent].store(seq, std::memory_order_release);
  }
  team_barrier(s, parent);
  int idx = -1;
  wait_for([&] { return s.seg->split_seq[parent].load(std::memory_order_acquire) == seq; }, "a team split");
  idx = s.seg->split_index[parent].load(std::memory_order_relaxed);
  team_barrier(s, parent);
  if (idx < 0) {
    say("nvshmem_team_split_strided: no team index left (NVSHMEM_MAX_TEAMS=%d)", s.max_teams);
    return 1;
  }
  const bool member = std::find(members.begin(), members.end(), s.mype) != members.end();
  if (member) {
    const int world_start = members[0];
    const int world_stride = members.size() > 1 ? members[1] - members[0] : 1;
    make_team(s, idx, members, world_start, world_stride);
    if (config && (config_mask & NVSHMEM_TEAM_CONFIG_MASK_NUM_CONTEXTS))
      s.teams[idx].config.num_contexts = config->num_contexts;
    if (config && (config_mask & NVSHMEM_TEAM_CONFIG_MASK_UNIQUEID)) s.teams[idx].config.uniqueid = config->uniqueid;
    *new_team = idx;
  }
  // The new team is ready on every member before any of them uses it.
  team_barrier(s, parent);
  return 0;
}

// Atomics, reductions and the arithmetic of the typed API.
#include "nvshmem_typed_impl.inc"

}  // namespace

#define NVSHMEM_EXPORT __attribute__((visibility("default")))

// ============================================================================
// Initialization
// ============================================================================

// The C++ entry point NVSHMEM's inline nvshmem_init() calls. Programs that link
// NVIDIA's device library get its definition instead, which calls
// nvshmemid_hostlib_init_attr below.
NVSHMEM_EXPORT int nvshmemi_init_thread(int requested, int* provided, unsigned int bootstrap_flags,
                                        nvshmemx_init_attr_t* attr, nvshmemi_version_t) {
  const int r = init(bootstrap_flags, attr);
  if (provided) *provided = std::min(requested, int(NVSHMEM_THREAD_MULTIPLE));
  st().provided = provided ? *provided : requested;
  return r;
}

extern "C" {

// The device library's callback: it reports where nvshmemi_device_state_d
// (and a transport state this library does not use) live on this PE's GPU.
typedef int (*nvshmemx_device_lib_init_cb)(void** dev_state_ptr, void** transport_dev_state_ptr);

// Which device transport the device library should expect: the proxy, which
// with every PE reachable peer to peer is never called.
NVSHMEM_EXPORT int nvshmem_selected_device_transport = 0;

NVSHMEM_EXPORT int nvshmemid_hostlib_init_attr(int requested, int* provided, unsigned int bootstrap_flags,
                                               nvshmemx_init_attr_t* attr, nvshmemi_version_t version,
                                               nvshmemx_device_lib_init_cb cb) {
  if (version.major != NVSHMEM_VENDOR_MAJOR_VERSION) {
    say("the device library is NVSHMEM %d.%d.%d; this host library implements %d.x", version.major, version.minor,
        version.patch, NVSHMEM_VENDOR_MAJOR_VERSION);
    return 1;
  }
  const int r = init(bootstrap_flags, attr);
  if (r) return r;
  if (provided) *provided = std::min(requested, int(NVSHMEM_THREAD_MULTIPLE));
  State& s = st();
  if (cb) {
    void* dev_state = nullptr;
    void* transport = nullptr;
    if (cb(&dev_state, &transport) != 0 || !dev_state) {
      say("the device library did not report nvshmemi_device_state_d");
      return 1;
    }
    std::lock_guard<std::recursive_mutex> lock(s.mu);
    s.state_targets.push_back(dev_state);
    write_device_state(s, dev_state);
  }
  return 0;
}

NVSHMEM_EXPORT void nvshmemid_hostlib_finalize(void*, void*) { finalize(); }

NVSHMEM_EXPORT int nvshmemid_init_status() { return st().status; }
NVSHMEM_EXPORT int nvshmemx_init_status() { return st().status; }

NVSHMEM_EXPORT int nvshmemx_hostlib_init_attr(unsigned int flags, nvshmemx_init_attr_t* attr) {
  return init(flags, attr);
}
NVSHMEM_EXPORT void nvshmemx_hostlib_finalize() { finalize(); }
NVSHMEM_EXPORT void nvshmemi_finalize() { finalize(); }

NVSHMEM_EXPORT void nvshmem_query_thread(int* provided) {
  if (provided) *provided = st().provided;
}

NVSHMEM_EXPORT void nvshmem_global_exit(int status) {
  std::fflush(stdout);
  std::fprintf(stderr, "[vgpu] nvshmem: PE %d called nvshmem_global_exit(%d)\n", st().mype, status);
  std::exit(status);
}

NVSHMEM_EXPORT int nvshmemx_get_uniqueid(nvshmemx_uniqueid_t* uniqueid) {
  if (!uniqueid) return 1;
  std::memset(uniqueid, 0, sizeof *uniqueid);
  uniqueid->version = (1 << 16) + int(sizeof(nvshmemx_uniqueid_t));
  UidPayload p{};
  p.magic = kUidMagic;
  const std::string t = random_token();
  std::snprintf(p.token, sizeof p.token, "%s", t.c_str());
  std::memcpy(uniqueid->internal, &p, sizeof p);
  return 0;
}

NVSHMEM_EXPORT int nvshmemx_set_attr_uniqueid_args(const int myrank, const int nranks,
                                                   const nvshmemx_uniqueid_t* uniqueid,
                                                   nvshmemx_init_attr_t* attr) {
  if (!attr || !uniqueid) return 1;
  attr->version = NVSHMEM_INIT_ATTR_V2_IDENTIFIER;
  attr->args.version = NVSHMEM_INIT_ARGS_V2_IDENTIFIER;
  attr->args.uid_args.version = (1 << 16) + int(sizeof(nvshmemx_uniqueid_args_t));
  attr->args.uid_args.id = const_cast<nvshmemx_uniqueid_t*>(uniqueid);
  attr->args.uid_args.myrank = myrank;
  attr->args.uid_args.nranks = nranks;
  return 0;
}

NVSHMEM_EXPORT int nvshmemx_set_attr_mpi_comm_args(void* mpi_comm, nvshmemx_init_attr_t* nvshmem_attr) {
  if (!nvshmem_attr) return 1;
  nvshmem_attr->version = NVSHMEM_INIT_ATTR_V2_IDENTIFIER;
  nvshmem_attr->args.version = NVSHMEM_INIT_ARGS_V2_IDENTIFIER;
  nvshmem_attr->mpi_comm = mpi_comm;
  return 0;
}

// A module loaded with the driver API carries its own nvshmemi_device_state_d.
NVSHMEM_EXPORT int nvshmemx_cumodule_init(CUmodule module) {
  using GetGlobal = CUresult (*)(CUdeviceptr*, size_t*, CUmodule, const char*);
  auto fn = reinterpret_cast<GetGlobal>(dlsym(RTLD_DEFAULT, "cuModuleGetGlobal_v2"));
  State& s = st();
  if (!fn || !initialized(s)) return 1;
  CUdeviceptr p = 0;
  size_t n = 0;
  if (fn(&p, &n, module, "nvshmemi_device_state_d") != CUDA_SUCCESS || n < sizeof(abi::DeviceState)) return 1;
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  write_device_state(s, reinterpret_cast<void*>(p));
  return 0;
}
NVSHMEM_EXPORT int nvshmemx_cumodule_finalize(CUmodule) { return 0; }

// The same for a library loaded with the driver's cuLibrary calls.
NVSHMEM_EXPORT int nvshmemx_culibrary_init(CUlibrary library) {
  using GetGlobal = CUresult (*)(CUdeviceptr*, size_t*, CUlibrary, const char*);
  auto fn = reinterpret_cast<GetGlobal>(dlsym(RTLD_DEFAULT, "cuLibraryGetGlobal"));
  State& s = st();
  if (!fn || !initialized(s)) return 1;
  CUdeviceptr p = 0;
  size_t n = 0;
  if (fn(&p, &n, library, "nvshmemi_device_state_d") != CUDA_SUCCESS || n < sizeof(abi::DeviceState)) return 1;
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  write_device_state(s, reinterpret_cast<void*>(p));
  return 0;
}
NVSHMEM_EXPORT int nvshmemx_culibrary_finalize(CUlibrary) { return 0; }

// Regions (hints for batching RMA), queue pairs (IBGDA) and externally mapped
// symmetric buffers belong to transports and drivers this library does not have.
NVSHMEM_EXPORT int nvshmemx_region_start(void*, const void*) {
  say("nvshmemx_region_start: regions are not available in VirtualGPU");
  return 1;
}
NVSHMEM_EXPORT int nvshmemx_region_stop(uint64_t) {
  say("nvshmemx_region_stop: regions are not available in VirtualGPU");
  return 1;
}
NVSHMEM_EXPORT int nvshmemx_region_is_active(uint32_t, int* active) {
  if (active) *active = 0;
  return 0;
}
NVSHMEM_EXPORT int nvshmemx_qp_create(int, void**) {
  say("nvshmemx_qp_create: there are no queue pairs: every PE is a peer");
  return 1;
}
NVSHMEM_EXPORT void* nvshmemx_buffer_register_symmetric(void*, size_t, int) {
  say("nvshmemx_buffer_register_symmetric: external buffers cannot join the symmetric heap");
  return nullptr;
}
NVSHMEM_EXPORT void* nvshmemx_buffer_register_symmetric_at_preferred_address(void*, size_t, void*, int) {
  say("nvshmemx_buffer_register_symmetric_at_preferred_address: external buffers cannot join the symmetric heap");
  return nullptr;
}
NVSHMEM_EXPORT int nvshmemx_buffer_unregister_symmetric(void*, size_t) { return 1; }

// A team from a unique ID: one PE asks for the ID, the others get it from it
// out of band, and every member calls nvshmemx_team_init with it, the team's
// size and its own index. Derived from the documentation.
NVSHMEM_EXPORT int nvshmemx_team_get_uniqueid(nvshmemx_team_uniqueid_t* uniqueid) {
  if (!uniqueid) return 1;
  uint64_t r = 0;
  if (::getrandom(&r, sizeof r, 0) != sizeof r) r = uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
  *uniqueid = (r & ~(1ull << 63)) | 1ull;  // not 0 (free slot) and not ~0 (no ID)
  return 0;
}
NVSHMEM_EXPORT int nvshmemx_team_init(nvshmem_team_t* team, nvshmem_team_config_t* config, long config_mask, int npes,
                                      int pe_idx_in_team) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (!team) return 1;
  *team = NVSHMEM_TEAM_INVALID;
  if (!initialized(s)) return say("nvshmemx_team_init before nvshmem_init"), 1;
  if (!config || !(config_mask & NVSHMEM_TEAM_CONFIG_MASK_UNIQUEID) || config->uniqueid == 0 ||
      config->uniqueid == ~0ull)
    return say("nvshmemx_team_init needs a unique ID from nvshmemx_team_get_uniqueid in the configuration"), 1;
  if (npes < 1 || npes > s.npes || pe_idx_in_team < 0 || pe_idx_in_team >= npes)
    return say("nvshmemx_team_init: team index %d of %d PEs in a job of %d", pe_idx_in_team, npes, s.npes), 1;
  const uint64_t uid = config->uniqueid;
  Segment::UidSlot* slot = nullptr;
  for (auto& u : s.seg->uid_slot) {
    uint64_t cur = u.uid.load(std::memory_order_acquire);
    if (cur == uid) { slot = &u; break; }
  }
  while (!slot) {
    for (auto& u : s.seg->uid_slot) {
      uint64_t free_slot = 0;
      if (u.uid.compare_exchange_strong(free_slot, uid) || free_slot == uid) { slot = &u; break; }
    }
    if (!slot) std::this_thread::yield();
  }
  if (slot->arrived.fetch_add(1, std::memory_order_acq_rel) == 0) {
    int idx = -1;
    if (!take_team_index(s, &idx)) idx = -1;
    slot->index.store(idx >= 0 ? idx + 1 : -1, std::memory_order_release);  // 0 means "not yet"
  }
  slot->member[pe_idx_in_team].store(s.mype + 1, std::memory_order_release);
  wait_for([&] { return slot->index.load(std::memory_order_acquire) != 0; }, "the team's first member");
  const int stored = slot->index.load(std::memory_order_acquire);
  const int idx = stored > 0 ? stored - 1 : -1;
  std::vector<int> members(size_t(npes), -1);
  for (int i = 0; i < npes; ++i) {
    wait_for([&] { return slot->member[i].load(std::memory_order_acquire) != 0; }, "the members of a team");
    members[size_t(i)] = slot->member[i].load(std::memory_order_acquire) - 1;
  }
  if (idx < 0) {
    say("nvshmemx_team_init: no team index left (NVSHMEM_MAX_TEAMS=%d)", s.max_teams);
  } else {
    make_team(s, idx, members, members[0], members.size() > 1 ? members[1] - members[0] : 1);
    s.teams[idx].config.uniqueid = uid;
    if (config_mask & NVSHMEM_TEAM_CONFIG_MASK_NUM_CONTEXTS) s.teams[idx].config.num_contexts = config->num_contexts;
    *team = idx;
    team_barrier(s, idx);  // the team exists on every member before any of them uses it
  }
  // The last member out frees the slot for the next team made with this ID.
  if (slot->departed.fetch_add(1, std::memory_order_acq_rel) + 1 == uint32_t(npes)) {
    for (auto& m : slot->member) m.store(0, std::memory_order_relaxed);
    slot->index.store(0, std::memory_order_relaxed);
    slot->arrived.store(0, std::memory_order_relaxed);
    slot->departed.store(0, std::memory_order_relaxed);
    slot->uid.store(0, std::memory_order_release);
  }
  return *team == NVSHMEM_TEAM_INVALID ? 1 : 0;
}

// ============================================================================
// Who and where
// ============================================================================

NVSHMEM_EXPORT int nvshmem_my_pe() { return initialized(st()) ? st().mype : -1; }
NVSHMEM_EXPORT int nvshmem_n_pes() { return initialized(st()) ? st().npes : -1; }

NVSHMEM_EXPORT void nvshmem_info_get_version(int* major, int* minor) {
  if (major) *major = NVSHMEM_MAJOR_VERSION;
  if (minor) *minor = NVSHMEM_MINOR_VERSION;
}
NVSHMEM_EXPORT void nvshmem_info_get_name(char* name) {
  if (name) std::snprintf(name, NVSHMEM_MAX_NAME_LEN, "NVSHMEM v%d.%d.%d", NVSHMEM_VENDOR_MAJOR_VERSION,
                          NVSHMEM_VENDOR_MINOR_VERSION, NVSHMEM_VENDOR_PATCH_VERSION);
}
NVSHMEM_EXPORT void nvshmemx_vendor_get_version_info(int* major, int* minor, int* patch) {
  if (major) *major = NVSHMEM_VENDOR_MAJOR_VERSION;
  if (minor) *minor = NVSHMEM_VENDOR_MINOR_VERSION;
  if (patch) *patch = NVSHMEM_VENDOR_PATCH_VERSION;
}

// ============================================================================
// The symmetric heap
// ============================================================================

// The allocation routines are collective: every PE allocates, then they meet,
// so that no PE writes to a block before every PE has it (calloc's zeros
// included: they are written before the meeting, or a fast PE's first put
// could land before a slow PE cleared the block).
static void* collective_alloc(size_t alignment, size_t size, bool zero) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (!initialized(s)) return nullptr;
  void* p = nullptr;
  if (alignment && !(alignment & (alignment - 1))) {
    p = heap_alloc(s, alignment, size);
    if (!p && size) say("nvshmem_malloc: %zu bytes do not fit in the symmetric heap (NVSHMEM_SYMMETRIC_SIZE)", size);
    if (p && zero) cuda_ok(cudaMemset(p, 0, size), "cudaMemset");
  }
  team_barrier(s, NVSHMEM_TEAM_WORLD);
  return p;
}
NVSHMEM_EXPORT void* nvshmem_align(size_t alignment, size_t size) { return collective_alloc(alignment, size, false); }
NVSHMEM_EXPORT void* nvshmem_malloc(size_t size) { return collective_alloc(kMinAlign, size, false); }
NVSHMEM_EXPORT void* nvshmem_calloc(size_t count, size_t size) {
  if (size && count > SIZE_MAX / size) return collective_alloc(kMinAlign, 0, false);
  return collective_alloc(kMinAlign, count * size, true);
}
NVSHMEM_EXPORT void nvshmem_free(void* ptr) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (!initialized(s)) return;
  // A barrier first: no PE may still be using the block another frees.
  cudaDeviceSynchronize();
  cudaGetLastError();
  team_barrier(s, NVSHMEM_TEAM_WORLD);
  if (ptr) heap_free(s, ptr);
}

NVSHMEM_EXPORT void* nvshmem_ptr(const void* dest, int pe) {
  State& s = st();
  if (!initialized(s) || pe < 0 || pe >= s.npes || !symmetric(s, dest)) return nullptr;
  return remote(s, dest, pe);
}
// No NVLink SHARP here: there is no multicast address, as on hardware without it.
NVSHMEM_EXPORT void* nvshmemx_mc_ptr(nvshmem_team_t, const void*) { return nullptr; }
// Local buffers need no registration to be a source or target of RMA here.
NVSHMEM_EXPORT int nvshmemx_buffer_register(void*, size_t) { return 0; }
NVSHMEM_EXPORT int nvshmemx_buffer_unregister(void*) { return 0; }
NVSHMEM_EXPORT void nvshmemx_buffer_unregister_all() {}

// ============================================================================
// Remote memory access
// ============================================================================

NVSHMEM_EXPORT void nvshmem_putmem(void* dest, const void* source, size_t bytes, int pe) {
  put(dest, source, bytes, pe, "nvshmem_putmem");
}
NVSHMEM_EXPORT void nvshmem_getmem(void* dest, const void* source, size_t bytes, int pe) {
  get(dest, source, bytes, pe, "nvshmem_getmem");
}
NVSHMEM_EXPORT void nvshmem_putmem_nbi(void* dest, const void* source, size_t bytes, int pe) {
  put(dest, source, bytes, pe, "nvshmem_putmem_nbi");
}
NVSHMEM_EXPORT void nvshmem_getmem_nbi(void* dest, const void* source, size_t bytes, int pe) {
  get(dest, source, bytes, pe, "nvshmem_getmem_nbi");
}
NVSHMEM_EXPORT void nvshmemx_putmem_on_stream(void* dest, const void* source, size_t bytes, int pe, cudaStream_t s) {
  in_stream_order(s, [=] { put(dest, source, bytes, pe, "nvshmemx_putmem_on_stream"); });
}
NVSHMEM_EXPORT void nvshmemx_getmem_on_stream(void* dest, const void* source, size_t bytes, int pe, cudaStream_t s) {
  in_stream_order(s, [=] { get(dest, source, bytes, pe, "nvshmemx_getmem_on_stream"); });
}
NVSHMEM_EXPORT void nvshmemx_putmem_nbi_on_stream(void* dest, const void* source, size_t bytes, int pe,
                                                  cudaStream_t s) {
  nvshmemx_putmem_on_stream(dest, source, bytes, pe, s);
}
NVSHMEM_EXPORT void nvshmemx_getmem_nbi_on_stream(void* dest, const void* source, size_t bytes, int pe,
                                                  cudaStream_t s) {
  nvshmemx_getmem_on_stream(dest, source, bytes, pe, s);
}

#define VGPU_SIZED(BITS)                                                                                       \
  NVSHMEM_EXPORT void nvshmem_put##BITS(void* d, const void* src, size_t n, int pe) {                         \
    put(d, src, n * (BITS / 8), pe, "nvshmem_put" #BITS);                                                     \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmem_get##BITS(void* d, const void* src, size_t n, int pe) {                         \
    get(d, src, n * (BITS / 8), pe, "nvshmem_get" #BITS);                                                     \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmem_put##BITS##_nbi(void* d, const void* src, size_t n, int pe) {                   \
    put(d, src, n * (BITS / 8), pe, "nvshmem_put" #BITS "_nbi");                                              \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmem_get##BITS##_nbi(void* d, const void* src, size_t n, int pe) {                   \
    get(d, src, n * (BITS / 8), pe, "nvshmem_get" #BITS "_nbi");                                              \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmem_iput##BITS(void* d, const void* src, ptrdiff_t dst, ptrdiff_t sst, size_t n,    \
                                         int pe) {                                                             \
    iput(d, src, dst, sst, n, BITS / 8, pe, true);                                                             \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmem_iget##BITS(void* d, const void* src, ptrdiff_t dst, ptrdiff_t sst, size_t n,    \
                                         int pe) {                                                             \
    iput(d, src, dst, sst, n, BITS / 8, pe, false);                                                            \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmemx_put##BITS##_on_stream(void* d, const void* src, size_t n, int pe,              \
                                                     cudaStream_t s) {                                         \
    nvshmemx_putmem_on_stream(d, src, n * (BITS / 8), pe, s);                                                  \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmemx_get##BITS##_on_stream(void* d, const void* src, size_t n, int pe,              \
                                                     cudaStream_t s) {                                         \
    nvshmemx_getmem_on_stream(d, src, n * (BITS / 8), pe, s);                                                  \
  }
VGPU_SIZED(8)
VGPU_SIZED(16)
VGPU_SIZED(32)
VGPU_SIZED(64)
VGPU_SIZED(128)
#undef VGPU_SIZED

#define VGPU_TYPED(NAME, TYPE)                                                                                 \
  NVSHMEM_EXPORT void nvshmem_##NAME##_put(TYPE* d, const TYPE* src, size_t n, int pe) {                      \
    put(d, src, n * sizeof(TYPE), pe, "nvshmem_" #NAME "_put");                                               \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmem_##NAME##_get(TYPE* d, const TYPE* src, size_t n, int pe) {                      \
    get(d, src, n * sizeof(TYPE), pe, "nvshmem_" #NAME "_get");                                               \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmem_##NAME##_p(TYPE* d, const TYPE v, int pe) {                                     \
    put(d, &v, sizeof(TYPE), pe, "nvshmem_" #NAME "_p");                                                      \
  }                                                                                                            \
  NVSHMEM_EXPORT TYPE nvshmem_##NAME##_g(const TYPE* src, int pe) {                                           \
    TYPE v{};                                                                                                  \
    get(&v, src, sizeof(TYPE), pe, "nvshmem_" #NAME "_g");                                                    \
    return v;                                                                                                  \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmemx_##NAME##_put_on_stream(TYPE* d, const TYPE* src, size_t n, int pe,             \
                                                      cudaStream_t s) {                                        \
    nvshmemx_putmem_on_stream(d, src, n * sizeof(TYPE), pe, s);                                                \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmemx_##NAME##_get_on_stream(TYPE* d, const TYPE* src, size_t n, int pe,             \
                                                      cudaStream_t s) {                                        \
    nvshmemx_getmem_on_stream(d, src, n * sizeof(TYPE), pe, s);                                                \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmemx_##NAME##_p_on_stream(TYPE* d, const TYPE v, int pe, cudaStream_t s) {          \
    in_stream_order(s, [=] { put(d, &v, sizeof(TYPE), pe, "nvshmemx_" #NAME "_p_on_stream"); });             \
  }
VGPU_NVSHMEM_FOR_TYPES(VGPU_TYPED)

NVSHMEM_EXPORT void nvshmemx_putmem_signal_on_stream(void* dest, const void* source, size_t bytes,
                                                     uint64_t* sig_addr, uint64_t signal, int sig_op, int pe,
                                                     cudaStream_t s) {
  in_stream_order(s, [=] {
    put(dest, source, bytes, pe, "nvshmemx_putmem_signal_on_stream");
    signal_op(sig_addr, signal, sig_op, pe, "nvshmemx_putmem_signal_on_stream");
  });
}
NVSHMEM_EXPORT void nvshmemx_putmem_signal_nbi_on_stream(void* dest, const void* source, size_t bytes,
                                                         uint64_t* sig_addr, uint64_t signal, int sig_op, int pe,
                                                         cudaStream_t s) {
  nvshmemx_putmem_signal_on_stream(dest, source, bytes, sig_addr, signal, sig_op, pe, s);
}
NVSHMEM_EXPORT void nvshmemx_signal_op_on_stream(uint64_t* sig_addr, uint64_t signal, int sig_op, int pe,
                                                 cudaStream_t s) {
  in_stream_order(s, [=] { signal_op(sig_addr, signal, sig_op, pe, "nvshmemx_signal_op_on_stream"); });
}
NVSHMEM_EXPORT uint64_t nvshmem_signal_fetch(uint64_t* sig_addr) {
  uint64_t v = 0;
  copy(&v, sig_addr, sizeof v, "nvshmem_signal_fetch");
  return v;
}
NVSHMEM_EXPORT void nvshmemx_signal_wait_until_on_stream(uint64_t* sig_addr, int cmp, uint64_t cmp_value,
                                                         cudaStream_t s) {
  in_stream_order(s, [=] {
    wait_for([&] {
      uint64_t v = 0;
      copy(&v, sig_addr, sizeof v, "nvshmemx_signal_wait_until_on_stream");
      return compare(v, cmp, cmp_value);
    }, "a signal");
  });
}

// ============================================================================
// Ordering and synchronization
// ============================================================================

// Host-side operations complete before they return, so there is nothing to
// order or wait for -- but device work queued before must have finished.
NVSHMEM_EXPORT void nvshmem_quiet() {
  cudaDeviceSynchronize();
  cudaGetLastError();
}
NVSHMEM_EXPORT void nvshmem_fence() {}
NVSHMEM_EXPORT void nvshmemx_quiet_on_stream(cudaStream_t s) { in_stream_order(s, [] {}); }
// Host-side puts are finished when they return, so a flush has nothing to wait for.
NVSHMEM_EXPORT void nvshmemx_flush() {}
NVSHMEM_EXPORT void nvshmemx_flush_on_stream(cudaStream_t s) { in_stream_order(s, [] {}); }
// A counted signal is a plain word that the host resets to zero.
NVSHMEM_EXPORT void nvshmemx_signal_counted_reset(uint64_t* signal_addr) {
  const uint64_t zero = 0;
  copy(signal_addr, &zero, sizeof zero, "nvshmemx_signal_counted_reset");
}

// Equivalent to the on-stream form on the default stream, then a wait for it.
NVSHMEM_EXPORT void nvshmem_barrier_all() {
  State& s = st();
  if (!initialized(s)) return say("nvshmem_barrier_all before nvshmem_init");
  cudaDeviceSynchronize();
  cudaGetLastError();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  team_barrier(s, NVSHMEM_TEAM_WORLD);
}
NVSHMEM_EXPORT void nvshmem_sync_all() { nvshmem_barrier_all(); }
NVSHMEM_EXPORT int nvshmem_team_sync(nvshmem_team_t team) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (team_rank_ok(s, team, "nvshmem_team_sync") < 0) return 1;
  cudaDeviceSynchronize();
  cudaGetLastError();
  team_barrier(s, team);
  return 0;
}
NVSHMEM_EXPORT int nvshmem_barrier(nvshmem_team_t team) { return nvshmem_team_sync(team); }
NVSHMEM_EXPORT void nvshmemx_barrier_all_on_stream(cudaStream_t stream) {
  in_stream_order(stream, [] {
    State& s = st();
    std::lock_guard<std::recursive_mutex> lock(s.mu);
    if (initialized(s)) team_barrier(s, NVSHMEM_TEAM_WORLD);
  });
}
NVSHMEM_EXPORT void nvshmemx_sync_all_on_stream(cudaStream_t stream) { nvshmemx_barrier_all_on_stream(stream); }
NVSHMEM_EXPORT int nvshmemx_team_sync_on_stream(nvshmem_team_t team, cudaStream_t stream) {
  State& s = st();
  if (team_rank_ok(s, team, "nvshmemx_team_sync_on_stream") < 0) return 1;
  in_stream_order(stream, [team] {
    State& s2 = st();
    std::lock_guard<std::recursive_mutex> lock(s2.mu);
    team_barrier(s2, team);
  });
  return 0;
}
NVSHMEM_EXPORT int nvshmemx_barrier_on_stream(nvshmem_team_t team, cudaStream_t stream) {
  return nvshmemx_team_sync_on_stream(team, stream);
}

// ============================================================================
// Teams
// ============================================================================

NVSHMEM_EXPORT int nvshmem_team_my_pe(nvshmem_team_t team) {
  State& s = st();
  if (!initialized(s) || team < 0 || team >= s.max_teams || !s.teams[team].valid) return -1;
  return s.teams[team].my_pe;
}
NVSHMEM_EXPORT int nvshmem_team_n_pes(nvshmem_team_t team) {
  State& s = st();
  if (!initialized(s) || team < 0 || team >= s.max_teams || !s.teams[team].valid) return -1;
  return s.teams[team].size;
}
NVSHMEM_EXPORT int nvshmem_team_translate_pe(nvshmem_team_t src_team, int src_pe, nvshmem_team_t dest_team) {
  State& s = st();
  if (!initialized(s)) return -1;
  for (nvshmem_team_t t : {src_team, dest_team})
    if (t < 0 || t >= s.max_teams || !s.teams[t].valid) return -1;
  const HostTeam& a = s.teams[src_team];
  const HostTeam& b = s.teams[dest_team];
  if (src_pe < 0 || src_pe >= a.size) return -1;
  const int world = a.members[size_t(src_pe)];
  const auto it = std::find(b.members.begin(), b.members.end(), world);
  return it == b.members.end() ? -1 : int(it - b.members.begin());
}
NVSHMEM_EXPORT int nvshmem_team_split_strided(nvshmem_team_t parent_team, int start, int stride, int size,
                                              const nvshmem_team_config_t* config, long config_mask,
                                              nvshmem_team_t* new_team) {
  return split_strided(parent_team, start, stride, size, config, config_mask, new_team);
}
// A 2-D grid of the parent's PEs, xrange wide: the x team is this PE's row,
// the y team its column. Both splits are collective over the parent.
NVSHMEM_EXPORT int nvshmem_team_split_2d(nvshmem_team_t parent_team, int xrange,
                                         const nvshmem_team_config_t* xaxis_config, long xaxis_mask,
                                         nvshmem_team_t* xaxis_team, const nvshmem_team_config_t* yaxis_config,
                                         long yaxis_mask, nvshmem_team_t* yaxis_team) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  const int me = team_rank_ok(s, parent_team, "nvshmem_team_split_2d");
  if (me < 0 || xrange <= 0 || !xaxis_team || !yaxis_team) return 1;
  const int n = s.teams[parent_team].size;
  const int xr = std::min(xrange, n);
  const int rows = (n + xr - 1) / xr;
  int rc = 0;
  *xaxis_team = *yaxis_team = NVSHMEM_TEAM_INVALID;
  // Every parent PE takes part in every row's and every column's split, so
  // that each split is collective over the whole parent.
  for (int r = 0; r < rows; ++r) {
    nvshmem_team_t t = NVSHMEM_TEAM_INVALID;
    const int start = r * xr, size = std::min(xr, n - start);
    rc |= split_strided(parent_team, start, 1, size, xaxis_config, xaxis_mask, &t);
    if (t != NVSHMEM_TEAM_INVALID) *xaxis_team = t;
  }
  for (int c = 0; c < xr; ++c) {
    nvshmem_team_t t = NVSHMEM_TEAM_INVALID;
    const int size = (n - c + xr - 1) / xr;
    rc |= split_strided(parent_team, c, xr, size, yaxis_config, yaxis_mask, &t);
    if (t != NVSHMEM_TEAM_INVALID) *yaxis_team = t;
  }
  return rc;
}
NVSHMEM_EXPORT void nvshmem_team_get_config(nvshmem_team_t team, nvshmem_team_config_t* config) {
  State& s = st();
  if (!config || !initialized(s) || team < 0 || team >= s.max_teams || !s.teams[team].valid) return;
  *config = s.teams[team].config;
}
// Collective over the team: its members meet, then the index is free again.
NVSHMEM_EXPORT void nvshmem_team_destroy(nvshmem_team_t team) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (!initialized(s) || team < NVSHMEM_TEAMS_MIN || team >= s.max_teams || !s.teams[team].valid) return;
  cudaDeviceSynchronize();
  cudaGetLastError();
  team_barrier(s, team);
  const bool first = s.teams[team].my_pe == 0;
  drop_team(s, team);
  if (first) release_team_index(s, team);
}

// ============================================================================
// Collectives
// ============================================================================

NVSHMEM_EXPORT int nvshmem_broadcastmem(nvshmem_team_t team, void* dest, const void* src, size_t nelems,
                                        int PE_root) {
  cudaDeviceSynchronize();
  cudaGetLastError();
  return broadcast(team, dest, src, nelems, PE_root);
}
NVSHMEM_EXPORT int nvshmem_fcollectmem(nvshmem_team_t team, void* dest, const void* src, size_t nelems) {
  cudaDeviceSynchronize();
  cudaGetLastError();
  return fcollect(team, dest, src, nelems);
}
NVSHMEM_EXPORT int nvshmem_alltoallmem(nvshmem_team_t team, void* dest, const void* src, size_t nelems) {
  cudaDeviceSynchronize();
  cudaGetLastError();
  return alltoall(team, dest, src, nelems);
}
NVSHMEM_EXPORT int nvshmemx_broadcastmem_on_stream(nvshmem_team_t team, void* dest, const void* src, size_t nelems,
                                                   int PE_root, cudaStream_t stream) {
  in_stream_order(stream, [=] { broadcast(team, dest, src, nelems, PE_root); });
  return 0;
}
NVSHMEM_EXPORT int nvshmemx_fcollectmem_on_stream(nvshmem_team_t team, void* dest, const void* src, size_t nelems,
                                                  cudaStream_t stream) {
  in_stream_order(stream, [=] { fcollect(team, dest, src, nelems); });
  return 0;
}
NVSHMEM_EXPORT int nvshmemx_alltoallmem_on_stream(nvshmem_team_t team, void* dest, const void* src, size_t nelems,
                                                  cudaStream_t stream) {
  in_stream_order(stream, [=] { alltoall(team, dest, src, nelems); });
  return 0;
}

// ============================================================================
// The typed API: every type, nonblocking and stream-ordered forms, collectives,
// reductions, atomics
// ============================================================================

#include "nvshmem_typed_api.inc"
#undef VGPU_TYPED

// ============================================================================
// Kernels that use the device API
// ============================================================================

// Every block of a kernel that synchronizes across the grid must be resident:
// a cooperative launch.
NVSHMEM_EXPORT int nvshmemx_collective_launch(const void* func, dim3 gridDims, dim3 blockDims, void** args,
                                              size_t sharedMem, cudaStream_t stream) {
  const cudaError_t e = cudaLaunchCooperativeKernel(func, gridDims, blockDims, args, sharedMem, stream);
  if (e != cudaSuccess) {
    say("nvshmemx_collective_launch: %s", cudaGetErrorString(e));
    return 1;
  }
  return 0;
}
// The attribute form: a cudaLaunchConfig_t, made cooperative unless the caller
// said otherwise. Derived from the header's description.
NVSHMEM_EXPORT int nvshmemx_collective_launch_attr(const nvshmemx_collective_launch_attr_t* attr, const void* func, void** args) {
  if (!attr || !func) return 1;
  cudaLaunchConfig_t cfg = attr->cuda_config;
  std::vector<cudaLaunchAttribute> attrs(cfg.attrs, cfg.attrs + (cfg.attrs ? cfg.numAttrs : 0));
  bool has_coop = false;
  for (const auto& a : attrs) has_coop = has_coop || a.id == cudaLaunchAttributeCooperative;
  if (!has_coop) {
    cudaLaunchAttribute c{};
    c.id = cudaLaunchAttributeCooperative;
    c.val.cooperative = 1;
    attrs.push_back(c);
  }
  cfg.attrs = attrs.data();
  cfg.numAttrs = static_cast<unsigned>(attrs.size());
  const cudaError_t e = cudaLaunchKernelExC(&cfg, func, args);
  if (e != cudaSuccess) {
    say("nvshmemx_collective_launch_attr: %s", cudaGetErrorString(e));
    cudaGetLastError();
    return 1;
  }
  return 0;
}
NVSHMEM_EXPORT int nvshmemx_collective_launch_query_gridsize(const void* func, dim3 blockDims, void**,
                                                             size_t sharedMem, int* gridsize) {
  if (!gridsize) return 1;
  int per_sm = 0, sms = 0, dev = 0;
  cudaGetDevice(&dev);
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, func, int(blockDims.x * blockDims.y * blockDims.z),
                                                    sharedMem) != cudaSuccess ||
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
    cudaGetLastError();
    return 1;
  }
  *gridsize = per_sm * sms;
  return 0;
}

}  // extern "C"
