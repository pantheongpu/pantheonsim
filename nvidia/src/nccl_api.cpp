// libvgpunccl -- VirtualGPU's NCCL, presented as libnccl.so.2.
//
// Real NCCL moves bytes between physical GPUs over NVLink or the network.
// VirtualGPU has neither, so the transport here is a directory of memory-mapped
// files: each rank publishes its contribution to its own file, the ranks
// rendezvous on a shared metadata segment, and every rank then computes its own
// output from the peers' files. That is slow compared to NVLink and exactly as
// correct, which is the trade this simulator always makes.
//
// The file-backed transport is what makes the common deployment work: one rank
// per *process* (torchrun and friends), where the ranks share no address space.
// The single-process forms -- ncclCommInitAll, and one thread issuing every
// rank's call inside ncclGroupStart/End -- go through the same path, so there is
// only one implementation to get right.
//
// Communicators can be split (ncclCommSplit), shrunk (ncclCommShrink), grown
// (ncclCommGetUniqueId + ncclCommGrow), revoked, suspended and made
// non-blocking (ncclConfig_t.blocking = 0, whose work then runs on a background
// thread and is polled through ncclCommGetAsyncError). Pre-multiplied sums
// (ncclRedOpCreatePreMulSum) are the one kind of user-defined reduction NCCL
// has. Where the API leaves behaviour open, it was measured against NCCL 2.29.7
// and 2.31.2 on two RTX 3060s, one rank per GPU; those measurements are noted
// where they are used, as "card:".
//
// Not implemented, by design rather than by omission:
//  - symmetric memory windows. ncclCommWindowRegister succeeds and returns a
//    NULL window, which is what NCCL itself does on a machine without the
//    peer-to-peer mappings windows are built on (the RTX 3060 pair measured);
//    collectives on the buffer work as they always do. A real window would
//    promise device-side loads and stores into peer memory, which another
//    process's simulated device cannot offer through a file.
//  - the network plugin interface. A plugin is something NCCL dlopens to drive
//    a NIC; there is no network transport here for one to replace, so
//    NCCL_NET_PLUGIN and NCCL_NET are not read.
//  - the device API (ncclDevCommCreate and the kernels built on it). Its
//    load/store window into peer memory needs peer mappings, which a file does
//    not give another process's simulated device; ncclCommQueryProperties says
//    so (deviceApiSupport = false) and ncclDevCommCreate refuses, as NCCL does
//    on the RTX 3060 pair.
#include <nccl.h>

#include "vgpu/runtime/capture.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0) && defined(__has_include)
#if __has_include(<nccl_device/lsa_barrier.h>)
#define VGPU_NCCL_HAVE_DEVICE_API 1
// The device API's host half (team queries, communicator properties, the
// requirement helpers). Its kernels-side half is compiled by the user's nvcc.
#include <nccl_device/core.h>
#include <nccl_device/impl/gin_barrier__types.h>
#include <nccl_device/impl/ll_a2a__types.h>
#include <nccl_device/impl/lsa_barrier__types.h>
#endif
#endif

namespace {

constexpr int kMaxRanks = 64;          // a simulated rack, not a real cluster
constexpr uint32_t kMagic = 0x4e470756;  // "V\a GN"
constexpr uint32_t kVersion = 3;         // 2: the abort flag; 3: the grow join mask

bool quiet() { const char* q = std::getenv("VGPU_QUIET"); return q && q[0] == '1'; }

double timeout_seconds() {
  if (const char* t = std::getenv("VGPU_NCCL_TIMEOUT")) {
    double v = std::atof(t);
    if (v > 0) return v;
  }
  return 300.0;
}

// Shared rendezvous state. Every field a peer reads is atomic; the file is
// mapped MAP_SHARED by every rank, in this process or another.
struct Meta {
  uint32_t magic;
  uint32_t version;
  uint32_t nranks;
  std::atomic<uint32_t> aborted;  // a rank called ncclCommAbort: nobody waits for it any more
  std::atomic<uint32_t> joined;
  std::atomic<uint64_t> grow_mask;  // ncclCommGrow: ranks that joined this generation (a unique id is used once)
  std::atomic<uint64_t> phase[kMaxRanks];  // last collective this rank deposited
  std::atomic<uint64_t> done[kMaxRanks];   // last collective this rank finished reading
  std::atomic<uint64_t> posted[kMaxRanks][kMaxRanks];  // p2p [src][dst] messages sent
  std::atomic<uint64_t> taken[kMaxRanks][kMaxRanks];   // ... and received
  std::atomic<uint64_t> bytes[kMaxRanks][kMaxRanks];   // size of the message in flight
};

// The rendezvous lives in a file two processes map at different addresses, so
// every atomic in it has to be address-free -- a lock-backed atomic would
// synchronise the wrong thing entirely.
static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "NCCL rendezvous needs lock-free 64-bit atomics in shared memory");
static_assert(std::atomic<uint32_t>::is_always_lock_free,
              "NCCL rendezvous needs lock-free 32-bit atomics in shared memory");

std::string rendezvous_dir() {
  if (const char* d = std::getenv("VGPU_NCCL_DIR")) return d;
  const char* tmp = std::getenv("TMPDIR");
  return std::string(tmp && *tmp ? tmp : "/tmp") + "/vgpu-nccl-" + std::to_string(getuid());
}

std::string hex16(const ncclUniqueId& id) {
  static const char* h = "0123456789abcdef";
  std::string s;
  for (int i = 0; i < 16; ++i) {
    s += h[(unsigned char)id.internal[i] >> 4];
    s += h[(unsigned char)id.internal[i] & 0xf];
  }
  return s;
}

// A file mapped into this process, grown on demand. Writers own their own file
// so growth never races; readers map exactly the length they need, which the
// writer has already committed by the time the barrier releases them.
struct Mapping {
  int fd = -1;
  void* addr = nullptr;
  size_t len = 0;

  ~Mapping() { reset(); }
  void reset() {
    if (addr) munmap(addr, len);
    if (fd >= 0) close(fd);
    addr = nullptr; fd = -1; len = 0;
  }
  bool open_write(const std::string& path, size_t want) {
    if (fd < 0) {
      fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0600);
      if (fd < 0) return false;
    }
    if (want <= len && addr) return true;
    struct stat st{};
    if (fstat(fd, &st) != 0) return false;
    if ((size_t)st.st_size < want && ftruncate(fd, (off_t)want) != 0) return false;
    if (addr) { munmap(addr, len); addr = nullptr; }
    addr = mmap(nullptr, want, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) { addr = nullptr; return false; }
    len = want;
    return true;
  }
  bool open_read(const std::string& path, size_t want) {
    reset();
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st{};
    // The writer grew the file before releasing the barrier, so a short file
    // here means the barrier logic is wrong -- fail loudly instead of taking
    // SIGBUS on the first touch past EOF.
    if (fstat(fd, &st) != 0 || (size_t)st.st_size < want) { reset(); return false; }
    addr = mmap(nullptr, want, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) { addr = nullptr; reset(); return false; }
    len = want;
    return true;
  }
  template <class T> T* as() { return static_cast<T*>(addr); }
};

struct Rendezvous {
  std::string base;      // path prefix shared by every rank
  Mapping meta_map;
  Meta* meta = nullptr;
  std::mutex mu;         // guards the per-rank writer map below
  std::unordered_map<int, Mapping*> writers;   // rank -> my own data file
  // Communicators in this process still attached. The rendezvous is shared by
  // every rank running here, so it outlives any one of them -- but not all of
  // them, which is what this counts. It used to live until the process exited,
  // and LeakSanitizer was right to call that a leak.
  int refs = 0;
  ~Rendezvous() {
    for (auto& [k, v] : writers) delete v;
  }
  std::string rank_path(int r) const { return base + ".r" + std::to_string(r); }
  // One file per message rather than one per pair: a group may post several
  // sends to the same peer before any of them is received, and a single slot
  // would have the second overwrite the first.
  std::string p2p_path(int s, int d, uint64_t seq) const {
    return base + ".p" + std::to_string(s) + "-" + std::to_string(d) + "." + std::to_string(seq);
  }
};

// What a communicator was configured with, as far as anything here acts on it.
// The rest of ncclConfig_t tunes CTAs, channels and NVLS, none of which exist.
struct CommConfig {
  int blocking = 1;
};

// NCCL_COMM_BLOCKING sets the default for a config that leaves blocking unset.
int default_blocking() {
  const char* e = std::getenv("NCCL_COMM_BLOCKING");
  return (e && e[0] == '0') ? 0 : 1;
}

// A pre-multiplied sum (ncclRedOpCreatePreMulSum). The scalar is read at
// creation for ncclScalarHostImmediate and when the collective runs otherwise.
struct UserOp {
  bool live = false;
  ncclDataType_t dt = ncclFloat32;
  bool on_device = false;
  unsigned char host[8] = {};
  const void* dev = nullptr;
};

struct Comm {
  Rendezvous* rz = nullptr;
  int rank = 0;
  int nranks = 1;
  int cuda_dev = 0;
  uint64_t seq = 0;      // collectives issued on this communicator so far
  uint64_t shrinks = 0;  // ncclCommShrink calls, which name the child's rendezvous
  CommConfig cfg;
  bool finalized = false;
  // ncclCommGetAsyncError's answer: an error that outlives the call that found
  // it (a peer aborting, a non-blocking batch failing).
  std::atomic<int> state{ncclSuccess};
  std::atomic<int> pending{0};             // non-blocking batches still running
  std::atomic<bool> init_pending{false};   // non-blocking init, until every rank joined
  std::atomic<bool> aborted{false};
  std::atomic<bool> revoked{false};        // ncclCommRevoke: no further collectives or p2p
  std::atomic<bool> suspended{false};      // ncclCommSuspend(NCCL_SUSPEND_MEM) took effect
  uint64_t grows = 0;                      // ncclCommGrow generations this rank has taken part in
  std::mutex mu;                           // guards tail and redops
  std::shared_future<void> tail;           // the last background batch on this comm
  std::vector<UserOp> redops;              // see redop_handle
  uint32_t salt = 0;                       // this communicator's tag in its redop handles
};

// card: user operators' handles differ from one communicator to the next, so
// destroying one on the wrong communicator is caught rather than destroying
// that communicator's own. Handles here are ncclNumOps + (salt << 16) + index.
constexpr uint32_t kRedopSlots = 1u << 16;
std::atomic<uint32_t> g_next_salt{1};
ncclRedOp_t redop_handle(const Comm* c, size_t idx) {
  return static_cast<ncclRedOp_t>(ncclNumOps + ((c->salt << 16) | (uint32_t)idx));
}
// The slot `op` names on `c`, or -1.
long redop_slot(const Comm* c, ncclRedOp_t op) {
  if (op < ncclNumOps) return -1;
  const uint32_t v = (uint32_t)op - ncclNumOps;
  if ((v >> 16) != c->salt) return -1;
  const size_t idx = v & (kRedopSlots - 1);
  return idx < c->redops.size() && c->redops[idx].live ? (long)idx : -1;
}

std::mutex g_mu;
std::unordered_map<std::string, Rendezvous*> g_rendezvous;

// Revoke interrupts whatever a communicator is waiting for. The interrupted call
// is not an error (card: the stream of a revoked rank completes, and no call
// returns a failure for it), so the batch carries this marker out and
// execute() turns it back into ncclSuccess. It is a value inside the enum's
// range (ncclNumResults is 8, so the range is 0..15) that no NCCL call returns:
// a batch's result sits in enum-typed locals, and an out-of-range value there
// is undefined behaviour that UBSan reports.
constexpr ncclResult_t kRevokedWait = static_cast<ncclResult_t>(15);
static_assert(ncclNumResults < 15, "kRevokedWait must not be a real result");

bool mkdir_p(const std::string& p) {
  if (::mkdir(p.c_str(), 0700) == 0 || errno == EEXIST) return true;
  return false;
}

// A child communicator (split, shrink, a scalable init) has no unique id of its
// own; every member derives the same rendezvous name from what they agree on.
void hash128(const std::string& key, uint64_t out[2]) {
  auto fnv = [&](uint64_t h) {
    for (unsigned char ch : key) { h ^= ch; h *= 0x100000001b3ull; }
    return h;
  };
  out[0] = fnv(0xcbf29ce484222325ull);
  out[1] = fnv(0x84222325cbf29ce4ull);
}

std::string derived_base(const std::string& dir, const std::string& key) {
  uint64_t h[2];
  hash128(key, h);
  char buf[40];
  std::snprintf(buf, sizeof buf, "%016llx%016llx", (unsigned long long)h[0], (unsigned long long)h[1]);
  return dir + "/" + buf;
}

std::string dir_of(const std::string& base) { return base.substr(0, base.rfind('/')); }

// Create the metadata file exactly once, whichever rank gets there first.
// Build it under a private name and link() it into place: link fails with
// EEXIST rather than truncating, so no rank can ever see a half-built segment.
Rendezvous* attach(const std::string& base, int nranks, std::string* err) {
  if (!mkdir_p(dir_of(base))) { *err = "cannot create " + dir_of(base); return nullptr; }
  const std::string meta_path = base + ".meta";

  std::lock_guard<std::mutex> lock(g_mu);
  if (auto it = g_rendezvous.find(base); it != g_rendezvous.end()) {
    ++it->second->refs;
    return it->second;
  }

  struct stat st{};
  if (stat(meta_path.c_str(), &st) != 0) {
    const std::string tmp = meta_path + "." + std::to_string(getpid());
    int fd = ::open(tmp.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) { *err = "cannot create " + tmp; return nullptr; }
    if (ftruncate(fd, (off_t)sizeof(Meta)) != 0) { close(fd); unlink(tmp.c_str()); *err = "ftruncate"; return nullptr; }
    void* p = mmap(nullptr, sizeof(Meta), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { close(fd); unlink(tmp.c_str()); *err = "mmap"; return nullptr; }
    auto* m = static_cast<Meta*>(p);
    std::memset(m, 0, sizeof(Meta));
    m->version = kVersion;
    m->nranks = (uint32_t)nranks;
    m->magic = kMagic;  // last: a reader that sees the magic sees a complete file
    msync(p, sizeof(Meta), MS_SYNC);
    munmap(p, sizeof(Meta));
    close(fd);
    if (link(tmp.c_str(), meta_path.c_str()) != 0 && errno != EEXIST) {
      unlink(tmp.c_str());
      *err = "cannot publish " + meta_path;
      return nullptr;
    }
    unlink(tmp.c_str());
  }

  auto* rz = new Rendezvous();
  rz->base = base;
  if (!rz->meta_map.open_write(meta_path, sizeof(Meta))) {
    delete rz; *err = "cannot map " + meta_path; return nullptr;
  }
  rz->meta = rz->meta_map.as<Meta>();
  if (rz->meta->magic != kMagic || rz->meta->version != kVersion) {
    delete rz; *err = "rendezvous file is not a VirtualGPU NCCL segment"; return nullptr;
  }
  if ((int)rz->meta->nranks != nranks) {
    *err = "rank count disagrees: this rendezvous was created with " +
           std::to_string(rz->meta->nranks) + " ranks, this rank passed " + std::to_string(nranks);
    delete rz;
    return nullptr;
  }
  rz->refs = 1;
  g_rendezvous[base] = rz;
  return rz;
}

void detach(Rendezvous* rz) {
  if (!rz) return;
  std::lock_guard<std::mutex> lock(g_mu);
  if (--rz->refs <= 0) {
    g_rendezvous.erase(rz->base);
    delete rz;  // takes its per-rank mappings with it
  }
}

// A communicator on the rendezvous at `base`, joined as `rank`.
Comm* make_comm(const std::string& base, int nranks, int rank, int dev, const CommConfig& cfg,
                std::string* err, bool grow = false) {
  Rendezvous* rz = attach(base, nranks, err);
  if (!rz) return nullptr;
  if (grow) {
    // A grow's unique id is good for one generation: a rank that joins twice is
    // using an id that was already consumed, or a rank number that is taken.
    const uint64_t bit = 1ull << rank;
    if (rz->meta->grow_mask.fetch_or(bit, std::memory_order_acq_rel) & bit) {
      *err = "rank " + std::to_string(rank) + " has already joined this grow: the unique id was "
             "consumed, or the rank is in use";
      detach(rz);
      return nullptr;
    }
  }
  auto* c = new Comm();
  c->rz = rz;
  c->rank = rank;
  c->nranks = nranks;
  c->cuda_dev = dev;
  c->cfg = cfg;
  // 15 bits of salt keep every handle below ncclMaxRedOp; past 32767
  // communicators in one process the tags repeat, which only weakens the check.
  c->salt = 1 + (g_next_salt.fetch_add(1, std::memory_order_relaxed) - 1) % 0x7ffe;
  rz->meta->joined.fetch_add(1, std::memory_order_release);
  return c;
}

// A non-blocking init is complete once every rank has joined.
void refresh_init(Comm* c) {
  if (c->init_pending.load(std::memory_order_acquire) &&
      (int)c->rz->meta->joined.load(std::memory_order_acquire) >= c->nranks)
    c->init_pending.store(false, std::memory_order_release);
}

// card: on a non-blocking communicator whose last operation has not completed,
// every call -- ncclCommCount included -- fails with ncclInvalidArgument
// ("Attempt to use communicator before the previous operation returned
// ncclSuccess"). A collective (`poison`) also leaves that as the
// communicator's async error; ncclCommCount does not.
bool busy(Comm* c, bool poison = false) {
  if (c->cfg.blocking) return false;
  refresh_init(c);
  if (c->pending.load(std::memory_order_acquire) == 0 &&
      !c->init_pending.load(std::memory_order_acquire))
    return false;
  std::fprintf(stderr,
               "[vgpu] nccl: rank %d: communicator used before its previous operation returned "
               "ncclSuccess (poll ncclCommGetAsyncError first)\n", c->rank);
  if (poison) c->state.store(ncclInvalidArgument, std::memory_order_release);
  return true;
}

// Reads a caller's ncclConfig_t. A NULL config inherits `parent`'s, as NCCL
// documents for ncclCommSplit and ncclCommShrink, or takes the defaults.
ncclResult_t parse_config(const ncclConfig_t* in, const CommConfig* parent, CommConfig* out) {
  if (!in) {
    *out = parent ? *parent : CommConfig{default_blocking()};
    return ncclSuccess;
  }
  // card: a config not from NCCL_CONFIG_INITIALIZER is ncclInvalidArgument, and
  // so is blocking = 5 ("Invalid config blocking attribute value 5").
  if (in->magic != NCCL_API_MAGIC || in->size < offsetof(ncclConfig_t, blocking) + sizeof(int)) {
    std::fprintf(stderr, "[vgpu] nccl: ncclConfig_t argument not initialized via NCCL_CONFIG_INITIALIZER\n");
    return ncclInvalidArgument;
  }
  int b = in->blocking;
  if (b == NCCL_CONFIG_UNDEF_INT) b = default_blocking();
  if (b != 0 && b != 1) {
    std::fprintf(stderr, "[vgpu] nccl: invalid config blocking attribute value %d\n", b);
    return ncclInvalidArgument;
  }
  out->blocking = b;
  return ncclSuccess;
}

// Spin-wait with a deadline. Collectives that never match up are the single
// most common NCCL bug, so time out with a diagnosis rather than hanging. An
// abort -- this communicator's, or a peer's -- ends the wait early.
// `interruptible` is whether a revoke ends the wait: it does for data movement,
// not for the management calls (split, suspend, resume) a revoked communicator
// still takes part in.
template <class Pred>
ncclResult_t wait_for(Pred done, const char* what, Comm* c, bool interruptible = true) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_seconds());
  int spins = 0;
  while (!done()) {
    if (++spins > 512) {
      timespec ts{0, 200000};  // 0.2 ms
      nanosleep(&ts, nullptr);
      if (c->aborted.load(std::memory_order_acquire)) return ncclInvalidUsage;
      if (interruptible && c->revoked.load(std::memory_order_acquire)) return kRevokedWait;
      if (c->rz->meta->aborted.load(std::memory_order_acquire)) {
        if (done()) break;
        std::fprintf(stderr, "[vgpu] nccl: rank %d gave up waiting for %s: a peer aborted the "
                             "communicator\n", c->rank, what);
        c->state.store(ncclRemoteError, std::memory_order_release);
        return ncclRemoteError;
      }
      if (std::chrono::steady_clock::now() > deadline) {
        std::fprintf(stderr,
                     "[vgpu] nccl: rank %d timed out waiting for %s after %.0fs. Ranks must "
                     "call the same collectives in the same order; set VGPU_NCCL_TIMEOUT to "
                     "raise the limit.\n",
                     c->rank, what, timeout_seconds());
        c->state.store(ncclTimeout, std::memory_order_release);
        return ncclTimeout;
      }
    }
  }
  return ncclSuccess;
}

size_t type_size(ncclDataType_t t) {
  switch (t) {
    case ncclInt8: case ncclUint8: case ncclFloat8e4m3: case ncclFloat8e5m2: return 1;
    case ncclFloat16: case ncclBfloat16: return 2;
    case ncclInt32: case ncclUint32: case ncclFloat32: return 4;
    case ncclInt64: case ncclUint64: case ncclFloat64: return 8;
    default: return 0;
  }
}

/* ---- reduced-precision conversions ----
   Through the vendor headers' own host-callable conversions rather than hand
   written bit twiddling. Four formats' worth of rounding rules is exactly the
   kind of thing that has silently produced wrong numbers in this repository
   before, and the correct version already ships with the toolkit. */

float half_to_float(uint16_t bits) {
  __half v;
  std::memcpy(&v, &bits, sizeof v);
  return __half2float(v);
}
uint16_t float_to_half(float f) {
  const __half v = __float2half(f);
  uint16_t bits;
  std::memcpy(&bits, &v, sizeof bits);
  return bits;
}
float bf16_to_float(uint16_t bits) {
  __nv_bfloat16 v;
  std::memcpy(&v, &bits, sizeof v);
  return __bfloat162float(v);
}
uint16_t float_to_bf16(float f) {
  const __nv_bfloat16 v = __float2bfloat16(f);
  uint16_t bits;
  std::memcpy(&bits, &v, sizeof bits);
  return bits;
}
float e4m3_to_f(uint8_t bits) {
  __nv_fp8_e4m3 v;
  v.__x = bits;
  return static_cast<float>(v);
}
uint8_t f_to_e4m3(float f) { return static_cast<uint8_t>(__nv_fp8_e4m3(f).__x); }
float e5m2_to_f(uint8_t bits) {
  __nv_fp8_e5m2 v;
  v.__x = bits;
  return static_cast<float>(v);
}
uint8_t f_to_e5m2(float f) { return static_cast<uint8_t>(__nv_fp8_e5m2(f).__x); }

/* ---- reductions ---- */

template <class T>
void reduce_typed(void* acc, const void* in, size_t n, ncclRedOp_t op) {
  T* a = static_cast<T*>(acc);
  const T* b = static_cast<const T*>(in);
  for (size_t i = 0; i < n; ++i) {
    switch (op) {
      case ncclProd: a[i] = a[i] * b[i]; break;
      case ncclMax: a[i] = b[i] > a[i] ? b[i] : a[i]; break;
      case ncclMin: a[i] = b[i] < a[i] ? b[i] : a[i]; break;
      default: a[i] = a[i] + b[i]; break;  // Sum, and Avg before the divide
    }
  }
}

template <class Raw, float (*ToF)(Raw), Raw (*FromF)(float)>
void reduce_narrow(void* acc, const void* in, size_t n, ncclRedOp_t op) {
  Raw* a = static_cast<Raw*>(acc);
  const Raw* b = static_cast<const Raw*>(in);
  for (size_t i = 0; i < n; ++i) {
    const float x = ToF(a[i]), y = ToF(b[i]);
    float r;
    switch (op) {
      case ncclProd: r = x * y; break;
      case ncclMax: r = y > x ? y : x; break;
      case ncclMin: r = y < x ? y : x; break;
      default: r = x + y; break;
    }
    a[i] = FromF(r);
  }
}

bool reduce(void* acc, const void* in, size_t n, ncclDataType_t dt, ncclRedOp_t op) {
  switch (dt) {
    case ncclInt8: reduce_typed<int8_t>(acc, in, n, op); return true;
    case ncclUint8: reduce_typed<uint8_t>(acc, in, n, op); return true;
    case ncclInt32: reduce_typed<int32_t>(acc, in, n, op); return true;
    case ncclUint32: reduce_typed<uint32_t>(acc, in, n, op); return true;
    case ncclInt64: reduce_typed<int64_t>(acc, in, n, op); return true;
    case ncclUint64: reduce_typed<uint64_t>(acc, in, n, op); return true;
    case ncclFloat32: reduce_typed<float>(acc, in, n, op); return true;
    case ncclFloat64: reduce_typed<double>(acc, in, n, op); return true;
    case ncclFloat16: reduce_narrow<uint16_t, half_to_float, float_to_half>(acc, in, n, op); return true;
    case ncclBfloat16: reduce_narrow<uint16_t, bf16_to_float, float_to_bf16>(acc, in, n, op); return true;
    case ncclFloat8e4m3: reduce_narrow<uint8_t, e4m3_to_f, f_to_e4m3>(acc, in, n, op); return true;
    case ncclFloat8e5m2: reduce_narrow<uint8_t, e5m2_to_f, f_to_e5m2>(acc, in, n, op); return true;
    default: return false;
  }
}

template <class T> void scale_typed(void* p, size_t n, double s) {
  T* a = static_cast<T*>(p);
  for (size_t i = 0; i < n; ++i) a[i] = static_cast<T>(a[i] * s);
}

template <class Raw, float (*ToF)(Raw), Raw (*FromF)(float)>
void scale_narrow(void* p, size_t n, double s) {
  Raw* a = static_cast<Raw*>(p);
  for (size_t i = 0; i < n; ++i) a[i] = FromF(static_cast<float>(ToF(a[i]) * s));
}

// ncclAvg is a sum followed by a divide by the rank count.
void average(void* p, size_t n, ncclDataType_t dt, int nranks) {
  const double s = 1.0 / nranks;
  switch (dt) {
    case ncclInt8: scale_typed<int8_t>(p, n, s); break;
    case ncclUint8: scale_typed<uint8_t>(p, n, s); break;
    case ncclInt32: scale_typed<int32_t>(p, n, s); break;
    case ncclUint32: scale_typed<uint32_t>(p, n, s); break;
    case ncclInt64: scale_typed<int64_t>(p, n, s); break;
    case ncclUint64: scale_typed<uint64_t>(p, n, s); break;
    case ncclFloat32: scale_typed<float>(p, n, s); break;
    case ncclFloat64: scale_typed<double>(p, n, s); break;
    case ncclFloat16: scale_narrow<uint16_t, half_to_float, float_to_half>(p, n, s); break;
    case ncclBfloat16: scale_narrow<uint16_t, bf16_to_float, float_to_bf16>(p, n, s); break;
    case ncclFloat8e4m3: scale_narrow<uint8_t, e4m3_to_f, f_to_e4m3>(p, n, s); break;
    case ncclFloat8e5m2: scale_narrow<uint8_t, e5m2_to_f, f_to_e5m2>(p, n, s); break;
    default: break;
  }
}

/* ---- pre-multiplied sums ----
   Each rank's input is multiplied by that rank's own scalar, then summed.
   card (NCCL 2.29.7, two ranks): fp64 results are a chain of fused multiply-adds
   in rank order -- acc = x0*s0 rounded, then acc = fma(x_r, s_r, acc) -- for
   every element. fp32 matches that chain on one of the ring's two chunks and the
   same chain started at the other rank on the other: the ring's chunking picks
   the order, and no implementation without it can reproduce both. fp16 rounds
   each product to fp16 before summing, for every element. Integers wrap. */

template <class T>
void premul_float(void* acc, const void* in, size_t n, const void* sc, bool first) {
  T* a = static_cast<T*>(acc);
  const T* x = static_cast<const T*>(in);
  T s;
  std::memcpy(&s, sc, sizeof s);
  for (size_t i = 0; i < n; ++i) a[i] = first ? x[i] * s : std::fma(x[i], s, a[i]);
}

template <class T>
void premul_int(void* acc, const void* in, size_t n, const void* sc, bool first) {
  using U = std::make_unsigned_t<T>;
  T* a = static_cast<T*>(acc);
  const T* x = static_cast<const T*>(in);
  T s;
  std::memcpy(&s, sc, sizeof s);
  for (size_t i = 0; i < n; ++i) {
    const U p = static_cast<U>(static_cast<U>(x[i]) * static_cast<U>(s));
    a[i] = static_cast<T>(first ? p : static_cast<U>(static_cast<U>(a[i]) + p));
  }
}

template <class Raw, float (*ToF)(Raw), Raw (*FromF)(float)>
void premul_narrow(void* acc, const void* in, size_t n, const void* sc, bool first) {
  Raw* a = static_cast<Raw*>(acc);
  const Raw* x = static_cast<const Raw*>(in);
  Raw s;
  std::memcpy(&s, sc, sizeof s);
  const float sf = ToF(s);
  for (size_t i = 0; i < n; ++i) {
    const Raw p = FromF(ToF(x[i]) * sf);
    a[i] = first ? p : FromF(ToF(a[i]) + ToF(p));
  }
}

// acc = in * scalar when `first`, acc += in * scalar after that.
bool premul(void* acc, const void* in, size_t n, ncclDataType_t dt, const void* sc, bool first) {
  switch (dt) {
    case ncclInt8: premul_int<int8_t>(acc, in, n, sc, first); return true;
    case ncclUint8: premul_int<uint8_t>(acc, in, n, sc, first); return true;
    case ncclInt32: premul_int<int32_t>(acc, in, n, sc, first); return true;
    case ncclUint32: premul_int<uint32_t>(acc, in, n, sc, first); return true;
    case ncclInt64: premul_int<int64_t>(acc, in, n, sc, first); return true;
    case ncclUint64: premul_int<uint64_t>(acc, in, n, sc, first); return true;
    case ncclFloat32: premul_float<float>(acc, in, n, sc, first); return true;
    case ncclFloat64: premul_float<double>(acc, in, n, sc, first); return true;
    case ncclFloat16: premul_narrow<uint16_t, half_to_float, float_to_half>(acc, in, n, sc, first); return true;
    case ncclBfloat16: premul_narrow<uint16_t, bf16_to_float, float_to_bf16>(acc, in, n, sc, first); return true;
    case ncclFloat8e4m3: premul_narrow<uint8_t, e4m3_to_f, f_to_e4m3>(acc, in, n, sc, first); return true;
    case ncclFloat8e5m2: premul_narrow<uint8_t, e5m2_to_f, f_to_e5m2>(acc, in, n, sc, first); return true;
    default: return false;
  }
}

/* ---- group buffering ----
   NCCL lets one thread issue every rank's call between ncclGroupStart and
   ncclGroupEnd; the calls only have to match up by the time the group closes.
   A barrier taken inside the call itself would deadlock that pattern, so the
   ops are recorded and run in four passes at ncclGroupEnd -- deposit all,
   collect all, release all, wait for all -- and each pass finishes for every
   op before the next begins. A call made outside a group is just a group of
   one, which reduces to deposit-collect-release-wait in order.

   A group on non-blocking communicators runs the same four passes on a
   background thread instead, and the call returns ncclInProgress. */

enum class Kind {
  AllReduce, Broadcast, Reduce, AllGather, ReduceScatter, AlltoAll, Gather, Scatter,
  Send, Recv,
  Split,    // a collective on the parent: every rank publishes (color, key)
  Shrink,   // local: the surviving ranks already agree on who survives
  Suspend,  // a barrier on the communicator; the state changes once every rank is in
  Resume,
};

struct Op {
  Kind kind;
  Comm* comm;
  const void* send;
  void* recv;
  size_t count;          // elements: per rank for AllGather/AlltoAll/Gather/Scatter, per-rank output for ReduceScatter
  // Kept as ints: a caller may pass a value that is no member of the enum
  // (NCCL answers ncclInvalidArgument), and copying such an enum is undefined
  // behavior that UBSan reports. enqueue() checks both before they are used.
  int dt;
  int red;
  ncclDataType_t type() const { return static_cast<ncclDataType_t>(dt); }
  ncclRedOp_t rop() const { return static_cast<ncclRedOp_t>(red); }
  int root = 0;
  int peer = 0;
  cudaStream_t stream = nullptr;
  uint64_t seq = 0;
  std::vector<char> staging{};   // host copy of the send buffer / assembled output
  bool ok = true;
  // A pre-multiplied sum carries its scalar; the rank file gets it as a trailer
  // after the data so every reducer can apply every rank's own.
  bool premul = false;
  UserOp scalar{};
  // Split and Shrink.
  int32_t color_key[2] = {0, 0};
  ncclComm_t* out = nullptr;
  CommConfig child{};
  std::vector<int> exclude{};
  uint64_t shrink_epoch = 0;
};

// True while a graph launch runs a recorded operation (see capture_op): the stream it was issued on is the
// graph's own, and waiting for it from inside the graph would wait for the graph.
thread_local bool t_replay = false;
thread_local int t_group_depth = 0;
thread_local std::vector<Op> t_pending;
// card: an argument error on a split or a collective inside a group is also
// what ncclGroupEnd returns, and the group does not run.
thread_local ncclResult_t t_group_error = ncclSuccess;

ncclResult_t fail_call(ncclResult_t r) {
  if (t_group_depth > 0 && t_group_error == ncclSuccess) t_group_error = r;
  return r;
}

bool is_collective(Kind k) { return k != Kind::Send && k != Kind::Recv && k != Kind::Shrink; }

// What moves data: the operations a revoked or suspended communicator refuses.
bool moves_data(Kind k) {
  return k != Kind::Split && k != Kind::Shrink && k != Kind::Suspend && k != Kind::Resume;
}

const char* kind_name(Kind k) {
  switch (k) {
    case Kind::AllReduce: return "AllReduce";
    case Kind::Broadcast: return "Broadcast";
    case Kind::Reduce: return "Reduce";
    case Kind::AllGather: return "AllGather";
    case Kind::ReduceScatter: return "ReduceScatter";
    case Kind::AlltoAll: return "AlltoAll";
    case Kind::Gather: return "Gather";
    case Kind::Scatter: return "Scatter";
    case Kind::Send: return "Send";
    case Kind::Recv: return "Recv";
    case Kind::Split: return "Split";
    case Kind::Shrink: return "Shrink";
    case Kind::Suspend: return "Suspend";
    case Kind::Resume: return "Resume";
  }
  return "?";
}

// Every device access has to happen with the communicator's device current:
// inside a group, one thread touches several devices in a row.
struct DeviceGuard {
  int prev = 0;
  explicit DeviceGuard(int dev) {
    cudaGetDevice(&prev);
    if (prev != dev) cudaSetDevice(dev);
  }
  ~DeviceGuard() { int cur = 0; cudaGetDevice(&cur); if (cur != prev) cudaSetDevice(prev); }
};

Mapping* writer_for(Rendezvous* rz, int rank, size_t want) {
  std::lock_guard<std::mutex> l(rz->mu);
  auto it = rz->writers.find(rank);
  if (it == rz->writers.end()) it = rz->writers.emplace(rank, new Mapping()).first;
  return it->second->open_write(rz->rank_path(rank), want) ? it->second : nullptr;
}

// How many bytes of data this rank publishes for the given op (not counting a
// pre-multiplied sum's scalar trailer).
size_t deposit_bytes(const Op& op) {
  const size_t es = type_size(op.type());
  const size_t n = (size_t)op.comm->nranks;
  const bool root = op.comm->rank == op.root;
  switch (op.kind) {
    case Kind::AllGather: case Kind::Gather: return op.count * es;        // each rank's slice
    case Kind::ReduceScatter: case Kind::AlltoAll: return op.count * n * es;  // whole input
    case Kind::Broadcast: return root ? op.count * es : 0;
    case Kind::Scatter: return root ? op.count * n * es : 0;
    case Kind::Split: return sizeof op.color_key;
    case Kind::Shrink: case Kind::Recv: case Kind::Suspend: case Kind::Resume: return 0;
    default: return op.count * es;
  }
}

ncclResult_t deposit(Op& op) {
  Comm* c = op.comm;
  const size_t bytes = deposit_bytes(op);
  if (op.kind == Kind::Recv || op.kind == Kind::Shrink) return ncclSuccess;  // nothing to publish

  if (op.kind == Kind::Send) {
    // The message number this send is: the file it writes is named for it, so
    // a second send to the same peer cannot overwrite an unread first.
    const uint64_t seq =
        c->rz->meta->posted[c->rank][op.peer].load(std::memory_order_acquire) + 1;
    Mapping m;
    if (!m.open_write(c->rz->p2p_path(c->rank, op.peer, seq), bytes ? bytes : 1))
      return ncclSystemError;
    {
      DeviceGuard g(c->cuda_dev);
      if (!t_replay) cudaStreamSynchronize(op.stream);
      if (bytes && cudaMemcpy(m.addr, op.send, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        return ncclUnhandledCudaError;
    }
    msync(m.addr, m.len, MS_SYNC);
    m.reset();   // close before publishing, so a reader never sees a short file
    c->rz->meta->bytes[c->rank][op.peer].store(bytes, std::memory_order_release);
    c->rz->meta->posted[c->rank][op.peer].store(seq, std::memory_order_release);
    return ncclSuccess;
  }

  if (op.kind == Kind::Split) {
    Mapping* m = writer_for(c->rz, c->rank, bytes);
    if (!m) return ncclSystemError;
    std::memcpy(m->addr, op.color_key, bytes);
  } else if (bytes) {
    const size_t es = type_size(op.type());
    const size_t total = bytes + (op.premul ? es : 0);
    Mapping* m = writer_for(c->rz, c->rank, total);
    if (!m) return ncclSystemError;
    DeviceGuard g(c->cuda_dev);
    if (!t_replay) cudaStreamSynchronize(op.stream);
    if (cudaMemcpy(m->addr, op.send, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
      return ncclUnhandledCudaError;
    if (op.premul) {
      // ncclScalarDevice is read now, while the collective runs, as documented.
      char* trailer = static_cast<char*>(m->addr) + bytes;
      if (!op.scalar.on_device) std::memcpy(trailer, op.scalar.host, es);
      else if (cudaMemcpy(trailer, op.scalar.dev, es, cudaMemcpyDefault) != cudaSuccess)
        return ncclUnhandledCudaError;
    }
    msync(m->addr, m->len, MS_ASYNC);
  } else if (op.kind != Kind::Suspend && op.kind != Kind::Resume) {
    DeviceGuard g(c->cuda_dev);
    if (!t_replay) cudaStreamSynchronize(op.stream);
  }
  c->rz->meta->phase[c->rank].store(op.seq, std::memory_order_release);
  return ncclSuccess;
}

// Read `bytes` of a peer's published file starting at `offset`. For the local
// rank we could reuse our own mapping, but reading the file keeps one code path
// and one set of bugs.
bool read_peer(Rendezvous* rz, int rank, size_t bytes, std::vector<char>* out, size_t offset = 0) {
  out->resize(bytes);
  if (!bytes) return true;
  Mapping m;
  if (!m.open_read(rz->rank_path(rank), offset + bytes)) return false;
  std::memcpy(out->data(), static_cast<const char*>(m.addr) + offset, bytes);
  return true;
}

// Sum of every rank's first `n` elements, each multiplied by its own scalar
// (the trailer at byte `n * es` of its file).
ncclResult_t premul_reduce(Comm* c, size_t n, ncclDataType_t dt, std::vector<char>* acc) {
  const size_t es = type_size(dt);
  std::vector<char> peer;
  acc->resize(n * es);
  for (int r = 0; r < c->nranks; ++r) {
    if (!read_peer(c->rz, r, n * es + es, &peer)) return ncclSystemError;
    if (!premul(acc->data(), peer.data(), n, dt, peer.data() + n * es, r == 0))
      return ncclInvalidArgument;
  }
  return ncclSuccess;
}

// A child communicator for this rank, named from what every member agrees on.
ncclResult_t make_child(Op& op, const std::string& key, int nranks, int rank) {
  Comm* p = op.comm;
  std::string err;
  Comm* child = make_comm(derived_base(dir_of(p->rz->base), p->rz->base + key), nranks, rank,
                          p->cuda_dev, op.child, &err);
  if (!child) {
    std::fprintf(stderr, "[vgpu] nccl: %s\n", err.c_str());
    return ncclSystemError;
  }
  *op.out = reinterpret_cast<ncclComm_t>(child);
  return ncclSuccess;
}

ncclResult_t collect(Op& op) {
  Comm* c = op.comm;
  Meta* meta = c->rz->meta;
  const size_t es = type_size(op.type());
  const size_t n = (size_t)c->nranks;

  if (op.kind == Kind::Send) return ncclSuccess;

  if (op.kind == Kind::Shrink) {
    // New ranks close the gaps the excluded ones leave, in the old order.
    int newrank = c->rank;
    for (int x : op.exclude) if (x < c->rank) --newrank;
    std::string key = "|shrink|" + std::to_string(op.shrink_epoch);
    for (int x : op.exclude) key += "," + std::to_string(x);
    return make_child(op, key, c->nranks - (int)op.exclude.size(), newrank);
  }

  if (op.kind == Kind::Recv) {
    const size_t want_msgs = meta->taken[op.peer][c->rank].load(std::memory_order_acquire) + 1;
    if (ncclResult_t r = wait_for([&] { return meta->posted[op.peer][c->rank].load(std::memory_order_acquire) >= want_msgs; },
                                  "a matching ncclSend", c); r != ncclSuccess)
      return r;
    const size_t bytes = op.count * es;
    if (bytes == 0) {   // an empty message: nothing to map (mmap refuses length 0)
      unlink(c->rz->p2p_path(op.peer, c->rank, want_msgs).c_str());
      meta->taken[op.peer][c->rank].store(want_msgs, std::memory_order_release);
      return ncclSuccess;
    }
    Mapping m;
    // The file's own length is the message length, so a size disagreement is
    // caught here rather than trusted from a shared counter that a later send
    // may already have moved on.
    const std::string path = c->rz->p2p_path(op.peer, c->rank, want_msgs);
    if (!m.open_read(path, bytes)) {
      std::fprintf(stderr,
                   "[vgpu] nccl: rank %d expected %zu bytes as message %llu from rank %d, and the "
                   "sender did not write that much\n",
                   c->rank, bytes, (unsigned long long)want_msgs, op.peer);
      return ncclInvalidUsage;
    }
    {
      DeviceGuard g(c->cuda_dev);
      if (cudaMemcpy(op.recv, m.addr, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        return ncclUnhandledCudaError;
    }
    m.reset();
    unlink(path.c_str());   // the message has been consumed; do not accumulate files
    meta->taken[op.peer][c->rank].store(want_msgs, std::memory_order_release);
    return ncclSuccess;
  }

  // Collectives: everyone must have deposited before anyone reads.
  if (ncclResult_t r = wait_for([&] {
        for (int i = 0; i < c->nranks; ++i)
          if (meta->phase[i].load(std::memory_order_acquire) < op.seq) return false;
        return true;
      }, "the other ranks to reach this collective", c, moves_data(op.kind)); r != ncclSuccess)
    return r;
  if (op.kind == Kind::Suspend || op.kind == Kind::Resume) {
    // card: both are collective -- a rank's call returns once every rank has
    // made it (measured: one rank waited exactly as long as the other was late).
    c->suspended.store(op.kind == Kind::Suspend, std::memory_order_release);
    return ncclSuccess;
  }
  // A collective of zero elements still synchronizes, and moves nothing. Going
  // on would hand memcpy the null data() of empty buffers, which UBSan rightly
  // calls undefined, and look for a pre-multiplied sum's scalar after data
  // that was never written.
  if (op.count == 0 && op.kind != Kind::Split) return ncclSuccess;

  std::vector<char> peer;
  switch (op.kind) {
    case Kind::Split: {
      // card: ranks are ordered by key, ties by their old rank; any color but
      // NCCL_SPLIT_NOCOLOR (-1) is a color, negative ones included.
      const int my_color = op.color_key[0];
      std::vector<std::pair<int32_t, int>> members;  // (key, old rank)
      for (int r = 0; r < c->nranks; ++r) {
        if (!read_peer(c->rz, r, sizeof op.color_key, &peer)) return ncclSystemError;
        int32_t ck[2];
        std::memcpy(ck, peer.data(), sizeof ck);
        if (ck[0] == my_color) members.emplace_back(ck[1], r);
      }
      if (my_color == NCCL_SPLIT_NOCOLOR) return ncclSuccess;  // *out stays NULL
      std::sort(members.begin(), members.end());
      int newrank = 0;
      while (members[newrank].second != c->rank) ++newrank;
      return make_child(op, "|split|" + std::to_string(op.seq) + "|" + std::to_string(my_color),
                        (int)members.size(), newrank);
    }
    case Kind::Broadcast: {
      if (!read_peer(c->rz, op.root, op.count * es, &op.staging)) return ncclSystemError;
      break;
    }
    case Kind::AllGather: case Kind::Gather: {
      if (op.kind == Kind::Gather && c->rank != op.root) break;  // only the root receives
      op.staging.resize(op.count * es * n);
      for (int r = 0; r < c->nranks; ++r) {
        if (!read_peer(c->rz, r, op.count * es, &peer)) return ncclSystemError;
        std::memcpy(op.staging.data() + (size_t)r * op.count * es, peer.data(), op.count * es);
      }
      break;
    }
    case Kind::AlltoAll: {
      // Block j of rank i's input lands as block i of rank j's output.
      op.staging.resize(op.count * es * n);
      for (int r = 0; r < c->nranks; ++r) {
        if (!read_peer(c->rz, r, op.count * es, &peer, (size_t)c->rank * op.count * es))
          return ncclSystemError;
        std::memcpy(op.staging.data() + (size_t)r * op.count * es, peer.data(), op.count * es);
      }
      break;
    }
    case Kind::Scatter: {
      if (!read_peer(c->rz, op.root, op.count * es, &op.staging, (size_t)c->rank * op.count * es))
        return ncclSystemError;
      break;
    }
    case Kind::ReduceScatter: {
      // Reduce every rank's whole input, then keep this rank's slice.
      const size_t total = op.count * n;
      std::vector<char> acc;
      if (op.premul) {
        if (ncclResult_t r = premul_reduce(c, total, op.type(), &acc); r != ncclSuccess) return r;
      } else {
        if (!read_peer(c->rz, 0, total * es, &acc)) return ncclSystemError;
        for (int r = 1; r < c->nranks; ++r) {
          if (!read_peer(c->rz, r, total * es, &peer)) return ncclSystemError;
          if (!reduce(acc.data(), peer.data(), total, op.type(), op.rop())) return ncclInvalidArgument;
        }
        if (op.red == ncclAvg) average(acc.data(), total, op.type(), c->nranks);
      }
      op.staging.assign(acc.begin() + (size_t)c->rank * op.count * es,
                        acc.begin() + (size_t)(c->rank + 1) * op.count * es);
      break;
    }
    default: {  // AllReduce and Reduce
      if (op.kind == Kind::Reduce && c->rank != op.root) break;  // only the root builds a result
      if (op.premul) {
        if (ncclResult_t r = premul_reduce(c, op.count, op.type(), &op.staging); r != ncclSuccess)
          return r;
        break;
      }
      if (!read_peer(c->rz, 0, op.count * es, &op.staging)) return ncclSystemError;
      for (int r = 1; r < c->nranks; ++r) {
        if (!read_peer(c->rz, r, op.count * es, &peer)) return ncclSystemError;
        if (!reduce(op.staging.data(), peer.data(), op.count, op.type(), op.rop()))
          return ncclInvalidArgument;
      }
      if (op.red == ncclAvg) average(op.staging.data(), op.count, op.type(), c->nranks);
      break;
    }
  }
  if (!op.staging.empty()) {
    DeviceGuard g(c->cuda_dev);
    if (cudaMemcpy(op.recv, op.staging.data(), op.staging.size(), cudaMemcpyHostToDevice) !=
        cudaSuccess)
      return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}

// Publishing "done" only after every rank has read is what makes it safe for
// the next collective to overwrite this rank's file.
void release(Op& op) {
  if (!is_collective(op.kind)) return;
  op.comm->rz->meta->done[op.comm->rank].store(op.seq, std::memory_order_release);
}

ncclResult_t settle(Op& op) {
  if (!is_collective(op.kind)) return ncclSuccess;
  Comm* c = op.comm;
  Meta* meta = c->rz->meta;
  return wait_for([&] {
        for (int r = 0; r < c->nranks; ++r)
          if (meta->done[r].load(std::memory_order_acquire) < op.seq) return false;
        return true;
      }, "the other ranks to finish this collective", c, moves_data(op.kind));
}

ncclResult_t execute(std::vector<Op>& ops) {
  ncclResult_t rc = ncclSuccess;
  auto keep = [&](ncclResult_t r) { if (rc == ncclSuccess) rc = r; };
  // Every collective publishes its data in its rank's one file on its communicator, so a
  // pass that deposited two collectives of the same communicator before collecting either
  // would have the second overwrite the first (two all-reduces in one group read each
  // other's data). Collectives of different communicators do not meet there, and must run
  // together: one thread drives every rank's communicator in a single group, and it
  // cannot wait for a rank whose operation it has not deposited yet. So the group runs
  // in rounds: the k-th collective of each communicator in round k, the point-to-point
  // operations (which have a file each) in round 0, every round in the four passes.
  // Collectives of a communicator are issued in the same order on every rank, so the
  // rounds line up.
  std::unordered_map<Comm*, size_t> seen;
  std::vector<size_t> round(ops.size(), 0);
  size_t rounds = 1;
  for (size_t i = 0; i < ops.size(); ++i) {
    if (!is_collective(ops[i].kind)) continue;
    round[i] = seen[ops[i].comm]++;
    rounds = std::max(rounds, round[i] + 1);
  }
  for (size_t r = 0; r < rounds && rc == ncclSuccess; ++r) {
    std::vector<Op*> in_round;
    for (size_t i = 0; i < ops.size(); ++i)
      if (round[i] == r) in_round.push_back(&ops[i]);
    for (Op* op : in_round) keep(deposit(*op));
    if (rc == ncclSuccess) for (Op* op : in_round) keep(collect(*op));
    for (Op* op : in_round) release(*op);
    if (rc == ncclSuccess) for (Op* op : in_round) keep(settle(*op));
  }
  return rc == kRevokedWait ? ncclSuccess : rc;
}

// Run a batch on non-blocking communicators off the calling thread. Batches on
// the same communicator still run in the order they were issued: each waits
// for the one before it on every communicator it touches.
void launch_async(std::vector<Op> ops) {
  std::vector<Comm*> comms;
  for (auto& op : ops)
    if (std::find(comms.begin(), comms.end(), op.comm) == comms.end()) comms.push_back(op.comm);
  auto done = std::make_shared<std::promise<void>>();
  std::shared_future<void> fut = done->get_future().share();
  std::vector<std::shared_future<void>> before;
  for (Comm* c : comms) {
    std::lock_guard<std::mutex> l(c->mu);
    if (c->tail.valid()) before.push_back(c->tail);
    c->tail = fut;
    c->pending.fetch_add(1, std::memory_order_acq_rel);
  }
  std::thread([ops = std::move(ops), comms, before, done]() mutable {
    for (auto& f : before) f.wait();
    const ncclResult_t rc = execute(ops);
    for (Comm* c : comms) {
      if (rc != ncclSuccess) c->state.store(rc, std::memory_order_release);
      c->pending.fetch_sub(1, std::memory_order_acq_rel);
    }
    done->set_value();
  }).detach();
}

// Runs the calling thread's recorded ops. `async` says whether they went to the
// background; the caller picks the return code, which differs by call.
ncclResult_t run_pending(bool* async) {
  std::vector<Op> ops;
  ops.swap(t_pending);
  *async = false;
  for (auto& op : ops) if (!op.comm->cfg.blocking) *async = true;
  if (ops.empty()) return ncclSuccess;
  if (*async) { launch_async(std::move(ops)); return ncclSuccess; }
  return execute(ops);
}

// Waits for every background batch on `c`.
void drain(Comm* c) {
  std::shared_future<void> t;
  {
    std::lock_guard<std::mutex> l(c->mu);
    t = c->tail;
  }
  if (t.valid()) t.wait();
}

ncclResult_t check_comm(Comm* c, bool poison = false) {
  if (!c) {
    std::fprintf(stderr, "[vgpu] nccl: comm argument is NULL\n");
    return ncclInvalidArgument;
  }
  return busy(c, poison) ? ncclInvalidArgument : ncclSuccess;
}

// A collective or point-to-point operation issued on a stream that is capturing is recorded as a host node of the
// graph, which does it when the graph runs (NCCL: a kernel node of the captured stream). Its place in the
// communicator's order of collectives is taken then (every rank's graphs launch in the same order, as the
// collectives of an eager program are issued), not at the capture; the operation's buffers are read and written
// then, and the stream is not waited for (it is the graph's own). The communicator and the buffers have to outlive
// the graph, as NCCL's do. Each rank's graph waits at its node for the other ranks' nodes, so the ranks' graphs
// have to be launched concurrently (a rank per thread or process, as a program that runs graphs on several GPUs
// does) -- a launch here runs the graph in the calling thread.
bool capturable(const Op& op) {
  const bool stream_op = op.kind != Kind::Split && op.kind != Kind::Shrink && op.kind != Kind::Suspend && op.kind != Kind::Resume;
  if (!stream_op || !op.stream) return false;
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  return cudaStreamIsCapturing(op.stream, &st) == cudaSuccess && st == cudaStreamCaptureStatusActive;
}

bool record_captured(const Op& op) {
  CUstream_st* stream = reinterpret_cast<CUstream_st*>(op.stream);
  auto shared = std::make_shared<Op>(op);
  return vgpu_record_host_op_if_capturing(stream, [shared] {
    Op o = *shared;
    Comm* c = o.comm;
    if (is_collective(o.kind)) o.seq = ++c->seq;
    std::vector<Op> ops;
    ops.push_back(std::move(o));
    t_replay = true;
    const ncclResult_t rc = execute(ops);
    t_replay = false;
    if (rc != ncclSuccess) c->state.store(rc, std::memory_order_release);
  });
}

// The captured operations of the open group. A group's operations run together on NCCL, so a send need not be
// issued before the receive that matches the one the peer issues; recorded one after another they have to be, so
// the sends go first.
thread_local std::vector<Op> t_captured;

ncclResult_t record_group_captured() {
  std::vector<Op> ops;
  ops.swap(t_captured);
  std::stable_partition(ops.begin(), ops.end(), [](const Op& o) { return o.kind == Kind::Send; });
  for (const Op& o : ops)
    if (!record_captured(o)) return ncclInvalidUsage;   // the capture ended meanwhile
  return ncclSuccess;
}

ncclResult_t enqueue(Op op) {
  Comm* c = op.comm;
  if (ncclResult_t r = check_comm(c, true); r != ncclSuccess) return fail_call(r);
  if (moves_data(op.kind)) {
    // card: a revoked communicator refuses new work with ncclInvalidUsage
    // ("communicator was revoked"). Work issued on a suspended one is undefined
    // there (an illegal memory access on the buffers it released); here it is
    // refused the same way instead of corrupting anything.
    if (c->revoked.load(std::memory_order_acquire)) {
      std::fprintf(stderr, "[vgpu] nccl: %s: communicator was revoked\n", kind_name(op.kind));
      return fail_call(ncclInvalidUsage);
    }
    if (c->suspended.load(std::memory_order_acquire)) {
      std::fprintf(stderr, "[vgpu] nccl: %s: communicator is suspended (call ncclCommResume first)\n",
                   kind_name(op.kind));
      return fail_call(ncclInvalidUsage);
    }
  }
  if (type_size(op.type()) == 0) {
    std::fprintf(stderr, "[vgpu] nccl: invalid type %d\n", op.dt);
    return fail_call(ncclInvalidArgument);
  }
  if ((op.kind == Kind::Broadcast || op.kind == Kind::Reduce || op.kind == Kind::Gather ||
       op.kind == Kind::Scatter) && (op.root < 0 || op.root >= c->nranks)) {
    std::fprintf(stderr, "[vgpu] nccl: root %d is not a rank of a %d-rank communicator\n", op.root,
                 c->nranks);
    return fail_call(ncclInvalidArgument);
  }
  if ((op.kind == Kind::Send || op.kind == Kind::Recv) && (op.peer < 0 || op.peer >= c->nranks)) {
    std::fprintf(stderr, "[vgpu] nccl: peer %d is not a rank of a %d-rank communicator\n", op.peer,
                 c->nranks);
    return fail_call(ncclInvalidArgument);
  }
  if (op.red < 0 || op.red >= ncclMaxRedOp) return fail_call(ncclInvalidArgument);
  if (op.red >= ncclNumOps) {
    // card: an operator from another communicator, or destroyed, is
    // ncclInvalidArgument, and so is one used with another datatype.
    std::lock_guard<std::mutex> l(c->mu);
    const long idx = redop_slot(c, op.rop());
    if (idx < 0) {
      std::fprintf(stderr, "[vgpu] nccl: reduction operation %d unknown to this communicator\n",
                   op.red);
      return fail_call(ncclInvalidArgument);
    }
    if (c->redops[idx].dt != op.type()) {
      std::fprintf(stderr, "[vgpu] nccl: data type supplied to user-created ncclRedOp_t does not "
                           "match type given to reduction operation\n");
      return fail_call(ncclInvalidArgument);
    }
    op.premul = true;
    op.scalar = c->redops[idx];
    op.red = ncclSum;
  }
  // On a capturing stream the operation is recorded into the graph and happens at each launch, among the same
  // launches of the other ranks' graphs (see capturable).
  if (capturable(op)) {
    if (t_group_depth > 0) {
      t_captured.push_back(std::move(op));
      return ncclSuccess;
    }
    return record_captured(op) ? ncclSuccess : fail_call(ncclInvalidUsage);
  }
  if (is_collective(op.kind)) op.seq = ++c->seq;  // p2p has its own handshake
  t_pending.push_back(std::move(op));
  if (t_group_depth > 0) return ncclSuccess;
  bool async = false;
  const ncclResult_t r = run_pending(&async);
  // card: a collective on a non-blocking communicator returns ncclInProgress,
  // even once it is connected.
  return async ? ncclInProgress : r;
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

/* ---- library-level ---- */

VGPU_EXPORT ncclResult_t ncclGetVersion(int* version) {
  if (!version) return ncclInvalidArgument;
  *version = NCCL_VERSION_CODE;
  return ncclSuccess;
}

VGPU_EXPORT const char* ncclGetErrorString(ncclResult_t r) {
  switch (r) {
    case ncclSuccess: return "no error";
    case ncclUnhandledCudaError: return "unhandled cuda error";
    case ncclSystemError: return "unhandled system error";
    case ncclInternalError: return "internal error";
    case ncclInvalidArgument: return "invalid argument";
    case ncclInvalidUsage: return "invalid usage";
    case ncclRemoteError: return "remote process exited or there was a network error";
    case ncclInProgress: return "NCCL operation in progress";
    case ncclTimeout: return "NCCL operation timed out";
    default: return "unknown result code";
  }
}
VGPU_EXPORT const char* ncclGetLastError(ncclComm_t) { return ""; }

VGPU_EXPORT ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  if (!out) return ncclInvalidArgument;
  static std::atomic<uint64_t> counter{0};
  std::memset(out, 0, sizeof(*out));
  const uint64_t a = (uint64_t)getpid() << 32 ^ counter.fetch_add(1);
  const uint64_t b = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count() ^
                     (uint64_t)std::chrono::system_clock::now().time_since_epoch().count();
  std::memcpy(out->internal, &a, 8);
  std::memcpy(out->internal + 8, &b, 8);
  std::memcpy(out->internal + 16, "VGPU-NCCL", 9);
  return ncclSuccess;
}

/* ---- communicators ---- */

namespace {

// Every init form lands here once it knows its rendezvous name. A non-blocking
// one returns ncclInProgress (card: so does NCCL's, for InitRankConfig and
// InitRankScalable alike) and completes when every rank has joined.
// `dflt` is what a NULL config means: ncclCommInitRank is always blocking, the
// config forms take NCCL_COMM_BLOCKING's default.
ncclResult_t init_rank(ncclComm_t* out, int nranks, const std::string& base, int rank,
                       const ncclConfig_t* config, const CommConfig* dflt = nullptr) {
  if (!out) return ncclInvalidArgument;
  *out = nullptr;
  CommConfig cfg;
  if (ncclResult_t r = parse_config(config, dflt, &cfg); r != ncclSuccess) return r;
  if (nranks < 1 || rank < 0 || rank >= nranks) {
    std::fprintf(stderr, "[vgpu] nccl: invalid rank requested : %d/%d\n", rank, nranks);
    return ncclInvalidArgument;
  }
  if (nranks > kMaxRanks) {
    std::fprintf(stderr, "[vgpu] nccl: %d ranks requested, this build supports %d\n", nranks,
                 kMaxRanks);
    return ncclInvalidArgument;
  }
  int dev = 0;
  cudaGetDevice(&dev);
  std::string err;
  Comm* c = make_comm(base, nranks, rank, dev, cfg, &err);
  if (!c) {
    std::fprintf(stderr, "[vgpu] nccl: %s\n", err.c_str());
    return ncclSystemError;
  }
  if (!quiet() && rank == 0)
    std::fprintf(stderr,
                 "[vgpu] nccl: %d ranks over %s (file-backed transport; see nvidia/docs/libraries.md)\n",
                 nranks, c->rz->base.c_str());
  *out = reinterpret_cast<ncclComm_t>(c);
  if (cfg.blocking) return ncclSuccess;
  c->init_pending.store(true, std::memory_order_release);
  refresh_init(c);
  return ncclInProgress;
}

}  // namespace

VGPU_EXPORT ncclResult_t ncclCommInitRankConfig(ncclComm_t* out, int nranks, ncclUniqueId id,
                                                int rank, ncclConfig_t* config) {
  return init_rank(out, nranks, rendezvous_dir() + "/" + hex16(id), rank, config);
}

VGPU_EXPORT ncclResult_t ncclCommInitRank(ncclComm_t* out, int nranks, ncclUniqueId id, int rank) {
  static const CommConfig kBlocking{1};
  return init_rank(out, nranks, rendezvous_dir() + "/" + hex16(id), rank, nullptr, &kBlocking);
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 23, 0)
// Several unique ids only spread NCCL's bootstrap over several roots; every rank
// passes the same ones, so they name one rendezvous together.
VGPU_EXPORT ncclResult_t ncclCommInitRankScalable(ncclComm_t* out, int nranks, int myrank, int nId,
                                                  ncclUniqueId* commIds, ncclConfig_t* config) {
  // card: nId outside 1..nranks, or no ids, crashes NCCL 2.29.7 after "improper
  // usage of ncclCommInitRank"; refusing them is the defined version of that.
  if (!out || !commIds || nId < 1 || nId > nranks) {
    std::fprintf(stderr, "[vgpu] nccl: improper usage of ncclCommInitRankScalable: nId = %d, "
                         "nranks=%d\n", nId, nranks);
    if (out) *out = nullptr;
    return ncclInvalidArgument;
  }
  std::string key = "scalable";
  for (int i = 0; i < nId; ++i) key += "|" + hex16(commIds[i]);
  return init_rank(out, nranks, derived_base(rendezvous_dir(), key), myrank, config);
}
#endif

VGPU_EXPORT ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  if (!comms || ndev < 1) return ncclInvalidArgument;
  ncclUniqueId id;
  ncclResult_t r = ncclGetUniqueId(&id);
  if (r != ncclSuccess) return r;
  int saved = 0;
  cudaGetDevice(&saved);
  for (int i = 0; i < ndev; ++i) {
    const int dev = devlist ? devlist[i] : i;
    if (cudaSetDevice(dev) != cudaSuccess) { cudaSetDevice(saved); return ncclUnhandledCudaError; }
    r = ncclCommInitRank(&comms[i], ndev, id, i);
    if (r != ncclSuccess) { cudaSetDevice(saved); return r; }
  }
  cudaSetDevice(saved);
  return ncclSuccess;
}

namespace {
void free_comm(Comm* c) {
  drain(c);
  detach(c->rz);
  delete c;
}
}  // namespace

// card: NULL is ncclSuccess for Destroy, Abort and Finalize alike.
VGPU_EXPORT ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  auto* c = reinterpret_cast<Comm*>(comm);
  if (!c) return ncclSuccess;
  if (busy(c)) return ncclInvalidArgument;
  free_comm(c);
  return ncclSuccess;
}

// Everything issued has already been flushed through the files by the time it
// returned (or, non-blocking, by the time its state went back to ncclSuccess),
// so there is nothing left to flush. card: a second finalize is
// ncclInvalidArgument; a non-blocking one returns ncclInProgress.
VGPU_EXPORT ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  auto* c = reinterpret_cast<Comm*>(comm);
  if (!c) return ncclSuccess;
  if (busy(c)) return ncclInvalidArgument;
  if (c->finalized) {
    std::fprintf(stderr, "[vgpu] nccl: communicator already finalized\n");
    return ncclInvalidArgument;
  }
  // card: Finalize on a revoked communicator is ncclInvalidArgument, as the header says.
  if (c->revoked.load(std::memory_order_acquire)) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommFinalize on a revoked communicator is invalid\n");
    return ncclInvalidArgument;
  }
  c->finalized = true;
  return c->cfg.blocking ? ncclSuccess : ncclInProgress;
}

// Abort stops whatever this communicator is waiting on, and tells the peers so
// that theirs give up with ncclRemoteError rather than running out the clock.
VGPU_EXPORT ncclResult_t ncclCommAbort(ncclComm_t comm) {
  auto* c = reinterpret_cast<Comm*>(comm);
  if (!c) return ncclSuccess;
  c->aborted.store(true, std::memory_order_release);
  if (c->rz) c->rz->meta->aborted.store(1, std::memory_order_release);
  free_comm(c);
  return ncclSuccess;
}

VGPU_EXPORT ncclResult_t ncclCommCount(const ncclComm_t comm, int* n) {
  auto* c = reinterpret_cast<Comm*>(comm);
  if (!n) return ncclInvalidArgument;
  if (ncclResult_t r = check_comm(c); r != ncclSuccess) return r;
  *n = c->nranks;
  return ncclSuccess;
}
VGPU_EXPORT ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* r) {
  auto* c = reinterpret_cast<Comm*>(comm);
  if (!r) return ncclInvalidArgument;
  if (ncclResult_t e = check_comm(c); e != ncclSuccess) return e;
  *r = c->rank;
  return ncclSuccess;
}
VGPU_EXPORT ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* d) {
  auto* c = reinterpret_cast<Comm*>(comm);
  if (!d) return ncclInvalidArgument;
  if (ncclResult_t r = check_comm(c); r != ncclSuccess) return r;
  *d = c->cuda_dev;
  return ncclSuccess;
}

// An error the communicator is holding comes first (card: one left by a call
// made too early stays, even after the init finishes); then any work still
// running; then success.
VGPU_EXPORT ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* e) {
  auto* c = reinterpret_cast<Comm*>(comm);
  if (!c || !e) return ncclInvalidArgument;
  refresh_init(c);
  const int s = c->state.load(std::memory_order_acquire);
  if (s != ncclSuccess) *e = static_cast<ncclResult_t>(s);
  else if (c->pending.load(std::memory_order_acquire) > 0 ||
           c->init_pending.load(std::memory_order_acquire))
    *e = ncclInProgress;
  else
    *e = ncclSuccess;
  return ncclSuccess;
}


/* ---- revoke, suspend, resume, memory statistics ---- */

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
// card (NCCL 2.31.2, two ranks): Revoke is local and idle-instant -- one rank
// can revoke alone, and the state reads ncclSuccess straight away -- and it
// unblocks a stream stuck in a collective its peer never joined. A revoked
// communicator refuses collectives and p2p with ncclInvalidUsage, still splits,
// shrinks, suspends, resumes and destroys, refuses ncclCommFinalize
// (ncclInvalidArgument), and cannot be revoked twice (ncclInvalidArgument). A
// non-zero flag is ncclInvalidArgument. NULL is ncclSuccess, as for Destroy.
// A non-blocking communicator answers ncclInProgress, then settles.
VGPU_EXPORT ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags) {
  auto* c = reinterpret_cast<Comm*>(comm);
  if (!c) return ncclSuccess;
  if (revokeFlags != NCCL_REVOKE_DEFAULT) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommRevoke: revokeFlags %d must be NCCL_REVOKE_DEFAULT (0)\n",
                 revokeFlags);
    return ncclInvalidArgument;
  }
  if (c->revoked.exchange(true, std::memory_order_acq_rel)) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommRevoke: communicator already revoked\n");
    return ncclInvalidArgument;
  }
  if (c->cfg.blocking) return ncclSuccess;
  // Quiescent once the batches that were running have unwound. This takes its
  // place at the end of the communicator's queue, so a destroy waits for it.
  auto done = std::make_shared<std::promise<void>>();
  std::shared_future<void> before;
  {
    std::lock_guard<std::mutex> l(c->mu);
    before = c->tail;
    c->tail = done->get_future().share();
    c->pending.fetch_add(1, std::memory_order_acq_rel);
  }
  std::thread([c, before, done] {
    if (before.valid()) before.wait();
    c->pending.fetch_sub(1, std::memory_order_acq_rel);
    done->set_value();
  }).detach();
  return ncclInProgress;
}

// card: Suspend and Resume are collective, one barrier each. Only bit 0
// (NCCL_SUSPEND_MEM) does anything -- any value with it set suspends, -1
// included -- and any other value is a successful no-op. Suspending twice, or
// resuming what is not suspended, is ncclInvalidUsage. A NULL communicator is
// ncclInvalidArgument. Both work inside a group and on a non-blocking
// communicator (ncclInProgress).
VGPU_EXPORT ncclResult_t ncclCommSuspend(ncclComm_t comm, int flags) {
  auto* c = reinterpret_cast<Comm*>(comm);
  if (!c) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommSuspend : comm argument is NULL\n");
    return fail_call(ncclInvalidArgument);
  }
  if (ncclResult_t r = check_comm(c); r != ncclSuccess) return fail_call(r);
  if (!(flags & NCCL_SUSPEND_MEM)) return ncclSuccess;
  if (c->suspended.load(std::memory_order_acquire)) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommSuspend: already suspended\n");
    return fail_call(ncclInvalidUsage);
  }
  Op o{Kind::Suspend, c, nullptr, nullptr, 0, ncclInt8, ncclSum};
  return enqueue(std::move(o));
}

VGPU_EXPORT ncclResult_t ncclCommResume(ncclComm_t comm) {
  auto* c = reinterpret_cast<Comm*>(comm);
  if (!c) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommResume : comm argument is NULL\n");
    return fail_call(ncclInvalidArgument);
  }
  if (ncclResult_t r = check_comm(c); r != ncclSuccess) return fail_call(r);
  if (!c->suspended.load(std::memory_order_acquire)) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommResume: not in suspended state\n");
    return fail_call(ncclInvalidUsage);
  }
  Op o{Kind::Resume, c, nullptr, nullptr, 0, ncclInt8, ncclSum};
  return enqueue(std::move(o));
}

// What NCCL tracks is GPU memory it allocated for the communicator: on the card
// a two-rank communicator held 12 MiB it can release (suspending frees exactly
// that: cudaMemGetInfo's free memory rose by 12582912), 4 MiB it cannot, and
// 16 MiB in all. This transport allocates no device memory -- the bytes live in
// files -- so there is nothing to release and the sizes are zero; the
// "suspended" statistic is live. Any other statistic is ncclInvalidArgument, and
// so are a NULL value or communicator.
VGPU_EXPORT ncclResult_t ncclCommMemStats(ncclComm_t comm, ncclCommMemStat_t stat, uint64_t* value) {
  auto* c = reinterpret_cast<Comm*>(comm);
  if (!c) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommMemStats : comm argument is NULL\n");
    return ncclInvalidArgument;
  }
  if (!value) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommMemStats : value argument is NULL\n");
    return ncclInvalidArgument;
  }
  const int which = static_cast<int>(stat);
  switch (which) {
    case ncclStatGpuMemSuspend: case ncclStatGpuMemPersist: case ncclStatGpuMemTotal:
      *value = 0;
      return ncclSuccess;
    case ncclStatGpuMemSuspended:
      *value = c->suspended.load(std::memory_order_acquire) ? 1 : 0;
      return ncclSuccess;
    default:
      std::fprintf(stderr, "[vgpu] nccl: ncclCommMemStats: unknown statistic %d\n", which);
      return ncclInvalidArgument;
  }
}

/* ---- growing a communicator ----
   A grow's unique id names the rendezvous the old and new ranks meet at. Its
   first 16 bytes are derived from the parent's rendezvous and how many grows
   it has seen, so every existing rank works out the same name without being
   handed the id (the non-root form: uniqueId NULL); the rest is a nonce, which
   is why every ncclCommGetUniqueId call returns a different id (card: it does)
   though only one grow can use them. */
namespace {
constexpr char kGrowTag[] = "VGPU-GROW";

void grow_id(const Comm* c, uint64_t generation, ncclUniqueId* out) {
  static std::atomic<uint64_t> nonce{0};
  std::memset(out, 0, sizeof(*out));
  uint64_t h[2];
  hash128("grow|" + c->rz->base + "|" + std::to_string(generation), h);
  std::memcpy(out->internal, h, 16);
  std::memcpy(out->internal + 16, kGrowTag, sizeof kGrowTag - 1);
  const uint64_t n = ((uint64_t)getpid() << 32) ^ nonce.fetch_add(1) ^
                     (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
  std::memcpy(out->internal + 16 + sizeof kGrowTag - 1, &n, 8);
}

bool is_grow_id(const ncclUniqueId& id) {
  return std::memcmp(id.internal + 16, kGrowTag, sizeof kGrowTag - 1) == 0;
}
}  // namespace

// card: any rank may ask, each call returns a fresh id, and the id stays good
// across the other management calls (revoked and suspended communicators
// included). A NULL communicator or id is ncclInvalidArgument.
VGPU_EXPORT ncclResult_t ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId* uniqueId) {
  auto* c = reinterpret_cast<Comm*>(comm);
  if (!c) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommGetUniqueId : comm argument is NULL\n");
    return ncclInvalidArgument;
  }
  if (!uniqueId) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommGetUniqueId : uniqueId argument is NULL\n");
    return ncclInvalidArgument;
  }
  if (ncclResult_t r = check_comm(c); r != ncclSuccess) return r;
  grow_id(c, c->grows + 1, uniqueId);
  return ncclSuccess;
}

// Existing ranks pass their communicator and rank = -1 (the root also passes
// the id, the others NULL); a new rank passes comm = NULL, the id and its own
// rank, which has to be a free one. Existing ranks keep their numbers.
// card: nRanks that is not larger than the communicator, an existing rank that
// names a rank, a NULL newcomm, a new rank without an id or with a rank outside
// 0..nRanks-1 are all ncclInvalidArgument; the non-root form on a one-rank
// communicator, which has no root to wait for, is ncclInternalError. A grow
// returns the way a split does: ncclSuccess, with ncclInProgress showing in the
// new communicator's state while it is non-blocking. NCCL needs the existing
// ranks to start before a new rank joins (a new rank that arrives first fails
// with ncclInternalError after a fraction of a second); the rendezvous here
// has no such race and the new rank simply waits, which is the lenient side of
// the same contract. A used id is refused rather than hanging, as the card
// does.
VGPU_EXPORT ncclResult_t ncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId* uniqueId,
                                      int rank, ncclComm_t* newcomm, ncclConfig_t* config) {
  auto* c = reinterpret_cast<Comm*>(comm);
  if (!newcomm) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommGrow : newcomm argument is NULL\n");
    return ncclInvalidArgument;
  }
  *newcomm = nullptr;
  CommConfig cfg;
  if (ncclResult_t r = parse_config(config, c ? &c->cfg : nullptr, &cfg); r != ncclSuccess) return r;
  if (nRanks < 2 || nRanks > kMaxRanks) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommGrow: invalid size %d (this build supports 2..%d)\n",
                 nRanks, kMaxRanks);
    return ncclInvalidArgument;
  }
  ncclUniqueId head;
  int dev = 0, my_rank = rank;
  if (c) {
    if (ncclResult_t r = check_comm(c); r != ncclSuccess) return r;
    if (nRanks <= c->nranks) {
      std::fprintf(stderr, "[vgpu] nccl: ncclCommGrow: nRanks %d must exceed the communicator's %d\n",
                   nRanks, c->nranks);
      return ncclInvalidArgument;
    }
    if (rank != -1) {
      std::fprintf(stderr, "[vgpu] nccl: ncclCommGrow: an existing rank passes rank = -1, not %d\n", rank);
      return ncclInvalidArgument;
    }
    if (!uniqueId && c->nranks == 1) {
      std::fprintf(stderr, "[vgpu] nccl: ncclCommGrow: no unique id: the only rank of a one-rank "
                           "communicator is the root\n");
      return ncclInternalError;
    }
    grow_id(c, c->grows + 1, &head);
    if (uniqueId && std::memcmp(uniqueId->internal, head.internal, 16) != 0) {
      std::fprintf(stderr, "[vgpu] nccl: ncclCommGrow: the unique id is not this communicator's "
                           "current one (already used, or from another communicator)\n");
      return ncclInvalidArgument;
    }
    ++c->grows;
    my_rank = c->rank;
    dev = c->cuda_dev;
  } else {
    if (!uniqueId) {
      std::fprintf(stderr, "[vgpu] nccl: ncclCommGrow: a new rank needs the unique id\n");
      return ncclInvalidArgument;
    }
    if (rank < 0 || rank >= nRanks) {
      std::fprintf(stderr, "[vgpu] nccl: ncclCommGrow: invalid rank requested : %d/%d\n", rank, nRanks);
      return ncclInvalidArgument;
    }
    if (!is_grow_id(*uniqueId)) {
      std::fprintf(stderr, "[vgpu] nccl: ncclCommGrow: not an id from ncclCommGetUniqueId\n");
      return ncclInvalidArgument;
    }
    head = *uniqueId;
    cudaGetDevice(&dev);
  }
  std::string dir = c ? dir_of(c->rz->base) : rendezvous_dir();
  std::string err;
  Comm* child = make_comm(derived_base(dir, "grow|" + hex16(head)), nRanks, my_rank, dev, cfg, &err, true);
  if (!child) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommGrow: %s\n", err.c_str());
    return err.find("already joined") != std::string::npos ? ncclInvalidArgument : ncclSystemError;
  }
  *newcomm = reinterpret_cast<ncclComm_t>(child);
  if (cfg.blocking) return ncclSuccess;
  child->init_pending.store(true, std::memory_order_release);
  refresh_init(child);
  return ncclSuccess;
}
#endif

VGPU_EXPORT ncclResult_t ncclCommRegister(const ncclComm_t, void*, size_t, void** handle) {
  if (handle) *handle = nullptr;  // no NIC to pin against
  return ncclSuccess;
}
VGPU_EXPORT ncclResult_t ncclCommDeregister(const ncclComm_t, void*) { return ncclSuccess; }

VGPU_EXPORT ncclResult_t ncclMemAlloc(void** ptr, size_t size) {
  if (!ptr) return ncclInvalidArgument;
  return cudaMalloc(ptr, size) == cudaSuccess ? ncclSuccess : ncclSystemError;
}
VGPU_EXPORT ncclResult_t ncclMemFree(void* ptr) {
  return cudaFree(ptr) == cudaSuccess ? ncclSuccess : ncclSystemError;
}

/* ---- new communicators from old ---- */

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 18, 0)
// A collective on the parent: every rank, NCCL_SPLIT_NOCOLOR ones included,
// publishes its (color, key), and each builds its own view of the result.
// card: on a non-blocking parent the call returns ncclSuccess outside a group
// (ncclInProgress from the ncclGroupEnd inside one), and *newcomm stays NULL
// until the parent's state is ncclSuccess again. A NULL config inherits the
// parent's, so the child of a non-blocking parent is non-blocking too.
VGPU_EXPORT ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* newcomm,
                                       ncclConfig_t* config) {
  auto* c = reinterpret_cast<Comm*>(comm);
  if (!newcomm) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommSplit : newcomm argument is NULL\n");
    return fail_call(ncclInvalidArgument);
  }
  *newcomm = nullptr;
  if (ncclResult_t r = check_comm(c); r != ncclSuccess) return fail_call(r);
  Op o{Kind::Split, c, nullptr, nullptr, 2, ncclInt32, ncclSum};
  if (ncclResult_t r = parse_config(config, &c->cfg, &o.child); r != ncclSuccess)
    return fail_call(r);
  o.color_key[0] = color;
  o.color_key[1] = key;
  o.out = newcomm;
  const ncclResult_t r = enqueue(std::move(o));
  return r == ncclInProgress ? ncclSuccess : r;
}
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
// Only the surviving ranks call this, and they all hold the same exclusion
// list, so nothing needs exchanging: each works out its new rank and joins.
VGPU_EXPORT ncclResult_t ncclCommShrink(ncclComm_t comm, int* excludeRanksList,
                                        int excludeRanksCount, ncclComm_t* newcomm,
                                        ncclConfig_t* config, int shrinkFlags) {
  (void)shrinkFlags;  // card: any value is accepted; ABORT has nothing in flight to stop here
  auto* c = reinterpret_cast<Comm*>(comm);
  if (!newcomm) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommShrink : newcomm argument is NULL\n");
    return fail_call(ncclInvalidArgument);
  }
  if (ncclResult_t r = check_comm(c); r != ncclSuccess) return fail_call(r);
  // card: a NULL list is refused, and so is a count of zero or less.
  if (!excludeRanksList || excludeRanksCount <= 0) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommShrink : excludeRanksList argument is NULL or empty\n");
    return fail_call(ncclInvalidArgument);
  }
  std::vector<int> ex(excludeRanksList, excludeRanksList + excludeRanksCount);
  std::sort(ex.begin(), ex.end());
  // card: excluding yourself is refused; a duplicate leaves an empty
  // communicator and an error. An out-of-range rank is accepted by NCCL 2.29.7
  // and counted as a removed rank, which leaves a communicator whose size
  // disagrees with its members; that is refused here.
  for (size_t i = 0; i < ex.size(); ++i) {
    if (ex[i] < 0 || ex[i] >= c->nranks || ex[i] == c->rank || (i && ex[i] == ex[i - 1])) {
      std::fprintf(stderr, "[vgpu] nccl: ncclCommShrink : cannot exclude rank %d\n", ex[i]);
      return fail_call(ncclInvalidArgument);
    }
  }
  Op o{Kind::Shrink, c, nullptr, nullptr, 0, ncclInt8, ncclSum};
  if (ncclResult_t r = parse_config(config, &c->cfg, &o.child); r != ncclSuccess)
    return fail_call(r);
  *newcomm = nullptr;
  o.out = newcomm;
  o.exclude = std::move(ex);
  o.shrink_epoch = ++c->shrinks;
  const ncclResult_t r = enqueue(std::move(o));
  return r == ncclInProgress ? ncclSuccess : r;
}

/* ---- symmetric memory windows ----
   card: on the RTX 3060 pair (no NVLink, no peer mappings) ncclCommWindowRegister
   returns ncclSuccess and a NULL window, for ncclMemAlloc and cudaMalloc buffers
   alike; a collective on the buffer then runs as it always does. That is what
   happens here, for the reason in the header comment. Measured argument errors:
   a NULL comm or win, and a NULL buffer or zero size, are ncclInvalidArgument;
   deregistering a NULL window is ncclSuccess. */
VGPU_EXPORT ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size,
                                                ncclWindow_t* win, int winFlags) {
  (void)winFlags;
  auto* c = reinterpret_cast<Comm*>(comm);
  if (!win) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommWindowRegister : win argument is NULL\n");
    return ncclInvalidArgument;
  }
  if (ncclResult_t r = check_comm(c); r != ncclSuccess) return r;
  if (!buff || size == 0) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommWindowRegister: invalid pointer %p / size %zu\n",
                 buff, size);
    return ncclInvalidArgument;
  }
  *win = nullptr;
  return ncclSuccess;
}

VGPU_EXPORT ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win) {
  auto* c = reinterpret_cast<Comm*>(comm);
  if (ncclResult_t r = check_comm(c); r != ncclSuccess) return r;
  return win ? ncclInvalidArgument : ncclSuccess;   // no window was ever handed out
}
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
// card: NCCL 2.29.7 refuses a NULL window (ncclInvalidArgument) and answers
// ncclSuccess with a NULL pointer for any other value; 2.31.2 and 2.32.3 answer
// ncclSuccess with NULL for every window, NULL included. No window here has a
// user pointer, and this follows the newer library, which is the version the
// header declares. A NULL communicator is ncclInvalidArgument in all of them.
VGPU_EXPORT ncclResult_t ncclWinGetUserPtr(ncclComm_t comm, ncclWindow_t win, void** outUserPtr) {
  (void)win;
  if (!comm) {
    std::fprintf(stderr, "[vgpu] nccl: ncclWinGetUserPtr : comm argument is NULL\n");
    return ncclInvalidArgument;
  }
  if (!outUserPtr) {
    std::fprintf(stderr, "[vgpu] nccl: ncclWinGetUserPtr : outUserPtr argument is NULL\n");
    return ncclInvalidArgument;
  }
  *outUserPtr = nullptr;
  return ncclSuccess;
}
#endif

/* ---- groups ---- */

VGPU_EXPORT ncclResult_t ncclGroupStart(void) { ++t_group_depth; return ncclSuccess; }

VGPU_EXPORT ncclResult_t ncclGroupEnd(void) {
  if (t_group_depth == 0) return ncclInvalidUsage;
  if (--t_group_depth > 0) return ncclSuccess;
  if (t_group_error != ncclSuccess) {
    const ncclResult_t r = t_group_error;
    t_group_error = ncclSuccess;
    t_pending.clear();
    t_captured.clear();
    return r;
  }
  if (!t_captured.empty())
    if (const ncclResult_t rc = record_group_captured(); rc != ncclSuccess) {
      t_pending.clear();
      return rc;
    }
  bool async = false;
  const ncclResult_t r = run_pending(&async);
  return async ? ncclInProgress : r;   // card: a non-blocking group returns ncclInProgress
}

VGPU_EXPORT ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t* info) {
  // Nothing here models time, so report the group as free and drop it.
  if (info) info->estimatedTime = 0.0f;
  if (t_group_depth > 0) --t_group_depth;
  t_pending.clear();
  t_captured.clear();
  t_group_error = ncclSuccess;
  return ncclSuccess;
}

/* ---- collectives ---- */

VGPU_EXPORT ncclResult_t ncclAllReduce(const void* send, void* recv, size_t count,
                                       ncclDataType_t dt, ncclRedOp_t op, ncclComm_t comm,
                                       cudaStream_t stream) {
  Op o{Kind::AllReduce, reinterpret_cast<Comm*>(comm), send, recv, count, dt, op, 0, 0, stream};
  return enqueue(std::move(o));
}

VGPU_EXPORT ncclResult_t ncclReduce(const void* send, void* recv, size_t count, ncclDataType_t dt,
                                    ncclRedOp_t op, int root, ncclComm_t comm,
                                    cudaStream_t stream) {
  Op o{Kind::Reduce, reinterpret_cast<Comm*>(comm), send, recv, count, dt, op, root, 0, stream};
  return enqueue(std::move(o));
}

VGPU_EXPORT ncclResult_t ncclBroadcast(const void* send, void* recv, size_t count,
                                       ncclDataType_t dt, int root, ncclComm_t comm,
                                       cudaStream_t stream) {
  Op o{Kind::Broadcast, reinterpret_cast<Comm*>(comm), send, recv, count, dt, ncclSum, root, 0,
       stream};
  return enqueue(std::move(o));
}

VGPU_EXPORT ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t dt, int root,
                                   ncclComm_t comm, cudaStream_t stream) {
  return ncclBroadcast(buff, buff, count, dt, root, comm, stream);
}

VGPU_EXPORT ncclResult_t ncclAllGather(const void* send, void* recv, size_t sendcount,
                                       ncclDataType_t dt, ncclComm_t comm, cudaStream_t stream) {
  Op o{Kind::AllGather, reinterpret_cast<Comm*>(comm), send, recv, sendcount, dt, ncclSum, 0, 0,
       stream};
  return enqueue(std::move(o));
}

VGPU_EXPORT ncclResult_t ncclReduceScatter(const void* send, void* recv, size_t recvcount,
                                           ncclDataType_t dt, ncclRedOp_t op, ncclComm_t comm,
                                           cudaStream_t stream) {
  Op o{Kind::ReduceScatter, reinterpret_cast<Comm*>(comm), send, recv, recvcount, dt, op, 0, 0,
       stream};
  return enqueue(std::move(o));
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
// NCCL 2.28 added these three; nccl.h declares them from that version on.
// card: rank j's output block i is rank i's input block j; count 0 is a no-op
// that succeeds. nccl.h promises no in-place all-to-all and on the card one
// races; here the input is copied out before any output is written, so it works.
VGPU_EXPORT ncclResult_t ncclAlltoAll(const void* send, void* recv, size_t count,
                                      ncclDataType_t dt, ncclComm_t comm, cudaStream_t stream) {
  Op o{Kind::AlltoAll, reinterpret_cast<Comm*>(comm), send, recv, count, dt, ncclSum, 0, 0, stream};
  return enqueue(std::move(o));
}

// card: only the root's buffer is written; the others' are left alone.
VGPU_EXPORT ncclResult_t ncclGather(const void* send, void* recv, size_t count, ncclDataType_t dt,
                                    int root, ncclComm_t comm, cudaStream_t stream) {
  Op o{Kind::Gather, reinterpret_cast<Comm*>(comm), send, recv, count, dt, ncclSum, root, 0, stream};
  return enqueue(std::move(o));
}

VGPU_EXPORT ncclResult_t ncclScatter(const void* send, void* recv, size_t count, ncclDataType_t dt,
                                     int root, ncclComm_t comm, cudaStream_t stream) {
  Op o{Kind::Scatter, reinterpret_cast<Comm*>(comm), send, recv, count, dt, ncclSum, root, 0,
       stream};
  return enqueue(std::move(o));
}
#endif


#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
/* ---- per-collective configuration: nccl*Config ----
   Every form checks its ncclCollConfig_t and then does what the plain
   collective does: the fields tune CTAs, channels and algorithm choice, none of
   which exist here, so a valid config changes nothing. What the card refuses
   is refused the same way, before the communicator, type and root are looked
   at (card: a bad config with a NULL communicator reports the config), and a
   refusal does not spoil the group it was issued in (card: the rest of the
   group ran and ncclGroupEnd returned ncclSuccess).
   card (NCCL 2.31.2): the size must be at least 64 (the struct as of 2.31) and
   the magic NCCL_API_MAGIC, in that order; the version is not looked at;
   forceAlgSelection must be 0 or 1; CTAPolicy is unset or 0..3; minCTAs,
   maxCTAs, nvlsCTAs, cgaClusterSize, userProfilerTag and ext are accepted
   whatever they hold. algSelection is a comma-separated list of algorithm
   names (ring, tree, collnetdirect, collnetchain, nvls, nvlstree, pat, in any
   case; blanks and empty items are ignored), each optionally prefixed with ^,
   which means every algorithm but that one; the selection is the union. With
   forceAlgSelection = 1 an unknown name, or a selection that leaves no
   algorithm available for the collective, is ncclInvalidArgument; with 0 it
   falls back to automatic. On the card's PCIe pair AllReduce has ring and tree;
   Broadcast, Reduce, AllGather and ReduceScatter ring only; AlltoAll, Gather
   and Scatter none, so any selection of theirs fails when forced. The same
   holds here: the one transport computes every collective the same way, so a
   selection that names ring or tree is honoured trivially, and NVLS, CollNet
   and PAT -- which need hardware or topology this does not have -- are not
   available, as on the card. */
namespace {
constexpr size_t kCollConfigMinSize = offsetof(ncclCollConfig_t, userProfilerTag) + sizeof(uint64_t);
enum AlgBit : uint32_t { kAlgRing = 1, kAlgTree = 2, kAlgCollnetDirect = 4, kAlgCollnetChain = 8,
                         kAlgNvls = 16, kAlgNvlsTree = 32, kAlgPat = 64, kAlgAll = 127 };

uint32_t algs_available(Kind k) {
  switch (k) {
    case Kind::AllReduce: return kAlgRing | kAlgTree;
    case Kind::Broadcast: case Kind::Reduce: case Kind::AllGather: case Kind::ReduceScatter:
      return kAlgRing;
    default: return 0;
  }
}

// false if a name is not one NCCL knows.
bool parse_alg_selection(const char* sel, uint32_t* mask) {
  static const struct { const char* name; uint32_t bit; } kNames[] = {
      {"ring", kAlgRing}, {"tree", kAlgTree}, {"collnetdirect", kAlgCollnetDirect},
      {"collnetchain", kAlgCollnetChain}, {"nvls", kAlgNvls}, {"nvlstree", kAlgNvlsTree},
      {"pat", kAlgPat}};
  *mask = 0;
  const std::string all(sel);
  size_t pos = 0;
  while (pos <= all.size()) {
    size_t end = all.find(',', pos);
    if (end == std::string::npos) end = all.size();
    std::string tok = all.substr(pos, end - pos);
    pos = end + 1;
    while (!tok.empty() && std::isspace((unsigned char)tok.front())) tok.erase(tok.begin());
    while (!tok.empty() && std::isspace((unsigned char)tok.back())) tok.pop_back();
    if (tok.empty()) continue;
    const bool neg = tok[0] == '^';
    if (neg) tok.erase(tok.begin());
    for (char& ch : tok) ch = (char)std::tolower((unsigned char)ch);
    uint32_t bit = 0;
    for (const auto& n : kNames) if (tok == n.name) bit = n.bit;
    if (!bit) return false;
    *mask |= neg ? (kAlgAll & ~bit) : bit;
  }
  return true;
}

ncclResult_t check_coll_config(const ncclCollConfig_t* cfg, Kind kind) {
  if (!cfg) return ncclSuccess;
  if (cfg->size < kCollConfigMinSize) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCollConfig_t size %zu too small (expected >= %zu)\n",
                 cfg->size, kCollConfigMinSize);
    return ncclInvalidArgument;
  }
  if (cfg->magic != NCCL_API_MAGIC) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCollConfig_t magic mismatch (got 0x%x, expected 0x%x)\n",
                 cfg->magic, (unsigned)NCCL_API_MAGIC);
    return ncclInvalidArgument;
  }
  if (cfg->forceAlgSelection != 0 && cfg->forceAlgSelection != 1) {
    std::fprintf(stderr, "[vgpu] nccl: config->forceAlgSelection=%d is invalid, accepted values are 0 or 1\n",
                 cfg->forceAlgSelection);
    return ncclInvalidArgument;
  }
  if (cfg->CTAPolicy != NCCL_CONFIG_UNDEF_INT && (cfg->CTAPolicy < 0 || cfg->CTAPolicy > 3)) {
    std::fprintf(stderr, "[vgpu] nccl: config->CTAPolicy=%d is invalid, accepted values are [0, 3]\n",
                 cfg->CTAPolicy);
    return ncclInvalidArgument;
  }
  if (cfg->algSelection && cfg->algSelection[0] && cfg->forceAlgSelection) {
    uint32_t mask = 0;
    if (!parse_alg_selection(cfg->algSelection, &mask)) {
      std::fprintf(stderr, "[vgpu] nccl: config->algSelection \"%s\" cannot be honored for %s (set "
                           "forceAlgSelection=0 to allow a fall back to automatic selection)\n",
                   cfg->algSelection, kind_name(kind));
      return ncclInvalidArgument;
    }
    if (!(mask & algs_available(kind))) {
      std::fprintf(stderr, "[vgpu] nccl: algSelection \"%s\": no algorithm in the selected set is "
                           "available for %s\n", cfg->algSelection, kind_name(kind));
      return ncclInvalidArgument;
    }
  }
  return ncclSuccess;
}
}  // namespace

VGPU_EXPORT ncclResult_t ncclAllReduceConfig(const void* send, void* recv, size_t count,
                                             ncclDataType_t dt, ncclRedOp_t op, ncclComm_t comm,
                                             cudaStream_t stream, const ncclCollConfig_t* config) {
  if (ncclResult_t r = check_coll_config(config, Kind::AllReduce); r != ncclSuccess) return r;
  return ncclAllReduce(send, recv, count, dt, op, comm, stream);
}
VGPU_EXPORT ncclResult_t ncclBroadcastConfig(const void* send, void* recv, size_t count,
                                             ncclDataType_t dt, int root, ncclComm_t comm,
                                             cudaStream_t stream, const ncclCollConfig_t* config) {
  if (ncclResult_t r = check_coll_config(config, Kind::Broadcast); r != ncclSuccess) return r;
  return ncclBroadcast(send, recv, count, dt, root, comm, stream);
}
VGPU_EXPORT ncclResult_t ncclReduceConfig(const void* send, void* recv, size_t count,
                                          ncclDataType_t dt, ncclRedOp_t op, int root,
                                          ncclComm_t comm, cudaStream_t stream,
                                          const ncclCollConfig_t* config) {
  if (ncclResult_t r = check_coll_config(config, Kind::Reduce); r != ncclSuccess) return r;
  return ncclReduce(send, recv, count, dt, op, root, comm, stream);
}
VGPU_EXPORT ncclResult_t ncclAllGatherConfig(const void* send, void* recv, size_t sendcount,
                                             ncclDataType_t dt, ncclComm_t comm,
                                             cudaStream_t stream, const ncclCollConfig_t* config) {
  if (ncclResult_t r = check_coll_config(config, Kind::AllGather); r != ncclSuccess) return r;
  return ncclAllGather(send, recv, sendcount, dt, comm, stream);
}
VGPU_EXPORT ncclResult_t ncclReduceScatterConfig(const void* send, void* recv, size_t recvcount,
                                                 ncclDataType_t dt, ncclRedOp_t op,
                                                 ncclComm_t comm, cudaStream_t stream,
                                                 const ncclCollConfig_t* config) {
  if (ncclResult_t r = check_coll_config(config, Kind::ReduceScatter); r != ncclSuccess) return r;
  return ncclReduceScatter(send, recv, recvcount, dt, op, comm, stream);
}
VGPU_EXPORT ncclResult_t ncclAlltoAllConfig(const void* send, void* recv, size_t count,
                                            ncclDataType_t dt, ncclComm_t comm, cudaStream_t stream,
                                            const ncclCollConfig_t* config) {
  if (ncclResult_t r = check_coll_config(config, Kind::AlltoAll); r != ncclSuccess) return r;
  return ncclAlltoAll(send, recv, count, dt, comm, stream);
}
VGPU_EXPORT ncclResult_t ncclGatherConfig(const void* send, void* recv, size_t count,
                                          ncclDataType_t dt, int root, ncclComm_t comm,
                                          cudaStream_t stream, const ncclCollConfig_t* config) {
  if (ncclResult_t r = check_coll_config(config, Kind::Gather); r != ncclSuccess) return r;
  return ncclGather(send, recv, count, dt, root, comm, stream);
}
VGPU_EXPORT ncclResult_t ncclScatterConfig(const void* send, void* recv, size_t count,
                                           ncclDataType_t dt, int root, ncclComm_t comm,
                                           cudaStream_t stream, const ncclCollConfig_t* config) {
  if (ncclResult_t r = check_coll_config(config, Kind::Scatter); r != ncclSuccess) return r;
  return ncclScatter(send, recv, count, dt, root, comm, stream);
}
#endif

VGPU_EXPORT ncclResult_t ncclSend(const void* send, size_t count, ncclDataType_t dt, int peer,
                                  ncclComm_t comm, cudaStream_t stream) {
  Op o{Kind::Send, reinterpret_cast<Comm*>(comm), send, nullptr, count, dt, ncclSum, 0, peer,
       stream};
  return enqueue(std::move(o));
}

VGPU_EXPORT ncclResult_t ncclRecv(void* recv, size_t count, ncclDataType_t dt, int peer,
                                  ncclComm_t comm, cudaStream_t stream) {
  Op o{Kind::Recv, reinterpret_cast<Comm*>(comm), nullptr, recv, count, dt, ncclSum, 0, peer,
       stream};
  return enqueue(std::move(o));
}


#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
/* ---- one-sided operations: ncclPutSignal, ncclSignal, ncclWaitSignal ----
   NCCL's host RMA moves data into a peer's registered window without the peer
   calling anything, through a GIN transport (InfiniBand or EFA) or, from 2.32,
   a socket proxy that needs GDRCopy. The RTX 3060 pair has none of them, and
   NCCL says so: ncclCommQueryProperties reports hostRmaSupport = false, a
   window registers as NULL, and all three calls fail with ncclInvalidArgument
   ("One sided RMA: host RMA is not supported in this communicator"), whether
   or not the communicator was configured with numRmaCtx and numRmaSig, whether
   the arguments are sane, and for any peer (measured on 2.31.2). That is the
   behaviour here, for the same two reasons: ncclCommQueryProperties says the
   same, and a put would need the window to be real -- a region of the target's
   device memory that another process's simulated device exposes -- which a
   file-backed transport cannot do without a progress thread in every process,
   a design this transport does not have and whose error behaviour (signal
   indices, contexts, ordering against the stream) there is no card here to
   measure. A program checks hostRmaSupport first, as the API intends. */
namespace {
ncclResult_t rma_refused(const char* fn, ncclComm_t comm) {
  if (!comm) std::fprintf(stderr, "[vgpu] nccl: %s : comm argument is NULL\n", fn);
  else std::fprintf(stderr, "[vgpu] nccl: One sided RMA: host RMA is not supported in this communicator (%s)\n", fn);
  return ncclInvalidArgument;
}
}  // namespace

VGPU_EXPORT ncclResult_t ncclPutSignal(const void*, size_t, ncclDataType_t, int, ncclWindow_t, size_t,
                                       int, int, unsigned int, ncclComm_t comm, cudaStream_t) {
  return rma_refused("ncclPutSignal", comm);
}
VGPU_EXPORT ncclResult_t ncclSignal(int, int, int, unsigned int, ncclComm_t comm, cudaStream_t) {
  return rma_refused("ncclSignal", comm);
}
VGPU_EXPORT ncclResult_t ncclWaitSignal(int, ncclWaitSignalDesc_t*, ncclComm_t comm, cudaStream_t) {
  return rma_refused("ncclWaitSignal", comm);
}
#endif

/* ---- user-defined reduction operators ---- */

// card: a NULL comm is ncclInvalidArgument, an invalid datatype
// ncclInternalError, and a residence other than ncclScalarHostImmediate is
// taken as device residence. NCCL 2.29.7 crashes on a NULL op or scalar; those
// are ncclInvalidArgument here. Handles are per communicator and a destroyed
// one's handle is handed out again.
VGPU_EXPORT ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* scalar,
                                                  ncclDataType_t dt,
                                                  ncclScalarResidence_t residence,
                                                  ncclComm_t comm) {
  auto* c = reinterpret_cast<Comm*>(comm);
  if (ncclResult_t r = check_comm(c); r != ncclSuccess) return r;
  if (!op || !scalar) return ncclInvalidArgument;
  const size_t es = type_size(dt);
  if (es == 0) return ncclInternalError;
  UserOp u;
  u.live = true;
  u.dt = dt;
  u.on_device = residence != ncclScalarHostImmediate;
  if (u.on_device) u.dev = scalar;
  else std::memcpy(u.host, scalar, es);   // dereferenced now, as documented
  std::lock_guard<std::mutex> l(c->mu);
  size_t idx = 0;
  while (idx < c->redops.size() && c->redops[idx].live) ++idx;
  if (idx == c->redops.size()) {
    if (idx >= kRedopSlots) return ncclInternalError;
    c->redops.push_back(u);
  } else {
    c->redops[idx] = u;
  }
  *op = redop_handle(c, idx);
  return ncclSuccess;
}

// card: a builtin, a NULL comm, another communicator's operator or one already
// destroyed are all ncclInvalidArgument.
VGPU_EXPORT ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  auto* c = reinterpret_cast<Comm*>(comm);
  if (op >= 0 && op < ncclNumOps) {
    std::fprintf(stderr, "[vgpu] nccl: ncclRedOpDestroy : operator is a NCCL builtin.\n");
    return ncclInvalidArgument;
  }
  if (!c) {
    std::fprintf(stderr, "[vgpu] nccl: ncclRedOpDestroy : invalid communicator passed.\n");
    return ncclInvalidArgument;
  }
  std::lock_guard<std::mutex> l(c->mu);
  const long idx = redop_slot(c, op);
  if (idx < 0) {
    std::fprintf(stderr, "[vgpu] nccl: ncclRedOpDestroy : operator unknown to this communicator.\n");
    return ncclInvalidArgument;
  }
  c->redops[idx].live = false;
  return ncclSuccess;
}


#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
/* ---- runtime parameters: ncclParam* ----
   NCCL 2.30 added a registry of its tunables, readable by key. What NCCL 2.31.2
   registers is small: six public parameters, all of them about debug output,
   and three private ones that ncclParamGetAllParameterKeys and
   ncclParamDumpAll show only when NCCL_PARAM_DUMP_ALL is true (measured on the
   RTX 3060 pair, after communicator creation and after collectives alike --
   the rest of NCCL's environment variables are not in the registry). The
   registry is reproduced as measured: each key's type, default, flags and
   documentation text, the way its environment variable is parsed, and what
   each getter says.
   card: a getter takes the parameter's own type -- ncclParamGetStr for strings,
   ncclParamGetU32 for NCCL_DEBUG_TIMESTAMP_LEVELS, ncclParamGetU64 for
   NCCL_DEBUG_SUBSYS -- and is ncclInvalidArgument for any other, which is every
   typed getter for the booleans and for NCCL_DEBUG; ncclParamGet copies the
   value's bytes (a boolean one byte, a string with its NUL, NCCL_DEBUG four)
   and is ncclInvalidArgument, leaving *len zero, when maxLen is too small; an
   unknown or NULL key, handle or output is ncclInvalidArgument, and keys are
   case-sensitive. An environment value that does not parse is ignored.
   Booleans are true for 1, t and true in any case, false for anything else.
   Flag lists are comma-separated, in any case, blanks ignored; "^" in front
   negates; ALL sets every bit; an empty one is NONE; one unknown name voids
   the lot. ncclParamDumpAll writes to stdout. */
namespace {
enum class ParamKind { Bool, Str, U32, U64, Level };
struct ParamDef {
  const char* key;
  ParamKind kind;
  const char* ctype;
  const char* flags;
  bool is_private;
  const char* desc;
  const char* accepted;
};
const ParamDef kParams[] = {
    {"NCCL_PARAM_DUMP_ALL", ParamKind::Bool, "bool", "Private", true, "Print all parameters including private ones",
     "Boolean: 1/T/TRUE or 0/F/FALSE"},
    {"NCCL_DEBUG_FILE", ParamKind::Str, "const char*", "NoEnvPluginInit", false, "Set the NCCL debug logging output to a file. The filename format can be set to filename.%h.%p where %h is replaced with the hostname and %p is replaced with the process PID. This does not accept the ~ character as part of the path, please convert to a relative or absolute path first.",
     "String"},
    {"NCCL_DEBUG_TIMESTAMP_FORMAT", ParamKind::Str, "const char*", "NoEnvPluginInit", false, "Set the format used when printing debug log messages",
     "String"},
    {"NCCL_DEBUG_TIMESTAMP_LEVELS", ParamKind::U32, "uint32_t", "NoEnvPluginInit", false, "Set which log lines get a timestamp depending upon the level of the log",
     "Comma-separated list of:\n        VERSION - on NCCL version info message\n        WARN - on Explicit error message\n        INFO - on Debug message\n        ABORT - \n        TRACE - on Replayable trace message\n        ALL - on All messages"},
    {"NCCL_NO_CACHE", ParamKind::Str, "const char*", "Private, Cached", true, "Comma-separated list of param keys to disable caching (or ALL)",
     "String"},
    {"NCCL_SET_THREAD_NAME", ParamKind::Bool, "bool", "Cached", false, "Allow NCCL to give meaningful names to NCCL CPU threads via pthread_setname_np",
     "Boolean: 1/T/TRUE or 0/F/FALSE"},
    {"NCCL_WARN_ENABLE_DEBUG_INFO", ParamKind::Bool, "bool", "Private, NoEnvPluginInit", true, "If enabled, the debug level will be set to INFO after a WARN level debug message is logged.",
     "Boolean: 1/T/TRUE or 0/F/FALSE"},
    {"NCCL_DEBUG_SUBSYS", ParamKind::U64, "uint64_t", "NoEnvPluginInit", false, "Filter debug output by (comma-separated)",
     "Comma-separated list of:\n        INIT - NCCL and comm initialization (included in default)\n        COLL - Collective operations\n        P2P - Peer-to-peer transport\n        SHM - Shared memory transport\n        NET - Network transport\n        GRAPH - Graph search and topology\n        TUNING - Algorithm tuning\n        ENV - Parameter settings by config file, EnvVar or EnvPlugins (included in default)\n        ALLOC - Device memory allocation\n        ALLOC_HOST - Host memory allocation\n        CALL - API call tracing\n        PROXY - Proxy thread operations\n        NVLS - NVLink SHARP operations\n        BOOTSTRAP - Bootstrap network (included in default)\n        REG - Buffer registration\n        PROFILE - Profiling\n        RAS - Reliability, availability, serviceability\n        DESTROY - Communicator destroy, abort, revoke, and plugin unload/close operations\n        ALL - All categories"},
    {"NCCL_DEBUG", ParamKind::Level, "ncclDebugLogLevel", "NoEnvPluginInit", false, "Set debug output level, the option is inclusive for any level that is less verbose than the set value",
     "One of:\n        VERSION - Prints the NCCL version info only\n        WARN - Prints only messages indicating a fatal error.\n        INFO - Prints debug message\n        ABORT - \n        TRACE - Prints replayable trace info on all calls"},
};
constexpr size_t kNumParams = sizeof kParams / sizeof kParams[0];

struct FlagName { const char* name; uint64_t bit; };
const FlagName kSubsysNames[] = {
    {"INIT", 0x1}, {"COLL", 0x2}, {"P2P", 0x4}, {"SHM", 0x8}, {"NET", 0x10}, {"GRAPH", 0x20},
    {"TUNING", 0x40}, {"ENV", 0x80}, {"ALLOC", 0x100}, {"ALLOC_HOST", 0x20000}, {"CALL", 0x200},
    {"PROXY", 0x400}, {"NVLS", 0x800}, {"BOOTSTRAP", 0x1000}, {"REG", 0x2000},
    {"PROFILE", 0x4000}, {"RAS", 0x8000}, {"DESTROY", 0x10000}};
const FlagName kTimestampNames[] = {
    {"VERSION", 0x2}, {"WARN", 0x4}, {"INFO", 0x8}, {"ABORT", 0x10}, {"TRACE", 0x20}};
const FlagName kLevelNames[] = {
    {"VERSION", 1}, {"WARN", 2}, {"INFO", 3}, {"ABORT", 4}, {"TRACE", 5}};

std::string lower(std::string v) {
  for (char& ch : v) ch = (char)std::tolower((unsigned char)ch);
  return v;
}
std::string trim(const std::string& v) {
  size_t a = 0, b = v.size();
  while (a < b && std::isspace((unsigned char)v[a])) ++a;
  while (b > a && std::isspace((unsigned char)v[b - 1])) --b;
  return v.substr(a, b - a);
}

// A flag list. `all` is what ALL stands for.
bool parse_flags(const char* text, const FlagName* names, size_t n, uint64_t all, uint64_t* out) {
  std::string v = trim(text);
  const bool neg = !v.empty() && v[0] == '^';
  if (neg) v.erase(0, 1);
  uint64_t bits = 0;
  size_t pos = 0;
  while (!v.empty() && pos <= v.size()) {
    size_t end = v.find(',', pos);
    if (end == std::string::npos) end = v.size();
    const std::string tok = lower(trim(v.substr(pos, end - pos)));
    pos = end + 1;
    if (tok.empty()) continue;
    if (tok == "all") { bits |= all; continue; }
    bool found = false;
    for (size_t i = 0; i < n; ++i)
      if (tok == lower(names[i].name)) { bits |= names[i].bit; found = true; }
    if (!found) return false;
  }
  *out = neg ? (all & ~bits) : bits;
  return true;
}

std::string render_flags(uint64_t v, const FlagName* names, size_t n, uint64_t all, bool all_is_ones) {
  if (v == 0) return "NONE";
  if (v == all || (all_is_ones && v == ~0ull)) return "ALL";
  std::string s;
  for (size_t i = 0; i < n; ++i)
    if (v & names[i].bit) s += (s.empty() ? "" : ",") + std::string(names[i].name);
  return s;
}

// A parameter's current value: what its environment variable says if that
// parses, else the default.
struct ParamValue {
  uint64_t num = 0;       // booleans, flag lists, levels
  std::string str;        // strings
  bool from_env = false;
  bool unset = false;     // a string parameter with no value
};

ParamValue param_value(const ParamDef& d, bool use_env = true) {
  ParamValue v;
  const char* env = use_env ? std::getenv(d.key) : nullptr;
  switch (d.kind) {
    case ParamKind::Bool:
      if (env) { const std::string t = lower(trim(env)); v.num = (t == "1" || t == "t" || t == "true"); v.from_env = true; }
      break;
    case ParamKind::Str:
      if (env) { v.str = env; v.from_env = true; }
      else if (std::strcmp(d.key, "NCCL_DEBUG_TIMESTAMP_FORMAT") == 0) v.str = "[%F %T] ";
      else v.unset = true;
      break;
    case ParamKind::U32:
      v.num = 0x4;   // WARN
      if (env) { uint64_t b; if (parse_flags(env, kTimestampNames, 5, 0x3e, &b)) { v.num = b; v.from_env = true; } }
      break;
    case ParamKind::U64:
      v.num = 0x1081;   // INIT,ENV,BOOTSTRAP
      if (env) { uint64_t b; if (parse_flags(env, kSubsysNames, 18, ~0ull, &b)) { v.num = b; v.from_env = true; } }
      break;
    case ParamKind::Level:
      if (env) {
        const std::string t = lower(trim(env));
        for (const auto& n : kLevelNames) if (t == lower(n.name)) { v.num = (uint64_t)n.bit; v.from_env = true; }
      }
      break;
  }
  return v;
}

std::string render_value(const ParamDef& d, const ParamValue& v) {
  switch (d.kind) {
    case ParamKind::Bool: return v.num ? "TRUE" : "FALSE";
    case ParamKind::Str: return v.str;
    case ParamKind::U32: return render_flags(v.num, kTimestampNames, 5, 0x3e, false);
    case ParamKind::U64: return render_flags(v.num, kSubsysNames, 18, ~0ull, true);
    case ParamKind::Level:
      for (const auto& n : kLevelNames) if (n.bit == v.num) return n.name;
      return "<unknown>";
  }
  return "";
}

// Bytes ncclParamGet hands out.
std::vector<unsigned char> param_bytes(const ParamDef& d, const ParamValue& v) {
  std::vector<unsigned char> b;
  switch (d.kind) {
    case ParamKind::Bool: b.push_back(v.num ? 1 : 0); break;
    case ParamKind::Str: b.assign(v.str.begin(), v.str.end()); b.push_back(0); break;
    case ParamKind::U32: case ParamKind::Level: {
      const uint32_t x = (uint32_t)v.num; b.resize(4); std::memcpy(b.data(), &x, 4); break;
    }
    case ParamKind::U64: b.resize(8); std::memcpy(b.data(), &v.num, 8); break;
  }
  return b;
}

// Handles are the addresses of table entries; anything else is not a handle.
const ParamDef* handle_def(ncclParamHandle_t h) {
  const uintptr_t p = reinterpret_cast<uintptr_t>(h);
  const uintptr_t lo = reinterpret_cast<uintptr_t>(kParams);
  if (p >= lo && p < lo + sizeof kParams && (p - lo) % sizeof(ParamDef) == 0)
    return reinterpret_cast<const ParamDef*>(p);
  return nullptr;
}
const ParamDef* find_param(const char* key) {
  if (!key) return nullptr;
  for (const auto& d : kParams) if (std::strcmp(d.key, key) == 0) return &d;
  return nullptr;
}
bool dump_all_private() {
  ParamValue v = param_value(*find_param("NCCL_PARAM_DUMP_ALL"));
  return v.num != 0;
}

template <class T>
ncclResult_t param_typed(ncclParamHandle_t h, T* out, ParamKind want) {
  const ParamDef* d = handle_def(h);
  if (!d || !out) {
    std::fprintf(stderr, "[vgpu] nccl: ncclParamGet: invalid handle or output\n");
    return ncclInvalidArgument;
  }
  if (d->kind != want) {
    std::fprintf(stderr, "[vgpu] nccl: PARAM: type mismatch for key \"%s\"\n", d->key);
    return ncclInvalidArgument;
  }
  *out = static_cast<T>(param_value(*d).num);
  return ncclSuccess;
}
}  // namespace

VGPU_EXPORT ncclResult_t ncclParamBind(ncclParamHandle_t* out, const char* key) {
  if (!out || !key) return ncclInvalidArgument;
  const ParamDef* d = find_param(key);
  if (!d) {
    std::fprintf(stderr, "[vgpu] nccl: PARAM: key \"%s\" not found in registry\n", key);
    return ncclInvalidArgument;
  }
  *out = reinterpret_cast<ncclParamHandle_t>(const_cast<ParamDef*>(d));
  return ncclSuccess;
}

// No signed or 8/16-bit parameter exists in the registry, so these are the
// type mismatch the card reports for everything -- but they are still the ABI.
namespace {
template <class T>
ncclResult_t param_never(ncclParamHandle_t h, T* out) {
  const ParamDef* d = handle_def(h);
  if (!d || !out) return ncclInvalidArgument;
  std::fprintf(stderr, "[vgpu] nccl: PARAM: type mismatch for key \"%s\"\n", d->key);
  return ncclInvalidArgument;
}
}  // namespace
VGPU_EXPORT ncclResult_t ncclParamGetI8(ncclParamHandle_t h, int8_t* out) { return param_never(h, out); }
VGPU_EXPORT ncclResult_t ncclParamGetI16(ncclParamHandle_t h, int16_t* out) { return param_never(h, out); }
VGPU_EXPORT ncclResult_t ncclParamGetI32(ncclParamHandle_t h, int32_t* out) { return param_never(h, out); }
VGPU_EXPORT ncclResult_t ncclParamGetI64(ncclParamHandle_t h, int64_t* out) { return param_never(h, out); }
VGPU_EXPORT ncclResult_t ncclParamGetU8(ncclParamHandle_t h, uint8_t* out) { return param_never(h, out); }
VGPU_EXPORT ncclResult_t ncclParamGetU16(ncclParamHandle_t h, uint16_t* out) { return param_never(h, out); }
VGPU_EXPORT ncclResult_t ncclParamGetU32(ncclParamHandle_t h, uint32_t* out) {
  return param_typed<uint32_t>(h, out, ParamKind::U32);
}
VGPU_EXPORT ncclResult_t ncclParamGetU64(ncclParamHandle_t h, uint64_t* out) {
  return param_typed<uint64_t>(h, out, ParamKind::U64);
}

// The pointer stays valid until this thread's next ncclParamGetStr.
VGPU_EXPORT ncclResult_t ncclParamGetStr(ncclParamHandle_t h, const char** out) {
  const ParamDef* d = handle_def(h);
  if (!d || !out) return ncclInvalidArgument;
  if (d->kind != ParamKind::Str) {
    std::fprintf(stderr, "[vgpu] nccl: PARAM: type mismatch for key \"%s\"\n", d->key);
    return ncclInvalidArgument;
  }
  thread_local std::string keep;
  keep = param_value(*d).str;
  *out = keep.c_str();
  return ncclSuccess;
}

VGPU_EXPORT ncclResult_t ncclParamGet(ncclParamHandle_t h, void* out, int maxLen, int* len) {
  const ParamDef* d = handle_def(h);
  if (!d || !out || !len || maxLen <= 0) return ncclInvalidArgument;
  const std::vector<unsigned char> b = param_bytes(*d, param_value(*d));
  if ((size_t)maxLen < b.size()) {
    *len = 0;
    return ncclInvalidArgument;
  }
  std::memcpy(out, b.data(), b.size());
  *len = (int)b.size();
  return ncclSuccess;
}

// The value as text, by key. The pointer stays valid until this thread's next call.
VGPU_EXPORT ncclResult_t ncclParamGetParameter(const char* key, const char** value, int* valueLen) {
  if (!key || !value || !valueLen) return ncclInvalidArgument;
  const ParamDef* d = find_param(key);
  if (!d) {
    std::fprintf(stderr, "[vgpu] nccl: PARAM: key \"%s\" not found in registry\n", key);
    *value = nullptr;
    *valueLen = 0;
    return ncclInvalidArgument;
  }
  thread_local std::string keep;
  keep = render_value(*d, param_value(*d));
  *value = keep.c_str();
  *valueLen = (int)keep.size();
  return ncclSuccess;
}

// The table and its keys are valid until this thread's next call.
VGPU_EXPORT ncclResult_t ncclParamGetAllParameterKeys(const char*** table, int* tableLen) {
  if (!table || !tableLen) return ncclInvalidArgument;
  thread_local std::vector<const char*> keys;
  keys.clear();
  const bool every = dump_all_private();
  for (const auto& d : kParams)
    if (every || !d.is_private) keys.push_back(d.key);
  *table = keys.data();
  *tableLen = (int)keys.size();
  return ncclSuccess;
}

VGPU_EXPORT void ncclParamDumpAll(void) {
  const bool every = dump_all_private();
  std::string out = "=== ncclParam Registry Dump ===\n";
  for (const auto& d : kParams) {
    if (d.is_private && !every) continue;
    const ParamValue cur = param_value(d);
    const ParamValue def = param_value(d, false);
    auto show = [&](const ParamValue& v) {
      const std::string t = render_value(d, v);
      return d.kind == ParamKind::Str && v.unset ? std::string("<unset>") : t;
    };
    out += std::string(d.key) + " (" + d.ctype + ") [" + d.flags + "] " + d.desc + "\n";
    out += "    Current value=" + show(cur) + " set_by=" + (cur.from_env ? "EnvPlugin" : "Default") +
           " default=" + show(def) + "\n";
    out += std::string("    Accepted value: ") + d.accepted + "\n\n";
  }
  std::fwrite(out.data(), 1, out.size(), stdout);
  std::fflush(stdout);
}
#endif


#if defined(VGPU_NCCL_HAVE_DEVICE_API)
/* ---- the device API, host side ----
   Kernels that use ncclDevComm (LSA, multimem, GIN) read and write peer memory
   through addresses the library builds into a device communicator. There is no
   such address for another process's simulated device, and a file cannot make
   one, so no communicator here supports the device API. That is the state the
   RTX 3060 pair reports -- deviceApiSupport = false, hostRmaSupport = false,
   no GIN, no multimem -- and everything below answers as NCCL answers there:
   ncclDevCommCreate is ncclInvalidUsage, windows are NULL (see
   ncclCommWindowRegister) and so every window query is ncclInvalidArgument. A
   program that checks ncclCommQueryProperties first, as the API intends, takes
   its fallback path; one that does not is told why. The team and requirement
   helpers are plain host arithmetic and work. ncclTeamLsa is the whole
   communicator, as NCCL reports for ranks that share a node (the card's
   ncclTeamLsa has two ranks too, although its device API is unsupported);
   nothing is reachable through it without a device communicator.
   card (2.31.2): ncclCommQueryProperties wants an initialized struct (magic
   NCCL_API_MAGIC and a version NCCL can read, else ncclInvalidUsage), and
   fills what fits in `size`; ncclDevCommCreate checks its pointers
   (ncclInvalidArgument), then that the requirements struct is initialized
   (ncclInvalidUsage), then the support; ncclDevCommDestroy accepts anything.
   devCommRuntimeVersionSize is 248 bytes for 2.31.2. */
namespace {
constexpr size_t kDevCommRuntimeVersionSize = 248;

Comm* comm_or_null(ncclComm_t c) { return reinterpret_cast<Comm*>(c); }

ncclResult_t window_query_refused(const char* fn, ncclWindow_t win) {
  std::fprintf(stderr, win ? "[vgpu] nccl: %s : no communicator owns this window (windows are NULL here)\n"
                           : "[vgpu] nccl: %s : window argument is NULL\n", fn);
  return ncclInvalidArgument;
}
}  // namespace

VGPU_EXPORT ncclResult_t ncclCommQueryProperties(ncclComm_t comm, ncclCommProperties_t* props) {
  Comm* c = comm_or_null(comm);
  if (!c) { std::fprintf(stderr, "[vgpu] nccl: ncclCommQueryProperties : comm argument is NULL\n"); return ncclInvalidArgument; }
  if (!props) { std::fprintf(stderr, "[vgpu] nccl: ncclCommQueryProperties : props argument is NULL\n"); return ncclInvalidArgument; }
  if (props->magic != NCCL_API_MAGIC) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommProperties_t argument must be initialized via NCCL_COMM_PROPERTIES_INITIALIZER\n");
    return ncclInvalidUsage;
  }
  if (props->version < NCCL_VERSION(2, 28, 0) || props->version > NCCL_VERSION_CODE) {
    std::fprintf(stderr, "[vgpu] nccl: ncclCommProperties_t was built for NCCL code %u, this is %d\n",
                 props->version, (int)NCCL_VERSION_CODE);
    return ncclInvalidUsage;
  }
  if (ncclResult_t r = check_comm(c); r != ncclSuccess) return r;
  ncclCommProperties_t out = *props;
  out.rank = c->rank;
  out.nRanks = c->nranks;
  out.cudaDev = c->cuda_dev;
  out.nvmlDev = c->cuda_dev;
  out.deviceApiSupport = false;
  out.multimemSupport = false;
  out.ginType = NCCL_GIN_TYPE_NONE;
  out.nLsaTeams = 1;
  out.hostRmaSupport = false;
  out.railedGinType = NCCL_GIN_TYPE_NONE;
  uint64_t h[2];
  hash128(c->rz->base, h);
  out.commHash = h[0];            // the same on every rank of the communicator
  out.ginMinStride = c->nranks;
  out.ginConnectionType = NCCL_GIN_CONNECTION_NONE;
  std::memset(out.ginSupport, 0, sizeof out.ginSupport);
  out.devCommRuntimeVersionSize = kDevCommRuntimeVersionSize;
  std::memcpy(props, &out, std::min(props->size, sizeof out));
  return ncclSuccess;
}

VGPU_EXPORT ncclResult_t ncclDevCommCreate(ncclComm_t comm, ncclDevCommRequirements_t const* reqs,
                                           ncclDevComm_t* outDevComm) {
  (void)outDevComm;
  Comm* c = comm_or_null(comm);
  if (!c) { std::fprintf(stderr, "[vgpu] nccl: ncclDevCommCreate : comm argument is NULL\n"); return ncclInvalidArgument; }
  if (!reqs) { std::fprintf(stderr, "[vgpu] nccl: ncclDevCommCreate : reqs argument is NULL\n"); return ncclInvalidArgument; }
  if (reqs->magic != NCCL_API_MAGIC) {
    std::fprintf(stderr, "[vgpu] nccl: ncclDevCommRequirements_t argument must be initialized via "
                         "NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER\n");
    return ncclInvalidUsage;
  }
  std::fprintf(stderr, "[vgpu] nccl: ncclDevCommCreate: this communicator does not support the device "
                       "API (no peer mappings between ranks; see ncclCommQueryProperties)\n");
  return ncclInvalidUsage;
}

VGPU_EXPORT ncclResult_t ncclDevCommDestroy(ncclComm_t comm, ncclDevComm_t const* devComm) {
  if (!comm_or_null(comm)) { std::fprintf(stderr, "[vgpu] nccl: ncclDevCommDestroy : comm argument is NULL\n"); return ncclInvalidArgument; }
  if (!devComm) { std::fprintf(stderr, "[vgpu] nccl: ncclDevCommDestroy : devComm argument is NULL\n"); return ncclInvalidArgument; }
  return ncclSuccess;   // nothing was ever created
}

VGPU_EXPORT ncclResult_t ncclGetLsaMultimemDevicePointer(ncclWindow_t window, size_t, void** outPtr) {
  (void)outPtr;
  return window_query_refused("ncclGetLsaMultimemDevicePointer", window);
}
VGPU_EXPORT ncclResult_t ncclGetMultimemDevicePointer(ncclWindow_t window, size_t,
                                                      ncclMultimemHandle_t, void** outPtr) {
  (void)outPtr;
  return window_query_refused("ncclGetMultimemDevicePointer", window);
}
VGPU_EXPORT ncclResult_t ncclGetLsaDevicePointer(ncclWindow_t window, size_t, int, void** outPtr) {
  (void)outPtr;
  return window_query_refused("ncclGetLsaDevicePointer", window);
}
VGPU_EXPORT ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t, int, void** outPtr) {
  (void)outPtr;
  return window_query_refused("ncclGetPeerDevicePointer", window);
}
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
VGPU_EXPORT ncclResult_t ncclGetMultimemDeviceLeInfo(ncclWindow_t window, size_t, ncclCftLeId*, size_t*) {
  return window_query_refused("ncclGetMultimemDeviceLeInfo", window);
}
VGPU_EXPORT ncclResult_t ncclGetCftDeviceLeInfo(ncclWindow_t window, size_t, int, ncclTeam_t,
                                                ncclCftLeId*, size_t*) {
  return window_query_refused("ncclGetCftDeviceLeInfo", window);
}
VGPU_EXPORT ncclResult_t ncclGetPeerDeviceLeInfo(ncclWindow_t window, size_t, int, ncclCftLeId*, size_t*) {
  return window_query_refused("ncclGetPeerDeviceLeInfo", window);
}
#endif

// card: world, LSA (every rank of the node: two on the RTX 3060 pair) and rail
// (the ranks one per node: one rank, with a stride of the LSA size) teams.
VGPU_EXPORT ncclTeam_t ncclTeamWorld(ncclComm_t comm) {
  Comm* c = comm_or_null(comm);
  return ncclTeam_t{c ? c->nranks : 0, c ? c->rank : 0, 1};
}
VGPU_EXPORT ncclTeam_t ncclTeamLsa(ncclComm_t comm) {
  Comm* c = comm_or_null(comm);
  return ncclTeam_t{c ? c->nranks : 0, c ? c->rank : 0, 1};
}
VGPU_EXPORT ncclTeam_t ncclTeamRail(ncclComm_t comm) {
  Comm* c = comm_or_null(comm);
  return ncclTeam_t{1, 0, c ? c->nranks : 1};
}
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
VGPU_EXPORT ncclTeam_t ncclTeamCft(ncclComm_t, ncclCftTeamMode_t) { return ncclTeam_t{1, 0, 1}; }
VGPU_EXPORT ncclTeam_t ncclTeamCftMultimem(ncclComm_t) { return ncclTeam_t{1, 0, 1}; }
#endif
// The world rank of `rank` within `team`, which contains the calling rank.
VGPU_EXPORT int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  Comm* c = comm_or_null(comm);
  return (c ? c->rank : 0) + (rank - team.rank) * team.stride;
}
// The LSA rank of `rank` within `team`: the LSA team is the whole communicator.
VGPU_EXPORT int ncclTeamRankToLsa(ncclComm_t comm, ncclTeam_t team, int rank) {
  return ncclTeamRankToWorld(comm, team, rank);
}

/* The requirement helpers fill in what a device communicator would have to
   reserve. card: they are plain arithmetic, reproduced as measured -- a
   barrier's buffer is nBarriers * (12 + 4 * nRanks) bytes, 4-aligned, the
   handle remembering nBarriers; the LL all-to-all's is nBlocks * (16 + 32 *
   nSlots), 16-aligned; a GIN barrier reserves nBarriers * nRanks signals and
   no buffer. ncclLLA2ACalcSlots is maxElts * ceil(maxEltSize / 8). The card
   dereferences a NULL handle or requirement; here that is ncclInvalidArgument. */
namespace {
void zero_requirements(ncclDevResourceRequirements_t* r) {
  std::memset(r, 0, sizeof *r);
}
}  // namespace

VGPU_EXPORT ncclResult_t ncclLsaBarrierCreateRequirement(ncclTeam_t team, int nBarriers,
                                                         ncclLsaBarrierHandle_t* outHandle,
                                                         ncclDevResourceRequirements_t* outReq) {
  if (!outHandle || !outReq || nBarriers < 0 || team.nRanks < 1) return ncclInvalidArgument;
  outHandle->nBarriers = nBarriers;
  zero_requirements(outReq);
  outReq->bufferSize = (size_t)nBarriers * (12 + 4 * (size_t)team.nRanks);
  outReq->bufferAlign = 4;
  outReq->outBufferHandle = &outHandle->bufHandle;
  return ncclSuccess;
}

VGPU_EXPORT int ncclLLA2ACalcSlots(int maxElts, int maxEltSize) {
  return maxElts * ((maxEltSize + 7) / 8);
}

VGPU_EXPORT ncclResult_t ncclLLA2ACreateRequirement(int nBlocks, int nSlots,
                                                    ncclLLA2AHandle_t* outHandle,
                                                    ncclDevResourceRequirements_t* outReq) {
  if (!outHandle || !outReq || nBlocks < 0 || nSlots < 0) return ncclInvalidArgument;
  outHandle->nSlots = (uint32_t)nSlots;
  zero_requirements(outReq);
  outReq->bufferSize = (size_t)nBlocks * (16 + 32 * (size_t)nSlots);
  outReq->bufferAlign = 16;
  outReq->outBufferHandle = &outHandle->bufHandle;
  return ncclSuccess;
}

VGPU_EXPORT ncclResult_t ncclGinBarrierCreateRequirement(ncclComm_t comm, ncclTeam_t team,
                                                         int nBarriers,
                                                         ncclGinBarrierHandle_t* outHandle,
                                                         ncclDevResourceRequirements_t* outReq) {
  if (!comm_or_null(comm) || !outHandle || !outReq || nBarriers < 0 || team.nRanks < 1)
    return ncclInvalidArgument;
  zero_requirements(outReq);
  outReq->ginSignalCount = nBarriers * team.nRanks;
  outReq->outGinSignalStart = &outHandle->signal0;
  return ncclSuccess;
}
#endif

// Deprecated in nccl.h ("not part of the NCCL API"); there is no cached debug level to reload.
VGPU_EXPORT void (ncclResetDebugInit)(void) {}
VGPU_EXPORT void ncclResetDebugInitInternal(void) {}

/* ---- profiling aliases ----
   Real NCCL exports every entry point twice: nccl* as a weak alias of the
   pnccl* symbol, so a profiler can interpose. Tools that link the pnccl names
   directly should find them here too. */
#define VGPU_ALIAS(name) \
  VGPU_EXPORT __typeof__(name) p##name __attribute__((alias(#name)))
VGPU_ALIAS(ncclGetVersion);
VGPU_ALIAS(ncclGetUniqueId);
VGPU_ALIAS(ncclGetErrorString);
VGPU_ALIAS(ncclGetLastError);
VGPU_ALIAS(ncclCommInitRank);
VGPU_ALIAS(ncclCommInitRankConfig);
VGPU_ALIAS(ncclCommInitAll);
VGPU_ALIAS(ncclCommDestroy);
VGPU_ALIAS(ncclCommFinalize);
VGPU_ALIAS(ncclCommAbort);
VGPU_ALIAS(ncclCommCount);
VGPU_ALIAS(ncclCommUserRank);
VGPU_ALIAS(ncclCommCuDevice);
VGPU_ALIAS(ncclCommGetAsyncError);
VGPU_ALIAS(ncclCommRegister);
VGPU_ALIAS(ncclCommDeregister);
VGPU_ALIAS(ncclMemAlloc);
VGPU_ALIAS(ncclMemFree);
VGPU_ALIAS(ncclGroupStart);
VGPU_ALIAS(ncclGroupEnd);
VGPU_ALIAS(ncclGroupSimulateEnd);
VGPU_ALIAS(ncclAllReduce);
VGPU_ALIAS(ncclReduce);
VGPU_ALIAS(ncclBroadcast);
VGPU_ALIAS(ncclBcast);
VGPU_ALIAS(ncclAllGather);
VGPU_ALIAS(ncclReduceScatter);
VGPU_ALIAS(ncclSend);
VGPU_ALIAS(ncclRecv);
VGPU_ALIAS(ncclRedOpCreatePreMulSum);
VGPU_ALIAS(ncclRedOpDestroy);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 18, 0)
VGPU_ALIAS(ncclCommSplit);
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 23, 0)
VGPU_ALIAS(ncclCommInitRankScalable);
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
VGPU_ALIAS(ncclCommShrink);
VGPU_ALIAS(ncclCommWindowRegister);
VGPU_ALIAS(ncclCommWindowDeregister);
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
VGPU_ALIAS(ncclAlltoAll);
VGPU_ALIAS(ncclGather);
VGPU_ALIAS(ncclScatter);
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
VGPU_ALIAS(ncclWinGetUserPtr);
VGPU_ALIAS(ncclCommRevoke);
VGPU_ALIAS(ncclCommSuspend);
VGPU_ALIAS(ncclCommResume);
VGPU_ALIAS(ncclCommMemStats);
VGPU_ALIAS(ncclCommGetUniqueId);
VGPU_ALIAS(ncclCommGrow);
VGPU_ALIAS(ncclPutSignal);
VGPU_ALIAS(ncclSignal);
VGPU_ALIAS(ncclWaitSignal);
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
VGPU_ALIAS(ncclAllReduceConfig);
VGPU_ALIAS(ncclBroadcastConfig);
VGPU_ALIAS(ncclReduceConfig);
VGPU_ALIAS(ncclAllGatherConfig);
VGPU_ALIAS(ncclReduceScatterConfig);
VGPU_ALIAS(ncclAlltoAllConfig);
VGPU_ALIAS(ncclGatherConfig);
VGPU_ALIAS(ncclScatterConfig);
VGPU_ALIAS(ncclParamBind);
VGPU_ALIAS(ncclParamGetI8);
VGPU_ALIAS(ncclParamGetI16);
VGPU_ALIAS(ncclParamGetI32);
VGPU_ALIAS(ncclParamGetI64);
VGPU_ALIAS(ncclParamGetU8);
VGPU_ALIAS(ncclParamGetU16);
VGPU_ALIAS(ncclParamGetU32);
VGPU_ALIAS(ncclParamGetU64);
VGPU_ALIAS(ncclParamGetStr);
VGPU_ALIAS(ncclParamGet);
VGPU_ALIAS(ncclParamGetParameter);
VGPU_ALIAS(ncclParamGetAllParameterKeys);
VGPU_ALIAS(ncclParamDumpAll);
#endif
#if defined(VGPU_NCCL_HAVE_DEVICE_API)
VGPU_ALIAS(ncclCommQueryProperties);
VGPU_ALIAS(ncclDevCommCreate);
VGPU_ALIAS(ncclDevCommDestroy);
VGPU_ALIAS(ncclGetLsaMultimemDevicePointer);
VGPU_ALIAS(ncclGetMultimemDevicePointer);
VGPU_ALIAS(ncclGetLsaDevicePointer);
VGPU_ALIAS(ncclGetPeerDevicePointer);
VGPU_ALIAS(ncclTeamWorld);
VGPU_ALIAS(ncclTeamLsa);
VGPU_ALIAS(ncclTeamRail);
VGPU_ALIAS(ncclTeamRankToWorld);
VGPU_ALIAS(ncclTeamRankToLsa);
VGPU_ALIAS(ncclLsaBarrierCreateRequirement);
VGPU_ALIAS(ncclLLA2ACalcSlots);
VGPU_ALIAS(ncclLLA2ACreateRequirement);
VGPU_ALIAS(ncclGinBarrierCreateRequirement);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
VGPU_ALIAS(ncclGetMultimemDeviceLeInfo);
VGPU_ALIAS(ncclGetCftDeviceLeInfo);
VGPU_ALIAS(ncclGetPeerDeviceLeInfo);
VGPU_ALIAS(ncclTeamCft);
VGPU_ALIAS(ncclTeamCftMultimem);
#endif
#endif
VGPU_ALIAS(ncclResetDebugInit);
