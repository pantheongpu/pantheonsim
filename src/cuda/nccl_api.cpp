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
// Not implemented: the network plugin interface, user-defined reduction
// operators (ncclRedOpCreatePreMulSum), symmetric memory windows, and the
// non-blocking config. Those return ncclInvalidUsage rather than a wrong answer.
#include <nccl.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace {

constexpr int kMaxRanks = 64;          // a simulated rack, not a real cluster
constexpr uint32_t kMagic = 0x4e470756;  // "V\a GN"
constexpr uint32_t kVersion = 1;

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
  uint32_t pad;
  std::atomic<uint32_t> joined;
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

struct Comm {
  Rendezvous* rz = nullptr;
  int rank = 0;
  int nranks = 1;
  int cuda_dev = 0;
  uint64_t seq = 0;   // collectives issued on this communicator so far
  ncclResult_t async = ncclSuccess;
};

std::mutex g_mu;
std::unordered_map<std::string, Rendezvous*> g_rendezvous;

bool mkdir_p(const std::string& p) {
  if (::mkdir(p.c_str(), 0700) == 0 || errno == EEXIST) return true;
  return false;
}

// Create the metadata file exactly once, whichever rank gets there first.
// Build it under a private name and link() it into place: link fails with
// EEXIST rather than truncating, so no rank can ever see a half-built segment.
Rendezvous* attach(const ncclUniqueId& id, int nranks, std::string* err) {
  const std::string dir = rendezvous_dir();
  if (!mkdir_p(dir)) { *err = "cannot create " + dir; return nullptr; }
  const std::string base = dir + "/" + hex16(id);
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

// Spin-wait with a deadline. Collectives that never match up are the single
// most common NCCL bug, so time out with a diagnosis rather than hanging.
template <class Pred>
bool wait_for(Pred done, const char* what, int rank) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_seconds());
  int spins = 0;
  while (!done()) {
    if (++spins > 512) {
      timespec ts{0, 200000};  // 0.2 ms
      nanosleep(&ts, nullptr);
      if (std::chrono::steady_clock::now() > deadline) {
        std::fprintf(stderr,
                     "[vgpu] nccl: rank %d timed out waiting for %s after %.0fs. Ranks must "
                     "call the same collectives in the same order; set VGPU_NCCL_TIMEOUT to "
                     "raise the limit.\n",
                     rank, what, timeout_seconds());
        return false;
      }
    }
  }
  return true;
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

/* ---- group buffering ----
   NCCL lets one thread issue every rank's call between ncclGroupStart and
   ncclGroupEnd; the calls only have to match up by the time the group closes.
   A barrier taken inside the call itself would deadlock that pattern, so the
   ops are recorded and run in four passes at ncclGroupEnd -- deposit all,
   collect all, release all, wait for all -- and each pass finishes for every
   op before the next begins. A call made outside a group is just a group of
   one, which reduces to deposit-collect-release-wait in order. */

enum class Kind { AllReduce, Broadcast, Reduce, AllGather, ReduceScatter, Send, Recv };

struct Op {
  Kind kind;
  Comm* comm;
  const void* send;
  void* recv;
  size_t count;          // elements: per-rank for AllGather, per-rank output for ReduceScatter
  ncclDataType_t dt;
  ncclRedOp_t red;
  int root = 0;
  int peer = 0;
  cudaStream_t stream = nullptr;
  uint64_t seq = 0;
  std::vector<char> staging;   // host copy of the send buffer / assembled output
  bool ok = true;
};

thread_local int t_group_depth = 0;
thread_local std::vector<Op> t_pending;

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

// How many bytes this rank publishes for the given op.
size_t deposit_bytes(const Op& op) {
  const size_t es = type_size(op.dt);
  switch (op.kind) {
    case Kind::AllGather: return op.count * es;                       // each rank's slice
    case Kind::ReduceScatter: return op.count * op.comm->nranks * es;  // whole input
    case Kind::Broadcast: return op.comm->rank == op.root ? op.count * es : 0;
    default: return op.count * es;
  }
}

ncclResult_t deposit(Op& op) {
  Comm* c = op.comm;
  const size_t bytes = deposit_bytes(op);
  if (op.kind == Kind::Recv) return ncclSuccess;  // receivers publish nothing

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
      cudaStreamSynchronize(op.stream);
      if (bytes && cudaMemcpy(m.addr, op.send, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        return ncclUnhandledCudaError;
    }
    msync(m.addr, m.len, MS_SYNC);
    m.reset();   // close before publishing, so a reader never sees a short file
    c->rz->meta->bytes[c->rank][op.peer].store(bytes, std::memory_order_release);
    c->rz->meta->posted[c->rank][op.peer].store(seq, std::memory_order_release);
    return ncclSuccess;
  }

  if (bytes) {
    Mapping* m = writer_for(c->rz, c->rank, bytes);
    if (!m) return ncclSystemError;
    DeviceGuard g(c->cuda_dev);
    cudaStreamSynchronize(op.stream);
    if (cudaMemcpy(m->addr, op.send, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
      return ncclUnhandledCudaError;
    msync(m->addr, m->len, MS_ASYNC);
  } else {
    DeviceGuard g(c->cuda_dev);
    cudaStreamSynchronize(op.stream);
  }
  c->rz->meta->phase[c->rank].store(op.seq, std::memory_order_release);
  return ncclSuccess;
}

// Read a peer's published bytes. For the local rank we could reuse our own
// mapping, but reading the file keeps one code path and one set of bugs.
bool read_peer(Rendezvous* rz, int rank, size_t bytes, std::vector<char>* out) {
  out->resize(bytes);
  if (!bytes) return true;
  Mapping m;
  if (!m.open_read(rz->rank_path(rank), bytes)) return false;
  std::memcpy(out->data(), m.addr, bytes);
  return true;
}

ncclResult_t collect(Op& op) {
  Comm* c = op.comm;
  Meta* meta = c->rz->meta;
  const size_t es = type_size(op.dt);

  if (op.kind == Kind::Send) return ncclSuccess;

  if (op.kind == Kind::Recv) {
    const size_t want_msgs = meta->taken[op.peer][c->rank].load(std::memory_order_acquire) + 1;
    if (!wait_for([&] { return meta->posted[op.peer][c->rank].load(std::memory_order_acquire) >= want_msgs; },
                  "a matching ncclSend", c->rank))
      return ncclTimeout;
    const size_t bytes = op.count * es;
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
  if (!wait_for([&] {
        for (int r = 0; r < c->nranks; ++r)
          if (meta->phase[r].load(std::memory_order_acquire) < op.seq) return false;
        return true;
      }, "the other ranks to reach this collective", c->rank))
    return ncclTimeout;

  std::vector<char> peer;
  switch (op.kind) {
    case Kind::Broadcast: {
      if (!read_peer(c->rz, op.root, op.count * es, &op.staging)) return ncclSystemError;
      break;
    }
    case Kind::AllGather: {
      op.staging.resize(op.count * es * c->nranks);
      for (int r = 0; r < c->nranks; ++r) {
        if (!read_peer(c->rz, r, op.count * es, &peer)) return ncclSystemError;
        std::memcpy(op.staging.data() + (size_t)r * op.count * es, peer.data(), op.count * es);
      }
      break;
    }
    case Kind::ReduceScatter: {
      // Reduce every rank's whole input, then keep this rank's slice.
      const size_t total = op.count * c->nranks;
      std::vector<char> acc;
      if (!read_peer(c->rz, 0, total * es, &acc)) return ncclSystemError;
      for (int r = 1; r < c->nranks; ++r) {
        if (!read_peer(c->rz, r, total * es, &peer)) return ncclSystemError;
        if (!reduce(acc.data(), peer.data(), total, op.dt, op.red)) return ncclInvalidArgument;
      }
      if (op.red == ncclAvg) average(acc.data(), total, op.dt, c->nranks);
      op.staging.assign(acc.begin() + (size_t)c->rank * op.count * es,
                        acc.begin() + (size_t)(c->rank + 1) * op.count * es);
      break;
    }
    default: {  // AllReduce and Reduce
      if (op.kind == Kind::Reduce && c->rank != op.root) break;  // only the root builds a result
      if (!read_peer(c->rz, 0, op.count * es, &op.staging)) return ncclSystemError;
      for (int r = 1; r < c->nranks; ++r) {
        if (!read_peer(c->rz, r, op.count * es, &peer)) return ncclSystemError;
        if (!reduce(op.staging.data(), peer.data(), op.count, op.dt, op.red))
          return ncclInvalidArgument;
      }
      if (op.red == ncclAvg) average(op.staging.data(), op.count, op.dt, c->nranks);
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
  if (op.kind == Kind::Send || op.kind == Kind::Recv) return;
  op.comm->rz->meta->done[op.comm->rank].store(op.seq, std::memory_order_release);
}

ncclResult_t settle(Op& op) {
  if (op.kind == Kind::Send || op.kind == Kind::Recv) return ncclSuccess;
  Comm* c = op.comm;
  Meta* meta = c->rz->meta;
  if (!wait_for([&] {
        for (int r = 0; r < c->nranks; ++r)
          if (meta->done[r].load(std::memory_order_acquire) < op.seq) return false;
        return true;
      }, "the other ranks to finish this collective", c->rank))
    return ncclTimeout;
  return ncclSuccess;
}

ncclResult_t run_pending() {
  ncclResult_t rc = ncclSuccess;
  auto keep = [&](ncclResult_t r) { if (rc == ncclSuccess) rc = r; };
  for (auto& op : t_pending) keep(deposit(op));
  if (rc == ncclSuccess) for (auto& op : t_pending) keep(collect(op));
  for (auto& op : t_pending) release(op);
  if (rc == ncclSuccess) for (auto& op : t_pending) keep(settle(op));
  t_pending.clear();
  return rc;
}

ncclResult_t enqueue(Op op) {
  if (!op.comm || !op.comm->rz) return ncclInvalidArgument;
  if (type_size(op.dt) == 0) return ncclInvalidArgument;
  if (op.red < 0 || op.red >= ncclNumOps) return ncclInvalidArgument;  // no custom reducers
  op.seq = ++op.comm->seq;
  if (op.kind == Kind::Send || op.kind == Kind::Recv) --op.comm->seq;  // p2p has its own handshake
  t_pending.push_back(std::move(op));
  if (t_group_depth > 0) return ncclSuccess;
  return run_pending();
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

VGPU_EXPORT ncclResult_t ncclCommInitRank(ncclComm_t* out, int nranks, ncclUniqueId id, int rank) {
  if (!out || nranks < 1 || rank < 0 || rank >= nranks) return ncclInvalidArgument;
  if (nranks > kMaxRanks) {
    std::fprintf(stderr, "[vgpu] nccl: %d ranks requested, this build supports %d\n", nranks,
                 kMaxRanks);
    return ncclInvalidArgument;
  }
  std::string err;
  Rendezvous* rz = attach(id, nranks, &err);
  if (!rz) {
    std::fprintf(stderr, "[vgpu] nccl: %s\n", err.c_str());
    return ncclSystemError;
  }
  auto* c = new Comm();
  c->rz = rz;
  c->rank = rank;
  c->nranks = nranks;
  cudaGetDevice(&c->cuda_dev);
  rz->meta->joined.fetch_add(1, std::memory_order_release);
  if (!quiet() && rank == 0)
    std::fprintf(stderr,
                 "[vgpu] nccl: %d ranks over %s (file-backed transport; see docs/libraries.md)\n",
                 nranks, rz->base.c_str());
  *out = reinterpret_cast<ncclComm_t>(c);
  return ncclSuccess;
}

VGPU_EXPORT ncclResult_t ncclCommInitRankConfig(ncclComm_t* out, int nranks, ncclUniqueId id,
                                                int rank, ncclConfig_t* config) {
  if (config && config->blocking == 0) {
    std::fprintf(stderr, "[vgpu] nccl: non-blocking communicators are not implemented\n");
    return ncclInvalidUsage;
  }
  return ncclCommInitRank(out, nranks, id, rank);
}

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

VGPU_EXPORT ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  auto* c = reinterpret_cast<Comm*>(comm);
  if (!c) return ncclSuccess;
  if (c->rz) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (--c->rz->refs <= 0) {
      g_rendezvous.erase(c->rz->base);
      delete c->rz;  // takes its per-rank mappings with it
    }
  }
  delete c;
  return ncclSuccess;
}
VGPU_EXPORT ncclResult_t ncclCommFinalize(ncclComm_t) { return ncclSuccess; }
VGPU_EXPORT ncclResult_t ncclCommAbort(ncclComm_t comm) { return ncclCommDestroy(comm); }

VGPU_EXPORT ncclResult_t ncclCommCount(const ncclComm_t comm, int* n) {
  if (!comm || !n) return ncclInvalidArgument;
  *n = reinterpret_cast<Comm*>(comm)->nranks;
  return ncclSuccess;
}
VGPU_EXPORT ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* r) {
  if (!comm || !r) return ncclInvalidArgument;
  *r = reinterpret_cast<Comm*>(comm)->rank;
  return ncclSuccess;
}
VGPU_EXPORT ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* d) {
  if (!comm || !d) return ncclInvalidArgument;
  *d = reinterpret_cast<Comm*>(comm)->cuda_dev;
  return ncclSuccess;
}
VGPU_EXPORT ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* e) {
  if (!comm || !e) return ncclInvalidArgument;
  *e = reinterpret_cast<Comm*>(comm)->async;
  return ncclSuccess;
}
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

/* ---- groups ---- */

VGPU_EXPORT ncclResult_t ncclGroupStart(void) { ++t_group_depth; return ncclSuccess; }

VGPU_EXPORT ncclResult_t ncclGroupEnd(void) {
  if (t_group_depth == 0) return ncclInvalidUsage;
  if (--t_group_depth > 0) return ncclSuccess;
  return run_pending();
}

VGPU_EXPORT ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t* info) {
  // Nothing here models time, so report the group as free and drop it.
  if (info) info->estimatedTime = 0.0f;
  if (t_group_depth > 0) --t_group_depth;
  t_pending.clear();
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

/* ---- explicitly unimplemented ---- */

VGPU_EXPORT ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t*, void*, ncclDataType_t,
                                                  ncclScalarResidence_t, ncclComm_t) {
  std::fprintf(stderr, "[vgpu] nccl: user-defined reduction operators are not implemented\n");
  return ncclInvalidUsage;
}
VGPU_EXPORT ncclResult_t ncclRedOpDestroy(ncclRedOp_t, ncclComm_t) { return ncclSuccess; }

/* ---- profiling aliases ----
   Real NCCL exports every entry point twice: nccl* as a weak alias of the
   pnccl* symbol, so a profiler can interpose. Tools that link the pnccl names
   directly should find them here too. */
#define VGPU_ALIAS(name) \
  VGPU_EXPORT __typeof__(name) p##name __attribute__((alias(#name)))
VGPU_ALIAS(ncclGetVersion);
VGPU_ALIAS(ncclGetUniqueId);
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
