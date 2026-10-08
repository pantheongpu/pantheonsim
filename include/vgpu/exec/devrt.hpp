// The CUDA device runtime (cudadevrt, dynamic parallelism), the half both
// engines share: the PTX interpreter (src/exec/interpreter.cpp) and the SASS
// executor (src/sass/exec.cpp) decode a device-side call their own way -- a
// PTX call slot, a register pair -- and hand what they decoded to this.
//
// What a kernel can call is cuda_device_runtime_api.h's list, and what each
// call does and returns, errors included, was measured on an RTX 3060 with
// CUDA 13.0's and 12.0's toolchains (nvidia/docs/sass.md, "Device runtime").
// The numbers that matter are named where they are used.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/profile.hpp"

namespace vgpu::exec::devrt {

// cudaError_t values the device runtime returns.
enum : int {
  kSuccess = 0,
  kInvalidValue = 1,
  kMemoryAllocation = 2,
  kInvalidConfiguration = 9,
  kInvalidPitchValue = 12,
  kInvalidMemcpyDirection = 21,
  kLaunchPendingCountExceeded = 69,
  kNotSupported = 801,
  kUnknown = 999,
};

// cudaDeviceAttr, cudaLimit and the like, as the ABI passes them.
constexpr int kLimitDevRuntimePendingLaunchCount = 4;
// The fixed pool of pending launches is 32 deep: a limit below that still
// allows 32 (RTX 3060: limits 0, 1, 10, 31 and 32 all failed at the 33rd).
constexpr int64_t kMinPendingLaunches = 32;
constexpr int64_t kDefaultPendingLaunches = 2048;

// What a kernel's calls into the device runtime ask of the host's runtime:
// the device attributes, limits and configuration the host's CUDA library
// reports, and the strings it prints errors with. The library that launched
// the kernel answers (the CUDA runtime shim, or the driver shim for a
// program with a static cudart), so a kernel sees what its host sees. Every
// method returns a cudaError_t.
struct Services {
  virtual int attribute(int device, int attr, int* value) const = 0;
  virtual int limit(int device, int limit, uint64_t* value) const = 0;
  virtual int cache_config(int device, int* value) const = 0;
  virtual int shared_mem_config(int device, int* value) const = 0;
  virtual const char* error_name(int code) const = 0;
  virtual const char* error_string(int code) const = 0;

 protected:
  ~Services() = default;
};

// ---- the entry points -------------------------------------------------------------------
//
// cuda_device_runtime_api.h's functions, under the names a compiled kernel
// calls: CUDA 12 and 13 compile to __cudaCDP2<Name> (CDP2), and a program
// built with -DCUDA_FORCE_CDP1_IF_SUPPORTED calls cuda<Name> (CDP1, which
// still has cudaDeviceSynchronize). The _ptsz forms are the same call with
// the per-thread default stream compiled in.
enum class Fn {
  None,
  GetParameterBufferV2, GetParameterBuffer, LaunchDeviceV2, LaunchDevice,
  GetLastError, PeekAtLastError, GetErrorString, GetErrorName,
  GetDevice, GetDeviceCount, RuntimeGetVersion,
  StreamCreateWithFlags, StreamDestroy, StreamWaitEvent,
  EventCreateWithFlags, EventDestroy, EventRecord, EventRecordWithFlags,
  Malloc, Free,
  MemcpyAsync, Memcpy2DAsync, Memcpy3DAsync, MemsetAsync, Memset2DAsync, Memset3DAsync,
  FuncGetAttributes, DeviceGetAttribute, DeviceGetLimit, DeviceGetCacheConfig, DeviceGetSharedMemConfig,
  OccupancyMaxActiveBlocks, OccupancyMaxActiveBlocksWithFlags,
  DeviceSynchronize,
};
// The function a callee name is, or Fn::None.
Fn lookup(const std::string& callee);
// Every name lookup() knows, in a fixed order.
const std::vector<std::string>& names();

// ---- streams and events ----------------------------------------------------------------
//
// A stream argument is 0 (the block's implicit stream), 1 (legacy), 2
// (per-thread), 3 (cudaStreamTailLaunch), 4 (cudaStreamFireAndForget) or a
// handle cudaStreamCreateWithFlags returned; anything else is
// cudaErrorInvalidValue (RTX 3060: 5 to 8, 0x10, 0x100, 0x1000, the graph
// streams 0x01.. to 0x03.. and ~0 all were). The card does not remember a
// stream was destroyed: a launch into one still succeeds, and destroying it
// twice does too.
enum class StreamKind { Default, Legacy, PerThread, Tail, FireAndForget, Named, Invalid };

constexpr uint64_t kStreamBase = 0x5654'4750'0000'0000ull;   // "VTGP": never a device address
constexpr uint64_t kEventBase = 0x5654'4751'0000'0000ull;

StreamKind stream_kind(uint64_t handle);
uint64_t new_stream();
uint64_t new_event();
bool valid_event(uint64_t handle);
// cudaStreamCreateWithFlags accepts 0 and cudaStreamNonBlocking; any other
// flag is cudaErrorInvalidValue.
inline bool stream_flags_ok(uint32_t flags) { return flags <= 1; }
// cudaEventCreateWithFlags needs cudaEventDisableTiming: 2 and 3 (with
// cudaEventBlockingSync) and 10 and 11 work, every other value of 0 to 15
// is cudaErrorInvalidValue (no timing, no interprocess).
inline bool event_flags_ok(uint32_t flags) { return flags == 2 || flags == 3 || flags == 10 || flags == 11; }

// ---- asynchronous copies and fills --------------------------------------------------------

// A cudaMemcpyAsync, cudaMemset2DAsync and the rest, kept with the launches:
// it runs where it was issued in its stream's order, which is its place in
// the queue.
struct MemOp {
  enum class Kind { Copy, Set } kind = Kind::Copy;
  uint64_t dst = 0, src = 0;
  uint64_t dpitch = 0, spitch = 0;   // bytes between rows
  uint64_t dslice = 0, sslice = 0;   // bytes between slices (3D)
  uint64_t width = 0;                // bytes per row
  uint64_t height = 1, depth = 1;
  uint8_t value = 0;
  // Runs it. Reads and writes go through the device's memory, so a bad
  // address is the fault a kernel's would be.
  void run(MemoryManager& mem) const;
};

// ---- launches -----------------------------------------------------------------------------

// Whether a child grid's configuration is one the device runtime accepts
// (cudaErrorInvalidConfiguration otherwise): no zero dimension, no dimension
// or thread count past the device's, shared memory (static and dynamic) within
// its opt-in maximum, and the kernel's own launch bounds. `max_threads` is
// __launch_bounds__' bound on the block's threads (0: none), `req_block` the
// exact block a kernel asks for (zeros: none). RTX 3060: block 1025, block
// z 65, grid y 65536, zero dimensions, 100000 bytes of shared memory and a
// block past __launch_bounds__ all gave 9; with 49153 bytes of shared memory
// it gave 9 until the host had raised the kernel's
// cudaFuncAttributeMaxDynamicSharedMemorySize and 0 after (this checks the
// raised limit, the device's opt-in maximum).
bool config_ok(const std::array<uint32_t, 3>& grid, const std::array<uint32_t, 3>& block, uint64_t shared_total,
               const DeviceProfile& profile, uint64_t max_threads, const std::array<uint32_t, 3>& req_block);

// The seven fields cudaFuncGetAttributes fills in a kernel (sharedSizeBytes,
// constSizeBytes, localSizeBytes, maxThreadsPerBlock, numRegs, ptxVersion,
// binaryVersion); the rest of the struct is left as it was (RTX 3060).
struct FuncAttrs {
  uint64_t shared = 0, constant = 0, local = 0;
  int32_t max_threads = 0, regs = 0, ptx = 0, binary = 0;
};
FuncAttrs func_attributes(const ptx::EntryFn& fn, const DeviceProfile& profile, int ptx_arch);
// Writes `a` where a cudaFuncAttributes starts: the layout is the toolkit's,
// stable since CUDA 6.
void put_func_attrs(const FuncAttrs& a, uint8_t out[40]);

// What a launch's kernel is: the engine's own reference.
template <class Kernel>
struct Child {
  Kernel kernel{};
  std::array<uint32_t, 3> grid{}, block{};
  uint32_t shared = 0;
  uint64_t buffer = 0;          // a parameter buffer handed out (device memory)
  uint32_t size = 0;            // bytes the buffer holds
  bool has_kernel = false;      // false for a cudaGetParameterBuffer(alignment, size) buffer
  uint64_t order = 0;           // parent block, then issue order within it
  bool tail = false;            // cudaStreamTailLaunch
  std::vector<uint8_t> params;  // the kernel's parameters, as they were when it was launched
  std::shared_ptr<const MemOp> op;   // a copy or a fill, not a grid
};

// Dynamic parallelism's bookkeeping for the grids of one launch. `pending` is
// the device launches that have not completed -- queued, running, or waiting
// for their own children -- which is what cudaLimitDevRuntimePendingLaunchCount
// bounds (RTX 3060: a chain of grids each launching the next failed at the
// 2049th with cudaErrorLaunchPendingCountExceeded, 69, and a parent launching
// children that wait failed at the 2049th too).
struct Tree {
  std::atomic<int64_t> pending{0};
  int64_t limit = -1;   // as the launch saw it; -1 until the first launch asks
};

template <class Kernel>
struct Launches {
  std::mutex mu;
  std::unordered_map<uint64_t, Child<Kernel>> by_buffer;   // handed out, not launched yet
  std::vector<Child<Kernel>> queue;                        // launched, not started
  std::unordered_map<uint64_t, uint64_t> per_block;        // launches each parent block has issued
  std::shared_ptr<Tree> tree = std::make_shared<Tree>();
  LaunchStats extra;                                       // children that ran while the grid did
  uint32_t depth = 1;
  MemoryManager* mem = nullptr;
  Launches() = default;
  Launches(const Launches&) = delete;
  Launches& operator=(const Launches&) = delete;
  ~Launches() {
    if (!mem) return;
    for (auto& [b, c] : by_buffer) {
      try {
        mem->free(b);
      } catch (...) {
      }
    }
  }
};

// A launch chain this deep is a program that recurses without end, not one
// the card's pending limit would allow to run on a stack.
constexpr uint32_t kMaxDepth = 8192;

// How the children of a grid run, once the grid's blocks have: first every
// launch that is not a tail launch, in the order issued (a grid's parent
// block, then its order within the block, which does not depend on how the
// blocks were spread over host threads); then the tail launches, in order.
// A grid is complete only when everything it launched is, so each child's
// own children run, and its tail launches, before the next. RTX 3060: a tail
// launch saw the effects of a fire-and-forget grid launched after it, of a
// grid in a named stream, of the grids those launched, and a tail launched
// by a child ran before its parent's tail did; two tail launches ran in
// order.
//
// `run` is the engine's: operator()(Child&, Launches&) runs one grid's
// blocks, leaving what it launched in the Launches, and add_stats(LaunchStats)
// takes counters. Both are declared first because they call each other.
template <class K, class Run>
void run_one(Child<K>& c, Launches<K>& dl, MemoryManager& mem, Run& run);
template <class K, class Run>
void complete_children(Launches<K>& dl, MemoryManager& mem, Run& run);

template <class K, class Run>
void run_one(Child<K>& c, Launches<K>& dl, MemoryManager& mem, Run& run) {
  if (c.op) {
    c.op->run(mem);
    return;
  }
  if (dl.depth > kMaxDepth)
    throw Error::make(Err::LaunchConfig, "device-side launches nested ", dl.depth,
                      " deep: a chain of grids each launching the next that this simulator will not run");
  Launches<K> sub;
  sub.tree = dl.tree;
  sub.depth = dl.depth + 1;
  sub.mem = &mem;
  run(c, sub);
  complete_children(sub, mem, run);
  dl.tree->pending.fetch_sub(1, std::memory_order_relaxed);
}

template <class K, class Run>
void run_list(std::vector<Child<K>>& list, Launches<K>& dl, MemoryManager& mem, Run& run) {
  std::stable_sort(list.begin(), list.end(), [](const Child<K>& a, const Child<K>& b) { return a.order < b.order; });
  for (Child<K>& c : list)
    if (!c.tail) run_one(c, dl, mem, run);
  for (Child<K>& c : list)
    if (c.tail) run_one(c, dl, mem, run);
}

template <class K, class Run>
void complete_children(Launches<K>& dl, MemoryManager& mem, Run& run) {
  std::vector<Child<K>> list;
  {
    std::lock_guard<std::mutex> g(dl.mu);
    // Buffers taken and never launched go back.
    for (auto& [b, c] : dl.by_buffer) mem.free(b);
    dl.by_buffer.clear();
    list = std::move(dl.queue);
    dl.queue.clear();
  }
  run.add_stats(dl.extra);
  dl.extra = LaunchStats{};
  run_list(list, dl, mem, run);
}

// Runs the launches already issued, ahead of the grid's end -- the tail
// launches stay for it. A grid on the card runs its children while it runs
// (they are concurrent, and a launch that would pass the pending limit waits
// for earlier ones to finish); this does the same by running the earlier ones
// to completion: when the pending count is full, and for cudaDeviceSynchronize
// (CUDA's CDP1, which waits for the launches of the calling block: `block` is
// its linear index; ~0 takes every block's).
template <class K, class Run>
void drain(Launches<K>& dl, MemoryManager& mem, Run& run, uint64_t block) {
  std::vector<Child<K>> list, keep;
  {
    std::lock_guard<std::mutex> g(dl.mu);
    for (Child<K>& c : dl.queue) {
      const bool mine = block == ~uint64_t{0} || (c.order >> 24) == block;
      (mine && !c.tail ? list : keep).push_back(std::move(c));
    }
    dl.queue = std::move(keep);
  }
  std::stable_sort(list.begin(), list.end(), [](const Child<K>& a, const Child<K>& b) { return a.order < b.order; });
  for (Child<K>& c : list) run_one(c, dl, mem, run);
}

// Queues a launch of block `block`, as cudaLaunchDevice does: false (and
// *err set) when the pending count is full even after earlier launches have
// run. Memory operations are not grids and are not counted.
template <class K, class Run>
bool enqueue(Launches<K>& dl, MemoryManager& mem, Run& run, Child<K> c, uint64_t block, int64_t limit, int* err) {
  const bool grid = !c.op;
  if (grid) {
    const int64_t cap = std::max<int64_t>(limit, kMinPendingLaunches);
    if (dl.tree->pending.load(std::memory_order_relaxed) >= cap)
      drain(dl, mem, run, ~uint64_t{0});
    if (dl.tree->pending.load(std::memory_order_relaxed) >= cap) {
      *err = kLaunchPendingCountExceeded;
      return false;
    }
    dl.tree->pending.fetch_add(1, std::memory_order_relaxed);
  }
  std::lock_guard<std::mutex> g(dl.mu);
  c.order = (block << 24) | (dl.per_block[block]++ & 0xFFFFFF);
  dl.queue.push_back(std::move(c));
  *err = kSuccess;
  return true;
}

}  // namespace vgpu::exec::devrt
