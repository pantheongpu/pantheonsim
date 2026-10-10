// The SIMT warp interpreter — VirtualGPU's CPU execution engine.
//
// Execution model (see ARCHITECTURE.md):
//  - A warp is 32 lanes advancing in lockstep over vector registers
//    (one 32-bit lane mask, one value per lane per register). We do NOT spawn
//    a CPU thread per GPU thread.
//  - Divergence: a branch that splits the active mask parks the not-taken
//    (pc, mask) on a per-warp divergence stack and continues with the taken
//    side; a path that retires pops the next parked path. Paths reconverge
//    implicitly at ret. Since then reconvergence is min-PC (paths at the same
//    pc merge, the lowest pc runs next), so a bar.sync after a divergent region
//    works; full IPDOM and irreducible control flow remain open (TODO.md,
//    "Partially implemented").
//  - Address spaces: device globals live in the MemoryManager VA range;
//    per-thread .local frames live in a reserved window (kLocalVaBase) that
//    generic loads/stores route to the executing lane's private buffer —
//    which is exactly PTX .local semantics. cvta is identity everywhere.
//  - Atomics are read-modify-write in fixed lane order — trivially atomic
//    and deterministic in this sequential engine.
//  - Warps in a block run under a pluggable Scheduler; a warp yields only at
//    barriers or retirement. Blocks run sequentially in a fixed order.
//    Everything is deterministic by construction.
#include <algorithm>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <thread>
#include <deque>
#include <memory>
#include <cfenv>
#include <climits>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <limits>
#include <map>
#include <set>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "vgpu/exec/host_atomic.hpp"
#include "vgpu/exec/device_printf.hpp"
#include "vgpu/exec/ldmatrix.hpp"
#include "vgpu/exec/numerics.hpp"
#include "vgpu/exec/wgmma.hpp"
#include "vgpu/exec/tensormap.hpp"
#include "vgpu/exec/tma.hpp"
#include "vgpu/exec/tcgen05.hpp"
#include "vgpu/exec/texture_grad.hpp"
#include "vgpu/error.hpp"
#include "vgpu/faults.hpp"
#include "vgpu/host_cpus.hpp"
#include "vgpu/exec/devrt.hpp"
#include "vgpu/exec/launch.hpp"

namespace vgpu::exec {
namespace {

using namespace vgpu::ptx;

// The widest warp any profile describes: 32 lanes on NVIDIA, 64 on a CDNA
// wavefront. This is a *ceiling* for sizing per-lane storage, not the width
// anything executes at -- that comes from the device profile at construction
// and lives in Interpreter::W_. Sizing to the maximum costs an NVIDIA warp
// twice the register-file footprint it needs and keeps every per-lane array a
// plain fixed-size member, which is the trade the alternative (a vector per
// register, indexed at every access) loses badly.
constexpr uint32_t kMaxWarpSize = 64;
// Per-thread .local window: distinct from both host pointers and device
// globals. Addresses here are lane-relative (each lane sees its own frame).
constexpr uint64_t kLocalVaBase = 0x6fff'0000'0000ull;
constexpr uint64_t kLocalVaSize = 1ull << 30;
// Where each thread's alloca stack sits in its local space: far above any
// function's static .local frame, cudaLimitStackSize bytes below this
// offset's top, growing down.
constexpr uint64_t kStackOffset = 0x1000'0000ull;
// Per-block .shared window, likewise distinct from host and device-global VAs.
// Kernel parameters get an address window of their own so a kernel can take a
// parameter's address and load through it -- CUB's segmented sort does exactly
// that. Nothing else may be addressed here, so a stray pointer into this range
// is still diagnosable.
constexpr uint64_t kParamVaBase = 0x6ffd'0000'0000ull;
constexpr uint64_t kParamVaSize = 1ull << 20;
constexpr uint64_t kSharedVaBase = 0x6ffe'0000'0000ull;
constexpr uint64_t kSharedVaSize = 1ull << 30;
// Distributed shared memory: the bits of a shared-window address from here up
// are the cluster rank of the block it is in, and the bits below the offset in
// that block's shared memory. A block's own addresses carry its own rank, so
// every .shared::cta address is already the .shared::cluster address of the
// same byte, as the ISA requires (the .shared::cta window is contained in the
// .shared::cluster one) -- and, as CUTLASS's 2-SM kernels rely on ("Set peer
// bit to 0 so that the transaction bytes will update CTA0's barrier", cute's
// copy_sm100_tma.hpp), clearing bit 24 of an odd block's address gives the
// same offset in its even peer. A block has at most 228 KiB of shared memory
// and a cluster at most 16 blocks, so both fit with room to spare.
constexpr uint32_t kClusterRankShift = 24;
constexpr uint64_t kClusterOffsetMask = (1ull << kClusterRankShift) - 1;

// Bit i == lane i active. 64 bits because a CDNA wavefront has 64 lanes; an
// NVIDIA warp uses the low 32 and leaves the rest clear.
using Mask = uint64_t;
// Operands are passed around as 64-bit lanes so every handler sees one type.
using Lanes = std::array<uint64_t, kMaxWarpSize>;
// Storage, however, is split by declared width: a 32-bit register costs
// 128 bytes per warp instead of 256. Most registers in real kernels are
// 32-bit, so this halves register-file traffic for the common case.
using Lanes32 = std::array<uint32_t, kMaxWarpSize>;

struct ParamBuffer {
  std::vector<uint8_t> bytes;
  std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> layout;  // name -> (offset, size)
};

// State for bar.red, which needs every warp in the block to arrive before it
// can produce a value. Held behind a pointer for the same reason shared memory
// is: the context is passed by const reference.
// bar.red in rounds. Votes go into the round being gathered; when the block
// releases, that round's answer is set aside and the next round starts empty,
// and each warp collects the answer of the round it voted in. A warp released
// first can loop back and vote again before the others have collected --
// CUTLASS's semaphore wait is `while (__syncthreads_and(state != k))` -- and
// with one shared accumulator that vote leaked into the answer they were
// about to read, and the round after started from a stale value.
struct BarrierReduction {
  uint64_t acc = 0;        // the round being gathered
  uint32_t arrived = 0;    // warps that have voted in it
  uint64_t round = 0;      // its number
  uint64_t result = 0;     // the answer of the last round released
};

// One mbarrier object. Lives in a side table keyed by its shared-memory
// address rather than in the shared bytes themselves: PTX says the contents
// are opaque, no kernel may read them as data, and keeping the real state
// outside means a kernel that does read them cannot accidentally appear to
// work.
// Bytes a TMA (cp.async.bulk) load has read from global memory and will write
// into shared memory, as runs of (shared offset, length) over `data`.
struct BlockCtx;
struct PendingBulk {
  std::vector<std::pair<uint32_t, uint32_t>> runs;
  std::vector<uint8_t> data;
  uint64_t tx = 0;         // bytes it completes on its barrier
  // The block whose shared memory the data goes to, when that is not the
  // barrier's: a .cta_group::2 copy may complete on the peer CTA's barrier.
  const BlockCtx* dst = nullptr;
};

struct Mbarrier {
  uint64_t expected = 0;   // arrivals per phase, from mbarrier.init
  uint64_t arrived = 0;    // arrivals so far in the current phase
  uint32_t phase = 0;      // flips each time the count is met
  bool valid = false;      // false before init and after inval
  // Transactions (bytes) the current phase still waits for: raised by
  // expect-tx, lowered by complete-tx. A phase completes only when the
  // arrivals are in *and* this is zero; it may dip below zero in between,
  // when a copy completes before its expect-tx is issued.
  int64_t tx = 0;
  // TMA loads that signal this barrier and have not landed yet. Their data
  // is written to shared memory, and their bytes completed, when a thread
  // next looks at the barrier -- a moment the asynchronous model allows, and
  // the latest one that cannot change a correct program's result. A kernel
  // that reads the tile without waiting sees the old contents, as it could
  // on hardware, instead of data a synchronous copy would have put there.
  std::vector<PendingBulk> pending;
};

// Every mbarrier a block has initialized, by shared address.
struct MbarrierTable {
  std::unordered_map<uint64_t, Mbarrier> bars;
};

// wgmma is one operation of the whole warpgroup, and it reads its shared
// memory operands as one: the first of the four warps to issue its n-th
// wgmma.mma_async reads A (all 64 rows) and B for all of them, and the
// others take their share of that copy. The interpreter runs the warps one
// after another, and when each read its own rows as it got there, a warp that
// finished early could wait_group, release the stage and let the cluster peer
// that multicasts A refill it before the last warp had read: CUTLASS's SM90
// group GEMM got one warp's 16 rows of a tile from the next K block. No warp
// can release a stage before some warp has issued the wgmma reading it, so
// the first issue is a safe moment to read.
struct WgmmaSnapshot {
  std::vector<double> A;   // 64 x K when A is in shared memory, else empty
  std::vector<double> B;   // K x N
  uint8_t taken = 0;       // warp ranks within the group that have used it
};
struct WgmmaSnapshots {
  std::map<std::pair<uint32_t, uint64_t>, WgmmaSnapshot> pending;   // (warpgroup, sequence)
};

// Shadow state for shared memory, one entry per 4-byte word, used only when
// race detection is on.
//
// The rule a CUDA block promises is simple: two warps may touch the same shared
// word without a barrier between them only if both are reading. Anything else
// is a race, and the answer depends on an order the program never specified.
// Hardware usually hides that -- warps advance together and the window is
// small -- which is exactly why it is worth checking here.
//
// An epoch is the count of barriers the block has completed, so "no barrier
// between them" is "same epoch". Warps per block cap at 32 (1024 threads), so
// the set of warps that read a word in an epoch fits in a uint32_t exactly.
struct WordShadow {
  uint32_t readers = 0;             // bitmask of warps that read it this epoch
  uint32_t read_epoch = 0xFFFFFFFF;
  uint32_t write_epoch = 0xFFFFFFFF;
  uint16_t writer = 0xFFFF;         // warp that wrote it, 0xFFFF for none
};

struct SharedShadow {
  std::vector<WordShadow> words;
  uint32_t epoch = 0;
};

struct ClusterState;
struct Warp;

// Dynamic parallelism: the child grids a launch's kernels ask for. A thread
// gets a parameter buffer for a kernel, fills it, and launches it; the child
// runs after the parent grid has finished and before the launch as a whole
// returns -- a schedule CUDA allows for every device-side stream, since it
// promises no concurrency between a parent and its children -- except a tail
// launch, which waits for the grid's other children too. Children run in the
// order they were launched, parent block by parent block, which makes the
// order reproducible when blocks run on several host threads. The shared
// half, with what each call does and returns, is vgpu/exec/devrt.hpp.
using ChildLaunch = devrt::Child<KernelRef>;
using DeviceLaunches = devrt::Launches<KernelRef>;

LaunchStats launch_grid(const EntryFn& fn, const LaunchConfig& cfg, const std::vector<std::vector<uint8_t>>& args,
                        MemoryManager& mem, const DeviceProfile& profile, const SymbolTable* symbols,
                        const ProgressFn& progress, DeviceLaunches* dl);

// A kernel's parameter space: each parameter at its alignment, in order --
// the layout the parent writes a child's arguments in.
uint32_t param_space_bytes(const EntryFn& fn);

// Runs one child grid: what devrt::run_children calls for each. `base` is
// the launch of the grid whose children these are (a child differs from it in
// its shape); `sink` takes the counters.
struct ChildRunner {
  const LaunchConfig& base;
  MemoryManager& mem;
  const DeviceProfile& profile;
  const ProgressFn& progress;
  LaunchStats* sink;
  void add_stats(const LaunchStats& s) { if (sink) sink->add(s); }
  void operator()(ChildLaunch& c, DeviceLaunches& out) {
    // The parameters back into one argument per parameter, at the offsets the
    // parent wrote them to.
    std::vector<std::vector<uint8_t>> args;
    uint32_t off = 0;
    for (const auto& p : c.kernel.fn->params) {
      const uint32_t align = p.align ? p.align : (p.size < 8 ? std::max<uint32_t>(p.size, 1) : 8);
      off = (off + align - 1) / align * align;
      if (off + p.size > c.params.size()) c.params.resize(off + p.size, 0);
      args.emplace_back(c.params.begin() + off, c.params.begin() + off + p.size);
      off += p.size;
    }
    LaunchConfig cc = base;
    cc.grid = c.grid;
    cc.block = c.block;
    cc.shared_bytes = c.shared;
    cc.cluster = {0, 0, 0};
    cc.cooperative = false;
    cc.coop_workspace = 0;
    add_stats(launch_grid(*c.kernel.fn, cc, args, mem, profile, c.kernel.symbols, progress, &out));
  }
};

// A kernel's parameter space: each parameter at its alignment, in order --
// the layout the parent writes a child's arguments in.
uint32_t param_space_bytes(const EntryFn& fn) {
  uint32_t end = 0;
  for (const auto& p : fn.params) {
    const uint32_t align = p.align ? p.align : (p.size < 8 ? std::max<uint32_t>(p.size, 1) : 8);
    end = (end + align - 1) / align * align + p.size;
  }
  return end;
}

// The CTA's sixteen barriers as bar.sync with a thread count and bar.arrive
// use them: arrivals counted in threads, a warp's arrival counting all of
// its threads (the ISA "marks warps' arrival"), and the warps waiting.
// A barrier without a count is the whole CTA, which the scheduler releases
// when every warp that has not exited is waiting (see step_block).
struct NamedBarrier {
  uint32_t arrived = 0;
  uint64_t waiting = 0;   // bit per warp of the block
  // A bar.red with a thread count gathers its answer here; the warps it
  // releases each carry the answer away (Warp::bar_red_result), since the
  // barrier is reusable the moment it completes.
  uint64_t acc = 0;
};
struct NamedBarriers {
  std::array<NamedBarrier, 16> bar{};
};

struct BlockCtx {
  std::array<uint32_t, 3> ctaid{};
  std::array<uint32_t, 3> ntid{};
  std::array<uint32_t, 3> nctaid{};
  // The cluster shape in CTAs, always at least 1x1x1: a launch with no
  // explicit cluster is a launch whose clusters hold one block each, which is
  // exactly what PTX says those registers report. Keeping the default at 1
  // rather than 0 means the cluster registers need no special case.
  std::array<uint32_t, 3> cluster{1, 1, 1};
  bool explicit_cluster = false;
  // Per-block shared memory. Zero-initialized at block start: real hardware
  // leaves it undefined, VirtualGPU makes it deterministic (documented).
  std::vector<uint8_t>* shared = nullptr;
  BarrierReduction* bar_red = nullptr;
  NamedBarriers* bars = nullptr;
  std::vector<Warp>* warps = nullptr;   // the block's, for releasing a named barrier's waiters
  MbarrierTable* mbar = nullptr;
  WgmmaSnapshots* wgmma = nullptr;
  TensorMemory* tmem = nullptr;
  SharedShadow* shadow = nullptr;   // non-null only when race detection is on
  // The thread-block cluster this block belongs to, for barrier.cluster. A
  // launch without clusters still has one per block.
  ClusterState* cluster_state = nullptr;
  // True once every thread of the block has exited, when its shared memory
  // is gone and another block of the cluster may no longer reach it.
  const bool* exited = nullptr;
  // Backs %clock/%clock64/%globaltimer. Advanced once per warp instruction,
  // per block -- see the note at sreg_value() for why this is a counter and
  // not a time.
  uint64_t* clock = nullptr;
};

// One diverged execution path: a set of lanes sharing a program counter.
struct Path {
  size_t pc = 0;
  Mask mask = 0;
  // Not runnable for now. kAtBarrier: waiting at a bar.sync for the warp's
  // other lanes, until another path reaches the same pc and merges with it.
  // kThisTurn: waiting on something outside the warp (barrier.cluster), so
  // the warp's other paths get to run; it is looked at again next turn.
  enum : uint8_t { kRunnable = 0, kAtBarrier = 1, kThisTurn = 2 };
  uint8_t parked = kRunnable;
  // The warp's step count when this path last issued, and how many more
  // instructions it is owed after waiting too long (see select_path).
  uint64_t issued_at = 0;
  uint32_t boost = 0;
};

// One cp.async copy that has been issued but not yet awaited.
//
// The source is read when the instruction issues and the destination is
// written when the thread waits. Both are points the hardware is allowed to
// pick, and splitting them this way is what makes the instruction mean
// anything: a kernel that reads the destination before its wait sees the old
// contents, exactly as it would on a device, instead of data that a
// synchronous copy would have put there early and hidden the bug.
struct PendingCopy {
  uint64_t dst = 0;                  // shared-window address
  uint32_t bytes = 0;                // 4, 8 or 16
  std::array<uint8_t, 16> data{};    // read at issue; zero past src-size
};

// Per-lane copy state, allocated only for warps that actually use cp.async --
// which is almost none of them, and this is 32 lanes of container otherwise.
struct AsyncCopies {
  // Issued but not yet committed to a group.
  std::array<std::vector<PendingCopy>, kMaxWarpSize> open;
  // Committed groups, oldest first. wait_group N drains until N remain.
  std::array<std::deque<std::vector<PendingCopy>>, kMaxWarpSize> groups;
};

// An allocator whose resize() leaves new elements uninitialized, for the
// register files: a register is never read before the write that sets its
// `written` flag (reads of unwritten ones answer zero), so zero-filling every
// declared register of every warp at every block start -- nvcc declares
// hundreds -- was pure cost: 29% of mma_virus's time.
template <class T>
struct UninitAlloc : std::allocator<T> {
  template <class U> struct rebind { using other = UninitAlloc<U>; };
  UninitAlloc() = default;
  template <class U> UninitAlloc(const UninitAlloc<U>&) noexcept {}
  template <class U> void construct(U* p) noexcept { ::new (static_cast<void*>(p)) U; }
  template <class U, class... A> void construct(U* p, A&&... a) {
    ::new (static_cast<void*>(p)) U(std::forward<A>(a)...);
  }
};
template <class T> using RegFile = std::vector<T, UninitAlloc<T>>;

struct Warp {
  enum class State { Ready, AtBarrier, Done };
  State state = State::Ready;
  // Set between contributing to a bar.red and collecting its result. The
  // instruction re-executes when the barrier releases, and this is how it knows
  // to collect rather than contribute a second time.
  bool bar_red_waiting = false;
  uint64_t bar_red_round = 0;   // the bar.red round this warp voted in
  // A bar.red with a thread count: the answer its completion handed this warp.
  uint64_t bar_red_result = 0;
  // At a barrier with a thread count, which that barrier's completion
  // releases; step_block's all-warps release leaves such a warp waiting.
  bool counted_barrier = false;
  uint8_t barrier_id = 0;
  // Set by an mbarrier wait that came back incomplete. The warp stays Ready --
  // the wait is a predicate and the kernel is free to spin on it -- but it
  // gives up the rest of its turn so the warps it is waiting for can run. With
  // the deterministic scheduler a turn otherwise lasts until the warp blocks,
  // and a spin never blocks, so the first waiter would hold the block forever.
  bool yield_now = false;
  // Set by an instruction that must run again rather than advance: a
  // tcgen05.alloc that found too few columns free waits by retrying, giving
  // up its turn each time so the warp that will free them can run.
  bool retry = false;
  // barrier.cluster: the lanes that have arrived in the cluster's current
  // phase, and the phase a later wait is waiting to see end.
  Mask cluster_arrived = 0;
  uint32_t cluster_wait_phase = 0;
  // wgmma.mma_async instructions issued, which pairs this warp's n-th with
  // the rest of its warpgroup's n-th (see WgmmaSnapshots), and the count at
  // each wgmma.commit_group not yet waited for.
  uint64_t wgmma_issued = 0;
  std::vector<uint64_t> wgmma_commits;
  // Instructions this warp has issued, for the step budget. Counted per warp
  // rather than per launch: the budget exists to catch a thread that never
  // finishes, and a launch's total grows with its grid -- a 12 GB sweep over
  // 43,008 threads is some 10^11 instructions of legitimate work, while each
  // warp's share stays small. It also keeps the verdict independent of how
  // many host threads the grid happens to be spread over.
  uint64_t steps = 0;
  // Per thread, the last error a device-runtime call returned: the device's
  // cudaGetLastError reports it and clears it. The CUDA Programming Guide
  // records device-side errors per thread, and an RTX 3060 keeps them so: a
  // thread whose launch failed sees the error, its warp-mates do not.
  Lanes device_error{};
  // Live paths. Reconvergence is by *lowest program counter*: the path with
  // the smallest pc always runs next, and paths that arrive at the same pc are
  // merged. For the structured control flow compilers emit, that reconverges
  // an if/else at its join point and lets a loop's lanes iterate until they
  // reach the exit -- which is what makes bar.sync after a divergent region
  // work, since every lane has merged back into one path by then.
  std::vector<Path> paths;
  Mask exited = 0;
  // Register files indexed by the parser's dense ids: narrow registers live
  // in regs32, 64-bit ones in regs64. `written*` tracks first assignment so a
  // read-before-write is still diagnosed.
  RegFile<Lanes32> regs32;
  RegFile<Lanes> regs64;
  std::vector<Mask> preds;
  std::vector<uint8_t> written32, written64;
  // Widening a 32-bit register into the 64-bit operand form needs somewhere to
  // land; reused per read to avoid touching the allocator.
  mutable std::array<Lanes, 4> widen_scratch;
  mutable uint32_t widen_next = 0;
  // Call-argument slots. Byte-addressable rather than one value per lane,
  // because a .param slot can hold a struct: "st.param.b32 [param0+8], %r"
  // writes at an offset, and a slot that stored a single 64-bit value per lane
  // had nowhere to put the rest. Laid out lane-major: bytes[lane*size + off].
  struct Slot {
    uint32_t size = 8;
    std::vector<uint8_t> bytes;
    void reset(uint32_t sz, uint32_t lanes) {
      size = sz ? sz : 8;
      bytes.assign(static_cast<size_t>(size) * lanes, 0);
    }
    // At least `need` bytes a lane, keeping what every lane already holds.
    void fit(uint32_t need, uint32_t lanes) {
      if (!bytes.empty() && size >= need) return;
      Slot grown;
      grown.reset(bytes.empty() || size < need ? need : size, lanes);
      for (uint32_t lane = 0; lane < lanes && !bytes.empty(); ++lane)
        std::memcpy(grown.bytes.data() + static_cast<size_t>(lane) * grown.size,
                    bytes.data() + static_cast<size_t>(lane) * size, size);
      *this = std::move(grown);
    }
    uint64_t read(uint32_t lane, uint32_t off, uint32_t nbytes) const {
      uint64_t v = 0;
      const size_t base = static_cast<size_t>(lane) * size + off;
      if (base + nbytes > bytes.size()) return 0;
      std::memcpy(&v, bytes.data() + base, nbytes);
      return v;
    }
    void write(uint32_t lane, uint32_t off, uint32_t nbytes, uint64_t v) {
      const size_t base = static_cast<size_t>(lane) * size + off;
      if (base + nbytes > bytes.size()) return;
      std::memcpy(bytes.data() + base, &v, nbytes);
    }
  };
  std::unordered_map<std::string, Slot> slots;
  // The device functions this warp is inside, innermost last. Each frame is
  // what its caller was doing: the paths (the calling one already past the
  // call), registers and slots it resumes with when the callee's last path
  // returns. A call is a frame rather than a nested run of the interpreter so
  // that a warp can stop in the middle of a device function -- at a bar.sync,
  // a bar.red, an mbarrier wait -- and let the other warps run, which is what
  // __syncthreads() in a function nvcc did not inline needs (every -G build).
  struct Frame {
    const EntryFn* fn = nullptr;
    uint64_t local_base = 0;
    std::vector<Path> paths;
    RegFile<Lanes32> regs32;
    RegFile<Lanes> regs64;
    std::vector<Mask> preds;
    std::vector<uint8_t> written32, written64;
    std::unordered_map<std::string, Slot> slots;
    std::string retval_from, retval_to;   // callee's return slot -> caller's
    Mask lanes = 0;                        // the lanes that made the call
    // The caller's stack pointer: what the callee allocas is freed when it
    // returns (9.7.19.3).
    std::array<uint64_t, kMaxWarpSize> sp{};
    bool sp_valid = false;
  };
  std::vector<Frame> frames;
  // The function this warp is executing (null: the kernel) and where its
  // .local frame starts; the interpreter's cur_ and local_base_ follow the
  // warp it is running.
  const EntryFn* fn = nullptr;
  uint64_t local_base = 0;
  // Set by a call that pushed a frame, so the caller's pc is not advanced a
  // second time: the frame already holds it past the call.
  bool entered_call = false;
  std::vector<std::vector<uint8_t>> local;       // per-lane .local frames (lazy)
  // The alloca stack: each lane's bytes, and its stack pointer (a local-space
  // offset, from kStackOffset up to kStackOffset + the stack size); unset
  // until the warp first touches the stack.
  std::vector<std::vector<uint8_t>> stack;
  std::array<uint64_t, kMaxWarpSize> sp{};
  bool sp_valid = false;
  std::array<uint32_t, kMaxWarpSize> tid_x{}, tid_y{}, tid_z{};
  // PTX's condition-code carry bit, one per lane. Written by ".cc" arithmetic
  // and read by addc/subc/madc; nothing else in the ISA touches it.
  Mask carry = 0;
  std::unique_ptr<AsyncCopies> cp;  // created on the first cp.async
};

// A thread-block cluster's shared state: its blocks' warps, which
// barrier.cluster waits on, and that barrier's phase. The blocks of a cluster
// are resident together and interleaved, the way a cooperative grid's are,
// because a cluster barrier only completes if every block can reach it.
struct ClusterState {
  std::vector<std::vector<Warp>*> blocks;
  // Each block's context by %cluster_ctarank, for distributed shared memory.
  std::vector<const BlockCtx*> ranks;
  uint32_t phase = 0;
};

// %cluster_ctarank: x fastest within the cluster.
uint32_t cluster_rank_of(const BlockCtx& ctx) {
  return ctx.ctaid[0] % ctx.cluster[0] + ctx.ctaid[1] % ctx.cluster[1] * ctx.cluster[0] +
         ctx.ctaid[2] % ctx.cluster[2] * ctx.cluster[0] * ctx.cluster[1];
}

// Strict mode, sampled once per launch: consulting it is a relaxed atomic load
// rather than a getenv on a hot path, and a process that changes the variable
// between launches gets what it asked for.
std::atomic<bool> g_strict{false};
// VGPU_FASTPATH=0 sends every instruction down the general path, for checking
// the fast paths against it (tests/unit/test_fastpath.cpp) and for ruling them
// out when a result looks wrong.
std::atomic<bool> g_fast_path{true};
std::atomic<bool> g_race{false};
// VGPU_RACE=2: also report writes that leave the bytes unchanged. Off by
// default -- see the comment on `unobservable_write` for why those are not
// races -- but the strict definition has its uses, so it stays reachable.
std::atomic<bool> g_race_strict{false};

// Sampled together, once per launch. Caching either in a function-local static
// makes it depend on which kernel in a process ran first -- which is how the
// race detector's own test came to pass a racy kernel: an earlier launch had
// already frozen the flag to false.
void refresh_modes() {
  faults::refresh();
  const char* strict = std::getenv("VGPU_STRICT");
  g_strict.store(strict && strict[0] == '1', std::memory_order_relaxed);
  const char* race = std::getenv("VGPU_RACE");
  g_race.store(race && (race[0] == '1' || race[0] == '2'), std::memory_order_relaxed);
  g_race_strict.store(race && race[0] == '2', std::memory_order_relaxed);
  const char* fast = std::getenv("VGPU_FASTPATH");
  g_fast_path.store(!(fast && fast[0] == '0'), std::memory_order_relaxed);
}

// IEEE 754 binary16 <-> double, implemented in software so the engine needs no
// host f16 support. Round-to-nearest-even, with subnormals and inf/NaN.
double f16_to_double_exact(uint64_t bits) {
  // Assembled field by field rather than through ldexp: every binary16 value
  // is exact in a double, so sign, exponent and mantissa carry straight across,
  // and the libm call this replaced was most of an f16 matrix kernel's time.
  // Bit-identical to the ldexp form on all 65536 inputs.
  const uint32_t h = static_cast<uint32_t>(bits) & 0xFFFFu;
  const uint64_t sign = uint64_t{h >> 15} << 63;
  const uint32_t exp = (h >> 10) & 0x1F;
  const uint64_t mant = h & 0x3FF;
  uint64_t out;
  if (exp == 31) {
    out = sign | (mant ? 0x7FF8'0000'0000'0000ull : 0x7FF0'0000'0000'0000ull);
  } else if (exp != 0) {
    out = sign | (uint64_t{exp - 15 + 1023} << 52) | (mant << 42);
  } else if (mant == 0) {
    out = sign;
  } else {
    // Subnormal: mant * 2^-24, normalized so its leading one is implicit.
    const int top = 63 - std::countl_zero(mant);
    out = sign | (uint64_t(top - 24 + 1023) << 52) | ((mant << (52 - top)) & ((1ull << 52) - 1));
  }
  return std::bit_cast<double>(out);
}

// Every binary16 value's double, computed once by the function above: f16
// kernels convert every operand of every lane, and a load is cheaper than the
// field assembly. The same values by construction.
const std::array<double, 65536> kF16ToDouble = [] {
  std::array<double, 65536> t{};
  for (uint32_t h = 0; h < 65536; ++h) t[h] = f16_to_double_exact(h);
  return t;
}();

// The same values as floats, for the f16 WMMA fast path: half the table, and
// every binary16 value is exact in f32.
const std::array<float, 65536> kF16ToFloat = [] {
  std::array<float, 65536> t{};
  for (uint32_t h = 0; h < 65536; ++h) t[h] = static_cast<float>(kF16ToDouble[h]);
  return t;
}();

double f16_to_double(uint64_t bits) {
  if (!g_fast_path.load(std::memory_order_relaxed)) return f16_to_double_exact(bits);
  return kF16ToDouble[bits & 0xFFFFu];
}

// How many threads are inside a float instruction with an explicit
// .rz/.rm/.rp, which sets the host rounding mode around its arithmetic. While
// it is zero every thread is rounding to nearest, and conversions can round on
// the bits; otherwise they take the path that follows the host mode. A global
// count rather than a per-thread flag: a thread_local in this shared library
// cost a __tls_get_addr call per conversion, and reading MXCSR every time was
// not much cheaper.
std::atomic<int> g_directed_rounding{0};

[[gnu::noinline]] uint64_t double_to_narrow(double d, int man, int ebits);

// NaN in a 16-bit float: what a real GPU writes for every NaN result, whatever
// the input's payload -- the all-ones pattern below the sign, 0x7FFF for both
// f16 and bf16 (and 0x7FFFFFFF for f32, which PTX calls canonical).
inline constexpr uint64_t kCanonicalNaN16 = 0x7FFF;

// bfloat16 is the top 16 bits of an f32: same exponent, mantissa truncated to
// 7 bits. Converting in rounds to nearest even on the discarded half, which is
// what cvt.rn asks for; converting out is exact.
double bf16_to_double(uint64_t in) {
  const uint32_t bits = static_cast<uint32_t>(in & 0xffffu) << 16;
  return static_cast<double>(std::bit_cast<float>(bits));
}
uint64_t double_to_bf16(double x) {
  const float f = static_cast<float>(x);
  if (std::isnan(f)) return kCanonicalNaN16;
  // A directed mode, or a double that is not already a float (rounding it to
  // float first could round twice), takes the general form.
  if (g_directed_rounding.load(std::memory_order_relaxed) != 0 || static_cast<double>(f) != x)
    return double_to_narrow(x, 7, 8);
  const uint32_t bits = std::bit_cast<uint32_t>(f);
  // Round to nearest, ties to even, on the 16 bits being dropped.
  const uint32_t lsb = (bits >> 16) & 1u;
  const uint32_t rounded = bits + 0x7fffu + lsb;
  return rounded >> 16;
}


inline uint64_t double_to_f16(double d) {
  // A normal binary16 result under round-to-nearest-even, done on the bits:
  // keep the top 10 bits of the double's mantissa and round on the 42 below,
  // with NaN, infinity and overflow answered as the general form answers them.
  // That form is kept for what remains: subnormal results, and anything under
  // a directed mode, where overflow can stop at the largest finite value.
  if (g_directed_rounding.load(std::memory_order_relaxed) == 0 &&
      g_fast_path.load(std::memory_order_relaxed)) {
    const uint64_t b = std::bit_cast<uint64_t>(d);
    const uint64_t mag = b & 0x7FFF'FFFF'FFFF'FFFFull;
    if (mag > 0x7FF0'0000'0000'0000ull) return kCanonicalNaN16;
    if (mag >= 0x40EF'FE00'0000'0000ull)                             // >= 65520, or infinity
      return (static_cast<uint32_t>(b >> 48) & 0x8000u) | 0x7C00;
    const int e = static_cast<int>(mag >> 52) - 1023;   // unbiased exponent
    if (e >= -14 && e <= 15) {                            // a normal half, before rounding
      const uint32_t sign = static_cast<uint32_t>(b >> 48) & 0x8000u;
      const uint64_t m = mag & ((1ull << 52) - 1);
      uint64_t mant = m >> 42;
      const uint64_t rest = m & ((1ull << 42) - 1), half = 1ull << 41;
      if (rest > half || (rest == half && (mant & 1))) ++mant;
      uint32_t e16 = static_cast<uint32_t>(e + 15);
      if (mant == 1024) {   // rounding carried into the exponent
        mant = 0;
        ++e16;
      }
      if (e16 >= 31) return sign | 0x7C00;
      return sign | (e16 << 10) | static_cast<uint32_t>(mant);
    }
  }
  return double_to_narrow(d, 10, 5);
}

// d rounded to a binary float with `man` stored mantissa bits and `ebits`
// exponent bits (f16 is 10/5, bf16 7/8), in the host's current rounding mode,
// subnormals included. The rounding is one nearbyint of d scaled so the
// result's last place is 1, which is what makes every mode come out right:
// scaling the magnitude instead rounded -x toward zero under .rm. A value
// between the smallest subnormal and half of it rounds up to that subnormal;
// flushing everything below the smallest one to zero lost silu(-20) in f16.
// Overflow gives infinity only where the mode rounds away from zero; .rz, and
// .rm/.rp against the sign, stop at the largest finite value.
[[gnu::noinline]] uint64_t double_to_narrow(double d, int man, int ebits) {
  const int bias = (1 << (ebits - 1)) - 1;
  const uint32_t sign = std::signbit(d) ? 1u << (man + ebits) : 0u;
  const uint32_t inf = ((1u << ebits) - 1) << man;
  if (std::isnan(d)) return kCanonicalNaN16;
  if (d == 0) return sign;
  if (std::isinf(d)) return sign | inf;
  int exp;
  std::frexp(d, &exp);                          // |d| in [2^(exp-1), 2^exp)
  int e = std::max(exp - 1, 1 - bias);          // the exponent the result is scaled by
  uint64_t mant = static_cast<uint64_t>(std::fabs(std::nearbyint(std::ldexp(d, man - e))));
  if (mant >> (man + 1)) {                      // rounding carried into the next binade
    mant >>= 1;
    ++e;
  }
  if (e > bias) {
    const int mode = std::fegetround();
    const bool to_inf = mode == FE_TONEAREST || (mode == FE_UPWARD && !sign) ||
                        (mode == FE_DOWNWARD && sign);
    return sign | (to_inf ? inf : inf - 1);
  }
  if (mant < (1ull << man)) return sign | static_cast<uint32_t>(mant);  // subnormal
  return sign | (static_cast<uint32_t>(e + bias) << man) |
         static_cast<uint32_t>(mant - (1ull << man));
}

// ---- FP8 -------------------------------------------------------------
//
// Two formats, and they are not the same shape with a different bias.
//
//   e4m3: 4 exponent bits (bias 7), 3 mantissa bits. NO infinity -- the
//         all-ones exponent is a normal range, and only S.1111.111 is NaN, so
//         the largest finite value is 448.
//   e5m2: 5 exponent bits (bias 15), 2 mantissa bits, and an IEEE-shaped top
//         exponent, so it does have infinity and its max finite is 57344.
//
// Getting e4m3's missing infinity wrong is the interesting failure: treating
// its top exponent as reserved costs half the representable range, and the
// values that vanish are the large activations FP8 inference is scaled to
// keep. Reading 0x7F as an ordinary number instead of NaN is the same mistake
// pointing the other way.
struct Fp8Format {
  int exp_bits, man_bits, bias;
  bool has_inf;
  double max_finite;
};
inline constexpr Fp8Format kE4M3{4, 3, 7, false, 448.0};
inline constexpr Fp8Format kE5M2{5, 2, 15, true, 57344.0};

double fp8_to_double(uint32_t byte, const Fp8Format& f) {
  const uint32_t man_mask = (1u << f.man_bits) - 1u;
  const uint32_t exp_mask = (1u << f.exp_bits) - 1u;
  const bool sign = (byte >> (f.exp_bits + f.man_bits)) & 1u;
  const uint32_t exp = (byte >> f.man_bits) & exp_mask;
  const uint32_t man = byte & man_mask;
  double v;
  if (exp == exp_mask) {
    if (f.has_inf) {
      v = man ? std::numeric_limits<double>::quiet_NaN()
              : std::numeric_limits<double>::infinity();
    } else {
      // e4m3 spends this exponent on ordinary numbers; only all-ones mantissa
      // is NaN.
      v = man == man_mask
              ? std::numeric_limits<double>::quiet_NaN()
              : std::ldexp(1.0 + static_cast<double>(man) / (man_mask + 1),
                           static_cast<int>(exp) - f.bias);
    }
  } else if (exp == 0) {
    v = std::ldexp(static_cast<double>(man) / (man_mask + 1), 1 - f.bias);
  } else {
    v = std::ldexp(1.0 + static_cast<double>(man) / (man_mask + 1),
                   static_cast<int>(exp) - f.bias);
  }
  return sign ? -v : v;
}

// satfinite clamps to the largest finite value instead of producing infinity
// or NaN, which is what every FP8 conversion nvcc emits actually asks for.
uint32_t double_to_fp8(double d, const Fp8Format& f, bool satfinite) {
  const uint32_t man_mask = (1u << f.man_bits) - 1u;
  const uint32_t exp_mask = (1u << f.exp_bits) - 1u;
  const uint32_t sign_bit = 1u << (f.exp_bits + f.man_bits);
  // Measured on an L4 (sm_89): a NaN converts to the all-ones code in both formats (0x7F for E5M2 too).
  const uint32_t nan_bits = (exp_mask << f.man_bits) | man_mask;
  if (std::isnan(d)) return nan_bits;
  const uint32_t sign = std::signbit(d) ? sign_bit : 0u;
  double a = std::fabs(d);
  const uint32_t max_bits =
      f.has_inf ? (((exp_mask - 1u) << f.man_bits) | man_mask)
                : ((exp_mask << f.man_bits) | (man_mask - 1u));
  if (std::isinf(a) || a > f.max_finite) {
    if (satfinite) return sign | max_bits;
    if (f.has_inf) return sign | (exp_mask << f.man_bits);
    return nan_bits;  // e4m3 has no infinity to overflow into
  }
  const int min_sub = 1 - f.bias - f.man_bits;   // exponent of the smallest subnormal
  if (a < std::ldexp(1.0, min_sub - 1)) return sign;  // rounds to zero
  int exp = 0;
  const double frac = std::frexp(a, &exp);       // a = frac * 2^exp, frac in [0.5, 1)
  int e = exp - 1 + f.bias;
  if (e <= 0) {                                   // subnormal
    const uint32_t man =
        static_cast<uint32_t>(std::nearbyint(std::ldexp(a, f.man_bits + f.bias - 1)));
    // Rounding a subnormal up can carry it into the smallest normal, which is
    // exactly representable and must not be truncated back down.
    return sign | (man & ((man_mask << 1) | 1u));
  }
  uint32_t man = static_cast<uint32_t>(
      std::nearbyint((frac * 2.0 - 1.0) * static_cast<double>(man_mask + 1)));
  if (man == man_mask + 1) {  // rounding carried into the exponent
    man = 0;
    ++e;
  }
  if (static_cast<uint32_t>(e) > exp_mask ||
      (f.has_inf && static_cast<uint32_t>(e) >= exp_mask) ||
      (!f.has_inf && static_cast<uint32_t>(e) == exp_mask && man == man_mask)) {
    if (satfinite) return sign | max_bits;
    return f.has_inf ? (sign | (exp_mask << f.man_bits)) : nan_bits;
  }
  return sign | (static_cast<uint32_t>(e) << f.man_bits) | (man & man_mask);
}

// The OCP MX small floats e2m3, e3m2 and e2m1 (PTX ISA 5.2.3): no infinity
// and no NaN, so cvt's mandatory .satfinite sends a magnitude past the largest
// normal, and infinity, to that normal with the sign kept. Rounds to nearest
// even; the caller handles NaN.
double small_float_value(uint32_t code, int eb, int mb, int bias) {
  const uint32_t mant = code & ((1u << mb) - 1), exp = (code >> mb) & ((1u << eb) - 1);
  const double mag = exp ? std::ldexp(1.0 + mant / double(1u << mb), int(exp) - bias)
                         : std::ldexp(mant / double(1u << mb), 1 - bias);
  return (code >> (eb + mb)) & 1 ? -mag : mag;
}
// `v` rounded toward zero onto the grid of a float format with `mb` mantissa bits and
// the given exponent bias (subnormals included), for cvt's .rz; NaN and infinity pass.
// Overflow is left to the encoder (it saturates).
double trunc_to_grid(double v, int mb, int bias) {
  if (!std::isfinite(v) || v == 0.0) return v;
  const double a = std::fabs(v);
  int e = 0;
  std::frexp(a, &e);
  const int E = std::max(e - 1, 1 - bias);
  const double t = std::ldexp(std::floor(std::ldexp(a, mb - E)), E - mb);
  return std::signbit(v) ? -t : t;
}
uint32_t double_to_small_float(double v, int eb, int mb, int bias) {
  const uint32_t sign = std::signbit(v) ? 1u << (eb + mb) : 0u;
  const uint32_t max_code = (((1u << eb) - 1) << mb) | ((1u << mb) - 1);
  const double a = std::fabs(v);
  if (a >= small_float_value(max_code, eb, mb, bias)) return sign | max_code;
  int e = 0;
  std::frexp(a, &e);                                      // a = f * 2^e, f in [0.5, 1)
  int E = a == 0.0 ? 1 - bias : std::max(e - 1, 1 - bias);   // unbiased, subnormals at 1 - bias
  uint32_t q = static_cast<uint32_t>(std::nearbyint(std::ldexp(a, mb - E)));
  if (q == 2u << mb) {   // rounding carried into the next binade
    q = 1u << mb;
    ++E;
  }
  const uint32_t code = q < (1u << mb) ? q : (static_cast<uint32_t>(E + bias) << mb) | (q - (1u << mb));
  return sign | std::min(code, max_code);
}

// tf32 is f32's sign and exponent with the mantissa cut to 10 bits. Round to
// nearest even rather than truncating: truncation biases every product toward
// zero, which accumulates over a reduction into a visible error.
float f32_to_tf32(float x) {
  uint32_t b = std::bit_cast<uint32_t>(x);
  if ((b & 0x7F800000u) == 0x7F800000u) return x;  // inf/NaN keep their payload
  const uint32_t lsb = (b >> 13) & 1u;
  b += 0x0FFFu + lsb;
  b &= ~0x1FFFu;
  return std::bit_cast<float>(b);
}

float f32(uint64_t bits) { return std::bit_cast<float>(static_cast<uint32_t>(bits)); }
uint64_t f32bits(float f) { return std::bit_cast<uint32_t>(f); }
double f64(uint64_t bits) { return std::bit_cast<double>(bits); }
uint64_t f64bits(double d) { return std::bit_cast<uint64_t>(d); }

// Every lane of a warp `width` lanes wide. Written as a shift of 2 rather than
// 1 so that width == 64 does not shift a 64-bit value by 64, which is
// undefined and on x86 produces 1 rather than 0.
inline constexpr Mask all_lanes(uint32_t width) {
  return static_cast<Mask>((Mask{2} << (width - 1)) - 1);
}

// Applies `f` to each active lane. The full-warp case is a straight loop the
// compiler can vectorize; a partial mask walks only the set bits instead of
// testing every lane. Both matter: this runs once per instruction per warp.
template <class F>
inline void for_active(Mask m, uint32_t width, F&& f) {
  if (m == all_lanes(width)) {
    for (uint32_t lane = 0; lane < width; ++lane) f(lane);
  } else {
    Mask rest = m;
    while (rest) {
      uint32_t lane = static_cast<uint32_t>(__builtin_ctzll(rest));
      rest &= rest - 1;
      f(lane);
    }
  }
}

// Lane loops for fma on the host's FMA unit. The build targets baseline
// x86-64, so std::fma is a call into libm per lane -- which glibc then routes
// to the same vfmadd instruction through an ifunc on any CPU that has one.
// Calling it once per instruction instead of once per lane gives the same
// bits (both are the correctly rounded fused multiply-add) without the call.
#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("fma"))) void fma_lanes_f32_hw(uint32_t* d, const uint32_t* a, const uint32_t* b,
                                                     const uint32_t* c, Mask m, uint32_t width) {
  for_active(m, width, [&](uint32_t l) {
    float x, y, z;
    std::memcpy(&x, &a[l], 4);
    std::memcpy(&y, &b[l], 4);
    std::memcpy(&z, &c[l], 4);
    const float r = __builtin_fmaf(x, y, z);
    std::memcpy(&d[l], &r, 4);
  });
}
__attribute__((target("fma"))) void fma_lanes_f64_hw(uint64_t* d, const uint64_t* a, const uint64_t* b,
                                                     const uint64_t* c, Mask m, uint32_t width) {
  for_active(m, width, [&](uint32_t l) {
    double x, y, z;
    std::memcpy(&x, &a[l], 8);
    std::memcpy(&y, &b[l], 8);
    std::memcpy(&z, &c[l], 8);
    const double r = __builtin_fma(x, y, z);
    std::memcpy(&d[l], &r, 8);
  });
}
// fma.f16x2 lanes on the FMA unit, with the decode and the rounding inlined:
// the same double fma and double_to_f16 as the general path, without two
// calls per half.
__attribute__((target("fma"))) void f16x2_fma_lanes_hw(uint32_t* d, const uint32_t* a, const uint32_t* b,
                                                      const uint32_t* c, Mask m, uint32_t width, int halves) {
  for_active(m, width, [&](uint32_t l) {
    uint64_t out = 0;
    for (int h = 0; h < halves; ++h) {
      const double v = __builtin_fma(kF16ToDouble[(a[l] >> (16 * h)) & 0xFFFF],
                                     kF16ToDouble[(b[l] >> (16 * h)) & 0xFFFF],
                                     kF16ToDouble[(c[l] >> (16 * h)) & 0xFFFF]);
      out |= double_to_f16(v) << (16 * h);
    }
    d[l] = static_cast<uint32_t>(out);
  });
}
const bool g_hw_fma = __builtin_cpu_supports("fma");
__attribute__((target("fma"))) double fma_hw(double a, double b, double c) { return __builtin_fma(a, b, c); }
#else
constexpr bool g_hw_fma = false;
double fma_hw(double a, double b, double c) { return std::fma(a, b, c); }
#endif
// The correctly rounded fused multiply-add, without libm's per-call dispatch.
inline double host_fma(double a, double b, double c) { return g_hw_fma ? fma_hw(a, b, c) : std::fma(a, b, c); }

// Per-element lanes for a vector load, store or register pack: up to four on
// the stack, as the parser allows, instead of a heap allocation on every
// memory instruction. More than four still works, from the heap.
struct LaneSet {
  explicit LaneSet(size_t n) : n_(n) {
    if (n > inline_.size()) heap_.resize(n);
  }
  Lanes& operator[](size_t i) { return n_ > inline_.size() ? heap_[i] : inline_[i]; }
  size_t size() const { return n_; }

 private:
  size_t n_;
  std::array<Lanes, 4> inline_;
  std::vector<Lanes> heap_;
};

// The position of an instruction kind in the Op variant, for switching on
// ins.op.index() instead of testing the alternatives one at a time.
template <class T>
constexpr size_t op_index() {
  return Op(std::in_place_type<T>).index();
}

// D = A x B + C for a 16x16 tile, K deep, in f32: each element accumulates
// its products in k order, one rounding for the product and one for the sum,
// as exec_wmma_mma always has. A row is two 8-float vectors, built for AVX2 as
// well as the baseline, and the CPU is asked which to use -- the way
// g_hw_fma chooses. Not target_clones: that picks through an IFUNC resolver,
// which the loader runs before main, and under ThreadSanitizer the resolver
// runs instrumented before the sanitizer's runtime is up, so every program
// linking the interpreter crashed at startup without a word. Contraction is
// off here so that no build -- -march=native included -- fuses the multiply
// and add into an FMA and changes the bits.
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
typedef float WmmaRow8 __attribute__((vector_size(32)));
#define VGPU_WMMA_TILE_BODY                                          \
  for (uint32_t i = 0; i < 16; ++i) {                                \
    WmmaRow8 lo, hi;                                                 \
    std::memcpy(&lo, &C[i][0], sizeof lo);                           \
    std::memcpy(&hi, &C[i][8], sizeof hi);                           \
    for (uint32_t k = 0; k < K; ++k) {                               \
      const float aik = A[i][k];                                     \
      const WmmaRow8 a = {aik, aik, aik, aik, aik, aik, aik, aik};   \
      WmmaRow8 bl, bh;                                               \
      std::memcpy(&bl, &B[k][0], sizeof bl);                         \
      std::memcpy(&bh, &B[k][8], sizeof bh);                         \
      const WmmaRow8 pl = a * bl, ph = a * bh;                       \
      lo = lo + pl;                                                  \
      hi = hi + ph;                                                  \
    }                                                                \
    std::memcpy(&D[i][0], &lo, sizeof lo);                           \
    std::memcpy(&D[i][8], &hi, sizeof hi);                           \
  }
void wmma_tile_base(float (&D)[16][16], const float (&A)[16][16], const float (&B)[16][16],
                    const float (&C)[16][16], uint32_t K) {
  VGPU_WMMA_TILE_BODY
}
#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx2"))) void wmma_tile_avx2(float (&D)[16][16], const float (&A)[16][16],
                                                    const float (&B)[16][16],
                                                    const float (&C)[16][16], uint32_t K) {
  VGPU_WMMA_TILE_BODY
}
const bool g_avx2 = __builtin_cpu_supports("avx2");
#endif
#undef VGPU_WMMA_TILE_BODY
void wmma_tile(float (&D)[16][16], const float (&A)[16][16], const float (&B)[16][16],
               const float (&C)[16][16], uint32_t K) {
#if defined(__x86_64__) && defined(__GNUC__)
  if (g_avx2) return wmma_tile_avx2(D, A, B, C, K);
#endif
  wmma_tile_base(D, A, B, C, K);
}
#pragma GCC pop_options

// Sixteen binary16 values -- eight 32-bit words, low half first -- as floats:
// one f16 WMMA fragment row. F16C converts eight at a time; it is used only
// if it gives the table's bits for all 65536 inputs, NaNs included, which is
// checked once rather than assumed.
void f16x16_to_float_table(const uint32_t* w8, float* out) {
  for (int i = 0; i < 8; ++i) {
    out[2 * i] = kF16ToFloat[w8[i] & 0xFFFF];
    out[2 * i + 1] = kF16ToFloat[w8[i] >> 16];
  }
}
#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx,f16c"))) void f16x16_to_float_f16c(const uint32_t* w8, float* out) {
  const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w8));
  const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w8 + 4));
  _mm256_storeu_ps(out, _mm256_cvtph_ps(lo));
  _mm256_storeu_ps(out + 8, _mm256_cvtph_ps(hi));
}
const bool g_f16c = [] {
  if (!__builtin_cpu_supports("avx") || !__builtin_cpu_supports("f16c")) return false;
  for (uint32_t h = 0; h < 65536; h += 16) {
    uint32_t w8[8];
    for (uint32_t i = 0; i < 8; ++i) w8[i] = (h + 2 * i) | ((h + 2 * i + 1) << 16);
    float got[16];
    f16x16_to_float_f16c(w8, got);
    if (std::memcmp(got, &kF16ToFloat[h], sizeof got) != 0) return false;
  }
  return true;
}();
#endif
// 8x8 transposes of 32-bit values, for moving WMMA fragments between the
// register file (one array of lanes per register) and tiles (one row per
// element group). Shuffles only: the bits are moved, never interpreted.
#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx2"))) inline void transpose8x8(__m256 r[8]) {
  const __m256 t0 = _mm256_unpacklo_ps(r[0], r[1]), t1 = _mm256_unpackhi_ps(r[0], r[1]);
  const __m256 t2 = _mm256_unpacklo_ps(r[2], r[3]), t3 = _mm256_unpackhi_ps(r[2], r[3]);
  const __m256 t4 = _mm256_unpacklo_ps(r[4], r[5]), t5 = _mm256_unpackhi_ps(r[4], r[5]);
  const __m256 t6 = _mm256_unpacklo_ps(r[6], r[7]), t7 = _mm256_unpackhi_ps(r[6], r[7]);
  const __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44), u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
  const __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44), u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
  const __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44), u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
  const __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44), u7 = _mm256_shuffle_ps(t5, t7, 0xEE);
  r[0] = _mm256_permute2f128_ps(u0, u4, 0x20);
  r[1] = _mm256_permute2f128_ps(u1, u5, 0x20);
  r[2] = _mm256_permute2f128_ps(u2, u6, 0x20);
  r[3] = _mm256_permute2f128_ps(u3, u7, 0x20);
  r[4] = _mm256_permute2f128_ps(u0, u4, 0x31);
  r[5] = _mm256_permute2f128_ps(u1, u5, 0x31);
  r[6] = _mm256_permute2f128_ps(u2, u6, 0x31);
  r[7] = _mm256_permute2f128_ps(u3, u7, 0x31);
}
// out[l * 8 + k] = rows[k][first + l], for k, l < 8.
__attribute__((target("avx2"))) void gather8x8(const uint32_t* const rows[8], uint32_t first, void* out) {
  __m256 r[8];
  for (int k = 0; k < 8; ++k)
    r[k] = _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(rows[k] + first)));
  transpose8x8(r);
  for (int l = 0; l < 8; ++l) _mm256_storeu_ps(static_cast<float*>(out) + l * 8, r[l]);
}
// rows[k][first + l] = in[l * 8 + k], for k, l < 8.
__attribute__((target("avx2"))) void scatter8x8(const void* in, uint32_t* const rows[8], uint32_t first) {
  __m256 r[8];
  for (int l = 0; l < 8; ++l) r[l] = _mm256_loadu_ps(static_cast<const float*>(in) + l * 8);
  transpose8x8(r);
  for (int k = 0; k < 8; ++k)
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(rows[k] + first), _mm256_castps_si256(r[k]));
}
#else
// Elsewhere the callers take their scalar paths; these keep the code compiling.
void gather8x8(const uint32_t* const rows[8], uint32_t first, void* out) {
  for (int l = 0; l < 8; ++l)
    for (int k = 0; k < 8; ++k) std::memcpy(static_cast<uint32_t*>(out) + l * 8 + k, rows[k] + first + l, 4);
}
void scatter8x8(const void* in, uint32_t* const rows[8], uint32_t first) {
  for (int l = 0; l < 8; ++l)
    for (int k = 0; k < 8; ++k) std::memcpy(rows[k] + first + l, static_cast<const uint32_t*>(in) + l * 8 + k, 4);
}
#endif

void f16x16_to_float(const uint32_t* w8, float* out) {
#if defined(__x86_64__) && defined(__GNUC__)
  if (g_f16c) return f16x16_to_float_f16c(w8, out);
#endif
  f16x16_to_float_table(w8, out);
}

// tcgen05.mma's rows: acc[i] += a[k] * b[k * ldb + i] for k = 0..K-1 in order,
// i < n -- each sum in k order, the product and the sum each rounded to f32,
// as a one-element-at-a-time loop does; contraction off, as above, so no
// build fuses them. The integer form is exact in any order.
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
void tc_rows_f32(float* __restrict acc, const float* __restrict a, const float* __restrict b, uint32_t K,
                 uint32_t n, size_t ldb) {
  for (uint32_t k = 0; k < K; ++k) {
    const float ak = a[k];
    const float* __restrict bk = b + k * ldb;
    for (uint32_t i = 0; i < n; ++i) acc[i] += ak * bk[i];
  }
}
#pragma GCC pop_options
void tc_rows_i64(int64_t* __restrict acc, const int64_t* __restrict a, const int64_t* __restrict b, uint32_t K,
                 uint32_t n, size_t ldb) {
  for (uint32_t k = 0; k < K; ++k) {
    const int64_t ak = a[k];
    const int64_t* __restrict bk = b + k * ldb;
    for (uint32_t i = 0; i < n; ++i) acc[i] += ak * bk[i];
  }
}

uint64_t mask_to_bits(uint64_t v, uint32_t bits) {
  return bits >= 64 ? v : (v & ((1ull << bits) - 1));
}

// ---- texture and surface objects ----
//
// A texture fetch is an addressed read with a format conversion on the end.
// There is no texture cache modelled here: what is implemented is the part
// that changes results rather than timing. Both engines -- the PTX
// interpreter and the SASS executor -- sample through these functions (see
// vgpu/exec/texture.hpp), so a texel means the same thing to either. They
// throw vgpu::Error with what went wrong; the caller adds where.

[[noreturn]] void tex_fail(Err code, const std::string& msg) { throw Error(code, msg); }

// The address mode a fetch actually uses. Wrap and mirror are defined only
// for normalized coordinates; with unnormalized ones a real GPU (measured on
// an RTX 3060, point and linear alike) clamps instead.
TexAddress effective_address(const TextureDesc& d, uint32_t axis) {
  const TexAddress a = d.address[axis];
  if (!d.normalized_coords && (a == TexAddress::Wrap || a == TexAddress::Mirror)) return TexAddress::Clamp;
  return a;
}

// Applies the addressing mode. Returns false when the texel is outside and
// the mode says to produce the border colour rather than clamp to an edge.
bool wrap_coord(TexAddress mode, int64_t v, uint32_t size, uint32_t* out) {
  if (size == 0) { *out = 0; return true; }
  const int64_t n = static_cast<int64_t>(size);
  switch (mode) {
    case TexAddress::Clamp:
      if (v < 0) v = 0;
      if (v >= n) v = n - 1;
      break;
    case TexAddress::Wrap:
      v %= n;
      if (v < 0) v += n;
      break;
    case TexAddress::Mirror: {
      const int64_t period = 2 * n;
      int64_t t = v % period;
      if (t < 0) t += period;
      v = (t < n) ? t : (period - 1 - t);
      break;
    }
    case TexAddress::Border:
      if (v < 0 || v >= n) return false;
      break;
  }
  *out = static_cast<uint32_t>(v);
  return true;
}

// Reads one channel's raw bits out of a texel.
uint64_t texel_channel_bits(const MemoryManager& mem, const TextureDesc& d, uint64_t texel_addr, uint32_t ch) {
  if (d.packed_1010102) {   // 10:10:10:2, x in the low bits
    const uint64_t w = mem.load_scalar(texel_addr, 4);
    static constexpr uint32_t kShift[4] = {0, 10, 20, 30}, kBits[4] = {10, 10, 10, 2};
    return (w >> kShift[ch & 3]) & ((1u << kBits[ch & 3]) - 1);
  }
  uint32_t offset = 0;
  for (uint32_t i = 0; i < ch; ++i) offset += d.channel_bits[i] / 8;
  const uint32_t bytes = d.channel_bits[ch] / 8;
  if (bytes == 0) return 0;
  return mem.load_scalar(texel_addr + offset, bytes);
}

// ---- block-compressed textures ----
//
// The texel at (x, y, z) of a block-compressed texture is a texel of the decoded block that holds it; a
// fetch looks at one or a few blocks (a filter's footprint), so the last block decoded is kept, for the
// length of the fetch (texture_fetch clears it first, the texture's memory being free to change between
// fetches).
struct BcCache {
  bool valid = false;
  uint64_t addr = 0;
  BlockFormat format = BlockFormat::None;
  DecodedBlock block{};
};
thread_local BcCache t_bc;

uint64_t bc_texel_bits(const MemoryManager& mem, const TextureDesc& d, uint32_t x, uint32_t y, uint32_t z, uint32_t ch) {
  const uint64_t bb = block_bytes(d.block);
  const uint64_t blocks_per_row = (uint64_t{d.width} + 3) / 4;
  const uint64_t block_rows = ((d.height ? d.height : 1u) + 3u) / 4u;
  const uint64_t addr = d.base + uint64_t{z} * blocks_per_row * block_rows * bb + (y / 4) * blocks_per_row * bb + (x / 4) * bb;
  if (!t_bc.valid || t_bc.addr != addr || t_bc.format != d.block) {
    uint8_t raw[16];
    mem.read(addr, raw, bb);
    if (!decode_block(d.block, raw, &t_bc.block)) tex_fail(Err::Unsupported, "a block-compressed texture of an unknown format");
    t_bc.valid = true;
    t_bc.addr = addr;
    t_bc.format = d.block;
  }
  return t_bc.block.v[(y & 3) * 4 + (x & 3)][ch & 3];
}

// One channel of the texel at (x, y, z), whose address in a plain layout is `addr`.
uint64_t texel_raw(const MemoryManager& mem, const TextureDesc& d, uint64_t addr, uint32_t x, uint32_t y, uint32_t z,
                   uint32_t ch) {
  if (d.block != BlockFormat::None) return ch < d.channels ? bc_texel_bits(mem, d, x, y, z, ch) : 0;
  return texel_channel_bits(mem, d, addr, ch);
}

// sRGB decoding, as an RTX 3060's texture unit does it: through a table, not
// the sRGB formula -- its entries (in 1/65536ths, measured for every code)
// have at most 8 significant bits, and the low codes run at 20/65536 a
// step where 1/(255 * 12.92) would give 19.9. It applies to an 8-bit
// unsigned normalized texture read as normalized float: x, y and z of a
// four-channel texture, x of a one- or two-channel one; alpha, and every
// other format, are left alone.
static constexpr uint32_t kSrgbTable[256] = {
    0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220,
    240, 264, 288, 312, 340, 368, 396, 428, 460, 492, 524, 560,
    600, 636, 676, 720, 760, 804, 852, 896, 944, 1000, 1048, 1104,
    1160, 1216, 1272, 1328, 1392, 1456, 1520, 1584, 1648, 1720, 1792, 1864,
    1936, 2016, 2096, 2176, 2256, 2336, 2416, 2496, 2592, 2688, 2768, 2864,
    2960, 3056, 3152, 3264, 3360, 3456, 3584, 3680, 3776, 3904, 4000, 4128,
    4256, 4352, 4480, 4608, 4736, 4864, 4992, 5120, 5248, 5408, 5536, 5664,
    5824, 5952, 6112, 6240, 6400, 6560, 6688, 6848, 7008, 7168, 7328, 7488,
    7680, 7808, 8000, 8192, 8320, 8512, 8704, 8896, 9088, 9280, 9472, 9664,
    9856, 10048, 10240, 10432, 10624, 10816, 11008, 11264, 11456, 11648, 11904, 12096,
    12288, 12544, 12736, 12992, 13184, 13440, 13696, 13888, 14144, 14400, 14656, 14848,
    15104, 15360, 15616, 15872, 16128, 16384, 16640, 16896, 17152, 17408, 17664, 18048,
    18304, 18560, 18816, 19072, 19456, 19712, 19968, 20224, 20608, 20864, 21120, 21504,
    21760, 22144, 22400, 22784, 23040, 23296, 23680, 24064, 24320, 24704, 24960, 25344,
    25600, 25984, 26368, 26752, 27008, 27392, 27776, 28032, 28416, 28800, 29184, 29568,
    29952, 30336, 30720, 30976, 31488, 31744, 32256, 32512, 33024, 33280, 33792, 34048,
    34560, 35072, 35328, 35840, 36096, 36608, 37120, 37376, 37888, 38400, 38656, 39168,
    39680, 39936, 40448, 40960, 41216, 41728, 42240, 42752, 43264, 43520, 44032, 44544,
    45056, 45568, 45824, 46336, 46848, 47360, 47872, 48384, 48896, 49408, 49920, 50432,
    50944, 51456, 51968, 52480, 52992, 53504, 54016, 54528, 55040, 55552, 56064, 56576,
    57088, 57600, 58112, 58624, 59392, 59904, 60416, 60928, 61440, 61952, 62464, 62976,
    64000, 64512, 65024, 65536,
};
bool tex_srgb(const TextureDesc& d, uint32_t ch) {
  return d.srgb && d.kind == ChannelKind::Unsigned && d.channel_bits[ch] == 8 && d.read_as_normalized_float &&
         (ch == 0 || (ch < 3 && d.channel_bits[3] != 0));
}

// The border colour's channel `ch` as the raw bits of a texel of this
// format -- which is what the card fetches, filters and gathers in place of
// a texel outside the texture. Measured on an RTX 3060, over 280,000 border
// colours:
//  - a 32-bit float channel takes the bits as they are (NaN payloads too);
//  - a half channel takes the float rounded toward zero (65520 gives 65504;
//    a NaN keeps the top of its payload, and stays a NaN);
//  - a normalized channel of m magnitude bits clamps the float to [0, 1] or
//    [-1, 1] (NaN is 0), truncates it toward zero to m + 4 fractional bits,
//    v, and takes (|v| * (2^m - 1) + 2^(m+3) - 1) >> (m + 4), with v's sign
//    -- 0.5 on an 8-bit channel is 127, not 128;
//  - an integer channel read as an integer takes the float's low bits.
// A channel the format lacks reads 0, w included.
uint64_t tex_border_raw(const TextureDesc& d, uint32_t ch) {
  const uint32_t bits = d.channel_bits[ch];
  // An absent channel is 0 whatever the colour -- and must return before
  // the signed width below, which would wrap to 2^32-1 bits.
  if (bits == 0) return 0;
  const uint32_t b = d.border_bits[ch];
  const uint64_t mask = bits >= 64 ? ~0ull : (1ull << bits) - 1;
  if (d.kind == ChannelKind::Float) {
    if (bits != 16) return b;
    const uint32_t sign = (b >> 16) & 0x8000u, exp = (b >> 23) & 0xFF, man = b & 0x7FFFFFu;
    if (exp == 0xFF) return sign | 0x7C00u | (man ? std::max<uint32_t>(man >> 13, 1u) : 0u);
    const double a = std::fabs(static_cast<double>(f32(b)));
    if (a >= 65504.0) return sign | 0x7BFFu;
    if (a < std::ldexp(1.0, -14)) return sign | static_cast<uint32_t>(std::floor(std::ldexp(a, 24)));
    int e;
    const double fr = std::frexp(a, &e);   // a = fr * 2^e, fr in [0.5, 1)
    return sign | (static_cast<uint32_t>(e + 14) << 10) | (static_cast<uint32_t>(std::floor(std::ldexp(fr, 11))) & 0x3FFu);
  }
  if (!d.read_as_normalized_float) return b & mask;
  if (tex_srgb(d, ch)) {
    // An sRGB channel's border colour is encoded to the nearest sRGB code
    // -- by the sRGB formula, rounding a shade early: a fraction of 0.4989
    // already rounds up (measured over 7,100 colours) -- and decoded
    // through the table like a texel.
    double x = static_cast<double>(f32(b));
    if (std::isnan(x)) x = 0;
    x = std::clamp(x, 0.0, 1.0);
    const double e = x <= 0.0031308 ? x * 12.92 : 1.055 * std::pow(x, 1 / 2.4) - 0.055;
    return static_cast<uint64_t>(std::clamp(std::floor(e * 255 + 0.50108), 0.0, 255.0));
  }
  const bool sgn = d.kind == ChannelKind::Signed;
  const uint32_t m = bits - (sgn ? 1 : 0), fb = m + 4;
  double x = static_cast<double>(f32(b));
  if (std::isnan(x)) x = 0;
  x = std::clamp(x, sgn ? -1.0 : 0.0, 1.0);
  const int64_t v = static_cast<int64_t>(std::trunc(std::ldexp(x, static_cast<int>(fb))));
  const uint64_t mag = static_cast<uint64_t>(v < 0 ? -v : v);
  const int64_t k = static_cast<int64_t>((mag * ((1ull << m) - 1) + (1ull << (fb - 1)) - 1) >> fb);
  return static_cast<uint64_t>(v < 0 ? -k : k) & mask;
}

// Converts one channel to the 32 bits the destination register wants.
uint32_t convert_channel(const TextureDesc& d, uint32_t ch, uint64_t raw, bool float_result) {
  const uint32_t bits = d.channel_bits[ch];
  // A channel the format does not have reads 0 -- w too, measured on an RTX
  // 3060 for every format, read mode, filter and resource type (the
  // graphics APIs' w = 1 is not what CUDA's fetches return).
  if (bits == 0) {
    // A block-compressed format without an alpha channel has alpha 1 (measured for BC4 and BC5).
    return d.block != BlockFormat::None && ch == 3 ? f32bits(1.0f) : 0;
  }
  if (d.kind == ChannelKind::Float) {
    if (bits == 32) return static_cast<uint32_t>(raw);
    if (bits == 16) {
      // A NaN widens bit for bit, payload and all (a half 0x7c01 reads as
      // 0x7f802000 on an RTX 3060), rather than quieted.
      if ((raw & 0x7C00u) == 0x7C00u && (raw & 0x3FFu))
        return static_cast<uint32_t>(((raw & 0x8000u) << 16) | 0x7F800000u | ((raw & 0x3FFu) << 13));
      return static_cast<uint32_t>(f32bits(static_cast<float>(f16_to_double(raw))));
    }
    return 0;
  }
  if (tex_srgb(d, ch)) return static_cast<uint32_t>(f32bits(static_cast<float>(kSrgbTable[raw & 0xFF]) / 65536.0f));
  // Integer channels. Sign-extend first, because everything downstream --
  // both the integer result and the normalized float -- depends on it.
  int64_t sv = static_cast<int64_t>(raw);
  if (d.kind == ChannelKind::Signed && bits < 64) {
    const uint64_t sign = 1ull << (bits - 1);
    if (raw & sign) sv = static_cast<int64_t>(raw | ~((1ull << bits) - 1));
  }
  if (d.read_as_normalized_float) {
    // cudaReadModeNormalizedFloat: unsigned maps onto [0,1], signed onto
    // [-1,1], both by the widest magnitude the channel can hold.
    const double scale = static_cast<double>((1ull << (bits - (d.kind == ChannelKind::Signed ? 1 : 0))) - 1);
    double f = static_cast<double>(sv) / scale;
    if (d.kind == ChannelKind::Signed && f < -1.0) f = -1.0;
    return static_cast<uint32_t>(f32bits(static_cast<float>(f)));
  }
  if (float_result) return static_cast<uint32_t>(f32bits(static_cast<float>(sv)));
  return static_cast<uint32_t>(sv);
}

// ---- linear filtering ----
//
// The CUDA programming guide gives the formula -- tex(x) = (1-a)T[i] +
// aT[i+1] with x_B = x - 0.5, i = floor(x_B), a = frac(x_B) held in 9-bit
// fixed point with 8 fractional bits -- and the rest was measured on an RTX
// 3060 (sm_86), sample by sample, until every result matched to the bit:
//
//  - a rounds to the nearest 1/256, halves up; a = 256 carries into i.
//  - Normalized coordinates scale by the size in f32 first. In clamp mode
//    the coordinate is clamped to [0.5, size - 0.5] before x_B is formed,
//    which only shows in 3D, where the weights of two clamped-together
//    texels are rounded separately.
//  - A 1D texture is a 2D one of height 1 sampled at y = 0, so in border
//    mode half of every result comes from the border row.
//  - The 8 (or 4) weights are integers summing to 256, split one axis at a
//    time -- z, then x, then y -- each split rounding half up. The y split
//    rounds the upper part on the x = 1 side and the lower part on the x = 0
//    side; in 2D that is w11 = round(a*b/256), w10 = a - w11, w01 = b - w11.
//  - The sum of weight x texel is exact, then rounded once: to f32 (or to
//    f16 for a half texture), ties away from zero. 8- and 16-bit unsigned
//    normalized texels are filtered as 16-bit integers (an 8-bit u is u*257)
//    and 16-bit signed ones as themselves, rounded half up and read out as
//    K/65535 or K/32767 (clamped to -32767 after the blend).
//
int tex_round_half_up(int64_t num, int64_t den) {   // floor(num/den + 1/2)
  const int64_t t = 2 * num + den, d = 2 * den;
  return static_cast<int>(t >= 0 ? t / d : -((-t + d - 1) / d));
}

// The weight splits, as measured (see above). f[] are the 8-bit fractions
// along x, y, z; w[] is indexed x + 2y + 4z.
void tex_weights(uint32_t dims, const int f[3], int w[8], int total = 256) {
  auto split = [](int total, int frac, bool round_upper, int* lo, int* hi) {
    if (round_upper) { *hi = tex_round_half_up(int64_t{total} * frac, 256); *lo = total - *hi; }
    else { *lo = tex_round_half_up(int64_t{total} * (256 - frac), 256); *hi = total - *lo; }
  };
  for (int i = 0; i < 8; ++i) w[i] = 0;
  if (dims == 1) {
    split(total, f[0], true, &w[0], &w[1]);
    return;
  }
  // A mip level's share of the weight (total < 256) is split the same way,
  // z first when there is one, rounding the lower slice's part (measured on
  // 3D mipmaps; for the full 256 the split is exact either way).
  int z0 = total, z1 = 0;
  if (dims == 3) split(total, f[2], false, &z0, &z1);
  const int zslices = dims == 3 ? 2 : 1;
  for (int dz = 0; dz < zslices; ++dz) {
    const int tz = dz ? z1 : z0;
    int x0, x1;
    split(tz, f[0], true, &x0, &x1);
    for (int dx = 0; dx < 2; ++dx) {
      int y0, y1;
      split(dx ? x1 : x0, f[1], dx == 1, &y0, &y1);
      w[dx + 4 * dz] = y0;
      w[dx + 2 + 4 * dz] = y1;
    }
  }
}

// Exact sum of up to eight (weight x value) terms, as a non-overlapping
// expansion (Shewchuk's TwoSum): each product of a 9-bit weight and a float
// is exact in a double, and the expansion keeps every bit of their sum.
struct ExactSum {
  std::array<double, 20> e{};
  int n = 0;
  void add(double x) {
    int k = 0;
    for (int i = 0; i < n; ++i) {
      const double s = x + e[i], bb = s - x, err = (x - (s - bb)) + (e[i] - bb);
      x = s;
      if (err != 0) e[k++] = err;
    }
    e[k++] = x;
    n = k;
  }
  // The sign of (sum - v), exactly.
  int compare(double v) const {
    ExactSum t = *this;
    t.add(-v);
    for (int i = t.n - 1; i >= 0; --i)
      if (t.e[i] != 0) return t.e[i] > 0 ? 1 : -1;
    return 0;
  }
  double approx() const {
    double s = 0;
    for (int i = 0; i < n; ++i) s += e[i];
    return s;
  }
};

// Rounds the exact sum to f32, or to f16 when `half`, to nearest with ties
// away from zero, and returns the result as a float.
float tex_round_sum(const ExactSum& sum, bool half) {
  const double a = sum.approx();
  auto next = [half](double v, int dir) -> double {   // the adjacent value in the format
    if (half) {
      const uint64_t hb = double_to_f16(v) & 0xFFFF;
      const double c = f16_to_double(hb);
      if (c != v) return c;   // v was not representable: nearest is already a neighbour
      int64_t mag = hb & 0x7FFF;
      const bool neg = hb & 0x8000;
      if (mag == 0) return dir > 0 ? f16_to_double(0x0001) : f16_to_double(0x8001);
      mag += ((dir > 0) != neg) ? 1 : -1;
      return f16_to_double(static_cast<uint64_t>(mag) | (neg ? 0x8000u : 0u));
    }
    return static_cast<double>(std::nextafter(static_cast<float>(v), dir > 0 ? INFINITY : -INFINITY));
  };
  const double c = half ? f16_to_double(double_to_f16(a) & 0xFFFF) : static_cast<double>(static_cast<float>(a));
  const int s = sum.compare(c);
  if (s == 0) return static_cast<float>(c);
  const double lo = s > 0 ? c : next(c, -1), hi = s > 0 ? next(c, +1) : c;
  const int t = sum.compare(lo + (hi - lo) / 2);   // the midpoint is exact in a double
  if (t < 0) return static_cast<float>(lo);
  if (t > 0) return static_cast<float>(hi);
  return static_cast<float>(std::fabs(lo) > std::fabs(hi) ? lo : hi);
}

// One texel's share of a filtered fetch, in 1/256ths: all the terms of a
// fetch sum to 256. A border term reads the border colour.
struct TexTerm {
  int w = 0;
  uint64_t addr = 0;
  // The texel's place, and the texture it is a texel of (a mip level of the fetched one): what a
  // block-compressed texel is found by.
  uint32_t x = 0, y = 0, z = 0;
  const TextureDesc* src = nullptr;
  bool border = false;
  // Which 2x2 footprint the texel belongs to -- a 3D fetch's z-slice, a mip
  // blend's level -- since the float blend aligns each on its own (see
  // tex_finish).
  int group = 0;
};

void tex_check_filterable(const TextureDesc& d) {
  const uint32_t bits = d.channel_bits[0];
  const bool packed = d.packed_1010102;   // 10:10:10:2 filters as 16-bit unsigned normalized
  const bool is_float = d.kind == ChannelKind::Float && (bits == 32 || bits == 16);
  const bool unorm = d.kind == ChannelKind::Unsigned && (bits == 8 || bits == 16 || packed) && d.read_as_normalized_float;
  const bool snorm = d.kind == ChannelKind::Signed && (bits == 8 || bits == 16) && d.read_as_normalized_float;
  if (!is_float && !unorm && !snorm)
    tex_fail(Err::Unsupported,
             "linear filtering needs a float, half, or normalized 8/16-bit texture read as "
             "normalized float; this texture's format has no filtered form");
  // The channels of a block-compressed or packed format may differ in width (BC3's colour and alpha, 10:10:10:2).
  if (d.block == BlockFormat::None && !packed)
    for (uint32_t ch = 1; ch < 4; ++ch)
      if (d.channel_bits[ch] && d.channel_bits[ch] != bits)
        tex_fail(Err::Unsupported, "linear filtering of a texture whose channels differ in width");
}

// A normalized texture coordinate in texels of a level that is `size` wide. For a power of two that is the
// coordinate times the size, exactly. For any other size the card first floors the coordinate to a fixed-point
// number of 21 fractional bits (the integer part whole, negative coordinates toward minus infinity) and then
// multiplies it by the size: a point fetch at 0.04 on a 25-texel level reads texel 0 where the exact product
// (1.0000000708 for the float above 0.04) says 1, and the filter's weights follow the floored coordinate.
// Measured on an RTX 3060 over 300,000 coordinates of five sizes (the switch from texel k-1 to k at
// ceil(k 2^21 / size) / 2^21, and every 8-bit weight of a linear fetch).
double texel_coordinate(float x, uint32_t size) {
  if (size != 0 && (size & (size - 1)) == 0) return static_cast<double>(x * static_cast<float>(size));
  return std::floor(static_cast<double>(x) * 2097152.0) / 2097152.0 * static_cast<double>(size);
}

// floor(v) as an integer, saturating (an infinite or huge coordinate lands past the edge: the address mode
// decides, the clamp mode reads the edge texel, as the card does).
int64_t floor_index(double v) {
  const double f = std::floor(v);
  if (f >= 4e18) return int64_t{4000000000000000000};
  if (f <= -4e18) return int64_t{-4000000000000000000};
  return static_cast<int64_t>(f);
}

// The texels a linear filter reads and their weights, which sum to `total`.
int tex_footprint_linear(const TextureDesc& d, uint32_t dims, const float coord[3], bool true_1d, int total,
                         TexTerm* out) {
  // A 1D texture filters as 2D, height 1, at y = 0 -- all but a layer of a
  // 1D layered texture, which filters as 1D.
  const uint32_t fdims = dims == 1 && !true_1d ? 2 : dims;
  const uint32_t size[3] = {d.width, dims == 1 ? 1u : d.height, d.depth};
  int base[3] = {0, 0, 0}, frac[3] = {0, 0, 0};
  for (uint32_t i = 0; i < fdims; ++i) {
    float x = i < dims ? coord[i] : 0.0f;
    if (std::isnan(x)) x = 0.0f;   // a NaN coordinate reads as 0 (measured on an RTX 3060, clamp mode, all filters)
    double xd = static_cast<double>(x);
    if (d.normalized_coords) xd = texel_coordinate(x, size[i]);
    // The offset moves the coordinate by whole texels before the clamp mode
    // limits it (measured on an RTX 3060: clamp and linear filtering with
    // offsets, where the weights show the difference).
    double v = xd + d.fetch_offset[i];
    if (effective_address(d, i) == TexAddress::Clamp) v = std::clamp(v, 0.5, size[i] - 0.5);
    const double xb = v - 0.5;
    double fl = std::floor(xb);
    int f = static_cast<int>(std::floor((xb - fl) * 256 + 0.5));
    if (f >= 256) { fl += 1; f = 0; }
    base[i] = static_cast<int>(fl);
    frac[i] = f;
  }
  int w[8];
  tex_weights(fdims, frac, w, total);
  const uint64_t row = d.pitch_bytes ? d.pitch_bytes : uint64_t{d.width} * d.texel_bytes;
  const uint64_t plane = row * (d.height ? d.height : 1);
  int n = 0;
  for (int k = 0; k < (fdims == 3 ? 8 : fdims == 2 ? 4 : 2); ++k) {
    if (w[k] == 0) continue;
    uint32_t idx[3] = {0, 0, 0};
    bool inside = true;
    const int off[3] = {k & 1, (k >> 1) & 1, (k >> 2) & 1};
    for (uint32_t i = 0; i < fdims; ++i)
      if (!wrap_coord(effective_address(d, i), int64_t{base[i]} + off[i], size[i], &idx[i])) inside = false;
    TexTerm& t = out[n++];
    t.w = w[k];
    t.border = !inside;
    t.group = off[2];
    t.src = &d;
    t.x = idx[0], t.y = idx[1], t.z = idx[2];
    if (inside) t.addr = d.base + idx[2] * plane + idx[1] * row + uint64_t{idx[0]} * d.texel_bytes;
  }
  return n;
}

// The one texel a point fetch at float coordinates reads, with the whole
// `total` weight (for a point-sampled level inside a linear mip blend).
int tex_footprint_point(const TextureDesc& d, uint32_t dims, const float coord[3], int total, TexTerm* out) {
  const uint32_t size[3] = {d.width, d.height, d.depth};
  uint32_t idx[3] = {0, 0, 0};
  bool inside = true;
  for (uint32_t i = 0; i < dims; ++i) {
    const float xi = std::isnan(coord[i]) ? 0.0f : coord[i];   // a NaN coordinate reads as 0
    double f = static_cast<double>(xi);
    if (d.normalized_coords) f = texel_coordinate(xi, size[i]);
    if (!wrap_coord(effective_address(d, i), floor_index(f) + d.fetch_offset[i], size[i], &idx[i]))
      inside = false;
  }
  const uint64_t row = d.pitch_bytes ? d.pitch_bytes : uint64_t{d.width} * d.texel_bytes;
  const uint64_t plane = row * (d.height ? d.height : 1);
  out[0].w = total;
  out[0].border = !inside;
  out[0].src = &d;
  out[0].x = idx[0], out[0].y = idx[1], out[0].z = idx[2];
  out[0].addr = inside ? d.base + idx[2] * plane + idx[1] * row + uint64_t{idx[0]} * d.texel_bytes : 0;
  return 1;
}

// An unsigned normalized channel of `bits` bits as the 16-bit integer the filter works in: the code's bits
// repeated to fill 16 (8-bit codes are u * 257, the 10 and 2 bits of 10:10:10:2 are (u << 6) | (u >> 4) and
// u * 21845), 16-bit ones as they are. Measured on an RTX 3060 with all 1024 codes of 10 bits (linear
// filtering at texel centres): bit replication, not a rounded scaling.
uint32_t unorm16(uint64_t raw, uint32_t bits) {
  if (bits >= 16) return static_cast<uint32_t>(raw);
  uint32_t v = static_cast<uint32_t>(raw) << (16 - bits);
  for (uint32_t s = bits; s < 16; s *= 2) v |= v >> s;
  return v;
}

// Sums weight x texel over the terms and rounds, by the format's rules.
void tex_finish(const MemoryManager& mem, const TextureDesc& d, const TexTerm* terms, int n, uint32_t out[4]) {
  const bool is_float = d.kind == ChannelKind::Float;
  const bool unorm = d.kind == ChannelKind::Unsigned;
  int64_t isum[4] = {0, 0, 0, 0};
  double fv[4][16];
  uint32_t codes[4][16] = {};
  for (int k = 0; k < n; ++k) {
    for (uint32_t ch = 0; ch < 4; ++ch) {
      if (!d.channel_bits[ch]) continue;
      if (terms[k].w == 0) { fv[ch][k] = 0; continue; }
      const uint32_t bits = d.channel_bits[ch];   // the channels of a block-compressed format differ in width
      const TextureDesc& at = terms[k].src ? *terms[k].src : d;
      const uint64_t raw = terms[k].border ? tex_border_raw(d, ch)
                                           : texel_raw(mem, at, terms[k].addr, terms[k].x, terms[k].y, terms[k].z, ch);
      if (is_float) {
        double t = bits == 32 ? static_cast<double>(f32(raw)) : f16_to_double(raw);
        if (bits == 32 && std::fabs(t) < std::ldexp(1.0, -126)) t = std::copysign(0.0, t);   // flushed
        fv[ch][k] = t;
      } else if (tex_srgb(d, ch)) {
        codes[ch][k] = static_cast<uint32_t>(raw & 0xFF);
      } else if (unorm) {
        isum[ch] += int64_t{terms[k].w} * static_cast<int64_t>(unorm16(raw, bits));
      } else if (bits == 8) {
        isum[ch] += int64_t{terms[k].w} * static_cast<int8_t>(raw);   // the 8-bit codes, see the blend below
      } else {
        isum[ch] += int64_t{terms[k].w} * static_cast<int16_t>(raw);
      }
    }
  }
  for (uint32_t ch = 0; ch < 4; ++ch) {
    float r;
    const uint32_t bits = d.channel_bits[ch];
    if (!d.channel_bits[ch]) {
      r = d.block != BlockFormat::None && ch == 3 ? 1.0f : 0.0f;   // a block-compressed format without alpha has alpha 1
    } else if (tex_srgb(d, ch)) {
      // An sRGB channel blends its table values (measured over 134,316
      // two-texel blends): the table holds codes in blocks of eight sharing
      // an exponent -- that of the block's last entry -- and within each
      // 2x2 footprint (a 3D fetch's z-slice, a mip blend's level) each value
      // is truncated to 2^(E - 7), E the largest block exponent among the
      // footprint's texels with weight; the footprints' weighted sums are
      // added exactly and rounded once to a half's precision, ties away
      // from zero.
      int Eg[4] = {INT_MIN, INT_MIN, INT_MIN, INT_MIN};
      for (int k = 0; k < n; ++k)
        if (terms[k].w && kSrgbTable[codes[ch][k]])
          Eg[terms[k].group] = std::max(Eg[terms[k].group], std::ilogb(static_cast<double>(kSrgbTable[codes[ch][k] | 7])) - 16);
      const int Emax = *std::max_element(Eg, Eg + 4);
      uint64_t sum = 0;   // in 2^-24ths
      for (int g = 0; g < 4 && Emax != INT_MIN; ++g) {
        if (Eg[g] == INT_MIN) continue;
        const int shift = Eg[g] - 7 + 16;   // the grid, in table units (2^-16)
        uint64_t sg = 0;
        for (int k = 0; k < n; ++k) {
          if (!terms[k].w || terms[k].group != g) continue;
          const uint64_t t = kSrgbTable[codes[ch][k]];
          sg += uint64_t(terms[k].w) * (shift > 0 ? (t >> shift) << shift : t);
        }
        sum += sg;
      }
      if (sum) {
        const int e = 63 - __builtin_clzll(sum) - 24;          // the value's exponent
        const int drop = std::max(e, -14) - 10 + 24;           // bits below a half's last
        if (drop > 0) sum = ((sum >> (drop - 1)) + 1) >> 1 << drop;
      }
      r = static_cast<float>(std::ldexp(static_cast<double>(sum), -24));
    } else if (is_float) {
      // As measured on an RTX 3060, the blend is not an exact sum. Within
      // each 2x2 footprint (a 3D fetch's z-slice, a mip blend's level) every
      // value is truncated toward zero to 2^(E - 27) -- 2^(E - 14) for a
      // half -- where E is the exponent of the footprint's largest value
      // with a non-zero weight: 2.75 with weight 1 and 0.1 with weight 255
      // gives 0x3de207ff, not 0x3de20800. Each footprint's weighted sum is
      // then floored to 2^(Emax - 28), Emax the largest E, the footprints
      // added exactly, and the total rounded once, ties away from zero.
      // (Scaling by 1/256 is exact, so the sum of weight x value is rounded
      // and divided after.) Over 15,360 trilinear fetches of random texels
      // spanning 2^-10..2^11, 20 still differ, by 1-4 ulp, all where the
      // slices' magnitudes differ widely and the result cancels to far
      // below them. A non-finite value takes the plain arithmetic.
      const int M = bits == 32 ? 27 : 14;
      int Eg[4] = {INT_MIN, INT_MIN, INT_MIN, INT_MIN};
      bool finite = true, all_neg_zero = true;
      for (int k = 0; k < n; ++k) {
        if (terms[k].w == 0) continue;
        const double t = fv[ch][k];
        if (!std::isfinite(t)) finite = false;
        else if (t != 0) Eg[terms[k].group] = std::max(Eg[terms[k].group], std::ilogb(t));
        if (!(t == 0 && std::signbit(t))) all_neg_zero = false;
      }
      const int Emax = *std::max_element(Eg, Eg + 4);
      ExactSum scaled;
      double plain = 0;
      for (int k = 0; k < n; ++k)
        if (terms[k].w) plain += terms[k].w * fv[ch][k];
      if (finite && Emax != INT_MIN) {
        const double q2 = std::ldexp(1.0, Emax - (M + 1));
        for (int g = 0; g < 4; ++g) {
          if (Eg[g] == INT_MIN) continue;
          // Exact in a double: at most 8 terms of a 9-bit weight times a
          // value of at most M + 1 bits, all multiples of q.
          const double q = std::ldexp(1.0, Eg[g] - M);
          double sg = 0;
          for (int k = 0; k < n; ++k)
            if (terms[k].w && terms[k].group == g) sg += terms[k].w * (std::trunc(fv[ch][k] / q) * q);
          scaled.add(std::floor(sg / q2) * q2 / 256);
        }
      }
      r = finite ? tex_round_sum(scaled, bits == 16) : static_cast<float>(plain / 256);
      if (finite && r == 0 && all_neg_zero) r = -0.0f;
      // A result below the smallest normal float is flushed, keeping its sign.
      if (bits == 32 && std::fabs(r) < std::ldexp(1.0f, -126)) r = std::copysign(0.0f, r);
      // A NaN comes out as the filter's own, all ones in the format.
      if (std::isnan(r)) { out[ch] = bits == 32 ? 0x7FFFFFFFu : 0x7FFFE000u; continue; }
    } else {
      int K;
      if (!unorm && bits == 8) {
        // A signed 8-bit normalized blend: the weighted sum S of the 8-bit
        // codes (weights summing to 256) comes out as S + (257 * (S >> 4) +
        // 1024) >> 11, over 32767 -- the sum's 16-bit form being a function
        // of S alone, found over all 65,281 sums an RTX 3060 can make of
        // two texels (every sum from -32768 to 32512), where the fit is
        // exact; clamped to -32767 after the blend.
        const int64_t S = isum[ch];
        K = static_cast<int>(std::max<int64_t>(S + ((257 * (S >> 4) + 1024) >> 11), -32767));
      } else {
        K = std::max(tex_round_half_up(isum[ch], 256), unorm ? 0 : -32767);
      }
      r = static_cast<float>(static_cast<double>(K) / (unorm ? 65535.0 : 32767.0));
    }
    out[ch] = static_cast<uint32_t>(f32bits(r));
  }
}

void tex_linear(const MemoryManager& mem, const TextureDesc& d, uint32_t dims, const float coord[3], uint32_t out[4],
                bool true_1d = false) {
  tex_check_filterable(d);
  TexTerm terms[8];
  const int n = tex_footprint_linear(d, dims, coord, true_1d, 256, terms);
  tex_finish(mem, d, terms, n, out);
}

// An explicit level of detail through a texture whose descriptor has a
// maxAnisotropy of 2 or more and whose mip filter is linear is blended between its
// two levels with a sharper weight than the fraction of the level (measured on an
// RTX 3060, every fraction of every geometry): the weight is 0 for the first part
// of the fraction and 256 (the upper level alone) for the last, and goes up at 3/2,
// 7/4 or 2 times the rate in between -- for a maxAnisotropy of 2-3, 4-7 and 8 or
// more. The ramp starts at lo = 128 (1 - 1/rate) 256ths of a level: 128/3, 128 * 3/7
// and 64. The bias is not added to the level of detail first, as it is without
// anisotropy: the card takes lod + trunc(bias - lo) + trunc(lo), with the bias in
// 256ths and not truncated, truncating toward zero both times -- so a bias of 0 starts
// the ramp at the floor of lo (42, 54, 64), a bias above lo at the ceiling of it, and a
// fractional bias moves the ramp a whole 256th when it passes lo's own fraction (checked
// for biases from -300 to 300 256ths in steps of 1/8, and next to 128/3 and 3 * 128/7
// to 1/2048). A texture whose mip filter is point, and a fetch with no explicit level,
// are not affected; the level clamps apply to the level of detail this makes.
int64_t tex_aniso_lod(const TextureDesc& d, int64_t lod256) {
  const uint32_t n = d.max_anisotropy;
  const double lo = n < 4 ? 128.0 / 3 : n < 8 ? 128.0 * 3 / 7 : 64.0;
  const int64_t lo_floor = static_cast<int64_t>(std::trunc(lo));
  const int64_t t = lod256 + static_cast<int64_t>(std::trunc(d.mip_bias_exact - lo)) + lo_floor;
  const int64_t level = t >> 8, k = t & 255;
  const int64_t num = n < 4 ? 3 : n < 8 ? 7 : 2, den = n < 4 ? 2 : n < 8 ? 4 : 1;
  const int64_t weight = k <= lo_floor ? 0 : std::min<int64_t>(256, num * (k - lo_floor) / den);
  return level * 256 + weight;
}

// A mipmapped fetch's level of detail, in 1/256ths of a level (measured on
// an RTX 3060): an explicit lod is truncated toward zero to 1/256 and the
// bias added, a plain fetch is level 0 without the bias; an anisotropic
// texture with a linear mip filter instead sharpens the blend between the two
// levels (tex_aniso_lod); then the texture's level clamps, then the levels
// that exist.
int32_t tex_mip_lod(const TextureDesc& d, bool explicit_lod, double lod) {
  int64_t q = 0;
  if (explicit_lod) {
    const double scaled = std::trunc(lod * 256);
    q = static_cast<int64_t>(std::clamp(scaled, -1e9, 1e9));
    q = d.max_anisotropy >= 2 && d.mip_filter == TexFilter::Linear ? tex_aniso_lod(d, q) : q + d.mip_bias;
  }
  q = std::clamp<int64_t>(q, d.mip_min, std::max(d.mip_min, d.mip_max));
  q = std::clamp<int64_t>(q, 0, int64_t{d.mip_levels - 1} * 256);
  return static_cast<int32_t>(q);
}

// A mip level of `d` as a texture of its own.
uint64_t tex_slice_bytes(const TextureDesc& d);

TextureDesc tex_level(const TextureDesc& d, uint32_t level) {
  TextureDesc v = d;
  v.base = d.level_base[level];
  v.width = std::max(1u, d.width >> level);
  v.height = d.height ? std::max(1u, d.height >> level) : 0;
  v.depth = d.depth ? std::max(1u, d.depth >> level) : 0;
  v.pitch_bytes = v.width * d.texel_bytes;
  v.base += uint64_t{d.mip_slice} * tex_slice_bytes(v);
  v.mip_levels = 0;
  v.mip_slice = 0;
  return v;
}

// The texture a layered or cubemap fetch actually reads: one layer, or one
// face, as an ordinary 1D or 2D texture. Measured on an RTX 3060: the layer
// (and a layered cubemap's cubemap) index is unsigned and an index past the
// end -- a negative one included -- reads the last; a cube direction picks
// the face of its largest-magnitude axis, ties going to z, then y, then x,
// with the minor axes from the CUDA programming guide's table and (s/m+1)/2
// as the face coordinate; and filtering stays inside the face, under the
// texture's address mode.
uint64_t tex_slice_bytes(const TextureDesc& d) {
  if (d.block != BlockFormat::None)   // a slice of blocks
    return ((uint64_t{d.width} + 3) / 4) * (((d.height ? d.height : 1u) + 3u) / 4) * block_bytes(d.block);
  const uint64_t row = d.pitch_bytes ? d.pitch_bytes : uint64_t{d.width} * d.texel_bytes;
  return row * (d.height ? d.height : 1);
}

void fetch_texel(const MemoryManager& mem, const TextureDesc& d, const TexFetch& f, uint32_t out[4]) {
  const bool indexed = f.layered, cube = f.cube;
  if ((d.layers != 0) != indexed || d.cubemap != cube)
    tex_fail(Err::InvalidValue, std::string("the fetch's geometry does not match the texture: the texture is ") +
                                    (d.cubemap ? (d.layers ? "a layered cubemap" : "a cubemap")
                                               : (d.layers ? "layered" : "neither layered nor a cubemap")));
  TextureDesc v = d;
  for (int i = 0; i < 3; ++i) v.fetch_offset[i] = f.offset[i];
  uint32_t dims = f.dims;
  bool true_1d = false;
  float cf[3] = {0, 0, 0};
  int64_t ci[3] = {0, 0, 0};
  for (uint32_t i = 0; i < f.dims; ++i) {
    cf[i] = f32(f.coord[i]);
    ci[i] = static_cast<int32_t>(f.coord[i]);
  }
  if (indexed) {
    const uint64_t layer = std::min<uint64_t>(f.layer, d.layers - 1);
    v.base += layer * (cube ? 6 : 1) * tex_slice_bytes(d);
    v.mip_slice += static_cast<uint32_t>(layer * (cube ? 6 : 1));
    v.layers = 0;
    if (f.dims == 1 && !cube) {
      true_1d = true;   // a layer of a 1D layered texture filters as 1D
      v.height = 0;
    }
  }
  if (cube) {
    const float x = cf[0], y = cf[1], z = cf[2];
    const float ax = std::fabs(x), ay = std::fabs(y), az = std::fabs(z);
    int face;
    float sc, tc, ma;
    if (az >= ax && az >= ay) { face = z >= 0 ? 4 : 5; sc = z >= 0 ? x : -x; tc = -y; ma = az; }
    else if (ay >= ax) { face = y >= 0 ? 2 : 3; sc = x; tc = y >= 0 ? z : -z; ma = ay; }
    else { face = x >= 0 ? 0 : 1; sc = x >= 0 ? -z : z; tc = -y; ma = ax; }
    v.base += static_cast<uint64_t>(face) * tex_slice_bytes(d);
    v.mip_slice += static_cast<uint32_t>(face);
    v.cubemap = false;
    v.normalized_coords = true;
    // Point sampling clamps to the face whatever the address mode -- unless
    // the cubemap is mipmapped, when it applies the mode like linear
    // filtering does; linear filtering applies the mode inside the face --
    // wrap takes the texel from the face's far edge, border blends in the
    // border colour (measured).
    if (v.filter != TexFilter::Linear && !d.mip_levels && f.gather < 0)
      for (auto& a : v.address) a = TexAddress::Clamp;
    cf[0] = (sc / ma + 1.0f) * 0.5f;
    cf[1] = (tc / ma + 1.0f) * 0.5f;
    dims = 2;
  }
  if (f.gather >= 0) {
    // tld4: the four texels of the bilinear footprint, counter-clockwise
    // from the lower left -- (i, j+1), (i+1, j+1), (i+1, j), (i, j) -- one
    // component each. Measured on an RTX 3060: i and j come from the
    // filter's coordinate (with its 8-bit weight rounding carrying into
    // them), and the address mode applies to each texel's index -- clamp
    // clamps the indices, not the coordinate.
    if (d.mip_levels) tex_fail(Err::Unsupported, "tld4 of a mipmapped texture");
    const uint32_t size[2] = {v.width, v.height ? v.height : 1};
    int64_t b[2];
    for (uint32_t i = 0; i < 2; ++i) {
      const float xi = std::isnan(cf[i]) ? 0.0f : cf[i];
      double xd = static_cast<double>(xi);
      if (v.normalized_coords) xd = texel_coordinate(xi, size[i]);
      const double xb = xd - 0.5;
      double fl = std::floor(xb);
      if (std::floor((xb - fl) * 256 + 0.5) >= 256) fl += 1;
      b[i] = static_cast<int64_t>(fl) + v.fetch_offset[i];
    }
    const uint64_t row = v.pitch_bytes ? v.pitch_bytes : uint64_t{v.width} * v.texel_bytes;
    static constexpr int kOrder[4][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    for (int k = 0; k < 4; ++k) {
      uint32_t ix, iy;
      const bool inside = wrap_coord(effective_address(v, 0), b[0] + kOrder[k][0], size[0], &ix) &&
                          wrap_coord(effective_address(v, 1), b[1] + kOrder[k][1], size[1], &iy);
      const uint32_t ch = static_cast<uint32_t>(f.gather);
      const uint64_t raw = inside ? texel_raw(mem, v, v.base + iy * row + uint64_t{ix} * v.texel_bytes, ix, iy, 0, ch)
                                  : tex_border_raw(v, ch);
      uint32_t r = convert_channel(v, ch, raw, f.float_result);
      if (v.kind == ChannelKind::Signed && v.channel_bits[ch] == 8 && v.read_as_normalized_float) {
        // A signed 8-bit normalized texel is gathered as the 16-bit value
        // the filter works in, read out over 32767, not as k / 127:
        // |k| * 258, plus one from |k| = 64 up (65 when negative) --
        // measured for every code on an RTX 3060.
        const int k8 = std::max(static_cast<int>(static_cast<int8_t>(raw)), -127);
        const int mag = k8 < 0 ? -k8 : k8;
        const int k16 = mag * 258 + (mag >= (k8 < 0 ? 65 : 64) ? 1 : 0);
        r = static_cast<uint32_t>(f32bits(static_cast<float>(static_cast<double>(k8 < 0 ? -k16 : k16) / 32767.0)));
      }
      // A NaN comes out as the gather's own, all ones in the format, and a
      // subnormal float as zero (measured).
      if (v.kind == ChannelKind::Float && v.channel_bits[ch]) {
        if (std::isnan(f32(r))) r = v.channel_bits[ch] == 32 ? 0x7FFFFFFFu : 0x7FFFE000u;
        else if ((r & 0x7F800000u) == 0) r &= 0x80000000u;
      }
      out[k] = r;
    }
    return;
  }
  if (d.mip_levels) {
    // A mipmapped texture: pick the level, or blend two (see tex_mip_lod).
    // Its coordinates are normalized whatever the descriptor says, and
    // without normalizedCoords wrap and mirror still act as clamp (measured:
    // x = 0.25 on an 8-wide level 0 reads texel 2).
    if (!v.normalized_coords) {
      for (auto& a : v.address)
        if (a == TexAddress::Wrap || a == TexAddress::Mirror) a = TexAddress::Clamp;
      v.normalized_coords = true;
    }
    const int32_t q = tex_mip_lod(d, f.explicit_lod, f.lod);
    if (d.mip_filter == TexFilter::Linear && (q & 255) != 0) {
      if (!f.float_coords) tex_fail(Err::Unsupported, "blending mip levels with integer coordinates");
      tex_check_filterable(d);
      const TextureDesc lo = tex_level(v, static_cast<uint32_t>(q >> 8));
      const TextureDesc hi = tex_level(v, static_cast<uint32_t>(q >> 8) + 1);
      TexTerm terms[16];
      int n = 0;
      const int wh = q & 255;
      if (v.filter == TexFilter::Linear) {
        n += tex_footprint_linear(lo, dims, cf, true_1d, 256 - wh, terms + n);
        const int first_hi = n;
        n += tex_footprint_linear(hi, dims, cf, true_1d, wh, terms + n);
        for (int k = first_hi; k < n; ++k) terms[k].group += 2;
      } else {
        n += tex_footprint_point(lo, dims, cf, 256 - wh, terms + n);
        n += tex_footprint_point(hi, dims, cf, wh, terms + n);
        terms[n - 1].group = 2;
      }
      tex_finish(mem, d, terms, n, out);
      return;
    }
    v = tex_level(v, static_cast<uint32_t>(d.mip_filter == TexFilter::Linear ? q >> 8 : (q + 128) >> 8));
  }
  // Integer coordinates name a texel, and the filter mode does not apply: an
  // RTX 3060 point-samples them even from a texture set up for linear
  // filtering (tex1Dfetch from linear memory, and tex.2d.*.s32 on an array
  // alike). CUDA Samples' convolutionFFT2D binds its buffers that way.
  if (v.filter == TexFilter::Linear && f.float_coords) {
    tex_linear(mem, v, dims, cf, out, true_1d);
    return;
  }

  const uint32_t size[3] = {v.width, v.height, v.depth};
  bool inside = true, int_outside = false;
  uint32_t idx[3] = {0, 0, 0};
  for (uint32_t i = 0; i < dims; ++i) {
    int64_t c;
    if (f.float_coords) {
      const float xi = std::isnan(cf[i]) ? 0.0f : cf[i];
      double xd = static_cast<double>(xi);
      if (v.normalized_coords) xd = texel_coordinate(xi, size[i]);
      // Point sampling takes the texel the coordinate falls in. CUDA's
      // sampled coordinates are texel-centred, so x+0.5 addresses texel x.
      c = floor_index(xd) + v.fetch_offset[i];
    } else {
      // An integer coordinate names a texel directly, and outside the extent
      // reads zero: an RTX 3060 applies neither the address mode nor the
      // border colour to it (clamp and border alike, with a border colour
      // of 7, give 0).
      c = ci[i] + v.fetch_offset[i];
      if (c < 0 || c >= static_cast<int64_t>(size[i])) { inside = false; int_outside = true; break; }
      idx[i] = static_cast<uint32_t>(c);
      continue;
    }
    if (!wrap_coord(effective_address(v, i), c, size[i], &idx[i])) { inside = false; break; }
  }

  // A 1D texture is a 2D one of height 1, and the offset the instruction carries is read as three
  // nibbles -- x, y, z -- so a negative 1D offset, which ptxas sign-extends into the nibbles above
  // x, moves the row to -1 (fetch_offset[1]): outside, under border addressing.
  if (inside && !int_outside && dims == 1 && !true_1d && v.fetch_offset[1] != 0 &&
      effective_address(v, 1) == TexAddress::Border)
    inside = false;
  if (int_outside) {
    for (uint32_t ch = 0; ch < 4; ++ch) out[ch] = 0;
    return;
  }
  if (!inside) {
    // Border addressing outside the extent: the border colour.
    for (uint32_t ch = 0; ch < 4; ++ch) out[ch] = convert_channel(v, ch, tex_border_raw(v, ch), f.float_result);
    return;
  }
  const uint64_t row = v.pitch_bytes ? v.pitch_bytes : uint64_t{v.width} * v.texel_bytes;
  const uint64_t plane = row * (v.height ? v.height : 1);
  const uint64_t addr = v.base + idx[2] * plane + idx[1] * row + uint64_t{idx[0]} * v.texel_bytes;
  for (uint32_t ch = 0; ch < 4; ++ch)
    out[ch] = convert_channel(v, ch, texel_raw(mem, v, addr, idx[0], idx[1], idx[2], ch), f.float_result);
}

// suld/sust address a surface in *bytes* along x and in whole rows along y
// and z, which is why they take no format: they move raw bytes.
//
// The out-of-range policy (9.7.13.1-2), as an RTX 3060 applies it:
//   .trap   faults;
//   .clamp  moves each coordinate to the nearest place in the surface: x to
//           the last position, aligned to the access, at which the whole
//           access fits (a 16-byte .v4 on a 24-byte row reads from 0, not
//           8), y and z into their range, and the layer to the last one;
//   .zero   reads zero and drops the store if any byte of the access is
//           out of range -- the whole access, not only its outside part.
// Null for a .zero access out of range. An x not aligned to the access
// faults under every policy, as it does on the card (the ISA leaves it
// undefined).
std::optional<uint64_t> surface_at(const TextureDesc& d, const SurfaceAccess& s) {
  // A layered surface's first coordinate is its layer. A cubemap surface is
  // read as a layered one, face by face (surfCubemapread compiles to
  // suld.a2d with the face as the layer), so it has 6 -- or 6 x layers.
  int64_t x = s.x, y = s.dims > 1 ? s.y : 0, z = s.dims > 2 ? s.z : 0;
  const uint32_t bytes = s.bytes;
  const uint8_t oob = s.oob;
  if (oob != kSurfaceTrap && x % static_cast<int64_t>(bytes) != 0)
    tex_fail(Err::MisalignedAccess, "surface access at byte x=" + std::to_string(x) + " is not aligned to its " +
                                        std::to_string(bytes) +
                                        "-byte size (an RTX 3060 faults; the ISA leaves it undefined)");
  uint64_t layer_base = d.base;
  if (s.layered) {
    const uint64_t layers = d.cubemap ? 6 * std::max<uint64_t>(d.layers, 1) : d.layers;
    if (layers == 0)
      tex_fail(Err::InvalidValue, "a layered surface access (.a1d/.a2d) on a surface that is not layered");
    // Reading a 2D layered surface with a 1D layered access (or the other
    // way round) returns zeros on an RTX 3060, a rule nobody can rely on;
    // it is refused instead.
    if ((s.dims == 1) != (d.height == 0))
      tex_fail(Err::InvalidValue, std::string("a ") + (s.dims == 1 ? "1D" : "2D") + " layered access (.a" +
                                      (s.dims == 1 ? "1d" : "2d") + ") on a " + (d.height ? "2D" : "1D") +
                                      " layered surface");
    uint64_t layer = s.layer;
    if (layer >= layers && oob == kSurfaceZero) return std::nullopt;
    if (layer >= layers && oob == kSurfaceClamp) layer = layers - 1;
    if (layer >= layers)
      tex_fail(Err::OutOfBounds, "surface layer " + std::to_string(layer) + " is past the surface's " +
                                     std::to_string(layers) + " layers, and the '.trap' policy faults");
    layer_base += layer * tex_slice_bytes(d);
  } else if (d.layers || d.cubemap) {
    tex_fail(Err::InvalidValue, "a layered or cubemap surface needs a layered access (.a1d/.a2d)");
  }
  const uint64_t row = d.pitch_bytes ? d.pitch_bytes : uint64_t{d.width} * d.texel_bytes;
  const uint64_t plane = row * (d.height ? d.height : 1);
  // ".trap" is the out-of-range policy ptxas emits, and it means what it
  // says: the access faults rather than being clamped or dropped.
  const int64_t row_bytes = static_cast<int64_t>(uint64_t{d.width} * d.texel_bytes);
  const int64_t size = bytes;
  const bool out = x < 0 || x + size > row_bytes || (d.height && (y < 0 || y >= static_cast<int64_t>(d.height))) ||
                   (d.depth && (z < 0 || z >= static_cast<int64_t>(d.depth)));
  if (out && oob == kSurfaceZero) return std::nullopt;
  if (out && oob == kSurfaceClamp) {
    if (row_bytes < size)
      tex_fail(Err::UnsupportedPtx, "a .clamp surface access of " + std::to_string(size) + " bytes on a row of " +
                                        std::to_string(row_bytes) +
                                        ": no position holds it, and what the card does was not measured");
    x = std::clamp<int64_t>(x, 0, (row_bytes - size) / size * size);
    if (d.height) y = std::clamp<int64_t>(y, 0, static_cast<int64_t>(d.height) - 1);
    if (d.depth) z = std::clamp<int64_t>(z, 0, static_cast<int64_t>(d.depth) - 1);
  }
  if (x < 0 || x + static_cast<int64_t>(bytes) > row_bytes ||
      (d.height && (y < 0 || y >= static_cast<int64_t>(d.height))) ||
      (d.depth && (z < 0 || z >= static_cast<int64_t>(d.depth))))
    tex_fail(Err::OutOfBounds, "surface access at byte x=" + std::to_string(x) + ", y=" + std::to_string(y) +
                                   " is outside the " + std::to_string(d.width) + "x" + std::to_string(d.height) +
                                   " surface (" + std::to_string(row_bytes) +
                                   " bytes per row). The instruction's '.trap' policy is what makes this a fault "
                                   "rather than a clamp");
  return layer_base + z * plane + y * row + static_cast<uint64_t>(x);
}

// Where stored element `k` of chunk `chunk` in row `row` (0-15) of a
// structured-sparse A goes, as a column of the K-wide row: the metadata
// rule of mma.sp, which wgmma.mma_async.sp follows for each warp's 16 rows.
// A chunk is four elements (16- and 8-bit types, 2:4), two (tf32, 1:2) or
// eight (int4, 4:8 in pairs), and its metadata is 4 bits, two 2-bit
// indices of "units": an element, half a tf32 element (0b0100 and 0b1110
// are tf32's only values), or a pair of int4 elements. Which lane of the
// group of four carries a chunk's metadata: for 16- and 32-bit types, lane
// 4g + H * selector + chunk / 4, four chunks of both rows to a lane, the
// second row in the high half; for 8- and 4-bit types, a row to a lane --
// row g from lane 4g + 2 * selector, row g + 8 from the next, chunks 8-15
// from the pair after. H is the number of lanes holding the group's
// metadata. All of it as an RTX 3060 (sm_86) places it.
uint32_t sparse_column(uint32_t bits, uint32_t K, uint32_t sel, const Lanes& meta, uint32_t row,
                              uint32_t chunk, uint32_t k) {
  const bool tf32 = bits == 32, int4 = bits == 4, narrow = bits <= 8;
  const uint32_t chunk_elems = tf32 ? 2 : int4 ? 8 : 4;
  const uint32_t row_bits = tf32 ? 2 * K : int4 ? K / 2 : K;
  const uint32_t holders = 2 * row_bits / 32;
  const uint32_t group = row % 8, half = row / 8;
  const uint32_t src = narrow ? 4 * group + 2 * sel + half + 2 * (chunk / 8)
                              : 4 * group + holders * sel + chunk / 4;
  const uint32_t shift = narrow ? (chunk % 8) * 4 : half * 16 + (chunk % 4) * 4;
  const uint32_t nib = static_cast<uint32_t>(meta[src]) >> shift;
  const uint32_t idx0 = nib & 3, idx1 = (nib >> 2) & 3;
  uint32_t pos;
  if (tf32) pos = idx0 / 2;
  else if (int4) pos = 2 * (k < 2 ? idx0 : idx1) + k % 2;
  else pos = k == 0 ? idx0 : idx1;
  return chunk * chunk_elems + pos;
}


// An element type of tcgen05.mma's A and B (9.7.18.10.4): how many bits it
// takes in shared memory and in a Tensor Memory container, and its value.
// fp8/fp6/fp4 under .kind::f8f6f4 and .kind::mxf8f6f4 sit 16 to a 16-byte
// group in shared memory, the 6- and 4-bit ones padded, and in 8-bit
// containers in Tensor Memory -- fp6 in bits 0-5, fp4 in bits 2-5; under
// .kind::mxf4/mxf4nvf4, fp4 is packed two to a byte in both.
enum class TcType { F16, BF16, TF32, E4M3, E5M2, E2M3, E3M2, E2M1, S8, U8, TI16 };
struct TcElem {
  TcType t = TcType::F16;
  uint32_t bits = 16;          // the element's own width
  uint32_t per16 = 8;          // elements in a 16-byte group of shared memory
  uint32_t cbits = 16;         // its container in Tensor Memory
};
double small_float(uint32_t v, int ebits, int mbits, int bias) {
  const uint32_t mant = v & ((1u << mbits) - 1);
  const uint32_t exp = (v >> mbits) & ((1u << ebits) - 1);
  const bool neg = (v >> (ebits + mbits)) & 1;
  const double mag = exp ? std::ldexp(1.0 + mant / double(1u << mbits), int(exp) - bias)
                         : std::ldexp(mant / double(1u << mbits), 1 - bias);
  return neg ? -mag : mag;
}
double tc_decode(TcType t, uint32_t raw) {
  switch (t) {
    case TcType::F16: return f16_to_double(raw & 0xFFFF);
    case TcType::BF16: return bf16_to_double(raw & 0xFFFF);
    // As for wgmma: the low 13 mantissa bits of a tf32 are not read.
    case TcType::TF32: return static_cast<double>(f32(raw & 0xFFFFE000u));
    case TcType::E4M3: return fp8_to_double(raw & 0xFF, kE4M3);
    case TcType::E5M2: return fp8_to_double(raw & 0xFF, kE5M2);
    // The OCP MX formats: no infinities or NaNs.
    case TcType::E2M3: return small_float(raw & 0x3F, 2, 3, 1);
    case TcType::E3M2: return small_float(raw & 0x3F, 3, 2, 3);
    case TcType::E2M1: return small_float(raw & 0xF, 2, 1, 1);
    case TcType::S8: return static_cast<double>(static_cast<int8_t>(raw & 0xFF));
    case TcType::U8: return static_cast<double>(raw & 0xFF);
    // s1z4m11 (PTX ISA 9.4): a sign, four bits that are zero and eleven of
    // magnitude -- the range -2047 to 2047 the ISA gives, so sign and magnitude.
    case TcType::TI16: return (raw & 0x8000) ? -static_cast<double>(raw & 0x7FF) : static_cast<double>(raw & 0x7FF);
  }
  return 0.0;
}


// spcompress's choice (PTX ISA 9.7.10.30): the two of four elements a selection keeps, as
// indices in ascending order. sel: 0 MAX, 1 MAXABS, 2 MIN, 3 MINABS. A NaN is always kept; -0.0 is
// below +0.0; where the ISA leaves ties to the implementation, the lower index goes first.
void sp24_pick(const double v[4], uint32_t sel, uint32_t pick[2]) {
  bool nan[4];
  for (uint32_t j = 0; j < 4; ++j) nan[j] = std::isnan(v[j]);
  const auto better = [&](uint32_t a, uint32_t b) {   // is a strictly better than b
    if (nan[a] != nan[b]) return nan[a];
    if (nan[a]) return false;
    double x = v[a], y = v[b];
    if (sel == 1 || sel == 3) { x = std::fabs(x); y = std::fabs(y); }
    if (x == y && (sel == 0 || sel == 2)) {
      const bool sx = std::signbit(x), sy = std::signbit(y);
      if (sx != sy) return sel == 0 ? sy : sx;
    }
    return (sel == 0 || sel == 1) ? x > y : x < y;
  };
  uint32_t order[4] = {0, 1, 2, 3};
  std::stable_sort(order, order + 4, [&](uint32_t a, uint32_t b) { return better(a, b); });
  pick[0] = std::min(order[0], order[1]);
  pick[1] = std::max(order[0], order[1]);
}

class Interpreter {
 public:
  Interpreter(const EntryFn& fn, const LaunchConfig& cfg, const ParamBuffer& params, MemoryManager& mem,
              const DeviceProfile& profile, const SymbolTable* symbols, LaunchStats& stats,
              const ProgressFn& progress)
      // Listed in declaration order, which is the order they are actually
      // initialised in. W_ and all_ sit between profile_ and symbols_, and
      // writing them last read correctly only because both come from
      // `profile` rather than from each other -- the day one is written as
      // all_(all_lanes(W_)) that stops being true, silently.
      : fn_(fn), cfg_(cfg), params_(params), mem_(mem), profile_(profile),
        W_(profile.warp_size), all_(all_lanes(profile.warp_size)), symbols_(symbols),
        stats_(stats), progress_(progress) {
    if (progress_) last_progress_ = std::chrono::steady_clock::now();
    // A host program that set its own rounding mode before launching gets
    // conversions that follow it, as before; see g_directed_rounding.
    host_directed_ = std::fegetround() != FE_TONEAREST;
    fast_enabled_ = g_fast_path.load(std::memory_order_relaxed);
    if (host_directed_) g_directed_rounding.fetch_add(1, std::memory_order_relaxed);
  }
  ~Interpreter() {
    if (host_directed_) g_directed_rounding.fetch_sub(1, std::memory_order_relaxed);
  }
  Interpreter(const Interpreter&) = delete;
  Interpreter& operator=(const Interpreter&) = delete;

  // The cluster shape this launch runs with, never zero in any dimension. A
  // launch that names no cluster is a launch of 1x1x1 clusters -- one block
  // each -- which is what PTX says the cluster registers report there.
  std::array<uint32_t, 3> cluster_shape() const {
    std::array<uint32_t, 3> c = cfg_.cluster;
    for (int i = 0; i < 3; ++i)
      if (c[i] == 0) c[i] = 1;
    return c;
  }
  bool has_explicit_cluster() const {
    return cfg_.cluster[0] > 1 || cfg_.cluster[1] > 1 || cfg_.cluster[2] > 1;
  }

  void set_concurrent(bool v) { concurrent_ = v; }
  // Which of the launch's host threads this is, for %smid.
  void set_worker(unsigned t) { worker_ = t; }
  void set_device_launches(DeviceLaunches* dl) { dl_ = dl; }
  void set_grid_id(uint64_t v) { grid_id_ = v; }

  void run_grid() {
    const uint64_t total = uint64_t{cfg_.grid[0]} * cfg_.grid[1] * cfg_.grid[2];
    if (cfg_.cooperative) run_grid_cooperative(total);
    else run_units(0, work_units());
  }

  // What a launch divides among host threads: whole clusters when the launch
  // has them -- a cluster's blocks must be resident together -- and blocks
  // otherwise.
  uint64_t work_units() const {
    if (!has_explicit_cluster()) return uint64_t{cfg_.grid[0]} * cfg_.grid[1] * cfg_.grid[2];
    const auto c = cluster_shape();
    return uint64_t{cfg_.grid[0] / c[0]} * (cfg_.grid[1] / c[1]) * (cfg_.grid[2] / c[2]);
  }
  void run_units(uint64_t first, uint64_t last) {
    if (has_explicit_cluster()) run_cluster_range(first, last);
    else run_block_range(first, last);
  }

  // Block (x, y, z) of cluster `k`, rank `r`: ranks run x fastest, as
  // %cluster_ctarank numbers them.
  std::array<uint32_t, 3> cluster_member(uint64_t k, uint32_t r) const {
    const auto c = cluster_shape();
    const uint64_t ncx = cfg_.grid[0] / c[0], ncy = cfg_.grid[1] / c[1];
    const uint64_t kx = k % ncx, ky = (k / ncx) % ncy, kz = k / (ncx * ncy);
    return {static_cast<uint32_t>(kx * c[0] + r % c[0]),
            static_cast<uint32_t>(ky * c[1] + (r / c[0]) % c[1]),
            static_cast<uint32_t>(kz * c[2] + r / (c[0] * c[1]))};
  }

  // Clusters [first, last), each with its blocks resident at once and taking
  // bounded turns, like a cooperative grid in miniature.
  void run_cluster_range(uint64_t first, uint64_t last) {
    const auto c = cluster_shape();
    if (cfg_.grid[0] % c[0] || cfg_.grid[1] % c[1] || cfg_.grid[2] % c[2])
      throw Error::make(Err::LaunchConfig, "the grid (", cfg_.grid[0], ",", cfg_.grid[1], ",",
                        cfg_.grid[2], ") is not a whole number of ", c[0], "x", c[1], "x", c[2],
                        " clusters");
    const uint32_t size = c[0] * c[1] * c[2];
    // Clusters are taken from the front of the range, so one that a running
    // cluster has cancelled (clusterlaunchcontrol.try_cancel) never starts.
    clc_next_ = first;
    clc_last_ = last;
    clc_clusters_ = true;
    while (clc_next_ < clc_last_) {
      const uint64_t k = clc_next_++;
      std::vector<BlockState> blocks(size);
      std::vector<std::unique_ptr<Scheduler>> scheds;   // per block, as for a cooperative grid
      for (uint32_t r = 0; r < size; ++r)
        scheds.push_back(make_scheduler(cfg_.scheduler, cfg_.scheduler_seed + (k * size + r) * 0x9E3779B97F4A7C15ull));
      ClusterState cs;
      cs.ranks.assign(size, nullptr);
      for (uint32_t r = 0; r < size; ++r) {
        BlockState& b = blocks[r];
        b.ctx.ctaid = cluster_member(k, r);
        b.ctx.ntid = cfg_.block;
        b.ctx.nctaid = cfg_.grid;
        b.ctx.cluster = c;
        b.ctx.explicit_cluster = true;
        setup_block(b);
        b.ctx.cluster_state = &cs;
        cs.blocks.push_back(&b.warps);
        cs.ranks[r] = &b.ctx;
      }
      constexpr uint64_t kSlice = 256;
      size_t live = blocks.size();
      while (live) {
        live = 0;
        for (uint32_t r = 0; r < size; ++r) {
          BlockState& b = blocks[r];
          if (b.done) continue;
          if (step_block(b, *scheds[r], kSlice)) ++live;
          else ++stats_.blocks;
        }
      }
    }
    clc_next_ = clc_last_ = 0;
  }

  // A cooperative launch: every block resident at once, interleaved.
  //
  // `cg::this_grid().sync()` is not an instruction. It compiles to an atomic
  // increment of a counter in global memory and then a spin on that counter --
  // so it only terminates if the blocks that have not arrived yet are still
  // able to run. Running blocks one at a time, which is what the ordinary
  // scheduler does and what the programming model permits, deadlocks on the
  // first block to arrive.
  //
  // So every block is set up before any of them runs, and each gets a bounded
  // turn in round-robin order. Deterministic, and the same order every time:
  // the point of this simulator is that a run is reproducible, which rules out
  // handing the blocks to OS threads and letting them race.
  void run_grid_cooperative(uint64_t total) {
    const uint64_t gx = cfg_.grid[0], gy = cfg_.grid[1];
    std::vector<BlockState> blocks(static_cast<size_t>(total));
    // A scheduler per block. One shared by all of them handed the blocks
    // alternate warp indices -- with two blocks of four warps, round-robin gave
    // block 0 only warps 0 and 2 -- and a barrier that needs every warp of a
    // block never completed.
    std::vector<std::unique_ptr<Scheduler>> scheds;
    for (uint64_t i = 0; i < total; ++i)
      scheds.push_back(make_scheduler(cfg_.scheduler, cfg_.scheduler_seed + i * 0x9E3779B97F4A7C15ull));
    for (uint64_t i = 0; i < total; ++i) {
      BlockState& b = blocks[static_cast<size_t>(i)];
      b.ctx.ctaid = {static_cast<uint32_t>(i % gx), static_cast<uint32_t>((i / gx) % gy),
                     static_cast<uint32_t>(i / (gx * gy))};
      b.ctx.ntid = cfg_.block;
      b.ctx.nctaid = cfg_.grid;
      b.ctx.cluster = cluster_shape();
      b.ctx.explicit_cluster = has_explicit_cluster();
      setup_block(b);
    }
    // Every block is resident, so each cluster's barrier sees all its blocks.
    const auto c = cluster_shape();
    const uint64_t ncx = gx / c[0], ncy = gy / c[1];
    std::vector<ClusterState> clusters(static_cast<size_t>(work_units()));
    for (BlockState& b : blocks) {
      const uint64_t k = (b.ctx.ctaid[0] / c[0]) + (b.ctx.ctaid[1] / c[1]) * ncx +
                         uint64_t{b.ctx.ctaid[2] / c[2]} * ncx * ncy;
      if (k < clusters.size()) {
        ClusterState& cs = clusters[static_cast<size_t>(k)];
        b.ctx.cluster_state = &cs;
        cs.blocks.push_back(&b.warps);
        if (cs.ranks.empty()) cs.ranks.assign(size_t{c[0]} * c[1] * c[2], nullptr);
        cs.ranks[cluster_rank_of(b.ctx)] = &b.ctx;
      }
    }
    // One warp-turn per block per round. Long enough that a block making real
    // progress is not paying scheduler overhead per instruction, short enough
    // that a spinning warp hands the grid back promptly.
    constexpr uint64_t kSlice = 256;
    size_t live = blocks.size();
    while (live) {
      live = 0;
      for (size_t i = 0; i < blocks.size(); ++i) {
        BlockState& b = blocks[i];
        if (b.done) continue;
        if (step_block(b, *scheds[i], kSlice)) ++live;
        else ++stats_.blocks;
      }
    }
  }

  // Runs the blocks with linear indices [first, last). CUDA blocks are
  // independent -- that is the programming model's central promise -- so a
  // range can run on its own thread with nothing shared but device memory.
  void run_block_range(uint64_t first, uint64_t last) {
    auto sched = make_scheduler(cfg_.scheduler, cfg_.scheduler_seed);
    const uint64_t gx = cfg_.grid[0], gy = cfg_.grid[1];
    clc_next_ = first;
    clc_last_ = last;
    clc_clusters_ = false;
    while (clc_next_ < clc_last_) {
      const uint64_t i = clc_next_++;
      BlockCtx ctx;
      ctx.ctaid = {static_cast<uint32_t>(i % gx), static_cast<uint32_t>((i / gx) % gy),
                   static_cast<uint32_t>(i / (gx * gy))};
      ctx.ntid = cfg_.block;
      ctx.nctaid = cfg_.grid;
      ctx.cluster = cluster_shape();
      ctx.explicit_cluster = has_explicit_cluster();
      run_block(ctx, *sched);
      ++stats_.blocks;
    }
    clc_next_ = clc_last_ = 0;
  }

 private:
  DeviceLaunches* dl_ = nullptr;

  // The units (clusters, or blocks when there are none) of the range this
  // interpreter is running that have not started yet: [clc_next_, clc_last_).
  // clusterlaunchcontrol.try_cancel takes the next of them, which then never
  // launches, and gives its first CTA to the canceller. Empty outside those
  // two loops -- a cooperative launch has every block running already -- so a
  // cancellation there fails, as it does on a device with nothing pending.
  uint64_t clc_next_ = 0, clc_last_ = 0;
  bool clc_clusters_ = false;

  std::optional<std::array<uint32_t, 3>> clc_take() {
    if (clc_next_ >= clc_last_) return std::nullopt;
    const uint64_t u = clc_next_++;
    if (clc_clusters_) return cluster_member(u, 0);
    const uint64_t gx = cfg_.grid[0], gy = cfg_.grid[1];
    return std::array<uint32_t, 3>{static_cast<uint32_t>(u % gx), static_cast<uint32_t>((u / gx) % gy),
                                   static_cast<uint32_t>(u / (gx * gy))};
  }

  // Re-throws a lower-level error with kernel/instruction context attached.
  [[noreturn]] void rethrow_with_context(const Error& e, const Instr& ins, int lane) {
    throw Error::make(e.code(), e.message(), "\n  in ", cur_ == &fn_ ? "kernel '" : "device function '",
                      cur_->name, "', PTX line ", ins.line,
                      lane >= 0 ? "\n  lane " + std::to_string(lane) : "",
                      "\n  instruction: ", ins.text.empty() ? "?" : ins.text,
                      "\n  GPU profile: ", profile_.id);
  }

  // Where every warp of the block is, for a hang: the warp that ran out of
  // steps is usually spinning in a wait, and which warp should have released
  // it -- and what that one is doing instead -- is the rest of the diagnosis.
  // Warps in the same place are listed together.
  std::string warp_positions(const BlockCtx& ctx) const {
    if (!ctx.warps) return {};
    auto where = [&](const Warp& w) {
      if (w.state == Warp::State::Done) return std::string("exited");
      std::string out = w.state == Warp::State::AtBarrier ? "at a barrier" : "running";
      const EntryFn& f = w.fn ? *w.fn : fn_;
      if (w.fn) out += " in device function '" + f.name + "'";
      for (const Path& p : w.paths) {
        out += p.pc < f.body.size()
                   ? "; PTX line " + std::to_string(f.body[p.pc].line) + ": " +
                         f.body[p.pc].text.substr(0, 100)
                   : std::string("; past the end");
        if (p.parked) out += " (parked)";
      }
      return out;
    };
    std::string out = "\n  warps of this block:";
    const auto& warps = *ctx.warps;
    for (size_t i = 0; i < warps.size();) {
      const std::string here = where(warps[i]);
      size_t j = i + 1;
      while (j < warps.size() && where(warps[j]) == here) ++j;
      out += "\n    " + (j - i == 1 ? "warp " + std::to_string(i)
                                     : "warps " + std::to_string(i) + "-" + std::to_string(j - 1)) +
             ": " + here;
      i = j;
    }
    // The block's mbarriers whose current phase is still open, by shared
    // offset: a wait that never ends is usually on one of these.
    if (ctx.mbar) {
      std::vector<std::pair<uint64_t, const Mbarrier*>> open;
      for (const auto& [addr, b] : ctx.mbar->bars)
        if (b.valid && (b.arrived || b.tx || !b.pending.empty())) open.emplace_back(addr, &b);
      std::sort(open.begin(), open.end());
      if (!open.empty()) out += "\n  mbarriers with an open phase (shared offset: phase, arrivals, bytes due):";
      for (size_t i = 0; i < open.size() && i < 16; ++i) {
        const Mbarrier& b = *open[i].second;
        char line[160];
        std::snprintf(line, sizeof line, "\n    0x%llx: phase %u, %llu of %llu arrived, %lld bytes due%s",
                      static_cast<unsigned long long>(open[i].first & 0xFFFFFF), b.phase,
                      static_cast<unsigned long long>(b.arrived), static_cast<unsigned long long>(b.expected),
                      static_cast<long long>(b.tx), b.pending.empty() ? "" : ", copies not yet landed");
        out += line;
      }
    }
    return out;
  }

  [[noreturn]] void ctx_fail(const Instr& ins, int lane, Err code, const std::string& msg) {
    try {
      throw Error::make(code, msg);
    } catch (const Error& e) {
      rethrow_with_context(e, ins, lane);
    }
  }

  // Everything a block owns while it is resident. An ordinary launch keeps one
  // of these on the stack and runs it to completion; a cooperative launch keeps
  // one per block and interleaves them, which is the whole difference between
  // the two scheduling modes.
  struct BlockState {
    BlockCtx ctx;
    std::vector<uint8_t> shared;
    BarrierReduction bar_red;
    NamedBarriers bars;
    MbarrierTable mbar;
    WgmmaSnapshots wgmma;
    TensorMemory tmem;
    SharedShadow shadow;
    std::vector<Warp> warps;
    uint64_t clock = 0;
    bool done = false;
    bool exited = false;   // every warp Done; see BlockCtx::exited
  };

  // The kernel's own shared memory as the hardware allocates it: static
  // declarations, then the dynamic bytes, rounded up to the allocation unit
  // (128 bytes from compute capability 8.0, 256 before; cuda_occupancy.h).
  // An RTX 3060 reports this as %total_smem_size -- 128 for a kernel with one
  // byte -- and puts its reserved shared memory straight after it.
  uint32_t kernel_shared_bytes() const {
    const uint64_t used = uint64_t{std::max(fn_.static_shared_size, fn_.dynamic_shared_offset)} +
                          cfg_.shared_bytes;
    const uint64_t unit = profile_.cc_major >= 8 ? 128 : 256;
    return static_cast<uint32_t>((used + unit - 1) / unit * unit);
  }
  // The driver's reserved shared memory per block: 1 KiB from Ampere on
  // (cudaDevAttrReservedSharedMemoryPerBlock; cuda_occupancy.h adds it to
  // every block from compute capability 8.0), none before. The profile holds
  // the rule, so the driver, the runtime and this agree.
  uint32_t reserved_smem_bytes() const { return profile_.reserved_smem_per_block(); }
  static constexpr uint32_t kReservedSmemUsed = 0x120;
  uint32_t reserved_smem_base() const {
    if (reserved_smem_bytes() == 0)
      throw Error::make(Err::UnsupportedPtx, "%reserved_smem_offset_* on ", profile_.id,
                        ": reserved shared memory exists from compute capability 8.0");
    return kernel_shared_bytes();
  }
  // Whether the kernel or any device function in its module reads a
  // %reserved_smem_offset_* register. Only then is the reserved region
  // backed: every other kernel keeps its shared window at exactly what it
  // declared, so an overrun past it is still caught rather than landing in
  // bytes no kernel of its own should touch.
  bool module_reads_reserved_smem() const {
    if (fn_.reads_reserved_smem) return true;
    for (const auto& f : fn_.module_funcs)
      if (f && f->reads_reserved_smem) return true;
    return false;
  }

  void setup_block(BlockState& b) {
    // Fresh, zeroed shared memory per block (static declarations + the
    // launch's dynamic bytes), and the reserved region behind them when the
    // kernel asks where it is.
    if (module_reads_reserved_smem() && reserved_smem_bytes())
      b.shared.assign(size_t{kernel_shared_bytes()} + reserved_smem_bytes(), 0);
    else
      b.shared.assign(std::max(fn_.static_shared_size, fn_.dynamic_shared_offset) + cfg_.shared_bytes, 0);
    b.ctx.shared = &b.shared;
    b.ctx.bar_red = &b.bar_red;
    b.ctx.mbar = &b.mbar;
    b.wgmma.pending.clear();
    b.ctx.wgmma = &b.wgmma;
    b.tmem = TensorMemory{};
    b.ctx.tmem = &b.tmem;
    b.ctx.bars = &b.bars;
    b.ctx.warps = &b.warps;
    b.ctx.clock = &b.clock;
    b.ctx.exited = &b.exited;
    if (detect_races() && !b.shared.empty()) {
      b.shadow.words.assign(b.shared.size() / 4 + 1, WordShadow{});
      b.ctx.shadow = &b.shadow;
    }
    const uint64_t total = uint64_t{b.ctx.ntid[0]} * b.ctx.ntid[1] * b.ctx.ntid[2];
    const size_t nwarps = static_cast<size_t>((total + W_ - 1) / W_);
    // resize, not assign: a Warp owns a unique_ptr and so is move-only.
    b.warps.clear();
    b.warps.resize(nwarps);
    for (size_t w = 0; w < nwarps; ++w) {
      Warp& warp = b.warps[w];
      warp.regs32.clear();
      warp.regs32.resize(fn_.num_regs32);
      warp.regs64.clear();
      warp.regs64.resize(fn_.num_regs64);
      warp.preds.assign(fn_.num_regs32 + fn_.num_regs64, 0);
      warp.written32.assign(fn_.num_regs32, 0);
      warp.written64.assign(fn_.num_regs64, 0);
      Mask live = 0;
      for (uint32_t lane = 0; lane < W_; ++lane) {
        uint64_t lin = uint64_t{static_cast<uint32_t>(w)} * W_ + lane;
        if (lin >= total) break;
        live |= (Mask{1} << lane);
        warp.tid_x[lane] = static_cast<uint32_t>(lin % b.ctx.ntid[0]);
        warp.tid_y[lane] = static_cast<uint32_t>((lin / b.ctx.ntid[0]) % b.ctx.ntid[1]);
        warp.tid_z[lane] = static_cast<uint32_t>(lin / (uint64_t{b.ctx.ntid[0]} * b.ctx.ntid[1]));
      }
      if (live) warp.paths.push_back({0, live});
      else warp.state = Warp::State::Done;
    }
    stats_.warps += nwarps;
  }

  // Gives the block one turn: picks a runnable warp and runs it, or releases a
  // barrier when every warp has arrived. Returns false once the block is
  // finished.
  //
  // `slice` bounds how many instructions the chosen warp may execute before
  // returning here. An ordinary launch leaves it unbounded, because nothing
  // else is waiting. A cooperative launch must bound it: a grid barrier is a
  // spin on a global flag another block has to set, and a warp spinning without
  // a bound would never give that block a turn.
  bool step_block(BlockState& b, Scheduler& sched, uint64_t slice) {
    std::vector<size_t> runnable;
    for (size_t i = 0; i < b.warps.size(); ++i)
      if (b.warps[i].state == Warp::State::Ready) runnable.push_back(i);
    if (runnable.empty()) {
      // Every warp that has not exited is waiting. Those at a whole-CTA
      // barrier (bar.sync without a count, bar.red) have all arrived, so it
      // completes; those at a counted barrier are released by the arrivals
      // that complete it, and cannot be released here.
      bool any_waiting = false, any_counted = false;
      for (auto& w : b.warps)
        if (w.state == Warp::State::AtBarrier) {
          if (w.counted_barrier) {
            any_counted = true;
            continue;
          }
          w.state = Warp::State::Ready;
          any_waiting = true;
        }
      if (!any_waiting && any_counted) {
        // Nothing can run and nothing more can arrive: a hang on hardware.
        std::string where;
        for (size_t i = 0; i < b.bars.bar.size(); ++i)
          if (b.bars.bar[i].waiting)
            where += "\n  barrier " + std::to_string(i) + ": " +
                     std::to_string(__builtin_popcountll(b.bars.bar[i].waiting)) + " warp(s) waiting, " +
                     std::to_string(b.bars.bar[i].arrived) + " thread(s) arrived";
        throw Error::make(Err::LaunchConfig,
                          "deadlock: every warp of the block that has not exited is waiting at a "
                          "barrier whose thread count can no longer be reached", where);
      }
      // Every warp has now arrived, so a bar.red round in flight has its
      // answer: set it aside and start the next.
      if (any_waiting && b.bar_red.arrived) {
        b.bar_red.result = b.bar_red.acc;
        b.bar_red.arrived = 0;
        ++b.bar_red.round;
      }
      // ...and the barrier they arrived at orders everything before it
      // against everything after, which is what ends the epoch.
      if (any_waiting && b.ctx.shadow) ++b.ctx.shadow->epoch;
      if (!any_waiting) {
        // "All of the Tensor Memory that was allocated ... must be explicitly
        // deallocated using tcgen05.dealloc before the kernel exits."
        if (!b.tmem.live.empty()) {
          uint32_t cols = 0;
          for (const auto& [c, n] : b.tmem.live) cols += n;
          throw Error::make(Err::LaunchConfig, "block (", b.ctx.ctaid[0], ",", b.ctx.ctaid[1], ",",
                            b.ctx.ctaid[2], ") exited with ", cols,
                            " Tensor Memory columns still allocated; every tcgen05.alloc needs a "
                            "tcgen05.dealloc before the kernel exits");
        }
        b.done = true;
        return false;  // all Done
      }
      return true;
    }
    const size_t picked = sched.pick(runnable);
    cur_warp_ = static_cast<uint32_t>(picked);
    // Two things can bound the turn: the scheduler, which preempts to
    // interleave, and a cooperative launch, which preempts so a spinning warp
    // lets another block run. Whichever is shorter wins; zero from either means
    // "no bound of mine".
    const uint64_t sched_slice = sched.slice();
    uint64_t turn = slice;
    if (sched_slice && (!turn || sched_slice < turn)) turn = sched_slice;
    run_warp_until_yield(b.warps[picked], b.ctx, turn);
    if (b.warps[picked].state == Warp::State::Done) {
      bool all = true;
      for (const Warp& w : b.warps) all = all && w.state == Warp::State::Done;
      b.exited = all;
    }
    return true;
  }

  void run_block(BlockCtx& ctx, Scheduler& sched) {
    BlockState b;
    b.ctx = ctx;
    setup_block(b);
    ClusterState solo;   // a block with no explicit cluster is a cluster of one
    solo.blocks.push_back(&b.warps);
    solo.ranks.push_back(&b.ctx);
    b.ctx.cluster_state = &solo;
    while (step_block(b, sched, /*slice=*/0)) {
    }
  }

  // Merges paths sitting at the same pc and returns the index of the one with
  // the lowest pc, which is the path that runs next.
  size_t select_path(Warp& w) {
    if (w.paths.size() == 1) {
      w.paths[0].parked = Path::kRunnable;   // alone: nothing left to wait for
      w.paths[0].issued_at = w.steps;
      return 0;
    }
    for (size_t i = 0; i < w.paths.size(); ++i) {
      for (size_t j = w.paths.size(); j-- > i + 1;) {
        if (w.paths[j].pc == w.paths[i].pc) {
          w.paths[i].mask |= w.paths[j].mask;
          // Merged paths run again: at a barrier, that is the arrival that
          // may now find every lane there.
          w.paths[i].parked = Path::kRunnable;
          w.paths.erase(w.paths.begin() + static_cast<long>(j));
        }
      }
    }
    // Lowest pc first, among the paths not waiting at a barrier.
    size_t best = w.paths.size();
    for (size_t i = 0; i < w.paths.size(); ++i)
      if (!w.paths[i].parked && (best == w.paths.size() || w.paths[i].pc < w.paths[best].pc))
        best = i;
    if (best == w.paths.size()) best = 0;   // cannot happen: bar.sync keeps one runnable
    // Lowest pc first is what reconverges a warp promptly, but on its own it
    // can starve a path for good: lanes spinning on a lock loop back to a
    // lower pc than the lane holding it, which then never reaches its
    // release. Since Volta every thread makes progress regardless of its
    // warp-mates, so a path that has not issued for kStarvedSteps gets
    // kBoostSteps instructions in a row. Paths that reconverge normally never
    // wait that long, so ordinary code runs in the same order as before.
    constexpr uint64_t kStarvedSteps = 1024;
    constexpr uint32_t kBoostSteps = 256;
    size_t pick = w.paths.size();
    for (size_t i = 0; i < w.paths.size() && pick == w.paths.size(); ++i)
      if (!w.paths[i].parked && w.paths[i].boost) pick = i;
    if (pick == w.paths.size()) {
      for (size_t i = 0; i < w.paths.size(); ++i)
        if (!w.paths[i].parked && w.steps - w.paths[i].issued_at > kStarvedSteps &&
            (pick == w.paths.size() || w.paths[i].issued_at < w.paths[pick].issued_at))
          pick = i;
      if (pick != w.paths.size()) w.paths[pick].boost = kBoostSteps;
    }
    if (pick != w.paths.size()) {
      best = pick;
      --w.paths[best].boost;
    }
    w.paths[best].issued_at = w.steps;
    return best;
  }

  // Runs one warp until it yields: a barrier, a return, or -- when `slice` is
  // non-zero -- that many instructions. Zero means no bound.
  void run_warp_until_yield(Warp& w, const BlockCtx& ctx, uint64_t slice = 0) {
    cur_ = w.fn ? w.fn : &fn_;
    local_base_ = w.local_base;
    uint64_t issued = 0;
    turn_start_ = w.steps;
    for (Path& p : w.paths)
      if (p.parked == Path::kThisTurn) p.parked = Path::kRunnable;
    while (w.state == Warp::State::Ready) {
      if (slice && issued++ >= slice) return;
      if (w.paths.empty()) {
        w.state = Warp::State::Done;
        return;
      }
      if (w.yield_now) {
        w.yield_now = false;
        return;
      }
      size_t idx = select_path(w);
      if (w.paths[idx].pc >= cur_->body.size()) {
        exec_ret(w, ctx, implicit_ret(), idx, w.paths[idx].mask);
        continue;
      }
      const Instr& ins = cur_->body[w.paths[idx].pc];
      if (progress_ && (stats_.instructions & 0xFFFFF) == 0) report_progress();
      // Warp-level issue count, plus the per-lane total: their ratio is the
      // average lane utilisation, which is divergence measured directly.
      const uint64_t lanes = static_cast<uint64_t>(popcount_mask(w.paths[idx].mask));
      stats_.thread_instructions += lanes;
      // Counted against the same mask as thread_instructions, so the classes
      // partition it exactly. Predication narrows the mask inside step(); it is
      // deliberately not subtracted here, for the same reason.
      const InstClass cls = class_of_pc(w.paths[idx].pc);
      stats_.inst_by_class[static_cast<size_t>(cls)] += lanes;
      if (cls == InstClass::Tensor) ++stats_.tensor_instructions;
      if (ctx.clock) ++*ctx.clock;
      // Per-opcode issue count, alongside the class histogram. Counted per
      // warp-level issue like `instructions`, so the two are comparable.
      if (ins.opcode_id) {
        if (ins.opcode_id >= stats_.inst_by_opcode.size())
          stats_.inst_by_opcode.resize(ins.opcode_id + 1u, 0);
        ++stats_.inst_by_opcode[ins.opcode_id];
      }
      ++stats_.instructions;
      // Named with the instruction the warp is on: in a hang, that is almost
      // always the wait it is spinning in, which is the whole diagnosis.
      if (++w.steps > cfg_.max_steps)
        ctx_fail(ins, -1, Err::ExecLimit,
                 "exceeded the step budget (" + std::to_string(cfg_.max_steps) +
                     " instructions in one warp) — possible infinite loop; block (" +
                     std::to_string(ctx.ctaid[0]) + "," + std::to_string(ctx.ctaid[1]) + "," +
                     std::to_string(ctx.ctaid[2]) + "), warp " + std::to_string(cur_warp_) +
                     warp_positions(ctx));
      // The fast-path kinds go straight to their handler: no trip through
      // step() and dispatch(), whose frames cost more than the arithmetic.
      // A form the fast path declines falls through to step() as before.
      if (fast_enabled_ && fast_kind(ins)) {
        Mask fm = w.paths[idx].mask;
        if (ins.has_pred) {
          Mask p = read_pred(w, ins, ins.pred);
          if (ins.pred_negated) p = ~p;
          fm &= p;
        }
        if (fm == 0 || fast_path(w, ctx, ins, fm)) {
          ++w.paths[idx].pc;
          continue;
        }
      }
      step(w, ctx, idx, ins);
    }
  }

  static bool fast_kind(const Instr& ins) {
    switch (ins.op.index()) {
      case op_index<OpIntBin>():
      case op_index<OpMadLo>():
      case op_index<OpMov>():
      case op_index<OpSetp>():
      case op_index<OpFma>():
      case op_index<OpFloatBin>():
      case op_index<OpF16x2Fma>():
      case op_index<OpCvt>():
      case op_index<OpMulWide>():
        return true;
      default:
        return false;
    }
  }

  InstClass class_of_pc(size_t pc) const {
    // The cache is per kernel. Inside a device function the pc indexes a
    // different body, so classify directly rather than reading another
    // function's entry.
    if (cur_ != &fn_) return classify(cur_->body[pc]);
    if (class_by_pc_.empty()) {
      class_by_pc_.resize(fn_.body.size());
      for (size_t i = 0; i < fn_.body.size(); ++i)
        class_by_pc_[i] = static_cast<uint8_t>(classify(fn_.body[i]));
    }
    return static_cast<InstClass>(class_by_pc_[pc]);
  }

  // Which category an instruction falls in. Worked out once per kernel rather
  // than per execution: a kernel is parsed once and its instructions run
  // millions of times, and a chain of variant tests on that path would cost
  // more than the counting is worth.
  static InstClass classify(const Instr& ins) {
    auto by_width = [](const Type& t) {
      if (t.bits == 16) return InstClass::Fp16;
      if (t.bits == 64) return InstClass::Fp64;
      return InstClass::Fp32;
    };
    // Arithmetic that carries a type: float by width, otherwise integer.
    if (const auto* o = std::get_if<OpFloatBin>(&ins.op)) return by_width(o->ty);
    if (const auto* o = std::get_if<OpFma>(&ins.op)) return by_width(o->ty);
    if (const auto* o = std::get_if<OpMath>(&ins.op)) return by_width(o->ty);
    if (const auto* o = std::get_if<OpCopysign>(&ins.op)) return by_width(o->ty);
    if (std::holds_alternative<OpF16x2Bin>(ins.op) ||
        std::holds_alternative<OpF16x2Fma>(ins.op) ||
        std::holds_alternative<OpF16x2Neg>(ins.op))
      return InstClass::Fp16;
    if (std::holds_alternative<OpF32x2>(ins.op)) return InstClass::Fp32;
    for (const Type* t : {std::get_if<OpNeg>(&ins.op) ? &std::get_if<OpNeg>(&ins.op)->ty : nullptr,
                          std::get_if<OpAbs>(&ins.op) ? &std::get_if<OpAbs>(&ins.op)->ty : nullptr})
      if (t) return t->is_real() ? by_width(*t) : InstClass::Integer;

    if (std::holds_alternative<OpIntBin>(ins.op) || std::holds_alternative<OpMadLo>(ins.op) ||
        std::holds_alternative<OpMulWide>(ins.op) || std::holds_alternative<OpMadWide>(ins.op) ||
        std::holds_alternative<OpMulHi>(ins.op) || std::holds_alternative<OpMadHi>(ins.op) ||
        std::holds_alternative<OpShf>(ins.op) || std::holds_alternative<OpBfe>(ins.op) ||
        std::holds_alternative<OpBfi>(ins.op) || std::holds_alternative<OpBrev>(ins.op) ||
        std::holds_alternative<OpPopcClz>(ins.op) || std::holds_alternative<OpPrmt>(ins.op) ||
        std::holds_alternative<OpDp4a>(ins.op) || std::holds_alternative<OpBmsk>(ins.op) ||
        std::holds_alternative<OpNot>(ins.op))
      return InstClass::Integer;

    if (std::holds_alternative<OpCvt>(ins.op) || std::holds_alternative<OpCvtF16x2>(ins.op) ||
        std::holds_alternative<OpCvtPack>(ins.op) ||
        std::holds_alternative<OpCvta>(ins.op) || std::holds_alternative<OpMovPack>(ins.op) ||
        std::holds_alternative<OpMovUnpack>(ins.op))
      return InstClass::BitConvert;

    if (std::holds_alternative<OpBra>(ins.op) || std::holds_alternative<OpBrx>(ins.op) ||
        std::holds_alternative<OpRet>(ins.op) ||
        std::holds_alternative<OpBar>(ins.op) || std::holds_alternative<OpBarRed>(ins.op) ||
        std::holds_alternative<OpTrap>(ins.op) || std::holds_alternative<OpCall>(ins.op))
      return InstClass::Control;

    if (std::holds_alternative<OpLd>(ins.op) || std::holds_alternative<OpSt>(ins.op) ||
        std::holds_alternative<OpAtom>(ins.op) || std::holds_alternative<OpCpAsync>(ins.op) ||
        std::holds_alternative<OpCpAsyncGroup>(ins.op) ||
        std::holds_alternative<OpBulkCopy>(ins.op) ||
        std::holds_alternative<OpLdMatrix>(ins.op) ||
        std::holds_alternative<OpStMatrix>(ins.op) ||
        std::holds_alternative<OpWmmaStore>(ins.op) ||
        std::holds_alternative<OpLdSlot>(ins.op) || std::holds_alternative<OpStSlot>(ins.op))
      return InstClass::Memory;

    if (std::holds_alternative<OpMma>(ins.op) || std::holds_alternative<OpWmmaMma>(ins.op) ||
        std::holds_alternative<OpWgmma>(ins.op) || std::holds_alternative<OpTcgen05>(ins.op) ||
        std::holds_alternative<OpMovMatrix>(ins.op))
      return InstClass::Tensor;

    return InstClass::Misc;
  }

  // ---- symbols / registers / operands ----

  uint64_t resolve_symbol(const BlockCtx& ctx, const Instr& ins, const std::string& name) {
    // PTX's one predefined constant: nvcc emits `mov.u32 %r, WARP_SZ` for
    // warpSize. It is the warp size of the device the kernel runs on.
    if (name == "WARP_SZ") return W_;
    // .local/.shared variables name an offset within their address space, not
    // a generic address; cvta converts when the kernel needs a generic pointer.
    // A .shared one is in this block's part of the cluster's window.
    if (auto it = cur_->locals.find(name); it != cur_->locals.end()) return local_base_ + it->second.offset;
    if (auto it = fn_.shared.find(name); it != fn_.shared.end())
      return cluster_address(ctx, cluster_rank_of(ctx), it->second.offset);
    if (symbols_) {
      if (auto it = symbols_->find(name); it != symbols_->end()) return it->second;
    }
    // A kernel may take a parameter's address rather than loading it by name:
    // CUB's segmented sort does "mov.b64 %rd, kernel_param_10" and then loads
    // through the register. Parameters live in their own window so those loads
    // resolve back to the parameter buffer.
    if (auto it = params_.layout.find(name); it != params_.layout.end())
      return kParamVaBase + it->second.first;
    // Taking the address of a *kernel* means one thing in device code: a
    // device-side launch. Saying so is worth a branch, because "unknown
    // symbol" sends you looking for a typo in a name that is right there in
    // the module.
    for (const std::string& k : fn_.module_entry_names)
      if (k == name)
        ctx_fail(ins, -1, Err::Unsupported,
                 "kernel '" + name +
                     "' had its address taken for a device-side launch (dynamic parallelism), "
                     "but this launch has no symbol table giving kernels addresses; the runtime "
                     "supplies one");
    ctx_fail(ins, -1, Err::NotFound,
             "unknown symbol '" + name + "' (not a .local depot, module .global variable, "
             "kernel parameter, or device function)");
  }

  // Returns a reference to the operand's lane vector. Register operands alias
  // the warp's register file directly; everything else is materialized into
  // `scratch`. Returning a reference keeps a 256-byte copy off the hot path.
  const Lanes& read_operand(Warp& w, const BlockCtx& ctx, const Instr& ins, const Operand& op,
                            Lanes& scratch) {
    if (const auto* r = std::get_if<RegOperand>(&op)) {
      // Reading a register that has not been written yet is undefined in PTX,
      // and compilers emit it on purpose. ggml's flash-attention kernels
      // unroll "pick the value belonging to my lane" into a chain of selp, each
      // step overwriting the accumulator only for the lane whose index matches;
      // the chain is seeded with a register nothing has written, and every
      // lane's copy of that seed is dead by the time the chain ends. NVIDIA's
      // own compute-sanitizer checks uninitialized *memory* and not registers,
      // for this reason.
      //
      // So an undefined register reads as zero here -- deterministically, which
      // is more than hardware promises -- and the diagnostic is kept where it
      // still means something: addr_base, read_pred and exec_st below refuse a
      // register nothing has written, because an address, a branch condition or
      // a stored value made of nothing is a bug in any program.
      if (r->reg.wide) {
        if (r->reg.id >= w.regs64.size() || !w.written64[r->reg.id]) {
          scratch.fill(0);
          return scratch;
        }
        return w.regs64[r->reg.id];
      }
      if (r->reg.id >= w.regs32.size() || !w.written32[r->reg.id]) {
        scratch.fill(0);
        return scratch;
      }
      return widen(w, w.regs32[r->reg.id]);
    }
    if (const auto* imm = std::get_if<ImmInt>(&op)) {
      scratch.fill(static_cast<uint64_t>(imm->value));
      return scratch;
    }
    if (const auto* immf = std::get_if<ImmFloatBits>(&op)) {
      scratch.fill(immf->bits);
      return scratch;
    }
    if (const auto* sym = std::get_if<SymbolOperand>(&op)) {
      scratch.fill(resolve_symbol(ctx, ins, sym->name));
      return scratch;
    }
    const auto& sr = std::get<SregOperand>(op);
    if (is_lanemask_sreg(sr.reg)) require_warp32(ins, "%lanemask_*");
    for (uint32_t lane = 0; lane < W_; ++lane)
      scratch[lane] = sreg_value(sr.reg, sr.index, w, ctx, lane);
    return scratch;
  }

  // PTX defines its warp-level primitives over a 32-lane warp, and defines
  // their masks as .b32: activemask, vote.ballot, shfl.sync's member mask and
  // the %lanemask_* registers all produce or consume 32 bits. There is no PTX
  // meaning for any of them at 64 lanes. Widening them would be inventing
  // semantics the ISA does not define, and truncating them to 32 would drop
  // the upper half of a wavefront silently -- the worse of the two, because
  // the kernel would run and be wrong. So they are refused by name, and the
  // message says which primitive and what the profile's width is.
  //
  // A CDNA wavefront has its own 64-bit equivalents. They belong to the AMD
  // front-end, which reads a different ISA; they are not these instructions
  // with a wider mask.
  void require_warp32(const Instr& ins, const char* what) {
    if (W_ != 32u)
      ctx_fail(ins, -1, Err::UnsupportedPtx,
               std::string(what) + " is defined by PTX over a 32-lane warp; this profile's warp"
               " is " + std::to_string(W_) + " lanes wide");
  }

  static bool is_lanemask_sreg(Sreg s) {
    return s == Sreg::LaneMaskEq || s == Sreg::LaneMaskLt || s == Sreg::LaneMaskLe ||
           s == Sreg::LaneMaskGt || s == Sreg::LaneMaskGe;
  }

  // Special registers.
  //
  // The clock family needs a word of explanation, because the honest answer is
  // "these are not times". VirtualGPU has no timing model: it does not know how
  // long anything takes, and inventing a number that looked like nanoseconds
  // would be the same mistake as reporting a cache hit rate. What it does have
  // is a deterministic count of instructions issued, and that count is
  // monotonic -- which is the only property most kernels actually use these
  // for. Spin-with-a-deadline and exponential backoff need the value to
  // *advance*, not to be accurate.
  //
  // So a kernel that waits on %clock64 terminates, and a kernel that measures
  // with it gets a reproducible number that is not a duration. That is a
  // documented divergence, in the same family as the SFU transcendentals being
  // more accurate here than on hardware. The alternative was refusing the
  // register, which failed the entire kernel over something it read only to
  // decide when to stop waiting.
  uint64_t sreg_value(Sreg s, uint32_t index, const Warp& w, const BlockCtx& ctx, uint32_t lane) {
    switch (s) {
      case Sreg::TidX: return w.tid_x[lane];
      case Sreg::TidY: return w.tid_y[lane];
      case Sreg::TidZ: return w.tid_z[lane];
      case Sreg::NtidX: return ctx.ntid[0];
      case Sreg::NtidY: return ctx.ntid[1];
      case Sreg::NtidZ: return ctx.ntid[2];
      case Sreg::CtaidX: return ctx.ctaid[0];
      case Sreg::CtaidY: return ctx.ctaid[1];
      case Sreg::CtaidZ: return ctx.ctaid[2];
      case Sreg::NctaidX: return ctx.nctaid[0];
      case Sreg::NctaidY: return ctx.nctaid[1];
      case Sreg::NctaidZ: return ctx.nctaid[2];
      // ---- thread-block clusters ----
      //
      // A cluster tiles the grid: block (bx,by,bz) sits in cluster
      // (bx/cx, by/cy, bz/cz) at position (bx%cx, by%cy, bz%cz) inside it.
      // With the default 1x1x1 cluster every block is its own cluster, which
      // makes %clusterid equal %ctaid and %cluster_ctaid zero -- exactly what
      // hardware reports for a launch with no cluster dimension.
      //
      // What is modelled here is the scheduling level and nothing more. A
      // cluster on real hardware also means the blocks are co-resident and can
      // read each other's shared memory; `.shared::cluster`, mapa and the
      // cluster barriers stay refused, because pretending distributed shared
      // memory works would let a kernel read a neighbour's data that was never
      // written.
      case Sreg::ClusterIdX: return ctx.ctaid[0] / ctx.cluster[0];
      case Sreg::ClusterIdY: return ctx.ctaid[1] / ctx.cluster[1];
      case Sreg::ClusterIdZ: return ctx.ctaid[2] / ctx.cluster[2];
      // Clusters per grid, rounded up: a grid that is not a whole number of
      // clusters still contains the partial one. validate() requires the grid
      // to divide evenly, as hardware does, so this rounding is belt and
      // braces rather than a second policy.
      case Sreg::NClusterIdX:
        return (ctx.nctaid[0] + ctx.cluster[0] - 1) / ctx.cluster[0];
      case Sreg::NClusterIdY:
        return (ctx.nctaid[1] + ctx.cluster[1] - 1) / ctx.cluster[1];
      case Sreg::NClusterIdZ:
        return (ctx.nctaid[2] + ctx.cluster[2] - 1) / ctx.cluster[2];
      case Sreg::ClusterCtaIdX: return ctx.ctaid[0] % ctx.cluster[0];
      case Sreg::ClusterCtaIdY: return ctx.ctaid[1] % ctx.cluster[1];
      case Sreg::ClusterCtaIdZ: return ctx.ctaid[2] % ctx.cluster[2];
      case Sreg::ClusterNCtaIdX: return ctx.cluster[0];
      case Sreg::ClusterNCtaIdY: return ctx.cluster[1];
      case Sreg::ClusterNCtaIdZ: return ctx.cluster[2];
      // The block's linear rank inside its cluster, x fastest. This is the one
      // a real kernel uses most: it indexes the per-block slot in a
      // cluster-wide array.
      case Sreg::ClusterCtaRank: return cluster_rank_of(ctx);
      case Sreg::ClusterNCtaRank:
        return uint64_t{ctx.cluster[0]} * ctx.cluster[1] * ctx.cluster[2];
      case Sreg::IsExplicitCluster: return ctx.explicit_cluster ? 1u : 0u;

      case Sreg::LaneId: return lane;
      case Sreg::LaneMaskEq: return Mask{1} << lane;
      case Sreg::LaneMaskLt: return lane == 0 ? 0u : (~0u >> (W_ - lane));
      case Sreg::LaneMaskLe: return static_cast<uint32_t>((uint64_t{2} << lane) - 1);
      case Sreg::LaneMaskGt: return lane == W_ - 1
                                        ? 0u
                                        : ~static_cast<uint32_t>((uint64_t{2} << lane) - 1);
      case Sreg::LaneMaskGe: return ~0u << lane;
      // Warps are numbered within the block, which is how a kernel uses this:
      // to index a per-warp slot in shared memory.
      case Sreg::WarpId: {
        const uint32_t linear = w.tid_x[lane] + w.tid_y[lane] * ctx.ntid[0] +
                                w.tid_z[lane] * ctx.ntid[0] * ctx.ntid[1];
        return linear / W_;
      }
      // The driver's parameter bank. Only two entries mean anything here: a
      // cooperative launch passes the address of its grid-barrier workspace as
      // %envreg1 (low half) and %envreg2 (high half). cooperative_groups reads
      // exactly those, and traps if the pair is zero -- which is how it detects
      // a grid.sync() outside a cooperative launch, and why every other entry
      // must stay zero rather than being given a plausible-looking value.
      // %envreg1 is the *high* half and %envreg2 the low one. That is not a
      // guess: the generated code reassembles them with
      // "bfi.b64 %rd1, %rd7, %rd6, 32, 32", which inserts %envreg1 into bits
      // 63:32 of %envreg2.
      case Sreg::EnvReg:
        if (index == 1) return static_cast<uint32_t>(cfg_.coop_workspace >> 32);
        if (index == 2) return static_cast<uint32_t>(cfg_.coop_workspace);
        return 0;
      case Sreg::NWarpId:
        return (ctx.ntid[0] * ctx.ntid[1] * ctx.ntid[2] + W_ - 1) / W_;

      // ---- the clock family: a counter, not a time (see above) ----
      case Sreg::Clock: return static_cast<uint32_t>(ctx.clock ? *ctx.clock : 0);
      case Sreg::ClockHi: return static_cast<uint32_t>((ctx.clock ? *ctx.clock : 0) >> 32);
      case Sreg::Clock64: return ctx.clock ? *ctx.clock : 0;
      // %globaltimer is nanoseconds on hardware. Scaled from the same counter
      // so the two stay consistent with each other: a kernel that compares them
      // sees one clock, not two that disagree.
      case Sreg::GlobalTimer: return ctx.clock ? *ctx.clock : 0;
      case Sreg::GlobalTimerLo: return static_cast<uint32_t>(ctx.clock ? *ctx.clock : 0);
      case Sreg::GlobalTimerHi: return static_cast<uint32_t>((ctx.clock ? *ctx.clock : 0) >> 32);

      // Which multiprocessor the block landed on. Blocks are assigned round
      // robin over the profile's SM count, which is a real assignment rather
      // than a made-up number: the block does run somewhere, and round robin is
      // the most even placement. Persistent kernels use this to partition work,
      // and they need distinct values far more than they need the exact one
      // hardware would have chosen.
      case Sreg::SmId: {
        // Distinct among the blocks resident at once, as a CTA's SM is on the
        // hardware. A grid that fits the device is all resident there, so
        // its linear order -- distinct across the whole grid, which a
        // persistent kernel partitioning work by %smid relies on -- and a
        // cooperative grid likewise. A larger grid reuses SMs as blocks
        // finish, and kernels index per-SM workspace by %smid (CUTLASS's
        // grouped GEMMs keep a tensor map per SM, and two resident CTAs that
        // shared one wrote each other's groups): each host thread runs one
        // cluster (or block) at a time, so thread t's rank-r block is SM
        // t * cluster size + r, and the launch caps its threads so that fits.
        const uint32_t sms = profile_.limits.multiprocessors;
        if (!sms) return 0;
        const uint64_t linear = uint64_t{ctx.ctaid[0]} +
                                uint64_t{ctx.ctaid[1]} * ctx.nctaid[0] +
                                uint64_t{ctx.ctaid[2]} * ctx.nctaid[0] * ctx.nctaid[1];
        const uint64_t blocks = uint64_t{ctx.nctaid[0]} * ctx.nctaid[1] * ctx.nctaid[2];
        if (cfg_.cooperative || blocks <= sms) return static_cast<uint32_t>(linear % sms);
        const auto c = cluster_shape();
        const uint64_t size = uint64_t{c[0]} * c[1] * c[2];
        const uint64_t rank = ctx.explicit_cluster ? cluster_rank_of(ctx) : 0;
        return static_cast<uint32_t>((uint64_t{worker_} * size + rank) % sms);
      }
      case Sreg::NSmId: return profile_.limits.multiprocessors;

      case Sreg::DynamicSmemSize: return cfg_.shared_bytes;
      case Sreg::TotalSmemSize: return kernel_shared_bytes();
      // Everything the block was allocated: its own shared memory, in
      // allocation units as %total_smem_size counts it, and the driver's
      // reserved part (1 KiB from compute capability 8.0) -- the PTX ISA's
      // definition. ptxas accepts it only from sm_90, so no card here could
      // measure it; the parts are the measured ones.
      case Sreg::AggrSmemSize: return kernel_shared_bytes() + reserved_smem_bytes();
      // Measured on an RTX 3060, for kernels from none to 48 KiB of shared
      // memory: begin, offset 0 and offset 1 are %total_smem_size; end is 288
      // bytes on (the space cooperative_groups' memory.h says it can expect);
      // cap is the reserved amount on (1 KiB, cudaDevAttrReservedSharedMemory-
      // PerBlock).
      case Sreg::ReservedSmemBegin:
      case Sreg::ReservedSmemOffset0:
      case Sreg::ReservedSmemOffset1:
        return reserved_smem_base();
      case Sreg::ReservedSmemEnd: return reserved_smem_base() + kReservedSmemUsed;
      case Sreg::ReservedSmemCap: return reserved_smem_base() + reserved_smem_bytes();
      case Sreg::GridId: return grid_id_;
      case Sreg::CurrentGraphExec: return cfg_.current_graph_exec;
    }
    return 0;
  }

  // Widens a 32-bit register into the 64-bit operand form. A small rotating
  // set of buffers keeps several operands of one instruction alive at once.
  // The returned reference lives in a small rotating ring, so it stays valid
  // only until a few more widening reads have happened. A caller that reads
  // other operands before using it must take a copy first -- stmatrix reads four
  // source registers between taking its addresses and storing through them, and
  // with a reference the fourth read overwrote the addresses.
  const Lanes& widen(Warp& w, const Lanes32& src) {
    Lanes& out = w.widen_scratch[w.widen_next];
    w.widen_next = (w.widen_next + 1) % w.widen_scratch.size();
    for (uint32_t l = 0; l < W_; ++l) out[l] = src[l];
    return out;
  }

  // True when an operand can be read as 32-bit lanes with no widening: a
  // narrow register, or an immediate (which is materialized either way).
  static bool narrow_operand(const Operand& o) {
    if (const auto* r = std::get_if<RegOperand>(&o)) return !r->reg.wide;
    return std::holds_alternative<ImmInt>(o) || std::holds_alternative<ImmFloatBits>(o);
  }

  // A 64-bit register read in place, the common case of an f64 operand; any
  // other operand takes read_operand.
  const Lanes& read_wide(Warp& w, const BlockCtx& ctx, const Instr& ins, const Operand& o, Lanes& scratch) {
    if (const auto* r = std::get_if<RegOperand>(&o);
        r && r->reg.wide && r->reg.id < w.regs64.size() && w.written64[r->reg.id])
      return w.regs64[r->reg.id];
    return read_operand(w, ctx, ins, o, scratch);
  }

  // Reads an operand as 32-bit lanes. Only valid when narrow_operand() holds.
  const Lanes32& read_narrow(Warp& w, const Instr& ins, const Operand& o, Lanes32& scratch) {
    if (const auto* r = std::get_if<RegOperand>(&o)) {
      // Undefined reads as zero, for the reason given on read_operand.
      if (r->reg.id >= w.regs32.size() || !w.written32[r->reg.id]) {
        (void)ins;
        scratch.fill(0);
        return scratch;
      }
      return w.regs32[r->reg.id];
    }
    uint32_t v = std::holds_alternative<ImmInt>(o)
                     ? static_cast<uint32_t>(std::get<ImmInt>(o).value)
                     : static_cast<uint32_t>(std::get<ImmFloatBits>(o).bits);
    scratch.fill(v);
    return scratch;
  }

  // The register a write with mask `m` goes to, marked written. Register
  // files are not zeroed when a block starts, so a register's first write
  // zero-fills it when that write does not cover the whole warp: lanes it
  // leaves alone must read zero afterwards, as they always have, rather
  // than whatever the memory held.
  uint32_t* dst32(Warp& w, uint32_t id, Mask m) {
    Lanes32& r = w.regs32[id];
    if (!w.written32[id]) {
      if (m != all_lanes(W_)) r.fill(0);
      w.written32[id] = 1;
    }
    return r.data();
  }
  uint64_t* dst64(Warp& w, uint32_t id, Mask m) {
    Lanes& r = w.regs64[id];
    if (!w.written64[id]) {
      if (m != all_lanes(W_)) r.fill(0);
      w.written64[id] = 1;
    }
    return r.data();
  }

  // Writes 32-bit lanes straight into the narrow file -- no widening, and half
  // the memory traffic of the 64-bit path.
  void write_narrow(Warp& w, const Reg& reg, Mask m, const Lanes32& vals) {
    uint32_t* dst = dst32(w, reg.id, m);
    for_active(m, W_, [&](uint32_t l) { dst[l] = vals[l]; });
  }

  void write_reg(Warp& w, const Reg& reg, Mask m, const Lanes& vals, uint32_t bits) {
    if (reg.wide) {
      uint64_t* dst = dst64(w, reg.id, m);
      if (bits >= 64)
        for_active(m, W_, [&](uint32_t l) { dst[l] = vals[l]; });
      else {
        const uint64_t keep = (1ull << bits) - 1;
        for_active(m, W_, [&](uint32_t l) { dst[l] = vals[l] & keep; });
      }
      return;
    }
    uint32_t* dst = dst32(w, reg.id, m);
    if (bits >= 32)
      for_active(m, W_, [&](uint32_t l) { dst[l] = static_cast<uint32_t>(vals[l]); });
    else {
      const uint32_t keep = (1u << bits) - 1;
      for_active(m, W_, [&](uint32_t l) { dst[l] = static_cast<uint32_t>(vals[l]) & keep; });
    }
  }

  // Predicates are declared .pred, so they live in the narrow numbering; the
  // slot index is offset past the 64-bit file to keep one predicate vector.
  uint32_t pred_index(const Reg& reg) const {
    return reg.wide ? cur_->num_regs32 + reg.id : reg.id;
  }

  Mask read_pred(Warp& w, const Instr& ins, const Reg& reg) {
    uint32_t idx = pred_index(reg);
    bool ok = reg.wide ? (reg.id < w.written64.size() && w.written64[reg.id])
                       : (reg.id < w.written32.size() && w.written32[reg.id]);
    if (idx >= w.preds.size() || !ok)
      ctx_fail(ins, -1, Err::UninitializedRegister,
               "predicate " + reg.name + " read before any write");
    return w.preds[idx];
  }

  static uint32_t popcount_mask(Mask m) {
    // Counted inline: without -mpopcnt the builtin is a call into libgcc, and
    // this runs for every instruction.
    m = m - ((m >> 1) & 0x5555'5555'5555'5555ull);
    m = (m & 0x3333'3333'3333'3333ull) + ((m >> 2) & 0x3333'3333'3333'3333ull);
    m = (m + (m >> 4)) & 0x0F0F'0F0F'0F0F'0F0Full;
    return static_cast<uint32_t>((m * 0x0101'0101'0101'0101ull) >> 56);
  }

  // Memory traffic, counted per active lane. Space matters: a shared access and
  // a global one cost very different things on hardware, and lumping them would
  // make the numbers useless for the comparison people actually want.
  // Sectors and bank conflicts, counted from the addresses the lanes actually
  // used. A device derives both by sampling; here they are exact, because every
  // lane's address is in hand at the moment of the access.
  //
  // `addrs` holds one address per lane, valid where `m` is set. `bytes` is the
  // per-lane access width, `count` the number of consecutive elements a vector
  // access touches from that address.
  void count_addresses(Space space, const Lanes& addrs, Mask m, uint32_t bytes, size_t count,
                       bool is_store) {
    if (m == 0) return;
    const uint64_t span = static_cast<uint64_t>(bytes) * count;

    if (space == Space::Shared) {
      // 32 banks of 4 bytes. Lanes reaching different words in one bank
      // serialize; lanes reaching the same word are broadcast and free. An
      // access wider than a word touches consecutive words, which the hardware
      // takes in separate passes -- counting each word is what makes a
      // conflict-free 8-byte access come out as conflict-free.
      uint32_t worst = 1;
      for (uint32_t bank = 0; bank < 32; ++bank) {
        std::array<uint64_t, kMaxWarpSize * 4> words{};
        uint32_t distinct = 0;
        for (uint32_t lane = 0; lane < W_; ++lane) {
          if (!(m & (Mask{1} << lane))) continue;
          for (uint64_t off = 0; off < span; off += 4) {
            const uint64_t word = (addrs[lane] + off) / 4;
            if (word % 32 != bank) continue;
            bool seen = false;
            for (uint32_t i = 0; i < distinct; ++i)
              if (words[i] == word) { seen = true; break; }
            if (!seen && distinct < words.size()) words[distinct++] = word;
          }
        }
        if (distinct > worst) worst = distinct;
      }
      stats_.shared_bank_conflicts += worst - 1;
      ++stats_.shared_requests;
      if (is_store) {
        stats_.shared_bank_conflicts_st += worst - 1;
        ++stats_.shared_requests_st;
      } else {
        stats_.shared_bank_conflicts_ld += worst - 1;
        ++stats_.shared_requests_ld;
      }
      return;
    }

    if (space == Space::Param) return;  // the constant bank, not device memory

    // Everything else moves in 32-byte sectors. Count the distinct ones the
    // warp touched: four for a coalesced 32-lane 4-byte load, up to 32 when
    // every lane lands in its own sector.
    std::array<uint64_t, kMaxWarpSize * 8> sectors{};
    uint32_t distinct = 0;
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const uint64_t first = addrs[lane] / 32;
      const uint64_t last = (addrs[lane] + (span ? span - 1 : 0)) / 32;
      for (uint64_t sec = first; sec <= last; ++sec) {
        bool seen = false;
        for (uint32_t i = 0; i < distinct; ++i)
          if (sectors[i] == sec) { seen = true; break; }
        if (!seen && distinct < sectors.size()) sectors[distinct++] = sec;
      }
    }
    if (space == Space::Local) {
      stats_.local_sectors += distinct;
      ++stats_.local_requests;
    } else {
      stats_.global_sectors += distinct;
      ++stats_.global_requests;
      const uint64_t asked = span * popcount_mask(m);   // bytes the lanes asked for
      if (is_store) {
        stats_.global_sectors_st += distinct;
        ++stats_.global_requests_st;
        stats_.global_bytes_st += asked;
      } else {
        stats_.global_sectors_ld += distinct;
        ++stats_.global_requests_ld;
        stats_.global_bytes_ld += asked;
      }
    }
  }

  void count_memory(Space space, uint32_t bytes, uint32_t lanes, bool is_store) {
    const uint64_t n = lanes;
    const uint64_t b = n * bytes;
    switch (space) {
      case Space::Shared:
        if (is_store) { stats_.shared_stores += n; stats_.shared_bytes_written += b; }
        else { stats_.shared_loads += n; stats_.shared_bytes_read += b; }
        break;
      case Space::Local:
        if (is_store) { stats_.local_stores += n; stats_.local_bytes_written += b; }
        else { stats_.local_loads += n; stats_.local_bytes_read += b; }
        break;
      case Space::Param:
        // Kernel parameters are read from the constant bank, not from device
        // memory. Counting them as global traffic inflated every kernel's load
        // count by one per thread and would make the byte totals wrong.
        break;
      case Space::Global:
      case Space::Generic:
        if (is_store) { stats_.global_stores += n; stats_.global_bytes_written += b; }
        else { stats_.global_loads += n; stats_.global_bytes_read += b; }
        break;
    }
  }

  Mask& pred_slot(Warp& w, const Reg& reg) {
    if (reg.wide) w.written64[reg.id] = 1;
    else w.written32[reg.id] = 1;
    return w.preds[pred_index(reg)];
  }

  // ---- routed memory access (device global VA range vs .local window) ----

  // Window base for an address space. Global/generic already alias the flat
  // device VA range, so their base is zero.
  static uint64_t space_base(Space sp) {
    switch (sp) {
      case Space::Shared: return kSharedVaBase;
      case Space::Local: return kLocalVaBase;
      case Space::Param: return kParamVaBase;
      default: return 0;
    }
  }

  bool is_local(uint64_t addr) const {
    return addr >= kLocalVaBase && addr < kLocalVaBase + kLocalVaSize;
  }

  bool is_shared(uint64_t addr) const {
    return addr >= kSharedVaBase && addr < kSharedVaBase + kSharedVaSize;
  }

  // ---- distributed shared memory ----

  // A block of this block's cluster by rank, failing when there is no such
  // block or it has exited: its shared memory went with it (the CUDA
  // programming guide's reason for ending a kernel that uses distributed
  // shared memory with cluster.sync()).
  const BlockCtx& cluster_block(const BlockCtx& ctx, const Instr& ins, int lane, uint64_t rank) {
    const ClusterState* cs = ctx.cluster_state;
    const size_t n = cs ? cs->ranks.size() : 1;
    if (!cs || rank >= n || !cs->ranks[rank])
      ctx_fail(ins, lane, Err::OutOfBounds,
               "a shared::cluster address names block rank " + std::to_string(rank) +
                   " of a cluster of " + std::to_string(n) + " block" + (n == 1 ? "" : "s"));
    const BlockCtx& b = *cs->ranks[rank];
    if (&b != &ctx && b.exited && *b.exited)
      ctx_fail(ins, lane, Err::OutOfBounds,
               "block rank " + std::to_string(rank) +
                   " of the cluster has exited, and its shared memory with it. A block whose "
                   "shared memory other blocks use must wait for them (barrier.cluster, or "
                   "cluster.sync()) before it exits");
    return b;
  }

  // The block a shared-window address belongs to and the offset in it --
  // see kClusterRankShift.
  struct SharedRef {
    const BlockCtx* owner;
    uint64_t off;
  };
  SharedRef shared_ref(const BlockCtx& ctx, const Instr& ins, int lane, uint64_t addr) {
    const uint64_t rel = addr - kSharedVaBase;
    const uint64_t rank = rel >> kClusterRankShift;
    if (rank == cluster_rank_of(ctx)) return {&ctx, rel & kClusterOffsetMask};
    return {&cluster_block(ctx, ins, lane, rank), rel & kClusterOffsetMask};
  }
  // The same, bounds-checked for `size` bytes.
  SharedRef shared_at(const BlockCtx& ctx, const Instr& ins, int lane, uint64_t addr, uint64_t size) {
    const SharedRef r = shared_ref(ctx, ins, lane, addr);
    const size_t have = r.owner->shared ? r.owner->shared->size() : 0;
    if (r.off + size > have)
      ctx_fail(ins, lane, Err::OutOfBounds,
               "shared memory access at offset " + std::to_string(r.off) + " (+" +
                   std::to_string(size) + " bytes) exceeds the " + std::to_string(have) +
                   "-byte shared allocation for " +
                   (r.owner == &ctx ? std::string("this block")
                                    : "block rank " + std::to_string(cluster_rank_of(*r.owner)) +
                                          " of the cluster"));
    return r;
  }
  // A shared::cluster address for `off` in block `rank`, as mapa returns it:
  // the plain offset when that is the issuing block.
  static uint64_t cluster_address(const BlockCtx& ctx, uint64_t rank, uint64_t off) {
    (void)ctx;
    return (rank << kClusterRankShift) | (off & kClusterOffsetMask);
  }

  // Race detection, off unless VGPU_RACE=1.
  static bool detect_races() { return g_race.load(std::memory_order_relaxed); }
  static bool races_strict() { return g_race_strict.load(std::memory_order_relaxed); }

  [[noreturn]] void report_race(const Instr& ins, const char* what, uint64_t word,
                                uint32_t other_warp) {
    ctx_fail(ins, -1, Err::DataRace,
             std::string(what) + " on shared memory at byte offset " +
                 std::to_string(word * 4) + ": warp " + std::to_string(cur_warp_) +
                 " and warp " + std::to_string(other_warp) +
                 " both reach it with no bar.sync between them, so which one wins is not "
                 "something the program decided. Hardware usually hides this because warps "
                 "advance together; it is a real race either way");
  }

  // True when the store leaves shared memory exactly as it found it.
  //
  // Such a store cannot be observed by anyone: no reader and no other writer
  // can tell whether it happened before or after, because the bytes are the
  // same either way. So it does not turn a conflicting access into a race.
  //
  // This is not a heuristic -- it is checked against the bytes actually there,
  // and the conflicting writer is what put them there. Real kernels do this on
  // purpose: llama.cpp's mul_mat_q clamps out-of-range tile rows with
  // `i = min(i, i_max)`, so several warps recompute the same source pointer and
  // write the same value to the same word. Reporting that as a bug makes the
  // detector useless on the code it exists to check.
  //
  // The write is still *recorded* as a write, so a later store of a different
  // value is still caught against it. Only the report is suppressed.
  bool unobservable_write(const BlockCtx& ctx, uint64_t addr, uint32_t bytes,
                          uint64_t value) const {
    if (races_strict()) return false;
    if (bytes == 0 || bytes > 8) return false;
    uint64_t existing = 0;
    std::memcpy(&existing, ctx.shared->data() + (addr - kSharedVaBase), bytes);
    uint64_t incoming = 0;
    std::memcpy(&incoming, &value, bytes);
    return existing == incoming;
  }

  void note_shared_access(const BlockCtx& ctx, const Instr& ins, uint64_t addr, uint32_t bytes,
                          bool is_write, uint64_t value = 0) {
    if (!ctx.shadow) return;
    SharedShadow& sh = *ctx.shadow;
    const bool silent = is_write && unobservable_write(ctx, addr, bytes, value);
    const uint64_t first = (addr - kSharedVaBase) / 4;
    const uint64_t last = (addr - kSharedVaBase + (bytes ? bytes - 1 : 0)) / 4;
    for (uint64_t w = first; w <= last && w < sh.words.size(); ++w) {
      WordShadow& s = sh.words[w];
      if (is_write) {
        if (!silent && s.write_epoch == sh.epoch && s.writer != 0xFFFF && s.writer != cur_warp_)
          report_race(ins, "write-write race", w, s.writer);
        if (!silent && s.read_epoch == sh.epoch) {
          const uint32_t others = s.readers & ~(1u << cur_warp_);
          if (others) report_race(ins, "read-write race", w, __builtin_ctzll(others));
        }
        s.writer = static_cast<uint16_t>(cur_warp_);
        s.write_epoch = sh.epoch;
      } else {
        if (s.write_epoch == sh.epoch && s.writer != 0xFFFF && s.writer != cur_warp_)
          report_race(ins, "write-read race", w, s.writer);
        if (s.read_epoch != sh.epoch) {
          s.readers = 0;
          s.read_epoch = sh.epoch;
        }
        s.readers |= (1u << cur_warp_);
      }
    }
  }

  // For copies that only ever reach the issuing block's own shared memory
  // (cp.async): an address naming another block is out of its bounds.
  void check_shared(const BlockCtx& ctx, const Instr& ins, int lane, uint64_t addr, uint32_t size) {
    const SharedRef r = shared_ref(ctx, ins, lane, addr);
    if (r.owner != &ctx)
      ctx_fail(ins, lane, Err::InvalidValue,
               "a .shared::cta access to another block of the cluster (rank " +
                   std::to_string(cluster_rank_of(*r.owner)) + ")");
    uint64_t off = r.off;
    size_t have = ctx.shared ? ctx.shared->size() : 0;
    if (off + size > have)
      ctx_fail(ins, lane, Err::OutOfBounds,
               "shared memory access at offset " + std::to_string(off) + " (+" +
                   std::to_string(size) + " bytes) exceeds the " + std::to_string(have) +
                   "-byte shared allocation for this block");
  }

  std::vector<uint8_t>& lane_local(Warp& w, uint32_t lane) {
    if (w.local.empty()) w.local.resize(W_);
    auto& buf = w.local[lane];
    const uint64_t top = local_top();
    if (buf.size() < top) buf.resize(top, 0);
    return buf;
  }

  // A lane's alloca stack (PTX ISA 9.7.19): its pointer starts at the top and
  // alloca moves it down. Memory below the pointer is not allocated -- never
  // yet, or freed again by stackrestore -- and touching it is reported, as is
  // anything past the top.
  uint64_t stack_top() const { return kStackOffset + cfg_.stack_bytes; }
  uint64_t& stack_ptr(Warp& w, uint32_t lane) {
    if (!w.sp_valid) {
      w.sp.fill(stack_top());
      w.sp_valid = true;
    }
    return w.sp[lane];
  }
  uint8_t* stack_bytes_at(Warp& w, const Instr& ins, uint32_t lane, uint64_t off, uint32_t size, bool store) {
    const uint64_t sp = stack_ptr(w, lane), top = stack_top();
    if (off < sp || off + size > top)
      ctx_fail(ins, static_cast<int>(lane), Err::OutOfBounds,
               std::string("local ") + (store ? "store to" : "load from") + " stack offset " +
                   std::to_string(off - kStackOffset) + " (+" + std::to_string(size) + " bytes), " +
                   (off + size > top ? "past the top of the thread's " + std::to_string(cfg_.stack_bytes) +
                                           "-byte stack (cudaLimitStackSize)"
                                     : "below the stack pointer at " + std::to_string(sp - kStackOffset) +
                                           ": memory no alloca holds, or one stackrestore has freed"));
    if (w.stack.size() < W_) w.stack.resize(W_);
    std::vector<uint8_t>& buf = w.stack[lane];
    if (buf.size() < cfg_.stack_bytes) buf.resize(cfg_.stack_bytes, 0);
    return buf.data() + (off - kStackOffset);
  }

  void exec_stack(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpStack& op, Mask m) {
    Lanes _s;
    const Lanes& src = read_operand(w, ctx, ins, op.src, _s);
    Lanes out{};
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      uint64_t& sp = stack_ptr(w, lane);
      const uint64_t v = op.wide ? src[lane] : (src[lane] & 0xFFFFFFFFu);
      switch (op.kind) {
        case OpStack::Kind::Save:
          out[lane] = sp;
          break;
        case OpStack::Kind::Restore:
          if (v < kStackOffset || v > stack_top())
            ctx_fail(ins, static_cast<int>(lane), Err::InvalidValue,
                     "stackrestore to " + std::to_string(v) +
                         ", which is not a stack pointer stacksave or alloca gave (9.7.19.2)");
          sp = v;
          break;
        case OpStack::Kind::Alloca: {
          // Down by size, then down to the alignment (the ISA's
          // alloc_stack_mem): the result is the block's lowest address.
          if (v > sp - kStackOffset || ((sp - v) & ~uint64_t{op.align - 1}) < kStackOffset)
            ctx_fail(ins, static_cast<int>(lane), Err::OutOfBounds,
                     "alloca of " + std::to_string(v) + " bytes overflows the thread's " +
                         std::to_string(cfg_.stack_bytes) + "-byte stack (" +
                         std::to_string(stack_top() - sp) +
                         " in use); cudaDeviceSetLimit(cudaLimitStackSize) raises it");
          sp = (sp - v) & ~uint64_t{op.align - 1};
          out[lane] = sp;
          break;
        }
      }
    }
    if (op.kind != OpStack::Kind::Restore) write_reg(w, op.dst, m, out, op.wide ? 64 : 32);
  }

  void check_local(const Instr& ins, int lane, uint64_t addr, uint32_t size) {
    uint64_t off = addr - kLocalVaBase;
    if (off + size > local_top())
      ctx_fail(ins, lane, Err::OutOfBounds,
               "local memory access at frame offset " + std::to_string(off) + " (+" +
                   std::to_string(size) + " bytes) exceeds the " +
                   std::to_string(local_top()) + "-byte .local frame");
  }

  // The end of the current function's .local frame, which is also the end of
  // every frame below it: a callee may be handed a pointer into its caller's.
  uint64_t local_top() const { return local_base_ + cur_->local_frame_size; }

  // A bit flip armed on arithmetic results (`vgpu fault arm --bitflip --on
  // alu`) corrupts one lane's result -- the lowest active one -- of the next
  // floating-point, math or matrix instruction: the values a result check
  // verifies. Integer results are left alone; they are mostly addresses and
  // loop counters, where a flip is a crash or a hang rather than silent
  // corruption. While nothing is armed this is one relaxed load per
  // instruction, not per lane.
  void alu_fault(Lanes& r, Mask m, uint32_t bits) {
    if (!m || !mem_.alu_fault_armed()) return;
    const uint32_t lane = static_cast<uint32_t>(__builtin_ctzll(static_cast<unsigned long long>(m)));
    r[lane] = mem_.alu_result(r[lane], bits);
  }

  uint64_t load_routed(Warp& w, const BlockCtx& ctx, const Instr& ins, uint32_t lane, uint64_t addr,
                       uint32_t size) {
    if (is_shared(addr)) {
      const SharedRef r = shared_at(ctx, ins, static_cast<int>(lane), addr, size);
      // The race detector orders warps by their own block's barriers, which
      // say nothing about another block's, so a remote access is not noted.
      if (r.owner == &ctx) note_shared_access(ctx, ins, kSharedVaBase + r.off, size, /*is_write=*/false);
      uint64_t v = 0;
      std::memcpy(&v, r.owner->shared->data() + r.off, size);
      try {
        return mem_.shared_loaded(r.off, size, v);
      } catch (const Error& e) {
        rethrow_with_context(e, ins, static_cast<int>(lane));
      }
    }
    if (is_local(addr)) {
      uint64_t v = 0;
      if (addr - kLocalVaBase >= kStackOffset) {
        std::memcpy(&v, stack_bytes_at(w, ins, lane, addr - kLocalVaBase, size, false), size);
        return v;
      }
      check_local(ins, static_cast<int>(lane), addr, size);
      std::memcpy(&v, lane_local(w, lane).data() + (addr - kLocalVaBase), size);
      return v;
    }
    // A generic pointer to a kernel parameter -- a __grid_constant__ taken by
    // address, which is how a TMA tensor map reaches its instruction.
    if (addr >= kParamVaBase && addr < kParamVaBase + kParamVaSize) {
      if (addr + size > kParamVaBase + params_.bytes.size())
        ctx_fail(ins, static_cast<int>(lane), Err::OutOfBounds,
                 "a generic load reads past the end of the kernel's parameters");
      uint64_t v = 0;
      std::memcpy(&v, params_.bytes.data() + (addr - kParamVaBase), size);
      return v;
    }
    try {
      return mem_.load_scalar(addr, size);
    } catch (const Error& e) {
      rethrow_with_context(e, ins, static_cast<int>(lane));
    }
  }

  void store_routed(Warp& w, const BlockCtx& ctx, const Instr& ins, uint32_t lane, uint64_t addr,
                    uint32_t size, uint64_t value) {
    if (is_shared(addr)) {
      const SharedRef r = shared_at(ctx, ins, static_cast<int>(lane), addr, size);
      if (r.owner == &ctx) note_shared_access(ctx, ins, kSharedVaBase + r.off, size, /*is_write=*/true, value);
      std::memcpy(r.owner->shared->data() + r.off, &value, size);
      return;
    }
    if (is_local(addr)) {
      if (addr - kLocalVaBase >= kStackOffset) {
        std::memcpy(stack_bytes_at(w, ins, lane, addr - kLocalVaBase, size, true), &value, size);
        return;
      }
      check_local(ins, static_cast<int>(lane), addr, size);
      std::memcpy(lane_local(w, lane).data() + (addr - kLocalVaBase), &value, size);
      return;
    }
    if (addr >= kParamVaBase && addr < kParamVaBase + kParamVaSize)
      ctx_fail(ins, static_cast<int>(lane), Err::InvalidValue,
               "a store through a generic pointer to a kernel parameter; parameters are read-only");
    try {
      mem_.store_scalar(addr, size, value);
    } catch (const Error& e) {
      rethrow_with_context(e, ins, static_cast<int>(lane));
    }
  }

  // ---- the dispatcher ----

  void step(Warp& w, const BlockCtx& ctx, size_t idx, const Instr& ins) {
    step_idx_ = idx;
    Mask active = w.paths[idx].mask;
    Mask m = active;
    if (ins.has_pred) {
      Mask p = read_pred(w, ins, ins.pred);
      if (ins.pred_negated) p = ~p;
      m &= p;
    }

    if (const auto* op = std::get_if<OpBra>(&ins.op)) {
      exec_bra(w, idx, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpBrx>(&ins.op)) {
      exec_brx(w, ctx, ins, idx, *op, m);
      return;
    }
    if (std::holds_alternative<OpRet>(ins.op)) {
      exec_ret(w, ctx, ins, idx, m);
      return;
    }
    if (const auto* op = std::get_if<OpBarRed>(&ins.op)) {
      if (ins.has_pred && !w.bar_red_waiting) {
        // As with bar.sync: the barrier is aligned, so a guard must be the
        // same for the whole warp. A warp that skips it does not arrive.
        if (m == 0) {
          ++w.paths[idx].pc;
          return;
        }
        if (m != active)
          ctx_fail(ins, -1, Err::UnsupportedPtx,
                   "a guarded bar.red that some lanes of the warp take and others skip; the "
                   "barrier is aligned, so the warp must agree");
      }
      if (!ctx.bar_red)
        ctx_fail(ins, -1, Err::UnsupportedPtx, "bar.red outside a block context");
      BarrierReduction& red = *ctx.bar_red;
      if (!w.bar_red_waiting && w.paths.size() > 1) {
        // Like bar.sync, bar.red waits for every live thread, so lanes of this
        // warp still on another path have to get here before the warp votes.
        // CUTLASS's semaphore wait sends thread 0 round a fetch at a higher pc
        // while its warp-mates go straight back to the bar.red; voting for the
        // partial warp here left thread 0 behind for good, and the loop never
        // saw the semaphore change.
        if (select_other_runnable(w, idx) != idx) {
          w.paths[idx].parked = Path::kAtBarrier;
          return;
        }
        ctx_fail(ins, -1, Err::UnsupportedPtx,
                 "bar.red cannot be reached by every lane of the warp: some lanes are on a path "
                 "that never arrives at this barrier");
      }
      if (!w.bar_red_waiting && !w.frames.empty())
        refuse_barrier_outside_the_call(w, ins, "bar.red");
      // Folds this warp's votes into an accumulator; `first` starts it.
      auto vote = [&](uint64_t& acc, bool first) {
        if (first) acc = op->op == BarRedOp::And ? ~uint64_t{0} : 0;
        Mask p = read_pred(w, ins, op->src);
        if (op->negate_src) p = ~p;
        const Mask voters = p & m;
        switch (op->op) {
          case BarRedOp::And:
            // True only if every participating lane of every warp voted true.
            acc &= (voters == m) ? 1u : 0u;
            break;
          case BarRedOp::Or:
            acc |= (voters != 0) ? 1u : 0u;
            break;
          case BarRedOp::Popc:
            acc += static_cast<uint64_t>(popcount_mask(voters));
            break;
        }
      };
      auto deliver = [&](uint64_t result) {
        if (op->op == BarRedOp::Popc) {
          Lanes r;
          for (uint32_t lane = 0; lane < W_; ++lane)
            if (m & (Mask{1} << lane)) r[lane] = result;
          write_reg(w, op->dst, m, r, 32);
        } else {
          Mask& dp = pred_slot(w, op->dst);
          dp = (result & 1u) ? (dp | m) : (dp & ~m);
        }
        ++w.paths[idx].pc;
      };
      const uint32_t lead = first_set(m);
      Lanes _s_id;
      const uint64_t id = read_operand(w, ctx, ins, op->id, _s_id)[lead];
      if (id > 15)
        ctx_fail(ins, -1, Err::InvalidValue,
                 "barrier " + std::to_string(id) + ": a CTA has barriers 0 to 15");
      if (op->have_count) {
        // A barrier with a thread count: only the warps that arrive at it take
        // part, and its completion hands each of them the answer.
        if (w.bar_red_waiting) {
          w.bar_red_waiting = false;
          deliver(w.bar_red_result);
          return;
        }
        Lanes _s_c;
        const uint64_t count = static_cast<uint32_t>(read_operand(w, ctx, ins, op->count, _s_c)[lead]);
        if (count == 0 || count % W_)
          ctx_fail(ins, -1, Err::InvalidValue,
                   "a barrier's thread count must be a non-zero multiple of the warp size (" +
                       std::to_string(W_) + "); got " + std::to_string(count));
        NamedBarrier& nb = ctx.bars->bar[id];
        vote(nb.acc, nb.arrived == 0);
        nb.arrived += W_;
        ++stats_.barriers;
        if (nb.arrived >= count) {
          const uint64_t result = nb.acc;
          for (size_t i = 0; i < ctx.warps->size(); ++i)
            if (nb.waiting >> i & 1) {
              Warp& other = (*ctx.warps)[i];
              other.state = Warp::State::Ready;
              other.counted_barrier = false;
              other.bar_red_result = result;
            }
          nb = NamedBarrier{};
          if (ctx.shadow) ++ctx.shadow->epoch;
          deliver(result);
          return;
        }
        nb.waiting |= uint64_t{1} << cur_warp_;
        w.state = Warp::State::AtBarrier;
        w.counted_barrier = true;
        w.barrier_id = static_cast<uint8_t>(id);
        w.bar_red_waiting = true;
        return;
      }
      if (!w.bar_red_waiting) {
        // Contribute and wait. The result is not known until every warp in the
        // block has arrived, so the pc stays put and this re-executes on
        // release rather than advancing now.
        w.bar_red_round = red.round;
        vote(red.acc, red.arrived == 0);
        ++red.arrived;
        ++stats_.barriers;
        w.bar_red_waiting = true;
        w.state = Warp::State::AtBarrier;
        return;
      }
      // Released: collect the block-wide result of the round this warp voted
      // in, which the release set aside (the round after it cannot have
      // completed: it needs this warp's vote).
      w.bar_red_waiting = false;
      if (w.bar_red_round + 1 != red.round)
        ctx_fail(ins, -1, Err::UnsupportedPtx,
                 "bar.red released a warp whose round has not completed; were some warps at a "
                 "bar.sync on the same barrier?");
      deliver(red.result);
      return;
    }
    if (const auto* op = std::get_if<OpClusterBarrier>(&ins.op)) {
      if (!ctx.cluster_state)
        ctx_fail(ins, -1, Err::UnsupportedPtx, "barrier.cluster outside a block context");
      ClusterState& cs = *ctx.cluster_state;
      if (!op->wait) {
        if (w.cluster_arrived & m)
          ctx_fail(ins, -1, Err::UnsupportedPtx,
                   "a thread arrived at barrier.cluster twice before it completed, which the ISA "
                   "does not allow");
        w.cluster_arrived |= m;
        w.cluster_wait_phase = cs.phase;
        ++stats_.barriers;
        complete_cluster_barrier_if_done(cs);
        ++w.paths[idx].pc;
        return;
      }
      // Waiting: the phase this warp arrived in has to end. Threads may have
      // exited since the last look, which can complete it, so look again;
      // if it still has not, give the turn away and re-execute later.
      if (cs.phase == w.cluster_wait_phase) complete_cluster_barrier_if_done(cs);
      if (cs.phase == w.cluster_wait_phase) {
        // Other lanes of this warp may be what the barrier waits for -- on
        // their way to exit, say -- and at a higher pc they would never run
        // behind a lower one spinning here. Step aside for them this turn.
        if (select_other_runnable(w, idx) != idx) w.paths[idx].parked = Path::kThisTurn;
        else w.yield_now = true;
        return;
      }
      ++w.paths[idx].pc;
      return;
    }
    if (const auto* bop = std::get_if<OpBar>(&ins.op)) {
      if (bop->warp) {   // bar.warp.sync: waits for its lanes (wait_for_members)
        sync_mask_groups(w, ctx, ins, bop->id, m);   // a lane outside its own mask traps
        if (wait_for_members(w, ctx, ins, idx, bop->id, m)) return;
        ++w.paths[idx].pc;
        return;
      }
      if (ins.has_pred) {
        // A guarded barrier is fine as long as the warp agrees: all of it
        // takes the barrier or none does. Lanes that disagree leave the
        // aligned barrier undefined.
        if (m == 0) {
          ++w.paths[idx].pc;
          return;
        }
        if (m != active)
          ctx_fail(ins, -1, Err::UnsupportedPtx,
                   "a guarded bar.sync that some lanes of the warp take and others skip; the "
                   "barrier is aligned, so the warp must agree");
      }
      ++stats_.barriers;
      // Every live lane must arrive before the warp yields. Lanes still on
      // other paths have a higher pc and will merge here first; if any path
      // can never reach this barrier the kernel is malformed, and the step
      // budget catches it rather than deadlocking silently.
      if (w.paths.size() > 1) {
        // Wait here for the other paths, wherever they are. Usually they are
        // behind -- a lower pc -- but not always: nvcc places a rarely taken
        // block after the kernel's ret and branches back from it (CUTLASS's
        // "if (threadIdx.x == 0) prefetch(...)" compiles that way), so a lane
        // can reach this barrier from a higher pc. Only when no path is left
        // that could still move is the barrier unreachable.
        size_t other = select_other_runnable(w, idx);
        if (other != idx) {
          w.paths[idx].parked = Path::kAtBarrier;
          return;
        }
        // Where the other lanes are is the diagnosis, so it is in the message.
        std::string where;
        for (size_t i = 0; i < w.paths.size(); ++i) {
          if (i == idx) continue;
          const Path& p = w.paths[i];
          where += "\n  lanes 0x" + [&] {
            char buf[32];
            std::snprintf(buf, sizeof buf, "%llx", static_cast<unsigned long long>(p.mask));
            return std::string(buf);
          }() + " are at PTX line " +
                   (p.pc < cur_->body.size() ? std::to_string(cur_->body[p.pc].line) + ": " +
                                                   cur_->body[p.pc].text
                                             : std::string("the end"));
        }
        ctx_fail(ins, -1, Err::UnsupportedPtx,
                 "bar.sync cannot be reached by every lane of the warp: some lanes are on a path "
                 "that never arrives at this barrier" + where);
      }
      if (!w.frames.empty()) refuse_barrier_outside_the_call(w, ins, "bar.sync");
      const uint32_t lead = first_set(m);
      Lanes _s_id;
      const uint64_t id = read_operand(w, ctx, ins, bop->id, _s_id)[lead];
      if (id > 15)
        ctx_fail(ins, -1, Err::InvalidValue,
                 "barrier " + std::to_string(id) + ": a CTA has barriers 0 to 15");
      ++w.paths[idx].pc;
      if (!bop->have_count) {
        w.state = Warp::State::AtBarrier;
        w.counted_barrier = false;
        w.barrier_id = static_cast<uint8_t>(id);
        return;
      }
      Lanes _s_c;
      const uint64_t count = static_cast<uint32_t>(read_operand(w, ctx, ins, bop->count, _s_c)[lead]);
      if (count == 0 || count % W_)
        ctx_fail(ins, -1, Err::InvalidValue,
                 "a barrier's thread count must be a non-zero multiple of the warp size (" +
                     std::to_string(W_) + "); got " + std::to_string(count));
      NamedBarrier& nb = ctx.bars->bar[id];
      nb.arrived += W_;
      if (nb.arrived >= count) {
        // Complete: the waiters go, and the barrier is ready for reuse.
        for (size_t i = 0; i < ctx.warps->size(); ++i)
          if (nb.waiting >> i & 1) {
            Warp& other = (*ctx.warps)[i];
            other.state = Warp::State::Ready;
            other.counted_barrier = false;
          }
        nb = NamedBarrier{};
        if (ctx.shadow) ++ctx.shadow->epoch;
        return;
      }
      if (!bop->arrive) {
        nb.waiting |= uint64_t{1} << cur_warp_;
        w.state = Warp::State::AtBarrier;
        w.counted_barrier = true;
        w.barrier_id = static_cast<uint8_t>(id);
      }
      return;
    }

    // wgmma.wait_group N: the groups before the N most recent are complete
    // only once the whole warpgroup has issued their operations -- wgmma is
    // one operation of all four warps. Until then the warp waits here. A warp
    // that went on alone could release a stage its warp-mates had not yet
    // waited for; the refill moved the barrier on two phases and the last
    // warp's parity wait never finished (CUTLASS's ping-pong GEMM).
    if (const auto* wop = std::get_if<OpWgmma>(&ins.op);
        wop && wop->kind == WgmmaKind::Wait && m != 0 && !wgmma_group_caught_up(w, ctx, wop->wait_n)) {
      w.yield_now = true;
      return;
    }

    if (const Operand* members = sync_members(ins); members && m != 0 && wait_for_members(w, ctx, ins, idx, *members, m))
      return;

    if (m != 0) {
      try {
        std::vector<Mask> groups;
        if (const Operand* members = sync_members(ins)) groups = sync_mask_groups(w, ctx, ins, *members, m);
        if (groups.empty()) dispatch(w, ctx, ins, m);
        else for (Mask g : groups) dispatch(w, ctx, ins, g);
      } catch (const Error& e) {
        if (std::string(e.what()).find("in kernel") == std::string::npos)
          rethrow_with_context(e, ins, -1);
        throw;
      }
    }
    if (w.entered_call) {
      w.entered_call = false;
      return;
    }
    if (w.retry) {
      w.retry = false;
      return;
    }
    ++w.paths[idx].pc;
  }

  // Whether every warp of w's warpgroup has issued the wgmma operations in the
  // groups a wait_group `keep` waits for, and if so forgets those groups.
  bool wgmma_group_caught_up(Warp& w, const BlockCtx& ctx, uint32_t keep) {
    auto& marks = w.wgmma_commits;
    if (marks.size() <= keep || !ctx.warps) return true;
    const uint64_t need = marks[marks.size() - keep - 1];
    const uint32_t linear0 = w.tid_x[0] + w.tid_y[0] * ctx.ntid[0] +
                             w.tid_z[0] * ctx.ntid[0] * ctx.ntid[1];
    const size_t first = (linear0 / W_) / 4 * 4;
    for (size_t r = first; r < first + 4 && r < ctx.warps->size(); ++r)
      if ((*ctx.warps)[r].wgmma_issued < need) return false;
    marks.erase(marks.begin(), marks.end() - keep);
    return true;
  }

  // Is there another path that can still run -- one not itself waiting at a
  // barrier? Used to decide whether a barrier is merely waiting for stragglers.
  // The member mask of a warp instruction that synchronizes the lanes it
  // names (shfl.sync, vote.sync, match.sync, redux.sync), or null.
  static const Operand* sync_members(const Instr& ins) {
    if (const auto* op = std::get_if<OpShfl>(&ins.op); op && op->has_members) return &op->member_mask;
    if (const auto* op = std::get_if<OpMatch>(&ins.op)) return &op->membermask;
    if (const auto* op = std::get_if<OpRedux>(&ins.op)) return &op->members;
    if (const auto* op = std::get_if<OpVote>(&ins.op); op && op->has_members) return &op->members;
    return nullptr;
  }

  // The member masks of a *.sync warp instruction, as ptxas's code treats
  // them. Measured on an RTX 3060 (sm_86, CUDA 13.0), with the mask a
  // register -- one the compiler cannot see through, such as a lane-dependent
  // value or a kernel argument; a constant mask compiles to the bare
  // instruction, which ignores it:
  //  - every lane the instruction runs for names the same mask: the
  //    instruction runs once over the lanes that are there (vote, shfl and
  //    match.sync ignore the mask, even one that leaves lanes of the warp
  //    out -- __ballot_sync(0xffff, p) from all 32 lanes is the 32-lane
  //    ballot); redux.sync's code waits with WARPSYNC.EXCLUSIVE first, which
  //    a lane outside the mask traps on;
  //  - lanes name different masks (for shfl, masks that disagree about which
  //    lanes they share: see below): the code runs the instruction once for
  //    each distinct mask value, over the lanes that named it (a lane's
  //    __ballot_sync(1u << lane, p) is its own bit of p), and each of those
  //    waits with WARPSYNC, which a lane outside its mask traps on: "an
  //    illegal instruction was encountered" (715), whatever the instruction.
  //    __ballot_sync((1u << lane) - 1, p), the exclusive prefix, is the common
  //    way into it: lane 0's mask is 0.
  // Returns the lane groups to run the instruction for one at a time, or
  // empty to run it as it stands.
  std::vector<Mask> sync_mask_groups(Warp& w, const BlockCtx& ctx, const Instr& ins, const Operand& members, Mask m) {
    if (!std::holds_alternative<RegOperand>(members) || m == 0) return {};
    Lanes _s_mm;
    const Lanes& mm = read_operand(w, ctx, ins, members, _s_mm);
    const uint32_t first = static_cast<uint32_t>(mm[first_set(m)]);
    bool uniform = true;
    for (uint32_t lane = 0; lane < W_; ++lane)
      if ((m & (Mask{1} << lane)) && static_cast<uint32_t>(mm[lane]) != first) uniform = false;
    // shfl's code asks something weaker than "one mask for all": that every
    // lane its own mask names, among those here, names that same mask. Two
    // halves of the warp with a mask each (0xffff and 0xffff0000) are
    // consistent, and run as the plain 32-lane shuffle.
    if (!uniform && std::holds_alternative<OpShfl>(ins.op)) {
      uniform = true;
      for (uint32_t lane = 0; lane < W_ && uniform; ++lane) {
        if (!(m & (Mask{1} << lane))) continue;
        for (uint32_t o = 0; o < W_; ++o)
          if ((m & (Mask{1} << o)) && ((static_cast<uint32_t>(mm[lane]) >> o) & 1u) &&
              static_cast<uint32_t>(mm[o]) != static_cast<uint32_t>(mm[lane]))
            uniform = false;
      }
    }
    if (uniform && !std::holds_alternative<OpRedux>(ins.op)) return {};
    for (uint32_t lane = 0; lane < W_; ++lane)
      if ((m & (Mask{1} << lane)) && !((static_cast<uint32_t>(mm[lane]) >> lane) & 1u))
        ctx_fail(ins, -1, Err::IllegalInstruction,
                 "a *.sync warp instruction run by lane " + std::to_string(lane) + ", which its member mask " +
                 std::to_string(static_cast<uint32_t>(mm[lane])) + " leaves out");
    if (uniform) return {};
    std::vector<Mask> groups;
    std::vector<uint32_t> values;
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const uint32_t v = static_cast<uint32_t>(mm[lane]);
      size_t g = 0;
      while (g < values.size() && values[g] != v) ++g;
      if (g == values.size()) { values.push_back(v); groups.push_back(0); }
      groups[g] |= Mask{1} << lane;
    }
    return groups;
  }

  // On the hardware a *.sync warp instruction and bar.warp.sync wait for every
  // lane their member mask names. Running the lowest pc first used to bring
  // those lanes here first, but a path that has waited too long now runs
  // ahead (select_path), and it must not go on without them: in
  // cooperative_groups' multi-warp tiles lane 0 spins on the other warps
  // while its warp-mates wait at __syncwarp, then all read the tile's result.
  // So a path that gets here while lanes it names are on another path parks,
  // as at bar.red, until they reach this pc and merge with it. Returns true
  // when it parked; if no other path can run, it goes on as before.
  bool wait_for_members(Warp& w, const BlockCtx& ctx, const Instr& ins, size_t idx, const Operand& members, Mask m) {
    if (w.paths.size() < 2 || m == 0) return false;
    Lanes _s_members;
    const Mask named = static_cast<Mask>(read_operand(w, ctx, ins, members, _s_members)[first_set(m)]) & 0xffffffffu;
    for (size_t i = 0; i < w.paths.size(); ++i) {
      if (i == idx || !(w.paths[i].mask & named)) continue;
      if (select_other_runnable(w, idx) == idx) return false;
      w.paths[idx].parked = Path::kAtBarrier;
      return true;
    }
    return false;
  }

  size_t select_other_runnable(Warp& w, size_t idx) {
    for (size_t i = 0; i < w.paths.size(); ++i)
      if (i != idx && !w.paths[i].parked) return i;
    return idx;
  }

  void exec_bra(Warp& w, size_t idx, const OpBra& op, Mask m) {
    // A warp that has run a long turn gives it up at its next backward
    // branch. Warps of a block make progress independently on the hardware,
    // so one may spin on a flag another warp sets -- cooperative_groups' tiles
    // of more than one warp synchronise that way, through plain loads and
    // atomics in shared memory -- and a turn that only ended at a barrier
    // never let the other warp run. Yielding only at a loop edge, and only
    // after this many instructions, keeps straight-line code and short loops
    // in one turn, so the deterministic order stays what it was for them.
    if (m != 0 && op.target <= w.paths[idx].pc && w.steps - turn_start_ >= kLongTurn)
      w.yield_now = true;
    Mask taken = m;
    Mask fallthrough = w.paths[idx].mask & ~taken;
    if (taken == 0) {
      ++w.paths[idx].pc;
      return;
    }
    if (fallthrough == 0) {
      w.paths[idx].pc = op.target;
      return;
    }
    // Diverge: both halves become live paths, and whichever has the lower pc
    // runs first. They merge again as soon as they reach the same pc. Only this
    // case is divergence -- a branch every lane agrees on took one of the two
    // early returns above and costs nothing.
    ++stats_.divergent_branches;
    size_t fall_pc = w.paths[idx].pc + 1;
    w.paths[idx].pc = op.target;
    w.paths[idx].mask = taken;
    Path fall{fall_pc, fallthrough};
    fall.issued_at = w.steps;
    w.paths.push_back(fall);
  }

  // brx.idx: each lane goes where its own index points, so the warp splits
  // into one path per distinct target (and one for the lanes a guard kept
  // out). Lanes that pick the same entry stay together, and paths that reach
  // a common pc merge again, exactly as after a divergent bra.
  void exec_brx(Warp& w, const BlockCtx& ctx, const Instr& ins, size_t idx, const OpBrx& op, Mask m) {
    if (m == 0) {
      ++w.paths[idx].pc;
      return;
    }
    Lanes scratch;
    const Lanes& index = read_operand(w, ctx, ins, op.a, scratch);
    std::vector<std::pair<size_t, Mask>> groups;   // target pc, lanes going there
    for (Mask rest = m; rest != 0; rest &= rest - 1) {
      const uint32_t lane = first_set(rest);
      // The ISA defines no behaviour for an index outside the list, so none
      // is chosen: the launch stops and names the lane and the value.
      const uint32_t i = static_cast<uint32_t>(index[lane]);
      if (i >= op.targets.size())
        ctx_fail(ins, static_cast<int>(lane), Err::OutOfBounds,
                 "brx.idx index " + std::to_string(i) + " is outside the branch target list '" + op.table +
                     "' of " + std::to_string(op.targets.size()) +
                     " entries; the PTX ISA does not define where it goes");
      const size_t target = op.targets[i];
      auto g = std::find_if(groups.begin(), groups.end(), [&](const auto& e) { return e.first == target; });
      if (g == groups.end()) groups.emplace_back(target, Mask{0}), g = groups.end() - 1;
      g->second |= Mask{1} << lane;
    }
    const size_t here = w.paths[idx].pc;
    if (w.steps - turn_start_ >= kLongTurn)
      for (const auto& g : groups)
        if (g.first <= here) w.yield_now = true;
    const Mask fallthrough = w.paths[idx].mask & ~m;
    if (groups.size() == 1 && fallthrough == 0) {
      w.paths[idx].pc = groups[0].first;
      return;
    }
    ++stats_.divergent_branches;
    w.paths[idx].pc = groups[0].first;
    w.paths[idx].mask = groups[0].second;
    for (size_t k = 1; k < groups.size(); ++k) {
      Path p{groups[k].first, groups[k].second};
      p.issued_at = w.steps;
      w.paths.push_back(p);
    }
    if (fallthrough != 0) {
      Path p{here + 1, fallthrough};
      p.issued_at = w.steps;
      w.paths.push_back(p);
    }
  }

  void exec_ret(Warp& w, const BlockCtx& ctx, const Instr& ins, size_t idx, Mask m) {
    // Inside a device function `ret` means "return to the caller", not "this
    // thread is finished". Retiring the lanes and marking the warp Done there
    // ended the whole thread at the first call that returned -- and because
    // the caller then resumed with its own saved state, the damage showed up
    // later as a warp that had silently stopped executing.
    const bool in_call = !w.frames.empty();
    if (!in_call) w.exited |= m;
    Mask survivors = w.paths[idx].mask & ~m;
    if (survivors == 0) {
      w.paths.erase(w.paths.begin() + static_cast<long>(idx));
    } else {
      // A predicated ret retires some lanes; the rest carry on.
      w.paths[idx].mask = survivors;
      ++w.paths[idx].pc;
    }
    if (w.paths.empty() && !in_call) {
      w.state = Warp::State::Done;
      drain_async_copies(w, ctx, ins);
    }
    if (w.paths.empty() && in_call) return_to_caller(w);
  }

  // ---- fast paths ----
  //
  // The instructions compiled kernels spend their time in -- 32-bit integer
  // arithmetic, mad.lo, mov, setp, and f32/f64 fma and arithmetic -- in the
  // forms that need nothing special: 32-bit registers and immediates, no
  // carry, no division, no rounding mode, no armed fault. Those read the
  // narrow register file directly instead of widening every operand into a
  // 64-lane scratch array, decide the operation once per instruction instead
  // of once per lane, and write the result in place. Every other form returns
  // false and takes the general path below, which computes the same bits;
  // tests/unit/test_fastpath.cpp holds the two to each other.

  bool fast_path(Warp& w, const BlockCtx& ctx, const Instr& ins, Mask m) {
    switch (ins.op.index()) {
      case op_index<OpIntBin>(): {
        const auto& op = std::get<OpIntBin>(ins.op);
        return op.ty.bits == 64 ? fast_int_bin64(w, ctx, ins, op, m) : fast_int_bin(w, ins, op, m);
      }
      case op_index<OpMadLo>(): return fast_mad_lo(w, ins, std::get<OpMadLo>(ins.op), m);
      case op_index<OpMov>(): return fast_mov(w, ins, std::get<OpMov>(ins.op), m);
      case op_index<OpSetp>(): return fast_setp(w, ins, std::get<OpSetp>(ins.op), m);
      case op_index<OpFma>(): return fast_fma(w, ctx, ins, std::get<OpFma>(ins.op), m);
      case op_index<OpFloatBin>(): return fast_float_bin(w, ins, std::get<OpFloatBin>(ins.op), m);
      case op_index<OpF16x2Fma>(): return fast_f16x2_fma(w, ins, std::get<OpF16x2Fma>(ins.op), m);
      case op_index<OpCvt>(): return fast_cvt_int(w, ctx, ins, std::get<OpCvt>(ins.op), m);
      case op_index<OpMulWide>(): return fast_mul_wide(w, ins, std::get<OpMulWide>(ins.op), m);
      default: return false;
    }
  }

  bool fast_int_bin(Warp& w, const Instr& ins, const OpIntBin& op, Mask m) {
    if (op.carry_in || op.carry_out || op.sat || op.ty.bits != 32 || op.dst.wide || !narrow_operand(op.a) ||
        !narrow_operand(op.b) || op.op == IntBinOp::Div || op.op == IntBinOp::Rem)
      return false;
    Lanes32 sa, sb;
    const Lanes32& a = read_narrow(w, ins, op.a, sa);
    const Lanes32& b = read_narrow(w, ins, op.b, sb);
    uint32_t* d = dst32(w, op.dst.id, m);
    const bool sig = op.ty.is_signed();
    auto run = [&](auto f) { for_active(m, W_, [&](uint32_t l) { d[l] = f(a[l], b[l]); }); };
    switch (op.op) {
      case IntBinOp::Add: run([](uint32_t x, uint32_t y) { return x + y; }); break;
      case IntBinOp::Sub: run([](uint32_t x, uint32_t y) { return x - y; }); break;
      case IntBinOp::Mul: run([](uint32_t x, uint32_t y) { return x * y; }); break;
      case IntBinOp::And: run([](uint32_t x, uint32_t y) { return x & y; }); break;
      case IntBinOp::Or: run([](uint32_t x, uint32_t y) { return x | y; }); break;
      case IntBinOp::Xor: run([](uint32_t x, uint32_t y) { return x ^ y; }); break;
      case IntBinOp::Min:
        if (sig) run([](uint32_t x, uint32_t y) {
            return static_cast<uint32_t>(std::min(static_cast<int32_t>(x), static_cast<int32_t>(y)));
          });
        else run([](uint32_t x, uint32_t y) { return std::min(x, y); });
        break;
      case IntBinOp::Max:
        if (sig) run([](uint32_t x, uint32_t y) {
            return static_cast<uint32_t>(std::max(static_cast<int32_t>(x), static_cast<int32_t>(y)));
          });
        else run([](uint32_t x, uint32_t y) { return std::max(x, y); });
        break;
      // PTX clamps shift amounts: past the width, all bits shift out (or,
      // for a signed right shift, the sign fills).
      case IntBinOp::Shl: run([](uint32_t x, uint32_t y) { return y >= 32 ? 0u : x << y; }); break;
      case IntBinOp::Shr:
        if (sig) run([](uint32_t x, uint32_t y) {
            const int32_t v = static_cast<int32_t>(x);
            return static_cast<uint32_t>(y >= 32 ? (v < 0 ? -1 : 0) : v >> y);
          });
        else run([](uint32_t x, uint32_t y) { return y >= 32 ? 0u : x >> y; });
        break;
      default: return false;   // unreachable: Div and Rem were sent to the general path
    }
    w.written32[op.dst.id] = 1;
    return true;
  }

  bool fast_mad_lo(Warp& w, const Instr& ins, const OpMadLo& op, Mask m) {
    if (op.carry_in || op.carry_out || op.ty.bits != 32 || op.dst.wide || !narrow_operand(op.a) ||
        !narrow_operand(op.b) || !narrow_operand(op.c))
      return false;
    Lanes32 sa, sb, sc;
    const Lanes32& a = read_narrow(w, ins, op.a, sa);
    const Lanes32& b = read_narrow(w, ins, op.b, sb);
    const Lanes32& c = read_narrow(w, ins, op.c, sc);
    uint32_t* d = dst32(w, op.dst.id, m);
    for_active(m, W_, [&](uint32_t l) { d[l] = a[l] * b[l] + c[l]; });
    w.written32[op.dst.id] = 1;
    return true;
  }

  bool fast_mov(Warp& w, const Instr& ins, const OpMov& op, Mask m) {
    if (op.ty.bits != 32 || op.dst.wide || !narrow_operand(op.src)) return false;
    Lanes32 ss;
    const Lanes32& v = read_narrow(w, ins, op.src, ss);
    uint32_t* d = dst32(w, op.dst.id, m);
    for_active(m, W_, [&](uint32_t l) { d[l] = v[l]; });
    w.written32[op.dst.id] = 1;
    return true;
  }

  bool fast_setp(Warp& w, const Instr& ins, const OpSetp& op, Mask m) {
    if (op.ty.bits != 32 || op.ty.is_bfloat() || !narrow_operand(op.a) || !narrow_operand(op.b) ||
        op.has_bop || op.has_q || op.packed || op.ftz)
      return false;
    Lanes32 sa, sb;
    const Lanes32& a = read_narrow(w, ins, op.a, sa);
    const Lanes32& b = read_narrow(w, ins, op.b, sb);
    Mask res = 0;
    auto run = [&](auto f) {
      for_active(m, W_, [&](uint32_t l) { res |= Mask{f(a[l], b[l]) ? 1u : 0u} << l; });
    };
    if (op.ty.is_float()) {
      const CmpOp cmp = op.cmp;
      run([cmp](uint32_t x, uint32_t y) { return compare_float(cmp, f32(x), f32(y)); });
    } else {
      auto by = [&](auto cast) {
        switch (op.cmp) {
          case CmpOp::Eq: run([&](uint32_t x, uint32_t y) { return cast(x) == cast(y); }); return true;
          case CmpOp::Ne: run([&](uint32_t x, uint32_t y) { return cast(x) != cast(y); }); return true;
          case CmpOp::Lt: run([&](uint32_t x, uint32_t y) { return cast(x) < cast(y); }); return true;
          case CmpOp::Le: run([&](uint32_t x, uint32_t y) { return cast(x) <= cast(y); }); return true;
          case CmpOp::Gt: run([&](uint32_t x, uint32_t y) { return cast(x) > cast(y); }); return true;
          case CmpOp::Ge: run([&](uint32_t x, uint32_t y) { return cast(x) >= cast(y); }); return true;
          default: return false;
        }
      };
      const bool done = op.ty.is_signed() ? by([](uint32_t v) { return static_cast<int32_t>(v); })
                                          : by([](uint32_t v) { return v; });
      if (!done) return false;
    }
    Mask& p = pred_slot(w, op.dst);
    p = (p & ~m) | (res & m);
    return true;
  }

  bool fast_fma(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpFma& op, Mask m) {
    if (mem_.alu_fault_armed() || op.round != FRound::Nearest || op.sat || op.ftz || op.neg_ab || op.neg_c)
      return false;
    if (op.ty.bits == 32) {
      if (op.dst.wide || !narrow_operand(op.a) || !narrow_operand(op.b) || !narrow_operand(op.c))
        return false;
      Lanes32 sa, sb, sc;
      const Lanes32& a = read_narrow(w, ins, op.a, sa);
      const Lanes32& b = read_narrow(w, ins, op.b, sb);
      const Lanes32& c = read_narrow(w, ins, op.c, sc);
      uint32_t* d = dst32(w, op.dst.id, m);
#if defined(__x86_64__) && defined(__GNUC__)
      if (g_hw_fma) {
        fma_lanes_f32_hw(d, a.data(), b.data(), c.data(), m, W_);
        // A NaN result is the canonical one (see canon32).
        for_active(m, W_, [&](uint32_t l) {
          if ((d[l] & 0x7F800000u) == 0x7F800000u && (d[l] & 0x7FFFFFu)) d[l] = 0x7FFFFFFFu;
        });
        w.written32[op.dst.id] = 1;
        return true;
      }
#endif
      for_active(m, W_, [&](uint32_t l) { d[l] = canon32(std::fma(f32(a[l]), f32(b[l]), f32(c[l]))); });
      w.written32[op.dst.id] = 1;
      return true;
    }
    if (op.ty.bits != 64 || !op.dst.wide) return false;
    Lanes sa, sb, sc;
    const Lanes& a = read_wide(w, ctx, ins, op.a, sa);
    const Lanes& b = read_wide(w, ctx, ins, op.b, sb);
    const Lanes& c = read_wide(w, ctx, ins, op.c, sc);
    uint64_t* d = dst64(w, op.dst.id, m);
#if defined(__x86_64__) && defined(__GNUC__)
    if (g_hw_fma) {
      fma_lanes_f64_hw(d, a.data(), b.data(), c.data(), m, W_);
      w.written64[op.dst.id] = 1;
      return true;
    }
#endif
    for_active(m, W_, [&](uint32_t l) { d[l] = f64bits(std::fma(f64(a[l]), f64(b[l]), f64(c[l]))); });
    w.written64[op.dst.id] = 1;
    return true;
  }

  bool fast_float_bin(Warp& w, const Instr& ins, const OpFloatBin& op, Mask m) {
    if (op.round != FRound::Nearest || op.nan_propagate || op.sat || op.xorsign_abs || op.ftz || op.approx || op.ty.bits != 32 || !op.ty.is_float() ||
        op.dst.wide || !narrow_operand(op.a) || !narrow_operand(op.b) || mem_.alu_fault_armed())
      return false;
    Lanes32 sa, sb;
    const Lanes32& a = read_narrow(w, ins, op.a, sa);
    const Lanes32& b = read_narrow(w, ins, op.b, sb);
    uint32_t* d = dst32(w, op.dst.id, m);
    auto run = [&](auto f) {
      for_active(m, W_, [&](uint32_t l) { d[l] = canon32(f(f32(a[l]), f32(b[l]))); });
    };
    switch (op.op) {
      case FloatBinOp::Add: run([](float x, float y) { return x + y; }); break;
      case FloatBinOp::Sub: run([](float x, float y) { return x - y; }); break;
      case FloatBinOp::Mul: run([](float x, float y) { return x * y; }); break;
      case FloatBinOp::Div: run([](float x, float y) { return x / y; }); break;
      case FloatBinOp::Min: for_active(m, W_, [&](uint32_t l) { d[l] = ptx_min32(a[l], b[l]); }); break;
      case FloatBinOp::Max: for_active(m, W_, [&](uint32_t l) { d[l] = ptx_max32(a[l], b[l]); }); break;
    }
    w.written32[op.dst.id] = 1;
    return true;
  }

  // 64-bit integer arithmetic: the address computations around every memory
  // access. 64-bit registers are read in place, never widened.
  bool fast_int_bin64(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpIntBin& op, Mask m) {
    if (op.carry_in || op.carry_out || op.sat || !op.dst.wide || op.op == IntBinOp::Div || op.op == IntBinOp::Rem)
      return false;
    Lanes sa, sb;
    const Lanes& a = read_operand(w, ctx, ins, op.a, sa);
    const Lanes& b = read_operand(w, ctx, ins, op.b, sb);
    uint64_t* d = dst64(w, op.dst.id, m);
    const bool sig = op.ty.is_signed();
    auto run = [&](auto f) { for_active(m, W_, [&](uint32_t l) { d[l] = f(a[l], b[l]); }); };
    switch (op.op) {
      case IntBinOp::Add: run([](uint64_t x, uint64_t y) { return x + y; }); break;
      case IntBinOp::Sub: run([](uint64_t x, uint64_t y) { return x - y; }); break;
      case IntBinOp::Mul: run([](uint64_t x, uint64_t y) { return x * y; }); break;
      case IntBinOp::And: run([](uint64_t x, uint64_t y) { return x & y; }); break;
      case IntBinOp::Or: run([](uint64_t x, uint64_t y) { return x | y; }); break;
      case IntBinOp::Xor: run([](uint64_t x, uint64_t y) { return x ^ y; }); break;
      case IntBinOp::Min:
        if (sig) run([](uint64_t x, uint64_t y) {
            return static_cast<uint64_t>(std::min(static_cast<int64_t>(x), static_cast<int64_t>(y)));
          });
        else run([](uint64_t x, uint64_t y) { return std::min(x, y); });
        break;
      case IntBinOp::Max:
        if (sig) run([](uint64_t x, uint64_t y) {
            return static_cast<uint64_t>(std::max(static_cast<int64_t>(x), static_cast<int64_t>(y)));
          });
        else run([](uint64_t x, uint64_t y) { return std::max(x, y); });
        break;
      // The amount is a .u32, as in the general path.
      case IntBinOp::Shl: run([](uint64_t x, uint64_t y) { y = static_cast<uint32_t>(y); return y >= 64 ? 0 : x << y; }); break;
      case IntBinOp::Shr:
        if (sig) run([](uint64_t x, uint64_t y) {
            const int64_t v = static_cast<int64_t>(x);
            y = static_cast<uint32_t>(y);
            return static_cast<uint64_t>(y >= 64 ? (v < 0 ? -1 : 0) : v >> y);
          });
        else run([](uint64_t x, uint64_t y) { y = static_cast<uint32_t>(y); return y >= 64 ? 0 : x >> y; });
        break;
      default: return false;
    }
    w.written64[op.dst.id] = 1;
    return true;
  }

  // cvt between integer types: sign- or zero-extend from the source width,
  // then truncate to the destination's -- the widenings and narrowings that
  // turn a 32-bit index into a 64-bit address and back.
  bool fast_cvt_int(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpCvt& op, Mask m) {
    const Type& st = op.src_ty;
    const Type& dt = op.dst_ty;
    if (st.is_real() || dt.is_real() || st.kind == Type::Kind::Pred || dt.kind == Type::Kind::Pred ||
        op.round != Round::None || op.sat || (dt.is_signed() && dt.bits < 32))
      return false;
    Lanes sv;
    const Lanes& v = read_operand(w, ctx, ins, op.src, sv);
    const uint64_t src_mask = st.bits >= 64 ? ~uint64_t{0} : (uint64_t{1} << st.bits) - 1;
    const uint64_t sign = st.bits >= 64 ? 0 : uint64_t{1} << (st.bits - 1);
    const bool extend = st.is_signed() && st.bits < 64;
    auto value = [&](uint64_t in) {
      uint64_t x = in & src_mask;
      if (extend && (x & sign)) x |= ~src_mask;
      return x;
    };
    if (op.dst.wide) {
      const uint64_t keep = dt.bits >= 64 ? ~uint64_t{0} : (uint64_t{1} << dt.bits) - 1;
      uint64_t* d = dst64(w, op.dst.id, m);
      for_active(m, W_, [&](uint32_t l) { d[l] = value(v[l]) & keep; });
      w.written64[op.dst.id] = 1;
    } else {
      const uint32_t keep = dt.bits >= 32 ? ~0u : (1u << dt.bits) - 1;
      uint32_t* d = dst32(w, op.dst.id, m);
      for_active(m, W_, [&](uint32_t l) { d[l] = static_cast<uint32_t>(value(v[l])) & keep; });
      w.written32[op.dst.id] = 1;
    }
    return true;
  }

  // mul.wide from 32 bits: the full 64-bit product of two 32-bit values.
  bool fast_mul_wide(Warp& w, const Instr& ins, const OpMulWide& op, Mask m) {
    if (op.src_bits != 32 || !op.dst.wide || !narrow_operand(op.a) || !narrow_operand(op.b))
      return false;
    Lanes32 sa, sb;
    const Lanes32& a = read_narrow(w, ins, op.a, sa);
    const Lanes32& b = read_narrow(w, ins, op.b, sb);
    uint64_t* d = dst64(w, op.dst.id, m);
    if (op.is_signed)
      for_active(m, W_, [&](uint32_t l) {
        d[l] = static_cast<uint64_t>(int64_t{static_cast<int32_t>(a[l])} * int64_t{static_cast<int32_t>(b[l])});
      });
    else
      for_active(m, W_, [&](uint32_t l) { d[l] = uint64_t{a[l]} * uint64_t{b[l]}; });
    w.written64[op.dst.id] = 1;
    return true;
  }

  // fma on f16 or bf16 halves: the same double-precision fma and rounding as
  // the general path, with the decode from the table and no widening.
  bool fast_f16x2_fma(Warp& w, const Instr& ins, const OpF16x2Fma& op, Mask m) {
    if (op.mods.any() || op.dst.wide || !narrow_operand(op.a) || !narrow_operand(op.b) || !narrow_operand(op.c))
      return false;
    Lanes32 sa, sb, sc;
    const Lanes32& a = read_narrow(w, ins, op.a, sa);
    const Lanes32& b = read_narrow(w, ins, op.b, sb);
    const Lanes32& c = read_narrow(w, ins, op.c, sc);
    uint32_t* d = dst32(w, op.dst.id, m);
    const int halves = op.packed ? 2 : 1;
    if (op.bf16) {
      for_active(m, W_, [&](uint32_t l) {
        uint64_t out = 0;
        for (int h = 0; h < halves; ++h) {
          out |= bf16_fma(bf16_to_double((a[l] >> (16 * h)) & 0xFFFF), bf16_to_double((b[l] >> (16 * h)) & 0xFFFF),
                          bf16_to_double((c[l] >> (16 * h)) & 0xFFFF))
                 << (16 * h);
        }
        d[l] = static_cast<uint32_t>(out);
      });
#if defined(__x86_64__) && defined(__GNUC__)
    } else if (g_hw_fma) {
      f16x2_fma_lanes_hw(d, a.data(), b.data(), c.data(), m, W_, halves);
#endif
    } else {
      for_active(m, W_, [&](uint32_t l) {
        uint64_t out = 0;
        for (int h = 0; h < halves; ++h) {
          const double v = host_fma(kF16ToDouble[(a[l] >> (16 * h)) & 0xFFFF],
                                    kF16ToDouble[(b[l] >> (16 * h)) & 0xFFFF],
                                    kF16ToDouble[(c[l] >> (16 * h)) & 0xFFFF]);
          out |= double_to_f16(v) << (16 * h);
        }
        d[l] = static_cast<uint32_t>(out);
      });
    }
    w.written32[op.dst.id] = 1;
    return true;
  }

  void dispatch(Warp& w, const BlockCtx& ctx, const Instr& ins, Mask m) {
    if (g_fast_path.load(std::memory_order_relaxed) && fast_path(w, ctx, ins, m)) return;
    if (const auto* op = std::get_if<OpMov>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      write_reg(w, op->dst, m, v, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpIntBin>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;  // written for every active lane below
      if (op->carry_in || op->carry_out) {
        exec_carry_add_sub(w, *op, a, b, m, r);
      } else if (op->sat) {
        // add/sub.sat.s32: the true sum, clamped.
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) {
            const int64_t x = static_cast<int32_t>(a[lane]), y = static_cast<int32_t>(b[lane]);
            const int64_t t = op->op == IntBinOp::Sub ? x - y : x + y;
            r[lane] = static_cast<uint32_t>(static_cast<int32_t>(std::clamp<int64_t>(t, INT32_MIN, INT32_MAX)));
          }
      } else {
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) r[lane] = int_bin(op->op, op->ty, a[lane], b[lane], ins);
      }
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (std::holds_alternative<OpNop>(ins.op)) return;
    if (std::holds_alternative<OpFence>(ins.op)) {
      // Blocks run on several host threads; see OpFence.
      std::atomic_thread_fence(std::memory_order_seq_cst);
      return;
    }
    if (const auto* op = std::get_if<OpActiveMask>(&ins.op)) {
      require_warp32(ins, "activemask");
      Lanes r;  // every active lane sees the same membership
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) r[lane] = m;
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpCpAsync>(&ins.op)) {
      exec_cp_async(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpCpAsyncGroup>(&ins.op)) {
      exec_cp_async_group(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpLd>(&ins.op)) {
      // Every element of a vector access: an ld.v4 moves four times the
      // element's bytes, and counting one element undercounted it that much.
      count_memory(op->space, op->ty.bytes() * static_cast<uint32_t>(op->dsts.size()),
                   popcount_mask(m), /*is_store=*/false);
      exec_ld(w, ctx, ins, *op, m);
      // ld.acquire: nothing after it may be seen to happen before it.
      if (op->acquire) std::atomic_thread_fence(std::memory_order_acquire);
      return;
    }
    if (const auto* op = std::get_if<OpSt>(&ins.op)) {
      count_memory(op->space, op->ty.bytes() * static_cast<uint32_t>(op->srcs.size()),
                   popcount_mask(m), /*is_store=*/true);
      // st.release: nothing before it may be seen to happen after it.
      if (op->release) std::atomic_thread_fence(std::memory_order_release);
      exec_st(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpSetp>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Mask t = 0, thi = 0;
      if (op->packed) {
        // f16x2/bf16x2: the low halves decide p, the high halves q.
        auto half = [&](uint64_t v, int h) {
          uint64_t x = (v >> (16 * h)) & 0xFFFF;
          if (op->ftz && !op->ty.is_bfloat()) x = half_ftz(x);
          return op->ty.is_bfloat() ? bf16_to_double(x) : f16_to_double(x);
        };
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) {
            if (compare_float(op->cmp, half(a[lane], 0), half(b[lane], 0))) t |= Mask{1} << lane;
            if (compare_float(op->cmp, half(a[lane], 1), half(b[lane], 1))) thi |= Mask{1} << lane;
          }
      } else {
        for (uint32_t lane = 0; lane < W_; ++lane)
          if ((m & (Mask{1} << lane)) &&
              compare(op->cmp, op->ty, cmp_ftz(op->ftz, op->ty, a[lane]), cmp_ftz(op->ftz, op->ty, b[lane])))
            t |= Mask{1} << lane;
      }
      // With a boolean operation: p = t bop c and q = !t bop c. c is read
      // before either destination is written, since it may be one of them.
      Mask pv = t, qv = ~t;
      if (op->has_bop) {
        Mask c = read_pred(w, ins, op->c);
        if (op->negate_c) c = ~c;
        auto combine = [&](Mask x) {
          return op->bop == PredBinOp::And ? (x & c) : op->bop == PredBinOp::Or ? (x | c) : (x ^ c);
        };
        pv = combine(t);
        qv = combine(op->packed ? thi : ~t);
      } else if (op->packed) {
        qv = thi;
      }
      Mask& p = pred_slot(w, op->dst);
      p = (p & ~m) | (pv & m);
      if (op->has_q) {
        Mask& q = pred_slot(w, op->dst2);
        q = (q & ~m) | (qv & m);
      }
      return;
    }
    if (const auto* op = std::get_if<OpFloatBin>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;  // written for every active lane below
      // An explicit rounding mode changes the result, so it has to be applied
      // rather than assumed to be round-to-nearest: quantization kernels use
      // .rz specifically because truncation is what they want. Set the host
      // mode around the arithmetic and put it back, so nothing else observes
      // the change.
      // Only an explicit mode touches the environment: fegetround is a libc call,
      // and round-to-nearest is what nearly every instruction asks for.
      const int prev_round = op->round == FRound::Nearest ? 0 : std::fegetround();
      if (op->round != FRound::Nearest) g_directed_rounding.fetch_add(1, std::memory_order_relaxed);
      switch (op->round) {
        case FRound::Zero: std::fesetround(FE_TOWARDZERO); break;
        case FRound::MinusInf: std::fesetround(FE_DOWNWARD); break;
        case FRound::PlusInf: std::fesetround(FE_UPWARD); break;
        case FRound::Nearest: break;
      }
      const uint64_t sign = uint64_t{1} << (op->ty.bits - 1);
      // .ftz (f32): subnormal inputs and results to signed zero.
      auto flush = [&](uint64_t x) {
        return op->ftz && (x & 0x7F800000u) == 0 ? (x & 0x80000000u) : x;
      };
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          const uint64_t av = flush(a[lane]), bvv = flush(b[lane]);
          if (op->xorsign_abs) {
            // The smaller/larger magnitude, signed sign(a) ^ sign(b); a NaN
            // operand is ignored as in plain min/max (unless .NaN).
            uint64_t v = float_bin(op->op, op->ty, av & ~sign, bvv & ~sign, op->nan_propagate);
            if (!float_is_nan(op->ty, v)) v = (v & ~sign) | ((av ^ bvv) & sign);
            r[lane] = v;
          } else if (op->approx) {
            r[lane] = div_approx32(static_cast<uint32_t>(av), static_cast<uint32_t>(bvv));
          } else {
            r[lane] = float_bin(op->op, op->ty, av, bvv, op->nan_propagate);
          }
          r[lane] = flush(r[lane]);
          if (op->sat) r[lane] = float_sat(op->ty, r[lane]);
        }
      if (op->round != FRound::Nearest) {
        std::fesetround(prev_round);
        g_directed_rounding.fetch_sub(1, std::memory_order_relaxed);
      }
      alu_fault(r, m, op->ty.bits);
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpMadLo>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r;  // written for every active lane below
      if (op->carry_in || op->carry_out) {
        Lanes prod;
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) prod[lane] = a[lane] * b[lane];
        exec_carry_mad(w, op->ty.bits, op->carry_in, op->carry_out, prod, c, m, r);
      } else {
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) r[lane] = a[lane] * b[lane] + c[lane];
      }
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpFma>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r;  // written for every active lane below
      // The rounding mode is applied as float arithmetic's is: set around the
      // fma (the host's fma honours it) and put back.
      const int prev_round = op->round == FRound::Nearest ? 0 : std::fegetround();
      if (op->round != FRound::Nearest) g_directed_rounding.fetch_add(1, std::memory_order_relaxed);
      switch (op->round) {
        case FRound::Zero: std::fesetround(FE_TOWARDZERO); break;
        case FRound::MinusInf: std::fesetround(FE_DOWNWARD); break;
        case FRound::PlusInf: std::fesetround(FE_UPWARD); break;
        case FRound::Nearest: break;
      }
      // .ftz: subnormal f32 inputs and results to signed zero.
      auto ftz = [&](float v) { return op->ftz && std::fpclassify(v) == FP_SUBNORMAL ? std::copysign(0.0f, v) : v; };
      // A contracted sub negates the product or the addend.
      const double sa = op->neg_ab ? -1.0 : 1.0, sc = op->neg_c ? -1.0 : 1.0;
      if (op->ty.bits == 32)
        for_active(m, W_, [&](uint32_t l) {
          const float av = op->neg_ab ? -f32(a[l]) : f32(a[l]), cv = op->neg_c ? -f32(c[l]) : f32(c[l]);
          r[l] = canon32(ftz(std::fma(ftz(av), ftz(f32(b[l])), ftz(cv))));
        });
      else
        for_active(m, W_, [&](uint32_t l) { r[l] = f64bits(std::fma(sa * f64(a[l]), f64(b[l]), sc * f64(c[l]))); });
      if (op->round != FRound::Nearest) {
        std::fesetround(prev_round);
        g_directed_rounding.fetch_sub(1, std::memory_order_relaxed);
      }
      if (op->sat) for_active(m, W_, [&](uint32_t l) { r[l] = float_sat(op->ty, r[l]); });
      alu_fault(r, m, op->ty.bits);
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpMulWide>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          if (op->src_bits == 16) {
            if (op->is_signed)
              r[lane] = static_cast<uint64_t>(static_cast<uint32_t>(
                  int32_t{static_cast<int16_t>(a[lane])} * int32_t{static_cast<int16_t>(b[lane])}));
            else
              r[lane] = uint32_t{static_cast<uint16_t>(a[lane])} *
                        uint32_t{static_cast<uint16_t>(b[lane])};
          } else if (op->is_signed)
            r[lane] = static_cast<uint64_t>(int64_t{static_cast<int32_t>(a[lane])} *
                                            int64_t{static_cast<int32_t>(b[lane])});
          else
            r[lane] = uint64_t{static_cast<uint32_t>(a[lane])} * uint64_t{static_cast<uint32_t>(b[lane])};
        }
      write_reg(w, op->dst, m, r, op->src_bits * 2);  // .wide doubles the width
      return;
    }
    if (const auto* op = std::get_if<OpSet>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;
      const bool dfloat = op->dty.is_real();
      const bool sbf = op->sty.is_bfloat();
      Mask cpred = 0;
      if (op->has_bop) {
        cpred = read_pred(w, ins, op->c);
        if (op->negate_c) cpred = ~cpred;
      }
      for (uint32_t lane = 0; lane < W_; ++lane) {
        if (!(m & (Mask{1} << lane))) continue;
        const bool cl = (cpred >> lane) & 1;
        const int halves = op->packed ? 2 : 1;
        uint64_t out = 0;
        for (int h = 0; h < halves; ++h) {
          double x, y;
          if (op->packed) {
            uint64_t ax = (a[lane] >> (16 * h)) & 0xFFFF;
            uint64_t bx = (b[lane] >> (16 * h)) & 0xFFFF;
            if (op->ftz && !sbf) { ax = half_ftz(ax); bx = half_ftz(bx); }
            x = sbf ? bf16_to_double(ax) : f16_to_double(ax);
            y = sbf ? bf16_to_double(bx) : f16_to_double(bx);
          } else if (op->sty.is_real()) {
            x = op->sty.bits == 64 ? f64(a[lane])
              : op->sty.bits == 32 ? static_cast<double>(f32(a[lane]))
              : sbf                ? bf16_to_double(a[lane] & 0xFFFF)
                                   : f16_to_double(a[lane] & 0xFFFF);
            y = op->sty.bits == 64 ? f64(b[lane])
              : op->sty.bits == 32 ? static_cast<double>(f32(b[lane]))
              : sbf                ? bf16_to_double(b[lane] & 0xFFFF)
                                   : f16_to_double(b[lane] & 0xFFFF);
          } else {
            x = y = 0;  // integer compare below
          }
          bool t;
          if (op->packed) {
            t = compare_float(op->cmp, x, y);
          } else {
            // The scalar path reuses setp's comparator, which already handles
            // the signed/unsigned and NaN-aware cases from the source type.
            t = compare(op->cmp, op->sty, cmp_ftz(op->ftz, op->sty, a[lane]), cmp_ftz(op->ftz, op->sty, b[lane]));
          }
          (void)x; (void)y;
          if (op->has_bop)
            t = op->bop == PredBinOp::And ? (t && cl) : op->bop == PredBinOp::Or ? (t || cl) : (t != cl);
          // True's encoding comes from the *destination* type: an integer
          // destination gets all ones, a float destination gets 1.0. Writing 1
          // into an integer destination is the easy mistake, and it makes
          // every use of the result as a mask select a single bit.
          if (op->packed_int_dst) {
            out |= (t ? 0xFFFFull : 0ull) << (16 * h);
          } else if (op->packed) {
            const uint64_t one = op->dty.is_bfloat() ? double_to_bf16(1.0) : double_to_f16(1.0);
            out |= (t ? one : 0ull) << (16 * h);
          } else if (dfloat) {
            out = op->dty.bits == 64 ? f64bits(t ? 1.0 : 0.0)
                : op->dty.bits == 32 ? f32bits(t ? 1.0f : 0.0f)
                : op->dty.is_bfloat() ? double_to_bf16(t ? 1.0 : 0.0)
                                      : double_to_f16(t ? 1.0 : 0.0);
          } else {
            out = t ? mask_to_bits(~0ull, op->dty.bits) : 0ull;
          }
        }
        r[lane] = out;
      }
      write_reg(w, op->dst, m, r, op->packed ? 32u : (op->dty.bits < 32 ? 32u : op->dty.bits));
      return;
    }
    if (const auto* op = std::get_if<OpSelp>(&ins.op)) {
      Mask p = read_pred(w, ins, op->pred);
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) r[lane] = (p & (Mask{1} << lane)) ? a[lane] : b[lane];
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpCvta>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      // A parameter's address is already its generic one here: taking it
      // ("mov.b64 %rd, param") yields the parameter window address, and
      // converting that again added the window base twice.
      const uint64_t base = op->space == Space::Param ? 0 : space_base(op->space);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane))
          r[lane] = op->to_space ? v[lane] - base : v[lane] + base;
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpMadWide>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          uint64_t prod;
          if (op->is_signed)
            prod = static_cast<uint64_t>(int64_t{static_cast<int32_t>(a[lane])} *
                                         int64_t{static_cast<int32_t>(b[lane])});
          else
            prod = uint64_t{static_cast<uint32_t>(a[lane])} * uint64_t{static_cast<uint32_t>(b[lane])};
          r[lane] = prod + c[lane];
        }
      write_reg(w, op->dst, m, r, 64);
      return;
    }
    if (const auto* op = std::get_if<OpCvt>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r;  // written for every active lane below
      // .rz/.rm/.rp into a float destination round in that mode, the way
      // float_bin does it: set the host mode around the conversions. (Into an
      // integer the mode is applied by round_int instead.)
      int host_mode = FE_TONEAREST;
      if (op->dst_ty.is_real())
        switch (op->round) {
          case Round::Rz: host_mode = FE_TOWARDZERO; break;
          case Round::Rm: host_mode = FE_DOWNWARD; break;
          case Round::Rp: host_mode = FE_UPWARD; break;
          default: break;
        }
      const int prev_round = host_mode == FE_TONEAREST ? 0 : std::fegetround();
      if (host_mode != FE_TONEAREST) {
        g_directed_rounding.fetch_add(1, std::memory_order_relaxed);
        std::fesetround(host_mode);
      }
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) r[lane] = convert(op, v[lane]);
      if (host_mode != FE_TONEAREST) {
        std::fesetround(prev_round);
        g_directed_rounding.fetch_sub(1, std::memory_order_relaxed);
      }
      // A signed integer type narrower than the register is sign-extended
      // into it (PTX's rule for destination operands): cvt.s8.s32 of 0xFF
      // leaves 0xFFFFFFFF.
      const Type& dt = op->dst_ty;
      if (dt.is_signed() && !dt.is_real() && dt.bits < 32) {
        const uint64_t sign = uint64_t{1} << (dt.bits - 1), mask = (sign << 1) - 1;
        for (uint32_t lane = 0; lane < W_; ++lane)
          if ((m & (Mask{1} << lane)) && (r[lane] & sign)) r[lane] |= ~mask;
        write_reg(w, op->dst, m, r, op->dst.wide ? 64 : 32);
        return;
      }
      write_reg(w, op->dst, m, r, dt.bits);
      return;
    }
    if (const auto* op = std::get_if<OpIsTypep>(&ins.op)) {
      // istypep (9.7.12.6): no handle here is a .texref/.samplerref/.surfref
      // variable -- those went with CUDA 12's texture references -- and on an
      // RTX 3060 it is false for texture and surface objects and any other
      // value alike, so it is false.
      Mask& p = pred_slot(w, op->dst);
      p &= ~m;
      return;
    }
    if (const auto* op = std::get_if<OpStBulk>(&ins.op)) {
      exec_st_bulk(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpAtom>(&ins.op)) {
      exec_atom(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpMulHi>(&ins.op)) {
      Lanes _s_a; const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b; const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) r[lane] = mul_hi(op->ty, a[lane], b[lane]);
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpMovPack>(&ins.op)) {
      uint32_t n = static_cast<uint32_t>(op->srcs.size());
      uint32_t piece = op->ty.bits / n;
      LaneSet vals(n);
      size_t vi = 0;
      for (const auto& src : op->srcs) {
        Lanes tmp;
        vals[vi++] = read_operand(w, ctx, ins, src, tmp);
      }
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          uint64_t out = 0;
          for (uint32_t i = 0; i < n; ++i)
            out |= mask_to_bits(vals[i][lane], piece) << (piece * i);
          r[lane] = out;
        }
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpMovUnpack>(&ins.op)) {
      uint32_t n = static_cast<uint32_t>(op->dsts.size());
      uint32_t piece = op->ty.bits / n;
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      for (uint32_t i = 0; i < n; ++i) {
        if (op->dsts[i].id == kNoReg) continue;   // `_`
        Lanes r;  // written for every active lane below
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) r[lane] = mask_to_bits(v[lane] >> (piece * i), piece);
        write_reg(w, op->dsts[i], m, r, piece);
      }
      return;
    }
    if (const auto* op = std::get_if<OpNot>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane))
          r[lane] = op->logical ? (mask_to_bits(v[lane], op->ty.bits) == 0 ? 1u : 0u) : ~v[lane];
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpNeg>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          if (op->ty.kind == Type::Kind::F)
            r[lane] = float_sign_op(op->ty.bits, v[lane], op->ftz, /*negate=*/true);
          else
            r[lane] = static_cast<uint64_t>(-static_cast<int64_t>(v[lane]));
        }
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpMovPred>(&ins.op)) {
      Mask& p = pred_slot(w, op->dst);
      if (const auto* imm = std::get_if<ImmInt>(&op->src)) {
        // A constant applies to every active lane; inactive lanes keep theirs.
        if (imm->value != 0) p |= m;
        else p &= ~m;
      } else if (const auto* r = std::get_if<RegOperand>(&op->src)) {
        const Mask src = w.preds[pred_index(r->reg)];
        p = (p & ~m) | (src & m);
      } else if (std::holds_alternative<SregOperand>(op->src)) {
        // A predicate special register (%is_explicit_cluster).
        Lanes _s_v;
        const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) p = v[lane] ? (p | (Mask{1} << lane)) : (p & ~(Mask{1} << lane));
      } else {
        ctx_fail(ins, -1, Err::UnsupportedPtx,
                 "mov.pred source must be an immediate, a predicate register or a predicate special register");
      }
      return;
    }
    if (const auto* op = std::get_if<OpLdMatrix>(&ins.op)) {
      exec_ldmatrix(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpStMatrix>(&ins.op)) {
      exec_stmatrix(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpMovMatrix>(&ins.op)) {
      exec_movmatrix(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpPackedInt>(&ins.op)) {
      Lanes _s_a, _s_b;
      const Lanes a = read_operand(w, ctx, ins, op->a, _s_a);
      const Lanes b = op->kind == PackedIntOp::Neg ? Lanes{} : Lanes(read_operand(w, ctx, ins, op->b, _s_b));
      Lanes r;
      const uint32_t bits = op->lane_bits, n = 32 / bits;
      const uint32_t mask = (1u << bits) - 1;
      const int64_t lo = op->is_signed ? -(int64_t{1} << (bits - 1)) : 0,
                    hi = op->is_signed ? (int64_t{1} << (bits - 1)) - 1 : (int64_t{1} << bits) - 1;
      for (uint32_t lane = 0; lane < W_; ++lane) {
        if (!(m & (Mask{1} << lane))) continue;
        uint32_t out = 0;
        for (uint32_t k = 0; k < n; ++k) {
          const auto part = [&](uint64_t v) -> int64_t {
            const uint32_t u = static_cast<uint32_t>(v >> (bits * k)) & mask;
            return op->is_signed ? static_cast<int64_t>(static_cast<int32_t>(u << (32 - bits)) >> (32 - bits)) : u;
          };
          const int64_t x = part(a[lane]), y = op->kind == PackedIntOp::Neg ? 0 : part(b[lane]);
          int64_t v = 0;
          switch (op->kind) {
            case PackedIntOp::Add: v = x + y; break;
            case PackedIntOp::Sub: v = x - y; break;
            case PackedIntOp::Neg: v = -x; break;
            case PackedIntOp::Min: v = std::min(x, y); break;
            case PackedIntOp::Max: v = std::max(x, y); break;
            case PackedIntOp::Set: {
              bool c = false;
              switch (op->cmp) {
                case CmpOp::Eq: c = x == y; break;
                case CmpOp::Ne: c = x != y; break;
                case CmpOp::Lt: c = x < y; break;
                case CmpOp::Le: c = x <= y; break;
                case CmpOp::Gt: c = x > y; break;
                case CmpOp::Ge: c = x >= y; break;
                default: break;
              }
              v = c ? mask : 0;
              break;
            }
          }
          if (op->sat) v = std::clamp(v, lo, hi);
          if (op->relu && v < 0) v = 0;
          out |= (static_cast<uint32_t>(v) & mask) << (bits * k);
        }
        r[lane] = out;
      }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpF8x4Arith>(&ins.op)) {
      exec_f8x4_arith(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpSpCompress>(&ins.op)) {
      exec_spcompress(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpSpDecompress>(&ins.op)) {
      exec_spdecompress(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpWgmma>(&ins.op)) {
      require_warp32(ins, "wgmma");
      exec_wgmma(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpTcgen05>(&ins.op)) {
      require_warp32(ins, "tcgen05");
      exec_tcgen05(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpClc>(&ins.op)) {
      exec_clc(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpMma>(&ins.op)) {
      require_warp32(ins, "mma.sync");
      exec_mma(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpRedux>(&ins.op)) {
      require_warp32(ins, "redux.sync");
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->src, _s_a);
      // Every participating lane contributes and every one receives the result.
      // The active mask is the membership: a lane not executing this
      // instruction is not in the warp's reduction.
      bool first = true;
      uint64_t acc = 0;
      for (uint32_t lane = 0; lane < W_; ++lane) {
        if (!(m & (Mask{1} << lane))) continue;
        const uint64_t v = a[lane];
        if (first) { acc = v; first = false; continue; }
        switch (op->op) {
          case ReduxOp::Add: acc = acc + v; break;
          case ReduxOp::And: acc = acc & v; break;
          case ReduxOp::Or: acc = acc | v; break;
          case ReduxOp::Xor: acc = acc ^ v; break;
          case ReduxOp::Min:
            acc = op->ty.is_signed()
                      ? static_cast<uint64_t>(std::min(narrow_s(acc, op->ty.bits),
                                                       narrow_s(v, op->ty.bits)))
                      : std::min(narrow_u(acc, op->ty.bits), narrow_u(v, op->ty.bits));
            break;
          case ReduxOp::Max:
            acc = op->ty.is_signed()
                      ? static_cast<uint64_t>(std::max(narrow_s(acc, op->ty.bits),
                                                       narrow_s(v, op->ty.bits)))
                      : std::max(narrow_u(acc, op->ty.bits), narrow_u(v, op->ty.bits));
            break;
        }
      }
      Lanes r;
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) r[lane] = acc;
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpCvtPack>(&ins.op)) {
      Lanes _s_a, _s_b, _s_c;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      const Lanes* c = op->has_c ? &read_operand(w, ctx, ins, op->c, _s_c) : nullptr;
      const uint32_t bits = op->bits;
      const int64_t lo = op->is_signed ? -(int64_t{1} << (bits - 1)) : 0;
      const int64_t hi = op->is_signed ? (int64_t{1} << (bits - 1)) - 1 : (int64_t{1} << bits) - 1;
      const uint32_t field = static_cast<uint32_t>((uint64_t{1} << bits) - 1);
      auto sat = [&](uint64_t v) {
        const int64_t x = std::clamp<int64_t>(static_cast<int32_t>(v), lo, hi);
        return static_cast<uint32_t>(x) & field;
      };
      Lanes r;
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          uint32_t d = sat(b[lane]) | sat(a[lane]) << bits;
          if (c) d |= static_cast<uint32_t>((*c)[lane]) << (2 * bits);
          r[lane] = d;
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpCvtF16x2>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;
      HalfRounding rounding(op->rz ? FRound::Zero : FRound::Nearest);
      auto one = [&](double x) -> uint64_t {
        if (op->relu) {
          if (std::isnan(x)) return kCanonicalNaN16;
          if (x < 0 || std::signbit(x)) x = 0.0;   // -0 too (measured)
        }
        uint64_t h = op->bf16 ? double_to_bf16(x) : double_to_f16(x);
        // .satfinite: an overflow stops at the largest finite value.
        if (op->satfinite && !std::isnan(x) && (op->bf16 ? (h & 0x7FFF) == 0x7F80 : (h & 0x7FFF) == 0x7C00))
          h = (h & 0x8000) | (op->bf16 ? 0x7F7F : 0x7BFF);
        if (op->pzo && (h & 0xFFFF) == 0x8000) h = 0;   // .pzo: a -0.0 result is +0.0
        return h & 0xFFFF;
      };
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          // The first source goes in the high half, the second in the low.
          r[lane] = (one(static_cast<double>(f32(a[lane]))) << 16) | one(static_cast<double>(f32(b[lane])));
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpCvtTf32>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r;
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          uint32_t x = static_cast<uint32_t>(v[lane]);
          const bool nan = (x & 0x7F800000u) == 0x7F800000u && (x & 0x7FFFFFu);
          if (nan) {
            // Truncated rather than rounded (an RTX 3060: 0xff801fff gives
            // 0xff800000); with .relu or .satfinite, the canonical NaN.
            r[lane] = (op->relu || op->satfinite) ? 0x7FFFFFFFu : (x & ~0x1FFFu);
            continue;
          }
          if (op->relu && (x & 0x80000000u)) x = 0;
          uint32_t t;
          if (op->rna) t = (x + 0x1000u) & ~0x1FFFu;                    // ties away, on the bits
          else if (op->rz) t = x & ~0x1FFFu;
          else t = (x + 0x0FFFu + ((x >> 13) & 1u)) & ~0x1FFFu;          // ties to even
          // Rounding up can carry into the exponent and out to infinity.
          if (op->satfinite && (t & 0x7FFFFFFFu) >= 0x7F800000u) t = (t & 0x80000000u) | 0x7F7FE000u;
          if (op->pzo && t == 0x80000000u) t = 0;   // .pzo: a -0.0 result is +0.0
          r[lane] = t;
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpCopysign>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          // Magnitude of b, sign of a -- and it must be bit-exact for zeros and
          // NaNs, so move the sign bit rather than going through comparisons.
          if (op->ty.bits == 64) {
            const uint64_t sign = a[lane] & (1ull << 63);
            r[lane] = sign | (b[lane] & ~(1ull << 63));
          } else {
            const uint32_t av = static_cast<uint32_t>(a[lane]);
            const uint32_t bv = static_cast<uint32_t>(b[lane]);
            r[lane] = (av & 0x80000000u) | (bv & 0x7fffffffu);
          }
        }
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpDp4a>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r;
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          const uint32_t av = static_cast<uint32_t>(a[lane]);
          const uint32_t bv = static_cast<uint32_t>(b[lane]);
          // Each operand contributes four bytes, sign- or zero-extended
          // according to its own type; the four products accumulate into c.
          int64_t acc = op->a_signed || op->b_signed
                            ? static_cast<int64_t>(static_cast<int32_t>(c[lane]))
                            : static_cast<int64_t>(static_cast<uint32_t>(c[lane]));
          if (op->two) {
            // dp2a: a's two halves against b's low or high two bytes.
            for (int i = 0; i < 2; ++i) {
              const uint16_t ah = static_cast<uint16_t>(av >> (i * 16));
              const uint8_t bb = static_cast<uint8_t>(bv >> ((i + (op->hi ? 2 : 0)) * 8));
              const int64_t ax = op->a_signed ? static_cast<int16_t>(ah) : static_cast<int64_t>(ah);
              const int64_t bx = op->b_signed ? static_cast<int8_t>(bb) : static_cast<int64_t>(bb);
              acc += ax * bx;
            }
          } else {
            for (int byte = 0; byte < 4; ++byte) {
              const uint8_t ab = static_cast<uint8_t>(av >> (byte * 8));
              const uint8_t bb = static_cast<uint8_t>(bv >> (byte * 8));
              const int64_t ax = op->a_signed ? static_cast<int8_t>(ab) : static_cast<int64_t>(ab);
              const int64_t bx = op->b_signed ? static_cast<int8_t>(bb) : static_cast<int64_t>(bb);
              acc += ax * bx;
            }
          }
          r[lane] = static_cast<uint32_t>(static_cast<int32_t>(acc));
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpBmsk>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          uint32_t base = static_cast<uint32_t>(a[lane]);
          uint32_t width = static_cast<uint32_t>(b[lane]);
          // .wrap takes both operands modulo 32; .clamp caps them at 32. These
          // are different: wrapping the start position for .clamp turns a
          // position of 40 into 8 and produces a mask in the wrong place,
          // rather than the empty mask the clamp is supposed to give.
          if (op->wrap) {
            base &= 31u;
            width &= 31u;
          } else {
            if (base > 32u) base = 32u;
            if (width > 32u) width = 32u;
          }
          // Build and shift in 64 bits: a width or shift of 32 is undefined on
          // a 32-bit type.
          const uint64_t bits = width >= 32u ? 0xffffffffull : ((1ull << width) - 1ull);
          const uint64_t shifted = base >= 32u ? 0ull : (bits << base);
          r[lane] = static_cast<uint32_t>(shifted & 0xffffffffull);
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpCvtFp8>(&ins.op)) {
      const Fp8Format& f = op->e5m2 ? kE5M2 : kE4M3;
      const NarrowFmt fmt = op->fmt;
      const uint32_t width = fmt == NarrowFmt::E2M1 ? 4 : 8;   // bits a value takes in the pair
      Lanes _s_a, _s_b, _s_sf;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      const Lanes& b = op->to_fp8 && op->src_f32_pair ? read_operand(w, ctx, ins, op->b, _s_b) : a;
      const Lanes* sfl = op->scaled ? &read_operand(w, ctx, ins, op->sf, _s_sf) : nullptr;
      // The scale factor of value h (1 = the upper one): a ue8m0 byte.
      auto scale = [&](uint32_t lane, int h) {
        if (!sfl) return 1.0;
        const uint32_t byte = ((*sfl)[lane] >> (op->scaled_n1 ? 0 : 8 * h)) & 0xFF;
        return byte == 0xFF ? std::numeric_limits<double>::quiet_NaN() : std::ldexp(1.0, int(byte) - 127);
      };
      // One value to the narrow type (9.7.10.24's .relu and .satfinite).
      auto encode = [&](double v, uint32_t lane) -> uint32_t {
        // Measured on an L4 (sm_89): .relu sends -0 and every negative to +0.
        if (op->relu && !std::isnan(v) && std::signbit(v)) v = 0.0;
        uint32_t code;
        switch (fmt) {
          case NarrowFmt::E4M3:
          case NarrowFmt::E5M2:
            // .rz (PTX ISA 9.4): truncate onto the grid first; what is past the largest
            // finite value is then the encoder's, as for .rn.
            if (op->rz) v = trunc_to_grid(v, op->e5m2 ? 2 : 3, op->e5m2 ? 15 : 7);
            code = double_to_fp8(v, f, op->satfinite);
            return op->pzo && code == 0x80 ? 0u : code;
          case NarrowFmt::E2M3:
          case NarrowFmt::E3M2:
          case NarrowFmt::E2M1: {
            const int eb = fmt == NarrowFmt::E3M2 ? 3 : 2, mb = fmt == NarrowFmt::E2M3 ? 3 : fmt == NarrowFmt::E3M2 ? 2 : 1;
            const int bias = fmt == NarrowFmt::E3M2 ? 3 : 1;
            if (op->rz) v = trunc_to_grid(v, mb, bias);
            code = exec::small_float_bits(v, eb, mb, bias);
            return op->pzo && code == (1u << (eb + mb)) ? 0u : code;
          }
          case NarrowFmt::UE8M0: {
            const uint32_t code = exec::ue8m0_bits(v, op->rp, op->satfinite);
            if (code == ~0u)
              ctx_fail(ins, static_cast<int>(lane), Err::UnsupportedPtx,
                       "cvt to ue8m0x2 of a negative value, which the ISA does not define");
            return code;
          }
          case NarrowFmt::S2F6: {
            // An s8 in units of 2^-6; NaN to the positive largest.
            if (std::isnan(v)) return 0x7F;
            const double q = op->rz ? std::trunc(v * 64.0) : std::nearbyint(v * 64.0);
            return static_cast<uint32_t>(static_cast<int32_t>(std::clamp(q, -128.0, 127.0))) & 0xFF;
          }
        }
        return 0;
      };
      auto decode = [&](uint32_t code) -> double {
        switch (fmt) {
          case NarrowFmt::E4M3:
          case NarrowFmt::E5M2: return fp8_to_double(code, f);
          case NarrowFmt::E2M3: return small_float_value(code & 0x3F, 2, 3, 1);
          case NarrowFmt::E3M2: return small_float_value(code & 0x3F, 3, 2, 3);
          case NarrowFmt::E2M1: return small_float_value(code & 0xF, 2, 1, 1);
          case NarrowFmt::UE8M0:
            return code == 0xFF ? std::numeric_limits<double>::quiet_NaN() : std::ldexp(1.0, int(code) - 127);
          case NarrowFmt::S2F6: return static_cast<int8_t>(code & 0xFF) / 64.0;
        }
        return 0.0;
      };
      Lanes r;
      for (uint32_t lane = 0; lane < W_; ++lane) {
        if (!(m & (Mask{1} << lane))) continue;
        uint64_t out = 0;
        for (int h = 0; h < 2; ++h) {   // h = 1 is the upper value
          if (op->to_fp8) {
            double v;
            if (op->src_f32_pair) v = f32(h ? a[lane] : b[lane]);
            else {
              const uint64_t bits = (a[lane] >> (16 * h)) & 0xFFFF;
              v = op->bf16 ? bf16_to_double(bits) : f16_to_double(bits);
            }
            if (fmt == NarrowFmt::S2F6 || op->scaled_n1) v /= scale(lane, h);
            out |= uint64_t{encode(v, lane)} << (width * h);
          } else {
            double v = decode((a[lane] >> (width * h)) & ((1u << width) - 1)) * scale(lane, h);
            uint64_t half;
            if (op->relu && std::isnan(v)) half = 0x7FFF;   // canonical NaN
            else {
              if (op->relu && v < 0) v = 0.0;
              half = op->bf16 ? double_to_bf16(v) : double_to_f16(v);
              // .satfinite (bf16x2): past the largest finite, the largest.
              if (op->satfinite && std::isinf(v)) half = (v < 0 ? 0x8000 : 0) | (op->bf16 ? 0x7F7F : 0x7BFF);
              if (op->satfinite && op->bf16 && (half & 0x7FFF) == 0x7F80) half = (half & 0x8000) | 0x7F7F;
            }
            out |= half << (16 * h);
          }
        }
        r[lane] = out;
      }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpVideo>(&ins.op)) {
      exec_video(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpBfind>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->src, _s_a);
      Lanes r;
      const uint32_t bits = op->ty.bits;
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          uint64_t v = mask_to_bits(a[lane], bits);
          // The signed form looks for the most significant bit that differs
          // from the sign, which is what makes it an integer log2 of the
          // magnitude for negatives too. Inverting a negative value first is
          // how PTX defines it.
          if (op->ty.is_signed()) {
            const bool neg = (v >> (bits - 1)) & 1u;
            if (neg) v = mask_to_bits(~v, bits);
          }
          uint32_t idx = 0xFFFFFFFFu;
          if (v != 0) {
            uint32_t i = bits;
            while (i-- > 0)
              if ((v >> i) & 1ull) { idx = i; break; }
            // .shiftamt reports the distance from the top rather than the
            // index, which is what a normalizing shift wants.
            if (op->shiftamt) idx = bits - 1 - idx;
          }
          r[lane] = idx;
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpElect>(&ins.op)) {
      require_warp32(ins, "elect.sync");
      Lanes _s_mm;
      const Lanes& mm = read_operand(w, ctx, ins, op->membermask, _s_mm);
      // One leader for the whole warp, not one per lane: every participating
      // lane must be told the *same* winner or the kernel has several leaders
      // and the copy it was electing someone to issue happens more than once.
      uint32_t leader = W_;
      const Mask members = static_cast<Mask>(mm[first_set(m)]) & m;
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (members & (Mask{1} << lane)) { leader = lane; break; }
      Lanes r;
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) r[lane] = leader < W_ ? leader : 0;
      if (op->dst.id != kNoReg) write_reg(w, op->dst, m, r, 32);
      if (op->pred_dst.id != kNoReg) {
        Mask& p = pred_slot(w, op->pred_dst);
        const Mask won = (leader < W_) ? (Mask{1} << leader) : Mask{0};
        p = (p & ~m) | (won & m);
      }
      return;
    }
    if (const auto* op = std::get_if<OpIsSpacep>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->src, _s_a);
      Mask& p = pred_slot(w, op->dst);
      for (uint32_t lane = 0; lane < W_; ++lane) {
        if (!(m & (Mask{1} << lane))) continue;
        const uint64_t v = a[lane];
        bool in_space = false;
        switch (op->space) {
          case Space::Shared:
            // .shared is this block's; .shared::cluster any block's of the cluster.
            in_space = v >= kSharedVaBase && v < kSharedVaBase + kSharedVaSize;
            if (in_space) {
              const uint64_t tag = (v - kSharedVaBase) >> kClusterRankShift;
              const size_t n = ctx.cluster_state ? ctx.cluster_state->ranks.size() : 1;
              in_space = op->cluster ? tag < n : tag == cluster_rank_of(ctx);
            }
            break;
          case Space::Local:
            in_space = v >= kLocalVaBase && v < kLocalVaBase + kLocalVaSize;
            break;
          default:
            // Global is every generic address outside the shared and local
            // windows -- a null pointer and all-ones included, as an RTX 3060
            // answers (this once required a device allocation, and said no to
            // an address the card calls global). The engine's own parameter
            // window is not global.
            in_space = !(v >= kSharedVaBase && v < kSharedVaBase + kSharedVaSize) &&
                       !(v >= kLocalVaBase && v < kLocalVaBase + kLocalVaSize) &&
                       !(v >= kParamVaBase && v < kParamVaBase + kParamVaSize);
            break;
        }
        p = in_space ? (p | (Mask{1} << lane)) : (p & ~(Mask{1} << lane));
      }
      return;
    }
    if (const auto* op = std::get_if<OpMapa>(&ins.op)) {
      Lanes _s_a, _s_r;
      const Lanes a = read_operand(w, ctx, ins, op->src, _s_a);
      const Lanes& rank = read_operand(w, ctx, ins, op->rank, _s_r);
      Lanes r{};
      for (uint32_t lane = 0; lane < W_; ++lane) {
        if (!(m & (Mask{1} << lane))) continue;
        uint64_t v = op->wide ? a[lane] : static_cast<uint32_t>(a[lane]);
        if (op->generic) {
          if (!is_shared(v))
            ctx_fail(ins, static_cast<int>(lane), Err::InvalidValue,
                     "mapa on a generic address that is not in shared memory");
          v -= kSharedVaBase;
        }
        // mapa computes an address and touches nothing, so a rank the cluster
        // does not have is refused where the address is used, not here: the
        // ISA does not make mapa itself an error, and CUTLASS's SM120
        // pingpong kernels map every lane's rank in a cluster of one and use
        // only their own.
        const uint64_t want = static_cast<uint32_t>(rank[lane]) & ((uint64_t{1} << (32 - kClusterRankShift)) - 1);
        r[lane] = cluster_address(ctx, want, v) + (op->generic ? kSharedVaBase : 0);
      }
      write_reg(w, op->dst, m, r, op->wide ? 64 : 32);
      return;
    }
    if (const auto* op = std::get_if<OpGetCtaRank>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->src, _s_a);
      Lanes r{};
      for (uint32_t lane = 0; lane < W_; ++lane) {
        if (!(m & (Mask{1} << lane))) continue;
        uint64_t v = a[lane];
        if (op->generic) {
          if (!is_shared(v))
            ctx_fail(ins, static_cast<int>(lane), Err::InvalidValue,
                     "getctarank on a generic address that is not in shared memory");
          v -= kSharedVaBase;
        }
        const uint64_t tag = (v & (kSharedVaSize - 1)) >> kClusterRankShift;
        r[lane] = tag;
      }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpStAsync>(&ins.op)) {
      exec_st_async(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpTensormapReplace>(&ins.op)) {
      exec_tensormap_replace(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpTensormapCopy>(&ins.op)) {
      // A plain 128-byte copy; the proxy fence it carries orders it before
      // the tensor-map reads that follow, which here are ordinary loads.
      Lanes _s_d, _s_s;
      const Lanes dst = addr_base(w, ctx, ins, op->dst, _s_d);
      const Lanes src = addr_base(w, ctx, ins, op->src, _s_s);
      for (uint32_t lane = 0; lane < W_; ++lane) {
        if (!(m & (Mask{1} << lane))) continue;
        const uint64_t d = dst[lane] + static_cast<uint64_t>(op->dst.offset);
        const uint64_t sa = kSharedVaBase + src[lane] + static_cast<uint64_t>(op->src.offset);
        for (uint64_t i = 0; i < 128; i += 8)
          store_routed(w, ctx, ins, lane, d + i, 8, load_routed(w, ctx, ins, lane, sa + i, 8));
      }
      return;
    }
    if (const auto* op = std::get_if<OpBulkCopy>(&ins.op)) {
      exec_bulk_copy(w, ctx, ins, *op, m);
      return;
    }
    if (std::holds_alternative<OpBulkGroup>(ins.op)) {
      // Bulk stores write global memory when issued (see exec_bulk_copy), so
      // a group has nothing left to wait for when it is committed.
      return;
    }
    if (const auto* op = std::get_if<OpMbarrier>(&ins.op)) {
      exec_mbarrier(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpMatch>(&ins.op)) {
      require_warp32(ins, "match.sync");
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes r;
      Mask all_agreed = 0;
      for (uint32_t lane = 0; lane < W_; ++lane) {
        if (!(m & (Mask{1} << lane))) continue;
        // Participants are the lanes named by the member mask that are also
        // actually active. A lane listed in the mask but not executing cannot
        // contribute a value, and reading its stale register would invent one.
        // (The mask is not consulted: see sync_mask_groups.)
        const Mask members = m;
        Mask same = 0;
        for (uint32_t o = 0; o < W_; ++o)
          if ((members & (Mask{1} << o)) && a[o] == a[lane]) same |= (Mask{1} << o);
        // match.all writes the member mask when every member holds the same
        // value and 0 otherwise; match.any, the members holding this lane's.
        if (same == members) all_agreed |= (Mask{1} << lane);
        r[lane] = op->all ? (same == members ? static_cast<uint64_t>(members) : 0) : static_cast<uint64_t>(same);
      }
      write_reg(w, op->dst, m, r, 32);
      if (op->all && op->pred_dst.id != kNoReg) {
        Mask& p = pred_slot(w, op->pred_dst);
        p = (p & ~m) | (all_agreed & m);
      }
      return;
    }
    if (const auto* op = std::get_if<OpMul24>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes* c = op->mad ? &read_operand(w, ctx, ins, op->c, _s_c) : nullptr;
      Lanes r;
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          // The product is 48 bits wide, which is why .hi cannot be had by
          // masking the inputs of a 32-bit multiply: it wants bits 47:16 (the
          // ISA's, and an RTX 3060's -- this once took 47:24).
          int64_t prod;
          if (op->is_signed) {
            auto s24 = [](uint64_t v) -> int64_t {
              const uint32_t f = static_cast<uint32_t>(v) & 0xFFFFFFu;
              return (f & 0x800000u) ? static_cast<int64_t>(f) - 0x1000000 : static_cast<int64_t>(f);
            };
            prod = s24(a[lane]) * s24(b[lane]);
          } else {
            prod = static_cast<int64_t>((a[lane] & 0xFFFFFFu) * (b[lane] & 0xFFFFFFu));
          }
          const uint64_t u = static_cast<uint64_t>(prod);
          uint64_t v = op->hi ? ((u >> 16) & 0xFFFFFFFFull) : (u & 0xFFFFFFFFull);
          if (op->sat) {
            // The true sum of bits 47:16 and c, clamped (measured).
            const int64_t t = (prod >> 16) + static_cast<int32_t>((*c)[lane]);
            v = static_cast<uint32_t>(static_cast<int32_t>(std::clamp<int64_t>(t, INT32_MIN, INT32_MAX)));
          } else if (op->mad) {
            v = (v + (*c)[lane]) & 0xFFFFFFFFull;
          }
          r[lane] = v;
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpSzext>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          uint32_t n = static_cast<uint32_t>(b[lane]);
          n = op->wrap ? (n & 31u) : (n > 32u ? 32u : n);
          const uint32_t v = static_cast<uint32_t>(a[lane]);
          if (n == 0) { r[lane] = op->is_signed ? 0u : 0u; continue; }
          if (n >= 32) { r[lane] = v; continue; }
          const uint32_t keep = v & ((1u << n) - 1u);
          r[lane] = (op->is_signed && (keep & (1u << (n - 1))))
                        ? (keep | ~((1u << n) - 1u))
                        : keep;
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpFns>(&ins.op)) {
      Lanes _s_mask;
      const Lanes& mv = read_operand(w, ctx, ins, op->mask, _s_mask);
      Lanes _s_base;
      const Lanes& bv = read_operand(w, ctx, ins, op->base, _s_base);
      Lanes _s_off;
      const Lanes& ov = read_operand(w, ctx, ins, op->offset, _s_off);
      Lanes r;
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          const uint32_t bits = static_cast<uint32_t>(mv[lane]);
          const uint32_t base = static_cast<uint32_t>(bv[lane]);
          const int32_t off = static_cast<int32_t>(static_cast<uint32_t>(ov[lane]));
          uint32_t found = 0xFFFFFFFFu;
          // An offset of INT_MIN gives 0 whatever the mask and base, and a
          // base past bit 31 otherwise finds nothing (it is not taken modulo
          // 32) -- both measured on an RTX 3060, over 650 cases.
          if (off == INT32_MIN) found = 0;
          else if (base > 31) ;
          else if (off > 0) {
            int32_t n = off;
            for (int i = static_cast<int>(base); i < 32; ++i)
              if ((bits >> i) & 1u) { if (--n == 0) { found = static_cast<uint32_t>(i); break; } }
          } else if (off < 0) {
            int32_t n = -off;
            for (int i = static_cast<int>(base); i >= 0; --i)
              if ((bits >> i) & 1u) { if (--n == 0) { found = static_cast<uint32_t>(i); break; } }
          } else {
            // offset 0 asks for the bit at `base` itself.
            if ((bits >> base) & 1u) found = base;
          }
          r[lane] = found;
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpLop3>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r;
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          // Evaluate the truth table bit-parallel. Each of the 8 table bits
          // names one (a,b,c) combination; for the bits where the table says
          // 1, OR in the mask of positions whose operand bits match that
          // combination. Building that mask from a, b and c themselves does
          // all 32 positions at once, so this is one pass rather than 32.
          const uint32_t av = static_cast<uint32_t>(a[lane]);
          const uint32_t bv = static_cast<uint32_t>(b[lane]);
          const uint32_t cv = static_cast<uint32_t>(c[lane]);
          uint32_t out = 0;
          for (int k = 0; k < 8; ++k) {
            if (!((op->lut >> k) & 1)) continue;
            const uint32_t ma = (k & 4) ? av : ~av;
            const uint32_t mb = (k & 2) ? bv : ~bv;
            const uint32_t mc = (k & 1) ? cv : ~cv;
            out |= ma & mb & mc;
          }
          r[lane] = out;
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpSlct>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r;
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          bool take_a;
          if (op->c_is_float) {
            // NaN is not >= 0, so it selects b. Comparing the bits instead
            // would put NaN on whichever side its sign bit fell.
            const float f = f32(c[lane]);
            take_a = f >= 0.0f;
          } else {
            take_a = static_cast<int32_t>(static_cast<uint32_t>(c[lane])) >= 0;
          }
          r[lane] = take_a ? a[lane] : b[lane];
        }
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpTestp>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Mask& dst = pred_slot(w, op->dst);
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          // Classified at the operand's own width: an f32 subnormal widened to
          // a double is a normal double, which made testp.normal.f32 true and
          // testp.subnormal.f32 false for every f32 subnormal.
          const int cls = op->ty.bits == 64 ? std::fpclassify(f64(a[lane])) : std::fpclassify(f32(a[lane]));
          const bool nan = cls == FP_NAN;
          const bool inf = cls == FP_INFINITE;
          // Zero counts as normal: finite and not subnormal is what the card
          // answers (testp.normal of +0 and -0 is true on an RTX 3060).
          const bool normal = cls == FP_NORMAL || cls == FP_ZERO;
          const bool subnormal = cls == FP_SUBNORMAL;
          bool t = false;
          switch (op->op) {
            case TestpOp::Finite: t = !nan && !inf; break;
            case TestpOp::Infinite: t = inf; break;
            case TestpOp::Number: t = !nan; break;
            case TestpOp::NotANumber: t = nan; break;
            case TestpOp::Normal: t = normal; break;
            case TestpOp::Subnormal: t = subnormal; break;
          }
          dst = t ? (dst | (Mask{1} << lane)) : (dst & ~(Mask{1} << lane));
        }
      return;
    }
    if (const auto* op = std::get_if<OpSad>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r;
      const uint32_t bits = op->ty.bits;
      const bool sgn = op->ty.kind == Type::Kind::S;
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          uint64_t diff;
          if (sgn) {
            // Sign-extend to 64 bits first: the absolute difference of two
            // 32-bit signed values does not fit in 32 bits, and truncating
            // before the subtraction gets the wrap case wrong.
            auto sext = [bits](uint64_t v) -> int64_t {
              if (bits >= 64) return static_cast<int64_t>(v);
              const uint64_t f = v & ((1ull << bits) - 1);
              return (f & (1ull << (bits - 1)))
                         ? static_cast<int64_t>(f | ~((1ull << bits) - 1))
                         : static_cast<int64_t>(f);
            };
            const int64_t x = sext(a[lane]);
            const int64_t y = sext(b[lane]);
            diff = static_cast<uint64_t>(x > y ? x - y : y - x);
          } else {
            const uint64_t x = mask_to_bits(a[lane], bits);
            const uint64_t y = mask_to_bits(b[lane], bits);
            diff = x > y ? x - y : y - x;
          }
          r[lane] = mask_to_bits(diff + c[lane], bits);
        }
      write_reg(w, op->dst, m, r, bits);
      return;
    }
    if (const auto* op = std::get_if<OpPrmt>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          uint8_t bytes[8];
          uint32_t lo = static_cast<uint32_t>(a[lane]), hi = static_cast<uint32_t>(b[lane]);
          for (int i = 0; i < 4; ++i) bytes[i] = (lo >> (8 * i)) & 0xFF;
          for (int i = 0; i < 4; ++i) bytes[4 + i] = (hi >> (8 * i)) & 0xFF;
          uint32_t sel = static_cast<uint32_t>(c[lane]);
          uint32_t out = 0;
          const uint32_t s2 = sel & 3;
          for (int i = 0; i < 4; ++i) {
            uint8_t byte = 0;
            switch (op->mode) {
              case PrmtMode::Generic: {
                const uint32_t nib = (sel >> (4 * i)) & 0xF;
                byte = bytes[nib & 0x7];
                if (nib & 0x8) byte = (byte & 0x80) ? 0xFF : 0x00;  // sign-replicate mode
                break;
              }
              case PrmtMode::F4e: byte = bytes[(s2 + i) & 7]; break;
              case PrmtMode::B4e: byte = bytes[(s2 - i) & 7]; break;
              case PrmtMode::Rc8: byte = bytes[s2]; break;
              case PrmtMode::Ecl: byte = bytes[std::max<uint32_t>(i, s2)]; break;
              case PrmtMode::Ecr: byte = bytes[std::min<uint32_t>(i, s2)]; break;
              case PrmtMode::Rc16: byte = bytes[2 * (s2 & 1) + (i & 1)]; break;
            }
            out |= static_cast<uint32_t>(byte) << (8 * i);
          }
          r[lane] = out;
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpAbs>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          if (op->ty.is_float())
            r[lane] = float_sign_op(op->ty.bits, v[lane], op->ftz, /*negate=*/false);
          else {
            int64_t x = op->ty.bits == 64 ? static_cast<int64_t>(v[lane])
                                          : int64_t{static_cast<int32_t>(v[lane])};
            r[lane] = static_cast<uint64_t>(x < 0 ? -x : x);
          }
        }
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpMath>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r;  // written for every active lane below
      auto apply = [&](double x) {
        switch (op->op) {
          case MathOp::Ex2: return std::exp2(x);
          case MathOp::Lg2: return std::log2(x);
          case MathOp::Sin: return std::sin(x);
          case MathOp::Cos: return std::cos(x);
          case MathOp::Sqrt: return std::sqrt(x);
          case MathOp::Rsqrt: return 1.0 / std::sqrt(x);
          case MathOp::Rcp: return 1.0 / x;
          case MathOp::Tanh: return std::tanh(x);
        }
        return 0.0;
      };
      const bool is_bf = op->ty.is_bfloat();
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          if (op->ty.bits == 16) {
            // The half forms compute at host precision like every other
            // transcendental here and round once at the end. Rounding to 16
            // bits *before* the function -- the shape a naive reuse of the f32
            // path would take -- loses more than the hardware's own
            // approximation does.
            uint64_t out = 0;
            const int halves = op->packed ? 2 : 1;
            for (int h = 0; h < halves; ++h) {
              const uint64_t bits = (v[lane] >> (16 * h)) & 0xFFFF;
              const double y = apply(is_bf ? bf16_to_double(bits) : f16_to_double(bits));
              out |= (is_bf ? double_to_bf16(y) : double_to_f16(y)) << (16 * h);
            }
            r[lane] = out;
            continue;
          }
          if (op->exact) {
            // sqrt/rcp.{rn,rz,rm,rp}: the IEEE operations, correctly rounded in
            // that mode (the host's own, set for the operation).
            const bool directed = op->round != FRound::Nearest;
            const int prev = directed ? std::fegetround() : 0;
            if (directed) {
              g_directed_rounding.fetch_add(1, std::memory_order_relaxed);
              std::fesetround(op->round == FRound::Zero       ? FE_TOWARDZERO
                              : op->round == FRound::MinusInf ? FE_DOWNWARD
                                                              : FE_UPWARD);
            }
            if (op->ty.bits == 32) {
              float x = f32(v[lane]);
              if (op->ftz && std::fpclassify(x) == FP_SUBNORMAL) x = std::copysign(0.0f, x);
              volatile float one = 1.0f;
              float y = op->op == MathOp::Sqrt ? std::sqrt(x) : one / x;
              if (op->ftz && std::fpclassify(y) == FP_SUBNORMAL) y = std::copysign(0.0f, y);
              r[lane] = canon32(y);
            } else {
              const double x = f64(v[lane]);
              volatile double one = 1.0;
              r[lane] = f64bits(op->op == MathOp::Sqrt ? std::sqrt(x) : one / x);
            }
            if (directed) {
              std::fesetround(prev);
              g_directed_rounding.fetch_sub(1, std::memory_order_relaxed);
            }
            continue;
          }
          if (op->ty.bits == 32) {
            // .ftz: subnormal inputs and results to signed zero, as the
            // rounded forms above -- so sqrt.approx.ftz of a negative
            // subnormal is -0, rsqrt's and lg2's are infinite, and a
            // reciprocal or ex2 too small to be normal is 0 (an RTX 3060's).
            float x = f32(v[lane]);
            if (op->ftz && std::fpclassify(x) == FP_SUBNORMAL) x = std::copysign(0.0f, x);
            float y = static_cast<float>(apply(static_cast<double>(x)));
            if (op->ftz && std::fpclassify(y) == FP_SUBNORMAL) y = std::copysign(0.0f, y);
            r[lane] = canon32(y);
            continue;
          }
          r[lane] = f64bits(apply(f64(v[lane])));
        }
      // A half result still occupies a 32-bit register, packed or not.
      alu_fault(r, m, op->ty.bits == 16 ? 16u : op->ty.bits);
      write_reg(w, op->dst, m, r, op->ty.bits == 16 ? 32u : op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpBfe>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) r[lane] = bfe(op->ty, a[lane], b[lane], c[lane]);
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpBfi>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes _s_d;
      const Lanes& d = read_operand(w, ctx, ins, op->d, _s_d);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) r[lane] = bfi(op->ty, a[lane], b[lane], c[lane], d[lane]);
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpBrev>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          uint64_t x = mask_to_bits(v[lane], op->ty.bits), out = 0;
          for (uint32_t i = 0; i < op->ty.bits; ++i)
            if (x & (1ull << i)) out |= 1ull << (op->ty.bits - 1 - i);
          r[lane] = out;
        }
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpPopcClz>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          uint64_t x = mask_to_bits(v[lane], op->ty.bits);
          if (op->popc) {
            r[lane] = static_cast<uint64_t>(__builtin_popcountll(x));
          } else {
            uint32_t n = 0;
            for (int i = static_cast<int>(op->ty.bits) - 1; i >= 0 && !(x & (1ull << i)); --i) ++n;
            r[lane] = n;
          }
        }
      write_reg(w, op->dst, m, r, 32);  // popc/clz always produce a .u32
      return;
    }
    if (const auto* op = std::get_if<OpShfl>(&ins.op)) {
      require_warp32(ins, "shfl.sync");
      exec_shfl(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpVote>(&ins.op)) {
      require_warp32(ins, "vote/ballot");
      Mask p = read_pred(w, ins, op->src);
      if (op->negate_src) p = ~p;
      Mask voters = p & m;
      if (op->ballot) {
        Lanes r;  // written for every active lane below
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) r[lane] = voters;
        write_reg(w, op->dst, m, r, 32);
      } else {
        bool all = (voters == m), any = (voters != 0);
        bool val = op->mode == VoteMode::All   ? all
                   : op->mode == VoteMode::Any ? any
                                               : (voters == m || voters == 0);  // uni
        Mask& dp = pred_slot(w, op->dst);
        dp = val ? (dp | m) : (dp & ~m);
      }
      return;
    }
    if (const auto* op = std::get_if<OpMadHi>(&ins.op)) {
      Lanes _s_a; const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b; const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c; const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r;  // written for every active lane below
      if (op->carry_in || op->carry_out) {
        Lanes prod;
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) prod[lane] = mul_hi(op->ty, a[lane], b[lane]);
        exec_carry_mad(w, op->ty.bits, op->carry_in, op->carry_out, prod, c, m, r);
      } else if (op->sat) {
        // mad.hi.sat.s32: the true sum of the product's high half and c, clamped.
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) {
            const int64_t t = int64_t{static_cast<int32_t>(mul_hi(op->ty, a[lane], b[lane]))} + static_cast<int32_t>(c[lane]);
            r[lane] = static_cast<uint32_t>(static_cast<int32_t>(std::clamp<int64_t>(t, INT32_MIN, INT32_MAX)));
          }
      } else {
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) r[lane] = mul_hi(op->ty, a[lane], b[lane]) + c[lane];
      }
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpShf>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          uint64_t hi = static_cast<uint32_t>(b[lane]);
          uint64_t lo = static_cast<uint32_t>(a[lane]);
          uint32_t n = op->wrap ? (static_cast<uint32_t>(c[lane]) & 31u)
                                : std::min<uint32_t>(static_cast<uint32_t>(c[lane]), 32u);
          uint64_t funnel = (hi << 32) | lo;
          r[lane] = op->left ? static_cast<uint32_t>((funnel << n) >> 32)
                             : static_cast<uint32_t>(funnel >> n);
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpF16x2Bin>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;  // written for every active lane below
      const HalfMods& md = op->mods;
      HalfRounding rounding(md.round);
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          uint64_t out = 0;
          const int halves = op->packed ? 2 : 1;
          for (int h = 0; h < halves; ++h) {
            uint64_t ax = (a[lane] >> (16 * h)) & 0xFFFF;
            uint64_t bx = (b[lane] >> (16 * h)) & 0xFFFF;
            if (md.ftz && !op->bf16) { ax = half_ftz(ax); bx = half_ftz(bx); }
            const uint64_t sign = (ax ^ bx) & 0x8000;
            if (md.xorsign_abs) { ax &= 0x7FFF; bx &= 0x7FFF; }
            double x = op->bf16 ? bf16_to_double(ax) : f16_to_double(ax);
            double y = op->bf16 ? bf16_to_double(bx) : f16_to_double(bx);
            double v;
            switch (op->op) {
              case FloatBinOp::Add: v = x + y; break;
              case FloatBinOp::Sub: v = x - y; break;
              case FloatBinOp::Mul: v = x * y; break;
              // PTX min/max return the non-NaN operand when exactly one is
              // NaN, which is std::fmin/fmax's rule and not what < gives --
              // unless .NaN asks for the NaN.
              // -0 orders below +0, as for f32 (measured on an RTX 3060:
              // min(+0, -0) is -0, which .ftz makes of every negative
              // subnormal).
              case FloatBinOp::Min:
                v = x == 0 && y == 0 ? (std::signbit(x) || std::signbit(y) ? -0.0 : 0.0) : std::fmin(x, y);
                break;
              case FloatBinOp::Max:
                v = x == 0 && y == 0 ? (std::signbit(x) && std::signbit(y) ? -0.0 : 0.0) : std::fmax(x, y);
                break;
              default: v = x * y; break;
            }
            if (md.nan_propagate && (std::isnan(x) || std::isnan(y))) v = std::numeric_limits<double>::quiet_NaN();
            uint64_t hv = op->bf16 ? double_to_bf16(v) : double_to_f16(v);
            if (md.xorsign_abs && !half_nan(hv, op->bf16)) hv = (hv & 0x7FFF) | sign;
            out |= half_post(hv, op->bf16, md) << (16 * h);
          }
          r[lane] = out;
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpF32x2>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = op->fma ? read_operand(w, ctx, ins, op->c, _s_c) : b;
      Lanes r;  // written for every active lane below
      // Each half is computed in float under the requested rounding mode, so
      // the result is the correctly rounded f32 that hardware gives.
      const int prev_round = op->round == FRound::Nearest ? 0 : std::fegetround();
      if (op->round != FRound::Nearest) g_directed_rounding.fetch_add(1, std::memory_order_relaxed);
      switch (op->round) {
        case FRound::Zero: std::fesetround(FE_TOWARDZERO); break;
        case FRound::MinusInf: std::fesetround(FE_DOWNWARD); break;
        case FRound::PlusInf: std::fesetround(FE_UPWARD); break;
        case FRound::Nearest: break;
      }
      auto flush = [&](float v) { return op->ftz && std::fpclassify(v) == FP_SUBNORMAL ? std::copysign(0.0f, v) : v; };
      auto half = [](uint64_t v, int h) { return f32(static_cast<uint32_t>(v >> (32 * h))); };
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          uint64_t out = 0;
          for (int h = 0; h < 2; ++h) {
            const float x = flush(half(a[lane], h)), y = flush(half(b[lane], h));
            float v;
            if (op->fma) v = std::fmaf(x, y, flush(half(c[lane], h)));
            else if (op->op == FloatBinOp::Add) v = x + y;
            else if (op->op == FloatBinOp::Sub) v = x - y;
            else v = x * y;
            v = flush(v);
            out |= (f32bits(v) & 0xFFFFFFFFull) << (32 * h);
          }
          r[lane] = out;
        }
      if (op->round != FRound::Nearest) {
        std::fesetround(prev_round);
        g_directed_rounding.fetch_sub(1, std::memory_order_relaxed);
      }
      write_reg(w, op->dst, m, r, 64);
      return;
    }
    if (const auto* op = std::get_if<OpF16x2Fma>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r;  // written for every active lane below
      HalfRounding rounding(op->mods.round);
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          uint64_t out = 0;
          const int halves = op->packed ? 2 : 1;
          for (int h = 0; h < halves; ++h) {
            const uint64_t ax = (a[lane] >> (16 * h)) & 0xFFFF;
            const uint64_t bx = (b[lane] >> (16 * h)) & 0xFFFF;
            const uint64_t cx = (c[lane] >> (16 * h)) & 0xFFFF;
            const bool ftz = op->mods.ftz && !op->bf16;
            double x = op->bf16 ? bf16_to_double(ax) : f16_to_double(ftz ? half_ftz(ax) : ax);
            double y = op->bf16 ? bf16_to_double(bx) : f16_to_double(ftz ? half_ftz(bx) : bx);
            double z = op->bf16 ? bf16_to_double(cx) : f16_to_double(ftz ? half_ftz(cx) : cx);
            const uint64_t hv = op->bf16 ? bf16_fma(x, y, z) : double_to_f16(host_fma(x, y, z));
            out |= half_post(hv, op->bf16, op->mods) << (16 * h);
          }
          r[lane] = out;
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpF16x2Neg>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          // The sign bit is the top bit of each 16-bit half for both f16 and
          // bf16, so this is one mask either way; only how many halves take
          // part differs. neg flips it, abs clears it.
          // A NaN half comes out as the canonical 0x7fff (an RTX 3060), not
          // with its sign flipped or cleared.
          uint64_t out = 0;
          for (int h = 0; h < (op->packed ? 2 : 1); ++h) {
            uint64_t x = (v[lane] >> (16 * h)) & 0xFFFF;
            if (op->ftz) x = half_ftz(x);   // a subnormal is a signed zero first
            if (half_nan(x, op->bf16)) x = kCanonicalNaN16;
            else x = op->absolute ? (x & 0x7FFF) : (x ^ 0x8000);
            out |= x << (16 * h);
          }
          r[lane] = op->packed ? out : (out | (v[lane] & 0xFFFF0000ull));
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpWmmaMma>(&ins.op)) {
      require_warp32(ins, "wmma.mma");
      if (g_fast_path.load(std::memory_order_relaxed) && fast_wmma_mma(w, *op, m)) return;
      exec_wmma_mma(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpWmmaLoad>(&ins.op)) {
      require_warp32(ins, "wmma.load");
      exec_wmma_load(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpWmmaStore>(&ins.op)) {
      exec_wmma_store(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpPredBin>(&ins.op)) {
      const Mask a = op->a_imm >= 0 ? (op->a_imm ? ~Mask{0} : 0) : read_pred(w, ins, op->a);
      const Mask b = op->b_imm >= 0 ? (op->b_imm ? ~Mask{0} : 0) : read_pred(w, ins, op->b);
      Mask r = op->op == PredBinOp::And ? (a & b) : op->op == PredBinOp::Or ? (a | b) : (a ^ b);
      Mask& p = pred_slot(w, op->dst);
      p = (p & ~m) | (r & m);
      return;
    }
    if (const auto* op = std::get_if<OpNotPred>(&ins.op)) {
      Mask s = read_pred(w, ins, op->src);
      Mask& p = pred_slot(w, op->dst);
      p = (p & ~m) | (~s & m);
      return;
    }
    // A .param declaration belongs to the lanes that run it. Every call site
    // names its slots param0, retval0 and so on, and a warp's diverged paths
    // can interleave: a path starved long enough is boosted, and the boost can
    // end between its st.param and its call. Clearing the slot for the whole
    // warp there wiped the waiting lanes' arguments (Boost.Math's ibetac, with
    // lanes of one warp in different series).
    if (const auto* op = std::get_if<OpDeclSlot>(&ins.op)) {
      Warp::Slot& slot = w.slots[op->name];
      slot.fit(op->size ? op->size : 8u, W_);
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane))
          std::memset(slot.bytes.data() + static_cast<size_t>(lane) * slot.size, 0, slot.size);
      return;
    }
    if (const auto* op = std::get_if<OpStSlot>(&ins.op)) {
      Warp::Slot& slot = w.slots[op->slot];
      const uint32_t nbytes = op->ty.bytes() ? op->ty.bytes() : 4u;
      // A slot written before it was declared (or wider than declared) grows
      // to fit rather than dropping the write silently.
      slot.fit(static_cast<uint32_t>(op->offset) + nbytes * static_cast<uint32_t>(op->srcs.size()), W_);
      for (size_t e = 0; e < op->srcs.size(); ++e) {
        Lanes _s_v;
        const Lanes& v = read_operand(w, ctx, ins, op->srcs[e], _s_v);
        const uint32_t at = static_cast<uint32_t>(op->offset) + nbytes * static_cast<uint32_t>(e);
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) slot.write(lane, at, nbytes, mask_to_bits(v[lane], op->ty.bits));
      }
      return;
    }
    if (const auto* op = std::get_if<OpLdSlot>(&ins.op)) {
      auto it = w.slots.find(op->slot);
      if (it == w.slots.end())
        ctx_fail(ins, -1, Err::UninitializedRegister, "call slot '" + op->slot + "' read before write");
      const uint32_t nbytes = op->ty.bytes() ? op->ty.bytes() : 4u;
      for (size_t e = 0; e < op->dsts.size(); ++e) {
        const uint32_t at = static_cast<uint32_t>(op->offset) + nbytes * static_cast<uint32_t>(e);
        Lanes r{};
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) r[lane] = it->second.read(lane, at, nbytes);
        write_loaded(w, op->dsts[e], m, r, op->ty);
      }
      return;
    }
    if (const auto* op = std::get_if<OpCall>(&ins.op)) {
      if (op->callee == "__assertfail") {
        exec_assertfail(w, ctx, ins, *op, m);
        return;
      }
      if (op->callee == "malloc" || op->callee == "free") {
        exec_device_heap(w, ctx, ins, *op, m);
        return;
      }
      // The device runtime's entry points -- the launch pair as the CUDA
      // programming guide documents them for code generators ("Device-side
      // Launch from PTX"), and the rest of cuda_device_runtime_api.h. CUDA 12's
      // CDP2 compiles to the __cudaCDP2 names, CDP1 to the cuda ones.
      if (const devrt::Fn dfn = devrt::lookup(op->callee); dfn != devrt::Fn::None) {
        exec_devrt(dfn, w, ctx, ins, *op, m);
        return;
      }
      // Device-side cudaGetDevice and cudaGetDeviceCount: the device runtime
      // library's wrappers call these driver entry points (CUDA 12's CDP2
      // names, and CDP1's cnp ones), which write one int and return an error
      // code. NanoVDB calls cudaGetDevice from a kernel.
      if (op->callee == "__cuda_syscall_cnpv2GetDevice" || op->callee == "cnpGetDevice") {
        exec_device_int_query(w, ctx, ins, *op, m, cfg_.device_ordinal);
        return;
      }
      if (op->callee == "__cuda_syscall_cnpv2GetDeviceCount" || op->callee == "cnpGetDeviceCount") {
        exec_device_int_query(w, ctx, ins, *op, m, cfg_.device_count);
        return;
      }
      // Device-side streams and events (CUDA Samples' cdpSimpleQuicksort
      // launches each child into a stream of its own). Child grids run here
      // one after another in launch order, which every ordering these can
      // ask for already satisfies: creation hands back a distinct handle,
      // and destroying, recording and waiting have nothing left to do.
      if (op->callee == "__cuda_syscall_cnpv2GetLastError") {
        exec_device_get_last_error(w, *op, m);
        return;
      }
      if (op->callee == "__cuda_syscall_cnpv2SetLastError") {
        exec_device_set_last_error(w, ctx, ins, *op, m);
        return;
      }
      // cudaGraphSetConditional(handle, value): nvcc leaves it a call to an
      // external function of that name, which the driver supplies.
      if (op->callee == "cudaGraphSetConditional") {
        exec_graph_set_conditional(w, ins, *op, m);
        return;
      }
      // Device-side cudaGraphLaunch(graphExec, stream): also a driver-supplied
      // external, which the device runtime's header declares and no library
      // defines.
      if (op->callee == "cudaGraphLaunch") {
        exec_device_graph_launch(w, ins, *op, m);
        return;
      }
      if (op->callee == "__cuda_syscall_cnpv2StreamCreate" || op->callee == "__cuda_syscall_cnpv2EventCreate") {
        exec_device_handle(w, ctx, ins, *op, m);
        return;
      }
      if (op->callee == "__cuda_syscall_cnpv2StreamDestroy" || op->callee == "__cuda_syscall_cnpv2EventDestroy" ||
          op->callee == "__cuda_syscall_cnpv2EventRecord" || op->callee == "__cuda_syscall_cnpv2StreamWaitEvent") {
        write_call_result(w, *op, m, Lanes{}, 4);   // cudaSuccess
        return;
      }
      if (op->indirect) {
        exec_indirect_call(w, ctx, ins, *op, m);
        return;
      }
      if (op->target) {
        exec_user_call(w, ctx, ins, *op, m);
        return;
      }
      // Everything that is neither a builtin nor a resolved device function.
      // Reported here rather than at parse time because a separately compiled
      // build links in CUDA's device-runtime library, which declares functions
      // the driver supplies and defines them nowhere -- refusing at load
      // rejected whole programs over a library function nobody calls.
      if (op->callee != "vprintf")
        ctx_fail(ins, -1, Err::UnsupportedPtx,
                 "call to '" + op->callee +
                     "', which this module neither defines nor implements as a builtin");
      exec_vprintf(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpTex>(&ins.op)) {
      exec_tex(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpSuld>(&ins.op)) {
      exec_suld(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpStack>(&ins.op)) {
      exec_stack(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpSust>(&ins.op)) {
      exec_sust(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpTrap>(&ins.op)) {
      if (op->breakpoint)
        ctx_fail(ins, -1, Err::Trap,
                 "the kernel executed 'brkpt'. With no debugger attached nothing can resume it, so "
                 "the launch ends with a device-side error; libraries place one on paths they "
                 "consider unreachable");
      // "trap" ends the kernel with an unrecoverable device-side error. On
      // hardware the launch fails and the context is left unusable; here it is
      // an error with the line that did it, which is more use and no less true.
      //
      // Compilers put it on paths that are supposed to be unreachable --
      // cooperative_groups emits one where the grid is not cooperative, and
      // assert() lowers to it -- so reaching one is a fact about the program
      // worth reporting rather than a gap in this interpreter.
      ctx_fail(ins, -1, Err::Trap,
               "the kernel executed 'trap', which ends it with a device-side error. This is what "
               "a failed device assert(), an unreachable path, or a cooperative-groups call "
               "outside a cooperative launch compiles to");
    }
    // The parser accepted it and nothing here runs it -- a gap between the two
    // halves, not a bad program. Naming the instruction is the difference
    // between a five-minute fix and a bisection.
    ctx_fail(ins, -1, Err::Internal,
             "the parser accepts this instruction but the interpreter has no handler for it");
  }

  // High half of a same-width product. 64-bit needs a 128-bit intermediate.
  static uint64_t mul_hi(Type ty, uint64_t a, uint64_t b) {
    if (ty.bits == 64) {
      if (ty.is_signed()) {
        __int128 prod = static_cast<__int128>(static_cast<int64_t>(a)) *
                        static_cast<__int128>(static_cast<int64_t>(b));
        return static_cast<uint64_t>(static_cast<unsigned __int128>(prod) >> 64);
      }
      unsigned __int128 prod = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
      return static_cast<uint64_t>(prod >> 64);
    }
    uint32_t bits = ty.bits;
    if (ty.is_signed()) {
      int64_t x = static_cast<int64_t>(sign_extend(a, bits));
      int64_t y = static_cast<int64_t>(sign_extend(b, bits));
      return mask_to_bits(static_cast<uint64_t>((x * y) >> bits), bits);
    }
    uint64_t prod = mask_to_bits(a, bits) * mask_to_bits(b, bits);
    return mask_to_bits(prod >> bits, bits);
  }

  static uint64_t sign_extend(uint64_t v, uint32_t bits) {
    if (bits >= 64) return v;
    uint64_t sign = 1ull << (bits - 1);
    v = mask_to_bits(v, bits);
    return (v & sign) ? (v | ~((sign << 1) - 1)) : v;
  }

  // The 32-bit forms take the ISA's 8-bit position and length; the 64-bit
  // forms the whole 32-bit operands (measured on an RTX 3060: bfe.u64 with
  // length 0x100 extracts nothing, not 0 bits wrapped to 256).
  static uint64_t bfe(Type ty, uint64_t a, uint64_t bpos, uint64_t clen) {
    const uint32_t fmask = ty.bits == 64 ? 0xFFFFFFFFu : 0xFFu;
    uint32_t pos = static_cast<uint32_t>(bpos) & fmask;
    uint32_t len = static_cast<uint32_t>(clen) & fmask;
    uint32_t bits = ty.bits;
    if (pos >= bits || len == 0) {
      // Signed extract of an empty//out-of-range field replicates the sign bit.
      if (!ty.is_signed() || len == 0) return 0;
      uint64_t sign = (a >> (bits - 1)) & 1;
      return sign ? mask_to_bits(~0ull, bits) : 0;
    }
    if (len > bits - pos) len = bits - pos;
    uint64_t field = (a >> pos) & ((len >= 64) ? ~0ull : ((1ull << len) - 1));
    // (A 64-bit field needs no extension -- and 1 << 64 is undefined, which
    // on x86 made the whole of a negative value all ones.)
    if (ty.is_signed() && len < 64 && (field & (1ull << (len - 1)))) field |= ~((1ull << len) - 1);
    return mask_to_bits(field, bits);
  }

  static uint64_t bfi(Type ty, uint64_t a, uint64_t b, uint64_t cpos, uint64_t dlen) {
    const uint32_t fmask = ty.bits == 64 ? 0xFFFFFFFFu : 0xFFu;
    uint32_t pos = static_cast<uint32_t>(cpos) & fmask;
    uint32_t len = static_cast<uint32_t>(dlen) & fmask;
    uint32_t bits = ty.bits;
    if (pos >= bits || len == 0) return mask_to_bits(b, bits);
    if (len > bits - pos) len = bits - pos;
    uint64_t field_mask = ((len >= 64) ? ~0ull : ((1ull << len) - 1)) << pos;
    return mask_to_bits((b & ~field_mask) | ((a << pos) & field_mask), bits);
  }

  // Warp shuffle: a lane reads another lane's register. This is a plain index
  // into the warp's lane vector — the operation the warp-vectorized register
  // file makes trivial (see ARCHITECTURE.md D1).
  void exec_shfl(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpShfl& op, Mask m) {
    Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op.a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op.b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op.c, _s_c);
    Lanes r;  // written for every active lane below
    Mask pred_out = 0;
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      uint32_t bval = static_cast<uint32_t>(b[lane]) & 0x1F;
      uint32_t cval = static_cast<uint32_t>(c[lane]) & 0x1F;
      uint32_t segmask = (static_cast<uint32_t>(c[lane]) >> 8) & 0x1F;
      uint32_t min_lane = lane & segmask;
      uint32_t max_lane = (lane & segmask) | (cval & ~segmask);
      uint32_t j = lane;
      bool pred;
      switch (op.mode) {
        case ShflMode::Up:
          // The ISA's pval = (j >= maxLane): for .up, c's clamp field is the
          // lowest source lane (this compared with minLane, which agrees only
          // when c's clamp bits are 0, as __shfl_up_sync passes them).
          j = lane - bval;
          pred = static_cast<int32_t>(lane - bval) >= static_cast<int32_t>(max_lane);
          break;
        case ShflMode::Down:
          j = lane + bval;
          pred = j <= max_lane;
          break;
        case ShflMode::Bfly:
          j = lane ^ bval;
          pred = j <= max_lane;
          break;
        case ShflMode::Idx:
          j = min_lane | (bval & ~segmask);
          pred = j <= max_lane;
          break;
      }
      if (!pred) j = lane;  // out-of-range source: the lane reads its own value
      // A source lane that is not running this shuffle gives 0, not what its
      // register holds (an RTX 3060: lanes 0-15 of a diverged warp shuffling
      // from lane 17 read 0).
      r[lane] = (m >> (j & 0x1F)) & 1 ? a[j & 0x1F] : 0;
      if (pred) pred_out |= (Mask{1} << lane);
    }
    write_reg(w, op.dst, m, r, 32);
    if (op.pred_dst.id != kNoReg) {
      Mask& p = pred_slot(w, op.pred_dst);
      p = (p & ~m) | (pred_out & m);
    }
  }

  // ---- tensor cores (wmma) ----
  //
  // VirtualGPU's fragment layout. PTX leaves the element->register mapping
  // unspecified, so we define a self-consistent one:
  //   A/B (f16): 8 x .b32 per lane = 16 halves. Lane L, register r, half h
  //              holds element (row = L % 16, col = r * 2 + h). Lanes 16-31
  //              mirror lanes 0-15, matching the hardware's 2x redundancy.
  //   C/D (f32): 8 x .f32 per lane = 256 slots exactly. Lane L register r
  //              holds linear element L * 8 + r (row-major 16x16).
  // A ".col" layout transposes the matrix on the way in/out.
  static constexpr uint32_t kMmaDim = 16;

  // ldmatrix: row r of matrix i comes from the address supplied by lane i*8+r,
  // and every lane leaves with two consecutive 16-bit elements of one row. The
  // distribution is the layout an mma A/B fragment expects, which is the whole
  // point of the instruction.
  void exec_ldmatrix(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpLdMatrix& op, Mask m) {
    Lanes _s_base;
    const Lanes& base = addr_base(w, ctx, ins, op.addr, _s_base);
    // Without ".shared" the register already holds an address in the shared
    // window -- it comes from cvta, which has done the space conversion, and
    // adding the window base again lands at twice it. With ".shared" the
    // instruction names the space itself and the register is a bare offset, so
    // the base has to be applied here instead.
    const uint64_t window = op.shared_space ? space_base(Space::Shared) : 0;
    auto row_addr = [&](uint32_t src_lane) {
      if (!(m & (Mask{1} << src_lane)))
        ctx_fail(ins, static_cast<int>(src_lane), Err::UnsupportedPtx,
                 "ldmatrix needs every lane that supplies a row address to be active");
      return window + base[src_lane] + static_cast<uint64_t>(op.addr.offset);
    };
    if (op.shape != LdmShape::M8N8) {
      // Rows of sixteen 8-bit values: from 16 bytes (.b8), from the packed
      // front of a padded group (.b4x16_p64: 8 bytes, .b6x16_p32: 12), each
      // value in the low bits of its byte -- CUTLASS shifts e2m1 up by 2
      // itself before an mma -- or from 8 bytes of .s4, sign-extended.
      const uint32_t R = op.shape == LdmShape::M16N16 ? 16 : 8;
      exec::LdmRow fmt;
      fmt.bits = op.fmt == LdmSrc::B6P32 ? 6 : op.fmt == LdmSrc::B8 ? 8 : 4;
      fmt.sign4 = op.fmt == LdmSrc::S4;
      for (uint32_t mat = 0; mat < op.count; ++mat) {
        uint8_t tile[16][16] = {};
        for (uint32_t r = 0; r < R; ++r) {
          const uint32_t src_lane = mat * R + r;
          const uint64_t addr = row_addr(src_lane);
          uint8_t raw[16];
          for (uint32_t b = 0; b < fmt.bytes(); ++b)
            raw[b] = static_cast<uint8_t>(load_routed(w, ctx, ins, src_lane, addr + b, 1));
          exec::ldm_unpack_row(raw, fmt, tile[r]);
        }
        const uint32_t regs = R / 8;
        for (uint32_t rr = 0; rr < regs; ++rr) {
          Lanes out;
          for (uint32_t lane = 0; lane < W_; ++lane)
            if (m & (Mask{1} << lane)) out[lane] = exec::ldm_word(tile, lane, rr, op.trans);
          write_reg(w, op.dsts[mat * regs + rr], m, out, 32);
        }
      }
      return;
    }
    for (uint32_t mat = 0; mat < op.count; ++mat) {
      // Pull the 8x8 matrix in, a row at a time.
      uint16_t tile[8][8] = {};
      for (uint32_t r = 0; r < 8; ++r) {
        const uint32_t src_lane = mat * 8 + r;
        const uint64_t addr = row_addr(src_lane);
        for (uint32_t c = 0; c < 8; ++c)
          tile[r][c] = static_cast<uint16_t>(
              load_routed(w, ctx, ins, src_lane, addr + c * 2, 2));
      }
      Lanes r;
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          const uint32_t row = lane / 4, colpair = lane % 4;
          uint16_t lo, hi;
          if (op.trans) {
            lo = tile[colpair * 2][row];
            hi = tile[colpair * 2 + 1][row];
          } else {
            lo = tile[row][colpair * 2];
            hi = tile[row][colpair * 2 + 1];
          }
          r[lane] = (static_cast<uint64_t>(hi) << 16) | lo;
        }
      write_reg(w, op.dsts[mat], m, r, 32);
    }
  }

  // add, sub, mul and fma on four packed narrow floats (PTX ISA 9.7.6): each lane's operands decoded,
  // combined exactly, and rounded to nearest-even into e5m2 or e4m3 (.satfinite clamps; without it an e4m3
  // result past its largest finite value is NaN, an e5m2 one infinity). A sum of two values whose
  // exponents are too far apart for a double breaks a rounding tie by the sign of what the double lost.
  void exec_f8x4_arith(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpF8x4Arith& op, Mask m) {
    Lanes sa, sb, sc;
    const Lanes a = read_operand(w, ctx, ins, op.a, sa);
    const Lanes b = read_operand(w, ctx, ins, op.b, sb);
    const bool fma = op.kind == F8x4Op::Fma;
    const Lanes c = fma ? Lanes(read_operand(w, ctx, ins, op.c, sc)) : Lanes{};
    const Fp8Format& df = op.dfmt == NarrowFmt::E5M2 ? kE5M2 : kE4M3;
    const auto element = [&](const F8x4Src& s, uint64_t reg, uint32_t i) -> double {
      uint32_t code;
      if (s.nibbles) code = static_cast<uint32_t>(reg >> (4 * i)) & 0xF;
      else {
        code = static_cast<uint32_t>(reg >> (8 * i)) & 0xFF;
        if (s.padded_nibble) code &= 0xF;
      }
      switch (s.fmt) {
        case NarrowFmt::E4M3: return fp8_to_double(code, kE4M3);
        case NarrowFmt::E5M2: return fp8_to_double(code, kE5M2);
        case NarrowFmt::E2M3: return small_float_value(code & 0x3F, 2, 3, 1);
        case NarrowFmt::E3M2: return small_float_value(code & 0x3F, 3, 2, 3);
        case NarrowFmt::E2M1: return small_float_value(code & 0xF, 2, 1, 1);
        case NarrowFmt::UE8M0: return code == 0xFF ? std::numeric_limits<double>::quiet_NaN() : std::ldexp(1.0, int(code) - 127);
        default: return 0.0;
      }
    };
    Lanes r;
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      uint32_t out = 0;
      for (uint32_t i = 0; i < 4; ++i) {
        const double x = element(op.asrc, a[lane], i), y = element(op.bsrc, b[lane], i);
        double v;
        switch (op.kind) {
          case F8x4Op::Add: v = x + y; break;
          case F8x4Op::Sub: v = x - y; break;
          case F8x4Op::Mul: v = x * y; break;
          default: {
            const double z = fp8_to_double((static_cast<uint32_t>(c[lane]) >> (8 * i)) & 0xFF, df);
            const double p = x * y;   // exact: at most 8 bits by 8
            v = p + z;
            if (std::isfinite(p) && std::isfinite(z) && std::isfinite(v)) {   // the sum's rounding error, as a nudge
              const double bb = v - p, err = (p - (v - bb)) + (z - bb);
              if (err != 0) v = std::nextafter(v, err > 0 ? INFINITY : -INFINITY);
            }
          }
        }
        if (op.kind == F8x4Op::Add || op.kind == F8x4Op::Sub) {
          if (std::isfinite(x) && std::isfinite(y) && std::isfinite(v)) {
            const double yy = op.kind == F8x4Op::Sub ? -y : y;
            const double bb = v - x, err = (x - (v - bb)) + (yy - bb);
            if (err != 0) v = std::nextafter(v, err > 0 ? INFINITY : -INFINITY);
          }
        }
        out |= double_to_fp8(v, df, op.satfinite) << (8 * i);
      }
      r[lane] = out;
    }
    write_reg(w, op.dst, m, r, 32);
  }

  // spcompress (PTX ISA 9.4, 9.7.10.30; figure 43): per thread, every group of four
  // elements of `data` gives its two selected ones, in index order, to `cdata`
  // and their indices to `mdata` (low bits first, group after group). The
  // descriptor picks the selection (MAX, MAXABS, MIN, MINABS) and the element
  // type. Where the ISA leaves a tie to the implementation, the lower index
  // goes first; a NaN is always selected, and -0.0 is below +0.0.
  void exec_spcompress(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpSpCompress& op, Mask m) {
    std::vector<Lanes> data(op.data.size());
    for (size_t i = 0; i < op.data.size(); ++i) {
      Lanes scratch;
      data[i] = read_operand(w, ctx, ins, op.data[i], scratch);
    }
    Lanes _s_desc;
    const Lanes desc = read_operand(w, ctx, ins, op.spdesc, _s_desc);
    std::vector<Lanes> cdata(op.cdata.size()), mdata(op.mdata.size());
    const uint32_t per_reg = 32 / op.elem_bits, groups_per_unit = 8 / (op.elem_bits / 8) / 4;   // 2 for b8, 1 for b16
    (void)per_reg;
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const uint32_t d = static_cast<uint32_t>(desc[lane]);
      const uint32_t sel = d & 3, type = (d >> 2) & 7;
      TcType tt = TcType::U8;
      if (op.elem_bits == 8) {
        switch (type) {
          case 0: tt = TcType::U8; break;
          case 1: tt = TcType::S8; break;
          case 2: tt = TcType::E5M2; break;
          case 3: tt = TcType::E4M3; break;
          case 4: tt = TcType::E3M2; break;
          case 5: tt = TcType::E2M3; break;
          default: ctx_fail(ins, static_cast<int>(lane), Err::InvalidValue, "spcompress: the descriptor's element type " + std::to_string(type) + " does not exist");
        }
      } else {
        if (type > 1) ctx_fail(ins, static_cast<int>(lane), Err::InvalidValue, "spcompress.b16: the descriptor's element type must be 0 (.f16) or 1 (.bf16), not " + std::to_string(type));
        tt = type == 0 ? TcType::F16 : TcType::BF16;
      }
      const uint32_t emask = op.elem_bits == 8 ? 0xFFu : 0xFFFFu;
      uint64_t cbits[64] = {}, mbits[64] = {};   // the output streams, 32-bit words packed into 64
      uint32_t cpos = 0, mpos = 0;
      const uint32_t ngroups = op.num * groups_per_unit * 1;
      for (uint32_t g = 0; g < ngroups; ++g) {
        uint32_t code[4];
        double val[4];
        for (uint32_t j = 0; j < 4; ++j) {
          const uint32_t elem = g * 4 + j, bit = elem * op.elem_bits;
          code[j] = static_cast<uint32_t>(data[bit / 32][lane] >> (bit % 32)) & emask;
          val[j] = tc_decode(tt, code[j]);
        }
        uint32_t pick[2];
        sp24_pick(val, sel, pick);
        for (uint32_t k = 0; k < 2; ++k) {
          const uint64_t cbit = uint64_t{cpos} * op.elem_bits, mbit = uint64_t{mpos} * op.idx_bits;
          cbits[cbit / 32] |= uint64_t{code[pick[k]]} << (cbit % 32);
          mbits[mbit / 32] |= uint64_t{pick[k]} << (mbit % 32);
          ++cpos;
          ++mpos;
        }
      }
      for (size_t i = 0; i < op.cdata.size(); ++i) cdata[i][lane] = static_cast<uint32_t>(cbits[i]);
      for (size_t i = 0; i < op.mdata.size(); ++i) mdata[i][lane] = static_cast<uint32_t>(mbits[i]);
    }
    for (size_t i = 0; i < op.cdata.size(); ++i) write_reg(w, op.cdata[i], m, cdata[i], 32);
    for (size_t i = 0; i < op.mdata.size(); ++i) write_reg(w, op.mdata[i], m, mdata[i], 32);
  }

  // spdecompress (9.7.10.31; figure 44): iteration N of .num puts each of its n_src
  // elements of `cdata` at the position `mdata` gives among n_target, the rest
  // zero. All three operands are bit streams across their 32-bit registers,
  // low bits first. An index past n_target is the ISA's implementation-specific
  // case: here the element is dropped.
  void exec_spdecompress(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpSpDecompress& op, Mask m) {
    std::vector<Lanes> mdata(op.mdata.size()), cdata(op.cdata.size()), data(op.data.size());
    for (size_t i = 0; i < op.mdata.size(); ++i) { Lanes s; mdata[i] = read_operand(w, ctx, ins, op.mdata[i], s); }
    for (size_t i = 0; i < op.cdata.size(); ++i) { Lanes s; cdata[i] = read_operand(w, ctx, ins, op.cdata[i], s); }
    const uint64_t emask = (uint64_t{1} << op.elem_bits) - 1, imask = (uint64_t{1} << op.idx_bits) - 1;
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const auto bits_at = [&](const std::vector<Lanes>& v, uint64_t bit, uint32_t n) {
        uint64_t x = 0;
        for (uint32_t k = 0; k < n; ++k) x |= ((v[(bit + k) / 32][lane] >> ((bit + k) % 32)) & 1) << k;
        return x;
      };
      std::vector<uint32_t> out(op.data.size(), 0);
      for (uint32_t n = 0; n < op.num; ++n)
        for (uint32_t s = 0; s < op.n_src; ++s) {
          const uint64_t c = bits_at(cdata, (uint64_t{n} * op.n_src + s) * op.elem_bits, op.elem_bits) & emask;
          const uint64_t idx = bits_at(mdata, (uint64_t{n} * op.n_src + s) * op.idx_bits, op.idx_bits) & imask;
          if (idx >= op.n_target) continue;
          const uint64_t at = (uint64_t{n} * op.n_target + idx) * op.elem_bits;
          out[at / 32] |= static_cast<uint32_t>(c << (at % 32));
        }
      for (size_t i = 0; i < out.size(); ++i) data[i][lane] = out[i];
    }
    for (size_t i = 0; i < op.data.size(); ++i) write_reg(w, op.data[i], m, data[i], 32);
  }

  // stmatrix: the inverse. Every lane hands over two consecutive 16-bit elements
  // of one row, the warp reassembles each 8x8 matrix, and row r of matrix i goes
  // to the address supplied by lane i*8+r -- the same lanes that would have
  // supplied it to ldmatrix, so a fragment loaded by one can be stored by the
  // other and land where it started.
  void exec_stmatrix(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpStMatrix& op, Mask m) {
    Lanes _s_base;
    // A copy, not a reference: reading the source registers below goes through
    // the widening ring, which would overwrite the addresses in place.
    const Lanes base = addr_base(w, ctx, ins, op.addr, _s_base);
    // Same reasoning as ldmatrix: with ".shared" the register is a bare offset
    // into the window, without it cvta has already made it an address.
    const uint64_t window = op.shared_space ? space_base(Space::Shared) : 0;
    if (op.m16n8) {
      // 16x8 of bytes (figure 111): element (row, col) is byte 2(row / 8) +
      // col % 2 of lane 4(row % 8) + col / 2 -- an mma accumulator's layout.
      // Stored transposed (.trans is mandatory): memory row c is column c,
      // 16 bytes, at the address lane 8 * matrix + c supplies.
      for (uint32_t mat = 0; mat < op.count; ++mat) {
        Lanes _s_val;
        const Lanes& val = read_operand(w, ctx, ins, op.srcs[mat], _s_val);
        for (uint32_t c = 0; c < 8; ++c) {
          const uint32_t dst_lane = mat * 8 + c;
          if (!(m & (Mask{1} << dst_lane)))
            ctx_fail(ins, static_cast<int>(dst_lane), Err::UnsupportedPtx,
                     "stmatrix needs every lane that supplies a row address to be active");
          const uint64_t addr = window + base[dst_lane] + static_cast<uint64_t>(op.addr.offset);
          for (uint32_t row = 0; row < 16; ++row) {
            const uint32_t lane = 4 * (row % 8) + c / 2, byte = 2 * (row / 8) + c % 2;
            if (!(m & (Mask{1} << lane)))
              ctx_fail(ins, static_cast<int>(lane), Err::UnsupportedPtx,
                       "stmatrix needs every lane of the warp active");
            store_routed(w, ctx, ins, dst_lane, addr + row, 1, (val[lane] >> (8 * byte)) & 0xFF);
          }
        }
      }
      return;
    }
    for (uint32_t mat = 0; mat < op.count; ++mat) {
      Lanes _s_val;
      const Lanes& val = read_operand(w, ctx, ins, op.srcs[mat], _s_val);
      // Put the matrix back together from what the lanes hold.
      uint16_t tile[8][8] = {};
      for (uint32_t lane = 0; lane < W_; ++lane) {
        if (!(m & (Mask{1} << lane))) continue;
        const uint32_t row = lane / 4, colpair = lane % 4;
        const uint16_t lo = static_cast<uint16_t>(val[lane] & 0xFFFFu);
        const uint16_t hi = static_cast<uint16_t>((val[lane] >> 16) & 0xFFFFu);
        if (op.trans) {
          tile[colpair * 2][row] = lo;
          tile[colpair * 2 + 1][row] = hi;
        } else {
          tile[row][colpair * 2] = lo;
          tile[row][colpair * 2 + 1] = hi;
        }
      }
      for (uint32_t r = 0; r < 8; ++r) {
        const uint32_t dst_lane = mat * 8 + r;
        if (!(m & (Mask{1} << dst_lane)))
          ctx_fail(ins, static_cast<int>(dst_lane), Err::UnsupportedPtx,
                   "stmatrix needs every lane that supplies a row address to be active");
        const uint64_t addr = window + base[dst_lane] + static_cast<uint64_t>(op.addr.offset);
        for (uint32_t c = 0; c < 8; ++c)
          store_routed(w, ctx, ins, dst_lane, addr + c * 2, 2, tile[r][c]);
      }
    }
  }

  // The warp holds an 8x8 matrix of 16-bit elements in the same layout ldmatrix
  // produces: lane L holds row L/4, columns 2*(L%4) and 2*(L%4)+1, packed into
  // one 32-bit register. Transposing it is a matter of re-gathering, since
  // every element already lives somewhere in the warp.
  // The addend half of mad.{lo,hi}.cc / madc: the multiply has already been
  // reduced to one value per lane, and only its addition with c touches the
  // carry bit. The product's own overflow is discarded -- ".lo" and ".hi" have
  // each already chosen which half of it survives.
  void exec_carry_mad(Warp& w, uint32_t bits, bool carry_in, bool carry_out, const Lanes& prod,
                      const Lanes& c, Mask m, Lanes& r) {
    Mask out = w.carry;
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const uint64_t p = mask_to_bits(prod[lane], bits);
      const uint64_t addend = mask_to_bits(c[lane], bits);
      const uint64_t cin = carry_in ? ((w.carry >> lane) & 1u) : 0u;
      const uint64_t sum = mask_to_bits(p + addend + cin, bits);
      r[lane] = sum;
      if (carry_out) {
        const bool cout = (sum < p) || (cin && sum == p);
        out = (out & ~(Mask{1} << lane)) | (Mask{cout} << lane);
      }
    }
    if (carry_out) w.carry = out;
  }

  // ---- texture and surface objects ----
  //
  // The sampling itself is shared with the SASS executor (fetch_texel and
  // surface_at, above the class); these read the operands, run it lane by
  // lane and add the instruction to any error.

  const TextureDesc& texture_for(const Instr& ins, int lane, uint64_t handle, TexKind want) {
    try {
      return texture_lookup(cfg_.textures, handle, want);
    } catch (const Error& e) {
      rethrow_with_context(e, ins, lane);
    }
  }

  void exec_tex(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpTex& op, Mask m) {
    Lanes _s_obj;
    const Lanes& obj = read_operand(w, ctx, ins, op.obj, _s_obj);
    const bool indexed = op.geom == TexGeom::A1D || op.geom == TexGeom::A2D || op.geom == TexGeom::ACube;
    const bool cube = op.geom == TexGeom::Cube || op.geom == TexGeom::ACube;
    const uint32_t first = indexed ? 1 : 0;
    std::array<Lanes, 4> coord;
    std::array<Lanes, 4> coord_scratch;
    for (uint32_t i = 0; i < op.dims + first; ++i)
      coord[i] = read_operand(w, ctx, ins, op.coords[i], coord_scratch[i]);
    Lanes _s_lod;
    const Lanes* lod = op.level ? &read_operand(w, ctx, ins, op.lod, _s_lod) : nullptr;
    // .grad: dPdx and dPdy, one operand per spatial coordinate (a 3D or cube texture's vectors carry a fourth,
    // unused one).
    std::array<Lanes, 3> ddx, ddy, ddx_scratch, ddy_scratch;
    if (op.grad)
      for (uint32_t i = 0; i < op.dims; ++i) {
        ddx[i] = read_operand(w, ctx, ins, op.ddx[i], ddx_scratch[i]);
        ddy[i] = read_operand(w, ctx, ins, op.ddy[i], ddy_scratch[i]);
      }
    std::array<Lanes, 3> offs;
    std::array<Lanes, 3> offs_scratch;
    const uint32_t noff = std::min<uint32_t>(3, static_cast<uint32_t>(op.offset.size()));
    for (uint32_t i = 0; i < noff; ++i) offs[i] = read_operand(w, ctx, ins, op.offset[i], offs_scratch[i]);
    // The depth-compare reference is read, so a bad register still faults, and
    // otherwise has no effect: CUDA gives a texture no depth-compare state, and
    // an RTX 3060 returns the plain fetch for tex and tld4 with a reference
    // (measured, point and linear).
    Lanes _s_dref;
    if (op.has_dref) (void)read_operand(w, ctx, ins, op.dref, _s_dref);

    std::array<Lanes, 4> out;
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const TextureDesc& d = texture_for(ins, static_cast<int>(lane), obj[lane], TexKind::Texture);
      TexFetch f;
      f.dims = op.dims;
      f.layered = indexed;
      f.cube = cube;
      f.layer = indexed ? static_cast<uint32_t>(coord[0][lane]) : 0;
      for (uint32_t i = 0; i < op.dims; ++i) f.coord[i] = static_cast<uint32_t>(coord[first + i][lane]);
      f.float_coords = op.ctype.is_float();
      // A half-precision result is the f32 one narrowed.
      f.float_result = op.dtype.is_float();
      f.explicit_lod = lod != nullptr;
      if (lod)
        f.lod = op.ctype.is_float() ? static_cast<double>(f32((*lod)[lane]))
                                    : static_cast<double>(static_cast<int32_t>((*lod)[lane]));
      f.gather = op.gather;
      if (op.grad) {
        f.grad = true;
        for (uint32_t i = 0; i < op.dims; ++i) {
          f.ddx[i] = static_cast<uint32_t>(ddx[i][lane]);
          f.ddy[i] = static_cast<uint32_t>(ddy[i][lane]);
        }
      }
      // Offsets are two's complement fields, as ptxas packs them into a
      // register for the instruction: four bits for tex (-8..7, a register
      // holding more wraps: measured), six for tld4 (to be measured).
      for (uint32_t i = 0; i < noff; ++i) {
        const uint32_t v = static_cast<uint32_t>(offs[i][lane]);
        f.offset[i] = op.gather >= 0 ? static_cast<int32_t>((v & 63u) ^ 32u) - 32   // tld4: six bits
                                     : static_cast<int32_t>((v & 15u) ^ 8u) - 8;
      }
      // A 1D texture's offset is one value, which ptxas hands the hardware as it is (an immediate
      // sign-extended, a register whole) and which reads as three nibbles: the second is the row of
      // the 2D texture a 1D one is, so a negative x offset (or one past 15) puts the fetch outside
      // it (measured with immediates and registers on an RTX 3060).
      if (op.geom == TexGeom::D1 && noff >= 1)
        f.offset[1] = static_cast<int32_t>((static_cast<uint32_t>(offs[0][lane]) >> 4 & 15u) ^ 8u) - 8;
      uint32_t r[4];
      try {
        texture_fetch(mem_, d, f, r);   // the shared entry: it resets the block cache and resolves .grad
      } catch (const Error& e) {
        rethrow_with_context(e, ins, static_cast<int>(lane));
      }
      for (uint32_t ch = 0; ch < 4; ++ch) out[ch][lane] = r[ch];
    }
    count_memory(Space::Global, 4 * 4, popcount_mask(m), /*is_store=*/false);
    if (op.pred_dst.id != kNoReg) {   // resident: every texel is
      Mask& p = pred_slot(w, op.pred_dst);
      p |= m;
    }
    if (op.dtype.is_float() && op.dtype.bits == 16) {
      // .f16: four halves, one per register; .f16x2: two registers of two.
      for (uint32_t lane = 0; lane < W_; ++lane) {
        if (!(m & (Mask{1} << lane))) continue;
        uint64_t h[4];
        for (uint32_t ch = 0; ch < 4; ++ch) h[ch] = double_to_f16(static_cast<double>(f32(static_cast<uint32_t>(out[ch][lane])))) & 0xFFFF;
        if (op.f16x2) {
          out[0][lane] = h[0] | (h[1] << 16);
          out[1][lane] = h[2] | (h[3] << 16);
        } else {
          for (uint32_t ch = 0; ch < 4; ++ch) out[ch][lane] = h[ch];
        }
      }
      for (uint32_t ch = 0; ch < op.dsts.size(); ++ch)
        write_reg(w, op.dsts[ch], m, out[ch], op.f16x2 ? 32 : 16);
      return;
    }
    for (uint32_t ch = 0; ch < 4 && ch < op.dsts.size(); ++ch)
      write_reg(w, op.dsts[ch], m, out[ch], 32);
  }

  std::optional<uint64_t> surface_address(const Instr& ins, uint32_t lane, const TextureDesc& d,
                                          const std::array<Lanes, 4>& coord, uint32_t dims, uint32_t bytes,
                                          bool layered, uint8_t oob) {
    const uint32_t first = layered ? 1 : 0;
    SurfaceAccess a;
    a.dims = dims;
    a.layered = layered;
    a.layer = layered ? static_cast<uint32_t>(coord[0][lane]) : 0;
    a.x = static_cast<int32_t>(coord[first][lane]);
    a.y = dims > 1 ? static_cast<int32_t>(coord[first + 1][lane]) : 0;
    a.z = dims > 2 ? static_cast<int32_t>(coord[first + 2][lane]) : 0;
    a.bytes = bytes;
    a.oob = oob;
    try {
      return surface_at(d, a);
    } catch (const Error& e) {
      rethrow_with_context(e, ins, static_cast<int>(lane));
    }
  }

  void exec_suld(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpSuld& op, Mask m) {
    Lanes _s_obj;
    const Lanes& obj = read_operand(w, ctx, ins, op.obj, _s_obj);
    std::array<Lanes, 4> coord;
    std::array<Lanes, 4> coord_scratch;
    for (uint32_t i = 0; i < op.dims + (op.layered ? 1 : 0); ++i)
      coord[i] = read_operand(w, ctx, ins, op.coords[i], coord_scratch[i]);
    std::vector<Lanes> out(op.dsts.size());
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const TextureDesc& d = texture_for(ins, static_cast<int>(lane), obj[lane], TexKind::Surface);
      const std::optional<uint64_t> base = surface_address(
          ins, lane, d, coord, op.dims, op.bytes * static_cast<uint32_t>(op.dsts.size()), op.layered, op.oob);
      for (size_t c = 0; c < op.dsts.size(); ++c) {
        if (!base) {   // .zero, out of range
          out[c][lane] = 0;
          continue;
        }
        try {
          out[c][lane] = mem_.load_scalar(*base + c * op.bytes, op.bytes);
        } catch (const Error& e) {
          rethrow_with_context(e, ins, static_cast<int>(lane));
        }
      }
    }
    count_memory(Space::Global, op.bytes * static_cast<uint32_t>(op.dsts.size()), popcount_mask(m),
                 /*is_store=*/false);
    for (size_t c = 0; c < op.dsts.size(); ++c)
      write_reg(w, op.dsts[c], m, out[c], op.bytes * 8 > 32 ? 64 : 32);
  }

  void exec_sust(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpSust& op, Mask m) {
    Lanes _s_obj;
    const Lanes& obj = read_operand(w, ctx, ins, op.obj, _s_obj);
    std::array<Lanes, 4> coord;
    std::array<Lanes, 4> coord_scratch;
    for (uint32_t i = 0; i < op.dims + (op.layered ? 1 : 0); ++i)
      coord[i] = read_operand(w, ctx, ins, op.coords[i], coord_scratch[i]);
    std::vector<Lanes> src(op.srcs.size());
    std::vector<Lanes> src_scratch(op.srcs.size());
    for (size_t c = 0; c < op.srcs.size(); ++c)
      src[c] = read_operand(w, ctx, ins, op.srcs[c], src_scratch[c]);
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const TextureDesc& d = texture_for(ins, static_cast<int>(lane), obj[lane], TexKind::Surface);
      if (op.formatted) {
        // sust.p: x counts texels, not bytes, and the values become the surface's format.
        uint32_t vals[4] = {0, 0, 0, 0};
        for (size_t c = 0; c < op.srcs.size() && c < 4; ++c) vals[c] = static_cast<uint32_t>(src[c][lane]);
        uint8_t texel[16];
        const uint32_t nb = surface_pack_texel(d, vals, static_cast<uint32_t>(std::min<size_t>(op.srcs.size(), 4)), texel);
        SurfaceAccess a;
        a.dims = op.dims;
        a.x = int64_t{static_cast<int32_t>(coord[0][lane])} * nb;
        a.y = op.dims > 1 ? static_cast<int32_t>(coord[1][lane]) : 0;
        a.z = op.dims > 2 ? static_cast<int32_t>(coord[2][lane]) : 0;
        a.bytes = nb;
        a.oob = op.oob;
        try {
          const std::optional<uint64_t> at = surface_at(d, a);
          if (at) mem_.write(*at, texel, nb);
        } catch (const Error& e) {
          rethrow_with_context(e, ins, static_cast<int>(lane));
        }
        continue;
      }
      const std::optional<uint64_t> base = surface_address(
          ins, lane, d, coord, op.dims, op.bytes * static_cast<uint32_t>(op.srcs.size()), op.layered, op.oob);
      if (!base) continue;   // .zero, out of range: the store is dropped
      for (size_t c = 0; c < op.srcs.size(); ++c) {
        try {
          mem_.store_scalar(*base + c * op.bytes, op.bytes, src[c][lane]);
        } catch (const Error& e) {
          rethrow_with_context(e, ins, static_cast<int>(lane));
        }
      }
    }
    count_memory(Space::Global, op.bytes * static_cast<uint32_t>(op.srcs.size()), popcount_mask(m),
                 /*is_store=*/true);
  }

  // add/sub with the condition-code carry bit. PTX defines subtraction's carry
  // as the carry-out of (a + ~b + 1), so a borrow clears the bit rather than
  // setting it -- which is what makes "sub.cc" then "subc" chain correctly into
  // a wider subtract.
  void exec_carry_add_sub(Warp& w, const OpIntBin& op, const Lanes& a, const Lanes& b, Mask m,
                          Lanes& r) {
    const uint32_t bits = op.ty.bits;
    const bool sub = (op.op == IntBinOp::Sub);
    Mask out = w.carry;
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const uint64_t x = mask_to_bits(a[lane], bits);
      // For a subtract the carry-in defaults to 1 (the +1 of two's complement);
      // subc supplies the previous carry-out in its place.
      const uint64_t cin =
          op.carry_in ? ((w.carry >> lane) & 1u) : (sub ? 1u : 0u);
      const uint64_t y = sub ? mask_to_bits(~b[lane], bits) : mask_to_bits(b[lane], bits);
      const uint64_t sum = mask_to_bits(x + y + cin, bits);
      r[lane] = sum;
      if (op.carry_out) {
        // Unsigned overflow: the sum wrapped below either addend, or landed
        // exactly on one because the carry-in pushed it around.
        const bool cout = (sum < x) || (cin && sum == x);
        out = (out & ~(Mask{1} << lane)) | (Mask{cout} << lane);
      }
    }
    if (op.carry_out) w.carry = out;
  }

  void exec_movmatrix(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpMovMatrix& op,
                      Mask m) {
    Lanes _s_a;
    const Lanes& a = read_operand(w, ctx, ins, op.src, _s_a);
    uint16_t tile[8][8] = {};
    for (uint32_t lane = 0; lane < W_; ++lane) {
      // .aligned means the whole warp executes this together; a partial warp
      // would be reading registers no lane wrote.
      if (!(m & (Mask{1} << lane)))
        ctx_fail(ins, static_cast<int>(lane), Err::UnsupportedPtx,
                 "movmatrix is warp-aligned: every lane must be active, because the "
                 "matrix is spread across all 32 of them");
      const uint32_t row = lane / 4, colpair = lane % 4;
      tile[row][colpair * 2] = static_cast<uint16_t>(a[lane] & 0xFFFFu);
      tile[row][colpair * 2 + 1] = static_cast<uint16_t>((a[lane] >> 16) & 0xFFFFu);
    }
    Lanes r{};
    for (uint32_t lane = 0; lane < W_; ++lane) {
      const uint32_t row = lane / 4, colpair = lane % 4;
      // d[row][c] = a[c][row]
      const uint16_t lo = tile[colpair * 2][row];
      const uint16_t hi = tile[colpair * 2 + 1][row];
      r[lane] = (static_cast<uint64_t>(hi) << 16) | lo;
    }
    write_reg(w, op.dst, m, r, 32);
  }

  // Bits in one element of an mma A/B fragment.
  static uint32_t mma_bits(MmaElem t) {
    switch (t) {
      case MmaElem::F16:
      case MmaElem::BF16: return 16;
      case MmaElem::TF32: return 32;
      case MmaElem::S4:
      case MmaElem::U4:
      case MmaElem::E2M1P: return 4;
      case MmaElem::B1: return 1;
      case MmaElem::F64: return 64;
      default: return 8;
    }
  }

  // Decodes element `slot` of an mma A/B fragment register.
  double mma_elem(MmaElem t, bool is_signed, uint64_t reg, uint32_t slot) {
    switch (t) {
      case MmaElem::F16: return f16_to_double((reg >> (16 * slot)) & 0xFFFF);
      case MmaElem::BF16: return bf16_to_double((reg >> (16 * slot)) & 0xFFFF);
      case MmaElem::TF32:
        // tf32 occupies a full 32-bit register, of which the tensor core
        // reads the top 19 bits: the low 13 mantissa bits are dropped, as an
        // RTX 3060 (sm_86) does with inputs that have them set.
        return static_cast<double>(f32(reg & 0xFFFFE000u));
      case MmaElem::F64: {
        double d;
        std::memcpy(&d, &reg, 8);
        return d;
      }
      case MmaElem::E4M3: return fp8_to_double(static_cast<uint32_t>(reg >> (8 * slot)) & 0xFF, kE4M3);
      case MmaElem::E5M2: return fp8_to_double(static_cast<uint32_t>(reg >> (8 * slot)) & 0xFF, kE5M2);
      case MmaElem::S4:
      case MmaElem::U4: {
        const uint32_t nib = static_cast<uint32_t>(reg >> (4 * slot)) & 0xF;
        return is_signed ? static_cast<double>(static_cast<int32_t>(nib << 28) >> 28) : static_cast<double>(nib);
      }
      case MmaElem::B1: return static_cast<double>((reg >> slot) & 1);
      // sm_120's narrow types (9.7.16.5.14): fp6 in the low 6 bits of its
      // byte, fp4 in the central 4 (bits 2-5); or fp4 packed, for the mxf4
      // kinds.
      case MmaElem::E3M2: return small_float_value(static_cast<uint32_t>(reg >> (8 * slot)) & 0x3F, 3, 2, 3);
      case MmaElem::E2M3: return small_float_value(static_cast<uint32_t>(reg >> (8 * slot)) & 0x3F, 2, 3, 1);
      // (An RTX PRO 6000 reads the byte's low six bits as an E2M3 for an E2M1 operand too -- the hash of a form
      // with random bytes in the containers says so -- which is the same number when bits 0-1 and 6-7 are zero,
      // as CUTLASS's and the ISA's layout leaves them.)
      case MmaElem::E2M1: return small_float_value(static_cast<uint32_t>(reg >> (8 * slot)) & 0x3F, 2, 3, 1);
      case MmaElem::E2M1P: return small_float_value(static_cast<uint32_t>(reg >> (4 * slot)) & 0xF, 2, 1, 1);
      case MmaElem::S8:
      case MmaElem::U8: {
        const uint8_t byte = static_cast<uint8_t>(reg >> (8 * slot));
        return is_signed ? static_cast<double>(static_cast<int8_t>(byte)) : static_cast<double>(byte);
      }
    }
    return 0.0;
  }

  // mma.sync (9.7.16.5). A, B, C and D are held as full tiles, filled from
  // and written back to the fragments by the per-shape formulas of
  // 9.7.16.5.1-13: for m16n8kK, with p elements to a register, A's register
  // r of lane (g, t) holds row g + 8(r % 2), columns tp + 4p(r / 2) on; B's
  // holds rows tp + 4pr on of column g; C/D element i is row g + 8(i / 2),
  // column 2t + i % 2. The m8n8kK shapes are the same with one register of
  // A and B and two elements of C. m8n8k4 f16 (Volta's) is four products at
  // once and has its own layouts.
  void exec_mma(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpMma& op, Mask m) {
    if (op.m == 8 && op.ab_type == MmaElem::F16) return exec_mma_m8n8k4(w, ctx, ins, op, m);
    constexpr uint32_t kN = 8;
    const uint32_t M = op.m, K = op.k;
    const uint32_t bits = mma_bits(op.ab_type);
    const bool wide = op.ab_type == MmaElem::F64;
    const uint32_t per_reg = wide ? 1u : 32u / bits;
    const uint32_t a_regs = (M * K / (op.sparse ? 2 : 1)) / (W_ * per_reg);
    const uint32_t b_regs = (K * kN) / (W_ * per_reg);
    const uint32_t c_elems = M * kN / W_;   // per lane: 4 for m16, 2 for m8
    if (op.a.size() != a_regs || op.b.size() != b_regs ||
        op.c.size() != (op.c_f16 ? c_elems / 2 : c_elems) || op.d.size() != (op.acc_f16 ? c_elems / 2 : c_elems))
      ctx_fail(ins, -1, Err::UnsupportedPtx, "mma fragment arity does not match the shape");
    // The largest shape is m16n8k256 (.b1): A 16 x 256, B 256 x 8.
    std::vector<double> A(size_t{M} * K, 0.0), B(size_t{K} * kN, 0.0);
    std::array<double, 16 * kN> C{};
    if (op.sparse) {
      // Sparse A (PTX ISA 9.7.16.6). Each K-wide row is cut into chunks --
      // four elements (f16, bf16, 8-bit), two (tf32, 1:2) or eight (int4,
      // 4:8 in pairs) -- and a chunk's metadata is 4 bits, two 2-bit indices
      // of "units": an element, half a tf32 element (0b0100 and 0b1110 are
      // tf32's only values), or a pair of int4 elements. Register r of lane
      // (g, t) holds the stored units of row g + 8(r % 2), chunks from
      // t * c + 4c(r / 2), c chunks to a register. The metadata for rows g
      // and g + 8 comes from the group's H holder lanes -- one, a pair, or all
      // four, as the ISA lists them; which lane carries which chunk is below.
      // Each form was checked against an RTX 3060 (sm_86): element by element
      // and nibble by nibble for f16 and int8, and by hashes of every result
      // for all of them.
      const bool tf32 = op.ab_type == MmaElem::TF32;
      // 4:8 in pairs for every 4-bit type: int4, and sm_120's packed e2m1.
      const bool int4 = mma_bits(op.ab_type) == 4;
      const uint32_t chunk_elems = tf32 ? 2 : int4 ? 8 : 4;
      const uint32_t stored = chunk_elems / 2;              // per chunk
      const uint32_t cpr = per_reg / stored;                // chunks per register
      Lanes _s_meta;
      const Lanes meta = read_operand(w, ctx, ins, op.meta, _s_meta);
      const uint32_t sel = static_cast<uint32_t>(std::get<ImmInt>(op.selector).value);
      for (uint32_t reg = 0; reg < a_regs; ++reg) {
        Lanes _s;
        const Lanes& v = read_operand(w, ctx, ins, Operand{RegOperand{op.a[reg]}}, _s);
        for (uint32_t lane = 0; lane < W_; ++lane) {
          const uint32_t group = lane / 4, tid = lane % 4;
          const uint32_t row = group + (reg % 2) * 8;
          for (uint32_t e = 0; e < per_reg; ++e) {
            const uint32_t chunk = tid * cpr + (reg / 2) * 4 * cpr + e / stored;
            const uint32_t col = sparse_column(mma_bits(op.ab_type), K, sel, meta, row, chunk, e % stored);
            A[row * K + col] = mma_elem(op.ab_type, op.ab_signed, v[lane], e);
          }
        }
      }
    } else
    for (uint32_t reg = 0; reg < a_regs; ++reg) {
      Lanes _s;
      const Lanes& v = read_operand(w, ctx, ins, Operand{RegOperand{op.a[reg]}}, _s);
      for (uint32_t lane = 0; lane < W_; ++lane) {
        const uint32_t group = lane / 4, tid = lane % 4;
        const uint32_t row = group + (reg % 2) * 8;
        const uint32_t col0 = tid * per_reg + (reg / 2) * (per_reg * 4);
        for (uint32_t e = 0; e < per_reg; ++e)
          A[row * K + col0 + e] = mma_elem(op.ab_type, op.ab_signed, v[lane], e);
      }
    }
    for (uint32_t reg = 0; reg < b_regs; ++reg) {
      Lanes _s;
      const Lanes& v = read_operand(w, ctx, ins, Operand{RegOperand{op.b[reg]}}, _s);
      for (uint32_t lane = 0; lane < W_; ++lane) {
        const uint32_t group = lane / 4, tid = lane % 4;
        const uint32_t row0 = tid * per_reg + reg * (per_reg * 4);
        for (uint32_t e = 0; e < per_reg; ++e)
          B[(row0 + e) * kN + group] = mma_elem(op.b_type, op.b_signed, v[lane], e);
      }
    }
    auto cd_index = [](uint32_t lane, uint32_t slot) {
      const uint32_t group = lane / 4, tid = lane % 4;
      return (group + (slot / 2) * 8) * kN + tid * 2 + (slot % 2);
    };
    auto as_f64 = [](uint64_t v) { double d; std::memcpy(&d, &v, 8); return d; };
    for (uint32_t reg = 0; reg < op.c.size(); ++reg) {
      Lanes _s;
      const Lanes& v = read_operand(w, ctx, ins, Operand{RegOperand{op.c[reg]}}, _s);
      for (uint32_t lane = 0; lane < W_; ++lane) {
        if (op.c_f16) {
          C[cd_index(lane, reg * 2)] = f16_to_double(v[lane] & 0xFFFF);
          C[cd_index(lane, reg * 2 + 1)] = f16_to_double((v[lane] >> 16) & 0xFFFF);
        } else if (op.acc_int) {
          C[cd_index(lane, reg)] = static_cast<double>(static_cast<int32_t>(v[lane]));
        } else if (op.acc_f64) {
          C[cd_index(lane, reg)] = as_f64(v[lane]);
        } else {
          C[cd_index(lane, reg)] = static_cast<double>(f32(v[lane]));
        }
      }
    }
    // Block scaling (9.7.16.3): row i of A takes its factors from lane
    // 4(i % 8) + 2 * thread-id-a + i / 8 -- the selected pair of each quad,
    // rows 0-7 from the first thread and 8-15 from the second, as figure 46
    // draws them and CuTe's SFALayout has them -- and column j of B from lane
    // 4j + thread-id-b (figure 47); factor b of each is byte byte-id + b
    // (figure 48). ue8m0 is 2^(v - 127), ue4m3 an unsigned e4m3.
    std::array<std::array<double, 4>, 16> SA{};
    std::array<std::array<double, 4>, kN> SB{};
    if (op.block_scale) {
      auto uniform = [&](const Operand& o, const char* what) {
        Lanes _s;
        const Lanes& v = read_operand(w, ctx, ins, o, _s);
        const uint32_t first = static_cast<uint32_t>(v[static_cast<uint32_t>(std::countr_zero(m))]) & 0xFFFF;
        for (uint32_t lane = 0; lane < W_; ++lane)
          if ((m >> lane & 1) && (static_cast<uint32_t>(v[lane]) & 0xFFFF) != first)
            ctx_fail(ins, static_cast<int>(lane), Err::InvalidValue,
                     std::string("mma's ") + what + " differs between the threads of the warp");
        return first;
      };
      const uint32_t bida = uniform(op.sfa_byte, "byte-id-a"), tida = uniform(op.sfa_thread, "thread-id-a");
      const uint32_t bidb = uniform(op.sfb_byte, "byte-id-b"), tidb = uniform(op.sfb_thread, "thread-id-b");
      // Table 46.
      const uint32_t sv = op.scale_vec;
      const bool ok = sv == 1 ? bida < 4 && tida < 2 && bidb < 4 && tidb < 4
                    : sv == 2 ? (bida == 0 || bida == 2) && tida < 2 && (bidb == 0 || bidb == 2) && tidb < 4
                              : bida == 0 && tida < 2 && bidb == 0 && tidb < 4;
      if (!ok)
        ctx_fail(ins, -1, Err::InvalidValue,
                 "mma's scale selectors {" + std::to_string(bida) + ", " + std::to_string(tida) + "} / {" +
                     std::to_string(bidb) + ", " + std::to_string(tidb) + "} are not valid for .scale_vec::" +
                     std::to_string(sv) + "X (Table 46)");
      Lanes _sa, _sb;
      const Lanes& sfa = read_operand(w, ctx, ins, op.sfa, _sa);
      const Lanes& sfb = read_operand(w, ctx, ins, op.sfb, _sb);
      auto factor = [&](uint32_t byte) {
        if (op.ue4m3) return fp8_to_double(byte & 0x7F, kE4M3);
        return byte == 0xFF ? std::numeric_limits<double>::quiet_NaN() : std::ldexp(1.0, int(byte) - 127);
      };
      for (uint32_t i = 0; i < M; ++i)
        for (uint32_t b = 0; b < sv; ++b)
          SA[i][b] = factor((static_cast<uint32_t>(sfa[4 * (i % 8) + 2 * tida + i / 8]) >> (8 * (bida + b))) & 0xFF);
      for (uint32_t j = 0; j < kN; ++j)
        for (uint32_t b = 0; b < sv; ++b)
          SB[j][b] = factor((static_cast<uint32_t>(sfb[4 * j + tidb]) >> (8 * (bidb + b))) & 0xFF);
    }
    const uint32_t sblock = op.block_scale ? K / op.scale_vec : K;
    // D = A x B + C. Float products are exact in f32 for every input type
    // here and the sum is kept in f32, in K order; f64 is a chain of fused
    // multiply-adds in the instruction's rounding mode, as the ISA says;
    // integers accumulate exactly and then wrap or saturate; .b1 counts the
    // bits of A's row and B's column that .and/.xor leave set.
    std::array<double, 16 * kN> D{};
    // The narrow-float tensor core of sm_120 (E4M3, E5M2, E3M2, E2M3, E2M1 and packed E2M1 operands, float accumulate).
    const auto narrow_elem = [](MmaElem t) {
      return t == MmaElem::E4M3 || t == MmaElem::E5M2 || t == MmaElem::E3M2 || t == MmaElem::E2M3 ||
             t == MmaElem::E2M1 || t == MmaElem::E2M1P;
    };
    const bool narrow_tensor_core = profile_.cc_major == 12 && narrow_elem(op.ab_type) && narrow_elem(op.b_type) &&
                                    !op.acc_int && !op.acc_f64 && !op.acc_f16;
    const int prev_round = std::fegetround();
    if (op.acc_f64) {
      switch (op.rnd) {
        case FRound::Nearest: std::fesetround(FE_TONEAREST); break;
        case FRound::Zero: std::fesetround(FE_TOWARDZERO); break;
        case FRound::MinusInf: std::fesetround(FE_DOWNWARD); break;
        case FRound::PlusInf: std::fesetround(FE_UPWARD); break;
      }
    }
    for (uint32_t i = 0; i < M; ++i)
      for (uint32_t j = 0; j < kN; ++j) {
        if (op.ab_type == MmaElem::B1) {
          int64_t acc = static_cast<int64_t>(C[i * kN + j]);
          for (uint32_t k = 0; k < K; ++k) {
            const bool a = A[i * K + k] != 0, b = B[k * kN + j] != 0;
            acc += op.b1_and ? (a && b) : (a != b);
          }
          D[i * kN + j] = static_cast<double>(static_cast<int32_t>(acc));
        } else if (op.acc_int) {
          // .satfinite clamps after every 128 bits of K -- 16 int8 or 32
          // int4 elements -- not once at the end: an RTX 3060 (sm_86) gives
          // sat(sat(C + first half) + second half) for m16n8k32 int8 and
          // m16n8k64 int4, and one clamp for the 128-bit shapes.
          const uint32_t block = op.satfinite ? 128 / bits : K;
          int64_t acc = static_cast<int64_t>(C[i * kN + j]);
          for (uint32_t k0 = 0; k0 < K; k0 += block) {
            for (uint32_t k = k0; k < k0 + block && k < K; ++k)
              acc += static_cast<int64_t>(A[i * K + k]) * static_cast<int64_t>(B[k * kN + j]);
            if (op.satfinite)
              acc = std::clamp<int64_t>(acc, std::numeric_limits<int32_t>::min(),
                                        std::numeric_limits<int32_t>::max());
          }
          D[i * kN + j] = static_cast<double>(static_cast<int32_t>(acc));
        } else if (op.acc_f64) {
          double acc = C[i * kN + j];
          for (uint32_t k = 0; k < K; ++k) acc = std::fma(A[i * K + k], B[k * kN + j], acc);
          D[i * kN + j] = acc;
        } else if (narrow_tensor_core) {
          // sm_120's narrow-float tensor core: the exact sum, rounded once toward zero (mma_narrow_sum).
          double terms[256];
          for (uint32_t k = 0; k < K; ++k)
            terms[k] = op.block_scale ? (A[i * K + k] * SA[i][k / sblock]) * (B[k * kN + j] * SB[j][k / sblock])
                                      : A[i * K + k] * B[k * kN + j];
          D[i * kN + j] = static_cast<double>(vgpu::exec::mma_narrow_sum(terms, K, static_cast<float>(C[i * kN + j])));
        } else {
          float acc = static_cast<float>(C[i * kN + j]);
          if (op.block_scale)
            for (uint32_t k = 0; k < K; ++k)
              acc += static_cast<float>(A[i * K + k] * SA[i][k / sblock]) *
                     static_cast<float>(B[k * kN + j] * SB[j][k / sblock]);
          else
            for (uint32_t k = 0; k < K; ++k)
              acc += static_cast<float>(A[i * K + k]) * static_cast<float>(B[k * kN + j]);
          D[i * kN + j] = static_cast<double>(acc);
        }
      }
    if (op.acc_f64) std::fesetround(prev_round);
    for (uint32_t reg = 0; reg < op.d.size(); ++reg) {
      Lanes r;
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          if (op.acc_f16) {
            const uint64_t lo = double_to_f16(D[cd_index(lane, reg * 2)]);
            const uint64_t hi = double_to_f16(D[cd_index(lane, reg * 2 + 1)]);
            r[lane] = ((hi & 0xFFFF) << 16) | (lo & 0xFFFF);
          } else if (op.acc_int) {
            r[lane] = static_cast<uint32_t>(static_cast<int32_t>(D[cd_index(lane, reg)]));
          } else if (op.acc_f64) {
            const double d = D[cd_index(lane, reg)];
            uint64_t u;
            std::memcpy(&u, &d, 8);
            r[lane] = u;
          } else {
            // A NaN is the card's canonical one: which NaN the host's
            // additions kept depended on the operand order the compiler
            // chose (e4m3's NaN codes, a -O3 build against -O2).
            r[lane] = canon32(static_cast<float>(D[cd_index(lane, reg)]));
          }
        }
      if (!op.acc_int) alu_fault(r, m, op.acc_f64 ? 64 : 32);
      write_reg(w, op.d[reg], m, r, op.acc_f64 ? 64 : 32);
    }
  }

  // mma.m8n8k4 with .f16 A and B (9.7.16.5.1): four 8x8x4 products, one per
  // quad pair -- lanes 4q..4q+3 (the low group) and 4q+16..4q+19 (the high
  // group) -- each with its own A, B, C and D.
  void exec_mma_m8n8k4(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpMma& op, Mask m) {
    if (op.a.size() != 2 || op.b.size() != 2 || op.c.size() != (op.c_f16 ? 4u : 8u) ||
        op.d.size() != (op.acc_f16 ? 4u : 8u))
      ctx_fail(ins, -1, Err::UnsupportedPtx, "mma.m8n8k4 fragment arity");
    std::vector<Lanes> av(2), bv(2), cv(op.c.size());
    for (size_t r = 0; r < 2; ++r) {
      Lanes _s;
      av[r] = read_operand(w, ctx, ins, Operand{RegOperand{op.a[r]}}, _s);
      Lanes _t;
      bv[r] = read_operand(w, ctx, ins, Operand{RegOperand{op.b[r]}}, _t);
    }
    for (size_t r = 0; r < op.c.size(); ++r) {
      Lanes _s;
      cv[r] = read_operand(w, ctx, ins, Operand{RegOperand{op.c[r]}}, _s);
    }
    // Element i (0-7) of a lane's C/D fragment, as (row, column) of its tile.
    auto cd_pos = [&](uint32_t lane, uint32_t i, bool f16, uint32_t* row, uint32_t* col) {
      const uint32_t hi = lane >= 16 ? 4 : 0;
      if (f16) {
        *row = lane % 4 + hi;
        *col = i;
      } else {
        *row = (lane & 1) + (i & 2) + hi;
        *col = (i & 4) + (lane & 2) + (i & 1);
      }
    };
    std::vector<Lanes> out(op.d.size(), Lanes{});
    for (uint32_t q = 0; q < 4; ++q) {
      double A[8][4] = {}, B[4][8] = {}, C[8][8] = {};
      for (uint32_t half = 0; half < 2; ++half)
        for (uint32_t t = 0; t < 4; ++t) {
          const uint32_t lane = half * 16 + q * 4 + t;
          const uint32_t hi = half * 4;
          for (uint32_t i = 0; i < 4; ++i) {
            const double a = f16_to_double((av[i / 2][lane] >> (16 * (i % 2))) & 0xFFFF);
            const double b = f16_to_double((bv[i / 2][lane] >> (16 * (i % 2))) & 0xFFFF);
            if (op.a_row) A[t + hi][i] = a; else A[i + hi][t] = a;
            if (op.b_col) B[i][t + hi] = b; else B[t][i + hi] = b;
          }
          for (uint32_t i = 0; i < 8; ++i) {
            uint32_t r, c;
            cd_pos(lane, i, op.c_f16, &r, &c);
            C[r][c] = op.c_f16 ? f16_to_double((cv[i / 2][lane] >> (16 * (i % 2))) & 0xFFFF)
                               : static_cast<double>(f32(cv[i][lane]));
          }
        }
      for (uint32_t half = 0; half < 2; ++half)
        for (uint32_t t = 0; t < 4; ++t) {
          const uint32_t lane = half * 16 + q * 4 + t;
          for (uint32_t i = 0; i < 8; ++i) {
            uint32_t r, c;
            cd_pos(lane, i, op.acc_f16, &r, &c);
            float acc = static_cast<float>(C[r][c]);
            for (uint32_t k = 0; k < 4; ++k) acc += static_cast<float>(A[r][k]) * static_cast<float>(B[k][c]);
            if (op.acc_f16)
              out[i / 2][lane] |= (double_to_f16(acc) & 0xFFFF) << (16 * (i % 2));
            else
              out[i][lane] = f32bits(acc);
          }
        }
    }
    for (size_t r = 0; r < op.d.size(); ++r) {
      alu_fault(out[r], m, 32);
      write_reg(w, op.d[r], m, out[r], 32);
    }
  }

  // ---- Hopper warpgroup MMA (wgmma, sm_90a) ----
  //
  // Four consecutive warps form a warpgroup and compute one 64xNxK product,
  // but the work divides exactly along M: warp r of the group holds rows
  // 16r..16r+15 of A (when A is in registers) and of D, and every warp reads
  // all of B (PTX ISA figures 151-158). So each warp computes its own sixteen
  // rows when it reaches the instruction, and the group's result is the same
  // as if all four had waited for each other.
  //
  // The product completes when it is issued. The ISA makes it asynchronous and
  // leaves the accumulator undefined until wgmma.wait_group, and completing
  // early is one of the orders that allows -- which also means a kernel that
  // reads its accumulator before waiting is not caught here. fence,
  // commit_group and wait_group have nothing left to order.

  WgmmaDesc decode_wgmma_desc(const Instr& ins, uint64_t d) {
    try {
      return exec::decode_wgmma_desc(d);
    } catch (const Error& e) {
      rethrow_with_context(e, ins, -1);
    }
  }

  static uint32_t wgmma_elem_bytes(WgmmaElem t) {
    switch (t) {
      case WgmmaElem::F16:
      case WgmmaElem::BF16: return 2;
      case WgmmaElem::TF32: return 4;
      default: return 1;
    }
  }

  static double wgmma_decode(WgmmaElem t, uint64_t bits) {
    switch (t) {
      case WgmmaElem::F16: return f16_to_double(bits & 0xFFFF);
      case WgmmaElem::BF16: return bf16_to_double(bits & 0xFFFF);
      // "wgmma.mma_async operation involving type .tf32 will truncate lower
      // 13 bits of the 32-bit input data before multiplication is issued."
      case WgmmaElem::TF32: return static_cast<double>(f32(bits & 0xFFFFE000u));
      case WgmmaElem::E4M3: return fp8_to_double(bits & 0xFF, kE4M3);
      case WgmmaElem::E5M2: return fp8_to_double(bits & 0xFF, kE5M2);
      case WgmmaElem::S8: return static_cast<double>(static_cast<int8_t>(bits & 0xFF));
      case WgmmaElem::U8: return static_cast<double>(bits & 0xFF);
      case WgmmaElem::B1: return static_cast<double>(bits & 1);
    }
    return 0.0;
  }

  void exec_wgmma(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpWgmma& op, Mask m) {
    // Every form is .sync.aligned: the whole warp, and the whole warpgroup,
    // executes it together.
    const uint32_t linear0 = w.tid_x[0] + w.tid_y[0] * ctx.ntid[0] +
                             w.tid_z[0] * ctx.ntid[0] * ctx.ntid[1];
    const uint32_t warp_index = linear0 / W_;
    const uint32_t block_threads = ctx.ntid[0] * ctx.ntid[1] * ctx.ntid[2];
    if ((warp_index / 4 + 1) * 4 * W_ > block_threads)
      ctx_fail(ins, -1, Err::UnsupportedPtx,
               "wgmma needs a whole warpgroup -- four warps, warp ranks 4i..4i+3 -- and this "
               "block of " + std::to_string(block_threads) + " threads has no warp " +
                   std::to_string((warp_index / 4) * 4 + 3));
    if (m != all_)
      ctx_fail(ins, -1, Err::UnsupportedPtx,
               "wgmma is .sync.aligned: every thread of the warpgroup must execute it, and this "
               "warp reached it with some lanes inactive or predicated off");
    if (op.kind == WgmmaKind::Commit) w.wgmma_commits.push_back(w.wgmma_issued);
    if (op.kind != WgmmaKind::Mma) return;

    const uint32_t N = op.n, K = op.k;
    const uint32_t row0 = (warp_index % 4) * 16;   // this warp's rows of A and D
    const uint32_t ea = wgmma_elem_bytes(op.a_type), eb = wgmma_elem_bytes(op.b_type);
    const bool b1 = op.a_type == WgmmaElem::B1;
    const uint32_t abits = b1 ? 1 : 8 * ea;
    const bool int_form = op.d_type == WgmmaAcc::S32;
    const double sa = op.scale_a, sb = op.scale_b;
    // A sparse A stores half of each row (9.7.17.6): Kp columns, which this
    // warp expands into its 16 rows with its own metadata, the way mma.sp
    // does for an m16n8kK -- sparse_column. A chunk is four elements (two
    // stored) for 16- and 8-bit types and two (one stored) for tf32.
    const uint32_t Kp = op.sparse ? K / 2 : K;
    const uint32_t stored = abits == 32 ? 1 : 2;
    Lanes sp_meta{};
    uint32_t sp_sel = 0;
    if (op.sparse) {
      Lanes _s;
      sp_meta = read_operand(w, ctx, ins, op.sp_meta, _s);
      sp_sel = static_cast<uint32_t>(std::get<ImmInt>(op.sp_sel).value);
    }
    // One element of shared memory: a byte-addressed read, or for .b1 the
    // bit it is in (K-major only, eight to a byte).
    auto smem_elem = [&](const WgmmaDesc& d, bool k_major, WgmmaElem t, uint32_t bytes, uint32_t mn,
                         uint32_t k) {
      if (t == WgmmaElem::B1) {
        const uint64_t at = wgmma_smem_offset(d, true, 1, mn, k / 8);
        const uint64_t byte =
            load_routed(w, ctx, ins, 0, kSharedVaBase + cluster_address(ctx, cluster_rank_of(ctx), at), 1);
        return static_cast<double>((byte >> (k % 8)) & 1);
      }
      const uint64_t at = wgmma_smem_offset(d, k_major, bytes, mn, k);
      return wgmma_decode(t, load_routed(w, ctx, ins, 0,
                                         kSharedVaBase + cluster_address(ctx, cluster_rank_of(ctx), at), bytes));
    };

    // A, this warp's 16 x K slice.
    std::vector<double> A(size_t{16} * K, 0.0);
    if (op.a_regs) {
      // The same fragment an mma.m16n8kK A operand uses (figures 151, 153, 155):
      // lane (g, t) holds rows g and g+8, and the register index picks the row
      // half and the column block. Sparse, it is mma.sp's (figures 180-182).
      const uint32_t per_reg = 32 / abits;
      const uint32_t cpr = per_reg / stored;   // sparse chunks per register
      for (uint32_t reg = 0; reg < 4; ++reg) {
        Lanes _s;
        const Lanes& v = read_operand(w, ctx, ins, Operand{RegOperand{op.a[reg]}}, _s);
        for (uint32_t lane = 0; lane < W_; ++lane) {
          const uint32_t g = lane / 4, t = lane % 4;
          const uint32_t row = g + (reg % 2) * 8;
          const uint32_t col0 = t * per_reg + (reg / 2) * (per_reg * 4);
          for (uint32_t e = 0; e < per_reg; ++e) {
            const uint32_t col =
                op.sparse ? sparse_column(abits, K, sp_sel, sp_meta, row,
                                          t * cpr + (reg / 2) * 4 * cpr + e / stored, e % stored)
                          : col0 + e;
            A[row * K + col] = sa * wgmma_decode(op.a_type, v[lane] >> (abits * e));
          }
        }
      }
    }
    // The shared memory operands, read once for the warpgroup.
    const uint32_t group = warp_index / 4;
    auto [snap_it, first] = ctx.wgmma->pending.try_emplace({group, w.wgmma_issued++});
    WgmmaSnapshot& snap = snap_it->second;
    if (first) {
      if (!op.a_regs) {
        Lanes _s;
        const Lanes& dv = read_operand(w, ctx, ins, op.a_desc, _s);
        const WgmmaDesc d = decode_wgmma_desc(ins, dv[0]);
        // Sparse: the packed 64 x K/2 matrix, in the canonical layout of its
        // stored columns.
        snap.A.assign(size_t{64} * Kp, 0.0);
        for (uint32_t r = 0; r < 64; ++r)
          for (uint32_t k = 0; k < Kp; ++k)
            snap.A[r * Kp + k] = sa * smem_elem(d, op.trans_a == 0, op.a_type, ea, r, k);
      }
      // B, all of it: K x N, read as N rows of K.
      snap.B.assign(size_t{K} * N, 0.0);
      Lanes _s;
      const Lanes& dv = read_operand(w, ctx, ins, op.b_desc, _s);
      const WgmmaDesc d = decode_wgmma_desc(ins, dv[0]);
      for (uint32_t n = 0; n < N; ++n)
        for (uint32_t k = 0; k < K; ++k)
          snap.B[size_t{k} * N + n] = sb * smem_elem(d, op.trans_b == 0, op.b_type, eb, n, k);
    }
    if (!op.a_regs) {
      if (snap.A.empty())
        ctx_fail(ins, -1, Err::UnsupportedPtx,
                 "the warps of a warpgroup disagree on whether this wgmma's A is in registers or "
                 "shared memory");
      for (uint32_t r = 0; r < 16; ++r)
        for (uint32_t j = 0; j < Kp; ++j) {
          const uint32_t col = op.sparse ? sparse_column(abits, K, sp_sel, sp_meta, r, j / stored, j % stored) : j;
          A[r * K + col] = snap.A[size_t{row0 + r} * Kp + j];
        }
    }
    const std::vector<double>& B = snap.B;
    // scale-d: false means D = A*B, per the ISA. Read per lane; a kernel
    // passes a uniform value, and per lane is what that value means for the
    // accumulator elements each lane owns.
    Mask keep_d = all_;
    if (const auto* imm = std::get_if<ImmInt>(&op.scale_d)) {
      keep_d = imm->value ? all_ : 0;
    } else if (const auto* r = std::get_if<RegOperand>(&op.scale_d)) {
      keep_d = read_pred(w, ins, r->reg);
    } else {
      ctx_fail(ins, -1, Err::UnsupportedPtx, "wgmma scale-d must be a predicate or 0/1");
    }

    // Accumulator element e of a lane: rows g and g+8, two adjacent columns,
    // repeated across N in blocks of eight (figures 152, 154, 156).
    auto d_pos = [](uint32_t lane, uint32_t e, uint32_t* row, uint32_t* col) {
      const uint32_t g = lane / 4, t = lane % 4, j = e / 4, s = e % 4;
      *row = g + 8 * (s / 2);
      *col = 8 * j + 2 * t + (s % 2);
    };
    const bool f16_acc = op.d_type == WgmmaAcc::F16;
    const uint32_t elems = N / 2;   // accumulator elements per lane
    std::vector<Lanes> out(op.d.size());
    for (uint32_t reg = 0; reg < op.d.size(); ++reg) {
      Lanes _s;
      out[reg] = read_operand(w, ctx, ins, Operand{RegOperand{op.d[reg]}}, _s);
    }
    for (uint32_t lane = 0; lane < W_; ++lane) {
      const bool accumulate = keep_d & (Mask{1} << lane);
      for (uint32_t e = 0; e < elems; ++e) {
        uint32_t row, col;
        d_pos(lane, e, &row, &col);
        uint64_t& slot = out[f16_acc ? e / 2 : e][lane];
        if (int_form) {
          int64_t acc = accumulate ? static_cast<int32_t>(slot) : 0;
          for (uint32_t k = 0; k < K; ++k)
            acc += b1 ? int64_t{A[row * K + k] != 0 && B[size_t{k} * N + col] != 0}
                      : static_cast<int64_t>(A[row * K + k]) * static_cast<int64_t>(B[size_t{k} * N + col]);
          if (op.satfinite)
            acc = std::clamp<int64_t>(acc, std::numeric_limits<int32_t>::min(),
                                      std::numeric_limits<int32_t>::max());
          slot = static_cast<uint32_t>(static_cast<int32_t>(acc));
          continue;
        }
        const uint32_t shift = f16_acc ? 16 * (e % 2) : 0;
        float acc = 0.0f;
        if (accumulate)
          acc = f16_acc ? static_cast<float>(f16_to_double((slot >> shift) & 0xFFFF))
                        : f32(slot);
        // Products of every input type here are exact in f32, and the sum is
        // kept in f32 -- "at least single precision" for an f32 accumulator,
        // and more than the half precision an f16 one promises.
        for (uint32_t k = 0; k < K; ++k)
          acc += static_cast<float>(A[row * K + k]) * static_cast<float>(B[size_t{k} * N + col]);
        if (f16_acc) {
          const uint64_t h = double_to_f16(acc) & 0xFFFF;
          slot = (slot & ~(uint64_t{0xFFFF} << shift) & 0xFFFFFFFFu) | (h << shift);
        } else {
          slot = f32bits(acc);
        }
      }
    }
    for (uint32_t reg = 0; reg < op.d.size(); ++reg) {
      if (!int_form) alu_fault(out[reg], m, 32);
      write_reg(w, op.d[reg], m, out[reg], 32);
    }
    snap.taken |= static_cast<uint8_t>(1u << (warp_index % 4));
    if (snap.taken == 0xF) ctx.wgmma->pending.erase(snap_it);
  }

  // clusterlaunchcontrol (9.7.15.18-19). The 16-byte response is this
  // engine's own encoding -- the ISA calls it opaque and a kernel may only
  // decode it with query_cancel: bit 0 says whether a cluster was cancelled,
  // bits 32-127 hold its first CTA's x, y and z.
  void exec_clc(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpClc& op, Mask m) {
    if (op.kind != ClcKind::TryCancel) {
      Lanes _s_lo, _s_hi;
      const Lanes lo = read_operand(w, ctx, ins, Operand{RegOperand{op.resp_lo}}, _s_lo);
      const Lanes hi = read_operand(w, ctx, ins, Operand{RegOperand{op.resp_hi}}, _s_hi);
      if (op.kind == ClcKind::IsCanceled) {
        Mask& p = pred_slot(w, op.dst[0]);
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) p = (lo[lane] & 1) ? (p | (Mask{1} << lane)) : (p & ~(Mask{1} << lane));
        return;
      }
      auto coord = [&](uint32_t lane, int d) -> uint64_t {
        return d == 0 ? lo[lane] >> 32 : d == 1 ? hi[lane] & 0xFFFFFFFFu : hi[lane] >> 32;
      };
      for (size_t i = 0; i < op.dst.size(); ++i) {
        if (op.dst[i].id == kNoReg || (op.dim < 0 && i == 3)) continue;   // `_`, or .v4's unspecified 4th
        Lanes v{};
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) v[lane] = coord(lane, op.dim >= 0 ? op.dim : static_cast<int>(i));
        write_reg(w, op.dst[i], m, v, 32);
      }
      return;
    }
    Lanes _s_a, _s_b;
    const Lanes abase = addr_base(w, ctx, ins, op.addr, _s_a);
    const Lanes bbase = addr_base(w, ctx, ins, op.mbar, _s_b);
    const size_t nranks = ctx.cluster_state ? ctx.cluster_state->ranks.size() : 1;
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const int li = static_cast<int>(lane);
      const uint64_t at = shared_window(abase[lane] + static_cast<uint64_t>(op.addr.offset));
      const uint64_t bar = shared_window(bbase[lane] + static_cast<uint64_t>(op.mbar.offset));
      if (shared_ref(ctx, ins, li, at).owner != &ctx || shared_ref(ctx, ins, li, bar).owner != &ctx)
        ctx_fail(ins, li, Err::InvalidValue,
                 "clusterlaunchcontrol.try_cancel's response and barrier are in this CTA's shared "
                 "memory (.shared::cta)");
      if (at % 16) ctx_fail(ins, li, Err::MisalignedAccess,
                            "clusterlaunchcontrol.try_cancel's response must be 16-byte aligned");
      const auto got = clc_take();
      const uint64_t lo = got ? 1u | uint64_t{(*got)[0]} << 32 : 0;
      const uint64_t hi = got ? (*got)[1] | uint64_t{(*got)[2]} << 32 : 0;
      // The response lands and its 16 bytes complete on the barrier -- in
      // this CTA, or with .multicast::cluster::all in every CTA of the
      // cluster at the same offsets.
      const uint64_t aoff = at - kSharedVaBase, boff = bar - kSharedVaBase;
      for (uint64_t r = 0; r < nranks; ++r) {
        if (!op.multicast && r != cluster_rank_of(ctx)) continue;
        const uint64_t ra = kSharedVaBase + cluster_address(ctx, r, aoff);
        store_routed(w, ctx, ins, lane, ra, 8, lo);
        store_routed(w, ctx, ins, lane, ra + 8, 8, hi);
        Mbarrier& b = cluster_mbarrier(ctx, ins, li, kSharedVaBase + cluster_address(ctx, r, boff),
                                       "clusterlaunchcontrol.try_cancel");
        b.tx -= 16;
        complete_phase_if_done(b);
      }
    }
  }

  // ---- Blackwell's fifth-generation tensor core (tcgen05, sm_100a) ----
  //
  // Every tcgen05 operation completes when it is issued. The asynchronous
  // ones (mma, ld, st) may complete any time up to their tcgen05.commit or
  // tcgen05.wait, and at once is one of the orders that allows; the fences
  // and waits then have nothing left to order. A kernel that reads an
  // accumulator before waiting for it is therefore not caught here.

  // The CTAs an operation of .cta_group::n works on: this one, or the pair
  // it belongs to, even rank first (9.7.18.5.1).
  std::vector<const BlockCtx*> tcgen05_ctas(const BlockCtx& ctx, const Instr& ins, uint32_t group) {
    if (group == 1) return {&ctx};
    const uint32_t rank = cluster_rank_of(ctx);
    const size_t n = ctx.cluster_state ? ctx.cluster_state->ranks.size() : 1;
    if ((rank | 1) >= n || !ctx.cluster_state->ranks[rank ^ 1])
      ctx_fail(ins, -1, Err::InvalidValue,
               ".cta_group::2 works on a CTA pair -- two CTAs of a cluster whose %cluster_ctarank "
               "differs in the last bit -- and this CTA (rank " + std::to_string(rank) + " of a " +
                   std::to_string(n) + "-CTA cluster) has no peer");
    const BlockCtx* even = ctx.cluster_state->ranks[rank & ~1u];
    const BlockCtx* odd = ctx.cluster_state->ranks[rank | 1u];
    return {even, odd};
  }

  static TensorMemory& tmem_of(const BlockCtx& c) { return *c.tmem; }

  // A shared-memory address operand as a window address. tcgen05 takes both
  // .shared::cta/.shared::cluster offsets and, with no state space, generic
  // addresses, which here are already in the window.
  static uint64_t shared_window(uint64_t v) { return v >= kSharedVaBase ? v : kSharedVaBase + v; }

  uint32_t tcgen05_uniform(Warp& w, const BlockCtx& ctx, const Instr& ins, const Operand& o, Mask m,
                           const char* what) {
    Lanes _s;
    const Lanes& v = read_operand(w, ctx, ins, o, _s);
    uint32_t lead = W_;
    for (uint32_t lane = 0; lane < W_; ++lane)
      if (m & (Mask{1} << lane)) {
        if (lead == W_) lead = lane;
        else if (static_cast<uint32_t>(v[lane]) != static_cast<uint32_t>(v[lead]))
          ctx_fail(ins, static_cast<int>(lane), Err::UnsupportedPtx,
                   std::string("every thread of the warp must give tcgen05 the same ") + what +
                       "; the ISA leaves the result undefined otherwise");
      }
    return static_cast<uint32_t>(v[lead]);
  }

  void exec_tcgen05(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpTcgen05& op, Mask m) {
    if (!ctx.tmem) ctx_fail(ins, -1, Err::UnsupportedPtx, "tcgen05 outside a block context");
    const bool collective = op.kind != Tcgen05Kind::Mma && op.kind != Tcgen05Kind::Cp &&
                            op.kind != Tcgen05Kind::Shift && op.kind != Tcgen05Kind::Commit &&
                            op.kind != Tcgen05Kind::FenceBefore && op.kind != Tcgen05Kind::FenceAfter;
    if (collective && m != all_)
      ctx_fail(ins, -1, Err::UnsupportedPtx,
               "this tcgen05 instruction is .sync.aligned: every thread of the warp must execute it, "
               "and this warp reached it with some lanes inactive or predicated off");
    switch (op.kind) {
      case Tcgen05Kind::FenceBefore:
      case Tcgen05Kind::FenceAfter:
      case Tcgen05Kind::WaitLd:
      case Tcgen05Kind::WaitSt:
        return;
      case Tcgen05Kind::Relinquish:
        for (const BlockCtx* c : tcgen05_ctas(ctx, ins, op.cta_group)) tmem_of(*c).relinquished = true;
        return;
      case Tcgen05Kind::Alloc:
      case Tcgen05Kind::Dealloc:
        exec_tmem_alloc(w, ctx, ins, op, m);
        return;
      case Tcgen05Kind::Ld:
      case Tcgen05Kind::St:
        exec_tmem_ldst(w, ctx, ins, op, m);
        return;
      case Tcgen05Kind::Commit:
        exec_tcgen05_commit(w, ctx, ins, op, m);
        return;
      case Tcgen05Kind::Mma:
        // Single-thread semantics: every active thread issues a whole MMA.
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) exec_tcgen05_mma(w, ctx, ins, op, lane);
        return;
      case Tcgen05Kind::Cp:
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) exec_tcgen05_cp(w, ctx, ins, op, lane);
        return;
      case Tcgen05Kind::Shift:
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) exec_tcgen05_shift(w, ctx, ins, op, lane);
        return;
    }
  }

  // tcgen05.alloc/dealloc (9.7.18.7.1).
  void exec_tmem_alloc(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpTcgen05& op, Mask m) {
    const uint32_t ncols = tcgen05_uniform(w, ctx, ins, op.ncols, m, "nCols");
    const bool pow2 = ncols && !(ncols & (ncols - 1));
    if (ncols < 32 || ncols > TensorMemory::kCols || ncols % 32 || (!op.exclusive && !pow2))
      ctx_fail(ins, -1, Err::InvalidValue,
               "tcgen05." + std::string(op.kind == Tcgen05Kind::Alloc ? "alloc" : "dealloc") +
                   " of " + std::to_string(ncols) + " columns; " +
                   (op.exclusive ? "an .exclusive allocation takes a multiple of 32 from 32 to 512"
                                 : "a power of two from 32 to 512"));
    const std::vector<const BlockCtx*> ctas = tcgen05_ctas(ctx, ins, op.cta_group);
    TensorMemory& mine = tmem_of(ctx);
    if (op.kind == Tcgen05Kind::Alloc) {
      if (mine.relinquished)
        ctx_fail(ins, -1, Err::InvalidValue,
                 "tcgen05.alloc after this CTA ran tcgen05.relinquish_alloc_permit");
      uint32_t col = 0;
      if (op.cta_group == 2 && !mine.peer_alloc.empty()) {
        // The peer CTA's warp got here first and allocated for both.
        const auto [c, n] = mine.peer_alloc.front();
        mine.peer_alloc.pop_front();
        if (n != ncols)
          ctx_fail(ins, -1, Err::InvalidValue,
                   "the two CTAs of a pair ask .cta_group::2 tcgen05.alloc for different column "
                   "counts (" + std::to_string(n) + " and " + std::to_string(ncols) + ")");
        col = c;
      } else {
        // The lowest run aligned to its size that is free in every CTA the
        // allocation covers. Where the hardware puts it is not documented;
        // a kernel uses the address it is given.
        bool found = false;
        for (col = 0; col + ncols <= TensorMemory::kCols; col += op.exclusive ? 32 : ncols) {
          bool ok = true;
          for (const BlockCtx* c : ctas) {
            const TensorMemory& t = tmem_of(*c);
            if (!t.free_run(col, ncols) || t.exclusive || (op.exclusive && !t.live.empty())) ok = false;
          }
          if (ok) { found = true; break; }
        }
        if (!found) {
          // "tcgen05.alloc is a potentially blocking instruction": it waits
          // until another warp's tcgen05.dealloc frees enough. Only a warp of
          // a CTA the allocation covers that can still run could do that; with
          // none, the columns will never be freed, and on hardware this warp
          // would hang.
          bool someone_can_free = false;
          for (const BlockCtx* c : ctas)
            if (c->warps)
              for (const Warp& other : *c->warps)
                if (&other != &w && other.state == Warp::State::Ready) someone_can_free = true;
          if (!someone_can_free)
            ctx_fail(ins, -1, Err::LaunchConfig,
                     "deadlock: tcgen05.alloc of " + std::to_string(ncols) +
                         " columns waits for Tensor Memory that no other warp of the CTA can "
                         "free (none is still running)");
          w.retry = true;
          w.yield_now = true;
          return;
        }
        for (const BlockCtx* c : ctas) {
          TensorMemory& t = tmem_of(*c);
          if (t.cells.empty()) t.cells.assign(size_t{TensorMemory::kLanes} * TensorMemory::kCols, 0);
          t.live[col] = ncols;
          if (op.exclusive) t.exclusive = true;
          if (c != &ctx) t.peer_alloc.emplace_back(col, ncols);
        }
      }
      // The address -- lane 0, the first column -- goes to shared memory, in
      // this CTA, as a weak store.
      Lanes _s;
      const Lanes& base = addr_base(w, ctx, ins, op.addr, _s);
      uint32_t lead = 0;
      while (!(m & (Mask{1} << lead))) ++lead;
      const uint64_t at = shared_window(base[lead] + static_cast<uint64_t>(op.addr.offset));
      if (shared_ref(ctx, ins, static_cast<int>(lead), at).owner != &ctx)
        ctx_fail(ins, static_cast<int>(lead), Err::InvalidValue,
                 "tcgen05.alloc writes its address into this CTA's shared memory (.shared::cta)");
      if (at % 4) ctx_fail(ins, static_cast<int>(lead), Err::MisalignedAccess,
                           "tcgen05.alloc's destination must be 4-byte aligned");
      store_routed(w, ctx, ins, lead, at, 4, col);
      return;
    }
    const uint32_t taddr = tcgen05_uniform(w, ctx, ins, op.taddr, m, "taddr");
    const uint32_t col = taddr & 0xFFFF;
    if (op.cta_group == 2 && !mine.peer_dealloc.empty()) {
      const auto [c, n] = mine.peer_dealloc.front();
      mine.peer_dealloc.pop_front();
      if (c != col || n != ncols)
        ctx_fail(ins, -1, Err::InvalidValue,
                 "the two CTAs of a pair deallocate different Tensor Memory with "
                 ".cta_group::2 tcgen05.dealloc");
      return;
    }
    for (const BlockCtx* c : ctas) {
      TensorMemory& t = tmem_of(*c);
      auto it = t.live.find(col);
      if ((taddr >> 16) != 0 || it == t.live.end() || it->second != ncols)
        ctx_fail(ins, -1, Err::InvalidValue,
                 "tcgen05.dealloc of " + std::to_string(ncols) + " columns at column " +
                     std::to_string(col) + " (lane " + std::to_string(taddr >> 16) +
                     "), which is not an allocation tcgen05.alloc made" +
                     (c != &ctx ? " in the peer CTA" : ""));
      if (t.exclusive != op.exclusive)
        ctx_fail(ins, -1, Err::InvalidValue,
                 "Tensor Memory is deallocated with .exclusive if and only if it was allocated "
                 "with it");
      t.live.erase(it);
      t.exclusive = false;
      if (c != &ctx) t.peer_dealloc.emplace_back(col, ncols);
    }
  }

  // tcgen05.ld/st (9.7.18.8). A warp reaches the 32 lanes of its quarter of
  // Tensor Memory: warp w of a warpgroup, lanes 32(w%4) to 32(w%4)+31.
  void exec_tmem_ldst(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpTcgen05& op, Mask m) {
    const uint32_t taddr = tcgen05_uniform(w, ctx, ins, op.taddr, m, "taddr");
    const uint32_t linear0 = w.tid_x[0] + w.tid_y[0] * ctx.ntid[0] +
                             w.tid_z[0] * ctx.ntid[0] * ctx.ntid[1];
    const uint32_t quarter = (linear0 / W_) % 4;
    TensorMemory& t = tmem_of(ctx);
    const bool ld = op.kind == Tcgen05Kind::Ld;
    const char* name = ld ? "tcgen05.ld" : "tcgen05.st";
    const uint32_t lane0 = taddr >> 16, col0 = taddr & 0xFFFF;
    const uint32_t width = op.pack16 ? 2 : 1;   // columns per register
    // With .spcompress the load is num columns and the registers are half as many.
    std::vector<Lanes> vals(op.spcompress ? op.num : op.regs.size());
    if (!ld)
      for (size_t j = 0; j < op.regs.size(); ++j) {
        Lanes _s;
        vals[j] = read_operand(w, ctx, ins, Operand{RegOperand{op.regs[j]}}, _s);
      }
    for (uint32_t j = 0; j < vals.size(); ++j)
      for (uint32_t th = 0; th < W_; ++th) {
        uint32_t dl, dc;
        tmem_fragment(op.shape, th, j, &dl, &dc);
        const uint32_t lane = lane0 + dl;
        uint32_t col = col0 + dc * width;
        if (op.shape == Tcgen05Shape::S16x32bx2 && th >= 16) col += op.half_split;
        if (lane / 32 != quarter || lane >= TensorMemory::kLanes)
          ctx_fail(ins, static_cast<int>(th), Err::InvalidValue,
                   std::string(name) + " reaches Tensor Memory lane " + std::to_string(lane) +
                       ", and warp " + std::to_string(quarter) +
                       " of its warpgroup may only reach lanes " + std::to_string(32 * quarter) +
                       "-" + std::to_string(32 * quarter + 31) + " (9.7.18.8.1)");
        for (uint32_t c = col; c < col + width; ++c)
          if (c >= TensorMemory::kCols || !t.allocated(c))
            ctx_fail(ins, static_cast<int>(th), Err::OutOfBounds,
                     std::string(name) + " reaches Tensor Memory column " + std::to_string(c) +
                         ", which no tcgen05.alloc of this CTA has allocated");
        if (ld) {
          vals[j][th] = op.pack16 ? (t.at(lane, col) & 0xFFFF) | (t.at(lane, col + 1) << 16)
                                  : t.at(lane, col);
        } else if (op.pack16) {
          t.at(lane, col) = static_cast<uint32_t>(vals[j][th] & 0xFFFF);
          t.at(lane, col + 1) = static_cast<uint32_t>((vals[j][th] >> 16) & 0xFFFF);
        } else {
          t.at(lane, col) = static_cast<uint32_t>(vals[j][th]);
        }
      }
    if (ld && !op.spcompress)
      for (size_t j = 0; j < op.regs.size(); ++j) write_reg(w, op.regs[j], m, vals[j], 32);
    if (op.spcompress) {
      // Two of each four loaded columns, by the rowop (.max keeps the greatest, .min the smallest,
      // .abs by magnitude), into the cdata registers in column order, with their indices, two bits
      // each, in the mdata registers. A NaN is always kept; the reduction below is of the whole
      // load, as without .spcompress.
      const uint32_t sel = (op.red_max ? 0u : 2u) + (op.red_abs ? 1u : 0u);
      std::vector<Lanes> cd(op.regs.size()), md(op.sp_mdata.size());
      for (auto& l : md) l.fill(0);
      for (uint32_t th = 0; th < W_; ++th)
        for (uint32_t g = 0; g < op.num / 4; ++g) {
          double v[4];
          for (uint32_t j = 0; j < 4; ++j) v[j] = static_cast<double>(f32(static_cast<uint32_t>(vals[4 * g + j][th])));
          uint32_t pick[2];
          sp24_pick(v, sel, pick);
          for (uint32_t k = 0; k < 2; ++k) {
            const uint32_t e = 2 * g + k;
            cd[e][th] = vals[4 * g + pick[k]][th];
            md[e * 2 / 32][th] |= uint64_t{pick[k]} << (e * 2 % 32);
          }
        }
      for (size_t j = 0; j < op.regs.size(); ++j) write_reg(w, op.regs[j], m, cd[j], 32);
      for (size_t j = 0; j < op.sp_mdata.size(); ++j) write_reg(w, op.sp_mdata[j], m, md[j], 32);
    }
    if (op.red) {
      // .red (9.7.18.8.3): each thread's loaded columns, reduced. f32 as min
      // and max define it: -0.0 below +0.0, a NaN input ignored unless .NaN
      // (then the result is the canonical NaN), .abs comparing magnitudes.
      Lanes out{};
      for (uint32_t th = 0; th < W_; ++th) {
        uint32_t acc = 0;
        bool have = false, nan = false;
        for (size_t j = 0; j < vals.size(); ++j) {
          uint32_t v = static_cast<uint32_t>(vals[j][th]);
          if (op.red_type == 'f') {
            if (op.red_abs) v &= 0x7FFFFFFFu;
            if (std::isnan(f32(v))) {
              nan = true;
              continue;
            }
          }
          if (!have) {
            acc = v;
            have = true;
            continue;
          }
          bool less;   // v < acc
          if (op.red_type == 'f') {
            const float a = f32(v), b = f32(acc);
            less = a < b || (a == b && std::signbit(a) && !std::signbit(b));
          } else if (op.red_type == 's') {
            less = static_cast<int32_t>(v) < static_cast<int32_t>(acc);
          } else {
            less = v < acc;
          }
          if (op.red_max ? (!less && v != acc) : less) acc = v;
        }
        if (op.red_type == 'f' && (!have || (nan && op.red_nan))) acc = 0x7FFFFFFFu;
        out[th] = acc;
      }
      write_reg(w, op.red_dst, m, out, 32);
    }
  }

  // tcgen05.commit: an arrive-on, count 1, on the barrier -- or with
  // .multicast::cluster, on the barrier at the same offset in every CTA the
  // mask names -- once this thread's earlier MMAs are done, which here they
  // already are.
  void exec_tcgen05_commit(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpTcgen05& op,
                           Mask m) {
    Lanes _s_a, _s_m;
    const Lanes base = addr_base(w, ctx, ins, op.addr, _s_a);
    const Lanes mask = op.multicast ? read_operand(w, ctx, ins, op.cta_mask, _s_m) : Lanes{};
    const size_t nranks = ctx.cluster_state ? ctx.cluster_state->ranks.size() : 1;
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const int li = static_cast<int>(lane);
      const uint64_t at = shared_window(base[lane] + static_cast<uint64_t>(op.addr.offset));
      auto arrive = [&](uint64_t addr) {
        Mbarrier& b = cluster_mbarrier(ctx, ins, li, addr, "tcgen05.commit");
        b.arrived += 1;
        complete_phase_if_done(b);
      };
      if (!op.multicast) {
        arrive(at);
        continue;
      }
      const uint64_t cta_mask = mask[lane] & 0xFFFF;
      if (cta_mask >> nranks)
        ctx_fail(ins, li, Err::InvalidValue,
                 "tcgen05.commit's ctaMask names a CTA past the cluster's " + std::to_string(nranks));
      const uint64_t off = shared_ref(ctx, ins, li, at).off;
      for (uint64_t r = 0; r < nranks; ++r)
        if (cta_mask >> r & 1) arrive(kSharedVaBase + cluster_address(ctx, r, off));
    }
  }

  // tcgen05 through exec/tcgen05.hpp: the CTAs an operation of .cta_group::n
  // works on, their Tensor Memory and shared memory, and failures with this
  // instruction's place.
  class PtxTcHost final : public Tcgen05Host {
   public:
    PtxTcHost(Interpreter& in, Warp& w, const BlockCtx& ctx, const Instr& ins, uint32_t lane, uint32_t group)
        : in_(in), w_(w), ctx_(ctx), ins_(ins), lane_(lane), cs_(in.tcgen05_ctas(ctx, ins, group)),
          direct_(in.fast_enabled_ && !in.detect_races() && !in.mem_.shared_fault_armed()) {
      for (uint32_t i = 0; i < cs_.size(); ++i)
        if (cs_[i] == &ctx) me_ = i;
    }
    uint32_t ctas() const override { return static_cast<uint32_t>(cs_.size()); }
    uint32_t self() const override { return me_; }
    TensorMemory& tmem(uint32_t c) override { return *cs_[c]->tmem; }
    uint64_t smem_load(uint32_t c, uint64_t off, uint32_t bytes) override {
      // Straight from the CTA's shared memory, unless something must see each
      // load -- the race detector, a shared-memory fault -- or it is out of
      // bounds; those take load_routed, which reports as any load does.
      if (direct_) {
        const std::vector<uint8_t>* sm = cs_[c]->shared;
        if (sm && off + bytes <= sm->size()) {
          uint64_t v = 0;
          std::memcpy(&v, sm->data() + off, bytes);
          return v;
        }
      }
      return in_.load_routed(w_, ctx_, ins_, lane_,
                             kSharedVaBase + in_.cluster_address(ctx_, cluster_rank_of(*cs_[c]), off), bytes);
    }
    [[noreturn]] void fail(Err code, const std::string& why) override {
      in_.ctx_fail(ins_, static_cast<int>(lane_), code, why);
    }

   private:
    Interpreter& in_;
    Warp& w_;
    const BlockCtx& ctx_;
    const Instr& ins_;
    uint32_t lane_;
    std::vector<const BlockCtx*> cs_;
    bool direct_;
    uint32_t me_ = 0;
  };

  void exec_tcgen05_cp(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpTcgen05& op, uint32_t lane) {
    auto value = [&](const Operand& o) {
      Lanes _s;
      return read_operand(w, ctx, ins, o, _s)[lane];
    };
    PtxTcHost h(*this, w, ctx, ins, lane, op.cta_group);
    exec::tcgen05_cp(Tcgen05Cp{op.cp_shape, op.cp_multicast, op.cp_decompress,
                               static_cast<uint32_t>(value(op.d_tmem)), value(op.a)},
                     h);
  }

  void exec_tcgen05_shift(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpTcgen05& op, uint32_t lane) {
    Lanes _s;
    const uint32_t taddr = static_cast<uint32_t>(read_operand(w, ctx, ins, op.d_tmem, _s)[lane]);
    PtxTcHost h(*this, w, ctx, ins, lane, op.cta_group);
    exec::tcgen05_shift(taddr, h);
  }

  // One thread's tcgen05.mma (9.7.18.10.10.1), through exec/tcgen05.hpp with
  // its operands read.
  void exec_tcgen05_mma(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpTcgen05& op,
                        uint32_t lane) {
    auto value = [&](const Operand& o) {
      Lanes _s;
      return read_operand(w, ctx, ins, o, _s)[lane];
    };
    Tcgen05Mma m;
    m.kind = op.mma_kind;
    m.cta_group = op.cta_group;
    m.block_scale = op.block_scale;
    m.sparse = op.sparse;
    m.ws = op.ws;
    m.a_tmem = op.a_tmem;
    m.scale_vec = op.scale_vec;
    m.target_sm = op.target_sm;
    m.target_arch = op.target_arch;
    m.idesc = static_cast<uint32_t>(value(op.idesc));
    m.a = value(op.a);
    m.b = value(op.b_desc);
    m.d = static_cast<uint32_t>(value(op.d_tmem));
    if (const auto* imm = std::get_if<ImmInt>(&op.enable_d)) m.accumulate = imm->value != 0;
    else if (const auto* r = std::get_if<RegOperand>(&op.enable_d)) m.accumulate = read_pred(w, ins, r->reg) >> lane & 1;
    else ctx_fail(ins, static_cast<int>(lane), Err::UnsupportedPtx, "tcgen05.mma: enable-input-d must be a predicate or 0/1");
    if (op.sparse) m.sp_meta = static_cast<uint32_t>(value(op.sp_meta));
    if (op.block_scale) {
      m.scale_a = static_cast<uint32_t>(value(op.scale_a));
      m.scale_b = static_cast<uint32_t>(value(op.scale_b));
    }
    if (op.has_zero_mask) {
      m.has_zero_mask = true;
      m.zero_mask = value(op.zero_mask);
    }
    for (size_t i = 0; i < op.disable_lanes.size() && i < 8; ++i)
      m.disable_lanes[i] = static_cast<uint32_t>(value(op.disable_lanes[i]));
    m.scale_d = op.scale_d;
    m.collector = op.collector;
    m.collector_buf = op.collector_buf;
    PtxTcHost h(*this, w, ctx, ins, lane, op.cta_group);
    exec::tcgen05_mma(m, h);
  }

  // The same fragments as exec_wmma_mma below, read from and written to the
  // 32-bit register file in place, with the halves decoded from the table.
  bool fast_wmma_mma(Warp& w, const OpWmmaMma& op, Mask m) {
    if (op.generic || op.elem == WmmaElem::TF32 || op.any_wide || mem_.alu_fault_armed() || W_ != 32)
      return false;
    const bool bf = op.elem == WmmaElem::BF16;
    const int ab_regs = bf ? 4 : 8;
    if (op.a.size() != size_t(ab_regs) || op.b.size() != size_t(ab_regs) || op.c.size() != 8 ||
        op.d.size() != 8)
      return false;
#if defined(__x86_64__) && defined(__GNUC__)
    const bool avx2 = g_avx2;
#else
    constexpr bool avx2 = false;
#endif
    // Every element of A, B and C is written below, so none is initialized.
    // The tiles are indexed flat, row * 16 + column.
    alignas(32) float A[kMmaDim][kMmaDim], B[kMmaDim][kMmaDim], C[kMmaDim][kMmaDim];
    float* const af = &A[0][0];
    float* const bff = &B[0][0];
    float* const cf = &C[0][0];
    // A register nothing has written reads as zero, as read_operand has it.
    static const Lanes32 kZero{};
    auto regs = [&](const Reg& r) -> const Lanes32& {
      return r.id < w.regs32.size() && w.written32[r.id] ? w.regs32[r.id] : kZero;
    };
    // Element (lane, register, half) of an A or B fragment is matrix element
    // `linear` in row-major order, and the fragment's layout says whether that
    // is (row, column) or (column, row) of the tile: f16 puts lane l's sixteen
    // halves in row l, bf16 packs eight per lane.
    auto unpack = [&](const std::vector<Reg>& frag, MatLayout layout, float* t) {
      const bool row = layout == MatLayout::Row;
      if (!bf) {
        // f16: lane l's eight registers are row l of the fragment.
        const uint32_t* v[8];
        for (int reg = 0; reg < 8; ++reg) v[reg] = regs(frag[reg]).data();
        alignas(32) uint32_t w8[kMmaDim][8];
        if (avx2) {
          gather8x8(v, 0, w8[0]);
          gather8x8(v, 8, w8[8]);
        } else {
          for (uint32_t lane = 0; lane < kMmaDim; ++lane)
            for (int reg = 0; reg < 8; ++reg) w8[lane][reg] = v[reg][lane];
        }
        if (row) {
          for (uint32_t lane = 0; lane < kMmaDim; ++lane) f16x16_to_float(w8[lane], t + lane * kMmaDim);
          return;
        }
        alignas(32) float r[kMmaDim][kMmaDim];
        for (uint32_t lane = 0; lane < kMmaDim; ++lane) f16x16_to_float(w8[lane], r[lane]);
        for (uint32_t i = 0; i < kMmaDim; ++i)
          for (uint32_t c = 0; c < kMmaDim; ++c) t[c * kMmaDim + i] = r[i][c];
        return;
      }
      for (int reg = 0; reg < ab_regs; ++reg) {
        const uint32_t* v = regs(frag[reg]).data();
        const uint32_t lanes = bf ? W_ : kMmaDim;
        for (uint32_t lane = 0; lane < lanes; ++lane) {
          const uint32_t word = v[lane];
          const uint32_t base = bf ? lane * 8 + static_cast<uint32_t>(reg * 2)
                                   : lane * kMmaDim + static_cast<uint32_t>(reg * 2);
          for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t bits = (word >> (16 * h)) & 0xFFFF;
            const float x = bf ? static_cast<float>(bf16_to_double(bits)) : kF16ToFloat[bits];
            const uint32_t linear = base + h;
            t[row ? linear : (linear % kMmaDim) * kMmaDim + linear / kMmaDim] = x;
          }
        }
      }
    };
    unpack(op.a, op.alayout, af);
    unpack(op.b, op.blayout, bff);
    // C and D: register r of lane l is element l * 8 + r -- an 8 x 32
    // transpose each way.
    if (avx2) {
      const uint32_t* cr[8];
      for (int reg = 0; reg < 8; ++reg) cr[reg] = regs(op.c[reg]).data();
      for (uint32_t lane = 0; lane < W_; lane += 8) gather8x8(cr, lane, cf + lane * 8);
    } else {
      for (int reg = 0; reg < 8; ++reg) {
        const uint32_t* v = regs(op.c[reg]).data();
        for (uint32_t lane = 0; lane < W_; ++lane) cf[lane * 8 + static_cast<uint32_t>(reg)] = f32(v[lane]);
      }
    }
    alignas(32) float D[kMmaDim][kMmaDim];
    wmma_tile(D, A, B, C, kMmaDim);
    const float* const df = &D[0][0];
    // Every register of D, written after all of A, B and C have been read:
    // D may name the same registers as C.
    if (avx2 && m == all_lanes(W_)) {
      uint32_t* dr[8];
      for (int reg = 0; reg < 8; ++reg) dr[reg] = dst32(w, op.d[reg].id, m);
      for (uint32_t lane = 0; lane < W_; lane += 8) scatter8x8(df + lane * 8, dr, lane);
      return true;
    }
    for (int reg = 0; reg < 8; ++reg) {
      uint32_t* d = dst32(w, op.d[reg].id, m);
      for_active(m, W_, [&](uint32_t lane) {
        d[lane] = static_cast<uint32_t>(f32bits(df[lane * 8 + static_cast<uint32_t>(reg)]));
      });
    }
    return true;
  }

  // ---- WMMA for every shape and type the ISA has (the `generic` ones) ----
  //
  // A generic fragment holds the logical matrix: slot (lane, register,
  // element) is matrix element (lane * per_lane + register * per_reg +
  // element) mod the matrix's size, row by row. The ISA leaves the
  // distribution unspecified, so any consistent one is correct; this one
  // holds whole copies when the register count has room for more than one.
  static std::pair<uint32_t, uint32_t> wmma_slot(const WmmaGeom& g, uint32_t lane, uint32_t reg, uint32_t e) {
    const uint32_t per_reg = g.reg_bits / g.bits;
    const uint64_t linear = uint64_t{lane} * g.regs * per_reg + uint64_t{reg} * per_reg + e;
    const uint64_t idx = linear % (uint64_t{g.rows} * g.cols);
    return {static_cast<uint32_t>(idx / g.cols), static_cast<uint32_t>(idx % g.cols)};
  }
  static char wmma_frag(OpWmmaLoad::Which w) {
    return w == OpWmmaLoad::Which::A ? 'a' : w == OpWmmaLoad::Which::B ? 'b' : 'c';
  }
  // Element `elem` (counted in elements) of a matrix in memory: sub-byte
  // elements are packed, low bits first.
  uint64_t wmma_read(Warp& w, const BlockCtx& ctx, const Instr& ins, uint32_t lane, uint64_t base,
                     uint64_t elem, uint32_t bits) {
    if (bits >= 8) return load_routed(w, ctx, ins, lane, base + elem * (bits / 8), bits / 8);
    const uint64_t bit = elem * bits;
    return (load_routed(w, ctx, ins, lane, base + bit / 8, 1) >> (bit % 8)) & ((1u << bits) - 1);
  }
  // A fragment element as a number: the float types as doubles (exact), the
  // integers sign- or zero-extended, b1 as 0 or 1.
  static double wmma_value(WmmaType t, uint64_t bits) {
    switch (t) {
      case WmmaType::F16: return f16_to_double(bits & 0xFFFF);
      case WmmaType::BF16: return bf16_to_double(bits & 0xFFFF);
      case WmmaType::TF32: case WmmaType::F32: return std::bit_cast<float>(static_cast<uint32_t>(bits));
      case WmmaType::F64: return std::bit_cast<double>(bits);
      case WmmaType::S8: return static_cast<int8_t>(bits);
      case WmmaType::U8: return static_cast<uint8_t>(bits);
      case WmmaType::S4: return static_cast<int32_t>((bits & 0xF) << 28) >> 28;
      case WmmaType::U4: return static_cast<double>(bits & 0xF);
      case WmmaType::B1: return static_cast<double>(bits & 1);
      case WmmaType::S32: return static_cast<int32_t>(bits);
    }
    return 0;
  }

  void exec_wmma_load_generic(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpWmmaLoad& op, Mask m) {
    Lanes _s_base, _s_stride;
    const Lanes base = addr_base(w, ctx, ins, op.addr, _s_base);
    const Lanes& stride = read_operand(w, ctx, ins, op.stride, _s_stride);
    const uint32_t lead = first_set(m);
    const uint64_t addr0 = space_base(op.space) + base[lead] + static_cast<uint64_t>(op.addr.offset);
    const uint64_t ld = stride[lead];
    const WmmaGeom g = wmma_geom(wmma_frag(op.which), op.shape, op.type);
    const uint32_t per_reg = g.reg_bits / g.bits;
    const uint64_t mask = g.bits == 64 ? ~0ull : (1ull << g.bits) - 1;
    for (uint32_t reg = 0; reg < g.regs; ++reg) {
      Lanes r{};
      for (uint32_t lane = 0; lane < W_; ++lane) {
        if (!(m & (Mask{1} << lane))) continue;
        uint64_t packed = 0;
        for (uint32_t e = 0; e < per_reg; ++e) {
          const auto [row, col] = wmma_slot(g, lane, reg, e);
          const uint64_t elem = op.layout == MatLayout::Row ? uint64_t{row} * ld + col : uint64_t{col} * ld + row;
          uint64_t v = wmma_read(w, ctx, ins, lane, addr0, elem, g.bits) & mask;
          if (op.type == WmmaType::TF32) v = f32bits(f32_to_tf32(f32(v)));
          packed |= v << (e * g.bits);
        }
        r[lane] = packed;
      }
      write_reg(w, op.dsts[reg], m, r, g.reg_bits);
    }
  }

  void exec_wmma_store_generic(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpWmmaStore& op, Mask m) {
    Lanes _s_base, _s_stride;
    const Lanes base = addr_base(w, ctx, ins, op.addr, _s_base);
    const Lanes& stride = read_operand(w, ctx, ins, op.stride, _s_stride);
    const uint32_t lead = first_set(m);
    const uint64_t addr0 = space_base(op.space) + base[lead] + static_cast<uint64_t>(op.addr.offset);
    const uint64_t ld = stride[lead];
    const WmmaGeom g = wmma_geom('d', op.shape, op.type);
    const uint32_t per_reg = g.reg_bits / g.bits;
    const uint64_t mask = g.bits == 64 ? ~0ull : (1ull << g.bits) - 1;
    for (uint32_t reg = 0; reg < g.regs; ++reg) {
      Lanes _s_v;
      const Lanes v = read_operand(w, ctx, ins, op.src[reg], _s_v);
      for (uint32_t lane = 0; lane < W_; ++lane) {
        if (!(m & (Mask{1} << lane))) continue;
        for (uint32_t e = 0; e < per_reg; ++e) {
          const auto [row, col] = wmma_slot(g, lane, reg, e);
          const uint64_t elem = op.layout == MatLayout::Row ? uint64_t{row} * ld + col : uint64_t{col} * ld + row;
          store_routed(w, ctx, ins, lane, addr0 + elem * (g.bits / 8), g.bits / 8, (v[lane] >> (e * g.bits)) & mask);
        }
      }
    }
  }

  // Gathers a generic fragment into its logical matrix.
  void wmma_gather(Warp& w, const BlockCtx& ctx, const Instr& ins, const std::vector<Reg>& regs, char frag,
                   WmmaShape shape, WmmaType t, std::vector<double>& out) {
    const WmmaGeom g = wmma_geom(frag, shape, t);
    const uint32_t per_reg = g.reg_bits / g.bits;
    const uint64_t mask = g.bits == 64 ? ~0ull : (1ull << g.bits) - 1;
    out.assign(size_t{g.rows} * g.cols, 0.0);
    for (uint32_t reg = 0; reg < g.regs; ++reg) {
      Lanes _s;
      const Lanes& v = read_operand(w, ctx, ins, Operand{RegOperand{regs[reg]}}, _s);
      for (uint32_t lane = 0; lane < W_; ++lane)
        for (uint32_t e = 0; e < per_reg; ++e) {
          const auto [row, col] = wmma_slot(g, lane, reg, e);
          out[size_t{row} * g.cols + col] = wmma_value(t, (v[lane] >> (e * g.bits)) & mask);
        }
    }
  }
  // The f16/bf16 A and B fragments at m16n16k16 keep the layout exec_wmma_mma
  // reads (memory order, the mma transposing a column-major one), since a load
  // does not know which accumulator type the mma will use.
  void wmma_gather_legacy(Warp& w, const BlockCtx& ctx, const Instr& ins, const std::vector<Reg>& regs,
                          bool bf, MatLayout layout, std::vector<double>& out) {
    out.assign(size_t{kMmaDim} * kMmaDim, 0.0);
    for (size_t reg = 0; reg < regs.size(); ++reg) {
      Lanes _s;
      const Lanes& v = read_operand(w, ctx, ins, Operand{RegOperand{regs[reg]}}, _s);
      const uint32_t lanes_used = bf ? W_ : kMmaDim;
      for (uint32_t lane = 0; lane < lanes_used; ++lane)
        for (int h = 0; h < 2; ++h) {
          uint32_t row, col;
          if (bf) {
            const uint32_t linear = lane * 8 + static_cast<uint32_t>(reg * 2 + h);
            row = linear / kMmaDim;
            col = linear % kMmaDim;
          } else {
            row = lane;
            col = static_cast<uint32_t>(reg * 2 + h);
          }
          const uint64_t bits = (v[lane] >> (16 * h)) & 0xFFFF;
          const double x = bf ? bf16_to_double(bits) : f16_to_double(bits);
          if (layout == MatLayout::Row) out[size_t{row} * kMmaDim + col] = x;
          else out[size_t{col} * kMmaDim + row] = x;
        }
    }
  }

  void exec_wmma_mma_generic(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpWmmaMma& op, Mask m) {
    const uint32_t M = op.shape.m, N = op.shape.n, K = op.shape.k;
    std::vector<double> A, B, C;
    const bool legacy_ab = op.shape == WmmaShape{16, 16, 16} &&
                           (op.atype == WmmaType::F16 || op.atype == WmmaType::BF16);
    if (legacy_ab) {
      wmma_gather_legacy(w, ctx, ins, op.a, op.atype == WmmaType::BF16, op.alayout, A);
      wmma_gather_legacy(w, ctx, ins, op.b, op.btype == WmmaType::BF16, op.blayout, B);
    } else {
      wmma_gather(w, ctx, ins, op.a, 'a', op.shape, op.atype, A);
      wmma_gather(w, ctx, ins, op.b, 'b', op.shape, op.btype, B);
    }
    wmma_gather(w, ctx, ins, op.c, 'c', op.shape, op.ctype, C);
    std::vector<uint64_t> D(size_t{M} * N);
    const WmmaType at = op.atype;
    if (at == WmmaType::F16 || at == WmmaType::BF16 || at == WmmaType::TF32) {
      // Products and sums in f32, k in order, as the f32-accumulator path does;
      // an f16 result is that rounded once.
      for (uint32_t i = 0; i < M; ++i)
        for (uint32_t j = 0; j < N; ++j) {
          float acc = static_cast<float>(C[size_t{i} * N + j]);
          for (uint32_t k = 0; k < K; ++k) {
            const float prod = static_cast<float>(A[size_t{i} * K + k]) * static_cast<float>(B[size_t{k} * N + j]);
            acc = acc + prod;
          }
          D[size_t{i} * N + j] = op.dtype == WmmaType::F16 ? double_to_f16(acc) : f32bits(acc);
        }
    } else if (at == WmmaType::F64) {
      const int prev = op.rnd == FRound::Nearest ? 0 : std::fegetround();
      if (op.rnd != FRound::Nearest) g_directed_rounding.fetch_add(1, std::memory_order_relaxed);
      switch (op.rnd) {
        case FRound::Zero: std::fesetround(FE_TOWARDZERO); break;
        case FRound::MinusInf: std::fesetround(FE_DOWNWARD); break;
        case FRound::PlusInf: std::fesetround(FE_UPWARD); break;
        case FRound::Nearest: break;
      }
      for (uint32_t i = 0; i < M; ++i)
        for (uint32_t j = 0; j < N; ++j) {
          double acc = C[size_t{i} * N + j];
          for (uint32_t k = 0; k < K; ++k) acc = std::fma(A[size_t{i} * K + k], B[size_t{k} * N + j], acc);
          D[size_t{i} * N + j] = std::bit_cast<uint64_t>(acc);
        }
      if (op.rnd != FRound::Nearest) {
        std::fesetround(prev);
        g_directed_rounding.fetch_sub(1, std::memory_order_relaxed);
      }
    } else {
      // Integers, exactly; single bits as a population count of the AND or XOR.
      for (uint32_t i = 0; i < M; ++i)
        for (uint32_t j = 0; j < N; ++j) {
          int64_t acc = static_cast<int64_t>(C[size_t{i} * N + j]);
          for (uint32_t k = 0; k < K; ++k) {
            const int64_t a = static_cast<int64_t>(A[size_t{i} * K + k]);
            const int64_t b = static_cast<int64_t>(B[size_t{k} * N + j]);
            acc += at == WmmaType::B1 ? (op.b1_and ? (a & b) : (a ^ b)) : a * b;
          }
          if (op.satfinite)
            acc = std::clamp<int64_t>(acc, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());
          D[size_t{i} * N + j] = static_cast<uint32_t>(static_cast<int32_t>(acc));
        }
    }
    // Scatter D into its fragment.
    const WmmaGeom g = wmma_geom('d', op.shape, op.dtype);
    const uint32_t per_reg = g.reg_bits / g.bits;
    const uint64_t mask = g.bits == 64 ? ~0ull : (1ull << g.bits) - 1;
    for (uint32_t reg = 0; reg < g.regs; ++reg) {
      Lanes r{};
      for (uint32_t lane = 0; lane < W_; ++lane) {
        uint64_t packed = 0;
        for (uint32_t e = 0; e < per_reg; ++e) {
          const auto [row, col] = wmma_slot(g, lane, reg, e);
          packed |= (D[size_t{row} * N + col] & mask) << (e * g.bits);
        }
        r[lane] = packed;
      }
      if (op.dtype == WmmaType::F32 || op.dtype == WmmaType::F64) alu_fault(r, m, g.reg_bits);
      write_reg(w, op.d[reg], m, r, g.reg_bits);
    }
  }

  void exec_wmma_mma(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpWmmaMma& op, Mask m) {
    if (op.generic) {
      exec_wmma_mma_generic(w, ctx, ins, op, m);
      return;
    }
    // Gather A and B (16x16 f16 each) and C (16x16 f32) from the warp.
    // Held as f32, which every element type here fits exactly (f16 and bf16 are
    // both exact in a float), since the product is accumulated in f32 anyway.
    float A[kMmaDim][kMmaDim] = {}, B[kMmaDim][kMmaDim] = {}, C[kMmaDim][kMmaDim] = {};
    const bool bf = op.elem == WmmaElem::BF16;
    const bool tf = op.elem == WmmaElem::TF32;
    const int ab_regs = op.elem == WmmaElem::F16 ? 8 : 4;
    // k is 8 for tf32's m16n16k8 and 16 otherwise.
    const uint32_t K = tf ? 8u : kMmaDim;
    if (tf) {
      for (int reg = 0; reg < ab_regs; ++reg) {
        Lanes _s_av;
        const Lanes& av = read_operand(w, ctx, ins, Operand{RegOperand{op.a[reg]}}, _s_av);
        Lanes _s_bv;
        const Lanes& bv = read_operand(w, ctx, ins, Operand{RegOperand{op.b[reg]}}, _s_bv);
        for (uint32_t lane = 0; lane < W_; ++lane) {
          const uint32_t linear = lane * 4 + static_cast<uint32_t>(reg);
          // A is 16x8, B is 8x16: different inner extents, so different maps.
          const uint32_t ar = linear / 8u, ac = linear % 8u;
          const uint32_t br = linear / kMmaDim, bc = linear % kMmaDim;
          A[ar][ac] = f32(av[lane]);
          B[br][bc] = f32(bv[lane]);
        }
      }
    } else
    for (int reg = 0; reg < ab_regs; ++reg) {
      Lanes _s_av;
      const Lanes& av = read_operand(w, ctx, ins, Operand{RegOperand{op.a[reg]}}, _s_av);
      Lanes _s_bv;
      const Lanes& bv = read_operand(w, ctx, ins, Operand{RegOperand{op.b[reg]}}, _s_bv);
      // A bf16 fragment covers the matrix exactly once across all 32 lanes; an
      // f16 fragment covers it twice, so only the first 16 lanes are read.
      const uint32_t lanes_used = bf ? W_ : kMmaDim;
      for (uint32_t lane = 0; lane < lanes_used; ++lane) {
        for (int h = 0; h < 2; ++h) {
          uint32_t row, col;
          if (bf) {
            const uint32_t linear = lane * 8 + static_cast<uint32_t>(reg * 2 + h);
            row = linear / kMmaDim;
            col = linear % kMmaDim;
          } else {
            row = lane;
            col = static_cast<uint32_t>(reg * 2 + h);
          }
          const uint64_t abits = (av[lane] >> (16 * h)) & 0xFFFF;
          const uint64_t bbits = (bv[lane] >> (16 * h)) & 0xFFFF;
          const double a = bf ? bf16_to_double(abits) : f16_to_double(abits);
          const double b = bf ? bf16_to_double(bbits) : f16_to_double(bbits);
          // .row: element (row, col); .col: transposed.
          if (op.alayout == MatLayout::Row) A[row][col] = a; else A[col][row] = a;
          if (op.blayout == MatLayout::Row) B[row][col] = b; else B[col][row] = b;
        }
      }
    }
    for (int reg = 0; reg < 8; ++reg) {
      Lanes _s_cv;
      const Lanes& cv = read_operand(w, ctx, ins, Operand{RegOperand{op.c[reg]}}, _s_cv);
      for (uint32_t lane = 0; lane < W_; ++lane) {
        uint32_t linear = lane * 8 + static_cast<uint32_t>(reg);
        C[linear / kMmaDim][linear % kMmaDim] = f32(cv[lane]);
      }
    }
    // D = A x B + C, accumulated in f32 (matching the .f32 accumulate type).
    // Each element still accumulates its products in k order, exactly as the
    // element-by-element form did; running j innermost lets it vectorize.
    float D[kMmaDim][kMmaDim];
    wmma_tile(D, A, B, C, K);
    // Scatter D back into the destination fragment.
    for (int reg = 0; reg < 8; ++reg) {
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < W_; ++lane) {
        uint32_t linear = lane * 8 + static_cast<uint32_t>(reg);
        r[lane] = f32bits(D[linear / kMmaDim][linear % kMmaDim]);
      }
      alu_fault(r, m, 32);
      write_reg(w, op.d[reg], m, r, 32);
    }
  }

  // wmma.load.{a,b,c}. The addressing mirrors exactly what exec_wmma_mma reads
  // back out, because a WMMA fragment is opaque: CUDA specifies no register
  // assignment, and the only thing that has to hold is that load, mma and
  // store agree.
  //
  // For A and B the layout flag cancels out. mma interprets a .col fragment as
  // transposed, and .col memory is itself transposed, so both paths land on
  // the same element -- which is the point: a column-major A read with .col is
  // the same logical matrix as a row-major A read with .row.
  //
  // Only 16 of the 32 lanes carry distinct A/B data (16x16 halves over 32
  // lanes is exactly a factor of two of duplication, which is what the
  // hardware fragment does too), so lanes 16-31 duplicate lanes 0-15 rather
  // than reading rows 16-31 of a 16-row matrix, which would be out of bounds.
  void exec_wmma_load(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpWmmaLoad& op,
                      Mask m) {
    if (op.generic) {
      exec_wmma_load_generic(w, ctx, ins, op, m);
      return;
    }
    Lanes _s_base;
    const Lanes& base = addr_base(w, ctx, ins, op.addr, _s_base);
    Lanes _s_stride;
    const Lanes& stride = read_operand(w, ctx, ins, op.stride, _s_stride);
    const uint64_t sbase = space_base(op.space);
    uint32_t lead = 0;
    while (lead < W_ && !(m & (Mask{1} << lead))) ++lead;
    if (lead == W_) return;
    const uint64_t addr0 = sbase + base[lead] + static_cast<uint64_t>(op.addr.offset);
    const uint64_t ld = stride[lead];

    const bool bf = op.elem == WmmaElem::BF16;
    const bool tf = op.elem == WmmaElem::TF32;
    const int nregs =
        (op.which == OpWmmaLoad::Which::C || op.elem == WmmaElem::F16) ? 8 : 4;
    for (int reg = 0; reg < nregs; ++reg) {
      Lanes r;
      for (uint32_t lane = 0; lane < W_; ++lane) {
        if (!(m & (Mask{1} << lane))) continue;
        if (op.which == OpWmmaLoad::Which::C) {
          const uint32_t linear = lane * 8 + static_cast<uint32_t>(reg);
          const uint32_t i = linear / kMmaDim, j = linear % kMmaDim;
          const uint64_t elem = op.layout == MatLayout::Row ? (uint64_t{i} * ld + j)
                                                            : (uint64_t{j} * ld + i);
          r[lane] = load_routed(w, ctx, ins, lane, addr0 + elem * 4, 4);
        } else if (tf) {
          // m16n16k8: A is 16x8 and B is 8x16, so the two fragments do not
          // share an index map and the square shapes' layout cancellation
          // does not apply. The address is computed from the layout directly
          // instead: row-major puts (r,c) at r*ld+c, column-major at c*ld+r.
          const uint32_t linear = lane * 4 + static_cast<uint32_t>(reg);
          const uint32_t inner = op.which == OpWmmaLoad::Which::A ? 8u : kMmaDim;
          const uint32_t rr = linear / inner, cc = linear % inner;
          const uint64_t elem = op.layout == MatLayout::Row ? (uint64_t{rr} * ld + cc)
                                                            : (uint64_t{cc} * ld + rr);
          const uint64_t bits = load_routed(w, ctx, ins, lane, addr0 + elem * 4, 4);
          // tf32 keeps f32's exponent and 10 mantissa bits. Rounding to that
          // here is what the hardware fragment holds; keeping all 23 would
          // make the simulator more accurate than the part it stands in for.
          r[lane] = f32bits(f32_to_tf32(f32(bits)));
        } else {
          uint64_t packed = 0;
          for (int h = 0; h < 2; ++h) {
            uint32_t row, col;
            if (bf) {
              // 4 registers x 2 halves x 32 lanes is exactly 16x16, so a bf16
              // fragment covers the matrix once with no duplication and every
              // lane carries distinct data.
              const uint32_t linear = lane * 8 + static_cast<uint32_t>(reg * 2 + h);
              row = linear / kMmaDim;
              col = linear % kMmaDim;
            } else {
              // f16 takes 8 registers, which is twice the matrix, so lanes
              // 16-31 duplicate lanes 0-15 rather than reading rows that do
              // not exist.
              row = lane % kMmaDim;
              col = static_cast<uint32_t>(reg * 2 + h);
            }
            const uint64_t elem = uint64_t{row} * ld + col;
            const uint64_t v = load_routed(w, ctx, ins, lane, addr0 + elem * 2, 2);
            packed |= (v & 0xFFFF) << (16 * h);
          }
          r[lane] = packed;
        }
      }
      write_reg(w, op.dsts[reg], m, r, 32);
    }
  }

  void exec_wmma_store(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpWmmaStore& op,
                       Mask m) {
    if (op.generic) {
      exec_wmma_store_generic(w, ctx, ins, op, m);
      return;
    }
    Lanes _s_base;
    const Lanes& base = addr_base(w, ctx, ins, op.addr, _s_base);
    Lanes _s_stride;
      const Lanes& stride = read_operand(w, ctx, ins, op.stride, _s_stride);
    uint64_t sbase = space_base(op.space);
    // The whole warp cooperates; lane 0's address and stride describe the tile.
    uint32_t lead = 0;
    while (lead < W_ && !(m & (Mask{1} << lead))) ++lead;
    if (lead == W_) return;
    uint64_t addr0 = sbase + base[lead] + static_cast<uint64_t>(op.addr.offset);
    uint64_t ld = stride[lead];
    for (int reg = 0; reg < 8; ++reg) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op.src[reg], _s_v);
      for (uint32_t lane = 0; lane < W_; ++lane) {
        if (!(m & (Mask{1} << lane))) continue;
        uint32_t linear = lane * 8 + static_cast<uint32_t>(reg);
        uint32_t i = linear / kMmaDim, j = linear % kMmaDim;
        uint64_t elem = op.layout == MatLayout::Row ? (uint64_t{i} * ld + j)
                                                    : (uint64_t{j} * ld + i);
        store_routed(w, ctx, ins, lane, addr0 + elem * 4, 4, v[lane] & 0xFFFFFFFFull);
      }
    }
  }

  // Applies an integer rounding mode to a real value.
  static double round_int(double x, Round rnd) {
    switch (rnd) {
      case Round::Rzi: return std::trunc(x);
      case Round::Rmi: [[fallthrough]];
      case Round::Rm: return std::floor(x);
      case Round::Rpi: [[fallthrough]];
      case Round::Rp: return std::ceil(x);
      case Round::Rni: [[fallthrough]];
      default: return std::nearbyint(x);  // round-to-nearest-even
    }
  }

  // A directed rounding mode for the half-precision arithmetic: the host mode
  // set for the scope (double_to_narrow follows it) and put back after.
  struct HalfRounding {
    FRound mode;
    int prev = 0;
    explicit HalfRounding(FRound m) : mode(m) {
      if (mode == FRound::Nearest) return;
      prev = std::fegetround();
      g_directed_rounding.fetch_add(1, std::memory_order_relaxed);
      std::fesetround(mode == FRound::Zero ? FE_TOWARDZERO : mode == FRound::MinusInf ? FE_DOWNWARD : FE_UPWARD);
    }
    ~HalfRounding() {
      if (mode == FRound::Nearest) return;
      std::fesetround(prev);
      g_directed_rounding.fetch_sub(1, std::memory_order_relaxed);
    }
  };
  static bool half_is_nan(uint64_t h) { return (h & 0x7FFF) > 0x7C00 && (h & 0x7C00) == 0x7C00; }
  static bool half_nan(uint64_t h, bool bf16) { return bf16 ? (h & 0x7FFF) > 0x7F80 : half_is_nan(h); }
  // .ftz on an f16: a subnormal to signed zero.
  static uint64_t half_ftz(uint64_t h) { return (h & 0x7C00) == 0 ? (h & 0x8000) : h; }
  // A comparison's .ftz: an f32 or f16 subnormal operand as its signed zero.
  static uint64_t cmp_ftz(bool ftz, Type ty, uint64_t v) {
    if (!ftz || !ty.is_float() || ty.is_bfloat()) return v;
    if (ty.bits == 32) return (v & 0x7F800000u) == 0 ? (v & 0x80000000u) : v;
    if (ty.bits == 16) return half_ftz(v & 0xFFFF);
    return v;
  }
  // The modifiers applied to a half result: .ftz, .relu, .sat.
  static uint64_t half_post(uint64_t h, bool bf16, const HalfMods& md) {
    const bool nan = half_nan(h, bf16);
    if (md.ftz && !bf16) h = half_ftz(h);
    if (md.relu && !nan && (h & 0x8000)) h = 0;
    if (md.sat) {
      if (nan || (h & 0x8000)) h = 0;
      else if (bf16 ? h > 0x3F80 : h > 0x3C00) h = bf16 ? 0x3F80 : 0x3C00;
    }
    return h;
  }

  // bf16_to_double and double_to_bf16 are at file scope, shared with the
  // SASS executor.

  // fma of three bf16 values, rounded once to bf16. The double fma rounds
  // first, and bf16's exponent range lets the addend sit far below the
  // product: a product exactly halfway between two bf16 values plus a tiny
  // positive c must round up, and rounding the double sum (which drops c)
  // took the tie to even instead (measured: 0x9ccc * 0xfe60 + 0x007f is
  // 0x5bb3 on an RTX 3060). The product is exact in a double, so the sum's
  // rounding error is too (TwoSum), and it breaks such ties.
  static uint64_t bf16_fma(double x, double y, double z) {
    const double p = x * y;               // 8 x 8 significant bits: exact
    const double s = p + z;
    const uint64_t h = double_to_bf16(s);
    if (!std::isfinite(s)) return h;
    const double bb = s - p, err = (p - (s - bb)) + (z - bb);
    const double hv = bf16_to_double(h);
    if (err == 0 || hv == s) return h;
    // s lies strictly between hv and its neighbour on s's side; only if it is
    // their midpoint does err decide.
    const bool up = s > hv;
    const uint64_t n = (up == !(h & 0x8000)) ? h + 1 : (h & 0x7FFF) == 0 ? ((h ^ 0x8000) | 1) : h - 1;
    const double nv = bf16_to_double(n);
    if (s != hv + (nv - hv) / 2) return h;
    return (err > 0) == (nv > hv) ? n : h;
  }

  uint64_t convert(const OpCvt* op, uint64_t in) {
    const Type& s = op->src_ty;
    const Type& d = op->dst_ty;
    // Read the source as a real number (float src) or integer (int src).
    if (s.is_real()) {
      // bf16 is the top half of an f32, and cvt.f32.bf16 is that shift on an
      // RTX 3060 for every input, NaNs included: 0xffff becomes 0xffff0000
      // and a signalling 0xff81 stays signalling.
      if (s.is_bfloat() && d.bits == 32 && !d.is_bfloat() && d.is_real() && !op->sat && !op->relu &&
          !op->ftz && (op->round == Round::None || op->round == Round::Rn))
        return (in & 0xFFFFu) << 16;
      double x = s.is_bfloat()  ? bf16_to_double(in)
                 : s.bits == 16 ? f16_to_double(in)
                 : s.bits == 32 ? static_cast<double>(f32(in))
                                : f64(in);
      if (d.is_real()) {
        // An "i" rounding mode rounds to an integral value and keeps the float
        // type: cvt.rpi.f32.f32 is ceilf. Skipping it here made ceilf, floorf
        // and truncf return their argument, and roundf return x + 0.5.
        switch (op->round) {
          case Round::Rni:
          case Round::Rzi:
          case Round::Rmi:
          case Round::Rpi:
            x = round_int(x, op->round);
            break;
          default:
            break;
        }
        const auto subnormal32 = [](double v) {
          return v != 0 && std::fabs(v) < static_cast<double>(std::numeric_limits<float>::min());
        };
        if (op->ftz && s.bits == 32 && !s.is_bfloat() && subnormal32(x)) x = std::copysign(0.0, x);
        if (op->sat) x = x > 1 ? 1.0 : x > 0 ? x : 0.0;   // NaN and -0 become +0
        if (op->relu) {
          // A negative result (and -0) to +0; a NaN to the canonical NaN.
          if (std::isnan(x)) return kCanonicalNaN16;
          if (x < 0 || std::signbit(x)) x = 0.0;
        }
        if (d.is_bfloat()) {
          const uint64_t h = double_to_bf16(x);
          return op->pzo && h == 0x8000 ? 0u : h;   // .pzo: a -0.0 result is +0.0
        }
        // A NaN from f64 keeps its sign and the top of its payload in f16 on a
        // real GPU, made quiet, where one from f32 comes out 0x7FFF whatever
        // it was.
        if (d.bits == 16 && s.bits == 64 && std::isnan(x)) {
          const uint64_t b = std::bit_cast<uint64_t>(x);
          return ((b >> 48) & 0x8000u) | 0x7E00u | ((b >> 42) & 0x3FFu);
        }
        if (d.bits == 16) {
          const uint64_t h = double_to_f16(x);
          return op->pzo && h == 0x8000 ? 0u : h;   // .pzo: a -0.0 result is +0.0
        }
        // With .ftz, an f32 NaN goes through f32 arithmetic first and comes
        // out as the canonical NaN widened (measured); without, its payload
        // is kept.
        if (d.bits == 64 && s.bits == 32 && op->ftz && std::isnan(x)) return 0x7FFF'FFFF'E000'0000ull;
        // An f16 NaN widened to f64 keeps its sign and its payload, at the top
        // of the f64's, and is made quiet (an RTX 3060: 0x7d55 becomes
        // 0x7ffd540000000000, 0xffff 0xfffffc0000000000).
        if (d.bits == 64 && s.bits == 16 && !s.is_bfloat() && std::isnan(x))
          return (uint64_t{(in >> 15) & 1} << 63) | 0x7FF8'0000'0000'0000ull | (uint64_t{in & 0x3FFu} << 42);
        if (d.bits == 64) return f64bits(x);
        const float f = static_cast<float>(x);
        if (op->ftz && subnormal32(f)) return f32bits(std::copysign(0.0f, f));
        // f32 to f32 goes through the float pipeline, which writes the
        // canonical NaN; a narrowing cvt.f32.f64 keeps the payload.
        if (std::isnan(f) && (s.bits == 32 || s.bits == 16)) return 0x7FFF'FFFFu;
        return f32bits(f);
      }
      // float -> int: round then clamp to the destination range.
      double rounded = round_int(x, op->round == Round::None ? Round::Rzi : op->round);
      // Saturate against the powers of two themselves, which a double holds
      // exactly. Clamping to INT64_MAX as a double rounded it up to 2^63, and
      // converting 2^63 back was undefined: +inf and 2^63 came out as INT64_MIN
      // for s64, and as 0 for u64. The 32-bit limits are exact doubles, which is
      // why only the 64-bit conversions were wrong.
      // NaN: 0 from an f32 or f16 into 32 bits or fewer; the destination's
      // sign bit alone from an f64 into any width, or from an f32 into 64
      // (an RTX 3060: cvt.rzi.s16.f64 of NaN is 0x8000, .s32.f64 and .u32.f64
      // 0x80000000, cvt.rzi.s64.f32 and .u64.f32 0x8000000000000000).
      if (std::isnan(rounded))
        return (s.bits == 64 || d.bits == 64) ? uint64_t{1} << (d.bits - 1) : 0;
      if (d.is_signed()) {
        const double limit = std::ldexp(1.0, static_cast<int>(d.bits) - 1);
        const uint64_t max = (uint64_t{1} << (d.bits - 1)) - 1;
        if (rounded >= limit) return mask_to_bits(max, d.bits);
        if (rounded < -limit) return mask_to_bits(~max, d.bits);
        return mask_to_bits(static_cast<uint64_t>(static_cast<int64_t>(rounded)), d.bits);
      }
      if (rounded < 0) return 0;
      if (rounded >= std::ldexp(1.0, static_cast<int>(d.bits))) return mask_to_bits(~uint64_t{0}, d.bits);
      return mask_to_bits(static_cast<uint64_t>(rounded), d.bits);
    }
    // Integer source: sign/zero-extend to 64 bits first.
    uint64_t sv = mask_to_bits(in, s.bits);
    if (s.is_signed() && s.bits < 64) {
      uint64_t sign_bit = 1ull << (s.bits - 1);
      if (sv & sign_bit) sv |= ~((sign_bit << 1) - 1);
    }
    if (d.is_real()) {
      double x = s.is_signed() ? static_cast<double>(static_cast<int64_t>(sv))
                               : static_cast<double>(sv);
      if (d.is_bfloat()) return double_to_bf16(x);
      if (d.bits == 16) return double_to_f16(x);
      return d.bits == 32 ? f32bits(static_cast<float>(x)) : f64bits(x);
    }
    if (op->sat && d.bits < 64) {
      // Clamp to the destination's range, reading the source with its own
      // signedness: cvt.sat.u32.s32 of -1 is 0, cvt.sat.s32.u32 of
      // 0xFFFFFFFF is INT32_MAX.
      const bool neg = s.is_signed() && static_cast<int64_t>(sv) < 0;
      const uint64_t hi = d.is_signed() ? (uint64_t{1} << (d.bits - 1)) - 1 : (uint64_t{1} << d.bits) - 1;
      if (neg) {
        const int64_t lo = d.is_signed() ? -(int64_t{1} << (d.bits - 1)) : 0;
        if (static_cast<int64_t>(sv) < lo) sv = static_cast<uint64_t>(lo);
      } else if (sv > hi) {
        sv = hi;
      }
    } else if (op->sat && d.bits == 64 && s.is_signed() != d.is_signed()) {
      if (s.is_signed() && static_cast<int64_t>(sv) < 0) sv = 0;   // s64 -> u64
      else if (!s.is_signed() && sv >> 63) sv = uint64_t{INT64_MAX};  // u64 -> s64
    }
    return mask_to_bits(sv, d.bits);  // int -> int: truncate/extend
  }

  // Registers are 32 or 64 bits wide, but an operand type can be narrower. The
  // bits above the operand's width belong to whatever was in the register
  // before and must take no part: shr.u16 of a register holding 0xFFFFFFFF has
  // to shift 0xFFFF, and shr.s16 has to take its sign from bit 15, not bit 31.
  // Treating every non-64-bit type as 32-bit got both wrong, which is why the
  // i-quant kernels -- almost entirely 16-bit bit manipulation -- produced
  // wrong values while the 32-bit paths around them were fine.
  static uint64_t narrow_u(uint64_t v, uint32_t bits) {
    return bits >= 64 ? v : (v & ((1ull << bits) - 1));
  }
  static int64_t narrow_s(uint64_t v, uint32_t bits) {
    if (bits >= 64) return static_cast<int64_t>(v);
    const uint64_t mask = (1ull << bits) - 1;
    uint64_t m = v & mask;
    const uint64_t sign = 1ull << (bits - 1);
    if (m & sign) m |= ~mask;
    return static_cast<int64_t>(m);
  }

  // Strict mode: the checks that hunt for bugs hardware would hide, but which
  // real compiler output trips over. Integer division by zero and a store of a
  // register nothing has written are both undefined-but-harmless in code ptxas
  // and CUB emit every day, so neither can be on by default -- and both are
  // worth having when you are looking for a bug rather than running a workload.
  static bool strict() { return g_strict.load(std::memory_order_relaxed); }

  uint64_t int_bin(IntBinOp op, Type ty, uint64_t a, uint64_t b, const Instr& ins) {
    bool sig = ty.is_signed();
    auto s = [&](uint64_t v) -> int64_t { return narrow_s(v, ty.bits); };
    auto u = [&](uint64_t v) -> uint64_t { return narrow_u(v, ty.bits); };
    switch (op) {
      case IntBinOp::Add: return a + b;
      case IntBinOp::Sub: return a - b;
      case IntBinOp::Mul: return a * b;
      case IntBinOp::Min: return sig ? static_cast<uint64_t>(std::min(s(a), s(b))) : std::min(u(a), u(b));
      case IntBinOp::Max: return sig ? static_cast<uint64_t>(std::max(s(a), s(b))) : std::max(u(a), u(b));
      case IntBinOp::Div:
      case IntBinOp::Rem: {
        if (u(b) == 0) {
          // PTX leaves this undefined and real GPUs do not trap. VirtualGPU
          // used to, on the reasoning that a div-by-zero is almost always a
          // bug -- but real code falsifies that: ggml's flash-attention passes
          // zero for a stride its configuration does not use, computes a
          // remainder from it, and discards the answer. Trapping made those
          // kernels unrunnable over arithmetic that was never going to matter.
          //
          // So the default now follows the hardware, deterministically: a
          // quotient of all-ones and a remainder of the dividend, which is what
          // the usual expansion of these instructions leaves behind. The trap
          // is still available for a debugging run, since it does find real
          // bugs -- it just cannot be the default.
          if (strict())
            ctx_fail(ins, -1, Err::InvalidValue, "integer division by zero");
          // All ones for the quotient and the remainder alike, signed or not
          // (an RTX 3060; the remainder was once taken to be the dividend).
          (void)a;
          return narrow_u(~uint64_t{0}, ty.bits);
        }
        // The one signed quotient that does not fit: INT64_MIN / -1 is a
        // SIGFPE on x86, and it is data, not a malformed kernel. It wraps, as
        // the narrower widths already do, and the remainder is zero.
        if (sig && s(b) == -1)
          return op == IntBinOp::Div ? uint64_t{0} - static_cast<uint64_t>(s(a)) : uint64_t{0};
        if (op == IntBinOp::Div)
          return sig ? static_cast<uint64_t>(s(a) / s(b)) : u(a) / u(b);
        return sig ? static_cast<uint64_t>(s(a) % s(b)) : u(a) % u(b);
      }
      case IntBinOp::And: return a & b;
      case IntBinOp::Or: return a | b;
      case IntBinOp::Xor: return a ^ b;
      case IntBinOp::Shl: {
        // The amount is a .u32 whatever the type: a 16-bit shift by 0x10000
        // shifts everything out (it was once cut to 16 bits, to 0).
        uint64_t sh = static_cast<uint32_t>(b);
        return sh >= ty.bits ? 0 : a << sh;  // PTX clamps shift amounts
      }
      case IntBinOp::Shr: {
        uint64_t sh = static_cast<uint32_t>(b);
        if (sig) {
          int64_t v = s(a);
          if (sh >= ty.bits) return static_cast<uint64_t>(v < 0 ? -1 : 0);
          return static_cast<uint64_t>(v >> sh);
        }
        return sh >= ty.bits ? 0 : u(a) >> sh;
      }
    }
    return 0;
  }

  // nan_propagate is min.NaN/max.NaN: NaN in, NaN out. Plain min/max return
  // the *other* operand when one is NaN, which is fmin/fmax's rule, and the
  // two disagree on exactly the inputs a numerically fragile kernel is
  // watching for -- a clamp written as max.NaN(x, lo) to keep NaNs visible
  // would quietly launder them away under fmax.
  // f32 arithmetic writes the canonical NaN, 0x7fffffff, for every NaN
  // result whatever the operands' payloads (an RTX 3060; f64 keeps the quieted
  // payload, as the host does).
  static uint32_t canon32(float r) { return std::isnan(r) ? 0x7FFFFFFFu : static_cast<uint32_t>(f32bits(r)); }
  // min/max.f32: a NaN operand, quiet or signalling, gives the other one; two
  // give the canonical NaN; -0 is below +0 (min(+0, -0) is -0 either way
  // round). The host's fmin/fmax turn a signalling NaN into a NaN result and
  // leave the zeros' order open.
  static uint32_t ptx_min32(uint32_t a, uint32_t b) {
    const float x = f32(a), y = f32(b);
    if (std::isnan(x) && std::isnan(y)) return 0x7FFFFFFFu;
    if (std::isnan(x)) return b;
    if (std::isnan(y)) return a;
    if (x == y) return (a | b) & 0x80000000u ? (a | 0x80000000u) : a;   // zeros: -0 wins
    return x < y ? a : b;
  }
  static uint32_t ptx_max32(uint32_t a, uint32_t b) {
    const float x = f32(a), y = f32(b);
    if (std::isnan(x) && std::isnan(y)) return 0x7FFFFFFFu;
    if (std::isnan(x)) return b;
    if (std::isnan(y)) return a;
    if (x == y) return a & b & 0x80000000u ? a : (a & 0x7FFFFFFFu);   // zeros: +0 wins
    return x > y ? a : b;
  }
  // neg/abs on a float, as an RTX 3060 does them: an f32 NaN becomes the
  // canonical NaN and .ftz flushes a subnormal to (signed) zero first; an f64
  // NaN is left exactly as it is, sign included. Otherwise the sign is flipped
  // or cleared.
  static uint64_t float_sign_op(uint32_t bits, uint64_t v, bool ftz, bool negate) {
    if (bits == 32) {
      uint32_t x = static_cast<uint32_t>(v);
      if ((x & 0x7F800000u) == 0x7F800000u && (x & 0x7FFFFFu)) return 0x7FFFFFFFu;
      if (ftz && (x & 0x7F800000u) == 0) x &= 0x80000000u;
      return negate ? x ^ 0x80000000u : x & 0x7FFFFFFFu;
    }
    const uint64_t sign = 0x8000'0000'0000'0000ull;
    if (std::isnan(f64(v))) return v;
    return negate ? v ^ sign : v & ~sign;
  }
  static bool float_is_nan(Type ty, uint64_t v) {
    if (ty.bits == 64) return std::isnan(f64(v));
    if (ty.bits == 32) return std::isnan(f32(v));
    return (v & 0x7C00) == 0x7C00 && (v & 0x3FF);
  }
  // .sat: clamp to [0, 1]; a NaN and -0 to +0.
  static uint64_t float_sat(Type ty, uint64_t v) {
    if (ty.bits == 32) {
      const float x = f32(v);
      return f32bits(std::isnan(x) || x <= 0.0f ? 0.0f : std::min(x, 1.0f));
    }
    const double x = f64(v);
    return f64bits(std::isnan(x) || x <= 0.0 ? 0.0 : std::min(x, 1.0));
  }

  // div.approx.f32, as the ISA documents it: a * (1/b) with the reciprocal
  // rcp.approx gives, for |b| in [2^-126, 2^126]; for 2^126 < |b| < 2^128, 0
  // -- signed as a quotient is -- or NaN when a is infinite. An RTX 3060
  // follows both to the bit (its a * rcp.approx(b) and a mul.rn of the two
  // agree on a million random pairs). Outside the documented range -- a
  // subnormal, zero or infinite b -- it gives the quotient (0 / 2^-128 is 0,
  // not 0 * inf), and so does this. The reciprocal is rcp.approx's here, at
  // host precision (the documented divergence), so the result is the card's
  // wherever the card's reciprocal is correctly rounded. .ftz is the
  // caller's: the operands arrive flushed and the result is flushed after.
  static uint64_t div_approx32(uint32_t a, uint32_t b) {
    const float x = f32(a), y = f32(b), ay = std::fabs(y);
    if (ay > 0x1p126f && ay <= std::numeric_limits<float>::max())
      return std::isfinite(x) ? (a ^ b) & 0x80000000u : 0x7FFFFFFFu;
    if (!(ay >= 0x1p-126f) || std::isinf(y)) return canon32(x / y);
    return canon32(x * rcp_approx32(y));
  }
  // rcp.approx.f32 at host precision: the correctly rounded reciprocal.
  static float rcp_approx32(float x) { return static_cast<float>(1.0 / static_cast<double>(x)); }

  uint64_t float_bin(FloatBinOp op, Type ty, uint64_t a, uint64_t b,
                     bool nan_propagate = false) {
    if (nan_propagate && (op == FloatBinOp::Min || op == FloatBinOp::Max)) {
      const double x = ty.bits == 64 ? f64(a) : static_cast<double>(f32(a));
      const double y = ty.bits == 64 ? f64(b) : static_cast<double>(f32(b));
      if (std::isnan(x) || std::isnan(y)) {
        // The canonical NaN, as every f32 NaN result is (0x7fffffff; 0x7fff
        // for a half).
        if (ty.bits == 64) return f64bits(std::numeric_limits<double>::quiet_NaN());
        if (ty.bits == 32) return 0x7FFFFFFFu;
        return kCanonicalNaN16;
      }
    }
    return float_bin_impl(op, ty, a, b);
  }

  uint64_t float_bin_impl(FloatBinOp op, Type ty, uint64_t a, uint64_t b) {
    if (ty.bits == 16) {
      double x = f16_to_double(a), y = f16_to_double(b), r = 0;
      switch (op) {
        case FloatBinOp::Add: r = x + y; break;
        case FloatBinOp::Sub: r = x - y; break;
        case FloatBinOp::Mul: r = x * y; break;
        case FloatBinOp::Div: r = x / y; break;
        case FloatBinOp::Min: r = std::fmin(x, y); break;
        case FloatBinOp::Max: r = std::fmax(x, y); break;
      }
      return double_to_f16(r);
    }
    if (ty.bits == 32) {
      float x = f32(a), y = f32(b), r = 0;
      switch (op) {
        case FloatBinOp::Add: r = x + y; break;
        case FloatBinOp::Sub: r = x - y; break;
        case FloatBinOp::Mul: r = x * y; break;
        case FloatBinOp::Div: r = x / y; break;
        case FloatBinOp::Min: return ptx_min32(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
        case FloatBinOp::Max: return ptx_max32(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
      }
      return canon32(r);
    }
    double x = f64(a), y = f64(b), r = 0;
    switch (op) {
      case FloatBinOp::Add: r = x + y; break;
      case FloatBinOp::Sub: r = x - y; break;
      case FloatBinOp::Mul: r = x * y; break;
      case FloatBinOp::Div: r = x / y; break;
      case FloatBinOp::Min: r = std::fmin(x, y); break;
      case FloatBinOp::Max: r = std::fmax(x, y); break;
    }
    return f64bits(r);
  }

  // Float comparison including the NaN-aware forms: the ordered ops are false
  // if either operand is NaN, the "u" (unordered) ops are true.
  static bool compare_float(CmpOp cmp, double x, double y) {
    bool unordered = std::isnan(x) || std::isnan(y);
    switch (cmp) {
      case CmpOp::Eq: return !unordered && x == y;
      case CmpOp::Ne: return !unordered && x != y;
      case CmpOp::Lt: return !unordered && x < y;
      case CmpOp::Le: return !unordered && x <= y;
      case CmpOp::Gt: return !unordered && x > y;
      case CmpOp::Ge: return !unordered && x >= y;
      case CmpOp::Equ: return unordered || x == y;
      case CmpOp::Neu: return unordered || x != y;
      case CmpOp::Ltu: return unordered || x < y;
      case CmpOp::Leu: return unordered || x <= y;
      case CmpOp::Gtu: return unordered || x > y;
      case CmpOp::Geu: return unordered || x >= y;
      case CmpOp::Num: return !unordered;
      case CmpOp::Nan: return unordered;
    }
    return false;
  }

  bool compare(CmpOp cmp, Type ty, uint64_t a, uint64_t b) {
    auto do_cmp = [&](auto x, auto y) {
      switch (cmp) {
        case CmpOp::Eq: return x == y;
        case CmpOp::Ne: return x != y;
        case CmpOp::Lt: return x < y;
        case CmpOp::Le: return x <= y;
        case CmpOp::Gt: return x > y;
        case CmpOp::Ge: return x >= y;
        default: return false;  // unordered forms are float-only; handled above
      }
    };
    if (ty.is_float()) {
      if (ty.bits == 16) return compare_float(cmp, f16_to_double(a), f16_to_double(b));
      return ty.bits == 32 ? compare_float(cmp, f32(a), f32(b)) : compare_float(cmp, f64(a), f64(b));
    }
    // Same narrowing as the arithmetic: setp.gt.s16 compares 16-bit values, so
    // the bits above them must not decide the result.
    if (ty.is_signed()) return do_cmp(narrow_s(a, ty.bits), narrow_s(b, ty.bits));
    return do_cmp(narrow_u(a, ty.bits), narrow_u(b, ty.bits));
  }

  // ---- memory ops ----

  // Resolves an address operand's base. Registers alias the register file
  // directly; a named variable resolves through the symbol table into
  // `scratch`.
  const Lanes& addr_base(Warp& w, const BlockCtx& ctx, const Instr& ins, const Addr& a,
                         Lanes& scratch) {
    if (a.base_kind == Addr::Base::Symbol) {
      scratch.fill(resolve_symbol(ctx, ins, a.base));
      return scratch;
    }
    // Address registers are .b64 in 64-bit PTX, but shared/local addressing
    // legitimately uses 32-bit registers, so honor whichever file it is in.
    if (a.base_wide) {
      if (a.base_id >= w.regs64.size() || !w.written64[a.base_id])
        ctx_fail(ins, -1, Err::UninitializedRegister,
                 "address register " + a.base + " read before any write");
      return w.regs64[a.base_id];
    }
    if (a.base_id >= w.regs32.size() || !w.written32[a.base_id])
      ctx_fail(ins, -1, Err::UninitializedRegister,
               "address register " + a.base + " read before any write");
    if (a.offset == 0) return widen(w, w.regs32[a.base_id]);
    // A 32-bit address is register plus offset in 32 bits: nvcc leaves a
    // "negative" base in the register and brings it back with the offset
    // (Rodinia's needle: [%r4+68] with %r4 = temp - 64 for a thread of the
    // second row). Every caller adds the offset to what this returns, so the
    // base comes back as the wrapped sum less the offset.
    const auto& r = w.regs32[a.base_id];
    const uint64_t off = static_cast<uint64_t>(a.offset);
    for (uint32_t lane = 0; lane < W_; ++lane)
      scratch[lane] = uint64_t{static_cast<uint32_t>(r[lane] + static_cast<uint32_t>(off))} - off;
    return scratch;
  }

  // ---- cp.async ----

  // Issue: read the source now, hold the bytes, and leave the destination
  // untouched until the thread waits for the group this copy lands in.
  void exec_cp_async(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpCpAsync& op, Mask m) {
    if (!w.cp) w.cp = std::make_unique<AsyncCopies>();
    Lanes _s_dst, _s_src;
    const Lanes& dstb = addr_base(w, ctx, ins, op.dst, _s_dst);
    const Lanes& srcb = addr_base(w, ctx, ins, op.src, _s_src);
    Lanes _s_size;
    const Lanes* size_lanes = nullptr;
    if (op.have_src_size) size_lanes = &read_operand(w, ctx, ins, op.src_size, _s_size);

    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      PendingCopy pc;
      pc.bytes = op.bytes;
      // The destination is a shared-space address; the source a global one.
      pc.dst = kSharedVaBase + dstb[lane] + static_cast<uint64_t>(op.dst.offset);
      const uint64_t src = srcb[lane] + static_cast<uint64_t>(op.src.offset);
      // Bytes past src-size are zero-filled rather than read, which is how a
      // tile that runs off the end of a tensor is handled without a branch.
      uint64_t readable = op.bytes;
      if (size_lanes) readable = std::min<uint64_t>((*size_lanes)[lane], op.bytes);
      // Fault now if the destination could not take the write: the instruction
      // that named the address is far more useful to report than the wait that
      // happens to drain it.
      check_shared(ctx, ins, static_cast<int>(lane), pc.dst, op.bytes);
      for (uint64_t off = 0; off < readable;) {
        const uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(8, readable - off));
        const uint64_t v = load_routed(w, ctx, ins, lane, src + off, chunk);
        std::memcpy(pc.data.data() + off, &v, chunk);
        off += chunk;
      }
      w.cp->open[lane].push_back(pc);
    }
    count_memory(Space::Global, op.bytes, popcount_mask(m), /*is_store=*/false);
  }

  // Land one group's copies in shared memory.
  void complete_group(Warp& w, const BlockCtx& ctx, const Instr& ins, uint32_t lane,
                      const std::vector<PendingCopy>& group) {
    (void)w;
    for (const PendingCopy& pc : group) {
      check_shared(ctx, ins, static_cast<int>(lane), pc.dst, pc.bytes);
      std::memcpy(ctx.shared->data() + shared_ref(ctx, ins, static_cast<int>(lane), pc.dst).off,
                  pc.data.data(), pc.bytes);
      // The shared write lands here, not where the copy was issued, so this is
      // where it is counted.
      count_memory(Space::Shared, pc.bytes, 1, /*is_store=*/true);
    }
  }

  void exec_cp_async_group(Warp& w, const BlockCtx& ctx, const Instr& ins,
                           const OpCpAsyncGroup& op, Mask m) {
    if (op.kind == OpCpAsyncGroup::Kind::MbarrierArrive) {
      // The copies this thread issued complete, and then it arrives on the
      // barrier. Completing first is the whole ordering guarantee: a consumer
      // released by this arrival must see the filled buffer, so making the
      // copies land after the arrival would hand it the old contents -- the
      // exact bug deferring cp.async exists to expose.
      if (w.cp) {
        for (uint32_t lane = 0; lane < W_; ++lane) {
          if (!(m & (Mask{1} << lane))) continue;
          auto& open = w.cp->open[lane];
          auto& groups = w.cp->groups[lane];
          if (!open.empty()) {
            groups.push_back(std::move(open));
            open.clear();
          }
          while (!groups.empty()) {
            complete_group(w, ctx, ins, lane, groups.front());
            groups.pop_front();
          }
        }
      }
      if (!ctx.mbar)
        ctx_fail(ins, -1, Err::UnsupportedPtx, "cp.async.mbarrier.arrive outside a block context");
      Lanes _s_base;
      const Lanes& base = addr_base(w, ctx, ins, op.bar, _s_base);
      const uint32_t lead = first_set(m);
      const uint64_t addr =
          space_base(Space::Shared) + base[lead] + static_cast<uint64_t>(op.bar.offset);
      // Resolved as every other mbarrier access is: a shared address carries
      // its CTA's cluster rank, so an odd CTA's own barrier (and the even
      // CTA's, which CUTLASS's 2-SM blockwise-scaled kernels arrive on) is
      // found through it rather than by the raw address.
      Mbarrier& b = cluster_mbarrier(ctx, ins, static_cast<int>(lead), addr, "cp.async.mbarrier.arrive");
      // Without .noinc the pending count goes up by one before the
      // asynchronous arrive-on, a net zero for the phase -- the arrival only
      // waits for the copies, which have completed above. With .noinc there
      // is no increment, so each thread's arrive-on counts toward the phase
      // and the barrier's initial count must include it (9.7.15.16.18).
      if (op.noinc) {
        b.arrived += popcount_mask(m);
        if (b.arrived >= b.expected) {
          b.arrived -= b.expected;
          b.phase ^= 1u;
        }
      }
      return;
    }
    if (!w.cp) {
      // Waiting with nothing outstanding is legal and common -- a loop's first
      // iteration waits before it has issued anything.
      return;
    }
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      auto& open = w.cp->open[lane];
      auto& groups = w.cp->groups[lane];
      // wait_all is defined as commit_group followed by wait_group 0, so both
      // it and commit close whatever is open first -- into a group even when
      // nothing is open. An empty group is trivially complete but it is still
      // one of the "N most recent" wait_group N may leave pending: dropping
      // it left an older, real group pending past the wait, and a CUTLASS
      // multistage mainloop whose masked-off threads commit empty groups read
      // stale stages.
      if (op.kind != OpCpAsyncGroup::Kind::WaitGroup) {
        groups.push_back(std::move(open));
        open.clear();
      }
      if (op.kind == OpCpAsyncGroup::Kind::Commit) continue;
      const size_t keep = op.kind == OpCpAsyncGroup::Kind::WaitAll ? 0 : op.keep;
      while (groups.size() > keep) {
        complete_group(w, ctx, ins, lane, groups.front());
        groups.pop_front();
      }
    }
  }

  // A thread that ends with copies still in flight still performs them: the
  // hardware does not cancel an issued copy because the thread finished, and
  // another warp in the block may yet read what it wrote.
  void drain_async_copies(Warp& w, const BlockCtx& ctx, const Instr& ins) {
    if (!w.cp) return;
    for (uint32_t lane = 0; lane < W_; ++lane) {
      for (auto& g : w.cp->groups[lane]) complete_group(w, ctx, ins, lane, g);
      w.cp->groups[lane].clear();
      if (!w.cp->open[lane].empty()) {
        complete_group(w, ctx, ins, lane, w.cp->open[lane]);
        w.cp->open[lane].clear();
      }
    }
  }

  void exec_ld(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpLd& op, Mask m) {
    uint32_t size = op.ty.bytes();
    size_t n = op.dsts.size();
    if (op.space == Space::Param && op.addr.base_kind == Addr::Base::Reg) {
      // Address form: the register holds a parameter-window address.
      Lanes _s_base;
      const Lanes& base = addr_base(w, ctx, ins, op.addr, _s_base);
      LaneSet results(n);
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) {
          const uint64_t addr =
              base[lane] + static_cast<uint64_t>(op.addr.offset);
          for (size_t e = 0; e < n; ++e) {
            const uint64_t at = addr + e * size;
            if (at < kParamVaBase || at + size > kParamVaBase + params_.bytes.size())
              ctx_fail(ins, static_cast<int>(lane), Err::OutOfBounds,
                       "ld.param through a register reads outside the parameter buffer");
            uint64_t v = 0;
            std::memcpy(&v, params_.bytes.data() + (at - kParamVaBase), size);
            results[e][lane] = v;
          }
        }
      for (size_t e = 0; e < n; ++e) write_loaded(w, op.dsts[e], m, results[e], op.ty);
      return;
    }
    if (op.space == Space::Param) {
      auto it = params_.layout.find(op.addr.base);
      if (it == params_.layout.end())
        ctx_fail(ins, -1, Err::PtxParse, "ld.param from non-parameter '" + op.addr.base + "'");
      auto [off, psize] = it->second;
      for (size_t e = 0; e < n; ++e) {
        int64_t at = int64_t{off} + op.addr.offset + static_cast<int64_t>(e * size);
        if (at < 0 || static_cast<uint64_t>(at) + size > uint64_t{off} + psize)
          ctx_fail(ins, -1, Err::OutOfBounds,
                   "ld.param reads past parameter '" + op.addr.base + "'");
        uint64_t v = 0;
        std::memcpy(&v, params_.bytes.data() + at, size);
        Lanes r;  // written for every active lane below
        r.fill(v);
        write_loaded(w, op.dsts[e], m, r, op.ty);
      }
      return;
    }
    Lanes _s_base;
    const Lanes& base = addr_base(w, ctx, ins, op.addr, _s_base);
    uint64_t sbase = space_base(op.space);
    uint32_t va = size * static_cast<uint32_t>(n);  // vector accesses need vector alignment
    {
      // Sectors and bank conflicts need the addresses, which only exist here.
      Lanes at{};
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane))
          at[lane] = sbase + base[lane] + static_cast<uint64_t>(op.addr.offset);
      count_addresses(op.space, at, m, size, n, /*is_store=*/false);
    }
    LaneSet results(n);
    for (uint32_t lane = 0; lane < W_; ++lane)
      if (m & (Mask{1} << lane)) {
        uint64_t addr = sbase + base[lane] + static_cast<uint64_t>(op.addr.offset);
        if (addr % va != 0)
          ctx_fail(ins, static_cast<int>(lane), Err::MisalignedAccess,
                   "vector load requires " + std::to_string(va) + "-byte alignment");
        std::unique_lock<std::mutex> whole;
        if (op.ordered && n > 1) whole = std::unique_lock<std::mutex>(vector_lock(addr));
        for (size_t e = 0; e < n; ++e)
          results[e][lane] = load_routed(w, ctx, ins, lane, addr + e * size, size);
      }
    for (size_t e = 0; e < n; ++e) write_loaded(w, op.dsts[e], m, results[e], op.ty);
  }

  // A signed narrow load sign-extends into the destination register: PTX says
  // ld.s8 delivers the byte's value, not its bit pattern. Masking to the type
  // width instead turns -1 into 255, and the cvt that follows reads the
  // positive number -- which is how a quantized weight of -1 became +255 and
  // corrupted every dequantized tensor while still looking like a plain copy.
  // Parameters too: a kernel taking a signed char or short reads it with
  // ld.param.s8 or .s16, and -128 arrived as 128.
  void write_loaded(Warp& w, const Reg& dst, Mask m, const Lanes& vals, const Type& ty) {
    const uint32_t dst_bits = dst.wide ? 64u : 32u;
    if (!(ty.is_signed() && ty.bits < dst_bits)) {
      write_reg(w, dst, m, vals, ty.bits);
      return;
    }
    Lanes ext = vals;
    const uint64_t sign_bit = 1ull << (ty.bits - 1);
    const uint64_t value_mask = (sign_bit << 1) - 1;
    for (uint32_t lane = 0; lane < W_; ++lane)
      if (m & (Mask{1} << lane)) {
        uint64_t v = ext[lane] & value_mask;
        if (v & sign_bit) v |= ~value_mask;
        ext[lane] = v;
      }
    write_reg(w, dst, m, ext, dst_bits);
  }

  void exec_st(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpSt& op, Mask m) {
    uint32_t size = op.ty.bytes();
    size_t n = op.srcs.size();
    LaneSet vals(n);
    size_t vi = 0;
    for (const auto& src : op.srcs) {
      // A value nothing has written, on its way to memory, looks like a bug --
      // but CUB's radix sort stores exactly that into the unused part of a
      // shared tile, and never reads it back. So this is a strict-mode check
      // rather than a default one; see strict().
      if (const auto* r = strict() ? std::get_if<RegOperand>(&src) : nullptr) {
        const bool init = r->reg.wide ? (r->reg.id < w.regs64.size() && w.written64[r->reg.id])
                                      : (r->reg.id < w.regs32.size() && w.written32[r->reg.id]);
        if (!init)
          ctx_fail(ins, -1, Err::UninitializedRegister,
                   "store writes register " + r->reg.name + ", which nothing has written");
      }
      Lanes tmp;
      vals[vi++] = read_operand(w, ctx, ins, src, tmp);
    }
    Lanes _s_base;
    const Lanes& base = addr_base(w, ctx, ins, op.addr, _s_base);
    uint64_t sbase = space_base(op.space);
    uint32_t va = size * static_cast<uint32_t>(n);
    {
      Lanes at{};
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane))
          at[lane] = sbase + base[lane] + static_cast<uint64_t>(op.addr.offset);
      count_addresses(op.space, at, m, size, n, /*is_store=*/true);
    }
    for (uint32_t lane = 0; lane < W_; ++lane)
      if (m & (Mask{1} << lane)) {
        uint64_t addr = sbase + base[lane] + static_cast<uint64_t>(op.addr.offset);
        if (addr % va != 0)
          ctx_fail(ins, static_cast<int>(lane), Err::MisalignedAccess,
                   "vector store requires " + std::to_string(va) + "-byte alignment");
        std::unique_lock<std::mutex> whole;
        if (op.ordered && n > 1) whole = std::unique_lock<std::mutex>(vector_lock(addr));
        for (size_t e = 0; e < n; ++e)
          store_routed(w, ctx, ins, lane, addr + e * size, size, mask_to_bits(vals[e][lane], op.ty.bits));
      }
  }

  // mbarrier: a split barrier. Arriving and waiting are different
  // instructions, so nothing here blocks -- a wait answers with a predicate
  // and the kernel spins, which is what it does on hardware. The scheduler
  // preempts a spinning warp after its slice, the other warps run and arrive,
  // and the spinner then sees the phase flip. A spin that can never be
  // satisfied is caught by the step budget rather than hanging.
  static uint32_t first_set(Mask m) {
    return m ? static_cast<uint32_t>(__builtin_ctzll(m)) : 0u;
  }

  // barrier.cluster completes when every thread of the cluster that has not
  // exited has arrived (9.7.15.3). Exited lanes are simply no longer live, so
  // "not exited" is "on some path".
  static void complete_cluster_barrier_if_done(ClusterState& cs) {
    for (std::vector<Warp>* warps : cs.blocks)
      for (const Warp& w : *warps) {
        Mask live = 0;
        for (const Path& p : w.paths) live |= p.mask;
        if (live & ~w.cluster_arrived) return;
      }
    for (std::vector<Warp>* warps : cs.blocks)
      for (Warp& w : *warps) w.cluster_arrived = 0;
    ++cs.phase;
  }

  // ---- Hopper's bulk copies (TMA) ----
  //
  // A load reads global memory when issued and reaches shared memory when its
  // mbarrier is next looked at (Mbarrier::pending). A store reads shared
  // memory and writes global memory when issued, so its bulk group is
  // already complete when the kernel waits on it; a kernel that overwrites
  // the source tile before cp.async.bulk.wait_group.read is not caught.

  // Reads the 128-byte tensor map at a generic address: a kernel parameter
  // (__grid_constant__), global memory or constant memory.
  exec::TensorMap read_tensor_map(Warp& w, const BlockCtx& ctx, const Instr& ins, uint32_t lane,
                                  uint64_t addr) {
    if (addr % 64)
      ctx_fail(ins, static_cast<int>(lane), Err::MisalignedAccess,
               "a tensor map must be 64-byte aligned");
    uint64_t q[16];
    for (int i = 0; i < 16; ++i) q[i] = load_routed(w, ctx, ins, lane, addr + 8 * i, 8);
    exec::TensorMap m;
    if (!m.decode(q))
      ctx_fail(ins, static_cast<int>(lane), Err::InvalidValue,
               "the tensor map this copy names was not made by cuTensorMapEncodeTiled (or was "
               "overwritten); its first bytes are not the encoder's");
    return m;
  }

  // tensormap.replace: decode the map, change one field, encode it back. A
  // value the hardware would take but that no map cuTensorMapEncodeTiled
  // makes could have (a box of 300, a traversal stride of 0) leaves the
  // behaviour undefined, so it is refused here rather than carried into a
  // copy that would then do something arbitrary.
  void exec_tensormap_replace(Warp& w, const BlockCtx& ctx, const Instr& ins,
                              const OpTensormapReplace& op, Mask m) {
    Lanes _s_a, _s_v;
    const Lanes base = addr_base(w, ctx, ins, op.addr, _s_a);
    const Lanes& val = read_operand(w, ctx, ins, op.value, _s_v);
    const uint64_t sbase = space_base(op.space);
    const bool wide = op.field == TmapField::GlobalAddress || op.field == TmapField::GlobalStride;
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const int li = static_cast<int>(lane);
      const uint64_t addr = sbase + base[lane] + static_cast<uint64_t>(op.addr.offset);
      if (addr % 64)
        ctx_fail(ins, li, Err::MisalignedAccess, "a tensor map must be 64-byte aligned");
      uint64_t q[16];
      for (int i = 0; i < 16; ++i) q[i] = load_routed(w, ctx, ins, lane, addr + 8 * i, 8);
      exec::TensorMap t;
      if (!t.decode(q))
        ctx_fail(ins, li, Err::InvalidValue,
                 "tensormap.replace on 128 bytes that are not a tensor map made by "
                 "cuTensorMapEncodeTiled (or a copy of one)");
      if (t.im2col)
        ctx_fail(ins, li, Err::UnsupportedPtx,
                 "tensormap.replace.tile on a map made by cuTensorMapEncodeIm2col; the ISA defines "
                 "only the .tile mode");
      const uint64_t v = wide ? val[lane] : static_cast<uint32_t>(val[lane]);
      auto bad = [&](const std::string& what) {
        ctx_fail(ins, li, Err::InvalidValue, "tensormap.replace: " + what);
      };
      auto refused = [&](const std::string& what) {
        ctx_fail(ins, li, Err::UnsupportedPtx, "tensormap.replace: " + what);
      };
      using exec::TmapType;
      using exec::TmapSwizzle;
      switch (op.field) {
        case TmapField::GlobalAddress:
          if (v % 16) bad("a tensor's global address must be 16-byte aligned");
          t.address = v;
          break;
        case TmapField::Rank:
          // Zero-based: the value is the rank less one.
          if (v > 4) bad("the rank field is the rank less one, 0 to 4; got " + std::to_string(v));
          t.rank = static_cast<uint32_t>(v) + 1;
          break;
        case TmapField::BoxDim:
          if (v < 1 || v > 256) bad("a box dimension is 1 to 256 elements; got " + std::to_string(v));
          t.box[op.ord] = static_cast<uint32_t>(v);
          break;
        case TmapField::GlobalDim:
          if (v < 1) bad("a global dimension is at least 1");
          t.dim[op.ord] = v;
          break;
        case TmapField::GlobalStride: {
          // Ordinal i is the stride of dimension i + 1; dimension 0's is the
          // element size.
          if (op.ord > 3) bad("a map has four strides, ordinals 0 to 3");
          const uint64_t bytes = op.stride_in_16b ? v << 4 : v;
          if (bytes % 16 || bytes >= (1ull << 40))
            bad("a global stride is a multiple of 16 bytes below 2^40; got " + std::to_string(bytes));
          t.stride[op.ord + 1] = bytes;
          break;
        }
        case TmapField::ElementStride:
          if (v < 1 || v > 8) bad("an element stride is 1 to 8; got " + std::to_string(v));
          t.elem_stride[op.ord] = static_cast<uint32_t>(v);
          break;
        case TmapField::ElemType: {
          // The ISA's own numbering (Table 36), which is not CUtensorMapDataType's.
          static const TmapType types[] = {
              TmapType::U8,  TmapType::U16,    TmapType::U32, TmapType::S32,  TmapType::U64,
              TmapType::S64, TmapType::F16,    TmapType::F32, TmapType::F32Ftz, TmapType::F64,
              TmapType::BF16, TmapType::TF32, TmapType::TF32Ftz};
          static_assert(sizeof types / sizeof types[0] == 13);
          // 13-15 are .b4x16, .b4x16_p64 and .b6x16_p32 / .b6p2x16.
          if (v > 15) bad("element type " + std::to_string(v) + " is not in the ISA's table");
          if (v >= 13) {
            t.type = v == 13 ? TmapType::U4x16Align8 : v == 14 ? TmapType::U4x16Align16 : TmapType::U6x16Align16;
            t.stride[0] = 0;
            break;
          }
          t.type = types[v];
          t.stride[0] = exec::TensorMap::type_bytes(t.type);
          break;
        }
        case TmapField::InterleaveLayout:
          if (v > 2) bad("interleave layout " + std::to_string(v) + " is not in the ISA's table");
          t.interleave = static_cast<uint8_t>(v);
          break;
        case TmapField::SwizzleMode:
          if (v == 4) refused("the 96-byte swizzle is sm_103a's and sm_107a's");
          if (v > 4) bad("swizzle mode " + std::to_string(v) + " is not in the ISA's table");
          // None, 32B, 64B, 128B, in the same order; a 128-byte swizzle keeps
          // the atomicity the map already has.
          if (!(v == 3 && exec::TensorMap::swizzle_bytes(t.swizzle) == 128))
            t.swizzle = static_cast<TmapSwizzle>(v);
          break;
        case TmapField::SwizzleAtomicity: {
          // Table 36: 16B, 32B, 32B with the 8-byte flip, 64B. The wider atoms
          // are sub-modes of the 128-byte swizzle.
          if (v > 3) bad("swizzle atomicity " + std::to_string(v) + " is not in the ISA's table");
          if (v == 2) refused("the 128-byte swizzle's 32-byte atomicity with the 8-byte flip");
          const bool wide = exec::TensorMap::swizzle_bytes(t.swizzle) == 128;
          if (v != 0 && !wide)
            refused("a 32- or 64-byte swizzle atomicity on a map whose swizzle is not 128 bytes");
          if (wide)
            t.swizzle = v == 1 ? TmapSwizzle::B128Atom32 : v == 3 ? TmapSwizzle::B128Atom64 : TmapSwizzle::B128;
          break;
        }
        case TmapField::FillMode:
          if (v > 1) bad("fill mode " + std::to_string(v) + " is not in the ISA's table");
          t.oob_nan = static_cast<uint8_t>(v);
          break;
      }
      t.encode(q);
      for (int i = 0; i < 16; ++i) store_routed(w, ctx, ins, lane, addr + 8 * i, 8, q[i]);
    }
  }

  // The mbarrier at a shared::cluster address, which must be initialized.
  Mbarrier& cluster_mbarrier(const BlockCtx& ctx, const Instr& ins, int lane, uint64_t addr,
                             const char* what) {
    const SharedRef r = shared_at(ctx, ins, lane, addr, 8);
    if (r.off % 8)
      ctx_fail(ins, lane, Err::MisalignedAccess, "an mbarrier must be 8-byte aligned");
    auto it = r.owner->mbar->bars.find(kSharedVaBase + r.off);
    if (it == r.owner->mbar->bars.end() || !it->second.valid)
      ctx_fail(ins, lane, Err::UnsupportedPtx,
               std::string(what) + " completes on an mbarrier that has not been initialized");
    return it->second;
  }

  // st.async and red.async: the write is made when issued -- one of the
  // moments the asynchronous proxy allows -- and its bytes are completed on
  // the barrier at once, so a consumer that waits on the barrier sees it.
  void exec_st_async(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpStAsync& op, Mask m) {
    Lanes _s_a, _s_b;
    const Lanes abase = addr_base(w, ctx, ins, op.addr, _s_a);
    const Lanes bbase = addr_base(w, ctx, ins, op.mbar, _s_b);
    std::vector<Lanes> vals(op.srcs.size());
    for (size_t e = 0; e < op.srcs.size(); ++e) {
      Lanes _s;
      vals[e] = read_operand(w, ctx, ins, op.srcs[e], _s);
    }
    const uint32_t size = op.ty.bytes();
    const uint32_t n = static_cast<uint32_t>(op.srcs.size());
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const int li = static_cast<int>(lane);
      const uint64_t addr = kSharedVaBase + abase[lane] + static_cast<uint64_t>(op.addr.offset);
      const uint64_t bar = kSharedVaBase + bbase[lane] + static_cast<uint64_t>(op.mbar.offset);
      if (addr % (size * n))
        ctx_fail(ins, li, Err::MisalignedAccess,
                 std::string(op.red ? "red" : "st") + ".async requires " + std::to_string(size * n) +
                     "-byte alignment");
      const SharedRef dst = shared_at(ctx, ins, li, addr, size * n);
      const SharedRef at_bar = shared_ref(ctx, ins, li, bar);
      if (at_bar.owner != dst.owner)
        ctx_fail(ins, li, Err::InvalidValue,
                 std::string(op.red ? "red" : "st") +
                     ".async's mbarrier must be in the same block as the memory it writes");
      Mbarrier& b = cluster_mbarrier(ctx, ins, li, bar, op.red ? "red.async" : "st.async");
      uint8_t* p = dst.owner->shared->data() + dst.off;
      for (uint32_t e = 0; e < n; ++e) {
        uint64_t v = vals[e][lane];
        if (op.red) {
          uint64_t old = 0;
          std::memcpy(&old, p, size);
          v = reduce_value(op.op, op.ty, old, v);
        }
        std::memcpy(p + e * size, &v, size);
      }
      b.tx -= static_cast<int64_t>(size * n);
      complete_phase_if_done(b);
    }
    count_memory(Space::Shared, size * n, popcount_mask(m), /*is_store=*/true);
  }

  // One element of red.async or cp.reduce.async.bulk: `old` combined with
  // `b` by `op`, both of type `ty`. Floating-point add rounds to nearest even
  // and keeps subnormals (the .noftz the ISA requires or defaults to); min
  // and max return the other operand when one is NaN, as min and max do.
  static uint64_t reduce_value(AtomOp op, const Type& ty, uint64_t old, uint64_t b) {
    return exec::reduce_value(op, ty, old, b);
  }


  // cp.reduce.async.bulk into global memory: element by element, each an
  // atomic read-modify-write, as the ISA makes them (and as they must be
  // here, where other blocks run on other host threads).
  void reduce_global(const Instr& ins, int lane, uint64_t addr, const uint8_t* src, uint64_t bytes,
                     const Type& ty, AtomOp op) {
    const uint32_t es = ty.bytes();
    for (uint64_t off = 0; off < bytes; off += es) {
      uint64_t b = 0;
      std::memcpy(&b, src + off, es);
      std::unique_lock<std::mutex> guard;
      if (concurrent_) guard = std::unique_lock<std::mutex>(atomic_lock_for(addr + off));
      try {
        const uint64_t old = mem_.load_scalar(addr + off, es);
        mem_.store_scalar(addr + off, es, reduce_value(op, ty, old, b));
      } catch (const Error& e) {
        rethrow_with_context(e, ins, lane);
      }
    }
    stats_.atomics += bytes / es;
    stats_.atomic_bytes += bytes;
  }

  // The element type a tensor map gives cp.reduce.async.bulk.tensor, and
  // whether the operation has a form for it (the ISA's table in 9.7.10.28.5.4).
  static std::optional<Type> tensor_reduce_type(exec::TmapType t, AtomOp op) {
    return exec::tensor_reduce_type(t, op);
  }


  void exec_bulk_copy(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpBulkCopy& op, Mask m) {
    // Copies, not references: a 32-bit address register is widened into a
    // small ring of scratch lanes, and the coordinates read below would reuse
    // the slot an address still pointed into.
    Lanes _s_smem, _s_gmem, _s_mbar;
    const Lanes smem_base = addr_base(w, ctx, ins, op.smem, _s_smem);
    const Lanes gmem_lanes = op.tensor ? Lanes{} : addr_base(w, ctx, ins, op.gmem, _s_gmem);
    const Lanes mbar_lanes = op.to_shared ? addr_base(w, ctx, ins, op.mbar, _s_mbar) : Lanes{};
    const Lanes* gmem_base = &gmem_lanes;
    const Lanes* mbar_base = &mbar_lanes;
    Lanes size_v{}, tmap_v{}, mask_v{};
    {
      Lanes _s;
      if (!op.tensor) size_v = read_operand(w, ctx, ins, op.size, _s);
      else tmap_v = read_operand(w, ctx, ins, op.tmap, _s);
    }
    if (op.multicast) {
      Lanes _s;
      mask_v = read_operand(w, ctx, ins, op.cta_mask, _s);
    }
    std::vector<Lanes> coords(op.coords.size());
    for (size_t i = 0; i < op.coords.size(); ++i) {
      Lanes _s;
      coords[i] = read_operand(w, ctx, ins, op.coords[i], _s);
    }
    std::vector<Lanes> im2col_off(op.im2col_offsets.size());
    for (size_t i = 0; i < op.im2col_offsets.size(); ++i) {
      Lanes _s;
      im2col_off[i] = read_operand(w, ctx, ins, op.im2col_offsets[i], _s);
    }
    const uint64_t shared_size = ctx.shared ? ctx.shared->size() : 0;
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const int li = static_cast<int>(lane);
      if (!op.shared_to_shared) ++stats_.global_loads;
      // The shared side, which a load may put in another block of the
      // cluster: `smem` is the offset within whichever block that is.
      const SharedRef dst = shared_ref(ctx, ins, li, kSharedVaBase + smem_base[lane] +
                                                         static_cast<uint64_t>(op.smem.offset));
      if (!op.to_shared && dst.owner != &ctx)
        ctx_fail(ins, li, Err::InvalidValue,
                 "a bulk store's source is in another block of the cluster; it must be this "
                 "block's shared memory (.shared::cta)");
      const uint64_t smem = dst.off;
      if (smem % 16)
        ctx_fail(ins, static_cast<int>(lane), Err::MisalignedAccess,
                 "a bulk copy's shared-memory address must be 16-byte aligned");
      PendingBulk pb;
      auto add_run = [&](uint64_t off, uint32_t len) {
        if (off + len > shared_size)
          ctx_fail(ins, static_cast<int>(lane), Err::OutOfBounds,
                   "a bulk copy reaches shared offset " + std::to_string(off + len) +
                       ", past the block's " + std::to_string(shared_size) + " bytes");
        if (!pb.runs.empty() && pb.runs.back().first + pb.runs.back().second == off)
          pb.runs.back().second += len;
        else
          pb.runs.emplace_back(static_cast<uint32_t>(off), len);
      };
      if (!op.tensor) {
        uint64_t g = (*gmem_base)[lane] + static_cast<uint64_t>(op.gmem.offset);
        const uint64_t n = static_cast<uint32_t>(size_v[lane]);
        if (n % 16 || g % 16)
          ctx_fail(ins, static_cast<int>(lane), Err::MisalignedAccess,
                   "a bulk copy's size and global address must be multiples of 16");
        if (op.shared_to_shared) {
          // From this block's shared memory, read now like any bulk load's
          // source; `g` is a shared::cta address.
          add_run(smem, static_cast<uint32_t>(n));
          const SharedRef src = shared_ref(ctx, ins, li, kSharedVaBase + g);
          if (src.owner != &ctx)
            ctx_fail(ins, li, Err::InvalidValue,
                     "a shared-to-shared bulk copy's source must be this block's shared memory");
          g = src.off;
          if (g + n > shared_size)
            ctx_fail(ins, li, Err::OutOfBounds,
                     "a bulk copy reads past this block's shared memory");
          pb.data.assign(ctx.shared->data() + g, ctx.shared->data() + g + n);
          pb.tx = n;
          count_memory(Space::Shared, static_cast<uint32_t>(n), 1, /*is_store=*/false);
        } else if (op.to_shared) {
          add_run(smem, static_cast<uint32_t>(n));
          pb.data.resize(n);
          try {
            mem_.read(g, pb.data.data(), n);
          } catch (const Error& e) {
            rethrow_with_context(e, ins, static_cast<int>(lane));
          }
          pb.tx = n;
        } else {
          if (smem + n > shared_size)
            ctx_fail(ins, static_cast<int>(lane), Err::OutOfBounds,
                     "a bulk store reads past the block's shared memory");
          if (op.reduce) {
            reduce_global(ins, li, g, ctx.shared->data() + smem, n, op.red_ty, op.red_op);
          } else {
            try {
              mem_.write(g, ctx.shared->data() + smem, n);
            } catch (const Error& e) {
              rethrow_with_context(e, ins, static_cast<int>(lane));
            }
          }
        }
      } else {
        exec::TensorMap map = read_tensor_map(w, ctx, ins, lane, tmap_v[lane]);
        int64_t cs[5] = {};
        for (size_t d = 0; d < coords.size() && d < 5; ++d)
          cs[d] = static_cast<int32_t>(static_cast<uint32_t>(coords[d][lane]));
        exec::TmaBox box;
        if (auto bad = exec::tma_box(map, op.dims, cs, op.four_rows, op.im2col, op.to_shared, op.reduce, &box))
          ctx_fail(ins, li, bad->code, bad->what);
        const bool packed = box.packed;
        const uint32_t es = box.es, gs = box.gs;
        const uint32_t swz = exec::TensorMap::swizzle_bytes(map.swizzle);
        std::optional<Type> red_ty;
        if (op.reduce) {
          red_ty = tensor_reduce_type(map.type, op.red_op);
          if (!red_ty)
            ctx_fail(ins, li, Err::UnsupportedPtx,
                     "cp.reduce.async.bulk.tensor has no form of this operation for the tensor "
                     "map's element type (9.7.10.28.5.4)");
        }
        std::array<int64_t, 3> off{};
        for (uint32_t i = 0; i < op.im2col_offsets.size() && i < 3; ++i)
          off[i] = static_cast<uint16_t>(im2col_off[i][lane]);
        const uint64_t total = box.total;
        if (op.to_shared) pb.data.resize(total * gs);
        // A tile-mode load takes a box row at a time when it can: dimension 0
        // is contiguous in global memory, so the row's in-tensor part is one
        // bulk read rather than an allocation lookup per element. load_run
        // declines what it cannot answer exactly (and a NaN fill must fail at
        // its element), and those rows go element by element below.
        // A store's row likewise: its in-tensor elements gathered out of
        // (swizzled) shared memory, then one bulk write.
        const bool row_path = fast_enabled_ && !packed && !op.four_rows && !map.im2col && map.elem_stride[0] == 1 &&
                              (op.to_shared || !op.reduce);
        std::vector<uint8_t> row_buf;
        const int64_t n0 = static_cast<int64_t>(box.count[0]);
        const auto row = [&](uint64_t e, uint64_t row_addr, int64_t lo, int64_t hi) -> bool {
          if (!row_path) return false;
          if (!op.to_shared) {
            bool ok = true;
            row_buf.resize(static_cast<size_t>(hi - lo) * es);
            for (int64_t k = lo; k < hi && ok; ++k) {
              const uint64_t soff = exec::swizzle_address(smem + (e + static_cast<uint64_t>(k)) * es, swz,
                                                          exec::TensorMap::swizzle_atom(map.swizzle));
              if (soff + es > shared_size) ok = false;
              else std::memcpy(row_buf.data() + (k - lo) * es, ctx.shared->data() + soff, es);
            }
            return ok && mem_.store_run(row_addr + static_cast<uint64_t>(box.start[0] + lo) * es, row_buf.data(),
                                        row_buf.size(), es);
          }
          uint8_t* rowp = pb.data.data() + e * es;
          if ((map.oob_nan && !(lo == 0 && hi == n0)) ||
              !mem_.load_run(row_addr + static_cast<uint64_t>(box.start[0] + lo) * es, rowp + lo * es,
                             static_cast<uint64_t>(hi - lo) * es, es))
            return false;
          std::memset(rowp, 0, static_cast<size_t>(lo) * es);
          std::memset(rowp + hi * es, 0, static_cast<size_t>(n0 - hi) * es);
          for (uint64_t k = 0; k < box.count[0]; ++k)
            add_run(exec::swizzle_address(smem + (e + k) * es, swz, exec::TensorMap::swizzle_atom(map.swizzle)), es);
          return true;
        };
        exec::tma_walk(map, box, cs + 1, off, op.four_rows, [&](uint64_t e, uint64_t gaddr, bool inside) {
          // Shared position: the box packed densely, then swizzled by
          // address the same way wgmma reads it back.
          const uint64_t soff = exec::swizzle_address(smem + e * es, swz, exec::TensorMap::swizzle_atom(map.swizzle));
          if (op.to_shared && packed) {
            // Outside the tensor the group reads as zeros, like any element.
            if (inside) {
              try {
                mem_.read(gaddr, pb.data.data() + e * gs, gs);
              } catch (const Error& ex) {
                rethrow_with_context(ex, ins, static_cast<int>(lane));
              }
            }
            add_run(soff, gs);
          } else if (op.to_shared) {
            uint64_t v = 0;
            if (inside) {
              try {
                v = mem_.load_scalar(gaddr, es);
              } catch (const Error& ex) {
                rethrow_with_context(ex, ins, static_cast<int>(lane));
              }
            } else if (map.oob_nan) {
              ctx_fail(ins, static_cast<int>(lane), Err::UnsupportedPtx,
                       "the box crosses the tensor's edge and the map asks for the NaN "
                       "out-of-bounds fill, whose value is not documented; only zero fill is "
                       "implemented");
            }
            std::memcpy(pb.data.data() + e * es, &v, es);
            add_run(soff, es);
          } else if (inside && packed) {
            if (soff + es > shared_size)
              ctx_fail(ins, static_cast<int>(lane), Err::OutOfBounds,
                       "a bulk tensor store reads past the block's shared memory");
            const uint8_t* src = ctx.shared->data() + soff;
            uint8_t out[16] = {};
            if (map.type == exec::TmapType::U6x16Align16) {
              for (int i = 0; i < 16; ++i)
                for (int b = 0; b < 6; ++b)
                  if (src[i] >> b & 1) out[(6 * i + b) / 8] |= static_cast<uint8_t>(1u << ((6 * i + b) % 8));
            } else {
              std::memcpy(out, src, gs);
            }
            try {
              mem_.write(gaddr, out, gs);
            } catch (const Error& ex) {
              rethrow_with_context(ex, ins, static_cast<int>(lane));
            }
          } else if (inside) {
            // A store writes only the elements inside the tensor.
            if (soff + es > shared_size)
              ctx_fail(ins, static_cast<int>(lane), Err::OutOfBounds,
                       "a bulk tensor store reads past the block's shared memory");
            if (op.reduce) {
              reduce_global(ins, li, gaddr, ctx.shared->data() + soff, es, *red_ty, op.red_op);
            } else {
              uint64_t v = 0;
              std::memcpy(&v, ctx.shared->data() + soff, es);
              try {
                mem_.store_scalar(gaddr, es, v);
              } catch (const Error& ex) {
                rethrow_with_context(ex, ins, static_cast<int>(lane));
              }
            }
          }
        }, row);
        // The barrier counts every byte of the box, the zero-filled ones too
        // -- for the packed types, the packed bytes, not the padded slots
        // (a 128 x 128 .b6x16_p32 tile completes 12288, as CUTLASS's
        // block-scaled kernels expect).
        pb.tx = total * gs;
      }
      if (!op.to_shared) continue;
      // The barrier the load completes on, in the destination block.
      const SharedRef bar = shared_ref(ctx, ins, li, kSharedVaBase + (*mbar_base)[lane] +
                                                         static_cast<uint64_t>(op.mbar.offset));
      // .cta_group::2 lets the barrier be in the destination's peer CTA
      // instead (9.7.9.25.5.1), which CUTLASS's 2-SM kernels use to count both
      // CTAs' tiles on the leader's barrier.
      const uint32_t bar_rank = cluster_rank_of(*bar.owner);
      if (!op.multicast) {
        const bool peer_ok = op.cta_group == 2 && (bar_rank ^ 1) == cluster_rank_of(*dst.owner);
        if (bar.owner != dst.owner && !peer_ok)
          ctx_fail(ins, li, Err::InvalidValue,
                   op.cta_group == 2
                       ? "a .cta_group::2 bulk copy's mbarrier must be in the block its data goes to or that block's peer"
                       : "a bulk copy's mbarrier must be in the block its data goes to");
        if (bar.owner != dst.owner) pb.dst = dst.owner;
        if (op.reduce) {
          // A reduction into another block's shared memory is made now, one
          // of the moments the asynchronous proxy allows, and its bytes are
          // completed with it, as st.async's are. (A copy's data instead waits
          // for the barrier to be looked at, see Mbarrier::pending.)
          Mbarrier& b = cluster_mbarrier(
              ctx, ins, li, kSharedVaBase + (*mbar_base)[lane] + static_cast<uint64_t>(op.mbar.offset),
              "cp.reduce.async.bulk");
          const uint32_t es = op.red_ty.bytes();
          uint8_t* d = dst.owner->shared->data() + smem;
          for (uint64_t off = 0; off < pb.data.size(); off += es) {
            uint64_t old = 0, v = 0;
            std::memcpy(&old, d + off, es);
            std::memcpy(&v, pb.data.data() + off, es);
            v = reduce_value(op.red_op, op.red_ty, old, v);
            std::memcpy(d + off, &v, es);
          }
          b.tx -= static_cast<int64_t>(pb.tx);
          complete_phase_if_done(b);
          continue;
        }
        cluster_mbarrier(ctx, ins, li, kSharedVaBase + (*mbar_base)[lane] + static_cast<uint64_t>(op.mbar.offset),
                         "a bulk copy")
            .pending.push_back(std::move(pb));
        continue;
      }
      // Multicast: the same data, at the same offset, into every block the
      // mask names, each completing on its own barrier at the barrier's
      // offset (9.7.9.25.4.1).
      const uint64_t cta_mask = mask_v[lane] & 0xFFFF;
      const size_t nranks = ctx.cluster_state ? ctx.cluster_state->ranks.size() : 1;
      if (!cta_mask || (cta_mask >> nranks))
        ctx_fail(ins, li, Err::InvalidValue,
                 "a multicast bulk copy's ctaMask (0x" + [&] {
                   char b[8];
                   std::snprintf(b, sizeof b, "%x", static_cast<unsigned>(cta_mask));
                   return std::string(b);
                 }() + ") names no block, or a block past the cluster's " + std::to_string(nranks));
      for (uint64_t r = 0; r < nranks; ++r) {
        if (!(cta_mask >> r & 1)) continue;
        // .cta_group::2: each destination's signal goes to whichever CTA of
        // its pair has the barrier's rank parity (9.7.9.25.5.1).
        const uint64_t br = op.cta_group == 2 ? (r & ~uint64_t{1}) | (bar_rank & 1) : r;
        if (br >= nranks)
          ctx_fail(ins, li, Err::InvalidValue,
                   "a .cta_group::2 multicast bulk copy to CTA " + std::to_string(r) + ", which has no peer");
        const uint64_t at = kSharedVaBase + cluster_address(ctx, br, bar.off);
        PendingBulk to = pb;
        if (br != r) to.dst = ctx.cluster_state->ranks[r];
        cluster_mbarrier(ctx, ins, li, at, "a multicast bulk copy").pending.push_back(std::move(to));
      }
    }
  }

  // A phase completes when its arrivals are all in and no transaction bytes
  // are outstanding (9.7.15.16.8). A phase can be over-subscribed only by a
  // malformed kernel; the surplus carries into the next phase rather than
  // being dropped, which is what the hardware counter does.
  static void complete_phase_if_done(Mbarrier& b) {
    if (b.expected && b.arrived >= b.expected && b.tx == 0) {
      b.arrived -= b.expected;
      b.phase ^= 1u;
    }
  }

  // Writes the TMA loads waiting on this barrier into shared memory and
  // completes their bytes -- see Mbarrier::pending for when and why.
  static void land_bulk_copies(const BlockCtx& ctx, Mbarrier& b) {
    if (b.pending.empty()) return;
    for (const PendingBulk& p : b.pending) {
      const BlockCtx& to = p.dst ? *p.dst : ctx;
      size_t at = 0;
      for (const auto& [off, len] : p.runs) {
        std::memcpy(to.shared->data() + off, p.data.data() + at, len);
        at += len;
      }
      b.tx -= static_cast<int64_t>(p.tx);
    }
    b.pending.clear();
    complete_phase_if_done(b);
  }

  void exec_mbarrier(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpMbarrier& op,
                     Mask m) {
    if (!ctx.mbar)
      ctx_fail(ins, -1, Err::UnsupportedPtx, "mbarrier outside a block context");
    if (op.op == MbarOp::PendingCount && op.have_state) {
      // The count an arrival's state token carries (see the arrive); no
      // barrier is touched.
      Lanes _s_st;
      const Lanes& st = read_operand(w, ctx, ins, op.state, _s_st);
      Lanes r;
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) r[lane] = static_cast<uint32_t>(st[lane] >> 1);
      write_reg(w, op.dst, m, r, 32);
      return;
    }
    Lanes _s_base;
    const Lanes& base = addr_base(w, ctx, ins, op.addr, _s_base);
    const uint64_t sbase = space_base(Space::Shared);

    // All lanes of a warp address the same barrier in every use this has been
    // seen in, but that is a convention rather than a rule, so the address is
    // taken per lane and lanes are grouped by it.
    uint32_t lead = W_;
    for (uint32_t lane = 0; lane < W_; ++lane)
      if (m & (Mask{1} << lane)) { lead = lane; break; }
    if (lead >= W_) return;
    const uint64_t addr = sbase + base[lead] + static_cast<uint64_t>(op.addr.offset);
    // Lanes may name different barriers -- CUTLASS's cluster pipelines have
    // each lane arrive on another block's -- and every mbarrier operation is
    // per thread, so each barrier's lanes are handled as a group of their own.
    Mask same = 0;
    for (uint32_t lane = 0; lane < W_; ++lane)
      if (m & (Mask{1} << lane) && sbase + base[lane] + static_cast<uint64_t>(op.addr.offset) == addr)
        same |= Mask{1} << lane;
    if (same != m) {
      exec_mbarrier(w, ctx, ins, op, same);
      exec_mbarrier(w, ctx, ins, op, m & ~same);
      return;
    }
    if (addr % 8 != 0)
      ctx_fail(ins, -1, Err::MisalignedAccess, "an mbarrier must be 8-byte aligned");

    // The barrier's block: this one, or with .shared::cluster whichever
    // block the address names.
    const SharedRef at = shared_at(ctx, ins, static_cast<int>(lead), addr, 8);
    if (at.owner != &ctx && !op.cluster)
      ctx_fail(ins, static_cast<int>(lead), Err::InvalidValue,
               "an mbarrier address in another block of the cluster, used without "
               ".shared::cluster");
    Mbarrier& b = at.owner->mbar->bars[kSharedVaBase + at.off];
    const uint32_t lanes = popcount_mask(m);

    auto require_valid = [&]() {
      if (!b.valid)
        ctx_fail(ins, -1, Err::UnsupportedPtx,
                 "this mbarrier has not been initialized (or was invalidated); "
                 "mbarrier.init must run, and be visible to this thread, first");
    };

    switch (op.op) {
      case MbarOp::Init: {
        Lanes _s_c;
        const Lanes& c = read_operand(w, ctx, ins, op.count, _s_c);
        const uint64_t want = c[lead];
        if (want == 0)
          ctx_fail(ins, -1, Err::UnsupportedPtx, "mbarrier.init with an expected count of 0");
        b = Mbarrier{};
        b.expected = want;
        b.valid = true;
        return;
      }
      case MbarOp::Inval:
        b.valid = false;
        return;
      case MbarOp::Arrive:
      case MbarOp::ArriveDrop: {
        require_valid();
        // Every active lane is a thread, and each arrives once -- or `count`
        // times when the instruction carries one. Counting the warp as a
        // single arrival is the mistake that makes a barrier initialized to
        // blockDim.x never complete.
        uint64_t inc = lanes;
        if (op.have_count) {
          Lanes _s_c;
          const Lanes& c = read_operand(w, ctx, ins, op.count, _s_c);
          inc = 0;
          for (uint32_t lane = 0; lane < W_; ++lane)
            if (m & (Mask{1} << lane)) inc += c[lane];
        }
        // arrive.expect_tx: the count operand is the transaction bytes, raised
        // before the arrival, and each thread arrives once.
        if (op.expect_tx) {
          b.tx += static_cast<int64_t>(inc);
          inc = lanes;
        }
        // The token names the phase this arrival belongs to, which is the
        // phase a later test_wait asks about (bit 0), captured before any
        // flip; above it, the arrivals still pending before this instruction,
        // which is what mbarrier.pending_count reads from a state (an RTX
        // 3060: 64 on a fresh barrier of 64, 54 after ten arrivals).
        const uint64_t pending_before = b.expected > b.arrived ? b.expected - b.arrived : 0;
        const uint64_t token = (b.phase & 1u) | (pending_before << 1);
        b.arrived += inc;
        const uint32_t before = b.phase;
        complete_phase_if_done(b);
        if (op.no_complete && b.phase != before)
          ctx_fail(ins, -1, Err::UnsupportedPtx,
                   "mbarrier.arrive.noComplete completed the phase, which the ISA leaves undefined");
        if (op.op == MbarOp::ArriveDrop) {
          // arrive_drop also removes this thread from every later phase.
          b.expected = inc >= b.expected ? 0 : b.expected - inc;
          if (b.expected == 0) b.valid = false;
        }
        if (op.dst.id != kNoReg) {   // `_` discards the token
          Lanes r;
          for (uint32_t lane = 0; lane < W_; ++lane)
            if (m & (Mask{1} << lane)) r[lane] = token;
          write_reg(w, op.dst, m, r, 64);
        }
        return;
      }
      case MbarOp::ExpectTx:
      case MbarOp::CompleteTx: {
        require_valid();
        Lanes _s_c;
        const Lanes& c = read_operand(w, ctx, ins, op.count, _s_c);
        int64_t n = 0;
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) n += static_cast<int64_t>(static_cast<uint32_t>(c[lane]));
        b.tx += op.op == MbarOp::ExpectTx ? n : -n;
        complete_phase_if_done(b);
        return;
      }
      case MbarOp::TestWait:
      case MbarOp::TryWait: {
        require_valid();
        land_bulk_copies(ctx, b);
        Mask& p = pred_slot(w, op.dst);
        if (!op.have_state)
          ctx_fail(ins, -1, Err::UnsupportedPtx,
                   "mbarrier.test_wait/try_wait needs a state token or a phase parity");
        Lanes _s_st;
        const Lanes& st = read_operand(w, ctx, ins, op.state, _s_st);
        for (uint32_t lane = 0; lane < W_; ++lane) {
          if (!(m & (Mask{1} << lane))) continue;
          // Both forms ask the same question: has the phase named by the
          // operand finished? It has exactly when the barrier has moved on
          // from it, so the test is a difference, not an equality -- and
          // getting that backwards produces a wait that returns true
          // immediately and a pipeline that reads a buffer nobody filled.
          const uint32_t want = static_cast<uint32_t>(st[lane]) & 1u;
          const bool complete = (b.phase & 1u) != want;
          p = complete ? (p | (Mask{1} << lane)) : (p & ~(Mask{1} << lane));
          // Not yet: give up the rest of this turn so whoever we are waiting
          // for gets to run. The warp remains Ready and will re-test.
          if (!complete) w.yield_now = true;
        }
        return;
      }
      case MbarOp::PendingCount: {
        Lanes r;
        require_valid();
        land_bulk_copies(ctx, b);
        const uint64_t pending = b.expected > b.arrived ? b.expected - b.arrived : 0;
        for (uint32_t lane = 0; lane < W_; ++lane)
          if (m & (Mask{1} << lane)) r[lane] = pending;
        write_reg(w, op.dst, m, r, 32);
        return;
      }
    }
  }

  // st.bulk (9.7.10.14, sm_100): each thread zeroes its `size` bytes of
  // shared memory, a multiple of 8 and at most 16 MiB. Outside the shared
  // window, or with any other size, the ISA leaves it undefined; here it is
  // reported.
  void exec_st_bulk(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpStBulk& op, Mask m) {
    Lanes _s_base, _s_size;
    const Lanes& base = addr_base(w, ctx, ins, op.addr, _s_base);
    const Lanes& size = read_operand(w, ctx, ins, op.size, _s_size);
    const uint64_t sbase = op.shared ? kSharedVaBase : 0;
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const int li = static_cast<int>(lane);
      const uint64_t at = sbase + base[lane] + static_cast<uint64_t>(op.addr.offset);
      const uint64_t n = size[lane];
      if (n % 8 || n > (uint64_t{1} << 24))
        ctx_fail(ins, li, Err::InvalidValue,
                 "st.bulk of " + std::to_string(n) + " bytes; the size must be a multiple of 8 up to 16 MiB");
      if (!is_shared(at) || (n && !is_shared(at + n - 1)))
        ctx_fail(ins, li, Err::InvalidValue,
                 "st.bulk outside shared memory, which the ISA leaves undefined");
      if (at % 8)
        ctx_fail(ins, li, Err::MisalignedAccess, "st.bulk at an address not aligned to 8 bytes");
      for (uint64_t off = 0; off < n; off += 8) store_routed(w, ctx, ins, lane, at + off, 8, 0);
    }
  }

  // atom.{exch,cas}.b128 (sm_90): 16 aligned bytes read, and replaced by b
  // (exch), or by c if they equal b (cas), under the same striped lock the
  // other atomics take -- 16-byte atomicity is promised against other atomics.
  void exec_atom128(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpAtom& op, Mask m) {
    Lanes _s_base, s1, s2, s3, s4;
    const Lanes& base = addr_base(w, ctx, ins, op.addr, _s_base);
    const Lanes& blo = read_operand(w, ctx, ins, op.b, s1);
    const Lanes& bhi = read_operand(w, ctx, ins, op.b_hi, s2);
    const bool cas = op.op == AtomOp::Cas;
    Lanes clo{}, chi{};
    if (cas) {
      clo = read_operand(w, ctx, ins, op.c, s3);
      chi = read_operand(w, ctx, ins, op.c_hi, s4);
    }
    const uint64_t sbase = space_base(op.space);
    const bool lock_needed = concurrent_ && op.space != Space::Shared && op.space != Space::Local;
    Lanes lo{}, hi{};
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const uint64_t addr = sbase + base[lane] + static_cast<uint64_t>(op.addr.offset);
      if (addr % 16)
        ctx_fail(ins, static_cast<int>(lane), Err::MisalignedAccess,
                 "atom.b128 at an address not aligned to 16 bytes");
      std::unique_lock<std::mutex> guard;
      if (lock_needed) guard = std::unique_lock<std::mutex>(atomic_lock_for(addr));
      ++stats_.atomics;
      stats_.atomic_bytes += 16;
      lo[lane] = load_routed(w, ctx, ins, lane, addr, 8);
      hi[lane] = load_routed(w, ctx, ins, lane, addr + 8, 8);
      if (!cas) {
        store_routed(w, ctx, ins, lane, addr, 8, blo[lane]);
        store_routed(w, ctx, ins, lane, addr + 8, 8, bhi[lane]);
      } else if (lo[lane] == blo[lane] && hi[lane] == bhi[lane]) {
        store_routed(w, ctx, ins, lane, addr, 8, clo[lane]);
        store_routed(w, ctx, ins, lane, addr + 8, 8, chi[lane]);
      }
    }
    write_reg(w, op.dst, m, lo, 64);
    write_reg(w, op.dst_hi, m, hi, 64);
  }

  // The video instructions (PTX ISA 9.7.20), as an RTX 3060 computes them
  // (nvidia/tests/e2e/video_forms.cu, 652 variants): ptxas emulates them on
  // sm_70+, and where that departs from the ISA's pseudocode the card is what
  // programs see. Among the things measured:
  //  - a four-operand scalar form with neither a secondary operation nor a
  //    merge ignores c (it is not |a - b| + c);
  //  - SIMD selectors pick each lane's source from a's then b's half-words or
  //    bytes, the highest lane's written first (.h10 is a's own two);
  //  - the SIMD accumulate adds the masked lanes' (signed) results to c.
  static int64_t video_part(uint32_t x, bool sgn, int sel) {
    if (sel < 0) return sgn ? static_cast<int64_t>(static_cast<int32_t>(x)) : static_cast<int64_t>(x);
    if (sel < 4) {
      const uint32_t v = (x >> (8 * sel)) & 0xFF;
      return sgn ? static_cast<int64_t>(static_cast<int8_t>(v)) : static_cast<int64_t>(v);
    }
    const uint32_t v = (x >> (16 * (sel - 4))) & 0xFFFF;
    return sgn ? static_cast<int64_t>(static_cast<int16_t>(v)) : static_cast<int64_t>(v);
  }
  static bool video_cmp(CmpOp c, int64_t x, int64_t y) {
    switch (c) {
      case CmpOp::Eq: return x == y;
      case CmpOp::Ne: return x != y;
      case CmpOp::Lt: return x < y;
      case CmpOp::Le: return x <= y;
      case CmpOp::Gt: return x > y;
      default: return x >= y;
    }
  }
  static __int128 video_clamp(__int128 v, bool sgn, int bits) {
    const __int128 lo = sgn ? -(static_cast<__int128>(1) << (bits - 1)) : 0;
    const __int128 hi = sgn ? (static_cast<__int128>(1) << (bits - 1)) - 1 : (static_cast<__int128>(1) << bits) - 1;
    return v < lo ? lo : v > hi ? hi : v;
  }
  static uint32_t video_scalar(const OpVideo& op, uint32_t a, uint32_t b, uint32_t c) {
    const int64_t ta = video_part(a, op.a_signed, op.asel);
    const bool shift = op.op == VideoOp::Shl || op.op == VideoOp::Shr;
    int64_t tb = video_part(b, shift ? false : op.b_signed, op.bsel);
    if (op.op == VideoOp::Mad) {
      // tmp = ta * tb + c; the negation of the product or of c, or .po, as
      // one's complement plus a carried-in one; then the scale (arithmetic)
      // and the saturation. Unlike the ISA's pseudocode, the card takes a
      // whole 32-bit a or b, and c, as signed whatever their types
      // (0xffffffff * 0x80000000 + 0x80 is 0x80000080, saturated u32), so
      // only a byte or half-word selection follows the type.
      const bool signed_final = op.a_signed || op.b_signed || op.neg_ab || op.neg_c;
      const int64_t ma = op.asel < 0 ? static_cast<int32_t>(a) : ta;
      const int64_t mb = op.bsel < 0 ? static_cast<int32_t>(b) : tb;
      __int128 tmp = static_cast<__int128>(ma) * mb;
      __int128 cc = static_cast<int32_t>(c);
      if (op.po) tmp += 1;
      else if (op.neg_ab) tmp = -tmp;
      else if (op.neg_c) cc = -cc;
      tmp += cc;
      if (op.scale) tmp >>= op.scale;
      // An unsigned saturation takes a negative result to 0 -- but after a
      // scale, to 0xffffffff (0 * 0x7fff + 0xffffffff, .shr15, gives
      // 0xffffffff), as if the shifted value were compared unsigned.
      if (op.sat) tmp = !signed_final && op.scale && tmp < 0 ? __int128{0xFFFFFFFF} : video_clamp(tmp, signed_final, 32);
      return static_cast<uint32_t>(tmp);
    }
    __int128 tmp = 0;
    switch (op.op) {
      case VideoOp::Add: tmp = static_cast<__int128>(ta) + tb; break;
      case VideoOp::Sub: tmp = static_cast<__int128>(ta) - tb; break;
      case VideoOp::AbsDiff: tmp = ta > tb ? ta - tb : tb - ta; break;
      case VideoOp::Min: tmp = std::min(ta, tb); break;
      case VideoOp::Max: tmp = std::max(ta, tb); break;
      case VideoOp::Shl:
      case VideoOp::Shr:
        if (op.shift_wrap) tb &= 0x1F;
        else if (tb > 32) tb = 32;
        tmp = op.op == VideoOp::Shl ? static_cast<__int128>(ta) << tb : static_cast<__int128>(ta) >> tb;
        // The ISA's .s34 intermediate, wrapping at 34 bits: 0x7fffffff << 31
        // saturates to 0x80000000 (signed) or 0 (unsigned) on the card.
        tmp = static_cast<__int128>(static_cast<int64_t>(static_cast<uint64_t>(tmp) << 30) >> 30);
        break;
      case VideoOp::Set: tmp = video_cmp(op.cmp, ta, tb) ? 1 : 0; break;
      default: break;
    }
    const bool dsgn = op.op == VideoOp::Set ? false : op.d_signed;
    uint32_t t = static_cast<uint32_t>(tmp);
    if (op.sat) {
      // As the card saturates: a 32-bit signed destination clamps both ways;
      // a 32-bit unsigned one clamps below at 0 and wraps above
      // (0xffffffff + 1 gives 0) after vadd, vsub and vabsdiff; a byte or half-word one takes the unsigned
      // minimum of the 32-bit result and the field's maximum, so a negative
      // result gives 0x7f/0x7fff (signed) or 0xff/0xffff (unsigned).
      if (op.dsel < 0) {
        // vabsdiff too: |-0x55fd0d9d - 0xb3280f47| gives 0x09251ce4.
        const bool wraps = op.op == VideoOp::Add || op.op == VideoOp::Sub || op.op == VideoOp::AbsDiff;
        t = dsgn ? static_cast<uint32_t>(video_clamp(tmp, true, 32))
                 : tmp < 0 ? 0 : wraps ? t : static_cast<uint32_t>(video_clamp(tmp, false, 32));
      }
      else t = std::min<uint32_t>(t, (op.dsel < 4 ? 0xFFu : 0xFFFFu) >> (dsgn ? 1 : 0));
    }
    // The secondary operation works on the 32-bit result, as the card does.
    // .min/.max compare it with c signed for a signed destination (for vset,
    // signed operands); for an unsigned one, unsigned -- except that after
    // vadd and vsub a result with bit 31 set counts as above every c (the
    // sum is sign-extended to 34 bits and then compared unsigned).
    // After vabsdiff and the shifts it is the result's true value
    // (|0 - -2^31| is 2^31, not negative) unless it was saturated; after vmin
    // and vmax, the 32-bit result.
    const bool sec_signed = op.op == VideoOp::Set ? (op.a_signed || op.b_signed) : dsgn;
    const bool addsub = op.op == VideoOp::Add || op.op == VideoOp::Sub;
    const __int128 true_t = op.sat ? static_cast<__int128>(sec_signed ? static_cast<int64_t>(static_cast<int32_t>(t))
                                                                      : static_cast<int64_t>(t))
                                   : tmp;
    auto key = [&](uint32_t v, bool is_t) -> __int128 {
      if (!is_t) return sec_signed ? static_cast<__int128>(static_cast<int32_t>(v)) : static_cast<__int128>(v);
      if (addsub) {
        if (sec_signed) return static_cast<int32_t>(v);
        return (v >> 31) ? (static_cast<__int128>(3) << 32) | v : static_cast<__int128>(v);
      }
      if (op.op == VideoOp::Min || op.op == VideoOp::Max)
        return sec_signed ? static_cast<__int128>(static_cast<int32_t>(v)) : static_cast<__int128>(v);
      return true_t;
    };
    switch (op.sec) {
      case OpVideo::Sec::Add: t += c; break;
      case OpVideo::Sec::Min: t = key(t, true) <= key(c, false) ? t : c; break;
      case OpVideo::Sec::Max: t = key(t, true) >= key(c, false) ? t : c; break;
      default: break;
    }
    // A merge puts the result's low byte or half-word at the selected place
    // and keeps c's other bits (the ISA's optMerge) -- except .h1, where the
    // card keeps the result's own upper half, in place.
    switch (op.dsel) {
      case 0: case 1: case 2: case 3: {
        const uint32_t sh = 8u * static_cast<uint32_t>(op.dsel);
        return ((t & 0xFFu) << sh) | (c & ~(0xFFu << sh));
      }
      case 4: return (t & 0xFFFFu) | (c & 0xFFFF0000u);
      case 5: return (t & 0xFFFF0000u) | (c & 0x0000FFFFu);
      default: return t;
    }
  }
  static uint32_t video_simd(const OpVideo& op, uint32_t a, uint32_t b, uint32_t c) {
    const uint32_t n = op.lanes, width = 32 / n;
    const uint32_t lmask = (1u << width) - 1;
    auto pool = [&](uint32_t i) { return i < n ? (a >> (width * i)) & lmask : (b >> (width * (i - n))) & lmask; };
    auto ext = [&](uint32_t v, bool sgn) -> int64_t {
      return sgn && (v >> (width - 1) & 1) ? static_cast<int64_t>(v) - (int64_t{1} << width) : static_cast<int64_t>(v);
    };
    int64_t t[4] = {};
    for (uint32_t i = 0; i < n; ++i) {
      const int64_t x = ext(pool(op.asel_v[i]), op.a_signed), y = ext(pool(op.bsel_v[i]), op.b_signed);
      switch (op.op) {
        case VideoOp::Add: t[i] = x + y; break;
        case VideoOp::Sub: t[i] = x - y; break;
        case VideoOp::Avrg: t[i] = x + y >= 0 ? (x + y + 1) >> 1 : (x + y) >> 1; break;
        case VideoOp::AbsDiff: t[i] = x > y ? x - y : y - x; break;
        case VideoOp::Min: t[i] = std::min(x, y); break;
        case VideoOp::Max: t[i] = std::max(x, y); break;
        case VideoOp::Set: t[i] = video_cmp(op.cmp, x, y) ? 1 : 0; break;
        default: break;
      }
      if (op.sat) t[i] = static_cast<int64_t>(video_clamp(t[i], op.d_signed, static_cast<int>(width)));
    }
    if (op.sec == OpVideo::Sec::Add) {
      uint64_t d = c;
      for (uint32_t i = 0; i < n; ++i)
        if (op.mask >> i & 1) d += static_cast<uint64_t>(t[i]);
      return static_cast<uint32_t>(d);
    }
    uint32_t d = 0;
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t v = op.mask >> i & 1 ? static_cast<uint32_t>(t[i]) & lmask : (c >> (width * i)) & lmask;
      d |= v << (width * i);
    }
    return d;
  }
  void exec_video(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpVideo& op, Mask m) {
    Lanes s1, s2, s3;
    const Lanes& a = read_operand(w, ctx, ins, op.a, s1);
    const Lanes& b = read_operand(w, ctx, ins, op.b, s2);
    const Lanes c = op.has_c ? read_operand(w, ctx, ins, op.c, s3) : Lanes{};
    Lanes r{};
    for (uint32_t lane = 0; lane < W_; ++lane)
      if (m & (Mask{1} << lane)) {
        const uint32_t x = static_cast<uint32_t>(a[lane]), y = static_cast<uint32_t>(b[lane]);
        const uint32_t z = op.has_c ? static_cast<uint32_t>(c[lane]) : 0;
        r[lane] = op.lanes == 1 ? video_scalar(op, x, y, z) : video_simd(op, x, y, z);
      }
    write_reg(w, op.dst, m, r, 32);
  }

  // atom/red .v2/.v4/.v8 (sm_90): the vector is a run of 32-bit words, aligned to
  // its whole size, each updated by its own atomic read-modify-write (the ISA
  // makes no promise about the vector as a whole). A .f32 element is a word, a
  // packed .f16x2/.bf16x2 element is a word, and two .f16/.bf16 elements share
  // one: register 2i is its low half and 2i+1 its high.
  void exec_atom_vec(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpAtom& op, Mask m) {
    const bool wide = op.ty.bits == 32 || op.packed_half;   // one register per word
    const uint32_t nwords = wide ? op.vec : op.vec / 2u;
    const uint32_t bytes = nwords * 4u;
    const int kind = op.ty.bits == 32 ? 0 : op.ty.is_bfloat() ? 2 : 1;
    Lanes _s_base;
    const Lanes& base = addr_base(w, ctx, ins, op.addr, _s_base);
    std::vector<Lanes> bs(op.vec);
    for (uint32_t i = 0; i < op.vec; ++i) {
      Lanes tmp;
      bs[i] = read_operand(w, ctx, ins, op.srcs[i], tmp);
    }
    std::vector<Lanes> rs(op.vec);
    const uint64_t sbase = space_base(op.space);
    const bool lock_needed = concurrent_ && op.space != Space::Shared && op.space != Space::Local;
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const uint64_t addr = sbase + base[lane] + static_cast<uint64_t>(op.addr.offset);
      if (addr % bytes)
        ctx_fail(ins, static_cast<int>(lane), Err::MisalignedAccess,
                 "a vector atomic of " + std::to_string(bytes) + " bytes at an address not aligned to them");
      for (uint32_t j = 0; j < nwords; ++j) {
        const uint64_t wa = addr + 4ull * j;
        const uint32_t bw = wide ? static_cast<uint32_t>(bs[j][lane])
                                 : (static_cast<uint32_t>(bs[2 * j][lane]) & 0xffffu) |
                                       (static_cast<uint32_t>(bs[2 * j + 1][lane]) << 16);
        std::unique_lock<std::mutex> guard;
        if (lock_needed) guard = std::unique_lock<std::mutex>(atomic_lock_for(wa));
        ++stats_.atomics;
        stats_.atomic_bytes += 4;
        const auto compute = [&](const uint64_t old) -> uint64_t {
          return exec::atom_word(op.op, kind, static_cast<uint32_t>(old), bw);
        };
        uint64_t old;
        uint8_t* host = !is_shared(wa) && !is_local(wa) ? mem_.host_address(wa, 4) : nullptr;
        if (host && reinterpret_cast<uintptr_t>(host) % 4 == 0) {
          old = vgpu::exec::host_atomic_rmw(host, 4, compute);
        } else {
          old = load_routed(w, ctx, ins, lane, wa, 4);
          store_routed(w, ctx, ins, lane, wa, 4, compute(old));
        }
        if (wide) {
          rs[j][lane] = old;
        } else {
          rs[2 * j][lane] = old & 0xffffu;
          rs[2 * j + 1][lane] = (old >> 16) & 0xffffu;
        }
      }
    }
    if (!op.discards_result)
      for (uint32_t i = 0; i < op.vec; ++i) write_reg(w, op.dsts[i], m, rs[i], wide ? 32u : 16u);
  }

  void exec_atom(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpAtom& op, Mask m) {
    if (op.vec > 1) {
      exec_atom_vec(w, ctx, ins, op, m);
      return;
    }
    if (op.b128) {
      exec_atom128(w, ctx, ins, op, m);
      return;
    }
    // The access width is not the element width for the half forms: an
    // f16x2 atomic reads and writes a whole 32-bit word, and a scalar f16 one
    // touches 2 bytes. Taking the width from the element type alone is what
    // made a packed atomic look like an unsupported 2-byte access.
    uint32_t size = op.ty.bytes();
    if (op.ty.bits == 16) size = op.packed_half ? 4u : 2u;
    if (size != 2 && size != 4 && size != 8)
      ctx_fail(ins, -1, Err::UnsupportedPtx,
               "atomics are implemented for 16-, 32- and 64-bit types");
    Lanes _s_base;
    const Lanes& base = addr_base(w, ctx, ins, op.addr, _s_base);
    uint64_t sbase = space_base(op.space);
    Lanes _s_bv;
      const Lanes& bv = read_operand(w, ctx, ins, op.b, _s_bv);
    Lanes cv;
    Lanes _s_cv;
    if (op.op == AtomOp::Cas) cv = read_operand(w, ctx, ins, op.c, _s_cv);
    Lanes r;  // written for every active lane below
    // A fixed lane order makes this atomic within a warp, and within a block,
    // because a block runs on one thread. Across blocks it does not: when the
    // grid is spread over several threads, two blocks can read-modify-write the
    // same global address at once and lose an update. The stripe lock closes
    // that, and is skipped entirely when the launch is single-threaded, where
    // the fixed order was already enough.
    const bool lock_needed = concurrent_ && op.space != Space::Shared && op.space != Space::Local;
    for (uint32_t lane = 0; lane < W_; ++lane)
      if (m & (Mask{1} << lane)) {
        uint64_t addr = sbase + base[lane] + static_cast<uint64_t>(op.addr.offset);
        std::unique_lock<std::mutex> guard;
        if (lock_needed)
          guard = std::unique_lock<std::mutex>(atomic_lock_for(addr));
        ++stats_.atomics;
        stats_.atomic_bytes += size;
        // Masked to the *access* width, not the element width: for an f16x2
        // atomic those differ, and masking to 16 bits threw away the high
        // half of every operand before it was ever added.
        const uint64_t b = mask_to_bits(bv[lane], size * 8u);
        // The value the atomic stores, given the one it finds.
        auto compute = [&](const uint64_t old) -> uint64_t {
        uint64_t nv = old;
        if (op.ty.is_real() && op.ty.bits == 16) {
          // f16/bf16 atomics. The packed forms update two independent halves
          // in one operation, which is the point of them: a gradient
          // accumulation touches both channels of a half2 with one atomic
          // rather than racing on two.
          const bool is_bf = op.ty.is_bfloat();
          const int halves = op.packed_half ? 2 : 1;
          uint64_t out = op.packed_half ? 0 : (old & ~0xFFFFull);
          for (int h = 0; h < halves; ++h) {
            const uint64_t ox = (old >> (16 * h)) & 0xFFFF;
            const uint64_t bx = (b >> (16 * h)) & 0xFFFF;
            const double x = is_bf ? bf16_to_double(ox) : f16_to_double(ox);
            const double y = is_bf ? bf16_to_double(bx) : f16_to_double(bx);
            out |= (is_bf ? double_to_bf16(x + y) : double_to_f16(x + y)) << (16 * h);
          }
          nv = out;
          // Store and continue, exactly as the f32/f64 branch does. Falling
          // through instead reaches the integer switch below, which overwrote
          // nv with old + b -- so a packed half atomic added the two operands'
          // *bit patterns* and stored that. It looked like an accumulation
          // because the number grew.
          return mask_to_bits(nv, size * 8u);
        } else if (op.ty.is_float()) {
          // Reinterpret and operate in the float domain: adding the bit
          // patterns of two floats produces a number unrelated to their sum.
          // Min/max follow CUDA and use the fmin/fmax ordering rather than <,
          // so a NaN operand yields the other value.
          if (size == 4) {
            float x = std::bit_cast<float>(static_cast<uint32_t>(old));
            float y = std::bit_cast<float>(static_cast<uint32_t>(b));
            // atom/red.add.f32 flush subnormal inputs and results to signed
            // zero (the ISA's .ftz, which the card applies).
            auto ftz = [](float v) { return std::fpclassify(v) == FP_SUBNORMAL ? std::copysign(0.0f, v) : v; };
            float res = x;
            switch (op.op) {
              case AtomOp::Add: res = ftz(ftz(x) + ftz(y)); break;
              case AtomOp::Exch: res = y; break;
              case AtomOp::Min: res = std::fmin(x, y); break;
              case AtomOp::Max: res = std::fmax(x, y); break;
              default:
                ctx_fail(ins, static_cast<int>(lane), Err::UnsupportedPtx,
                         "this atomic operation has no float form");
            }
            // Exchange stores the bits it was given; an arithmetic result that
            // is NaN is the canonical one, as f32 arithmetic writes it
            // (atom.add.f32 onto a NaN stores 0x7fffffff on an RTX 3060).
            nv = op.op == AtomOp::Exch ? static_cast<uint32_t>(b) : canon32(res);
          } else {
            const double x = std::bit_cast<double>(old);
            const double y = std::bit_cast<double>(b);
            double res = x;
            switch (op.op) {
              // A NaN comes through as it is, signalling or not: the one
              // added if that is a NaN, else the one in memory (an RTX 3060).
              // Left to the host's +, which NaN survived depended on the
              // operand order the compiler chose, and a sanitizer build chose
              // differently.
              case AtomOp::Add:
                res = std::isnan(y) ? y : std::isnan(x) ? x : x + y;
                break;
              case AtomOp::Exch: res = y; break;
              case AtomOp::Min: res = std::fmin(x, y); break;
              case AtomOp::Max: res = std::fmax(x, y); break;
              default:
                ctx_fail(ins, static_cast<int>(lane), Err::UnsupportedPtx,
                         "this atomic operation has no float form");
            }
            nv = std::bit_cast<uint64_t>(res);
          }
          return mask_to_bits(nv, size * 8u);
        }
        switch (op.op) {
          case AtomOp::Add: nv = old + b; break;
          case AtomOp::And: nv = old & b; break;
          case AtomOp::Or: nv = old | b; break;
          case AtomOp::Xor: nv = old ^ b; break;
          case AtomOp::Exch: nv = b; break;
          case AtomOp::Min:
            nv = op.ty.is_signed()
                     ? (size == 8 ? static_cast<uint64_t>(std::min(static_cast<int64_t>(old),
                                                                   static_cast<int64_t>(b)))
                                  : static_cast<uint64_t>(static_cast<uint32_t>(
                                        std::min(static_cast<int32_t>(old), static_cast<int32_t>(b)))))
                     : std::min(old, b);
            break;
          case AtomOp::Max:
            nv = op.ty.is_signed()
                     ? (size == 8 ? static_cast<uint64_t>(std::max(static_cast<int64_t>(old),
                                                                   static_cast<int64_t>(b)))
                                  : static_cast<uint64_t>(static_cast<uint32_t>(
                                        std::max(static_cast<int32_t>(old), static_cast<int32_t>(b)))))
                     : std::max(old, b);
            break;
          // atom.cas d, [a], b, c: compare with b, store c. This once compared
          // with c and stored b, so a CAS that matched never stored anything
          // (atomicCAS(p, 5, 7) on 5 left 5) -- every lock and every
          // CAS-built atomic, float atomicMax included, was wrong.
          case AtomOp::Cas: nv = (old == b) ? mask_to_bits(cv[lane], size * 8u) : old; break;
          // The wrapping forms. atomicInc counts up to b and then rolls to
          // zero, which is what makes it a ring-buffer index rather than a
          // counter; atomicDec counts down and rolls to b. Implementing them
          // as +1/-1 gives a value that is right until the first wrap and
          // wrong forever after.
          case AtomOp::Inc: nv = (old >= b) ? 0ull : old + 1ull; break;
          case AtomOp::Dec: nv = (old == 0ull || old > b) ? b : old - 1ull; break;
        }
        return mask_to_bits(nv, size * 8u);
        };
        // Memory the host maps -- managed, pinned, or shared with another
        // process through CUDA IPC -- takes the CPU's own atomic, which host
        // code and other processes' kernels respect; the stripe lock is this
        // process's alone (host_atomic.hpp).
        if (!is_shared(addr) && !is_local(addr) && addr % size == 0) {
          uint8_t* host = mem_.host_address(addr, size);
          if (host && reinterpret_cast<uintptr_t>(host) % size == 0) {
            r[lane] = vgpu::exec::host_atomic_rmw(host, size, compute);
            continue;
          }
        }
        const uint64_t old = load_routed(w, ctx, ins, lane, addr, size);
        store_routed(w, ctx, ins, lane, addr, size, compute(old));
        r[lane] = old;
      }
    // `red` performed the read-modify-write and has nowhere to put the old
    // value. The memory side above already happened, which is the whole
    // instruction; only the write-back is skipped.
    // The width here is the access width too: an f16x2 atomic returns the
    // whole 32-bit word it replaced, not one half of it.
    if (!op.discards_result) write_reg(w, op.dst, m, r, size * 8u);
  }

  // ---- device printf (the vprintf builtin) ----

  std::string read_cstring(Warp& w, const BlockCtx& ctx, const Instr& ins, uint32_t lane,
                           uint64_t addr) {
    std::string out;
    for (size_t i = 0; i < 8192; ++i) {
      uint64_t b = load_routed(w, ctx, ins, lane, addr + i, 1);
      if (b == 0) return out;
      out += static_cast<char>(b);
    }
    ctx_fail(ins, static_cast<int>(lane), Err::InvalidValue,
             "printf format string exceeds 8 KiB (missing NUL terminator?)");
  }

  // Device-side assert(). nvcc lowers a failing assert to a call to
  //   __assertfail(message, file, line, function, charSize)
  // followed by a trap. Reporting the message is the whole value: an assert
  // that failed with only "trap" tells you a kernel died and nothing about
  // which invariant it died on, and the source location is right there in the
  // arguments.
  void exec_assertfail(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpCall& op, Mask m) {
    if (op.param_slots.size() < 4)
      ctx_fail(ins, -1, Err::UnsupportedPtx,
               "__assertfail expects (message, file, line, function, charSize)");
    uint32_t lane = 0;
    while (lane < W_ && !(m & (Mask{1} << lane))) ++lane;
    if (lane >= W_) return;
    auto slot = [&](size_t i) -> uint64_t {
      auto it = w.slots.find(op.param_slots[i]);
      if (it == w.slots.end())
        ctx_fail(ins, -1, Err::UninitializedRegister, "__assertfail argument slot read before write");
      return it->second.read(lane, 0, std::min<uint32_t>(it->second.size, 8));
    };
    const std::string msg = read_cstring(w, ctx, ins, lane, slot(0));
    const std::string file = read_cstring(w, ctx, ins, lane, slot(1));
    // `unsigned int line`: a 4-byte argument. Reading 8 bytes of its slot
    // reported whatever sat above it, and CuTe's asserts arrived at line
    // 236223201335.
    const uint64_t line = slot(2) & 0xFFFFFFFFu;
    const std::string fn = read_cstring(w, ctx, ins, lane, slot(3));
    ctx_fail(ins, static_cast<int>(lane), Err::DeviceAssert,
             "device assertion failed: " + msg + "\n  at " + file + ":" + std::to_string(line) +
             " in " + fn);
  }

  // An indirect call: the callee is whatever function the target register
  // points at. Addresses come from kFuncVaBase and encode the index into the
  // module's function table, so decoding one is arithmetic rather than a
  // lookup that could go stale.
  //
  // Lanes may hold different targets -- a virtual call over objects of
  // different types. The lanes that share the first lane's target make the
  // call; the others split off as a path of their own still at the call, which
  // runs next time round with the next target, and the paths merge after it
  // as any diverged paths do.
  void exec_indirect_call(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpCall& op,
                          Mask m) {
    Lanes _s_t;
    const Lanes& target = read_operand(w, ctx, ins, Operand{RegOperand{op.target_reg}}, _s_t);
    uint32_t lead = W_;
    for (uint32_t lane = 0; lane < W_; ++lane)
      if (m & (Mask{1} << lane)) { lead = lane; break; }
    if (lead >= W_) return;
    const uint64_t addr = target[lead];
    Mask same = 0;
    for (uint32_t lane = 0; lane < W_; ++lane)
      if ((m & (Mask{1} << lane)) && target[lane] == addr) same |= Mask{1} << lane;
    if (addr < kFuncVaBase || addr >= kFuncVaBase + kFuncVaSize ||
        (addr - kFuncVaBase) % kFuncVaStride != 0)
      ctx_fail(ins, static_cast<int>(lead), Err::InvalidPointer,
               "indirect call through a pointer that is not the address of a device function");
    const uint64_t index = (addr - kFuncVaBase) / kFuncVaStride;
    if (index >= fn_.module_funcs.size())
      ctx_fail(ins, static_cast<int>(lead), Err::InvalidPointer,
               "indirect call to function index " + std::to_string(index) +
                   ", which this module does not define");
    if (same != m) {
      Path rest = w.paths[step_idx_];
      rest.mask &= ~same;
      w.paths[step_idx_].mask = same;
      w.paths.push_back(rest);
    }
    OpCall resolved = op;
    resolved.target = fn_.module_funcs[static_cast<size_t>(index)].get();
    resolved.indirect = false;
    exec_user_call(w, ctx, ins, resolved, same);
  }

  // ---- calls to device functions ----
  //
  // A call pushes a frame (Warp::Frame) holding the caller's paths, registers
  // and slots, and the warp carries on in the callee's body with a single path
  // of the calling lanes. The scheduler runs it like any other code: it can
  // diverge, wait at a barrier and yield, and its last path's ret pops the
  // frame (return_to_caller). Running the callee to completion inside the
  // call instruction, as this used to, could not yield, so a __syncthreads()
  // in a function nvcc had not inlined was refused.
  //
  // Parameters and the return value travel as call slots, not as a parameter
  // buffer: each lane passes its own arguments, so there is no single set of
  // bytes to read them from. The caller has already written its slots with
  // st.param; this binds them to the names the callee's body reads.
  // Running off the end of a function is a return. nvcc emits kernels whose
  // last block ends without one, after a call whose result nothing uses
  // (Boost.Math's inverse Gaussian quantile), and a .func can end the same
  // way; an RTX 3060 returns from both, the .func with its return value.
  // Built by a plain function rather than a lambda: a lambda is implicitly
  // constexpr, so g++-12 tried to evaluate the whole Instr variant at compile
  // time for this static's initializer, and compiling this file took over
  // 8 GB (GitHub's ubuntu-22.04 runners ran out of memory and died).
  static Instr make_implicit_ret() {
    Instr i;
    i.op.emplace<OpRet>();
    return i;
  }
  static const Instr& implicit_ret() {
    static const Instr r = make_implicit_ret();
    return r;
  }

  void exec_user_call(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpCall& op, Mask m) {
    (void)ctx;
    const EntryFn& callee = *op.target;
    if (op.param_slots.size() != callee.param_slot_names.size())
      ctx_fail(ins, -1, Err::UnsupportedPtx,
               "call to '" + callee.name + "' passes " + std::to_string(op.param_slots.size()) +
                   " arguments; it takes " + std::to_string(callee.param_slot_names.size()));
    if (w.frames.size() >= kMaxCallDepth)
      ctx_fail(ins, -1, Err::ExecLimit,
               "device call nested more than " + std::to_string(kMaxCallDepth) +
                   " deep in '" + callee.name + "' (runaway recursion?)");

    // Bind arguments before anything is swapped out: they live in the caller's
    // slot map and are read per lane.
    std::unordered_map<std::string, Warp::Slot> args;
    for (size_t i = 0; i < op.param_slots.size(); ++i) {
      auto it = w.slots.find(op.param_slots[i]);
      if (it == w.slots.end())
        ctx_fail(ins, -1, Err::UninitializedRegister,
                 "argument slot '" + op.param_slots[i] + "' read before write");
      // Copied whole, bytes and all: a struct argument is as much a slot as a
      // scalar one, and only the name changes across the call boundary.
      args[callee.param_slot_names[i]] = it->second;
    }

    // The caller resumes after the call, so its path is saved already past it.
    ++w.paths[step_idx_].pc;
    Warp::Frame f;
    f.fn = w.fn;
    f.local_base = w.local_base;
    f.paths = std::move(w.paths);
    f.regs32 = std::move(w.regs32);
    f.regs64 = std::move(w.regs64);
    f.preds = std::move(w.preds);
    f.written32 = std::move(w.written32);
    f.written64 = std::move(w.written64);
    f.slots = std::move(w.slots);
    f.retval_from = callee.retval_slot_name;
    f.retval_to = op.retval_slot;
    f.lanes = m;
    f.sp = w.sp;
    f.sp_valid = w.sp_valid;
    w.frames.push_back(std::move(f));

    // The callee's .local frame sits above its caller's, the way a stack
    // grows, so a pointer into the caller's frame stays valid in the callee.
    w.local_base = (local_base_ + cur_->local_frame_size + 15) / 16 * 16;
    w.fn = &callee;
    cur_ = &callee;
    local_base_ = w.local_base;
    w.paths.clear();
    Path entry{0, m};
    entry.issued_at = w.steps;
    w.paths.push_back(entry);
    w.regs32.clear();
    w.regs32.resize(callee.num_regs32);
    w.regs64.clear();
    w.regs64.resize(callee.num_regs64);
    w.preds.assign(callee.num_regs32 + callee.num_regs64, 0);
    w.written32.assign(callee.num_regs32, 0);
    w.written64.assign(callee.num_regs64, 0);
    w.slots = std::move(args);
    w.entered_call = true;
  }

  // The callee's last path has returned: the caller's world comes back, with
  // the return value, if the callee set one, in the caller's slot.
  void return_to_caller(Warp& w) {
    Warp::Frame f = std::move(w.frames.back());
    w.frames.pop_back();
    Warp::Slot retval;
    bool have_ret = false;
    if (!f.retval_from.empty()) {
      auto it = w.slots.find(f.retval_from);
      if (it != w.slots.end()) {
        retval = std::move(it->second);
        have_ret = true;
      }
    }
    w.fn = f.fn;
    w.local_base = f.local_base;
    cur_ = w.fn ? w.fn : &fn_;
    local_base_ = w.local_base;
    w.paths = std::move(f.paths);
    w.regs32 = std::move(f.regs32);
    w.regs64 = std::move(f.regs64);
    w.preds = std::move(f.preds);
    w.written32 = std::move(f.written32);
    w.written64 = std::move(f.written64);
    w.slots = std::move(f.slots);
    w.sp = f.sp;
    w.sp_valid = f.sp_valid;
    // Only the calling lanes' return values: another path's lanes may have a
    // result in the same-named slot that they have not loaded yet.
    if (have_ret && !f.retval_to.empty()) {
      Warp::Slot& out = w.slots[f.retval_to];
      out.fit(retval.size, W_);
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (f.lanes & (Mask{1} << lane))
          std::memcpy(out.bytes.data() + static_cast<size_t>(lane) * out.size,
                      retval.bytes.data() + static_cast<size_t>(lane) * retval.size, retval.size);
    }
  }

  // Lanes of the warp that are live but outside the function it is running:
  // on the caller's other paths, frozen in a frame until the call returns. A
  // barrier inside the call cannot wait for them -- they will not move until
  // it completes -- so a barrier that needs them is refused, not deadlocked.
  Mask lanes_outside_the_call(const Warp& w) const {
    Mask in_call = 0, all = 0;
    for (const Path& p : w.paths) in_call |= p.mask;
    for (const auto& f : w.frames)
      for (const Path& p : f.paths) all |= p.mask;
    return all & ~in_call;
  }

  void refuse_barrier_outside_the_call(const Warp& w, const Instr& ins, const char* what) {
    if (const Mask out = lanes_outside_the_call(w))
      ctx_fail(ins, -1, Err::UnsupportedPtx,
               std::string(what) + " inside device function '" + cur_->name +
                   "' needs lanes that did not make the call (0x" + [&] {
                     char buf[32];
                     std::snprintf(buf, sizeof buf, "%llx", static_cast<unsigned long long>(out));
                     return std::string(buf);
                   }() + "); they wait in the caller until it returns, so the barrier can never "
                         "complete for this warp");
  }

  static constexpr uint32_t kMaxCallDepth = 256;
  // The path step() is executing, for a call that has to save it.
  size_t step_idx_ = 0;
  // The step count at which the running warp's turn began (see exec_bra).
  uint64_t turn_start_ = 0;
  static constexpr uint64_t kLongTurn = uint64_t{1} << 16;

  // ---- the device heap: malloc() and free() called from a kernel ----
  //
  // Backed by the same allocator host-side cudaMalloc uses, so a device
  // allocation gets the same out-of-bounds and use-after-free checking every
  // other device pointer gets -- which is worth more here than a bump
  // allocator would be.
  //
  // The heap is the size cudaLimitMallocHeapSize gives the device -- CUDA's
  // default 8 MiB unless the program set it -- so a runaway allocation fails the
  // way it does on hardware rather than exhausting the host, and a program that
  // raises the limit gets the room it asked for. Each device has its own heap,
  // as each has its own limit. One documented divergence: memory allocated here
  // is reachable from the host,
  // where on a device it is not -- a permissive difference, so a program that
  // works on hardware works here, but one that copies a device-malloc'd
  // pointer to the host will pass here and fail there.
  const Warp::Slot& call_slot(const Warp& w, const Instr& ins, const OpCall& op, size_t i) {
    auto it = w.slots.find(op.param_slots[i]);
    if (it == w.slots.end())
      ctx_fail(ins, -1, Err::UninitializedRegister, op.callee + " argument read before it was written");
    return it->second;
  }
  void write_call_result(Warp& w, const OpCall& op, Mask m, const Lanes& r, uint32_t bytes) {
    if (op.retval_slot.empty()) return;
    Warp::Slot& out = w.slots[op.retval_slot];
    if (out.bytes.empty()) out.reset(bytes, W_);
    for (uint32_t lane = 0; lane < W_; ++lane)
      if (m & (Mask{1} << lane)) out.write(lane, 0, bytes, r[lane]);
  }

  // A device-runtime call's error, kept as the thread's last one when it is
  // an error at all (a success does not clear it, as on the host).
  static void record_device_errors(Warp& w, Mask m, const Lanes& r) {
    for (uint32_t lane = 0; lane < kMaxWarpSize; ++lane)
      if ((m & (Mask{1} << lane)) && r[lane]) w.device_error[lane] = r[lane];
  }

  // The device runtime's per-thread error, through the two driver entry points
  // its library is built on: __cuda_syscall_cnpv2GetLastError reads it
  // without clearing it (cudaPeekAtLastError is just that call), and
  // __cuda_syscall_cnpv2SetLastError writes it -- the library's own
  // cudaGetLastError reads and then sets 0.
  void exec_device_get_last_error(Warp& w, const OpCall& op, Mask m) {
    Lanes r{};
    for (uint32_t lane = 0; lane < W_; ++lane)
      if (m & (Mask{1} << lane)) r[lane] = w.device_error[lane];
    write_call_result(w, op, m, r, 4);
  }
  void exec_device_set_last_error(Warp& w, const BlockCtx&, const Instr& ins, const OpCall& op, Mask m) {
    if (op.param_slots.size() != 1)
      ctx_fail(ins, -1, Err::UnsupportedPtx, op.callee + " takes one argument");
    const Warp::Slot& err = call_slot(w, ins, op, 0);
    for (uint32_t lane = 0; lane < W_; ++lane)
      if (m & (Mask{1} << lane)) w.device_error[lane] = static_cast<uint32_t>(err.read(lane, 0, 4));
    write_call_result(w, op, m, Lanes{}, 4);
  }

  // cudaGraphSetConditional: sets a conditional handle's value, which the
  // graph's conditional node reads once the kernel is done. A kernel launched
  // outside a graph has no handles to set, and on an RTX 3060 the call faults
  // with an illegal address; so does a handle no graph made. Racing calls are
  // undefined, as CUDA documents; here the last lane to run wins.
  void exec_graph_set_conditional(Warp& w, const Instr& ins, const OpCall& op, Mask m) {
    if (op.param_slots.size() != 2)
      ctx_fail(ins, -1, Err::UnsupportedPtx, op.callee + " takes two arguments");
    const Warp::Slot& handle = call_slot(w, ins, op, 0);
    const Warp::Slot& value = call_slot(w, ins, op, 1);
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const uint64_t h = handle.read(lane, 0, 8);
      GraphConditionals* table = cfg_.conditionals;
      if (table) {
        std::lock_guard<std::mutex> guard(table->mu);
        const auto it = table->values.find(h);
        if (it != table->values.end()) {
          it->second = static_cast<uint32_t>(value.read(lane, 0, 4));
          continue;
        }
      }
      ctx_fail(ins, static_cast<int>(lane), Err::InvalidPointer,
               table ? "cudaGraphSetConditional with a handle no graph created (" + std::to_string(h) + ")"
                     : std::string("cudaGraphSetConditional in a kernel that is not running in a graph"));
    }
    write_call_result(w, op, m, Lanes{}, 4);
  }

  // cudaGraphLaunch(graphExec, stream) from a kernel. The runtime that launched
  // the graph this kernel runs in checks and queues it (DeviceGraphLauncher);
  // each calling thread is its own launch, in lane order. Outside any graph,
  // and for a handle that is no device graph, an RTX 3060 faults with an
  // illegal address rather than returning an error.
  void exec_device_graph_launch(Warp& w, const Instr& ins, const OpCall& op, Mask m) {
    if (op.param_slots.size() != 2)
      ctx_fail(ins, -1, Err::UnsupportedPtx, op.callee + " takes two arguments");
    const Warp::Slot& exec = call_slot(w, ins, op, 0);
    const Warp::Slot& stream = call_slot(w, ins, op, 1);
    Lanes r{};
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const uint64_t g = exec.read(lane, 0, 8);
      if (!cfg_.graph_launcher)
        ctx_fail(ins, static_cast<int>(lane), Err::InvalidPointer,
                 "device-side cudaGraphLaunch in a kernel that is not running in a graph");
      const int rc = cfg_.graph_launcher->launch(g, stream.read(lane, 0, 8));
      if (rc < 0)
        ctx_fail(ins, static_cast<int>(lane), Err::InvalidPointer,
                 "device-side cudaGraphLaunch of a handle that is not a graph instantiated for device "
                 "launch (cudaGraphInstantiateFlagDeviceLaunch)");
      r[lane] = static_cast<uint64_t>(rc);
    }
    write_call_result(w, op, m, r, 4);
  }

  // int f(int* out): writes `value` through the pointer and returns success.
  void exec_device_int_query(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpCall& op, Mask m,
                             int value) {
    if (op.param_slots.size() != 1)
      ctx_fail(ins, -1, Err::UnsupportedPtx, op.callee + " takes one argument");
    const Warp::Slot& ptr = call_slot(w, ins, op, 0);
    Lanes r{};   // cudaSuccess
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const uint64_t addr = ptr.read(lane, 0, 8);
      if (addr == 0) {
        r[lane] = 1;   // cudaErrorInvalidValue
        continue;
      }
      // Usually the address of a local variable (int d; cudaGetDevice(&d)),
      // so it goes wherever a generic store would.
      store_routed(w, ctx, ins, lane, addr, 4, static_cast<uint32_t>(value));
    }
    record_device_errors(w, m, r);
    write_call_result(w, op, m, r, 4);
  }

  // cudaStreamCreateWithFlags / cudaEventCreateWithFlags on the device: a
  // non-null handle, distinct per call, written where the first argument
  // points; the flags are accepted as they come.
  void exec_device_handle(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpCall& op, Mask m) {
    if (op.param_slots.size() != 2)
      ctx_fail(ins, -1, Err::UnsupportedPtx, op.callee + " takes two arguments");
    const Warp::Slot& ptr = call_slot(w, ins, op, 0);
    Lanes r{};
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const uint64_t addr = ptr.read(lane, 0, 8);
      if (addr == 0) {
        r[lane] = 1;   // cudaErrorInvalidValue
        continue;
      }
      const uint64_t handle = kDeviceHandleBase + device_handles_.fetch_add(1, std::memory_order_relaxed);
      store_routed(w, ctx, ins, lane, addr, 8, handle);
    }
    record_device_errors(w, m, r);
    write_call_result(w, op, m, r, 4);
  }
  static constexpr uint64_t kDeviceHandleBase = 0x5654'4750'0000'0000ull;   // "VTGP": never a device address
  static inline std::atomic<uint64_t> device_handles_{1};

  // ---- the device runtime (vgpu/exec/devrt.hpp) ----
  //
  // Every entry point of cuda_device_runtime_api.h a kernel can call, with the
  // arguments in call slots (a pointer or size_t in eight bytes, a dim3 in
  // twelve, a struct passed by value in its own size) and a result in the
  // return slot. What each does, and what it returns for a bad argument, was
  // measured on an RTX 3060 (CUDA 13.0 and 12.0): vgpu/exec/devrt.hpp and
  // nvidia/docs/sass.md have the list. The SASS executor runs the same calls
  // (Runner::device_runtime_call in src/sass/exec_ops.inc); the two must
  // agree.

  int64_t devrt_pending_limit() {
    devrt::Tree& t = *dl_->tree;
    if (t.limit < 0) {
      uint64_t v = static_cast<uint64_t>(devrt::kDefaultPendingLaunches);
      if (cfg_.devrt && cfg_.devrt->limit(cfg_.device_ordinal, devrt::kLimitDevRuntimePendingLaunchCount, &v) != 0)
        v = static_cast<uint64_t>(devrt::kDefaultPendingLaunches);
      t.limit = static_cast<int64_t>(std::min<uint64_t>(v, uint64_t{1} << 40));
    }
    return t.limit;
  }

  // A parameter buffer `c` describes, handed out and remembered until it is
  // launched.
  uint64_t devrt_new_buffer(ChildLaunch c) {
    c.buffer = mem_.alloc(std::max<uint64_t>(c.size, 16));
    std::lock_guard<std::mutex> guard(dl_->mu);
    dl_->mem = &mem_;
    dl_->by_buffer.emplace(c.buffer, c);
    return c.buffer;
  }

  // Takes a handed-out buffer back, as a launch consumes it: its description,
  // and the parameter bytes in it now. False for an address that is none.
  bool devrt_take_buffer(uint64_t buffer, ChildLaunch* out) {
    {
      std::lock_guard<std::mutex> guard(dl_->mu);
      const auto it = dl_->by_buffer.find(buffer);
      if (it == dl_->by_buffer.end()) return false;
      *out = std::move(it->second);
      dl_->by_buffer.erase(it);
    }
    if (out->size) {
      out->params.resize(out->size);
      mem_.read(out->buffer, out->params.data(), out->size);
    }
    mem_.free(out->buffer);
    out->buffer = 0;
    return true;
  }

  // Queues a launch or a copy of the block `ctx` is running; returns the
  // cudaError_t the call gives.
  int devrt_issue(ChildLaunch c, uint64_t stream, const BlockCtx& ctx, bool memory_op) {
    const devrt::StreamKind kind = devrt::stream_kind(stream);
    if (kind == devrt::StreamKind::Invalid) return memory_op ? devrt::kUnknown : devrt::kInvalidValue;
    if (!memory_op) {
      const EntryFn& fn = *c.kernel.fn;
      const uint64_t bound = uint64_t{fn.max_ntid[0]} * std::max(1u, fn.max_ntid[1]) * std::max(1u, fn.max_ntid[2]);
      if (!devrt::config_ok(c.grid, c.block, uint64_t{fn.static_shared_size} + c.shared, profile_,
                            fn.max_ntid[0] ? bound : 0, fn.req_ntid))
        return devrt::kInvalidConfiguration;
    }
    c.tail = kind == devrt::StreamKind::Tail;
    const uint64_t block_linear =
        (uint64_t{ctx.ctaid[2]} * ctx.nctaid[1] + ctx.ctaid[1]) * ctx.nctaid[0] + ctx.ctaid[0];
    ChildRunner runner{cfg_, mem_, profile_, progress_, &dl_->extra};
    int err = devrt::kSuccess;
    devrt::enqueue(*dl_, mem_, runner, std::move(c), block_linear, devrt_pending_limit(), &err);
    return err;
  }

  // cudaDeviceSynchronize (CDP1): the launches of this block, run to
  // completion.
  void devrt_sync(const BlockCtx& ctx) {
    const uint64_t block_linear =
        (uint64_t{ctx.ctaid[2]} * ctx.nctaid[1] + ctx.ctaid[1]) * ctx.nctaid[0] + ctx.ctaid[0];
    ChildRunner runner{cfg_, mem_, profile_, progress_, &dl_->extra};
    devrt::drain(*dl_, mem_, runner, block_linear);
  }

  void exec_devrt(devrt::Fn f, Warp& w, const BlockCtx& ctx, const Instr& ins, const OpCall& op, Mask m) {
    using devrt::Fn;
    const auto need = [&](size_t n) {
      if (op.param_slots.size() != n)
        ctx_fail(ins, -1, Err::UnsupportedPtx, op.callee + " takes " + std::to_string(n) + " arguments");
    };
    const bool launches = f == Fn::GetParameterBufferV2 || f == Fn::GetParameterBuffer || f == Fn::LaunchDeviceV2 ||
                          f == Fn::LaunchDevice || f == Fn::MemcpyAsync || f == Fn::Memcpy2DAsync ||
                          f == Fn::Memcpy3DAsync || f == Fn::MemsetAsync || f == Fn::Memset2DAsync ||
                          f == Fn::Memset3DAsync || f == Fn::DeviceSynchronize;
    if (launches && !dl_)
      ctx_fail(ins, -1, Err::Unsupported, "a device-side launch outside a launch that can run it");
    const auto slot = [&](size_t i) -> const Warp::Slot& { return call_slot(w, ins, op, i); };
    const auto fail = [&](uint32_t lane, Err e, const std::string& why) {
      ctx_fail(ins, static_cast<int>(lane), e, why);
    };
    const auto put = [&](uint32_t lane, uint64_t addr, uint32_t n, uint64_t v) {
      store_routed(w, ctx, ins, lane, addr, n, v);
    };
    const auto get = [&](uint32_t lane, uint64_t addr, uint32_t n) { return load_routed(w, ctx, ins, lane, addr, n); };
    const auto kernel_at = [&](uint32_t lane, uint64_t addr) -> const KernelRef& {
      if (!cfg_.kernels)
        fail(lane, Err::Unsupported,
             "a device-side call naming a kernel, with no kernel table to find it in (the runtime supplies one; a "
             "bare exec::launch does not)");
      const auto k = cfg_.kernels->find(addr);
      if (k == cfg_.kernels->end()) {
        char b[24];
        std::snprintf(b, sizeof b, "%llx", static_cast<unsigned long long>(addr));
        fail(lane, Err::InvalidValue,
             std::string("a device-side call names 0x") + b +
                 ", which is not the address of a kernel loaded on this device");
      }
      return k->second;
    };
    // A device pointer for a copy or a fill: shared and local addresses are
    // this thread's and this block's, not memory the device runtime moves.
    const auto device_addr = [&](uint32_t lane, uint64_t addr) {
      if (is_shared(addr) || is_local(addr))
        fail(lane, Err::Unsupported,
             "a device-side copy or fill of shared or local memory; the device runtime moves global memory");
      return addr;
    };
    const auto copy_kind_ok = [](uint32_t kind) { return kind == 3 || kind == 4; };   // device to device, default
    Lanes r{};
    uint32_t rbytes = 4;
    bool is_error = true;   // the result is a cudaError_t, kept as the thread's last when it is one
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      switch (f) {
        case Fn::GetParameterBufferV2: {
          need(4);
          rbytes = 8;
          is_error = false;
          ChildLaunch c;
          c.kernel = kernel_at(lane, slot(0).read(lane, 0, 8));
          c.has_kernel = true;
          for (int i = 0; i < 3; ++i) {
            c.grid[i] = static_cast<uint32_t>(slot(1).read(lane, 4 * i, 4));
            c.block[i] = static_cast<uint32_t>(slot(2).read(lane, 4 * i, 4));
          }
          c.shared = static_cast<uint32_t>(slot(3).read(lane, 0, 4));
          c.size = param_space_bytes(*c.kernel.fn);
          r[lane] = devrt_new_buffer(std::move(c));
          break;
        }
        case Fn::GetParameterBuffer: {   // (alignment, size): any size, as the card hands out
          need(2);
          rbytes = 8;
          is_error = false;
          const uint64_t size = slot(1).read(lane, 0, 8);
          if (size > (uint64_t{1} << 26)) break;   // null
          ChildLaunch c;
          c.size = static_cast<uint32_t>(size);
          r[lane] = devrt_new_buffer(std::move(c));
          break;
        }
        case Fn::LaunchDeviceV2: {   // (parameterBuffer, stream)
          need(2);
          ChildLaunch c;
          if (!devrt_take_buffer(slot(0).read(lane, 0, 8), &c) || !c.has_kernel) {
            r[lane] = devrt::kInvalidValue;   // not a buffer cudaGetParameterBufferV2 handed out
            break;
          }
          r[lane] = static_cast<uint64_t>(devrt_issue(std::move(c), slot(1).read(lane, 0, 8), ctx, false));
          break;
        }
        case Fn::LaunchDevice: {   // (func, parameterBuffer, gridDim, blockDim, sharedMem, stream)
          need(6);
          ChildLaunch c;
          c.kernel = kernel_at(lane, slot(0).read(lane, 0, 8));
          c.has_kernel = true;
          for (int i = 0; i < 3; ++i) {
            c.grid[i] = static_cast<uint32_t>(slot(2).read(lane, 4 * i, 4));
            c.block[i] = static_cast<uint32_t>(slot(3).read(lane, 4 * i, 4));
          }
          c.shared = static_cast<uint32_t>(slot(4).read(lane, 0, 4));
          const uint32_t want = param_space_bytes(*c.kernel.fn);
          if (const uint64_t b = slot(1).read(lane, 0, 8)) {   // a null buffer is allowed: no parameters
            ChildLaunch held;
            if (!devrt_take_buffer(b, &held)) {
              r[lane] = devrt::kInvalidValue;
              break;
            }
            c.params = std::move(held.params);
          }
          c.params.resize(want, 0);
          r[lane] = static_cast<uint64_t>(devrt_issue(std::move(c), slot(5).read(lane, 0, 8), ctx, false));
          break;
        }
        case Fn::GetLastError:
        case Fn::PeekAtLastError:
          r[lane] = w.device_error[lane];
          if (f == Fn::GetLastError) w.device_error[lane] = 0;
          is_error = false;
          break;
        case Fn::GetDevice:
        case Fn::GetDeviceCount: {   // (int*)
          need(1);
          const uint64_t p = slot(0).read(lane, 0, 8);
          if (!p) {
            r[lane] = devrt::kInvalidValue;
            break;
          }
          put(lane, p, 4, static_cast<uint32_t>(f == Fn::GetDevice ? cfg_.device_ordinal : cfg_.device_count));
          break;
        }
        case Fn::RuntimeGetVersion: {   // (int*): the device runtime's own, 6000 whatever the toolkit
          need(1);
          const uint64_t p = slot(0).read(lane, 0, 8);
          if (!p) {
            r[lane] = devrt::kInvalidValue;
            break;
          }
          put(lane, p, 4, 6000);
          break;
        }
        case Fn::StreamCreateWithFlags:
        case Fn::EventCreateWithFlags: {   // (handle*, flags)
          need(2);
          const uint64_t p = slot(0).read(lane, 0, 8);
          const uint32_t flags = static_cast<uint32_t>(slot(1).read(lane, 0, 4));
          const bool stream = f == Fn::StreamCreateWithFlags;
          if (!p || !(stream ? devrt::stream_flags_ok(flags) : devrt::event_flags_ok(flags))) {
            r[lane] = devrt::kInvalidValue;
            break;
          }
          put(lane, p, 8, stream ? devrt::new_stream() : devrt::new_event());
          break;
        }
        case Fn::StreamDestroy:   // (stream): only a created stream, and again is no error
          need(1);
          if (devrt::stream_kind(slot(0).read(lane, 0, 8)) != devrt::StreamKind::Named) r[lane] = devrt::kInvalidValue;
          break;
        case Fn::EventDestroy:
          need(1);
          if (!devrt::valid_event(slot(0).read(lane, 0, 8))) r[lane] = devrt::kInvalidValue;
          break;
        case Fn::EventRecord:
        case Fn::EventRecordWithFlags:   // (event, stream[, flags])
          need(f == Fn::EventRecord ? 2 : 3);
          if (!devrt::valid_event(slot(0).read(lane, 0, 8)) ||
              devrt::stream_kind(slot(1).read(lane, 0, 8)) == devrt::StreamKind::Invalid)
            r[lane] = devrt::kInvalidValue;
          break;
        case Fn::StreamWaitEvent:   // (stream, event, flags)
          need(3);
          if (devrt::stream_kind(slot(0).read(lane, 0, 8)) == devrt::StreamKind::Invalid ||
              !devrt::valid_event(slot(1).read(lane, 0, 8)))
            r[lane] = devrt::kInvalidValue;
          break;
        case Fn::Malloc: {   // (void** p, size): the device heap, as malloc; *p is left alone on failure
          need(2);
          const uint64_t p = slot(0).read(lane, 0, 8), size = slot(1).read(lane, 0, 8);
          if (!p || !size) {
            r[lane] = devrt::kInvalidValue;
            break;
          }
          const uint64_t a = mem_.heap_alloc(size, cfg_.device_heap_bytes);
          if (!a) {
            r[lane] = devrt::kMemoryAllocation;
            break;
          }
          put(lane, p, 8, a);
          break;
        }
        case Fn::Free: {   // (ptr): a heap block goes back; any other address the card lets be (a
                           // local, a global variable); an unmapped one faults
          need(1);
          const uint64_t a = slot(0).read(lane, 0, 8);
          if (!a || is_local(a) || is_shared(a) || mem_.heap_free(a)) break;
          uint64_t base = 0, size = 0;
          if (!mem_.find_allocation(a, &base, &size))
            fail(lane, Err::InvalidFree,
                 "device cudaFree of a pointer that is neither live device memory nor a block of the device heap "
                 "(freed twice, or never allocated)");
          break;
        }
        case Fn::MemcpyAsync: {   // (dst, src, count, kind, stream): device to device only
          need(5);
          const uint64_t dst = slot(0).read(lane, 0, 8), src = slot(1).read(lane, 0, 8), n = slot(2).read(lane, 0, 8);
          if (!copy_kind_ok(static_cast<uint32_t>(slot(3).read(lane, 0, 4)))) {
            r[lane] = devrt::kInvalidMemcpyDirection;
            break;
          }
          if (n && (!dst || !src)) {
            r[lane] = devrt::kInvalidValue;
            break;
          }
          ChildLaunch c;
          auto op2 = std::make_shared<devrt::MemOp>();
          if (n) {
            op2->dst = device_addr(lane, dst);
            op2->src = device_addr(lane, src);
          }
          op2->width = n;
          c.op = std::move(op2);
          r[lane] = static_cast<uint64_t>(devrt_issue(std::move(c), slot(4).read(lane, 0, 8), ctx, true));
          break;
        }
        case Fn::Memcpy2DAsync: {   // (dst, dpitch, src, spitch, width, height, kind, stream)
          need(8);
          const uint64_t dst = slot(0).read(lane, 0, 8), dp = slot(1).read(lane, 0, 8), src = slot(2).read(lane, 0, 8),
                         sp = slot(3).read(lane, 0, 8), wd = slot(4).read(lane, 0, 8), ht = slot(5).read(lane, 0, 8);
          if (!copy_kind_ok(static_cast<uint32_t>(slot(6).read(lane, 0, 4)))) {
            r[lane] = devrt::kInvalidMemcpyDirection;
            break;
          }
          if (wd > dp || wd > sp) {
            r[lane] = devrt::kInvalidPitchValue;
            break;
          }
          if (wd && ht && (!dst || !src)) {
            r[lane] = devrt::kInvalidValue;
            break;
          }
          ChildLaunch c;
          auto op2 = std::make_shared<devrt::MemOp>();
          if (wd && ht) {
            op2->dst = device_addr(lane, dst);
            op2->src = device_addr(lane, src);
          }
          op2->dpitch = dp;
          op2->spitch = sp;
          op2->width = wd;
          op2->height = ht;
          c.op = std::move(op2);
          r[lane] = static_cast<uint64_t>(devrt_issue(std::move(c), slot(7).read(lane, 0, 8), ctx, true));
          break;
        }
        case Fn::Memcpy3DAsync: {   // (const cudaMemcpy3DParms*, stream)
          need(2);
          const uint64_t pa = slot(0).read(lane, 0, 8);
          if (!pa) {
            r[lane] = devrt::kInvalidValue;
            break;
          }
          // cudaMemcpy3DParms: srcArray 0, srcPos 8, srcPtr 32 (ptr, pitch, xsize, ysize), dstArray 64, dstPos 72,
          // dstPtr 96, extent 128, kind 152.
          const auto q = [&](uint64_t off) { return get(lane, pa + off, 8); };
          if (q(0) || q(64)) {   // arrays: not through the device runtime
            r[lane] = devrt::kInvalidValue;
            break;
          }
          if (!copy_kind_ok(static_cast<uint32_t>(get(lane, pa + 152, 4)))) {
            r[lane] = devrt::kInvalidMemcpyDirection;
            break;
          }
          const uint64_t sx = q(8), sy = q(16), sz = q(24), sptr = q(32), spitch = q(40), sys = q(56);
          const uint64_t dx = q(72), dy = q(80), dz = q(88), dptr = q(96), dpitch = q(104), dys = q(120);
          const uint64_t wd = q(128), ht = q(136), dep = q(144);
          if (wd > spitch || wd > dpitch) {
            r[lane] = devrt::kInvalidPitchValue;
            break;
          }
          if (wd && ht && dep && (!sptr || !dptr)) {
            r[lane] = devrt::kInvalidValue;
            break;
          }
          ChildLaunch c;
          auto op2 = std::make_shared<devrt::MemOp>();
          if (wd && ht && dep) {
            op2->src = device_addr(lane, sptr) + sx + sy * spitch + sz * spitch * sys;
            op2->dst = device_addr(lane, dptr) + dx + dy * dpitch + dz * dpitch * dys;
          }
          op2->spitch = spitch;
          op2->dpitch = dpitch;
          op2->sslice = spitch * sys;
          op2->dslice = dpitch * dys;
          op2->width = wd;
          op2->height = ht;
          op2->depth = dep;
          c.op = std::move(op2);
          r[lane] = static_cast<uint64_t>(devrt_issue(std::move(c), slot(1).read(lane, 0, 8), ctx, true));
          break;
        }
        case Fn::MemsetAsync:
        case Fn::Memset2DAsync:
        case Fn::Memset3DAsync: {
          // (ptr, value, count, stream) | (ptr, pitch, value, width, height, stream) |
          // (cudaPitchedPtr, value, cudaExtent, stream). The card's own fill ignores the value and writes zeros
          // (RTX 3060, driver 596.36: 0x5a, 0xff, 1 and -1 all gave 0); the documented fill is done here.
          uint64_t ptr, pitch = 0, wd, ht = 1, dep = 1, ysize = 1, stream;
          uint32_t value;
          if (f == Fn::MemsetAsync) {
            need(4);
            ptr = slot(0).read(lane, 0, 8);
            value = static_cast<uint32_t>(slot(1).read(lane, 0, 4));
            wd = slot(2).read(lane, 0, 8);
            pitch = wd;
            stream = slot(3).read(lane, 0, 8);
          } else if (f == Fn::Memset2DAsync) {
            need(6);
            ptr = slot(0).read(lane, 0, 8);
            pitch = slot(1).read(lane, 0, 8);
            value = static_cast<uint32_t>(slot(2).read(lane, 0, 4));
            wd = slot(3).read(lane, 0, 8);
            ht = slot(4).read(lane, 0, 8);
            stream = slot(5).read(lane, 0, 8);
          } else {
            need(4);
            ptr = slot(0).read(lane, 0, 8);
            pitch = slot(0).read(lane, 8, 8);
            ysize = slot(0).read(lane, 24, 8);
            value = static_cast<uint32_t>(slot(1).read(lane, 0, 4));
            wd = slot(2).read(lane, 0, 8);
            ht = slot(2).read(lane, 8, 8);
            dep = slot(2).read(lane, 16, 8);
            stream = slot(3).read(lane, 0, 8);
          }
          if (f != Fn::MemsetAsync && wd > pitch) {
            r[lane] = devrt::kInvalidValue;
            break;
          }
          if (wd && ht && dep && !ptr) {
            r[lane] = devrt::kInvalidValue;
            break;
          }
          ChildLaunch c;
          auto op2 = std::make_shared<devrt::MemOp>();
          op2->kind = devrt::MemOp::Kind::Set;
          if (wd && ht && dep) op2->dst = device_addr(lane, ptr);
          op2->dpitch = pitch;
          op2->dslice = pitch * ysize;
          op2->width = wd;
          op2->height = ht;
          op2->depth = dep;
          op2->value = static_cast<uint8_t>(value);
          c.op = std::move(op2);
          r[lane] = static_cast<uint64_t>(devrt_issue(std::move(c), stream, ctx, true));
          break;
        }
        case Fn::FuncGetAttributes: {   // (cudaFuncAttributes*, func): the first seven fields
          need(2);
          const uint64_t p = slot(0).read(lane, 0, 8);
          if (!p) {
            r[lane] = devrt::kInvalidValue;
            break;
          }
          const KernelRef& k = kernel_at(lane, slot(1).read(lane, 0, 8));
          uint8_t out[40];
          devrt::put_func_attrs(devrt::func_attributes(*k.fn, profile_, k.arch), out);
          for (uint32_t off = 0; off < 40; off += 8) {
            uint64_t v;
            std::memcpy(&v, out + off, 8);
            put(lane, p + off, 8, v);
          }
          break;
        }
        case Fn::DeviceGetAttribute: {   // (int* value, attr, device)
          need(3);
          const uint64_t p = slot(0).read(lane, 0, 8);
          const int attr = static_cast<int>(slot(1).read(lane, 0, 4));
          const int dev = static_cast<int>(slot(2).read(lane, 0, 4));
          if (!p) {
            r[lane] = devrt::kInvalidValue;
            break;
          }
          // The device runtime answers attributes 1 to 148 but 131 as the host does, 0 with 0, and nothing else.
          if (attr == 0) {
            put(lane, p, 4, 0);
            break;
          }
          if (attr < 0 || attr == 131 || attr >= 149) {
            r[lane] = devrt::kInvalidValue;
            break;
          }
          int v = 0;
          const int e = cfg_.devrt ? cfg_.devrt->attribute(dev, attr, &v) : devrt::kNotSupported;
          if (e) {
            r[lane] = static_cast<uint64_t>(e);
            break;
          }
          put(lane, p, 4, static_cast<uint32_t>(v));
          break;
        }
        case Fn::DeviceGetLimit: {   // (size_t*, limit)
          need(2);
          const uint64_t p = slot(0).read(lane, 0, 8);
          if (!p) {
            r[lane] = devrt::kInvalidValue;
            break;
          }
          uint64_t v = 0;
          const int e = cfg_.devrt ? cfg_.devrt->limit(cfg_.device_ordinal, static_cast<int>(slot(1).read(lane, 0, 4)), &v)
                                   : devrt::kNotSupported;
          if (e) {
            r[lane] = static_cast<uint64_t>(e);
            break;
          }
          put(lane, p, 8, v);
          break;
        }
        case Fn::DeviceGetCacheConfig:
        case Fn::DeviceGetSharedMemConfig: {   // (enum*)
          need(1);
          const uint64_t p = slot(0).read(lane, 0, 8);
          if (!p) {
            r[lane] = devrt::kInvalidValue;
            break;
          }
          int v = 0;
          int e = devrt::kNotSupported;
          if (cfg_.devrt)
            e = f == Fn::DeviceGetCacheConfig ? cfg_.devrt->cache_config(cfg_.device_ordinal, &v)
                                              : cfg_.devrt->shared_mem_config(cfg_.device_ordinal, &v);
          if (e) {
            r[lane] = static_cast<uint64_t>(e);
            break;
          }
          put(lane, p, 4, static_cast<uint32_t>(v));
          break;
        }
        case Fn::OccupancyMaxActiveBlocks:
        case Fn::OccupancyMaxActiveBlocksWithFlags: {   // (int* n, func, blockSize, dynamicSmem[, flags])
          need(f == Fn::OccupancyMaxActiveBlocks ? 4 : 5);
          const uint64_t p = slot(0).read(lane, 0, 8);
          const int32_t bs = static_cast<int32_t>(slot(2).read(lane, 0, 4));
          if (!p) {
            r[lane] = devrt::kInvalidValue;
            break;
          }
          if (bs <= 0) {   // the card writes 0 and refuses
            put(lane, p, 4, 0);
            r[lane] = devrt::kInvalidValue;
            break;
          }
          const KernelRef& k = kernel_at(lane, slot(1).read(lane, 0, 8));
          const auto res = kernel_resources(*k.fn, profile_, static_cast<uint32_t>(bs),
                                            static_cast<uint32_t>(slot(3).read(lane, 0, 8)));
          put(lane, p, 4, res.occupancy.blocks_per_sm);
          break;
        }
        case Fn::GetErrorString:
        case Fn::GetErrorName: {   // (cudaError_t): a string in device memory
          need(1);
          rbytes = 8;
          is_error = false;
          const int code = static_cast<int>(slot(0).read(lane, 0, 4));
          const char* text = cfg_.devrt ? (f == Fn::GetErrorString ? cfg_.devrt->error_string(code)
                                                                    : cfg_.devrt->error_name(code))
                                        : "unrecognized error code";
          r[lane] = mem_.intern_string(text);
          break;
        }
        case Fn::DeviceSynchronize:   // CDP1's: wait for the block's launches. Gone from sm_90 on.
          need(0);
          if (profile_.cc_major >= 9)
            fail(lane, Err::Unsupported,
                 "cudaDeviceSynchronize from device code, which NVIDIA's driver refuses to load a module for on "
                 "sm_90 and later (CUDA 12 removed it; build for CDP2 and use tail launches)");
          devrt_sync(ctx);
          break;
        case Fn::None:
          break;
      }
    }
    if (is_error) record_device_errors(w, m, r);
    write_call_result(w, op, m, r, rbytes);
  }

  void exec_device_heap(Warp& w, [[maybe_unused]] const BlockCtx& ctx, const Instr& ins, const OpCall& op, Mask m) {
    const bool allocating = op.callee == "malloc";
    if (op.param_slots.size() != 1)
      ctx_fail(ins, -1, Err::UnsupportedPtx,
               std::string(allocating ? "malloc" : "free") + " expects exactly one argument");
    auto it = w.slots.find(op.param_slots[0]);
    if (it == w.slots.end())
      ctx_fail(ins, -1, Err::UninitializedRegister, "device heap argument slot read before write");

    Lanes result{};
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      const uint64_t arg = it->second.read(lane, 0, 8);
      // The device's heap, shared with the SASS executor (MemoryManager::
      // heap_alloc). Each thread allocates independently, exactly as on
      // hardware -- this is not a warp-collective call.
      if (allocating) {
        result[lane] = mem_.heap_alloc(arg, cfg_.device_heap_bytes);   // 0: out of heap
      } else if (arg != 0 && !mem_.heap_free(arg)) {   // free(nullptr) is a no-op
        ctx_fail(ins, static_cast<int>(lane), Err::InvalidFree,
                 "device free() of a pointer the device heap did not allocate, or already freed");
      }
    }
    if (allocating && !op.retval_slot.empty()) {
      Warp::Slot& out = w.slots[op.retval_slot];
      if (out.bytes.empty()) out.reset(8, W_);
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) out.write(lane, 0, 8, result[lane]);
    }
  }

  void exec_vprintf(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpCall& op, Mask m) {
    if (op.param_slots.size() != 2)
      ctx_fail(ins, -1, Err::UnsupportedPtx, "vprintf expects exactly 2 arguments (format, valist)");
    auto fmt_it = w.slots.find(op.param_slots[0]);
    auto va_it = w.slots.find(op.param_slots[1]);
    if (fmt_it == w.slots.end() || va_it == w.slots.end())
      ctx_fail(ins, -1, Err::UninitializedRegister, "vprintf argument slot read before write");

    Lanes counts{};
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (Mask{1} << lane))) continue;
      std::string fmt = read_cstring(w, ctx, ins, lane, fmt_it->second.read(lane, 0, 8));
      uint64_t valist = va_it->second.read(lane, 0, 8);
      uint64_t cursor = 0;

      auto fetch = [&](uint32_t size) -> uint64_t {
        cursor = (cursor + size - 1) / size * size;  // natural alignment in the valist
        if (valist == 0)
          ctx_fail(ins, static_cast<int>(lane), Err::InvalidValue,
                   "printf format consumes arguments but no argument buffer was passed");
        uint64_t v = load_routed(w, ctx, ins, lane, valist + cursor, size);
        cursor += size;
        return v;
      };

      const std::string out = vgpu::exec::format_device_printf(
          fmt, fetch, [&](uint64_t a) { return read_cstring(w, ctx, ins, lane, a); },
          [&](Err e, const char* why) { ctx_fail(ins, static_cast<int>(lane), e, why); });
      vgpu::exec::emit_device_printf(out);
      counts[lane] = out.size();
    }
    if (!op.retval_slot.empty()) {
      Warp::Slot& slot = w.slots[op.retval_slot];
      if (slot.bytes.empty()) slot.reset(4, W_);
      for (uint32_t lane = 0; lane < W_; ++lane)
        if (m & (Mask{1} << lane)) slot.write(lane, 0, 4, counts[lane]);
    }
  }

  // Publishes elapsed busy time mid-launch so telemetry stays live during a
  // long kernel instead of freezing until the launch returns.
  void report_progress() {
    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - last_progress_).count();
    last_progress_ = now;
    progress_(dt);
  }

  // Striped locks for device-side atomics when the grid is running on more
  // than one thread. Sized well above the core count so unrelated addresses
  // rarely collide.
  static std::mutex& atomic_lock_for(uint64_t addr) {
    static std::array<std::mutex, 251> locks;
    return locks[(addr >> 2) % locks.size()];
  }

  // A vector access marked .relaxed, .volatile, .acquire or .release is done
  // element by element, so another block's thread could see half of it. CUB's
  // decoupled look-back stores a tile's {status, value} with one
  // st.relaxed.gpu.v2 and polls it with one ld.relaxed.gpu.v2, and a torn read
  // is a prefix sum from a new status and an old value. Hardware performs an
  // aligned vector access as one; so does holding this lock, keyed by the
  // 16-byte line, across all of its elements. Unordered vector accesses -- every
  // vectorized load in a GEMM -- stay lock-free.
  static std::mutex& vector_lock(uint64_t addr) {
    static std::array<std::mutex, 251> locks;
    return locks[(addr >> 4) % locks.size()];
  }

  const EntryFn& fn_;
  // The function currently executing. Equal to &fn_ except while a call to a
  // device function is in flight, when everything that reads a body, a
  // register count or a .local frame must follow the callee instead.
  const EntryFn* cur_ = &fn_;
  // Where cur_'s .local frame starts in each lane's local buffer: 0 in the
  // kernel, and above its caller's frame in a device function, the way a
  // stack grows. With every frame at 0, a callee's locals overwrote its
  // caller's, and a pointer into the caller's frame (NanoVDB's device-side
  // cudaGetDevice(&dev)) was out of bounds in a callee with no frame of its own.
  uint64_t local_base_ = 0;
  const LaunchConfig& cfg_;
  const ParamBuffer& params_;
  MemoryManager& mem_;
  const DeviceProfile& profile_;
  // Lanes per warp, from the device profile: 32 on NVIDIA, 64 on a CDNA
  // wavefront. Every per-lane loop is bounded by this rather than by the array
  // size, so an NVIDIA warp does not walk 32 lanes that are not there.
  const uint32_t W_ = 32;
  const Mask all_ = all_lanes(32);
  const SymbolTable* symbols_;
  LaunchStats& stats_;
  ProgressFn progress_;
  // One entry per instruction, filled on first use. Classifying costs a chain
  // of variant tests, and the instruction stream is the hottest path there is.
  mutable std::vector<uint8_t> class_by_pc_;
  uint32_t cur_warp_ = 0;     // which warp of the block is running, for race reports
  bool fast_enabled_ = true;   // VGPU_FASTPATH, sampled at launch
  bool host_directed_ = false;   // this thread was not rounding to nearest at launch
  // %gridid: a serial number distinguishing this launch from every other one in
  // the process. Assigned once per launch rather than per interpreter, so the
  // workers of one launch agree.
  uint64_t grid_id_ = 0;
  bool concurrent_ = false;   // set when the grid is split across threads
  unsigned worker_ = 0;       // this interpreter's host thread, 0..threads-1
  std::chrono::steady_clock::time_point last_progress_;
};

ParamBuffer build_params(const EntryFn& fn, const std::vector<std::vector<uint8_t>>& args) {
  if (args.size() != fn.params.size())
    throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "' expects ", fn.params.size(),
                      " parameters, launch supplied ", args.size());
  ParamBuffer pb;
  for (size_t i = 0; i < args.size(); ++i) {
    uint32_t size = fn.params[i].size;
    if (args[i].size() != size)
      throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "' parameter '", fn.params[i].name,
                        "' is ", size, " bytes, launch supplied ", args[i].size(), " bytes");
    uint32_t align = fn.params[i].align ? fn.params[i].align : (size < 8 ? size : 8);
    uint32_t off = static_cast<uint32_t>((pb.bytes.size() + align - 1) / align * align);
    pb.bytes.resize(off + size);
    std::memcpy(pb.bytes.data() + off, args[i].data(), size);
    pb.layout[fn.params[i].name] = {off, size};
  }
  return pb;
}

}  // namespace

// Shared with the SASS executor (tcgen05.hpp).
// A tcgen05 shared-memory matrix descriptor (9.7.18.4.1). The same
// canonical layouts as wgmma's; the swizzle field is three bits wide here.
WgmmaDesc tcgen05_desc(uint64_t d, Tcgen05Host& h, bool lbo_abs_ok) {
  WgmmaDesc out;
  out.start = (d & 0x3FFF) << 4;
  out.lbo = ((d >> 16) & 0x3FFF) << 4;
  out.sbo = ((d >> 32) & 0x3FFF) << 4;
  switch ((d >> 61) & 7) {
    case 0: out.swizzle = 0; break;
    case 2: out.swizzle = 128; break;
    case 4: out.swizzle = 64; break;
    case 6: out.swizzle = 32; break;
    // 128 bytes swizzled in 32-byte atoms: the same canonical strides, the
    // swizzle moving 32-byte chunks (Swizzle<2,5,2>, as TMA's
    // CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B writes the tile).
    case 1: out.swizzle = 128; out.atom = 32; break;
    default:
      h.fail(Err::InvalidValue,
               "tcgen05 matrix descriptor swizzle mode " + std::to_string((d >> 61) & 7) +
                   " (bits 61-63), which the ISA makes invalid");
  }
  if (((d >> 46) & 7) != 1)
    h.fail(Err::InvalidValue,
             "a tcgen05 matrix descriptor needs the fixed value 0b001 in bits 46-48");
  if (((d >> 49) & 7) != 0)
    h.fail(Err::UnsupportedPtx,
             "a tcgen05 matrix descriptor with a nonzero base offset (bits 49-51); only swizzle "
             "patterns that start on their repeat boundary are implemented");
  if ((d >> 52) & 1) {
    if (!lbo_abs_ok)
      h.fail(Err::InvalidValue,
             "a tcgen05 matrix descriptor with an absolute leading-dimension address (bit 52) "
             "needs an sm_103a target (9.7.18.4.1.1)");
    out.lbo_abs = true;
  }
  return out;
}

void tmem_fragment(ptx::Tcgen05Shape s, uint32_t t, uint32_t j, uint32_t* lane, uint32_t* col) {
  switch (s) {
    case ptx::Tcgen05Shape::S32x32b: *lane = t; *col = j; return;
    case ptx::Tcgen05Shape::S16x64b: *lane = t / 4 + 8 * (t % 2); *col = 2 * j + (t / 2) % 2; return;
    case ptx::Tcgen05Shape::S16x128b: *lane = t / 4 + 8 * (j % 2); *col = t % 4 + 4 * (j / 2); return;
    case ptx::Tcgen05Shape::S16x256b:
      *lane = t / 4 + 8 * ((j / 2) % 2);
      *col = 2 * (t % 4) + j % 2 + 8 * (j / 4);
      return;
    case ptx::Tcgen05Shape::S16x32bx2: *lane = t % 16; *col = j; return;
  }
}


// tcgen05.cp (9.7.18.9.2): rows of the shared-memory matrix the descriptor
// describes -- K-major, 16 or 32 bytes a row -- into Tensor Memory lanes,
// one row to a lane from the address's lane on, packed into the columns
// from its column. .32x128b.warpx4 writes its 32 rows into every 32-lane
// partition. With .cta_group::2 each CTA of the pair copies its own shared
// memory into its own Tensor Memory. It completes when issued, like the
// other asynchronous tcgen05 operations; tcgen05.commit tracks it.
void tcgen05_cp(const Tcgen05Cp& op, Tcgen05Host& h) {
  uint32_t rows = 128, bytes = 32;
  switch (op.shape) {
    case ptx::Tcgen05CpShape::S128x256b: break;
    case ptx::Tcgen05CpShape::S4x256b: rows = 4; break;
    case ptx::Tcgen05CpShape::S128x128b: bytes = 16; break;
    case ptx::Tcgen05CpShape::S64x128b: rows = 64; bytes = 16; break;
    case ptx::Tcgen05CpShape::S32x128b: rows = 32; bytes = 16; break;
  }
  const uint32_t taddr = op.taddr;
  const uint32_t lane0 = taddr >> 16, col0 = taddr & 0xFFFF;
  const WgmmaDesc d = tcgen05_desc(op.desc, h);
  // Where source row r lands: .warpx4 in all four 32-lane quarters;
  // .warpx2::02_13 at lanes r and r + 64, .warpx2::01_23 at 64(r / 32) +
  // r % 32 and 32 lanes on -- warps 0 and 2 (or 0 and 1) receiving the
  // same half, as CuTe's UTCCP 2x64dp copy traits lay the destination out.
  const uint32_t copies = op.multicast == 4 ? 4 : op.multicast ? 2 : 1;
  auto dst_lane = [&](uint32_t r, uint32_t p) {
    switch (op.multicast) {
      case 4: return r + 32 * p;
      case 2: return r + 64 * p;
      case 3: return 64 * (r / 32) + r % 32 + 32 * p;
      default: return r;
    }
  };
  // Decompression (9.7.18.9.1): each 16-byte group of the source holds
  // sixteen 4-bit values in its first 8 bytes (.b4x16_p64) or sixteen 6-bit
  // ones in its first 12 (.b6x16_p32), and becomes sixteen bytes -- fp4 in
  // bits 2-5, fp6 in bits 0-5 (figures 198, 200-201).
  auto source_byte = [&](uint32_t rank, uint32_t r, uint32_t byte) -> uint32_t {
    auto raw = [&](uint32_t b) {
      const uint64_t at = wgmma_smem_offset(d, true, 1, r, b);
      return static_cast<uint32_t>(h.smem_load(rank, at, 1));
    };
    if (!op.decompress) return raw(byte);
    const uint32_t bits = static_cast<uint32_t>(op.decompress), group = byte / 16, j = byte % 16;
    const uint32_t bit = j * bits;
    uint32_t v = raw(16 * group + bit / 8) | raw(16 * group + (bit + bits - 1) / 8) << 8;
    v = (v >> (bit % 8)) & ((1u << bits) - 1);
    return bits == 4 ? v << 2 : v;
  };
  for (uint32_t c = 0; c < h.ctas(); ++c) {
    TensorMemory& t = h.tmem(c);
    const uint32_t rank = c;
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t cw = 0; cw < bytes / 4; ++cw) {
        uint32_t word = 0;
        for (uint32_t b = 0; b < 4; ++b) word |= source_byte(rank, r, cw * 4 + b) << (8 * b);
        for (uint32_t p = 0; p < copies; ++p) {
          const uint32_t l = lane0 + dst_lane(r, p), col = col0 + cw;
          if (l >= TensorMemory::kLanes || col >= TensorMemory::kCols || !t.allocated(col))
            h.fail(Err::OutOfBounds,
                     "tcgen05.cp writes Tensor Memory lane " + std::to_string(l) + ", column " +
                         std::to_string(col) + ", outside what tcgen05.alloc allocated");
          t.at(l, col) = word;
        }
      }
  }
}

// tcgen05.shift.down (9.7.18.9.3): the implicit .31x256b shape -- rows 0-30
// of the 32 lanes at the (32-aligned) address move down one row, 256 bits
// (eight columns) each; row 0 is not written. In this CTA, or both of a pair.
void tcgen05_shift(uint32_t taddr, Tcgen05Host& h) {

  const uint32_t lane0 = taddr >> 16, col0 = taddr & 0xFFFF;
  if (lane0 % 32)
    h.fail(Err::InvalidValue, "tcgen05.shift's address must have a lane aligned to 32");
  for (uint32_t c = 0; c < h.ctas(); ++c) {
    TensorMemory& t = h.tmem(c);
    for (uint32_t col = col0; col < col0 + 8; ++col)
      if (lane0 + 32 > TensorMemory::kLanes || col >= TensorMemory::kCols || !t.allocated(col))
        h.fail(Err::OutOfBounds,
                 "tcgen05.shift touches Tensor Memory column " + std::to_string(col) +
                     ", which no tcgen05.alloc has allocated");
    for (uint32_t r = 31; r >= 1; --r)
      for (uint32_t col = col0; col < col0 + 8; ++col) t.at(lane0 + r, col) = t.at(lane0 + r - 1, col);
  }
}

// One thread's tcgen05.mma (9.7.18.10.10.1): D = A*B (+ D), M x N x K, on
// this CTA's Tensor Memory or the pair's; with .block_scale,
// (A * scale_A) * (B * scale_B) + D (9.7.18.10.7).
void tcgen05_mma(const Tcgen05Mma& op, Tcgen05Host& h) {
  auto refuse = [&](const std::string& why) { h.fail(Err::UnsupportedPtx, "tcgen05.mma: " + why); };
  const uint32_t id = op.idesc;
  auto bad = [&]() {
    h.fail(Err::InvalidValue,
             "tcgen05.mma instruction descriptor 0x" + [&] {
               char b[12];
               std::snprintf(b, sizeof b, "%08x", id);
               return std::string(b);
             }() + " has fields this .kind does not define (Tables 51-53)");
  };
  const bool mx = op.block_scale;
  bool lane128 = false, k64 = false;
  const bool sm103a = op.target_sm == 103 && op.target_arch;
  const bool mxf4 = op.kind == Tcgen05MmaKind::MXF4 || op.kind == Tcgen05MmaKind::MXF4NVF4;
  // The instruction descriptor: Table 51, or 52/53 for the block-scaled kinds.
  const bool sp = op.sparse;
  if (bool(id >> 2 & 1) != sp)
    h.fail(Err::InvalidValue,
             sp ? "tcgen05.mma.sp needs the instruction descriptor's sparsity bit (2) set"
                : "the instruction descriptor asks for sparse A (bit 2) on a dense tcgen05.mma; that is "
                  "tcgen05.mma.sp");
  const uint32_t sp_sel = sp ? id & 3 : 0;
  const bool sat = !mx && (id >> 3 & 1);
  const uint32_t dtype = mx ? 1 : id >> 4 & 3;
  const uint32_t atype = id >> 7 & 7, btype = mxf4 ? id >> 10 & 3 : id >> 10 & 7;
  const bool neg_a = id >> 13 & 1, neg_b = id >> 14 & 1;
  const bool trans_a = id >> 15 & 1, trans_b = id >> 16 & 1;
  const uint32_t N = (id >> 17 & 0x3F) << 3;
  const uint32_t M = mx ? (id >> 27 & 3) << 7 : (id >> 24 & 0x1F) << 4;
  const uint32_t sfb_id = id >> 4 & 3, sfa_id = id >> 29 & 3;
  if (mx) {
    if ((!sp && (id & 3)) || (id & 0x40) || (!mxf4 && (id & 8)) || (mxf4 && (id >> 25 & 1)) ||
        (!mxf4 && (id >> 24 & 3)))
      h.fail(Err::InvalidValue, "tcgen05.mma instruction descriptor sets a reserved bit");
    // Bit 26: scale factor A in the 128-lane layout (figures 238-242, 256), which only sm_107f has.
    lane128 = id >> 26 & 1;
    if (lane128 && op.target_sm != 107)
      refuse("the 128-lane scale-factor A layout (instruction descriptor bit 26) is sm_107f's; this module targets sm_" +
             std::to_string(op.target_sm));
    // Bit 31 of .kind::mxf8f6f4: K = 64 (dense), sm_107f's. The mxf4 kinds' bit 3 (K = 128) is not implemented:
    // its scale-factor layouts for B are drawn only in figures.
    if (!mxf4 && (id >> 31)) {
      if (op.target_sm != 107)
        refuse("K = 64 (instruction descriptor bit 31) is sm_107f's; this module targets sm_" + std::to_string(op.target_sm));
      k64 = true;
    }
    if (mxf4 && (id >> 3 & 1))
      refuse(".kind::mxf4's K = 128 (instruction descriptor bit 3) is not implemented: the layout of its scale factors "
             "for B is drawn in figures only");
    // .kind::mxf4/mxf4nvf4 with bit 31: K = 96, dense, sm_103a (9.7.18.2.1.1;
    // sm_107a's is not implemented). CuTe's descriptor calls the sparse
    // K = 192 invalid, and the ISA's target note names only K = 96.
    if (mxf4 && (id >> 31)) {
      if (sp) refuse("sparse A with the K = 96 bit (31): a K of 192, which the ISA does not define");
      if (!sm103a)
        refuse("K = 96 (instruction descriptor bit 31) needs an sm_103a target; this module targets sm_" +
               std::to_string(op.target_sm) + (op.target_arch ? "a" : ""));
    }
    if (mxf4 && (id >> 12 & 1)) refuse("sparsity version v1 (bit 12) is sm_107's");
  } else {
    if ((id & 0x40) || (id & (1u << 23)))
      h.fail(Err::InvalidValue, "tcgen05.mma instruction descriptor sets a reserved bit");
    if ((id >> 30) && !op.ws)
      refuse("the instruction descriptor's B-reuse shift (bits 30-31) is for .ws only");
    if (id >> 29 & 1) {
      // Bit 29: K = 64 for .kind::f8f6f4 (dense), sm_107f's; no other kind defines it.
      if (op.kind != Tcgen05MmaKind::F8F6F4) bad();
      if (op.target_sm != 107)
        refuse("K = 64 (instruction descriptor bit 29) is sm_107f's; this module targets sm_" + std::to_string(op.target_sm));
      k64 = true;
    }
  }
  // Element types by kind, and K: 256 bits of an 8-bit-container row.
  auto f8f6f4_elem = [&](uint32_t t) {
    TcElem e;
    switch (t) {
      case 0: e.t = TcType::E4M3; e.bits = 8; break;
      case 1: e.t = TcType::E5M2; e.bits = 8; break;
      case 3: e.t = TcType::E2M3; e.bits = 6; break;
      case 4: e.t = TcType::E3M2; e.bits = 6; break;
      case 5: e.t = TcType::E2M1; e.bits = 4; break;
      default: bad();
    }
    e.per16 = 16;
    e.cbits = 8;
    return e;
  };
  TcElem ea, eb2;
  uint32_t K = 0;
  bool d_f16 = false, d_int = false;
  switch (op.kind) {
    case Tcgen05MmaKind::F16:
      if (atype > 1 || btype != atype || dtype > 1 || (dtype == 0 && atype != 0)) bad();
      ea.t = atype ? TcType::BF16 : TcType::F16;
      eb2 = ea;
      d_f16 = dtype == 0;
      K = 16;
      break;
    case Tcgen05MmaKind::TF32:
      if (atype != 2 || btype != 2 || dtype != 1) bad();
      ea = TcElem{TcType::TF32, 32, 4, 32};
      eb2 = ea;
      K = 8;
      break;
    case Tcgen05MmaKind::F8F6F4:
    case Tcgen05MmaKind::MXF8F6F4:
      if (dtype > 1) bad();
      ea = f8f6f4_elem(atype);
      eb2 = f8f6f4_elem(btype);
      d_f16 = dtype == 0;
      K = 32;
      if (k64) {
        // K = 64 takes e4m3 and e5m2 only (Table 48), dense -- the sparse K = 128's metadata is not drawn --
        // at M = 128 (a CTA) or 256 (a pair).
        if (atype > 1 || btype > 1) bad();
        if (sp) refuse("sparse A with K = 64 (a K of 128): the metadata for it is not drawn in the ISA's figures");
        K = 64;
      }
      break;
    case Tcgen05MmaKind::I8:
      if (atype > 1 || btype > 1 || dtype != 2) bad();
      ea = TcElem{atype ? TcType::S8 : TcType::U8, 8, 16, 8};
      eb2 = TcElem{btype ? TcType::S8 : TcType::U8, 8, 16, 8};
      d_int = true;
      K = 32;
      if (neg_a || neg_b) bad();
      break;
    case Tcgen05MmaKind::TI16:
      // sm_107's: s1z4m11 elements into s32, K = 16 (32 sparse), with negate and
      // transpose but no saturation (Tables 48, 51 and 62); from the ISA alone.
      if (atype != 3 || btype != 3 || dtype != 2 || sat) bad();
      ea = eb2 = TcElem{TcType::TI16, 16, 8, 16};
      d_int = true;
      K = 16;
      break;
    case Tcgen05MmaKind::MXF4:
    case Tcgen05MmaKind::MXF4NVF4:
      if (atype != 1 || btype != 1) bad();
      ea = eb2 = TcElem{TcType::E2M1, 4, 32, 4};
      K = id >> 31 ? 96 : 64;
      if (trans_a || trans_b) bad();   // Table 62: no transpose for mxf4
      break;
  }
  // Sparse A (9.7.18.10.9): K doubles and A holds half of each row. Each
  // chunk of sp_w elements keeps half of them, placed by a 4-bit metadata
  // field: 2:4 for most kinds, 1:2 for tf32 and 4:8 in pairs for mxf4*.
  // The 8-bit kinds keep a row's metadata in one lane and take no
  // selector; f16 and tf32 pick the column with it (figures 287-292).
  const bool meta_rows = op.kind != Tcgen05MmaKind::F16 && op.kind != Tcgen05MmaKind::TF32 && op.kind != Tcgen05MmaKind::TI16;
  uint32_t sp_w = 0;
  if (sp) {
    K *= 2;
    sp_w = op.kind == Tcgen05MmaKind::TF32 ? 2 : mxf4 ? 8 : 4;
    if (meta_rows && !mx && sp_sel)
      h.fail(Err::InvalidValue,
               "the sparsity selector must be 0 for .kind::i8 and .kind::f8f6f4 (9.7.18.10.9.4)");
    if (!meta_rows && sp_sel > 1)
      refuse("sparsity selector " + std::to_string(sp_sel) +
             ": the ISA's figures 287-290 place the metadata for selectors 0 and 1 only");
  }
  const uint32_t Ka = sp ? K / 2 : K;   // A's stored elements a row
  if (sat && !d_int) bad();
  // Table 62: the fp6/fp4 types transpose too, except at the dense K = 64
  // (sm_107's, refused above); the mxf4 kinds never do.
  // Block scaling: scale factors per row of K, and their type.
  uint32_t sv = 0;          // scale factors per row of A / column of B
  bool ue4m3 = false;
  if (mx) {
    switch (op.kind) {
      case Tcgen05MmaKind::MXF8F6F4:
        if (op.scale_vec != 0 && op.scale_vec != 1 && op.scale_vec != 32) bad();
        sv = 1;
        if (!(id >> 23 & 1)) bad();   // UE8M0 is its only scale type
        break;
      case Tcgen05MmaKind::MXF4:
        if (op.scale_vec != 0 && op.scale_vec != 2 && op.scale_vec != 32) bad();
        sv = 2;
        if ((id >> 23 & 3) != 1) bad();
        break;
      default: {   // mxf4nvf4: the size must be named
        if (op.scale_vec == 0) refuse(".kind::mxf4nvf4 needs a scale vector size");
        sv = op.scale_vec == 2 || op.scale_vec == 32 ? 2 : op.scale_vec == 4 || op.scale_vec == 16 ? 4 : 0;
        if (!sv) bad();
        const uint32_t st = id >> 23 & 3;
        if (st == 2) refuse("UE5M3 scale factors are sm_107f's");
        if (st > 1) bad();
        ue4m3 = st == 0;
        if (ue4m3 && sv == 2 && op.target_sm != 107) refuse("UE4M3 scale factors with .block32 are sm_107f's");
        break;
      }
    }
    // K = 96 (Table 68): .block32 is three factors a row, .block16 six; the
    // .scale_vec::NX spellings name no K = 96 form.
    if (K == 96) {
      if (op.scale_vec != 0 && op.scale_vec != 16 && op.scale_vec != 32)
        refuse(".scale_vec::" + std::to_string(op.scale_vec) + "X with K = 96; Table 68 gives K = 96 "
               ".block16 and .block32 only");
      sv = op.scale_vec == 16 ? 6 : 3;
      // Figures 243-275: .block32's factors start at any byte of the word,
      // .block16's at byte 0 or 2.
      if (sv == 6 && (sfa_id % 2 || sfb_id % 2)) bad();
    }
    // .block16/.block32 are aliases of 4X/2X at K = 64 and 128 -- a sparse
    // K = 128 included (9.7.18.10.10.1's "Aliased .scale_vectorsize
    // variants") -- so the factors stay four or two, each covering K/4 or
    // K/2. Table 68's six and eight belong to sm_103/107's larger K,
    // refused above; CUTLASS's sparse nvf4 GEMMs issue .block16 over
    // K = 128 with a factor per 32.
    // Byte-aligned sub-columns: 1X any byte, 2X a half word, 4X all four.
    if (K != 96 && ((sv == 2 && (sfa_id % 2 || sfb_id % 2)) || (sv == 4 && (sfa_id || sfb_id)))) bad();
  }
  const uint32_t G = op.cta_group;
  // .ws (Table 48): M = 32, 64 or 128 by N = 64, 128 or 256; sparse, N up
  // to 128.
  const bool shape_ok =
      op.ws ? G == 1 && (M == 32 || M == 64 || M == 128) && (N == 64 || N == 128 || (N == 256 && !sp))
      : mx ? (G == 1 ? M == 128 && N >= 8 && N <= 256 && N % 8 == 0
                   : (M == 128 || M == 256) && N >= 16 && N <= 256 && N % 16 == 0)
      : G == 1 ? (M == 64 || M == 128) && N >= 8 && N <= 256 && N % 8 == 0 &&
                     !(d_int && M == 128 && N % 16)
               : (M == 128 || M == 256) && N >= 16 && N <= 256 && N % (d_int ? 32 : 16) == 0;
  if (!shape_ok)
    h.fail(Err::InvalidValue,
             "tcgen05.mma.cta_group::" + std::to_string(G) + " of shape M=" + std::to_string(M) +
                 " N=" + std::to_string(N) + ", which Table 48 does not define");
  if (lane128 && (op.ws || (G == 2 && M == 128)))
    refuse("the 128-lane scale-factor A layout with a CTA pair at M = 128: each CTA's rows then sit in D lanes the "
           "figures do not tie to a scale lane");
  if (k64 && !((G == 1 && M == 128) || (G == 2 && M == 256)) || (k64 && op.ws))
    h.fail(Err::InvalidValue, "tcgen05.mma with K = 64 is 128xNxK1 on one CTA or 256xNxK1 on a pair (Table 48), and not .ws");
  if (op.a_tmem && trans_a) bad();
  // Below M = 128 the ISA draws only D's .ws layouts (E and G), not where A
  // or the sparsity metadata sit in Tensor Memory.
  if (op.ws && M < 128 && op.a_tmem)
    refuse(".ws with A in Tensor Memory at M = " + std::to_string(M) +
           ": the ISA does not draw A's layout for it (layouts E and G are D's)");
  if (op.ws && M < 128 && sp)
    refuse(".ws.sp at M = " + std::to_string(M) +
           ": figures 287-292 place the sparsity metadata for M = 64 without .ws and for M >= 128 only");

  const uint32_t Mloc = M / G, Nloc = N / G;
  const uint32_t d_addr = op.d;
  const uint32_t d_lane0 = d_addr >> 16, d_col0 = d_addr & 0xFFFF;
  // Where D element (row m of this CTA's Mloc, column n) lives: the
  // data-path layouts of 9.7.18.10.5 -- D (M=128), F (M=64, lanes 0-15 or
  // 16-31 of each warp's quarter), A (M=256 over a pair) and B (M=128 over a
  // pair, the upper half of N in lanes 64-127).
  // Sparse A over a pair with M = 128 is layout C instead of B: 64 rows
  // a CTA, placed as layout F places its 64 (figures 215-216). .ws spreads
  // N over the idle lanes instead: layout E (M = 64) puts N's upper half
  // in lanes 64-127 as B does, and layout G (M = 32) its quarters in each
  // warp's 32 (figures 219 and 223; CUTLASS's tmem_frg_ws agrees).
  // Figures 220 and 224 address those regions at lanes 0 and 32 only, but
  // a warp reaches only its own quarter of the lanes (9.7.18.5), so those
  // can only be the figures' slip.
  const bool layout_b = (G == 2 && M == 128 && !sp) || (op.ws && M == 64);
  const bool layout_g = op.ws && M == 32;
  const bool layout_f = !op.ws && ((G == 1 && M == 64) || (G == 2 && M == 128 && sp));
  const uint32_t n_parts = layout_b ? 2 : layout_g ? 4 : 1;
  auto lane_align_ok = [&](uint32_t l) { return layout_f ? (l == 0 || l == 16) : l == 0; };
  if (!lane_align_ok(d_lane0))
    h.fail(Err::InvalidValue,
             "tcgen05.mma's D address has lane " + std::to_string(d_lane0) +
                 "; this shape's data-path layout starts at lane 0" +
                 (layout_f ? " or 16" : ""));
  auto d_pos = [&](uint32_t m, uint32_t n, uint32_t lane0, uint32_t* dl, uint32_t* dc) {
    if (layout_f) { *dl = (m / 16) * 32 + m % 16 + lane0; *dc = n; }
    else { *dl = m + 128 / n_parts * (n / (N / n_parts)); *dc = n % (N / n_parts); }
  };
  const uint32_t d_cols = N / n_parts;
  for (uint32_t c = 0; c < G; ++c)
    for (uint32_t col = d_col0; col < d_col0 + d_cols; ++col)
      if (col >= TensorMemory::kCols || !h.tmem(c).allocated(col))
        h.fail(Err::OutOfBounds,
                 "tcgen05.mma writes D to Tensor Memory column " + std::to_string(col) +
                     ", which no tcgen05.alloc has allocated" + (c != h.self() ? " in the peer CTA" : ""));
  const double sa = neg_a ? -1.0 : 1.0, sb = neg_b ? -1.0 : 1.0;

  // Element (mn, k) of an operand in shared memory. K-major, it is bits at
  // (k % per16) * bits within 16-byte group k / per16 of the row, through
  // the canonical layout byte by byte; MN-major, the canonical layout of
  // whole elements, or for fp6/fp4 the same 16-byte groups running along
  // MN (as TMA's .b6x16_p32/.b4x16_p64 write an MN-major tile).
  auto smem_raw = [&](const WgmmaDesc& d, bool k_major, const TcElem& e, uint32_t rank, uint32_t mn,
                      uint32_t k) -> uint32_t {
    auto byte_at = [&](uint64_t off) { return static_cast<uint32_t>(h.smem_load(rank, off, 1)); };
    if (!k_major && e.bits < 8) {
      const uint32_t bit = (mn % e.per16) * e.bits;
      const uint32_t byte = 16 * (mn / e.per16) + bit / 8;
      uint32_t raw = 0;
      for (uint32_t i = 0; i < (bit % 8 + e.bits + 7) / 8; ++i)
        raw |= byte_at(wgmma_smem_offset(d, false, 1, byte + i, k)) << (8 * i);
      return (raw >> (bit % 8)) & ((1u << e.bits) - 1);
    }
    if (!k_major) {
      const uint32_t eb = e.bits / 8;
      const uint64_t at = wgmma_smem_offset(d, false, eb, mn, k);
      return static_cast<uint32_t>(h.smem_load(rank, at, eb));
    }
    const uint32_t bit = (k % e.per16) * e.bits;
    const uint32_t byte = 16 * (k / e.per16) + bit / 8;
    uint32_t raw = 0;
    const uint32_t nbytes = (bit % 8 + e.bits + 7) / 8;
    for (uint32_t i = 0; i < nbytes; ++i)
      raw |= byte_at(wgmma_smem_offset(d, true, 1, mn, byte + i)) << (8 * i);
    return (raw >> (bit % 8)) & ((e.bits >= 32) ? 0xFFFFFFFFu : ((1u << e.bits) - 1));
  };
  // B: K x N, CTA v supplying columns [v*Nloc, (v+1)*Nloc) from its own
  // shared memory at the descriptor's offsets (the peer's half of a pair
  // sits at the same offsets there).
  //
  // .ws's zero-column mask descriptor (9.7.18.4.3) zeroes whole columns of B
  // and shifts which columns are read: MMA column n reads B's column
  // n + shift. The mask is one sub-mask per N / 1, 2 or 4 columns as M is
  // 128, 64 or 32, each a run of fs_i's value, sc_i bits short, then runs
  // alternating. The ISA's four worked examples make a run of 1s (zeroed
  // columns) Skip Span + 1 long and a run of 0s Use Span + 1 long -- as the
  // names say, though Table 54's one-line descriptions have them the other
  // way round; the examples are what is followed here.
  std::vector<uint8_t> zero_col(N, 0);
  uint32_t col_shift = 0;
  if (op.has_zero_mask) {
    const uint64_t zm = op.zero_mask;
    if ((zm >> 36 & 7) || (zm >> 62))
      h.fail(Err::InvalidValue,
               "tcgen05.mma.ws zero-column mask descriptor sets a reserved bit (36-38 or 62-63)");
    col_shift = zm >> 56 & 0x3F;
    if (col_shift > (M == 32 ? 16u : 32u))
      h.fail(Err::InvalidValue,
               "tcgen05.mma.ws column shift " + std::to_string(col_shift) + " is over the " +
                   (M == 32 ? "16" : "32") + " allowed at M = " + std::to_string(M) + " (Table 54)");
    if (zm >> 39 & 1) {
      const uint32_t subs = M == 128 ? 1 : M == 64 ? 2 : 4, width = N / subs;
      const uint32_t run1 = (zm >> 40 & 0xFF) + 1, run0 = (zm >> 48 & 0xFF) + 1;
      for (uint32_t i = 0; i < subs; ++i) {
        bool v = zm >> (32 + i) & 1;
        const uint32_t sc = zm >> (8 * i) & 0xFF;
        if (sc >= (v ? run1 : run0))
          refuse("zero-column sub-mask " + std::to_string(i) + "'s start count " + std::to_string(sc) +
                 " skips its whole first run; the ISA's examples never do, and do not say what follows");
        uint32_t left = (v ? run1 : run0) - sc;
        for (uint32_t b = 0; b < width; ++b) {
          zero_col[i * width + b] = v;
          if (--left == 0) {
            v = !v;
            left = v ? run1 : run0;
          }
        }
      }
    }
  }
  const uint64_t b_desc_bits = op.b;
  // .ws's collector buffers. Reuse is permission (the tensor core may
  // reload B anyway), so B is read from memory every time; what is checked
  // is that a ::use or ::lastuse follows a fill of the same B that no
  // ::lastuse or ::discard has ended -- otherwise the hardware may
  // multiply by whatever the buffer holds.
  if (op.ws) {
    auto& cb = h.tmem(h.self()).collector_b[op.collector_buf];
    const uint32_t b_fields = (id >> 10 & 7) | (id >> 16 & 1) << 3 | (id >> 17 & 0x3F) << 4;
    const std::string name = ".collector::b" + std::to_string(op.collector_buf);
    switch (op.collector) {
      case Tcgen05Collector::Fill:
        cb = {true, b_desc_bits, b_fields};
        break;
      case Tcgen05Collector::Use:
      case Tcgen05Collector::LastUse:
        if (!cb.valid)
          h.fail(Err::InvalidValue,
                   "tcgen05.mma.ws" + name + (op.collector == Tcgen05Collector::Use ? "::use" : "::lastuse") +
                       " with no fill of that buffer still valid (9.7.18.10.10.3)");
        if (cb.desc != b_desc_bits || cb.b_fields != b_fields)
          h.fail(Err::InvalidValue,
                   "tcgen05.mma.ws" + name + " reuses a buffer filled from a different B (descriptor, "
                   "type, transpose or N); the tensor core may multiply by the filled one");
        if (op.collector == Tcgen05Collector::LastUse) cb.valid = false;
        break;
      case Tcgen05Collector::Discard:
        cb.valid = false;
        break;
    }
  }
  std::vector<double> B(size_t{K} * N);
  {
    const WgmmaDesc d = tcgen05_desc(b_desc_bits, h, sm103a);
    for (uint32_t v = 0; v < G; ++v) {
      const uint32_t rank = v;
      for (uint32_t n = 0; n < Nloc; ++n)
        for (uint32_t k = 0; k < K; ++k)
          B[size_t{k} * N + v * Nloc + n] =
              zero_col[v * Nloc + n]
                  ? 0.0
                  : sb * tc_decode(eb2.t, smem_raw(d, !trans_b, eb2, rank, n + col_shift, k));
    }
  }
  // A, by the Tensor Memory lane each D row is written to: from shared
  // memory, the row that lane holds; from Tensor Memory, whatever that
  // lane holds, in its containers packed 32 bits to a column
  // (9.7.18.10.4) -- so layout B's duplicated A must really be in both
  // halves, as on the hardware.
  const uint32_t a_addr = op.a_tmem ? static_cast<uint32_t>(op.a) : 0;
  if (op.a_tmem && !lane_align_ok(a_addr >> 16))
    h.fail(Err::InvalidValue,
             "tcgen05.mma's A address has lane " + std::to_string(a_addr >> 16) +
                 ", which must match D's data-path lane alignment");
  if (op.a_tmem && layout_f && (a_addr >> 16) != d_lane0)
    h.fail(Err::InvalidValue,
             "for M = 64, A and D must use the same Tensor Memory lane alignment (9.7.18.10.5)");
  const WgmmaDesc a_desc = op.a_tmem ? WgmmaDesc{} : tcgen05_desc(op.a, h, sm103a);
  const bool accumulate = op.accumulate;
  // disable-output-lane: bit l of the vector leaves D's lane l alone.
  const std::array<uint32_t, 8> off_mask = op.disable_lanes;
  if (G == 2)
    for (uint32_t x : off_mask)
      if (x) refuse("a nonzero disable-output-lane with .cta_group::2; the ISA does not say which "
                    "CTA's lanes its upper half covers");
  const double d_scale = op.scale_d > 0 ? std::ldexp(1.0, -op.scale_d) : 1.0;
  // Scale factors (9.7.18.10.7.2-3): row m's j-th for A, in byte SFA_ID + j
  // of the cell at lane m % 32, column (scale-A address) + m / 32; column
  // n's for B likewise. Both are duplicated into every 32-lane partition
  // (CUTLASS's tcgen05.cp .32x128b.warpx4 does it), and each D row reads
  // the copies in its own partition.
  // Bits 30-31 of these addresses are not read: CuTe keeps a factor's byte
  // (the SF_ID it also puts in the instruction descriptor) there in its
  // Tensor Memory pointers and passes them on as they are, which the
  // hardware accepts -- no lane is that high.
  const uint32_t sfa_addr = mx ? op.scale_a & 0x3FFFFFFFu : 0;
  const uint32_t sfb_addr = mx ? op.scale_b & 0x3FFFFFFFu : 0;
  const uint32_t blk = mx ? K / sv : K;
  // At K = 96 a row's factors are one byte stream from byte SF_ID: byte b of
  // it in byte b % 4 of the word `stride` columns on per four bytes -- four
  // columns on, or eight for a B of more than 128 columns, past the columns
  // the rows' first words take (figures 243-275). Below four bytes that is
  // every other size's layout.
  const uint32_t sfb_stride = N > 128 ? 8 : 4;
  auto scale_of = [&](TensorMemory& t, uint32_t addr, uint32_t idx, uint32_t part, uint32_t sfid,
                      uint32_t j, uint32_t stride = 4, bool lanes128 = false) -> double {
    const uint32_t b = sfid + j;
    // The 128-lane layout (sm_107f's, A only): row idx's factors in lane idx, a byte stream from
    // SFA_ID one column on every four bytes, with none of the 32-lane layout's column per 32 rows.
    const uint32_t l = lanes128 ? (addr >> 16) + idx : (addr >> 16) + idx % 32 + 32 * part;
    const uint32_t col = lanes128 ? (addr & 0xFFFF) + b / 4 : (addr & 0xFFFF) + idx / 32 + stride * (b / 4);
    if (l >= TensorMemory::kLanes || col >= TensorMemory::kCols || !t.allocated(col))
      h.fail(Err::OutOfBounds,
               "tcgen05.mma reads a scale factor from Tensor Memory lane " + std::to_string(l) +
                   ", column " + std::to_string(col) + ", which no tcgen05.alloc has allocated");
    const uint32_t byte = (t.at(l, col) >> (8 * (b % 4))) & 0xFF;
    if (ue4m3) return fp8_to_double(byte & 0x7F, kE4M3);
    if (byte == 0xFF) return std::numeric_limits<double>::quiet_NaN();
    return std::ldexp(1.0, int(byte) - 127);   // UE8M0
  };

  // The metadata of the row D lane dl holds: its partition's, at the row
  // that lane's D row has within it.
  const uint32_t meta_addr = sp ? op.sp_meta : 0;
  if (sp && (layout_f ? (meta_addr >> 16) != d_lane0 : (meta_addr >> 16) != 0))
    h.fail(Err::InvalidValue,
             "the sparsity metadata's Tensor Memory lane must match D's data-path lane alignment "
             "(9.7.18.10.9.5)");
  auto meta_of = [&](TensorMemory& t, uint32_t dl, uint32_t c) -> uint32_t {
    const uint32_t r = dl % 32 - (layout_f ? d_lane0 : 0);
    uint32_t lane, col, bit;
    if (meta_rows) {   // figures 291-292: row r in lane r, 16 fields over two columns
      lane = r;
      col = c / 8;
      bit = 4 * (c % 8);
    } else {           // figures 287-290: rows r and r + 8 share a lane, K's upper half 8 lanes on
      lane = 16 * (r / 16) + r % 8 + 8 * (c / 4);
      col = sp_sel;
      bit = 16 * (r % 16 / 8) + 4 * (c % 4);
    }
    const uint32_t l = (meta_addr >> 16) + 32 * (dl / 32) + lane, cc = (meta_addr & 0xFFFF) + col;
    if (l >= TensorMemory::kLanes || cc >= TensorMemory::kCols || !t.allocated(cc))
      h.fail(Err::OutOfBounds,
               "tcgen05.mma.sp reads metadata from Tensor Memory lane " + std::to_string(l) + ", column " +
                   std::to_string(cc) + ", which no tcgen05.alloc has allocated");
    return t.at(l, cc) >> bit & 0xF;
  };

  std::vector<double> A(K), Ap(Ka), SA(8, 1.0), SB(8, 1.0);
  // Without block scaling, D is computed a row at a time: for each k, every
  // column's product is added to its sum (tc_rows_*), so each element still
  // sums its products in k order in f32 -- the same roundings as one
  // element at a time -- while B is read along its rows. B once, in the
  // arithmetic's own type.
  std::vector<float> Bf, Af, accf;
  std::vector<int64_t> Bi, Ai, acci;
  if (!mx) {
    if (d_int) {
      Bi.resize(B.size());
      for (size_t i = 0; i < B.size(); ++i) Bi[i] = static_cast<int64_t>(B[i]);
      Ai.resize(K);
      acci.resize(N);
    } else {
      Bf.resize(B.size());
      for (size_t i = 0; i < B.size(); ++i) Bf[i] = static_cast<float>(B[i]);
      Af.resize(K);
      accf.resize(N);
    }
  }
  for (uint32_t v = 0; v < G; ++v) {
    TensorMemory& t = h.tmem(v);
    const uint32_t rank = v;
    // A's rows of this CTA from shared memory, read once.
    std::vector<double> As;
    if (!op.a_tmem) {
      As.resize(size_t{Mloc} * Ka);
      for (uint32_t m = 0; m < Mloc; ++m)
        for (uint32_t k = 0; k < Ka; ++k)
          As[size_t{m} * Ka + k] = sa * tc_decode(ea.t, smem_raw(a_desc, !trans_a, ea, rank, m, k));
    }
    // Row m of A as D lane dl sees it, into A: its own row from shared
    // memory, or whatever that lane holds in Tensor Memory; expanded by the
    // lane's metadata when sparse.
    auto load_a = [&](uint32_t m, uint32_t dl) {
      std::vector<double>& Arow = sp ? Ap : A;
      if (op.a_tmem) {
        const uint32_t al = dl - d_lane0 + (a_addr >> 16), ac0 = a_addr & 0xFFFF;
        for (uint32_t k = 0; k < Ka; ++k) {
          const uint32_t col = ac0 + k * ea.cbits / 32;
          if (col >= TensorMemory::kCols || !t.allocated(col))
            h.fail(Err::OutOfBounds,
                     "tcgen05.mma reads A from Tensor Memory column " + std::to_string(col) +
                         ", which no tcgen05.alloc has allocated");
          uint32_t raw = t.at(al, col) >> (k * ea.cbits % 32);
          // fp4 in an 8-bit container sits in bits 2-5 (figure 202).
          if (ea.cbits == 8 && ea.bits == 4) raw >>= 2;
          Arow[k] = sa * tc_decode(ea.t, raw);
        }
      } else {
        std::copy_n(As.begin() + size_t{m} * Ka, Ka, Arow.begin());
      }
      if (sp) {
        // 2:4: bits 0-1 and 2-3 place the chunk's two stored elements;
        // 1:2 (tf32): 0b0100 is position 0 and 0b1110 position 1; 4:8
        // (mxf4): the two fields place two-element pairs. A field that
        // places two elements at one position is undefined; both are
        // added here.
        std::fill(A.begin(), A.end(), 0.0);
        const uint32_t per = sp_w / 2;
        for (uint32_t c = 0; c < K / sp_w; ++c) {
          const uint32_t f = meta_of(t, dl, c);
          for (uint32_t s = 0; s < per; ++s) {
            const uint32_t pos = sp_w == 2 ? (f & 3) / 2
                               : sp_w == 4 ? f >> (2 * s) & 3
                                           : 2 * (f >> (2 * (s / 2)) & 3) + s % 2;
            A[c * sp_w + pos] += Ap[c * per + s];
          }
        }
      }
    };
    for (uint32_t m = 0; m < Mloc; ++m) {
      if (!mx) {
        // A D row's columns sit in one lane for each of the layout's
        // parts of N (one part except layouts B, E and G).
        const uint32_t w = N / n_parts;
        for (uint32_t p = 0; p < n_parts; ++p) {
          uint32_t dl, dc0;
          d_pos(m, p * w, d_lane0, &dl, &dc0);
          if (off_mask[dl / 32] >> (dl % 32) & 1) continue;
          load_a(m, dl);
          uint32_t* cells = &t.at(dl, d_col0 + dc0);
          if (d_int) {
            for (uint32_t k = 0; k < K; ++k) Ai[k] = static_cast<int64_t>(A[k]);
            for (uint32_t i = 0; i < w; ++i) acci[i] = accumulate ? static_cast<int32_t>(cells[i]) : 0;
            tc_rows_i64(acci.data(), Ai.data(), Bi.data() + p * w, K, w, N);
            for (uint32_t i = 0; i < w; ++i) {
              int64_t acc = acci[i];
              if (sat)
                acc = std::clamp<int64_t>(acc, std::numeric_limits<int32_t>::min(),
                                          std::numeric_limits<int32_t>::max());
              cells[i] = static_cast<uint32_t>(static_cast<int32_t>(acc));
            }
            continue;
          }
          // As wgmma: every product here is exact in f32, and the sum is
          // kept in f32. An f16 D is one 16-bit value in the low half of
          // its cell (9.7.18.10.4.1).
          for (uint32_t k = 0; k < K; ++k) Af[k] = static_cast<float>(A[k]);
          for (uint32_t i = 0; i < w; ++i) {
            float acc = 0.0f;
            if (accumulate) {
              const double old = d_f16 ? f16_to_double(cells[i] & 0xFFFF) : static_cast<double>(f32(cells[i]));
              acc = static_cast<float>(old * d_scale);
            }
            accf[i] = acc;
          }
          tc_rows_f32(accf.data(), Af.data(), Bf.data() + p * w, K, w, N);
          for (uint32_t i = 0; i < w; ++i)
            cells[i] = d_f16 ? static_cast<uint32_t>(double_to_f16(accf[i]) & 0xFFFF) : f32bits(accf[i]);
        }
        continue;
      }
      // Block-scaled: element by element, each operand multiplied by its
      // block's factor first -- exact too, for these element and scale
      // types. A is loaded again only when the lane changes.
      uint32_t a_lane = UINT32_MAX;
      for (uint32_t n = 0; n < N; ++n) {
        uint32_t dl, dc;
        d_pos(m, n, d_lane0, &dl, &dc);
        if (off_mask[dl / 32] >> (dl % 32) & 1) continue;
        if (dl != a_lane) {
          load_a(m, dl);
          a_lane = dl;
        }
        for (uint32_t j = 0; j < sv; ++j) {
          SA[j] = scale_of(t, sfa_addr, m, dl / 32, sfa_id, j, 4, lane128);
          SB[j] = scale_of(t, sfb_addr, n, dl / 32, sfb_id, j, sfb_stride);
        }
        uint32_t& cell = t.at(dl, d_col0 + dc);
        float acc = 0.0f;
        if (accumulate) {
          const double old = d_f16 ? f16_to_double(cell & 0xFFFF) : static_cast<double>(f32(cell));
          acc = static_cast<float>(old * d_scale);
        }
        for (uint32_t k = 0; k < K; ++k) {
          const double a = A[k] * SA[k / blk];
          const double b = B[size_t{k} * N + n] * SB[k / blk];
          acc += static_cast<float>(a) * static_cast<float>(b);
        }
        cell = d_f16 ? static_cast<uint32_t>(double_to_f16(acc) & 0xFFFF) : f32bits(acc);
      }
    }
  }
}

// Shared with the SASS executor (numerics.hpp).
uint64_t reduce_value(AtomOp op, const Type& ty, uint64_t old, uint64_t b) {
  const uint64_t mask = ty.bits == 64 ? ~0ull : (1ull << ty.bits) - 1;
  old &= mask;
  b &= mask;
  if (ty.is_real()) {
    double x, y;
    if (ty.bits == 16) {
      x = ty.is_bfloat() ? bf16_to_double(old) : f16_to_double(old);
      y = ty.is_bfloat() ? bf16_to_double(b) : f16_to_double(b);
    } else if (ty.bits == 32) {
      x = std::bit_cast<float>(static_cast<uint32_t>(old));
      y = std::bit_cast<float>(static_cast<uint32_t>(b));
    } else {
      x = std::bit_cast<double>(old);
      y = std::bit_cast<double>(b);
    }
    if (op == AtomOp::Min || op == AtomOp::Max) {
      if (std::isnan(x)) return b;
      if (std::isnan(y)) return old;
      // -0 below +0, so the answer does not depend on operand order.
      const bool y_less = y < x || (y == x && std::signbit(y) && !std::signbit(x));
      return (op == AtomOp::Min) == y_less ? b : old;
    }
    // Add, in double and then rounded to the element type. A double has
    // more than twice the precision of a float, a half or a bfloat16, so
    // rounding twice gives the correctly rounded sum (Figueroa, 1995).
    if (ty.bits == 16) return ty.is_bfloat() ? double_to_bf16(x + y) : double_to_f16(x + y);
    if (ty.bits == 32) return std::bit_cast<uint32_t>(static_cast<float>(x + y));
    return std::bit_cast<uint64_t>(x + y);
  }
  const bool s = ty.is_signed();
  auto sx = [&](uint64_t v) { return ty.bits == 64 ? static_cast<int64_t>(v) : int64_t{static_cast<int32_t>(v)}; };
  switch (op) {
    case AtomOp::Add: return (old + b) & mask;
    case AtomOp::Min: return s ? (sx(b) < sx(old) ? b : old) : std::min(old, b);
    case AtomOp::Max: return s ? (sx(b) > sx(old) ? b : old) : std::max(old, b);
    case AtomOp::And: return old & b;
    case AtomOp::Or: return old | b;
    case AtomOp::Xor: return old ^ b;
    case AtomOp::Inc: return old >= b ? 0 : old + 1;
    case AtomOp::Dec: return (old == 0 || old > b) ? b : old - 1;
    default: return b;
  }
}
uint32_t atom_word(AtomOp op, int kind, uint32_t old, uint32_t b) {
  if (kind == 0) {
    const auto ftz = [](float v) { return std::fpclassify(v) == FP_SUBNORMAL ? std::copysign(0.0f, v) : v; };
    const float r = ftz(ftz(std::bit_cast<float>(old)) + ftz(std::bit_cast<float>(b)));
    return std::isnan(r) ? 0x7fffffffu : std::bit_cast<uint32_t>(r);
  }
  const Type ht{kind == 2 ? Type::Kind::BF : Type::Kind::F, 16};
  const uint32_t lo = static_cast<uint32_t>(reduce_value(op, ht, old & 0xffff, b & 0xffff));
  const uint32_t hi = static_cast<uint32_t>(reduce_value(op, ht, old >> 16, b >> 16));
  return (lo & 0xffff) | (hi << 16);
}
std::optional<Type> tensor_reduce_type(exec::TmapType t, AtomOp op) {
  using K = Type::Kind;
  using exec::TmapType;
  Type ty;
  switch (t) {
    case TmapType::U32: ty = {K::U, 32}; break;
    case TmapType::S32: ty = {K::S, 32}; break;
    case TmapType::U64: ty = {K::U, 64}; break;
    case TmapType::S64: ty = {K::S, 64}; break;
    case TmapType::F16: ty = {K::F, 16}; break;
    case TmapType::BF16: ty = {K::BF, 16}; break;
    case TmapType::F32: ty = {K::F, 32}; break;
    default: return std::nullopt;
  }
  bool ok = false;
  switch (op) {
    case AtomOp::Add: ok = !(ty.kind == K::S && ty.bits == 64); break;
    case AtomOp::Min:
    case AtomOp::Max: ok = !(ty.kind == K::F && ty.bits == 32); break;
    case AtomOp::Inc:
    case AtomOp::Dec: ok = ty.kind == K::U && ty.bits == 32; break;
    default: ok = !ty.is_real(); break;
  }
  if (!ok) return std::nullopt;
  return ty;
}

KernelResources kernel_resources(const EntryFn& fn, const DeviceProfile& profile,
                                 uint32_t block_threads, uint32_t dynamic_shared) {
  // The register analysis depends only on the kernel, so memoize it on the
  // kernel itself. Occupancy also depends on the launch shape and is cheap.
  ptx::RegisterUsage usage;
  if (fn.regs_analyzed) {
    usage.regs_per_thread = fn.cached_regs_per_thread;
    usage.pred_regs = fn.cached_pred_regs;
    usage.peak_live = fn.cached_peak_live;
    usage.local_bytes = fn.local_frame_size;
  } else {
    usage = ptx::analyze_registers(fn);
    fn.cached_regs_per_thread = usage.regs_per_thread;
    fn.cached_pred_regs = usage.pred_regs;
    fn.cached_peak_live = usage.peak_live;
    fn.regs_analyzed = true;
  }
  // A thread cannot occupy more architectural registers than the ISA has. When
  // the data flow needs more, ptxas does not give up -- it spills the excess to
  // local memory and the kernel runs, slower. Treating the analysis figure as a
  // hard requirement instead turned a register-hungry but perfectly legal
  // kernel into a launch that could never happen: ggml's flash-attention
  // kernels want 272 by this measure, and refusing them meant the whole
  // attention path was unreachable.
  //
  // The interpreter has no architectural register file to run out of, so this
  // only corrects what is *reported* -- the occupancy figure and the launch
  // decision, which are exactly the things the number exists to inform.
  //
  // The kernel's own bounds lower that ceiling, the same way: .maxnreg names
  // one, and .maxntid (with .minnctapersm) promises that many threads -- that
  // many blocks -- fit, which ptxas meets by allocating no more registers per
  // thread than the block's and the multiprocessor's files allow, in its
  // allocation unit of 8. CUTLASS's Hopper kernels declare .maxntid 384 and
  // need more than 170 by this measure; hardware launches them, spilling.
  uint32_t arch_max = profile.limits.max_registers_per_thread;
  if (fn.max_nreg && (!arch_max || fn.max_nreg < arch_max)) arch_max = fn.max_nreg;
  const uint64_t bound_threads = uint64_t{fn.max_ntid[0]} * std::max(1u, fn.max_ntid[1]) *
                                 std::max(1u, fn.max_ntid[2]);
  if (fn.max_ntid[0] && bound_threads) {
    uint64_t cap = profile.limits.registers_per_block / bound_threads;
    if (profile.limits.registers_per_sm)
      cap = std::min<uint64_t>(cap, profile.limits.registers_per_sm /
                                        (bound_threads * std::max(1u, fn.min_ctas_per_sm)));
    cap = cap / 8 * 8;
    if (cap && (!arch_max || cap < arch_max)) arch_max = static_cast<uint32_t>(cap);
  }
  if (arch_max && usage.regs_per_thread > arch_max) {
    const uint32_t spilled = usage.regs_per_thread - arch_max;
    usage.spilled_regs = spilled;
    usage.regs_per_thread = arch_max;
    usage.local_bytes += spilled * 4u;  // a spilled 32-bit register
  }
  KernelResources r;
  r.usage = usage;
  // NVIDIA's allocation rules, as cuda_occupancy.h gives them and an RTX
  // 3060's cudaOccupancyMaxActiveBlocksPerMultiprocessor applies them
  // (e2e_occupancy_rules): registers per warp in units of 256 from four
  // sub-partitions, shared memory in units of 128 bytes from compute
  // capability 8.0 (256 before) with the driver's reserved shared memory added
  // to every block.
  ptx::OccupancyRules rules;
  if (profile.vendor == "nvidia") {
    rules.reg_alloc_unit = 256;
    rules.sub_partitions = profile.cc_major == 6 && profile.cc_minor == 0 ? 2 : 4;
    rules.regs_per_block = profile.limits.registers_per_block;
    rules.smem_alloc_unit = profile.cc_major >= 8 ? 128 : 256;
    rules.reserved_smem = profile.reserved_smem_per_block();
  }
  r.occupancy = ptx::compute_occupancy(
      usage.regs_per_thread, block_threads, fn.static_shared_size, dynamic_shared,
      profile.limits.registers_per_sm, profile.limits.max_threads_per_sm,
      profile.limits.max_blocks_per_sm, profile.limits.shared_mem_per_sm, profile.warp_size, rules);
  return r;
}

namespace {

void validate(const EntryFn& fn, const LaunchConfig& cfg, const DeviceProfile& p) {
  // 32 for an NVIDIA warp, 64 for a CDNA wavefront. Anything else is not a
  // width this engine has storage for, and rounding it would silently execute
  // a different machine than the profile describes.
  if (p.warp_size != 32u && p.warp_size != kMaxWarpSize)
    throw Error::make(Err::Unsupported, "profile ", p.id, " has warp size ", p.warp_size,
                      "; only 32 and 64 are implemented");
  // ---- thread-block clusters ----
  const bool wants_cluster =
      cfg.cluster[0] > 1 || cfg.cluster[1] > 1 || cfg.cluster[2] > 1;
  if (wants_cluster) {
    // Clusters arrived with Hopper. Running one on an older profile would
    // report a scheduling level that part does not have, which is the kind of
    // wrong answer this whole engine exists to avoid.
    if (p.cc_major < 9)
      throw Error::make(Err::LaunchConfig, "kernel '", fn.name,
                        "': thread-block clusters require compute capability 9.0 or later; ",
                        p.id, " is ", p.cc_major, ".", p.cc_minor);
    // A launch that contradicts the kernel's own __cluster_dims__ fails on
    // hardware rather than being silently overridden, so it fails here.
    if (fn.req_cluster != std::array<uint32_t, 3>{0, 0, 0} && cfg.cluster != fn.req_cluster)
      throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "': launched with cluster ",
                        cfg.cluster[0], "x", cfg.cluster[1], "x", cfg.cluster[2],
                        " but compiled with __cluster_dims__ ", fn.req_cluster[0], "x",
                        fn.req_cluster[1], "x", fn.req_cluster[2]);
    uint64_t ctas = 1;
    for (int i = 0; i < 3; ++i) {
      const uint32_t c = cfg.cluster[i] ? cfg.cluster[i] : 1;
      // The grid is tiled by clusters, so a grid that is not a whole number of
      // them has blocks belonging to no cluster. Hardware rejects this; so
      // does this, rather than inventing a partial cluster.
      if (cfg.grid[i] % c != 0)
        throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "': grid dimension ", i,
                          " is ", cfg.grid[i], ", which is not a multiple of the cluster "
                          "dimension ", c);
      ctas *= c;
    }
    // 8 is the portable maximum CUDA guarantees. A kernel that opts in with
    // cudaFuncAttributeNonPortableClusterSizeAllowed may have up to 16 on
    // Hopper and Blackwell, the most those parts schedule.
    const uint64_t nonportable_max = p.cc_major >= 9 ? 16 : 8;
    if (ctas > 8 && !(cfg.nonportable_cluster && ctas <= nonportable_max))
      throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "': cluster of ", ctas,
                        " blocks exceeds ",
                        cfg.nonportable_cluster
                            ? "the " + std::to_string(nonportable_max) + " this part can schedule"
                            : std::string("the portable maximum of 8 (cudaFuncAttributeNonPortable"
                                          "ClusterSizeAllowed raises it on Hopper and later)"));
  } else if (fn.explicit_cluster) {
    // .explicitcluster means the kernel refuses to run without one.
    throw Error::make(Err::LaunchConfig, "kernel '", fn.name,
                      "': compiled with .explicitcluster and must be launched with a "
                      "cluster dimension");
  }

  uint64_t threads = 1;
  for (int i = 0; i < 3; ++i) {
    if (cfg.block[i] == 0 || cfg.grid[i] == 0)
      throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "': grid/block dimensions must be "
                        ">= 1, got grid ", cfg.grid[0], "x", cfg.grid[1], "x", cfg.grid[2],
                        " block ", cfg.block[0], "x", cfg.block[1], "x", cfg.block[2]);
    if (cfg.block[i] > p.limits.max_block_dim[i])
      throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "': block dim ", i, " is ", cfg.block[i],
                        ", profile ", p.id, " allows at most ", p.limits.max_block_dim[i]);
    if (cfg.grid[i] > p.limits.max_grid_dim[i])
      throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "': grid dim ", i, " is ", cfg.grid[i],
                        ", profile ", p.id, " allows at most ", p.limits.max_grid_dim[i]);
    threads *= cfg.block[i];
  }
  if (threads > p.limits.max_threads_per_block)
    throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "': ", threads,
                      " threads per block exceeds profile limit ", p.limits.max_threads_per_block, " (",
                      p.id, ")");
  // Launch bounds declared by the kernel (__launch_bounds__). Hardware
  // refuses a block larger than the kernel was compiled for.
  for (int i = 0; i < 3; ++i) {
    if (fn.req_ntid[i] && cfg.block[i] != fn.req_ntid[i])
      throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "' requires exactly ",
                        fn.req_ntid[0], "x", fn.req_ntid[1], "x", fn.req_ntid[2],
                        " threads per block (.reqntid), launch asked for ", cfg.block[0], "x",
                        cfg.block[1], "x", cfg.block[2]);
  }
  // .maxntid bounds the *product* of the block dimensions, not each one
  // separately: __launch_bounds__(128) emits ".maxntid 128, 1, 1", and a launch
  // of 32x4x1 is 128 threads, which hardware accepts. Comparing dimension by
  // dimension rejected shapes that are within the bound -- every block whose y
  // extent was greater than 1, which is most of them.
  const uint64_t max_ntid_total = uint64_t{fn.max_ntid[0]} * fn.max_ntid[1] * fn.max_ntid[2];
  const uint64_t block_total = uint64_t{cfg.block[0]} * cfg.block[1] * cfg.block[2];
  if (max_ntid_total && block_total > max_ntid_total)
    throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "' was compiled for at most ",
                      max_ntid_total, " threads per block (__launch_bounds__), launch asked for ",
                      cfg.block[0], "x", cfg.block[1], "x", cfg.block[2], " = ", block_total);

  // Register budget. A block whose threads collectively need more registers
  // than the device provides cannot be launched -- the same
  // "too many resources requested for launch" a real driver reports.
  KernelResources res = kernel_resources(fn, p, static_cast<uint32_t>(threads),
                                         cfg.shared_bytes);
  uint64_t block_regs = uint64_t{res.usage.regs_per_thread} * threads;
  if (block_regs > p.limits.registers_per_block)
    throw Error::make(Err::LaunchConfig, "too many resources requested for launch: kernel '",
                      fn.name, "' uses ", res.usage.regs_per_thread, " registers per thread and ",
                      threads, " threads per block (", block_regs,
                      " registers), but profile ", p.id, " allows ", p.limits.registers_per_block,
                      " per block. Use a smaller block or fewer registers.");
  if (res.occupancy.blocks_per_sm == 0)
    throw Error::make(Err::LaunchConfig, "kernel '", fn.name,
                      "' cannot place a single block on a multiprocessor of ", p.id,
                      " (limited by ", res.occupancy.limited_by, ")");

  uint64_t total_shared = uint64_t{fn.static_shared_size} + cfg.shared_bytes;
  uint32_t shared_limit = std::max(p.limits.shared_mem_per_block, p.limits.shared_mem_per_block_optin);
  if (total_shared > shared_limit)
    throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "' needs ", total_shared,
                      " bytes of shared memory (", fn.static_shared_size, " static + ",
                      cfg.shared_bytes, " dynamic); profile ", p.id, " allows at most ", shared_limit);
}

}  // namespace

namespace {

// Whether a kernel can allocate device memory while it runs: malloc/free, or
// a dynamic-parallelism parameter buffer, in its own code or in a device
// function it calls (an indirect call counts, since where it lands is not
// known here). The allocator's table of live allocations is read without a
// lock by every load and store, so allocating in one block while another
// block's thread reads it is a data race (ThreadSanitizer found it in the
// dynamic-parallelism test). Such a grid runs on one host thread.
bool allocates_while_running(const ptx::EntryFn& fn, std::set<const ptx::EntryFn*>& seen) {
  if (!seen.insert(&fn).second) return false;
  for (const ptx::Instr& ins : fn.body) {
    const auto* op = std::get_if<ptx::OpCall>(&ins.op);
    if (!op) continue;
    if (op->indirect || op->callee == "malloc" || op->callee == "free" || devrt::lookup(op->callee) != devrt::Fn::None)
      return true;
    if (op->target && allocates_while_running(*op->target, seen)) return true;
  }
  return false;
}

// How many host threads to spread the grid over. One thread reproduces the old
// strictly serial block order exactly, which is what a kernel with a data race
// needs to stay reproducible; more threads is faster and is what CUDA's own
// model already allows, since blocks may run in any order and concurrently.
unsigned worker_count(uint64_t blocks) {
  unsigned want = 0;
  if (const char* t = std::getenv("VGPU_THREADS")) {
    const int v = std::atoi(t);
    want = v > 0 ? static_cast<unsigned>(v) : 1;
  } else {
    // The CPUs this process may use, quota included, not the host's count.
    want = vgpu::host_cpus();
  }
  if (want > blocks) want = static_cast<unsigned>(blocks);
  return want ? want : 1;
}

// The step budget is a guard against a kernel that loops forever. A kernel that
// is legitimately long -- a soak or bake kernel that runs for minutes on
// hardware -- trips it too, and the interpreter is slow enough that this is not
// rare. VGPU_MAX_STEPS raises it without a rebuild; 0 turns the guard off, for
// when you know the kernel terminates and only need it to finish.
uint64_t effective_max_steps(uint64_t configured) {
  const char* e = std::getenv("VGPU_MAX_STEPS");
  if (!e || !*e) return configured;
  // strtoull accepts a leading sign and wraps a negative around, so "-5" would
  // parse as an enormous budget and quietly remove the very guard this is
  // protecting. Reject a sign before parsing.
  const char* p = e;
  while (*p == ' ' || *p == '\t') ++p;
  if (*p == '-' || *p == '+') return configured;
  errno = 0;
  char* end = nullptr;
  const unsigned long long v = std::strtoull(e, &end, 10);
  // Anything that is not a whole number leaves the guard as configured rather
  // than silently disabling it.
  if (errno != 0 || end == e || *end != '\0') return configured;
  return v == 0 ? std::numeric_limits<uint64_t>::max() : static_cast<uint64_t>(v);
}

}  // namespace

namespace {

// One grid, without the child grids it launches: those are left in `dl`.
LaunchStats launch_grid(const EntryFn& fn, const LaunchConfig& cfg,
                        const std::vector<std::vector<uint8_t>>& args, MemoryManager& mem,
                        const DeviceProfile& profile, const SymbolTable* symbols,
                        const ProgressFn& progress, DeviceLaunches* dl) {
  LaunchConfig with_cluster = cfg;
  // __cluster_dims__ compiles to .reqnctapercluster and is a property of the
  // kernel, so it applies whether or not the launch asked for a cluster. An
  // explicit cudaLaunchAttributeClusterDimension still wins: validate() then
  // checks the two agree, because on hardware a launch that contradicts
  // .reqnctapercluster fails rather than being quietly overridden.
  if (with_cluster.cluster == std::array<uint32_t, 3>{0, 0, 0} &&
      fn.req_cluster != std::array<uint32_t, 3>{0, 0, 0})
    with_cluster.cluster = fn.req_cluster;
  const LaunchConfig& cfg_ref = with_cluster;
  validate(fn, cfg_ref, profile);
  // One %gridid per launch. Starts at 1 so an unset value reads as "no launch"
  // rather than as the first one.
  static std::atomic<uint64_t> g_next_grid_id{1};
  const uint64_t grid_id = g_next_grid_id.fetch_add(1, std::memory_order_relaxed);
  LaunchConfig eff = cfg_ref;
  eff.max_steps = effective_max_steps(cfg_ref.max_steps);
  // VGPU_SCHEDULER overrides whatever the caller asked for, so a racy program
  // can be re-run under a different order without touching its source. A
  // caller that set the mode explicitly still loses to the environment, which
  // is the right way round: the environment is the person debugging.
  {
    SchedulerKind k = eff.scheduler;
    uint64_t seed = eff.scheduler_seed;
    if (scheduler_from_env(&k, &seed)) {
      eff.scheduler = k;
      eff.scheduler_seed = seed;
    }
  }
  // A non-deterministic order is only useful if it can be replayed, and only
  // reproducible if the blocks are not also being raced across host threads.
  const bool ordered = eff.scheduler != SchedulerKind::Deterministic;

  // A cooperative launch needs the grid-barrier workspace the driver would
  // reserve on hardware: cg::this_grid().sync() finds it through %envreg1 and
  // %envreg2, and the barrier counter it spins on lives a few bytes in. It must
  // start zeroed, because the counter's sign is what the barrier reads.
  //
  // Freed on the way out however that happens -- a kernel that faults must not
  // leak a buffer the caller never asked for.
  struct CoopWorkspace {
    MemoryManager& mem;
    uint64_t addr = 0;
    ~CoopWorkspace() { if (addr) mem.free(addr); }
  } coop{mem};
  if (eff.cooperative) {
    constexpr uint64_t kCoopWorkspaceBytes = 64;
    coop.addr = mem.alloc(kCoopWorkspaceBytes);
    const uint8_t zero = 0;   // not a kernel's store, so no armed fault takes it
    mem.fill(coop.addr, &zero, 1, kCoopWorkspaceBytes);
    eff.coop_workspace = coop.addr;
  }
  ParamBuffer pb = build_params(fn, args);
  const uint64_t blocks = uint64_t{cfg.grid[0]} * cfg.grid[1] * cfg.grid[2];
  // A cooperative launch runs on one worker whatever VGPU_THREADS says. Its
  // blocks wait on each other, so they cannot be split into independent ranges
  // -- that is precisely the promise a cooperative launch does not make.
  const unsigned nthreads = [&] {
    if (eff.cooperative || ordered || blocks <= 1) return 1u;
    std::set<const ptx::EntryFn*> seen;
    if (allocates_while_running(fn, seen)) return 1u;
    // No more threads than the device has SMs for their clusters, so the
    // blocks resident at once can all have distinct %smid values.
    uint64_t csize = 1;
    for (int i = 0; i < 3; ++i) csize *= eff.cluster[i] ? eff.cluster[i] : 1;
    const uint64_t sms = profile.limits.multiprocessors;
    const unsigned cap = sms ? static_cast<unsigned>(std::max<uint64_t>(1, sms / csize)) : 1u;
    return std::min(worker_count(blocks), cap);
  }();

  if (nthreads <= 1) {
    LaunchStats stats;
    Interpreter interp(fn, eff, pb, mem, profile, symbols, stats, progress);
    interp.set_grid_id(grid_id);
    interp.set_device_launches(dl);
    interp.run_grid();
    return stats;
  }

  // Each worker gets its own interpreter and its own statistics; the only thing
  // they share is device memory, whose chunk table is lock-free for exactly
  // this. Progress reporting stays on one worker so the callback is never
  // re-entered.
  std::vector<LaunchStats> per_thread(nthreads);
  std::vector<std::thread> workers;
  std::mutex err_mu;
  std::exception_ptr first_error;
  workers.reserve(nthreads);
  // Divided by clusters when the launch has them, since a cluster's blocks
  // must run together.
  const uint64_t units = [&] {
    uint32_t c[3];
    for (int i = 0; i < 3; ++i) c[i] = eff.cluster[i] ? eff.cluster[i] : 1;
    if (c[0] * c[1] * c[2] <= 1) return blocks;
    return uint64_t{eff.grid[0] / c[0]} * (eff.grid[1] / c[1]) * (eff.grid[2] / c[2]);
  }();
  // Work is handed out in small chunks as threads free up, not split into
  // equal ranges up front: on a host whose cores differ in speed (performance
  // and efficiency cores), equal ranges left the fast cores idle while the
  // slow ones finished theirs. Blocks are independent, so which thread runs
  // one changes nothing it computes.
  std::atomic<uint64_t> next_unit{0};
  const uint64_t chunk = std::max<uint64_t>(1, units / (uint64_t{nthreads} * 8));
  for (unsigned t = 0; t < nthreads; ++t) {
    workers.emplace_back([&, t] {
      try {
        Interpreter interp(fn, eff, pb, mem, profile, symbols, per_thread[t],
                           t == 0 ? progress : ProgressFn{});
        interp.set_concurrent(true);
        interp.set_worker(t);
        interp.set_grid_id(grid_id);
        interp.set_device_launches(dl);
        for (;;) {
          const uint64_t begin = next_unit.fetch_add(chunk, std::memory_order_relaxed);
          if (begin >= units) break;
          interp.run_units(begin, std::min(units, begin + chunk));
        }
      } catch (...) {
        next_unit.store(units, std::memory_order_relaxed);   // the launch has failed; stop
        std::lock_guard<std::mutex> lock(err_mu);
        if (!first_error) first_error = std::current_exception();
      }
    });
  }
  for (auto& w : workers) w.join();
  if (first_error) std::rethrow_exception(first_error);

  LaunchStats total;
  for (const auto& s : per_thread) total.add(s);
  return total;
}

}  // namespace

LaunchStats launch(const EntryFn& fn, const LaunchConfig& cfg,
                   const std::vector<std::vector<uint8_t>>& args, MemoryManager& mem,
                   const DeviceProfile& profile, const SymbolTable* symbols,
                   const ProgressFn& progress) {
  refresh_modes();
  DeviceLaunches dl;
  dl.mem = &mem;
  LaunchStats stats = launch_grid(fn, cfg, args, mem, profile, symbols, progress, &dl);
  if (!dl.queue.empty() || !dl.by_buffer.empty() || dl.extra.blocks) {
    ChildRunner runner{cfg, mem, profile, progress, &stats};
    devrt::complete_children(dl, mem, runner);
  }
  return stats;
}

// ---- textures, for the SASS executor (vgpu/exec/texture.hpp) ----

const TextureDesc& texture_lookup(const TextureTable* table, uint64_t handle, TexKind want) {
  if (!table || table->empty())
    throw Error(Err::InvalidValue,
                "this launch has no texture or surface objects, but the kernel used one. The "
                "handle a kernel receives is created by cudaCreateTextureObject or "
                "cudaCreateSurfaceObject on the host");
  auto it = table->find(handle);
  if (it == table->end())
    throw Error(Err::InvalidValue,
                "texture/surface handle " + std::to_string(handle) + " was never created, or was already destroyed");
  if (it->second.object != want)
    throw Error(Err::InvalidValue, std::string("this is a ") +
                                       (it->second.object == TexKind::Surface ? "surface" : "texture") +
                                       " object, but the instruction is a " +
                                       (want == TexKind::Surface ? "surface" : "texture") +
                                       " access. The two have the same shape of handle and are not "
                                       "interchangeable");
  return it->second;
}

void texture_fetch(const MemoryManager& mem, const TextureDesc& d, const TexFetch& f, uint32_t out[4]) {
  t_bc.valid = false;   // the memory a block was decoded from may have changed since the last fetch
  if (!f.grad) {
    fetch_texel(mem, d, f, out);
    return;
  }
  // tex.grad: the level of detail comes from the gradients (see exec/texture_grad.hpp for what is reproduced).
  TexFetch g = f;
  g.grad = false;
  if (d.max_anisotropy > 1)
    tex_fail(Err::Unsupported,
             "tex.grad on a texture with maxAnisotropy above 1: the card then filters along the major axis of "
             "the gradients' ellipse, which is not reproduced (set maxAnisotropy to 0 or 1)");
  if (d.mip_levels == 0) {   // one level: the gradients choose nothing
    fetch_texel(mem, d, g, out);
    return;
  }
  if (f.cube)
    tex_fail(Err::Unsupported,
             "tex.grad on a cube texture: the card finds the level of detail from the four coordinates of a quad "
             "of normalized directions, which is not reproduced bit for bit");
  if (f.dims == 3) {   // see exec/texture_grad.hpp: the card's unit sees (P + d) - P, scaled by the size
    const uint32_t size[3] = {d.width, d.height, d.depth};
    double dx[3], dy[3];
    for (int i = 0; i < 3; ++i) {
      const float p = std::bit_cast<float>(f.coord[i]);
      dx[i] = std::clamp(tex_grad::scale_by_size(tex_grad::quad_difference(p, std::bit_cast<float>(f.ddx[i])), size[i]),
                         -1e300, 1e300);
      dy[i] = std::clamp(tex_grad::scale_by_size(tex_grad::quad_difference(p, std::bit_cast<float>(f.ddy[i])), size[i]),
                         -1e300, 1e300);
    }
    g.explicit_lod = true;
    g.lod = static_cast<double>(tex_grad::lod_q_3d(dx, dy)) / 256.0;
    fetch_texel(mem, d, g, out);
    return;
  }
  const bool one_d = f.dims == 1;
  // A gradient in texels of the base level, as the card's multiplier gives it (exact for a power-of-two size);
  // an infinite gradient (1e300 here) times a size stays finite.
  auto c = [](uint32_t bits, uint32_t size) {
    return std::clamp(tex_grad::scale_by_size(tex_grad::component(std::bit_cast<float>(bits)), size), -1e300, 1e300);
  };
  const uint32_t w = d.width, h = one_d ? 1u : d.height;
  g.explicit_lod = true;
  g.lod = static_cast<double>(tex_grad::lod_q_2d(c(f.ddx[0], w), one_d ? 0.0 : c(f.ddx[1], h), c(f.ddy[0], w),
                                                 one_d ? 0.0 : c(f.ddy[1], h))) / 256.0;
  fetch_texel(mem, d, g, out);
}

std::optional<uint64_t> surface_address(const TextureDesc& d, const SurfaceAccess& a) { return surface_at(d, a); }

// A f32 rounded toward zero to a half: the top ten mantissa bits, a magnitude past the largest finite
// half the largest finite one (an RTX 3060's sust.p), and a NaN keeping the top ten bits of its payload (quiet
// or not; a payload that truncates to nothing becomes 1, so it stays a NaN).
static uint16_t f32_to_f16_rz(uint32_t b) {
  const uint32_t s = b >> 31, e = (b >> 23) & 0xFF, m = b & 0x7FFFFF;
  if (e == 0xFF) return static_cast<uint16_t>((s << 15) | 0x7C00 | (m ? std::max<uint32_t>(m >> 13, 1u) : 0u));
  const int ex = static_cast<int>(e) - 127;
  if (ex > 15) return static_cast<uint16_t>((s << 15) | 0x7BFF);
  if (ex >= -14) return static_cast<uint16_t>((s << 15) | (static_cast<uint32_t>(ex + 15) << 10) | (m >> 13));
  const int shift = 13 + (-14 - ex);   // a half subnormal: the significand shifted right
  if (e == 0 || shift >= 24) return static_cast<uint16_t>(s << 15);
  return static_cast<uint16_t>((s << 15) | ((0x800000u | m) >> shift));
}

uint32_t surface_pack_texel(const TextureDesc& d, const uint32_t* values, uint32_t n, uint8_t out[16]) {
  uint32_t at = 0;
  for (uint32_t c = 0; c < 4 && d.channel_bits[c]; ++c) {
    const uint32_t bits = d.channel_bits[c], v = c < n ? values[c] : 0;
    uint64_t raw;
    if (d.kind == ChannelKind::Float) {
      raw = bits == 16 ? f32_to_f16_rz(v) : v;
    } else if (d.kind == ChannelKind::Signed) {
      const int64_t sv = static_cast<int32_t>(v), lo = -(int64_t{1} << (bits - 1)), hi = (int64_t{1} << (bits - 1)) - 1;
      raw = static_cast<uint64_t>(std::clamp(sv, lo, hi)) & ((uint64_t{1} << bits) - 1);
    } else {
      raw = std::min<uint64_t>(v, (bits >= 32 ? uint64_t{0xFFFFFFFFull} : (uint64_t{1} << bits) - 1));
    }
    for (uint32_t i = 0; i < bits / 8; ++i) out[at + i] = static_cast<uint8_t>(raw >> (8 * i));
    at += bits / 8;
  }
  return at;
}

// ---- FP8, for the SASS executor (vgpu/exec/numerics.hpp) ----

double fp8_value(uint32_t byte, bool e5m2) { return fp8_to_double(byte & 0xFF, e5m2 ? kE5M2 : kE4M3); }
double mx_float_value(uint32_t code, int eb, int mb, int bias) { return small_float_value(code, eb, mb, bias); }
uint32_t fp8_bits(double v, bool e5m2, bool satfinite) { return double_to_fp8(v, e5m2 ? kE5M2 : kE4M3, satfinite); }
uint32_t small_float_bits(double v, int eb, int mb, int bias) {
  if (std::isnan(v)) return (((1u << eb) - 1) << mb) | ((1u << mb) - 1);   // .satfinite: the positive largest
  return double_to_small_float(v, eb, mb, bias);
}
uint32_t ue8m0_bits(double v, bool round_up, bool satfinite) {
  if (std::isnan(v)) return 0xFF;
  if (v < 0) return ~0u;
  if (v == 0) return 0;
  if (std::isinf(v)) return satfinite ? 0xFE : 0xFF;
  int e = std::ilogb(v);
  if (round_up && std::ldexp(1.0, e) != v) ++e;
  if (e + 127 < 0) return 0;
  if (e + 127 > 254) return satfinite ? 0xFE : 0xFF;
  return static_cast<uint32_t>(e + 127);
}
uint16_t f16_bits(double v) { return static_cast<uint16_t>(double_to_f16(v)); }
uint16_t bf16_bits(double v) { return static_cast<uint16_t>(double_to_bf16(v)); }

float mma_narrow_sum(const double* terms, size_t n, float c) {
  bool nan = std::isnan(c), pinf = std::isinf(c) && c > 0, ninf = std::isinf(c) && c < 0;
  bool all_neg_zero = c == 0.0f && std::signbit(c);
  int max_e = std::numeric_limits<int>::min();
  auto look = [&](double t) {
    if (std::isnan(t)) { nan = true; all_neg_zero = false; return; }
    if (std::isinf(t)) { (t > 0 ? pinf : ninf) = true; all_neg_zero = false; return; }
    if (t != 0) { max_e = std::max(max_e, std::ilogb(t)); all_neg_zero = false; }
    else if (!std::signbit(t)) all_neg_zero = false;
  };
  for (size_t i = 0; i < n; ++i) look(terms[i]);
  if (std::isfinite(c) && c != 0.0f) max_e = std::max(max_e, std::ilogb(static_cast<double>(c)));
  if (nan || (pinf && ninf)) return std::bit_cast<float>(0x7fffffffu);
  if (pinf) return std::numeric_limits<float>::infinity();
  if (ninf) return -std::numeric_limits<float>::infinity();
  if (max_e == std::numeric_limits<int>::min()) return all_neg_zero ? -0.0f : 0.0f;
  const int lsb = max_e - 118;   // the unit of the fixed-point sum: 2^lsb, the largest term in 119 bits of an __int128
  __int128 acc = 0;
  auto add = [&](double t) {
    if (t == 0) return;
    acc += static_cast<__int128>(std::floor(std::ldexp(t, -lsb)));   // below the unit: toward minus infinity
  };
  for (size_t i = 0; i < n; ++i) add(terms[i]);
  add(static_cast<double>(c));
  if (acc == 0) return 0.0f;
  const bool neg = acc < 0;
  unsigned __int128 mag = neg ? -static_cast<unsigned __int128>(acc) : static_cast<unsigned __int128>(acc);
  const uint64_t hi = static_cast<uint64_t>(mag >> 64), lo = static_cast<uint64_t>(mag);
  const int top = hi ? 127 - std::countl_zero(hi) : 63 - std::countl_zero(lo);
  // fp32 keeps 24 bits and nothing below 2^-149
  const int shift = std::max(top - 23, -149 - lsb);
  const uint64_t m = shift >= 0 ? static_cast<uint64_t>(mag >> shift) : static_cast<uint64_t>(mag << -shift);
  if (m == 0) return 0.0f;
  const double v = std::ldexp(static_cast<double>(m), lsb + shift);
  if (v >= 0x1p128) return neg ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
  return neg ? -static_cast<float>(v) : static_cast<float>(v);
}

void validate_launch(const ptx::EntryFn& fn, const LaunchConfig& cfg, const DeviceProfile& profile) {
  validate(fn, cfg, profile);
}

uint32_t mma_sparse_column(uint32_t bits, uint32_t K, uint32_t sel, const uint32_t meta[32], uint32_t row,
                           uint32_t chunk, uint32_t k) {
  Lanes m{};
  for (int i = 0; i < 32; ++i) m[i] = meta[i];
  return sparse_column(bits, K, sel, m, row, chunk, k);
}

WgmmaDesc decode_wgmma_desc(uint64_t d) {
  WgmmaDesc out;
  out.start = (d & 0x3FFF) << 4;
  out.lbo = ((d >> 16) & 0x3FFF) << 4;
  out.sbo = ((d >> 32) & 0x3FFF) << 4;
  static constexpr uint32_t kSwizzle[4] = {0, 128, 64, 32};
  out.swizzle = kSwizzle[(d >> 62) & 3];
  // The base offset says where a swizzle pattern starts when that is not the
  // boundary the pattern repeats on. CUTLASS always leaves it zero and
  // aligns its buffers instead, and the ISA does not say how a nonzero value
  // moves the pattern; guessing would read plausible wrong matrices.
  if (((d >> 49) & 7) != 0)
    throw Error(Err::UnsupportedPtx,
                "a wgmma matrix descriptor with a nonzero base offset (bits 51-49); only swizzle "
                "patterns that start on their repeat boundary are implemented");
  return out;
}

uint64_t wgmma_smem_offset(const WgmmaDesc& d, bool k_major, uint32_t eb, uint32_t mn, uint32_t k) {
  const uint64_t W = d.swizzle;
  uint64_t off;
  if (k_major) {
    const uint64_t kb = uint64_t{k} * eb;
    // Absolute mode (sm_103a): a row's bytes run to the end of its swizzle
    // row, then on at the same row of the atom at `lbo` -- how CUTLASS's
    // SM103 kernels have a K = 96 fp4 block straddle two pipeline buffers.
    if (d.lbo_abs && W && d.start % W + kb >= W)
      return swizzle_address(d.lbo + (mn % 8) * W + (mn / 8) * d.sbo + (d.start % W + kb - W),
                             static_cast<uint32_t>(W), d.atom);
    off = W ? (mn % 8) * W + (mn / 8) * d.sbo + kb
            : (mn % 8) * 16 + (mn / 8) * d.sbo + kb % 16 + (kb / 16) * d.lbo;
  } else {
    const uint64_t mb = uint64_t{mn} * eb;
    off = W ? mb % W + (mb / W) * d.lbo + (k % 8) * W + (k / 8) * d.sbo
            : mb % 16 + (mb / 16) * d.sbo + (k % 8) * 16 + (k / 8) * d.lbo;
  }
  return swizzle_address(d.start + off, static_cast<uint32_t>(W), d.atom);
}

}  // namespace vgpu::exec
