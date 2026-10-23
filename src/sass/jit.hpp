// Tier 2 of the SASS executor: generated C++ for hot straight-line regions of a code section, compiled with the host's compiler
// and loaded with dlopen (docs/perf-design/06-tier2-jit.md; milestone 1 notes: docs/perf-design/round11-jit.md).
//
// VGPU_JIT=1 turns it on (off by default; also off when the tier is off, and in sanitizer builds). Milestone 1 is the skeleton:
// a region is a run of consecutive straight MOps of the tier, each with a handler, none a store; the generated function for it
// counts exactly what Runner::tier_run counts and calls the MOp's own tier handler through `Ctx::escape` for every op, so it is
// deliberately not faster than the tier. What it proves is the machinery: discovery, key, cache with verification, compile with
// a time and memory limit, dlopen, background compile and promotion, exceptions through the generated frames, and a fall-back to
// the tier for every failure.
//
// The Engine (key, source, compile, cache, load) knows nothing about the executor, so the unit tests drive it with synthetic
// regions. The Unit (one per code section, owned by the tier's Program) and the global Manager (queue, worker thread, notices,
// statistics) connect it to Runner::tier_run (tier_ops.inc).
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vgpu::sass::tier {
struct Program;
}

namespace vgpu::sass::jit {

// Bumped when the layout of Ctx, the symbols of an object or the meaning of its metadata change: an object of another ABI is not loaded.
inline constexpr uint32_t kAbiVersion = 2;
// Bumped when the text the generator writes for the same regions changes. (The key also hashes the text itself, so a forgotten bump
// cannot make a stale object load; the number is for people reading a cache directory.)
inline constexpr uint32_t kGeneratorVersion = 2;
inline constexpr uint32_t kMinRegion = 2;
// The run loop of the tier executes at most this many straight instructions in one pass (n < 256), then one more that ends the pass.
inline constexpr unsigned kRunLimit = 256;

// Whether a region of k ops may run in one go in a pass that has run n ops, for a warp that has taken `steps` steps under a budget of
// max_steps, whose file holds nregs registers when the region's handlers touch maxreg of them: what the loop checks op by op (the run of 256,
// the step budget, the register file) holds for every op of the region. When it does not, the ops go through the loop one at a time.
inline bool admissible(unsigned n, uint32_t k, uint64_t steps, uint64_t max_steps, uint32_t maxreg, uint32_t nregs) {
  return n + k <= kRunLimit && steps + k <= max_steps && maxreg <= nregs;
}

// What a generated region sees of the warp and the launch. A plain C layout; the generated source repeats it and static_asserts
// every offset against this one's, so a drift fails the compile instead of corrupting memory.
struct Ctx {
  uint32_t abi;               // kAbiVersion
  uint32_t flags;             // bit 0: the tier's statistics are on (every op is reported, even one with no lane)
  const bool* up;             // Warp::up, the uniform predicates
  uint32_t* p;                // Warp::p, the predicates as lane masks (generated code writes them)
  const uint32_t* ur;         // Warp::ur, the uniform registers
  uint64_t* steps;            // Warp::steps
  uint64_t* instructions;     // LaunchStats::instructions
  uint64_t* thread_instructions;
  uint64_t* by_class;         // LaunchStats::inst_by_class
  void (*escape)(Ctx*, uint32_t op, uint32_t ex);   // op: index inside the region; ex: the lanes that execute it
  void (*note)(Ctx*, uint32_t op);                  // the tier's statistics for an op the generated code ran itself
  void* host;                 // the executor's, opaque to the generated code
  uint32_t* R;                // the warp's register file: register n is R + 32 * n
  const uint8_t* b0;          // constant bank 0 of the launch
  uint64_t b0n;
  const void* pool;
};

using RegionFn = void (*)(Ctx*, uint32_t group, uint32_t glanes);

// A source operand of a generated op, as the tier's MOp holds it (tier.hpp), with a Pool source's row replaced by the constant it holds
// (all 32 lanes are equal): `val`, and for a 64-bit constant pair `val2`, the high word's.
struct SrcFacts {
  uint8_t kind = 2;     // tier::SrcKind: 0 Row, 1 Pool, 2 Dyn
  uint8_t dyn = 0;      // tier::Dyn
  uint8_t mod = 0;
  uint8_t pad = 0;
  uint32_t off = 0;     // Row: register * 32; Dyn: the register, byte offset or uniform register
  uint32_t val = 0, val2 = 0;
};

// What the generator needs of one op: the instruction's 128 bits, the facts the run loop uses to count it, and, when the generated code
// runs the op itself rather than calling the tier's handler, the form and the MOp's fields.
struct OpFacts {
  uint64_t lo = 0, hi = 0;
  uint8_t cls = 0;
  uint8_t guard = 7;           // Instr::guard (7: PT)
  bool guard_not = false;
  bool guard_uniform = false;
  bool unguarded = true;       // MOp::unguarded: the op runs on the whole group
  uint8_t form = 0;            // tier::Form of an op the generated code computes; 0: an escape to the tier's handler
  SrcFacts s[3], p, p2;
  int64_t imm = 0;
  uint32_t a = 0, b = 0, c = 0;
  uint16_t dst = 0xffff, dst2 = 0xffff;
  uint8_t aux = 0;
};
// Whether the generator computes this form itself (a form it does not, or a source it cannot, stays an escape). Called on a facts record
// with `form` set to the MOp's, by make_unit, which then clears `form` if this says no.
bool generates(const OpFacts& o);
// The C++ for a generated op: a block that reads `R` (the register file), `c` (the Ctx) and `ex` (the executing lanes).
std::string gen_op_body(const OpFacts& o);
struct RegionSpec {
  std::vector<OpFacts> ops;
};
struct UnitSpec {
  int sm = 0;
  std::vector<RegionSpec> regions;
};

struct Config {
  std::string cache_dir;            // "": $VGPU_JIT_CACHE, else $XDG_CACHE_HOME/pantheonsim/jit/<build>, else $HOME/.cache/...
  std::string cxx;                  // "": c++, g++, clang++ on PATH
  std::string path;                 // PATH to search; "": the process's
  std::string extra_flags;          // appended to the compile command (and part of the key)
  unsigned timeout_s = 120;         // wall limit of one compile
  unsigned mem_mb = 4096;           // RLIMIT_AS of the compiler
  // For tests: replaces dlopen (returns null and fills the error on a failure), and the ISA level the key and flags use.
  std::function<void*(const char* path, std::string* err)> dlopen_fn;
  std::string isa;                  // "": the running CPU's
  static Config from_env();
};

enum class Status {
  Ok,
  NoCompiler,       // none found, or it cannot run
  CompileFailed,    // the compiler said no (the message has its output)
  CompileTimeout,
  CacheUnusable,    // cannot create or trust the directory, cannot write to it, or the object cannot be mapped (noexec)
  Mismatch,         // the object's metadata is not what was asked for
};
const char* status_name(Status s);
// The checksum the cache keeps beside an object (and the key's hash): 32 hex digits.
std::string checksum(const std::string& bytes);

// A loaded object: one function per region of the unit it was made for. Never unloaded while the process runs (a region may be on a stack).
struct Loaded {
  std::vector<RegionFn> fns;
  void* handle = nullptr;
};

struct Result {
  Status status = Status::Ok;
  std::shared_ptr<Loaded> lib;      // set when status is Ok and `found`
  bool found = false;               // lookup(): false is a plain miss
  bool hit = false;                 // came from the cache
  bool compiled = false;            // this call ran the compiler
  bool replaced_corrupt = false;    // a file of the key was damaged and has been removed
  double compile_ms = 0, lookup_ms = 0;
  std::string key;
  std::string message;
};

class Engine {
 public:
  explicit Engine(Config cfg);
  ~Engine();
  // The C++ the generator writes for the unit (deterministic: a function of the spec only).
  static std::string source(const UnitSpec& spec, bool avx2 = true);
  // The words the object's metadata array must hold for the spec.
  static std::vector<uint32_t> meta(const UnitSpec& spec);
  // The 32-hex-digit key: the source, the compiler's identity, flags, ISA level, ABI and generator versions. Empty (and `why` set) with no compiler.
  std::string key(const UnitSpec& spec, std::string* why = nullptr);
  // The cache only: found, damaged (removed), or absent. Never compiles.
  Result lookup(const UnitSpec& spec);
  // The cache, else the compiler (under a per-key lock file, so that threads and processes compile one key once).
  Result obtain(const UnitSpec& spec);
  // A unit's selected regions: the object holds only those (a compile costs by the size of the source), and a small file beside the objects
  // names the selection of each section, so that a later run finds it without counting. `sel` is region numbers in increasing order.
  static UnitSpec subspec(const UnitSpec& spec, const std::vector<uint32_t>& sel);
  Result lookup_selected(const UnitSpec& full, std::vector<uint32_t>* sel);   // the cache alone, by the section's own key
  Result obtain_selected(const UnitSpec& full, const std::vector<uint32_t>& sel);   // compile (or find) the selection, and record it
  std::string section_key(const UnitSpec& full, std::string* why = nullptr);
  uint64_t compiles() const { return compiles_.load(); }
  void cancel();                                   // kills a compile in progress (process exit)
  const Config& config() const { return cfg_; }
  std::string cache_path() const;                  // the directory in use (created on demand)
  std::string object_path(const std::string& key) const;
  std::string compiler_id(std::string* why = nullptr);
  std::string ensure_rt();                         // writes the embedded headers the generated code includes; "" or the error

 private:
  struct Impl;
  Config cfg_;
  std::unique_ptr<Impl> p_;
  std::atomic<uint64_t> compiles_{0};
  Result load_(const UnitSpec& spec, const std::string& key, bool fresh);
};

// ---- the executor's side ---------------------------------------------------------------------------------------------

struct Region {
  uint32_t first = 0;     // index of its first op in the section
  uint32_t k = 0;         // its length
  uint32_t maxreg = 0;    // one past the highest register any of its ops touches: the warp's file must hold it
  uint32_t gen = 0;       // how many of its ops the generated code computes itself (the others are escapes to the tier's handlers)
};

// Idle: counting only (a section that runs little never asks the cache or the compiler anything); Looking: the cache is being asked; Counting:
// the cache had nothing, counting towards the compile; Queued: a compile is queued or running.
enum class State : int { Idle, Looking, Counting, Queued, Ready, Failed };

// One per code section: the regions found in its tier program, and where the JIT stands for it.
struct Unit : std::enable_shared_from_this<Unit> {
  UnitSpec spec;
  std::vector<Region> regions;
  std::vector<uint32_t> region_at;     // op index -> region number + 1 (0: no region starts there)
  std::atomic<int> state{static_cast<int>(State::Idle)};
  std::atomic<uint64_t> executed{0};   // warp instructions the tier ran in the regions while counting
  std::unique_ptr<std::atomic<uint64_t>[]> counts;   // ... per region: which ones are hot decides which are compiled
  std::shared_ptr<Loaded> lib;         // valid once state is Ready (published with release)
  std::vector<int32_t> fn_index;       // region -> its function in lib, -1 for a region that was not compiled (valid once Ready)
  bool ready() const { return state.load(std::memory_order_acquire) == static_cast<int>(State::Ready); }
  bool counting() const { return state.load(std::memory_order_relaxed) <= static_cast<int>(State::Counting); }
  // The tier ran region `region` (k ops) uncompiled: count it, and ask for the compile once the threshold is passed.
  void note(uint32_t region, uint32_t k);
  // The compiled function of a region, null if it is not compiled (yet, or not selected).
  RegionFn fn(uint32_t region) const {
    if (!ready()) return nullptr;
    const int32_t i = fn_index[region];
    return i < 0 ? nullptr : lib->fns[static_cast<size_t>(i)];
  }
};

// Whether VGPU_JIT=1 is set, the tier is on, and this is not a sanitizer build. Read once.
bool requested();

// Region discovery over the tier's program; null if the unit has none. `max_region` caps a region's length.
std::shared_ptr<Unit> make_unit(const tier::Program& prog, int sm);

// Test and statistics access.
struct Stats {
  std::atomic<uint64_t> units{0}, regions{0};
  std::atomic<uint64_t> cache_hits{0}, compiles{0}, failures{0}, corrupt{0};
  std::atomic<uint64_t> regions_run{0}, instructions_run{0}, generated_run{0}, region_throws{0};   // calls of generated regions, and the warp instructions in them
  std::atomic<uint64_t> compile_us{0}, lookup_us{0}, units_ready{0};
};
Stats& stats();
// Whether the statistics are being kept (VGPU_JIT_STATS): the executor counts nothing otherwise, an atomic add per region call being a visible cost on small regions.
namespace detail { extern bool g_stats_on __attribute__((visibility("hidden"))); }   // libcuda and libcudart both contain this code (tests/lint/check_odr_globals.py)
inline bool stats_on() { return detail::g_stats_on; }
// Waits until the worker has nothing queued (tests).
void drain();

}  // namespace vgpu::sass::jit
