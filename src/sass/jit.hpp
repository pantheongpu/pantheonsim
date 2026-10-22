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
inline constexpr uint32_t kAbiVersion = 1;
// Bumped when the text the generator writes for the same regions changes. (The key also hashes the text itself, so a forgotten bump
// cannot make a stale object load; the number is for people reading a cache directory.)
inline constexpr uint32_t kGeneratorVersion = 1;
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
  uint32_t flags;             // bit 0: the tier's statistics are on (every op is reported to the escape, even one with no lane)
  const bool* up;             // Warp::up, the uniform predicates
  const uint32_t* p;          // Warp::p, the predicates as lane masks
  uint64_t* steps;            // Warp::steps
  uint64_t* instructions;     // LaunchStats::instructions
  uint64_t* thread_instructions;
  uint64_t* by_class;         // LaunchStats::inst_by_class
  void (*escape)(Ctx*, uint32_t op, uint32_t ex);   // op: index inside the region; ex: the lanes that execute it
  void* host;                 // the executor's, opaque to the generated code
  // (reserved for milestone 2: the warp's register file, constant bank 0, the pool)
  uint32_t* R;
  const uint8_t* b0;
  uint64_t b0n;
  const void* pool;
};

using RegionFn = void (*)(Ctx*, uint32_t group, uint32_t glanes);

// What the generator needs of one op: the instruction's 128 bits and the facts the run loop uses to count it.
struct OpFacts {
  uint64_t lo = 0, hi = 0;
  uint8_t cls = 0;
  uint8_t guard = 7;           // Instr::guard (7: PT)
  bool guard_not = false;
  bool guard_uniform = false;
  bool unguarded = true;       // MOp::unguarded: the op runs on the whole group
};
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
  static std::string source(const UnitSpec& spec);
  // The words the object's metadata array must hold for the spec.
  static std::vector<uint32_t> meta(const UnitSpec& spec);
  // The 32-hex-digit key: the source, the compiler's identity, flags, ISA level, ABI and generator versions. Empty (and `why` set) with no compiler.
  std::string key(const UnitSpec& spec, std::string* why = nullptr);
  // The cache only: found, damaged (removed), or absent. Never compiles.
  Result lookup(const UnitSpec& spec);
  // The cache, else the compiler (under a per-key lock file, so that threads and processes compile one key once).
  Result obtain(const UnitSpec& spec);
  uint64_t compiles() const { return compiles_.load(); }
  void cancel();                                   // kills a compile in progress (process exit)
  const Config& config() const { return cfg_; }
  std::string cache_path() const;                  // the directory in use (created on demand)
  std::string object_path(const std::string& key) const;
  std::string compiler_id(std::string* why = nullptr);

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
};

enum class State : int { Looking, Counting, Queued, Ready, Failed };

// One per code section: the regions found in its tier program, and where the JIT stands for it.
struct Unit : std::enable_shared_from_this<Unit> {
  UnitSpec spec;
  std::vector<Region> regions;
  std::vector<uint32_t> region_at;     // op index -> region number + 1 (0: no region starts there)
  std::atomic<int> state{static_cast<int>(State::Looking)};
  std::atomic<uint64_t> executed{0};   // warp instructions the tier ran in the regions while counting
  std::shared_ptr<Loaded> lib;         // valid once state is Ready (published with release)
  bool ready() const { return state.load(std::memory_order_acquire) == static_cast<int>(State::Ready); }
  // The tier ran a region of k ops that was not yet compiled: count it, and ask for the compile once the threshold is passed.
  void note(uint32_t k);
  RegionFn fn(uint32_t region) const { return lib->fns[region]; }
};

// Whether VGPU_JIT=1 is set, the tier is on, and this is not a sanitizer build. Read once.
bool requested();

// Region discovery over the tier's program; null if the unit has none. `max_region` caps a region's length.
std::shared_ptr<Unit> make_unit(const tier::Program& prog, int sm);

// Test and statistics access.
struct Stats {
  std::atomic<uint64_t> units{0}, regions{0};
  std::atomic<uint64_t> cache_hits{0}, compiles{0}, failures{0}, corrupt{0};
  std::atomic<uint64_t> regions_run{0}, instructions_run{0}, region_throws{0};   // calls of generated regions, and the warp instructions in them
  std::atomic<uint64_t> compile_us{0}, lookup_us{0}, units_ready{0};
};
Stats& stats();
// Waits until the worker has nothing queued (tests).
void drain();

}  // namespace vgpu::sass::jit
