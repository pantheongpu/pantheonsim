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
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace vgpu::sass::tier {
struct Program;
}

namespace vgpu::sass::jit {

// Bumped when the layout of Ctx, the symbols of an object or the meaning of its metadata change: an object of another ABI is not loaded.
inline constexpr uint32_t kAbiVersion = 4;
// Bumped when the text the generator writes for the same regions changes. (The key also hashes the text itself, so a forgotten bump
// cannot make a stale object load; the number is for people reading a cache directory.)
inline constexpr uint32_t kGeneratorVersion = 6;
inline constexpr uint32_t kMinRegion = 2;
// The size of tier::MOp (static_assert'ed in jit.cpp): the generated code finds the MOp of op i at Ctx::ops + kMopSize * i, its handler in the first word.
inline constexpr uint32_t kMopSize = 96;
// Ctx::flags
inline constexpr uint32_t kFlagTierStats = 1;   // the tier's statistics are on (every op is reported, even one with no lane)
inline constexpr uint32_t kFlagJitStats = 2;    // the JIT's statistics are on (generated ops are counted into Ctx::gen_run)
inline constexpr uint32_t kFlagChain = 4;       // the run loop would go on past a branch or a register-only op that is not straight (see "Control flow" below)
inline constexpr uint32_t kFlagLink = 8;        // the chain bit is down only because other runnable lanes of the warp stand elsewhere: with Ctx::olo, see "Linking" below
// Ctx::exit_ps when the generated code took no branch or register-only op that resets the loop's 256-instruction run
inline constexpr int32_t kNoPassStart = INT32_MIN;
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
  uint32_t flags;             // kFlag*
  bool* up;                   // Warp::up, the uniform predicates (generated code writes them)
  uint32_t* p;                // Warp::p, the predicates as lane masks (generated code writes them)
  uint32_t* ur;               // Warp::ur, the uniform registers (generated code writes them)
  uint64_t* steps;            // Warp::steps
  uint64_t* instructions;     // LaunchStats::instructions
  uint64_t* thread_instructions;
  uint64_t* by_class;         // LaunchStats::inst_by_class
  void* tctx;                 // the tier handlers' context (the executor's TCtx), what an escape passes to the handler
  const uint8_t* ops;         // the region's first MOp (kMopSize bytes each, the handler first): an escape calls ops[op].fn(tctx, ops[op], ex)
  void (*note)(Ctx*, uint32_t op);                  // the tier's statistics for an op the generated code ran itself
  void* host;                 // the executor's, opaque to the generated code
  uint32_t* R;                // the warp's register file: register n is R + 32 * n
  const uint8_t* b0;          // constant bank 0 of the launch
  uint64_t b0n;
  const void* pool;
  // Control flow (ABI 3): what the run loop's chain looks at and changes when it goes on past a branch.
  uint64_t* divergent;        // LaunchStats::divergent_branches
  bool* gate;                 // Warp::store_gate: up until the warp's first store (the loop ends that warp's turn at it, once)
  uint32_t* budget;           // Warp::budget: the calls the warp's turn has left
  uint32_t* extra;            // Warp::extra: the calls the loop has taken over
  uint64_t* gen_run;          // statistics (kFlagJitStats): the ops the generated code computed
  void (*sreg)(Ctx*, uint32_t sr, uint32_t* d, uint32_t ex);   // S2R: the special register `sr` of the executing lanes into d (the tier's h_s2r)
  uint64_t max_steps;         // the warp's step budget
  // Where the generated code stopped, written once on the way out: the region-relative index of the op the loop runs next (a region may
  // leave through a branch to an op outside it, so it can be negative or past the end), and the region-relative index at which the loop's 256-op
  // run last restarted (kNoPassStart: it did not).
  int32_t exit_idx;
  int32_t exit_ps;
  uint32_t last;              // the index of the op whose handler the generated code called last (a fault names it); the host sets ~0u before a call
  // Linking (ABI 4, round sp52; read only by code generated with Config::link, and only when kFlagLink is up). In a warp whose runnable lanes stand at several
  // addresses the loop ends its pass at every branch and cont op and its chain picks the group at the lowest address. While that is this group again -- the
  // op the group goes to lies strictly below every other runnable lane -- the next pass starts where the generated code is, so it goes on, as it does when the
  // chain bit is up: the call is counted against the turn, the 256-op run starts afresh. `olo` is the lowest address of the other runnable lanes as a
  // region-relative op index (rounded up), `spins` the warp's count of taken backward branches while other lanes can run (the loop's spin rule: the 64th makes
  // the group give way, so the generated code leaves before it).
  int64_t olo;
  uint32_t* spins;
  // A BSSY inside a region (OpFacts::hostctl): the lanes of `ex` join the barrier of op `op` and the barrier's nesting stamp is renewed (the run loop's own lines).
  void (*ctl)(Ctx*, uint32_t op, uint32_t ex);
  // A branch that splits the group, run by the generated code (linking): the lanes of `take` stand at the region-relative op `tgt` from now on, the others of
  // `group` at `fall` (the loop's select_pcs). The generated code then goes on with the half at the lower place, and says so: `group` (the host sets it to the
  // group it calls with; it differs afterwards when a split was run), `pcs_at` (the op the group's lanes' addresses were last set to) and `olo` (above).
  void (*split)(Ctx*, uint32_t group, uint32_t take, int32_t tgt, int32_t fall);
  uint32_t group;
  int32_t pcs_at;
  // An EXIT inside a region that no lane of the group takes (OpFacts::hostctl 2): whether the run loop would run it itself (no lane of the warp waits, no block
  // barrier has an arrival; else it is the interpreter's and the generated code stops before it), and then the rest of what the loop does for it.
  uint32_t (*exit0)(Ctx*);
  // A CALL.REL inside a region (OpFacts::call): whether the place op `op` calls is plain code of the section (not a function the simulator provides, which the
  // interpreter runs). A RET inside a region (hostctl 3): the region-relative op every lane of `group` returns to, or INT64_MIN when the guard lets only
  // some of them run it (ex != group), they return to different places, or the place is not an op of the section (all of these are the loop's).
  uint32_t (*callok)(Ctx*, uint32_t op);
  int64_t (*ret)(Ctx*, uint32_t op, uint32_t ex, uint32_t group);
};

// A region is entered at any of its stretch starts (stretch_starts) with the number of ops the loop's current pass has taken since its start (n0);
// it runs until it must leave, and says where in exit_idx and exit_ps.
using RegionFn = void (*)(Ctx*, uint32_t group, uint32_t glanes, uint32_t entry, uint32_t n0);


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
  bool cont = false;           // MOp::cont: not straight, a handler that only writes registers: the run loop may go on past it (a stretch ends here)
  uint8_t hostctl = 0;         // (with cont; linking only) an op without a handler that the generated code goes on past. 1, a BSSY: the host's Ctx::ctl does its
                               // bookkeeping, whatever the guard says. 2, an EXIT (aux, p, p2: its conditions): gone past only when no lane leaves and
                               // the host's Ctx::exit0 says the loop itself would run it; else the generated code stops before it. 3, a RET (dst: the
                               // register pair that holds the address): gone past when every lane of the group returns to one place in the section
                               // (the host's Ctx::ret reads it), which the generated code then looks up among the region's entries
  bool call = false;           // (with br; linking only) a CALL.REL to a fixed place in the section: a branch every lane its guard lets run takes, without
                               // the spin rule and the divergent-branch count of a BRA, and only when the place is no function the simulator provides (Ctx::callok)
  bool stop = false;           // (merged regions; linking only) not an op of the region at all: an op between two of the regions that were merged, which the
                               // run loop runs. The generated code leaves before it; it is no entry, no branch target and belongs to no stretch
  bool store = false;          // MOp::store: a store to global memory (an escape): while the warp's store gate is up the generated code leaves before it
  bool br = false;             // a plain relative BRA (tier::Form::Bra) that the generated code runs itself: imm is its target in the section, tgt in the region, p/p2/aux its conditions
  int32_t tgt = 0;             // a branch: the region-relative index of its target (outside [0, k): the generated code leaves there)
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

// Register residence (milestone 3). Within a region the generated code keeps the register-file rows it touches in local arrays (and
// predicates in locals): a row is loaded at its first read (or, for a write that may not cover every lane, at its first write, since the
// lanes outside the executing mask keep their old value), kept, and stored back before every escape (the tier's handler reads and writes the
// real register file) and at the region's end; after an escape no row is trusted. RowCtx is the generator's account of that, per region.
struct RowCtx {
  bool enabled = true;                  // false: every access goes straight to the register file (milestone 2's code)
  unsigned cap = 32;                    // at most this many rows get an array; others are accessed in place
  std::map<unsigned, uint8_t> st;       // rows with a valid local copy: 1 clean, 2 dirty
  std::set<unsigned> decl;              // rows that have an array
  uint8_t pst[8] = {};                  // predicates P0..P6 likewise (0: not loaded)
  std::set<unsigned> pdecl;
  std::set<unsigned> reads;             // rows the op being generated reads (for the mutation checks)
  std::string pre;                      // loads the op being generated needs; emitted before its body, outside the guard
  std::string row(unsigned reg, bool write);                 // a pointer expression for the row
  std::string pred(unsigned reg, bool neg);                  // the lane mask of a predicate register
  std::string setp(unsigned reg, const std::string& value);  // the statement that writes the executing lanes of a predicate
  std::string flush_pre();                                    // takes `pre`
  void sync(std::string& out, bool invalidate);              // stores every dirty row and predicate; invalidate: forget all copies (an escape follows)
  std::string decls() const;
};

// The C++ for a generated op: a block that reads `R` (the register file), `c` (the Ctx) and `ex` (the executing lanes).
std::string gen_op_body(const OpFacts& o, RowCtx& rc);
// Fused runs (round sp48, docs/perf-design/rounds/11-rounds-sp36-sp41.md). A run of consecutive ops that only read and write register rows, each for the
// whole group and of a form whose result in a lane depends on that lane alone, is generated as one loop over the warp's four blocks of eight lanes
// with every op of the run inside it: a row the run writes is a local vector until the block's end, so the value an op reads from an earlier op of the
// run comes from a host register (the op-by-op form stores every result's 32 lanes and loads them again). Each row is read from the register file once,
// at its first use, and each row written is stored once, at the block's end; the result, lane for lane, is what the ops one after the other give,
// because no op of the run reads a lane other than its own. `fusable` says whether an op may be in a run; `gen_fused_run` writes the loop for `n`
// such ops (`full`: for a whole warp, where a result is the row; otherwise the lanes outside `ex` keep the register's value). The text reads `R`,
// `c` and `ex`. Only the AVX2 level and only without register residence (Engine::source).
bool fusable(const OpFacts& o);
std::string gen_fused_run(const OpFacts* ops, uint32_t n, bool full);
struct RegionSpec {
  std::vector<OpFacts> ops;
  uint32_t first = 0;          // the section's op its first op is (make_unit sets it; merging needs it; not part of the metadata)
};
// Control flow inside a region (milestone 5). A region is a run of consecutive ops that may hold uniform branches (every lane of the group takes it, or none),
// ops that are not straight but only write registers (`cont`), and stores. It is cut into *stretches*: a stretch starts at the region's first op, at the target
// of a branch inside the region, and at the op after a branch or a cont op, and it ends at the next branch or cont op (which belongs to it) or at the region's
// end. Each stretch is what one pass of the run loop does between two of its chain points: the 256-op run bound, the step budget and the step count are checked
// for the whole stretch on its way in. The generated code runs stretch after stretch, as the loop does when it goes on past a branch (tier_ops.inc, the "cont"
// path), and leaves -- with nothing of the next op done -- at the first op the loop would not go on past: a branch that splits the group, a branch or cont op
// when the loop's chain conditions do not hold or the turn has no calls left (Ctx::budget), the start of a stretch that does not fit the run bound or the step
// budget, or an op outside the region.
// The stretch starts of a region, in increasing order (the host marks each as an entry; the generator labels each).
std::vector<uint32_t> stretch_starts(const RegionSpec& r);
// One past the last op of the stretch that starts at `start`, and how many of its ops are straight (the ones the loop's 256-op run bound counts).
uint32_t stretch_end(const RegionSpec& r, uint32_t start);
inline bool is_ctl(const OpFacts& o) { return o.cont || o.br; }
struct UnitSpec {
  int sm = 0;
  std::vector<RegionSpec> regions;
  bool nolink = false;         // (linking; the loop rule of make_unit) this section is generated without linking whatever Config::link says: the text, the object and
                               // the selection are those of a run without VGPU_JIT_LINK. Not part of the metadata
};

struct Config {
  std::string cache_dir;            // "": $VGPU_JIT_CACHE, else $XDG_CACHE_HOME/pantheonsim/jit/<build>, else $HOME/.cache/...
  std::string cxx;                  // "": c++, g++, clang++ on PATH
  std::string path;                 // PATH to search; "": the process's
  std::string extra_flags;          // appended to the compile command (and part of the key)
  bool residence = true;            // keep register-file rows in locals within a region (from_env: off, VGPU_JIT_RESIDENCE=1: on)
  bool link = false;                // go on past a branch or cont op in a diverged warp while the chain would pick the group again (Ctx::olo; from_env: VGPU_JIT_LINK=1 turns it on)
  bool specialize = true;           // also generate the body for a whole warp (all 32 lanes) apart from the one for any group (from_env: off, VGPU_JIT_SPECIALIZE=1: on; off is only the general one: half the code, a faster compile)
  bool fuse = false;                // generate runs of register-only ops block by block of eight lanes (gen_fused_run; from_env: VGPU_JIT_FUSE=1 turns it on)
  unsigned timeout_s = 120;         // wall limit of one compile
  bool pch = true;                  // GCC: the runtime header precompiled once in the runtime directory, read by every compile after (round sp64; from_env: VGPU_JIT_PCH=0 off)
  unsigned mem_mb = 4096;           // RLIMIT_AS of the compiler
  uint64_t max_bytes = uint64_t{256} << 20;   // the cache's size cap over its directories (Engine::trim; from_env: VGPU_JIT_CACHE_MAX_MB, 0: none)
  unsigned stale_days = 14;         // another build's directory none of whose files changed for this long is removed (0: never)
  bool trim_siblings = false;       // (tests) trim the cache directory's siblings as with the default directory
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
  bool pending = false;             // lookup_selected(): a miss of a section that has a selection on record (a compile of an earlier run that did not finish)
  bool cancelled = false;           // obtain(): the compile was cancelled (the process is ending)
  double compile_ms = 0, lookup_ms = 0;
  uint64_t source_bytes = 0;        // obtain(): the size of the generated source it compiled (the compile log of VGPU_JIT_STATS, round sp64)
  bool pch = false;                 // obtain(): the compile read the precompiled runtime header (round sp64)
  std::string key;
  std::string message;
};

class Engine {
 public:
  explicit Engine(Config cfg);
  ~Engine();
  // The C++ the generator writes for the unit (deterministic: a function of the spec only).
  static std::string source(const UnitSpec& spec, bool avx2 = true, bool residence = true, bool specialize = true, bool link = false, bool fuse = false);
  // The words the object's metadata array must hold for the spec.
  static std::vector<uint32_t> meta(const UnitSpec& spec);
  // The 32-hex-digit key: the source, the compiler's identity, flags, ISA level, ABI and generator versions. Empty (and `why` set) with no compiler.
  std::string key(const UnitSpec& spec, std::string* why = nullptr);
  // Merging (linking). The selected regions of a section as ONE region that runs from the first selected region's first op to the last one's last: the ops
  // between them are `stop` ops, branch targets are rebased. One function then holds every compiled op of the section, so a branch, a call or a return
  // from one of the old regions into another is a jump inside generated code. merges(): whether a selection is merged (linking on, and the span is not huge).
  static RegionSpec merge_regions(const UnitSpec& spec, const std::vector<uint32_t>& sel);
  bool merges(const UnitSpec& spec, const std::vector<uint32_t>& sel) const;
  UnitSpec selected_spec(const UnitSpec& full, const std::vector<uint32_t>& sel) const;   // subspec, or the one merged region
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
  // (Round sp64) Makes the precompiled runtime header that the last compile found missing, once an engine; true if it made one.
  bool build_pch();
  void trim(const std::string& keep);              // the size cap and the age rule (round sp63; run after an install): `keep` is a key never removed

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
  bool has_store = false; // it holds a store to global memory (the generated code leaves before it while the warp's store gate is up)
  bool ctl = false;       // it holds a branch or a cont op
  std::vector<uint32_t> starts;   // its stretch starts, region-relative (stretch_starts)
  uint32_t maxreg = 0;    // one past the highest register any of its ops touches: the warp's file must hold it
  uint32_t gen = 0;       // how many of its ops the generated code computes itself (the others are escapes to the tier's handlers)
};

// Idle: counting only (a section that runs little never asks the cache or the compiler anything); Looking: the cache is being asked; Counting:
// the cache had nothing, counting towards the compile; Queued: a compile is queued or running.
// Failed: nothing to compile, or the compile failed. Declined: the sample found the section's regions too small a share of its work to be worth the
// hook (the share rule below). Both go back to the plain run loop for good.
// Waiting (round sp57): the cache had nothing and the threshold is far: the section runs on the plain loop, which costs nothing, and only the warp
// instructions of each of its calls are added up (Unit::wait_note); `profile` instructions before the threshold it is Counting again, with the hook, so
// that the regions' counts of the compile are fresh. (Until that round a section counted in the hook loop all the way: 2 to 12 % of a program that
// never reaches the threshold.)
enum class State : int { Idle, Looking, Counting, Queued, Ready, Failed, Declined, Waiting };
inline constexpr uint64_t kWaitBatch = 65536;        // warp instructions a thread adds up by itself while a section waits
inline constexpr uint64_t kWaitBatchFrom = 1 << 20;  // ... when the wait is at least this long
// (round sp63) A section the share rule has kept waits on the plain loop for its lookup too (Unit::sample), when the lookup is at least this many instructions away;
// that wait ends in Idle (Unit::wait_then), the hook loop's counting up to the lookup.
inline constexpr uint64_t kPrewaitMin = 4096;

// The share rule (round 14). The hook (tier_run_t<true>, the counting, the cache lookup) costs the instructions of a section that are not in a compiled
// region something, and a section whose regions are a small part of its work never earns it back (`warp`, `shared`, `f64_fma`, `divergent`, `texture`
// lost 5% to 19% against the tier on 0.1 to 0.4 s runs). So a section is *sampled*: once it has run `sample_after` warp instructions in passes of the hook
// loop, the share of them that generated code would compute (the entries of its regions, each worth the region's `gen` ops) is compared with
// `min_share_permille`, once; below it the section is Declined and runs on the plain loop. Nothing asks the cache or the compiler before the sample is in.
// A sample is a decision about a rule's inputs only (no timing), so it is the same on every run of the same program.
struct ShareRule {
  bool on = false;                    // VGPU_JIT_MIN_SHARE=0, or VGPU_JIT_HOT_PERMILLE=0 ("compile every region", the differential tests): off
  unsigned min_share_permille = 0;    // generated-code share of the sample, in thousandths, below which the section is declined
  uint64_t sample_after = 0;          // warp instructions of the section to sample
};
// Whether `gen_work` of `total` instructions (both counted the same way) is enough: gen_work * 1000 >= total * min_share_permille. Off, or nothing sampled: true.
inline bool share_ok(uint64_t gen_work, uint64_t total, const ShareRule& r) {
  return !r.on || total == 0 || gen_work * 1000 >= total * r.min_share_permille;
}
const ShareRule& share_rule();

// One per code section: the regions found in its tier program, and where the JIT stands for it.
struct Unit : std::enable_shared_from_this<Unit> {
  UnitSpec spec;
  std::vector<Region> regions;
  std::vector<uint32_t> region_at;     // op index -> region number + 1 at each stretch start of a region (0: no entry there): what counting uses
  std::vector<uint16_t> entry_ops;     // op index -> the ops of the stretch that starts there (the weight of an entry while counting)
  std::vector<uint8_t> jrun;           // op index -> the length of the plain run (MOp::run) that stops short of the next entry: what the loop issues back to back
  // What the run loop reads once a section is Ready: the entries of the regions that were compiled only (a region that was not selected costs no call), and
  // the function of each region. Published before `state` says Ready; the pointer the loop reads is region_at's until then.
  std::vector<uint32_t> active_at;
  std::vector<RegionFn> fast;
  std::atomic<const uint32_t*> active{nullptr};
  std::atomic<int> state{static_cast<int>(State::Idle)};
  std::atomic<uint64_t> executed{0};   // warp instructions the tier ran in the regions while counting
  std::atomic<bool> sampling{false};   // the share rule has not decided yet (set by make_unit when the rule is on)
  std::atomic<uint64_t> sampled{0};    // warp instructions run in the hook loop while sampling
  std::unique_ptr<std::atomic<uint64_t>[]> counts;   // ... per region: which ones are hot decides which are compiled
  std::atomic<uint64_t> waited{0};     // warp instructions of the section run on the plain loop while Waiting (all of them, in regions or not)
  std::atomic<uint64_t> wait_for{0};   // ... and how many of them end the wait
  std::atomic<uint64_t> wait_batch{1}; // ... and how many a thread adds up before it brings them to `waited` (set with wait_for)
  std::atomic<uint64_t> credit{0};     // what the waits count for towards the lookup and the threshold once over (their wait_for; Unit::note adds it to the regions' counts)
  std::atomic<int> wait_then{static_cast<int>(State::Counting)};   // the state a wait ends in: Counting (after a miss of the lookup), Idle (a wait before the lookup)
  // Merging (linking): `regions` has one more element than spec.regions, a placeholder until a merged selection is published; then it is the merged region
  // (its spec is `merged`), the only one with entries in active_at and a function.
  RegionSpec merged;
  std::shared_ptr<Loaded> lib;         // valid once state is Ready (published with release)
  std::vector<int32_t> fn_index;       // region -> its function in lib, -1 for a region that was not compiled (valid once Ready)
  bool ready() const { return state.load(std::memory_order_acquire) == static_cast<int>(State::Ready); }
  bool counting() const { return state.load(std::memory_order_relaxed) <= static_cast<int>(State::Counting); }
  // Whether the run loop with the region hook is wanted: not once the section has failed or been declined (the plain loop is then used: step_warp asks),
  // and not while its compile runs (round sp64: Queued counts nothing, and its regions have no functions yet; the hook loop cost the section 3 to 8 % for
  // as long as the compile took, a whole run when the compile started at the lookup and did not finish before the end).
  bool hook() const {
    const int s = state.load(std::memory_order_relaxed);
    return s <= static_cast<int>(State::Counting) || s == static_cast<int>(State::Ready);
  }
  // Waiting: the plain loop is used, and step_warp tells wait_note what each call of it ran.
  bool waiting() const { return state.load(std::memory_order_relaxed) == static_cast<int>(State::Waiting); }
  // The plain loop ran `instr` warp instructions of this section while it was Waiting. Each thread adds them up by itself and brings them to the shared
  // count a batch at a time (kWaitBatch instructions when the wait is a long one, so that sixteen threads running short calls do not pass the count's cache
  // line around: that cost a 16-thread run up to 13 %; at once when the wait is short, as in the tests). The thread whose batch completes the wait makes
  // the section Counting again, up to a batch a thread late.
  void wait_note(uint64_t instr);
  void wait_over();
  // The hook loop ran `instr` warp instructions of this section (the whole pass, regions or not). The one that brings the sample to `sample_after` decides.
  void sample(uint64_t instr);
  // The work generated code would do for what the regions ran while counting: sum over regions of entries * gen, entries = counts / k.
  uint64_t gen_work() const;
  // The tier ran region `region` (k ops) uncompiled: count it, and ask for the compile once the threshold is passed.
  void note(uint32_t region, uint32_t k);
  // The compiled function of a region, null if it is not compiled (yet, or not selected).
  RegionFn fn(uint32_t region) const { return ready() ? fast[region] : nullptr; }
};

// Mutation checks of the executor's side (tests only: compiled in with -DVGPU_JIT_MUTANTS=ON, VGPU_JIT_MUTANT=<host_ps|host_chain|host_maxreg|host_n0|host_olo|host_spins>); 0 otherwise.
inline constexpr uint32_t kMutHostPs = 2, kMutHostChain = 4, kMutHostMaxreg = 8, kMutHostN0 = 16;
inline constexpr uint32_t kMutHostOlo = 32, kMutHostSpins = 64;   // host_olo: the other lanes are taken to stand above everything; host_spins: the generated code counts its taken backward branches somewhere else
uint32_t host_mutants();

// Whether VGPU_JIT=1 (or auto) is set, the tier is on, and this is not a sanitizer build. Read once.
bool requested();
// Whether VGPU_JIT=auto is set (round sp63): on as with 1, the way a default-on JIT would run, without the notices that only say it cannot run here
// (no C++ compiler, a cache directory that cannot be used). Read once.
bool auto_mode();
// Whether VGPU_JIT_LINK=1 is set (linking, see Ctx::olo): the host then works out kFlagLink and Ctx::olo for a region call in a diverged warp. Read once.
bool link_requested();

// Region discovery over the tier's program; null if the unit has none. `max_region` caps a region's length.
std::shared_ptr<Unit> make_unit(const tier::Program& prog, int sm);
// Gives `prog` its unit when the JIT is on (VGPU_JIT=1); otherwise does nothing.
void attach(tier::Program& prog, int sm);

// Test and statistics access.
struct Stats {
  std::atomic<uint64_t> units{0}, regions{0};
  std::atomic<uint64_t> cache_hits{0}, compiles{0}, failures{0}, corrupt{0};
  std::atomic<uint64_t> regions_run{0}, instructions_run{0}, generated_run{0}, region_throws{0};   // calls of generated regions, and the warp instructions in them
  std::atomic<uint64_t> compile_us{0}, lookup_us{0}, units_ready{0};
  std::atomic<uint64_t> declined{0}, accepted{0};   // the share rule's decisions
  std::atomic<uint64_t> waits{0}, wait_ends{0};     // sections that went to the plain loop on the way to the threshold (State::Waiting), and waits that ended (either kind)
  std::atomic<uint64_t> prewaits{0};                // sections that went to the plain loop on the way to the lookup (round sp63)
  std::atomic<uint64_t> history{0};                 // lookups that found a selection without its object and compiled it at once (round sp63)
  std::atomic<uint64_t> evicted{0};                 // objects and directories the cache's size cap and age rule removed (Engine::trim)
  std::atomic<uint64_t> pch_used{0}, pch_built{0};  // compiles that read the precompiled runtime header, and headers made (round sp64)
  std::atomic<uint64_t> continued{0};   // branches and register-only ops the generated code went on past (the loop's chain steps it took over)
  // Why generated code was left (round sp52), per reason (ExitWhy): [0] the group was every runnable lane of the warp, [1] other runnable lanes stood elsewhere
  // (a diverged warp: the loop's chain conditions do not hold), [2] the calls of these that left at their entry with nothing run.
  std::atomic<uint64_t> exits[24][3] = {};
  std::atomic<uint64_t> splits{0};      // region calls that ran a branch that split the group and went on with one half (linking)
  std::atomic<uint64_t> linked{0};      // ... of `continued`, those of region calls in a diverged warp (linking, Ctx::olo)
  std::atomic<uint64_t> link_declined{0};   // sections the loop rule left unlinked (make_unit)
};
// Why a region call ended, as the host works it out from where it ended (jit_exit_why, tier_ops.inc); kept only with VGPU_JIT_STATS.
enum ExitWhy : unsigned {
  kExSplit,          // a branch that splits the group (some lanes take it, some do not)
  kExBraTaken,       // a branch the whole group takes, when the loop would not go on past it (the chain's conditions do not hold)
  kExBraFall,        // ... that no lane of the group takes
  kExCont,           // a register-only op that is not straight, likewise
  kExBudget,         // a branch or such an op when the turn has no calls left
  kExBound,          // the start of a stretch that does not fit the loop's 256-op run
  kExSteps,          // ... or the warp's step budget
  kExGate,           // a store while the warp's store gate is up
  kExToRegion,       // left the region (its end, or a branch out of it) at an entry of a compiled region
  kExBssy,           // ... at a BSSY or a NOP
  kExBsync,          // ... at a BSYNC
  kExExit,           // ... at an EXIT that lanes of the group take
  kExExitNone,       // ... at an EXIT that no lane of the group takes (a guard or a condition holds for none: the group goes on)
  kExCallRet,        // ... at a CALL or a RET
  kExOtherCtl,       // ... at a BAR, YIELD or WARPSYNC
  kExBraOther,       // ... at a branch that is in no compiled region
  kExStraight,       // ... at a straight op that is in no compiled region
  kExHandler,        // ... at an op with a handler that is not straight (SHFL, VOTE, ...)
  kExInterp,         // ... at an op the interpreter runs
  kExCount
};
const char* exit_name(unsigned why);
Stats& stats();
// Whether the statistics are being kept (VGPU_JIT_STATS): the executor counts nothing otherwise, an atomic add per region call being a visible cost on small regions.
namespace detail { extern bool g_stats_on __attribute__((visibility("hidden"))); }   // libcuda and libcudart both contain this code (tests/lint/check_odr_globals.py)
inline bool stats_on() { return detail::g_stats_on; }
// Waits until the worker has nothing queued (tests).
void drain();

}  // namespace vgpu::sass::jit
