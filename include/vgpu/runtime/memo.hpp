// Launch-level result caching for one device: the history chain, the launch key, the write-set capture, the store
// (memory and disk) and the rules that say when a launch cannot be cached. Opt-in (VGPU_MEMO=1); everything here is
// idle, and Device::launch does not reach it, when it is off. vgpu/memo.hpp has the idea; docs/perf-design/rounds/10-rounds-sp19-sp27.md,
// "Round sp35", has the rules and the measurements; docs/cli.md the knobs.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "vgpu/exec/launch.hpp"
#include "vgpu/exec/launch_env.hpp"
#include "vgpu/memo.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/profile.hpp"

namespace vgpu::memo {

// Whether a function a kernel calls is the heap, printf or the device runtime (what makes a launch uncacheable), as against code of
// ptxas's own (`__cuda_sm20_div_rn_f64_full`) that only computes.
bool is_device_runtime_name(const std::string& name);

struct Config {
  bool enabled = false;
  std::string dir;                    // VGPU_MEMO_DIR: entries persist here (empty: in memory only)
  uint64_t disk_cap = 4ull << 30;     // VGPU_MEMO_MAX_MB: least recently used entries go past this
  uint64_t mem_cap = 512ull << 20;    // VGPU_MEMO_MEM_MB: entries held in memory
  uint64_t entry_cap = 256ull << 20;  // the largest write set an entry may hold
  double budget = 0.03;               // VGPU_MEMO_BUDGET: what capturing write sets may cost, as a share of kernel time
  bool force_record = false;          // VGPU_MEMO_BUDGET=inf: capture every miss whatever it costs (measurement, tests)
  int verify = 0;                     // VGPU_MEMO_VERIFY=1: run every hit for real and compare (abort on a difference); 2: count
  uint64_t credit_ns = 150000;        // recording is free until it has cost this much
  uint64_t dormant_after = 256;       // a cache whose launches have not hit once in this many, and whose fixed cost is above the share below, goes quiet
  double dormant_fixed_share = 0.05;  // (of the kernel time; both are for the tests, which make it happen at once)
  uint64_t event_credit_ns = 8000000;    // VGPU_MEMO_EVENT_CREDIT_MS: hashing the bytes of host writes is free until it has cost this much
                                         // (past it, it eats the pool above: a program's uploads before its first launch are not poison)
  std::string mutant;                 // VGPU_MEMO_TEST_MUTANT: a deliberately wrong rule, for the tests that must catch it
  static Config from_env();
};

struct Run {
  uint64_t addr = 0, len = 0;
  bool uniform = false;   // every byte is `byte`
  uint8_t byte = 0;
  std::vector<uint8_t> data;   // empty when uniform
};

// What a hit replays.
struct Entry {
  exec::LaunchStats stats;
  std::vector<Run> runs;
  uint64_t exec_ns = 0;   // what the launch took when it ran
  uint64_t bytes() const {
    uint64_t n = 256 + stats.inst_by_opcode.size() * 8;
    for (const Run& r : runs) n += 32 + r.data.size();
    return n;
  }
};

struct Stats {
  uint64_t launches = 0, hits = 0, disk_hits = 0, recorded = 0, unrecorded = 0, poisoned = 0, dormant_launches = 0;
  uint64_t replay_refused = 0, verify_checked = 0, verify_mismatch = 0, entries_put = 0, entries_oversize = 0;
  uint64_t replay_bytes = 0, entry_bytes = 0, disk_evicted = 0;
  uint64_t replayed_instructions = 0, recorded_instructions = 0;   // warp instructions of the hits' counters / of the launches recorded
  uint64_t exec_ns = 0;      // kernel time of the launches that ran
  uint64_t fixed_ns = 0;     // what the key, the lookup and the checks cost those launches
  uint64_t record_ns = 0;    // what capturing write sets and storing entries cost them
  uint64_t event_ns = 0;     // what hashing the bytes of host writes cost (part of the same budget as record_ns)
  uint64_t events_skipped = 0;   // host writes not hashed because the budget was spent (each poisoned the history instead)
  uint64_t hit_ns = 0;       // what hits cost (lookup and replay)
  uint64_t saved_ns = 0;     // what the hits' launches took when they ran, less hit_ns
  std::map<std::string, uint64_t> reasons;   // why launches were not cached, by rule
};

// What Device::launch tells the cache about a launch.
struct KernelRef {
  Hash128 code;               // the module's content hash
  bool code_ok = false;       // false: the module has no content hash (built without source)
  const std::string* name = nullptr;
  bool sass = false;
  const ptx::EntryFn* fn = nullptr;
  const exec::LaunchConfig* cfg = nullptr;
  const std::vector<std::vector<uint8_t>>* args = nullptr;
  const std::vector<uint64_t>* module_globals = nullptr;   // addresses of the module's global variables
  const exec::LaunchEnv* env = nullptr;
  bool symbols_known = false;
  int device_count = 1;
};

struct Ticket {
  bool active = false;       // end() or abort() must be called
  bool eligible = false;
  bool record = false;
  bool sequential = false;   // VGPU_THREADS=1: blocks run in order
  uint64_t blocks = 0;
  Hash128 key;
  Hash128 desc;                  // the launch alone, without the state before it
  std::shared_ptr<const Entry> verify_against;
  std::vector<uint64_t> bases;   // allocations a launch could write
  uint64_t seq0 = 0;             // host events seen when the launch began
  uint64_t t_begin = 0;          // ns, for the fixed cost
  uint64_t begin_ns = 0;         // what begin() took
  uint64_t t_pre = 0;            // ns spent capturing the before-state
  std::string why;               // why it is not cacheable
};

// The hash of every live allocation of a device, and the end-of-process record of it (VGPU_MEMO_FINAL_HASH).
Hash128 memory_hash(const MemoryManager& mem);
// The part of a launch's key that is the device: every profile value that can change what a kernel computes or whether it may
// launch (identity, compute capability, limits, features, and the numerics). One function, so a test can hold it to that.
Hash128 profile_hash(const DeviceProfile& p);
void final_hash_register(const MemoryManager* m);
void final_hash_write(const MemoryManager* m);   // (the device is going away: write its line now)

class DeviceMemo final : public EventSink {
 public:
  DeviceMemo(MemoryManager& mem, const Config& cfg, const DeviceProfile& profile);
  ~DeviceMemo() override;

  // Before a launch. True: a stored result was applied (memory, `*stats`) and the caller reports it and returns.
  // False: run the kernel, then end() (or abort() if it threw) when the ticket is active.
  bool begin(const KernelRef& k, Ticket& t, exec::LaunchStats* stats);
  void end(Ticket& t, const exec::LaunchStats& stats, uint64_t exec_ns);
  void abort(Ticket& t, const char* why);

  // EventSink
  void on_write(uint64_t dst, const void* src, uint64_t len) override;
  void on_fill(uint64_t dst, const uint8_t* pattern, uint32_t pattern_len, uint64_t len) override;
  void on_copy(uint64_t dst, uint64_t src, uint64_t len) override;
  void on_alloc(uint64_t base, uint64_t size) override;
  void on_free(uint64_t ptr) override;
  void on_poison(const char* why) override;

  // Forget what is known about kernels (a module was unloaded: a pointer may name another kernel next).
  void forget_kernels();
  Stats stats() const;
  Hash128 state() const;
  const Config& config() const { return cfg_; }
  // Writes the summary line (and, with VGPU_MEMO_FINAL_HASH, the hash of all live memory).
  void report(std::FILE* f) const;
  // The hash of every live allocation: what the differential tests compare at the end of a program.
  Hash128 memory_hash() const;
  // Drops everything held in memory (not on disk): a test's way of asking whether the disk alone serves.
  void drop_memory_store();

 private:
  struct PageCache {
    uint64_t size = 0;
    std::vector<uint64_t> pages;
    bool valid = false;
  };
  struct Store;
  bool begin_impl(const KernelRef& k, Ticket& t, exec::LaunchStats* stats);
  const char* ineligible(const KernelRef& k, Ticket& t);
  Hash128 launch_key(const KernelRef& k, const Hash128& state, const Ticket& t);
  void poison_locked(const char* why);
  void advance_locked(const Ticket& t);
  void poison(const char* why);
  bool collect_bases(const KernelRef& k, Ticket& t);
  bool capture_before(Ticket& t, uint64_t* est_bytes);
  uint64_t estimate_ns(const Ticket& t);
  uint64_t estimate_all_ns();
  bool replay(const Entry& e);
  void invalidate_range_locked(uint64_t addr, uint64_t len);
  void scan_pointers(uint64_t addr, const uint8_t* p, uint64_t len);
  bool kernel_has_device_calls(const ptx::EntryFn& fn);
  void note_reason(const char* why);
  Hash128 profile_hash(const DeviceProfile& p) const;
  Hash128 env_hash();
  bool mut(const char* name) const { return !mutant_.empty() && mutant_ == name; }

  MemoryManager& mem_;
  Config cfg_;
  Hash128 profile_;
  Hash128 build_;
  mutable std::mutex mu_;
  Hash128 state_;
  uint64_t poison_seq_ = 0;
  uint64_t event_seq_ = 0;       // host events (writes, fills, copies, allocations, frees)
  bool ptrs_ = true;             // device pointers may be stored in memory (until shown otherwise)
  std::atomic<bool> dormant_{false};
  std::atomic<int> active_launches_{0};
  bool event_budget_allows(uint64_t len);   // false: the write's bytes are not hashed (the caller poisons)
  std::atomic<bool> tainted_{false};
  std::map<uint64_t, PageCache> cache_;
  struct KernelInfo {
    bool calls = false;    // reaches malloc, free, printf or a device-runtime call
    Hash128 name_hash;
  };
  std::unordered_map<const ptx::EntryFn*, KernelInfo> device_calls_;
  std::unordered_map<uint64_t, uint64_t> mat_hint_;   // bytes in real chunks of each allocation, as last seen (for the estimate)
  uint64_t alloc_epoch_ = 0;                          // moves with every allocation, free and poison
  uint64_t all_est_bytes_ = 0, all_est_epoch_ = ~0ull;
  unsigned all_est_age_ = 0;
  Hash128 env_hash_;
  uint64_t env_n_ = 0, env_sig_ = 0;
  bool env_valid_ = false;
  Stats stats_;
  double bw_ = 6.0e9;            // bytes a second hashing runs at, from what it has measured
  std::unique_ptr<Store> store_;
  std::string mutant_;
  uint64_t executed_ = 0;
  mutable std::atomic<bool> reported_{false};
};

}  // namespace vgpu::memo
