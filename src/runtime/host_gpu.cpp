// Host-GPU mode, `exact` (docs/perf-design/09-host-gpu-spike.md): a launch whose kernel passes the eligibility scan and whose
// size rule says the card path wins runs on the machine's real GPU, on the simulator's own device memory.
//
// How, in this first form (write-through, no state between launches): the simulator's device memory stays the truth. For a launch
// every live allocation is copied to the card at the same virtual address (the driver's virtual-memory calls reserve the
// simulator's device window and back the granules in use), the kernel runs there from the module's own cubin, and what changed
// is copied back into the simulator. A launch that fails on the card changes nothing and runs in the simulator after all.
// The real driver is loaded next to the shim with dlopen (RTLD_DEEPBIND: its symbols are its own), from a thread that is given
// a deadline, because a reservation outside the card's address range can hang instead of failing (WSL2).
#include "vgpu/host_gpu_rt.hpp"

#include <dlfcn.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>

#include "vgpu/error.hpp"
#include "vgpu/sass/exec.hpp"

namespace vgpu::hostgpu {
namespace {

// The few driver types and calls used, declared here so that this file needs no CUDA header (the simulator builds without one).
using CUresult = int;
using CUdevice = int;
using CUcontext = struct CUctx_st*;
using CUmodule = struct CUmod_st*;
using CUfunction = struct CUfunc_st*;
using CUdeviceptr = unsigned long long;
using CUhandle = unsigned long long;

struct CUmemLocation { int type; int id; };
struct CUmemAllocationProp {
  int type;
  int requestedHandleTypes;
  CUmemLocation location;
  void* win32HandleMetaData;
  struct { unsigned char compressionType; unsigned char gpuDirectRDMACapable; unsigned short usage; unsigned char reserved[4]; } allocFlags;
};
struct CUmemAccessDesc { CUmemLocation location; int flags; };

constexpr uint64_t kGranule = 2ull << 20;
constexpr int kAttrMultiprocessors = 16, kAttrCcMajor = 75, kAttrCcMinor = 76;
constexpr int kFuncAttrMaxDynamicShared = 8;

struct Api {
  void* lib = nullptr;
  CUresult (*cuInit)(unsigned) = nullptr;
  CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
  CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice) = nullptr;
  CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
  CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
  CUresult (*cuCtxSynchronize)() = nullptr;
  CUresult (*cuMemAddressReserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) = nullptr;
  CUresult (*cuMemCreate)(CUhandle*, size_t, const CUmemAllocationProp*, unsigned long long) = nullptr;
  CUresult (*cuMemMap)(CUdeviceptr, size_t, size_t, CUhandle, unsigned long long) = nullptr;
  CUresult (*cuMemSetAccess)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) = nullptr;
  CUresult (*cuMemsetD8)(CUdeviceptr, unsigned char, size_t) = nullptr;
  CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;
  CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
  CUresult (*cuMemGetInfo)(size_t*, size_t*) = nullptr;
  CUresult (*cuModuleLoadData)(CUmodule*, const void*) = nullptr;
  CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
  CUresult (*cuFuncSetAttribute)(CUfunction, int, int) = nullptr;
  CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*, void**, void**) = nullptr;

  bool load(const char* path) {
    lib = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!lib) return false;
    // The names the driver exports: cuda.h renames these calls to their _v2 versions with a macro, and the plain names are the
    // old 32-bit-size ones.
#define VGPU_HG_SYM(n) \
  if (!(n = reinterpret_cast<decltype(n)>(dlsym(lib, #n)))) return false;
#define VGPU_HG_SYM_V2(n) \
  if (!(n = reinterpret_cast<decltype(n)>(dlsym(lib, #n "_v2")))) return false;
    VGPU_HG_SYM(cuInit) VGPU_HG_SYM(cuDeviceGet) VGPU_HG_SYM(cuDeviceGetAttribute) VGPU_HG_SYM(cuDevicePrimaryCtxRetain)
    VGPU_HG_SYM(cuCtxSetCurrent) VGPU_HG_SYM(cuCtxSynchronize) VGPU_HG_SYM(cuMemAddressReserve) VGPU_HG_SYM(cuMemCreate)
    VGPU_HG_SYM(cuMemMap) VGPU_HG_SYM(cuMemSetAccess) VGPU_HG_SYM_V2(cuMemsetD8) VGPU_HG_SYM_V2(cuMemcpyHtoD) VGPU_HG_SYM_V2(cuMemcpyDtoH)
    VGPU_HG_SYM_V2(cuMemGetInfo) VGPU_HG_SYM(cuModuleLoadData) VGPU_HG_SYM(cuModuleGetFunction) VGPU_HG_SYM(cuFuncSetAttribute)
    VGPU_HG_SYM(cuLaunchKernel)
#undef VGPU_HG_SYM
#undef VGPU_HG_SYM_V2
    return true;
  }
};

// VGPU_HOST_GPU_TRACE=<file> (spike, step A): the stream of allocations, frees, host copies and launches, for an offline model of the transfers
// a persistent card copy would still need (docs/perf-design/data/round13/residency_model.py). Events the mode itself causes are not recorded.
thread_local bool t_internal = false;
FILE* g_trace_file = nullptr;
std::mutex g_trace_mu;
void trace_hook(char op, uint64_t a, uint64_t b) {
  if (t_internal || !g_trace_file) return;
  std::lock_guard<std::mutex> g(g_trace_mu);
  std::fprintf(g_trace_file, "%c %llx %llx\n", op, static_cast<unsigned long long>(a), static_cast<unsigned long long>(b));
}
// Installed when the library loads, so the allocations and copies before the first launch are in the trace.
struct TraceInit {
  TraceInit() {
    const char* tf = std::getenv("VGPU_HOST_GPU_TRACE");
    if (!tf || !*tf) return;
    g_trace_file = std::fopen(tf, "w");
    if (!g_trace_file) return;
    std::setvbuf(g_trace_file, nullptr, _IOFBF, 1 << 20);
    set_mem_trace(trace_hook);
    std::atexit([] { std::lock_guard<std::mutex> g(g_trace_mu); std::fflush(g_trace_file); });
  }
} g_trace_init;
struct InternalScope {
  bool prev;
  InternalScope() : prev(t_internal) { t_internal = true; }
  ~InternalScope() { t_internal = prev; }
};

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && *q && *q != '0';
}
bool logging() {
  const char* q = std::getenv("VGPU_HOST_GPU_LOG");
  return q && *q && *q != '0';
}

// Diagnostics that need the simulator: with any of them on, the mode refuses to start rather than quietly dropping them.
const char* const kIncompatible[] = {"VGPU_COUNTERS", "VGPU_SASS_TRACE", "VGPU_KERNEL_DIGEST", "VGPU_HOSTGPU_CAPTURE", "VGPU_TRACE",
                                     "VGPU_MAX_STEPS", "VGPU_FAULT", "VGPU_RACE", "VGPU_STRICT", "VGPU_SCHEDULER"};

}  // namespace

struct Runtime::Impl : CardGate {
  Api api;
  CUcontext ctx = nullptr;
  uint64_t window_base = 0;
  uint64_t mapped_end = 0;   // granules [window_base, mapped_end) are backed
  Stats stats;
  Costs costs;
  SimTimeTable sim_times;
  std::map<std::string, double> card_kernel_s;
  TrafficTable traffic;
  std::set<std::string> banned;   // kernels whose offloaded launch took longer than the simulator was expected to
  std::map<const sass::Module*, CUmodule> modules;
  struct Verdict {
    bool ok = false;
    uint64_t instrs = 0;
    bool runnable = false;   // everything but the instruction scan passed: the card could run it
    bool back_edge = false;  // the code has a loop (a branch to itself or an earlier instruction)
  };
  std::map<std::pair<const sass::Module*, std::string>, Verdict> eligible;
  std::mutex mu;
  uint64_t max_bytes = 2ull << 30;
  bool dead = false;
  uint32_t multiprocessors = 0;   // of the simulated profile (and the card: start() checks they agree)
  // The card copy outlives a launch (docs/perf-design/09-host-gpu-spike.md section 4) unless VGPU_HOST_GPU_WRITE_THROUGH=1.
  bool resident = true;
  MemoryManager* gate_mem = nullptr;
  struct Res {
    uint64_t base = 0, size = 0;
    Residency r;
    std::atomic<uint32_t>* slot = nullptr;
  };
  std::map<uint64_t, Res> res;
  // Scratch for the copies between the card and the simulator, kept across launches: a fresh 16 to 32 MiB vector per launch is an mmap, a zero fill and a
  // page fault per page each time (4.7 s of system time in 1,400 launches of PolyBench LU, against 0.6 s with the write-through path).
  std::vector<uint8_t> pull_buf, push_buf, sim_buf;
  bool verify_card_ran = false;   // verify + resident: this launch ran on the card and the simulator runs it next, on memory that is the truth
  struct Last {
    bool in_set = false;
    double card_s = -1;
  } last;
  // VGPU_HOST_GPU_VERIFY=1 runs the launches the mode would offload on the card and then in the simulator, and compares the memory
  // (the simulator's result is the one the program gets); =all does it for every kernel the card can run, in or out of the exact set.
  int verify = 0;
  // VGPU_HOST_GPU_STATIC=1: a kernel the simulator has not run is estimated from its code and grid. NOT a lower bound (threads that leave early
  // execute less: PolyBench 3DCONV was 38x slower with it), so it is an experiment switch, off by default.
  bool static_estimate = false;
  // VGPU_HOST_GPU_NO_FIRST_LAUNCH=1: the first launch of a kernel is always the simulator's (the behaviour before the first-launch rule; for A/B runs).
  bool first_launch_off = false;
  struct Pending {
    bool active = false, in_exact_set = false;
    std::string kernel;
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    std::vector<std::vector<uint8_t>> card;
  } pending;


  // ---- the gate (vgpu::CardGate): called by the memory manager on the slow path of any access to an allocation with a valid card copy ----
  static uint64_t chunk_len(const Res& a, uint64_t c) { return std::min<uint64_t>(kChunkSize, a.size - c * kChunkSize); }

  void update_slot(Res& a) {
    const uint32_t v = static_cast<uint32_t>(a.r.card_valid());
    const uint32_t old = a.slot->exchange(v, std::memory_order_relaxed);
    if (old == 0 && v > 0) gate_mem->card_resident_count().fetch_add(1, std::memory_order_relaxed);
    if (old > 0 && v == 0) gate_mem->card_resident_count().fetch_sub(1, std::memory_order_relaxed);
  }

  // Copies chunks the card alone holds into the simulator's memory (not in a verify run, where the simulator's memory is the truth already).
  void pull_chunks(Res& a, const std::vector<uint64_t>& idx) {
    if (idx.empty() || verify) return;
    api.cuCtxSetCurrent(ctx);
    std::vector<uint8_t>& buf = pull_buf;
    for (const auto& [first, count] : chunk_runs(idx)) {
      for (uint64_t c = first; c < first + count;) {
        const uint64_t n_chunks = std::min<uint64_t>(first + count - c, (32ull << 20) / kChunkSize);
        const uint64_t off = c * kChunkSize, bytes = std::min<uint64_t>(n_chunks * kChunkSize, a.size - off);
        buf.resize(bytes);
        if (api.cuMemcpyDtoH(buf.data(), a.base + off, bytes)) {
          dead = true;
          throw Error(Err::Internal, "host_gpu: reading the card's memory back failed: data only the card held is lost");
        }
        gate_mem->write(a.base + off, buf.data(), bytes);
        stats.pulled_bytes += bytes;
        c += n_chunks;
      }
    }
  }

  void before_access(uint64_t addr, uint64_t len, bool write) override {
    if (t_internal) return;   // the mode's own copies
    std::lock_guard<std::mutex> g(mu);
    auto it = res.upper_bound(addr);
    if (it == res.begin()) return;
    Res& a = std::prev(it)->second;
    if (a.slot->load(std::memory_order_relaxed) == 0) return;
    const ChunkSpan sp = chunk_span(a.base, a.size, addr, len);
    if (!sp.any) return;
    InternalScope internal;
    if (!write) {
      pull_chunks(a, a.r.sim_read(sp.first, sp.last));
    } else {
      for (uint64_t c = sp.first; c <= sp.last; ++c) {
        const bool whole = c == sp.first ? sp.first_whole : c == sp.last ? sp.last_whole : true;
        pull_chunks(a, a.r.sim_write(c, c, whole));
      }
    }
    update_slot(a);
  }

  void before_free(uint64_t base) override {
    if (t_internal) return;
    std::lock_guard<std::mutex> g(mu);
    for (auto it = base == UINT64_MAX ? res.begin() : res.find(base); it != res.end();) {
      Res& a = it->second;
      a.r = Residency(a.r.chunks());   // dropped: nothing of it is valid on the card any more
      update_slot(a);
      if (base == UINT64_MAX) it = res.erase(it);
      else { res.erase(it); break; }
    }
  }

  // The simulator is about to run a launch of its own (mu held by the caller).
  void reset_for_simulator() {
    InternalScope internal;
    for (auto& [b, a] : res) {
      if (a.slot->load(std::memory_order_relaxed) == 0) continue;
      pull_chunks(a, a.r.sim_launch());
      update_slot(a);
    }
  }

  // The tracked allocations are the live ones: new ones start Host (nothing of them is on the card).
  void sync_tracking(MemoryManager& mem, const std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
    if (gate_mem != &mem) {
      gate_mem = &mem;
      mem.set_card_gate(this);
    }
    for (const auto& [b, s] : ranges) {
      const auto it = res.find(b);
      if (it != res.end() && it->second.size == s) continue;
      Res a;
      a.base = b;
      a.size = s;
      a.r = Residency((s + kChunkSize - 1) / kChunkSize);
      a.slot = mem.card_slot(b);
      if (!a.slot) continue;
      res[b] = std::move(a);
    }
  }

  // ---- start-up of the real driver: lazy, and off the launch path ----
  // Loading the driver, cuInit, the primary context and the 1 TiB reservation cost 0.15 to 0.35 s (09 doc, section 5), and used to be paid synchronously at the
  // first launch of any program, offloaded or not. Now (a) nothing happens until a launch of a kernel in the exact set is big enough that it could be worth a card
  // (`start_worthwhile`), so a program with no such kernel never touches the driver, and (b) the work runs on a thread that the simulator does not wait for until a
  // launch is actually about to go to the card (`await_start`): the first simulated launches of the kernel overlap with it. Everything that can hang runs on a
  // thread with a deadline, as before.
  struct StartShared {
    std::mutex mu;
    std::condition_variable cv;
    bool done = false, ok = false;
    std::string why;
  };
  std::shared_ptr<StartShared> start_sh;     // set once the thread is launched
  std::chrono::steady_clock::time_point start_began{};
  bool start_failed = false;                 // the driver could not be used (said once); the mode is off for the rest of the process
  bool start_ready = false;                  // the driver is up: window_base and mapped_end are set
  int want_sm = 0;
  uint32_t want_mp = 0;
  uint64_t va_base = 0;
  std::string lib_path;

  void configure_start(const DeviceProfile& profile, uint64_t base) {
    const char* path = std::getenv("VGPU_HOST_GPU_LIBCUDA");
    lib_path = path && *path ? path : "/usr/lib/x86_64-linux-gnu/libcuda.so.1";
    want_sm = profile.cc_major * 10 + profile.cc_minor;
    want_mp = profile.limits.multiprocessors;
    multiprocessors = want_mp;
    va_base = base;
  }

  // Starts the driver on a thread and returns at once (idempotent). The thread only touches `api` and `ctx`, which nobody reads before await_start.
  void begin_start() {
    if (start_sh || start_failed) return;
    auto sh = std::make_shared<StartShared>();
    start_sh = sh;
    start_began = std::chrono::steady_clock::now();
    Impl* self = this;
    const std::string lib = lib_path;
    const int sm = want_sm;
    const uint32_t mp_want = want_mp;
    const uint64_t base = va_base;
    std::thread([sh, self, lib, sm, mp_want, base] {
      bool ok = false;
      std::string why;
      do {
        if (!self->api.load(lib.c_str())) { why = "cannot load the real driver " + lib; break; }
        if (self->api.cuInit(0)) { why = "cuInit failed"; break; }
        CUdevice d;
        if (self->api.cuDeviceGet(&d, 0)) { why = "no device 0"; break; }
        int mp = 0, cmaj = 0, cmin = 0;
        self->api.cuDeviceGetAttribute(&mp, kAttrMultiprocessors, d);
        self->api.cuDeviceGetAttribute(&cmaj, kAttrCcMajor, d);
        self->api.cuDeviceGetAttribute(&cmin, kAttrCcMinor, d);
        if (cmaj * 10 + cmin != sm || static_cast<uint32_t>(mp) != mp_want) {
          why = "the card (sm_" + std::to_string(cmaj * 10 + cmin) + ", " + std::to_string(mp) + " SMs) is not the simulated device (sm_" +
                std::to_string(sm) + ", " + std::to_string(mp_want) + " SMs)";
          break;
        }
        if (self->api.cuDevicePrimaryCtxRetain(&self->ctx, d) || self->api.cuCtxSetCurrent(self->ctx)) { why = "no context"; break; }
        CUdeviceptr p = 0;
        const CUresult r = self->api.cuMemAddressReserve(&p, kDeviceVaStride, 0, base, 0);
        if (r || p != base) { why = "the card did not give the simulator's address window"; break; }
        ok = true;
      } while (false);
      std::lock_guard<std::mutex> g(sh->mu);
      sh->done = true;
      sh->ok = ok;
      sh->why = why;
      sh->cv.notify_all();
    }).detach();
  }

  // Waits (to the 10 s deadline, counted from the start) for the driver. False: it cannot be used; the mode is off from now on.
  bool await_start() {
    if (start_ready) return true;
    if (start_failed) return false;
    begin_start();
    auto sh = start_sh;
    std::unique_lock<std::mutex> lk(sh->mu);
    if (!sh->cv.wait_until(lk, start_began + std::chrono::seconds(10), [&] { return sh->done; })) {
      if (!quiet()) std::fprintf(stderr, "host_gpu: the driver did not answer within 10 s (an address outside the card's range hangs on some systems); mode off\n");
      start_failed = dead = true;
      return false;
    }
    if (!sh->ok) {
      if (!quiet()) std::fprintf(stderr, "host_gpu: %s; mode off\n", sh->why.c_str());
      start_failed = dead = true;
      return false;
    }
    window_base = va_base;
    mapped_end = va_base;
    start_ready = true;
    return true;
  }

  // At exit: a driver start that is still running must not race the process's teardown (the real driver's own exit handlers). Bounded by the same deadline.
  void settle_start() {
    if (!start_sh) return;
    std::unique_lock<std::mutex> lk(start_sh->mu);
    start_sh->cv.wait_until(lk, start_began + std::chrono::seconds(10), [&] { return start_sh->done; });
  }

  // The granules of [lo, hi) are backed by physical memory on the card (filled with zeros when first mapped).
  bool ensure_mapped(uint64_t hi) {
    hi = (hi + kGranule - 1) & ~(kGranule - 1);
    if (hi <= mapped_end) return true;
    const uint64_t lo = mapped_end, size = hi - lo;
    size_t freeb = 0, total = 0;
    api.cuMemGetInfo(&freeb, &total);
    if (size + (256ull << 20) > freeb) return false;
    CUmemAllocationProp prop = {};
    prop.type = 1;   // CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = 1;   // CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = 0;
    CUhandle h;
    CUresult r = api.cuMemCreate(&h, size, &prop, 0);
    if (!r) r = api.cuMemMap(lo, size, 0, h, 0);
    CUmemAccessDesc ad = {};
    ad.location = prop.location;
    ad.flags = 3;   // CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    if (!r) r = api.cuMemSetAccess(lo, size, &ad, 1);
    if (r) {
      if (logging()) std::fprintf(stderr, "host_gpu: mapping 0x%llx + %llu MiB failed (%d)\n", static_cast<unsigned long long>(lo), static_cast<unsigned long long>(size >> 20), r);
      return false;
    }
    api.cuMemsetD8(lo, 0, size);
    mapped_end = hi;
    return true;
  }

  // VGPU_HOST_GPU_DIV=1: kernels whose only outside-the-set instructions are the IEEE division idiom (host_gpu.hpp division_idiom_only) are in the set. Off until
  // the card's exhaustive division check (tools/probes/hostgpu_sfu_exhaustive.cu div) has run clean.
  bool div_idiom = false;

  static std::vector<CodeInsn> idiom_summary(const sass::Code& code) {
    std::vector<CodeInsn> out;
    out.reserve(code.count);
    for (size_t i = 0; i < code.count; ++i) {
      const sass::Instr& ins = code.instr(i);
      CodeInsn c;
      c.kind = classify_for_idiom(ins.mnemonic, ins.mods);
      c.pc = ins.pc;
      const auto reg_of = [&](size_t k) { return ins.src.size() > k && ins.src[k].kind == sass::Kind::Reg && !ins.src[k].neg && !ins.src[k].abs ? static_cast<int>(ins.src[k].reg) : -1; };
      if (c.kind == CodeInsn::Kind::Call && !ins.src.empty() && ins.src[0].kind == sass::Kind::Label) c.target = ins.src[0].imm;
      if (c.kind == CodeInsn::Kind::MufuRcp) c.src0 = reg_of(0);
      if (c.kind == CodeInsn::Kind::Fchk) { c.src0 = reg_of(0); c.src1 = reg_of(1); }
      out.push_back(c);
    }
    return out;
  }

  bool kernel_eligible(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg, const std::vector<std::vector<uint8_t>>& args,
                       MemoryManager& mem, const char** why, uint64_t* instrs, bool* runnable, bool* back_edge) {
    *runnable = false;
    *back_edge = false;
    if (!fn.sass || fn.sass->image.empty()) { *why = "not a SASS module"; return false; }
    if (cfg.cooperative || cfg.cluster != std::array<uint32_t, 3>{0, 0, 0} || fn.req_cluster != std::array<uint32_t, 3>{0, 0, 0}) { *why = "cooperative or cluster launch"; return false; }
    if (cfg.textures) { *why = "texture objects"; return false; }
    if (cfg.conditionals) { *why = "inside a graph with conditionals"; return false; }
    if (cfg.warp_size && cfg.warp_size != 32) { *why = "warp size"; return false; }
    for (const auto& a : args)
      for (size_t i = 0; i + 8 <= a.size(); i += 8) {
        uint64_t v;
        std::memcpy(&v, &a[i], 8);
        if (v && mem.is_host_mapped(v)) { *why = "argument points into mapped host memory"; return false; }
      }
    const sass::Module& m = *fn.sass;
    const auto key = std::make_pair(&m, fn.name);
    if (const auto it = eligible.find(key); it != eligible.end()) {
      if (!it->second.ok) *why = "kernel code not in the exact set (cached)";
      *instrs = it->second.instrs;
      *runnable = it->second.runnable;
      *back_edge = it->second.back_edge;
      return it->second.ok;
    }
    bool ok = true;
    uint64_t n_instrs = 0;
    std::vector<std::pair<uint64_t, uint64_t>> branches;
    if (!m.symbol_va.empty() || !m.cubin.externs.empty() || m.device_launches) { ok = false; *why = "module variables, device calls or device-side launches"; }
    const auto ci = m.code_index.find(".text." + fn.name);
    if (ok && ci == m.code_index.end()) { ok = false; *why = "no code section"; }
    const bool runnable_here = ok;
    if (ok) {
      const sass::Code& code = m.code[ci->second];
      n_instrs = code.count;
      bool idiom_checked = false, idiom_ok = false;
      for (size_t i = 0; i < code.count && ok; ++i) {
        const sass::Instr& ins = code.instr(i);
        if (ins.op == sass::Op::BRA)
          for (const auto& o : ins.src)
            if (o.kind == sass::Kind::Label) branches.emplace_back(ins.pc, static_cast<uint64_t>(o.imm));
        if (!exact_opcode_allowed(ins.mnemonic, ins.mods)) {
          if (div_idiom && (ins.op == sass::Op::MUFU || ins.op == sass::Op::CALL || ins.op == sass::Op::RET)) {
            if (!idiom_checked) {
              idiom_checked = true;
              idiom_ok = division_idiom_only(idiom_summary(code));
            }
            if (idiom_ok) continue;
          }
          ok = false;
          *why = "instruction outside the exact set";
          if (logging()) std::fprintf(stderr, "host_gpu: %s refused: %s\n", fn.name.c_str(), sass::to_text(ins).c_str());
        }
      }
    }
    eligible[key] = {ok, n_instrs, runnable_here, has_back_edge(branches)};
    *instrs = n_instrs;
    *runnable = runnable_here;
    *back_edge = eligible[key].back_edge;
    return ok;
  }

  CUfunction function_for(const ptx::EntryFn& fn) {
    const sass::Module* m = fn.sass.get();
    CUmodule mod;
    if (const auto it = modules.find(m); it != modules.end()) mod = it->second;
    else {
      if (const CUresult r = api.cuModuleLoadData(&mod, m->image.data())) {
        if (logging()) std::fprintf(stderr, "host_gpu: cuModuleLoadData failed (%d)\n", r);
        return nullptr;
      }
      modules[m] = mod;
    }
    CUfunction f = nullptr;
    if (api.cuModuleGetFunction(&f, mod, fn.name.c_str())) return nullptr;
    return f;
  }
};

Runtime::Runtime() : impl_(new Impl) {}
Runtime::~Runtime() { delete impl_; }

namespace {
void print_summary(Runtime::Impl* im) {
  if (quiet() || !im || im->start_failed) return;   // (a driver that could not be used already said "mode off")
  const Stats& s = im->stats;
  std::fprintf(stderr,
               "host_gpu: %llu launches ran on the card, %llu in the simulator (%llu kernels not in the exact set, %llu too small, %llu unsupported launch shapes, %llu failed on the card)\n",
               static_cast<unsigned long long>(s.offloaded), static_cast<unsigned long long>(s.simulated), static_cast<unsigned long long>(s.refused_kernel),
               static_cast<unsigned long long>(s.refused_size), static_cast<unsigned long long>(s.refused_launch), static_cast<unsigned long long>(s.failed));
  std::fprintf(stderr, "host_gpu: %.1f MiB went to the card, %.1f MiB came back\n", s.pushed_bytes / 1048576.0, s.pulled_bytes / 1048576.0);
  if (im->verify)
    std::fprintf(stderr, "host_gpu: verify: kernels in the exact set: %llu identical, %llu different; other kernels: %llu identical, %llu different\n",
                 static_cast<unsigned long long>(s.same_exact_set), static_cast<unsigned long long>(s.differ_exact_set), static_cast<unsigned long long>(s.same_other),
                 static_cast<unsigned long long>(s.differ_other));
}
Runtime::Impl* g_summary_impl = nullptr;
}  // namespace

Runtime* Runtime::for_device(const DeviceProfile& profile, int ordinal) {
  static std::once_flag once;
  static Runtime* rt = nullptr;
  std::call_once(once, [&] {
    if (mode_from_text(std::getenv("VGPU_HOST_GPU")) != Mode::Exact) return;
    for (const char* v : kIncompatible)
      if (const char* e = std::getenv(v); e && *e && *e != '0') {
        if (!quiet()) std::fprintf(stderr, "host_gpu: %s is set and needs the simulator's launches; mode off\n", v);
        return;
      }
    if (profile.vendor != "nvidia") {
      if (!quiet()) std::fprintf(stderr, "host_gpu: only NVIDIA devices; mode off\n");
      return;
    }
    auto* r = new Runtime;
    // The driver is not started here: see Impl::begin_start. The switches are read now so that they do not depend on whether it came up.
    r->impl_->configure_start(profile, kDeviceVaBase);
    if (const char* v = std::getenv("VGPU_HOST_GPU_VERIFY"); v && *v && *v != '0') r->impl_->verify = std::string(v) == "all" ? 2 : 1;
    if (const char* v = std::getenv("VGPU_HOST_GPU_WRITE_THROUGH"); v && *v && *v != '0') r->impl_->resident = false;
    if (const char* v = std::getenv("VGPU_HOST_GPU_NO_FIRST_LAUNCH"); v && *v && *v != '0') r->impl_->first_launch_off = true;
    if (const char* v = std::getenv("VGPU_HOST_GPU_DIV"); v && *v && *v != '0') r->impl_->div_idiom = true;
    if (const char* v = std::getenv("VGPU_HOST_GPU_STATIC"); v && *v && *v != '0') r->impl_->static_estimate = true;
    if (const char* mb = std::getenv("VGPU_HOST_GPU_MAX_MIB")) r->impl_->max_bytes = std::strtoull(mb, nullptr, 10) << 20;
    g_summary_impl = r->impl_;
    std::atexit([] {
      g_summary_impl->settle_start();
      print_summary(g_summary_impl);
    });
    rt = r;
  });
  return ordinal == 0 ? rt : nullptr;
}

void Runtime::before_simulated(MemoryManager& mem) {
  if (mem.card_resident_count().load(std::memory_order_relaxed) <= 0) {
    impl_->verify_card_ran = false;
    return;
  }
  std::lock_guard<std::mutex> g(impl_->mu);
  if (impl_->verify_card_ran) {   // verifying: this launch already ran on the card and the simulator runs it too; the card's copy stays valid
    impl_->verify_card_ran = false;
    return;
  }
  impl_->reset_for_simulator();
}

void Runtime::note_simulated(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg, double sim_seconds, const MemoryManager* mem,
                             const std::vector<std::vector<uint8_t>>* args) {
  std::lock_guard<std::mutex> g(impl_->mu);
  if (impl_->start_failed) return;   // the driver could not be used: nothing to learn for
  InternalScope internal;
  if (g_trace_file) {
    std::string bases;
    if (mem && args)
      for (const auto& a : *args)
        for (size_t i = 0; i + 8 <= a.size(); i += 8) {
          uint64_t v, b, sz;
          std::memcpy(&v, &a[i], 8);
          if (v && mem->find_allocation(v, &b, &sz)) {
            char t[24];
            std::snprintf(t, sizeof t, " %llx", static_cast<unsigned long long>(b));
            bases += t;
          }
        }
    std::lock_guard<std::mutex> tg(g_trace_mu);
    std::fprintf(g_trace_file, "L %s %llu %.9f %d %.9f |%s\n", fn.name.c_str(), static_cast<unsigned long long>(uint64_t{cfg.grid[0]} * cfg.grid[1] * cfg.grid[2]),
                 sim_seconds, impl_->last.in_set ? 1 : 0, impl_->last.card_s, bases.c_str());
  }
  Impl::Pending& p = impl_->pending;
  if (p.active && mem && p.kernel == fn.name) {
    bool same = true;
    uint64_t bad_chunks = 0;
    std::vector<uint8_t> sim_bytes;
    for (size_t i = 0; i < p.ranges.size(); ++i) {
      sim_bytes.resize(p.ranges[i].second);
      mem->read(p.ranges[i].first, sim_bytes.data(), p.ranges[i].second);
      for (uint64_t c = 0; c < sim_bytes.size(); c += kChunkSize)
        if (std::memcmp(sim_bytes.data() + c, p.card[i].data() + c, std::min<uint64_t>(kChunkSize, sim_bytes.size() - c)) != 0) {
          same = false;
          ++bad_chunks;
        }
    }
    Stats& st = impl_->stats;
    ++(p.in_exact_set ? (same ? st.same_exact_set : st.differ_exact_set) : (same ? st.same_other : st.differ_other));
    if (logging() || (!same && p.in_exact_set))
      std::fprintf(stderr, "host_gpu: verify %s (%s): %s%s\n", fn.name.c_str(), p.in_exact_set ? "in the exact set" : "outside the exact set",
                   same ? "identical" : "DIFFERENT, 64 KiB chunks: ", same ? "" : std::to_string(bad_chunks).c_str());
    p = {};
  }
  impl_->sim_times.record(fn.name, uint64_t{cfg.grid[0]} * cfg.grid[1] * cfg.grid[2], sim_seconds);
}

bool Runtime::try_launch(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg, const std::vector<std::vector<uint8_t>>& args,
                         MemoryManager& mem) {
  Impl& im = *impl_;
  std::lock_guard<std::mutex> g(im.mu);   // one offloaded launch at a time: the card path shares the reserved window
  InternalScope internal;
  im.last = {};
  if (im.dead) return false;
  im.pending = {};
  const char* why = "";
  uint64_t static_instrs = 0;
  bool runnable = false, back_edge = false;
  const bool in_exact_set = im.kernel_eligible(fn, cfg, args, mem, &why, &static_instrs, &runnable, &back_edge);
  im.last.in_set = in_exact_set;
  const uint64_t blocks = uint64_t{cfg.grid[0]} * cfg.grid[1] * cfg.grid[2];
  // A kernel that could go to the card, in a launch big enough to be worth one: start the driver now, on a thread; the simulator runs this launch meanwhile.
  if ((in_exact_set || (im.verify == 2 && runnable)) && (im.verify || start_worthwhile(blocks, cfg.block[0] * cfg.block[1] * cfg.block[2]))) im.begin_start();
  if (!in_exact_set && !(im.verify == 2 && runnable)) {
    ++(std::string(why).find("exact set") != std::string::npos ? im.stats.refused_kernel : im.stats.refused_launch);
    ++im.stats.simulated;
    if (logging()) std::fprintf(stderr, "host_gpu: %s stays in the simulator: %s\n", fn.name.c_str(), why);
    return false;
  }
  // What the simulator took before for this kernel, else a lower bound from the code and the grid.
  std::optional<double> sim_s = im.sim_times.predict(fn.name, blocks);
  if (!sim_s && im.static_estimate) sim_s = static_sim_seconds(blocks, cfg.block[0] * cfg.block[1] * cfg.block[2], static_instrs, im.costs);
  if (!sim_s && im.verify) sim_s = 0;   // verifying: every launch runs on both, whatever the rule would say
  bool first_launch_rule = false;
  if (!sim_s && !im.verify && im.sim_times.samples(fn.name) == 0 && im.card_kernel_s.count(fn.name) == 0 && !im.first_launch_off) {
    // Nothing is known of this kernel: the conservative first-launch rule (host_gpu.hpp) may send this one launch to the card.
    uint64_t live = 0;
    for (const auto& r : mem.live_ranges()) live += r.second;
    const FirstLaunch fl{blocks, cfg.block[0] * cfg.block[1] * cfg.block[2], static_instrs, back_edge, im.multiprocessors, live};
    if (first_launch_offload(fl, im.costs)) {
      sim_s = static_sim_seconds(blocks, fl.threads_per_block, static_instrs, im.costs);
      first_launch_rule = true;
    }
  }
  if (!sim_s) {   // the first launches of a kernel are the simulator's: they teach the estimate (VGPU_HOST_GPU_STATIC=1 is the exception, and is not sound)
    ++im.stats.simulated;
    return false;
  }
  const auto ranges = mem.live_ranges();
  uint64_t total = 0, hi = 0;
  for (const auto& r : ranges) {
    total += r.second;
    hi = std::max(hi, r.first + r.second);
  }
  uint64_t push_bytes = total;
  if (im.resident) {   // only the chunks the simulator wrote since the card last had them; what the host reads later is paid when it reads
    im.sync_tracking(mem, ranges);
    push_bytes = 0;
    for (const auto& [b, a] : im.res)
      for (uint64_t c : a.r.host_chunks()) push_bytes += Impl::chunk_len(a, c);
  }
  Estimate e;
  e.sim_s = *sim_s;
  e.push_bytes = push_bytes;
  e.pull_bytes = im.resident ? 0 : total;   // write-through: an upper bound, only what changed is written back, but the card's copy is read in full
  if (const auto it = im.card_kernel_s.find(fn.name); it != im.card_kernel_s.end()) e.card_kernel_s = it->second;
  if (!im.verify && im.banned.count(fn.name)) {
    ++im.stats.refused_size;
    ++im.stats.simulated;
    return false;
  }
  if (total > im.max_bytes || (!im.verify && !should_offload(e, im.costs))) {
    ++im.stats.refused_size;
    ++im.stats.simulated;
    if (logging()) std::fprintf(stderr, "host_gpu: %s stays in the simulator (sim %.3f ms, card path %.3f ms, %.1f MiB)\n", fn.name.c_str(), e.sim_s * 1e3, card_path_seconds(e, im.costs) * 1e3, total / 1048576.0);
    return false;
  }

  // The card path. Anything that goes wrong before the kernel has run leaves the simulator's memory as it was.
  if (!im.await_start()) {   // the driver is unusable (said once on stderr): this launch and all later ones are the simulator's
    ++im.stats.simulated;
    return false;
  }
  const auto t0 = std::chrono::steady_clock::now();
  im.api.cuCtxSetCurrent(im.ctx);
  const uint64_t base = mem.va_base();
  if (hi < base || !im.ensure_mapped(hi)) {
    if (logging()) std::fprintf(stderr, "host_gpu: %s: could not back the simulator's window up to 0x%llx on the card\n", fn.name.c_str(), static_cast<unsigned long long>(hi));
    ++im.stats.failed;
    ++im.stats.simulated;
    return false;
  }
  CUfunction f = im.function_for(fn);
  if (!f) {
    if (logging()) std::fprintf(stderr, "host_gpu: %s: the real driver did not load the module or find the kernel\n", fn.name.c_str());
    ++im.stats.failed;
    ++im.stats.simulated;
    return false;
  }
  std::vector<uint8_t>& buf = im.push_buf;
  bool card_only_data = false;   // chunks only the card holds, from earlier launches: lost if the card fails now
  if (im.resident) {
    for (auto& [b, a] : im.res) {
      card_only_data = card_only_data || a.r.count(Where::Card) > 0;
      for (const auto& [first, count] : chunk_runs(a.r.host_chunks()))
        for (uint64_t c = first; c < first + count;) {
          const uint64_t n_chunks = std::min<uint64_t>(first + count - c, (32ull << 20) / kChunkSize);
          const uint64_t off = c * kChunkSize, bytes = std::min<uint64_t>(n_chunks * kChunkSize, a.size - off);
          buf.resize(bytes);
          mem.read(a.base + off, buf.data(), bytes);
          if (im.api.cuMemcpyHtoD(a.base + off, buf.data(), bytes)) {
            im.dead = true;
            ++im.stats.failed;
            ++im.stats.simulated;
            if (card_only_data) throw Error(Err::Internal, "host_gpu: copying to the card failed while it holds data the simulator does not");
            im.reset_for_simulator();
            return false;
          }
          im.stats.pushed_bytes += bytes;
          c += n_chunks;
        }
      a.r.card_prepare();   // the pushed chunks are Both now
    }
  } else {
    for (const auto& [rb, rs] : ranges)
      for (uint64_t at = 0; at < rs;) {
        const uint64_t n = std::min<uint64_t>(rs - at, 32ull << 20);
        buf.resize(n);
        mem.read(rb + at, buf.data(), n);
        if (im.api.cuMemcpyHtoD(rb + at, buf.data(), n)) { im.dead = true; ++im.stats.failed; ++im.stats.simulated; return false; }
        im.stats.pushed_bytes += n;
        at += n;
      }
  }
  if (cfg.shared_bytes > 48 * 1024) im.api.cuFuncSetAttribute(f, kFuncAttrMaxDynamicShared, static_cast<int>(cfg.shared_bytes));
  std::vector<void*> params;
  std::vector<std::vector<uint8_t>> keep = args;
  for (auto& a : keep) params.push_back(a.data());
  const auto k0 = std::chrono::steady_clock::now();
  CUresult r = im.api.cuLaunchKernel(f, cfg.grid[0], cfg.grid[1], cfg.grid[2], cfg.block[0], cfg.block[1], cfg.block[2], cfg.shared_bytes, nullptr,
                                     params.data(), nullptr);
  if (!r) r = im.api.cuCtxSynchronize();
  const double kernel_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - k0).count();
  if (r) {   // a failed kernel poisons the context (an illegal address): the mode ends, the simulator takes this launch
    im.dead = true;
    ++im.stats.failed;
    ++im.stats.simulated;
    if (im.resident) {
      if (card_only_data) throw Error(Err::Internal, "host_gpu: " + fn.name + " failed on the card, which holds data from earlier launches that the simulator does not");
      im.reset_for_simulator();
    }
    if (!quiet()) std::fprintf(stderr, "host_gpu: %s failed on the card (error %d); mode off, the simulator runs it\n", fn.name.c_str(), r);
    return false;
  }
  im.last.card_s = kernel_s;
  if (im.resident) {   // the kernel may have written any live chunk: every one is the card's now
    std::vector<uint64_t> newly;
    for (auto& [b, a] : im.res) {
      const bool was = a.slot->load(std::memory_order_relaxed) > 0;
      a.r.card_launched();
      im.update_slot(a);
      if (!was) newly.push_back(a.base);
    }
    mem.card_void_lookups(newly);
  }
  if (im.verify) {   // keep the card's bytes; the simulator runs the launch next and note_simulated compares
    im.pending = {};
    im.pending.active = true;
    im.pending.in_exact_set = in_exact_set;
    im.pending.kernel = fn.name;
    im.pending.ranges = ranges;
    for (const auto& [rb, rs] : ranges) {
      im.pending.card.emplace_back(rs);
      for (uint64_t at = 0; at < rs; at += (32ull << 20))
        if (im.api.cuMemcpyDtoH(im.pending.card.back().data() + at, rb + at, std::min<uint64_t>(rs - at, 32ull << 20))) { im.pending = {}; return false; }
    }
    im.verify_card_ran = true;
    ++im.stats.simulated;
    return false;
  }
  if (im.resident) {   // nothing comes back now: the simulator's memory is stale where the card wrote, and the gate pulls it when it is read
    const double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const bool first_card_launch = im.card_kernel_s.count(fn.name) == 0;
    if (!first_card_launch && !offload_paid_off(total_s, e.sim_s)) {
      im.banned.insert(fn.name);
      if (logging()) std::fprintf(stderr, "host_gpu: %s took %.3f ms on the card path against an estimate of %.3f ms in the simulator: offloading it ends\n", fn.name.c_str(), total_s * 1e3, e.sim_s * 1e3);
    }
    if (im.traffic.record(fn.name, push_bytes, total, total_s, e.sim_s, !first_card_launch) && im.banned.insert(fn.name).second && logging())
      std::fprintf(stderr, "host_gpu: %s pushed %.1f MiB of %.1f MiB live on each of its last %u launches: offloading it ends (traffic)\n", fn.name.c_str(),
                   push_bytes / 1048576.0, total / 1048576.0, kTrafficLaunches);
    im.card_kernel_s[fn.name] = kernel_s;
    ++im.stats.offloaded;
    im.sim_times.note_offload(fn.name);
    if (first_launch_rule && logging()) std::fprintf(stderr, "host_gpu: %s: first launch decided by the static rule (loop, %llu blocks)\n", fn.name.c_str(), static_cast<unsigned long long>(blocks));
    if (logging())
      std::fprintf(stderr, "host_gpu: %s on the card (resident): kernel %.3f ms, total %.3f ms (sim estimate %.3f ms), %.2f MiB pushed\n", fn.name.c_str(), kernel_s * 1e3,
                   total_s * 1e3, e.sim_s * 1e3, push_bytes / 1048576.0);
    return true;
  }
  // Back into the simulator: only the 64 KiB chunks whose bytes differ.
  std::vector<uint8_t>& sim_bytes = im.sim_buf;
  for (const auto& [rb, rs] : ranges)
    for (uint64_t at = 0; at < rs;) {
      const uint64_t n = std::min<uint64_t>(rs - at, 32ull << 20);
      buf.resize(n);
      sim_bytes.resize(n);
      if (im.api.cuMemcpyDtoH(buf.data(), rb + at, n)) {
        // The kernel has run: the simulator can no longer run it again on its own memory, which is half written back.
        im.dead = true;
        throw Error(Err::Internal, "host_gpu: reading the card's memory back failed after the kernel ran");
      }
      im.stats.pulled_bytes += n;
      mem.read(rb + at, sim_bytes.data(), n);
      for (uint64_t c = 0; c < n; c += kChunkSize) {
        const uint64_t m = std::min<uint64_t>(kChunkSize, n - c);
        if (std::memcmp(buf.data() + c, sim_bytes.data() + c, m) != 0) mem.write(rb + at + c, buf.data() + c, m);
      }
      at += n;
    }
  const double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  // The first launch of a kernel on the card also pays the module's load: not held against the estimate.
  if (im.card_kernel_s.count(fn.name) && !offload_paid_off(total_s, e.sim_s)) {
    im.banned.insert(fn.name);
    if (logging()) std::fprintf(stderr, "host_gpu: %s took %.3f ms on the card path against an estimate of %.3f ms in the simulator: offloading it ends\n", fn.name.c_str(), total_s * 1e3, e.sim_s * 1e3);
  }
  im.card_kernel_s[fn.name] = kernel_s;
  ++im.stats.offloaded;
  im.sim_times.note_offload(fn.name);
  if (logging())
    std::fprintf(stderr, "host_gpu: %s on the card: kernel %.3f ms, total %.3f ms (sim estimate %.3f ms), %.1f MiB each way\n", fn.name.c_str(), kernel_s * 1e3,
                 total_s * 1e3, e.sim_s * 1e3, total / 1048576.0);
  return true;
}

}  // namespace vgpu::hostgpu
