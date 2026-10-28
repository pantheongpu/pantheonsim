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

struct Runtime::Impl {
  Api api;
  CUcontext ctx = nullptr;
  uint64_t window_base = 0;
  uint64_t mapped_end = 0;   // granules [window_base, mapped_end) are backed
  Stats stats;
  Costs costs;
  SimTimeTable sim_times;
  std::map<std::string, double> card_kernel_s;
  std::set<std::string> banned;   // kernels whose offloaded launch took longer than the simulator was expected to
  std::map<const sass::Module*, CUmodule> modules;
  struct Verdict {
    bool ok = false;
    uint64_t instrs = 0;
    bool runnable = false;   // everything but the instruction scan passed: the card could run it
  };
  std::map<std::pair<const sass::Module*, std::string>, Verdict> eligible;
  std::mutex mu;
  uint64_t max_bytes = 2ull << 30;
  bool dead = false;
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
  struct Pending {
    bool active = false, in_exact_set = false;
    std::string kernel;
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    std::vector<std::vector<uint8_t>> card;
  } pending;

  bool start(const DeviceProfile& profile, uint64_t base) {
    const char* path = std::getenv("VGPU_HOST_GPU_LIBCUDA");
    if (!path || !*path) path = "/usr/lib/x86_64-linux-gnu/libcuda.so.1";
    // Everything that can hang runs on a thread with a deadline.
    struct Shared {
      std::mutex mu;
      std::condition_variable cv;
      bool done = false, ok = false;
      std::string why;
    };
    auto sh = std::make_shared<Shared>();
    Impl* self = this;
    const std::string lib = path;
    const int want_sm = profile.cc_major * 10 + profile.cc_minor;
    const uint32_t want_mp = profile.limits.multiprocessors;
    std::thread([sh, self, lib, want_sm, want_mp, base] {
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
        if (cmaj * 10 + cmin != want_sm || static_cast<uint32_t>(mp) != want_mp) {
          why = "the card (sm_" + std::to_string(cmaj * 10 + cmin) + ", " + std::to_string(mp) + " SMs) is not the simulated device (sm_" +
                std::to_string(want_sm) + ", " + std::to_string(want_mp) + " SMs)";
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
    std::unique_lock<std::mutex> lk(sh->mu);
    if (!sh->cv.wait_for(lk, std::chrono::seconds(10), [&] { return sh->done; })) {
      if (!quiet()) std::fprintf(stderr, "host_gpu: the driver did not answer within 10 s (an address outside the card's range hangs on some systems); mode off\n");
      return false;
    }
    if (!sh->ok && !quiet()) std::fprintf(stderr, "host_gpu: %s; mode off\n", sh->why.c_str());
    window_base = base;
    mapped_end = base;
    return sh->ok;
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

  bool kernel_eligible(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg, const std::vector<std::vector<uint8_t>>& args,
                       MemoryManager& mem, const char** why, uint64_t* instrs, bool* runnable) {
    *runnable = false;
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
      return it->second.ok;
    }
    bool ok = true;
    uint64_t n_instrs = 0;
    if (!m.symbol_va.empty() || !m.cubin.externs.empty() || m.device_launches) { ok = false; *why = "module variables, device calls or device-side launches"; }
    const auto ci = m.code_index.find(".text." + fn.name);
    if (ok && ci == m.code_index.end()) { ok = false; *why = "no code section"; }
    const bool runnable_here = ok;
    if (ok) {
      const sass::Code& code = m.code[ci->second];
      n_instrs = code.count;
      for (size_t i = 0; i < code.count && ok; ++i) {
        const sass::Instr& ins = code.instr(i);
        if (!exact_opcode_allowed(ins.mnemonic, ins.mods)) {
          ok = false;
          *why = "instruction outside the exact set";
          if (logging()) std::fprintf(stderr, "host_gpu: %s refused: %s\n", fn.name.c_str(), sass::to_text(ins).c_str());
        }
      }
    }
    eligible[key] = {ok, n_instrs, runnable_here};
    *instrs = n_instrs;
    *runnable = runnable_here;
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
  if (quiet() || !im) return;
  const Stats& s = im->stats;
  std::fprintf(stderr,
               "host_gpu: %llu launches ran on the card, %llu in the simulator (%llu kernels not in the exact set, %llu too small, %llu unsupported launch shapes, %llu failed on the card)\n",
               static_cast<unsigned long long>(s.offloaded), static_cast<unsigned long long>(s.simulated), static_cast<unsigned long long>(s.refused_kernel),
               static_cast<unsigned long long>(s.refused_size), static_cast<unsigned long long>(s.refused_launch), static_cast<unsigned long long>(s.failed));
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
    if (!r->impl_->start(profile, kDeviceVaBase)) {
      // The detached start-up thread may still run (the hang case): the runtime is leaked, not destroyed.
      return;
    }
    if (const char* v = std::getenv("VGPU_HOST_GPU_VERIFY"); v && *v && *v != '0') r->impl_->verify = std::string(v) == "all" ? 2 : 1;
    if (const char* v = std::getenv("VGPU_HOST_GPU_STATIC"); v && *v && *v != '0') r->impl_->static_estimate = true;
    if (const char* mb = std::getenv("VGPU_HOST_GPU_MAX_MIB")) r->impl_->max_bytes = std::strtoull(mb, nullptr, 10) << 20;
    g_summary_impl = r->impl_;
    std::atexit([] { print_summary(g_summary_impl); });
    rt = r;
  });
  return ordinal == 0 ? rt : nullptr;
}

void Runtime::note_simulated(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg, double sim_seconds, const MemoryManager* mem,
                             const std::vector<std::vector<uint8_t>>* args) {
  std::lock_guard<std::mutex> g(impl_->mu);
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
  bool runnable = false;
  const bool in_exact_set = im.kernel_eligible(fn, cfg, args, mem, &why, &static_instrs, &runnable);
  im.last.in_set = in_exact_set;
  if (!in_exact_set && !(im.verify == 2 && runnable)) {
    ++(std::string(why).find("exact set") != std::string::npos ? im.stats.refused_kernel : im.stats.refused_launch);
    ++im.stats.simulated;
    if (logging()) std::fprintf(stderr, "host_gpu: %s stays in the simulator: %s\n", fn.name.c_str(), why);
    return false;
  }
  const uint64_t blocks = uint64_t{cfg.grid[0]} * cfg.grid[1] * cfg.grid[2];
  // What the simulator took before for this kernel, else a lower bound from the code and the grid.
  std::optional<double> sim_s = im.sim_times.predict(fn.name, blocks);
  if (!sim_s && im.static_estimate) sim_s = static_sim_seconds(blocks, cfg.block[0] * cfg.block[1] * cfg.block[2], static_instrs, im.costs);
  if (!sim_s && im.verify) sim_s = 0;   // verifying: every launch runs on both, whatever the rule would say
  if (!sim_s) {   // the first launch of every kernel is the simulator's: it teaches the estimate (VGPU_HOST_GPU_STATIC=1 is the exception, and is not sound)
    ++im.stats.simulated;
    return false;
  }
  const auto ranges = mem.live_ranges();
  uint64_t total = 0, hi = 0;
  for (const auto& r : ranges) {
    total += r.second;
    hi = std::max(hi, r.first + r.second);
  }
  Estimate e;
  e.sim_s = *sim_s;
  e.push_bytes = total;
  e.pull_bytes = total;   // an upper bound: only what changed is written back, but the card's copy has to be read in full
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
  std::vector<uint8_t> buf;
  for (const auto& [rb, rs] : ranges)
    for (uint64_t at = 0; at < rs;) {
      const uint64_t n = std::min<uint64_t>(rs - at, 32ull << 20);
      buf.resize(n);
      mem.read(rb + at, buf.data(), n);
      if (im.api.cuMemcpyHtoD(rb + at, buf.data(), n)) { im.dead = true; ++im.stats.failed; ++im.stats.simulated; return false; }
      at += n;
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
    if (!quiet()) std::fprintf(stderr, "host_gpu: %s failed on the card (error %d); mode off, the simulator runs it\n", fn.name.c_str(), r);
    return false;
  }
  im.last.card_s = kernel_s;
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
    ++im.stats.simulated;
    return false;
  }
  // Back into the simulator: only the 64 KiB chunks whose bytes differ.
  std::vector<uint8_t> sim_bytes;
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
  if (logging())
    std::fprintf(stderr, "host_gpu: %s on the card: kernel %.3f ms, total %.3f ms (sim estimate %.3f ms), %.1f MiB each way\n", fn.name.c_str(), kernel_s * 1e3,
                 total_s * 1e3, e.sim_s * 1e3, total / 1048576.0);
  return true;
}

}  // namespace vgpu::hostgpu
