// Launch-level result caching (VGPU_MEMO, vgpu/memo.hpp, src/runtime/memo.cpp): the history chain, the launch key, the
// poisoning table (one test per row), exact replay, the on-disk store, and the mutation evidence (each deliberately wrong
// rule, VGPU_MEMO_TEST_MUTANT, must make a test here fail).
//
// Every check compares a device with the cache against the same program on a device without it: the ground truth is
// never the cache's own answer. Programs are hand-written PTX run through the Device API, so this needs no toolkit.
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memo.hpp"
#include "vgpu/profiling.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/memo.hpp"
#include "vgpu/runtime/runtime.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

namespace {

// Two modules with the same kernel names and different code, to tell a key that ignores the code.
std::string module_text(int mul) {
  return std::string(".version 8.0\n.target sm_80\n.address_size 64\n") + R"(
.extern .func (.param .b64 func_retval0) malloc (.param .b64 size);
.extern .func (.param .b32 func_retval0) vprintf (.param .b64 p0, .param .b64 p1);
.global .align 4 .u32 gvar[4];

// y[i] = y[i] * a + x[i]
.visible .entry axpy(.param .u64 y, .param .u64 x, .param .u32 a)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [y];
    ld.param.u64 %rd2, [x];
    ld.param.u32 %r1, [a];
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.u32 %r5, %r3, %r4, %r2;
    mul.wide.u32 %rd3, %r5, 4;
    add.u64 %rd4, %rd1, %rd3;
    add.u64 %rd5, %rd2, %rd3;
    ld.global.u32 %r6, [%rd4];
    ld.global.u32 %r7, [%rd5];
    mad.lo.u32 %r6, %r6, %r1, %r7;
    st.global.u32 [%rd4], %r6;
    ret;
}

// y[i] = MUL (the same name in every module, a different constant)
.visible .entry fixed(.param .u64 y)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [y];
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.u32 %r5, %r3, %r4, %r2;
    mul.wide.u32 %rd3, %r5, 4;
    add.u64 %rd4, %rd1, %rd3;
    mov.u32 %r6, )" + std::to_string(mul) + R"(;
    st.global.u32 [%rd4], %r6;
    ret;
}

// atomicAdd(&y[0], 1) by every thread
.visible .entry atomadd(.param .u64 y)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [y];
    atom.global.add.u32 %r1, [%rd1], 1;
    ret;
}

// p = table[0]; p[tid] = tid + 1: writes through a pointer stored in memory
.visible .entry viaptr(.param .u64 table)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [table];
    ld.global.u64 %rd2, [%rd1];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd3, %r1, 4;
    add.u64 %rd4, %rd2, %rd3;
    add.u32 %r2, %r1, 1;
    st.global.u32 [%rd4], %r2;
    ret;
}

// gvar[0] = v
.visible .entry setglobal(.param .u32 v)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;
    ld.param.u32 %r1, [v];
    mov.u64 %rd1, gvar;
    st.global.u32 [%rd1], %r1;
    ret;
}

// y[i] = i + gvar[0]
.visible .entry readglobal(.param .u64 y)
{
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [y];
    mov.u64 %rd2, gvar;
    ld.global.u32 %r1, [%rd2];
    mov.u32 %r2, %tid.x;
    add.u32 %r3, %r1, %r2;
    mul.wide.u32 %rd3, %r2, 4;
    add.u64 %rd4, %rd1, %rd3;
    st.global.u32 [%rd4], %r3;
    ret;
}

// y[off / 4] = 1
.visible .entry poke(.param .u64 y, .param .u64 off)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [y];
    ld.param.u64 %rd2, [off];
    add.u64 %rd3, %rd1, %rd2;
    mov.u32 %r1, 1;
    st.global.u32 [%rd3], %r1;
    ret;
}

// out[0] = y[off / 4]
.visible .entry peek(.param .u64 y, .param .u64 off, .param .u64 out)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<5>;
    ld.param.u64 %rd1, [y];
    ld.param.u64 %rd2, [off];
    ld.param.u64 %rd4, [out];
    add.u64 %rd3, %rd1, %rd2;
    ld.global.u32 %r1, [%rd3];
    st.global.u32 [%rd4], %r1;
    ret;
}

// vprintf(y, 0) with an empty format (y holds zeros), then y[2] = 5
.visible .entry useprintf(.param .u64 y)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [y];
    {
    .param .b64 param0;
    st.param.b64 [param0+0], %rd1;
    .param .b64 param1;
    st.param.b64 [param1+0], 0;
    .param .b32 retval0;
    call.uni (retval0), vprintf, (param0, param1);
    ld.param.b32 %r1, [retval0+0];
    }
    mov.u32 %r2, 5;
    st.global.u32 [%rd1+8], %r2;
    ret;
}

// y[0] = a malloc'd block's address is not stored; the call is the point
.visible .entry usemalloc(.param .u64 y)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [y];
    mov.u64 %rd2, 16;
    {
    .param .b64 param0;
    st.param.b64 [param0+0], %rd2;
    .param .b64 retval0;
    call.uni (retval0), malloc, (param0);
    ld.param.b64 %rd3, [retval0+0];
    }
    mov.u32 %r1, 5;
    st.global.u32 [%rd1], %r1;
    ret;
}

// y[0] = %smid, %nsmid in y[1], %dynamic_smem_size in y[2]
.visible .entry whoami(.param .u64 y)
{
    .reg .b32 %r<6>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [y];
    mov.u32 %r1, %smid;
    st.global.u32 [%rd1], %r1;
    mov.u32 %r2, %nsmid;
    st.global.u32 [%rd1+4], %r2;
    mov.u32 %r3, %dynamic_smem_size;
    st.global.u32 [%rd1+8], %r3;
    ret;
}

// y[ctaid.x] = 1 if %smid is a sane number: it reads %smid, and writes the same whichever thread ran the block
.visible .entry smids(.param .u64 y)
{
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    .reg .pred %p<2>;
    ld.param.u64 %rd1, [y];
    mov.u32 %r1, %smid;
    setp.lt.u32 %p1, %r1, 100000;
    selp.u32 %r3, 1, 2, %p1;
    mov.u32 %r2, %ctaid.x;
    mul.wide.u32 %rd2, %r2, 4;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r3;
    ret;
}
)";
}

struct EnvGuard {
  std::string name;
  explicit EnvGuard(const char* n, const char* v) : name(n) { ::setenv(n, v, 1); }
  ~EnvGuard() { ::unsetenv(name.c_str()); }
};

memo::Config base_config(const std::string& dir = "") {
  memo::Config c;
  c.enabled = true;
  c.dir = dir;
  c.force_record = true;   // the tests are about what is cached, not about what it costs
  return c;
}

// The cache as shipped in the one respect the host-write budget tests are about: nothing is forced, recording is free (a second of
// credit), and the hashing of the bytes of host writes has `event_credit_ns` of credit and no share of the kernel time.
memo::Config budget_config(const std::string& dir, uint64_t event_credit_ns) {
  memo::Config c = base_config(dir);
  c.force_record = false;
  c.budget = 0.0;
  c.credit_ns = 1000000000ull;
  c.event_credit_ns = event_credit_ns;
  return c;
}

// Runs a shell command; its status is of no interest (a directory that is not there is what rm -rf is asked about).
void sh(const std::string& cmd) {
  const int rc = std::system(cmd.c_str());
  (void)rc;
}

std::string temp_dir(const char* name) {
  char b[256];
  std::snprintf(b, sizeof b, "/tmp/vgpu-memo-test-%d-%s", static_cast<int>(::getpid()), name);
  const std::string cmd = std::string("rm -rf ") + b;
  sh(cmd.c_str());
  return b;
}

void remove_dir(const std::string& d) {
  const std::string cmd = "rm -rf " + d;
  sh(cmd.c_str());
}

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}
std::vector<uint8_t> arg_u32(uint32_t v) {
  std::vector<uint8_t> b(4);
  std::memcpy(b.data(), &v, 4);
  return b;
}

// A device, with the cache (cfg) or without (null), and one module.
struct Rig {
  runtime::Runtime rt;
  runtime::Device& dev;
  uint64_t mod = 0;
  uint64_t y = 0, x = 0, a = 0;   // the program's buffers, for the rows to use
  Rig(const memo::Config* cfg, int mul = 7, const char* gpu = "nvidia/a10") : rt(load_gpu(gpu)), dev(rt.device(0)) {
    if (cfg) dev.enable_memo(*cfg);
    mod = dev.load_module(module_text(mul));
  }
  MemoryManager& mem() { return dev.memory(); }
  uint64_t alloc(uint64_t n) { return mem().alloc(n); }
  void write_words(uint64_t a, const std::vector<uint32_t>& w) { mem().write(a, w.data(), w.size() * 4); }
  std::vector<uint32_t> words(uint64_t a, size_t n) {
    std::vector<uint32_t> w(n);
    mem().read(a, w.data(), n * 4);
    return w;
  }
  void launch(const char* name, uint32_t grid, uint32_t block, const std::vector<std::vector<uint8_t>>& args, uint32_t shared = 0) {
    LaunchConfig cfg;
    cfg.grid = {grid, 1, 1};
    cfg.block = {block, 1, 1};
    cfg.shared_bytes = shared;
    dev.launch(*dev.get_function(mod, name), cfg, args, dev.symbols(mod));
  }
  void launch_cfg(const char* name, const LaunchConfig& cfg, const std::vector<std::vector<uint8_t>>& args) {
    dev.launch(*dev.get_function(mod, name), cfg, args, dev.symbols(mod));
  }
  memo::DeviceMemo* memo() { return dev.memo(); }
  memo::Stats stats() { return dev.memo() ? dev.memo()->stats() : memo::Stats{}; }
  // All live memory, as one hash: the program's whole observable state on the device.
  std::string everything() {
    memo::Hasher h(1);
    for (const auto& [base, size] : mem().live_ranges()) {
      std::vector<uint8_t> b(size);
      mem().read(base, b.data(), size);
      h.u64(base), h.u64(size), h.hash(memo::hash_bytes(b.data(), b.size()));
    }
    const memo::Hash128 d = h.done();
    char s[40];
    std::snprintf(s, sizeof s, "%016llx%016llx", static_cast<unsigned long long>(d.a), static_cast<unsigned long long>(d.b));
    return s;
  }
};

using Program = std::function<void(Rig&, std::vector<std::string>&)>;
struct Outcome {
  std::vector<std::string> log;
  memo::Stats stats;
  memo::Hash128 state;
  std::string error;
};

// Runs `p` on a fresh device (cfg null: no cache); the log is what the program saw.
Outcome run_program(const Program& p, const memo::Config* cfg, int mul = 7, const char* gpu = "nvidia/a10") {
  Outcome o;
  Rig r(cfg, mul, gpu);
  try {
    p(r, o.log);
  } catch (const Error& e) {
    o.error = e.what();
    o.log.push_back(std::string("error: ") + e.what());
  }
  o.log.push_back("end " + r.everything());
  o.stats = r.stats();
  if (r.memo()) o.state = r.memo()->state();
  return o;
}

void check_same(const Outcome& a, const Outcome& b, const char* what) {
  if (a.log.size() != b.log.size()) {
    for (const auto& l : a.log) std::fprintf(stderr, "%s: left  %s\n", what, l.c_str());
    for (const auto& l : b.log) std::fprintf(stderr, "%s: right %s\n", what, l.c_str());
  }
  VCHECK_EQ(a.log.size(), b.log.size());
  for (size_t i = 0; i < a.log.size() && i < b.log.size(); ++i) {
    if (a.log[i] != b.log[i]) std::fprintf(stderr, "%s: step %zu differs: [%s] vs [%s]\n", what, i, a.log[i].c_str(), b.log[i].c_str());
    VCHECK_EQ(a.log[i], b.log[i]);
  }
}

// A program that touches every kind of event and repeats launches on unchanged state.
void typical(Rig& r, std::vector<std::string>& log) {
  const uint64_t y = r.alloc(4096), x = r.alloc(4096);
  std::vector<uint32_t> xs(256);
  for (size_t i = 0; i < xs.size(); ++i) xs[i] = static_cast<uint32_t>(i * 3 + 1);
  r.write_words(x, xs);
  r.mem().fill(y, reinterpret_cast<const uint8_t*>("\1"), 1, 1024);
  for (int i = 0; i < 3; ++i) {
    r.launch("axpy", 2, 128, {arg_u64(y), arg_u64(x), arg_u32(2)});
    log.push_back("axpy " + r.everything());
  }
  for (int i = 0; i < 4; ++i) {
    r.launch("fixed", 2, 128, {arg_u64(y)});   // a fixed point after the first
    log.push_back("fixed " + r.everything());
  }
  const uint64_t z = r.alloc(4096);
  r.mem().copy_to(r.mem(), z, y, 1024);
  r.launch("axpy", 2, 128, {arg_u64(z), arg_u64(x), arg_u32(3)});
  log.push_back("z " + r.everything());
  r.mem().free(x);
  r.launch("fixed", 1, 64, {arg_u64(z)});
  log.push_back("tail " + r.everything());
}

}  // namespace

// ---- the hash ------------------------------------------------------------------------------------------------------------

VTEST(the_hash_sees_every_byte_and_the_length) {
  std::vector<uint8_t> b(300);
  for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<uint8_t>(i * 7 + 3);
  const memo::Hash128 base = memo::hash_bytes(b.data(), b.size());
  for (size_t i = 0; i < b.size(); ++i) {
    b[i] ^= 1;
    VCHECK(memo::hash_bytes(b.data(), b.size()) != base);
    b[i] ^= 1;
  }
  VCHECK(memo::hash_bytes(b.data(), 299) != base);
  VCHECK(memo::hash_bytes(b.data(), 300) == base);
  const std::vector<uint8_t> zeros(4096, 0);
  std::vector<uint8_t> one = zeros;
  one[4095] = 1;
  VCHECK(memo::page_hash(zeros.data(), 4096) != memo::page_hash(one.data(), 4096));
  VCHECK(memo::page_hash(zeros.data(), 4096) != memo::page_hash(zeros.data(), 4095));
}

// The device's part of a launch's key holds the numerics: two profiles that differ only in which sum the tensor core makes (or in
// the published model VGPU_TC_EXACT=published would turn on) give different results, so they must not share a cached one. What a
// value rests on (provenance, evidence) changes no result and is not in the key.
VTEST(the_profile_hash_sees_the_numerics) {
  const auto yaml = [](const std::string& numerics) {
    return std::string("id: nvidia/testcard\nvendor: nvidia\nmodel: \"Test Card\"\narchitecture: hopper\ncompute_capability: \"9.0\"\n"
                       "warp_size: 32\nvram_bytes: 85028896768\nverified: false\nlimits:\n  max_threads_per_block: 1024\n"
                       "  max_block_dim: [1024, 1024, 64]\n  max_grid_dim: [2147483647, 65535, 65535]\n  shared_mem_per_block_bytes: 49152\n"
                       "  shared_mem_per_block_optin_bytes: 232448\n  registers_per_block: 65536\n  multiprocessors: 132\n") + numerics;
  };
  const std::string four = "[numerics.tensor_sum, numerics.tensor_sum_chunk, numerics.tensor_sum_chunk_tf32, numerics.tensor_sum_grid]";
  const std::string ampere = "numerics:\n  tensor_sum: ampere\n  tensor_sum_chunk: 8\n  tensor_sum_chunk_tf32: 4\n  tensor_sum_grid: 25\n";
  const std::string hopper = "numerics:\n  tensor_sum: hopper\n  tensor_sum_chunk: 16\n  tensor_sum_chunk_tf32: 8\n  tensor_sum_grid: 26\n";
  const std::string measured = "provenance:\n  measured: " + four + "\nevidence:\n  numerics: \"a card\"\n";
  const std::string inferred = "provenance:\n  inferred: " + four + "\nevidence:\n  numerics: \"another card of the same capability\"\n";
  const std::string seq = "numerics:\n  tensor_sum: sequential\nprovenance:\n  unknown: [numerics.tensor_sum]\n";
  const std::string seq_pub = "numerics:\n  tensor_sum: sequential\n  tensor_sum_published: hopper\nprovenance:\n  unknown: [numerics.tensor_sum]\n"
                              "  inferred: [numerics.tensor_sum_published]\nevidence:\n  numerics.tensor_sum_published: \"a paper\"\n";
  const auto hash = [&](const std::string& numerics) { return memo::profile_hash(DeviceProfile::from_yaml(yaml(numerics), "t")); };
  const memo::Hash128 a = hash(ampere + measured), h = hash(hopper + measured), s = hash(seq), sp = hash(seq_pub), none = hash("");
  // Profiles that differ only in a numerics field: each pair apart.
  VCHECK(a != h);
  VCHECK(a != s && h != s);
  VCHECK(s != sp);            // the published model alone
  VCHECK(none != s);          // (a profile without the section is not one that says "sequential": a different document)
  VCHECK(none != a && none != h && none != sp);
  // The same numerics on another class of evidence: the same key.
  VCHECK(hash(ampere + inferred) == a);
  VCHECK(hash(ampere + measured) == a);
  // Each field by itself, on a loaded profile.
  const DeviceProfile base = load_gpu("nvidia/a10");
  const memo::Hash128 hb = memo::profile_hash(base);
  VCHECK(memo::profile_hash(load_gpu("nvidia/a10")) == hb);
  const auto changed = [&](const std::function<void(DeviceProfile&)>& edit) {
    DeviceProfile p = base;
    edit(p);
    return memo::profile_hash(p) != hb;
  };
  VCHECK(changed([](DeviceProfile& p) { p.numerics.tensor_sum = vgpu::TensorSum::Hopper; }));
  VCHECK(changed([](DeviceProfile& p) { p.numerics.tensor_sum = vgpu::TensorSum::Sequential; }));
  VCHECK(changed([](DeviceProfile& p) { p.numerics.tensor_sum_chunk = 16; }));
  VCHECK(changed([](DeviceProfile& p) { p.numerics.tensor_sum_chunk_tf32 = 8; }));
  VCHECK(changed([](DeviceProfile& p) { p.numerics.tensor_sum_grid = 26; }));
  VCHECK(changed([](DeviceProfile& p) { p.numerics.tensor_sum_published = vgpu::TensorSum::Hopper; }));
  VCHECK(changed([](DeviceProfile& p) { p.numerics.present = false; }));
  VCHECK(!changed([](DeviceProfile& p) { p.provenance.clear(), p.evidence.clear(); }));
  VCHECK(changed([](DeviceProfile& p) { p.limits.multiprocessors += 1; }));   // (and what it always saw)
  // Two built-in profiles that are the same compute capability and differ in their sum's class only still differ (their ids do).
  VCHECK(memo::profile_hash(load_gpu("nvidia/h100")) != memo::profile_hash(load_gpu("nvidia/h100-pcie")));
}

// ---- which called functions make a launch uncacheable ----------------------------------------------------------------------------

VTEST(the_device_runtime_is_told_from_ptxas_helpers_by_name) {
  for (const char* n : {"malloc", "free", "vprintf", "printf", "cudaMalloc", "cudaGraphLaunch", "cudaLaunchDevice", "cudaGetDevice",
                        "cudaCGGetIntrinsicHandle", "__cudaCDP2GetParameterBufferV2", "__cudaCDP2LaunchDeviceV2", "__cudart_internal", "__cuda_syscall_malloc"})
    VCHECK(memo::is_device_runtime_name(n));
  // Code ptxas links into kernels that divide, take remainders and so on: ordinary computation.
  for (const char* n : {"__cuda_sm20_div_rn_f64_full", "__cuda_sm20_rem_u64", "__cuda_sm3x_div_rn_noftz_f32_slowpath", "$__cuda_sm20_dsqrt_rn_f64_mediumpath_v1",
                        "$__internal_0_$__cuda_sm20_rcp_f64", "__nv_sinf", "_Z6kernelPf", "__assertfail", "memcpy", "memset"})
    VCHECK(!memo::is_device_runtime_name(n));
}

// ---- off is off --------------------------------------------------------------------------------------------------------------

VTEST(without_the_cache_nothing_is_attached) {
  Rig r(nullptr);
  VCHECK(r.memo() == nullptr);
  const Outcome off = run_program(typical, nullptr);
  const memo::Config cfg = base_config();
  const Outcome on = run_program(typical, &cfg);
  check_same(off, on, "off vs on");
}

// ---- hits: in a run, and from disk ----------------------------------------------------------------------------------------------

VTEST(a_launch_that_changes_nothing_hits_in_the_same_run_and_replays_exactly) {
  const memo::Config cfg = base_config();
  const Outcome off = run_program(typical, nullptr);
  const Outcome on = run_program(typical, &cfg);
  check_same(off, on, "memory only");
  // fixed x4: the first changes memory, the second finds the fixed point (recorded, empty), the third and fourth hit.
  VCHECK(on.stats.hits >= 2);
  VCHECK_EQ(on.stats.verify_mismatch, 0u);
  VCHECK_EQ(on.stats.replay_refused, 0u);
}

VTEST(a_second_run_hits_from_the_first_launch_through_the_disk) {
  const std::string dir = temp_dir("warm");
  const memo::Config cfg = base_config(dir);
  const Outcome off = run_program(typical, nullptr);
  const Outcome cold = run_program(typical, &cfg);
  const Outcome warm = run_program(typical, &cfg);
  check_same(off, cold, "cold");
  check_same(off, warm, "warm");
  VCHECK_EQ(cold.stats.disk_hits, 0u);
  // 9 launches: every one of the warm run's is a hit, and from the disk.
  VCHECK_EQ(warm.stats.launches, 9u);
  VCHECK_EQ(warm.stats.hits, 9u);
  VCHECK(warm.stats.disk_hits >= 1);
  // The counters a hit reports are the ones the real run counted.
  VCHECK(warm.stats.replayed_instructions >= cold.stats.recorded_instructions);
  VCHECK(warm.stats.replayed_instructions > 0);
  remove_dir(dir);
}

VTEST(a_warm_run_through_a_dropped_memory_store_is_served_by_the_disk_alone) {
  const std::string dir = temp_dir("disk");
  const memo::Config cfg = base_config(dir);
  const Outcome off = run_program(typical, nullptr);
  (void)run_program(typical, &cfg);
  Outcome warm;
  {
    Rig r(&cfg);
    r.memo()->drop_memory_store();
    typical(r, warm.log);
    warm.log.push_back("end " + r.everything());
    warm.stats = r.stats();
  }
  check_same(off, warm, "disk only");
  VCHECK(warm.stats.disk_hits >= 1);
  remove_dir(dir);
}

VTEST(a_corrupt_entry_on_disk_is_a_miss_and_is_not_trusted) {
  const std::string dir = temp_dir("corrupt");
  const memo::Config cfg = base_config(dir);
  const Outcome off = run_program(typical, nullptr);
  (void)run_program(typical, &cfg);
  // Flip a byte in the middle of every entry.
  const std::string cmd = "find " + dir + " -name '*.vgm' -exec sh -c 'printf X | dd of=\"$1\" bs=1 seek=100 conv=notrunc 2>/dev/null' _ {} \\;";
  sh(cmd.c_str());
  const Outcome warm = run_program(typical, &cfg);
  check_same(off, warm, "corrupt");
  remove_dir(dir);
}

VTEST(the_disk_cache_stays_under_its_cap_and_still_serves_what_is_left) {
  const std::string dir = temp_dir("cap");
  memo::Config cfg = base_config(dir);
  cfg.disk_cap = 6000;   // bytes: a few entries
  const Outcome off = run_program(typical, nullptr);
  const Outcome cold = run_program(typical, &cfg);
  check_same(off, cold, "capped");
  VCHECK(cold.stats.disk_evicted > 0);
  const std::string cmd = "du -sb " + dir + " | cut -f1 > " + dir + ".size";
  sh(cmd.c_str());
  std::FILE* f = std::fopen((dir + ".size").c_str(), "r");
  unsigned long long bytes = 0;
  if (f) {
    VCHECK(std::fscanf(f, "%llu", &bytes) == 1);
    std::fclose(f);
  }
  VCHECK(bytes < 6000 + 4096 * 40);   // (directory entries count in du)
  const Outcome warm = run_program(typical, &cfg);
  check_same(off, warm, "capped warm");
  remove_dir(dir);
  ::unlink((dir + ".size").c_str());
}

// ---- the events of the history, one row each ----------------------------------------------------------------------------------------

namespace {

// The state after `row` has run on a device whose history so far is the same every time.
memo::Hash128 state_after(const std::function<void(Rig&)>& row) {
  const memo::Config cfg = base_config();
  Rig r(&cfg);
  r.a = r.alloc(8192);
  r.write_words(r.a, {1, 2, 3, 4});
  row(r);
  return r.memo()->state();
}

}  // namespace

VTEST(a_host_write_is_an_event_with_its_bytes) {
  const auto s1 = state_after([](Rig& r) { r.mem().write(r.a, "abcd", 4); });
  const auto s2 = state_after([](Rig& r) { r.mem().write(r.a, "abcd", 4); });
  const auto s3 = state_after([](Rig& r) { r.mem().write(r.a, "abce", 4); });
  const auto s4 = state_after([](Rig& r) { r.mem().write(r.a + 4, "abcd", 4); });
  const auto s0 = state_after([](Rig&) {});
  VCHECK(s1 == s2);
  VCHECK(s1 != s3);
  VCHECK(s1 != s4);
  VCHECK(s1 != s0);
}

VTEST(a_fill_is_an_event_with_its_pattern_and_length) {
  const auto f = [](uint8_t b, uint64_t n) {
    return state_after([=](Rig& r) { r.mem().fill(r.a, &b, 1, n); });
  };
  VCHECK(f(1, 64) == f(1, 64));
  VCHECK(f(1, 64) != f(2, 64));
  VCHECK(f(1, 64) != f(1, 65));
  const uint8_t p2[2] = {1, 2}, p2b[2] = {1, 3};
  const auto g = [&](const uint8_t* p) {
    return state_after([=](Rig& r) { r.mem().fill(r.a, p, 2, 64); });
  };
  VCHECK(g(p2) != g(p2b));
}

VTEST(a_device_to_device_copy_is_an_event_with_its_addresses) {
  const auto c = [](uint64_t from, uint64_t to) {
    return state_after([=](Rig& r) {
      const uint64_t a = r.a;
      r.mem().copy_to(r.mem(), a + to, a + from, 64);
    });
  };
  VCHECK(c(0, 4096) == c(0, 4096));
  VCHECK(c(0, 4096) != c(64, 4096));
  VCHECK(c(0, 4096) != c(0, 4160));
  VCHECK(c(0, 4096) != state_after([](Rig&) {}));
}

VTEST(allocation_and_free_are_events_and_reads_are_not) {
  const auto s0 = state_after([](Rig&) {});
  const auto sa = state_after([](Rig& r) { r.alloc(100); });
  const auto sb = state_after([](Rig& r) { r.alloc(101); });
  const auto sf = state_after([](Rig& r) { r.mem().free(r.a); });
  VCHECK(s0 != sa);
  VCHECK(sa != sb);
  VCHECK(s0 != sf);
  const auto sr = state_after([](Rig& r) {
    uint8_t b[16];
    r.mem().read(r.a, b, 16);   // a read
    r.mem().live_ranges();
    uint64_t base = 0, size = 0;
    r.mem().find_allocation(r.a, &base, &size);
    r.mem().load_scalar(r.a, 4);
  });
  VCHECK(sr == s0);
}

VTEST(a_replayed_hit_is_not_an_event_of_its_own) {
  // Two runs of one program end in the same state whether the second replayed from the disk or ran.
  const std::string dir = temp_dir("replay-state");
  const memo::Config cfg = base_config(dir);
  const Outcome cold = run_program(typical, &cfg);
  const Outcome warm = run_program(typical, &cfg);
  VCHECK(cold.state == warm.state);
  remove_dir(dir);
}

// ---- the poisoning table: a row each -----------------------------------------------------------------------------------------------

namespace {

// The program every row interrupts: two launches on a changing state, the row, two more launches. Run twice against one
// directory, the second run must find every launch before the row and none after it (the row restarts the history from a
// value no other run has), and both must equal a device without the cache.
struct RowResult {
  Outcome off, cold, warm;
  memo::Stats before_row_cold, before_row_warm;
};

RowResult run_row(const std::string& name, const std::function<void(Rig&)>& row, int mul = 7, const char* gpu = "nvidia/a10") {
  const std::string dir = temp_dir(name.c_str());
  const memo::Config cfg = base_config(dir);
  std::map<int, memo::Stats> marks;
  const auto prog = [&](Rig& r, std::vector<std::string>& log, int which) {
    const uint64_t y = r.alloc(4096), x = r.alloc(4096);
    r.y = y, r.x = x;
    r.write_words(x, {1, 2, 3, 4, 5, 6, 7, 8});
    r.launch("axpy", 1, 8, {arg_u64(y), arg_u64(x), arg_u32(2)});
    r.launch("axpy", 1, 8, {arg_u64(y), arg_u64(x), arg_u32(2)});
    log.push_back("before " + r.everything());
    marks[which] = r.stats();
    row(r);
    r.launch("axpy", 1, 8, {arg_u64(y), arg_u64(x), arg_u32(3)});
    r.launch("fixed", 1, 8, {arg_u64(y)});
    r.launch("fixed", 1, 8, {arg_u64(y)});
    log.push_back("after " + r.everything());
  };
  RowResult res;
  const auto go = [&](const memo::Config* c, int which) {
    return run_program([&](Rig& r, std::vector<std::string>& log) { prog(r, log, which); }, c, mul, gpu);
  };
  res.off = go(nullptr, 0);
  res.cold = go(&cfg, 1);
  res.warm = go(&cfg, 2);
  res.before_row_cold = marks[1];
  res.before_row_warm = marks[2];
  remove_dir(dir);
  return res;
}

// What a row must do: the program answers as without the cache; before the row the second run hits; after it the
// second run does not (hits after = hits in total less hits before).
void expect_poisoned(const char* name, const RowResult& r, const char* reason) {
  check_same(r.off, r.cold, name);
  check_same(r.off, r.warm, name);
  VCHECK_EQ(r.warm.stats.verify_mismatch, 0u);
  VCHECK(r.before_row_warm.hits >= 2);   // the two launches before the row hit from the disk
  const uint64_t after_hits = r.warm.stats.hits - r.before_row_warm.hits;
  if (after_hits != 0) std::fprintf(stderr, "%s: %llu hits after the row\n", name, static_cast<unsigned long long>(after_hits));
  VCHECK_EQ(after_hits, 0u);
  if (reason) {
    const auto it = r.cold.stats.reasons.find(reason);
    if (it == r.cold.stats.reasons.end()) std::fprintf(stderr, "%s: no reason '%s' counted\n", name, reason);
    VCHECK(it != r.cold.stats.reasons.end());
  }
}

// A row that is only an event: the second run still hits after it.
void expect_survives(const char* name, const RowResult& r) {
  check_same(r.off, r.cold, name);
  check_same(r.off, r.warm, name);
  const uint64_t after_hits = r.warm.stats.hits - r.before_row_warm.hits;
  if (after_hits < 1) std::fprintf(stderr, "%s: no hits after the row\n", name);
  VCHECK(after_hits >= 1);
}

}  // namespace

VTEST(row_vmm_reserve_and_free_poison) {
  expect_poisoned("vmm", run_row("vmm", [](Rig& r) {
    const uint64_t va = r.mem().reserve(2u << 20, 0);
    r.mem().address_free(va, 2u << 20);
  }), "reserve");
}

VTEST(row_vmm_handle_map_poisons_and_a_live_mapping_refuses_launches) {
  expect_poisoned("vmm-map", run_row("vmmmap", [](Rig& r) {
    const uint64_t va = r.mem().reserve(2u << 20, 0);
    const uint64_t h = r.mem().create_handle(2u << 20);
    r.mem().map(va, 2u << 20, 0, h);   // left mapped: every launch after this one is refused
  }), "vmm-or-shared-memory");
}

VTEST(row_a_host_mapping_no_kernel_touches_costs_nothing) {
  // Pinned memory a program only copies through (the usual case) is no reason to stop caching: what matters is whether a kernel reaches it.
  static std::vector<uint8_t> host(8192);
  expect_survives("host-map-idle", run_row("hostmapidle", [](Rig& r) {
    r.mem().map_host(reinterpret_cast<uint64_t>(host.data()), host.data(), host.size());
  }));
}

VTEST(row_a_kernel_that_touches_a_host_mapping_poisons) {
  static std::vector<uint8_t> host(8192);
  expect_poisoned("host-map-touch", run_row("hostmaptouch", [](Rig& r) {
    r.mem().map_host(reinterpret_cast<uint64_t>(host.data()), host.data(), host.size());
    r.launch("fixed", 1, 8, {arg_u64(reinterpret_cast<uint64_t>(host.data()))});   // writes host memory, which no diff sees
  }), "touched-host-memory");
}

VTEST(row_device_reset_poisons) {
  // A reset empties the device: the program's own pointers are gone, so the row is the end of the program.
  const memo::Config cfg = base_config();
  Rig r(&cfg);
  const uint64_t y = r.alloc(4096), x = r.alloc(4096);
  r.launch("axpy", 1, 8, {arg_u64(y), arg_u64(x), arg_u32(2)});
  const memo::Hash128 before = r.memo()->state();
  const uint64_t poisoned_before = r.stats().poisoned;
  r.dev.reset();
  VCHECK(r.memo()->state() != before);
  VCHECK(r.stats().poisoned > poisoned_before);
}

VTEST(row_rebinding_a_module_global_poisons) {
  expect_poisoned("rebind", run_row("rebind", [](Rig& r) {
    const uint64_t other = r.alloc(64);
    r.dev.rebind_global(r.mod, "gvar", other);
  }), "rebind_global");
}

VTEST(row_ipc_export_import_poison) {
  static int n = 0;
  expect_poisoned("ipc", run_row("ipc", [&](Rig& r) {
    const std::string path = "/tmp/vgpu-memo-ipc-" + std::to_string(::getpid()) + "-" + std::to_string(n++);
    const uint64_t p = r.alloc(4096);
    r.mem().share(p, path);
    r.mem().free(p);   // the owner's free takes the file with it
  }), "share");
}

VTEST(row_a_host_write_is_only_an_event) {
  expect_survives("write", run_row("write", [](Rig& r) { r.mem().write(r.y, "abcd", 4); }));
}
VTEST(row_a_fill_is_only_an_event) {
  expect_survives("fill", run_row("fill", [](Rig& r) {
    const uint8_t b = 9;
    r.mem().fill(r.y + 2048, &b, 1, 64);
  }));
}
VTEST(row_a_d2d_copy_is_only_an_event) {
  expect_survives("copy", run_row("copy", [](Rig& r) {
    const uint64_t a = r.y;
    r.mem().copy_to(r.mem(), a + 2048, a, 64);
  }));
}
VTEST(row_alloc_and_free_are_only_events) {
  expect_survives("allocfree", run_row("allocfree", [](Rig& r) {
    const uint64_t p = r.alloc(512);
    r.mem().free(p);
  }));
}
VTEST(row_a_read_back_is_not_even_an_event) {
  expect_survives("read", run_row("read", [](Rig& r) {
    uint8_t b[64];
    r.mem().read(r.y, b, 64);
  }));
}

// Launch rows: a launch the cache cannot represent runs, and poisons what follows.
VTEST(row_launch_calling_malloc_is_refused) {
  expect_poisoned("malloc", run_row("malloc", [](Rig& r) {
    const uint64_t y = r.alloc(64);
    r.launch("usemalloc", 1, 1, {arg_u64(y)});
  }), "device-runtime-call");
}

VTEST(row_launch_calling_printf_is_refused) {
  expect_poisoned("printf", run_row("printf", [](Rig& r) { r.launch("useprintf", 1, 1, {arg_u64(r.y + 1024)}); }), "device-runtime-call");
}

VTEST(row_launch_with_atomics_on_several_threads_poisons) {
  ::unsetenv("VGPU_THREADS");
  expect_poisoned("atomics", run_row("atomics", [](Rig& r) {
    const uint64_t y = r.alloc(64);
    r.launch("atomadd", 4, 32, {arg_u64(y)});
  }), "atomics-on-several-threads");
}

VTEST(row_launch_with_atomics_in_block_order_is_cached) {
  RowResult r;
  {
    EnvGuard threads("VGPU_THREADS", "1");
    r = run_row("atomics1", [](Rig& rr) {
      const uint64_t y = rr.alloc(64);
      rr.launch("atomadd", 4, 32, {arg_u64(y)});
    });
  }
  expect_survives("atomics-in-order", r);
}

VTEST(row_cooperative_cluster_and_warp_size_launches_are_refused) {
  expect_poisoned("coop", run_row("coop", [](Rig& r) {
    const uint64_t y = r.alloc(4096);
    LaunchConfig c;
    c.block = {8, 1, 1};
    c.cooperative = true;
    r.launch_cfg("fixed", c, {arg_u64(y)});
  }), "cooperative");
  expect_poisoned("cluster", run_row("cluster", [](Rig& r) {
    const uint64_t y = r.alloc(4096);
    LaunchConfig c;
    c.grid = {2, 1, 1};
    c.block = {8, 1, 1};
    c.cluster = {2, 1, 1};
    r.launch_cfg("fixed", c, {arg_u64(y)});
  }, 7, "nvidia/h100"), "cluster");
}

VTEST(row_a_graph_instantiated_for_device_launch_refuses_its_kernels) {
  expect_poisoned("device-graph", run_row("devgraph", [](Rig& r) {
    LaunchConfig c;
    c.grid = {1, 1, 1};
    c.block = {8, 1, 1};
    c.current_graph_exec = 0x1234;   // what a kernel of a device-launch graph is told its own handle is
    r.launch_cfg("fixed", c, {arg_u64(r.y)});
  }), "device-graph");
}

VTEST(row_texture_objects_make_launches_uncacheable) {
  expect_poisoned("textures", run_row("textures", [](Rig& r) {
    r.dev.textures()[0x1000] = exec::TextureDesc{};
    const uint64_t y = r.alloc(4096);
    r.launch("fixed", 1, 8, {arg_u64(y)});
  }), "textures");
}

VTEST(row_environment_that_changes_what_runs_is_refused) {
  for (const char* var : {"VGPU_RACE", "VGPU_TRACE", "VGPU_MAX_STEPS", "VGPU_FAIL_LAUNCH", "VGPU_TIER_STATS"}) {
    const RowResult r = run_row(std::string("env") + var, [var](Rig& rr) {
      EnvGuard g(var, var == std::string("VGPU_MAX_STEPS") ? "100000000" : "0");
      const uint64_t y = rr.alloc(4096);
      rr.launch("fixed", 1, 8, {arg_u64(y)});
    });
    expect_poisoned(var, r, nullptr);
  }
}

VTEST(row_profiling_and_memory_tracing_refuse_launches) {
  expect_poisoned("profiling", run_row("prof", [](Rig& r) {
    profiling::set_enabled(true);
    const uint64_t y = r.alloc(4096);
    r.launch("fixed", 1, 8, {arg_u64(y)});
    profiling::set_enabled(false);
  }), "profiling");
  expect_poisoned("mem-trace", run_row("trace", [](Rig& r) {
    set_mem_trace([](char, uint64_t, uint64_t) {});
    const uint64_t y = r.alloc(4096);
    r.launch("fixed", 1, 8, {arg_u64(y)});
    set_mem_trace(nullptr);
  }), "mem-trace");
}

namespace {
struct ArmedFault final : MemoryManager::AccessFault {
  uint64_t armed = 1;
  ArmedFault() { load_pending = &armed; }
  uint64_t on_load(uint64_t, uint32_t, uint64_t v) override { return v; }
  uint64_t on_store(uint64_t, uint32_t, uint64_t v) override { return v; }
  uint64_t on_shared_load(uint64_t, uint32_t, uint64_t v) override { return v; }
  void on_read(uint64_t, uint8_t*, uint64_t) override {}
  uint64_t on_alu(uint64_t v, uint32_t) override { return v; }
  void on_copy(uint64_t, uint8_t*, uint64_t) override {}
};
}  // namespace

VTEST(row_an_armed_fault_refuses_launches) {
  static ArmedFault f;
  f.armed = 1;
  expect_poisoned("faults", run_row("faults", [](Rig& r) {
    r.mem().set_access_fault(&f);
    const uint64_t y = r.alloc(4096);
    r.launch("fixed", 1, 8, {arg_u64(y)});
    r.mem().set_access_fault(nullptr);
  }), "faults-armed");
}

VTEST(row_a_kernel_reading_smid_on_a_big_grid_poisons) {
  ::unsetenv("VGPU_THREADS");
  expect_poisoned("smid", run_row("smid", [](Rig& r) {
    const uint64_t y = r.alloc(4096);
    r.launch("smids", 400, 1, {arg_u64(y)});   // more blocks than the part has SMs
  }), "smid-by-host-thread");
}

VTEST(row_launch_without_a_symbol_table_is_refused) {
  expect_poisoned("nosyms", run_row("nosyms", [](Rig& r) {
    const uint64_t y = r.alloc(4096);
    LaunchConfig c;
    c.block = {8, 1, 1};
    r.dev.launch(*r.dev.get_function(r.mod, "fixed"), c, {arg_u64(y)}, nullptr);
  }), "unknown-symbols");
}

// The rows only the ticket sees: a host event or a device-side allocation during the launch, an exception, a module with
// no content hash, a second device.
namespace {
memo::KernelRef fake_ref(const std::string* name, const LaunchConfig* cfg, const std::vector<std::vector<uint8_t>>* args, const exec::LaunchEnv* env) {
  memo::KernelRef k;
  k.code = memo::hash_bytes("code", 4);
  k.code_ok = true;
  k.name = name;
  k.cfg = cfg;
  k.args = args;
  k.env = env;
  k.symbols_known = true;
  static const std::vector<uint64_t> none;
  k.module_globals = &none;
  return k;
}
}  // namespace

VTEST(rows_seen_between_begin_and_end) {
  const memo::Config cfg = base_config();
  const std::string name = "k";
  LaunchConfig lc;
  const exec::LaunchEnv env{};
  const auto run = [&](const std::function<void(Rig&, memo::DeviceMemo&, memo::Ticket&)>& between, const char* expect_reason) {
    Rig r(&cfg);
    const uint64_t y = r.alloc(4096);
    const std::vector<std::vector<uint8_t>> args{arg_u64(y)};
    const memo::KernelRef k = fake_ref(&name, &lc, &args, &env);
    memo::Ticket t;
    exec::LaunchStats st;
    const memo::Hash128 s0 = r.memo()->state();
    VCHECK(!r.memo()->begin(k, t, &st));
    VCHECK(t.active);
    between(r, *r.memo(), t);
    exec::LaunchStats ran;
    r.memo()->end(t, ran, 1000);
    const memo::Stats s = r.memo()->stats();
    if (!s.reasons.count(expect_reason)) std::fprintf(stderr, "no reason '%s'\n", expect_reason);
    VCHECK(s.reasons.count(expect_reason) == 1);
    VCHECK(r.memo()->state() != s0);
  };
  run([](Rig& r, memo::DeviceMemo&, memo::Ticket&) { r.mem().write(r.mem().live_ranges()[0].first, "x", 1); }, "concurrent-host-event");
  run([](Rig& r, memo::DeviceMemo&, memo::Ticket&) {
    memo::ScopeGuard g(memo::kEngine);   // a kernel's device-side allocation
    r.alloc(64);
  }, "device-side-alloc-or-free");
  run([](Rig& r, memo::DeviceMemo&, memo::Ticket&) {
    memo::ScopeGuard g(memo::kEngine);
    r.mem().free(r.mem().live_ranges()[0].first);
  }, "device-side-alloc-or-free");
  // A launch that threw.
  {
    Rig r(&cfg);
    const uint64_t y = r.alloc(4096);
    const std::vector<std::vector<uint8_t>> args{arg_u64(y)};
    const memo::KernelRef k = fake_ref(&name, &lc, &args, &env);
    memo::Ticket t;
    exec::LaunchStats st;
    r.memo()->begin(k, t, &st);
    const memo::Hash128 s0 = r.memo()->state();
    r.memo()->abort(t, "launch-threw");
    VCHECK(r.memo()->state() != s0);
    VCHECK(r.memo()->stats().reasons.count("launch-threw") == 1);
  }
  // Refused up front: no content hash, two devices, a second launch in flight.
  {
    Rig r(&cfg);
    const uint64_t y = r.alloc(4096);
    const std::vector<std::vector<uint8_t>> args{arg_u64(y)};
    memo::KernelRef k = fake_ref(&name, &lc, &args, &env);
    memo::Ticket t;
    exec::LaunchStats st;
    k.code_ok = false;
    r.memo()->begin(k, t, &st);
    VCHECK(!t.eligible);
    r.memo()->end(t, st, 1);
    k.code_ok = true;
    k.device_count = 2;
    r.memo()->begin(k, t, &st);
    VCHECK(!t.eligible);
    r.memo()->end(t, st, 1);
    k.device_count = 1;
    memo::Ticket first;
    r.memo()->begin(k, first, &st);
    VCHECK(first.eligible);
    memo::Ticket second;
    r.memo()->begin(k, second, &st);
    VCHECK(!second.eligible);   // the second launch overlapped the first
    r.memo()->end(second, st, 1);
    r.memo()->end(first, st, 1);
    const memo::Stats s = r.memo()->stats();
    VCHECK(s.reasons.count("unhashable-module") == 1);
    VCHECK(s.reasons.count("multi-device") == 1);
    VCHECK(s.reasons.count("concurrent-launch") == 1);
  }
}

// ---- write sets: what a launch could write -----------------------------------------------------------------------------------------

VTEST(a_kernel_that_writes_through_a_pointer_in_memory_is_captured_and_replayed) {
  const std::string dir = temp_dir("ptr");
  const memo::Config cfg = base_config(dir);
  const Program p = [](Rig& r, std::vector<std::string>& log) {
    const uint64_t table = r.alloc(64), target = r.alloc(4096);
    uint64_t t = target;
    r.mem().write(table, &t, 8);
    r.launch("viaptr", 1, 32, {arg_u64(table)});   // `target` is not an argument
    log.push_back("viaptr " + r.everything());
  };
  const Outcome off = run_program(p, nullptr);
  const Outcome cold = run_program(p, &cfg);
  const Outcome warm = run_program(p, &cfg);
  check_same(off, cold, "ptr cold");
  check_same(off, warm, "ptr warm");
  VCHECK_EQ(warm.stats.hits, 1u);
  remove_dir(dir);
}

VTEST(a_kernel_that_writes_a_module_global_is_captured_and_replayed) {
  const std::string dir = temp_dir("glob");
  const memo::Config cfg = base_config(dir);
  const Program p = [](Rig& r, std::vector<std::string>& log) {
    r.launch("setglobal", 1, 1, {arg_u32(41)});
    const uint64_t y = r.alloc(256);
    r.launch("readglobal", 1, 8, {arg_u64(y)});
    log.push_back("y " + r.everything());
    uint64_t g = 0, gs = 0;
    VCHECK(r.dev.global(r.mod, "gvar", &g, &gs));
    log.push_back("g " + std::to_string(r.words(g, 1)[0]));
  };
  const Outcome off = run_program(p, nullptr);
  const Outcome cold = run_program(p, &cfg);
  const Outcome warm = run_program(p, &cfg);
  check_same(off, cold, "glob cold");
  check_same(off, warm, "glob warm");
  VCHECK_EQ(warm.stats.hits, 2u);
  remove_dir(dir);
}

VTEST(a_write_set_of_a_uniform_run_is_stored_as_one_byte_and_replays) {
  const std::string dir = temp_dir("uniform");
  const memo::Config cfg = base_config(dir);
  const Program p = [](Rig& r, std::vector<std::string>& log) {
    const uint64_t y = r.alloc(1 << 20);
    for (int i = 0; i < 2; ++i) r.launch("fixed", 256, 128, {arg_u64(y)});   // 128 KiB of the same word
    log.push_back("y " + r.everything());
  };
  const Outcome off = run_program(p, nullptr);
  const Outcome cold = run_program(p, &cfg);
  const Outcome warm = run_program(p, &cfg);
  check_same(off, warm, "uniform warm");
  VCHECK_EQ(warm.stats.hits, 2u);
  remove_dir(dir);
}

// ---- verify mode ------------------------------------------------------------------------------------------------------------------------

VTEST(verify_mode_runs_every_hit_for_real_and_finds_no_difference) {
  const std::string dir = temp_dir("verify");
  memo::Config cfg = base_config(dir);
  const Outcome off = run_program(typical, nullptr);
  (void)run_program(typical, &cfg);
  cfg.verify = 2;   // count, do not abort
  const Outcome warm = run_program(typical, &cfg);
  check_same(off, warm, "verify");
  VCHECK(warm.stats.verify_checked >= 9);
  VCHECK_EQ(warm.stats.verify_mismatch, 0u);
  remove_dir(dir);
}

// ---- the mutants: each wrong rule must be caught by a check here --------------------------------------------------------------------------

namespace {

// Runs scenario `sc` (cold, then warm, then off) with the mutant on; true if a check of ours would fail.
struct Killed {
  bool by_answer = false;   // the warm run answered differently from a device without the cache
  bool by_hits = false;     // it hit where it must not, or missed where it must
};

// Two programs whose histories are the same up to the point they differ: run A cold, then B warm with the same directory
// and compare B against B without the cache.
Killed two_runs(const std::string& mutant, const Program& a, const Program& b, int mul_a, int mul_b, uint64_t expect_hits_in_b,
                bool hits_exact = true, const char* gpu_a = "nvidia/a10", const char* gpu_b = "nvidia/a10") {
  const std::string dir = temp_dir(("mut-" + mutant).c_str());
  memo::Config cfg = base_config(dir);
  cfg.mutant = mutant;
  (void)run_program(a, &cfg, mul_a, gpu_a);
  const Outcome warm = run_program(b, &cfg, mul_b, gpu_b);
  const Outcome truth = run_program(b, nullptr, mul_b, gpu_b);
  Killed k;
  k.by_answer = warm.log != truth.log;
  k.by_hits = hits_exact ? warm.stats.hits != expect_hits_in_b : warm.stats.hits > expect_hits_in_b;
  if (mutant.empty() && (k.by_answer || k.by_hits)) {   // the control: say why it is not clean
    std::fprintf(stderr, "control: by_answer=%d by_hits=%d hits=%llu expected %llu\n", k.by_answer, k.by_hits,
                 static_cast<unsigned long long>(warm.stats.hits), static_cast<unsigned long long>(expect_hits_in_b));
    for (const auto& l : warm.log) std::fprintf(stderr, "  warm:  %s\n", l.c_str());
    for (const auto& l : truth.log) std::fprintf(stderr, "  truth: %s\n", l.c_str());
  }
  remove_dir(dir);
  return k;
}

Program axpy_prog(uint32_t grid, uint32_t a, uint32_t x0, bool fixed_instead = false, uint32_t shared = 0) {
  return [=](Rig& r, std::vector<std::string>& log) {
    const uint64_t y = r.alloc(4096), x = r.alloc(4096);
    r.write_words(x, {x0, x0 + 1, x0 + 2, x0 + 3});
    if (fixed_instead) r.launch("fixed", grid, 4, {arg_u64(y)}, shared);
    else r.launch("axpy", grid, 4, {arg_u64(y), arg_u64(x), arg_u32(a)}, shared);
    log.push_back(r.everything());
  };
}

// For a mutant that should have been caught: the scenario that catches it, as (program in the first run, program in the
// second, modules, hits the second run may have).
bool caught(const std::string& m, bool mutate = true) {
  const std::string mut = mutate ? m : std::string();
  if (m == "drop_byte") {
    const Killed k = two_runs(mut, axpy_prog(1, 2, 1), axpy_prog(1, 2, 1), 7, 7, 1);
    return k.by_answer;
  }
  if (m == "key_no_grid") {   // the same call on the same history with a bigger grid
    const Killed k = two_runs(mut, axpy_prog(1, 2, 1, true), axpy_prog(2, 2, 1, true), 7, 7, 0);
    return k.by_answer || k.by_hits;
  }
  if (m == "key_no_args") {
    const Killed k = two_runs(mut, axpy_prog(1, 2, 1), axpy_prog(1, 3, 1), 7, 7, 0);
    return k.by_answer || k.by_hits;
  }
  if (m == "key_no_code") {   // same kernel name, same arguments, another module
    const Killed k = two_runs(mut, axpy_prog(1, 0, 1, true), axpy_prog(1, 0, 1, true), 7, 9, 0);
    return k.by_answer || k.by_hits;
  }
  if (m == "key_no_profile") {   // %nsmid is the part's SM count
    const Program who = [](Rig& r, std::vector<std::string>& log) {
      const uint64_t y = r.alloc(4096);
      r.launch("whoami", 1, 1, {arg_u64(y)});
      log.push_back("w " + std::to_string(r.words(y, 3)[1]));
    };
    const Killed k = two_runs(mut, who, who, 7, 7, 0, true, "nvidia/a10", "nvidia/h100");
    return k.by_answer || k.by_hits;
  }
  if (m == "key_no_state") {   // the same launch on a different memory state
    const Killed k = two_runs(mut, axpy_prog(1, 2, 1), axpy_prog(1, 2, 50), 7, 7, 0);
    return k.by_answer || k.by_hits;
  }
  if (m == "key_no_shared") {   // %dynamic_smem_size
    const Program who = [](Rig& r, std::vector<std::string>& log) {
      const uint64_t y = r.alloc(4096);
      r.launch("whoami", 1, 1, {arg_u64(y)}, 0);
      log.push_back("s " + std::to_string(r.words(y, 3)[2]));
    };
    const Program who2 = [](Rig& r, std::vector<std::string>& log) {
      const uint64_t y = r.alloc(4096);
      r.launch("whoami", 1, 1, {arg_u64(y)}, 1024);
      log.push_back("s " + std::to_string(r.words(y, 3)[2]));
    };
    const Killed k = two_runs(mut, who, who2, 7, 7, 0);
    return k.by_answer || k.by_hits;
  }
  if (m == "key_no_env") {   // blocks in order or not: the one environment fact in the key
    const Program at = [](Rig& r, std::vector<std::string>& log) {
      const uint64_t y = r.alloc(64);
      r.launch("atomadd", 2, 8, {arg_u64(y)});
      log.push_back("a " + std::to_string(r.words(y, 1)[0]));
    };
    ::setenv("VGPU_THREADS", "1", 1);
    const std::string dir = temp_dir("mut-env");
    memo::Config cfg = base_config(dir);
    cfg.mutant = mut;
    (void)run_program(at, &cfg);   // cacheable in block order
    ::unsetenv("VGPU_THREADS");
    const Outcome warm = run_program(at, &cfg);   // not in block order: must not hit
    remove_dir(dir);
    return warm.stats.hits != 0;
  }
  if (m == "no_write_event" || m == "no_fill_event") {   // the same history but for the bytes a host wrote
    const bool fill = m == "no_fill_event";
    const auto prog = [fill](uint32_t v) {
      return [=](Rig& r, std::vector<std::string>& log) {
        const uint64_t y = r.alloc(4096), x = r.alloc(4096);
        if (fill) {
          const uint8_t b = static_cast<uint8_t>(v);
          r.mem().fill(x, &b, 1, 16);
        } else {
          r.write_words(x, {v, v, v, v});
        }
        r.launch("axpy", 1, 4, {arg_u64(y), arg_u64(x), arg_u32(2)});
        log.push_back(r.everything());
      };
    };
    const Killed k = two_runs(mut, prog(1), prog(2), 7, 7, 0);
    return k.by_answer || k.by_hits;
  }
  if (m == "no_copy_event") {   // the same copy call on different sources
    const auto prog = [](bool from_first) {
      return [=](Rig& r, std::vector<std::string>& log) {
        const uint64_t s1 = r.alloc(4096), s2 = r.alloc(4096), x = r.alloc(4096), y = r.alloc(4096);
        r.write_words(s1, {1, 1, 1, 1});
        r.write_words(s2, {9, 9, 9, 9});
        r.mem().copy_to(r.mem(), x, from_first ? s1 : s2, 16);
        r.launch("axpy", 1, 4, {arg_u64(y), arg_u64(x), arg_u32(2)});
        log.push_back(r.everything());
      };
    };
    const Killed k = two_runs(mut, prog(true), prog(false), 7, 7, 0);
    return k.by_answer || k.by_hits;
  }
  if (m == "no_free_event") {   // x freed in one run and live in the other: the launch must fail in the first
    const auto prog = [](bool free_x) {
      return [=](Rig& r, std::vector<std::string>& log) {
        const uint64_t y = r.alloc(4096), x = r.alloc(4096);
        if (free_x) r.mem().free(x);
        r.launch("axpy", 1, 4, {arg_u64(y), arg_u64(x), arg_u32(2)});
        log.push_back(r.everything());
      };
    };
    const Killed k = two_runs(mut, prog(false), prog(true), 7, 7, 0);
    return k.by_answer || k.by_hits;
  }
  if (m == "no_alloc_event") {   // the same address, a smaller allocation: the launch must fail on the small one
    const auto prog = [](uint64_t bytes) {
      return [=](Rig& r, std::vector<std::string>& log) {
        const uint64_t out = r.alloc(64), y = r.alloc(bytes);
        r.launch("peek", 1, 1, {arg_u64(y), arg_u64(512), arg_u64(out)});   // reads byte 512 of y
        log.push_back(r.everything());
      };
    };
    const Killed k = two_runs(mut, prog(1024), prog(256), 7, 7, 0);
    return k.by_answer || k.by_hits;
  }
  if (m == "commutative_chain") {   // A then B against B then A: the next launch must tell them apart
    const auto prog = [](bool ab) {
      return [=](Rig& r, std::vector<std::string>& log) {
        const uint64_t y = r.alloc(4096), x = r.alloc(4096);
        r.write_words(x, {1, 2, 3, 4});
        r.write_words(y, {5, 6, 7, 8});
        const auto A = [&] { r.launch("axpy", 1, 4, {arg_u64(y), arg_u64(x), arg_u32(2)}); };
        const auto B = [&] { r.launch("axpy", 1, 4, {arg_u64(y), arg_u64(x), arg_u32(3)}); };
        if (ab) A(), B();
        else B(), A();
        r.launch("axpy", 1, 4, {arg_u64(y), arg_u64(x), arg_u32(1)});   // C, after either order
        log.push_back(r.everything());
      };
    };
    const Killed k = two_runs(mut, prog(true), prog(false), 7, 7, 0);
    return k.by_answer || k.by_hits;
  }
  if (m == "state_never_advances") {   // A1 then B against A2 then B
    const auto prog = [](uint32_t a) {
      return [=](Rig& r, std::vector<std::string>& log) {
        const uint64_t y = r.alloc(4096), x = r.alloc(4096);
        r.write_words(x, {1, 2, 3, 4});
        r.launch("axpy", 1, 4, {arg_u64(y), arg_u64(x), arg_u32(a)});
        r.launch("axpy", 1, 4, {arg_u64(y), arg_u64(x), arg_u32(3)});
        log.push_back(r.everything());
      };
    };
    const Killed k = two_runs(mut, prog(2), prog(5), 7, 7, 0);
    return k.by_answer || k.by_hits;
  }
  if (m == "no_cache_invalidation") {   // the kernel puts back what the page held before a host changed it
    const auto prog = [](Rig& r, std::vector<std::string>& log) {
      const uint64_t y = r.alloc(4096);
      r.launch("fixed", 1, 128, {arg_u64(y)});         // y[0..128) = 7
      r.write_words(y, {1, 2, 3, 4});                  // a host changes them
      r.launch("fixed", 1, 128, {arg_u64(y)});         // the kernel puts 7 back
      log.push_back(r.everything());
    };
    const Killed k = two_runs(mut, prog, prog, 7, 7, 2);
    return k.by_answer || k.by_hits;
  }
  if (m == "ptrs_never") {
    const auto prog = [](Rig& r, std::vector<std::string>& log) {
      const uint64_t table = r.alloc(64), target = r.alloc(4096);
      uint64_t t = target;
      r.mem().write(table, &t, 8);
      r.launch("viaptr", 1, 32, {arg_u64(table)});
      log.push_back(r.everything());
    };
    const Killed k = two_runs(mut, prog, prog, 7, 7, 1);
    return k.by_answer || k.by_hits;
  }
  if (m == "skip_module_globals") {
    const auto prog = [](Rig& r, std::vector<std::string>& log) {
      r.launch("setglobal", 1, 1, {arg_u32(41)});
      uint64_t g = 0, gs = 0;
      r.dev.global(r.mod, "gvar", &g, &gs);
      log.push_back(std::to_string(r.words(g, 1)[0]));
    };
    const Killed k = two_runs(mut, prog, prog, 7, 7, 1);
    return k.by_answer || k.by_hits;
  }
  if (m == "hit_zero_counters") {
    const std::string dir = temp_dir("mut-zc");
    memo::Config cfg = base_config(dir);
    cfg.mutant = mut;
    const Outcome cold = run_program(axpy_prog(1, 2, 1), &cfg);
    const Outcome warm = run_program(axpy_prog(1, 2, 1), &cfg);
    remove_dir(dir);
    return warm.stats.hits == 1 && warm.stats.replayed_instructions != cold.stats.recorded_instructions;
  }
  if (m == "event_budget_no_poison") {   // a host write the budget refuses is an event without its bytes: another run's bytes match it
    const std::string dir = temp_dir("mut-evb");
    memo::Config cfg = budget_config(dir, 0);
    cfg.mutant = mut;
    (void)run_program(axpy_prog(1, 2, 1), &cfg);
    const Outcome warm = run_program(axpy_prog(1, 2, 5), &cfg);
    const Outcome truth = run_program(axpy_prog(1, 2, 5), nullptr);
    remove_dir(dir);
    return warm.log != truth.log || warm.stats.hits != 0;
  }
  if (m == "dormant_keeps_history") {   // a cache that went quiet still hashes the host writes it is told about
    memo::Config cfg = budget_config("", 1000000000ull);
    cfg.dormant_after = 4;
    cfg.dormant_fixed_share = 0.0;
    cfg.mutant = mut;
    Rig r(&cfg);
    const uint64_t y = r.alloc(4096), x = r.alloc(4096);
    r.write_words(x, {1, 2, 3, 4});
    for (uint32_t i = 0; i < 8; ++i) r.launch("axpy", 1, 4, {arg_u64(y), arg_u64(x), arg_u32(i + 2)});
    if (r.stats().dormant_launches == 0) return false;   // (the scenario did not make it quiet: not a catch)
    const memo::Hash128 s0 = r.memo()->state();
    r.write_words(x, {9, 9, 9, 9});
    return r.memo()->state() != s0;   // moved: the write was hashed into a history nobody will read
  }
  // The poisoning rows: with the rule off, a launch that must not be cached is, and the second run hits after it.
  struct Row {
    const char* mutant;
    std::function<void(Rig&)> row;
    const char* env = nullptr;
  };
  static const std::vector<Row> rows = {
      {"no_poison_vmm", [](Rig& r) { r.mem().address_free(r.mem().reserve(2u << 20, 0), 2u << 20); }},
      {"no_poison_atomics", [](Rig& r) { r.launch("atomadd", 4, 32, {arg_u64(r.y + 2048)}); }},
      {"no_poison_device_calls", [](Rig& r) { r.launch("useprintf", 1, 1, {arg_u64(r.y + 1024)}); }},
      {"no_poison_smid", [](Rig& r) { r.launch("smids", 400, 1, {arg_u64(r.alloc(4096))}); }},
      {"no_poison_textures", [](Rig& r) {
         r.dev.textures()[0x1000] = exec::TextureDesc{};
         r.launch("fixed", 1, 8, {arg_u64(r.alloc(4096))});
       }},
  };
  for (const Row& row : rows) {
    if (m != row.mutant) continue;
    ::unsetenv("VGPU_THREADS");
    const std::string name = std::string("mut-") + m;
    const std::string dir = temp_dir(name.c_str());
    memo::Config cfg = base_config(dir);
    cfg.mutant = mut;
    memo::Stats before{};
    const auto prog = [&](Rig& r, std::vector<std::string>& log) {
      const uint64_t y = r.alloc(4096), x = r.alloc(4096);
      r.y = y, r.x = x;
      r.write_words(x, {1, 2, 3, 4, 5, 6, 7, 8});
      r.launch("axpy", 1, 8, {arg_u64(y), arg_u64(x), arg_u32(2)});
      before = r.stats();
      row.row(r);
      r.launch("axpy", 1, 8, {arg_u64(y), arg_u64(x), arg_u32(3)});
      r.launch("fixed", 1, 8, {arg_u64(y)});
      r.launch("fixed", 1, 8, {arg_u64(y)});
      log.push_back(r.everything());
    };
    (void)run_program(prog, &cfg);
    const Outcome warm = run_program(prog, &cfg);
    remove_dir(dir);
    return warm.stats.hits - before.hits != 0;   // hits after the row: a poisoned history has none
  }
  if (m == "no_poison_host_maps") {   // a kernel writes host memory; a host changes it behind the cache's back
    static std::vector<uint32_t> host(2048);
    const auto prog = [](Rig& r, std::vector<std::string>& log) {
      std::fill(host.begin(), host.end(), 0u);
      r.mem().map_host(reinterpret_cast<uint64_t>(host.data()), host.data(), host.size() * 4);
      r.launch("fixed", 1, 128, {arg_u64(reinterpret_cast<uint64_t>(host.data()))});
      host[0] = 1;                       // the host, directly
      r.launch("fixed", 1, 128, {arg_u64(reinterpret_cast<uint64_t>(host.data()))});
      log.push_back("host " + std::to_string(host[0]));
      r.mem().unmap_host(reinterpret_cast<uint64_t>(host.data()));
    };
    memo::Config cfg = base_config();
    cfg.mutant = mut;
    const Outcome on = run_program(prog, &cfg);
    const Outcome off = run_program(prog, nullptr);
    return on.log != off.log;
  }
  // The rows only the ticket sees.
  if (m == "no_poison_taint" || m == "no_poison_concurrent") {
    const memo::Config cfg = [&] {
      memo::Config c = base_config();
      c.mutant = mut;
      return c;
    }();
    Rig r(&cfg);
    const std::string name = "k";
    LaunchConfig lc;
    const exec::LaunchEnv env{};
    const uint64_t y = r.alloc(4096);
    const uint64_t other = r.alloc(4096);
    const std::vector<std::vector<uint8_t>> args{arg_u64(y)};
    const memo::KernelRef k = fake_ref(&name, &lc, &args, &env);
    memo::Ticket t;
    exec::LaunchStats st;
    r.memo()->begin(k, t, &st);
    if (m == "no_poison_taint") {
      memo::ScopeGuard g(memo::kEngine);
      r.alloc(64);
    } else {
      r.mem().write(other, "x", 1);   // a host thread's write to something the launch cannot reach
    }
    r.memo()->end(t, st, 1);
    const memo::Stats s = r.memo()->stats();
    return s.reasons.empty();   // the row left no trace: the launch was cached
  }
  if (m == "no_poison_faults") {
    static ArmedFault f;
    f.armed = 1;
    const memo::Config cfg = [&] {
      memo::Config c = base_config();
      c.mutant = mut;
      return c;
    }();
    Rig r(&cfg);
    r.mem().set_access_fault(&f);
    const uint64_t y = r.alloc(4096);
    r.launch("fixed", 1, 8, {arg_u64(y)});
    r.mem().set_access_fault(nullptr);
    return r.stats().recorded != 0;
  }
  std::fprintf(stderr, "no scenario for mutant %s\n", m.c_str());
  return false;
}

}  // namespace

// ---- the hashing of host writes is part of the budget ----------------------------------------------------------------------------

VTEST(host_writes_over_the_budget_poison_and_never_make_a_wrong_hit) {
  const std::string dir = temp_dir("evbudget");
  const memo::Config cfg = budget_config(dir, 0);   // every write is over budget
  const Outcome t1 = run_program(axpy_prog(1, 2, 1), nullptr);
  const Outcome t5 = run_program(axpy_prog(1, 2, 5), nullptr);
  const Outcome a = run_program(axpy_prog(1, 2, 1), &cfg);
  const Outcome b = run_program(axpy_prog(1, 2, 5), &cfg);
  const Outcome c = run_program(axpy_prog(1, 2, 1), &cfg);   // the same program as the first: a poisoned history still never matches
  check_same(t1, a, "over budget, first");
  check_same(t5, b, "over budget, other bytes");
  check_same(t1, c, "over budget, same bytes again");
  for (const Outcome* o : {&a, &b, &c}) {
    VCHECK(o->stats.events_skipped >= 1);
    VCHECK_EQ(o->stats.hits, 0u);
    VCHECK(o->stats.reasons.count("event-budget") == 1);
    VCHECK_EQ(o->stats.event_ns, 0u);   // nothing was hashed
    VCHECK(o->stats.recorded >= 1);     // (the launch itself was still recorded: it is the history that is poisoned)
  }
  remove_dir(dir);
}

VTEST(host_writes_within_the_budget_are_hashed_and_a_second_run_hits) {
  const std::string dir = temp_dir("evbudget-ok");
  const memo::Config cfg = budget_config(dir, 1000000000ull);
  const Outcome t1 = run_program(axpy_prog(1, 2, 1), nullptr);
  const Outcome t5 = run_program(axpy_prog(1, 2, 5), nullptr);
  const Outcome cold = run_program(axpy_prog(1, 2, 1), &cfg);
  const Outcome same = run_program(axpy_prog(1, 2, 1), &cfg);
  const Outcome other = run_program(axpy_prog(1, 2, 5), &cfg);
  check_same(t1, cold, "within budget, cold");
  check_same(t1, same, "within budget, warm");
  check_same(t5, other, "within budget, other bytes");
  VCHECK_EQ(cold.stats.events_skipped, 0u);
  VCHECK(cold.stats.event_ns > 0);
  VCHECK_EQ(same.stats.hits, 1u);    // the same bytes: the hit
  VCHECK_EQ(other.stats.hits, 0u);   // other bytes: none
  remove_dir(dir);
}

VTEST(a_big_write_over_the_budget_poisons_and_the_small_writes_after_it_are_hashed_again) {
  const std::string dir = temp_dir("evbudget-big");
  // 8 MiB at the cache's starting hash rate is 1.4 ms; the credit is 1 ms. A 64-byte write is 0.3 us.
  const memo::Config cfg = budget_config(dir, 1000000);
  const Program prog = [](Rig& r, std::vector<std::string>& log) {
    const uint64_t big = r.alloc(8u << 20), y = r.alloc(4096), x = r.alloc(4096);
    std::vector<uint8_t> bytes(8u << 20, 3);
    r.mem().write(big, bytes.data(), bytes.size());   // refused: poisons
    r.write_words(x, {1, 2, 3, 4});                   // hashed
    r.write_words(x, {5, 6, 7, 8});                   // hashed
    r.launch("axpy", 1, 4, {arg_u64(y), arg_u64(x), arg_u32(2)});
    r.launch("fixed", 1, 4, {arg_u64(y)});
    r.launch("fixed", 1, 4, {arg_u64(y)});
    r.launch("fixed", 1, 4, {arg_u64(y)});
    log.push_back(r.everything());
  };
  const Outcome truth = run_program(prog, nullptr);
  const Outcome cold = run_program(prog, &cfg);
  const Outcome warm = run_program(prog, &cfg);
  check_same(truth, cold, "big write, cold");
  check_same(truth, warm, "big write, warm");
  VCHECK_EQ(warm.stats.events_skipped, 1u);
  VCHECK_EQ(warm.stats.disk_hits, 0u);   // nothing carries over from the first run: the history is poisoned
  VCHECK(warm.stats.hits >= 1);          // but the loop of unchanged launches after the two small writes hits in the run
  VCHECK(warm.stats.event_ns > 0);       // the small writes were hashed (charged by estimate)
  remove_dir(dir);
}

VTEST(the_hashing_of_host_writes_stops_at_its_credit) {
  const memo::Config cfg = budget_config("", 10000000);   // 10 ms
  Rig r(&cfg);
  const uint64_t buf = r.alloc(1u << 20);
  const std::vector<uint8_t> bytes(1u << 20, 7);
  for (int i = 0; i < 700; ++i) r.mem().write(buf, bytes.data(), bytes.size());   // 700 MiB: tens of ms of hashing, if it were all done
  const memo::Stats s = r.stats();
  VCHECK(s.events_skipped > 0);
  VCHECK(s.event_ns > 0);
  VCHECK(s.event_ns < 30000000ull);   // (the credit and the last write's error, not the whole)
  VCHECK(s.events_skipped + 1 <= 700u);
}

VTEST(a_dormant_cache_keeps_no_history_and_the_answers_are_the_same) {
  memo::Config cfg = budget_config("", 1000000000ull);
  cfg.dormant_after = 4;            // quiet after 4 launches that did not hit, whatever they cost
  cfg.dormant_fixed_share = 0.0;
  const Program prog = [](Rig& r, std::vector<std::string>& log) {
    const uint64_t y = r.alloc(4096), x = r.alloc(4096);
    r.write_words(x, {1, 2, 3, 4});
    for (uint32_t i = 0; i < 8; ++i) r.launch("axpy", 1, 4, {arg_u64(y), arg_u64(x), arg_u32(i + 2)});   // every launch changes y: no hit
    log.push_back("before " + r.everything());
    const memo::Hash128 s0 = r.memo() ? r.memo()->state() : memo::Hash128{};
    r.write_words(x, {9, 9, 9, 9});    // a host write after the cache went quiet
    const memo::Hash128 s1 = r.memo() ? r.memo()->state() : memo::Hash128{};
    log.push_back(std::string("history ") + (s0 == s1 ? "kept still" : "moved"));
    r.launch("axpy", 1, 4, {arg_u64(y), arg_u64(x), arg_u32(3)});
    log.push_back("after " + r.everything());
  };
  const Outcome off = run_program(prog, nullptr);
  const Outcome on = run_program(prog, &cfg);
  VCHECK(on.stats.dormant_launches >= 1);
  VCHECK(on.log.size() == off.log.size());
  for (size_t i = 0; i < on.log.size() && i < off.log.size(); ++i) {
    if (off.log[i].rfind("history", 0) == 0) VCHECK_EQ(on.log[i], std::string("history kept still"));   // (without a cache there is nothing to compare)
    else VCHECK_EQ(on.log[i], off.log[i]);
  }
}

VTEST(every_mutant_is_caught) {
  ::unsetenv("VGPU_THREADS");
  const char* mutants[] = {"drop_byte",        "key_no_grid",         "key_no_args",         "key_no_code",       "key_no_profile",
                           "key_no_state",     "key_no_env",          "key_no_shared",       "no_write_event",    "no_fill_event",
                           "no_copy_event",    "no_free_event",       "no_alloc_event",      "commutative_chain", "state_never_advances",
                           "no_cache_invalidation", "ptrs_never",     "skip_module_globals", "hit_zero_counters", "no_poison_vmm",
                           "no_poison_atomics", "no_poison_device_calls", "no_poison_smid",   "no_poison_textures", "no_poison_host_maps",
                           "no_poison_taint",  "no_poison_concurrent", "no_poison_faults",  "event_budget_no_poison", "dormant_keeps_history"};
  int missed = 0;
  for (const char* m : mutants) {
    const bool c = caught(m);
    if (!c) {
      std::fprintf(stderr, "MUTANT NOT CAUGHT: %s\n", m);
      ++missed;
    } else {
      std::fprintf(stderr, "mutant caught: %s\n", m);
    }
  }
  VCHECK_EQ(missed, 0);
}

VTEST(the_unmutated_cache_passes_the_same_scenarios) {
  // The scenarios that catch the mutants compare a cache against a device without one; with no mutant applied every one of
  // them must come out clean, or "caught" would only mean the scenario is broken.
  ::unsetenv("VGPU_THREADS");
  const char* names[] = {"drop_byte",        "key_no_grid",         "key_no_args",         "key_no_code",       "key_no_profile",
                         "key_no_state",     "key_no_env",          "key_no_shared",       "no_write_event",    "no_fill_event",
                         "no_copy_event",    "no_free_event",       "no_alloc_event",      "commutative_chain", "state_never_advances",
                         "no_cache_invalidation", "ptrs_never",     "skip_module_globals", "hit_zero_counters", "no_poison_vmm",
                         "no_poison_atomics", "no_poison_device_calls", "no_poison_smid",   "no_poison_textures", "no_poison_host_maps",
                         "no_poison_taint",  "no_poison_concurrent", "no_poison_faults",  "event_budget_no_poison", "dormant_keeps_history"};
  int broken = 0;
  for (const char* m : names)
    if (caught(m, false)) {
      std::fprintf(stderr, "SCENARIO FAILS WITH NO MUTANT: %s\n", m);
      ++broken;
    }
  VCHECK_EQ(broken, 0);
}

VTEST_MAIN
