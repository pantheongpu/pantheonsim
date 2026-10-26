// The PTX interpreter keeps a block's state between blocks and launches (a thread's pool of BlockState: shared memory, barriers, the warps'
// register vectors, tensor memory) and takes the environment it reads from one pass the runtime made (LaunchConfig::env).
//
// A block must start from nothing whatever ran before it on the thread. Each kernel here is run on a thread of its own (the pool is
// empty: the reference), then in a long sequence of different kernels on one thread (every block a reuse of what the kernel before it
// left): the shared memory a kernel never wrote reads zero, a named barrier starts with no arrivals, a block's clock starts at zero,
// blocks of different warp counts follow one another. Memory and every counter must match the reference. The environment half:
// a launch reads VGPU_MAX_STEPS from the pass it is given, not from the process, and from the process when it is given none.
// And the launch's parameters, which are looked up by name (a kernel that takes a parameter's address, or the general path's loads) from the
// kernel's own list and not from a table made for every launch.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/exec/launch_env.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

namespace {

const char* kHeader = ".version 8.3\n.target sm_86\n.address_size 64\n";

// Every kernel stores one word a thread to out[tid].
const char* kPrelude = R"(
    .reg .pred %p<4>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd4, %r1, 4;
    add.s64 %rd6, %rd2, %rd4;
)";

std::string kernel(const std::string& name, const std::string& decl, const std::string& body) {
  return std::string(kHeader) + ".visible .entry " + name + "(.param .u64 out)\n{\n" + decl + kPrelude + body + "\n    ret;\n}\n";
}

// Writes shared memory, leaves registers non-zero, and passes a block barrier.
const std::string kDirty = kernel("dirty", "    .shared .align 4 .b8 buf[1024];\n", R"(
    mov.u64 %rd3, buf;
    add.s64 %rd5, %rd3, %rd4;
    add.u32 %r2, %r1, 1000;
    st.shared.u32 [%rd5], %r2;
    mov.u32 %r10, 77;
    mov.u32 %r11, 78;
    bar.sync 0;
    ld.shared.u32 %r3, [%rd5];
    st.global.u32 [%rd6], %r3;
)");

// Reads shared memory it never wrote (and the word after the end of what the kernel before it wrote, in the same place).
const std::string kClean = kernel("clean", "    .shared .align 4 .b8 buf[256];\n", R"(
    mov.u64 %rd3, buf;
    add.s64 %rd5, %rd3, %rd4;
    ld.shared.u32 %r3, [%rd5];
    add.u32 %r4, %r3, 5;
    st.global.u32 [%rd6], %r4;
)");

// A barrier with a thread count that two warps arrive at (one of them has stored the word the other reads), and bar.red: the barrier's
// arrival count and the reduction's round carry nothing over.
const std::string kCounted = kernel("counted", "    .shared .align 4 .b8 buf[16];\n", R"(
    mov.u64 %rd3, buf;
    mov.u32 %r7, 0;
    setp.ge.u32 %p1, %r1, 64;
    @%p1 bra SKIP;
    setp.eq.u32 %p3, %r1, 0;
    @%p3 st.shared.u32 [%rd3], 7;
    bar.sync 1, 64;
    ld.shared.u32 %r7, [%rd3];
SKIP:
    setp.lt.u32 %p2, %r1, 100;
    bar.red.popc.u32 %r5, 0, %p2;
    add.u32 %r6, %r5, %r1;
    add.u32 %r6, %r6, %r7;
    st.global.u32 [%rd6], %r6;
)");

// A warp arrives at a counted barrier and goes on, and the block ends with the barrier half full: the next block must not find it so.
const std::string kArrive = kernel("arrive", "", R"(
    setp.lt.u32 %p1, %r1, 32;
    @!%p1 bra SKIP;
    bar.arrive 1, 64;
SKIP:
    add.u32 %r6, %r1, 9;
    st.global.u32 [%rd6], %r6;
)");

// The block's clock counts from the block's first instruction.
const std::string kClock = kernel("clock", "", R"(
    mov.u64 %rd3, %clock64;
    cvt.u32.u64 %r5, %rd3;
    st.global.u32 [%rd6], %r5;
)");

// Many warps, a barrier in the middle.
const std::string kBig = kernel("big", "", R"(
    mul.lo.u32 %r2, %r1, 3;
    bar.sync 0;
    add.u32 %r3, %r2, 1;
    st.global.u32 [%rd6], %r3;
)");

struct Case {
  const std::string* ptx;
  uint32_t threads;
  const char* name;
};
const Case kCases[] = {
    {&kDirty, 256, "dirty"}, {&kClean, 64, "clean"}, {&kCounted, 128, "counted"}, {&kClock, 32, "clock"}, {&kBig, 512, "big"},
    {&kClean, 32, "clean (one warp)"}, {&kDirty, 96, "dirty (three warps)"}, {&kArrive, 128, "arrive"},
};

struct Outcome {
  std::vector<uint32_t> out;
  std::vector<uint64_t> counters;
};

Outcome run(const Case& c, bool* failed) {
  Outcome o;
  setenv("VGPU_THREADS", "1", 1);   // the blocks run on this thread, one after another
  MemoryManager mem{1 << 20};
  DeviceProfile prof = load_gpu("nvidia/a10");
  static std::vector<std::pair<const std::string*, ptx::Module>> parsed;
  const ptx::Module* mod = nullptr;
  for (auto& p : parsed)
    if (p.first == c.ptx) mod = &p.second;
  if (!mod) {
    parsed.emplace_back(c.ptx, ptx::parse(*c.ptx));
    mod = &parsed.back().second;
  }
  constexpr uint32_t kBlocks = 3;
  const uint64_t total = uint64_t{kBlocks} * c.threads;
  const uint64_t pout = mem.alloc(c.threads * 4 + 64);
  LaunchConfig cfg;
  cfg.grid = {kBlocks, 1, 1};
  cfg.block = {c.threads, 1, 1};
  std::vector<uint8_t> arg(8);
  std::memcpy(arg.data(), &pout, 8);
  try {
    exec::LaunchStats st = exec::launch(mod->entries[0], cfg, {arg}, mem, prof);
    const auto* w = reinterpret_cast<const uint64_t*>(&st);
    o.counters.assign(w, w + exec::kCounterWords);
    o.counters.insert(o.counters.end(), st.inst_by_opcode.begin(), st.inst_by_opcode.end());
  } catch (const Error& e) {
    std::fprintf(stderr, "%s: %s\n", c.name, e.what());
    *failed = true;
  }
  (void)total;
  o.out.resize(c.threads);
  mem.read(pout, o.out.data(), c.threads * 4);
  unsetenv("VGPU_THREADS");
  return o;
}

Outcome on_a_new_thread(const Case& c) {
  Outcome o;
  bool failed = false;
  std::thread t([&] { o = run(c, &failed); });
  t.join();
  if (failed) throw vtest::Failure(std::string(c.name) + " failed to run");
  return o;
}

}  // namespace

VTEST(a_block_starts_from_nothing_after_other_kernels_on_the_thread) {
  const size_t n = sizeof kCases / sizeof kCases[0];
  std::vector<Outcome> ref;
  for (const Case& c : kCases) ref.push_back(on_a_new_thread(c));
  // The references say what they should (the kernels are not all the same trivial thing).
  VCHECK_EQ(ref[1].out[3], 5u);                // clean: shared memory nothing wrote reads zero
  VCHECK_EQ(ref[0].out[3], 1003u);             // dirty: its own words
  VCHECK_EQ(ref[2].out[7], 100u + 7u + 7u);    // counted: 100 lanes voted, and the word the other warp stored
  VCHECK_EQ(ref[7].out[3], 12u);               // arrive
  VCHECK_EQ(ref[4].out[300], 901u);
  // The same kernels, in orders that put every one after every other, all on this thread.
  const size_t order[] = {0, 1, 2, 3, 4, 1, 3, 5, 0, 6, 1, 2, 2, 3, 4, 0, 5, 6, 1, 4, 3, 2, 6, 5, 0, 0, 1, 1, 3, 3,
                          7, 2, 7, 7, 2, 3, 7, 0, 2};
  for (size_t k : order) {
    bool failed = false;
    const Outcome got = run(kCases[k], &failed);
    VCHECK(!failed);
    const std::string where = std::string(kCases[k].name) + " (case " + std::to_string(k) + ")";
    if (got.out != ref[k].out) throw vtest::Failure(where + ": memory differs from a block on a new thread");
    if (got.counters != ref[k].counters) throw vtest::Failure(where + ": the counters differ from a block on a new thread");
  }
  (void)n;
}

namespace {

// A loop of a few hundred instructions.
const std::string kLoop = kernel("spin", "", R"(
    mov.u32 %r5, 0;
L:
    add.u32 %r5, %r5, 1;
    setp.lt.u32 %p1, %r5, 200;
    @%p1 bra L;
    st.global.u32 [%rd6], %r5;
)");

std::string launch_error(const exec::LaunchEnv* env) {
  MemoryManager mem{1 << 20};
  DeviceProfile prof = load_gpu("nvidia/a10");
  static const ptx::Module mod = ptx::parse(kLoop);
  const uint64_t pout = mem.alloc(256);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  cfg.env = env;
  std::vector<uint8_t> arg(8);
  std::memcpy(arg.data(), &pout, 8);
  try {
    exec::launch(mod.entries[0], cfg, {arg}, mem, prof);
  } catch (const Error& e) {
    return e.what();
  }
  return "";
}

}  // namespace

VTEST(a_launch_reads_the_environment_it_is_given_or_the_process_s) {
  unsetenv("VGPU_MAX_STEPS");
  VCHECK_EQ(launch_error(nullptr), std::string(""));
  // Given a pass that names a budget the loop exceeds: the launch stops, though the process says nothing.
  exec::LaunchEnv tight;
  tight.max_steps = "50";
  VCHECK(launch_error(&tight).find("step budget") != std::string::npos);
  // The process names the budget, the pass given does not: the pass is the launch's environment (the runtime read the process's before).
  setenv("VGPU_MAX_STEPS", "50", 1);
  exec::LaunchEnv none;
  VCHECK_EQ(launch_error(&none), std::string(""));
  // No pass given: the launch reads the process's.
  VCHECK(launch_error(nullptr).find("step budget") != std::string::npos);
  unsetenv("VGPU_MAX_STEPS");
}

namespace {

// Takes the address of its third parameter by name, and reads it through the register (the window the parameters live in): the value of
// `third`, not of the first or the second. Each of the others is read by name too (VGPU_FASTPATH=0 loads by name, not by position).
const std::string kParams = std::string(kHeader) + R"(
.visible .entry params(.param .u32 first, .param .u64 out, .param .u32 third, .param .u32 fourth)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.b64 %rd3, third;
    ld.param.u32 %r1, [%rd3];
    ld.param.u32 %r2, [first];
    ld.param.u32 %r3, [fourth];
    st.global.u32 [%rd2], %r1;
    st.global.u32 [%rd2+4], %r2;
    st.global.u32 [%rd2+8], %r3;
    ret;
}
)";

uint32_t u32(std::vector<uint8_t> b, MemoryManager& mem, uint64_t at) {
  (void)b;
  uint32_t v = 0;
  mem.read(at, &v, 4);
  return v;
}

}  // namespace

VTEST(parameters_are_found_by_name_and_by_position) {
  static const ptx::Module mod = ptx::parse(kParams);
  for (const bool fast : {true, false}) {
    setenv("VGPU_FASTPATH", fast ? "1" : "0", 1);
    MemoryManager mem{1 << 20};
    DeviceProfile prof = load_gpu("nvidia/a10");
    const uint64_t pout = mem.alloc(256);
    auto bytes = [](const void* p, size_t n) {
      std::vector<uint8_t> b(n);
      std::memcpy(b.data(), p, n);
      return b;
    };
    const uint32_t first = 111, third = 333, fourth = 444;
    LaunchConfig cfg;
    cfg.block = {1, 1, 1};
    exec::launch(mod.entries[0], cfg, {bytes(&first, 4), bytes(&pout, 8), bytes(&third, 4), bytes(&fourth, 4)}, mem, prof);
    VCHECK_EQ(u32({}, mem, pout), 333u);       // the third, through its address
    VCHECK_EQ(u32({}, mem, pout + 4), 111u);   // the first
    VCHECK_EQ(u32({}, mem, pout + 8), 444u);   // the fourth
  }
  unsetenv("VGPU_FASTPATH");
}

VTEST_MAIN
