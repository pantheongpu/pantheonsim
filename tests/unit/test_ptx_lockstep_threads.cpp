// How many host threads a lockstep launch of a small grid takes on the PTX engine (the SASS engine's twin is test_lockstep_shape and
// e2e_lockstep_blocks).
//
// A grid of fewer than 16 blocks whose kernel polls global memory runs on ONE host thread, every block resident from the start, in a fixed
// order, when the kernel takes locks (the answer of RXMesh's patch locks depends on the order blocks start in). A kernel whose waits are
// only loads (CUB's radix-sort look-back, a flag chain) gives the same answer in any order, so its small grid uses the host threads, two
// blocks to a thread at least, every block resident from the start. The PTX engine used to give every small grid one thread.
//
// These launch three PTX kernels of 10 to 12 blocks with VGPU_THREADS=4 and read the lockstep log (VGPU_LOCKSTEP_LOG=<file>):
//   flag_wait   the last block stores a flag, the others spin on it with ld.volatile        (loads only)    -> 2 or more host threads
//   lock_chain  every block takes a lock with atom.exch, adds to a counter, releases it     (an atomic)     -> 1 host thread
//   both        a load-polling loop and, after it, a lock loop                             (both)          -> 1 host thread
// and VGPU_LOCKSTEP=all, which runs every launch this way to check that nothing depends on the order blocks start in, keeps one thread
// for the same small grid. Each kernel also has to give the right answer.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

namespace {

const char* kPtx = R"(.version 8.3
.target sm_86
.address_size 64

.visible .entry flag_wait(.param .u64 pflag, .param .u64 pout)
{
    .reg .pred %p<4>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [pflag];
    ld.param.u64 %rd2, [pout];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %nctaid.x;
    sub.u32 %r3, %r2, 1;
    setp.eq.u32 %p1, %r1, %r3;
    @%p1 bra PRODUCE;
WAIT:
    ld.volatile.global.u32 %r4, [%rd1];
    setp.eq.u32 %p2, %r4, 0;
    @%p2 bra WAIT;
    mul.wide.u32 %rd3, %r1, 4;
    add.u64 %rd4, %rd2, %rd3;
    st.global.u32 [%rd4], %r4;
    ret;
PRODUCE:
    mov.u32 %r5, 77;
    st.global.u32 [%rd1], %r5;
    ret;
}

// pout[0] counts the blocks that went through the lock; pflag is the lock word.
.visible .entry lock_chain(.param .u64 pflag, .param .u64 pout)
{
    .reg .pred %p<4>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [pflag];
    ld.param.u64 %rd2, [pout];
LOCK:
    atom.global.exch.b32 %r4, [%rd1], 1;
    setp.ne.u32 %p2, %r4, 0;
    @%p2 bra LOCK;
    ld.global.u32 %r5, [%rd2];
    add.u32 %r5, %r5, 1;
    st.global.u32 [%rd2], %r5;
    membar.gl;
    st.global.u32 [%rd1], 0;
    ret;
}

// The last block stores the flag; the others wait for it with a load, then take the lock.
.visible .entry both(.param .u64 pflag, .param .u64 pout)
{
    .reg .pred %p<4>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [pflag];
    ld.param.u64 %rd2, [pout];
    add.u64 %rd5, %rd1, 8;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %nctaid.x;
    sub.u32 %r3, %r2, 1;
    setp.eq.u32 %p1, %r1, %r3;
    @%p1 bra PRODUCE;
WAIT:
    ld.volatile.global.u32 %r4, [%rd1];
    setp.eq.u32 %p2, %r4, 0;
    @%p2 bra WAIT;
    bra LOCK;
PRODUCE:
    mov.u32 %r6, 1;
    st.global.u32 [%rd1], %r6;
LOCK:
    atom.global.exch.b32 %r4, [%rd5], 1;
    setp.ne.u32 %p2, %r4, 0;
    @%p2 bra LOCK;
    ld.global.u32 %r7, [%rd2];
    add.u32 %r7, %r7, 1;
    st.global.u32 [%rd2], %r7;
    membar.gl;
    st.global.u32 [%rd5], 0;
    ret;
}
)";

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

struct Run {
  std::vector<uint32_t> out;
  int host_threads = 0;   // as the lockstep log says; 0: the launch did not take the lockstep path
};

// Launches `kernel` on `blocks` one-thread blocks, all resident, with the environment given; the lockstep log is read back.
Run launch(const char* kernel, unsigned blocks, const char* lockstep) {
  char path[] = "/tmp/vgpu_ptx_lockstep_threads_XXXXXX";
  const int fd = ::mkstemp(path);
  if (fd < 0) throw vtest::Failure("no temporary file for the lockstep log");
  ::close(fd);
  ::setenv("VGPU_THREADS", "4", 1);
  ::setenv("VGPU_LOCKSTEP_LOG", path, 1);
  if (lockstep) ::setenv("VGPU_LOCKSTEP", lockstep, 1);
  else ::unsetenv("VGPU_LOCKSTEP");

  MemoryManager mem{1 << 24};
  DeviceProfile prof = load_gpu("nvidia/a10");
  auto m = ptx::parse(kPtx);
  const ptx::EntryFn* fn = nullptr;
  for (const auto& e : m.entries)
    if (e.name == kernel) fn = &e;
  if (!fn) throw vtest::Failure(std::string("no kernel ") + kernel);
  const uint64_t flag = mem.alloc(256), out = mem.alloc(256);
  const uint32_t zeros[64] = {};
  mem.write(flag, zeros, sizeof zeros);
  mem.write(out, zeros, sizeof zeros);
  LaunchConfig cfg;
  cfg.grid = {blocks, 1, 1};
  cfg.block = {1, 1, 1};
  cfg.resident_blocks = blocks;   // the device holds them all at once
  cfg.max_steps = 50'000'000;     // a run that does not interleave the blocks fails instead of hanging
  exec::launch(*fn, cfg, {arg_u64(flag), arg_u64(out)}, mem, prof);

  Run r;
  r.out.resize(64);
  mem.read(out, r.out.data(), 64 * 4);
  std::ifstream in(path);
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string log = ss.str();
  const std::string tag = std::string("kernel ") + kernel + ",";
  if (const size_t at = log.find(tag); at != std::string::npos) {
    const size_t host = log.find("blocks, ", at);
    if (host != std::string::npos) r.host_threads = std::atoi(log.c_str() + host + 8);
  }
  ::unlink(path);
  ::unsetenv("VGPU_THREADS");
  ::unsetenv("VGPU_LOCKSTEP_LOG");
  ::unsetenv("VGPU_LOCKSTEP");
  return r;
}

}  // namespace

VTEST(a_small_grid_that_waits_on_loads_uses_the_host_threads_on_the_ptx_engine) {
  const Run r = launch("flag_wait", 10, nullptr);
  VCHECK(r.host_threads >= 2);
  VCHECK(r.host_threads <= 4);
  for (unsigned b = 0; b < 9; ++b) VCHECK_EQ(r.out[b], 77u);   // every consumer saw the producer's flag
}

VTEST(a_small_grid_that_takes_a_lock_stays_on_one_thread_on_the_ptx_engine) {
  const Run r = launch("lock_chain", 12, nullptr);
  VCHECK_EQ(r.host_threads, 1);
  VCHECK_EQ(r.out[0], 12u);   // every block went through the lock once
}

VTEST(a_kernel_with_a_load_poll_and_a_lock_loop_stays_on_one_thread) {
  // (a kernel that takes a lock anywhere keeps the fixed order, whatever its other loops wait on)
  const Run r = launch("both", 10, nullptr);
  VCHECK_EQ(r.host_threads, 1);
  VCHECK_EQ(r.out[0], 10u);
}

VTEST(vgpu_lockstep_all_keeps_a_small_grid_on_one_thread) {
  // (VGPU_LOCKSTEP=all checks that nothing depends on the order blocks start in: it keeps the fixed order for every small grid)
  const Run r = launch("flag_wait", 10, "all");
  VCHECK_EQ(r.host_threads, 1);
  for (unsigned b = 0; b < 9; ++b) VCHECK_EQ(r.out[b], 77u);
}

VTEST_MAIN
