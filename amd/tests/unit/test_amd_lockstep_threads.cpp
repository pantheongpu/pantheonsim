// How many host threads a lockstep launch of a small grid takes on the AMD engine (the NVIDIA engines' twins are test_lockstep_shape,
// test_ptx_lockstep_threads and e2e_lockstep_blocks).
//
// A grid of fewer than 16 work-groups whose kernel polls global memory runs on ONE host thread, every group resident from the start, in a
// fixed order, when the kernel takes locks (the answer of RXMesh's patch locks depends on the order groups start in). A kernel whose waits
// are only loads (a decoupled look-back, a flag chain) gives the same answer in any order, so its small grid uses the host threads, two
// groups to a thread at least, every group resident from the start. The AMD engine used to give every small grid one thread.
//
// The kernels are amd/tests/data/lockstep_threads.c, launched with VGPU_THREADS=4 and the lockstep log (VGPU_LOCKSTEP_LOG=<file>) read back:
//   flag_wait   the last group stores a flag, the others spin on it with loads           (loads only)  -> 2 or more host threads
//   lock_chain  every group takes a lock with an exchange, counts itself, lets go        (an atomic)   -> 1 host thread
//   both        a load-polling loop and, after it, a lock loop                           (both)        -> 1 host thread
// and VGPU_LOCKSTEP=all, which runs every launch this way to check that nothing depends on the order groups start in, keeps one thread for
// the same small grid. Each kernel also has to give the right answer.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_decode_cache.hpp"
#include "vgpu/amd_exec.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

amd::CodeObject object() {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/lockstep_threads." + vtest::amd_target() + ".o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

uint64_t kernargs(MemoryManager& mem, const amd::Kernel& k, const std::vector<uint64_t>& values) {
  std::vector<uint8_t> buf(k.kernarg_size, 0);
  for (size_t i = 0; i < values.size() && i < k.args.size(); ++i) {
    const amd::KernelArg& a = k.args[i];
    for (uint32_t b = 0; b < a.size && b < 8; ++b) buf[a.offset + b] = static_cast<uint8_t>(values[i] >> (8 * b));
  }
  const uint64_t p = mem.alloc(buf.empty() ? 1 : buf.size());
  if (!buf.empty()) mem.write(p, buf.data(), buf.size());
  return p;
}

struct Run {
  std::vector<uint32_t> out;   // the first 64 words of the output buffer
  int host_threads = 0;        // as the lockstep log says; 0: the launch did not take the lockstep path
};

// Launches `kernel` on `groups` one-wave groups, all resident, under the given VGPU_LOCKSTEP (null: the default).
Run launch(const char* kernel, unsigned groups, const char* lockstep) {
  char path[] = "/tmp/vgpu_amd_lockstep_threads_XXXXXX";
  const int fd = ::mkstemp(path);
  if (fd < 0) throw vtest::Failure("no temporary file for the lockstep log");
  ::close(fd);
  ::setenv("VGPU_THREADS", "4", 1);
  ::setenv("VGPU_LOCKSTEP_LOG", path, 1);
  if (lockstep) ::setenv("VGPU_LOCKSTEP", lockstep, 1);
  else ::unsetenv("VGPU_LOCKSTEP");

  const amd::CodeObject o = object();
  const amd::Kernel* k = amd::find_kernel(o, kernel);
  if (!k) throw vtest::Failure(std::string("no kernel named ") + kernel);
  MemoryManager mem(64ull << 20);
  const uint64_t flag = mem.alloc(256), out = mem.alloc(256);
  const uint32_t zeros[64] = {};
  mem.write(flag, zeros, sizeof zeros);
  mem.write(out, zeros, sizeof zeros);
  // flag_wait(flag, out, groups); lock_chain(lock, count, groups); both(flag, count, groups)
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = kernargs(mem, *k, {flag, out, groups});
  d.groups[0] = groups;
  d.group_size[0] = 64;
  d.resident_groups = groups;   // the device holds them all at once
  amd::execute(d, mem);

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

VTEST(a_small_grid_that_waits_on_loads_uses_the_host_threads_on_the_amd_engine) {
  const Run r = launch("flag_wait", 10, nullptr);
  VCHECK(r.host_threads >= 2);
  VCHECK(r.host_threads <= 4);
  for (unsigned b = 0; b < 9; ++b) VCHECK_EQ(r.out[b], 77u);   // every consumer saw the producer's flag
}

VTEST(a_small_grid_that_takes_a_lock_stays_on_one_thread_on_the_amd_engine) {
  const Run r = launch("lock_chain", 12, nullptr);
  VCHECK_EQ(r.host_threads, 1);
  VCHECK_EQ(r.out[0], 12u);   // every group went through the lock once
}

VTEST(a_kernel_with_a_load_poll_and_a_lock_loop_stays_on_one_thread_on_the_amd_engine) {
  // (a kernel that takes a lock anywhere keeps the fixed order, whatever its other loops wait on)
  const Run r = launch("both", 10, nullptr);
  VCHECK_EQ(r.host_threads, 1);
  VCHECK_EQ(r.out[0], 10u);
}

VTEST(vgpu_lockstep_all_keeps_a_small_grid_on_one_thread_on_the_amd_engine) {
  // (VGPU_LOCKSTEP=all checks that nothing depends on the order groups start in: it keeps the fixed order for every small grid)
  const Run r = launch("flag_wait", 10, "all");
  VCHECK_EQ(r.host_threads, 1);
  for (unsigned b = 0; b < 9; ++b) VCHECK_EQ(r.out[b], 77u);
}

VTEST_MAIN
