// A dispatch whose host threads cannot be started. execute() spreads a grid's work-groups over host threads; where the
// host refuses one (EAGAIN at a process or user limit, or no address space left) a std::thread that never started was
// destroyed joinable and the whole process died in std::terminate, which a loaded shared machine showed as a launch
// that ended the program. The ranges of work-groups are independent, so now the range whose thread was refused runs on
// the thread that launched, and the dispatch finishes with the answers it would have had. (The limit is RLIMIT_NPROC;
// a process allowed past it, root for one, cannot make the host refuse, and the test says so and passes.)
#include <sys/resource.h>

#include <cstdio>
#include <atomic>
#include <cstdlib>
#include <system_error>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include "../../src/gcn_pool.hpp"
#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

amd::CodeObject object() {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/lds." + vtest::amd_target() + ".o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

// Makes the host refuse new threads for as long as it lives; says whether it could.
struct NoNewThreads {
  rlimit saved{};
  bool effective = false;
  NoNewThreads() {
    getrlimit(RLIMIT_NPROC, &saved);
    rlimit r = saved;
    r.rlim_cur = 1;
    if (setrlimit(RLIMIT_NPROC, &r) != 0) return;
    try {
      std::thread([] {}).join();   // (allowed anyway: root, or CAP_SYS_RESOURCE)
    } catch (const std::system_error&) {
      effective = true;
    }
  }
  ~NoNewThreads() { setrlimit(RLIMIT_NPROC, &saved); }
};

}  // namespace

VTEST(a_dispatch_finishes_when_the_host_will_not_start_its_threads) {
  const amd::CodeObject o = object();
  const int kN = 64, kGroups = 8;
  ::setenv("VGPU_THREADS", "4", 1);
  MemoryManager mem(64ull << 20);
  std::vector<double> in(kN);
  for (int i = 0; i < kN; ++i) in[i] = static_cast<double>(i) * 1.5 - 20.0;
  const uint64_t pin = mem.alloc(kN * 8), pout = mem.alloc(kN * 8);
  mem.write(pin, in.data(), kN * 8);
  const amd::Kernel* k = amd::find_kernel(o, "doubles_in_lds");
  VCHECK(k != nullptr);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  std::vector<uint8_t> args(k->kernarg_size, 0);
  const uint64_t values[3] = {pin, pout, static_cast<uint64_t>(kN)};
  for (size_t i = 0; i < 3 && i < k->args.size(); ++i)
    for (uint32_t b = 0; b < k->args[i].size && b < 8; ++b) args[k->args[i].offset + b] = static_cast<uint8_t>(values[i] >> (8 * b));
  d.kernarg = mem.alloc(args.size());
  mem.write(d.kernarg, args.data(), args.size());
  d.groups[0] = kGroups;   // (every group writes the same answers)
  d.group_size[0] = 64;
  amd::DispatchStats stats;
  {
    NoNewThreads refuse;
    if (!refuse.effective) {
      std::printf("SKIP: this process is allowed to start threads past RLIMIT_NPROC\n");
      return;
    }
    stats = amd::execute(d, mem);
  }
  std::vector<double> out(kN);
  mem.read(pout, out.data(), kN * 8);
  for (int i = 0; i < kN; ++i) VCHECK_EQ(out[i], in[(i + 1) & 63] * 2.0 + in[(i + 5) & 63] * 2.0);
  VCHECK_EQ(stats.waves, static_cast<uint64_t>(kGroups));   // every group ran, on the launching thread
}

// The worker pool (amd/src/gcn_pool.hpp) the dispatch's threads come from: tasks that are independent run, every one,
// on the launching thread when the host gives no thread; tasks that wait on one another start none and throw.
VTEST(the_pool_runs_tasks_here_when_the_host_gives_no_thread_and_refuses_tasks_that_must_run_together) {
  NoNewThreads refuse;
  if (!refuse.effective) {
    std::printf("SKIP: this process is allowed to start threads past RLIMIT_NPROC\n");
    return;
  }
  std::atomic<unsigned> ran{0}, mask{0};
  const std::thread::id here = std::this_thread::get_id();
  std::atomic<bool> all_here{true};
  amd::pool::run(5, [&](unsigned i) {
    ++ran;
    mask |= 1u << i;
    if (std::this_thread::get_id() != here) all_here = false;
  });
  VCHECK_EQ(ran.load(), 5u);
  VCHECK_EQ(mask.load(), 31u);
  VCHECK(all_here.load());   // (a worker the pool made before the limit may have taken some; none was made, as it is the first launch)
  ran = 0;
  bool threw = false;
  try {
    amd::pool::run_together(5, [&](unsigned) { ++ran; });
  } catch (const std::system_error&) {
    threw = true;
  }
  VCHECK(threw);
  VCHECK_EQ(ran.load(), 0u);   // none started, so none is left waiting for a task that was never run
}

VTEST_MAIN
