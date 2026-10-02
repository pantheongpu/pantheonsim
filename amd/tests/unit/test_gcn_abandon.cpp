// Dispatches abandoned as the process exits (vgpu::amd::abandon_dispatches):
// one running stops within a few thousand instructions a wave, throwing
// Err::DeviceLost, and none starts after. The HIP shim calls it at exit, so
// that a kernel a program leaves running stops before the static objects it
// reads are destroyed. Its own binary, since the switch is the process's and
// stays thrown.
#include <atomic>
#include <chrono>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vgpu/error.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

amd::CodeObject object() {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/vector_add.gfx942.o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

// vector_add over `groups` work-groups of 256, with n = 0: every group is
// cheap, and there are as many as asked for.
amd::Dispatch dispatch(const amd::CodeObject& o, MemoryManager& mem, uint32_t groups) {
  const amd::Kernel* k = amd::find_kernel(o, "vector_add");
  if (!k) throw vtest::Failure("no kernel named vector_add");
  std::vector<uint8_t> args(k->kernarg_size, 0);   // null pointers, n = 0
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(args.size());
  mem.write(d.kernarg, args.data(), args.size());
  d.groups[0] = groups;
  d.group_size[0] = 256;
  return d;
}

}  // namespace

VTEST(a_running_dispatch_stops_when_abandoned_and_none_starts_after) {
  const amd::CodeObject o = object();
  MemoryManager mem(16ull << 20);
  // Runs as it should before.
  const amd::DispatchStats small = amd::execute(dispatch(o, mem, 4), mem);
  VCHECK_EQ(small.waves, 16u);
  VCHECK(!amd::dispatches_abandoned());

  // One that would take a long while, stopped from another thread.
  std::atomic<int> outcome{0};   // 1 finished, 2 abandoned (DeviceLost), 3 anything else
  const amd::Dispatch big = dispatch(o, mem, 1u << 22);
  std::thread runner([&] {
    try {
      amd::execute(big, mem);
      outcome = 1;
    } catch (const Error& e) {
      outcome = e.code() == Err::DeviceLost ? 2 : 3;
    } catch (...) {
      outcome = 3;
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  const auto asked = std::chrono::steady_clock::now();
  amd::abandon_dispatches();
  runner.join();
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - asked).count();
  VCHECK_EQ(outcome.load(), 2);
  VCHECK(seconds < 5.0);
  VCHECK(amd::dispatches_abandoned());

  // And none starts.
  bool refused = false;
  try {
    amd::execute(dispatch(o, mem, 1), mem);
  } catch (const Error& e) {
    refused = e.code() == Err::DeviceLost;
  }
  VCHECK(refused);
}

VTEST_MAIN
