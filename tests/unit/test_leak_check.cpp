// VGPU_LEAK_CHECK (vgpu/leak_check.hpp): the line is clean for a machine whose program gave everything back, and names each
// resource that did not come back -- device memory (bytes and allocations), pinned host blocks, file descriptors and threads. A check that has never failed proves nothing, so every resource has a case that leaks on purpose.
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <string>
#include <thread>

#include "vgpu/leak_check.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {
// The mode is read once, so it is set before anything asks.
const bool g_env = (setenv("VGPU_LEAK_CHECK", "1", 1), true);

// The machine first, the baseline after it: as the shims do (shared_runtime.cpp), so what the machine itself holds is not growth.
runtime::Runtime& machine() {
  static runtime::Runtime* rt = [] {
    auto* r = new runtime::Runtime(load_gpu("nvidia/a10"));
    runtime::leak_check_arm();
    return r;
  }();
  return *rt;
}

std::string line_of(runtime::Runtime& rt, bool* leaked) { return runtime::leak_check_line(&rt, leaked); }
}  // namespace

VTEST(a_machine_that_was_given_everything_back_is_clean) {
  (void)g_env;
  runtime::Runtime& rt = machine();
  bool leaked = true;
  std::string line = line_of(rt, &leaked);
  VCHECK_CONTAINS(line, "device_bytes=0 device_allocs=0 pinned_allocs=0");
  VCHECK_CONTAINS(line, "fds=+0");
  // Allocate, use and free: back to clean.
  MemoryManager& mm = rt.device(0).memory();
  const uint64_t p = mm.alloc(1 << 20);
  mm.free(p);
  line = line_of(rt, &leaked);
  VCHECK_CONTAINS(line, "device_bytes=0 device_allocs=0");
}

VTEST(device_memory_not_freed_is_a_leak_in_bytes_and_in_allocations) {
  runtime::Runtime& rt = machine();
  bool leaked = false;
  MemoryManager& mm = rt.device(0).memory();
  const uint64_t p = mm.alloc(4096);
  const std::string line = line_of(rt, &leaked);
  VCHECK(leaked);
  VCHECK_CONTAINS(line, "device_allocs=1");
  VCHECK_CONTAINS(line, "LEAK device_bytes,device_allocs");
  mm.free(p);
  VCHECK_CONTAINS(line_of(rt, &leaked), "device_bytes=0 device_allocs=0");
}

long threads_of(const std::string& line) {   // the "threads=+N" figure
  const auto at = line.find("threads=");
  return at == std::string::npos ? -1000 : std::strtol(line.c_str() + at + 8, nullptr, 10);
}

VTEST(a_descriptor_left_behind_is_a_leak) {
  runtime::Runtime& rt = machine();
  bool leaked = false;
  const int fd = open("/dev/null", O_RDONLY);
  std::string line = line_of(rt, &leaked);
  VCHECK(leaked);
  VCHECK_CONTAINS(line, "fds=+1");
  VCHECK_CONTAINS(line, "/dev/null");
  close(fd);
  line = line_of(rt, &leaked);
  VCHECK_CONTAINS(line, "fds=+0");
}

// (Relative to what the line says before: a sanitizer runs a thread of its own, which the baseline may not have had.)
VTEST(a_thread_left_running_is_a_leak) {
  runtime::Runtime& rt = machine();
  bool leaked = false;
  const long base = threads_of(line_of(rt, &leaked));
  // The thread count the machine sees comes from the process's task list, which lags a thread's start and its end by a moment (more under a
  // sanitizer, and on a loaded runner): ask again until it is what it must be, for up to two seconds.
  const auto settle = [&](long want, std::string* line) {
    for (int i = 0; i < 200; ++i) {
      *line = line_of(rt, &leaked);
      if (threads_of(*line) == want) return;
      usleep(10000);
    }
  };
  std::atomic<bool> stop{false};
  std::thread t([&] { while (!stop) usleep(1000); });
  std::string during, after;
  settle(base + 1, &during);
  stop = true;
  t.join();
  settle(base, &after);
  VCHECK_EQ(threads_of(during), base + 1);
  VCHECK_EQ(threads_of(after), base);
  VCHECK_CONTAINS(during, "LEAK");
}

VTEST(pinned_host_blocks_not_freed_are_a_leak) {
  runtime::Runtime& rt = machine();
  bool leaked = false;
  char block[16];
  rt.host_allocations()[block] = runtime::HostRange{64, 0, 0};
  const std::string line = line_of(rt, &leaked);
  VCHECK(leaked);
  VCHECK_CONTAINS(line, "pinned_allocs=1 pinned_bytes=64");
  rt.host_allocations().erase(block);
  VCHECK_CONTAINS(line_of(rt, &leaked), "pinned_allocs=0");
}

VTEST_MAIN
