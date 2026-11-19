// VGPU_LEAK_CHECK (vgpu/leak_check.hpp): the line is clean for a machine whose program gave everything back, and names each
// resource that did not come back -- device memory (bytes and allocations), pinned host blocks, file descriptors, threads and
// mappings. A check that has never failed proves nothing, so every resource has a case that leaks on purpose.
#include <fcntl.h>
#include <sys/mman.h>
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
  VCHECK(!leaked);
  VCHECK_CONTAINS(line, "device_bytes=0 device_allocs=0");
  VCHECK_CONTAINS(line, "-> clean");
  // Allocate, use and free: back to clean.
  MemoryManager& mm = rt.device(0).memory();
  const uint64_t p = mm.alloc(1 << 20);
  mm.free(p);
  line = line_of(rt, &leaked);
  VCHECK(!leaked);
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
  line_of(rt, &leaked);
  VCHECK(!leaked);
}

VTEST(a_descriptor_a_thread_and_a_mapping_left_behind_are_leaks) {
  runtime::Runtime& rt = machine();
  bool leaked = false;
  const int fd = open("/dev/null", O_RDONLY);
  std::string line = line_of(rt, &leaked);
  VCHECK(leaked);
  VCHECK_CONTAINS(line, "fds=+1");
  VCHECK_CONTAINS(line, "/dev/null");
  close(fd);
  line_of(rt, &leaked);
  VCHECK(!leaked);

  std::atomic<bool> stop{false};
  std::thread t([&] { while (!stop) usleep(1000); });
  line = line_of(rt, &leaked);
  VCHECK(leaked);
  VCHECK_CONTAINS(line, "threads=+1");
  stop = true;
  t.join();
  line_of(rt, &leaked);
  VCHECK(!leaked);

  // A mapping with a different protection from its neighbours is a line of its own in /proc/self/maps.
  void* m = mmap(nullptr, 1 << 16, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  void* guard = mmap(nullptr, 1 << 16, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  line = line_of(rt, &leaked);
  VCHECK(leaked);
  VCHECK_CONTAINS(line, "maps=");
  munmap(m, 1 << 16);
  munmap(guard, 1 << 16);
  line_of(rt, &leaked);
  VCHECK(!leaked);
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
  line_of(rt, &leaked);
  VCHECK(!leaked);
}

VTEST_MAIN
