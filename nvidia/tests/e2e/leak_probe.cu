// The resource check (VGPU_LEAK_CHECK, vgpu/leak_check.hpp) must say "clean" for a program that gives everything back, through every
// kind of handle leak_cycle.h uses, and name the resource for a program that does not. Case "clean" does the cycle and frees all of
// it; each "leak-*" case keeps exactly one thing besides.
// run_leak_probe.sh runs one process per case and reads the line the simulator wrote.
#include "leak_cycle.h"

#include <atomic>
#include <cstring>

int main(int argc, char** argv) {
  const char* which = argc > 1 ? argv[1] : "clean";
  WANT_OK(cuInit(0));
  everything_and_give_it_back();
  if (!std::strcmp(which, "clean")) {
    // nothing kept
  } else if (!std::strcmp(which, "leak-device")) {
    void* p = nullptr;
    WANT_OK(cudaMalloc(&p, 1 << 20));
  } else if (!std::strcmp(which, "leak-driver")) {
    CUdeviceptr p;
    WANT_OK(cuMemAlloc(&p, 4096));
  } else if (!std::strcmp(which, "leak-pinned")) {
    void* p = nullptr;
    WANT_OK(cudaMallocHost(&p, 4096));
  } else if (!std::strcmp(which, "leak-fd")) {
    if (open("/dev/null", O_RDONLY) < 0) ++fails;
  } else if (!std::strcmp(which, "leak-thread")) {
    static std::atomic<bool> stop{false};
    static std::thread* t = new std::thread([] { while (!stop) usleep(1000); });
    (void)t;
  } else {
    std::printf("FAIL: unknown case %s\n", which);
    return 2;
  }
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  std::fflush(stdout);
  return fails ? 1 : 0;
}
