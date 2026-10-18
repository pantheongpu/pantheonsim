// The agent rocprofiler-sdk gives the profiler for the GPU, in every field (shader engines and arrays, SDMA
// engines, queues, clocks, memory banks, caches, io links, firmware). The program only has to start the
// runtime: the comparison is the agent block of the profiler's output
// (amd/tests/rocprof_normalize.py --agents), not what the program does.
#include "common.h"

int main() {
  int n = 0;
  CHECK(hipGetDeviceCount(&n));
  CHECK(hipSetDevice(0));
  CHECK(hipFree(nullptr));
  std::printf("agents %d\n", n);
  return n != 1;
}
