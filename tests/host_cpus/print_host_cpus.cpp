// Prints vgpu::host_cpus(): what the interpreters spread a grid over when
// VGPU_THREADS is unset. run_host_cpus.sh runs it under an affinity mask and a
// CPU quota and checks the answer.
#include <cstdio>

#include "vgpu/host_cpus.hpp"

int main() {
  std::printf("%u\n", vgpu::host_cpus());
  return 0;
}
