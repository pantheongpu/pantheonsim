// A module that cannot be put on the device because the host is short of memory (the simulator's guard: below
// VGPU_MEMORY_HOST_RESERVE_MB of free host memory it refuses device memory) is hipErrorOutOfMemory, as a card
// that has no room for a code object says. It used to come back as hipErrorInvalidImage ("device kernel image is
// invalid"), which hid the cause: on a shared machine under memory pressure, a launch of a loaded program then failed
// once in a while, with a message that blamed the program's own code. The reserve here is far more memory than any host
// has, so the first chunk of real bytes the module needs is refused (the test loads a linked code object, whose image
// is written to device memory; chunks that cost nothing are not refused). ctest sets the reserve in the environment.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

#include "vgpu_hip.h"

int main() {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/devmalloc.gfx942.hsaco";
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::printf("FAIL no code object at %s\n", path.c_str());
    return 1;
  }
  const std::string image((std::istreambuf_iterator<char>(in)), {});
  hipModule_t module = nullptr;
  const hipError_t e = hipModuleLoadData(&module, image.data());
  if (e != hipErrorOutOfMemory) {
    std::printf("FAIL loading a module with no host memory to spare gave %d (%s), not hipErrorOutOfMemory\n", static_cast<int>(e),
                hipGetErrorString(e));
    return 1;
  }
  std::printf("ok a module the host has no room for is hipErrorOutOfMemory\n");
  return 0;
}
