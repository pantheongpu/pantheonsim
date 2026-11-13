// Issue #585: ROCm's HIP gives a kernel with a dynamic stack (.amdhsa_uses_dynamic_stack 1) max(hipLimitStackSize, its static
// private_segment_fixed_size) bytes of scratch per work-item -- CLR's VirtualGPU::submitKernelInternal sets the packet's
// private_segment_size so when the kernel has a dynamic call stack -- and keeps the stack size a multiple of 16, refusing one past what
// a wave's scratch offset can address (Device::UpdateStackSize, GetMaxStackSize: 131004 bytes on gfx9). Before, the simulator gave such
// a kernel its static 16 bytes and stopped at its first access past them. The kernel is amd/tests/data/asm_dyn_stack.s (gfx942).
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>

#include "vgpu_hip.h"

extern "C" {
hipError_t hipDeviceSetLimit(int limit, size_t value);
hipError_t hipDeviceGetLimit(size_t* value, int limit);
}

namespace {
int failures = 0;
void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
// dyn_stack writing at byte `at`: whether it ran and every lane read back what it wrote.
bool launch(hipFunction_t f, uint32_t at) {
  void* out = nullptr;
  if (hipMalloc(&out, 256) != hipSuccess) return false;
  void* params[2] = {&out, &at};
  bool ok = hipModuleLaunchKernel(f, 1, 1, 1, 64, 1, 1, 0, nullptr, params, nullptr) == hipSuccess &&
            hipDeviceSynchronize() == hipSuccess;
  uint32_t got[64] = {};
  ok = ok && hipMemcpy(got, out, sizeof got, hipMemcpyDeviceToHost) == hipSuccess;
  for (uint32_t l = 0; ok && l < 64; ++l) ok = got[l] == 1000 + l;
  (void)hipFree(out);
  return ok;
}
}  // namespace

int main() {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/asm_dyn_stack.gfx942.o";
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::printf("FAIL no code object at %s\n", path.c_str());
    return 1;
  }
  const std::string image((std::istreambuf_iterator<char>(in)), {});
  hipModule_t module = nullptr;
  hipFunction_t f = nullptr;
  if (hipModuleLoadData(&module, image.data()) != hipSuccess || hipModuleGetFunction(&f, module, "dyn_stack") != hipSuccess) {
    std::printf("FAIL could not load dyn_stack\n");
    return 1;
  }
  size_t stack = 0;
  check(hipDeviceGetLimit(&stack, 0) == hipSuccess && stack == 1024, "the stack size is 1024 bytes to begin with");
  check(launch(f, 1000), "a dynamic-stack kernel reaches byte 1000 of its scratch with the default stack size");
  check(hipDeviceSetLimit(0, 2001) == hipSuccess && hipDeviceGetLimit(&stack, 0) == hipSuccess && stack == 2016,
        "a stack size is rounded up to a multiple of 16");
  check(launch(f, 2008), "and the kernel then reaches byte 2008");
  check(hipDeviceSetLimit(0, 131005) == hipErrorInvalidValue, "a stack size past 131004 bytes is refused on gfx942");
  check(hipDeviceSetLimit(0, 131004) == hipSuccess, "131004 bytes is not");
  (void)hipDeviceSetLimit(0, 1024);
  (void)hipModuleUnload(module);
  return failures ? 1 : 0;
}
