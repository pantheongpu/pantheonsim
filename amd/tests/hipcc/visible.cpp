// How many devices a program is shown, and what hipGetDeviceCount says:
// run_hipcc.sh runs it under ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES.
// Then, a line a device, which of the machine's devices each one is: its PCI
// bus and the tail of its UUID, the machine device's whatever its place.
#include <hip/hip_runtime.h>

#include <cstdio>

int main() {
  int count = -1;
  const hipError_t e = hipGetDeviceCount(&count);
  std::printf("%s %d\n", hipGetErrorName(e), count);
  for (int i = 0; i < count; ++i) {
    char bus[32] = {};
    hipDeviceProp_t p;
    if (hipDeviceGetPCIBusId(bus, sizeof bus, i) != hipSuccess || hipGetDeviceProperties(&p, i) != hipSuccess) return 1;
    std::printf("device %d: bus %s uuid ...%.2s\n", i, bus, p.uuid.bytes + 14);
  }
  return 0;
}
