// How many devices a program is shown, and what hipGetDeviceCount says:
// run_hipcc.sh runs it under ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES.
#include <hip/hip_runtime.h>

#include <cstdio>

int main() {
  int count = -1;
  const hipError_t e = hipGetDeviceCount(&count);
  std::printf("%s %d\n", hipGetErrorName(e), count);
  return 0;
}
