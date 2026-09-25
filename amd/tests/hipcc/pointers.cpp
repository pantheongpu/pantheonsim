// What hipPointerGetAttributes says of device memory, pinned host memory and
// memory the runtime knows nothing of. ROCm 6 renumbered the memory types
// (as CUDA numbers them) and made the last an answer rather than an error,
// under a new library name; a program built with ROCm 5 reads the old ones.
#include <hip/hip_runtime.h>

#include <cstdio>

int main() {
  void *device = nullptr, *pinned = nullptr;
  int stack = 0;
  if (hipMalloc(&device, 64) != hipSuccess || hipHostMalloc(&pinned, 64, 0) != hipSuccess) return 1;
  hipPointerAttribute_t a;
  const void* ptrs[] = {device, pinned, &stack};
  const char* what[] = {"device memory", "pinned host memory", "memory the runtime knows nothing of"};
  for (int i = 0; i < 3; ++i) {
    const hipError_t e = hipPointerGetAttributes(&a, ptrs[i]);
#if HIP_VERSION_MAJOR < 6
    const int type = a.memoryType;   // renamed in ROCm 6
#else
    const int type = a.type;
#endif
    if (e == hipSuccess) std::printf("%s: type %d\n", what[i], type);
    else std::printf("%s: %s\n", what[i], hipGetErrorName(e));
  }
  return 0;
}
