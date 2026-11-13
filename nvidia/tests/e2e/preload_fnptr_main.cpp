// Loads preload_fnptr_lib.so privately and calls its initialisation, as `import ThrustRTC` does (issue #597). Names no CUDA library,
// so `vgpu run --preload` preloads every one the shim carries.
#include <dlfcn.h>
#include <cstdio>

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    std::printf("cannot load %s: %s\n", argv[1], dlerror());
    return 2;
  }
  const auto init = reinterpret_cast<int (*)()>(dlsym(h, "fnptr_init"));
  if (!init) return 2;
  const int r = init();
  std::printf("%s\n", r == 0 ? "ok" : "failed");
  return r;
}
