// A library that loads the driver and NVRTC itself and keeps what dlsym gives in exported function pointers named like the API, as
// ThrustRTC's libPyThrustRTC.so does (issue #597). With a shim library that defines `cuInit` in the global scope (LD_PRELOAD), the
// library's own references to its variable `cuInit` bind to the shim's function instead, and the first store writes over that
// function's code. Built by run_preload_fnptr.sh; loaded with dlopen by preload_fnptr_main.cpp, as Python loads an extension.
#include <dlfcn.h>
#include <cstdio>

extern "C" {
__attribute__((visibility("default"))) int (*cuInit)(unsigned) = nullptr;
__attribute__((visibility("default"))) int (*cuDriverGetVersion)(int*) = nullptr;
__attribute__((visibility("default"))) int (*nvrtcVersion)(int*, int*) = nullptr;

__attribute__((visibility("default"))) int fnptr_init() {
  void* cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!cuda) {
    std::printf("libcuda.so.1 not found: %s\n", dlerror());
    return 1;
  }
  cuInit = reinterpret_cast<int (*)(unsigned)>(dlsym(cuda, "cuInit"));
  cuDriverGetVersion = reinterpret_cast<int (*)(int*)>(dlsym(cuda, "cuDriverGetVersion"));
  if (!cuInit || !cuDriverGetVersion) {
    std::printf("the driver lacks cuInit or cuDriverGetVersion\n");
    return 1;
  }
  int v = 0;
  const int init = cuInit(0), ver = cuDriverGetVersion(&v);
  std::printf("cuInit %d, cuDriverGetVersion %d (%s)\n", init, ver, v > 0 ? "a version" : "none");
  void* nvrtc = dlopen("libnvrtc.so", RTLD_NOW | RTLD_LOCAL);
  if (!nvrtc) {
    std::printf("libnvrtc.so not found: %s\n", dlerror());
    return 1;
  }
  nvrtcVersion = reinterpret_cast<int (*)(int*, int*)>(dlsym(nvrtc, "nvrtcVersion"));
  int major = 0, minor = 0;
  const int r = nvrtcVersion ? nvrtcVersion(&major, &minor) : -1;
  std::printf("nvrtcVersion %d (%s)\n", r, major > 0 ? "a version" : "none");
  return init != 0 || ver != 0 || r != 0;
}
}
