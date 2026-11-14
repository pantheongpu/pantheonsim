// A primary context released from a library's destructor at exit, as NVHPC's OpenACC runtime releases it
// (pantheonsim #646).
//
// NVHPC 26.9's libaccdevice dlopens libcuda.so.1 at the first compute region and makes these calls, traced on an
// RTX 3060 (driver 610.88) with an LD_PRELOAD logger: cuInit(0), cuCtxGetCurrent (none), cuDevicePrimaryCtxRetain(0),
// cuCtxSetCurrent(primary) three times, cuCtxGetCurrent; and, from its destructor, which the dynamic linker runs
// after exit()'s handlers, cuDevicePrimaryCtxRelease(0). One retain, one release. The driver's own teardown has run
// by then (registered at cuInit, so before the destructors of libraries loaded earlier) and the card answers that
// release CUDA_ERROR_DEINITIALIZED, which NVHPC ignores. The simulator answered it from its destroyed state, as a
// release with nothing retained: CUDA_ERROR_INVALID_CONTEXT, and NVHPC aborted with "Accelerator Fatal Error" and
// exit status 1 after a correct run.
//
// This program does the same: the driver is dlopened by path, after the program has started (so this program's
// destructor runs after the driver's teardown, as libaccdevice's does), and the release is made from a destructor.
// VGPU_TEST_LIBCUDA names another libcuda to run it against (the card's: it passes there). No toolkit header needed.
#include <dlfcn.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kSuccess = 0, kDeinitialized = 4;

using InitFn = int (*)(unsigned);
using GetCurrentFn = int (*)(void**);
using SetCurrentFn = int (*)(void*);
using RetainFn = int (*)(void**, int);
using ReleaseFn = int (*)(int);
using GetStateFn = int (*)(int, unsigned*, int*);

ReleaseFn g_release = nullptr;
int g_failures = 0;
bool g_ran = false;

void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++g_failures;
}

template <class T>
T sym(void* h, const char* name) {
  void* p = dlsym(h, name);
  if (!p) {
    std::printf("FAIL dlsym(%s): %s\n", name, dlerror());
    std::fflush(stdout);
    _exit(1);
  }
  return reinterpret_cast<T>(p);
}

// libaccdevice's destructor: the release that pairs with the retain in main. Runs from _dl_fini, after the
// driver's state is gone.
__attribute__((destructor)) void release_at_exit() {
  if (!g_ran) return;
  const int r = g_release(0);
  std::printf("     cuDevicePrimaryCtxRelease at exit -> %d\n", r);
  check(r == kDeinitialized, "cuDevicePrimaryCtxRelease from a destructor after exit() -> CUDA_ERROR_DEINITIALIZED");
  std::printf("%s\n", g_failures ? "FAIL" : "PASS");
  std::fflush(stdout);
  if (g_failures) _exit(1);
}

}  // namespace

int main() {
  const char* path = std::getenv("VGPU_TEST_LIBCUDA");
  if (!path || !*path) path = VGPU_LIBCUDA_PATH;
  void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    std::printf("FAIL dlopen(%s): %s\n", path, dlerror());
    return 1;
  }
  const auto init = sym<InitFn>(h, "cuInit");
  const auto get_current = sym<GetCurrentFn>(h, "cuCtxGetCurrent");
  const auto set_current = sym<SetCurrentFn>(h, "cuCtxSetCurrent");
  const auto retain = sym<RetainFn>(h, "cuDevicePrimaryCtxRetain");
  const auto get_state = sym<GetStateFn>(h, "cuDevicePrimaryCtxGetState");
  g_release = sym<ReleaseFn>(h, "cuDevicePrimaryCtxRelease");

  check(init(0) == kSuccess, "cuInit(0)");
  void* cur = reinterpret_cast<void*>(0x1);
  check(get_current(&cur) == kSuccess && cur == nullptr, "cuCtxGetCurrent: no context yet");
  void* ctx = nullptr;
  check(retain(&ctx, 0) == kSuccess && ctx != nullptr, "cuDevicePrimaryCtxRetain(0)");
  for (int i = 0; i < 3; ++i) check(set_current(ctx) == kSuccess, "cuCtxSetCurrent(primary)");
  check(get_current(&cur) == kSuccess && cur == ctx, "cuCtxGetCurrent: the primary context");
  unsigned flags = 0;
  int active = 0;
  check(get_state(0, &flags, &active) == kSuccess && active == 1, "cuDevicePrimaryCtxGetState: active");
  g_ran = true;
  std::fflush(stdout);
  // The release is the destructor's (release_at_exit), as NVHPC's is; the library stays loaded.
  return g_failures ? 1 : 0;
}
