// libcudss.so.0 and libcudss.so.1 side by side in one process, each answering for its own major: a
// program (or a plugin) that loads both gets two independent libraries, not one shadowing the other.
// The shim's directory is found from this program's own location (build/ holds libcudss.so.0.8 and
// cudss1/libcudss.so.1.0.0).
#include <dlfcn.h>
#include <unistd.h>

#include <string>

#define VGPU_CUDSS_API_0_8 /* the declarations only; every call goes through dlsym */
#include "../../include/vgpu_cudss.h"
#include "vtest.hpp"

extern "C" const char* __asan_default_options() { return "detect_leaks=1:detect_odr_violation=0"; }

static std::string exe_dir() {
  char buf[4096];
  const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
  std::string p(buf, n > 0 ? (size_t)n : 0);
  return p.substr(0, p.rfind('/'));
}

struct Lib {
  void* h = nullptr;
  int (*get_property)(libraryPropertyType, int*) = nullptr;
  int (*config_create)(cudssConfig_t*) = nullptr;
  int (*config_get)(cudssConfig_t, cudssConfigParam_t, void*, size_t, size_t*) = nullptr;
  int (*config_destroy)(cudssConfig_t) = nullptr;
  int (*create)(cudssHandle_t*) = nullptr;
  int (*destroy)(cudssHandle_t) = nullptr;
  void* jit = nullptr;
};

static Lib open_lib(const std::string& path) {
  Lib l;
  l.h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  VCHECK(l.h != nullptr);
  l.get_property = (int (*)(libraryPropertyType, int*))dlsym(l.h, "cudssGetProperty");
  l.config_create = (int (*)(cudssConfig_t*))dlsym(l.h, "cudssConfigCreate");
  l.config_get = (int (*)(cudssConfig_t, cudssConfigParam_t, void*, size_t, size_t*))dlsym(l.h, "cudssConfigGet");
  l.config_destroy = (int (*)(cudssConfig_t))dlsym(l.h, "cudssConfigDestroy");
  l.create = (int (*)(cudssHandle_t*))dlsym(l.h, "cudssCreate");
  l.destroy = (int (*)(cudssHandle_t))dlsym(l.h, "cudssDestroy");
  l.jit = dlsym(l.h, "cudssSetJitCacheDir");
  VCHECK(l.get_property && l.config_create && l.config_get && l.config_destroy && l.create && l.destroy);
  return l;
}

static int matching_default(const Lib& l) {
  cudssConfig_t c = nullptr;
  VCHECK_EQ(l.config_create(&c), 0);
  int v = -1;
  size_t w = 0;
  VCHECK_EQ(l.config_get(c, CUDSS_CONFIG_MATCHING_ALG, &v, sizeof v, &w), 0);
  VCHECK_EQ(l.config_destroy(c), 0);
  return v;
}

VTEST(both_majors_load_together_and_each_answers_for_its_own) {
  const std::string dir = exe_dir();
  Lib v0 = open_lib(dir + "/libcudss.so.0.8");
  Lib v1 = open_lib(dir + "/cudss1/libcudss.so.1.0.0");
  int a = -1, b = -1;
  VCHECK_EQ(v0.get_property(MAJOR_VERSION, &a), 0);
  VCHECK_EQ(v1.get_property(MAJOR_VERSION, &b), 0);
  VCHECK_EQ(a, 0);
  VCHECK_EQ(b, 1);
  VCHECK_EQ(v0.get_property(MINOR_VERSION, &a), 0);
  VCHECK_EQ(a, 8);
  // Different defaults prove they are two implementations, not one library seen twice.
  VCHECK_EQ(matching_default(v0), 0);
  VCHECK_EQ(matching_default(v1), 6);
  VCHECK(v0.jit == nullptr);
  VCHECK(v1.jit != nullptr);
  // Both can hold a handle at the same time.
  cudssHandle_t h0 = nullptr, h1 = nullptr;
  VCHECK_EQ(v0.create(&h0), 0);
  VCHECK_EQ(v1.create(&h1), 0);
  VCHECK_EQ(v1.destroy(h1), 0);
  VCHECK_EQ(v0.destroy(h0), 0);
  dlclose(v1.h);
  dlclose(v0.h);
}

VTEST_MAIN
