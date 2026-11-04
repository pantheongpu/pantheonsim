// What `vgpu run` has to put in front of a program that was linked against cuDSS: this one calls
// cudssGetProperty and cudssConfigGet, prints the version the library reports next to the version
// its header (the one the program was built with) says, and exits 0 only when they agree and the
// library's own defaults are those of that major. Built twice (CMakeLists.txt), against
// libcudss.so.0 (VGPU_CUDSS_API_0_8) and libcudss.so.1, with no RUNPATH to the build tree, so
// the library that answers is whichever one the launcher's shim directory names.
#include <cstdio>

#include "../../include/vgpu_cudss.h"

int main() {
  int major = -1, minor = -1;
  cudssGetProperty(MAJOR_VERSION, &major);
  cudssGetProperty(MINOR_VERSION, &minor);
  cudssConfig_t cfg = nullptr;
  int matching = -1;
  size_t written = 0;
  if (cudssConfigCreate(&cfg) != CUDSS_STATUS_SUCCESS ||
      cudssConfigGet(cfg, CUDSS_CONFIG_MATCHING_ALG, &matching, sizeof matching, &written) != CUDSS_STATUS_SUCCESS)
    return 2;
  cudssConfigDestroy(cfg);
  std::printf("cudss header %d.%d, library %d.%d, default matching %d\n", CUDSS_VERSION_MAJOR, CUDSS_VERSION_MINOR,
              major, minor, matching);
  // 1.0 matches by default (CUDSS_MATCHING_ALG_AUTO = 6), 0.8 did not.
  const int want_matching = CUDSS_VERSION_MAJOR >= 1 ? 6 : 0;
  return major == CUDSS_VERSION_MAJOR && minor == CUDSS_VERSION_MINOR && matching == want_matching ? 0 : 1;
}
