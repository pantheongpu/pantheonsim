// Every error code's name and text, from hipGetErrorName/String and their
// driver forms, for each value from 0 to 1100 that either knows: what
// rocm/errors.expected holds as ROCm 7.1's own libamdhip64 answers it (over
// this HSA runtime), line for line. Built by build.sh; run_hipcc.sh diffs.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <string>

int main() {
  for (int c = 0; c <= 1100; ++c) {
    const auto e = static_cast<hipError_t>(c);
    const char* dname = nullptr;
    const char* dtext = nullptr;
    const hipError_t rn = hipDrvGetErrorName(e, &dname);
    const hipError_t rt = hipDrvGetErrorString(e, &dtext);
    const std::string name = hipGetErrorName(e), text = hipGetErrorString(e);
    if (rn != hipSuccess && rt != hipSuccess && name == "hipErrorUnknown" && text == "unknown error") continue;
    std::printf("%d|%s|%s|%s|%s\n", c, name.c_str(), text.c_str(), rn == hipSuccess ? dname : "(refused)",
                rt == hipSuccess ? dtext : "(refused)");
  }
  return 0;
}
