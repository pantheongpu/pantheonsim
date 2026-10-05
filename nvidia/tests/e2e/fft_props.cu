// cuFFT's per-plan properties (cufftSetPlanPropertyInt64, cufftGetPlanPropertyInt64,
// cufftResetPlanProperty): the patient-JIT flag is kept and read back (any
// non-zero value as 1), an unknown property is refused, and a plan already
// made refuses changes. Every expectation below is what an RTX 3060's cuFFT
// 12.0 (CUDA 13.0) answered, and the program passes against it. The entry
// points are looked up by name: CUDA 12.0's header does not declare them.
#include <cuda_runtime.h>
#include <cufft.h>
#include <dlfcn.h>

#include <cstdio>

static int failures = 0;
static void check(bool ok, const char* what, long long v) {
  std::printf("%-4s %s (%lld)\n", ok ? "ok" : "FAIL", what, v);
  if (!ok) ++failures;
}

typedef int (*SetFn)(cufftHandle, int, long long);
typedef int (*GetFn)(cufftHandle, int, long long*);
typedef int (*ResetFn)(cufftHandle, int);

int main() {
  auto set = (SetFn)dlsym(RTLD_DEFAULT, "cufftSetPlanPropertyInt64");
  auto get = (GetFn)dlsym(RTLD_DEFAULT, "cufftGetPlanPropertyInt64");
  auto reset = (ResetFn)dlsym(RTLD_DEFAULT, "cufftResetPlanProperty");
  if (!set || !get || !reset) {
    std::printf("SKIP: this cuFFT has no plan properties\n");
    return 0;
  }
  cufftHandle p;
  if (cufftCreate(&p)) { std::printf("FAIL: cufftCreate\n"); return 1; }
  long long v = -77;
  check(get(p, 1, &v) == CUFFT_SUCCESS && v == 0, "patient JIT defaults to 0", v);
  check(set(p, 1, 5) == CUFFT_SUCCESS, "set patient JIT to 5", 0);
  check(get(p, 1, &v) == CUFFT_SUCCESS && v == 1, "any non-zero value reads back as 1", v);
  check(set(p, 1, -1) == CUFFT_SUCCESS && get(p, 1, &v) == CUFFT_SUCCESS && v == 1, "negative is non-zero too", v);
  check(set(p, 1, 0) == CUFFT_SUCCESS && get(p, 1, &v) == CUFFT_SUCCESS && v == 0, "zero clears it", v);
  set(p, 1, 1);
  check(reset(p, 1) == CUFFT_SUCCESS && get(p, 1, &v) == CUFFT_SUCCESS && v == 0, "reset restores 0", v);
  check(set(p, 2, 4) == CUFFT_NOT_SUPPORTED, "host-thread cap: set is not supported", 0);
  check(get(p, 2, &v) == CUFFT_NOT_SUPPORTED, "host-thread cap: get is not supported", 0);
  check(reset(p, 2) == CUFFT_NOT_SUPPORTED, "host-thread cap: reset is not supported", 0);
  for (int bad : {0, 3, -1, 256}) {
    char what[64];
    std::snprintf(what, sizeof what, "property %d is refused", bad);
    check(set(p, bad, 1) == CUFFT_INVALID_VALUE && get(p, bad, &v) == CUFFT_INVALID_VALUE &&
              reset(p, bad) == CUFFT_INVALID_VALUE, what, bad);
  }
  check(get(p, 1, nullptr) == CUFFT_INVALID_VALUE, "a NULL result pointer", 0);
  set(p, 1, 1);
  size_t work = 0;
  check(cufftMakePlan1d(p, 64, CUFFT_C2C, 1, &work) == CUFFT_SUCCESS, "make the plan", 0);
  check(get(p, 1, &v) == CUFFT_SUCCESS && v == 1, "the flag survives the make", v);
  check(set(p, 1, 0) == CUFFT_NOT_SUPPORTED, "a made plan refuses set", 0);
  check(reset(p, 1) == CUFFT_NOT_SUPPORTED, "a made plan refuses reset", 0);
  check(get(p, 1, &v) == CUFFT_SUCCESS && v == 1, "a made plan still answers get", v);
  cufftDestroy(p);
  check(set(p, 1, 1) == CUFFT_INVALID_PLAN && get(p, 1, &v) == CUFFT_INVALID_PLAN &&
            reset(p, 1) == CUFFT_INVALID_PLAN, "a destroyed handle is an invalid plan", 0);
  std::printf(failures ? "FAIL: %d cuFFT property checks\n" : "PASS: every cuFFT property check\n", failures);
  return failures ? 1 : 0;
}
