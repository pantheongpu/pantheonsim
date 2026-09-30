// Modules as ROCm's HIP answers them: a bundle carrying only generic code
// (gfx9-4-generic) loads and runs on gfx942, and the module calls answer
// hip-tests' negative cases (ModuleTest) as ROCm does. Each check prints
// "ok <what>" or "FAIL <what>: <why>", and the last line counts them. Built by
// build.sh with hipcc, with modules_kernel.generic.co beside it; run by
// amd/tests/e2e/run_hipcc.sh.
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static int checks = 0, failures = 0;
static void check(bool ok, const char* what, const std::string& why = "") {
  ++checks;
  if (ok) {
    std::printf("ok    %s\n", what);
  } else {
    ++failures;
    std::printf("FAIL  %s%s%s\n", what, why.empty() ? "" : ": ", why.c_str());
  }
}
static std::string err(hipError_t e) { return hipGetErrorName(e); }
#define EXPECT(call, want, what) \
  do { \
    const hipError_t got_ = (call); \
    check(got_ == (want), what, "got " + err(got_) + ", want " + err(want)); \
  } while (0)

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: modules <modules_kernel.generic.co>\n");
    return 2;
  }
  const char* path = argv[1];
  (void)hipFree(nullptr);

  // Generic code: the bundle names gfx9-4-generic and nothing else.
  hipModule_t m = nullptr;
  EXPECT(hipModuleLoad(&m, path), hipSuccess, "a gfx9-4-generic code object loads on gfx942");
  hipFunction_t f = nullptr;
  EXPECT(hipModuleGetFunction(&f, m, "scale"), hipSuccess, "its kernel is found");
  const int n = 256;
  std::vector<int> host(n, 3);
  int* d = nullptr;
  (void)hipMalloc(&d, n * sizeof(int));
  (void)hipMemcpy(d, host.data(), n * sizeof(int), hipMemcpyHostToDevice);
  int count = n;
  void* args[] = {&d, &count};
  EXPECT(hipModuleLaunchKernel(f, n / 64, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr), hipSuccess,
         "and launches");
  (void)hipMemcpy(host.data(), d, n * sizeof(int), hipMemcpyDeviceToHost);
  check(host[0] == 21 && host[n - 1] == 21, "and runs, reading its global", std::to_string(host[0]));

  // Launches the device cannot take: hipModuleLaunchKernel answers them as
  // wrong values, hipExtModuleLaunchKernel (whose grid is in work-items) as
  // wrong configurations, and both refuse no function as a bad handle.
  EXPECT(hipModuleLaunchKernel(nullptr, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr), hipErrorInvalidResourceHandle,
         "a launch of no function");
  EXPECT(hipModuleLaunchKernel(f, 0, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr), hipErrorInvalidValue,
         "a module launch of an empty grid");
  EXPECT(hipModuleLaunchKernel(f, 1, 1, 1, 2048, 1, 1, 0, nullptr, args, nullptr), hipErrorInvalidValue,
         "a module launch of a block wider than the device's");
  EXPECT(hipExtModuleLaunchKernel(f, 64, 1, 1, 2048, 1, 1, 0, nullptr, args, nullptr, nullptr, nullptr, 0),
         hipErrorInvalidConfiguration, "an extended launch of a block wider than the device's");
  void* extra[] = {HIP_LAUNCH_PARAM_END};
  EXPECT(hipModuleLaunchKernel(f, 1, 1, 1, 64, 1, 1, 0, nullptr, args, extra), hipErrorInvalidValue,
         "a launch passing both kernelParams and extra");
  // hipcc builds HIP kernels for uniform work-groups: a grid of work-items
  // that is not a whole number of them is refused.
  EXPECT(hipExtModuleLaunchKernel(f, 100, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr, nullptr, nullptr, 0),
         hipErrorInvalidValue, "an extended launch of a partial work-group, for a uniform kernel");
  EXPECT(hipExtModuleLaunchKernel(f, n, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr, nullptr, nullptr, 0), hipSuccess,
         "an extended launch of whole work-groups");
  (void)hipDeviceSynchronize();
  int devices = 0;
  (void)hipGetDeviceCount(&devices);
  if (devices > 1) {
    hipStream_t other = nullptr;
    (void)hipSetDevice(1);
    (void)hipStreamCreate(&other);
    (void)hipSetDevice(0);
    EXPECT(hipModuleLaunchKernel(f, 1, 1, 1, 64, 1, 1, 0, other, args, nullptr), hipErrorInvalidResourceHandle,
           "a launch on another device's stream");
    (void)hipStreamDestroy(other);
  }

  // A library's kernel launched by hipLaunchKernel, as a host function is:
  // ROCm's hipLaunchKernel takes either.
  {
    hipLibrary_t lib = nullptr;
    EXPECT(hipLibraryLoadFromFile(&lib, path, nullptr, nullptr, 0, nullptr, nullptr, 0), hipSuccess,
           "the code object loads as a library");
    hipKernel_t kernel = nullptr;
    EXPECT(hipLibraryGetKernel(&kernel, lib, "scale"), hipSuccess, "its kernel is found");
    std::vector<int> ones(n, 1);
    (void)hipMemcpy(d, ones.data(), n * sizeof(int), hipMemcpyHostToDevice);
    EXPECT(hipLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(n / 64), dim3(64), args, 0, nullptr),
           hipSuccess, "hipLaunchKernel launches a library's kernel");
    (void)hipMemcpy(ones.data(), d, n * sizeof(int), hipMemcpyDeviceToHost);
    check(ones[0] == 7 && ones[n - 1] == 7, "and it runs", std::to_string(ones[0]));
    EXPECT(hipLibraryUnload(lib), hipSuccess, "the library unloads");
  }
  // Libraries load lazily, as ROCm's do: an image that is no code object
  // loads, and is refused when a kernel is asked of it.
  {
    hipLibrary_t lib = nullptr;
    hipKernel_t kernel = nullptr;
    EXPECT(hipLibraryLoadData(&lib, "call me ishmael", nullptr, nullptr, 0, nullptr, nullptr, 0), hipSuccess,
           "a library of an image that is no code object loads");
    EXPECT(hipLibraryGetKernel(&kernel, lib, "moby"), hipErrorInvalidImage, "and is refused when a kernel is asked");
    EXPECT(hipLibraryUnload(lib), hipSuccess, "and unloads");
    EXPECT(hipLibraryLoadData(nullptr, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0), hipErrorInvalidValue,
           "a library load with nowhere to put it");
    EXPECT(hipLibraryUnload(nullptr), hipErrorInvalidValue, "an unload of no library");
  }

  // Arguments packed in extra: the kernel's own come from the buffer, as
  // many as it takes, whatever the size beside it says -- hip-tests' RTC
  // reduce passes an int there, and a size too small for its arguments.
  {
    std::vector<int> fives(n, 5);
    (void)hipMemcpy(d, fives.data(), n * sizeof(int), hipMemcpyHostToDevice);
    struct {
      int* p;
      int n;
    } packed{d, n};
    int size = 4;
    void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &packed, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
    EXPECT(hipModuleLaunchKernel(f, n / 64, 1, 1, 64, 1, 1, 0, nullptr, nullptr, extra), hipSuccess,
           "a launch with its arguments in extra, beside an int's size");
    (void)hipMemcpy(fives.data(), d, n * sizeof(int), hipMemcpyDeviceToHost);
    check(fives[0] == 35 && fives[n - 1] == 35, "and the kernel has all its arguments", std::to_string(fives[0]));
  }

  hipDeviceptr_t global = nullptr;
  size_t bytes = 0;
  EXPECT(hipModuleGetGlobal(&global, &bytes, m, "int_var"), hipSuccess, "its global is found");
  check(bytes == sizeof(int), "the global's size", std::to_string(bytes));
  EXPECT(hipModuleGetGlobal(&global, &bytes, nullptr, "int_var"), hipErrorInvalidResourceHandle,
         "a global of no module");
  EXPECT(hipModuleGetGlobal(&global, &bytes, m, ""), hipErrorInvalidValue, "a global with an empty name");
  EXPECT(hipModuleGetGlobal(nullptr, nullptr, m, "int_var"), hipErrorInvalidValue,
         "a global asked for with nowhere to put it");
  EXPECT(hipModuleGetGlobal(&global, &bytes, m, "dummy"), hipErrorNotFound, "a global that is not there");

  // What hipFuncGetAttribute says of a module's kernel: the device's version.
  int value = 0, major = 0, minor = 0;
  (void)hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor, 0);
  (void)hipDeviceGetAttribute(&minor, hipDeviceAttributeComputeCapabilityMinor, 0);
  EXPECT(hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_BINARY_VERSION, f), hipSuccess, "a kernel's binary version");
  check(value == major * 10 + minor && value == 94, "is the device's, 9.4", std::to_string(value));
  EXPECT(hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_PTX_VERSION, f), hipSuccess, "its PTX version");
  check(value > 0, "is set", std::to_string(value));
  EXPECT(hipFuncGetAttribute(nullptr, HIP_FUNC_ATTRIBUTE_BINARY_VERSION, f), hipErrorInvalidValue,
         "an attribute with nowhere to put it");
  EXPECT(hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_BINARY_VERSION, nullptr), hipErrorInvalidResourceHandle,
         "an attribute of no function");

  // Loading and unloading.
  hipModule_t other = nullptr;
  EXPECT(hipModuleLoad(nullptr, path), hipErrorInvalidValue, "a load with nowhere to put the module");
  EXPECT(hipModuleLoad(&other, nullptr), hipErrorInvalidValue, "a load of no file");
  EXPECT(hipModuleLoad(&other, ""), hipErrorInvalidValue, "a load of an empty file name");
  EXPECT(hipModuleLoad(&other, "no such file"), hipErrorFileNotFound, "a load of a file that is not there");
  EXPECT(hipModuleUnload(nullptr), hipErrorInvalidResourceHandle, "an unload of no module");
  EXPECT(hipModuleUnload(m), hipSuccess, "an unload");
  EXPECT(hipModuleUnload(m), hipErrorNotFound, "the same module unloaded twice");

  (void)hipFree(d);
  std::printf("modules: %d checks, %d failed\n", checks, failures);
  return failures ? 1 : 0;
}
