// The rest of what ROCm's libamdhip64 exports, answered as it answers them:
// __managed__ variables, modules loaded as libraries and fat binaries, the
// run-time linker, HCC's launch (in C and C++), compiler-rt's half
// conversions, OpenGL interop on a device with no OpenGL, and dma-bufs. Each
// check prints "ok <what>" or "FAIL <what>: <why>", and the last line counts
// them. Built by build.sh with hipcc, with exports_kernel.gfx942.co beside it;
// run by amd/tests/e2e/run_hipcc.sh.
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>
#include <vector>

extern "C" {
uint16_t __gnu_f2h_ieee(float);
float __gnu_h2f_ieee(uint16_t);
hipError_t hipGLGetDevices(unsigned int* count, int* devices, unsigned int n, int list);
}

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

__managed__ int counter = 5;
__global__ void bump() { counter += 1; }

// Runs the loaded add_to on n ints of 1 and says whether each is 1 + v.
static bool adds(hipFunction_t f, int v) {
  constexpr int n = 64;
  int* d = nullptr;
  (void)hipMalloc(&d, n * sizeof(int));
  std::vector<int> h(n, 1);
  (void)hipMemcpy(d, h.data(), n * sizeof(int), hipMemcpyHostToDevice);
  int count = n;
  void* args[] = {&d, &v, &count};
  const hipError_t e = hipModuleLaunchKernel(f, 1, 1, 1, n, 1, 1, 0, nullptr, args, nullptr);
  (void)hipMemcpy(h.data(), d, n * sizeof(int), hipMemcpyDeviceToHost);
  (void)hipFree(d);
  if (e != hipSuccess) return false;
  for (int x : h)
    if (x != 1 + v) return false;
  return true;
}

int main(int argc, char** argv) {
  const std::string co = argc > 1 ? argv[1] : "exports_kernel.gfx942.co";
  std::ifstream in(co, std::ios::binary);
  const std::vector<char> image((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  check(!image.empty(), "the kernel's code object is there to load", co);

  // ---- __managed__ variables: one memory, the host's and every device's
  bump<<<1, 1>>>();
  (void)hipDeviceSynchronize();
  check(counter == 6, "a kernel updates a __managed__ variable the host then reads", std::to_string(counter));
  counter = 40;
  bump<<<1, 1>>>();
  (void)hipDeviceSynchronize();
  check(counter == 41, "and sees what the host wrote there", std::to_string(counter));

  // ---- Modules by other names
  hipModule_t fat = nullptr;
  EXPECT(hipModuleLoadFatBinary(&fat, nullptr), hipErrorInvalidValue, "a fat binary that is not there");
  EXPECT(hipModuleLoadFatBinary(&fat, image.data()), hipSuccess, "a fat binary loads as a module");
  hipFunction_t f = nullptr;
  (void)hipModuleGetFunction(&f, fat, "add_to");
  check(f && adds(f, 2), "and its kernel runs");
  (void)hipModuleUnload(fat);

  hipLibrary_t lib = nullptr;
  EXPECT(hipLibraryLoadData(&lib, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0), hipErrorInvalidValue,
         "a library from nothing");
  EXPECT(hipLibraryLoadData(&lib, image.data(), nullptr, nullptr, 0, nullptr, nullptr, 0), hipSuccess,
         "a library from a code object");
  unsigned int kernels = 0;
  (void)hipLibraryGetKernelCount(&kernels, lib);
  check(kernels == 1, "counts its one kernel", std::to_string(kernels));
  hipKernel_t k = nullptr;
  EXPECT(hipLibraryGetKernel(&k, lib, "add_to"), hipSuccess, "gives it by name");
  check(adds(reinterpret_cast<hipFunction_t>(k), 3), "a library's kernel launches as a module's function");
  EXPECT(hipLibraryUnload(lib), hipSuccess, "and the library unloads");
  EXPECT(hipLibraryLoadFromFile(&lib, "/nonexistent.co", nullptr, nullptr, 0, nullptr, nullptr, 0),
         hipErrorInvalidValue, "a library from a file that is not there");
  EXPECT(hipLibraryLoadFromFile(&lib, co.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0), hipSuccess,
         "a library from a file");
  (void)hipLibraryUnload(lib);

  // ---- HCC's launch, by its C name and its C++ one
  hipModule_t m = nullptr;
  (void)hipModuleLoadData(&m, image.data());
  (void)hipModuleGetFunction(&f, m, "add_to");
  int* d = nullptr;
  (void)hipMalloc(&d, 64 * sizeof(int));
  (void)hipMemset(d, 0, 64 * sizeof(int));
  int v = 4, n = 64;
  void* args[] = {&d, &v, &n};
  EXPECT(hipHccModuleLaunchKernel(f, 64, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr, nullptr, nullptr), hipSuccess,
         "hipHccModuleLaunchKernel launches in work-items");
  hipError_t (*cxx)(hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, hipStream_t,
                    void**, void**, hipEvent_t, hipEvent_t, uint32_t) = &hipExtModuleLaunchKernel;
  EXPECT(cxx(f, 64, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr, nullptr, nullptr, 0), hipSuccess,
         "and hipExtModuleLaunchKernel by its C++ name");
  std::vector<int> back(64);
  (void)hipMemcpy(back.data(), d, 64 * sizeof(int), hipMemcpyDeviceToHost);
  check(back[0] == 8 && back[63] == 8, "both ran", std::to_string(back[0]));
  (void)hipFree(d);
  (void)hipModuleUnload(m);

  // ---- The run-time linker
  hipLinkState_t link = nullptr;
  EXPECT(hipLinkCreate(0, nullptr, nullptr, nullptr), hipErrorInvalidValue, "a link with nowhere to go");
  EXPECT(hipLinkCreate(0, nullptr, nullptr, &link), hipSuccess, "a link");
  EXPECT(hipLinkAddData(link, hipJitInputObject, nullptr, 0, "none", 0, nullptr, nullptr), hipErrorInvalidImage,
         "takes no empty input");
  EXPECT(hipLinkAddData(link, hipJitInputObject, const_cast<char*>(image.data()), image.size(), "add_to", 0, nullptr,
                        nullptr),
         hipSuccess, "takes a code object");
  void* linked = nullptr;
  size_t linked_size = 0;
  EXPECT(hipLinkComplete(link, &linked, &linked_size), hipSuccess, "and completes");
  check(linked_size == image.size(), "with that code object");
  (void)hipModuleLoadData(&m, linked);
  (void)hipModuleGetFunction(&f, m, "add_to");
  check(adds(f, 5), "which loads and runs");
  (void)hipModuleUnload(m);
  EXPECT(hipLinkDestroy(link), hipSuccess, "and the link goes");
  (void)hipLinkCreate(0, nullptr, nullptr, &link);
  (void)hipLinkAddData(link, hipJitInputLLVMBitcode, const_cast<char*>(image.data()), image.size(), "bc", 0, nullptr,
                       nullptr);
  EXPECT(hipLinkComplete(link, &linked, &linked_size), hipErrorNotSupported,
         "LLVM bitcode needs AMD's compiler library to link");
  (void)hipLinkDestroy(link);

  // ---- Half conversions, OpenGL, dma-bufs
  check(__gnu_f2h_ieee(1.5f) == 0x3e00 && __gnu_h2f_ieee(0x3e00) == 1.5f, "1.5 to half and back");
  check(__gnu_f2h_ieee(65520.0f) == 0x7c00 && __gnu_f2h_ieee(-0.0f) == 0x8000, "overflow to infinity, and -0");
  check(__gnu_f2h_ieee(5.96046448e-8f) == 0x0001 && __gnu_h2f_ieee(0x0001) == 5.96046448e-8f,
        "the smallest subnormal both ways");
  check(__gnu_f2h_ieee(1.0f + 1.0f / 2048) == 0x3c00, "a tie rounds to even");
  unsigned int gl = 9;
  int devices[2];
  EXPECT(hipGLGetDevices(&gl, devices, 2, 1), hipErrorNoDevice, "no device is an OpenGL one");
  check(gl == 0, "and none is counted");
  (void)hipMalloc(&d, 256);
  int fd = -1;
  EXPECT(hipMemGetHandleForAddressRange(&fd, nullptr, 256, hipMemRangeHandleTypeDmaBufFd, 0), hipErrorInvalidValue,
         "a dma-buf of nothing");
  EXPECT(hipMemGetHandleForAddressRange(&fd, d, 256, hipMemRangeHandleTypeDmaBufFd, 0), hipSuccess,
         "device memory as a file descriptor");
  check(fd >= 0, "which is one");
  if (fd >= 0) close(fd);
  (void)hipFree(d);
  (void)hipGetLastError();

  std::printf("exports: %d checks, %d failed\n", checks, failures);
  return failures ? 1 : 0;
}
