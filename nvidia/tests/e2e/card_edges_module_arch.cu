// cuModuleLoadData and the other module loaders handed code for another GPU, as an RTX 3060 (sm_86, CUDA 13.0) answers.
//
//   - A fatbin with nothing the device runs -- SASS only for sm_90, sm_89 or sm_75 (or both of sm_90 and sm_75), or
//     PTX only for compute_90 -- is CUDA_ERROR_NO_BINARY_FOR_GPU from cuModuleLoadData, cuModuleLoadDataEx,
//     cuModuleLoadFatBinary and cuModuleLoad. The simulator answered CUDA_ERROR_NOT_SUPPORTED ("fatbin contains no PTX
//     image") for the SASS-only ones and CUDA_ERROR_INVALID_PTX for the PTX one. A bare sm_90 cubin is the same 209.
//   - What it can run loads and runs: sm_80 SASS (an older minor of its major), sm_75 SASS beside compute_75 PTX.
// The fatbins are built here with the nvcc on PATH (the one that built this program).
#include <cuda.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static int fails = 0;
#define WANT(x, want) do { const long long got_ = static_cast<long long>(x); if (got_ != static_cast<long long>(want)) { \
  std::printf("FAIL %s line %d: %s -> %lld, expected %lld\n", what.c_str(), __LINE__, #x, got_, static_cast<long long>(want)); \
  ++fails; } } while (0)

static std::string what;

static std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  b.push_back(0);   // (a PTX image is read up to its NUL)
  return b;
}

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const std::string dir = std::string(tmp && *tmp ? tmp : "/tmp") + "/vgpu_card_edges_module_arch_" + std::to_string(getpid());
  if (std::system(("mkdir -p '" + dir + "'").c_str()) != 0) { std::printf("FAIL mkdir\n"); return 1; }
  {
    std::ofstream src(dir + "/k.cu");
    src << "extern \"C\" __global__ void kern(int* p) { if (p) *p = 42; }\n";
  }
  struct Case { const char* name; const char* flags; bool runs; };
  const Case cases[] = {
      {"sm_90 SASS", "-fatbin -gencode arch=compute_90,code=sm_90", false},
      {"sm_89 SASS", "-fatbin -gencode arch=compute_89,code=sm_89", false},
      {"sm_75 SASS", "-fatbin -gencode arch=compute_75,code=sm_75", false},
      {"sm_90 and sm_75 SASS", "-fatbin -gencode arch=compute_90,code=sm_90 -gencode arch=compute_75,code=sm_75", false},
      {"compute_90 PTX", "-fatbin -gencode arch=compute_90,code=compute_90", false},
      {"bare sm_90 cubin", "-cubin -arch=sm_90", false},
      {"sm_80 SASS", "-fatbin -gencode arch=compute_80,code=sm_80", true},
      {"sm_75 SASS and PTX", "-fatbin -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75", true},
  };
  CUdevice dev = 0;
  CUcontext ctx = nullptr;
  what = "setup";
  WANT(cuInit(0), CUDA_SUCCESS);
  WANT(cuDeviceGet(&dev, 0), CUDA_SUCCESS);
  WANT(cuDevicePrimaryCtxRetain(&ctx, dev), CUDA_SUCCESS);
  WANT(cuCtxSetCurrent(ctx), CUDA_SUCCESS);
  CUdeviceptr out = 0;
  WANT(cuMemAlloc(&out, sizeof(int)), CUDA_SUCCESS);
  int n = 0;
  for (const Case& c : cases) {
    what = c.name;
    const std::string bin = dir + "/k" + std::to_string(n++) + ".bin";
    const std::string cmd = std::string("nvcc -Wno-deprecated-gpu-targets ") + c.flags + " '" + dir + "/k.cu' -o '" + bin + "'";
    if (std::system(cmd.c_str()) != 0) {
      std::printf("FAIL %s: %s\n", c.name, cmd.c_str());
      ++fails;
      continue;
    }
    const std::vector<char> image = read_file(bin);
    const CUresult want = c.runs ? CUDA_SUCCESS : CUDA_ERROR_NO_BINARY_FOR_GPU;
    CUmodule m = nullptr;
    WANT(cuModuleLoadData(&m, image.data()), want);
    if (m) {
      CUfunction f = nullptr;
      WANT(cuModuleGetFunction(&f, m, "kern"), CUDA_SUCCESS);
      WANT(cuMemsetD32(out, 0, 1), CUDA_SUCCESS);
      void* args[] = {&out};
      if (f) WANT(cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr), CUDA_SUCCESS);
      WANT(cuCtxSynchronize(), CUDA_SUCCESS);
      int v = 0;
      WANT(cuMemcpyDtoH(&v, out, sizeof v), CUDA_SUCCESS);
      WANT(v, 42);
      WANT(cuModuleUnload(m), CUDA_SUCCESS);
    }
    m = nullptr;
    WANT(cuModuleLoadDataEx(&m, image.data(), 0, nullptr, nullptr), want);
    if (m) WANT(cuModuleUnload(m), CUDA_SUCCESS);
    m = nullptr;
    WANT(cuModuleLoadFatBinary(&m, image.data()), want);
    if (m) WANT(cuModuleUnload(m), CUDA_SUCCESS);
    m = nullptr;
    WANT(cuModuleLoad(&m, bin.c_str()), want);
    if (m) WANT(cuModuleUnload(m), CUDA_SUCCESS);
    std::remove(bin.c_str());
  }
  what = "cleanup";
  WANT(cuMemFree(out), CUDA_SUCCESS);
  WANT(cuDevicePrimaryCtxRelease(dev), CUDA_SUCCESS);
  std::remove((dir + "/k.cu").c_str());
  std::system(("rmdir '" + dir + "'").c_str());
  std::printf("%s\n", fails ? "FAILED" : "PASS");
  return fails ? 1 : 0;
}
