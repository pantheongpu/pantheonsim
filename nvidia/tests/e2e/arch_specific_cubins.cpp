// An sm_XYa cubin runs on compute capability X.Y and on nothing newer, an sm_XYf one on every minor of its
// major from Y up, a plain sm_XY one the same -- the driver's rule, and the one a B300 (10.3) and a Vera
// Rubin (10.7) test: an sm_100a cubin is no candidate there, so a fatbin that also carries plain PTX runs
// that, and one that does not has nothing to run.
//
//   arch_specific_cubins <fatbin> run|fail
//
// run_arch_specific.sh builds the fatbins with nvcc and runs this on each simulated GPU. The flag is read
// from the cubin (the .nv.compat record CUDA 13's nvcc writes for "a", or the e_flags bit CUDA 12.8's does); that the driver treats it so is the
// documented rule, derived from documentation and not checked against a card (the local RTX 3060 is sm_86,
// which has no "a" target).
#include <cuda.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  std::ifstream f(argv[1], std::ios::binary);
  const std::string image((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  const bool want_run = std::strcmp(argv[2], "run") == 0;
  CUdevice dev;
  CUcontext ctx;
  if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
      cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS || cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
    std::printf("FAIL: no device\n");
    return 1;
  }
  CUmodule mod = nullptr;
  const CUresult loaded = cuModuleLoadData(&mod, image.data());
  if (!want_run) {
    std::printf(loaded != CUDA_SUCCESS ? "PASS: the load is refused (%d)\n" : "FAIL: the load took an image nothing can run\n",
                static_cast<int>(loaded));
    return loaded != CUDA_SUCCESS ? 0 : 1;
  }
  if (loaded != CUDA_SUCCESS) {
    std::printf("FAIL: cuModuleLoadData returned %d\n", static_cast<int>(loaded));
    return 1;
  }
  CUfunction fn;
  CUdeviceptr out;
  int got[32] = {};
  void* args[] = {&out};
  const bool ok = cuModuleGetFunction(&fn, mod, "fill") == CUDA_SUCCESS && cuMemAlloc(&out, sizeof got) == CUDA_SUCCESS &&
                  cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr) == CUDA_SUCCESS &&
                  cuCtxSynchronize() == CUDA_SUCCESS && cuMemcpyDtoH(got, out, sizeof got) == CUDA_SUCCESS;
  bool right = ok;
  for (int i = 0; i < 32; ++i) right = right && got[i] == i * 5 + 2;
  std::printf(right ? "PASS: the kernel runs and is right\n" : "FAIL: the kernel did not run right\n");
  return right ? 0 : 1;
}
