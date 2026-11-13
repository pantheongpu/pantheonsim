// NPP 12's own stream (nppGetStreamContext, nppGetStream, nppSetStream), which CUDA 12 programs use to fill the
// NppStreamContext of their _Ctx calls (saiga's NppiHelper.h: pantheonsim #617), and a _Ctx call made with what it
// fills. CUDA 13's NPP has none of the three, so a CUDA 13 build skips. Passes with NPP 12.3 (CUDA 12.6) on an RTX 3060.
#include <cuda_runtime.h>
#include <npp.h>

#include <cstdio>
#include <cstring>

int main() {
#if NPP_VER_MAJOR >= 13
  std::printf("SKIP: NPP %d has no nppGetStreamContext (it is NPP 12's)\n", NPP_VER_MAJOR);
  std::printf("PASS\n");
  return 0;
#else
  int failures = 0;
  const auto check = [&](bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
  };
  cudaDeviceProp p{};
  cudaGetDeviceProperties(&p, 0);

  NppStreamContext c;
  std::memset(&c, 0x5a, sizeof c);
  check(nppGetStreamContext(&c) == NPP_SUCCESS, "nppGetStreamContext");
  check(c.hStream == nullptr && c.nStreamFlags == 0, "the stream is 0 before nppSetStream");
  check(c.nCudaDeviceId == 0, "the current device");
  check(c.nMultiProcessorCount == p.multiProcessorCount && c.nMaxThreadsPerMultiProcessor == p.maxThreadsPerMultiProcessor &&
            c.nMaxThreadsPerBlock == p.maxThreadsPerBlock && c.nSharedMemPerBlock == p.sharedMemPerBlock,
        "the device's properties");
  check(c.nCudaDevAttrComputeCapabilityMajor == p.major && c.nCudaDevAttrComputeCapabilityMinor == p.minor,
        "the compute capability");
  check(c.nReserved0 == 0x5a5a5a5a, "nReserved0 left as it was");
  check(nppGetStream() == nullptr, "nppGetStream is 0");

  cudaStream_t s = nullptr;
  cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  check(nppSetStream(s) == NPP_SUCCESS, "nppSetStream");
  check(nppGetStream() == s, "nppGetStream gives it back");
  std::memset(&c, 0, sizeof c);
  check(nppGetStreamContext(&c) == NPP_SUCCESS && c.hStream == s && c.nStreamFlags == cudaStreamNonBlocking,
        "the context has the stream and its flags");
  check(nppGetStreamContext(nullptr) == NPP_INVALID_HOST_POINTER_ERROR, "a null context is refused");

  // A _Ctx call on that context: an 8x4 image plus a constant.
  Npp8u *img = nullptr, *out = nullptr;
  cudaMalloc(&img, 32);
  cudaMalloc(&out, 32);
  const NppiSize roi{8, 4};
  check(nppiSet_8u_C1R_Ctx(40, img, 8, roi, c) == NPP_SUCCESS, "nppiSet_8u_C1R_Ctx");
  check(nppiAddC_8u_C1RSfs_Ctx(img, 8, 2, out, 8, roi, 0, c) == NPP_SUCCESS, "nppiAddC_8u_C1RSfs_Ctx");
  cudaStreamSynchronize(s);
  Npp8u h[32] = {};
  cudaMemcpy(h, out, 32, cudaMemcpyDeviceToHost);
  bool all = true;
  for (Npp8u v : h) all = all && v == 42;
  check(all, "the image is 42 everywhere");
  cudaFree(img);
  cudaFree(out);
  nppSetStream(nullptr);
  cudaStreamDestroy(s);
  std::printf("%s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
#endif
}
