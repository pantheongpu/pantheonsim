// Runtime compilation end to end: NVRTC turns a source string into PTX, the
// driver API loads it, and the kernel runs. This is the path CuPy, Numba and
// Triton take, so it is worth proving against hardware rather than in
// isolation -- the same binary runs on a physical GPU with NVIDIA's NVRTC and
// on VirtualGPU with this one, and the results have to agree.
#include <nvrtc.h>
#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define NR(x) do { nvrtcResult r_ = (x); if (r_ != NVRTC_SUCCESS) { \
  printf("%s -> %s\n", #x, nvrtcGetErrorString(r_)); return 1; } } while (0)
#define DR(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { \
  const char* n_ = nullptr; cuGetErrorName(r_, &n_); \
  printf("%s -> %s\n", #x, n_ ? n_ : "?"); return 1; } } while (0)

static const char* kSource = R"(
extern "C" __global__ void scale_add(const float* a, const float* b, float* c, float k, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = k * a[i] + b[i];
}

template <typename T>
__global__ void fill_ramp(T* out, T start, T step, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = start + step * (T)i;
}
template __global__ void fill_ramp<float>(float*, float, float, int);
)";

int main() {
  int major = 0, minor = 0;
  NR(nvrtcVersion(&major, &minor));
  printf("nvrtc major %d\n", major);   // minor moves between patch releases

  int narch = 0;
  NR(nvrtcGetNumSupportedArchs(&narch));
  std::vector<int> archs(narch);
  NR(nvrtcGetSupportedArchs(archs.data()));
  bool has86 = false;
  bool ascending = true;
  for (int i = 0; i < narch; ++i) {
    if (archs[i] == 86) has86 = true;
    if (i && archs[i] <= archs[i - 1]) ascending = false;
  }
  printf("supported archs include sm_86: %s, ascending: %s\n", has86 ? "yes" : "no",
         ascending ? "yes" : "no");

  // Compile for the device that will run it. A fixed architecture works until
  // the machine is older than it: PTX runs forward, not backward, and the
  // driver rejects a newer target with CUDA_ERROR_INVALID_PTX.
  DR(cuInit(0));
  int cc_major = 0, cc_minor = 0;
  CUdevice probe;
  DR(cuDeviceGet(&probe, 0));
  DR(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, probe));
  DR(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, probe));
  char arch_opt[64];
  snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d%d", cc_major, cc_minor);
  char target_needle[32];
  snprintf(target_needle, sizeof(target_needle), ".target sm_%d%d", cc_major, cc_minor);

  nvrtcProgram prog;
  NR(nvrtcCreateProgram(&prog, kSource, "jit.cu", 0, nullptr, nullptr));
  NR(nvrtcAddNameExpression(prog, "fill_ramp<float>"));
  const char* opts[] = {arch_opt, "-std=c++14"};
  const nvrtcResult crc = nvrtcCompileProgram(prog, 2, opts);
  size_t logsz = 0;
  NR(nvrtcGetProgramLogSize(prog, &logsz));
  std::string log(logsz, '\0');
  NR(nvrtcGetProgramLog(prog, &log[0]));
  printf("compile: %s, log empty: %s\n", crc == NVRTC_SUCCESS ? "ok" : "FAILED",
         logsz <= 1 ? "yes" : "no");
  if (crc != NVRTC_SUCCESS) { printf("%s\n", log.c_str()); return 1; }

  const char* lowered = nullptr;
  NR(nvrtcGetLoweredName(prog, "fill_ramp<float>", &lowered));
  // The mangled name is the compiler's, so check its shape rather than
  // hard-coding a string: it must be an Itanium-ABI name for this template.
  printf("lowered name mangled: %s, names the template: %s\n",
         std::strncmp(lowered, "_Z", 2) == 0 ? "yes" : "no",
         std::strstr(lowered, "fill_ramp") ? "yes" : "no");

  size_t ptxsz = 0;
  NR(nvrtcGetPTXSize(prog, &ptxsz));
  std::string ptx(ptxsz, '\0');
  NR(nvrtcGetPTX(prog, &ptx[0]));
  printf("ptx targets this device: %s, has scale_add entry: %s\n",
         ptx.find(target_needle) != std::string::npos ? "yes" : "no",
         ptx.find(".entry scale_add") != std::string::npos ? "yes" : "no");

  CUdevice dev;
  DR(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  // Retain the primary context rather than cuCtxCreate: the latter changed
  // signature in CUDA 13, and this is what the runtime API uses anyway.
  DR(cuDevicePrimaryCtxRetain(&ctx, dev));
  DR(cuCtxSetCurrent(ctx));
  CUmodule mod;
  DR(cuModuleLoadData(&mod, ptx.c_str()));

  const int n = 1024;
  CUdeviceptr da, db, dc;
  DR(cuMemAlloc(&da, n * sizeof(float)));
  DR(cuMemAlloc(&db, n * sizeof(float)));
  DR(cuMemAlloc(&dc, n * sizeof(float)));

  // The JIT-compiled template fills A; the extern "C" kernel consumes it.
  CUfunction ramp, scale;
  DR(cuModuleGetFunction(&ramp, mod, lowered));
  DR(cuModuleGetFunction(&scale, mod, "scale_add"));
  float start = 1.0f, step = 0.25f;
  int count = n;
  void* ramp_args[] = {&da, &start, &step, &count};
  DR(cuLaunchKernel(ramp, (n + 127) / 128, 1, 1, 128, 1, 1, 0, nullptr, ramp_args, nullptr));
  float bstart = 0.5f, bstep = -0.125f;
  void* ramp_b[] = {&db, &bstart, &bstep, &count};
  DR(cuLaunchKernel(ramp, (n + 127) / 128, 1, 1, 128, 1, 1, 0, nullptr, ramp_b, nullptr));
  float k = 3.0f;
  void* scale_args[] = {&da, &db, &dc, &k, &count};
  DR(cuLaunchKernel(scale, (n + 127) / 128, 1, 1, 128, 1, 1, 0, nullptr, scale_args, nullptr));
  DR(cuCtxSynchronize());

  std::vector<float> h(n);
  DR(cuMemcpyDtoH(h.data(), dc, n * sizeof(float)));
  double sum = 0;
  int wrong = 0;
  for (int i = 0; i < n; ++i) {
    sum += h[i];
    const float want = 3.0f * (1.0f + 0.25f * i) + (0.5f - 0.125f * i);
    if (h[i] != want) ++wrong;
  }
  printf("jit result sum=%.4f first=%.4f last=%.4f wrong=%d\n", sum, h[0], h[n - 1], wrong);

  cuMemFree(da); cuMemFree(db); cuMemFree(dc);
  cuModuleUnload(mod);
  cuDevicePrimaryCtxRelease(dev);
  nvrtcDestroyProgram(&prog);

  {  // A program that does not compile must report the failure, with a log.
    nvrtcProgram bad;
    NR(nvrtcCreateProgram(&bad, "__global__ void k() { this_does_not_exist(); }", "bad.cu", 0,
                          nullptr, nullptr));
    const nvrtcResult r = nvrtcCompileProgram(bad, 0, nullptr);
    size_t sz = 0;
    NR(nvrtcGetProgramLogSize(bad, &sz));
    printf("bad program: %s, log non-empty: %s\n",
           r == NVRTC_ERROR_COMPILATION ? "NVRTC_ERROR_COMPILATION" : "unexpected success",
           sz > 1 ? "yes" : "no");
    nvrtcDestroyProgram(&bad);
  }
  return 0;
}
