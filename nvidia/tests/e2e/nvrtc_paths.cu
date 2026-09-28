// NVRTC as PyTorch's jiterator and the JIT frameworks use it: compile a
// string to PTX, load it through the driver API, launch it, check the result.
//
// The first program is shaped like a jiterator kernel. It defines int64_t and
// INFINITY itself, as jiterator source does, because NVRTC provides no host
// headers; a compile that pulled in the host's <stdint.h> (nvcc pre-includes
// cuda_runtime.h, which does) fails on it. It also asks for a real
// architecture, sm_80, which NVRTC compiles to SASS, while VirtualGPU needs
// PTX; the shim targets the matching virtual architecture instead.
#include <cuda.h>
#include <nvrtc.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}

#define CK(x)                                                                    \
  do {                                                                           \
    const int r_ = (int)(x);                                                     \
    if (r_ != 0) {                                                               \
      std::printf("FAIL %s returned %d\n", #x, r_);                             \
      ++failures;                                                                \
      return;                                                                    \
    }                                                                            \
  } while (0)

static std::string log_of(nvrtcProgram p) {
  size_t n = 0;
  nvrtcGetProgramLogSize(p, &n);
  std::string s(n, '\0');
  if (n) nvrtcGetProgramLog(p, &s[0]);
  return s;
}

static std::string compile(const char* src, const char* name, std::vector<const char*> opts,
                           std::vector<std::string> exprs = {}, std::vector<std::string>* lowered = nullptr,
                           int nheaders = 0, const char* const* headers = nullptr,
                           const char* const* header_names = nullptr) {
  nvrtcProgram p;
  if (nvrtcCreateProgram(&p, src, name, nheaders, headers, header_names) != NVRTC_SUCCESS) return {};
  for (const auto& e : exprs) nvrtcAddNameExpression(p, e.c_str());
  const nvrtcResult rc = nvrtcCompileProgram(p, (int)opts.size(), opts.data());
  std::string ptx;
  if (rc == NVRTC_SUCCESS) {
    size_t n = 0;
    nvrtcGetPTXSize(p, &n);
    ptx.assign(n, '\0');
    nvrtcGetPTX(p, &ptx[0]);
    for (const auto& e : exprs) {
      const char* low = nullptr;
      nvrtcGetLoweredName(p, e.c_str(), &low);
      if (lowered) lowered->push_back(low ? low : "");
    }
  } else {
    std::printf("     compile log: %s\n", log_of(p).substr(0, 400).c_str());
  }
  nvrtcDestroyProgram(&p);
  return ptx;
}

static bool run(const std::string& ptx, const char* kernel, std::vector<float>& io, int n) {
  CUmodule mod;
  CUfunction fn;
  if (cuModuleLoadData(&mod, ptx.c_str()) != CUDA_SUCCESS) return false;
  if (cuModuleGetFunction(&fn, mod, kernel) != CUDA_SUCCESS) return false;
  CUdeviceptr d;
  cuMemAlloc(&d, io.size() * sizeof(float));
  cuMemcpyHtoD(d, io.data(), io.size() * sizeof(float));
  void* args[] = {&d, &n};
  const bool ok = cuLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr) == CUDA_SUCCESS &&
                  cuCtxSynchronize() == CUDA_SUCCESS;
  cuMemcpyDtoH(io.data(), d, io.size() * sizeof(float));
  cuMemFree(d);
  cuModuleUnload(mod);
  return ok;
}

static void jiterator_shaped() {
  const char* src = R"(
typedef long long int int64_t;
typedef unsigned int uint32_t;
#define POS_INFINITY __int_as_float(0x7f800000)
#define INFINITY POS_INFINITY
#define NAN __int_as_float(0x7fffffff)
template <typename T> __device__ T clamp_inf(T x) { return x > T(1e30) ? INFINITY : x; }
extern "C" __global__ void jit_kernel(float* x, int n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = clamp_inf(x[i] * 2.0f + 1.0f);
}
)";
  const std::string ptx = compile(src, "default_program", {"--std=c++17", "--gpu-architecture=sm_80", "-default-device"});
  {  // What PyTorch loads for an sm_ target: the CUBIN, which here is the PTX.
    nvrtcProgram p;
    nvrtcCreateProgram(&p, src, "default_program", 0, nullptr, nullptr);
    const char* opts[] = {"--std=c++17", "--gpu-architecture=sm_80", "-default-device"};
    size_t n = 0;
    const bool ok = nvrtcCompileProgram(p, 3, opts) == NVRTC_SUCCESS && nvrtcGetCUBINSize(p, &n) == NVRTC_SUCCESS && n > 0;
    std::string cubin(n, '\0');
    CUmodule mod;
    CUfunction fn;
    const bool loads = ok && nvrtcGetCUBIN(p, &cubin[0]) == NVRTC_SUCCESS &&
                       cuModuleLoadData(&mod, cubin.data()) == CUDA_SUCCESS &&
                       cuModuleGetFunction(&fn, mod, "jit_kernel") == CUDA_SUCCESS;
    check(loads, "for sm_80 it hands back a CUBIN that cuModuleLoadData loads");
    if (loads) cuModuleUnload(mod);
    nvrtcDestroyProgram(&p);
  }
  check(!ptx.empty(), "a jiterator-shaped program (its own int64_t, INFINITY, NAN) compiles for sm_80");
  if (ptx.empty()) return;
  check(ptx.find(".target sm_80") != std::string::npos, "the PTX targets the virtual architecture of sm_80");
  std::vector<float> x(64);
  for (int i = 0; i < 64; ++i) x[i] = i == 63 ? 1e31f : 0.5f * i;
  const bool ran = run(ptx, "jit_kernel", x, 64);
  bool right = ran;
  for (int i = 0; i < 63; ++i) right = right && x[i] == 0.5f * i * 2.0f + 1.0f;
  check(right && std::isinf(x[63]), "and runs, with the right answers");
}

static void name_expressions_and_headers() {
  const char* header = "#pragma once\ntemplate <int K> __device__ float scale(float v) { return v * K; }\n";
  const char* header_name = "scale.h";
  const char* src = R"(
#include "scale.h"
template <int K> __global__ void times(float* x, int n) {
  int i = threadIdx.x;
  if (i < n) x[i] = scale<K>(x[i]);
}
)";
  std::vector<std::string> lowered;
  const std::string ptx = compile(src, "templated.cu", {"--gpu-architecture=compute_80"}, {"times<3>"}, &lowered, 1,
                                  &header, &header_name);
  check(!ptx.empty() && lowered.size() == 1 && lowered[0].rfind("_Z", 0) == 0,
        "a template kernel from a string header compiles and its name expression lowers to a mangled name");
  if (ptx.empty() || lowered.empty()) return;
  std::vector<float> x(8, 2.0f);
  const bool ran = run(ptx, lowered[0].c_str(), x, 8);
  check(ran && x[0] == 6.0f && x[7] == 6.0f, "the kernel found by its lowered name runs");
}

// A struct passed by value, as the jiterator passes its arrays of pointers
// and its offset calculators: in PTX a .b8 array parameter, launched through
// the driver API's kernelParams.
static void struct_parameter() {
  const char* src = R"(
struct Pack { float scale[4]; };
extern "C" __global__ void packed(Pack p, float* x) { x[threadIdx.x] *= p.scale[threadIdx.x % 4]; }
)";
  const std::string ptx = compile(src, "packed.cu", {"--gpu-architecture=compute_80"});
  if (ptx.empty()) { check(false, "a kernel taking a struct by value compiles"); return; }
  CUmodule mod;
  CUfunction fn;
  CK(cuModuleLoadData(&mod, ptx.c_str()));
  CK(cuModuleGetFunction(&fn, mod, "packed"));
  struct Pack { float scale[4]; } pack = {{1.0f, 2.0f, 3.0f, 4.0f}};
  std::vector<float> x(8, 1.0f);
  CUdeviceptr d;
  cuMemAlloc(&d, x.size() * sizeof(float));
  cuMemcpyHtoD(d, x.data(), x.size() * sizeof(float));
  void* args[] = {&pack, &d};
  const bool ran = cuLaunchKernel(fn, 1, 1, 1, 8, 1, 1, 0, nullptr, args, nullptr) == CUDA_SUCCESS &&
                   cuCtxSynchronize() == CUDA_SUCCESS;
  cuMemcpyDtoH(x.data(), d, x.size() * sizeof(float));
  check(ran && x[0] == 1.0f && x[3] == 4.0f && x[6] == 3.0f,
        "a kernel taking a 16-byte struct by value launches through cuLaunchKernel with it intact");
  cuMemFree(d);
  cuModuleUnload(mod);
}

static void compile_error() {
  nvrtcProgram p;
  CK(nvrtcCreateProgram(&p, "__global__ void k() { undeclared_thing(); }", "bad.cu", 0, nullptr, nullptr));
  const char* opts[] = {"--gpu-architecture=compute_80"};
  const nvrtcResult rc = nvrtcCompileProgram(p, 1, opts);
  const std::string log = log_of(p);
  check(rc == NVRTC_ERROR_COMPILATION && log.find("undeclared_thing") != std::string::npos,
        "a compile error comes back as NVRTC_ERROR_COMPILATION with the diagnostic in the log");
  nvrtcDestroyProgram(&p);
}

int main() {
  CUdevice dev;
  CUcontext ctx;
  // The primary context: cuCtxCreate's signature differs between CUDA 12 and 13.
  if (cuInit(0) || cuDeviceGet(&dev, 0) || cuDevicePrimaryCtxRetain(&ctx, dev) || cuCtxSetCurrent(ctx)) {
    std::printf("FAIL: no CUDA context\n");
    return 1;
  }
  jiterator_shaped();
  name_expressions_and_headers();
  struct_parameter();
  compile_error();
  cuDevicePrimaryCtxRelease(dev);
  std::printf(failures ? "FAIL: %d NVRTC checks\n" : "PASS: every NVRTC check\n", failures);
  return failures ? 1 : 0;
}
