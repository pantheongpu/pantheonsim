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
  compile_error();
  cuDevicePrimaryCtxRelease(dev);
  std::printf(failures ? "FAIL: %d NVRTC checks\n" : "PASS: every NVRTC check\n", failures);
  return failures ? 1 : 0;
}
