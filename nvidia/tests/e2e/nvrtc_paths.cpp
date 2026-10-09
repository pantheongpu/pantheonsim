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
#include <dlfcn.h>
#include <nvrtc.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
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

// A kernel from one module launched again after others have loaded, as the
// jiterator does: each of its kernels is a module of its own, and a function
// handle has to stay good while more arrive. The kernel reads a __device__
// variable, so the launch needs its module's symbol table.
static void function_outlives_later_modules() {
  const char* src = R"(
__device__ float bias = 5.0f;
extern "C" __global__ void add_bias(float* x, int n) { if (threadIdx.x < n) x[threadIdx.x] += bias; }
)";
  const std::string ptx = compile(src, "bias.cu", {"--gpu-architecture=compute_80"});
  if (ptx.empty()) { check(false, "the first module compiles"); return; }
  CUmodule first;
  CUfunction fn;
  CK(cuModuleLoadData(&first, ptx.c_str()));
  CK(cuModuleGetFunction(&fn, first, "add_bias"));
  std::vector<float> x(4, 1.0f);
  bool ok = run(ptx, "add_bias", x, 4) && x[0] == 6.0f;  // a module of its own, loaded and unloaded
  std::vector<CUmodule> more;
  const std::string other = compile("__device__ int k; extern \"C\" __global__ void other(int* p) { *p = k; }",
                                    "other.cu", {"--gpu-architecture=compute_80"});
  for (int i = 0; i < 20 && !other.empty(); ++i) {
    CUmodule m;
    if (cuModuleLoadData(&m, other.c_str()) == CUDA_SUCCESS) more.push_back(m);
  }
  CUdeviceptr d;
  cuMemAlloc(&d, 4 * sizeof(float));
  std::vector<float> y(4, 1.0f);
  cuMemcpyHtoD(d, y.data(), y.size() * sizeof(float));
  int n = 4;
  void* args[] = {&d, &n};
  ok = ok && more.size() == 20 && cuLaunchKernel(fn, 1, 1, 1, 4, 1, 1, 0, nullptr, args, nullptr) == CUDA_SUCCESS &&
       cuCtxSynchronize() == CUDA_SUCCESS;
  cuMemcpyDtoH(y.data(), d, y.size() * sizeof(float));
  check(ok && y[3] == 6.0f, "a function still launches, reading its module's globals, after 20 more modules load");
  cuMemFree(d);
  for (CUmodule m : more) cuModuleUnload(m);
  cuModuleUnload(first);
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


// ---- what the card's NVRTC does beyond PTX: measured on an RTX 3060 with CUDA 13.0's
// libnvrtc, and the same here ----

static bool is_vgpu_shim() { return dlsym(RTLD_DEFAULT, "vgpu_nvrtc_ptx_for_cubin") != nullptr; }
static bool ptx_engine() {
  const char* v = std::getenv("VGPU_SASS");
  const char* c = std::getenv("VGPU_NVRTC_CUBIN");
  return (v && v[0] == '0') || (c && std::strcmp(c, "ptx") == 0);
}
static std::string bytes_of(nvrtcResult (*size)(nvrtcProgram, size_t*), nvrtcResult (*get)(nvrtcProgram, char*),
                            nvrtcProgram p, size_t* n_out = nullptr) {
  size_t n = 0;
  if (size(p, &n) != NVRTC_SUCCESS) return {};
  std::string s(n, '\0');
  if (n && get(p, &s[0]) != NVRTC_SUCCESS) return {};
  if (n_out) *n_out = n;
  return s;
}

// The cubin of a real architecture is an ELF image of SASS, which the driver
// loads and runs; PTX alone for a virtual one (size 0 for the cubin).
static void cubin_is_real() {
  int cc_major = 0, cc_minor = 0;
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
  cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
  const int sm = cc_major * 10 + cc_minor;
  int n = 0;
  nvrtcGetNumSupportedArchs(&n);
  std::vector<int> archs(n);
  nvrtcGetSupportedArchs(archs.data());
  bool ascending = n > 0, has = false;
  for (int i = 0; i < n; ++i) {
    has = has || archs[i] == sm;
    if (i && archs[i] <= archs[i - 1]) ascending = false;
  }
  check(ascending, "the supported architectures come back in ascending order");
  if (!has) {
    std::printf("skip this device's architecture sm_%d is not one this NVRTC compiles for\n", sm);
    return;
  }
  const char* src = "extern \"C\" __global__ void fill(int* p) { p[threadIdx.x] = (int)threadIdx.x * 3 + 1; }\n";
  const std::string arch = "--gpu-architecture=sm_" + std::to_string(sm);
  nvrtcProgram p;
  nvrtcCreateProgram(&p, src, "fill.cu", 0, nullptr, nullptr);
  const char* opts[] = {arch.c_str()};
  const bool compiled = nvrtcCompileProgram(p, 1, opts) == NVRTC_SUCCESS;
  check(compiled, "a program compiles for the device's own sm_ architecture");
  if (!compiled) { nvrtcDestroyProgram(&p); return; }
  size_t csize = 0, psize = 0;
  const std::string cubin = bytes_of(nvrtcGetCUBINSize, nvrtcGetCUBIN, p, &csize);
  nvrtcGetPTXSize(p, &psize);
  if (ptx_engine()) {
    check(csize > 1 && cubin.find(".version") != std::string::npos, "the PTX engine's \"cubin\" is the PTX");
  } else {
    check(csize > 64 && cubin.compare(0, 4, "\x7f" "ELF") == 0, "the cubin of an sm_ target is an ELF image");
    check(psize > 1, "and the PTX comes with it");
  }
  CUmodule mod;
  CUfunction fn;
  bool ok = cuModuleLoadData(&mod, cubin.data()) == CUDA_SUCCESS && cuModuleGetFunction(&fn, mod, "fill") == CUDA_SUCCESS;
  int out[32] = {0};
  if (ok) {
    CUdeviceptr d;
    cuMemAlloc(&d, sizeof out);
    void* args[] = {&d};
    ok = cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr) == CUDA_SUCCESS && cuCtxSynchronize() == CUDA_SUCCESS;
    cuMemcpyDtoH(out, d, sizeof out);
    cuMemFree(d);
    cuModuleUnload(mod);
  }
  bool right = ok;
  for (int i = 0; i < 32; ++i) right = right && out[i] == i * 3 + 1;
  check(right, "the cubin loads through cuModuleLoadData and its kernel gives the right answers");
  if (is_vgpu_shim() && !ptx_engine()) {
    // The SASS engine refusing an instruction of the cubin (VGPU_SASS_REFUSE tests this): the PTX
    // NVRTC made it from runs in its place, as for a fatbin that carries both.
    setenv("VGPU_SASS_REFUSE", "EXIT", 1);
    CUmodule again;
    CUfunction fn2;
    int out2[32] = {0};
    bool ok2 = cuModuleLoadData(&again, cubin.data()) == CUDA_SUCCESS && cuModuleGetFunction(&fn2, again, "fill") == CUDA_SUCCESS;
    if (ok2) {
      CUdeviceptr d;
      cuMemAlloc(&d, sizeof out2);
      void* args[] = {&d};
      ok2 = cuLaunchKernel(fn2, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr) == CUDA_SUCCESS && cuCtxSynchronize() == CUDA_SUCCESS;
      cuMemcpyDtoH(out2, d, sizeof out2);
      cuMemFree(d);
      cuModuleUnload(again);
    }
    unsetenv("VGPU_SASS_REFUSE");
    for (int i = 0; i < 32; ++i) ok2 = ok2 && out2[i] == i * 3 + 1;
    check(ok2, "and where the SASS engine refuses an instruction, the PTX it came from runs instead");
  }
  nvrtcDestroyProgram(&p);

  // A virtual architecture has PTX and no cubin.
  nvrtcProgram v;
  nvrtcCreateProgram(&v, src, "fill.cu", 0, nullptr, nullptr);
  const char* vopts[] = {"--gpu-architecture=compute_80"};
  nvrtcCompileProgram(v, 1, vopts);
  size_t vc = 99, vp = 0;
  const nvrtcResult rc = nvrtcGetCUBINSize(v, &vc);
  nvrtcGetPTXSize(v, &vp);
  check(rc == NVRTC_SUCCESS && vc == 0 && vp > 1, "for compute_80 the PTX is there and the cubin's size is 0");
  nvrtcDestroyProgram(&v);
}

// LTO-IR (-dlto) and OptiX-IR (--optix-ir) are NVVM bitcode that only NVIDIA's compiler makes; with
// the toolkit's NVRTC they come back as it makes them, without it they are refused by name.
static void lto_and_optix() {
  const char* src = "extern \"C\" __global__ void k(int* p) { p[threadIdx.x] = (int)threadIdx.x * 2; }\n";
  struct Case { const char* flag; bool lto; };
  for (const Case& c : {Case{"-dlto", true}, Case{"--optix-ir", false}}) {
    nvrtcProgram p;
    nvrtcCreateProgram(&p, src, "k.cu", 0, nullptr, nullptr);
    const char* opts[] = {"--gpu-architecture=compute_80", c.flag};
    const nvrtcResult rc = nvrtcCompileProgram(p, 2, opts);
    if (rc == NVRTC_ERROR_INVALID_OPTION && is_vgpu_shim() && log_of(p).find("libnvrtc") != std::string::npos) {
      std::printf("skip %s needs the toolkit's libnvrtc, which is not installed\n", c.flag);
      nvrtcDestroyProgram(&p);
      continue;
    }
    size_t ptx = 0, cubin = 7, lto = 0, optix = 0;
    nvrtcGetPTXSize(p, &ptx);
    nvrtcGetCUBINSize(p, &cubin);
    nvrtcGetLTOIRSize(p, &lto);
    nvrtcGetOptiXIRSize(p, &optix);
    const std::string bits = c.lto ? bytes_of(nvrtcGetLTOIRSize, nvrtcGetLTOIR, p)
                                   : bytes_of(nvrtcGetOptiXIRSize, nvrtcGetOptiXIR, p);
    check(rc == NVRTC_SUCCESS && (c.lto ? lto : optix) > 4 && (c.lto ? optix : lto) == 0 && cubin == 0,
          c.lto ? "-dlto makes LTO-IR and neither a cubin nor OptiX-IR" : "--optix-ir makes OptiX-IR and neither a cubin nor LTO-IR");
    check(ptx == 1, c.lto ? "and no PTX (its size is the terminator alone)" : "and no PTX either");
    check(bits.size() > 4 && bits.compare(0, 4, "\xed" "CN\x7f") == 0, "the bitcode starts with NVVM's magic (0x7f4e43ed)");
    nvrtcDestroyProgram(&p);
  }
  nvrtcProgram p;
  nvrtcCreateProgram(&p, src, "k.cu", 0, nullptr, nullptr);
  const char* both[] = {"--gpu-architecture=compute_80", "-dlto", "--optix-ir"};
  const nvrtcResult rc = nvrtcCompileProgram(p, 3, both);
  check(rc == NVRTC_ERROR_INVALID_OPTION, "-dlto and --optix-ir together are refused as options");
  nvrtcDestroyProgram(&p);
}

// What every query answers before a compile, with no output pointer, and for no program.
static void argument_checks() {
  nvrtcProgram p;
  nvrtcCreateProgram(&p, "extern \"C\" __global__ void k() {}", "k.cu", 0, nullptr, nullptr);
  size_t n = 77;
  char c[8];
  check(nvrtcGetPTXSize(p, &n) == NVRTC_SUCCESS && nvrtcGetCUBINSize(p, &n) == NVRTC_SUCCESS &&
            nvrtcGetLTOIRSize(p, &n) == NVRTC_SUCCESS && nvrtcGetOptiXIRSize(p, &n) == NVRTC_SUCCESS &&
            nvrtcGetProgramLogSize(p, &n) == NVRTC_SUCCESS,
        "the size queries succeed before any compile");
  check(nvrtcGetPTX(p, c) == NVRTC_SUCCESS && nvrtcGetCUBIN(p, c) == NVRTC_SUCCESS && nvrtcGetLTOIR(p, c) == NVRTC_SUCCESS &&
            nvrtcGetOptiXIR(p, c) == NVRTC_SUCCESS,
        "and so do the outputs, there being nothing to copy");
  const char* opts[] = {"--gpu-architecture=sm_80"};
  nvrtcCompileProgram(p, 1, opts);
  check(nvrtcGetCUBINSize(p, nullptr) == NVRTC_ERROR_INVALID_INPUT && nvrtcGetCUBIN(p, nullptr) == NVRTC_ERROR_INVALID_INPUT &&
            nvrtcGetLTOIR(p, nullptr) == NVRTC_ERROR_INVALID_INPUT && nvrtcGetOptiXIR(p, nullptr) == NVRTC_ERROR_INVALID_INPUT,
        "a null destination is INVALID_INPUT");
  check(nvrtcGetCUBINSize(nullptr, &n) == NVRTC_ERROR_INVALID_PROGRAM && nvrtcGetCUBIN(nullptr, c) == NVRTC_ERROR_INVALID_PROGRAM &&
            nvrtcGetLTOIRSize(nullptr, &n) == NVRTC_ERROR_INVALID_PROGRAM && nvrtcGetOptiXIRSize(nullptr, &n) == NVRTC_ERROR_INVALID_PROGRAM,
        "a null program is INVALID_PROGRAM");
  nvrtcProgram none = nullptr;
  nvrtcProgram q;
  check(nvrtcDestroyProgram(&none) == NVRTC_ERROR_INVALID_PROGRAM && nvrtcDestroyProgram(nullptr) == NVRTC_ERROR_INVALID_PROGRAM &&
            nvrtcCreateProgram(nullptr, "x", "a", 0, nullptr, nullptr) == NVRTC_ERROR_INVALID_PROGRAM &&
            nvrtcCreateProgram(&q, nullptr, "a", 0, nullptr, nullptr) == NVRTC_ERROR_INVALID_INPUT &&
            nvrtcCreateProgram(&q, "x", "a", -1, nullptr, nullptr) == NVRTC_ERROR_INVALID_INPUT &&
            nvrtcCreateProgram(&q, "x", "a", 1, nullptr, nullptr) == NVRTC_ERROR_INVALID_INPUT,
        "a null handle, source or header array is refused as NVIDIA's refuses it");
  check(nvrtcAddNameExpression(p, "k") == NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION,
        "a name expression after the compile is refused by name");
  nvrtcDestroyProgram(&p);
  // The names of the result codes, and what an unknown one reads as.
  static const char* kNames[] = {"NVRTC_SUCCESS", "NVRTC_ERROR_OUT_OF_MEMORY", "NVRTC_ERROR_PROGRAM_CREATION_FAILURE",
                                 "NVRTC_ERROR_INVALID_INPUT", "NVRTC_ERROR_INVALID_PROGRAM", "NVRTC_ERROR_INVALID_OPTION",
                                 "NVRTC_ERROR_COMPILATION", "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE",
                                 "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION",
                                 "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION", "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID",
                                 "NVRTC_ERROR_INTERNAL_ERROR", "NVRTC_ERROR_TIME_FILE_WRITE_FAILED",
                                 "NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED", "NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED",
                                 "NVRTC_ERROR_PCH_CREATE", "NVRTC_ERROR_CANCELLED", "NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED"};
  int major = 0, minor = 0;
  nvrtcVersion(&major, &minor);
  // 13.0 has the names up to 17; CUDA 12's NVRTC answers only as far as it goes.
  const int top = is_vgpu_shim() || major >= 13 ? 17 : 11;
  bool names = true;
  for (int i = 0; i <= top; ++i) names = names && std::strcmp(nvrtcGetErrorString(static_cast<nvrtcResult>(i)), kNames[i]) == 0;
  check(names, "the result codes have NVIDIA's names");
  check(std::strcmp(nvrtcGetErrorString(static_cast<nvrtcResult>(99)), "NVRTC_ERROR unknown") == 0 ||
            (major < 13 && !is_vgpu_shim()),
        "an unknown code reads \"NVRTC_ERROR unknown\"");
}

// Precompiled headers, the time table, and the flow callback: NVRTC's own, reached by name since
// older headers lack them. Skipped where the library has no such entry (CUDA 12 before 12.8) or,
// for this shim without the toolkit's NVRTC, refuses the option.
static void pch_time_and_flow() {
  using PchStatus = int (*)(nvrtcProgram);   // (results past 11 are ints: CUDA 12's enum has no such values)
  using HeapGet = int (*)(size_t*);
  using HeapSet = int (*)(size_t);
  using FlowSet = int (*)(nvrtcProgram, int (*)(void*, void*), void*);
  auto status = reinterpret_cast<PchStatus>(dlsym(RTLD_DEFAULT, "nvrtcGetPCHCreateStatus"));
  auto heap_get = reinterpret_cast<HeapGet>(dlsym(RTLD_DEFAULT, "nvrtcGetPCHHeapSize"));
  auto heap_set = reinterpret_cast<HeapSet>(dlsym(RTLD_DEFAULT, "nvrtcSetPCHHeapSize"));
  auto flow = reinterpret_cast<FlowSet>(dlsym(RTLD_DEFAULT, "nvrtcSetFlowCallback"));
  if (!status || !heap_get || !heap_set || !flow) {
    std::printf("skip this libnvrtc has no precompiled-header or flow-callback entry points\n");
    return;
  }
  // The heap rounds a request up to 4096.
  size_t before = 0, h = 0;
  heap_get(&before);
  bool round = heap_set(12345) == NVRTC_SUCCESS && heap_get(&h) == NVRTC_SUCCESS && h == 16384;
  round = round && heap_set(1) == NVRTC_SUCCESS && heap_get(&h) == NVRTC_SUCCESS && h == 4096;
  round = round && heap_set(1 << 20) == NVRTC_SUCCESS && heap_get(&h) == NVRTC_SUCCESS && h == (1u << 20);
  heap_set(before);
  check(round && heap_get(nullptr) == NVRTC_ERROR_INVALID_INPUT, "the PCH heap size rounds up to 4096, and a null destination is refused");

  const char* hdr = "#pragma once\n__device__ inline int twice(int x) { return 2 * x; }\n";
  const char* hdr_name = "hdr.h";
  const char* src = "#include \"hdr.h\"\nextern \"C\" __global__ void k(int* p) { p[threadIdx.x] = twice((int)threadIdx.x); }\n";
  char dir[] = "/tmp/vgpu_nvrtc_pch_XXXXXX";
  if (!mkdtemp(dir)) { check(false, "a scratch directory"); return; }
  const std::string d = dir;
  auto build = [&](std::vector<std::string> extra, std::string* log, int* st) {
    nvrtcProgram p;
    nvrtcCreateProgram(&p, src, "p.cu", 1, &hdr, &hdr_name);
    std::vector<std::string> o = {"--gpu-architecture=compute_80"};
    o.insert(o.end(), extra.begin(), extra.end());
    std::vector<const char*> argv;
    for (auto& s : o) argv.push_back(s.c_str());
    const int rc = static_cast<int>(nvrtcCompileProgram(p, (int)argv.size(), argv.data()));
    *log = log_of(p);
    *st = status(p);
    nvrtcDestroyProgram(&p);
    return rc;
  };
  std::string log;
  int st;
  int rc = build({"--create-pch=" + d + "/a.pch"}, &log, &st);
  if (rc == NVRTC_ERROR_INVALID_OPTION && is_vgpu_shim() && log.find("libnvrtc") != std::string::npos) {
    std::printf("skip precompiled headers need the toolkit's libnvrtc, which is not installed\n");
  } else {
    struct stat sb;
    check(rc == NVRTC_SUCCESS && st == NVRTC_SUCCESS && stat((d + "/a.pch").c_str(), &sb) == 0 && sb.st_size > 0,
          "--create-pch writes the header and the creation status reads SUCCESS");
    const int rc2 = build({"--use-pch=" + d + "/a.pch"}, &log, &st);
    check(rc2 == NVRTC_SUCCESS && st == 13 && log.find("using precompiled header") != std::string::npos,
          "--use-pch uses it, and no creation was attempted");
    const int rc3 = build({"--use-pch=" + d + "/missing.pch"}, &log, &st);
    check(rc3 == NVRTC_ERROR_COMPILATION && log.find("missing.pch") != std::string::npos, "a missing PCH file is a compile error naming it");
    rc = build({"--pch", "--pch-dir=" + d}, &log, &st);
    const int again = build({"--pch", "--pch-dir=" + d}, &log, &st);
    check(rc == NVRTC_SUCCESS && again == NVRTC_SUCCESS && st == 13, "--pch creates a header once and then reuses it");
    // A heap too small for the header: the program still compiles, the status says so, the required size is reported.
    heap_set(4096);
    nvrtcProgram p;
    nvrtcCreateProgram(&p, src, "p.cu", 1, &hdr, &hdr_name);
    const std::string tiny = "--create-pch=" + d + "/tiny.pch";
    const char* o[] = {"--gpu-architecture=compute_80", tiny.c_str()};
    rc = static_cast<int>(nvrtcCompileProgram(p, 2, o));
    using Req = int (*)(nvrtcProgram, size_t*);
    auto required = reinterpret_cast<Req>(dlsym(RTLD_DEFAULT, "nvrtcGetPCHHeapSizeRequired"));
    size_t need = 0;
    check(rc == NVRTC_SUCCESS && status(p) == 14 && required && required(p, &need) == NVRTC_SUCCESS && need > 4096,
          "a PCH heap that is too small reports PCH_CREATE_HEAP_EXHAUSTED and the size it needs");
    nvrtcDestroyProgram(&p);
    heap_set(before);
  }
  // --time writes a CSV table of the compile's phases; an unwritable path is its own error.
  nvrtcProgram t;
  nvrtcCreateProgram(&t, src, "p.cu", 1, &hdr, &hdr_name);
  const std::string csv = "--time=" + d + "/time.csv";
  const char* to[] = {"--gpu-architecture=compute_80", csv.c_str()};
  rc = static_cast<int>(nvrtcCompileProgram(t, 2, to));
  if (rc == NVRTC_ERROR_INVALID_OPTION && is_vgpu_shim()) {
    std::printf("skip --time needs the toolkit's libnvrtc, which is not installed\n");
  } else {
    std::string first;
    if (FILE* f = std::fopen((d + "/time.csv").c_str(), "r")) {
      char line[200] = {0};
      if (std::fgets(line, sizeof line, f)) first = line;
      std::fclose(f);
    }
    check(rc == NVRTC_SUCCESS && first.rfind("File name, phase name, metric, unit", 0) == 0, "--time writes a CSV table with NVRTC's header row");
    nvrtcProgram u;
    nvrtcCreateProgram(&u, src, "p.cu", 1, &hdr, &hdr_name);
    const char* bad[] = {"--gpu-architecture=compute_80", "--time=/nonexistent_vgpu_dir/t.csv"};
    const int brc = static_cast<int>(nvrtcCompileProgram(u, 2, bad));
    check(brc == 12 && log_of(u).find("failed to open file") != std::string::npos,
          "an unwritable --time path is NVRTC_ERROR_TIME_FILE_WRITE_FAILED");
    nvrtcDestroyProgram(&u);
  }
  nvrtcDestroyProgram(&t);
  (void)std::system(("rm -rf '" + d + "'").c_str());

  // The flow callback may cancel a compile; nothing else about when or how often it is called is promised.
  static int calls;
  struct Cb { static int cancel(void*, void*) { ++calls; return 1; } static int go_on(void*, void*) { ++calls; return 0; } };
  nvrtcProgram f;
  nvrtcCreateProgram(&f, "extern \"C\" __global__ void k() {}", "k.cu", 0, nullptr, nullptr);
  check(flow(f, nullptr, nullptr) == NVRTC_ERROR_INVALID_INPUT, "a null flow callback is refused");
  flow(f, Cb::cancel, nullptr);
  const char* fo[] = {"--gpu-architecture=sm_80"};
  calls = 0;
  const int frc = static_cast<int>(nvrtcCompileProgram(f, 1, fo));
  size_t cs = 7;
  check(frc == 16 && calls >= 1 && nvrtcGetCUBINSize(f, &cs) == NVRTC_SUCCESS && cs == 0,
        "a callback that returns 1 cancels the compile: NVRTC_ERROR_CANCELLED, no cubin");
  nvrtcDestroyProgram(&f);
  nvrtcProgram g;
  nvrtcCreateProgram(&g, "extern \"C\" __global__ void k() {}", "k.cu", 0, nullptr, nullptr);
  flow(g, Cb::go_on, nullptr);
  calls = 0;
  check(nvrtcCompileProgram(g, 1, fo) == NVRTC_SUCCESS && calls >= 1, "and one that returns 0 lets it finish, having been called");
  nvrtcDestroyProgram(&g);
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
  function_outlives_later_modules();
  compile_error();
  cubin_is_real();
  lto_and_optix();
  argument_checks();
  pch_time_and_flow();
  cuDevicePrimaryCtxRelease(dev);
  std::printf(failures ? "FAIL: %d NVRTC checks\n" : "PASS: every NVRTC check\n", failures);
  return failures ? 1 : 0;
}
