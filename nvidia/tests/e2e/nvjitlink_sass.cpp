// nvJitLink linking machine code: relocatable SASS (nvcc -rdc / -dc, SASS
// only) linked into a cubin, loaded with cuModuleLoadData and run. The device
// code is sass_link_main.cu and sass_link_lib.cu, which between them make
// every kind of reference only a link resolves (see sass_link_main.cu); the
// answers are checked exactly.
//
//   nvjitlink_sass <arch> <main.cubin> <lib.cubin> [<other-arch.cubin> [<lib.o> <lib.a>]]
//
// run_jit_link_sass.sh builds the inputs. Also checked: an undefined reference
// fails the link and names the symbol; a second definition is reported and
// the link still succeeds; a cubin for an architecture -arch cannot run is
// refused when added; a SASS link has no PTX to hand out; a host object's and
// a static library's relocatable SASS link like a cubin.
//
// Every check passes against NVIDIA's libnvJitLink 13.0 and driver on an RTX
// 3060 (sm_86), and with VirtualGPU's libnvJitLink in its place there, whose
// cubin NVIDIA's driver then runs.
#include <cuda.h>
#ifdef VGPU_OWN_NVJITLINK_H   // a toolkit without nvJitLink (run_jit_link_sass.sh)
#include "../../include/vgpu_nvjitlink.h"
#else
#include <nvJitLink.h>
#endif

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
static void is(long long got, long long want, const char* what) {
  check(got == want, what);
  if (got != want) std::printf("     got %lld, want %lld\n", got, want);
}

static std::vector<char> slurp(const char* path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static std::string error_log(nvJitLinkHandle h) {
  size_t n = 0;
  if (nvJitLinkGetErrorLogSize(h, &n) != NVJITLINK_SUCCESS || n == 0) return {};
  std::string s(n, '\0');
  nvJitLinkGetErrorLog(h, &s[0]);
  return s;
}

struct Input {
  nvJitLinkInputType type;
  const char* path;
};

// Links the inputs; returns the result of nvJitLinkComplete (or of the first
// add that failed), the cubin in *out and the error log in *log.
static int link(const std::string& arch, const std::vector<Input>& inputs, std::vector<char>* out, std::string* log) {
  const std::string opt = "-arch=" + arch;
  const char* opts[] = {opt.c_str()};
  nvJitLinkHandle h = nullptr;
  int r = nvJitLinkCreate(&h, 1, opts);
  if (r != NVJITLINK_SUCCESS) return r;
  for (const Input& in : inputs) {
    const std::vector<char> bytes = slurp(in.path);
    r = nvJitLinkAddData(h, in.type, bytes.data(), bytes.size(), in.path);
    if (r != NVJITLINK_SUCCESS) {
      *log = error_log(h);
      nvJitLinkDestroy(&h);
      return r;
    }
  }
  r = nvJitLinkComplete(h);
  *log = error_log(h);
  if (r == NVJITLINK_SUCCESS && out) {
    size_t n = 0;
    nvJitLinkGetLinkedCubinSize(h, &n);
    out->assign(n, 0);
    nvJitLinkGetLinkedCubin(h, out->data());
    // A SASS link has no PTX to give.
    size_t pn = 0;
    is(nvJitLinkGetLinkedPtxSize(h, &pn), NVJITLINK_ERROR_INVALID_INPUT, "a SASS link has no linked PTX");
  }
  nvJitLinkDestroy(&h);
  return r;
}

// Loads a linked cubin and runs both modules' kernels, checking every answer.
static void run(const std::vector<char>& cubin, const char* what) {
  std::printf("-- %s\n", what);
  check(cubin.size() > 64 && cubin[0] == 0x7f && cubin[1] == 'E', "the linked image is a cubin");
  CUmodule m = nullptr;
  if (cuModuleLoadData(&m, cubin.data()) != CUDA_SUCCESS) {
    check(false, "cuModuleLoadData takes it");
    return;
  }
  CUfunction run_all = nullptr, lib_kernel = nullptr;
  check(cuModuleGetFunction(&run_all, m, "_Z7run_allPii") == CUDA_SUCCESS, "the main module's kernel is there");
  check(cuModuleGetFunction(&lib_kernel, m, "_Z10lib_kernelPi") == CUDA_SUCCESS, "and the library's");
  if (!run_all || !lib_kernel) return;
  CUdeviceptr out;
  cuMemAlloc(&out, 64 * sizeof(int));
  cuMemsetD32(out, 0, 64);
  int sel = 5;
  void* args[] = {&out, &sel};
  check(cuLaunchKernel(run_all, 1, 1, 1, 64, 1, 1, 64 * sizeof(int), nullptr, args, nullptr) == CUDA_SUCCESS,
        "run_all launches, with 256 bytes of dynamic shared memory");
  check(cuCtxSynchronize() == CUDA_SUCCESS, "and finishes");
  int got[10] = {};
  cuMemcpyDtoH(got, out, sizeof got);
  is(got[0], 5 + 17 + 3, "a call into the library, which reads both modules' variables");
  is(got[1], 17, "the library's variable");
  is(got[2], 4 + 6, "the library's constants, direct and indexed");
  is(got[3], 30 + 60, "the main module's constants, direct and indexed");
  is(got[4], 1001 + 2001, "each module's file-scope function of the same name is its own");
  is(got[5], 42 - 4 - 1, "calls through pointers to functions in both modules");
  is(got[6], 11 + 11, "a template both modules instantiate");
  is(got[7], 300 + 6, "pointers initialised to other variables' addresses");
  is(got[8], 11, "malloc in linked code");
  is(got[9], 64 * 5, "no two shared arrays overlap (kernel, functions in both modules, dynamic)");
  CUdeviceptr counter;
  size_t bytes = 0;
  int count = 0;
  check(cuModuleGetGlobal(&counter, &bytes, m, "main_counter") == CUDA_SUCCESS && bytes == 4,
        "cuModuleGetGlobal finds a variable");
  cuMemcpyDtoH(&count, counter, 4);
  is(count, 64, "which every thread added to");
  check(cuModuleGetGlobal(&counter, &bytes, m, "lib_counter") == CUDA_SUCCESS, "and the library's");
  cuMemcpyDtoH(&count, counter, 4);
  is(count, 64, "which every thread added to");
  // New constants, through the module, read by the library's kernel.
  CUdeviceptr c;
  check(cuModuleGetGlobal(&c, &bytes, m, "main_const") == CUDA_SUCCESS && bytes == 32, "a __constant__ array");
  int consts[8];
  for (int i = 0; i < 8; ++i) consts[i] = 1000 + i;
  cuMemcpyHtoD(c, consts, sizeof consts);
  void* args2[] = {&out};
  check(cuLaunchKernel(lib_kernel, 1, 1, 1, 8, 1, 1, 0, nullptr, args2, nullptr) == CUDA_SUCCESS &&
            cuCtxSynchronize() == CUDA_SUCCESS,
        "the library's kernel runs");
  cuMemcpyDtoH(got, out, 8 * sizeof(int));
  bool right = true;
  for (int t = 0; t < 8; ++t) right = right && got[t] == t + 17 + 3 + 1000 + t;
  check(right, "and reads the other module's constants as the host set them");
  cuMemFree(out);
  cuModuleUnload(m);
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::printf("usage: %s <arch> <main.cubin> <lib.cubin> [other-arch.cubin [lib.o lib.a]]\n", argv[0]);
    return 2;
  }
  const std::string arch = argv[1];
  const char* main_cubin = argv[2];
  const char* lib_cubin = argv[3];
  CUdevice dev;
  CUcontext ctx;
  if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
      cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS || cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
    std::printf("FAIL: no CUDA device\n");
    return 1;
  }
  std::vector<char> cubin;
  std::string log;

  is(link(arch, {{NVJITLINK_INPUT_CUBIN, main_cubin}, {NVJITLINK_INPUT_CUBIN, lib_cubin}}, &cubin, &log),
     NVJITLINK_SUCCESS, "two relocatable cubins link");
  if (!log.empty()) std::printf("     log: %s", log.c_str());
  run(cubin, "the linked cubin");

  is(link(arch, {{NVJITLINK_INPUT_CUBIN, lib_cubin}, {NVJITLINK_INPUT_CUBIN, main_cubin}}, &cubin, &log),
     NVJITLINK_SUCCESS, "the other order links");
  run(cubin, "linked library first");

  // A reference nothing defines.
  is(link(arch, {{NVJITLINK_INPUT_CUBIN, main_cubin}}, nullptr, &log), 6 /* NVJITLINK_ERROR_INTERNAL */,
     "an undefined reference fails the link");
  check(log.find("Undefined reference to 'lib_var'") != std::string::npos, "and names the symbol");

  // A second definition does not fail the link. (NVIDIA's 13.0 prints
  // "Multiple definition of ..." to stderr, and its driver then refuses the
  // image; VirtualGPU's puts the line in the error log and links the first.)
  is(link(arch, {{NVJITLINK_INPUT_CUBIN, main_cubin}, {NVJITLINK_INPUT_CUBIN, lib_cubin}, {NVJITLINK_INPUT_CUBIN, lib_cubin}},
          &cubin, &log),
     NVJITLINK_SUCCESS, "a module linked twice still links");

  if (argc > 4) {
    is(link(arch, {{NVJITLINK_INPUT_CUBIN, main_cubin}, {NVJITLINK_INPUT_CUBIN, argv[4]}}, nullptr, &log),
       NVJITLINK_ERROR_INVALID_INPUT, "a cubin for an architecture -arch cannot run is refused");
  }
  if (argc > 6) {
    is(link(arch, {{NVJITLINK_INPUT_CUBIN, main_cubin}, {NVJITLINK_INPUT_OBJECT, argv[5]}}, &cubin, &log),
       NVJITLINK_SUCCESS, "a host object's relocatable SASS links");
    run(cubin, "linked with a host object");
    is(link(arch, {{NVJITLINK_INPUT_LIBRARY, argv[6]}, {NVJITLINK_INPUT_CUBIN, main_cubin}}, &cubin, &log),
       NVJITLINK_SUCCESS, "and a static library's");
    run(cubin, "linked with a static library");
  }
  std::printf(failures ? "FAIL: %d checks\n" : "PASS\n", failures);
  return failures != 0;
}
