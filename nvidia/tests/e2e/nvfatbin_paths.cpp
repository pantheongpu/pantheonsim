// nvFatbin as a program packaging its JIT output uses it: PTX for several
// architectures, a cubin from nvJitLink, LTO-IR and a host object's
// relocatable PTX in one fatbin, which cuModuleLoadData and
// cuModuleLoadFatBinary then load -- choosing the image for the device, as the
// driver does -- and nvJitLink takes as an input. Plus the result codes for
// each misuse.
//
//   nvfatbin_paths [object.o]
//
// The object, built by run_jit_link.sh from jitlink_lib.cu with nvcc -dc,
// feeds nvFatbinAddReloc; without it that check is skipped.
//
// Built against VirtualGPU's declarations (CUDA 12.0 has no nvFatbin.h),
// which follow NVIDIA's ABI: every check passes against NVIDIA's
// libnvfatbin 13.0 on an RTX 3060, and the fatbins it writes load there.
#include <cuda.h>
#ifdef VGPU_OWN_NVJITLINK_H   // a toolkit without nvJitLink (run_jit_link.sh)
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

#include "../../include/vgpu_nvfatbin.h"

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
static void is(int got, int want, const char* what) {
  check(got == want, what);
  if (got != want) std::printf("     returned %d\n", got);
}
#define IS(call, want) is((int)(call), (int)(want), #call " -> " #want)

#include "jitlink_ptx.inc"

// Two builds of one kernel, each writing which one ran: for sm_80, and for
// sm_90, which neither an A100 nor an RTX 3060 can run.
static std::string which(int arch, int value) {
  return ".version 7.8\n.target sm_" + std::to_string(arch) +
         "\n.address_size 64\n"
         ".visible .entry which(.param .u64 out)\n{\n"
         "\t.reg .b32 %r<2>;\n\t.reg .b64 %rd<3>;\n"
         "\tld.param.u64 %rd1, [out];\n\tcvta.to.global.u64 %rd2, %rd1;\n"
         "\tmov.u32 %r1, " + std::to_string(value) + ";\n\tst.global.u32 [%rd2], %r1;\n\tret;\n}\n";
}

static std::vector<char> image_of(nvFatbinHandle h) {
  size_t n = 0;
  if (nvFatbinSize(h, &n) != NVFATBIN_SUCCESS || n == 0) return {};
  std::vector<char> b(n);
  if (nvFatbinGet(h, b.data()) != NVFATBIN_SUCCESS) return {};
  return b;
}

// Runs `kernel` with one pointer argument over `threads` threads; the ints it
// wrote, or nothing when the image does not load or run.
static std::vector<int> run(CUmodule mod, const char* kernel, int threads) {
  CUfunction fn;
  if (cuModuleGetFunction(&fn, mod, kernel) != CUDA_SUCCESS) return {};
  CUdeviceptr d = 0;
  if (cuMemAlloc(&d, threads * sizeof(int)) != CUDA_SUCCESS) return {};
  void* args[] = {&d};
  std::vector<int> out(threads);
  const bool ok = cuLaunchKernel(fn, 1, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr) == CUDA_SUCCESS &&
                  cuCtxSynchronize() == CUDA_SUCCESS &&
                  cuMemcpyDtoH(out.data(), d, threads * sizeof(int)) == CUDA_SUCCESS;
  cuMemFree(d);
  return ok ? out : std::vector<int>{};
}

static bool computes_k(const std::vector<int>& out) {
  if (out.size() != 32) return false;
  for (int t = 0; t < 32; ++t)
    if (out[t] != expected(t)) return false;
  return true;
}

static std::vector<char> linked_cubin(std::vector<std::pair<nvJitLinkInputType, std::vector<char>>> in,
                                      const char* arch) {
  const char* opts[] = {arch};
  nvJitLinkHandle h;
  if (nvJitLinkCreate(&h, 1, opts) != NVJITLINK_SUCCESS) return {};
  bool ok = true;
  for (auto& [type, bytes] : in)
    ok = ok && nvJitLinkAddData(h, type, bytes.data(), bytes.size(), "input") == NVJITLINK_SUCCESS;
  size_t n = 0;
  ok = ok && nvJitLinkComplete(h) == NVJITLINK_SUCCESS && nvJitLinkGetLinkedCubinSize(h, &n) == NVJITLINK_SUCCESS;
  std::vector<char> cubin(ok ? n : 0);
  if (ok) nvJitLinkGetLinkedCubin(h, cubin.data());
  nvJitLinkDestroy(&h);
  return cubin;
}

// PTX as bytes, its terminator included: NVIDIA's nvJitLink reads PTX as a
// C string whatever size it is given, so without one it reads on into
// whatever follows (and fails now and then, as an RTX 3060 run showed).
static std::vector<char> bytes_of(const char* s) { return std::vector<char>(s, s + std::strlen(s) + 1); }

int main(int argc, char** argv) {
  if (cuInit(0) != CUDA_SUCCESS) {
    std::printf("FAIL: cuInit\n");
    return 1;
  }
  CUdevice dev;
  CUcontext ctx;
  cuDeviceGet(&dev, 0);
  cuDevicePrimaryCtxRetain(&ctx, dev);
  cuCtxSetCurrent(ctx);

  unsigned major = 0, minor = 0;
  IS(nvFatbinVersion(&major, &minor), NVFATBIN_SUCCESS);
  check(major >= 12, "nvFatbinVersion is a CUDA 12 or later version");
  for (int r = NVFATBIN_ERROR_INTERNAL; r <= NVFATBIN_ERROR_INTERNAL_PTX_OPTION; ++r)
    if (!nvFatbinGetErrorString(static_cast<nvFatbinResult>(r))) {
      check(false, "every result has a description");
      break;
    }

  // Creation.
  nvFatbinHandle h = nullptr;
  IS(nvFatbinCreate(nullptr, nullptr, 0), NVFATBIN_ERROR_NULL_POINTER);
  const char* bogus[] = {"-bogus"};
  IS(nvFatbinCreate(&h, bogus, 1), NVFATBIN_ERROR_UNRECOGNIZED_OPTION);
  if (h) nvFatbinDestroy(&h);
  const char* fine[] = {"-64", "-compress=false", "-host=linux", "-cuda", "-g"};
  IS(nvFatbinCreate(&h, fine, 5), NVFATBIN_SUCCESS);
  nvFatbinDestroy(&h);
  IS(nvFatbinCreate(&h, nullptr, 0), NVFATBIN_SUCCESS);
  size_t size = 0;
  IS(nvFatbinSize(h, &size), NVFATBIN_SUCCESS);
  check(size == 16, "an empty fatbin is its 16-byte container header");
  IS(nvFatbinSize(h, nullptr), NVFATBIN_ERROR_NULL_POINTER);

  // PTX, and what is refused on the way in. The architecture is the bare
  // number; PTX whose .target differs from it is still accepted.
  const std::string sm80 = which(80, 1), sm90 = which(90, 2);
  IS(nvFatbinAddPTX(h, sm80.c_str(), sm80.size() + 1, "sm_80", "which", nullptr), NVFATBIN_ERROR_INVALID_ARCH);
  IS(nvFatbinAddPTX(h, sm80.c_str(), sm80.size() + 1, nullptr, "which", nullptr), NVFATBIN_ERROR_NULL_POINTER);
  IS(nvFatbinAddPTX(h, nullptr, 10, "80", "which", nullptr), NVFATBIN_ERROR_NULL_POINTER);
  IS(nvFatbinAddPTX(h, sm80.c_str(), 0, "80", "which", nullptr), NVFATBIN_ERROR_EMPTY_INPUT);
  const std::string unversioned = "// no version\n.target sm_80\n.address_size 64\n";
  IS(nvFatbinAddPTX(h, unversioned.c_str(), unversioned.size() + 1, "80", "u", nullptr),
     NVFATBIN_ERROR_MISSING_PTX_VERSION);
  IS(nvFatbinAddPTX(h, sm90.c_str(), sm90.size() + 1, "90", "which", nullptr), NVFATBIN_SUCCESS);
  IS(nvFatbinAddPTX(h, sm80.c_str(), sm80.size(), "80", "which", "-O3"), NVFATBIN_SUCCESS);
  {
    // The driver JITs the newest PTX the device can run: sm_80's.
    std::vector<char> image = image_of(h);
    CUmodule mod;
    IS(cuModuleLoadData(&mod, image.data()), CUDA_SUCCESS);
    std::vector<int> out = run(mod, "which", 1);
    check(out.size() == 1 && out[0] == 1, "cuModuleLoadData picks the sm_80 PTX");
    cuModuleUnload(mod);
    IS(cuModuleLoadFatBinary(&mod, image.data()), CUDA_SUCCESS);
    out = run(mod, "which", 1);
    check(out.size() == 1 && out[0] == 1, "and so does cuModuleLoadFatBinary");
    cuModuleUnload(mod);
  }
  IS(nvFatbinDestroy(&h), NVFATBIN_SUCCESS);
  check(h == nullptr, "destroy clears the handle");
  IS(nvFatbinDestroy(nullptr), NVFATBIN_ERROR_NULL_POINTER);

  // A cubin from nvJitLink -- on VirtualGPU that is PTX, and it goes in as
  // what it is -- and the fatbin loads and runs it.
  std::vector<char> cubin = linked_cubin({{NVJITLINK_INPUT_PTX, bytes_of(kMain)},
                                          {NVJITLINK_INPUT_PTX, bytes_of(kLib)}},
                                         "-arch=sm_80");
  check(!cubin.empty(), "nvJitLink links the kernel");
  nvFatbinCreate(&h, nullptr, 0);
  IS(nvFatbinAddCubin(h, cubin.data(), cubin.size(), "80", "k"), NVFATBIN_SUCCESS);
  const char garbage[] = "this is not an ELF image";
  IS(nvFatbinAddCubin(h, garbage, sizeof garbage, "80", "g"), NVFATBIN_ERROR_ELF_SIZE_MISMATCH);
  IS(nvFatbinAddLTOIR(h, garbage, sizeof garbage, "80", "g", nullptr), NVFATBIN_ERROR_INTERNAL);
  IS(nvFatbinAddIndex(h, garbage, sizeof garbage, "g"), NVFATBIN_ERROR_INVALID_INDEX);
  {
    std::vector<char> image = image_of(h);
    CUmodule mod;
    IS(cuModuleLoadData(&mod, image.data()), CUDA_SUCCESS);
    check(computes_k(run(mod, "k", 32)), "a fatbin around nvJitLink's cubin runs it");
    cuModuleUnload(mod);
  }
  nvFatbinDestroy(&h);
  // A real cubin's architecture must be the one named. (VirtualGPU's cubin
  // is PTX, which carries no such claim.)
  if (!cubin.empty() && cubin[0] == 0x7f) {
    std::vector<char> sm86 = linked_cubin({{NVJITLINK_INPUT_PTX, bytes_of(kMain)},
                                           {NVJITLINK_INPUT_PTX, bytes_of(kLib)}},
                                          "-arch=sm_86");
    nvFatbinCreate(&h, nullptr, 0);
    IS(nvFatbinAddCubin(h, sm86.data(), sm86.size(), "80", "k"), NVFATBIN_ERROR_ELF_ARCH_MISMATCH);
    IS(nvFatbinAddCubin(h, sm86.data(), sm86.size(), "86", "k"), NVFATBIN_SUCCESS);
    nvFatbinDestroy(&h);
  }

  // A fatbin of the library's PTX is an nvJitLink input.
  nvFatbinCreate(&h, nullptr, 0);
  IS(nvFatbinAddPTX(h, kLib, std::strlen(kLib), "80", "lib", nullptr), NVFATBIN_SUCCESS);
  {
    std::vector<char> image = image_of(h);
    std::vector<char> linked = linked_cubin({{NVJITLINK_INPUT_PTX, bytes_of(kMain)},
                                             {NVJITLINK_INPUT_FATBIN, image}},
                                            "-arch=sm_80");
    CUmodule mod;
    check(!linked.empty() && cuModuleLoadData(&mod, linked.data()) == CUDA_SUCCESS &&
              computes_k(run(mod, "k", 32)),
          "nvJitLink links the kernel with a fatbin nvFatbin wrote");
  }
  nvFatbinDestroy(&h);

  // A host object's relocatable PTX, under its module identifier, once per
  // architecture; the fatbin it makes links with the kernel too.
  if (argc >= 2) {
    std::ifstream f(argv[1], std::ios::binary);
    const std::vector<char> object((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    nvFatbinCreate(&h, nullptr, 0);
    IS(nvFatbinAddReloc(h, object.data(), object.size()), NVFATBIN_SUCCESS);
    IS(nvFatbinAddReloc(h, object.data(), object.size()), NVFATBIN_ERROR_IDENTIFIER_REUSE);
    std::vector<char> image = image_of(h);
    std::vector<char> linked = linked_cubin({{NVJITLINK_INPUT_PTX, bytes_of(kMain)},
                                             {NVJITLINK_INPUT_FATBIN, image}},
                                            "-arch=sm_80");
    CUmodule mod;
    check(!linked.empty() && cuModuleLoadData(&mod, linked.data()) == CUDA_SUCCESS &&
              computes_k(run(mod, "k", 32)),
          "a host object's relocatable PTX, packaged, links with the kernel");
    nvFatbinDestroy(&h);
  } else {
    std::printf("note: no host object given; nvFatbinAddReloc is not checked\n");
  }

  std::printf(failures ? "FAIL: %d nvFatbin checks\n" : "PASS: every nvFatbin check\n", failures);
  return failures != 0;
}
