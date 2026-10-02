// nvJitLink as JIT frameworks call it (numba-cuda's and cuda-python's
// linkers, CuPy): link PTX modules for an architecture, load the linked
// cubin with cuModuleLoadData, launch, check the answer -- plus the device
// linker's rules (one definition per symbol, a strong one over a weak one,
// file-scope names private to their module, undefined references refused),
// a host object's and a static library's device code, and the result codes
// for each misuse.
//
//   nvjitlink_paths [object.o library.a lto.fatbin]
//
// The object, the library and an LTO-IR fatbin are built by
// run_nvjitlink.sh from jitlink_lib.cu (nvcc -dc, ar, nvcc -dlto -fatbin);
// without them those checks are skipped.
//
// Built against NVIDIA's nvJitLink.h where the toolkit has one, so
// it calls the versioned entry points a real program imports. Every check
// passes against NVIDIA's libnvJitLink 13.0 on an RTX 3060, where the cubin
// is SASS. VirtualGPU's cubin is the linked PTX, which is the one place the
// two may answer differently; where they do, both answers are named.
#include <cuda.h>
#include <dlfcn.h>
#ifdef VGPU_OWN_NVJITLINK_H   // a toolkit without nvJitLink (run_jit_link.sh)
#include "../../include/vgpu_nvjitlink.h"
#else
#include <nvJitLink.h>
#endif

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

// Result codes and input types newer than CUDA 12.0's header, by value.
constexpr int kInternal = 6, kUnrecognizedInput = 8, kNullInput = 10, kIncompatibleOptions = 11,
              kIncorrectInputType = 12, kUnrecognizedArch = 16, kUnsupportedArch = 17,
              kLtoNotEnabled = 18;
const nvJitLinkInputType kInputAny = static_cast<nvJitLinkInputType>(10);

#include "jitlink_ptx.inc"

static std::string error_log(nvJitLinkHandle h) {
  size_t n = 0;
  if (nvJitLinkGetErrorLogSize(h, &n) != NVJITLINK_SUCCESS || n == 0) return {};
  std::string s(n, '\0');
  nvJitLinkGetErrorLog(h, &s[0]);
  return s;
}

static nvJitLinkHandle create(std::vector<const char*> opts) {
  nvJitLinkHandle h = nullptr;
  if (nvJitLinkCreate(&h, (uint32_t)opts.size(), opts.data()) != NVJITLINK_SUCCESS) {
    std::printf("FAIL nvJitLinkCreate\n");
    ++failures;
  }
  return h;
}

static void add(nvJitLinkHandle h, const char* ptx, const char* name) {
  const int r = nvJitLinkAddData(h, NVJITLINK_INPUT_PTX, ptx, std::strlen(ptx), name);
  if (r != NVJITLINK_SUCCESS) {
    std::printf("FAIL nvJitLinkAddData(%s) returned %d: %s\n", name, r, error_log(h).c_str());
    ++failures;
  }
}

static std::vector<char> cubin_of(nvJitLinkHandle h) {
  size_t n = 0;
  if (nvJitLinkGetLinkedCubinSize(h, &n) != NVJITLINK_SUCCESS || n == 0) return {};
  std::vector<char> c(n);
  if (nvJitLinkGetLinkedCubin(h, c.data()) != NVJITLINK_SUCCESS) return {};
  return c;
}

// Loads a linked image and runs k over 32 threads; true when every thread's
// answer is expected(t).
static bool runs_right(const std::vector<char>& image) {
  CUmodule mod;
  CUfunction fn;
  if (image.empty() || cuModuleLoadData(&mod, image.data()) != CUDA_SUCCESS) return false;
  bool ok = cuModuleGetFunction(&fn, mod, "k") == CUDA_SUCCESS;
  CUdeviceptr d = 0;
  ok = ok && cuMemAlloc(&d, 32 * sizeof(int)) == CUDA_SUCCESS;
  void* args[] = {&d};
  ok = ok && cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr) == CUDA_SUCCESS &&
       cuCtxSynchronize() == CUDA_SUCCESS;
  int out[32] = {};
  ok = ok && cuMemcpyDtoH(out, d, sizeof out) == CUDA_SUCCESS;
  for (int t = 0; ok && t < 32; ++t)
    if (out[t] != expected(t)) {
      std::printf("     thread %d: %d, expected %d\n", t, out[t], expected(t));
      ok = false;
    }
  if (d) cuMemFree(d);
  cuModuleUnload(mod);
  return ok;
}

static void options(unsigned major) {
  nvJitLinkHandle h = nullptr;
  // A failed create still hands back a handle, so the log can say why.
  const char* none[] = {"-O3"};
  IS(nvJitLinkCreate(&h, 1, none), NVJITLINK_ERROR_MISSING_ARCH);
  check(h != nullptr, "a handle comes back from a create that failed");
  if (h) IS(nvJitLinkDestroy(&h), NVJITLINK_SUCCESS);
  const char* bogus[] = {"-arch=sm_80", "-bogus"};
  IS(nvJitLinkCreate(&h, 2, bogus), NVJITLINK_ERROR_UNRECOGNIZED_OPTION);
  check(h && error_log(h).find("-bogus") != std::string::npos, "the log names the unrecognized option");
  if (h) nvJitLinkDestroy(&h);
  const char* virt[] = {"-arch=compute_80"};
  IS(nvJitLinkCreate(&h, 1, virt), kIncompatibleOptions);   // a virtual arch needs -ptx
  if (h) nvJitLinkDestroy(&h);
  const char* ptx_only[] = {"-arch=sm_80", "-ptx"};
  IS(nvJitLinkCreate(&h, 2, ptx_only), kIncompatibleOptions);   // and -ptx needs -lto
  if (h) nvJitLinkDestroy(&h);
  const char* future[] = {"-arch=sm_999"};
  IS(nvJitLinkCreate(&h, 1, future), kUnrecognizedArch);
  if (h) nvJitLinkDestroy(&h);
  // CUDA 13 dropped every architecture before Turing.
  const char* volta[] = {"-arch=sm_70"};
  const int r = nvJitLinkCreate(&h, 1, volta);
  check(major >= 13 ? r == kUnsupportedArch : r == NVJITLINK_SUCCESS,
        "sm_70 is unsupported from CUDA 13 on, and supported before");
  if (h) nvJitLinkDestroy(&h);
  // Code-generation options are accepted, though nothing here generates code.
  const char* many[] = {"-arch=sm_80", "-O3", "-maxrregcount=32", "-lineinfo", "-ftz=1",
                        "-prec-div=0", "-fma=1", "-Xptxas=-v", "-no-cache", "-lto"};
  IS(nvJitLinkCreate(&h, 10, many), NVJITLINK_SUCCESS);
  if (h) nvJitLinkDestroy(&h);
  IS(nvJitLinkCreate(&h, 0, nullptr), kNullInput);
  IS(nvJitLinkCreate(nullptr, 1, virt), kNullInput);
  IS(nvJitLinkDestroy(nullptr), kNullInput);
}

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

  // nvJitLinkVersion arrived after CUDA 12.0, whose library does not have
  // it; looked up rather than linked, so this builds against either.
  unsigned major = 12, minor = 0;
  using VersionFn = nvJitLinkResult (*)(unsigned*, unsigned*);
  if (auto v = reinterpret_cast<VersionFn>(dlsym(RTLD_DEFAULT, "nvJitLinkVersion"))) {
    IS(v(&major, &minor), NVJITLINK_SUCCESS);
    check(major >= 12, "nvJitLinkVersion is a CUDA 12 or later version");
  }
  options(major);

  // The link itself.
  nvJitLinkHandle h = create({"-arch=sm_80"});
  if (!h) return 1;
  add(h, kMain, "main.ptx");
  size_t size = 0;
  IS(nvJitLinkGetLinkedCubinSize(h, &size), kInternal);   // nothing linked yet
  add(h, kLib, "lib.ptx");
  IS(nvJitLinkComplete(h), NVJITLINK_SUCCESS);
  check(error_log(h).empty(), "a clean link logs no errors");
  std::vector<char> cubin = cubin_of(h);
  check(!cubin.empty(), "the linked cubin has a size and contents");
  check(runs_right(cubin), "the linked cubin loads and k computes across both modules");
  IS(nvJitLinkGetLinkedCubinSize(h, nullptr), kNullInput);
  IS(nvJitLinkComplete(h), kInternal);   // once only
  IS(nvJitLinkAddData(h, NVJITLINK_INPUT_PTX, kDup, std::strlen(kDup), "late"), kInternal);
  // NVIDIA's hands out PTX only for -lto -ptx (INVALID_INPUT otherwise);
  // VirtualGPU's cubin is PTX, and it hands that out.
  const bool ptx_cubin = !cubin.empty() && cubin[0] != 0x7f;
  {
    const int r = nvJitLinkGetLinkedPtxSize(h, &size);
    std::string ptx;
    if (r == NVJITLINK_SUCCESS) {
      ptx.assign(size, '\0');
      nvJitLinkGetLinkedPtx(h, &ptx[0]);
    }
    check(ptx_cubin ? r == NVJITLINK_SUCCESS && ptx.find(".entry k") != std::string::npos
                    : r == NVJITLINK_ERROR_INVALID_INPUT,
          "linked PTX: the module (VirtualGPU) or INVALID_INPUT without -lto -ptx (NVIDIA)");
  }
  IS(nvJitLinkDestroy(&h), NVJITLINK_SUCCESS);
  check(h == nullptr, "destroy clears the handle");

  // VirtualGPU's cubin is PTX, so it links again as a CUBIN input -- the
  // round trip a framework that caches linked code makes.
  if (ptx_cubin) {
    h = create({"-arch=sm_80"});
    IS(nvJitLinkAddData(h, NVJITLINK_INPUT_CUBIN, cubin.data(), cubin.size(), "cached.cubin"),
       NVJITLINK_SUCCESS);
    IS(nvJitLinkComplete(h), NVJITLINK_SUCCESS);
    check(runs_right(cubin_of(h)), "a VirtualGPU cubin relinks as an input");
    nvJitLinkDestroy(&h);
  }

  // An undefined reference fails the link and is named.
  h = create({"-arch=sm_80"});
  add(h, kMain, "main.ptx");
  IS(nvJitLinkComplete(h), kInternal);
  {
    const std::string log = error_log(h);
    check(log.find("Undefined reference to 'add_base' in 'main.ptx'") != std::string::npos &&
              log.find("Undefined reference to 'base'") != std::string::npos,
          "the error log names each undefined reference and its module");
  }
  nvJitLinkDestroy(&h);

  // A second strong definition is reported, and the first one is kept: the
  // link still succeeds. VirtualGPU's reports it in the error log; NVIDIA's
  // 13.0 does that in some processes and prints it to stdout in others.
  h = create({"-arch=sm_80"});
  add(h, kMain, "main.ptx");
  add(h, kLib, "lib.ptx");
  add(h, kDup, "dup.ptx");
  IS(nvJitLinkComplete(h), NVJITLINK_SUCCESS);
  if (ptx_cubin)
    check(error_log(h).find("Multiple definition of 'add_base' in 'dup.ptx', first defined in "
                            "'lib.ptx'") != std::string::npos,
          "a second definition is named in the error log");
  check(runs_right(cubin_of(h)), "and the first definition is the one that runs");
  nvJitLinkDestroy(&h);

  // A strong definition beats a weak one that came first.
  h = create({"-arch=sm_80"});
  add(h, kMain, "main.ptx");
  add(h, kWeak, "weak.ptx");
  add(h, kLib, "lib.ptx");
  IS(nvJitLinkComplete(h), NVJITLINK_SUCCESS);
  check(error_log(h).empty(), "a weak and a strong definition link without complaint");
  check(runs_right(cubin_of(h)), "and the strong one runs");
  nvJitLinkDestroy(&h);

  // Inputs refused when added.
  h = create({"-arch=sm_80"});
  IS(nvJitLinkAddData(h, NVJITLINK_INPUT_PTX, kHopper, std::strlen(kHopper), "hopper.ptx"),
     NVJITLINK_ERROR_PTX_COMPILE);   // .target sm_90 cannot be compiled for sm_80
  IS(nvJitLinkAddData(h, NVJITLINK_INPUT_FATBIN, kLib, std::strlen(kLib), "lib.ptx"),
     kIncorrectInputType);
  const char garbage[] = "this is not device code";
  IS(nvJitLinkAddData(h, NVJITLINK_INPUT_PTX, garbage, sizeof garbage, "garbage"), kIncorrectInputType);
  IS(nvJitLinkAddData(h, kInputAny, garbage, sizeof garbage, "garbage"), kUnrecognizedInput);
  IS(nvJitLinkAddData(h, NVJITLINK_INPUT_NONE, kLib, std::strlen(kLib), "none"), kIncorrectInputType);
  IS(nvJitLinkAddData(h, NVJITLINK_INPUT_PTX, kLib, 0, "empty"), NVJITLINK_ERROR_INVALID_INPUT);
  IS(nvJitLinkAddFile(h, NVJITLINK_INPUT_PTX, "/nonexistent/lib.ptx"), NVJITLINK_ERROR_INVALID_INPUT);
  check(error_log(h).find("/nonexistent/lib.ptx") != std::string::npos, "a missing file is named");
  IS(nvJitLinkAddFile(h, NVJITLINK_INPUT_PTX, nullptr), kNullInput);
  // Nothing refused was kept: the two good modules still link.
  IS(nvJitLinkAddData(h, kInputAny, kMain, std::strlen(kMain), "main.ptx"), NVJITLINK_SUCCESS);
  add(h, kLib, "lib.ptx");
  IS(nvJitLinkComplete(h), NVJITLINK_SUCCESS);
  check(runs_right(cubin_of(h)), "refused inputs leave the link as it was");
  nvJitLinkDestroy(&h);

  // Device code from a host object and from a static library, as nvcc -dc
  // writes them (PTX in the object's __nv_relfatbin section).
  if (argc >= 3) {
    for (int i = 1; i <= 2; ++i) {
      const nvJitLinkInputType type = i == 1 ? NVJITLINK_INPUT_OBJECT : NVJITLINK_INPUT_LIBRARY;
      for (nvJitLinkInputType t : {type, kInputAny}) {
        h = create({"-arch=sm_80"});
        add(h, kMain, "main.ptx");
        const int r = nvJitLinkAddFile(h, t, argv[i]);
        check(r == NVJITLINK_SUCCESS, i == 1 ? "a host object's device code is an input"
                                            : "a static library's device code is an input");
        if (r != NVJITLINK_SUCCESS) std::printf("     %s\n", error_log(h).c_str());
        IS(nvJitLinkComplete(h), NVJITLINK_SUCCESS);
        check(runs_right(cubin_of(h)), "and links with the kernel");
        nvJitLinkDestroy(&h);
      }
    }
    // A host object is not PTX.
    h = create({"-arch=sm_80"});
    IS(nvJitLinkAddFile(h, NVJITLINK_INPUT_PTX, argv[1]), kIncorrectInputType);
    nvJitLinkDestroy(&h);
  } else {
    std::printf("note: no object or library given; their checks are skipped\n");
  }

  // LTO-IR, as nvcc -dlto puts it in a fatbin, needs -lto. With it NVIDIA's
  // compiles the bitcode and links it; VirtualGPU cannot compile NVVM bitcode,
  // and refuses it by name instead.
  if (argc >= 4) {
    h = create({"-arch=sm_80"});
    IS(nvJitLinkAddFile(h, NVJITLINK_INPUT_FATBIN, argv[3]), kLtoNotEnabled);
    check(error_log(h).find("-lto") != std::string::npos, "the log says LTO-IR needs -lto");
    nvJitLinkDestroy(&h);
    h = create({"-arch=sm_80", "-lto"});
    add(h, kMain, "main.ptx");
    const int r = nvJitLinkAddFile(h, NVJITLINK_INPUT_FATBIN, argv[3]);
    if (ptx_cubin) {
      is(r, NVJITLINK_ERROR_NVVM_COMPILE, "LTO-IR with -lto is refused (VirtualGPU)");
      check(error_log(h).find("LTO-IR input '") != std::string::npos, "and the input is named");
    } else {
      is(r, NVJITLINK_SUCCESS, "LTO-IR with -lto is an input (NVIDIA)");
      IS(nvJitLinkComplete(h), NVJITLINK_SUCCESS);
      check(runs_right(cubin_of(h)), "and links with the kernel's PTX");
    }
    nvJitLinkDestroy(&h);
  }

  std::printf(failures ? "FAIL: %d nvJitLink checks\n" : "PASS: every nvJitLink check\n", failures);
  return failures != 0;
}
