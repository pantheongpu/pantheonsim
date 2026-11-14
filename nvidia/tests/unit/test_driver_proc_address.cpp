// cuGetProcAddress resolves a function's base name by the CUDA version asked for, as an RTX 3060 does
// (pantheonsim #626), and the driver exports cuProfilerInitialize (pantheonsim #627).
//
// A caller asks for the plain name and the version it was built against, and gets the export with that ABI:
// cuGraphExecUpdate at 12000 is cuGraphExecUpdate_v2 (cuda-core 1.3 asks for it so, and stopped every numba-cuda
// launch when the shim found nothing). nvidia/tests/data/driver_proc_address.rtx3060.txt is what the card's driver
// answered for every name of NVIDIA's typedef headers at 11040, 12000, 12040 and 13000, with the default flags and
// with the per-thread default stream (tools/probes/driver_proc_address_probe.cpp); the shim must hand back the
// export of the same name, with the same result and status, for each.
//
// No toolkit header is needed: the few prototypes are declared here.
#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "vtest.hpp"

extern "C" {
int cuInit(unsigned int flags);
int cuDeviceGet(int* dev, int ordinal);
int cuDevicePrimaryCtxRetain(void** ctx, int dev);
int cuDevicePrimaryCtxRelease_v2(int dev);
int cuCtxSetCurrent(void* ctx);
int cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, unsigned long long flags);
int cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, unsigned long long flags, int* status);
int cuProfilerInitialize(const char* configFile, const char* outputFile, int outputMode);
int cuMemAlloc_v2(unsigned long long* dptr, size_t bytes);
int cuMemFree_v2(unsigned long long dptr);
int cuMemGetAddressRange(unsigned long long* base, size_t* size, unsigned long long dptr);
}

namespace {

constexpr int kSuccess = 0, kNotInitialized = 3, kNotSupported = 801;
constexpr int kFound = 0, kSymbolNotFound = 1, kVersionNotSufficient = 2;

void* exported(const char* name) { return dlsym(RTLD_DEFAULT, name); }

// Rows where the card's answer is its platform's rather than the driver's: the WSL2 driver it was recorded on has
// no EGL, which a Linux driver has and the shim exports (and refuses when called: test_driver_gl_stubs).
const std::map<std::string, std::string> kPlatform = {
    {"cuEventCreateFromEGLSync", "cuEventCreateFromEGLSync"},
};

}  // namespace

// First, before anything initialises the driver.
VTEST(profiler_initialize_before_cuInit_is_not_initialized) {
  VCHECK_EQ(cuProfilerInitialize("cfg.txt", "out.txt", 0), kNotInitialized);
  VCHECK_EQ(cuProfilerInitialize(nullptr, nullptr, 0), kNotInitialized);
}

VTEST(profiler_initialize_is_not_supported_whatever_the_arguments) {
  VCHECK_EQ(cuInit(0), kSuccess);
  VCHECK(exported("cuProfilerInitialize") != nullptr);
  // No current context.
  VCHECK_EQ(cuProfilerInitialize("cfg.txt", "out.txt", 0), kNotSupported);
  VCHECK_EQ(cuProfilerInitialize(nullptr, nullptr, 0), kNotSupported);
  VCHECK_EQ(cuProfilerInitialize("/nonexistent/cfg", "out.txt", 1), kNotSupported);
  VCHECK_EQ(cuProfilerInitialize("cfg.txt", "out.txt", -1), kNotSupported);
  // With one.
  int dev = -1;
  void* ctx = nullptr;
  VCHECK_EQ(cuDeviceGet(&dev, 0), kSuccess);
  VCHECK_EQ(cuDevicePrimaryCtxRetain(&ctx, dev), kSuccess);
  VCHECK_EQ(cuCtxSetCurrent(ctx), kSuccess);
  VCHECK_EQ(cuProfilerInitialize("cfg.txt", "out.txt", 0), kNotSupported);
  VCHECK_EQ(cuProfilerInitialize(nullptr, "/nonexistent/out", 2), kNotSupported);
  VCHECK_EQ(cuCtxSetCurrent(nullptr), kSuccess);
  VCHECK_EQ(cuDevicePrimaryCtxRelease_v2(dev), kSuccess);
  // And a loader finds it by name.
  void* fn = nullptr;
  int status = -1;
  VCHECK_EQ(cuGetProcAddress_v2("cuProfilerInitialize", &fn, 13000, 0, &status), kSuccess);
  VCHECK_EQ(status, kFound);
  VCHECK(fn == exported("cuProfilerInitialize"));
}

// The calls of the issue: cuda-core 1.3.0 and the reproducer.
VTEST(graph_functions_resolve_by_version) {
  VCHECK_EQ(cuInit(0), kSuccess);
  struct Case {
    const char* name;
    int version;
    const char* want;
  } cases[] = {
      {"cuGraphExecUpdate", 11040, "cuGraphExecUpdate"},
      {"cuGraphExecUpdate", 12000, "cuGraphExecUpdate_v2"},
      {"cuGraphExecUpdate", 13000, "cuGraphExecUpdate_v2"},
      // The card hands the 11.0 ABI for every version from 11.0 on (not cuGraphInstantiateWithFlags, which
      // cuda.h 12 renames the call to; that name is asked for by its own name).
      {"cuGraphInstantiate", 11040, "cuGraphInstantiate_v2"},
      {"cuGraphInstantiate", 13000, "cuGraphInstantiate_v2"},
      {"cuGraphInstantiateWithFlags", 13000, "cuGraphInstantiateWithFlags"},
      {"cuCtxCreate", 12000, "cuCtxCreate_v3"},
      {"cuCtxCreate", 13000, "cuCtxCreate_v4"},
      {"cuStreamGetCaptureInfo", 13000, "cuStreamGetCaptureInfo_v3"},
      {"cuMemAlloc", 13000, "cuMemAlloc_v2"},
  };
  for (const auto& c : cases) {
    void* fn = nullptr;
    int status = -1;
    VCHECK_EQ(cuGetProcAddress_v2(c.name, &fn, c.version, 0, &status), kSuccess);
    if (fn != exported(c.want)) std::printf("  %s at %d is not %s\n", c.name, c.version, c.want);
    VCHECK_EQ(status, kFound);
    VCHECK(fn != nullptr && fn == exported(c.want));
    // The status-less form answers the same.
    void* fn1 = nullptr;
    VCHECK_EQ(cuGetProcAddress(c.name, &fn1, c.version, 0), kSuccess);
    VCHECK(fn1 == fn);
  }
}

VTEST(a_missing_or_too_new_name_is_success_with_no_function) {
  VCHECK_EQ(cuInit(0), kSuccess);
  void* fn = reinterpret_cast<void*>(1);
  int status = -1;
  VCHECK_EQ(cuGetProcAddress_v2("cuNoSuchFunction", &fn, 13000, 0, &status), kSuccess);
  VCHECK(fn == nullptr);
  VCHECK_EQ(status, kSymbolNotFound);
  fn = reinterpret_cast<void*>(1);
  // cuTensorMapEncodeTiled came with 12.0.
  VCHECK_EQ(cuGetProcAddress_v2("cuTensorMapEncodeTiled", &fn, 11040, 0, &status), kSuccess);
  VCHECK(fn == nullptr);
  VCHECK_EQ(status, kVersionNotSufficient);
  VCHECK_EQ(cuGetProcAddress_v2(nullptr, &fn, 13000, 0, &status), 1);   // CUDA_ERROR_INVALID_VALUE
}

// A suffixed name, which the card does not resolve at all, is still taken here as the export it names, and as no
// other: cuStreamGetCaptureInfo_v3 was cuStreamGetCaptureInfo_v2.
VTEST(a_suffixed_name_is_its_own_export) {
  VCHECK_EQ(cuInit(0), kSuccess);
  for (const char* n : {"cuStreamGetCaptureInfo_v3", "cuGraphExecUpdate_v2", "cuMemAlloc_v2", "cuCtxCreate_v4"}) {
    void* fn = nullptr;
    int status = -1;
    VCHECK_EQ(cuGetProcAddress_v2(n, &fn, 13000, 0, &status), kSuccess);
    if (fn != exported(n)) std::printf("  %s is not its own export\n", n);
    VCHECK(fn != nullptr && fn == exported(n));
  }
}

VTEST(every_name_resolves_as_on_an_rtx3060) {
  VCHECK_EQ(cuInit(0), kSuccess);
  std::ifstream in(DRIVER_PROC_ADDRESS_TABLE);
  VCHECK(in.good());
  int rows = 0, wrong = 0;
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    std::string name, symbol;
    int version = 0, card_result = -1, card_status = -1;
    unsigned long long flags = 0;
    if (!(ls >> name >> version >> flags >> card_result >> card_status >> symbol)) continue;
    ++rows;
    const auto platform = kPlatform.find(name);
    if (platform != kPlatform.end()) {
      symbol = platform->second;
      card_status = kFound;
    }
    void* fn = nullptr;
    int status = -1;
    const int r = cuGetProcAddress_v2(name.c_str(), &fn, version, flags, &status);
    void* want = symbol == "-" ? nullptr : exported(symbol.c_str());
    const bool ok = r == card_result && status == card_status && fn == want && (symbol == "-" || want != nullptr);
    if (!ok) {
      ++wrong;
      Dl_info d{};
      const char* got = fn && dladdr(fn, &d) && d.dli_sname ? d.dli_sname : (fn ? "?" : "-");
      std::printf("  %s %d flags %llu: card %d/%d %s, shim %d/%d %s%s\n", name.c_str(), version, flags, card_result,
                  card_status, symbol.c_str(), r, status, got,
                  symbol != "-" && !want ? " (the shim does not export it)" : "");
    }
  }
  std::printf("  %d rows, %d differ\n", rows, wrong);
  VCHECK(rows > 3000);
  VCHECK_EQ(wrong, 0);
}

// The names the CUDA 13.0 driver stub exports that the shim lacked besides cuProfilerInitialize: the per-thread forms
// of the CUDA 11 stream memory-operation and capture ABIs, and the unsuffixed cuMemGetAddressRange.
VTEST(the_driver_stubs_other_names_are_exported) {
  for (const char* n : {"cuStreamBatchMemOp_ptsz", "cuStreamBeginCapture_ptsz", "cuStreamGetCaptureInfo_ptsz",
                        "cuStreamWaitValue32_ptsz", "cuStreamWaitValue64_ptsz", "cuStreamWriteValue32_ptsz",
                        "cuStreamWriteValue64_ptsz", "cuMemGetAddressRange"}) {
    if (!exported(n)) std::printf("  not exported: %s\n", n);
    VCHECK(exported(n) != nullptr);
  }
  VCHECK_EQ(cuInit(0), kSuccess);
  int dev = -1;
  void* ctx = nullptr;
  VCHECK_EQ(cuDeviceGet(&dev, 0), kSuccess);
  VCHECK_EQ(cuDevicePrimaryCtxRetain(&ctx, dev), kSuccess);
  VCHECK_EQ(cuCtxSetCurrent(ctx), kSuccess);
  unsigned long long p = 0, base = 0;
  size_t size = 0;
  VCHECK_EQ(cuMemAlloc_v2(&p, 4096), kSuccess);
  VCHECK_EQ(cuMemGetAddressRange(&base, &size, p + 100), kSuccess);
  VCHECK_EQ(base, p);
  VCHECK_EQ(size, size_t{4096});
  VCHECK_EQ(cuMemFree_v2(p), kSuccess);
  VCHECK_EQ(cuCtxSetCurrent(nullptr), kSuccess);
  VCHECK_EQ(cuDevicePrimaryCtxRelease_v2(dev), kSuccess);
}

VTEST_MAIN
