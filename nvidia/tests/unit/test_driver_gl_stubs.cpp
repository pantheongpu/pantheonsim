// The OpenGL, EGL and VDPAU interop entry points of libcuda (pantheonsim #470). A simulated GPU has no display
// stack, but a loader that looks the names up when it loads libcuda (Dr.Jit asks for cuGraphicsGLRegisterBuffer and
// cuGraphicsGLRegisterImage and turns its CUDA backend off without them) must find them. Each must
//   - be an exported symbol (dlsym) and be found by cuGetProcAddress, at the version a loader asks for;
//   - answer CUDA_ERROR_NOT_SUPPORTED when called, and cuGLGetDevices report no device.
// When the toolkit's cudaGL.h, cudaEGL.h and cudaVDPAU.h are at hand (VGPU_CUDA_INCLUDE), every function they
// declare is checked, so a name a later toolkit adds shows up here as a missing one.
#include <cuda.h>
#include <dlfcn.h>

#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "vtest.hpp"

namespace {

// What a loader that must not fail looks for, by name; the rest come from the headers.
const char* const kNames[] = {
    "cuGraphicsGLRegisterBuffer", "cuGraphicsGLRegisterImage", "cuGLGetDevices", "cuGLGetDevices_v2", "cuGLCtxCreate",
    "cuGLCtxCreate_v2", "cuGLInit", "cuGLRegisterBufferObject", "cuGLMapBufferObject", "cuGLMapBufferObject_v2",
    "cuGLMapBufferObject_v2_ptds", "cuGLUnmapBufferObject", "cuGLUnregisterBufferObject", "cuGLSetBufferObjectMapFlags",
    "cuGLMapBufferObjectAsync", "cuGLMapBufferObjectAsync_v2", "cuGLMapBufferObjectAsync_v2_ptsz",
    "cuGLUnmapBufferObjectAsync", "cuGraphicsEGLRegisterImage", "cuEGLStreamConsumerConnect",
    "cuEGLStreamConsumerConnectWithFlags", "cuEGLStreamConsumerDisconnect", "cuEGLStreamConsumerAcquireFrame",
    "cuEGLStreamConsumerReleaseFrame", "cuEGLStreamProducerConnect", "cuEGLStreamProducerDisconnect",
    "cuEGLStreamProducerPresentFrame", "cuEGLStreamProducerReturnFrame", "cuGraphicsResourceGetMappedEglFrame",
    "cuEventCreateFromEGLSync", "cuVDPAUGetDevice", "cuVDPAUCtxCreate", "cuVDPAUCtxCreate_v2",
    "cuGraphicsVDPAURegisterVideoSurface", "cuGraphicsVDPAURegisterOutputSurface"};

using Fn = CUresult (*)(void*, void*, void*, void*, void*);

std::set<std::string> header_names() {
  std::set<std::string> names;
#ifdef VGPU_CUDA_INCLUDE
  for (const char* h : {"cudaGL.h", "cudaEGL.h", "cudaVDPAU.h"}) {
    std::ifstream in(std::string(VGPU_CUDA_INCLUDE) + "/" + h);
    std::string line;
    while (std::getline(in, line)) {
      const size_t at = line.find("CUresult CUDAAPI ");
      if (at == std::string::npos || line.compare(0, 2, " *") == 0) continue;
      size_t b = at + std::strlen("CUresult CUDAAPI "), e = b;
      while (e < line.size() && (std::isalnum(static_cast<unsigned char>(line[e])) || line[e] == '_')) ++e;
      const std::string n = line.substr(b, e - b);
      if (n.rfind("cu", 0) == 0 && n != "cuWGLGetDevice") names.insert(n);   // cuWGLGetDevice is Windows'
    }
  }
#endif
  return names;
}

}  // namespace

VTEST(every_interop_name_is_an_export_of_libcuda) {
  std::set<std::string> all(std::begin(kNames), std::end(kNames));
  for (const auto& n : header_names()) all.insert(n);
  for (const auto& n : all) {
    void* sym = dlsym(RTLD_DEFAULT, n.c_str());
    if (!sym) std::printf("  not exported: %s\n", n.c_str());
    VCHECK(sym != nullptr);
  }
}

VTEST(cuGetProcAddress_finds_every_interop_name) {
  std::set<std::string> all(std::begin(kNames), std::end(kNames));
  for (const auto& n : header_names()) all.insert(n);
  for (const auto& n : all) {
    if (n.size() > 5 && (n.compare(n.size() - 5, 5, "_ptds") == 0 || n.compare(n.size() - 5, 5, "_ptsz") == 0)) continue;
    void* fn = nullptr;
    CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT;
    const CUresult r = cuGetProcAddress(n.c_str(), &fn, 12000, CU_GET_PROC_ADDRESS_DEFAULT, &status);
    if (r != CUDA_SUCCESS || !fn) std::printf("  cuGetProcAddress cannot find: %s\n", n.c_str());
    VCHECK_EQ(r, CUDA_SUCCESS);
    VCHECK(fn != nullptr);
  }
}

VTEST(the_two_names_dr_jit_asks_for_resolve_to_the_exports) {
  for (const char* n : {"cuGraphicsGLRegisterBuffer", "cuGraphicsGLRegisterImage"}) {
    void* fn = nullptr;
    VCHECK_EQ(cuGetProcAddress(n, &fn, 12000, 0, nullptr), CUDA_SUCCESS);
    VCHECK(fn == dlsym(RTLD_DEFAULT, n));
  }
}

VTEST(calling_one_answers_not_supported) {
  CUgraphicsResource res = nullptr;
  auto reg_buffer = reinterpret_cast<CUresult (*)(CUgraphicsResource*, unsigned, unsigned)>(dlsym(RTLD_DEFAULT, "cuGraphicsGLRegisterBuffer"));
  auto reg_image = reinterpret_cast<CUresult (*)(CUgraphicsResource*, unsigned, unsigned, unsigned)>(dlsym(RTLD_DEFAULT, "cuGraphicsGLRegisterImage"));
  VCHECK(reg_buffer != nullptr && reg_image != nullptr);
  if (!reg_buffer || !reg_image) return;
  VCHECK_EQ(reg_buffer(&res, 1, 0), CUDA_ERROR_NOT_SUPPORTED);
  VCHECK_EQ(reg_image(&res, 1, 0x0DE1, 0), CUDA_ERROR_NOT_SUPPORTED);
  VCHECK(res == nullptr);
  // The rest, through the name a loader holds them by.
  std::set<std::string> all(std::begin(kNames), std::end(kNames));
  for (const auto& n : header_names()) all.insert(n);
  for (const auto& n : all) {
    if (n.rfind("cuGLGetDevices", 0) == 0) continue;
    Fn fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, n.c_str()));
    VCHECK(fn != nullptr);
    if (!fn) continue;
    const CUresult r = fn(nullptr, nullptr, nullptr, nullptr, nullptr);
    if (r != CUDA_ERROR_NOT_SUPPORTED) std::printf("  %s answered %d\n", n.c_str(), static_cast<int>(r));
    VCHECK_EQ(r, CUDA_ERROR_NOT_SUPPORTED);
  }
}

VTEST(cuGLGetDevices_finds_no_device) {
  for (const char* n : {"cuGLGetDevices", "cuGLGetDevices_v2"}) {
    auto fn = reinterpret_cast<CUresult (*)(unsigned*, CUdevice*, unsigned, int)>(dlsym(RTLD_DEFAULT, n));
    VCHECK(fn != nullptr);
    if (!fn) continue;
    unsigned count = 99;
    CUdevice dev[4];
    VCHECK_EQ(fn(&count, dev, 4, 1), CUDA_ERROR_NOT_SUPPORTED);
    VCHECK_EQ(count, 0u);
  }
}

VTEST_MAIN
