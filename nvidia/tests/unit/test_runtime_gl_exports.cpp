// The cudaGL* interop functions of libcudart (pantheonsim #528). The real libcudart exports nine; a program that only
// names one of them fails to start under -z now when the shim lacks it. Each must be an exported symbol and, with no
// OpenGL context on a simulated GPU, answer what an RTX 3060 under CUDA 13.0 answers with no GL context:
//   cudaGLRegisterBufferObject, cudaGLUnregisterBufferObject, cudaGLGetDevices   cudaErrorInvalidGraphicsContext (219)
//   cudaGLMapBufferObject(Async) (*devPtr = NULL), cudaGLUnmapBufferObject(Async),
//   cudaGLSetBufferObjectMapFlags                                                cudaErrorInvalidValue (1)
//   cudaGLSetGLDevice                                                            cudaErrorInvalidDevice (101) for a
//       device that does not exist, otherwise cudaErrorDevicesUnavailable (46).
// cuda_gl_interop.h needs the GL headers, so the prototypes are declared here; when the header is at hand
// (VGPU_CUDA_INCLUDE) every cudaGL* name it declares is also looked up, so a name a later toolkit adds shows up.
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cctype>
#include <cstring>
#include <fstream>
#include <set>
#include <string>

#include "vtest.hpp"

extern "C" {
cudaError_t cudaGLGetDevices(unsigned int*, int*, unsigned int, int);
cudaError_t cudaGLSetGLDevice(int);
cudaError_t cudaGLRegisterBufferObject(unsigned int);
cudaError_t cudaGLMapBufferObject(void**, unsigned int);
cudaError_t cudaGLMapBufferObjectAsync(void**, unsigned int, cudaStream_t);
cudaError_t cudaGLUnmapBufferObject(unsigned int);
cudaError_t cudaGLUnmapBufferObjectAsync(unsigned int, cudaStream_t);
cudaError_t cudaGLUnregisterBufferObject(unsigned int);
cudaError_t cudaGLSetBufferObjectMapFlags(unsigned int, unsigned int);
}

VTEST(every_cudaGL_name_is_an_export) {
  std::set<std::string> names = {"cudaGLGetDevices", "cudaGLSetGLDevice", "cudaGLRegisterBufferObject",
                                 "cudaGLMapBufferObject", "cudaGLMapBufferObjectAsync", "cudaGLUnmapBufferObject",
                                 "cudaGLUnmapBufferObjectAsync", "cudaGLUnregisterBufferObject",
                                 "cudaGLSetBufferObjectMapFlags"};
#ifdef VGPU_CUDA_INCLUDE
  std::ifstream in(std::string(VGPU_CUDA_INCLUDE) + "/cuda_gl_interop.h");
  std::string line;
  while (std::getline(in, line)) {
    if (line.compare(0, 6, "extern") != 0) continue;
    const size_t at = line.find("cudaGL");
    if (at == std::string::npos) continue;
    size_t e = at;
    while (e < line.size() && (std::isalnum(static_cast<unsigned char>(line[e])) || line[e] == '_')) ++e;
    if (e < line.size() && line[e] == '(') names.insert(line.substr(at, e - at));
  }
#endif
  VCHECK(names.size() >= 9);
  for (const std::string& n : names) {
    const bool found = dlsym(RTLD_DEFAULT, n.c_str()) != nullptr;
    if (!found) std::fprintf(stderr, "missing export: %s\n", n.c_str());
    VCHECK(found);
  }
}

VTEST(registering_and_listing_say_there_is_no_graphics_context) {
  const cudaError_t k = cudaErrorInvalidGraphicsContext;
  VCHECK_EQ(cudaGLRegisterBufferObject(1), k);
  VCHECK_EQ(cudaGLUnregisterBufferObject(1), k);
  unsigned int count = 77;
  int devices[4] = {9, 9, 9, 9};
  VCHECK_EQ(cudaGLGetDevices(&count, devices, 4, 1), k);
  VCHECK_EQ(count, 77u);   // left alone, as the card leaves it
  VCHECK_EQ(cudaGLGetDevices(nullptr, nullptr, 0, 1), k);
  VCHECK_EQ(cudaGetLastError(), k);   // and the error is the sticky last one, as for any runtime call
}

VTEST(mapping_an_unregistered_buffer_is_an_invalid_value) {
  void* p = reinterpret_cast<void*>(0x1);
  VCHECK_EQ(cudaGLMapBufferObject(&p, 1), cudaErrorInvalidValue);
  VCHECK(p == nullptr);
  p = reinterpret_cast<void*>(0x1);
  VCHECK_EQ(cudaGLMapBufferObjectAsync(&p, 1, nullptr), cudaErrorInvalidValue);
  VCHECK(p == nullptr);
  VCHECK_EQ(cudaGLUnmapBufferObject(1), cudaErrorInvalidValue);
  VCHECK_EQ(cudaGLUnmapBufferObjectAsync(1, nullptr), cudaErrorInvalidValue);
  VCHECK_EQ(cudaGLSetBufferObjectMapFlags(1, 0), cudaErrorInvalidValue);
  VCHECK_EQ(cudaGetLastError(), cudaErrorInvalidValue);
}

VTEST(setting_the_gl_device_checks_the_device) {
  VCHECK_EQ(cudaGLSetGLDevice(99), cudaErrorInvalidDevice);
  VCHECK_EQ(cudaGLSetGLDevice(-1), cudaErrorInvalidDevice);
  VCHECK_EQ(cudaGLSetGLDevice(0), cudaErrorDevicesUnavailable);
  (void)cudaGetLastError();
}

VTEST_MAIN
