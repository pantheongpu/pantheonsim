// The graphics-interop entry points of libcuda that need a display stack the simulated GPU does not have: OpenGL
// (cudaGL.h), EGL (cudaEGL.h) and VDPAU (cudaVDPAU.h). Included in driver_api.cpp at file scope, before the
// cuGetProcAddress table that lists them.
//
// A loader looks these up when it loads libcuda and treats a missing name as no CUDA at all (Dr.Jit fetches
// cuGraphicsGLRegisterBuffer and cuGraphicsGLRegisterImage and turns its CUDA backend off without them; pantheonsim
// #470). So each name resolves, by dlsym and by cuGetProcAddress, and answers CUDA_ERROR_NOT_SUPPORTED: there is
// no OpenGL, EGL or VDPAU on a simulated device. cuGLGetDevices says it found no device. The Windows-only
// cuWGLGetDevice is not exported on Linux and is left out here too.
//
// The parameters are declared as cudaGL.h / cudaEGL.h / cudaVDPAU.h declare them where they are plain types and as
// pointers or integers where the header's type comes from a graphics library this one does not include; none is
// read, so the signature only has to match in size.

namespace {
CUresult gl_refuse(const char* api, const char* what) {
  static std::mutex mu;
  static std::set<std::string> said;
  std::lock_guard<std::mutex> lock(mu);
  if (!quiet() && said.insert(api).second)
    std::fprintf(stderr, "[vgpu] %s is not implemented by VirtualGPU: there is no %s on a simulated GPU; returning %d\n", api,
                 what, static_cast<int>(CUDA_ERROR_NOT_SUPPORTED));
  return CUDA_ERROR_NOT_SUPPORTED;
}
}  // namespace

#define VGPU_GL_STUB(name, what, params) \
  VGPU_EXPORT CUresult name params { return gl_refuse(#name, what); }

// ---- OpenGL (cudaGL.h) ----
VGPU_GL_STUB(cuGraphicsGLRegisterBuffer, "OpenGL", (CUgraphicsResource*, unsigned int, unsigned int))
VGPU_GL_STUB(cuGraphicsGLRegisterImage, "OpenGL", (CUgraphicsResource*, unsigned int, unsigned int, unsigned int))
VGPU_EXPORT CUresult cuGLGetDevices_v2(unsigned int* count, CUdevice*, unsigned int, int) {
  if (count) *count = 0;
  return gl_refuse("cuGLGetDevices", "OpenGL");
}
VGPU_EXPORT CUresult cuGLGetDevices(unsigned int* count, CUdevice* devices, unsigned int n, int list) {
  return cuGLGetDevices_v2(count, devices, n, list);
}
VGPU_GL_STUB(cuGLCtxCreate_v2, "OpenGL", (CUcontext*, unsigned int, CUdevice))
VGPU_GL_STUB(cuGLCtxCreate, "OpenGL", (CUcontext*, unsigned int, CUdevice))
VGPU_GL_STUB(cuGLInit, "OpenGL", (void))
VGPU_GL_STUB(cuGLRegisterBufferObject, "OpenGL", (unsigned int))
VGPU_GL_STUB(cuGLMapBufferObject_v2, "OpenGL", (CUdeviceptr*, size_t*, unsigned int))
VGPU_GL_STUB(cuGLMapBufferObject, "OpenGL", (void*, void*, unsigned int))
VGPU_GL_STUB(cuGLUnmapBufferObject, "OpenGL", (unsigned int))
VGPU_GL_STUB(cuGLUnregisterBufferObject, "OpenGL", (unsigned int))
VGPU_GL_STUB(cuGLSetBufferObjectMapFlags, "OpenGL", (unsigned int, unsigned int))
VGPU_GL_STUB(cuGLMapBufferObjectAsync_v2, "OpenGL", (CUdeviceptr*, size_t*, unsigned int, CUstream))
VGPU_GL_STUB(cuGLMapBufferObjectAsync, "OpenGL", (void*, void*, unsigned int, CUstream))
VGPU_GL_STUB(cuGLUnmapBufferObjectAsync, "OpenGL", (unsigned int, CUstream))

// ---- EGL (cudaEGL.h) ----
VGPU_GL_STUB(cuGraphicsEGLRegisterImage, "EGL", (CUgraphicsResource*, void*, unsigned int))
VGPU_GL_STUB(cuEGLStreamConsumerConnect, "EGL", (void*, void*))
VGPU_GL_STUB(cuEGLStreamConsumerConnectWithFlags, "EGL", (void*, void*, unsigned int))
VGPU_GL_STUB(cuEGLStreamConsumerDisconnect, "EGL", (void*))
VGPU_GL_STUB(cuEGLStreamConsumerAcquireFrame, "EGL", (void*, CUgraphicsResource*, CUstream*, unsigned int))
VGPU_GL_STUB(cuEGLStreamConsumerReleaseFrame, "EGL", (void*, CUgraphicsResource, CUstream*))
VGPU_GL_STUB(cuEGLStreamProducerConnect, "EGL", (void*, void*, int, int))
VGPU_GL_STUB(cuEGLStreamProducerDisconnect, "EGL", (void*))
VGPU_GL_STUB(cuEGLStreamProducerPresentFrame, "EGL", (void*, void*, CUstream*))
VGPU_GL_STUB(cuEGLStreamProducerReturnFrame, "EGL", (void*, void*, CUstream*))
VGPU_GL_STUB(cuGraphicsResourceGetMappedEglFrame, "EGL", (void*, CUgraphicsResource, unsigned int, unsigned int))
VGPU_GL_STUB(cuEventCreateFromEGLSync, "EGL", (CUevent*, void*, unsigned int))

// ---- VDPAU (cudaVDPAU.h) ----
VGPU_GL_STUB(cuVDPAUGetDevice, "VDPAU", (CUdevice*, unsigned int, void*))
VGPU_GL_STUB(cuVDPAUCtxCreate_v2, "VDPAU", (CUcontext*, unsigned int, CUdevice, unsigned int, void*))
VGPU_GL_STUB(cuVDPAUCtxCreate, "VDPAU", (CUcontext*, unsigned int, CUdevice, unsigned int, void*))
VGPU_GL_STUB(cuGraphicsVDPAURegisterVideoSurface, "VDPAU", (CUgraphicsResource*, unsigned int, unsigned int))
VGPU_GL_STUB(cuGraphicsVDPAURegisterOutputSurface, "VDPAU", (CUgraphicsResource*, unsigned int, unsigned int))
#undef VGPU_GL_STUB
