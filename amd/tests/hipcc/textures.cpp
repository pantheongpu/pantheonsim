// The texture API on a GPU with no texture units, as the MI300 family is:
// hipcc refuses textures in its device code (__HIP_NO_IMAGE_SUPPORT), and
// the host API says so. Each line is what the runtime answers; the same
// program on ROCm's own HIP (amd/tests/e2e/run_hip_on_hsa.sh) must print the
// same lines, which is what says these answers are HIP's.
#include <hip/hip_runtime.h>

#include <cstdio>

#define SAY(x) std::printf("%s: %s\n", #x, hipGetErrorName(x))

int main() {
  int images = -1;
  (void)hipDeviceGetAttribute(&images, hipDeviceAttributeImageSupport, 0);
  hipDeviceProp_t p;
  (void)hipGetDeviceProperties(&p, 0);
  std::printf("image support %d, largest 1D texture %d, 2D %dx%d, 3D %dx%dx%d, texture alignment %zu\n", images,
              p.maxTexture1D, p.maxTexture2D[0], p.maxTexture2D[1], p.maxTexture3D[0], p.maxTexture3D[1],
              p.maxTexture3D[2], p.textureAlignment);

  const hipChannelFormatDesc f = hipCreateChannelDesc<float>();
  const hipChannelFormatDesc u8x2 = hipCreateChannelDesc(8, 8, 0, 0, hipChannelFormatKindUnsigned);
  std::printf("channel descriptions: float %d/%d/%d/%d kind %d, uchar2 %d/%d/%d/%d kind %d\n", f.x, f.y, f.z, f.w,
              static_cast<int>(f.f), u8x2.x, u8x2.y, u8x2.z, u8x2.w, static_cast<int>(u8x2.f));
#if HIP_VERSION_MAJOR >= 7   // ROCm 6 declared it with the wrong parameters
  size_t width = 7;
  SAY(hipDeviceGetTexture1DLinearMaxWidth(&width, &f, 0));
  std::printf("widest 1D linear texture %zu\n", width);
#endif

  hipArray_t array = nullptr;
  SAY(hipMallocArray(&array, &f, 64, 64, 0));
  hipArray_t cube = nullptr;
  SAY(hipMalloc3DArray(&cube, &f, make_hipExtent(8, 8, 8), 0));
  hipMipmappedArray_t mips = nullptr;
  SAY(hipMallocMipmappedArray(&mips, &f, make_hipExtent(8, 8, 8), 3, 0));
  HIP_ARRAY_DESCRIPTOR ad = {};
  ad.Width = ad.Height = 16;
  ad.Format = HIP_AD_FORMAT_FLOAT;
  ad.NumChannels = 1;
  SAY(hipArrayCreate(&array, &ad));

  float* linear = nullptr;
  (void)hipMalloc(&linear, 4096);
  hipResourceDesc r = {};
  r.resType = hipResourceTypeLinear;
  r.res.linear.devPtr = linear;
  r.res.linear.desc = f;
  r.res.linear.sizeInBytes = 4096;
  hipTextureDesc t = {};
  t.filterMode = hipFilterModePoint;
  t.readMode = hipReadModeElementType;
  hipTextureObject_t tex = 0;
  SAY(hipCreateTextureObject(&tex, &r, &t, nullptr));
  hipSurfaceObject_t surf = 0;
  SAY(hipCreateSurfaceObject(&surf, &r));

  hipChannelFormatDesc got = {};
  SAY(hipGetChannelDesc(&got, array));
  SAY(hipMemcpy2DToArray(array, 0, 0, linear, 64, 64, 4, hipMemcpyDeviceToDevice));
  SAY(hipFreeArray(array));
  SAY(hipDestroyTextureObject(0));
  SAY(hipDestroySurfaceObject(0));
  // The failures above are kept for hipGetLastError, as any other is.
  SAY(hipGetLastError());
  SAY(hipGetLastError());
  (void)hipFree(linear);
  return 0;
}
