// CUDA arrays through the driver API, as bindings that load libcuda alone use
// them (rust-cuda's cust): which shapes cuArray3DCreate takes and refuses,
// the descriptors read back, copies into and out of 1D arrays, and 2D and 3D
// copies between host, device and array memory, sub-rectangles and bounds
// included. Also cuMemAllocAsync and cuMemFreeAsync. The expected answers are
// what an RTX 3060's and an RTX 3080 Ti's drivers give (they agree).
#include <cuda.h>

#include <cstdio>
#include <cstring>

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

static CUresult make(CUarray* a, size_t w, size_t h, size_t d, CUarray_format f, unsigned ch, unsigned flags) {
  CUDA_ARRAY3D_DESCRIPTOR desc{};
  desc.Width = w; desc.Height = h; desc.Depth = d; desc.Format = f; desc.NumChannels = ch; desc.Flags = flags;
  return cuArray3DCreate(a, &desc);
}

// Created, read back with both descriptor calls, destroyed.
static void shape(size_t w, size_t h, size_t d, unsigned ch, unsigned flags, const char* what) {
  CUarray a = nullptr;
  const bool made = make(&a, w, h, d, CU_AD_FORMAT_FLOAT, ch, flags) == CUDA_SUCCESS;
  CUDA_ARRAY3D_DESCRIPTOR got{};
  CUDA_ARRAY_DESCRIPTOR got2{};
  const bool ok = made && cuArray3DGetDescriptor(&got, a) == CUDA_SUCCESS &&
                  cuArrayGetDescriptor(&got2, a) == CUDA_SUCCESS && got.Width == w && got.Height == h &&
                  got.Depth == d && got.Format == CU_AD_FORMAT_FLOAT && got.NumChannels == ch &&
                  got.Flags == flags && got2.Width == w && got2.Height == h && got2.NumChannels == ch;
  check(ok && cuArrayDestroy(a) == CUDA_SUCCESS, what);
}

static void refused(size_t w, size_t h, size_t d, CUarray_format f, unsigned ch, unsigned flags, const char* what) {
  CUarray a = nullptr;
  check(make(&a, w, h, d, f, ch, flags) == CUDA_ERROR_INVALID_VALUE, what);
}

int main() {
  CUdevice dev;
  CUcontext ctx;
  if (cuInit(0) || cuDeviceGet(&dev, 0) || cuDevicePrimaryCtxRetain(&ctx, dev) || cuCtxSetCurrent(ctx)) {
    std::printf("FAIL: no device\n");
    return 1;
  }
  const CUarray_format F = CU_AD_FORMAT_FLOAT;
  shape(1, 2, 3, 2, 0, "a 1x2x3 array of float2");
  shape(10, 0, 0, 1, 0, "a 1D array");
  shape(10, 20, 0, 1, 0, "a 2D array");
  shape(10, 0, 20, 1, CUDA_ARRAY3D_LAYERED, "a layered 1D array");
  shape(4, 4, 6, 1, CUDA_ARRAY3D_CUBEMAP, "a cubemap");
  shape(4, 4, 24, 1, CUDA_ARRAY3D_CUBEMAP | CUDA_ARRAY3D_LAYERED, "a layered cubemap");
  shape(4, 4, 6, 1, CUDA_ARRAY3D_SURFACE_LDST, "a surface array");
  shape(1, 2, 3, 4, 0, "four channels");
  refused(0, 0, 0, F, 1, 0, "zero width refused");
  refused(0, 10, 20, F, 1, 0, "zero width with a height refused");
  refused(10, 0, 20, F, 1, 0, "a depth without a height refused");
  refused(2, 3, 6, F, 1, CUDA_ARRAY3D_CUBEMAP, "a cubemap that is not square refused");
  refused(4, 4, 5, F, 1, CUDA_ARRAY3D_CUBEMAP, "a cubemap of depth 5 refused");
  refused(4, 4, 10, F, 1, CUDA_ARRAY3D_CUBEMAP | CUDA_ARRAY3D_LAYERED, "layered cubemaps not a multiple of 6 refused");
  refused(4, 4, 0, F, 1, CUDA_ARRAY3D_LAYERED, "a layered array of no layers refused");
  refused(1, 2, 3, F, 3, 0, "three channels refused");
  refused(1, 2, 3, F, 0, 0, "no channels refused");
  refused(1, 2, 3, (CUarray_format)0x55, 1, 0, "an unknown format refused");
  refused(4, 4, 6, F, 1, 0x80000, "an unknown flag refused");

  float h[32], back[32];
  for (int i = 0; i < 32; ++i) h[i] = i * 1.5f;

  // A 1D array: host and device copies at a byte offset, and the bounds.
  CUarray one;
  IS(make(&one, 32, 0, 0, F, 1, 0), CUDA_SUCCESS);
  IS(cuMemcpyHtoA(one, 0, h, sizeof h), CUDA_SUCCESS);
  for (float& v : back) v = -1;
  IS(cuMemcpyAtoH(back, one, 16, 64), CUDA_SUCCESS);
  IS(cuCtxSynchronize(), CUDA_SUCCESS);
  check(back[0] == h[4] && back[15] == h[19] && back[16] == -1, "a 1D array's bytes from an offset");
  IS(cuMemcpyHtoA(one, 64, h, sizeof h), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemcpyAtoH(back, one, 120, 16), CUDA_ERROR_INVALID_VALUE);
  CUdeviceptr d;
  IS(cuMemAlloc(&d, 128), CUDA_SUCCESS);
  IS(cuMemsetD8(d, 0, 128), CUDA_SUCCESS);
  IS(cuMemcpyAtoD(d, one, 8, 32), CUDA_SUCCESS);
  IS(cuMemcpyDtoA(one, 64, d, 32), CUDA_SUCCESS);
  for (float& v : back) v = -1;
  IS(cuMemcpyAtoH(back, one, 0, sizeof back), CUDA_SUCCESS);
  IS(cuCtxSynchronize(), CUDA_SUCCESS);
  check(back[16] == h[2] && back[23] == h[9] && back[24] == h[24], "device to and from a 1D array");

  // A 2D array: 8 floats by 4 rows, filled from the host with cuMemcpy2D.
  CUarray two;
  IS(make(&two, 8, 4, 0, F, 1, 0), CUDA_SUCCESS);
  // The linear copies reach an array's first row, whatever its shape.
  IS(cuMemcpyHtoA(two, 16, h, 16), CUDA_SUCCESS);
  IS(cuMemcpyHtoA(two, 0, h, 64), CUDA_ERROR_INVALID_VALUE);
  CUDA_MEMCPY2D w{};
  w.srcMemoryType = CU_MEMORYTYPE_HOST; w.srcHost = h; w.srcPitch = 32;
  w.dstMemoryType = CU_MEMORYTYPE_ARRAY; w.dstArray = two; w.WidthInBytes = 32; w.Height = 4;
  IS(cuMemcpy2D(&w), CUDA_SUCCESS);
  // The 3-by-2 sub-rectangle at column 1, row 1, to a host buffer 12 bytes a row.
  for (float& v : back) v = -1;
  CUDA_MEMCPY2D c{};
  c.srcMemoryType = CU_MEMORYTYPE_ARRAY; c.srcArray = two; c.srcXInBytes = 4; c.srcY = 1;
  c.dstMemoryType = CU_MEMORYTYPE_HOST; c.dstHost = back; c.dstPitch = 12; c.WidthInBytes = 12; c.Height = 2;
  IS(cuMemcpy2D(&c), CUDA_SUCCESS);
  IS(cuCtxSynchronize(), CUDA_SUCCESS);
  check(back[0] == h[9] && back[2] == h[11] && back[3] == h[17] && back[5] == h[19] && back[6] == -1,
        "a sub-rectangle of a 2D array");
  c.WidthInBytes = 32;   // column 1 plus a whole row runs past the array's edge
  IS(cuMemcpy2D(&c), CUDA_ERROR_INVALID_VALUE);
  // The whole array through device memory, with cuMemcpy3D.
  CUDA_MEMCPY3D t{};
  t.srcMemoryType = CU_MEMORYTYPE_ARRAY; t.srcArray = two;
  t.dstMemoryType = CU_MEMORYTYPE_DEVICE; t.dstDevice = d; t.dstPitch = 32; t.dstHeight = 4;
  t.WidthInBytes = 32; t.Height = 4; t.Depth = 1;
  IS(cuMemcpy3D(&t), CUDA_SUCCESS);
  for (float& v : back) v = -1;
  IS(cuMemcpyDtoH(back, d, 128), CUDA_SUCCESS);
  check(back[0] == h[0] && back[9] == h[9] && back[31] == h[31], "a 2D array to device memory with cuMemcpy3D");
  // Device to device, pitched: rows of 12 bytes out of rows of 32.
  CUdeviceptr e;
  IS(cuMemAlloc(&e, 64), CUDA_SUCCESS);
  CUDA_MEMCPY2D p{};
  p.srcMemoryType = CU_MEMORYTYPE_DEVICE; p.srcDevice = d; p.srcPitch = 32; p.srcXInBytes = 8;
  p.dstMemoryType = CU_MEMORYTYPE_DEVICE; p.dstDevice = e; p.dstPitch = 12; p.WidthInBytes = 12; p.Height = 3;
  IS(cuMemcpy2D(&p), CUDA_SUCCESS);
  for (float& v : back) v = -1;
  IS(cuMemcpyDtoH(back, e, 36), CUDA_SUCCESS);
  check(back[0] == h[2] && back[3] == h[10] && back[8] == h[20], "a pitched device-to-device copy");

  IS(cuArrayDestroy(one), CUDA_SUCCESS);
  IS(cuArrayDestroy(two), CUDA_SUCCESS);
  IS(cuMemFree(d), CUDA_SUCCESS);
  IS(cuMemFree(e), CUDA_SUCCESS);

  // Stream-ordered allocation.
  CUstream s;
  IS(cuStreamCreate(&s, 0), CUDA_SUCCESS);
  CUdeviceptr q = 0;
  IS(cuMemAllocAsync(&q, 1024, s), CUDA_SUCCESS);
  IS(cuMemsetD8Async(q, 7, 1024, s), CUDA_SUCCESS);
  unsigned char byte = 0;
  IS(cuMemcpyDtoHAsync(&byte, q + 1023, 1, s), CUDA_SUCCESS);
  IS(cuMemFreeAsync(q, s), CUDA_SUCCESS);
  IS(cuStreamSynchronize(s), CUDA_SUCCESS);
  check(byte == 7, "cuMemAllocAsync memory is usable on its stream");
  IS(cuStreamDestroy(s), CUDA_SUCCESS);

  std::printf(failures ? "FAIL: %d array checks\n" : "PASS: every array check\n", failures);
  return failures ? 1 : 0;
}
