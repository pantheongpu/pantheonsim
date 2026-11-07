// Driver calls that need no current context. On a real GPU the context of a stream-ordered call comes from its
// stream (cuMemAllocFromPoolAsync, cuMemcpy3DAsync, cuStreamSynchronize), a free finds it from the pointer
// (cuMemFree of device and managed memory, cuMemFreeHost), and a virtual address range belongs to no context
// (cuMemAddressReserve, cuMemAddressFree). cuda-api-wrappers retains the primary context, pushes it around
// every call and pops it after, so none is current between calls, and these failed with
// CUDA_ERROR_INVALID_CONTEXT. What does need a context still says so: cuMemcpy3D (no stream) and the null
// stream's synchronize fail the same way on the card.
#include <cuda.h>

#include <cstdint>
#include <cstring>

#include "vtest.hpp"

namespace {

struct Env {
  CUcontext ctx = nullptr;
  CUdevice dev = 0;
  Env() {
    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuDevicePrimaryCtxRetain(&ctx, dev);
  }
  ~Env() { cuDevicePrimaryCtxRelease(dev); }
};

CUcontext current_context() {
  CUcontext c = nullptr;
  VCHECK_EQ(cuCtxGetCurrent(&c), CUDA_SUCCESS);
  return c;
}

}  // namespace

VTEST(stream_ordered_calls_take_the_context_from_the_stream) {
  Env e;
  VCHECK_EQ(cuCtxPushCurrent(e.ctx), CUDA_SUCCESS);
  CUstream s = nullptr;
  VCHECK_EQ(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  CUmemPoolProps props;
  std::memset(&props, 0, sizeof props);
  props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
  props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  props.location.id = 0;
  CUmemoryPool pool = nullptr;
  VCHECK_EQ(cuMemPoolCreate(&pool, &props), CUDA_SUCCESS);
  CUDA_ARRAY3D_DESCRIPTOR ad;
  std::memset(&ad, 0, sizeof ad);
  ad.Width = 4; ad.Height = 4; ad.Depth = 4; ad.Format = CU_AD_FORMAT_FLOAT; ad.NumChannels = 1;
  CUarray arr = nullptr;
  VCHECK_EQ(cuArray3DCreate(&arr, &ad), CUDA_SUCCESS);
  CUcontext popped = nullptr;
  VCHECK_EQ(cuCtxPopCurrent(&popped), CUDA_SUCCESS);
  VCHECK(current_context() == nullptr);

  CUdeviceptr p = 0;
  VCHECK_EQ(cuMemAllocFromPoolAsync(&p, 1024, pool, s), CUDA_SUCCESS);
  VCHECK(p != 0);
  VCHECK_EQ(cuMemFreeAsync(p, s), CUDA_SUCCESS);

  static float host[64];
  CUDA_MEMCPY3D c;
  std::memset(&c, 0, sizeof c);
  c.srcMemoryType = CU_MEMORYTYPE_HOST; c.srcHost = host; c.srcPitch = 16; c.srcHeight = 4;
  c.dstMemoryType = CU_MEMORYTYPE_ARRAY; c.dstArray = arr;
  c.WidthInBytes = 16; c.Height = 4; c.Depth = 4;
  VCHECK_EQ(cuMemcpy3DAsync(&c, s), CUDA_SUCCESS);
  VCHECK_EQ(cuStreamSynchronize(s), CUDA_SUCCESS);

  // What has no stream to name a context still needs the current one.
  VCHECK_EQ(cuMemcpy3D(&c), CUDA_ERROR_INVALID_CONTEXT);
  VCHECK_EQ(cuStreamSynchronize(nullptr), CUDA_ERROR_INVALID_CONTEXT);
  VCHECK_EQ(cuMemAllocFromPoolAsync(&p, 1024, pool, nullptr), CUDA_ERROR_INVALID_CONTEXT);
  // Nothing was made current on the way.
  VCHECK(current_context() == nullptr);

  VCHECK_EQ(cuCtxPushCurrent(e.ctx), CUDA_SUCCESS);
  VCHECK_EQ(cuArrayDestroy(arr), CUDA_SUCCESS);
  VCHECK_EQ(cuMemPoolDestroy(pool), CUDA_SUCCESS);
  VCHECK_EQ(cuStreamDestroy(s), CUDA_SUCCESS);
  VCHECK_EQ(cuCtxPopCurrent(&popped), CUDA_SUCCESS);
}

VTEST(address_ranges_and_frees_need_no_current_context) {
  Env e;
  VCHECK_EQ(cuCtxPushCurrent(e.ctx), CUDA_SUCCESS);
  CUdeviceptr dptr = 0, managed = 0;
  void* pinned = nullptr;
  VCHECK_EQ(cuMemAlloc(&dptr, 4096), CUDA_SUCCESS);
  VCHECK_EQ(cuMemAllocManaged(&managed, 4096, CU_MEM_ATTACH_GLOBAL), CUDA_SUCCESS);
  VCHECK_EQ(cuMemHostAlloc(&pinned, 4096, CU_MEMHOSTALLOC_DEVICEMAP), CUDA_SUCCESS);
  CUcontext popped = nullptr;
  VCHECK_EQ(cuCtxPopCurrent(&popped), CUDA_SUCCESS);
  VCHECK(current_context() == nullptr);

  CUdeviceptr base = 0;
  VCHECK_EQ(cuMemAddressReserve(&base, 2 << 20, 0, 0, 0), CUDA_SUCCESS);
  VCHECK(base != 0);
  VCHECK_EQ(cuMemAddressFree(base, 2 << 20), CUDA_SUCCESS);

  VCHECK_EQ(cuMemFree(managed), CUDA_SUCCESS);
  VCHECK_EQ(cuMemFree(dptr), CUDA_SUCCESS);
  VCHECK_EQ(cuMemFreeHost(pinned), CUDA_SUCCESS);
  VCHECK(current_context() == nullptr);

  // The freed device pointer is gone, which the pointer query says with a context current.
  VCHECK_EQ(cuCtxPushCurrent(e.ctx), CUDA_SUCCESS);
  unsigned int type = 0;
  VCHECK_EQ(cuPointerGetAttribute(&type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dptr), CUDA_ERROR_INVALID_VALUE);
  // Allocating still needs the context, with or without the calls above having run.
  VCHECK_EQ(cuCtxPopCurrent(&popped), CUDA_SUCCESS);
  CUdeviceptr q = 0;
  VCHECK_EQ(cuMemAlloc(&q, 16), CUDA_ERROR_INVALID_CONTEXT);
}

VTEST_MAIN
