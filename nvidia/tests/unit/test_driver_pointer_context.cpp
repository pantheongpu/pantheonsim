// cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_CONTEXT) is the context that owns the allocation, not the one
// that is current when it is asked. With two contexts on a device and a third current, the simulator named the
// third for both pointers (cuda-api-wrappers' unified_addressing example checks for the owner).
#include <cuda.h>

#include "vtest.hpp"

namespace {
CUcontext context_of(CUdeviceptr p, CUresult want = CUDA_SUCCESS) {
  CUcontext c = reinterpret_cast<CUcontext>(0x1);
  VCHECK_EQ(cuPointerGetAttribute(&c, CU_POINTER_ATTRIBUTE_CONTEXT, p), want);
  return c;
}
}  // namespace

VTEST(a_device_pointer_names_the_context_it_was_allocated_in) {
  cuInit(0);
  CUdevice dev = 0;
  cuDeviceGet(&dev, 0);
  CUcontext c1 = nullptr, c2 = nullptr, primary = nullptr;
  VCHECK_EQ(cuCtxCreate(&c1, nullptr, 0, dev), CUDA_SUCCESS);
  CUdeviceptr p1 = 0, m1 = 0;
  VCHECK_EQ(cuMemAlloc(&p1, 123), CUDA_SUCCESS);
  VCHECK_EQ(cuMemAllocManaged(&m1, 4096, CU_MEM_ATTACH_GLOBAL), CUDA_SUCCESS);
  VCHECK_EQ(cuCtxCreate(&c2, nullptr, 0, dev), CUDA_SUCCESS);
  CUdeviceptr p2 = 0, pitched = 0;
  size_t pitch = 0;
  VCHECK_EQ(cuMemAlloc(&p2, 123), CUDA_SUCCESS);
  VCHECK_EQ(cuMemAllocPitch(&pitched, &pitch, 100, 4, 4), CUDA_SUCCESS);
  VCHECK_EQ(cuDevicePrimaryCtxRetain(&primary, dev), CUDA_SUCCESS);
  VCHECK_EQ(cuCtxPushCurrent(primary), CUDA_SUCCESS);   // ask from a third context

  VCHECK(context_of(p1) == c1);
  VCHECK(context_of(p1 + 100) == c1);   // anywhere in the allocation
  VCHECK(context_of(m1) == c1);
  VCHECK(context_of(p2) == c2);
  VCHECK(context_of(pitched) == c2);
  // The batch query agrees.
  CUpointer_attribute attr = CU_POINTER_ATTRIBUTE_CONTEXT;
  CUcontext got = nullptr;
  void* data = &got;
  VCHECK_EQ(cuPointerGetAttributes(1, &attr, &data, p1), CUDA_SUCCESS);
  VCHECK(got == c1);

  // After a free and a new allocation from the other context the answer follows the new owner.
  VCHECK_EQ(cuCtxSetCurrent(c2), CUDA_SUCCESS);
  VCHECK_EQ(cuMemFree(p1), CUDA_SUCCESS);
  CUdeviceptr p3 = 0;
  VCHECK_EQ(cuMemAlloc(&p3, 123), CUDA_SUCCESS);
  VCHECK(context_of(p3) == c2);
  VCHECK_EQ(cuMemFree(p3), CUDA_SUCCESS);
  VCHECK_EQ(cuMemFree(m1), CUDA_SUCCESS);
  VCHECK_EQ(cuMemFree(p2), CUDA_SUCCESS);
  VCHECK_EQ(cuMemFree(pitched), CUDA_SUCCESS);

  VCHECK_EQ(cuCtxSetCurrent(nullptr), CUDA_SUCCESS);
  VCHECK_EQ(cuCtxDestroy(c1), CUDA_SUCCESS);
  VCHECK_EQ(cuCtxDestroy(c2), CUDA_SUCCESS);
  VCHECK_EQ(cuDevicePrimaryCtxRelease(dev), CUDA_SUCCESS);
}

VTEST_MAIN
