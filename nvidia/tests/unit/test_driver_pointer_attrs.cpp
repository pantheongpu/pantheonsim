// cuPointerGetAttribute, cuPointerGetAttributes and cuPointerSetAttribute.
// CUDA-aware MPI and UCX ask cuPointerGetAttributes -- memory type, device
// ordinal, buffer id -- of every buffer they are handed, on the strength of the
// answer for a pointer CUDA has never heard of being "none of the above" rather
// than an error. The single query answered only the memory type, the context
// (as NULL) and the device pointer, and the batch and the setter were absent.
//   - what each attribute reports for device, managed, pinned and registered memory;
//   - a pointer that is none of those: INVALID_VALUE alone, NULL defaults in a batch;
//   - SYNC_MEMOPS, the one attribute that can be set.
#include <cuda.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "vtest.hpp"

// The driver shim loads the runtime shim on demand, and both carry src/core/regs.cpp: AddressSanitizer's default
// ODR check aborts a run of this program under it. Say so here, as test_runtime_user_objects does (the
// ASAN_OPTIONS variable still wins when set). Leak detection stays on.
extern "C" const char* __asan_default_options() { return "detect_leaks=1:detect_odr_violation=0"; }

namespace {

struct Env {
  CUcontext ctx[2] = {nullptr, nullptr};
  Env() {
    cuInit(0);
    for (int d = 0; d < 2; ++d) {
      CUdevice dev = 0;
      cuDeviceGet(&dev, d);
      cuDevicePrimaryCtxRetain(&ctx[d], dev);
    }
    cuCtxSetCurrent(ctx[0]);
  }
  ~Env() {
    for (int d = 0; d < 2; ++d) {
      CUdevice dev = 0;
      cuDeviceGet(&dev, d);
      cuDevicePrimaryCtxRelease(dev);
    }
  }
};

template <class T>
T get(CUpointer_attribute a, CUdeviceptr p, CUresult want = CUDA_SUCCESS) {
  T v{};
  std::memset(&v, 0xAA, sizeof v);
  VCHECK_EQ(cuPointerGetAttribute(&v, a, p), want);
  return v;
}

}  // namespace

VTEST(device_memory_is_described_by_the_allocation_that_holds_it) {
  Env e;
  CUdeviceptr base = 0;
  VCHECK_EQ(cuMemAlloc(&base, 4096), CUDA_SUCCESS);
  const CUdeviceptr inside = base + 1000;
  VCHECK_EQ(get<unsigned int>(CU_POINTER_ATTRIBUTE_MEMORY_TYPE, inside), static_cast<unsigned>(CU_MEMORYTYPE_DEVICE));
  VCHECK_EQ(get<CUdeviceptr>(CU_POINTER_ATTRIBUTE_DEVICE_POINTER, inside), inside);
  VCHECK_EQ(get<int>(CU_POINTER_ATTRIBUTE_IS_MANAGED, inside), 0);
  VCHECK_EQ(get<int>(CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, inside), 0);
  VCHECK_EQ(get<int>(CU_POINTER_ATTRIBUTE_MAPPED, inside), 1);
  VCHECK(get<void*>(CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, inside) == reinterpret_cast<void*>(base));
  VCHECK_EQ(get<size_t>(CU_POINTER_ATTRIBUTE_RANGE_SIZE, inside), size_t{4096});
  VCHECK(get<CUcontext>(CU_POINTER_ATTRIBUTE_CONTEXT, inside) == e.ctx[0]);
  // The host cannot address device memory: no host pointer.
  get<void*>(CU_POINTER_ATTRIBUTE_HOST_POINTER, inside, CUDA_ERROR_INVALID_VALUE);
  // Anything of the allocation answers alike; the one past its end is no pointer.
  VCHECK_EQ(get<unsigned long long>(CU_POINTER_ATTRIBUTE_BUFFER_ID, base),
            get<unsigned long long>(CU_POINTER_ATTRIBUTE_BUFFER_ID, inside));
  get<unsigned int>(CU_POINTER_ATTRIBUTE_MEMORY_TYPE, base + 4096 + 65536, CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(cuMemFree(base), CUDA_SUCCESS);
  // Freed: no longer a pointer.
  get<unsigned int>(CU_POINTER_ATTRIBUTE_MEMORY_TYPE, base, CUDA_ERROR_INVALID_VALUE);
}

VTEST(each_allocation_has_its_own_buffer_id_and_none_is_reused) {
  Env e;
  CUdeviceptr a = 0, b = 0, c = 0;
  VCHECK_EQ(cuMemAlloc(&a, 256), CUDA_SUCCESS);
  VCHECK_EQ(cuMemAlloc(&b, 256), CUDA_SUCCESS);
  const auto ida = get<unsigned long long>(CU_POINTER_ATTRIBUTE_BUFFER_ID, a);
  const auto idb = get<unsigned long long>(CU_POINTER_ATTRIBUTE_BUFFER_ID, b);
  VCHECK(ida != idb);
  VCHECK_EQ(cuMemFree(a), CUDA_SUCCESS);
  VCHECK_EQ(cuMemAlloc(&c, 256), CUDA_SUCCESS);   // a later allocation never takes a freed one's id
  const auto idc = get<unsigned long long>(CU_POINTER_ATTRIBUTE_BUFFER_ID, c);
  VCHECK(idc != ida && idc != idb);
  VCHECK_EQ(cuMemFree(b), CUDA_SUCCESS);
  VCHECK_EQ(cuMemFree(c), CUDA_SUCCESS);
}

VTEST(the_device_ordinal_is_the_device_the_memory_is_on) {
  Env e;
  CUdeviceptr on1 = 0;
  VCHECK_EQ(cuCtxSetCurrent(e.ctx[1]), CUDA_SUCCESS);
  VCHECK_EQ(cuMemAlloc(&on1, 512), CUDA_SUCCESS);
  VCHECK_EQ(get<int>(CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, on1), 1);
  VCHECK(get<CUcontext>(CU_POINTER_ATTRIBUTE_CONTEXT, on1) == e.ctx[1]);
  // Asked from device 0's context, the memory is still device 1's, in the
  // context it was made in (the device's primary).
  VCHECK_EQ(cuCtxSetCurrent(e.ctx[0]), CUDA_SUCCESS);
  VCHECK_EQ(get<int>(CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, on1), 1);
  VCHECK(get<CUcontext>(CU_POINTER_ATTRIBUTE_CONTEXT, on1) == e.ctx[1]);
  CUdeviceptr mgd = 0;
  VCHECK_EQ(cuCtxSetCurrent(e.ctx[1]), CUDA_SUCCESS);
  VCHECK_EQ(cuMemAllocManaged(&mgd, 4096, CU_MEM_ATTACH_GLOBAL), CUDA_SUCCESS);
  VCHECK_EQ(cuCtxSetCurrent(e.ctx[0]), CUDA_SUCCESS);
  VCHECK_EQ(get<int>(CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, mgd), 1);   // allocated against device 1
  VCHECK_EQ(cuMemFree(mgd), CUDA_SUCCESS);
  VCHECK_EQ(cuMemFree(on1), CUDA_SUCCESS);
}

VTEST(managed_memory_is_device_memory_the_host_reaches_at_the_same_address) {
  Env e;
  CUdeviceptr m = 0;
  VCHECK_EQ(cuMemAllocManaged(&m, 8192, CU_MEM_ATTACH_GLOBAL), CUDA_SUCCESS);
  VCHECK_EQ(get<int>(CU_POINTER_ATTRIBUTE_IS_MANAGED, m + 5), 1);
  VCHECK_EQ(get<unsigned int>(CU_POINTER_ATTRIBUTE_MEMORY_TYPE, m), static_cast<unsigned>(CU_MEMORYTYPE_DEVICE));
  VCHECK(get<void*>(CU_POINTER_ATTRIBUTE_HOST_POINTER, m + 8) == reinterpret_cast<void*>(m + 8));
  VCHECK_EQ(get<CUdeviceptr>(CU_POINTER_ATTRIBUTE_DEVICE_POINTER, m + 8), m + 8);
  VCHECK(get<void*>(CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, m + 100) == reinterpret_cast<void*>(m));
  VCHECK_EQ(get<size_t>(CU_POINTER_ATTRIBUTE_RANGE_SIZE, m + 100), size_t{8192});
  VCHECK_EQ(get<int>(CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, m), 0);
  VCHECK_EQ(cuMemFree(m), CUDA_SUCCESS);
}

VTEST(pinned_and_registered_host_memory_is_host_memory) {
  Env e;
  void* pinned = nullptr;
  VCHECK_EQ(cuMemHostAlloc(&pinned, 10000, 0), CUDA_SUCCESS);
  const CUdeviceptr pp = reinterpret_cast<CUdeviceptr>(pinned) + 777;
  VCHECK_EQ(get<unsigned int>(CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pp), static_cast<unsigned>(CU_MEMORYTYPE_HOST));
  VCHECK(get<void*>(CU_POINTER_ATTRIBUTE_HOST_POINTER, pp) == reinterpret_cast<void*>(pp));
  VCHECK_EQ(get<CUdeviceptr>(CU_POINTER_ATTRIBUTE_DEVICE_POINTER, pp), pp);   // unified addressing
  VCHECK(get<void*>(CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, pp) == pinned);
  VCHECK_EQ(get<size_t>(CU_POINTER_ATTRIBUTE_RANGE_SIZE, pp), size_t{10000});
  VCHECK_EQ(get<int>(CU_POINTER_ATTRIBUTE_IS_MANAGED, pp), 0);
  VCHECK_EQ(get<int>(CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, pp), 0);
  VCHECK_EQ(cuMemFreeHost(pinned), CUDA_SUCCESS);
  get<unsigned int>(CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pp, CUDA_ERROR_INVALID_VALUE);

  void* plain = std::aligned_alloc(4096, 8192);
  const CUdeviceptr q = reinterpret_cast<CUdeviceptr>(plain);
  get<unsigned int>(CU_POINTER_ATTRIBUTE_MEMORY_TYPE, q, CUDA_ERROR_INVALID_VALUE);   // not CUDA's yet
  VCHECK_EQ(cuMemHostRegister(plain, 8192, 0), CUDA_SUCCESS);
  VCHECK_EQ(get<unsigned int>(CU_POINTER_ATTRIBUTE_MEMORY_TYPE, q + 10), static_cast<unsigned>(CU_MEMORYTYPE_HOST));
  VCHECK_EQ(get<size_t>(CU_POINTER_ATTRIBUTE_RANGE_SIZE, q + 10), size_t{8192});
  VCHECK_EQ(cuMemHostUnregister(plain), CUDA_SUCCESS);
  get<unsigned int>(CU_POINTER_ATTRIBUTE_MEMORY_TYPE, q, CUDA_ERROR_INVALID_VALUE);
  std::free(plain);
}

VTEST(a_pointer_cuda_does_not_know_is_an_error_alone_and_nulls_in_a_batch) {
  Env e;
  int on_stack = 0;
  const CUdeviceptr nowhere = reinterpret_cast<CUdeviceptr>(&on_stack);
  get<unsigned int>(CU_POINTER_ATTRIBUTE_MEMORY_TYPE, nowhere, CUDA_ERROR_INVALID_VALUE);
  get<int>(CU_POINTER_ATTRIBUTE_IS_MANAGED, nowhere, CUDA_ERROR_INVALID_VALUE);
  get<CUcontext>(CU_POINTER_ATTRIBUTE_CONTEXT, 0, CUDA_ERROR_INVALID_VALUE);

  CUpointer_attribute attrs[] = {CU_POINTER_ATTRIBUTE_MEMORY_TYPE, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                 CU_POINTER_ATTRIBUTE_BUFFER_ID, CU_POINTER_ATTRIBUTE_IS_MANAGED,
                                 CU_POINTER_ATTRIBUTE_CONTEXT, CU_POINTER_ATTRIBUTE_RANGE_SIZE};
  unsigned int mem_type;
  int ordinal, managed;
  unsigned long long buffer;
  CUcontext ctx;
  size_t range;
  std::memset(&mem_type, 0xAA, sizeof mem_type);
  std::memset(&ordinal, 0xAA, sizeof ordinal);
  std::memset(&buffer, 0xAA, sizeof buffer);
  std::memset(&managed, 0xAA, sizeof managed);
  std::memset(&ctx, 0xAA, sizeof ctx);
  std::memset(&range, 0xAA, sizeof range);
  void* data[] = {&mem_type, &ordinal, &buffer, &managed, &ctx, &range};
  VCHECK_EQ(cuPointerGetAttributes(6, attrs, data, nowhere), CUDA_SUCCESS);
  VCHECK_EQ(mem_type, 0u);   // not device (2), not host (1)
  VCHECK_EQ(ordinal, 0);
  VCHECK_EQ(buffer, 0ull);
  VCHECK_EQ(managed, 0);
  VCHECK(ctx == nullptr);
  VCHECK_EQ(range, size_t{0});
}

VTEST(a_batch_of_attributes_matches_the_single_queries) {
  Env e;
  CUdeviceptr p = 0;
  VCHECK_EQ(cuMemAlloc(&p, 2048), CUDA_SUCCESS);
  CUpointer_attribute attrs[] = {CU_POINTER_ATTRIBUTE_MEMORY_TYPE, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                 CU_POINTER_ATTRIBUTE_BUFFER_ID, CU_POINTER_ATTRIBUTE_IS_MANAGED,
                                 CU_POINTER_ATTRIBUTE_RANGE_SIZE, CU_POINTER_ATTRIBUTE_DEVICE_POINTER};
  unsigned int mem_type = 0;
  int ordinal = -1, managed = -1;
  unsigned long long buffer = 0;
  size_t range = 0;
  CUdeviceptr dptr = 0;
  void* data[] = {&mem_type, &ordinal, &buffer, &managed, &range, &dptr};
  VCHECK_EQ(cuPointerGetAttributes(6, attrs, data, p + 3), CUDA_SUCCESS);
  VCHECK_EQ(mem_type, static_cast<unsigned>(CU_MEMORYTYPE_DEVICE));
  VCHECK_EQ(ordinal, 0);
  VCHECK_EQ(buffer, get<unsigned long long>(CU_POINTER_ATTRIBUTE_BUFFER_ID, p));
  VCHECK_EQ(managed, 0);
  VCHECK_EQ(range, size_t{2048});
  VCHECK_EQ(dptr, p + 3);

  // What it cannot answer is refused, and so is a malformed request.
  CUpointer_attribute p2p[] = {CU_POINTER_ATTRIBUTE_P2P_TOKENS};
  void* one[] = {&mem_type};
  VCHECK_EQ(cuPointerGetAttributes(1, p2p, one, p), CUDA_ERROR_INVALID_VALUE);
  CUpointer_attribute unknown[] = {static_cast<CUpointer_attribute>(99)};
  VCHECK_EQ(cuPointerGetAttributes(1, unknown, one, p), CUDA_ERROR_INVALID_VALUE);
  void* nulldata[] = {nullptr};
  VCHECK_EQ(cuPointerGetAttributes(1, attrs, nulldata, p), CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(cuPointerGetAttributes(1, nullptr, one, p), CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(cuPointerGetAttributes(0, nullptr, nullptr, p), CUDA_SUCCESS);
  VCHECK_EQ(cuMemFree(p), CUDA_SUCCESS);
}

VTEST(sync_memops_is_a_boolean_that_can_be_set_on_an_allocation) {
  Env e;
  CUdeviceptr p = 0;
  VCHECK_EQ(cuMemAlloc(&p, 1024), CUDA_SUCCESS);
  VCHECK_EQ(get<int>(CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, p), 0);
  unsigned int on = 1, off = 0, bad = 2;
  VCHECK_EQ(cuPointerSetAttribute(&on, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, p + 64), CUDA_SUCCESS);   // any byte of it
  VCHECK_EQ(get<int>(CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, p), 1);
  VCHECK_EQ(get<int>(CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, p + 1000), 1);
  VCHECK_EQ(cuPointerSetAttribute(&off, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, p), CUDA_SUCCESS);
  VCHECK_EQ(get<int>(CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, p), 0);
  VCHECK_EQ(cuPointerSetAttribute(&bad, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, p), CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(cuPointerSetAttribute(&on, CU_POINTER_ATTRIBUTE_BUFFER_ID, p), CUDA_ERROR_INVALID_VALUE);   // read-only
  VCHECK_EQ(cuPointerSetAttribute(nullptr, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, p), CUDA_ERROR_INVALID_VALUE);
  int on_stack = 0;
  VCHECK_EQ(cuPointerSetAttribute(&on, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, reinterpret_cast<CUdeviceptr>(&on_stack)),
            CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(cuMemFree(p), CUDA_SUCCESS);
}

VTEST_MAIN
