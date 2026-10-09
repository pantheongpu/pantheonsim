// The driver's stream-ordered memory pools: cuMemPoolCreate and the rest of
// the cuMemPool* family, cuDeviceGet/SetMemPool, cuMemAllocFromPoolAsync, and
// cuMemAllocAsync / cuMemFreeAsync going through the device's current pool.
// A program that asks cuDeviceGetAttribute whether pools are supported was
// told no, and the entry points were absent; cuda-python's memory resources
// and anything else that manages a pool through the driver need them.
//   - the accounting the pool attributes report (used, reserved, high-water marks);
//   - a release threshold keeps freed memory and hands it out again;
//   - cuMemPoolTrimTo gives it back;
//   - pools are made, made current for a device, and destroyed under the rules
//     the documentation states;
//   - what cannot be done here -- sharing a pool with another process, a pool
//     on the host -- is refused by name.
#include <cuda.h>

#include <cstdint>
#include <cstring>

#include "vtest.hpp"

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

cuuint64_t attr64(CUmemoryPool pool, CUmemPool_attribute a) {
  cuuint64_t v = ~cuuint64_t{0};
  VCHECK_EQ(cuMemPoolGetAttribute(pool, a, &v), CUDA_SUCCESS);
  return v;
}

CUmemoryPool make_pool(int device) {
  CUmemPoolProps props;
  std::memset(&props, 0, sizeof props);
  props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
  props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  props.location.id = device;
  CUmemoryPool pool = nullptr;
  VCHECK_EQ(cuMemPoolCreate(&pool, &props), CUDA_SUCCESS);
  return pool;
}

}  // namespace

VTEST(pools_are_supported_and_each_device_has_a_default) {
  Env e;
  CUdevice dev = 0;
  cuDeviceGet(&dev, 0);
  int supported = 0;
  VCHECK_EQ(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, dev), CUDA_SUCCESS);
  VCHECK_EQ(supported, 1);
  CUmemoryPool def0 = nullptr, def1 = nullptr, cur0 = nullptr;
  VCHECK_EQ(cuDeviceGetDefaultMemPool(&def0, 0), CUDA_SUCCESS);
  VCHECK_EQ(cuDeviceGetDefaultMemPool(&def1, 1), CUDA_SUCCESS);
  VCHECK(def0 != nullptr && def1 != nullptr && def0 != def1);
  VCHECK_EQ(cuDeviceGetMemPool(&cur0, 0), CUDA_SUCCESS);   // until told otherwise, the default
  VCHECK(cur0 == def0);
  CUmemoryPool again = nullptr;
  VCHECK_EQ(cuDeviceGetDefaultMemPool(&again, 0), CUDA_SUCCESS);
  VCHECK(again == def0);
  VCHECK_EQ(cuDeviceGetDefaultMemPool(nullptr, 0), CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(cuDeviceGetDefaultMemPool(&again, 7), CUDA_ERROR_INVALID_DEVICE);
}

VTEST(the_attributes_count_what_is_used_and_what_is_reserved) {
  Env e;
  CUmemoryPool pool = make_pool(0);
  // Defaults: a release threshold of zero, and the three reuse policies on.
  VCHECK_EQ(attr64(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD), cuuint64_t{0});
  int policy = 0;
  VCHECK_EQ(cuMemPoolGetAttribute(pool, CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, &policy), CUDA_SUCCESS);
  VCHECK_EQ(policy, 1);
  VCHECK_EQ(cuMemPoolGetAttribute(pool, CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC, &policy), CUDA_SUCCESS);
  VCHECK_EQ(policy, 1);
  VCHECK_EQ(cuMemPoolGetAttribute(pool, CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES, &policy), CUDA_SUCCESS);
  VCHECK_EQ(policy, 1);

  CUdeviceptr a = 0, b = 0;
  VCHECK_EQ(cuMemAllocFromPoolAsync(&a, 1 << 20, pool, nullptr), CUDA_SUCCESS);
  VCHECK_EQ(cuMemAllocFromPoolAsync(&b, 3 << 20, pool, nullptr), CUDA_SUCCESS);
  VCHECK(a != 0 && b != 0 && a != b);
  VCHECK_EQ(attr64(pool, CU_MEMPOOL_ATTR_USED_MEM_CURRENT), cuuint64_t{4u << 20});
  VCHECK_EQ(attr64(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT), cuuint64_t{4u << 20});
  VCHECK_EQ(cuMemFreeAsync(b, nullptr), CUDA_SUCCESS);
  VCHECK_EQ(attr64(pool, CU_MEMPOOL_ATTR_USED_MEM_CURRENT), cuuint64_t{1u << 20});
  // Threshold zero: what is freed goes straight back to the device.
  VCHECK_EQ(attr64(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT), cuuint64_t{1u << 20});
  VCHECK_EQ(attr64(pool, CU_MEMPOOL_ATTR_USED_MEM_HIGH), cuuint64_t{4u << 20});
  VCHECK_EQ(attr64(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH), cuuint64_t{4u << 20});
  VCHECK_EQ(cuMemFreeAsync(a, nullptr), CUDA_SUCCESS);
  VCHECK_EQ(attr64(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT), cuuint64_t{0});

  // A high-water mark is reset by writing zero to it, and only by that.
  cuuint64_t zero = 0, other = 5;
  VCHECK_EQ(cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_USED_MEM_HIGH, &other), CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_USED_MEM_HIGH, &zero), CUDA_SUCCESS);
  VCHECK_EQ(cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH, &zero), CUDA_SUCCESS);
  VCHECK_EQ(attr64(pool, CU_MEMPOOL_ATTR_USED_MEM_HIGH), cuuint64_t{0});
  VCHECK_EQ(attr64(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH), cuuint64_t{0});
  // The current totals are read-only; an attribute that is not one is invalid.
  VCHECK_EQ(cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_USED_MEM_CURRENT, &zero), CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT, &zero), CUDA_ERROR_INVALID_VALUE);
  cuuint64_t v;
  VCHECK_EQ(cuMemPoolGetAttribute(pool, static_cast<CUmemPool_attribute>(99), &v), CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(cuMemPoolDestroy(pool), CUDA_SUCCESS);
}

VTEST(a_release_threshold_keeps_freed_memory_and_hands_it_out_again) {
  Env e;
  CUmemoryPool pool = make_pool(0);
  cuuint64_t threshold = 8u << 20;
  VCHECK_EQ(cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &threshold), CUDA_SUCCESS);
  CUdeviceptr a = 0;
  VCHECK_EQ(cuMemAllocFromPoolAsync(&a, 2 << 20, pool, nullptr), CUDA_SUCCESS);
  VCHECK_EQ(cuMemFreeAsync(a, nullptr), CUDA_SUCCESS);
  // Kept: nothing is in use, the two megabytes are still reserved.
  VCHECK_EQ(attr64(pool, CU_MEMPOOL_ATTR_USED_MEM_CURRENT), cuuint64_t{0});
  VCHECK_EQ(attr64(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT), cuuint64_t{2u << 20});
  // The next request that fits it is given the same block.
  CUdeviceptr b = 0;
  VCHECK_EQ(cuMemAllocFromPoolAsync(&b, 2 << 20, pool, nullptr), CUDA_SUCCESS);
  VCHECK_EQ(b, a);
  VCHECK_EQ(attr64(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT), cuuint64_t{2u << 20});
  // A request far smaller than the cached block does not take it.
  VCHECK_EQ(cuMemFreeAsync(b, nullptr), CUDA_SUCCESS);
  CUdeviceptr small = 0;
  VCHECK_EQ(cuMemAllocFromPoolAsync(&small, 1024, pool, nullptr), CUDA_SUCCESS);
  VCHECK(small != a);
  VCHECK_EQ(attr64(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT), cuuint64_t{(2u << 20) + 1024});
  VCHECK_EQ(cuMemFreeAsync(small, nullptr), CUDA_SUCCESS);

  // Trim: keep what is asked and give the rest back, oldest first.
  VCHECK_EQ(cuMemPoolTrimTo(pool, 1 << 30), CUDA_SUCCESS);   // keeping more than is held releases nothing
  VCHECK_EQ(attr64(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT), cuuint64_t{(2u << 20) + 1024});
  VCHECK_EQ(cuMemPoolTrimTo(pool, 0), CUDA_SUCCESS);
  VCHECK_EQ(attr64(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT), cuuint64_t{0});

  // Lowering the threshold gives memory back at once.
  VCHECK_EQ(cuMemAllocFromPoolAsync(&a, 2 << 20, pool, nullptr), CUDA_SUCCESS);
  VCHECK_EQ(cuMemFreeAsync(a, nullptr), CUDA_SUCCESS);
  VCHECK_EQ(attr64(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT), cuuint64_t{2u << 20});
  threshold = 0;
  VCHECK_EQ(cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &threshold), CUDA_SUCCESS);
  VCHECK_EQ(attr64(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT), cuuint64_t{0});
  VCHECK_EQ(cuMemPoolDestroy(pool), CUDA_SUCCESS);
}

VTEST(memory_from_a_pool_is_ordinary_device_memory) {
  Env e;
  CUmemoryPool pool = make_pool(0);
  CUdeviceptr p = 0;
  VCHECK_EQ(cuMemAllocFromPoolAsync(&p, 256, pool, nullptr), CUDA_SUCCESS);
  uint8_t out[256], in[256];
  for (int i = 0; i < 256; ++i) in[i] = static_cast<uint8_t>(i * 7);
  VCHECK_EQ(cuMemcpyHtoD(p, in, sizeof in), CUDA_SUCCESS);
  VCHECK_EQ(cuMemcpyDtoH(out, p, sizeof out), CUDA_SUCCESS);
  VCHECK_EQ(std::memcmp(in, out, sizeof in), 0);
  VCHECK_EQ(cuMemFreeAsync(p, nullptr), CUDA_SUCCESS);
  // Freed with a zero threshold, it is the device's again: not reachable.
  VCHECK(cuMemcpyDtoH(out, p, sizeof out) != CUDA_SUCCESS);
  VCHECK_EQ(cuMemPoolDestroy(pool), CUDA_SUCCESS);
}

VTEST(allocating_without_naming_a_pool_uses_the_devices_current_pool) {
  Env e;
  CUmemoryPool def = nullptr, mine = make_pool(0);
  VCHECK_EQ(cuDeviceGetMemPool(&def, 0), CUDA_SUCCESS);
  const cuuint64_t def_before = attr64(def, CU_MEMPOOL_ATTR_USED_MEM_CURRENT);
  CUdeviceptr p = 0;
  VCHECK_EQ(cuMemAllocAsync(&p, 1 << 16, nullptr), CUDA_SUCCESS);
  VCHECK_EQ(attr64(def, CU_MEMPOOL_ATTR_USED_MEM_CURRENT), def_before + (1 << 16));
  VCHECK_EQ(cuMemFreeAsync(p, nullptr), CUDA_SUCCESS);
  VCHECK_EQ(attr64(def, CU_MEMPOOL_ATTR_USED_MEM_CURRENT), def_before);

  VCHECK_EQ(cuDeviceSetMemPool(0, mine), CUDA_SUCCESS);
  CUmemoryPool cur = nullptr;
  VCHECK_EQ(cuDeviceGetMemPool(&cur, 0), CUDA_SUCCESS);
  VCHECK(cur == mine);
  VCHECK_EQ(cuMemAllocAsync(&p, 1 << 16, nullptr), CUDA_SUCCESS);
  VCHECK_EQ(attr64(mine, CU_MEMPOOL_ATTR_USED_MEM_CURRENT), cuuint64_t{1u << 16});
  VCHECK_EQ(attr64(def, CU_MEMPOOL_ATTR_USED_MEM_CURRENT), def_before);   // the default was not touched
  VCHECK_EQ(cuMemFreeAsync(p, nullptr), CUDA_SUCCESS);

  // The default pool is still the default, and a pool made for device 0 is
  // not device 1's to use.
  CUmemoryPool still_default = nullptr;
  VCHECK_EQ(cuDeviceGetDefaultMemPool(&still_default, 0), CUDA_SUCCESS);
  VCHECK(still_default == def);
  VCHECK_EQ(cuDeviceSetMemPool(1, mine), CUDA_ERROR_INVALID_VALUE);

  // Destroying the current pool puts the default back.
  VCHECK_EQ(cuMemPoolDestroy(mine), CUDA_SUCCESS);
  VCHECK_EQ(cuDeviceGetMemPool(&cur, 0), CUDA_SUCCESS);
  VCHECK(cur == def);
  // A destroyed pool is no handle any longer.
  VCHECK_EQ(cuMemPoolTrimTo(mine, 0), CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(cuDeviceSetMemPool(0, mine), CUDA_ERROR_INVALID_VALUE);
}

VTEST(a_pool_allocates_on_the_device_it_was_made_for) {
  Env e;
  CUmemoryPool pool1 = make_pool(1);
  CUdeviceptr p = 0;
  // Device 0 is current; the allocation still comes from device 1's memory.
  VCHECK_EQ(cuMemAllocFromPoolAsync(&p, 4096, pool1, nullptr), CUDA_SUCCESS);
  CUdeviceptr base = 0;
  size_t size = 0;
  VCHECK_EQ(cuMemGetAddressRange(&base, &size, p), CUDA_SUCCESS);
  VCHECK_EQ(size, size_t{4096});
  // Each device's memory sits in its own window of the address space, so the
  // pointer is nearer an allocation made on device 1 than one made on device 0.
  CUdeviceptr on0 = 0, on1 = 0;
  VCHECK_EQ(cuMemAlloc(&on0, 4096), CUDA_SUCCESS);
  VCHECK_EQ(cuCtxSetCurrent(e.ctx[1]), CUDA_SUCCESS);
  VCHECK_EQ(cuMemAlloc(&on1, 4096), CUDA_SUCCESS);
  auto dist = [](CUdeviceptr a, CUdeviceptr b) { return a > b ? a - b : b - a; };
  VCHECK(dist(p, on1) < dist(p, on0));
  VCHECK_EQ(cuMemFree(on1), CUDA_SUCCESS);
  VCHECK_EQ(cuCtxSetCurrent(e.ctx[0]), CUDA_SUCCESS);
  VCHECK_EQ(cuMemFree(on0), CUDA_SUCCESS);
  VCHECK_EQ(cuMemFreeAsync(p, nullptr), CUDA_SUCCESS);
  VCHECK_EQ(cuMemPoolDestroy(pool1), CUDA_SUCCESS);
}

VTEST(pools_are_made_and_destroyed_under_the_documented_rules) {
  Env e;
  CUmemPoolProps props;
  std::memset(&props, 0, sizeof props);
  props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
  props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  props.location.id = 0;
  CUmemoryPool pool = nullptr;
  VCHECK_EQ(cuMemPoolCreate(nullptr, &props), CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(cuMemPoolCreate(&pool, nullptr), CUDA_ERROR_INVALID_VALUE);

  CUmemPoolProps bad = props;
  bad.allocType = CU_MEM_ALLOCATION_TYPE_INVALID;
  VCHECK_EQ(cuMemPoolCreate(&pool, &bad), CUDA_ERROR_INVALID_VALUE);
  bad = props;
  bad.location.id = 9;
  VCHECK_EQ(cuMemPoolCreate(&pool, &bad), CUDA_ERROR_INVALID_VALUE);   // an RTX 3060 says invalid value, not invalid device
  bad = props;
  // CU_MEM_LOCATION_TYPE_HOST, spelled by value: cuda.h declares it from 12.2 and the older toolkits CI builds with do not
  // (INVALID 0, DEVICE 1, HOST 2, HOST_NUMA 3, HOST_NUMA_CURRENT 4).
  bad.location.type = static_cast<CUmemLocationType>(2);   // a pool on the host
  VCHECK_EQ(cuMemPoolCreate(&pool, &bad), CUDA_ERROR_NOT_SUPPORTED);
  bad = props;
  bad.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;   // a pool another process could use
  VCHECK_EQ(cuMemPoolCreate(&pool, &bad), CUDA_ERROR_INVALID_VALUE);   // the device supports no handle type

  VCHECK_EQ(cuMemPoolCreate(&pool, &props), CUDA_SUCCESS);
  CUdeviceptr p = 0;
  VCHECK_EQ(cuMemAllocFromPoolAsync(&p, 64, pool, nullptr), CUDA_SUCCESS);
  VCHECK_EQ(cuMemPoolDestroy(pool), CUDA_ERROR_INVALID_VALUE);   // an allocation is still out
  VCHECK_EQ(cuMemFreeAsync(p, nullptr), CUDA_SUCCESS);
  VCHECK_EQ(cuMemPoolDestroy(pool), CUDA_SUCCESS);
  VCHECK_EQ(cuMemPoolDestroy(pool), CUDA_ERROR_INVALID_VALUE);   // and again

  CUmemoryPool def = nullptr;
  VCHECK_EQ(cuDeviceGetDefaultMemPool(&def, 0), CUDA_SUCCESS);
  VCHECK_EQ(cuMemPoolDestroy(def), CUDA_ERROR_INVALID_VALUE);    // the device's own
  VCHECK_EQ(cuMemAllocFromPoolAsync(&p, 64, reinterpret_cast<CUmemoryPool>(0x1234), nullptr), CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(cuMemAllocFromPoolAsync(nullptr, 64, def, nullptr), CUDA_ERROR_INVALID_VALUE);
}

VTEST(zero_bytes_and_other_memory_through_the_async_calls) {
  Env e;
  CUdeviceptr p = 1;
  VCHECK_EQ(cuMemAllocAsync(&p, 0, nullptr), CUDA_SUCCESS);   // a zero-byte request succeeds, with no pointer
  VCHECK_EQ(p, CUdeviceptr{0});
  VCHECK_EQ(cuMemFreeAsync(0, nullptr), CUDA_SUCCESS);
  VCHECK_EQ(cuMemAllocAsync(nullptr, 8, nullptr), CUDA_ERROR_INVALID_VALUE);
  // cuMemFreeAsync frees an ordinary allocation too, as it always did here.
  CUdeviceptr plain = 0;
  VCHECK_EQ(cuMemAlloc(&plain, 512), CUDA_SUCCESS);
  VCHECK_EQ(cuMemFreeAsync(plain, nullptr), CUDA_SUCCESS);
  VCHECK(cuMemFree(plain) != CUDA_SUCCESS);   // it is gone
}

VTEST(a_pools_access_starts_as_its_own_device_and_follows_what_is_set) {
  Env e;
  CUmemoryPool pool = make_pool(0);
  CUmemLocation loc0, loc1;
  std::memset(&loc0, 0, sizeof loc0);
  std::memset(&loc1, 0, sizeof loc1);
  loc0.type = loc1.type = CU_MEM_LOCATION_TYPE_DEVICE;
  loc0.id = 0;
  loc1.id = 1;
  CUmemAccess_flags flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
  VCHECK_EQ(cuMemPoolGetAccess(&flags, pool, &loc0), CUDA_SUCCESS);
  VCHECK_EQ(static_cast<int>(flags), static_cast<int>(CU_MEM_ACCESS_FLAGS_PROT_READWRITE));
  VCHECK_EQ(cuMemPoolGetAccess(&flags, pool, &loc1), CUDA_SUCCESS);
  VCHECK_EQ(static_cast<int>(flags), static_cast<int>(CU_MEM_ACCESS_FLAGS_PROT_NONE));

  CUmemAccessDesc desc;
  std::memset(&desc, 0, sizeof desc);
  desc.location = loc1;
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  VCHECK_EQ(cuMemPoolSetAccess(pool, &desc, 1), CUDA_SUCCESS);
  VCHECK_EQ(cuMemPoolGetAccess(&flags, pool, &loc1), CUDA_SUCCESS);
  VCHECK_EQ(static_cast<int>(flags), static_cast<int>(CU_MEM_ACCESS_FLAGS_PROT_READWRITE));
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
  VCHECK_EQ(cuMemPoolSetAccess(pool, &desc, 1), CUDA_SUCCESS);
  VCHECK_EQ(cuMemPoolGetAccess(&flags, pool, &loc1), CUDA_SUCCESS);
  VCHECK_EQ(static_cast<int>(flags), static_cast<int>(CU_MEM_ACCESS_FLAGS_PROT_NONE));

  desc.location.id = 5;   // no such device
  VCHECK_EQ(cuMemPoolSetAccess(pool, &desc, 1), CUDA_ERROR_INVALID_VALUE);
  desc.location = loc1;
  desc.flags = static_cast<CUmemAccess_flags>(2);   // write-only is not a protection
  VCHECK_EQ(cuMemPoolSetAccess(pool, &desc, 1), CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(cuMemPoolSetAccess(pool, nullptr, 1), CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(cuMemPoolSetAccess(pool, nullptr, 0), CUDA_SUCCESS);
  loc1.id = 5;
  VCHECK_EQ(cuMemPoolGetAccess(&flags, pool, &loc1), CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(cuMemPoolDestroy(pool), CUDA_SUCCESS);
}

VTEST(sharing_a_pool_with_another_process_is_refused_as_the_card_refuses_it) {
  Env e;
  CUmemoryPool pool = make_pool(0);
  int fd = 0;
  VCHECK_EQ(cuMemPoolExportToShareableHandle(&fd, pool, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0), CUDA_ERROR_INVALID_VALUE);
  CUmemoryPool imported = nullptr;
  VCHECK_EQ(cuMemPoolImportFromShareableHandle(&imported, &fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0),
            CUDA_ERROR_NOT_SUPPORTED);
  CUmemPoolPtrExportData data;
  CUdeviceptr p = 0;
  VCHECK_EQ(cuMemPoolExportPointer(&data, 0), CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(cuMemPoolImportPointer(&p, pool, &data), CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(cuMemPoolDestroy(pool), CUDA_SUCCESS);
}

VTEST_MAIN
