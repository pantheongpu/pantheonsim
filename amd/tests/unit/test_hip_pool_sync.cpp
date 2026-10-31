// A synchronization gives back what the stream-ordered pools keep beyond their release thresholds
// (release_pools_at_sync in hip_api.cpp): hipStreamSynchronize, hipDeviceSynchronize and hipEventSynchronize each do,
// and a program that has made no pool is not made to take the runtime's lock to find that out.
#include <cstdint>
#include <cstdio>

#include "vgpu_hip.h"

extern "C" {
hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream);
hipError_t hipFreeAsync(void* ptr, hipStream_t stream);
hipError_t hipDeviceGetDefaultMemPool(void** pool, int ordinal);
hipError_t hipMemPoolSetAttribute(void* pool, int attr, void* value);
hipError_t hipMemPoolGetAttribute(void* pool, int attr, void* value);
}

namespace {
constexpr int kReleaseThreshold = 4, kReservedCurrent = 5;
int failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL %s\n", what);
    ++failures;
  }
}
uint64_t reserved(void* pool) {
  uint64_t v = ~uint64_t{0};
  if (hipMemPoolGetAttribute(pool, kReservedCurrent, &v) != hipSuccess) return ~uint64_t{0};
  return v;
}
}  // namespace

int main() {
  // Before any pool exists a synchronization is only a synchronization.
  hipStream_t s = nullptr;
  check(hipStreamCreate(&s) == hipSuccess, "stream");
  check(hipStreamSynchronize(s) == hipSuccess && hipDeviceSynchronize() == hipSuccess, "synchronize without a pool");
  hipEvent_t e = nullptr;
  check(hipEventCreate(&e) == hipSuccess && hipEventRecord(e, s) == hipSuccess && hipEventSynchronize(e) == hipSuccess,
        "event synchronize without a pool");

  void* pool = nullptr;
  check(hipDeviceGetDefaultMemPool(&pool, 0) == hipSuccess && pool, "default pool");
  uint64_t keep_all = ~uint64_t{0}, keep_none = 0;
  for (int round = 0; round < 3; ++round) {
    check(hipMemPoolSetAttribute(pool, kReleaseThreshold, &keep_all) == hipSuccess, "keep everything");
    void* p = nullptr;
    check(hipMallocAsync(&p, 1 << 20, s) == hipSuccess && p, "allocate");
    check(hipFreeAsync(p, s) == hipSuccess, "free");
    check(hipStreamSynchronize(s) == hipSuccess, "synchronize");
    check(reserved(pool) >= (1u << 20), "the pool keeps what was freed while its threshold allows");
    check(hipMemPoolSetAttribute(pool, kReleaseThreshold, &keep_none) == hipSuccess, "keep nothing");
    check(reserved(pool) >= (1u << 20), "(lowering the threshold gives nothing back by itself)");
    const char* which = "?";
    hipError_t rc = hipErrorUnknown;
    if (round == 0) {
      which = "hipStreamSynchronize gives the pool's excess back";
      rc = hipStreamSynchronize(s);
    } else if (round == 1) {
      which = "hipDeviceSynchronize gives the pool's excess back";
      rc = hipDeviceSynchronize();
    } else {
      which = "hipEventSynchronize gives the pool's excess back";
      rc = hipEventRecord(e, s) == hipSuccess ? hipEventSynchronize(e) : hipErrorUnknown;
    }
    check(rc == hipSuccess, "the synchronization");
    check(reserved(pool) == 0, which);
  }
  hipEventDestroy(e);
  hipStreamDestroy(s);
  if (failures) return 1;
  std::printf("ok synchronizations give back what the pools keep beyond their thresholds\n");
  return 0;
}
