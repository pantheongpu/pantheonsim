// Allocation records: what CUPTI reports for device, pinned and stream-ordered
// memory, for the pools the stream-ordered allocator draws from, and the older
// memory kind made from the same events. The program allocates and frees memory
// of each kind and the trace is printed in the order it arrived; run against
// NVIDIA's libcupti on a card it gives nvidia/tests/data/cupti_memory.expected.
//
// What this shows, and the shim must reproduce (measured on an RTX 3060):
//  * a record for each allocation and each release, with the allocation's kind,
//    size and the stream it was ordered on; plain cudaMalloc is not ordered;
//  * for pool memory, the pool at that moment: its size (in steps of 32 MiB, the
//    size the driver has grown it to), its release threshold and what is handed
//    out;
//  * a pool is created the first time it is used (the device's own) or when the
//    program makes it, destroyed when the program destroys it, and "trimmed"
//    whenever it is: by the driver when the stream that freed its memory has been
//    waited on (every pool of the device is then reported), and by
//    cudaMemPoolTrimTo. A trim that gave memory back says 1 where the structure
//    has the number of bytes to keep (so does NVIDIA's).
//  * the older memory kind lists each allocation once, when it is freed, or when
//    the buffer is flushed if it still lives.
#include "cupti_test_util.h"

using cupti_test::fmt;
using cupti_test::out;

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_) {                                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

#if CUPTI_API_VERSION >= 24
using Memory2Record = CUpti_ActivityMemory4;
#else
using Memory2Record = CUpti_ActivityMemory3;
#endif
#if CUPTI_API_VERSION >= 130000
using PoolRecord = CUpti_ActivityMemoryPool3;
#else
using PoolRecord = CUpti_ActivityMemoryPool2;
#endif

namespace {

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid) {
  CUpti_Activity* r = nullptr;
  while (cuptiActivityGetNextRecord(buffer, valid, &r) == CUPTI_SUCCESS) {
    switch (static_cast<int>(r->kind)) {
      case CUPTI_ACTIVITY_KIND_MEMORY2: {
        const auto* m = reinterpret_cast<const Memory2Record*>(r);
        out(fmt("REC memory2 op=%s kind=%s bytes=%llu async=%u stream=%s pool=%s poolsize=%llu threshold=%llu "
                "utilized=%llu corr=#c%u",
                m->memoryOperationType == CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION ? "alloc" : "release",
                cupti_test::mem_kind(m->memoryKind), (unsigned long long)m->bytes, m->isAsync,
                m->streamId == 0xffffffffu ? "none" : "set",
                m->memoryPoolConfig.memoryPoolType == CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL ? "local" : "none",
                (unsigned long long)m->memoryPoolConfig.pool.size,
                (unsigned long long)m->memoryPoolConfig.releaseThreshold,
                (unsigned long long)m->memoryPoolConfig.utilizedSize, m->correlationId));
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMORY_POOL: {
        const auto* p = reinterpret_cast<const PoolRecord*>(r);
        const char* op = p->memoryPoolOperationType == CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_CREATED     ? "created"
                         : p->memoryPoolOperationType == CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_DESTROYED ? "destroyed"
                                                                                                             : "trimmed";
        out(fmt("REC pool op=%s type=%s size=%llu threshold=%llu utilized=%llu minkeep=%zu corr=#c%u", op,
                p->memoryPoolType == CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL ? "local" : "other",
                (unsigned long long)p->size, (unsigned long long)p->releaseThreshold,
                (unsigned long long)p->utilizedSize, p->minBytesToKeep, p->correlationId));
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMORY: {
        const auto* m = reinterpret_cast<const CUpti_ActivityMemory*>(r);
        out(fmt("REC memory1 kind=%s bytes=%llu", cupti_test::mem_kind(m->memoryKind), (unsigned long long)m->bytes));
        break;
      }
      case CUPTI_ACTIVITY_KIND_RUNTIME: {
        const auto* a = reinterpret_cast<const CUpti_ActivityAPI*>(r);
        const char* name = nullptr;
        cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, a->cbid, &name);
        out(fmt("REC runtime %s corr=#c%u", name ? name : "?", a->correlationId));
        break;
      }
      default: break;
    }
  }
}

void section(const char* name) {
  out(std::string("# ") + name);
  cuptiActivityFlushAll(0);
}

}  // namespace

int main() {
  cuptiActivityRegisterCallbacks(cupti_test::request_buffer, buffer_completed);
  for (int kind : {CUPTI_ACTIVITY_KIND_MEMORY, CUPTI_ACTIVITY_KIND_MEMORY2, CUPTI_ACTIVITY_KIND_MEMORY_POOL,
                   CUPTI_ACTIVITY_KIND_RUNTIME})
    if (cuptiActivityEnable(static_cast<CUpti_ActivityKind>(kind)) != CUPTI_SUCCESS) {
      std::fprintf(stderr, "FAIL: cannot enable kind %d\n", kind);
      return 1;
    }
  CK(cudaSetDevice(0));
  CK(cudaFree(nullptr));
  cuptiActivityFlushAll(0);
  cupti_test::lines().clear();

  // ---- device and pinned memory ----
  float* a = nullptr;
  void* h2 = nullptr;
  CK(cudaMalloc(&a, 4096));
  CK(cudaFree(a));
  CK(cudaHostAlloc(&h2, 4096, cudaHostAllocDefault));
  CK(cudaFreeHost(h2));
  section("plain");

  // ---- the device's own pool ----
  cudaStream_t s1, s2;
  CK(cudaStreamCreate(&s1));
  CK(cudaStreamCreate(&s2));
  void *p1 = nullptr, *p2 = nullptr;
  CK(cudaMallocAsync(&p1, 1 << 20, s1));
  CK(cudaMallocAsync(&p2, 2 << 20, s1));
  CK(cudaFreeAsync(p1, s1));
  CK(cudaFreeAsync(p2, s1));
  CK(cudaStreamSynchronize(s1));
  section("default pool, one wait");

  // A pool grows in steps of 32 MiB, to hold what is handed out.
  CK(cudaMallocAsync(&p1, 33 << 20, s1));
  CK(cudaMallocAsync(&p2, 100 << 20, s1));
  CK(cudaFreeAsync(p1, s1));
  CK(cudaFreeAsync(p2, s1));
  CK(cudaStreamSynchronize(s1));
  section("growing");

  // Memory freed on one stream is not complete until that stream is waited on:
  // waiting on another does not trim, and a trim before the wait gives nothing back.
  cudaMemPool_t def;
  CK(cudaDeviceGetDefaultMemPool(&def, 0));
  unsigned long long threshold = 64ull << 20;
  CK(cudaMemPoolSetAttribute(def, cudaMemPoolAttrReleaseThreshold, &threshold));
  CK(cudaMallocAsync(&p1, 100 << 20, s1));
  CK(cudaFreeAsync(p1, s1));
  CK(cudaStreamSynchronize(s2));
  CK(cudaMemPoolTrimTo(def, 0));
  section("wait on another stream");
  CK(cudaStreamSynchronize(s1));
  CK(cudaMemPoolTrimTo(def, 12345));
  CK(cudaMemPoolTrimTo(def, 0));
  CK(cudaMemPoolTrimTo(def, 0));
  section("trims");
  threshold = 0;
  CK(cudaMemPoolSetAttribute(def, cudaMemPoolAttrReleaseThreshold, &threshold));

  // ---- a pool of the program's own ----
  cudaMemPool_t pool;
  cudaMemPoolProps props;
  std::memset(&props, 0, sizeof props);
  props.allocType = cudaMemAllocationTypePinned;
  props.handleTypes = cudaMemHandleTypeNone;
  props.location.type = cudaMemLocationTypeDevice;
  props.location.id = 0;
  CK(cudaMemPoolCreate(&pool, &props));
  threshold = 1ull << 30;
  CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
  CK(cudaMallocFromPoolAsync(&p1, 1 << 20, pool, s1));
  CK(cudaFreeAsync(p1, s1));
  CK(cudaStreamSynchronize(s1));
  CK(cudaMemPoolTrimTo(pool, 0));
  CK(cudaMemPoolDestroy(pool));
  section("own pool");

  CK(cudaMalloc(&a, 8192));
  section("left allocated");
  cuptiActivityFlushAll(0);
  CK(cudaFree(a));
  CK(cudaStreamDestroy(s1));
  CK(cudaStreamDestroy(s2));
  cuptiActivityFlushAll(0);

  cupti_test::print_all();
  std::printf("# end\n");
  return 0;
}
