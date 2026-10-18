// ROCTx's profiler control and naming: roctxProfilerPause and roctxProfilerResume around HIP calls and a
// mark (what the profiler does with them is the comparison), a name given to a stream, and a range that is
// opened on one thread and has a mark and a HIP call made inside it on another.
#include <rocprofiler-sdk-roctx/roctx.h>

#include <thread>

#include "common.h"

int main() {
  float *a, *b, *c;
  hipStream_t stream;
  roctxMarkA("before");
  CHECK(hipMalloc(&a, 1024));
  roctxProfilerPause(0);
  CHECK(hipMalloc(&b, 1024));
  roctxMarkA("paused");
  roctxProfilerResume(0);
  CHECK(hipMalloc(&c, 1024));
  CHECK(hipStreamCreate(&stream));
  roctxNameHipStream("work", stream);
  roctxRangePushA("outer");
  roctxRangePushA("inner");
  CHECK(hipMemsetAsync(a, 0, 1024, stream));
  roctxRangePop();
  CHECK(hipStreamSynchronize(stream));
  roctxRangePop();
  roctx_range_id_t id = roctxRangeStartA("elsewhere");
  std::thread t([&] {
    roctxMarkA("worker");
    CHECK(hipDeviceSynchronize());
    roctxRangePushA("worker range");
    roctxRangePop();
  });
  t.join();
  roctxRangeStop(id);
  CHECK(hipStreamDestroy(stream));
  CHECK(hipFree(a));
  CHECK(hipFree(b));
  CHECK(hipFree(c));
  roctxMarkA("after");
  std::printf("markers_control wrong 0 of 0\n");
  return 0;
}
