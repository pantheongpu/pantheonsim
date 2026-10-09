// When CUPTI asks the program for a buffer. It does so as the first record of
// a batch is made -- as a kernel launch returns, as the first call returns
// while the runtime kind is on -- not when the batch is delivered, and not at
// all while nothing is recorded; a flush that completes the buffer is followed
// by a new request at the next record. Profilers that count the buffers they
// were handed and flush only if there are any (Kineto, under torch.profiler)
// depend on it. Run against NVIDIA's libcupti on a card it gives
// nvidia/tests/data/cupti_buffers.expected.
#include "cupti_test_util.h"

using cupti_test::fmt;
using cupti_test::out;

namespace {

int g_requested = 0, g_completed = 0;
size_t g_last_valid = 0;
uint8_t g_buffer[1 << 16] __attribute__((aligned(8)));

void CUPTIAPI request(uint8_t** buffer, size_t* size, size_t* max_records) {
  ++g_requested;
  *buffer = g_buffer;
  *size = sizeof g_buffer;
  *max_records = 0;
}
void CUPTIAPI completed(CUcontext, uint32_t, uint8_t*, size_t, size_t valid) {
  ++g_completed;
  g_last_valid = valid;
}

void say(const char* when) { out(fmt("%-34s requested %d, completed %d", when, g_requested, g_completed)); }

}  // namespace

__global__ void bump(int* p) { *p += 1; }

int main() {
  cuptiActivityRegisterCallbacks(request, completed);
  cudaFree(nullptr);
  say("context made");
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  say("kernel kind enabled");
  int* d = nullptr;
  cudaMalloc(&d, sizeof(int));
  cudaMemset(d, 0, sizeof(int));
  say("allocated, nothing to record");
  bump<<<1, 1>>>(d);
  say("kernel launched");
  cudaDeviceSynchronize();
  say("synchronized");
  cuptiActivityFlushAll(0);
  say("flushed");
  out(fmt("delivered anything: %s", g_last_valid ? "yes" : "no"));
  cuptiActivityFlushAll(0);
  say("flushed again");
  bump<<<1, 1>>>(d);
  say("second kernel launched");
  cudaDeviceSynchronize();
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  say("kind disabled, not flushed");
  cuptiActivityFlushAll(0);
  say("flushed");
  bump<<<1, 1>>>(d);
  cudaDeviceSynchronize();
  cuptiActivityFlushAll(0);
  say("launch while disabled, flushed");

  // The runtime kind: the first call to return is the first record.
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
  say("runtime kind enabled");
  cudaFree(d);
  say("one call made");
  cuptiActivityFlushAll(0);
  say("flushed");
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);
  cupti_test::print_all();
  std::printf("# end\n");
  return 0;
}
