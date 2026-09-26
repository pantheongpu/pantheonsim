// Streams that run at once, as a card's do, checked by a program built by
// hipcc: two kernels on two streams that hand a value to each other (which
// only finishes if both run at the same time), a stream and an event that say
// their work is not done while a kernel waits on the host, one stream made to
// wait for another's event, the null stream waiting for the blocking streams
// and not for a non-blocking one, a host function in stream order, an
// asynchronous copy from memory the program changes right after, and a
// kernel's fault told at the synchronization after it.
//
// A kernel that waits does so for a bounded number of turns and reports
// running out, so a runtime that serializes streams fails here rather than
// hanging.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>

#define CHECK(x)                                                                        \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

constexpr long kTurns = 400000;   // how long a waiting kernel waits before it gives up

// Waits until *flag holds `want`; says whether it came before the turns ran out.
__device__ bool wait_for(volatile int* flag, int want) {
  for (long i = 0; i < kTurns; ++i)
    if (*flag == want) return true;
  return false;
}

// One side of the handshake: waits for `in`, then answers on `out`.
__global__ void answer(volatile int* in, volatile int* out, int* ok) {
  const bool got = wait_for(in, 1);
  *out = 1;
  *ok = got;
}
// The other: speaks first on `out`, then waits for the answer on `in`.
__global__ void speak(volatile int* out, volatile int* in, int* ok) {
  *out = 1;
  *ok = wait_for(in, 1);
}
// Waits on a flag the host sets, then writes `value`.
__global__ void after_host(volatile int* host_flag, int* where, int value, int* ok) {
  *ok = wait_for(host_flag, 1);
  *where = value;
}
__global__ void copy_value(const int* from, int* to) { *to = *from; }
__global__ void write_value(int* where, int value) { *where = value; }
__global__ void fault() { *reinterpret_cast<volatile int*>(0x10) = 1; }

void set_flag(void* p) { *static_cast<int*>(p) = 7; }

int main() {
  int* dev;
  CHECK(hipMalloc(&dev, 64 * sizeof(int)));
  CHECK(hipMemset(dev, 0, 64 * sizeof(int)));
  int* host_flag;   // pinned: the device reaches it where it is
  CHECK(hipHostMalloc(reinterpret_cast<void**>(&host_flag), sizeof(int), 0));
  int h[64];

  // 1. Two streams at once: each kernel waits for the other's word.
  hipStream_t a, b;
  CHECK(hipStreamCreate(&a));
  CHECK(hipStreamCreate(&b));
  answer<<<1, 1, 0, a>>>(dev + 0, dev + 1, dev + 2);
  speak<<<1, 1, 0, b>>>(dev + 0, dev + 1, dev + 3);
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(h, dev, 4 * sizeof(int), hipMemcpyDeviceToHost));
  std::printf("two kernels on two streams run at once %d\n", h[2] == 1 && h[3] == 1);

  // 2. A kernel waiting on the host: its stream, and an event behind it, are
  // not done until the host lets it go.
  *host_flag = 0;
  hipEvent_t after;
  CHECK(hipEventCreate(&after));
  after_host<<<1, 1, 0, a>>>(host_flag, dev + 4, 42, dev + 5);
  CHECK(hipEventRecord(after, a));
  const bool busy = hipStreamQuery(a) == hipErrorNotReady && hipEventQuery(after) == hipErrorNotReady;
  *host_flag = 1;
  CHECK(hipEventSynchronize(after));
  const bool done = hipStreamQuery(a) == hipSuccess && hipEventQuery(after) == hipSuccess;
  CHECK(hipMemcpy(h, dev + 4, 2 * sizeof(int), hipMemcpyDeviceToHost));
  std::printf("a stream is busy while its kernel waits, and done after %d\n", busy && done && h[0] == 42 && h[1]);

  // 3. One stream waits for another's event: b's copy must see a's write,
  // though a's kernel is held until after b's work was queued.
  *host_flag = 0;
  hipEvent_t written;
  CHECK(hipEventCreate(&written));
  after_host<<<1, 1, 0, a>>>(host_flag, dev + 6, 99, dev + 7);
  CHECK(hipEventRecord(written, a));
  CHECK(hipStreamWaitEvent(b, written, 0));
  copy_value<<<1, 1, 0, b>>>(dev + 6, dev + 8);
  *host_flag = 1;
  CHECK(hipStreamSynchronize(b));
  CHECK(hipMemcpy(h, dev + 6, 3 * sizeof(int), hipMemcpyDeviceToHost));
  std::printf("a stream waits for another's event %d\n", h[2] == 99 && h[1]);

  // 4. The null stream waits for the blocking streams' work; a non-blocking
  // stream does not wait for the null stream's.
  *host_flag = 0;
  hipStream_t nb;
  CHECK(hipStreamCreateWithFlags(&nb, hipStreamNonBlocking));
  after_host<<<1, 1, 0, a>>>(host_flag, dev + 9, 5, dev + 10);
  copy_value<<<1, 1, 0, 0>>>(dev + 9, dev + 11);    // behind a's kernel
  write_value<<<1, 1, 0, nb>>>(dev + 12, 6);        // behind nothing
  CHECK(hipStreamSynchronize(nb));
  const bool null_waits = hipStreamQuery(0) == hipErrorNotReady;
  *host_flag = 1;
  CHECK(hipStreamSynchronize(0));
  CHECK(hipMemcpy(h, dev + 9, 4 * sizeof(int), hipMemcpyDeviceToHost));
  std::printf("the null stream waits for the blocking streams, not a non-blocking one %d\n",
              null_waits && h[2] == 5 && h[3] == 6 && h[1]);

  // 5. A host function runs after the work before it on its stream.
  int seen = 0;
  write_value<<<1, 1, 0, a>>>(dev + 13, 11);
  CHECK(hipLaunchHostFunc(a, set_flag, &seen));
  CHECK(hipStreamSynchronize(a));
  CHECK(hipMemcpy(h, dev + 13, sizeof(int), hipMemcpyDeviceToHost));
  std::printf("a host function runs in stream order %d\n", seen == 7 && h[0] == 11);

  // 6. An asynchronous copy from ordinary host memory takes its bytes when
  // it is called: changing them after does not change what arrives.
  *host_flag = 0;
  int source = 1234;
  after_host<<<1, 1, 0, a>>>(host_flag, dev + 14, 0, dev + 15);   // holds the stream
  CHECK(hipMemcpyAsync(dev + 16, &source, sizeof(int), hipMemcpyHostToDevice, a));
  source = 5678;
  *host_flag = 1;
  CHECK(hipStreamSynchronize(a));
  CHECK(hipMemcpy(h, dev + 16, sizeof(int), hipMemcpyDeviceToHost));
  std::printf("an asynchronous copy takes its bytes when it is called %d\n", h[0] == 1234);

  // 7. A kernel's fault is told when its stream is synchronized, and not
  // before: the launch itself succeeds.
  const hipError_t launched = (fault<<<1, 1, 0, b>>>(), hipGetLastError());
  const hipError_t synced = hipStreamSynchronize(b);
  std::printf("a kernel's fault is told at the synchronization after it %d\n",
              launched == hipSuccess && synced != hipSuccess);

  CHECK(hipStreamDestroy(a));
  CHECK(hipStreamDestroy(b));
  CHECK(hipStreamDestroy(nb));
  CHECK(hipEventDestroy(after));
  CHECK(hipEventDestroy(written));
  CHECK(hipHostFree(host_flag));
  CHECK(hipFree(dev));
  return 0;
}
