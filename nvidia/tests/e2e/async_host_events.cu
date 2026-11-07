// Host functions and events that are not complete when the call returns (issue #487): cudaLaunchHostFunc
// enqueues and returns; an event recorded behind it, and the stream, are not ready until it has finished
// (cudaErrorNotReady from cudaEventQuery, cudaStreamQuery and cudaEventElapsedTime); a host function that waits for
// something the host does after the call does not hang the program; work issued on the stream after it runs after
// it, cudaStreamWaitEvent holds another stream back, and the legacy default stream orders against blocking streams.
//
// Every check here is what an RTX 3060 (driver 610, CUDA 13.0 runtime) does too -- the program is also built and run
// against NVIDIA's runtime on the card, where all of them hold -- so each is a fact about CUDA, not about this
// simulator. The kernel-launch case of #523 (an event recorded behind a long kernel is not ready) is async_launch_events.cu.
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

using clk = std::chrono::steady_clock;
static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  std::fflush(stdout);
  if (!ok) ++failures;
}
static void sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

// A host function that waits for the main thread to release it (with a safety net so a broken runtime fails
// instead of hanging CI).
struct Gate {
  std::atomic<bool> release{false}, entered{false}, done{false};
};
static void CUDART_CB wait_gate(void* p) {
  Gate* g = static_cast<Gate*>(p);
  g->entered = true;
  const auto t0 = clk::now();
  while (!g->release && clk::now() - t0 < std::chrono::seconds(20)) sleep_ms(1);
  g->done = true;
}
static void CUDART_CB sleep_then_set(void* p) {   // a host function that sets a pinned flag after 80 ms
  sleep_ms(80);
  *static_cast<volatile int*>(p) = 1;
}
static void CUDART_CB sleep_50(void*) { sleep_ms(50); }
static void CUDART_CB set_flag(void* p) { static_cast<std::atomic<int>*>(p)->store(1); }

__global__ void copy_flag(const volatile int* flag, int* out) { *out = *flag; }

int main() {
  cudaStream_t s1, s2;
  check(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking) == cudaSuccess, "stream s1");
  check(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking) == cudaSuccess, "stream s2");
  cudaEvent_t a, b, c;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  cudaEventCreate(&c);

  // 1. The call returns while the function still waits; queries say not ready; the last error stays clean.
  {
    Gate g;
    cudaEventRecord(a, s1);
    const auto t0 = clk::now();
    check(cudaLaunchHostFunc(s1, wait_gate, &g) == cudaSuccess, "cudaLaunchHostFunc succeeds");
    const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    check(ms < 1000 && !g.done, "cudaLaunchHostFunc returned before the host function finished");
    cudaEventRecord(b, s1);
    check(cudaEventQuery(b) == cudaErrorNotReady, "cudaEventQuery behind a host function: not ready");
    check(cudaStreamQuery(s1) == cudaErrorNotReady, "cudaStreamQuery behind a host function: not ready");
    float t = -1;
    check(cudaEventElapsedTime(&t, a, b) == cudaErrorNotReady, "cudaEventElapsedTime on a pending event: not ready");
    check(cudaGetLastError() == cudaSuccess, "not ready is not left as the last error");
    g.release = true;   // the main thread lets it finish, after the call returned
    check(cudaEventSynchronize(b) == cudaSuccess, "cudaEventSynchronize returns once it finished");
    check(g.done, "the host function ran before the event completed");
    check(cudaEventQuery(b) == cudaSuccess && cudaStreamQuery(s1) == cudaSuccess, "complete after it finished");
    check(cudaEventElapsedTime(&t, a, b) == cudaSuccess && t >= 0, "cudaEventElapsedTime after it finished");
  }

  // 2. Elapsed time spans the host function.
  {
    cudaEventRecord(a, s1);
    cudaLaunchHostFunc(s1, sleep_50, nullptr);
    cudaEventRecord(b, s1);
    cudaEventSynchronize(b);
    float t = -1;
    check(cudaEventElapsedTime(&t, a, b) == cudaSuccess && t >= 40.0f, "elapsed time covers a 50 ms host function");
  }

  // 3. Stream and device synchronize wait for a host function.
  {
    Gate g;
    cudaLaunchHostFunc(s1, wait_gate, &g);
    std::thread rel([&] { while (!g.entered) sleep_ms(1); sleep_ms(50); g.release = true; });
    check(cudaStreamSynchronize(s1) == cudaSuccess && g.done, "cudaStreamSynchronize waits for the host function");
    rel.join();
    Gate g2;
    cudaLaunchHostFunc(s2, wait_gate, &g2);
    std::thread rel2([&] { while (!g2.entered) sleep_ms(1); sleep_ms(50); g2.release = true; });
    check(cudaDeviceSynchronize() == cudaSuccess && g2.done, "cudaDeviceSynchronize waits for the host function");
    rel2.join();
  }

  // 4. Work issued after a host function runs after it: a kernel reads a pinned flag the host function sets.
  {
    volatile int* flag = nullptr;
    int* dout = nullptr;
    cudaMallocHost((void**)&flag, sizeof(int));
    cudaMalloc((void**)&dout, sizeof(int));
    *flag = 0;
    cudaMemset(dout, 0, sizeof(int));
    cudaLaunchHostFunc(s1, sleep_then_set, (void*)flag);
    copy_flag<<<1, 1, 0, s1>>>(flag, dout);
    cudaStreamSynchronize(s1);
    int got = -1;
    cudaMemcpy(&got, dout, sizeof got, cudaMemcpyDeviceToHost);
    check(got == 1, "a kernel launched after a host function sees what it wrote");
    // Host functions on one stream run in order.
    static std::atomic<int> seq{0};
    struct Step { std::atomic<int>* s; int expect; std::atomic<int>* bad; };
    static std::atomic<int> bad{0};
    static Step steps[3] = {{&seq, 0, &bad}, {&seq, 1, &bad}, {&seq, 2, &bad}};
    for (auto& st : steps)
      cudaLaunchHostFunc(s1, [](void* p) {
        Step* st = static_cast<Step*>(p);
        if (st->s->fetch_add(1) != st->expect) st->bad->fetch_add(1);
      }, &st);
    cudaStreamSynchronize(s1);
    check(seq == 3 && bad == 0, "host functions on a stream run in order");
    cudaFree(dout);
    cudaFreeHost((void*)flag);
  }

  // 5. cudaStreamWaitEvent on a pending event holds the other stream back.
  {
    Gate g;
    cudaLaunchHostFunc(s1, wait_gate, &g);
    cudaEventRecord(b, s1);
    cudaStreamWaitEvent(s2, b, 0);
    cudaEventRecord(c, s2);
    check(cudaEventQuery(c) == cudaErrorNotReady, "an event behind a wait on a pending event: not ready");
    check(cudaStreamQuery(s2) == cudaErrorNotReady, "a stream waiting on a pending event: not ready");
    g.release = true;
    check(cudaStreamSynchronize(s2) == cudaSuccess && cudaEventQuery(c) == cudaSuccess,
          "the waiting stream completes once the event does");
    cudaStreamSynchronize(s1);
  }

  // 6. The legacy default stream orders against a blocking stream.
  {
    cudaStream_t blk;
    cudaStreamCreate(&blk);
    Gate g;
    cudaLaunchHostFunc(blk, wait_gate, &g);
    cudaEventRecord(c, 0);   // legacy default stream: behind the blocking stream's work
    check(cudaEventQuery(c) == cudaErrorNotReady, "an event on the legacy stream waits for a blocking stream");
    check(cudaStreamQuery(0) == cudaErrorNotReady, "the legacy stream is not ready while a blocking stream is busy");
    g.release = true;
    check(cudaStreamSynchronize(0) == cudaSuccess && cudaEventQuery(c) == cudaSuccess, "and completes after it");
    cudaStreamDestroy(blk);
  }

  // 7. A host function that waits for what the host does after enqueueing it does not hang (alpaka's shape).
  {
    std::atomic<int> flag{0};
    Gate g;
    cudaLaunchHostFunc(s1, wait_gate, &g);
    cudaLaunchHostFunc(s1, set_flag, &flag);
    cudaEventRecord(b, s1);
    check(cudaEventQuery(b) == cudaErrorNotReady && flag == 0, "nothing after a waiting host function ran yet");
    g.release = true;
    cudaEventSynchronize(b);
    check(flag == 1 && cudaEventQuery(b) == cudaSuccess, "the queue drains once the host releases it");
  }

  // 8. Nothing pending: everything behaves as before.
  {
    cudaEventRecord(a, s2);
    cudaStreamSynchronize(s2);
    check(cudaEventQuery(a) == cudaSuccess && cudaStreamQuery(s2) == cudaSuccess, "idle stream, recorded event: ready");
  }

  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
  std::printf("%s\n", failures ? "FAILED" : "all passed");
  return failures ? 1 : 0;
}
