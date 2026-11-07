// A kernel launch returns before the kernel has finished (issue #523, the kernel-launch half of #487): an event recorded
// behind a long kernel, and the stream, are not ready (cudaErrorNotReady from cudaEventQuery, cudaStreamQuery and
// cudaEventElapsedTime) until it has run; everything that depends on the kernel still orders after it; a kernel's fault
// is the context's, found at the next call that touches it, and a launch the device refuses for its configuration is
// refused in the launch.
//
// Every check here is what an RTX 3060 (driver 610, CUDA 13.0 runtime) does too -- the program is also built and run
// against NVIDIA's runtime on the card, where all of them hold (the first section is the issue's own repro: query 600,
// elapsed 600, then 0 after the synchronize) -- so each is a fact about CUDA, not about this simulator. Copies that are
// checked against a kernel on a non-blocking stream are on that stream (a copy on the legacy default stream is not
// ordered with it on a card).
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  std::fflush(stdout);
  if (!ok) ++failures;
}

// A loop long enough to still be running when the host asks (tens of milliseconds on a card), then a result.
__global__ void spin(volatile unsigned* out, int n, unsigned seed) {
  unsigned acc = seed;
  for (int i = 0; i < n; i++) acc = acc * 1664525u + 1013904223u;
  out[0] = acc;
}
__global__ void add_one(unsigned* p) { p[0] += 1; }
__global__ void poke_null(int* p) { *p = 7; }

static unsigned host_spin(int n, unsigned seed) {
  unsigned acc = seed;
  for (int i = 0; i < n; i++) acc = acc * 1664525u + 1013904223u;
  return acc;
}

int main() {
  const int kIters = 20000000;
  unsigned* d = nullptr;
  cudaMalloc(&d, 4);
  unsigned* pinned = nullptr;
  cudaMallocHost(&pinned, 4);
  cudaStream_t s, s2;
  cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
  cudaEvent_t e1, e2, e3;
  cudaEventCreate(&e1);
  cudaEventCreate(&e2);
  cudaEventCreate(&e3);

  // 1. The issue's repro: an event recorded behind a long kernel is not ready, and the stream is not either.
  cudaEventRecord(e1, s);
  spin<<<1, 1, 0, s>>>(d, kIters, 1u);
  cudaEventRecord(e2, s);
  const cudaError_t q = cudaEventQuery(e2);
  float ms = -1;
  const cudaError_t t = cudaEventElapsedTime(&ms, e1, e2);
  const cudaError_t sq = cudaStreamQuery(s);
  std::printf("cudaEventQuery(e2) = %d, cudaEventElapsedTime = %d, cudaStreamQuery = %d\n", (int)q, (int)t, (int)sq);
  check(q == cudaErrorNotReady, "cudaEventQuery behind a running kernel is cudaErrorNotReady");
  check(t == cudaErrorNotReady, "cudaEventElapsedTime behind a running kernel is cudaErrorNotReady");
  check(sq == cudaErrorNotReady, "cudaStreamQuery while the kernel runs is cudaErrorNotReady");
  check(cudaGetLastError() == cudaSuccess, "a query that says not ready leaves the last error clean");
  check(cudaStreamSynchronize(s) == cudaSuccess, "cudaStreamSynchronize");
  check(cudaEventQuery(e2) == cudaSuccess, "after the synchronize the event is complete");
  check(cudaStreamQuery(s) == cudaSuccess, "after the synchronize the stream is idle");
  check(cudaEventElapsedTime(&ms, e1, e2) == cudaSuccess && ms > 0.0f, "after the synchronize the elapsed time is positive");

  // 2. A copy on the same stream orders after the kernel.
  *pinned = 0;
  cudaEventRecord(e1, s);
  spin<<<1, 1, 0, s>>>(d, kIters, 2u);
  cudaMemcpyAsync(pinned, d, 4, cudaMemcpyDeviceToHost, s);
  check(cudaStreamSynchronize(s) == cudaSuccess, "stream synchronize after kernel and copy");
  check(*pinned == host_spin(kIters, 2u), "a copy on the stream sees the kernel's result");

  // 3. Kernels on one stream run in order, one after the other.
  spin<<<1, 1, 0, s>>>(d, kIters / 4, 3u);
  add_one<<<1, 1, 0, s>>>(d);
  add_one<<<1, 1, 0, s>>>(d);
  cudaMemcpyAsync(pinned, d, 4, cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);
  check(*pinned == host_spin(kIters / 4, 3u) + 2u, "three kernels on a stream ran in order");

  // 4. An event on the first stream and a wait for it on the second: the second stream's kernel sees the first's result.
  cudaEventRecord(e3, s);   // an idle stream: complete at once
  spin<<<1, 1, 0, s>>>(d, kIters / 2, 4u);
  cudaEventRecord(e3, s);
  check(cudaEventQuery(e3) == cudaErrorNotReady, "an event behind a kernel is not ready");
  cudaStreamWaitEvent(s2, e3, 0);
  add_one<<<1, 1, 0, s2>>>(d);
  cudaMemcpyAsync(pinned, d, 4, cudaMemcpyDeviceToHost, s2);
  check(cudaStreamSynchronize(s2) == cudaSuccess, "synchronize of the waiting stream");
  check(*pinned == host_spin(kIters / 2, 4u) + 1u, "a stream that waited for the event sees the kernel's result");
  check(cudaEventQuery(e3) == cudaSuccess, "the event is complete once the waiting stream is");

  // 5. cudaFree after a launch with nothing between: it waits for the kernel (the kernel's store, at its end, is to
  //    memory that is still there).
  unsigned* f = nullptr;
  cudaMalloc(&f, 4);
  spin<<<1, 1, 0, s>>>(f, kIters / 2, 5u);
  check(cudaFree(f) == cudaSuccess, "cudaFree while a kernel is still queued on a stream");
  check(cudaDeviceSynchronize() == cudaSuccess, "the kernel was not run against freed memory");
  check(cudaGetLastError() == cudaSuccess, "no error left behind by it");

  // 6. cudaDeviceSynchronize and cudaEventSynchronize wait for the kernel.
  spin<<<1, 1, 0, s>>>(d, kIters / 2, 6u);
  cudaEventRecord(e2, s);
  check(cudaEventSynchronize(e2) == cudaSuccess, "cudaEventSynchronize");
  check(cudaEventQuery(e2) == cudaSuccess, "the event is complete after cudaEventSynchronize");
  spin<<<1, 1, 0, s2>>>(d, kIters / 4, 7u);
  check(cudaDeviceSynchronize() == cudaSuccess, "cudaDeviceSynchronize");
  check(cudaStreamQuery(s2) == cudaSuccess, "idle after cudaDeviceSynchronize");

  // 7. A launch the device refuses for its configuration is refused in the launch, and leaves nothing pending.
  spin<<<1, 2048, 0, s>>>(d, 10, 1u);
  check(cudaGetLastError() == cudaErrorInvalidValue, "a block of 2048 threads is cudaErrorInvalidValue from the launch");
  check(cudaStreamQuery(s) == cudaSuccess, "a refused launch queues nothing");

  // 7b. Several host threads, each with a stream of its own, launch and wait at once (launch, query, copy, synchronize).
  {
    std::atomic<int> good{0};
    std::vector<std::thread> pool;
    for (int i = 0; i < 4; ++i)
      pool.emplace_back([&good, i] {
        cudaStream_t st;
        cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
        unsigned* buf = nullptr;
        unsigned* host = nullptr;
        cudaMalloc(&buf, 4);
        cudaMallocHost(&host, 4);
        bool ok = true;
        for (int round = 0; round < 3; ++round) {
          const unsigned seed = 100u + i * 10u + round;
          spin<<<1, 1, 0, st>>>(buf, 1000000, seed);
          add_one<<<1, 1, 0, st>>>(buf);
          while (cudaStreamQuery(st) == cudaErrorNotReady) std::this_thread::yield();
          cudaMemcpyAsync(host, buf, 4, cudaMemcpyDeviceToHost, st);
          ok = ok && cudaStreamSynchronize(st) == cudaSuccess && *host == host_spin(1000000, seed) + 1u;
        }
        cudaFree(buf);
        cudaFreeHost(host);
        cudaStreamDestroy(st);
        if (ok) ++good;
      });
    for (auto& t : pool) t.join();
    check(good == 4, "four host threads, each launching on a stream of its own, all get the right results");
  }

  // 8. A kernel's fault: the launch succeeds, the next call that touches the context reports it, and it is sticky.
  poke_null<<<1, 1, 0, s>>>(nullptr);
  check(cudaGetLastError() == cudaSuccess, "right after the launch of a faulting kernel the last error is clean");
  const cudaError_t sync = cudaStreamSynchronize(s);
  std::printf("cudaStreamSynchronize after the fault = %d\n", (int)sync);
  check(sync == cudaErrorIllegalAddress, "the synchronize reports the illegal address");
  void* again = nullptr;
  check(cudaMalloc(&again, 16) == cudaErrorIllegalAddress, "the context stays dead: cudaMalloc reports it too");
  check(cudaDeviceSynchronize() == cudaErrorIllegalAddress, "and so does cudaDeviceSynchronize");

  std::printf(failures ? "FAILED: %d check(s)\n" : "all checks passed\n", failures);
  return failures ? 1 : 0;
}
