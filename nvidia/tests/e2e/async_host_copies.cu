// Copies, fills and kernels issued behind a host function that has not finished (issue #634). A host function that
// waits for something the program does after it has queued more work on the stream must not hang the program: the
// later cudaMemcpyAsync, cudaMemsetAsync, 2D and 3D copies, symbol copies, cudaFreeAsync and kernel launches return
// at once, as they do on a GPU, and run after the host function, in stream order. So:
//   - each call returns while the host function is still waiting (the deadlock of #634: the simulator used to run
//     the copy inside the call after waiting for the host function);
//   - the work sees what the host function wrote (a host function fills pinned memory, a copy reads it), and the
//     host function does not see what the work writes (it reads the copy's destination before the copy runs);
//   - cudaMemcpy, cudaStreamSynchronize, cudaDeviceSynchronize and cudaEventSynchronize wait for the queued work;
//   - the legacy default stream and blocking streams order against each other, and a non-blocking stream does not
//     wait for either.
// The first case is the program of the issue, whose output was taken on an RTX 3060 (CUDA 13.0); the others follow
// CUDA's documented stream semantics for page-locked memory. A broken runtime fails here instead of hanging: every
// host function gives up waiting after 20 s.
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

using clk = std::chrono::steady_clock;
static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  std::fflush(stdout);
  if (!ok) ++failures;
}
static void sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

// A host function that waits for the main thread to release it, then (optionally) reads one int and writes another:
// `seen` is what `*watch` held when it ran, after which it stores `value` to `*where` (both pinned memory).
struct Gate {
  std::atomic<bool> release{false}, entered{false}, done{false};
  const volatile int* watch = nullptr;
  int seen = -12345;
  volatile int* where = nullptr;
  int value = 0;
  volatile unsigned char* fill = nullptr;   // or fills `fill_bytes` bytes with a pattern
  size_t fill_bytes = 0;
};
static void CUDART_CB gate_fn(void* p) {
  Gate* g = static_cast<Gate*>(p);
  g->entered = true;
  const auto t0 = clk::now();
  while (!g->release && clk::now() - t0 < std::chrono::seconds(20)) std::this_thread::yield();
  if (g->watch) g->seen = *g->watch;
  if (g->where) *g->where = g->value;
  for (size_t i = 0; i < g->fill_bytes; ++i) g->fill[i] = static_cast<unsigned char>(i * 7 + 3);
  g->done = true;
}
// Releases the gate from another thread after `ms`, for the calls that are meant to wait for it.
static std::thread release_later(Gate& g, int ms) {
  return std::thread([&g, ms] {
    while (!g.entered) sleep_ms(1);
    sleep_ms(ms);
    g.release = true;
  });
}

__global__ void copy_flag(const volatile int* flag, int* out) { *out = *flag + 1; }
__device__ int g_sym;

int main() {
  cudaStream_t blk, nb;
  check(cudaStreamCreate(&blk) == cudaSuccess, "blocking stream");
  check(cudaStreamCreateWithFlags(&nb, cudaStreamNonBlocking) == cudaSuccess, "non-blocking stream");
  int* d = nullptr;
  int* d2 = nullptr;
  cudaMalloc((void**)&d, 64 * sizeof(int));
  cudaMalloc((void**)&d2, 64 * sizeof(int));
  int* h = nullptr;     // pinned
  int* hin = nullptr;   // pinned
  cudaMallocHost((void**)&h, 64 * sizeof(int));
  cudaMallocHost((void**)&hin, 64 * sizeof(int));
  const int five = 5;
  cudaEvent_t e;
  cudaEventCreateWithFlags(&e, cudaEventDisableTiming);

  // 1. The issue's program: a device-to-pinned copy behind a host function that waits for the host returns at once,
  //    the event behind it is not ready, and once the host releases the function the copy has run.
  {
    cudaMemcpy(d, &five, sizeof(int), cudaMemcpyHostToDevice);
    *h = 0;
    Gate g;
    g.watch = h;
    cudaLaunchHostFunc(blk, gate_fn, &g);
    const cudaError_t rc = cudaMemcpyAsync(h, d, sizeof(int), cudaMemcpyDeviceToHost, blk);
    check(rc == cudaSuccess && !g.done, "cudaMemcpyAsync behind a waiting host function returns at once");
    cudaEventRecord(e, blk);
    check(cudaEventQuery(e) == cudaErrorNotReady, "the event behind the copy is not ready");
    check(cudaStreamQuery(blk) == cudaErrorNotReady, "the stream is not ready");
    g.release = true;
    check(cudaEventSynchronize(e) == cudaSuccess && *h == 5, "after the release the copy has run (value 5)");
    check(g.seen == 0, "the copy did not run before the host function");
    check(cudaGetLastError() == cudaSuccess, "no error left behind");
  }

  // 2. Order: the host function writes pinned memory; a host-to-device copy of it and a device-to-host copy back,
  //    queued after it, carry its value; it does not see the second copy's result.
  {
    *hin = 0;
    h[1] = -1;
    Gate g;
    g.where = hin;
    g.value = 42;
    g.watch = h + 1;
    cudaLaunchHostFunc(blk, gate_fn, &g);
    const cudaError_t a = cudaMemcpyAsync(d2, hin, sizeof(int), cudaMemcpyHostToDevice, blk);
    const cudaError_t b = cudaMemcpyAsync(h + 1, d2, sizeof(int), cudaMemcpyDeviceToHost, blk);
    check(a == cudaSuccess && b == cudaSuccess && !g.done, "H2D and D2H behind a waiting host function return at once");
    g.release = true;
    check(cudaStreamSynchronize(blk) == cudaSuccess, "cudaStreamSynchronize");
    check(h[1] == 42, "the H2D copy read what the host function wrote (42)");
    check(g.seen == -1, "the host function ran before the D2H copy wrote its destination");
  }

  // 3. Fills, on a non-blocking stream: device memory and pinned memory, after the host function.
  {
    std::memset(h, 0, 16);
    Gate g;
    g.watch = h + 2;
    cudaLaunchHostFunc(nb, gate_fn, &g);
    const cudaError_t a = cudaMemsetAsync(d, 0xab, 4 * sizeof(int), nb);
    const cudaError_t b = cudaMemsetAsync(h + 2, 0x11, sizeof(int), nb);
    const cudaError_t c = cudaMemcpyAsync(h, d, sizeof(int), cudaMemcpyDeviceToHost, nb);
    check(a == cudaSuccess && b == cudaSuccess && c == cudaSuccess && !g.done,
          "cudaMemsetAsync behind a waiting host function returns at once");
    g.release = true;
    cudaStreamSynchronize(nb);
    check(static_cast<unsigned>(h[0]) == 0xababababu, "the device fill ran, and the copy after it saw it");
    check(static_cast<unsigned>(h[2]) == 0x11111111u, "the pinned fill ran");
    check(g.seen == 0, "the host function ran before the pinned fill");
  }

  // 4. 2D and 3D copies: the host function fills a pinned rectangle, a 2D copy takes it to pitched device memory and
  //    a 3D copy brings it back into another pinned buffer.
  {
    const size_t w = 12, rows = 4;
    size_t pitch = 0;
    void* dp = nullptr;
    cudaMallocPitch(&dp, &pitch, w, rows);
    unsigned char* src = nullptr;
    unsigned char* dst = nullptr;
    cudaMallocHost((void**)&src, w * rows);
    cudaMallocHost((void**)&dst, w * rows);
    std::memset(src, 0, w * rows);
    std::memset(dst, 0, w * rows);
    Gate g;
    g.fill = src;
    g.fill_bytes = w * rows;
    cudaLaunchHostFunc(blk, gate_fn, &g);
    const cudaError_t a = cudaMemcpy2DAsync(dp, pitch, src, w, w, rows, cudaMemcpyHostToDevice, blk);
    cudaMemcpy3DParms p{};
    p.srcPtr = make_cudaPitchedPtr(dp, pitch, w, rows);
    p.dstPtr = make_cudaPitchedPtr(dst, w, w, rows);
    p.extent = make_cudaExtent(w, rows, 1);
    p.kind = cudaMemcpyDeviceToHost;
    const cudaError_t b = cudaMemcpy3DAsync(&p, blk);
    check(a == cudaSuccess && b == cudaSuccess && !g.done, "cudaMemcpy2DAsync and cudaMemcpy3DAsync return at once");
    g.release = true;
    cudaStreamSynchronize(blk);
    bool same = true;
    for (size_t i = 0; i < w * rows; ++i) same = same && dst[i] == static_cast<unsigned char>(i * 7 + 3);
    check(same, "the 2D and 3D copies carried what the host function wrote");
    cudaFreeHost(src);
    cudaFreeHost(dst);
    cudaFree(dp);
  }

  // 5. A kernel launched behind the host function on a blocking stream reads what it wrote, and the copy after the
  //    kernel sees the kernel's result.
  {
    *hin = 0;
    Gate g;
    g.where = hin;
    g.value = 9;
    cudaLaunchHostFunc(blk, gate_fn, &g);
    copy_flag<<<1, 1, 0, blk>>>(hin, d);
    const cudaError_t launched = cudaGetLastError();
    h[3] = 0;
    cudaMemcpyAsync(h + 3, d, sizeof(int), cudaMemcpyDeviceToHost, blk);
    check(launched == cudaSuccess && !g.done, "a kernel launch behind a waiting host function returns at once");
    g.release = true;
    cudaStreamSynchronize(blk);
    check(h[3] == 10, "the kernel read what the host function wrote, and the copy read the kernel's result");
  }

  // 6. Symbol copies.
  {
    *hin = 0;
    h[4] = 0;
    Gate g;
    g.where = hin;
    g.value = 13;
    cudaLaunchHostFunc(blk, gate_fn, &g);
    const cudaError_t a = cudaMemcpyToSymbolAsync(g_sym, hin, sizeof(int), 0, cudaMemcpyHostToDevice, blk);
    const cudaError_t b = cudaMemcpyFromSymbolAsync(h + 4, g_sym, sizeof(int), 0, cudaMemcpyDeviceToHost, blk);
    check(a == cudaSuccess && b == cudaSuccess && !g.done, "symbol copies behind a waiting host function return at once");
    g.release = true;
    cudaStreamSynchronize(blk);
    check(h[4] == 13, "the symbol copies carried what the host function wrote");
  }

  // 7. Stream-ordered free: memory a queued copy still reads is freed after the copy.
  {
    int* tmp = nullptr;
    cudaMallocAsync((void**)&tmp, sizeof(int), blk);
    const int seven = 7;
    cudaMemcpyAsync(tmp, &seven, sizeof(int), cudaMemcpyHostToDevice, blk);   // pageable, nothing pending: done now
    cudaStreamSynchronize(blk);
    h[5] = 0;
    Gate g;
    cudaLaunchHostFunc(blk, gate_fn, &g);
    const cudaError_t a = cudaMemcpyAsync(h + 5, tmp, sizeof(int), cudaMemcpyDeviceToHost, blk);
    const cudaError_t b = cudaFreeAsync(tmp, blk);
    check(a == cudaSuccess && b == cudaSuccess && !g.done, "cudaFreeAsync behind a waiting host function returns at once");
    g.release = true;
    check(cudaStreamSynchronize(blk) == cudaSuccess && h[5] == 7, "the copy before the free read the memory");
  }

  // 8. The synchronous calls wait for the queued work.
  {
    cudaMemcpy(d, &five, sizeof(int), cudaMemcpyHostToDevice);
    {
      h[6] = 0;
      Gate g;
      cudaLaunchHostFunc(blk, gate_fn, &g);
      cudaMemcpyAsync(h + 6, d, sizeof(int), cudaMemcpyDeviceToHost, blk);
      std::thread rel = release_later(g, 50);
      int sink = 0;
      cudaMemcpy(d2 + 8, &sink, sizeof(int), cudaMemcpyHostToDevice);   // legacy default stream: after blk's work
      check(g.done && h[6] == 5, "cudaMemcpy waits for the copy queued on a blocking stream");
      rel.join();
    }
    {
      h[6] = 0;
      Gate g;
      cudaLaunchHostFunc(blk, gate_fn, &g);
      cudaMemcpyAsync(h + 6, d, sizeof(int), cudaMemcpyDeviceToHost, blk);
      std::thread rel = release_later(g, 50);
      check(cudaStreamSynchronize(blk) == cudaSuccess && h[6] == 5, "cudaStreamSynchronize waits for the queued copy");
      rel.join();
    }
    {
      h[6] = 0;
      Gate g;
      cudaLaunchHostFunc(nb, gate_fn, &g);
      cudaMemcpyAsync(h + 6, d, sizeof(int), cudaMemcpyDeviceToHost, nb);
      std::thread rel = release_later(g, 50);
      check(cudaDeviceSynchronize() == cudaSuccess && h[6] == 5, "cudaDeviceSynchronize waits for the queued copy");
      rel.join();
    }
    {
      h[6] = 0;
      Gate g;
      cudaLaunchHostFunc(nb, gate_fn, &g);
      cudaMemcpyAsync(h + 6, d, sizeof(int), cudaMemcpyDeviceToHost, nb);
      cudaEventRecord(e, nb);
      std::thread rel = release_later(g, 50);
      check(cudaEventSynchronize(e) == cudaSuccess && h[6] == 5, "cudaEventSynchronize waits for the queued copy");
      rel.join();
    }
    {
      // A pageable destination: the copy is done when the call returns, as on a GPU, so it waits.
      int pageable = 0;
      Gate g;
      cudaLaunchHostFunc(blk, gate_fn, &g);
      std::thread rel = release_later(g, 50);
      cudaMemcpyAsync(&pageable, d, sizeof(int), cudaMemcpyDeviceToHost, blk);
      check(g.done && pageable == 5, "a copy to pageable memory is done when the call returns");
      rel.join();
      cudaStreamSynchronize(blk);
    }
  }

  // 9. The legacy default stream.
  {
    cudaMemcpy(d, &five, sizeof(int), cudaMemcpyHostToDevice);
    // A host function on the legacy stream holds a blocking stream's copy back, not a non-blocking stream's.
    h[7] = 0;
    h[8] = 0;
    Gate g;
    cudaLaunchHostFunc(0, gate_fn, &g);
    const cudaError_t a = cudaMemcpyAsync(h + 7, d, sizeof(int), cudaMemcpyDeviceToHost, blk);
    check(a == cudaSuccess && !g.done, "a blocking stream's copy behind the legacy stream's host function returns at once");
    cudaMemcpyAsync(h + 8, d, sizeof(int), cudaMemcpyDeviceToHost, nb);
    check(cudaStreamSynchronize(nb) == cudaSuccess && h[8] == 5 && !g.done,
          "a non-blocking stream's copy does not wait for the legacy stream");
    check(cudaStreamQuery(blk) == cudaErrorNotReady && h[7] == 0, "the blocking stream's copy has not run yet");
    g.release = true;
    check(cudaStreamSynchronize(blk) == cudaSuccess && h[7] == 5, "and runs after the legacy stream's host function");

    // A copy on the legacy stream behind a blocking stream's host function: queued too, and the legacy stream waits.
    h[9] = 0;
    Gate g2;
    g2.watch = h + 9;
    cudaLaunchHostFunc(blk, gate_fn, &g2);
    const cudaError_t b = cudaMemcpyAsync(h + 9, d, sizeof(int), cudaMemcpyDeviceToHost, 0);
    check(b == cudaSuccess && !g2.done, "a legacy-stream copy behind a blocking stream's host function returns at once");
    check(cudaStreamQuery(0) == cudaErrorNotReady, "the legacy stream is not ready");
    g2.release = true;
    check(cudaStreamSynchronize(0) == cudaSuccess && h[9] == 5 && g2.seen == 0,
          "the legacy-stream copy ran after the blocking stream's host function");
  }

  // 10. Nothing pending: copies run in the call, as before.
  {
    h[10] = 0;
    cudaMemcpyAsync(h + 10, d, sizeof(int), cudaMemcpyDeviceToHost, nb);
    cudaStreamSynchronize(nb);
    check(h[10] == 5, "an idle stream's copy");
  }

  cudaEventDestroy(e);
  cudaFreeHost(h);
  cudaFreeHost(hin);
  cudaFree(d);
  cudaFree(d2);
  cudaStreamDestroy(blk);
  cudaStreamDestroy(nb);
  std::printf("%s\n", failures ? "FAILED" : "all passed");
  return failures ? 1 : 0;
}
