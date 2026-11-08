// The driver API's host functions (issue #487, the cuLaunchHostFunc half; async_host_events.cu is the runtime's): cuLaunchHostFunc
// enqueues and returns; an event recorded behind it and the stream are not ready until it has finished (CUDA_ERROR_NOT_READY from
// cuEventQuery, cuStreamQuery and cuEventElapsedTime); a host function that waits for something the host does after the call
// does not hang the program; kernels, copies and stream callbacks issued on the stream afterwards run after it,
// cuStreamWaitEvent holds another stream back, and the legacy default stream orders against blocking streams only.
//
// Every check is what an RTX 3060 (driver 610, CUDA 13.0) does too -- the program is built with nvcc -lcuda and also run against
// NVIDIA's driver on the card, where all of them hold -- so each is a fact about CUDA, not about this simulator. Pure driver API,
// kernels as PTX. Argument: a case name (all by default), so a mutation can be shown to fail one case at a time.
#include <cuda.h>

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
#define OK(call) check((call) == CUDA_SUCCESS, #call " succeeds")
static void sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
static double since_ms(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

// A host function that waits for the main thread to release it (with a safety net so a broken driver fails instead of hanging CI).
struct Gate {
  std::atomic<bool> release{false}, entered{false}, done{false};
};
static void wait_gate(void* p) {
  Gate* g = static_cast<Gate*>(p);
  g->entered = true;
  const auto t0 = clk::now();
  while (!g->release && clk::now() - t0 < std::chrono::seconds(20)) sleep_ms(1);
  g->done = true;
}
static void sleep_then_set(void* p) {   // sets a pinned flag after 80 ms
  sleep_ms(80);
  *static_cast<volatile int*>(p) = 1;
}
static void sleep_50(void*) { sleep_ms(50); }
static void set_flag(void* p) { static_cast<std::atomic<int>*>(p)->store(1); }

static std::atomic<int> g_seq{0};
static std::atomic<int> g_host_at{-1}, g_callback_at{-1};
static void note_host(void*) { sleep_ms(50); g_host_at = g_seq.fetch_add(1); }
static void note_callback(CUstream, CUresult, void*) { g_callback_at = g_seq.fetch_add(1); }

static const char* kPtx = R"(
.version 7.0
.target sm_75
.address_size 64
.visible .entry copy_flag(.param .u64 pf, .param .u64 po)
{
  .reg .b64 %rd<3>;
  .reg .b32 %r<2>;
  ld.param.u64 %rd1, [pf];
  ld.param.u64 %rd2, [po];
  ld.volatile.u32 %r1, [%rd1];
  st.u32 [%rd2], %r1;
  ret;
}
)";

static const CUresult kNotReady = CUDA_ERROR_NOT_READY;
static CUstream s1, s2, blk;
static CUevent a, b, c;
static CUfunction copy_flag;
static CUdeviceptr dout;
static volatile int* flag;

static void reset_out() {
  int zero = 0;
  cuMemcpyHtoD(dout, &zero, sizeof zero);
}
static int read_out() {
  int got = -1;
  cuMemcpyDtoH(&got, dout, sizeof got);
  return got;
}

// 1. The call returns while the function still waits; queries say not ready.
static void c_not_ready() {
  Gate g;
  cuEventRecord(a, s1);
  const auto t0 = clk::now();
  check(cuLaunchHostFunc(s1, wait_gate, &g) == CUDA_SUCCESS, "cuLaunchHostFunc succeeds");
  check(since_ms(t0) < 1000 && !g.done, "cuLaunchHostFunc returned before the host function finished");
  cuEventRecord(b, s1);
  check(cuEventQuery(b) == kNotReady, "cuEventQuery behind a host function: not ready");
  check(cuStreamQuery(s1) == kNotReady, "cuStreamQuery behind a host function: not ready");
  float t = -1;
  check(cuEventElapsedTime(&t, a, b) == kNotReady, "cuEventElapsedTime on a pending event: not ready");
  g.release = true;   // the main thread lets it finish, after the call returned
  check(cuEventSynchronize(b) == CUDA_SUCCESS, "cuEventSynchronize returns once it finished");
  check(g.done, "the host function ran before the event completed");
  check(cuEventQuery(b) == CUDA_SUCCESS && cuStreamQuery(s1) == CUDA_SUCCESS, "complete after it finished");
  check(cuEventElapsedTime(&t, a, b) == CUDA_SUCCESS && t >= 0, "cuEventElapsedTime after it finished");
}

// 2. Elapsed time spans the host function.
static void c_elapsed() {
  cuEventRecord(a, s1);
  cuLaunchHostFunc(s1, sleep_50, nullptr);
  cuEventRecord(b, s1);
  cuEventSynchronize(b);
  float t = -1;
  check(cuEventElapsedTime(&t, a, b) == CUDA_SUCCESS && t >= 40.0f, "elapsed time covers a 50 ms host function");
}

// 3. Stream and context synchronize wait for a host function.
static void c_sync() {
  Gate g;
  cuLaunchHostFunc(s1, wait_gate, &g);
  std::thread rel([&] { while (!g.entered) sleep_ms(1); sleep_ms(50); g.release = true; });
  check(cuStreamSynchronize(s1) == CUDA_SUCCESS && g.done, "cuStreamSynchronize waits for the host function");
  rel.join();
  Gate g2;
  cuLaunchHostFunc(s2, wait_gate, &g2);
  std::thread rel2([&] { while (!g2.entered) sleep_ms(1); sleep_ms(50); g2.release = true; });
  check(cuCtxSynchronize() == CUDA_SUCCESS && g2.done, "cuCtxSynchronize waits for the host function");
  rel2.join();
}

// 4. Work issued after a host function runs after it: each reads the pinned flag the host function sets.
static void c_order() {
  const int* pflag = const_cast<const int*>(reinterpret_cast<volatile int*>(flag));
  // a kernel
  *flag = 0;
  reset_out();
  cuLaunchHostFunc(s1, sleep_then_set, (void*)flag);
  void* params[] = {&flag, &dout};
  OK(cuLaunchKernel(copy_flag, 1, 1, 1, 1, 1, 1, 0, s1, params, nullptr));
  cuStreamSynchronize(s1);
  check(read_out() == 1, "a kernel launched after a host function sees what it wrote");
  // a copy
  *flag = 0;
  reset_out();
  cuLaunchHostFunc(s1, sleep_then_set, (void*)flag);
  OK(cuMemcpyHtoDAsync(dout, (const void*)pflag, sizeof(int), s1));
  cuStreamSynchronize(s1);
  check(read_out() == 1, "a copy issued after a host function sees what it wrote");
  // a 2D copy
  *flag = 0;
  reset_out();
  cuLaunchHostFunc(s1, sleep_then_set, (void*)flag);
  CUDA_MEMCPY2D m{};
  m.srcMemoryType = CU_MEMORYTYPE_HOST;
  m.srcHost = (const void*)pflag;
  m.srcPitch = sizeof(int);
  m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  m.dstDevice = dout;
  m.dstPitch = sizeof(int);
  m.WidthInBytes = sizeof(int);
  m.Height = 1;
  OK(cuMemcpy2DAsync(&m, s1));
  cuStreamSynchronize(s1);
  check(read_out() == 1, "a 2D copy issued after a host function sees what it wrote");
  // the legacy default stream behind a blocking stream
  *flag = 0;
  reset_out();
  cuLaunchHostFunc(blk, sleep_then_set, (void*)flag);
  OK(cuMemcpyHtoD(dout, (const void*)pflag, sizeof(int)));
  check(read_out() == 1, "a synchronous copy (legacy stream) waits for a blocking stream's host function");
  cuStreamSynchronize(blk);
  // a stream callback
  g_seq = 0;
  g_host_at = g_callback_at = -1;
  cuLaunchHostFunc(s1, note_host, nullptr);
  OK(cuStreamAddCallback(s1, note_callback, nullptr, 0));
  cuStreamSynchronize(s1);
  check(g_host_at == 0 && g_callback_at == 1, "a stream callback runs after the host function queued before it");
  // host functions on one stream run in order
  static std::atomic<int> seq{0};
  struct Step { int expect; };
  static std::atomic<int> bad{0};
  static Step steps[3] = {{0}, {1}, {2}};
  for (auto& st : steps)
    cuLaunchHostFunc(s1, [](void* p) { if (seq.fetch_add(1) != static_cast<Step*>(p)->expect) bad.fetch_add(1); }, &st);
  cuStreamSynchronize(s1);
  check(seq == 3 && bad == 0, "host functions on a stream run in order");
}

// 5. cuStreamWaitEvent on a pending event holds the other stream back.
static void c_wait_event() {
  Gate g;
  cuLaunchHostFunc(s1, wait_gate, &g);
  cuEventRecord(b, s1);
  OK(cuStreamWaitEvent(s2, b, 0));
  cuEventRecord(c, s2);
  check(cuEventQuery(c) == kNotReady, "an event behind a wait on a pending event: not ready");
  check(cuStreamQuery(s2) == kNotReady, "a stream waiting on a pending event: not ready");
  g.release = true;
  check(cuStreamSynchronize(s2) == CUDA_SUCCESS && cuEventQuery(c) == CUDA_SUCCESS,
        "the waiting stream completes once the event does");
  cuStreamSynchronize(s1);
}

// 6. The legacy default stream orders against a blocking stream, and not against a non-blocking one.
static void c_legacy() {
  Gate g;
  cuLaunchHostFunc(blk, wait_gate, &g);
  cuEventRecord(c, nullptr);   // legacy default stream: behind the blocking stream's work
  check(cuEventQuery(c) == kNotReady, "an event on the legacy stream waits for a blocking stream");
  check(cuStreamQuery(nullptr) == kNotReady, "the legacy stream is not ready while a blocking stream is busy");
  g.release = true;
  check(cuStreamSynchronize(nullptr) == CUDA_SUCCESS && cuEventQuery(c) == CUDA_SUCCESS, "and completes after it");
  Gate h;
  cuLaunchHostFunc(s1, wait_gate, &h);   // s1 is non-blocking: the legacy stream does not wait for it
  reset_out();
  const auto t0 = clk::now();
  check(cuMemcpyHtoD(dout, (const void*)flag, sizeof(int)) == CUDA_SUCCESS && since_ms(t0) < 5000 && !h.done,
        "a synchronous copy does not wait for a non-blocking stream's host function");
  check(cuStreamQuery(nullptr) == CUDA_SUCCESS, "the legacy stream is ready while only a non-blocking stream is busy");
  // And work on another stream is not held back by it either.
  CUdeviceptr other = 0;
  cuMemAlloc(&other, 16);
  OK(cuMemsetD32Async(other, 7, 4, s2));
  check(cuStreamSynchronize(s2) == CUDA_SUCCESS && !h.done, "work on another stream does not wait for it");
  cuMemFree(other);
  h.release = true;
  cuStreamSynchronize(s1);
}

// 7. A host function that waits for what the host does after enqueueing it does not hang (alpaka's shape).
static void c_hang() {
  std::atomic<int> f{0};
  Gate g;
  cuLaunchHostFunc(s1, wait_gate, &g);
  cuLaunchHostFunc(s1, set_flag, &f);
  cuEventRecord(b, s1);
  check(cuEventQuery(b) == kNotReady && f == 0, "nothing after a waiting host function ran yet");
  g.release = true;
  cuEventSynchronize(b);
  check(f == 1 && cuEventQuery(b) == CUDA_SUCCESS, "the queue drains once the host releases it");
}

// 8. Nothing pending: everything behaves as before.
static void c_idle() {
  cuEventRecord(a, s2);
  cuStreamSynchronize(s2);
  check(cuEventQuery(a) == CUDA_SUCCESS && cuStreamQuery(s2) == CUDA_SUCCESS, "idle stream, recorded event: ready");
  check(cuEventQuery(b) == CUDA_SUCCESS, "an event that finished stays ready");
}

int main(int argc, char** argv) {
  const char* which = argc > 1 ? argv[1] : "all";
  CUdevice dev = 0;
  CUcontext ctx = nullptr;
  CUmodule mod = nullptr;
  OK(cuInit(0));
  OK(cuDeviceGet(&dev, 0));
  OK(cuDevicePrimaryCtxRetain(&ctx, dev));
  OK(cuCtxSetCurrent(ctx));
  OK(cuModuleLoadData(&mod, kPtx));
  OK(cuModuleGetFunction(&copy_flag, mod, "copy_flag"));
  OK(cuStreamCreate(&s1, CU_STREAM_NON_BLOCKING));
  OK(cuStreamCreate(&s2, CU_STREAM_NON_BLOCKING));
  OK(cuStreamCreate(&blk, CU_STREAM_DEFAULT));
  OK(cuEventCreate(&a, CU_EVENT_DEFAULT));
  OK(cuEventCreate(&b, CU_EVENT_DEFAULT));
  OK(cuEventCreate(&c, CU_EVENT_DEFAULT));
  void* pinned = nullptr;
  OK(cuMemHostAlloc(&pinned, sizeof(int), CU_MEMHOSTALLOC_DEVICEMAP));
  flag = static_cast<volatile int*>(pinned);
  OK(cuMemAlloc(&dout, sizeof(int)));
  const struct { const char* name; void (*fn)(); } cases[] = {
      {"not_ready", c_not_ready}, {"elapsed", c_elapsed},     {"sync", c_sync},   {"order", c_order},
      {"wait_event", c_wait_event}, {"legacy", c_legacy}, {"hang", c_hang}, {"idle", c_idle}};
  for (const auto& cs : cases)
    if (!std::strcmp(which, "all") || !std::strcmp(which, cs.name)) {
      std::printf("-- %s\n", cs.name);
      cs.fn();
    }
  cuStreamSynchronize(nullptr);
  cuStreamDestroy(s1);
  cuStreamDestroy(s2);
  cuStreamDestroy(blk);
  std::printf("%s\n", failures ? "FAILED" : "all passed");
  return failures ? 1 : 0;
}
