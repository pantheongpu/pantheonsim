// Stream capture modes: which calls a thread may make while a capture is going
// on. Every check passes on an RTX 3060's runtime (driver 13.2) as well as on
// this simulator; the expected values are what the card answered.
//
// The rule, from the capture modes' documentation and the card: a thread has a
// mode (cudaThreadExchangeStreamCaptureMode, Global at first) and a capture has
// the mode it began with. A "potentially unsafe" call is refused with
// cudaErrorStreamCaptureUnsupported -- which also invalidates the capture -- when
// the thread's mode is not Relaxed and either the thread has a capture of its own
// that was not begun Relaxed, or (in Global mode) another thread has one begun
// Global. cudaDeviceSynchronize fails during any capture, in any mode.
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <string>
#include <thread>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

__global__ void bump(int* p) { *p += 1; }

static int* g_dev = nullptr;
static cudaStream_t g_other = nullptr;

struct Call {
  const char* name;
  std::function<cudaError_t()> fn;
  bool unsafe;   // refused by the mode rule (cudaDeviceSynchronize is separate)
};

// The calls under test. Each releases what it makes, so a success leaks nothing.
static std::vector<Call> calls() {
  return {
      {"cudaMalloc", [] { void* p; cudaError_t e = cudaMalloc(&p, 64); if (e == cudaSuccess) cudaFree(p); return e; }, true},
      {"cudaFree(NULL)", [] { return cudaFree(nullptr); }, true},
      {"cudaMallocHost", [] { void* p; cudaError_t e = cudaMallocHost(&p, 64); if (e == cudaSuccess) cudaFreeHost(p); return e; }, true},
      {"cudaHostAlloc", [] { void* p; cudaError_t e = cudaHostAlloc(&p, 64, 0); if (e == cudaSuccess) cudaFreeHost(p); return e; }, true},
      {"cudaMallocManaged", [] { void* p; cudaError_t e = cudaMallocManaged(&p, 64); if (e == cudaSuccess) cudaFree(p); return e; }, true},
      {"cudaMallocPitch", [] { void* p; size_t pitch; cudaError_t e = cudaMallocPitch(&p, &pitch, 64, 4); if (e == cudaSuccess) cudaFree(p); return e; }, true},
      {"cudaMallocArray",
       [] { cudaChannelFormatDesc f = cudaCreateChannelDesc<float>(); cudaArray_t a; cudaError_t e = cudaMallocArray(&a, &f, 8, 8); if (e == cudaSuccess) cudaFreeArray(a); return e; },
       true},
      {"cudaMallocAsync (a stream not capturing)",
       [] { void* p; cudaError_t e = cudaMallocAsync(&p, 64, g_other); if (e == cudaSuccess) { cudaFreeAsync(p, g_other); cudaStreamSynchronize(g_other); } return e; },
       true},
      {"cudaHostRegister",
       [] { static char buf[4096 * 4]; cudaError_t e = cudaHostRegister(buf, sizeof buf, 0); if (e == cudaSuccess) cudaHostUnregister(buf); return e; },
       true},
      {"cudaMemPoolCreate",
       [] { cudaMemPoolProps pp = {}; pp.allocType = cudaMemAllocationTypePinned; pp.location.type = cudaMemLocationTypeDevice; cudaMemPool_t p; cudaError_t e = cudaMemPoolCreate(&p, &pp); if (e == cudaSuccess) cudaMemPoolDestroy(p); return e; },
       true},
      {"cudaMemPoolTrimTo", [] { cudaMemPool_t p; cudaDeviceGetDefaultMemPool(&p, 0); return cudaMemPoolTrimTo(p, 0); }, true},
      {"cudaStreamSynchronize (a stream not capturing)", [] { return cudaStreamSynchronize(g_other); }, true},
      {"cudaStreamQuery (a stream not capturing)", [] { return cudaStreamQuery(g_other); }, true},
      {"cudaGraphInstantiate",
       [] { cudaGraph_t g; cudaGraphCreate(&g, 0); cudaGraphExec_t ge; cudaError_t e = cudaGraphInstantiate(&ge, g, 0); if (e == cudaSuccess) cudaGraphExecDestroy(ge); cudaGraphDestroy(g); return e; },
       true},
      {"cudaDeviceSetLimit", [] { return cudaDeviceSetLimit(cudaLimitStackSize, 2048); }, true},
      {"cudaProfilerStart", [] { return cudaProfilerStart(); }, true},
      // Calls a capture does not forbid.
      {"cudaEventCreate", [] { cudaEvent_t e; cudaError_t r = cudaEventCreate(&e); if (r == cudaSuccess) cudaEventDestroy(e); return r; }, false},
      {"cudaStreamCreate", [] { cudaStream_t s; cudaError_t r = cudaStreamCreate(&s); if (r == cudaSuccess) cudaStreamDestroy(s); return r; }, false},
      {"cudaMemGetInfo", [] { size_t a, b; return cudaMemGetInfo(&a, &b); }, false},
      {"cudaDeviceGetAttribute", [] { int v; return cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); }, false},
      {"cudaGraphCreate", [] { cudaGraph_t g; cudaError_t e = cudaGraphCreate(&g, 0); if (e == cudaSuccess) cudaGraphDestroy(g); return e; }, false},
      {"cudaMemcpyAsync D2D (another stream)", [] { return cudaMemcpyAsync(g_dev + 4, g_dev, 4, cudaMemcpyDeviceToDevice, g_other); }, false},
      {"cudaGetDeviceCount", [] { int n; return cudaGetDeviceCount(&n); }, false},
  };
}

static const char* mode_name(int m) { return m == 0 ? "Global" : m == 1 ? "ThreadLocal" : "Relaxed"; }

// One capture, one call, then the capture is ended and its result checked.
// `cap_mode` is what the capture began with, `thread_mode` what the calling thread's mode
// is (-1: the default, Global), `other_thread` whether the call comes from a thread that
// began no capture.
static void trial(const Call& c, int cap_mode, int thread_mode, bool other_thread) {
  cudaStream_t cs;
  cudaStreamCreate(&cs);
  cudaStreamBeginCapture(cs, static_cast<cudaStreamCaptureMode>(cap_mode));
  bump<<<1, 1, 0, cs>>>(g_dev);
  cudaError_t r = cudaSuccess;
  auto run = [&] {
    if (thread_mode >= 0) {
      cudaStreamCaptureMode m = static_cast<cudaStreamCaptureMode>(thread_mode);
      cudaThreadExchangeStreamCaptureMode(&m);
    }
    r = c.fn();
  };
  if (other_thread) {
    std::thread t(run);
    t.join();
  } else {
    run();
    if (thread_mode >= 0) {   // push-pop: back to the default
      cudaStreamCaptureMode m = cudaStreamCaptureModeGlobal;
      cudaThreadExchangeStreamCaptureMode(&m);
    }
  }
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(cs, &st);
  cudaGraph_t g = nullptr;
  const cudaError_t end = cudaStreamEndCapture(cs, &g);
  // Refused when the thread's mode is not Relaxed and a capture of its own, not begun Relaxed, is going on
  // -- or, in Global mode, another thread's Global capture is.
  const int tm = thread_mode < 0 ? 0 : thread_mode;
  const bool mine_counts = !other_thread && cap_mode != cudaStreamCaptureModeRelaxed;
  const bool others_counts = other_thread && cap_mode == cudaStreamCaptureModeGlobal && tm != cudaStreamCaptureModeThreadLocal;
  const bool refused = c.unsafe && tm != cudaStreamCaptureModeRelaxed && (mine_counts || others_counts);
  char what[256];
  std::snprintf(what, sizeof what, "%s: capture %s, %s thread in mode %s -> %s", c.name, mode_name(cap_mode),
                other_thread ? "another" : "the same", mode_name(tm < 3 ? tm : 0), refused ? "refused, capture invalidated" : "allowed");
  check(refused ? (r == cudaErrorStreamCaptureUnsupported && st == cudaStreamCaptureStatusInvalidated &&
                   end == cudaErrorStreamCaptureInvalidated)
                : (r == cudaSuccess && st == cudaStreamCaptureStatusActive && end == cudaSuccess),
        what);
  if (g) cudaGraphDestroy(g);
  cudaGetLastError();
  cudaStreamDestroy(cs);
}

static void exchange() {
  cudaStreamCaptureMode m = cudaStreamCaptureModeRelaxed;
  IS(cudaThreadExchangeStreamCaptureMode(&m), cudaSuccess);
  check(m == cudaStreamCaptureModeGlobal, "a thread starts in Global mode");
  m = cudaStreamCaptureModeThreadLocal;
  IS(cudaThreadExchangeStreamCaptureMode(&m), cudaSuccess);
  check(m == cudaStreamCaptureModeRelaxed, "the exchange hands back the previous mode");
  m = cudaStreamCaptureModeGlobal;
  IS(cudaThreadExchangeStreamCaptureMode(&m), cudaSuccess);
  check(m == cudaStreamCaptureModeThreadLocal, "and the one before that");
  // A value that is no mode is kept as given, as the card keeps it.
  int odd = 7;
  std::memcpy(&m, &odd, sizeof odd);
  IS(cudaThreadExchangeStreamCaptureMode(&m), cudaSuccess);
  m = cudaStreamCaptureModeGlobal;
  IS(cudaThreadExchangeStreamCaptureMode(&m), cudaSuccess);
  int back = 0;
  std::memcpy(&back, &m, sizeof back);
  check(back == 7, "a mode that is none of the three is stored as given");

  cudaStream_t s;
  cudaStreamCreate(&s);
  // These numbers are no mode, and holding one in an enum-typed object is undefined behaviour (UBSan reports it):
  // the mode is an int in the ABI, so call the entry point as taking one.
  using BeginWithInt = cudaError_t (*)(cudaStream_t, int);
  const BeginWithInt begin_with_int = reinterpret_cast<BeginWithInt>(&cudaStreamBeginCapture);
  IS(begin_with_int(s, 3), cudaErrorInvalidValue);
  IS(begin_with_int(s, -1), cudaErrorInvalidValue);
  cudaStreamDestroy(s);
}

static void always_refused() {
  for (int cap_mode = 0; cap_mode < 3; ++cap_mode) {
    for (bool other : {false, true}) {
      cudaStream_t cs;
      cudaStreamCreate(&cs);
      cudaStreamBeginCapture(cs, static_cast<cudaStreamCaptureMode>(cap_mode));
      bump<<<1, 1, 0, cs>>>(g_dev);
      cudaError_t r = cudaSuccess;
      if (other) {
        std::thread t([&] { r = cudaDeviceSynchronize(); });
        t.join();
      } else {
        r = cudaDeviceSynchronize();
      }
      cudaStreamCaptureStatus st;
      cudaStreamIsCapturing(cs, &st);
      cudaGraph_t g = nullptr;
      const cudaError_t end = cudaStreamEndCapture(cs, &g);
      char what[128];
      std::snprintf(what, sizeof what, "cudaDeviceSynchronize during a %s capture, from %s thread: refused in every mode",
                    mode_name(cap_mode), other ? "another" : "the same");
      check(r == cudaErrorStreamCaptureUnsupported && st == cudaStreamCaptureStatusInvalidated && end == cudaErrorStreamCaptureInvalidated, what);
      if (g) cudaGraphDestroy(g);
      cudaGetLastError();
      cudaStreamDestroy(cs);
    }
  }
}

static void wrong_thread() {
  for (int cap_mode = 0; cap_mode < 3; ++cap_mode) {
    cudaStream_t cs;
    cudaStreamCreate(&cs);
    cudaStreamBeginCapture(cs, static_cast<cudaStreamCaptureMode>(cap_mode));
    bump<<<1, 1, 0, cs>>>(g_dev);
    cudaError_t r = cudaSuccess;
    cudaGraph_t g = nullptr;
    std::thread t([&] { r = cudaStreamEndCapture(cs, &g); });
    t.join();
    char what[128];
    if (cap_mode == cudaStreamCaptureModeRelaxed) {
      std::snprintf(what, sizeof what, "a Relaxed capture can be ended by another thread");
      check(r == cudaSuccess && g, what);
    } else {
      std::snprintf(what, sizeof what, "a %s capture ended by another thread is cudaErrorStreamCaptureWrongThread, and over", mode_name(cap_mode));
      check(r == cudaErrorStreamCaptureWrongThread && !g, what);
    }
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusActive;
    cudaStreamIsCapturing(cs, &st);
    check(st == cudaStreamCaptureStatusNone, "the stream is not capturing afterwards");
    if (g) cudaGraphDestroy(g);
    cudaGetLastError();
    cudaStreamDestroy(cs);
  }
}

// Another thread's Relaxed capture does not exempt it from a Global capture's rule; its own thread mode does.
static void two_threads() {
  cudaStream_t cs, cs2;
  cudaStreamCreate(&cs);
  cudaStreamCreate(&cs2);
  cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal);
  bump<<<1, 1, 0, cs>>>(g_dev);
  cudaError_t begin = cudaSuccess, refused = cudaSuccess, allowed = cudaErrorUnknown;
  std::thread t([&] {
    begin = cudaStreamBeginCapture(cs2, cudaStreamCaptureModeRelaxed);
    void* q = nullptr;
    refused = cudaMalloc(&q, 64);
    cudaStreamCaptureMode m = cudaStreamCaptureModeRelaxed;
    cudaThreadExchangeStreamCaptureMode(&m);
    allowed = cudaMalloc(&q, 64);
    if (allowed == cudaSuccess) cudaFree(q);
    cudaGraph_t g2 = nullptr;
    cudaStreamEndCapture(cs2, &g2);
    if (g2) cudaGraphDestroy(g2);
  });
  t.join();
  IS(begin, cudaSuccess);
  IS(refused, cudaErrorStreamCaptureUnsupported);
  IS(allowed, cudaSuccess);
  cudaGraph_t g = nullptr;
  IS(cudaStreamEndCapture(cs, &g), cudaErrorStreamCaptureInvalidated);
  if (g) cudaGraphDestroy(g);
  cudaGetLastError();
  cudaStreamDestroy(cs);
  cudaStreamDestroy(cs2);
}

// A capture into a graph the program made fails like any other; the graph goes with it.
static void to_graph() {
#if CUDART_VERSION >= 12030  // cudaStreamBeginCaptureToGraph arrived in CUDA 12.3
  cudaStream_t cs;
  cudaStreamCreate(&cs);
  cudaGraph_t graph;
  cudaGraphCreate(&graph, 0);
  IS(cudaStreamBeginCaptureToGraph(cs, graph, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal), cudaSuccess);
  void* p = nullptr;
  IS(cudaMalloc(&p, 64), cudaErrorStreamCaptureUnsupported);
  cudaGraph_t out = nullptr;
  IS(cudaStreamEndCapture(cs, &out), cudaErrorStreamCaptureInvalidated);
  cudaGetLastError();
  cudaStreamDestroy(cs);
#endif
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (cudaFree(nullptr) != cudaSuccess) {
    std::printf("FAIL: no device\n");
    return 1;
  }
  cudaMalloc(&g_dev, 64);
  cudaStreamCreate(&g_other);
  exchange();
  for (const Call& c : calls()) {
    for (int cap = 0; cap < 3; ++cap) {
      trial(c, cap, -1, false);        // the thread's default mode
      trial(c, cap, cudaStreamCaptureModeThreadLocal, false);
      trial(c, cap, cudaStreamCaptureModeRelaxed, false);
      trial(c, cap, -1, true);         // another thread, in its default mode
      trial(c, cap, cudaStreamCaptureModeRelaxed, true);
    }
  }
  always_refused();
  wrong_thread();
  two_threads();
  to_graph();
  cudaStreamDestroy(g_other);
  cudaFree(g_dev);
  std::printf(failures ? "FAIL: %d runtime checks\n" : "PASS: every runtime check\n", failures);
  return failures ? 1 : 0;
}
