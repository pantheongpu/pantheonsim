// The driver API's stream calls inside a captured CUDA graph (see graph_capture_common.h): the
// capture, instantiation and launch themselves are the driver API's (cuStreamBeginCapture,
// cuStreamEndCapture, cuGraphInstantiateWithFlags, cuGraphLaunch), and what is captured is the driver
// API's own stream-ordered calls -- copies of every shape, fills, host functions, events and waits across
// streams, stream memory operations and stream-ordered allocation -- followed by the calls CUDA refuses
// while a stream captures. run_graph_capture.sh driver --card runs the same program on NVIDIA's driver.
#define GC_DRIVER_API 1
#include <cuda.h>

#include <algorithm>
#include <atomic>

#include "graph_capture_common.h"

using namespace gc;

#define OK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { \
  std::printf("     %s -> %d\n", #x, (int)r_); ok = false; } } while (0)

static CUstream S(const Runner& r) { return reinterpret_cast<CUstream>(r.st); }
static CUdeviceptr D(const void* p) { return reinterpret_cast<CUdeviceptr>(p); }

/* ---- copies ----------------------------------------------------------------------------- */

static void copies(Runner& r) {
  const int n = 1000;
  float *src = r.alloc<float>(n), *mid = r.alloc<float>(n), *dst = r.alloc<float>(n);
  r.run("cuMemcpyAsync + cuMemcpyDtoDAsync", [&] {
    bool ok = true;
    fill(r.st, r.counter, src, n);
    OK(cuMemcpyAsync(D(mid), D(src), n * 4, S(r)));
    OK(cuMemcpyDtoDAsync_v2(D(dst), D(mid), n * 4, S(r)));
    return ok;
  }, {{mid, (size_t)n, Dt::F32}, {dst, (size_t)n, Dt::F32}});

  // Through host memory: the device value goes down to pinned memory and back, in stream order.
  void* pinned = nullptr;
  cuMemAllocHost(&pinned, n * 4);
  float* out = r.alloc<float>(n);
  r.run("cuMemcpyDtoHAsync + cuMemcpyHtoDAsync (pinned)", [&] {
    bool ok = true;
    fill(r.st, r.counter, src, n, 0.5f, 1.0f);
    OK(cuMemcpyDtoHAsync_v2(pinned, D(src), n * 4, S(r)));
    OK(cuMemcpyHtoDAsync_v2(D(out), pinned, n * 4, S(r)));
    return ok;
  }, {{out, (size_t)n, Dt::F32}});
  cuMemFreeHost(pinned);

  // The 2D and 3D descriptors are locals: gone when the capture function returns.
  float *a = r.alloc<float>(32 * 8), *b = r.alloc<float>(32 * 8);
  r.run("cuMemcpy2DAsync", [&] {
    bool ok = true;
    fill(r.st, r.counter, a, 32 * 8);
    CUDA_MEMCPY2D c;
    std::memset(&c, 0, sizeof c);
    c.srcMemoryType = CU_MEMORYTYPE_DEVICE, c.srcDevice = D(a), c.srcPitch = 32 * 4;
    c.dstMemoryType = CU_MEMORYTYPE_DEVICE, c.dstDevice = D(b), c.dstPitch = 32 * 4;
    c.WidthInBytes = 20 * 4, c.Height = 8;
    OK(cuMemcpy2DAsync(&c, S(r)));
    return ok;
  }, {{b, 32 * 8, Dt::F32}});
  r.run("cuMemcpy3DAsync", [&] {
    bool ok = true;
    fill(r.st, r.counter, a, 32 * 8);
    CUDA_MEMCPY3D c;
    std::memset(&c, 0, sizeof c);
    c.srcMemoryType = CU_MEMORYTYPE_DEVICE, c.srcDevice = D(a), c.srcPitch = 32 * 4, c.srcHeight = 4;
    c.dstMemoryType = CU_MEMORYTYPE_DEVICE, c.dstDevice = D(b), c.dstPitch = 32 * 4, c.dstHeight = 4;
    c.WidthInBytes = 16 * 4, c.Height = 4, c.Depth = 2;
    OK(cuMemcpy3DAsync(&c, S(r)));
    return ok;
  }, {{b, 32 * 8, Dt::F32}});

  // Arrays.
  CUarray arr = nullptr;
  CUDA_ARRAY_DESCRIPTOR ad = {256, 0, CU_AD_FORMAT_FLOAT, 1};   // one-dimensional: the only kind these copies take
  cuArrayCreate(&arr, &ad);
  void* host = nullptr;
  cuMemAllocHost(&host, 256 * 4);
  float* aout = r.alloc<float>(256);
  r.run("cuMemcpyHtoAAsync + cuMemcpyAtoHAsync", [&] {
    bool ok = true;
    fill(r.st, r.counter, a, 256);
    OK(cuMemcpyDtoHAsync_v2(host, D(a), 256 * 4, S(r)));
    OK(cuMemcpyHtoAAsync(arr, 0, host, 256 * 4, S(r)));
    OK(cuMemcpyAtoHAsync(host, arr, 0, 256 * 4, S(r)));
    OK(cuMemcpyHtoDAsync_v2(D(aout), host, 256 * 4, S(r)));
    return ok;
  }, {{aout, 256, Dt::F32}});
  cuMemFreeHost(host);
  cuArrayDestroy(arr);
}

/* ---- fills ------------------------------------------------------------------------------ */

static void fills(Runner& r) {
  const int n = 256;
  unsigned char* p = r.alloc<unsigned char>(n * 4);
  r.run("cuMemsetD8Async + D16 + D32", [&] {
    bool ok = true;
    // the values are the call's, not the counter's: the graph repeats them
    OK(cuMemsetD8Async(D(p), 0x5a, 100, S(r)));
    OK(cuMemsetD16Async(D(p + 100), 0x1234, 50, S(r)));
    OK(cuMemsetD32Async(D(p + 200), 0xdeadbeef, 100, S(r)));
    return ok;
  }, {{p, (size_t)n * 4, Dt::Bytes}});
  unsigned char* q = r.alloc<unsigned char>(64 * 8);
  r.run("cuMemsetD2D8Async + D2D16 + D2D32", [&] {
    bool ok = true;
    OK(cuMemsetD2D8Async(D(q), 64, 0x11, 32, 3, S(r)));
    OK(cuMemsetD2D16Async(D(q + 64 * 3), 64, 0x2222, 16, 2, S(r)));
    OK(cuMemsetD2D32Async(D(q + 64 * 5), 64, 0x33333333, 8, 3, S(r)));
    return ok;
  }, {{q, 64 * 8, Dt::Bytes}});
}

/* ---- host functions --------------------------------------------------------------------- */

static std::atomic<int> g_calls{0};
static void host_fn(void* user) { g_calls += *static_cast<int*>(user); }

static void host_function(Runner& r) {
  static int amount = 3;
  int before = 0;
  float* x = r.alloc<float>(4);
  r.run("cuLaunchHostFunc", [&] {
    bool ok = true;
    fill(r.st, r.counter, x, 4);
    OK(cuLaunchHostFunc(S(r), host_fn, &amount));
    return ok;
  }, {{x, 4, Dt::F32}}, 0, [&](int, int launch) {
    // eager (1), the capture (0), and each launch runs it once: the capture itself must not.
    if (launch == 1) before = g_calls;
    const int per = 3;
    return g_calls.load() == before + (launch == 1 ? 0 : per * (launch - 1));
  });
}

/* ---- events across streams -------------------------------------------------------------- */

static void fork_join(Runner& r) {
  const int n = 500;
  float *a = r.alloc<float>(n), *b = r.alloc<float>(n), *c = r.alloc<float>(n);
  CUstream side = nullptr;
  cuStreamCreate(&side, CU_STREAM_NON_BLOCKING);
  CUevent fork = nullptr, join = nullptr;
  cuEventCreate(&fork, CU_EVENT_DISABLE_TIMING);
  cuEventCreate(&join, CU_EVENT_DISABLE_TIMING);
  r.run("cuEventRecord + cuStreamWaitEvent (a fork and a join)", [&] {
    bool ok = true;
    fill(r.st, r.counter, a, n);
    OK(cuEventRecord(fork, S(r)));
    OK(cuStreamWaitEvent(side, fork, 0));   // the side stream joins the capture here
    OK(cuMemcpyAsync(D(b), D(a), n * 4, side));
    OK(cuMemsetD32Async(D(c), 0x3f800000, n, S(r)));   // 1.0f, on the first stream
    OK(cuEventRecord(join, side));
    OK(cuStreamWaitEvent(S(r), join, 0));
    return ok;
  }, {{b, (size_t)n, Dt::F32}, {c, (size_t)n, Dt::F32}});
  cuEventDestroy(fork);
  cuEventDestroy(join);
  cuStreamDestroy(side);
}

/* ---- stream memory operations ----------------------------------------------------------- */

static void stream_memory(Runner& r) {
  unsigned* w = r.alloc<unsigned>(8);
  r.run("cuStreamWriteValue32/64 + cuStreamWaitValue32/64", [&] {
    bool ok = true;
    OK(cuStreamWriteValue32(S(r), D(w), 0xabcd, 0));
    OK(cuStreamWriteValue64(S(r), D(w + 2), 0x1122334455667788ull, 0));
    OK(cuStreamWaitValue32(S(r), D(w), 0xabcd, CU_STREAM_WAIT_VALUE_EQ));      // already true
    OK(cuStreamWaitValue64(S(r), D(w + 2), 0x1122334455667788ull, CU_STREAM_WAIT_VALUE_GEQ));
    OK(cuStreamWriteValue32(S(r), D(w + 5), 7, 0));
    return ok;
  }, {{w, 8, Dt::I32}});
  r.run("cuStreamBatchMemOp", [&] {
    bool ok = true;
    CUstreamBatchMemOpParams ops[3];
    std::memset(ops, 0, sizeof ops);
    ops[0].operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    ops[0].writeValue.address = D(w + 1), ops[0].writeValue.value = 41;
    ops[1].operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    ops[1].writeValue.address = D(w + 4), ops[1].writeValue.value = 99;
    ops[2].operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    ops[2].waitValue.address = D(w + 1), ops[2].waitValue.value = 41, ops[2].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    OK(cuStreamBatchMemOp(S(r), 3, ops, 0));
    return ok;
  }, {{w, 8, Dt::I32}});
}

/* ---- stream-ordered allocation ---------------------------------------------------------- */

static void stream_ordered_allocation(Runner& r) {
  const int n = 1000;
  float *src = r.alloc<float>(n), *dst = r.alloc<float>(n);
  r.run("cuMemAllocAsync + cuMemFreeAsync (graph memory)", [&] {
    bool ok = true;
    fill(r.st, r.counter, src, n);
    CUdeviceptr tmp = 0;
    OK(cuMemAllocAsync(&tmp, n * 4, S(r)));
    OK(cuMemcpyAsync(tmp, D(src), n * 4, S(r)));
    OK(cuMemcpyAsync(D(dst), tmp, n * 4, S(r)));
    OK(cuMemFreeAsync(tmp, S(r)));
    return ok;
  }, {{dst, (size_t)n, Dt::F32}});
}

/* ---- what a capture refuses -------------------------------------------------------------- */

// Each is called in a capture of its own: the call's answer, whether the stream still captures after
// it, and what ending the capture says. Measured on the RTX 3060.
static void refusals(Runner& r) {
  struct Case {
    const char* name;
    std::function<CUresult()> call;
    CUresult want_rc;
    CUstreamCaptureStatus want_after;   // active (untouched) or invalidated
  };
  CUdeviceptr d = D(r.alloc<float>(1024));
  float* host = nullptr;
  cuMemAllocHost(reinterpret_cast<void**>(&host), 4096);
  CUevent never = nullptr, outside = nullptr;
  cuEventCreate(&never, CU_EVENT_DISABLE_TIMING);
  cuEventCreate(&outside, CU_EVENT_DISABLE_TIMING);
  cuEventRecord(outside, 0);
  cuCtxSynchronize();
  const CUresult unsupported = static_cast<CUresult>(900), isolation = static_cast<CUresult>(905);
  const std::vector<Case> cases = {
    {"cuMemAlloc", [&] { CUdeviceptr p; return cuMemAlloc(&p, 4096); }, unsupported, CU_STREAM_CAPTURE_STATUS_INVALIDATED},
    {"cuMemFree", [&] { CUdeviceptr p = 0; cuMemAlloc(&p, 64); return cuMemFree(p); }, unsupported, CU_STREAM_CAPTURE_STATUS_INVALIDATED},
    {"cuMemAllocHost", [&] { void* p; return cuMemAllocHost(&p, 4096); }, unsupported, CU_STREAM_CAPTURE_STATUS_INVALIDATED},
    {"cuMemAllocManaged", [&] { CUdeviceptr p; return cuMemAllocManaged(&p, 4096, CU_MEM_ATTACH_GLOBAL); }, unsupported, CU_STREAM_CAPTURE_STATUS_INVALIDATED},
    {"cuStreamSynchronize", [&] { return cuStreamSynchronize(S(r)); }, unsupported, CU_STREAM_CAPTURE_STATUS_INVALIDATED},
    {"cuStreamQuery", [&] { return cuStreamQuery(S(r)); }, unsupported, CU_STREAM_CAPTURE_STATUS_INVALIDATED},
    {"cuCtxSynchronize", [&] { return cuCtxSynchronize(); }, unsupported, CU_STREAM_CAPTURE_STATUS_INVALIDATED},
    {"cuMemPrefetchAsync", [&] { CUmemLocation l; l.type = CU_MEM_LOCATION_TYPE_DEVICE; l.id = 0; return cuMemPrefetchAsync(d, 4096, l, 0, S(r)); }, unsupported, CU_STREAM_CAPTURE_STATUS_INVALIDATED},
    {"cuStreamAttachMemAsync", [&] { return cuStreamAttachMemAsync(S(r), d, 0, CU_MEM_ATTACH_GLOBAL); }, unsupported, CU_STREAM_CAPTURE_STATUS_INVALIDATED},
    {"cuMemcpyPeerAsync", [&] { CUcontext c; cuCtxGetCurrent(&c); return cuMemcpyPeerAsync(d, c, d + 2048, c, 64, S(r)); }, unsupported, CU_STREAM_CAPTURE_STATUS_INVALIDATED},
    {"cuStreamAddCallback (refused, the capture goes on)", [&] { return cuStreamAddCallback(S(r), [](CUstream, CUresult, void*) {}, nullptr, 0); }, unsupported, CU_STREAM_CAPTURE_STATUS_ACTIVE},
    {"cuCtxSetLimit", [&] { return cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, 1 << 20); }, unsupported, CU_STREAM_CAPTURE_STATUS_INVALIDATED},
    {"cuMemFreeAsync of a pointer that is no graph allocation", [&] { return cuMemFreeAsync(d, S(r)); }, CUDA_ERROR_INVALID_VALUE, CU_STREAM_CAPTURE_STATUS_ACTIVE},
    {"cuStreamBeginCapture on a capturing stream", [&] { return cuStreamBeginCapture(S(r), CU_STREAM_CAPTURE_MODE_GLOBAL); }, CUDA_ERROR_ILLEGAL_STATE, CU_STREAM_CAPTURE_STATUS_ACTIVE},
    {"cuStreamWaitEvent on an event never recorded (no-op)", [&] { return cuStreamWaitEvent(S(r), never, 0); }, CUDA_SUCCESS, CU_STREAM_CAPTURE_STATUS_ACTIVE},
    {"cuStreamWaitEvent on an event recorded outside the capture", [&] { return cuStreamWaitEvent(S(r), outside, 0); }, isolation, CU_STREAM_CAPTURE_STATUS_INVALIDATED},
    {"cuMemcpyHtoD (synchronous: runs at once, nothing recorded)", [&] { return cuMemcpyHtoD(d, host, 64); }, CUDA_SUCCESS, CU_STREAM_CAPTURE_STATUS_ACTIVE},
    {"cuMemsetD8 (synchronous: runs at once, nothing recorded)", [&] { return cuMemsetD8(d, 1, 64); }, CUDA_SUCCESS, CU_STREAM_CAPTURE_STATUS_ACTIVE},
  };
  for (const Case& c : cases) {
    CUgraph g = nullptr;
    const CUresult b = cuStreamBeginCapture(S(r), CU_STREAM_CAPTURE_MODE_GLOBAL);
    cuMemsetD32Async(d, 1, 16, S(r));   // a node, so the graph is never empty
    const CUresult rc = c.call();
    CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
    cuStreamIsCapturing(S(r), &status);
    const CUresult e = cuStreamEndCapture(S(r), &g);
    if (g) cuGraphDestroy(g);
    cudaGetLastError();
    const bool invalidated = c.want_after == CU_STREAM_CAPTURE_STATUS_INVALIDATED;
    expect(std::string(c.name) + " in a capture", b == CUDA_SUCCESS && rc == c.want_rc && status == c.want_after &&
                                                     (invalidated ? e == static_cast<CUresult>(901) : e == CUDA_SUCCESS),
           static_cast<double>(rc));
    cuCtxSynchronize();
  }
  cuEventDestroy(never);
  cuEventDestroy(outside);
  cuMemFreeHost(host);
}

int main() {
  Runner r;
  copies(r);
  fills(r);
  host_function(r);
  fork_join(r);
  stream_memory(r);
  stream_ordered_allocation(r);
  refusals(r);
  return finish();
}
