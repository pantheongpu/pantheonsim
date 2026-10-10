// What the graph_capture_* programs share: a harness that checks a library call
// the way a CUDA graph uses it, and nothing else.
//
// On NVIDIA's libraries a call made while its stream is capturing is recorded,
// not run: it executes at each launch of the graph, over whatever the graph's
// own kernels have written by then, and it must not depend on anything the
// caller destroys after the capture (descriptors, plans, host scalars on the
// stack). So, for each case:
//   1. the call is made once eagerly (counter 3) and its outputs are kept;
//   2. the outputs are zeroed and the call -- preceded, in the same capture, by
//      kernels that write every input from a counter in device memory -- is
//      captured; whatever the call built (descriptors, plans, scalars) is gone
//      when the capture function returns;
//   3. the capture must not have run the call (the outputs are still zero);
//   4. for two other counter values the call is made eagerly (the reference),
//      the outputs are zeroed, and the graph is launched: its outputs must match
//      the reference, which they do only if the call ran at launch, over the
//      inputs the counter produced.
// Nothing here knows what the call computes, so the same program checks the
// shims and, with the card's libraries, what NVIDIA's do (which is what these
// expectations were written from). A case whose call the card does not capture
// says so in its name and the program expects what the card does.
#pragma once

// With GC_DRIVER_API defined before this header, the capture itself (begin, end, instantiate, launch,
// destroy) goes through the driver API's cuStreamBeginCapture and cuGraph*, and the stream is a driver
// stream: the program checks those calls as well as what is captured.
#ifdef GC_DRIVER_API
#include <cuda.h>
#endif
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace gc {

// The capture mode the runtime-API harness captures in (the driver API's is always global).
inline cudaStreamCaptureMode capture_mode = cudaStreamCaptureModeGlobal;
inline int fails = 0;
inline int checks = 0;

inline void expect(const std::string& what, bool ok, double detail = 0) {
  ++checks;
  std::printf("%s %s", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok && detail != 0) std::printf(" (%g)", detail);
  std::printf("\n");
  fails += !ok;
}

/* ---- inputs written by kernels, from the counter ----------------------------- */

__host__ __device__ inline float pattern(size_t i, int counter, int salt, int mode) {
  const int v = static_cast<int>((i * 7 + 3 * static_cast<size_t>(counter) + 5 * static_cast<size_t>(salt)) % 11);
  // mode 0: centred on zero, mode 1: strictly positive, mode 2: 0 or 1.
  return mode == 0 ? static_cast<float>(v - 5) : mode == 1 ? static_cast<float>(v + 1) : static_cast<float>(v % 2);
}
__device__ inline void put(float* p, float v) { *p = v; }
__device__ inline void put(double* p, float v) { *p = v; }
__device__ inline void put(__half* p, float v) { *p = __float2half(v); }
__device__ inline void put(int* p, float v) { *p = static_cast<int>(v); }
__device__ inline void put(unsigned char* p, float v) { *p = static_cast<unsigned char>(v); }

template <class T>
__global__ void fill_kernel(T* p, size_t n, const int* counter, float scale, float shift, int salt, int mode) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) put(p + i, shift + scale * pattern(i, *counter, salt, mode));
}

// p[i] = shift + scale * pattern(i, counter, salt, mode) -- on the stream, so it is part of a capture.
template <class T>
inline void fill(cudaStream_t st, const int* counter, T* p, size_t n, float scale = 0.25f, float shift = 0.0f,
                 int salt = 0, int mode = 0) {
  if (!n) return;
  fill_kernel<T><<<static_cast<unsigned>((n + 127) / 128), 128, 0, st>>>(p, n, counter, scale, shift, salt, mode);
}

/* ---- outputs ----------------------------------------------------------------- */

enum class Dt { F32, F64, F16, I32, Bytes };
struct Out {
  void* p;
  size_t n;   // elements
  Dt t;
  const char* label = "";
};
inline size_t dt_bytes(Dt t) { return t == Dt::F64 ? 8 : t == Dt::F16 ? 2 : t == Dt::Bytes ? 1 : 4; }

// The graph calls, in whichever API the program checks. 0 is success.
struct GraphApi {
#ifdef GC_DRIVER_API
  static int begin(cudaStream_t st) { return cuStreamBeginCapture(reinterpret_cast<CUstream>(st), CU_STREAM_CAPTURE_MODE_GLOBAL); }
  static int end(cudaStream_t st, void** graph) { return cuStreamEndCapture(reinterpret_cast<CUstream>(st), reinterpret_cast<CUgraph*>(graph)); }
  static int instantiate(void** exec, void* graph) {
    return cuGraphInstantiateWithFlags(reinterpret_cast<CUgraphExec*>(exec), reinterpret_cast<CUgraph>(graph), 0);
  }
  static int launch(void* exec, cudaStream_t st) { return cuGraphLaunch(reinterpret_cast<CUgraphExec>(exec), reinterpret_cast<CUstream>(st)); }
  static void destroy_exec(void* exec) { cuGraphExecDestroy(reinterpret_cast<CUgraphExec>(exec)); }
  static void destroy(void* graph) { cuGraphDestroy(reinterpret_cast<CUgraph>(graph)); }
  static int sync(cudaStream_t st) { return cuStreamSynchronize(reinterpret_cast<CUstream>(st)); }
  static cudaStream_t make_stream() {
    cudaFree(nullptr);   // brings up the primary context the driver stream belongs to
    CUstream s = nullptr;
    cuStreamCreate(&s, CU_STREAM_NON_BLOCKING);
    return reinterpret_cast<cudaStream_t>(s);
  }
  static void destroy_stream(cudaStream_t st) { cuStreamDestroy(reinterpret_cast<CUstream>(st)); }
#else
  static int begin(cudaStream_t st) { return cudaStreamBeginCapture(st, capture_mode); }
  static int end(cudaStream_t st, void** graph) { return cudaStreamEndCapture(st, reinterpret_cast<cudaGraph_t*>(graph)); }
  static int instantiate(void** exec, void* graph) {
    return cudaGraphInstantiate(reinterpret_cast<cudaGraphExec_t*>(exec), static_cast<cudaGraph_t>(graph), 0);
  }
  static int launch(void* exec, cudaStream_t st) { return cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec), st); }
  static void destroy_exec(void* exec) { cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(exec)); }
  static void destroy(void* graph) { cudaGraphDestroy(static_cast<cudaGraph_t>(graph)); }
  static int sync(cudaStream_t st) { return cudaStreamSynchronize(st); }
  static cudaStream_t make_stream() {
    cudaStream_t s = nullptr;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    return s;
  }
  static void destroy_stream(cudaStream_t st) { cudaStreamDestroy(st); }
#endif
};

struct Runner {
  cudaStream_t st = nullptr;
  int* counter = nullptr;
  std::vector<void*> allocs;

  Runner() {
    st = GraphApi::make_stream();
    cudaMalloc(&counter, sizeof(int));
  }
  ~Runner() {
    release();
    cudaFree(counter);
    GraphApi::destroy_stream(st);
  }
  template <class T>
  T* alloc(size_t n) {
    void* p = nullptr;
    if (cudaMalloc(&p, (n ? n : 1) * sizeof(T)) != cudaSuccess) return nullptr;
    cudaMemset(p, 0, (n ? n : 1) * sizeof(T));
    allocs.push_back(p);
    return static_cast<T*>(p);
  }
  void release() {
    for (void* p : allocs) cudaFree(p);
    allocs.clear();
  }
  void set_counter(int c) {
    cudaMemcpy(counter, &c, sizeof c, cudaMemcpyHostToDevice);
  }

  static std::vector<unsigned char> read(const Out& o) {
    std::vector<unsigned char> h(o.n * dt_bytes(o.t));
    cudaMemcpy(h.data(), o.p, h.size(), cudaMemcpyDeviceToHost);
    return h;
  }
  void zero(const std::vector<Out>& outs) {
    for (const Out& o : outs) cudaMemset(o.p, 0, o.n * dt_bytes(o.t));
    cudaDeviceSynchronize();
  }
  static double value(const unsigned char* b, Dt t) {
    switch (t) {
      case Dt::F32: { float v; std::memcpy(&v, b, 4); return v; }
      case Dt::F64: { double v; std::memcpy(&v, b, 8); return v; }
      case Dt::F16: { __half v; std::memcpy(&v, b, 2); return __half2float(v); }
      case Dt::I32: { int v; std::memcpy(&v, b, 4); return v; }
      default: return *b;
    }
  }
  // The largest difference, relative to the reference's magnitude plus one.
  static double distance(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b, Dt t) {
    double worst = 0;
    const size_t w = dt_bytes(t);
    for (size_t i = 0; i + w <= a.size(); i += w) {
      const double x = value(a.data() + i, t), y = value(b.data() + i, t);
      if (std::isnan(x) && std::isnan(y)) continue;
      if (std::isnan(x) != std::isnan(y)) return 1e30;
      worst = std::fmax(worst, std::fabs(x - y) / (1.0 + std::fabs(y)));
    }
    return worst;
  }
  static bool any_nonzero(const std::vector<unsigned char>& a) {
    for (unsigned char c : a) if (c) return true;
    return false;
  }

  // `issue` enqueues the inputs' fill kernels and the call(s) on `st` and says
  // whether every call succeeded. `check`, when given, replaces the comparison
  // with the eager run (a call that draws random numbers has no reference): it
  // is called after each graph launch with the counter, and the launch index.
  void run(const std::string& name, const std::function<bool()>& issue, const std::vector<Out>& outs,
           double tol = 1e-5, const std::function<bool(int, int)>& check = nullptr, int launches = 2) {
    const int counters[3] = {3, 6, 9};
    set_counter(counters[0]);
    zero(outs);
    const bool eager_ok = issue();
    GraphApi::sync(st);
    expect(name + ", eager", eager_ok);
    if (!eager_ok) return;
    bool produced = false;
    for (const Out& o : outs) produced = produced || any_nonzero(read(o));
    expect(name + ": the call writes something", produced);
    bool finite = true;
    for (const Out& o : outs)
      if (o.t == Dt::F32 || o.t == Dt::F64 || o.t == Dt::F16) {
        const auto b = read(o);
        for (size_t i = 0; i + dt_bytes(o.t) <= b.size(); i += dt_bytes(o.t)) finite = finite && std::isfinite(value(b.data() + i, o.t));
      }
    expect(name + ": its results are finite", finite);

    zero(outs);
    void* graph = nullptr;
    void* exec = nullptr;
    const int b = GraphApi::begin(st);
    bool rec_ok = b == 0 && issue();
    const int e = GraphApi::end(st, &graph);
    expect(name + ", recorded into a capture", rec_ok && e == 0, e);
    if (!rec_ok || e != 0) {
      if (graph) GraphApi::destroy(graph);
      cudaGetLastError();
      GraphApi::sync(st);
      return;
    }
    if (GraphApi::instantiate(&exec, graph) != 0) {
      expect(name + ", graph instantiates", false);
      GraphApi::destroy(graph);
      return;
    }
    GraphApi::sync(st);
    bool untouched = true;
    for (const Out& o : outs) untouched = untouched && !any_nonzero(read(o));
    expect(name + ": the capture did not run the call", untouched);

    for (int r = 1; r <= launches; ++r) {
      const int c = counters[r];
      set_counter(c);
      std::vector<std::vector<unsigned char>> want;
      if (!check) {
        zero(outs);
        const bool ok = issue();
        GraphApi::sync(st);
        if (!ok) { expect(name + ", reference run " + std::to_string(r), false); break; }
        for (const Out& o : outs) want.push_back(read(o));
      }
      zero(outs);
      const int l = GraphApi::launch(exec, st);
      const int s = GraphApi::sync(st);
      const std::string what = name + ", graph launch " + std::to_string(r) + " (counter " + std::to_string(c) + ")";
      if (l != 0 || s != 0) { expect(what, false, l ? l : s); break; }
      if (check) {
        expect(what, check(c, r));
        continue;
      }
      double worst = 0;
      for (size_t i = 0; i < outs.size(); ++i) worst = std::fmax(worst, distance(read(outs[i]), want[i], outs[i].t));
      expect(what, worst <= tol, worst);
    }
    GraphApi::destroy_exec(exec);
    GraphApi::destroy(graph);
    GraphApi::sync(st);
  }
};

inline int finish() {
  std::printf("%d checks\n", checks);
  std::puts(fails ? "FAIL" : "PASS");
  return fails != 0;
}

}  // namespace gc
