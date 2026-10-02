// Device-side graph launch (CUDA 12): kernels that launch graphs instantiated
// with cudaGraphInstantiateFlagDeviceLaunch, and %current_graph_exec
// (cudaGetCurrentGraphExec). Built -rdc=true and linked with cudadevrt, as a
// program using them is. Every expected value below is what an RTX 3060
// (sm_86, CUDA 13.0) gave, and the program passes there as on the simulator.
//
// 1. cudaGetCurrentGraphExec: 0 outside a graph and in a graph not
//    instantiated for device launch; the executable graph's own handle in a
//    device graph however it was launched, a child graph's kernel included.
// 2. Fire-and-forget, tail and sibling launches, and when each has run: all
//    before the host launch is complete, tail launches in the guide's order.
// 3. A tail self-launch loop, and the one-at-a-time rule for it.
// 4. What a device launch refuses (cudaErrorInvalidValue): the wrong streams,
//    a null handle, a graph not uploaded, a graph already running or queued,
//    more than 120 fire-and-forget or 255 tail launches -- and that a refused
//    launch leaves the device-side last error alone.
// 5. What instantiation for device launch refuses, and what it accepts.
// 6. Upload: a device launch runs the graph as last uploaded.
// 7. A device launch of a graph not instantiated for device launch faults
//    ("host-handle" argument); a kernel that launches graphs, launched outside
//    any graph, is refused.
//
// Prints PASS on the last line.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef cudaStreamGraphFireAndForgetAsSibling   // CUDA 12.0's header has the other two
#define cudaStreamGraphFireAndForgetAsSibling (cudaStream_t)0x0300000000000000
#endif

static int fails = 0;
#define CK(x)                                                                                  \
  do {                                                                                         \
    cudaError_t e_ = (x);                                                                      \
    if (e_ != cudaSuccess) {                                                                   \
      std::printf("FAIL: %s: %s (line %d)\n", #x, cudaGetErrorName(e_), __LINE__);            \
      ++fails;                                                                                 \
    }                                                                                          \
  } while (0)
#define WANT(got, want)                                                                        \
  do {                                                                                         \
    long long g_ = (long long)(got), w_ = (long long)(want);                                   \
    if (g_ != w_) {                                                                            \
      std::printf("FAIL: %s is %lld, want %lld (line %d)\n", #got, g_, w_, __LINE__);         \
      ++fails;                                                                                 \
    }                                                                                          \
  } while (0)

__device__ int g_log[512];
__device__ int g_n;
__device__ unsigned long long g_cur[16];
__device__ int g_err[16];

// Logs its id, and the graph it runs in under that id.
__global__ void logk(int id) {
  g_log[atomicAdd(&g_n, 1)] = id;
  g_cur[id & 15] = (unsigned long long)cudaGetCurrentGraphExec();
}
__global__ void launch1(cudaGraphExec_t g, cudaStream_t s, int slot) { g_err[slot] = cudaGraphLaunch(g, s); }
__global__ void launch2(cudaGraphExec_t a, cudaGraphExec_t b, cudaStream_t s, int slot, int id) {
  g_log[atomicAdd(&g_n, 1)] = id;
  g_err[slot] = cudaGraphLaunch(a, s);
  g_err[slot + 1] = cudaGraphLaunch(b, s);
}
// Launches g[0..n) into s; records the first that failed and its error.
__global__ void launch_many(cudaGraphExec_t* g, cudaStream_t s, int n, int slot) {
  int first = -1, err = 0;
  for (int i = 0; i < n; ++i) {
    const int e = cudaGraphLaunch(g[i], s);
    if (e && first < 0) { first = i; err = e; }
  }
  g_err[slot] = first;
  g_err[slot + 1] = err;
}
// Relaunches its own graph by tail launch until it has run `times` times; the
// second self-launch of one run is refused.
__global__ void self_tail(int times) {
  g_log[atomicAdd(&g_n, 1)] = 7;
  if (g_n < times) {
    g_err[0] = cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
    g_err[1] = cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
  }
}
__global__ void self_fire_and_forget() {
  g_err[0] = cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphFireAndForget);
}
__global__ void launch_then_last_error(cudaGraphExec_t g) {
  g_err[0] = cudaGraphLaunch(g, cudaStreamGraphFireAndForget);
  g_err[1] = cudaGetLastError();
  g_err[2] = cudaPeekAtLastError();
}
__global__ void child_k() {}
__global__ void launches_a_kernel() { child_k<<<1, 1>>>(); }

static cudaStream_t s;
static const unsigned long long DL = cudaGraphInstantiateFlagDeviceLaunch;
static cudaStream_t FAF = cudaStreamGraphFireAndForget, TAIL = cudaStreamGraphTailLaunch,
                    SIB = cudaStreamGraphFireAndForgetAsSibling;

static cudaGraph_t one_kernel(const void* f, void** args) {
  cudaGraph_t g;
  CK(cudaGraphCreate(&g, 0));
  cudaKernelNodeParams p = {};
  p.func = const_cast<void*>(f);
  p.gridDim = dim3(1);
  p.blockDim = dim3(1);
  p.kernelParams = args;
  cudaGraphNode_t n;
  CK(cudaGraphAddKernelNode(&n, g, nullptr, 0, &p));
  return g;
}
static cudaGraphExec_t inst(cudaGraph_t g, unsigned long long flags, bool upload) {
  cudaGraphExec_t e = nullptr;
  CK(cudaGraphInstantiate(&e, g, flags));
  if (upload) CK(cudaGraphUpload(e, s));
  return e;
}
static cudaGraphExec_t log_graph(int id, unsigned long long flags, bool upload) {
  void* a[] = {&id};
  return inst(one_kernel((const void*)logk, a), flags, upload);
}
// A host graph whose one kernel launches `g` into `st`, its result in g_err[slot].
static cudaGraphExec_t launcher(cudaGraphExec_t g, cudaStream_t st, int slot, unsigned long long flags = 0) {
  void* a[] = {&g, &st, &slot};
  return inst(one_kernel((const void*)launch1, a), flags, flags != 0);
}
static void reset() {
  int z[512] = {};
  unsigned long long c[16] = {};
  CK(cudaMemcpyToSymbol(g_n, z, 4));
  CK(cudaMemcpyToSymbol(g_log, z, sizeof g_log));
  CK(cudaMemcpyToSymbol(g_err, z, sizeof g_err));
  CK(cudaMemcpyToSymbol(g_cur, c, sizeof c));
}
static int n_logged() {
  int n = 0;
  CK(cudaMemcpyFromSymbol(&n, g_n, 4));
  return n;
}
static void logged(int* out, int n) { CK(cudaMemcpyFromSymbol(out, g_log, sizeof(int) * n)); }
static int err(int i) {
  int e[16];
  CK(cudaMemcpyFromSymbol(e, g_err, sizeof e));
  return e[i];
}
static unsigned long long cur(int id) {
  unsigned long long c[16];
  CK(cudaMemcpyFromSymbol(c, g_cur, sizeof c));
  return c[id & 15];
}
static void run(cudaGraphExec_t g) {
  CK(cudaGraphLaunch(g, s));
  CK(cudaStreamSynchronize(s));
}
// The log, as a string of ids, to compare orders.
static void want_log(const char* want, int line) {
  int n = n_logged(), l[64] = {};
  logged(l, n < 64 ? n : 64);
  char got[256] = "";
  for (int i = 0; i < n && i < 64; ++i) std::snprintf(got + std::strlen(got), sizeof got - std::strlen(got), "%s%d", i ? " " : "", l[i]);
  if (std::strcmp(got, want) != 0) {
    std::printf("FAIL: log is \"%s\", want \"%s\" (line %d)\n", got, want, line);
    ++fails;
  }
}
#define WANT_LOG(w) want_log(w, __LINE__)

int main(int argc, char** argv) {
  CK(cudaStreamCreate(&s));
  reset();

  if (argc > 1 && std::strcmp(argv[1], "host-handle") == 0) {
    // A device launch of a graph that was not instantiated for device launch:
    // the card does not return an error, the kernel faults.
    cudaGraphExec_t h = log_graph(1, 0, false);
    CK(cudaGraphLaunch(launcher(h, FAF, 0), s));
    WANT(cudaStreamSynchronize(s), cudaErrorIllegalAddress);
    std::printf(fails ? "FAIL\n" : "PASS\n");
    return fails != 0;
  }

  // ---- 1. cudaGetCurrentGraphExec ---------------------------------------------
  {
    logk<<<1, 1, 0, s>>>(0);
    cudaGraphExec_t host = log_graph(1, 0, false);
    run(host);
    cudaGraphExec_t dev = log_graph(2, DL, true);
    run(dev);
    cudaGraphExec_t dev3 = log_graph(3, DL, true);
    run(launcher(dev3, FAF, 0));
    // A child graph's kernel, inside a device graph: the outer graph's handle.
    int id = 4;
    void* a[] = {&id};
    cudaGraph_t outer;
    cudaGraphNode_t n;
    CK(cudaGraphCreate(&outer, 0));
    CK(cudaGraphAddChildGraphNode(&n, outer, nullptr, 0, one_kernel((const void*)logk, a)));
    cudaGraphExec_t dev4 = inst(outer, DL, true);
    run(launcher(dev4, FAF, 1));
    WANT(err(0), 0);
    WANT(err(1), 0);
    WANT_LOG("0 1 2 3 4");
    WANT(cur(0), 0);
    WANT(cur(1), 0);
    WANT(cur(2), (unsigned long long)dev);
    WANT(cur(3), (unsigned long long)dev3);
    WANT(cur(4), (unsigned long long)dev4);
  }

  // ---- 2. launch modes ----------------------------------------------------------
  {
    // Tail launches run once the launching graph is done, in order; those a
    // tail graph queues run before the ones queued ahead of it: A queues B
    // then C, B queues D then E, and they run A B D E C.
    reset();
    cudaGraphExec_t C = log_graph(3, DL, true), D = log_graph(4, DL, true), E = log_graph(5, DL, true);
    int slot_b = 4, id_b = 2;
    void* ab[] = {&D, &E, &TAIL, &slot_b, &id_b};
    cudaGraphExec_t B = inst(one_kernel((const void*)launch2, ab), DL, true);
    int slot_a = 0, id_a = 1;
    void* aa[] = {&B, &C, &TAIL, &slot_a, &id_a};
    cudaGraphExec_t A = inst(one_kernel((const void*)launch2, aa), 0, false);
    run(A);
    WANT_LOG("1 2 4 5 3");
    for (int i : {0, 1, 4, 5}) WANT(err(i), 0);
    WANT(cur(3), (unsigned long long)C);

    // Each mode's work is done before the stream's next work starts. (The
    // order of two fire-and-forget or sibling graphs is the card's to pick,
    // so only what ran is checked.)
    for (cudaStream_t mode : {FAF, SIB, TAIL}) {
      reset();
      cudaGraphExec_t x = log_graph(2, DL, true), y = log_graph(3, DL, true);
      int slot = 0, id = 1;
      void* a[] = {&x, &y, &mode, &slot, &id};
      CK(cudaGraphLaunch(inst(one_kernel((const void*)launch2, a), 0, false), s));
      logk<<<1, 1, 0, s>>>(9);
      CK(cudaStreamSynchronize(s));
      int l[4] = {};
      WANT(n_logged(), 4);
      logged(l, 4);
      WANT(l[0], 1);
      WANT(l[1] + l[2], 5);
      WANT(l[3], 9);
      if (mode == TAIL) WANT_LOG("1 2 3 9");
      WANT(err(0), 0);
      WANT(err(1), 0);
    }

    // A device graph whose kernel tail-launches a graph that launches another
    // fire-and-forget.
    reset();
    cudaGraphExec_t inner = log_graph(8, DL, true);
    cudaGraphExec_t mid = launcher(inner, FAF, 3, DL);
    run(launcher(mid, TAIL, 4, DL));
    WANT_LOG("8");
    WANT(err(3), 0);
    WANT(err(4), 0);
    WANT(cur(8), (unsigned long long)inner);
  }

  // ---- 3. tail self-launch ------------------------------------------------------
  {
    reset();
    int times = 4;
    void* a[] = {&times};
    cudaGraphExec_t g = inst(one_kernel((const void*)self_tail, a), DL, true);
    run(launcher(g, FAF, 2));
    WANT_LOG("7 7 7 7");
    WANT(err(0), 0);
    WANT(err(1), cudaErrorInvalidValue);   // only one self-launch queued at a time
    WANT(err(2), 0);
    reset();
    run(g);                                // launched from the host, the same
    WANT_LOG("7 7 7 7");
    WANT(err(1), cudaErrorInvalidValue);
  }

  // ---- 4. what a device launch refuses ------------------------------------------
  {
    reset();
    cudaGraphExec_t d = log_graph(1, DL, true), not_uploaded = log_graph(2, DL, false);
    cudaStream_t user;
    CK(cudaStreamCreate(&user));
    run(launcher(not_uploaded, FAF, 0));
    run(launcher(nullptr, FAF, 1));
    run(launcher(d, (cudaStream_t)0, 2));
    run(launcher(d, user, 3));
    run(launcher(d, cudaStreamPerThread, 4));
    run(launcher(d, cudaStreamTailLaunch, 5));      // dynamic parallelism's names
    run(launcher(d, cudaStreamFireAndForget, 6));
    run(launcher(d, (cudaStream_t)0x0400000000000000ull, 7));
    for (int i = 0; i < 8; ++i) WANT(err(i), cudaErrorInvalidValue);
    WANT(n_logged(), 0);

    // A graph launching itself fire-and-forget is already running.
    reset();
    run(inst(one_kernel((const void*)self_fire_and_forget, nullptr), DL, true));
    WANT(err(0), cudaErrorInvalidValue);
    // One graph queued twice from one kernel: the second is refused, in
    // either mode.
    for (cudaStream_t mode : {FAF, TAIL}) {
      reset();
      cudaGraphExec_t same[2] = {d, d};
      cudaGraphExec_t* dd;
      CK(cudaMalloc(&dd, sizeof same));
      CK(cudaMemcpy(dd, same, sizeof same, cudaMemcpyHostToDevice));
      int n = 2, slot = 0;
      void* a[] = {&dd, &mode, &n, &slot};
      cudaGraphExec_t o = inst(one_kernel((const void*)launch_many, a), 0, false);
      run(o);
      WANT(err(0), 1);
      WANT(err(1), cudaErrorInvalidValue);
      WANT(n_logged(), 1);
      CK(cudaFree(dd));
    }
    // 120 fire-and-forget launches in one execution, and 255 queued tail
    // launches; the count starts again with the next launch.
    for (cudaStream_t mode : {FAF, TAIL}) {
      const int n = mode == FAF ? 125 : 260, limit = mode == FAF ? 120 : 255;
      static cudaGraphExec_t many[260];
      for (int i = 0; i < n; ++i) many[i] = log_graph(1, DL, true);
      cudaGraphExec_t* dd;
      CK(cudaMalloc(&dd, sizeof many));
      CK(cudaMemcpy(dd, many, sizeof many, cudaMemcpyHostToDevice));
      int count = n, slot = 0;
      void* a[] = {&dd, &mode, &count, &slot};
      cudaGraphExec_t o = inst(one_kernel((const void*)launch_many, a), 0, false);
      for (int again = 0; again < 2; ++again) {
        reset();
        run(o);
        WANT(err(0), limit);
        WANT(err(1), cudaErrorInvalidValue);
        WANT(n_logged(), limit);
      }
      CK(cudaFree(dd));
    }
    // A refused launch is a return value only.
    reset();
    void* a[] = {&not_uploaded};
    run(inst(one_kernel((const void*)launch_then_last_error, a), 0, false));
    WANT(err(0), cudaErrorInvalidValue);
    WANT(err(1), cudaSuccess);
    WANT(err(2), cudaSuccess);
    CK(cudaStreamDestroy(user));
  }

  // ---- 5. instantiation for device launch ---------------------------------------
  {
    cudaGraph_t g;
    cudaGraphNode_t n;
    cudaGraphExec_t e;
    int id = 1;
    void* a[] = {&id};
    const auto refused = [&](cudaGraph_t graph, unsigned long long flags, int line) {
      const cudaError_t rc = cudaGraphInstantiate(&e, graph, flags);
      if (rc != cudaErrorInvalidValue) {
        std::printf("FAIL: instantiate gave %s, want cudaErrorInvalidValue (line %d)\n", cudaGetErrorName(rc), line);
        ++fails;
      }
    };
    const auto accepted = [&](cudaGraph_t graph, int line) {
      const cudaError_t rc = cudaGraphInstantiate(&e, graph, DL);
      if (rc != cudaSuccess) {
        std::printf("FAIL: instantiate gave %s, want cudaSuccess (line %d)\n", cudaGetErrorName(rc), line);
        ++fails;
      }
    };
    CK(cudaGraphCreate(&g, 0));
    refused(g, DL, __LINE__);                                   // nothing in it
    CK(cudaGraphAddEmptyNode(&n, g, nullptr, 0));
    refused(g, DL, __LINE__);                                   // an empty node
    cudaGraph_t with_empty_child;
    CK(cudaGraphCreate(&with_empty_child, 0));
    CK(cudaGraphAddChildGraphNode(&n, with_empty_child, nullptr, 0, g));
    refused(with_empty_child, DL, __LINE__);                    // ... in a child graph too
    CK(cudaGraphCreate(&g, 0));
    cudaHostNodeParams hp = {};
    hp.fn = [](void*) {};
    CK(cudaGraphAddHostNode(&n, g, nullptr, 0, &hp));
    refused(g, DL, __LINE__);                                   // a host node
    cudaEvent_t ev;
    CK(cudaEventCreate(&ev));
    CK(cudaGraphCreate(&g, 0));
    CK(cudaGraphAddEventRecordNode(&n, g, nullptr, 0, ev));
    refused(g, DL, __LINE__);                                   // an event node
    refused(one_kernel((const void*)launches_a_kernel, nullptr), DL, __LINE__);   // dynamic parallelism
    int* dev;
    int* pinned;
    int* managed;
    static int pageable[16];
    CK(cudaMalloc(&dev, 64));
    CK(cudaMallocHost(&pinned, 64));
    CK(cudaMallocManaged(&managed, 64));
    CK(cudaGraphCreate(&g, 0));
    CK(cudaGraphAddMemcpyNode1D(&n, g, nullptr, 0, dev, pageable, 64, cudaMemcpyHostToDevice));
    refused(g, DL, __LINE__);                                   // pageable memory
    CK(cudaGraphCreate(&g, 0));
    CK(cudaGraphAddMemcpyNode1D(&n, g, nullptr, 0, dev, managed, 64, cudaMemcpyDefault));
    refused(g, DL, __LINE__);                                   // managed memory
    refused(one_kernel((const void*)logk, a), DL | cudaGraphInstantiateFlagAutoFreeOnLaunch, __LINE__);
    refused(one_kernel((const void*)logk, a), DL | cudaGraphInstantiateFlagUpload, __LINE__);
    refused(one_kernel((const void*)logk, a), cudaGraphInstantiateFlagUpload, __LINE__);   // WithParams' flag

    CK(cudaGraphCreate(&g, 0));
    CK(cudaGraphAddMemcpyNode1D(&n, g, nullptr, 0, dev, pinned, 64, cudaMemcpyHostToDevice));
    accepted(g, __LINE__);
    CK(cudaGraphCreate(&g, 0));
    CK(cudaGraphAddMemcpyNode1D(&n, g, nullptr, 0, dev + 8, dev, 32, cudaMemcpyDeviceToDevice));
    accepted(g, __LINE__);
    cudaMemsetParams mp = {};
    mp.dst = pinned;
    mp.value = 1;
    mp.elementSize = 4;
    mp.width = 4;
    mp.height = 1;
    CK(cudaGraphCreate(&g, 0));
    CK(cudaGraphAddMemsetNode(&n, g, nullptr, 0, &mp));
    accepted(g, __LINE__);
    CK(cudaGraphCreate(&g, 0));
    CK(cudaGraphAddChildGraphNode(&n, g, nullptr, 0, one_kernel((const void*)logk, a)));
    accepted(g, __LINE__);
    cudaGetLastError();
  }

  // ---- 6. upload ----------------------------------------------------------------
  {
    // A graph changed by cudaGraphExecUpdate runs from the device as it was
    // last uploaded.
    reset();
    cudaGraphExec_t d = log_graph(1, DL, true);
    int id = 2;
    void* a[] = {&id};
    cudaGraphExecUpdateResultInfo info;
    CK(cudaGraphExecUpdate(d, one_kernel((const void*)logk, a), &info));
    cudaGraphExec_t o = launcher(d, FAF, 0);
    run(o);
    WANT_LOG("1");
    reset();
    CK(cudaGraphUpload(d, s));
    run(o);
    WANT_LOG("2");
    // Uploaded by cudaGraphInstantiateWithParams, and by a host launch.
    reset();
    cudaGraphInstantiateParams ip = {};
    ip.flags = DL | cudaGraphInstantiateFlagUpload;
    ip.uploadStream = s;
    id = 6;
    cudaGraphExec_t w;
    CK(cudaGraphInstantiateWithParams(&w, one_kernel((const void*)logk, a), &ip));
    WANT(ip.result_out, cudaGraphInstantiateSuccess);
    run(launcher(w, FAF, 0));
    WANT_LOG("6");
    reset();
    cudaGraphExec_t h = log_graph(5, DL, false);
    run(h);
    run(launcher(h, FAF, 0));
    WANT_LOG("5 5");
    WANT(err(0), 0);
    WANT(cudaGraphUpload(nullptr, s), cudaErrorInvalidValue);
  }

  // ---- 7. from a kernel outside any graph ----------------------------------------
  //
  // A kernel that launches graphs, launched on its own: the card refuses the
  // launch (cudaErrorNotSupported), the kernel does not run, and the context
  // lives on. Kernels of the same module that do not launch graphs still run.
  {
    reset();
    cudaGraphExec_t d = log_graph(1, DL, true);
    CK(cudaStreamSynchronize(s));
    launch1<<<1, 1, 0, s>>>(d, FAF, 0);
    WANT(cudaGetLastError(), cudaErrorNotSupported);
    CK(cudaStreamSynchronize(s));
    logk<<<1, 1, 0, s>>>(3);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(s));
    WANT_LOG("3");
  }

  std::printf(fails ? "FAIL\n" : "PASS\n");
  return fails != 0;
}
