// A CUgraph and a cudaGraph_t are one handle, as are a CUstream and a
// cudaStream_t, a CUevent and a cudaEvent_t, a CUgraphExec and a
// cudaGraphExec_t. A program that uses both APIs -- as every framework does --
// builds a graph with one and launches it with the other, captures a stream
// one API made with kernels the other launched, and records events in nodes
// the other API made. This program does each, on kernels from nvcc (which run
// as SASS or PTX, as the runner chooses) and from PTX a module loaded; it
// passes against NVIDIA's libraries on an RTX 3060 as well.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

static int failures = 0;
static void check(bool ok, const char* what, int line) {
  if (!ok) {
    std::printf("FAIL line %d: %s\n", line, what);
    ++failures;
  }
}
#define CHECK(c) check((c), #c, __LINE__)
#define RK(call) do { const cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  std::printf("FAIL line %d: %s -> %s\n", __LINE__, #call, cudaGetErrorName(e_)); ++failures; } } while (0)
#define CK(call) do { const CUresult r_ = (call); if (r_ != CUDA_SUCCESS) { \
  std::printf("FAIL line %d: %s -> %d\n", __LINE__, #call, (int)r_); ++failures; } } while (0)

constexpr int kN = 64;

__global__ void addk(int* p, int v) { p[threadIdx.x] += v; }

static const char* kPtx = R"(
.version 7.0
.target sm_80
.address_size 64
.visible .entry addp(.param .u64 p, .param .u32 v)
{
  .reg .b32 %r<4>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd1, [p];
  ld.param.u32 %r1, [v];
  mov.u32 %r2, %tid.x;
  mul.wide.u32 %rd2, %r2, 4;
  add.u64 %rd3, %rd1, %rd2;
  ld.global.u32 %r3, [%rd3];
  add.u32 %r3, %r3, %r1;
  st.global.u32 [%rd3], %r3;
  ret;
}
)";

static int first(int* d) {
  int h[kN];
  cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  return h[0];
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  RK(cudaFree(nullptr));
  CUmodule mod;
  CUfunction drv_fn;
  CK(cuModuleLoadData(&mod, kPtx));
  CK(cuModuleGetFunction(&drv_fn, mod, "addp"));
  int* d = nullptr;
  RK(cudaMalloc(&d, kN * sizeof(int)));
  RK(cudaMemset(d, 0, kN * sizeof(int)));

  // ---- a graph built with the driver API, read and launched by the runtime ----
  CUgraph g = nullptr;
  CK(cuGraphCreate(&g, 0));
  int v5 = 5;
  void* args5[] = {&d, &v5};
  CUDA_KERNEL_NODE_PARAMS kp;
  std::memset(&kp, 0, sizeof kp);
  kp.func = drv_fn;
  kp.gridDimX = kp.gridDimY = kp.gridDimZ = 1;
  kp.blockDimX = kN;
  kp.blockDimY = kp.blockDimZ = 1;
  kp.kernelParams = args5;
  CUgraphNode k1 = nullptr;
  CK(cuGraphAddKernelNode(&k1, g, nullptr, 0, &kp));
  cudaGraphNodeType t;
  RK(cudaGraphNodeGetType(reinterpret_cast<cudaGraphNode_t>(k1), &t));
  CHECK(t == cudaGraphNodeTypeKernel);
  // The runtime has no host stub for a function a module loaded.
  cudaKernelNodeParams rp;
  std::memset(&rp, 0, sizeof rp);
  CHECK(cudaGraphKernelNodeGetParams(reinterpret_cast<cudaGraphNode_t>(k1), &rp) == cudaErrorInvalidDeviceFunction);
  cudaGetLastError();
  // The runtime adds a node to the driver's graph.
  cudaGraphNode_t en = nullptr;
  RK(cudaGraphAddEmptyNode(&en, reinterpret_cast<cudaGraph_t>(g), reinterpret_cast<cudaGraphNode_t*>(&k1), 1));
  size_t nodes = 0;
  CK(cuGraphGetNodes(g, nullptr, &nodes));
  CHECK(nodes == 2);
  cudaGraphExec_t rexec = nullptr;
  RK(cudaGraphInstantiate(&rexec, reinterpret_cast<cudaGraph_t>(g), 0));
  CK(cuGraphLaunch(reinterpret_cast<CUgraphExec>(rexec), 0));
  RK(cudaDeviceSynchronize());
  CHECK(first(d) == 5);
  CUgraphExec dexec = nullptr;
  CK(cuGraphInstantiate(&dexec, g, 0));
  RK(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(dexec), 0));
  RK(cudaDeviceSynchronize());
  CHECK(first(d) == 10);

  // ---- a graph built with the runtime, read by the driver ----
  cudaGraph_t rg = nullptr;
  RK(cudaGraphCreate(&rg, 0));
  int v7 = 7;
  void* args7[] = {&d, &v7};
  cudaKernelNodeParams rk;
  std::memset(&rk, 0, sizeof rk);
  rk.func = reinterpret_cast<void*>(addk);
  rk.gridDim = dim3(1);
  rk.blockDim = dim3(kN);
  rk.kernelParams = args7;
  cudaGraphNode_t rn = nullptr;
  RK(cudaGraphAddKernelNode(&rn, rg, nullptr, 0, &rk));
  CUDA_KERNEL_NODE_PARAMS dk;
  std::memset(&dk, 0xcd, sizeof dk);
  CK(cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(rn), &dk));
  CHECK(dk.func != nullptr && dk.blockDimX == kN && dk.gridDimX == 1);
  CHECK(dk.kernelParams && *static_cast<int*>(dk.kernelParams[1]) == 7);
  CUgraphExec rg_exec = nullptr;
  CK(cuGraphInstantiate(&rg_exec, reinterpret_cast<CUgraph>(rg), 0));
  CK(cuGraphLaunch(rg_exec, 0));
  CK(cuCtxSynchronize());
  CHECK(first(d) == 17);
  // Either API destroys what the other made.
  RK(cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(g)));
  CK(cuGraphDestroy(reinterpret_cast<CUgraph>(rg)));
  CK(cuGraphExecDestroy(reinterpret_cast<CUgraphExec>(rexec)));
  RK(cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(rg_exec)));
  CHECK(cuGraphDestroy(g) == CUDA_ERROR_INVALID_VALUE);

  // The driver's failures are not the runtime's last error.
  RK(cudaGetLastError());
  CHECK(cuGraphCreate(nullptr, 0) == CUDA_ERROR_INVALID_VALUE);
  CHECK(cudaGetLastError() == cudaSuccess);

  // ---- capture ----
  // A stream the runtime made, captured through the driver, with a runtime kernel.
  cudaStream_t rs;
  RK(cudaStreamCreate(&rs));
  RK(cudaMemset(d, 0, kN * sizeof(int)));
  CK(cuStreamBeginCapture(reinterpret_cast<CUstream>(rs), CU_STREAM_CAPTURE_MODE_GLOBAL));
  cudaStreamCaptureStatus st;
  RK(cudaStreamIsCapturing(rs, &st));
  CHECK(st == cudaStreamCaptureStatusActive);
  addk<<<1, kN, 0, rs>>>(d, 1);
  CK(cuLaunchKernel(drv_fn, 1, 1, 1, kN, 1, 1, 0, reinterpret_cast<CUstream>(rs), args5, nullptr));
  CUgraph cg = nullptr;
  CK(cuStreamEndCapture(reinterpret_cast<CUstream>(rs), &cg));
  CHECK(first(d) == 0);   // neither ran
  CK(cuGraphGetNodes(cg, nullptr, &nodes));
  CHECK(nodes == 2);
  CUgraphExec cexec = nullptr;
  CK(cuGraphInstantiate(&cexec, cg, 0));
  RK(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(cexec), rs));
  RK(cudaStreamSynchronize(rs));
  CHECK(first(d) == 6);

  // A stream the driver made, captured through the runtime.
  CUstream ds;
  CK(cuStreamCreate(&ds, 0));
  RK(cudaMemset(d, 0, kN * sizeof(int)));
  RK(cudaStreamBeginCapture(reinterpret_cast<cudaStream_t>(ds), cudaStreamCaptureModeGlobal));
  CK(cuStreamIsCapturing(ds, reinterpret_cast<CUstreamCaptureStatus*>(&st)));
  CHECK(st == cudaStreamCaptureStatusActive);
  addk<<<1, kN, 0, reinterpret_cast<cudaStream_t>(ds)>>>(d, 2);
  CK(cuLaunchKernel(drv_fn, 1, 1, 1, kN, 1, 1, 0, ds, args5, nullptr));
  CK(cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(d) + 4, 9, 1, ds));
  cudaGraph_t rcg = nullptr;
  RK(cudaStreamEndCapture(reinterpret_cast<cudaStream_t>(ds), &rcg));
  size_t rnodes = 0;
  RK(cudaGraphGetNodes(rcg, nullptr, &rnodes));
  CHECK(rnodes == 3);
  cudaGraphExec_t rce = nullptr;
  RK(cudaGraphInstantiate(&rce, rcg, 0));
  CK(cuGraphLaunch(reinterpret_cast<CUgraphExec>(rce), ds));
  CK(cuStreamSynchronize(ds));
  int h[kN];
  RK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
  CHECK(h[0] == 7 && h[1] == 9);

  // Events: one stream joins another's capture through a driver event, the way a
  // framework's backward pass forks.
  CUevent fork, join;
  CK(cuEventCreate(&fork, CU_EVENT_DISABLE_TIMING));
  CK(cuEventCreate(&join, CU_EVENT_DISABLE_TIMING));
  cudaStream_t rs2;
  RK(cudaStreamCreate(&rs2));
  RK(cudaMemset(d, 0, kN * sizeof(int)));
  RK(cudaStreamBeginCapture(rs, cudaStreamCaptureModeGlobal));
  addk<<<1, kN, 0, rs>>>(d, 1);
  CK(cuEventRecord(fork, reinterpret_cast<CUstream>(rs)));
  CK(cuStreamWaitEvent(reinterpret_cast<CUstream>(rs2), fork, 0));
  addk<<<1, kN, 0, rs2>>>(d, 1);
  CK(cuEventRecord(join, reinterpret_cast<CUstream>(rs2)));
  CK(cuStreamWaitEvent(reinterpret_cast<CUstream>(rs), join, 0));
  cudaGraph_t forked = nullptr;
  RK(cudaStreamEndCapture(rs, &forked));
  RK(cudaGraphGetNodes(forked, nullptr, &rnodes));
  CHECK(rnodes == 2);

  // An event node on a driver event, recorded each time a runtime-built graph runs.
  CUevent timed;
  CK(cuEventCreate(&timed, 0));
  cudaGraph_t eg;
  RK(cudaGraphCreate(&eg, 0));
  cudaGraphNode_t e1 = nullptr;
  RK(cudaGraphAddEventRecordNode(&e1, eg, nullptr, 0, reinterpret_cast<cudaEvent_t>(timed)));
  cudaGraphExec_t ee;
  RK(cudaGraphInstantiate(&ee, eg, 0));
  RK(cudaGraphLaunch(ee, 0));
  RK(cudaDeviceSynchronize());
  float ms = -1;
  CK(cuEventElapsedTime(&ms, timed, timed));   // recorded: it can be timed against itself
  CHECK(ms >= 0.0f);

  std::printf(failures ? "FAIL: %d checks\n" : "PASS: the two APIs share graphs, streams and events\n", failures);
  return failures ? 1 : 0;
}
