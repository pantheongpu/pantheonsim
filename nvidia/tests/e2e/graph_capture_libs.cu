// cuBLASLt's fused matmul, cuDNN's graph-API convolution and a kernel launched
// through the driver API (cuLaunchKernel, as Triton does) inside a captured
// CUDA graph, as PyTorch's CUDA graphs (torch.cuda.graph, make_graphed_callables,
// torch.compile mode="reduce-overhead") use them.
//
// On NVIDIA's libraries a call made while its stream is capturing is recorded,
// not run: it executes at each launch of the graph, over whatever the graph's
// own kernels have written by then. So, for each library:
//   1. the call is made once eagerly and checked (so the call itself is right);
//   2. a kernel that writes the input from a counter in device memory, and then
//      the library call, are captured; the descriptors, plan and variant pack are
//      destroyed right after the capture (callers do, and the graph must not
//      depend on them);
//   3. the counter is changed and the graph launched several times; every launch
//      must produce the result for the counter it saw (a call that ran during
//      capture would give the result of the stale input on every launch).
// Everything is checked against a host reference. The same program passes
// against NVIDIA's libraries on a card (run_graph_capture_libs.sh --card), which
// is what these expectations were written from. Prints PASS on the last line.
#include <cublasLt.h>
#include <cuda.h>
#include <cudnn.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int fails = 0;
static void expect(const char* what, bool ok, double detail = 0) {
  std::printf("%s %s", ok ? "ok  " : "FAIL", what);
  if (!ok && detail != 0) std::printf(" (%g)", detail);
  std::printf("\n");
  fails += !ok;
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  std::printf("FAIL %s -> %s\n", #x, cudaGetErrorString(e_)); ++fails; return; } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  std::printf("FAIL %s -> %d\n", #x, (int)s_); ++fails; return; } } while (0)
#define CD(x) do { cudnnStatus_t s_ = (x); if (s_ != CUDNN_STATUS_SUCCESS) { \
  std::printf("FAIL %s -> %d\n", #x, (int)s_); ++fails; return; } } while (0)

// Input element i for a counter value c: small values, exact in float.
__global__ void fill_from_counter(float* p, int n, const int* counter, float scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = scale * static_cast<float>(((i * 7 + 3 * *counter) % 11) - 5);
}
static float host_fill(int i, int c, float scale) { return scale * static_cast<float>(((i * 7 + 3 * c) % 11) - 5); }

/* ---- cuBLASLt ------------------------------------------------------------- */

static void lt_matmul(cudaStream_t st) {
  // D(m x n) = A(m x k) * B(k x n) + bias(m), column-major, as PyTorch's addmm.
  const int m = 6, n = 5, k = 7;
  cublasLtHandle_t lt;
  CB(cublasLtCreate(&lt));
  float *A, *B, *D, *bias;
  int* counter;
  CK(cudaMalloc(&A, m * k * 4));
  CK(cudaMalloc(&B, k * n * 4));
  CK(cudaMalloc(&D, m * n * 4));
  CK(cudaMalloc(&bias, m * 4));
  CK(cudaMalloc(&counter, 4));
  std::vector<float> hb(k * n), hbias(m);
  for (int i = 0; i < k * n; ++i) hb[i] = 0.25f * static_cast<float>((i % 5) - 2);
  for (int i = 0; i < m; ++i) hbias[i] = 0.5f * static_cast<float>(i);
  CK(cudaMemcpy(B, hb.data(), k * n * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(bias, hbias.data(), m * 4, cudaMemcpyHostToDevice));

  auto reference = [&](int c) {
    std::vector<float> d(m * n);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        double acc = 0;
        for (int p = 0; p < k; ++p) acc += static_cast<double>(host_fill(i + p * m, c, 0.5f)) * hb[p + j * k];
        d[i + j * m] = static_cast<float>(acc + hbias[i]);
      }
    return d;
  };
  auto check = [&](const char* what, int c) {
    CK(cudaStreamSynchronize(st));
    std::vector<float> got(m * n);
    CK(cudaMemcpy(got.data(), D, m * n * 4, cudaMemcpyDeviceToHost));
    const auto want = reference(c);
    double d = 0;
    for (int i = 0; i < m * n; ++i) d = std::fmax(d, std::fabs(got[i] - want[i]));
    expect(what, d < 1e-4, d);
  };

  // The descriptors are built in a function the caller leaves before the graph
  // is launched.
  auto record = [&]() -> cudaError_t {
    cublasLtMatmulDesc_t md;
    cublasLtMatrixLayout_t la, lb, ld;
    cublasLtMatmulDescCreate(&md, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(md, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof ep);
    cublasLtMatmulDescSetAttribute(md, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof bias);
    cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, m, k, m);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, k, n, k);
    cublasLtMatrixLayoutCreate(&ld, CUDA_R_32F, m, n, m);
    fill_from_counter<<<1, 64, 0, st>>>(A, m * k, counter, 0.5f);
    float alpha = 1.0f, beta = 0.0f;   // locals: gone when this returns
    const cublasStatus_t s = cublasLtMatmul(lt, md, &alpha, A, la, B, lb, &beta, D, ld, D, ld, nullptr, nullptr, 0, st);
    cublasLtMatmulDescDestroy(md);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(ld);
    return s == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
  };

  int c = 2;
  CK(cudaMemcpy(counter, &c, 4, cudaMemcpyHostToDevice));
  if (record() != cudaSuccess) { expect("cuBLASLt matmul + bias, eager", false); return; }
  check("cuBLASLt matmul + bias, eager", c);

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  CK(cudaMemset(D, 0, m * n * 4));
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  const cudaError_t rec = record();
  const cudaError_t end = cudaStreamEndCapture(st, &graph);
  expect("cuBLASLt matmul + bias, recorded into a capture", rec == cudaSuccess && end == cudaSuccess, end);
  if (rec != cudaSuccess || end != cudaSuccess) return;
  CK(cudaGraphInstantiate(&exec, graph, 0));
  // Nothing ran during the capture.
  CK(cudaStreamSynchronize(st));
  std::vector<float> zero(m * n);
  CK(cudaMemcpy(zero.data(), D, m * n * 4, cudaMemcpyDeviceToHost));
  bool untouched = true;
  for (float v : zero) untouched = untouched && v == 0.0f;
  expect("cuBLASLt matmul: the capture did not run the call", untouched);
  for (int r = 0; r < 3; ++r) {
    c = 5 + 4 * r;
    CK(cudaMemcpy(counter, &c, 4, cudaMemcpyHostToDevice));
    CK(cudaGraphLaunch(exec, st));
    char what[96];
    std::snprintf(what, sizeof what, "cuBLASLt matmul + bias, graph launch %d (counter %d)", r, c);
    check(what, c);
  }
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cublasLtDestroy(lt);
  cudaFree(A), cudaFree(B), cudaFree(D), cudaFree(bias), cudaFree(counter);
}

/* ---- cuDNN's graph API ------------------------------------------------------ */

using Desc = cudnnBackendDescriptor_t;
static Desc make(cudnnBackendDescriptorType_t t) {
  Desc d = nullptr;
  cudnnBackendCreateDescriptor(t, &d);
  return d;
}
static void set(Desc d, cudnnBackendAttributeName_t n, cudnnBackendAttributeType_t t, int64_t c, const void* v) {
  cudnnBackendSetAttribute(d, n, t, c, v);
}
static Desc tensor(int64_t uid, const std::vector<int64_t>& dims) {
  Desc t = make(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
  std::vector<int64_t> strides(dims.size(), 1);
  for (size_t i = dims.size() - 1; i-- > 0;) strides[i] = strides[i + 1] * dims[i + 1];
  const cudnnDataType_t f = CUDNN_DATA_FLOAT;
  const int64_t align = 16;
  set(t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &f);
  set(t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, (int64_t)dims.size(), dims.data());
  set(t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, (int64_t)strides.size(), strides.data());
  set(t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
  set(t, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align);
  cudnnBackendFinalize(t);
  return t;
}

static void cudnn_conv(cudaStream_t st) {
  // y = conv(x, w): x 2x3x6x6, w 4x3x3x3, padding 1, stride 1.
  const std::vector<int64_t> xd{2, 3, 6, 6}, wd{4, 3, 3, 3}, yd{2, 4, 6, 6};
  const int nx = 2 * 3 * 6 * 6, nw = 4 * 3 * 3 * 3, ny = 2 * 4 * 6 * 6;
  cudnnHandle_t h;
  CD(cudnnCreate(&h));
  CD(cudnnSetStream(h, st));
  float *x, *w, *y;
  int* counter;
  CK(cudaMalloc(&x, nx * 4));
  CK(cudaMalloc(&w, nw * 4));
  CK(cudaMalloc(&y, ny * 4));
  CK(cudaMalloc(&counter, 4));
  std::vector<float> hw(nw);
  for (int i = 0; i < nw; ++i) hw[i] = 0.125f * static_cast<float>((i % 7) - 3);
  CK(cudaMemcpy(w, hw.data(), nw * 4, cudaMemcpyHostToDevice));

  auto reference = [&](int c) {
    std::vector<float> out(ny);
    for (int n = 0; n < 2; ++n)
      for (int k = 0; k < 4; ++k)
        for (int oh = 0; oh < 6; ++oh)
          for (int ow = 0; ow < 6; ++ow) {
            double acc = 0;
            for (int ch = 0; ch < 3; ++ch)
              for (int fh = 0; fh < 3; ++fh)
                for (int fw = 0; fw < 3; ++fw) {
                  const int ih = oh - 1 + fh, iw = ow - 1 + fw;
                  if (ih < 0 || ih >= 6 || iw < 0 || iw >= 6) continue;
                  acc += static_cast<double>(host_fill(((n * 3 + ch) * 6 + ih) * 6 + iw, c, 0.25f)) * hw[((k * 3 + ch) * 3 + fh) * 3 + fw];
                }
            out[((n * 4 + k) * 6 + oh) * 6 + ow] = static_cast<float>(acc);
          }
    return out;
  };
  auto check = [&](const char* what, int c) {
    CK(cudaStreamSynchronize(st));
    std::vector<float> got(ny);
    CK(cudaMemcpy(got.data(), y, ny * 4, cudaMemcpyDeviceToHost));
    const auto want = reference(c);
    double d = 0;
    for (int i = 0; i < ny; ++i) d = std::fmax(d, std::fabs(got[i] - want[i]));
    expect(what, d < 1e-3, d);
  };

  // Plan and variant pack, built once and kept for the eager call; the capture
  // gets its own pair, destroyed as soon as the capture ends.
  struct Built { Desc graph, heur, cfg, plan, pack; void* work; bool ok; };
  auto build = [&]() {
    Built b{};
    Desc tx = tensor(1, xd), tw = tensor(2, wd), ty = tensor(3, yd);
    Desc cd = make(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
    const cudnnDataType_t f = CUDNN_DATA_FLOAT;
    const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
    const int64_t spatial = 2;
    const int64_t pad[2] = {1, 1}, stride[2] = {1, 1}, dil[2] = {1, 1};
    set(cd, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &f);
    set(cd, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode);
    set(cd, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial);
    set(cd, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, pad);
    set(cd, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, pad);
    set(cd, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, stride);
    set(cd, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, dil);
    cudnnBackendFinalize(cd);
    const float one = 1.0f, zero = 0.0f;
    Desc op = make(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tx);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tw);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &ty);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cd);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &one);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &zero);
    cudnnBackendFinalize(op);
    b.graph = make(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    set(b.graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &h);
    set(b.graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op);
    cudnnBackendFinalize(b.graph);
    for (Desc d : {tx, tw, ty, cd, op}) cudnnBackendDestroyDescriptor(d);
    b.heur = make(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    const cudnnBackendHeurMode_t hm = CUDNN_HEUR_MODE_INSTANT;
    set(b.heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &b.graph);
    set(b.heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &hm);
    cudnnBackendFinalize(b.heur);
    // Take the first configuration that plans and runs under capture; NVIDIA's
    // heuristics list more than one.
    int64_t n_cfg = 0;
    b.cfg = make(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    cudnnBackendGetAttribute(b.heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &n_cfg, &b.cfg);
    b.plan = make(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    set(b.plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &h);
    set(b.plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &b.cfg);
    b.ok = n_cfg >= 1 && cudnnBackendFinalize(b.plan) == CUDNN_STATUS_SUCCESS;
    int64_t ws = 0, one_i = 1;
    if (b.ok) cudnnBackendGetAttribute(b.plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &one_i, &ws);
    if (ws > 0) cudaMalloc(&b.work, (size_t)ws);
    b.pack = make(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    void* ptrs[3] = {x, w, y};
    const int64_t uids[3] = {1, 2, 3};
    set(b.pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, ptrs);
    set(b.pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids);
    set(b.pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &b.work);
    cudnnBackendFinalize(b.pack);
    return b;
  };
  auto destroy = [&](Built& b) {
    for (Desc d : {b.graph, b.heur, b.cfg, b.plan, b.pack}) cudnnBackendDestroyDescriptor(d);
  };

  int c = 3;
  CK(cudaMemcpy(counter, &c, 4, cudaMemcpyHostToDevice));
  {
    Built b = build();
    if (!b.ok) { expect("cuDNN convolution, planned", false); return; }
    fill_from_counter<<<4, 128, 0, st>>>(x, nx, counter, 0.25f);
    CD(cudnnBackendExecute(h, b.plan, b.pack));
    check("cuDNN convolution, eager", c);
    destroy(b);
    cudaFree(b.work);
  }

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  CK(cudaMemset(y, 0, ny * 4));
  // Planned before the capture (planning allocates, which a capture forbids);
  // the descriptors go as soon as it ends, the workspace when the graph does.
  Built b = build();
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  fill_from_counter<<<4, 128, 0, st>>>(x, nx, counter, 0.25f);
  const cudnnStatus_t rs = cudnnBackendExecute(h, b.plan, b.pack);
  const cudaError_t end = cudaStreamEndCapture(st, &graph);
  destroy(b);
  expect("cuDNN convolution, recorded into a capture", rs == CUDNN_STATUS_SUCCESS && end == cudaSuccess, end);
  if (rs != CUDNN_STATUS_SUCCESS || end != cudaSuccess) return;
  CK(cudaGraphInstantiate(&exec, graph, 0));
  CK(cudaStreamSynchronize(st));
  std::vector<float> zero(ny);
  CK(cudaMemcpy(zero.data(), y, ny * 4, cudaMemcpyDeviceToHost));
  bool untouched = true;
  for (float v : zero) untouched = untouched && v == 0.0f;
  expect("cuDNN convolution: the capture did not run the call", untouched);
  for (int r = 0; r < 3; ++r) {
    c = 6 + 5 * r;
    CK(cudaMemcpy(counter, &c, 4, cudaMemcpyHostToDevice));
    CK(cudaGraphLaunch(exec, st));
    char what[96];
    std::snprintf(what, sizeof what, "cuDNN convolution, graph launch %d (counter %d)", r, c);
    check(what, c);
  }
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaFree(b.work);
  cudnnDestroy(h);
  cudaFree(x), cudaFree(w), cudaFree(y), cudaFree(counter);
}

/* ---- a kernel launched through the driver API -------------------------------- */

static const char* kAddCounterPtx = R"(
.version 7.0
.target sm_50
.address_size 64
.visible .entry addc(.param .u64 py, .param .u64 px, .param .u64 pc, .param .u32 pn)
{
  .reg .pred %p<2>;
  .reg .b32 %r<8>;
  .reg .f32 %f<5>;
  .reg .b64 %rd<8>;
  ld.param.u64 %rd1, [py];
  ld.param.u64 %rd2, [px];
  ld.param.u64 %rd3, [pc];
  ld.param.u32 %r1, [pn];
  mov.u32 %r2, %tid.x;
  mov.u32 %r3, %ctaid.x;
  mov.u32 %r4, %ntid.x;
  mad.lo.s32 %r5, %r3, %r4, %r2;
  setp.ge.s32 %p1, %r5, %r1;
  @%p1 bra DONE;
  cvta.to.global.u64 %rd1, %rd1;
  cvta.to.global.u64 %rd2, %rd2;
  cvta.to.global.u64 %rd3, %rd3;
  ld.global.u32 %r6, [%rd3];
  cvt.rn.f32.s32 %f1, %r6;
  mul.wide.s32 %rd4, %r5, 4;
  add.s64 %rd5, %rd2, %rd4;
  ld.global.f32 %f2, [%rd5];
  add.f32 %f3, %f2, %f1;
  add.s64 %rd6, %rd1, %rd4;
  st.global.f32 [%rd6], %f3;
DONE:
  ret;
}
)";

static void driver_launch(cudaStream_t st) {
  const int n = 100;
  CUmodule mod;
  CUfunction fn;
  CK(cudaFree(nullptr));   // the runtime's context is the driver's current one
  if (cuModuleLoadData(&mod, kAddCounterPtx) != CUDA_SUCCESS || cuModuleGetFunction(&fn, mod, "addc") != CUDA_SUCCESS) {
    expect("a driver-API kernel loads", false);
    return;
  }
  float *x, *y;
  int* counter;
  CK(cudaMalloc(&x, n * 4));
  CK(cudaMalloc(&y, n * 4));
  CK(cudaMalloc(&counter, 4));
  CK(cudaMemset(y, 0, n * 4));
  std::vector<float> hx(n), got(n);
  auto set_x = [&](float scale) {
    for (int i = 0; i < n; ++i) hx[i] = scale * static_cast<float>(i);
    return cudaMemcpy(x, hx.data(), n * 4, cudaMemcpyHostToDevice);
  };
  auto launch = [&] {
    void* args[] = {&y, &x, &counter, const_cast<int*>(&n)};
    return cuLaunchKernel(fn, 4, 1, 1, 32, 1, 1, 0, st, args, nullptr);
  };
  auto check = [&](const char* what, float scale, int c) {
    CK(cudaStreamSynchronize(st));
    CK(cudaMemcpy(got.data(), y, n * 4, cudaMemcpyDeviceToHost));
    double d = 0;
    for (int i = 0; i < n; ++i) d = std::fmax(d, std::fabs(got[i] - (scale * static_cast<float>(i) + static_cast<float>(c))));
    expect(what, d == 0, d);
  };

  int c = 4;
  CK(set_x(1.0f));
  CK(cudaMemcpy(counter, &c, 4, cudaMemcpyHostToDevice));
  expect("cuLaunchKernel, eager", launch() == CUDA_SUCCESS);
  check("cuLaunchKernel, eager result", 1.0f, c);

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  CK(cudaMemset(y, 0, n * 4));
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  expect("cuStreamIsCapturing says so during a capture", cuStreamIsCapturing(st, &status) == CUDA_SUCCESS && status == CU_STREAM_CAPTURE_STATUS_ACTIVE, status);
  const CUresult rl = launch();
  const cudaError_t end = cudaStreamEndCapture(st, &graph);
  expect("cuLaunchKernel, recorded into a capture", rl == CUDA_SUCCESS && end == cudaSuccess, end);
  if (rl != CUDA_SUCCESS || end != cudaSuccess) return;
  size_t nodes = 0;
  CK(cudaGraphGetNodes(graph, nullptr, &nodes));
  expect("the capture holds the kernel as a node", nodes == 1, static_cast<double>(nodes));
  CK(cudaGraphInstantiate(&exec, graph, 0));
  CK(cudaStreamSynchronize(st));
  CK(cudaMemcpy(got.data(), y, n * 4, cudaMemcpyDeviceToHost));
  bool untouched = true;
  for (float v : got) untouched = untouched && v == 0.0f;
  expect("cuLaunchKernel: the capture did not run the kernel", untouched);
  for (int r = 0; r < 3; ++r) {
    c = 7 + 3 * r;
    const float scale = 0.5f * static_cast<float>(r + 1);
    CK(set_x(scale));
    CK(cudaMemcpy(counter, &c, 4, cudaMemcpyHostToDevice));
    CK(cudaGraphLaunch(exec, st));
    char what[96];
    std::snprintf(what, sizeof what, "cuLaunchKernel, graph launch %d (counter %d)", r, c);
    check(what, scale, c);
  }
  status = CU_STREAM_CAPTURE_STATUS_ACTIVE;
  expect("cuStreamIsCapturing says no once it ended", cuStreamIsCapturing(st, &status) == CUDA_SUCCESS && status == CU_STREAM_CAPTURE_STATUS_NONE, status);
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cuModuleUnload(mod);
  cudaFree(x), cudaFree(y), cudaFree(counter);
}

int main() {
  cudaStream_t st;
  if (cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess) { std::printf("FAIL no stream\n"); return 1; }
  lt_matmul(st);
  cudnn_conv(st);
  driver_launch(st);
  std::puts(fails ? "FAIL" : "PASS");
  return fails != 0;
}
