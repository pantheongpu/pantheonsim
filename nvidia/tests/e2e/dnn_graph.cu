// cuDNN's backend (graph) API beyond single convolutions: operation graphs
// joined through virtual tensors, as cudnn-frontend builds them --
//
//   - convolution forward + bias + ReLU (conv-bias-activation fusion)
//   - backward-data + ReLU backward (dgrad-drelu)
//   - batched matmul + bias + GELU (a matmul epilogue)
//   - reductions (sum, max, L2 norm) over chosen dimensions
//   - pointwise forward and backward modes, with broadcasting and alpha
//
// each checked against a reference computed here on the host, and a graph
// holding an operation the library has no engine for refused rather than run.
// The same program runs against NVIDIA's libcudnn.so.9: where its heuristics
// offer no engine for a graph on that GPU, the check says so and is skipped,
// since that is the library declining, not answering wrongly; on VirtualGPU
// every one must run.
#include <cudnn.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int fails = 0;
static void expect(const char* what, bool ok, double detail = 0) {
  std::printf("%s %s", ok ? "ok  " : "FAIL", what);
  if (!ok && detail != 0) std::printf(" (%g)", detail);
  std::printf("\n");
  fails += !ok;
}

static cudnnHandle_t H;
using Desc = cudnnBackendDescriptor_t;

static Desc make(cudnnBackendDescriptorType_t t) {
  Desc d = nullptr;
  cudnnBackendCreateDescriptor(t, &d);
  return d;
}
static void set(Desc d, cudnnBackendAttributeName_t n, cudnnBackendAttributeType_t t, int64_t c, const void* v) {
  cudnnBackendSetAttribute(d, n, t, c, v);
}
static void set_desc(Desc d, cudnnBackendAttributeName_t n, Desc v) { set(d, n, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &v); }
static std::vector<int64_t> packed(const std::vector<int64_t>& dims) {
  std::vector<int64_t> s(dims.size(), 1);
  for (size_t i = dims.size() - 1; i-- > 0;) s[i] = s[i + 1] * dims[i + 1];
  return s;
}
static size_t count(const std::vector<int64_t>& d) {
  size_t n = 1;
  for (int64_t x : d) n *= (size_t)x;
  return n;
}

static Desc tensor(int64_t uid, const std::vector<int64_t>& dims, bool is_virtual = false) {
  Desc t = make(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
  const cudnnDataType_t f = CUDNN_DATA_FLOAT;
  const int64_t align = 16;
  const std::vector<int64_t> strides = packed(dims);
  set(t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &f);
  set(t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, (int64_t)dims.size(), dims.data());
  set(t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, (int64_t)strides.size(), strides.data());
  set(t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
  set(t, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align);
  set(t, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &is_virtual);
  cudnnBackendFinalize(t);
  return t;
}

static Desc pointwise_desc(cudnnPointwiseMode_t mode) {
  Desc p = make(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
  const cudnnDataType_t f = CUDNN_DATA_FLOAT;
  set(p, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &mode);
  set(p, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &f);
  cudnnBackendFinalize(p);
  return p;
}
// y = mode(alpha1 * x [, alpha2 * b]).
static Desc pointwise(cudnnPointwiseMode_t mode, Desc x, Desc b, Desc y, float a1 = 1.0f, float a2 = 1.0f) {
  Desc pd = pointwise_desc(mode);
  Desc op = make(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
  set_desc(op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, pd);
  set_desc(op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, x);
  if (b) set_desc(op, CUDNN_ATTR_OPERATION_POINTWISE_BDESC, b);
  set_desc(op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, y);
  set(op, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, CUDNN_TYPE_FLOAT, 1, &a1);
  if (b) set(op, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2, CUDNN_TYPE_FLOAT, 1, &a2);
  cudnnBackendFinalize(op);
  cudnnBackendDestroyDescriptor(pd);
  return op;
}
// dx = mode_bwd(dy, x).
static Desc pointwise_bwd(cudnnPointwiseMode_t mode, Desc dy, Desc x, Desc dx) {
  Desc pd = pointwise_desc(mode);
  Desc op = make(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
  set_desc(op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, pd);
  set_desc(op, CUDNN_ATTR_OPERATION_POINTWISE_DYDESC, dy);
  set_desc(op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, x);
  set_desc(op, CUDNN_ATTR_OPERATION_POINTWISE_DXDESC, dx);
  cudnnBackendFinalize(op);
  cudnnBackendDestroyDescriptor(pd);
  return op;
}

static Desc conv_desc(int64_t pad) {
  Desc cd = make(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
  const cudnnDataType_t f = CUDNN_DATA_FLOAT;
  const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
  const int64_t spatial = 2, pads[2] = {pad, pad}, ones[2] = {1, 1};
  set(cd, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &f);
  set(cd, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode);
  set(cd, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial);
  set(cd, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, pads);
  set(cd, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, pads);
  set(cd, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, ones);
  set(cd, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, ones);
  cudnnBackendFinalize(cd);
  return cd;
}

// Builds the graph, asks the heuristics, and runs the first configuration
// that makes a plan. 1 ran, 0 no engine was offered, -1 an error.
static int run(const char* what, const std::vector<Desc>& ops, const std::vector<int64_t>& uids,
               const std::vector<void*>& ptrs) {
  Desc graph = make(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
  set(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &H);
  set(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, (int64_t)ops.size(), ops.data());
  cudnnStatus_t s = cudnnBackendFinalize(graph);
  if (s != CUDNN_STATUS_SUCCESS) {
    std::printf("     %s: the graph did not finalize (%d)\n", what, (int)s);
    cudnnBackendDestroyDescriptor(graph);
    return s == CUDNN_STATUS_NOT_SUPPORTED ? 0 : -1;
  }
  int result = 0;
  for (cudnnBackendHeurMode_t hm : {CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_FALLBACK}) {
    Desc heur = make(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    set_desc(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, graph);
    set(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &hm);
    if (cudnnBackendFinalize(heur) != CUDNN_STATUS_SUCCESS) { cudnnBackendDestroyDescriptor(heur); continue; }
    int64_t n = 0;
    cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &n, nullptr);
    std::vector<Desc> cfgs((size_t)n);
    for (auto& c : cfgs) c = make(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    if (n) cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, n, &n, cfgs.data());
    for (int64_t i = 0; i < n && !result; ++i) {
      Desc plan = make(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
      set(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &H);
      set_desc(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, cfgs[i]);
      if (cudnnBackendFinalize(plan) == CUDNN_STATUS_SUCCESS) {
        int64_t ws = 0, one = 1;
        cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &one, &ws);
        void* work = nullptr;
        if (ws > 0) cudaMalloc(&work, (size_t)ws);
        Desc pack = make(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        set(pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, (int64_t)ptrs.size(), ptrs.data());
        set(pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, (int64_t)uids.size(), uids.data());
        set(pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &work);
        cudnnBackendFinalize(pack);
        s = cudnnBackendExecute(H, plan, pack);
        cudaDeviceSynchronize();
        result = s == CUDNN_STATUS_SUCCESS ? 1 : -1;
        if (s != CUDNN_STATUS_SUCCESS) std::printf("     %s: execute -> %d\n", what, (int)s);
        cudnnBackendDestroyDescriptor(pack);
        cudaFree(work);
      }
      cudnnBackendDestroyDescriptor(plan);
    }
    for (auto& c : cfgs) cudnnBackendDestroyDescriptor(c);
    cudnnBackendDestroyDescriptor(heur);
    if (result) break;
  }
  cudnnBackendDestroyDescriptor(graph);
  return result;
}

// On VirtualGPU (the tests set VGPU_GPU) every graph here must run.
static const bool kOnSim = std::getenv("VGPU_GPU") != nullptr;

static void report(const char* what, int ran, double err, double tol) {
  if (ran == 0 && !kOnSim) {
    std::printf("skip %s (no engine offered for this graph)\n", what);
    return;
  }
  expect(what, ran == 1 && err < tol, ran == 1 ? err : -1);
}

static float* to_dev(const std::vector<float>& h) {
  float* p = nullptr;
  cudaMalloc(&p, h.size() * sizeof(float) + 64);
  cudaMemcpy(p, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
  return p;
}
static std::vector<float> from_dev(const float* p, size_t n) {
  std::vector<float> h(n);
  cudaMemcpy(h.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost);
  return h;
}
static std::vector<float> filled(size_t n, float scale, int seed) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = scale * std::sin(0.7f * (i + seed) + 0.3f) + 0.05f * ((i + seed) % 7) - 0.1f;
  return v;
}
static double max_rel(const std::vector<float>& got, const std::vector<double>& want) {
  double m = 0;
  for (size_t i = 0; i < want.size(); ++i) m = std::fmax(m, std::fabs(got[i] - want[i]) / (std::fabs(want[i]) + 1.0));
  return m;
}

/* ---- conv + bias + relu, and dgrad + drelu ---- */

static void conv_fusions() {
  const int N = 2, C = 3, Hh = 6, W = 5, K = 4, R = 3;
  const std::vector<int64_t> xd = {N, C, Hh, W}, wd = {K, C, R, R}, yd = {N, K, Hh, W}, bd = {1, K, 1, 1};
  const auto hx = filled(count(xd), 1.0f, 1), hw = filled(count(wd), 0.5f, 2), hb = filled(K, 0.3f, 3);
  // Host reference of the convolution (3x3, pad 1).
  std::vector<double> conv(count(yd), 0.0);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      for (int i = 0; i < Hh; ++i)
        for (int j = 0; j < W; ++j) {
          double s = 0;
          for (int c = 0; c < C; ++c)
            for (int r = 0; r < R; ++r)
              for (int q = 0; q < R; ++q) {
                const int ii = i - 1 + r, jj = j - 1 + q;
                if (ii < 0 || ii >= Hh || jj < 0 || jj >= W) continue;
                s += (double)hx[((n * C + c) * Hh + ii) * W + jj] * hw[((k * C + c) * R + r) * R + q];
              }
          conv[((n * K + k) * Hh + i) * W + j] = s;
        }
  {
    Desc x = tensor(1, xd), w = tensor(2, wd), b = tensor(3, bd), y = tensor(4, yd);
    Desc t1 = tensor(101, yd, true), t2 = tensor(102, yd, true);
    Desc cd = conv_desc(1);
    Desc op = make(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    const float one = 1.0f, zero = 0.0f;
    set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, x);
    set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, w);
    set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, t1);
    set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, cd);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &one);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &zero);
    cudnnBackendFinalize(op);
    Desc add = pointwise(CUDNN_POINTWISE_ADD, t1, b, t2);
    Desc relu = pointwise(CUDNN_POINTWISE_RELU_FWD, t2, nullptr, y);
    float *dx = to_dev(hx), *dw = to_dev(hw), *db = to_dev(hb), *dy = to_dev(std::vector<float>(count(yd), 7.0f));
    const int ran = run("conv + bias + relu", {op, add, relu}, {1, 2, 3, 4}, {dx, dw, db, dy});
    std::vector<double> want(conv.size());
    for (size_t i = 0; i < want.size(); ++i) want[i] = std::fmax(0.0, conv[i] + hb[(i / (Hh * W)) % K]);
    report("graph: convolution forward + bias + ReLU", ran, ran == 1 ? max_rel(from_dev(dy, want.size()), want) : 0,
           1e-4);
    for (Desc d : {x, w, b, y, t1, t2, cd, op, add, relu}) cudnnBackendDestroyDescriptor(d);
    cudaFree(dx), cudaFree(dw), cudaFree(db), cudaFree(dy);
  }
  {
    // dx = relu_bwd(conv_bwd_data(dy, w), x): the gradient through a ReLU
    // whose input was x.
    const auto hdy = filled(count(yd), 0.8f, 5);
    Desc gy = tensor(1, yd), w = tensor(2, wd), x = tensor(3, xd), gx = tensor(4, xd), t = tensor(101, xd, true);
    Desc cd = conv_desc(1);
    Desc op = make(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);
    const float one = 1.0f, zero = 0.0f;
    set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, gy);
    set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, w);
    set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, t);
    set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC, cd);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA, CUDNN_TYPE_FLOAT, 1, &one);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA, CUDNN_TYPE_FLOAT, 1, &zero);
    cudnnBackendFinalize(op);
    Desc drelu = pointwise_bwd(CUDNN_POINTWISE_RELU_BWD, t, x, gx);
    float *ddy = to_dev(hdy), *dw = to_dev(hw), *dx = to_dev(hx), *dgx = to_dev(std::vector<float>(count(xd), 7.0f));
    const int ran = run("dgrad + drelu", {op, drelu}, {1, 2, 3, 4}, {ddy, dw, dx, dgx});
    std::vector<double> want(count(xd), 0.0);
    for (int n = 0; n < N; ++n)
      for (int c = 0; c < C; ++c)
        for (int ii = 0; ii < Hh; ++ii)
          for (int jj = 0; jj < W; ++jj) {
            double s = 0;
            for (int k = 0; k < K; ++k)
              for (int r = 0; r < R; ++r)
                for (int q = 0; q < R; ++q) {
                  const int i = ii + 1 - r, j = jj + 1 - q;
                  if (i < 0 || i >= Hh || j < 0 || j >= W) continue;
                  s += (double)hdy[((n * K + k) * Hh + i) * W + j] * hw[((k * C + c) * R + r) * R + q];
                }
            const size_t at = ((n * C + c) * Hh + ii) * W + jj;
            want[at] = hx[at] > 0 ? s : 0.0;
          }
    report("graph: convolution backward-data + ReLU backward", ran,
           ran == 1 ? max_rel(from_dev(dgx, want.size()), want) : 0, 1e-4);
    for (Desc d : {gy, w, x, gx, t, cd, op, drelu}) cudnnBackendDestroyDescriptor(d);
    cudaFree(ddy), cudaFree(dw), cudaFree(dx), cudaFree(dgx);
  }
}

/* ---- matmul + bias + gelu ---- */

static void matmul() {
  const int B = 2, M = 4, K = 8, N = 6;
  const std::vector<int64_t> ad = {B, M, K}, bd = {B, K, N}, cd = {B, M, N}, biasd = {1, 1, N};
  const auto ha = filled(count(ad), 1.0f, 1), hb = filled(count(bd), 0.5f, 2), hbias = filled(N, 0.4f, 3);
  Desc a = tensor(1, ad), b = tensor(2, bd), bias = tensor(3, biasd), out = tensor(4, cd);
  Desc c = tensor(101, cd, true), t = tensor(102, cd, true);
  Desc md = make(CUDNN_BACKEND_MATMUL_DESCRIPTOR);
  const cudnnDataType_t f = CUDNN_DATA_FLOAT;
  set(md, CUDNN_ATTR_MATMUL_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &f);
  cudnnBackendFinalize(md);
  Desc op = make(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);
  set_desc(op, CUDNN_ATTR_OPERATION_MATMUL_ADESC, a);
  set_desc(op, CUDNN_ATTR_OPERATION_MATMUL_BDESC, b);
  set_desc(op, CUDNN_ATTR_OPERATION_MATMUL_CDESC, c);
  set_desc(op, CUDNN_ATTR_OPERATION_MATMUL_DESC, md);
  cudnnBackendFinalize(op);
  Desc add = pointwise(CUDNN_POINTWISE_ADD, c, bias, t);
  Desc gelu = pointwise(CUDNN_POINTWISE_GELU_FWD, t, nullptr, out);
  float *da = to_dev(ha), *db = to_dev(hb), *dbias = to_dev(hbias), *dout = to_dev(std::vector<float>(count(cd), 7.0f));
  const int ran = run("matmul + bias + gelu", {op, add, gelu}, {1, 2, 3, 4}, {da, db, dbias, dout});
  std::vector<double> want(count(cd));
  for (int q = 0; q < B; ++q)
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) {
        double s = 0;
        for (int k = 0; k < K; ++k) s += (double)ha[(q * M + i) * K + k] * hb[(q * K + k) * N + j];
        s += hbias[j];
        want[(q * M + i) * N + j] = 0.5 * s * (1.0 + std::erf(s / std::sqrt(2.0)));
      }
  report("graph: batched matmul + bias + GELU", ran, ran == 1 ? max_rel(from_dev(dout, want.size()), want) : 0, 1e-3);
  for (Desc d : {a, b, bias, out, c, t, md, op, add, gelu}) cudnnBackendDestroyDescriptor(d);
  cudaFree(da), cudaFree(db), cudaFree(dbias), cudaFree(dout);
}

/* ---- reductions ---- */

static void reductions() {
  const std::vector<int64_t> xd = {2, 3, 4, 5}, yd = {2, 3, 1, 1};
  const auto hx = filled(count(xd), 1.5f, 4);
  struct R { const char* name; cudnnReduceTensorOp_t op; };
  for (const R& r : {R{"sum", CUDNN_REDUCE_TENSOR_ADD}, R{"max", CUDNN_REDUCE_TENSOR_MAX},
                     R{"L2 norm", CUDNN_REDUCE_TENSOR_NORM2}}) {
    Desc x = tensor(1, xd), y = tensor(2, yd);
    Desc rd = make(CUDNN_BACKEND_REDUCTION_DESCRIPTOR);
    const cudnnDataType_t f = CUDNN_DATA_FLOAT;
    set(rd, CUDNN_ATTR_REDUCTION_OPERATOR, CUDNN_TYPE_REDUCTION_OPERATOR_TYPE, 1, &r.op);
    set(rd, CUDNN_ATTR_REDUCTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &f);
    cudnnBackendFinalize(rd);
    Desc op = make(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR);
    set_desc(op, CUDNN_ATTR_OPERATION_REDUCTION_XDESC, x);
    set_desc(op, CUDNN_ATTR_OPERATION_REDUCTION_YDESC, y);
    set_desc(op, CUDNN_ATTR_OPERATION_REDUCTION_DESC, rd);
    cudnnBackendFinalize(op);
    float *dx = to_dev(hx), *dy = to_dev(std::vector<float>(6, 7.0f));
    char what[96];
    std::snprintf(what, sizeof what, "graph: reduction (%s) over the spatial dimensions", r.name);
    const int ran = run(what, {op}, {1, 2}, {dx, dy});
    std::vector<double> want(6);
    for (int nc = 0; nc < 6; ++nc) {
      double s = r.op == CUDNN_REDUCE_TENSOR_MAX ? -1e30 : 0.0;
      for (int i = 0; i < 20; ++i) {
        const double v = hx[nc * 20 + i];
        if (r.op == CUDNN_REDUCE_TENSOR_ADD) s += v;
        else if (r.op == CUDNN_REDUCE_TENSOR_MAX) s = std::fmax(s, v);
        else s += v * v;
      }
      want[nc] = r.op == CUDNN_REDUCE_TENSOR_NORM2 ? std::sqrt(s) : s;
    }
    report(what, ran, ran == 1 ? max_rel(from_dev(dy, 6), want) : 0, 1e-5);
    for (Desc d : {x, y, rd, op}) cudnnBackendDestroyDescriptor(d);
    cudaFree(dx), cudaFree(dy);
  }
}

/* ---- pointwise: broadcasting, alpha, backward modes ---- */

static void pointwise_ops() {
  const std::vector<int64_t> xd = {2, 3, 4}, bd = {1, 3, 1};
  const auto hx = filled(count(xd), 1.2f, 6), hb = filled(3, 1.0f, 7), hdy = filled(count(xd), 0.9f, 8);
  {
    Desc x = tensor(1, xd), b = tensor(2, bd), y = tensor(3, xd);
    Desc op = pointwise(CUDNN_POINTWISE_MUL, x, b, y, 1.5f, -0.5f);
    float *dx = to_dev(hx), *db = to_dev(hb), *dy = to_dev(std::vector<float>(count(xd), 7.0f));
    const int ran = run("mul", {op}, {1, 2, 3}, {dx, db, dy});
    std::vector<double> want(count(xd));
    for (size_t i = 0; i < want.size(); ++i) want[i] = 1.5 * hx[i] * -0.5 * hb[(i / 4) % 3];
    report("graph: pointwise multiply, broadcast and scaled", ran, ran == 1 ? max_rel(from_dev(dy, want.size()), want) : 0,
           1e-6);
    for (Desc d : {x, b, y, op}) cudnnBackendDestroyDescriptor(d);
    cudaFree(dx), cudaFree(db), cudaFree(dy);
  }
  struct B { const char* name; cudnnPointwiseMode_t mode; };
  for (const B& m : {B{"sigmoid", CUDNN_POINTWISE_SIGMOID_BWD}, B{"tanh", CUDNN_POINTWISE_TANH_BWD},
                     B{"GELU", CUDNN_POINTWISE_GELU_BWD}}) {
    Desc gy = tensor(1, xd), x = tensor(2, xd), gx = tensor(3, xd);
    Desc op = pointwise_bwd(m.mode, gy, x, gx);
    float *ddy = to_dev(hdy), *dx = to_dev(hx), *dgx = to_dev(std::vector<float>(count(xd), 7.0f));
    char what[96];
    std::snprintf(what, sizeof what, "graph: pointwise %s backward", m.name);
    const int ran = run(what, {op}, {1, 2, 3}, {ddy, dx, dgx});
    std::vector<double> want(count(xd));
    for (size_t i = 0; i < want.size(); ++i) {
      const double v = hx[i], g = hdy[i];
      if (m.mode == CUDNN_POINTWISE_SIGMOID_BWD) {
        const double s = 1.0 / (1.0 + std::exp(-v));
        want[i] = g * s * (1 - s);
      } else if (m.mode == CUDNN_POINTWISE_TANH_BWD) {
        want[i] = g * (1 - std::tanh(v) * std::tanh(v));
      } else {
        want[i] = g * (0.5 * (1 + std::erf(v / std::sqrt(2.0))) + v * std::exp(-0.5 * v * v) / std::sqrt(2 * M_PI));
      }
    }
    report(what, ran, ran == 1 ? max_rel(from_dev(dgx, want.size()), want) : 0, 1e-4);
    for (Desc d : {gy, x, gx, op}) cudnnBackendDestroyDescriptor(d);
    cudaFree(ddy), cudaFree(dx), cudaFree(dgx);
  }
}

// An operation this library has no engine for is refused when it is
// finalized, by name, not accepted and run wrongly. (NVIDIA's library has an
// engine for it, so this is VirtualGPU's own contract.)
static void refusals() {
  if (!kOnSim) {
    std::printf("skip refusal of an operation with no engine (VirtualGPU's contract)\n");
    return;
  }
  Desc rs = make(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR);
  expect("an operation with no engine here is refused, not run",
         cudnnBackendFinalize(rs) == CUDNN_STATUS_NOT_SUPPORTED);
  cudnnBackendDestroyDescriptor(rs);
}

int main() {
  if (cudnnCreate(&H) != CUDNN_STATUS_SUCCESS) {
    std::printf("FAIL cudnnCreate\n");
    return 1;
  }
  conv_fusions();
  matmul();
  reductions();
  pointwise_ops();
  refusals();
  cudnnDestroy(H);
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
