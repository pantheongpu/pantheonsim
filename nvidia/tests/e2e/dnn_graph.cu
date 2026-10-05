// cuDNN's backend (graph) API beyond single convolutions: operation graphs
// joined through virtual tensors, as cudnn-frontend builds them --
//
//   - convolution forward + bias + ReLU (conv-bias-activation fusion)
//   - backward-data + ReLU backward (dgrad-drelu)
//   - batched matmul + bias + GELU (a matmul epilogue)
//   - reductions (sum, max, L2 norm) over chosen dimensions
//   - pointwise forward and backward modes, with broadcasting and alpha
//   - layer, RMS and batch normalization, training forward (saved and
//     running statistics) and backward
//   - max and average pooling (resampling), forward and backward
//   - a convolution padded asymmetrically, and a concatenation
//   - NHWC max pooling with its index tensor, and the backward pass from it
//   - group normalization, and normalization backward without the saved
//     statistics
//   - statistics generation, the RNG operation, reshape, transpose, slice
//   - an INT8x4 vectorized convolution
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

/* ---- normalization: layer, batch and RMS, forward and backward ---- */

static Desc scalar_tensor(int64_t uid) {
  Desc t = make(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
  const cudnnDataType_t f = CUDNN_DATA_FLOAT;
  const int64_t one[4] = {1, 1, 1, 1}, align = 4;
  const bool by_value = true;
  set(t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &f);
  set(t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, one);
  set(t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, one);
  set(t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
  set(t, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align);
  set(t, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN, 1, &by_value);
  cudnnBackendFinalize(t);
  return t;
}

// Statistics per group of x (groups: the positions where the stats shape
// keeps x's index), the host reference for every normalization below.
struct Stats { std::vector<double> mean, inv, var; std::vector<size_t> slot; std::vector<double> count; };
static Stats stats(const std::vector<float>& x, const std::vector<int64_t>& xd, const std::vector<int64_t>& sd,
                   bool rms, double eps) {
  Stats s;
  const size_t n = x.size(), ns = count(sd);
  s.mean.assign(ns, 0), s.var.assign(ns, 0), s.inv.assign(ns, 0), s.count.assign(ns, 0), s.slot.resize(n);
  for (size_t i = 0; i < n; ++i) {
    size_t rem = i, k = 0, mul = 1;
    std::vector<int64_t> idx(xd.size());
    for (size_t d = xd.size(); d-- > 0;) idx[d] = rem % xd[d], rem /= xd[d];
    for (size_t d = xd.size(); d-- > 0;) k += (sd[d] == 1 ? 0 : idx[d]) * mul, mul *= sd[d];
    s.slot[i] = k;
    s.count[k] += 1;
    s.mean[k] += rms ? 0.0 : x[i];
  }
  for (size_t k = 0; k < ns; ++k) s.mean[k] /= s.count[k];
  for (size_t i = 0; i < n; ++i) s.var[s.slot[i]] += (x[i] - s.mean[s.slot[i]]) * (x[i] - s.mean[s.slot[i]]);
  for (size_t k = 0; k < ns; ++k) s.var[k] /= s.count[k], s.inv[k] = 1 / std::sqrt(s.var[k] + eps);
  return s;
}

static void norms() {
  const std::vector<int64_t> xd = {4, 6, 1, 1}, pd = {1, 6, 1, 1};
  const float eps = 1e-5f;
  const auto hx = filled(count(xd), 1.5f, 3), hs = filled(6, 0.5f, 4), hb = filled(6, 0.3f, 5);
  struct M { const char* name; cudnnBackendNormMode_t mode; std::vector<int64_t> sd; };
  for (const M& m : {M{"layer", CUDNN_LAYER_NORM, {4, 1, 1, 1}}, M{"RMS", CUDNN_RMS_NORM, {4, 1, 1, 1}},
                     M{"batch", CUDNN_BATCH_NORM, {1, 6, 1, 1}}}) {
    const bool rms = m.mode == CUDNN_RMS_NORM, batch = m.mode == CUDNN_BATCH_NORM;
    // Forward, training: y and the saved statistics (and, for batch
    // normalization, the running ones).
    Desc x = tensor(1, xd), sc = tensor(2, pd), b = tensor(3, pd), e = scalar_tensor(4), y = tensor(5, xd),
         mean = tensor(6, m.sd), inv = tensor(7, m.sd), f = scalar_tensor(8), rmi = tensor(9, pd), rvi = tensor(10, pd),
         rmo = tensor(11, pd), rvo = tensor(12, pd);
    Desc op = make(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR);
    const cudnnBackendNormFwdPhase_t phase = CUDNN_NORM_FWD_TRAINING;
    set(op, CUDNN_ATTR_OPERATION_NORM_FWD_MODE, CUDNN_TYPE_NORM_MODE, 1, &m.mode);
    set(op, CUDNN_ATTR_OPERATION_NORM_FWD_PHASE, CUDNN_TYPE_NORM_FWD_PHASE, 1, &phase);
    set_desc(op, CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, x);
    set_desc(op, CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, sc);
    if (!rms) set_desc(op, CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, b);
    set_desc(op, CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, e);
    set_desc(op, CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, y);
    if (!rms) set_desc(op, CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC, mean);
    set_desc(op, CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC, inv);
    if (batch) {
      set_desc(op, CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC, f);
      set_desc(op, CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC, rmi);
      set_desc(op, CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC, rvi);
      set_desc(op, CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC, rmo);
      set_desc(op, CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC, rvo);
    }
    cudnnBackendFinalize(op);
    const float factor = 0.25f;
    float *dx = to_dev(hx), *ds = to_dev(hs), *db = to_dev(hb), *dyv = to_dev(std::vector<float>(count(xd), 7.0f)),
          *dmean = to_dev(std::vector<float>(count(m.sd), 7.0f)), *dinv = to_dev(std::vector<float>(count(m.sd), 7.0f)),
          *drmi = to_dev(std::vector<float>(6, 0.5f)), *drvi = to_dev(std::vector<float>(6, 2.0f)),
          *drmo = to_dev(std::vector<float>(6, 7.0f)), *drvo = to_dev(std::vector<float>(6, 7.0f));
    std::vector<int64_t> uids = {1, 2, 4, 5, 7};
    std::vector<void*> ptrs = {dx, ds, (void*)&eps, dyv, dinv};
    if (!rms) uids.push_back(3), ptrs.push_back(db), uids.push_back(6), ptrs.push_back(dmean);
    if (batch)
      for (auto [u, p] : std::vector<std::pair<int64_t, void*>>{{8, (void*)&factor}, {9, drmi}, {10, drvi}, {11, drmo}, {12, drvo}})
        uids.push_back(u), ptrs.push_back(p);
    char what[96];
    std::snprintf(what, sizeof what, "graph: %s normalization forward (training)", m.name);
    const int ran = run(what, {op}, uids, ptrs);
    const Stats st = stats(hx, xd, m.sd, rms, eps);
    std::vector<double> want(hx.size());
    for (size_t i = 0; i < want.size(); ++i) {
      const size_t c = (i / 1) % 6, k = st.slot[i];
      want[i] = hs[c] * (hx[i] - st.mean[k]) * st.inv[k] + (rms ? 0.0 : hb[c]);
    }
    double err = 0;
    if (ran == 1) {
      err = max_rel(from_dev(dyv, want.size()), want);
      err = std::fmax(err, max_rel(from_dev(dinv, st.inv.size()), st.inv));
      if (!rms) err = std::fmax(err, max_rel(from_dev(dmean, st.mean.size()), st.mean));
      if (batch) {
        std::vector<double> rm(6), rv(6);
        for (int c = 0; c < 6; ++c) {
          rm[c] = 0.75 * 0.5 + 0.25 * st.mean[c];
          rv[c] = 0.75 * 2.0 + 0.25 * st.var[c] * st.count[c] / (st.count[c] - 1);
        }
        err = std::fmax(err, max_rel(from_dev(drmo, 6), rm));
        err = std::fmax(err, max_rel(from_dev(drvo, 6), rv));
      }
    }
    report(what, ran, err, 1e-4);

    // Backward, from the saved statistics.
    const auto hdy = filled(count(xd), 0.9f, 6);
    Desc gy = tensor(13, xd), gx = tensor(14, xd), gs = tensor(15, pd), gb = tensor(16, pd);
    Desc bop = make(CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR);
    set(bop, CUDNN_ATTR_OPERATION_NORM_BWD_MODE, CUDNN_TYPE_NORM_MODE, 1, &m.mode);
    set_desc(bop, CUDNN_ATTR_OPERATION_NORM_BWD_XDESC, x);
    set_desc(bop, CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC, gy);
    set_desc(bop, CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC, sc);
    if (!rms) set_desc(bop, CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC, mean);
    set_desc(bop, CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC, inv);
    set_desc(bop, CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC, gx);
    set_desc(bop, CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC, gs);
    if (!rms) set_desc(bop, CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC, gb);
    cudnnBackendFinalize(bop);
    std::vector<float> fmean(st.mean.begin(), st.mean.end()), finv(st.inv.begin(), st.inv.end());
    float *ddy = to_dev(hdy), *dgx = to_dev(std::vector<float>(count(xd), 7.0f)), *dgs = to_dev(std::vector<float>(6, 7.0f)),
          *dgb = to_dev(std::vector<float>(6, 7.0f)), *smean = to_dev(fmean), *sinv = to_dev(finv);
    std::vector<int64_t> buids = {1, 13, 2, 7, 14, 15};
    std::vector<void*> bptrs = {dx, ddy, ds, sinv, dgx, dgs};
    if (!rms) buids.push_back(6), bptrs.push_back(smean), buids.push_back(16), bptrs.push_back(dgb);
    std::snprintf(what, sizeof what, "graph: %s normalization backward", m.name);
    const int bran = run(what, {bop}, buids, bptrs);
    double berr = 0;
    if (bran == 1) {
      // dx = inv * (g - mean(g) - xhat * mean(g * xhat)), g = dy * scale.
      const size_t ns = st.mean.size();
      std::vector<double> mg(ns, 0), mgx(ns, 0), dscale(6, 0), dbias(6, 0), want_dx(hx.size());
      for (size_t i = 0; i < hx.size(); ++i) {
        const size_t k = st.slot[i], c = i % 6;
        const double xh = (hx[i] - st.mean[k]) * st.inv[k], g = hdy[i] * hs[c];
        mg[k] += g, mgx[k] += g * xh, dscale[c] += hdy[i] * xh, dbias[c] += hdy[i];
      }
      for (size_t i = 0; i < hx.size(); ++i) {
        const size_t k = st.slot[i], c = i % 6;
        const double xh = (hx[i] - st.mean[k]) * st.inv[k], g = hdy[i] * hs[c];
        want_dx[i] = st.inv[k] * (g - (rms ? 0.0 : mg[k] / st.count[k]) - xh * mgx[k] / st.count[k]);
      }
      berr = max_rel(from_dev(dgx, want_dx.size()), want_dx);
      berr = std::fmax(berr, max_rel(from_dev(dgs, 6), dscale));
      if (!rms) berr = std::fmax(berr, max_rel(from_dev(dgb, 6), dbias));
    }
    report(what, bran, berr, 1e-4);
    for (Desc d : {x, sc, b, e, y, mean, inv, f, rmi, rvi, rmo, rvo, op, gy, gx, gs, gb, bop}) cudnnBackendDestroyDescriptor(d);
    for (float* p : {dx, ds, db, dyv, dmean, dinv, drmi, drvi, drmo, drvo, ddy, dgx, dgs, dgb, smean, sinv}) cudaFree(p);
  }
}

/* ---- pooling as resampling ---- */

static void pooling() {
  const int N = 2, C = 3, Hh = 7, W = 6, win = 3, pad = 1, str = 2;
  const int Ho = 1 + (Hh + 2 * pad - win) / str, Wo = 1 + (W + 2 * pad - win) / str;
  const std::vector<int64_t> xd = {N, C, Hh, W}, yd = {N, C, Ho, Wo};
  const auto hx = filled(count(xd), 2.0f, 2), hdy = filled(count(yd), 1.0f, 3);
  struct P { const char* name; cudnnResampleMode_t mode; cudnnPaddingMode_t pad; };
  for (const P& p : {P{"max", CUDNN_RESAMPLE_MAXPOOL, CUDNN_NEG_INF_PAD},
                     P{"average excluding padding", CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING, CUDNN_ZERO_PAD},
                     P{"average including padding", CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING, CUDNN_ZERO_PAD}}) {
    Desc rd = make(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
    const int64_t nsp = 2, w2[2] = {win, win}, p2[2] = {pad, pad}, s2[2] = {str, str};
    const cudnnDataType_t f = CUDNN_DATA_FLOAT;
    const cudnnNanPropagation_t nan = CUDNN_NOT_PROPAGATE_NAN;
    set(rd, CUDNN_ATTR_RESAMPLE_MODE, CUDNN_TYPE_RESAMPLE_MODE, 1, &p.mode);
    set(rd, CUDNN_ATTR_RESAMPLE_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &f);
    set(rd, CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION, CUDNN_TYPE_NAN_PROPOGATION, 1, &nan);
    set(rd, CUDNN_ATTR_RESAMPLE_PADDING_MODE, CUDNN_TYPE_PADDING_MODE, 1, &p.pad);
    set(rd, CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &nsp);
    set(rd, CUDNN_ATTR_RESAMPLE_WINDOW_DIMS, CUDNN_TYPE_INT64, 2, w2);
    set(rd, CUDNN_ATTR_RESAMPLE_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, p2);
    set(rd, CUDNN_ATTR_RESAMPLE_POST_PADDINGS, CUDNN_TYPE_INT64, 2, p2);
    set(rd, CUDNN_ATTR_RESAMPLE_STRIDES, CUDNN_TYPE_INT64, 2, s2);
    cudnnBackendFinalize(rd);
    // The host reference: windows, divisors and the first maximum.
    std::vector<double> want_y(count(yd)), want_dx(count(xd), 0.0);
    for (int nc = 0; nc < N * C; ++nc)
      for (int i = 0; i < Ho; ++i)
        for (int j = 0; j < Wo; ++j) {
          double best = -1e30, sum = 0;
          int arg = -1, valid = 0;
          for (int a = 0; a < win; ++a)
            for (int b = 0; b < win; ++b) {
              const int r = i * str - pad + a, c = j * str - pad + b;
              if (r < 0 || r >= Hh || c < 0 || c >= W) continue;
              const double v = hx[(nc * Hh + r) * W + c];
              if (arg < 0 || v > best) best = v, arg = (nc * Hh + r) * W + c;
              sum += v, ++valid;
            }
          const int yi = (nc * Ho + i) * Wo + j;
          const double div = p.mode == CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING ? valid : win * win;
          want_y[yi] = p.mode == CUDNN_RESAMPLE_MAXPOOL ? best : sum / div;
          if (p.mode == CUDNN_RESAMPLE_MAXPOOL) {
            want_dx[arg] += hdy[yi];
            continue;
          }
          for (int a = 0; a < win; ++a)
            for (int b = 0; b < win; ++b) {
              const int r = i * str - pad + a, c = j * str - pad + b;
              if (r >= 0 && r < Hh && c >= 0 && c < W) want_dx[(nc * Hh + r) * W + c] += hdy[yi] / div;
            }
        }
    {
      Desc x = tensor(1, xd), y = tensor(2, yd);
      Desc op = make(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR);
      set_desc(op, CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC, rd);
      set_desc(op, CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, x);
      set_desc(op, CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, y);
      cudnnBackendFinalize(op);
      float *dx = to_dev(hx), *dy = to_dev(std::vector<float>(count(yd), 7.0f));
      char what[96];
      std::snprintf(what, sizeof what, "graph: %s pooling forward", p.name);
      const int ran = run(what, {op}, {1, 2}, {dx, dy});
      report(what, ran, ran == 1 ? max_rel(from_dev(dy, want_y.size()), want_y) : 0, 1e-5);
      for (Desc d : {x, y, op}) cudnnBackendDestroyDescriptor(d);
      cudaFree(dx), cudaFree(dy);
    }
    {
      // x and y too, as the hardware requires for NCHW.
      Desc gy = tensor(1, yd), gx = tensor(2, xd), x = tensor(3, xd), y = tensor(4, yd);
      Desc op = make(CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR);
      set_desc(op, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC, rd);
      set_desc(op, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC, gy);
      set_desc(op, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC, gx);
      set_desc(op, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC, x);
      set_desc(op, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC, y);
      std::vector<float> fy(want_y.begin(), want_y.end());
      float *ddy = to_dev(hdy), *dgx = to_dev(std::vector<float>(count(xd), 7.0f)), *dx = to_dev(hx), *dyv = to_dev(fy);
      const std::vector<int64_t> uids = {1, 2, 3, 4};
      const std::vector<void*> ptrs = {ddy, dgx, dx, dyv};
      cudnnBackendFinalize(op);
      char what[96];
      std::snprintf(what, sizeof what, "graph: %s pooling backward", p.name);
      const int ran = run(what, {op}, uids, ptrs);
      report(what, ran, ran == 1 ? max_rel(from_dev(dgx, want_dx.size()), want_dx) : 0, 1e-5);
      for (Desc d : {gy, gx, x, y, op}) cudnnBackendDestroyDescriptor(d);
      cudaFree(ddy), cudaFree(dgx), cudaFree(dx), cudaFree(dyv);
    }
    cudnnBackendDestroyDescriptor(rd);
  }
}

/* ---- asymmetric convolution padding, and concatenation ---- */

static void asymmetric_and_concat() {
  {
    // TensorFlow's SAME padding of an even filter: one more row after than before.
    const int N = 1, C = 2, Hh = 5, W = 6, K = 3, R = 2;
    const int64_t pre[2] = {0, 1}, post[2] = {1, 0}, ones[2] = {1, 1}, spatial = 2;
    const int Ho = Hh + pre[0] + post[0] - R + 1, Wo = W + pre[1] + post[1] - R + 1;
    const std::vector<int64_t> xd = {N, C, Hh, W}, wd = {K, C, R, R}, yd = {N, K, Ho, Wo};
    const auto hx = filled(count(xd), 1.0f, 1), hw = filled(count(wd), 0.5f, 2);
    Desc x = tensor(1, xd), w = tensor(2, wd), y = tensor(3, yd);
    Desc cd = make(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
    const cudnnDataType_t f = CUDNN_DATA_FLOAT;
    const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
    set(cd, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &f);
    set(cd, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode);
    set(cd, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial);
    set(cd, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, pre);
    set(cd, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, post);
    set(cd, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, ones);
    set(cd, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, ones);
    cudnnBackendFinalize(cd);
    Desc op = make(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    const float one = 1.0f, zero = 0.0f;
    set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, x);
    set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, w);
    set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, y);
    set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, cd);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &one);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &zero);
    cudnnBackendFinalize(op);
    float *dx = to_dev(hx), *dw = to_dev(hw), *dy = to_dev(std::vector<float>(count(yd), 7.0f));
    const int ran = run("asymmetric padding", {op}, {1, 2, 3}, {dx, dw, dy});
    std::vector<double> want(count(yd), 0.0);
    for (int k = 0; k < K; ++k)
      for (int i = 0; i < Ho; ++i)
        for (int j = 0; j < Wo; ++j) {
          double acc = 0;
          for (int c = 0; c < C; ++c)
            for (int r = 0; r < R; ++r)
              for (int q = 0; q < R; ++q) {
                const int ii = i - (int)pre[0] + r, jj = j - (int)pre[1] + q;
                if (ii < 0 || ii >= Hh || jj < 0 || jj >= W) continue;
                acc += (double)hx[(c * Hh + ii) * W + jj] * hw[((k * C + c) * R + r) * R + q];
              }
          want[(k * Ho + i) * Wo + j] = acc;
        }
    report("graph: convolution with asymmetric padding", ran, ran == 1 ? max_rel(from_dev(dy, want.size()), want) : 0,
           1e-5);
    for (Desc d : {x, w, y, cd, op}) cudnnBackendDestroyDescriptor(d);
    cudaFree(dx), cudaFree(dw), cudaFree(dy);
  }
  {
    const std::vector<int64_t> ad = {2, 3, 4}, bd = {2, 5, 4}, cd = {2, 8, 4};
    const auto ha = filled(count(ad), 1.0f, 1), hb = filled(count(bd), 1.0f, 2);
    Desc a = tensor(1, ad), b = tensor(2, bd), c = tensor(3, cd);
    Desc op = make(CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR);
    const int64_t axis = 1;
    const Desc ins[2] = {a, b};
    set(op, CUDNN_ATTR_OPERATION_CONCAT_AXIS, CUDNN_TYPE_INT64, 1, &axis);
    set(op, CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 2, ins);
    set_desc(op, CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC, c);
    cudnnBackendFinalize(op);
    float *da = to_dev(ha), *db = to_dev(hb), *dc = to_dev(std::vector<float>(count(cd), 7.0f));
    const int ran = run("concatenate", {op}, {1, 2, 3}, {da, db, dc});
    std::vector<double> want(count(cd));
    for (int n = 0; n < 2; ++n)
      for (int k = 0; k < 8; ++k)
        for (int j = 0; j < 4; ++j)
          want[(n * 8 + k) * 4 + j] = k < 3 ? ha[(n * 3 + k) * 4 + j] : hb[(n * 5 + k - 3) * 4 + j];
    report("graph: concatenation along the channels", ran, ran == 1 ? max_rel(from_dev(dc, want.size()), want) : 0,
           1e-7);
    for (Desc d : {a, b, c, op}) cudnnBackendDestroyDescriptor(d);
    cudaFree(da), cudaFree(db), cudaFree(dc);
  }
}

/* ---- the graph API's other operations ---- */

// A tensor of any type and strides (packed when strides is empty).
static Desc tensor_of(int64_t uid, const std::vector<int64_t>& dims, cudnnDataType_t t, std::vector<int64_t> strides = {},
                      bool is_virtual = false) {
  Desc d = make(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
  const int64_t align = 16;
  if (strides.empty()) strides = packed(dims);
  set(d, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &t);
  set(d, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, (int64_t)dims.size(), dims.data());
  set(d, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, (int64_t)strides.size(), strides.data());
  set(d, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
  set(d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align);
  set(d, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &is_virtual);
  cudnnBackendFinalize(d);
  return d;
}

// Max pooling in NHWC with the INT8 index tensor: the maximum's position in
// its window, row-major over the window's taps, padded taps counted (as an
// RTX 3060 writes it), and the backward pass taken from the index alone.
static void pooling_index() {
  const int N = 1, C = 2, Hh = 4, W = 6, wh = 2, ww = 3, sh = 2, sw = 2, ph = 0, pw = 1;
  const int Ho = 1 + (Hh + 2 * ph - wh) / sh, Wo = 1 + (W + 2 * pw - ww) / sw;
  const std::vector<int64_t> xd = {N, C, Hh, W}, yd = {N, C, Ho, Wo};
  const std::vector<int64_t> xs = {Hh * W * C, 1, W * C, C}, ys = {Ho * Wo * C, 1, Wo * C, C};
  const auto hx = filled(count(xd), 2.0f, 9);  // logical NCHW order
  // In memory, NHWC.
  std::vector<float> mx(hx.size());
  for (int c = 0; c < C; ++c)
    for (int h = 0; h < Hh; ++h)
      for (int w = 0; w < W; ++w) mx[(h * W + w) * C + c] = hx[(c * Hh + h) * W + w];
  std::vector<double> want_y(count(yd)), want_i(count(yd)), want_dx(count(xd), 0.0);
  const auto hdy = filled(count(yd), 1.0f, 4);
  for (int c = 0; c < C; ++c)
    for (int i = 0; i < Ho; ++i)
      for (int j = 0; j < Wo; ++j) {
        double best = 0;
        int arg = -1, at = -1;
        for (int a = 0; a < wh; ++a)
          for (int b = 0; b < ww; ++b) {
            const int r = i * sh - ph + a, q = j * sw - pw + b;
            if (r < 0 || r >= Hh || q < 0 || q >= W) continue;
            const double v = hx[(c * Hh + r) * W + q];
            if (arg < 0 || v > best) best = v, arg = a * ww + b, at = (c * Hh + r) * W + q;
          }
        const int yi = (c * Ho + i) * Wo + j;
        want_y[yi] = best, want_i[yi] = arg;
        want_dx[at] += hdy[yi];
      }
  Desc rd = make(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
  const cudnnResampleMode_t mode = CUDNN_RESAMPLE_MAXPOOL;
  const cudnnPaddingMode_t pad = CUDNN_NEG_INF_PAD;
  const int64_t nsp = 2, w2[2] = {wh, ww}, p2[2] = {ph, pw}, s2[2] = {sh, sw};
  const cudnnDataType_t f = CUDNN_DATA_FLOAT;
  const cudnnNanPropagation_t nan = CUDNN_NOT_PROPAGATE_NAN;
  set(rd, CUDNN_ATTR_RESAMPLE_MODE, CUDNN_TYPE_RESAMPLE_MODE, 1, &mode);
  set(rd, CUDNN_ATTR_RESAMPLE_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &f);
  set(rd, CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION, CUDNN_TYPE_NAN_PROPOGATION, 1, &nan);
  set(rd, CUDNN_ATTR_RESAMPLE_PADDING_MODE, CUDNN_TYPE_PADDING_MODE, 1, &pad);
  set(rd, CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &nsp);
  set(rd, CUDNN_ATTR_RESAMPLE_WINDOW_DIMS, CUDNN_TYPE_INT64, 2, w2);
  set(rd, CUDNN_ATTR_RESAMPLE_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, p2);
  set(rd, CUDNN_ATTR_RESAMPLE_POST_PADDINGS, CUDNN_TYPE_INT64, 2, p2);
  set(rd, CUDNN_ATTR_RESAMPLE_STRIDES, CUDNN_TYPE_INT64, 2, s2);
  cudnnBackendFinalize(rd);
  Desc x = tensor_of(1, xd, CUDNN_DATA_FLOAT, xs), y = tensor_of(2, yd, CUDNN_DATA_FLOAT, ys),
       ix = tensor_of(3, yd, CUDNN_DATA_INT8, ys);
  Desc op = make(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR);
  set_desc(op, CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC, rd);
  set_desc(op, CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, x);
  set_desc(op, CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, y);
  set_desc(op, CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC, ix);
  cudnnBackendFinalize(op);
  float *dx = to_dev(mx), *dy = to_dev(std::vector<float>(count(yd), 7.0f));
  int8_t* di = nullptr;
  cudaMalloc(&di, count(yd) + 64);
  cudaMemset(di, 0x7f, count(yd));
  const int ran = run("graph: NHWC max pooling with its index tensor", {op}, {1, 2, 3}, {dx, dy, di});
  double err = 0;
  std::vector<int8_t> hi(count(yd));
  if (ran == 1) {
    const auto gy = from_dev(dy, count(yd));
    cudaMemcpy(hi.data(), di, hi.size(), cudaMemcpyDeviceToHost);
    for (int c = 0; c < C; ++c)
      for (int i = 0; i < Ho; ++i)
        for (int j = 0; j < Wo; ++j) {
          const int yi = (c * Ho + i) * Wo + j, mi = (i * Wo + j) * C + c;
          err = std::fmax(err, std::fabs(gy[mi] - want_y[yi]) + std::fabs(hi[mi] - want_i[yi]));
        }
  }
  report("graph: NHWC max pooling with its index tensor", ran, err, 1e-6);
  // Backward from the index alone (no x): the RTX 3060 offers no engine for
  // it, so on the hardware this is skipped.
  {
    std::vector<int8_t> idx_mem(count(yd));
    std::vector<float> dy_mem(count(yd));
    for (int c = 0; c < C; ++c)
      for (int i = 0; i < Ho; ++i)
        for (int j = 0; j < Wo; ++j) {
          const int yi = (c * Ho + i) * Wo + j, mi = (i * Wo + j) * C + c;
          idx_mem[mi] = (int8_t)want_i[yi], dy_mem[mi] = hdy[yi];
        }
    cudaMemcpy(di, idx_mem.data(), idx_mem.size(), cudaMemcpyHostToDevice);
    Desc gy = tensor_of(4, yd, CUDNN_DATA_FLOAT, ys), gx = tensor_of(5, xd, CUDNN_DATA_FLOAT, xs);
    Desc bop = make(CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR);
    set_desc(bop, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC, rd);
    set_desc(bop, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC, gy);
    set_desc(bop, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC, gx);
    set_desc(bop, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC, ix);
    cudnnBackendFinalize(bop);
    float *ddy = to_dev(dy_mem), *dgx = to_dev(std::vector<float>(count(xd), 7.0f));
    const int bran = run("graph: max pooling backward from the index tensor", {bop}, {4, 5, 3}, {ddy, dgx, di});
    double berr = 0;
    if (bran == 1) {
      const auto g = from_dev(dgx, count(xd));
      for (int c = 0; c < C; ++c)
        for (int h = 0; h < Hh; ++h)
          for (int w = 0; w < W; ++w) berr = std::fmax(berr, std::fabs(g[(h * W + w) * C + c] - want_dx[(c * Hh + h) * W + w]));
    }
    report("graph: max pooling backward from the index tensor", bran, berr, 1e-6);
    for (Desc d : {gy, gx, bop}) cudnnBackendDestroyDescriptor(d);
    cudaFree(ddy), cudaFree(dgx);
  }
  for (Desc d : {rd, x, y, ix, op}) cudnnBackendDestroyDescriptor(d);
  cudaFree(dx), cudaFree(dy), cudaFree(di);
}

// Group normalization (the norm operation's GROUP_NORM mode, the groups
// given by the statistics' [N, G, 1, 1] shape) forward and backward, and
// layer normalization's backward pass without the saved statistics (from x
// and epsilon).
static void group_and_unsaved_norms() {
  const int N = 2, C = 6, G = 3, S = 4;
  const std::vector<int64_t> xd = {N, C, S, 1}, pd = {1, C, 1, 1}, sd = {N, G, 1, 1};
  const float eps = 1e-5f;
  const auto hx = filled(count(xd), 1.5f, 11), hs = filled(C, 0.5f, 12), hb = filled(C, 0.3f, 13), hdy = filled(count(xd), 0.8f, 14);
  // Host reference: groups of C / G consecutive channels.
  const int cg = C / G;
  std::vector<double> mean(N * G, 0), var(N * G, 0), inv(N * G);
  auto slot = [&](size_t i) { const int n = (int)(i / (C * S)), c = (int)(i / S) % C; return n * G + c / cg; };
  for (size_t i = 0; i < hx.size(); ++i) mean[slot(i)] += hx[i] / (cg * S);
  for (size_t i = 0; i < hx.size(); ++i) var[slot(i)] += (hx[i] - mean[slot(i)]) * (hx[i] - mean[slot(i)]) / (cg * S);
  for (int k = 0; k < N * G; ++k) inv[k] = 1 / std::sqrt(var[k] + eps);
  std::vector<double> want_y(hx.size());
  for (size_t i = 0; i < hx.size(); ++i) {
    const int c = (int)(i / S) % C;
    want_y[i] = hs[c] * (hx[i] - mean[slot(i)]) * inv[slot(i)] + hb[c];
  }
  const cudnnBackendNormMode_t gm = CUDNN_GROUP_NORM;
  Desc x = tensor(1, xd), sc = tensor(2, pd), b = tensor(3, pd), e = scalar_tensor(4), y = tensor(5, xd), m = tensor(6, sd),
       iv = tensor(7, sd);
  Desc op = make(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR);
  const cudnnBackendNormFwdPhase_t phase = CUDNN_NORM_FWD_TRAINING;
  set(op, CUDNN_ATTR_OPERATION_NORM_FWD_MODE, CUDNN_TYPE_NORM_MODE, 1, &gm);
  set(op, CUDNN_ATTR_OPERATION_NORM_FWD_PHASE, CUDNN_TYPE_NORM_FWD_PHASE, 1, &phase);
  set_desc(op, CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, x);
  set_desc(op, CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, sc);
  set_desc(op, CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, b);
  set_desc(op, CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, e);
  set_desc(op, CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, y);
  set_desc(op, CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC, m);
  set_desc(op, CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC, iv);
  const cudnnStatus_t fs = cudnnBackendFinalize(op);
  float *dx = to_dev(hx), *ds = to_dev(hs), *db = to_dev(hb), *dyv = to_dev(std::vector<float>(hx.size(), 7.0f)),
        *dm = to_dev(std::vector<float>(N * G, 7.0f)), *di = to_dev(std::vector<float>(N * G, 7.0f));
  int ran = fs == CUDNN_STATUS_SUCCESS ? run("graph: group normalization forward (training)", {op}, {1, 2, 3, 4, 5, 6, 7},
                                             {dx, ds, db, (void*)&eps, dyv, dm, di})
                                       : 0;
  double err = 0;
  if (ran == 1)
    err = std::fmax(std::fmax(max_rel(from_dev(dyv, hx.size()), want_y), max_rel(from_dev(dm, N * G), mean)),
                    max_rel(from_dev(di, N * G), inv));
  report("graph: group normalization forward (training)", ran, err, 1e-4);

  // Backward: with the saved statistics (group norm), and without them
  // (layer norm, recomputed from x and epsilon).
  auto backward_ref = [&](const std::vector<double>& mn, const std::vector<double>& in, auto slotf, int per,
                          std::vector<double>* gx, std::vector<double>* gs, std::vector<double>* gb) {
    const size_t ns = mn.size();
    std::vector<double> mg(ns, 0), mgx(ns, 0);
    gx->assign(hx.size(), 0), gs->assign(C, 0), gb->assign(C, 0);
    for (size_t i = 0; i < hx.size(); ++i) {
      const int k = slotf(i), c = (int)(i / S) % C;
      const double xh = (hx[i] - mn[k]) * in[k], g = hdy[i] * hs[c];
      mg[k] += g, mgx[k] += g * xh, (*gs)[c] += hdy[i] * xh, (*gb)[c] += hdy[i];
    }
    for (size_t i = 0; i < hx.size(); ++i) {
      const int k = slotf(i), c = (int)(i / S) % C;
      const double xh = (hx[i] - mn[k]) * in[k], g = hdy[i] * hs[c];
      (*gx)[i] = in[k] * (g - mg[k] / per - xh * mgx[k] / per);
    }
  };
  for (int unsaved = 0; unsaved < 2; ++unsaved) {
    const char* what = unsaved ? "graph: layer normalization backward without saved statistics"
                               : "graph: group normalization backward";
    const cudnnBackendNormMode_t mode = unsaved ? CUDNN_LAYER_NORM : CUDNN_GROUP_NORM;
    std::vector<double> lmean(N, 0), lvar(N, 0), linv(N);
    for (size_t i = 0; i < hx.size(); ++i) lmean[i / (C * S)] += hx[i] / (C * S);
    for (size_t i = 0; i < hx.size(); ++i) lvar[i / (C * S)] += (hx[i] - lmean[i / (C * S)]) * (hx[i] - lmean[i / (C * S)]) / (C * S);
    for (int n = 0; n < N; ++n) linv[n] = 1 / std::sqrt(lvar[n] + eps);
    std::vector<double> wgx, wgs, wgb;
    if (unsaved) backward_ref(lmean, linv, [&](size_t i) { return (int)(i / (C * S)); }, C * S, &wgx, &wgs, &wgb);
    else backward_ref(mean, inv, slot, cg * S, &wgx, &wgs, &wgb);
    Desc gy = tensor(13, xd), gx = tensor(14, xd), gs = tensor(15, pd), gb = tensor(16, pd);
    Desc bop = make(CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR);
    set(bop, CUDNN_ATTR_OPERATION_NORM_BWD_MODE, CUDNN_TYPE_NORM_MODE, 1, &mode);
    set_desc(bop, CUDNN_ATTR_OPERATION_NORM_BWD_XDESC, x);
    set_desc(bop, CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC, gy);
    set_desc(bop, CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC, sc);
    if (unsaved) {
      set_desc(bop, CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC, e);
    } else {
      set_desc(bop, CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC, m);
      set_desc(bop, CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC, iv);
    }
    set_desc(bop, CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC, gx);
    set_desc(bop, CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC, gs);
    set_desc(bop, CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC, gb);
    const cudnnStatus_t bs = cudnnBackendFinalize(bop);
    std::vector<float> fm(mean.begin(), mean.end()), fi(inv.begin(), inv.end());
    float *ddy = to_dev(hdy), *dgx = to_dev(std::vector<float>(hx.size(), 7.0f)), *dgs = to_dev(std::vector<float>(C, 7.0f)),
          *dgb = to_dev(std::vector<float>(C, 7.0f)), *sm = to_dev(fm), *si = to_dev(fi);
    std::vector<int64_t> uids = {1, 13, 2, 14, 15, 16};
    std::vector<void*> ptrs = {dx, ddy, ds, dgx, dgs, dgb};
    if (unsaved) uids.push_back(4), ptrs.push_back((void*)&eps);
    else uids.push_back(6), ptrs.push_back(sm), uids.push_back(7), ptrs.push_back(si);
    const int bran = bs == CUDNN_STATUS_SUCCESS ? run(what, {bop}, uids, ptrs) : 0;
    double berr = 0;
    if (bran == 1)
      berr = std::fmax(std::fmax(max_rel(from_dev(dgx, hx.size()), wgx), max_rel(from_dev(dgs, C), wgs)), max_rel(from_dev(dgb, C), wgb));
    report(what, bran, berr, 1e-4);
    for (Desc d : {gy, gx, gs, gb, bop}) cudnnBackendDestroyDescriptor(d);
    for (float* p : {ddy, dgx, dgs, dgb, sm, si}) cudaFree(p);
  }
  for (Desc d : {x, sc, b, e, y, m, iv, op}) cudnnBackendDestroyDescriptor(d);
  for (float* p : {dx, ds, db, dyv, dm, di}) cudaFree(p);
}

// Statistics generation: per-channel sum and sum of squares.
static void genstats() {
  const std::vector<int64_t> xd = {2, 3, 4, 4}, cd = {1, 3, 1, 1};
  const auto hx = filled(count(xd), 1.0f, 21);
  std::vector<double> sum(3, 0), sq(3, 0);
  for (size_t i = 0; i < hx.size(); ++i) sum[(i / 16) % 3] += hx[i], sq[(i / 16) % 3] += hx[i] * hx[i];
  Desc x = tensor(1, xd), s = tensor(2, cd), q = tensor(3, cd);
  Desc op = make(CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR);
  const cudnnGenStatsMode_t mode = CUDNN_GENSTATS_SUM_SQSUM;
  const cudnnDataType_t f = CUDNN_DATA_FLOAT;
  set(op, CUDNN_ATTR_OPERATION_GENSTATS_MODE, CUDNN_TYPE_GENSTATS_MODE, 1, &mode);
  set(op, CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &f);
  set_desc(op, CUDNN_ATTR_OPERATION_GENSTATS_XDESC, x);
  set_desc(op, CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC, s);
  set_desc(op, CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC, q);
  const cudnnStatus_t st = cudnnBackendFinalize(op);
  float *dx = to_dev(hx), *dsum = to_dev(std::vector<float>(3, 7.0f)), *dsq = to_dev(std::vector<float>(3, 7.0f));
  const int ran = st == CUDNN_STATUS_SUCCESS ? run("graph: statistics generation (sum, sum of squares)", {op}, {1, 2, 3}, {dx, dsum, dsq}) : 0;
  report("graph: statistics generation (sum, sum of squares)", ran,
         ran == 1 ? std::fmax(max_rel(from_dev(dsum, 3), sum), max_rel(from_dev(dsq, 3), sq)) : 0, 1e-5);
  for (Desc d : {x, s, q, op}) cudnnBackendDestroyDescriptor(d);
  cudaFree(dx), cudaFree(dsum), cudaFree(dsq);
}

// The RNG operation (Philox; the hardware offers no engine for it alone, so
// this checks VirtualGPU's own: Bernoulli's rate and values, uniform's
// range, and that the seed and offset decide the stream).
static void rng() {
  if (!kOnSim) {
    std::printf("skip the RNG operation alone (no engine for it on this GPU)\n");
    return;
  }
  const std::vector<int64_t> yd = {1, 1, 64, 64};
  auto draw = [&](cudnnRngDistribution_t dist, int64_t seed, int64_t offset, std::vector<float>* out) {
    Desc rd = make(CUDNN_BACKEND_RNG_DESCRIPTOR);
    const double p = 0.3, lo = 2.0, hi = 5.0;
    set(rd, CUDNN_ATTR_RNG_DISTRIBUTION, CUDNN_TYPE_RNG_DISTRIBUTION, 1, &dist);
    set(rd, CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY, CUDNN_TYPE_DOUBLE, 1, &p);
    set(rd, CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM, CUDNN_TYPE_DOUBLE, 1, &lo);
    set(rd, CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM, CUDNN_TYPE_DOUBLE, 1, &hi);
    cudnnBackendFinalize(rd);
    Desc y = tensor(1, yd), off = tensor_of(2, {1, 1, 1, 1}, CUDNN_DATA_INT64);
    Desc op = make(CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR);
    set_desc(op, CUDNN_ATTR_OPERATION_RNG_DESC, rd);
    set_desc(op, CUDNN_ATTR_OPERATION_RNG_YDESC, y);
    set(op, CUDNN_ATTR_OPERATION_RNG_SEED, CUDNN_TYPE_INT64, 1, &seed);
    set_desc(op, CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC, off);
    cudnnBackendFinalize(op);
    float* dy = to_dev(std::vector<float>(count(yd), -1.0f));
    int64_t* doff = nullptr;
    cudaMalloc(&doff, 8);
    cudaMemcpy(doff, &offset, 8, cudaMemcpyHostToDevice);
    const int ran = run("graph: RNG", {op}, {1, 2}, {dy, doff});
    *out = from_dev(dy, count(yd));
    for (Desc d : {rd, y, off, op}) cudnnBackendDestroyDescriptor(d);
    cudaFree(dy), cudaFree(doff);
    return ran;
  };
  std::vector<float> a, b, c, u;
  const int r1 = draw(CUDNN_RNG_DISTRIBUTION_BERNOULLI, 42, 0, &a), r2 = draw(CUDNN_RNG_DISTRIBUTION_BERNOULLI, 42, 0, &b),
            r3 = draw(CUDNN_RNG_DISTRIBUTION_BERNOULLI, 42, 4096, &c), r4 = draw(CUDNN_RNG_DISTRIBUTION_UNIFORM, 7, 0, &u);
  double ones = 0;
  bool binary = true, range = true;
  for (float v : a) ones += v, binary &= v == 0.0f || v == 1.0f;
  for (float v : u) range &= v >= 2.0f && v < 5.0f;
  expect("graph: RNG runs", r1 == 1 && r2 == 1 && r3 == 1 && r4 == 1);
  expect("graph: RNG Bernoulli draws 0 or 1, about 30% ones", binary && std::fabs(ones / a.size() - 0.3) < 0.03, ones / a.size());
  expect("graph: RNG repeats a seed and offset, and a new offset changes it", a == b && a != c);
  expect("graph: RNG uniform stays in [min, max)", range);
}

// Reshape (a view: a transpose through permuted strides), transpose and
// slice, chained through virtual tensors.
static void data_movement() {
  const int64_t A = 2, B = 3, Cc = 4;
  const auto hx = filled(A * B * Cc, 1.0f, 31);
  // y[a, c, b] = x[a, b, c] by a view-only reshape whose strides permute
  // x's: [A, Cc, B] with strides {B*Cc, 1, Cc}.
  Desc x = tensor(1, {A, B, Cc}), v = tensor_of(2, {A, Cc, B}, CUDNN_DATA_FLOAT, {B * Cc, 1, Cc}, true), y = tensor(3, {A, Cc, B});
  Desc rs = make(CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR);
  set_desc(rs, CUDNN_ATTR_OPERATION_RESHAPE_XDESC, x);
  set_desc(rs, CUDNN_ATTR_OPERATION_RESHAPE_YDESC, v);
  cudnnBackendFinalize(rs);
  Desc id = pointwise(CUDNN_POINTWISE_IDENTITY, v, nullptr, y);
  std::vector<double> want(hx.size());
  for (int64_t a = 0; a < A; ++a)
    for (int64_t b = 0; b < B; ++b)
      for (int64_t c = 0; c < Cc; ++c) want[(a * Cc + c) * B + b] = hx[(a * B + b) * Cc + c];
  float *dx = to_dev(hx), *dy = to_dev(std::vector<float>(hx.size(), 7.0f));
  const int ran = run("graph: reshape (a transposing view) + identity", {rs, id}, {1, 3}, {dx, dy});
  report("graph: reshape (a transposing view) + identity", ran, ran == 1 ? max_rel(from_dev(dy, want.size()), want) : 0, 1e-6);
  for (Desc d : {x, v, y, rs, id}) cudnnBackendDestroyDescriptor(d);
  cudaFree(dy);
  // transpose {2, 0, 1} then slice [.., 1:3, ::2]: [Cc, A, B] -> [Cc, A, 2]... over B with start 0 step 2.
  Desc x2 = tensor(1, {A, B, Cc}), t = tensor_of(2, {Cc, A, B}, CUDNN_DATA_FLOAT, {}, true), s = tensor(3, {2, A, 2});
  Desc tr = make(CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR);
  const int64_t perm[3] = {2, 0, 1};
  set_desc(tr, CUDNN_ATTR_OPERATION_TRANSPOSE_XDESC, x2);
  set_desc(tr, CUDNN_ATTR_OPERATION_TRANSPOSE_YDESC, t);
  set(tr, CUDNN_ATTR_OPERATION_TRANSPOSE_PERMUTATION, CUDNN_TYPE_INT64, 3, perm);
  cudnnBackendFinalize(tr);
  Desc sl = make(CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR);
  const int64_t st[3] = {1, 0, 0}, lim[3] = {3, A, B}, str[3] = {1, 1, 2};
  set_desc(sl, CUDNN_ATTR_OPERATION_SLICE_XDESC, t);
  set_desc(sl, CUDNN_ATTR_OPERATION_SLICE_YDESC, s);
  set(sl, CUDNN_ATTR_OPERATION_SLICE_START_INDICES, CUDNN_TYPE_INT64, 3, st);
  set(sl, CUDNN_ATTR_OPERATION_SLICE_LIMIT_INDICES, CUDNN_TYPE_INT64, 3, lim);
  set(sl, CUDNN_ATTR_OPERATION_SLICE_STRIDES, CUDNN_TYPE_INT64, 3, str);
  cudnnBackendFinalize(sl);
  std::vector<double> want2(2 * A * 2);
  for (int64_t c = 0; c < 2; ++c)
    for (int64_t a = 0; a < A; ++a)
      for (int64_t b = 0; b < 2; ++b) want2[(c * A + a) * 2 + b] = hx[(a * B + 2 * b) * Cc + (c + 1)];
  float* ds = to_dev(std::vector<float>(want2.size(), 7.0f));
  const int ran2 = run("graph: transpose + slice", {tr, sl}, {1, 3}, {dx, ds});
  report("graph: transpose + slice", ran2, ran2 == 1 ? max_rel(from_dev(ds, want2.size()), want2) : 0, 1e-6);
  for (Desc d : {x2, t, s, tr, sl}) cudnnBackendDestroyDescriptor(d);
  cudaFree(dx), cudaFree(ds);
}

// A vectorized INT8 convolution: tensors whose channel dimension holds
// vectors of 4 (the graph API's form of NCHW_VECT_C / INT8x4, the DP4A
// path), dims and strides counting vectors, accumulated in INT32 and
// saturated to INT8.
static Desc vtensor(int64_t uid, const std::vector<int64_t>& dims) {
  Desc d = make(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
  const cudnnDataType_t t = CUDNN_DATA_INT8;
  const int64_t align = 16, vc = 4, vd = 1;
  const std::vector<int64_t> strides = packed(dims);
  const bool v = false;
  set(d, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &t);
  set(d, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, (int64_t)dims.size(), dims.data());
  set(d, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, (int64_t)strides.size(), strides.data());
  set(d, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
  set(d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align);
  set(d, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &v);
  set(d, CUDNN_ATTR_TENSOR_VECTOR_COUNT, CUDNN_TYPE_INT64, 1, &vc);
  set(d, CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION, CUDNN_TYPE_INT64, 1, &vd);
  cudnnBackendFinalize(d);
  return d;
}
static void vectorized_conv() {
  const int C = 8, K = 8, Hh = 5, W = 5, R = 3, pad = 1, cv = C / 4, kv = K / 4;
  // Memory order: [n][c/4][h][w][4] and [k][c/4][r][s][4].
  std::vector<int8_t> hx(C * Hh * W), hw(K * C * R * R);
  auto xat = [&](int c, int h, int w) { return ((c / 4) * Hh * W + h * W + w) * 4 + c % 4; };
  auto wat = [&](int k, int c, int r, int q) { return ((k * cv + c / 4) * R * R + r * R + q) * 4 + c % 4; };
  auto yat = [&](int k, int h, int w) { return ((k / 4) * Hh * W + h * W + w) * 4 + k % 4; };
  for (int c = 0; c < C; ++c)
    for (int h = 0; h < Hh; ++h)
      for (int w = 0; w < W; ++w) hx[xat(c, h, w)] = (int8_t)((c * 7 + h * 3 + w * 5) % 5 - 2);
  for (int k = 0; k < K; ++k)
    for (int c = 0; c < C; ++c)
      for (int r = 0; r < R; ++r)
        for (int q = 0; q < R; ++q) hw[wat(k, c, r, q)] = (int8_t)((k * 5 + c * 3 + r * 2 + q + (k > 5 ? 1 : 0)) % 3 - (k > 5 ? 0 : 1));
  std::vector<int> want(K * Hh * W);
  for (int k = 0; k < K; ++k)
    for (int h = 0; h < Hh; ++h)
      for (int w = 0; w < W; ++w) {
        int acc = 0;
        for (int c = 0; c < C; ++c)
          for (int r = 0; r < R; ++r)
            for (int q = 0; q < R; ++q) {
              const int ih = h - pad + r, iw = w - pad + q;
              if (ih >= 0 && ih < Hh && iw >= 0 && iw < W) acc += hx[xat(c, ih, iw)] * hw[wat(k, c, r, q)];
            }
        want[yat(k, h, w)] = acc > 127 ? 127 : acc < -128 ? -128 : acc;
      }
  Desc x = vtensor(1, {1, cv, Hh, W}), wd = vtensor(2, {K, cv, R, R}), y = vtensor(3, {1, kv, Hh, W});
  Desc cd = make(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
  const cudnnDataType_t i32 = CUDNN_DATA_INT32;
  const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
  const int64_t spatial = 2, pads[2] = {pad, pad}, ones[2] = {1, 1};
  set(cd, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &i32);
  set(cd, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode);
  set(cd, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial);
  set(cd, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, pads);
  set(cd, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, pads);
  set(cd, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, ones);
  set(cd, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, ones);
  cudnnBackendFinalize(cd);
  Desc op = make(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
  const float alpha = 1.0f, beta = 0.0f;
  set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, x);
  set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, wd);
  set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, y);
  set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, cd);
  set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha);
  set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta);
  const cudnnStatus_t st = cudnnBackendFinalize(op);
  int8_t *dx, *dw, *dy;
  cudaMalloc(&dx, hx.size() + 64), cudaMalloc(&dw, hw.size() + 64), cudaMalloc(&dy, want.size() + 64);
  cudaMemcpy(dx, hx.data(), hx.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice);
  cudaMemset(dy, 0x55, want.size());
  const int ran = st == CUDNN_STATUS_SUCCESS ? run("graph: INT8x4 vectorized convolution", {op}, {1, 2, 3}, {dx, dw, dy}) : 0;
  double err = 0;
  if (ran == 1) {
    std::vector<int8_t> got(want.size());
    cudaMemcpy(got.data(), dy, got.size(), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < got.size(); ++i) err = std::fmax(err, std::fabs(got[i] - want[i]));
  }
  report("graph: INT8x4 vectorized convolution", ran, err, 0.5);
  for (Desc d : {x, wd, y, cd, op}) cudnnBackendDestroyDescriptor(d);
  cudaFree(dx), cudaFree(dw), cudaFree(dy);
}

// An operation this library has no engine for is refused when it is
// finalized, by name, not accepted and run wrongly. (NVIDIA's library has an
// engine for it, so this is VirtualGPU's own contract.)
static void refusals() {
  if (!kOnSim) {
    std::printf("skip refusal of an operation with no engine (VirtualGPU's contract)\n");
    return;
  }
  Desc q = make(CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR);
  expect("an operation with no engine here is refused, not run",
         cudnnBackendFinalize(q) == CUDNN_STATUS_NOT_SUPPORTED);
  cudnnBackendDestroyDescriptor(q);
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
  norms();
  pooling();
  asymmetric_and_concat();
  pooling_index();
  group_and_unsaved_norms();
  genstats();
  rng();
  data_movement();
  vectorized_conv();
  refusals();
  cudnnDestroy(H);
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
