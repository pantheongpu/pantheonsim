// The library paths PyTorch's CUDA build takes, called directly, each checked
// against a reference computed here on the host (or, for gradients, against
// finite differences of the library's own forward pass). No framework, so it
// runs on every CI machine; the same program passes on a real GPU with
// NVIDIA's libraries, since it checks documented behaviour, not VirtualGPU's.
//
//   - cuDNN's backend (graph) API: convolution forward, backward-data and
//     backward-filter; groups, stride, dilation, padding, channels-last
//     strides, a 3-D convolution; descriptors destroyed once the graph holds
//     them; a graph cuDNN would need an engine this library lacks is refused.
//   - cuDNN's BatchNorm training forward and backward (the Ex forms), and
//     N-dimensional tensor descriptors.
//   - cuDNN's RNN API: LSTM (two layers, both directions, packed sequences of
//     different lengths) and GRU, forward against a reference and gradients
//     against finite differences; where cudnnGetRNNWeightParams says the
//     weights are is where the forward pass reads them.
//   - cuDNN's version, as cudnnGetVersion and cudnnGetProperty both give it.
//   - cuBLASLt's fused matmul + bias with a cuBLAS handle as its handle, as
//     PyTorch calls it.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

static int fails = 0;
static void expect(const char* what, bool ok, double detail = 0) {
  std::printf("%s %s", ok ? "ok  " : "FAIL", what);
  if (!ok && detail != 0) std::printf(" (%g)", detail);
  std::printf("\n");
  fails += !ok;
}
#define CK(x) do { cudnnStatus_t s_ = (x); if (s_ != CUDNN_STATUS_SUCCESS) { \
  std::printf("FAIL %s -> %d\n", #x, (int)s_); ++fails; return; } } while (0)

static float fill(int i, float scale) { return scale * std::sin(0.7f * i + 0.3f) + 0.05f * (i % 7); }
static std::vector<float> filled(size_t n, float scale) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = fill((int)i, scale);
  return v;
}
static float* to_dev(const std::vector<float>& h) {
  float* p = nullptr;
  cudaMalloc(&p, h.size() * sizeof(float));
  cudaMemcpy(p, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
  return p;
}
static std::vector<float> from_dev(const float* p, size_t n) {
  std::vector<float> h(n);
  cudaMemcpy(h.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost);
  return h;
}
static double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double d = a.size() == b.size() ? 0 : 1e30;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) d = std::fmax(d, std::fabs((double)a[i] - b[i]));
  return d;
}

/* ---- cuDNN's graph API ---------------------------------------------------- */

using Desc = cudnnBackendDescriptor_t;
static Desc make(cudnnBackendDescriptorType_t t) {
  Desc d = nullptr;
  cudnnBackendCreateDescriptor(t, &d);
  return d;
}
static void set(Desc d, cudnnBackendAttributeName_t n, cudnnBackendAttributeType_t t, int64_t c, const void* v) {
  cudnnBackendSetAttribute(d, n, t, c, v);
}
static void set_i64s(Desc d, cudnnBackendAttributeName_t n, const std::vector<int64_t>& v) {
  set(d, n, CUDNN_TYPE_INT64, (int64_t)v.size(), v.data());
}

static Desc tensor(int64_t uid, const std::vector<int64_t>& dims, const std::vector<int64_t>& strides) {
  Desc t = make(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
  const cudnnDataType_t f = CUDNN_DATA_FLOAT;
  const int64_t align = 16;
  set(t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &f);
  set_i64s(t, CUDNN_ATTR_TENSOR_DIMENSIONS, dims);
  set_i64s(t, CUDNN_ATTR_TENSOR_STRIDES, strides);
  set(t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
  set(t, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align);
  cudnnBackendFinalize(t);
  return t;
}
static std::vector<int64_t> packed(const std::vector<int64_t>& dims) {
  std::vector<int64_t> s(dims.size(), 1);
  for (size_t i = dims.size() - 1; i-- > 0;) s[i] = s[i + 1] * dims[i + 1];
  return s;
}
// NHWC-style strides for an NC(D)HW tensor: channels innermost.
static std::vector<int64_t> channels_last(const std::vector<int64_t>& d) {
  std::vector<int64_t> s(d.size());
  s[1] = 1;
  int64_t acc = d[1];
  for (size_t i = d.size(); i-- > 2;) s[i] = acc, acc *= d[i];
  s[0] = acc;
  return s;
}
static size_t count(const std::vector<int64_t>& d) {
  size_t n = 1;
  for (int64_t x : d) n *= (size_t)x;
  return n;
}
static size_t at(const std::vector<int64_t>& strides, const std::vector<int64_t>& idx) {
  size_t o = 0;
  for (size_t i = 0; i < idx.size(); ++i) o += (size_t)(idx[i] * strides[i]);
  return o;
}

enum class Dir { Fwd, Data, Filter };

struct ConvCase {
  const char* name;
  std::vector<int64_t> x, w;   // dims: N C spatial... and K C/g spatial...
  int64_t pad, stride, dil;
  bool nhwc;
};

// y[n,k,o] = sum x[n, g*Cg+c, o*s - p + f*d] * w[k,c,f]  (cross-correlation)
static void reference(const ConvCase& cc, const std::vector<int64_t>& yd, Dir dir, const std::vector<float>& x,
                      const std::vector<float>& w, const std::vector<float>& y, std::vector<double>* out,
                      const std::vector<int64_t>& xs, const std::vector<int64_t>& ys) {
  const int S = (int)cc.x.size() - 2;
  const int64_t N = cc.x[0], C = cc.x[1], K = cc.w[0], Cg = cc.w[1], Kg = K / (C / Cg);
  const std::vector<int64_t> ws = packed(cc.w);
  std::vector<int64_t> o(S, 0), f(S, 0), i(S, 0);
  std::function<void(int, int64_t, int64_t, int64_t)> walk;
  for (int64_t n = 0; n < N; ++n)
    for (int64_t k = 0; k < K; ++k)
      for (int64_t c = 0; c < Cg; ++c) {
        const int64_t in_c = (k / Kg) * Cg + c;
        // every output position and filter tap
        std::vector<int64_t> pos(2 * S, 0);
        for (;;) {
          bool inside = true;
          for (int d = 0; d < S; ++d) {
            o[d] = pos[d], f[d] = pos[S + d];
            i[d] = o[d] * cc.stride - cc.pad + f[d] * cc.dil;
            inside &= i[d] >= 0 && i[d] < cc.x[2 + d];
          }
          if (inside) {
            std::vector<int64_t> xi{n, in_c}, wi{k, c}, yi{n, k};
            for (int d = 0; d < S; ++d) xi.push_back(i[d]), wi.push_back(f[d]), yi.push_back(o[d]);
            const size_t X = at(xs, xi), W = at(ws, wi), Y = at(ys, yi);
            if (dir == Dir::Fwd) (*out)[Y] += (double)x[X] * w[W];
            else if (dir == Dir::Data) (*out)[X] += (double)y[Y] * w[W];
            else (*out)[W] += (double)y[Y] * x[X];
          }
          int d = 2 * S - 1;
          for (; d >= 0; --d) {
            const int64_t lim = d < S ? yd[2 + d] : cc.w[2 + d - S];
            if (++pos[d] < lim) break;
            pos[d] = 0;
          }
          if (d < 0) break;
        }
      }
}

static void conv(cudnnHandle_t h, const ConvCase& cc, Dir dir) {
  const int S = (int)cc.x.size() - 2;
  std::vector<int64_t> yd{cc.x[0], cc.w[0]};
  for (int d = 0; d < S; ++d) yd.push_back((cc.x[2 + d] + 2 * cc.pad - cc.dil * (cc.w[2 + d] - 1) - 1) / cc.stride + 1);
  const auto xs = cc.nhwc ? channels_last(cc.x) : packed(cc.x);
  const auto ys = cc.nhwc ? channels_last(yd) : packed(yd);
  const auto hx = filled(count(cc.x), 1.0f), hw = filled(count(cc.w), 0.5f), hy = filled(count(yd), 0.8f);
  float *dx = to_dev(hx), *dw = to_dev(hw), *dy = to_dev(hy);

  Desc tx = tensor(1, cc.x, xs), tw = tensor(2, cc.w, packed(cc.w)), ty = tensor(3, yd, ys);
  Desc cd = make(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
  const cudnnDataType_t f = CUDNN_DATA_FLOAT;
  const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
  const int64_t spatial = S;
  set(cd, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &f);
  set(cd, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode);
  set(cd, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial);
  set_i64s(cd, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, std::vector<int64_t>(S, cc.pad));
  set_i64s(cd, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, std::vector<int64_t>(S, cc.pad));
  set_i64s(cd, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, std::vector<int64_t>(S, cc.stride));
  set_i64s(cd, CUDNN_ATTR_CONVOLUTION_DILATIONS, std::vector<int64_t>(S, cc.dil));
  CK(cudnnBackendFinalize(cd));

  const float one = 1.0f, zero = 0.0f;
  Desc op;
  if (dir == Dir::Fwd) {
    op = make(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tx);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tw);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &ty);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cd);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &one);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &zero);
  } else if (dir == Dir::Data) {
    op = make(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tx);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tw);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &ty);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cd);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA, CUDNN_TYPE_FLOAT, 1, &one);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA, CUDNN_TYPE_FLOAT, 1, &zero);
  } else {
    op = make(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tx);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tw);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &ty);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cd);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA, CUDNN_TYPE_FLOAT, 1, &one);
    set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA, CUDNN_TYPE_FLOAT, 1, &zero);
  }
  CK(cudnnBackendFinalize(op));
  Desc graph = make(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
  set(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &h);
  set(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op);
  CK(cudnnBackendFinalize(graph));
  // cudnn-frontend destroys these as soon as the graph is built; the graph
  // holds its own copies.
  for (Desc d : {tx, tw, ty, cd, op}) cudnnBackendDestroyDescriptor(d);

  Desc heur = make(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
  const cudnnBackendHeurMode_t hm = CUDNN_HEUR_MODE_INSTANT;
  set(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph);
  set(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &hm);
  CK(cudnnBackendFinalize(heur));
  int64_t n_cfg = 0;
  Desc cfg = make(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
  CK(cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &n_cfg, &cfg));
  Desc plan = make(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
  set(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &h);
  set(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cfg);
  CK(cudnnBackendFinalize(plan));
  int64_t ws = -1, one_i = 1;
  CK(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &one_i, &ws));
  void* work = nullptr;
  if (ws > 0) cudaMalloc(&work, (size_t)ws);
  Desc pack = make(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
  void* ptrs[3] = {dx, dw, dy};
  const int64_t uids[3] = {1, 2, 3};
  set(pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, ptrs);
  set(pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids);
  set(pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &work);
  CK(cudnnBackendFinalize(pack));
  CK(cudnnBackendExecute(h, plan, pack));
  cudaDeviceSynchronize();

  const size_t n_out = dir == Dir::Fwd ? count(yd) : dir == Dir::Data ? count(cc.x) : count(cc.w);
  std::vector<double> want(n_out, 0.0);
  reference(cc, yd, dir, hx, hw, hy, &want, xs, ys);
  const auto got = from_dev(dir == Dir::Fwd ? dy : dir == Dir::Data ? dx : dw, n_out);
  double d = 0;
  for (size_t i = 0; i < n_out; ++i) d = std::fmax(d, std::fabs(got[i] - want[i]));
  char what[160];
  std::snprintf(what, sizeof what, "graph API convolution %s, %s", dir == Dir::Fwd ? "forward" : dir == Dir::Data ? "backward-data" : "backward-filter", cc.name);
  expect(what, n_cfg == 1 && d < 1e-3, d);
  for (Desc x : {graph, heur, cfg, plan, pack}) cudnnBackendDestroyDescriptor(x);
  cudaFree(dx), cudaFree(dw), cudaFree(dy), cudaFree(work);
}

static void graph_api(cudnnHandle_t h) {
  const ConvCase cases[] = {
      {"groups 2, 3x3, padded", {2, 4, 7, 7}, {6, 2, 3, 3}, 1, 1, 1, false},
      {"stride 2, dilation 2", {1, 3, 9, 8}, {4, 3, 3, 2}, 2, 2, 2, false},
      {"channels-last strides", {2, 3, 5, 6}, {4, 3, 3, 3}, 1, 1, 1, true},
      {"3-D", {1, 2, 4, 5, 5}, {3, 2, 2, 3, 3}, 1, 1, 1, false},
  };
  for (const auto& cc : cases)
    for (Dir d : {Dir::Fwd, Dir::Data, Dir::Filter}) conv(h, cc, d);
  // A graph whose operation this library has no engine for is refused when
  // it is finalized, not accepted and run wrongly.
  Desc pw = make(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
  expect("an operation with no engine here is refused, not run", cudnnBackendFinalize(pw) != CUDNN_STATUS_SUCCESS);
  cudnnBackendDestroyDescriptor(pw);
}

/* ---- BatchNorm training ---------------------------------------------------- */

static void batchnorm(cudnnHandle_t h) {
  const int N = 3, C = 4, H = 5, W = 2, M = N * H * W;
  const int dims[4] = {N, C, H, W}, strides[4] = {C * H * W, H * W, W, 1};
  const int pd[4] = {1, C, 1, 1}, ps[4] = {C, 1, 1, 1};
  cudnnTensorDescriptor_t xd, bd;
  CK(cudnnCreateTensorDescriptor(&xd));
  CK(cudnnCreateTensorDescriptor(&bd));
  CK(cudnnSetTensorNdDescriptor(xd, CUDNN_DATA_FLOAT, 4, dims, strides));
  CK(cudnnSetTensorNdDescriptor(bd, CUDNN_DATA_FLOAT, 4, pd, ps));
  int nb = 0, gd[4], gs[4];
  cudnnDataType_t t;
  CK(cudnnGetTensorNdDescriptor(xd, 4, &t, &nb, gd, gs));
  expect("an N-D tensor descriptor reads back as set", nb == 4 && gd[2] == H && gs[1] == H * W && t == CUDNN_DATA_FLOAT);
  const auto hx = filled((size_t)N * C * H * W, 1.0f), hdy = filled((size_t)N * C * H * W, 0.7f);
  const auto hs = filled(C, 0.5f), hb = filled(C, 0.2f);
  float *x = to_dev(hx), *y = to_dev(std::vector<float>(hx.size())), *dy = to_dev(hdy), *dx = to_dev(std::vector<float>(hx.size()));
  float *s = to_dev(hs), *b = to_dev(hb), *mean = to_dev(std::vector<float>(C)), *inv = to_dev(std::vector<float>(C));
  float *rm = to_dev(std::vector<float>(C, 0.0f)), *rv = to_dev(std::vector<float>(C, 1.0f));
  float *ds = to_dev(std::vector<float>(C)), *db = to_dev(std::vector<float>(C));
  const float one = 1.0f, zero = 0.0f;
  const double eps = 1e-5;
  CK(cudnnBatchNormalizationForwardTrainingEx(h, CUDNN_BATCHNORM_SPATIAL, CUDNN_BATCHNORM_OPS_BN, &one, &zero, xd, x,
                                              nullptr, nullptr, xd, y, bd, s, b, 0.1, rm, rv, eps, mean, inv, nullptr,
                                              nullptr, 0, nullptr, 0));
  CK(cudnnBatchNormalizationBackwardEx(h, CUDNN_BATCHNORM_SPATIAL, CUDNN_BATCHNORM_OPS_BN, &one, &zero, &one, &zero,
                                       xd, x, nullptr, nullptr, xd, dy, nullptr, nullptr, xd, dx, bd, s, b, ds, db,
                                       eps, mean, inv, nullptr, nullptr, 0, nullptr, 0));
  cudaDeviceSynchronize();
  // The reference: per channel, over N, H and W.
  std::vector<float> wy(hx.size()), wdx(hx.size()), wds(C), wdb(C);
  for (int c = 0; c < C; ++c) {
    double m = 0, v = 0;
    auto idx = [&](int n, int k) { return ((size_t)n * C + c) * H * W + k; };
    for (int n = 0; n < N; ++n) for (int k = 0; k < H * W; ++k) m += hx[idx(n, k)];
    m /= M;
    for (int n = 0; n < N; ++n) for (int k = 0; k < H * W; ++k) v += (hx[idx(n, k)] - m) * (hx[idx(n, k)] - m);
    v /= M;
    const double iv = 1 / std::sqrt(v + eps);
    double sdy = 0, sdyx = 0;
    for (int n = 0; n < N; ++n) for (int k = 0; k < H * W; ++k) {
      const double xh = (hx[idx(n, k)] - m) * iv;
      wy[idx(n, k)] = (float)(hs[c] * xh + hb[c]);
      sdy += hdy[idx(n, k)], sdyx += hdy[idx(n, k)] * xh;
    }
    wdb[c] = (float)sdy, wds[c] = (float)sdyx;
    for (int n = 0; n < N; ++n) for (int k = 0; k < H * W; ++k) {
      const double xh = (hx[idx(n, k)] - m) * iv;
      wdx[idx(n, k)] = (float)(hs[c] * iv / M * (M * hdy[idx(n, k)] - sdy - xh * sdyx));
    }
  }
  expect("BatchNorm training forward", max_diff(from_dev(y, hx.size()), wy) < 1e-4, max_diff(from_dev(y, hx.size()), wy));
  expect("BatchNorm backward: data", max_diff(from_dev(dx, hx.size()), wdx) < 1e-4, max_diff(from_dev(dx, hx.size()), wdx));
  expect("BatchNorm backward: scale and bias",
         max_diff(from_dev(ds, C), wds) < 1e-3 && max_diff(from_dev(db, C), wdb) < 1e-3,
         std::fmax(max_diff(from_dev(ds, C), wds), max_diff(from_dev(db, C), wdb)));
  for (float* p : {x, y, dy, dx, s, b, mean, inv, rm, rv, ds, db}) cudaFree(p);
  cudnnDestroyTensorDescriptor(xd), cudnnDestroyTensorDescriptor(bd);
}

/* ---- RNNs ------------------------------------------------------------------- */

struct Rnn {
  cudnnRNNMode_t mode;
  int layers, dirs, in = 3, hid = 4, T = 5, B = 3;
  std::vector<int> lens;
  int G() const { return mode == CUDNN_LSTM ? 4 : mode == CUDNN_GRU ? 3 : 1; }
};

static float sig(float x) { return 1 / (1 + std::exp(-x)); }

// A reference forward pass over the weights as cudnnGetRNNWeightParams lays
// them out: W[pl][gate] (hid x in_l), R[pl][gate] (hid x hid), bW, bR.
struct Weights { std::vector<std::vector<std::vector<float>>> W, R, bW, bR; };

static std::vector<float> reference_rnn(const Rnn& r, const Weights& w, const std::vector<float>& x /* [T][B][in] */,
                                        const std::vector<float>& hx, const std::vector<float>& cx) {
  const int H = r.hid, D = r.dirs, G = r.G();
  std::vector<float> in = x;
  int I = r.in;
  for (int l = 0; l < r.layers; ++l) {
    std::vector<float> out((size_t)r.T * r.B * H * D, 0.0f);
    for (int dir = 0; dir < D; ++dir) {
      const int pl = l * D + dir;
      for (int b = 0; b < r.B; ++b) {
        std::vector<float> h(hx.begin() + ((size_t)pl * r.B + b) * H, hx.begin() + ((size_t)pl * r.B + b + 1) * H);
        std::vector<float> c(cx.begin() + ((size_t)pl * r.B + b) * H, cx.begin() + ((size_t)pl * r.B + b + 1) * H);
        for (int s = 0; s < r.lens[b]; ++s) {
          const int t = dir == 0 ? s : r.lens[b] - 1 - s;
          const float* xt = &in[((size_t)t * r.B + b) * I];
          std::vector<float> zi(G * H), zr(G * H);
          for (int g = 0; g < G; ++g)
            for (int j = 0; j < H; ++j) {
              double a = w.bW[pl][g][j], q = w.bR[pl][g][j];
              for (int k = 0; k < I; ++k) a += w.W[pl][g][(size_t)j * I + k] * xt[k];
              for (int k = 0; k < H; ++k) q += w.R[pl][g][(size_t)j * H + k] * h[k];
              zi[g * H + j] = (float)a, zr[g * H + j] = (float)q;
            }
          std::vector<float> nh(H);
          for (int j = 0; j < H; ++j) {
            if (r.mode == CUDNN_LSTM) {
              const float ig = sig(zi[j] + zr[j]), fg = sig(zi[H + j] + zr[H + j]);
              const float gg = std::tanh(zi[2 * H + j] + zr[2 * H + j]), og = sig(zi[3 * H + j] + zr[3 * H + j]);
              c[j] = fg * c[j] + ig * gg;
              nh[j] = og * std::tanh(c[j]);
            } else {  // GRU
              const float rg = sig(zi[j] + zr[j]), z = sig(zi[H + j] + zr[H + j]);
              const float n = std::tanh(zi[2 * H + j] + rg * zr[2 * H + j]);
              nh[j] = (1 - z) * n + z * h[j];
            }
          }
          h = nh;
          for (int j = 0; j < H; ++j) out[((size_t)t * r.B + b) * H * D + (size_t)dir * H + j] = h[j];
        }
      }
    }
    in = out;
    I = H * D;
  }
  return in;
}

static void rnn(cudnnHandle_t h, Rnn r, const char* name) {
  const int H = r.hid, D = r.dirs, G = r.G(), states = r.layers * D * r.B * H;
  cudnnDropoutDescriptor_t drop;
  cudnnRNNDescriptor_t rd;
  cudnnRNNDataDescriptor_t xd, yd;
  cudnnTensorDescriptor_t hd, wmd, wbd;
  CK(cudnnCreateDropoutDescriptor(&drop));
  size_t ss = 0;
  CK(cudnnDropoutGetStatesSize(h, &ss));
  void* st = nullptr;
  cudaMalloc(&st, ss);
  CK(cudnnSetDropoutDescriptor(drop, h, 0.0f, st, ss, 0));
  CK(cudnnCreateRNNDescriptor(&rd));
  CK(cudnnSetRNNDescriptor_v8(rd, CUDNN_RNN_ALGO_STANDARD, r.mode, CUDNN_RNN_DOUBLE_BIAS,
                              D == 2 ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, CUDNN_LINEAR_INPUT,
                              CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH, r.in, H, H, r.layers, drop, 0));
  CK(cudnnCreateRNNDataDescriptor(&xd));
  CK(cudnnCreateRNNDataDescriptor(&yd));
  // Packed, time-major: the lengths sorted longest first, as packing requires.
  CK(cudnnSetRNNDataDescriptor(xd, CUDNN_DATA_FLOAT, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED, r.T, r.B, r.in, r.lens.data(), nullptr));
  CK(cudnnSetRNNDataDescriptor(yd, CUDNN_DATA_FLOAT, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED, r.T, r.B, H * D, r.lens.data(), nullptr));
  CK(cudnnCreateTensorDescriptor(&hd));
  const int hdims[3] = {r.layers * D, r.B, H}, hstr[3] = {r.B * H, H, 1};
  CK(cudnnSetTensorNdDescriptor(hd, CUDNN_DATA_FLOAT, 3, hdims, hstr));
  CK(cudnnCreateTensorDescriptor(&wmd));
  CK(cudnnCreateTensorDescriptor(&wbd));
  size_t wbytes = 0, work = 0, reserve = 0;
  CK(cudnnGetRNNWeightSpaceSize(h, rd, &wbytes));
  CK(cudnnGetRNNTempSpaceSizes(h, rd, CUDNN_FWD_MODE_TRAINING, xd, &work, &reserve));
  const size_t nw = wbytes / sizeof(float);
  std::vector<float> hw = filled(nw, 0.4f);
  float* wsp = to_dev(hw);
  // The reference reads the weights where cudnnGetRNNWeightParams says they are.
  Weights w;
  w.W.resize(r.layers * D), w.R.resize(r.layers * D), w.bW.resize(r.layers * D), w.bR.resize(r.layers * D);
  for (int pl = 0; pl < r.layers * D; ++pl)
    for (int lin = 0; lin < 2 * G; ++lin) {
      void *m = nullptr, *b = nullptr;
      CK(cudnnGetRNNWeightParams(h, rd, pl, wbytes, wsp, lin, wmd, &m, wbd, &b));
      int nbd = 0, md[3], ms[3];
      cudnnDataType_t t;
      CK(cudnnGetTensorNdDescriptor(wmd, 3, &t, &nbd, md, ms));
      const size_t mo = (float*)m - wsp, bo = (float*)b - wsp;
      std::vector<float> mat(hw.begin() + mo, hw.begin() + mo + (size_t)md[1] * md[2]);
      std::vector<float> bias(hw.begin() + bo, hw.begin() + bo + H);
      (lin < G ? w.W : w.R)[pl].push_back(mat);
      (lin < G ? w.bW : w.bR)[pl].push_back(bias);
    }
  // Inputs in the packed layout, and densely for the reference.
  std::vector<float> dense((size_t)r.T * r.B * r.in, 0.0f), packed_x;
  for (int t = 0; t < r.T; ++t)
    for (int b = 0; b < r.B; ++b)
      if (t < r.lens[b])
        for (int k = 0; k < r.in; ++k) {
          const float v = fill((t * r.B + b) * r.in + k, 1.0f);
          dense[((size_t)t * r.B + b) * r.in + k] = v;
          packed_x.push_back(v);
        }
  size_t ny = 0;
  for (int l : r.lens) ny += (size_t)l * H * D;
  const auto hhx = filled(states, 0.3f), hcx = filled(states, 0.2f);
  float *x = to_dev(packed_x), *y = to_dev(std::vector<float>(ny)), *hx = to_dev(hhx), *cx = to_dev(hcx);
  float *hy = to_dev(std::vector<float>(states)), *cy = to_dev(std::vector<float>(states));
  void *wk = nullptr, *rs = nullptr;
  cudaMalloc(&wk, work + 16), cudaMalloc(&rs, reserve + 16);
  const bool lstm = r.mode == CUDNN_LSTM;
  CK(cudnnRNNForward(h, rd, CUDNN_FWD_MODE_TRAINING, nullptr, xd, x, yd, y, hd, hx, hy, hd, lstm ? cx : nullptr,
                     lstm ? cy : nullptr, wbytes, wsp, work, wk, reserve, rs));
  cudaDeviceSynchronize();
  // Compare y with the reference, unpacked.
  const auto want = reference_rnn(r, w, dense, hhx, lstm ? hcx : std::vector<float>(states, 0.0f));
  const auto got_packed = from_dev(y, ny);
  std::vector<float> got_dense(want.size(), 0.0f);
  size_t p = 0;
  for (int t = 0; t < r.T; ++t)
    for (int b = 0; b < r.B; ++b)
      if (t < r.lens[b])
        for (int k = 0; k < H * D; ++k) got_dense[((size_t)t * r.B + b) * H * D + k] = got_packed[p++];
  char what[160];
  std::snprintf(what, sizeof what, "%s forward matches the reference", name);
  expect(what, max_diff(got_dense, want) < 1e-5, max_diff(got_dense, want));

  // Gradients of L = sum(y * c), c fixed, against finite differences of the
  // library's own forward pass.
  std::vector<float> coef(ny);
  for (size_t i = 0; i < ny; ++i) coef[i] = std::cos(0.37f * (float)i);
  float *dy = to_dev(coef), *dx = to_dev(std::vector<float>(packed_x.size())), *dhx = to_dev(std::vector<float>(states));
  float *dcx = to_dev(std::vector<float>(states)), *dw = to_dev(std::vector<float>(nw, 0.0f));
  CK(cudnnRNNBackwardData_v8(h, rd, nullptr, yd, y, dy, xd, dx, hd, hx, nullptr, dhx, hd, lstm ? cx : nullptr, nullptr,
                             lstm ? dcx : nullptr, wbytes, wsp, work, wk, reserve, rs));
  CK(cudnnRNNBackwardWeights_v8(h, rd, CUDNN_WGRAD_MODE_ADD, nullptr, xd, x, hd, hx, yd, y, wbytes, dw, work, wk,
                                reserve, rs));
  cudaDeviceSynchronize();
  auto loss = [&](const std::vector<float>& xs, const std::vector<float>& ws) {
    float *x2 = to_dev(xs), *w2 = to_dev(ws), *y2 = to_dev(std::vector<float>(ny));
    cudnnRNNForward(h, rd, CUDNN_FWD_MODE_INFERENCE, nullptr, xd, x2, yd, y2, hd, hx, nullptr, hd,
                    lstm ? cx : nullptr, nullptr, wbytes, w2, work, wk, 0, nullptr);
    const auto yy = from_dev(y2, ny);
    double s = 0;
    for (size_t i = 0; i < ny; ++i) s += (double)yy[i] * coef[i];
    cudaFree(x2), cudaFree(w2), cudaFree(y2);
    return s;
  };
  const auto gx = from_dev(dx, packed_x.size()), gw = from_dev(dw, nw);
  const float e = 1e-2f;
  double worst = 0;
  for (size_t i : {size_t(0), packed_x.size() / 2, packed_x.size() - 1}) {
    auto a = packed_x, b = packed_x;
    a[i] += e, b[i] -= e;
    worst = std::fmax(worst, std::fabs((loss(a, hw) - loss(b, hw)) / (2 * e) - gx[i]));
  }
  std::snprintf(what, sizeof what, "%s backward-data matches finite differences", name);
  expect(what, worst < 5e-3, worst);
  worst = 0;
  for (size_t i : {size_t(1), nw / 3, nw / 2, nw - 2}) {
    auto a = hw, b = hw;
    a[i] += e, b[i] -= e;
    worst = std::fmax(worst, std::fabs((loss(packed_x, a) - loss(packed_x, b)) / (2 * e) - gw[i]));
  }
  std::snprintf(what, sizeof what, "%s backward-weights matches finite differences", name);
  expect(what, worst < 5e-3, worst);
  for (float* q : {x, y, hx, cx, hy, cy, dy, dx, dhx, dcx, dw, wsp}) cudaFree(q);
  cudaFree(wk), cudaFree(rs), cudaFree(st);
  cudnnDestroyRNNDataDescriptor(xd), cudnnDestroyRNNDataDescriptor(yd), cudnnDestroyRNNDescriptor(rd);
  cudnnDestroyTensorDescriptor(hd), cudnnDestroyTensorDescriptor(wmd), cudnnDestroyTensorDescriptor(wbd);
  cudnnDestroyDropoutDescriptor(drop);
}

/* ---- cuBLASLt with a cuBLAS handle ------------------------------------------ */

static void lt_with_cublas_handle() {
  const int M = 5, N = 4, K = 3;  // D (M x N, column-major) = A (M x K) * B (K x N) + bias[M]
  const auto ha = filled(M * K, 1.0f), hb = filled(K * N, 0.5f), hbias = filled(M, 0.3f);
  float *a = to_dev(ha), *b = to_dev(hb), *bias = to_dev(hbias), *d = to_dev(std::vector<float>(M * N));
  cublasHandle_t blas;
  cublasCreate(&blas);
  // What PyTorch does: its cuBLAS handle, cast, is its cuBLASLt handle.
  const auto lt = reinterpret_cast<cublasLtHandle_t>(blas);
  cublasLtMatmulDesc_t md;
  cublasLtMatrixLayout_t la, lb, ld;
  cublasLtMatmulDescCreate(&md, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  const cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
  cublasLtMatmulDescSetAttribute(md, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof ep);
  cublasLtMatmulDescSetAttribute(md, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof bias);
  cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, M, K, M);
  cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, K, N, K);
  cublasLtMatrixLayoutCreate(&ld, CUDA_R_32F, M, N, M);
  const float one = 1.0f, zero = 0.0f;
  const cublasStatus_t s = cublasLtMatmul(lt, md, &one, a, la, b, lb, &zero, d, ld, d, ld, nullptr, nullptr, 0, 0);
  cudaDeviceSynchronize();
  std::vector<float> want(M * N);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double acc = hbias[i];
      for (int k = 0; k < K; ++k) acc += (double)ha[k * M + i] * hb[j * K + k];
      want[j * M + i] = (float)acc;
    }
  expect("cuBLASLt matmul + bias, called with a cuBLAS handle", s == CUBLAS_STATUS_SUCCESS && max_diff(from_dev(d, M * N), want) < 1e-4);
  cublasLtMatrixLayoutDestroy(la), cublasLtMatrixLayoutDestroy(lb), cublasLtMatrixLayoutDestroy(ld);
  cublasLtMatmulDescDestroy(md);
  cublasDestroy(blas);
  for (float* q : {a, b, bias, d}) cudaFree(q);
}

int main() {
  int major = 0, minor = 0, patch = 0;
  cudnnGetProperty(MAJOR_VERSION, &major);
  cudnnGetProperty(MINOR_VERSION, &minor);
  cudnnGetProperty(PATCH_LEVEL, &patch);
  expect("cudnnGetProperty and cudnnGetVersion agree on the version",
         (size_t)(major * 10000 + minor * 100 + patch) == cudnnGetVersion() && major == 9);
  cudnnHandle_t h;
  if (cudnnCreate(&h) != CUDNN_STATUS_SUCCESS) { std::printf("FAIL cudnnCreate\n"); return 1; }
  graph_api(h);
  batchnorm(h);
  rnn(h, Rnn{CUDNN_LSTM, 2, 2, 3, 4, 5, 3, {5, 3, 2}}, "LSTM, 2 layers, both directions, packed");
  rnn(h, Rnn{CUDNN_GRU, 1, 1, 3, 4, 5, 3, {5, 4, 1}}, "GRU, packed");
  cudnnDestroy(h);
  lt_with_cublas_handle();
  std::printf(fails ? "FAIL\n" : "PASS\n");
  return fails != 0;
}
