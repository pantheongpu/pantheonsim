// An INT8x32 convolution through cuDNN's backend (graph) API with the filter in the layout
// cudnnReorderFilterAndBias makes for it (CUDNN_ATTR_TENSOR_REORDERING_MODE =
// CUDNN_TENSOR_REORDERING_INT8x32): tensors whose channel dimension holds vectors of 32, the
// filter reordered for the tensor cores, accumulated in INT32 and saturated to INT8.
//
// The program runs against NVIDIA's libcudnn.so.9 on a GPU with INT8 tensor cores (sm_75 and
// later) and against VirtualGPU; both must print the same and agree with a host reference.
// Where NVIDIA's heuristics offer no engine the check says so and is skipped.
#include <cudnn.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int fails = 0;
static void expect(const char* what, bool ok) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
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
static const bool kOnSim = std::getenv("VGPU_GPU") != nullptr;

static std::vector<int64_t> packed(const std::vector<int64_t>& dims) {
  std::vector<int64_t> s(dims.size(), 1);
  for (size_t i = dims.size() - 1; i-- > 0;) s[i] = s[i + 1] * dims[i + 1];
  return s;
}

// An INT8 tensor whose dimension 1 holds vectors of 32; dims and strides count vectors.
static Desc vtensor(int64_t uid, const std::vector<int64_t>& dims, bool reordered, cudnnStatus_t* finalize = nullptr) {
  Desc d = make(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
  const cudnnDataType_t t = CUDNN_DATA_INT8;
  const int64_t align = 16, vc = 32, vd = 1;
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
  if (reordered) {
    const cudnnBackendTensorReordering_t r = CUDNN_TENSOR_REORDERING_INT8x32;
    set(d, CUDNN_ATTR_TENSOR_REORDERING_MODE, CUDNN_TYPE_TENSOR_REORDERING_MODE, 1, &r);
  }
  const cudnnStatus_t s = cudnnBackendFinalize(d);
  if (finalize) *finalize = s;
  return d;
}

// Builds the graph and runs the first engine configuration that makes a plan: 1 ran, 0 none, -1 error.
static int run(const std::vector<Desc>& ops, const std::vector<int64_t>& uids, const std::vector<void*>& ptrs) {
  Desc graph = make(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
  set(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &H);
  set(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, (int64_t)ops.size(), ops.data());
  cudnnStatus_t s = cudnnBackendFinalize(graph);
  if (s != CUDNN_STATUS_SUCCESS) {
    std::printf("     the graph did not finalize (%d)\n", (int)s);
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
        if (s != CUDNN_STATUS_SUCCESS) std::printf("     execute -> %d\n", (int)s);
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

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  if (cudnnCreate(&H) != CUDNN_STATUS_SUCCESS) {
    std::printf("FAIL cudnnCreate\n");
    return 1;
  }
  // N = 1, C = K = 32 (one vector), 6 x 6, 3 x 3 filter, padding 1.
  const int C = 32, K = 32, Hh = 6, W = 6, R = 3, pad = 1;
  const int L = C / 32 * R * R * 32;   // bytes of one filter's row in the vectorized [K][C/32][R][S][32] layout
  auto xat = [&](int c, int h, int w) { return ((c / 32) * Hh * W + h * W + w) * 32 + c % 32; };
  auto wat = [&](int k, int c, int r, int q) { return k * L + ((c / 32) * R * R + r * R + q) * 32 + c % 32; };
  auto yat = [&](int k, int h, int w) { return ((k / 32) * Hh * W + h * W + w) * 32 + k % 32; };
  std::vector<int8_t> hx(C * Hh * W), hw((size_t)K * L);
  for (int c = 0; c < C; ++c)
    for (int h = 0; h < Hh; ++h)
      for (int w = 0; w < W; ++w) hx[xat(c, h, w)] = (int8_t)((c * 7 + h * 3 + w * 5) % 5 - 2);
  for (int k = 0; k < K; ++k)
    for (int c = 0; c < C; ++c)
      for (int r = 0; r < R; ++r)
        for (int q = 0; q < R; ++q) hw[wat(k, c, r, q)] = (int8_t)((k * 5 + c * 3 + r * 2 + q) % 3 - 1);
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

  // The filter in the order the tensor cores want it: cudnnReorderFilterAndBias, DEFAULT_REORDER.
  cudnnFilterDescriptor_t fd;
  cudnnCreateFilterDescriptor(&fd);
  cudnnSetFilter4dDescriptor(fd, CUDNN_DATA_INT8x32, CUDNN_TENSOR_NCHW_VECT_C, K, C, R, R);
  int8_t *dw_plain, *dw, *dx, *dy;
  cudaMalloc(&dw_plain, hw.size() + 64), cudaMalloc(&dw, hw.size() + 64);
  cudaMalloc(&dx, hx.size() + 64), cudaMalloc(&dy, want.size() + 64);
  cudaMemcpy(dw_plain, hw.data(), hw.size(), cudaMemcpyHostToDevice);
  cudaMemset(dw, 0, hw.size() + 64);
  const cudnnStatus_t rs = cudnnReorderFilterAndBias(H, fd, CUDNN_DEFAULT_REORDER, dw_plain, dw, 0, nullptr, nullptr);
  expect("cudnnReorderFilterAndBias of an INT8x32 filter", rs == CUDNN_STATUS_SUCCESS);
  cudaMemcpy(dx, hx.data(), hx.size(), cudaMemcpyHostToDevice);
  cudaMemset(dy, 0x55, want.size());

  cudnnStatus_t fin = CUDNN_STATUS_SUCCESS;
  Desc x = vtensor(1, {1, C / 32, Hh, W}, false), w = vtensor(2, {K, C / 32, R, R}, true, &fin),
       y = vtensor(3, {1, K / 32, Hh, W}, false);
  std::printf("     a reordered filter tensor finalizes with %d\n", (int)fin);
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
  set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, w);
  set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, y);
  set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, cd);
  set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha);
  set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta);
  const cudnnStatus_t st = cudnnBackendFinalize(op);
  std::printf("     the convolution op finalizes with %d\n", (int)st);
  int ran = st == CUDNN_STATUS_SUCCESS ? run({op}, {1, 2, 3}, {dx, dw, dy}) : 0;
  if (ran == 0 && !kOnSim) {
    std::printf("skip the INT8x32 graph convolution (no engine offered for it)\n");
  } else {
    bool same = ran == 1;
    if (same) {
      std::vector<int8_t> got(want.size());
      cudaMemcpy(got.data(), dy, got.size(), cudaMemcpyDeviceToHost);
      for (size_t i = 0; i < got.size(); ++i) same = same && got[i] == want[i];
    }
    expect("the graph's INT8x32 convolution with a reordered filter matches the host", same);
  }
  // The plain filter, given as reordered, must not be taken for the right answer (the data differ).
  cudaMemset(dy, 0x55, want.size());
  ran = st == CUDNN_STATUS_SUCCESS ? run({op}, {1, 2, 3}, {dx, dw_plain, dy}) : 0;
  if (ran == 1) {
    std::vector<int8_t> got(want.size());
    cudaMemcpy(got.data(), dy, got.size(), cudaMemcpyDeviceToHost);
    bool same = true;
    for (size_t i = 0; i < got.size(); ++i) same = same && got[i] == want[i];
    expect("the plain filter, taken as reordered, gives another answer", !same);
  }
  for (Desc d : {x, w, y, cd, op}) cudnnBackendDestroyDescriptor(d);
  cudnnDestroyFilterDescriptor(fd);
  cudaFree(dw_plain), cudaFree(dw), cudaFree(dx), cudaFree(dy);
  cudnnDestroy(H);
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
