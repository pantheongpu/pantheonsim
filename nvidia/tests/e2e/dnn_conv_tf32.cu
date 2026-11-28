// PyTorch's default FP32 convolution on a card with TF32 tensor cores (issue "Cross-architecture (cuDNN conv): PyTorch's default TF32 convolution on Ampere
// and newer is not modelled"): torch.backends.cudnn.allow_tf32 = True is the default, and then cuDNN's engine for an FP32 convolution reads its inputs as
// TF32 (numerical note CUDNN_NUMERICAL_NOTE_TENSOR_CORE: cuDNN 9 has no separate TF32 note; the A10G's other engines note WINOGRAD or FFT or nothing); with it False PyTorch drops the engines that note TENSOR_CORE. A Turing card has no TF32 and offers none.
//
// The program convolves 8x64x28x28 by 64x64x3x3 (pad 1) of random values, through cudnnConvolutionForward with the default math type and with
// CUDNN_FMA_MATH, and through the backend API: every engine configuration the heuristics return, with its numerical notes read, is planned and run.
// Each result is classed by its error against the double-precision convolution of the same inputs: TF32 operands give about 1e-4 to 1e-3, an FP32
// computation 1e-6 and below. The output is a summary that does not depend on how many engines a cuDNN version lists:
//   legacy default math: <class>      legacy FMA math: <class>
//   engines: some note TENSOR_CORE: yes|no; every engine that notes it gives class tf32: yes|no|-; some engine without the note exists: yes|no;
//   the first engine notes it: yes|no
// (an engine without the note is not held to a class: a Winograd or FFT engine of a Turing card is less accurate than 2e-6 and no TF32.)
// and is pinned for a card (nvidia/tests/data/dnn_conv_tf32.<card>.txt, run by run_dnn_conv_tf32.sh). What it does not hold: the bits. Which sum order
// cuDNN's TF32 engines use is not modelled (the simulator sums in double and rounds once), so the error class is the check.
#include <cudnn.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Desc = cudnnBackendDescriptor_t;
static cudnnHandle_t H;
static Desc make(cudnnBackendDescriptorType_t t) { Desc d = nullptr; cudnnBackendCreateDescriptor(t, &d); return d; }
static void set(Desc d, cudnnBackendAttributeName_t n, cudnnBackendAttributeType_t t, int64_t c, const void* v) { cudnnBackendSetAttribute(d, n, t, c, v); }
static void set_desc(Desc d, cudnnBackendAttributeName_t n, Desc v) { set(d, n, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &v); }

static Desc tensor(int64_t uid, const std::vector<int64_t>& dims) {
  Desc t = make(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
  const cudnnDataType_t f = CUDNN_DATA_FLOAT;
  const int64_t align = 16;
  std::vector<int64_t> strides(dims.size(), 1);
  for (size_t i = dims.size() - 1; i-- > 0;) strides[i] = strides[i + 1] * dims[i + 1];
  const bool virt = false;
  set(t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &f);
  set(t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, (int64_t)dims.size(), dims.data());
  set(t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, (int64_t)strides.size(), strides.data());
  set(t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
  set(t, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align);
  set(t, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &virt);
  cudnnBackendFinalize(t);
  return t;
}

static uint64_t g_state = 0x9e3779b97f4a7c15ull;
static float next_value() {
  g_state ^= g_state << 13; g_state ^= g_state >> 7; g_state ^= g_state << 17;
  return static_cast<float>((static_cast<double>(g_state >> 11) / 9007199254740992.0 - 0.5) * 2.0);
}

constexpr int N = 8, C = 64, HH = 28, K = 64, R = 3;

static std::vector<double> reference(const std::vector<float>& x, const std::vector<float>& w) {
  std::vector<double> y((size_t)N * K * HH * HH, 0.0);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      for (int i = 0; i < HH; ++i)
        for (int j = 0; j < HH; ++j) {
          double s = 0;
          for (int c = 0; c < C; ++c)
            for (int r = 0; r < R; ++r)
              for (int q = 0; q < R; ++q) {
                const int ii = i + r - 1, jj = j + q - 1;
                if (ii < 0 || jj < 0 || ii >= HH || jj >= HH) continue;
                s += (double)x[(((size_t)n * C + c) * HH + ii) * HH + jj] * (double)w[(((size_t)k * C + c) * R + r) * R + q];
              }
          y[(((size_t)n * K + k) * HH + i) * HH + j] = s;
        }
  return y;
}
static const char* klass(const std::vector<float>& got, const std::vector<double>& ref) {
  double err = 0, mag = 0;
  for (size_t i = 0; i < ref.size(); ++i) { err = std::fmax(err, std::fabs(got[i] - ref[i])); mag = std::fmax(mag, std::fabs(ref[i])); }
  return err / mag < 2e-6 ? "fp32" : "tf32";
}

int main() {
  cudnnCreate(&H);
  g_state ^= 734;
  std::vector<float> x((size_t)N * C * HH * HH), w((size_t)K * C * R * R), y((size_t)N * K * HH * HH);
  for (auto& v : x) v = next_value();
  for (auto& v : w) v = next_value();
  const std::vector<double> ref = reference(x, w);
  float *dx, *dw, *dy;
  cudaMalloc(&dx, x.size() * 4); cudaMalloc(&dw, w.size() * 4); cudaMalloc(&dy, y.size() * 4);
  cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice);

  // The classic API.
  cudnnTensorDescriptor_t xd, yd;
  cudnnFilterDescriptor_t wd;
  cudnnConvolutionDescriptor_t cd;
  cudnnCreateTensorDescriptor(&xd); cudnnCreateTensorDescriptor(&yd); cudnnCreateFilterDescriptor(&wd); cudnnCreateConvolutionDescriptor(&cd);
  cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, HH, HH);
  cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, HH, HH);
  cudnnSetFilter4dDescriptor(wd, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, R);
  cudnnSetConvolution2dDescriptor(cd, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
  const char* legacy[2] = {"?", "?"};
  for (int pass = 0; pass < 2; ++pass) {
    cudnnSetConvolutionMathType(cd, pass == 0 ? CUDNN_DEFAULT_MATH : CUDNN_FMA_MATH);
    int n_res = 0;
    cudnnConvolutionFwdAlgoPerf_t perf[16];
    cudnnGetConvolutionForwardAlgorithm_v7(H, xd, wd, cd, yd, 16, &n_res, perf);
    size_t ws = 0;
    cudnnGetConvolutionForwardWorkspaceSize(H, xd, wd, cd, yd, perf[0].algo, &ws);
    void* work = nullptr;
    cudaMalloc(&work, ws + 1);
    const float al = 1.f, be = 0.f;
    cudaMemset(dy, 0, y.size() * 4);
    if (cudnnConvolutionForward(H, &al, xd, dx, wd, dw, cd, perf[0].algo, work, ws, &be, yd, dy) == CUDNN_STATUS_SUCCESS) {
      cudaDeviceSynchronize();
      cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
      legacy[pass] = klass(y, ref);
    }
    cudaFree(work);
  }
  std::printf("legacy default math: %s\nlegacy FMA math: %s\n", legacy[0], legacy[1]);

  // The backend API: the heuristics' configurations, their notes, their results.
  Desc tx = tensor(1, {N, C, HH, HH}), tw = tensor(2, {K, C, R, R}), ty = tensor(3, {N, K, HH, HH});
  Desc cv = make(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
  const cudnnDataType_t f = CUDNN_DATA_FLOAT;
  const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
  const int64_t spatial = 2, pads[2] = {1, 1}, ones[2] = {1, 1};
  set(cv, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &f);
  set(cv, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode);
  set(cv, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial);
  set(cv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, pads);
  set(cv, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, pads);
  set(cv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, ones);
  set(cv, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, ones);
  cudnnBackendFinalize(cv);
  Desc op = make(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
  const float one = 1.f, zero = 0.f;
  set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, tx);
  set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, tw);
  set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, ty);
  set_desc(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, cv);
  set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &one);
  set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &zero);
  cudnnBackendFinalize(op);
  Desc graph = make(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
  set(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &H);
  set(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op);
  cudnnBackendFinalize(graph);
  Desc heur = make(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
  const cudnnBackendHeurMode_t hm = CUDNN_HEUR_MODE_A;
  set_desc(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, graph);
  set(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &hm);
  cudnnBackendFinalize(heur);
  int64_t n = 0;
  cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &n, nullptr);
  std::vector<Desc> cfgs((size_t)n);
  for (auto& c : cfgs) c = make(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
  if (n) cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, n, &n, cfgs.data());
  bool any_tf32 = false, tf32_all_tf32 = true, any_plain = false, first_notes = false;
  for (int64_t i = 0; i < n; ++i) {
    Desc eng = make(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    int64_t got = 0;
    cudnnBackendGetAttribute(cfgs[i], CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &got, &eng);
    int64_t nn = 0;
    cudnnBackendNumericalNote_t notes[32];
    cudnnBackendGetAttribute(eng, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE, CUDNN_TYPE_NUMERICAL_NOTE, 32, &nn, notes);
    bool tf = false;
    for (int64_t q = 0; q < nn; ++q) tf |= notes[q] == CUDNN_NUMERICAL_NOTE_TENSOR_CORE;
    if (std::getenv("DNN_CONV_VERBOSE")) { std::fprintf(stderr, "config %lld notes:", (long long)i); for (int64_t q = 0; q < nn; ++q) std::fprintf(stderr, " %d", (int)notes[q]); std::fprintf(stderr, "\n"); }
    cudnnBackendDestroyDescriptor(eng);
    Desc plan = make(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    set(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &H);
    set_desc(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, cfgs[i]);
    if (cudnnBackendFinalize(plan) != CUDNN_STATUS_SUCCESS) { cudnnBackendDestroyDescriptor(plan); continue; }
    int64_t ws = 0, o = 1;
    cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &o, &ws);
    void* work = nullptr;
    if (ws > 0) cudaMalloc(&work, (size_t)ws);
    const int64_t uids[3] = {1, 2, 3};
    void* ptrs[3] = {dx, dw, dy};
    Desc pack = make(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    set(pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, ptrs);
    set(pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids);
    set(pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &work);
    cudnnBackendFinalize(pack);
    cudaMemset(dy, 0, y.size() * 4);
    if (cudnnBackendExecute(H, plan, pack) == CUDNN_STATUS_SUCCESS) {
      cudaDeviceSynchronize();
      cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
      const bool is_tf32 = !std::strcmp(klass(y, ref), "tf32");
      if (i == 0) first_notes = tf;
      if (tf) { any_tf32 = true; tf32_all_tf32 &= is_tf32; }
      else any_plain = true;
      if (std::getenv("DNN_CONV_VERBOSE")) std::fprintf(stderr, "config %lld note-tf32 %d class %s\n", (long long)i, (int)tf, klass(y, ref));
    }
    cudnnBackendDestroyDescriptor(pack);
    cudnnBackendDestroyDescriptor(plan);
    cudaFree(work);
  }
  auto yn = [](bool b) { return b ? "yes" : "no"; };
  std::printf("engines: some note TENSOR_CORE: %s; every engine that notes it gives class tf32: %s; some engine without the note exists: %s;"
              " the first engine notes it: %s\n",
              yn(any_tf32), any_tf32 ? yn(tf32_all_tf32) : "-", yn(any_plain), yn(first_notes));
  for (Desc c : cfgs) cudnnBackendDestroyDescriptor(c);
  for (Desc d : {heur, graph, op, cv, tx, tw, ty}) cudnnBackendDestroyDescriptor(d);
  cudnnDestroyTensorDescriptor(xd); cudnnDestroyTensorDescriptor(yd); cudnnDestroyFilterDescriptor(wd); cudnnDestroyConvolutionDescriptor(cd);
  cudnnDestroy(H);
  cudaFree(dx); cudaFree(dw); cudaFree(dy);
  return 0;
}
