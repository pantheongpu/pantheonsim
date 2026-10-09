// cuDNN's per-tensor FP8 scaled dot-product attention (CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR
// with FP8 Q, K, V, their descale factors, the scale factors of S and O and the amax of S and O),
// forward, built from the backend API directly.
//
// DOCUMENTATION-DERIVED: no card was available for it (cuDNN's FP8 attention engines are Hopper's
// and Blackwell's; AWS had no capacity for either), so this checks VirtualGPU against the formulas
// cuDNN's header and cudnn-frontend's documentation give, computed here on the host:
//   S = (Q K^T) descale_Q descale_K scale, P = softmax(S), amax_S = max P,
//   P8 = E4M3(P scale_S), O = (P8 V) descale_S descale_V, amax_O = max |O|, O8 = E4M3(O scale_O).
// On a card the program prints SKIP. MXFP8 attention (E8M0 block scales) is refused.
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cudnn.h>

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

static Desc tensor(int64_t uid, const std::vector<int64_t>& dims, cudnnDataType_t t) {
  Desc d = make(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
  std::vector<int64_t> strides(dims.size(), 1);
  for (size_t i = dims.size() - 1; i-- > 0;) strides[i] = strides[i + 1] * dims[i + 1];
  const int64_t align = 16;
  const bool v = false;
  set(d, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &t);
  set(d, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, (int64_t)dims.size(), dims.data());
  set(d, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, (int64_t)strides.size(), strides.data());
  set(d, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
  set(d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align);
  set(d, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &v);
  cudnnBackendFinalize(d);
  return d;
}

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

static uint8_t e4m3(float f) { return (uint8_t)__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3); }
static float from_e4m3(uint8_t b) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
  return __half2float(__half(h));
}

template <class T> static T* up(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(T) + 64);
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  if (!std::getenv("VGPU_GPU")) {
    std::printf("SKIP: documentation-derived; there is no card to compare with (VirtualGPU only)\nPASS\n");
    return 0;
  }
  if (cudnnCreate(&H) != CUDNN_STATUS_SUCCESS) {
    std::printf("FAIL cudnnCreate\n");
    return 1;
  }
  const int B = 1, Hh = 2, Sq = 8, Skv = 8, D = 16;
  const float vals[8] = {1, -1, 0.5f, -0.5f, 2, -2, 0.25f, 3};
  auto gen = [&](size_t n, int salt) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = e4m3(vals[(i * 7 + salt * 3 + i / 5) % 8]);
    return v;
  };
  const size_t nq = (size_t)B * Hh * Sq * D, nk = (size_t)B * Hh * Skv * D;
  const auto hq = gen(nq, 1), hk = gen(nk, 2), hv = gen(nk, 3);
  const float dq = 0.5f, dk = 0.25f, dv = 2.0f, scale_s = 256.0f, ds = 1.0f / 256.0f, scale_o = 4.0f, attn = 0.25f;

  // The reference, on the host, by the formulas above.
  std::vector<float> want((size_t)B * Hh * Sq * D);
  double want_amax_s = 0, want_amax_o = 0;
  std::vector<float> Oacc(want.size());
  for (int h = 0; h < Hh; ++h)
    for (int i = 0; i < Sq; ++i) {
      std::vector<double> s(Skv), p(Skv);
      double mx = -1e300;
      for (int j = 0; j < Skv; ++j) {
        double acc = 0;
        for (int d = 0; d < D; ++d) acc += (double)from_e4m3(hq[(size_t)(h * Sq + i) * D + d]) * from_e4m3(hk[(size_t)(h * Skv + j) * D + d]);
        s[j] = (float)(acc * attn * dq * dk);
        mx = std::fmax(mx, s[j]);
      }
      double sum = 0;
      for (int j = 0; j < Skv; ++j) sum += (p[j] = std::exp(s[j] - mx));
      for (int j = 0; j < Skv; ++j) p[j] /= sum, want_amax_s = std::fmax(want_amax_s, p[j]);
      for (int e = 0; e < D; ++e) {
        double acc = 0;
        for (int j = 0; j < Skv; ++j) acc += (double)from_e4m3(e4m3((float)(p[j] * scale_s))) * from_e4m3(hv[(size_t)(h * Skv + j) * D + e]);
        const float o = (float)(acc * ds * dv);
        want_amax_o = std::fmax(want_amax_o, std::fabs(o));
        Oacc[(size_t)(h * Sq + i) * D + e] = o;
      }
    }
  for (size_t i = 0; i < want.size(); ++i) want[i] = from_e4m3(e4m3(Oacc[i] * scale_o));

  const std::vector<int64_t> qd = {B, Hh, Sq, D}, kd = {B, Hh, Skv, D}, one = {1, 1, 1, 1};
  Desc q = tensor(1, qd, CUDNN_DATA_FP8_E4M3), k = tensor(2, kd, CUDNN_DATA_FP8_E4M3), v = tensor(3, kd, CUDNN_DATA_FP8_E4M3),
       o = tensor(4, qd, CUDNN_DATA_FP8_E4M3), sc = tensor(5, one, CUDNN_DATA_FLOAT), dqt = tensor(6, one, CUDNN_DATA_FLOAT),
       dkt = tensor(7, one, CUDNN_DATA_FLOAT), dvt = tensor(8, one, CUDNN_DATA_FLOAT), dst = tensor(9, one, CUDNN_DATA_FLOAT),
       sst = tensor(10, one, CUDNN_DATA_FLOAT), sot = tensor(11, one, CUDNN_DATA_FLOAT),
       as = tensor(12, one, CUDNN_DATA_FLOAT), ao = tensor(13, one, CUDNN_DATA_FLOAT);
  auto build = [&](bool with_amax_o) {
    Desc op = make(CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR);
    set_desc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC, q);
    set_desc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC, k);
    set_desc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC, v);
    set_desc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC, o);
    set_desc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC, sc);
    set_desc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_QDESC, dqt);
    set_desc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_KDESC, dkt);
    set_desc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_VDESC, dvt);
    set_desc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_SDESC, dst);
    set_desc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_SCALE_SDESC, sst);
    set_desc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_SCALE_ODESC, sot);
    set_desc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_AMAX_SDESC, as);
    if (with_amax_o) set_desc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_AMAX_ODESC, ao);
    return op;
  };
  std::vector<float> f1 = {attn}, fq = {dq}, fk = {dk}, fv = {dv}, fs = {ds}, fss = {scale_s}, fso = {scale_o}, z1 = {-1}, z2 = {-1};
  uint8_t *dQ = up(hq), *dK = up(hk), *dV = up(hv), *dO = up(std::vector<uint8_t>(want.size(), 0x55));
  float *dsc = up(f1), *ddq = up(fq), *ddk = up(fk), *ddv = up(fv), *dds = up(fs), *dss = up(fss), *dso = up(fso), *das = up(z1), *dao = up(z2);
  Desc op = build(true);
  const cudnnStatus_t st = cudnnBackendFinalize(op);
  expect("an FP8 attention operation finalizes", st == CUDNN_STATUS_SUCCESS, (double)st);
  if (st == CUDNN_STATUS_SUCCESS) {
    const int ran = run({op}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, {dQ, dK, dV, dO, dsc, ddq, ddk, ddv, dds, dss, dso, das, dao});
    expect("it runs", ran == 1);
    if (ran == 1) {
      std::vector<uint8_t> got(want.size());
      cudaMemcpy(got.data(), dO, got.size(), cudaMemcpyDeviceToHost);
      int off = 0;
      double worst = 0;
      for (size_t i = 0; i < got.size(); ++i) {
        const double g = from_e4m3(got[i]), w = want[i];
        const double tol = 0.13 * std::fabs(w) + 0.02;   // one E4M3 step: a softmax tie may round the other way
        worst = std::fmax(worst, std::fabs(g - w) / tol);
        off += std::fabs(g - w) > tol;
      }
      expect("O, quantized to E4M3 with scale_O, is the documented attention within one FP8 step", off == 0, worst);
      float am_s = 0, am_o = 0;
      cudaMemcpy(&am_s, das, 4, cudaMemcpyDeviceToHost);
      cudaMemcpy(&am_o, dao, 4, cudaMemcpyDeviceToHost);
      expect("amax of S is the largest softmax probability", std::fabs(am_s - want_amax_s) <= 1e-6 * want_amax_s + 1e-9, am_s);
      expect("amax of O is the largest magnitude before scale_O", std::fabs(am_o - want_amax_o) <= 0.13 * want_amax_o, am_o);
    }
  }
  // Without O's amax, and with E8M0 descales (MXFP8), the graph is refused when it is built.
  auto graph_status = [&](Desc oper) {
    Desc g = make(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    set(g, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &H);
    set(g, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &oper);
    const cudnnStatus_t r = cudnnBackendFinalize(g);
    cudnnBackendDestroyDescriptor(g);
    return r;
  };
  Desc op2 = build(false);
  cudnnBackendFinalize(op2);
  expect("an FP8 attention without O's amax is refused", graph_status(op2) == CUDNN_STATUS_NOT_SUPPORTED);
  Desc e8 = tensor(20, one, CUDNN_DATA_FP8_E8M0);
  Desc mx = make(CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR);
  set_desc(mx, CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC, q);
  set_desc(mx, CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC, k);
  set_desc(mx, CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC, v);
  set_desc(mx, CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC, o);
  set_desc(mx, CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_QDESC, e8);
  cudnnBackendFinalize(mx);
  expect("MXFP8 attention (E8M0 descale) is refused", graph_status(mx) == CUDNN_STATUS_NOT_SUPPORTED);
  for (Desc d : {q, k, v, o, sc, dqt, dkt, dvt, dst, sst, sot, as, ao, e8, op, op2, mx}) cudnnBackendDestroyDescriptor(d);
  for (void* p : {(void*)dQ, (void*)dK, (void*)dV, (void*)dO, (void*)dsc, (void*)ddq, (void*)ddk, (void*)ddv, (void*)dds, (void*)dss,
                  (void*)dso, (void*)das, (void*)dao})
    cudaFree(p);
  cudnnDestroy(H);
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
