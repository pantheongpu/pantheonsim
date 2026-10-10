// What cuDNN's per-tensor FP8 attention (CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR) computes, asked cell by
// cell. dnn_fp8_attention.cu and dnn_fp8_attention_frontend.cpp check the documented formulas
//   S = (Q K^T) descale_Q descale_K attn, P = softmax(S), amax_S = max P,
//   P8 = E4M3(P scale_S), O = (P8 V) descale_S descale_V, amax_O = max |O|, O8 = E4M3(O scale_O)
// and on an RTX PRO 6000 (sm_120, cuDNN 9.27) both fail the O8 check while the two amax values agree: the output
// quantized with scale_O is 4.76 FP8 steps from the formula in the backend test (a want of 0.25 against 0 or 0.5, or
// 5.5 against 2 or 9, say) and 29 in the frontend's, so something between O and O8 -- or the layout O8 is written
// in -- is not what the formulas say. The log of that run holds only the verdict, so this probe is what has to be run
// on a g7e (or any Hopper or Blackwell card) to say what the card does.
//
// It runs the same graph through the backend API in several configurations and, for each, prints the card's O8 bytes,
// both amax values and a table of candidate explanations, each a variant of the formulas, with the number of elements
// of O8 that it reproduces bit for bit:
//
//   documented       the formulas above
//   no_quant_P       P is not rounded to E4M3 (O = P V descale_V)
//   P8_unscaled      P8 = E4M3(P), descale_S still applied (scale_S not used to quantize)
//   no_descale_S     O = P8 V descale_V
//   no_scale_O       O8 = E4M3(O)
//   scale_O_twice    O8 = E4M3(O scale_O scale_O)
//   rz_O8            the documented O8 rounded toward zero instead of to nearest
//   fp16_O           O rounded to fp16 before scale_O
//   bshd / hdsd      the documented O8 read as if the card had written [S][H][D] or [H][D][S]
//   rowmax_P         P8 = E4M3(P / rowmax(P) scale_S) (the flash-attention running max rescales P)
//
// and a "uniform" family (Q = 0, so S = 0 and every probability is 1/8; V constant along the sequence) in which the
// correct O8 is E4M3(descale_V V_d scale_O) exactly, for dv and scale_O in {1, 2, 4}: if it is not, the first line that
// is off says which factor the card handles differently. Run with PROBE_DUMP=<dir> to also write the raw O8 bytes of
// every configuration to <dir>/fp8attn_<config>.bin.
//
// On the simulator (a Hopper or Blackwell profile) the documented row matches in full, which checks the probe itself.
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

// Plans and runs one operation; 1 on success, 0 when no engine takes it, -1 on an execution error.
static int run(const std::vector<Desc>& ops, const std::vector<int64_t>& uids, const std::vector<void*>& ptrs) {
  Desc graph = make(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
  set(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &H);
  set(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, (int64_t)ops.size(), ops.data());
  cudnnStatus_t s = cudnnBackendFinalize(graph);
  if (s != CUDNN_STATUS_SUCCESS) {
    std::printf("     the graph did not finalize (%d)\n", (int)s);
    cudnnBackendDestroyDescriptor(graph);
    return 0;
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

// ---- E4M3 ----
static uint8_t e4m3(float f) { return (uint8_t)__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3); }
static float from_e4m3(uint8_t b) { return __half2float(__half(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3))); }
// toward zero: the largest E4M3 value not above |f|
static uint8_t e4m3_rz(float f) {
  const uint8_t n = e4m3(f);
  if (std::fabs(from_e4m3(n)) <= std::fabs(f)) return n;
  for (int c = (n & 0x7f) - 1; c >= 0; --c) {
    const uint8_t b = (uint8_t)(c | (f < 0 ? 0x80 : 0));
    if (std::fabs(from_e4m3(b)) <= std::fabs(f)) return b;
  }
  return 0;
}
static float round_half(float f) { return __half2float(__float2half_rn(f)); }

template <class T> static T* up(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(T) + 64);
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

static const int B = 1, Hh = 2, Sq = 8, Skv = 8, D = 16;

struct Cfg {
  const char* name;
  float dq, dk, dv, ds, scale_s, scale_o, attn;
  bool zero_q;       // Q = 0 (S = 0, uniform P)
  bool v_const;      // V constant along the sequence: V[h][j][d] = pattern(h, d)
};

enum Variant { DOCUMENTED, NO_QUANT_P, P8_UNSCALED, NO_DESCALE_S, NO_SCALE_O, SCALE_O_TWICE, RZ_O8, FP16_O, ROWMAX_P, N_VARIANTS };
static const char* kVariantName[N_VARIANTS] = {"documented", "no_quant_P", "P8_unscaled", "no_descale_S", "no_scale_O",
                                                "scale_O_twice", "rz_O8", "fp16_O", "rowmax_P"};

// The O8 a variant of the formulas gives, [H][Sq][D].
static std::vector<uint8_t> reference(const Cfg& c, const std::vector<uint8_t>& hq, const std::vector<uint8_t>& hk,
                                      const std::vector<uint8_t>& hv, Variant var) {
  std::vector<uint8_t> out((size_t)B * Hh * Sq * D);
  for (int h = 0; h < Hh; ++h)
    for (int i = 0; i < Sq; ++i) {
      std::vector<double> s(Skv), p(Skv);
      double mx = -1e300;
      for (int j = 0; j < Skv; ++j) {
        double acc = 0;
        for (int d = 0; d < D; ++d) acc += (double)from_e4m3(hq[(size_t)(h * Sq + i) * D + d]) * from_e4m3(hk[(size_t)(h * Skv + j) * D + d]);
        s[j] = (float)(acc * c.attn * c.dq * c.dk);
        mx = std::fmax(mx, s[j]);
      }
      double sum = 0, pmax = 0;
      for (int j = 0; j < Skv; ++j) sum += (p[j] = std::exp(s[j] - mx));
      for (int j = 0; j < Skv; ++j) pmax = std::fmax(pmax, p[j] /= sum);
      for (int e = 0; e < D; ++e) {
        double acc = 0;
        for (int j = 0; j < Skv; ++j) {
          double pq;
          const float pf = (float)p[j];
          switch (var) {
            case NO_QUANT_P: pq = p[j] * c.scale_s; break;
            case P8_UNSCALED: pq = from_e4m3(e4m3(pf)) * c.scale_s; break;
            case ROWMAX_P: pq = from_e4m3(e4m3((float)(p[j] / pmax * c.scale_s))); break;
            default: pq = from_e4m3(e4m3((float)(p[j] * c.scale_s)));
          }
          acc += pq * from_e4m3(hv[(size_t)(h * Skv + j) * D + e]);
        }
        float o = (float)(acc * (var == NO_DESCALE_S ? 1.0 : c.ds) * c.dv);
        if (var == FP16_O) o = round_half(o);
        float q = o * (var == NO_SCALE_O ? 1.0f : c.scale_o);
        if (var == SCALE_O_TWICE) q *= c.scale_o;
        out[(size_t)(h * Sq + i) * D + e] = var == RZ_O8 ? e4m3_rz(q) : e4m3(q);
      }
    }
  return out;
}

static int g_fails = 0;

static void one_config(const Cfg& c) {
  const float vals[8] = {1, -1, 0.5f, -0.5f, 2, -2, 0.25f, 3};
  auto gen = [&](size_t n, int salt) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = e4m3(vals[(i * 7 + salt * 3 + i / 5) % 8]);
    return v;
  };
  const size_t nq = (size_t)B * Hh * Sq * D, nk = (size_t)B * Hh * Skv * D;
  std::vector<uint8_t> hq = gen(nq, 1), hk = gen(nk, 2), hv = gen(nk, 3);
  if (c.zero_q) std::fill(hq.begin(), hq.end(), (uint8_t)0);
  if (c.v_const)
    for (int h = 0; h < Hh; ++h)
      for (int j = 0; j < Skv; ++j)
        for (int d = 0; d < D; ++d) hv[(size_t)(h * Skv + j) * D + d] = e4m3(vals[(d + 3 * h) % 8]);

  const std::vector<int64_t> qd = {B, Hh, Sq, D}, kd = {B, Hh, Skv, D}, one = {1, 1, 1, 1};
  Desc q = tensor(1, qd, CUDNN_DATA_FP8_E4M3), k = tensor(2, kd, CUDNN_DATA_FP8_E4M3), v = tensor(3, kd, CUDNN_DATA_FP8_E4M3),
       o = tensor(4, qd, CUDNN_DATA_FP8_E4M3), sc = tensor(5, one, CUDNN_DATA_FLOAT), dqt = tensor(6, one, CUDNN_DATA_FLOAT),
       dkt = tensor(7, one, CUDNN_DATA_FLOAT), dvt = tensor(8, one, CUDNN_DATA_FLOAT), dst = tensor(9, one, CUDNN_DATA_FLOAT),
       sst = tensor(10, one, CUDNN_DATA_FLOAT), sot = tensor(11, one, CUDNN_DATA_FLOAT),
       as = tensor(12, one, CUDNN_DATA_FLOAT), ao = tensor(13, one, CUDNN_DATA_FLOAT);
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
  set_desc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_AMAX_ODESC, ao);
  const cudnnStatus_t st = cudnnBackendFinalize(op);
  std::printf("== %s  (dq %g dk %g dv %g descale_S %g scale_S %g scale_O %g attn %g%s%s)\n", c.name, c.dq, c.dk, c.dv, c.ds,
              c.scale_s, c.scale_o, c.attn, c.zero_q ? ", Q = 0" : "", c.v_const ? ", V constant along the sequence" : "");
  if (st != CUDNN_STATUS_SUCCESS) {
    std::printf("   the operation did not finalize (%d)\n", (int)st);
    ++g_fails;
    for (Desc d : {q, k, v, o, sc, dqt, dkt, dvt, dst, sst, sot, as, ao, op}) cudnnBackendDestroyDescriptor(d);
    return;
  }
  std::vector<float> f1 = {c.attn}, fq = {c.dq}, fk = {c.dk}, fv = {c.dv}, fs = {c.ds}, fss = {c.scale_s}, fso = {c.scale_o}, z1 = {-1}, z2 = {-1};
  uint8_t *dQ = up(hq), *dK = up(hk), *dV = up(hv), *dO = up(std::vector<uint8_t>(nq, 0x55));
  float *dsc = up(f1), *ddq = up(fq), *ddk = up(fk), *ddv = up(fv), *dds = up(fs), *dss = up(fss), *dso = up(fso), *das = up(z1), *dao = up(z2);
  const int ran = run({op}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, {dQ, dK, dV, dO, dsc, ddq, ddk, ddv, dds, dss, dso, das, dao});
  if (ran == 1) {
    std::vector<uint8_t> got(nq);
    cudaMemcpy(got.data(), dO, got.size(), cudaMemcpyDeviceToHost);
    float am_s = 0, am_o = 0;
    cudaMemcpy(&am_s, das, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&am_o, dao, 4, cudaMemcpyDeviceToHost);
    std::printf("   amax_S %.9g  amax_O %.9g\n", am_s, am_o);
    for (size_t r = 0; r < got.size() / D; ++r) {
      std::printf("   O8[h%zu,i%zu]", r / Sq, r % Sq);
      for (int e = 0; e < D; ++e) std::printf(" %02x", got[r * D + e]);
      std::printf("\n");
    }
    for (int var = 0; var < N_VARIANTS; ++var) {
      const std::vector<uint8_t> want = reference(c, hq, hk, hv, (Variant)var);
      size_t same = 0;
      for (size_t i = 0; i < got.size(); ++i) same += got[i] == want[i];
      std::printf("   %-14s %3zu of %zu bytes match\n", kVariantName[var], same, got.size());
      // On the simulator the documented formulas are the model, so the probe itself is checked there.
      if (var == DOCUMENTED && std::getenv("VGPU_GPU") && same != got.size()) {
        std::printf("FAIL the documented row does not match on the simulator\n");
        ++g_fails;
      }
    }
    {   // the documented bytes as if written in another layout
      const std::vector<uint8_t> want = reference(c, hq, hk, hv, DOCUMENTED);
      size_t bshd = 0, hdsd = 0;
      for (int h = 0; h < Hh; ++h)
        for (int i = 0; i < Sq; ++i)
          for (int e = 0; e < D; ++e) {
            const uint8_t w = want[(size_t)(h * Sq + i) * D + e];
            bshd += got[(size_t)(i * Hh + h) * D + e] == w;      // [S][H][D]
            hdsd += got[(size_t)(h * D + e) * Sq + i] == w;      // [H][D][S]
          }
      std::printf("   %-14s %3zu of %zu bytes match\n", "bshd", bshd, got.size());
      std::printf("   %-14s %3zu of %zu bytes match\n", "hdsd", hdsd, got.size());
    }
    if (const char* dir = std::getenv("PROBE_DUMP")) {
      const std::string path = std::string(dir) + "/fp8attn_" + c.name + ".bin";
      if (FILE* f = std::fopen(path.c_str(), "wb")) { std::fwrite(got.data(), 1, got.size(), f); std::fclose(f); }
    }
    if (c.zero_q && c.v_const) {   // the closed form: O8[d] = E4M3(descale_V V_d scale_O), amax_O = max |descale_V V_d|
      size_t same = 0;
      for (size_t i = 0; i < got.size(); ++i) {
        const int d = (int)(i % D), h = (int)(i / ((size_t)Sq * D));
        same += got[i] == e4m3(from_e4m3(hv[(size_t)(h * Skv) * D + d]) * c.dv * c.scale_o);
      }
      std::printf("   %-14s %3zu of %zu bytes match (O8 = E4M3(V descale_V scale_O), no softmax involved)\n", "closed_form", same, got.size());
    }
  } else {
    std::printf("   did not run (%d)\n", ran);
    ++g_fails;
  }
  for (Desc d : {q, k, v, o, sc, dqt, dkt, dvt, dst, sst, sot, as, ao, op}) cudnnBackendDestroyDescriptor(d);
  for (void* p : {(void*)dQ, (void*)dK, (void*)dV, (void*)dO, (void*)dsc, (void*)ddq, (void*)ddk, (void*)ddv, (void*)dds, (void*)dss,
                  (void*)dso, (void*)das, (void*)dao})
    cudaFree(p);
}

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  int major = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  if (major < 9) {
    std::printf("SKIP: FP8 attention needs a Hopper or Blackwell GPU (or profile)\nPASS\n");
    return 0;
  }
  if (cudnnCreate(&H) != CUDNN_STATUS_SUCCESS) {
    std::printf("FAIL cudnnCreate\n");
    return 1;
  }
  const float attn = 0.25f;
  const Cfg cfgs[] = {
      // the configuration of dnn_fp8_attention.cu
      {"documented_test", 0.5f, 0.25f, 2.0f, 1.0f / 256.0f, 256.0f, 4.0f, attn, false, false},
      // each factor on its own
      {"scale_O_1", 0.5f, 0.25f, 2.0f, 1.0f / 256.0f, 256.0f, 1.0f, attn, false, false},
      {"scale_O_16", 0.5f, 0.25f, 2.0f, 1.0f / 256.0f, 256.0f, 16.0f, attn, false, false},
      {"scale_S_16", 0.5f, 0.25f, 2.0f, 1.0f / 16.0f, 16.0f, 4.0f, attn, false, false},
      // scales that are not powers of two, so that where a factor is applied (before or after a rounding) shows
      {"scale_S_100", 0.5f, 0.25f, 2.0f, 0.01f, 100.0f, 4.0f, attn, false, false},
      {"scale_O_3", 0.5f, 0.25f, 2.0f, 1.0f / 256.0f, 256.0f, 3.0f, attn, false, false},
      {"descale_V_1", 0.5f, 0.25f, 1.0f, 1.0f / 256.0f, 256.0f, 4.0f, attn, false, false},
      {"unit_scales", 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, false, false},
      // Q = 0 and V constant along the sequence: O = descale_V V_d, scale_O only
      {"uniform_dv1_so1", 1.0f, 1.0f, 1.0f, 1.0f / 256.0f, 256.0f, 1.0f, attn, true, true},
      {"uniform_dv2_so1", 1.0f, 1.0f, 2.0f, 1.0f / 256.0f, 256.0f, 1.0f, attn, true, true},
      {"uniform_dv1_so4", 1.0f, 1.0f, 1.0f, 1.0f / 256.0f, 256.0f, 4.0f, attn, true, true},
      {"uniform_dv2_so4", 1.0f, 1.0f, 2.0f, 1.0f / 256.0f, 256.0f, 4.0f, attn, true, true},
      {"uniform_dv1_so2_ss1", 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, attn, true, true},
  };
  for (const Cfg& c : cfgs) one_config(c);
  cudnnDestroy(H);
  std::printf("%s\n", g_fails ? "FAIL" : "PASS");
  return g_fails ? 1 : 0;
}
