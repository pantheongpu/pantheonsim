// cuBLASLt's narrow-precision matmuls (FP8, MXFP8, NVFP4 and their scale modes),
// swept over descriptors and printed one line per case, so the same program can
// run against NVIDIA's cuBLASLt on a card and against VirtualGPU's and the two
// transcripts compared. Each line is the heuristic's status and count, the
// matmul's status, and an FNV-1a hash of D (and of the auxiliary output, the
// amax values and the block scales D's quantization writes), so a rounding,
// saturation or layout difference changes the line.
//
//   lowprec_lt [--group <prefix>]... [--list]
//
// The data are small exact values (halves, integers) so every product and sum is
// exact in fp32 and the result does not depend on the order the library
// accumulates in. Lines that start with '#' name the machine (device, library
// version) and are not compared.
//
// nvidia/tests/e2e/run_lowprec.sh runs it: against the shims on a simulated GPU
// with the profile named by the expected file, or with --card against NVIDIA's
// libraries on a real GPU (--update rewrites the expected file from the card).
// Set PROBE_DUMP=<dir> to write every case's inputs and outputs there as files,
// for finding which element differs.
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "lowprec_common.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

using namespace lp;

// Named by value: CUDA 12.0's headers, which CI builds with, have none of these.
namespace {
constexpr int M_SCALAR = 0, M_VEC16 = 1, M_VEC32 = 2, M_OUTER = 3, M_VEC128 = 4, M_BLK128 = 5, M_BATCH = 6,
              M_MN_K4_128 = 12, M_MN_K4_32 = 13;
constexpr int A_AUX_DATA_TYPE = 22, A_AUX_SCALE = 23, A_AUX_AMAX = 24, A_FAST = 25, A_BIAS_TYPE = 26;
constexpr int A_ASM = 31, A_BSM = 32, A_CSM = 33, A_DSM = 34, A_DOSP = 36, A_DOSM = 37;
constexpr int EP_DEFAULT = 1, EP_RELU = 2, EP_RELU_AUX = 2 | 128, EP_BIAS = 4, EP_RELU_BIAS = 6, EP_RELU_AUX_BIAS = 6 | 128,
              EP_GELU = 32, EP_GELU_AUX = 32 | 128, EP_GELU_BIAS = 36, EP_GELU_AUX_BIAS = 36 | 128,
              EP_DRELU = 8 | 128, EP_DRELU_BGRAD = 8 | 16 | 128, EP_DGELU = 64 | 128, EP_DGELU_BGRAD = 64 | 16 | 128,
              EP_BGRADA = 256, EP_BGRADB = 512;

struct Spec {
  std::string name;
  int ta = 1, tb = 0;                 // cublasOperation_t of A and B
  int at = T_E4M3, bt = T_E4M3, ct = T_BF16, dt = T_BF16;
  int compute = CUBLAS_COMPUTE_32F, stype = CUDA_R_32F;
  int m = 64, n = 64, k = 64, batch = 1;
  int epilogue = EP_DEFAULT, bias_type = -1, aux_type = -1;
  int am = M_SCALAR, bm = M_SCALAR, cm = M_SCALAR, dm = M_SCALAR, dom = M_SCALAR, auxm = -1;
  bool a_scale = true, b_scale = true, c_scale = false, d_scale = false, amax = false, aux_scale = false,
       aux_amax = false, d_out = false;
  float alpha = 1.f, beta = 0.f;
  int fast = -1;
  int pointer_mode = 0;
  bool big = false;                   // large values, to saturate narrow outputs
  bool c_null = false;
  int oa = 0, ob = 0, oc = 0, od = 0;           // 1: CUBLASLT_ORDER_ROW
  int pad_a = 0, pad_b = 0, pad_c = 0, pad_d = 0;   // leading dimension beyond the minimum
};

// The bytes one scale tensor of this mode takes for `outer` rows and `inner` elements along the blocks.
size_t scale_bytes(int mode, size_t outer, size_t inner) {
  switch (mode) {
    case M_SCALAR: return 4;
    case M_OUTER: return outer * 4;
    case M_VEC16: return tiled_bytes(outer, cdiv(inner, 16));
    case M_VEC32: return tiled_bytes(outer, cdiv(inner, 32));
    case M_VEC128: return outer * cdiv(inner, 128) * 4;
    case M_BLK128: return cdiv(cdiv(inner, 128), 4) * 4 * cdiv(outer, 128) * 4;
    case M_MN_K4_128: return cdiv(outer, 4) * 4 * cdiv(inner, 512) * 4;
    case M_MN_K4_32: return cdiv(outer, 4) * 4 * cdiv(inner, 128) * 4;
    default: return 4;
  }
}
std::vector<uint8_t> make_scales(int mode, size_t bytes, uint32_t salt) {
  std::vector<uint8_t> b(bytes + 64, 0);
  static const float f32s[4] = {0.5f, 1.f, 2.f, 0.25f};
  for (size_t i = 0; i < bytes; ++i) {
    const uint32_t r = mix((uint32_t)i * 40503u + salt);
    switch (mode) {
      case M_VEC16: { static const uint8_t t[4] = {0x30, 0x38, 0x40, 0x28}; b[i] = t[r & 3]; break; }
      case M_VEC32: case M_MN_K4_128: case M_MN_K4_32: { static const uint8_t t[4] = {126, 127, 128, 125}; b[i] = t[r & 3]; break; }
      default: { if (i % 4 == 0 && i + 4 <= bytes) { const float f = f32s[r & 3]; std::memcpy(&b[i], &f, 4); } break; }
    }
  }
  return b;
}

cublasLtHandle_t g_lt;
void* g_ws = nullptr;
constexpr size_t kWs = 32u << 20;

struct Setter {
  cublasLtMatmulDesc_t d;
  std::string bad;
  template <class V> void set(const char* what, int attr, V v) {
    const int st = (int)cublasLtMatmulDescSetAttribute(d, (cublasLtMatmulDescAttributes_t)attr, &v, sizeof v);
    if (st && bad.empty()) bad = std::string(what) + ":" + std::to_string(st);
  }
};


// A GELU case's check: D against the exact tanh-approximation GELU (or its derivative) of the product,
// computed on the host from what the library was given. Scalar-scaled FP8 products of one batch only.
std::string gelu_check(const Spec& s, const std::vector<uint8_t>& D, const std::vector<uint8_t>& A, const std::vector<uint8_t>& B,
                       const std::vector<uint8_t>& C, const std::vector<uint8_t>& SA, const std::vector<uint8_t>& SB,
                       const std::vector<uint8_t>& SC, const std::vector<uint8_t>& bias, const std::vector<uint8_t>& auxin,
                       int64_t lda, int64_t ldb, int64_t ldc, int64_t ldd, int64_t aux_ld) {
  const bool fp8 = (s.at == T_E4M3 || s.at == T_E5M2) && (s.bt == T_E4M3 || s.bt == T_E5M2);
  if (!fp8 || s.am || s.bm || s.cm || s.batch != 1 || s.pointer_mode || s.big) return "na";
  const size_t m = s.m, n = s.n, k = s.k;
  auto at = [](size_t r, size_t c, int64_t ld, int order) { return order ? r * (size_t)ld + c : c * (size_t)ld + r; };
  auto scale = [](const std::vector<uint8_t>& v, bool set) { float f = 1.f; if (set && v.size() >= 4) std::memcpy(&f, v.data(), 4); return (double)f; };
  const double sa = scale(SA, s.a_scale), sb = scale(SB, s.b_scale), sc = scale(SC, s.c_scale);
  const bool narrow_d = s.dt == T_E4M3 || s.dt == T_E5M2;
  const double sd = narrow_d && s.d_scale ? (s.big ? 8.0 : 4.0) : 1.0;
  const int bias_t = s.bias_type >= 0 ? s.bias_type : (narrow_d || s.dt == T_F32 ? T_BF16 : s.dt);
  const int aux_t = s.aux_type >= 0 ? s.aux_type : s.dt;
  const bool fwd = (s.epilogue & 32) != 0, has_bias = (s.epilogue & 4) != 0;
  double worst = 0;
  bool ok = true;
  for (size_t j = 0; j < n && ok; ++j)
    for (size_t i = 0; i < m; ++i) {
      double acc = 0;
      for (size_t p = 0; p < k; ++p) {
        const double a = dec_elem(s.at, A.data(), s.ta ? at(p, i, lda, s.oa) : at(i, p, lda, s.oa));
        const double b = dec_elem(s.bt, B.data(), s.tb ? at(j, p, ldb, s.ob) : at(p, j, ldb, s.ob));
        acc += a * b;
      }
      double x = s.alpha * sa * sb * acc;
      if (s.beta != 0.f) x += s.beta * sc * dec_elem(s.ct, C.data(), at(i, j, ldc, s.oc));
      if (has_bias) x += dec_elem(bias_t, bias.data(), i);
      double ref;
      if (fwd) {
        ref = 0.5 * x * (1 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
      } else {
        const double a = dec_elem(aux_t, auxin.data(), j * (size_t)aux_ld + i);
        const double t = std::tanh(0.7978845608028654 * (a + 0.044715 * a * a * a));
        ref = x * (0.5 * (1 + t) + 0.5 * a * (1 - t * t) * 0.7978845608028654 * (1 + 3 * 0.044715 * a * a));
      }
      ref *= sd;
      const double got = dec_elem(s.dt, D.data(), at(i, j, ldd, s.od));
      const double rel = narrow_d ? (s.dt == T_E4M3 ? 0.13 : 0.26) : (s.dt == T_F32 ? 1e-3 : 0.02);
      const double tol = rel * (std::fabs(ref) + 0.25 * std::fabs(x) * sd) + 1e-3 * sd;
      const double err = std::fabs(got - ref);
      worst = std::fmax(worst, err / tol);
      if (!(err <= tol)) { ok = false; break; }
    }
  return ok ? "ok" : "BAD";
}

void run(const Spec& s) {
  std::string line = s.name;
  auto emit = [&] { std::printf("%s\n", line.c_str()); std::fflush(stdout); };
  std::fprintf(stderr, "RUN %s\n", s.name.c_str());
  cublasLtMatmulDesc_t desc;
  int st = (int)cublasLtMatmulDescCreate(&desc, (cublasComputeType_t)s.compute, (cudaDataType)s.stype);
  if (st) { line += " create=" + std::to_string(st); emit(); return; }
  Setter S{desc, ""};
  const int ta = s.ta, tb = s.tb;
  S.set("transa", CUBLASLT_MATMUL_DESC_TRANSA, (int32_t)ta);
  S.set("transb", CUBLASLT_MATMUL_DESC_TRANSB, (int32_t)tb);
  const size_t m = s.m, n = s.n, k = s.k;
  const size_t la_r = ta ? k : m, la_c = ta ? m : k, lb_r = tb ? n : k, lb_c = tb ? k : n;
  const int batch = s.batch;
  // Leading dimensions: rows for column order, columns for row order, plus the padding.
  const int64_t lda = (s.oa ? la_c : la_r) + s.pad_a, ldb = (s.ob ? lb_c : lb_r) + s.pad_b, ldc = (s.oc ? n : m) + s.pad_c,
                ldd = (s.od ? n : m) + s.pad_d;
  const size_t a_elems = (size_t)lda * (s.oa ? la_r : la_c), b_elems = (size_t)ldb * (s.ob ? lb_r : lb_c),
               c_elems = (size_t)ldc * (s.oc ? m : n), d_elems = (size_t)ldd * (s.od ? m : n);
  std::vector<uint8_t> hSA, hSB, hSC, hBias, hAuxIn;
  Dev dA, dB, dC, dD, dSA, dSB, dSC, dSD, dSO, dBias, dAux, dAuxS, dAuxMax, dAmax, dAlpha, dBeta;
  // Host copies of what the library is given, for the GELU checks below.
  const std::vector<uint8_t> hA = make_matrix(s.at, a_elems * batch, 1, s.big), hB = make_matrix(s.bt, b_elems * batch, 2, s.big),
                             hC = make_matrix(s.ct, c_elems * batch, 3, false);
  dA.put(hA);
  dB.put(hB);
  dC.put(hC);
  dD.fill(bytes_of(s.dt, d_elems * batch), 0xA5);
  // Scale tensors (a batch of block-scaled matmuls gets one tensor per batch, one after another).
  auto mk = [&](Dev& d, int mode, size_t outer, size_t inner, uint32_t salt, int copies) -> std::vector<uint8_t> {
    const size_t one = scale_bytes(mode, outer, inner);
    std::vector<uint8_t> b = make_scales(mode, one * (mode == M_BATCH ? 1 : copies), salt);
    if (mode == M_BATCH) b = make_scales(mode, 4 * copies, salt);
    d.put(b);
    return b;
  };
  if (s.a_scale) { hSA = mk(dSA, s.am, m, k, 11, batch); S.set("a_scale", CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, (const void*)dSA.p); }
  if (s.b_scale) { hSB = mk(dSB, s.bm, n, k, 12, batch); S.set("b_scale", CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, (const void*)dSB.p); }
  if (s.c_scale) { hSC = mk(dSC, s.cm, n, m, 13, batch); S.set("c_scale", CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, (const void*)dSC.p); }
  if (s.d_scale) {
    const float sc = s.big ? 8.f : 4.f;
    std::vector<uint8_t> b(68, 0);
    std::memcpy(b.data(), &sc, 4);
    dSD.put(b);
    S.set("d_scale", CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, (const void*)dSD.p);
  }
  if (s.am) S.set("a_mode", A_ASM, (int32_t)s.am);
  if (s.bm) S.set("b_mode", A_BSM, (int32_t)s.bm);
  if (s.cm) S.set("c_mode", A_CSM, (int32_t)s.cm);
  if (s.dm) S.set("d_mode", A_DSM, (int32_t)s.dm);
  if (s.dom) {
    S.set("dout_mode", A_DOSM, (int32_t)s.dom);
    dSO.fill(scale_bytes(s.dom, n, m) * batch, 0xEE);
    S.set("dout_ptr", A_DOSP, (void*)dSO.p);
  }
  if (s.amax) {
    const float sentinel = 12345.f;
    std::vector<uint8_t> b(68, 0);
    std::memcpy(b.data(), &sentinel, 4);
    dAmax.put(b);
    S.set("amax", CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, (void*)dAmax.p);
  }
  if (s.fast >= 0) S.set("fast", A_FAST, (int8_t)s.fast);
  if (s.epilogue != EP_DEFAULT) S.set("epilogue", CUBLASLT_MATMUL_DESC_EPILOGUE, (uint32_t)s.epilogue);
  const int e = s.epilogue;
  const bool bias = (e & 4) != 0, aux = (e & 128) != 0;
  const bool bgrad = (e & 16) != 0 || e == EP_BGRADA || e == EP_BGRADB;   // the bias pointer is an output
  const bool aux_in = aux && ((e & 8) || (e & 64));                       // DRELU and DGELU read the buffer
  const bool relu = (e & 2) != 0 || (e & 8) != 0;                         // a bit mask, not values
  if (bias || bgrad) {
    // The bias is generated in the type the library reads it as: the one asked for, else BF16 under an FP8 or fp32 D.
    const int bt = s.bias_type >= 0 ? s.bias_type : (s.dt == T_E4M3 || s.dt == T_E5M2 || s.dt == T_F32 ? T_BF16 : s.dt);
    if (bias) { hBias = make_matrix(bt, m, 5, false); dBias.put(hBias); }
    else dBias.fill(bytes_of(bt, std::max(m, n)), 0xA5);
    S.set("bias", CUBLASLT_MATMUL_DESC_BIAS_POINTER, (const void*)dBias.p);
    if (s.bias_type >= 0) S.set("bias_type", A_BIAS_TYPE, (int32_t)s.bias_type);
  }
  const int64_t aux_ld = relu ? 128 : (int64_t)m;
  if (aux) {
    const int auxt = s.aux_type >= 0 ? s.aux_type : s.dt;
    if (aux_in && relu) dAux.fill((size_t)(aux_ld * n + 7) / 8 + 64, 0xA5);
    else if (aux_in) { hAuxIn = make_matrix(auxt, (size_t)aux_ld * n, 7, false); dAux.put(hAuxIn); }
    else dAux.fill(relu ? (size_t)(aux_ld * n + 7) / 8 + 64 : bytes_of(auxt, (size_t)aux_ld * n), 0xA5);
    S.set("aux", CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, (void*)dAux.p);
    S.set("aux_ld", CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, (int64_t)aux_ld);
    if (s.aux_type >= 0) S.set("aux_type", A_AUX_DATA_TYPE, (int32_t)s.aux_type);
    if (s.aux_scale) {
      const float sc = 2.f;
      std::vector<uint8_t> b(68, 0);
      std::memcpy(b.data(), &sc, 4);
      dAuxS.put(b);
      S.set("aux_scale", A_AUX_SCALE, (const void*)dAuxS.p);
    }
    if (s.aux_amax) {
      const float sentinel = 12345.f;
      std::vector<uint8_t> b(68, 0);
      std::memcpy(b.data(), &sentinel, 4);
      dAuxMax.put(b);
      S.set("aux_amax", A_AUX_AMAX, (void*)dAuxMax.p);
    }
    if (s.auxm >= 0) S.set("aux_mode", 35, (int32_t)s.auxm);
  }
  if (s.pointer_mode) {
    S.set("pointer_mode", CUBLASLT_MATMUL_DESC_POINTER_MODE, (int32_t)s.pointer_mode);
    dAlpha.put(std::vector<uint8_t>((const uint8_t*)&s.alpha, (const uint8_t*)&s.alpha + 4));
    dBeta.put(std::vector<uint8_t>((const uint8_t*)&s.beta, (const uint8_t*)&s.beta + 4));
  }

  cublasLtMatrixLayout_t LA, LB, LC, LD;
  int ls = 0;
  ls |= (int)cublasLtMatrixLayoutCreate(&LA, (cudaDataType)s.at, la_r, la_c, lda);
  ls |= (int)cublasLtMatrixLayoutCreate(&LB, (cudaDataType)s.bt, lb_r, lb_c, ldb);
  ls |= (int)cublasLtMatrixLayoutCreate(&LC, (cudaDataType)s.ct, m, n, ldc);
  ls |= (int)cublasLtMatrixLayoutCreate(&LD, (cudaDataType)s.dt, m, n, ldd);
  {
    cublasLtMatrixLayout_t all[4] = {LA, LB, LC, LD};
    const int orders[4] = {s.oa, s.ob, s.oc, s.od};
    for (int i = 0; i < 4; ++i)
      if (orders[i]) {
        const int32_t ord = CUBLASLT_ORDER_ROW;
        ls |= (int)cublasLtMatrixLayoutSetAttribute(all[i], CUBLASLT_MATRIX_LAYOUT_ORDER, &ord, sizeof ord);
      }
  }
  if (batch > 1) {
    const int32_t bc = batch;
    cublasLtMatrixLayout_t ls_[4] = {LA, LB, LC, LD};
    const int64_t strides[4] = {(int64_t)a_elems, (int64_t)b_elems, (int64_t)c_elems, (int64_t)d_elems};
    for (int i = 0; i < 4; ++i) {
      cublasLtMatrixLayoutSetAttribute(ls_[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof bc);
      cublasLtMatrixLayoutSetAttribute(ls_[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strides[i], sizeof(int64_t));
    }
  }
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  size_t ws = kWs;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws);
  cublasLtMatmulHeuristicResult_t hr;
  std::memset(&hr, 0, sizeof hr);
  int got = 0;
  const int hs = (int)cublasLtMatmulAlgoGetHeuristic(g_lt, desc, LA, LB, LC, LD, pref, 1, &hr, &got);
  line += " h=" + std::to_string(hs) + ":" + std::to_string(got);
  if (!S.bad.empty()) line += " set=" + S.bad;
  if (ls) line += " layout=" + std::to_string(ls);

  const float alpha = s.alpha, beta = s.beta;
  const void* pa = s.pointer_mode ? (const void*)dAlpha.p : (const void*)&alpha;
  const void* pb = s.pointer_mode ? (const void*)dBeta.p : (const void*)&beta;
  const int ms = (int)cublasLtMatmul(g_lt, desc, pa, dA.p, LA, dB.p, LB, pb, s.c_null ? nullptr : dC.p, LC, dD.p, LD,
                                     got > 0 ? &hr.algo : nullptr, g_ws, kWs, 0);
  const cudaError_t sync = cudaDeviceSynchronize();
  line += " m=" + std::to_string(ms);
  if (sync != cudaSuccess) line += " rt=" + std::to_string((int)sync);
  if (ms == 0) {
    const auto d = dD.get();
    char buf[64];
    // GELU is approximated differently by the library on each GPU (and by VirtualGPU), so a GELU case is not
    // hashed: D is checked against the exact function instead, within a tolerance.
    const bool gelu_fwd = (e & 32) != 0, gelu_bwd = (e & 64) != 0;
    if (gelu_fwd || gelu_bwd) {
      line += " gelu=" + gelu_check(s, d, hA, hB, hC, hSA, hSB, hSC, hBias, hAuxIn, lda, ldb, ldc, ldd, aux_ld);
    } else {
      std::snprintf(buf, sizeof buf, " d=%016llx", (unsigned long long)fnv(d.data(), d.size()));
      line += buf;
    }
    dump(s.name, "d", d.data(), d.size());
    dump(s.name, "a", dA.get().data(), dA.n);
    dump(s.name, "b", dB.get().data(), dB.n);
    auto hex = [&](const char* tag, Dev& dev, size_t off) {
      const auto v = dev.get();
      float f;
      std::memcpy(&f, v.data() + off, 4);
      uint32_t u;
      std::memcpy(&u, &f, 4);
      std::snprintf(buf, sizeof buf, " %s=%08x", tag, u);
      line += buf;
    };
    if (s.amax) hex("amax", dAmax, 0);
    if (aux && !aux_in && s.aux_amax) hex("auxamax", dAuxMax, 0);
    if (aux && !aux_in) {
      const auto x = dAux.get();
      std::snprintf(buf, sizeof buf, " aux=%016llx", (unsigned long long)fnv(x.data(), x.size()));
      line += buf;
      dump(s.name, "aux", x.data(), x.size());
    }
    if (bgrad) {
      const auto x = dBias.get();
      if (!gelu_bwd) {
        std::snprintf(buf, sizeof buf, " bgrad=%016llx", (unsigned long long)fnv(x.data(), x.size()));
        line += buf;
      }
      dump(s.name, "bgrad", x.data(), x.size());
    }
    if (s.dom) {
      const auto x = dSO.get();
      std::snprintf(buf, sizeof buf, " dscale=%016llx", (unsigned long long)fnv(x.data(), x.size()));
      line += buf;
      dump(s.name, "dscale", x.data(), x.size());
    }
  }
  emit();
  cublasLtMatmulPreferenceDestroy(pref);
  for (auto l : {LA, LB, LC, LD}) cublasLtMatrixLayoutDestroy(l);
  cublasLtMatmulDescDestroy(desc);
}

std::vector<Spec> g_specs;
std::string tn(int t) {
  switch (t) {
    case T_E4M3: return "e4m3"; case T_E5M2: return "e5m2"; case T_FP4: return "fp4"; case T_BF16: return "bf16";
    case T_F16: return "f16"; case T_F32: return "f32"; default: return std::to_string(t);
  }
}
std::string mn(int m) {
  switch (m) {
    case M_SCALAR: return "sc"; case M_VEC16: return "v16"; case M_VEC32: return "v32"; case M_OUTER: return "outer";
    case M_VEC128: return "v128"; case M_BLK128: return "b128"; case M_BATCH: return "pb";
    case M_MN_K4_128: return "mk128"; case M_MN_K4_32: return "mk32"; default: return std::to_string(m);
  }
}
std::string op(const Spec& s) { return std::string(s.ta ? "T" : "N") + (s.tb ? "T" : "N"); }
void add(Spec s, const std::string& group, const std::string& tag) {
  s.name = group + "/" + tag;
  g_specs.push_back(s);
}


std::string spec_str(const Spec& s) {
  char b[640];
  std::snprintf(b, sizeof b,
                "%s ta=%d tb=%d at=%d bt=%d ct=%d dt=%d compute=%d stype=%d m=%d n=%d k=%d batch=%d ep=%d bias=%d aux=%d "
                "am=%d bm=%d cm=%d dm=%d dom=%d as=%d bs=%d cs=%d ds=%d amax=%d auxs=%d auxa=%d alpha=%g beta=%g fast=%d pm=%d "
                "big=%d cnull=%d ord=%d%d%d%d pad=%d,%d,%d,%d",
                s.name.c_str(), s.ta, s.tb, s.at, s.bt, s.ct, s.dt, s.compute, s.stype, s.m, s.n, s.k, s.batch, s.epilogue,
                s.bias_type, s.aux_type, s.am, s.bm, s.cm, s.dm, s.dom, s.a_scale, s.b_scale, s.c_scale, s.d_scale, s.amax,
                s.aux_scale, s.aux_amax, s.alpha, s.beta, s.fast, s.pointer_mode, s.big, s.c_null, s.oa, s.ob, s.oc, s.od,
                s.pad_a, s.pad_b, s.pad_c, s.pad_d);
  return b;
}

// ---- the case lists ----
void fp8_cases() {
  // Types and layouts: A, B in E4M3/E5M2, C/D combinations, every transpose.
  const int ab[4][2] = {{T_E4M3, T_E4M3}, {T_E4M3, T_E5M2}, {T_E5M2, T_E4M3}, {T_E5M2, T_E5M2}};
  const int cd[][2] = {{T_BF16, T_BF16}, {T_F16, T_F16}, {T_F32, T_F32}, {T_BF16, T_E4M3}, {T_F16, T_E4M3},
                       {T_BF16, T_E5M2}, {T_F16, T_E5M2}, {T_E4M3, T_E4M3}, {T_E5M2, T_E5M2}, {T_F16, T_BF16},
                       {T_F32, T_BF16}, {T_BF16, T_F32}, {T_F32, T_E4M3}, {T_E4M3, T_BF16}, {T_E4M3, T_F16}};
  for (int ta = 0; ta < 2; ++ta)
    for (int tb = 0; tb < 2; ++tb)
      for (auto& x : ab)
        for (auto& y : cd) {
          Spec s;
          s.ta = ta; s.tb = tb; s.at = x[0]; s.bt = x[1]; s.ct = y[0]; s.dt = y[1];
          const bool narrow_d = y[1] == T_E4M3 || y[1] == T_E5M2;
          s.d_scale = narrow_d; s.amax = narrow_d;
          add(s, "fp8types", op(s) + "_" + tn(x[0]) + tn(x[1]) + "_c" + tn(y[0]) + "_d" + tn(y[1]));
        }
  // amax and D scale for non-narrow outputs; no D scale for a narrow one.
  for (int dt : {T_BF16, T_F16, T_F32}) {
    Spec s; s.dt = dt; s.ct = dt; s.amax = true; add(s, "fp8amax", "amax_d" + tn(dt));
    s.d_scale = true; add(s, "fp8amax", "amax_scale_d" + tn(dt));
  }
  { Spec s; s.dt = T_E4M3; s.ct = T_BF16; s.d_scale = false; s.amax = true; add(s, "fp8amax", "e4m3_noscale_amax"); }
  { Spec s; s.dt = T_E4M3; s.ct = T_BF16; s.d_scale = true; s.amax = false; add(s, "fp8amax", "e4m3_scale_noamax"); }
  { Spec s; s.dt = T_E4M3; s.ct = T_BF16; s.d_scale = false; s.amax = false; add(s, "fp8amax", "e4m3_noscale_noamax"); }
  { Spec s; s.a_scale = false; s.b_scale = false; add(s, "fp8amax", "no_ab_scales"); }
  { Spec s; s.a_scale = true; s.b_scale = false; add(s, "fp8amax", "a_scale_only"); }
  { Spec s; s.a_scale = false; s.b_scale = true; add(s, "fp8amax", "b_scale_only"); }
  // Saturation: values that overflow a narrow D.
  for (int dt : {T_E4M3, T_E5M2}) {
    Spec s; s.dt = dt; s.ct = T_BF16; s.d_scale = true; s.amax = true; s.big = true;
    add(s, "fp8sat", "d" + tn(dt));
    s.at = T_E5M2; s.bt = T_E4M3; add(s, "fp8sat", "d" + tn(dt) + "_e5m2xe4m3");
  }
  { Spec s; s.dt = T_F16; s.ct = T_F16; s.big = true; s.at = T_E5M2; s.bt = T_E5M2; s.amax = true; add(s, "fp8sat", "f16_overflow"); }
  // Fast accumulation.
  for (int f : {0, 1}) {
    Spec s; s.fast = f; add(s, "fp8fast", "fast" + std::to_string(f));
    s.dt = T_E4M3; s.d_scale = true; s.amax = true; add(s, "fp8fast", "fast" + std::to_string(f) + "_e4m3");
  }
  // Dimensions and alignment.
  const int dims[][3] = {{16, 16, 16}, {8, 8, 8},    {8, 8, 16},  {16, 16, 8},  {24, 24, 24}, {32, 16, 48}, {17, 16, 16},
                         {16, 17, 16}, {16, 16, 17}, {4, 4, 16},  {1, 1, 16},   {64, 8, 64},  {64, 64, 128}, {128, 128, 128},
                         {256, 128, 256}, {16, 64, 16}, {48, 80, 32}, {12, 12, 12}, {64, 64, 24}, {64, 64, 40}};
  for (auto& d : dims) {
    Spec s; s.m = d[0]; s.n = d[1]; s.k = d[2];
    add(s, "fp8dims", "TN_" + std::to_string(d[0]) + "x" + std::to_string(d[1]) + "x" + std::to_string(d[2]));
    s.dt = T_E4M3; s.ct = T_BF16; s.d_scale = true; s.amax = true;
    add(s, "fp8dims", "TN_e4m3out_" + std::to_string(d[0]) + "x" + std::to_string(d[1]) + "x" + std::to_string(d[2]));
  }
  // Scale modes: scalars, outer vectors, missing pointers.
  for (int am : {M_SCALAR, M_OUTER})
    for (int bm : {M_SCALAR, M_OUTER})
      for (int dt : {T_BF16, T_E4M3}) {
        Spec s; s.am = am; s.bm = bm; s.dt = dt; s.d_scale = dt == T_E4M3; s.amax = dt == T_E4M3;
        add(s, "fp8scales", mn(am) + "_" + mn(bm) + "_d" + tn(dt));
        s.ta = 0; add(s, "fp8scales", "NN_" + mn(am) + "_" + mn(bm) + "_d" + tn(dt));
      }
  // C input.
  for (float beta : {1.f, 0.5f})
    for (int ct : {T_BF16, T_F16, T_F32, T_E4M3})
      for (int dt : {T_BF16, T_F16, T_F32, T_E4M3}) {
        Spec s; s.beta = beta; s.ct = ct; s.dt = dt; s.c_scale = true; s.d_scale = dt == T_E4M3; s.amax = dt == T_E4M3;
        add(s, "fp8beta", "beta" + std::to_string((int)(beta * 2)) + "_c" + tn(ct) + "_d" + tn(dt));
      }
  { Spec s; s.beta = 1.f; s.ct = T_BF16; s.c_scale = false; add(s, "fp8beta", "beta1_noCscale"); }
  { Spec s; s.beta = 1.f; s.c_null = true; add(s, "fp8beta", "beta1_Cnull"); }
  // Alpha: not 1.
  { Spec s; s.alpha = 0.5f; add(s, "fp8alpha", "alpha_half"); s.alpha = 3.f; add(s, "fp8alpha", "alpha_3"); }
  { Spec s; s.pointer_mode = 1; s.alpha = 2.f; s.beta = 0.f; add(s, "fp8alpha", "device_pointer_mode"); }
  { Spec s; s.pointer_mode = 1; s.alpha = 2.f; s.beta = 1.f; s.ct = T_BF16; add(s, "fp8alpha", "device_pointer_mode_beta1"); }
  // Batches.
  for (int b : {2, 3}) {
    Spec s; s.batch = b; add(s, "fp8batch", "scalar_b" + std::to_string(b));
    s.dt = T_E4M3; s.d_scale = true; s.amax = true; add(s, "fp8batch", "scalar_e4m3_b" + std::to_string(b));
    s = Spec(); s.batch = b; s.am = M_BATCH; s.bm = M_BATCH; add(s, "fp8batch", "perbatch_b" + std::to_string(b));
    s.am = M_BATCH; s.bm = M_SCALAR; add(s, "fp8batch", "perbatch_a_b" + std::to_string(b));
    s.am = M_SCALAR; s.bm = M_BATCH; add(s, "fp8batch", "perbatch_b_b" + std::to_string(b));
    s.am = M_OUTER; s.bm = M_OUTER; add(s, "fp8batch", "outer_b" + std::to_string(b));
  }
  // Compute and scale types.
  { Spec s; s.compute = CUBLAS_COMPUTE_16F; s.stype = T_F16; add(s, "fp8compute", "16F"); }
  { Spec s; s.compute = CUBLAS_COMPUTE_32F_FAST_16F; add(s, "fp8compute", "32F_FAST_16F"); }
  { Spec s; s.compute = CUBLAS_COMPUTE_32F_FAST_TF32; add(s, "fp8compute", "32F_FAST_TF32"); }
  { Spec s; s.stype = T_F16; add(s, "fp8compute", "scale_f16"); }
  { Spec s; s.stype = T_BF16; add(s, "fp8compute", "scale_bf16"); }
  { Spec s; s.compute = CUBLAS_COMPUTE_64F; s.stype = CUDA_R_64F; add(s, "fp8compute", "64F"); }
}

void fp8_epilogues() {
  const int eps[] = {EP_DEFAULT, EP_RELU, EP_RELU_AUX, EP_BIAS, EP_RELU_BIAS, EP_RELU_AUX_BIAS, EP_GELU, EP_GELU_AUX,
                     EP_GELU_BIAS, EP_GELU_AUX_BIAS};
  const char* en[] = {"default", "relu", "relu_aux", "bias", "relu_bias", "relu_aux_bias", "gelu", "gelu_aux", "gelu_bias",
                      "gelu_aux_bias"};
  for (int dt : {T_BF16, T_F16, T_F32, T_E4M3, T_E5M2})
    for (int e = 0; e < 10; ++e) {
      const bool narrow_d = dt == T_E4M3 || dt == T_E5M2;
      const bool bias = (eps[e] & 4) != 0, aux = (eps[e] & 128) != 0;
      std::vector<int> bts = bias ? std::vector<int>{-1, T_BF16, T_F16, T_F32} : std::vector<int>{-1};
      std::vector<int> axs = aux ? std::vector<int>{-1, T_F16, T_BF16, T_E4M3, T_F32} : std::vector<int>{-1};
      for (int bt : bts)
        for (int ax : axs) {
          Spec s; s.dt = dt; s.ct = narrow_d ? T_BF16 : dt; s.epilogue = eps[e]; s.bias_type = bt; s.aux_type = ax;
          if (dt == T_E5M2) s.at = T_E5M2;   // E5M2 outputs need an E5M2 operand
          s.d_scale = narrow_d; s.amax = narrow_d;
          add(s, "fp8epi", std::string(en[e]) + "_d" + tn(dt) + (bt >= 0 ? "_bias" + tn(bt) : "") + (ax >= 0 ? "_aux" + tn(ax) : ""));
          if (aux && (ax == T_E4M3 || ax == T_F16 || ax == -1) && narrow_d) {
            s.aux_scale = true; s.aux_amax = true;
            add(s, "fp8epi", std::string(en[e]) + "_d" + tn(dt) + (bt >= 0 ? "_bias" + tn(bt) : "") + "_aux" + (ax >= 0 ? tn(ax) : "dflt") + "_auxscale");
          }
        }
    }
  // The backward epilogues (a transformer's training step): DRELU/DGELU read the auxiliary buffer
  // a forward pass wrote, the _BGRAD forms also write the bias gradient; BGRADA/BGRADB alone.
  {
    const int beps[] = {EP_DRELU, EP_DRELU_BGRAD, EP_DGELU, EP_DGELU_BGRAD, EP_BGRADA, EP_BGRADB};
    const char* bn[] = {"drelu", "drelu_bgrad", "dgelu", "dgelu_bgrad", "bgrada", "bgradb"};
    for (int dt : {T_BF16, T_F16, T_F32, T_E4M3, T_E5M2})
      for (int ta = 0; ta < 2; ++ta)
        for (int tb = 0; tb < 2; ++tb)
          for (int b = 0; b < 6; ++b) {
            if ((ta != 1 || tb != 0) && b < 4 && dt != T_BF16) continue;
            const bool narrow_d = dt == T_E4M3 || dt == T_E5M2;
            Spec s; s.ta = ta; s.tb = tb; s.dt = dt; s.ct = narrow_d ? T_BF16 : dt; s.epilogue = beps[b];
            if (dt == T_E5M2) s.at = T_E5M2;
            s.d_scale = narrow_d; s.amax = narrow_d;
            add(s, "fp8bwd", std::string(bn[b]) + "_" + (ta ? "T" : "N") + (tb ? "T" : "N") + "_d" + tn(dt));
          }
  }
  // Row-major layouts and padded leading dimensions.
  for (int mask = 0; mask < 16; ++mask) {
    Spec s; s.oa = mask & 1; s.ob = (mask >> 1) & 1; s.oc = (mask >> 2) & 1; s.od = (mask >> 3) & 1;
    s.ct = s.dt = T_BF16;
    add(s, "fp8order", std::string("TN_") + (s.oa ? "r" : "c") + (s.ob ? "r" : "c") + (s.oc ? "r" : "c") + (s.od ? "r" : "c"));
    s.ta = 0; s.tb = 1; add(s, "fp8order", std::string("NT_") + (s.oa ? "r" : "c") + (s.ob ? "r" : "c") + (s.oc ? "r" : "c") + (s.od ? "r" : "c"));
  }
  {
    Spec s; s.pad_a = 16; add(s, "fp8ld", "pad_a16");
    s = Spec(); s.pad_b = 16; add(s, "fp8ld", "pad_b16");
    s = Spec(); s.pad_c = 8; s.pad_d = 8; add(s, "fp8ld", "pad_cd8");
    s = Spec(); s.pad_d = 8; add(s, "fp8ld", "pad_d8");
    s = Spec(); s.pad_d = 1; add(s, "fp8ld", "pad_d1");
    s = Spec(); s.pad_a = 1; add(s, "fp8ld", "pad_a1");
    s = Spec(); s.pad_a = 16; s.pad_b = 32; s.pad_c = 8; s.pad_d = 24; s.dt = T_E4M3; s.d_scale = true; s.amax = true; add(s, "fp8ld", "pad_all_e4m3");
    s = Spec(); s.pad_a = 16; s.batch = 2; add(s, "fp8ld", "pad_a16_b2");
  }
  // The aux scale and amax, with a non-FP8 D and with an FP8 aux.
  for (int dt : {T_BF16, T_F16}) {
    Spec s; s.dt = dt; s.ct = dt; s.epilogue = EP_GELU_AUX; s.aux_scale = true; s.aux_amax = true; s.amax = true;
    add(s, "fp8aux", "gelu_aux_scale_amax_d" + tn(dt));
    s.aux_type = T_E4M3; add(s, "fp8aux", "gelu_aux_e4m3_scale_amax_d" + tn(dt));
    s.aux_type = T_E5M2; add(s, "fp8aux", "gelu_aux_e5m2_scale_amax_d" + tn(dt));
    s.epilogue = EP_GELU_AUX_BIAS; s.aux_type = -1; add(s, "fp8aux", "gelu_aux_bias_scale_amax_d" + tn(dt));
    s.epilogue = EP_RELU_AUX; s.aux_type = -1; add(s, "fp8aux", "relu_aux_scale_amax_d" + tn(dt));
  }
}

// The block-scaled modes.
void block_cases() {
  struct Sh { int m, n, k; const char* tag; };
  const Sh shapes[] = {{128, 128, 256, "128x128x256"}, {64, 64, 64, "64x64x64"}, {32, 32, 32, "32x32x32"},
                       {160, 96, 192, "160x96x192"}, {128, 128, 96, "128x128x96"}, {192, 64, 512, "192x64x512"},
                       {256, 256, 1024, "256x256x1024"}, {128, 128, 640, "128x128x640"}, {16, 16, 64, "16x16x64"}};
  // MXFP8: FP8 operands, UE8M0 per 32.
  const int ab[4][2] = {{T_E4M3, T_E4M3}, {T_E4M3, T_E5M2}, {T_E5M2, T_E4M3}, {T_E5M2, T_E5M2}};
  for (int ta = 0; ta < 2; ++ta)
    for (int tb = 0; tb < 2; ++tb)
      for (auto& x : ab)
        for (int dt : {T_BF16, T_F16, T_F32}) {
          Spec s; s.ta = ta; s.tb = tb; s.at = x[0]; s.bt = x[1]; s.dt = dt; s.ct = dt; s.am = M_VEC32; s.bm = M_VEC32;
          s.m = 128; s.n = 128; s.k = 256;
          add(s, "mx8", op(s) + "_" + tn(x[0]) + tn(x[1]) + "_d" + tn(dt));
        }
  for (auto& sh : shapes) {
    Spec s; s.am = M_VEC32; s.bm = M_VEC32; s.m = sh.m; s.n = sh.n; s.k = sh.k;
    add(s, "mx8shape", std::string("TN_") + sh.tag);
    s.dt = T_F32; s.ct = T_F32; add(s, "mx8shape", std::string("TN_f32_") + sh.tag);
  }
  // MXFP8 outputs: quantized with D_OUT scales (and a plain E4M3 D with a scalar D scale).
  for (int dt : {T_E4M3, T_E5M2}) {
    for (auto& sh : shapes) {
      Spec s; s.am = M_VEC32; s.bm = M_VEC32; s.m = sh.m; s.n = sh.n; s.k = sh.k; s.dt = dt; s.ct = T_BF16; s.dom = M_VEC32;
      add(s, "mx8out", "dout_v32_d" + tn(dt) + "_" + sh.tag);
    }
    Spec s; s.am = M_VEC32; s.bm = M_VEC32; s.m = 128; s.n = 128; s.k = 256; s.dt = dt; s.ct = T_BF16; s.d_scale = true; s.amax = true;
    add(s, "mx8out", "scalar_dscale_d" + tn(dt));
    s.d_scale = false; s.amax = false; add(s, "mx8out", "no_dscale_d" + tn(dt));
  }
  // Mixed scale modes for A and B.
  const int modes[] = {M_SCALAR, M_VEC32, M_VEC16, M_OUTER, M_VEC128, M_BLK128, M_BATCH, M_MN_K4_128, M_MN_K4_32};
  for (int am : modes)
    for (int bm : modes) {
      Spec s; s.am = am; s.bm = bm; s.m = 128; s.n = 128; s.k = 256;
      add(s, "scalemix", mn(am) + "_" + mn(bm) + "_e4m3");
    }
  // NVFP4: E2M1 operands with UE4M3 per 16; and the MX variant (UE8M0 per 32).
  for (int ta = 0; ta < 2; ++ta)
    for (int tb = 0; tb < 2; ++tb)
      for (int mode : {M_VEC16, M_VEC32})
        for (int dt : {T_BF16, T_F16, T_F32}) {
          Spec s; s.ta = ta; s.tb = tb; s.at = T_FP4; s.bt = T_FP4; s.dt = dt; s.ct = dt; s.am = mode; s.bm = mode;
          s.m = 128; s.n = 128; s.k = 256;
          add(s, "fp4", op(s) + "_" + mn(mode) + "_d" + tn(dt));
        }
  for (auto& sh : shapes) {
    Spec s; s.at = T_FP4; s.bt = T_FP4; s.am = M_VEC16; s.bm = M_VEC16; s.m = sh.m; s.n = sh.n; s.k = sh.k;
    add(s, "fp4shape", std::string("TN_") + sh.tag);
    s.dt = T_F32; s.ct = T_F32; add(s, "fp4shape", std::string("TN_f32_") + sh.tag);
    s.dt = T_BF16; s.ct = T_BF16; s.dt = T_FP4; s.ct = T_BF16; s.dom = M_VEC16; s.d_scale = true;
    add(s, "fp4shape", std::string("TN_fp4out_") + sh.tag);
  }
  for (int dt : {T_E4M3, T_E5M2, T_FP4}) {
    Spec s; s.at = T_FP4; s.bt = T_FP4; s.am = M_VEC16; s.bm = M_VEC16; s.m = 128; s.n = 128; s.k = 256; s.dt = dt; s.ct = T_BF16;
    s.d_scale = true; s.dom = dt == T_FP4 ? M_VEC16 : M_SCALAR; s.amax = dt != T_FP4;
    add(s, "fp4out", "d" + tn(dt) + "_dscale");
    s.d_scale = false; add(s, "fp4out", "d" + tn(dt) + "_nodscale");
    s.dom = M_VEC32; add(s, "fp4out", "d" + tn(dt) + "_dout_v32");
    s.dom = M_VEC16; s.d_scale = true; add(s, "fp4out", "d" + tn(dt) + "_dout_v16");
  }
  // FP4 with FP8 (mixed), and scale/type mismatches.
  { Spec s; s.at = T_FP4; s.bt = T_E4M3; s.am = M_VEC16; s.bm = M_VEC16; s.m = 128; s.n = 128; s.k = 256; add(s, "fp4mix", "fp4_e4m3_v16"); }
  { Spec s; s.at = T_E4M3; s.bt = T_FP4; s.am = M_VEC32; s.bm = M_VEC32; s.m = 128; s.n = 128; s.k = 256; add(s, "fp4mix", "e4m3_fp4_v32"); }
  { Spec s; s.at = T_E4M3; s.bt = T_E4M3; s.am = M_VEC16; s.bm = M_VEC16; s.m = 128; s.n = 128; s.k = 256; add(s, "fp4mix", "e4m3_v16"); }
  { Spec s; s.at = T_FP4; s.bt = T_FP4; s.m = 128; s.n = 128; s.k = 256; add(s, "fp4mix", "fp4_noscale_mode"); }
  { Spec s; s.at = T_FP4; s.bt = T_FP4; s.am = M_VEC16; s.bm = M_VEC16; s.a_scale = false; s.m = 128; s.n = 128; s.k = 256; add(s, "fp4mix", "fp4_no_a_scale_ptr"); }
  { Spec s; s.at = T_E4M3; s.bt = T_E4M3; s.am = M_VEC32; s.bm = M_VEC32; s.a_scale = false; s.b_scale = false; s.m = 128; s.n = 128; s.k = 256; add(s, "fp4mix", "e4m3_v32_no_ptrs"); }
  // The 128-element and 128x128 FP32 scales.
  for (int ta = 0; ta < 2; ++ta)
    for (int tb = 0; tb < 2; ++tb)
      for (int am : {M_VEC128, M_BLK128})
        for (int bm : {M_VEC128, M_BLK128})
          for (int dt : {T_BF16, T_F32}) {
            Spec s; s.ta = ta; s.tb = tb; s.am = am; s.bm = bm; s.dt = dt; s.ct = dt; s.m = 128; s.n = 128; s.k = 256;
            add(s, "hop", op(s) + "_" + mn(am) + "_" + mn(bm) + "_d" + tn(dt));
          }
  for (auto& sh : shapes) {
    Spec s; s.am = M_VEC128; s.bm = M_BLK128; s.m = sh.m; s.n = sh.n; s.k = sh.k;
    add(s, "hopshape", std::string("TN_v128_b128_") + sh.tag);
    s.am = M_VEC128; s.bm = M_VEC128; add(s, "hopshape", std::string("TN_v128_v128_") + sh.tag);
    s.am = M_BLK128; s.bm = M_VEC128; add(s, "hopshape", std::string("TN_b128_v128_") + sh.tag);
  }
  for (int dt : {T_E4M3, T_FP4}) {
    Spec s; s.am = M_VEC128; s.bm = M_BLK128; s.m = 128; s.n = 128; s.k = 256; s.dt = dt; s.d_scale = true; s.amax = true;
    add(s, "hopout", "d" + tn(dt));
  }
  // The MN x K4 modes (experimental, newer than CUDA 13.2's cuBLAS).
  for (int ta = 0; ta < 2; ++ta)
    for (int tb = 0; tb < 2; ++tb)
      for (int mode : {M_MN_K4_128, M_MN_K4_32})
        for (int dt : {T_BF16, T_F32}) {
          Spec s; s.ta = ta; s.tb = tb; s.am = mode; s.bm = mode; s.dt = dt; s.ct = dt; s.m = 128; s.n = 128; s.k = 512;
          add(s, "mnk4", op(s) + "_" + mn(mode) + "_d" + tn(dt));
        }
  for (auto& sh : shapes)
    for (int mode : {M_MN_K4_128, M_MN_K4_32}) {
      Spec s; s.am = mode; s.bm = mode; s.m = sh.m; s.n = sh.n; s.k = sh.k;
      add(s, "mnk4shape", std::string("TN_") + mn(mode) + "_" + sh.tag);
    }
  // Per-batch block scales: batch 2 and 3, scale tensors one after another.
  for (int b : {2, 3}) {
    { Spec s; s.batch = b; s.am = M_VEC32; s.bm = M_VEC32; s.m = 128; s.n = 128; s.k = 256; add(s, "bsbatch", "v32_b" + std::to_string(b)); }
    { Spec s; s.batch = b; s.am = M_VEC32; s.bm = M_VEC32; s.m = 160; s.n = 96; s.k = 192; add(s, "bsbatch", "v32_odd_b" + std::to_string(b)); }
    { Spec s; s.batch = b; s.at = T_FP4; s.bt = T_FP4; s.am = M_VEC16; s.bm = M_VEC16; s.m = 128; s.n = 128; s.k = 256; add(s, "bsbatch", "v16_b" + std::to_string(b)); }
    { Spec s; s.batch = b; s.am = M_VEC128; s.bm = M_BLK128; s.m = 128; s.n = 128; s.k = 256; add(s, "bsbatch", "hop_b" + std::to_string(b)); }
    { Spec s; s.batch = b; s.am = M_MN_K4_32; s.bm = M_MN_K4_32; s.m = 128; s.n = 128; s.k = 256; add(s, "bsbatch", "mnk4_b" + std::to_string(b)); }
    { Spec s; s.batch = b; s.am = M_VEC32; s.bm = M_VEC32; s.m = 128; s.n = 128; s.k = 256; s.dt = T_E4M3; s.dom = M_VEC32; add(s, "bsbatch", "v32_dout_b" + std::to_string(b)); }
  }
  // Epilogues and C with block scales.
  const int eps[] = {EP_BIAS, EP_RELU, EP_GELU, EP_GELU_AUX, EP_RELU_AUX_BIAS};
  const char* en[] = {"bias", "relu", "gelu", "gelu_aux", "relu_aux_bias"};
  for (int e = 0; e < 5; ++e) {
    Spec s; s.am = M_VEC32; s.bm = M_VEC32; s.m = 128; s.n = 128; s.k = 256; s.epilogue = eps[e];
    add(s, "bsepi", std::string("mx8_") + en[e]);
    Spec f; f.at = T_FP4; f.bt = T_FP4; f.am = M_VEC16; f.bm = M_VEC16; f.m = 128; f.n = 128; f.k = 256; f.epilogue = eps[e];
    add(f, "bsepi", std::string("nv4_") + en[e]);
  }
  for (float beta : {1.f, 0.5f}) {
    Spec s; s.am = M_VEC32; s.bm = M_VEC32; s.m = 128; s.n = 128; s.k = 256; s.beta = beta; s.ct = T_BF16;
    add(s, "bsbeta", "mx8_c_bf16_beta" + std::to_string((int)(beta * 2)));
    s.ct = T_E4M3; s.cm = M_VEC32; s.c_scale = true; add(s, "bsbeta", "mx8_c_e4m3_v32_beta" + std::to_string((int)(beta * 2)));
    s.ct = T_E4M3; s.cm = M_SCALAR; add(s, "bsbeta", "mx8_c_e4m3_scalar_beta" + std::to_string((int)(beta * 2)));
    Spec f; f.at = T_FP4; f.bt = T_FP4; f.am = M_VEC16; f.bm = M_VEC16; f.m = 128; f.n = 128; f.k = 256; f.beta = beta; f.ct = T_BF16;
    add(f, "bsbeta", "nv4_c_bf16_beta" + std::to_string((int)(beta * 2)));
  }
  // The scale and compute types for the block-scaled modes.
  { Spec s; s.am = M_VEC32; s.bm = M_VEC32; s.m = 128; s.n = 128; s.k = 256; s.compute = CUBLAS_COMPUTE_32F_FAST_TF32; add(s, "bscompute", "fast_tf32"); }
  { Spec s; s.am = M_VEC32; s.bm = M_VEC32; s.m = 128; s.n = 128; s.k = 256; s.stype = T_F16; add(s, "bscompute", "scale_f16"); }
  { Spec s; s.am = M_VEC32; s.bm = M_VEC32; s.m = 128; s.n = 128; s.k = 256; s.alpha = 0.5f; add(s, "bscompute", "alpha_half"); }
  { Spec s; s.am = M_VEC32; s.bm = M_VEC32; s.m = 128; s.n = 128; s.k = 256; s.pointer_mode = 1; s.alpha = 2.f; add(s, "bscompute", "device_alpha"); }
}
}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> groups;
  bool list = false, specs = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--group") && i + 1 < argc) groups.push_back(argv[++i]);
    else if (!std::strcmp(argv[i], "--list")) list = true;
    else if (!std::strcmp(argv[i], "--specs")) specs = true;
  }
  fp8_cases();
  fp8_epilogues();
  block_cases();
  if (list || specs) {
    for (auto& s : g_specs) std::printf("%s\n", specs ? spec_str(s).c_str() : s.name.c_str());
    return 0;
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) { std::printf("FAIL: no device\n"); return 1; }
  std::printf("# device %s sm_%d%d\n", prop.name, prop.major, prop.minor);
  std::printf("# cublasLt %zu\n", cublasLtGetVersion());
  if (cublasLtCreate(&g_lt)) { std::printf("FAIL: cublasLtCreate\n"); return 1; }
  cudaMalloc(&g_ws, kWs);
  int ran = 0;
  for (auto& s : g_specs) {
    bool pick = groups.empty();
    for (auto& g : groups) pick = pick || s.name.compare(0, g.size(), g) == 0;
    if (!pick) continue;
    run(s);
    ++ran;
  }
  cudaFree(g_ws);
  cublasLtDestroy(g_lt);
  std::printf("# %d cases\nPASS\n", ran);
  return 0;
}
