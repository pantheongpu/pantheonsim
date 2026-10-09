// cuSPARSELt's narrow-precision structured-sparse matmuls (FP8, FP4 and the
// scale modes), swept over descriptors and printed one line per case, so the
// same program can run against NVIDIA's libcusparseLt on a card and against
// VirtualGPU's and the two transcripts compared (nvidia/tests/e2e/run_lowprec.sh).
// Each line is the status of every call of the usual sequence (descriptors,
// matmul descriptor, attributes, algorithm, plan, prune, compress, matmul) and
// FNV-1a hashes of the compressed operand and of D.
//
//   lowprec_sparselt [--group <prefix>]... [--list]
//
// The data are small exact values, so every product and sum is exact and the
// order of accumulation does not matter. Lines that start with '#' name the
// machine and are not compared. PROBE_DUMP=<dir> dumps each case's data.
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../../include/vgpu_cusparselt.h"
#include "lowprec_common.h"

using namespace lp;

namespace {
constexpr cudaDataType E4M3 = (cudaDataType)T_E4M3, E5M2 = (cudaDataType)T_E5M2, E2M1 = (cudaDataType)T_FP4;
const cusparseOperation_t N = CUSPARSE_OPERATION_NON_TRANSPOSE, T = CUSPARSE_OPERATION_TRANSPOSE;
const cusparseOrder_t ROW = CUSPARSE_ORDER_ROW, COL = CUSPARSE_ORDER_COL;
// Scale modes by value (cusparseLtMatmulMatrixScale_t).
constexpr int S_NONE = 0, S_SCALAR = 1, S_V32 = 2, S_V64 = 3;

struct Spec {
  std::string name;
  int ab = CUDA_R_16F, cd = CUDA_R_16F, compute = CUSPARSE_COMPUTE_32F;
  cusparseOperation_t opA = N, opB = T;   // row-major, both contiguous along K: the layout the 8-bit types take
  cusparseOrder_t oA = ROW, oB = ROW, oC = ROW;
  int m = 64, n = 64, k = 64, batches = 1;
  bool sparseB = false;
  int am = -1, bm = -1, cm = -1, dm = -1, dom = -1;     // scale modes (-1: not set)
  bool a_ptr = true, b_ptr = true, c_ptr = false, d_ptr = false, dout_ptr = false;
  int relu = 0, gelu = 0;
  float gelu_scale = 1.f;
  float alpha = 1.f, beta = 0.f;
  bool bias = false;
  bool alpha_vec = false;
  bool search = false;
};

std::vector<Spec> g_specs;
void add(Spec s, const std::string& group, const std::string& tag) {
  s.name = group + "/" + tag;
  g_specs.push_back(s);
}
std::string tn(int t) {
  switch (t) {
    case E4M3: return "e4m3"; case E5M2: return "e5m2"; case E2M1: return "e2m1"; case CUDA_R_16BF: return "bf16";
    case CUDA_R_16F: return "f16"; case CUDA_R_32F: return "f32"; case CUDA_R_8I: return "i8"; case CUDA_R_32I: return "i32";
    default: return std::to_string(t);
  }
}
std::string cn(int c) { return c == CUSPARSE_COMPUTE_32F ? "c32f" : c == CUSPARSE_COMPUTE_16F ? "c16f" : "c32i"; }
bool narrow(int t) { return t == E4M3 || t == E5M2 || t == E2M1; }

cusparseLtHandle_t g_h;

std::string num(int v) { return std::to_string(v); }

void run(const Spec& s) {
  std::fprintf(stderr, "RUN %s\n", s.name.c_str());
  std::string line = s.name;
  Dev dA, dB, dC, dD, dSA, dSB, dSC, dSD, dSO, dBias, dAv, dBv;
  const int64_t m = s.m, n = s.n, k = s.k;
  const int64_t ar = s.opA == N ? m : k, ac = s.opA == N ? k : m, br = s.opB == N ? k : n, bc = s.opB == N ? n : k;
  const int64_t lda = s.oA == ROW ? ac : ar, ldb = s.oB == ROW ? bc : br, ldc = s.oC == ROW ? n : m;
  const int64_t rowsA = s.oA == ROW ? ar : ac, rowsB = s.oB == ROW ? br : bc, rowsC = s.oC == ROW ? m : n;
  cusparseLtMatDescriptor_t A, B, C;
  const int sA = s.sparseB ? cusparseLtDenseDescriptorInit(&g_h, &A, ar, ac, lda, 16, (cudaDataType)s.ab, s.oA)
                           : cusparseLtStructuredDescriptorInit(&g_h, &A, ar, ac, lda, 16, (cudaDataType)s.ab, s.oA,
                                                                CUSPARSELT_SPARSITY_50_PERCENT);
  const int sB = s.sparseB ? cusparseLtStructuredDescriptorInit(&g_h, &B, br, bc, ldb, 16, (cudaDataType)s.ab, s.oB,
                                                                CUSPARSELT_SPARSITY_50_PERCENT)
                           : cusparseLtDenseDescriptorInit(&g_h, &B, br, bc, ldb, 16, (cudaDataType)s.ab, s.oB);
  const int sC = cusparseLtDenseDescriptorInit(&g_h, &C, m, n, ldc, 16, (cudaDataType)s.cd, s.oC);
  if (s.batches > 1)
    for (cusparseLtMatDescriptor_t* d : {&A, &B, &C}) {
      cusparseLtMatDescSetAttribute(&g_h, d, CUSPARSELT_MAT_NUM_BATCHES, &s.batches, 4);
    }
  line += " desc=" + num(sA) + "," + num(sB) + "," + num(sC);
  if (sA || sB || sC) { std::printf("%s\n", line.c_str()); std::fflush(stdout); return; }
  cusparseLtMatmulDescriptor_t md;
  const int smd = cusparseLtMatmulDescriptorInit(&g_h, &md, s.opA, s.opB, &A, &B, &C, &C, (cusparseComputeType)s.compute);
  line += " md=" + num(smd);
  if (smd) { std::printf("%s\n", line.c_str()); std::fflush(stdout); return; }

  std::string bad;
  auto note = [&](const char* what, int st) { if (st && bad.empty()) bad = std::string(what) + ":" + num(st); };
  const int nb = s.batches;
  // Scale tensors.
  auto ptr_attr = [&](int attr, Dev& d, const void* p) { (void)d; note("ptr", cusparseLtMatmulDescSetAttribute(&g_h, &md, (cusparseLtMatmulDescAttribute_t)attr, &p, sizeof p)); };
  auto mode_attr = [&](int attr, int mode) { note("mode", cusparseLtMatmulDescSetAttribute(&g_h, &md, (cusparseLtMatmulDescAttribute_t)attr, &mode, 4)); };
  auto scale_buf = [&](Dev& d, int mode, uint32_t salt) {
    std::vector<uint8_t> b(mode == S_SCALAR || mode < 0 ? 64 : 256 * 64 * 4, 0);
    static const float f[4] = {0.5f, 1.f, 2.f, 0.25f};
    for (size_t i = 0; i < b.size(); ++i) {
      const uint32_t r = mix((uint32_t)i * 40503u + salt);
      if (mode == S_V32) { static const uint8_t t[4] = {0x30, 0x38, 0x40, 0x28}; b[i] = t[r & 3]; }
      else if (mode == S_V64) { static const uint8_t t[4] = {126, 127, 128, 125}; b[i] = t[r & 3]; }
      else if (i % 4 == 0) { float v = mode == S_SCALAR ? (salt == 21 ? 2.f : salt == 22 ? 0.25f : salt == 24 ? 4.f : 1.f) : f[r & 3]; std::memcpy(&b[i], &v, 4); }
    }
    d.put(b);
  };
  if (s.am >= 0) mode_attr(CUSPARSELT_MATMUL_A_SCALE_MODE, s.am);
  if (s.bm >= 0) mode_attr(CUSPARSELT_MATMUL_B_SCALE_MODE, s.bm);
  if (s.cm >= 0) mode_attr(CUSPARSELT_MATMUL_C_SCALE_MODE, s.cm);
  if (s.dm >= 0) mode_attr(CUSPARSELT_MATMUL_D_SCALE_MODE, s.dm);
  if (s.dom >= 0) mode_attr(CUSPARSELT_MATMUL_D_OUT_SCALE_MODE, s.dom);
  if (s.a_ptr) { scale_buf(dSA, s.am < 0 ? S_SCALAR : s.am, 21); ptr_attr(CUSPARSELT_MATMUL_A_SCALE_POINTER, dSA, dSA.p); }
  if (s.b_ptr) { scale_buf(dSB, s.bm < 0 ? S_SCALAR : s.bm, 22); ptr_attr(CUSPARSELT_MATMUL_B_SCALE_POINTER, dSB, dSB.p); }
  if (s.c_ptr) { scale_buf(dSC, s.cm < 0 ? S_SCALAR : s.cm, 23); ptr_attr(CUSPARSELT_MATMUL_C_SCALE_POINTER, dSC, dSC.p); }
  if (s.d_ptr) { scale_buf(dSD, s.dm < 0 ? S_SCALAR : s.dm, 24); ptr_attr(CUSPARSELT_MATMUL_D_SCALE_POINTER, dSD, dSD.p); }
  if (s.dout_ptr) { dSO.fill(256 * 64 * 4, 0xEE); ptr_attr(CUSPARSELT_MATMUL_D_OUT_SCALE_POINTER, dSO, dSO.p); }
  const int one = 1;
  if (s.relu) note("relu", cusparseLtMatmulDescSetAttribute(&g_h, &md, CUSPARSELT_MATMUL_ACTIVATION_RELU, &one, 4));
  if (s.gelu) {
    note("gelu", cusparseLtMatmulDescSetAttribute(&g_h, &md, CUSPARSELT_MATMUL_ACTIVATION_GELU, &one, 4));
    note("gelus", cusparseLtMatmulDescSetAttribute(&g_h, &md, CUSPARSELT_MATMUL_ACTIVATION_GELU_SCALING, &s.gelu_scale, 4));
  }
  if (s.alpha_vec) note("av", cusparseLtMatmulDescSetAttribute(&g_h, &md, CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &one, 4));
  if (s.bias) {
    dBias.put(make_matrix(narrow(s.ab) ? s.cd : (s.ab == CUDA_R_8I ? CUDA_R_32F : s.cd), (size_t)m * nb, 5, false));
    const void* bp = dBias.p;
    note("bias", cusparseLtMatmulDescSetAttribute(&g_h, &md, CUSPARSELT_MATMUL_BIAS_POINTER, &bp, sizeof bp));
  }
  if (!bad.empty()) line += " set=" + bad;

  cusparseLtMatmulAlgSelection_t alg;
  cusparseLtMatmulPlan_t plan;
  int st = cusparseLtMatmulAlgSelectionInit(&g_h, &alg, &md, CUSPARSELT_MATMUL_ALG_DEFAULT);
  line += " alg=" + num(st);
  if (st) { std::printf("%s\n", line.c_str()); std::fflush(stdout); return; }
  st = cusparseLtMatmulPlanInit(&g_h, &plan, &md, &alg);
  line += " plan=" + num(st);
  if (st) { cusparseLtMatmulAlgSelectionDestroy(&alg); std::printf("%s\n", line.c_str()); std::fflush(stdout); return; }

  // Operands.
  const int64_t szA = rowsA * lda, szB = rowsB * ldb, szC = rowsC * ldc;
  dA.put(make_matrix(s.ab, (size_t)szA * nb, 1, false));
  dB.put(make_matrix(s.ab, (size_t)szB * nb, 2, false));
  dC.put(make_matrix(s.cd, (size_t)szC * nb, 3, false));
  dD.fill(bytes_of(s.cd, (size_t)szC * nb), 0);
  Dev& sp = s.sparseB ? dB : dA;
  Dev valid;
  valid.fill(4, 0);
  const int pr = cusparseLtSpMMAPrune(&g_h, &md, sp.p, sp.p, CUSPARSELT_PRUNE_SPMMA_STRIP, nullptr);
  const int pc = cusparseLtSpMMAPruneCheck(&g_h, &md, sp.p, (int*)valid.p, nullptr);
  line += " prune=" + num(pr) + "," + num(pc);
  size_t cs = 0, cbs = 0;
  const int zs = cusparseLtSpMMACompressedSize(&g_h, &plan, &cs, &cbs);
  Dev comp, cbuf;
  comp.fill(cs, 0);
  cbuf.fill(cbs, 0);
  const int cp = cusparseLtSpMMACompress(&g_h, &plan, sp.p, comp.p, cbuf.p, nullptr);
  line += " comp=" + num(zs) + "," + num(cp) + ":" + std::to_string(cs) + ":" + std::to_string(cbs);
  size_t wsz = 0;
  const int wst = cusparseLtMatmulGetWorkspace(&g_h, &plan, &wsz);
  Dev ws;
  ws.fill(wsz, 0);
  line += " ws=" + num(wst) + ":" + std::to_string(wsz);
  const float alpha = s.alpha, beta = s.beta;
  // With alpha-vector scaling the alpha argument is a device vector of m floats (0.5, 1, 1.5, 2 ... cycling).
  if (s.alpha_vec) {
    std::vector<uint8_t> v((size_t)m * 4 + 64, 0);
    for (int64_t i = 0; i < m; ++i) { const float f = 0.5f * (float)(1 + i % 4); std::memcpy(&v[(size_t)i * 4], &f, 4); }
    dAv.put(v);
  }
  const void* alpha_arg = s.alpha_vec ? dAv.p : (const void*)&alpha;
  const void* opA = s.sparseB ? dA.p : comp.p;
  const void* opB = s.sparseB ? comp.p : dB.p;
  int mm;
  if (s.search) mm = cusparseLtMatmulSearch(&g_h, &plan, alpha_arg, opA, opB, &beta, dC.p, dD.p, ws.p, nullptr, 0);
  else mm = cusparseLtMatmul(&g_h, &plan, alpha_arg, opA, opB, &beta, dC.p, dD.p, ws.p, nullptr, 0);
  const cudaError_t sync = cudaDeviceSynchronize();
  line += " mm=" + num(mm);
  if (sync != cudaSuccess) line += " rt=" + num((int)sync);
  // The compressed buffer's layout is NVIDIA's own and measured only for the shapes the library's
  // tests (sparselt_paths.cpp) cover, so only its size is compared; PROBE_DUMP keeps the bytes.
  if (cp == 0 && cs) {
    const auto c = comp.get();
    dump(s.name, "comp", c.data(), c.size());
  }
  if (mm == 0) {
    const auto d = dD.get();
    char buf[48];
    if (s.gelu && s.cd != CUDA_R_8I) {
      // GELU is approximated differently on each GPU (and by VirtualGPU), so it is not hashed: the same product
      // is run again without it and D must be the GELU (scaled) of that product, within a tolerance.
      cusparseLtMatmulDescriptor_t md2 = md;
      const int zero = 0;
      cusparseLtMatmulDescSetAttribute(&g_h, &md2, CUSPARSELT_MATMUL_ACTIVATION_GELU, &zero, 4);
      cusparseLtMatmulAlgSelection_t alg2;
      cusparseLtMatmulPlan_t plan2;
      std::string verdict = "na";
      const bool checkable = bad.find("gelu") == std::string::npos && (s.cd == CUDA_R_16F || s.cd == CUDA_R_16BF || s.cd == CUDA_R_32F);
      if (checkable && !cusparseLtMatmulAlgSelectionInit(&g_h, &alg2, &md2, CUSPARSELT_MATMUL_ALG_DEFAULT) &&
          !cusparseLtMatmulPlanInit(&g_h, &plan2, &md2, &alg2)) {
        Dev d2;
        d2.fill(bytes_of(s.cd, (size_t)szC * nb), 0);
        if (!cusparseLtMatmul(&g_h, &plan2, alpha_arg, opA, opB, &beta, dC.p, d2.p, ws.p, nullptr, 0) && cudaDeviceSynchronize() == cudaSuccess) {
          const auto x = d2.get();
          verdict = "ok";
          for (int64_t i = 0; i < m && verdict == "ok"; ++i)
            for (int64_t j = 0; j < n; ++j) {
              const size_t off = (size_t)(s.oC == ROW ? i * ldc + j : j * ldc + i);
              const double v = dec_elem(s.cd, x.data(), off), got = dec_elem(s.cd, d.data(), off);
              const double ref = (double)s.gelu_scale * 0.5 * v * (1 + std::tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
              const double rel = s.cd == CUDA_R_32F ? 1e-3 : 0.03;
              if (!(std::fabs(got - ref) <= rel * (std::fabs(ref) + 0.25 * std::fabs(v)) + (s.cd == CUDA_R_8I ? 1.01 : 1e-3))) { verdict = "BAD"; break; }
            }
        }
        cusparseLtMatmulPlanDestroy(&plan2);
        cusparseLtMatmulAlgSelectionDestroy(&alg2);
      }
      line += " gelu=" + verdict;
    } else {
      std::snprintf(buf, sizeof buf, " d=%016llx", (unsigned long long)fnv(d.data(), d.size()));
      line += buf;
    }
    dump(s.name, "d", d.data(), d.size());
    dump(s.name, "a", dA.get().data(), dA.n);
    dump(s.name, "b", dB.get().data(), dB.n);
    if (s.dout_ptr) {
      const auto o = dSO.get();
      std::snprintf(buf, sizeof buf, " dscale=%016llx", (unsigned long long)fnv(o.data(), o.size()));
      line += buf;
    }
  }
  std::printf("%s\n", line.c_str());
  std::fflush(stdout);
  cusparseLtMatmulPlanDestroy(&plan);
  cusparseLtMatmulAlgSelectionDestroy(&alg);
}

void cases() {
  const int abs_[] = {CUDA_R_8I, CUDA_R_16F, CUDA_R_16BF, E4M3, E5M2, E2M1};
  const int cds[] = {CUDA_R_16F, CUDA_R_16BF, CUDA_R_32F, CUDA_R_8I, CUDA_R_32I, E4M3, E5M2, E2M1};
  const int computes[] = {CUSPARSE_COMPUTE_32F, CUSPARSE_COMPUTE_16F, CUSPARSE_COMPUTE_32I};
  // The type matrix, A structured, K contiguous in both (the 8-bit rule), every transpose.
  for (int ab : abs_)
    for (int cd : cds)
      for (int c : computes)
        for (int ta = 0; ta < 2; ++ta)
          for (int tb = 0; tb < 2; ++tb) {
            Spec s; s.ab = ab; s.cd = cd; s.compute = c;
            s.opA = ta ? T : N; s.opB = tb ? T : N;
            // Row-major storage: op N for A and T for B are K contiguous.
            s.d_ptr = narrow(cd); s.dm = narrow(cd) ? S_SCALAR : -1;
            s.am = narrow(ab) ? S_SCALAR : -1; s.bm = s.am;
            s.a_ptr = s.b_ptr = narrow(ab);
            add(s, "types", std::string(ta ? "T" : "N") + (tb ? "T" : "N") + "_" + tn(ab) + "_" + tn(cd) + "_" + cn(c));
          }
  // Orders: column-major operands, for the narrow types.
  for (int ab : {E4M3, E5M2, E2M1, CUDA_R_8I, CUDA_R_16F})
    for (int oa = 0; oa < 2; ++oa)
      for (int ob = 0; ob < 2; ++ob)
        for (int oc = 0; oc < 2; ++oc)
          for (int ta = 0; ta < 2; ++ta)
            for (int tb = 0; tb < 2; ++tb) {
              Spec s; s.ab = ab; s.cd = narrow(ab) ? CUDA_R_16BF : (ab == CUDA_R_8I ? CUDA_R_32I : CUDA_R_16F);
              s.compute = ab == CUDA_R_8I ? CUSPARSE_COMPUTE_32I : CUSPARSE_COMPUTE_32F;
              s.oA = oa ? COL : ROW; s.oB = ob ? COL : ROW; s.oC = oc ? COL : ROW;
              s.opA = ta ? T : N; s.opB = tb ? T : N;
              s.am = narrow(ab) ? S_SCALAR : -1; s.bm = s.am; s.a_ptr = s.b_ptr = narrow(ab);
              add(s, "orders", tn(ab) + "_" + (oa ? "cA" : "rA") + (ob ? "cB" : "rB") + (oc ? "cC" : "rC") + "_" + (ta ? "T" : "N") + (tb ? "T" : "N"));
            }
  // Structured B.
  for (int ab : {E4M3, E2M1, CUDA_R_8I, CUDA_R_16F})
    for (int ta = 0; ta < 2; ++ta)
      for (int tb = 0; tb < 2; ++tb) {
        Spec s; s.ab = ab; s.cd = narrow(ab) ? CUDA_R_16BF : (ab == CUDA_R_8I ? CUDA_R_32I : CUDA_R_16F); s.sparseB = true;
        s.compute = ab == CUDA_R_8I ? CUSPARSE_COMPUTE_32I : CUSPARSE_COMPUTE_32F;
        s.opA = ta ? T : N; s.opB = tb ? T : N; s.am = narrow(ab) ? S_SCALAR : -1; s.bm = s.am; s.a_ptr = s.b_ptr = narrow(ab);
        add(s, "sparseB", tn(ab) + "_" + (ta ? "T" : "N") + (tb ? "T" : "N"));
      }
  // Scale pointers missing, set or in the wrong mode.
  for (int ab : {E4M3, E5M2}) {
    Spec s; s.ab = ab; s.cd = CUDA_R_16BF;
    s.a_ptr = s.b_ptr = false; add(s, "scales", tn(ab) + "_noptr_mode_unset");
    s.am = s.bm = S_NONE; add(s, "scales", tn(ab) + "_noptr_none");
    s.am = s.bm = S_SCALAR; add(s, "scales", tn(ab) + "_noptr_scalar");
    s.a_ptr = s.b_ptr = true; s.am = s.bm = -1; add(s, "scales", tn(ab) + "_ptr_mode_unset");
    s.am = s.bm = S_NONE; add(s, "scales", tn(ab) + "_ptr_none");
    s.am = s.bm = S_SCALAR; add(s, "scales", tn(ab) + "_ptr_scalar");
    s.am = S_SCALAR; s.bm = -1; add(s, "scales", tn(ab) + "_a_scalar_only");
    s.am = -1; s.bm = S_SCALAR; add(s, "scales", tn(ab) + "_b_scalar_only");
    for (int vm : {S_V32, S_V64}) {
      s.am = s.bm = vm; s.m = 128; s.n = 128; s.k = 256; add(s, "scales", tn(ab) + "_vec" + (vm == S_V32 ? "32" : "64") + "_128x128x256");
      s.m = s.n = s.k = 64; add(s, "scales", tn(ab) + "_vec" + (vm == S_V32 ? "32" : "64") + "_64");
    }
    Spec d = Spec(); d.ab = ab; d.cd = ab; d.d_ptr = true; d.dm = S_SCALAR; d.am = d.bm = S_SCALAR;
    add(d, "scales", tn(ab) + "_dscalar_d" + tn(ab));
    d.dm = -1; add(d, "scales", tn(ab) + "_dptr_modeunset_d" + tn(ab));
    d.dm = S_V32; d.dout_ptr = true; add(d, "scales", tn(ab) + "_dvec32_d" + tn(ab));
    d = Spec(); d.ab = ab; d.cd = ab; d.am = d.bm = S_SCALAR; d.d_ptr = false; d.dm = S_SCALAR; add(d, "scales", tn(ab) + "_dscalar_noptr");
    d.dm = -1; add(d, "scales", tn(ab) + "_d_noscale_at_all");
    d = Spec(); d.ab = ab; d.cd = ab; d.am = d.bm = S_SCALAR; d.dom = S_V32; d.dout_ptr = true; d.d_ptr = true; d.dm = S_SCALAR;
    add(d, "scales", tn(ab) + "_dout_v32");
    d.dom = S_V64; add(d, "scales", tn(ab) + "_dout_v64");
    // C scale.
    d = Spec(); d.ab = ab; d.cd = ab; d.am = d.bm = S_SCALAR; d.d_ptr = true; d.dm = S_SCALAR; d.c_ptr = true; d.cm = S_SCALAR; d.beta = 1.f;
    add(d, "scales", tn(ab) + "_cscalar_beta1");
  }
  for (int vm : {S_V32, S_V64}) {
    Spec s; s.ab = E2M1; s.cd = CUDA_R_16BF; s.am = s.bm = vm; s.m = 128; s.n = 128; s.k = 256;
    add(s, "scales", std::string("e2m1_vec") + (vm == S_V32 ? "32" : "64") + "_bf16");
    s.cd = E2M1; s.d_ptr = true; s.dm = S_SCALAR; s.dout_ptr = true; s.dom = vm; add(s, "scales", std::string("e2m1_vec") + (vm == S_V32 ? "32" : "64") + "_e2m1_dout");
    s.dom = vm == S_V32 ? S_V64 : S_V32; add(s, "scales", std::string("e2m1_vec") + (vm == S_V32 ? "32" : "64") + "_e2m1_dout_other");
    s.dout_ptr = false; s.dom = -1; add(s, "scales", std::string("e2m1_vec") + (vm == S_V32 ? "32" : "64") + "_e2m1_nodout");
  }
  { Spec s; s.ab = E2M1; s.cd = CUDA_R_16BF; s.am = S_V32; s.bm = S_V64; s.m = 128; s.n = 128; s.k = 256; add(s, "scales", "e2m1_mixed_a32_b64"); }
  { Spec s; s.ab = E2M1; s.cd = CUDA_R_16BF; s.am = S_SCALAR; s.bm = S_SCALAR; add(s, "scales", "e2m1_scalar"); }
  { Spec s; s.ab = E2M1; s.cd = CUDA_R_16BF; add(s, "scales", "e2m1_unset"); }
  // Non-narrow types with scale attributes set.
  for (int ab : {CUDA_R_16F, CUDA_R_8I}) {
    Spec s; s.ab = ab; s.cd = ab == CUDA_R_8I ? CUDA_R_32I : CUDA_R_16F; s.compute = ab == CUDA_R_8I ? CUSPARSE_COMPUTE_32I : CUSPARSE_COMPUTE_32F;
    s.am = S_SCALAR; s.bm = S_SCALAR; s.a_ptr = s.b_ptr = true; add(s, "scalesnarrow", tn(ab) + "_scalar");
    s.am = s.bm = S_V32; add(s, "scalesnarrow", tn(ab) + "_v32");
    s.am = s.bm = S_NONE; add(s, "scalesnarrow", tn(ab) + "_none");
    s.a_ptr = s.b_ptr = false; add(s, "scalesnarrow", tn(ab) + "_none_noptr");
  }
  // Dimensions and alignment.
  const int dims[][3] = {{16, 16, 16}, {32, 32, 32}, {8, 8, 8}, {16, 16, 32}, {16, 16, 64}, {24, 24, 24}, {48, 48, 48},
                         {17, 16, 16}, {16, 17, 16}, {16, 16, 17}, {64, 8, 64}, {128, 128, 128}, {256, 64, 128},
                         {64, 64, 128}, {64, 64, 256}, {64, 64, 512}, {32, 64, 96}, {96, 64, 64}, {128, 16, 128}};
  for (int ab : {E4M3, E2M1, CUDA_R_8I, CUDA_R_16F})
    for (auto& d : dims) {
      Spec s; s.ab = ab; s.cd = narrow(ab) ? CUDA_R_16BF : (ab == CUDA_R_8I ? CUDA_R_32I : CUDA_R_16F);
      s.compute = ab == CUDA_R_8I ? CUSPARSE_COMPUTE_32I : CUSPARSE_COMPUTE_32F;
      s.m = d[0]; s.n = d[1]; s.k = d[2]; s.am = narrow(ab) ? S_SCALAR : -1; s.bm = s.am; s.a_ptr = s.b_ptr = narrow(ab);
      add(s, "dims", tn(ab) + "_" + num(d[0]) + "x" + num(d[1]) + "x" + num(d[2]));
    }
  // Activations.
  for (int ab : {E4M3, E5M2, E2M1, CUDA_R_8I, CUDA_R_16F, CUDA_R_16BF})
    for (int cd : cds) {
      if ((ab == CUDA_R_8I) != (cd == CUDA_R_8I) && cd != CUDA_R_32I) {
        if (ab == CUDA_R_8I) continue;
      }
      for (int act = 0; act < 3; ++act) {
        Spec s; s.ab = ab; s.cd = cd; s.compute = ab == CUDA_R_8I ? CUSPARSE_COMPUTE_32I : CUSPARSE_COMPUTE_32F;
        s.relu = act == 1; s.gelu = act == 2; s.gelu_scale = act == 2 ? 0.5f : 1.f;
        s.am = narrow(ab) ? S_SCALAR : -1; s.bm = s.am; s.a_ptr = s.b_ptr = narrow(ab);
        s.d_ptr = narrow(cd); s.dm = narrow(cd) ? S_SCALAR : -1;
        add(s, "act", std::string(act == 0 ? "none" : act == 1 ? "relu" : "gelu") + "_" + tn(ab) + "_" + tn(cd));
      }
    }
  // Bias, alpha, beta, batches, the search.
  for (int ab : {E4M3, E5M2, E2M1, CUDA_R_16F, CUDA_R_8I})
    for (int cd : {CUDA_R_16F, CUDA_R_16BF, CUDA_R_32F, E4M3, CUDA_R_32I}) {
      Spec s; s.ab = ab; s.cd = cd; s.bias = true; s.compute = ab == CUDA_R_8I ? CUSPARSE_COMPUTE_32I : CUSPARSE_COMPUTE_32F;
      s.am = narrow(ab) ? S_SCALAR : -1; s.bm = s.am; s.a_ptr = s.b_ptr = narrow(ab); s.d_ptr = narrow(cd); s.dm = narrow(cd) ? S_SCALAR : -1;
      add(s, "bias", tn(ab) + "_" + tn(cd));
      s.bias = false; s.alpha = 0.5f; s.beta = 1.f; add(s, "alphabeta", tn(ab) + "_" + tn(cd) + "_a05_b1");
      s.alpha = 2.f; s.beta = 0.f; s.alpha_vec = true; add(s, "alphabeta", tn(ab) + "_" + tn(cd) + "_alphavec");
    }
  for (int ab : {E4M3, E2M1}) {
    Spec s; s.ab = ab; s.cd = CUDA_R_16BF; s.batches = 3; s.am = S_SCALAR; s.bm = S_SCALAR; add(s, "batch", tn(ab) + "_b3");
    s = Spec(); s.ab = ab; s.cd = CUDA_R_16BF; s.search = true; s.am = S_SCALAR; s.bm = S_SCALAR; add(s, "search", tn(ab));
  }
  // fp16 compute and gelu outside int8 for the old types.
  { Spec s; s.ab = CUDA_R_16F; s.cd = CUDA_R_16F; s.compute = CUSPARSE_COMPUTE_16F; add(s, "fp16", "c16f"); }
  { Spec s; s.ab = CUDA_R_16F; s.cd = CUDA_R_16F; s.compute = CUSPARSE_COMPUTE_16F; s.relu = 1; add(s, "fp16", "c16f_relu"); }
  { Spec s; s.ab = CUDA_R_16F; s.cd = CUDA_R_16F; s.compute = CUSPARSE_COMPUTE_16F; s.bias = true; add(s, "fp16", "c16f_bias"); }
  { Spec s; s.ab = CUDA_R_16F; s.cd = CUDA_R_16F; s.compute = CUSPARSE_COMPUTE_16F; s.alpha = 0.5f; s.beta = 1.f; add(s, "fp16", "c16f_a05b1"); }
}
}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> groups;
  bool list = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--group") && i + 1 < argc) groups.push_back(argv[++i]);
    else if (!std::strcmp(argv[i], "--list")) list = true;
  }
  cases();
  if (list) {
    for (auto& s : g_specs) std::printf("%s\n", s.name.c_str());
    return 0;
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) { std::printf("FAIL: no device\n"); return 1; }
  std::printf("# device %s sm_%d%d\n", prop.name, prop.major, prop.minor);
  if (cusparseLtInit(&g_h)) { std::printf("FAIL: cusparseLtInit\n"); return 1; }
  int v = 0;
  cusparseLtGetVersion(&g_h, &v);
  std::printf("# cusparseLt %d\n", v);
  int ran = 0;
  for (auto& s : g_specs) {
    bool pick = groups.empty();
    for (auto& g : groups) pick = pick || s.name.compare(0, g.size(), g) == 0;
    if (!pick) continue;
    run(s);
    ++ran;
  }
  cusparseLtDestroy(&g_h);
  std::printf("# %d cases\nPASS\n", ran);
  return 0;
}
