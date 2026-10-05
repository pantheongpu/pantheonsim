// cuBLASLt's epilogues with auxiliary data, the ones a training step's
// backward pass uses: RELU_AUX and GELU_AUX (forward, writing the ReLU bit
// mask or the GELU input), DRELU and DGELU (backward, reading them), their
// _BGRAD forms and BGRADA/BGRADB (bias gradients), in fp16, bf16, fp32 and
// fp64; and the descriptors an RTX 3060's cuBLASLt 13.0 refuses, with what it
// answers. The matmul operands are small dyadic values, so every product is
// exact; GELU and its derivative are compared with a tolerance. Every check
// passes on the card too.
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}

static double F(float v) { return v; }
static double F(double v) { return v; }
static double F(__half v) { return __half2float(v); }
static double F(__nv_bfloat16 v) { return __bfloat162float(v); }
template <class T> static T mk(double v);
template <> float mk<float>(double v) { return (float)v; }
template <> double mk<double>(double v) { return v; }
template <> __half mk<__half>(double v) { return __float2half((float)v); }
template <> __nv_bfloat16 mk<__nv_bfloat16>(double v) { return __float2bfloat16((float)v); }

template <class T> static T* up(const std::vector<T>& h, size_t extra = 256) {
  T* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(T) + extra);
  cudaMemset(d, 0, h.size() * sizeof(T) + extra);
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> static std::vector<T> down(const T* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

static cublasLtHandle_t lt;

static double gelu(double x) { return 0.5 * x * (1 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))); }
static double gelu_grad(double x) {
  const double k = 0.7978845608028654, c = 0.044715, t = std::tanh(k * (x + c * x * x * x));
  return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * k * (1 + 3 * c * x * x);
}

// One matmul: D (m x n) = alpha op(A) op(B) + beta C (C = D's buffer) with an
// epilogue; returns the heuristic's status in *heur and the matmul's.
struct Call {
  cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_DEFAULT;
  cublasOperation_t ta = CUBLAS_OP_N, tb = CUBLAS_OP_N;
  void* aux = nullptr;
  int64_t aux_ld = 0;
  int32_t aux_type = -1;
  void* bias = nullptr;
  double alpha = 1, beta = 0;
};
template <class T>
static int run(cudaDataType dt, int m, int n, int k, const std::vector<T>& A, const std::vector<T>& B, std::vector<T>* D,
               const Call& c, int* heur) {
  const bool f64 = dt == CUDA_R_64F;
  cublasLtMatmulDesc_t desc;
  cublasLtMatmulDescCreate(&desc, f64 ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_32F, f64 ? CUDA_R_64F : CUDA_R_32F);
  cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &c.ta, sizeof c.ta);
  cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &c.tb, sizeof c.tb);
  cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &c.epi, sizeof c.epi);
  if (c.aux) {
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &c.aux, sizeof c.aux);
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &c.aux_ld, sizeof c.aux_ld);
  }
  if (c.aux_type >= 0)
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, &c.aux_type, sizeof c.aux_type);
  if (c.bias) cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &c.bias, sizeof c.bias);
  const bool na = c.ta == CUBLAS_OP_N, nb = c.tb == CUBLAS_OP_N;
  cublasLtMatrixLayout_t la, lb, ld;
  cublasLtMatrixLayoutCreate(&la, dt, na ? m : k, na ? k : m, na ? m : k);
  cublasLtMatrixLayoutCreate(&lb, dt, nb ? k : n, nb ? n : k, nb ? k : n);
  cublasLtMatrixLayoutCreate(&ld, dt, m, n, m);
  T *dA = up(A), *dB = up(B), *dD = up(*D);
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  const size_t ws_bytes = 1 << 22;
  void* ws;
  cudaMalloc(&ws, ws_bytes);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof ws_bytes);
  cublasLtMatmulHeuristicResult_t hr;
  int got = 0;
  *heur = (int)cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, ld, ld, pref, 1, &hr, &got);
  const float a32 = (float)c.alpha, b32 = (float)c.beta;
  const double a64 = c.alpha, b64 = c.beta;
  const void* al = f64 ? (const void*)&a64 : (const void*)&a32;
  const void* be = f64 ? (const void*)&b64 : (const void*)&b32;
  const int st = (int)cublasLtMatmul(lt, desc, al, dA, la, dB, lb, be, dD, ld, dD, ld, got ? &hr.algo : nullptr, ws,
                                     ws_bytes, 0);
  cudaDeviceSynchronize();
  *D = down(dD, D->size());
  for (void* p : {(void*)dA, (void*)dB, (void*)dD, ws}) cudaFree(p);
  cublasLtMatmulPreferenceDestroy(pref);
  for (auto l : {la, lb, ld}) cublasLtMatrixLayoutDestroy(l);
  cublasLtMatmulDescDestroy(desc);
  return st;
}

template <class T> static void suite(const char* ty, cudaDataType dt, double tol) {
  const int n = 4, k = 3;
  char what[160];
  for (int m : {8, 5}) {
    std::vector<T> A((size_t)m * k), B((size_t)k * n), C0((size_t)m * n);
    for (int i = 0; i < m * k; ++i) A[i] = mk<T>((double)((i * 7) % 5) - 2);
    for (int i = 0; i < k * n; ++i) B[i] = mk<T>((double)((i * 3) % 4) - 1.5);
    for (int i = 0; i < m * n; ++i) C0[i] = mk<T>((double)((i * 5) % 7) - 3);
    // op(A) op(B) for each transpose pair, A stored m x k, B k x n.
    auto prod = [&](int i, int j) {
      double s = 0;
      for (int p = 0; p < k; ++p) s += F(A[(size_t)p * m + i]) * F(B[(size_t)j * k + p]);
      return s;
    };
    int heur = -1, st = -1;
    // RELU_AUX_BIAS with beta: D = relu(AB + C + bias), the mask set where the
    // input is >= 0 (zero included, as the card sets it), the
    // mask's bits past m left as they were.
    {
      std::vector<T> bias(m);
      for (int i = 0; i < m; ++i) bias[i] = mk<T>(0.5 * i - 1);
      T* dbias = up(bias);
      std::vector<uint8_t> mask(64, 0xAA);
      uint8_t* dmask = up(mask);
      std::vector<T> D = C0;
      Call c;
      c.epi = CUBLASLT_EPILOGUE_RELU_AUX_BIAS;
      c.aux = dmask;
      c.aux_ld = 128;
      c.bias = dbias;
      c.beta = 1;
      st = run(dt, m, n, k, A, B, &D, c, &heur);
      mask = down(dmask, 64);
      bool ok = st == 0 && heur == 0;
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < 8; ++i) {
          const double v = prod(i < m ? i : 0, j) + F(C0[(size_t)j * m + (i < m ? i : 0)]) + F(bias[i < m ? i : 0]);
          const bool bit = (mask[(size_t)j * 16] >> i) & 1;
          if (i < m) ok = ok && bit == (v >= 0) && F(D[(size_t)j * m + i]) == F(mk<T>(v > 0 ? v : 0));
          else ok = ok && bit == ((0xAA >> i) & 1);
        }
      for (int j = 0; j < n; ++j) ok = ok && mask[(size_t)j * 16 + 1] == 0xAA;
      if (!ok) {
        std::printf("     st %d heur %d mask:", st, heur);
        for (int j = 0; j < 4 * 16; ++j) std::printf(" %02x", mask[j]);
        std::printf("\n     D:");
        for (auto v : D) std::printf(" %g", F(v));
        std::printf("\n     want:");
        for (int j = 0; j < n; ++j)
          for (int i = 0; i < m; ++i) std::printf(" %g", prod(i, j) + F(C0[(size_t)j * m + i]) + F(bias[i]));
        std::printf("\n");
      }
      std::snprintf(what, sizeof what, "%s m %d: RELU_AUX_BIAS with beta, the mask's bits past m untouched", ty, m);
      check(ok, what);

      // DRELU reads that mask back: D = mask ? AB + beta C : 0; its _BGRAD
      // writes each row's sum.
      std::vector<T> bg((size_t)m + 4, mk<T>(-99));
      T* dbg = up(bg);
      D = C0;
      c = Call();
      c.epi = CUBLASLT_EPILOGUE_DRELU_BGRAD;
      c.aux = dmask;
      c.aux_ld = 128;
      c.bias = dbg;
      c.beta = 1;
      st = run(dt, m, n, k, A, B, &D, c, &heur);
      bg = down(dbg, bg.size());
      ok = st == 0 && heur == 0 && F(bg[m]) == -99;
      for (int i = 0; i < m; ++i) {
        double sum = 0;
        for (int j = 0; j < n; ++j) {
          const bool bit = (mask[(size_t)j * 16] >> i) & 1;
          const double v = bit ? prod(i, j) + F(C0[(size_t)j * m + i]) : 0;
          sum += v;
          ok = ok && F(D[(size_t)j * m + i]) == F(mk<T>(v));
        }
        ok = ok && F(bg[i]) == F(mk<T>(sum));
      }
      std::snprintf(what, sizeof what, "%s m %d: DRELU_BGRAD reads the mask, the bias gradient is each row's sum", ty, m);
      check(ok, what);
      cudaFree(dbias);
      cudaFree(dmask);
      cudaFree(dbg);
    }
    // GELU_AUX_BIAS: the aux is AB/2 + bias, exactly; D GELU of it.
    {
      std::vector<T> bias(m);
      for (int i = 0; i < m; ++i) bias[i] = mk<T>(0.25 * i);
      T* dbias = up(bias);
      const int64_t ld = m + 3;
      std::vector<T> aux((size_t)ld * n, mk<T>(-7));
      T* daux = up(aux);
      std::vector<T> D((size_t)m * n);
      Call c;
      c.epi = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
      c.aux = daux;
      c.aux_ld = ld;
      c.bias = dbias;
      c.alpha = 0.5;
      st = run(dt, m, n, k, A, B, &D, c, &heur);
      aux = down(daux, aux.size());
      bool exact = st == 0 && heur == 0, close = true;
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < ld; ++i) {
          if (i >= m) {
            exact = exact && F(aux[(size_t)j * ld + i]) == -7;
            continue;
          }
          const double v = 0.5 * prod(i, j) + F(bias[i]);
          exact = exact && F(aux[(size_t)j * ld + i]) == F(mk<T>(v));
          close = close && std::fabs(F(D[(size_t)j * m + i]) - gelu(v)) <= tol * (std::fabs(gelu(v)) + 0.05);
          if (std::getenv("LT_DEBUG")) std::printf("     gelu(%g) = %.9g, got %.9g\n", v, gelu(v), F(D[(size_t)j * m + i]));
        }
      std::snprintf(what, sizeof what, "%s m %d: GELU_AUX_BIAS writes GELU's input exactly, D within %g", ty, m, tol);
      check(exact && close, what);
      cudaFree(dbias);

      // DGELU_BGRAD: D = AB * GELU'(aux), each row summed.
      std::vector<T> g((size_t)ld * n);
      for (size_t i = 0; i < g.size(); ++i) g[i] = mk<T>(-2.0 + 0.125 * (double)(i % 32));
      cudaMemcpy(daux, g.data(), g.size() * sizeof(T), cudaMemcpyHostToDevice);
      std::vector<T> bg((size_t)m, mk<T>(-99));
      T* dbg = up(bg);
      c = Call();
      c.epi = CUBLASLT_EPILOGUE_DGELU_BGRAD;
      c.aux = daux;
      c.aux_ld = ld;
      c.bias = dbg;
      st = run(dt, m, n, k, A, B, &D, c, &heur);
      bg = down(dbg, bg.size());
      close = st == 0 && heur == 0;
      for (int i = 0; i < m; ++i) {
        double sum = 0;
        for (int j = 0; j < n; ++j) {
          const double v = prod(i, j) * gelu_grad(F(g[(size_t)j * ld + i]));
          sum += v;
          close = close && std::fabs(F(D[(size_t)j * m + i]) - v) <= tol * (std::fabs(v) + 0.05);
          if (std::getenv("LT_DEBUG")) std::printf("     dgelu %.9g, got %.9g\n", v, F(D[(size_t)j * m + i]));
        }
        close = close && std::fabs(F(bg[i]) - sum) <= tol * (std::fabs(sum) + 0.2);
      }
      std::snprintf(what, sizeof what, "%s m %d: DGELU_BGRAD, D = AB GELU'(aux), within %g", ty, m, tol);
      check(close, what);
      cudaFree(daux);
      cudaFree(dbg);
    }
    // BGRADA (op(A) = N) and BGRADB (op(B) = T): the sums over K, not scaled
    // by alpha; D an ordinary matmul.
    for (int which = 0; which < 2; ++which)
      for (cublasOperation_t other : {CUBLAS_OP_N, CUBLAS_OP_T}) {
        Call c;
        c.epi = which ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BGRADA;
        c.ta = which ? other : CUBLAS_OP_N;
        c.tb = which ? CUBLAS_OP_T : other;
        c.alpha = 2;
        // The same A and B values, stored for the transposes.
        std::vector<T> As = A, Bs = B;
        if (c.ta == CUBLAS_OP_T)
          for (int i = 0; i < m; ++i)
            for (int p = 0; p < k; ++p) As[(size_t)i * k + p] = A[(size_t)p * m + i];
        if (c.tb == CUBLAS_OP_T)
          for (int p = 0; p < k; ++p)
            for (int j = 0; j < n; ++j) Bs[(size_t)p * n + j] = B[(size_t)j * k + p];
        const int len = which ? n : m;
        std::vector<T> bg((size_t)len + 2, mk<T>(-99));
        T* dbg = up(bg);
        c.bias = dbg;
        std::vector<T> D((size_t)m * n);
        st = run(dt, m, n, k, As, Bs, &D, c, &heur);
        bg = down(dbg, bg.size());
        bool ok = st == 0 && heur == 0 && F(bg[len]) == -99;
        for (int i = 0; i < len; ++i) {
          double s = 0;
          for (int p = 0; p < k; ++p) s += which ? F(B[(size_t)i * k + p]) : F(A[(size_t)p * m + i]);
          ok = ok && F(bg[i]) == F(mk<T>(s));
        }
        for (int j = 0; j < n; ++j)
          for (int i = 0; i < m; ++i) ok = ok && F(D[(size_t)j * m + i]) == F(mk<T>(2 * prod(i, j)));
        std::snprintf(what, sizeof what, "%s m %d: %s, op(A) %d, op(B) %d", ty, m, which ? "BGRADB" : "BGRADA",
                      (int)c.ta, (int)c.tb);
        check(ok, what);
        cudaFree(dbg);
      }
    // What the card refuses, in the heuristic and the matmul alike.
    {
      std::vector<uint8_t> mask(64, 0);
      uint8_t* dmask = up(mask);
      T* dbuf = up(std::vector<T>(64));
      std::vector<T> D((size_t)m * n);
      struct Case {
        const char* name;
        Call c;
        int heur, matmul;
      };
      std::vector<Case> cases;
      Call c;
      c.epi = CUBLASLT_EPILOGUE_RELU_AUX;
      c.aux = dmask;
      c.aux_ld = 64;
      cases.push_back({"RELU_AUX, mask ld 64", c, 7, 7});
      c.aux_ld = 0;
      cases.push_back({"RELU_AUX, mask ld 0", c, 7, 7});
      c.aux = nullptr;
      cases.push_back({"RELU_AUX, no aux pointer", c, 0, 15});
      c = Call();
      c.epi = CUBLASLT_EPILOGUE_GELU_AUX;
      c.aux = dbuf;
      c.aux_ld = m - 1;
      cases.push_back({"GELU_AUX, ld below m", c, 7, 7});
      c.aux_ld = m;
      if (dt != CUDA_R_32F) {
        c.aux_type = CUDA_R_32F;
        cases.push_back({"GELU_AUX, an aux type that is not D's", c, 7, 7});
        c.aux_type = -1;
      }
      c.aux = nullptr;
      cases.push_back({"GELU_AUX, no aux pointer", c, 0, 15});
      c = Call();
      c.epi = CUBLASLT_EPILOGUE_DGELU;
      cases.push_back({"DGELU, no aux pointer", c, 0, 7});
      c = Call();
      c.epi = CUBLASLT_EPILOGUE_DRELU_BGRAD;
      c.aux = dmask;
      c.aux_ld = 128;
      cases.push_back({"DRELU_BGRAD, no bias pointer", c, 0, 7});
      c = Call();
      c.epi = CUBLASLT_EPILOGUE_BGRADA;
      c.ta = CUBLAS_OP_T;
      c.bias = dbuf;
      cases.push_back({"BGRADA with op(A) = T", c, 15, 15});
      c = Call();
      c.epi = CUBLASLT_EPILOGUE_BGRADB;
      c.bias = dbuf;
      cases.push_back({"BGRADB with op(B) = N", c, 15, 15});
      c.tb = CUBLAS_OP_T;
      c.bias = nullptr;
      cases.push_back({"BGRADB, no bias pointer", c, 0, 7});
      for (const auto& cs : cases) {
        std::vector<T> As(A.size()), Bs(B.size());
        st = run(dt, m, n, k, As, Bs, &D, cs.c, &heur);
        std::snprintf(what, sizeof what, "%s m %d: %s -> heuristic %d, matmul %d (got %d, %d)", ty, m, cs.name, cs.heur,
                      cs.matmul, heur, st);
        check(heur == cs.heur && st == cs.matmul, what);
      }
      cudaFree(dmask);
      cudaFree(dbuf);
    }
  }
}

// The epilogue attributes read back, and their defaults.
static void attributes() {
  cublasLtMatmulDesc_t d;
  cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  int32_t i32 = 0;
  int64_t i64 = -1;
  void* p = (void*)1;
  size_t w = 0;
  bool ok = cublasLtMatmulDescGetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE, &i32, sizeof i32, &w) == 0 &&
            i32 == CUBLASLT_EPILOGUE_DEFAULT && w == 4;
  ok = ok && cublasLtMatmulDescGetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, &i32, sizeof i32, &w) == 0 &&
       i32 == -1;
  ok = ok && cublasLtMatmulDescGetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &i64, sizeof i64, &w) == 0 &&
       i64 == 0 && w == 8;
  ok = ok && cublasLtMatmulDescGetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &p, sizeof p, &w) == 0 &&
       p == nullptr;
  check(ok, "the epilogue attributes' defaults");
  const int64_t ld = 256;
  void* q = &ok;
  const cublasLtEpilogue_t e = CUBLASLT_EPILOGUE_DRELU_BGRAD;
  cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ld, sizeof ld);
  cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &q, sizeof q);
  cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE, &e, sizeof e);
  ok = cublasLtMatmulDescGetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &i64, sizeof i64, &w) == 0 && i64 == 256;
  ok = ok && cublasLtMatmulDescGetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &p, sizeof p, &w) == 0 && p == q;
  ok = ok && cublasLtMatmulDescGetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE, &i32, sizeof i32, &w) == 0 && i32 == e;
  ok = ok && cublasLtMatmulDescGetAttribute(d, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &i32, sizeof i32, &w) != 0;
  check(ok, "the epilogue attributes read back as set; a short buffer is refused");
  cublasLtMatmulDescDestroy(d);
}

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  if (cublasLtCreate(&lt)) {
    std::printf("FAIL: cublasLtCreate\n");
    return 1;
  }
  suite<__half>("fp16", CUDA_R_16F, 4e-3);
  suite<__nv_bfloat16>("bf16", CUDA_R_16BF, 1.6e-2);
  suite<float>("fp32", CUDA_R_32F, 2e-4);   // the card's fp32 tanh is approximate
  suite<double>("fp64", CUDA_R_64F, 1e-5);
  attributes();
  cublasLtDestroy(lt);
  std::printf(failures ? "FAIL: %d cuBLASLt epilogue checks\n" : "PASS: every cuBLASLt epilogue check\n", failures);
  return failures ? 1 : 0;
}
