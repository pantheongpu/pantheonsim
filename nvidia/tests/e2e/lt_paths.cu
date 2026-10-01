// cuBLASLt as PyTorch drives it: fp16 and bf16 matmuls with a bias epilogue
// (addmm, Linear), strided batches (bmm), and torch._scaled_mm's FP8 matmuls
// with tensor-wise and row-wise scales, an FP8 output with its scale and amax.
//
// The reference is computed here on the host, and the conversions to and from
// fp8, fp16 and bf16 come from CUDA's own headers, so the shim's hand-written
// encoders are checked against NVIDIA's rather than against themselves.
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;

static void check(bool ok, const char* what, double err) {
  std::printf("%-4s %s (%.2e)\n", ok ? "ok" : "FAIL", what, err);
  if (!ok) ++failures;
}

#define CK(x)                                                                    \
  do {                                                                           \
    const int r_ = (int)(x);                                                     \
    if (r_ != 0) {                                                               \
      std::printf("FAIL %s returned %d\n", #x, r_);                             \
      ++failures;                                                                \
      return;                                                                    \
    }                                                                            \
  } while (0)

template <class T> T* upload(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T));
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> std::vector<T> download(const T* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

static double val(int i, int j, int seed) { return std::sin(0.37 * i + 0.61 * j + seed) * 1.5; }

// Host conversions, from NVIDIA's headers.
static float f(__half h) { return __half2float(h); }
static float f(__nv_bfloat16 h) { return __bfloat162float(h); }
static float f(__nv_fp8_e4m3 h) { return (float)h; }

static double max_rel(const std::vector<double>& ref, const std::vector<double>& got) {
  double err = 0, scale = 1e-30;
  for (size_t i = 0; i < ref.size(); ++i) {
    err = std::fmax(err, std::fabs(ref[i] - got[i]));
    scale = std::fmax(scale, std::fabs(ref[i]));
  }
  return err / scale;
}

static cublasLtHandle_t lt;

struct Layouts {
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr, d = nullptr;
  ~Layouts() {
    for (auto l : {a, b, c, d})
      if (l) cublasLtMatrixLayoutDestroy(l);
  }
};

// D = A^T B + bias in fp16, ReLU'd, as a Linear layer's forward: A is k x m
// (weights, transposed), B is k x n, column-major.
static void half_bias_relu() {
  const int m = 5, n = 4, k = 7;
  std::vector<__half> a((size_t)k * m), b((size_t)k * n), bias(m), d((size_t)m * n);
  for (int i = 0; i < k * m; ++i) a[i] = __float2half((float)val(i % k, i / k, 1));
  for (int i = 0; i < k * n; ++i) b[i] = __float2half((float)val(i % k, i / k, 2));
  for (int i = 0; i < m; ++i) bias[i] = __float2half((float)(0.25 * i - 0.5));
  __half *da = upload(a), *db = upload(b), *dbias = upload(bias), *dd = upload(d);
  cublasLtMatmulDesc_t desc;
  CK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t T = CUBLAS_OP_T;
  const cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_RELU_BIAS;
  CK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T));
  CK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof epi));
  CK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dbias, sizeof dbias));
  Layouts L;
  CK(cublasLtMatrixLayoutCreate(&L.a, CUDA_R_16F, k, m, k));
  CK(cublasLtMatrixLayoutCreate(&L.b, CUDA_R_16F, k, n, k));
  CK(cublasLtMatrixLayoutCreate(&L.d, CUDA_R_16F, m, n, m));
  const float alpha = 1.0f, beta = 0.0f;
  CK(cublasLtMatmul(lt, desc, &alpha, da, L.a, db, L.b, &beta, dd, L.d, dd, L.d, nullptr, nullptr, 0, 0));
  const auto out = download(dd, d.size());
  std::vector<double> ref, got;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      double s = f(bias[i]);
      for (int p = 0; p < k; ++p) s += (double)f(a[(size_t)i * k + p]) * f(b[(size_t)j * k + p]);
      ref.push_back(s > 0 ? s : 0);
      got.push_back(f(out[(size_t)j * m + i]));
    }
  const double e = max_rel(ref, got);
  check(e < 2e-3, "fp16 A^T B + bias, ReLU epilogue", e);
  cublasLtMatmulDescDestroy(desc);
  for (void* p : {(void*)da, (void*)db, (void*)dbias, (void*)dd}) cudaFree(p);
}

// A strided batch in bf16, beta != 0 reading C, row-major layouts.
static void bf16_batched_rowmajor() {
  const int m = 3, n = 4, k = 5, batch = 3;
  std::vector<__nv_bfloat16> a((size_t)m * k * batch), b((size_t)k * n * batch), c((size_t)m * n * batch),
      d((size_t)m * n * batch);
  for (size_t i = 0; i < a.size(); ++i) a[i] = __float2bfloat16((float)val((int)i, 1, 3));
  for (size_t i = 0; i < b.size(); ++i) b[i] = __float2bfloat16((float)val((int)i, 2, 4));
  for (size_t i = 0; i < c.size(); ++i) c[i] = __float2bfloat16((float)val((int)i, 3, 5));
  auto *da = upload(a), *db = upload(b), *dc = upload(c), *dd = upload(d);
  cublasLtMatmulDesc_t desc;
  CK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  Layouts L;
  const int32_t row = CUBLASLT_ORDER_ROW, count = batch;
  auto make = [&](cublasLtMatrixLayout_t* l, int r, int cols) {
    const int64_t stride = (int64_t)r * cols;
    cublasLtMatrixLayoutCreate(l, CUDA_R_16BF, r, cols, cols);
    cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof row);
    cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof count);
    cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof stride);
  };
  make(&L.a, m, k);
  make(&L.b, k, n);
  make(&L.c, m, n);
  make(&L.d, m, n);
  int32_t got_order = -1;
  size_t written = 0;
  CK(cublasLtMatrixLayoutGetAttribute(L.a, CUBLASLT_MATRIX_LAYOUT_ORDER, &got_order, sizeof got_order, &written));
  check(got_order == CUBLASLT_ORDER_ROW && written == sizeof got_order,
        "a layout reports the attribute asked for, not its batch count", 0);
  const float alpha = 0.5f, beta = -1.0f;
  CK(cublasLtMatmul(lt, desc, &alpha, da, L.a, db, L.b, &beta, dc, L.c, dd, L.d, nullptr, nullptr, 0, 0));
  const auto out = download(dd, d.size());
  std::vector<double> ref, got;
  for (int s = 0; s < batch; ++s)
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        double acc = 0;
        for (int p = 0; p < k; ++p)
          acc += (double)f(a[(size_t)s * m * k + (size_t)i * k + p]) * f(b[(size_t)s * k * n + (size_t)p * n + j]);
        ref.push_back(alpha * acc + beta * f(c[(size_t)s * m * n + (size_t)i * n + j]));
        got.push_back(f(out[(size_t)s * m * n + (size_t)i * n + j]));
      }
  const double e = max_rel(ref, got);
  check(e < 1e-2, "bf16 strided batch, row-major, alpha A B + beta C", e);
  cublasLtMatmulDescDestroy(desc);
  for (void* p : {(void*)da, (void*)db, (void*)dc, (void*)dd}) cudaFree(p);
}

// torch._scaled_mm: E4M3 A^T B with fp32 scales on the device, bf16 out; then
// row-wise (outer-vector) scales; then an E4M3 output with its scale and amax.
static void fp8_scaled() {
  const int m = 4, n = 6, k = 8;
  std::vector<__nv_fp8_e4m3> a((size_t)k * m), b((size_t)k * n);
  for (int i = 0; i < k * m; ++i) a[i] = __nv_fp8_e4m3((float)val(i % k, i / k, 6) * 3);
  for (int i = 0; i < k * n; ++i) b[i] = __nv_fp8_e4m3((float)val(i % k, i / k, 7) * 3);
  auto *da = upload(a), *db = upload(b);
  const std::vector<float> sa_t = {0.5f}, sb_t = {2.5f};
  std::vector<float> sa_r(m), sb_r(n);
  for (int i = 0; i < m; ++i) sa_r[i] = 0.25f * (i + 1);
  for (int j = 0; j < n; ++j) sb_r[j] = 1.0f + 0.5f * j;
  float *dsa = upload(sa_t), *dsb = upload(sb_t), *dsar = upload(sa_r), *dsbr = upload(sb_r);
  auto dot = [&](int i, int j) {
    double s = 0;
    for (int p = 0; p < k; ++p) s += (double)f(a[(size_t)i * k + p]) * f(b[(size_t)j * k + p]);
    return s;
  };

  cublasLtMatmulDesc_t desc;
  CK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t T = CUBLAS_OP_T;
  const int8_t fast = 1;
  CK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T));
  CK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast, sizeof fast));
  CK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dsa, sizeof dsa));
  CK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dsb, sizeof dsb));
  Layouts L;
  CK(cublasLtMatrixLayoutCreate(&L.a, CUDA_R_8F_E4M3, k, m, k));
  CK(cublasLtMatrixLayoutCreate(&L.b, CUDA_R_8F_E4M3, k, n, k));
  CK(cublasLtMatrixLayoutCreate(&L.c, CUDA_R_16BF, m, n, m));
  CK(cublasLtMatrixLayoutCreate(&L.d, CUDA_R_16BF, m, n, m));
  auto* dd = upload(std::vector<__nv_bfloat16>((size_t)m * n));
  const float alpha = 1.0f, beta = 0.0f;
  cublasLtMatmulPreference_t pref;
  CK(cublasLtMatmulPreferenceCreate(&pref));
  cublasLtMatmulHeuristicResult_t heur;
  int found = 0;
  CK(cublasLtMatmulAlgoGetHeuristic(lt, desc, L.a, L.b, L.c, L.d, pref, 1, &heur, &found));
  CK(cublasLtMatmul(lt, desc, &alpha, da, L.a, db, L.b, &beta, dd, L.c, dd, L.d, &heur.algo, nullptr, 0, 0));
  auto out = download(dd, (size_t)m * n);
  std::vector<double> ref, got;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      ref.push_back(sa_t[0] * sb_t[0] * dot(i, j));
      got.push_back(f(out[(size_t)j * m + i]));
    }
  double e = max_rel(ref, got);
  check(e < 1e-2 && found == 1, "E4M3 A^T B with tensor-wise scales, bf16 out", e);

  // Row-wise: one scale per row of A^T (m of them) and per column of B (n).
  const int32_t outer = 3;  // CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F
  CK(cublasLtMatmulDescSetAttribute(desc, (cublasLtMatmulDescAttributes_t)31, &outer, sizeof outer));
  CK(cublasLtMatmulDescSetAttribute(desc, (cublasLtMatmulDescAttributes_t)32, &outer, sizeof outer));
  CK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dsar, sizeof dsar));
  CK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dsbr, sizeof dsbr));
  CK(cublasLtMatmul(lt, desc, &alpha, da, L.a, db, L.b, &beta, dd, L.c, dd, L.d, nullptr, nullptr, 0, 0));
  out = download(dd, (size_t)m * n);
  ref.clear();
  got.clear();
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      ref.push_back(sa_r[i] * sb_r[j] * dot(i, j));
      got.push_back(f(out[(size_t)j * m + i]));
    }
  e = max_rel(ref, got);
  check(e < 1e-2, "E4M3 A^T B with row-wise (outer-vector) scales", e);

  // A block-scaled mode is refused, not approximated.
  const int32_t block = 2;  // CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
  CK(cublasLtMatmulDescSetAttribute(desc, (cublasLtMatmulDescAttributes_t)31, &block, sizeof block));
  check(cublasLtMatmul(lt, desc, &alpha, da, L.a, db, L.b, &beta, dd, L.c, dd, L.d, nullptr, nullptr, 0, 0) ==
            CUBLAS_STATUS_NOT_SUPPORTED,
        "a block-scaled mode is refused", 0);
  cublasLtMatmulDescDestroy(desc);
  cublasLtMatmulPreferenceDestroy(pref);

  {  // E4M3 out: D = scaleD * (A^T B), saturating; amax is taken before scaleD.
    cublasLtMatmulDesc_t d8;
    CK(cublasLtMatmulDescCreate(&d8, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CK(cublasLtMatmulDescSetAttribute(d8, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T));
    float* dscale = upload(std::vector<float>{40.0f});
    float* damax = upload(std::vector<float>{-1.0f});
    CK(cublasLtMatmulDescSetAttribute(d8, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &dscale, sizeof dscale));
    CK(cublasLtMatmulDescSetAttribute(d8, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &damax, sizeof damax));
    Layouts L8;
    CK(cublasLtMatrixLayoutCreate(&L8.a, CUDA_R_8F_E4M3, k, m, k));
    CK(cublasLtMatrixLayoutCreate(&L8.b, CUDA_R_8F_E4M3, k, n, k));
    CK(cublasLtMatrixLayoutCreate(&L8.c, CUDA_R_16BF, m, n, m));
    CK(cublasLtMatrixLayoutCreate(&L8.d, CUDA_R_8F_E4M3, m, n, m));
    auto* d8out = upload(std::vector<__nv_fp8_e4m3>((size_t)m * n));
    CK(cublasLtMatmul(lt, d8, &alpha, da, L8.a, db, L8.b, &beta, nullptr, L8.c, d8out, L8.d, nullptr, nullptr, 0, 0));
    const auto o8 = download(d8out, (size_t)m * n);
    double amax = 0;
    bool bits = true;
    int saturated = 0;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        const double v = dot(i, j);
        amax = std::fmax(amax, std::fabs(v));
        // NVIDIA's own saturating conversion is the reference, bit for bit.
        const __nv_fp8_storage_t want = __nv_cvt_double_to_fp8(v * 40.0, __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_storage_t have;
        std::memcpy(&have, &o8[(size_t)j * m + i], 1);
        if (have != want) bits = false;
        saturated += std::fabs(v * 40.0) > 448.0;
      }
    const float got_amax = download(damax, 1)[0];
    check(bits && saturated > 0, "E4M3 out matches NVIDIA's saturating conversion bit for bit", saturated);
    check(std::fabs(got_amax - amax) <= 1e-5 * amax, "amax of D is taken before D's scale", std::fabs(got_amax - amax));
    cublasLtMatmulDescDestroy(d8);
    for (void* p : {(void*)dscale, (void*)damax, (void*)d8out}) cudaFree(p);
  }
  for (void* p : {(void*)da, (void*)db, (void*)dsa, (void*)dsb, (void*)dsar, (void*)dsbr, (void*)dd}) cudaFree(p);
}

// alpha in device memory, and an epilogue it does not implement.
static void pointer_mode_and_refusal() {
  const int m = 3, n = 2, k = 4;
  std::vector<float> a((size_t)m * k), b((size_t)k * n);
  for (int i = 0; i < m * k; ++i) a[i] = (float)val(i, 0, 8);
  for (int i = 0; i < k * n; ++i) b[i] = (float)val(i, 1, 9);
  float *da = upload(a), *db = upload(b), *dd = upload(std::vector<float>((size_t)m * n));
  float* dalpha = upload(std::vector<float>{-2.0f, 0.0f});  // alpha, beta
  cublasLtMatmulDesc_t desc;
  CK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const int32_t mode = CUBLASLT_POINTER_MODE_DEVICE;
  CK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &mode, sizeof mode));
  Layouts L;
  CK(cublasLtMatrixLayoutCreate(&L.a, CUDA_R_32F, m, k, m));
  CK(cublasLtMatrixLayoutCreate(&L.b, CUDA_R_32F, k, n, k));
  CK(cublasLtMatrixLayoutCreate(&L.d, CUDA_R_32F, m, n, m));
  CK(cublasLtMatmul(lt, desc, dalpha, da, L.a, db, L.b, dalpha + 1, dd, L.d, dd, L.d, nullptr, nullptr, 0, 0));
  const auto out = download(dd, (size_t)m * n);
  std::vector<double> ref, got(out.begin(), out.end());
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      double s = 0;
      for (int p = 0; p < k; ++p) s += (double)a[(size_t)p * m + i] * b[(size_t)j * k + p];
      ref.push_back(-2.0 * s);
    }
  const double e = max_rel(ref, got);
  check(e < 1e-6, "alpha and beta read from device memory in the device pointer mode", e);
  const cublasLtEpilogue_t bgrad = CUBLASLT_EPILOGUE_BGRADB;
  CK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &bgrad, sizeof bgrad));
  check(cublasLtMatmul(lt, desc, dalpha, da, L.a, db, L.b, dalpha + 1, dd, L.d, dd, L.d, nullptr, nullptr, 0, 0) ==
            CUBLAS_STATUS_NOT_SUPPORTED,
        "an epilogue it does not implement is refused, not skipped", 0);
  cublasLtMatmulDescDestroy(desc);
  for (void* p : {(void*)da, (void*)db, (void*)dd, (void*)dalpha}) cudaFree(p);
}

// The logger calls frameworks and bindings make at start-up (cudarc sets a
// level). An RTX 3060's cuBLASLt takes levels 0 to 6 and refuses the rest,
// and takes any mask.
static void logger() {
  check(cublasLtLoggerSetLevel(0) == CUBLAS_STATUS_SUCCESS, "logger level 0 taken", 0);
  check(cublasLtLoggerSetLevel(5) == CUBLAS_STATUS_SUCCESS, "logger level 5 taken", 0);
  check(cublasLtLoggerSetLevel(6) == CUBLAS_STATUS_SUCCESS, "logger level 6 taken", 0);
  check(cublasLtLoggerSetLevel(7) == CUBLAS_STATUS_INVALID_VALUE, "logger level 7 refused", 0);
  check(cublasLtLoggerSetLevel(-1) == CUBLAS_STATUS_INVALID_VALUE, "logger level -1 refused", 0);
  check(cublasLtLoggerSetMask(31) == CUBLAS_STATUS_SUCCESS, "logger mask 31 taken", 0);
  check(cublasLtLoggerSetMask(1000) == CUBLAS_STATUS_SUCCESS, "logger mask 1000 taken", 0);
  check(cublasLtLoggerSetCallback(nullptr) == CUBLAS_STATUS_SUCCESS, "logger callback cleared", 0);
  check(cublasLtLoggerSetFile(nullptr) == CUBLAS_STATUS_SUCCESS, "logger file cleared", 0);
  check(cublasLtLoggerSetLevel(0) == CUBLAS_STATUS_SUCCESS, "logger back off", 0);
}

int main() {
  if (cublasLtCreate(&lt)) {
    std::printf("FAIL: cublasLtCreate\n");
    return 1;
  }
  half_bias_relu();
  bf16_batched_rowmajor();
  fp8_scaled();
  pointer_mode_and_refusal();
  logger();
  cublasLtDestroy(lt);
  std::printf(failures ? "FAIL: %d cuBLASLt checks\n" : "PASS: every cuBLASLt check\n", failures);
  return failures ? 1 : 0;
}
