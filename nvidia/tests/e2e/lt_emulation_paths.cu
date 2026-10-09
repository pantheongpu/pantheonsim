// cublasLtEmulationDesc_t, the settings of floating-point emulation, and how
// cublasLtMatmul treats one. What an RTX 3060's cuBLASLt 13.0 does, found by
// experiment, and every check here passes on it too:
//
//   * the defaults: strategy 0, special values 0xFFFF, mantissa control 0, max
//     mantissa bit count 0, offset 0, bit count pointer NULL;
//   * a buffer of the wrong size, a NULL buffer, a NULL descriptor and an
//     unknown attribute are INVALID_VALUE; a size query needs sizeWritten; no
//     value is range-checked when it is set;
//   * a descriptor lives in caller memory too (cublasLtEmulationDescInit);
//   * CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR is a pointer, read back as set;
//   * cublasLtMatmul with a descriptor: INVALID_VALUE unless the compute type
//     is an emulated one, and for a negative max mantissa bit count; otherwise
//     it runs in plain double precision here -- the card's cuBLASLt did not
//     emulate with the descriptor on that GPU (the bit count pointer stayed
//     untouched).
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#if CUBLAS_VER_MAJOR < 13
int main() {
  std::printf("SKIP: the emulation descriptor needs the CUDA 13 cuBLASLt headers\n");
  return 0;
}
#else

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want)                                                                 \
  do {                                                                                 \
    const int got_ = (int)(call);                                                      \
    char what_[300];                                                                   \
    std::snprintf(what_, sizeof what_, "%s -> %s (got %d)", #call, #want, got_);       \
    check(got_ == (int)(want), what_);                                                 \
  } while (0)

int main() {
  cublasLtHandle_t h;
  cublasLtCreate(&h);
  cublasLtEmulationDesc_t ed;
  IS(cublasLtEmulationDescCreate(&ed), CUBLAS_STATUS_SUCCESS);
  IS(cublasLtEmulationDescCreate(nullptr), CUBLAS_STATUS_INVALID_VALUE);

  // Defaults.
  {
    const int32_t want[5] = {0, 0xFFFF, 0, 0, 0};
    bool ok = true;
    for (int a = 0; a < 5; ++a) {
      int32_t v = -7;
      size_t w = 0;
      ok = ok && cublasLtEmulationDescGetAttribute(ed, (cublasLtEmulationDescAttributes_t)a, &v, 4, &w) == CUBLAS_STATUS_SUCCESS && v == want[a] && w == 4;
    }
    int32_t* p = (int32_t*)0x1;
    size_t w = 0;
    ok = ok && cublasLtEmulationDescGetAttribute(ed, CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_COUNT_POINTER, &p, 8, &w) == CUBLAS_STATUS_SUCCESS &&
         p == nullptr && w == 8;
    check(ok, "the defaults: 0, 0xFFFF, 0, 0, 0 and a NULL pointer");
  }
  // Set and get.
  {
    int32_t v = 2, got = -1;
    size_t w = 0;
    IS(cublasLtEmulationDescSetAttribute(ed, CUBLASLT_EMULATION_DESC_STRATEGY, &v, 4), CUBLAS_STATUS_SUCCESS);
    IS(cublasLtEmulationDescGetAttribute(ed, CUBLASLT_EMULATION_DESC_STRATEGY, &got, 4, &w), CUBLAS_STATUS_SUCCESS);
    check(got == 2, "a strategy set is read back");
    v = 7;
    IS(cublasLtEmulationDescSetAttribute(ed, CUBLASLT_EMULATION_DESC_STRATEGY, &v, 4), CUBLAS_STATUS_SUCCESS);   // not range-checked
    IS(cublasLtEmulationDescSetAttribute(ed, CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL, &v, 4), CUBLAS_STATUS_SUCCESS);
    v = -3;
    IS(cublasLtEmulationDescSetAttribute(ed, CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT, &v, 4), CUBLAS_STATUS_SUCCESS);
    IS(cublasLtEmulationDescGetAttribute(ed, CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT, &got, 4, &w), CUBLAS_STATUS_SUCCESS);
    check(got == -3, "a negative maximum is taken and read back");
    IS(cublasLtEmulationDescSetAttribute(ed, CUBLASLT_EMULATION_DESC_STRATEGY, &v, 8), CUBLAS_STATUS_INVALID_VALUE);
    IS(cublasLtEmulationDescSetAttribute(ed, CUBLASLT_EMULATION_DESC_STRATEGY, nullptr, 4), CUBLAS_STATUS_INVALID_VALUE);
    IS(cublasLtEmulationDescSetAttribute(ed, (cublasLtEmulationDescAttributes_t)9, &v, 4), CUBLAS_STATUS_INVALID_VALUE);
    IS(cublasLtEmulationDescSetAttribute(nullptr, CUBLASLT_EMULATION_DESC_STRATEGY, &v, 4), CUBLAS_STATUS_INVALID_VALUE);
    IS(cublasLtEmulationDescGetAttribute(ed, CUBLASLT_EMULATION_DESC_STRATEGY, nullptr, 0, &w), CUBLAS_STATUS_SUCCESS);
    check(w == 4, "a size query answers 4");
    IS(cublasLtEmulationDescGetAttribute(ed, CUBLASLT_EMULATION_DESC_STRATEGY, nullptr, 0, nullptr), CUBLAS_STATUS_INVALID_VALUE);
    IS(cublasLtEmulationDescGetAttribute(ed, CUBLASLT_EMULATION_DESC_STRATEGY, &got, 8, &w), CUBLAS_STATUS_INVALID_VALUE);
    IS(cublasLtEmulationDescGetAttribute(ed, CUBLASLT_EMULATION_DESC_STRATEGY, nullptr, 4, &w), CUBLAS_STATUS_INVALID_VALUE);
    IS(cublasLtEmulationDescGetAttribute(ed, (cublasLtEmulationDescAttributes_t)9, &got, 4, &w), CUBLAS_STATUS_INVALID_VALUE);
    IS(cublasLtEmulationDescGetAttribute(nullptr, CUBLASLT_EMULATION_DESC_STRATEGY, &got, 4, &w), CUBLAS_STATUS_INVALID_VALUE);
  }
  // In caller memory.
  {
    cublasLtEmulationDescOpaque_t buf;
    IS(cublasLtEmulationDescInit(&buf), CUBLAS_STATUS_SUCCESS);
    int32_t v = 1, got = -1;
    size_t w = 0;
    IS(cublasLtEmulationDescSetAttribute(&buf, CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL, &v, 4), CUBLAS_STATUS_SUCCESS);
    IS(cublasLtEmulationDescGetAttribute(&buf, CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL, &got, 4, &w), CUBLAS_STATUS_SUCCESS);
    check(got == 1, "a descriptor initialised in caller memory works");
    IS(cublasLtEmulationDescInit_internal(&buf, 4), CUBLAS_STATUS_ALLOC_FAILED);
  }
  IS(cublasLtEmulationDescDestroy(nullptr), CUBLAS_STATUS_SUCCESS);

  // On a matmul.
  const int n = 24;
  std::vector<double> A((size_t)n * n), B((size_t)n * n), plain((size_t)n * n), got((size_t)n * n);
  for (size_t i = 0; i < A.size(); ++i) {
    A[i] = (double)((i * 7919u) % 1000) / 500.0 - 1.0;
    B[i] = (double)((i * 104729u) % 1000) / 500.0 - 1.0;
  }
  double *a, *b, *c, *d;
  int* dbits;
  cudaMalloc(&a, A.size() * 8);
  cudaMalloc(&b, B.size() * 8);
  cudaMalloc(&c, A.size() * 8);
  cudaMalloc(&d, A.size() * 8);
  cudaMalloc(&dbits, 4);
  cudaMemcpy(a, A.data(), A.size() * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(b, B.data(), B.size() * 8, cudaMemcpyHostToDevice);
  auto run = [&](cublasComputeType_t ct, const cublasLtEmulationDesc_t* desc, int* status, int* bits) {
    cublasLtMatmulDesc_t md;
    cublasLtMatrixLayout_t l;
    cublasLtMatmulDescCreate(&md, ct, CUDA_R_64F);
    if (desc) cublasLtMatmulDescSetAttribute(md, CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR, desc, sizeof *desc);
    cublasLtMatrixLayoutCreate(&l, CUDA_R_64F, n, n, n);
    const int sentinel = -99;
    cudaMemcpy(dbits, &sentinel, 4, cudaMemcpyHostToDevice);
    cudaMemset(d, 0, A.size() * 8);
    const double one = 1, zero = 0;
    *status = (int)cublasLtMatmul(h, md, &one, a, l, b, l, &zero, c, l, d, l, nullptr, nullptr, 0, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), d, got.size() * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(bits, dbits, 4, cudaMemcpyDeviceToHost);
    cublasLtMatmulDescDestroy(md);
    cublasLtMatrixLayoutDestroy(l);
  };
  int st, bits;
  run(CUBLAS_COMPUTE_64F, nullptr, &st, &bits);
  plain = got;
  check(st == 0 && bits == -99, "a plain double matmul");

  // The descriptor attribute itself.
  {
    cublasLtMatmulDesc_t md;
    cublasLtMatmulDescCreate(&md, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUDA_R_64F);
    cublasLtEmulationDesc_t read = (cublasLtEmulationDesc_t)0x1;
    size_t w = 0;
    IS(cublasLtMatmulDescGetAttribute(md, CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR, &read, sizeof read, &w), CUBLAS_STATUS_SUCCESS);
    check(read == nullptr && w == sizeof read, "the descriptor attribute is NULL by default");
    IS(cublasLtMatmulDescSetAttribute(md, CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR, &ed, sizeof ed), CUBLAS_STATUS_SUCCESS);
    read = nullptr;
    IS(cublasLtMatmulDescGetAttribute(md, CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR, &read, sizeof read, &w), CUBLAS_STATUS_SUCCESS);
    check(read == ed, "and reads back as set");
    IS(cublasLtMatmulDescSetAttribute(md, CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR, &ed, 4), CUBLAS_STATUS_INVALID_VALUE);
    cublasLtMatmulDescDestroy(md);
  }

  cublasLtEmulationDesc_t good;
  cublasLtEmulationDescCreate(&good);
  int32_t v = 2;
  cublasLtEmulationDescSetAttribute(good, CUBLASLT_EMULATION_DESC_STRATEGY, &v, 4);
  v = 1;
  cublasLtEmulationDescSetAttribute(good, CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL, &v, 4);
  v = 16;
  cublasLtEmulationDescSetAttribute(good, CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT, &v, 4);
  cublasLtEmulationDescSetAttribute(good, CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_COUNT_POINTER, &dbits, 8);
  run(CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, nullptr, &st, &bits);
  check(st == 0 && got == plain && bits == -99, "COMPUTE_64F_EMULATED_FIXEDPOINT without a descriptor: the plain product");
  run(CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, &good, &st, &bits);
  check(st == 0 && got == plain && bits == -99, "with an EAGER, fixed 16-bit descriptor: still the plain product, the pointer untouched");
  run(CUBLAS_COMPUTE_64F, &good, &st, &bits);
  check(st == CUBLAS_STATUS_INVALID_VALUE, "a descriptor on a plain COMPUTE_64F matmul is INVALID_VALUE");
  cublasLtEmulationDesc_t negative;
  cublasLtEmulationDescCreate(&negative);
  v = -3;
  cublasLtEmulationDescSetAttribute(negative, CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT, &v, 4);
  run(CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, &negative, &st, &bits);
  check(st == CUBLAS_STATUS_INVALID_VALUE, "a negative maximum mantissa bit count is INVALID_VALUE at the matmul");
  cublasLtEmulationDesc_t odd;
  cublasLtEmulationDescCreate(&odd);
  v = 7;
  cublasLtEmulationDescSetAttribute(odd, CUBLASLT_EMULATION_DESC_STRATEGY, &v, 4);
  cublasLtEmulationDescSetAttribute(odd, CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL, &v, 4);
  run(CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, &odd, &st, &bits);
  check(st == 0 && got == plain, "a strategy and a mantissa control the enums do not name are taken");

  cublasLtEmulationDescDestroy(ed);
  cublasLtEmulationDescDestroy(good);
  cublasLtEmulationDescDestroy(negative);
  cublasLtEmulationDescDestroy(odd);
  cublasLtDestroy(h);
  std::printf("%s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
#endif
