// cuFFT's legacy callbacks (cufftXtSetCallback), the way NVIDIA documents them:
// device functions of the program itself, built with -rdc=true, whose addresses
// are read out of __device__ variables and handed to the plan.
//
// NVIDIA ships these only in its static library. The program is linked with
// libcufft_static.a and libculibos.a, so it carries NVIDIA's own cuFFT and its
// kernels, which then run on whichever device the program finds -- a card, or
// VirtualGPU's simulated one. libcufft.so (what the shim stands in for) answers
// every legacy callback call with CUFFT_NOT_IMPLEMENTED, and the other test of
// this family (fft_callbacks.cu) pins that.
//
//   nvcc -rdc=true fft_legacy_callbacks.cu -lcufft_static -lculibos
//
// Each transform is checked against a direct DFT on the host. With the
// dynamic library (no libcufft_static.a to link), the program checks that the
// calls answer NOT_IMPLEMENTED and passes.
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

using cd = std::complex<double>;

static int failures = 0;

static void check(bool ok, const char* what, double err = 0) {
  std::printf("%-4s %s (%.2e)\n", ok ? "ok" : "FAIL", what, err);
  if (!ok) ++failures;
}

static double sig(int i) { return std::sin(0.37 * i) + 0.5 * std::cos(0.11 * i) + 0.05 * (i % 9); }

static std::vector<cd> dft(const std::vector<cd>& a, int sign) {
  const size_t n = a.size();
  std::vector<cd> o(n);
  for (size_t k = 0; k < n; ++k) {
    cd acc = 0;
    for (size_t j = 0; j < n; ++j) acc += a[j] * std::polar(1.0, sign * 2 * M_PI * (double)(j * k % n) / n);
    o[k] = acc;
  }
  return o;
}

// ---- the callbacks ----

// Load: the input element scaled by the factor callerInfo points at.
__device__ cufftComplex ld_scale(void* in, size_t offset, void* info, void*) {
  cufftComplex v = static_cast<cufftComplex*>(in)[offset];
  const float s = *static_cast<float*>(info);
  return make_cuFloatComplex(v.x * s, v.y * s);
}
__device__ cufftCallbackLoadC d_ld_scale = ld_scale;

// Store: the conjugate of the result.
__device__ void st_conj(void* out, size_t offset, cufftComplex e, void*, void*) {
  static_cast<cufftComplex*>(out)[offset] = make_cuFloatComplex(e.x, -e.y);
}
__device__ cufftCallbackStoreC d_st_conj = st_conj;

// Load (real input): the element plus its index.
__device__ cufftReal ld_real(void* in, size_t offset, void*, void*) {
  return static_cast<cufftReal*>(in)[offset] + (float)offset;
}
__device__ cufftCallbackLoadR d_ld_real = ld_real;

// Store (double complex): the result halved.
__device__ void st_half(void* out, size_t offset, cufftDoubleComplex e, void*, void*) {
  static_cast<cufftDoubleComplex*>(out)[offset] = make_cuDoubleComplex(e.x / 2, e.y / 2);
}
__device__ cufftCallbackStoreZ d_st_half = st_half;

// Store (real output of a C2R): the result plus one.
__device__ void st_real(void* out, size_t offset, cufftReal e, void*, void*) {
  static_cast<cufftReal*>(out)[offset] = e + 1.0f;
}
__device__ cufftCallbackStoreR d_st_real = st_real;

#define CF(x)                                                \
  do {                                                       \
    cufftResult r_ = (x);                                    \
    if (r_ != CUFFT_SUCCESS) {                               \
      std::printf("FAIL %s returned %d\n", #x, (int)r_);     \
      ++failures;                                            \
      return;                                                \
    }                                                        \
  } while (0)

template <class T>
static T from_symbol(const T& sym) {
  T host{};
  cudaMemcpyFromSymbol(&host, sym, sizeof host);
  return host;
}

static void run() {
  const int n = 64;
  cufftHandle h;
  CF(cufftPlan1d(&h, n, CUFFT_C2C, 1));
  cufftCallbackLoadC ld = nullptr;
  cudaMemcpyFromSymbol(&ld, d_ld_scale, sizeof ld);
  float* d_scale = nullptr;
  cudaMalloc(&d_scale, sizeof(float));
  const float scale = 0.5f;
  cudaMemcpy(d_scale, &scale, sizeof scale, cudaMemcpyHostToDevice);
  void* info[1] = {d_scale};
  const cufftResult r = cufftXtSetCallback(h, (void**)&ld, CUFFT_CB_LD_COMPLEX, info);
  if (r == CUFFT_NOT_IMPLEMENTED) {
    // libcufft.so: legacy callbacks live in the static library only.
    check(true, "libcufft.so answers legacy callbacks with CUFFT_NOT_IMPLEMENTED");
    cufftDestroy(h);
    cudaFree(d_scale);
    return;
  }
  check(r == CUFFT_SUCCESS, "cufftXtSetCallback (load, complex) succeeds");
  if (r != CUFFT_SUCCESS) {
    cufftDestroy(h);
    return;
  }

  std::vector<cufftComplex> in(n);
  std::vector<cd> ref(n);
  for (int i = 0; i < n; ++i) {
    in[i] = make_cuFloatComplex((float)sig(i), (float)sig(i + 5));
    ref[i] = cd(in[i].x * scale, in[i].y * scale);
  }
  const std::vector<cd> want = dft(ref, -1);
  cufftComplex *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, n * sizeof(cufftComplex));
  cudaMalloc(&d_out, n * sizeof(cufftComplex));
  cudaMemcpy(d_in, in.data(), n * sizeof(cufftComplex), cudaMemcpyHostToDevice);
  CF(cufftExecC2C(h, d_in, d_out, CUFFT_FORWARD));
  std::vector<cufftComplex> got(n);
  cudaMemcpy(got.data(), d_out, n * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
  double err = 0;
  for (int i = 0; i < n; ++i) err = std::fmax(err, std::abs(cd(got[i].x, got[i].y) - want[i]));
  check(err < 1e-4, "C2C with a load callback (scaled by callerInfo)", err);

  // Add a store callback: the result is conjugated.
  cufftCallbackStoreC st = nullptr;
  cudaMemcpyFromSymbol(&st, d_st_conj, sizeof st);
  CF(cufftXtSetCallback(h, (void**)&st, CUFFT_CB_ST_COMPLEX, nullptr));
  cudaMemset(d_out, 0, n * sizeof(cufftComplex));
  CF(cufftExecC2C(h, d_in, d_out, CUFFT_FORWARD));
  cudaMemcpy(got.data(), d_out, n * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
  err = 0;
  for (int i = 0; i < n; ++i) err = std::fmax(err, std::abs(cd(got[i].x, got[i].y) - std::conj(want[i])));
  check(err < 1e-4, "C2C with both callbacks (store conjugates)", err);

  // Clearing the load callback leaves the store one.
  CF(cufftXtClearCallback(h, CUFFT_CB_LD_COMPLEX));
  std::vector<cd> plain(n);
  for (int i = 0; i < n; ++i) plain[i] = cd(in[i].x, in[i].y);
  const std::vector<cd> want2 = dft(plain, -1);
  CF(cufftExecC2C(h, d_in, d_out, CUFFT_FORWARD));
  cudaMemcpy(got.data(), d_out, n * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
  err = 0;
  for (int i = 0; i < n; ++i) err = std::fmax(err, std::abs(cd(got[i].x, got[i].y) - std::conj(want2[i])));
  check(err < 1e-4, "after cufftXtClearCallback the load callback is gone", err);
  cufftDestroy(h);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_scale);

  // R2C with a real load callback.
  {
    cufftHandle p;
    CF(cufftPlan1d(&p, n, CUFFT_R2C, 1));
    cufftCallbackLoadR lr = from_symbol(d_ld_real);
    CF(cufftXtSetCallback(p, (void**)&lr, CUFFT_CB_LD_REAL, nullptr));
    std::vector<float> x(n);
    std::vector<cd> xr(n);
    for (int i = 0; i < n; ++i) {
      x[i] = (float)sig(i);
      xr[i] = cd(x[i] + (float)i, 0);
    }
    const std::vector<cd> w = dft(xr, -1);
    float* dx = nullptr;
    cufftComplex* dy = nullptr;
    cudaMalloc(&dx, n * sizeof(float));
    cudaMalloc(&dy, (n / 2 + 1) * sizeof(cufftComplex));
    cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    CF(cufftExecR2C(p, dx, dy));
    std::vector<cufftComplex> y(n / 2 + 1);
    cudaMemcpy(y.data(), dy, y.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    double e = 0;
    for (int i = 0; i <= n / 2; ++i) e = std::fmax(e, std::abs(cd(y[i].x, y[i].y) - w[i]));
    check(e < 1e-3, "R2C with a real load callback", e);
    cufftDestroy(p);
    cudaFree(dx);
    cudaFree(dy);
  }

  // Z2Z with a double-complex store callback.
  {
    cufftHandle p;
    CF(cufftPlan1d(&p, n, CUFFT_Z2Z, 1));
    cufftCallbackStoreZ sz = from_symbol(d_st_half);
    CF(cufftXtSetCallback(p, (void**)&sz, CUFFT_CB_ST_COMPLEX_DOUBLE, nullptr));
    std::vector<cufftDoubleComplex> x(n);
    std::vector<cd> xr(n);
    for (int i = 0; i < n; ++i) {
      x[i] = make_cuDoubleComplex(sig(i), sig(i + 3));
      xr[i] = cd(x[i].x, x[i].y);
    }
    const std::vector<cd> w = dft(xr, 1);
    cufftDoubleComplex *dx = nullptr, *dy = nullptr;
    cudaMalloc(&dx, n * sizeof(cufftDoubleComplex));
    cudaMalloc(&dy, n * sizeof(cufftDoubleComplex));
    cudaMemcpy(dx, x.data(), n * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    CF(cufftExecZ2Z(p, dx, dy, CUFFT_INVERSE));
    std::vector<cufftDoubleComplex> y(n);
    cudaMemcpy(y.data(), dy, n * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    double e = 0;
    for (int i = 0; i < n; ++i) e = std::fmax(e, std::abs(cd(y[i].x, y[i].y) - w[i] / 2.0));
    check(e < 1e-10, "Z2Z with a double-complex store callback", e);
    cufftDestroy(p);
    cudaFree(dx);
    cudaFree(dy);
  }

  // C2R with a real store callback.
  {
    cufftHandle p;
    CF(cufftPlan1d(&p, n, CUFFT_C2R, 1));
    cufftCallbackStoreR sr = from_symbol(d_st_real);
    CF(cufftXtSetCallback(p, (void**)&sr, CUFFT_CB_ST_REAL, nullptr));
    std::vector<cufftComplex> x(n / 2 + 1);
    std::vector<cd> full(n);
    for (int i = 0; i <= n / 2; ++i) {
      const float re = (float)sig(i);
      const float im = (i == 0 || i == n / 2) ? 0.0f : (float)sig(i + 7);
      x[i] = make_cuFloatComplex(re, im);
      full[i] = cd(re, im);
      if (i > 0 && i < n / 2) full[n - i] = std::conj(full[i]);
    }
    const std::vector<cd> w = dft(full, 1);
    cufftComplex* dx = nullptr;
    float* dy = nullptr;
    cudaMalloc(&dx, x.size() * sizeof(cufftComplex));
    cudaMalloc(&dy, n * sizeof(float));
    cudaMemcpy(dx, x.data(), x.size() * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    CF(cufftExecC2R(p, dx, dy));
    std::vector<float> y(n);
    cudaMemcpy(y.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost);
    double e = 0;
    for (int i = 0; i < n; ++i) e = std::fmax(e, std::fabs(y[i] - (w[i].real() + 1.0)));
    check(e < 1e-3, "C2R with a real store callback", e);
    cufftDestroy(p);
    cudaFree(dx);
    cudaFree(dy);
  }
}

int main() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
    std::printf("SKIP: no device\n");
    return 0;
  }
  run();
  std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
