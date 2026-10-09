// cuFFT's LTO callbacks (cufftXtSetJITCallback, CUDA 12.6 and later) given as LTO-IR: the
// fatbin nvcc -dlto writes, which holds NVVM bitcode and no PTX or machine code.
// fft_callbacks.cu covers the callbacks given as PTX; this is the form NVIDIA documents.
//
//   fft_lto_callbacks <loads.fatbin> <stores.fatbin>
//
// run_fft_lto_callbacks.sh builds the fatbins from fft_lto_callback_loads.cu and
// fft_lto_callback_stores.cu and runs this
// on the simulator, or with --card against NVIDIA's cuFFT. Each transform is checked against a
// direct DFT on the host. On the simulator the LTO-IR is turned into machine code by the CUDA
// toolkit's libnvJitLink (the host's, as its NVRTC uses the host's libnvrtc); without the
// toolkit the plan fails as a callback that does not link does, and the test SKIPs.
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

using cd = std::complex<double>;

static int failures = 0;

static void check(bool ok, const char* what, double err = 0) {
  std::printf("%-4s %s (%.2e)\n", ok ? "ok" : "FAIL", what, err);
  if (!ok) ++failures;
}

#if CUFFT_VERSION >= 11300
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

static std::vector<char> g_loads, g_stores;

// Makes a plan, sets the callbacks, makes it. Returns the status of the final cufftMakePlan.
static cufftResult plan_with(cufftHandle* h, cufftType type, int n, const char* ld, int ldtype, void* ldinfo, const char* st,
                             int sttype) {
  cufftCreate(h);
  cufftSetAutoAllocation(*h, 1);
  if (ld) {
    void* infos[1] = {ldinfo};
    const cufftResult r = cufftXtSetJITCallback(*h, ld, g_loads.data(), g_loads.size(), (cufftXtCallbackType)ldtype, infos);
    if (r != CUFFT_SUCCESS) return r;
  }
  if (st) {
    const cufftResult r = cufftXtSetJITCallback(*h, st, g_stores.data(), g_stores.size(), (cufftXtCallbackType)sttype, nullptr);
    if (r != CUFFT_SUCCESS) return r;
  }
  size_t ws = 0;
  return cufftMakePlan1d(*h, n, type, 1, &ws);
}

static int run() {
  const int n = 64;
  cufftHandle h;
  float* d_scale = nullptr;
  cudaMalloc(&d_scale, sizeof(float));
  const float scale = 0.5f;
  cudaMemcpy(d_scale, &scale, sizeof scale, cudaMemcpyHostToDevice);
  cufftResult r = plan_with(&h, CUFFT_C2C, n, "ld_scale", CUFFT_CB_LD_COMPLEX, d_scale, "st_conj", CUFFT_CB_ST_COMPLEX);
  if (r != CUFFT_SUCCESS) {
    // No toolkit nvJitLink to read LTO-IR with: the plan does not link.
    cufftHandle h2;
    const cufftResult rl = plan_with(&h2, CUFFT_C2C, n, "ld_scale", CUFFT_CB_LD_COMPLEX, d_scale, nullptr, 0);
    cufftDestroy(h2);
    const cufftResult rs = plan_with(&h2, CUFFT_C2C, n, nullptr, 0, nullptr, "st_conj", CUFFT_CB_ST_COMPLEX);
    std::printf("SKIP: the plan with LTO-IR callbacks failed (%d; load alone %d, store alone %d); the host needs the CUDA "
                "toolkit's libnvJitLink\n", (int)r, (int)rl, (int)rs);
    cufftDestroy(h);
    cufftDestroy(h2);
    cudaFree(d_scale);
    return 0;
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
  r = cufftExecC2C(h, d_in, d_out, CUFFT_FORWARD);
  check(r == CUFFT_SUCCESS, "C2C with LTO-IR load and store callbacks executes");
  std::vector<cufftComplex> got(n);
  cudaMemcpy(got.data(), d_out, n * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
  double err = 0;
  for (int i = 0; i < n; ++i) err = std::fmax(err, std::abs(cd(got[i].x, got[i].y) - std::conj(want[i])));
  check(err < 1e-4, "C2C: load scaled by callerInfo, store conjugates", err);
  cufftDestroy(h);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_scale);

  // R2C with a load callback only.
  {
    r = plan_with(&h, CUFFT_R2C, n, "ld_real", CUFFT_CB_LD_REAL, nullptr, nullptr, 0);
    check(r == CUFFT_SUCCESS, "R2C with an LTO-IR load callback plans");
    if (r == CUFFT_SUCCESS) {
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
      r = cufftExecR2C(h, dx, dy);
      std::vector<cufftComplex> y(n / 2 + 1);
      cudaMemcpy(y.data(), dy, y.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
      double e = 0;
      for (int i = 0; i <= n / 2; ++i) e = std::fmax(e, std::abs(cd(y[i].x, y[i].y) - w[i]));
      check(r == CUFFT_SUCCESS && e < 1e-3, "R2C: the load callback adds the offset", e);
      cudaFree(dx);
      cudaFree(dy);
      cufftDestroy(h);
    }
  }

  // Z2Z with a store callback that calls a helper.
  {
    r = plan_with(&h, CUFFT_Z2Z, n, nullptr, 0, nullptr, "st_half", CUFFT_CB_ST_COMPLEX_DOUBLE);
    check(r == CUFFT_SUCCESS, "Z2Z with an LTO-IR store callback plans");
    if (r == CUFFT_SUCCESS) {
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
      r = cufftExecZ2Z(h, dx, dy, CUFFT_INVERSE);
      std::vector<cufftDoubleComplex> y(n);
      cudaMemcpy(y.data(), dy, n * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
      double e = 0;
      for (int i = 0; i < n; ++i) e = std::fmax(e, std::abs(cd(y[i].x, y[i].y) - w[i] / 2.0));
      check(r == CUFFT_SUCCESS && e < 1e-10, "Z2Z: the store callback halves the result", e);
      cudaFree(dx);
      cudaFree(dy);
      cufftDestroy(h);
    }
  }

  // A symbol the fatbin does not define: the plan does not link.
  r = plan_with(&h, CUFFT_C2C, n, "no_such_callback", CUFFT_CB_LD_COMPLEX, nullptr, nullptr, 0);
  check(r != CUFFT_SUCCESS, "a callback the image does not define fails the plan");
  cufftDestroy(h);
  return failures ? 1 : 0;
}
#endif

int main(int argc, char** argv) {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
    std::printf("SKIP: no device\n");
    return 0;
  }
#if CUFFT_VERSION >= 11300
  if (argc < 3) {
    std::printf("usage: fft_lto_callbacks <loads.fatbin> <stores.fatbin>\n");
    return 2;
  }
  for (int k = 1; k <= 2; ++k) {
    std::ifstream f(argv[k], std::ios::binary);
    (k == 1 ? g_loads : g_stores).assign(std::istreambuf_iterator<char>(f), {});
    if ((k == 1 ? g_loads : g_stores).empty()) {
      std::printf("FAIL cannot read %s\n", argv[k]);
      return 2;
    }
  }
  const int rc = run();
  if (rc == 0 && !failures) std::printf("PASS (0 failures)\n");
  else if (failures) std::printf("FAIL (%d failures)\n", failures);
  return rc;
#else
  (void)argc, (void)argv;
  std::printf("SKIP: this cuFFT predates LTO callbacks (CUDA 12.6)\n");
  return 0;
#endif
}
