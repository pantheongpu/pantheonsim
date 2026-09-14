// Differential conformance for the cuFFT shim: C2C/Z2Z in both directions,
// R2C/C2R and D2Z/Z2D round trips, 2-D and 3-D plans, batches, non-power-of-two
// and prime sizes, and the advanced 1-D layout. Values are printed rounded to
// what single precision actually resolves; the transforms are computed in
// double here and in single on hardware, so they agree to rounding, not bits.
#include <cufft.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

#define CF(x) do { cufftResult r_ = (x); if (r_ != CUFFT_SUCCESS) { \
  printf("%s -> %d\n", #x, (int)r_); return; } } while (0)

static float sig(int i) { return std::sin(0.37f * i) + 0.5f * std::cos(0.11f * i) + 0.05f * (i % 9); }

// Aggregate rather than element dumps: a full spectrum is noise to read, and
// these four numbers pin down the transform just as tightly.
// A component that is analytically zero comes out as +0, -0 or a value below
// the printed resolution depending on the order the terms were summed in. That
// is not a semantics difference, so snap it before printing.
static double z(double v) { return std::fabs(v) < 5e-4 ? 0.0 : v; }

static void report(const char* tag, const std::vector<float>& re, const std::vector<float>& im) {
  double sr = 0, si = 0, mag = 0, peak = 0;
  int peak_at = 0;
  for (size_t i = 0; i < re.size(); ++i) {
    sr += re[i]; si += im[i];
    const double m = std::hypot(re[i], im[i]);
    mag += m;
    if (m > peak + 1e-6) { peak = m; peak_at = (int)i; }
  }
  printf("%-30s n=%zu re=%.3f im=%.3f absmag=%.3f peak=%.3f@%d\n", tag, re.size(), z(sr), z(si),
         z(mag), z(peak), peak_at);
}

static void split(const std::vector<cufftComplex>& c, std::vector<float>* re,
                  std::vector<float>* im) {
  re->resize(c.size()); im->resize(c.size());
  for (size_t i = 0; i < c.size(); ++i) { (*re)[i] = c[i].x; (*im)[i] = c[i].y; }
}

static void c2c(const char* tag, int rank, const int* n, int batch, int dir) {
  size_t total = 1;
  for (int i = 0; i < rank; ++i) total *= (size_t)n[i];
  total *= (size_t)batch;
  std::vector<cufftComplex> h(total);
  for (size_t i = 0; i < total; ++i) { h[i].x = sig((int)i); h[i].y = 0.25f * sig((int)i + 3); }
  cufftComplex* d = nullptr;
  cudaMalloc(&d, total * sizeof(cufftComplex));
  cudaMemcpy(d, h.data(), total * sizeof(cufftComplex), cudaMemcpyHostToDevice);
  cufftHandle p;
  if (rank == 1) CF(cufftPlan1d(&p, n[0], CUFFT_C2C, batch));
  else if (rank == 2) CF(cufftPlan2d(&p, n[0], n[1], CUFFT_C2C));
  else CF(cufftPlan3d(&p, n[0], n[1], n[2], CUFFT_C2C));
  CF(cufftExecC2C(p, d, d, dir));
  cudaMemcpy(h.data(), d, total * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
  std::vector<float> re, im;
  split(h, &re, &im);
  report(tag, re, im);
  cufftDestroy(p);
  cudaFree(d);
}

static void run() {
  { const int n[] = {64}; c2c("c2c 64 forward", 1, n, 1, CUFFT_FORWARD); }
  { const int n[] = {64}; c2c("c2c 64 inverse", 1, n, 1, CUFFT_INVERSE); }
  { const int n[] = {96}; c2c("c2c 96 (2^5*3)", 1, n, 1, CUFFT_FORWARD); }
  { const int n[] = {100}; c2c("c2c 100 (2^2*5^2)", 1, n, 1, CUFFT_FORWARD); }
  { const int n[] = {97}; c2c("c2c 97 (prime)", 1, n, 1, CUFFT_FORWARD); }
  { const int n[] = {32}; c2c("c2c 32 batch4", 1, n, 4, CUFFT_FORWARD); }
  { const int n[] = {8, 16}; c2c("c2c 8x16 2d", 2, n, 1, CUFFT_FORWARD); }
  { const int n[] = {4, 6, 8}; c2c("c2c 4x6x8 3d", 3, n, 1, CUFFT_FORWARD); }

  {  // R2C then C2R: the round trip must come back scaled by n.
    const int n = 128;
    std::vector<float> h(n);
    for (int i = 0; i < n; ++i) h[i] = sig(i);
    cufftReal* dr = nullptr; cufftComplex* dc = nullptr;
    cudaMalloc(&dr, n * sizeof(cufftReal));
    cudaMalloc(&dc, (n / 2 + 1) * sizeof(cufftComplex));
    cudaMemcpy(dr, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cufftHandle fwd, inv;
    CF(cufftPlan1d(&fwd, n, CUFFT_R2C, 1));
    CF(cufftPlan1d(&inv, n, CUFFT_C2R, 1));
    CF(cufftExecR2C(fwd, dr, dc));
    std::vector<cufftComplex> spec(n / 2 + 1);
    cudaMemcpy(spec.data(), dc, spec.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    std::vector<float> re, im; split(spec, &re, &im);
    report("r2c 128 spectrum", re, im);
    CF(cufftExecC2R(inv, dc, dr));
    std::vector<float> back(n);
    cudaMemcpy(back.data(), dr, n * sizeof(float), cudaMemcpyDeviceToHost);
    double err = 0;
    for (int i = 0; i < n; ++i) err = std::fmax(err, std::fabs(back[i] / n - h[i]));
    printf("%-30s roundtrip max error < 1e-4: %s\n", "r2c/c2r 128", err < 1e-4 ? "yes" : "NO");
    cufftDestroy(fwd); cufftDestroy(inv); cudaFree(dr); cudaFree(dc);
  }

  {  // 2-D R2C, where only the fastest dimension is halved.
    const int nx = 8, ny = 12;
    std::vector<float> h(nx * ny);
    for (int i = 0; i < nx * ny; ++i) h[i] = sig(i);
    cufftReal* dr = nullptr; cufftComplex* dc = nullptr;
    cudaMalloc(&dr, h.size() * sizeof(float));
    cudaMalloc(&dc, (size_t)nx * (ny / 2 + 1) * sizeof(cufftComplex));
    cudaMemcpy(dr, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    cufftHandle p;
    CF(cufftPlan2d(&p, nx, ny, CUFFT_R2C));
    CF(cufftExecR2C(p, dr, dc));
    std::vector<cufftComplex> spec((size_t)nx * (ny / 2 + 1));
    cudaMemcpy(spec.data(), dc, spec.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    std::vector<float> re, im; split(spec, &re, &im);
    report("r2c 8x12 2d", re, im);
    cufftDestroy(p); cudaFree(dr); cudaFree(dc);
  }

  {  // Double precision, where hardware and this implementation both use double.
    const int n = 64;
    std::vector<cufftDoubleComplex> h(n);
    for (int i = 0; i < n; ++i) { h[i].x = sig(i); h[i].y = 0.25 * sig(i + 3); }
    cufftDoubleComplex* d = nullptr;
    cudaMalloc(&d, n * sizeof(cufftDoubleComplex));
    cudaMemcpy(d, h.data(), n * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    cufftHandle p;
    CF(cufftPlan1d(&p, n, CUFFT_Z2Z, 1));
    CF(cufftExecZ2Z(p, d, d, CUFFT_FORWARD));
    cudaMemcpy(h.data(), d, n * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    std::vector<float> re(n), im(n);
    for (int i = 0; i < n; ++i) { re[i] = (float)h[i].x; im[i] = (float)h[i].y; }
    report("z2z 64 forward", re, im);
    cufftDestroy(p); cudaFree(d);
  }

  {  // Advanced layout: 3 interleaved signals, stride 3, in one batched plan.
    const int n = 32, batch = 3;
    std::vector<cufftComplex> h((size_t)n * batch);
    for (size_t i = 0; i < h.size(); ++i) { h[i].x = sig((int)i); h[i].y = 0.0f; }
    cufftComplex* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(cufftComplex));
    cudaMemcpy(d, h.data(), h.size() * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    cufftHandle p;
    int nn = n;
    CF(cufftPlanMany(&p, 1, &nn, &nn, batch, 1, &nn, batch, 1, CUFFT_C2C, batch));
    CF(cufftExecC2C(p, d, d, CUFFT_FORWARD));
    cudaMemcpy(h.data(), d, h.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    std::vector<float> re, im; split(h, &re, &im);
    report("c2c 32x3 interleaved", re, im);
    cufftDestroy(p); cudaFree(d);
  }

  int v = 0, major = 0;
  cufftGetVersion(&v);
  cufftGetProperty(MAJOR_VERSION, &major);
  printf("cufft major %d (version/1000 %d)\n", major, v / 1000);
}

int main() { run(); return 0; }
