// AMD's hipFFT, unmodified (over rocFFT), on a simulated MI300X: transforms
// of the shapes programs ask for, each checked against a discrete Fourier
// transform done on the host in double precision.
//
//   complex to complex, single precision: 1D of 256 points (a power of two)
//     and 100 (2^2 * 5^2), batched three at a time, forward and back; 2D of
//     16 x 24
//   real to complex and back, 1D of 100 points, two at a time
//   complex to complex, double precision: 1D of 128 points
//
// Built ahead of time by build.sh, from hipFFT's documented API.
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

namespace {

using cd = std::complex<double>;

// The DFT of `n` points, `sign` -1 forward and +1 back, unscaled as FFT
// libraries leave it.
std::vector<cd> dft(const std::vector<cd>& x, int sign) {
  const size_t n = x.size();
  std::vector<cd> y(n);
  for (size_t k = 0; k < n; ++k) {
    cd s = 0;
    for (size_t j = 0; j < n; ++j) s += x[j] * std::polar(1.0, sign * 2 * M_PI * double(j * k % n) / double(n));
    y[k] = s;
  }
  return y;
}

// The 2D DFT of rows x cols points, row-major.
std::vector<cd> dft2(const std::vector<cd>& x, size_t rows, size_t cols, int sign) {
  std::vector<cd> t(x.size()), y(x.size());
  for (size_t r = 0; r < rows; ++r) {
    const std::vector<cd> row = dft(std::vector<cd>(x.begin() + r * cols, x.begin() + (r + 1) * cols), sign);
    std::copy(row.begin(), row.end(), t.begin() + r * cols);
  }
  for (size_t c = 0; c < cols; ++c) {
    std::vector<cd> col(rows);
    for (size_t r = 0; r < rows; ++r) col[r] = t[r * cols + c];
    col = dft(col, sign);
    for (size_t r = 0; r < rows; ++r) y[r * cols + c] = col[r];
  }
  return y;
}

// Whether `got` is `want` to within `tol` of the largest magnitude in it.
template <class C>
bool close(const std::vector<C>& got, const std::vector<cd>& want, double tol) {
  double scale = 0, err = 0;
  for (const cd& w : want) scale = std::max(scale, std::abs(w));
  for (size_t i = 0; i < want.size(); ++i) err = std::max(err, std::abs(cd(got[i].x, got[i].y) - want[i]));
  return err <= tol * std::max(scale, 1.0);
}

// Deterministic inputs in [-1, 1).
double value(size_t i) { return double((i * 2654435761u) % 2001) / 1000.0 - 1.0; }

int ok = 0, total = 0;
void check(const char* what, bool good) {
  ++total;
  ok += good;
  if (!good) std::printf("wrong: %s\n", what);
}

bool fine(hipfftResult r, const char* what) {
  if (r != HIPFFT_SUCCESS) std::printf("%s: hipFFT error %d\n", what, int(r));
  return r == HIPFFT_SUCCESS;
}

// Single-precision complex transforms of `n` points, `batch` of them, forward
// and then back (which gives n times the input).
void c2c(int n, int batch) {
  const size_t all = size_t(n) * batch;
  std::vector<hipfftComplex> h(all);
  std::vector<cd> in(all);
  for (size_t i = 0; i < all; ++i) {
    in[i] = cd(value(i), value(i + all));
    h[i] = {float(in[i].real()), float(in[i].imag())};
  }
  hipfftComplex* d;
  hipfftHandle plan;
  if (hipMalloc(&d, all * sizeof(hipfftComplex)) != hipSuccess) return check("memory", false);
  (void)hipMemcpy(d, h.data(), all * sizeof(hipfftComplex), hipMemcpyHostToDevice);
  bool good = fine(hipfftPlan1d(&plan, n, HIPFFT_C2C, batch), "plan") &&
              fine(hipfftExecC2C(plan, d, d, HIPFFT_FORWARD), "forward");
  (void)hipMemcpy(h.data(), d, all * sizeof(hipfftComplex), hipMemcpyDeviceToHost);
  bool forward = good;
  for (int b = 0; b < batch && forward; ++b) {
    const std::vector<cd> want = dft(std::vector<cd>(in.begin() + b * n, in.begin() + (b + 1) * n), -1);
    forward = close(std::vector<hipfftComplex>(h.begin() + b * n, h.begin() + (b + 1) * n), want, 1e-5);
  }
  char what[64];
  std::snprintf(what, sizeof what, "C2C forward, %d points x %d", n, batch);
  check(what, forward);
  good = good && fine(hipfftExecC2C(plan, d, d, HIPFFT_BACKWARD), "backward");
  (void)hipMemcpy(h.data(), d, all * sizeof(hipfftComplex), hipMemcpyDeviceToHost);
  std::vector<cd> scaled(all);
  for (size_t i = 0; i < all; ++i) scaled[i] = in[i] * double(n);
  std::snprintf(what, sizeof what, "C2C back, %d points x %d", n, batch);
  check(what, good && close(h, scaled, 1e-5));
  (void)hipfftDestroy(plan);
  (void)hipFree(d);
}

void c2c_2d(int rows, int cols) {
  const size_t all = size_t(rows) * cols;
  std::vector<hipfftComplex> h(all);
  std::vector<cd> in(all);
  for (size_t i = 0; i < all; ++i) {
    in[i] = cd(value(3 * i), value(3 * i + 1));
    h[i] = {float(in[i].real()), float(in[i].imag())};
  }
  hipfftComplex* d;
  hipfftHandle plan;
  if (hipMalloc(&d, all * sizeof(hipfftComplex)) != hipSuccess) return check("memory", false);
  (void)hipMemcpy(d, h.data(), all * sizeof(hipfftComplex), hipMemcpyHostToDevice);
  const bool good = fine(hipfftPlan2d(&plan, rows, cols, HIPFFT_C2C), "plan 2D") &&
                    fine(hipfftExecC2C(plan, d, d, HIPFFT_FORWARD), "forward 2D");
  (void)hipMemcpy(h.data(), d, all * sizeof(hipfftComplex), hipMemcpyDeviceToHost);
  check("C2C 2D forward, 16 x 24", good && close(h, dft2(in, rows, cols, -1), 1e-5));
  (void)hipfftDestroy(plan);
  (void)hipFree(d);
}

// Real to complex of `n` points, `batch` of them: n/2 + 1 outputs each, the
// rest being their conjugates. Then back, to n times the input.
void r2c(int n, int batch) {
  const int out_n = n / 2 + 1;
  std::vector<float> h(size_t(n) * batch);
  for (size_t i = 0; i < h.size(); ++i) h[i] = float(value(5 * i));
  float* d_in;
  hipfftComplex* d_out;
  hipfftHandle fwd, back;
  if (hipMalloc(&d_in, h.size() * sizeof(float)) != hipSuccess ||
      hipMalloc(&d_out, size_t(out_n) * batch * sizeof(hipfftComplex)) != hipSuccess)
    return check("memory", false);
  (void)hipMemcpy(d_in, h.data(), h.size() * sizeof(float), hipMemcpyHostToDevice);
  bool good = fine(hipfftPlan1d(&fwd, n, HIPFFT_R2C, batch), "plan R2C") &&
              fine(hipfftPlan1d(&back, n, HIPFFT_C2R, batch), "plan C2R") &&
              fine(hipfftExecR2C(fwd, d_in, d_out), "R2C");
  std::vector<hipfftComplex> out(size_t(out_n) * batch);
  (void)hipMemcpy(out.data(), d_out, out.size() * sizeof(hipfftComplex), hipMemcpyDeviceToHost);
  bool forward = good;
  for (int b = 0; b < batch && forward; ++b) {
    std::vector<cd> x(n);
    for (int i = 0; i < n; ++i) x[i] = h[size_t(b) * n + i];
    std::vector<cd> want = dft(x, -1);
    want.resize(out_n);
    forward = close(std::vector<hipfftComplex>(out.begin() + b * out_n, out.begin() + (b + 1) * out_n), want, 1e-5);
  }
  check("R2C, 100 points x 2", forward);
  good = good && fine(hipfftExecC2R(back, d_out, d_in), "C2R");
  std::vector<float> round(h.size());
  (void)hipMemcpy(round.data(), d_in, round.size() * sizeof(float), hipMemcpyDeviceToHost);
  double err = 0;
  for (size_t i = 0; i < h.size(); ++i) err = std::max(err, std::abs(double(round[i]) - double(h[i]) * n));
  check("C2R back to n times the input", good && err <= 1e-4 * n);
  (void)hipfftDestroy(fwd);
  (void)hipfftDestroy(back);
  (void)hipFree(d_in);
  (void)hipFree(d_out);
}

void z2z(int n) {
  std::vector<hipfftDoubleComplex> h(n);
  std::vector<cd> in(n);
  for (int i = 0; i < n; ++i) {
    in[i] = cd(value(7 * i), value(7 * i + 3));
    h[i] = {in[i].real(), in[i].imag()};
  }
  hipfftDoubleComplex* d;
  hipfftHandle plan;
  if (hipMalloc(&d, n * sizeof(hipfftDoubleComplex)) != hipSuccess) return check("memory", false);
  (void)hipMemcpy(d, h.data(), n * sizeof(hipfftDoubleComplex), hipMemcpyHostToDevice);
  const bool good =
      fine(hipfftPlan1d(&plan, n, HIPFFT_Z2Z, 1), "plan Z2Z") && fine(hipfftExecZ2Z(plan, d, d, HIPFFT_FORWARD), "Z2Z");
  (void)hipMemcpy(h.data(), d, n * sizeof(hipfftDoubleComplex), hipMemcpyDeviceToHost);
  check("Z2Z forward, 128 points", good && close(h, dft(in, -1), 1e-12));
  (void)hipfftDestroy(plan);
  (void)hipFree(d);
}

}  // namespace

int main() {
  c2c(256, 3);
  c2c(100, 3);
  c2c_2d(16, 24);
  r2c(100, 2);
  z2z(128);
  std::printf("hipFFT: %d of %d transforms match the host's\n", ok, total);
  return ok == total ? 0 : 1;
}
