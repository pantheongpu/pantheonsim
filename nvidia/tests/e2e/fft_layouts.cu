// cuFFT's layouts and its cufftXt API, checked against a direct DFT.
//
// The conformance test (nvidia/tests/conformance/cufft_transforms.cu) is a
// differential one: it needs hardware to compare against. This one checks
// itself, so it runs in CI: every transform here is compared, element by
// element, with an O(n^2) DFT computed on the host, which shares no code with
// the shim's Cooley-Tukey.
//
// What it covers is what PyTorch's torch.fft asks for: cufftXtMakePlanMany and
// cufftXtExec, strided and padded multi-dimensional layouts, the complex-to-real
// inverse of a 2-D and 3-D spectrum (whose Hermitian symmetry spans every axis,
// not just the fastest), and half precision.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

using cd = std::complex<double>;

static int failures = 0;

static void check(bool ok, const char* what, double err) {
  std::printf("%-4s %s (max error %.2e)\n", ok ? "ok" : "FAIL", what, err);
  if (!ok) ++failures;
}

#define CF(x)                                                              \
  do {                                                                     \
    cufftResult r_ = (x);                                                  \
    if (r_ != CUFFT_SUCCESS) {                                             \
      std::printf("FAIL %s returned %d\n", #x, (int)r_);                  \
      ++failures;                                                          \
      return;                                                              \
    }                                                                      \
  } while (0)

static double sig(int i) { return std::sin(0.37 * i) + 0.5 * std::cos(0.11 * i) + 0.05 * (i % 9); }

// The reference: a direct DFT along each axis of a packed array.
static std::vector<cd> dft(std::vector<cd> a, const std::vector<int>& n, int sign) {
  size_t inner = 1;
  for (size_t d = n.size(); d-- > 0;) {
    const int len = n[d];
    const size_t outer = a.size() / (inner * len);
    std::vector<cd> line(len);
    for (size_t o = 0; o < outer; ++o)
      for (size_t i = 0; i < inner; ++i) {
        const size_t base = o * inner * len + i;
        for (int k = 0; k < len; ++k) {
          cd acc = 0;
          for (int j = 0; j < len; ++j)
            acc += a[base + j * inner] * std::polar(1.0, sign * 2 * M_PI * (double)j * k / len);
          line[k] = acc;
        }
        for (int k = 0; k < len; ++k) a[base + k * inner] = line[k];
      }
    inner *= len;
  }
  return a;
}

static size_t prod(const std::vector<int>& n) {
  size_t p = 1;
  for (int d : n) p *= d;
  return p;
}

// The stored half of a real transform: the fastest axis cut to n/2+1.
static std::vector<cd> halve(const std::vector<cd>& full, const std::vector<int>& n) {
  const int last = n.back(), h = last / 2 + 1;
  std::vector<cd> out;
  for (size_t l = 0; l < full.size() / last; ++l)
    for (int k = 0; k < h; ++k) out.push_back(full[l * last + k]);
  return out;
}

template <class T> T* upload(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(T));
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> std::vector<T> download(const T* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

// Relative to the largest reference magnitude, so one tolerance fits every size.
template <class Get>
static double error(const std::vector<cd>& ref, Get got) {
  double scale = 1e-30, err = 0;
  for (const cd& v : ref) scale = std::fmax(scale, std::abs(v));
  for (size_t i = 0; i < ref.size(); ++i) err = std::fmax(err, std::abs(got(i) - ref[i]));
  return err / scale;
}

// The packed 2-D and 3-D complex transforms, through the classic API.
static void packed_c2c() {
  const std::vector<int> n = {6, 10};
  std::vector<cufftComplex> h(prod(n));
  std::vector<cd> x(h.size());
  for (size_t i = 0; i < h.size(); ++i) {
    x[i] = cd(sig(i), 0.3 * sig(i + 7));
    h[i] = make_cuComplex((float)x[i].real(), (float)x[i].imag());
  }
  cufftComplex* d = upload(h);
  cufftHandle p;
  CF(cufftPlan2d(&p, n[0], n[1], CUFFT_C2C));
  CF(cufftExecC2C(p, d, d, CUFFT_INVERSE));
  auto out = download(d, h.size());
  const auto ref = dft(x, n, +1);
  const double e = error(ref, [&](size_t i) { return cd(out[i].x, out[i].y); });
  check(e < 1e-5, "2-D 6x10 C2C inverse matches the DFT", e);
  cufftDestroy(p);
  cudaFree(d);
}

// Real to complex and back in 2-D and 3-D. The C2R inverse is the one that
// needs the spectrum's symmetry across every axis.
static void real_round_trip(const std::vector<int>& n, const char* what_fwd,
                            const char* what_inv) {
  const size_t total = prod(n), half = total / n.back() * (n.back() / 2 + 1);
  std::vector<float> h(total);
  std::vector<cd> x(total);
  for (size_t i = 0; i < total; ++i) { h[i] = (float)sig(i); x[i] = h[i]; }
  float* dr = upload(h);
  cufftComplex* dc = nullptr;
  cudaMalloc(&dc, half * sizeof(cufftComplex));
  cufftHandle fwd, inv;
  const int rank = (int)n.size();
  std::vector<int> nn = n;
  CF(cufftPlanMany(&fwd, rank, nn.data(), nullptr, 1, 0, nullptr, 1, 0, CUFFT_R2C, 1));
  CF(cufftPlanMany(&inv, rank, nn.data(), nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2R, 1));
  CF(cufftExecR2C(fwd, dr, dc));
  auto spec = download(dc, half);
  const auto ref = halve(dft(x, n, -1), n);
  double e = error(ref, [&](size_t i) { return cd(spec[i].x, spec[i].y); });
  check(e < 1e-5, what_fwd, e);
  CF(cufftExecC2R(inv, dc, dr));
  auto back = download(dr, total);
  std::vector<cd> scaled(total);
  for (size_t i = 0; i < total; ++i) scaled[i] = x[i] * (double)total;
  e = error(scaled, [&](size_t i) { return cd(back[i], 0); });
  check(e < 1e-5, what_inv, e);
  cufftDestroy(fwd);
  cufftDestroy(inv);
  cudaFree(dr);
  cudaFree(dc);
}

// A batch of 2-D transforms in a padded, strided layout, the way a slice of a
// larger tensor is laid out: rows padded to 7 and 9 complex elements, every
// element 2 apart, batches 200 apart. The elements the layout skips must come
// through untouched.
static void padded_strided() {
  const std::vector<int> n = {4, 5};
  int nn[2] = {4, 5}, iemb[2] = {4, 7}, oemb[2] = {4, 9};
  const int istride = 2, ostride = 2, idist = 70, odist = 80, batch = 2;
  const size_t ispan = (batch - 1) * idist + ((3 * 7 + 4) * istride) + 1;
  const size_t ospan = (batch - 1) * odist + ((3 * 9 + 4) * ostride) + 1;
  std::vector<cufftComplex> hin(ispan), hout(ospan);
  for (size_t i = 0; i < ispan; ++i) hin[i] = make_cuComplex((float)sig(i), (float)sig(i + 1));
  for (size_t i = 0; i < ospan; ++i) hout[i] = make_cuComplex(-1234.0f, 4321.0f);
  cufftComplex* din = upload(hin);
  cufftComplex* dout = upload(hout);
  cufftHandle p;
  CF(cufftPlanMany(&p, 2, nn, iemb, istride, idist, oemb, ostride, odist, CUFFT_C2C, batch));
  CF(cufftExecC2C(p, din, dout, CUFFT_FORWARD));
  auto out = download(dout, ospan);
  std::vector<bool> written(ospan, false);
  double err = 0;
  for (int b = 0; b < batch; ++b) {
    std::vector<cd> x;
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 5; ++c) {
        const auto v = hin[b * idist + (r * 7 + c) * istride];
        x.push_back(cd(v.x, v.y));
      }
    const auto ref = dft(x, n, -1);
    err = std::fmax(err, error(ref, [&](size_t i) {
      const size_t at = b * odist + ((i / 5) * 9 + i % 5) * ostride;
      written[at] = true;
      return cd(out[at].x, out[at].y);
    }));
  }
  check(err < 1e-5, "batched 2-D C2C in a padded, strided layout matches the DFT", err);
  bool kept = true;
  for (size_t i = 0; i < ospan; ++i)
    if (!written[i] && (out[i].x != -1234.0f || out[i].y != 4321.0f)) kept = false;
  check(kept, "the elements a strided output skips keep what they held", 0);
  cufftDestroy(p);
  cudaFree(din);
  cudaFree(dout);
}

// torch.fft.rfft2 and irfft2 on a contiguous float tensor: cufftXt plans with
// explicit embeds, a batch, and the direction passed to cufftXtExec.
static void xt_rfft2() {
  const std::vector<int> n = {8, 6};
  const int batch = 3;
  const size_t total = prod(n), half = 8 * 4;
  long long ln[2] = {8, 6}, iemb[2] = {8, 6}, oemb[2] = {8, 4};
  std::vector<float> h(total * batch);
  for (size_t i = 0; i < h.size(); ++i) h[i] = (float)sig(i * 3);
  float* dr = upload(h);
  cufftComplex* dc = nullptr;
  cudaMalloc(&dc, half * batch * sizeof(cufftComplex));
  cufftHandle fwd, inv;
  size_t work = 1;
  CF(cufftCreate(&fwd));
  CF(cufftCreate(&inv));
  CF(cufftSetAutoAllocation(fwd, 0));
  CF(cufftXtMakePlanMany(fwd, 2, ln, iemb, 1, (long long)total, CUDA_R_32F, oemb, 1,
                         (long long)half, CUDA_C_32F, batch, &work, CUDA_C_32F));
  CF(cufftXtMakePlanMany(inv, 2, ln, oemb, 1, (long long)half, CUDA_C_32F, iemb, 1,
                         (long long)total, CUDA_R_32F, batch, &work, CUDA_C_32F));
  CF(cufftSetStream(fwd, 0));
  CF(cufftXtExec(fwd, dr, dc, CUFFT_FORWARD));
  auto spec = download(dc, half * batch);
  double err = 0;
  for (int b = 0; b < batch; ++b) {
    std::vector<cd> x(h.begin() + b * total, h.begin() + (b + 1) * total);
    const auto ref = halve(dft(x, n, -1), n);
    err = std::fmax(err, error(ref, [&](size_t i) {
      const auto v = spec[b * half + i];
      return cd(v.x, v.y);
    }));
  }
  check(err < 1e-5, "cufftXt batched 2-D R2C (rfft2) matches the DFT", err);
  CF(cufftXtExec(inv, dc, dr, CUFFT_INVERSE));
  auto back = download(dr, total * batch);
  err = 0;
  for (size_t i = 0; i < back.size(); ++i)
    err = std::fmax(err, std::fabs(back[i] / (double)total - h[i]));
  check(err < 1e-5, "cufftXt batched 2-D C2R (irfft2) inverts it", err);
  cufftDestroy(fwd);
  cufftDestroy(inv);
  cudaFree(dr);
  cudaFree(dc);
}

// Double and half precision through cufftXt, and a pair of types cuFFT has no
// transform for.
static void xt_precisions() {
  {
    const std::vector<int> n = {12};
    std::vector<cufftDoubleComplex> h(12);
    std::vector<cd> x(12);
    for (int i = 0; i < 12; ++i) { x[i] = cd(sig(i), -sig(i + 2)); h[i] = {x[i].real(), x[i].imag()}; }
    cufftDoubleComplex* d = upload(h);
    cufftHandle p;
    long long ln = 12;
    size_t work;
    CF(cufftCreate(&p));
    CF(cufftXtMakePlanMany(p, 1, &ln, nullptr, 1, 0, CUDA_C_64F, nullptr, 1, 0, CUDA_C_64F, 1,
                           &work, CUDA_C_64F));
    CF(cufftXtExec(p, d, d, CUFFT_FORWARD));
    auto out = download(d, 12);
    const auto ref = dft(x, n, -1);
    const double e = error(ref, [&](size_t i) { return cd(out[i].x, out[i].y); });
    check(e < 1e-12, "cufftXt Z2Z matches the DFT to double precision", e);
    cufftDestroy(p);
    cudaFree(d);
  }
  {
    const std::vector<int> n = {16};
    std::vector<__half2> h(16);
    std::vector<cd> x(16);
    for (int i = 0; i < 16; ++i) {
      h[i] = __floats2half2_rn((float)sig(i), (float)sig(i + 5));
      x[i] = cd(__low2float(h[i]), __high2float(h[i]));  // the values as stored
    }
    __half2* d = upload(h);
    cufftHandle p;
    long long ln = 16;
    size_t work;
    CF(cufftCreate(&p));
    CF(cufftXtMakePlanMany(p, 1, &ln, nullptr, 1, 0, CUDA_C_16F, nullptr, 1, 0, CUDA_C_16F, 1,
                           &work, CUDA_C_16F));
    CF(cufftXtExec(p, d, d, CUFFT_FORWARD));
    auto out = download(d, 16);
    const auto ref = dft(x, n, -1);
    const double e = error(ref, [&](size_t i) {
      return cd(__low2float(out[i]), __high2float(out[i]));
    });
    check(e < 2e-3, "cufftXt half-precision C2C matches the DFT to half rounding", e);
    // cufftExecC2C runs a half-precision C2C plan too, at the plan's
    // precision, as it does on hardware: the inverse brings back 16 x.
    CF(cufftExecC2C(p, (cufftComplex*)d, (cufftComplex*)d, CUFFT_INVERSE));
    auto back = download(d, 16);
    std::vector<cd> scaled(16);
    for (int i = 0; i < 16; ++i) scaled[i] = x[i] * 16.0;
    const double r = error(scaled, [&](size_t i) { return cd(__low2float(back[i]), __high2float(back[i])); });
    check(r < 4e-3, "cufftExecC2C runs a half plan at its precision: the inverse returns 16 x", r);
    cufftDestroy(p);
    cudaFree(d);
  }
  {
    cufftHandle p;
    long long ln = 8;
    size_t work;
    CF(cufftCreate(&p));
    const cufftResult r = cufftXtMakePlanMany(p, 1, &ln, nullptr, 1, 0, CUDA_R_32F, nullptr, 1, 0,
                                              CUDA_R_32F, 1, &work, CUDA_C_32F);
    check(r == CUFFT_INVALID_TYPE, "cufftXt refuses real-to-real", 0);
    cufftDestroy(p);
  }
}

int main() {
  packed_c2c();
  real_round_trip({6, 8}, "2-D 6x8 R2C matches the DFT", "2-D 6x8 C2R inverts it");
  real_round_trip({3, 5, 6}, "3-D 3x5x6 R2C matches the DFT", "3-D 3x5x6 C2R inverts it");
  real_round_trip({4, 7}, "2-D 4x7 (odd fastest axis) R2C matches the DFT",
                  "2-D 4x7 (odd fastest axis) C2R inverts it");
  padded_strided();
  xt_rfft2();
  xt_precisions();
  std::printf(failures ? "FAIL: %d cuFFT checks\n" : "PASS: every cuFFT check\n", failures);
  return failures ? 1 : 0;
}
