// cuFFT across several GPUs: cufftXtSetGPUs, cufftXtMalloc / cufftXtMemcpy /
// cufftXtFree descriptors and cufftXtExecDescriptor*, checked against a direct
// DFT on the host and against the layouts two RTX 3060s showed with CUDA
// 13.0's cuFFT, which this program passes against too:
//
//   batched       whole transforms dealt out, the first batch % G GPUs taking
//                 one more; output natural, in place.
//   2-D and 3-D   natural order splits x (the first nx % G GPUs one plane
//                 more); a transform leaves the data split on y
//                 (CUFFT_XT_FORMAT_INPLACE_SHUFFLED), [x][its y][z] on each
//                 GPU, and the next transform brings it back.
//   1-D           the output is in strings (cuFFT documentation,
//                 permuted2Linear), from cufftXtQueryPlan's factors;
//                 CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED takes the input
//                 redistributed.
//
// Each descriptor part is checked to live on its own device. The program
// needs two devices (the RTX 3060 pair, or VGPU_DEVICE_COUNT=2) and uses the
// first two.
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <vector>

using cd = std::complex<double>;

static int failures = 0;
static int gpus[2] = {0, 1};

static void check(bool ok, const char* what, double err = 0) {
  std::printf("%-4s %s (%.2e)\n", ok ? "ok" : "FAIL", what, err);
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

// A direct DFT along every axis of a packed array.
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
            acc += a[base + j * inner] *
                   std::polar(1.0, sign * 2 * M_PI * (double)((long long)j * k % len) / len);
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

static double scale_of(const std::vector<cd>& v) {
  double s = 0;
  for (const cd& x : v) s = std::max(s, std::abs(x));
  return s > 0 ? s : 1;
}

// One GPU's part of a descriptor, read straight from that device's memory, as
// complex numbers of the plan's precision.
template <class T>
static std::vector<cd> part(const cudaLibXtDesc* d, int g) {
  std::vector<T> raw(d->descriptor->size[g] / sizeof(T));
  cudaMemcpy(raw.data(), d->descriptor->data[g], raw.size() * sizeof(T), cudaMemcpyDefault);
  std::vector<cd> out;
  for (size_t i = 0; i + 1 < raw.size(); i += 2) out.push_back(cd(raw[i], raw[i + 1]));
  return out;
}

static bool on_own_devices(const cudaLibXtDesc* d) {
  if (d->descriptor->nGPUs != 2) return false;
  for (int g = 0; g < 2; ++g) {
    cudaPointerAttributes a{};
    if (d->descriptor->GPUs[g] != gpus[g]) return false;
    if (cudaPointerGetAttributes(&a, d->descriptor->data[g]) != cudaSuccess) return false;
    if (a.type != cudaMemoryTypeDevice || a.device != gpus[g]) return false;
  }
  return true;
}

static cufftResult make(cufftHandle* h, std::vector<long long> n, cudaDataType in, cudaDataType out,
                        long long batch, size_t* ws) {
  cufftResult r = cufftCreate(h);
  if (r == CUFFT_SUCCESS) r = cufftXtSetGPUs(*h, 2, gpus);
  if (r != CUFFT_SUCCESS) return r;
  const cudaDataType exec = in == CUDA_C_64F || out == CUDA_C_64F   ? CUDA_C_64F
                            : in == CUDA_C_16F || out == CUDA_C_16F ? CUDA_C_16F
                                                                    : CUDA_C_32F;
  return cufftXtMakePlanMany(*h, (int)n.size(), n.data(), nullptr, 1, 0, in, nullptr, 1, 0, out,
                             batch, ws, exec);
}

// What cufftXtSetGPUs and planning refuse, with NVIDIA's status for each.
static void plan_rules() {
  cufftHandle h;
  cufftCreate(&h);
  int one[1] = {0}, bad[2] = {0, 99};
  check(cufftXtSetGPUs(h, 1, one) == CUFFT_INVALID_VALUE, "cufftXtSetGPUs refuses one GPU");
  check(cufftXtSetGPUs(h, 2, bad) == CUFFT_INVALID_DEVICE, "cufftXtSetGPUs refuses a device that does not exist");
  check(cufftXtSetGPUs(h, 2, nullptr) == CUFFT_INVALID_VALUE, "cufftXtSetGPUs refuses a NULL list");
  check(cufftXtSetGPUs(h, 2, gpus) == CUFFT_SUCCESS, "cufftXtSetGPUs takes two GPUs");
  check(cufftMakePlan2d(h, 64, 64, CUFFT_C2C, nullptr) == CUFFT_INVALID_VALUE,
        "a multi-GPU plan needs somewhere to put its work sizes");
  size_t ws[2] = {7, 7};
  check(cufftMakePlan2d(h, 64, 64, CUFFT_C2C, ws) == CUFFT_SUCCESS, "multi-GPU 64x64 C2C plan");
  check(cufftXtSetGPUs(h, 2, gpus) == CUFFT_INVALID_PLAN, "cufftXtSetGPUs after planning is INVALID_PLAN");
  {
    cudaLibXtDesc* d = nullptr;
    void* p = nullptr;
    cudaMalloc(&p, 64 * 64 * 8);
    check(cufftExecC2C(h, (cufftComplex*)p, (cufftComplex*)p, CUFFT_FORWARD) == CUFFT_INTERNAL_ERROR,
          "cufftExecC2C on a multi-GPU plan is INTERNAL_ERROR");
    check(cufftXtMalloc(h, &d, CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED) == CUFFT_INVALID_PLAN,
          "a 2-D plan has no 1-D input-shuffled descriptor");
    cudaFree(p);
  }
  check(cufftSetWorkArea(h, nullptr) == CUFFT_INVALID_PLAN, "cufftSetWorkArea is not for multi-GPU plans");
  cufftDestroy(h);

  struct Case { std::vector<long long> n; cudaDataType in, out; long long batch; cufftResult want; const char* what; };
  const Case cases[] = {
      {{32}, CUDA_C_32F, CUDA_C_32F, 1, CUFFT_INVALID_SIZE, "a single 1-D transform under 64 points is refused"},
      {{96}, CUDA_C_32F, CUDA_C_32F, 1, CUFFT_INVALID_SIZE, "a single 1-D transform not a power of two is refused"},
      {{1024}, CUDA_R_32F, CUDA_C_32F, 1, CUFFT_INVALID_SIZE, "a single 1-D R2C is refused"},
      {{16, 64}, CUDA_C_32F, CUDA_C_32F, 1, CUFFT_INVALID_SIZE, "a single 2-D transform needs x and y of 32 or more"},
      {{3, 32, 32}, CUDA_C_32F, CUDA_C_32F, 1, CUFFT_INVALID_SIZE, "a single 3-D transform needs x of 32 or more"},
      {{4099, 32}, CUDA_C_32F, CUDA_C_32F, 1, CUFFT_SUCCESS, "an axis past 4096 with a prime over 127 is planned all the same"},
      {{32, 32, 2}, CUDA_C_32F, CUDA_C_32F, 1, CUFFT_SUCCESS, "a 3-D transform's z may be small"},
      {{1, 1, 64}, CUDA_C_32F, CUDA_C_32F, 1, CUFFT_SUCCESS, "axes of size 1 are dropped: 1x1x64 is a 1-D plan"},
      {{1024}, CUDA_C_16F, CUDA_C_16F, 1, CUFFT_SETUP_FAILED, "half precision is SETUP_FAILED"},
      {{7}, CUDA_R_32F, CUDA_C_32F, 3, CUFFT_SUCCESS, "batched plans take any size"},
  };
  for (const Case& c : cases) {
    size_t w[2];
    const cufftResult r = make(&h, c.n, c.in, c.out, c.batch, w);
    check(r == c.want, c.what, (double)r);
    cufftDestroy(h);
  }
}

// Batched: transforms dealt out whole, results in natural order.
static void batched() {
  const int n = 8, batch = 3;
  cufftHandle h;
  size_t ws[2];
  CF(make(&h, {n}, CUDA_C_32F, CUDA_C_32F, batch, ws));
  cudaLibXtDesc* d = nullptr;
  CF(cufftXtMalloc(h, &d, CUFFT_XT_FORMAT_INPLACE));
  check(on_own_devices(d), "batched descriptor: each part on its own GPU");
  check(d->descriptor->size[0] == 2 * n * 8 && d->descriptor->size[1] == 1 * n * 8,
        "3 transforms on 2 GPUs: 2 on the first, 1 on the second");
  cudaLibXtDesc* none = nullptr;
  check(cufftXtMalloc(h, &none, CUFFT_XT_FORMAT_INPLACE_SHUFFLED) == CUFFT_INVALID_VALUE,
        "a batched plan has no shuffled descriptor");
  std::vector<float2> host(n * batch);
  std::vector<cd> in(n * batch);
  for (int i = 0; i < n * batch; ++i) {
    in[i] = cd(sig(i), sig(i + 50));
    host[i] = {(float)in[i].real(), (float)in[i].imag()};
  }
  CF(cufftXtMemcpy(h, d, host.data(), CUFFT_COPY_HOST_TO_DEVICE));
  CF(cufftXtExecDescriptorC2C(h, d, d, CUFFT_FORWARD));
  check(d->subFormat == CUFFT_XT_FORMAT_INPLACE, "a batched result stays in natural order");
  std::vector<cd> ref;
  for (int b = 0; b < batch; ++b) {
    auto o = dft(std::vector<cd>(in.begin() + b * n, in.begin() + (b + 1) * n), {n}, -1);
    ref.insert(ref.end(), o.begin(), o.end());
  }
  const double s = scale_of(ref);
  double e = 0;
  for (int g = 0; g < 2; ++g) {
    const auto v = part<float>(d, g);
    for (size_t i = 0; i < v.size(); ++i) e = std::max(e, std::abs(v[i] - ref[g * 2 * n + i]) / s);
  }
  check(e < 1e-5, "batched C2C on two GPUs: each GPU holds its transforms' spectra", e);
  std::vector<float2> back(n * batch);
  CF(cufftXtMemcpy(h, back.data(), d, CUFFT_COPY_DEVICE_TO_HOST));
  e = 0;
  for (int i = 0; i < n * batch; ++i) e = std::max(e, std::abs(cd(back[i].x, back[i].y) - ref[i]) / s);
  check(e < 1e-5, "cufftXtMemcpy gathers the batched result", e);
  CF(cufftXtFree(d));
  cufftDestroy(h);

  // R2C in place: the real input padded to the stored half's width.
  CF(make(&h, {n}, CUDA_R_32F, CUDA_C_32F, batch, ws));
  cudaLibXtDesc *dr = nullptr, *packed = nullptr;
  CF(cufftXtMalloc(h, &dr, CUFFT_XT_FORMAT_INPLACE));
  CF(cufftXtMalloc(h, &packed, CUFFT_XT_FORMAT_INPUT));
  const int half = n / 2 + 1;
  check(dr->descriptor->size[0] == 2 * half * 8 && packed->descriptor->size[0] == 2 * n * 4,
        "a batched R2C in-place descriptor is padded, an input one packed");
  std::vector<float> rin(batch * 2 * half, 0.f);
  for (int b = 0; b < batch; ++b)
    for (int k = 0; k < n; ++k) rin[b * 2 * half + k] = (float)sig(b * n + k);
  CF(cufftXtMemcpy(h, dr, rin.data(), CUFFT_COPY_HOST_TO_DEVICE));
  CF(cufftXtExecDescriptorR2C(h, dr, dr));
  std::vector<float2> rout(batch * half);
  CF(cufftXtMemcpy(h, rout.data(), dr, CUFFT_COPY_DEVICE_TO_HOST));
  e = 0;
  for (int b = 0; b < batch; ++b) {
    std::vector<cd> x(n);
    for (int k = 0; k < n; ++k) x[k] = sig(b * n + k);
    const auto o = dft(x, {n}, -1);
    for (int k = 0; k < half; ++k)
      e = std::max(e, std::abs(cd(rout[b * half + k].x, rout[b * half + k].y) - o[k]) / scale_of(o));
  }
  check(e < 1e-5, "batched R2C in place on two GPUs matches the DFT", e);
  cufftXtFree(dr);
  cufftXtFree(packed);
  cufftDestroy(h);
}

// One 2-D or 3-D transform: x split before, y split after.
template <class T>
static void slabs(std::vector<int> n, const char* what) {
  const bool dbl = sizeof(T) == 8;
  const cudaDataType ct = dbl ? CUDA_C_64F : CUDA_C_32F;
  cufftHandle h;
  size_t ws[2];
  CF(make(&h, std::vector<long long>(n.begin(), n.end()), ct, ct, 1, ws));
  cudaLibXtDesc* d = nullptr;
  CF(cufftXtMalloc(h, &d, CUFFT_XT_FORMAT_INPLACE));
  check(on_own_devices(d), "single transform descriptor: each part on its own GPU");
  const size_t total = prod(n), inner = total / n[0] / n[1];
  const int x0 = (n[0] + 1) / 2, y0 = (n[1] + 1) / 2;  // the first GPU's share
  std::vector<cd> in(total);
  std::vector<T> host(2 * total);
  for (size_t i = 0; i < total; ++i) {
    in[i] = cd(sig((int)i), sig((int)i + 7));
    host[2 * i] = (T)in[i].real();
    host[2 * i + 1] = (T)in[i].imag();
  }
  CF(cufftXtMemcpy(h, d, host.data(), CUFFT_COPY_HOST_TO_DEVICE));
  // Natural order: the first x0 planes on GPU 0, the rest on GPU 1.
  double e = 0;
  for (int g = 0; g < 2; ++g) {
    const auto v = part<T>(d, g);
    const size_t first = g ? (size_t)x0 * n[1] * inner : 0;
    const size_t count = (size_t)(g ? n[0] - x0 : x0) * n[1] * inner;
    for (size_t i = 0; i < count; ++i) e = std::max(e, std::abs(v[i] - in[first + i]));
  }
  check(e < 1e-6, "natural order: x split, first GPU one plane more", e);
  CF(cufftXtExecDescriptor(h, d, d, CUFFT_FORWARD));
  check(d->subFormat == CUFFT_XT_FORMAT_INPLACE_SHUFFLED, "a single transform leaves its output shuffled");
  const auto ref = dft(in, n, -1);
  const double s = scale_of(ref);
  e = 0;
  for (int g = 0; g < 2; ++g) {
    const auto v = part<T>(d, g);
    const int ys = g ? y0 : 0, yc = g ? n[1] - y0 : y0;
    size_t i = 0;
    for (int x = 0; x < n[0]; ++x)
      for (int y = ys; y < ys + yc; ++y)
        for (size_t r = 0; r < inner; ++r, ++i)
          e = std::max(e, std::abs(v[i] - ref[((size_t)x * n[1] + y) * inner + r]) / s);
  }
  check(e < (dbl ? 1e-12 : 1e-5), what, e);
  std::vector<T> back(2 * total);
  CF(cufftXtMemcpy(h, back.data(), d, CUFFT_COPY_DEVICE_TO_HOST));
  e = 0;
  for (size_t i = 0; i < total; ++i) e = std::max(e, std::abs(cd(back[2 * i], back[2 * i + 1]) - ref[i]) / s);
  check(e < (dbl ? 1e-12 : 1e-5), "cufftXtMemcpy returns a shuffled result in natural order", e);
  // Device to device: natural order in the destination.
  cudaLibXtDesc* nat = nullptr;
  CF(cufftXtMalloc(h, &nat, CUFFT_XT_FORMAT_INPLACE));
  CF(cufftXtMemcpy(h, nat, d, CUFFT_COPY_DEVICE_TO_DEVICE));
  const auto v0 = part<T>(nat, 0);
  e = 0;
  for (size_t i = 0; i < (size_t)x0 * n[1] * inner; ++i) e = std::max(e, std::abs(v0[i] - ref[i]) / s);
  check(e < (dbl ? 1e-12 : 1e-5) && nat->subFormat == CUFFT_XT_FORMAT_INPLACE,
        "device-to-device copy puts the spectrum in natural order", e);
  check(cufftXtMemcpy(h, d, nat, CUFFT_COPY_DEVICE_TO_DEVICE) == CUFFT_INTERNAL_ERROR,
        "a copy into a shuffled descriptor is INTERNAL_ERROR");
  check(cufftXtExecDescriptor(h, d, nat, CUFFT_INVERSE) == CUFFT_EXEC_FAILED,
        "a single multi-GPU transform out of place is EXEC_FAILED");
  // The inverse of the shuffled spectrum comes back in natural order.
  CF(cufftXtExecDescriptor(h, d, d, CUFFT_INVERSE));
  check(d->subFormat == CUFFT_XT_FORMAT_INPLACE, "the inverse of a shuffled spectrum is natural");
  CF(cufftXtMemcpy(h, back.data(), d, CUFFT_COPY_DEVICE_TO_HOST));
  e = 0;
  for (size_t i = 0; i < total; ++i)
    e = std::max(e, std::abs(cd(back[2 * i], back[2 * i + 1]) / (double)total - in[i]));
  check(e < (dbl ? 1e-12 : 1e-5), "forward then inverse returns the input times N", e);
  cufftXtFree(nat);
  cufftXtFree(d);
  cufftDestroy(h);
}

// A single 2-D R2C in place, and its C2R from the shuffled spectrum.
static void real_slabs() {
  const std::vector<int> n = {33, 34};
  const int half = n[1] / 2 + 1, rows = n[0];
  cufftHandle h;
  size_t ws[2];
  CF(make(&h, {33, 34}, CUDA_R_32F, CUDA_C_32F, 1, ws));
  cudaLibXtDesc* d = nullptr;
  CF(cufftXtMalloc(h, &d, CUFFT_XT_FORMAT_INPLACE));
  std::vector<float> host(rows * 2 * half, 0.f);
  std::vector<cd> x(rows * n[1]);
  for (int r = 0; r < rows; ++r)
    for (int k = 0; k < n[1]; ++k) {
      x[r * n[1] + k] = sig(r * n[1] + k);
      host[r * 2 * half + k] = (float)x[r * n[1] + k].real();
    }
  CF(cufftXtMemcpy(h, d, host.data(), CUFFT_COPY_HOST_TO_DEVICE));
  CF(cufftXtExecDescriptorR2C(h, d, d));
  check(d->subFormat == CUFFT_XT_FORMAT_INPLACE_SHUFFLED, "2-D R2C leaves its output shuffled");
  const auto full = dft(x, n, -1);
  const double s = scale_of(full);
  // Shuffled: the stored half's columns split, 9 and 8, all rows on each GPU.
  const int c0 = (half + 1) / 2;
  double e = 0;
  for (int g = 0; g < 2; ++g) {
    const auto v = part<float>(d, g);
    const int cs = g ? c0 : 0, cc = g ? half - c0 : c0;
    size_t i = 0;
    for (int r = 0; r < rows; ++r)
      for (int c = cs; c < cs + cc; ++c, ++i) e = std::max(e, std::abs(v[i] - full[r * n[1] + c]) / s);
  }
  check(e < 1e-5, "2-D R2C 33x34 on two GPUs: each holds its columns of the half spectrum", e);
  check(cufftXtExecDescriptorC2R(h, d, d) == CUFFT_INVALID_PLAN, "a C2R entry point on an R2C plan is INVALID_PLAN");
  cufftXtFree(d);
  cufftDestroy(h);

  // The C2R takes the spectrum shuffled: a 2-D C2R plan's natural descriptor
  // is empty, and the host's natural spectrum is dealt into the shuffled one.
  CF(make(&h, {33, 34}, CUDA_C_32F, CUDA_R_32F, 1, ws));
  cudaLibXtDesc *nat = nullptr, *sh = nullptr;
  CF(cufftXtMalloc(h, &nat, CUFFT_XT_FORMAT_INPLACE));
  CF(cufftXtMalloc(h, &sh, CUFFT_XT_FORMAT_INPLACE_SHUFFLED));
  check(nat->descriptor->size[0] == 0 && sh->descriptor->size[0] > 0,
        "a 2-D C2R plan allocates only its shuffled descriptor");
  std::vector<float2> spec(rows * half);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < half; ++c)
      spec[r * half + c] = {(float)full[r * n[1] + c].real(), (float)full[r * n[1] + c].imag()};
  CF(cufftXtMemcpy(h, sh, spec.data(), CUFFT_COPY_HOST_TO_DEVICE));
  e = 0;
  for (int g = 0; g < 2; ++g) {
    const auto v = part<float>(sh, g);
    const int cs = g ? c0 : 0, cc = g ? half - c0 : c0;
    size_t i = 0;
    for (int r = 0; r < rows; ++r)
      for (int c = cs; c < cs + cc; ++c, ++i) e = std::max(e, std::abs(v[i] - full[r * n[1] + c]) / s);
  }
  check(e < 1e-6, "the host spectrum lands in shuffled order", e);
  CF(cufftXtExecDescriptorC2R(h, sh, sh));
  check(sh->subFormat == CUFFT_XT_FORMAT_INPLACE, "C2R of a shuffled spectrum is natural");
  std::vector<float> back(rows * 2 * half);
  CF(cufftXtMemcpy(h, back.data(), sh, CUFFT_COPY_DEVICE_TO_HOST));
  e = 0;
  for (int r = 0; r < rows; ++r)
    for (int k = 0; k < n[1]; ++k)
      e = std::max(e, std::abs(back[r * 2 * half + k] / (double)(rows * n[1]) - x[r * n[1] + k].real()));
  check(e < 1e-5, "2-D C2R 33x34 on two GPUs inverts the R2C", e);
  cufftXtFree(nat);
  cufftXtFree(sh);
  cufftDestroy(h);
}

// One 1-D transform: the output in strings.
template <class T>
static void line(int n) {
  const bool dbl = sizeof(T) == 8;
  const cudaDataType ct = dbl ? CUDA_C_64F : CUDA_C_32F;
  cufftHandle h;
  size_t ws[2];
  CF(make(&h, {n}, ct, ct, 1, ws));
  cufftXt1dFactors f{};
  CF(cufftXtQueryPlan(h, &f, CUFFT_QUERY_1D_FACTORS));
  // What NVIDIA's chose for these sizes on two GPUs.
  const bool known = (n == 128 && f.factor1 == 16 && f.factor2 == 8 && f.stringCount == 8) ||
                     (n == 1024 && f.factor1 == 64 && f.factor2 == 16 && f.stringCount == 8) ||
                     (n == 64 && f.factor1 == 16 && f.factor2 == 4 && f.stringCount == 2);
  check(known && f.size == n && f.substringLength == n / f.stringCount / f.factor2 &&
            f.substringMask == f.substringLength - 1 && (1LL << f.stringShift) == f.stringLength,
        "cufftXtQueryPlan: NVIDIA's 1-D factors");
  cudaLibXtDesc* d = nullptr;
  CF(cufftXtMalloc(h, &d, CUFFT_XT_FORMAT_INPLACE));
  check(on_own_devices(d) && d->descriptor->size[0] == (size_t)n / 2 * 2 * sizeof(T),
        "1-D descriptor: half the points on each GPU");
  std::vector<cd> in(n);
  std::vector<T> host(2 * n);
  for (int i = 0; i < n; ++i) {
    in[i] = cd(sig(i), sig(3 * i + 1));
    host[2 * i] = (T)in[i].real();
    host[2 * i + 1] = (T)in[i].imag();
  }
  CF(cufftXtMemcpy(h, d, host.data(), CUFFT_COPY_HOST_TO_DEVICE));
  CF(cufftXtExecDescriptor(h, d, d, CUFFT_FORWARD));
  check(d->subFormat == CUFFT_XT_FORMAT_INPLACE_SHUFFLED, "1-D output is shuffled");
  const auto ref = dft(in, {n}, -1);
  const double s = scale_of(ref);
  // The documentation's permuted2Linear, from the queried factors.
  double e = 0;
  for (int g = 0; g < 2; ++g) {
    const auto v = part<T>(d, g);
    for (size_t i = 0; i < v.size(); ++i) {
      const size_t in_sub = i & f.substringMask;
      const size_t sub = (i >> f.substringShift) & f.factor2Mask;
      size_t str = (i >> f.stringShift) & f.stringMask;
      if (g) str += f.stringCount / 2;
      const size_t lin = in_sub + (str << f.substringShift) + (sub << f.factor1Shift);
      e = std::max(e, std::abs(v[i] - ref[lin]) / s);
    }
  }
  check(e < (dbl ? 1e-12 : 1e-5), "1-D output in strings, as permuted2Linear maps them", e);
  std::vector<T> back(2 * n);
  CF(cufftXtMemcpy(h, back.data(), d, CUFFT_COPY_DEVICE_TO_HOST));
  e = 0;
  for (int i = 0; i < n; ++i) e = std::max(e, std::abs(cd(back[2 * i], back[2 * i + 1]) - ref[i]) / s);
  check(e < (dbl ? 1e-12 : 1e-5), "cufftXtMemcpy returns the 1-D result in natural order", e);
  check(cufftXtExecDescriptor(h, d, d, CUFFT_INVERSE) == CUFFT_INVALID_TYPE,
        "a 1-D transform of string-ordered data is INVALID_TYPE");
  check(cufftXtMemcpy(h, d, host.data(), CUFFT_COPY_HOST_TO_DEVICE) == CUFFT_INVALID_TYPE,
        "the host cannot write a string-ordered descriptor");
  // Back to natural order on the devices, then the inverse.
  cudaLibXtDesc* nat = nullptr;
  CF(cufftXtMalloc(h, &nat, CUFFT_XT_FORMAT_INPLACE));
  CF(cufftXtMemcpy(h, nat, d, CUFFT_COPY_DEVICE_TO_DEVICE));
  CF(cufftXtExecDescriptor(h, nat, nat, CUFFT_INVERSE));
  CF(cufftXtMemcpy(h, back.data(), nat, CUFFT_COPY_DEVICE_TO_HOST));
  e = 0;
  for (int i = 0; i < n; ++i) e = std::max(e, std::abs(cd(back[2 * i], back[2 * i + 1]) / (double)n - in[i]));
  check(e < (dbl ? 1e-12 : 1e-5), "1-D forward, device-to-device copy, inverse returns the input", e);
  // The input already redistributed: point a + factor2*b on the GPU owning a.
  cudaLibXtDesc* pre = nullptr;
  CF(cufftXtMalloc(h, &pre, CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED));
  CF(cufftXtMemcpy(h, pre, host.data(), CUFFT_COPY_HOST_TO_DEVICE));
  const long long per = f.factor2 / 2, S = std::max<long long>(1, f.stringCount / 2);
  const long long q = std::max<long long>(1, per / S);
  e = 0;
  for (int g = 0; g < 2; ++g) {
    const auto v = part<T>(pre, g);
    for (size_t i = 0; i < v.size(); ++i) {
      const long long st = i / (q * f.factor1), rem = i % (q * f.factor1);
      const long long a = g * per + st * q + rem % q, b = rem / q;
      e = std::max(e, std::abs(v[i] - in[a + f.factor2 * b]));
    }
  }
  check(e < 1e-6, "1-D input-shuffled descriptor: the first pass's order", e);
  CF(cufftXtExecDescriptor(h, pre, pre, CUFFT_FORWARD));
  const auto v0 = part<T>(pre, 0), w0 = part<T>(d, 0);
  e = 0;
  for (size_t i = 0; i < v0.size(); ++i) e = std::max(e, std::abs(v0[i] - w0[i]) / s);
  check(e < (dbl ? 1e-12 : 1e-5) && pre->subFormat == CUFFT_XT_FORMAT_INPLACE_SHUFFLED,
        "a transform of input-shuffled data gives the same strings", e);
  cufftXtFree(pre);
  cufftXtFree(nat);
  cufftXtFree(d);
  cufftDestroy(h);
}

// A single-GPU plan's descriptor, and the callback entry points libcufft.so
// has (legacy callbacks are only in the static library).
static void odds() {
  cufftHandle h;
  CF(cufftPlan1d(&h, 64, CUFFT_C2C, 1));
  cudaLibXtDesc* d = nullptr;
  CF(cufftXtMalloc(h, &d, CUFFT_XT_FORMAT_INPLACE));
  check(d->descriptor->nGPUs == 1 && d->descriptor->size[0] == 0,
        "a single-GPU plan's descriptor is one empty part");
  check(cufftXtExecDescriptorC2C(h, d, d, CUFFT_FORWARD) == CUFFT_EXEC_FAILED,
        "executing a single-GPU plan's descriptor is EXEC_FAILED");
  cufftXtFree(d);
  void* fn = nullptr;
  check(cufftXtSetCallback(h, &fn, CUFFT_CB_LD_COMPLEX, nullptr) == CUFFT_NOT_IMPLEMENTED,
        "legacy callbacks are NOT_IMPLEMENTED in libcufft.so");
  check(cufftXtClearCallback(h, CUFFT_CB_LD_COMPLEX) == CUFFT_NOT_IMPLEMENTED,
        "cufftXtClearCallback is NOT_IMPLEMENTED in libcufft.so");
  check(cufftXtSetCallbackSharedSize(h, CUFFT_CB_LD_COMPLEX, 16) == CUFFT_INVALID_PLAN,
        "cufftXtSetCallbackSharedSize without a callback is INVALID_PLAN");
  cufftDestroy(h);
}

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);  // a crash still shows how far it got
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
    std::printf("SKIP: needs two GPUs (VGPU_DEVICE_COUNT=2 on the simulator)\n");
    return 0;
  }
  plan_rules();
  batched();
  slabs<float>({33, 34}, "2-D C2C 33x34: y split after the transform, [x][y] on each GPU");
  slabs<double>({33, 32, 3}, "3-D Z2Z 33x32x3: y split after the transform, [x][y][z] on each GPU");
  real_slabs();
  line<float>(128);
  line<double>(1024);
  line<float>(64);
  odds();
  std::printf("%s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
