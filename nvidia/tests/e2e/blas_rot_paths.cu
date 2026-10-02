// cuBLAS's plane rotations -- rot, rotg, rotm and rotmg, typed and Ex --
// checked against what an RTX 3060's cuBLAS answers: which types the Ex forms
// take, the arithmetic bit for bit (the card fuses each rotation in a fixed
// order, which the expected values below repeat with fmaf), BLAS's special
// cases, negative increments and the device pointer mode. Every check passes
// on the card too.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

template <class T> static T* dev(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(T) + 16);
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> static std::vector<T> host(const T* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}
template <class T> static bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
template <class T> static bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
// Values with every bit of the mantissa busy, so that the order of rounding shows.
static float val(int i) { return std::sin(1.7f * i + 0.3f) * 3.1f; }

// ---- rot ----
static void rot(cublasHandle_t h) {
  const int n = 64;
  std::vector<float> x(n), y(n);
  for (int i = 0; i < n; ++i) x[i] = val(i), y[i] = val(i + 100);
  const float c = 0.7312345f, s = -0.6821313f;
  float* dx = dev(x);
  float* dy = dev(y);
  IS(cublasSrot(h, n, dx, 1, dy, 1, &c, &s), CUBLAS_STATUS_SUCCESS);
  std::vector<float> wx(n), wy(n);
  for (int i = 0; i < n; ++i) wx[i] = std::fma(c, x[i], s * y[i]), wy[i] = std::fma(c, y[i], -(s * x[i]));
  check(same_bits(host(dx, n), wx) && same_bits(host(dy, n), wy), "srot: x' = fma(c, x, s y), y' = fma(c, y, -(s x))");

  // x walked from its far end: x's element i is x[n-1-i].
  cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dy, y.data(), n * 4, cudaMemcpyHostToDevice);
  IS(cublasSrot(h, n / 2, dx, -2, dy, 1, &c, &s), CUBLAS_STATUS_SUCCESS);
  bool ok = true;
  const auto rx = host(dx, n), ry = host(dy, n);
  for (int i = 0; i < n / 2; ++i) {
    const float xi = x[(n / 2 - 1 - i) * 2], yi = y[i];
    ok = ok && same_bits(rx[(n / 2 - 1 - i) * 2], std::fma(c, xi, s * yi)) &&
         same_bits(ry[i], std::fma(c, yi, -(s * xi)));
  }
  for (int i = 0; i < n / 2; ++i) ok = ok && same_bits(rx[2 * i + 1], x[2 * i + 1]);
  check(ok, "srot with incx -2 walks x from its end and leaves the gaps");
  cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice);
  IS(cublasSrot(h, 0, dx, 1, dy, 1, &c, &s), CUBLAS_STATUS_SUCCESS);
  IS(cublasSrot(h, -1, dx, 1, dy, 1, &c, &s), CUBLAS_STATUS_SUCCESS);
  check(same_bits(host(dx, n), x), "and n <= 0 does nothing");
  cudaMemcpy(dy, y.data(), n * 4, cudaMemcpyHostToDevice);
  IS(cublasSrot_v2_64(h, n, dx, 1, dy, 1, &c, &s), CUBLAS_STATUS_SUCCESS);
  check(same_bits(host(dx, n), wx) && same_bits(host(dy, n), wy), "cublasSrot_v2_64");

  // Double, the same fused order.
  std::vector<double> xd(n), yd(n);
  for (int i = 0; i < n; ++i) xd[i] = val(i) / 3.0, yd[i] = val(i + 7) / 7.0;
  const double cd_ = 0.123456789012345, sd = 0.987654321098765;
  double* dxd = dev(xd);
  double* dyd = dev(yd);
  IS(cublasDrot(h, n, dxd, 1, dyd, 1, &cd_, &sd), CUBLAS_STATUS_SUCCESS);
  std::vector<double> wxd(n), wyd(n);
  for (int i = 0; i < n; ++i) wxd[i] = std::fma(cd_, xd[i], sd * yd[i]), wyd[i] = std::fma(cd_, yd[i], -(sd * xd[i]));
  check(same_bits(host(dxd, n), wxd) && same_bits(host(dyd, n), wyd), "drot, fused the same way");

  // Complex: c real, y' = c y - conj(s) x, each complex product with its
  // first part fused, and c times the operand fused onto that.
  std::vector<cuComplex> xc(n), yc(n);
  for (int i = 0; i < n; ++i) xc[i] = make_cuComplex(val(i), val(i + 1)), yc[i] = make_cuComplex(val(i + 2), val(i + 3));
  const cuComplex sc = make_cuComplex(0.3141593f, -0.5772157f);
  cuComplex* dxc = dev(xc);
  cuComplex* dyc = dev(yc);
  IS(cublasCrot(h, n, dxc, 1, dyc, 1, &c, &sc), CUBLAS_STATUS_SUCCESS);
  std::vector<cuComplex> wxc(n), wyc(n);
  for (int i = 0; i < n; ++i) {
    const float xr = xc[i].x, xi = xc[i].y, yr = yc[i].x, yi = yc[i].y, sr = sc.x, si = sc.y;
    wxc[i] = make_cuComplex(std::fma(c, xr, std::fma(sr, yr, -(si * yi))), std::fma(c, xi, std::fma(si, yr, sr * yi)));
    wyc[i] = make_cuComplex(std::fma(c, yr, -std::fma(sr, xr, si * xi)), std::fma(c, yi, -std::fma(-si, xr, sr * xi)));
  }
  check(same_bits(host(dxc, n), wxc) && same_bits(host(dyc, n), wyc), "crot: y' = c y - conj(s) x, bit for bit");
  // csrot: s real, so each part rotates as srot would.
  cudaMemcpy(dxc, xc.data(), n * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(dyc, yc.data(), n * 8, cudaMemcpyHostToDevice);
  IS(cublasCsrot(h, n, dxc, 1, dyc, 1, &c, &s), CUBLAS_STATUS_SUCCESS);
  ok = true;
  const auto rxc = host(dxc, n), ryc = host(dyc, n);
  for (int i = 0; i < n; ++i)
    ok = ok && same_bits(rxc[i].x, std::fma(c, xc[i].x, s * yc[i].x)) &&
         same_bits(rxc[i].y, std::fma(c, xc[i].y, s * yc[i].y)) &&
         same_bits(ryc[i].x, std::fma(c, yc[i].x, -(s * xc[i].x))) &&
         same_bits(ryc[i].y, std::fma(c, yc[i].y, -(s * xc[i].y)));
  check(ok, "csrot rotates the real and imaginary parts as srot");
  // RotEx with a complex c: its imaginary part is not used.
  cudaMemcpy(dxc, xc.data(), n * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(dyc, yc.data(), n * 8, cudaMemcpyHostToDevice);
  const cuComplex cc = make_cuComplex(c, 13.0f);
  IS(cublasRotEx(h, n, dxc, CUDA_C_32F, 1, dyc, CUDA_C_32F, 1, &cc, &sc, CUDA_C_32F, CUDA_C_32F), CUBLAS_STATUS_SUCCESS);
  check(same_bits(host(dxc, n), wxc) && same_bits(host(dyc, n), wyc), "RotEx in C_32F is crot; c's imaginary part is ignored");
  std::vector<cuDoubleComplex> xz(4, make_cuDoubleComplex(1, 2)), yz(4, make_cuDoubleComplex(3, 5));
  cuDoubleComplex* dxz = dev(xz);
  cuDoubleComplex* dyz = dev(yz);
  const double two = 2, seven = 7;
  const cuDoubleComplex sz = make_cuDoubleComplex(7, 11);
  IS(cublasZrot(h, 4, dxz, 1, dyz, 1, &two, &sz), CUBLAS_STATUS_SUCCESS);
  const auto rz = host(dxz, 1)[0], ryz = host(dyz, 1)[0];
  check(rz.x == -32 && rz.y == 72 && ryz.x == -23 && ryz.y == 7, "zrot: (1+2i, 3+5i) by c 2, s 7+11i");
  IS(cublasZdrot(h, 4, dxz, 1, dyz, 1, &two, &seven), CUBLAS_STATUS_SUCCESS);
  check(host(dxz, 1)[0].x == -32 * 2 + 7 * -23 && host(dyz, 1)[0].y == 7 * 2 - 7 * 72, "zdrot");

  // Half precision rotates in single and rounds back.
  std::vector<__half> xh(n), yh(n);
  for (int i = 0; i < n; ++i) xh[i] = __float2half(val(i)), yh[i] = __float2half(val(i + 50));
  const __half ch = __float2half(0.6f), sh = __float2half(-0.8f);
  __half* dxh = dev(xh);
  __half* dyh = dev(yh);
  IS(cublasRotEx(h, n, dxh, CUDA_R_16F, 1, dyh, CUDA_R_16F, 1, &ch, &sh, CUDA_R_16F, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  ok = true;
  const auto rxh = host(dxh, n), ryh = host(dyh, n);
  for (int i = 0; i < n; ++i) {
    const float cf = __half2float(ch), sf = __half2float(sh), xf = __half2float(xh[i]), yf = __half2float(yh[i]);
    ok = ok && same_bits(rxh[i], __float2half(std::fma(cf, xf, sf * yf))) &&
         same_bits(ryh[i], __float2half(std::fma(cf, yf, -(sf * xf))));
  }
  check(ok, "half RotEx: single-precision arithmetic, rounded to half");
  IS(cublasRotEx_64(h, n, dxh, CUDA_R_16F, 1, dyh, CUDA_R_16F, 1, &ch, &sh, CUDA_R_16F, CUDA_R_32F),
     CUBLAS_STATUS_SUCCESS);

  // The types RotEx takes: x, y, c and s alike, the narrow ones executing in
  // single precision; a complex x with c and s of its real type, too.
  const double big[2] = {0.5, 0.5};
  IS(cublasRotEx(h, n, dx, CUDA_R_32F, 1, dy, CUDA_R_32F, 1, big, big, CUDA_R_32F, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  IS(cublasRotEx(h, n, dxc, CUDA_C_32F, 1, dyc, CUDA_C_32F, 1, big, big, CUDA_R_32F, CUDA_C_32F),
     CUBLAS_STATUS_SUCCESS);
  IS(cublasRotEx(h, n, dxc, CUDA_C_32F, 1, dyc, CUDA_C_32F, 1, big, big, CUDA_R_64F, CUDA_C_32F),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasRotEx(h, n, dxh, CUDA_R_16F, 1, dyh, CUDA_R_16F, 1, big, big, CUDA_R_32F, CUDA_R_32F),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasRotEx(h, n, dxh, CUDA_R_16F, 1, dyh, CUDA_R_16F, 1, big, big, CUDA_R_16F, CUDA_R_16F),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasRotEx(h, n, dx, CUDA_R_32F, 1, dy, CUDA_R_64F, 1, big, big, CUDA_R_32F, CUDA_R_32F),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasRotEx(h, n, dx, CUDA_R_32F, 1, dy, CUDA_R_32F, 1, big, big, CUDA_R_32F, CUDA_R_64F),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasRotEx(h, -1, dx, CUDA_R_8I, 1, dy, CUDA_R_8I, 1, big, big, CUDA_R_8I, CUDA_R_32F),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasSrot(nullptr, n, dx, 1, dy, 1, &c, &s), CUBLAS_STATUS_NOT_INITIALIZED);
  for (void* p : {(void*)dx, (void*)dy, (void*)dxd, (void*)dyd, (void*)dxc, (void*)dyc, (void*)dxz, (void*)dyz,
                  (void*)dxh, (void*)dyh})
    cudaFree(p);
}

// ---- rotg ----
static void rotg(cublasHandle_t h) {
  auto s4 = [&](float a, float b, float wa, float wb, float wc, float ws, const char* what) {
    float c = 9, s = 9;
    const int st = cublasSrotg(h, &a, &b, &c, &s);
    check(st == 0 && same_bits(a, wa) && same_bits(b, wb) && same_bits(c, wc) && same_bits(s, ws), what);
  };
  s4(3, -4, -5, -1.0f / 0.6f, -0.6f, 0.8f, "srotg(3, -4): r takes the sign of the larger, z = 1/c");
  s4(-4, 3, -5, -0.6f, 0.8f, -0.6f, "srotg(-4, 3): z = s");
  s4(0, -3, -3, 1, -0.0f, 1, "srotg(0, -3): c = -0, no shortcut for a zero a");
  s4(-3, 0, -3, -0.0f, 1, -0.0f, "srotg(-3, 0): s = -0");
  s4(0, 0, 0, 0, 1, 0, "srotg(0, 0): the identity");
  // The larger of c and s is a/r (or b/r), the smaller that times b/a (a/b).
  const float a0 = 0x1.88813cp+12f, b0 = -0x1.ff64bp+1f;
  {
    const float p = a0 / a0, q = b0 / a0;
    const float r = 1.0f * (a0 * std::sqrt(p * p + q * q)), c = a0 / r, s = (b0 / a0) * c;
    s4(a0, b0, r, s, c, s, "srotg bit for bit: r scaled by the larger, s = (b/a) c");
  }
  double a = 1e200, b = -3e199, c, s;
  IS(cublasDrotg(h, &a, &b, &c, &s), CUBLAS_STATUS_SUCCESS);
  check(std::fabs(a - std::hypot(1e200, 3e199)) < 1e186 && c > 0 && s < 0 && b == s, "drotg scales, so 1e200 does not overflow");

  cuComplex ca = make_cuComplex(3, 4), cb = make_cuComplex(1, -2), cs;
  float cc;
  IS(cublasCrotg(h, &ca, &cb, &cc, &cs), CUBLAS_STATUS_SUCCESS);
  const float nrm = std::sqrt(30.0f);
  check(std::fabs(cc - 5 / nrm) < 1e-6f && std::fabs(cs.x + 1 / nrm) < 1e-6f && std::fabs(cs.y - 2 / nrm) < 1e-6f &&
            std::fabs(ca.x - 0.6f * nrm) < 1e-5f && std::fabs(ca.y - 0.8f * nrm) < 1e-5f && cb.x == 1 && cb.y == -2,
        "crotg: c = |a|/norm, s = (a/|a|) conj(b)/norm, a = (a/|a|) norm, b untouched");
  ca = make_cuComplex(0, 0);
  IS(cublasCrotg(h, &ca, &cb, &cc, &cs), CUBLAS_STATUS_SUCCESS);
  check(cc == 0 && cs.x == 1 && cs.y == 0 && ca.x == 1 && ca.y == -2, "crotg with a zero a: c 0, s 1, a = b");
  cuDoubleComplex za = make_cuDoubleComplex(3, 4), zb = make_cuDoubleComplex(0, 0), zs;
  double zc;
  IS(cublasZrotg(h, &za, &zb, &zc, &zs), CUBLAS_STATUS_SUCCESS);
  check(zc == 1 && zs.x == 0 && zs.y == 0 && std::fabs(za.x - 3) < 1e-15 && std::fabs(za.y - 4) < 1e-15,
        "zrotg with a zero b: c 1, s 0");

  // RotgEx: real types only, and the narrow ones in single precision.
  __half ha = __float2half(3), hb = __float2half(-4), hc, hs;
  IS(cublasRotgEx(h, &ha, &hb, CUDA_R_16F, &hc, &hs, CUDA_R_16F, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  check(__half2float(ha) == -5 && __half2float(hc) == __half2float(__float2half(-0.6f)) &&
            __half2float(hb) == __half2float(__float2half(-1.0f / 0.6f)),
        "half RotgEx");
  float fa = 3, fb = 4, fc, fs;
  IS(cublasRotgEx(h, &fa, &fb, CUDA_R_32F, &fc, &fs, CUDA_R_32F, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  IS(cublasRotgEx(h, &fa, &fb, CUDA_R_32F, &fc, &fs, CUDA_R_32F, CUDA_R_64F), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasRotgEx(h, &fa, &fb, CUDA_R_16F, &fc, &fs, CUDA_R_16F, CUDA_R_16F), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasRotgEx(h, &ca, &cb, CUDA_C_32F, &fc, &cs, CUDA_R_32F, CUDA_C_32F), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasRotgEx(h, &ca, &cb, CUDA_C_32F, &cs, &cs, CUDA_C_32F, CUDA_C_32F), CUBLAS_STATUS_NOT_SUPPORTED);

  // Every operand through device memory.
  cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
  float* d = dev(std::vector<float>{3, -4, 9, 9});
  IS(cublasSrotg(h, d, d + 1, d + 2, d + 3), CUBLAS_STATUS_SUCCESS);
  cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
  check(same_bits(host(d, 4), std::vector<float>{-5, -1.0f / 0.6f, -0.6f, 0.8f}), "srotg in device memory");
  cudaFree(d);
}

// ---- rotm ----
static void rotm(cublasHandle_t h) {
  const int n = 32;
  std::vector<float> x(n), y(n);
  for (int i = 0; i < n; ++i) x[i] = val(i), y[i] = val(i + 40);
  float* dx = dev(x);
  float* dy = dev(y);
  const float h11 = 0.7312345f, h21 = -1.2345678f, h12 = 0.4567891f, h22 = 2.3456789f;
  auto run = [&](float flag, int incx, const char* what) {
    cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.data(), n * 4, cudaMemcpyHostToDevice);
    const float p[5] = {flag, h11, h21, h12, h22};
    const int m = incx == 1 ? n : n / 2;
    const int st = cublasSrotm(h, m, dx, incx, dy, 1, p);
    std::vector<float> wx = x, wy = y;
    for (int i = 0; i < m; ++i) {
      const size_t ix = incx > 0 ? (size_t)i * incx : (size_t)(m - 1 - i) * -incx;
      const float a = x[ix], b = y[i];
      if (flag == -1) wx[ix] = std::fma(h11, a, h12 * b), wy[i] = std::fma(h21, a, h22 * b);
      if (flag == 0) wx[ix] = std::fma(h12, b, a), wy[i] = std::fma(h21, a, b);
      if (flag == 1) wx[ix] = std::fma(h11, a, b), wy[i] = std::fma(h22, b, -a);
    }
    check(st == 0 && same_bits(host(dx, n), wx) && same_bits(host(dy, n), wy), what);
  };
  run(-1, 1, "srotm flag -1: the full H, fused as rot");
  run(0, 1, "srotm flag 0: h11 = h22 = 1");
  run(1, 1, "srotm flag 1: h12 = 1, h21 = -1");
  run(-2, 1, "srotm flag -2: the identity");
  run(5, 1, "srotm with an unknown flag does nothing");
  run(-1, -2, "srotm with incx -2");
  run(0, 2, "srotm with incx 2");
  const double pd[5] = {1, 2, 0, 0, 3};
  double* dxd = dev(std::vector<double>{1, 2});
  double* dyd = dev(std::vector<double>{10, 20});
  IS(cublasDrotm(h, 2, dxd, 1, dyd, 1, pd), CUBLAS_STATUS_SUCCESS);
  check(host(dxd, 2) == std::vector<double>({12, 24}) && host(dyd, 2) == std::vector<double>({29, 58}), "drotm");
  IS(cublasDrotm_v2_64(h, 2, dxd, 1, dyd, 1, pd), CUBLAS_STATUS_SUCCESS);
  const float pf[5] = {0, 0, 0.5f, 0.25f, 0};
  IS(cublasRotmEx(h, n, dx, CUDA_R_32F, 1, dy, CUDA_R_32F, 1, pf, CUDA_R_32F, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  IS(cublasRotmEx_64(h, n, dx, CUDA_R_32F, 1, dy, CUDA_R_32F, 1, pf, CUDA_R_32F, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  IS(cublasRotmEx(h, n, dx, CUDA_R_32F, 1, dy, CUDA_R_32F, 1, pf, CUDA_R_64F, CUDA_R_32F), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasRotmEx(h, n, dx, CUDA_C_32F, 1, dy, CUDA_C_32F, 1, pf, CUDA_C_32F, CUDA_C_32F), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasRotmEx(h, n, dx, CUDA_R_16BF, 1, dy, CUDA_R_16BF, 1, pf, CUDA_R_16BF, CUDA_R_16BF),
     CUBLAS_STATUS_NOT_SUPPORTED);
  // bfloat16, the parameters bfloat16 too.
  std::vector<__nv_bfloat16> xb = {__float2bfloat16(1), __float2bfloat16(2)}, yb = {__float2bfloat16(3), __float2bfloat16(5)};
  __nv_bfloat16* dxb = dev(xb);
  __nv_bfloat16* dyb = dev(yb);
  const __nv_bfloat16 pb[5] = {__float2bfloat16(1), __float2bfloat16(2), __float2bfloat16(0), __float2bfloat16(0),
                               __float2bfloat16(-1)};
  IS(cublasRotmEx(h, 2, dxb, CUDA_R_16BF, 1, dyb, CUDA_R_16BF, 1, pb, CUDA_R_16BF, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  const auto rxb = host(dxb, 2), ryb = host(dyb, 2);
  check(__bfloat162float(rxb[1]) == 9 && __bfloat162float(ryb[1]) == -7, "bfloat16 RotmEx, flag 1");
  // The parameters in device memory.
  cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
  float* dp = dev(std::vector<float>{-2, 1, 1, 1, 1});
  IS(cublasSrotm(h, n, dx, 1, dy, 1, dp), CUBLAS_STATUS_SUCCESS);
  cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
  for (void* p : {(void*)dx, (void*)dy, (void*)dxd, (void*)dyd, (void*)dxb, (void*)dyb, (void*)dp}) cudaFree(p);
}

// ---- rotmg ----
static void rotmg(cublasHandle_t h) {
  // {d1, d2, x1, y1} in; {d1, d2, x1, param[0..4]} out, as an RTX 3060 answers
  // (7 marks a parameter the flag leaves alone).
  struct Case {
    float in[4], out[8];
    const char* what;
  } cases[] = {
      {{2, 3, 5, 7}, {0x1.1e89cp+1f, 0x1.7e0cfep+0f, 0x1.2c30c4p+3f, 1, 0x1.e79e7ap-2f, 7, 7, 0x1.6db6dcp-1f},
       "srotmg flag 1: |d1 x1^2| <= |d2 y1^2|"},
      {{4, 1, 3, 1}, {0x1.f22984p+1f, 0x1.f22984p-1f, 0x1.8aaaacp+1f, 0, 7, -0x1.555556p-2f, 0x1.555556p-4f, 7},
       "srotmg flag 0"},
      {{-1, 2, 3, 4}, {0, 0, 0, -1, 0, 0, 0, 0}, "srotmg with d1 < 0 zeroes everything"},
      {{2, 3, 5, 0}, {2, 3, 5, -2, 7, 7, 7, 7}, "srotmg with d2 y1 = 0: flag -2, nothing else"},
      {{1e-10f, 1, 1, 1e-3f}, {0x1.fff2e4p-1f, 0x1.b7c2bcp-10f, 0x1.062b94p-10f, -1, 0x1.ad7f28p-24f, -0x1p-12f, 1,
                               0x1.f3fffep-3f},
       "srotmg rescales a tiny d2 by 4096^2, the matrix becoming full"},
  };
  for (const Case& k : cases) {
    float d1 = k.in[0], d2 = k.in[1], x1 = k.in[2], y1 = k.in[3], p[5] = {7, 7, 7, 7, 7};
    const int st = cublasSrotmg(h, &d1, &d2, &x1, &y1, p);
    const float got[8] = {d1, d2, x1, p[0], p[1], p[2], p[3], p[4]};
    bool ok = st == 0;
    for (int i = 0; i < 8; ++i) ok = ok && same_bits(got[i], k.out[i]);
    if (!ok)
      for (int i = 0; i < 8; ++i) std::printf("     got %a want %a\n", got[i], k.out[i]);
    check(ok, k.what);
  }
  double d1 = 2, d2 = 3, x1 = 5, y1 = 7, p[5] = {7, 7, 7, 7, 7};
  IS(cublasDrotmg(h, &d1, &d2, &x1, &y1, p), CUBLAS_STATUS_SUCCESS);
  check(p[0] == 1 && p[2] == 7 && std::fabs(p[1] - 10.0 / 21) < 1e-15 && std::fabs(p[4] - 5.0 / 7) < 1e-15 &&
            std::fabs(x1 - 7 * (1 + 50.0 / 147)) < 1e-14,
        "drotmg");
  // RotmgEx: all five operands one real type.
  float f[4] = {2, 3, 5, 7}, pf[5];
  IS(cublasRotmgEx(h, f, CUDA_R_32F, f + 1, CUDA_R_32F, f + 2, CUDA_R_32F, f + 3, CUDA_R_32F, pf, CUDA_R_32F,
                   CUDA_R_32F),
     CUBLAS_STATUS_SUCCESS);
  IS(cublasRotmgEx(h, f, CUDA_R_32F, f + 1, CUDA_R_64F, f + 2, CUDA_R_32F, f + 3, CUDA_R_32F, pf, CUDA_R_32F,
                   CUDA_R_32F),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasRotmgEx(h, f, CUDA_R_32F, f + 1, CUDA_R_32F, f + 2, CUDA_R_32F, f + 3, CUDA_R_32F, pf, CUDA_R_64F,
                   CUDA_R_32F),
     CUBLAS_STATUS_NOT_SUPPORTED);
  __half hh[4] = {__float2half(2), __float2half(3), __float2half(5), __float2half(7)}, hp[5];
  IS(cublasRotmgEx(h, hh, CUDA_R_16F, hh + 1, CUDA_R_16F, hh + 2, CUDA_R_16F, hh + 3, CUDA_R_16F, hp, CUDA_R_16F,
                   CUDA_R_32F),
     CUBLAS_STATUS_SUCCESS);
  check(__half2float(hp[0]) == 1 && same_bits(hh[2], __float2half(0x1.2c30c4p+3f)), "half RotmgEx, in single precision");
  // Through device memory: d1, d2, x1, y1 and the parameters.
  cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
  float* dm = dev(std::vector<float>{-1, 2, 3, 4, 7, 7, 7, 7, 7});
  IS(cublasSrotmg(h, dm, dm + 1, dm + 2, dm + 3, dm + 4), CUBLAS_STATUS_SUCCESS);
  cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
  check(host(dm, 9) == std::vector<float>({0, 0, 0, 4, -1, 0, 0, 0, 0}), "srotmg in device memory");
  cudaFree(dm);
}

int main() {
  cublasHandle_t h;
  if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
    std::printf("FAIL: cublasCreate\n");
    return 1;
  }
  rot(h);
  rotg(h);
  rotm(h);
  rotmg(h);
  cublasDestroy(h);
  std::printf(failures ? "FAIL: %d cuBLAS rotation checks\n" : "PASS: every cuBLAS rotation check\n", failures);
  return failures ? 1 : 0;
}
