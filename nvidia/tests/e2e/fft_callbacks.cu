// cuFFT's LTO callbacks (cufftXtSetJITCallback, CUDA 12.6 and later), and the
// legacy ones, as libcufft.so answers them.
//
// The callbacks are device functions handed over as PTX text -- which
// NVIDIA's library accepts in place of an LTO-IR fatbin, and which is the
// form VirtualGPU can run (it executes PTX; LTO-IR is NVVM bitcode, which only
// NVIDIA's compiler reads). The PTX below is what nvcc -ptx -rdc=true
// -arch=compute_75 makes of these functions:
//
//   cufftComplex ld_scale(void* in, unsigned long long off, void* info, void*)
//     { v = in[off]; v *= *(float*)info; return v; }
//   void st_conj(void* out, unsigned long long off, cufftComplex e, void*, void*)
//     { out[off] = conj(e); }
//   cufftReal ld_real(void* in, unsigned long long off, void*, void*)
//     { return in[off] + off; }
//   void st_half(void* out, unsigned long long off, cufftDoubleComplex e, void*, void*)
//     { out[off] = e / 2; }
//   void st_real(void* out, unsigned long long off, cufftReal e, void*, void*)
//     { out[off] = e + 1; }
//
// Each transform is checked against a direct DFT on the host. Passes against
// NVIDIA's cuFFT on an RTX 3060 (CUDA 13.0); needs a cuFFT of CUDA 12.6 or
// later to compile the LTO part, and SKIPs it otherwise.
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

[[maybe_unused]] static double sig(int i) { return std::sin(0.37 * i) + 0.5 * std::cos(0.11 * i) + 0.05 * (i % 9); }

[[maybe_unused]] static std::vector<cd> dft(const std::vector<cd>& a, int sign) {
  const size_t n = a.size();
  std::vector<cd> o(n);
  for (size_t k = 0; k < n; ++k) {
    cd acc = 0;
    for (size_t j = 0; j < n; ++j) acc += a[j] * std::polar(1.0, sign * 2 * M_PI * (double)(j * k % n) / n);
    o[k] = acc;
  }
  return o;
}

// The legacy entry points: libcufft.so has them, and answers NOT_IMPLEMENTED
// (legacy callbacks are only in libcufft_static).
static void legacy() {
  cufftHandle h;
  CF(cufftPlan1d(&h, 64, CUFFT_C2C, 1));
  void* fn = nullptr;
  check(cufftXtSetCallback(h, &fn, CUFFT_CB_LD_COMPLEX, nullptr) == CUFFT_NOT_IMPLEMENTED,
        "cufftXtSetCallback is NOT_IMPLEMENTED in libcufft.so");
  check(cufftXtSetCallback(h + 1000, &fn, CUFFT_CB_LD_COMPLEX, nullptr) == CUFFT_NOT_IMPLEMENTED,
        "... whatever the plan");
  check(cufftXtClearCallback(h, CUFFT_CB_LD_COMPLEX) == CUFFT_NOT_IMPLEMENTED,
        "cufftXtClearCallback is NOT_IMPLEMENTED in libcufft.so");
  check(cufftXtSetCallbackSharedSize(h, CUFFT_CB_LD_COMPLEX, 16) == CUFFT_INVALID_PLAN,
        "cufftXtSetCallbackSharedSize on a plan without callbacks is INVALID_PLAN");
  cufftDestroy(h);
}

#if CUFFT_VERSION >= 11300
static const char kLdScale[] =
    ".version 7.0\n"
    ".target sm_75\n"
    ".address_size 64\n"
    ".visible .func  (.param .align 8 .b8 func_retval0[8]) _Z8ld_scalePvyS_S_(\n"
    " .param .b64 _Z8ld_scalePvyS_S__param_0,\n"
    " .param .b64 _Z8ld_scalePvyS_S__param_1,\n"
    " .param .b64 _Z8ld_scalePvyS_S__param_2,\n"
    " .param .b64 _Z8ld_scalePvyS_S__param_3\n"
    ")\n"
    "{\n"
    " .reg .f32  %f<8>;\n"
    " .reg .b64  %rd<6>;\n"
    " ld.param.u64  %rd1, [_Z8ld_scalePvyS_S__param_0];\n"
    " ld.param.u64  %rd2, [_Z8ld_scalePvyS_S__param_1];\n"
    " ld.param.u64  %rd3, [_Z8ld_scalePvyS_S__param_2];\n"
    " shl.b64  %rd4, %rd2, 3;\n"
    " add.s64  %rd5, %rd1, %rd4;\n"
    " ld.v2.f32  {%f1, %f2}, [%rd5];\n"
    " ld.f32  %f5, [%rd3];\n"
    " mul.f32  %f6, %f2, %f5;\n"
    " mul.f32  %f7, %f1, %f5;\n"
    " st.param.f32  [func_retval0+0], %f7;\n"
    " st.param.f32  [func_retval0+4], %f6;\n"
    " ret;\n"
    "}\n";
static const char kStConj[] =
    ".version 7.0\n"
    ".target sm_75\n"
    ".address_size 64\n"
    ".visible .func _Z7st_conjPvy6float2S_S_(\n"
    " .param .b64 _Z7st_conjPvy6float2S_S__param_0,\n"
    " .param .b64 _Z7st_conjPvy6float2S_S__param_1,\n"
    " .param .align 8 .b8 _Z7st_conjPvy6float2S_S__param_2[8],\n"
    " .param .b64 _Z7st_conjPvy6float2S_S__param_3,\n"
    " .param .b64 _Z7st_conjPvy6float2S_S__param_4\n"
    ")\n"
    "{\n"
    " .reg .f32  %f<4>;\n"
    " .reg .b64  %rd<5>;\n"
    " ld.param.u64  %rd1, [_Z7st_conjPvy6float2S_S__param_0];\n"
    " ld.param.u64  %rd2, [_Z7st_conjPvy6float2S_S__param_1];\n"
    " ld.param.f32  %f1, [_Z7st_conjPvy6float2S_S__param_2+4];\n"
    " shl.b64  %rd3, %rd2, 3;\n"
    " add.s64  %rd4, %rd1, %rd3;\n"
    " neg.f32  %f2, %f1;\n"
    " ld.param.f32  %f3, [_Z7st_conjPvy6float2S_S__param_2];\n"
    " st.v2.f32  [%rd4], {%f3, %f2};\n"
    " ret;\n"
    "}\n";
static const char kLdReal[] =
    ".version 7.0\n"
    ".target sm_75\n"
    ".address_size 64\n"
    ".visible .func  (.param .b32 func_retval0) _Z7ld_realPvyS_S_(\n"
    " .param .b64 _Z7ld_realPvyS_S__param_0,\n"
    " .param .b64 _Z7ld_realPvyS_S__param_1,\n"
    " .param .b64 _Z7ld_realPvyS_S__param_2,\n"
    " .param .b64 _Z7ld_realPvyS_S__param_3\n"
    ")\n"
    "{\n"
    " .reg .f32  %f<4>;\n"
    " .reg .b64  %rd<5>;\n"
    " ld.param.u64  %rd1, [_Z7ld_realPvyS_S__param_0];\n"
    " ld.param.u64  %rd2, [_Z7ld_realPvyS_S__param_1];\n"
    " shl.b64  %rd3, %rd2, 2;\n"
    " add.s64  %rd4, %rd1, %rd3;\n"
    " ld.f32  %f1, [%rd4];\n"
    " cvt.rn.f32.u64  %f2, %rd2;\n"
    " add.f32  %f3, %f1, %f2;\n"
    " st.param.f32  [func_retval0+0], %f3;\n"
    " ret;\n"
    "}\n";
static const char kStHalf[] =
    ".version 7.0\n"
    ".target sm_75\n"
    ".address_size 64\n"
    ".visible .func _Z7st_halfPvy7double2S_S_(\n"
    " .param .b64 _Z7st_halfPvy7double2S_S__param_0,\n"
    " .param .b64 _Z7st_halfPvy7double2S_S__param_1,\n"
    " .param .align 16 .b8 _Z7st_halfPvy7double2S_S__param_2[16],\n"
    " .param .b64 _Z7st_halfPvy7double2S_S__param_3,\n"
    " .param .b64 _Z7st_halfPvy7double2S_S__param_4\n"
    ")\n"
    "{\n"
    " .reg .f64  %fd<5>;\n"
    " .reg .b64  %rd<5>;\n"
    " ld.param.u64  %rd1, [_Z7st_halfPvy7double2S_S__param_0];\n"
    " ld.param.u64  %rd2, [_Z7st_halfPvy7double2S_S__param_1];\n"
    " ld.param.f64  %fd1, [_Z7st_halfPvy7double2S_S__param_2];\n"
    " ld.param.f64  %fd2, [_Z7st_halfPvy7double2S_S__param_2+8];\n"
    " shl.b64  %rd3, %rd2, 4;\n"
    " add.s64  %rd4, %rd1, %rd3;\n"
    " mul.f64  %fd3, %fd2, 0d3FE0000000000000;\n"
    " mul.f64  %fd4, %fd1, 0d3FE0000000000000;\n"
    " st.v2.f64  [%rd4], {%fd4, %fd3};\n"
    " ret;\n"
    "}\n";
static const char kStReal[] =
    ".version 7.0\n"
    ".target sm_75\n"
    ".address_size 64\n"
    ".visible .func _Z7st_realPvyfS_S_(\n"
    " .param .b64 _Z7st_realPvyfS_S__param_0,\n"
    " .param .b64 _Z7st_realPvyfS_S__param_1,\n"
    " .param .b32 _Z7st_realPvyfS_S__param_2,\n"
    " .param .b64 _Z7st_realPvyfS_S__param_3,\n"
    " .param .b64 _Z7st_realPvyfS_S__param_4\n"
    ")\n"
    "{\n"
    " .reg .f32  %f<3>;\n"
    " .reg .b64  %rd<5>;\n"
    " ld.param.u64  %rd1, [_Z7st_realPvyfS_S__param_0];\n"
    " ld.param.u64  %rd2, [_Z7st_realPvyfS_S__param_1];\n"
    " ld.param.f32  %f1, [_Z7st_realPvyfS_S__param_2];\n"
    " add.f32  %f2, %f1, 0f3F800000;\n"
    " shl.b64  %rd3, %rd2, 2;\n"
    " add.s64  %rd4, %rd1, %rd3;\n"
    " st.f32  [%rd4], %f2;\n"
    " ret;\n"
    "}\n";

// A callback that does not link fails the plan: CUDA 13.2's cuFFT says so
// with NVJITLINK_FAILURE, 13.0's with INTERNAL_ERROR (RTX 3060).
static bool link_failed(cufftResult r) { return r == (cufftResult)0x13 || r == CUFFT_INTERNAL_ERROR; }

template <size_t N>
static cufftResult set_jit(cufftHandle h, const char* name, const char (&image)[N],
                           cufftXtCallbackType type, void* info) {
  void* infos[1] = {info};
  return cufftXtSetJITCallback(h, name, image, N, type, info ? infos : nullptr);
}

// What cufftXtSetJITCallback and planning refuse, as NVIDIA's answers.
static void rules() {
  cufftHandle h;
  size_t ws;
  CF(cufftCreate(&h));
  check(cufftXtSetJITCallback(h, "ld_scale", nullptr, 16, CUFFT_CB_LD_COMPLEX, nullptr) == CUFFT_INVALID_VALUE,
        "no callback image is INVALID_VALUE");
  check(cufftXtSetJITCallback(h, "ld_scale", kLdScale, 0, CUFFT_CB_LD_COMPLEX, nullptr) == CUFFT_INVALID_VALUE,
        "an empty callback image is INVALID_VALUE");
  check(set_jit(h, "ld_scale", kLdScale, CUFFT_CB_UNDEFINED, nullptr) == CUFFT_INVALID_TYPE,
        "CUFFT_CB_UNDEFINED is INVALID_TYPE");
  check(set_jit(h, "no_such_function", kLdScale, CUFFT_CB_LD_COMPLEX, nullptr) == CUFFT_SUCCESS,
        "the callback's name is not looked at until the plan is made");
  cufftResult r = cufftMakePlan1d(h, 64, CUFFT_C2C, 1, &ws);
  check(link_failed(r), "a callback name the image does not define fails the plan", r);
  cufftDestroy(h);
  CF(cufftCreate(&h));
  CF(set_jit(h, "ld_scale", kLdScale, CUFFT_CB_ST_REAL, nullptr));
  r = cufftMakePlan1d(h, 64, CUFFT_C2C, 1, &ws);
  check(link_failed(r), "a load routine set as a real store does not link", r);
  cufftDestroy(h);
  CF(cufftCreate(&h));
  CF(cufftXtSetJITCallback(h, "ld_scale", "not a fatbin", 12, CUFFT_CB_LD_COMPLEX, nullptr));
  r = cufftMakePlan1d(h, 64, CUFFT_C2C, 1, &ws);
  check(link_failed(r), "an image that is not code fails the plan", r);
  cufftDestroy(h);
  // A real load has the mangled name of a complex one (the return type is not
  // part of it), so it links; a C2C plan just never calls it.
  CF(cufftCreate(&h));
  CF(set_jit(h, "ld_scale", kLdScale, CUFFT_CB_LD_REAL, nullptr));
  check(cufftMakePlan1d(h, 64, CUFFT_C2C, 1, &ws) == CUFFT_SUCCESS, "a load of another type that links is accepted");
  cufftDestroy(h);
  CF(cufftPlan1d(&h, 64, CUFFT_C2C, 1));
  check(set_jit(h, "ld_scale", kLdScale, CUFFT_CB_LD_COMPLEX, nullptr) == CUFFT_INVALID_PLAN,
        "LTO callbacks after the plan is made are INVALID_PLAN");
  cufftDestroy(h);
}

// C2C, batched and strided, with a load and a store callback.
static void c2c() {
  const int n = 64, batch = 2, stride = 2, dist = n * stride + 3;
  float* scale = nullptr;
  cudaMalloc(&scale, sizeof(float));
  const float three = 3.0f;
  cudaMemcpy(scale, &three, sizeof three, cudaMemcpyHostToDevice);
  cufftHandle h;
  CF(cufftCreate(&h));
  CF(set_jit(h, "ld_scale", kLdScale, CUFFT_CB_LD_COMPLEX, scale));
  CF(set_jit(h, "st_conj", kStConj, CUFFT_CB_ST_COMPLEX, nullptr));
  int nn[1] = {n}, emb[1] = {n};
  size_t ws;
  CF(cufftMakePlanMany(h, 1, nn, emb, stride, dist, emb, stride, dist, CUFFT_C2C, batch, &ws));
  check(cufftXtSetCallbackSharedSize(h, CUFFT_CB_LD_COMPLEX, 64) == CUFFT_SUCCESS,
        "shared memory for an LTO callback, once the plan is made");
  check(cufftXtClearCallback(h, CUFFT_CB_LD_COMPLEX) == CUFFT_NOT_IMPLEMENTED,
        "an LTO callback cannot be cleared");
  const size_t span = (size_t)dist * batch;
  std::vector<float2> host(span);
  for (size_t i = 0; i < span; ++i) host[i] = {(float)sig((int)i), (float)sig((int)i + 9)};
  cufftComplex* d = nullptr;
  cudaMalloc(&d, span * sizeof(float2));
  cudaMemcpy(d, host.data(), span * sizeof(float2), cudaMemcpyHostToDevice);
  CF(cufftExecC2C(h, d, d, CUFFT_FORWARD));
  std::vector<float2> out(span);
  cudaMemcpy(out.data(), d, span * sizeof(float2), cudaMemcpyDeviceToHost);
  double e = 0, s = 0;
  bool gaps = true;
  for (int b = 0; b < batch; ++b) {
    std::vector<cd> x(n);
    for (int k = 0; k < n; ++k) {
      const float2 v = host[(size_t)b * dist + (size_t)k * stride];
      x[k] = cd(v.x, v.y) * 3.0;
    }
    const auto ref = dft(x, -1);
    for (int k = 0; k < n; ++k) {
      const float2 v = out[(size_t)b * dist + (size_t)k * stride];
      e = std::max(e, std::abs(cd(v.x, v.y) - std::conj(ref[k])));
      s = std::max(s, std::abs(ref[k]));
    }
    for (int k = 0; k < n; ++k) {  // the elements the stride skips
      const size_t at = (size_t)b * dist + (size_t)k * stride + 1;
      gaps = gaps && out[at].x == host[at].x && out[at].y == host[at].y;
    }
  }
  check(e / s < 1e-5, "strided batched C2C: the load callback scales by callerInfo, the store conjugates", e / s);
  check(gaps, "the elements a strided store skips keep what they held");
  cudaFree(d);
  cudaFree(scale);
  cufftDestroy(h);
}

// R2C: the load callback sees each element's offset in the input buffer.
static void r2c() {
  const int n = 16, batch = 3, idist = n + 2, odist = n / 2 + 1;
  cufftHandle h;
  CF(cufftCreate(&h));
  CF(set_jit(h, "ld_real", kLdReal, CUFFT_CB_LD_REAL, nullptr));
  int nn[1] = {n}, ie[1] = {idist}, oe[1] = {odist};
  size_t ws;
  CF(cufftMakePlanMany(h, 1, nn, ie, 1, idist, oe, 1, odist, CUFFT_R2C, batch, &ws));
  std::vector<float> host(idist * batch);
  for (size_t i = 0; i < host.size(); ++i) host[i] = (float)sig((int)i);
  float* in = nullptr;
  float2* out = nullptr;
  cudaMalloc(&in, host.size() * sizeof(float));
  cudaMalloc(&out, (size_t)odist * batch * sizeof(float2));
  cudaMemcpy(in, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
  CF(cufftExecR2C(h, in, out));
  std::vector<float2> res((size_t)odist * batch);
  cudaMemcpy(res.data(), out, res.size() * sizeof(float2), cudaMemcpyDeviceToHost);
  double e = 0, s = 0;
  for (int b = 0; b < batch; ++b) {
    std::vector<cd> x(n);
    for (int k = 0; k < n; ++k) x[k] = host[b * idist + k] + (double)(b * idist + k);
    const auto ref = dft(x, -1);
    for (int k = 0; k <= n / 2; ++k) {
      e = std::max(e, std::abs(cd(res[b * odist + k].x, res[b * odist + k].y) - ref[k]));
      s = std::max(s, std::abs(ref[k]));
    }
  }
  check(e / s < 1e-5, "R2C: the load callback is given each element's offset from the input's start", e / s);
  cudaFree(in);
  cudaFree(out);
  cufftDestroy(h);
}

// Z2Z with a double store; C2R with a real store and shared memory.
static void z2z_c2r() {
  const int n = 32;
  cufftHandle h;
  size_t ws;
  CF(cufftCreate(&h));
  CF(set_jit(h, "st_half", kStHalf, CUFFT_CB_ST_COMPLEX_DOUBLE, nullptr));
  CF(cufftMakePlan1d(h, n, CUFFT_Z2Z, 1, &ws));
  std::vector<double2> host(n);
  std::vector<cd> x(n);
  for (int i = 0; i < n; ++i) {
    x[i] = cd(sig(i), sig(2 * i));
    host[i] = {x[i].real(), x[i].imag()};
  }
  double2* d = nullptr;
  cudaMalloc(&d, n * sizeof(double2));
  cudaMemcpy(d, host.data(), n * sizeof(double2), cudaMemcpyHostToDevice);
  CF(cufftExecZ2Z(h, d, d, CUFFT_INVERSE));
  cudaMemcpy(host.data(), d, n * sizeof(double2), cudaMemcpyDeviceToHost);
  const auto ref = dft(x, 1);
  double e = 0;
  for (int i = 0; i < n; ++i) e = std::max(e, std::abs(cd(host[i].x, host[i].y) - ref[i] * 0.5));
  check(e < 1e-12, "Z2Z: the double store callback halves the result", e);
  cudaFree(d);
  cufftDestroy(h);

  CF(cufftCreate(&h));
  CF(set_jit(h, "st_real", kStReal, CUFFT_CB_ST_REAL, nullptr));
  CF(cufftMakePlan1d(h, n, CUFFT_C2R, 1, &ws));
  CF(cufftXtSetCallbackSharedSize(h, CUFFT_CB_ST_REAL, 128));
  std::vector<cd> r(n);
  for (int i = 0; i < n; ++i) r[i] = sig(5 * i);
  const auto spec = dft(r, -1);
  std::vector<float2> hs(n / 2 + 1);
  for (int k = 0; k <= n / 2; ++k) hs[k] = {(float)spec[k].real(), (float)spec[k].imag()};
  float2* in = nullptr;
  float* out = nullptr;
  cudaMalloc(&in, hs.size() * sizeof(float2));
  cudaMalloc(&out, n * sizeof(float));
  cudaMemcpy(in, hs.data(), hs.size() * sizeof(float2), cudaMemcpyHostToDevice);
  CF(cufftExecC2R(h, in, out));
  std::vector<float> res(n);
  cudaMemcpy(res.data(), out, n * sizeof(float), cudaMemcpyDeviceToHost);
  e = 0;
  for (int i = 0; i < n; ++i) e = std::max(e, std::abs(res[i] - (r[i].real() * n + 1.0)));
  check(e < 1e-3, "C2R: the real store callback adds one to N times the signal", e);
  cudaFree(in);
  cudaFree(out);
  cufftDestroy(h);
}
#endif

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  legacy();
#if CUFFT_VERSION >= 11300
  rules();
  c2c();
  r2c();
  z2z_c2r();
#else
  std::printf("SKIP: LTO callbacks need cuFFT 11.3 (CUDA 12.6) headers; this is %d\n", CUFFT_VERSION);
#endif
  std::printf("%s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
