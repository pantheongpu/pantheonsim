// A cuBLAS reduction in the default (host) pointer mode with its result pointer in device memory (pantheonsim #638).
// cuBLAS documents a host pointer there, but GauXC's gdot passes a cudaMalloc one. Measured on an RTX 3060 (CUDA 13.0,
// cuBLAS 13.1.0.3) over every reduction and its _64 form, at n = 1, 1000 and 700920:
//   - ?dot, ?dotu, ?dotc, DotEx and DotcEx with n > 0 return SUCCESS and store the result through the device pointer;
//   - the same with n <= 0, and ?nrm2, ?asum, i?amax, i?amin, Nrm2Ex, AsumEx, IamaxEx and IaminEx at any n, crash
//     (SIGSEGV), which the simulator answers with CUBLAS_STATUS_INVALID_VALUE and leaves the device word alone;
//   - a managed result is host memory, and every reduction writes it in host mode.
// Before the fix every one of these calls died with SIGSEGV in the host-computed routine. The plain host and device
// pointer modes are checked alongside, so the fix cannot have moved them.
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

#include "vtest.hpp"

namespace {

constexpr int kN = 1000;

// x[i] = (1 + (i % 3) / 2, (i % 2) / 4), with x[7] = (-3, 1) and x[11] = (0.25, 0): every sum below is exact in
// single precision, so the results are compared bit for bit. y[i] = (2, -(i % 2)). Real routines use the real parts.
struct Data {
  std::vector<double> xr, yr;
  std::vector<double> xi, yi;
  double dot = 0, nrm2sq = 0, asum = 0, cnrm2sq = 0, casum = 0;
  double dotu_re = 0, dotu_im = 0, dotc_re = 0, dotc_im = 0;
  int amax = 1, amin = 1, camax = 1, camin = 1;
  Data() {
    for (int i = 0; i < kN; ++i) {
      xr.push_back(1 + (i % 3) * 0.5);
      xi.push_back((i % 2) * 0.25);
      yr.push_back(2);
      yi.push_back(-(i % 2));
    }
    xr[7] = -3; xi[7] = 1; xr[11] = 0.25; xi[11] = 0;
    auto mag = [&](int i) { return (xr[i] < 0 ? -xr[i] : xr[i]) + xi[i]; };
    auto ab = [&](int i) { return xr[i] < 0 ? -xr[i] : xr[i]; };
    for (int i = 0; i < kN; ++i) {
      dot += xr[i] * yr[i];
      nrm2sq += xr[i] * xr[i];
      asum += ab(i);
      cnrm2sq += xr[i] * xr[i] + xi[i] * xi[i];
      casum += mag(i);
      dotu_re += xr[i] * yr[i] - xi[i] * yi[i];
      dotu_im += xr[i] * yi[i] + xi[i] * yr[i];
      dotc_re += xr[i] * yr[i] + xi[i] * yi[i];
      dotc_im += xr[i] * yi[i] - xi[i] * yr[i];
      if (ab(i) > ab(amax - 1)) amax = i + 1;
      if (ab(i) < ab(amin - 1)) amin = i + 1;
      if (mag(i) > mag(camax - 1)) camax = i + 1;
      if (mag(i) < mag(camin - 1)) camin = i + 1;
    }
  }
};

template <class T>
T* upload(const std::vector<T>& v) {
  T* p = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&p), v.size() * sizeof(T));
  cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return p;
}

struct Fixture {
  Data d;
  cublasHandle_t h = nullptr;
  float *sx, *sy;
  double *dx, *dy;
  cuComplex *cx, *cy;
  cuDoubleComplex *zx, *zy;
  Fixture() {
    VCHECK_EQ(cublasCreate(&h), CUBLAS_STATUS_SUCCESS);
    std::vector<float> fx(kN), fy(kN);
    std::vector<cuComplex> vcx(kN), vcy(kN);
    std::vector<cuDoubleComplex> vzx(kN), vzy(kN);
    for (int i = 0; i < kN; ++i) {
      fx[i] = (float)d.xr[i];
      fy[i] = (float)d.yr[i];
      vcx[i] = make_cuComplex((float)d.xr[i], (float)d.xi[i]);
      vcy[i] = make_cuComplex((float)d.yr[i], (float)d.yi[i]);
      vzx[i] = make_cuDoubleComplex(d.xr[i], d.xi[i]);
      vzy[i] = make_cuDoubleComplex(d.yr[i], d.yi[i]);
    }
    sx = upload(fx); sy = upload(fy);
    dx = upload(d.xr); dy = upload(d.yr);
    cx = upload(vcx); cy = upload(vcy);
    zx = upload(vzx); zy = upload(vzy);
  }
  ~Fixture() {
    for (void* p : std::initializer_list<void*>{sx, sy, dx, dy, cx, cy, zx, zy}) cudaFree(p);
    cublasDestroy(h);
  }
};

// Where the result goes: host memory, device memory or managed memory.
enum class Where { host, device, managed };

// Runs `call` with a 16-byte result slot filled with 0xA5 in `where`, the handle in `mode`, and returns the status
// and the slot's bytes afterwards.
struct Out {
  cublasStatus_t status;
  unsigned char bytes[16];
  template <class T> T as() const { T v; std::memcpy(&v, bytes, sizeof v); return v; }
  bool untouched() const {
    for (unsigned char b : bytes) if (b != 0xA5) return false;
    return true;
  }
};
Out run(Fixture& f, cublasPointerMode_t mode, Where where, const std::function<cublasStatus_t(void*)>& call) {
  Out o{};
  unsigned char fill[16];
  std::memset(fill, 0xA5, sizeof fill);
  void* slot = nullptr;
  unsigned char host[16];
  std::memcpy(host, fill, sizeof host);
  if (where == Where::device) {
    VCHECK_EQ(cudaMalloc(&slot, 16), cudaSuccess);
    cudaMemcpy(slot, fill, 16, cudaMemcpyHostToDevice);
  } else if (where == Where::managed) {
    VCHECK_EQ(cudaMallocManaged(&slot, 16), cudaSuccess);
    std::memcpy(slot, fill, 16);
  } else {
    slot = host;
  }
  VCHECK_EQ(cublasSetPointerMode(f.h, mode), CUBLAS_STATUS_SUCCESS);
  o.status = call(slot);
  VCHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  if (where == Where::device) cudaMemcpy(o.bytes, slot, 16, cudaMemcpyDeviceToHost);
  else std::memcpy(o.bytes, slot, 16);
  if (where != Where::host) cudaFree(slot);
  cublasSetPointerMode(f.h, CUBLAS_POINTER_MODE_HOST);
  return o;
}

// One reduction: its name, the call, and a check of the value it should leave.
struct Case {
  std::string name;
  bool dot;   // a dot product: the card stores a device result in host mode
  std::function<cublasStatus_t(Fixture&, int64_t n, void* r)> call;
  std::function<bool(const Fixture&, const Out&)> good;
};

template <class T> bool eq(const Out& o, T want) { return o.as<T>() == want; }
bool eqc(const Out& o, float re, float im) {
  const cuComplex v = o.as<cuComplex>();
  return v.x == re && v.y == im;
}
bool eqz(const Out& o, double re, double im) {
  const cuDoubleComplex v = o.as<cuDoubleComplex>();
  return v.x == re && v.y == im;
}
template <class T> bool near(const Out& o, double want) {
  const double v = o.as<T>(), e = (v - want) / want;
  return e < 1e-6 && e > -1e-6;
}

std::vector<Case> cases() {
  using F = Fixture;
  using O = Out;
  std::vector<Case> c;
  auto add = [&](std::string n, bool dot, std::function<cublasStatus_t(F&, int64_t, void*)> call,
                 std::function<bool(const F&, const O&)> good) { c.push_back({std::move(n), dot, call, good}); };
  // The dot products.
  add("Sdot", true, [](F& f, int64_t n, void* r) { return cublasSdot(f.h, (int)n, f.sx, 1, f.sy, 1, (float*)r); },
      [](const F& f, const O& o) { return eq<float>(o, (float)f.d.dot); });
  add("Ddot", true, [](F& f, int64_t n, void* r) { return cublasDdot(f.h, (int)n, f.dx, 1, f.dy, 1, (double*)r); },
      [](const F& f, const O& o) { return eq<double>(o, f.d.dot); });
  add("Cdotu", true, [](F& f, int64_t n, void* r) { return cublasCdotu(f.h, (int)n, f.cx, 1, f.cy, 1, (cuComplex*)r); },
      [](const F& f, const O& o) { return eqc(o, (float)f.d.dotu_re, (float)f.d.dotu_im); });
  add("Cdotc", true, [](F& f, int64_t n, void* r) { return cublasCdotc(f.h, (int)n, f.cx, 1, f.cy, 1, (cuComplex*)r); },
      [](const F& f, const O& o) { return eqc(o, (float)f.d.dotc_re, (float)f.d.dotc_im); });
  add("Zdotu", true,
      [](F& f, int64_t n, void* r) { return cublasZdotu(f.h, (int)n, f.zx, 1, f.zy, 1, (cuDoubleComplex*)r); },
      [](const F& f, const O& o) { return eqz(o, f.d.dotu_re, f.d.dotu_im); });
  add("Zdotc", true,
      [](F& f, int64_t n, void* r) { return cublasZdotc(f.h, (int)n, f.zx, 1, f.zy, 1, (cuDoubleComplex*)r); },
      [](const F& f, const O& o) { return eqz(o, f.d.dotc_re, f.d.dotc_im); });
  add("Sdot_64", true, [](F& f, int64_t n, void* r) { return cublasSdot_64(f.h, n, f.sx, 1, f.sy, 1, (float*)r); },
      [](const F& f, const O& o) { return eq<float>(o, (float)f.d.dot); });
  add("Ddot_64", true, [](F& f, int64_t n, void* r) { return cublasDdot_64(f.h, n, f.dx, 1, f.dy, 1, (double*)r); },
      [](const F& f, const O& o) { return eq<double>(o, f.d.dot); });
  add("Cdotu_64", true,
      [](F& f, int64_t n, void* r) { return cublasCdotu_64(f.h, n, f.cx, 1, f.cy, 1, (cuComplex*)r); },
      [](const F& f, const O& o) { return eqc(o, (float)f.d.dotu_re, (float)f.d.dotu_im); });
  add("Zdotc_64", true,
      [](F& f, int64_t n, void* r) { return cublasZdotc_64(f.h, n, f.zx, 1, f.zy, 1, (cuDoubleComplex*)r); },
      [](const F& f, const O& o) { return eqz(o, f.d.dotc_re, f.d.dotc_im); });
  add("DotEx_R64F", true,
      [](F& f, int64_t n, void* r) {
        return cublasDotEx(f.h, (int)n, f.dx, CUDA_R_64F, 1, f.dy, CUDA_R_64F, 1, r, CUDA_R_64F, CUDA_R_64F);
      },
      [](const F& f, const O& o) { return eq<double>(o, f.d.dot); });
  add("DotEx_C32F", true,
      [](F& f, int64_t n, void* r) {
        return cublasDotEx(f.h, (int)n, f.cx, CUDA_C_32F, 1, f.cy, CUDA_C_32F, 1, r, CUDA_C_32F, CUDA_C_32F);
      },
      [](const F& f, const O& o) { return eqc(o, (float)f.d.dotu_re, (float)f.d.dotu_im); });
  add("DotcEx_C64F", true,
      [](F& f, int64_t n, void* r) {
        return cublasDotcEx(f.h, (int)n, f.zx, CUDA_C_64F, 1, f.zy, CUDA_C_64F, 1, r, CUDA_C_64F, CUDA_C_64F);
      },
      [](const F& f, const O& o) { return eqz(o, f.d.dotc_re, f.d.dotc_im); });
  add("DotEx_64_R32F", true,
      [](F& f, int64_t n, void* r) {
        return cublasDotEx_64(f.h, n, f.sx, CUDA_R_32F, 1, f.sy, CUDA_R_32F, 1, r, CUDA_R_32F, CUDA_R_32F);
      },
      [](const F& f, const O& o) { return eq<float>(o, (float)f.d.dot); });
  add("DotcEx_64_C32F", true,
      [](F& f, int64_t n, void* r) {
        return cublasDotcEx_64(f.h, n, f.cx, CUDA_C_32F, 1, f.cy, CUDA_C_32F, 1, r, CUDA_C_32F, CUDA_C_32F);
      },
      [](const F& f, const O& o) { return eqc(o, (float)f.d.dotc_re, (float)f.d.dotc_im); });
  // The other reductions.
  add("Snrm2", false, [](F& f, int64_t n, void* r) { return cublasSnrm2(f.h, (int)n, f.sx, 1, (float*)r); },
      [](const F& f, const O& o) { return near<float>(o, std::sqrt(f.d.nrm2sq)); });
  add("Dnrm2", false, [](F& f, int64_t n, void* r) { return cublasDnrm2(f.h, (int)n, f.dx, 1, (double*)r); },
      [](const F& f, const O& o) { return near<double>(o, std::sqrt(f.d.nrm2sq)); });
  add("Scnrm2", false, [](F& f, int64_t n, void* r) { return cublasScnrm2(f.h, (int)n, f.cx, 1, (float*)r); },
      [](const F& f, const O& o) { return near<float>(o, std::sqrt(f.d.cnrm2sq)); });
  add("Dznrm2", false, [](F& f, int64_t n, void* r) { return cublasDznrm2(f.h, (int)n, f.zx, 1, (double*)r); },
      [](const F& f, const O& o) { return near<double>(o, std::sqrt(f.d.cnrm2sq)); });
  add("Dnrm2_64", false, [](F& f, int64_t n, void* r) { return cublasDnrm2_64(f.h, n, f.dx, 1, (double*)r); },
      [](const F& f, const O& o) { return near<double>(o, std::sqrt(f.d.nrm2sq)); });
  add("Sasum", false, [](F& f, int64_t n, void* r) { return cublasSasum(f.h, (int)n, f.sx, 1, (float*)r); },
      [](const F& f, const O& o) { return eq<float>(o, (float)f.d.asum); });
  add("Dasum", false, [](F& f, int64_t n, void* r) { return cublasDasum(f.h, (int)n, f.dx, 1, (double*)r); },
      [](const F& f, const O& o) { return eq<double>(o, f.d.asum); });
  add("Scasum", false, [](F& f, int64_t n, void* r) { return cublasScasum(f.h, (int)n, f.cx, 1, (float*)r); },
      [](const F& f, const O& o) { return eq<float>(o, (float)f.d.casum); });
  add("Dzasum_64", false, [](F& f, int64_t n, void* r) { return cublasDzasum_64(f.h, n, f.zx, 1, (double*)r); },
      [](const F& f, const O& o) { return eq<double>(o, f.d.casum); });
  add("Isamax", false, [](F& f, int64_t n, void* r) { return cublasIsamax(f.h, (int)n, f.sx, 1, (int*)r); },
      [](const F& f, const O& o) { return eq<int>(o, f.d.amax); });
  add("Idamax", false, [](F& f, int64_t n, void* r) { return cublasIdamax(f.h, (int)n, f.dx, 1, (int*)r); },
      [](const F& f, const O& o) { return eq<int>(o, f.d.amax); });
  add("Izamax", false, [](F& f, int64_t n, void* r) { return cublasIzamax(f.h, (int)n, f.zx, 1, (int*)r); },
      [](const F& f, const O& o) { return eq<int>(o, f.d.camax); });
  add("Idamin", false, [](F& f, int64_t n, void* r) { return cublasIdamin(f.h, (int)n, f.dx, 1, (int*)r); },
      [](const F& f, const O& o) { return eq<int>(o, f.d.amin); });
  add("Icamin", false, [](F& f, int64_t n, void* r) { return cublasIcamin(f.h, (int)n, f.cx, 1, (int*)r); },
      [](const F& f, const O& o) { return eq<int>(o, f.d.camin); });
  add("Idamax_64", false, [](F& f, int64_t n, void* r) { return cublasIdamax_64(f.h, n, f.dx, 1, (int64_t*)r); },
      [](const F& f, const O& o) { return eq<int64_t>(o, f.d.amax); });
  add("Isamin_64", false, [](F& f, int64_t n, void* r) { return cublasIsamin_64(f.h, n, f.sx, 1, (int64_t*)r); },
      [](const F& f, const O& o) { return eq<int64_t>(o, f.d.amin); });
  add("Nrm2Ex_R32F", false,
      [](F& f, int64_t n, void* r) { return cublasNrm2Ex(f.h, (int)n, f.sx, CUDA_R_32F, 1, r, CUDA_R_32F, CUDA_R_32F); },
      [](const F& f, const O& o) { return near<float>(o, std::sqrt(f.d.nrm2sq)); });
  add("Nrm2Ex_64_C64F", false,
      [](F& f, int64_t n, void* r) { return cublasNrm2Ex_64(f.h, n, f.zx, CUDA_C_64F, 1, r, CUDA_R_64F, CUDA_R_64F); },
      [](const F& f, const O& o) { return near<double>(o, std::sqrt(f.d.cnrm2sq)); });
  add("AsumEx_R64F", false,
      [](F& f, int64_t n, void* r) { return cublasAsumEx(f.h, (int)n, f.dx, CUDA_R_64F, 1, r, CUDA_R_64F, CUDA_R_64F); },
      [](const F& f, const O& o) { return eq<double>(o, f.d.asum); });
  add("IamaxEx_R64F", false, [](F& f, int64_t n, void* r) { return cublasIamaxEx(f.h, (int)n, f.dx, CUDA_R_64F, 1, (int*)r); },
      [](const F& f, const O& o) { return eq<int>(o, f.d.amax); });
  add("IaminEx_64_C32F", false,
      [](F& f, int64_t n, void* r) { return cublasIaminEx_64(f.h, n, f.cx, CUDA_C_32F, 1, (int64_t*)r); },
      [](const F& f, const O& o) { return eq<int64_t>(o, f.d.camin); });
  return c;
}

void fail(const std::string& what) { throw ::vtest::Failure(what); }

}  // namespace

// The card's answer: a dot product stores through the device pointer; the rest crash there and are refused here,
// with the device word left as it was.
VTEST(cublas_host_mode_device_result) {
  Fixture f;
  for (const Case& c : cases()) {
    const Out o = run(f, CUBLAS_POINTER_MODE_HOST, Where::device, [&](void* r) { return c.call(f, kN, r); });
    if (c.dot) {
      if (o.status != CUBLAS_STATUS_SUCCESS || !c.good(f, o))
        fail(c.name + ": host mode, device result: status " + std::to_string(o.status) + ", wrong or missing value");
    } else if (o.status != CUBLAS_STATUS_INVALID_VALUE || !o.untouched()) {
      fail(c.name + ": host mode, device result: status " + std::to_string(o.status) + ", expected INVALID_VALUE and no write");
    }
  }
}

// A dot product with nothing to sum writes its 0 as host memory on the card, which crashes: refused here too.
VTEST(cublas_host_mode_device_result_empty_dot) {
  Fixture f;
  for (const Case& c : cases()) {
    if (!c.dot) continue;
    for (int64_t n : {0, -1}) {
      const Out o = run(f, CUBLAS_POINTER_MODE_HOST, Where::device, [&](void* r) { return c.call(f, n, r); });
      if (o.status != CUBLAS_STATUS_INVALID_VALUE || !o.untouched())
        fail(c.name + ": n = " + std::to_string(n) + ": status " + std::to_string(o.status) + ", expected INVALID_VALUE and no write");
    }
  }
}

// The documented modes are unchanged: a host result in host mode, a device result in device mode, and a managed
// result (host memory) in host mode, every reduction, with the right value.
VTEST(cublas_reductions_in_their_documented_modes) {
  Fixture f;
  const struct { cublasPointerMode_t mode; Where where; const char* name; } modes[] = {
      {CUBLAS_POINTER_MODE_HOST, Where::host, "host mode, host result"},
      {CUBLAS_POINTER_MODE_DEVICE, Where::device, "device mode, device result"},
      {CUBLAS_POINTER_MODE_HOST, Where::managed, "host mode, managed result"},
  };
  for (const auto& m : modes)
    for (const Case& c : cases()) {
      const Out o = run(f, m.mode, m.where, [&](void* r) { return c.call(f, kN, r); });
      if (o.status != CUBLAS_STATUS_SUCCESS || !c.good(f, o))
        fail(c.name + ": " + m.name + ": status " + std::to_string(o.status) + ", wrong or missing value");
    }
  // An empty dot product into a host result is 0, as before.
  const Out o = run(f, CUBLAS_POINTER_MODE_HOST, Where::host, [&](void* r) { return cublasDdot(f.h, 0, f.dx, 1, f.dy, 1, (double*)r); });
  VCHECK_EQ(o.status, CUBLAS_STATUS_SUCCESS);
  VCHECK(o.as<double>() == 0.0);
}

VTEST_MAIN
