// cuTensorNet's decompositions beyond the default SVD, and the gradients of a network:
//
//  - SVD with every algorithm (gesvd, gesvdj, gesvdp, gesvdr): reconstruction, orthogonality,
//    the singular values against a host reference, the information each reports (the
//    residual and sweeps of gesvdj, the error in sigma of gesvdp), the parameters each takes
//    and the host scratch gesvdr needs;
//  - what is refused: half precision (every call), the complex half type (the descriptor),
//    decompositions under stream capture (CUDA_ERROR; a QR also invalidates the capture),
//    and a workspace without memory (NO_DEVICE_ALLOCATOR);
//  - gradients of a network: values for real and complex data (a complex gradient is the
//    adjoint times the conjugate of the other tensors), overwriting and accumulating, and the
//    statuses of the argument errors.
//
// Every expectation was measured against NVIDIA's libcutensornet 2.14 on an RTX 3060, and
// the program passes against it as against this library. Numbers are checked against a
// reference computed here, never against the other library's.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "../../include/vgpu_cutensornet.h"

using cd = std::complex<double>;

static int failures = 0;
static cutensornetHandle_t h;

static void check(bool ok, const std::string& what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

static const char* tname(cudaDataType_t t) {
  switch (t) {
    case CUDA_R_16F: return "R16F";
    case CUDA_R_16BF: return "R16BF";
    case CUDA_C_16F: return "C16F";
    case CUDA_R_32F: return "R32F";
    case CUDA_R_64F: return "R64F";
    case CUDA_C_32F: return "C32F";
    default: return "C64F";
  }
}
static bool is_cx(cudaDataType_t t) { return t == CUDA_C_32F || t == CUDA_C_64F; }
static size_t eb(cudaDataType_t t) { return t == CUDA_R_32F ? 4 : t == CUDA_R_64F ? 8 : t == CUDA_C_32F ? 8 : 16; }
static size_t rb(cudaDataType_t t) { return (t == CUDA_R_64F || t == CUDA_C_64F) ? 8 : 4; }

// ---- host helpers ----

static unsigned rng_state = 424242;
static double rnd() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return ((double)(rng_state % 20001) - 10000.0) / 10000.0;
}

static void* dev(size_t bytes) {
  void* p = nullptr;
  cudaMalloc(&p, bytes ? bytes : 1);
  cudaMemset(p, 0, bytes ? bytes : 1);
  return p;
}

static void put(void* p, cudaDataType_t t, const std::vector<cd>& v) {
  std::vector<char> raw(v.size() * eb(t));
  for (size_t i = 0; i < v.size(); ++i) {
    char* e = raw.data() + i * eb(t);
    if (t == CUDA_R_32F) { float x = (float)v[i].real(); std::memcpy(e, &x, 4); }
    else if (t == CUDA_R_64F) { double x = v[i].real(); std::memcpy(e, &x, 8); }
    else if (t == CUDA_C_32F) { float x[2] = {(float)v[i].real(), (float)v[i].imag()}; std::memcpy(e, x, 8); }
    else { double x[2] = {v[i].real(), v[i].imag()}; std::memcpy(e, x, 16); }
  }
  if (!raw.empty()) cudaMemcpy(p, raw.data(), raw.size(), cudaMemcpyHostToDevice);
}

static std::vector<cd> get(const void* p, cudaDataType_t t, size_t n) {
  std::vector<char> raw(n * eb(t));
  if (n) cudaMemcpy(raw.data(), p, raw.size(), cudaMemcpyDeviceToHost);
  std::vector<cd> o(n);
  for (size_t i = 0; i < n; ++i) {
    const char* e = raw.data() + i * eb(t);
    if (t == CUDA_R_32F) { float x; std::memcpy(&x, e, 4); o[i] = x; }
    else if (t == CUDA_R_64F) { double x; std::memcpy(&x, e, 8); o[i] = x; }
    else if (t == CUDA_C_32F) { float x[2]; std::memcpy(x, e, 8); o[i] = cd(x[0], x[1]); }
    else { double x[2]; std::memcpy(x, e, 16); o[i] = cd(x[0], x[1]); }
  }
  return o;
}

static std::vector<double> get_real(const void* p, cudaDataType_t t, size_t n) {
  std::vector<double> o(n);
  if (rb(t) == 8) {
    cudaMemcpy(o.data(), p, n * 8, cudaMemcpyDeviceToHost);
  } else {
    std::vector<float> f(n);
    cudaMemcpy(f.data(), p, n * 4, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n; ++i) o[i] = f[i];
  }
  return o;
}

// Singular values of an m x n column-major matrix by one-sided Jacobi.
static std::vector<double> host_singular_values(std::vector<cd> a, int m, int n) {
  for (int sweep = 0; sweep < 60; ++sweep) {
    double off = 0;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) {
        double alpha = 0, beta = 0;
        cd gamma = 0;
        for (int i = 0; i < m; ++i) {
          alpha += std::norm(a[(size_t)(i + p * m)]);
          beta += std::norm(a[(size_t)(i + q * m)]);
          gamma += std::conj(a[(size_t)(i + p * m)]) * a[(size_t)(i + q * m)];
        }
        const double g = std::abs(gamma);
        off = std::max(off, g / std::sqrt(alpha * beta + 1e-300));
        if (g < 1e-300) continue;
        const cd phase = gamma / g;
        const double zeta = (beta - alpha) / (2 * g);
        const double tt = (zeta >= 0 ? 1.0 : -1.0) / (std::fabs(zeta) + std::sqrt(1 + zeta * zeta));
        const double c = 1 / std::sqrt(1 + tt * tt), s = c * tt;
        for (int i = 0; i < m; ++i) {
          const cd x = a[(size_t)(i + p * m)], y = a[(size_t)(i + q * m)];
          a[(size_t)(i + p * m)] = c * x - s * std::conj(phase) * y;
          a[(size_t)(i + q * m)] = s * phase * x + c * y;
        }
      }
    if (off < 1e-15) break;
  }
  std::vector<double> sv(n);
  for (int j = 0; j < n; ++j) {
    double s = 0;
    for (int i = 0; i < m; ++i) s += std::norm(a[(size_t)(i + j * m)]);
    sv[(size_t)j] = std::sqrt(s);
  }
  std::sort(sv.begin(), sv.end(), std::greater<double>());
  return sv;
}

// ---- descriptors, workspaces ----

struct Desc {
  cutensornetTensorDescriptor_t d = nullptr;
  cutensornetStatus_t status = CUTENSORNET_STATUS_SUCCESS;
  Desc(std::vector<int32_t> modes, std::vector<int64_t> ext, cudaDataType_t t) {
    status = cutensornetCreateTensorDescriptor(h, (int)modes.size(), ext.data(), nullptr, modes.data(), t, &d);
  }
  ~Desc() {
    if (d) cutensornetDestroyTensorDescriptor(d);
  }
};

struct Work {
  cutensornetWorkspaceDescriptor_t w = nullptr;
  void* dev_ptr = nullptr;
  void* cache = nullptr;
  std::vector<char> host;
  Work() { cutensornetCreateWorkspaceDescriptor(h, &w); }
  ~Work() {
    cudaFree(dev_ptr);
    cudaFree(cache);
    cutensornetDestroyWorkspaceDescriptor(w);
  }
  // Gives the scratch (device, and host when asked) and the cache the sizes reported.
  void provide(bool host_too = true, bool with_cache = false) {
    int64_t sz = 0, hs = 0, cs = 0;
    cutensornetWorkspaceGetMemorySize(h, w, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE,
                                      CUTENSORNET_WORKSPACE_SCRATCH, &sz);
    cutensornetWorkspaceGetMemorySize(h, w, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_HOST,
                                      CUTENSORNET_WORKSPACE_SCRATCH, &hs);
    cutensornetWorkspaceGetMemorySize(h, w, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE,
                                      CUTENSORNET_WORKSPACE_CACHE, &cs);
    cudaFree(dev_ptr);
    dev_ptr = dev((size_t)sz + 256);
    cutensornetWorkspaceSetMemory(h, w, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, dev_ptr, sz + 256);
    if (host_too && hs > 0) {
      host.assign((size_t)hs, 0);
      cutensornetWorkspaceSetMemory(h, w, CUTENSORNET_MEMSPACE_HOST, CUTENSORNET_WORKSPACE_SCRATCH, host.data(), hs);
    }
    if (with_cache) {
      cudaFree(cache);
      cache = dev((size_t)cs + 256);
      cutensornetWorkspaceSetMemory(h, w, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_CACHE, cache, cs + 256);
    }
  }
  int64_t size(cutensornetMemspace_t m, cutensornetWorkspaceKind_t k = CUTENSORNET_WORKSPACE_SCRATCH) const {
    int64_t v = -1;
    cutensornetWorkspaceGetMemorySize(h, w, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, m, k, &v);
    return v;
  }
};

// ---- SVD ----

// A is m x n = (i j | k l), made of a rank-four product plus noise, so that the leading
// four singular values stand clear of the rest and a randomized SVD finds them.
static const int64_t EI = 3, EJ = 4, EK = 5, EL = 2;
static const int64_t M = EI * EJ, N = EK * EL, FULL = std::min(M, N);

static std::vector<cd> low_rank_matrix(bool cx) {
  std::vector<cd> p((size_t)(M * 4)), q((size_t)(4 * N)), a((size_t)(M * N));
  for (auto& e : p) { const double r = rnd(); e = cx ? cd(r, rnd()) : cd(r, 0); }
  for (auto& e : q) { const double r = rnd(); e = cx ? cd(r, rnd()) : cd(r, 0); }
  const double scale[4] = {5, 3, 1.5, 0.7};
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j) {
      cd acc = 0;
      for (int k = 0; k < 4; ++k) acc += p[(size_t)(i + k * M)] * scale[k] * q[(size_t)(k + 4 * j)];
      const double r = 1e-7 * rnd();
      a[(size_t)(i + j * M)] = acc + (cx ? cd(r, 1e-7 * rnd()) : cd(r, 0));
    }
  return a;
}

static std::vector<cd> random_matrix(bool cx) {
  std::vector<cd> a((size_t)(M * N));
  for (auto& e : a) { const double r = rnd(); e = cx ? cd(r, rnd()) : cd(r, 0); }
  return a;
}

struct SvdRun {
  cutensornetStatus_t sizes = CUTENSORNET_STATUS_SUCCESS, svd = CUTENSORNET_STATUS_SUCCESS;
  int64_t full = -1, reduced = -1;
  double discarded = -1;
  int32_t algo = -1;
  std::vector<cd> u, v;
  std::vector<double> s;
  int64_t kx = 0;
  cutensornetGesvdjStatus_t jstat{-1, -1};
  cutensornetGesvdpStatus_t pstat{-1};
  cutensornetStatus_t jstat_rc = CUTENSORNET_STATUS_SUCCESS, pstat_rc = CUTENSORNET_STATUS_SUCCESS;
};

static SvdRun run_svd(cudaDataType_t t, const std::vector<cd>& A, int algo, int64_t xext, bool host_scratch = true,
                      const cutensornetGesvdrParams_t* rp = nullptr) {
  SvdRun r;
  void* dA = dev((size_t)(M * N) * eb(t));
  put(dA, t, A);
  Desc in({'i', 'j', 'k', 'l'}, {EI, EJ, EK, EL}, t);
  Desc U({'i', 'j', 'x'}, {EI, EJ, xext}, t);
  Desc V({'x', 'k', 'l'}, {xext, EK, EL}, t);
  void* dU = dev((size_t)(M * xext) * eb(t));
  void* dV = dev((size_t)(xext * N) * eb(t));
  void* dS = dev((size_t)xext * rb(t));
  cutensornetTensorSVDConfig_t cfg;
  cutensornetCreateTensorSVDConfig(h, &cfg);
  int a = algo;
  cutensornetTensorSVDConfigSetAttribute(h, cfg, CUTENSORNET_TENSOR_SVD_CONFIG_ALGO, &a, sizeof a);
  if (algo == CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ) {
    cutensornetGesvdjParams_t p{rb(t) == 4 ? 1e-6 : 1e-12, 80};
    cutensornetTensorSVDConfigSetAttribute(h, cfg, CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS, &p, sizeof p);
  } else if (algo == CUTENSORNET_TENSOR_SVD_ALGO_GESVDR) {
    cutensornetGesvdrParams_t p = rp ? *rp : cutensornetGesvdrParams_t{2, 4};
    cutensornetTensorSVDConfigSetAttribute(h, cfg, CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS, &p, sizeof p);
  }
  Work w;
  r.sizes = cutensornetWorkspaceComputeSVDSizes(h, in.d, U.d, V.d, cfg, w.w);
  cutensornetTensorSVDInfo_t info;
  cutensornetCreateTensorSVDInfo(h, &info);
  if (r.sizes == CUTENSORNET_STATUS_SUCCESS) {
    w.provide(host_scratch);
    r.svd = cutensornetTensorSVD(h, in.d, dA, U.d, dU, dS, V.d, dV, cfg, info, w.w, 0);
    cudaDeviceSynchronize();
    if (r.svd == CUTENSORNET_STATUS_SUCCESS) {
      cutensornetTensorSVDInfoGetAttribute(h, info, CUTENSORNET_TENSOR_SVD_INFO_FULL_EXTENT, &r.full, 8);
      cutensornetTensorSVDInfoGetAttribute(h, info, CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT, &r.reduced, 8);
      cutensornetTensorSVDInfoGetAttribute(h, info, CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT, &r.discarded, 8);
      cutensornetTensorSVDInfoGetAttribute(h, info, CUTENSORNET_TENSOR_SVD_INFO_ALGO, &r.algo, 4);
      r.jstat_rc = (cutensornetStatus_t)cutensornetTensorSVDInfoGetAttribute(h, info, CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS,
                                                                            &r.jstat, sizeof r.jstat);
      r.pstat_rc = (cutensornetStatus_t)cutensornetTensorSVDInfoGetAttribute(h, info, CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS,
                                                                            &r.pstat, sizeof r.pstat);
      int32_t nm;
      size_t ds;
      int32_t lab[3];
      int64_t ex[3], str[3];
      cutensornetGetTensorDetails(h, U.d, &nm, &ds, lab, ex, str);
      r.kx = ex[2];
      r.u = get(dU, t, (size_t)(M * r.kx));
      r.v = get(dV, t, (size_t)(r.kx * N));
      r.s = get_real(dS, t, (size_t)r.kx);
    }
  }
  cutensornetDestroyTensorSVDInfo(info);
  cutensornetDestroyTensorSVDConfig(cfg);
  cudaFree(dA);
  cudaFree(dU);
  cudaFree(dV);
  cudaFree(dS);
  return r;
}

// The relative error of U S V against A, and how far U^H U is from the identity.
static void quality(const SvdRun& r, const std::vector<cd>& A, double* recon, double* orth) {
  double err = 0, nrm = 0;
  const int64_t kx = r.kx;
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j) {
      cd acc = 0;
      for (int64_t q = 0; q < kx; ++q) acc += r.u[(size_t)(i + M * q)] * r.s[(size_t)q] * r.v[(size_t)(q + kx * j)];
      err += std::norm(acc - A[(size_t)(i + j * M)]);
      nrm += std::norm(A[(size_t)(i + j * M)]);
    }
  *recon = std::sqrt(err / nrm);
  double o = 0;
  for (int64_t p = 0; p < kx; ++p)
    for (int64_t q = 0; q < kx; ++q) {
      cd acc = 0;
      for (int64_t i = 0; i < M; ++i) acc += std::conj(r.u[(size_t)(i + M * p)]) * r.u[(size_t)(i + M * q)];
      o = std::max(o, std::abs(acc - (p == q ? 1.0 : 0.0)));
    }
  *orth = o;
}

static void svd_algorithms() {
  const char* names[] = {"gesvd", "gesvdj", "gesvdp", "gesvdr"};
  for (cudaDataType_t t : {CUDA_R_32F, CUDA_R_64F, CUDA_C_32F, CUDA_C_64F}) {
    const bool fp32 = rb(t) == 4;
    const double tol = fp32 ? 5e-5 : 1e-11;
    const std::vector<cd> A = random_matrix(is_cx(t));   // well conditioned: every algorithm finds all of it
    const std::vector<cd> L = low_rank_matrix(is_cx(t)); // a rank-four matrix and noise
    const std::vector<double> ref = host_singular_values(A, (int)M, (int)N);
    const std::vector<double> lref = host_singular_values(L, (int)M, (int)N);
    double ltotal = 0;
    for (double x : lref) ltotal += x * x;
    for (int algo = 0; algo < 4; ++algo) {
      const std::string tag = std::string(names[algo]) + " " + tname(t) + ": ";
      if (algo != CUTENSORNET_TENSOR_SVD_ALGO_GESVDR) {
        // The whole spectrum.
        SvdRun r = run_svd(t, A, algo, FULL);
        check(r.sizes == 0 && r.svd == 0, tag + "SVD of the full extent succeeds");
        if (r.svd == 0) {
          double recon, orth;
          quality(r, A, &recon, &orth);
          check(recon < tol && orth < tol, tag + "U S V rebuilds A and U is orthonormal");
          double serr = 0;
          for (int64_t j = 0; j < FULL; ++j) serr = std::max(serr, std::fabs(r.s[(size_t)j] - ref[(size_t)j]));
          check(serr < tol * ref[0], tag + "the singular values are the reference's");
          bool sorted = true;
          for (int64_t j = 1; j < FULL; ++j) sorted = sorted && r.s[(size_t)j] <= r.s[(size_t)j - 1];
          check(sorted, tag + "in descending order");
          check(r.full == FULL && r.reduced == FULL && r.discarded < 1e-12 && r.algo == algo, tag + "extents, weight and algorithm in the info");
          if (algo == CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ)
            check(r.jstat_rc == 0 && r.jstat.sweeps >= 1 && r.jstat.sweeps <= 80 && r.jstat.residual >= 0 && r.jstat.residual < 1e-3,
                  tag + "reports its residual and sweeps");
          else if (algo == CUTENSORNET_TENSOR_SVD_ALGO_GESVDP)
            check(r.pstat_rc == 0 && r.pstat.errSigma >= 0 && r.pstat.errSigma < 1e-3, tag + "reports the error in sigma");
          else
            check(r.jstat_rc == CUTENSORNET_STATUS_INVALID_VALUE, tag + "has no algorithm status");
        }
      }
      // Truncated to four: the rank-four matrix survives.
      SvdRun r = run_svd(t, L, algo, 4);
      check(r.sizes == 0 && r.svd == 0, tag + "SVD truncated to extent 4 succeeds");
      if (r.svd == 0) {
        double recon, orth;
        quality(r, L, &recon, &orth);
        check(recon < (fp32 ? 2e-3 : 1e-5) && orth < (fp32 ? 1e-3 : 1e-9), tag + "extent 4 keeps the rank-four matrix");
        double serr = 0;
        for (int j = 0; j < 4; ++j) serr = std::max(serr, std::fabs(r.s[(size_t)j] - lref[(size_t)j]));
        check(serr < (fp32 ? 2e-3 : 1e-5) * lref[0], tag + "and its four singular values");
        double tail = 0;
        for (int j = 4; j < FULL; ++j) tail += lref[(size_t)j] * lref[(size_t)j];
        if (algo == CUTENSORNET_TENSOR_SVD_ALGO_GESVDR)
          check(r.discarded == 0 && r.reduced == 4 && r.full == FULL, tag + "reports no discarded weight, as NVIDIA's does");
        else
          check(r.reduced == 4 && r.full == FULL && std::fabs(r.discarded - tail / ltotal) < 1e-5, tag + "reports what the extent drops");
      }
    }
  }
}

static void svd_parameters() {
  const std::vector<cd> A = low_rank_matrix(false);
  cutensornetTensorSVDConfig_t cfg;
  cutensornetCreateTensorSVDConfig(h, &cfg);
  unsigned char buf[40];
  for (int algo = 0; algo < 4; ++algo) {
    int a = algo;
    cutensornetTensorSVDConfigSetAttribute(h, cfg, CUTENSORNET_TENSOR_SVD_CONFIG_ALGO, &a, sizeof a);
    std::memset(buf, 0x5a, sizeof buf);
    const int rc16 = cutensornetTensorSVDConfigGetAttribute(h, cfg, CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS, buf, 16);
    const int rc8 = cutensornetTensorSVDConfigGetAttribute(h, cfg, CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS, buf, 8);
    const int rc24 = cutensornetTensorSVDConfigGetAttribute(h, cfg, CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS, buf, 24);
    const bool takes = algo == CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ || algo == CUTENSORNET_TENSOR_SVD_ALGO_GESVDR;
    check(rc16 == (takes ? 0 : 7) && rc8 == 7 && rc24 == (takes ? 0 : 7),
          std::string("algorithm ") + std::to_string(algo) + (takes ? ": parameters read back at 16 bytes and more, not at 8" : ": takes no parameters"));
    if (takes) {
      std::memset(buf, 0x5a, sizeof buf);
      cutensornetTensorSVDConfigGetAttribute(h, cfg, CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS, buf, 16);
      bool zero;
      if (algo == CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ) {
        cutensornetGesvdjParams_t p;
        std::memcpy(&p.tol, buf, 8);
        std::memcpy(&p.maxSweeps, buf + 8, 4);
        zero = p.tol == 0 && p.maxSweeps == 0;
      } else {
        cutensornetGesvdrParams_t p;
        std::memcpy(&p, buf, 16);
        zero = p.oversampling == 0 && p.niters == 0;
      }
      check(zero, std::string("algorithm ") + std::to_string(algo) + ": parameters default to zero");
    }
  }
  cutensornetDestroyTensorSVDConfig(cfg);

  // gesvdr: the rank plus the oversampling must fit in min(m, n), and it needs host scratch.
  const cudaDataType_t t = CUDA_R_64F;
  cutensornetGesvdrParams_t p{2, 4};
  SvdRun r = run_svd(t, A, CUTENSORNET_TENSOR_SVD_ALGO_GESVDR, 8, true, &p);
  check(r.sizes == 0 && r.svd == 0, "gesvdr: rank 8 with oversampling 2 fits in 10");
  r = run_svd(t, A, CUTENSORNET_TENSOR_SVD_ALGO_GESVDR, 9, true, &p);
  check(r.sizes == CUTENSORNET_STATUS_INVALID_VALUE, "gesvdr: rank 9 with oversampling 2 does not");
  cutensornetGesvdrParams_t big{10, 4};
  r = run_svd(t, A, CUTENSORNET_TENSOR_SVD_ALGO_GESVDR, 4, true, &big);
  check(r.sizes == CUTENSORNET_STATUS_INVALID_VALUE, "gesvdr: oversampling 10 at rank 4 does not");
  r = run_svd(t, A, CUTENSORNET_TENSOR_SVD_ALGO_GESVDR, 4, false, &p);
  check(r.sizes == 0 && r.svd == CUTENSORNET_STATUS_INTERNAL_ERROR, "gesvdr without host scratch memory is INTERNAL_ERROR");
  cutensornetGesvdrParams_t none{0, 0};
  r = run_svd(t, A, CUTENSORNET_TENSOR_SVD_ALGO_GESVDR, 4, true, &none);
  check(r.sizes == 0 && r.svd == 0, "gesvdr with the default parameters (no oversampling, no iterations) runs");
}

// ---- half precision ----

static void half_precision() {
  for (cudaDataType_t t : {CUDA_R_16F, CUDA_R_16BF}) {
    Desc in({'i', 'j', 'k', 'l'}, {EI, EJ, EK, EL}, t);
    Desc U({'i', 'j', 'x'}, {EI, EJ, FULL}, t);
    Desc V({'x', 'k', 'l'}, {FULL, EK, EL}, t);
    Desc Q({'i', 'j', 'x'}, {EI, EJ, FULL}, t);
    Desc R({'x', 'k', 'l'}, {FULL, EK, EL}, t);
    check(in.status == 0 && U.status == 0, std::string(tname(t)) + ": tensor descriptors are made");
    Work w;
    cutensornetTensorSVDConfig_t cfg;
    cutensornetCreateTensorSVDConfig(h, &cfg);
    IS(cutensornetWorkspaceComputeSVDSizes(h, in.d, U.d, V.d, cfg, w.w), 7);
    IS(cutensornetWorkspaceComputeQRSizes(h, in.d, Q.d, R.d, w.w), 7);
    void* mem = dev(1 << 20);
    cutensornetWorkspaceSetMemory(h, w.w, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, mem, 1 << 20);
    void* a = dev(1 << 12);
    void* o1 = dev(1 << 12);
    void* o2 = dev(1 << 12);
    void* s = dev(1 << 12);
    IS(cutensornetTensorQR(h, in.d, a, Q.d, o1, R.d, o2, w.w, 0), 7);
    IS(cutensornetTensorSVD(h, in.d, a, U.d, o1, s, V.d, o2, cfg, nullptr, w.w, 0), 7);
    cudaFree(mem); cudaFree(a); cudaFree(o1); cudaFree(o2); cudaFree(s);
    cutensornetDestroyTensorSVDConfig(cfg);
  }
  cutensornetTensorDescriptor_t d = nullptr;
  int64_t ext[1] = {4};
  int32_t mode[1] = {'a'};
  IS(cutensornetCreateTensorDescriptor(h, 1, ext, nullptr, mode, CUDA_C_16F, &d), 15);
}

// ---- stream capture ----

static void capture() {
  cudaStream_t s;
  cudaStreamCreate(&s);
  const cudaDataType_t t = CUDA_R_32F;
  const std::vector<cd> A = low_rank_matrix(false);
  void* dA = dev((size_t)(M * N) * 4);
  put(dA, t, A);
  Desc in({'i', 'j', 'k', 'l'}, {EI, EJ, EK, EL}, t);
  Desc Q({'i', 'j', 'x'}, {EI, EJ, FULL}, t);
  Desc R({'x', 'k', 'l'}, {FULL, EK, EL}, t);
  void* dQ = dev((size_t)(M * FULL) * 4);
  void* dR = dev((size_t)(FULL * N) * 4);
  void* dS = dev((size_t)FULL * 4);
  Work wq;
  IS(cutensornetWorkspaceComputeQRSizes(h, in.d, Q.d, R.d, wq.w), 0);
  wq.provide();
  cutensornetTensorSVDConfig_t cfg;
  cutensornetCreateTensorSVDConfig(h, &cfg);
  Work ws;
  IS(cutensornetWorkspaceComputeSVDSizes(h, in.d, Q.d, R.d, cfg, ws.w), 0);
  ws.provide();
  for (int what = 0; what < 2; ++what) {
    cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    cutensornetStatus_t st = what == 0 ? cutensornetTensorQR(h, in.d, dA, Q.d, dQ, R.d, dR, wq.w, s)
                                       : cutensornetTensorSVD(h, in.d, dA, Q.d, dQ, dS, R.d, dR, cfg, nullptr, ws.w, s);
    cudaGraph_t g = nullptr;
    const cudaError_t e = cudaStreamEndCapture(s, &g);
    cudaGetLastError();
    size_t nodes = 0;
    if (g) cudaGraphGetNodes(g, nullptr, &nodes);
    const std::string name = what == 0 ? "QR" : "SVD";
    check(st == CUTENSORNET_STATUS_CUDA_ERROR, name + " during stream capture is CUDA_ERROR");
    // A QR leaves the capture invalidated; an SVD fails before touching the stream and the empty capture ends normally.
    if (what == 0) check(e == cudaErrorStreamCaptureInvalidated && g == nullptr, "and the QR's capture is invalidated, so there is no graph");
    else check(e == cudaSuccess && g != nullptr && nodes == 0, "and the SVD's capture ends with an empty graph");
    if (g) cudaGraphDestroy(g);
  }
  IS(cutensornetTensorQR(h, in.d, dA, Q.d, dQ, R.d, dR, wq.w, s), 0);
  cutensornetDestroyTensorSVDConfig(cfg);
  cudaFree(dA); cudaFree(dQ); cudaFree(dR); cudaFree(dS);
  cudaStreamDestroy(s);
}


// ---- gradients ----

static std::map<int32_t, int64_t> EXT = {{'i', 3}, {'j', 4}, {'k', 5}, {'l', 2}};
static size_t vol(const std::vector<int32_t>& m) {
  size_t n = 1;
  for (int32_t x : m) n *= (size_t)EXT[x];
  return n;
}

struct HT {
  std::vector<int32_t> modes;
  std::vector<cd> v;
};
static HT mk(std::vector<int32_t> modes, bool cx) {
  HT t;
  t.modes = modes;
  t.v.resize(vol(modes));
  for (auto& e : t.v) { const double a = rnd(); e = cx ? cd(a, rnd()) : cd(a, 0); }
  return t;
}

static std::vector<cd> einsum(const std::vector<const HT*>& in, const std::vector<int32_t>& out) {
  std::vector<int32_t> all;
  for (auto* t : in)
    for (int32_t m : t->modes)
      if (std::find(all.begin(), all.end(), m) == all.end()) all.push_back(m);
  std::vector<int32_t> sum;
  for (int32_t m : all)
    if (std::find(out.begin(), out.end(), m) == out.end()) sum.push_back(m);
  std::vector<cd> r(vol(out), 0);
  std::map<int32_t, int64_t> idx;
  auto off = [&](const std::vector<int32_t>& modes) {
    size_t o = 0, s = 1;
    for (int32_t m : modes) { o += (size_t)idx[m] * s; s *= (size_t)EXT[m]; }
    return o;
  };
  for (size_t o = 0; o < r.size(); ++o) {
    size_t rem = o;
    for (int32_t m : out) { idx[m] = (int64_t)(rem % (size_t)EXT[m]); rem /= (size_t)EXT[m]; }
    size_t ns = 1;
    for (int32_t m : sum) ns *= (size_t)EXT[m];
    cd acc = 0;
    for (size_t si = 0; si < ns; ++si) {
      size_t rem2 = si;
      for (int32_t m : sum) { idx[m] = (int64_t)(rem2 % (size_t)EXT[m]); rem2 /= (size_t)EXT[m]; }
      cd p = 1;
      for (auto* t : in) p *= t->v[off(t->modes)];
      acc += p;
    }
    r[o] = acc;
  }
  return r;
}

static double rel(const std::vector<cd>& a, const std::vector<cd>& b) {
  double e = 0, n = 0;
  for (size_t i = 0; i < a.size(); ++i) { e += std::norm(a[i] - b[i]); n += std::norm(b[i]); }
  return std::sqrt(e / (n > 0 ? n : 1));
}

static void* filled(size_t n, cudaDataType_t t, double v) {
  void* p = dev(n * eb(t));
  put(p, t, std::vector<cd>(n, cd(v, 0)));
  return p;
}

static void gradients() {
  for (cudaDataType_t t : {CUDA_R_32F, CUDA_R_64F, CUDA_C_32F, CUDA_C_64F}) {
    const bool cx = is_cx(t);
    const double tol = rb(t) == 4 ? 2e-5 : 1e-12;
    for (int variant = 0; variant < 3; ++variant) {
      const std::string tag = std::string(tname(t)) + " gradients of " + (variant == 0 ? "A and C" : variant == 1 ? "B" : "A, B and C") + ": ";
      // ijk,kl,jl->i
      HT A = mk({'i', 'j', 'k'}, cx), B = mk({'k', 'l'}, cx), C = mk({'j', 'l'}, cx);
      HT adj = mk({'i'}, cx);
      const HT* ts[3] = {&A, &B, &C};
      std::vector<int32_t> out = {'i'};
      cutensornetNetworkDescriptor_t net;
      cutensornetCreateNetwork(h, &net);
      int64_t id[3];
      void* d[3];
      for (int q = 0; q < 3; ++q) {
        std::vector<int64_t> ex;
        for (int32_t m : ts[q]->modes) ex.push_back(EXT[m]);
        cutensornetNetworkAppendTensor(h, net, (int)ts[q]->modes.size(), ex.data(), ts[q]->modes.data(), nullptr, t, &id[q]);
        d[q] = dev(ts[q]->v.size() * eb(t));
        put(d[q], t, ts[q]->v);
        cutensornetNetworkSetInputTensorMemory(h, net, id[q], d[q], nullptr);
      }
      cutensornetNetworkSetOutputTensor(h, net, 1, out.data(), t);
      void* dout = dev(vol(out) * eb(t));
      cutensornetNetworkSetOutputTensorMemory(h, net, dout, nullptr);
      std::vector<int> want = variant == 0 ? std::vector<int>{0, 2} : variant == 1 ? std::vector<int>{1} : std::vector<int>{0, 1, 2};
      std::vector<int32_t> ids;
      for (int q : want) ids.push_back((int32_t)id[q]);
      cutensornetTensorIDList_t lst{(int32_t)ids.size(), ids.data()};
      IS(cutensornetNetworkSetAttribute(h, net, CUTENSORNET_NETWORK_INPUT_TENSORS_REQUIRE_GRAD, &lst, sizeof lst), 0);
      int32_t nreq = -1;
      cutensornetNetworkGetAttribute(h, net, CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_REQUIRE_GRAD, &nreq, sizeof nreq);
      check(nreq == (int32_t)want.size(), tag + "the count of tensors that require gradients reads back");
      void* dadj = dev(adj.v.size() * eb(t));
      put(dadj, t, adj.v);
      IS(cutensornetNetworkSetAdjointTensorMemory(h, net, dadj, nullptr), 0);
      void* dg[3] = {nullptr, nullptr, nullptr};
      for (int q : want) {
        dg[q] = filled(ts[q]->v.size(), t, 5.0);  // a sentinel, to tell overwriting from adding
        IS(cutensornetNetworkSetGradientTensorMemory(h, net, id[q], dg[q], nullptr), 0);
      }
      cutensornetContractionOptimizerConfig_t cfg;
      cutensornetCreateContractionOptimizerConfig(h, &cfg);
      cutensornetContractionOptimizerInfo_t info;
      cutensornetCreateContractionOptimizerInfo(h, net, &info);
      IS(cutensornetContractionOptimize(h, net, cfg, 1ull << 30, info), 0);
      Work w;
      IS(cutensornetWorkspaceComputeContractionSizes(h, net, info, w.w), 0);
      check(w.size(CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_CACHE) > 0, tag + "gradients ask for cache workspace");
      w.provide(false, true);
      IS(cutensornetNetworkPrepareContraction(h, net, w.w), 0);
      IS(cutensornetNetworkPrepareGradientsBackward(h, net, w.w), 0);
      IS(cutensornetNetworkContract(h, net, 0, w.w, nullptr, 0), 0);
      IS(cutensornetNetworkComputeGradientsBackward(h, net, 0, w.w, nullptr, 0), 0);
      cudaDeviceSynchronize();
      bool ok = true;
      for (int q : want) {
        HT adjx = adj;
        std::vector<HT> others;
        for (int r = 0; r < 3; ++r) {
          if (r == q) continue;
          HT o = *ts[r];
          for (auto& e : o.v) e = std::conj(e);  // the gradient takes the conjugate of the other tensors
          others.push_back(o);
        }
        std::vector<const HT*> ins = {&adjx};
        for (auto& o : others) ins.push_back(&o);
        const std::vector<cd> ref = einsum(ins, ts[q]->modes);
        ok = ok && rel(get(dg[q], t, ref.size()), ref) < tol;
      }
      check(ok, tag + "each is the adjoint times the conjugate of the other tensors");
      IS(cutensornetNetworkComputeGradientsBackward(h, net, 1, w.w, nullptr, 0), 0);
      cudaDeviceSynchronize();
      ok = true;
      for (int q : want) {
        std::vector<HT> others;
        for (int r = 0; r < 3; ++r) {
          if (r == q) continue;
          HT o = *ts[r];
          for (auto& e : o.v) e = std::conj(e);
          others.push_back(o);
        }
        HT adjx = adj;
        std::vector<const HT*> ins = {&adjx};
        for (auto& o : others) ins.push_back(&o);
        std::vector<cd> ref = einsum(ins, ts[q]->modes);
        for (auto& e : ref) e *= 2.0;
        ok = ok && rel(get(dg[q], t, ref.size()), ref) < tol;
      }
      check(ok, tag + "accumulating adds a second copy");
      // The forward result is untouched by the backward pass.
      {
        std::vector<const HT*> ins = {&A, &B, &C};
        check(rel(get(dout, t, vol(out)), einsum(ins, out)) < tol, tag + "the forward result is still the network's");
      }
      cutensornetDestroyContractionOptimizerInfo(info);
      cutensornetDestroyContractionOptimizerConfig(cfg);
      cutensornetDestroyNetwork(net);
      for (int q = 0; q < 3; ++q) { cudaFree(d[q]); cudaFree(dg[q]); }
      cudaFree(dout);
      cudaFree(dadj);
    }
  }
}

static void gradient_errors() {
  const cudaDataType_t t = CUDA_R_64F;
  HT A = mk({'i', 'j'}, false), B = mk({'j', 'k'}, false);
  for (int variant = 0; variant < 4; ++variant) {
    cutensornetNetworkDescriptor_t net;
    cutensornetCreateNetwork(h, &net);
    int64_t id0, id1;
    int64_t e0[2] = {EXT['i'], EXT['j']}, e1[2] = {EXT['j'], EXT['k']};
    int32_t m0[2] = {'i', 'j'}, m1[2] = {'j', 'k'};
    cutensornetNetworkAppendTensor(h, net, 2, e0, m0, nullptr, t, &id0);
    cutensornetNetworkAppendTensor(h, net, 2, e1, m1, nullptr, t, &id1);
    int32_t mo[2] = {'i', 'k'};
    cutensornetNetworkSetOutputTensor(h, net, 2, mo, t);
    void* d0 = dev(A.v.size() * 8);
    void* d1 = dev(B.v.size() * 8);
    put(d0, t, A.v);
    put(d1, t, B.v);
    void* dout = dev((size_t)(EXT['i'] * EXT['k']) * 8);
    cutensornetNetworkSetInputTensorMemory(h, net, id0, d0, nullptr);
    cutensornetNetworkSetInputTensorMemory(h, net, id1, d1, nullptr);
    cutensornetNetworkSetOutputTensorMemory(h, net, dout, nullptr);
    void* g0 = dev((size_t)(EXT['i'] * EXT['j']) * 8);
    int32_t ids[1] = {(int32_t)id0};
    cutensornetTensorIDList_t lst{1, ids};
    if (variant == 0) {
      // Before any tensor is asked for: gradient memory for one that does not require it, ids that do not exist.
      IS(cutensornetNetworkSetGradientTensorMemory(h, net, id0, g0, nullptr), 7);
      int32_t bad[1] = {7};
      cutensornetTensorIDList_t lb{1, bad};
      IS(cutensornetNetworkSetAttribute(h, net, CUTENSORNET_NETWORK_INPUT_TENSORS_REQUIRE_GRAD, &lb, sizeof lb), 7);
      cutensornetTensorIDList_t all{-1, nullptr};
      IS(cutensornetNetworkSetAttribute(h, net, CUTENSORNET_NETWORK_INPUT_TENSORS_REQUIRE_GRAD, &all, sizeof all), 7);
    }
    cutensornetNetworkSetAttribute(h, net, CUTENSORNET_NETWORK_INPUT_TENSORS_REQUIRE_GRAD, &lst, sizeof lst);
    void* dadj = dev((size_t)(EXT['i'] * EXT['k']) * 8);
    put(dadj, t, std::vector<cd>((size_t)(EXT['i'] * EXT['k']), cd(1, 0)));
    if (variant != 2) IS(cutensornetNetworkSetAdjointTensorMemory(h, net, dadj, nullptr), 0);
    if (variant != 3) IS(cutensornetNetworkSetGradientTensorMemory(h, net, id0, g0, nullptr), 0);
    cutensornetContractionOptimizerConfig_t cfg;
    cutensornetCreateContractionOptimizerConfig(h, &cfg);
    cutensornetContractionOptimizerInfo_t info;
    cutensornetCreateContractionOptimizerInfo(h, net, &info);
    cutensornetContractionOptimize(h, net, cfg, 1ull << 30, info);
    Work w;
    cutensornetWorkspaceComputeContractionSizes(h, net, info, w.w);
    if (variant == 0) IS(cutensornetNetworkPrepareGradientsBackward(h, net, w.w), 7);  // the contraction is not prepared
    w.provide(false, variant != 1);
    cutensornetNetworkPrepareContraction(h, net, w.w);
    IS(cutensornetNetworkPrepareGradientsBackward(h, net, w.w), 0);
    cutensornetNetworkContract(h, net, 0, w.w, nullptr, 0);
    const char* what[] = {"everything set", "no cache memory", "no adjoint", "no memory for the gradient"};
    const cutensornetStatus_t want = variant == 1 ? CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE : variant == 2 ? CUTENSORNET_STATUS_INVALID_VALUE : CUTENSORNET_STATUS_SUCCESS;
    check(cutensornetNetworkComputeGradientsBackward(h, net, 0, w.w, nullptr, 0) == want,
          std::string("computing gradients with ") + what[variant] + (want ? " is refused" : " succeeds"));
    if (variant == 1) IS(cutensornetNetworkComputeGradientsBackward(h, net, 0, nullptr, nullptr, 0), CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE);
    IS(cutensornetNetworkComputeGradientsBackward(h, nullptr, 0, w.w, nullptr, 0), 7);
    cutensornetDestroyContractionOptimizerInfo(info);
    cutensornetDestroyContractionOptimizerConfig(cfg);
    cutensornetDestroyNetwork(net);
    cudaFree(d0); cudaFree(d1); cudaFree(dout); cudaFree(g0); cudaFree(dadj);
  }
}

// ---- workspaces without enough memory ----

static void workspaces() {
  const cudaDataType_t t = CUDA_R_64F;
  const std::vector<cd> A = random_matrix(false);
  void* dA = dev((size_t)(M * N) * 8);
  put(dA, t, A);
  Desc in({'i', 'j', 'k', 'l'}, {EI, EJ, EK, EL}, t);
  Desc Q({'i', 'j', 'x'}, {EI, EJ, FULL}, t);
  Desc R({'x', 'k', 'l'}, {FULL, EK, EL}, t);
  void* dQ = dev((size_t)(M * FULL) * 8);
  void* dR = dev((size_t)(FULL * N) * 8);
  void* dS = dev((size_t)FULL * 8);
  cutensornetTensorSVDConfig_t cfg;
  cutensornetCreateTensorSVDConfig(h, &cfg);
  for (int variant = 0; variant < 3; ++variant) {
    Work w;
    cutensornetWorkspaceComputeQRSizes(h, in.d, Q.d, R.d, w.w);
    cutensornetWorkspaceComputeSVDSizes(h, in.d, Q.d, R.d, cfg, w.w);
    void* mem = dev(16);
    if (variant == 1) cutensornetWorkspaceSetMemory(h, w.w, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, mem, 16);
    cutensornetWorkspaceDescriptor_t use = variant == 2 ? nullptr : w.w;
    const char* what[] = {"a workspace with no memory", "a workspace of 16 bytes", "no workspace"};
    check(cutensornetTensorQR(h, in.d, dA, Q.d, dQ, R.d, dR, use, 0) == CUTENSORNET_STATUS_NO_DEVICE_ALLOCATOR,
          std::string("QR with ") + what[variant] + " is NO_DEVICE_ALLOCATOR");
    check(cutensornetTensorSVD(h, in.d, dA, Q.d, dQ, dS, R.d, dR, cfg, nullptr, use, 0) == CUTENSORNET_STATUS_NO_DEVICE_ALLOCATOR,
          std::string("SVD with ") + what[variant] + " is NO_DEVICE_ALLOCATOR");
    cudaFree(mem);
  }
  cutensornetDestroyTensorSVDConfig(cfg);
  cudaFree(dA); cudaFree(dQ); cudaFree(dR); cudaFree(dS);

  // Contracting: preparing needs the scratch the sizes asked for.
  for (int variant = 0; variant < 4; ++variant) {
    HT a = mk({'i', 'j'}, false), b = mk({'j', 'k'}, false), c = mk({'k', 'l'}, false);
    cutensornetNetworkDescriptor_t net;
    cutensornetCreateNetwork(h, &net);
    const HT* ts[3] = {&a, &b, &c};
    int64_t id[3];
    void* d[3];
    for (int q = 0; q < 3; ++q) {
      std::vector<int64_t> ex;
      for (int32_t m : ts[q]->modes) ex.push_back(EXT[m]);
      cutensornetNetworkAppendTensor(h, net, 2, ex.data(), ts[q]->modes.data(), nullptr, t, &id[q]);
      d[q] = dev(ts[q]->v.size() * 8);
      put(d[q], t, ts[q]->v);
      cutensornetNetworkSetInputTensorMemory(h, net, id[q], d[q], nullptr);
    }
    int32_t mo[2] = {'i', 'l'};
    cutensornetNetworkSetOutputTensor(h, net, 2, mo, t);
    void* dout = dev((size_t)(EXT['i'] * EXT['l']) * 8);
    cutensornetNetworkSetOutputTensorMemory(h, net, dout, nullptr);
    cutensornetContractionOptimizerConfig_t ocfg;
    cutensornetCreateContractionOptimizerConfig(h, &ocfg);
    cutensornetContractionOptimizerInfo_t oinfo;
    cutensornetCreateContractionOptimizerInfo(h, net, &oinfo);
    cutensornetContractionOptimize(h, net, ocfg, 1ull << 30, oinfo);
    Work w;
    cutensornetWorkspaceComputeContractionSizes(h, net, oinfo, w.w);
    void* mem = dev(16);
    if (variant == 1) cutensornetWorkspaceSetMemory(h, w.w, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, mem, 16);
    if (variant == 3) w.provide(false);
    cutensornetWorkspaceDescriptor_t use = variant == 2 ? nullptr : w.w;
    const char* what[] = {"no memory", "16 bytes", "no workspace", "the sizes' memory"};
    const cutensornetStatus_t prep = cutensornetNetworkPrepareContraction(h, net, use);
    check(prep == (variant == 3 ? CUTENSORNET_STATUS_SUCCESS : CUTENSORNET_STATUS_INTERNAL_ERROR),
          std::string("preparing a contraction with ") + what[variant] + (variant == 3 ? " succeeds" : " is INTERNAL_ERROR"));
    if (variant != 3) check(cutensornetNetworkContract(h, net, 0, use, nullptr, 0) == CUTENSORNET_STATUS_INVALID_VALUE, "and contracting what was not prepared is INVALID_VALUE");
    cutensornetDestroyContractionOptimizerInfo(oinfo);
    cutensornetDestroyContractionOptimizerConfig(ocfg);
    cutensornetDestroyNetwork(net);
    for (int q = 0; q < 3; ++q) cudaFree(d[q]);
    cudaFree(dout); cudaFree(mem);
  }
}

int main() {
  if (cutensornetCreate(&h)) {
    std::printf("FAIL cutensornetCreate\n");
    return 1;
  }
  svd_algorithms();
  svd_parameters();
  half_precision();
  capture();
  gradients();
  gradient_errors();
  workspaces();
  cutensornetDestroy(h);
  std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
