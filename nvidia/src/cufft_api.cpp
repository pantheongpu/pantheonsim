// libvgpucufft -- VirtualGPU's cuFFT, presented as libcufft.so.12.
//
// Library math runs on the host, as it does for cuBLAS and cuDNN. The transform
// is a generic mixed-radix Cooley-Tukey: it splits on the smallest prime factor
// at each level, so every size works, and only a large prime factor degrades to
// the O(n*p) direct sum.
//
// Everything is computed in double and rounded on the way out, so the
// single-precision results are, if anything, slightly more accurate than
// hardware's. They agree to single-precision rounding, not bit-for-bit.
//
// Layouts: the packed layouts that cufftPlan1d/2d/3d produce are supported for
// every rank, and the advanced (inembed/istride/idist) layout is supported for
// 1-D batched plans. A multi-dimensional advanced layout returns
// CUFFT_NOT_SUPPORTED rather than quietly reading the wrong elements.
#include <cufft.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace {

using cd = std::complex<double>;

bool quiet() { const char* q = std::getenv("VGPU_QUIET"); return q && q[0] == '1'; }

struct Plan {
  cufftType type = CUFFT_C2C;
  std::vector<int> n;        // logical transform size, slowest dimension first
  int batch = 1;
  bool advanced = false;     // user supplied inembed/onembed
  int istride = 1, idist = 0, ostride = 1, odist = 0;
  cudaStream_t stream = nullptr;
};

std::mutex g_mu;
std::unordered_map<int, Plan> g_plans;
int g_next = 1;

Plan* find(cufftHandle h) {
  std::lock_guard<std::mutex> l(g_mu);
  auto it = g_plans.find(h);
  return it == g_plans.end() ? nullptr : &it->second;
}

bool is_c2c(cufftType t) { return t == CUFFT_C2C || t == CUFFT_Z2Z; }
bool is_r2c(cufftType t) { return t == CUFFT_R2C || t == CUFFT_D2Z; }
bool is_c2r(cufftType t) { return t == CUFFT_C2R || t == CUFFT_Z2D; }

size_t logical_elems(const std::vector<int>& n) {
  size_t p = 1;
  for (int d : n) p *= (size_t)d;
  return p;
}

// R2C and C2R store only the non-redundant half of the fastest dimension.
size_t complex_elems(const std::vector<int>& n) {
  if (n.empty()) return 0;
  size_t p = 1;
  for (size_t i = 0; i + 1 < n.size(); ++i) p *= (size_t)n[i];
  return p * (size_t)(n.back() / 2 + 1);
}

int smallest_factor(int n) {
  for (int p = 2; (long long)p * p <= n; ++p)
    if (n % p == 0) return p;
  return n;  // prime
}

// Generic Cooley-Tukey. `in` is read with `stride`, `out` is packed. The
// scratch buffer is sized once by the caller and never grown here: a resize
// mid-recursion would invalidate the `tmp` pointers held by every level above.
void fft_rec(const cd* in, cd* out, int n, int stride, int sign, cd* scratch) {
  if (n == 1) { out[0] = in[0]; return; }
  const int p = smallest_factor(n);
  const int m = n / p;
  cd* tmp = scratch;
  // Children run one after another and hand their results up into `tmp` before
  // the next starts, so they can all share the region past this level's.
  for (int j = 0; j < p; ++j)
    fft_rec(in + (size_t)j * stride, tmp + (size_t)j * m, m, stride * p, sign, scratch + n);
  const double base = sign * 2.0 * M_PI / n;
  for (int q = 0; q < p; ++q)
    for (int k = 0; k < m; ++k) {
      cd acc = tmp[k];  // j = 0 has a unit twiddle
      for (int j = 1; j < p; ++j) {
        const double ang = base * (double)j * (double)(k + q * m);
        acc += tmp[(size_t)j * m + k] * cd(std::cos(ang), std::sin(ang));
      }
      out[(size_t)q * m + k] = acc;
    }
}

void fft_1d(cd* data, int n, int sign) {
  if (n <= 1) return;
  // Each recursion level needs n slots and every level at least halves n, so
  // 2n bounds the whole tree.
  std::vector<cd> out((size_t)n), scratch((size_t)n * 2 + 8);
  fft_rec(data, out.data(), n, 1, sign, scratch.data());
  std::memcpy(data, out.data(), sizeof(cd) * (size_t)n);
}

// Multi-dimensional transform of a packed array: 1-D transforms along each axis.
void fft_nd(std::vector<cd>& a, const std::vector<int>& n, int sign) {
  const size_t total = a.size();
  size_t inner = 1;
  for (size_t d = n.size(); d-- > 0;) {
    const int len = n[d];
    const size_t outer = total / (inner * (size_t)len);
    std::vector<cd> line((size_t)len);
    for (size_t o = 0; o < outer; ++o)
      for (size_t i = 0; i < inner; ++i) {
        const size_t base = o * inner * (size_t)len + i;
        for (int k = 0; k < len; ++k) line[k] = a[base + (size_t)k * inner];
        fft_1d(line.data(), len, sign);
        for (int k = 0; k < len; ++k) a[base + (size_t)k * inner] = line[k];
      }
    inner *= (size_t)len;
  }
}

// Rebuild the full spectrum of one real-to-complex line from its stored half,
// using Hermitian symmetry, so the C2R inverse can run as a plain complex FFT.
void expand_hermitian(const cd* half, cd* full, int n) {
  const int h = n / 2 + 1;
  for (int k = 0; k < h; ++k) full[k] = half[k];
  for (int k = h; k < n; ++k) full[k] = std::conj(half[n - k]);
}

template <class T> void to_host(const void* dev, std::vector<T>& out, size_t n) {
  out.resize(n);
  if (n) cudaMemcpy(out.data(), dev, n * sizeof(T), cudaMemcpyDeviceToHost);
}
template <class T> void to_dev(void* dev, const std::vector<T>& in) {
  if (!in.empty()) cudaMemcpy(dev, in.data(), in.size() * sizeof(T), cudaMemcpyHostToDevice);
}

cufftResult make_plan(cufftHandle h, int rank, const int* n, const int* inembed, int istride,
                      int idist, const int* onembed, int ostride, int odist, cufftType type,
                      int batch) {
  if (rank < 1 || rank > 3 || !n || batch < 1) return CUFFT_INVALID_VALUE;
  Plan p;
  p.type = type;
  p.n.assign(n, n + rank);
  for (int d : p.n) if (d < 1) return CUFFT_INVALID_SIZE;
  p.batch = batch;
  p.advanced = inembed != nullptr || onembed != nullptr;
  if (p.advanced) {
    if (rank != 1) {
      std::fprintf(stderr,
                   "[vgpu] cufft: advanced data layouts are implemented for 1-D plans only\n");
      return CUFFT_NOT_SUPPORTED;
    }
    // An embed equal to the transform size is just the packed layout described
    // the long way; anything else would need real strided gathering.
    if ((inembed && inembed[0] != n[0]) || (onembed && onembed[0] != n[0] &&
                                            onembed[0] != n[0] / 2 + 1)) {
      std::fprintf(stderr, "[vgpu] cufft: padded embeds are not implemented\n");
      return CUFFT_NOT_SUPPORTED;
    }
    p.istride = istride < 1 ? 1 : istride;
    p.ostride = ostride < 1 ? 1 : ostride;
    p.idist = idist;
    p.odist = odist;
  }
  std::lock_guard<std::mutex> l(g_mu);
  g_plans[h] = p;
  return CUFFT_SUCCESS;
}

cufftHandle alloc_handle() {
  std::lock_guard<std::mutex> l(g_mu);
  return g_next++;
}

// One batch element's input/output offsets, honoring the advanced layout.
size_t in_offset(const Plan& p, int b, size_t packed) {
  const size_t dist = p.advanced && p.idist ? (size_t)p.idist : packed;
  return (size_t)b * dist;
}
size_t out_offset(const Plan& p, int b, size_t packed) {
  const size_t dist = p.advanced && p.odist ? (size_t)p.odist : packed;
  return (size_t)b * dist;
}

// The single implementation behind all six Exec entry points. Reads the whole
// input, transforms batch by batch, writes the whole output.
template <class Real>
cufftResult exec(cufftHandle handle, const void* idata, void* odata, int direction) {
  Plan* p = find(handle);
  if (!p) return CUFFT_INVALID_PLAN;
  if (!idata || !odata) return CUFFT_INVALID_VALUE;
  cudaStreamSynchronize(p->stream);

  const size_t nlog = logical_elems(p->n);
  const size_t nhalf = complex_elems(p->n);
  const int sign = is_c2c(p->type) ? direction : (is_r2c(p->type) ? CUFFT_FORWARD : CUFFT_INVERSE);

  const size_t in_pack = is_c2r(p->type) ? nhalf : nlog;
  const size_t out_pack = is_r2c(p->type) ? nhalf : nlog;
  // The exact number of elements the user's buffer must hold: the last batch's
  // base plus its last strided element. Rounding this up would read past the
  // end of the allocation, which VirtualGPU catches and hardware does not.
  const size_t istride = p->advanced ? (size_t)p->istride : 1;
  const size_t ostride = p->advanced ? (size_t)p->ostride : 1;
  const size_t in_span = in_offset(*p, p->batch - 1, in_pack) + (in_pack - 1) * istride + 1;
  const size_t out_span = out_offset(*p, p->batch - 1, out_pack) + (out_pack - 1) * ostride + 1;

  std::vector<Real> hin_r, hout_r;
  std::vector<std::complex<Real>> hin_c, hout_c;
  if (is_c2r(p->type) || is_c2c(p->type)) to_host(idata, hin_c, in_span);
  else to_host(idata, hin_r, in_span);
  if (is_r2c(p->type) || is_c2c(p->type)) hout_c.assign(out_span, std::complex<Real>(0, 0));
  else hout_r.assign(out_span, Real(0));

  std::vector<cd> work(nlog);
  for (int b = 0; b < p->batch; ++b) {
    const size_t io = in_offset(*p, b, in_pack);
    const size_t oo = out_offset(*p, b, out_pack);
    const size_t is = istride, os = ostride;

    if (is_c2r(p->type)) {
      // Expand the stored half-spectrum of the fastest dimension, then invert.
      const int last = p->n.back();
      const size_t lines = nhalf / (size_t)(last / 2 + 1);
      std::vector<cd> half((size_t)(last / 2 + 1));
      for (size_t l = 0; l < lines; ++l) {
        for (int k = 0; k <= last / 2; ++k) {
          const auto v = hin_c[io + (l * (size_t)(last / 2 + 1) + (size_t)k) * is];
          half[k] = cd((double)v.real(), (double)v.imag());
        }
        expand_hermitian(half.data(), work.data() + l * (size_t)last, last);
      }
      fft_nd(work, p->n, CUFFT_INVERSE);
      for (size_t k = 0; k < nlog; ++k) hout_r[oo + k * os] = (Real)work[k].real();
    } else if (is_r2c(p->type)) {
      for (size_t k = 0; k < nlog; ++k) work[k] = cd((double)hin_r[io + k * is], 0.0);
      fft_nd(work, p->n, CUFFT_FORWARD);
      // Keep the non-redundant half of the fastest dimension.
      const int last = p->n.back();
      const size_t lines = nlog / (size_t)last;
      for (size_t l = 0; l < lines; ++l)
        for (int k = 0; k <= last / 2; ++k) {
          const cd v = work[l * (size_t)last + (size_t)k];
          hout_c[oo + (l * (size_t)(last / 2 + 1) + (size_t)k) * os] =
              std::complex<Real>((Real)v.real(), (Real)v.imag());
        }
    } else {
      for (size_t k = 0; k < nlog; ++k) {
        const auto v = hin_c[io + k * is];
        work[k] = cd((double)v.real(), (double)v.imag());
      }
      fft_nd(work, p->n, sign);
      for (size_t k = 0; k < nlog; ++k)
        hout_c[oo + k * os] = std::complex<Real>((Real)work[k].real(), (Real)work[k].imag());
    }
  }

  if (is_r2c(p->type) || is_c2c(p->type)) to_dev(odata, hout_c);
  else to_dev(odata, hout_r);
  return CUFFT_SUCCESS;
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

/* ---- plans ---- */

VGPU_EXPORT cufftResult cufftCreate(cufftHandle* handle) {
  if (!handle) return CUFFT_INVALID_VALUE;
  *handle = alloc_handle();
  std::lock_guard<std::mutex> l(g_mu);
  g_plans[*handle] = Plan();
  if (!quiet())
    std::fprintf(stderr, "[vgpu] cuFFT plan created (host-computed mixed-radix transform)\n");
  return CUFFT_SUCCESS;
}

VGPU_EXPORT cufftResult cufftDestroy(cufftHandle handle) {
  std::lock_guard<std::mutex> l(g_mu);
  return g_plans.erase(handle) ? CUFFT_SUCCESS : CUFFT_INVALID_PLAN;
}

VGPU_EXPORT cufftResult cufftPlan1d(cufftHandle* plan, int nx, cufftType type, int batch) {
  if (!plan) return CUFFT_INVALID_VALUE;
  *plan = alloc_handle();
  return make_plan(*plan, 1, &nx, nullptr, 1, 0, nullptr, 1, 0, type, batch);
}
VGPU_EXPORT cufftResult cufftPlan2d(cufftHandle* plan, int nx, int ny, cufftType type) {
  if (!plan) return CUFFT_INVALID_VALUE;
  const int n[2] = {nx, ny};
  *plan = alloc_handle();
  return make_plan(*plan, 2, n, nullptr, 1, 0, nullptr, 1, 0, type, 1);
}
VGPU_EXPORT cufftResult cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, cufftType type) {
  if (!plan) return CUFFT_INVALID_VALUE;
  const int n[3] = {nx, ny, nz};
  *plan = alloc_handle();
  return make_plan(*plan, 3, n, nullptr, 1, 0, nullptr, 1, 0, type, 1);
}
VGPU_EXPORT cufftResult cufftPlanMany(cufftHandle* plan, int rank, int* n, int* inembed,
                                      int istride, int idist, int* onembed, int ostride, int odist,
                                      cufftType type, int batch) {
  if (!plan) return CUFFT_INVALID_VALUE;
  *plan = alloc_handle();
  return make_plan(*plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);
}

VGPU_EXPORT cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch,
                                        size_t* work) {
  if (work) *work = 0;
  return make_plan(plan, 1, &nx, nullptr, 1, 0, nullptr, 1, 0, type, batch);
}
VGPU_EXPORT cufftResult cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type,
                                        size_t* work) {
  if (work) *work = 0;
  const int n[2] = {nx, ny};
  return make_plan(plan, 2, n, nullptr, 1, 0, nullptr, 1, 0, type, 1);
}
VGPU_EXPORT cufftResult cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type,
                                        size_t* work) {
  if (work) *work = 0;
  const int n[3] = {nx, ny, nz};
  return make_plan(plan, 3, n, nullptr, 1, 0, nullptr, 1, 0, type, 1);
}
VGPU_EXPORT cufftResult cufftMakePlanMany(cufftHandle plan, int rank, int* n, int* inembed,
                                          int istride, int idist, int* onembed, int ostride,
                                          int odist, cufftType type, int batch, size_t* work) {
  if (work) *work = 0;
  return make_plan(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);
}
VGPU_EXPORT cufftResult cufftMakePlanMany64(cufftHandle plan, int rank, long long* n,
                                            long long* inembed, long long istride, long long idist,
                                            long long* onembed, long long ostride, long long odist,
                                            cufftType type, long long batch, size_t* work) {
  if (!n || rank < 1 || rank > 3) return CUFFT_INVALID_VALUE;
  int ni[3], ie[3], oe[3];
  for (int i = 0; i < rank; ++i) {
    ni[i] = (int)n[i];
    if (inembed) ie[i] = (int)inembed[i];
    if (onembed) oe[i] = (int)onembed[i];
  }
  if (work) *work = 0;
  return make_plan(plan, rank, ni, inembed ? ie : nullptr, (int)istride, (int)idist,
                   onembed ? oe : nullptr, (int)ostride, (int)odist, type, (int)batch);
}

/* ---- work area: everything is computed on the host, so there is none ---- */

VGPU_EXPORT cufftResult cufftGetSize(cufftHandle handle, size_t* work) {
  if (!find(handle)) return CUFFT_INVALID_PLAN;
  if (work) *work = 0;
  return CUFFT_SUCCESS;
}
VGPU_EXPORT cufftResult cufftGetSize1d(cufftHandle, int, cufftType, int, size_t* w) {
  if (w) *w = 0; return CUFFT_SUCCESS;
}
VGPU_EXPORT cufftResult cufftGetSize2d(cufftHandle, int, int, cufftType, size_t* w) {
  if (w) *w = 0; return CUFFT_SUCCESS;
}
VGPU_EXPORT cufftResult cufftGetSize3d(cufftHandle, int, int, int, cufftType, size_t* w) {
  if (w) *w = 0; return CUFFT_SUCCESS;
}
VGPU_EXPORT cufftResult cufftGetSizeMany(cufftHandle, int, int*, int*, int, int, int*, int, int,
                                         cufftType, int, size_t* w) {
  if (w) *w = 0; return CUFFT_SUCCESS;
}
VGPU_EXPORT cufftResult cufftGetSizeMany64(cufftHandle, int, long long*, long long*, long long,
                                           long long, long long*, long long, long long, cufftType,
                                           long long, size_t* w) {
  if (w) *w = 0; return CUFFT_SUCCESS;
}
VGPU_EXPORT cufftResult cufftEstimate1d(int, cufftType, int, size_t* w) { if (w) *w = 0; return CUFFT_SUCCESS; }
VGPU_EXPORT cufftResult cufftEstimate2d(int, int, cufftType, size_t* w) { if (w) *w = 0; return CUFFT_SUCCESS; }
VGPU_EXPORT cufftResult cufftEstimate3d(int, int, int, cufftType, size_t* w) { if (w) *w = 0; return CUFFT_SUCCESS; }
VGPU_EXPORT cufftResult cufftEstimateMany(int, int*, int*, int, int, int*, int, int, cufftType,
                                          int, size_t* w) { if (w) *w = 0; return CUFFT_SUCCESS; }
VGPU_EXPORT cufftResult cufftSetWorkArea(cufftHandle handle, void*) {
  return find(handle) ? CUFFT_SUCCESS : CUFFT_INVALID_PLAN;
}
VGPU_EXPORT cufftResult cufftSetAutoAllocation(cufftHandle handle, int) {
  return find(handle) ? CUFFT_SUCCESS : CUFFT_INVALID_PLAN;
}

VGPU_EXPORT cufftResult cufftSetStream(cufftHandle handle, cudaStream_t stream) {
  Plan* p = find(handle);
  if (!p) return CUFFT_INVALID_PLAN;
  p->stream = stream;
  return CUFFT_SUCCESS;
}

VGPU_EXPORT cufftResult cufftGetVersion(int* version) {
  if (!version) return CUFFT_INVALID_VALUE;
  *version = CUFFT_VERSION;
  return CUFFT_SUCCESS;
}
VGPU_EXPORT cufftResult cufftGetProperty(libraryPropertyType type, int* value) {
  if (!value) return CUFFT_INVALID_VALUE;
  switch (type) {
    case MAJOR_VERSION: *value = CUFFT_VER_MAJOR; break;
    case MINOR_VERSION: *value = CUFFT_VER_MINOR; break;
    case PATCH_LEVEL: *value = CUFFT_VER_PATCH; break;
    default: return CUFFT_INVALID_VALUE;
  }
  return CUFFT_SUCCESS;
}

/* ---- execution ---- */

VGPU_EXPORT cufftResult cufftExecC2C(cufftHandle h, cufftComplex* in, cufftComplex* out, int dir) {
  Plan* p = find(h);
  if (!p) return CUFFT_INVALID_PLAN;
  if (p->type != CUFFT_C2C) return CUFFT_INVALID_TYPE;
  return exec<float>(h, in, out, dir);
}
VGPU_EXPORT cufftResult cufftExecR2C(cufftHandle h, cufftReal* in, cufftComplex* out) {
  Plan* p = find(h);
  if (!p) return CUFFT_INVALID_PLAN;
  if (p->type != CUFFT_R2C) return CUFFT_INVALID_TYPE;
  return exec<float>(h, in, out, CUFFT_FORWARD);
}
VGPU_EXPORT cufftResult cufftExecC2R(cufftHandle h, cufftComplex* in, cufftReal* out) {
  Plan* p = find(h);
  if (!p) return CUFFT_INVALID_PLAN;
  if (p->type != CUFFT_C2R) return CUFFT_INVALID_TYPE;
  return exec<float>(h, in, out, CUFFT_INVERSE);
}
VGPU_EXPORT cufftResult cufftExecZ2Z(cufftHandle h, cufftDoubleComplex* in,
                                     cufftDoubleComplex* out, int dir) {
  Plan* p = find(h);
  if (!p) return CUFFT_INVALID_PLAN;
  if (p->type != CUFFT_Z2Z) return CUFFT_INVALID_TYPE;
  return exec<double>(h, in, out, dir);
}
VGPU_EXPORT cufftResult cufftExecD2Z(cufftHandle h, cufftDoubleReal* in,
                                     cufftDoubleComplex* out) {
  Plan* p = find(h);
  if (!p) return CUFFT_INVALID_PLAN;
  if (p->type != CUFFT_D2Z) return CUFFT_INVALID_TYPE;
  return exec<double>(h, in, out, CUFFT_FORWARD);
}
VGPU_EXPORT cufftResult cufftExecZ2D(cufftHandle h, cufftDoubleComplex* in,
                                     cufftDoubleReal* out) {
  Plan* p = find(h);
  if (!p) return CUFFT_INVALID_PLAN;
  if (p->type != CUFFT_Z2D) return CUFFT_INVALID_TYPE;
  return exec<double>(h, in, out, CUFFT_INVERSE);
}
