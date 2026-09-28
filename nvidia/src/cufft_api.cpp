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
// Layouts: the packed layouts of cufftPlan1d/2d/3d and the advanced
// (inembed/istride/idist) layout of cufftPlanMany and cufftXtMakePlanMany, for
// every rank, padded embeds included. Elements a strided output layout skips
// keep what they held.
//
// Precision: single and double through the classic API; cufftXt adds half,
// computed in double like the rest and rounded to half on the way out.
#include <cufft.h>
#include <cufftXt.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace {

using cd = std::complex<double>;

bool quiet() { const char* q = std::getenv("VGPU_QUIET"); return q && q[0] == '1'; }

struct Plan {
  cufftType type = CUFFT_C2C;  // the transform's kind; the precision is `prec`
  int prec = 32;             // element width in bits: 16 (cufftXt only), 32 or 64
  std::vector<int> n;        // logical transform size, slowest dimension first
  long long batch = 1;
  // The advanced layout, as cufftPlanMany describes it: element (i0, ..., ir-1)
  // of batch b sits at b*dist + (sum of i_d * pitch_d) * stride, where each
  // pitch is the product of the embeds of the dimensions faster than it. The
  // basic (packed) layout is the same thing with embed = shape and stride 1.
  std::vector<long long> iembed, oembed;  // empty for the packed layout
  long long istride = 1, idist = 0, ostride = 1, odist = 0;
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

// Multi-dimensional transform of a packed array: 1-D transforms along each
// axis. With `skip_last` the fastest axis is left alone, which is what the
// complex-to-real inverse needs before its last, Hermitian, step.
void fft_nd(std::vector<cd>& a, const std::vector<int>& n, int sign, bool skip_last = false) {
  const size_t total = a.size();
  size_t inner = 1;
  for (size_t d = n.size(); d-- > 0;) {
    const int len = n[d];
    if (skip_last && d + 1 == n.size()) { inner *= (size_t)len; continue; }
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
// This holds line by line only once every slower axis has been inverted: before
// that, the mirror of X[i, k] is X[-i, n-k], in another line.
void expand_hermitian(const cd* half, cd* full, int n) {
  const int h = n / 2 + 1;
  for (int k = 0; k < h; ++k) full[k] = half[k];
  for (int k = h; k < n; ++k) full[k] = std::conj(half[n - k]);
}

// IEEE binary16, for cufftXt's half-precision plans. Converted by hand so the
// library needs no CUDA half header on the host side.
double half_to_double(uint16_t h) {
  const int sign = h >> 15, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
  double v;
  if (exp == 0) v = std::ldexp((double)man, -24);
  else if (exp == 31) v = man ? NAN : INFINITY;
  else v = std::ldexp((double)(man | 0x400), exp - 25);
  return sign ? -v : v;
}
uint16_t double_to_half(double d) {
  float f = (float)d;
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000;
  const int exp = (int)((x >> 23) & 0xff) - 127 + 15;
  uint32_t man = x & 0x7fffff;
  if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00 | (man ? 0x200 : 0));
  if (exp >= 31) return (uint16_t)(sign | 0x7c00);
  if (exp <= 0) {  // subnormal or zero, round to nearest even
    if (exp < -10) return (uint16_t)sign;
    man |= 0x800000;
    const int shift = 14 - exp;
    uint32_t h = man >> shift;
    const uint32_t rem = man & ((1u << shift) - 1), halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (h & 1))) ++h;
    return (uint16_t)(sign | h);
  }
  uint32_t h = ((uint32_t)exp << 10) | (man >> 13);
  const uint32_t rem = man & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) ++h;  // may carry into the exponent
  return (uint16_t)(sign | h);
}

// One element of a device buffer, read or written as the plan's type.
struct Elems {
  int prec;       // 16, 32 or 64
  bool complex;
  size_t size() const { return (size_t)prec / 8 * (complex ? 2 : 1); }
  double comp(const char* p, int c) const {
    p += (size_t)c * (prec / 8);
    if (prec == 64) { double v; std::memcpy(&v, p, 8); return v; }
    if (prec == 32) { float v; std::memcpy(&v, p, 4); return v; }
    uint16_t v; std::memcpy(&v, p, 2); return half_to_double(v);
  }
  void put(char* p, int c, double v) const {
    p += (size_t)c * (prec / 8);
    if (prec == 64) std::memcpy(p, &v, 8);
    else if (prec == 32) { const float f = (float)v; std::memcpy(p, &f, 4); }
    else { const uint16_t h = double_to_half(v); std::memcpy(p, &h, 2); }
  }
  cd load(const std::vector<char>& buf, size_t i) const {
    const char* p = buf.data() + i * size();
    return cd(comp(p, 0), complex ? comp(p, 1) : 0.0);
  }
  void store(std::vector<char>& buf, size_t i, cd v) const {
    char* p = buf.data() + i * size();
    put(p, 0, v.real());
    if (complex) put(p, 1, v.imag());
  }
};

// The element offsets (in elements, from the batch's base) of every point of a
// packed array of `shape`, laid out with `embed` and `stride`.
std::vector<size_t> layout(const std::vector<int>& shape, const std::vector<long long>& embed,
                           long long stride) {
  const size_t r = shape.size();
  std::vector<size_t> pitch(r, 1);
  for (size_t d = r - 1; d-- > 0;)
    pitch[d] = pitch[d + 1] * (size_t)(embed.empty() ? shape[d + 1] : embed[d + 1]);
  std::vector<size_t> off(1, 0);
  for (size_t d = 0; d < r; ++d) {
    std::vector<size_t> next;
    next.reserve(off.size() * (size_t)shape[d]);
    for (size_t o : off)
      for (int i = 0; i < shape[d]; ++i) next.push_back(o + (size_t)i * pitch[d]);
    off.swap(next);
  }
  const size_t s = embed.empty() ? 1 : (size_t)stride;
  for (size_t& o : off) o *= s;
  return off;
}

bool is_double(cufftType t) { return t == CUFFT_Z2Z || t == CUFFT_D2Z || t == CUFFT_Z2D; }

// Every plan entry point lands here. `prec` is 0 to take it from the type; only
// cufftXt plans ask for half precision.
template <class I>
cufftResult make_plan(cufftHandle h, int rank, const I* n, const std::type_identity_t<I>* inembed,
                      long long istride, long long idist,
                      const std::type_identity_t<I>* onembed, long long ostride, long long odist,
                      cufftType type, long long batch, int prec = 0) {
  if (rank < 1 || rank > 3 || !n || batch < 1) return CUFFT_INVALID_VALUE;
  if (!is_c2c(type) && !is_r2c(type) && !is_c2r(type)) return CUFFT_INVALID_TYPE;
  Plan p;
  p.type = type;
  p.prec = prec ? prec : is_double(type) ? 64 : 32;
  for (int d = 0; d < rank; ++d) {
    if (n[d] < 1 || n[d] > INT32_MAX) return CUFFT_INVALID_SIZE;
    p.n.push_back((int)n[d]);
  }
  p.batch = batch;
  // Either embed left NULL means the basic layout for both sides, and the
  // strides and distances are ignored, as cufftPlanMany documents.
  if (inembed && onembed) {
    p.iembed.assign(inembed, inembed + rank);
    p.oembed.assign(onembed, onembed + rank);
    for (int d = 1; d < rank; ++d)
      if (p.iembed[d] < 1 || p.oembed[d] < 1) return CUFFT_INVALID_VALUE;
    p.istride = istride < 1 ? 1 : istride;
    p.ostride = ostride < 1 ? 1 : ostride;
    p.idist = idist;
    p.odist = odist;
  }
  std::lock_guard<std::mutex> l(g_mu);
  auto it = g_plans.find(h);
  if (it != g_plans.end()) p.stream = it->second.stream;  // cufftSetStream may come first
  g_plans[h] = p;
  return CUFFT_SUCCESS;
}

cufftHandle alloc_handle() {
  std::lock_guard<std::mutex> l(g_mu);
  return g_next++;
}

// The single implementation behind every Exec entry point. Reads the whole
// input, transforms batch by batch, writes the whole output.
cufftResult exec(cufftHandle handle, const void* idata, void* odata, int direction) {
  Plan* p = find(handle);
  if (!p) return CUFFT_INVALID_PLAN;
  if (p->n.empty()) return CUFFT_INVALID_PLAN;  // created but never made
  if (!idata || !odata) return CUFFT_INVALID_VALUE;
  cudaStreamSynchronize(p->stream);

  const bool c2r = is_c2r(p->type), r2c = is_r2c(p->type);
  std::vector<int> half = p->n;
  half.back() = half.back() / 2 + 1;
  const std::vector<int>& ishape = c2r ? half : p->n;
  const std::vector<int>& oshape = r2c ? half : p->n;
  const Elems ie{p->prec, !r2c}, oe{p->prec, !c2r};
  const std::vector<size_t> ioff = layout(ishape, p->iembed, p->istride);
  const std::vector<size_t> ooff = layout(oshape, p->oembed, p->ostride);
  const size_t idist = !p->iembed.empty() && p->idist ? (size_t)p->idist : logical_elems(ishape);
  const size_t odist = !p->oembed.empty() && p->odist ? (size_t)p->odist : logical_elems(oshape);
  // The exact number of elements the user's buffer must hold: the last batch's
  // base plus its furthest element. Rounding this up would read past the end
  // of the allocation, which VirtualGPU catches and hardware does not.
  const size_t in_span = (size_t)(p->batch - 1) * idist + ioff.back() + 1;
  const size_t out_span = (size_t)(p->batch - 1) * odist + ooff.back() + 1;

  // The output is read first so the gaps a strided or padded layout leaves
  // keep what they held; an in-place transform reads the input just after.
  std::vector<char> hout(out_span * oe.size()), hin(in_span * ie.size());
  cudaMemcpy(hout.data(), odata, hout.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hin.data(), idata, hin.size(), cudaMemcpyDeviceToHost);

  const int sign = c2r ? CUFFT_INVERSE : r2c ? CUFFT_FORWARD : direction;
  const int last = p->n.back();
  std::vector<cd> work;
  for (long long b = 0; b < p->batch; ++b) {
    const size_t ib = (size_t)b * idist, ob = (size_t)b * odist;
    work.resize(ioff.size());
    for (size_t k = 0; k < ioff.size(); ++k) work[k] = ie.load(hin, ib + ioff[k]);

    if (c2r) {
      // Invert every axis but the fastest on the stored half, then each
      // fastest-axis line is the half spectrum of a real signal.
      fft_nd(work, half, CUFFT_INVERSE, /*skip_last=*/true);
      const size_t lines = work.size() / (size_t)half.back();
      std::vector<cd> full((size_t)lines * (size_t)last);
      for (size_t l = 0; l < lines; ++l) {
        cd* line = full.data() + l * (size_t)last;
        expand_hermitian(work.data() + l * (size_t)half.back(), line, last);
        fft_1d(line, last, CUFFT_INVERSE);
      }
      work.swap(full);
    } else {
      fft_nd(work, p->n, sign);
      if (r2c) {  // keep the non-redundant half of the fastest dimension
        const size_t lines = work.size() / (size_t)last, h = (size_t)half.back();
        for (size_t l = 0; l < lines; ++l)
          for (size_t k = 0; k < h; ++k) work[l * h + k] = work[l * (size_t)last + k];
        work.resize(lines * h);
      }
    }
    for (size_t k = 0; k < ooff.size(); ++k) oe.store(hout, ob + ooff[k], work[k]);
  }

  cudaMemcpy(odata, hout.data(), hout.size(), cudaMemcpyHostToDevice);
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
  if (work) *work = 0;
  return make_plan(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);
}

/* ---- cufftXt: the same plans, typed by cudaDataType; PyTorch plans this way ---- */

namespace {
// The transform a pair of cudaDataTypes describes, or 0 for a pair cuFFT has no
// transform for. Half precision keeps the single-precision kind; the plan's
// `prec` says which.
cufftType xt_type(cudaDataType in, cudaDataType out, int* prec) {
  struct Row { cudaDataType in, out; cufftType type; int prec; };
  static const Row rows[] = {
      {CUDA_C_32F, CUDA_C_32F, CUFFT_C2C, 32}, {CUDA_R_32F, CUDA_C_32F, CUFFT_R2C, 32},
      {CUDA_C_32F, CUDA_R_32F, CUFFT_C2R, 32}, {CUDA_C_64F, CUDA_C_64F, CUFFT_Z2Z, 64},
      {CUDA_R_64F, CUDA_C_64F, CUFFT_D2Z, 64}, {CUDA_C_64F, CUDA_R_64F, CUFFT_Z2D, 64},
      {CUDA_C_16F, CUDA_C_16F, CUFFT_C2C, 16}, {CUDA_R_16F, CUDA_C_16F, CUFFT_R2C, 16},
      {CUDA_C_16F, CUDA_R_16F, CUFFT_C2R, 16},
  };
  for (const Row& r : rows)
    if (r.in == in && r.out == out) { *prec = r.prec; return r.type; }
  return (cufftType)0;
}
}  // namespace

VGPU_EXPORT cufftResult cufftXtMakePlanMany(cufftHandle plan, int rank, long long* n,
                                            long long* inembed, long long istride,
                                            long long idist, cudaDataType inputtype,
                                            long long* onembed, long long ostride,
                                            long long odist, cudaDataType outputtype,
                                            long long batch, size_t* work,
                                            cudaDataType executiontype) {
  if (!find(plan)) return CUFFT_INVALID_PLAN;
  int prec = 0;
  const cufftType type = xt_type(inputtype, outputtype, &prec);
  if (!type) return CUFFT_INVALID_TYPE;
  // The execution type is the complex type of the transform's precision.
  const cudaDataType exec_type = prec == 64 ? CUDA_C_64F : prec == 32 ? CUDA_C_32F : CUDA_C_16F;
  if (executiontype != exec_type) return CUFFT_INVALID_TYPE;
  if (work) *work = 0;
  return make_plan(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch,
                   prec);
}

VGPU_EXPORT cufftResult cufftXtGetSizeMany(cufftHandle plan, int, long long*, long long*,
                                           long long, long long, cudaDataType, long long*,
                                           long long, long long, cudaDataType, long long,
                                           size_t* work, cudaDataType) {
  if (!find(plan)) return CUFFT_INVALID_PLAN;
  if (work) *work = 0;
  return CUFFT_SUCCESS;
}

// The direction is ignored for real-to-complex and complex-to-real plans, as on
// hardware: their kind fixes it.
VGPU_EXPORT cufftResult cufftXtExec(cufftHandle plan, void* input, void* output, int direction) {
  Plan* p = find(plan);
  if (!p) return CUFFT_INVALID_PLAN;
  if (is_c2c(p->type) && direction != CUFFT_FORWARD && direction != CUFFT_INVERSE)
    return CUFFT_INVALID_VALUE;
  return exec(plan, input, output, direction);
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
  return exec(h, in, out, dir);
}
VGPU_EXPORT cufftResult cufftExecR2C(cufftHandle h, cufftReal* in, cufftComplex* out) {
  Plan* p = find(h);
  if (!p) return CUFFT_INVALID_PLAN;
  if (p->type != CUFFT_R2C) return CUFFT_INVALID_TYPE;
  return exec(h, in, out, CUFFT_FORWARD);
}
VGPU_EXPORT cufftResult cufftExecC2R(cufftHandle h, cufftComplex* in, cufftReal* out) {
  Plan* p = find(h);
  if (!p) return CUFFT_INVALID_PLAN;
  if (p->type != CUFFT_C2R) return CUFFT_INVALID_TYPE;
  return exec(h, in, out, CUFFT_INVERSE);
}
VGPU_EXPORT cufftResult cufftExecZ2Z(cufftHandle h, cufftDoubleComplex* in,
                                     cufftDoubleComplex* out, int dir) {
  Plan* p = find(h);
  if (!p) return CUFFT_INVALID_PLAN;
  if (p->type != CUFFT_Z2Z) return CUFFT_INVALID_TYPE;
  return exec(h, in, out, dir);
}
VGPU_EXPORT cufftResult cufftExecD2Z(cufftHandle h, cufftDoubleReal* in,
                                     cufftDoubleComplex* out) {
  Plan* p = find(h);
  if (!p) return CUFFT_INVALID_PLAN;
  if (p->type != CUFFT_D2Z) return CUFFT_INVALID_TYPE;
  return exec(h, in, out, CUFFT_FORWARD);
}
VGPU_EXPORT cufftResult cufftExecZ2D(cufftHandle h, cufftDoubleComplex* in,
                                     cufftDoubleReal* out) {
  Plan* p = find(h);
  if (!p) return CUFFT_INVALID_PLAN;
  if (p->type != CUFFT_Z2D) return CUFFT_INVALID_TYPE;
  return exec(h, in, out, CUFFT_INVERSE);
}
