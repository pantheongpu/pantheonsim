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

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
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
  // cufftXtSetGPUs: the devices a multi-GPU plan spreads its data over, in
  // the order the data is dealt to them. Empty for a single-GPU plan.
  std::vector<int> gpus;
  int spread = 0;            // how a multi-GPU plan lays its data out (Spread)
  // cufftXtSetJITCallback: the LTO callbacks, kept until the plan is made.
  struct JitCallback {
    std::string name;
    std::vector<char> image;
    void* info = nullptr;
  };
  std::map<int, JitCallback> jit;  // by cufftXtCallbackType
  size_t jit_shared[8] = {};       // cufftXtSetCallbackSharedSize, by type
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

cufftResult multi_check(Plan& p);
void zero_work(cufftHandle h, size_t* work);
cufftResult jit_prepare(Plan& p);

// Every plan entry point lands here. `prec` is 0 to take it from the type; only
// cufftXt plans ask for half precision.
template <class I>
cufftResult make_plan(cufftHandle h, int rank, const I* n, const std::type_identity_t<I>* inembed,
                      long long istride, long long idist,
                      const std::type_identity_t<I>* onembed, long long ostride, long long odist,
                      cufftType type, long long batch, int prec = 0, size_t* work = nullptr,
                      bool make_call = false) {
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
  if (it != g_plans.end()) {  // what was set on the handle before it was made
    p.stream = it->second.stream;
    p.gpus = it->second.gpus;
    p.jit = it->second.jit;
    std::memcpy(p.jit_shared, it->second.jit_shared, sizeof p.jit_shared);
  }
  if (p.gpus.size() > 1) {
    // A multi-GPU plan reports one work size per GPU, so it needs somewhere
    // to put them: NVIDIA's answers a NULL workSize with INVALID_VALUE before
    // it looks at anything else (RTX 3060 pair, CUDA 13.0).
    if (make_call && !work) return CUFFT_INVALID_VALUE;
    if (const cufftResult r = multi_check(p)) return r;
  }
  if (!p.jit.empty())
    if (const cufftResult r = jit_prepare(p)) return r;
  // Everything is computed on the host, so no plan needs a work area.
  if (work)
    for (size_t g = 0; g < std::max<size_t>(1, p.gpus.size()); ++g) work[g] = 0;
  g_plans[h] = p;
  return CUFFT_SUCCESS;
}

cufftHandle alloc_handle() {
  std::lock_guard<std::mutex> l(g_mu);
  return g_next++;
}

// The element counts of one transform's input and output, and where each of
// their elements sits in the user's buffer, as the plan's layout describes.
struct Geometry {
  Elems ie, oe;
  std::vector<int> half;            // the stored half of a real transform's shape
  std::vector<size_t> ioff, ooff;   // offsets of every element of one batch
  size_t idist, odist;
  size_t in_span, out_span;         // elements a whole call touches, batches included
};

Geometry geometry(const Plan& p) {
  const bool c2r = is_c2r(p.type), r2c = is_r2c(p.type);
  Geometry g{Elems{p.prec, !r2c}, Elems{p.prec, !c2r}, p.n, {}, {}, 0, 0, 0, 0};
  g.half.back() = g.half.back() / 2 + 1;
  const std::vector<int>& ishape = c2r ? g.half : p.n;
  const std::vector<int>& oshape = r2c ? g.half : p.n;
  g.ioff = layout(ishape, p.iembed, p.istride);
  g.ooff = layout(oshape, p.oembed, p.ostride);
  g.idist = !p.iembed.empty() && p.idist ? (size_t)p.idist : logical_elems(ishape);
  g.odist = !p.oembed.empty() && p.odist ? (size_t)p.odist : logical_elems(oshape);
  // The exact number of elements the user's buffer must hold: the last batch's
  // base plus its furthest element. Rounding this up would read past the end
  // of the allocation, which VirtualGPU catches and hardware does not.
  g.in_span = (size_t)(p.batch - 1) * g.idist + g.ioff.back() + 1;
  g.out_span = (size_t)(p.batch - 1) * g.odist + g.ooff.back() + 1;
  return g;
}

// The transform itself, on host copies of the input and output buffers. The
// output arrives holding what the user's buffer held, so the gaps a strided or
// padded layout leaves keep it.
void transform(const Plan& p, const Geometry& g, const std::vector<char>& hin,
               std::vector<char>& hout, int direction) {
  const bool c2r = is_c2r(p.type), r2c = is_r2c(p.type);
  const int sign = c2r ? CUFFT_INVERSE : r2c ? CUFFT_FORWARD : direction;
  const int last = p.n.back();
  const std::vector<int>& half = g.half;
  std::vector<cd> work;
  for (long long b = 0; b < p.batch; ++b) {
    const size_t ib = (size_t)b * g.idist, ob = (size_t)b * g.odist;
    work.resize(g.ioff.size());
    for (size_t k = 0; k < g.ioff.size(); ++k) work[k] = g.ie.load(hin, ib + g.ioff[k]);

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
      fft_nd(work, p.n, sign);
      if (r2c) {  // keep the non-redundant half of the fastest dimension
        const size_t lines = work.size() / (size_t)last, h = (size_t)half.back();
        for (size_t l = 0; l < lines; ++l)
          for (size_t k = 0; k < h; ++k) work[l * h + k] = work[l * (size_t)last + k];
        work.resize(lines * h);
      }
    }
    for (size_t k = 0; k < g.ooff.size(); ++k) g.oe.store(hout, ob + g.ooff[k], work[k]);
  }
}

// The single implementation behind every single-GPU Exec entry point. Reads
// the whole input, transforms batch by batch, writes the whole output.
cufftResult exec(cufftHandle handle, const void* idata, void* odata, int direction) {
  Plan* p = find(handle);
  if (!p) return CUFFT_INVALID_PLAN;
  if (p->n.empty()) return CUFFT_INVALID_PLAN;  // created but never made
  // A multi-GPU plan runs on descriptors (cufftXtExecDescriptor*): handed
  // plain pointers, or a descriptor cast to one, NVIDIA's answers
  // INTERNAL_ERROR (RTX 3060 pair).
  if (p->gpus.size() > 1) return CUFFT_INTERNAL_ERROR;
  if (!idata || !odata) return CUFFT_INVALID_VALUE;
  cudaStreamSynchronize(p->stream);
  const Geometry g = geometry(*p);
  // The output is read first so the gaps a strided or padded layout leaves
  // keep what they held; an in-place transform reads the input just after.
  std::vector<char> hout(g.out_span * g.oe.size()), hin(g.in_span * g.ie.size());
  cudaMemcpy(hout.data(), odata, hout.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hin.data(), idata, hin.size(), cudaMemcpyDeviceToHost);
  transform(*p, g, hin, hout, direction);
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
  return make_plan(plan, 1, &nx, nullptr, 1, 0, nullptr, 1, 0, type, batch, 0, work, true);
}
VGPU_EXPORT cufftResult cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type,
                                        size_t* work) {
  const int n[2] = {nx, ny};
  return make_plan(plan, 2, n, nullptr, 1, 0, nullptr, 1, 0, type, 1, 0, work, true);
}
VGPU_EXPORT cufftResult cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type,
                                        size_t* work) {
  const int n[3] = {nx, ny, nz};
  return make_plan(plan, 3, n, nullptr, 1, 0, nullptr, 1, 0, type, 1, 0, work, true);
}
VGPU_EXPORT cufftResult cufftMakePlanMany(cufftHandle plan, int rank, int* n, int* inembed,
                                          int istride, int idist, int* onembed, int ostride,
                                          int odist, cufftType type, int batch, size_t* work) {
  return make_plan(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, 0,
                   work, true);
}
VGPU_EXPORT cufftResult cufftMakePlanMany64(cufftHandle plan, int rank, long long* n,
                                            long long* inembed, long long istride, long long idist,
                                            long long* onembed, long long ostride, long long odist,
                                            cufftType type, long long batch, size_t* work) {
  return make_plan(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, 0,
                   work, true);
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
  return make_plan(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch,
                   prec, work, true);
}

VGPU_EXPORT cufftResult cufftXtGetSizeMany(cufftHandle plan, int, long long*, long long*,
                                           long long, long long, cudaDataType, long long*,
                                           long long, long long, cudaDataType, long long,
                                           size_t* work, cudaDataType) {
  if (!find(plan)) return CUFFT_INVALID_PLAN;
  zero_work(plan, work);
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

// A multi-GPU plan's work sizes are an array, one per GPU (cuFFT docs, "Plan
// Specification and Work Areas").
namespace {
void zero_work(cufftHandle h, size_t* work) {
  if (!work) return;
  const Plan* p = find(h);
  const size_t count = p && p->gpus.size() > 1 ? p->gpus.size() : 1;
  for (size_t g = 0; g < count; ++g) work[g] = 0;
}
}  // namespace

VGPU_EXPORT cufftResult cufftGetSize(cufftHandle handle, size_t* work) {
  if (!find(handle)) return CUFFT_INVALID_PLAN;
  zero_work(handle, work);
  return CUFFT_SUCCESS;
}
VGPU_EXPORT cufftResult cufftGetSize1d(cufftHandle h, int, cufftType, int, size_t* w) {
  zero_work(h, w); return CUFFT_SUCCESS;
}
VGPU_EXPORT cufftResult cufftGetSize2d(cufftHandle h, int, int, cufftType, size_t* w) {
  zero_work(h, w); return CUFFT_SUCCESS;
}
VGPU_EXPORT cufftResult cufftGetSize3d(cufftHandle h, int, int, int, cufftType, size_t* w) {
  zero_work(h, w); return CUFFT_SUCCESS;
}
VGPU_EXPORT cufftResult cufftGetSizeMany(cufftHandle h, int, int*, int*, int, int, int*, int, int,
                                         cufftType, int, size_t* w) {
  zero_work(h, w); return CUFFT_SUCCESS;
}
VGPU_EXPORT cufftResult cufftGetSizeMany64(cufftHandle h, int, long long*, long long*, long long,
                                           long long, long long*, long long, long long, cufftType,
                                           long long, size_t* w) {
  zero_work(h, w); return CUFFT_SUCCESS;
}
VGPU_EXPORT cufftResult cufftEstimate1d(int, cufftType, int, size_t* w) { if (w) *w = 0; return CUFFT_SUCCESS; }
VGPU_EXPORT cufftResult cufftEstimate2d(int, int, cufftType, size_t* w) { if (w) *w = 0; return CUFFT_SUCCESS; }
VGPU_EXPORT cufftResult cufftEstimate3d(int, int, int, cufftType, size_t* w) { if (w) *w = 0; return CUFFT_SUCCESS; }
VGPU_EXPORT cufftResult cufftEstimateMany(int, int*, int*, int, int, int*, int, int, cufftType,
                                          int, size_t* w) { if (w) *w = 0; return CUFFT_SUCCESS; }
VGPU_EXPORT cufftResult cufftSetWorkArea(cufftHandle handle, void*) {
  // A multi-GPU plan takes its work areas through cufftXtSetWorkArea, and
  // NVIDIA's answers this one INVALID_PLAN for it.
  const Plan* p = find(handle);
  return p && p->gpus.size() < 2 ? CUFFT_SUCCESS : CUFFT_INVALID_PLAN;
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

/* ---- multiple GPUs: cufftXtSetGPUs, descriptors, cufftXtExecDescriptor ----
 *
 * The data of a multi-GPU plan really is spread over the simulated devices:
 * cufftXtMalloc allocates each GPU's part on that device, cufftXtMemcpy deals
 * the host array out (and gathers it back) in NVIDIA's order, and the
 * transform gathers every part, runs on the host like the rest of this
 * library, and writes each part back where NVIDIA's leaves it.
 *
 * The orders are the ones cuFFT's documentation describes ("Multiple GPU Data
 * Organization"), measured on two RTX 3060s with CUDA 13.0's cuFFT 12.0
 * (nvidia/tests/e2e/fft_multigpu.cu checks each against the card):
 *
 *   batched (batch > 1)  whole transforms dealt out in order, the first
 *                        batch % G GPUs taking one more; output in place, in
 *                        natural order, the descriptor's subFormat unchanged.
 *   single 2-D / 3-D     natural order (CUFFT_XT_FORMAT_INPLACE) splits the
 *                        slowest axis, x, the first nx % G GPUs taking one
 *                        more plane, each GPU's planes contiguous; the
 *                        transform leaves it split on y instead
 *                        (CUFFT_XT_FORMAT_INPLACE_SHUFFLED), each GPU holding
 *                        [x][its y][z...], and a transform of shuffled data
 *                        returns it to natural order. Each part is allocated
 *                        for the larger of the two. Real transforms count the
 *                        stored half of the fastest axis (padded in place).
 *   single 1-D           natural order is n/G contiguous points per GPU; the
 *                        output is in "strings" (the cuFFT documentation's
 *                        permuted2Linear); CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED
 *                        takes the input already redistributed.
 */
namespace {

enum Spread { kSingle = 0, kBatches, kLine, kSlabs };

std::vector<int> squeezed(const std::vector<int>& n) {
  std::vector<int> s;
  for (int d : n)
    if (d != 1) s.push_back(d);
  return s;
}

bool pow2(long long v) { return v > 0 && (v & (v - 1)) == 0; }
int log2i(long long v) { int k = 0; while ((1LL << k) < v) ++k; return k; }

// The rules a multi-GPU plan has to meet, with the status NVIDIA's returns
// when it does not, each measured on the RTX 3060 pair:
//   - half precision: SETUP_FAILED, even batched.
//   - batch > 1: anything else goes.
//   - batch 1, after dropping axes of size 1 (1x1x64 is a 1-D 64-point plan,
//     32x1x32 a 2-D one): 1-D only C2C/Z2Z of a power of two of at least 64
//     (128 on 8 GPUs, 1024 on 16) on 2, 4, 8 or 16 GPUs; 2-D and 3-D with the
//     two slowest axes at least 32 (the third is free: 32x32x2 is accepted).
//     Anything else is INVALID_SIZE -- a 1-D R2C too, and 8x8, 16x64, 31x32,
//     3x32x32. (The documentation also asks every axis to factor into primes
//     up to 127 or stay within 4096 points; NVIDIA's plans 4099x32 all the
//     same, so this does too.)
//   - an advanced layout (inembed/onembed) is accepted, batched or not; a
//     single transform's descriptors still hold it packed.
cufftResult multi_check(Plan& p) {
  const long long G = (long long)p.gpus.size();
  if (p.prec == 16) return CUFFT_SETUP_FAILED;
  if (p.batch > 1) { p.spread = kBatches; return CUFFT_SUCCESS; }
  const std::vector<int> s = squeezed(p.n);
  if (s.empty()) return CUFFT_INVALID_SIZE;
  if (s.size() == 1) {
    // Two GPUs is all the card pair can show; the other counts follow the
    // documentation's table. (NVIDIA's accepted 64 points on a three-entry
    // GPU list, {0,1,0}, while refusing 1024 on it.)
    const long long min = G >= 16 ? 1024 : G >= 8 ? 128 : 64;
    if (!is_c2c(p.type) || !pow2(s[0]) || s[0] < min || !pow2(G) || G > 16)
      return CUFFT_INVALID_SIZE;
    p.spread = kLine;
    return CUFFT_SUCCESS;
  }
  if (s[0] < 32 || s[1] < 32 || G > 16) return CUFFT_INVALID_SIZE;
  p.spread = kSlabs;
  return CUFFT_SUCCESS;
}

// The factors a single 1-D transform is split into (cufftXtQueryPlan's
// cufftXt1dFactors). factor2 for 2^k points, k = 6..27, as NVIDIA's chose it
// on two GPUs, single and double precision alike; factor1 = n / factor2; 8
// strings, but 2 for 64 points. Past 2^27 the last choice is kept.
struct Factors { long long n, f1, f2, strings; };
Factors factors_1d(long long n, int G) {
  static const int f2_log[] = {2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6,
                               7, 8, 9, 9, 9, 8, 9, 9, 9};
  const int k = log2i(n);
  const int f2l = f2_log[std::min(std::max(k, 6), 27) - 6];
  Factors f{n, n >> f2l, 1LL << f2l, k == 6 ? 2 : 8};
  // More GPUs than the card pair has: every GPU needs a string of its own.
  if (f.strings < G) f.strings = G;
  return f;
}

// Where each GPU's elements sit in the natural host array: at[g][i] is the
// host index of local element i on GPU g, in `unit`-byte elements.
struct Placement {
  size_t unit = 0;
  std::vector<std::vector<size_t>> at;
};

size_t complex_bytes(const Plan& p) { return (size_t)p.prec / 4; }

// The shape a single multi-GPU transform's elements form: the stored half of
// the fastest axis for a real transform, axes of size 1 dropped.
std::vector<int> unit_shape(const Plan& p) {
  std::vector<int> u = p.n;
  if (!is_c2c(p.type)) u.back() = u.back() / 2 + 1;
  return squeezed(u);
}

// A part of `total` for each of G GPUs, the first total % G one larger.
void split(long long total, int G, int g, long long* start, long long* count) {
  const long long base = total / G, extra = total % G;
  *count = base + (g < extra ? 1 : 0);
  *start = g * base + std::min<long long>(g, extra);
}

// Whether a batched real transform's descriptor of this format holds the real
// side packed (n reals a transform) rather than in place (the stored half,
// padded): CUFFT_XT_FORMAT_INPUT of an R2C plan, CUFFT_XT_FORMAT_OUTPUT of a
// C2R one, as their sizes on the card show.
bool packed_real(const Plan& p, int fmt) {
  return (is_r2c(p.type) && fmt == CUFFT_XT_FORMAT_INPUT) ||
         (is_c2r(p.type) && fmt == CUFFT_XT_FORMAT_OUTPUT);
}

Placement place(const Plan& p, int fmt) {
  const int G = (int)p.gpus.size();
  Placement pl;
  pl.at.resize(G);
  pl.unit = complex_bytes(p);
  if (p.spread == kLine) {
    const long long n = squeezed(p.n)[0], part = n / G;
    const Factors f = factors_1d(n, G);
    for (int g = 0; g < G; ++g) {
      pl.at[g].resize((size_t)part);
      for (long long i = 0; i < part; ++i) {
        long long lin = g * part + i;
        if (fmt == CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
          // The documentation's permuted2Linear: strings of factor2
          // substrings, each substring factor1 apart, the strings in order
          // over the GPUs.
          const long long sl = n / f.strings, ssl = sl / f.f2;
          const long long in_sub = i % ssl, sub = (i / ssl) % f.f2;
          const long long str = i / sl + g * (f.strings / G);
          lin = in_sub + str * ssl + sub * f.f1;
        } else if (fmt == CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED) {
          // The input of the first pass, measured: point a + factor2*b
          // (b < factor1) goes to the GPU owning a, each GPU's a values in
          // its strings, q of them a string, b before a within one.
          const long long per_gpu = f.f2 / G, S = std::max<long long>(1, f.strings / G);
          const long long q = std::max<long long>(1, per_gpu / S);
          const long long st = i / (q * f.f1), rem = i % (q * f.f1);
          const long long b = rem / q, a = g * per_gpu + st * q + rem % q;
          lin = a + f.f2 * b;
        }
        pl.at[g][(size_t)i] = (size_t)lin;
      }
    }
    return pl;
  }
  // Slabs: natural order splits axis 0, shuffled order axis 1.
  const std::vector<int> u = unit_shape(p);
  const size_t inner = logical_elems(std::vector<int>(u.begin() + 2, u.end()));
  for (int g = 0; g < G; ++g) {
    long long s0, c;
    if (fmt == CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
      split(u[1], G, g, &s0, &c);
      for (long long x = 0; x < u[0]; ++x)
        for (long long y = s0; y < s0 + c; ++y)
          for (size_t r = 0; r < inner; ++r)
            pl.at[g].push_back(((size_t)x * u[1] + (size_t)y) * inner + r);
    } else {
      split(u[0], G, g, &s0, &c);
      const size_t slab = (size_t)u[1] * inner;
      for (size_t i = 0; i < (size_t)c * slab; ++i) pl.at[g].push_back((size_t)s0 * slab + i);
    }
  }
  return pl;
}

// What cufftXtMalloc allocates on each GPU, in bytes (measured sizes): a
// batched descriptor holds its transforms; a 1-D one n/G points whatever the
// format; a 2-D/3-D one the larger of its natural and shuffled parts, but
// only for the formats a transform can start from -- CUFFT_XT_FORMAT_INPUT,
// _OUTPUT and the formats past _1D_INPUT_SHUFFLED come back empty, and so do
// a 2-D R2C plan's shuffled and a 2-D C2R plan's natural descriptors (each
// can only start from the other).
size_t part_bytes(const Plan& p, int fmt, int g) {
  if (p.spread == kBatches) {
    long long b0, cnt;
    split(p.batch, (int)p.gpus.size(), g, &b0, &cnt);
    if (!cnt) return 0;
    if (!p.iembed.empty()) {
      // An advanced layout: the span its batches cover, as the plan lays them
      // out (8 points at stride 2, distance 16: 31 elements for two batches,
      // 15 for one).
      Plan local = p;
      local.batch = cnt;
      const Geometry geo = geometry(local);
      const size_t in = geo.in_span * geo.ie.size(), out = geo.out_span * geo.oe.size();
      return fmt == CUFFT_XT_FORMAT_INPUT ? in : fmt == CUFFT_XT_FORMAT_OUTPUT ? out : std::max(in, out);
    }
    if (packed_real(p, fmt)) return (size_t)cnt * logical_elems(p.n) * (size_t)p.prec / 8;
    return (size_t)cnt * (is_c2c(p.type) ? logical_elems(p.n) : complex_elems(p.n)) * complex_bytes(p);
  }
  if (p.spread == kSlabs) {
    if (fmt != CUFFT_XT_FORMAT_INPLACE && fmt != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) return 0;
    if (unit_shape(p).size() == 2 && is_r2c(p.type) && fmt == CUFFT_XT_FORMAT_INPLACE_SHUFFLED)
      return 0;
    if (unit_shape(p).size() == 2 && is_c2r(p.type) && fmt == CUFFT_XT_FORMAT_INPLACE) return 0;
    const Placement a = place(p, CUFFT_XT_FORMAT_INPLACE), b = place(p, CUFFT_XT_FORMAT_INPLACE_SHUFFLED);
    return std::max(a.at[g].size(), b.at[g].size()) * a.unit;
  }
  const Placement pl = place(p, fmt);
  return pl.at[g].size() * pl.unit;
}

// Which formats cufftXtMalloc takes, by spread, with NVIDIA's status for the
// others: batched plans take INPUT, OUTPUT and INPLACE (INPLACE_SHUFFLED is
// INVALID_VALUE, 1D_INPUT_SHUFFLED INVALID_PLAN); 1-D plans take the first
// five; 2-D/3-D plans take all but 1D_INPUT_SHUFFLED (INVALID_PLAN), any
// value past it included.
cufftResult format_ok(const Plan& p, int fmt) {
  if (fmt < 0) return CUFFT_INVALID_VALUE;
  if (p.spread == kSlabs) return fmt == CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED ? CUFFT_INVALID_PLAN : CUFFT_SUCCESS;
  if (p.spread == kLine) return fmt <= CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED ? CUFFT_SUCCESS : CUFFT_INVALID_VALUE;
  if (fmt == CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED) return CUFFT_INVALID_PLAN;
  return fmt <= CUFFT_XT_FORMAT_INPLACE ? CUFFT_SUCCESS : CUFFT_INVALID_VALUE;
}

// Restores the current device on the way out of a call that visits others.
struct DeviceGuard {
  int saved = 0;
  DeviceGuard() { cudaGetDevice(&saved); }
  ~DeviceGuard() { cudaSetDevice(saved); }
};

// Every GPU's part of a descriptor, gathered into the natural host array
// (`bytes` long; elements no part holds are left as they are).
// A batched descriptor's parts are the host array's consecutive pieces, each
// as long as the part: NVIDIA's copies them so even when an advanced layout
// leaves the last batch on a GPU short of the next GPU's first.
void gather(const Plan& p, const cudaLibXtDesc* d, std::vector<char>& host) {
  if (p.spread == kBatches) {
    size_t at = 0;
    for (int g = 0; g < d->descriptor->nGPUs; ++g) {
      const size_t bytes = std::min(d->descriptor->size[g], host.size() - std::min(host.size(), at));
      if (bytes) cudaMemcpy(host.data() + at, d->descriptor->data[g], bytes, cudaMemcpyDefault);
      at += d->descriptor->size[g];
    }
    return;
  }
  const Placement pl = place(p, d->subFormat);
  for (int g = 0; g < (int)pl.at.size(); ++g) {
    const size_t bytes = std::min(pl.at[g].size() * pl.unit, d->descriptor->size[g]);
    if (!bytes) continue;
    std::vector<char> part(bytes);
    cudaMemcpy(part.data(), d->descriptor->data[g], bytes, cudaMemcpyDefault);
    for (size_t i = 0; i < bytes / pl.unit; ++i)
      std::memcpy(host.data() + pl.at[g][i] * pl.unit, part.data() + i * pl.unit, pl.unit);
  }
}

void scatter(const Plan& p, cudaLibXtDesc* d, const std::vector<char>& host) {
  if (p.spread == kBatches) {
    size_t at = 0;
    for (int g = 0; g < d->descriptor->nGPUs; ++g) {
      const size_t bytes = std::min(d->descriptor->size[g], host.size() - std::min(host.size(), at));
      if (bytes) cudaMemcpy(d->descriptor->data[g], host.data() + at, bytes, cudaMemcpyDefault);
      at += d->descriptor->size[g];
    }
    return;
  }
  const Placement pl = place(p, d->subFormat);
  for (int g = 0; g < (int)pl.at.size(); ++g) {
    const size_t bytes = std::min(pl.at[g].size() * pl.unit, d->descriptor->size[g]);
    if (!bytes) continue;
    std::vector<char> part(bytes);
    for (size_t i = 0; i < bytes / pl.unit; ++i)
      std::memcpy(part.data() + i * pl.unit, host.data() + pl.at[g][i] * pl.unit, pl.unit);
    cudaMemcpy(d->descriptor->data[g], part.data(), bytes, cudaMemcpyDefault);
  }
}

// The natural host array a descriptor of this format stands for, in bytes.
size_t host_bytes(const Plan& p, int fmt) {
  if (p.spread == kBatches) {
    size_t n = 0;
    for (int g = 0; g < (int)p.gpus.size(); ++g) n += part_bytes(p, fmt, g);
    return n;
  }
  const Placement pl = place(p, fmt);
  size_t n = 0;
  for (const auto& a : pl.at) n += a.size();
  return n * pl.unit;
}

// The single-GPU plan that transforms host copies of the data: every batch
// packed, a real side packed (out of place) or padded in place to the stored
// half's width. A batched plan with an advanced layout keeps it: each GPU runs
// its batches in that layout on its own part.
Plan host_plan(const Plan& p, bool packed_real_side) {
  Plan h = p;
  h.gpus.clear();
  h.spread = kSingle;
  if (p.spread == kBatches && !p.iembed.empty()) return h;
  h.iembed.clear();
  h.oembed.clear();
  h.istride = h.ostride = 1;
  if (is_c2c(p.type)) return h;
  std::vector<long long> real(p.n.begin(), p.n.end()), cplx = real;
  cplx.back() = cplx.back() / 2 + 1;
  std::vector<long long> padded = cplx;
  padded.back() *= 2;
  const bool r2c = is_r2c(p.type);
  const std::vector<long long>& rside = packed_real_side ? real : padded;
  long long rdist = 1, cdist = 1;
  for (long long v : rside) rdist *= v;
  for (long long v : cplx) cdist *= v;
  h.iembed = r2c ? rside : cplx;
  h.oembed = r2c ? cplx : rside;
  h.idist = r2c ? rdist : cdist;
  h.odist = r2c ? cdist : rdist;
  return h;
}

std::mutex g_desc_mu;
std::set<const cudaLibXtDesc*> g_descs;  // the descriptors cufftXtMalloc made
bool live_desc(const cudaLibXtDesc* d) {
  std::lock_guard<std::mutex> l(g_desc_mu);
  return d && g_descs.count(d);
}

// Runs `hp` on host copies of `in` and `out` (`in_bytes` and `out_bytes`
// long, from the given device pointers), writing the result back.
void run_part(const Plan& hp, const void* in, size_t in_bytes, void* out, size_t out_bytes,
              bool in_place, int direction) {
  const Geometry geo = geometry(hp);
  std::vector<char> hin(std::max(in_bytes, geo.in_span * geo.ie.size()));
  std::vector<char> hout(std::max(out_bytes, geo.out_span * geo.oe.size()));
  if (in_bytes) cudaMemcpy(hin.data(), in, in_bytes, cudaMemcpyDefault);
  if (in_place) std::memcpy(hout.data(), hin.data(), std::min(hin.size(), hout.size()));
  else if (out_bytes) cudaMemcpy(hout.data(), out, out_bytes, cudaMemcpyDefault);
  transform(hp, geo, hin, hout, direction);
  if (out_bytes) cudaMemcpy(out, hout.data(), out_bytes, cudaMemcpyDefault);
}

// cufftXtExecDescriptor*: `want` is the plan type the entry point is for, or
// 0 for cufftXtExecDescriptor itself.
cufftResult exec_desc(cufftHandle plan, cudaLibXtDesc* in, cudaLibXtDesc* out, int direction,
                      cufftType want) {
  Plan* pp = find(plan);
  // NVIDIA's answers a missing descriptor, and an entry point for another
  // type, with INVALID_PLAN.
  if (!pp || pp->n.empty() || !in || !out) return CUFFT_INVALID_PLAN;
  const Plan p = *pp;
  if (want && p.type != want) return CUFFT_INVALID_PLAN;
  if (is_c2c(p.type) && direction != CUFFT_FORWARD && direction != CUFFT_INVERSE)
    return CUFFT_INVALID_VALUE;
  // A single-GPU plan's descriptor holds nothing (cufftXtMalloc gives it no
  // memory), and NVIDIA's answers EXEC_FAILED.
  if (p.gpus.size() < 2) return CUFFT_EXEC_FAILED;
  if (!live_desc(in) || !live_desc(out)) return CUFFT_INVALID_VALUE;
  DeviceGuard dg;
  for (int g : p.gpus) {
    cudaSetDevice(g);
    cudaDeviceSynchronize();
  }
  if (p.spread == kBatches) {
    // Each GPU transforms its own batches where they are, whatever the
    // descriptors' subFormats say, which are left as they were. Out of place,
    // a real side is read or written packed whatever the descriptor's format
    // (an R2C from an in-place descriptor reads n reals a transform).
    const bool in_place = in == out;
    for (int g = 0; g < (int)p.gpus.size() && g < in->descriptor->nGPUs; ++g) {
      long long b0, cnt;
      split(p.batch, (int)p.gpus.size(), g, &b0, &cnt);
      if (!cnt) continue;
      Plan hp = host_plan(p, !in_place);
      hp.batch = cnt;
      run_part(hp, in->descriptor->data[g], in->descriptor->size[g], out->descriptor->data[g],
               out->descriptor->size[g], in_place, direction);
    }
    return CUFFT_SUCCESS;
  }
  const int f = in->subFormat;
  int out_fmt;
  if (p.spread == kLine) {
    // Natural or input-shuffled data in, strings out. Anything else is
    // INVALID_TYPE: strings (inverting a 1-D result takes a cufftXtMemcpy
    // device to device first) and the formats with no layout of their own.
    if (f != CUFFT_XT_FORMAT_INPLACE && f != CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED)
      return CUFFT_INVALID_TYPE;
    out_fmt = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
  } else {
    // Natural in, shuffled out, and back again; any other subFormat is
    // INTERNAL_ERROR. A 2-D R2C takes only natural input and a 2-D C2R only
    // shuffled (the documentation's table; the other is INTERNAL_ERROR on
    // the card), where 3-D real transforms take either.
    if (f != CUFFT_XT_FORMAT_INPLACE && f != CUFFT_XT_FORMAT_INPLACE_SHUFFLED)
      return CUFFT_INTERNAL_ERROR;
    if (unit_shape(p).size() == 2 && ((is_r2c(p.type) && f != CUFFT_XT_FORMAT_INPLACE) ||
                                      (is_c2r(p.type) && f != CUFFT_XT_FORMAT_INPLACE_SHUFFLED)))
      return CUFFT_INTERNAL_ERROR;
    out_fmt = f == CUFFT_XT_FORMAT_INPLACE ? CUFFT_XT_FORMAT_INPLACE_SHUFFLED
                                           : CUFFT_XT_FORMAT_INPLACE;
  }
  // A single transform runs in place only: out of place is EXEC_FAILED.
  if (in != out) return CUFFT_EXEC_FAILED;
  const Plan hp = host_plan(p, false);
  const Geometry geo = geometry(hp);
  std::vector<char> hin(std::max(host_bytes(p, f), geo.in_span * geo.ie.size()));
  gather(p, in, hin);
  std::vector<char> hout = hin;  // the padding of an in-place real layout keeps its contents
  hout.resize(std::max(hout.size(), geo.out_span * geo.oe.size()));
  transform(hp, geo, hin, hout, direction);
  out->subFormat = out_fmt;
  scatter(p, out, hout);
  return CUFFT_SUCCESS;
}

}  // namespace

VGPU_EXPORT cufftResult cufftXtSetGPUs(cufftHandle handle, int nGPUs, int* whichGPUs) {
  Plan* p = find(handle);
  // RTX 3060 pair: an unknown handle, or one already made, is INVALID_PLAN;
  // fewer than two GPUs or no list INVALID_VALUE; a device that does not
  // exist INVALID_DEVICE. The same device twice is accepted.
  if (!p || !p->n.empty()) return CUFFT_INVALID_PLAN;
  if (nGPUs < 2 || !whichGPUs) return CUFFT_INVALID_VALUE;
  int count = 0;
  cudaGetDeviceCount(&count);
  for (int i = 0; i < nGPUs; ++i)
    if (whichGPUs[i] < 0 || whichGPUs[i] >= count) return CUFFT_INVALID_DEVICE;
  std::lock_guard<std::mutex> l(g_mu);
  p->gpus.assign(whichGPUs, whichGPUs + nGPUs);
  return CUFFT_SUCCESS;
}

VGPU_EXPORT cufftResult cufftXtMalloc(cufftHandle plan, cudaLibXtDesc** descriptor,
                                      cufftXtSubFormat format) {
  Plan* pp = find(plan);
  if (!pp || pp->n.empty()) return CUFFT_INVALID_PLAN;
  if (!descriptor) return CUFFT_INVALID_VALUE;
  const Plan p = *pp;
  auto* d = new cudaLibXtDesc();
  d->version = 0;  // what NVIDIA's writes, here and in the inner descriptor
  d->descriptor = new cudaXtDesc();
  d->library = LIB_FORMAT_CUFFT;
  d->subFormat = format;
  d->libDescriptor = d;  // NVIDIA's points it at a structure of its own
  int current = 0;
  cudaGetDevice(&current);
  if (p.gpus.size() < 2) {
    // A single-GPU plan gets a descriptor of one empty part on the current
    // device: NVIDIA's allocates nothing for it.
    d->descriptor->nGPUs = 1;
    d->descriptor->GPUs[0] = current;
  } else {
    if (const cufftResult r = format_ok(p, format)) {
      delete d->descriptor;
      delete d;
      return r;
    }
    DeviceGuard dg;
    d->descriptor->nGPUs = (int)p.gpus.size();
    for (int g = 0; g < (int)p.gpus.size(); ++g) {
      d->descriptor->GPUs[g] = p.gpus[g];
      const size_t bytes = part_bytes(p, format, g);
      d->descriptor->size[g] = bytes;
      if (!bytes) continue;
      cudaSetDevice(p.gpus[g]);
      if (cudaMalloc(&d->descriptor->data[g], bytes) != cudaSuccess) {
        for (int k = 0; k < g; ++k) cudaFree(d->descriptor->data[k]);
        delete d->descriptor;
        delete d;
        return CUFFT_ALLOC_FAILED;
      }
    }
  }
  {
    std::lock_guard<std::mutex> l(g_desc_mu);
    g_descs.insert(d);
  }
  *descriptor = d;
  return CUFFT_SUCCESS;
}

VGPU_EXPORT cufftResult cufftXtFree(cudaLibXtDesc* descriptor) {
  if (!descriptor) return CUFFT_SUCCESS;  // as NVIDIA's answers it
  {
    std::lock_guard<std::mutex> l(g_desc_mu);
    if (!g_descs.erase(descriptor)) return CUFFT_INVALID_VALUE;
  }
  DeviceGuard dg;
  for (int g = 0; g < descriptor->descriptor->nGPUs; ++g)
    if (descriptor->descriptor->data[g]) {
      cudaSetDevice(descriptor->descriptor->GPUs[g]);
      cudaFree(descriptor->descriptor->data[g]);
    }
  delete descriptor->descriptor;
  delete descriptor;
  return CUFFT_SUCCESS;
}

VGPU_EXPORT cufftResult cufftXtMemcpy(cufftHandle plan, void* dstPointer, void* srcPointer,
                                      cufftXtCopyType type) {
  Plan* pp = find(plan);
  if (!pp || pp->n.empty()) return CUFFT_INVALID_PLAN;
  // A copy type out of range, or a NULL either side, is INVALID_VALUE.
  if (type != CUFFT_COPY_HOST_TO_DEVICE && type != CUFFT_COPY_DEVICE_TO_HOST &&
      type != CUFFT_COPY_DEVICE_TO_DEVICE)
    return CUFFT_INVALID_VALUE;
  if (!dstPointer || !srcPointer) return CUFFT_INVALID_VALUE;
  const Plan p = *pp;
  // A single-GPU plan's descriptors hold nothing to copy.
  if (p.gpus.size() < 2) return CUFFT_SUCCESS;
  DeviceGuard dg;
  for (int g : p.gpus) {
    cudaSetDevice(g);
    cudaDeviceSynchronize();
  }
  if (type == CUFFT_COPY_HOST_TO_DEVICE) {
    auto* d = static_cast<cudaLibXtDesc*>(dstPointer);
    if (!live_desc(d)) return CUFFT_INVALID_VALUE;
    // A 1-D result's string order cannot be written from the host:
    // INVALID_TYPE, and nothing copied.
    if (p.spread == kLine && d->subFormat == CUFFT_XT_FORMAT_INPLACE_SHUFFLED)
      return CUFFT_INVALID_TYPE;
    const size_t bytes = host_bytes(p, d->subFormat);
    std::vector<char> host(static_cast<const char*>(srcPointer),
                           static_cast<const char*>(srcPointer) + bytes);
    scatter(p, d, host);
    return CUFFT_SUCCESS;
  }
  auto* s = static_cast<cudaLibXtDesc*>(srcPointer);
  if (!live_desc(s)) return CUFFT_INVALID_VALUE;
  // Input-shuffled 1-D data only goes to a transform: copying it out, to the
  // host or another descriptor, is INVALID_TYPE.
  if (p.spread == kLine && s->subFormat == CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED)
    return CUFFT_INVALID_TYPE;
  if (type == CUFFT_COPY_DEVICE_TO_HOST) {
    const size_t bytes = host_bytes(p, s->subFormat);
    std::vector<char> host(static_cast<const char*>(dstPointer),
                           static_cast<const char*>(dstPointer) + bytes);
    gather(p, s, host);
    std::memcpy(dstPointer, host.data(), bytes);
    return CUFFT_SUCCESS;
  }
  auto* d = static_cast<cudaLibXtDesc*>(dstPointer);
  if (!live_desc(d)) return CUFFT_INVALID_VALUE;
  // Device to device puts the data in natural order. NVIDIA's copies nothing
  // for a batched plan, and still answers SUCCESS; refuses a shuffled
  // destination, and a 2-D real transform's shuffled source, with
  // INTERNAL_ERROR (RTX 3060 pair).
  if (p.spread == kBatches) return CUFFT_SUCCESS;
  if (d->subFormat == CUFFT_XT_FORMAT_INPLACE_SHUFFLED) return CUFFT_INTERNAL_ERROR;
  if (p.spread == kSlabs && !is_c2c(p.type) && unit_shape(p).size() == 2 &&
      s->subFormat == CUFFT_XT_FORMAT_INPLACE_SHUFFLED)
    return CUFFT_INTERNAL_ERROR;
  std::vector<char> host(host_bytes(p, s->subFormat));
  gather(p, s, host);
  d->subFormat = CUFFT_XT_FORMAT_INPLACE;
  scatter(p, d, host);
  return CUFFT_SUCCESS;
}

VGPU_EXPORT cufftResult cufftXtExecDescriptorC2C(cufftHandle plan, cudaLibXtDesc* input,
                                                 cudaLibXtDesc* output, int direction) {
  return exec_desc(plan, input, output, direction, CUFFT_C2C);
}
VGPU_EXPORT cufftResult cufftXtExecDescriptorZ2Z(cufftHandle plan, cudaLibXtDesc* input,
                                                 cudaLibXtDesc* output, int direction) {
  return exec_desc(plan, input, output, direction, CUFFT_Z2Z);
}
VGPU_EXPORT cufftResult cufftXtExecDescriptorR2C(cufftHandle plan, cudaLibXtDesc* input,
                                                 cudaLibXtDesc* output) {
  return exec_desc(plan, input, output, CUFFT_FORWARD, CUFFT_R2C);
}
VGPU_EXPORT cufftResult cufftXtExecDescriptorD2Z(cufftHandle plan, cudaLibXtDesc* input,
                                                 cudaLibXtDesc* output) {
  return exec_desc(plan, input, output, CUFFT_FORWARD, CUFFT_D2Z);
}
VGPU_EXPORT cufftResult cufftXtExecDescriptorC2R(cufftHandle plan, cudaLibXtDesc* input,
                                                 cudaLibXtDesc* output) {
  return exec_desc(plan, input, output, CUFFT_INVERSE, CUFFT_C2R);
}
VGPU_EXPORT cufftResult cufftXtExecDescriptorZ2D(cufftHandle plan, cudaLibXtDesc* input,
                                                 cudaLibXtDesc* output) {
  return exec_desc(plan, input, output, CUFFT_INVERSE, CUFFT_Z2D);
}
VGPU_EXPORT cufftResult cufftXtExecDescriptor(cufftHandle plan, cudaLibXtDesc* input,
                                              cudaLibXtDesc* output, int direction) {
  const Plan* p = find(plan);
  if (!p) return CUFFT_INVALID_PLAN;
  if (!is_c2c(p->type)) direction = is_r2c(p->type) ? CUFFT_FORWARD : CUFFT_INVERSE;
  return exec_desc(plan, input, output, direction, (cufftType)0);
}

VGPU_EXPORT cufftResult cufftXtQueryPlan(cufftHandle plan, void* queryStruct,
                                         cufftXtQueryType queryType) {
  // NULL, or a query type other than the 1-D factors, is INVALID_VALUE; a
  // plan that is not a single multi-GPU 1-D transform INVALID_PLAN. (NVIDIA's
  // divides by zero on a batched multi-GPU plan; this answers INVALID_PLAN.)
  if (!queryStruct || queryType != CUFFT_QUERY_1D_FACTORS) return CUFFT_INVALID_VALUE;
  const Plan* p = find(plan);
  if (!p || p->spread != kLine) return CUFFT_INVALID_PLAN;
  const long long n = squeezed(p->n)[0];
  const Factors f = factors_1d(n, (int)p->gpus.size());
  auto* q = static_cast<cufftXt1dFactors*>(queryStruct);
  q->size = n;
  q->stringCount = f.strings;
  q->stringLength = n / f.strings;
  q->substringLength = q->stringLength / f.f2;
  q->factor1 = f.f1;
  q->factor2 = f.f2;
  // The masks and shifts of a power-of-two size, as NVIDIA's fills them:
  // each length's mask is length - 1 and its shift log2(length).
  q->stringMask = q->stringLength - 1;
  q->substringMask = q->substringLength - 1;
  q->factor1Mask = f.f1 - 1;
  q->factor2Mask = f.f2 - 1;
  q->stringShift = log2i(q->stringLength);
  q->substringShift = log2i(q->substringLength);
  q->factor1Shift = log2i(f.f1);
  q->factor2Shift = log2i(f.f2);
  return CUFFT_SUCCESS;
}

VGPU_EXPORT cufftResult cufftXtSetWorkArea(cufftHandle plan, void** workArea) {
  const Plan* p = find(plan);
  if (!p) return CUFFT_INVALID_PLAN;
  // A multi-GPU plan takes one pointer per GPU and needs none of them; a
  // single-GPU plan answers a NULL list INVALID_VALUE (RTX 3060).
  if (p->gpus.size() < 2 && !workArea) return CUFFT_INVALID_VALUE;
  return CUFFT_SUCCESS;
}

// There is no work area to shrink. What NVIDIA's answers (RTX 3060): a plan
// not yet made, or a multi-GPU one, is INVALID_PLAN; CUFFT_WORKAREA_MINIMAL
// succeeds and leaves *workSize alone; CUFFT_WORKAREA_USER, and
// CUFFT_WORKAREA_PERFORMANCE with no workSize, are INVALID_PLAN too.
VGPU_EXPORT cufftResult cufftXtSetWorkAreaPolicy(cufftHandle plan, cufftXtWorkAreaPolicy policy,
                                                 size_t* workSize) {
  const Plan* p = find(plan);
  if (!p || p->n.empty() || p->gpus.size() > 1) return CUFFT_INVALID_PLAN;
  if (policy == CUFFT_WORKAREA_MINIMAL) return CUFFT_SUCCESS;
  if (policy == CUFFT_WORKAREA_PERFORMANCE && workSize) return CUFFT_SUCCESS;
  return CUFFT_INVALID_PLAN;
}

/* ---- callbacks ----
 *
 * Legacy callbacks (cufftXtSetCallback with a device function pointer) exist
 * only in NVIDIA's static library, libcufft_static.a, as cuFFT's
 * documentation says: its libcufft.so answers every legacy callback call with
 * NOT_IMPLEMENTED, a valid plan or not (RTX 3060, CUDA 13.0). This library
 * stands in for libcufft.so, so it answers the same. A program linked against
 * libcufft_static carries NVIDIA's own cuFFT and never reaches this one.
 */
VGPU_EXPORT cufftResult cufftXtSetCallback(cufftHandle, void**, cufftXtCallbackType, void**) {
  return CUFFT_NOT_IMPLEMENTED;
}
VGPU_EXPORT cufftResult cufftXtClearCallback(cufftHandle, cufftXtCallbackType) {
  return CUFFT_NOT_IMPLEMENTED;
}
// Shared memory for a callback: only a plan with LTO callbacks, once made,
// takes it, for any callback type; anything else is INVALID_PLAN, as NVIDIA's
// answers. (NVIDIA's crashes when asked before the plan is made.)
VGPU_EXPORT cufftResult cufftXtSetCallbackSharedSize(cufftHandle plan, cufftXtCallbackType type,
                                                     size_t sharedSize) {
  std::lock_guard<std::mutex> l(g_mu);
  auto it = g_plans.find(plan);
  if (it == g_plans.end() || it->second.jit.empty() || it->second.n.empty())
    return CUFFT_INVALID_PLAN;
  if ((int)type >= 0 && (int)type < CUFFT_CB_UNDEFINED) it->second.jit_shared[type] = sharedSize;
  return CUFFT_SUCCESS;
}

namespace {
cufftResult jit_prepare(Plan&) { return CUFFT_SUCCESS; }
}  // namespace
