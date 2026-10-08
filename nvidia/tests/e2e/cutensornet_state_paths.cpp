// cuTensorNet's state API: tensor network states of quantum circuits, the
// tensor operators applied to them (plain, diagonal, controlled, network
// operators, channels), and what is computed from them -- amplitudes,
// reduced density matrices, expectation values, samples -- by contraction or
// from a matrix product state (MPS).
//
// The expectations hold for NVIDIA's libcutensornet 2.14 on an RTX 3060 too,
// except three that are still the simulator's own and unmatched on the card
// (the final successful AccessorCompute after the argument errors, and the
// two CreateExpectation refusals for another set of modes and another data
// type, which the card accepts). Conventions (the matrix layout of an operator,
// the order of modes, which operators cancel in a norm, what an integer id
// is) were measured on the card and are checked here against a dense host
// simulation. The contraction paths, workspace sizes and FLOP counts are each
// library's own and are only taken as the library reports them.
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

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

static cutensornetHandle_t h;

// ---- host helpers ----

static unsigned rng_state = 12345;
static double rnd() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return ((double)(rng_state % 20001) - 10000.0) / 10000.0;
}
static std::vector<cd> randv(size_t n, bool cx = true) {
  std::vector<cd> v(n);
  for (auto& x : v) {
    const double a = rnd();
    x = cx ? cd(a, rnd()) : cd(a, 0);
  }
  return v;
}

static size_t type_bytes(cudaDataType_t t) { return t == CUDA_C_64F ? 16 : t == CUDA_C_32F ? 8 : t == CUDA_R_64F ? 8 : 4; }

// A device array of numbers of the given type, filled from complex doubles.
struct Dev {
  void* p = nullptr;
  size_t n = 0;
  cudaDataType_t t = CUDA_C_64F;
  explicit Dev(size_t count, cudaDataType_t ty = CUDA_C_64F) : n(count), t(ty) {
    cudaMalloc(&p, std::max<size_t>(1, count) * type_bytes(ty));
    cudaMemset(p, 0, std::max<size_t>(1, count) * type_bytes(ty));
  }
  Dev(const std::vector<cd>& v, cudaDataType_t ty = CUDA_C_64F) : Dev(v.size(), ty) { put(v); }
  void put(const std::vector<cd>& v) {
    if (t == CUDA_C_64F) cudaMemcpy(p, v.data(), v.size() * 16, cudaMemcpyHostToDevice);
    else if (t == CUDA_C_32F) {
      std::vector<std::complex<float>> f(v.begin(), v.end());
      cudaMemcpy(p, f.data(), f.size() * 8, cudaMemcpyHostToDevice);
    } else if (t == CUDA_R_64F) {
      std::vector<double> f;
      for (auto& x : v) f.push_back(x.real());
      cudaMemcpy(p, f.data(), f.size() * 8, cudaMemcpyHostToDevice);
    } else {
      std::vector<float> f;
      for (auto& x : v) f.push_back((float)x.real());
      cudaMemcpy(p, f.data(), f.size() * 4, cudaMemcpyHostToDevice);
    }
  }
  std::vector<cd> get() const {
    cudaDeviceSynchronize();
    std::vector<cd> v(n);
    if (t == CUDA_C_64F) cudaMemcpy(v.data(), p, n * 16, cudaMemcpyDeviceToHost);
    else if (t == CUDA_C_32F) {
      std::vector<std::complex<float>> f(n);
      cudaMemcpy(f.data(), p, n * 8, cudaMemcpyDeviceToHost);
      for (size_t i = 0; i < n; ++i) v[i] = f[i];
    } else if (t == CUDA_R_64F) {
      std::vector<double> f(n);
      cudaMemcpy(f.data(), p, n * 8, cudaMemcpyDeviceToHost);
      for (size_t i = 0; i < n; ++i) v[i] = f[i];
    } else {
      std::vector<float> f(n);
      cudaMemcpy(f.data(), p, n * 4, cudaMemcpyDeviceToHost);
      for (size_t i = 0; i < n; ++i) v[i] = f[i];
    }
    return v;
  }
  ~Dev() { cudaFree(p); }
};

static double maxdiff(const std::vector<cd>& a, const std::vector<cd>& b) {
  if (a.size() != b.size()) return 1e30;
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}
static std::vector<cd> conjv(std::vector<cd> v) {
  for (auto& x : v) x = std::conj(x);
  return v;
}
static std::vector<cd> dagger(const std::vector<cd>& M, size_t D) {
  std::vector<cd> t(M.size());
  for (size_t r = 0; r < D; ++r)
    for (size_t c = 0; c < D; ++c) t[c * D + r] = std::conj(M[r * D + c]);
  return t;
}

// A dense state over modes with extents `ext` (mode 0 fastest) and a matrix
// applied to listed modes (row-major, the first listed mode most significant).
struct Ref {
  std::vector<int64_t> ext;
  std::vector<cd> v;
  explicit Ref(std::vector<int64_t> e) : ext(e) {
    size_t n = 1;
    for (auto x : e) n *= x;
    v.assign(n, 0);
    v[0] = 1;
  }
  void apply(const std::vector<int>& modes, const std::vector<cd>& M) {
    const size_t k = modes.size();
    size_t D = 1;
    for (int m : modes) D *= ext[m];
    std::vector<int64_t> stride(ext.size());
    {
      int64_t s = 1;
      for (size_t i = 0; i < ext.size(); ++i) stride[i] = s, s *= ext[i];
    }
    std::vector<cd> out(v.size(), 0);
    for (size_t idx = 0; idx < v.size(); ++idx) {
      if (v[idx] == cd(0)) continue;
      std::vector<int64_t> dig(ext.size());
      size_t r0 = idx;
      for (size_t i = 0; i < ext.size(); ++i) dig[i] = r0 % ext[i], r0 /= ext[i];
      size_t c = 0;
      for (size_t j = 0; j < k; ++j) c = c * ext[modes[j]] + dig[modes[j]];
      for (size_t r = 0; r < D; ++r) {
        size_t rr = r;
        std::vector<int64_t> od = dig;
        for (int j = (int)k - 1; j >= 0; --j) od[modes[j]] = rr % ext[modes[j]], rr /= ext[modes[j]];
        size_t oi = 0;
        for (size_t i = 0; i < ext.size(); ++i) oi += od[i] * stride[i];
        out[oi] += M[r * D + c] * v[idx];
      }
    }
    v = out;
  }
};

// The workspace a prepare call asked for: scratch (and cache, if any) of the
// recommended sizes, set on the descriptor.
struct WS {
  cutensornetWorkspaceDescriptor_t d = nullptr;
  void* scratch = nullptr;
  void* cache = nullptr;
  WS() { cutensornetCreateWorkspaceDescriptor(h, &d); }
  ~WS() {
    cudaFree(scratch);
    cudaFree(cache);
    cutensornetDestroyWorkspaceDescriptor(d);
  }
  // after a prepare call
  void alloc() {
    int64_t rc = 0, cmax = 0;
    cutensornetWorkspaceGetMemorySize(h, d, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE,
                                      CUTENSORNET_WORKSPACE_SCRATCH, &rc);
    cutensornetWorkspaceGetMemorySize(h, d, CUTENSORNET_WORKSIZE_PREF_MAX, CUTENSORNET_MEMSPACE_DEVICE,
                                      CUTENSORNET_WORKSPACE_CACHE, &cmax);
    cudaFree(scratch);
    cudaFree(cache);
    scratch = cache = nullptr;
    cudaMalloc(&scratch, std::max<int64_t>(rc, 256));
    cutensornetWorkspaceSetMemory(h, d, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, scratch,
                                  std::max<int64_t>(rc, 256));
    if (cmax > 0) {
      cudaMalloc(&cache, cmax);
      cutensornetWorkspaceSetMemory(h, d, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_CACHE, cache, cmax);
    }
  }
};

// A state with the dense reference it must equal.
struct Circuit {
  cutensornetState_t st = nullptr;
  Ref ref;
  int n;
  bool mixed;
  cudaDataType_t type;
  std::vector<Dev*> keep;
  Circuit(std::vector<int64_t> ext, bool mixed_ = false, cudaDataType_t t = CUDA_C_64F)
      : ref(ext), n((int)ext.size()), mixed(mixed_), type(t) {
    cutensornetCreateState(h, mixed ? CUTENSORNET_STATE_PURITY_MIXED : CUTENSORNET_STATE_PURITY_PURE, n, ext.data(), t,
                           &st);
  }
  ~Circuit() {
    cutensornetDestroyState(st);
    for (Dev* d : keep) delete d;
  }
  size_t dim(const std::vector<int>& modes) const {
    size_t D = 1;
    for (int m : modes) D *= ref.ext[m];
    return D;
  }
  std::vector<cd> random_matrix(const std::vector<int>& modes, bool real = false) {
    const size_t D = dim(modes);
    return randv(D * D, !real && (type == CUDA_C_64F || type == CUDA_C_32F));
  }
  // a tensor operator; the reference is updated
  int64_t gate(std::vector<int> modes, const std::vector<cd>& M, int unitary = 0, bool adjoint = false,
               int immutable = 1) {
    Dev* d = new Dev(M, type);
    keep.push_back(d);
    std::vector<int32_t> m(modes.begin(), modes.end());
    int64_t id = -1;
    const cutensornetStatus_t s = cutensornetStateApplyTensorOperator(h, st, (int)m.size(), m.data(), d->p, nullptr,
                                                                      immutable, adjoint, unitary, &id);
    if (s) std::printf("apply -> %d\n", (int)s);
    ref.apply(modes, adjoint ? dagger(M, dim(modes)) : M);
    return id;
  }
};

static std::vector<cd> accessor_full(Circuit& c, size_t count, cd* norm = nullptr, const int64_t* strides = nullptr) {
  cutensornetStateAccessor_t acc;
  cutensornetCreateAccessor(h, c.st, 0, nullptr, strides, &acc);
  WS ws;
  cutensornetAccessorPrepare(h, acc, (size_t)1 << 30, ws.d, 0);
  ws.alloc();
  Dev out(count, c.type);
  unsigned char nb[16] = {0};
  cutensornetStatus_t s = cutensornetAccessorCompute(h, acc, nullptr, ws.d, out.p, norm ? nb : nullptr, 0);
  if (s) std::printf("accessor compute -> %d\n", (int)s);
  if (norm) {
    if (c.type == CUDA_C_64F) *norm = cd(((double*)nb)[0], ((double*)nb)[1]);
    else if (c.type == CUDA_C_32F) *norm = cd(((float*)nb)[0], ((float*)nb)[1]);
    else if (c.type == CUDA_R_64F) *norm = ((double*)nb)[0];
    else *norm = ((float*)nb)[0];
  }
  cutensornetDestroyAccessor(acc);
  return out.get();
}

// ---- a pure circuit: the matrix convention, adjoints, projections, strides ----

static void test_conventions() {
  std::printf("-- conventions\n");
  Circuit c({2, 3, 2});
  int64_t id0 = c.gate({0}, c.random_matrix({0}));
  // a pure state's tensors hold ids 0..n for the vacuum; the first operator is next
  check(id0 == 4, "the first operator of a 3-mode pure state has id 4");
  int64_t id1 = c.gate({1}, c.random_matrix({1}));
  check(id1 == 5, "the next operator has the next id");
  c.gate({2}, c.random_matrix({2}));
  c.gate({2, 1}, c.random_matrix({2, 1}));
  c.gate({0, 2}, c.random_matrix({0, 2}));
  c.gate({1, 2, 0}, c.random_matrix({1, 2, 0}));
  c.gate({1, 0}, c.random_matrix({1, 0}), 0, true);  // applied as an adjoint
  cd nrm;
  auto v = accessor_full(c, 12, &nrm);
  check(maxdiff(v, c.ref.v) < 1e-9, "the state's amplitudes follow the row-major matrix convention");
  double want = 0;
  for (auto& x : c.ref.v) want += std::norm(x);
  check(std::abs(nrm.real() - want) < 1e-9 * want && std::abs(nrm.imag()) < 1e-9 * want, "the squared norm is returned");
  // row-major amplitudes: strides (3, 1) over modes (0, 1, 2)? the open modes ascend
  {
    int64_t strd[3] = {6, 2, 1};
    auto r = accessor_full(c, 12, nullptr, strd);
    std::vector<cd> rm(12);
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 3; ++b)
        for (int e = 0; e < 2; ++e) rm[a * 6 + b * 2 + e] = c.ref.v[a + 2 * b + 6 * e];
    check(maxdiff(r, rm) < 1e-9, "amplitudes are written with the caller's strides");
  }
  // projected modes: an amplitude, and a slice (open modes in ascending order)
  {
    int32_t pm[3] = {0, 1, 2};
    int64_t vals[3] = {1, 2, 1};
    cutensornetStateAccessor_t acc;
    IS(cutensornetCreateAccessor(h, c.st, 3, pm, nullptr, &acc), 0);
    WS ws;
    IS(cutensornetAccessorPrepare(h, acc, (size_t)1 << 30, ws.d, 0), 0);
    ws.alloc();
    Dev o(1);
    IS(cutensornetAccessorCompute(h, acc, vals, ws.d, o.p, nullptr, 0), 0);
    check(std::abs(o.get()[0] - c.ref.v[1 + 2 * 2 + 6 * 1]) < 1e-9, "a single amplitude");
    cutensornetDestroyAccessor(acc);
    int32_t pm2[2] = {2, 0};
    int64_t v2[2] = {1, 0};
    IS(cutensornetCreateAccessor(h, c.st, 2, pm2, nullptr, &acc), 0);
    WS w2;
    IS(cutensornetAccessorPrepare(h, acc, (size_t)1 << 30, w2.d, 0), 0);
    w2.alloc();
    Dev o2(3);
    IS(cutensornetAccessorCompute(h, acc, v2, w2.d, o2.p, nullptr, 0), 0);
    auto g = o2.get();
    double d = 0;
    for (int b = 0; b < 3; ++b) d = std::max(d, std::abs(g[b] - c.ref.v[0 + 2 * b + 6 * 1]));
    check(d < 1e-9, "a slice with modes 2 and 0 fixed");
    // changing a value needs no new preparation; another call
    int64_t v3[2] = {0, 1};
    IS(cutensornetAccessorCompute(h, acc, v3, w2.d, o2.p, nullptr, 0), 0);
    g = o2.get();
    d = 0;
    for (int b = 0; b < 3; ++b) d = std::max(d, std::abs(g[b] - c.ref.v[1 + 2 * b + 6 * 0]));
    check(d < 1e-9, "the same accessor with other values");
    cutensornetDestroyAccessor(acc);
  }
}

// ---- which operators cancel in a norm ----
//
// A unitary operator outside the light cone of what is computed cancels
// against its adjoint, so the flag's truth is not checked: these use
// operators whose flag is false, as measured on the card.

static void test_unitary_flags() {
  std::printf("-- unitary flags\n");
  struct Step {
    int mode;
    double scale;
    int flag;
  };
  auto norm_of = [&](std::vector<Step> steps) {
    Circuit c({2, 2});
    for (auto& s : steps) {
      std::vector<cd> G = {s.scale, 0, 0, s.scale};
      c.gate({s.mode}, G, s.flag);
    }
    cd nrm;
    accessor_full(c, 4, &nrm);
    return nrm.real();
  };
  check(std::abs(norm_of({{0, 3, 0}, {1, 2, 1}}) - 9) < 1e-9, "a flagged operator on another mode cancels (after)");
  check(std::abs(norm_of({{0, 2, 1}, {0, 3, 0}}) - 36) < 1e-9, "a flagged operator before a non-unitary one stays");
  check(std::abs(norm_of({{0, 3, 0}, {0, 2, 1}}) - 9) < 1e-9, "a flagged operator after a non-unitary one cancels");
  check(std::abs(norm_of({{1, 2, 1}, {0, 3, 0}}) - 9) < 1e-9, "a flagged operator on a mode outside the light cone cancels");
  check(std::abs(norm_of({{1, 2, 0}, {0, 3, 0}}) - 36) < 1e-9, "unflagged operators all count");
  check(std::abs(norm_of({{0, 2, 1}}) - 1) < 1e-9, "a state of unitary operators has norm 1");
  check(std::abs(norm_of({{0, 2, 1}, {0, 5, 1}, {1, 3, 0}}) - 9) < 1e-9, "a chain of flagged operators cancels");
}

// ---- explicit strides, diagonal and controlled operators ----

static void test_operator_kinds() {
  std::printf("-- diagonal, controlled and network operators\n");
  {  // strides: (q1 column, q0 column, q1 row, q0 row)
    Circuit c({2, 2});
    c.gate({0}, c.random_matrix({0}));
    c.gate({1}, c.random_matrix({1}));
    auto M = c.random_matrix({1, 0});
    std::vector<cd> Bt(16);
    for (int r = 0; r < 4; ++r)
      for (int cc = 0; cc < 4; ++cc) Bt[r + 4 * cc] = M[r * 4 + cc];
    Dev d(Bt);
    int32_t m10[2] = {1, 0};
    int64_t strd[4] = {4, 8, 1, 2}, id;
    IS(cutensornetStateApplyTensorOperator(h, c.st, 2, m10, d.p, strd, 1, 0, 0, &id), 0);
    c.ref.apply({1, 0}, M);
    cd nrm;
    check(maxdiff(accessor_full(c, 4, &nrm), c.ref.v) < 1e-9, "an operator with the caller's strides");
  }
  {  // diagonal
    Circuit c({2, 2, 2});
    for (int q = 0; q < 3; ++q) c.gate({q}, c.random_matrix({q}));
    auto dg = randv(4);
    Dev d(dg);
    int32_t m[2] = {2, 0};
    int64_t id = -1;
    IS(cutensornetStateApplyDiagonalTensorOperator(h, c.st, 2, m, d.p, nullptr, 1, 0, 0, &id), 0);
    check(id == 7, "a diagonal operator holds one id");
    std::vector<cd> M(16, 0);
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b) M[(a * 2 + b) * 4 + (a * 2 + b)] = dg[a * 2 + b];
    c.ref.apply({2, 0}, M);
    cd nrm;
    check(maxdiff(accessor_full(c, 8, &nrm), c.ref.v) < 1e-9, "a diagonal operator (first listed mode most significant)");
    // explicit strides: given in the reverse of the listed modes
    Circuit c2({2, 2, 2});
    for (int q = 0; q < 3; ++q) c2.gate({q}, c.random_matrix({q}));
    Ref r2 = c2.ref;
    Dev d2(dg);
    int64_t strd[2] = {2, 1};
    IS(cutensornetStateApplyDiagonalTensorOperator(h, c2.st, 2, m, d2.p, strd, 1, 0, 0, &id), 0);
    std::vector<cd> M2(16, 0);
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b) M2[(a * 2 + b) * 4 + (a * 2 + b)] = dg[2 * b + a];  // stride 2 on the last mode
    r2.apply({2, 0}, M2);
    check(maxdiff(accessor_full(c2, 8, &nrm), r2.v) < 1e-9, "a diagonal operator's strides run over the modes in reverse");
  }
  {  // controlled, one and two controls, one target
    for (int cfg = 0; cfg < 3; ++cfg) {
      Circuit c({2, 2, 2, 2});
      for (int q = 0; q < 4; ++q) c.gate({q}, c.random_matrix({q}));
      c.gate({0, 3}, c.random_matrix({0, 3}));
      std::vector<int> ctl, tgt = {0};
      std::vector<int64_t> cv;
      bool nullvals = false;
      if (cfg == 0) ctl = {2}, cv = {1};
      if (cfg == 1) ctl = {2}, cv = {0};
      if (cfg == 2) ctl = {3, 1}, cv = {1, 0};
      const bool adj = cfg == 1;
      auto G = randv(4);
      Dev d(G);
      std::vector<int32_t> c32(ctl.begin(), ctl.end()), t32(tgt.begin(), tgt.end());
      int64_t id = -1;
      IS(cutensornetStateApplyControlledTensorOperator(h, c.st, (int)ctl.size(), c32.data(), nullvals ? nullptr : cv.data(),
                                                       1, t32.data(), d.p, nullptr, 1, adj, 0, &id),
         0);
      std::vector<int> modes = ctl;
      modes.push_back(0);
      const size_t DT = 1 << modes.size();
      std::vector<cd> M(DT * DT, 0);
      auto Ge = adj ? dagger(G, 2) : G;
      for (size_t r = 0; r < DT; ++r)
        for (size_t cc = 0; cc < DT; ++cc) {
          const size_t rc = r >> 1, cc_c = cc >> 1, rt = r & 1, ct = cc & 1;
          if (rc != cc_c) continue;
          size_t want = 0;
          for (size_t j = 0; j < ctl.size(); ++j) want = want * 2 + cv[j];
          if (rc == want) M[r * DT + cc] = Ge[rt * 2 + ct];
          else if (r == cc) M[r * DT + cc] = 1;
        }
      c.ref.apply(modes, M);
      cd nrm;
      char what[96];
      std::snprintf(what, sizeof what, "a controlled operator (%zu control(s)%s)", ctl.size(), adj ? ", adjoint" : "");
      check(maxdiff(accessor_full(c, 16, &nrm), c.ref.v) < 1e-9, what);
    }
    Circuit c({2, 2, 2});
    int32_t ctl[1] = {1}, tgt2[2] = {2, 0};
    Dev d(randv(16));
    int64_t id;
    IS(cutensornetStateApplyControlledTensorOperator(h, c.st, 1, ctl, nullptr, 2, tgt2, d.p, nullptr, 1, 0, 0, &id), 7);
  }
  {  // a network operator applied to the state: a product of tensors in forward mode order, coefficient not applied
    Circuit c({2, 2, 2});
    for (int q = 0; q < 3; ++q) c.gate({q}, c.random_matrix({q}));
    c.gate({2, 0}, c.random_matrix({2, 0}));
    cutensornetNetworkOperator_t op;
    int64_t e3[3] = {2, 2, 2};
    IS(cutensornetCreateNetworkOperator(h, 3, e3, CUDA_C_64F, &op), 0);
    auto A = randv(4), B = randv(16);
    Dev dA(A), dB(B);
    int32_t nm[2] = {1, 2}, m0[1] = {0}, m1[2] = {2, 1};
    const int32_t* mp[2] = {m0, m1};
    const void* dat[2] = {dA.p, dB.p};
    int64_t oid = -1;
    IS(cutensornetStateApplyNetworkOperator(h, c.st, op, 1, 0, 0, &oid), 7);  // no component yet
    int64_t cid = -1;
    IS(cutensornetNetworkOperatorAppendProduct(h, op, make_cuDoubleComplex(0.5, 0.5), 2, nm, mp, nullptr, dat, &cid), 0);
    check(cid == 0, "the first component of a network operator is 0");
    IS(cutensornetStateApplyNetworkOperator(h, c.st, op, 1, 0, 0, &oid), 0);
    check(oid == 8, "a network operator holds one id per tensor");
    // the matrices: tensors' modes (column modes forward, then row modes forward)
    std::vector<cd> A2(4), B2(16);
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b) A2[b * 2 + a] = A[a + 2 * b];
    for (int a0 = 0; a0 < 2; ++a0)
      for (int a1 = 0; a1 < 2; ++a1)
        for (int b0 = 0; b0 < 2; ++b0)
          for (int b1 = 0; b1 < 2; ++b1) B2[(b0 * 2 + b1) * 4 + (a0 * 2 + a1)] = B[a0 + 2 * a1 + 4 * b0 + 8 * b1];
    c.ref.apply({0}, A2);
    c.ref.apply({2, 1}, B2);
    cd nrm;
    auto v = accessor_full(c, 8, &nrm);
    check(maxdiff(v, c.ref.v) < 1e-9, "a network operator applied to a state, without its coefficient");
    int64_t oid2 = -1, nid = -1;
    IS(cutensornetStateApplyNetworkOperator(h, c.st, op, 1, 1, 0, &oid2), 0);  // its adjoint
    check(oid2 == oid + 2, "the next operator follows the range");
    c.ref.apply({0}, dagger(A2, 2));
    c.ref.apply({2, 1}, dagger(B2, 4));
    check(maxdiff(accessor_full(c, 8, &nrm), c.ref.v) < 1e-9, "the adjoint of a network operator");
    nid = c.gate({1}, c.random_matrix({1}));
    check(nid == oid2 + 2, "the ids after it");
    // two components cannot be applied
    IS(cutensornetNetworkOperatorAppendProduct(h, op, make_cuDoubleComplex(1, 0), 1, nm, mp, nullptr, dat, &cid), 0);
    IS(cutensornetStateApplyNetworkOperator(h, c.st, op, 1, 0, 0, &oid), 7);
    cutensornetDestroyNetworkOperator(op);
  }
}

// ---- mixed states, marginals and their diagonals ----

static std::vector<cd> density(const std::vector<cd>& psi) {
  const size_t N = psi.size();
  std::vector<cd> rho(N * N);
  for (size_t b = 0; b < N; ++b)
    for (size_t k = 0; k < N; ++k) rho[k + N * b] = psi[k] * std::conj(psi[b]);
  return rho;
}

static void test_marginals() {
  std::printf("-- marginals and mixed states\n");
  for (int mixed = 0; mixed < 2; ++mixed) {
    Circuit c({2, 2, 2}, mixed != 0);
    for (int q = 0; q < 3; ++q) c.gate({q}, c.random_matrix({q}));
    c.gate({2, 0}, c.random_matrix({2, 0}));
    c.gate({1, 0}, c.random_matrix({1, 0}));
    const char* kind = mixed ? "mixed" : "pure";
    char what[128];
    if (mixed) {
      cd nrm;
      auto rho = accessor_full(c, 64, &nrm);
      std::snprintf(what, sizeof what, "%s: the density matrix (kets fastest, then bras), and its trace as the norm", kind);
      double tr = 0;
      for (auto& x : c.ref.v) tr += std::norm(x);
      check(maxdiff(rho, density(c.ref.v)) < 1e-9 && std::abs(nrm.real() - tr) < 1e-9 * tr, what);
      // projected ket 0 and bra 1: open kets 1, 2 then bras 0, 2
      int32_t pm[2] = {0, 4};
      int64_t vals[2] = {1, 0};
      cutensornetStateAccessor_t acc;
      IS(cutensornetCreateAccessor(h, c.st, 2, pm, nullptr, &acc), 0);
      WS ws;
      IS(cutensornetAccessorPrepare(h, acc, (size_t)1 << 30, ws.d, 0), 0);
      ws.alloc();
      Dev o(16);
      IS(cutensornetAccessorCompute(h, acc, vals, ws.d, o.p, nullptr, 0), 0);
      auto g = o.get();
      double d = 0;
      for (int k1 = 0; k1 < 2; ++k1)
        for (int k2 = 0; k2 < 2; ++k2)
          for (int b0 = 0; b0 < 2; ++b0)
            for (int b2 = 0; b2 < 2; ++b2) {
              const size_t ket = 1 + 2 * k1 + 4 * k2, bra = b0 + 0 + 4 * b2;
              d = std::max(d, std::abs(c.ref.v[ket] * std::conj(c.ref.v[bra]) - g[k1 + 2 * k2 + 4 * b0 + 8 * b2]));
            }
      check(d < 1e-9, "mixed: a slice of the density matrix with a ket and a bra mode fixed");
      cutensornetDestroyAccessor(acc);
    }
    // marginal over (2, 0), mode 1 fixed to 1
    {
      int32_t mm[2] = {2, 0}, pj[1] = {1};
      cutensornetStateMarginal_t mg;
      IS(cutensornetCreateMarginal(h, c.st, 2, mm, 1, pj, nullptr, &mg), 0);
      WS ws;
      IS(cutensornetMarginalPrepare(h, mg, (size_t)1 << 30, ws.d, 0), 0);
      ws.alloc();
      Dev out(16);
      int64_t pv[1] = {1};
      IS(cutensornetMarginalCompute(h, mg, pv, ws.d, out.p, 0), 0);
      auto g = out.get();
      std::vector<cd> hyp(16);
      for (int k2 = 0; k2 < 2; ++k2)
        for (int k0 = 0; k0 < 2; ++k0)
          for (int b2 = 0; b2 < 2; ++b2)
            for (int b0 = 0; b0 < 2; ++b0)
              hyp[k2 + 2 * k0 + 4 * b2 + 8 * b0] =
                  c.ref.v[k0 + 2 + 4 * k2] * std::conj(c.ref.v[b0 + 2 + 4 * b2]);
      std::snprintf(what, sizeof what, "%s: a marginal with a projected mode (ket modes, then bra modes)", kind);
      check(maxdiff(g, hyp) < 1e-9, what);
      int kind_v = -1;
      IS(cutensornetMarginalGetInfo(h, mg, CUTENSORNET_MARGINAL_INFO_KIND, &kind_v, 4), 0);
      check(kind_v == CUTENSORNET_MARGINAL_KIND_FULL, "a marginal's kind");
      double fl = -1;
      IS(cutensornetMarginalGetInfo(h, mg, CUTENSORNET_MARGINAL_INFO_FLOPS, &fl, 8), 0);
      check(fl > 0, "a prepared marginal reports its FLOPs");
      int64_t bad[1] = {2};
      IS(cutensornetMarginalCompute(h, mg, bad, ws.d, out.p, 0), 7);
      cutensornetDestroyMarginal(mg);
    }
    // marginal of one mode, everything else traced
    {
      int32_t mm[1] = {1};
      cutensornetStateMarginal_t mg;
      IS(cutensornetCreateMarginal(h, c.st, 1, mm, 0, nullptr, nullptr, &mg), 0);
      WS ws;
      IS(cutensornetMarginalPrepare(h, mg, (size_t)1 << 30, ws.d, 0), 0);
      ws.alloc();
      Dev out(4);
      IS(cutensornetMarginalCompute(h, mg, nullptr, ws.d, out.p, 0), 0);
      auto g = out.get();
      std::vector<cd> hyp(4, 0);
      for (int k = 0; k < 2; ++k)
        for (int b = 0; b < 2; ++b)
          for (int r = 0; r < 4; ++r)
            hyp[k + 2 * b] += c.ref.v[(r & 1) + 2 * k + 4 * (r >> 1)] * std::conj(c.ref.v[(r & 1) + 2 * b + 4 * (r >> 1)]);
      std::snprintf(what, sizeof what, "%s: the reduced density matrix of one mode", kind);
      check(maxdiff(g, hyp) < 1e-9, what);
      cutensornetDestroyMarginal(mg);
    }
    // the diagonal of the marginal
    {
      int32_t mm[2] = {2, 0}, pj[1] = {1};
      cutensornetStateMarginal_t mg;
      IS(cutensornetCreateMarginalDiagonal(h, c.st, 2, mm, 1, pj, nullptr, &mg), 0);
      WS ws;
      IS(cutensornetMarginalPrepare(h, mg, (size_t)1 << 30, ws.d, 0), 0);
      ws.alloc();
      Dev out(4);
      int64_t pv[1] = {1};
      IS(cutensornetMarginalCompute(h, mg, pv, ws.d, out.p, 0), 0);
      auto g = out.get();
      double d = 0;
      for (int idx = 0; idx < 4; ++idx) d = std::max(d, std::abs(g[idx] - std::norm(c.ref.v[(idx >> 1) + 2 + 4 * (idx & 1)])));
      std::snprintf(what, sizeof what, "%s: the diagonal of a marginal (real values in the state's type)", kind);
      check(d < 1e-9, what);
      int kind_v = -1;
      IS(cutensornetMarginalGetInfo(h, mg, CUTENSORNET_MARGINAL_INFO_KIND, &kind_v, 4), 0);
      check(kind_v == CUTENSORNET_MARGINAL_KIND_DIAGONAL, "a diagonal marginal's kind");
      cutensornetDestroyMarginal(mg);
    }
  }
}

// ---- expectation values ----

static void test_expectation() {
  std::printf("-- expectation values\n");
  for (int mixed = 0; mixed < 2; ++mixed) {
    Circuit c({2, 2, 2}, mixed != 0);
    for (int q = 0; q < 3; ++q) c.gate({q}, c.random_matrix({q}));
    c.gate({2, 0}, c.random_matrix({2, 0}));
    c.gate({1, 0}, c.random_matrix({1, 0}));
    cutensornetNetworkOperator_t op;
    int64_t e3[3] = {2, 2, 2};
    IS(cutensornetCreateNetworkOperator(h, 3, e3, CUDA_C_64F, &op), 0);
    auto A = randv(4), B = randv(16), C = randv(4);
    Dev dA(A), dB(B), dC(C);
    int32_t nm[2] = {1, 2}, m0[1] = {0}, m1[2] = {2, 1};
    const int32_t* mp[2] = {m0, m1};
    const void* dat[2] = {dA.p, dB.p};
    int64_t cid = -1;
    IS(cutensornetNetworkOperatorAppendProduct(h, op, make_cuDoubleComplex(0.7, -0.3), 2, nm, mp, nullptr, dat, &cid), 0);
    int32_t nm2[1] = {1}, ms[1] = {1};
    const int32_t* mp2[1] = {ms};
    const void* dat2[1] = {dC.p};
    IS(cutensornetNetworkOperatorAppendProduct(h, op, make_cuDoubleComplex(-0.4, 0.2), 1, nm2, mp2, nullptr, dat2, &cid), 0);
    check(cid == 1, "components are numbered in order");
    cutensornetStateExpectation_t ex;
    IS(cutensornetCreateExpectation(h, c.st, op, &ex), 0);
    WS ws;
    IS(cutensornetExpectationPrepare(h, ex, (size_t)1 << 30, ws.d, 0), 0);
    ws.alloc();
    cd val = 0, nrm = 0;
    IS(cutensornetExpectationCompute(h, ex, ws.d, &val, &nrm, 0), 0);
    cudaDeviceSynchronize();
    // reference: sum_c coef_c <psi| O_c |psi>, a tensor's first half of modes contracting with the ket
    std::vector<cd> A2(4), B2(16), C2(4);
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b) A2[b * 2 + a] = A[a + 2 * b], C2[b * 2 + a] = C[a + 2 * b];
    for (int a0 = 0; a0 < 2; ++a0)
      for (int a1 = 0; a1 < 2; ++a1)
        for (int b0 = 0; b0 < 2; ++b0)
          for (int b1 = 0; b1 < 2; ++b1) B2[(b0 * 2 + b1) * 4 + (a0 * 2 + a1)] = B[a0 + 2 * a1 + 4 * b0 + 8 * b1];
    auto dotc = [](const std::vector<cd>& a, const std::vector<cd>& b) {
      cd s = 0;
      for (size_t i = 0; i < a.size(); ++i) s += std::conj(a[i]) * b[i];
      return s;
    };
    Ref r1 = c.ref;
    r1.apply({0}, A2);
    r1.apply({2, 1}, B2);
    Ref r2 = c.ref;
    r2.apply({1}, C2);
    const cd E = cd(0.7, -0.3) * dotc(c.ref.v, r1.v) + cd(-0.4, 0.2) * dotc(c.ref.v, r2.v);
    char what[96];
    std::snprintf(what, sizeof what, "%s: an expectation value of a sum of products", mixed ? "mixed" : "pure");
    check(std::abs(val - E) < 1e-9, what);
    double nn = 0;
    for (auto& x : c.ref.v) nn += std::norm(x);
    check(std::abs(nrm.real() - nn) < 1e-9 * nn, "and the squared norm of the state");
    double fl = -1;
    IS(cutensornetExpectationGetInfo(h, ex, CUTENSORNET_EXPECTATION_INFO_FLOPS, &fl, 8), 0);
    check(fl > 0, "a prepared expectation reports its FLOPs");
    IS(cutensornetExpectationCompute(h, ex, ws.d, &val, nullptr, 0), 0);
    check(std::abs(val - E) < 1e-9, "the norm is optional");
    cutensornetDestroyExpectation(ex);
    cutensornetDestroyNetworkOperator(op);
    // a matrix product operator on modes (2, 0, 1), bond extent 3
    cutensornetNetworkOperator_t mpo;
    IS(cutensornetCreateNetworkOperator(h, 3, e3, CUDA_C_64F, &mpo), 0);
    int32_t sm[3] = {2, 0, 1};
    int64_t ex0[3] = {2, 3, 2}, ex1[4] = {3, 2, 3, 2}, ex2[3] = {3, 2, 2};
    const int64_t* exts[3] = {ex0, ex1, ex2};
    auto T0 = randv(12), T1 = randv(36), T2 = randv(12);
    Dev d0(T0), d1(T1), d2(T2);
    const void* td[3] = {d0.p, d1.p, d2.p};
    IS(cutensornetNetworkOperatorAppendMPO(h, mpo, make_cuDoubleComplex(1, 0), 3, sm, exts, nullptr, td,
                                           CUTENSORNET_BOUNDARY_CONDITION_OPEN, &cid), 0);
    cutensornetStateExpectation_t ex2h;
    IS(cutensornetCreateExpectation(h, c.st, mpo, &ex2h), 0);
    WS w2;
    IS(cutensornetExpectationPrepare(h, ex2h, (size_t)1 << 30, w2.d, 0), 0);
    w2.alloc();
    cd val2 = 0;
    IS(cutensornetExpectationCompute(h, ex2h, w2.d, &val2, nullptr, 0), 0);
    // O[bra2 bra0 bra1][ket2 ket0 ket1] = sum T0[ket2, a, bra2] T1[a, ket0, b, bra0] T2[b, ket1, bra1]
    std::vector<cd> M(64, 0);
    for (int k2 = 0; k2 < 2; ++k2)
      for (int k0 = 0; k0 < 2; ++k0)
        for (int k1 = 0; k1 < 2; ++k1)
          for (int b2 = 0; b2 < 2; ++b2)
            for (int b0 = 0; b0 < 2; ++b0)
              for (int b1 = 0; b1 < 2; ++b1) {
                cd s = 0;
                for (int a = 0; a < 3; ++a)
                  for (int b = 0; b < 3; ++b) s += T0[k2 + 2 * a + 6 * b2] * T1[a + 3 * k0 + 6 * b + 18 * b0] * T2[b + 3 * k1 + 6 * b1];
                M[((b2 * 2 + b0) * 2 + b1) * 8 + ((k2 * 2 + k0) * 2 + k1)] = s;
              }
    Ref rm = c.ref;
    rm.apply({2, 0, 1}, M);
    std::snprintf(what, sizeof what, "%s: an expectation value of a matrix product operator", mixed ? "mixed" : "pure");
    check(std::abs(val2 - dotc(c.ref.v, rm.v)) < 1e-9, what);
    // appending to an operator invalidates the expectation made from it
    cutensornetDestroyExpectation(ex2h);
    cutensornetDestroyNetworkOperator(mpo);
  }
  // real data: the value is real, the coefficient's imaginary part drops
  {
    Circuit c({2, 2}, false, CUDA_R_64F);
    for (int q = 0; q < 2; ++q) c.gate({q}, c.random_matrix({q}, true));
    cutensornetNetworkOperator_t op;
    int64_t e2[2] = {2, 2};
    IS(cutensornetCreateNetworkOperator(h, 2, e2, CUDA_R_64F, &op), 0);
    auto A = randv(4, false);
    Dev dA(A, CUDA_R_64F);
    int32_t nm[1] = {1}, m0[1] = {0};
    const int32_t* mp[1] = {m0};
    const void* dat[1] = {dA.p};
    int64_t cid;
    IS(cutensornetNetworkOperatorAppendProduct(h, op, make_cuDoubleComplex(2, 1), 1, nm, mp, nullptr, dat, &cid), 0);
    cutensornetStateExpectation_t ex;
    IS(cutensornetCreateExpectation(h, c.st, op, &ex), 0);
    WS ws;
    IS(cutensornetExpectationPrepare(h, ex, (size_t)1 << 30, ws.d, 0), 0);
    ws.alloc();
    double val = -99, nrm = -99;
    IS(cutensornetExpectationCompute(h, ex, ws.d, &val, &nrm, 0), 0);
    std::vector<cd> A2(4);
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b) A2[b * 2 + a] = A[a + 2 * b];
    Ref r2 = c.ref;
    r2.apply({0}, A2);
    cd s = 0;
    double nn = 0;
    for (size_t i = 0; i < 4; ++i) s += std::conj(c.ref.v[i]) * r2.v[i], nn += std::norm(c.ref.v[i]);
    check(std::abs(val - (s * cd(2, 1)).real()) < 1e-9 && std::abs(nrm - nn) < 1e-9, "a real state: a real value, the coefficient's real part");
    cutensornetDestroyExpectation(ex);
    cutensornetDestroyNetworkOperator(op);
  }
}

// ---- channels ----

static void test_channels() {
  std::printf("-- channels\n");
  // a mixed state's channel holds five ids and is named by the fifth
  {
    Circuit c({2, 2}, true);
    std::vector<cd> I2 = {1, 0, 0, 1}, X = {0, 1, 1, 0}, Z = {1, 0, 0, -1};
    Dev dI(I2), dX(X), dZ(Z);
    void* td[3] = {dI.p, dX.p, dZ.p};
    double pr[3] = {0.5, 0.3, 0.2};
    int32_t m[1] = {1};
    int64_t cid = -1;
    const int64_t before = c.gate({0}, c.random_matrix({0}), 1);
    IS(cutensornetStateApplyUnitaryChannel(h, c.st, 1, m, 3, td, nullptr, pr, &cid), 0);
    check(cid == before + 2 + 4, "a mixed state's channel holds five ids and is named by the fifth");
    const int64_t nid = c.gate({1}, c.random_matrix({1}), 1);
    check(nid == cid + 1, "the id after a channel");
  }
  // exact reference: |00> through gates, a channel, then compare the whole density matrix
  {
    std::vector<cd> I2 = {1, 0, 0, 1}, X = {0, 1, 1, 0};
    Circuit c({2, 2}, true);
    auto H1 = randv(4), H2 = randv(4);
    Dev dH1(H1), dH2(H2), dI(I2), dX(X);
    int32_t m0[1] = {0}, m1[1] = {1};
    int64_t id;
    IS(cutensornetStateApplyTensorOperator(h, c.st, 1, m0, dH1.p, nullptr, 1, 0, 0, &id), 0);
    IS(cutensornetStateApplyTensorOperator(h, c.st, 1, m1, dH2.p, nullptr, 1, 0, 0, &id), 0);
    void* td[2] = {dI.p, dX.p};
    double pr[2] = {0.7, 0.3};
    IS(cutensornetStateApplyUnitaryChannel(h, c.st, 1, m1, 2, td, nullptr, pr, &id), 0);
    Ref a({2, 2});
    a.apply({0}, H1);
    a.apply({1}, H2);
    Ref b = a;
    b.apply({1}, X);
    const size_t N = 4;
    std::vector<cd> want(N * N);
    for (size_t bi = 0; bi < N; ++bi)
      for (size_t k = 0; k < N; ++k)
        want[k + N * bi] = 0.7 * a.v[k] * std::conj(a.v[bi]) + 0.3 * b.v[k] * std::conj(b.v[bi]);
    cd nrm;
    check(maxdiff(accessor_full(c, 16, &nrm), want) < 1e-9, "a mixed state through a unitary channel: sum p U rho U^H");
    // a general channel (Kraus operators): sum K rho K^H
    Circuit g({2, 2}, true);
    int64_t gid;
    IS(cutensornetStateApplyTensorOperator(h, g.st, 1, m0, dH1.p, nullptr, 1, 0, 0, &gid), 0);
    std::vector<cd> K0 = {1, 0, 0, std::sqrt(0.5)}, K1 = {0, std::sqrt(0.5), 0, 0};
    Dev dK0(K0), dK1(K1);
    void* kd[2] = {dK0.p, dK1.p};
    IS(cutensornetStateApplyGeneralChannel(h, g.st, 1, m0, 2, kd, nullptr, &gid), 0);
    Ref s0({2, 2});
    s0.apply({0}, H1);
    Ref k0 = s0, k1 = s0;
    k0.apply({0}, K0);
    k1.apply({0}, K1);
    std::vector<cd> want2(N * N);
    for (size_t bi = 0; bi < N; ++bi)
      for (size_t k = 0; k < N; ++k) want2[k + N * bi] = k0.v[k] * std::conj(k0.v[bi]) + k1.v[k] * std::conj(k1.v[bi]);
    check(maxdiff(accessor_full(g, 16, &nrm), want2) < 1e-9, "a mixed state through a general channel: sum K rho K^H");
  }
  // a pure state: a unitary channel is resolved at random at every compute; a general one cannot be contracted
  {
    Circuit c({2, 2});
    std::vector<cd> I2 = {1, 0, 0, 1}, X = {0, 1, 1, 0};
    Dev dI(I2), dX(X);
    void* td[2] = {dI.p, dX.p};
    double pr[2] = {0.5, 0.5};
    int32_t m0[1] = {0};
    int64_t id;
    IS(cutensornetStateApplyUnitaryChannel(h, c.st, 1, m0, 2, td, nullptr, pr, &id), 0);
    check(id == 3, "a pure state's channel holds one id");
    cutensornetStateAccessor_t acc;
    IS(cutensornetCreateAccessor(h, c.st, 0, nullptr, nullptr, &acc), 0);
    WS ws;
    IS(cutensornetAccessorPrepare(h, acc, (size_t)1 << 30, ws.d, 0), 0);
    ws.alloc();
    int ones = 0, bad = 0;
    for (int r = 0; r < 24; ++r) {
      Dev o(4);
      IS(cutensornetAccessorCompute(h, acc, nullptr, ws.d, o.p, nullptr, 0), 0);
      auto v = o.get();
      const bool zero = std::abs(v[0] - 1.0) < 1e-12 && std::abs(v[1]) < 1e-12;
      const bool one = std::abs(v[1] - 1.0) < 1e-12 && std::abs(v[0]) < 1e-12;
      ones += one;
      bad += !(zero || one);
    }
    check(bad == 0 && ones > 2 && ones < 22, "a unitary channel on a pure state picks one operator per compute");
    cutensornetDestroyAccessor(acc);
    Circuit g({2, 2});
    std::vector<cd> K0 = {1, 0, 0, 1};
    Dev dK(K0);
    void* kd[1] = {dK.p};
    IS(cutensornetStateApplyGeneralChannel(h, g.st, 1, m0, 1, kd, nullptr, &id), 0);
    cutensornetStateAccessor_t a2;
    IS(cutensornetCreateAccessor(h, g.st, 0, nullptr, nullptr, &a2), 0);
    WS w2;
    IS(cutensornetAccessorPrepare(h, a2, (size_t)1 << 30, w2.d, 0), 15);
    cutensornetDestroyAccessor(a2);
  }
}

// ---- sampling ----

static void test_sampler() {
  std::printf("-- sampling\n");
  Circuit c({2, 3, 2});
  for (int q = 0; q < 3; ++q) c.gate({q}, c.random_matrix({q}));
  c.gate({2, 1}, c.random_matrix({2, 1}));
  c.gate({0, 1}, c.random_matrix({0, 1}));
  double nn = 0;
  for (auto& x : c.ref.v) nn += std::norm(x);
  auto run = [&](std::vector<int32_t> modes, int64_t shots, int seed, std::vector<int64_t>& out) {
    cutensornetStateSampler_t sp;
    IS(cutensornetCreateSampler(h, c.st, modes.empty() ? 3 : (int)modes.size(), modes.empty() ? nullptr : modes.data(), &sp), 0);
    if (seed >= 0) {
      int32_t sd = seed;
      IS(cutensornetSamplerConfigure(h, sp, CUTENSORNET_SAMPLER_CONFIG_DETERMINISTIC, &sd, 4), 0);
    }
    WS ws;
    IS(cutensornetSamplerPrepare(h, sp, (size_t)1 << 30, ws.d, 0), 0);
    ws.alloc();
    const size_t nm = modes.empty() ? 3 : modes.size();
    out.assign((size_t)shots * nm, -7);
    IS(cutensornetSamplerSample(h, sp, shots, ws.d, out.data(), 0), 0);
    double fl = -1;
    IS(cutensornetSamplerGetInfo(h, sp, CUTENSORNET_SAMPLER_INFO_FLOPS, &fl, 8), 0);
    check(fl > 0, "a prepared sampler reports its FLOPs");
    cutensornetDestroySampler(sp);
  };
  std::vector<int64_t> s1;
  const int64_t N = 20000;
  run({}, N, -1, s1);
  {
    std::map<int, int> cnt;
    bool inrange = true;
    for (int64_t s = 0; s < N; ++s) {
      const int64_t a = s1[3 * s], b = s1[3 * s + 1], e = s1[3 * s + 2];
      inrange = inrange && a >= 0 && a < 2 && b >= 0 && b < 3 && e >= 0 && e < 2;
      cnt[(int)(a + 2 * b + 6 * e)]++;
    }
    double chi = 0;
    for (int i = 0; i < 12; ++i) {
      const double e = N * std::norm(c.ref.v[i]) / nn;
      chi += (cnt[i] - e) * (cnt[i] - e) / e;
    }
    check(inrange && chi < 45, "samples follow the state's probabilities (chi-square, 11 degrees of freedom)");
  }
  std::vector<int64_t> s2;
  run({2, 0}, N, -1, s2);
  {
    std::map<int, int> cnt;
    for (int64_t s = 0; s < N; ++s) cnt[(int)(s2[2 * s] * 2 + s2[2 * s + 1])]++;
    double chi = 0;
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b) {
        double p = 0;
        for (int m1 = 0; m1 < 3; ++m1) p += std::norm(c.ref.v[b + 2 * m1 + 6 * a]);
        p /= nn;
        const double e = N * p;
        chi += (cnt[a * 2 + b] - e) * (cnt[a * 2 + b] - e) / e;
      }
    check(chi < 25, "samples of modes (2, 0) are in that order, from the marginal distribution (3 degrees of freedom)");
  }
  std::vector<int64_t> a, b, d;
  run({}, 40, 42, a);
  run({}, 40, 42, b);
  run({}, 40, 43, d);
  check(a == b, "a positive seed makes two samplers agree");
  check(a != d, "another seed gives other samples");
  // the generator advances, a re-configured seed starts again
  {
    cutensornetStateSampler_t sp;
    IS(cutensornetCreateSampler(h, c.st, 3, nullptr, &sp), 0);
    int32_t sd = 5;
    IS(cutensornetSamplerConfigure(h, sp, CUTENSORNET_SAMPLER_CONFIG_DETERMINISTIC, &sd, 4), 0);
    WS ws;
    IS(cutensornetSamplerPrepare(h, sp, (size_t)1 << 30, ws.d, 0), 0);
    ws.alloc();
    std::vector<int64_t> x(120), y(120), z(120);
    IS(cutensornetSamplerSample(h, sp, 40, ws.d, x.data(), 0), 0);
    IS(cutensornetSamplerSample(h, sp, 40, ws.d, y.data(), 0), 0);
    check(x != y, "a sampler's next call continues its sequence");
    IS(cutensornetSamplerConfigure(h, sp, CUTENSORNET_SAMPLER_CONFIG_DETERMINISTIC, &sd, 4), 0);
    IS(cutensornetSamplerSample(h, sp, 40, ws.d, z.data(), 0), 0);
    check(x == z, "configuring the seed again restarts the sequence");
    int32_t got = -1;
    IS(cutensornetSamplerGetInfo(h, sp, CUTENSORNET_SAMPLER_CONFIG_DETERMINISTIC, &got, 4), 0);
    check(got == 5, "the seed can be read back");
    IS(cutensornetSamplerSample(h, sp, 0, ws.d, x.data(), 0), 7);
    IS(cutensornetSamplerSample(h, sp, -1, ws.d, x.data(), 0), 7);
    int32_t neg = -1;
    IS(cutensornetSamplerConfigure(h, sp, CUTENSORNET_SAMPLER_CONFIG_DETERMINISTIC, &neg, 4), 7);
    cutensornetDestroySampler(sp);
  }
}

// ---- computing the state itself, and the library's bookkeeping ----

static void test_state_compute() {
  std::printf("-- the state tensor\n");
  Circuit c({2, 3, 2});
  int32_t nt = -1, nmodes[4] = {0};
  IS(cutensornetGetOutputStateDetails(h, c.st, &nt, nmodes, nullptr, nullptr), 0);
  check(nt == 1 && nmodes[0] == 3, "an unfactorized state has one tensor of one mode per state mode");
  for (int q = 0; q < 3; ++q) c.gate({q}, c.random_matrix({q}));
  c.gate({2, 1}, c.random_matrix({2, 1}));
  double fl = -1;
  IS(cutensornetStateGetInfo(h, c.st, CUTENSORNET_STATE_INFO_FLOPS, &fl, 8), 0);
  check(fl == 0, "no FLOP count before the state is prepared");
  WS ws;
  Dev out(12);
  void* ptrs[1] = {out.p};
  IS(cutensornetStateCompute(h, c.st, ws.d, nullptr, nullptr, ptrs, 0), 7);  // not prepared (refused before the scratch is looked at)
  IS(cutensornetStatePrepare(h, c.st, (size_t)1 << 30, ws.d, 0), 0);
  ws.alloc();
  IS(cutensornetStateGetInfo(h, c.st, CUTENSORNET_STATE_INFO_FLOPS, &fl, 8), 0);
  check(fl > 0, "the FLOP count after preparation");
  int64_t eo[3] = {0}, so[3] = {0};
  int64_t* ep[1] = {eo};
  int64_t* sp[1] = {so};
  IS(cutensornetStateCompute(h, c.st, ws.d, ep, sp, ptrs, 0), 0);
  check(maxdiff(out.get(), c.ref.v) < 1e-9, "the state tensor");
  check(eo[0] == 2 && eo[1] == 3 && eo[2] == 2 && so[0] == 1 && so[1] == 2 && so[2] == 6, "its extents and strides");
  int64_t e2[3] = {0}, s2[3] = {0};
  int64_t* e2p[1] = {e2};
  int64_t* s2p[1] = {s2};
  IS(cutensornetGetOutputStateDetails(h, c.st, &nt, nmodes, e2p, s2p), 0);
  check(e2[1] == 3 && s2[2] == 6, "the output state's details after computing it");
  // a new operator makes the preparation outdated, and prepare resets the workspace's memory
  c.gate({1}, c.random_matrix({1}));
  IS(cutensornetStateCompute(h, c.st, ws.d, nullptr, nullptr, ptrs, 0), 7);
  IS(cutensornetStatePrepare(h, c.st, (size_t)1 << 30, ws.d, 0), 0);
  IS(cutensornetStateCompute(h, c.st, ws.d, nullptr, nullptr, ptrs, 0), 15);
  ws.alloc();
  IS(cutensornetStateCompute(h, c.st, ws.d, nullptr, nullptr, ptrs, 0), 0);
  check(maxdiff(out.get(), c.ref.v) < 1e-9, "the state after another operator");
  void* none[1] = {nullptr};
  IS(cutensornetStateCompute(h, c.st, ws.d, nullptr, nullptr, none, 0), 7);
  // mixed: the density matrix tensor
  Circuit m({2, 2}, true);
  m.gate({0}, m.random_matrix({0}));
  m.gate({1, 0}, m.random_matrix({1, 0}));
  WS w2;
  Dev o2(16);
  void* p2[1] = {o2.p};
  IS(cutensornetStatePrepare(h, m.st, (size_t)1 << 30, w2.d, 0), 0);
  w2.alloc();
  IS(cutensornetStateCompute(h, m.st, w2.d, nullptr, nullptr, p2, 0), 0);
  check(maxdiff(o2.get(), density(m.ref.v)) < 1e-9, "a mixed state's density matrix tensor");
  // integer ids of a mixed state: 2n + 1 reserved, two per operator
  Circuit m2({2, 2}, true);
  const int64_t i0 = m2.gate({0}, m2.random_matrix({0}));
  const int64_t i1 = m2.gate({1}, m2.random_matrix({1}));
  check(i0 == 5 && i1 == 7, "a mixed state's operators hold two ids each, after 2n + 1");
  // updating a mutable operator
  {
    Circuit u({2, 2});
    auto G1 = randv(4), G2 = randv(4);
    Dev d1(G1), d2(G2);
    int32_t m0[1] = {0};
    int64_t id;
    IS(cutensornetStateApplyTensorOperator(h, u.st, 1, m0, d1.p, nullptr, 0, 0, 0, &id), 0);  // mutable
    cutensornetStateAccessor_t acc;
    IS(cutensornetCreateAccessor(h, u.st, 0, nullptr, nullptr, &acc), 0);
    WS wa;
    IS(cutensornetAccessorPrepare(h, acc, (size_t)1 << 30, wa.d, 0), 0);
    wa.alloc();
    Dev o(4);
    IS(cutensornetAccessorCompute(h, acc, nullptr, wa.d, o.p, nullptr, 0), 0);
    Ref r({2, 2});
    r.apply({0}, G1);
    check(maxdiff(o.get(), r.v) < 1e-9, "before the update");
    IS(cutensornetStateUpdateTensorOperator(h, u.st, id, d2.p, 0), 0);
    IS(cutensornetAccessorCompute(h, acc, nullptr, wa.d, o.p, nullptr, 0), 0);
    Ref r2({2, 2});
    r2.apply({0}, G2);
    check(maxdiff(o.get(), r2.v) < 1e-9, "after updating the operator's data, with no new preparation");
    IS(cutensornetStateUpdateTensorOperator(h, u.st, 9999, d2.p, 0), 7);
    IS(cutensornetStateUpdateTensorOperator(h, u.st, 0, d2.p, 0), 7);  // a vacuum tensor
    // an immutable operator cannot be updated
    int64_t id2;
    int32_t m1[1] = {1};
    IS(cutensornetStateApplyTensorOperator(h, u.st, 1, m1, d1.p, nullptr, 1, 0, 0, &id2), 0);
    IS(cutensornetStateUpdateTensorOperator(h, u.st, id2, d2.p, 0), 7);
    // a new operator outdates the accessor
    IS(cutensornetAccessorCompute(h, acc, nullptr, wa.d, o.p, nullptr, 0), 7);
    IS(cutensornetAccessorPrepare(h, acc, (size_t)1 << 30, wa.d, 0), 7);
    cutensornetDestroyAccessor(acc);
  }
}

// ---- matrix product states ----

struct MpsOut {
  std::vector<std::vector<int64_t>> ext, str;
  std::vector<std::vector<cd>> data;
};

// The dense state (mode 0 fastest) an MPS of open boundary stands for.
static std::vector<cd> mps_dense(const MpsOut& m, int n) {
  std::vector<cd> out((size_t)1 << n);
  for (size_t idx = 0; idx < out.size(); ++idx) {
    std::vector<cd> v = {1};
    for (int i = 0; i < n; ++i) {
      const int phys = (int)((idx >> i) & 1);
      const size_t L = i == 0 ? 1 : (size_t)m.ext[i][0], R = i == n - 1 ? 1 : (size_t)m.ext[i].back();
      std::vector<cd> nv(R, 0);
      for (size_t l = 0; l < L; ++l)
        for (size_t r = 0; r < R; ++r) {
          int64_t off;
          if (i == 0) off = phys * m.str[i][0] + r * m.str[i][1];
          else if (i == n - 1) off = l * m.str[i][0] + phys * m.str[i][1];
          else off = l * m.str[i][0] + phys * m.str[i][1] + r * m.str[i][2];
          nv[r] += v[l] * m.data[i][off];
        }
      v = nv;
    }
    out[idx] = v[0];
  }
  return out;
}

struct MpsSetup {
  int n = 5;
  std::vector<int64_t> maxb = {2, 4, 4, 2};
  bool row_major = false;
  int center = -1;
  double abs_cut = 0, rel_cut = 0, dw_cut = 0;
  int norm = CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE;
};

// Builds a circuit of gates on a pure state, finalizes and computes an MPS.
// `gates` lists the modes of each operator (random matrices, seeded).
static bool run_mps(const std::vector<std::vector<int>>& gates, const MpsSetup& su, Circuit& c, MpsOut& out,
                    std::vector<int64_t>* ext_flat = nullptr) {
  const int n = su.n;
  for (auto& modes : gates) c.gate(modes, c.random_matrix(modes));
  std::vector<std::vector<int64_t>> ex((size_t)n), st((size_t)n);
  for (int i = 0; i < n; ++i) {
    if (i == 0) ex[i] = {2, su.maxb[0]};
    else if (i == n - 1) ex[i] = {su.maxb[i - 1], 2};
    else ex[i] = {su.maxb[i - 1], 2, su.maxb[i]};
    if (su.row_major) {
      int64_t s = 1;
      st[i].resize(ex[i].size());
      for (int j = (int)ex[i].size() - 1; j >= 0; --j) st[i][j] = s, s *= ex[i][j];
    }
  }
  std::vector<const int64_t*> ep, sp;
  for (int i = 0; i < n; ++i) ep.push_back(ex[i].data());
  if (su.row_major)
    for (int i = 0; i < n; ++i) sp.push_back(st[i].data());
  if (su.center >= 0) {
    int32_t cc = su.center;
    if (cutensornetStateConfigure(h, c.st, CUTENSORNET_STATE_CONFIG_MPS_CANONICAL_CENTER, &cc, 4)) return false;
  }
  if (su.abs_cut > 0 && cutensornetStateConfigure(h, c.st, CUTENSORNET_STATE_CONFIG_MPS_SVD_ABS_CUTOFF, &su.abs_cut, 8)) return false;
  if (su.rel_cut > 0 && cutensornetStateConfigure(h, c.st, CUTENSORNET_STATE_CONFIG_MPS_SVD_REL_CUTOFF, &su.rel_cut, 8)) return false;
  if (su.dw_cut > 0 &&
      cutensornetStateConfigure(h, c.st, CUTENSORNET_STATE_CONFIG_MPS_SVD_DISCARDED_WEIGHT_CUTOFF, &su.dw_cut, 8))
    return false;
  if (su.norm != 0) {
    int32_t nm = su.norm;
    if (cutensornetStateConfigure(h, c.st, CUTENSORNET_STATE_CONFIG_MPS_SVD_S_NORMALIZATION, &nm, 4)) return false;
  }
  if (cutensornetStateFinalizeMPS(h, c.st, CUTENSORNET_BOUNDARY_CONDITION_OPEN, ep.data(), su.row_major ? sp.data() : nullptr))
    return false;
  static WS* ws = nullptr;
  delete ws;
  ws = new WS;
  if (cutensornetStatePrepare(h, c.st, (size_t)1 << 30, ws->d, 0)) return false;
  ws->alloc();
  static std::vector<Dev*> bufs;
  for (Dev* d : bufs) delete d;
  bufs.clear();
  std::vector<void*> ptr((size_t)n);
  for (int i = 0; i < n; ++i) {
    size_t v = 1;
    for (auto x : ex[i]) v *= x;
    bufs.push_back(new Dev(v));
    ptr[i] = bufs[i]->p;
  }
  std::vector<std::vector<int64_t>> eo((size_t)n, std::vector<int64_t>(3, -1)), so((size_t)n, std::vector<int64_t>(3, -1));
  std::vector<int64_t*> eop((size_t)n), sop((size_t)n);
  for (int i = 0; i < n; ++i) eop[i] = eo[i].data(), sop[i] = so[i].data();
  if (cutensornetStateCompute(h, c.st, ws->d, eop.data(), sop.data(), ptr.data(), 0)) return false;
  out.ext.assign((size_t)n, {});
  out.str.assign((size_t)n, {});
  out.data.clear();
  for (int i = 0; i < n; ++i) {
    const size_t r = (i == 0 || i == n - 1) ? 2 : 3;
    out.ext[i].assign(eo[i].begin(), eo[i].begin() + r);
    out.str[i].assign(so[i].begin(), so[i].begin() + r);
    out.data.push_back(bufs[i]->get());
  }
  (void)ext_flat;
  return true;
}

static double fidelity(const std::vector<cd>& a, const std::vector<cd>& b, double* nb = nullptr) {
  cd ov = 0;
  double na = 0, nbb = 0;
  for (size_t i = 0; i < a.size(); ++i) ov += std::conj(a[i]) * b[i], na += std::norm(a[i]), nbb += std::norm(b[i]);
  if (nb) *nb = nbb;
  return std::norm(ov) / (na * nbb);
}

static void test_mps() {
  std::printf("-- matrix product states\n");
  const std::vector<std::vector<int>> g_exact = {{0}, {1}, {2}, {3}, {4}, {0, 1}, {2, 3}, {1, 2}, {4, 3}, {0, 3}, {4, 1}};
  {
    rng_state = 777;
    Circuit c({2, 2, 2, 2, 2});
    int32_t nt = -1, nmod[8] = {0};
    MpsSetup su;
    MpsOut m;
    check(run_mps(g_exact, su, c, m), "an exact MPS is computed");
    IS(cutensornetGetOutputStateDetails(h, c.st, &nt, nmod, nullptr, nullptr), 0);
    check(nt == 5 && nmod[0] == 2 && nmod[1] == 3 && nmod[4] == 2, "an MPS state has one tensor per mode, the ends with two modes");
    check(m.ext[0][1] == 2 && m.ext[1][2] == 4 && m.ext[2][0] == 4 && m.ext[3][2] == 2 &&
              m.str[1][0] == 1 && m.str[1][1] == 2 && m.str[1][2] == 4,
          "bond extents and column-major strides");
    check(maxdiff(mps_dense(m, 5), c.ref.v) < 1e-9, "the MPS contracts to the state of the circuit");
    // properties of an MPS state
    cutensornetStateAccessor_t acc;
    IS(cutensornetCreateAccessor(h, c.st, 0, nullptr, nullptr, &acc), 0);
    WS ws;
    IS(cutensornetAccessorPrepare(h, acc, (size_t)1 << 30, ws.d, 0), 0);
    ws.alloc();
    Dev o(32);
    cd nrm;
    unsigned char nb[16];
    IS(cutensornetAccessorCompute(h, acc, nullptr, ws.d, o.p, nb, 0), 0);
    nrm = cd(((double*)nb)[0], ((double*)nb)[1]);
    double want = 0;
    for (auto& x : c.ref.v) want += std::norm(x);
    check(maxdiff(o.get(), c.ref.v) < 1e-9 && std::abs(nrm.real() - want) < 1e-9 * want, "an accessor of an MPS state");
    cutensornetDestroyAccessor(acc);
    int32_t mm[1] = {2};
    cutensornetStateMarginal_t mg;
    IS(cutensornetCreateMarginalDiagonal(h, c.st, 1, mm, 0, nullptr, nullptr, &mg), 0);
    WS w2;
    IS(cutensornetMarginalPrepare(h, mg, (size_t)1 << 30, w2.d, 0), 0);
    w2.alloc();
    Dev o2(2);
    IS(cutensornetMarginalCompute(h, mg, nullptr, w2.d, o2.p, 0), 0);
    double p0 = 0;
    for (size_t i = 0; i < 32; ++i)
      if (!((i >> 2) & 1)) p0 += std::norm(c.ref.v[i]);
    check(std::abs(o2.get()[0].real() - p0) < 1e-9 * want, "a marginal of an MPS state");
    cutensornetDestroyMarginal(mg);
    // a new operator makes the MPS outdated
    c.gate({0}, c.random_matrix({0}));
    cutensornetStateAccessor_t a2;
    IS(cutensornetCreateAccessor(h, c.st, 0, nullptr, nullptr, &a2), 0);   // measured: still the MPS computed last
    cutensornetDestroyAccessor(a2);
    std::vector<void*> none(5, nullptr);
    IS(cutensornetStateCompute(h, c.st, ws.d, nullptr, nullptr, none.data(), 0), 7);
  }
  {  // the canonical center: sites before it are left-orthogonal, after it right-orthogonal
    rng_state = 777;
    Circuit c({2, 2, 2, 2, 2});
    MpsSetup su;
    su.center = 2;
    MpsOut m;
    check(run_mps(g_exact, su, c, m), "an MPS with a canonical center is computed");
    check(maxdiff(mps_dense(m, 5), c.ref.v) < 1e-9, "its state is unchanged");
    double dl = 0, dr = 0;
    const int n = 5;
    for (int i = 0; i < n; ++i) {
      const size_t L = i == 0 ? 1 : (size_t)m.ext[i][0], R = i == n - 1 ? 1 : (size_t)m.ext[i].back();
      auto at = [&](size_t l, int p, size_t r) {
        int64_t off;
        if (i == 0) off = p * m.str[i][0] + r * m.str[i][1];
        else if (i == n - 1) off = l * m.str[i][0] + p * m.str[i][1];
        else off = l * m.str[i][0] + p * m.str[i][1] + r * m.str[i][2];
        return m.data[i][off];
      };
      if (i < 2) {
        for (size_t r = 0; r < R; ++r)
          for (size_t r2 = 0; r2 < R; ++r2) {
            cd s = 0;
            for (size_t l = 0; l < L; ++l)
              for (int p = 0; p < 2; ++p) s += std::conj(at(l, p, r)) * at(l, p, r2);
            dl = std::max(dl, std::abs(s - (r == r2 ? 1.0 : 0.0)));
          }
      } else if (i > 2) {
        for (size_t l = 0; l < L; ++l)
          for (size_t l2 = 0; l2 < L; ++l2) {
            cd s = 0;
            for (size_t r = 0; r < R; ++r)
              for (int p = 0; p < 2; ++p) s += std::conj(at(l, p, r)) * at(l2, p, r);
            dr = std::max(dr, std::abs(s - (l == l2 ? 1.0 : 0.0)));
          }
      }
    }
    check(dl < 1e-9 && dr < 1e-9, "sites left of the center are left-orthogonal, those right of it right-orthogonal");
  }
  {  // row-major output strides
    rng_state = 777;
    Circuit c({2, 2, 2, 2, 2});
    MpsSetup su;
    su.row_major = true;
    MpsOut m;
    check(run_mps(g_exact, su, c, m), "an MPS with the caller's strides is computed");
    check(m.str[1][0] == 8 && m.str[1][1] == 4 && m.str[1][2] == 1 && maxdiff(mps_dense(m, 5), c.ref.v) < 1e-9,
          "the strides are the caller's");
  }
  {  // a product state: the bond extents are what the operators made, not the largest allowed
    rng_state = 777;
    Circuit c({2, 2, 2, 2, 2});
    MpsSetup su;
    MpsOut m;
    check(run_mps({{0}, {1}, {2}, {3}, {4}, {1, 2}}, su, c, m), "a nearly product MPS is computed");
    check(m.ext[0][1] == 1 && m.ext[1][2] == 2 && m.ext[2][2] == 1 && m.ext[3][2] == 1 && m.ext[4][0] == 1,
          "bonds without a two-site operator stay 1");
    check(maxdiff(mps_dense(m, 5), c.ref.v) < 1e-9, "and it is exact");
  }
  {  // truncation: the card's results for these seeded circuits, which pin the algorithm
    struct Case {
      const char* what;
      std::vector<std::vector<int>> gates;
      MpsSetup su;
      double norm2, fid;
    };
    std::vector<Case> cases;
    const std::vector<std::vector<int>> gadj = {{0}, {1}, {2}, {3}, {4}, {0, 1}, {2, 3}, {1, 2}, {3, 4}, {0, 1}, {1, 2}, {2, 3}};
    const std::vector<std::vector<int>> gfar = {{0}, {1}, {2}, {3}, {4}, {0, 2}, {1, 3}, {2, 4}, {0, 4}, {3, 1}, {4, 0}};
    MpsSetup s2;
    s2.maxb = {2, 2, 2, 2};
    cases.push_back({"adjacent operators, bond extent 2", gadj, s2, NAN, NAN});
    cases.push_back({"non-adjacent operators, bond extent 2", gfar, s2, NAN, NAN});
    MpsSetup sn = s2;
    sn.norm = CUTENSORNET_TENSOR_SVD_NORMALIZATION_L2;
    cases.push_back({"adjacent, singular values L2-normalized", gadj, sn, NAN, NAN});
    MpsSetup sa = s2;
    sa.maxb = {4, 4, 4, 4};
    sa.abs_cut = 1.0;
    cases.push_back({"adjacent, absolute cutoff 1", gadj, sa, NAN, NAN});
    MpsSetup sr = sa;
    sr.abs_cut = 0;
    sr.rel_cut = 0.3;
    cases.push_back({"adjacent, relative cutoff 0.3", gadj, sr, NAN, NAN});
    MpsSetup sd = sa;
    sd.abs_cut = 0;
    sd.dw_cut = 0.05;
    cases.push_back({"adjacent, discarded weight 0.05", gadj, sd, NAN, NAN});
    for (auto& k : cases) {
      rng_state = 777;
      Circuit c({2, 2, 2, 2, 2});
      MpsOut m;
      char what[160];
      if (!run_mps(k.gates, k.su, c, m)) {
        check(false, k.what);
        continue;
      }
      double nb;
      const double f = fidelity(c.ref.v, mps_dense(m, 5), &nb);
      std::printf("   measured %-44s norm^2 %.12g fidelity %.12g\n", k.what, nb, f);
      std::snprintf(what, sizeof what, "truncated MPS: %s", k.what);
      check(std::isnan(k.fid) || (std::abs(nb / k.norm2 - 1) < 1e-7 && std::abs(f - k.fid) < 1e-7), what);
    }
  }
}

// ---- initializing and capturing an MPS ----

static void test_mps_init_capture() {
  std::printf("-- initial and captured MPS\n");
  const int n = 4;
  std::vector<int64_t> e4(n, 2);
  Circuit c(e4);
  // a random MPS of bond extent 2 as the initial state
  std::vector<std::vector<int64_t>> ex = {{2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2}}, st((size_t)n);
  std::vector<std::vector<cd>> td;
  std::vector<Dev*> t;
  std::vector<void*> tp;
  std::vector<const int64_t*> ep;
  for (int i = 0; i < n; ++i) {
    size_t v = 1;
    int64_t s = 1;
    for (auto x : ex[i]) v *= x, st[i].push_back(s), s *= x;
    td.push_back(randv(v));
    t.push_back(new Dev(td[i]));
    tp.push_back(t[i]->p);
    ep.push_back(ex[i].data());
  }
  IS(cutensornetStateInitializeMPS(h, c.st, CUTENSORNET_BOUNDARY_CONDITION_OPEN, ep.data(), nullptr, tp.data()), 0);
  MpsOut init;
  init.ext = ex;
  init.str = st;
  init.data = td;
  c.ref.v = mps_dense(init, n);
  auto G = c.random_matrix({1, 2});
  Dev dg(G);
  int32_t mm[2] = {1, 2};
  int64_t id = -1;
  IS(cutensornetStateApplyTensorOperator(h, c.st, 2, mm, dg.p, nullptr, 1, 0, 0, &id), 0);
  check(id == 10, "initializing an MPS reserves a further n + 1 ids");
  c.ref.apply({1, 2}, G);
  cd nrm;
  check(maxdiff(accessor_full(c, 16, &nrm), c.ref.v) < 1e-9, "an operator applied to an initial MPS");
  // the vacuum when no tensors are given
  Circuit v(e4);
  IS(cutensornetStateInitializeMPS(h, v.st, CUTENSORNET_BOUNDARY_CONDITION_OPEN, ep.data(), nullptr, nullptr), 0);
  auto a = accessor_full(v, 16, &nrm);
  double rest = 0;
  for (size_t i = 1; i < 16; ++i) rest = std::max(rest, std::abs(a[i]));
  check(std::abs(a[0] - 1.0) < 1e-12 && rest < 1e-12, "no tensors: the vacuum");
  for (Dev* d : t) delete d;
  // capture: the computed MPS becomes the initial state
  rng_state = 777;
  Circuit k({2, 2, 2, 2, 2});
  MpsSetup su;
  MpsOut m;
  check(run_mps({{0}, {1}, {2}, {3}, {4}, {0, 1}, {1, 2}, {3, 4}}, su, k, m), "an MPS is computed");
  IS(cutensornetStateCaptureMPS(h, k.st), 0);
  auto G2 = k.random_matrix({2, 3});
  Dev dg2(G2);
  int32_t m23[2] = {2, 3};
  IS(cutensornetStateApplyTensorOperator(h, k.st, 2, m23, dg2.p, nullptr, 1, 0, 0, &id), 0);
  // measured on an RTX 3060: an accessor still reads the captured MPS (the operator joins the state when
  // the MPS is computed again), so it returns the amplitudes from before the operator.
  check(maxdiff(accessor_full(k, 32, &nrm), k.ref.v) < 1e-9, "an accessor reads the captured MPS, not the operators applied after it");
  k.ref.apply({2, 3}, G2);
  Circuit nc(e4);
  IS(cutensornetStateCaptureMPS(h, nc.st), 7);  // nothing computed
}

// ---- argument errors ----

static void test_errors() {
  std::printf("-- errors\n");
  int64_t e3[3] = {2, 2, 2};
  cutensornetState_t st = nullptr;
  IS(cutensornetCreateState(nullptr, CUTENSORNET_STATE_PURITY_PURE, 3, e3, CUDA_C_64F, &st), 7);
  IS(cutensornetCreateState(h, CUTENSORNET_STATE_PURITY_PURE, 3, e3, CUDA_C_64F, nullptr), 7);
  IS(cutensornetCreateState(h, CUTENSORNET_STATE_PURITY_PURE, 0, e3, CUDA_C_64F, &st), 7);
  IS(cutensornetCreateState(h, CUTENSORNET_STATE_PURITY_PURE, 3, nullptr, CUDA_C_64F, &st), 7);
  {
    int64_t bad[2] = {2, 0};
    IS(cutensornetCreateState(h, CUTENSORNET_STATE_PURITY_PURE, 2, bad, CUDA_C_64F, &st), 7);
  }
  IS(cutensornetCreateState(h, CUTENSORNET_STATE_PURITY_PURE, 3, e3, CUDA_C_16F, &st), 15);
  IS(cutensornetCreateState(h, CUTENSORNET_STATE_PURITY_PURE, 3, e3, CUDA_R_32F, &st), 0);
  cutensornetDestroyState(st);
  Circuit c({2, 2, 2});
  Dev g(randv(4)), g2(randv(16));
  int64_t id = -1;
  int32_t m0[1] = {0}, mdup[2] = {1, 1}, moob[1] = {3}, mneg[1] = {-1};
  IS(cutensornetStateApplyTensorOperator(nullptr, c.st, 1, m0, g.p, nullptr, 1, 0, 0, &id), 7);
  IS(cutensornetStateApplyTensorOperator(h, nullptr, 1, m0, g.p, nullptr, 1, 0, 0, &id), 7);
  IS(cutensornetStateApplyTensorOperator(h, c.st, 0, m0, g.p, nullptr, 1, 0, 0, &id), 7);
  IS(cutensornetStateApplyTensorOperator(h, c.st, 1, nullptr, g.p, nullptr, 1, 0, 0, &id), 7);
  IS(cutensornetStateApplyTensorOperator(h, c.st, 1, m0, nullptr, nullptr, 1, 0, 0, &id), 7);
  IS(cutensornetStateApplyTensorOperator(h, c.st, 1, m0, g.p, nullptr, 1, 0, 0, nullptr), 7);
  IS(cutensornetStateApplyTensorOperator(h, c.st, 2, mdup, g2.p, nullptr, 1, 0, 0, &id), 7);
  IS(cutensornetStateApplyTensorOperator(h, c.st, 1, moob, g.p, nullptr, 1, 0, 0, &id), 7);
  IS(cutensornetStateApplyTensorOperator(h, c.st, 1, mneg, g.p, nullptr, 1, 0, 0, &id), 7);
  c.gate({0}, c.random_matrix({0}));
  // accessors
  cutensornetStateAccessor_t acc = nullptr;
  int32_t pm4[4] = {0, 1, 2, 0}, pm2[2] = {1, 1}, pm3[1] = {3}, pm1[1] = {1};
  IS(cutensornetCreateAccessor(h, c.st, 0, nullptr, nullptr, nullptr), 7);
  IS(cutensornetCreateAccessor(h, c.st, -1, nullptr, nullptr, &acc), 7);
  IS(cutensornetCreateAccessor(h, c.st, 1, nullptr, nullptr, &acc), 7);
  IS(cutensornetCreateAccessor(h, c.st, 4, pm4, nullptr, &acc), 7);
  IS(cutensornetCreateAccessor(h, c.st, 2, pm2, nullptr, &acc), 7);
  IS(cutensornetCreateAccessor(h, c.st, 1, pm3, nullptr, &acc), 7);
  IS(cutensornetCreateAccessor(h, c.st, 1, pm1, nullptr, &acc), 0);
  {
    WS ws;
    Dev o(4);
    int64_t v1[1] = {1};
    // no scratch memory set (and not prepared): a memory pool is not supported
    IS(cutensornetAccessorCompute(h, acc, v1, ws.d, o.p, nullptr, 0), 15);
    IS(cutensornetAccessorPrepare(h, acc, 100, ws.d, 0), 7);
    IS(cutensornetAccessorPrepare(h, acc, (size_t)1 << 30, ws.d, 0), 0);
    int64_t big[1] = {2};
    ws.alloc();
    IS(cutensornetAccessorCompute(h, acc, big, ws.d, o.p, nullptr, 0), 7);
    IS(cutensornetAccessorCompute(h, acc, nullptr, ws.d, o.p, nullptr, 0), 7);
    IS(cutensornetAccessorCompute(h, acc, v1, ws.d, nullptr, nullptr, 0), 7);
    IS(cutensornetAccessorCompute(h, acc, v1, nullptr, o.p, nullptr, 0), 7);
    IS(cutensornetAccessorCompute(h, acc, v1, ws.d, o.p, nullptr, 0), 0);
    // attributes
    int32_t hv = 4;
    IS(cutensornetAccessorConfigure(h, acc, CUTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES, &hv, 4), 0);
    IS(cutensornetAccessorConfigure(h, acc, CUTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES, &hv, 8), 7);
    IS(cutensornetAccessorConfigure(h, acc, CUTENSORNET_ACCESSOR_INFO_FLOPS, &hv, 4), 7);
    int32_t got = -1;
    IS(cutensornetAccessorGetInfo(h, acc, CUTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES, &got, 4), 0);
    check(got == 4, "a configured attribute reads back");
    double f;
    IS(cutensornetAccessorGetInfo(h, acc, CUTENSORNET_ACCESSOR_INFO_FLOPS, &f, 4), 7);
  }
  IS(cutensornetDestroyAccessor(acc), 0);
  IS(cutensornetDestroyAccessor(nullptr), 0);
  // state attributes
  {
    int32_t hs = 3, got = -1;
    IS(cutensornetStateConfigure(h, c.st, CUTENSORNET_STATE_CONFIG_NUM_HYPER_SAMPLES, &hs, 4), 0);
    IS(cutensornetStateGetInfo(h, c.st, CUTENSORNET_STATE_CONFIG_NUM_HYPER_SAMPLES, &got, 4), 0);
    check(got == 3, "the number of hyper samples reads back");
    int32_t cc;
    IS(cutensornetStateGetInfo(h, c.st, CUTENSORNET_STATE_CONFIG_MPS_CANONICAL_CENTER, &cc, 4), 0);
    check(cc == -1, "no canonical center by default");
    double ac = 0.1;
    IS(cutensornetStateConfigure(h, c.st, CUTENSORNET_STATE_CONFIG_MPS_SVD_ABS_CUTOFF, &ac, 8), 0);
    IS(cutensornetStateConfigure(h, c.st, CUTENSORNET_STATE_CONFIG_MPS_SVD_ABS_CUTOFF, &ac, 4), 7);
    IS(cutensornetStateConfigure(h, c.st, CUTENSORNET_STATE_CONFIG_MPS_SVD_ABS_CUTOFF, nullptr, 8), 7);
    ac = -1;
    IS(cutensornetStateConfigure(h, c.st, CUTENSORNET_STATE_CONFIG_MPS_SVD_ABS_CUTOFF, &ac, 8), 7);
    IS(cutensornetStateConfigure(h, c.st, CUTENSORNET_STATE_INFO_FLOPS, &ac, 8), 15);
    IS(cutensornetStateConfigure(h, c.st, (cutensornetStateAttributes_t)99, &ac, 8), 15);
    IS(cutensornetStateGetInfo(h, c.st, CUTENSORNET_STATE_CONFIG_MPS_SVD_ABS_CUTOFF, &ac, 8), 15);
    IS(cutensornetStateGetInfo(h, c.st, CUTENSORNET_STATE_INFO_FLOPS, &ac, 4), 7);
    int32_t cen = 99;
    IS(cutensornetStateConfigure(h, c.st, CUTENSORNET_STATE_CONFIG_MPS_CANONICAL_CENTER, &cen, 4), 7);
    int32_t gauge = 5;
    IS(cutensornetStateConfigure(h, c.st, CUTENSORNET_STATE_CONFIG_MPS_GAUGE_OPTION, &gauge, 4), 7);
    // MPS: not for mixed states or one mode
    int64_t two[2] = {2, 2}, bonds[2] = {2, 2};
    const int64_t* ep[2] = {bonds, bonds};
    cutensornetState_t mixed, one;
    IS(cutensornetCreateState(h, CUTENSORNET_STATE_PURITY_MIXED, 2, two, CUDA_C_64F, &mixed), 0);
    IS(cutensornetStateFinalizeMPS(h, mixed, CUTENSORNET_BOUNDARY_CONDITION_OPEN, ep, nullptr), 15);
    int64_t one_e[1] = {2};
    IS(cutensornetCreateState(h, CUTENSORNET_STATE_PURITY_PURE, 1, one_e, CUDA_C_64F, &one), 0);
    IS(cutensornetStateFinalizeMPS(h, one, CUTENSORNET_BOUNDARY_CONDITION_OPEN, ep, nullptr), 7);
    IS(cutensornetStateFinalizeMPS(h, c.st, CUTENSORNET_BOUNDARY_CONDITION_OPEN, nullptr, nullptr), 7);
    cutensornetDestroyState(mixed);
    cutensornetDestroyState(one);
  }
  // marginals, samplers, expectations
  {
    int32_t mm[2] = {0, 1}, pj[1] = {1}, mdup2[2] = {0, 0}, pj2[1] = {2}, mo[1] = {3};
    cutensornetStateMarginal_t mg = nullptr;
    IS(cutensornetCreateMarginal(h, c.st, 0, mm, 0, nullptr, nullptr, &mg), 7);
    IS(cutensornetCreateMarginal(h, c.st, 2, nullptr, 0, nullptr, nullptr, &mg), 7);
    IS(cutensornetCreateMarginal(h, c.st, 2, mdup2, 0, nullptr, nullptr, &mg), 7);
    IS(cutensornetCreateMarginal(h, c.st, 1, mo, 0, nullptr, nullptr, &mg), 7);
    IS(cutensornetCreateMarginal(h, c.st, 2, mm, 1, pj, nullptr, &mg), 7);  // a mode both open and projected
    IS(cutensornetCreateMarginal(h, c.st, 2, mm, 1, nullptr, nullptr, &mg), 7);
    IS(cutensornetCreateMarginal(h, c.st, 2, mm, 1, pj2, nullptr, &mg), 0);
    cutensornetDestroyMarginal(mg);
    cutensornetStateSampler_t sp = nullptr;
    IS(cutensornetCreateSampler(h, c.st, 0, nullptr, &sp), 7);
    IS(cutensornetCreateSampler(h, c.st, 2, nullptr, &sp), 7);
    IS(cutensornetCreateSampler(h, c.st, 2, mdup2, &sp), 7);
    IS(cutensornetCreateSampler(h, c.st, 1, mo, &sp), 7);
    int32_t order[3] = {2, 0, 1};
    IS(cutensornetCreateSampler(h, c.st, 3, order, &sp), 0);
    cutensornetDestroySampler(sp);
    cutensornetNetworkOperator_t op, op2, op3;
    IS(cutensornetCreateNetworkOperator(nullptr, 3, e3, CUDA_C_64F, &op), 7);
    IS(cutensornetCreateNetworkOperator(h, 0, e3, CUDA_C_64F, &op), 7);
    IS(cutensornetCreateNetworkOperator(h, 3, e3, CUDA_C_64F, &op), 0);
    int64_t e2[2] = {2, 2};
    IS(cutensornetCreateNetworkOperator(h, 2, e2, CUDA_C_64F, &op2), 0);
    IS(cutensornetCreateNetworkOperator(h, 3, e3, CUDA_C_32F, &op3), 0);
    cutensornetStateExpectation_t ex = nullptr;
    IS(cutensornetCreateExpectation(h, c.st, op, &ex), 7);   // no components
    Dev dA(randv(4));
    int32_t nm1[1] = {1}, ms0[1] = {0}, msb[1] = {5};
    const int32_t* mp0[1] = {ms0};
    const int32_t* mpb[1] = {msb};
    const void* d0[1] = {dA.p};
    int64_t cid;
    cuDoubleComplex one = make_cuDoubleComplex(1, 0);
    IS(cutensornetNetworkOperatorAppendProduct(h, op, one, 1, nm1, mpb, nullptr, d0, &cid), 7);
    IS(cutensornetNetworkOperatorAppendProduct(h, op, one, 0, nm1, mp0, nullptr, d0, &cid), 7);
    IS(cutensornetNetworkOperatorAppendProduct(h, op, one, 1, nm1, mp0, nullptr, nullptr, &cid), 7);
    IS(cutensornetNetworkOperatorAppendProduct(h, op, one, 1, nm1, mp0, nullptr, d0, &cid), 0);
    IS(cutensornetNetworkOperatorAppendProduct(h, op2, one, 1, nm1, mp0, nullptr, d0, &cid), 0);
    IS(cutensornetNetworkOperatorAppendProduct(h, op3, one, 1, nm1, mp0, nullptr, d0, &cid), 0);
    IS(cutensornetCreateExpectation(h, c.st, op, &ex), 0);
    cutensornetDestroyExpectation(ex);
    ex = nullptr;
    IS(cutensornetCreateExpectation(h, c.st, op2, &ex), 7);  // other modes
    IS(cutensornetCreateExpectation(h, c.st, op3, &ex), 7);  // another data type
    IS(cutensornetCreateExpectation(h, c.st, nullptr, &ex), 7);
    cutensornetDestroyNetworkOperator(op);
    cutensornetDestroyNetworkOperator(op2);
    cutensornetDestroyNetworkOperator(op3);
  }
}

int main() {
  if (cutensornetCreate(&h)) {
    std::printf("FAIL cutensornetCreate\n");
    return 1;
  }
  test_conventions();
  test_unitary_flags();
  test_operator_kinds();
  test_marginals();
  test_expectation();
  test_channels();
  test_sampler();
  test_state_compute();
  test_mps();
  test_mps_init_capture();
  test_errors();
  cutensornetDestroy(h);
  std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
