// cuTensorNet's gradients of an expectation value with respect to the tensor operators
// of a state: cutensornetStateApplyTensorOperatorWithGradient registers a buffer for the
// gradient of a gate, cutensornetExpectationComputeWithGradientsBackward writes it.
//
// The convention for complex data is that of PyTorch. With E the expectation value
// (the sum over the operator's terms of coefficient * <psi|term|psi>), N the squared norm
// <psi|psi> of the state, and A a gate's tensor, the gradient written for A is
//
//     Ebar conj(dE/dA) + conj(Ebar) dE/dA*  +  Nbar conj(dN/dA) + conj(Nbar) dN/dA*
//
// with Ebar and Nbar the upstream adjoints and d/dA, d/dA* the Wirtinger derivatives. The
// expectations hold for NVIDIA's libcutensornet 2.14 on an RTX 3060 too (the convention, the
// zeros for a gate applied as an adjoint or cancelled against its adjoint outside the light cone of
// the operator, the checks of the arguments and their order, and what is left alone were measured
// there); the derivatives are taken here by finite differences of the library's own
// forward value, so the program needs no reference simulator.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static unsigned rng_state = 777;
static double rnd() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return ((double)(rng_state % 20001) - 10000.0) / 10000.0;
}
static std::vector<cd> randm(size_t n) {
  std::vector<cd> v(n);
  for (auto& x : v) {
    const double a = rnd();
    x = cd(a, rnd());
  }
  return v;
}

static void* dev(const std::vector<cd>& v) {
  void* d = nullptr;
  cudaMalloc(&d, v.size() * 16 + 256);
  cudaMemcpy(d, v.data(), v.size() * 16, cudaMemcpyHostToDevice);
  return d;
}
static std::vector<cd> fetch(const void* d, size_t n) {
  std::vector<cd> v(n);
  cudaDeviceSynchronize();
  cudaMemcpy(v.data(), d, n * 16, cudaMemcpyDeviceToHost);
  return v;
}

struct WS {
  cutensornetWorkspaceDescriptor_t d = nullptr;
  void* scratch = nullptr;
  void* cache = nullptr;
  WS() { cutensornetCreateWorkspaceDescriptor(h, &d); }
  void alloc(bool with_scratch = true) {
    int64_t rc = 0, cm = 0;
    cutensornetWorkspaceGetMemorySize(h, d, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE,
                                      CUTENSORNET_WORKSPACE_SCRATCH, &rc);
    cutensornetWorkspaceGetMemorySize(h, d, CUTENSORNET_WORKSIZE_PREF_MAX, CUTENSORNET_MEMSPACE_DEVICE,
                                      CUTENSORNET_WORKSPACE_CACHE, &cm);
    if (with_scratch) {
      cudaMalloc(&scratch, (size_t)std::max<int64_t>(rc, 256));
      cutensornetWorkspaceSetMemory(h, d, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, scratch,
                                    std::max<int64_t>(rc, 256));
    }
    if (cm > 0) {
      cudaMalloc(&cache, (size_t)cm);
      cutensornetWorkspaceSetMemory(h, d, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_CACHE, cache, cm);
    }
  }
  ~WS() {
    cudaFree(scratch);
    cudaFree(cache);
    cutensornetDestroyWorkspaceDescriptor(d);
  }
};

// ---- a circuit of two qubits ----

struct Gate {
  std::vector<int32_t> modes;
  std::vector<cd> m;
  int unitary = 0, adjoint = 0;
  bool registered = false;
  const int64_t* gradient_strides = nullptr;
};

static std::vector<cd> M0, M1, M2;

struct Result {
  cd E, N;
  std::vector<std::vector<cd>> gradients;  // of the registered gates, in order
  int status = 0;
};

enum Terms { TWO_TERMS, ONE_TERM };

// Builds the state and the expectation value, and either computes the forward values or the gradients.
static Result run(const std::vector<Gate>& gates, bool mixed, bool backward, cd Ebar, bool use_norm, cd Nbar,
                  Terms terms = TWO_TERMS, int calls = 1, int accumulate = 0, const std::vector<cd>* start = nullptr) {
  Result r;
  cutensornetState_t st;
  int64_t ext[2] = {2, 2};
  cutensornetCreateState(h, mixed ? CUTENSORNET_STATE_PURITY_MIXED : CUTENSORNET_STATE_PURITY_PURE, 2, ext, CUDA_C_64F, &st);
  std::vector<void*> keep, grads;
  for (const Gate& g : gates) {
    void* d = dev(g.m);
    keep.push_back(d);
    int64_t id = -1;
    if (g.registered) {
      void* gd = dev(start ? *start : std::vector<cd>(g.m.size(), cd(0, 0)));
      grads.push_back(gd);
      cutensornetStateApplyTensorOperatorWithGradient(h, st, (int)g.modes.size(), g.modes.data(), d, nullptr, 1, g.adjoint,
                                                     g.unitary, gd, g.gradient_strides, &id);
    } else {
      cutensornetStateApplyTensorOperator(h, st, (int)g.modes.size(), g.modes.data(), d, nullptr, 1, g.adjoint, g.unitary, &id);
    }
  }
  cutensornetNetworkOperator_t op;
  cutensornetCreateNetworkOperator(h, 2, ext, CUDA_C_64F, &op);
  void *d0 = dev(M0), *d1 = dev(M1), *d2 = dev(M2);
  if (terms == TWO_TERMS) {
    int32_t nm[2] = {1, 1}, m0[1] = {0}, m1[1] = {1};
    const int32_t* mp[2] = {m0, m1};
    const void* dat[2] = {d0, d1};
    int64_t cid;
    cutensornetNetworkOperatorAppendProduct(h, op, make_cuDoubleComplex(0.7, 0.3), 2, nm, mp, nullptr, dat, &cid);
  }
  {
    int32_t nm[1] = {1}, m1[1] = {1};
    const int32_t* mp[1] = {m1};
    const void* dat[1] = {d2};
    int64_t cid;
    cutensornetNetworkOperatorAppendProduct(h, op, make_cuDoubleComplex(-0.4, 0.2), 1, nm, mp, nullptr, dat, &cid);
  }
  cutensornetStateExpectation_t ex;
  cutensornetCreateExpectation(h, st, op, &ex);
  WS w;
  cutensornetExpectationPrepare(h, ex, (size_t)1 << 30, w.d, 0);
  w.alloc();
  if (!backward) {
    cutensornetExpectationCompute(h, ex, w.d, &r.E, &r.N, 0);
  } else {
    for (int i = 0; i < calls; ++i)
      r.status = cutensornetExpectationComputeWithGradientsBackward(h, ex, i == 0 ? accumulate : 1, &Ebar,
                                                                   use_norm ? &Nbar : nullptr, w.d, &r.E,
                                                                   use_norm ? &r.N : nullptr, 0);
    size_t k = 0;
    for (const Gate& g : gates)
      if (g.registered) r.gradients.push_back(fetch(grads[k++], g.m.size()));
  }
  cutensornetDestroyExpectation(ex);
  cutensornetDestroyNetworkOperator(op);
  cutensornetDestroyState(st);
  for (void* p : keep) cudaFree(p);
  for (void* p : grads) cudaFree(p);
  cudaFree(d0);
  cudaFree(d1);
  cudaFree(d2);
  return r;
}

// dE/dA and dE/dA*, dN/dA and dN/dA* of the elements of gate `which`, by central differences of the forward values.
struct Wirtinger {
  std::vector<cd> dE, dEc, dN, dNc;
};
static Wirtinger derivatives(const std::vector<Gate>& gates, size_t which, bool mixed, Terms terms) {
  const size_t n = gates[which].m.size();
  Wirtinger w;
  const double dl = 1e-5;
  for (size_t k = 0; k < n; ++k) {
    auto gp = gates, gm = gates;
    gp[which].m[k] += dl;
    gm[which].m[k] -= dl;
    const Result p = run(gp, mixed, false, 0, false, 0, terms), m = run(gm, mixed, false, 0, false, 0, terms);
    const cd eRe = (p.E - m.E) / (2 * dl), nRe = (p.N - m.N) / (2 * dl);
    gp = gates;
    gm = gates;
    gp[which].m[k] += cd(0, dl);
    gm[which].m[k] -= cd(0, dl);
    const Result pi = run(gp, mixed, false, 0, false, 0, terms), mi = run(gm, mixed, false, 0, false, 0, terms);
    const cd eIm = (pi.E - mi.E) / (2 * dl), nIm = (pi.N - mi.N) / (2 * dl);
    w.dE.push_back((eRe - cd(0, 1) * eIm) / 2.0);
    w.dEc.push_back((eRe + cd(0, 1) * eIm) / 2.0);
    w.dN.push_back((nRe - cd(0, 1) * nIm) / 2.0);
    w.dNc.push_back((nRe + cd(0, 1) * nIm) / 2.0);
  }
  return w;
}

static double rel(const std::vector<cd>& got, const std::vector<cd>& want) {
  double e = 0, m = 1e-300;
  for (size_t i = 0; i < want.size(); ++i) e = std::max(e, std::abs(got[i] - want[i])), m = std::max(m, std::abs(want[i]));
  return got.size() == want.size() ? e / std::max(m, 1.0) : 1e300;
}

static std::vector<cd> expect_gradient(const Wirtinger& w, cd Ebar, bool use_norm, cd Nbar) {
  std::vector<cd> g(w.dE.size());
  for (size_t k = 0; k < g.size(); ++k) {
    g[k] = Ebar * std::conj(w.dE[k]) + std::conj(Ebar) * w.dEc[k];
    if (use_norm) g[k] += Nbar * std::conj(w.dN[k]) + std::conj(Nbar) * w.dNc[k];
  }
  return g;
}

static void test_convention() {
  std::printf("-- the gradient of a gate\n");
  rng_state = 777;
  M0 = randm(4);
  M1 = randm(4);
  M2 = randm(4);
  const std::vector<cd> G1 = randm(4), A = randm(4), Bm = randm(16), C = randm(4), A2 = randm(16);
  struct Adj {
    cd e;
    bool use_norm;
    cd n;
  };
  const std::vector<Adj> adjoints = {{cd(1, 0), false, 0},        {cd(0, 1), false, 0},          {cd(0.5, -1.5), false, 0},
                                     {cd(0, 0), true, cd(1, 0)},  {cd(0, 0), true, cd(0, 1)},   {cd(0, 0), true, cd(2, 3)},
                                     {cd(0.5, -1.5), true, cd(2, 3)}};
  const int64_t transposed1[2] = {2, 1}, transposed2[4] = {8, 4, 2, 1};
  struct Case {
    const char* name;
    std::vector<Gate> gates;
    bool mixed;
  };
  std::vector<Case> cases;
  cases.push_back({"a gate on one mode", {{{0}, G1, 0, 0, false}, {{1}, C, 0, 0, false}, {{0}, A, 0, 0, true}, {{0, 1}, Bm, 0, 0, false}, {{1}, C, 0, 0, false}}, false});
  cases.push_back({"a gate marked unitary in the light cone", {{{0}, G1, 0, 0, false}, {{0}, A, 1, 0, true}, {{0, 1}, Bm, 0, 0, false}, {{1}, C, 0, 0, false}}, false});
  {
    Gate g{{0}, A, 0, 0, true, transposed1};
    cases.push_back({"a gradient buffer in the transposed layout", {{{0}, G1, 0, 0, false}, g, {{0, 1}, Bm, 0, 0, false}, {{1}, C, 0, 0, false}}, false});
  }
  cases.push_back({"a gate on two modes", {{{0}, G1, 0, 0, false}, {{1}, C, 0, 0, false}, {{0, 1}, A2, 0, 0, true}, {{1}, C, 0, 0, false}}, false});
  {
    Gate g{{0, 1}, A2, 0, 0, true, transposed2};
    cases.push_back({"a gate on two modes in the transposed layout", {{{0}, G1, 0, 0, false}, g, {{1}, C, 0, 0, false}}, false});
  }
  cases.push_back({"a mixed state", {{{0}, G1, 0, 0, false}, {{0}, A, 0, 0, true}, {{0, 1}, Bm, 0, 0, false}, {{1}, C, 0, 0, false}}, true});
  for (const Case& c : cases) {
    size_t which = 0;
    for (size_t i = 0; i < c.gates.size(); ++i)
      if (c.gates[i].registered) which = i;
    const Wirtinger w = derivatives(c.gates, which, c.mixed, TWO_TERMS);
    bool all = true, values = true;
    const bool strided = c.gates[which].gradient_strides != nullptr;
    for (const Adj& a : adjoints) {
      const Result r = run(c.gates, c.mixed, true, a.e, a.use_norm, a.n);
      std::vector<cd> want = expect_gradient(w, a.e, a.use_norm, a.n);
      if (strided) {  // the strides {2, 1} / {8, 4, 2, 1} lay the buffer out with the modes in reverse order
        const size_t modes = want.size() == 4 ? 2 : 4;
        std::vector<cd> t(want.size());
        for (size_t k = 0; k < want.size(); ++k) {
          size_t off = 0, rem = k;
          for (size_t d = 0; d < modes; ++d) {  // 2 per mode: index d of k's binary digits, stride 2^(modes - 1 - d)
            off += (rem & 1) << (modes - 1 - d);
            rem >>= 1;
          }
          t[off] = want[k];
        }
        want = t;
      }
      all = all && r.status == 0 && !r.gradients.empty() && rel(r.gradients[0], want) < 1e-6;
      const Result f = run(c.gates, c.mixed, false, 0, false, 0);
      values = values && std::abs(r.E - f.E) < 1e-12 * (1 + std::abs(f.E));
    }
    check(all, (std::string(c.name) + ": Ebar conj(dE/dA) + conj(Ebar) dE/dA* + Nbar conj(dN/dA) + conj(Nbar) dN/dA*").c_str());
    check(values, (std::string(c.name) + ": the call also gives the expectation value").c_str());
  }
}

static void test_zeros_and_overwrite() {
  std::printf("-- what is left out, overwritten and added to\n");
  rng_state = 4242;
  M0 = randm(4);
  M1 = randm(4);
  M2 = randm(4);
  const std::vector<cd> G1 = randm(4), A = randm(4), A2 = randm(4), Bm = randm(16);
  // an adjoint gate: zeros
  {
    const std::vector<Gate> g = {{{0}, G1, 0, 0, false}, {{0}, A, 0, 1, true}, {{0, 1}, Bm, 0, 0, false}};
    const Result r = run(g, false, true, cd(1, 0), false, 0);
    bool zeros = r.gradients.size() == 1;
    for (cd x : r.gradients[0]) zeros = zeros && x == cd(0, 0);
    check(r.status == 0 && zeros, "a gate applied as an adjoint gets a gradient of zeros");
  }
  // a unitary gate outside the light cone of the operator (which acts on mode 1) cancels against its adjoint
  {
    for (int unitary = 0; unitary < 2; ++unitary) {
      const std::vector<Gate> g = {{{1}, G1, 0, 0, false}, {{0}, A, unitary, 0, true}};
      const Result r = run(g, false, true, cd(1, 0), true, cd(1, 0), ONE_TERM);
      bool zeros = true, any = false;
      for (cd x : r.gradients[0]) zeros = zeros && x == cd(0, 0), any = any || x != cd(0, 0);
      check(r.status == 0 && (unitary ? zeros : any),
            unitary ? "a gate marked unitary outside the light cone gets zeros (it cancels against its adjoint)"
                    : "the same gate not marked unitary has a gradient");
    }
  }
  // two registered gates
  {
    const std::vector<Gate> g = {{{0}, A, 0, 0, true}, {{1}, G1, 0, 0, false}, {{0, 1}, Bm, 0, 0, false}, {{1}, A2, 0, 0, true}};
    const Result r = run(g, false, true, cd(1, 0), false, 0);
    check(r.status == 0 && r.gradients.size() == 2, "two gates with gradients are both written");
    const Wirtinger w0 = derivatives(g, 0, false, TWO_TERMS), w1 = derivatives(g, 3, false, TWO_TERMS);
    check(rel(r.gradients[0], expect_gradient(w0, cd(1, 0), false, 0)) < 1e-6 &&
              rel(r.gradients[1], expect_gradient(w1, cd(1, 0), false, 0)) < 1e-6,
          "each with its own derivative");
  }
  // overwrite and accumulate
  {
    const std::vector<Gate> g = {{{0}, A, 0, 0, true}, {{0, 1}, Bm, 0, 0, false}, {{1}, G1, 0, 0, false}};
    const Result base = run(g, false, true, cd(1, 0), false, 0);
    const std::vector<cd> start(4, cd(5, -7));
    const Result over = run(g, false, true, cd(1, 0), false, 0, TWO_TERMS, 1, 0, &start);
    check(rel(over.gradients[0], base.gradients[0]) < 1e-12, "without accumulating, the buffer is overwritten");
    const Result acc = run(g, false, true, cd(1, 0), false, 0, TWO_TERMS, 1, 1, &start);
    std::vector<cd> want = base.gradients[0];
    for (size_t k = 0; k < want.size(); ++k) want[k] += start[k];
    check(rel(acc.gradients[0], want) < 1e-12, "accumulating adds to what the buffer held");
    const Result twice = run(g, false, true, cd(1, 0), false, 0, TWO_TERMS, 2, 0);
    want = base.gradients[0];
    for (auto& x : want) x *= 2.0;
    check(rel(twice.gradients[0], want) < 1e-12, "a second call after an overwriting one accumulates onto it (accumulate = 1)");
  }
}

static void test_arguments() {
  std::printf("-- arguments and states\n");
  rng_state = 99;
  M2 = randm(4);
  const std::vector<cd> A = randm(4), Bm = randm(16);
  cutensornetState_t st;
  int64_t ext[2] = {2, 2};
  cutensornetCreateState(h, CUTENSORNET_STATE_PURITY_PURE, 2, ext, CUDA_C_64F, &st);
  void *dA = dev(A), *dB = dev(Bm), *gd = dev(std::vector<cd>(4)), *gd2 = dev(std::vector<cd>(4, cd(9, 9)));
  int32_t m0[1] = {0}, m01[2] = {0, 1};
  int64_t id = -1, id2 = -1;
  IS(cutensornetStateApplyTensorOperatorWithGradient(h, st, 1, m0, dA, nullptr, 1, 0, 0, nullptr, nullptr, &id),
     CUTENSORNET_STATUS_INVALID_VALUE);
  IS(cutensornetStateApplyTensorOperatorWithGradient(h, st, 1, m0, dA, nullptr, 1, 0, 0, gd, nullptr, &id), 0);
  IS(cutensornetStateApplyTensorOperator(h, st, 2, m01, dB, nullptr, 1, 0, 0, &id2), 0);
  cutensornetNetworkOperator_t op;
  cutensornetCreateNetworkOperator(h, 2, ext, CUDA_C_64F, &op);
  void* d2 = dev(M2);
  {
    int32_t nm[1] = {1}, m1[1] = {1};
    const int32_t* mp[1] = {m1};
    const void* dat[1] = {d2};
    int64_t cid;
    cutensornetNetworkOperatorAppendProduct(h, op, make_cuDoubleComplex(1, 0), 1, nm, mp, nullptr, dat, &cid);
  }
  cutensornetStateExpectation_t ex;
  cutensornetCreateExpectation(h, st, op, &ex);
  WS w, empty;
  cd E, Eb(1, 0), N, Nb(1, 0);
  const cutensornetStatus_t INV = CUTENSORNET_STATUS_INVALID_VALUE;
  IS(cutensornetExpectationComputeWithGradientsBackward(h, ex, 0, &Eb, &Nb, w.d, &E, &N, 0), INV);  // not prepared
  IS(cutensornetExpectationPrepare(h, ex, (size_t)1 << 30, w.d, 0), 0);
  w.alloc();
  IS(cutensornetExpectationPrepare(h, ex, (size_t)1 << 30, empty.d, 0), 0);  // empty gets sizes but no memory
  // The arguments are checked in this order: handle, expectation, workspace, Ebar, expectation value, the norm pair.
  IS(cutensornetExpectationComputeWithGradientsBackward(nullptr, nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0), INV);
  IS(cutensornetExpectationComputeWithGradientsBackward(h, nullptr, 0, &Eb, nullptr, w.d, &E, nullptr, 0), INV);
  IS(cutensornetExpectationComputeWithGradientsBackward(h, ex, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0), INV);
  IS(cutensornetExpectationComputeWithGradientsBackward(h, ex, 0, nullptr, nullptr, w.d, nullptr, &N, 0), INV);
  IS(cutensornetExpectationComputeWithGradientsBackward(h, ex, 0, &Eb, nullptr, w.d, nullptr, &N, 0), INV);
  IS(cutensornetExpectationComputeWithGradientsBackward(h, ex, 0, &Eb, &Nb, w.d, &E, nullptr, 0), INV);   // an adjoint of the norm, no norm
  IS(cutensornetExpectationComputeWithGradientsBackward(h, ex, 0, &Eb, nullptr, w.d, &E, &N, 0), INV);    // a norm, no adjoint
  IS(cutensornetExpectationComputeWithGradientsBackward(h, ex, 0, &Eb, nullptr, empty.d, &E, nullptr, 0),
     CUTENSORNET_STATUS_NOT_SUPPORTED);  // the scratch memory must be set
  IS(cutensornetExpectationComputeWithGradientsBackward(h, ex, 0, nullptr, nullptr, empty.d, &E, nullptr, 0), INV);
  IS(cutensornetExpectationComputeWithGradientsBackward(h, ex, 7, &Eb, nullptr, w.d, &E, nullptr, 0), 0);  // any nonzero accumulates
  IS(cutensornetExpectationComputeWithGradientsBackward(h, ex, 0, &Eb, &Nb, w.d, &E, &N, 0), 0);
  // the gradient buffer can be moved
  IS(cutensornetStateUpdateTensorOperatorGradient(h, st, id, gd2), 0);
  IS(cutensornetExpectationComputeWithGradientsBackward(h, ex, 0, &Eb, nullptr, w.d, &E, nullptr, 0), 0);
  const std::vector<cd> moved = fetch(gd2, 4), old = fetch(gd, 4);
  bool differs = false;
  for (size_t k = 0; k < 4; ++k) differs = differs || std::abs(moved[k] - cd(9, 9)) > 1e-9;
  check(differs, "after cutensornetStateUpdateTensorOperatorGradient the gradient goes to the new buffer");
  IS(cutensornetStateUpdateTensorOperatorGradient(h, st, id2, gd2), 0);  // a tensor applied without a gradient: accepted, no effect
  IS(cutensornetStateUpdateTensorOperatorGradient(h, st, id, nullptr), INV);
  IS(cutensornetStateUpdateTensorOperatorGradient(nullptr, st, id, gd2), INV);
  IS(cutensornetStateUpdateTensorOperatorGradient(h, nullptr, id, gd2), INV);
  // a state that changes after the preparation outdates the expectation value
  int64_t id3 = -1;
  void *dA3 = dev(A), *gd3 = dev(std::vector<cd>(4));
  IS(cutensornetStateApplyTensorOperatorWithGradient(h, st, 1, m0, dA3, nullptr, 1, 0, 0, gd3, nullptr, &id3), 0);
  IS(cutensornetExpectationComputeWithGradientsBackward(h, ex, 0, &Eb, nullptr, w.d, &E, nullptr, 0), INV);
  // with no gradient registered the call is the forward evaluation: the values are written, no buffer is touched
  {
    cutensornetState_t st2;
    cutensornetCreateState(h, CUTENSORNET_STATE_PURITY_PURE, 2, ext, CUDA_C_64F, &st2);
    int64_t i2;
    cutensornetStateApplyTensorOperator(h, st2, 1, m0, dA, nullptr, 1, 0, 0, &i2);
    cutensornetStateExpectation_t ex2;
    cutensornetCreateExpectation(h, st2, op, &ex2);
    WS w2;
    cutensornetExpectationPrepare(h, ex2, (size_t)1 << 30, w2.d, 0);
    w2.alloc();
    cd E2(0, 0), N2(0, 0), Ef(0, 0), Nf(0, 0);
    IS(cutensornetExpectationComputeWithGradientsBackward(h, ex2, 0, &Eb, &Nb, w2.d, &E2, &N2, 0), 0);
    IS(cutensornetExpectationCompute(h, ex2, w2.d, &Ef, &Nf, 0), 0);
    check(std::abs(E2 - Ef) < 1e-12 && std::abs(N2 - Nf) < 1e-12 && std::abs(Nf) > 0.1, "and writes the expectation value and the norm");
    cutensornetDestroyExpectation(ex2);
    cutensornetDestroyState(st2);
  }
  cutensornetDestroyExpectation(ex);
  cutensornetDestroyNetworkOperator(op);
  cutensornetDestroyState(st);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(gd);
  cudaFree(gd2);
  cudaFree(d2);
  cudaFree(dA3);
  cudaFree(gd3);
}

// ---- other data types ----

// One gate on a one-qubit-wide state with the observable Z, in the data type `t`: the gradient and the expectation value as doubles.
static bool one_gate(cudaDataType_t t, cd Ebar, cd Nbar, std::vector<cd>& gradient, cd& E) {
  const double a[4] = {0.3, 0.2, -0.5, 0.7}, m[4] = {1, 0, 0, -1};
  const bool complex_type = t == CUDA_C_32F || t == CUDA_C_64F;
  const size_t es = t == CUDA_R_32F ? 4 : t == CUDA_R_64F ? 8 : t == CUDA_C_32F ? 8 : 16;
  auto pack = [&](const double* v, bool imaginary) {
    std::vector<unsigned char> b(4 * es);
    for (int i = 0; i < 4; ++i) {
      const double re = v[i], im = imaginary ? 0.1 * i : 0.0;
      if (t == CUDA_C_64F) {
        const double q[2] = {re, im};
        std::memcpy(&b[(size_t)i * 16], q, 16);
      } else if (t == CUDA_C_32F) {
        const float q[2] = {(float)re, (float)im};
        std::memcpy(&b[(size_t)i * 8], q, 8);
      } else if (t == CUDA_R_64F) {
        std::memcpy(&b[(size_t)i * 8], &re, 8);
      } else {
        const float q = (float)re;
        std::memcpy(&b[(size_t)i * 4], &q, 4);
      }
    }
    return b;
  };
  auto scalar = [&](cd v) {
    std::vector<unsigned char> b(16, 0);
    if (t == CUDA_C_64F) {
      const double q[2] = {v.real(), v.imag()};
      std::memcpy(b.data(), q, 16);
    } else if (t == CUDA_C_32F) {
      const float q[2] = {(float)v.real(), (float)v.imag()};
      std::memcpy(b.data(), q, 8);
    } else if (t == CUDA_R_64F) {
      const double q = v.real();
      std::memcpy(b.data(), &q, 8);
    } else {
      const float q = (float)v.real();
      std::memcpy(b.data(), &q, 4);
    }
    return b;
  };
  auto unscalar = [&](const unsigned char* p) -> cd {
    if (t == CUDA_C_64F) {
      double q[2];
      std::memcpy(q, p, 16);
      return cd(q[0], q[1]);
    }
    if (t == CUDA_C_32F) {
      float q[2];
      std::memcpy(q, p, 8);
      return cd(q[0], q[1]);
    }
    if (t == CUDA_R_64F) {
      double q;
      std::memcpy(&q, p, 8);
      return cd(q, 0);
    }
    float q;
    std::memcpy(&q, p, 4);
    return cd(q, 0);
  };
  cutensornetState_t st;
  int64_t ext[2] = {2, 2};
  cutensornetCreateState(h, CUTENSORNET_STATE_PURITY_PURE, 2, ext, t, &st);
  const auto ab = pack(a, complex_type), mb = pack(m, false);
  void *da, *dm, *dg;
  cudaMalloc(&da, 64);
  cudaMalloc(&dm, 64);
  cudaMalloc(&dg, 64);
  cudaMemcpy(da, ab.data(), ab.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dm, mb.data(), mb.size(), cudaMemcpyHostToDevice);
  cudaMemset(dg, 0, 64);
  int32_t m0[1] = {0};
  int64_t gid;
  bool ok = cutensornetStateApplyTensorOperatorWithGradient(h, st, 1, m0, da, nullptr, 1, 0, 0, dg, nullptr, &gid) == 0;
  cutensornetNetworkOperator_t op;
  cutensornetCreateNetworkOperator(h, 2, ext, t, &op);
  {
    int32_t nm[1] = {1}, q0[1] = {0};
    const int32_t* mp[1] = {q0};
    const void* dat[1] = {dm};
    int64_t cid;
    cutensornetNetworkOperatorAppendProduct(h, op, make_cuDoubleComplex(1, 0), 1, nm, mp, nullptr, dat, &cid);
  }
  cutensornetStateExpectation_t ex;
  cutensornetCreateExpectation(h, st, op, &ex);
  WS w;
  cutensornetExpectationPrepare(h, ex, (size_t)1 << 30, w.d, 0);
  w.alloc();
  const auto eb = scalar(Ebar), nb = scalar(Nbar);
  alignas(16) unsigned char ev[16] = {0}, nv[16] = {0};
  ok = ok && cutensornetExpectationComputeWithGradientsBackward(h, ex, 0, eb.data(), nb.data(), w.d, ev, nv, 0) == 0;
  std::vector<unsigned char> gb(4 * es);
  cudaDeviceSynchronize();
  cudaMemcpy(gb.data(), dg, gb.size(), cudaMemcpyDeviceToHost);
  gradient.clear();
  for (size_t i = 0; i < 4; ++i) gradient.push_back(unscalar(&gb[i * es]));
  E = unscalar(ev);
  cutensornetDestroyExpectation(ex);
  cutensornetDestroyNetworkOperator(op);
  cutensornetDestroyState(st);
  cudaFree(da);
  cudaFree(dm);
  cudaFree(dg);
  return ok;
}

static void test_types() {
  std::printf("-- real and single precision states\n");
  std::vector<cd> ref, got;
  cd Eref, E;
  // the real types agree with the complex double one on data with no imaginary part (the gate's data is real for them)
  for (cudaDataType_t t : {CUDA_C_32F, CUDA_R_64F, CUDA_R_32F}) {
    const bool complex_type = t == CUDA_C_32F;
    const cd Eb = complex_type ? cd(1.0, 0.5) : cd(1.5, 0.0), Nb = complex_type ? cd(1.0, 0.5) : cd(1.5, 0.0);
    bool ok = one_gate(CUDA_C_64F, Eb, Nb, ref, Eref);
    if (!complex_type) {  // the complex double reference has an imaginary part in its gate: use the real part of a real gate by comparing with the real double type
      ok = one_gate(CUDA_R_64F, Eb, Nb, ref, Eref);
    }
    ok = one_gate(t, Eb, Nb, got, E) && ok;
    check(ok, (std::string("the gradient of a state in type ") + std::to_string((int)t) + " is computed").c_str());
    check(rel(got, ref) < 1e-5 && std::abs(E - Eref) < 1e-5 * (1 + std::abs(Eref)),
          (std::string("and agrees with double precision (type ") + std::to_string((int)t) + ")").c_str());
    if (!complex_type) {
      bool real = true;
      for (cd x : got) real = real && x.imag() == 0;
      check(real, (std::string("a real state's gradient is real (type ") + std::to_string((int)t) + ")").c_str());
    }
  }
  // measured: Ebar = Nbar = 1.5 on the real gate [0.3 0.2; -0.5 0.7] with Z: the gradient is 1.8 in its first element, zero elsewhere
  one_gate(CUDA_R_64F, cd(1.5, 0), cd(1.5, 0), got, E);
  check(std::abs(got[0].real() - 1.8) < 1e-12 && std::abs(got[1]) + std::abs(got[2]) + std::abs(got[3]) < 1e-12,
        "d(Ebar E + Nbar N)/dA of a real gate with Z: 2 a0 (Ebar + Nbar) in the first element");
}

int main() {
  if (cutensornetCreate(&h) != 0) {
    std::printf("cutensornetCreate failed\n");
    return 1;
  }
  test_convention();
  test_zeros_and_overwrite();
  test_arguments();
  test_types();
  cutensornetDestroy(h);
  std::printf("%s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
