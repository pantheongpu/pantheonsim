// cuStateVec as QuEST's cuQuantum backend calls it: dense gates (both
// layouts, adjoint, controls with chosen values, a matrix in device memory,
// single precision), diagonal gates, controlled index-bit swaps,
// probabilities, projection and Pauli expectation values, plus the argument
// checks and the handle's stream and memory handler. Each result is checked
// against a host reference computed differently (by gathering each output
// amplitude, where the library scatters), on a 4-qubit state whose amplitudes
// all differ so a bit-order mistake cannot hide.
//
// Built against VirtualGPU's own declarations, which follow NVIDIA's ABI, so
// the same program also runs against NVIDIA's libcustatevec on a real GPU:
// that is how the references, and so the simulator's answers, were checked.
#include <cuda_runtime_api.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../../include/vgpu_custatevec.h"

using cd = std::complex<double>;
using cf = std::complex<float>;

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

constexpr int kBits = 4;
constexpr int kN = 1 << kBits;

static std::vector<cd> initial() {
  std::vector<cd> a(kN);
  double norm = 0;
  for (int i = 0; i < kN; ++i) {
    a[i] = cd(i + 1, (i % 3) - 1);
    norm += std::norm(a[i]);
  }
  for (cd& x : a) x /= std::sqrt(norm);
  return a;
}

static bool close(const std::vector<cd>& x, const std::vector<cd>& y, double tol) {
  if (x.size() != y.size()) return false;
  for (size_t i = 0; i < x.size(); ++i)
    if (std::abs(x[i] - y[i]) > tol) {
      std::printf("     [%zu] got (%.17g, %.17g) want (%.17g, %.17g)\n", i, x[i].real(), x[i].imag(),
                  y[i].real(), y[i].imag());
      return false;
    }
  return true;
}

static bool close(double x, double y, double tol) {
  if (std::fabs(x - y) <= tol) return true;
  std::printf("     got %.17g want %.17g\n", x, y);
  return false;
}

// Host reference for a controlled gate. Out[i] is gathered: when i satisfies
// the controls, it is row r of M (r = i's target bits, targets[0] lowest)
// applied to the amplitudes that share i's other bits.
static std::vector<cd> ref_gate(const std::vector<cd>& in, const std::vector<cd>& Mrow,
                                const std::vector<int>& targets, const std::vector<int>& controls,
                                const std::vector<int>& values) {
  const int dim = 1 << targets.size();
  std::vector<cd> out(in.size());
  for (int i = 0; i < kN; ++i) {
    bool on = true;
    for (size_t k = 0; k < controls.size(); ++k)
      if (((i >> controls[k]) & 1) != (values.empty() ? 1 : values[k])) on = false;
    if (!on) { out[i] = in[i]; continue; }
    int r = 0, rest = i;
    for (size_t k = 0; k < targets.size(); ++k) {
      r |= ((i >> targets[k]) & 1) << k;
      rest &= ~(1 << targets[k]);
    }
    cd acc = 0;
    for (int c = 0; c < dim; ++c) {
      int j = rest;
      for (size_t k = 0; k < targets.size(); ++k) j |= ((c >> k) & 1) << targets[k];
      acc += Mrow[r * dim + c] * in[j];
    }
    out[i] = acc;
  }
  return out;
}

static std::vector<cd> transpose_conj(const std::vector<cd>& M, int dim, bool conj) {
  std::vector<cd> T(M.size());
  for (int r = 0; r < dim; ++r)
    for (int c = 0; c < dim; ++c) T[c * dim + r] = conj ? std::conj(M[r * dim + c]) : M[r * dim + c];
  return T;
}

struct Dev {
  void* p = nullptr;
  Dev(size_t bytes) { cudaMalloc(&p, bytes); }
  ~Dev() { cudaFree(p); }
};

static void put(void* d, const std::vector<cd>& a) { cudaMemcpy(d, a.data(), a.size() * sizeof(cd), cudaMemcpyHostToDevice); }
static std::vector<cd> get(const void* d) {
  std::vector<cd> a(kN);
  cudaMemcpy(a.data(), d, kN * sizeof(cd), cudaMemcpyDeviceToHost);
  return a;
}

static int alloc_cb(void*, void** ptr, size_t size, cudaStream_t) { return (int)cudaMalloc(ptr, size); }
static int free_cb(void*, void* ptr, size_t, cudaStream_t) { return (int)cudaFree(ptr); }

int main() {
  custatevecHandle_t h = nullptr;
  IS(custatevecCreate(&h), CUSTATEVEC_STATUS_SUCCESS);
  if (!h) { std::printf("FAIL\n"); return 1; }

  // Library management.
  int32_t major = -1, minor = -1;
  IS(custatevecGetProperty(MAJOR_VERSION, &major), CUSTATEVEC_STATUS_SUCCESS);
  IS(custatevecGetProperty(MINOR_VERSION, &minor), CUSTATEVEC_STATUS_SUCCESS);
  check(major == 1 && (size_t)(major * 10000 + minor * 100) <= custatevecGetVersion(),
        "GetProperty and GetVersion agree on the version");
  check(std::strcmp(custatevecGetErrorName(CUSTATEVEC_STATUS_INVALID_VALUE),
                    "CUSTATEVEC_STATUS_INVALID_VALUE") == 0, "GetErrorName names a status");
  cudaStream_t s = nullptr, got = nullptr;
  cudaStreamCreate(&s);
  IS(custatevecSetStream(h, s), CUSTATEVEC_STATUS_SUCCESS);
  IS(custatevecGetStream(h, &got), CUSTATEVEC_STATUS_SUCCESS);
  check(got == s, "GetStream returns the stream SetStream set");
  custatevecDeviceMemHandler_t mh{}, back{};
  mh.device_alloc = alloc_cb;
  mh.device_free = free_cb;
  std::strcpy(mh.name, "test pool");
  IS(custatevecSetDeviceMemHandler(h, &mh), CUSTATEVEC_STATUS_SUCCESS);
  IS(custatevecGetDeviceMemHandler(h, &back), CUSTATEVEC_STATUS_SUCCESS);
  check(back.device_alloc == alloc_cb && std::strcmp(back.name, "test pool") == 0,
        "GetDeviceMemHandler returns the handler SetDeviceMemHandler set");

  const std::vector<cd> psi = initial();
  Dev sv(kN * sizeof(cd));

  // Dense gates. A deliberately non-unitary, non-symmetric matrix, so layout,
  // adjoint and target order each change the answer.
  {
    const std::vector<cd> M = {cd(1, 2), cd(3, -1), cd(0.5, 0), cd(0, -2)};
    put(sv.p, psi);
    int t = 2;
    IS(custatevecApplyMatrix(h, sv.p, CUDA_C_64F, kBits, M.data(), CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                             0, &t, 1, nullptr, nullptr, 0, CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),
       CUSTATEVEC_STATUS_SUCCESS);
    check(close(get(sv.p), ref_gate(psi, M, {2}, {}, {}), 1e-12), "one target, row-major");

    put(sv.p, psi);
    IS(custatevecApplyMatrix(h, sv.p, CUDA_C_64F, kBits, M.data(), CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_COL,
                             0, &t, 1, nullptr, nullptr, 0, CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),
       CUSTATEVEC_STATUS_SUCCESS);
    check(close(get(sv.p), ref_gate(psi, transpose_conj(M, 2, false), {2}, {}, {}), 1e-12),
          "one target, column-major");

    put(sv.p, psi);
    IS(custatevecApplyMatrix(h, sv.p, CUDA_C_64F, kBits, M.data(), CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                             1, &t, 1, nullptr, nullptr, 0, CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),
       CUSTATEVEC_STATUS_SUCCESS);
    check(close(get(sv.p), ref_gate(psi, transpose_conj(M, 2, true), {2}, {}, {}), 1e-12), "adjoint");
  }
  {
    // Two targets given high bit first, one control that must be 0.
    std::vector<cd> M(16);
    for (int k = 0; k < 16; ++k) M[k] = cd(0.25 * (k + 1), 0.1 * (7 - k));
    const int targets[] = {3, 1}, controls[] = {0}, values[] = {0};
    put(sv.p, psi);
    IS(custatevecApplyMatrix(h, sv.p, CUDA_C_64F, kBits, M.data(), CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                             0, targets, 2, controls, values, 1, CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),
       CUSTATEVEC_STATUS_SUCCESS);
    check(close(get(sv.p), ref_gate(psi, M, {3, 1}, {0}, {0}), 1e-12),
          "two targets (targets[0] is the low bit), control on 0");

    // The same matrix from device memory.
    Dev dm(M.size() * sizeof(cd));
    cudaMemcpy(dm.p, M.data(), M.size() * sizeof(cd), cudaMemcpyHostToDevice);
    put(sv.p, psi);
    IS(custatevecApplyMatrix(h, sv.p, CUDA_C_64F, kBits, dm.p, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                             0, targets, 2, controls, values, 1, CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),
       CUSTATEVEC_STATUS_SUCCESS);
    check(close(get(sv.p), ref_gate(psi, M, {3, 1}, {0}, {0}), 1e-12), "matrix in device memory");

    // Controls with no values array mean 1.
    put(sv.p, psi);
    const int ctl2[] = {0, 2};
    IS(custatevecApplyMatrix(h, sv.p, CUDA_C_64F, kBits, M.data(), CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                             0, targets, 2, ctl2, nullptr, 2, CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),
       CUSTATEVEC_STATUS_SUCCESS);
    check(close(get(sv.p), ref_gate(psi, M, {3, 1}, {0, 2}, {}), 1e-12), "controls default to 1");
  }
  {
    // Single precision state and matrix.
    const std::vector<cf> M = {cf(0, 1), cf(1, 0), cf(1, 0), cf(0, -1)};
    std::vector<cf> f(kN);
    for (int i = 0; i < kN; ++i) f[i] = cf((float)psi[i].real(), (float)psi[i].imag());
    Dev sf(kN * sizeof(cf));
    cudaMemcpy(sf.p, f.data(), kN * sizeof(cf), cudaMemcpyHostToDevice);
    int t = 1;
    IS(custatevecApplyMatrix(h, sf.p, CUDA_C_32F, kBits, M.data(), CUDA_C_32F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                             0, &t, 1, nullptr, nullptr, 0, CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),
       CUSTATEVEC_STATUS_SUCCESS);
    cudaMemcpy(f.data(), sf.p, kN * sizeof(cf), cudaMemcpyDeviceToHost);
    std::vector<cd> fd(kN), Md = {cd(0, 1), cd(1, 0), cd(1, 0), cd(0, -1)};
    for (int i = 0; i < kN; ++i) fd[i] = cd(f[i].real(), f[i].imag());
    check(close(fd, ref_gate(psi, Md, {1}, {}, {}), 1e-6), "single precision");
  }

  // Argument checks.
  {
    const std::vector<cd> M(16, cd(1, 0));
    const int dup[] = {1, 1}, t[] = {1, 2}, c[] = {2};
    put(sv.p, psi);
    IS(custatevecApplyMatrix(h, sv.p, CUDA_C_64F, kBits, M.data(), CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                             0, dup, 2, nullptr, nullptr, 0, CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),
       CUSTATEVEC_STATUS_INVALID_VALUE);
    IS(custatevecApplyMatrix(h, sv.p, CUDA_C_64F, kBits, M.data(), CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                             0, t, 2, c, nullptr, 1, CUSTATEVEC_COMPUTE_DEFAULT, nullptr, 0),
       CUSTATEVEC_STATUS_INVALID_VALUE);
    check(close(get(sv.p), psi, 0), "a refused call leaves the state alone");
  }

  // Diagonal gates through the generalized permutation matrix, no permutation.
  {
    const std::vector<cd> d = {cd(1, 0), cd(0, 1), cd(-0.5, 0.5), cd(2, -1)};
    const int targets[] = {1, 3}, controls[] = {2}, values[] = {1};
    put(sv.p, psi);
    IS(custatevecApplyGeneralizedPermutationMatrix(h, sv.p, CUDA_C_64F, kBits, nullptr, d.data(), CUDA_C_64F,
                                                   1, targets, 2, controls, values, 1, nullptr, 0),
       CUSTATEVEC_STATUS_SUCCESS);
    std::vector<cd> D(16, cd(0, 0));
    for (int k = 0; k < 4; ++k) D[k * 4 + k] = std::conj(d[k]);
    check(close(get(sv.p), ref_gate(psi, D, {1, 3}, {2}, {1}), 1e-12), "diagonal, adjoint, controlled");
  }

  // Controlled index-bit swap: bits 0 and 3 exchanged where bit 1 is 1.
  {
    put(sv.p, psi);
    const int2 swaps[] = {{0, 3}};
    const int32_t maskBits[] = {1}, maskOrder[] = {1};
    IS(custatevecSwapIndexBits(h, sv.p, CUDA_C_64F, kBits, swaps, 1, maskBits, maskOrder, 1),
       CUSTATEVEC_STATUS_SUCCESS);
    const std::vector<cd> SWAP = {1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1};
    check(close(get(sv.p), ref_gate(psi, SWAP, {0, 3}, {1}, {1}), 1e-15), "swap index bits under a mask");
  }

  // Probabilities.
  put(sv.p, psi);
  {
    double p[4] = {-1, -1, -1, -1};
    const int32_t order[] = {2, 0}, maskBits[] = {0}, maskOrder[] = {3};
    IS(custatevecAbs2SumArray(h, sv.p, CUDA_C_64F, kBits, p, order, 2, maskBits, maskOrder, 1),
       CUSTATEVEC_STATUS_SUCCESS);
    bool ok = true;
    for (int k = 0; k < 4; ++k) {
      double want = 0;
      for (int i = 0; i < kN; ++i)
        if (((i >> 3) & 1) == 0 && ((i >> 2) & 1) == (k & 1) && ((i >> 0) & 1) == (k >> 1))
          want += std::norm(psi[i]);
      ok = close(p[k], want, 1e-14) && ok;
    }
    check(ok, "Abs2SumArray: output bit k is bitOrdering[k], under a mask");

    Dev dp(4 * sizeof(double));
    IS(custatevecAbs2SumArray(h, sv.p, CUDA_C_64F, kBits, (double*)dp.p, order, 2, maskBits, maskOrder, 1),
       CUSTATEVEC_STATUS_SUCCESS);
    double q[4];
    cudaMemcpy(q, dp.p, sizeof q, cudaMemcpyDeviceToHost);
    check(std::memcmp(p, q, sizeof q) == 0, "Abs2SumArray into device memory");

    // QuEST's probability of one outcome: no bit ordering, all bits masked.
    double one = -1;
    const int32_t outcome[] = {1, 0}, qubits[] = {0, 3};
    IS(custatevecAbs2SumArray(h, sv.p, CUDA_C_64F, kBits, &one, nullptr, 0, outcome, qubits, 2),
       CUSTATEVEC_STATUS_SUCCESS);
    double want = 0;
    for (int i = 0; i < kN; ++i)
      if ((i & 1) == 1 && ((i >> 3) & 1) == 0) want += std::norm(psi[i]);
    check(close(one, want, 1e-14), "Abs2SumArray of a single outcome");
  }
  {
    double p0 = -1, p1 = -1;
    const int32_t basis[] = {0, 3};
    IS(custatevecAbs2SumOnZBasis(h, sv.p, CUDA_C_64F, kBits, &p0, &p1, basis, 2), CUSTATEVEC_STATUS_SUCCESS);
    double w0 = 0, w1 = 0;
    for (int i = 0; i < kN; ++i) ((((i >> 0) ^ (i >> 3)) & 1) ? w1 : w0) += std::norm(psi[i]);
    check(close(p0, w0, 1e-14) && close(p1, w1, 1e-14), "Abs2SumOnZBasis splits by parity");
  }

  // Projection onto bits 1 = 1 and 3 = 0, renormalised by that outcome's probability.
  {
    const int32_t bits[] = {1, 0}, order[] = {1, 3};
    double prob = 0;
    for (int i = 0; i < kN; ++i)
      if (((i >> 1) & 1) == 1 && ((i >> 3) & 1) == 0) prob += std::norm(psi[i]);
    put(sv.p, psi);
    IS(custatevecCollapseByBitString(h, sv.p, CUDA_C_64F, kBits, bits, order, 2, prob), CUSTATEVEC_STATUS_SUCCESS);
    std::vector<cd> want(kN);
    for (int i = 0; i < kN; ++i)
      want[i] = (((i >> 1) & 1) == 1 && ((i >> 3) & 1) == 0) ? psi[i] / std::sqrt(prob) : cd(0, 0);
    check(close(get(sv.p), want, 1e-14), "CollapseByBitString");
  }

  // Pauli expectation values, against <psi| (P|psi>) built gate by gate.
  {
    put(sv.p, psi);
    const custatevecPauli_t t0[] = {CUSTATEVEC_PAULI_X, CUSTATEVEC_PAULI_Y, CUSTATEVEC_PAULI_Z};
    const custatevecPauli_t t1[] = {CUSTATEVEC_PAULI_Z};
    const custatevecPauli_t t2[] = {CUSTATEVEC_PAULI_I, CUSTATEVEC_PAULI_Y, CUSTATEVEC_PAULI_Y};
    const int32_t b0[] = {0, 2, 3}, b1[] = {1}, b2[] = {0, 1, 3};
    const custatevecPauli_t* ops[] = {t0, t1, t2};
    const int32_t* bits[] = {b0, b1, b2};
    const uint32_t lens[] = {3, 1, 3};
    double e[3] = {0, 0, 0};
    IS(custatevecComputeExpectationsOnPauliBasis(h, sv.p, CUDA_C_64F, kBits, e, ops, 3, bits, lens),
       CUSTATEVEC_STATUS_SUCCESS);
    const std::vector<cd> I = {1, 0, 0, 1}, X = {0, 1, 1, 0}, Y = {0, cd(0, -1), cd(0, 1), 0}, Z = {1, 0, 0, -1};
    auto expect = [&](std::vector<std::pair<const std::vector<cd>*, int>> term) {
      std::vector<cd> v = psi;
      for (auto& [m, b] : term) v = ref_gate(v, *m, {b}, {}, {});
      cd acc = 0;
      for (int i = 0; i < kN; ++i) acc += std::conj(psi[i]) * v[i];
      return acc.real();
    };
    bool ok = close(e[0], expect({{&X, 0}, {&Y, 2}, {&Z, 3}}), 1e-14);
    ok = close(e[1], expect({{&Z, 1}}), 1e-14) && ok;
    ok = close(e[2], expect({{&I, 0}, {&Y, 1}, {&Y, 3}}), 1e-14) && ok;
    check(ok, "ComputeExpectationsOnPauliBasis, three terms");
  }

  IS(custatevecDestroy(h), CUSTATEVEC_STATUS_SUCCESS);
  cudaStreamDestroy(s);
  std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
