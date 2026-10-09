// cuTENSOR: contractions (binary and trinary) in every type and compute
// combination an RTX 3060 takes, permutations with their type conversions,
// unary operators and padding, elementwise binary and trinary operations,
// reductions, plan preferences, the plan cache, workspace estimation, graph
// capture, and the argument checks.
//
// The expectations are NVIDIA's: this program also runs against NVIDIA's
// libcutensor 2.8.1 on an RTX 3060, and every status, default, scalar type,
// FLOP count and padded layout below is what that library answered. Values
// that are the library's own choice -- workspace estimates, which plans the
// plan cache keeps -- are only checked for what the documentation promises.
// Inputs are small integers or quarters, which every element type and
// compute precision represents exactly, so results compare exactly.
#include <cuda_runtime_api.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

#include "../../include/vgpu_cutensor.h"

using cd = std::complex<double>;

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

static cutensorHandle_t h;
static cutensorPlanPreference_t pref;

static const cutensorOperator_t ID = CUTENSOR_OP_IDENTITY;

// ---- element conversion on the host ----

static uint16_t to_half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000, ax = x & 0x7fffffff;
  if (ax < 0x38800000) return (uint16_t)(sign | (uint32_t)std::nearbyint(std::fabs(f) * 16777216.0f));
  uint32_t m = ax + 0xfff + ((ax >> 13) & 1);
  m -= 0x38000000;
  return (uint16_t)(sign | (m >> 13));
}
static double from_half(uint16_t h16) {
  const int s = h16 >> 15, e = (h16 >> 10) & 0x1f, m = h16 & 0x3ff;
  double v = e == 0 ? std::ldexp((double)m, -24) : std::ldexp((double)(m | 0x400), e - 25);
  return s ? -v : v;
}
static uint16_t to_bf16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  x += 0x7fff + ((x >> 16) & 1);
  return (uint16_t)(x >> 16);
}
static double from_bf16(uint16_t b) {
  uint32_t x = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

static size_t esize(cudaDataType_t t) {
  switch (t) {
    case CUDA_R_16F: case CUDA_R_16BF: return 2;
    case CUDA_R_32F: return 4;
    case CUDA_R_64F: case CUDA_C_32F: return 8;
    case CUDA_C_64F: return 16;
    default: return 1;
  }
}
static bool cplx(cudaDataType_t t) { return t == CUDA_C_32F || t == CUDA_C_64F; }

static std::vector<unsigned char> encode(const std::vector<cd>& v, cudaDataType_t t) {
  std::vector<unsigned char> b(v.size() * esize(t));
  for (size_t i = 0; i < v.size(); ++i) {
    unsigned char* p = b.data() + i * esize(t);
    switch (t) {
      case CUDA_R_16F: { uint16_t x = to_half((float)v[i].real()); std::memcpy(p, &x, 2); break; }
      case CUDA_R_16BF: { uint16_t x = to_bf16((float)v[i].real()); std::memcpy(p, &x, 2); break; }
      case CUDA_R_32F: { float x = (float)v[i].real(); std::memcpy(p, &x, 4); break; }
      case CUDA_R_64F: { double x = v[i].real(); std::memcpy(p, &x, 8); break; }
      case CUDA_C_32F: { float x[2] = {(float)v[i].real(), (float)v[i].imag()}; std::memcpy(p, x, 8); break; }
      case CUDA_C_64F: { double x[2] = {v[i].real(), v[i].imag()}; std::memcpy(p, x, 16); break; }
      default: break;
    }
  }
  return b;
}
static std::vector<cd> decode(const std::vector<unsigned char>& b, cudaDataType_t t) {
  std::vector<cd> v(b.size() / esize(t));
  for (size_t i = 0; i < v.size(); ++i) {
    const unsigned char* p = b.data() + i * esize(t);
    switch (t) {
      case CUDA_R_16F: { uint16_t x; std::memcpy(&x, p, 2); v[i] = from_half(x); break; }
      case CUDA_R_16BF: { uint16_t x; std::memcpy(&x, p, 2); v[i] = from_bf16(x); break; }
      case CUDA_R_32F: { float x; std::memcpy(&x, p, 4); v[i] = x; break; }
      case CUDA_R_64F: { double x; std::memcpy(&x, p, 8); v[i] = x; break; }
      case CUDA_C_32F: { float x[2]; std::memcpy(x, p, 8); v[i] = cd(x[0], x[1]); break; }
      case CUDA_C_64F: { double x[2]; std::memcpy(x, p, 16); v[i] = cd(x[0], x[1]); break; }
      default: break;
    }
  }
  return v;
}

static void* upload(const std::vector<cd>& v, cudaDataType_t t) {
  std::vector<unsigned char> b = encode(v, t);
  void* p = nullptr;
  cudaMalloc(&p, b.size() + 256);
  cudaMemcpy(p, b.data(), b.size(), cudaMemcpyHostToDevice);
  return p;
}
static std::vector<cd> download(const void* p, size_t n, cudaDataType_t t) {
  std::vector<unsigned char> b(n * esize(t));
  cudaDeviceSynchronize();
  cudaMemcpy(b.data(), p, b.size(), cudaMemcpyDeviceToHost);
  return decode(b, t);
}

// Small integers (and, for complex types, integer imaginary parts).
static std::vector<cd> ints(size_t n, int seed, bool complex_values) {
  std::vector<cd> v(n);
  unsigned s = 2463534242u + (unsigned)seed * 77u;
  for (size_t i = 0; i < n; ++i) {
    s ^= s << 13, s ^= s >> 17, s ^= s << 5;
    const double re = (double)((int)(s % 7) - 3);
    s ^= s << 13, s ^= s >> 17, s ^= s << 5;
    const double im = complex_values ? (double)((int)(s % 5) - 2) : 0.0;
    v[i] = cd(re, im);
  }
  return v;
}

static double maxdiff(const std::vector<cd>& a, const std::vector<cd>& b) {
  double d = 0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) d = std::fmax(d, std::abs(a[i] - b[i]));
  return a.size() == b.size() ? d : 1e300;
}

// ---- ownership: everything created is destroyed, so the program runs clean
//      under LeakSanitizer ----

// A variable that is handed to Create calls again and again: each time its
// address is taken, the previous object is kept for destruction at exit.
template <class H, cutensorStatus_t (*Destroy)(H)>
struct Owned {
  H d = nullptr;
  std::vector<H> old;
  H* operator&() {
    if (d) old.push_back(d);
    d = nullptr;
    return &d;
  }
  operator H() const { return d; }
  ~Owned() {
    if (d) old.push_back(d);
    for (H x : old) Destroy(x);
  }
};
using OpVar = Owned<cutensorOperationDescriptor_t, cutensorDestroyOperationDescriptor>;
using TdVar = Owned<cutensorTensorDescriptor_t, cutensorDestroyTensorDescriptor>;
using BsVar = Owned<cutensorBlockSparseTensorDescriptor_t, cutensorDestroyBlockSparseTensorDescriptor>;

static std::vector<cutensorTensorDescriptor_t> g_descs;
static std::vector<cutensorPlan_t> g_plans;
static std::vector<void*> g_buffers;

static void release_all() {
  for (cutensorPlan_t p : g_plans) cutensorDestroyPlan(p);
  for (cutensorTensorDescriptor_t d : g_descs) cutensorDestroyTensorDescriptor(d);
  for (void* b : g_buffers) cudaFree(b);
  g_plans.clear(), g_descs.clear(), g_buffers.clear();
}

static cutensorTensorDescriptor_t desc(cudaDataType_t t, std::vector<int64_t> e, std::vector<int64_t> s = {},
                                       uint32_t align = 256) {
  cutensorTensorDescriptor_t d = nullptr;
  if (cutensorCreateTensorDescriptor(h, &d, (uint32_t)e.size(), e.data(), s.empty() ? nullptr : s.data(), t, align))
    std::printf("     (descriptor refused)\n");
  else
    g_descs.push_back(d);
  return d;
}

static std::vector<unsigned char> scalar_bytes(cudaDataType_t t, cd v) { return encode({v}, t); }

// The scalar type the operation reports, then a plan.
static cudaDataType_t scalar_of(cutensorOperationDescriptor_t op) {
  cudaDataType_t st = CUDA_R_8I;
  cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, &st, sizeof st);
  return st;
}

struct Planned {
  cutensorPlan_t plan = nullptr;
  void* ws = nullptr;
  uint64_t wsize = 0;
  int status = -1;
};

static Planned plan(cutensorOperationDescriptor_t op) {
  Planned p;
  cutensorEstimateWorkspaceSize(h, op, pref, CUTENSOR_WORKSPACE_DEFAULT, &p.wsize);
  p.status = cutensorCreatePlan(h, &p.plan, op, pref, p.wsize);
  if (p.status == 0) {
    g_plans.push_back(p.plan);
    uint64_t need = ~0ull;
    cutensorPlanGetAttribute(h, p.plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &need, sizeof need);
    if (need > p.wsize) std::printf("     (plan needs more than the estimate)\n");
    if (p.wsize) {
      cudaMalloc(&p.ws, p.wsize);
      g_buffers.push_back(p.ws);
    }
  }
  return p;
}

// ---- host references ----

// Column-major index of element `idx` (a mode -> index map in `modes` order).
static int64_t offset(const std::vector<int32_t>& modes, const std::vector<int64_t>& strides,
                      const std::vector<int32_t>& all, const std::vector<int64_t>& idx) {
  int64_t o = 0;
  for (size_t i = 0; i < modes.size(); ++i)
    for (size_t k = 0; k < all.size(); ++k)
      if (all[k] == modes[i]) o += idx[k] * strides[i];
  return o;
}

static std::vector<int64_t> packed(const std::vector<int64_t>& e) {
  std::vector<int64_t> s(e.size());
  int64_t p = 1;
  for (size_t i = 0; i < e.size(); ++i) s[i] = p, p *= e[i];
  return s;
}

// ---- contraction across types and computes ----

static void contraction_types() {
  struct Combo {
    cudaDataType_t a, b, c;
    cutensorComputeDescriptor_t comp;
    const char* name;
    cudaDataType_t scalar;
    int plan;  // the plan status on an RTX 3060
  };
  const cutensorComputeDescriptor_t C16F = CUTENSOR_COMPUTE_DESC_16F, C16BF = CUTENSOR_COMPUTE_DESC_16BF,
                                    CTF32 = CUTENSOR_COMPUTE_DESC_TF32, C3X = CUTENSOR_COMPUTE_DESC_3XTF32,
                                    C32 = CUTENSOR_COMPUTE_DESC_32F, C64 = CUTENSOR_COMPUTE_DESC_64F,
                                    C9X = CUTENSOR_COMPUTE_DESC_9X16BF, C8X = CUTENSOR_COMPUTE_DESC_8XINT8,
                                    C4X = CUTENSOR_COMPUTE_DESC_4X16F;
  const Combo combos[] = {
      {CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, C32, "R16F 32F", CUDA_R_32F, 0},
      {CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, C16F, "R16F 16F", CUDA_R_32F, 0},
      {CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, C64, "R16F 64F", CUDA_R_64F, 15},
      {CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, C32, "R16BF 32F", CUDA_R_32F, 0},
      {CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, C16BF, "R16BF 16BF", CUDA_R_32F, 0},
      {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, C32, "R32F 32F", CUDA_R_32F, 0},
      {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CTF32, "R32F TF32", CUDA_R_32F, 0},
      {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, C3X, "R32F 3XTF32", CUDA_R_32F, 0},
      {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, C16BF, "R32F 16BF", CUDA_R_32F, 0},
      {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, C16F, "R32F 16F", CUDA_R_32F, 0},
      {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, C9X, "R32F 9X16BF", CUDA_R_32F, 0},
      {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, C4X, "R32F 4X16F", CUDA_R_32F, 0},
      {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, C64, "R32F 64F", CUDA_R_64F, 15},
      {CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, C64, "R64F 64F", CUDA_R_64F, 0},
      {CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, C32, "R64F 32F", CUDA_R_64F, 0},
      {CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, C8X, "R64F 8XINT8", CUDA_R_64F, 0},
      {CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, C32, "C32F 32F", CUDA_C_32F, 0},
      {CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CTF32, "C32F TF32", CUDA_C_32F, 0},
      {CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, C3X, "C32F 3XTF32", CUDA_C_32F, 0},
      {CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, C64, "C32F 64F", CUDA_C_64F, 15},
      {CUDA_C_64F, CUDA_C_64F, CUDA_C_64F, C64, "C64F 64F", CUDA_C_64F, 0},
      {CUDA_C_64F, CUDA_C_64F, CUDA_C_64F, C32, "C64F 32F", CUDA_C_64F, 0},
      {CUDA_R_64F, CUDA_C_64F, CUDA_C_64F, C64, "R64F x C64F", CUDA_C_64F, 0},
      {CUDA_C_64F, CUDA_R_64F, CUDA_C_64F, C64, "C64F x R64F", CUDA_C_64F, 0},
      {CUDA_R_32F, CUDA_C_32F, CUDA_C_32F, C32, "R32F x C32F", CUDA_C_32F, 15},
  };
  // D[m,n,b] = alpha A[k,m,b] B[n,k,b] + beta C[m,n,b], with a strided A.
  const int64_t M = 5, N = 3, K = 4, Bt = 2;
  const std::vector<int32_t> ma = {'k', 'm', 'b'}, mb = {'n', 'k', 'b'}, mc = {'m', 'n', 'b'};
  const std::vector<int64_t> ea = {K, M, Bt}, eb = {N, K, Bt}, ec = {M, N, Bt};
  const std::vector<int64_t> sa = {1, K + 1, (K + 1) * M};  // a padded leading dimension
  const std::vector<int32_t> all = {'m', 'n', 'b', 'k'};
  for (const Combo& c : combos) {
    std::string name = std::string("contraction ") + c.name;
    auto dA = desc(c.a, ea, sa), dB = desc(c.b, eb), dC = desc(c.c, ec);
    OpVar op;
    const int cs = cutensorCreateContraction(h, &op, dA, ma.data(), ID, dB, mb.data(), ID, dC, mc.data(), ID, dC,
                                             mc.data(), c.comp);
    check(cs == 0, (name + ": created").c_str());
    if (cs) continue;
    check(scalar_of(op) == c.scalar, (name + ": scalar type").c_str());
    Planned p = plan(op);
    check(p.status == c.plan, (name + ": plan status " + std::to_string(c.plan)).c_str());
    if (p.status) {
      continue;
    }
    const size_t na = (size_t)(sa[2] * Bt), nb = (size_t)(N * K * Bt), nc = (size_t)(M * N * Bt);
    std::vector<cd> a = ints(na, 1, cplx(c.a)), b = ints(nb, 2, cplx(c.b)), cc = ints(nc, 3, cplx(c.c));
    void *pa = upload(a, c.a), *pb = upload(b, c.b), *pc = upload(cc, c.c);
    const cd alpha = 2, beta = -1;
    auto al = scalar_bytes(c.scalar, alpha), be = scalar_bytes(c.scalar, beta);
    IS(cutensorContract(h, p.plan, al.data(), pa, pb, be.data(), pc, pc, p.ws, p.wsize, 0), CUTENSOR_STATUS_SUCCESS);
    std::vector<cd> got = download(pc, nc, c.c), want(nc);
    for (int64_t bb = 0; bb < Bt; ++bb)
      for (int64_t n = 0; n < N; ++n)
        for (int64_t m = 0; m < M; ++m) {
          cd s = 0;
          for (int64_t k = 0; k < K; ++k) {
            std::vector<int64_t> idx = {m, n, bb, k};
            s += a[(size_t)offset(ma, sa, all, idx)] * b[(size_t)offset(mb, packed(eb), all, idx)];
          }
          const size_t o = (size_t)(m + M * (n + N * bb));
          want[o] = alpha * s + beta * cc[o];
        }
    check(maxdiff(got, want) == 0, (name + ": values exact").c_str());
    cudaFree(pa), cudaFree(pb), cudaFree(pc);
  }
  // C16F contracts nowhere on the 3060 (INTERNAL_ERROR at planning), and
  // 8XINT8 refuses a non-double output at creation (INTERNAL_ERROR).
  {
    auto d16 = desc(CUDA_C_16F, {4, 4});
    OpVar op;
    const int32_t m1[] = {'i', 'k'}, m2[] = {'k', 'j'}, m3[] = {'i', 'j'};
    IS(cutensorCreateContraction(h, &op, d16, m1, ID, d16, m2, ID, d16, m3, ID, d16, m3, CUTENSOR_COMPUTE_DESC_32F), 0);
    Planned p = plan(op);
    IS(p.status, CUTENSOR_STATUS_INTERNAL_ERROR);
    auto d32 = desc(CUDA_R_32F, {4, 4});
    IS(cutensorCreateContraction(h, &op, d32, m1, ID, d32, m2, ID, d32, m3, ID, d32, m3, CUTENSOR_COMPUTE_DESC_8XINT8),
       CUTENSOR_STATUS_INTERNAL_ERROR);
  }
}

// ---- contraction semantics and checks ----

static void contraction_checks() {
  const cutensorComputeDescriptor_t C32 = CUTENSOR_COMPUTE_DESC_32F;
  auto A = desc(CUDA_R_32F, {4, 6}), B = desc(CUDA_R_32F, {6, 5}), C = desc(CUDA_R_32F, {4, 5});
  auto C2 = desc(CUDA_R_32F, {4, 5}, {1, 8}), D64 = desc(CUDA_R_64F, {4, 5}), Bx = desc(CUDA_R_32F, {7, 5});
  auto A0 = desc(CUDA_R_32F, {4, 6}, {}, 0);
  const int32_t mk[] = {'m', 'k'}, kn[] = {'k', 'n'}, mn[] = {'m', 'n'}, nm[] = {'n', 'm'}, kx[] = {'k', 'x'};
  OpVar op;
  IS(cutensorCreateContraction(nullptr, &op, A, mk, ID, B, kn, ID, C, mn, ID, C, mn, C32), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorCreateContraction(h, nullptr, A, mk, ID, B, kn, ID, C, mn, ID, C, mn, C32), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorCreateContraction(h, &op, nullptr, mk, ID, B, kn, ID, C, mn, ID, C, mn, C32), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorCreateContraction(h, &op, A, nullptr, ID, B, kn, ID, C, mn, ID, C, mn, C32), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorCreateContraction(h, &op, A, mk, ID, B, kn, ID, C, mn, ID, C, mn, nullptr), CUTENSOR_STATUS_INVALID_VALUE);
  // C and D must agree (a current limitation: NOT_SUPPORTED).
  IS(cutensorCreateContraction(h, &op, A, mk, ID, B, kn, ID, C, mn, ID, C, nm, C32), CUTENSOR_STATUS_NOT_SUPPORTED);
  IS(cutensorCreateContraction(h, &op, A, mk, ID, B, kn, ID, C, mn, ID, C2, mn, C32), CUTENSOR_STATUS_NOT_SUPPORTED);
  IS(cutensorCreateContraction(h, &op, A, mk, ID, B, kn, ID, C, mn, ID, D64, mn, C32), CUTENSOR_STATUS_NOT_SUPPORTED);
  // A mode only the output has.
  IS(cutensorCreateContraction(h, &op, A, mk, ID, B, kx, ID, C, mn, ID, C, mn, C32), CUTENSOR_STATUS_INVALID_VALUE);
  // Operators: opC the identity; opA, opB the identity or (complex) CONJ.
  IS(cutensorCreateContraction(h, &op, A, mk, ID, B, kn, ID, C, mn, CUTENSOR_OP_CONJ, C, mn, C32), CUTENSOR_STATUS_NOT_SUPPORTED);
  IS(cutensorCreateContraction(h, &op, A, mk, CUTENSOR_OP_SQRT, B, kn, ID, C, mn, ID, C, mn, C32), CUTENSOR_STATUS_NOT_SUPPORTED);
  IS(cutensorCreateContraction(h, &op, A, mk, CUTENSOR_OP_CONJ, B, kn, ID, C, mn, ID, C, mn, C32), CUTENSOR_STATUS_NOT_SUPPORTED);
  // An alignment left at 0 is refused by a contraction (and taken elsewhere).
  IS(cutensorCreateContraction(h, &op, A0, mk, ID, B, kn, ID, C, mn, ID, C, mn, C32), CUTENSOR_STATUS_INVALID_VALUE);
  // Extents that disagree: created, then refused at planning.
  IS(cutensorCreateContraction(h, &op, A, mk, ID, Bx, kn, ID, C, mn, ID, C, mn, C32), 0);
  IS(plan(op).status, CUTENSOR_STATUS_INTERNAL_ERROR);

  // Attributes of D[m,n] = A[m,k] B[k,n] (m4 n5 k6).
  IS(cutensorCreateContraction(h, &op, A, mk, ID, B, kn, ID, C, mn, ID, C, mn, C32), 0);
  float flops = 0, bytes = 0;
  int32_t tag = -1;
  IS(cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_FLOPS, &flops, 4), 0);
  check(flops == 240, "contraction FLOPS = 2 m n k = 240");
  IS(cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES, &bytes, 4), 0);
  check(bytes == 376, "contraction MOVED_BYTES = (A + B + C + D) * 4 = 376");
  IS(cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_TAG, &tag, 4), 0);
  check(tag == 0, "the tag defaults to 0");
  IS(cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_TAG, &tag, 8), CUTENSOR_STATUS_INVALID_VALUE);
  tag = 7;
  IS(cutensorOperationDescriptorSetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_TAG, &tag, 4), 0);
  IS(cutensorOperationDescriptorSetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_TAG, &tag, 8), CUTENSOR_STATUS_INVALID_VALUE);
  tag = 0;
  cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_TAG, &tag, 4);
  check(tag == 7, "the tag reads back");
  cudaDataType_t st = CUDA_R_32F;
  IS(cutensorOperationDescriptorSetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, &st, 4), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorOperationDescriptorSetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_FLOPS, &flops, 4), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, &st, 8), CUTENSOR_STATUS_INVALID_VALUE);
  uint32_t pads[2] = {0, 0};
  IS(cutensorOperationDescriptorSetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT, pads, 8), CUTENSOR_STATUS_INVALID_VALUE);
  int32_t rep = 0;
  IS(cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_BLOCKSPARSE_REPRODUCIBLE, &rep, 4), CUTENSOR_STATUS_INVALID_VALUE);
  // Workspace estimates: preferences 1 to 3 only, the preference required.
  uint64_t ws = 77;
  IS(cutensorEstimateWorkspaceSize(h, op, pref, (cutensorWorksizePreference_t)0, &ws), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorEstimateWorkspaceSize(h, op, pref, (cutensorWorksizePreference_t)4, &ws), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorEstimateWorkspaceSize(h, op, nullptr, CUTENSOR_WORKSPACE_DEFAULT, &ws), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorEstimateWorkspaceSize(h, op, pref, CUTENSOR_WORKSPACE_DEFAULT, nullptr), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorEstimateWorkspaceSize(h, op, pref, CUTENSOR_WORKSPACE_MIN, &ws), 0);
  cutensorPlan_t pl = nullptr;
  IS(cutensorCreatePlan(h, &pl, op, nullptr, 0), CUTENSOR_STATUS_INVALID_VALUE);  // a preference is required
  IS(cutensorCreatePlan(h, &pl, op, pref, 0), 0);
  uint64_t need = 99;
  IS(cutensorPlanGetAttribute(h, pl, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &need, 8), 0);
  check(need == 0, "a plan made with no workspace needs none");
  IS(cutensorPlanGetAttribute(h, pl, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &need, 4), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorPlanGetAttribute(h, pl, (cutensorPlanAttribute_t)1, &need, 8), CUTENSOR_STATUS_INVALID_VALUE);

  // alpha = 0 leaves A and B unread; beta = 0 leaves C unread.
  std::vector<cd> a = ints(24, 4, false), b = ints(30, 5, false);
  std::vector<cd> nan20(20, cd(NAN, 0)), two(20, cd(2, 0)), nan24(24, cd(NAN, 0));
  void *pa = upload(a, CUDA_R_32F), *pb = upload(b, CUDA_R_32F), *pnanA = upload(nan24, CUDA_R_32F);
  void *pc = upload(nan20, CUDA_R_32F), *pc2 = upload(two, CUDA_R_32F);
  float one = 1, zero = 0;
  IS(cutensorContract(h, pl, &one, pa, pb, &zero, pc, pc, nullptr, 0, 0), 0);
  std::vector<cd> got = download(pc, 20, CUDA_R_32F);
  bool finite = true;
  for (auto& x : got) finite = finite && std::isfinite(x.real());
  check(finite, "beta = 0: a NaN in C does not reach D");
  IS(cutensorContract(h, pl, &zero, pnanA, pb, &one, pc2, pc2, nullptr, 0, 0), 0);
  got = download(pc2, 20, CUDA_R_32F);
  check(got[0] == cd(2, 0) && got[19] == cd(2, 0), "alpha = 0: a NaN in A does not reach D");
  IS(cutensorContract(h, pl, nullptr, pa, pb, &zero, pc, pc, nullptr, 0, 0), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorContract(h, pl, &one, nullptr, pb, &zero, pc, pc, nullptr, 0, 0), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorContract(h, pl, &one, pa, pb, nullptr, pc, pc, nullptr, 0, 0), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorContract(h, pl, &one, pa, pb, &zero, nullptr, pc, nullptr, 0, 0), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorContract(h, pl, &one, pa, pb, &zero, pc, nullptr, nullptr, 0, 0), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorContract(h, nullptr, &one, pa, pb, &zero, pc, pc, nullptr, 0, 0), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorContract(nullptr, pl, &one, pa, pb, &zero, pc, pc, nullptr, 0, 0), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorContract(h, pl, &one, pa, pb, &zero, pc, pc, nullptr, 1000, 0), CUTENSOR_STATUS_INVALID_VALUE);

  // A contraction recorded into a CUDA graph runs when the graph does.
  {
    cudaStream_t s;
    cudaStreamCreate(&s);
    void* pd = upload(std::vector<cd>(20, cd(0, 0)), CUDA_R_32F);
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ge = nullptr;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    const int rs = cutensorContract(h, pl, &one, pa, pb, &zero, pd, pd, nullptr, 0, s);
    cudaStreamEndCapture(s, &g);
    check(rs == 0, "a contraction is recorded while the stream captures");
    std::vector<cd> before = download(pd, 20, CUDA_R_32F);
    check(before[0] == cd(0, 0), "nothing is computed at capture");
    cudaGraphInstantiate(&ge, g, 0);
    cudaGraphLaunch(ge, s);
    cudaStreamSynchronize(s);
    std::vector<cd> after = download(pd, 20, CUDA_R_32F);
    cd ref = 0;
    for (int k = 0; k < 6; ++k) ref += a[(size_t)(0 + 4 * k)] * b[(size_t)(k + 6 * 0)];
    check(after[0] == ref, "the graph computes it at launch");
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    cudaStreamDestroy(s);
  }
  cutensorDestroyPlan(pl);

  // A mode only A has is summed; a mode A repeats is its diagonal.
  {
    auto A3 = desc(CUDA_R_32F, {4, 6, 3});
    const int32_t mkq[] = {'m', 'k', 'q'};
    IS(cutensorCreateContraction(h, &op, A3, mkq, ID, B, kn, ID, C, mn, ID, C, mn, C32), 0);
    Planned p = plan(op);
    IS(p.status, 0);
    std::vector<cd> a3 = ints(72, 6, false);
    void* pa3 = upload(a3, CUDA_R_32F);
    void* pd = upload(std::vector<cd>(20, cd(0, 0)), CUDA_R_32F);
    IS(cutensorContract(h, p.plan, &one, pa3, pb, &zero, pd, pd, p.ws, p.wsize, 0), 0);
    got = download(pd, 20, CUDA_R_32F);
    cd ref = 0;
    for (int q = 0; q < 3; ++q)
      for (int k = 0; k < 6; ++k) ref += a3[(size_t)(1 + 4 * (k + 6 * q))] * b[(size_t)(k + 6 * 2)];
    check(got[(size_t)(1 + 4 * 2)] == ref, "a mode only A has is summed over");

    auto Asq = desc(CUDA_R_32F, {4, 4}), Bk = desc(CUDA_R_32F, {3});
    const int32_t mm[] = {'m', 'm'}, k1[] = {'k'};
    std::vector<cd> q(16);
    for (int i = 0; i < 16; ++i) q[(size_t)i] = i + 1;
    void* pq = upload(q, CUDA_R_32F);
    void* pk = upload({1, 2, 3}, CUDA_R_32F);
    void* pr = upload({0, 0, 0}, CUDA_R_32F);
    IS(cutensorCreateContraction(h, &op, Asq, mm, ID, Bk, k1, ID, Bk, k1, ID, Bk, k1, C32), 0);
    Planned t = plan(op);
    IS(t.status, 0);
    IS(cutensorContract(h, t.plan, &one, pq, pk, &zero, pr, pr, t.ws, t.wsize, 0), 0);
    got = download(pr, 3, CUDA_R_32F);
    check(got[0] == cd(34, 0) && got[2] == cd(102, 0), "A[m,m] is A's diagonal: trace(A) b = 34, 68, 102");
  }

  // Complex: CONJ on A, FLOPS counted 8 per multiply-add (4 with one real
  // operand).
  {
    auto Az = desc(CUDA_C_32F, {4, 6}), Bz = desc(CUDA_C_32F, {6, 4}), Cz = desc(CUDA_C_32F, {4, 4});
    const int32_t m2[] = {'m', 'n'};
    IS(cutensorCreateContraction(h, &op, Az, mk, CUTENSOR_OP_CONJ, Bz, kn, ID, Cz, m2, ID, Cz, m2, C32), 0);
    cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_FLOPS, &flops, 4);
    check(flops == 768, "complex contraction FLOPS = 8 m n k = 768");
    Planned p = plan(op);
    IS(p.status, 0);
    std::vector<cd> az = ints(24, 7, true), bz = ints(24, 8, true);
    void *paz = upload(az, CUDA_C_32F), *pbz = upload(bz, CUDA_C_32F), *pcz = upload(std::vector<cd>(16), CUDA_C_32F);
    float al[2] = {1, 0}, be[2] = {0, 0};
    IS(cutensorContract(h, p.plan, al, paz, pbz, be, pcz, pcz, p.ws, p.wsize, 0), 0);
    got = download(pcz, 16, CUDA_C_32F);
    cd ref = 0;
    for (int k = 0; k < 6; ++k) ref += std::conj(az[(size_t)(2 + 4 * k)]) * bz[(size_t)(k + 6 * 3)];
    check(got[(size_t)(2 + 4 * 3)] == ref, "opA = CONJ conjugates A");
    IS(cutensorCreateContraction(h, &op, Az, mk, CUTENSOR_OP_ABS, Bz, kn, ID, Cz, m2, ID, Cz, m2, C32), CUTENSOR_STATUS_NOT_SUPPORTED);
    auto Ad = desc(CUDA_R_64F, {4, 6}), Bd = desc(CUDA_C_64F, {6, 4}), Cd = desc(CUDA_C_64F, {4, 4});
    IS(cutensorCreateContraction(h, &op, Ad, mk, ID, Bd, kn, ID, Cd, m2, ID, Cd, m2, CUTENSOR_COMPUTE_DESC_64F), 0);
    cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_FLOPS, &flops, 4);
    cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES, &bytes, 4);
    check(flops == 384 && bytes == 1088, "R64F x C64F: FLOPS 4 m n k = 384, MOVED_BYTES 1088");
  }
}

// ---- trinary contraction ----

static void trinary_contraction() {
  // E[m,n] = alpha A[m,k] B[k,l] C[l,n] + beta D[m,n]  (m4 k6 l3 n5)
  auto A = desc(CUDA_R_64F, {4, 6}), B = desc(CUDA_R_64F, {6, 3}), C = desc(CUDA_R_64F, {3, 5}),
       D = desc(CUDA_R_64F, {4, 5});
  const int32_t ma[] = {'m', 'k'}, mb[] = {'k', 'l'}, mc[] = {'l', 'n'}, md[] = {'m', 'n'};
  OpVar op;
  IS(cutensorCreateContractionTrinary(h, &op, A, ma, ID, B, mb, ID, C, mc, ID, D, md, ID, D, md,
                                      CUTENSOR_COMPUTE_DESC_64F), 0);
  float flops = 0, bytes = 0;
  cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_FLOPS, &flops, 4);
  cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES, &bytes, 4);
  check(flops == 264, "trinary contraction FLOPS: (AB)C, 2*4*6*3 + 2*4*3*5 = 264");
  check(bytes == 776, "trinary contraction MOVED_BYTES = (24+18+15+20+20)*8 = 776");
  check(scalar_of(op) == CUDA_R_64F, "trinary contraction scalar type");
  Planned p = plan(op);
  IS(p.status, 0);
  std::vector<cd> a = ints(24, 11, false), b = ints(18, 12, false), c = ints(15, 13, false), d = ints(20, 14, false);
  void *pa = upload(a, CUDA_R_64F), *pb = upload(b, CUDA_R_64F), *pc = upload(c, CUDA_R_64F), *pd = upload(d, CUDA_R_64F);
  double al = 0.5, be = 2;
  IS(cutensorContractTrinary(h, p.plan, &al, pa, pb, pc, &be, pd, pd, p.ws, p.wsize, 0), 0);
  std::vector<cd> got = download(pd, 20, CUDA_R_64F), want(20);
  for (int n = 0; n < 5; ++n)
    for (int m = 0; m < 4; ++m) {
      cd s = 0;
      for (int k = 0; k < 6; ++k)
        for (int l = 0; l < 3; ++l) s += a[(size_t)(m + 4 * k)] * b[(size_t)(k + 6 * l)] * c[(size_t)(l + 3 * n)];
      want[(size_t)(m + 4 * n)] = al * s + be * d[(size_t)(m + 4 * n)];
    }
  check(maxdiff(got, want) == 0, "trinary contraction values exact");
  auto H = desc(CUDA_R_16F, {4, 6});
  IS(cutensorCreateContractionTrinary(h, &op, H, ma, ID, B, mb, ID, C, mc, ID, D, md, ID, D, md,
                                      CUTENSOR_COMPUTE_DESC_64F), 0);
  IS(plan(op).status, CUTENSOR_STATUS_NOT_SUPPORTED);
}

// ---- permutations: types, operators, broadcasting, padding ----

static double unary_ref(int op, double x) {
  switch (op) {
    case 1: return x;
    case 2: return std::sqrt(x);
    case 8: return x > 0 ? x : 0;
    case 10: return 1 / x;
    case 11: return 1 / (1 + std::exp(-x));
    case 12: return std::tanh(x);
    case 22: return std::exp(x);
    case 23: return std::log(x);
    case 24: return std::fabs(x);
    case 25: return -x;
    case 26: return std::sin(x);
    case 27: return std::cos(x);
    case 28: return std::tan(x);
    case 29: return std::sinh(x);
    case 30: return std::cosh(x);
    case 31: return std::asin(x);
    case 32: return std::acos(x);
    case 33: return std::atan(x);
    case 34: return std::asinh(x);
    case 35: return std::acosh(x);
    case 36: return std::atanh(x);
    case 37: return std::ceil(x);
    case 38: return std::floor(x);
    case 39: return x * std::tanh(std::log1p(std::exp(x)));
    case 40: return x / (1 + std::exp(-x));
    case 41: return std::log1p(std::exp(x));
    case 42: return x / (std::fabs(x) + 1);
    default: return NAN;
  }
}

static void permutations() {
  const cutensorComputeDescriptor_t C16F = CUTENSOR_COMPUTE_DESC_16F, C16BF = CUTENSOR_COMPUTE_DESC_16BF,
                                    C32 = CUTENSOR_COMPUTE_DESC_32F, C64 = CUTENSOR_COMPUTE_DESC_64F;
  struct Combo {
    cudaDataType_t a, b;
    cutensorComputeDescriptor_t c;
    cudaDataType_t scalar;
    const char* name;
  };
  const Combo ok[] = {
      {CUDA_R_16F, CUDA_R_16F, C16F, CUDA_R_16F, "R16F->R16F 16F"},   {CUDA_R_16F, CUDA_R_16F, C32, CUDA_R_32F, "R16F->R16F 32F"},
      {CUDA_R_16F, CUDA_R_32F, C32, CUDA_R_32F, "R16F->R32F 32F"},    {CUDA_R_16BF, CUDA_R_16BF, C16BF, CUDA_R_16BF, "R16BF->R16BF 16BF"},
      {CUDA_R_16BF, CUDA_R_16BF, C32, CUDA_R_32F, "R16BF->R16BF 32F"}, {CUDA_R_32F, CUDA_R_16F, C32, CUDA_R_32F, "R32F->R16F 32F"},
      {CUDA_R_32F, CUDA_R_32F, C32, CUDA_R_32F, "R32F->R32F 32F"},    {CUDA_R_32F, CUDA_R_64F, C64, CUDA_R_64F, "R32F->R64F 64F"},
      {CUDA_R_64F, CUDA_R_32F, C64, CUDA_R_64F, "R64F->R32F 64F"},    {CUDA_R_64F, CUDA_R_64F, C64, CUDA_R_64F, "R64F->R64F 64F"},
      {CUDA_C_32F, CUDA_C_32F, C32, CUDA_C_32F, "C32F->C32F 32F"},    {CUDA_C_32F, CUDA_C_64F, C64, CUDA_C_64F, "C32F->C64F 64F"},
      {CUDA_C_64F, CUDA_C_32F, C64, CUDA_C_64F, "C64F->C32F 64F"},    {CUDA_C_64F, CUDA_C_64F, C64, CUDA_C_64F, "C64F->C64F 64F"},
  };
  const int32_t abc[] = {'a', 'b', 'c'}, cab[] = {'c', 'a', 'b'};
  for (const Combo& c : ok) {
    std::string name = std::string("permutation ") + c.name;
    auto dA = desc(c.a, {4, 5, 6}), dB = desc(c.b, {6, 4, 5});
    OpVar op;
    IS(cutensorCreatePermutation(h, &op, dA, abc, ID, dB, cab, c.c), 0);
    check(scalar_of(op) == c.scalar, (name + ": scalar type").c_str());
    Planned p = plan(op);
    check(p.status == 0, (name + ": planned").c_str());
    if (p.status) continue;
    std::vector<cd> a = ints(120, 21, cplx(c.a));
    void *pa = upload(a, c.a), *pb = upload(std::vector<cd>(120), c.b);
    auto al = scalar_bytes(c.scalar, cd(-2, 0));
    IS(cutensorPermute(h, p.plan, al.data(), pa, pb, 0), 0);
    std::vector<cd> got = download(pb, 120, c.b), want(120);
    for (int x = 0; x < 4; ++x)
      for (int y = 0; y < 5; ++y)
        for (int z = 0; z < 6; ++z) want[(size_t)(z + 6 * (x + 4 * y))] = -2.0 * a[(size_t)(x + 4 * (y + 5 * z))];
    check(maxdiff(got, want) == 0, (name + ": values exact").c_str());
  }
  // Not taken: R16F with 16BF (NOT_SUPPORTED at planning), TF32 descriptors
  // (INVALID_VALUE at creation), Hopper descriptors (NOT_SUPPORTED), 16BF on
  // complex data (INVALID_VALUE), integers as output (NOT_SUPPORTED).
  {
    auto h16 = desc(CUDA_R_16F, {4}), f32 = desc(CUDA_R_32F, {4}), z32 = desc(CUDA_C_32F, {4}), i8 = desc(CUDA_R_8I, {4});
    const int32_t i1[] = {'i'};
    OpVar op;
    IS(cutensorCreatePermutation(h, &op, h16, i1, ID, h16, i1, C16BF), 0);
    IS(plan(op).status, CUTENSOR_STATUS_NOT_SUPPORTED);
    IS(cutensorCreatePermutation(h, &op, f32, i1, ID, f32, i1, CUTENSOR_COMPUTE_DESC_TF32), CUTENSOR_STATUS_INVALID_VALUE);
    IS(cutensorCreatePermutation(h, &op, f32, i1, ID, f32, i1, CUTENSOR_COMPUTE_DESC_9X16BF), CUTENSOR_STATUS_NOT_SUPPORTED);
    IS(cutensorCreatePermutation(h, &op, z32, i1, ID, z32, i1, C16BF), CUTENSOR_STATUS_INVALID_VALUE);
    IS(cutensorCreatePermutation(h, &op, f32, i1, ID, i8, i1, C32), 0);
    IS(plan(op).status, CUTENSOR_STATUS_NOT_SUPPORTED);
  }
  // Every unary operator on real data, against the host's math.
  {
    auto S = desc(CUDA_R_32F, {4});
    const int32_t i1[] = {'i'};
    const std::vector<cd> x = {0.25, 0.75, -0.5, 2.0};
    void* px = upload(x, CUDA_R_32F);
    void* py = upload(std::vector<cd>(4), CUDA_R_32F);
    float one = 1;
    bool all_ok = true, refusals_ok = true;
    for (int o = 1; o <= 42; ++o) {
      OpVar op;
      const int cs = cutensorCreatePermutation(h, &op, S, i1, (cutensorOperator_t)o, S, i1, C32);
      const bool valid = (o == 1 || o == 2 || o == 8 || (o >= 10 && o <= 12) || o >= 22);
      if (!valid) {
        refusals_ok = refusals_ok && cs == CUTENSOR_STATUS_INVALID_VALUE;
        continue;
      }
      Planned p = plan(op);
      if (cs || p.status || cutensorPermute(h, p.plan, &one, px, py, 0)) {
        all_ok = false;
        std::printf("     operator %d refused\n", o);
        continue;
      }
      std::vector<cd> got = download(py, 4, CUDA_R_32F);
      for (int i = 0; i < 4; ++i) {
        const double want = (double)(float)unary_ref(o, x[(size_t)i].real());
        const double g = got[(size_t)i].real();
        const bool same = (std::isnan(want) && std::isnan(g)) || std::fabs(g - want) <= 2e-6 * std::fmax(1.0, std::fabs(want));
        if (!same) {
          all_ok = false;
          std::printf("     operator %d at %g: %.9g, want %.9g\n", o, x[(size_t)i].real(), g, want);
        }
      }

    }
    check(refusals_ok, "binary operators, CONJ and unknown values are INVALID_VALUE as a real unary operator");
    check(all_ok, "the 27 real unary operators compute their functions");
    auto Z = desc(CUDA_C_32F, {4});
    OpVar op;
    IS(cutensorCreatePermutation(h, &op, Z, i1, CUTENSOR_OP_SQRT, Z, i1, C32), CUTENSOR_STATUS_INVALID_VALUE);
    IS(cutensorCreatePermutation(h, &op, Z, i1, CUTENSOR_OP_CONJ, Z, i1, C32), 0);
    Planned p = plan(op);
    std::vector<cd> z = {cd(1, 2), cd(-3, 1), cd(0, -1), cd(2, 2)};
    void *pz = upload(z, CUDA_C_32F), *pw = upload(std::vector<cd>(4), CUDA_C_32F);
    float al[2] = {1, 0};
    IS(cutensorPermute(h, p.plan, al, pz, pw, 0), 0);
    std::vector<cd> got = download(pw, 4, CUDA_C_32F);
    check(got[0] == cd(1, -2) && got[3] == cd(2, -2), "CONJ conjugates complex data");
  }
  // Broadcasting a mode the input lacks; extents that disagree.
  {
    auto A = desc(CUDA_R_32F, {4, 6}), B3 = desc(CUDA_R_32F, {6, 4, 2}), Bx = desc(CUDA_R_32F, {6, 5});
    const int32_t mk[] = {'m', 'k'}, kmz[] = {'k', 'm', 'z'}, km[] = {'k', 'm'};
    OpVar op;
    IS(cutensorCreatePermutation(h, &op, A, mk, ID, B3, kmz, CUTENSOR_COMPUTE_DESC_32F), 0);
    Planned p = plan(op);
    IS(p.status, 0);
    std::vector<cd> a = ints(24, 31, false);
    void *pa = upload(a, CUDA_R_32F), *pb = upload(std::vector<cd>(48), CUDA_R_32F);
    float one = 1;
    IS(cutensorPermute(h, p.plan, &one, pa, pb, 0), 0);
    std::vector<cd> got = download(pb, 48, CUDA_R_32F);
    check(got[(size_t)(5 + 6 * (3 + 4 * 1))] == a[(size_t)(3 + 4 * 5)] && got[5 + 6 * 3] == a[3 + 4 * 5],
          "a mode only the output has is broadcast");
    IS(cutensorCreatePermutation(h, &op, A, mk, ID, Bx, km, CUTENSOR_COMPUTE_DESC_32F), 0);
    IS(plan(op).status, CUTENSOR_STATUS_INVALID_VALUE);
    IS(cutensorCreatePermutation(h, &op, A, mk, CUTENSOR_OP_ADD, B3, kmz, CUTENSOR_COMPUTE_DESC_32F), CUTENSOR_STATUS_INVALID_VALUE);
  }
  // Padding: the output descriptor keeps the unpadded extents, its strides
  // lay out the padded buffer.
  {
    std::vector<cd> a(24);
    for (int i = 0; i < 24; ++i) a[(size_t)i] = i + 1;
    void* pa = upload(a, CUDA_R_32F);
    auto A = desc(CUDA_R_32F, {4, 6});
    const int32_t mk[] = {'m', 'k'}, km[] = {'k', 'm'};
    auto B = desc(CUDA_R_32F, {6, 4}, {1, 8});
    OpVar op;
    IS(cutensorCreatePermutation(h, &op, A, mk, ID, B, km, CUTENSOR_COMPUTE_DESC_32F), 0);
    uint32_t pl[2] = {1, 1}, pr[2] = {1, 1}, got_pl[2] = {9, 9};
    float pv = -7, got_pv = 0;
    IS(cutensorOperationDescriptorSetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT, pl, 8), 0);
    IS(cutensorOperationDescriptorSetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT, pr, 8), 0);
    IS(cutensorOperationDescriptorSetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE, &pv, 4), 0);
    IS(cutensorOperationDescriptorSetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT, pl, 4), CUTENSOR_STATUS_INVALID_VALUE);
    IS(cutensorOperationDescriptorSetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE, &pv, 8), CUTENSOR_STATUS_INVALID_VALUE);
    IS(cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT, got_pl, 8), 0);
    IS(cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE, &got_pv, 4), 0);
    check(got_pl[0] == 1 && got_pl[1] == 1 && got_pv == -7, "padding attributes read back");
    Planned p = plan(op);
    IS(p.status, 0);
    void* pb = upload(std::vector<cd>(64, cd(100, 0)), CUDA_R_32F);
    float one = 1;
    IS(cutensorPermute(h, p.plan, &one, pa, pb, 0), 0);
    std::vector<cd> got = download(pb, 64, CUDA_R_32F);
    bool good = true;
    for (int m = -1; m <= 4; ++m)
      for (int k = -1; k <= 6; ++k) {
        const double want = (m < 0 || m > 3 || k < 0 || k > 5) ? -7 : a[(size_t)(m + 4 * k)].real();
        good = good && got[(size_t)((k + 1) + 8 * (m + 1))].real() == want;
      }
    for (int i = 48; i < 64; ++i) good = good && got[(size_t)i].real() == 100;
    check(good, "a permutation padded by one on every side writes the 8x6 padded box");
    // Left padding of two along k only, default (zero) value, packed strides:
    // the box overlaps itself and the result wins.
    auto Bp = desc(CUDA_R_32F, {6, 4});
    IS(cutensorCreatePermutation(h, &op, A, mk, ID, Bp, km, CUTENSOR_COMPUTE_DESC_32F), 0);
    uint32_t l2[2] = {2, 0};
    IS(cutensorOperationDescriptorSetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT, l2, 8), 0);
    p = plan(op);
    IS(p.status, 0);
    void* pc = upload(std::vector<cd>(64, cd(100, 0)), CUDA_R_32F);
    IS(cutensorPermute(h, p.plan, &one, pa, pc, 0), 0);
    got = download(pc, 64, CUDA_R_32F);
    const double want2[] = {0, 0, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23, 4, 8, 12, 16, 20, 24, 100};
    bool good2 = true;
    for (int i = 0; i < 27; ++i) good2 = good2 && got[(size_t)i].real() == want2[i];
    check(good2, "left padding with the default value and overlapping rows");
  }
  // FLOPS and MOVED_BYTES of the elementwise family (2 per element, 4 for
  // complex; every operand counted once).
  {
    auto X = desc(CUDA_R_32F, {4, 6}), Y = desc(CUDA_R_32F, {6, 4}), Z = desc(CUDA_C_32F, {4, 6}),
         W = desc(CUDA_C_32F, {6, 4});
    const int32_t mk[] = {'m', 'k'}, km[] = {'k', 'm'};
    OpVar op;
    float fl = 0, mb = 0;
    cutensorCreatePermutation(h, &op, X, mk, ID, Y, km, CUTENSOR_COMPUTE_DESC_32F);
    cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_FLOPS, &fl, 4);
    cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES, &mb, 4);
    check(fl == 48 && mb == 192, "permutation FLOPS 48, MOVED_BYTES 192");
    cutensorCreateElementwiseBinary(h, &op, X, mk, ID, Y, km, ID, Y, km, CUTENSOR_OP_ADD, CUTENSOR_COMPUTE_DESC_32F);
    cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_FLOPS, &fl, 4);
    cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES, &mb, 4);
    check(fl == 48 && mb == 288, "binary FLOPS 48, MOVED_BYTES 288");
    cutensorCreateElementwiseTrinary(h, &op, X, mk, ID, X, mk, ID, Y, km, ID, Y, km, CUTENSOR_OP_ADD, CUTENSOR_OP_ADD,
                                     CUTENSOR_COMPUTE_DESC_32F);
    cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_FLOPS, &fl, 4);
    cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES, &mb, 4);
    check(fl == 48 && mb == 384, "trinary FLOPS 48, MOVED_BYTES 384");
    cutensorCreatePermutation(h, &op, Z, mk, ID, W, km, CUTENSOR_COMPUTE_DESC_32F);
    cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_FLOPS, &fl, 4);
    check(fl == 96, "complex permutation FLOPS 96");
  }
}

// ---- elementwise binary and trinary ----

static void elementwise() {
  auto S = desc(CUDA_R_32F, {4});
  const int32_t i1[] = {'i'};
  const cutensorComputeDescriptor_t C32 = CUTENSOR_COMPUTE_DESC_32F;
  void* x = upload({1, -2, 3, cd(NAN, 0)}, CUDA_R_32F);
  void* y = upload({2, 2, -1, 5}, CUDA_R_32F);
  void* w = upload({0, 16, 1, 4}, CUDA_R_32F);
  void* z = upload(std::vector<cd>(4), CUDA_R_32F);
  OpVar op;
  // D = max(2 * -x, -1 * |y|): a NaN on either side gives NaN.
  IS(cutensorCreateElementwiseBinary(h, &op, S, i1, CUTENSOR_OP_NEG, S, i1, CUTENSOR_OP_ABS, S, i1, CUTENSOR_OP_MAX, C32), 0);
  check(scalar_of(op) == CUDA_R_32F, "binary scalar type");
  Planned p = plan(op);
  float al = 2, ga = -1, zero = 0;
  IS(cutensorElementwiseBinaryExecute(h, p.plan, &al, x, &ga, y, z, 0), 0);
  std::vector<cd> got = download(z, 4, CUDA_R_32F);
  check(got[0] == cd(-2, 0) && got[1] == cd(4, 0) && got[2] == cd(-1, 0) && std::isnan(got[3].real()),
        "binary MAX with operators and scalars, NaN propagating");
  IS(cutensorElementwiseBinaryExecute(h, p.plan, &zero, x, &ga, y, z, 0), 0);
  got = download(z, 4, CUDA_R_32F);
  check(got[0] == cd(0, 0) && got[3] == cd(0, 0), "alpha = 0: A contributes zero, unread");
  IS(cutensorElementwiseBinaryExecute(h, p.plan, nullptr, x, &ga, y, z, 0), CUTENSOR_STATUS_INVALID_VALUE);
  // D = min((1.5 x) * (2 sqrt(w)), y)
  IS(cutensorCreateElementwiseTrinary(h, &op, S, i1, ID, S, i1, CUTENSOR_OP_SQRT, S, i1, ID, S, i1, CUTENSOR_OP_MUL,
                                      CUTENSOR_OP_MIN, C32), 0);
  p = plan(op);
  float a15 = 1.5f, b2 = 2, g1 = 1;
  IS(cutensorElementwiseTrinaryExecute(h, p.plan, &a15, x, &b2, w, &g1, y, z, 0), 0);
  got = download(z, 4, CUDA_R_32F);
  check(got[0] == cd(0, 0) && got[1] == cd(-24, 0) && got[2] == cd(-1, 0) && std::isnan(got[3].real()),
        "trinary MUL then MIN");
  IS(cutensorElementwiseTrinaryExecute(h, p.plan, &a15, x, &b2, w, &g1, y, y, 0), 0);  // D may be C
  // Operators, layouts and modes.
  auto Z = desc(CUDA_C_32F, {4});
  for (int o = 1; o <= 8; ++o) {
    const bool real_ok = o == 3 || o == 5 || o == 6 || o == 7, cplx_ok = o == 3 || o == 5;
    const int rs = cutensorCreateElementwiseBinary(h, &op, S, i1, ID, S, i1, ID, S, i1, (cutensorOperator_t)o, C32);
    const int zs = cutensorCreateElementwiseBinary(h, &op, Z, i1, ID, Z, i1, ID, Z, i1, (cutensorOperator_t)o, C32);
    check(rs == (real_ok ? 0 : 7) && zs == (cplx_ok ? 0 : 7),
          ("binary operator " + std::to_string(o) + ": ADD MUL MAX MIN for real, ADD MUL for complex").c_str());
  }
  auto X = desc(CUDA_R_32F, {4, 6}), Y = desc(CUDA_R_32F, {6, 4}), Yx = desc(CUDA_R_32F, {6, 4}, {1, 8}),
       Y3 = desc(CUDA_R_32F, {6, 4, 2});
  const int32_t mk[] = {'m', 'k'}, km[] = {'k', 'm'}, kmz[] = {'k', 'm', 'z'};
  IS(cutensorCreateElementwiseBinary(h, &op, X, mk, ID, Y, km, ID, Yx, km, CUTENSOR_OP_ADD, C32), CUTENSOR_STATUS_NOT_SUPPORTED);
  IS(cutensorCreateElementwiseBinary(h, &op, X, mk, ID, Y, km, ID, Y, mk, CUTENSOR_OP_ADD, C32), CUTENSOR_STATUS_NOT_SUPPORTED);
  IS(cutensorCreateElementwiseBinary(h, &op, X, mk, CUTENSOR_OP_ADD, Y, km, ID, Y, km, CUTENSOR_OP_ADD, C32), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorCreateElementwiseBinary(h, &op, X, mk, ID, Y3, kmz, ID, Y3, kmz, CUTENSOR_OP_ADD, C32), 0);
  p = plan(op);
  IS(p.status, 0);
  std::vector<cd> xa = ints(24, 41, false), ya = ints(48, 42, false);
  void *px = upload(xa, CUDA_R_32F), *py = upload(ya, CUDA_R_32F);
  float one = 1;
  IS(cutensorElementwiseBinaryExecute(h, p.plan, &one, px, &one, py, py, 0), 0);
  got = download(py, 48, CUDA_R_32F);
  const size_t at = (size_t)(5 + 6 * (2 + 4 * 1));
  check(got[at] == xa[2 + 4 * 5] + ya[at], "binary: A broadcast along the mode only C and D have");
  // Mixed precision: R16F A into an R32F output.
  {
    auto Ah = desc(CUDA_R_16F, {4}), Df = desc(CUDA_R_32F, {4});
    IS(cutensorCreateElementwiseBinary(h, &op, Ah, i1, ID, Df, i1, ID, Df, i1, CUTENSOR_OP_ADD, C32), 0);
    p = plan(op);
    IS(p.status, 0);
    void* ph = upload({0.25, -1.5, 3, 1024}, CUDA_R_16F);
    void* pf = upload({1, 1, 1, 1}, CUDA_R_32F);
    IS(cutensorElementwiseBinaryExecute(h, p.plan, &one, ph, &one, pf, pf, 0), 0);
    got = download(pf, 4, CUDA_R_32F);
    check(got[0] == cd(1.25, 0) && got[1] == cd(-0.5, 0) && got[3] == cd(1025, 0), "R16F + R32F into R32F");
  }
}

// ---- reductions ----

static void reductions() {
  auto X = desc(CUDA_R_32F, {4, 6}), R = desc(CUDA_R_32F, {6}), Rx = desc(CUDA_R_32F, {7}), S = desc(CUDA_R_32F, {});
  const int32_t mk[] = {'m', 'k'}, k1[] = {'k'}, z1[] = {'z'};
  const cutensorComputeDescriptor_t C32 = CUTENSOR_COMPUTE_DESC_32F;
  std::vector<cd> x = ints(24, 51, false);
  void* px = upload(x, CUDA_R_32F);
  OpVar op;
  for (int o : {3, 5, 6, 7}) {
    IS(cutensorCreateReduction(h, &op, X, mk, ID, R, k1, ID, R, k1, (cutensorOperator_t)o, C32), 0);
    Planned p = plan(op);
    IS(p.status, 0);
    void* pr = upload({1, 2, 3, 4, 5, 6}, CUDA_R_32F);
    float al = 2, be = 1;
    IS(cutensorReduce(h, p.plan, &al, px, &be, pr, pr, p.ws, p.wsize, 0), 0);
    std::vector<cd> got = download(pr, 6, CUDA_R_32F);
    bool good = true;
    for (int k = 0; k < 6; ++k) {
      double r = o == 5 ? 1 : (o == 6 ? -INFINITY : (o == 7 ? INFINITY : 0));
      for (int m = 0; m < 4; ++m) {
        const double v = x[(size_t)(m + 4 * k)].real();
        r = o == 3 ? r + v : o == 5 ? r * v : o == 6 ? std::fmax(r, v) : std::fmin(r, v);
      }
      good = good && got[(size_t)k].real() == 2 * r + (k + 1);
    }
    check(good, ("reduction with operator " + std::to_string(o) + ": D = 2 reduce(A) + C").c_str());
  }
  float fl = 0, mb = 0;
  cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_FLOPS, &fl, 4);
  cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES, &mb, 4);
  check(fl == 48 && mb == 144, "reduction FLOPS 48, MOVED_BYTES 144");
  check(scalar_of(op) == CUDA_R_32F, "reduction scalar type");
  // A full reduction to a scalar.
  IS(cutensorCreateReduction(h, &op, X, mk, ID, S, nullptr, ID, S, nullptr, CUTENSOR_OP_ADD, C32), 0);
  Planned p = plan(op);
  IS(p.status, 0);
  void* ps = upload({0}, CUDA_R_32F);
  float one = 1, zero = 0;
  IS(cutensorReduce(h, p.plan, &one, px, &zero, ps, ps, p.ws, p.wsize, 0), 0);
  cd total = 0;
  for (auto& v : x) total += v;
  check(download(ps, 1, CUDA_R_32F)[0] == total, "a full reduction sums every element");
  // Checks.
  IS(cutensorCreateReduction(h, &op, X, mk, ID, R, k1, ID, R, k1, CUTENSOR_OP_SQRT, C32), CUTENSOR_STATUS_NOT_SUPPORTED);
  auto Z = desc(CUDA_C_32F, {4, 6}), ZR = desc(CUDA_C_32F, {6});
  IS(cutensorCreateReduction(h, &op, Z, mk, ID, ZR, k1, ID, ZR, k1, CUTENSOR_OP_MAX, C32), CUTENSOR_STATUS_NOT_SUPPORTED);
  IS(cutensorCreateReduction(h, &op, Z, mk, ID, ZR, k1, ID, ZR, k1, CUTENSOR_OP_MUL, C32), 0);
  IS(cutensorCreateReduction(h, &op, X, mk, ID, R, z1, ID, R, z1, CUTENSOR_OP_ADD, C32), 0);
  IS(plan(op).status, CUTENSOR_STATUS_NOT_SUPPORTED);  // broadcasting into a reduction's output
  IS(cutensorCreateReduction(h, &op, X, mk, ID, Rx, k1, ID, Rx, k1, CUTENSOR_OP_ADD, C32), 0);
  IS(plan(op).status, CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorCreateReduction(h, &op, X, mk, ID, X, mk, ID, X, mk, CUTENSOR_OP_ADD, C32), 0);
  IS(plan(op).status, 0);  // nothing to reduce
  auto H = desc(CUDA_R_16F, {4, 6}), HR = desc(CUDA_R_16F, {6});
  IS(cutensorCreateReduction(h, &op, H, mk, ID, HR, k1, ID, HR, k1, CUTENSOR_OP_ADD, CUTENSOR_COMPUTE_DESC_64F), 0);
  IS(plan(op).status, CUTENSOR_STATUS_NOT_SUPPORTED);
  auto CF = desc(CUDA_C_16F, {6});
  IS(cutensorCreateReduction(h, &op, X, mk, ID, CF, k1, ID, CF, k1, CUTENSOR_OP_ADD, C32), 0);
  IS(plan(op).status, CUTENSOR_STATUS_INTERNAL_ERROR);
}

// ---- descriptors, preferences, caches, logger, block-sparse ----

static void infrastructure() {
  check(cutensorGetVersion() >= 20500, "version is 2.5 or later");
  check(std::strcmp(cutensorGetErrorString(CUTENSOR_STATUS_NOT_SUPPORTED), "CUTENSOR_STATUS_NOT_SUPPORTED") == 0,
        "error string");
  check(std::strcmp(cutensorGetErrorString((cutensorStatus_t)2), "<unknown>") == 0, "unknown error string");
  IS(cutensorCreate(nullptr), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorDestroy(nullptr), 0);

  // Tensor descriptors.
  TdVar d;
  const int64_t e2[] = {4, 5}, s2[] = {1, 4}, e0[] = {0, 5}, en[] = {-1, 5}, sz[] = {0, 4}, sn[] = {-1, 4};
  IS(cutensorCreateTensorDescriptor(nullptr, &d, 2, e2, s2, CUDA_R_32F, 256), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorCreateTensorDescriptor(h, nullptr, 2, e2, s2, CUDA_R_32F, 256), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorCreateTensorDescriptor(h, &d, 2, nullptr, s2, CUDA_R_32F, 256), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorCreateTensorDescriptor(h, &d, 2, e0, s2, CUDA_R_32F, 256), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorCreateTensorDescriptor(h, &d, 2, en, s2, CUDA_R_32F, 256), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorCreateTensorDescriptor(h, &d, 2, e2, sn, CUDA_R_32F, 256), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorCreateTensorDescriptor(h, &d, 2, e2, sz, CUDA_R_32F, 256), 0);
  IS(cutensorCreateTensorDescriptor(h, &d, 0, nullptr, nullptr, CUDA_R_32F, 256), 0);
  std::vector<int64_t> big(70, 2);
  IS(cutensorCreateTensorDescriptor(h, &d, 63, big.data(), nullptr, CUDA_R_32F, 256), 0);
  IS(cutensorCreateTensorDescriptor(h, &d, 64, big.data(), nullptr, CUDA_R_32F, 256), CUTENSOR_STATUS_NOT_SUPPORTED);
  // Types: these are taken, the complex integers are NOT_SUPPORTED, the rest
  // INVALID_VALUE.
  const int taken[] = {CUDA_R_32F, CUDA_R_64F, CUDA_R_16F, CUDA_R_8I, CUDA_C_32F, CUDA_C_64F, CUDA_C_16F,
                       CUDA_R_8U, CUDA_R_32I, CUDA_R_32U, CUDA_R_16BF};
  bool types_ok = true;
  for (int t : taken) types_ok = types_ok && cutensorCreateTensorDescriptor(h, &d, 2, e2, s2, (cudaDataType_t)t, 256) == 0;
  for (int t : {CUDA_C_8I, CUDA_C_8U, CUDA_C_32I, CUDA_C_32U})
    types_ok = types_ok && cutensorCreateTensorDescriptor(h, &d, 2, e2, s2, (cudaDataType_t)t, 256) == 15;
  for (int t : {CUDA_C_16BF, CUDA_R_4I, CUDA_R_16I, CUDA_R_64I})
    types_ok = types_ok && cutensorCreateTensorDescriptor(h, &d, 2, e2, s2, (cudaDataType_t)t, 256) == 7;
  check(types_ok, "descriptor element types");
  // Alignment: 0, or a power of two that is a multiple of the element size.
  bool align_ok = true;
  struct A_ {
    cudaDataType_t t;
    uint32_t al;
    int want;
  } aligns[] = {{CUDA_R_32F, 0, 0}, {CUDA_R_32F, 1, 7}, {CUDA_R_32F, 2, 7}, {CUDA_R_32F, 3, 7}, {CUDA_R_32F, 4, 0},
                {CUDA_R_32F, 6, 7}, {CUDA_R_32F, 8, 0}, {CUDA_R_16F, 1, 7}, {CUDA_R_16F, 2, 0}, {CUDA_R_64F, 4, 7},
                {CUDA_R_64F, 8, 0}, {CUDA_C_64F, 8, 7}, {CUDA_C_64F, 16, 0}, {CUDA_R_8I, 1, 0}, {CUDA_R_32F, 1024, 0}};
  for (const A_& a : aligns) align_ok = align_ok && cutensorCreateTensorDescriptor(h, &d, 2, e2, nullptr, a.t, a.al) == a.want;
  check(align_ok, "descriptor alignment rules");
  IS(cutensorDestroyTensorDescriptor(nullptr), 0);

  // Plan preferences.
  cutensorPlanPreference_t p2 = nullptr;
  IS(cutensorCreatePlanPreference(h, nullptr, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorCreatePlanPreference(h, &p2, (cutensorAlgo_t)5, CUTENSOR_JIT_MODE_NONE), CUTENSOR_STATUS_NOT_SUPPORTED);
  IS(cutensorCreatePlanPreference(h, &p2, CUTENSOR_ALGO_DEFAULT, (cutensorJitMode_t)2), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorCreatePlanPreference(h, &p2, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE), 0);
  const int32_t defaults[] = {0, 1, 4, -1, 0, 0, -1};
  bool def_ok = true;
  for (int a = 0; a <= 6; ++a) {
    int32_t v = 99;
    def_ok = def_ok && cutensorPlanPreferenceGetAttribute(h, p2, (cutensorPlanPreferenceAttribute_t)a, &v, 4) == 0 &&
             v == defaults[a];
    int64_t w = 0;
    def_ok = def_ok && cutensorPlanPreferenceGetAttribute(h, p2, (cutensorPlanPreferenceAttribute_t)a, &w, 8) == 7;
  }
  int32_t v32 = 0;
  def_ok = def_ok && cutensorPlanPreferenceGetAttribute(h, p2, (cutensorPlanPreferenceAttribute_t)7, &v32, 4) == 7;
  check(def_ok, "plan preference defaults: autotune 0, cache 1, incremental 4, algo -1, rank 0, JIT 0, arch -1");
  struct Set_ {
    int attr, value, want;
  } sets[] = {{3, -6, 0}, {3, -5, 7}, {3, -4, 0}, {3, 0, 7}, {0, -1, 7}, {0, 1, 0}, {1, 0, 0}, {1, -1, 7},
              {2, 0, 7},  {2, 1, 0},  {4, -1, 7}, {4, 5, 0}, {5, 2, 7},  {5, 1, 0}, {6, 86, 7}, {6, 80, 0},
              {6, 90, 0}, {6, 100, 0}, {6, 120, 7}};
  bool set_ok = true;
  for (const Set_& s : sets) {
    const int32_t x = s.value;
    set_ok = set_ok && cutensorPlanPreferenceSetAttribute(h, p2, (cutensorPlanPreferenceAttribute_t)s.attr, &x, 4) == s.want;
  }
  int64_t w8 = 1;
  set_ok = set_ok && cutensorPlanPreferenceSetAttribute(h, p2, CUTENSOR_PLAN_PREFERENCE_CACHE_MODE, &w8, 8) == 7;
  check(set_ok, "plan preference values accepted and refused");
  IS(cutensorDestroyPlanPreference(p2), 0);
  IS(cutensorDestroyPlanPreference(nullptr), 0);

  // The plan cache and its file.
  cutensorHandle_t h2 = nullptr, h3 = nullptr;
  cutensorCreate(&h2);
  cutensorCreate(&h3);
  const char* dir = std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp";
  const std::string f1 = std::string(dir) + "/vgpu_cutensor_cache_" + std::to_string((long)getpid()) + ".bin";
  const std::string fj = f1 + ".junk", fe = f1 + ".empty", fk = f1 + ".kernels";
  {
    // Plan a contraction on h2 so its cache holds something.
    auto A = desc(CUDA_R_32F, {4, 6}), B = desc(CUDA_R_32F, {6, 5}), C = desc(CUDA_R_32F, {4, 5});
    const int32_t mk[] = {'m', 'k'}, kn[] = {'k', 'n'}, mn[] = {'m', 'n'};
    OpVar op;
    cutensorCreateContraction(h2, &op, A, mk, ID, B, kn, ID, C, mn, ID, C, mn, CUTENSOR_COMPUTE_DESC_32F);
    cutensorPlanPreference_t pp = nullptr;
    cutensorCreatePlanPreference(h2, &pp, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE);
    cutensorPlan_t pl = nullptr;
    IS(cutensorCreatePlan(h2, &pl, op, pp, 0), 0);
    cutensorDestroyPlan(pl);
    cutensorDestroyPlanPreference(pp);
  }
  IS(cutensorHandleWritePlanCacheToFile(h2, f1.c_str()), 0);
  IS(cutensorHandleWritePlanCacheToFile(h2, "/nonexistent/dir/cache.bin"), CUTENSOR_STATUS_IO_ERROR);
  uint32_t nr = 0;
  // Which plans a cache keeps is the library's choice; reading back what
  // was written succeeds.
  IS(cutensorHandleReadPlanCacheFromFile(h3, f1.c_str(), &nr), 0);
  check(nr <= 64, "no more cachelines than the default capacity");
  IS(cutensorHandleReadPlanCacheFromFile(h3, "/nonexistent/cache.bin", &nr), CUTENSOR_STATUS_IO_ERROR);
  FILE* f = std::fopen(fj.c_str(), "w");
  std::fputs("not a cache", f);
  std::fclose(f);
  std::fclose(std::fopen(fe.c_str(), "w"));
  IS(cutensorHandleReadPlanCacheFromFile(h3, fj.c_str(), &nr), CUTENSOR_STATUS_IO_ERROR);
  IS(cutensorHandleReadPlanCacheFromFile(h3, fe.c_str(), &nr), CUTENSOR_STATUS_IO_ERROR);
  IS(cutensorHandleResizePlanCache(nullptr, 4), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorHandleResizePlanCache(h3, 0), 0);
  IS(cutensorHandleReadPlanCacheFromFile(h3, f1.c_str(), &nr), CUTENSOR_STATUS_INVALID_VALUE);  // no cache attached
  IS(cutensorHandleWritePlanCacheToFile(h3, f1.c_str()), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorWriteKernelCacheToFile(h3, fk.c_str()), 0);
  IS(cutensorReadKernelCacheFromFile(h3, fk.c_str()), 0);
  IS(cutensorReadKernelCacheFromFile(h3, fe.c_str()), 0);
  IS(cutensorReadKernelCacheFromFile(h3, fj.c_str()), CUTENSOR_STATUS_INTERNAL_ERROR);
  IS(cutensorReadKernelCacheFromFile(h3, "/nonexistent/kernels.bin"), CUTENSOR_STATUS_IO_ERROR);
  IS(cutensorWriteKernelCacheToFile(h3, "/nonexistent/kernels.bin"), 0);   // the card: SUCCESS, there is nothing to write
  for (const std::string& s : {f1, fj, fe, fk}) std::remove(s.c_str());
  cutensorDestroy(h2);
  cutensorDestroy(h3);

  // The logger.
  IS(cutensorLoggerSetLevel(0), 0);
  IS(cutensorLoggerSetLevel(-1), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorLoggerSetMask(0), 0);
  IS(cutensorLoggerSetCallback(nullptr), 0);
  IS(cutensorLoggerSetFile(nullptr), 0);
  IS(cutensorLoggerOpenFile("/nonexistent/dir/cutensor.log"), CUTENSOR_STATUS_INVALID_VALUE);

  // Block-sparse descriptors and operations (an RTX 3060 cannot plan them).
  BsVar bA, bB, bC;
  const uint32_t sec[] = {2, 2};
  const int64_t ext[] = {2, 3, 4, 1}, ext0[] = {2, 0, 4, 1}, extB[] = {4, 1, 2, 2}, extC[] = {2, 3, 2, 2};
  const int32_t nz[] = {0, 0, 1, 1}, bad[] = {0, 0, 2, 1}, nzc[] = {0, 0, 1, 0, 0, 1, 1, 1};
  IS(cutensorCreateBlockSparseTensorDescriptor(h, &bA, 2, 2, sec, ext, nz, nullptr, CUDA_R_32F), 0);
  IS(cutensorCreateBlockSparseTensorDescriptor(h, &bB, 2, 2, sec, extB, nz, nullptr, CUDA_R_32F), 0);
  IS(cutensorCreateBlockSparseTensorDescriptor(h, &bC, 2, 4, sec, extC, nzc, nullptr, CUDA_R_32F), 0);
  BsVar bx;
  IS(cutensorCreateBlockSparseTensorDescriptor(h, &bx, 2, 2, sec, ext, bad, nullptr, CUDA_R_32F), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorCreateBlockSparseTensorDescriptor(h, &bx, 2, 2, sec, ext, nz, nullptr, CUDA_R_16F), CUTENSOR_STATUS_NOT_SUPPORTED);
  IS(cutensorCreateBlockSparseTensorDescriptor(h, &bx, 2, 2, sec, ext0, nz, nullptr, CUDA_R_32F), CUTENSOR_STATUS_INVALID_VALUE);
  IS(cutensorCreateBlockSparseTensorDescriptor(h, &bx, 2, 0, sec, ext, nz, nullptr, CUDA_R_32F), 0);
  const int32_t mA[] = {'m', 'k'}, mB[] = {'k', 'n'}, mC[] = {'m', 'n'};
  OpVar op;
  IS(cutensorCreateBlockSparseContraction(h, &op, bA, mA, ID, bB, mB, ID, bC, mC, ID, bC, mC, CUTENSOR_COMPUTE_DESC_32F), 0);
  check(scalar_of(op) == CUDA_R_32F, "block-sparse scalar type");
  int32_t rep = -1;
  IS(cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_BLOCKSPARSE_REPRODUCIBLE, &rep, 4), 0);
  check(rep == 0, "block-sparse results are not reproducible by default");
  float fl = 0;
  IS(cutensorOperationDescriptorGetAttribute(h, op, CUTENSOR_OPERATION_DESCRIPTOR_FLOPS, &fl, 4), CUTENSOR_STATUS_NOT_SUPPORTED);
  IS(cutensorDestroyBlockSparseTensorDescriptor(nullptr), 0);
}

int main() {
  if (cutensorCreate(&h) != CUTENSOR_STATUS_SUCCESS) {
    std::printf("FAIL: cutensorCreate\n");
    return 1;
  }
  cutensorCreatePlanPreference(h, &pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE);
  infrastructure();
  contraction_types();
  contraction_checks();
  trinary_contraction();
  permutations();
  elementwise();
  reductions();
  release_all();
  cutensorDestroyPlanPreference(pref);
  cutensorDestroy(h);
  if (failures) {
    std::printf("FAIL: %d checks failed\n", failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
