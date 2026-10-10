// Which cuDNN graph operations have an engine on a Hopper (or later) GPU: the ones an RTX 3060 (sm_86, cuDNN 9.27)
// offers none for -- the standalone rotary embedding (forward and backward), the MoE grouped matmul backward pass, and
// the backend's band-matrix expand and contract operations (which cudnn-frontend never emits).
//
// A probe, not a test: it prints one line per graph saying whether the heuristics offer an engine and, when one runs,
// how far its output is from the formulas the simulator computes (nvidia/src/cudnn_backend.cpp), then the first values
// of the output (PROBE_DUMP=1 prints every value). Run it against NVIDIA's cuDNN on a Hopper card and commit what it
// printed; nothing here asserts that an engine exists. The last line is PASS when nothing crashed.
//
//   nvcc -std=c++17 -cudart shared -arch=sm_90a -I<cudnn include> -I<cudnn-frontend include> dnn_hopper_engines.cpp \
//        -o dnn_hopper_engines -L<cudnn lib> -lcudnn -ldl
#define NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#include <cudnn_frontend.h>
#include <cuda_runtime.h>

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;
namespace cudnn_frontend {
void* cudnn_dlhandle = dlopen("libcudnn.so.9", RTLD_NOW | RTLD_GLOBAL);
}
using TA = fe::graph::Tensor_attributes;

static bool dump_all = false;
struct Dev {
  void* p = nullptr;
  size_t n;
  explicit Dev(size_t bytes) : n(bytes ? bytes : 1) { cudaMalloc(&p, n); cudaMemset(p, 0, n); }
  ~Dev() { cudaFree(p); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
};

static uint16_t f2bf(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  x += 0x7fffu + ((x >> 16) & 1u);
  return static_cast<uint16_t>(x >> 16);
}
static float bf2f(uint16_t b) {
  const uint32_t x = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}
static double bfround(double v) { return bf2f(f2bf(static_cast<float>(v))); }
static std::vector<double> randoms(size_t n, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> v(n);
  for (double& x : v) x = u(g);
  return v;
}
static std::vector<uint16_t> to_bf16(const std::vector<double>& v) {
  std::vector<uint16_t> b(v.size());
  for (size_t i = 0; i < v.size(); ++i) b[i] = f2bf(static_cast<float>(v[i]));
  return b;
}
static std::vector<double> from_bf16(const std::vector<uint16_t>& b) {
  std::vector<double> v(b.size());
  for (size_t i = 0; i < b.size(); ++i) v[i] = bf2f(b[i]);
  return v;
}
static void show(const char* what, const std::vector<double>& v, size_t first = 12) {
  std::printf("  %s:", what);
  for (size_t i = 0; i < v.size() && (dump_all || i < first); ++i) std::printf(" %.6g", v[i]);
  std::printf("%s\n", !dump_all && v.size() > first ? " ..." : "");
}

// Builds the graph in cudnn-frontend's stages and says where it stopped.
static bool plan(const char* name, fe::graph::Graph& g, cudnnHandle_t h) {
  auto bad = [&](const char* at, fe::error_t st) {
    std::printf("%s: no engine (%s: %s)\n", name, at, st.get_message().substr(0, 140).c_str());
    return false;
  };
  fe::error_t st = g.validate();
  if (!st.is_good()) return bad("validate", st);
  st = g.build_operation_graph(h);
  if (!st.is_good()) return bad("build_operation_graph", st);
  st = g.create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
  if (!st.is_good()) return bad("create_execution_plans", st);
  st = g.check_support();
  if (!st.is_good()) return bad("check_support", st);
  st = g.build_plans();
  if (!st.is_good()) return bad("build_plans", st);
  std::printf("%s: an engine is offered and its plan builds\n", name);
  return true;
}
static bool run(fe::graph::Graph& g, cudnnHandle_t h, std::unordered_map<int64_t, void*>& v) {
  int64_t ws = 0;
  if (!g.get_workspace_size(ws).is_good()) return false;
  Dev work(static_cast<size_t>(ws));
  auto st = g.execute(h, v, work.p);
  cudaError_t ce = cudaDeviceSynchronize();
  if (!st.is_good() || ce != cudaSuccess) {
    std::printf("  execute failed: %s (cuda %d)\n", st.get_message().substr(0, 140).c_str(), static_cast<int>(ce));
    return false;
  }
  return true;
}

// ---- standalone rotary embedding -----------------------------------------------------------------

static void rope_case(cudnnHandle_t h, bool bwd, int64_t rope_dim) {
  const int64_t B = 1, H = 2, S = 16, D = 64, r = rope_dim ? rope_dim : D;
  const std::vector<int64_t> st{H * S * D, S * D, D, 1};
  const fe::DataType_t bf = fe::DataType_t::BFLOAT16, fl = fe::DataType_t::FLOAT;
  char name[96];
  std::snprintf(name, sizeof name, "rotary embedding %s alone, bfloat16, rotated width %lld of %lld", bwd ? "backward" : "forward",
                static_cast<long long>(r), static_cast<long long>(D));
  try {
    fe::graph::Graph g;
    g.set_intermediate_data_type(fl).set_compute_data_type(fl);
    auto x = g.tensor(TA().set_name("x").set_dim({B, H, S, D}).set_stride(st).set_data_type(bf).set_uid(1));
    auto f = g.tensor(TA().set_name("freqs").set_dim({S, 1, 1, r}).set_stride({r, r, r, 1}).set_data_type(fl).set_uid(2));
    std::shared_ptr<TA> y;
    if (bwd) {
      auto a = fe::graph::RoPE_backward_attributes().set_name("rope_bwd");
      if (rope_dim) a.set_rope_dim(rope_dim);
      y = g.rope_backward(x, f, a);
    } else {
      auto a = fe::graph::RoPE_attributes().set_name("rope");
      if (rope_dim) a.set_rope_dim(rope_dim);
      y = g.rope(x, f, a);
    }
    y->set_output(true).set_data_type(bf).set_dim({B, H, S, D}).set_stride(st).set_uid(3);
    if (!plan(name, g, h)) return;
    std::vector<double> xv = randoms(B * H * S * D, 21), fv = randoms(S * r, 22);
    for (double& v : xv) v = bfround(v);
    for (double& v : fv) v = static_cast<float>(v * 3.0);
    const std::vector<uint16_t> bx = to_bf16(xv);
    std::vector<float> ff(fv.begin(), fv.end());
    Dev dx(bx.size() * 2), df(ff.size() * 4), dy(bx.size() * 2);
    cudaMemcpy(dx.p, bx.data(), bx.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(df.p, ff.data(), ff.size() * 4, cudaMemcpyHostToDevice);
    std::unordered_map<int64_t, void*> v{{1, dx.p}, {2, df.p}, {3, dy.p}};
    if (!run(g, h, v)) return;
    std::vector<uint16_t> raw(bx.size());
    cudaMemcpy(raw.data(), dy.p, raw.size() * 2, cudaMemcpyDeviceToHost);
    const std::vector<double> got = from_bf16(raw);
    // The formulas the simulator uses (non-interleaved halves of the rotated tail), and the interleaved pairing for contrast.
    const int64_t nope = D - r, half = r / 2;
    std::vector<double> want_halves(xv.size()), want_pairs(xv.size());
    for (int64_t bh = 0; bh < B * H; ++bh)
      for (int64_t s = 0; s < S; ++s) {
        const size_t row = static_cast<size_t>((bh * S + s) * D);
        for (int64_t d = 0; d < nope; ++d) want_halves[row + d] = want_pairs[row + d] = xv[row + d];
        for (int64_t j = 0; j < half; ++j) {
          const double a = static_cast<float>(fv[static_cast<size_t>(s * r + j)]), c = std::cos(a), sn = std::sin(a);
          const size_t i1 = row + nope + j, i2 = row + nope + half + j;
          want_halves[i1] = bwd ? xv[i1] * c + xv[i2] * sn : xv[i1] * c - xv[i2] * sn;
          want_halves[i2] = bwd ? xv[i2] * c - xv[i1] * sn : xv[i2] * c + xv[i1] * sn;
          const size_t p1 = row + nope + 2 * j, p2 = p1 + 1;
          const double b = static_cast<float>(fv[static_cast<size_t>(s * r + 2 * j)]), c2 = std::cos(b), s2 = std::sin(b);
          want_pairs[p1] = bwd ? xv[p1] * c2 + xv[p2] * s2 : xv[p1] * c2 - xv[p2] * s2;
          want_pairs[p2] = bwd ? xv[p2] * c2 - xv[p1] * s2 : xv[p2] * c2 + xv[p1] * s2;
        }
      }
    auto err = [&](const std::vector<double>& w) {
      double e = 0;
      for (size_t i = 0; i < got.size(); ++i) e = std::max(e, std::fabs(got[i] - w[i]) / (1.0 + std::fabs(w[i])));
      return e;
    };
    std::printf("  max relative error against the halves formula %.4g, against interleaved pairs %.4g\n", err(want_halves), err(want_pairs));
    show("output", got);
  } catch (const std::exception& ex) {
    std::printf("%s: frontend exception: %s\n", name, ex.what());
  }
}

// ---- the MoE grouped matmul, backward -------------------------------------------------------------

static void moe_bwd_case(cudnnHandle_t h, fe::DataType_t ty, const char* tn) {
  const int64_t E = 3, T = 64, K = 32, N = 16;
  const fe::DataType_t fl = fe::DataType_t::FLOAT;
  const std::string name = std::string("MoE grouped matmul backward, ") + tn;
  try {
    fe::graph::Graph g;
    g.set_intermediate_data_type(fl).set_compute_data_type(fl);
    auto dout = g.tensor(TA().set_name("dout").set_dim({1, T, N}).set_stride({T * N, N, 1}).set_data_type(ty).set_uid(1));
    auto tok = g.tensor(TA().set_name("tok").set_dim({1, T, K}).set_stride({T * K, K, 1}).set_data_type(ty).set_uid(2));
    auto off = g.tensor(TA().set_name("off").set_dim({E, 1, 1}).set_stride({1, 1, 1}).set_data_type(fe::DataType_t::INT32).set_uid(3));
    auto dw = g.moe_grouped_matmul_bwd(dout, tok, off, fe::graph::Moe_grouped_matmul_bwd_attributes().set_name("moe_bwd").set_compute_data_type(fl));
    dw->set_output(true).set_data_type(ty).set_uid(4);
    if (!plan(name.c_str(), g, h)) return;
    // dW[e] = sum over the rows of expert e of tok[row]^T dout[row]; the output is [E, K, N] with column-major strides.
    const std::vector<int32_t> offsets{0, 20, 41};
    std::vector<double> dv = randoms(T * N, 31), tv = randoms(T * K, 32);
    const bool is_bf = ty == fe::DataType_t::BFLOAT16;
    if (!is_bf) { std::printf("  (only bfloat16 data is run)\n"); return; }
    for (double& v : dv) v = bfround(v);
    for (double& v : tv) v = bfround(v);
    const std::vector<uint16_t> bd = to_bf16(dv), bt = to_bf16(tv);
    Dev ddout(bd.size() * 2), dtok(bt.size() * 2), doff(offsets.size() * 4), ddw(static_cast<size_t>(E * K * N) * 2);
    cudaMemcpy(ddout.p, bd.data(), bd.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dtok.p, bt.data(), bt.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(doff.p, offsets.data(), offsets.size() * 4, cudaMemcpyHostToDevice);
    std::unordered_map<int64_t, void*> v{{1, ddout.p}, {2, dtok.p}, {3, doff.p}, {4, ddw.p}};
    if (!run(g, h, v)) return;
    std::vector<uint16_t> raw(static_cast<size_t>(E * K * N));
    cudaMemcpy(raw.data(), ddw.p, raw.size() * 2, cudaMemcpyDeviceToHost);
    const std::vector<double> got = from_bf16(raw);
    std::vector<double> want(got.size(), 0.0);
    for (int64_t e = 0; e < E; ++e)
      for (int64_t k = 0; k < K; ++k)
        for (int64_t n = 0; n < N; ++n) {
          double acc = 0;
          const int64_t lo = offsets[static_cast<size_t>(e)], hi = e + 1 < E ? offsets[static_cast<size_t>(e + 1)] : T;
          for (int64_t i = lo; i < hi; ++i) acc += tv[static_cast<size_t>(i * K + k)] * dv[static_cast<size_t>(i * N + n)];
          want[static_cast<size_t>(e * K * N + k + n * K)] = bfround(acc);   // [E, K, N] with K contiguous
        }
    double e1 = 0, e2 = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      e1 = std::max(e1, std::fabs(got[i] - want[i]) / (1.0 + std::fabs(want[i])));
      // The other layout, N contiguous.
      const size_t e = i / static_cast<size_t>(K * N), rem = i % static_cast<size_t>(K * N);
      const size_t k = rem / static_cast<size_t>(N), n = rem % static_cast<size_t>(N);
      e2 = std::max(e2, std::fabs(got[i] - want[e * K * N + k + n * K]) / (1.0 + std::fabs(want[e * K * N + k + n * K])));
    }
    std::printf("  max relative error with K contiguous (column-major [K, N] per expert) %.4g, with N contiguous %.4g\n", e1, e2);
    show("output", got);
  } catch (const std::exception& ex) {
    std::printf("%s: frontend exception: %s\n", name.c_str(), ex.what());
  }
}

// ---- band-matrix expand and contract (backend API) --------------------------------------------------

static cudnnBackendDescriptor_t mk(cudnnBackendDescriptorType_t t) {
  cudnnBackendDescriptor_t d = nullptr;
  cudnnBackendCreateDescriptor(t, &d);
  return d;
}
static cudnnStatus_t setd(cudnnBackendDescriptor_t d, cudnnBackendAttributeName_t a, cudnnBackendAttributeType_t t, int64_t n, const void* p) {
  return cudnnBackendSetAttribute(d, a, t, n, p);
}
static cudnnBackendDescriptor_t tensor_desc(int64_t uid, std::vector<int64_t> dims, cudnnDataType_t ty) {
  std::vector<int64_t> st(dims.size());
  int64_t s = 1;
  for (size_t i = dims.size(); i-- > 0;) { st[i] = s; s *= dims[i]; }
  cudnnBackendDescriptor_t t = mk(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
  int64_t align = 16;
  setd(t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
  setd(t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &ty);
  setd(t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, static_cast<int64_t>(dims.size()), dims.data());
  setd(t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, static_cast<int64_t>(st.size()), st.data());
  setd(t, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align);
  if (cudnnBackendFinalize(t) != CUDNN_STATUS_SUCCESS) std::printf("  tensor %lld did not finalize\n", static_cast<long long>(uid));
  return t;
}

// An expand or contract operation over x and y of the given dimensions, offered an engine or not.
static void band_case(cudnnHandle_t h, bool expand, std::vector<int64_t> xd, std::vector<int64_t> yd, int64_t lower, int64_t upper, int64_t axis,
                      const char* label) {
  char name[128];
  std::snprintf(name, sizeof name, "band matrix %s, %s (lower %lld, upper %lld, axis %lld)", expand ? "expand" : "contract", label,
                static_cast<long long>(lower), static_cast<long long>(upper), static_cast<long long>(axis));
  auto X = tensor_desc(1, xd, CUDNN_DATA_BFLOAT16), Y = tensor_desc(2, yd, CUDNN_DATA_BFLOAT16);
  cudnnBackendDescriptor_t op = mk(expand ? CUDNN_BACKEND_OPERATION_EXPAND_BAND_MATRIX_DESCRIPTOR : CUDNN_BACKEND_OPERATION_CONTRACT_BAND_MATRIX_DESCRIPTOR);
  setd(op, expand ? CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_XDESC : CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &X);
  setd(op, expand ? CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_YDESC : CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &Y);
  setd(op, expand ? CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_LOWER_BANDWIDTH : CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_LOWER_BANDWIDTH, CUDNN_TYPE_INT64, 1, &lower);
  setd(op, expand ? CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_UPPER_BANDWIDTH : CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_UPPER_BANDWIDTH, CUDNN_TYPE_INT64, 1, &upper);
  setd(op, expand ? CUDNN_ATTR_OPERATION_EXPAND_BAND_MATRIX_AXIS : CUDNN_ATTR_OPERATION_CONTRACT_BAND_MATRIX_AXIS, CUDNN_TYPE_INT64, 1, &axis);
  if (!expand) {
    int64_t maxtok = xd.back();
    setd(op, CUDNN_ATTR_OPERATION_CONTRACT_BAND_MAX_TOKEN_VALUE, CUDNN_TYPE_INT64, 1, &maxtok);
  }
  cudnnStatus_t s = cudnnBackendFinalize(op);
  if (s != CUDNN_STATUS_SUCCESS) {
    std::printf("%s: operation does not finalize (status %d: %s)\n", name, static_cast<int>(s), "run with CUDNN_LOGERR_DBG=1 for the reason");
    return;
  }
  cudnnBackendDescriptor_t og = mk(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
  setd(og, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op);
  setd(og, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &h);
  s = cudnnBackendFinalize(og);
  if (s != CUDNN_STATUS_SUCCESS) {
    std::printf("%s: operation graph does not finalize (status %d)\n", name, static_cast<int>(s));
    return;
  }
  cudnnBackendDescriptor_t heur = mk(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
  cudnnBackendHeurMode_t mode = CUDNN_HEUR_MODE_A;
  setd(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &og);
  setd(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &mode);
  s = cudnnBackendFinalize(heur);
  int64_t n = 0;
  if (s == CUDNN_STATUS_SUCCESS) cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &n, nullptr);
  std::printf("%s: heuristics status %d, %lld engine configs\n", name, static_cast<int>(s), static_cast<long long>(n));
  cudnnBackendDestroyDescriptor(heur);
  cudnnBackendDestroyDescriptor(og);
  cudnnBackendDestroyDescriptor(op);
  cudnnBackendDestroyDescriptor(X);
  cudnnBackendDestroyDescriptor(Y);
}

int main() {
  dump_all = std::getenv("PROBE_DUMP") != nullptr;
  cudnnHandle_t h;
  if (cudnnCreate(&h) != CUDNN_STATUS_SUCCESS) {
    std::printf("SKIP: cudnnCreate failed\n");
    return 0;
  }
  int dev = 0, major = 0, minor = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  std::printf("# cuDNN %zu, cudnn-frontend %d.%d.%d, compute capability %d.%d\n", cudnnGetVersion(), CUDNN_FRONTEND_MAJOR_VERSION,
              CUDNN_FRONTEND_MINOR_VERSION, CUDNN_FRONTEND_PATCH_VERSION, major, minor);
  rope_case(h, false, 0);
  rope_case(h, true, 0);
  rope_case(h, false, 32);
  rope_case(h, true, 32);
  moe_bwd_case(h, fe::DataType_t::BFLOAT16, "bfloat16");
  moe_bwd_case(h, fe::DataType_t::HALF, "half");
  // Shapes are guesses: the header names the attributes and not the layout. The rows say which finalize and which get engines.
  band_case(h, true, {1, 2, 8, 3}, {1, 2, 8, 8}, 1, 1, 3, "band stored as [B, H, S, lower + upper + 1] to [B, H, S, S]");
  band_case(h, false, {1, 2, 8, 8}, {1, 2, 8, 3}, 1, 1, 3, "[B, H, S, S] to [B, H, S, lower + upper + 1]");
  band_case(h, true, {1, 2, 8, 3}, {1, 2, 8, 10}, 1, 1, 3, "band to a wider matrix");
  band_case(h, false, {1, 2, 8, 10}, {1, 2, 8, 3}, 1, 1, 3, "wider matrix to a band");
  cudnnDestroy(h);
  std::printf("PASS\n");
  return 0;
}
