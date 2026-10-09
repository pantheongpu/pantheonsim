// FP8 attention as cudnn-frontend builds it (graph.sdpa_fp8) on a Hopper profile: the graph the
// frontend makes must plan and run here, and give the documented attention.
//
// DOCUMENTATION-DERIVED, like dnn_fp8_attention.cu: no card was available for cuDNN's FP8
// attention (Hopper and Blackwell engines), so this checks VirtualGPU against the formulas of
// cudnn-frontend's sdpa_fp8 documentation, on the host:
//   S = (Q K^T) descale_Q descale_K attn_scale, P = softmax(S), amax_S = max P,
//   P8 = E4M3(P scale_S), O = (P8 V) descale_S descale_V, amax_O = max |O|, O8 = E4M3(O scale_O).
// Below compute capability 9 cudnn-frontend refuses FP8 attention itself (dnn_frontend_ops.cpp
// checks that); this program prints SKIP there and on a card.
#define NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#include <cudnn_frontend.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <dlfcn.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;
namespace cudnn_frontend {
void* cudnn_dlhandle = dlopen("libcudnn.so.9", RTLD_NOW | RTLD_GLOBAL);
}
using TA = fe::graph::Tensor_attributes;

static int fails = 0;
static void expect(const std::string& what, bool ok, double detail = 0) {
  std::printf("%s %s", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok && detail != 0) std::printf(" (%g)", detail);
  std::printf("\n");
  fails += !ok;
}
static uint8_t e4m3(float f) { return (uint8_t)__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3); }
static float from_e4m3(uint8_t b) { return __half2float(__half(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3))); }
template <class T> static T* up(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(T) + 64);
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  int major = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  if (!std::getenv("VGPU_GPU") || major < 9) {
    std::printf("SKIP: FP8 attention needs a Hopper or Blackwell profile on VirtualGPU (documentation-derived)\nPASS\n");
    return 0;
  }
  cudnnHandle_t h;
  if (cudnnCreate(&h) != CUDNN_STATUS_SUCCESS) {
    std::printf("FAIL cudnnCreate\n");
    return 1;
  }
  const int64_t B = 1, H = 2, S = 8, D = 16;
  const auto fl = fe::DataType_t::FLOAT;
  fe::graph::Graph g;
  g.set_io_data_type(fe::DataType_t::FP8_E4M3).set_intermediate_data_type(fl).set_compute_data_type(fl);
  auto q = g.tensor(TA().set_name("Q").set_dim({B, H, S, D}).set_stride({H * S * D, S * D, D, 1}).set_uid(1));
  auto k = g.tensor(TA().set_name("K").set_dim({B, H, S, D}).set_stride({H * S * D, S * D, D, 1}).set_uid(2));
  auto v = g.tensor(TA().set_name("V").set_dim({B, H, S, D}).set_stride({H * S * D, S * D, D, 1}).set_uid(3));
  auto sc = [&](const char* n, int uid) {
    return g.tensor(TA().set_name(n).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fl).set_uid(uid));
  };
  const float attn = 0.25f, dq = 0.5f, dk = 0.25f, dv = 2.0f, ds = 1.0f / 256.0f, scale_s = 256.0f, scale_o = 4.0f;
  auto r = g.sdpa_fp8(q, k, v, sc("dq", 11), sc("dk", 12), sc("dv", 13), sc("ds", 14), sc("sS", 15), sc("sO", 16),
                      fe::graph::SDPA_fp8_attributes().set_name("sdpa_fp8").set_generate_stats(false).set_attn_scale(attn));
  r[0]->set_output(true).set_dim({B, H, S, D}).set_stride({H * S * D, S * D, D, 1}).set_uid(4);
  r[2]->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fl).set_uid(5);
  r[3]->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fl).set_uid(6);
  fe::error_t st = g.validate();
  if (st.is_good()) st = g.build_operation_graph(h);
  if (st.is_good()) st = g.create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
  if (st.is_good()) st = g.check_support();
  if (st.is_good()) st = g.build_plans();
  expect("cudnn-frontend's sdpa_fp8 plans on a Hopper profile", st.is_good());
  if (!st.is_good()) {
    std::printf("     %s\n", st.get_message().c_str());
    return 1;
  }
  const float vals[8] = {1, -1, 0.5f, -0.5f, 2, -2, 0.25f, 3};
  auto gen = [&](size_t n, int salt) {
    std::vector<uint8_t> x(n);
    for (size_t i = 0; i < n; ++i) x[i] = e4m3(vals[(i * 7 + salt * 3 + i / 5) % 8]);
    return x;
  };
  const size_t n = (size_t)B * H * S * D;
  const auto hq = gen(n, 1), hk = gen(n, 2), hv = gen(n, 3);
  uint8_t *dQ = up(hq), *dK = up(hk), *dV = up(hv), *dO = up(std::vector<uint8_t>(n, 0x55));
  float *d11 = up(std::vector<float>{dq}), *d12 = up(std::vector<float>{dk}), *d13 = up(std::vector<float>{dv}),
        *d14 = up(std::vector<float>{ds}), *d15 = up(std::vector<float>{scale_s}), *d16 = up(std::vector<float>{scale_o}),
        *d5 = up(std::vector<float>{-1}), *d6 = up(std::vector<float>{-1});
  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> pack = {{1, dQ},  {2, dK},  {3, dV},  {4, dO},  {5, d5},  {6, d6},
                                                                          {11, d11}, {12, d12}, {13, d13}, {14, d14}, {15, d15}, {16, d16}};
  int64_t ws = 0;
  g.get_workspace_size(ws);
  void* work = nullptr;
  if (ws > 0) cudaMalloc(&work, (size_t)ws);
  st = g.execute(h, pack, work);
  cudaDeviceSynchronize();
  expect("it runs", st.is_good());
  if (!st.is_good()) std::printf("     %s\n", st.get_message().c_str());
  // Reference.
  std::vector<float> want(n);
  double want_as = 0, want_ao = 0;
  for (int hh = 0; hh < H; ++hh)
    for (int i = 0; i < S; ++i) {
      std::vector<double> s(S), p(S);
      double mx = -1e300;
      for (int j = 0; j < S; ++j) {
        double acc = 0;
        for (int d = 0; d < D; ++d) acc += (double)from_e4m3(hq[(size_t)(hh * S + i) * D + d]) * from_e4m3(hk[(size_t)(hh * S + j) * D + d]);
        s[j] = (float)(acc * attn * dq * dk);
        mx = std::fmax(mx, s[j]);
      }
      double sum = 0;
      for (int j = 0; j < S; ++j) sum += (p[j] = std::exp(s[j] - mx));
      for (int j = 0; j < S; ++j) p[j] /= sum, want_as = std::fmax(want_as, p[j]);
      for (int e = 0; e < D; ++e) {
        double acc = 0;
        for (int j = 0; j < S; ++j) acc += (double)from_e4m3(e4m3((float)(p[j] * scale_s))) * from_e4m3(hv[(size_t)(hh * S + j) * D + e]);
        const float o = (float)(acc * ds * dv);
        want_ao = std::fmax(want_ao, std::fabs(o));
        want[(size_t)(hh * S + i) * D + e] = from_e4m3(e4m3(o * scale_o));
      }
    }
  if (st.is_good()) {
    std::vector<uint8_t> got(n);
    cudaMemcpy(got.data(), dO, n, cudaMemcpyDeviceToHost);
    int off = 0;
    for (size_t i = 0; i < n; ++i) off += std::fabs(from_e4m3(got[i]) - want[i]) > 0.13 * std::fabs(want[i]) + 0.02;
    expect("O is the documented FP8 attention within one FP8 step", off == 0, off);
    float as = 0, ao = 0;
    cudaMemcpy(&as, d5, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&ao, d6, 4, cudaMemcpyDeviceToHost);
    expect("amax of S", std::fabs(as - want_as) <= 1e-6 * want_as + 1e-9, as);
    expect("amax of O", std::fabs(ao - want_ao) <= 0.13 * want_ao, ao);
  }
  for (void* p : {(void*)dQ, (void*)dK, (void*)dV, (void*)dO, (void*)d11, (void*)d12, (void*)d13, (void*)d14, (void*)d15, (void*)d16, (void*)d5, (void*)d6, work})
    cudaFree(p);
  cudnnDestroy(h);
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
