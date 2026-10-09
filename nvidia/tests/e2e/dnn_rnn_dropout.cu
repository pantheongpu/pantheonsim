// The dropout between an RNN's layers, mask for mask. cuDNN draws it from the dropout descriptor's states
// exactly as cudnnDropoutForward does (see dnn_dropout.cu): after each layer but the last, every direction of
// the next layer gets a mask of its own over the whole of the lower layer's output, [T][B][hidden * dirs] in
// that order, forward direction first, drawn as one application of the dropout kernel each -- element i from
// generator i % T, which goes on from where the last application left it. Measured on an RTX 3060 with cuDNN
// 9.27 through the states buffer and the outputs of RNNs whose weights are identity matrices, so that a
// layer's output is the previous layer's, dropped and scaled by 1 / (1 - p) in float. This program holds that
// on the host and checks the library against it, on a real GPU with NVIDIA's libcudnn.so.9 (it passes) and on
// VirtualGPU. What is not covered: sequences shorter than the longest (padded batches), whose masks were not
// measured, and the attention API's dropout descriptors.
#include <cudnn.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "xorwow_model.h"

static int fails = 0;
static void expect(const char* what, bool ok, double detail = 0) {
  std::printf("%s %s", ok ? "ok  " : "FAIL", what);
  if (!ok && detail != 0) std::printf(" (%g)", detail);
  std::printf("\n");
  fails += !ok;
}
#define CK(x) do { cudnnStatus_t s_ = (x); if (s_ != CUDNN_STATUS_SUCCESS) { \
  std::printf("FAIL %s -> %d\n", #x, (int)s_); ++fails; return; } } while (0)

template <class T>
struct Buf {
  T* p = nullptr;
  size_t n = 0;
  explicit Buf(size_t n_) : n(n_) {
    cudaMalloc(&p, n * sizeof(T) + 64);
    cudaMemset(p, 0, n * sizeof(T) + 64);
  }
  Buf(const std::vector<T>& h) : Buf(h.size()) { put(h); }
  void put(const std::vector<T>& h) { cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); }
  std::vector<T> get() const {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
  }
  ~Buf() { cudaFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
};

static cudnnHandle_t H;

// A ReLU RNN of L layers (one or two directions) whose weights are identity matrices (the input matrix of every
// pseudo-layer, on its first H input channels), no recurrence, no bias, run in training with dropout p.
static void one(const char* name, int L, bool bidir, int T, int B, int Hd, float p, unsigned long long seed) {
  const int D = bidir ? 2 : 1;
  size_t ssz = 0;
  CK(cudnnDropoutGetStatesSize(H, &ssz));
  const size_t threads = ssz / 48;
  Buf<uint8_t> states(ssz);
  cudnnDropoutDescriptor_t drop;
  CK(cudnnCreateDropoutDescriptor(&drop));
  CK(cudnnSetDropoutDescriptor(drop, H, p, states.p, ssz, seed));
  cudnnRNNDescriptor_t rd;
  CK(cudnnCreateRNNDescriptor(&rd));
  CK(cudnnSetRNNDescriptor_v8(rd, CUDNN_RNN_ALGO_STANDARD, CUDNN_RNN_RELU, CUDNN_RNN_SINGLE_INP_BIAS,
                              bidir ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT,
                              CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH, Hd, Hd, Hd, L, drop, 0));
  cudnnRNNDataDescriptor_t xd, yd;
  CK(cudnnCreateRNNDataDescriptor(&xd));
  CK(cudnnCreateRNNDataDescriptor(&yd));
  const std::vector<int> lens(B, T);
  CK(cudnnSetRNNDataDescriptor(xd, CUDNN_DATA_FLOAT, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, T, B, Hd, lens.data(), nullptr));
  CK(cudnnSetRNNDataDescriptor(yd, CUDNN_DATA_FLOAT, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, T, B, Hd * D, lens.data(), nullptr));
  size_t wbytes = 0, work = 0, reserve = 0;
  CK(cudnnGetRNNWeightSpaceSize(H, rd, &wbytes));
  CK(cudnnGetRNNTempSpaceSizes(H, rd, CUDNN_FWD_MODE_TRAINING, xd, &work, &reserve));
  Buf<uint8_t> w(wbytes);  // zeros
  for (int pl = 0; pl < L * D; ++pl) {
    cudnnTensorDescriptor_t md, bd;
    CK(cudnnCreateTensorDescriptor(&md));
    CK(cudnnCreateTensorDescriptor(&bd));
    void *ma = nullptr, *ba = nullptr;
    CK(cudnnGetRNNWeightParams(H, rd, pl, wbytes, w.p, 0, md, &ma, bd, &ba));
    cudnnDataType_t dt;
    int nd = 0, dims[8], strides[8];
    CK(cudnnGetTensorNdDescriptor(md, 8, &dt, &nd, dims, strides));
    const int rows = dims[1], cols = dims[2];
    std::vector<float> m((size_t)rows * cols, 0.0f);
    for (int i = 0; i < rows && i < cols; ++i) m[(size_t)i * cols + i] = 1.0f;
    cudaMemcpy(ma, m.data(), m.size() * 4, cudaMemcpyHostToDevice);
    cudnnDestroyTensorDescriptor(md);
    cudnnDestroyTensorDescriptor(bd);
  }
  const size_t nx = (size_t)T * B * Hd, ny = nx * D;
  std::vector<float> hx(nx);
  for (size_t i = 0; i < nx; ++i) hx[i] = 1.0f + (float)(i % 97) * 0.01f;
  Buf<float> x(hx), y(ny);
  Buf<uint8_t> wk(work), rs(reserve);
  cudnnTensorDescriptor_t hd;
  CK(cudnnCreateTensorDescriptor(&hd));
  const int h3[3] = {L * D, B, Hd}, s3[3] = {B * Hd, Hd, 1};
  CK(cudnnSetTensorNdDescriptor(hd, CUDNN_DATA_FLOAT, 3, h3, s3));
  CK(cudnnRNNForward(H, rd, CUDNN_FWD_MODE_TRAINING, nullptr, xd, x.p, yd, y.p, hd, nullptr, nullptr, hd, nullptr, nullptr,
                     wbytes, w.p, work, wk.p, reserve, rs.p));
  cudaDeviceSynchronize();
  // The model: after each layer but the last, a mask per direction over the output, [T][B][Hd * D].
  std::vector<host::State> model = host::seeded(seed, threads);
  const size_t n = (size_t)T * B * Hd * D;
  std::vector<std::vector<std::vector<uint8_t>>> keeps;  // [boundary][direction]
  for (int b = 0; b + 1 < L; ++b) {
    keeps.emplace_back();
    for (int d = 0; d < D; ++d) keeps.back().push_back(host::draw(model, n, p));
  }
  const float scale = 1.0f / (1.0f - p);
  const std::vector<float> gy = y.get();
  size_t wrong = 0;
  for (int t = 0; t < T; ++t)
    for (int b = 0; b < B; ++b)
      for (int d = 0; d < D; ++d)
        for (int j = 0; j < Hd; ++j) {
          // Layer l's unit j of direction dir reads channel j of the layer below -- its forward direction's --
          // through the mask drawn for dir at the boundary below it.
          float v = hx[((size_t)t * B + b) * Hd + j];
          bool zero = false;
          for (int l = 1; l < L; ++l) {
            const int dir = l == L - 1 ? d : 0;  // only the top layer's direction shows in y; below it the forward one feeds on
            const size_t e = ((size_t)t * B + b) * (Hd * D) + j;
            if (!keeps[l - 1][dir][e]) zero = true;
            v = v * scale;
          }
          const float got = gy[((size_t)t * B + b) * (Hd * D) + (size_t)d * Hd + j];
          const float want = zero ? 0.0f : v;
          wrong += std::fabs(got - want) > 1e-5f * (1.0f + std::fabs(want));
        }
  char what[200];
  std::snprintf(what, sizeof what, "RNN dropout, %s: the outputs follow the host model's masks", name);
  expect(what, wrong == 0, (double)wrong);
  // The generators then stand where the model's draws left them.
  std::vector<uint32_t> words(ssz / 4);
  cudaMemcpy(words.data(), states.p, ssz, cudaMemcpyDeviceToHost);
  size_t bad = 0;
  for (size_t t = 0; t < threads; ++t) {
    bool ok = words[t * 12] == model[t].d;
    for (int k = 0; k < 5; ++k) ok &= words[t * 12 + 1 + k] == model[t].v[k];
    bad += !ok;
  }
  std::snprintf(what, sizeof what, "RNN dropout, %s: the states then stand where the draws left them", name);
  expect(what, bad == 0, (double)bad);
  cudnnDestroyTensorDescriptor(hd);
  cudnnDestroyRNNDataDescriptor(xd);
  cudnnDestroyRNNDataDescriptor(yd);
  cudnnDestroyRNNDescriptor(rd);
  cudnnDestroyDropoutDescriptor(drop);
}

int main() {
  if (cudnnCreate(&H) != CUDNN_STATUS_SUCCESS) {
    std::printf("SKIP: cudnnCreate failed\n");
    return 0;
  }
  std::printf("cuDNN %zu\n", cudnnGetVersion());
  one("two layers", 2, false, 8, 4, 64, 0.5f, 77);
  one("three layers, odd sizes", 3, false, 5, 3, 100, 0.3f, 12345);
  one("two bidirectional layers", 2, true, 8, 4, 64, 0.5f, 77);
  one("three bidirectional layers", 3, true, 6, 3, 48, 0.4f, 99);
  cudnnDestroy(H);
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
