// cuDNN's classic dropout, bit for bit. The mask is not a statistic here but a function of the seed:
// cudnnSetDropoutDescriptor fills the states buffer with one cuRAND XORWOW state (48 bytes) per thread of the
// dropout kernel, thread t seeded as curand_init(seed, t, 0) seeds it; cudnnDropoutForward gives element i the
// next output of thread i % T's generator and keeps it when curand_uniform of the output exceeds p, writing
// x * (1 / (1 - p)) in float. This program holds that function on the host and checks the library against it:
// the states buffer after a set, the output and the reserve space after a forward pass, the states after it,
// the backward pass, resuming through cudnnRestoreDropoutDescriptor, and the statuses at the edges (a
// probability outside [0, 1] is accepted, a states buffer that is too small is BAD_PARAM, ...). It passes on a
// real GPU with NVIDIA's libcudnn.so.9 (an RTX 3060 with cuDNN 9.27, where T = 21504) and on VirtualGPU,
// where hosted CI runs it. T is read off cudnnDropoutGetStatesSize (48 bytes a thread), so it holds on
// whatever GPU the program runs on.
#include <cudnn.h>
#include <cuda_fp16.h>
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

// The library's states buffer, as 12-word records, against the host's model.
static bool states_match(const void* dev, size_t bytes, const std::vector<host::State>& want, double* bad) {
  std::vector<uint32_t> w(bytes / 4);
  cudaMemcpy(w.data(), dev, bytes, cudaMemcpyDeviceToHost);
  size_t wrong = 0;
  for (size_t t = 0; t < want.size(); ++t) {
    const uint32_t* r = &w[t * 12];
    bool ok = r[0] == want[t].d;
    for (int j = 0; j < 5; ++j) ok &= r[1 + j] == want[t].v[j];
    ok &= r[6] == 0 && r[7] == 0 && r[8] == 0 && r[10] == 0 && r[11] == 0;  // the Box-Muller fields; word 9 is padding
    wrong += !ok;
  }
  *bad = (double)wrong;
  return wrong == 0;
}

static cudnnTensorDescriptor_t make_tensor(cudnnDataType_t t, int n, int c, int h, int w) {
  cudnnTensorDescriptor_t d;
  cudnnCreateTensorDescriptor(&d);
  cudnnSetTensor4dDescriptor(d, CUDNN_TENSOR_NCHW, t, n, c, h, w);
  return d;
}

static float h2f(__half h) { return __half2float(h); }

static void run() {
  size_t ssz = 0;
  CK(cudnnDropoutGetStatesSize(H, &ssz));
  expect("the states are whole 48-byte XORWOW records", ssz > 0 && ssz % 48 == 0);
  const size_t T = ssz / 48;
  Buf<uint8_t> states(ssz);
  cudnnDropoutDescriptor_t dd;
  CK(cudnnCreateDropoutDescriptor(&dd));
  const float p = 0.3f;
  const unsigned long long seed = 0x1234abcd5678ull;
  CK(cudnnSetDropoutDescriptor(dd, H, p, states.p, ssz, seed));
  std::vector<host::State> model = host::seeded(seed, T);
  double bad = 0;
  expect("cudnnSetDropoutDescriptor fills the states buffer with curand_init(seed, thread, 0)'s states",
         states_match(states.p, ssz, model, &bad), bad);
  {
    float gp = 0;
    void* gs = nullptr;
    unsigned long long gseed = 0;
    CK(cudnnGetDropoutDescriptor(dd, H, &gp, &gs, &gseed));
    expect("cudnnGetDropoutDescriptor returns p, the buffer and the seed", gp == p && gs == states.p && gseed == seed);
  }

  // ---- float forward: more elements than threads, not a multiple of them
  const int N = (int)(T * 2 + T / 3 + 5);
  cudnnTensorDescriptor_t d = make_tensor(CUDNN_DATA_FLOAT, 1, 1, 1, N);
  size_t rs = 0;
  CK(cudnnDropoutGetReserveSpaceSize(d, &rs));
  expect("the reserve space is one bit an element, in whole 32-bit words", rs == ((size_t)N + 31) / 32 * 4);
  std::vector<float> hx(N);
  for (int i = 0; i < N; ++i) hx[i] = 1.0f + 0.000123f * (float)(i % 4001) - 0.25f * (float)(i % 3);
  Buf<float> x(hx), y((size_t)N), dy(hx), dx((size_t)N);
  Buf<uint8_t> res(rs);
  CK(cudnnDropoutForward(H, dd, d, x.p, d, y.p, res.p, rs));
  const float scale = 1.0f / (1.0f - p);
  {
    const std::vector<uint8_t> keep = host::draw(model, (size_t)N, p);
    const std::vector<float> gy = y.get();
    size_t wrong = 0, kept = 0;
    for (int i = 0; i < N; ++i) {
      const float want = keep[i] ? hx[i] * scale : 0.0f;
      uint32_t a, b;
      std::memcpy(&a, &gy[i], 4);
      std::memcpy(&b, &want, 4);
      wrong += a != b;
      kept += keep[i];
    }
    expect("cudnnDropoutForward (float): the kept elements are x * (1 / (1 - p)), bit for bit, and the rest 0", wrong == 0, (double)wrong);
    expect("... and about 1 - p of them are kept", std::fabs((double)kept / N - 0.7) < 0.02, (double)kept / N);
    const std::vector<uint8_t> gr = res.get();
    size_t rwrong = 0;
    for (int i = 0; i < N; ++i) rwrong += (((gr[i / 8] >> (i % 8)) & 1) != 0) != (keep[i] != 0);
    expect("... the reserve space holds the mask, element i in bit i % 8 of byte i / 8", rwrong == 0, (double)rwrong);
    expect("... the states then stand where each thread's draws left them", states_match(states.p, ssz, model, &bad), bad);
    CK(cudnnDropoutBackward(H, dd, d, dy.p, d, dx.p, res.p, rs));
    const std::vector<float> gdx = dx.get();
    size_t bwrong = 0;
    for (int i = 0; i < N; ++i) {
      const float want = keep[i] ? hx[i] * scale : 0.0f;
      uint32_t a, b;
      std::memcpy(&a, &gdx[i], 4);
      std::memcpy(&b, &want, 4);
      bwrong += a != b;
    }
    expect("cudnnDropoutBackward: dy * (1 / (1 - p)) where the mask kept the element, bit for bit", bwrong == 0, (double)bwrong);
  }
  // ---- a second pass goes on from where the first stopped; so does a restored descriptor
  {
    CK(cudnnDropoutForward(H, dd, d, x.p, d, y.p, res.p, rs));
    const std::vector<uint8_t> keep = host::draw(model, (size_t)N, p);
    const std::vector<float> gy = y.get();
    size_t wrong = 0;
    for (int i = 0; i < N; ++i) wrong += (gy[i] != 0.0f) != (keep[i] != 0);
    expect("a second forward pass continues each thread's stream", wrong == 0, (double)wrong);
    cudnnDropoutDescriptor_t rd;
    CK(cudnnCreateDropoutDescriptor(&rd));
    std::vector<uint8_t> before = states.get();
    const cudnnStatus_t rst = cudnnRestoreDropoutDescriptor(rd, H, p, states.p, ssz, 99);
    const bool untouched = states.get() == before;
    CK(rst);
    CK(cudnnDropoutForward(H, rd, d, x.p, d, y.p, res.p, rs));
    const std::vector<uint8_t> keep3 = host::draw(model, (size_t)N, p);
    const std::vector<float> gy3 = y.get();
    size_t wrong3 = 0;
    for (int i = 0; i < N; ++i) wrong3 += (gy3[i] != 0.0f) != (keep3[i] != 0);
    expect("cudnnRestoreDropoutDescriptor leaves the states alone, and the pass through it goes on from them", untouched && wrong3 == 0, (double)wrong3);
    cudnnDestroyDropoutDescriptor(rd);
    // Setting it again starts over.
    CK(cudnnSetDropoutDescriptor(dd, H, p, states.p, ssz, seed));
    model = host::seeded(seed, T);
    expect("setting the descriptor again starts the streams over", states_match(states.p, ssz, model, &bad), bad);
  }
  cudnnDestroyTensorDescriptor(d);

  // ---- half, 4-D, a different probability and seed
  {
    const float p2 = 0.6f;
    const unsigned long long seed2 = 5;
    CK(cudnnSetDropoutDescriptor(dd, H, p2, states.p, ssz, seed2));
    std::vector<host::State> m2 = host::seeded(seed2, T);
    const int n = 2, c = 3, h = 5, w = 7, M = n * c * h * w;
    cudnnTensorDescriptor_t hd = make_tensor(CUDNN_DATA_HALF, n, c, h, w);
    size_t hrs = 0;
    CK(cudnnDropoutGetReserveSpaceSize(hd, &hrs));
    std::vector<__half> hxh(M);
    for (int i = 0; i < M; ++i) hxh[i] = __float2half(0.5f + 0.01f * (float)i);
    Buf<__half> xh(hxh), yh((size_t)M);
    Buf<uint8_t> rh(hrs);
    CK(cudnnDropoutForward(H, dd, hd, xh.p, hd, yh.p, rh.p, hrs));
    const std::vector<uint8_t> keep = host::draw(m2, (size_t)M, p2);
    const float s2 = 1.0f / (1.0f - p2);
    const std::vector<__half> g = yh.get();
    size_t wrong = 0;
    for (int i = 0; i < M; ++i) {
      const float want = keep[i] ? __half2float(hxh[i]) * s2 : 0.0f;
      wrong += __half2float(__float2half(want)) != h2f(g[i]);
    }
    expect("cudnnDropoutForward (half, 4-D, p = 0.6): kept = round(x * (1 / (1 - p))), bit for bit", wrong == 0, (double)wrong);
    cudnnDestroyTensorDescriptor(hd);
  }

  // ---- the edges
  {
    cudnnTensorDescriptor_t e = make_tensor(CUDNN_DATA_FLOAT, 1, 1, 1, 1000);
    size_t ers = 0;
    cudnnDropoutGetReserveSpaceSize(e, &ers);
    std::vector<float> ones(1000, 1.0f);
    Buf<float> ex(ones), ey(1000);
    Buf<uint8_t> er(ers);
    Buf<uint8_t> big(ssz * 2);
    expect("a states buffer one byte short is BAD_PARAM",
           cudnnSetDropoutDescriptor(dd, H, 0.5f, states.p, ssz - 1, 1) == CUDNN_STATUS_BAD_PARAM);
    expect("a larger buffer is fine", cudnnSetDropoutDescriptor(dd, H, 0.5f, big.p, ssz * 2, 1) == CUDNN_STATUS_SUCCESS);
    expect("zero bytes for a buffer is BAD_PARAM", cudnnSetDropoutDescriptor(dd, H, 0.5f, states.p, 0, 1) == CUDNN_STATUS_BAD_PARAM);
    expect("a restore with a buffer too small is BAD_PARAM",
           cudnnRestoreDropoutDescriptor(dd, H, 0.5f, states.p, ssz - 1, 1) == CUDNN_STATUS_BAD_PARAM);
    {
      cudnnDropoutDescriptor_t nd;
      CK(cudnnCreateDropoutDescriptor(&nd));
      expect("no buffer at all is accepted whatever its size", cudnnSetDropoutDescriptor(nd, H, 0.5f, nullptr, 0, 1) == CUDNN_STATUS_SUCCESS);
      expect("... but a forward pass through a descriptor that never had a buffer is BAD_PARAM",
             cudnnDropoutForward(H, nd, e, ex.p, e, ey.p, er.p, ers) == CUDNN_STATUS_BAD_PARAM);
      cudnnDestroyDropoutDescriptor(nd);
    }
    CK(cudnnSetDropoutDescriptor(dd, H, 0.5f, states.p, ssz, 1));
    {
      float gp = 0;
      void* gs = nullptr;
      unsigned long long gseed = 0;
      CK(cudnnSetDropoutDescriptor(dd, H, 0.25f, nullptr, 0, 9));
      CK(cudnnGetDropoutDescriptor(dd, H, &gp, &gs, &gseed));
      expect("setting without a buffer leaves the descriptor's buffer, and a forward pass uses it",
             gs == states.p && cudnnDropoutForward(H, dd, e, ex.p, e, ey.p, er.p, ers) == CUDNN_STATUS_SUCCESS);
    }
    CK(cudnnSetDropoutDescriptor(dd, H, -0.1f, states.p, ssz, 1));
    CK(cudnnDropoutForward(H, dd, e, ex.p, e, ey.p, er.p, ers));
    {
      const std::vector<float> g = ey.get();
      const float s = 1.0f / (1.0f - (-0.1f));
      size_t wrong = 0;
      for (float v : g) wrong += v != 1.0f * s;
      expect("a probability below 0 is accepted: everything is kept, scaled by 1 / (1 - p)", wrong == 0, (double)wrong);
    }
    CK(cudnnSetDropoutDescriptor(dd, H, 1.1f, states.p, ssz, 1));
    CK(cudnnDropoutForward(H, dd, e, ex.p, e, ey.p, er.p, ers));
    {
      const std::vector<float> g = ey.get();
      size_t kept = 0;
      for (float v : g) kept += v != 0.0f;
      expect("a probability above 1 is accepted: nothing is kept", kept == 0, (double)kept);
    }
    CK(cudnnSetDropoutDescriptor(dd, H, 0.0f, states.p, ssz, 1));
    CK(cudnnDropoutForward(H, dd, e, ex.p, e, ey.p, er.p, ers));
    {
      const std::vector<float> g = ey.get();
      size_t same = 0;
      for (float v : g) same += v == 1.0f;
      expect("p = 0 keeps everything unscaled", same == 1000, (double)same);
    }
    CK(cudnnSetDropoutDescriptor(dd, H, 1.0f, states.p, ssz, 1));
    CK(cudnnDropoutForward(H, dd, e, ex.p, e, ey.p, er.p, ers));
    {
      const std::vector<float> g = ey.get();
      size_t kept = 0;
      for (float v : g) kept += v != 0.0f;
      expect("p = 1 keeps nothing", kept == 0, (double)kept);
    }
    CK(cudnnSetDropoutDescriptor(dd, H, 0.5f, states.p, ssz, 1));
    expect("a reserve space one byte short is BAD_PARAM", cudnnDropoutForward(H, dd, e, ex.p, e, ey.p, er.p, ers - 1) == CUDNN_STATUS_BAD_PARAM);
    expect("a null reserve space is BAD_PARAM", cudnnDropoutForward(H, dd, e, ex.p, e, ey.p, nullptr, ers) == CUDNN_STATUS_BAD_PARAM);
    expect("a null x is BAD_PARAM", cudnnDropoutForward(H, dd, e, nullptr, e, ey.p, er.p, ers) == CUDNN_STATUS_BAD_PARAM);
    expect("backward with a reserve space one byte short is BAD_PARAM",
           cudnnDropoutBackward(H, dd, e, ex.p, e, ey.p, er.p, ers - 1) == CUDNN_STATUS_BAD_PARAM);
    expect("in place (x == y) is fine", cudnnDropoutForward(H, dd, e, ex.p, e, ex.p, er.p, ers) == CUDNN_STATUS_SUCCESS);
    {
      cudnnTensorDescriptor_t e2 = make_tensor(CUDNN_DATA_FLOAT, 1, 1, 1, 999);
      expect("x and y of different sizes are BAD_PARAM", cudnnDropoutForward(H, dd, e, ex.p, e2, ey.p, er.p, ers) == CUDNN_STATUS_BAD_PARAM);
      cudnnDestroyTensorDescriptor(e2);
    }
    {  // double: the scale is 1 / (1 - p) in double
      const float p3 = 0.3f;
      CK(cudnnSetDropoutDescriptor(dd, H, p3, states.p, ssz, 3));
      std::vector<host::State> m3 = host::seeded(3, T);
      cudnnTensorDescriptor_t dbl = make_tensor(CUDNN_DATA_DOUBLE, 1, 1, 1, 1000);
      std::vector<double> hd(1000);
      for (int i = 0; i < 1000; ++i) hd[i] = 1.0 + 0.000123 * i;
      Buf<double> dx2(hd), dy2(1000);
      CK(cudnnDropoutForward(H, dd, dbl, dx2.p, dbl, dy2.p, er.p, ers));
      const std::vector<uint8_t> keep = host::draw(m3, 1000, p3);
      const std::vector<double> g = dy2.get();
      size_t wrong = 0;
      for (int i = 0; i < 1000; ++i) wrong += g[i] != (keep[i] ? hd[i] * (1.0 / (1.0 - (double)p3)) : 0.0);
      expect("cudnnDropoutForward (double): the kept elements are x * (1.0 / (1.0 - p)) in double, bit for bit", wrong == 0, (double)wrong);
      cudnnDestroyTensorDescriptor(dbl);
      cudnnTensorDescriptor_t bf = make_tensor(CUDNN_DATA_BFLOAT16, 1, 1, 1, 1000);
      Buf<uint16_t> bx(1000), by(1000);
      expect("bfloat16 is NOT_SUPPORTED", cudnnDropoutForward(H, dd, bf, bx.p, bf, by.p, er.p, ers) == CUDNN_STATUS_NOT_SUPPORTED);
      cudnnDestroyTensorDescriptor(bf);
    }
    cudnnDestroyTensorDescriptor(e);
  }
  cudnnDestroyDropoutDescriptor(dd);
}

int main() {
  if (cudnnCreate(&H) != CUDNN_STATUS_SUCCESS) {
    std::printf("SKIP: cudnnCreate failed\n");
    return 0;
  }
  std::printf("cuDNN %zu\n", cudnnGetVersion());
  run();
  cudnnDestroy(H);
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
