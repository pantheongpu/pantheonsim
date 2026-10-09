// The classic multi-head attention API's dropout, mask for mask. cudnnMultiHeadAttnForward in training draws its
// two dropouts from the dropout descriptors' states like cudnnDropoutForward does (see dnn_dropout.cu): the
// attention dropout, on the softmax probabilities, as one application of the dropout kernel over
// [batch][beam][head][query step][key step], and the post dropout, on the attention's output (after the output
// projection, before the residual is added), over [batch][beam][query step][output element] -- over the
// dimensions of the sequence data the call was given, padded steps included. Measured on an RTX 3060 with
// cuDNN 9.27 through the states buffers and through outputs of attention layers whose V is an identity matrix, so
// that the output is the dropped probabilities. This program holds that on the host and checks the library
// against it, on a real GPU with NVIDIA's libcudnn.so.9 (it passes) and on VirtualGPU.
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

struct Run {
  int NH, B, J, Tq, Tk;        // heads, batch, query beams, query and key steps
  bool attn_drop, post_drop;   // which dropouts
  bool project;                // identity V and O projections with a residual of 10, else no projections
  std::vector<int> lq, lk;     // sequence lengths: per (batch, beam), per batch
  float p;
  const char* name;
};

static void one(const Run& r) {
  const int S = r.Tk;  // vector sizes: queries, keys, values and outputs are all S wide
  size_t ssz = 0;
  CK(cudnnDropoutGetStatesSize(H, &ssz));
  const size_t threads = ssz / 48;
  Buf<uint8_t> st_attn(ssz), st_post(ssz);
  cudnnDropoutDescriptor_t da, dp;
  CK(cudnnCreateDropoutDescriptor(&da));
  CK(cudnnCreateDropoutDescriptor(&dp));
  const unsigned long long seed_a = 31, seed_p = 32;
  CK(cudnnSetDropoutDescriptor(da, H, r.p, st_attn.p, ssz, seed_a));
  CK(cudnnSetDropoutDescriptor(dp, H, r.p, st_post.p, ssz, seed_p));
  cudnnAttnDescriptor_t ad;
  CK(cudnnCreateAttnDescriptor(&ad));
  const int proj = r.project ? S : 0;
  CK(cudnnSetAttnDescriptor(ad, CUDNN_ATTN_QUERYMAP_ALL_TO_ONE, r.NH, 1.0, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH,
                            r.attn_drop ? da : nullptr, r.post_drop ? dp : nullptr, S, S, S, proj, proj, proj, proj, r.Tq, r.Tk,
                            r.B, r.J));
  size_t wb = 0, ws = 0, rs = 0;
  CK(cudnnGetMultiHeadAttnBuffers(H, ad, &wb, &ws, &rs));
  Buf<uint8_t> w(wb);  // zeros
  if (r.project) {
    // W_V and W_O identity (per head, [proj][size] with the strides cuDNN reports), W_Q and W_K zero: uniform attention.
    for (int kind : {2, 3}) {
      cudnnTensorDescriptor_t wd;
      CK(cudnnCreateTensorDescriptor(&wd));
      void* addr = nullptr;
      CK(cudnnGetMultiHeadAttnWeights(H, ad, (cudnnMultiHeadAttnWeightKind_t)kind, wb, w.p, wd, &addr));
      cudnnDataType_t dt;
      int nb = 0, dims[3], str[3];
      CK(cudnnGetTensorNdDescriptor(wd, 3, &dt, &nb, dims, str));
      std::vector<float> m((size_t)(dims[0] - 1) * str[0] + (size_t)(dims[1] - 1) * str[1] + (size_t)(dims[2] - 1) * str[2] + 1, 0.0f);
      for (int i = 0; i < S; ++i) m[(size_t)i * str[1] + (size_t)i * str[2]] = 1.0f;
      cudaMemcpy(addr, m.data(), m.size() * 4, cudaMemcpyHostToDevice);
      cudnnDestroyTensorDescriptor(wd);
    }
  }
  cudnnSeqDataDescriptor_t qd, kd, vd, od;
  CK(cudnnCreateSeqDataDescriptor(&qd));
  CK(cudnnCreateSeqDataDescriptor(&kd));
  CK(cudnnCreateSeqDataDescriptor(&vd));
  CK(cudnnCreateSeqDataDescriptor(&od));
  const cudnnSeqDataAxis_t axes[4] = {CUDNN_SEQDATA_BATCH_DIM, CUDNN_SEQDATA_BEAM_DIM, CUDNN_SEQDATA_TIME_DIM, CUDNN_SEQDATA_VECT_DIM};
  const int out_size = r.project ? S : r.NH * S;  // an output vector: the projection, or the heads' outputs side by side
  auto dims = [&](int T, int beam, int vect = -1) {
    std::vector<int> d(4);
    d[CUDNN_SEQDATA_TIME_DIM] = T, d[CUDNN_SEQDATA_BATCH_DIM] = r.B, d[CUDNN_SEQDATA_BEAM_DIM] = beam;
    d[CUDNN_SEQDATA_VECT_DIM] = vect < 0 ? S : vect;
    return d;
  };
  CK(cudnnSetSeqDataDescriptor(qd, CUDNN_DATA_FLOAT, 4, dims(r.Tq, r.J).data(), axes, r.B * r.J, r.lq.data(), nullptr));
  CK(cudnnSetSeqDataDescriptor(kd, CUDNN_DATA_FLOAT, 4, dims(r.Tk, 1).data(), axes, r.B, r.lk.data(), nullptr));
  CK(cudnnSetSeqDataDescriptor(vd, CUDNN_DATA_FLOAT, 4, dims(r.Tk, 1).data(), axes, r.B, r.lk.data(), nullptr));
  CK(cudnnSetSeqDataDescriptor(od, CUDNN_DATA_FLOAT, 4, dims(r.Tq, r.J, out_size).data(), axes, r.B * r.J, r.lq.data(), nullptr));
  const size_t nq = (size_t)r.B * r.J * r.Tq * S, nk = (size_t)r.B * r.Tk * S;
  std::vector<float> hq(nq), hk(nk), hv(nk, 0.0f);
  const bool uniform = r.project;
  for (size_t i = 0; i < nq; ++i) hq[i] = uniform ? 0.0f : 0.3f * std::sin(0.7f * (float)i + 1.0f);
  for (size_t i = 0; i < nk; ++i) hk[i] = uniform ? 0.0f : 0.3f * std::cos(0.5f * (float)i + 2.0f);
  for (int b = 0; b < r.B; ++b)
    for (int k = 0; k < r.Tk; ++k) hv[((size_t)b * r.Tk + k) * S + k] = 1.0f;
  std::vector<float> hres(nq, 10.0f);
  Buf<float> q(hq), k(hk), v(hv), o((size_t)r.B * r.J * r.Tq * out_size), res(hres);
  Buf<int> dlq(r.lq), dlk(r.lk);
  const std::vector<int> lo(r.Tq, 0), hi(r.Tq, r.Tk);
  Buf<uint8_t> work(ws), reserve(rs);
  std::vector<host::State> ma = host::seeded(seed_a, threads), mp = host::seeded(seed_p, threads);
  CK(cudnnMultiHeadAttnForward(H, ad, -1, lo.data(), hi.data(), dlq.p, dlk.p, qd, q.p, r.project ? res.p : nullptr, kd, k.p, vd,
                               v.p, od, o.p, wb, w.p, ws, work.p, rs, reserve.p));
  cudaDeviceSynchronize();
  const std::vector<float> go = o.get();
  const size_t osz = (size_t)out_size;
  // The model's masks.
  std::vector<uint8_t> ka, kp;
  if (r.attn_drop) ka = host::draw(ma, (size_t)r.B * r.J * r.NH * r.Tq * r.Tk, r.p);
  if (r.post_drop) kp = host::draw(mp, (size_t)r.B * r.J * r.Tq * osz, r.p);
  const float scale = 1.0f / (1.0f - r.p);
  size_t wrong = 0, cells = 0;
  for (int b = 0; b < r.B; ++b)
    for (int j = 0; j < r.J; ++j)
      for (int t = 0; t < r.lq[(size_t)b * r.J + j]; ++t) {
        for (size_t c = 0; c < osz; ++c) {
          double want;
          if (r.project) {
            // Uniform attention over the valid keys, V and O identity: the output vector is 1 / lk per element ...
            double a = 1.0 / r.lk[b];
            // With V an identity matrix, element c of the output is key c's share of the attention (head 0 is the
            // only head here), kept or not by the attention dropout.
            if (r.attn_drop)
              a = ka[((((size_t)b * r.J + j) * r.NH + 0) * r.Tq + t) * r.Tk + c] && (int)c < r.lk[b] ? (1.0 / r.lk[b]) * scale : 0.0;
            else
              a = (int)c < r.lk[b] ? 1.0 / r.lk[b] : 0.0;
            if (r.post_drop) a = kp[(((size_t)b * r.J + j) * r.Tq + t) * osz + c] ? a * scale : 0.0;
            want = a + 10.0;
          } else {
            const int h = (int)(c / (size_t)S), kk = (int)(c % (size_t)S);
            if (kk >= r.lk[b]) {
              want = 0.0;
            } else {
              // softmax over the valid keys of q . k
              double m = -1e30, sum = 0.0, e[64];
              for (int k2 = 0; k2 < r.lk[b]; ++k2) {
                double s = 0.0;
                for (int ch = 0; ch < S; ++ch)
                  s += (double)hq[(((size_t)b * r.J + j) * r.Tq + t) * S + ch] * hk[((size_t)b * r.Tk + k2) * S + ch];
                e[k2] = s, m = std::fmax(m, s);
              }
              for (int k2 = 0; k2 < r.lk[b]; ++k2) e[k2] = std::exp(e[k2] - m), sum += e[k2];
              double pr = e[kk] / sum;
              if (r.attn_drop) pr = ka[((((size_t)b * r.J + j) * r.NH + h) * r.Tq + t) * r.Tk + kk] ? pr * scale : 0.0;
              if (r.post_drop) pr = kp[(((size_t)b * r.J + j) * r.Tq + t) * osz + c] ? pr * scale : 0.0;
              want = pr;
            }
          }
          const double got = go[(((size_t)b * r.J + j) * r.Tq + t) * osz + c];
          wrong += std::fabs(got - want) > 2e-5 * (1.0 + std::fabs(want));
          ++cells;
        }
      }
  char what[240];
  std::snprintf(what, sizeof what, "attention dropout, %s: the outputs follow the host model's masks", r.name);
  expect(what, wrong == 0 && cells > 0, (double)wrong);
  auto states_ok = [&](const Buf<uint8_t>& st, const std::vector<host::State>& m) {
    std::vector<uint32_t> words(ssz / 4);
    cudaMemcpy(words.data(), st.p, ssz, cudaMemcpyDeviceToHost);
    for (size_t t = 0; t < threads; ++t) {
      bool ok = words[t * 12] == m[t].d;
      for (int kk = 0; kk < 5; ++kk) ok &= words[t * 12 + 1 + kk] == m[t].v[kk];
      if (!ok) return false;
    }
    return true;
  };
  std::snprintf(what, sizeof what, "attention dropout, %s: the descriptors' states stand where the draws left them", r.name);
  expect(what, states_ok(st_attn, ma) && states_ok(st_post, mp));
  cudnnDestroySeqDataDescriptor(qd);
  cudnnDestroySeqDataDescriptor(kd);
  cudnnDestroySeqDataDescriptor(vd);
  cudnnDestroySeqDataDescriptor(od);
  cudnnDestroyAttnDescriptor(ad);
  cudnnDestroyDropoutDescriptor(da);
  cudnnDestroyDropoutDescriptor(dp);
}

int main() {
  if (cudnnCreate(&H) != CUDNN_STATUS_SUCCESS) {
    std::printf("SKIP: cudnnCreate failed\n");
    return 0;
  }
  std::printf("cuDNN %zu\n", cudnnGetVersion());
  one({2, 3, 1, 5, 4, true, false, false, {5, 5, 5}, {4, 4, 4}, 0.5f, "probabilities, 2 heads, 3 batches"});
  one({2, 3, 2, 5, 4, true, false, false, {5, 5, 5, 5, 5, 5}, {4, 4, 4}, 0.5f, "probabilities, 2 beams"});
  one({2, 3, 2, 5, 4, true, false, false, {5, 3, 4, 5, 2, 1}, {4, 2, 3}, 0.5f, "probabilities, 2 beams, padded sequences"});
  one({2, 3, 2, 5, 4, false, true, false, {5, 5, 5, 5, 5, 5}, {4, 4, 4}, 0.5f, "output vectors"});
  one({1, 2, 1, 4, 4, false, true, true, {4, 4}, {4, 4}, 0.5f, "output projection and residual: post dropout before the residual"});
  one({1, 2, 1, 4, 4, true, true, true, {4, 4}, {4, 4}, 0.5f, "both dropouts"});
  cudnnDestroy(H);
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
