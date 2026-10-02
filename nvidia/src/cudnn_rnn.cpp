// libvgpucudnn's recurrent networks: cuDNN's RNN API (the v8 forms PyTorch's
// nn.LSTM, nn.GRU and nn.RNN call) and the dropout descriptor it takes.
//
// As everywhere in this library, the math runs on the host over copies of the
// device buffers (see cudnn_api.cpp). What the API leaves to the library, and
// how it is done here:
//
//   - The weight space: for each pseudo-layer (layer x direction), the input
//     matrices of every gate, then the recurrent ones, then the input biases,
//     then the recurrent biases -- each gate's block contiguous with the next,
//     which PyTorch requires when it views one gate block as one parameter.
//     cudnnGetRNNWeightParams says where each is, as it does in cuDNN.
//   - Gate order is cuDNN's documented one, which is PyTorch's too: LSTM
//     input, forget, cell, output; GRU reset, update, new.
//   - A training forward pass keeps what the backward passes need in the
//     reserve space the caller allocated: every layer's input, the gates, the
//     cell and hidden states, and GRU's recurrent term; backward-data leaves
//     the gate gradients there for backward-weights.
//
// Supported: LSTM, GRU, and ReLU/tanh RNNs; one or two directions; any number
// of layers, with dropout between them in training (the classic API's
// generator, so a reseeded descriptor repeats its masks); no, single or
// double biases; padded (sequence- or batch-major) and packed sequences of
// varying lengths; LSTM recurrent projections (projSize < hiddenSize) and
// cell clipping; float, half, bfloat16 or double data, computed in double.
// Refused by name: skip-input mode.
//
// Measured on an RTX 3060 with cuDNN 9.27 and matched:
//   - The weight space holds every pseudo-layer's matrices first (input,
//     recurrent, then the projection [projSize, hiddenSize]), then every
//     pseudo-layer's biases. With a projection the recurrent matrices are
//     [4 * hidden, projSize], later layers' inputs projSize * directions wide,
//     h states and y projSize wide; c stays hiddenSize. linLayerID 8 is the
//     projection (no bias); IDs up to 2 * gates without a matrix answer with
//     NULL and zero-dimension descriptors, past that BAD_PARAM.
//   - Clipping limits the cell state where it is read, not where it is
//     stored: c_t = f * clip(c_{t-1}) + i * g (cx clipped too) and
//     h = o * tanh(clip(c_t)), while cy is the unclipped c_T (to 1e-7 against
//     a host model; storing clipped states misses by 0.06).
//   - Unpacked data with sequences shorter than maxSeqLength needs
//     CUDNN_RNN_PADDED_IO_ENABLED (else BAD_PARAM).
#include "cudnn_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <vector>

#include <cuda_runtime.h>

namespace {

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

using vgpu_cudnn::known;

// Every pass computes in double, whatever the data's type; the buffers a
// caller gives are converted on the way in and out. The reserve space is this
// library's own, laid out in reals.
using real = double;

cudnnStatus_t refuse(const char* fn, const char* why) {
  if (!quiet()) std::fprintf(stderr, "[vgpu] %s: %s\n", fn, why);
  return CUDNN_STATUS_NOT_SUPPORTED;
}

struct Rnn {
  bool set = false;  // cudnnSetRNNDescriptor_v8 has run
  cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;
  cudnnRNNMode_t mode = CUDNN_LSTM;
  cudnnDataType_t type = CUDNN_DATA_FLOAT;  // of the weights, states and data
  cudnnDataType_t math_prec = CUDNN_DATA_FLOAT;
  cudnnMathType_t math_type = CUDNN_DEFAULT_MATH;
  cudnnRNNInputMode_t input = CUDNN_LINEAR_INPUT;
  uint32_t aux = 0;
  size_t esize() const { return vgpu_cudnn::type_bytes(type); }
  cudnnRNNBiasMode_t bias = CUDNN_RNN_DOUBLE_BIAS;
  int dirs = 1, in = 0, hid = 0, proj = 0, layers = 1;
  // Cell clipping (cudnnRNNSetClip_v8/_v9).
  cudnnRNNClipMode_t clip = CUDNN_RNN_CLIP_NONE;
  cudnnNanPropagation_t clip_nan = CUDNN_NOT_PROPAGATE_NAN;
  double lclip = 0.0, rclip = 0.0;
  bool clips() const { return mode == CUDNN_LSTM && clip == CUDNN_RNN_CLIP_MINMAX; }
  // The bounds as the hardware applies them, in float (measured: a double
  // LSTM clipped at -0.3 agrees with float bounds, not double ones, to 1e-12).
  double lo() const { return static_cast<float>(lclip); }
  double hi() const { return static_cast<float>(rclip); }
  double clipped(double c) const { return clips() ? std::min(std::max(c, lo()), hi()) : c; }
  double clip_slope(double c) const { return !clips() || (c >= lo() && c <= hi()) ? 1.0 : 0.0; }
  // An LSTM's recurrent projection, and the width of h (and of y per direction).
  bool projects() const { return mode == CUDNN_LSTM && proj < hid; }
  int out() const { return projects() ? proj : hid; }
  // The dropout descriptor, read when a training pass runs: its probability
  // and generator are the caller's to change in between.
  vgpu_cudnn::DropoutDesc* drop = nullptr;
  float set_p = 0.0f;  // its probability when the RNN was set, for when it is gone
  bool drop_alive() const { return drop && known(drop); }
  real dropout() const { return drop_alive() ? drop->p : set_p; }
  int gates() const { return mode == CUDNN_LSTM ? 4 : mode == CUDNN_GRU ? 3 : 1; }
  bool input_bias() const { return bias == CUDNN_RNN_DOUBLE_BIAS || bias == CUDNN_RNN_SINGLE_INP_BIAS; }
  bool rec_bias() const { return bias == CUDNN_RNN_DOUBLE_BIAS || bias == CUDNN_RNN_SINGLE_REC_BIAS; }
  int layer_in(int layer) const { return layer == 0 ? in : out() * dirs; }
  // Where pseudo-layer pl's matrices start in the weight space, in elements:
  // input, recurrent, projection. Then all the biases, pseudo-layer by
  // pseudo-layer: input, recurrent.
  size_t mat_size(int pl) const {
    const size_t G = gates(), H = hid, I = layer_in(pl / dirs), O = out();
    return G * H * I + G * H * O + (projects() ? O * H : 0);
  }
  size_t mat(int pl) const {
    size_t off = 0;
    for (int p = 0; p < pl; ++p) off += mat_size(p);
    return off;
  }
  size_t bias_size() const { return (input_bias() ? gates() * hid : 0) + (rec_bias() ? gates() * hid : 0); }
  size_t bias_at(int pl) const { return mat(layers * dirs) + static_cast<size_t>(pl) * bias_size(); }
  size_t weights() const { return bias_at(layers * dirs); }
  // One pseudo-layer's pieces within a weight-space copy.
  template <class P>
  struct Parts {
    P* W;   // [G*H][I]
    P* R;   // [G*H][O]
    P* Pj;  // [O][H], or null
    P* bW;  // [G*H], or null
    P* bR;
  };
  // The same, as element offsets; npos where there is none.
  static constexpr size_t npos = ~size_t(0);
  struct Offsets { size_t W, R, Pj, bW, bR; };
  Offsets offsets(int pl) const {
    const size_t G = gates(), H = hid, I = layer_in(pl / dirs), O = out();
    Offsets o;
    o.W = mat(pl);
    o.R = o.W + G * H * I;
    o.Pj = projects() ? o.R + G * H * O : npos;
    o.bW = input_bias() ? bias_at(pl) : npos;
    o.bR = rec_bias() ? bias_at(pl) + (input_bias() ? G * H : 0) : npos;
    return o;
  }
  template <class P>
  Parts<P> parts(P* base, int pl) const {
    const Offsets o = offsets(pl);
    auto at = [&](size_t e) { return e == npos ? nullptr : base + e; };
    return Parts<P>{at(o.W), at(o.R), at(o.Pj), at(o.bW), at(o.bR)};
  }
};

struct Data {
  cudnnRNNDataLayout_t layout = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
  int T = 0, B = 0, V = 0;
  std::vector<int> len;
  cudnnDataType_t type = CUDNN_DATA_FLOAT;
  bool has_fill = false;
  real fill = 0.0f;
  // Where (t, b) is in the buffer, in elements; -1 for a position the
  // sequence does not reach in a packed layout.
  long offset(int t, int b) const {
    switch (layout) {
      case CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED: return (long)(b * T + t) * V;
      case CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED: {
        if (t >= len[b]) return -1;
        long row = 0;
        for (int s = 0; s < t; ++s)
          for (int k = 0; k < B; ++k) row += len[k] > s;
        return (row + b) * V;
      }
      default: return (long)(t * B + b) * V;
    }
  }
  size_t elements() const {
    if (layout != CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED) return (size_t)T * B * V;
    size_t n = 0;
    for (int l : len) n += (size_t)l;
    return n * V;
  }
};

// The library's one registry (cudnn_common.hpp): the dropout descriptor an
// RNN takes is the classic API's.
using vgpu_cudnn::track;
using vgpu_cudnn::untrack;

// n elements of a caller's buffer of type t, as reals, and back.
std::vector<real> fetch(const void* dev, size_t n, cudnnDataType_t t) {
  std::vector<real> h(n, 0.0);
  if (!dev || !n) return h;
  const size_t es = vgpu_cudnn::type_bytes(t);
  std::vector<uint8_t> raw(n * es);
  cudaMemcpy(raw.data(), dev, raw.size(), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < n; ++i) h[i] = vgpu_cudnn::decode(t, raw.data() + i * es);
  return h;
}
void store(void* dev, const std::vector<real>& h, cudnnDataType_t t) {
  if (!dev || h.empty()) return;
  const size_t es = vgpu_cudnn::type_bytes(t);
  std::vector<uint8_t> raw(h.size() * es);
  for (size_t i = 0; i < h.size(); ++i) vgpu_cudnn::encode(t, h[i], raw.data() + i * es);
  cudaMemcpy(dev, raw.data(), raw.size(), cudaMemcpyHostToDevice);
}
// The reserve space: reals, as this library lays it out.
std::vector<real> fetch_res(const void* dev, size_t n) {
  std::vector<real> h(n, 0.0);
  if (dev && n) cudaMemcpy(h.data(), dev, n * sizeof(real), cudaMemcpyDeviceToHost);
  return h;
}
void store_res(void* dev, const std::vector<real>& h) {
  if (dev && !h.empty()) cudaMemcpy(dev, h.data(), h.size() * sizeof(real), cudaMemcpyHostToDevice);
}
void drain(cudnnHandle_t h) {
  cudaStream_t s = nullptr;
  cudnnGetStream(h, &s);
  cudaStreamSynchronize(s);
}

real sigmoid(real x) { return 1.0f / (1.0f + std::exp(-x)); }

// Dense [T][B][V] from an RNN data buffer, positions past a sequence's end 0.
std::vector<real> unpack(const Data& d, const void* dev) {
  const auto raw = fetch(dev, d.elements(), d.type);
  std::vector<real> out((size_t)d.T * d.B * d.V, 0.0f);
  for (int t = 0; t < d.T; ++t)
    for (int b = 0; b < d.B; ++b) {
      if (t >= d.len[b]) continue;
      const long o = d.offset(t, b);
      std::copy_n(raw.begin() + o, d.V, out.begin() + ((size_t)t * d.B + b) * d.V);
    }
  return out;
}
// And back: positions a padded layout has past a sequence's end get the
// padding fill if one was given, and are left as they were otherwise.
void pack(const Data& d, void* dev, const std::vector<real>& dense) {
  auto raw = fetch(dev, d.elements(), d.type);
  for (int t = 0; t < d.T; ++t)
    for (int b = 0; b < d.B; ++b) {
      const long o = d.offset(t, b);
      if (o < 0) continue;
      if (t < d.len[b]) std::copy_n(dense.begin() + ((size_t)t * d.B + b) * d.V, d.V, raw.begin() + o);
      else if (d.has_fill) std::fill_n(raw.begin() + o, d.V, d.fill);
    }
  store(dev, raw, d.type);
}

// The reserve space's layout, in reals.
struct Reserve {
  size_t T, B, G, H, O;
  std::vector<size_t> input;   // per layer: [T][B][in_l]
  std::vector<size_t> mask;    // per layer but the last: dropout's scale on its output, [T][B][O*D]
  size_t per_pl_start = 0;     // then per pseudo-layer: gates, c, h, rn, dgi, dgr, hr, dh
  size_t pl_stride = 0;
  size_t total = 0;
  Reserve(const Rnn& r, int T_, int B_) : T(T_), B(B_), G(r.gates()), H(r.hid), O(r.out()) {
    size_t off = 0;
    for (int l = 0; l < r.layers; ++l) {
      input.push_back(off);
      off += T * B * r.layer_in(l);
    }
    for (int l = 0; l + 1 < r.layers; ++l) {
      mask.push_back(off);
      off += T * B * O * r.dirs;
    }
    per_pl_start = off;
    pl_stride = T * B * (G * H + H + O + H + G * H + G * H + H + O);
    total = off + pl_stride * r.layers * r.dirs;
  }
  size_t gates(int pl) const { return per_pl_start + pl * pl_stride; }
  size_t c(int pl) const { return gates(pl) + T * B * G * H; }    // the cell state, unclipped
  size_t h(int pl) const { return c(pl) + T * B * H; }           // the output state (projected)
  size_t rn(int pl) const { return h(pl) + T * B * O; }
  size_t dgi(int pl) const { return rn(pl) + T * B * H; }
  size_t dgr(int pl) const { return dgi(pl) + T * B * G * H; }
  size_t hr(int pl) const { return dgr(pl) + T * B * G * H; }    // o * tanh(c), before the projection
  size_t dh(int pl) const { return hr(pl) + T * B * H; }          // the gradient reaching h (projected)
};

// Checks the descriptors a call was given agree with each other.
cudnnStatus_t check(const Rnn& r, const Data& x, const Data& y) {
  if (!r.set || x.type != r.type || y.type != r.type) return CUDNN_STATUS_BAD_PARAM;
  if (x.V != r.in || y.V != r.out() * r.dirs || x.T != y.T || x.B != y.B) return CUDNN_STATUS_BAD_PARAM;
  if ((int)x.len.size() != x.B) return CUDNN_STATUS_BAD_PARAM;
  // Measured: unpacked sequences shorter than maxSeqLength need padded I/O.
  if (x.layout != CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED && !(r.aux & CUDNN_RNN_PADDED_IO_ENABLED))
    for (int l : x.len)
      if (l != x.T) return CUDNN_STATUS_BAD_PARAM;
  if (x.layout == CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED)
    for (int b = 1; b < x.B; ++b)
      if (x.len[b] > x.len[b - 1]) return refuse("cudnnRNNForward", "a packed batch must be sorted by length, longest first");
  return CUDNN_STATUS_SUCCESS;
}

// The step order of one direction for batch entry b: forward, or from its own
// last position back to the first.
inline int step_t(int dir, int s, int len) { return dir == 0 ? s : len - 1 - s; }

// Runs every layer forward. Writes y (dense), hy, cy, and the reserve when
// training (res != nullptr).
// In training, dropout scales each layer's output but the last's before the
// next layer reads it: by 0 or 1 / (1 - p), drawn from the dropout
// descriptor and kept in the reserve for the backward pass.
bool forward(const Rnn& r, const Data& xd, const std::vector<real>& x, const std::vector<real>& w,
             const std::vector<real>& hx, const std::vector<real>& cx, std::vector<real>* y,
             std::vector<real>* hy, std::vector<real>* cy, std::vector<real>* res, const Reserve& rv) {
  const int T = xd.T, B = xd.B, H = r.hid, G = r.gates(), D = r.dirs, O = r.out();
  std::vector<real> in = x;  // this layer's input, dense [T][B][I]
  for (int l = 0; l < r.layers; ++l) {
    const int I = r.layer_in(l);
    if (res) std::copy(in.begin(), in.end(), res->begin() + rv.input[l]);
    std::vector<real> out((size_t)T * B * O * D, 0.0f);
    for (int dir = 0; dir < D; ++dir) {
      const int pl = l * D + dir;
      const auto q = r.parts(w.data(), pl);
      for (int b = 0; b < B; ++b) {
        std::vector<real> h(hx.begin() + ((size_t)pl * B + b) * O, hx.begin() + ((size_t)pl * B + b + 1) * O);
        std::vector<real> c = r.mode == CUDNN_LSTM
                                   ? std::vector<real>(cx.begin() + ((size_t)pl * B + b) * H,
                                                        cx.begin() + ((size_t)pl * B + b + 1) * H)
                                   : std::vector<real>(H, 0.0f);
        std::vector<real> zi((size_t)G * H), zr((size_t)G * H), hr(H);
        for (int s = 0; s < xd.len[b]; ++s) {
          const int t = step_t(dir, s, xd.len[b]);
          const real* xt = in.data() + ((size_t)t * B + b) * I;
          for (int g = 0; g < G; ++g)
            for (int j = 0; j < H; ++j) {
              double a = q.bW ? q.bW[g * H + j] : 0.0, qq = q.bR ? q.bR[g * H + j] : 0.0;
              const real* wr = q.W + ((size_t)g * H + j) * I;
              for (int k = 0; k < I; ++k) a += (double)wr[k] * xt[k];
              const real* rr = q.R + ((size_t)g * H + j) * O;
              for (int k = 0; k < O; ++k) qq += (double)rr[k] * h[k];
              zi[g * H + j] = (real)a, zr[g * H + j] = (real)qq;
            }
          const size_t tb = (size_t)t * B + b;
          real* gates = res ? res->data() + rv.gates(pl) + tb * G * H : nullptr;
          for (int j = 0; j < H; ++j) {
            if (r.mode == CUDNN_LSTM) {
              const real i = sigmoid(zi[j] + zr[j]), f = sigmoid(zi[H + j] + zr[H + j]);
              const real gg = std::tanh(zi[2 * H + j] + zr[2 * H + j]), o = sigmoid(zi[3 * H + j] + zr[3 * H + j]);
              // Clipping applies where the state is read (see the top).
              c[j] = f * r.clipped(c[j]) + i * gg;
              hr[j] = o * std::tanh(r.clipped(c[j]));
              if (gates) gates[j] = i, gates[H + j] = f, gates[2 * H + j] = gg, gates[3 * H + j] = o;
            } else if (r.mode == CUDNN_GRU) {
              const real rg = sigmoid(zi[j] + zr[j]), z = sigmoid(zi[H + j] + zr[H + j]);
              const real rn = zr[2 * H + j];
              const real n = std::tanh(zi[2 * H + j] + rg * rn);
              hr[j] = (1 - z) * n + z * h[j];
              if (gates) gates[j] = rg, gates[H + j] = z, gates[2 * H + j] = n;
              if (res) (*res)[rv.rn(pl) + tb * H + j] = rn;
            } else {
              const real a = zi[j] + zr[j];
              hr[j] = r.mode == CUDNN_RNN_RELU ? std::max(0.0, a) : std::tanh(a);
              if (gates) gates[j] = hr[j];
            }
          }
          if (q.Pj) {
            for (int p = 0; p < O; ++p) {
              double a = 0.0;
              for (int j = 0; j < H; ++j) a += (double)q.Pj[(size_t)p * H + j] * hr[j];
              h[p] = (real)a;
            }
          } else {
            h = hr;
          }
          if (res) {
            std::copy(h.begin(), h.end(), res->begin() + rv.h(pl) + tb * O);
            std::copy(c.begin(), c.end(), res->begin() + rv.c(pl) + tb * H);
            std::copy(hr.begin(), hr.end(), res->begin() + rv.hr(pl) + tb * H);
          }
          std::copy(h.begin(), h.end(), out.begin() + tb * O * D + (size_t)dir * O);
        }
        std::copy(h.begin(), h.end(), hy->begin() + ((size_t)pl * B + b) * O);
        if (r.mode == CUDNN_LSTM) std::copy(c.begin(), c.end(), cy->begin() + ((size_t)pl * B + b) * H);
      }
    }
    if (res && l + 1 < r.layers) {
      const real p = r.dropout();
      real* m = res->data() + rv.mask[l];
      if (p > 0.0f) {
        std::vector<uint8_t> keep;
        if (!vgpu_cudnn::dropout_draw(r.drop, out.size(), &keep)) return false;
        const real scale = p < 1.0f ? 1.0f / (1.0f - p) : 0.0f;
        for (size_t i = 0; i < out.size(); ++i) m[i] = keep[i] ? scale : 0.0f, out[i] *= m[i];
      } else {
        std::fill(m, m + out.size(), 1.0f);
      }
    }
    in = std::move(out);
  }
  *y = std::move(in);
  return true;
}

struct Rnns {
  const Rnn* r;
  const Data* x;
  const Data* y;
};

// The h and c state descriptors, when given, must be [layers * dirs, batch,
// projSize or hiddenSize] and [layers * dirs, batch, hiddenSize] (measured:
// an h descriptor hiddenSize wide with a projection is BAD_PARAM).
bool state_shape(cudnnTensorDescriptor_t d, int a, int b, int c) {
  if (!d) return true;
  cudnnDataType_t t;
  int nb = 0, dims[8], strides[8];
  if (cudnnGetTensorNdDescriptor(d, 8, &t, &nb, dims, strides) != CUDNN_STATUS_SUCCESS || nb != 3) return true;
  return dims[0] == a && dims[1] == b && dims[2] == c;
}

cudnnStatus_t resolve(cudnnRNNDescriptor_t rd, cudnnRNNDataDescriptor_t xd, cudnnRNNDataDescriptor_t yd,
                      cudnnTensorDescriptor_t hd, cudnnTensorDescriptor_t cd, Rnns* out) {
  if (!known(rd) || !known(xd) || !known(yd)) return CUDNN_STATUS_BAD_PARAM;
  out->r = reinterpret_cast<const Rnn*>(rd);
  out->x = reinterpret_cast<const Data*>(xd);
  out->y = reinterpret_cast<const Data*>(yd);
  const Rnn& r = *out->r;
  const int L = r.layers * r.dirs, B = out->x->B;
  if (!state_shape(hd, L, B, r.out()) || (r.mode == CUDNN_LSTM && !state_shape(cd, L, B, r.hid)))
    return CUDNN_STATUS_BAD_PARAM;
  return check(r, *out->x, *out->y);
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

// The dropout descriptor and its calls are the classic API's (cudnn_api.cpp).

/* ---- descriptors ---- */

VGPU_EXPORT cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnRNNDescriptor_t>(track(new Rnn()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d);
  delete reinterpret_cast<Rnn*>(d);
  return CUDNN_STATUS_SUCCESS;
}
// Measured: projSize 0 is BAD_PARAM; projSize above hiddenSize, below it for
// anything but an LSTM, or with an algorithm other than STANDARD, is
// NOT_SUPPORTED.
VGPU_EXPORT cudnnStatus_t cudnnSetRNNDescriptor_v8(cudnnRNNDescriptor_t d, cudnnRNNAlgo_t algo, cudnnRNNMode_t mode,
                                                   cudnnRNNBiasMode_t bias, cudnnDirectionMode_t dir,
                                                   cudnnRNNInputMode_t input, cudnnDataType_t type,
                                                   cudnnDataType_t math, cudnnMathType_t math_type, int32_t in,
                                                   int32_t hid, int32_t proj, int32_t layers,
                                                   cudnnDropoutDescriptor_t drop, uint32_t aux) {
  if (!known(d) || in <= 0 || hid <= 0 || proj <= 0 || layers <= 0) return CUDNN_STATUS_BAD_PARAM;
  // cuDNN's pairs: float, half or bfloat16 data with float math (half also
  // with half math), double with double. All compute in double here.
  const bool pair = (math == CUDNN_DATA_FLOAT && (type == CUDNN_DATA_FLOAT || type == CUDNN_DATA_HALF ||
                                                 type == CUDNN_DATA_BFLOAT16)) ||
                    (math == CUDNN_DATA_HALF && type == CUDNN_DATA_HALF) ||
                    (math == CUDNN_DATA_DOUBLE && type == CUDNN_DATA_DOUBLE);
  if (!pair) return refuse("cudnnSetRNNDescriptor_v8", "the data and math types are not one of cuDNN's pairs");
  if (proj > hid) return refuse("cudnnSetRNNDescriptor_v8", "projSize is larger than hiddenSize");
  if (proj < hid && (mode != CUDNN_LSTM || algo != CUDNN_RNN_ALGO_STANDARD))
    return refuse("cudnnSetRNNDescriptor_v8", "a recurrent projection needs an LSTM and CUDNN_RNN_ALGO_STANDARD");
  if (input != CUDNN_LINEAR_INPUT) return refuse("cudnnSetRNNDescriptor_v8", "only CUDNN_LINEAR_INPUT is supported");
  auto* r = reinterpret_cast<Rnn*>(d);
  r->set = true;
  r->algo = algo;
  r->type = type;
  r->math_prec = math;
  r->math_type = math_type;
  r->input = input;
  r->aux = aux;
  r->mode = mode;
  r->bias = bias;
  r->dirs = dir == CUDNN_BIDIRECTIONAL ? 2 : 1;
  r->in = in, r->hid = hid, r->proj = proj, r->layers = layers;
  r->drop = known(drop) ? reinterpret_cast<vgpu_cudnn::DropoutDesc*>(drop) : nullptr;
  r->set_p = r->drop ? r->drop->p : 0.0f;
  return CUDNN_STATUS_SUCCESS;
}
// Measured: NOT_INITIALIZED for a descriptor never set; any output may be NULL.
VGPU_EXPORT cudnnStatus_t cudnnGetRNNDescriptor_v8(cudnnRNNDescriptor_t d, cudnnRNNAlgo_t* algo, cudnnRNNMode_t* mode,
                                                   cudnnRNNBiasMode_t* bias, cudnnDirectionMode_t* dir,
                                                   cudnnRNNInputMode_t* input, cudnnDataType_t* type,
                                                   cudnnDataType_t* math, cudnnMathType_t* math_type, int32_t* in,
                                                   int32_t* hid, int32_t* proj, int32_t* layers,
                                                   cudnnDropoutDescriptor_t* drop, uint32_t* aux) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  const Rnn& r = *reinterpret_cast<const Rnn*>(d);
  if (!r.set) return CUDNN_STATUS_NOT_INITIALIZED;
  if (algo) *algo = r.algo;
  if (mode) *mode = r.mode;
  if (bias) *bias = r.bias;
  if (dir) *dir = r.dirs == 2 ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
  if (input) *input = r.input;
  if (type) *type = r.type;
  if (math) *math = r.math_prec;
  if (math_type) *math_type = r.math_type;
  if (in) *in = r.in;
  if (hid) *hid = r.hid;
  if (proj) *proj = r.proj;
  if (layers) *layers = r.layers;
  if (drop) *drop = r.drop_alive() ? reinterpret_cast<cudnnDropoutDescriptor_t>(r.drop) : nullptr;
  if (aux) *aux = r.aux;
  return CUDNN_STATUS_SUCCESS;
}

// Cell clipping. Measured: lclip > rclip is BAD_PARAM whatever the mode; the
// mode and NaN option are stored unchecked; it may be set on any cell type
// (only an LSTM clips); _v9 sets NaN propagation on.
VGPU_EXPORT cudnnStatus_t cudnnRNNSetClip_v8(cudnnRNNDescriptor_t d, cudnnRNNClipMode_t mode,
                                             cudnnNanPropagation_t nan, double lclip, double rclip) {
  if (!known(d) || lclip > rclip) return CUDNN_STATUS_BAD_PARAM;
  auto* r = reinterpret_cast<Rnn*>(d);
  r->clip = mode, r->clip_nan = nan, r->lclip = lclip, r->rclip = rclip;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnRNNSetClip_v9(cudnnRNNDescriptor_t d, cudnnRNNClipMode_t mode, double lclip,
                                             double rclip) {
  return cudnnRNNSetClip_v8(d, mode, CUDNN_PROPAGATE_NAN, lclip, rclip);
}
VGPU_EXPORT cudnnStatus_t cudnnRNNGetClip_v8(cudnnRNNDescriptor_t d, cudnnRNNClipMode_t* mode,
                                             cudnnNanPropagation_t* nan, double* lclip, double* rclip) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  const Rnn& r = *reinterpret_cast<const Rnn*>(d);
  if (mode) *mode = r.clip;
  if (nan) *nan = r.clip_nan;
  if (lclip) *lclip = r.lclip;
  if (rclip) *rclip = r.rclip;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnRNNGetClip_v9(cudnnRNNDescriptor_t d, cudnnRNNClipMode_t* mode, double* lclip,
                                             double* rclip) {
  return cudnnRNNGetClip_v8(d, mode, nullptr, lclip, rclip);
}
// Prepares CUDNN_RNN_ALGO_PERSIST_DYNAMIC kernels for a batch size on the
// hardware; nothing to build here. (Measured: SUCCESS for any descriptor,
// set or not, and any algorithm.)
VGPU_EXPORT cudnnStatus_t cudnnBuildRNNDynamic(cudnnHandle_t h, cudnnRNNDescriptor_t d, int) {
  if (!known(h) || !known(d)) return CUDNN_STATUS_BAD_PARAM;
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnRNNDataDescriptor_t>(track(new Data()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d);
  delete reinterpret_cast<Data*>(d);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetRNNDataDescriptor(cudnnRNNDataDescriptor_t d, cudnnDataType_t type,
                                                    cudnnRNNDataLayout_t layout, int T, int B, int V,
                                                    const int lens[], void* fill) {
  if (!known(d) || T <= 0 || B <= 0 || V <= 0 || !lens) return CUDNN_STATUS_BAD_PARAM;
  if (type != CUDNN_DATA_FLOAT && type != CUDNN_DATA_HALF && type != CUDNN_DATA_BFLOAT16 && type != CUDNN_DATA_DOUBLE)
    return refuse("cudnnSetRNNDataDescriptor", "only float, half, bfloat16 and double data are supported");
  auto* x = reinterpret_cast<Data*>(d);
  x->type = type;
  x->layout = layout;
  x->T = T, x->B = B, x->V = V;
  x->len.assign(lens, lens + B);
  for (int l : x->len)
    if (l < 0 || l > T) return CUDNN_STATUS_BAD_PARAM;
  x->has_fill = fill != nullptr;
  if (fill) x->fill = vgpu_cudnn::decode(type, static_cast<const uint8_t*>(fill));
  return CUDNN_STATUS_SUCCESS;
}
// Measured: BAD_PARAM for a descriptor never set and for an array shorter than
// the batch; array entries past the batch are zeroed; the padding fill comes
// back in the data type (zero when none was given).
VGPU_EXPORT cudnnStatus_t cudnnGetRNNDataDescriptor(cudnnRNNDataDescriptor_t d, cudnnDataType_t* type,
                                                    cudnnRNNDataLayout_t* layout, int* T, int* B, int* V,
                                                    int requested, int lens[], void* fill) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  const Data& x = *reinterpret_cast<const Data*>(d);
  if (!x.T || requested < x.B || !lens) return CUDNN_STATUS_BAD_PARAM;
  if (type) *type = x.type;
  if (layout) *layout = x.layout;
  if (T) *T = x.T;
  if (B) *B = x.B;
  if (V) *V = x.V;
  for (int i = 0; i < requested; ++i) lens[i] = i < x.B ? x.len[i] : 0;
  if (fill) vgpu_cudnn::encode(x.type, x.has_fill ? x.fill : 0.0, static_cast<uint8_t*>(fill));
  return CUDNN_STATUS_SUCCESS;
}

/* ---- sizes and where the weights are ---- */

VGPU_EXPORT cudnnStatus_t cudnnGetRNNWeightSpaceSize(cudnnHandle_t, cudnnRNNDescriptor_t d, size_t* size) {
  if (!known(d) || !size) return CUDNN_STATUS_BAD_PARAM;
  *size = reinterpret_cast<const Rnn*>(d)->weights() * reinterpret_cast<const Rnn*>(d)->esize();
  return CUDNN_STATUS_SUCCESS;
}

// Where pseudo-layer pl's gate linLayerID lives: IDs below the gate count are
// the input matrices, the rest the recurrent ones, each with its bias; ID
// 2 * gates is an LSTM's projection. The descriptors say [1, hidden, columns]
// and [1, hidden, 1] ([1, projSize, hidden] for the projection); a matrix or
// bias that does not exist comes back NULL with a zero-dimension descriptor.
VGPU_EXPORT cudnnStatus_t cudnnGetRNNWeightParams(cudnnHandle_t, cudnnRNNDescriptor_t d, int32_t pl, size_t size,
                                                  const void* space, int32_t lin, cudnnTensorDescriptor_t mDesc,
                                                  void** mAddr, cudnnTensorDescriptor_t bDesc, void** bAddr) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  const Rnn& r = *reinterpret_cast<const Rnn*>(d);
  const int G = r.gates(), H = r.hid, O = r.out();
  if (pl < 0 || pl >= r.layers * r.dirs || lin < 0 || lin > 2 * G || size < r.weights() * r.esize())
    return CUDNN_STATUS_BAD_PARAM;
  auto* at = static_cast<const char*>(space);
  auto addr = [&](size_t elem) { return space ? const_cast<char*>(at + elem * r.esize()) : nullptr; };
  auto describe = [&](cudnnTensorDescriptor_t t, int rows, int cols) {
    const int dims[3] = {1, rows, cols}, strides[3] = {rows * cols, cols, 1};
    return !t || cudnnSetTensorNdDescriptor(t, r.type, 3, dims, strides) == CUDNN_STATUS_SUCCESS;
  };
  const Rnn::Offsets q = r.offsets(pl);
  if (lin == 2 * G) {  // the projection, or nothing
    if (bAddr) *bAddr = nullptr;
    if (bDesc) vgpu_cudnn::clear_tensor(bDesc);
    if (!r.projects()) {
      if (mAddr) *mAddr = nullptr;
      if (mDesc) vgpu_cudnn::clear_tensor(mDesc);
      return CUDNN_STATUS_SUCCESS;
    }
    if (mAddr) *mAddr = addr(q.Pj);
    return describe(mDesc, O, H) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
  }
  const bool rec = lin >= G;
  const int g = rec ? lin - G : lin;
  const int I = r.layer_in(pl / r.dirs);
  const int cols = rec ? O : I;
  const size_t m = (rec ? q.R : q.W) + (size_t)g * H * cols;
  const size_t b = rec ? q.bR : q.bW;
  const bool has_b = b != Rnn::npos;
  if (mAddr) *mAddr = addr(m);
  if (bAddr) *bAddr = has_b ? addr(b + (size_t)g * H) : nullptr;
  if (!describe(mDesc, H, cols)) return CUDNN_STATUS_BAD_PARAM;
  if (!has_b) {
    if (bDesc) vgpu_cudnn::clear_tensor(bDesc);
  } else if (!describe(bDesc, H, 1)) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnGetRNNTempSpaceSizes(cudnnHandle_t, cudnnRNNDescriptor_t d, cudnnForwardMode_t mode,
                                                    cudnnRNNDataDescriptor_t xd, size_t* work, size_t* reserve) {
  if (!known(d) || !known(xd)) return CUDNN_STATUS_BAD_PARAM;
  const Rnn& r = *reinterpret_cast<const Rnn*>(d);
  const Data& x = *reinterpret_cast<const Data*>(xd);
  if (work) *work = 0;
  if (reserve) *reserve = mode == CUDNN_FWD_MODE_TRAINING ? Reserve(r, x.T, x.B).total * sizeof(real) : 0;
  return CUDNN_STATUS_SUCCESS;
}

/* ---- the passes ---- */

VGPU_EXPORT cudnnStatus_t cudnnRNNForward(cudnnHandle_t h, cudnnRNNDescriptor_t rd, cudnnForwardMode_t mode,
                                          const int32_t*, cudnnRNNDataDescriptor_t xd, const void* x,
                                          cudnnRNNDataDescriptor_t yd, void* y, cudnnTensorDescriptor_t hd,
                                          const void* hx, void* hy, cudnnTensorDescriptor_t cd, const void* cx,
                                          void* cy, size_t wsize, const void* w, size_t, void*, size_t rsize,
                                          void* reserve) {
  Rnns d;
  if (const cudnnStatus_t s = resolve(rd, xd, yd, hd, cd, &d); s != CUDNN_STATUS_SUCCESS) return s;
  const Rnn& r = *d.r;
  const bool training = mode == CUDNN_FWD_MODE_TRAINING;
  if (training && r.layers > 1 && r.dropout() > 0.0 && !r.drop_alive())
    return refuse("cudnnRNNForward", "the dropout descriptor this RNN was set with has been destroyed");
  if (wsize < r.weights() * r.esize()) return CUDNN_STATUS_BAD_PARAM;
  const Reserve rv(r, d.x->T, d.x->B);
  if (training && (!reserve || rsize < rv.total * sizeof(real))) return CUDNN_STATUS_BAD_PARAM;
  drain(h);
  const size_t hs = (size_t)r.layers * r.dirs * d.x->B * r.out(), cs = (size_t)r.layers * r.dirs * d.x->B * r.hid;
  const auto hw = fetch(w, r.weights(), r.type);
  const auto hx_ = hx ? fetch(hx, hs, r.type) : std::vector<real>(hs, 0.0f);
  const auto cx_ = r.mode == CUDNN_LSTM && cx ? fetch(cx, cs, r.type) : std::vector<real>(cs, 0.0f);
  std::vector<real> dense_y, hy_(hs, 0.0f), cy_(cs, 0.0f);
  std::vector<real> res(training ? rv.total : 0, 0.0f);
  if (!forward(r, *d.x, unpack(*d.x, x), hw, hx_, cx_, &dense_y, &hy_, &cy_, training ? &res : nullptr, rv))
    return CUDNN_STATUS_EXECUTION_FAILED;
  pack(*d.y, y, dense_y);
  if (hy) store(hy, hy_, r.type);
  if (cy && r.mode == CUDNN_LSTM) store(cy, cy_, r.type);
  if (training) store_res(reserve, res);
  return CUDNN_STATUS_SUCCESS;
}

// Backpropagation through time, from the top layer down, each direction's
// steps in the reverse of the order they ran in. Gate gradients are left in
// the reserve for backward-weights: dgi for the input side, dgr for the
// recurrent side (they differ only in GRU's new gate), and with a
// projection the gradient reaching each projected h. The clip passes a
// gradient where the state it limited was inside its bounds.
VGPU_EXPORT cudnnStatus_t cudnnRNNBackwardData_v8(cudnnHandle_t h, cudnnRNNDescriptor_t rd, const int32_t*,
                                                  cudnnRNNDataDescriptor_t yd, const void*, const void* dy,
                                                  cudnnRNNDataDescriptor_t xd, void* dx, cudnnTensorDescriptor_t hd,
                                                  const void* hx, const void* dhy, void* dhx,
                                                  cudnnTensorDescriptor_t cd, const void* cx, const void* dcy,
                                                  void* dcx, size_t wsize, const void* w, size_t, void*,
                                                  size_t rsize, void* reserve) {
  Rnns d;
  if (const cudnnStatus_t s = resolve(rd, xd, yd, hd, cd, &d); s != CUDNN_STATUS_SUCCESS) return s;
  const Rnn& r = *d.r;
  const Reserve rv(r, d.x->T, d.x->B);
  if (wsize < r.weights() * r.esize() || !reserve || rsize < rv.total * sizeof(real)) return CUDNN_STATUS_BAD_PARAM;
  drain(h);
  const int T = d.x->T, B = d.x->B, H = r.hid, G = r.gates(), D = r.dirs, O = r.out();
  const size_t hs = (size_t)r.layers * D * B * O, cs = (size_t)r.layers * D * B * H;
  const auto hw = fetch(w, r.weights(), r.type);
  auto res = fetch_res(reserve, rv.total);
  const auto hx_ = hx ? fetch(hx, hs, r.type) : std::vector<real>(hs, 0.0f);
  const auto cx_ = r.mode == CUDNN_LSTM && cx ? fetch(cx, cs, r.type) : std::vector<real>(cs, 0.0f);
  const auto dhy_ = dhy ? fetch(dhy, hs, r.type) : std::vector<real>(hs, 0.0f);
  const auto dcy_ = r.mode == CUDNN_LSTM && dcy ? fetch(dcy, cs, r.type) : std::vector<real>(cs, 0.0f);
  std::vector<real> dhx_(hs, 0.0f), dcx_(cs, 0.0f);
  std::vector<real> dout = unpack(*d.y, dy);  // gradient of this layer's output, dense [T][B][O*D]
  for (int l = r.layers - 1; l >= 0; --l) {
    const int I = r.layer_in(l);
    std::vector<real> din((size_t)T * B * I, 0.0f);
    for (int dir = 0; dir < D; ++dir) {
      const int pl = l * D + dir;
      const auto q = r.parts(hw.data(), pl);
      for (int b = 0; b < B; ++b) {
        const int len = d.x->len[b];
        std::vector<real> dh(dhy_.begin() + ((size_t)pl * B + b) * O, dhy_.begin() + ((size_t)pl * B + b + 1) * O);
        std::vector<real> dc(dcy_.begin() + ((size_t)pl * B + b) * H, dcy_.begin() + ((size_t)pl * B + b + 1) * H);
        std::vector<real> dhr(H);
        for (int s = len - 1; s >= 0; --s) {
          const int t = step_t(dir, s, len);
          const size_t tb = (size_t)t * B + b;
          // The state before this step: the previous step's, or hx/cx.
          const bool first = s == 0;
          const int tp = first ? -1 : step_t(dir, s - 1, len);
          const real* hprev = first ? hx_.data() + ((size_t)pl * B + b) * O : res.data() + rv.h(pl) + ((size_t)tp * B + b) * O;
          const real* cprev = first ? cx_.data() + ((size_t)pl * B + b) * H : res.data() + rv.c(pl) + ((size_t)tp * B + b) * H;
          const real* gt = res.data() + rv.gates(pl) + tb * G * H;
          const real* ct = res.data() + rv.c(pl) + tb * H;
          const real* ht = res.data() + rv.h(pl) + tb * O;
          real* dgi = res.data() + rv.dgi(pl) + tb * G * H;
          real* dgr = res.data() + rv.dgr(pl) + tb * G * H;
          for (int j = 0; j < O; ++j) dh[j] += dout[tb * O * D + (size_t)dir * O + j];
          if (q.Pj) {
            std::copy(dh.begin(), dh.end(), res.begin() + rv.dh(pl) + tb * O);
            for (int j = 0; j < H; ++j) {
              double a = 0.0;
              for (int p = 0; p < O; ++p) a += (double)q.Pj[(size_t)p * H + j] * dh[p];
              dhr[j] = (real)a;
            }
          } else {
            std::copy(dh.begin(), dh.end(), dhr.begin());
          }
          std::vector<real> dhp(O, 0.0f), dcp(H, 0.0f);
          for (int j = 0; j < H; ++j) {
            if (r.mode == CUDNN_LSTM) {
              const real i = gt[j], f = gt[H + j], g = gt[2 * H + j], o = gt[3 * H + j];
              const real tc = std::tanh(r.clipped(ct[j])), cp = r.clipped(cprev[j]);
              const real dct = dc[j] + dhr[j] * o * (1 - tc * tc) * r.clip_slope(ct[j]);
              dgi[j] = dct * g * i * (1 - i);
              dgi[H + j] = dct * cp * f * (1 - f);
              dgi[2 * H + j] = dct * i * (1 - g * g);
              dgi[3 * H + j] = dhr[j] * tc * o * (1 - o);
              dcp[j] = dct * f * r.clip_slope(cprev[j]);
            } else if (r.mode == CUDNN_GRU) {
              const real rg = gt[j], z = gt[H + j], n = gt[2 * H + j];
              const real rn = res[rv.rn(pl) + tb * H + j];
              const real dn = dhr[j] * (1 - z), dz = dhr[j] * (hprev[j] - n);
              const real dan = dn * (1 - n * n);
              dgi[j] = dan * rn * rg * (1 - rg);
              dgi[H + j] = dz * z * (1 - z);
              dgi[2 * H + j] = dan;
              dhp[j] += dhr[j] * z;
            } else {
              const real a = ht[j];
              dgi[j] = dhr[j] * (r.mode == CUDNN_RNN_RELU ? (a > 0 ? 1.0f : 0.0f) : 1 - a * a);
            }
          }
          for (int k = 0; k < G * H; ++k) dgr[k] = dgi[k];
          if (r.mode == CUDNN_GRU)
            for (int j = 0; j < H; ++j) dgr[2 * H + j] = dgi[2 * H + j] * gt[j];  // through the reset gate
          // Into the previous state, and into this layer's input.
          for (int g = 0; g < G; ++g)
            for (int j = 0; j < H; ++j) {
              const real* rr = q.R + ((size_t)g * H + j) * O;
              for (int k = 0; k < O; ++k) dhp[k] += rr[k] * dgr[g * H + j];
              const real* wr = q.W + ((size_t)g * H + j) * I;
              real* di = din.data() + tb * I;
              for (int k = 0; k < I; ++k) di[k] += wr[k] * dgi[g * H + j];
            }
          dh = dhp;
          dc = dcp;
        }
        std::copy(dh.begin(), dh.end(), dhx_.begin() + ((size_t)pl * B + b) * O);
        if (r.mode == CUDNN_LSTM) std::copy(dc.begin(), dc.end(), dcx_.begin() + ((size_t)pl * B + b) * H);
      }
    }
    // Into the layer below, through the dropout that scaled its output.
    if (l > 0) {
      const real* m = res.data() + rv.mask[l - 1];
      for (size_t i = 0; i < din.size(); ++i) din[i] *= m[i];
    }
    dout = std::move(din);
  }
  pack(*d.x, dx, dout);
  if (dhx) store(dhx, dhx_, r.type);
  if (dcx && r.mode == CUDNN_LSTM) store(dcx, dcx_, r.type);
  store_res(reserve, res);
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnRNNBackwardWeights_v8(cudnnHandle_t h, cudnnRNNDescriptor_t rd, cudnnWgradMode_t add,
                                                     const int32_t*, cudnnRNNDataDescriptor_t xd, const void*,
                                                     cudnnTensorDescriptor_t hd, const void* hx,
                                                     cudnnRNNDataDescriptor_t yd, const void*, size_t wsize,
                                                     void* dw, size_t, void*, size_t rsize, void* reserve) {
  Rnns d;
  if (const cudnnStatus_t s = resolve(rd, xd, yd, hd, nullptr, &d); s != CUDNN_STATUS_SUCCESS) return s;
  const Rnn& r = *d.r;
  const Reserve rv(r, d.x->T, d.x->B);
  if (wsize < r.weights() * r.esize() || !reserve || rsize < rv.total * sizeof(real)) return CUDNN_STATUS_BAD_PARAM;
  drain(h);
  const int B = d.x->B, H = r.hid, G = r.gates(), D = r.dirs, O = r.out();
  const size_t hs = (size_t)r.layers * D * B * O;
  const auto res = fetch_res(reserve, rv.total);
  const auto hx_ = hx ? fetch(hx, hs, r.type) : std::vector<real>(hs, 0.0f);
  std::vector<real> g = add == CUDNN_WGRAD_MODE_ADD ? fetch(dw, r.weights(), r.type) : std::vector<real>(r.weights(), 0.0f);
  for (int l = 0; l < r.layers; ++l) {
    const int I = r.layer_in(l);
    const real* in = res.data() + rv.input[l];
    for (int dir = 0; dir < D; ++dir) {
      const int pl = l * D + dir;
      const auto q = r.parts(g.data(), pl);
      for (int b = 0; b < B; ++b) {
        const int len = d.x->len[b];
        for (int s = 0; s < len; ++s) {
          const int t = step_t(dir, s, len);
          const size_t tb = (size_t)t * B + b;
          const real* hprev = s == 0 ? hx_.data() + ((size_t)pl * B + b) * O
                                      : res.data() + rv.h(pl) + ((size_t)step_t(dir, s - 1, len) * B + b) * O;
          const real* dgi = res.data() + rv.dgi(pl) + tb * G * H;
          const real* dgr = res.data() + rv.dgr(pl) + tb * G * H;
          const real* xt = in + tb * I;
          for (int k = 0; k < G * H; ++k) {
            for (int c = 0; c < I; ++c) q.W[(size_t)k * I + c] += dgi[k] * xt[c];
            for (int c = 0; c < O; ++c) q.R[(size_t)k * O + c] += dgr[k] * hprev[c];
            if (q.bW) q.bW[k] += dgi[k];
            if (q.bR) q.bR[k] += dgr[k];
          }
          if (q.Pj) {
            const real* hr = res.data() + rv.hr(pl) + tb * H;
            const real* dhp = res.data() + rv.dh(pl) + tb * O;
            for (int p = 0; p < O; ++p)
              for (int j = 0; j < H; ++j) q.Pj[(size_t)p * H + j] += dhp[p] * hr[j];
          }
        }
      }
    }
  }
  store(dw, g, r.type);
  return CUDNN_STATUS_SUCCESS;
}
