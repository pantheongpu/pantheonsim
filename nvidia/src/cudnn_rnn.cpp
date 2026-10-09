// libvgpucudnn's recurrent networks: cuDNN's RNN API (the v8 forms PyTorch's
// nn.LSTM, nn.GRU and nn.RNN call) and the dropout descriptor it takes; and,
// at the end, the classic multi-head attention API, the other half of
// cuDNN's adv library.
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
  int D;
  std::vector<size_t> input;   // per layer and direction: [T][B][in_l], the layer's input after the dropout on it
  std::vector<size_t> mask;    // per layer but the last, and direction: dropout's scale on its output, [T][B][O*D]
  size_t per_pl_start = 0;     // then per pseudo-layer: gates, c, h, rn, dgi, dgr, hr, dh
  size_t pl_stride = 0;
  size_t total = 0;
  Reserve(const Rnn& r, int T_, int B_) : T(T_), B(B_), G(r.gates()), H(r.hid), O(r.out()), D(r.dirs) {
    size_t off = 0;
    for (int l = 0; l < r.layers; ++l)
      for (int d = 0; d < D; ++d) {
        input.push_back(off);
        off += T * B * r.layer_in(l);
      }
    for (int l = 0; l + 1 < r.layers; ++l)
      for (int d = 0; d < D; ++d) {
        mask.push_back(off);
        off += T * B * O * D;
      }
    per_pl_start = off;
    pl_stride = T * B * (G * H + H + O + H + G * H + G * H + H + O);
    total = off + pl_stride * r.layers * r.dirs;
  }
  size_t input_at(int l, int dir) const { return input[(size_t)l * D + dir]; }
  size_t mask_at(int l, int dir) const { return mask[(size_t)l * D + dir]; }
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
// descriptor and kept in the reserve for the backward pass. Each direction of
// the next layer gets a mask of its own over the whole of the lower layer's
// output, forward direction first -- as cuDNN draws them (measured on an RTX
// 3060 through the descriptor's states: one grid-stride application of the
// dropout kernel over [T][B][O*D] per direction, layer after layer).
bool forward(const Rnn& r, const Data& xd, const std::vector<real>& x, const std::vector<real>& w,
             const std::vector<real>& hx, const std::vector<real>& cx, std::vector<real>* y,
             std::vector<real>* hy, std::vector<real>* cy, std::vector<real>* res, const Reserve& rv) {
  const int T = xd.T, B = xd.B, H = r.hid, G = r.gates(), D = r.dirs, O = r.out();
  std::vector<std::vector<real>> ins(D, x);  // each direction's input to this layer, dense [T][B][I]
  std::vector<real> out;
  for (int l = 0; l < r.layers; ++l) {
    const int I = r.layer_in(l);
    if (res)
      for (int dir = 0; dir < D; ++dir) std::copy(ins[dir].begin(), ins[dir].end(), res->begin() + rv.input_at(l, dir));
    out.assign((size_t)T * B * O * D, 0.0f);
    for (int dir = 0; dir < D; ++dir) {
      const std::vector<real>& in = ins[dir];
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
    if (l + 1 < r.layers) {
      const real p = r.dropout();
      for (int dir = 0; dir < D; ++dir) {
        ins[dir] = out;
        if (!res) continue;  // inference: no dropout
        real* m = res->data() + rv.mask_at(l, dir);
        if (p > 0.0f) {
          std::vector<uint8_t> keep;
          if (!vgpu_cudnn::dropout_draw(r.drop, out.size(), &keep)) return false;
          const real scale = p < 1.0f ? 1.0f / (1.0f - p) : 0.0f;
          for (size_t i = 0; i < out.size(); ++i) m[i] = keep[i] ? scale : 0.0f, ins[dir][i] *= m[i];
        } else {
          std::fill(m, m + out.size(), 1.0f);
        }
      }
    }
  }
  *y = std::move(out);
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
    std::vector<std::vector<real>> dind(D, std::vector<real>((size_t)T * B * I, 0.0f));  // each direction's input gradient
    for (int dir = 0; dir < D; ++dir) {
      std::vector<real>& din = dind[dir];
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
    // Into the layer below, each direction's gradient through the dropout that scaled
    // the output for it.
    std::vector<real> below((size_t)T * B * I, 0.0f);
    for (int dir = 0; dir < D; ++dir) {
      const real* m = l > 0 ? res.data() + rv.mask_at(l - 1, dir) : nullptr;
      for (size_t i = 0; i < below.size(); ++i) below[i] += m ? dind[dir][i] * m[i] : dind[dir][i];
    }
    dout = std::move(below);
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
    for (int dir = 0; dir < D; ++dir) {
      const real* in = res.data() + rv.input_at(l, dir);
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

/* ---- multi-head attention (the classic API) ---- */

// cudnnMultiHeadAttnForward and its gradients, computed on the host as the
// documentation's equations give them: per head i,
//   h_i = sum over the window of softmax(smScaler (W_K,i k + b) . (W_Q,i q + b)) (W_V,i v + b)
//   out = sum_i W_O,i h_i + b_O (+ residuals)
// with a projection of size 0 meaning none (every head sees the whole
// vector). Measured on an RTX 3060 (cuDNN 9.27) and matched (to 1e-7):
//   - The weight buffer holds W_Q, W_K, W_V, W_O, then the biases b_Q, b_K,
//     b_V, b_O (one b_O for all heads), only those that exist. W_Q is
//     [nHeads, qProjSize, qSize] with strides [qProjSize, 1, nHeads *
//     qProjSize] (input element slowest), W_K and W_V likewise, W_O
//     [nHeads, oProjSize, vProj] with strides [oProjSize * vProj, 1,
//     oProjSize]; biases [nHeads, proj, 1], b_O [1, oProjSize, 1].
//   - The window of query step t is [loWinIdx[t], hiWinIdx[t]) clipped to the
//     key sequence's length; query steps past their sequence's length come
//     out as b_O plus the residual (zeros without either); with currIdx >= 0
//     (inference only: BAD_PARAM with a reserve space) only that step is
//     written.
//   - Status codes: qProjSize != kProjSize, a negative smScaler, unequal q
//     and k sizes without projections, or a type other than half, float or
//     double are BAD_PARAM; ONE_TO_ONE query mapping with beams is
//     NOT_SUPPORTED; a sequence descriptor of other than 4 dimensions is
//     NOT_SUPPORTED, one whose innermost axis is not VECT or whose length
//     array is short is BAD_PARAM.
// Dropout (the attention and post-attention descriptors) uses this library's
// generator, as the RNNs do: the fraction kept and the scaling are cuDNN's,
// the mask is not the hardware's.
namespace {

struct Attn {
  bool set = false;
  unsigned mode = 0;
  int heads = 0;
  double sm = 1.0;
  cudnnDataType_t type = CUDNN_DATA_FLOAT, prec = CUDNN_DATA_FLOAT;
  cudnnMathType_t math = CUDNN_DEFAULT_MATH;
  vgpu_cudnn::DropoutDesc *attn_drop = nullptr, *post_drop = nullptr;
  int qS = 0, kS = 0, vS = 0, qP = 0, kP = 0, vP = 0, oP = 0, Tq = 0, Tk = 0, maxB = 0, maxBeam = 0;
  bool biases() const { return mode & CUDNN_ATTN_ENABLE_PROJ_BIASES; }
  int qe() const { return qP ? qP : qS; }   // q after projection (= k after projection)
  int ve() const { return vP ? vP : vS; }
  int oS() const { return oP ? oP : heads * ve(); }
  size_t esize() const { return vgpu_cudnn::type_bytes(type); }
  // Element offsets of the eight tensors, npos where absent; the total.
  static constexpr size_t npos = ~size_t(0);
  void offsets(size_t* o, size_t* total) const {
    size_t at = 0;
    auto take = [&](bool has, size_t n) {
      const size_t r = has ? at : npos;
      if (has) at += n;
      return r;
    };
    const size_t H = static_cast<size_t>(heads);
    o[0] = take(qP > 0, H * qP * qS);
    o[1] = take(kP > 0, H * kP * kS);
    o[2] = take(vP > 0, H * vP * vS);
    o[3] = take(oP > 0, H * oP * ve());
    o[4] = take(biases() && qP > 0, H * qP);
    o[5] = take(biases() && kP > 0, H * kP);
    o[6] = take(biases() && vP > 0, H * vP);
    o[7] = take(biases() && oP > 0, static_cast<size_t>(oP));
    *total = at;
  }
};

struct SeqData {
  bool set = false;
  cudnnDataType_t type = CUDNN_DATA_FLOAT;
  int dims[CUDNN_SEQDATA_DIM_COUNT] = {};
  cudnnSeqDataAxis_t axes[CUDNN_SEQDATA_DIM_COUNT] = {};
  std::vector<int> lens;
  size_t stride[CUDNN_SEQDATA_DIM_COUNT] = {};
  int T() const { return dims[CUDNN_SEQDATA_TIME_DIM]; }
  int B() const { return dims[CUDNN_SEQDATA_BATCH_DIM]; }
  int beam() const { return dims[CUDNN_SEQDATA_BEAM_DIM]; }
  int V() const { return dims[CUDNN_SEQDATA_VECT_DIM]; }
  size_t count() const { return static_cast<size_t>(T()) * B() * beam() * V(); }
  // Element offset of vector (t, b, j).
  size_t at(int t, int b, int j) const {
    return stride[CUDNN_SEQDATA_TIME_DIM] * t + stride[CUDNN_SEQDATA_BATCH_DIM] * b + stride[CUDNN_SEQDATA_BEAM_DIM] * j;
  }
  int len(int b, int j) const { return lens[static_cast<size_t>(b) * beam() + j]; }
};

const Attn* attn(const void* d) { return known(d) && static_cast<const Attn*>(d)->set ? static_cast<const Attn*>(d) : nullptr; }
const SeqData* seq(const void* d) { return known(d) && static_cast<const SeqData*>(d)->set ? static_cast<const SeqData*>(d) : nullptr; }

// The dropout masks a training pass keeps, in the reserve: one byte per
// attention probability (query step, head, key step) and per output element.
// Then the attention windows, which cudnnMultiHeadAttnBackwardWeights is not
// given.
struct AttnReserve {
  size_t probs = 0, outs = 0, wins = 0;
  AttnReserve(const Attn& a) {
    probs = static_cast<size_t>(a.maxB) * a.maxBeam * a.Tq * a.heads * a.Tk;
    outs = static_cast<size_t>(a.maxB) * a.maxBeam * a.Tq * a.oS();
    wins = (probs + outs + 3) / 4 * 4;  // int-aligned
  }
  size_t win_ints(const Attn& a) const { return 2 * static_cast<size_t>(a.Tq); }
  size_t bytes(const Attn& a) const { return wins + win_ints(a) * sizeof(int); }
};

// One forward evaluation, keeping what the gradients need.
struct AttnPass {
  const Attn& a;
  const SeqData &q, &k, &v, &o;
  std::vector<real> w, Q, K, V;
  size_t off[8], wtotal = 0;
  AttnPass(const Attn& a_, const SeqData& q_, const SeqData& k_, const SeqData& v_, const SeqData& o_)
      : a(a_), q(q_), k(k_), v(v_), o(o_) {
    a.offsets(off, &wtotal);
  }
  // Projections: W (offset, proj P, input S) applied to vector x for head h.
  void project(int which, int P, int S, int h, const real* x, real* y) const {
    const size_t base = off[which], bias = off[which + 4];
    for (int p = 0; p < P; ++p) {
      double s = bias == Attn::npos ? 0.0 : w[bias + (size_t)h * P + p];
      for (int c = 0; c < S; ++c) s += w[base + (size_t)h * P + p + (size_t)c * a.heads * P] * x[c];
      y[p] = (real)s;
    }
  }
  void qbar(int h, const real* x, real* y) const {
    if (a.qP) project(0, a.qP, a.qS, h, x, y);
    else std::copy(x, x + a.qS, y);
  }
  void kbar(int h, const real* x, real* y) const {
    if (a.kP) project(1, a.kP, a.kS, h, x, y);
    else std::copy(x, x + a.kS, y);
  }
  void vbar(int h, const real* x, real* y) const {
    if (a.vP) project(2, a.vP, a.vS, h, x, y);
    else std::copy(x, x + a.vS, y);
  }
  int kv_beam(int j) const { return k.beam() == 1 ? 0 : j; }
  // The key steps of query step t's window for batch b, beam j.
  void window(const int* lo, const int* hi, int t, int b, int j, int* first, int* last) const {
    const int kl = k.len(b, kv_beam(j));
    *first = std::max(lo[t], 0);
    *last = std::min(hi[t], kl);
  }
};

}  // namespace

VGPU_EXPORT cudnnStatus_t cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnAttnDescriptor_t>(track(new Attn()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d);
  delete reinterpret_cast<Attn*>(d);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetAttnDescriptor(cudnnAttnDescriptor_t d, unsigned mode, int heads, double sm,
                                                 cudnnDataType_t type, cudnnDataType_t prec, cudnnMathType_t math,
                                                 cudnnDropoutDescriptor_t attn_drop, cudnnDropoutDescriptor_t post_drop,
                                                 int qS, int kS, int vS, int qP, int kP, int vP, int oP, int Tq, int Tk,
                                                 int maxB, int maxBeam) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  if (heads <= 0 || qS <= 0 || kS <= 0 || vS <= 0 || Tq <= 0 || Tk <= 0 || maxB <= 0 || maxBeam <= 0 || qP < 0 ||
      kP < 0 || vP < 0 || oP < 0 || !(sm >= 0.0))
    return CUDNN_STATUS_BAD_PARAM;
  if (qP != kP || (!qP && qS != kS)) return CUDNN_STATUS_BAD_PARAM;
  const bool pair = (type == CUDNN_DATA_DOUBLE && prec == CUDNN_DATA_DOUBLE) ||
                    (type == CUDNN_DATA_FLOAT && prec == CUDNN_DATA_FLOAT) ||
                    (type == CUDNN_DATA_HALF && (prec == CUDNN_DATA_HALF || prec == CUDNN_DATA_FLOAT));
  if (!pair) return CUDNN_STATUS_BAD_PARAM;
  if ((mode & CUDNN_ATTN_QUERYMAP_ONE_TO_ONE) && maxBeam > 1)
    return refuse("cudnnSetAttnDescriptor", "ONE_TO_ONE query mapping with beams (NOT_SUPPORTED on the hardware measured)");
  auto* a = reinterpret_cast<Attn*>(d);
  a->set = true;
  a->mode = mode, a->heads = heads, a->sm = sm, a->type = type, a->prec = prec, a->math = math;
  a->attn_drop = known(attn_drop) ? reinterpret_cast<vgpu_cudnn::DropoutDesc*>(attn_drop) : nullptr;
  a->post_drop = known(post_drop) ? reinterpret_cast<vgpu_cudnn::DropoutDesc*>(post_drop) : nullptr;
  a->qS = qS, a->kS = kS, a->vS = vS, a->qP = qP, a->kP = kP, a->vP = vP, a->oP = oP;
  a->Tq = Tq, a->Tk = Tk, a->maxB = maxB, a->maxBeam = maxBeam;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetAttnDescriptor(cudnnAttnDescriptor_t d, unsigned* mode, int* heads, double* sm,
                                                 cudnnDataType_t* type, cudnnDataType_t* prec, cudnnMathType_t* math,
                                                 cudnnDropoutDescriptor_t* attn_drop,
                                                 cudnnDropoutDescriptor_t* post_drop, int* qS, int* kS, int* vS,
                                                 int* qP, int* kP, int* vP, int* oP, int* Tq, int* Tk, int* maxB,
                                                 int* maxBeam) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  const Attn& a = *reinterpret_cast<const Attn*>(d);
  if (mode) *mode = a.mode;
  if (heads) *heads = a.heads;
  if (sm) *sm = a.sm;
  if (type) *type = a.type;
  if (prec) *prec = a.prec;
  if (math) *math = a.math;
  if (attn_drop) *attn_drop = reinterpret_cast<cudnnDropoutDescriptor_t>(a.attn_drop);
  if (post_drop) *post_drop = reinterpret_cast<cudnnDropoutDescriptor_t>(a.post_drop);
  if (qS) *qS = a.qS;
  if (kS) *kS = a.kS;
  if (vS) *vS = a.vS;
  if (qP) *qP = a.qP;
  if (kP) *kP = a.kP;
  if (vP) *vP = a.vP;
  if (oP) *oP = a.oP;
  if (Tq) *Tq = a.Tq;
  if (Tk) *Tk = a.Tk;
  if (maxB) *maxB = a.maxB;
  if (maxBeam) *maxBeam = a.maxBeam;
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnSeqDataDescriptor_t>(track(new SeqData()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroySeqDataDescriptor(cudnnSeqDataDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d);
  delete reinterpret_cast<SeqData*>(d);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetSeqDataDescriptor(cudnnSeqDataDescriptor_t d, cudnnDataType_t type, int nb,
                                                    const int dims[], const cudnnSeqDataAxis_t axes[], size_t nlens,
                                                    const int lens[], void*) {
  if (!known(d) || !dims || !axes || nb <= 0) return CUDNN_STATUS_BAD_PARAM;
  if (nb != CUDNN_SEQDATA_DIM_COUNT) return refuse("cudnnSetSeqDataDescriptor", "only 4 dimensions");
  if (type != CUDNN_DATA_HALF && type != CUDNN_DATA_FLOAT && type != CUDNN_DATA_DOUBLE) return CUDNN_STATUS_BAD_PARAM;
  SeqData s;
  s.set = true;
  s.type = type;
  bool seen[CUDNN_SEQDATA_DIM_COUNT] = {};
  for (int i = 0; i < CUDNN_SEQDATA_DIM_COUNT; ++i) {
    if (dims[i] <= 0) return CUDNN_STATUS_BAD_PARAM;
    s.dims[i] = dims[i];
    const int ax = axes[i];
    if (ax < 0 || ax >= CUDNN_SEQDATA_DIM_COUNT || seen[ax]) return CUDNN_STATUS_BAD_PARAM;
    seen[ax] = true;
    s.axes[i] = axes[i];
  }
  if (axes[CUDNN_SEQDATA_DIM_COUNT - 1] != CUDNN_SEQDATA_VECT_DIM) return CUDNN_STATUS_BAD_PARAM;
  const size_t want = static_cast<size_t>(s.B()) * s.beam();
  if (nlens != want || !lens) return CUDNN_STATUS_BAD_PARAM;
  s.lens.assign(lens, lens + want);
  for (int l : s.lens)
    if (l < 0 || l > s.T()) return CUDNN_STATUS_BAD_PARAM;
  size_t stride = 1;
  for (int i = CUDNN_SEQDATA_DIM_COUNT; i-- > 0;) {
    s.stride[s.axes[i]] = stride;
    stride *= static_cast<size_t>(s.dims[s.axes[i]]);
  }
  *reinterpret_cast<SeqData*>(d) = s;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetSeqDataDescriptor(const cudnnSeqDataDescriptor_t d, cudnnDataType_t* type, int* nb,
                                                    int requested, int dims[], cudnnSeqDataAxis_t axes[],
                                                    size_t* nlens, size_t requested_lens, int lens[], void* fill) {
  const SeqData* s = seq(d);
  if (!s) return CUDNN_STATUS_BAD_PARAM;
  if (type) *type = s->type;
  if (nb) *nb = CUDNN_SEQDATA_DIM_COUNT;
  for (int i = 0; i < std::min(requested, CUDNN_SEQDATA_DIM_COUNT); ++i) {
    if (dims) dims[i] = s->dims[i];
    if (axes) axes[i] = s->axes[i];
  }
  if (nlens) *nlens = s->lens.size();
  for (size_t i = 0; lens && i < std::min(requested_lens, s->lens.size()); ++i) lens[i] = s->lens[i];
  if (fill) vgpu_cudnn::encode(s->type, 0.0, static_cast<uint8_t*>(fill));
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnGetMultiHeadAttnBuffers(cudnnHandle_t h, const cudnnAttnDescriptor_t d, size_t* wbytes,
                                                       size_t* work, size_t* reserve) {
  const Attn* a = attn(d);
  if (!known(h) || !a || !wbytes || !work) return CUDNN_STATUS_BAD_PARAM;
  size_t off[8], total = 0;
  a->offsets(off, &total);
  *wbytes = total * a->esize();
  *work = 0;
  if (reserve) *reserve = AttnReserve(*a).bytes(*a);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetMultiHeadAttnWeights(cudnnHandle_t h, const cudnnAttnDescriptor_t d,
                                                       cudnnMultiHeadAttnWeightKind_t kind, size_t wbytes,
                                                       const void* weights, cudnnTensorDescriptor_t wd, void** addr) {
  const Attn* a = attn(d);
  if (!known(h) || !a || !wd || !addr || kind < 0 || kind >= CUDNN_ATTN_WKIND_COUNT) return CUDNN_STATUS_BAD_PARAM;
  size_t off[8], total = 0;
  a->offsets(off, &total);
  if (wbytes < total * a->esize()) return CUDNN_STATUS_BAD_PARAM;
  if (off[kind] == Attn::npos) {
    *addr = nullptr;
    vgpu_cudnn::clear_tensor(wd);
    return CUDNN_STATUS_SUCCESS;
  }
  const int H = a->heads;
  int dims[3], str[3];
  switch (kind) {
    case CUDNN_MH_ATTN_Q_WEIGHTS: dims[0] = H, dims[1] = a->qP, dims[2] = a->qS, str[0] = a->qP, str[1] = 1, str[2] = H * a->qP; break;
    case CUDNN_MH_ATTN_K_WEIGHTS: dims[0] = H, dims[1] = a->kP, dims[2] = a->kS, str[0] = a->kP, str[1] = 1, str[2] = H * a->kP; break;
    case CUDNN_MH_ATTN_V_WEIGHTS: dims[0] = H, dims[1] = a->vP, dims[2] = a->vS, str[0] = a->vP, str[1] = 1, str[2] = H * a->vP; break;
    case CUDNN_MH_ATTN_O_WEIGHTS:
      dims[0] = H, dims[1] = a->oP, dims[2] = a->ve(), str[0] = a->oP * a->ve(), str[1] = 1, str[2] = a->oP;
      break;
    case CUDNN_MH_ATTN_Q_BIASES: dims[0] = H, dims[1] = a->qP, dims[2] = 1, str[0] = a->qP, str[1] = 1, str[2] = 1; break;
    case CUDNN_MH_ATTN_K_BIASES: dims[0] = H, dims[1] = a->kP, dims[2] = 1, str[0] = a->kP, str[1] = 1, str[2] = 1; break;
    case CUDNN_MH_ATTN_V_BIASES: dims[0] = H, dims[1] = a->vP, dims[2] = 1, str[0] = a->vP, str[1] = 1, str[2] = 1; break;
    default: dims[0] = 1, dims[1] = a->oP, dims[2] = 1, str[0] = a->oP, str[1] = 1, str[2] = 1; break;
  }
  if (cudnnSetTensorNdDescriptor(wd, a->type, 3, dims, str) != CUDNN_STATUS_SUCCESS) return CUDNN_STATUS_BAD_PARAM;
  *addr = weights ? const_cast<char*>(static_cast<const char*>(weights) + off[kind] * a->esize()) : nullptr;
  return CUDNN_STATUS_SUCCESS;
}

namespace {
// The descriptors of one call, checked against each other and the attention
// descriptor.
cudnnStatus_t attn_check(const Attn* a, const SeqData* q, const SeqData* k, const SeqData* v, const SeqData* o) {
  if (!a || !q || !k || !v || !o) return CUDNN_STATUS_BAD_PARAM;
  for (const SeqData* s : {q, k, v, o}) {
    if (s->type != a->type) return CUDNN_STATUS_BAD_PARAM;
    for (int i = 0; i < CUDNN_SEQDATA_DIM_COUNT; ++i)
      if (s->axes[i] != q->axes[i]) return CUDNN_STATUS_BAD_PARAM;  // one layout for all
    if (s->B() > a->maxB || s->beam() > a->maxBeam) return CUDNN_STATUS_BAD_PARAM;
  }
  if (q->V() != a->qS || k->V() != a->kS || v->V() != a->vS || o->V() != a->oS()) return CUDNN_STATUS_BAD_PARAM;
  if (q->T() > a->Tq || k->T() > a->Tk || o->T() != q->T() || v->T() != k->T()) return CUDNN_STATUS_BAD_PARAM;
  if (k->B() != q->B() || v->B() != q->B() || o->B() != q->B() || o->beam() != q->beam()) return CUDNN_STATUS_BAD_PARAM;
  if (k->beam() != v->beam() || (k->beam() != 1 && k->beam() != q->beam())) return CUDNN_STATUS_BAD_PARAM;
  if (k->lens != v->lens || o->lens != q->lens) return CUDNN_STATUS_BAD_PARAM;
  return CUDNN_STATUS_SUCCESS;
}

// Everything of one (batch, beam, query step), per head: the projected query,
// the window's projected keys and values, the probabilities (before and after
// dropout) and the head's output.
struct StepState {
  int first = 0, last = 0;
  std::vector<std::vector<real>> qb, p, pd, hv;          // per head
  std::vector<std::vector<std::vector<real>>> kb, vb;    // per head, per key step
};

void attn_step(const AttnPass& P, const int* lo, const int* hi, int t, int b, int j, const uint8_t* amask, double ap,
               StepState* st) {
  const Attn& a = P.a;
  const int H = a.heads, E = a.qe(), Ve = a.ve();
  P.window(lo, hi, t, b, j, &st->first, &st->last);
  const int n = std::max(0, st->last - st->first), jk = P.kv_beam(j);
  st->qb.assign(H, std::vector<real>(E));
  st->p.assign(H, std::vector<real>(n));
  st->pd.assign(H, std::vector<real>(n));
  st->hv.assign(H, std::vector<real>(Ve, 0.0));
  st->kb.assign(H, std::vector<std::vector<real>>(n, std::vector<real>(E)));
  st->vb.assign(H, std::vector<std::vector<real>>(n, std::vector<real>(Ve)));
  const real* qx = P.Q.data() + P.q.at(t, b, j);
  for (int h = 0; h < H; ++h) {
    P.qbar(h, qx, st->qb[h].data());
    double mx = -INFINITY;
    std::vector<double> s(n);
    for (int i = 0; i < n; ++i) {
      const int kt = st->first + i;
      P.kbar(h, P.K.data() + P.k.at(kt, b, jk), st->kb[h][i].data());
      P.vbar(h, P.V.data() + P.v.at(kt, b, jk), st->vb[h][i].data());
      double d = 0.0;
      for (int e = 0; e < E; ++e) d += (double)st->kb[h][i][e] * st->qb[h][e];
      s[i] = a.sm * d;
      mx = std::max(mx, s[i]);
    }
    double z = 0.0;
    for (int i = 0; i < n; ++i) z += std::exp(s[i] - mx);
    for (int i = 0; i < n; ++i) {
      st->p[h][i] = (real)(std::exp(s[i] - mx) / z);
      const double keep = amask ? (amask[(size_t)h * a.Tk + st->first + i] ? 1.0 / (1.0 - ap) : 0.0) : 1.0;
      st->pd[h][i] = (real)(st->p[h][i] * keep);
      for (int c = 0; c < Ve; ++c) st->hv[h][c] += st->pd[h][i] * st->vb[h][i][c];
    }
  }
}

// The output vector of one step from its heads' outputs (before dropout and residuals).
void attn_out(const AttnPass& P, const StepState& st, real* out) {
  const Attn& a = P.a;
  const int H = a.heads, Ve = a.ve(), O = a.oS();
  if (!a.oP) {
    for (int h = 0; h < H; ++h) std::copy(st.hv[h].begin(), st.hv[h].end(), out + (size_t)h * Ve);
    return;
  }
  for (int r = 0; r < O; ++r) {
    double s = P.off[7] == Attn::npos ? 0.0 : P.w[P.off[7] + r];
    for (int h = 0; h < H; ++h)
      for (int c = 0; c < Ve; ++c) s += P.w[P.off[3] + (size_t)h * a.oP * Ve + r + (size_t)c * a.oP] * st.hv[h][c];
    out[r] = (real)s;
  }
}

// Where the masks of (b, j, t) live in the reserve.
size_t mask_probs(const Attn& a, int b, int j, int t) { return (((size_t)b * a.maxBeam + j) * a.Tq + t) * a.heads * a.Tk; }
size_t mask_outs(const Attn& a, const AttnReserve& R, int b, int j, int t) {
  return R.probs + (((size_t)b * a.maxBeam + j) * a.Tq + t) * a.oS();
}
double drop_p(const vgpu_cudnn::DropoutDesc* d) { return d && known(d) ? d->p : 0.0; }
}  // namespace

VGPU_EXPORT cudnnStatus_t cudnnMultiHeadAttnForward(cudnnHandle_t h, const cudnnAttnDescriptor_t d, int curr,
                                                    const int lo[], const int hi[], const int*, const int*,
                                                    const cudnnSeqDataDescriptor_t qd, const void* queries,
                                                    const void* residuals, const cudnnSeqDataDescriptor_t kd,
                                                    const void* keys, const cudnnSeqDataDescriptor_t vd,
                                                    const void* values, const cudnnSeqDataDescriptor_t od, void* out,
                                                    size_t wbytes, const void* weights, size_t, void*, size_t rbytes,
                                                    void* reserve) {
  const Attn* a = attn(d);
  const SeqData *q = seq(qd), *k = seq(kd), *v = seq(vd), *o = seq(od);
  if (!known(h)) return CUDNN_STATUS_BAD_PARAM;
  if (cudnnStatus_t s = attn_check(a, q, k, v, o); s != CUDNN_STATUS_SUCCESS) return s;
  if (!lo || !hi || !queries || !keys || !values || !out || curr >= q->T()) return CUDNN_STATUS_BAD_PARAM;
  if (residuals && a->qS != a->oS()) return CUDNN_STATUS_BAD_PARAM;
  AttnPass P(*a, *q, *k, *v, *o);
  if (wbytes < P.wtotal * a->esize() || (P.wtotal && !weights)) return CUDNN_STATUS_BAD_PARAM;
  const AttnReserve R(*a);
  const bool training = reserve && rbytes > 0;
  if (training && rbytes < R.bytes(*a)) return CUDNN_STATUS_BAD_PARAM;
  if (training && curr >= 0) return CUDNN_STATUS_BAD_PARAM;  // one step at a time is inference only (measured)
  drain(h);
  P.w = fetch(weights, P.wtotal, a->type);
  P.Q = fetch(queries, q->count(), a->type);
  P.K = fetch(keys, k->count(), a->type);
  P.V = fetch(values, v->count(), a->type);
  const std::vector<real> res = residuals ? fetch(residuals, q->count(), a->type) : std::vector<real>();
  std::vector<real> y = fetch(out, o->count(), a->type);  // steps not computed keep what they held
  std::vector<uint8_t> masks(training ? R.bytes(*a) : 0, 1);
  if (training) {
    // The windows of every query step, for the weights' gradient.
    std::vector<int> win(R.win_ints(*a), 0);
    for (int t = 0; t < q->T(); ++t) win[t] = lo[t], win[a->Tq + t] = hi[t];
    std::memcpy(masks.data() + R.wins, win.data(), win.size() * sizeof(int));
  }
  const double ap = training ? drop_p(a->attn_drop) : 0.0, pp = training ? drop_p(a->post_drop) : 0.0;
  if (training && ap > 0.0) {
    std::vector<uint8_t> keep;
    if (!vgpu_cudnn::dropout_draw(a->attn_drop, R.probs, &keep)) return CUDNN_STATUS_EXECUTION_FAILED;
    std::copy(keep.begin(), keep.end(), masks.begin());
  }
  if (training && pp > 0.0) {
    std::vector<uint8_t> keep;
    if (!vgpu_cudnn::dropout_draw(a->post_drop, R.outs, &keep)) return CUDNN_STATUS_EXECUTION_FAILED;
    std::copy(keep.begin(), keep.end(), masks.begin() + R.probs);
  }
  const int O = a->oS();
  std::vector<real> ov(O);
  StepState st;
  for (int b = 0; b < q->B(); ++b)
    for (int j = 0; j < q->beam(); ++j)
      for (int t = 0; t < q->T(); ++t) {
        if (curr >= 0 && t != curr) continue;
        real* dst = y.data() + o->at(t, b, j);
        if (t >= q->len(b, j)) {
          // Past the sequence: the output bias and the residual (measured).
          for (int r = 0; r < O; ++r)
            dst[r] = (P.off[7] == Attn::npos ? 0.0 : P.w[P.off[7] + r]) + (res.empty() ? 0.0 : res[q->at(t, b, j) + r]);
          continue;
        }
        attn_step(P, lo, hi, t, b, j, ap > 0.0 ? masks.data() + mask_probs(*a, b, j, t) : nullptr, ap, &st);
        attn_out(P, st, ov.data());
        const uint8_t* pm = pp > 0.0 ? masks.data() + mask_outs(*a, R, b, j, t) : nullptr;
        for (int r = 0; r < O; ++r) {
          double e = ov[r];
          if (pm) e = pm[r] ? e / (1.0 - pp) : 0.0;
          if (!res.empty()) e += res[q->at(t, b, j) + r];
          dst[r] = (real)e;
        }
      }
  store(out, y, a->type);
  if (training && cudaMemcpy(reserve, masks.data(), masks.size(), cudaMemcpyHostToDevice) != cudaSuccess)
    return CUDNN_STATUS_EXECUTION_FAILED;
  return CUDNN_STATUS_SUCCESS;
}

namespace {
// The gradients of one call: data (dQ, dK, dV) and/or weights.
cudnnStatus_t attn_backward(cudnnHandle_t h, const Attn* a, const int* lo, const int* hi, const SeqData* q,
                            const SeqData* k, const SeqData* v, const SeqData* o, const void* dout, const void* queries,
                            const void* keys, const void* values, size_t wbytes, const void* weights, size_t rbytes,
                            const void* reserve, void* dq, void* dk, void* dv, void* dw, bool add) {
  if (cudnnStatus_t s = attn_check(a, q, k, v, o); s != CUDNN_STATUS_SUCCESS) return s;
  if (!dout || !queries || !keys || !values) return CUDNN_STATUS_BAD_PARAM;
  AttnPass P(*a, *q, *k, *v, *o);
  const AttnReserve R(*a);
  if (wbytes < P.wtotal * a->esize() || (P.wtotal && !weights) || !reserve || rbytes < R.bytes(*a))
    return CUDNN_STATUS_BAD_PARAM;
  drain(h);
  P.w = fetch(weights, P.wtotal, a->type);
  P.Q = fetch(queries, q->count(), a->type);
  P.K = fetch(keys, k->count(), a->type);
  P.V = fetch(values, v->count(), a->type);
  const std::vector<real> g = fetch(dout, o->count(), a->type);
  std::vector<uint8_t> masks(R.bytes(*a));
  if (cudaMemcpy(masks.data(), reserve, masks.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
    return CUDNN_STATUS_EXECUTION_FAILED;
  std::vector<int> win;
  if (!lo || !hi) {  // the weights' gradient: the forward pass's windows
    win.resize(R.win_ints(*a));
    std::memcpy(win.data(), masks.data() + R.wins, win.size() * sizeof(int));
    lo = win.data(), hi = win.data() + a->Tq;
  }
  const double ap = drop_p(a->attn_drop), pp = drop_p(a->post_drop);
  std::vector<real> gq(q->count(), 0.0), gk(k->count(), 0.0), gv(v->count(), 0.0);
  std::vector<real> gw = dw && add ? fetch(dw, P.wtotal, a->type) : std::vector<real>(P.wtotal, 0.0);
  const int H = a->heads, E = a->qe(), Ve = a->ve(), O = a->oS();
  StepState st;
  std::vector<real> go(O);
  for (int b = 0; b < q->B(); ++b)
    for (int j = 0; j < q->beam(); ++j)
      for (int t = 0; t < q->len(b, j); ++t) {
        attn_step(P, lo, hi, t, b, j, ap > 0.0 ? masks.data() + mask_probs(*a, b, j, t) : nullptr, ap, &st);
        const uint8_t* pm = pp > 0.0 ? masks.data() + mask_outs(*a, R, b, j, t) : nullptr;
        for (int r = 0; r < O; ++r) {
          double e = g[o->at(t, b, j) + r];
          if (pm) e = pm[r] ? e / (1.0 - pp) : 0.0;
          go[r] = (real)e;
        }
        if (P.off[7] != Attn::npos)
          for (int r = 0; r < O; ++r) gw[P.off[7] + r] += go[r];
        const int jk = P.kv_beam(j), n = st.last > st.first ? st.last - st.first : 0;
        const real* qx = P.Q.data() + q->at(t, b, j);
        for (int hh = 0; hh < H; ++hh) {
          // Into the head's output.
          std::vector<real> dh(Ve, 0.0);
          if (a->oP) {
            for (int r = 0; r < a->oP; ++r)
              for (int c = 0; c < Ve; ++c) {
                const size_t wi = P.off[3] + (size_t)hh * a->oP * Ve + r + (size_t)c * a->oP;
                dh[c] += P.w[wi] * go[r];
                gw[wi] += go[r] * st.hv[hh][c];
              }
          } else {
            std::copy(go.begin() + (size_t)hh * Ve, go.begin() + (size_t)(hh + 1) * Ve, dh.begin());
          }
          // Through the probabilities and the softmax.
          std::vector<double> dp(n);
          double dot = 0.0;
          for (int i = 0; i < n; ++i) {
            double s = 0.0;
            for (int c = 0; c < Ve; ++c) s += (double)dh[c] * st.vb[hh][i][c];
            const double keep = st.p[hh][i] != 0.0 ? st.pd[hh][i] / st.p[hh][i] : (ap > 0.0 ? 0.0 : 1.0);
            dp[i] = s * keep;
            dot += dp[i] * st.p[hh][i];
          }
          std::vector<real> dqb(E, 0.0);
          for (int i = 0; i < n; ++i) {
            const int kt = st.first + i;
            const double ds = st.p[hh][i] * (dp[i] - dot) * a->sm;
            std::vector<real> dkb(E), dvb(Ve);
            for (int e = 0; e < E; ++e) dqb[e] += ds * st.kb[hh][i][e], dkb[e] = ds * st.qb[hh][e];
            for (int c = 0; c < Ve; ++c) dvb[c] = st.pd[hh][i] * dh[c];
            // dK and the K projection.
            const real* kx = P.K.data() + k->at(kt, b, jk);
            real* gkx = gk.data() + k->at(kt, b, jk);
            if (a->kP) {
              for (int e = 0; e < E; ++e) {
                for (int c = 0; c < a->kS; ++c) {
                  const size_t wi = P.off[1] + (size_t)hh * E + e + (size_t)c * H * E;
                  gkx[c] += P.w[wi] * dkb[e];
                  gw[wi] += dkb[e] * kx[c];
                }
                if (P.off[5] != Attn::npos) gw[P.off[5] + (size_t)hh * E + e] += dkb[e];
              }
            } else {
              for (int c = 0; c < a->kS; ++c) gkx[c] += dkb[c];
            }
            const real* vx = P.V.data() + v->at(kt, b, jk);
            real* gvx = gv.data() + v->at(kt, b, jk);
            if (a->vP) {
              for (int e = 0; e < Ve; ++e) {
                for (int c = 0; c < a->vS; ++c) {
                  const size_t wi = P.off[2] + (size_t)hh * Ve + e + (size_t)c * H * Ve;
                  gvx[c] += P.w[wi] * dvb[e];
                  gw[wi] += dvb[e] * vx[c];
                }
                if (P.off[6] != Attn::npos) gw[P.off[6] + (size_t)hh * Ve + e] += dvb[e];
              }
            } else {
              for (int c = 0; c < a->vS; ++c) gvx[c] += dvb[c];
            }
          }
          real* gqx = gq.data() + q->at(t, b, j);
          if (a->qP) {
            for (int e = 0; e < E; ++e) {
              for (int c = 0; c < a->qS; ++c) {
                const size_t wi = P.off[0] + (size_t)hh * E + e + (size_t)c * H * E;
                gqx[c] += P.w[wi] * dqb[e];
                gw[wi] += dqb[e] * qx[c];
              }
              if (P.off[4] != Attn::npos) gw[P.off[4] + (size_t)hh * E + e] += dqb[e];
            }
          } else {
            for (int c = 0; c < a->qS; ++c) gqx[c] += dqb[c];
          }
        }
      }
  if (dq) store(dq, gq, a->type);
  if (dk) store(dk, gk, a->type);
  if (dv) store(dv, gv, a->type);
  if (dw) store(dw, gw, a->type);
  return CUDNN_STATUS_SUCCESS;
}
}  // namespace

VGPU_EXPORT cudnnStatus_t cudnnMultiHeadAttnBackwardData(
    cudnnHandle_t h, const cudnnAttnDescriptor_t d, const int lo[], const int hi[], const int*, const int*,
    const cudnnSeqDataDescriptor_t dod, const void* dout, const cudnnSeqDataDescriptor_t dqd, void* dq,
    const void* queries, const cudnnSeqDataDescriptor_t dkd, void* dk, const void* keys,
    const cudnnSeqDataDescriptor_t dvd, void* dv, const void* values, size_t wbytes, const void* weights, size_t,
    void*, size_t rbytes, void* reserve) {
  if (!known(h) || !lo || !hi || !dq || !dk || !dv) return CUDNN_STATUS_BAD_PARAM;
  return attn_backward(h, attn(d), lo, hi, seq(dqd), seq(dkd), seq(dvd), seq(dod), dout, queries, keys, values, wbytes,
                       weights, rbytes, reserve, dq, dk, dv, nullptr, false);
}

// The windows are the forward pass's, which it kept in the reserve.
VGPU_EXPORT cudnnStatus_t cudnnMultiHeadAttnBackwardWeights(
    cudnnHandle_t h, const cudnnAttnDescriptor_t d, cudnnWgradMode_t add, const cudnnSeqDataDescriptor_t qd,
    const void* queries, const cudnnSeqDataDescriptor_t kd, const void* keys, const cudnnSeqDataDescriptor_t vd,
    const void* values, const cudnnSeqDataDescriptor_t dod, const void* dout, size_t wbytes, const void* weights,
    void* dw, size_t, void*, size_t rbytes, void* reserve) {
  if (!known(h) || !dw) return CUDNN_STATUS_BAD_PARAM;
  return attn_backward(h, attn(d), nullptr, nullptr, seq(qd), seq(kd), seq(vd), seq(dod), dout, queries, keys,
                       values, wbytes, weights, rbytes, reserve, nullptr, nullptr, nullptr, dw,
                       add == CUDNN_WGRAD_MODE_ADD);
}
