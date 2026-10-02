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
// varying lengths; float, half, bfloat16 or double data, computed in double.
// Refused by name: projections (projSize != hiddenSize) and skip-input mode.
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
  cudnnRNNMode_t mode = CUDNN_LSTM;
  cudnnDataType_t type = CUDNN_DATA_FLOAT;  // of the weights, states and data
  size_t esize() const { return vgpu_cudnn::type_bytes(type); }
  cudnnRNNBiasMode_t bias = CUDNN_RNN_DOUBLE_BIAS;
  int dirs = 1, in = 0, hid = 0, layers = 1;
  // The dropout descriptor, read when a training pass runs: its probability
  // and generator are the caller's to change in between.
  vgpu_cudnn::DropoutDesc* drop = nullptr;
  float set_p = 0.0f;  // its probability when the RNN was set, for when it is gone
  bool drop_alive() const { return drop && known(drop); }
  real dropout() const { return drop_alive() ? drop->p : set_p; }
  int gates() const { return mode == CUDNN_LSTM ? 4 : mode == CUDNN_GRU ? 3 : 1; }
  bool input_bias() const { return bias == CUDNN_RNN_DOUBLE_BIAS || bias == CUDNN_RNN_SINGLE_INP_BIAS; }
  bool rec_bias() const { return bias == CUDNN_RNN_DOUBLE_BIAS || bias == CUDNN_RNN_SINGLE_REC_BIAS; }
  int layer_in(int layer) const { return layer == 0 ? in : hid * dirs; }
  // Where pseudo-layer pl's pieces start in the weight space, in elements.
  size_t block(int pl) const {
    size_t off = 0;
    for (int p = 0; p < pl; ++p) off += block_size(p);
    return off;
  }
  size_t block_size(int pl) const {
    const size_t G = gates(), H = hid, I = layer_in(pl / dirs);
    return G * H * I + G * H * H + (input_bias() ? G * H : 0) + (rec_bias() ? G * H : 0);
  }
  size_t weights() const { return block(layers * dirs); }
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

// The reserve space's layout, in floats.
struct Reserve {
  size_t T, B, G, H;
  std::vector<size_t> input;   // per layer: [T][B][in_l]
  std::vector<size_t> mask;    // per layer but the last: dropout's scale on its output, [T][B][H*D]
  size_t per_pl_start = 0;     // then per pseudo-layer: gates, c, h, rn, dgi, dgr
  size_t pl_stride = 0;
  size_t total = 0;
  Reserve(const Rnn& r, int T_, int B_) : T(T_), B(B_), G(r.gates()), H(r.hid) {
    size_t off = 0;
    for (int l = 0; l < r.layers; ++l) {
      input.push_back(off);
      off += T * B * r.layer_in(l);
    }
    for (int l = 0; l + 1 < r.layers; ++l) {
      mask.push_back(off);
      off += T * B * H * r.dirs;
    }
    per_pl_start = off;
    pl_stride = T * B * (G * H + H + H + H + G * H + G * H);
    total = off + pl_stride * r.layers * r.dirs;
  }
  size_t gates(int pl) const { return per_pl_start + pl * pl_stride; }
  size_t c(int pl) const { return gates(pl) + T * B * G * H; }
  size_t h(int pl) const { return c(pl) + T * B * H; }
  size_t rn(int pl) const { return h(pl) + T * B * H; }
  size_t dgi(int pl) const { return rn(pl) + T * B * H; }
  size_t dgr(int pl) const { return dgi(pl) + T * B * G * H; }
};

// Checks the descriptors a call was given agree with each other.
cudnnStatus_t check(const Rnn& r, const Data& x, const Data& y) {
  if (x.type != r.type || y.type != r.type) return CUDNN_STATUS_BAD_PARAM;
  if (x.V != r.in || y.V != r.hid * r.dirs || x.T != y.T || x.B != y.B) return CUDNN_STATUS_BAD_PARAM;
  if ((int)x.len.size() != x.B) return CUDNN_STATUS_BAD_PARAM;
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
  const int T = xd.T, B = xd.B, H = r.hid, G = r.gates(), D = r.dirs;
  std::vector<real> in = x;  // this layer's input, dense [T][B][I]
  for (int l = 0; l < r.layers; ++l) {
    const int I = r.layer_in(l);
    if (res) std::copy(in.begin(), in.end(), res->begin() + rv.input[l]);
    std::vector<real> out((size_t)T * B * H * D, 0.0f);
    for (int dir = 0; dir < D; ++dir) {
      const int pl = l * D + dir;
      const size_t base = r.block(pl);
      const real* W = w.data() + base;
      const real* R = W + (size_t)G * H * I;
      const real* bW = r.input_bias() ? R + (size_t)G * H * H : nullptr;
      const real* bR = r.rec_bias() ? R + (size_t)G * H * H + (r.input_bias() ? (size_t)G * H : 0) : nullptr;
      for (int b = 0; b < B; ++b) {
        std::vector<real> h(hx.begin() + ((size_t)pl * B + b) * H, hx.begin() + ((size_t)pl * B + b + 1) * H);
        std::vector<real> c = r.mode == CUDNN_LSTM
                                   ? std::vector<real>(cx.begin() + ((size_t)pl * B + b) * H,
                                                        cx.begin() + ((size_t)pl * B + b + 1) * H)
                                   : std::vector<real>(H, 0.0f);
        std::vector<real> zi((size_t)G * H), zr((size_t)G * H);
        for (int s = 0; s < xd.len[b]; ++s) {
          const int t = step_t(dir, s, xd.len[b]);
          const real* xt = in.data() + ((size_t)t * B + b) * I;
          for (int g = 0; g < G; ++g)
            for (int j = 0; j < H; ++j) {
              double a = bW ? bW[g * H + j] : 0.0, q = bR ? bR[g * H + j] : 0.0;
              const real* wr = W + ((size_t)g * H + j) * I;
              for (int k = 0; k < I; ++k) a += (double)wr[k] * xt[k];
              const real* rr = R + ((size_t)g * H + j) * H;
              for (int k = 0; k < H; ++k) q += (double)rr[k] * h[k];
              zi[g * H + j] = (real)a, zr[g * H + j] = (real)q;
            }
          const size_t tb = (size_t)t * B + b;
          real* gates = res ? res->data() + rv.gates(pl) + tb * G * H : nullptr;
          std::vector<real> nh(H);
          for (int j = 0; j < H; ++j) {
            if (r.mode == CUDNN_LSTM) {
              const real i = sigmoid(zi[j] + zr[j]), f = sigmoid(zi[H + j] + zr[H + j]);
              const real gg = std::tanh(zi[2 * H + j] + zr[2 * H + j]), o = sigmoid(zi[3 * H + j] + zr[3 * H + j]);
              c[j] = f * c[j] + i * gg;
              nh[j] = o * std::tanh(c[j]);
              if (gates) gates[j] = i, gates[H + j] = f, gates[2 * H + j] = gg, gates[3 * H + j] = o;
            } else if (r.mode == CUDNN_GRU) {
              const real rg = sigmoid(zi[j] + zr[j]), z = sigmoid(zi[H + j] + zr[H + j]);
              const real rn = zr[2 * H + j];
              const real n = std::tanh(zi[2 * H + j] + rg * rn);
              nh[j] = (1 - z) * n + z * h[j];
              if (gates) gates[j] = rg, gates[H + j] = z, gates[2 * H + j] = n;
              if (res) (*res)[rv.rn(pl) + tb * H + j] = rn;
            } else {
              const real a = zi[j] + zr[j];
              nh[j] = r.mode == CUDNN_RNN_RELU ? std::max(0.0, a) : std::tanh(a);
              if (gates) gates[j] = nh[j];
            }
          }
          h = nh;
          if (res) {
            std::copy(h.begin(), h.end(), res->begin() + rv.h(pl) + tb * H);
            std::copy(c.begin(), c.end(), res->begin() + rv.c(pl) + tb * H);
          }
          std::copy(h.begin(), h.end(), out.begin() + tb * H * D + (size_t)dir * H);
        }
        std::copy(h.begin(), h.end(), hy->begin() + ((size_t)pl * B + b) * H);
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

cudnnStatus_t resolve(cudnnRNNDescriptor_t rd, cudnnRNNDataDescriptor_t xd, cudnnRNNDataDescriptor_t yd, Rnns* out) {
  if (!known(rd) || !known(xd) || !known(yd)) return CUDNN_STATUS_BAD_PARAM;
  out->r = reinterpret_cast<const Rnn*>(rd);
  out->x = reinterpret_cast<const Data*>(xd);
  out->y = reinterpret_cast<const Data*>(yd);
  return check(*out->r, *out->x, *out->y);
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
VGPU_EXPORT cudnnStatus_t cudnnSetRNNDescriptor_v8(cudnnRNNDescriptor_t d, cudnnRNNAlgo_t, cudnnRNNMode_t mode,
                                                   cudnnRNNBiasMode_t bias, cudnnDirectionMode_t dir,
                                                   cudnnRNNInputMode_t input, cudnnDataType_t type,
                                                   cudnnDataType_t math, cudnnMathType_t, int32_t in, int32_t hid,
                                                   int32_t proj, int32_t layers, cudnnDropoutDescriptor_t drop,
                                                   uint32_t) {
  if (!known(d) || in <= 0 || hid <= 0 || layers <= 0) return CUDNN_STATUS_BAD_PARAM;
  // cuDNN's pairs: float, half or bfloat16 data with float math (half also
  // with half math), double with double. All compute in double here.
  const bool pair = (math == CUDNN_DATA_FLOAT && (type == CUDNN_DATA_FLOAT || type == CUDNN_DATA_HALF ||
                                                 type == CUDNN_DATA_BFLOAT16)) ||
                    (math == CUDNN_DATA_HALF && type == CUDNN_DATA_HALF) ||
                    (math == CUDNN_DATA_DOUBLE && type == CUDNN_DATA_DOUBLE);
  if (!pair) return refuse("cudnnSetRNNDescriptor_v8", "the data and math types are not one of cuDNN's pairs");
  if (proj != hid) return refuse("cudnnSetRNNDescriptor_v8", "LSTM projections (projSize != hiddenSize) are not supported");
  if (input != CUDNN_LINEAR_INPUT) return refuse("cudnnSetRNNDescriptor_v8", "only CUDNN_LINEAR_INPUT is supported");
  auto* r = reinterpret_cast<Rnn*>(d);
  r->type = type;
  r->mode = mode;
  r->bias = bias;
  r->dirs = dir == CUDNN_BIDIRECTIONAL ? 2 : 1;
  r->in = in, r->hid = hid, r->layers = layers;
  r->drop = known(drop) ? reinterpret_cast<vgpu_cudnn::DropoutDesc*>(drop) : nullptr;
  r->set_p = r->drop ? r->drop->p : 0.0f;
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

/* ---- sizes and where the weights are ---- */

VGPU_EXPORT cudnnStatus_t cudnnGetRNNWeightSpaceSize(cudnnHandle_t, cudnnRNNDescriptor_t d, size_t* size) {
  if (!known(d) || !size) return CUDNN_STATUS_BAD_PARAM;
  *size = reinterpret_cast<const Rnn*>(d)->weights() * reinterpret_cast<const Rnn*>(d)->esize();
  return CUDNN_STATUS_SUCCESS;
}

// Where pseudo-layer pl's gate linLayerID lives: IDs below the gate count are
// the input matrices, the rest the recurrent ones, each with its bias. The
// descriptors say [1, hidden, columns] and [1, hidden, 1].
VGPU_EXPORT cudnnStatus_t cudnnGetRNNWeightParams(cudnnHandle_t, cudnnRNNDescriptor_t d, int32_t pl, size_t size,
                                                  const void* space, int32_t lin, cudnnTensorDescriptor_t mDesc,
                                                  void** mAddr, cudnnTensorDescriptor_t bDesc, void** bAddr) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  const Rnn& r = *reinterpret_cast<const Rnn*>(d);
  const int G = r.gates(), H = r.hid;
  if (pl < 0 || pl >= r.layers * r.dirs || lin < 0 || lin >= 2 * G || size < r.weights() * r.esize())
    return CUDNN_STATUS_BAD_PARAM;
  const bool rec = lin >= G;
  const int g = rec ? lin - G : lin;
  const int I = r.layer_in(pl / r.dirs);
  const int cols = rec ? H : I;
  const size_t base = r.block(pl);
  const size_t mat = base + (rec ? (size_t)G * H * I : 0) + (size_t)g * H * cols;
  const size_t bias0 = base + (size_t)G * H * I + (size_t)G * H * H;
  const bool has_bias = rec ? r.rec_bias() : r.input_bias();
  const size_t bias = bias0 + (rec && r.input_bias() ? (size_t)G * H : 0) + (size_t)g * H;
  auto* at = static_cast<const char*>(space);
  if (mAddr) *mAddr = space ? const_cast<char*>(at + mat * r.esize()) : nullptr;
  if (bAddr) *bAddr = space && has_bias ? const_cast<char*>(at + bias * r.esize()) : nullptr;
  if (mDesc) {
    const int dims[3] = {1, H, cols}, strides[3] = {H * cols, cols, 1};
    if (cudnnSetTensorNdDescriptor(mDesc, r.type, 3, dims, strides) != CUDNN_STATUS_SUCCESS)
      return CUDNN_STATUS_BAD_PARAM;
  }
  if (bDesc && has_bias) {
    const int dims[3] = {1, H, 1}, strides[3] = {H, 1, 1};
    if (cudnnSetTensorNdDescriptor(bDesc, r.type, 3, dims, strides) != CUDNN_STATUS_SUCCESS)
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
                                          cudnnRNNDataDescriptor_t yd, void* y, cudnnTensorDescriptor_t,
                                          const void* hx, void* hy, cudnnTensorDescriptor_t, const void* cx,
                                          void* cy, size_t wsize, const void* w, size_t, void*, size_t rsize,
                                          void* reserve) {
  Rnns d;
  if (const cudnnStatus_t s = resolve(rd, xd, yd, &d); s != CUDNN_STATUS_SUCCESS) return s;
  const Rnn& r = *d.r;
  const bool training = mode == CUDNN_FWD_MODE_TRAINING;
  if (training && r.layers > 1 && r.dropout() > 0.0 && !r.drop_alive())
    return refuse("cudnnRNNForward", "the dropout descriptor this RNN was set with has been destroyed");
  if (wsize < r.weights() * r.esize()) return CUDNN_STATUS_BAD_PARAM;
  const Reserve rv(r, d.x->T, d.x->B);
  if (training && (!reserve || rsize < rv.total * sizeof(real))) return CUDNN_STATUS_BAD_PARAM;
  drain(h);
  const size_t states = (size_t)r.layers * r.dirs * d.x->B * r.hid;
  const auto hw = fetch(w, r.weights(), r.type);
  const auto hx_ = hx ? fetch(hx, states, r.type) : std::vector<real>(states, 0.0f);
  const auto cx_ = r.mode == CUDNN_LSTM && cx ? fetch(cx, states, r.type) : std::vector<real>(states, 0.0f);
  std::vector<real> dense_y, hy_(states, 0.0f), cy_(states, 0.0f);
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
// recurrent side (they differ only in GRU's new gate).
VGPU_EXPORT cudnnStatus_t cudnnRNNBackwardData_v8(cudnnHandle_t h, cudnnRNNDescriptor_t rd, const int32_t*,
                                                  cudnnRNNDataDescriptor_t yd, const void*, const void* dy,
                                                  cudnnRNNDataDescriptor_t xd, void* dx, cudnnTensorDescriptor_t,
                                                  const void* hx, const void* dhy, void* dhx,
                                                  cudnnTensorDescriptor_t, const void* cx, const void* dcy,
                                                  void* dcx, size_t wsize, const void* w, size_t, void*,
                                                  size_t rsize, void* reserve) {
  Rnns d;
  if (const cudnnStatus_t s = resolve(rd, xd, yd, &d); s != CUDNN_STATUS_SUCCESS) return s;
  const Rnn& r = *d.r;
  const Reserve rv(r, d.x->T, d.x->B);
  if (wsize < r.weights() * r.esize() || !reserve || rsize < rv.total * sizeof(real)) return CUDNN_STATUS_BAD_PARAM;
  drain(h);
  const int T = d.x->T, B = d.x->B, H = r.hid, G = r.gates(), D = r.dirs;
  const size_t states = (size_t)r.layers * D * B * H;
  const auto hw = fetch(w, r.weights(), r.type);
  auto res = fetch_res(reserve, rv.total);
  const auto hx_ = hx ? fetch(hx, states, r.type) : std::vector<real>(states, 0.0f);
  const auto cx_ = r.mode == CUDNN_LSTM && cx ? fetch(cx, states, r.type) : std::vector<real>(states, 0.0f);
  const auto dhy_ = dhy ? fetch(dhy, states, r.type) : std::vector<real>(states, 0.0f);
  const auto dcy_ = r.mode == CUDNN_LSTM && dcy ? fetch(dcy, states, r.type) : std::vector<real>(states, 0.0f);
  std::vector<real> dhx_(states, 0.0f), dcx_(states, 0.0f);
  std::vector<real> dout = unpack(*d.y, dy);  // gradient of this layer's output, dense [T][B][H*D]
  for (int l = r.layers - 1; l >= 0; --l) {
    const int I = r.layer_in(l);
    std::vector<real> din((size_t)T * B * I, 0.0f);
    for (int dir = 0; dir < D; ++dir) {
      const int pl = l * D + dir;
      const real* W = hw.data() + r.block(pl);
      const real* R = W + (size_t)G * H * I;
      for (int b = 0; b < B; ++b) {
        const int len = d.x->len[b];
        std::vector<real> dh(dhy_.begin() + ((size_t)pl * B + b) * H, dhy_.begin() + ((size_t)pl * B + b + 1) * H);
        std::vector<real> dc(dcy_.begin() + ((size_t)pl * B + b) * H, dcy_.begin() + ((size_t)pl * B + b + 1) * H);
        for (int s = len - 1; s >= 0; --s) {
          const int t = step_t(dir, s, len);
          const size_t tb = (size_t)t * B + b;
          // The state before this step: the previous step's, or hx/cx.
          const bool first = s == 0;
          const int tp = first ? -1 : step_t(dir, s - 1, len);
          const real* hprev = first ? hx_.data() + ((size_t)pl * B + b) * H : res.data() + rv.h(pl) + ((size_t)tp * B + b) * H;
          const real* cprev = first ? cx_.data() + ((size_t)pl * B + b) * H : res.data() + rv.c(pl) + ((size_t)tp * B + b) * H;
          const real* gt = res.data() + rv.gates(pl) + tb * G * H;
          const real* ct = res.data() + rv.c(pl) + tb * H;
          const real* ht = res.data() + rv.h(pl) + tb * H;
          real* dgi = res.data() + rv.dgi(pl) + tb * G * H;
          real* dgr = res.data() + rv.dgr(pl) + tb * G * H;
          for (int j = 0; j < H; ++j) dh[j] += dout[tb * H * D + (size_t)dir * H + j];
          std::vector<real> dhp(H, 0.0f), dcp(H, 0.0f);
          for (int j = 0; j < H; ++j) {
            if (r.mode == CUDNN_LSTM) {
              const real i = gt[j], f = gt[H + j], g = gt[2 * H + j], o = gt[3 * H + j];
              const real tc = std::tanh(ct[j]);
              const real dct = dc[j] + dh[j] * o * (1 - tc * tc);
              dgi[j] = dct * g * i * (1 - i);
              dgi[H + j] = dct * cprev[j] * f * (1 - f);
              dgi[2 * H + j] = dct * i * (1 - g * g);
              dgi[3 * H + j] = dh[j] * tc * o * (1 - o);
              dcp[j] = dct * f;
            } else if (r.mode == CUDNN_GRU) {
              const real rg = gt[j], z = gt[H + j], n = gt[2 * H + j];
              const real rn = res[rv.rn(pl) + tb * H + j];
              const real dn = dh[j] * (1 - z), dz = dh[j] * (hprev[j] - n);
              const real dan = dn * (1 - n * n);
              dgi[j] = dan * rn * rg * (1 - rg);
              dgi[H + j] = dz * z * (1 - z);
              dgi[2 * H + j] = dan;
              dhp[j] += dh[j] * z;
            } else {
              const real a = ht[j];
              dgi[j] = dh[j] * (r.mode == CUDNN_RNN_RELU ? (a > 0 ? 1.0f : 0.0f) : 1 - a * a);
            }
          }
          for (int k = 0; k < G * H; ++k) dgr[k] = dgi[k];
          if (r.mode == CUDNN_GRU)
            for (int j = 0; j < H; ++j) dgr[2 * H + j] = dgi[2 * H + j] * gt[j];  // through the reset gate
          // Into the previous state, and into this layer's input.
          for (int g = 0; g < G; ++g)
            for (int j = 0; j < H; ++j) {
              const real* rr = R + ((size_t)g * H + j) * H;
              for (int k = 0; k < H; ++k) dhp[k] += rr[k] * dgr[g * H + j];
              const real* wr = W + ((size_t)g * H + j) * I;
              real* di = din.data() + tb * I;
              for (int k = 0; k < I; ++k) di[k] += wr[k] * dgi[g * H + j];
            }
          dh = dhp;
          dc = dcp;
        }
        std::copy(dh.begin(), dh.end(), dhx_.begin() + ((size_t)pl * B + b) * H);
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
                                                     cudnnTensorDescriptor_t, const void* hx,
                                                     cudnnRNNDataDescriptor_t yd, const void*, size_t wsize,
                                                     void* dw, size_t, void*, size_t rsize, void* reserve) {
  Rnns d;
  if (const cudnnStatus_t s = resolve(rd, xd, yd, &d); s != CUDNN_STATUS_SUCCESS) return s;
  const Rnn& r = *d.r;
  const Reserve rv(r, d.x->T, d.x->B);
  if (wsize < r.weights() * r.esize() || !reserve || rsize < rv.total * sizeof(real)) return CUDNN_STATUS_BAD_PARAM;
  drain(h);
  const int T = d.x->T, B = d.x->B, H = r.hid, G = r.gates(), D = r.dirs;
  (void)T;
  const size_t states = (size_t)r.layers * D * B * H;
  const auto res = fetch_res(reserve, rv.total);
  const auto hx_ = hx ? fetch(hx, states, r.type) : std::vector<real>(states, 0.0f);
  std::vector<real> g = add == CUDNN_WGRAD_MODE_ADD ? fetch(dw, r.weights(), r.type) : std::vector<real>(r.weights(), 0.0f);
  for (int l = 0; l < r.layers; ++l) {
    const int I = r.layer_in(l);
    const real* in = res.data() + rv.input[l];
    for (int dir = 0; dir < D; ++dir) {
      const int pl = l * D + dir;
      real* dW = g.data() + r.block(pl);
      real* dR = dW + (size_t)G * H * I;
      real* dbW = r.input_bias() ? dR + (size_t)G * H * H : nullptr;
      real* dbR = r.rec_bias() ? dR + (size_t)G * H * H + (r.input_bias() ? (size_t)G * H : 0) : nullptr;
      for (int b = 0; b < B; ++b) {
        const int len = d.x->len[b];
        for (int s = 0; s < len; ++s) {
          const int t = step_t(dir, s, len);
          const size_t tb = (size_t)t * B + b;
          const real* hprev = s == 0 ? hx_.data() + ((size_t)pl * B + b) * H
                                      : res.data() + rv.h(pl) + ((size_t)step_t(dir, s - 1, len) * B + b) * H;
          const real* dgi = res.data() + rv.dgi(pl) + tb * G * H;
          const real* dgr = res.data() + rv.dgr(pl) + tb * G * H;
          const real* xt = in + tb * I;
          for (int q = 0; q < G * H; ++q) {
            for (int k = 0; k < I; ++k) dW[(size_t)q * I + k] += dgi[q] * xt[k];
            for (int k = 0; k < H; ++k) dR[(size_t)q * H + k] += dgr[q] * hprev[k];
            if (dbW) dbW[q] += dgi[q];
            if (dbR) dbR[q] += dgr[q];
          }
        }
      }
    }
  }
  store(dw, g, r.type);
  return CUDNN_STATUS_SUCCESS;
}
