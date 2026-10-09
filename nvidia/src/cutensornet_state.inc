// The state API of libvgpucutensornet, included by cutensornet_api.cpp after
// the network and decomposition code it is built on.
//
// A tensor network state is a record of what was applied to it: the vacuum
// (or a matrix product state the caller initialised it with), then the
// tensor operators, channels and network operators in order. Nothing is
// contracted until a property is asked for. An accessor, marginal, sampler or
// expectation value then builds the tensor network of that property from the
// record (a pure state's ket circuit and its conjugate, a mixed state's ket
// and bra legs), finds a contraction path for it at prepare time, and
// contracts it at compute time with the network engine of this file. A
// matrix product state is different: the state is computed at StateCompute,
// by applying the operators to the tensors one at a time and truncating after
// each two-site operator with an SVD, and the properties are then contracted
// from those tensors.
//
// Conventions here are measured on an RTX 3060 against NVIDIA's cuTensorNet
// 2.14 (cuquantum-cu13 26.9), and each is noted where it is used:
//
//   - a tensor operator on state modes (q_0 .. q_{k-1}) is a matrix with q_0
//     the most significant index, in row-major C layout [row][column]: the
//     column (first half of the operator's modes) is contracted with the
//     state, the row (second half) is the new state's. Its default storage
//     order is (q_{k-1}, .., q_0 columns, q_{k-1}, .., q_0 rows), column-major;
//   - a tensor of a network operator (a sum of products, or a matrix product
//     operator) has its modes in forward order (q_0 .. q_{k-1} columns, then
//     rows); the coefficient of a component is applied only to expectation
//     values, not when the operator is applied to a state;
//   - the state's amplitudes are column-major with mode 0 fastest; a mixed
//     state is a tensor with the ket modes (all of them) fastest, then the
//     bra modes; the density matrix of a pure state is psi psi^*;
//   - unitary tensor operators are cancelled against their adjoint in the
//     network of a property wherever the operator's modes lie outside the
//     "light cone" of the property (the modes it touches), which is also why
//     a state of unitary operators reports a squared norm of exactly 1;
//   - an MPS is evolved by gates applied to adjacent sites with an SVD that
//     splits sqrt(S) onto both sides, after swapping the first listed mode of
//     a non-adjacent gate next to the last.
namespace {

constexpr uint32_t kMagicState = 0x54535431, kMagicNetOp = 0x544e4f50, kMagicAccessor = 0x54414331,
                   kMagicExpectation = 0x54455831, kMagicMarginal = 0x544d4731, kMagicSampler = 0x54534d31,
                   kMagicProjection = 0x54505231;

// ---- small helpers ----

// Device memory the library owns for a state or a property.
struct DevBuf {
  void* p = nullptr;
  size_t bytes = 0;
  DevBuf() = default;
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  ~DevBuf() { release(); }
  void release() {
    if (p) {
      cudaFree(p);
      cudaGetLastError();
    }
    p = nullptr;
    bytes = 0;
  }
  bool alloc(size_t n) {
    release();
    n = std::max<size_t>(n, 16);
    if (cudaMalloc(&p, n) != cudaSuccess) {
      cudaGetLastError();
      p = nullptr;
      return false;
    }
    bytes = n;
    return true;
  }
};

bool real_type(cudaDataType_t t) { return t == CUDA_R_32F || t == CUDA_R_64F; }
bool state_type(cudaDataType_t t) { return t == CUDA_R_32F || t == CUDA_R_64F || t == CUDA_C_32F || t == CUDA_C_64F; }

int64_t span_of(const std::vector<int64_t>& ext, const std::vector<int64_t>& str) {
  int64_t s = 1;
  for (size_t i = 0; i < ext.size(); ++i) s += (ext[i] - 1) * std::max<int64_t>(str[i], 0);
  return s;
}

// The elements of a strided device tensor, packed column-major, as complex doubles.
bool read_tensor(const void* dev, cudaDataType_t t, const std::vector<int64_t>& ext, const std::vector<int64_t>& str,
                 std::vector<cd>& out) {
  const std::vector<int64_t> s = str.empty() ? packed_strides(ext) : str;
  std::vector<cd> raw;
  if (!download(dev, t, (size_t)span_of(ext, s), raw)) return false;
  const int64_t total = volume(ext);
  out.assign((size_t)total, cd{});
  std::vector<int64_t> ix(ext.size(), 0);
  for (int64_t k = 0; k < total; ++k) {
    int64_t off = 0;
    for (size_t j = 0; j < ext.size(); ++j) off += ix[j] * s[j];
    out[(size_t)k] = raw[(size_t)off];
    for (size_t j = 0; j < ext.size(); ++j) {
      if (++ix[j] < ext[j]) break;
      ix[j] = 0;
    }
  }
  return true;
}

// Writes packed column-major elements into a strided device tensor, leaving
// the elements between the strides alone.
bool write_tensor(void* dev, cudaDataType_t t, const std::vector<int64_t>& ext, const std::vector<int64_t>& str,
                  const std::vector<cd>& v) {
  const std::vector<int64_t> s = str.empty() ? packed_strides(ext) : str;
  if (s == packed_strides(ext)) return upload(dev, t, v);
  std::vector<cd> raw;
  if (!download(dev, t, (size_t)span_of(ext, s), raw)) return false;
  const int64_t total = volume(ext);
  std::vector<int64_t> ix(ext.size(), 0);
  for (int64_t k = 0; k < total; ++k) {
    int64_t off = 0;
    for (size_t j = 0; j < ext.size(); ++j) off += ix[j] * s[j];
    raw[(size_t)off] = v[(size_t)k];
    for (size_t j = 0; j < ext.size(); ++j) {
      if (++ix[j] < ext[j]) break;
      ix[j] = 0;
    }
  }
  return upload(dev, t, raw);
}

// A host scalar of the state's type, written at `p` (complex: both parts;
// real: the real part).
void put_scalar(void* p, cudaDataType_t t, cd v) {
  switch (t) {
    case CUDA_R_32F: { float x = (float)v.real(); std::memcpy(p, &x, 4); break; }
    case CUDA_R_64F: { double x = v.real(); std::memcpy(p, &x, 8); break; }
    case CUDA_C_32F: { float x[2] = {(float)v.real(), (float)v.imag()}; std::memcpy(p, x, 8); break; }
    default: { double x[2] = {v.real(), v.imag()}; std::memcpy(p, x, 16); break; }
  }
}

cd get_scalar(const void* p, cudaDataType_t t) {
  switch (t) {
    case CUDA_R_32F: { float x; std::memcpy(&x, p, 4); return x; }
    case CUDA_R_64F: { double x; std::memcpy(&x, p, 8); return x; }
    case CUDA_C_32F: { float x[2]; std::memcpy(x, p, 8); return cd(x[0], x[1]); }
    default: { double x[2]; std::memcpy(x, p, 16); return cd(x[0], x[1]); }
  }
}

// ---- what is applied to a state ----

struct UserTensor {
  const void* data = nullptr;
  std::vector<int64_t> strides;  // empty: the default layout
};

enum OpKind { OP_GATE, OP_DIAG, OP_CONTROLLED, OP_UCHANNEL, OP_GCHANNEL, OP_NETPRODUCT, OP_NETMPO };

struct Op {
  OpKind kind = OP_GATE;
  int64_t id = 0;                         // the first of the integer ids it holds
  int64_t id_count = 1;
  bool immutable = true, adjoint = false, unitary = false;
  std::vector<int32_t> modes;             // acted state modes (a controlled operator's: the target)
  std::vector<int32_t> ctl_modes;
  std::vector<int64_t> ctl_vals;
  std::vector<UserTensor> tensors;        // one; a channel's; a network operator's factors
  std::vector<std::vector<int32_t>> fmodes;  // a network operator's: the state modes of each tensor
  std::vector<std::vector<int64_t>> fext;    // an MPO's: the extents of each tensor
  std::vector<double> probs;
  void* grad = nullptr;
  std::vector<int64_t> grad_strides;
};

// One component of a network operator: a product of tensors on disjoint
// modes, or a matrix product operator, with a complex coefficient.
struct NetComponent {
  cd coef{1, 0};
  bool mpo = false;
  std::vector<std::vector<int32_t>> fmodes;
  std::vector<std::vector<int64_t>> fext;  // an MPO's tensors' extents
  std::vector<UserTensor> tensors;
};

struct NetOp {
  uint32_t magic = kMagicNetOp;
  Handle* h = nullptr;
  int n = 0;
  std::vector<int64_t> ext;
  cudaDataType_t type = CUDA_C_64F;
  std::vector<NetComponent> comps;
  uint64_t version = 0;  // bumped when a component is appended
};

// A computed or initial matrix product state, in the caller's buffers.
struct MpsTensors {
  std::vector<std::vector<int64_t>> ext, str;
  std::vector<void*> ptr;
};

struct State {
  uint32_t magic = kMagicState;
  Handle* h = nullptr;
  bool mixed = false;
  int n = 0;
  std::vector<int64_t> ext;
  cudaDataType_t type = CUDA_C_64F;
  std::vector<Op> ops;
  int64_t next_id = 0;
  uint64_t epoch = 0;  // bumped by every structural change; properties made earlier are outdated
  mutable std::map<int64_t, std::unique_ptr<DevBuf>> vacuum;
  // configuration
  int32_t num_hyper = 0;
  int32_t canonical_center = -1;
  double svd_abs = 0, svd_rel = 0, svd_discarded = 0;
  int32_t svd_norm = CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE;
  int32_t svd_algo = CUTENSORNET_TENSOR_SVD_ALGO_GESVD;
  int64_t algo_params[2] = {0, 0};
  int32_t mpo_application = 0;
  int32_t gauge = 0;
  // an initial MPS, and the finalised target MPS
  bool has_init_mps = false;
  MpsTensors init_mps;
  bool mps_finalized = false;
  std::vector<std::vector<int64_t>> mps_max_ext;
  std::vector<std::vector<int64_t>> mps_out_str;  // the caller's output strides; empty: default
  // preparation and the computed state
  bool prepared = false;
  uint64_t prepared_epoch = 0;
  int64_t need = 0;
  double flops = 0;
  Path path;
  bool mps_computed = false;
  bool dense_computed = false;
  uint64_t mps_epoch = 0;
  // s.mps holds an MPS the state's properties read (measured: after more operators are applied, an
  // accessor still returns the MPS last computed or captured, until it is computed again).
  bool mps_held = false;
  MpsTensors mps;
  // an RNG for channels resolved at random
  std::mt19937_64 rng{std::random_device{}()};
};

State* state_of(void* p) { return as<State>(p, kMagicState); }
NetOp* netop_of(void* p) { return as<NetOp>(p, kMagicNetOp); }

// The device vector e_0 of dimension d, for the vacuum state.
const void* vacuum_vector(const State& s, int64_t d) {
  auto it = s.vacuum.find(d);
  if (it == s.vacuum.end()) {
    auto b = std::make_unique<DevBuf>();
    if (!b->alloc((size_t)d * elem_bytes(s.type))) return nullptr;
    std::vector<cd> v((size_t)d, cd{});
    v[0] = 1;
    upload(b->p, s.type, v);
    it = s.vacuum.emplace(d, std::move(b)).first;
  }
  return it->second->p;
}

// ---- building the network of a property ----

// A tensor network under construction: tensors with their data pointers,
// and the device memory of tensors made for this network alone.
struct NetBuild {
  Network net;
  std::vector<const void*> data;
  int32_t next_label = 0;
  std::vector<std::unique_ptr<DevBuf>> temps;
  bool dry = false;  // preparing: shapes only, no tensors are made

  int32_t fresh() { return next_label++; }
  void add(std::vector<int32_t> modes, std::vector<int64_t> ext, std::vector<int64_t> strides, const void* ptr,
           bool conj) {
    NetTensor t;
    t.modes = std::move(modes);
    t.extents = std::move(ext);
    t.strides = std::move(strides);
    t.q.isConjugate = conj ? 1 : 0;
    net.in.push_back(std::move(t));
    data.push_back(ptr);
  }
  // A tensor made here, owned until the network is dropped.
  const void* make(cudaDataType_t type, const std::vector<cd>& v) {
    if (dry) return nullptr;
    auto b = std::make_unique<DevBuf>();
    if (!b->alloc(v.size() * elem_bytes(type)) || !upload(b->p, type, v)) return nullptr;
    temps.push_back(std::move(b));
    return temps.back()->p;
  }
};

// Replaces mode label `from` by `to` in every tensor (the legs are joined).
void rename_label(NetBuild& b, int32_t from, int32_t to) {
  for (NetTensor& t : b.net.in)
    for (int32_t& m : t.modes)
      if (m == from) m = to;
}

struct Side {
  std::vector<int32_t> wire;  // the current label of each state mode
  bool conj = false;
};

// What the two sides of one network share: which operators are cancelled,
// which tensor a stochastic channel picked, the index of a mixed channel.
struct Circ {
  std::vector<char> cancelled;
  std::vector<int> choice;                    // a pure state's unitary channels: the picked tensor
  std::map<size_t, int32_t> chan_label;       // a mixed channel's summation index
};

bool op_unitary_like(const State& s, const Op& op) {
  switch (op.kind) {
    case OP_UCHANNEL: return true;
    case OP_GCHANNEL: return s.mixed;  // trace preserving: its Kraus operators cancel in a trace
    default: return op.unitary;
  }
}

std::vector<int32_t> op_modes(const Op& op) {
  std::vector<int32_t> m = op.modes;
  m.insert(m.end(), op.ctl_modes.begin(), op.ctl_modes.end());
  for (const auto& f : op.fmodes) m.insert(m.end(), f.begin(), f.end());
  return m;
}

// The operators that cancel against their adjoint when the state's wires are
// joined to its conjugate's, except on the `touched` modes: working back from
// the last operator, a unitary operator none of whose modes are touched by
// an operator that stays is dropped. This is what a measured RTX 3060 does
// (also with an operator whose unitary flag is wrong).
std::vector<char> lightcone(const State& s, const std::set<int32_t>& touched) {
  std::vector<char> gone(s.ops.size(), 0);
  std::set<int32_t> blocked = touched;
  for (size_t i = s.ops.size(); i-- > 0;) {
    const Op& op = s.ops[i];
    const std::vector<int32_t> ms = op_modes(op);
    bool free = op_unitary_like(s, op);
    for (int32_t m : ms) free = free && !blocked.count(m);
    if (free) gone[i] = 1;
    else blocked.insert(ms.begin(), ms.end());
  }
  return gone;
}

// The tensor of a gate-like piece: the legs `first` then `second` (the two
// halves of its modes), the data as the caller laid it out.
void add_gate_tensor(NetBuild& b, const State& s, const std::vector<int32_t>& first,
                     const std::vector<int32_t>& second, const std::vector<int64_t>& ext_first,
                     const std::vector<int64_t>& ext_second, const UserTensor& t, bool conj) {
  std::vector<int32_t> modes = first;
  modes.insert(modes.end(), second.begin(), second.end());
  std::vector<int64_t> ext = ext_first;
  ext.insert(ext.end(), ext_second.begin(), ext_second.end());
  (void)s;
  b.add(std::move(modes), std::move(ext), t.strides, t.data, conj);
}

std::vector<int64_t> extents_of(const State& s, const std::vector<int32_t>& modes) {
  std::vector<int64_t> e;
  for (int32_t m : modes) e.push_back(s.ext[(size_t)m]);
  return e;
}

std::vector<int32_t> rev32(std::vector<int32_t> v) {
  std::reverse(v.begin(), v.end());
  return v;
}
std::vector<int64_t> rev64(std::vector<int64_t> v) {
  std::reverse(v.begin(), v.end());
  return v;
}

// An MPS's tensors (the caller's, or computed ones) for one side of a network.
void emit_mps_tensors(NetBuild& b, const State& s, const MpsTensors& m, Side& side) {
  const int n = s.n;
  std::vector<int32_t> bond((size_t)std::max(n - 1, 0));
  for (int32_t& l : bond) l = b.fresh();
  for (int i = 0; i < n; ++i) {
    const int32_t w = b.fresh();
    std::vector<int32_t> modes;
    if (i > 0) modes.push_back(bond[(size_t)i - 1]);
    modes.push_back(w);
    if (i + 1 < n) modes.push_back(bond[(size_t)i]);
    b.add(modes, m.ext[(size_t)i], m.str[(size_t)i], m.ptr[(size_t)i], side.conj);
    side.wire[(size_t)i] = w;
  }
}

// Whether the properties start from the stored MPS instead of the operators.
bool reads_mps(const State& s) { return s.mps_computed || (s.mps_finalized && s.mps_held); }

// The start of a side: the computed MPS, the caller's initial MPS, or the vacuum.
void emit_initial(NetBuild& b, const State& s, Side& side) {
  if (reads_mps(s)) {
    emit_mps_tensors(b, s, s.mps, side);
    return;
  }
  if (s.has_init_mps) {
    emit_mps_tensors(b, s, s.init_mps, side);
    return;
  }
  for (int i = 0; i < s.n; ++i) {
    const int32_t w = b.fresh();
    b.add({w}, {s.ext[(size_t)i]}, {}, vacuum_vector(s, s.ext[(size_t)i]), false);
    side.wire[(size_t)i] = w;
  }
}

// A tensor of a product or an MPO as a gate: `cols` (the first half of its
// modes) and `rows` are the labels of each acted mode, in the order of
// `modes` (forward order within each half).
void add_product_tensor(NetBuild& b, const State& s, const std::vector<int32_t>& modes, const UserTensor& t,
                        const std::vector<int32_t>& cols, const std::vector<int32_t>& rows, bool swap_halves,
                        bool conj) {
  const auto ext = extents_of(s, modes);
  if (!swap_halves) add_gate_tensor(b, s, cols, rows, ext, ext, t, conj);
  else add_gate_tensor(b, s, rows, cols, ext, ext, t, conj);
}

// A matrix product operator's tensors between `cols` and `rows` wires (one
// per site, indexed by state mode). Site tensors have the modes (left bond,
// column, right bond, row); the first has no left bond and the last no right.
void add_mpo_tensors(NetBuild& b, const State& s, const std::vector<std::vector<int32_t>>& fmodes,
                     const std::vector<std::vector<int64_t>>& fext, const std::vector<UserTensor>& tensors,
                     const std::vector<int32_t>& cols, const std::vector<int32_t>& rows, bool swap_halves,
                     bool conj) {
  (void)s;
  const size_t L = fmodes.size();
  std::vector<int32_t> bond(L > 0 ? L - 1 : 0);
  for (int32_t& l : bond) l = b.fresh();
  for (size_t i = 0; i < L; ++i) {
    const int32_t m = fmodes[i][0];
    const int32_t col = swap_halves ? rows[(size_t)m] : cols[(size_t)m], row = swap_halves ? cols[(size_t)m] : rows[(size_t)m];
    std::vector<int32_t> modes;
    if (i > 0) modes.push_back(bond[i - 1]);
    modes.push_back(col);
    if (i + 1 < L) modes.push_back(bond[i]);
    modes.push_back(row);
    b.add(modes, fext[i], tensors[i].strides, tensors[i].data, conj);
  }
}

// The operator `op` on one side of the network. A gate-like operator moves
// the wires of the modes it acts on to fresh labels; a diagonal operator and
// the control legs of a controlled one keep theirs.
Status emit_op(NetBuild& b, const State& s, const Op& op, size_t idx, Circ& c, Side& side, const char* api) {
  const bool cj = op.adjoint != side.conj;
  auto gate = [&](const UserTensor& t, bool conj, bool adjoint) {
    std::vector<int32_t> in, out;
    for (int32_t m : op.modes) {
      in.push_back(side.wire[(size_t)m]);
      out.push_back(b.fresh());
    }
    const auto ext = rev64(extents_of(s, op.modes));
    if (!adjoint) add_gate_tensor(b, s, rev32(in), rev32(out), ext, ext, t, conj);
    else add_gate_tensor(b, s, rev32(out), rev32(in), ext, ext, t, conj);
    for (size_t j = 0; j < op.modes.size(); ++j) side.wire[(size_t)op.modes[j]] = out[j];
  };
  switch (op.kind) {
    case OP_GATE: gate(op.tensors[0], cj, op.adjoint); break;
    case OP_DIAG: {
      std::vector<int32_t> w;
      for (int32_t m : op.modes) w.push_back(side.wire[(size_t)m]);
      b.add(rev32(w), rev64(extents_of(s, op.modes)), op.tensors[0].strides, op.tensors[0].data, cj);
      break;
    }
    case OP_CONTROLLED: {
      // controlled-U = I + P (U - I), P the product of the controls' projectors:
      // a sum over a two-valued index s that every piece shares. Controls are
      // diagonal (their wires stay), the target carries (U - I) for s = 1.
      const int32_t t = op.modes[0];
      const int64_t d = s.ext[(size_t)t];
      const int32_t sl = b.fresh();
      for (size_t j = 0; j < op.ctl_modes.size(); ++j) {
        const int32_t cm = op.ctl_modes[j];
        const int64_t dj = s.ext[(size_t)cm];
        std::vector<cd> v((size_t)(dj * 2), cd{});
        for (int64_t x = 0; x < dj; ++x) {
          v[(size_t)x] = 1;
          v[(size_t)(x + dj)] = x == op.ctl_vals[j] ? 1.0 : 0.0;
        }
        b.add({side.wire[(size_t)cm], sl}, {dj, 2}, {}, b.make(s.type, v), false);
      }
      const int32_t in = side.wire[(size_t)t], out = b.fresh();
      const void* tp = nullptr;
      if (!b.dry) {
        std::vector<cd> u;  // u[c + d*r] = U[r][c]
        if (!read_tensor(op.tensors[0].data, s.type, {d, d}, op.tensors[0].strides, u))
          return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "a controlled operator's tensor could not be read");
        std::vector<cd> v((size_t)(d * d * 2), cd{});
        for (int64_t r = 0; r < d; ++r)
          for (int64_t cc = 0; cc < d; ++cc) {
            cd val = op.adjoint ? std::conj(u[(size_t)(r + d * cc)]) : u[(size_t)(cc + d * r)];
            val -= r == cc ? 1.0 : 0.0;
            if (side.conj) val = std::conj(val);
            v[(size_t)(cc + d * r)] = r == cc ? cd(side.conj ? 1.0 : 1.0, 0) : cd{};
            v[(size_t)(cc + d * r + d * d)] = val;
          }
        tp = b.make(s.type, v);
        if (!tp) return fail(CUTENSORNET_STATUS_ALLOC_FAILED, api, "no device memory for a controlled operator");
      }
      b.add({in, out, sl}, {d, d, 2}, {}, tp, false);
      side.wire[(size_t)t] = out;
      break;
    }
    case OP_UCHANNEL:
    case OP_GCHANNEL: {
      if (op.kind == OP_UCHANNEL && !s.mixed) {  // a pure state: the tensor picked at random
        gate(op.tensors[(size_t)std::max(c.choice[idx], 0)], side.conj, false);
        break;
      }
      // a sum over the channel's operators: a stacked tensor with the index k that
      // the ket and bra copies share
      const int64_t K = (int64_t)op.tensors.size();
      auto it = c.chan_label.find(idx);
      if (it == c.chan_label.end()) it = c.chan_label.emplace(idx, b.fresh()).first;
      std::vector<int32_t> in, out;
      for (int32_t m : op.modes) {
        in.push_back(side.wire[(size_t)m]);
        out.push_back(b.fresh());
      }
      const auto ext = rev64(extents_of(s, op.modes));
      std::vector<int32_t> modes = rev32(in);
      for (int32_t o : rev32(out)) modes.push_back(o);
      modes.push_back(it->second);
      std::vector<int64_t> ex = ext;
      ex.insert(ex.end(), ext.begin(), ext.end());
      ex.push_back(K);
      const void* tp = nullptr;
      if (!b.dry) {
        std::vector<cd> v;
        const int64_t vol = volume(ex) / K;
        for (int64_t k = 0; k < K; ++k) {
          std::vector<cd> u;
          std::vector<int64_t> ue = ext;
          ue.insert(ue.end(), ext.begin(), ext.end());
          if (!read_tensor(op.tensors[(size_t)k].data, s.type, ue, op.tensors[0].strides, u))
            return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "a channel's tensor could not be read");
          const double f = op.kind == OP_UCHANNEL ? std::sqrt(op.probs[(size_t)k]) : 1.0;
          for (auto& x : u) x = (side.conj ? std::conj(x) : x) * f;
          v.insert(v.end(), u.begin(), u.end());
        }
        (void)vol;
        tp = b.make(s.type, v);
        if (!tp) return fail(CUTENSORNET_STATUS_ALLOC_FAILED, api, "no device memory for a channel");
      }
      b.add(modes, ex, {}, tp, false);
      for (size_t j = 0; j < op.modes.size(); ++j) side.wire[(size_t)op.modes[j]] = out[j];
      break;
    }
    case OP_NETPRODUCT: {
      std::vector<std::vector<int32_t>> colsets;
      std::vector<int32_t> rows((size_t)s.n, -1), cols = side.wire;
      for (size_t f = 0; f < op.fmodes.size(); ++f)
        for (int32_t m : op.fmodes[f]) rows[(size_t)m] = b.fresh();
      for (size_t f = 0; f < op.fmodes.size(); ++f) {
        std::vector<int32_t> cl, rw;
        for (int32_t m : op.fmodes[f]) cl.push_back(cols[(size_t)m]), rw.push_back(rows[(size_t)m]);
        add_product_tensor(b, s, op.fmodes[f], op.tensors[f], cl, rw, op.adjoint, cj);
      }
      for (size_t f = 0; f < op.fmodes.size(); ++f)
        for (int32_t m : op.fmodes[f]) side.wire[(size_t)m] = rows[(size_t)m];
      break;
    }
    case OP_NETMPO: {
      std::vector<int32_t> rows(side.wire);
      for (const auto& f : op.fmodes) rows[(size_t)f[0]] = b.fresh();
      add_mpo_tensors(b, s, op.fmodes, op.fext, op.tensors, side.wire, rows, op.adjoint, cj);
      for (const auto& f : op.fmodes) side.wire[(size_t)f[0]] = rows[(size_t)f[0]];
      break;
    }
  }
  (void)api;
  return CUTENSORNET_STATUS_SUCCESS;
}

// A side's whole circuit: its start, then the operators that are not cancelled.
Status emit_side(NetBuild& b, const State& s, Circ& c, Side& side, const char* api) {
  side.wire.assign((size_t)s.n, -1);
  emit_initial(b, s, side);
  if (reads_mps(s)) return CUTENSORNET_STATUS_SUCCESS;  // the operators are in the MPS
  for (size_t i = 0; i < s.ops.size(); ++i) {
    if (!c.cancelled.empty() && c.cancelled[i]) continue;
    if (Status st = emit_op(b, s, s.ops[i], i, c, side, api)) return st;
  }
  return CUTENSORNET_STATUS_SUCCESS;
}

// The ket and bra wires of a state and its conjugate (for a mixed state, the
// density matrix's own two sides).
struct Wires {
  std::vector<int32_t> ket, bra;
};

Status build_doubled(NetBuild& b, const State& s, Circ& c, Wires& w, const char* api) {
  Side k, r;
  r.conj = true;
  if (Status st = emit_side(b, s, c, k, api)) return st;
  if (Status st = emit_side(b, s, c, r, api)) return st;
  w.ket = k.wire;
  w.bra = r.wire;
  return CUTENSORNET_STATUS_SUCCESS;
}

// A pure state's unitary channels pick one operator each, at random.
void resolve_channels(const State& s, Circ& c, std::mt19937_64& rng, bool random) {
  c.choice.assign(s.ops.size(), 0);
  if (s.mixed) return;
  for (size_t i = 0; i < s.ops.size(); ++i) {
    const Op& op = s.ops[i];
    if (op.kind != OP_UCHANNEL) continue;
    if (!random) continue;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng), acc = 0;
    int pick = (int)op.probs.size() - 1;
    for (size_t k = 0; k < op.probs.size(); ++k) {
      acc += op.probs[k];
      if (r < acc) {
        pick = (int)k;
        break;
      }
    }
    c.choice[i] = pick;
  }
}

// Fixes the network's output: its modes (their extents come from the inputs)
// and strides, the data type and the compute type.
void finish_net(NetBuild& b, const State& s, std::vector<int32_t> out_modes, std::vector<int64_t> out_strides) {
  const auto ext = extent_map(b.net);
  b.net.out = NetTensor{};
  b.net.out.modes = std::move(out_modes);
  for (int32_t m : b.net.out.modes) b.net.out.extents.push_back(ext.at(m));
  b.net.out.strides = std::move(out_strides);
  b.net.out_set = true;
  b.net.type = s.type;
  b.net.type_set = true;
  b.net.compute = default_compute(s.type);
  b.net.compute_set = true;
}

// A one-hot vector's host elements.
std::vector<cd> onehot_values(int64_t d, int64_t at) {
  std::vector<cd> v((size_t)d, cd{});
  if (at >= 0 && at < d) v[(size_t)at] = 1;
  return v;
}

// ---- planning and running a network ----

struct Part {
  Path path;
  int64_t need = 0;
  double flops = 0;
  int64_t slices = 1;
};

// The path and slicing of the network within `limit` bytes of scratch.
Status plan_part(Handle* h, NetBuild& b, uint64_t limit, Part& part, const char* api) {
  if (b.net.in.empty()) return fail(CUTENSORNET_STATUS_INTERNAL_ERROR, api, "an empty network");
  const auto ext = extent_map(b.net);
  Path p;
  p.steps = greedy_path(b.net, ext);
  // Distributed: at least a slice for each rank (as the optimizer of the network API).
  World world;
  if (Status st = world_of(h, world, api)) return st;
  if (!choose_slicing(b.net, p, limit, std::max<int32_t>(1, world.size)))
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "Max device workspace size is less than minimally required!");
  Info info;
  describe(b.net, p, info);
  part.path = p;
  part.need = scratch_need(b.net, p);
  part.flops = info.flops;
  part.slices = info.num_slices;
  return CUTENSORNET_STATUS_SUCCESS;
}

// Contracts the planned network into `out`. With a communicator, the
// computations of values (amplitudes, marginals, expectation values and the
// norms) deal the slices to the ranks and sum their outputs in place
// (Dist::Sum, measured on NVIDIA's library: one in-place sum for each output
// tensor and for each norm, and one for an expectation value's components
// together); Dist::Part leaves the sum to the caller.
enum class Dist { None, Sum, Part };

Status run_part(Handle* h, NetBuild& b, const Part& part, void* out, const Workspace* w, cudaStream_t stream,
                const char* api, Dist dist = Dist::None) {
  for (const void* d : b.data)
    if (!d) return fail(CUTENSORNET_STATUS_INTERNAL_ERROR, api, "a tensor of the network has no memory");
  const auto ext = extent_map(b.net);
  const int64_t total = slices_of(part.path, ext);
  std::vector<int64_t> ids((size_t)total);
  std::iota(ids.begin(), ids.end(), (int64_t)0);
  if (dist != Dist::None) {
    World world;
    if (Status st = world_of(h, world, api)) return st;
    if (world.distributed)
      return contract_distributed(h, world, b.net, part.path, b.data, out, false, w, nullptr, ids, stream, api,
                                  dist == Dist::Sum);
  }
  return contract_slices(h, b.net, part.path, b.data, out, false, w, ids, stream, api);
}

// The scratch of a compute call, checked in the order measured on the card:
// the scratch memory must be set (a memory pool is not supported), then the
// size must cover what was prepared.
Status check_scratch(const Workspace* w, int64_t need, const char* api) {
  const void* ptr = w->ptr[CUTENSORNET_MEMSPACE_DEVICE][CUTENSORNET_WORKSPACE_SCRATCH];
  if (!ptr)
    return fail(CUTENSORNET_STATUS_NOT_SUPPORTED, api, "The use of a user-provided memory pool allocator is not supported!");
  if (w->size[CUTENSORNET_MEMSPACE_DEVICE][CUTENSORNET_WORKSPACE_SCRATCH] < need)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "Device scratch workspace size is insufficient for internal tensor storage!");
  return CUTENSORNET_STATUS_SUCCESS;
}

constexpr int64_t kExtra = 256;  // room for a scalar result, ahead of the contraction's scratch

// The caller's scratch, split: a few scalars, then the rest for contractions.
struct ScratchSplit {
  char* scalars = nullptr;
  Workspace rest;
};
ScratchSplit split_scratch(const Workspace* w, int64_t scalars_bytes) {
  ScratchSplit s;
  char* base = static_cast<char*>(w->ptr[CUTENSORNET_MEMSPACE_DEVICE][CUTENSORNET_WORKSPACE_SCRATCH]);
  const int64_t size = w->size[CUTENSORNET_MEMSPACE_DEVICE][CUTENSORNET_WORKSPACE_SCRATCH];
  s.scalars = base;
  s.rest.ptr[CUTENSORNET_MEMSPACE_DEVICE][CUTENSORNET_WORKSPACE_SCRATCH] = base + scalars_bytes;
  s.rest.size[CUTENSORNET_MEMSPACE_DEVICE][CUTENSORNET_WORKSPACE_SCRATCH] = size - scalars_bytes;
  return s;
}

// Records what a prepare call needs in the workspace descriptor: every
// preference asks for the same scratch, there is no cache, and (as measured)
// the memory set before is forgotten.
void set_prepared_sizes(Workspace* w, int64_t need) {
  set_scratch_need(w, need);
  for (auto& mem : w->ptr)
    for (void*& p : mem) p = nullptr;
  for (auto& mem : w->size)
    for (int64_t& z : mem) z = 0;
}

// A scalar the network left in the device scratch, on the host.
bool fetch_scalar(const void* dev, cudaDataType_t t, cudaStream_t stream, cd& v) {
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  std::vector<cd> r;
  if (!download(dev, t, 1, r)) return false;
  v = r[0];
  return true;
}

bool graph_capture_refused(cudaStream_t stream, const char* api, Status& st) {
  if (!capturing(stream)) return false;
  st = refuse(api, "this call reads its results back to the host and cannot be captured into a CUDA graph; call it "
                   "outside capture");
  return true;
}

}  // namespace

namespace {

// ---- the objects computed from a state ----

struct Accessor {
  uint32_t magic = kMagicAccessor;
  State* s = nullptr;
  uint64_t epoch = 0;
  std::vector<int32_t> proj;      // projected modes ([0, 2N) for a mixed state)
  std::vector<int64_t> strides;   // of the amplitudes tensor; empty: default
  int32_t hyper = 0;
  bool prepared = false;
  bool broken = false;  // a projected value out of range was given: it computes nothing after that
  int64_t need = 0;
  Part amp, norm;
  std::vector<std::unique_ptr<DevBuf>> onehot;
  double flops = 0;
};

struct Marginal {
  uint32_t magic = kMagicMarginal;
  State* s = nullptr;
  uint64_t epoch = 0;
  bool diagonal = false;
  std::vector<int32_t> open, proj;
  std::vector<int64_t> strides;
  int32_t hyper = 0;
  bool prepared = false;
  int64_t need = 0;
  Part part;
  std::vector<std::unique_ptr<DevBuf>> onehot;
  double flops = 0;
};

struct Sampler {
  uint32_t magic = kMagicSampler;
  State* s = nullptr;
  uint64_t epoch = 0;
  std::vector<int32_t> modes;  // modes to sample, in the caller's order
  int32_t hyper = 0;
  int32_t seed = 0;
  std::mt19937_64 rng{std::random_device{}()};
  bool prepared = false;
  int64_t need = 0;
  // the levels: the sampled modes in chunks, each a diagonal marginal given the earlier chunks
  std::vector<std::vector<int>> chunks;  // indices into `modes`
  std::vector<Part> parts;
  std::vector<std::unique_ptr<DevBuf>> onehot;  // one per sampled mode
  double flops = 0;
};

struct Expectation {
  uint32_t magic = kMagicExpectation;
  State* s = nullptr;
  NetOp* op = nullptr;
  uint64_t epoch = 0, op_version = 0;
  int32_t hyper = 0;
  bool prepared = false;
  int64_t need = 0;
  std::vector<Part> parts;  // one per component
  Part norm;
  double flops = 0;
};

template <class T>
Status check_state_handle(const cutensornetHandle_t handle, Handle*& h, State*& s, void* state, const char* api) {
  h = handle_of(handle);
  s = state_of(state);
  if (!h || !s) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid handle or state");
  return CUTENSORNET_STATUS_SUCCESS;
}

Status outdated(const char* what, const char* api) {
  return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, std::string(what) + " is outdated due to new changes to the tensor circuit!");
}

// Sets the sizes of the workspace a prepare call fills, checks the limit.
Status prepare_workspace(Workspace* w, int64_t need, uint64_t limit, const char* api) {
  if (!w) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid workspace descriptor");
  if ((uint64_t)need > limit)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "Max device workspace size is less than minimally required!");
  set_prepared_sizes(w, need);
  return CUTENSORNET_STATUS_SUCCESS;
}

// A property's hyper-sample configuration, and its read-back.
Status config_hyper(int32_t& hyper, int attr, int num_attr, int opt_attr, const void* v, size_t sz, const char* api) {
  if (attr != num_attr && attr != opt_attr) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "Invalid attribute!");
  if (!v || sz != 4) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "Unexpected attribute size!");
  std::memcpy(&hyper, v, 4);
  return CUTENSORNET_STATUS_SUCCESS;
}

Status get_info_value(const void* src, size_t n, void* buf, size_t sz, const char* api) {
  if (!buf || sz != n) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "attributeSize is invalid!");
  std::memcpy(buf, src, n);
  return CUTENSORNET_STATUS_SUCCESS;
}

// The extents of the amplitudes a state's accessor produces, and the
// wire each open mode is read from.
struct AccessorShape {
  std::vector<int32_t> open;  // logical modes: [0, N) kets, [N, 2N) bras
};

AccessorShape accessor_shape(const State& s, const std::vector<int32_t>& proj) {
  AccessorShape a;
  const int total = s.mixed ? 2 * s.n : s.n;
  std::vector<char> fixed((size_t)total, 0);
  for (int32_t p : proj) fixed[(size_t)p] = 1;
  for (int m = 0; m < total; ++m)
    if (!fixed[(size_t)m]) a.open.push_back(m);
  return a;
}

// Every network built for a property starts from the caller's choices of the
// stochastic channels (one per operator), and builds its own operator masks.
void begin_build(const State& s, Circ& c) {
  if (c.choice.size() != s.ops.size()) c.choice.assign(s.ops.size(), 0);
  c.chan_label.clear();
  c.cancelled.clear();
}

// The network of an accessor's amplitudes. The projected values are in the
// accessor's one-hot buffers (set by the caller); `dry` makes shapes only.
Status build_amplitudes(const State& s, Accessor& a, bool dry, NetBuild& b, Circ& c, const char* api) {
  b.dry = dry;
  begin_build(s, c);
  c.cancelled.clear();
  Wires w;
  std::vector<int32_t> ket, bra;
  if (!s.mixed) {
    Side k;
    if (Status st = emit_side(b, s, c, k, api)) return st;
    ket = k.wire;
  } else {
    if (Status st = build_doubled(b, s, c, w, api)) return st;
    ket = w.ket;
    bra = w.bra;
  }
  auto wire_of = [&](int32_t m) { return m < s.n ? ket[(size_t)m] : bra[(size_t)(m - s.n)]; };
  const auto ext_of = [&](int32_t m) { return s.ext[(size_t)(m < s.n ? m : m - s.n)]; };
  for (size_t j = 0; j < a.proj.size(); ++j)
    b.add({wire_of(a.proj[j])}, {ext_of(a.proj[j])}, {}, a.onehot[j]->p, false);
  const AccessorShape sh = accessor_shape(s, a.proj);
  std::vector<int32_t> out;
  for (int32_t m : sh.open) out.push_back(wire_of(m));
  finish_net(b, s, out, a.strides);
  return CUTENSORNET_STATUS_SUCCESS;
}

// The squared norm network: the state's circuit joined to its conjugate.
Status build_norm(const State& s, bool dry, NetBuild& b, Circ& c, const char* api) {
  b.dry = dry;
  begin_build(s, c);
  c.cancelled = lightcone(s, {});
  Wires w;
  if (Status st = build_doubled(b, s, c, w, api)) return st;
  for (int i = 0; i < s.n; ++i) rename_label(b, w.bra[(size_t)i], w.ket[(size_t)i]);
  finish_net(b, s, {}, {});
  return CUTENSORNET_STATUS_SUCCESS;
}

// A marginal's network: the circuit and its conjugate, the traced modes
// joined, the projected ones fixed, the open ones left (or, for the
// diagonal, joined and left once).
Status build_marginal(const State& s, const std::vector<int32_t>& open, const std::vector<int32_t>& proj, bool diagonal,
                      const std::vector<int64_t>& strides, const std::vector<const void*>& onehot, bool dry,
                      NetBuild& b, Circ& c, const char* api) {
  b.dry = dry;
  begin_build(s, c);
  std::set<int32_t> touched(open.begin(), open.end());
  touched.insert(proj.begin(), proj.end());
  c.cancelled = lightcone(s, touched);
  Wires w;
  if (Status st = build_doubled(b, s, c, w, api)) return st;
  for (int i = 0; i < s.n; ++i)
    if (!touched.count(i)) rename_label(b, w.bra[(size_t)i], w.ket[(size_t)i]);
  for (size_t j = 0; j < proj.size(); ++j) {
    const int32_t q = proj[j];
    b.add({w.ket[(size_t)q]}, {s.ext[(size_t)q]}, {}, onehot[j], false);
    b.add({w.bra[(size_t)q]}, {s.ext[(size_t)q]}, {}, onehot[j], false);
  }
  std::vector<int32_t> out;
  if (diagonal) {
    for (int32_t q : open) rename_label(b, w.bra[(size_t)q], w.ket[(size_t)q]);
    for (int32_t q : open) out.push_back(w.ket[(size_t)q]);
  } else {
    for (int32_t q : open) out.push_back(w.ket[(size_t)q]);
    for (int32_t q : open) out.push_back(w.bra[(size_t)q]);
  }
  finish_net(b, s, out, strides);
  return CUTENSORNET_STATUS_SUCCESS;
}

std::set<int32_t> component_modes(const NetComponent& comp) {
  std::set<int32_t> m;
  for (const auto& f : comp.fmodes) m.insert(f.begin(), f.end());
  return m;
}

// <psi| component |psi> as a scalar network.
Status build_component(const State& s, const NetComponent& comp, bool dry, NetBuild& b, Circ& c, const char* api) {
  b.dry = dry;
  begin_build(s, c);
  const std::set<int32_t> touched = component_modes(comp);
  c.cancelled = lightcone(s, touched);
  Wires w;
  if (Status st = build_doubled(b, s, c, w, api)) return st;
  for (int i = 0; i < s.n; ++i)
    if (!touched.count(i)) rename_label(b, w.bra[(size_t)i], w.ket[(size_t)i]);
  if (!comp.mpo) {
    for (size_t f = 0; f < comp.fmodes.size(); ++f) {
      std::vector<int32_t> cl, rw;
      for (int32_t m : comp.fmodes[f]) cl.push_back(w.ket[(size_t)m]), rw.push_back(w.bra[(size_t)m]);
      add_product_tensor(b, s, comp.fmodes[f], comp.tensors[f], cl, rw, false, false);
    }
  } else {
    add_mpo_tensors(b, s, comp.fmodes, comp.fext, comp.tensors, w.ket, w.bra, false, false);
  }
  finish_net(b, s, {}, {});
  return CUTENSORNET_STATUS_SUCCESS;
}

// The densely computed state (no factorization): a pure state's amplitudes
// or a mixed state's density matrix.
Status build_dense_state(const State& s, bool dry, NetBuild& b, Circ& c, const char* api) {
  b.dry = dry;
  begin_build(s, c);
  c.cancelled.clear();
  std::vector<int32_t> out;
  if (!s.mixed) {
    Side k;
    if (Status st = emit_side(b, s, c, k, api)) return st;
    out = k.wire;
  } else {
    Wires w;
    if (Status st = build_doubled(b, s, c, w, api)) return st;
    out = w.ket;
    out.insert(out.end(), w.bra.begin(), w.bra.end());
  }
  finish_net(b, s, out, {});
  return CUTENSORNET_STATUS_SUCCESS;
}

// A pure state with a general (Kraus) channel cannot be contracted: the
// channel's outcome depends on the state (measured: NOT_SUPPORTED).
Status check_channels(const State& s, const char* api) {
  if (!s.mixed && !s.mps_computed)
    for (const Op& op : s.ops)
      if (op.kind == OP_GCHANNEL)
        return fail(CUTENSORNET_STATUS_NOT_SUPPORTED, api,
                    "#ERROR(NotSupported): General noise channels are not supported for pure tensor network states!");
  return CUTENSORNET_STATUS_SUCCESS;
}

// A property needs the finalised MPS computed first (measured: creating one on
// an uncomputed MPS fails with an internal error).
bool awaits_mps(const State& s) { return s.mps_finalized && !s.mps_held; }

}  // namespace

// ---- the API: states and what is applied to them ----

namespace {

// The common checks of every call that applies something to a state.
Status state_args(const cutensornetHandle_t handle, cutensornetState_t state, Handle*& h, State*& s, const char* api) {
  h = handle_of(handle);
  s = state_of(state);
  if (!h) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid handle");
  if (!s) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid tensor network state");
  return CUTENSORNET_STATUS_SUCCESS;
}

// k distinct state modes in range.
Status check_modes(const State& s, int32_t k, const int32_t* modes, const char* api) {
  if (k <= 0 || k > s.n || !modes) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid number of state modes");
  std::set<int32_t> seen;
  for (int32_t i = 0; i < k; ++i) {
    if (modes[i] < 0 || modes[i] >= s.n)
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "State mode id exceeds the total number of state modes!");
    if (!seen.insert(modes[i]).second)
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "State modes must not repeat!");
  }
  return CUTENSORNET_STATUS_SUCCESS;
}

// Records `op` as the next operator: its ids, and that the state changed.
// A pure state's operator holds `count` integer ids, a mixed state's twice as many
// (a ket and a bra copy of each tensor).
void add_op(State& s, Op&& op, int64_t count, int64_t* id_out) {
  op.id = s.next_id;
  op.id_count = count;
  s.next_id += s.mixed ? 2 * count : count;
  if (id_out) *id_out = op.id;
  s.ops.push_back(std::move(op));
  s.epoch++;
  s.prepared = false;
  s.mps_computed = false;
  s.dense_computed = false;
}

std::vector<int64_t> copy_strides(const int64_t* p, size_t n) {
  return p ? std::vector<int64_t>(p, p + n) : std::vector<int64_t>();
}

Status apply_one(const cutensornetHandle_t handle, cutensornetState_t state, OpKind kind, int32_t k,
                 const int32_t* modes, void* data, const int64_t* strides, int32_t immutable, int32_t adjoint,
                 int32_t unitary, int64_t* tensorId, const char* api, int32_t nctl = 0, const int32_t* ctl = nullptr,
                 const int64_t* ctl_vals = nullptr, void* grad = nullptr, const int64_t* grad_strides = nullptr) {
  Handle* h;
  State* s;
  if (Status st = state_args(handle, state, h, s, api)) return st;
  if (Status st = check_modes(*s, k, modes, api)) return st;
  if (!data || !tensorId) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "a null pointer was passed");
  Op op;
  op.kind = kind;
  op.immutable = immutable != 0;
  op.adjoint = adjoint != 0;
  op.unitary = unitary != 0;
  op.modes.assign(modes, modes + k);
  if (kind == OP_CONTROLLED) {
    if (nctl <= 0 || !ctl) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid control modes");
    if (k != 1)
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "Currently, multi-controlled single-target gates are supported!");
    std::set<int32_t> seen(op.modes.begin(), op.modes.end());
    for (int32_t i = 0; i < nctl; ++i) {
      if (ctl[i] < 0 || ctl[i] >= s->n)
        return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "State mode id exceeds the total number of state modes!");
      if (!seen.insert(ctl[i]).second)
        return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "State modes must not repeat!");
      const int64_t v = ctl_vals ? ctl_vals[i] : s->ext[(size_t)ctl[i]] - 1;
      if (v < 0 || v >= s->ext[(size_t)ctl[i]])
        return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "A control value is out of the mode extent!");
      op.ctl_modes.push_back(ctl[i]);
      op.ctl_vals.push_back(v);
    }
  }
  UserTensor t;
  t.data = data;
  t.strides = copy_strides(strides, kind == OP_DIAG ? (size_t)k : 2 * (size_t)k);
  op.tensors.push_back(std::move(t));
  if (grad) {
    op.grad = grad;
    op.grad_strides = copy_strides(grad_strides, 2 * (size_t)k);
  }
  add_op(*s, std::move(op), kind == OP_CONTROLLED ? 1 + nctl : 1, tensorId);
  return CUTENSORNET_STATUS_SUCCESS;
}

// The operator and tensor an integer id names: an operator's ids are a
// contiguous range, one per tensor (twice as many in a mixed state).
bool find_tensor(State& s, int64_t id, Op*& op, size_t& tensor, bool& found_internal) {
  found_internal = false;
  for (Op& o : s.ops) {
    const int64_t per = s.mixed ? 2 : 1;
    const int64_t span = o.id_count * per;
    if (id >= o.id && id < o.id + span) {
      op = &o;
      tensor = (size_t)((id - o.id) / per);
      return true;
    }
  }
  found_internal = id >= 0 && id < s.next_id;
  return false;
}

Status update_operator(const cutensornetHandle_t handle, cutensornetState_t state, int64_t id, void* data,
                       int32_t unitary, const char* api) {
  Handle* h;
  State* s;
  if (Status st = state_args(handle, state, h, s, api)) return st;
  Op* op = nullptr;
  size_t tensor = 0;
  bool internal = false;
  if (!find_tensor(*s, id, op, tensor, internal)) {
    if (internal)
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "Tensor operator to be updated is not mutable, thus cannot be updated!");
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "Tensor operator to be updated is not found by its integer id!");
  }
  if (!data) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "a null pointer was passed");
  if (op->kind == OP_UCHANNEL || op->kind == OP_GCHANNEL)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "a channel cannot be updated");
  if (op->immutable)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "Tensor operator to be updated is not mutable, thus cannot be updated!");
  op->tensors[std::min(tensor, op->tensors.size() - 1)].data = data;
  const bool u = unitary != 0;
  if (op->kind != OP_NETPRODUCT && op->kind != OP_NETMPO && u != op->unitary) {
    op->unitary = u;  // a structural change: what is prepared was built for the old flag
    s->epoch++;
    s->prepared = false;
  }
  return CUTENSORNET_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

cutensornetStatus_t cutensornetCreateState(const cutensornetHandle_t handle, cutensornetStatePurity_t purity,
                                           int32_t numStateModes, const int64_t* stateModeExtents,
                                           cudaDataType_t dataType, cutensornetState_t* tensorNetworkState) {
  const char* api = "cutensornetCreateState";
  Handle* h = handle_of(handle);
  if (!h) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid handle");
  const int pv = enum_value(purity), dt = enum_value(dataType);
  if (!tensorNetworkState || (pv != 0 && pv != 1) || numStateModes <= 0 || !stateModeExtents)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  for (int32_t i = 0; i < numStateModes; ++i)
    if (stateModeExtents[i] <= 0) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "extents must be positive");
  if (dt != (int)CUDA_R_32F && dt != (int)CUDA_R_64F && dt != (int)CUDA_C_32F && dt != (int)CUDA_C_64F)
    return fail(CUTENSORNET_STATUS_NOT_SUPPORTED, api, "unsupported data type");
  auto s = std::make_unique<State>();
  s->h = h;
  s->mixed = pv == 1;
  s->n = numStateModes;
  s->ext.assign(stateModeExtents, stateModeExtents + numStateModes);
  s->type = dataType;
  // the vacuum's tensors take the first integer ids: one per mode and one
  // more for a pure state, a ket and a bra copy of them for a mixed state
  s->next_id = s->mixed ? 2 * (int64_t)numStateModes + 1 : (int64_t)numStateModes + 1;
  *tensorNetworkState = s.release();
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetDestroyState(cutensornetState_t tensorNetworkState) {
  State* s = state_of(tensorNetworkState);
  if (!s) return tensorNetworkState ? CUTENSORNET_STATUS_INVALID_VALUE : CUTENSORNET_STATUS_SUCCESS;
  s->magic = 0;
  delete s;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetStateApplyTensorOperator(const cutensornetHandle_t handle,
                                                        cutensornetState_t tensorNetworkState,
                                                        int32_t numStateModes, const int32_t* stateModes,
                                                        void* tensorData, const int64_t* tensorModeStrides,
                                                        const int32_t immutable, const int32_t adjoint,
                                                        const int32_t unitary, int64_t* tensorId) {
  return apply_one(handle, tensorNetworkState, OP_GATE, numStateModes, stateModes, tensorData, tensorModeStrides,
                   immutable, adjoint, unitary, tensorId, "cutensornetStateApplyTensorOperator");
}

cutensornetStatus_t cutensornetStateApplyTensor(const cutensornetHandle_t handle,
                                                cutensornetState_t tensorNetworkState, int32_t numStateModes,
                                                const int32_t* stateModes, void* tensorData,
                                                const int64_t* tensorModeStrides, const int32_t immutable,
                                                const int32_t adjoint, const int32_t unitary, int64_t* tensorId) {
  return apply_one(handle, tensorNetworkState, OP_GATE, numStateModes, stateModes, tensorData, tensorModeStrides,
                   immutable, adjoint, unitary, tensorId, "cutensornetStateApplyTensor");
}

cutensornetStatus_t cutensornetStateApplyDiagonalTensorOperator(
    const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes,
    const int32_t* stateModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable,
    const int32_t adjoint, const int32_t unitary, int64_t* tensorId) {
  return apply_one(handle, tensorNetworkState, OP_DIAG, numStateModes, stateModes, tensorData, tensorModeStrides,
                   immutable, adjoint, unitary, tensorId, "cutensornetStateApplyDiagonalTensorOperator");
}

cutensornetStatus_t cutensornetStateApplyControlledTensorOperator(
    const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numControlModes,
    const int32_t* stateControlModes, const int64_t* stateControlValues, int32_t numTargetModes,
    const int32_t* stateTargetModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable,
    const int32_t adjoint, const int32_t unitary, int64_t* tensorId) {
  return apply_one(handle, tensorNetworkState, OP_CONTROLLED, numTargetModes, stateTargetModes, tensorData,
                   tensorModeStrides, immutable, adjoint, unitary, tensorId,
                   "cutensornetStateApplyControlledTensorOperator", numControlModes, stateControlModes,
                   stateControlValues);
}

cutensornetStatus_t cutensornetStateApplyTensorOperatorWithGradient(
    const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes,
    const int32_t* stateModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable,
    const int32_t adjoint, const int32_t unitary, void* gradientData, const int64_t* gradientModeStrides,
    int64_t* tensorId) {
  if (!gradientData)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, "cutensornetStateApplyTensorOperatorWithGradient",
                "a null pointer was passed");
  return apply_one(handle, tensorNetworkState, OP_GATE, numStateModes, stateModes, tensorData, tensorModeStrides,
                   immutable, adjoint, unitary, tensorId, "cutensornetStateApplyTensorOperatorWithGradient", 0, nullptr,
                   nullptr, gradientData, gradientModeStrides);
}

cutensornetStatus_t cutensornetStateUpdateTensorOperator(const cutensornetHandle_t handle,
                                                         cutensornetState_t tensorNetworkState, int64_t tensorId,
                                                         void* tensorData, int32_t unitary) {
  return update_operator(handle, tensorNetworkState, tensorId, tensorData, unitary,
                         "cutensornetStateUpdateTensorOperator");
}

cutensornetStatus_t cutensornetStateUpdateTensor(const cutensornetHandle_t handle,
                                                 cutensornetState_t tensorNetworkState, int64_t tensorId,
                                                 void* tensorData, int32_t unitary) {
  return update_operator(handle, tensorNetworkState, tensorId, tensorData, unitary, "cutensornetStateUpdateTensor");
}

cutensornetStatus_t cutensornetStateUpdateTensorOperatorGradient(const cutensornetHandle_t handle,
                                                                 cutensornetState_t tensorNetworkState,
                                                                 int64_t tensorId, void* gradientData) {
  const char* api = "cutensornetStateUpdateTensorOperatorGradient";
  Handle* h;
  State* s;
  if (Status st = state_args(handle, tensorNetworkState, h, s, api)) return st;
  Op* op = nullptr;
  size_t t = 0;
  bool internal = false;
  if (!gradientData || !find_tensor(*s, tensorId, op, t, internal) || !op->grad)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "no tensor operator with a gradient has this id");
  op->grad = gradientData;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetStateApplyUnitaryChannel(const cutensornetHandle_t handle,
                                                        cutensornetState_t tensorNetworkState,
                                                        int32_t numStateModes, const int32_t* stateModes,
                                                        int32_t numTensors, void* tensorData[],
                                                        const int64_t* tensorModeStrides,
                                                        const double probabilities[], int64_t* channelId) {
  const char* api = "cutensornetStateApplyUnitaryChannel";
  Handle* h;
  State* s;
  if (Status st = state_args(handle, tensorNetworkState, h, s, api)) return st;
  if (Status st = check_modes(*s, numStateModes, stateModes, api)) return st;
  if (numTensors <= 0 || !tensorData || !probabilities || !channelId)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  Op op;
  op.kind = OP_UCHANNEL;
  op.unitary = true;
  op.modes.assign(stateModes, stateModes + numStateModes);
  for (int32_t i = 0; i < numTensors; ++i) {
    if (!tensorData[i] || !(probabilities[i] >= 0))
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid tensor or probability");
    UserTensor t;
    t.data = tensorData[i];
    t.strides = copy_strides(tensorModeStrides, 2 * (size_t)numStateModes);
    op.tensors.push_back(std::move(t));
    op.probs.push_back(probabilities[i]);
  }
  // measured: a mixed state's channel holds five integer ids and is named by the fifth
  if (s->mixed) {
    op.id = s->next_id;
    op.id_count = 5;
    s->next_id += 5;
    *channelId = op.id + 4;
    s->ops.push_back(std::move(op));
    s->epoch++;
    s->prepared = false;
    s->mps_computed = false;
  } else {
    add_op(*s, std::move(op), 1, channelId);
  }
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetStateApplyGeneralChannel(const cutensornetHandle_t handle,
                                                        cutensornetState_t tensorNetworkState,
                                                        int32_t numStateModes, const int32_t* stateModes,
                                                        int32_t numTensors, void* tensorData[],
                                                        const int64_t* tensorModeStrides, int64_t* channelId) {
  const char* api = "cutensornetStateApplyGeneralChannel";
  Handle* h;
  State* s;
  if (Status st = state_args(handle, tensorNetworkState, h, s, api)) return st;
  if (Status st = check_modes(*s, numStateModes, stateModes, api)) return st;
  if (numTensors <= 0 || !tensorData || !channelId) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  Op op;
  op.kind = OP_GCHANNEL;
  op.modes.assign(stateModes, stateModes + numStateModes);
  for (int32_t i = 0; i < numTensors; ++i) {
    if (!tensorData[i]) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid tensor");
    UserTensor t;
    t.data = tensorData[i];
    t.strides = copy_strides(tensorModeStrides, 2 * (size_t)numStateModes);
    op.tensors.push_back(std::move(t));
  }
  if (s->mixed) {
    op.id = s->next_id;
    op.id_count = 5;
    s->next_id += 5;
    *channelId = op.id + 4;
    s->ops.push_back(std::move(op));
    s->epoch++;
    s->prepared = false;
    s->mps_computed = false;
  } else {
    add_op(*s, std::move(op), 1, channelId);
  }
  return CUTENSORNET_STATUS_SUCCESS;
}

}  // extern "C"

// ---- the API: network operators ----

extern "C" {

cutensornetStatus_t cutensornetCreateNetworkOperator(const cutensornetHandle_t handle, int32_t numStateModes,
                                                     const int64_t stateModeExtents[], cudaDataType_t dataType,
                                                     cutensornetNetworkOperator_t* tensorNetworkOperator) {
  const char* api = "cutensornetCreateNetworkOperator";
  Handle* h = handle_of(handle);
  if (!h || !tensorNetworkOperator || numStateModes <= 0 || !stateModeExtents)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  for (int32_t i = 0; i < numStateModes; ++i)
    if (stateModeExtents[i] <= 0) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "extents must be positive");
  auto op = std::make_unique<NetOp>();
  op->h = h;
  op->n = numStateModes;
  op->ext.assign(stateModeExtents, stateModeExtents + numStateModes);
  op->type = dataType;
  *tensorNetworkOperator = op.release();
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetNetworkOperatorAppendProduct(
    const cutensornetHandle_t handle, cutensornetNetworkOperator_t tensorNetworkOperator,
    cuDoubleComplex coefficient, int32_t numTensors, const int32_t numStateModes[], const int32_t* stateModes[],
    const int64_t* tensorModeStrides[], const void* tensorData[], int64_t* componentId) {
  const char* api = "cutensornetNetworkOperatorAppendProduct";
  Handle* h = handle_of(handle);
  NetOp* op = netop_of(tensorNetworkOperator);
  if (!h || !op) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid handle or operator");
  if (numTensors <= 0 || !numStateModes || !stateModes || !tensorData)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  NetComponent comp;
  comp.coef = cd(cuCreal(coefficient), cuCimag(coefficient));
  std::set<int32_t> seen;
  for (int32_t i = 0; i < numTensors; ++i) {
    if (numStateModes[i] <= 0 || !stateModes[i] || !tensorData[i])
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid tensor factor");
    std::vector<int32_t> m(stateModes[i], stateModes[i] + numStateModes[i]);
    for (int32_t q : m) {
      if (q < 0 || q >= op->n)
        return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "State mode id exceeds the total number of state modes!");
      if (!seen.insert(q).second)
        return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "tensor factors must act on disjoint state modes");
    }
    UserTensor t;
    t.data = tensorData[i];
    if (tensorModeStrides && tensorModeStrides[i]) t.strides.assign(tensorModeStrides[i], tensorModeStrides[i] + 2 * m.size());
    comp.fmodes.push_back(std::move(m));
    comp.tensors.push_back(std::move(t));
  }
  if (componentId) *componentId = (int64_t)op->comps.size();
  op->comps.push_back(std::move(comp));
  op->version++;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetNetworkOperatorAppendMPO(
    const cutensornetHandle_t handle, cutensornetNetworkOperator_t tensorNetworkOperator,
    cuDoubleComplex coefficient, int32_t numStateModes, const int32_t stateModes[],
    const int64_t* tensorModeExtents[], const int64_t* tensorModeStrides[], const void* tensorData[],
    cutensornetBoundaryCondition_t boundaryCondition, int64_t* componentId) {
  const char* api = "cutensornetNetworkOperatorAppendMPO";
  Handle* h = handle_of(handle);
  NetOp* op = netop_of(tensorNetworkOperator);
  if (!h || !op) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid handle or operator");
  if (numStateModes <= 0 || !stateModes || !tensorModeExtents || !tensorData || enum_value(boundaryCondition) != 0)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  NetComponent comp;
  comp.mpo = true;
  comp.coef = cd(cuCreal(coefficient), cuCimag(coefficient));
  std::set<int32_t> seen;
  const int32_t L = numStateModes;
  int64_t prev_right = 1;
  for (int32_t i = 0; i < L; ++i) {
    const int32_t q = stateModes[i];
    if (q < 0 || q >= op->n)
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "State mode id exceeds the total number of state modes!");
    if (!seen.insert(q).second) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "State modes must not repeat!");
    if (!tensorModeExtents[i] || !tensorData[i]) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid MPO tensor");
    const bool first = i == 0, last = i == L - 1;
    const int rank = (first ? 0 : 1) + 1 + (last ? 0 : 1) + 1;
    std::vector<int64_t> e(tensorModeExtents[i], tensorModeExtents[i] + rank);
    // modes: (left bond), ket, (right bond), bra
    size_t at = 0;
    const int64_t left = first ? 1 : e[at++];
    const int64_t ket = e[at++];
    const int64_t right = last ? 1 : e[at++];
    const int64_t bra = e[at++];
    if (ket != op->ext[(size_t)q] || bra != op->ext[(size_t)q] || left != prev_right || left < 1 || right < 1)
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "inconsistent MPO tensor extents");
    prev_right = right;
    UserTensor t;
    t.data = tensorData[i];
    if (tensorModeStrides && tensorModeStrides[i]) t.strides.assign(tensorModeStrides[i], tensorModeStrides[i] + rank);
    comp.fmodes.push_back({q});
    comp.fext.push_back(std::move(e));
    comp.tensors.push_back(std::move(t));
  }
  if (componentId) *componentId = (int64_t)op->comps.size();
  op->comps.push_back(std::move(comp));
  op->version++;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetDestroyNetworkOperator(cutensornetNetworkOperator_t tensorNetworkOperator) {
  NetOp* op = netop_of(tensorNetworkOperator);
  if (!op) return tensorNetworkOperator ? CUTENSORNET_STATUS_INVALID_VALUE : CUTENSORNET_STATUS_SUCCESS;
  op->magic = 0;
  delete op;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetStateApplyNetworkOperator(const cutensornetHandle_t handle,
                                                         cutensornetState_t tensorNetworkState,
                                                         const cutensornetNetworkOperator_t tensorNetworkOperator,
                                                         const int32_t immutable, const int32_t adjoint,
                                                         const int32_t unitary, int64_t* operatorId) {
  const char* api = "cutensornetStateApplyNetworkOperator";
  Handle* h;
  State* s;
  if (Status st = state_args(handle, tensorNetworkState, h, s, api)) return st;
  NetOp* no = netop_of(tensorNetworkOperator);
  if (!no || !operatorId) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  if (no->n != s->n || no->ext != s->ext)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the operator's state modes differ from the state's");
  if (no->comps.size() != 1)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api,
                "cutensornetStateApplyNetworkOperator currently only supports tensor network operators with a single component!");
  const NetComponent& c = no->comps[0];
  Op op;
  op.kind = c.mpo ? OP_NETMPO : OP_NETPRODUCT;
  op.immutable = immutable != 0;
  op.adjoint = adjoint != 0;
  op.unitary = unitary != 0;
  op.fmodes = c.fmodes;
  op.fext = c.fext;
  op.tensors = c.tensors;
  add_op(*s, std::move(op), (int64_t)c.tensors.size(), operatorId);
  return CUTENSORNET_STATUS_SUCCESS;
}

}  // extern "C"

// ---- matrix product states, on the host ----
//
// The tensors of an MPS being evolved are kept as complex doubles on the
// host (column-major (left, physical, right), left bond fastest). The SVD
// of each two-site operator is cuSOLVER's, on the device, like NVIDIA's.

namespace {

struct Site {
  int64_t l = 1, d = 1, r = 1;
  std::vector<cd> a;  // a[x + l * (p + d * y)]
};
using Mps = std::vector<Site>;

struct MpsCfg {
  std::vector<int64_t> maxb;  // the largest bond extent between site i and i + 1
  double abs_cutoff = 0, rel_cutoff = 0, discarded = 0;
  int normalization = CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE;
  int algo = CUTENSORNET_TENSOR_SVD_ALGO_GESVD;
  double gesvdj_tol = 0;
  int gesvdj_sweeps = 0;
  int64_t gesvdr_oversampling = 0, gesvdr_niters = 0;
  Handle* h = nullptr;
};

// An SVD on the device: A (m x n, column-major) = U diag(S) Vh with U (m x k),
// Vh (k x n), k = min(m, n).
Status host_svd(const MpsCfg& cfg, int64_t m, int64_t n, const std::vector<cd>& A, std::vector<cd>& U,
                std::vector<double>& S, std::vector<cd>& Vh, const char* api) {
  cusolverDnHandle_t sh = solver_of(cfg.h);
  if (!sh) return fail(CUTENSORNET_STATUS_CUSOLVER_ERROR, api, "cusolverDnCreate failed");
  const bool tr = m < n;  // gesvd needs rows >= columns: factor the adjoint
  const int64_t mm = tr ? n : m, nn = tr ? m : n;
  std::vector<cd> M((size_t)(mm * nn));
  if (!tr) M = A;
  else
    for (int64_t j = 0; j < n; ++j)
      for (int64_t i = 0; i < m; ++i) M[(size_t)(j + n * i)] = std::conj(A[(size_t)(i + m * j)]);
  DevBuf dA, dS, dU, dV, dW, dInfo, dR;
  int lwork = 0;
  if (cusolverDnZgesvd_bufferSize(sh, (int)mm, (int)nn, &lwork) != CUSOLVER_STATUS_SUCCESS)
    return fail(CUTENSORNET_STATUS_CUSOLVER_ERROR, api, "cuSOLVER failed");
  if (!dA.alloc((size_t)(mm * nn) * 16) || !dS.alloc((size_t)nn * 8) || !dU.alloc((size_t)(mm * nn) * 16) ||
      !dV.alloc((size_t)(nn * nn) * 16) || !dW.alloc((size_t)std::max(lwork, 1) * 16) || !dInfo.alloc(16) ||
      !dR.alloc((size_t)nn * 8 + 16))
    return fail(CUTENSORNET_STATUS_ALLOC_FAILED, api, "no device memory for an SVD");
  if (!upload(dA.p, CUDA_C_64F, M)) return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "a copy failed");
  cudaStreamSynchronize(nullptr);
  if (cusolverDnZgesvd(sh, 'S', 'S', (int)mm, (int)nn, (cuDoubleComplex*)dA.p, (int)mm, (double*)dS.p,
                       (cuDoubleComplex*)dU.p, (int)mm, (cuDoubleComplex*)dV.p, (int)nn, (cuDoubleComplex*)dW.p, lwork,
                       (double*)dR.p, (int*)dInfo.p) != CUSOLVER_STATUS_SUCCESS)
    return fail(CUTENSORNET_STATUS_CUSOLVER_ERROR, api, "cuSOLVER failed");
  cudaStreamSynchronize(nullptr);
  std::vector<cd> hu, hv, hs;
  if (!download(dU.p, CUDA_C_64F, (size_t)(mm * nn), hu) || !download(dV.p, CUDA_C_64F, (size_t)(nn * nn), hv) ||
      !download(dS.p, CUDA_R_64F, (size_t)nn, hs))
    return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "a copy failed");
  const int64_t k = nn;
  S.resize((size_t)k);
  for (int64_t j = 0; j < k; ++j) S[(size_t)j] = hs[(size_t)j].real();
  U.assign((size_t)(m * k), cd{});
  Vh.assign((size_t)(k * n), cd{});
  if (!tr) {
    U = hu;
    for (int64_t j = 0; j < n; ++j)
      for (int64_t i = 0; i < k; ++i) Vh[(size_t)(i + k * j)] = hv[(size_t)(i + nn * j)];
  } else {
    // A^H = Ua S Va^H, so A = Va S Ua^H: U = Va (m x k), Vh = Ua^H (k x n)
    for (int64_t j = 0; j < k; ++j)
      for (int64_t i = 0; i < m; ++i) U[(size_t)(i + m * j)] = std::conj(hv[(size_t)(j + nn * i)]);
    for (int64_t j = 0; j < n; ++j)
      for (int64_t i = 0; i < k; ++i) Vh[(size_t)(i + k * j)] = std::conj(hu[(size_t)(j + mm * i)]);
  }
  return CUTENSORNET_STATUS_SUCCESS;
}

// How many singular values to keep: the bond limit and every cutoff (the
// same rules as cutensornetTensorSVD), at least one.
int64_t kept_count(const MpsCfg& cfg, const std::vector<double>& S, int64_t limit) {
  const int64_t full = (int64_t)S.size();
  int64_t keep = std::min(full, std::max<int64_t>(limit, 1));
  if (cfg.abs_cutoff > 0 || cfg.rel_cutoff > 0) {
    const double thr = std::max(cfg.abs_cutoff, cfg.rel_cutoff * (S.empty() ? 0 : S[0]));
    int64_t k = 0;
    while (k < full && S[(size_t)k] > thr) ++k;
    keep = std::min(keep, k);
  }
  if (cfg.discarded > 0) {
    double total = 0;
    for (double x : S) total += x * x;
    if (total > 0) {
      int64_t k = full;
      double tail = 0;
      while (k > 1 && (tail + S[(size_t)(k - 1)] * S[(size_t)(k - 1)]) / total <= cfg.discarded) {
        tail += S[(size_t)(k - 1)] * S[(size_t)(k - 1)];
        --k;
      }
      keep = std::min(keep, k);
    }
  }
  return std::max<int64_t>(keep, 1);
}

// Splits the (rows x cols) matrix M by SVD into U sqrt(S) (rows x k) and
// sqrt(S) Vh (k x cols), truncated to the bond limit and cutoffs, with the
// singular values normalized if asked.
Status split_matrix(const MpsCfg& cfg, int64_t rows, int64_t cols, const std::vector<cd>& M, int64_t limit,
                    std::vector<cd>& left, std::vector<cd>& right, int64_t& k, const char* api) {
  std::vector<cd> U, Vh;
  std::vector<double> S;
  if (Status st = host_svd(cfg, rows, cols, M, U, S, Vh, api)) return st;
  k = kept_count(cfg, S, limit);
  std::vector<double> kept(S.begin(), S.begin() + k);
  double norm = 0;
  switch (cfg.normalization) {
    case CUTENSORNET_TENSOR_SVD_NORMALIZATION_L1: for (double y : kept) norm += std::fabs(y); break;
    case CUTENSORNET_TENSOR_SVD_NORMALIZATION_L2: for (double y : kept) norm += y * y; norm = std::sqrt(norm); break;
    case CUTENSORNET_TENSOR_SVD_NORMALIZATION_LINF: for (double y : kept) norm = std::max(norm, std::fabs(y)); break;
    default: break;
  }
  if (norm > 0)
    for (double& y : kept) y /= norm;
  const int64_t full = (int64_t)S.size();
  left.assign((size_t)(rows * k), cd{});
  right.assign((size_t)(k * cols), cd{});
  for (int64_t j = 0; j < k; ++j) {
    const double f = std::sqrt(kept[(size_t)j]);
    for (int64_t i = 0; i < rows; ++i) left[(size_t)(i + rows * j)] = U[(size_t)(i + rows * j)] * f;
    for (int64_t c = 0; c < cols; ++c) right[(size_t)(j + k * c)] = Vh[(size_t)(j + full * c)] * f;
  }
  return CUTENSORNET_STATUS_SUCCESS;
}

// Householder QR of the (rows x cols) column-major matrix: Q (rows x q), R (q x cols), q = min.
void host_qr(int64_t rows, int64_t cols, std::vector<cd> A, std::vector<cd>& Q, std::vector<cd>& R, int64_t& q) {
  q = std::min(rows, cols);
  std::vector<std::vector<cd>> vs;
  std::vector<cd> w((size_t)rows);
  for (int64_t j = 0; j < q; ++j) {
    double nrm = 0;
    for (int64_t i = j; i < rows; ++i) nrm += std::norm(A[(size_t)(i + rows * j)]);
    nrm = std::sqrt(nrm);
    std::vector<cd> v((size_t)rows, cd{});
    const cd x0 = A[(size_t)(j + rows * j)];
    const cd phase = std::abs(x0) > 0 ? x0 / std::abs(x0) : cd(1, 0);
    for (int64_t i = j; i < rows; ++i) v[(size_t)i] = A[(size_t)(i + rows * j)];
    v[(size_t)j] += phase * nrm;
    double vn = 0;
    for (int64_t i = j; i < rows; ++i) vn += std::norm(v[(size_t)i]);
    if (vn > 0) {
      const double inv = 1.0 / std::sqrt(vn);
      for (int64_t i = j; i < rows; ++i) v[(size_t)i] *= inv;
      for (int64_t c = j; c < cols; ++c) {
        cd d = 0;
        for (int64_t i = j; i < rows; ++i) d += std::conj(v[(size_t)i]) * A[(size_t)(i + rows * c)];
        for (int64_t i = j; i < rows; ++i) A[(size_t)(i + rows * c)] -= 2.0 * v[(size_t)i] * d;
      }
    }
    vs.push_back(std::move(v));
  }
  R.assign((size_t)(q * cols), cd{});
  for (int64_t c = 0; c < cols; ++c)
    for (int64_t i = 0; i < std::min<int64_t>(c + 1, q); ++i) R[(size_t)(i + q * c)] = A[(size_t)(i + rows * c)];
  Q.assign((size_t)(rows * q), cd{});
  for (int64_t j = 0; j < q; ++j) Q[(size_t)(j + rows * j)] = 1;
  for (int64_t j = q; j-- > 0;) {
    const auto& v = vs[(size_t)j];
    for (int64_t c = 0; c < q; ++c) {
      cd d = 0;
      for (int64_t i = j; i < rows; ++i) d += std::conj(v[(size_t)i]) * Q[(size_t)(i + rows * c)];
      for (int64_t i = j; i < rows; ++i) Q[(size_t)(i + rows * c)] -= 2.0 * v[(size_t)i] * d;
    }
  }
}

// The two-site tensor of sites lo .. lo + k - 1 contracted into one:
// (l, p_0 .. p_{k-1}, r), column-major.
std::vector<cd> contract_block(const Mps& m, int64_t lo, int64_t k, int64_t& L, int64_t& D, int64_t& R) {
  std::vector<cd> t = m[(size_t)lo].a;
  L = m[(size_t)lo].l;
  D = m[(size_t)lo].d;
  int64_t cur_r = m[(size_t)lo].r;  // t is (L, D, cur_r)
  for (int64_t j = 1; j < k; ++j) {
    const Site& s = m[(size_t)(lo + j)];
    std::vector<cd> n((size_t)(L * D * s.d * s.r), cd{});
    // n[x + L*(pp + D*(q + d*y))] = sum_b t[x + L*(pp + D*b)] * s[b + s.l*(q + s.d*y)]
    for (int64_t y = 0; y < s.r; ++y)
      for (int64_t q = 0; q < s.d; ++q)
        for (int64_t b = 0; b < cur_r; ++b) {
          const cd sv = s.a[(size_t)(b + s.l * (q + s.d * y))];
          if (sv == cd{}) continue;
          for (int64_t pp = 0; pp < D; ++pp)
            for (int64_t x = 0; x < L; ++x)
              n[(size_t)(x + L * (pp + D * (q + s.d * y)))] += t[(size_t)(x + L * (pp + D * b))] * sv;
        }
    t = std::move(n);
    D *= s.d;
    cur_r = s.r;
  }
  R = cur_r;
  return t;
}

// Applies the matrix G (row-major over the block's sites, the first the most
// significant) to the block's physical legs, then splits the block back into
// k sites left to right by SVD.
Status apply_block(Mps& m, int64_t lo, int64_t k, const std::vector<cd>& G, const MpsCfg& cfg, const char* api) {
  int64_t L, D, R;
  std::vector<cd> t = contract_block(m, lo, k, L, D, R);
  // the block's physical dimensions, first site fastest in t
  std::vector<int64_t> dims;
  for (int64_t j = 0; j < k; ++j) dims.push_back(m[(size_t)(lo + j)].d);
  // t's physical index pp = sum_j p_j * prod_{i<j} d_i (site 0 fastest);
  // G's index is the reverse: site 0 most significant
  auto g_index = [&](int64_t pp) {
    int64_t idx = 0, rest = pp;
    std::vector<int64_t> digs;
    for (int64_t j = 0; j < k; ++j) {
      digs.push_back(rest % dims[(size_t)j]);
      rest /= dims[(size_t)j];
    }
    for (int64_t j = 0; j < k; ++j) idx = idx * dims[(size_t)j] + digs[(size_t)j];
    return idx;
  };
  std::vector<int64_t> map((size_t)D);
  for (int64_t pp = 0; pp < D; ++pp) map[(size_t)pp] = g_index(pp);
  std::vector<cd> u((size_t)(L * D * R), cd{});
  for (int64_t y = 0; y < R; ++y)
    for (int64_t q = 0; q < D; ++q)
      for (int64_t pp = 0; pp < D; ++pp) {
        const cd g = G[(size_t)(map[(size_t)q] * D + map[(size_t)pp])];
        if (g == cd{}) continue;
        for (int64_t x = 0; x < L; ++x) u[(size_t)(x + L * (q + D * y))] += g * t[(size_t)(x + L * (pp + D * y))];
      }
  // split left to right: at step j the tensor is (curL, p_j .. p_{k-1}, R)
  int64_t curL = L, rest_D = D;
  std::vector<cd> cur = std::move(u);
  for (int64_t j = 0; j + 1 < k; ++j) {
    const int64_t dj = dims[(size_t)j];
    rest_D /= dj;
    const int64_t rows = curL * dj, cols = rest_D * R;
    std::vector<cd> left, right;
    int64_t kk;
    const int64_t limit = cfg.maxb.empty() ? INT64_MAX : cfg.maxb[(size_t)(lo + j)];
    if (Status st = split_matrix(cfg, rows, cols, cur, limit, left, right, kk, api)) return st;
    Site& s = m[(size_t)(lo + j)];
    s.l = curL;
    s.d = dj;
    s.r = kk;
    s.a = std::move(left);
    cur = std::move(right);
    curL = kk;
  }
  Site& last = m[(size_t)(lo + k - 1)];
  last.l = curL;
  last.d = dims[(size_t)(k - 1)];
  last.r = R;
  last.a = std::move(cur);
  return CUTENSORNET_STATUS_SUCCESS;
}

std::vector<cd> swap_matrix(int64_t d) {
  std::vector<cd> g((size_t)(d * d * d * d), cd{});
  for (int64_t a = 0; a < d; ++a)
    for (int64_t b = 0; b < d; ++b) g[(size_t)((b * d + a) * d * d + (a * d + b))] = 1;
  return g;
}

// Applies the matrix G (over the listed modes, the first the most significant)
// to an MPS whose site i is state mode i. The listed modes are gathered next
// to each other by swapping, the operator is applied to the block, and the
// swaps are undone. A two-mode operator moves its first mode next to its
// second, as measured on the card.
Status apply_gate_mps(Mps& m, const State& s, const std::vector<int32_t>& modes, const std::vector<cd>& G,
                      const MpsCfg& cfg, const char* api) {
  const int64_t k = (int64_t)modes.size();
  if (k == 1) {
    Site& st = m[(size_t)modes[0]];
    const int64_t d = st.d;
    std::vector<cd> n(st.a.size(), cd{});
    for (int64_t y = 0; y < st.r; ++y)
      for (int64_t q = 0; q < d; ++q)
        for (int64_t p = 0; p < d; ++p) {
          const cd g = G[(size_t)(q * d + p)];
          if (g == cd{}) continue;
          for (int64_t x = 0; x < st.l; ++x) n[(size_t)(x + st.l * (q + d * y))] += g * st.a[(size_t)(x + st.l * (p + d * y))];
        }
    st.a = std::move(n);
    return CUTENSORNET_STATUS_SUCCESS;
  }
  // gather
  std::vector<int64_t> where((size_t)s.n);  // position of each mode
  std::vector<int64_t> at((size_t)s.n);     // the mode at each position
  for (int64_t i = 0; i < s.n; ++i) where[(size_t)i] = at[(size_t)i] = i;
  std::vector<int64_t> swaps;  // swapped positions (i, i + 1), in order
  auto do_swap = [&](int64_t i) -> Status {
    const int64_t d = m[(size_t)i].d;
    if (m[(size_t)i].d != m[(size_t)(i + 1)].d)
      return refuse(api, "an operator on non-adjacent modes of different extents cannot be applied to an MPS");
    if (Status st = apply_block(m, i, 2, swap_matrix(d), cfg, api)) return st;
    std::swap(at[(size_t)i], at[(size_t)(i + 1)]);
    where[(size_t)at[(size_t)i]] = i;
    where[(size_t)at[(size_t)(i + 1)]] = i + 1;
    swaps.push_back(i);
    return CUTENSORNET_STATUS_SUCCESS;
  };
  int64_t lo = where[(size_t)modes[(size_t)(k - 1)]], hi = lo;
  for (int64_t j = k - 2; j >= 0; --j) {
    int64_t px = where[(size_t)modes[(size_t)j]];
    if (px < lo) {
      while (px < lo - 1) {
        if (Status st = do_swap(px)) return st;
        ++px;
      }
      lo = px;
    } else if (px > hi) {
      while (px > hi + 1) {
        if (Status st = do_swap(px - 1)) return st;
        --px;
      }
      hi = px;
    }
  }
  // the block [lo, hi] holds the modes in some order: build G in the block's order
  std::vector<int64_t> bdims;
  for (int64_t p = lo; p <= hi; ++p) bdims.push_back(m[(size_t)p].d);
  int64_t D = 1;
  for (int64_t x : bdims) D *= x;
  std::vector<cd> GB((size_t)(D * D), cd{});
  // the position in the block of each listed mode
  std::vector<int64_t> pos_in_block((size_t)k);
  for (int64_t j = 0; j < k; ++j) pos_in_block[(size_t)j] = where[(size_t)modes[(size_t)j]] - lo;
  std::vector<int64_t> ldims;
  for (int64_t j = 0; j < k; ++j) ldims.push_back(s.ext[(size_t)modes[(size_t)j]]);
  auto listed_index = [&](int64_t bidx) {  // block index (first block site most significant) -> G's index
    std::vector<int64_t> digs((size_t)k);
    int64_t rest = bidx;
    for (int64_t j = k - 1; j >= 0; --j) {
      digs[(size_t)j] = rest % bdims[(size_t)j];
      rest /= bdims[(size_t)j];
    }
    int64_t idx = 0;
    for (int64_t j = 0; j < k; ++j) idx = idx * ldims[(size_t)j] + digs[(size_t)pos_in_block[(size_t)j]];
    return idx;
  };
  std::vector<int64_t> lmap((size_t)D);
  for (int64_t b = 0; b < D; ++b) lmap[(size_t)b] = listed_index(b);
  for (int64_t r = 0; r < D; ++r)
    for (int64_t c = 0; c < D; ++c) GB[(size_t)(r * D + c)] = G[(size_t)(lmap[(size_t)r] * D + lmap[(size_t)c])];
  if (Status st = apply_block(m, lo, hi - lo + 1, GB, cfg, api)) return st;
  for (size_t j = swaps.size(); j-- > 0;) {
    const int64_t i = swaps[j];
    if (Status st = apply_block(m, i, 2, swap_matrix(m[(size_t)i].d), cfg, api)) return st;
  }
  return CUTENSORNET_STATUS_SUCCESS;
}

// The squared norm of an MPS (the transfer matrices contracted).
double mps_norm2(const Mps& m) {
  std::vector<cd> E = {1};  // environment (conj bra bond, ket bond)
  int64_t eb = 1;
  for (const Site& s : m) {
    std::vector<cd> n((size_t)(s.r * s.r), cd{});
    for (int64_t p = 0; p < s.d; ++p)
      for (int64_t b2 = 0; b2 < s.r; ++b2)
        for (int64_t b1 = 0; b1 < s.r; ++b1) {
          cd acc = 0;
          for (int64_t x2 = 0; x2 < eb; ++x2)
            for (int64_t x1 = 0; x1 < eb; ++x1)
              acc += E[(size_t)(x2 + eb * x1)] * std::conj(s.a[(size_t)(x2 + s.l * (p + s.d * b2))]) *
                     s.a[(size_t)(x1 + s.l * (p + s.d * b1))];
          n[(size_t)(b2 + s.r * b1)] += acc;
        }
    E = std::move(n);
    eb = s.r;
  }
  return E.empty() ? 0 : E[0].real();
}

}  // namespace

// ---- evolving an MPS by the operators of a state ----

namespace {

// An operator as a matrix over the modes it lists, row-major, the first mode
// the most significant (the convention of a tensor operator).
Status gate_matrix(const State& s, const Op& op, std::vector<int32_t>& modes, std::vector<cd>& G, size_t tensor,
                   const char* api) {
  auto read = [&](const UserTensor& t, const std::vector<int64_t>& ext, std::vector<cd>& u) -> Status {
    if (!read_tensor(t.data, s.type, ext, t.strides, u))
      return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "an operator's tensor could not be read");
    return CUTENSORNET_STATUS_SUCCESS;
  };
  auto adjoint_of = [](const std::vector<cd>& g, int64_t D) {
    std::vector<cd> a((size_t)(D * D));
    for (int64_t r = 0; r < D; ++r)
      for (int64_t c = 0; c < D; ++c) a[(size_t)(r * D + c)] = std::conj(g[(size_t)(c * D + r)]);
    return a;
  };
  switch (op.kind) {
    case OP_GATE:
    case OP_UCHANNEL:
    case OP_GCHANNEL: {
      modes = op.modes;
      int64_t D = 1;
      for (int32_t m : modes) D *= s.ext[(size_t)m];
      auto ext = rev64(extents_of(s, modes));
      ext.insert(ext.end(), ext.begin(), ext.end());
      std::vector<cd> u;
      if (Status st = read(op.tensors[tensor], ext, u)) return st;
      G.assign((size_t)(D * D), cd{});
      for (int64_t r = 0; r < D; ++r)
        for (int64_t c = 0; c < D; ++c) G[(size_t)(r * D + c)] = u[(size_t)(c + D * r)];
      if (op.adjoint) G = adjoint_of(G, D);
      return CUTENSORNET_STATUS_SUCCESS;
    }
    case OP_DIAG: {
      modes = op.modes;
      int64_t D = 1;
      for (int32_t m : modes) D *= s.ext[(size_t)m];
      std::vector<cd> v;
      if (Status st = read(op.tensors[0], rev64(extents_of(s, modes)), v)) return st;
      G.assign((size_t)(D * D), cd{});
      for (int64_t i = 0; i < D; ++i) G[(size_t)(i * D + i)] = op.adjoint ? std::conj(v[(size_t)i]) : v[(size_t)i];
      return CUTENSORNET_STATUS_SUCCESS;
    }
    case OP_CONTROLLED: {
      modes = op.ctl_modes;
      modes.push_back(op.modes[0]);
      const int64_t d = s.ext[(size_t)op.modes[0]];
      int64_t Dc = 1;
      for (int32_t m : op.ctl_modes) Dc *= s.ext[(size_t)m];
      std::vector<cd> u;
      if (Status st = read(op.tensors[0], {d, d}, u)) return st;
      int64_t want = 0;
      for (size_t j = 0; j < op.ctl_modes.size(); ++j) want = want * s.ext[(size_t)op.ctl_modes[j]] + op.ctl_vals[j];
      const int64_t D = Dc * d;
      G.assign((size_t)(D * D), cd{});
      for (int64_t cc = 0; cc < Dc; ++cc)
        for (int64_t r = 0; r < d; ++r)
          for (int64_t c = 0; c < d; ++c) {
            cd v = cc == want ? (op.adjoint ? std::conj(u[(size_t)(r + d * c)]) : u[(size_t)(c + d * r)]) : (r == c ? 1.0 : 0.0);
            G[(size_t)((cc * d + r) * D + (cc * d + c))] = v;
          }
      return CUTENSORNET_STATUS_SUCCESS;
    }
    default: break;
  }
  return fail(CUTENSORNET_STATUS_INTERNAL_ERROR, api, "not a gate");
}

// Applies a matrix product operator to the MPS: the bonds multiply, then
// the MPS is compressed back to its bond limits by an exact left-to-right QR
// sweep and a right-to-left truncating SVD sweep.
Status apply_mpo_mps(Mps& m, const State& s, const Op& op, const MpsCfg& cfg, const char* api) {
  const size_t L = op.fmodes.size();
  std::vector<int32_t> sites;
  for (const auto& f : op.fmodes) sites.push_back(f[0]);
  bool asc = true, desc = true;
  for (size_t i = 1; i < L; ++i) {
    asc = asc && sites[i] == sites[i - 1] + 1;
    desc = desc && sites[i] == sites[i - 1] - 1;
  }
  if (!asc && !desc)
    return refuse(api, "a matrix product operator on an MPS must act on consecutive state modes");
  const int64_t lo = asc ? sites.front() : sites.back();
  // the MPO's tensors, in the order of increasing site
  std::vector<std::vector<cd>> W(L);
  std::vector<std::array<int64_t, 4>> Wd(L);  // (left, ket, right, bra) extents
  for (size_t i = 0; i < L; ++i) {
    const size_t at = asc ? i : L - 1 - i;
    const auto& e = op.fext[at];
    const bool first = at == 0, last = at == L - 1;
    size_t p = 0;
    int64_t wl = first ? 1 : e[p++], ket = e[p++], wr = last ? 1 : e[p++], bra = e[p++];
    std::vector<cd> w;
    if (!read_tensor(op.tensors[at].data, s.type, e, op.tensors[at].strides, w))
      return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "an operator's tensor could not be read");
    // w is (wl?, ket, wr?, bra) column-major: index = a + wl*(k + ket*(b + wr*q))
    if (!asc) {  // reverse the chain: the roles of the left and right bonds swap
      std::vector<cd> t(w.size());
      for (int64_t q = 0; q < bra; ++q)
        for (int64_t b = 0; b < wr; ++b)
          for (int64_t k = 0; k < ket; ++k)
            for (int64_t a = 0; a < wl; ++a) t[(size_t)(b + wr * (k + ket * (a + wl * q)))] = w[(size_t)(a + wl * (k + ket * (b + wr * q)))];
      w = std::move(t);
      std::swap(wl, wr);
    }
    if (op.adjoint) {  // the adjoint: ket and bra swap, elements conjugated
      std::vector<cd> t(w.size());
      for (int64_t q = 0; q < bra; ++q)
        for (int64_t b = 0; b < wr; ++b)
          for (int64_t k = 0; k < ket; ++k)
            for (int64_t a = 0; a < wl; ++a) t[(size_t)(a + wl * (q + bra * (b + wr * k)))] = std::conj(w[(size_t)(a + wl * (k + ket * (b + wr * q)))]);
      w = std::move(t);
      std::swap(ket, bra);
    }
    W[i] = std::move(w);
    Wd[i] = {wl, ket, wr, bra};
  }
  for (size_t i = 0; i < L; ++i) {
    Site& a = m[(size_t)(lo + (int64_t)i)];
    const int64_t wl = Wd[i][0], ket = Wd[i][1], wr = Wd[i][2], bra = Wd[i][3];
    if (a.d != ket || a.d != bra) return refuse(api, "a matrix product operator must keep the state's mode extents");
    Site n;
    n.l = a.l * wl;
    n.d = bra;
    n.r = a.r * wr;
    n.a.assign((size_t)(n.l * n.d * n.r), cd{});
    for (int64_t y = 0; y < a.r; ++y)
      for (int64_t b = 0; b < wr; ++b)
        for (int64_t q = 0; q < bra; ++q)
          for (int64_t p = 0; p < ket; ++p)
            for (int64_t x = 0; x < a.l; ++x) {
              const cd av = a.a[(size_t)(x + a.l * (p + a.d * y))];
              if (av == cd{}) continue;
              for (int64_t w0 = 0; w0 < wl; ++w0)
                n.a[(size_t)((x + a.l * w0) + n.l * (q + n.d * (y + a.r * b)))] +=
                    W[i][(size_t)(w0 + wl * (p + ket * (b + wr * q)))] * av;
            }
    a = std::move(n);
  }
  // compress the affected sites: QR left to right, truncating SVD right to left
  const int64_t hi = lo + (int64_t)L - 1;
  for (int64_t i = lo; i < hi; ++i) {
    Site& a = m[(size_t)i];
    std::vector<cd> Q, R;
    int64_t q;
    host_qr(a.l * a.d, a.r, a.a, Q, R, q);
    Site& b = m[(size_t)(i + 1)];
    std::vector<cd> nb((size_t)(q * b.d * b.r), cd{});
    for (int64_t c = 0; c < b.d * b.r; ++c)
      for (int64_t row = 0; row < q; ++row) {
        cd acc = 0;
        for (int64_t kk = 0; kk < a.r; ++kk) acc += R[(size_t)(row + q * kk)] * b.a[(size_t)(kk + b.l * c)];
        nb[(size_t)(row + q * c)] = acc;
      }
    a.a = std::move(Q);
    a.r = q;
    b.a = std::move(nb);
    b.l = q;
  }
  for (int64_t i = hi; i > lo; --i) {
    Site& b = m[(size_t)i];
    std::vector<cd> left, right;
    int64_t kk;
    const int64_t limit = cfg.maxb.empty() ? INT64_MAX : cfg.maxb[(size_t)(i - 1)];
    MpsCfg c2 = cfg;
    c2.normalization = CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE;
    // M = (l) x (d r): split so that b becomes the isometry (right factor)
    std::vector<cd> U, Vh;
    std::vector<double> S;
    if (Status st = host_svd(cfg, b.l, b.d * b.r, b.a, U, S, Vh, api)) return st;
    kk = kept_count(c2, S, limit);
    const int64_t full = (int64_t)S.size();
    std::vector<cd> nb((size_t)(kk * b.d * b.r), cd{});
    for (int64_t c = 0; c < b.d * b.r; ++c)
      for (int64_t j = 0; j < kk; ++j) nb[(size_t)(j + kk * c)] = Vh[(size_t)(j + full * c)];
    // the left site absorbs U S
    Site& a = m[(size_t)(i - 1)];
    std::vector<cd> na((size_t)(a.l * a.d * kk), cd{});
    for (int64_t y = 0; y < kk; ++y)
      for (int64_t row = 0; row < a.l * a.d; ++row) {
        cd acc = 0;
        for (int64_t x = 0; x < a.r; ++x) acc += a.a[(size_t)(row + a.l * a.d * x)] * U[(size_t)(x + b.l * y)] * S[(size_t)y];
        na[(size_t)(row + a.l * a.d * y)] = acc;
      }
    a.a = std::move(na);
    a.r = kk;
    b.a = std::move(nb);
    b.l = kk;
  }
  return CUTENSORNET_STATUS_SUCCESS;
}

// Moves the canonical center of the MPS to site c: sites before it become
// left-orthogonal (QR), sites after it right-orthogonal.
void move_center(Mps& m, int64_t c) {
  const int64_t n = (int64_t)m.size();
  for (int64_t i = 0; i < c; ++i) {
    Site& a = m[(size_t)i];
    std::vector<cd> Q, R;
    int64_t q;
    host_qr(a.l * a.d, a.r, a.a, Q, R, q);
    Site& b = m[(size_t)(i + 1)];
    std::vector<cd> nb((size_t)(q * b.d * b.r), cd{});
    for (int64_t col = 0; col < b.d * b.r; ++col)
      for (int64_t row = 0; row < q; ++row) {
        cd acc = 0;
        for (int64_t kk = 0; kk < a.r; ++kk) acc += R[(size_t)(row + q * kk)] * b.a[(size_t)(kk + b.l * col)];
        nb[(size_t)(row + q * col)] = acc;
      }
    a.a = std::move(Q);
    a.r = q;
    b.a = std::move(nb);
    b.l = q;
  }
  for (int64_t i = n - 1; i > c; --i) {
    Site& b = m[(size_t)i];
    // b (l x d r) = L Qr: QR of the conjugate transpose
    const int64_t rows = b.d * b.r, cols = b.l;
    std::vector<cd> At((size_t)(rows * cols));
    for (int64_t x = 0; x < b.l; ++x)
      for (int64_t col = 0; col < rows; ++col) At[(size_t)(col + rows * x)] = std::conj(b.a[(size_t)(x + b.l * col)]);
    std::vector<cd> Q, R;
    int64_t q;
    host_qr(rows, cols, At, Q, R, q);  // At = Q R, so b = R^H Q^H
    std::vector<cd> nb((size_t)(q * rows));
    for (int64_t j = 0; j < q; ++j)
      for (int64_t col = 0; col < rows; ++col) nb[(size_t)(j + q * col)] = std::conj(Q[(size_t)(col + rows * j)]);
    Site& a = m[(size_t)(i - 1)];
    std::vector<cd> na((size_t)(a.l * a.d * q), cd{});
    for (int64_t y = 0; y < q; ++y)
      for (int64_t row = 0; row < a.l * a.d; ++row) {
        cd acc = 0;
        for (int64_t x = 0; x < a.r; ++x) acc += a.a[(size_t)(row + a.l * a.d * x)] * std::conj(R[(size_t)(y + q * x)]);
        na[(size_t)(row + a.l * a.d * y)] = acc;
      }
    a.a = std::move(na);
    a.r = q;
    b.a = std::move(nb);
    b.l = q;
  }
}

MpsCfg mps_config(const State& s) {
  MpsCfg c;
  c.h = s.h;
  c.abs_cutoff = s.svd_abs;
  c.rel_cutoff = s.svd_rel;
  c.discarded = s.svd_discarded;
  c.normalization = s.svd_norm;
  c.algo = s.svd_algo;
  for (int i = 0; i + 1 < s.n; ++i)
    c.maxb.push_back(std::min(s.mps_max_ext[(size_t)i].back(), s.mps_max_ext[(size_t)i + 1].front()));
  return c;
}

// The MPS the state's operators produce, starting from its initial state.
Status mps_evolve(State& s, Mps& m, const char* api) {
  const MpsCfg cfg = mps_config(s);
  m.assign((size_t)s.n, Site{});
  if (s.has_init_mps) {
    for (int i = 0; i < s.n; ++i) {
      const auto& e = s.init_mps.ext[(size_t)i];
      Site& a = m[(size_t)i];
      a.l = i == 0 ? 1 : e[0];
      a.d = e[i == 0 ? 0 : 1];
      a.r = i == s.n - 1 ? 1 : e.back();
      if (!read_tensor(s.init_mps.ptr[(size_t)i], s.type, e, s.init_mps.str[(size_t)i], a.a))
        return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "an initial MPS tensor could not be read");
    }
  } else {
    for (int i = 0; i < s.n; ++i) {
      Site& a = m[(size_t)i];
      a.d = s.ext[(size_t)i];
      a.a.assign((size_t)a.d, cd{});
      a.a[0] = 1;
    }
  }
  for (const Op& op : s.ops) {
    switch (op.kind) {
      case OP_NETPRODUCT: {
        for (size_t f = 0; f < op.fmodes.size(); ++f) {
          // a product factor stores its modes in forward order, q_0 the fastest
          // index: the matrix is over the reversed list, q_{k-1} the most significant
          std::vector<int32_t> modes = rev32(op.fmodes[f]);
          std::vector<cd> G;
          const auto ext = extents_of(s, op.fmodes[f]);
          std::vector<int64_t> e2 = ext;
          e2.insert(e2.end(), ext.begin(), ext.end());
          int64_t D = 1;
          for (int64_t x : ext) D *= x;
          std::vector<cd> u;
          if (!read_tensor(op.tensors[f].data, s.type, e2, op.tensors[f].strides, u))
            return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "an operator's tensor could not be read");
          G.assign((size_t)(D * D), cd{});
          for (int64_t r = 0; r < D; ++r)
            for (int64_t c = 0; c < D; ++c) G[(size_t)(r * D + c)] = u[(size_t)(c + D * r)];
          if (op.adjoint)
            for (int64_t r = 0; r < D; ++r)
              for (int64_t c = r; c < D; ++c) {
                const cd a = std::conj(G[(size_t)(c * D + r)]), b = std::conj(G[(size_t)(r * D + c)]);
                G[(size_t)(r * D + c)] = a;
                G[(size_t)(c * D + r)] = b;
              }
          if (Status st = apply_gate_mps(m, s, modes, G, cfg, api)) return st;
        }
        break;
      }
      case OP_NETMPO:
        if (Status st = apply_mpo_mps(m, s, op, cfg, api)) return st;
        break;
      case OP_GCHANNEL: {
        const double n0 = mps_norm2(m);
        std::vector<Mps> branch;
        std::vector<double> w;
        for (size_t k = 0; k < op.tensors.size(); ++k) {
          Mps b = m;
          std::vector<int32_t> modes;
          std::vector<cd> G;
          if (Status st = gate_matrix(s, op, modes, G, k, api)) return st;
          if (Status st = apply_gate_mps(b, s, modes, G, cfg, api)) return st;
          w.push_back(mps_norm2(b));
          branch.push_back(std::move(b));
        }
        double tot = 0;
        for (double x : w) tot += x;
        if (!(tot > 0)) return fail(CUTENSORNET_STATUS_INTERNAL_ERROR, api, "a channel annihilated the state");
        std::uniform_real_distribution<double> u(0.0, tot);
        double r = u(s.rng), acc = 0;
        size_t pick = w.size() - 1;
        for (size_t k = 0; k < w.size(); ++k) {
          acc += w[k];
          if (r < acc) {
            pick = k;
            break;
          }
        }
        m = std::move(branch[pick]);
        const double f = std::sqrt(n0 / w[pick]);
        for (cd& x : m[0].a) x *= f;
        break;
      }
      case OP_UCHANNEL: {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double r = u(s.rng), acc = 0;
        size_t pick = op.probs.size() - 1;
        for (size_t k = 0; k < op.probs.size(); ++k) {
          acc += op.probs[k];
          if (r < acc) {
            pick = k;
            break;
          }
        }
        std::vector<int32_t> modes;
        std::vector<cd> G;
        if (Status st = gate_matrix(s, op, modes, G, pick, api)) return st;
        if (Status st = apply_gate_mps(m, s, modes, G, cfg, api)) return st;
        break;
      }
      default: {
        std::vector<int32_t> modes;
        std::vector<cd> G;
        if (Status st = gate_matrix(s, op, modes, G, 0, api)) return st;
        if (Status st = apply_gate_mps(m, s, modes, G, cfg, api)) return st;
        break;
      }
    }
  }
  if (s.canonical_center >= 0) move_center(m, s.canonical_center);
  return CUTENSORNET_STATUS_SUCCESS;
}

// The extents of a site tensor as the caller sees them (no bond at an end).
std::vector<int64_t> site_extents(const Mps& m, int i) {
  const Site& a = m[(size_t)i];
  const int n = (int)m.size();
  std::vector<int64_t> e;
  if (i > 0) e.push_back(a.l);
  e.push_back(a.d);
  if (i < n - 1) e.push_back(a.r);
  return e;
}

}  // namespace

// ---- the API: configuring, preparing and computing a state ----

namespace {

// The deprecated attribute numbers name the same attributes as the new ones.
int canonical_state_attr(int a) {
  switch (a) {
    case CUTENSORNET_STATE_MPS_CANONICAL_CENTER: return CUTENSORNET_STATE_CONFIG_MPS_CANONICAL_CENTER;
    case CUTENSORNET_STATE_MPS_SVD_CONFIG_ABS_CUTOFF: return CUTENSORNET_STATE_CONFIG_MPS_SVD_ABS_CUTOFF;
    case CUTENSORNET_STATE_MPS_SVD_CONFIG_REL_CUTOFF: return CUTENSORNET_STATE_CONFIG_MPS_SVD_REL_CUTOFF;
    case CUTENSORNET_STATE_MPS_SVD_CONFIG_S_NORMALIZATION: return CUTENSORNET_STATE_CONFIG_MPS_SVD_S_NORMALIZATION;
    case CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO: return CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO;
    case CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO_PARAMS: return CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO_PARAMS;
    case CUTENSORNET_STATE_MPS_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF: return CUTENSORNET_STATE_CONFIG_MPS_SVD_DISCARDED_WEIGHT_CUTOFF;
    case CUTENSORNET_STATE_NUM_HYPER_SAMPLES: return CUTENSORNET_STATE_CONFIG_NUM_HYPER_SAMPLES;
    default: return a;
  }
}

std::string attr_name(int a) { return "attribute " + std::to_string(a); }

}  // namespace

extern "C" {

cutensornetStatus_t cutensornetStateConfigure(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState,
                                              cutensornetStateAttributes_t attribute, const void* attributeValue,
                                              size_t attributeSize) {
  const char* api = "cutensornetStateConfigure";
  Handle* h;
  State* s;
  if (Status st = state_args(handle, tensorNetworkState, h, s, api)) return st;
  const int a = canonical_state_attr(enum_value(attribute));
  auto size_ok = [&](size_t want) -> Status {
    if (attributeSize != want)
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "Unexpected attribute size for " + attr_name(a) + "!");
    return CUTENSORNET_STATUS_SUCCESS;
  };
  auto unexpected = [&]() { return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "Unexpected value for " + attr_name(a) + "!"); };
  const bool known = a == CUTENSORNET_STATE_CONFIG_MPS_CANONICAL_CENTER || a == CUTENSORNET_STATE_CONFIG_MPS_SVD_ABS_CUTOFF ||
                     a == CUTENSORNET_STATE_CONFIG_MPS_SVD_REL_CUTOFF || a == CUTENSORNET_STATE_CONFIG_MPS_SVD_S_NORMALIZATION ||
                     a == CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO || a == CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO_PARAMS ||
                     a == CUTENSORNET_STATE_CONFIG_MPS_SVD_DISCARDED_WEIGHT_CUTOFF ||
                     a == CUTENSORNET_STATE_CONFIG_MPS_MPO_APPLICATION || a == CUTENSORNET_STATE_CONFIG_MPS_GAUGE_OPTION ||
                     a == CUTENSORNET_STATE_CONFIG_NUM_HYPER_SAMPLES;
  if (!known) return fail(CUTENSORNET_STATUS_NOT_SUPPORTED, api, "Not Supported: Attribute not supported!");
  if (!attributeValue) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "a null pointer was passed");
  auto get_i32 = [&](int32_t& v) -> Status {
    if (Status st = size_ok(4)) return st;
    std::memcpy(&v, attributeValue, 4);
    return CUTENSORNET_STATUS_SUCCESS;
  };
  auto get_f64 = [&](double& v) -> Status {
    if (Status st = size_ok(8)) return st;
    std::memcpy(&v, attributeValue, 8);
    if (v < 0) return unexpected();
    return CUTENSORNET_STATUS_SUCCESS;
  };
  int32_t v32 = 0;
  switch (a) {
    case CUTENSORNET_STATE_CONFIG_NUM_HYPER_SAMPLES:
      if (Status st = get_i32(s->num_hyper)) return st;
      return CUTENSORNET_STATUS_SUCCESS;
    case CUTENSORNET_STATE_CONFIG_MPS_CANONICAL_CENTER:
      if (Status st = get_i32(v32)) return st;
      if (v32 < -1 || v32 >= s->n) return unexpected();
      s->canonical_center = v32;
      return CUTENSORNET_STATUS_SUCCESS;
    case CUTENSORNET_STATE_CONFIG_MPS_SVD_ABS_CUTOFF: return get_f64(s->svd_abs);
    case CUTENSORNET_STATE_CONFIG_MPS_SVD_REL_CUTOFF: return get_f64(s->svd_rel);
    case CUTENSORNET_STATE_CONFIG_MPS_SVD_DISCARDED_WEIGHT_CUTOFF: return get_f64(s->svd_discarded);
    case CUTENSORNET_STATE_CONFIG_MPS_SVD_S_NORMALIZATION:
      if (Status st = get_i32(v32)) return st;
      if (v32 < 0 || v32 > 3) return unexpected();
      s->svd_norm = v32;
      return CUTENSORNET_STATUS_SUCCESS;
    case CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO:
      if (Status st = get_i32(v32)) return st;
      if (v32 < 0 || v32 > 3) return unexpected();
      s->svd_algo = v32;
      return CUTENSORNET_STATUS_SUCCESS;
    case CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO_PARAMS:
      // only the algorithms that take parameters accept them (both structs are 16 bytes)
      if (s->svd_algo != CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ && s->svd_algo != CUTENSORNET_TENSOR_SVD_ALGO_GESVDR)
        return fail(CUTENSORNET_STATUS_NOT_SUPPORTED, api, "Not Supported: Attribute not supported for " + attr_name(a) + "!");
      if (Status st = size_ok(16)) return st;
      std::memcpy(s->algo_params, attributeValue, 16);
      return CUTENSORNET_STATUS_SUCCESS;
    case CUTENSORNET_STATE_CONFIG_MPS_MPO_APPLICATION:
      if (Status st = get_i32(v32)) return st;
      if (v32 < 0 || v32 > 1) return unexpected();
      s->mpo_application = v32;
      return CUTENSORNET_STATUS_SUCCESS;
    default:  // the gauge
      if (Status st = get_i32(v32)) return st;
      if (v32 < 0 || v32 > 1) return unexpected();
      s->gauge = v32;
      return CUTENSORNET_STATUS_SUCCESS;
  }
}

cutensornetStatus_t cutensornetStateGetInfo(const cutensornetHandle_t handle,
                                            const cutensornetState_t tensorNetworkState,
                                            cutensornetStateAttributes_t attribute, void* attributeValue,
                                            size_t attributeSize) {
  const char* api = "cutensornetStateGetInfo";
  Handle* h;
  State* s;
  if (Status st = state_args(handle, const_cast<cutensornetState_t>(tensorNetworkState), h, s, api)) return st;
  const int a = canonical_state_attr(enum_value(attribute));
  switch (a) {
    case CUTENSORNET_STATE_CONFIG_NUM_HYPER_SAMPLES: return get_info_value(&s->num_hyper, 4, attributeValue, attributeSize, api);
    case CUTENSORNET_STATE_CONFIG_MPS_CANONICAL_CENTER: return get_info_value(&s->canonical_center, 4, attributeValue, attributeSize, api);
    case CUTENSORNET_STATE_CONFIG_MPS_GAUGE_OPTION: return get_info_value(&s->gauge, 4, attributeValue, attributeSize, api);
    case CUTENSORNET_STATE_INFO_FLOPS: {
      const double f = s->prepared && s->prepared_epoch == s->epoch ? s->flops : 0.0;
      return get_info_value(&f, 8, attributeValue, attributeSize, api);
    }
    default: return fail(CUTENSORNET_STATUS_NOT_SUPPORTED, api, "Not Supported: Attribute not supported!");
  }
}

cutensornetStatus_t cutensornetStatePrepare(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState,
                                            size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc,
                                            cudaStream_t cudaStream) {
  const char* api = "cutensornetStatePrepare";
  (void)cudaStream;
  Handle* h;
  State* s;
  if (Status st = state_args(handle, tensorNetworkState, h, s, api)) return st;
  Workspace* w = work_of(workDesc);
  if (!w) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid workspace descriptor");
  if (s->mps_finalized) {
    // an MPS is evolved by operator and SVD: the library's own device temporaries, a
    // token amount of scratch (NVIDIA's asks for hundreds of megabytes)
    const int64_t need = kAlign;
    if (Status st = prepare_workspace(w, need, maxWorkspaceSizeDevice, api)) return st;
    s->need = need;
    s->flops = 0;
  } else {
    if (Status st = check_channels(*s, api)) return st;
    NetBuild b;
    Circ c;
    c.choice.assign(s->ops.size(), 0);
    if (Status st = build_dense_state(*s, true, b, c, api)) return st;
    Part part;
    if (Status st = plan_part(h, b, maxWorkspaceSizeDevice, part, api)) return st;
    if (Status st = prepare_workspace(w, part.need, maxWorkspaceSizeDevice, api)) return st;
    s->path = part.path;
    s->need = part.need;
    s->flops = part.flops;
  }
  s->prepared = true;
  s->prepared_epoch = s->epoch;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetStateCompute(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState,
                                            cutensornetWorkspaceDescriptor_t workDesc, int64_t* extentsOut[],
                                            int64_t* stridesOut[], void* stateTensorsOut[], cudaStream_t cudaStream) {
  const char* api = "cutensornetStateCompute";
  Handle* h;
  State* s;
  if (Status st = state_args(handle, tensorNetworkState, h, s, api)) return st;
  Workspace* w = work_of(workDesc);
  if (!w || !stateTensorsOut) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  const int ntensors = s->mps_finalized ? s->n : 1;
  for (int i = 0; i < ntensors; ++i)
    if (!stateTensorsOut[i]) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "an output tensor has no memory");
  // measured: unlike the properties, a state that is not prepared is refused before its scratch
  if (!s->prepared || s->prepared_epoch != s->epoch)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api,
                s->mps_finalized ? "MPS factorization is outdated due to new changes to the tensor circuit!"
                                 : "The tensor network state is outdated due to new changes to the tensor circuit!");
  if (Status st = check_scratch(w, s->need, api)) return st;
  Status cap;
  if (graph_capture_refused(cudaStream, api, cap)) return cap;
  cudaStreamSynchronize(cudaStream);
  if (s->mps_finalized) {
    Mps m;
    if (Status st = mps_evolve(*s, m, api)) return st;
    MpsTensors out;
    for (int i = 0; i < s->n; ++i) {
      const auto e = site_extents(m, i);
      std::vector<int64_t> str = s->mps_out_str[(size_t)i].empty() ? packed_strides(e) : s->mps_out_str[(size_t)i];
      if (!write_tensor(stateTensorsOut[i], s->type, e, str, m[(size_t)i].a))
        return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "an output tensor could not be written");
      if (extentsOut && extentsOut[i]) std::copy(e.begin(), e.end(), extentsOut[i]);
      if (stridesOut && stridesOut[i]) std::copy(str.begin(), str.end(), stridesOut[i]);
      out.ext.push_back(e);
      out.str.push_back(str);
      out.ptr.push_back(stateTensorsOut[i]);
    }
    s->mps = std::move(out);
    s->mps_computed = true;
    s->mps_held = true;
    s->mps_epoch = s->epoch;
    return CUTENSORNET_STATUS_SUCCESS;
  }
  NetBuild b;
  Circ c;
  resolve_channels(*s, c, s->rng, true);
  if (Status st = build_dense_state(*s, false, b, c, api)) return st;
  Part part;
  part.path = s->path;
  if (Status st = run_part(h, b, part, stateTensorsOut[0], w, cudaStream, api)) return st;
  cudaStreamSynchronize(cudaStream);
  const auto& oe = b.net.out.extents;
  if (extentsOut && extentsOut[0]) std::copy(oe.begin(), oe.end(), extentsOut[0]);
  if (stridesOut && stridesOut[0]) {
    const auto ps = packed_strides(oe);
    std::copy(ps.begin(), ps.end(), stridesOut[0]);
  }
  s->dense_computed = true;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetGetOutputStateDetails(const cutensornetHandle_t handle,
                                                     const cutensornetState_t tensorNetworkState,
                                                     int32_t* numTensorsOut, int32_t numModesOut[],
                                                     int64_t* extentsOut[], int64_t* stridesOut[]) {
  const char* api = "cutensornetGetOutputStateDetails";
  Handle* h;
  State* s;
  if (Status st = state_args(handle, const_cast<cutensornetState_t>(tensorNetworkState), h, s, api)) return st;
  if (!numTensorsOut) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "numTensorsOut must not be null");
  const int nt = s->mps_finalized ? s->n : 1;
  *numTensorsOut = nt;
  if (numModesOut)
    for (int i = 0; i < nt; ++i)
      numModesOut[i] = s->mps_finalized ? (i == 0 || i == s->n - 1 ? 2 : 3) : (s->mixed ? 2 * s->n : s->n);
  if (extentsOut || stridesOut) {
    // the extents and strides exist once the state has been computed
    const bool done = s->mps_finalized ? s->mps_computed : s->dense_computed;
    if (!done) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the state has not been computed");
    for (int i = 0; i < nt; ++i) {
      std::vector<int64_t> e, str;
      if (s->mps_finalized) {
        e = s->mps.ext[(size_t)i];
        str = s->mps.str[(size_t)i];
      } else {
        e = s->ext;
        if (s->mixed) e.insert(e.end(), s->ext.begin(), s->ext.end());
        str = packed_strides(e);
      }
      if (extentsOut && extentsOut[i]) std::copy(e.begin(), e.end(), extentsOut[i]);
      if (stridesOut && stridesOut[i]) std::copy(str.begin(), str.end(), stridesOut[i]);
    }
  }
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetStateFinalizeMPS(const cutensornetHandle_t handle,
                                                cutensornetState_t tensorNetworkState,
                                                cutensornetBoundaryCondition_t boundaryCondition,
                                                const int64_t* const extentsOut[], const int64_t* const stridesOut[]) {
  const char* api = "cutensornetStateFinalizeMPS";
  Handle* h;
  State* s;
  if (Status st = state_args(handle, tensorNetworkState, h, s, api)) return st;
  if (!extentsOut || enum_value(boundaryCondition) != 0)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  if (s->mixed) return fail(CUTENSORNET_STATUS_NOT_SUPPORTED, api, "MPS factorization is supported for pure states only");
  if (s->n < 2) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "an MPS needs at least two state modes");
  std::vector<std::vector<int64_t>> ext((size_t)s->n), str((size_t)s->n);
  for (int i = 0; i < s->n; ++i) {
    if (!extentsOut[i]) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid extents");
    const int rank = (i == 0 || i == s->n - 1) ? 2 : 3;
    ext[(size_t)i].assign(extentsOut[i], extentsOut[i] + rank);
    const int64_t phys = ext[(size_t)i][i == 0 ? 0 : 1];
    if (phys != s->ext[(size_t)i])
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "an MPS tensor's physical extent differs from the state mode's");
    for (int64_t x : ext[(size_t)i])
      if (x < 1) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "extents must be positive");
    if (stridesOut && stridesOut[i]) str[(size_t)i].assign(stridesOut[i], stridesOut[i] + rank);
  }
  s->mps_finalized = true;
  s->mps_max_ext = std::move(ext);
  s->mps_out_str = std::move(str);
  s->mps_computed = false;
  s->mps_held = false;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetStateInitializeMPS(const cutensornetHandle_t handle,
                                                  cutensornetState_t tensorNetworkState,
                                                  cutensornetBoundaryCondition_t boundaryCondition,
                                                  const int64_t* const extentsIn[], const int64_t* const stridesIn[],
                                                  void* stateTensorsIn[]) {
  const char* api = "cutensornetStateInitializeMPS";
  Handle* h;
  State* s;
  if (Status st = state_args(handle, tensorNetworkState, h, s, api)) return st;
  if (!extentsIn || enum_value(boundaryCondition) != 0)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  if (s->mixed) return fail(CUTENSORNET_STATUS_NOT_SUPPORTED, api, "MPS factorization is supported for pure states only");
  if (s->n < 2) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "an MPS needs at least two state modes");
  MpsTensors m;
  for (int i = 0; i < s->n; ++i) {
    if (!extentsIn[i]) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid extents");
    const int rank = (i == 0 || i == s->n - 1) ? 2 : 3;
    std::vector<int64_t> e(extentsIn[i], extentsIn[i] + rank);
    if (e[i == 0 ? 0 : 1] != s->ext[(size_t)i])
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "an MPS tensor's physical extent differs from the state mode's");
    for (int64_t x : e)
      if (x < 1) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "extents must be positive");
    m.ext.push_back(e);
    m.str.push_back(stridesIn && stridesIn[i] ? std::vector<int64_t>(stridesIn[i], stridesIn[i] + rank) : std::vector<int64_t>());
    m.ptr.push_back(stateTensorsIn ? stateTensorsIn[i] : nullptr);
  }
  if (stateTensorsIn) {
    for (void* p : m.ptr)
      if (!p) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "an initial MPS tensor has no memory");
    s->has_init_mps = true;
    s->init_mps = std::move(m);
  } else {
    s->has_init_mps = false;  // the vacuum
  }
  s->next_id += s->n + 1;
  s->epoch++;
  s->prepared = false;
  s->mps_computed = false;
  s->mps_held = false;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetStateCaptureMPS(const cutensornetHandle_t handle,
                                               cutensornetState_t tensorNetworkState) {
  const char* api = "cutensornetStateCaptureMPS";
  Handle* h;
  State* s;
  if (Status st = state_args(handle, tensorNetworkState, h, s, api)) return st;
  if (!s->mps_finalized || !s->mps_computed)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "no MPS has been computed");
  s->init_mps = s->mps;
  s->has_init_mps = true;
  s->ops.clear();
  s->epoch++;
  s->prepared = false;
  s->mps_computed = true;
  s->mps_held = true;
  return CUTENSORNET_STATUS_SUCCESS;
}

}  // extern "C"

// ---- the API: amplitudes ----

namespace {

Accessor* accessor_of(void* p) { return as<Accessor>(p, kMagicAccessor); }
Marginal* marginal_of(void* p) { return as<Marginal>(p, kMagicMarginal); }
Sampler* sampler_of(void* p) { return as<Sampler>(p, kMagicSampler); }
Expectation* expectation_of(void* p) { return as<Expectation>(p, kMagicExpectation); }

Status make_onehots(const State& s, const std::vector<int32_t>& modes, std::vector<std::unique_ptr<DevBuf>>& out,
                    const char* api) {
  for (int32_t m : modes) {
    const int64_t d = s.ext[(size_t)(m < s.n ? m : m - s.n)];
    auto b = std::make_unique<DevBuf>();
    if (!b->alloc((size_t)d * elem_bytes(s.type)))
      return fail(CUTENSORNET_STATUS_ALLOC_FAILED, api, "no device memory");
    out.push_back(std::move(b));
  }
  return CUTENSORNET_STATUS_SUCCESS;
}

Status set_onehot(const State& s, DevBuf& buf, int64_t d, int64_t at, const char* api) {
  if (!upload(buf.p, s.type, onehot_values(d, at))) return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "a copy failed");
  return CUTENSORNET_STATUS_SUCCESS;
}

// Mode lists given by the caller: counts, ranges and repeats.
Status check_mode_list(int32_t count, const int32_t* modes, int32_t range, bool allow_null_when_all,
                       std::vector<int32_t>& out, const char* api) {
  if (count < 0 || count > range) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid number of modes");
  if (count > 0 && !modes) {
    if (!(allow_null_when_all && count == range)) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "no modes were given");
    for (int32_t i = 0; i < range; ++i) out.push_back(i);
    return CUTENSORNET_STATUS_SUCCESS;
  }
  std::set<int32_t> seen;
  for (int32_t i = 0; i < count; ++i) {
    if (modes[i] < 0 || modes[i] >= range)
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "State mode id exceeds the total number of state modes!");
    if (!seen.insert(modes[i]).second) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "State modes must not repeat!");
    out.push_back(modes[i]);
  }
  return CUTENSORNET_STATUS_SUCCESS;
}

Status check_values(const State& s, const std::vector<int32_t>& modes, const int64_t* values, const char* api) {
  if (!modes.empty() && !values) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "no projected mode values were given");
  for (size_t j = 0; j < modes.size(); ++j) {
    const int32_t m = modes[j];
    const int64_t d = s.ext[(size_t)(m < s.n ? m : m - s.n)];
    if (values[j] < 0 || values[j] >= d)
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "Projected mode value is out of the mode extent!");
  }
  return CUTENSORNET_STATUS_SUCCESS;
}

// The part of an accessor's compute after its tensors are in order.
Status check_epoch_prepared(const State& s, uint64_t epoch, bool prepared, const char* what, const char* api) {
  if (s.epoch != epoch) return outdated(what, api);
  if (!prepared) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, std::string(what) + " has not been prepared");
  return CUTENSORNET_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

cutensornetStatus_t cutensornetCreateAccessor(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState,
                                              int32_t numProjectedModes, const int32_t* projectedModes,
                                              const int64_t* amplitudesTensorStrides,
                                              cutensornetStateAccessor_t* tensorNetworkAccessor) {
  const char* api = "cutensornetCreateAccessor";
  Handle* h;
  State* s;
  if (Status st = state_args(handle, tensorNetworkState, h, s, api)) return st;
  if (!tensorNetworkAccessor) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "a null pointer was passed");
  if (awaits_mps(*s)) return fail(CUTENSORNET_STATUS_INTERNAL_ERROR, api, "cuTensorNet internal error");
  auto a = std::make_unique<Accessor>();
  const int total = s->mixed ? 2 * s->n : s->n;
  if (Status st = check_mode_list(numProjectedModes, projectedModes, total, true, a->proj, api)) return st;
  a->s = s;
  a->epoch = s->epoch;
  const int rank = total - (int)a->proj.size();
  a->strides = copy_strides(amplitudesTensorStrides, (size_t)rank);
  if (Status st = make_onehots(*s, a->proj, a->onehot, api)) return st;
  *tensorNetworkAccessor = a.release();
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetAccessorConfigure(const cutensornetHandle_t handle,
                                                 cutensornetStateAccessor_t tensorNetworkAccessor,
                                                 cutensornetAccessorAttributes_t attribute, const void* attributeValue,
                                                 size_t attributeSize) {
  const char* api = "cutensornetAccessorConfigure";
  Accessor* a = accessor_of(tensorNetworkAccessor);
  if (!handle_of(handle) || !a) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid handle or accessor");
  return config_hyper(a->hyper, enum_value(attribute), CUTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES,
                      CUTENSORNET_ACCESSOR_OPT_NUM_HYPER_SAMPLES, attributeValue, attributeSize, api);
}

cutensornetStatus_t cutensornetAccessorPrepare(const cutensornetHandle_t handle,
                                               cutensornetStateAccessor_t tensorNetworkAccessor,
                                               size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc,
                                               cudaStream_t cudaStream) {
  const char* api = "cutensornetAccessorPrepare";
  (void)cudaStream;
  Accessor* a = accessor_of(tensorNetworkAccessor);
  if (!handle_of(handle) || !a) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid handle or accessor");
  Workspace* w = work_of(workDesc);
  if (!w) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid workspace descriptor");
  if (a->s->epoch != a->epoch) return outdated("Amplitudes Accessor", api);
  if (Status st = check_channels(*a->s, api)) return st;
  const uint64_t limit = maxWorkspaceSizeDevice > (uint64_t)kExtra ? maxWorkspaceSizeDevice - kExtra : 0;
  NetBuild b1, b2;
  Circ c1, c2;
  c1.choice.assign(a->s->ops.size(), 0);
  c2.choice = c1.choice;
  if (Status st = build_amplitudes(*a->s, *a, true, b1, c1, api)) return st;
  if (Status st = plan_part(handle_of(handle), b1, limit, a->amp, api)) return st;
  if (Status st = build_norm(*a->s, true, b2, c2, api)) return st;
  if (Status st = plan_part(handle_of(handle), b2, limit, a->norm, api)) return st;
  const int64_t need = std::max(a->amp.need, a->norm.need) + kExtra;
  if (Status st = prepare_workspace(w, need, maxWorkspaceSizeDevice, api)) return st;
  a->need = need;
  a->flops = a->amp.flops;
  a->prepared = true;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetAccessorGetInfo(const cutensornetHandle_t handle,
                                               const cutensornetStateAccessor_t tensorNetworkAccessor,
                                               cutensornetAccessorAttributes_t attribute, void* attributeValue,
                                               size_t attributeSize) {
  const char* api = "cutensornetAccessorGetInfo";
  Accessor* a = accessor_of(const_cast<cutensornetStateAccessor_t>(tensorNetworkAccessor));
  if (!handle_of(handle) || !a) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid handle or accessor");
  if (a->s->epoch != a->epoch) return outdated("Amplitudes Accessor", api);
  const int at = enum_value(attribute);
  if (at == CUTENSORNET_ACCESSOR_INFO_FLOPS) {
    const double f = a->prepared ? a->flops : 0.0;
    return get_info_value(&f, 8, attributeValue, attributeSize, api);
  }
  if (at == CUTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES || at == CUTENSORNET_ACCESSOR_OPT_NUM_HYPER_SAMPLES)
    return get_info_value(&a->hyper, 4, attributeValue, attributeSize, api);
  return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "Invalid attribute!");
}

cutensornetStatus_t cutensornetAccessorCompute(const cutensornetHandle_t handle,
                                               cutensornetStateAccessor_t tensorNetworkAccessor,
                                               const int64_t* projectedModeValues,
                                               cutensornetWorkspaceDescriptor_t workDesc, void* amplitudesTensor,
                                               void* stateNorm, cudaStream_t cudaStream) {
  const char* api = "cutensornetAccessorCompute";
  Handle* h = handle_of(handle);
  Accessor* a = accessor_of(tensorNetworkAccessor);
  Workspace* w = work_of(workDesc);
  if (!h || !a || !w || !amplitudesTensor) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  State& s = *a->s;
  // RTX 3060, cuTensorNet 2.14: a call with a projected value out of range leaves the accessor
  // refusing every call after it (INVALID_VALUE, valid arguments included); a NULL list of
  // values, a NULL output or a NULL workspace do not.
  if (a->broken) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "an earlier call gave a projected value out of range");
  if (Status st = check_values(s, a->proj, projectedModeValues, api)) {
    if (projectedModeValues) a->broken = true;
    return st;
  }
  if (Status st = check_scratch(w, 0, api)) return st;
  if (Status st = check_epoch_prepared(s, a->epoch, a->prepared, "Amplitudes Accessor", api)) return st;
  if (Status st = check_scratch(w, a->need, api)) return st;
  Status cap;
  if (graph_capture_refused(cudaStream, api, cap)) return cap;
  cudaStreamSynchronize(cudaStream);
  for (size_t j = 0; j < a->proj.size(); ++j) {
    const int32_t m = a->proj[j];
    if (Status st = set_onehot(s, *a->onehot[j], s.ext[(size_t)(m < s.n ? m : m - s.n)], projectedModeValues[j], api)) return st;
  }
  ScratchSplit sp = split_scratch(w, kExtra);
  NetBuild b;
  Circ c;
  resolve_channels(s, c, s.rng, true);
  if (Status st = build_amplitudes(s, *a, false, b, c, api)) return st;
  if (Status st = run_part(h, b, a->amp, amplitudesTensor, &sp.rest, cudaStream, api, Dist::Sum)) return st;
  if (stateNorm) {
    NetBuild nb;
    Circ nc = c;
    if (Status st = build_norm(s, false, nb, nc, api)) return st;
    if (Status st = run_part(h, nb, a->norm, sp.scalars, &sp.rest, cudaStream, api, Dist::Sum)) return st;
    cd v;
    if (!fetch_scalar(sp.scalars, s.type, cudaStream, v)) return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "a copy failed");
    put_scalar(stateNorm, s.type, v);
  }
  cudaStreamSynchronize(cudaStream);
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetDestroyAccessor(cutensornetStateAccessor_t tensorNetworkAccessor) {
  Accessor* a = accessor_of(tensorNetworkAccessor);
  if (!a) return tensorNetworkAccessor ? CUTENSORNET_STATUS_INVALID_VALUE : CUTENSORNET_STATUS_SUCCESS;
  a->magic = 0;
  delete a;
  return CUTENSORNET_STATUS_SUCCESS;
}

// ---- marginals ----

}  // extern "C"

namespace {

Status create_marginal(const cutensornetHandle_t handle, cutensornetState_t state, bool diagonal,
                       int32_t numMarginalModes, const int32_t* marginalModes, int32_t numProjectedModes,
                       const int32_t* projectedModes, const int64_t* strides, cutensornetStateMarginal_t* out,
                       const char* api) {
  Handle* h;
  State* s;
  if (Status st = state_args(handle, state, h, s, api)) return st;
  if (!out) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "a null pointer was passed");
  if (awaits_mps(*s)) return fail(CUTENSORNET_STATUS_INTERNAL_ERROR, api, "cuTensorNet internal error");
  if (numMarginalModes < 1) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "a marginal needs open modes");
  auto m = std::make_unique<Marginal>();
  m->diagonal = diagonal;
  if (Status st = check_mode_list(numMarginalModes, marginalModes, s->n, false, m->open, api)) return st;
  if (Status st = check_mode_list(numProjectedModes, projectedModes, s->n, false, m->proj, api)) return st;
  for (int32_t q : m->proj)
    if (std::find(m->open.begin(), m->open.end(), q) != m->open.end())
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "a mode is both open and projected");
  m->s = s;
  m->epoch = s->epoch;
  m->strides = copy_strides(strides, (diagonal ? 1 : 2) * m->open.size());
  if (Status st = make_onehots(*s, m->proj, m->onehot, api)) return st;
  *out = m.release();
  return CUTENSORNET_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

cutensornetStatus_t cutensornetCreateMarginal(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState,
                                              int32_t numMarginalModes, const int32_t* marginalModes,
                                              int32_t numProjectedModes, const int32_t* projectedModes,
                                              const int64_t* marginalTensorStrides,
                                              cutensornetStateMarginal_t* tensorNetworkMarginal) {
  return create_marginal(handle, tensorNetworkState, false, numMarginalModes, marginalModes, numProjectedModes,
                         projectedModes, marginalTensorStrides, tensorNetworkMarginal, "cutensornetCreateMarginal");
}

cutensornetStatus_t cutensornetCreateMarginalDiagonal(const cutensornetHandle_t handle,
                                                      cutensornetState_t tensorNetworkState, int32_t numMarginalModes,
                                                      const int32_t* marginalModes, int32_t numProjectedModes,
                                                      const int32_t* projectedModes,
                                                      const int64_t* marginalDiagonalTensorStrides,
                                                      cutensornetStateMarginal_t* tensorNetworkMarginal) {
  return create_marginal(handle, tensorNetworkState, true, numMarginalModes, marginalModes, numProjectedModes,
                         projectedModes, marginalDiagonalTensorStrides, tensorNetworkMarginal,
                         "cutensornetCreateMarginalDiagonal");
}

cutensornetStatus_t cutensornetMarginalConfigure(const cutensornetHandle_t handle,
                                                 cutensornetStateMarginal_t tensorNetworkMarginal,
                                                 cutensornetMarginalAttributes_t attribute, const void* attributeValue,
                                                 size_t attributeSize) {
  const char* api = "cutensornetMarginalConfigure";
  Marginal* m = marginal_of(tensorNetworkMarginal);
  if (!handle_of(handle) || !m) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid handle or marginal");
  const int at = enum_value(attribute);
  if (at != CUTENSORNET_MARGINAL_CONFIG_NUM_HYPER_SAMPLES && at != CUTENSORNET_MARGINAL_OPT_NUM_HYPER_SAMPLES)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "Unexpected attribute, please refer to enum cutensornetMarginalAttributes_t!");
  return config_hyper(m->hyper, at, CUTENSORNET_MARGINAL_CONFIG_NUM_HYPER_SAMPLES, CUTENSORNET_MARGINAL_OPT_NUM_HYPER_SAMPLES,
                      attributeValue, attributeSize, api);
}

cutensornetStatus_t cutensornetMarginalPrepare(const cutensornetHandle_t handle,
                                               cutensornetStateMarginal_t tensorNetworkMarginal,
                                               size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc,
                                               cudaStream_t cudaStream) {
  const char* api = "cutensornetMarginalPrepare";
  (void)cudaStream;
  Marginal* m = marginal_of(tensorNetworkMarginal);
  if (!handle_of(handle) || !m) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid handle or marginal");
  Workspace* w = work_of(workDesc);
  if (!w) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid workspace descriptor");
  if (m->s->epoch != m->epoch) return outdated("Marginal", api);
  if (Status st = check_channels(*m->s, api)) return st;
  const uint64_t limit = maxWorkspaceSizeDevice > (uint64_t)kExtra ? maxWorkspaceSizeDevice - kExtra : 0;
  NetBuild b;
  Circ c;
  c.choice.assign(m->s->ops.size(), 0);
  if (Status st = build_marginal(*m->s, m->open, m->proj, m->diagonal, m->strides, std::vector<const void*>(m->proj.size(), nullptr), true, b, c, api)) return st;
  if (Status st = plan_part(handle_of(handle), b, limit, m->part, api)) return st;
  const int64_t need = m->part.need + kExtra;
  if (Status st = prepare_workspace(w, need, maxWorkspaceSizeDevice, api)) return st;
  m->need = need;
  m->flops = m->part.flops;
  m->prepared = true;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetMarginalGetInfo(const cutensornetHandle_t handle,
                                               const cutensornetStateMarginal_t tensorNetworkMarginal,
                                               cutensornetMarginalAttributes_t attribute, void* attributeValue,
                                               size_t attributeSize) {
  const char* api = "cutensornetMarginalGetInfo";
  Marginal* m = marginal_of(const_cast<cutensornetStateMarginal_t>(tensorNetworkMarginal));
  if (!handle_of(handle) || !m) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid handle or marginal");
  const int at = enum_value(attribute);
  if (at == CUTENSORNET_MARGINAL_INFO_FLOPS) {
    const double f = m->prepared ? m->flops : 0.0;
    return get_info_value(&f, 8, attributeValue, attributeSize, api);
  }
  if (at == CUTENSORNET_MARGINAL_INFO_KIND) {
    const int32_t k = m->diagonal ? CUTENSORNET_MARGINAL_KIND_DIAGONAL : CUTENSORNET_MARGINAL_KIND_FULL;
    return get_info_value(&k, 4, attributeValue, attributeSize, api);
  }
  if (at == CUTENSORNET_MARGINAL_CONFIG_NUM_HYPER_SAMPLES || at == CUTENSORNET_MARGINAL_OPT_NUM_HYPER_SAMPLES)
    return get_info_value(&m->hyper, 4, attributeValue, attributeSize, api);
  return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "Invalid attribute!");
}

cutensornetStatus_t cutensornetMarginalCompute(const cutensornetHandle_t handle,
                                               cutensornetStateMarginal_t tensorNetworkMarginal,
                                               const int64_t* projectedModeValues,
                                               cutensornetWorkspaceDescriptor_t workDesc, void* marginalTensor,
                                               cudaStream_t cudaStream) {
  const char* api = "cutensornetMarginalCompute";
  Handle* h = handle_of(handle);
  Marginal* m = marginal_of(tensorNetworkMarginal);
  Workspace* w = work_of(workDesc);
  if (!h || !m || !w || !marginalTensor) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  State& s = *m->s;
  if (Status st = check_values(s, m->proj, projectedModeValues, api)) return st;
  if (Status st = check_scratch(w, 0, api)) return st;
  if (Status st = check_epoch_prepared(s, m->epoch, m->prepared, "Marginal", api)) return st;
  if (Status st = check_scratch(w, m->need, api)) return st;
  Status cap;
  if (graph_capture_refused(cudaStream, api, cap)) return cap;
  cudaStreamSynchronize(cudaStream);
  for (size_t j = 0; j < m->proj.size(); ++j)
    if (Status st = set_onehot(s, *m->onehot[j], s.ext[(size_t)m->proj[j]], projectedModeValues[j], api)) return st;
  ScratchSplit sp = split_scratch(w, kExtra);
  NetBuild b;
  Circ c;
  resolve_channels(s, c, s.rng, true);
  std::vector<const void*> oh;
  for (auto& o : m->onehot) oh.push_back(o->p);
  if (Status st = build_marginal(s, m->open, m->proj, m->diagonal, m->strides, oh, false, b, c, api)) return st;
  if (Status st = run_part(h, b, m->part, marginalTensor, &sp.rest, cudaStream, api, Dist::Sum)) return st;
  cudaStreamSynchronize(cudaStream);
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetDestroyMarginal(cutensornetStateMarginal_t tensorNetworkMarginal) {
  Marginal* m = marginal_of(tensorNetworkMarginal);
  if (!m) return tensorNetworkMarginal ? CUTENSORNET_STATUS_INVALID_VALUE : CUTENSORNET_STATUS_SUCCESS;
  m->magic = 0;
  delete m;
  return CUTENSORNET_STATUS_SUCCESS;
}

}  // extern "C"

// ---- samplers ----

namespace {

// The sampled modes in chunks whose joint distribution is computed by one
// contraction: each chunk's values are drawn given the values drawn before.
constexpr int64_t kChunkStates = 4096;

std::vector<std::vector<int>> sampler_chunks(const State& s, const std::vector<int32_t>& modes) {
  std::vector<std::vector<int>> chunks;
  std::vector<int> cur;
  int64_t vol = 1;
  for (size_t j = 0; j < modes.size(); ++j) {
    const int64_t d = s.ext[(size_t)modes[j]];
    if (!cur.empty() && vol * d > kChunkStates) {
      chunks.push_back(cur);
      cur.clear();
      vol = 1;
    }
    cur.push_back((int)j);
    vol *= d;
  }
  if (!cur.empty()) chunks.push_back(cur);
  return chunks;
}

int64_t chunk_volume(const State& s, const Sampler& sp, const std::vector<int>& chunk) {
  int64_t v = 1;
  for (int j : chunk) v *= s.ext[(size_t)sp.modes[(size_t)j]];
  return v;
}

int64_t sampler_scalar_bytes(const State& s, const Sampler& sp) {
  int64_t m = 1;
  for (const auto& c : sp.chunks) m = std::max(m, chunk_volume(s, sp, c));
  return std::max<int64_t>(kExtra, align_up(m * (int64_t)elem_bytes(s.type)));
}

}  // namespace

extern "C" {

cutensornetStatus_t cutensornetCreateSampler(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState,
                                             int32_t numModesToSample, const int32_t* modesToSample,
                                             cutensornetStateSampler_t* tensorNetworkSampler) {
  const char* api = "cutensornetCreateSampler";
  Handle* h;
  State* s;
  if (Status st = state_args(handle, tensorNetworkState, h, s, api)) return st;
  if (!tensorNetworkSampler) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "a null pointer was passed");
  if (awaits_mps(*s)) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "cuTensorNet internal error");
  auto sp = std::make_unique<Sampler>();
  if (numModesToSample < 1) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "no modes to sample");
  if (Status st = check_mode_list(numModesToSample, modesToSample, s->n, true, sp->modes, api)) return st;
  sp->s = s;
  sp->epoch = s->epoch;
  sp->chunks = sampler_chunks(*s, sp->modes);
  if (Status st = make_onehots(*s, sp->modes, sp->onehot, api)) return st;
  *tensorNetworkSampler = sp.release();
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetSamplerConfigure(const cutensornetHandle_t handle,
                                                cutensornetStateSampler_t tensorNetworkSampler,
                                                cutensornetSamplerAttributes_t attribute, const void* attributeValue,
                                                size_t attributeSize) {
  const char* api = "cutensornetSamplerConfigure";
  Sampler* sp = sampler_of(tensorNetworkSampler);
  if (!handle_of(handle) || !sp) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid handle or sampler");
  const int at = enum_value(attribute);
  if (at == CUTENSORNET_SAMPLER_CONFIG_DETERMINISTIC) {
    if (!attributeValue || attributeSize != 4) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "Unexpected attribute size!");
    int32_t seed;
    std::memcpy(&seed, attributeValue, 4);
    if (seed < 0) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "Unexpected value for the seed!");
    sp->seed = seed;
    // a positive seed makes the samples reproducible (and starts the sequence again);
    // zero seeds the generator at random
    sp->rng.seed(seed > 0 ? (uint64_t)seed : (uint64_t)std::random_device{}());
    return CUTENSORNET_STATUS_SUCCESS;
  }
  return config_hyper(sp->hyper, at, CUTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES, CUTENSORNET_SAMPLER_OPT_NUM_HYPER_SAMPLES,
                      attributeValue, attributeSize, api);
}

cutensornetStatus_t cutensornetSamplerPrepare(const cutensornetHandle_t handle,
                                              cutensornetStateSampler_t tensorNetworkSampler,
                                              size_t maxWorkspaceSizeDevice, cutensornetWorkspaceDescriptor_t workDesc,
                                              cudaStream_t cudaStream) {
  const char* api = "cutensornetSamplerPrepare";
  (void)cudaStream;
  Sampler* sp = sampler_of(tensorNetworkSampler);
  if (!handle_of(handle) || !sp) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid handle or sampler");
  Workspace* w = work_of(workDesc);
  if (!w) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid workspace descriptor");
  State& s = *sp->s;
  if (s.epoch != sp->epoch) return outdated("Sampler", api);
  if (Status st = check_channels(s, api)) return st;
  const int64_t extra = sampler_scalar_bytes(s, *sp);
  const uint64_t limit = maxWorkspaceSizeDevice > (uint64_t)extra ? maxWorkspaceSizeDevice - (uint64_t)extra : 0;
  sp->parts.assign(sp->chunks.size(), Part{});
  int64_t need = 0;
  double flops = 0;
  std::vector<int32_t> open, proj;
  for (size_t l = 0; l < sp->chunks.size(); ++l) {
    open.clear();
    for (int j : sp->chunks[l]) open.push_back(sp->modes[(size_t)j]);
    NetBuild b;
    Circ c;
    c.choice.assign(s.ops.size(), 0);
    if (Status st = build_marginal(s, open, proj, true, {}, std::vector<const void*>(proj.size(), nullptr), true, b, c, api))
      return st;
    if (Status st = plan_part(handle_of(handle), b, limit, sp->parts[l], api)) return st;
    need = std::max(need, sp->parts[l].need);
    flops += sp->parts[l].flops;
    proj.insert(proj.end(), open.begin(), open.end());
  }
  need += extra;
  if (Status st = prepare_workspace(w, need, maxWorkspaceSizeDevice, api)) return st;
  sp->need = need;
  sp->flops = flops;
  sp->prepared = true;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetSamplerGetInfo(const cutensornetHandle_t handle,
                                              const cutensornetStateSampler_t tensorNetworkSampler,
                                              cutensornetSamplerAttributes_t attribute, void* attributeValue,
                                              size_t attributeSize) {
  const char* api = "cutensornetSamplerGetInfo";
  Sampler* sp = sampler_of(const_cast<cutensornetStateSampler_t>(tensorNetworkSampler));
  if (!handle_of(handle) || !sp) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid handle or sampler");
  const int at = enum_value(attribute);
  if (at == CUTENSORNET_SAMPLER_INFO_FLOPS) {
    const double f = sp->prepared ? sp->flops : 0.0;
    return get_info_value(&f, 8, attributeValue, attributeSize, api);
  }
  if (at == CUTENSORNET_SAMPLER_CONFIG_DETERMINISTIC) return get_info_value(&sp->seed, 4, attributeValue, attributeSize, api);
  if (at == CUTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES || at == CUTENSORNET_SAMPLER_OPT_NUM_HYPER_SAMPLES)
    return get_info_value(&sp->hyper, 4, attributeValue, attributeSize, api);
  return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "Invalid attribute!");
}

cutensornetStatus_t cutensornetSamplerSample(const cutensornetHandle_t handle,
                                             cutensornetStateSampler_t tensorNetworkSampler, int64_t numShots,
                                             cutensornetWorkspaceDescriptor_t workDesc, int64_t* samples,
                                             cudaStream_t cudaStream) {
  const char* api = "cutensornetSamplerSample";
  Handle* h = handle_of(handle);
  Sampler* sp = sampler_of(tensorNetworkSampler);
  Workspace* w = work_of(workDesc);
  if (!h || !sp || !w || numShots <= 0 || !samples) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  State& s = *sp->s;
  if (Status st = check_scratch(w, 0, api)) return st;
  if (Status st = check_epoch_prepared(s, sp->epoch, sp->prepared, "Sampler", api)) return st;
  if (Status st = check_scratch(w, sp->need, api)) return st;
  Status cap;
  if (graph_capture_refused(cudaStream, api, cap)) return cap;
  cudaStreamSynchronize(cudaStream);
  const size_t K = sp->modes.size();
  const int64_t extra = sampler_scalar_bytes(s, *sp);
  ScratchSplit split = split_scratch(w, extra);
  // a pure state's stochastic channels are resolved once per call
  Circ c0;
  resolve_channels(s, c0, sp->rng, true);
  // one network per level, reused for every prefix
  std::vector<NetBuild> nets(sp->chunks.size());
  std::vector<int> first_of(sp->chunks.size());
  {
    std::vector<int32_t> open, proj;
    std::vector<const void*> oh;
    for (size_t l = 0; l < sp->chunks.size(); ++l) {
      open.clear();
      first_of[l] = sp->chunks[l].front();
      for (int j : sp->chunks[l]) open.push_back(sp->modes[(size_t)j]);
      Circ c = c0;
      if (Status st = build_marginal(s, open, proj, true, {}, oh, false, nets[l], c, api)) return st;
      proj.insert(proj.end(), open.begin(), open.end());
      for (int j : sp->chunks[l]) oh.push_back(sp->onehot[(size_t)j]->p);
    }
  }
  std::vector<std::map<std::vector<int64_t>, std::vector<double>>> cache(sp->chunks.size());
  std::vector<int64_t> prefix;
  for (int64_t shot = 0; shot < numShots; ++shot) {
    prefix.clear();
    for (size_t l = 0; l < sp->chunks.size(); ++l) {
      const auto& chunk = sp->chunks[l];
      auto it = cache[l].find(prefix);
      if (it == cache[l].end()) {
        for (size_t j = 0; j < prefix.size(); ++j)
          if (Status st = set_onehot(s, *sp->onehot[j], s.ext[(size_t)sp->modes[j]], prefix[j], api)) return st;
        const int64_t vol = chunk_volume(s, *sp, chunk);
        if (Status st = run_part(h, nets[l], sp->parts[l], split.scalars, &split.rest, cudaStream, api)) return st;
        cd dummy;
        if (!fetch_scalar(split.scalars, s.type, cudaStream, dummy)) return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "a copy failed");
        std::vector<cd> vals;
        if (!download(split.scalars, s.type, (size_t)vol, vals)) return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "a copy failed");
        std::vector<double> dist((size_t)vol);
        for (int64_t i = 0; i < vol; ++i) dist[(size_t)i] = std::max(vals[(size_t)i].real(), 0.0);
        it = cache[l].emplace(prefix, std::move(dist)).first;
      }
      const std::vector<double>& dist = it->second;
      double total = 0;
      for (double x : dist) total += x;
      if (!(total > 0)) return fail(CUTENSORNET_STATUS_INTERNAL_ERROR, api, "the state has no probability to sample from");
      std::uniform_real_distribution<double> u(0.0, total);
      const double r = u(sp->rng);
      double acc = 0;
      size_t pick = dist.size() - 1;
      for (size_t i = 0; i < dist.size(); ++i) {
        acc += dist[i];
        if (r < acc && dist[i] > 0) {
          pick = i;
          break;
        }
      }
      size_t rest = pick;
      for (int j : chunk) {
        const int64_t d = s.ext[(size_t)sp->modes[(size_t)j]];
        const int64_t v = (int64_t)(rest % (size_t)d);
        rest /= (size_t)d;
        prefix.push_back(v);
        samples[(size_t)shot * K + (size_t)j] = v;
      }
    }
  }
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetDestroySampler(cutensornetStateSampler_t tensorNetworkSampler) {
  Sampler* sp = sampler_of(tensorNetworkSampler);
  if (!sp) return tensorNetworkSampler ? CUTENSORNET_STATUS_INVALID_VALUE : CUTENSORNET_STATUS_SUCCESS;
  sp->magic = 0;
  delete sp;
  return CUTENSORNET_STATUS_SUCCESS;
}

// ---- expectation values ----

cutensornetStatus_t cutensornetCreateExpectation(const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState,
                                                 cutensornetNetworkOperator_t tensorNetworkOperator,
                                                 cutensornetStateExpectation_t* tensorNetworkExpectation) {
  const char* api = "cutensornetCreateExpectation";
  Handle* h;
  State* s;
  if (Status st = state_args(handle, tensorNetworkState, h, s, api)) return st;
  NetOp* op = netop_of(tensorNetworkOperator);
  if (!op || !tensorNetworkExpectation) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  // RTX 3060, cuTensorNet 2.14: an operator made for another number of modes, or in another data
  // type, is accepted -- its terms act on the state modes they name, and the rest of the state
  // is left alone -- and so is computing with it. Here its terms must name modes of the state,
  // and a data type other than the state's is refused when the expectation is prepared.
  if (op->comps.empty()) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the operator has no components");
  for (const NetComponent& comp : op->comps)
    for (const auto& f : comp.fmodes)
      for (int32_t m : f)
        if (m < 0 || m >= s->n) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the operator names a mode the state does not have");
  if (awaits_mps(*s)) return fail(CUTENSORNET_STATUS_INTERNAL_ERROR, api, "cuTensorNet internal error");
  auto e = std::make_unique<Expectation>();
  e->s = s;
  e->op = op;
  e->epoch = s->epoch;
  e->op_version = op->version;
  *tensorNetworkExpectation = e.release();
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetExpectationConfigure(const cutensornetHandle_t handle,
                                                    cutensornetStateExpectation_t tensorNetworkExpectation,
                                                    cutensornetExpectationAttributes_t attribute,
                                                    const void* attributeValue, size_t attributeSize) {
  const char* api = "cutensornetExpectationConfigure";
  Expectation* e = expectation_of(tensorNetworkExpectation);
  if (!handle_of(handle) || !e) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid handle or expectation");
  return config_hyper(e->hyper, enum_value(attribute), CUTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES,
                      CUTENSORNET_EXPECTATION_OPT_NUM_HYPER_SAMPLES, attributeValue, attributeSize, api);
}

cutensornetStatus_t cutensornetExpectationPrepare(const cutensornetHandle_t handle,
                                                  cutensornetStateExpectation_t tensorNetworkExpectation,
                                                  size_t maxWorkspaceSizeDevice,
                                                  cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream) {
  const char* api = "cutensornetExpectationPrepare";
  (void)cudaStream;
  Expectation* e = expectation_of(tensorNetworkExpectation);
  if (!handle_of(handle) || !e) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid handle or expectation");
  Workspace* w = work_of(workDesc);
  if (!w) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid workspace descriptor");
  State& s = *e->s;
  if (s.epoch != e->epoch || e->op->version != e->op_version) return outdated("Expectation value", api);
  if (e->op->type != s.type)
    return refuse(api, "an operator in another data type than the state's is not read in its own type here");
  if (Status st = check_channels(s, api)) return st;
  const int64_t extra = 2 * kExtra;
  const uint64_t limit = maxWorkspaceSizeDevice > (uint64_t)extra ? maxWorkspaceSizeDevice - (uint64_t)extra : 0;
  e->parts.assign(e->op->comps.size(), Part{});
  int64_t need = 0;
  double flops = 0;
  for (size_t k = 0; k < e->op->comps.size(); ++k) {
    NetBuild b;
    Circ c;
    c.choice.assign(s.ops.size(), 0);
    if (Status st = build_component(s, e->op->comps[k], true, b, c, api)) return st;
    if (Status st = plan_part(handle_of(handle), b, limit, e->parts[k], api)) return st;
    need = std::max(need, e->parts[k].need);
    flops += e->parts[k].flops;
  }
  NetBuild nb;
  Circ nc;
  nc.choice.assign(s.ops.size(), 0);
  if (Status st = build_norm(s, true, nb, nc, api)) return st;
  if (Status st = plan_part(handle_of(handle), nb, limit, e->norm, api)) return st;
  need = std::max(need, e->norm.need) + extra;
  if (Status st = prepare_workspace(w, need, maxWorkspaceSizeDevice, api)) return st;
  e->need = need;
  e->flops = flops;
  e->prepared = true;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetExpectationGetInfo(const cutensornetHandle_t handle,
                                                  const cutensornetStateExpectation_t tensorNetworkExpectation,
                                                  cutensornetExpectationAttributes_t attribute, void* attributeValue,
                                                  size_t attributeSize) {
  const char* api = "cutensornetExpectationGetInfo";
  Expectation* e = expectation_of(const_cast<cutensornetStateExpectation_t>(tensorNetworkExpectation));
  if (!handle_of(handle) || !e) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid handle or expectation");
  const int at = enum_value(attribute);
  if (at == CUTENSORNET_EXPECTATION_INFO_FLOPS) {
    const double f = e->prepared ? e->flops : 0.0;
    return get_info_value(&f, 8, attributeValue, attributeSize, api);
  }
  if (at == CUTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES || at == CUTENSORNET_EXPECTATION_OPT_NUM_HYPER_SAMPLES)
    return get_info_value(&e->hyper, 4, attributeValue, attributeSize, api);
  return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "Invalid attribute!");
}

cutensornetStatus_t cutensornetExpectationCompute(const cutensornetHandle_t handle,
                                                  cutensornetStateExpectation_t tensorNetworkExpectation,
                                                  cutensornetWorkspaceDescriptor_t workDesc, void* expectationValue,
                                                  void* stateNorm, cudaStream_t cudaStream) {
  const char* api = "cutensornetExpectationCompute";
  Handle* h = handle_of(handle);
  Expectation* e = expectation_of(tensorNetworkExpectation);
  Workspace* w = work_of(workDesc);
  if (!h || !e || !w || !expectationValue) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  State& s = *e->s;
  if (Status st = check_scratch(w, 0, api)) return st;
  if (s.epoch != e->epoch || e->op->version != e->op_version) return outdated("Expectation value", api);
  if (!e->prepared) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the expectation value has not been prepared");
  if (Status st = check_scratch(w, e->need, api)) return st;
  Status cap;
  if (graph_capture_refused(cudaStream, api, cap)) return cap;
  cudaStreamSynchronize(cudaStream);
  ScratchSplit sp = split_scratch(w, 2 * kExtra);
  Circ c0;
  resolve_channels(s, c0, s.rng, true);
  cd total = 0;
  for (size_t k = 0; k < e->op->comps.size(); ++k) {
    NetBuild b;
    Circ c = c0;
    if (Status st = build_component(s, e->op->comps[k], false, b, c, api)) return st;
    if (Status st = run_part(h, b, e->parts[k], sp.scalars, &sp.rest, cudaStream, api, Dist::Part)) return st;
    cd v;
    if (!fetch_scalar(sp.scalars, s.type, cudaStream, v)) return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "a copy failed");
    total += e->op->comps[k].coef * v;
  }
  World world;
  if (Status st = world_of(h, world, api)) return st;
  if (world.distributed && world.size > 1) {  // the ranks' parts of the value, summed
    alignas(16) unsigned char host[16] = {0};
    put_scalar(host, s.type, total);
    const auto comm = comm_of(h);
    if (cudaMemcpy(sp.scalars, host, elem_bytes(s.type), cudaMemcpyHostToDevice) != cudaSuccess ||
        g_comm_iface->AllreduceInPlace(&comm, sp.scalars, 1, s.type) ||
        cudaMemcpy(host, sp.scalars, elem_bytes(s.type), cudaMemcpyDeviceToHost) != cudaSuccess)
      return fail(CUTENSORNET_STATUS_DISTRIBUTED_FAILURE, api, "the distributed communication service failed");
    total = get_scalar(host, s.type);
  }
  put_scalar(expectationValue, s.type, total);
  if (stateNorm) {
    NetBuild nb;
    Circ nc = c0;
    if (Status st = build_norm(s, false, nb, nc, api)) return st;
    if (Status st = run_part(h, nb, e->norm, sp.scalars + kExtra, &sp.rest, cudaStream, api, Dist::Sum)) return st;
    cd v;
    if (!fetch_scalar(sp.scalars + kExtra, s.type, cudaStream, v)) return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "a copy failed");
    put_scalar(stateNorm, s.type, v);
  }
  cudaStreamSynchronize(cudaStream);
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetDestroyExpectation(cutensornetStateExpectation_t tensorNetworkExpectation) {
  Expectation* e = expectation_of(tensorNetworkExpectation);
  if (!e) return tensorNetworkExpectation ? CUTENSORNET_STATUS_INVALID_VALUE : CUTENSORNET_STATUS_SUCCESS;
  e->magic = 0;
  delete e;
  return CUTENSORNET_STATUS_SUCCESS;
}

}  // extern "C"

// ---- gradients of a network ----
//
// The gradient of each input that requires one is the adjoint of the output contracted with
// the other inputs (conjugated, for complex data): computed here as a contraction of its own
// per input rather than as the backward sweep of the forward path, so it needs no
// intermediates from the forward pass. The cache workspace the sizes report is checked, not
// used. Checked against NVIDIA's library on an RTX 3060 (2.14): the values for real and
// complex networks, overwrite and accumulate, and the statuses of the argument errors.

extern "C" {

cutensornetStatus_t cutensornetNetworkSetGradientTensorMemory(const cutensornetHandle_t handle,
                                                              cutensornetNetworkDescriptor_t networkDesc,
                                                              int64_t correspondingTensorId, void* const buffer,
                                                              const int64_t strides[]) {
  const char* api = "cutensornetNetworkSetGradientTensorMemory";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  if (!n || correspondingTensorId < 0 || correspondingTensorId >= (int64_t)n->in.size())
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid tensor id");
  NetTensor& t = n->in[(size_t)correspondingTensorId];
  if (!t.q.requiresGradient)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "this tensor does not require a gradient");
  t.grad = buffer;
  if (strides) t.grad_strides.assign(strides, strides + t.modes.size());
  else t.grad_strides.clear();
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetNetworkSetAdjointTensorMemory(const cutensornetHandle_t handle,
                                                             cutensornetNetworkDescriptor_t networkDesc,
                                                             const void* const buffer, const int64_t strides[]) {
  const char* api = "cutensornetNetworkSetAdjointTensorMemory";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  if (!n) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid network");
  fill_output(*n);
  n->adjoint = buffer;  // a NULL buffer is taken here and fails when the gradients are computed
  if (strides) n->adjoint_strides.assign(strides, strides + n->out.modes.size());
  else n->adjoint_strides.clear();
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetNetworkPrepareGradientsBackward(const cutensornetHandle_t handle,
                                                               cutensornetNetworkDescriptor_t networkDesc,
                                                               const cutensornetWorkspaceDescriptor_t workDesc) {
  const char* api = "cutensornetNetworkPrepareGradientsBackward";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  if (!n || (workDesc && !work_of(workDesc))) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  if (!n->prepared) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "prepare the contraction first");
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetNetworkComputeGradientsBackward(const cutensornetHandle_t handle,
                                                               cutensornetNetworkDescriptor_t networkDesc,
                                                               int32_t accumulateOutput,
                                                               const cutensornetWorkspaceDescriptor_t workDesc,
                                                               const cutensornetSliceGroup_t sliceGroup,
                                                               cudaStream_t stream) {
  const char* api = "cutensornetNetworkComputeGradientsBackward";
  Handle* h = handle_of(handle);
  if (!h) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  Workspace* w = work_of(workDesc);
  if (!n || (workDesc && !w)) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  const SliceGroup* group = as<SliceGroup>(sliceGroup, kMagicSlices);
  if (sliceGroup && !group) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid slice group");
  if (!n->prepared) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "prepare the contraction first");
  if (!n->adjoint) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the output's adjoint has no memory");
  std::vector<size_t> wanted;
  for (size_t i = 0; i < n->in.size(); ++i)
    if (n->in[i].q.requiresGradient) wanted.push_back(i);
  if (wanted.empty()) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "no input tensor requires a gradient");
  // RTX 3060, cuTensorNet 2.14: the cache workspace must have been given memory -- no cache
  // (or no workspace at all) is INSUFFICIENT_WORKSPACE, whatever size is set once there is one.
  if (!w || !w->ptr[CUTENSORNET_MEMSPACE_DEVICE][CUTENSORNET_WORKSPACE_CACHE] ||
      w->size[CUTENSORNET_MEMSPACE_DEVICE][CUTENSORNET_WORKSPACE_CACHE] <= 0)
    return fail(CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE, api, "the gradients need cache workspace memory");
  for (const NetTensor& t : n->in)
    if (!t.data) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "an input tensor has no memory");
  fill_output(*n);
  const Path fwd = n->has_path ? n->path : Path{};
  std::vector<int64_t> ids;
  if (Status st = slice_ids(group, slices_of(fwd, extent_map(*n)), ids, api)) return st;
  for (size_t i : wanted) {
    if (!n->in[i].grad) continue;  // no memory for this gradient: it is left out, and the call succeeds (RTX 3060, 2.14)
    GradNet g;
    if (Status st = build_grad_net(*n, i, fwd, g, api)) return st;
    std::vector<const void*> in;
    for (const NetTensor& t : g.net.in) in.push_back(t.data);
    if (Status st = contract_slices(h, g.net, g.path, in, n->in[i].grad, accumulateOutput != 0, w, ids, stream, api))
      return st;
  }
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetExpectationComputeWithGradientsBackward(
    const cutensornetHandle_t, cutensornetStateExpectation_t, int32_t, const void*, const void*,
    cutensornetWorkspaceDescriptor_t, void*, void*, cudaStream_t) {
  return refuse("cutensornetExpectationComputeWithGradientsBackward", "gradients are not implemented");
}

}  // extern "C"
