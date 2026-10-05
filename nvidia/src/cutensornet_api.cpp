// libvgpucutensornet -- VirtualGPU's cuTensorNet, presented as libcutensornet.so.2.
//
// NVIDIA's libcutensornet cannot run on a simulated GPU: it carries a
// statically linked CUDA runtime, which reaches the driver through NVIDIA's
// undocumented internal interface. This is the documented cuTensorNet 2.14
// API (nvidia/include/vgpu_cutensornet.h) built, as NVIDIA's is, on cuTENSOR
// and cuSOLVER -- the simulator's own libcutensor and libcusolver. A network
// is contracted pairwise with cutensorContract into intermediates carved from
// the caller's workspace, one slice at a time; QR and SVD permute the tensor
// into a matrix with cutensorPermute and factor it with cusolverDn.
//
// What a typical cuQuantum Python program calls is the subset implemented
// first and in full (traced on an RTX 3060 against NVIDIA's library): the
// network API (create, append tensors, set the output, set tensor memory,
// optimize, compute workspace sizes, prepare, contract), the older
// descriptor-and-plan API, slice groups, the optimizer configuration and
// information (with packing), tensor descriptors, QR, SVD with every
// truncation and partition option, and gate splitting. The state API
// (states, operators, expectations, marginals, samplers, MPS projection),
// gradients and distributed execution answer NOT_SUPPORTED.
//
// What is ours. The contraction path is found by a greedy pairwise search
// and slicing picks the contracted modes that shrink the largest
// intermediate most, so paths, slicing, FLOP counts and workspace sizes
// differ from NVIDIA's hyper-optimizer; the contracted result is the same
// tensor. Statuses and attribute sizes follow NVIDIA's library on an RTX
// 3060 where the documentation leaves them open.
#include "../include/vgpu_cutensornet.h"
#include "../include/vgpu_cutensor.h"
#include "enum_value.hpp"

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

using Status = cutensornetStatus_t;
using cd = std::complex<double>;
using cf = std::complex<float>;

// ---- errors and logging ----

thread_local std::string t_last_error;

struct Logger {
  cutensornetLoggerCallback_t callback = nullptr;
  cutensornetLoggerCallbackData_t callback_data = nullptr;
  void* user = nullptr;
  FILE* file = nullptr;
  bool own_file = false, disabled = false;
  int level = 0, mask = 0;
};
Logger& logger() {
  static Logger* l = new Logger;
  return *l;
}
std::mutex& log_mu() {
  static std::mutex* m = new std::mutex;
  return *m;
}

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

// Records the message for cutensornetGetLastError and the logger, and
// returns the status, so a refusal reads `return fail(s, api, why)`.
Status fail(Status s, const char* api, const std::string& why) {
  t_last_error = std::string(api) + ": " + why;
  std::lock_guard<std::mutex> g(log_mu());
  Logger& l = logger();
  if (l.disabled || (l.level < 1 && !(l.mask & 1))) return s;
  if (l.callback) l.callback(1, api, why.c_str());
  if (l.callback_data) l.callback_data(1, api, why.c_str(), l.user);
  if (l.file) std::fprintf(l.file, "[cuTensorNet][Error][%s] %s\n", api, why.c_str());
  return s;
}

// A refusal the user should see: the library's own NOT_SUPPORTED.
Status refuse(const char* api, const std::string& why) {
  if (!quiet()) std::fprintf(stderr, "[vgpu] %s: %s\n", api, why.c_str());
  return fail(CUTENSORNET_STATUS_NOT_SUPPORTED, api, why);
}

Status from_cutensor(cutensorStatus_t s) {
  switch (s) {
    case CUTENSOR_STATUS_SUCCESS: return CUTENSORNET_STATUS_SUCCESS;
    case CUTENSOR_STATUS_INVALID_VALUE: return CUTENSORNET_STATUS_INVALID_VALUE;
    case CUTENSOR_STATUS_NOT_SUPPORTED: return CUTENSORNET_STATUS_NOT_SUPPORTED;
    case CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE: return CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE;
    case CUTENSOR_STATUS_CUDA_ERROR: return CUTENSORNET_STATUS_CUDA_ERROR;
    default: return CUTENSORNET_STATUS_CUTENSOR_ERROR;
  }
}

// ---- types ----

size_t elem_bytes(cudaDataType_t t) {
  switch (t) {
    case CUDA_R_16F: case CUDA_R_16BF: return 2;
    case CUDA_R_32F: case CUDA_C_16F: return 4;
    case CUDA_R_64F: case CUDA_C_32F: return 8;
    case CUDA_C_64F: return 16;
    default: return 0;
  }
}
bool is_complex(cudaDataType_t t) { return t == CUDA_C_16F || t == CUDA_C_32F || t == CUDA_C_64F; }
bool network_type(cudaDataType_t t) {
  return t == CUDA_R_16F || t == CUDA_R_16BF || t == CUDA_R_32F || t == CUDA_R_64F || t == CUDA_C_32F ||
         t == CUDA_C_64F;
}

// The default compute type for each data type (documented table).
cutensornetComputeType_t default_compute(cudaDataType_t t) {
  return (t == CUDA_R_64F || t == CUDA_C_64F) ? CUTENSORNET_COMPUTE_64F : CUTENSORNET_COMPUTE_32F;
}

bool compute_ok(cudaDataType_t t, cutensornetComputeType_t c) {
  switch (t) {
    case CUDA_R_16F: case CUDA_R_16BF: return c == CUTENSORNET_COMPUTE_32F;
    case CUDA_R_32F:
      return c == CUTENSORNET_COMPUTE_32F || c == CUTENSORNET_COMPUTE_TF32 || c == CUTENSORNET_COMPUTE_3XTF32 ||
             c == CUTENSORNET_COMPUTE_16BF || c == CUTENSORNET_COMPUTE_16F;
    case CUDA_C_32F:
      return c == CUTENSORNET_COMPUTE_32F || c == CUTENSORNET_COMPUTE_TF32 || c == CUTENSORNET_COMPUTE_3XTF32;
    case CUDA_R_64F: case CUDA_C_64F: return c == CUTENSORNET_COMPUTE_64F || c == CUTENSORNET_COMPUTE_32F;
    default: return false;
  }
}

cutensorComputeDescriptor_t to_cutensor(cutensornetComputeType_t c) {
  switch (c) {
    case CUTENSORNET_COMPUTE_16F: return CUTENSOR_COMPUTE_DESC_16F;
    case CUTENSORNET_COMPUTE_16BF: return CUTENSOR_COMPUTE_DESC_16BF;
    case CUTENSORNET_COMPUTE_TF32: return CUTENSOR_COMPUTE_DESC_TF32;
    case CUTENSORNET_COMPUTE_3XTF32: return CUTENSOR_COMPUTE_DESC_3XTF32;
    case CUTENSORNET_COMPUTE_64F: return CUTENSOR_COMPUTE_DESC_64F;
    default: return CUTENSOR_COMPUTE_DESC_32F;
  }
}

std::vector<int64_t> packed_strides(const std::vector<int64_t>& ext) {
  std::vector<int64_t> s(ext.size());
  uint64_t p = 1;  // unsigned: the product past the last mode is never used
  for (size_t i = 0; i < ext.size(); ++i) {
    s[i] = (int64_t)p;
    p *= (uint64_t)ext[i];
  }
  return s;
}

int64_t volume(const std::vector<int64_t>& ext) {
  int64_t v = 1;
  for (int64_t e : ext) v *= e;
  return v;
}

constexpr int64_t kAlign = 256;
int64_t align_up(int64_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

}  // namespace

namespace {

// ---- objects ----

constexpr uint32_t kMagicHandle = 0x544e4831, kMagicNet = 0x544e4e31, kMagicTensor = 0x544e5431,
                   kMagicWork = 0x544e5731, kMagicConfig = 0x544e4331, kMagicInfo = 0x544e4931,
                   kMagicPlan = 0x544e5031, kMagicSlices = 0x544e5331, kMagicTune = 0x544e4131,
                   kMagicSvdConfig = 0x544e5643, kMagicSvdInfo = 0x544e5649;

struct Handle {
  uint32_t magic = kMagicHandle;
  int device = 0;
  cutensorHandle_t ct = nullptr;
  cusolverDnHandle_t solver = nullptr;
  bool has_mem = false;
  cutensornetDeviceMemHandler_t mem{};
};

struct TensorDesc {
  uint32_t magic = kMagicTensor;
  std::vector<int32_t> modes;
  std::vector<int64_t> extents, strides;
  cudaDataType_t type = CUDA_R_32F;
};

struct NetTensor {
  std::vector<int32_t> modes;
  std::vector<int64_t> extents, strides;  // strides empty: column-major
  cutensornetTensorQualifiers_t q{0, 0, 0};
  const void* data = nullptr;
};

struct Path {
  std::vector<std::pair<int32_t, int32_t>> steps;
  std::vector<std::pair<int32_t, int64_t>> slices;  // (mode, number of slices along it)
};

struct Network {
  uint32_t magic = kMagicNet;
  bool legacy = false;
  std::vector<NetTensor> in;
  NetTensor out;
  bool out_set = false;
  cudaDataType_t type = CUDA_R_32F;
  bool type_set = false;
  cutensornetComputeType_t compute = CUTENSORNET_COMPUTE_32F;
  bool compute_set = false;
  void* out_data = nullptr;
  bool has_path = false;
  Path path;
  bool prepared = false;
};

struct Workspace {
  uint32_t magic = kMagicWork;
  int64_t need[3][2][2] = {};  // [pref][memspace][kind]
  void* ptr[2][2] = {};        // [memspace][kind]
  int64_t size[2][2] = {};
};

struct Config {
  uint32_t magic = kMagicConfig;
  std::map<int, int32_t> v;
};

struct Network;
struct Info {
  uint32_t magic = kMagicInfo;
  Network* net = nullptr;  // the network it was created for
  int32_t num_inputs = 0;
  int64_t num_slices = 1;
  Path path;
  bool has_path = false;
  double phase1_flops = 0, flops = 0, largest = 0, overhead = 1;
  std::vector<std::vector<int32_t>> inter_modes;
};

struct Plan {
  uint32_t magic = kMagicPlan;
  Network* net = nullptr;
  Path path;
};

struct SliceGroup {
  uint32_t magic = kMagicSlices;
  bool range = false;
  int64_t start = 0, stop = 0, step = 1;
  std::vector<int64_t> ids;
};

struct TunePref {
  uint32_t magic = kMagicTune;
  int32_t max_iterations = 3, intermediate_modes = 2;
};

struct SvdConfig {
  uint32_t magic = kMagicSvdConfig;
  double abs_cutoff = 0, rel_cutoff = 0, discarded_cutoff = 0;
  int32_t normalization = CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE;
  int32_t partition = CUTENSORNET_TENSOR_SVD_PARTITION_NONE;
  int32_t algo = CUTENSORNET_TENSOR_SVD_ALGO_GESVD;
  cutensornetGesvdjParams_t gesvdj{0, 0};
  cutensornetGesvdrParams_t gesvdr{0, 0};
};

struct SvdInfo {
  uint32_t magic = kMagicSvdInfo;
  int64_t full = 0, reduced = 0;
  double discarded = 0;
  int32_t algo = CUTENSORNET_TENSOR_SVD_ALGO_GESVD;
};

template <class T>
T* as(void* p, uint32_t magic) {
  T* t = static_cast<T*>(p);
  return (t && t->magic == magic) ? t : nullptr;
}

Handle* handle_of(const cutensornetHandle_t h) { return as<Handle>(h, kMagicHandle); }
Network* net_of(const cutensornetNetworkDescriptor_t n) { return as<Network>(n, kMagicNet); }
TensorDesc* tensor_of(const cutensornetTensorDescriptor_t t) { return as<TensorDesc>(t, kMagicTensor); }
Workspace* work_of(const cutensornetWorkspaceDescriptor_t w) { return as<Workspace>(w, kMagicWork); }
Config* config_of(const cutensornetContractionOptimizerConfig_t c) { return as<Config>(c, kMagicConfig); }
Info* info_of(const cutensornetContractionOptimizerInfo_t i) { return as<Info>(i, kMagicInfo); }

}  // namespace

namespace {

// ---- the network as modes ----

std::map<int32_t, int64_t> extent_map(const Network& n) {
  std::map<int32_t, int64_t> e;
  for (const NetTensor& t : n.in)
    for (size_t i = 0; i < t.modes.size(); ++i) e.emplace(t.modes[i], t.extents[i]);
  return e;
}

// Output modes when the caller gave none: every mode that appears in exactly
// one input, in increasing label order (einsum's implicit output).
std::vector<int32_t> implicit_output(const Network& n) {
  std::map<int32_t, int> count;
  for (const NetTensor& t : n.in)
    for (int32_t m : t.modes) ++count[m];
  std::vector<int32_t> out;
  for (auto& [m, c] : count)
    if (c == 1) out.push_back(m);
  return out;
}

void fill_output(Network& n) {
  if (n.out_set) return;
  n.out.modes = implicit_output(n);
  auto e = extent_map(n);
  n.out.extents.clear();
  for (int32_t m : n.out.modes) n.out.extents.push_back(e[m]);
}

bool contains(const std::vector<int32_t>& v, int32_t m) { return std::find(v.begin(), v.end(), m) != v.end(); }

// The modes a pairwise contraction keeps: those still needed by another
// tensor or by the output, in the order they first appear in A then B.
std::vector<int32_t> kept_modes(const std::vector<int32_t>& a, const std::vector<int32_t>& b,
                                const std::vector<std::vector<int32_t>>& others, const std::vector<int32_t>& out) {
  std::vector<int32_t> r;
  for (const auto* src : {&a, &b})
    for (int32_t m : *src) {
      if (contains(r, m)) continue;
      bool needed = contains(out, m);
      for (const auto& o : others) needed = needed || contains(o, m);
      if (needed) r.push_back(m);
    }
  return r;
}

double mode_volume(const std::vector<int32_t>& modes, const std::map<int32_t, int64_t>& ext) {
  double v = 1;
  for (int32_t m : modes) v *= (double)ext.at(m);
  return v;
}

std::vector<int32_t> union_modes(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  std::vector<int32_t> u = a;
  for (int32_t m : b)
    if (!contains(u, m)) u.push_back(m);
  return u;
}

// The intermediates a path produces (the last one is the output) and the
// FLOP count: 2 per multiply-add over the union of the pair's modes (8 for
// complex data), as cuTENSOR counts a contraction.
struct PathCost {
  std::vector<std::vector<int32_t>> inter;
  double flops = 0, largest = 0;
};

bool cost_of(const Network& n, const std::vector<std::pair<int32_t, int32_t>>& steps,
             const std::map<int32_t, int64_t>& ext, PathCost& pc) {
  std::vector<std::vector<int32_t>> live;
  for (const NetTensor& t : n.in) live.push_back(t.modes);
  const double fm = is_complex(n.type) ? 8 : 2;
  pc = PathCost{};
  for (size_t s = 0; s < steps.size(); ++s) {
    int32_t i = steps[s].first, j = steps[s].second;
    if (i == j || i < 0 || j < 0 || i >= (int32_t)live.size() || j >= (int32_t)live.size()) return false;
    if (i > j) std::swap(i, j);
    std::vector<std::vector<int32_t>> others;
    for (int32_t k = 0; k < (int32_t)live.size(); ++k)
      if (k != i && k != j) others.push_back(live[k]);
    std::vector<int32_t> r = s + 1 == steps.size() ? n.out.modes : kept_modes(live[i], live[j], others, n.out.modes);
    pc.flops += fm * mode_volume(union_modes(live[i], live[j]), ext);
    if (s + 1 < steps.size()) pc.largest = std::max(pc.largest, mode_volume(r, ext));
    pc.inter.push_back(r);
    live.erase(live.begin() + j);
    live.erase(live.begin() + i);
    live.push_back(r);
  }
  return live.size() == 1 || n.in.empty();
}

// Greedy pairwise search: at each step the pair that shares a mode and costs
// the fewest multiply-adds (ties: the smaller result, then the lowest
// indices); pairs that share nothing only when no pair shares anything.
std::vector<std::pair<int32_t, int32_t>> greedy_path(const Network& n, const std::map<int32_t, int64_t>& ext) {
  std::vector<std::vector<int32_t>> live;
  for (const NetTensor& t : n.in) live.push_back(t.modes);
  std::vector<std::pair<int32_t, int32_t>> steps;
  while (live.size() > 1) {
    int bi = -1, bj = -1;
    bool best_shares = false;
    double best_cost = 0, best_size = 0;
    for (int i = 0; i < (int)live.size(); ++i)
      for (int j = i + 1; j < (int)live.size(); ++j) {
        bool shares = false;
        for (int32_t m : live[i]) shares = shares || contains(live[j], m);
        std::vector<std::vector<int32_t>> others;
        for (int k = 0; k < (int)live.size(); ++k)
          if (k != i && k != j) others.push_back(live[k]);
        const double cost = mode_volume(union_modes(live[i], live[j]), ext);
        const double size = mode_volume(kept_modes(live[i], live[j], others, n.out.modes), ext);
        const bool better = bi < 0 || (shares && !best_shares) ||
                            (shares == best_shares && (cost < best_cost || (cost == best_cost && size < best_size)));
        if (better) {
          bi = i, bj = j, best_shares = shares, best_cost = cost, best_size = size;
        }
      }
    std::vector<std::vector<int32_t>> others;
    for (int k = 0; k < (int)live.size(); ++k)
      if (k != bi && k != bj) others.push_back(live[k]);
    std::vector<int32_t> r = kept_modes(live[bi], live[bj], others, n.out.modes);
    steps.emplace_back(bi, bj);
    live.erase(live.begin() + bj);
    live.erase(live.begin() + bi);
    live.push_back(r);
  }
  return steps;
}

}  // namespace

namespace {

// ---- contracting through cuTENSOR ----

// The index range [lo, hi) each sliced mode takes in one slice.
struct SliceView {
  std::map<int32_t, std::pair<int64_t, int64_t>> range;
};
SliceView slice_view_of(const Path& p, const std::map<int32_t, int64_t>& ext, int64_t id);
int64_t scratch_need(const Network& n, const Path& p);

// A tensor as cuTENSOR sees it: modes, extents, strides (elements), base.
struct View {
  std::vector<int32_t> modes;
  std::vector<int64_t> ext, str;
  const void* ptr = nullptr;
  bool conj = false;
};

View view_of(const NetTensor& t, const void* data, const SliceView& sv, size_t eb) {
  View v;
  v.modes = t.modes;
  v.ext = t.extents;
  v.str = t.strides.empty() ? packed_strides(t.extents) : t.strides;
  v.conj = t.q.isConjugate != 0;
  const char* p = static_cast<const char*>(data);
  for (size_t i = 0; i < v.modes.size(); ++i) {
    auto it = sv.range.find(v.modes[i]);
    if (it == sv.range.end()) continue;
    p += it->second.first * v.str[i] * (int64_t)eb;
    v.ext[i] = it->second.second - it->second.first;
  }
  v.ptr = p;
  return v;
}

// Scalars of a cuTENSOR operation's scalar type.
struct Scalar {
  alignas(16) unsigned char b[16] = {};
  Scalar(cudaDataType_t t, double v) {
    switch (t) {
      case CUDA_R_32F: { float f = (float)v; std::memcpy(b, &f, 4); break; }
      case CUDA_R_64F: std::memcpy(b, &v, 8); break;
      case CUDA_C_32F: { float f[2] = {(float)v, 0}; std::memcpy(b, f, 8); break; }
      case CUDA_C_64F: { double f[2] = {v, 0}; std::memcpy(b, f, 16); break; }
      default: { float f = (float)v; std::memcpy(b, &f, 4); break; }
    }
  }
};

struct TensorGuard {
  cutensorTensorDescriptor_t d = nullptr;
  ~TensorGuard() { cutensorDestroyTensorDescriptor(d); }
};
struct OpGuard {
  cutensorOperationDescriptor_t d = nullptr;
  ~OpGuard() { cutensorDestroyOperationDescriptor(d); }
};
struct PlanGuard {
  cutensorPlan_t p = nullptr;
  ~PlanGuard() { cutensorDestroyPlan(p); }
};
struct PrefGuard {
  cutensorPlanPreference_t p = nullptr;
  ~PrefGuard() { cutensorDestroyPlanPreference(p); }
};

cutensorStatus_t make_desc(cutensorHandle_t h, const View& v, cudaDataType_t t, TensorGuard& g) {
  return cutensorCreateTensorDescriptor(h, &g.d, (uint32_t)v.modes.size(), v.ext.data(), v.str.data(), t,
                                        (uint32_t)elem_bytes(t));
}

// D = A * B (+ D when accumulating), or D = reduce(A) (+ D) when B is null.
cutensorStatus_t pairwise(cutensorHandle_t h, cudaDataType_t type, cutensorComputeDescriptor_t compute,
                          const View& a, const View* b, const View& d, bool accumulate, cudaStream_t stream) {
  TensorGuard da, db, dd;
  if (cutensorStatus_t s = make_desc(h, a, type, da)) return s;
  if (b)
    if (cutensorStatus_t s = make_desc(h, *b, type, db)) return s;
  if (cutensorStatus_t s = make_desc(h, d, type, dd)) return s;
  const bool cx = is_complex(type);
  auto opa = a.conj && cx ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
  auto opb = b && b->conj && cx ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
  OpGuard op;
  cutensorStatus_t s =
      b ? cutensorCreateContraction(h, &op.d, da.d, a.modes.data(), opa, db.d, b->modes.data(), opb, dd.d,
                                    d.modes.data(), CUTENSOR_OP_IDENTITY, dd.d, d.modes.data(), compute)
        : cutensorCreateReduction(h, &op.d, da.d, a.modes.data(), opa, dd.d, d.modes.data(), CUTENSOR_OP_IDENTITY,
                                  dd.d, d.modes.data(), CUTENSOR_OP_ADD, compute);
  if (s) return s;
  cudaDataType_t st = CUDA_R_32F;
  cutensorOperationDescriptorGetAttribute(h, op.d, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, &st, sizeof st);
  PrefGuard pref;
  if ((s = cutensorCreatePlanPreference(h, &pref.p, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE))) return s;
  PlanGuard plan;
  if ((s = cutensorCreatePlan(h, &plan.p, op.d, pref.p, 0))) return s;
  Scalar one(st, 1.0), beta(st, accumulate ? 1.0 : 0.0);
  void* dp = const_cast<void*>(d.ptr);
  return b ? cutensorContract(h, plan.p, one.b, a.ptr, b->ptr, beta.b, dp, dp, nullptr, 0, stream)
           : cutensorReduce(h, plan.p, one.b, a.ptr, beta.b, dp, dp, nullptr, 0, stream);
}

// Device memory for a call's scratch: the caller's workspace, or the memory
// handler's pool when the workspace leaves it to the pool.
struct Scratch {
  Handle* h = nullptr;
  void* ptr = nullptr;
  size_t size = 0;
  bool pooled = false;
  cudaStream_t stream = nullptr;
  ~Scratch() {
    if (pooled && ptr) h->mem.device_free(h->mem.ctx, ptr, size, stream);
  }
};

Status get_scratch(Handle* h, const Workspace* w, int64_t need, cudaStream_t stream, Scratch& s, const char* api) {
  s.h = h;
  s.stream = stream;
  void* ptr = w ? w->ptr[CUTENSORNET_MEMSPACE_DEVICE][CUTENSORNET_WORKSPACE_SCRATCH] : nullptr;
  const int64_t size = w ? w->size[CUTENSORNET_MEMSPACE_DEVICE][CUTENSORNET_WORKSPACE_SCRATCH] : 0;
  if (ptr) {
    if (size < need) return fail(CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE, api, "the scratch workspace is too small");
    s.ptr = ptr;
    s.size = (size_t)size;
    return CUTENSORNET_STATUS_SUCCESS;
  }
  if (need == 0) return CUTENSORNET_STATUS_SUCCESS;
  if (!h->has_mem)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "no scratch workspace was provided and no memory handler is set");
  const size_t want = (size_t)std::max<int64_t>(need, size > 0 ? size : 0);
  if (h->mem.device_alloc(h->mem.ctx, &s.ptr, want, stream) != 0 || !s.ptr)
    return fail(CUTENSORNET_STATUS_DEVICE_ALLOCATOR_ERROR, api, "the memory handler could not allocate the workspace");
  s.size = want;
  s.pooled = true;
  return CUTENSORNET_STATUS_SUCCESS;
}

// Contracts the slices `ids` of network `n` along `p` into the output.
Status contract_slices(Handle* h, Network& n, const Path& p, const std::vector<const void*>& in, void* out,
                       bool accumulate, const Workspace* w, const std::vector<int64_t>& ids, cudaStream_t stream,
                       const char* api) {
  const auto ext = extent_map(n);
  const size_t eb = elem_bytes(n.type);
  PathCost pc;
  if (!cost_of(n, p.steps, ext, pc)) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the contraction path is invalid");
  if (n.in.size() > 1 && p.steps.size() + 1 != n.in.size())
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the contraction path does not contract the whole network");
  Scratch scratch;
  if (Status s = get_scratch(h, w, scratch_need(n, p), stream, scratch, api)) return s;
  const cutensorComputeDescriptor_t compute = to_cutensor(n.compute);
  // Which output blocks a slice writes: a sliced output mode splits the
  // output, a sliced contracted mode makes slices add into the same block.
  std::set<std::vector<int64_t>> written;
  for (int64_t id : ids) {
    const SliceView sv = slice_view_of(p, ext, id);
    std::vector<int64_t> block;
    for (int32_t m : n.out.modes) {
      auto it = sv.range.find(m);
      if (it != sv.range.end()) block.push_back(it->second.first);
    }
    const bool acc = accumulate || written.count(block);
    written.insert(block);
    std::vector<View> live;
    for (size_t i = 0; i < n.in.size(); ++i) live.push_back(view_of(n.in[i], in[i], sv, eb));
    NetTensor ot = n.out;
    View ov = view_of(ot, out, sv, eb);
    ov.conj = false;
    if (n.in.size() == 1) {
      if (cutensorStatus_t s = pairwise(h->ct, n.type, compute, live[0], nullptr, ov, acc, stream))
        return fail(from_cutensor(s), api, "a reduction in cuTENSOR failed");
      continue;
    }
    char* base = static_cast<char*>(scratch.ptr);
    for (size_t st = 0; st < p.steps.size(); ++st) {
      int32_t i = p.steps[st].first, j = p.steps[st].second;
      if (i > j) std::swap(i, j);
      View r;
      const bool last = st + 1 == p.steps.size();
      if (last) {
        r = ov;
      } else {
        r.modes = pc.inter[st];
        for (int32_t m : r.modes) {
          auto it = sv.range.find(m);
          r.ext.push_back(it == sv.range.end() ? ext.at(m) : it->second.second - it->second.first);
        }
        r.str = packed_strides(r.ext);
        r.ptr = base;
        base += align_up(volume(r.ext) * (int64_t)eb);
      }
      if (cutensorStatus_t s = pairwise(h->ct, n.type, compute, live[i], &live[j], r, last && acc, stream))
        return fail(from_cutensor(s), api, "a pairwise contraction in cuTENSOR failed");
      live.erase(live.begin() + j);
      live.erase(live.begin() + i);
      live.push_back(r);
    }
  }
  return CUTENSORNET_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

// ---- library management ----

size_t cutensornetGetVersion(void) { return CUTENSORNET_VERSION; }

size_t cutensornetGetCudartVersion(void) {
  int v = 0;
  cudaRuntimeGetVersion(&v);
  cudaGetLastError();
  return (size_t)v;
}

const char* cutensornetGetErrorString(cutensornetStatus_t error) {
  switch (error) {
    case CUTENSORNET_STATUS_SUCCESS: return "CUTENSORNET_STATUS_SUCCESS";
    case CUTENSORNET_STATUS_NOT_INITIALIZED: return "CUTENSORNET_STATUS_NOT_INITIALIZED";
    case CUTENSORNET_STATUS_ALLOC_FAILED: return "CUTENSORNET_STATUS_ALLOC_FAILED";
    case CUTENSORNET_STATUS_INVALID_VALUE: return "CUTENSORNET_STATUS_INVALID_VALUE";
    case CUTENSORNET_STATUS_ARCH_MISMATCH: return "CUTENSORNET_STATUS_ARCH_MISMATCH";
    case CUTENSORNET_STATUS_MAPPING_ERROR: return "CUTENSORNET_STATUS_MAPPING_ERROR";
    case CUTENSORNET_STATUS_EXECUTION_FAILED: return "CUTENSORNET_STATUS_EXECUTION_FAILED";
    case CUTENSORNET_STATUS_INTERNAL_ERROR: return "CUTENSORNET_STATUS_INTERNAL_ERROR";
    case CUTENSORNET_STATUS_NOT_SUPPORTED: return "CUTENSORNET_STATUS_NOT_SUPPORTED";
    case CUTENSORNET_STATUS_LICENSE_ERROR: return "CUTENSORNET_STATUS_LICENSE_ERROR";
    case CUTENSORNET_STATUS_CUBLAS_ERROR: return "CUTENSORNET_STATUS_CUBLAS_ERROR";
    case CUTENSORNET_STATUS_CUDA_ERROR: return "CUTENSORNET_STATUS_CUDA_ERROR";
    case CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE: return "CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE";
    case CUTENSORNET_STATUS_INSUFFICIENT_DRIVER: return "CUTENSORNET_STATUS_INSUFFICIENT_DRIVER";
    case CUTENSORNET_STATUS_IO_ERROR: return "CUTENSORNET_STATUS_IO_ERROR";
    case CUTENSORNET_STATUS_CUTENSOR_VERSION_MISMATCH: return "CUTENSORNET_STATUS_CUTENSOR_VERSION_MISMATCH";
    case CUTENSORNET_STATUS_NO_DEVICE_ALLOCATOR: return "CUTENSORNET_STATUS_NO_DEVICE_ALLOCATOR";
    case CUTENSORNET_STATUS_ALL_HYPER_SAMPLES_FAILED: return "CUTENSORNET_STATUS_ALL_HYPER_SAMPLES_FAILED";
    case CUTENSORNET_STATUS_CUSOLVER_ERROR: return "CUTENSORNET_STATUS_CUSOLVER_ERROR";
    case CUTENSORNET_STATUS_DEVICE_ALLOCATOR_ERROR: return "CUTENSORNET_STATUS_DEVICE_ALLOCATOR_ERROR";
    case CUTENSORNET_STATUS_DISTRIBUTED_FAILURE: return "CUTENSORNET_STATUS_DISTRIBUTED_FAILURE";
    case CUTENSORNET_STATUS_INTERRUPTED: return "CUTENSORNET_STATUS_INTERRUPTED";
    case CUTENSORNET_STATUS_CUTENSOR_ERROR: return "CUTENSORNET_STATUS_CUTENSOR_ERROR";
    default: return "<unknown>";
  }
}

const char* cutensornetGetLastError(void) { return t_last_error.c_str(); }

cutensornetStatus_t cutensornetCreate(cutensornetHandle_t* handle) {
  if (!handle) return fail(CUTENSORNET_STATUS_INVALID_VALUE, "cutensornetCreate", "handle must not be null");
  auto h = std::make_unique<Handle>();
  cudaGetDevice(&h->device);
  cudaGetLastError();
  if (cutensorCreate(&h->ct) != CUTENSOR_STATUS_SUCCESS)
    return fail(CUTENSORNET_STATUS_CUTENSOR_ERROR, "cutensornetCreate", "cutensorCreate failed");
  *handle = h.release();
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetDestroy(cutensornetHandle_t handle) {
  Handle* h = handle_of(handle);
  if (!h) return CUTENSORNET_STATUS_INVALID_VALUE;
  if (h->solver) cusolverDnDestroy(h->solver);
  cutensorDestroy(h->ct);
  h->magic = 0;
  delete h;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetGetDeviceMemHandler(const cutensornetHandle_t handle,
                                                   cutensornetDeviceMemHandler_t* devMemHandler) {
  Handle* h = handle_of(handle);
  if (!h) return CUTENSORNET_STATUS_NOT_INITIALIZED;
  if (!devMemHandler) return CUTENSORNET_STATUS_INVALID_VALUE;
  if (!h->has_mem) return CUTENSORNET_STATUS_NO_DEVICE_ALLOCATOR;
  *devMemHandler = h->mem;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetSetDeviceMemHandler(cutensornetHandle_t handle,
                                                   const cutensornetDeviceMemHandler_t* devMemHandler) {
  Handle* h = handle_of(handle);
  if (!h) return CUTENSORNET_STATUS_NOT_INITIALIZED;
  if (!devMemHandler || !devMemHandler->device_alloc || !devMemHandler->device_free)
    return CUTENSORNET_STATUS_INVALID_VALUE;
  h->mem = *devMemHandler;
  h->has_mem = true;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetLoggerSetCallback(cutensornetLoggerCallback_t callback) {
  std::lock_guard<std::mutex> g(log_mu());
  logger().callback = callback;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetLoggerSetCallbackData(cutensornetLoggerCallbackData_t callback, void* userData) {
  std::lock_guard<std::mutex> g(log_mu());
  logger().callback_data = callback;
  logger().user = userData;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetLoggerSetFile(FILE* file) {
  std::lock_guard<std::mutex> g(log_mu());
  Logger& l = logger();
  if (l.own_file && l.file) std::fclose(l.file);
  l.file = file;
  l.own_file = false;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetLoggerOpenFile(const char* logFile) {
  if (!logFile) return CUTENSORNET_STATUS_INVALID_VALUE;
  FILE* f = std::fopen(logFile, "w");
  if (!f) return CUTENSORNET_STATUS_INVALID_VALUE;
  std::lock_guard<std::mutex> g(log_mu());
  Logger& l = logger();
  if (l.own_file && l.file) std::fclose(l.file);
  l.file = f;
  l.own_file = true;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetLoggerSetLevel(int32_t level) {
  if (level < 0 || level > 5) return CUTENSORNET_STATUS_INVALID_VALUE;
  std::lock_guard<std::mutex> g(log_mu());
  logger().level = level;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetLoggerSetMask(int32_t mask) {
  std::lock_guard<std::mutex> g(log_mu());
  logger().mask = mask;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetLoggerForceDisable(void) {
  std::lock_guard<std::mutex> g(log_mu());
  logger().disabled = true;
  return CUTENSORNET_STATUS_SUCCESS;
}

// ---- tensor descriptors ----

cutensornetStatus_t cutensornetCreateTensorDescriptor(const cutensornetHandle_t handle, int32_t numModes,
                                                      const int64_t extents[], const int64_t strides[],
                                                      const int32_t modeLabels[], cudaDataType_t dataType,
                                                      cutensornetTensorDescriptor_t* tensorDesc) {
  const char* api = "cutensornetCreateTensorDescriptor";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  if (!tensorDesc || numModes < 0 || (numModes > 0 && (!extents || !modeLabels)))
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  if (!network_type(dataType)) return fail(CUTENSORNET_STATUS_NOT_SUPPORTED, api, "unsupported data type");
  auto t = std::make_unique<TensorDesc>();
  for (int32_t i = 0; i < numModes; ++i) {
    if (extents[i] <= 0) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "extents must be positive");
    t->modes.push_back(modeLabels[i]);
    t->extents.push_back(extents[i]);
  }
  t->strides = strides ? std::vector<int64_t>(strides, strides + numModes) : packed_strides(t->extents);
  t->type = dataType;
  *tensorDesc = t.release();
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetGetTensorDetails(const cutensornetHandle_t handle,
                                                const cutensornetTensorDescriptor_t tensorDesc, int32_t* numModes,
                                                size_t* dataSize, int32_t* modeLabels, int64_t* extents,
                                                int64_t* strides) {
  const char* api = "cutensornetGetTensorDetails";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  TensorDesc* t = tensor_of(tensorDesc);
  if (!t || !numModes) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  *numModes = (int32_t)t->modes.size();
  if (dataSize) {
    int64_t span = 1;
    for (size_t i = 0; i < t->extents.size(); ++i) span += (t->extents[i] - 1) * t->strides[i];
    *dataSize = (size_t)(span * (int64_t)elem_bytes(t->type));
  }
  if (modeLabels) std::copy(t->modes.begin(), t->modes.end(), modeLabels);
  if (extents) std::copy(t->extents.begin(), t->extents.end(), extents);
  if (strides) std::copy(t->strides.begin(), t->strides.end(), strides);
  return CUTENSORNET_STATUS_SUCCESS;
}

// A local descriptor answers the distributed attributes as a single rank.
cutensornetStatus_t cutensornetTensorDescriptorGetAttribute(const cutensornetHandle_t handle,
                                                            const cutensornetTensorDescriptor_t tensorDesc,
                                                            cutensornetTensorDescriptorAttributes_t attr,
                                                            void* buffer, size_t sizeInBytes) {
  const char* api = "cutensornetTensorDescriptorGetAttribute";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  TensorDesc* t = tensor_of(tensorDesc);
  if (!t || !buffer) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  const size_t n = t->modes.size();
  auto put_array = [&](const std::vector<int64_t>& v) -> Status {
    if (sizeInBytes != n * 8) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
    if (n) std::memcpy(buffer, v.data(), n * 8);
    return CUTENSORNET_STATUS_SUCCESS;
  };
  switch (attr) {
    case CUTENSORNET_TENSOR_DESCRIPTOR_IS_DISTRIBUTED: {
      if (sizeInBytes != 4) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
      const int32_t z = 0;
      std::memcpy(buffer, &z, 4);
      return CUTENSORNET_STATUS_SUCCESS;
    }
    case CUTENSORNET_TENSOR_DESCRIPTOR_ELEMENT_STRIDES: return put_array(t->strides);
    case CUTENSORNET_TENSOR_DESCRIPTOR_BLOCK_SIZES: return put_array(std::vector<int64_t>(n, 0));
    case CUTENSORNET_TENSOR_DESCRIPTOR_BLOCK_STRIDES: return put_array(std::vector<int64_t>(n, 0));
    case CUTENSORNET_TENSOR_DESCRIPTOR_NRANKS_PER_MODE: return put_array(std::vector<int64_t>(n, 1));
    case CUTENSORNET_TENSOR_DESCRIPTOR_LOCAL_EXTENTS: return put_array(t->extents);
    case CUTENSORNET_TENSOR_DESCRIPTOR_LOCAL_DATA_SIZE: {
      if (sizeInBytes != sizeof(size_t)) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
      const size_t b = (size_t)volume(t->extents) * elem_bytes(t->type);
      std::memcpy(buffer, &b, sizeof b);
      return CUTENSORNET_STATUS_SUCCESS;
    }
    default: return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "unknown attribute");
  }
}

cutensornetStatus_t cutensornetDestroyTensorDescriptor(cutensornetTensorDescriptor_t tensorDesc) {
  TensorDesc* t = tensor_of(tensorDesc);
  if (!t) return CUTENSORNET_STATUS_INVALID_VALUE;
  t->magic = 0;
  delete t;
  return CUTENSORNET_STATUS_SUCCESS;
}

}  // extern "C"

extern "C" {

// ---- networks ----

cutensornetStatus_t cutensornetCreateNetworkDescriptor(
    const cutensornetHandle_t handle, int32_t numInputs, const int32_t numModesIn[],
    const int64_t* const extentsIn[], const int64_t* const stridesIn[], const int32_t* const modesIn[],
    const cutensornetTensorQualifiers_t qualifiersIn[], int32_t numModesOut, const int64_t extentsOut[],
    const int64_t stridesOut[], const int32_t modesOut[], cudaDataType_t dataType,
    cutensornetComputeType_t computeType, cutensornetNetworkDescriptor_t* networkDesc) {
  const char* api = "cutensornetCreateNetworkDescriptor";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  if (!networkDesc || numInputs <= 0 || !numModesIn || !extentsIn || !modesIn)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  if (!network_type(dataType)) return fail(CUTENSORNET_STATUS_NOT_SUPPORTED, api, "unsupported data type");
  if (!compute_ok(dataType, computeType))
    return fail(CUTENSORNET_STATUS_NOT_SUPPORTED, api, "unsupported data and compute type combination");
  auto n = std::make_unique<Network>();
  n->legacy = true;
  n->type = dataType;
  n->type_set = true;
  n->compute = computeType;
  n->compute_set = true;
  std::map<int32_t, int64_t> ext;
  for (int32_t i = 0; i < numInputs; ++i) {
    NetTensor t;
    const int32_t nm = numModesIn[i];
    if (nm < 0 || (nm > 0 && (!extentsIn[i] || !modesIn[i])))
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid input tensor");
    for (int32_t j = 0; j < nm; ++j) {
      const int32_t m = modesIn[i][j];
      const int64_t e = extentsIn[i][j];
      if (e <= 0) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "extents must be positive");
      auto it = ext.find(m);
      if (it != ext.end() && it->second != e)
        return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "a mode has two different extents");
      ext[m] = e;
      t.modes.push_back(m);
      t.extents.push_back(e);
    }
    if (stridesIn && stridesIn[i]) t.strides.assign(stridesIn[i], stridesIn[i] + nm);
    if (qualifiersIn) t.q = qualifiersIn[i];
    n->in.push_back(std::move(t));
  }
  if (numModesOut >= 0) {
    // The output's extents are required with its modes (measured).
    if (numModesOut > 0 && !modesOut) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "modesOut must not be null");
    if (numModesOut > 0 && !extentsOut) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "extentsOut must not be null");
    n->out_set = true;
    for (int32_t j = 0; j < numModesOut; ++j) {
      auto it = ext.find(modesOut[j]);
      if (it == ext.end()) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "an output mode is in no input");
      if (extentsOut && extentsOut[j] != it->second)
        return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "an output extent does not match its input");
      n->out.modes.push_back(modesOut[j]);
      n->out.extents.push_back(it->second);
    }
    if (stridesOut) n->out.strides.assign(stridesOut, stridesOut + numModesOut);
  }
  fill_output(*n);
  *networkDesc = n.release();
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetCreateNetwork(const cutensornetHandle_t handle,
                                             cutensornetNetworkDescriptor_t* networkDesc) {
  const char* api = "cutensornetCreateNetwork";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  if (!networkDesc) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "networkDesc must not be null");
  *networkDesc = new Network;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetDestroyNetwork(cutensornetNetworkDescriptor_t networkDesc) {
  Network* n = net_of(networkDesc);
  if (!n) return CUTENSORNET_STATUS_INVALID_VALUE;
  n->magic = 0;
  delete n;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetDestroyNetworkDescriptor(cutensornetNetworkDescriptor_t networkDesc) {
  return cutensornetDestroyNetwork(networkDesc);
}

cutensornetStatus_t cutensornetNetworkAppendTensor(const cutensornetHandle_t handle,
                                                   cutensornetNetworkDescriptor_t networkDesc, int32_t numModes,
                                                   const int64_t extents[], const int32_t modeLabels[],
                                                   const cutensornetTensorQualifiers_t* const qualifiers,
                                                   cudaDataType_t dataType, int64_t* tensorId) {
  const char* api = "cutensornetNetworkAppendTensor";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  if (!n || n->legacy || numModes < 0 || (numModes > 0 && (!extents || !modeLabels)))
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  if (!network_type(dataType)) return fail(CUTENSORNET_STATUS_NOT_SUPPORTED, api, "unsupported data type");
  if (n->type_set && dataType != n->type)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "every tensor of a network must have the same data type");
  auto ext = extent_map(*n);
  NetTensor t;
  for (int32_t j = 0; j < numModes; ++j) {
    if (extents[j] <= 0) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "extents must be positive");
    auto it = ext.find(modeLabels[j]);
    if (it != ext.end() && it->second != extents[j])
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "a mode has two different extents");
    t.modes.push_back(modeLabels[j]);
    t.extents.push_back(extents[j]);
  }
  if (qualifiers) t.q = *qualifiers;
  n->type = dataType;
  n->type_set = true;
  if (!n->compute_set) n->compute = default_compute(dataType);
  n->in.push_back(std::move(t));
  n->prepared = false;
  n->has_path = false;
  if (!n->out_set) fill_output(*n);
  if (tensorId) *tensorId = (int64_t)n->in.size() - 1;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetNetworkSetOutputTensor(const cutensornetHandle_t handle,
                                                      cutensornetNetworkDescriptor_t networkDesc, int32_t numModes,
                                                      const int32_t modeLabels[], cudaDataType_t dataType) {
  const char* api = "cutensornetNetworkSetOutputTensor";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  if (!n || n->legacy || numModes < 0 || (numModes > 0 && !modeLabels))
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  if (n->type_set && dataType != n->type)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the output must have the network's data type");
  auto ext = extent_map(*n);
  NetTensor o;
  for (int32_t j = 0; j < numModes; ++j) {
    auto it = ext.find(modeLabels[j]);
    if (it == ext.end()) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "an output mode is in no input");
    o.modes.push_back(modeLabels[j]);
    o.extents.push_back(it->second);
  }
  n->out = std::move(o);
  n->out_set = true;
  n->prepared = false;
  n->has_path = false;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetNetworkSetInputTensorMemory(const cutensornetHandle_t handle,
                                                           cutensornetNetworkDescriptor_t networkDesc,
                                                           int64_t tensorId, const void* const buffer,
                                                           const int64_t strides[]) {
  const char* api = "cutensornetNetworkSetInputTensorMemory";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  if (!n || tensorId < 0 || tensorId >= (int64_t)n->in.size())
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid tensor id");
  NetTensor& t = n->in[(size_t)tensorId];
  t.data = buffer;
  if (strides) t.strides.assign(strides, strides + t.modes.size());
  else t.strides.clear();
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetNetworkSetOutputTensorMemory(const cutensornetHandle_t handle,
                                                            cutensornetNetworkDescriptor_t networkDesc,
                                                            void* const buffer, const int64_t strides[]) {
  const char* api = "cutensornetNetworkSetOutputTensorMemory";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  if (!n) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid network");
  fill_output(*n);
  n->out_data = buffer;
  if (strides) n->out.strides.assign(strides, strides + n->out.modes.size());
  else n->out.strides.clear();
  return CUTENSORNET_STATUS_SUCCESS;
}

}  // extern "C"

namespace {

Status get_id_list(const std::vector<NetTensor>& in, int which, void* buffer, size_t size, const char* api) {
  if (size != sizeof(cutensornetTensorIDList_t)) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
  auto* l = static_cast<cutensornetTensorIDList_t*>(buffer);
  int32_t k = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    const int32_t flag = which == 0 ? in[i].q.isConstant : which == 1 ? in[i].q.isConjugate : in[i].q.requiresGradient;
    if (!flag) continue;
    if (l->data && k < l->numTensors) l->data[k] = (int32_t)i;
    ++k;
  }
  l->numTensors = k;
  return CUTENSORNET_STATUS_SUCCESS;
}

int32_t count_flag(const std::vector<NetTensor>& in, int which) {
  int32_t k = 0;
  for (const NetTensor& t : in) k += (which == 0 ? t.q.isConstant : which == 1 ? t.q.isConjugate : t.q.requiresGradient) != 0;
  return k;
}

}  // namespace

extern "C" {

cutensornetStatus_t cutensornetNetworkGetAttribute(const cutensornetHandle_t handle,
                                                   const cutensornetNetworkDescriptor_t networkDesc,
                                                   cutensornetNetworkAttributes_t attr, void* buffer,
                                                   size_t sizeInBytes) {
  const char* api = "cutensornetNetworkGetAttribute";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  if (!n || !buffer) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  auto put32 = [&](int32_t v) -> Status {
    if (sizeInBytes != 4) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
    std::memcpy(buffer, &v, 4);
    return CUTENSORNET_STATUS_SUCCESS;
  };
  switch (attr) {
    case CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_CONSTANT: return put32(count_flag(n->in, 0));
    case CUTENSORNET_NETWORK_INPUT_TENSORS_CONSTANT: return get_id_list(n->in, 0, buffer, sizeInBytes, api);
    case CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_CONJUGATED: return put32(count_flag(n->in, 1));
    case CUTENSORNET_NETWORK_INPUT_TENSORS_CONJUGATED: return get_id_list(n->in, 1, buffer, sizeInBytes, api);
    case CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_REQUIRE_GRAD: return put32(count_flag(n->in, 2));
    case CUTENSORNET_NETWORK_INPUT_TENSORS_REQUIRE_GRAD: return get_id_list(n->in, 2, buffer, sizeInBytes, api);
    case CUTENSORNET_NETWORK_COMPUTE_TYPE: return put32((int32_t)n->compute);
    default: return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "unknown attribute");
  }
}

cutensornetStatus_t cutensornetNetworkSetAttribute(const cutensornetHandle_t handle,
                                                   cutensornetNetworkDescriptor_t networkDesc,
                                                   cutensornetNetworkAttributes_t attr, const void* const buffer,
                                                   size_t sizeInBytes) {
  const char* api = "cutensornetNetworkSetAttribute";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  if (!n || !buffer) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  if (attr == CUTENSORNET_NETWORK_COMPUTE_TYPE) {
    if (sizeInBytes != 4) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
    int32_t c;
    std::memcpy(&c, buffer, 4);
    if (n->type_set && !compute_ok(n->type, (cutensornetComputeType_t)c))
      return fail(CUTENSORNET_STATUS_NOT_SUPPORTED, api, "unsupported data and compute type combination");
    n->compute = (cutensornetComputeType_t)c;
    n->compute_set = true;
    n->prepared = false;
    return CUTENSORNET_STATUS_SUCCESS;
  }
  const int which = attr == CUTENSORNET_NETWORK_INPUT_TENSORS_CONSTANT    ? 0
                    : attr == CUTENSORNET_NETWORK_INPUT_TENSORS_CONJUGATED ? 1
                    : attr == CUTENSORNET_NETWORK_INPUT_TENSORS_REQUIRE_GRAD ? 2
                                                                            : -1;
  if (which < 0) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "unknown or read-only attribute");
  if (sizeInBytes != sizeof(cutensornetTensorIDList_t)) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
  const auto* l = static_cast<const cutensornetTensorIDList_t*>(buffer);
  if (which == 2 && l->numTensors != 0)
    return refuse(api, "gradients are not implemented, so no tensor can require one");
  for (NetTensor& t : n->in) (which == 0 ? t.q.isConstant : t.q.isConjugate) = l->numTensors < 0 ? 1 : 0;
  for (int32_t k = 0; k < l->numTensors; ++k) {
    if (!l->data || l->data[k] < 0 || l->data[k] >= (int32_t)n->in.size())
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid tensor id");
    (which == 0 ? n->in[l->data[k]].q.isConstant : n->in[l->data[k]].q.isConjugate) = 1;
  }
  n->prepared = false;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetGetOutputTensorDetails(const cutensornetHandle_t handle,
                                                      const cutensornetNetworkDescriptor_t networkDesc,
                                                      int32_t* numModes, size_t* dataSize, int32_t* modeLabels,
                                                      int64_t* extents, int64_t* strides) {
  const char* api = "cutensornetGetOutputTensorDetails";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  if (!n || !numModes) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  fill_output(*n);
  const NetTensor& o = n->out;
  const std::vector<int64_t> st = o.strides.empty() ? packed_strides(o.extents) : o.strides;
  *numModes = (int32_t)o.modes.size();
  if (dataSize) *dataSize = (size_t)volume(o.extents) * elem_bytes(n->type);
  if (modeLabels) std::copy(o.modes.begin(), o.modes.end(), modeLabels);
  if (extents) std::copy(o.extents.begin(), o.extents.end(), extents);
  if (strides) std::copy(st.begin(), st.end(), strides);
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetGetOutputTensorDescriptor(const cutensornetHandle_t handle,
                                                         const cutensornetNetworkDescriptor_t networkDesc,
                                                         cutensornetTensorDescriptor_t* outputTensorDesc) {
  const char* api = "cutensornetGetOutputTensorDescriptor";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  if (!n || !outputTensorDesc) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  fill_output(*n);
  auto t = std::make_unique<TensorDesc>();
  t->modes = n->out.modes;
  t->extents = n->out.extents;
  t->strides = n->out.strides.empty() ? packed_strides(t->extents) : n->out.strides;
  t->type = n->type;
  *outputTensorDesc = t.release();
  return CUTENSORNET_STATUS_SUCCESS;
}

}  // extern "C"

namespace {

// ---- paths, costs, slicing ----

int64_t slices_of(const Path& p, const std::map<int32_t, int64_t>& ext) {
  int64_t n = 1;
  for (auto& [m, se] : p.slices) n *= (ext.at(m) + se - 1) / se;
  return n;
}

// A slice's index ranges: the sliced extent `se` of a mode of extent E cuts
// it into ceil(E / se) pieces of se indices (the last may be shorter). This
// is what NVIDIA's library reports: slicing a mode of extent 8 completely
// shows as (mode, 1) and 8 slices; (mode, 2) as 4 slices.
SliceView slice_view_of(const Path& p, const std::map<int32_t, int64_t>& ext, int64_t id) {
  SliceView v;
  for (auto& [mode, se] : p.slices) {
    const int64_t e = ext.at(mode), count = (e + se - 1) / se, k = id % count;
    id /= count;
    v.range[mode] = {k * se, std::min(e, (k + 1) * se)};
  }
  return v;
}

std::map<int32_t, int64_t> sliced_extents(const Path& p, std::map<int32_t, int64_t> ext) {
  for (auto& [m, se] : p.slices) ext[m] = std::min(ext[m], se);
  return ext;
}

int64_t scratch_need(const Network& n, const Path& p) {
  const auto ext = extent_map(n);
  PathCost pc;
  if (!cost_of(n, p.steps, ext, pc)) return 0;
  const auto se = sliced_extents(p, ext);
  int64_t b = 0;
  for (size_t s = 0; s + 1 < pc.inter.size(); ++s) b += align_up((int64_t)mode_volume(pc.inter[s], se) * (int64_t)elem_bytes(n.type));
  return b;
}

// Fills `info` for path `p` on network `n`: FLOP counts before and after
// slicing (each slice repeats the path; all but one add into the output),
// the largest tensor (intermediates and output, sliced), intermediate modes.
void describe(const Network& n, const Path& p, Info& info) {
  const auto ext = extent_map(n);
  info.num_inputs = (int32_t)n.in.size();
  info.path = p;
  info.has_path = true;
  PathCost full, sl;
  cost_of(n, p.steps, ext, full);
  const auto se = sliced_extents(p, ext);
  cost_of(n, p.steps, se, sl);
  const int64_t ns = slices_of(p, ext);
  info.num_slices = ns;
  info.phase1_flops = full.flops;
  info.inter_modes = full.inter;
  double out_vol = mode_volume(n.out.modes, se);
  info.flops = sl.flops * (double)ns + (double)(ns - 1) * out_vol;
  info.largest = std::max(sl.largest, out_vol);
  if (n.in.size() == 1) info.largest = out_vol;
  info.overhead = info.phase1_flops > 0 ? info.flops / info.phase1_flops : 1;
}

int32_t config_value(const Config* c, int attr, int32_t dflt) {
  if (!c) return dflt;
  auto it = c->v.find(attr);
  return it == c->v.end() ? dflt : it->second;
}

// Slicing: while the intermediates do not fit in `limit` bytes, or there are
// fewer than the configured minimum of slices, slice completely the
// contracted mode (one no output has) that shrinks the scratch most (ties:
// the larger extent, then the lower label).
bool choose_slicing(const Network& n, Path& p, uint64_t limit, int32_t min_slices) {
  const auto ext = extent_map(n);
  std::set<int32_t> candidates;
  for (const NetTensor& t : n.in)
    for (int32_t m : t.modes)
      if (!contains(n.out.modes, m) && ext.at(m) > 1) candidates.insert(m);
  for (;;) {
    const int64_t need = scratch_need(n, p);
    if ((uint64_t)need <= limit && slices_of(p, ext) >= min_slices) return true;
    int32_t best = 0;
    int64_t best_need = -1, best_ext = 0;
    for (int32_t m : candidates) {
      bool taken = false;
      for (auto& s : p.slices) taken = taken || s.first == m;
      if (taken) continue;
      Path q = p;
      q.slices.emplace_back(m, 1);
      const int64_t nq = scratch_need(n, q);
      if (best_need < 0 || nq < best_need || (nq == best_need && ext.at(m) > best_ext)) {
        best = m, best_need = nq, best_ext = ext.at(m);
      }
    }
    if (best_need < 0) return (uint64_t)need <= limit;  // nothing left to slice
    p.slices.emplace_back(best, 1);
  }
}

// ---- packing an optimizer info: our own format ----
const char kPackMagic[8] = {'V', 'G', 'P', 'U', 'T', 'N', 'I', '1'};

std::vector<uint8_t> pack(const Info& info) {
  std::vector<uint8_t> b(kPackMagic, kPackMagic + 8);
  auto put = [&](const void* p, size_t n) { b.insert(b.end(), (const uint8_t*)p, (const uint8_t*)p + n); };
  const int32_t ni = info.num_inputs, ns = (int32_t)info.path.steps.size(), nsl = (int32_t)info.path.slices.size();
  put(&ni, 4);
  put(&ns, 4);
  put(&nsl, 4);
  for (auto& s : info.path.steps) {
    const int32_t ij[2] = {s.first, s.second};
    put(ij, 8);
  }
  for (auto& s : info.path.slices) {
    put(&s.first, 4);
    put(&s.second, 8);
  }
  return b;
}

bool unpack(const void* buf, size_t size, Info& info) {
  const uint8_t* p = static_cast<const uint8_t*>(buf);
  if (!p || size < 20 || std::memcmp(p, kPackMagic, 8) != 0) return false;
  int32_t ni, ns, nsl;
  std::memcpy(&ni, p + 8, 4);
  std::memcpy(&ns, p + 12, 4);
  std::memcpy(&nsl, p + 16, 4);
  if (ns < 0 || nsl < 0 || size != 20 + (size_t)ns * 8 + (size_t)nsl * 12) return false;
  Path path;
  size_t at = 20;
  for (int32_t i = 0; i < ns; ++i, at += 8) {
    int32_t ij[2];
    std::memcpy(ij, p + at, 8);
    path.steps.emplace_back(ij[0], ij[1]);
  }
  for (int32_t i = 0; i < nsl; ++i, at += 12) {
    int32_t m;
    int64_t e;
    std::memcpy(&m, p + at, 4);
    std::memcpy(&e, p + at + 4, 8);
    path.slices.emplace_back(m, e);
  }
  info.num_inputs = ni;
  info.path = path;
  info.has_path = true;
  return true;
}

}  // namespace

extern "C" {

// ---- workspaces ----
//
// This library's needs are its own: the scratch is the sliced intermediates
// of the path (or a decomposition's matrices and cuSOLVER's work), the cache
// and host memory are never used. Every preference asks for the same size.

cutensornetStatus_t cutensornetCreateWorkspaceDescriptor(const cutensornetHandle_t handle,
                                                         cutensornetWorkspaceDescriptor_t* workDesc) {
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, "cutensornetCreateWorkspaceDescriptor", "invalid handle");
  if (!workDesc) return fail(CUTENSORNET_STATUS_INVALID_VALUE, "cutensornetCreateWorkspaceDescriptor", "workDesc must not be null");
  *workDesc = new Workspace;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetDestroyWorkspaceDescriptor(cutensornetWorkspaceDescriptor_t workDesc) {
  Workspace* w = work_of(workDesc);
  if (!w) return CUTENSORNET_STATUS_INVALID_VALUE;
  w->magic = 0;
  delete w;
  return CUTENSORNET_STATUS_SUCCESS;
}

}  // extern "C"

namespace {

// At least one aligned block, so that a caller who allocates what is asked
// for never allocates nothing (cuQuantum Python allocates the recommended
// size and passes the pointer).
void set_scratch_need(Workspace* w, int64_t need) {
  need = std::max<int64_t>(need, kAlign);
  for (auto& pref : w->need)
    for (auto& mem : pref)
      for (int64_t& k : mem) k = 0;
  for (int p = 0; p < 3; ++p) w->need[p][CUTENSORNET_MEMSPACE_DEVICE][CUTENSORNET_WORKSPACE_SCRATCH] = need;
}

bool valid_ws_enums(int pref, int mem, int kind) {
  return pref >= 0 && pref <= 2 && (mem == 0 || mem == 1) && (kind == 0 || kind == 1);
}

const Path* path_for(Network* n, Info* info) {
  if (info && info->has_path) return &info->path;
  if (n && n->has_path) return &n->path;
  return nullptr;
}

}  // namespace

extern "C" {

cutensornetStatus_t cutensornetWorkspaceComputeContractionSizes(const cutensornetHandle_t handle,
                                                                const cutensornetNetworkDescriptor_t networkDesc,
                                                                const cutensornetContractionOptimizerInfo_t optimizerInfo,
                                                                cutensornetWorkspaceDescriptor_t workDesc) {
  const char* api = "cutensornetWorkspaceComputeContractionSizes";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  Info* info = info_of(optimizerInfo);
  Workspace* w = work_of(workDesc);
  if (!n || !w) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  const Path* p = path_for(n, info);
  if (!p) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the network has no contraction path yet");
  fill_output(*n);
  set_scratch_need(w, scratch_need(*n, *p));
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetWorkspaceComputeSizes(const cutensornetHandle_t handle,
                                                     const cutensornetNetworkDescriptor_t networkDesc,
                                                     const cutensornetContractionOptimizerInfo_t optimizerInfo,
                                                     cutensornetWorkspaceDescriptor_t workDesc) {
  return cutensornetWorkspaceComputeContractionSizes(handle, networkDesc, optimizerInfo, workDesc);
}

cutensornetStatus_t cutensornetWorkspaceGetMemorySize(const cutensornetHandle_t handle,
                                                      const cutensornetWorkspaceDescriptor_t workDesc,
                                                      cutensornetWorksizePref_t workPref,
                                                      cutensornetMemspace_t memSpace,
                                                      cutensornetWorkspaceKind_t workKind, int64_t* memorySize) {
  const char* api = "cutensornetWorkspaceGetMemorySize";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Workspace* w = work_of(workDesc);
  if (!w || !memorySize || !valid_ws_enums(workPref, memSpace, workKind))
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  *memorySize = w->need[workPref][memSpace][workKind];
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetWorkspaceGetSize(const cutensornetHandle_t handle,
                                                const cutensornetWorkspaceDescriptor_t workDesc,
                                                cutensornetWorksizePref_t workPref, cutensornetMemspace_t memSpace,
                                                uint64_t* workspaceSize) {
  int64_t v = 0;
  cutensornetStatus_t s = cutensornetWorkspaceGetMemorySize(handle, workDesc, workPref, memSpace,
                                                            CUTENSORNET_WORKSPACE_SCRATCH, &v);
  if (s == CUTENSORNET_STATUS_SUCCESS) {
    if (!workspaceSize) return CUTENSORNET_STATUS_INVALID_VALUE;
    *workspaceSize = (uint64_t)v;
  }
  return s;
}

cutensornetStatus_t cutensornetWorkspaceSetMemory(const cutensornetHandle_t handle,
                                                  cutensornetWorkspaceDescriptor_t workDesc,
                                                  cutensornetMemspace_t memSpace,
                                                  cutensornetWorkspaceKind_t workKind, void* const memoryPtr,
                                                  int64_t memorySize) {
  const char* api = "cutensornetWorkspaceSetMemory";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Workspace* w = work_of(workDesc);
  if (!w || !valid_ws_enums(0, memSpace, workKind)) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  if (memoryPtr && memorySize < 0)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "a workspace pointer needs a size");
  w->ptr[memSpace][workKind] = memoryPtr;
  w->size[memSpace][workKind] = memorySize;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetWorkspaceSet(const cutensornetHandle_t handle, cutensornetWorkspaceDescriptor_t workDesc,
                                            cutensornetMemspace_t memSpace, void* const workspacePtr,
                                            uint64_t workspaceSize) {
  return cutensornetWorkspaceSetMemory(handle, workDesc, memSpace, CUTENSORNET_WORKSPACE_SCRATCH, workspacePtr,
                                       (int64_t)workspaceSize);
}

cutensornetStatus_t cutensornetWorkspaceGetMemory(const cutensornetHandle_t handle,
                                                  const cutensornetWorkspaceDescriptor_t workDesc,
                                                  cutensornetMemspace_t memSpace,
                                                  cutensornetWorkspaceKind_t workKind, void** memoryPtr,
                                                  int64_t* memorySize) {
  const char* api = "cutensornetWorkspaceGetMemory";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Workspace* w = work_of(workDesc);
  if (!w || !memoryPtr || !memorySize || !valid_ws_enums(0, memSpace, workKind))
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  *memoryPtr = w->ptr[memSpace][workKind];
  *memorySize = w->size[memSpace][workKind];
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetWorkspaceGet(const cutensornetHandle_t handle,
                                            const cutensornetWorkspaceDescriptor_t workDesc,
                                            cutensornetMemspace_t memSpace, void** workspacePtr,
                                            uint64_t* workspaceSize) {
  int64_t v = 0;
  cutensornetStatus_t s =
      cutensornetWorkspaceGetMemory(handle, workDesc, memSpace, CUTENSORNET_WORKSPACE_SCRATCH, workspacePtr, &v);
  if (s == CUTENSORNET_STATUS_SUCCESS) {
    if (!workspaceSize) return CUTENSORNET_STATUS_INVALID_VALUE;
    *workspaceSize = (uint64_t)std::max<int64_t>(v, 0);
  }
  return s;
}

// Nothing is cached, so there is nothing to purge.
cutensornetStatus_t cutensornetWorkspacePurgeCache(const cutensornetHandle_t handle,
                                                   cutensornetWorkspaceDescriptor_t workDesc,
                                                   cutensornetMemspace_t memSpace) {
  if (!handle_of(handle)) return CUTENSORNET_STATUS_NOT_INITIALIZED;
  if (!work_of(workDesc) || (memSpace != 0 && memSpace != 1)) return CUTENSORNET_STATUS_INVALID_VALUE;
  return CUTENSORNET_STATUS_SUCCESS;
}

// ---- the optimizer configuration ----

cutensornetStatus_t cutensornetCreateContractionOptimizerConfig(const cutensornetHandle_t handle,
                                                                cutensornetContractionOptimizerConfig_t* optimizerConfig) {
  const char* api = "cutensornetCreateContractionOptimizerConfig";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  if (!optimizerConfig) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "optimizerConfig must not be null");
  *optimizerConfig = new Config;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetDestroyContractionOptimizerConfig(cutensornetContractionOptimizerConfig_t optimizerConfig) {
  Config* c = config_of(optimizerConfig);
  if (!c) return CUTENSORNET_STATUS_INVALID_VALUE;
  c->magic = 0;
  delete c;
  return CUTENSORNET_STATUS_SUCCESS;
}

}  // extern "C"

namespace {

// Each configuration attribute is an int32_t; its documented default and
// range. The values steer NVIDIA's hyper-optimizer; this library keeps them
// and uses the slicer's (disable, minimum slices).
struct ConfigAttr {
  int attr;
  int32_t dflt, lo, hi;
};
const ConfigAttr kConfigAttrs[] = {
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_PARTITIONS, 8, 2, 30},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_CUTOFF_SIZE, 8, 4, 50},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM, CUTENSORNET_GRAPH_ALGO_KWAY, 0, 1},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_IMBALANCE_FACTOR, 200, 30, 2000},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_ITERATIONS, 60, 1, 500},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_CUTS, 10, 1, 40},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS, 500, 0, INT32_MAX},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_LEAVES, 8, 2, INT32_MAX},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING, 0, 0, 1},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL, CUTENSORNET_MEMORY_MODEL_CUTENSOR, 0, 1},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_FACTOR, 80, 1, 100},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES, 1, 1, INT32_MAX},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_SLICE_FACTOR, 32, 2, INT32_MAX},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES, 0, 0, INT32_MAX},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_THREADS, 1, 1, INT32_MAX},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SIMPLIFICATION_DISABLE_DR, 0, 0, 1},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED, 0, INT32_MIN, INT32_MAX},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE, CUTENSORNET_OPTIMIZER_COST_FLOPS, 0, 1},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_CACHE_REUSE_NRUNS, 0, 0, INT32_MAX},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SMART_OPTION, CUTENSORNET_SMART_OPTION_ENABLED, 0, 1},
    {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GPU_ARCH, 0, INT32_MIN, INT32_MAX},
};

const ConfigAttr* config_attr(int attr) {
  for (const ConfigAttr& a : kConfigAttrs)
    if (a.attr == attr) return &a;
  return nullptr;
}

}  // namespace

extern "C" {

cutensornetStatus_t cutensornetContractionOptimizerConfigGetAttribute(
    const cutensornetHandle_t handle, const cutensornetContractionOptimizerConfig_t optimizerConfig,
    cutensornetContractionOptimizerConfigAttributes_t attr, void* buffer, size_t sizeInBytes) {
  const char* api = "cutensornetContractionOptimizerConfigGetAttribute";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Config* c = config_of(optimizerConfig);
  const ConfigAttr* a = config_attr(attr);
  if (!c || !buffer || !a) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  // A buffer larger than the int32_t is taken (measured).
  if (sizeInBytes < 4) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
  const int32_t v = config_value(c, attr, a->dflt);
  std::memcpy(buffer, &v, 4);
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetContractionOptimizerConfigSetAttribute(
    const cutensornetHandle_t handle, cutensornetContractionOptimizerConfig_t optimizerConfig,
    cutensornetContractionOptimizerConfigAttributes_t attr, const void* buffer, size_t sizeInBytes) {
  const char* api = "cutensornetContractionOptimizerConfigSetAttribute";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Config* c = config_of(optimizerConfig);
  const ConfigAttr* a = config_attr(attr);
  if (!c || !buffer || !a) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  if (sizeInBytes < 4) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
  int32_t v;
  std::memcpy(&v, buffer, 4);
  if (v < a->lo || v > a->hi) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "value out of range");
  c->v[attr] = v;
  return CUTENSORNET_STATUS_SUCCESS;
}

// ---- the optimizer information ----

cutensornetStatus_t cutensornetCreateContractionOptimizerInfo(const cutensornetHandle_t handle,
                                                              const cutensornetNetworkDescriptor_t networkDesc,
                                                              cutensornetContractionOptimizerInfo_t* optimizerInfo) {
  const char* api = "cutensornetCreateContractionOptimizerInfo";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  if (!n || !optimizerInfo) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  auto* i = new Info;
  i->net = n;
  i->num_inputs = (int32_t)n->in.size();
  *optimizerInfo = i;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetDestroyContractionOptimizerInfo(cutensornetContractionOptimizerInfo_t optimizerInfo) {
  Info* i = info_of(optimizerInfo);
  if (!i) return CUTENSORNET_STATUS_INVALID_VALUE;
  i->magic = 0;
  delete i;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetContractionOptimize(const cutensornetHandle_t handle,
                                                   const cutensornetNetworkDescriptor_t networkDesc,
                                                   const cutensornetContractionOptimizerConfig_t optimizerConfig,
                                                   uint64_t workspaceSizeConstraint,
                                                   cutensornetContractionOptimizerInfo_t optimizerInfo) {
  const char* api = "cutensornetContractionOptimize";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  Info* info = info_of(optimizerInfo);
  Config* c = config_of(optimizerConfig);
  if (!n || !info || !c) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  if (n->in.empty()) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the network has no tensors");
  fill_output(*n);
  const auto ext = extent_map(*n);
  Path p;
  p.steps = greedy_path(*n, ext);
  const bool no_slicing = config_value(c, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING, 0) != 0;
  const int32_t min_slices = config_value(c, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES, 1);
  if (!no_slicing && !choose_slicing(*n, p, workspaceSizeConstraint, min_slices))
    return fail(CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE, api, "no slicing fits the workspace constraint");
  if (no_slicing && (uint64_t)scratch_need(*n, p) > workspaceSizeConstraint)
    return fail(CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE, api, "the path does not fit the workspace constraint");
  describe(*n, p, *info);
  n->path = p;
  n->has_path = true;
  n->prepared = false;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetNetworkSetOptimizerInfo(const cutensornetHandle_t handle,
                                                       cutensornetNetworkDescriptor_t networkDesc,
                                                       const cutensornetContractionOptimizerInfo_t optimizerInfo) {
  const char* api = "cutensornetNetworkSetOptimizerInfo";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  Info* info = info_of(optimizerInfo);
  if (!n || !info || !info->has_path) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  n->path = info->path;
  n->has_path = true;
  n->prepared = false;
  return CUTENSORNET_STATUS_SUCCESS;
}

}  // extern "C"

namespace {

// The info's counts are recomputed whenever its path or slicing is set, from
// the network it was created for.
Status info_get(Info* info, int attr, void* buffer, size_t size, const char* api) {
  auto put = [&](const void* v, size_t n) -> Status {
    if (size != n) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
    std::memcpy(buffer, v, n);
    return CUTENSORNET_STATUS_SUCCESS;
  };
  const Path& p = info->path;
  switch (attr) {
    case CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH: {
      if (size != sizeof(cutensornetContractionPath_t)) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
      auto* cp = static_cast<cutensornetContractionPath_t*>(buffer);
      if (cp->data)
        for (size_t i = 0; i < p.steps.size() && (int32_t)i < cp->numContractions; ++i)
          cp->data[i] = cutensornetNodePair_t{p.steps[i].first, p.steps[i].second};
      cp->numContractions = (int32_t)p.steps.size();
      return CUTENSORNET_STATUS_SUCCESS;
    }
    case CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES: return put(&info->num_slices, 8);
    case CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES: {
      const int32_t k = (int32_t)p.slices.size();
      return put(&k, 4);
    }
    case CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_MODE: {
      std::vector<int32_t> v;
      for (auto& s : p.slices) v.push_back(s.first);
      if (size != v.size() * 4) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
      if (!v.empty()) std::memcpy(buffer, v.data(), size);
      return CUTENSORNET_STATUS_SUCCESS;
    }
    case CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_EXTENT: {
      std::vector<int64_t> v;
      for (auto& s : p.slices) v.push_back(s.second);
      if (size != v.size() * 8) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
      if (!v.empty()) std::memcpy(buffer, v.data(), size);
      return CUTENSORNET_STATUS_SUCCESS;
    }
    case CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_CONFIG: {
      if (size != sizeof(cutensornetSlicingConfig_t)) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
      auto* sc = static_cast<cutensornetSlicingConfig_t*>(buffer);
      if (sc->data)
        for (size_t i = 0; i < p.slices.size() && i < sc->numSlicedModes; ++i)
          sc->data[i] = cutensornetSliceInfoPair_t{p.slices[i].first, p.slices[i].second};
      sc->numSlicedModes = (uint32_t)p.slices.size();
      return CUTENSORNET_STATUS_SUCCESS;
    }
    case CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_OVERHEAD: return put(&info->overhead, 8);
    case CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PHASE1_FLOP_COUNT: return put(&info->phase1_flops, 8);
    case CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT:
    case CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_EFFECTIVE_FLOPS_EST: return put(&info->flops, 8);
    case CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_RUNTIME_EST: {
      const double t = 0;  // no timing model: the simulator does not estimate run time
      return put(&t, 8);
    }
    case CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_LARGEST_TENSOR: return put(&info->largest, 8);
    case CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_INTERMEDIATE_MODES: {
      std::vector<int32_t> v;
      for (auto& m : info->inter_modes) v.push_back((int32_t)m.size());
      if (size != v.size() * 4) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
      if (!v.empty()) std::memcpy(buffer, v.data(), size);
      return CUTENSORNET_STATUS_SUCCESS;
    }
    case CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_INTERMEDIATE_MODES: {
      std::vector<int32_t> v;
      for (auto& m : info->inter_modes) v.insert(v.end(), m.begin(), m.end());
      if (size != v.size() * 4) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
      if (!v.empty()) std::memcpy(buffer, v.data(), size);
      return CUTENSORNET_STATUS_SUCCESS;
    }
    default: return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "unknown attribute");
  }
}

}  // namespace

namespace {

Network* info_network(Info* info) {
  return info->net && info->net->magic == kMagicNet ? info->net : nullptr;
}

Status refresh(Info* info, const Path& p, const char* api) {
  Network* n = info_network(info);
  if (!n) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the info's network no longer exists");
  fill_output(*n);
  PathCost pc;
  if (!cost_of(*n, p.steps, extent_map(*n), pc) || (n->in.size() > 1 && p.steps.size() + 1 != n->in.size()))
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the path does not contract the network");
  const auto ext = extent_map(*n);
  for (auto& [m, se] : p.slices)
    if (!ext.count(m) || se <= 0 || se > ext.at(m))
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid sliced mode or extent");
  describe(*n, p, *info);
  return CUTENSORNET_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

cutensornetStatus_t cutensornetContractionOptimizerInfoGetAttribute(
    const cutensornetHandle_t handle, const cutensornetContractionOptimizerInfo_t optimizerInfo,
    cutensornetContractionOptimizerInfoAttributes_t attr, void* buffer, size_t sizeInBytes) {
  const char* api = "cutensornetContractionOptimizerInfoGetAttribute";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Info* info = info_of(optimizerInfo);
  if (!info || !buffer) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  return info_get(info, attr, buffer, sizeInBytes, api);
}

cutensornetStatus_t cutensornetContractionOptimizerInfoSetAttribute(
    const cutensornetHandle_t handle, cutensornetContractionOptimizerInfo_t optimizerInfo,
    cutensornetContractionOptimizerInfoAttributes_t attr, const void* buffer, size_t sizeInBytes) {
  const char* api = "cutensornetContractionOptimizerInfoSetAttribute";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Info* info = info_of(optimizerInfo);
  if (!info || !buffer) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  Path p = info->path;
  switch (attr) {
    case CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH: {
      if (sizeInBytes != sizeof(cutensornetContractionPath_t)) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
      const auto* cp = static_cast<const cutensornetContractionPath_t*>(buffer);
      if (cp->numContractions < 0 || (cp->numContractions > 0 && !cp->data))
        return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid path");
      p.steps.clear();
      for (int32_t i = 0; i < cp->numContractions; ++i) {
        const int32_t f = cp->data[i].first, s2 = cp->data[i].second;
        p.steps.emplace_back(f, s2);
      }
      return refresh(info, p, api);
    }
    case CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_CONFIG: {
      if (sizeInBytes != sizeof(cutensornetSlicingConfig_t)) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
      const auto* sc = static_cast<const cutensornetSlicingConfig_t*>(buffer);
      if (sc->numSlicedModes > 0 && !sc->data) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid slicing");
      p.slices.clear();
      for (uint32_t i = 0; i < sc->numSlicedModes; ++i) p.slices.emplace_back(sc->data[i].slicedMode, sc->data[i].slicedExtent);
      if (!info->has_path) {
        Network* n = info_network(info);
        if (!n) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the info's network no longer exists");
        fill_output(*n);
        p.steps = greedy_path(*n, extent_map(*n));
      }
      return refresh(info, p, api);
    }
    case CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES: {
      if (sizeInBytes != 8) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
      int64_t want;
      std::memcpy(&want, buffer, 8);
      Network* n = info_network(info);
      if (!n || want < 1) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
      fill_output(*n);
      if (!info->has_path) p.steps = greedy_path(*n, extent_map(*n));
      p.slices.clear();
      choose_slicing(*n, p, UINT64_MAX, (int32_t)std::min<int64_t>(want, INT32_MAX));
      return refresh(info, p, api);
    }
    default: return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "unknown or read-only attribute");
  }
}

cutensornetStatus_t cutensornetContractionOptimizerInfoGetPackedSize(
    const cutensornetHandle_t handle, const cutensornetContractionOptimizerInfo_t optimizerInfo, size_t* sizeInBytes) {
  const char* api = "cutensornetContractionOptimizerInfoGetPackedSize";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Info* info = info_of(optimizerInfo);
  if (!info || !sizeInBytes) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  *sizeInBytes = pack(*info).size();
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetContractionOptimizerInfoPackData(
    const cutensornetHandle_t handle, const cutensornetContractionOptimizerInfo_t optimizerInfo, void* buffer,
    size_t sizeInBytes) {
  const char* api = "cutensornetContractionOptimizerInfoPackData";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Info* info = info_of(optimizerInfo);
  if (!info || !buffer) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  const std::vector<uint8_t> b = pack(*info);
  if (sizeInBytes < b.size()) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the buffer is too small");
  std::memcpy(buffer, b.data(), b.size());
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetCreateContractionOptimizerInfoFromPackedData(
    const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t networkDesc, const void* buffer,
    size_t sizeInBytes, cutensornetContractionOptimizerInfo_t* optimizerInfo) {
  const char* api = "cutensornetCreateContractionOptimizerInfoFromPackedData";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  if (!n || !optimizerInfo) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  auto info = std::make_unique<Info>();
  info->net = n;
  Info tmp;
  if (!unpack(buffer, sizeInBytes, tmp) || tmp.num_inputs != (int32_t)n->in.size())
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the buffer is not a packed info for this network");
  if (Status s = refresh(info.get(), tmp.path, api)) return s;
  *optimizerInfo = info.release();
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetUpdateContractionOptimizerInfoFromPackedData(
    const cutensornetHandle_t handle, const void* buffer, size_t sizeInBytes,
    cutensornetContractionOptimizerInfo_t optimizerInfo) {
  const char* api = "cutensornetUpdateContractionOptimizerInfoFromPackedData";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Info* info = info_of(optimizerInfo);
  Info tmp;
  if (!info || !unpack(buffer, sizeInBytes, tmp) || tmp.num_inputs != info->num_inputs)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the buffer is not a packed info for this network");
  return refresh(info, tmp.path, api);
}

// ---- plans ----

cutensornetStatus_t cutensornetCreateContractionPlan(const cutensornetHandle_t handle,
                                                     const cutensornetNetworkDescriptor_t networkDesc,
                                                     const cutensornetContractionOptimizerInfo_t optimizerInfo,
                                                     const cutensornetWorkspaceDescriptor_t workDesc,
                                                     cutensornetContractionPlan_t* plan) {
  const char* api = "cutensornetCreateContractionPlan";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  Info* info = info_of(optimizerInfo);
  if (!n || !info || !plan || !info->has_path) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  Workspace* w = work_of(workDesc);
  if (workDesc && !w) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid workspace descriptor");
  fill_output(*n);
  if (w && w->size[0][0] >= 0 && w->size[0][0] < scratch_need(*n, info->path) &&
      !(w->size[0][0] == 0 && !w->ptr[0][0] && handle_of(handle)->has_mem))
    return fail(CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE, api, "the workspace is smaller than the plan needs");
  auto* p = new Plan;
  p->net = n;
  p->path = info->path;
  *plan = p;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetDestroyContractionPlan(cutensornetContractionPlan_t plan) {
  Plan* p = as<Plan>(plan, kMagicPlan);
  if (!p) return CUTENSORNET_STATUS_INVALID_VALUE;
  p->magic = 0;
  delete p;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetNetworkPrepareContraction(const cutensornetHandle_t handle,
                                                         cutensornetNetworkDescriptor_t networkDesc,
                                                         const cutensornetWorkspaceDescriptor_t workDesc) {
  const char* api = "cutensornetNetworkPrepareContraction";
  Handle* h = handle_of(handle);
  if (!h) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  Workspace* w = work_of(workDesc);
  if (!n || (workDesc && !w)) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  if (!n->has_path) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "optimize the network or set an optimizer info first");
  fill_output(*n);
  const int64_t have = w ? w->size[0][0] : 0;
  if (have >= 0 && have < scratch_need(*n, n->path) && !(!w || (!w->ptr[0][0] && h->has_mem)))
    return fail(CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE, api, "the workspace is smaller than the contraction needs");
  n->prepared = true;
  return CUTENSORNET_STATUS_SUCCESS;
}

// Nothing to tune: one way of computing. The arguments are still checked.
cutensornetStatus_t cutensornetContractionAutotune(const cutensornetHandle_t handle, cutensornetContractionPlan_t plan,
                                                   const void* const rawDataIn[], void* rawDataOut,
                                                   cutensornetWorkspaceDescriptor_t workDesc,
                                                   const cutensornetContractionAutotunePreference_t pref,
                                                   cudaStream_t stream) {
  (void)pref, (void)stream;
  if (!handle_of(handle)) return CUTENSORNET_STATUS_NOT_INITIALIZED;
  if (!as<Plan>(plan, kMagicPlan) || !rawDataIn || !rawDataOut || (workDesc && !work_of(workDesc)))
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, "cutensornetContractionAutotune", "invalid argument");
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetNetworkAutotuneContraction(const cutensornetHandle_t handle,
                                                          cutensornetNetworkDescriptor_t networkDesc,
                                                          const cutensornetWorkspaceDescriptor_t workDesc,
                                                          const cutensornetNetworkAutotunePreference_t pref,
                                                          cudaStream_t stream) {
  (void)pref, (void)stream;
  if (!handle_of(handle)) return CUTENSORNET_STATUS_NOT_INITIALIZED;
  Network* n = net_of(networkDesc);
  if (!n || (workDesc && !work_of(workDesc)))
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, "cutensornetNetworkAutotuneContraction", "invalid argument");
  if (!n->prepared)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, "cutensornetNetworkAutotuneContraction", "prepare the network first");
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetCreateContractionAutotunePreference(const cutensornetHandle_t handle,
                                                                   cutensornetContractionAutotunePreference_t* autotunePreference) {
  if (!handle_of(handle)) return CUTENSORNET_STATUS_NOT_INITIALIZED;
  if (!autotunePreference) return CUTENSORNET_STATUS_INVALID_VALUE;
  *autotunePreference = new TunePref;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetCreateNetworkAutotunePreference(const cutensornetHandle_t handle,
                                                               cutensornetNetworkAutotunePreference_t* autotunePreference) {
  return cutensornetCreateContractionAutotunePreference(handle, autotunePreference);
}

cutensornetStatus_t cutensornetContractionAutotunePreferenceGetAttribute(
    const cutensornetHandle_t handle, const cutensornetContractionAutotunePreference_t autotunePreference,
    cutensornetContractionAutotunePreferenceAttributes_t attr, void* buffer, size_t sizeInBytes) {
  if (!handle_of(handle)) return CUTENSORNET_STATUS_NOT_INITIALIZED;
  TunePref* t = as<TunePref>(autotunePreference, kMagicTune);
  if (!t || !buffer || sizeInBytes != 4 || (attr != 0 && attr != 1)) return CUTENSORNET_STATUS_INVALID_VALUE;
  std::memcpy(buffer, attr == 0 ? &t->max_iterations : &t->intermediate_modes, 4);
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetNetworkAutotunePreferenceGetAttribute(
    const cutensornetHandle_t handle, const cutensornetNetworkAutotunePreference_t autotunePreference,
    cutensornetNetworkAutotunePreferenceAttributes_t attr, void* buffer, size_t sizeInBytes) {
  return cutensornetContractionAutotunePreferenceGetAttribute(
      handle, autotunePreference, (cutensornetContractionAutotunePreferenceAttributes_t)attr, buffer, sizeInBytes);
}

cutensornetStatus_t cutensornetContractionAutotunePreferenceSetAttribute(
    const cutensornetHandle_t handle, cutensornetContractionAutotunePreference_t autotunePreference,
    cutensornetContractionAutotunePreferenceAttributes_t attr, const void* buffer, size_t sizeInBytes) {
  if (!handle_of(handle)) return CUTENSORNET_STATUS_NOT_INITIALIZED;
  TunePref* t = as<TunePref>(autotunePreference, kMagicTune);
  if (!t || !buffer || sizeInBytes != 4 || (attr != 0 && attr != 1)) return CUTENSORNET_STATUS_INVALID_VALUE;
  int32_t v;
  std::memcpy(&v, buffer, 4);
  if (attr == 0 && v < 1) return CUTENSORNET_STATUS_INVALID_VALUE;
  if (attr == 1 && (v < 0 || v > 2)) return CUTENSORNET_STATUS_INVALID_VALUE;
  (attr == 0 ? t->max_iterations : t->intermediate_modes) = v;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetNetworkAutotunePreferenceSetAttribute(
    const cutensornetHandle_t handle, cutensornetNetworkAutotunePreference_t autotunePreference,
    cutensornetNetworkAutotunePreferenceAttributes_t attr, const void* buf, size_t sizeInBytes) {
  return cutensornetContractionAutotunePreferenceSetAttribute(
      handle, autotunePreference, (cutensornetContractionAutotunePreferenceAttributes_t)attr, buf, sizeInBytes);
}

cutensornetStatus_t cutensornetDestroyContractionAutotunePreference(cutensornetContractionAutotunePreference_t autotunePreference) {
  TunePref* t = as<TunePref>(autotunePreference, kMagicTune);
  if (!t) return CUTENSORNET_STATUS_INVALID_VALUE;
  t->magic = 0;
  delete t;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetDestroyNetworkAutotunePreference(cutensornetNetworkAutotunePreference_t autotunePreference) {
  return cutensornetDestroyContractionAutotunePreference(autotunePreference);
}

// ---- slices and contraction ----

cutensornetStatus_t cutensornetCreateSliceGroupFromIDRange(const cutensornetHandle_t handle, int64_t sliceIdStart,
                                                           int64_t sliceIdStop, int64_t sliceIdStep,
                                                           cutensornetSliceGroup_t* sliceGroup) {
  const char* api = "cutensornetCreateSliceGroupFromIDRange";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  if (!sliceGroup || sliceIdStep == 0 || sliceIdStart < 0 || sliceIdStop < -1)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid range");
  auto* g = new SliceGroup;
  g->range = true;
  g->start = sliceIdStart;
  g->stop = sliceIdStop;
  g->step = sliceIdStep;
  *sliceGroup = g;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetCreateSliceGroupFromIDs(const cutensornetHandle_t handle,
                                                       const int64_t* beginIDSequence, const int64_t* endIDSequence,
                                                       cutensornetSliceGroup_t* sliceGroup) {
  const char* api = "cutensornetCreateSliceGroupFromIDs";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  if (!sliceGroup || !beginIDSequence || !endIDSequence || endIDSequence < beginIDSequence)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid sequence");
  auto* g = new SliceGroup;
  std::set<int64_t> seen;
  for (const int64_t* p = beginIDSequence; p != endIDSequence; ++p) {
    if (*p < 0) {
      delete g;
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "slice ids must not be negative");
    }
    if (seen.insert(*p).second) g->ids.push_back(*p);
  }
  *sliceGroup = g;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetDestroySliceGroup(cutensornetSliceGroup_t sliceGroup) {
  SliceGroup* g = as<SliceGroup>(sliceGroup, kMagicSlices);
  if (!g) return CUTENSORNET_STATUS_INVALID_VALUE;
  g->magic = 0;
  delete g;
  return CUTENSORNET_STATUS_SUCCESS;
}

}  // extern "C"

namespace {

Status slice_ids(const SliceGroup* g, int64_t total, std::vector<int64_t>& ids, const char* api) {
  ids.clear();
  if (!g) {
    for (int64_t i = 0; i < total; ++i) ids.push_back(i);
    return CUTENSORNET_STATUS_SUCCESS;
  }
  if (g->range) {
    for (int64_t i = g->start; g->step > 0 ? i < g->stop : i > g->stop; i += g->step) ids.push_back(i);
  } else {
    ids = g->ids;
  }
  for (int64_t i : ids)
    if (i < 0 || i >= total) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "a slice id is out of range");
  return CUTENSORNET_STATUS_SUCCESS;
}

Status run_network(Handle* h, Network& n, const Path& p, const std::vector<const void*>& in, void* out,
                   bool accumulate, cutensornetWorkspaceDescriptor_t workDesc, const SliceGroup* group,
                   cudaStream_t stream, const char* api) {
  Workspace* w = work_of(workDesc);
  if (workDesc && !w) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid workspace descriptor");
  if (!out) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the output has no memory");
  for (const void* x : in)
    if (!x) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "an input tensor has no memory");
  fill_output(n);
  std::vector<int64_t> ids;
  if (Status s = slice_ids(group, slices_of(p, extent_map(n)), ids, api)) return s;
  return contract_slices(h, n, p, in, out, accumulate, w, ids, stream, api);
}

}  // namespace

extern "C" {

cutensornetStatus_t cutensornetContraction(const cutensornetHandle_t handle, cutensornetContractionPlan_t plan,
                                           const void* const rawDataIn[], void* rawDataOut,
                                           cutensornetWorkspaceDescriptor_t workDesc, int64_t sliceId,
                                           cudaStream_t stream) {
  const char* api = "cutensornetContraction";
  Handle* h = handle_of(handle);
  if (!h) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Plan* p = as<Plan>(plan, kMagicPlan);
  if (!p || !rawDataIn || !net_of(p->net)) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  std::vector<const void*> in(rawDataIn, rawDataIn + p->net->in.size());
  SliceGroup one;
  one.ids = {sliceId};
  // Slices run in ascending order from 0: the first writes the output, the
  // rest add to it (the documented contract of this deprecated call).
  return run_network(h, *p->net, p->path, in, rawDataOut, sliceId > 0, workDesc, &one, stream, api);
}

cutensornetStatus_t cutensornetContractSlices(const cutensornetHandle_t handle, cutensornetContractionPlan_t plan,
                                              const void* const rawDataIn[], void* rawDataOut,
                                              int32_t accumulateOutput, cutensornetWorkspaceDescriptor_t workDesc,
                                              const cutensornetSliceGroup_t sliceGroup, cudaStream_t stream) {
  const char* api = "cutensornetContractSlices";
  Handle* h = handle_of(handle);
  if (!h) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Plan* p = as<Plan>(plan, kMagicPlan);
  if (!p || !rawDataIn || !net_of(p->net)) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  const SliceGroup* g = as<SliceGroup>(sliceGroup, kMagicSlices);
  if (sliceGroup && !g) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid slice group");
  std::vector<const void*> in(rawDataIn, rawDataIn + p->net->in.size());
  return run_network(h, *p->net, p->path, in, rawDataOut, accumulateOutput != 0, workDesc, g, stream, api);
}

cutensornetStatus_t cutensornetNetworkContract(const cutensornetHandle_t handle,
                                               cutensornetNetworkDescriptor_t networkDesc, int32_t accumulateOutput,
                                               const cutensornetWorkspaceDescriptor_t workDesc,
                                               const cutensornetSliceGroup_t sliceGroup, cudaStream_t stream) {
  const char* api = "cutensornetNetworkContract";
  Handle* h = handle_of(handle);
  if (!h) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  Network* n = net_of(networkDesc);
  if (!n) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid network");
  if (!n->prepared) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "prepare the network first");
  const SliceGroup* g = as<SliceGroup>(sliceGroup, kMagicSlices);
  if (sliceGroup && !g) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid slice group");
  std::vector<const void*> in;
  for (const NetTensor& t : n->in) in.push_back(t.data);
  return run_network(h, *n, n->path, in, n->out_data, accumulateOutput != 0, workDesc, g, stream, api);
}

}  // extern "C"

namespace {

// ---- decompositions: cuTENSOR lays the tensor out as a matrix, cuSOLVER
//      factors it ----

cusolverDnHandle_t solver_of(Handle* h) {
  if (!h->solver && cusolverDnCreate(&h->solver) != CUSOLVER_STATUS_SUCCESS) h->solver = nullptr;
  return h->solver;
}

// How a tensor splits into the rows (modes of X but the shared one) and the
// columns (modes of Y but the shared one) of a matrix.
struct Split {
  std::vector<int32_t> rows, cols;
  std::vector<int64_t> row_ext, col_ext;
  int32_t shared = 0;
  int64_t m = 1, n = 1, k = 0;
};

Status split_of(const TensorDesc& in, const TensorDesc& x, const TensorDesc& y, Split& s, const char* api) {
  std::vector<int32_t> common;
  for (int32_t m : x.modes)
    if (contains(y.modes, m)) common.push_back(m);
  if (common.size() != 1) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the two outputs must share exactly one mode");
  s.shared = common[0];
  std::map<int32_t, int64_t> ext;
  for (size_t i = 0; i < in.modes.size(); ++i) ext[in.modes[i]] = in.extents[i];
  auto take = [&](const TensorDesc& t, std::vector<int32_t>& modes, std::vector<int64_t>& e, int64_t& vol) -> bool {
    for (size_t i = 0; i < t.modes.size(); ++i) {
      if (t.modes[i] == s.shared) {
        s.k = t.extents[i];
        continue;
      }
      auto it = ext.find(t.modes[i]);
      if (it == ext.end() || it->second != t.extents[i]) return false;
      modes.push_back(t.modes[i]);
      e.push_back(t.extents[i]);
      vol *= t.extents[i];
    }
    return true;
  };
  if (!take(x, s.rows, s.row_ext, s.m) || !take(y, s.cols, s.col_ext, s.n))
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "an output mode is not an input mode with the same extent");
  if (s.rows.size() + s.cols.size() != in.modes.size() || ext.count(s.shared))
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the outputs must partition the input's modes");
  for (int32_t m : in.modes)
    if (!contains(s.rows, m) && !contains(s.cols, m))
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the outputs must partition the input's modes");
  int64_t kx = 0, ky = 0;
  for (size_t i = 0; i < x.modes.size(); ++i)
    if (x.modes[i] == s.shared) kx = x.extents[i];
  for (size_t i = 0; i < y.modes.size(); ++i)
    if (y.modes[i] == s.shared) ky = y.extents[i];
  if (kx != ky) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the shared mode has two extents");
  return CUTENSORNET_STATUS_SUCCESS;
}

size_t real_bytes(cudaDataType_t t) { return (t == CUDA_R_64F || t == CUDA_C_64F) ? 8 : 4; }

// cuSOLVER's workspace for the type, in elements.
int solver_lwork(cusolverDnHandle_t sh, cudaDataType_t t, int m, int n, bool qr) {
  int a = 0, b = 0;
  switch (t) {
    case CUDA_R_32F:
      if (qr) cusolverDnSgeqrf_bufferSize(sh, m, n, nullptr, m, &a), cusolverDnSorgqr_bufferSize(sh, m, std::min(m, n), std::min(m, n), nullptr, m, nullptr, &b);
      else cusolverDnSgesvd_bufferSize(sh, std::max(m, n), std::min(m, n), &a);
      break;
    case CUDA_R_64F:
      if (qr) cusolverDnDgeqrf_bufferSize(sh, m, n, nullptr, m, &a), cusolverDnDorgqr_bufferSize(sh, m, std::min(m, n), std::min(m, n), nullptr, m, nullptr, &b);
      else cusolverDnDgesvd_bufferSize(sh, std::max(m, n), std::min(m, n), &a);
      break;
    case CUDA_C_32F:
      if (qr) cusolverDnCgeqrf_bufferSize(sh, m, n, nullptr, m, &a), cusolverDnCungqr_bufferSize(sh, m, std::min(m, n), std::min(m, n), nullptr, m, nullptr, &b);
      else cusolverDnCgesvd_bufferSize(sh, std::max(m, n), std::min(m, n), &a);
      break;
    case CUDA_C_64F:
      if (qr) cusolverDnZgeqrf_bufferSize(sh, m, n, nullptr, m, &a), cusolverDnZungqr_bufferSize(sh, m, std::min(m, n), std::min(m, n), nullptr, m, nullptr, &b);
      else cusolverDnZgesvd_bufferSize(sh, std::max(m, n), std::min(m, n), &a);
      break;
    default: break;
  }
  return std::max({a, b, 1});
}

// A carve-up of the scratch, in order, each piece 256-byte aligned.
struct Carve {
  char* base = nullptr;
  int64_t used = 0;
  void* take(int64_t bytes) {
    void* p = base ? base + used : nullptr;
    used += align_up(std::max<int64_t>(bytes, 1));
    return p;
  }
};

int64_t qr_need(cusolverDnHandle_t sh, cudaDataType_t t, const Split& s) {
  const int64_t eb = (int64_t)elem_bytes(t), k = std::min(s.m, s.n);
  Carve c;
  c.take(s.m * s.n * eb);                                    // the matrix, then Q
  c.take(k * eb);                                            // tau
  c.take((int64_t)solver_lwork(sh, t, (int)s.m, (int)s.n, true) * eb);
  c.take(4);                                                 // devInfo
  c.take(k * s.n * eb);                                      // R
  return c.used;
}

int64_t svd_need(cusolverDnHandle_t sh, cudaDataType_t t, const Split& s) {
  const int64_t eb = (int64_t)elem_bytes(t), rb = (int64_t)real_bytes(t);
  const int64_t mm = std::max(s.m, s.n), nn = std::min(s.m, s.n);
  Carve c;
  c.take(mm * nn * eb);                                      // the matrix (or its adjoint)
  c.take(nn * rb);                                           // S
  c.take(mm * nn * eb);                                      // U
  c.take(nn * nn * eb);                                      // VT
  c.take((int64_t)solver_lwork(sh, t, (int)s.m, (int)s.n, false) * eb);
  c.take(nn * rb);                                           // rwork
  c.take(4);                                                 // devInfo
  c.take(s.m * nn * eb);                                     // U, laid out for the output
  c.take(nn * s.n * eb);                                     // V
  return c.used;
}

// Copies tensor `src` (modes/ext/str) into `dst` (its own modes/ext/str),
// conjugating if asked, on the stream, with cuTENSOR.
Status permute(Handle* h, cudaDataType_t t, const View& src, const View& dst, bool conj, cudaStream_t stream,
               const char* api) {
  TensorGuard a, b;
  if (make_desc(h->ct, src, t, a) || make_desc(h->ct, dst, t, b))
    return fail(CUTENSORNET_STATUS_CUTENSOR_ERROR, api, "cuTENSOR refused a tensor descriptor");
  OpGuard op;
  const cutensorComputeDescriptor_t c =
      (t == CUDA_R_64F || t == CUDA_C_64F) ? CUTENSOR_COMPUTE_DESC_64F : CUTENSOR_COMPUTE_DESC_32F;
  if (cutensorStatus_t s = cutensorCreatePermutation(h->ct, &op.d, a.d, src.modes.data(),
                                                    conj ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY, b.d,
                                                    dst.modes.data(), c))
    return fail(from_cutensor(s), api, "cuTENSOR refused a permutation");
  PrefGuard pref;
  PlanGuard plan;
  if (cutensorCreatePlanPreference(h->ct, &pref.p, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE) ||
      cutensorCreatePlan(h->ct, &plan.p, op.d, pref.p, 0))
    return fail(CUTENSORNET_STATUS_CUTENSOR_ERROR, api, "cuTENSOR could not plan a permutation");
  cudaDataType_t st = CUDA_R_32F;
  cutensorOperationDescriptorGetAttribute(h->ct, op.d, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, &st, sizeof st);
  Scalar one(st, 1.0);
  if (cutensorStatus_t s = cutensorPermute(h->ct, plan.p, one.b, src.ptr, const_cast<void*>(dst.ptr), stream))
    return fail(from_cutensor(s), api, "a cuTENSOR permutation failed");
  return CUTENSORNET_STATUS_SUCCESS;
}

View tensor_view(const TensorDesc& t, const void* p) {
  View v;
  v.modes = t.modes;
  v.ext = t.extents;
  v.str = t.strides;
  v.ptr = p;
  return v;
}

View packed_view(std::vector<int32_t> modes, std::vector<int64_t> ext, const void* p) {
  View v;
  v.modes = std::move(modes);
  v.ext = std::move(ext);
  v.str = packed_strides(v.ext);
  v.ptr = p;
  return v;
}

template <class A, class B>
std::vector<A> cat(std::vector<A> a, const B& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

// Host copies of device matrices, widened to complex double.
bool download(const void* p, cudaDataType_t t, size_t count, std::vector<cd>& out) {
  out.assign(count, cd{});
  const size_t eb = elem_bytes(t);
  std::vector<unsigned char> raw(count * eb);
  if (count && cudaMemcpy(raw.data(), p, raw.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    const unsigned char* e = raw.data() + i * eb;
    switch (t) {
      case CUDA_R_32F: { float v; std::memcpy(&v, e, 4); out[i] = v; break; }
      case CUDA_R_64F: { double v; std::memcpy(&v, e, 8); out[i] = v; break; }
      case CUDA_C_32F: { float v[2]; std::memcpy(v, e, 8); out[i] = cd(v[0], v[1]); break; }
      case CUDA_C_64F: { double v[2]; std::memcpy(v, e, 16); out[i] = cd(v[0], v[1]); break; }
      default: break;
    }
  }
  return true;
}

bool upload(void* p, cudaDataType_t t, const std::vector<cd>& v) {
  const size_t eb = elem_bytes(t);
  std::vector<unsigned char> raw(v.size() * eb);
  for (size_t i = 0; i < v.size(); ++i) {
    unsigned char* e = raw.data() + i * eb;
    switch (t) {
      case CUDA_R_32F: { float x = (float)v[i].real(); std::memcpy(e, &x, 4); break; }
      case CUDA_R_64F: { double x = v[i].real(); std::memcpy(e, &x, 8); break; }
      case CUDA_C_32F: { float x[2] = {(float)v[i].real(), (float)v[i].imag()}; std::memcpy(e, x, 8); break; }
      case CUDA_C_64F: { double x[2] = {v[i].real(), v[i].imag()}; std::memcpy(e, x, 16); break; }
      default: break;
    }
  }
  const bool ok = raw.empty() || cudaMemcpy(p, raw.data(), raw.size(), cudaMemcpyHostToDevice) == cudaSuccess;
  cudaGetLastError();
  return ok;
}

bool upload_real(void* p, cudaDataType_t t, const std::vector<double>& v) {
  if (real_bytes(t) == 8) return v.empty() || cudaMemcpy(p, v.data(), v.size() * 8, cudaMemcpyHostToDevice) == cudaSuccess;
  std::vector<float> f(v.begin(), v.end());
  return f.empty() || cudaMemcpy(p, f.data(), f.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess;
}

bool capturing(cudaStream_t s) {
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  const bool on = cudaStreamIsCapturing(s, &st) == cudaSuccess && st == cudaStreamCaptureStatusActive;
  cudaGetLastError();
  return on;
}

Status settle(cudaStream_t stream, const char* api) {
  if (capturing(stream))
    return refuse(api, "decompositions read their results back to the host and cannot be captured into a CUDA "
                       "graph; call it outside capture");
  cudaGetLastError();
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    cudaGetLastError();
    return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "the stream failed");
  }
  return CUTENSORNET_STATUS_SUCCESS;
}

}  // namespace

namespace {

bool solver_type(cudaDataType_t t) {
  return t == CUDA_R_32F || t == CUDA_R_64F || t == CUDA_C_32F || t == CUDA_C_64F;
}

#define VGPU_SOLVE(call)                                                         \
  do {                                                                           \
    if ((call) != CUSOLVER_STATUS_SUCCESS)                                       \
      return fail(CUTENSORNET_STATUS_CUSOLVER_ERROR, api, "cuSOLVER failed");    \
  } while (0)

Status do_qr(Handle* h, const TensorDesc& in, const void* x, const TensorDesc& qd, void* q, const TensorDesc& rd,
             void* r, const Workspace* w, cudaStream_t stream, const char* api) {
  if (!x || !q || !r) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "a data pointer is null");
  const cudaDataType_t t = in.type;
  if (qd.type != t || rd.type != t) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the tensors' types differ");
  if (!solver_type(t)) return refuse(api, "QR of half-precision tensors is not implemented");
  Split s;
  if (Status st = split_of(in, qd, rd, s, api)) return st;
  const int64_t k = std::min(s.m, s.n);
  if (s.k != k) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the shared mode's extent must be min(m, n)");
  if (Status st = settle(stream, api)) return st;
  cusolverDnHandle_t sh = solver_of(h);
  if (!sh) return fail(CUTENSORNET_STATUS_CUSOLVER_ERROR, api, "cusolverDnCreate failed");
  Scratch scratch;
  if (Status st = get_scratch(h, w, qr_need(sh, t, s), stream, scratch, api)) return st;
  const int64_t eb = (int64_t)elem_bytes(t);
  const int lwork = solver_lwork(sh, t, (int)s.m, (int)s.n, true);
  Carve c{static_cast<char*>(scratch.ptr)};
  void* M = c.take(s.m * s.n * eb);
  void* tau = c.take(k * eb);
  void* work = c.take((int64_t)lwork * eb);
  int* info = static_cast<int*>(c.take(4));
  void* R = c.take(k * s.n * eb);
  if (Status st = permute(h, t, tensor_view(in, x), packed_view(cat(s.rows, s.cols), cat(s.row_ext, s.col_ext), M),
                          false, stream, api))
    return st;
  cudaStreamSynchronize(stream);
  cusolverDnSetStream(sh, stream);
  const int m = (int)s.m, n = (int)s.n, kk = (int)k;
  switch (t) {
    case CUDA_R_32F: VGPU_SOLVE(cusolverDnSgeqrf(sh, m, n, (float*)M, m, (float*)tau, (float*)work, lwork, info)); break;
    case CUDA_R_64F: VGPU_SOLVE(cusolverDnDgeqrf(sh, m, n, (double*)M, m, (double*)tau, (double*)work, lwork, info)); break;
    case CUDA_C_32F: VGPU_SOLVE(cusolverDnCgeqrf(sh, m, n, (cuComplex*)M, m, (cuComplex*)tau, (cuComplex*)work, lwork, info)); break;
    default: VGPU_SOLVE(cusolverDnZgeqrf(sh, m, n, (cuDoubleComplex*)M, m, (cuDoubleComplex*)tau, (cuDoubleComplex*)work, lwork, info)); break;
  }
  cudaStreamSynchronize(stream);
  // R: the upper triangle (trapezoid) of the factored matrix's first k rows.
  std::vector<cd> fm, rv((size_t)(k * s.n));
  if (!download(M, t, (size_t)(s.m * s.n), fm)) return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "a copy failed");
  for (int64_t j = 0; j < s.n; ++j)
    for (int64_t i = 0; i < k; ++i) rv[(size_t)(i + j * k)] = j >= i ? fm[(size_t)(i + j * s.m)] : cd{};
  if (!upload(R, t, rv)) return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "a copy failed");
  switch (t) {
    case CUDA_R_32F: VGPU_SOLVE(cusolverDnSorgqr(sh, m, kk, kk, (float*)M, m, (float*)tau, (float*)work, lwork, info)); break;
    case CUDA_R_64F: VGPU_SOLVE(cusolverDnDorgqr(sh, m, kk, kk, (double*)M, m, (double*)tau, (double*)work, lwork, info)); break;
    case CUDA_C_32F: VGPU_SOLVE(cusolverDnCungqr(sh, m, kk, kk, (cuComplex*)M, m, (cuComplex*)tau, (cuComplex*)work, lwork, info)); break;
    default: VGPU_SOLVE(cusolverDnZungqr(sh, m, kk, kk, (cuDoubleComplex*)M, m, (cuDoubleComplex*)tau, (cuDoubleComplex*)work, lwork, info)); break;
  }
  cudaStreamSynchronize(stream);
  if (Status st = permute(h, t, packed_view(cat(s.rows, std::vector<int32_t>{s.shared}), cat(s.row_ext, std::vector<int64_t>{k}), M),
                          tensor_view(qd, q), false, stream, api))
    return st;
  return permute(h, t, packed_view(cat(std::vector<int32_t>{s.shared}, s.cols), cat(std::vector<int64_t>{k}, s.col_ext), R),
                 tensor_view(rd, r), false, stream, api);
}

const SvdConfig kDefaultSvd{};

// U S V^H of the tensor split as rows (U's modes) by columns (V's modes),
// truncated, normalized and partitioned as the configuration asks.
Status do_svd(Handle* h, const TensorDesc& in, const void* x, TensorDesc& ud, void* u, void* sv, TensorDesc& vd, void* v,
              const SvdConfig* cfg, SvdInfo* out_info, const Workspace* w, cudaStream_t stream, const char* api) {
  if (!x || !u || !v) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "a data pointer is null");
  if (!cfg) cfg = &kDefaultSvd;
  const cudaDataType_t t = in.type;
  if (ud.type != t || vd.type != t) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the tensors' types differ");
  if (!solver_type(t)) return refuse(api, "SVD of half-precision tensors is not implemented");
  if (cfg->algo != CUTENSORNET_TENSOR_SVD_ALGO_GESVD)
    return refuse(api, "only the gesvd algorithm is implemented (gesvdj, gesvdp and gesvdr are not)");
  if (cfg->partition == CUTENSORNET_TENSOR_SVD_PARTITION_NONE && !sv)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "S is needed when the singular values are not partitioned");
  Split s;
  if (Status st = split_of(in, ud, vd, s, api)) return st;
  const int64_t full = std::min(s.m, s.n);
  if (s.k < 1 || s.k > full) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the shared extent exceeds min(m, n)");
  if (Status st = settle(stream, api)) return st;
  cusolverDnHandle_t sh = solver_of(h);
  if (!sh) return fail(CUTENSORNET_STATUS_CUSOLVER_ERROR, api, "cusolverDnCreate failed");
  Scratch scratch;
  if (Status st = get_scratch(h, w, svd_need(sh, t, s), stream, scratch, api)) return st;
  const int64_t eb = (int64_t)elem_bytes(t), rb = (int64_t)real_bytes(t);
  const bool tr = s.m < s.n;  // factor the adjoint: gesvd needs rows >= columns
  const int64_t mm = std::max(s.m, s.n), nn = full;
  const int lwork = solver_lwork(sh, t, (int)s.m, (int)s.n, false);
  Carve c{static_cast<char*>(scratch.ptr)};
  void* A = c.take(mm * nn * eb);
  void* S = c.take(nn * rb);
  void* U = c.take(mm * nn * eb);
  void* VT = c.take(nn * nn * eb);
  void* work = c.take((int64_t)lwork * eb);
  void* rwork = c.take(nn * rb);
  int* info = static_cast<int*>(c.take(4));
  void* Ubuf = c.take(s.m * nn * eb);
  void* Vbuf = c.take(nn * s.n * eb);
  const View mat = tr ? packed_view(cat(s.cols, s.rows), cat(s.col_ext, s.row_ext), A)
                      : packed_view(cat(s.rows, s.cols), cat(s.row_ext, s.col_ext), A);
  if (Status st = permute(h, t, tensor_view(in, x), mat, tr && is_complex(t), stream, api)) return st;
  cudaStreamSynchronize(stream);
  cusolverDnSetStream(sh, stream);
  const int M = (int)mm, N = (int)nn;
  switch (t) {
    case CUDA_R_32F: VGPU_SOLVE(cusolverDnSgesvd(sh, 'S', 'S', M, N, (float*)A, M, (float*)S, (float*)U, M, (float*)VT, N, (float*)work, lwork, (float*)rwork, info)); break;
    case CUDA_R_64F: VGPU_SOLVE(cusolverDnDgesvd(sh, 'S', 'S', M, N, (double*)A, M, (double*)S, (double*)U, M, (double*)VT, N, (double*)work, lwork, (double*)rwork, info)); break;
    case CUDA_C_32F: VGPU_SOLVE(cusolverDnCgesvd(sh, 'S', 'S', M, N, (cuComplex*)A, M, (float*)S, (cuComplex*)U, M, (cuComplex*)VT, N, (cuComplex*)work, lwork, (float*)rwork, info)); break;
    default: VGPU_SOLVE(cusolverDnZgesvd(sh, 'S', 'S', M, N, (cuDoubleComplex*)A, M, (double*)S, (cuDoubleComplex*)U, M, (cuDoubleComplex*)VT, N, (cuDoubleComplex*)work, lwork, (double*)rwork, info)); break;
  }
  cudaStreamSynchronize(stream);
  std::vector<cd> hu, hvt, hs;
  if (!download(U, t, (size_t)(mm * nn), hu) || !download(VT, t, (size_t)(nn * nn), hvt) ||
      !download(S, real_bytes(t) == 8 ? CUDA_R_64F : CUDA_R_32F, (size_t)nn, hs))
    return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "a copy failed");
  // U_A (m x full) and V_A^H (full x n), column-major.
  std::vector<cd> ua((size_t)(s.m * full)), vha((size_t)(full * s.n));
  for (int64_t j = 0; j < full; ++j)
    for (int64_t i = 0; i < s.m; ++i)
      ua[(size_t)(i + j * s.m)] = tr ? std::conj(hvt[(size_t)(j + i * nn)]) : hu[(size_t)(i + j * s.m)];
  for (int64_t c2 = 0; c2 < s.n; ++c2)
    for (int64_t j = 0; j < full; ++j)
      vha[(size_t)(j + c2 * full)] = tr ? std::conj(hu[(size_t)(c2 + j * mm)]) : hvt[(size_t)(j + c2 * nn)];
  std::vector<double> sing(full);
  for (int64_t j = 0; j < full; ++j) sing[(size_t)j] = hs[(size_t)j].real();
  // Truncation: the lowest of the fixed extent and every cutoff, at least one.
  double total = 0;
  for (double x2 : sing) total += x2 * x2;
  int64_t keep = s.k;
  if (cfg->abs_cutoff > 0 || cfg->rel_cutoff > 0) {
    int64_t kk = 0;
    const double thr = std::max(cfg->abs_cutoff, cfg->rel_cutoff * (sing.empty() ? 0 : sing[0]));
    while (kk < full && sing[(size_t)kk] > thr) ++kk;
    keep = std::min(keep, kk);
  }
  if (cfg->discarded_cutoff > 0 && total > 0) {
    int64_t kk = full;
    double tail = 0;
    while (kk > 1 && (tail + sing[(size_t)(kk - 1)] * sing[(size_t)(kk - 1)]) / total <= cfg->discarded_cutoff) {
      tail += sing[(size_t)(kk - 1)] * sing[(size_t)(kk - 1)];
      --kk;
    }
    keep = std::min(keep, kk);
  }
  keep = std::max<int64_t>(keep, 1);
  double discarded = 0;
  for (int64_t j = keep; j < full; ++j) discarded += sing[(size_t)j] * sing[(size_t)j];
  discarded = total > 0 ? discarded / total : 0;
  std::vector<double> kept(sing.begin(), sing.begin() + keep);
  double norm = 0;
  switch (cfg->normalization) {
    case CUTENSORNET_TENSOR_SVD_NORMALIZATION_L1: for (double y : kept) norm += std::fabs(y); break;
    case CUTENSORNET_TENSOR_SVD_NORMALIZATION_L2: for (double y : kept) norm += y * y; norm = std::sqrt(norm); break;
    case CUTENSORNET_TENSOR_SVD_NORMALIZATION_LINF: for (double y : kept) norm = std::max(norm, std::fabs(y)); break;
    default: break;
  }
  if (norm > 0)
    for (double& y : kept) y /= norm;
  std::vector<cd> uo((size_t)(s.m * keep)), vo((size_t)(keep * s.n));
  for (int64_t j = 0; j < keep; ++j) {
    const double sj = kept[(size_t)j];
    const double fu = cfg->partition == CUTENSORNET_TENSOR_SVD_PARTITION_US ? sj
                      : cfg->partition == CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL ? std::sqrt(sj) : 1.0;
    const double fv = cfg->partition == CUTENSORNET_TENSOR_SVD_PARTITION_SV ? sj
                      : cfg->partition == CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL ? std::sqrt(sj) : 1.0;
    for (int64_t i = 0; i < s.m; ++i) uo[(size_t)(i + j * s.m)] = ua[(size_t)(i + j * s.m)] * fu;
    for (int64_t c2 = 0; c2 < s.n; ++c2) vo[(size_t)(j + c2 * keep)] = vha[(size_t)(j + c2 * full)] * fv;
  }
  // A truncated result adopts the packed (Fortran) layout of the reduced
  // extent, and the descriptors say so.
  if (keep < s.k) {
    for (TensorDesc* d : {&ud, &vd}) {
      for (size_t i = 0; i < d->modes.size(); ++i)
        if (d->modes[i] == s.shared) d->extents[i] = keep;
      d->strides = packed_strides(d->extents);
    }
  }
  if (!upload(Ubuf, t, uo) || !upload(Vbuf, t, vo)) return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "a copy failed");
  if (Status st = permute(h, t, packed_view(cat(s.rows, std::vector<int32_t>{s.shared}), cat(s.row_ext, std::vector<int64_t>{keep}), Ubuf),
                          tensor_view(ud, u), false, stream, api))
    return st;
  if (Status st = permute(h, t, packed_view(cat(std::vector<int32_t>{s.shared}, s.cols), cat(std::vector<int64_t>{keep}, s.col_ext), Vbuf),
                          tensor_view(vd, v), false, stream, api))
    return st;
  cudaStreamSynchronize(stream);
  if (sv && !upload_real(sv, t, kept)) return fail(CUTENSORNET_STATUS_CUDA_ERROR, api, "a copy failed");
  if (out_info) {
    out_info->full = full;
    out_info->reduced = keep;
    out_info->discarded = discarded;
    out_info->algo = cfg->algo;
  }
  return CUTENSORNET_STATUS_SUCCESS;
}

#undef VGPU_SOLVE

}  // namespace

extern "C" {

cutensornetStatus_t cutensornetCreateTensorSVDConfig(const cutensornetHandle_t handle,
                                                     cutensornetTensorSVDConfig_t* svdConfig) {
  if (!handle_of(handle)) return CUTENSORNET_STATUS_NOT_INITIALIZED;
  if (!svdConfig) return CUTENSORNET_STATUS_INVALID_VALUE;
  *svdConfig = new SvdConfig;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetDestroyTensorSVDConfig(cutensornetTensorSVDConfig_t svdConfig) {
  SvdConfig* c = as<SvdConfig>(svdConfig, kMagicSvdConfig);
  if (!c) return CUTENSORNET_STATUS_INVALID_VALUE;
  c->magic = 0;
  delete c;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetTensorSVDConfigGetAttribute(const cutensornetHandle_t handle,
                                                           const cutensornetTensorSVDConfig_t svdConfig,
                                                           cutensornetTensorSVDConfigAttributes_t attr, void* buffer,
                                                           size_t sizeInBytes) {
  const char* api = "cutensornetTensorSVDConfigGetAttribute";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  SvdConfig* c = as<SvdConfig>(svdConfig, kMagicSvdConfig);
  if (!c || !buffer) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  auto put = [&](const void* v, size_t n) -> Status {
    if (sizeInBytes != n) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
    std::memcpy(buffer, v, n);
    return CUTENSORNET_STATUS_SUCCESS;
  };
  switch (attr) {
    case CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF: return put(&c->abs_cutoff, 8);
    case CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF: return put(&c->rel_cutoff, 8);
    case CUTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION: return put(&c->normalization, 4);
    case CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION: return put(&c->partition, 4);
    case CUTENSORNET_TENSOR_SVD_CONFIG_ALGO: return put(&c->algo, 4);
    case CUTENSORNET_TENSOR_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF: return put(&c->discarded_cutoff, 8);
    case CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS:
      if (c->algo == CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ) return put(&c->gesvdj, sizeof c->gesvdj);
      if (c->algo == CUTENSORNET_TENSOR_SVD_ALGO_GESVDR) return put(&c->gesvdr, sizeof c->gesvdr);
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the algorithm takes no parameters");
    default: return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "unknown attribute");
  }
}

cutensornetStatus_t cutensornetTensorSVDConfigSetAttribute(const cutensornetHandle_t handle,
                                                           cutensornetTensorSVDConfig_t svdConfig,
                                                           cutensornetTensorSVDConfigAttributes_t attr,
                                                           const void* buffer, size_t sizeInBytes) {
  const char* api = "cutensornetTensorSVDConfigSetAttribute";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  SvdConfig* c = as<SvdConfig>(svdConfig, kMagicSvdConfig);
  if (!c || !buffer) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  // Buffers larger than the value are taken (measured: a 4-byte enum set
  // from 8 bytes).
  auto get_d = [&](double& d) -> Status {
    if (sizeInBytes < 8) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
    double v;
    std::memcpy(&v, buffer, 8);
    if (v < 0) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "a cutoff must not be negative");
    d = v;
    return CUTENSORNET_STATUS_SUCCESS;
  };
  auto get_i = [&](int32_t& d, int32_t lo, int32_t hi) -> Status {
    if (sizeInBytes < 4) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
    int32_t v;
    std::memcpy(&v, buffer, 4);
    if (v < lo || v > hi) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "value out of range");
    d = v;
    return CUTENSORNET_STATUS_SUCCESS;
  };
  switch (attr) {
    case CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF: return get_d(c->abs_cutoff);
    case CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF: return get_d(c->rel_cutoff);
    case CUTENSORNET_TENSOR_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF: return get_d(c->discarded_cutoff);
    case CUTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION: return get_i(c->normalization, 0, 3);
    case CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION: return get_i(c->partition, 0, 3);
    case CUTENSORNET_TENSOR_SVD_CONFIG_ALGO: return get_i(c->algo, 0, 3);
    case CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS:
      if (c->algo == CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ && sizeInBytes == sizeof c->gesvdj) {
        std::memcpy(&c->gesvdj, buffer, sizeInBytes);
        return CUTENSORNET_STATUS_SUCCESS;
      }
      if (c->algo == CUTENSORNET_TENSOR_SVD_ALGO_GESVDR && sizeInBytes == sizeof c->gesvdr) {
        std::memcpy(&c->gesvdr, buffer, sizeInBytes);
        return CUTENSORNET_STATUS_SUCCESS;
      }
      return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "parameters that do not fit the algorithm");
    default: return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "unknown attribute");
  }
}

cutensornetStatus_t cutensornetCreateTensorSVDInfo(const cutensornetHandle_t handle, cutensornetTensorSVDInfo_t* svdInfo) {
  if (!handle_of(handle)) return CUTENSORNET_STATUS_NOT_INITIALIZED;
  if (!svdInfo) return CUTENSORNET_STATUS_INVALID_VALUE;
  *svdInfo = new SvdInfo;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetDestroyTensorSVDInfo(cutensornetTensorSVDInfo_t svdInfo) {
  SvdInfo* i = as<SvdInfo>(svdInfo, kMagicSvdInfo);
  if (!i) return CUTENSORNET_STATUS_INVALID_VALUE;
  i->magic = 0;
  delete i;
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetTensorSVDInfoGetAttribute(const cutensornetHandle_t handle,
                                                         const cutensornetTensorSVDInfo_t svdInfo,
                                                         cutensornetTensorSVDInfoAttributes_t attr, void* buffer,
                                                         size_t sizeInBytes) {
  const char* api = "cutensornetTensorSVDInfoGetAttribute";
  if (!handle_of(handle)) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  SvdInfo* i = as<SvdInfo>(svdInfo, kMagicSvdInfo);
  if (!i || !buffer) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  auto put = [&](const void* v, size_t n) -> Status {
    if (sizeInBytes != n) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "size mismatch");
    std::memcpy(buffer, v, n);
    return CUTENSORNET_STATUS_SUCCESS;
  };
  switch (attr) {
    case CUTENSORNET_TENSOR_SVD_INFO_FULL_EXTENT: return put(&i->full, 8);
    case CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT: return put(&i->reduced, 8);
    case CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT: return put(&i->discarded, 8);
    case CUTENSORNET_TENSOR_SVD_INFO_ALGO: return put(&i->algo, 4);
    default: return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "no status for this algorithm");
  }
}

cutensornetStatus_t cutensornetWorkspaceComputeQRSizes(const cutensornetHandle_t handle,
                                                       const cutensornetTensorDescriptor_t descTensorIn,
                                                       const cutensornetTensorDescriptor_t descTensorQ,
                                                       const cutensornetTensorDescriptor_t descTensorR,
                                                       cutensornetWorkspaceDescriptor_t workDesc) {
  const char* api = "cutensornetWorkspaceComputeQRSizes";
  Handle* h = handle_of(handle);
  if (!h) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  TensorDesc *in = tensor_of(descTensorIn), *q = tensor_of(descTensorQ), *r = tensor_of(descTensorR);
  Workspace* w = work_of(workDesc);
  if (!in || !q || !r || !w) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  Split s;
  if (Status st = split_of(*in, *q, *r, s, api)) return st;
  if (!solver_type(in->type)) return refuse(api, "QR of half-precision tensors is not implemented");
  set_scratch_need(w, qr_need(solver_of(h), in->type, s));
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetWorkspaceComputeSVDSizes(const cutensornetHandle_t handle,
                                                        const cutensornetTensorDescriptor_t descTensorIn,
                                                        const cutensornetTensorDescriptor_t descTensorU,
                                                        const cutensornetTensorDescriptor_t descTensorV,
                                                        const cutensornetTensorSVDConfig_t svdConfig,
                                                        cutensornetWorkspaceDescriptor_t workDesc) {
  const char* api = "cutensornetWorkspaceComputeSVDSizes";
  Handle* h = handle_of(handle);
  if (!h) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  TensorDesc *in = tensor_of(descTensorIn), *u = tensor_of(descTensorU), *v = tensor_of(descTensorV);
  Workspace* w = work_of(workDesc);
  if (!in || !u || !v || !w || (svdConfig && !as<SvdConfig>(svdConfig, kMagicSvdConfig)))
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  Split s;
  if (Status st = split_of(*in, *u, *v, s, api)) return st;
  if (!solver_type(in->type)) return refuse(api, "SVD of half-precision tensors is not implemented");
  set_scratch_need(w, svd_need(solver_of(h), in->type, s));
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetTensorQR(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorIn,
                                        const void* const rawDataIn, const cutensornetTensorDescriptor_t descTensorQ,
                                        void* q, const cutensornetTensorDescriptor_t descTensorR, void* r,
                                        const cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream) {
  const char* api = "cutensornetTensorQR";
  Handle* h = handle_of(handle);
  if (!h) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  TensorDesc *in = tensor_of(descTensorIn), *qd = tensor_of(descTensorQ), *rd = tensor_of(descTensorR);
  Workspace* w = work_of(workDesc);
  if (!in || !qd || !rd || (workDesc && !w)) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  return do_qr(h, *in, rawDataIn, *qd, q, *rd, r, w, stream, api);
}

cutensornetStatus_t cutensornetTensorSVD(const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorIn,
                                         const void* const rawDataIn, cutensornetTensorDescriptor_t descTensorU, void* u,
                                         void* s, cutensornetTensorDescriptor_t descTensorV, void* v,
                                         const cutensornetTensorSVDConfig_t svdConfig, cutensornetTensorSVDInfo_t svdInfo,
                                         const cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream) {
  const char* api = "cutensornetTensorSVD";
  Handle* h = handle_of(handle);
  if (!h) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  TensorDesc *in = tensor_of(descTensorIn), *ud = tensor_of(descTensorU), *vd = tensor_of(descTensorV);
  SvdConfig* cfg = as<SvdConfig>(svdConfig, kMagicSvdConfig);
  SvdInfo* info = as<SvdInfo>(svdInfo, kMagicSvdInfo);
  Workspace* w = work_of(workDesc);
  if (!in || !ud || !vd || (svdConfig && !cfg) || (svdInfo && !info) || (workDesc && !w))
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  return do_svd(h, *in, rawDataIn, *ud, u, s, *vd, v, cfg, info, w, stream, api);
}

}  // extern "C"

namespace {

// Gate splitting: theta = A B G contracted to U's and V's open modes, then
// the SVD of theta. (The REDUCED algorithm's QR steps change the cost, not
// the result, so both algorithms take this route.)
struct GateSetup {
  Network net;
  TensorDesc theta;
  Path path;
  int64_t theta_bytes = 0, contract_need = 0;
  Split s;
};

Status gate_setup(const TensorDesc& a, const TensorDesc& b, const TensorDesc& g, const TensorDesc& u,
                  const TensorDesc& v, cutensornetComputeType_t compute, GateSetup& gs, const char* api) {
  if (b.type != a.type || g.type != a.type || u.type != a.type || v.type != a.type)
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "the tensors' types differ");
  if (!compute_ok(a.type, compute)) return fail(CUTENSORNET_STATUS_NOT_SUPPORTED, api, "unsupported compute type");
  gs.net.type = a.type;
  gs.net.type_set = true;
  gs.net.compute = compute;
  for (const TensorDesc* t : {&a, &b, &g}) {
    NetTensor nt;
    nt.modes = t->modes;
    nt.extents = t->extents;
    nt.strides = t->strides;
    gs.net.in.push_back(nt);
  }
  std::vector<int32_t> shared;
  for (int32_t m : u.modes)
    if (contains(v.modes, m)) shared.push_back(m);
  if (shared.size() != 1) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "U and V must share exactly one mode");
  const auto ext = extent_map(gs.net);
  for (const TensorDesc* t : {&u, &v})
    for (size_t i = 0; i < t->modes.size(); ++i) {
      if (t->modes[i] == shared[0]) continue;
      auto it = ext.find(t->modes[i]);
      if (it == ext.end() || it->second != t->extents[i])
        return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "an output mode is not an input mode with the same extent");
      gs.net.out.modes.push_back(t->modes[i]);
      gs.net.out.extents.push_back(t->extents[i]);
    }
  gs.net.out_set = true;
  gs.path.steps = greedy_path(gs.net, ext);
  gs.theta.modes = gs.net.out.modes;
  gs.theta.extents = gs.net.out.extents;
  gs.theta.strides = packed_strides(gs.theta.extents);
  gs.theta.type = a.type;
  gs.theta_bytes = align_up(volume(gs.theta.extents) * (int64_t)elem_bytes(a.type));
  gs.contract_need = scratch_need(gs.net, gs.path);
  return split_of(gs.theta, u, v, gs.s, api);
}

}  // namespace

extern "C" {

cutensornetStatus_t cutensornetWorkspaceComputeGateSplitSizes(
    const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorInA,
    const cutensornetTensorDescriptor_t descTensorInB, const cutensornetTensorDescriptor_t descTensorInG,
    const cutensornetTensorDescriptor_t descTensorU, const cutensornetTensorDescriptor_t descTensorV,
    const cutensornetGateSplitAlgo_t gateAlgo, const cutensornetTensorSVDConfig_t svdConfig,
    cutensornetComputeType_t computeType, cutensornetWorkspaceDescriptor_t workDesc) {
  const char* api = "cutensornetWorkspaceComputeGateSplitSizes";
  Handle* h = handle_of(handle);
  if (!h) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  TensorDesc *a = tensor_of(descTensorInA), *b = tensor_of(descTensorInB), *g = tensor_of(descTensorInG),
             *u = tensor_of(descTensorU), *v = tensor_of(descTensorV);
  Workspace* w = work_of(workDesc);
  if (!a || !b || !g || !u || !v || !w || (gateAlgo != 0 && gateAlgo != 1) ||
      (svdConfig && !as<SvdConfig>(svdConfig, kMagicSvdConfig)))
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  if (!solver_type(a->type)) return refuse(api, "gate splitting of half-precision tensors is not implemented");
  GateSetup gs;
  if (Status st = gate_setup(*a, *b, *g, *u, *v, computeType, gs, api)) return st;
  set_scratch_need(w, gs.theta_bytes + std::max(gs.contract_need, svd_need(solver_of(h), a->type, gs.s)));
  return CUTENSORNET_STATUS_SUCCESS;
}

cutensornetStatus_t cutensornetGateSplit(
    const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorInA, const void* rawDataInA,
    const cutensornetTensorDescriptor_t descTensorInB, const void* rawDataInB,
    const cutensornetTensorDescriptor_t descTensorInG, const void* rawDataInG, cutensornetTensorDescriptor_t descTensorU,
    void* u, void* s, cutensornetTensorDescriptor_t descTensorV, void* v, const cutensornetGateSplitAlgo_t gateAlgo,
    const cutensornetTensorSVDConfig_t svdConfig, cutensornetComputeType_t computeType,
    cutensornetTensorSVDInfo_t svdInfo, const cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream) {
  const char* api = "cutensornetGateSplit";
  Handle* h = handle_of(handle);
  if (!h) return fail(CUTENSORNET_STATUS_NOT_INITIALIZED, api, "invalid handle");
  TensorDesc *a = tensor_of(descTensorInA), *b = tensor_of(descTensorInB), *g = tensor_of(descTensorInG),
             *ud = tensor_of(descTensorU), *vd = tensor_of(descTensorV);
  SvdConfig* cfg = as<SvdConfig>(svdConfig, kMagicSvdConfig);
  SvdInfo* info = as<SvdInfo>(svdInfo, kMagicSvdInfo);
  Workspace* w = work_of(workDesc);
  if (!a || !b || !g || !ud || !vd || (gateAlgo != 0 && gateAlgo != 1) || (svdConfig && !cfg) ||
      (svdInfo && !info) || (workDesc && !w))
    return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "invalid argument");
  if (!rawDataInA || !rawDataInB || !rawDataInG) return fail(CUTENSORNET_STATUS_INVALID_VALUE, api, "a data pointer is null");
  if (!solver_type(a->type)) return refuse(api, "gate splitting of half-precision tensors is not implemented");
  GateSetup gs;
  if (Status st = gate_setup(*a, *b, *g, *ud, *vd, computeType, gs, api)) return st;
  if (Status st = settle(stream, api)) return st;
  const int64_t need = gs.theta_bytes + std::max(gs.contract_need, svd_need(solver_of(h), a->type, gs.s));
  Scratch scratch;
  if (Status st = get_scratch(h, w, need, stream, scratch, api)) return st;
  // theta first, then the rest of the scratch for the contraction and, after
  // it, the SVD.
  Workspace rest;
  rest.ptr[0][0] = static_cast<char*>(scratch.ptr) + gs.theta_bytes;
  rest.size[0][0] = (int64_t)scratch.size - gs.theta_bytes;
  std::vector<int64_t> ids{0};
  if (Status st = contract_slices(h, gs.net, gs.path, {rawDataInA, rawDataInB, rawDataInG}, scratch.ptr, false, &rest,
                                  ids, stream, api))
    return st;
  return do_svd(h, gs.theta, scratch.ptr, *ud, u, s, *vd, v, cfg, info, &rest, stream, api);
}

}  // extern "C"

// ---- the state API ----
#include "cutensornet_state.inc"

// ---- not implemented ----
//
// Gradients, the MPS projection, distributed execution and NVIDIA's
// undocumented exports. Each is exported, so a program linked against
// NVIDIA's library loads, and answers NOT_SUPPORTED with a message.
extern "C" {
#define VGPU_TN_NOT_IMPLEMENTED(name)                                           \
  cutensornetStatus_t name(...) {                                               \
    return refuse(#name, "this part of cuTensorNet is not implemented");        \
  }
VGPU_TN_NOT_IMPLEMENTED(cutensornetBinaryTensorContractionCompute)
VGPU_TN_NOT_IMPLEMENTED(cutensornetBinaryTensorContractionPrepare)
VGPU_TN_NOT_IMPLEMENTED(cutensornetCreateBinaryTensorContraction)
VGPU_TN_NOT_IMPLEMENTED(cutensornetCreateCopyContractionOptimizerInfo)
VGPU_TN_NOT_IMPLEMENTED(cutensornetCreateDistributedTensorDescriptor)
VGPU_TN_NOT_IMPLEMENTED(cutensornetCreateStateProjectionMPS)
VGPU_TN_NOT_IMPLEMENTED(cutensornetDestroyBinaryTensorContraction)
VGPU_TN_NOT_IMPLEMENTED(cutensornetDestroyStateProjectionMPS)
VGPU_TN_NOT_IMPLEMENTED(cutensornetDistributedGetNumRanks)
VGPU_TN_NOT_IMPLEMENTED(cutensornetDistributedGetProcRank)
VGPU_TN_NOT_IMPLEMENTED(cutensornetDistributedResetConfiguration)
VGPU_TN_NOT_IMPLEMENTED(cutensornetDistributedSynchronize)
VGPU_TN_NOT_IMPLEMENTED(cutensornetStateProjectionMPSComputeTensorEnv)
VGPU_TN_NOT_IMPLEMENTED(cutensornetStateProjectionMPSConfigure)
VGPU_TN_NOT_IMPLEMENTED(cutensornetStateProjectionMPSExtractTensor)
VGPU_TN_NOT_IMPLEMENTED(cutensornetStateProjectionMPSGetTensorInfo)
VGPU_TN_NOT_IMPLEMENTED(cutensornetStateProjectionMPSInsertTensor)
VGPU_TN_NOT_IMPLEMENTED(cutensornetStateProjectionMPSPrepare)
VGPU_TN_NOT_IMPLEMENTED(cutensornetStateProjectionMPSUpdateCoefficients)
VGPU_TN_NOT_IMPLEMENTED(cutensornetStateProjectionMPSUpdateDualTensors)
#undef VGPU_TN_NOT_IMPLEMENTED
}  // extern "C"
