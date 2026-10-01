// libvgpucudss -- VirtualGPU's cuDSS, presented as libcudss.so.0.
//
// NVIDIA's libcudss cannot run on a simulated GPU: it carries a statically
// linked CUDA runtime, which reaches the driver through NVIDIA's
// undocumented internal interface, so its first call (cudssCreate) fails.
// This is the documented API (cuDSS 0.8) implemented the way the simulator's
// other vendor libraries are: a call reads its operands out of simulated
// device memory, solves on the host in double precision with the sparse
// direct solver in cudss_solver.cpp, and writes the results back.
// Application kernels are simulated; vendor library calls are implemented.
//
// SCS's GPU direct backend is the first user: a symmetric indefinite KKT
// matrix in lower CSR, analysed and factored once, solved every iteration,
// refactored when its diagonal changes, its inertia read back. Beyond that
// the whole API answers: every matrix type, view, index base, index width
// and value type cuDSS takes, several right-hand sides, the solve
// sub-phases, iterative refinement, the pivot epsilon, user permutations,
// uniform and non-uniform batches, and graph capture of the factorization
// and solve phases.
//
// What is measured and what is our own. The status codes, defaults, sizes
// and phase rules follow NVIDIA's library on an RTX 3060 where the
// documentation leaves them open (exact-size checks in the Get/Set calls, a
// zero size asking for the size, which calls refuse and with what). What
// depends on the factorization -- the permutation, LU_NNZ, FLOPS, DIAG, the
// pivots that needed the epsilon -- is this solver's own: it orders by
// minimum degree and pivots with 1x1 and 2x2 pivots and delays, where NVIDIA
// orders by nested dissection and pivots on the diagonal of each
// supernode. Inertia, solutions and residuals are the matrix's, so those
// agree. Refused, with a message: Schur complements, nested dissection
// trees, double-double values and a matrix split across processes.
#include "../include/vgpu_cudss.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "cudss_solver.hpp"
#include "vgpu/runtime/capture.hpp"

namespace {

using cd = std::complex<double>;
using cf = std::complex<float>;
using Status = cudssStatus_t;
namespace dss = vgpu_dss;

// ---- logging ----
//
// The library's own messages are its refusals. They go to stderr as every
// VirtualGPU library's do (VGPU_QUIET=1 silences them), and to the caller's
// logger callback or file when one is set and its level admits errors.
struct Logger {
  cudssLoggerCallback_t callback = nullptr;
  FILE* file = nullptr;
  bool own_file = false, disabled = false;
  int level = 0, mask = 0;
};
Logger& logger() {
  static Logger* l = new Logger;
  return *l;
}

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

void say(const char* api, const std::string& why) {
  if (!quiet()) std::fprintf(stderr, "[vgpu] %s: %s\n", api, why.c_str());
  Logger& l = logger();
  if (l.disabled || (l.level < 1 && !(l.mask & 1))) return;
  if (l.callback) l.callback(1, api, why.c_str());
  if (l.file) std::fprintf(l.file, "[cuDSS][Error][%s] %s\n", api, why.c_str());
}

Status refuse(const char* api, const std::string& why) {
  say(api, why);
  return CUDSS_STATUS_NOT_SUPPORTED;
}

// ---- types ----

size_t type_size(cudssDataType_t t) {
  switch (t) {
    case CUDSS_R_32F: case CUDSS_R_32I: return 4;
    case CUDSS_R_64F: case CUDSS_C_32F: case CUDSS_R_64I: return 8;
    case CUDSS_C_64F: case CUDSS_R_64F_64F: return 16;
    default: return 0;
  }
}
bool value_type(cudssDataType_t t) {
  return t == CUDSS_R_32F || t == CUDSS_R_64F || t == CUDSS_C_32F || t == CUDSS_C_64F || t == CUDSS_R_64F_64F;
}
bool index_type(cudssDataType_t t) { return t == CUDSS_R_32I || t == CUDSS_R_64I; }
bool complex_type(cudssDataType_t t) { return t == CUDSS_C_32F || t == CUDSS_C_64F; }
bool single_type(cudssDataType_t t) { return t == CUDSS_R_32F || t == CUDSS_C_32F; }

// ---- simulated device memory ----
//
// cuDSS takes the CSR arrays, the user permutation and DataGet's outputs from
// host or device memory alike. Every access is a cudaMemcpy with
// cudaMemcpyDefault, so the runtime decides which it is and a device side
// goes through the simulator's bounds-checked memory: a copy past the end of
// an allocation fails (and the call with it) rather than writing on.
bool read_bytes(void* dst, const void* src, size_t n) {
  if (!n) return true;
  if (!src || !dst) return false;
  const bool ok = cudaMemcpy(dst, src, n, cudaMemcpyDefault) == cudaSuccess;
  cudaGetLastError();
  return ok;
}

bool write_bytes(void* dst, const void* src, size_t n) {
  if (!n) return true;
  if (!dst || !src) return false;
  const bool ok = cudaMemcpy(dst, src, n, cudaMemcpyDefault) == cudaSuccess;
  cudaGetLastError();
  return ok;
}

bool read_indices(const void* p, cudssDataType_t t, size_t count, std::vector<int64_t>& out) {
  out.assign(count, 0);
  if (!count) return true;
  if (t == CUDSS_R_64I) return read_bytes(out.data(), p, count * 8);
  std::vector<int32_t> v(count);
  if (!read_bytes(v.data(), p, count * 4)) return false;
  for (size_t i = 0; i < count; ++i) out[i] = v[i];
  return true;
}

template <class T>
bool read_values(const void* p, cudssDataType_t t, size_t count, std::vector<T>& out) {
  out.assign(count, T(0));
  if (!count) return true;
  switch (t) {
    case CUDSS_R_64F: {
      std::vector<double> v(count);
      if (!read_bytes(v.data(), p, count * 8)) return false;
      for (size_t i = 0; i < count; ++i) out[i] = T(v[i]);
      return true;
    }
    case CUDSS_R_32F: {
      std::vector<float> v(count);
      if (!read_bytes(v.data(), p, count * 4)) return false;
      for (size_t i = 0; i < count; ++i) out[i] = T(v[i]);
      return true;
    }
    case CUDSS_C_64F:
      if constexpr (std::is_same_v<T, cd>) return read_bytes(out.data(), p, count * 16);
      return false;
    case CUDSS_C_32F:
      if constexpr (std::is_same_v<T, cd>) {
        std::vector<cf> v(count);
        if (!read_bytes(v.data(), p, count * 8)) return false;
        for (size_t i = 0; i < count; ++i) out[i] = cd(v[i].real(), v[i].imag());
        return true;
      }
      return false;
    default: return false;
  }
}

template <class T>
std::vector<uint8_t> encode(const T* v, size_t count, cudssDataType_t t) {
  std::vector<uint8_t> out(count * type_size(t));
  for (size_t i = 0; i < count; ++i) {
    uint8_t* d = out.data() + i * type_size(t);
    const cd z(v[i]);
    if (t == CUDSS_R_64F) { const double x = z.real(); std::memcpy(d, &x, 8); }
    else if (t == CUDSS_R_32F) { const float x = (float)z.real(); std::memcpy(d, &x, 4); }
    else if (t == CUDSS_C_64F) std::memcpy(d, &z, 16);
    else if (t == CUDSS_C_32F) { const cf x((float)z.real(), (float)z.imag()); std::memcpy(d, &x, 8); }
  }
  return out;
}

template <class T>
bool write_values(void* p, cudssDataType_t t, const T* v, size_t count) {
  const auto bytes = encode(v, count, t);
  return write_bytes(p, bytes.data(), bytes.size());
}

std::vector<uint8_t> encode_indices(const int64_t* v, size_t count, cudssDataType_t t, int64_t add = 0) {
  std::vector<uint8_t> out(count * type_size(t));
  for (size_t i = 0; i < count; ++i) {
    const int64_t x = v[i] + add;
    if (t == CUDSS_R_64I) std::memcpy(out.data() + i * 8, &x, 8);
    else { const int32_t y = (int32_t)x; std::memcpy(out.data() + i * 4, &y, 4); }
  }
  return out;
}

}  // namespace

// ---- the opaque objects ----

struct cudssContext {
  int device = 0;
  cudaStream_t stream = nullptr;
  cudssDeviceMemHandler_t mem{};
  bool has_mem = false;
  std::string threading_lib, comm_lib;  // recorded; neither is loaded
  int mg_count = 0;                     // devices of a cudssCreateMg handle
  std::vector<int> mg_devices;
  std::vector<cudaStream_t> mg_streams;
};

struct cudssConfig {
  int reordering = 0, factorization = 0, solve = 0, matching = 0, solve_mode = 0, ir_steps = 0;
  double ir_tol = 0;
  int pivot_type = 0;
  double pivot_threshold = 1.0, pivot_epsilon = -1.0;  // -1: the value type's default
  int64_t max_lu_nnz = -1;
  int hybrid_memory = 0;
  int64_t hybrid_limit = -1;
  int register_memory = 1, host_threads = -1, hybrid_execute = 0, epsilon_alg = 0, nd_levels = 10;
  int ubatch_size = 1, ubatch_index = -1, superpanels = 1, device_count = 1;
  std::vector<int> device_indices;  // empty: 0 .. device_count - 1
  int schur_mode = 0, deterministic = 0, nd_ubfactor = 20;
};

// A thin wrapper over the caller's buffers, read when the matrix is used.
struct cudssMatrix {
  int format = 0;
  int64_t rows = 0, cols = 0;
  // dense
  int64_t ld = 0;
  cudssLayout_t layout = CUDSS_LAYOUT_COL_MAJOR;
  // CSR
  int64_t nnz = 0;
  const void *row_start = nullptr, *row_end = nullptr, *col_ind = nullptr;
  cudssDataType_t otype = CUDSS_R_32I, itype = CUDSS_R_32I;
  cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;
  cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;
  cudssIndexBase_t base = CUDSS_BASE_ZERO;
  // both
  const void* values = nullptr;
  cudssDataType_t vtype = CUDSS_R_64F;
  // a non-uniform batch: the caller's per-matrix arrays (sizes in host or
  // device memory, pointer arrays likewise)
  int64_t batch = 0;
  const void *b_rows = nullptr, *b_cols = nullptr, *b_third = nullptr;  // nnz (CSR) or ld (dense)
  cudssDataType_t b_int = CUDSS_R_32I;
  const void* const* b_row_start = nullptr;
  const void* const* b_row_end = nullptr;
  const void* const* b_col_ind = nullptr;
  const void* const* b_values = nullptr;
  // MGMN row distribution
  bool distributed = false;
  int64_t first_row = 0, last_row = -1;
};

namespace {

// One system of the batch (or the only one): its factorization, and the
// matrix as it was factored, for iterative refinement.
struct System {
  dss::Factor<double> fr;
  dss::Factor<cd> fc;
  bool factored = false;
  dss::Pattern pattern;
  std::vector<double> vr;
  std::vector<cd> vc;
};

dss::Factor<double>& factor_of(System& s, double*) { return s.fr; }
dss::Factor<cd>& factor_of(System& s, cd*) { return s.fc; }
std::vector<double>& values_of(System& s, double*) { return s.vr; }
std::vector<cd>& values_of(System& s, cd*) { return s.vc; }

enum Stage { kNone = 0, kReordered = 1, kAnalysed = 2, kFactored = 3 };

}  // namespace

struct cudssData {
  Stage stage = kNone;
  bool pending_factor = false;  // a factorization recorded into a graph, not yet run
  // What the analysis saw.
  int64_t n = 0;
  cudssDataType_t itype = CUDSS_R_32I, vtype = CUDSS_R_64F;
  cudssIndexBase_t base = CUDSS_BASE_ZERO;
  dss::Kind kind = dss::Kind::General;
  bool uniform = true;           // one pattern shared by every system
  bool matching = false;         // asked for (and not performed)
  std::vector<dss::Symbolic> sym;  // one per pattern
  std::vector<System> sys;
  std::vector<int64_t> sizes;    // each system's n (a non-uniform batch's differ)
  // Results.
  int info = 0, ir_steps = 0;
  // Set by the caller.
  std::vector<uint8_t> user_perm, user_schur, user_tree;
  void *comm_device = nullptr, *comm_host = nullptr;
  int* interrupt = nullptr;
  int64_t ubatch_mask = -1;
  cudssMatrix_t schur_matrix = nullptr;
};

namespace {

// ---- reading a system's matrix ----

// The b-th matrix of a batch as if it were a matrix of its own.
bool batch_member(const cudssMatrix& m, int64_t b, cudssMatrix& out) {
  out = m;
  out.batch = 0;
  out.format &= ~CUDSS_MFORMAT_BATCH;
  std::vector<int64_t> r, c, t;
  if (!read_indices((const char*)m.b_rows + b * type_size(m.b_int), m.b_int, 1, r) ||
      !read_indices((const char*)m.b_cols + b * type_size(m.b_int), m.b_int, 1, c) ||
      !read_indices((const char*)m.b_third + b * type_size(m.b_int), m.b_int, 1, t))
    return false;
  out.rows = r[0];
  out.cols = c[0];
  const void* p[4] = {nullptr, nullptr, nullptr, nullptr};
  auto get = [&](const void* const* arr, int k) {
    return !arr || read_bytes(&p[k], (const char*)arr + b * sizeof(void*), sizeof(void*));
  };
  if (!get(m.b_values, 3)) return false;
  out.values = p[3];
  if (m.format & CUDSS_MFORMAT_CSR) {
    if (!get(m.b_row_start, 0) || !get(m.b_row_end, 1) || !get(m.b_col_ind, 2)) return false;
    out.nnz = t[0];
    out.row_start = p[0];
    out.row_end = p[1];
    out.col_ind = p[2];
  } else {
    out.ld = t[0];
  }
  return true;
}

dss::Kind kind_of(cudssMatrixType_t t, bool complex) {
  switch (t) {
    case CUDSS_MTYPE_SYMMETRIC: return dss::Kind::Symmetric;
    case CUDSS_MTYPE_HERMITIAN: return complex ? dss::Kind::Hermitian : dss::Kind::Symmetric;
    case CUDSS_MTYPE_SPD: return dss::Kind::Spd;
    case CUDSS_MTYPE_HPD: return complex ? dss::Kind::Hpd : dss::Kind::Spd;
    default: return dss::Kind::General;
  }
}

dss::View view_of(cudssMatrixViewType_t v) {
  return v == CUDSS_MVIEW_LOWER ? dss::View::Lower : v == CUDSS_MVIEW_UPPER ? dss::View::Upper : dss::View::Full;
}

// A CSR matrix in canonical form, values optional (`values_at` null: the
// pattern only). The b-th set of values of a uniform batch starts b * nnz in.
template <class T>
Status read_csr(const cudssMatrix& m, bool with_values, int64_t value_set, dss::Pattern* pattern,
                std::vector<T>* values) {
  if (m.rows != m.cols || m.nnz < 0 || !m.row_start || !m.col_ind) return CUDSS_STATUS_INVALID_VALUE;
  if (with_values && !m.values) return CUDSS_STATUS_INVALID_VALUE;
  const int64_t n = m.rows, off = m.base == CUDSS_BASE_ONE ? 1 : 0;
  std::vector<int64_t> start, end, col;
  if (!read_indices(m.row_start, m.otype, (size_t)(m.row_end ? n : n + 1), start) ||
      (m.row_end && !read_indices(m.row_end, m.otype, (size_t)n, end)) ||
      !read_indices(m.col_ind, m.itype, (size_t)m.nnz, col))
    return CUDSS_STATUS_EXECUTION_FAILED;
  for (auto& x : start) x -= off;
  for (auto& x : end) x -= off;
  for (auto& x : col) x -= off;
  std::vector<T> vals;
  if (with_values) {
    const char* base = (const char*)m.values + value_set * m.nnz * (int64_t)type_size(m.vtype);
    if (!read_values(base, m.vtype, (size_t)m.nnz, vals)) return CUDSS_STATUS_EXECUTION_FAILED;
  }
  if (!dss::canonical(kind_of(m.mtype, complex_type(m.vtype)), view_of(m.mview), n, start, end, col, vals,
                      pattern, values))
    return CUDSS_STATUS_INVALID_VALUE;
  return CUDSS_STATUS_SUCCESS;
}

// ---- the phases ----

// Work issued on the handle's streams must have finished before their memory
// is read.
Status settle(cudssHandle_t h) {
  cudaGetLastError();
  bool ok = cudaStreamSynchronize(h->stream) == cudaSuccess;
  for (cudaStream_t s : h->mg_streams) ok = cudaStreamSynchronize(s) == cudaSuccess && ok;
  return ok ? CUDSS_STATUS_SUCCESS : CUDSS_STATUS_EXECUTION_FAILED;
}

bool capturing(cudssHandle_t h) {
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  const bool on = cudaStreamIsCapturing(h->stream, &st) == cudaSuccess && st == cudaStreamCaptureStatusActive;
  cudaGetLastError();
  return on;
}

const int kSolveBits = CUDSS_PHASE_SOLVE;

// The systems a factorization or solve touches: one with UBATCH_INDEX set,
// those in UBATCH_MASK otherwise.
std::vector<int64_t> selected(const cudssConfig& cfg, const cudssData& d, bool* single) {
  std::vector<int64_t> out;
  const int64_t count = (int64_t)d.sys.size();
  *single = false;
  if (d.uniform && count > 1 && cfg.ubatch_index >= 0) {
    if (cfg.ubatch_index < count) out.push_back(cfg.ubatch_index);
    *single = true;
    return out;
  }
  for (int64_t b = 0; b < count; ++b)
    if (count == 1 || b >= 64 || ((d.ubatch_mask >> b) & 1)) out.push_back(b);
  return out;
}

Status analyse(const cudssConfig& cfg, cudssData* d, const cudssMatrix& A) {
  if (cfg.reordering < CUDSS_REORDERING_ALG_DEFAULT || cfg.reordering > CUDSS_REORDERING_ALG_NONE)
    return CUDSS_STATUS_INVALID_VALUE;
  if (cfg.schur_mode)
    return refuse("cudssExecute", "the Schur complement mode is not supported by this simulator's cuDSS");
  const bool complex = complex_type(A.vtype);
  const int64_t count = A.batch ? A.batch : std::max(cfg.ubatch_size, 1);
  std::vector<dss::Symbolic> syms;
  std::vector<int64_t> sizes;
  for (int64_t b = 0; b < (A.batch ? A.batch : 1); ++b) {
    cudssMatrix m = A;
    if (A.batch && !batch_member(A, b, m)) return CUDSS_STATUS_EXECUTION_FAILED;
    dss::Pattern pattern;
    std::vector<double> none;  // the pattern alone: values are not read
    if (Status s = read_csr<double>(m, false, 0, &pattern, &none)) return s;
    dss::Order order = cfg.reordering == CUDSS_REORDERING_ALG_NONE ? dss::Order::Natural
                                                                    : dss::Order::MinimumDegree;
    std::vector<int64_t> user;
    if (!d->user_perm.empty() && !A.batch) {
      // The caller's permutation replaces the reordering; cuDSS reads it in
      // the matrix's index type and base.
      if (d->user_perm.size() != (size_t)m.rows * type_size(m.itype)) return CUDSS_STATUS_INVALID_VALUE;
      read_indices(d->user_perm.data(), m.itype, (size_t)m.rows, user);
      std::vector<char> seen((size_t)m.rows, 0);
      for (auto& v : user) {
        v -= m.base == CUDSS_BASE_ONE ? 1 : 0;
        if (v < 0 || v >= m.rows || seen[v]) return CUDSS_STATUS_INVALID_VALUE;
        seen[v] = 1;
      }
      order = dss::Order::User;
    }
    syms.push_back(dss::analyse(kind_of(A.mtype, complex), pattern, order, user));
    sizes.push_back(m.rows);
  }
  // n is the system's size, or a non-uniform batch's total: what DIAG holds.
  d->n = 0;
  for (int64_t s : sizes) d->n += s;
  d->sym = std::move(syms);
  d->sizes = std::move(sizes);
  d->uniform = !A.batch;
  d->sys.assign((size_t)count, System{});
  if (d->uniform) d->sizes.assign((size_t)count, d->sizes[0]);
  d->matching = cfg.matching != CUDSS_MATCHING_ALG_NONE;
  d->itype = A.itype;
  d->vtype = A.vtype;
  d->base = A.base;
  d->kind = kind_of(A.mtype, complex);
  d->info = 0;
  d->ir_steps = 0;
  return CUDSS_STATUS_SUCCESS;
}

template <class T>
Status factor(const cudssConfig& cfg, cudssData* d, const cudssMatrix& A) {
  bool single = false;
  const auto which = selected(cfg, *d, &single);
  dss::Options opt;
  opt.pivoting = cfg.pivot_type != CUDSS_PIVOT_NONE;
  opt.eps = cfg.pivot_epsilon >= 0 ? cfg.pivot_epsilon : single_type(d->vtype) ? 1e-5 : 1e-13;
  opt.eps_scaled = cfg.epsilon_alg == CUDSS_PIVOT_EPSILON_ALG_SCALED;
  d->info = 0;
  for (int64_t b : which) {
    cudssMatrix m = A;
    if (A.batch && !batch_member(A, b, m)) return CUDSS_STATUS_EXECUTION_FAILED;
    if (m.rows != d->sizes[b]) return CUDSS_STATUS_INVALID_VALUE;
    System& s = d->sys[b];
    std::vector<T>& vals = values_of(s, (T*)nullptr);
    if (Status st = read_csr<T>(m, true, d->uniform && !single ? b : 0, &s.pattern, &vals)) return st;
    dss::Symbolic& sym = d->sym[d->uniform ? 0 : b];
    // A pattern other than the analysed one keeps the analysed order but
    // needs its own symbolic factorization.
    if (!(s.pattern == sym.pattern)) sym = dss::analyse(sym.kind, s.pattern, dss::Order::User, sym.perm);
    factor_of(s, (T*)nullptr) = dss::factorize(sym, s.pattern, vals, opt);
    s.factored = true;
    if (!d->info) d->info = factor_of(s, (T*)nullptr).info;
  }
  return CUDSS_STATUS_SUCCESS;
}

// The dense right-hand side or solution of system b: its first element and
// shape.
struct Dense {
  char* p = nullptr;
  int64_t rows = 0, cols = 0, ld = 0;
};

bool dense_of(const cudssMatrix& m, int64_t b, bool uniform_offset, Dense* out) {
  cudssMatrix x = m;
  if (m.batch && !batch_member(m, b, x)) return false;
  out->rows = x.rows;
  out->cols = x.cols;
  out->ld = x.ld;
  out->p = (char*)x.values;
  if (uniform_offset) out->p += b * x.ld * x.cols * (int64_t)type_size(x.vtype);
  return true;
}

template <class T>
Status solve(const cudssConfig& cfg, cudssData* d, const cudssMatrix& A, const cudssMatrix* X,
             const cudssMatrix* B, int phase) {
  if (!X || !B || !(X->format & CUDSS_MFORMAT_DENSE) || !(B->format & CUDSS_MFORMAT_DENSE))
    return CUDSS_STATUS_INVALID_VALUE;
  if (X->vtype != d->vtype || B->vtype != d->vtype)
    return refuse("cudssExecute", "the right-hand side and solution must have the matrix's value type");
  if (X->layout != CUDSS_LAYOUT_COL_MAJOR || B->layout != CUDSS_LAYOUT_COL_MAJOR)
    return refuse("cudssExecute", "only column-major right-hand sides and solutions are supported");
  bool single = false;
  const auto which = selected(cfg, *d, &single);
  const bool offset = d->uniform && d->sys.size() > 1 && !single;
  const int ir = (phase & CUDSS_PHASE_SOLVE_REFINEMENT) ? cfg.ir_steps : 0;
  const bool substitute = phase & (kSolveBits & ~CUDSS_PHASE_SOLVE_REFINEMENT);
  Status result = CUDSS_STATUS_SUCCESS;
  int steps_done = 0;
  for (int64_t b : which) {
    System& s = d->sys[b];
    if (!s.factored) return CUDSS_STATUS_INVALID_VALUE;
    const dss::Symbolic& sym = d->sym[d->uniform ? 0 : b];
    const auto& f = factor_of(s, (T*)nullptr);
    const int64_t n = sym.n;
    Dense xb, bb;
    if (!dense_of(*B, b, offset, &bb) || !dense_of(*X, b, offset, &xb)) return CUDSS_STATUS_EXECUTION_FAILED;
    if (bb.rows != n || xb.rows != n || bb.ld < n || xb.ld < n || xb.cols < bb.cols || (n && (!bb.p || !xb.p)))
      return CUDSS_STATUS_INVALID_VALUE;
    const int64_t nrhs = bb.cols;
    if (!n || !nrhs) continue;
    std::vector<T> rhs, sol;
    const size_t span = (size_t)((nrhs - 1) * bb.ld + n);
    if (!read_values(bb.p, B->vtype, span, rhs)) return CUDSS_STATUS_EXECUTION_FAILED;
    if (ir && !substitute && !read_values(xb.p, X->vtype, (size_t)((nrhs - 1) * xb.ld + n), sol))
      return CUDSS_STATUS_EXECUTION_FAILED;
    // Refinement measures the residual against the matrix this call was
    // given, which cuDSS allows to differ from the factored one.
    dss::Pattern pat = s.pattern;
    std::vector<T> vals = values_of(s, (T*)nullptr);
    if (ir) {
      cudssMatrix m = A;
      if (A.batch && !batch_member(A, b, m)) return CUDSS_STATUS_EXECUTION_FAILED;
      if (read_csr<T>(m, true, offset ? b : 0, &pat, &vals)) pat = s.pattern, vals = values_of(s, (T*)nullptr);
    }
    std::vector<T> w(n), z(n), x(n), r(n), ax(n);
    auto full_solve = [&](const T* in, T* out) {
      dss::permute(sym, in, w.data());
      dss::forward(f, w.data());
      dss::diagonal(f, w.data());
      dss::backward(f, w.data(), z.data());
      dss::unpermute(sym, z.data(), out);
    };
    for (int64_t j = 0; j < nrhs; ++j) {
      const T* bj = rhs.data() + j * bb.ld;
      if (substitute) {
        // Each sub-phase reads what the one before it wrote; the first
        // reads the right-hand side.
        std::copy(bj, bj + n, w.begin());
        if (phase & CUDSS_PHASE_SOLVE_FWD_PERM) dss::permute(sym, bj, w.data());
        if (phase & CUDSS_PHASE_SOLVE_FWD) dss::forward(f, w.data());
        if (phase & CUDSS_PHASE_SOLVE_DIAG) dss::diagonal(f, w.data());
        if (phase & CUDSS_PHASE_SOLVE_BWD) {
          dss::backward(f, w.data(), z.data());
          w.swap(z);
        }
        if (phase & CUDSS_PHASE_SOLVE_BWD_PERM) dss::unpermute(sym, w.data(), x.data());
        else x = w;
      } else if (ir) {
        std::copy(sol.begin() + j * xb.ld, sol.begin() + j * xb.ld + n, x.begin());
      }
      if (ir) {
        // Up to IR_N_STEPS corrections x += A^-1 (b - A x). With a tolerance,
        // it stops once ||b - A x|| / ||b|| is below it, and fails if it never
        // gets there.
        double bnorm = 0;
        for (int64_t i = 0; i < n; ++i) bnorm += std::norm(cd(bj[i]));
        bnorm = std::sqrt(bnorm);
        int steps = 0;
        auto residual = [&] {
          dss::multiply(sym.kind, pat, vals, x.data(), ax.data());
          double rn = 0;
          for (int64_t i = 0; i < n; ++i) r[i] = bj[i] - ax[i], rn += std::norm(cd(r[i]));
          return std::sqrt(rn);
        };
        for (; steps < ir; ++steps) {
          const double rn = residual();
          if (cfg.ir_tol > 0 && rn <= cfg.ir_tol * bnorm) break;
          std::vector<T> dx(n);
          full_solve(r.data(), dx.data());
          for (int64_t i = 0; i < n; ++i) x[i] += dx[i];
        }
        if (cfg.ir_tol > 0 && steps == ir && residual() > cfg.ir_tol * bnorm) result = CUDSS_STATUS_IR_FAILED;
        steps_done = std::max(steps_done, steps);
      }
      if (substitute || ir)
        if (!write_values(xb.p + j * xb.ld * (int64_t)type_size(X->vtype), X->vtype, x.data(), (size_t)n))
          return CUDSS_STATUS_EXECUTION_FAILED;
    }
  }
  d->ir_steps = steps_done;
  return result;
}

// Everything cudssExecute checks before it does any work, so that a call
// recorded into a graph fails when it is made, as the hardware's does.
Status check(cudssHandle_t h, int phase, const cudssConfig* cfg, cudssData* d, const cudssMatrix* A,
             const cudssMatrix* X, const cudssMatrix* B, bool* pending_ok) {
  if (!h || !cfg || !d || !A) return CUDSS_STATUS_INVALID_VALUE;
  const int known = 0x3ff;
  if (phase <= 0 || (phase & ~known)) return CUDSS_STATUS_INVALID_VALUE;
  // Phases combine only in their order, without gaps: reordering, symbolic
  // factorization, (re)factorization, solve.
  const bool R = phase & CUDSS_PHASE_REORDERING, S = phase & CUDSS_PHASE_SYMBOLIC_FACTORIZATION,
             F = phase & CUDSS_PHASE_FACTORIZATION, RF = phase & CUDSS_PHASE_REFACTORIZATION,
             V = phase & kSolveBits;
  if (F && RF) return CUDSS_STATUS_NOT_SUPPORTED;
  const bool groups[4] = {R, S, F || RF, V};
  int first = -1, last = -1;
  for (int g = 0; g < 4; ++g)
    if (groups[g]) last = g, first = first < 0 ? g : first;
  for (int g = first; g <= last; ++g)
    if (!groups[g]) return CUDSS_STATUS_NOT_SUPPORTED;
  if (!(A->format & CUDSS_MFORMAT_CSR)) return CUDSS_STATUS_NOT_SUPPORTED;
  if (A->mtype < CUDSS_MTYPE_GENERAL || A->mtype > CUDSS_MTYPE_HPD || A->mview < CUDSS_MVIEW_FULL ||
      A->mview > CUDSS_MVIEW_UPPER || A->base < CUDSS_BASE_ZERO || A->base > CUDSS_BASE_ONE)
    return CUDSS_STATUS_INVALID_VALUE;
  if (A->vtype == CUDSS_R_64F_64F)
    return refuse("cudssExecute", "double-double (CUDSS_R_64F_64F) values are not supported by this "
                                  "simulator's cuDSS");
  for (const cudssMatrix* m : {A, X, B})
    if (m && m->distributed && !(m->first_row == 0 && m->last_row == m->rows - 1))
      return refuse("cudssExecute", "a matrix distributed across processes (MGMN mode) is not supported "
                                    "by this simulator's cuDSS; one process holding every row is");
  if (d->interrupt && *d->interrupt) return CUDSS_STATUS_EXECUTION_FAILED;
  // The order the phases must come in, across calls.
  Stage at = d->stage;
  if (!R && S && at != kReordered) return CUDSS_STATUS_INVALID_VALUE;
  if (R) at = kReordered;
  if (S) at = kAnalysed;
  const bool factored = at == kFactored || (*pending_ok && d->pending_factor);
  if (F && at < kAnalysed) return CUDSS_STATUS_INVALID_VALUE;
  if (RF && !factored) return CUDSS_STATUS_INVALID_VALUE;
  if (!F && !RF && V && !factored) return CUDSS_STATUS_INVALID_VALUE;
  if ((F || RF || V) && (A->vtype != d->vtype || A->itype != d->itype) && !(R || S))
    return CUDSS_STATUS_INVALID_VALUE;
  if (V && (!X || !B)) return CUDSS_STATUS_INVALID_VALUE;
  if (V && (X->vtype != A->vtype || B->vtype != A->vtype))
    return refuse("cudssExecute", "the right-hand side and solution must have the matrix's value type");
  if (V && B->rows != (A->batch ? B->rows : A->rows)) return CUDSS_STATUS_INVALID_VALUE;
  if (V && !B->batch && B->ld < B->rows) return CUDSS_STATUS_INVALID_VALUE;
  return CUDSS_STATUS_SUCCESS;
}

template <class T>
Status run_phases(const cudssConfig& cfg, cudssData* d, const cudssMatrix& A, const cudssMatrix* X,
                  const cudssMatrix* B, int phase) {
  if (phase & (CUDSS_PHASE_REORDERING | CUDSS_PHASE_SYMBOLIC_FACTORIZATION)) {
    if (phase & CUDSS_PHASE_REORDERING) {
      d->stage = kNone;
      if (Status s = analyse(cfg, d, A)) return s;
      d->stage = kReordered;
    }
    if (phase & CUDSS_PHASE_SYMBOLIC_FACTORIZATION) d->stage = kAnalysed;
  }
  if (phase & (CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_REFACTORIZATION)) {
    if (Status s = factor<T>(cfg, d, A)) return s;
    d->stage = kFactored;
  }
  if (phase & kSolveBits) {
    if (d->stage != kFactored) return CUDSS_STATUS_INVALID_VALUE;
    return solve<T>(cfg, d, A, X, B, phase);
  }
  return CUDSS_STATUS_SUCCESS;
}

Status run(const cudssConfig& cfg, cudssData* d, const cudssMatrix& A, const cudssMatrix* X,
           const cudssMatrix* B, int phase) {
  return complex_type(A.vtype) ? run_phases<cd>(cfg, d, A, X, B, phase) : run_phases<double>(cfg, d, A, X, B, phase);
}

// ---- the Get/Set protocol ----
//
// Measured on NVIDIA's library: a size of 0 asks for the size (written to
// sizeWritten); otherwise the buffer must be exactly that size.
Status put(void* value, size_t size, size_t* written, const void* src, size_t need) {
  if (size == 0 || need == 0) {  // nothing set (a user permutation, say) reads as nothing
    if (written) *written = need;
    return CUDSS_STATUS_SUCCESS;
  }
  if (!value || size != need) return CUDSS_STATUS_INVALID_VALUE;
  if (!write_bytes(value, src, need)) return CUDSS_STATUS_EXECUTION_FAILED;
  if (written) *written = need;
  return CUDSS_STATUS_SUCCESS;
}

template <class V>
Status put_scalar(void* value, size_t size, size_t* written, V v) {
  return put(value, size, written, &v, sizeof v);
}

Status put_index(void* value, size_t size, size_t* written, const int64_t* v, size_t count,
                 cudssDataType_t t, int64_t add = 0) {
  const auto bytes = encode_indices(v, count, t, add);
  return put(value, size, written, bytes.data(), bytes.size());
}

}  // namespace

extern "C" {

// ---- library handle ----

cudssStatus_t cudssCreate(cudssHandle_t* handle) {
  if (!handle) return CUDSS_STATUS_INVALID_VALUE;
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    cudaGetLastError();
    return CUDSS_STATUS_NOT_INITIALIZED;
  }
  *handle = new cudssContext{};
  (*handle)->device = dev;
  return CUDSS_STATUS_SUCCESS;
}

// Several devices: the handle remembers them and their streams, and computes
// on the calling device's host, which is where all of this simulator's
// arithmetic happens anyway.
cudssStatus_t cudssCreateMg(cudssHandle_t* handle, int device_count, const int* device_indices) {
  if (!handle || device_count < 1 || device_count > 16) return CUDSS_STATUS_INVALID_VALUE;
  if (cudssStatus_t s = cudssCreate(handle)) return s;
  (*handle)->mg_count = device_count;
  for (int i = 0; i < device_count; ++i) (*handle)->mg_devices.push_back(device_indices ? device_indices[i] : i);
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssDestroy(cudssHandle_t handle) {
  if (!handle) return CUDSS_STATUS_INVALID_VALUE;
  delete handle;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssGetProperty(libraryPropertyType type, int* value) {
  if (!value) return CUDSS_STATUS_INVALID_VALUE;
  switch (type) {
    case MAJOR_VERSION: *value = CUDSS_VERSION_MAJOR; return CUDSS_STATUS_SUCCESS;
    case MINOR_VERSION: *value = CUDSS_VERSION_MINOR; return CUDSS_STATUS_SUCCESS;
    case PATCH_LEVEL: *value = CUDSS_VERSION_PATCH; return CUDSS_STATUS_SUCCESS;
    default: return CUDSS_STATUS_NOT_SUPPORTED;
  }
}

cudssStatus_t cudssSetStream(cudssHandle_t handle, cudaStream_t stream) {
  if (!handle) return CUDSS_STATUS_INVALID_VALUE;
  handle->stream = stream;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssSetMgStreams(cudssHandle_t handle, const cudaStream_t* streams, int stream_count) {
  if (!handle || !streams || stream_count != std::max(handle->mg_count, 1)) return CUDSS_STATUS_INVALID_VALUE;
  handle->mg_streams.assign(streams, streams + stream_count);
  return CUDSS_STATUS_SUCCESS;
}

// The communication and threading layers are libraries cuDSS loads for
// multi-process and multi-threaded runs. This one needs neither -- it is one
// process computing on the host -- so it records the name (or the
// environment's, as cuDSS reads it when the name is NULL) and loads nothing.
cudssStatus_t cudssSetCommLayer(cudssHandle_t handle, const char* name) {
  if (!handle) return CUDSS_STATUS_INVALID_VALUE;
  if (!name) name = std::getenv("CUDSS_COMM_LIB");
  if (!name || !*name) return CUDSS_STATUS_INVALID_VALUE;
  handle->comm_lib = name;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssSetThreadingLayer(cudssHandle_t handle, const char* name) {
  if (!handle) return CUDSS_STATUS_INVALID_VALUE;
  if (!name) name = std::getenv("CUDSS_THREADING_LIB");
  if (!name || !*name) return CUDSS_STATUS_INVALID_VALUE;
  handle->threading_lib = name;
  return CUDSS_STATUS_SUCCESS;
}

// The handler is kept and handed back. The factors live in host memory here,
// so nothing is ever allocated through it.
cudssStatus_t cudssSetDeviceMemHandler(cudssHandle_t handle, const cudssDeviceMemHandler_t* handler) {
  if (!handle) return CUDSS_STATUS_INVALID_VALUE;
  handle->has_mem = handler != nullptr;
  if (handler) handle->mem = *handler;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssGetDeviceMemHandler(const cudssHandle_t handle, cudssDeviceMemHandler_t* handler) {
  if (!handle || !handler || !handle->has_mem) return CUDSS_STATUS_INVALID_VALUE;
  *handler = handle->mem;
  return CUDSS_STATUS_SUCCESS;
}

// ---- configuration ----

cudssStatus_t cudssConfigCreate(cudssConfig_t* config) {
  if (!config) return CUDSS_STATUS_INVALID_VALUE;
  *config = new cudssConfig{};
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssConfigDestroy(cudssConfig_t config) {
  if (!config) return CUDSS_STATUS_INVALID_VALUE;
  delete config;
  return CUDSS_STATUS_SUCCESS;
}

}  // extern "C"

namespace {

// Where a setting lives and how big it is. Unknown parameters are an
// internal error, as NVIDIA's library answers.
struct Field {
  void* p = nullptr;
  size_t size = 0;
};

Field field(cudssConfig* c, cudssConfigParam_t param) {
  switch (param) {
    case CUDSS_CONFIG_REORDERING_ALG: return {&c->reordering, 4};
    case CUDSS_CONFIG_FACTORIZATION_ALG: return {&c->factorization, 4};
    case CUDSS_CONFIG_SOLVE_ALG: return {&c->solve, 4};
    case CUDSS_CONFIG_MATCHING_ALG: return {&c->matching, 4};
    case CUDSS_CONFIG_SOLVE_MODE: return {&c->solve_mode, 4};
    case CUDSS_CONFIG_IR_N_STEPS: return {&c->ir_steps, 4};
    case CUDSS_CONFIG_IR_TOL: return {&c->ir_tol, 8};
    case CUDSS_CONFIG_PIVOT_TYPE: return {&c->pivot_type, 4};
    case CUDSS_CONFIG_PIVOT_THRESHOLD: return {&c->pivot_threshold, 8};
    case CUDSS_CONFIG_PIVOT_EPSILON: return {&c->pivot_epsilon, 8};
    case CUDSS_CONFIG_MAX_LU_NNZ: return {&c->max_lu_nnz, 8};
    case CUDSS_CONFIG_HYBRID_MEMORY_MODE: return {&c->hybrid_memory, 4};
    case CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT: return {&c->hybrid_limit, 8};
    case CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY: return {&c->register_memory, 4};
    case CUDSS_CONFIG_HOST_NTHREADS: return {&c->host_threads, 4};
    case CUDSS_CONFIG_HYBRID_EXECUTE_MODE: return {&c->hybrid_execute, 4};
    case CUDSS_CONFIG_PIVOT_EPSILON_ALG: return {&c->epsilon_alg, 4};
    case CUDSS_CONFIG_ND_NLEVELS: return {&c->nd_levels, 4};
    case CUDSS_CONFIG_UBATCH_SIZE: return {&c->ubatch_size, 4};
    case CUDSS_CONFIG_UBATCH_INDEX: return {&c->ubatch_index, 4};
    case CUDSS_CONFIG_USE_SUPERPANELS: return {&c->superpanels, 4};
    case CUDSS_CONFIG_DEVICE_COUNT: return {&c->device_count, 4};
    case CUDSS_CONFIG_SCHUR_MODE: return {&c->schur_mode, 4};
    case CUDSS_CONFIG_DETERMINISTIC_MODE: return {&c->deterministic, 4};
    case CUDSS_CONFIG_ND_UBFACTOR: return {&c->nd_ubfactor, 4};
    default: return {};
  }
}

// The range checks NVIDIA's library makes when a setting is set.
bool in_range(cudssConfigParam_t param, const void* value, Status* why) {
  int i = 0;
  double x = 0;
  std::memcpy(&i, value, 4);
  if (param == CUDSS_CONFIG_IR_TOL) std::memcpy(&x, value, 8);
  *why = CUDSS_STATUS_INVALID_VALUE;
  switch (param) {
    case CUDSS_CONFIG_IR_N_STEPS: return i >= 0;
    case CUDSS_CONFIG_IR_TOL: return x >= 0;
    case CUDSS_CONFIG_PIVOT_TYPE:
      if (i == CUDSS_PIVOT_BUNCH_KAUFMAN) *why = CUDSS_STATUS_NOT_SUPPORTED;
      return i >= CUDSS_PIVOT_AUTO && i < CUDSS_PIVOT_BUNCH_KAUFMAN;
    case CUDSS_CONFIG_ND_NLEVELS: return i >= 1;
    case CUDSS_CONFIG_HYBRID_MEMORY_MODE:
    case CUDSS_CONFIG_HYBRID_EXECUTE_MODE: return i == 0 || i == 1;
    case CUDSS_CONFIG_ND_UBFACTOR: return i >= 0 && i <= 100;
    case CUDSS_CONFIG_DEVICE_COUNT: return i >= 1 && i <= 16;
    default: return true;
  }
}

}  // namespace

extern "C" {

cudssStatus_t cudssConfigSet(cudssConfig_t config, cudssConfigParam_t param, const void* value, size_t size) {
  if (!config || !value) return CUDSS_STATUS_INVALID_VALUE;
  if (param == CUDSS_CONFIG_DEVICE_INDICES) {
    if (size == 0 || size % sizeof(int)) return CUDSS_STATUS_INVALID_VALUE;
    config->device_indices.resize(size / sizeof(int));
    std::memcpy(config->device_indices.data(), value, size);
    return CUDSS_STATUS_SUCCESS;
  }
  const Field f = field(config, param);
  if (!f.p) return CUDSS_STATUS_INTERNAL_ERROR;
  if (size != f.size) return CUDSS_STATUS_INVALID_VALUE;
  Status why;
  if (!in_range(param, value, &why)) return why;
  std::memcpy(f.p, value, size);
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssConfigGet(const cudssConfig_t config, cudssConfigParam_t param, void* value, size_t size,
                             size_t* written) {
  if (!config) return CUDSS_STATUS_INVALID_VALUE;
  if (param == CUDSS_CONFIG_DEVICE_INDICES) {
    // The device list, as many as the device count: there is no size query.
    std::vector<int> idx = config->device_indices;
    if (idx.empty())
      for (int i = 0; i < config->device_count; ++i) idx.push_back(i);
    const size_t need = idx.size() * sizeof(int);
    if (!value || size < need) return CUDSS_STATUS_INVALID_VALUE;
    std::memcpy(value, idx.data(), need);
    if (written) *written = need;
    return CUDSS_STATUS_SUCCESS;
  }
  const Field f = field(config, param);
  if (!f.p) return CUDSS_STATUS_INTERNAL_ERROR;
  if (size == 0) {
    if (written) *written = f.size;
    return CUDSS_STATUS_SUCCESS;
  }
  if (!value || size != f.size) return CUDSS_STATUS_INVALID_VALUE;
  std::memcpy(value, f.p, size);
  if (written) *written = size;
  return CUDSS_STATUS_SUCCESS;
}

// ---- solver data ----

cudssStatus_t cudssDataCreate(const cudssHandle_t handle, cudssData_t* data) {
  if (!handle || !data) return CUDSS_STATUS_INVALID_VALUE;
  *data = new cudssData{};
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssDataDestroy(cudssHandle_t handle, cudssData_t data) {
  if (!handle || !data) return CUDSS_STATUS_INVALID_VALUE;
  delete data;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssDataSet(const cudssHandle_t handle, cudssData_t d, cudssDataParam_t param, const void* value,
                           size_t size) {
  if (!handle || !d) return CUDSS_STATUS_INVALID_VALUE;
  auto bytes = [&](std::vector<uint8_t>& to) {
    if (!value && size) return CUDSS_STATUS_INVALID_VALUE;
    to.assign(size, 0);
    return read_bytes(to.data(), value, size) ? CUDSS_STATUS_SUCCESS : CUDSS_STATUS_EXECUTION_FAILED;
  };
  auto pointer = [&](void* to, size_t want) {
    if (!value || size != want) return CUDSS_STATUS_INVALID_VALUE;
    std::memcpy(to, value, want);
    return CUDSS_STATUS_SUCCESS;
  };
  const bool input = param == CUDSS_DATA_USER_PERM || param == CUDSS_DATA_USER_SCHUR_INDICES ||
                     param == CUDSS_DATA_USER_ND_PARTITION_TREE || param == CUDSS_DATA_USER_HOST_INTERRUPT ||
                     param == CUDSS_DATA_COMM_DEVICE || param == CUDSS_DATA_COMM_HOST ||
                     param == CUDSS_DATA_UBATCH_MASK || param == CUDSS_DATA_SCHUR_MATRIX;
  if (!input && d->stage == kNone) return CUDSS_STATUS_NOT_INITIALIZED;
  switch (param) {
    case CUDSS_DATA_INFO:
      // Clearing a reported error (a non-positive minor) before trying again.
      return pointer(&d->info, sizeof(int));
    case CUDSS_DATA_USER_PERM: return bytes(d->user_perm);
    case CUDSS_DATA_USER_SCHUR_INDICES: return bytes(d->user_schur);
    case CUDSS_DATA_USER_ND_PARTITION_TREE: return bytes(d->user_tree);  // kept, not used
    case CUDSS_DATA_USER_HOST_INTERRUPT:
      // `value` is the flag itself, as NVIDIA's library reads it (the
      // documentation's "pointer to a 4-byte integer"); NULL turns it off.
      d->interrupt = static_cast<int*>(const_cast<void*>(value));
      return CUDSS_STATUS_SUCCESS;
    case CUDSS_DATA_COMM_DEVICE: return pointer(&d->comm_device, sizeof(void*));
    case CUDSS_DATA_COMM_HOST: return pointer(&d->comm_host, sizeof(void*));
    case CUDSS_DATA_UBATCH_MASK: return pointer(&d->ubatch_mask, sizeof(int64_t));
    case CUDSS_DATA_SCHUR_MATRIX: return pointer(&d->schur_matrix, sizeof(cudssMatrix_t));
    case CUDSS_DATA_LU_NNZ: case CUDSS_DATA_NPIVOTS: case CUDSS_DATA_INERTIA: case CUDSS_DATA_PERM_REORDER_ROW:
    case CUDSS_DATA_PERM_REORDER_COL: case CUDSS_DATA_PERM_ROW: case CUDSS_DATA_PERM_COL: case CUDSS_DATA_DIAG:
    case CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN: case CUDSS_DATA_MEMORY_ESTIMATES: case CUDSS_DATA_PERM_MATCHING:
    case CUDSS_DATA_SCALE_ROW: case CUDSS_DATA_SCALE_COL: case CUDSS_DATA_NSUPERPANELS:
    case CUDSS_DATA_SCHUR_SHAPE: case CUDSS_DATA_ND_PARTITION_TREE: case CUDSS_DATA_IR_N_STEPS:
    case CUDSS_DATA_FLOPS:
      return CUDSS_STATUS_NOT_SUPPORTED;  // outputs
    default: return CUDSS_STATUS_INVALID_VALUE;
  }
}

}  // extern "C"

namespace {

template <class T>
void collect(cudssData* d, int64_t which, int64_t* pert, int64_t* pos, int64_t* neg, int64_t* nnz,
             std::vector<T>* diag) {
  for (size_t b = 0; b < d->sys.size(); ++b) {
    System& s = d->sys[b];
    if (!s.factored) continue;
    const auto& f = factor_of(s, (T*)nullptr);
    *pert += f.perturbed;
    *pos += f.positive;
    *neg += f.negative;
    if (!d->uniform || (int64_t)b == which) *nnz += f.nnz;
    if (!d->uniform || (int64_t)b == which) diag->insert(diag->end(), f.diag.begin(), f.diag.end());
  }
}

template <class T>
Status get_factored(cudssData* d, cudssDataParam_t param, void* value, size_t size, size_t* written,
                    int64_t which) {
  int64_t pert = 0, pos = 0, neg = 0, nnz = 0;
  std::vector<T> diag;
  collect(d, which, &pert, &pos, &neg, &nnz, &diag);
  if (param == CUDSS_DATA_NPIVOTS) return put_index(value, size, written, &pert, 1, d->itype);
  if (param == CUDSS_DATA_INERTIA) {
    const int64_t in[2] = {pos, neg};
    return put_index(value, size, written, in, 2, d->itype);
  }
  if (param == CUDSS_DATA_LU_NNZ) return put_scalar(value, size, written, nnz);
  const auto bytes = encode(diag.data(), diag.size(), d->vtype);
  return put(value, size, written, bytes.data(), bytes.size());
}

}  // namespace

extern "C" {

cudssStatus_t cudssDataGet(const cudssHandle_t handle, const cudssData_t d, cudssDataParam_t param, void* value,
                           size_t size, size_t* written) {
  if (!handle || !d) return CUDSS_STATUS_INVALID_VALUE;
  // What the caller set can be read back at any time.
  switch (param) {
    case CUDSS_DATA_USER_PERM: return put(value, size, written, d->user_perm.data(), d->user_perm.size());
    case CUDSS_DATA_USER_SCHUR_INDICES: return put(value, size, written, d->user_schur.data(), d->user_schur.size());
    case CUDSS_DATA_USER_ND_PARTITION_TREE: return put(value, size, written, d->user_tree.data(), d->user_tree.size());
    case CUDSS_DATA_USER_HOST_INTERRUPT: return put_scalar(value, size, written, d->interrupt);
    case CUDSS_DATA_COMM_DEVICE: return put_scalar(value, size, written, d->comm_device);
    case CUDSS_DATA_COMM_HOST: return put_scalar(value, size, written, d->comm_host);
    case CUDSS_DATA_SCHUR_MATRIX: return CUDSS_STATUS_INVALID_VALUE;
    default: break;
  }
  // Results exist once the matrix has been reordered.
  if (d->stage == kNone) return CUDSS_STATUS_NOT_INITIALIZED;
  const size_t isz = type_size(d->itype), n = (size_t)d->n;
  const bool batch = !d->uniform && d->sys.size() > 1;
  const bool factored = d->stage == kFactored;
  int64_t which = 0;  // a uniform batch reports one system's factors: the first factored
  for (size_t b = 0; b < d->sys.size(); ++b)
    if (d->sys[b].factored) {
      which = (int64_t)b;
      break;
    }
  const dss::Symbolic& sym = d->sym.front();
  switch (param) {
    case CUDSS_DATA_INFO: return put_scalar(value, size, written, d->info);
    case CUDSS_DATA_IR_N_STEPS: return put_scalar(value, size, written, d->ir_steps);
    case CUDSS_DATA_UBATCH_MASK: return put_scalar(value, size, written, d->ubatch_mask);
    case CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN: return put_scalar(value, size, written, int64_t{0});
    case CUDSS_DATA_NSUPERPANELS: {
      const int64_t zero = 0;
      return put_index(value, size, written, &zero, 1, d->itype);
    }
    case CUDSS_DATA_SCHUR_SHAPE: {
      const int64_t shape[3] = {0, 0, 0};
      return put(value, size, written, shape, sizeof shape);
    }
    case CUDSS_DATA_FLOPS: {
      int64_t flops = 0;
      for (const auto& s : d->sym) flops += s.flops;
      return put_scalar(value, size, written, flops);
    }
    case CUDSS_DATA_MEMORY_ESTIMATES: {
      // Device memory: none, the factors are kept on the host. Host memory:
      // the predicted factors, and at the peak a dense front as well.
      int64_t est[16] = {0};
      int64_t factors = 0, front = 0;
      for (const auto& s : d->sym) {
        factors += s.nnz;
        for (size_t k = 0; k < s.sn_rows.size(); ++k) {
          const int64_t m = s.sn_first[k + 1] - s.sn_first[k] + (int64_t)s.sn_rows[k].size();
          front = std::max(front, m * m);
        }
      }
      const int64_t vb = (int64_t)type_size(d->vtype);
      est[2] = factors * vb;
      est[3] = (factors + front) * vb;
      return put(value, size, written, est, sizeof est);
    }
    case CUDSS_DATA_LU_NNZ:
      if (!factored) {
        int64_t nnz = 0;
        for (const auto& s : d->sym) nnz += s.nnz;
        return put_scalar(value, size, written, nnz);
      }
      break;
    case CUDSS_DATA_PERM_REORDER_ROW:
    case CUDSS_DATA_PERM_REORDER_COL:
      // Indices in the matrix's base, as cuDSS writes every index.
      if (batch) return CUDSS_STATUS_NOT_SUPPORTED;
      return put_index(value, size, written, sym.perm.data(), n, d->itype, d->base == CUDSS_BASE_ONE ? 1 : 0);
    case CUDSS_DATA_PERM_ROW:
    case CUDSS_DATA_PERM_COL: {
      // The final permutations, reordering and pivoting together: the
      // original row (column) of each pivot in elimination order.
      if (batch) return CUDSS_STATUS_NOT_SUPPORTED;
      if (size == 0) return put_index(value, size, written, sym.perm.data(), n, d->itype);
      if (!factored) return CUDSS_STATUS_INVALID_VALUE;
      const System& s = d->sys[which];
      const auto& piv = complex_type(d->vtype) ? (param == CUDSS_DATA_PERM_ROW ? s.fc.pivrow : s.fc.pivcol)
                                               : (param == CUDSS_DATA_PERM_ROW ? s.fr.pivrow : s.fr.pivcol);
      std::vector<int64_t> p(piv.size());
      for (size_t k = 0; k < piv.size(); ++k) p[k] = sym.perm[piv[k]];
      return put_index(value, size, written, p.data(), p.size(), d->itype, d->base == CUDSS_BASE_ONE ? 1 : 0);
    }
    case CUDSS_DATA_PERM_MATCHING:
    case CUDSS_DATA_SCALE_ROW:
    case CUDSS_DATA_SCALE_COL: {
      // Matching is accepted but not performed: the identity permutation, and
      // no scaling.
      const size_t elem = param == CUDSS_DATA_PERM_MATCHING ? isz : single_type(d->vtype) ? 4 : 8;
      if (size == 0) return put(value, size, written, nullptr, n * elem);
      if (batch || !d->matching) return CUDSS_STATUS_NOT_SUPPORTED;
      if (param == CUDSS_DATA_PERM_MATCHING) {
        std::vector<int64_t> id(n);
        for (size_t k = 0; k < n; ++k) id[k] = (int64_t)k;
        return put_index(value, size, written, id.data(), n, d->itype, d->base == CUDSS_BASE_ONE ? 1 : 0);
      }
      std::vector<double> ones(n, 1.0);
      const auto bytes = encode(ones.data(), n, single_type(d->vtype) ? CUDSS_R_32F : CUDSS_R_64F);
      return put(value, size, written, bytes.data(), bytes.size());
    }
    case CUDSS_DATA_ND_PARTITION_TREE:
      return refuse("cudssDataGet", "this simulator's cuDSS orders by minimum degree and has no nested "
                                    "dissection partition tree (CUDSS_DATA_ND_PARTITION_TREE)");
    case CUDSS_DATA_NPIVOTS:
    case CUDSS_DATA_INERTIA:
    case CUDSS_DATA_DIAG:
      break;
    default: return CUDSS_STATUS_INVALID_VALUE;
  }
  // The rest describe the numeric factorization. Their size can be asked for
  // once the matrix is analysed; their values exist once it is factored.
  if (size == 0) {
    const size_t need = param == CUDSS_DATA_NPIVOTS ? isz
                        : param == CUDSS_DATA_INERTIA ? 2 * isz
                        : param == CUDSS_DATA_LU_NNZ ? 8
                        : n * type_size(d->vtype);
    if (written) *written = need;
    return CUDSS_STATUS_SUCCESS;
  }
  if (!factored) return CUDSS_STATUS_INVALID_VALUE;
  return complex_type(d->vtype) ? get_factored<cd>(d, param, value, size, written, which)
                                : get_factored<double>(d, param, value, size, written, which);
}

// ---- execute ----

cudssStatus_t cudssExecute(cudssHandle_t handle, int phase, const cudssConfig_t config, cudssData_t data,
                           const cudssMatrix_t matrix, cudssMatrix_t solution, const cudssMatrix_t rhs) {
  bool pending_ok = true;
  if (Status s = check(handle, phase, config, data, matrix, solution, rhs, &pending_ok)) return s;
  if (capturing(handle)) {
    // Factorization and solve are recorded and run when the graph is
    // launched, over whatever its kernels have produced by then. The
    // analysis is synchronous on hardware too and cannot be captured.
    if (phase & CUDSS_PHASE_ANALYSIS)
      return refuse("cudssExecute", "the analysis phase cannot be captured into a CUDA graph; run it "
                                    "before capture begins");
    auto cfg = std::make_shared<cudssConfig>(*config);
    auto A = std::make_shared<cudssMatrix>(*matrix);
    auto X = solution ? std::make_shared<cudssMatrix>(*solution) : nullptr;
    auto B = rhs ? std::make_shared<cudssMatrix>(*rhs) : nullptr;
    cudssData* d = data;
    std::function<void()> op = [cfg, A, X, B, d, phase] {
      Status s = run(*cfg, d, *A, X.get(), B.get(), phase);
      d->pending_factor = false;
      if (s && s != CUDSS_STATUS_IR_FAILED) say("cudssExecute (graph)", "a recorded phase failed");
    };
    if (phase & (CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_REFACTORIZATION)) data->pending_factor = true;
    return vgpu_record_host_op_if_capturing(handle->stream, std::move(op)) ? CUDSS_STATUS_SUCCESS
                                                                           : CUDSS_STATUS_EXECUTION_FAILED;
  }
  if (Status s = settle(handle)) return s;
  return run(*config, data, *matrix, solution, rhs, phase);
}

// ---- matrices ----

cudssStatus_t cudssMatrixCreateDn(cudssMatrix_t* matrix, int64_t nrows, int64_t ncols, int64_t ld,
                                  const void* values, cudssDataType_t valueType, cudssLayout_t layout) {
  if (!matrix || nrows < 0 || ncols < 0 || !value_type(valueType)) return CUDSS_STATUS_INVALID_VALUE;
  if (layout == CUDSS_LAYOUT_ROW_MAJOR) return CUDSS_STATUS_NOT_SUPPORTED;  // as in cuDSS 0.8
  auto* m = new cudssMatrix{};
  m->format = CUDSS_MFORMAT_DENSE;
  m->rows = nrows;
  m->cols = ncols;
  m->ld = ld;
  m->values = values;
  m->vtype = valueType;
  m->layout = layout;
  *matrix = m;
  return CUDSS_STATUS_SUCCESS;
}

}  // extern "C"

namespace {

// cuDSS takes 64-bit offsets with 32-bit indices, but not the other way round.
bool csr_types(cudssDataType_t offsetType, cudssDataType_t indexType, cudssDataType_t valueType) {
  return index_type(offsetType) && index_type(indexType) && value_type(valueType) &&
         !(offsetType == CUDSS_R_32I && indexType == CUDSS_R_64I);
}

}  // namespace

extern "C" {

cudssStatus_t cudssMatrixCreateCsr(cudssMatrix_t* matrix, int64_t nrows, int64_t ncols, int64_t nnz,
                                   const void* rowStart, const void* rowEnd, const void* colIndices,
                                   const void* values, cudssDataType_t offsetType, cudssDataType_t indexType,
                                   cudssDataType_t valueType, cudssMatrixType_t mtype,
                                   cudssMatrixViewType_t mview, cudssIndexBase_t indexBase) {
  if (!matrix || nrows < 0 || ncols < 0 || !csr_types(offsetType, indexType, valueType))
    return CUDSS_STATUS_INVALID_VALUE;
  auto* m = new cudssMatrix{};
  m->format = CUDSS_MFORMAT_CSR;
  m->rows = nrows;
  m->cols = ncols;
  m->nnz = nnz;
  m->row_start = rowStart;
  m->row_end = rowEnd;
  m->col_ind = colIndices;
  m->values = values;
  m->otype = offsetType;
  m->itype = indexType;
  m->vtype = valueType;
  m->mtype = mtype;
  m->mview = mview;
  m->base = indexBase;
  *matrix = m;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssMatrixCreateBatchDn(cudssMatrix_t* matrix, int64_t batchCount, const void* nrows,
                                       const void* ncols, const void* ld, const void* const* values,
                                       cudssDataType_t integerType, cudssDataType_t valueType,
                                       cudssLayout_t layout) {
  if (!matrix || batchCount < 1 || !nrows || !ncols || !ld || !values || !index_type(integerType) ||
      !value_type(valueType))
    return CUDSS_STATUS_INVALID_VALUE;
  if (layout == CUDSS_LAYOUT_ROW_MAJOR) return CUDSS_STATUS_NOT_SUPPORTED;
  auto* m = new cudssMatrix{};
  m->format = CUDSS_MFORMAT_DENSE | CUDSS_MFORMAT_BATCH;
  m->batch = batchCount;
  m->b_rows = nrows;
  m->b_cols = ncols;
  m->b_third = ld;
  m->b_int = integerType;
  m->b_values = values;
  m->vtype = valueType;
  m->layout = layout;
  *matrix = m;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssMatrixCreateBatchCsr(cudssMatrix_t* matrix, int64_t batchCount, const void* nrows,
                                        const void* ncols, const void* nnz, const void* const* rowStart,
                                        const void* const* rowEnd, const void* const* colIndices,
                                        const void* const* values, cudssDataType_t offsetType,
                                        cudssDataType_t indexType, cudssDataType_t valueType,
                                        cudssMatrixType_t mtype, cudssMatrixViewType_t mview,
                                        cudssIndexBase_t indexBase) {
  if (!matrix || batchCount < 1 || !nrows || !ncols || !nnz || !rowStart || !colIndices ||
      !csr_types(offsetType, indexType, valueType))
    return CUDSS_STATUS_INVALID_VALUE;
  auto* m = new cudssMatrix{};
  m->format = CUDSS_MFORMAT_CSR | CUDSS_MFORMAT_BATCH;
  m->batch = batchCount;
  m->b_rows = nrows;
  m->b_cols = ncols;
  m->b_third = nnz;
  m->b_int = indexType;
  m->b_row_start = rowStart;
  m->b_row_end = rowEnd;
  m->b_col_ind = colIndices;
  m->b_values = values;
  m->otype = offsetType;
  m->itype = indexType;
  m->vtype = valueType;
  m->mtype = mtype;
  m->mview = mview;
  m->base = indexBase;
  *matrix = m;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssMatrixDestroy(cudssMatrix_t matrix) {
  if (!matrix) return CUDSS_STATUS_INVALID_VALUE;
  delete matrix;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssMatrixGetDn(const cudssMatrix_t m, int64_t* nrows, int64_t* ncols, int64_t* ld, void** values,
                               cudssDataType_t* type, cudssLayout_t* layout) {
  if (!m || m->format != CUDSS_MFORMAT_DENSE) return CUDSS_STATUS_INVALID_VALUE;
  if (nrows) *nrows = m->rows;
  if (ncols) *ncols = m->cols;
  if (ld) *ld = m->ld;
  if (values) *values = const_cast<void*>(m->values);
  if (type) *type = m->vtype;
  if (layout) *layout = m->layout;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssMatrixGetCsr(const cudssMatrix_t m, int64_t* nrows, int64_t* ncols, int64_t* nnz,
                                void** rowStart, void** rowEnd, void** colIndices, void** values,
                                cudssDataType_t* offsetType, cudssDataType_t* indexType,
                                cudssDataType_t* valueType, cudssMatrixType_t* mtype,
                                cudssMatrixViewType_t* mview, cudssIndexBase_t* indexBase) {
  if (!m || (m->format & ~CUDSS_MFORMAT_DISTRIBUTED) != CUDSS_MFORMAT_CSR) return CUDSS_STATUS_INVALID_VALUE;
  if (nrows) *nrows = m->rows;
  if (ncols) *ncols = m->cols;
  if (nnz) *nnz = m->nnz;
  if (rowStart) *rowStart = const_cast<void*>(m->row_start);
  if (rowEnd) *rowEnd = const_cast<void*>(m->row_end);
  if (colIndices) *colIndices = const_cast<void*>(m->col_ind);
  if (values) *values = const_cast<void*>(m->values);
  if (offsetType) *offsetType = m->otype;
  if (indexType) *indexType = m->itype;
  if (valueType) *valueType = m->vtype;
  if (mtype) *mtype = m->mtype;
  if (mview) *mview = m->mview;
  if (indexBase) *indexBase = m->base;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssMatrixSetValues(cudssMatrix_t m, const void* values) {
  if (!m || !values || m->batch) return CUDSS_STATUS_INVALID_VALUE;
  m->values = values;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssMatrixSetCsrPointers(cudssMatrix_t m, const void* rowOffsets, const void* rowEnd,
                                        const void* colIndices, const void* values) {
  if (!m || !(m->format & CUDSS_MFORMAT_CSR) || m->batch || !rowOffsets || !colIndices || !values)
    return CUDSS_STATUS_INVALID_VALUE;
  m->row_start = rowOffsets;
  m->row_end = rowEnd;
  m->col_ind = colIndices;
  m->values = values;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssMatrixGetBatchDn(const cudssMatrix_t m, int64_t* batchCount, void** nrows, void** ncols,
                                    void** ld, void*** values, cudssDataType_t* indexType,
                                    cudssDataType_t* valueType, cudssLayout_t* layout) {
  if (!m || m->format != (CUDSS_MFORMAT_DENSE | CUDSS_MFORMAT_BATCH)) return CUDSS_STATUS_INVALID_VALUE;
  if (batchCount) *batchCount = m->batch;
  if (nrows) *nrows = const_cast<void*>(m->b_rows);
  if (ncols) *ncols = const_cast<void*>(m->b_cols);
  if (ld) *ld = const_cast<void*>(m->b_third);
  if (values) *values = const_cast<void**>(m->b_values);
  if (indexType) *indexType = m->b_int;
  if (valueType) *valueType = m->vtype;
  if (layout) *layout = m->layout;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssMatrixGetBatchCsr(const cudssMatrix_t m, int64_t* batchCount, void** nrows, void** ncols,
                                     void** nnz, void*** rowStart, void*** rowEnd, void*** colIndices,
                                     void*** values, cudssDataType_t* offsetType, cudssDataType_t* indexType,
                                     cudssDataType_t* valueType, cudssMatrixType_t* mtype,
                                     cudssMatrixViewType_t* mview, cudssIndexBase_t* indexBase) {
  if (!m || m->format != (CUDSS_MFORMAT_CSR | CUDSS_MFORMAT_BATCH)) return CUDSS_STATUS_INVALID_VALUE;
  if (batchCount) *batchCount = m->batch;
  if (nrows) *nrows = const_cast<void*>(m->b_rows);
  if (ncols) *ncols = const_cast<void*>(m->b_cols);
  if (nnz) *nnz = const_cast<void*>(m->b_third);
  if (rowStart) *rowStart = const_cast<void**>(m->b_row_start);
  if (rowEnd) *rowEnd = const_cast<void**>(m->b_row_end);
  if (colIndices) *colIndices = const_cast<void**>(m->b_col_ind);
  if (values) *values = const_cast<void**>(m->b_values);
  if (offsetType) *offsetType = m->otype;
  if (indexType) *indexType = m->itype;
  if (valueType) *valueType = m->vtype;
  if (mtype) *mtype = m->mtype;
  if (mview) *mview = m->mview;
  if (indexBase) *indexBase = m->base;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssMatrixSetBatchValues(cudssMatrix_t m, const void* const* values) {
  if (!m || !m->batch || !values) return CUDSS_STATUS_INVALID_VALUE;
  m->b_values = values;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssMatrixSetBatchCsrPointers(cudssMatrix_t m, const void* const* rowOffsets,
                                             const void* const* rowEnd, const void* const* colIndices,
                                             const void* const* values) {
  if (!m || !m->batch || !(m->format & CUDSS_MFORMAT_CSR) || !rowOffsets || !colIndices || !values)
    return CUDSS_STATUS_INVALID_VALUE;
  m->b_row_start = rowOffsets;
  m->b_row_end = rowEnd;
  m->b_col_ind = colIndices;
  m->b_values = values;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssMatrixGetFormat(const cudssMatrix_t m, int* format) {
  if (!m || !format) return CUDSS_STATUS_INVALID_VALUE;
  *format = m->format;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssMatrixSetDistributionRow1d(cudssMatrix_t m, int64_t first_row, int64_t last_row) {
  if (!m) return CUDSS_STATUS_INVALID_VALUE;
  m->distributed = true;
  m->first_row = first_row;
  m->last_row = last_row;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssMatrixGetDistributionRow1d(const cudssMatrix_t m, int64_t* first_row, int64_t* last_row) {
  if (!m || !first_row || !last_row) return CUDSS_STATUS_INVALID_VALUE;
  *first_row = m->distributed ? m->first_row : 0;
  *last_row = m->distributed ? m->last_row : m->rows - 1;
  return CUDSS_STATUS_SUCCESS;
}

// ---- logging ----

cudssStatus_t cudssLoggerSetCallback(cudssLoggerCallback_t callback) {
  logger().callback = callback;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssLoggerSetFile(FILE* file) {
  Logger& l = logger();
  if (l.own_file && l.file) std::fclose(l.file);
  l.file = file;
  l.own_file = false;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssLoggerOpenFile(const char* path) {
  if (!path) return CUDSS_STATUS_INVALID_VALUE;
  FILE* f = std::fopen(path, "w");
  if (!f) return CUDSS_STATUS_INVALID_VALUE;
  cudssLoggerSetFile(f);
  logger().own_file = true;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssLoggerSetLevel(int level) {
  if (level < 0 || level > 5) return CUDSS_STATUS_INVALID_VALUE;
  logger().level = level;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssLoggerSetMask(int mask) {
  if (mask < 0 || mask > 31) return CUDSS_STATUS_INVALID_VALUE;
  logger().mask = mask;
  return CUDSS_STATUS_SUCCESS;
}

cudssStatus_t cudssLoggerForceDisable(void) {
  Logger& l = logger();
  l.disabled = true;
  l.callback = nullptr;
  if (l.own_file && l.file) std::fclose(l.file);
  l.file = nullptr;
  l.own_file = false;
  return CUDSS_STATUS_SUCCESS;
}

}  // extern "C"
