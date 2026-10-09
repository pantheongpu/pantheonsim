// cuTensorNet's distributed execution: a job of MPI processes, each with its
// own GPU, giving the library a communicator (cutensornetDistributedResetConfiguration)
// and a library of communication primitives ($CUTENSORNET_COMM_LIB, the table
// cutensornetCommInterface of cutensor/typesDistributed.h; here
// cutn_comm_mpi.c, which logs each primitive the library asks for).
//
// The expectations hold for NVIDIA's libcutensornet 2.14 on two RTX 3060s too
// (run_cutensornet_mpi.sh --card): the sequence of primitives behind each call,
// which rank contracts which slice, what a failing primitive answers, and the
// statuses of a library that cannot be loaded were all measured there. What
// differs and is not compared: the contraction path, the number of slices (both
// libraries slice to at least one per rank), and the exchanges of the
// hyper-optimizer in the preparation calls (NVIDIA's ranks search with different
// seeds and agree on the best; the simulator's search is deterministic).
//
// mpirun -np N ./cutensornet_mpi_paths [main|nolib|badlib|nosym|version]
//   main     the communicator, the slices, the state API, failing primitives
//   nolib    $CUTENSORNET_COMM_LIB unset or naming nothing
//   nosym    ... naming a library without the table
//   version  ... naming one whose table has another version (CUTN_COMM_VERSION)
#include <mpi.h>

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../include/vgpu_cutensornet.h"

using cd = std::complex<double>;

static int rk = 0, np = 1, failures = 0;
static std::string report, failed;
static bool card = false;  // NVIDIA's library, not the simulator's

static void check(bool ok, const char* what) {
  char b[640];
  std::snprintf(b, sizeof b, "%-4s %s\n", ok ? "ok" : "FAIL", what);
  report += b;
  if (!ok) {
    ++failures;
    failed += "rank " + std::to_string(rk) + ": " + b;
  }
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

static cutensornetHandle_t h;
static MPI_Comm comm;

// ---- the communication library's log and hooks ----

static int (*log_size)(void);
static const char* (*log_line)(int);
static void (*log_clear)(void);
static void (*comm_fail)(const char*);

static bool bind_comm() {
  const char* path = std::getenv("CUTENSORNET_COMM_LIB");
  void* d = path ? dlopen(path, RTLD_NOW | RTLD_NOLOAD) : nullptr;
  if (!d) return false;
  log_size = (int (*)(void))dlsym(d, "cutn_comm_log_size");
  log_line = (const char* (*)(int))dlsym(d, "cutn_comm_log_line");
  log_clear = (void (*)(void))dlsym(d, "cutn_comm_log_clear");
  comm_fail = (void (*)(const char*))dlsym(d, "cutn_comm_fail");
  return log_size && log_line && log_clear && comm_fail;
}

// The primitives called since the last call, as "name" or "name:count:type"
// (a query of the world is "name"), separated by spaces.
static std::string calls() {
  std::string s;
  if (!log_size) return s;
  for (int i = 0; i < log_size(); ++i) {
    char name[64];
    int count = 0, dtype = 0, x = 0;
    std::sscanf(log_line(i), "%63s count=%d dtype=%d x=%d", name, &count, &dtype, &x);
    if (!s.empty()) s += ' ';
    s += name;
    if (count) s += ":" + std::to_string(count) + ":" + std::to_string(dtype);
  }
  log_clear();
  return s;
}

#define CALLS(want) check(calls() == (want), "the primitives called are " want)

// The same without the queries of the world that every distributed call starts with.
static std::string data_calls() {
  std::string in = calls(), out, word;
  for (size_t i = 0; i <= in.size(); ++i) {
    if (i == in.size() || in[i] == ' ') {
      if (word != "getNumRanks" && word != "getProcRank") out += (out.empty() ? "" : " ") + word;
      word.clear();
    } else {
      word += in[i];
    }
  }
  return out;
}

// ---- host and device tensors (column-major doubles) ----

static double rnd(unsigned& s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return ((double)(s % 20001) - 10000.0) / 10000.0;
}

static void* up(const std::vector<double>& v) {
  void* d = nullptr;
  cudaMalloc(&d, v.size() * 8 + 256);
  cudaMemcpy(d, v.data(), v.size() * 8, cudaMemcpyHostToDevice);
  return d;
}

static std::vector<double> down(const void* d, size_t n) {
  std::vector<double> v(n);
  cudaDeviceSynchronize();
  cudaMemcpy(v.data(), d, n * 8, cudaMemcpyDeviceToHost);
  return v;
}

static double max_err(const std::vector<double>& a, const std::vector<double>& b) {
  double e = 0, m = 1e-300;
  if (a.size() != b.size()) return 1e300;
  for (size_t i = 0; i < a.size(); ++i) e = std::max(e, std::fabs(a[i] - b[i])), m = std::max(m, std::fabs(b[i]));
  return e / m;
}

// ---- a network ijk,kl,jlm->im, tensor 0 scaled by (rank + 1) ----

struct Net {
  cutensornetNetworkDescriptor_t net = nullptr;
  cutensornetWorkspaceDescriptor_t work = nullptr;
  cutensornetContractionOptimizerConfig_t cfg = nullptr;
  cutensornetContractionOptimizerInfo_t info = nullptr;
  void* in[3] = {};
  void* out = nullptr;
  void* scratch = nullptr;
  int64_t ns = 0;
  int status = 0;  // of the optimizer
};

static Net build(int min_slices, int* opt_status = nullptr) {
  Net n;
  cutensornetCreateNetwork(h, &n.net);
  const int32_t modes[3][3] = {{'i', 'j', 'k'}, {'k', 'l', 0}, {'j', 'l', 'm'}};
  const int32_t nm[3] = {3, 2, 3};
  std::map<int, int64_t> ext = {{'i', 4}, {'j', 5}, {'k', 8}, {'l', 3}, {'m', 2}};
  for (int t = 0; t < 3; ++t) {
    int64_t e[3];
    size_t sz = 1;
    for (int k = 0; k < nm[t]; ++k) e[k] = ext[modes[t][k]], sz *= (size_t)e[k];
    cutensornetTensorQualifiers_t q{0, 0, 0};
    int64_t id = -1;
    cutensornetNetworkAppendTensor(h, n.net, nm[t], e, modes[t], &q, CUDA_R_64F, &id);
    std::vector<double> v(sz);
    unsigned s = 77u + (unsigned)t;
    for (auto& x : v) x = (t == 0 ? (double)(rk + 1) : 1.0) * rnd(s);
    n.in[t] = up(v);
    cutensornetNetworkSetInputTensorMemory(h, n.net, id, n.in[t], nullptr);
  }
  const int32_t om[2] = {'i', 'm'};
  cutensornetNetworkSetOutputTensor(h, n.net, 2, om, CUDA_R_64F);
  n.out = up(std::vector<double>(8, 0.0));
  cutensornetNetworkSetOutputTensorMemory(h, n.net, n.out, nullptr);
  cutensornetCreateWorkspaceDescriptor(h, &n.work);
  cutensornetCreateContractionOptimizerConfig(h, &n.cfg);
  cutensornetCreateContractionOptimizerInfo(h, n.net, &n.info);
  int32_t ms = min_slices;
  cutensornetContractionOptimizerConfigSetAttribute(h, n.cfg, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES, &ms, 4);
  const int r = cutensornetContractionOptimize(h, n.net, n.cfg, 1ull << 30, n.info);
  if (opt_status) *opt_status = r;
  n.ns = -1;
  cutensornetContractionOptimizerInfoGetAttribute(h, n.info, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES, &n.ns, 8);
  return n;
}

static void prepare(Net& n) {
  cutensornetWorkspaceComputeContractionSizes(h, n.net, n.info, n.work);
  int64_t rec = 0;
  cutensornetWorkspaceGetMemorySize(h, n.work, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE,
                                    CUTENSORNET_WORKSPACE_SCRATCH, &rec);
  cudaMalloc(&n.scratch, (size_t)rec + 256);
  cutensornetWorkspaceSetMemory(h, n.work, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, n.scratch, rec);
  cutensornetNetworkPrepareContraction(h, n.net, n.work);
}

static void destroy(Net& n) {
  for (void* p : n.in) cudaFree(p);
  cudaFree(n.out);
  cudaFree(n.scratch);
  cutensornetDestroyContractionOptimizerInfo(n.info);
  cutensornetDestroyContractionOptimizerConfig(n.cfg);
  cutensornetDestroyWorkspaceDescriptor(n.work);
  cutensornetDestroyNetwork(n.net);
}

static void set_comm() {
  const cutensornetStatus_t s = cutensornetDistributedResetConfiguration(h, &comm, sizeof comm);
  if (s) std::printf("reset -> %d\n", (int)s);
}

// ---- the communicator ----

static void test_communicator() {
  report += "-- the communicator\n";
  int32_t v = -5;
  IS(cutensornetDistributedGetNumRanks(h, &v), 0);
  check(v == 1, "without a communicator the world is one rank");
  v = -5;
  IS(cutensornetDistributedGetProcRank(h, &v), 0);
  check(v == 0, "and this is rank 0");
  IS(cutensornetDistributedSynchronize(h), 0);
  IS(cutensornetDistributedGetNumRanks(h, nullptr), CUTENSORNET_STATUS_INVALID_VALUE);
  IS(cutensornetDistributedGetNumRanks(nullptr, &v), CUTENSORNET_STATUS_INVALID_VALUE);
  IS(cutensornetDistributedGetProcRank(h, nullptr), CUTENSORNET_STATUS_INVALID_VALUE);
  IS(cutensornetDistributedGetProcRank(nullptr, &v), CUTENSORNET_STATUS_INVALID_VALUE);
  IS(cutensornetDistributedSynchronize(nullptr), CUTENSORNET_STATUS_INVALID_VALUE);
  IS(cutensornetDistributedResetConfiguration(nullptr, &comm, sizeof comm), CUTENSORNET_STATUS_INVALID_VALUE);
  IS(cutensornetDistributedResetConfiguration(h, &comm, 0), CUTENSORNET_STATUS_INVALID_VALUE);
  IS(cutensornetDistributedResetConfiguration(h, nullptr, 0), 0);  // opens the library; nothing to communicate yet
  check(bind_comm(), "the communication library is loaded and exports the test hooks");
  calls();
  IS(cutensornetDistributedResetConfiguration(h, nullptr, sizeof comm), 0);
  CALLS("");
  IS(cutensornetDistributedResetConfiguration(h, &comm, sizeof comm), 0);
  CALLS("getProcRank Barrier getNumRanksShared");
  v = -5;
  IS(cutensornetDistributedGetNumRanks(h, &v), 0);
  check(v == np, "the world has every rank");
  CALLS("getNumRanks");
  v = -5;
  IS(cutensornetDistributedGetProcRank(h, &v), 0);
  check(v == rk, "this process has its rank in the communicator");
  CALLS("getProcRank");
  IS(cutensornetDistributedSynchronize(h), 0);
  CALLS("Barrier");
  IS(cutensornetDistributedResetConfiguration(h, &comm, sizeof comm), 0);
  CALLS("Barrier getProcRank Barrier getNumRanksShared");
  IS(cutensornetDistributedResetConfiguration(h, nullptr, 0), 0);
  CALLS("Barrier");
  v = -5;
  IS(cutensornetDistributedGetNumRanks(h, &v), 0);
  check(v == 1, "a reset to no communicator leaves one rank");
  IS(cutensornetDistributedResetConfiguration(h, nullptr, 0), 0);
  CALLS("");
  set_comm();
  calls();
}

// ---- a failing primitive ----

static void test_failures() {
  report += "-- failing primitives\n";
  int32_t v = -5;
  comm_fail("getNumRanks");
  IS(cutensornetDistributedGetNumRanks(h, &v), CUTENSORNET_STATUS_INTERNAL_ERROR);
  comm_fail(nullptr);
  comm_fail("getProcRank");
  IS(cutensornetDistributedGetProcRank(h, &v), CUTENSORNET_STATUS_INTERNAL_ERROR);
  IS(cutensornetDistributedResetConfiguration(h, &comm, sizeof comm), CUTENSORNET_STATUS_INTERNAL_ERROR);
  comm_fail(nullptr);
  comm_fail("Barrier");
  IS(cutensornetDistributedSynchronize(h), CUTENSORNET_STATUS_DISTRIBUTED_FAILURE);
  IS(cutensornetDistributedResetConfiguration(h, &comm, sizeof comm), CUTENSORNET_STATUS_DISTRIBUTED_FAILURE);
  IS(cutensornetDistributedResetConfiguration(h, nullptr, 0), CUTENSORNET_STATUS_DISTRIBUTED_FAILURE);
  comm_fail(nullptr);
  comm_fail("getNumRanksShared");
  IS(cutensornetDistributedResetConfiguration(h, &comm, sizeof comm), 0);  // the size of the node's group is not needed
  comm_fail(nullptr);
  set_comm();
  // A contraction and the optimizer fail with the service; the user's output is then this rank's part.
  comm_fail("getNumRanks");
  int os = 0;
  Net bad = build(2, &os);
  IS(os, CUTENSORNET_STATUS_DISTRIBUTED_FAILURE);
  comm_fail(nullptr);
  destroy(bad);
  if (np < 2) return;
  for (const char* name : {"AllreduceInPlaceMin", "AllreduceDoubleIntMinloc", "Bcast"}) {
    comm_fail(name);
    Net n = build(2, &os);
    check(os == CUTENSORNET_STATUS_DISTRIBUTED_FAILURE, (std::string("the optimizer fails with ") + name).c_str());
    comm_fail(nullptr);
    destroy(n);
  }
  Net n = build(2, &os);
  IS(os, 0);
  prepare(n);
  comm_fail("getProcRank");
  IS(cutensornetNetworkContract(h, n.net, 0, n.work, nullptr, 0), CUTENSORNET_STATUS_DISTRIBUTED_FAILURE);
  comm_fail("AllreduceInPlace");
  IS(cutensornetNetworkContract(h, n.net, 0, n.work, nullptr, 0), CUTENSORNET_STATUS_DISTRIBUTED_FAILURE);
  comm_fail(nullptr);
  IS(cutensornetNetworkContract(h, n.net, 0, n.work, nullptr, 0), 0);
  destroy(n);
}

// ---- slices dealt to the ranks ----

// The rank that contracts each slice of a group: the order in which a hash set of
// its ids iterates (room for one more than the ids), dealt round robin. A range, and
// all the slices, in ascending order.
static std::vector<int> owners_of(const std::vector<int64_t>& ids, bool ordered, size_t ns, std::vector<int64_t>& order_out) {
  std::vector<int64_t> order = ids;
  if (!ordered) {
    std::unordered_set<int64_t> set;
    set.reserve(ids.size() + 1);
    for (int64_t id : ids) set.insert(id);
    order.assign(set.begin(), set.end());
  }
  std::vector<int> owner(ns, -1);
  for (size_t i = 0; i < order.size(); ++i) owner[(size_t)order[i]] = (int)(i % (size_t)np);
  order_out = order;
  return owner;
}

struct Slices {
  std::vector<std::vector<double>> base;  // each slice's contribution, for this rank's scaling taken out
};

// Each slice contracted alone with no communicator (the scaling of tensor 0 taken out).
static Slices slice_parts(Net& n) {
  Slices s;
  cutensornetDistributedResetConfiguration(h, nullptr, 0);
  for (int64_t sl = 0; sl < n.ns; ++sl) {
    cutensornetSliceGroup_t g = nullptr;
    cutensornetCreateSliceGroupFromIDRange(h, sl, sl + 1, 1, &g);
    cutensornetNetworkContract(h, n.net, 0, n.work, g, 0);
    auto p = down(n.out, 8);
    for (auto& x : p) x /= (double)(rk + 1);
    s.base.push_back(p);
    cutensornetDestroySliceGroup(g);
  }
  set_comm();
  calls();
  return s;
}

// What the ranks' outputs hold after contracting `owner`'s slices, rank r's data being
// scaled by r + 1 and its output starting from pre(r).
static std::vector<double> expected(const Slices& s, const std::vector<int>& owner, bool accumulate) {
  std::vector<double> r(8, 0.0);
  for (size_t sl = 0; sl < owner.size(); ++sl) {
    if (owner[sl] < 0) continue;
    for (int i = 0; i < 8; ++i) r[(size_t)i] += (owner[sl] + 1) * s.base[sl][(size_t)i];
  }
  if (accumulate)
    for (int q = 0; q < np; ++q)
      for (int i = 0; i < 8; ++i) r[(size_t)i] += 100.0 * (q + 1) + i;
  return r;
}

static void test_slices() {
  report += "-- slices dealt to the ranks\n";
  int os = 0;
  Net n = build(1, &os);
  IS(os, 0);
  check(np == 1 ? n.ns >= 1 : n.ns >= np, "the optimizer slices to at least one slice for each rank");
  if (n.ns < 8) {  // the groups below need eight slices; this network has them in both libraries
    check(false, "the network has eight slices");
    destroy(n);
    return;
  }
  destroy(n);
  n = build(2, &os);
  IS(os, 0);
  prepare(n);
  check(n.ns >= 8, "a network of at least eight slices for the groups below");
  const Slices parts = slice_parts(n);
  calls();
  {  // the slices add up to the network, whoever contracts them
    std::vector<double> sum(8, 0.0);
    for (auto& p : parts.base)
      for (int i = 0; i < 8; ++i) sum[(size_t)i] += p[(size_t)i];
    cutensornetSliceGroup_t g = nullptr;
    cutensornetDistributedResetConfiguration(h, nullptr, 0);
    IS(cutensornetNetworkContract(h, n.net, 0, n.work, g, 0), 0);
    auto whole = down(n.out, 8);
    for (auto& x : whole) x /= (double)(rk + 1);
    check(max_err(whole, sum) < 1e-12, "the slices contracted alone add up to the whole network");
    set_comm();
    calls();
  }
  struct Case {
    const char* name;
    std::vector<int64_t> ids;  // empty: no group (all slices)
    bool range;
    int64_t step;
    bool accumulate;
  };
  const std::vector<Case> cases = {
      {"all slices", {}, true, 1, false},         {"ids 1 3 5", {1, 3, 5}, false, 1, false},
      {"id 5", {5}, false, 1, false},             {"range 2..7 step 2", {2, 4, 6}, true, 2, false},
      {"id 0", {0}, false, 1, false},             {"ids 7 6 5 4", {7, 6, 5, 4}, false, 1, false},
      {"ids 2 3", {2, 3}, false, 1, false},       {"ids 9 3 7 1 5 2 6", {9, 3, 7, 1, 5, 2, 6}, false, 1, false},
      {"all slices, accumulating", {}, true, 1, true},
      {"ids 1 3 5, accumulating", {1, 3, 5}, false, 1, true},
      {"id 5, accumulating", {5}, false, 1, true}};
  for (const Case& c : cases) {
    if (!c.ids.empty() && *std::max_element(c.ids.begin(), c.ids.end()) >= n.ns) continue;
    std::vector<double> pre(8);
    for (int i = 0; i < 8; ++i) pre[(size_t)i] = c.accumulate ? 100.0 * (rk + 1) + i : 9999.0 + rk;
    cudaMemcpy(n.out, pre.data(), 64, cudaMemcpyHostToDevice);
    cutensornetSliceGroup_t g = nullptr;
    if (!c.ids.empty()) {
      if (c.range) cutensornetCreateSliceGroupFromIDRange(h, c.ids.front(), c.ids.back() + 1, c.step, &g);
      else cutensornetCreateSliceGroupFromIDs(h, c.ids.data(), c.ids.data() + c.ids.size(), &g);
    }
    std::vector<int64_t> all(n.ns), order;
    for (int64_t i = 0; i < n.ns; ++i) all[(size_t)i] = i;
    const auto owner = owners_of(c.ids.empty() ? all : c.ids, c.ids.empty() || c.range, (size_t)n.ns, order);
    calls();
    IS(cutensornetNetworkContract(h, n.net, c.accumulate, n.work, g, 0), 0);
    const std::string seen = data_calls();
    check(np < 2 || seen == "AllreduceInPlace:8:1", (std::string(c.name) + ": the outputs are summed with one in-place sum of 8 doubles").c_str());
    auto got = down(n.out, 8);
    check(max_err(got, expected(parts, owner, c.accumulate)) < 1e-12,
          (std::string(c.name) + ": every slice is contracted by the rank it is dealt to").c_str());
    cutensornetDestroySliceGroup(g);
  }
  // A network with an output that has strides: the library writes it as a compact tensor.
  destroy(n);
}

// ---- the older descriptor-and-plan API ----

static void test_legacy() {
  report += "-- the descriptor and plan API\n";
  int32_t mi[3][3] = {{'i', 'j', 'k'}, {'k', 'l', 0}, {'j', 'l', 'm'}};
  int32_t nmi[3] = {3, 2, 3};
  int64_t e0[3] = {4, 5, 8}, e1[2] = {8, 3}, e2[3] = {5, 3, 2};
  const int64_t* exts[3] = {e0, e1, e2};
  const int32_t* mds[3] = {mi[0], mi[1], mi[2]};
  int32_t om[2] = {'i', 'm'};
  int64_t oe[2] = {4, 2};
  cutensornetNetworkDescriptor_t net = nullptr;
  IS(cutensornetCreateNetworkDescriptor(h, 3, nmi, exts, nullptr, mds, nullptr, 2, oe, nullptr, om, CUDA_R_64F, CUTENSORNET_COMPUTE_64F, &net), 0);
  const size_t sz[3] = {160, 24, 30};
  void* in[3];
  for (int t = 0; t < 3; ++t) {
    std::vector<double> v(sz[t]);
    unsigned sd = 77u + (unsigned)t;
    for (auto& x : v) x = (t == 0 ? (double)(rk + 1) : 1.0) * rnd(sd);
    in[t] = up(v);
  }
  void* o = up(std::vector<double>(8, 0.0));
  cutensornetContractionOptimizerConfig_t cfg;
  cutensornetContractionOptimizerInfo_t info;
  cutensornetWorkspaceDescriptor_t work;
  cutensornetCreateContractionOptimizerConfig(h, &cfg);
  cutensornetCreateContractionOptimizerInfo(h, net, &info);
  cutensornetCreateWorkspaceDescriptor(h, &work);
  int32_t ms = 2;
  cutensornetContractionOptimizerConfigSetAttribute(h, cfg, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES, &ms, 4);
  IS(cutensornetContractionOptimize(h, net, cfg, 1ull << 30, info), 0);
  int64_t ns = -1;
  cutensornetContractionOptimizerInfoGetAttribute(h, info, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES, &ns, 8);
  cutensornetWorkspaceComputeContractionSizes(h, net, info, work);
  int64_t need = 0;
  cutensornetWorkspaceGetMemorySize(h, work, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE,
                                    CUTENSORNET_WORKSPACE_SCRATCH, &need);
  void* scratch;
  cudaMalloc(&scratch, (size_t)need + 256);
  cutensornetWorkspaceSetMemory(h, work, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, scratch, need);
  cutensornetContractionPlan_t plan;
  IS(cutensornetCreateContractionPlan(h, net, info, work, &plan), 0);
  // each slice alone, locally, for the reference
  cutensornetDistributedResetConfiguration(h, nullptr, 0);
  Slices parts;
  for (int64_t sl = 0; sl < ns; ++sl) {
    cudaMemset(o, 0, 64);  // a slice after the first adds to the output
    cutensornetContraction(h, plan, (const void* const*)in, o, work, sl, 0);
    auto p = down(o, 8);
    for (auto& x : p) x /= (double)(rk + 1);
    parts.base.push_back(p);
  }
  set_comm();
  calls();
  std::vector<int64_t> all((size_t)ns), order;
  for (int64_t i = 0; i < ns; ++i) all[(size_t)i] = i;
  cudaMemset(o, 0, 64);
  IS(cutensornetContractSlices(h, plan, (const void* const*)in, o, 0, work, nullptr, 0), 0);
  check(np < 2 || data_calls() == "AllreduceInPlace:8:1", "ContractSlices sums the ranks' outputs");
  {
    const auto got = down(o, 8), want = expected(parts, owners_of(all, true, (size_t)ns, order), false);
    if (max_err(got, want) >= 1e-12) {
      std::string row;
      for (size_t i = 0; i < 8; ++i) row += std::to_string(got[i]) + "/" + std::to_string(want[i]) + " ";
      report += "     got/want: " + row + " ns=" + std::to_string(ns) + "\n";
    }
    check(max_err(got, want) < 1e-12, "ContractSlices deals the slices to the ranks");
  }
  if (ns > 3) {
    cudaMemset(o, 0, 64);
    IS(cutensornetContraction(h, plan, (const void* const*)in, o, work, 3, 0), 0);
    check(calls().empty(), "Contraction of one slice does not communicate");
    std::vector<double> mine(8);
    for (int i = 0; i < 8; ++i) mine[(size_t)i] = (rk + 1) * parts.base[3][(size_t)i];
    check(max_err(down(o, 8), mine) < 1e-12, "and every rank contracts the slice it is given");
  }
  cutensornetDestroyContractionPlan(plan);
  cudaFree(scratch);
  cutensornetDestroyWorkspaceDescriptor(work);
  cutensornetDestroyContractionOptimizerInfo(info);
  cutensornetDestroyContractionOptimizerConfig(cfg);
  cutensornetDestroyNetworkDescriptor(net);
  for (void* p : in) cudaFree(p);
  cudaFree(o);
}

// ---- an output with padding ----

static void test_output_strides() {
  report += "-- output strides\n";
  if (np < 2) return;
  cutensornetNetworkDescriptor_t net;
  cutensornetCreateNetwork(h, &net);
  int32_t ma[2] = {'i', 'k'}, mb[2] = {'k', 'm'}, mo[2] = {'i', 'm'};
  int64_t ea[2] = {4, 8}, eb[2] = {8, 2};
  cutensornetTensorQualifiers_t q{0, 0, 0};
  int64_t ida, idb;
  cutensornetNetworkAppendTensor(h, net, 2, ea, ma, &q, CUDA_R_64F, &ida);
  cutensornetNetworkAppendTensor(h, net, 2, eb, mb, &q, CUDA_R_64F, &idb);
  cutensornetNetworkSetOutputTensor(h, net, 2, mo, CUDA_R_64F);
  void* a = up(std::vector<double>(32, 1.0));
  void* b = up(std::vector<double>(16, 1.0));
  void* out = up(std::vector<double>(16, -7.0));
  cutensornetNetworkSetInputTensorMemory(h, net, ida, a, nullptr);
  cutensornetNetworkSetInputTensorMemory(h, net, idb, b, nullptr);
  const int64_t padded[2] = {1, 6};
  IS(cutensornetNetworkSetOutputTensorMemory(h, net, out, padded), 0);
  cutensornetWorkspaceDescriptor_t work;
  cutensornetCreateWorkspaceDescriptor(h, &work);
  cutensornetContractionOptimizerConfig_t cfg;
  cutensornetCreateContractionOptimizerConfig(h, &cfg);
  cutensornetContractionOptimizerInfo_t info;
  cutensornetCreateContractionOptimizerInfo(h, net, &info);
  IS(cutensornetContractionOptimize(h, net, cfg, 1ull << 30, info), 0);
  cutensornetWorkspaceComputeContractionSizes(h, net, info, work);
  int64_t rec = 0;
  cutensornetWorkspaceGetMemorySize(h, work, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE,
                                    CUTENSORNET_WORKSPACE_SCRATCH, &rec);
  void* scratch;
  cudaMalloc(&scratch, (size_t)rec + 256);
  cutensornetWorkspaceSetMemory(h, work, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, scratch, rec);
  IS(cutensornetNetworkPrepareContraction(h, net, work), 0);
  IS(cutensornetNetworkContract(h, net, 0, work, nullptr, 0), 0);
  const auto got = down(out, 16);
  // The ranks wrote their slices with the strides on top of the output the first 8 elements of which were cleared, and
  // summed those first 8 elements: positions 0-3 and 6-7 hold the sum, 4-5 are cleared padding, 8-9 hold this rank's part.
  const double want[8] = {8, 8, 8, 8, 0, 0, 8, 8};
  bool sums = true, untouched = true;
  for (int i = 0; i < 8; ++i) sums = sums && got[(size_t)i] == want[i];
  for (int i = 10; i < 16; ++i) untouched = untouched && got[(size_t)i] == -7.0;
  check(sums, "the strides are honoured by the contraction, the sum covers the first 8 elements and the clearing too (measured)");
  check(untouched, "and nothing past the strided tensor is written");
  cudaFree(scratch);
  cutensornetDestroyContractionOptimizerInfo(info);
  cutensornetDestroyContractionOptimizerConfig(cfg);
  cutensornetDestroyWorkspaceDescriptor(work);
  cutensornetDestroyNetwork(net);
  cudaFree(a);
  cudaFree(b);
  cudaFree(out);
}

// ---- the state API ----

struct WS {
  cutensornetWorkspaceDescriptor_t d = nullptr;
  void* scratch = nullptr;
  void* cache = nullptr;
  WS() { cutensornetCreateWorkspaceDescriptor(h, &d); }
  ~WS() {
    cudaFree(scratch);
    cudaFree(cache);
    cutensornetDestroyWorkspaceDescriptor(d);
  }
  void alloc() {
    int64_t rc = 0, cm = 0;
    cutensornetWorkspaceGetMemorySize(h, d, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE,
                                      CUTENSORNET_WORKSPACE_SCRATCH, &rc);
    cutensornetWorkspaceGetMemorySize(h, d, CUTENSORNET_WORKSIZE_PREF_MAX, CUTENSORNET_MEMSPACE_DEVICE,
                                      CUTENSORNET_WORKSPACE_CACHE, &cm);
    cudaMalloc(&scratch, (size_t)std::max<int64_t>(rc, 256));
    cutensornetWorkspaceSetMemory(h, d, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, scratch,
                                  std::max<int64_t>(rc, 256));
    if (cm > 0) {
      cudaMalloc(&cache, (size_t)cm);
      cutensornetWorkspaceSetMemory(h, d, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_CACHE, cache, cm);
    }
  }
};

struct StateResults {
  std::vector<cd> amps, marginal;
  cd norm, expectation, exp_norm;
  std::string accessor_calls, expectation_calls, marginal_calls;
};

// A state of four qubits after seven random gates (the same on every rank).
static cutensornetState_t make_state(std::vector<void*>& keep) {
  cutensornetState_t st;
  int64_t ext4[4] = {2, 2, 2, 2};
  cutensornetCreateState(h, CUTENSORNET_STATE_PURITY_PURE, 4, ext4, CUDA_C_64F, &st);
  unsigned sd = 4242;
  auto gate = [&](std::vector<int32_t> modes, size_t D) {
    std::vector<cd> M(D * D);
    for (auto& x : M) {
      const double a = rnd(sd);
      x = cd(a, rnd(sd));
    }
    void* d;
    cudaMalloc(&d, D * D * 16);
    cudaMemcpy(d, M.data(), D * D * 16, cudaMemcpyHostToDevice);
    keep.push_back(d);
    int64_t id;
    cutensornetStateApplyTensorOperator(h, st, (int)modes.size(), modes.data(), d, nullptr, 1, 0, 0, &id);
  };
  for (int q = 0; q < 4; ++q) gate({q}, 2);
  gate({0, 1}, 4);
  gate({2, 3}, 4);
  gate({1, 2}, 4);
  return st;
}

static StateResults state_run() {
  StateResults r;
  std::vector<void*> keep;
  const cutensornetState_t st = make_state(keep);
  int64_t ext4[4] = {2, 2, 2, 2};
  calls();
  {  // amplitudes and the norm
    cutensornetStateAccessor_t acc;
    cutensornetCreateAccessor(h, st, 0, nullptr, nullptr, &acc);
    WS w;
    cutensornetAccessorPrepare(h, acc, (size_t)1 << 30, w.d, 0);
    w.alloc();
    void* amp;
    cudaMalloc(&amp, 16 * 16 + 256);
    cd nrm;
    calls();
    cutensornetAccessorCompute(h, acc, nullptr, w.d, amp, &nrm, 0);
    r.accessor_calls = data_calls();
    r.amps.resize(16);
    cudaDeviceSynchronize();
    cudaMemcpy(r.amps.data(), amp, 256, cudaMemcpyDeviceToHost);
    r.norm = nrm;
    cudaFree(amp);
    cutensornetDestroyAccessor(acc);
  }
  {  // an expectation value of two terms
    cutensornetNetworkOperator_t op;
    cutensornetCreateNetworkOperator(h, 4, ext4, CUDA_C_64F, &op);
    std::vector<cd> Z = {1, 0, 0, -1};
    void* dz;
    cudaMalloc(&dz, 64);
    cudaMemcpy(dz, Z.data(), 64, cudaMemcpyHostToDevice);
    int32_t nm1[1] = {1}, m0[1] = {0}, m3[1] = {3};
    const int32_t* mp0[1] = {m0};
    const int32_t* mp3[1] = {m3};
    const void* dat[1] = {dz};
    int64_t cid;
    cutensornetNetworkOperatorAppendProduct(h, op, make_cuDoubleComplex(0.7, 0.1), 1, nm1, mp0, nullptr, dat, &cid);
    cutensornetNetworkOperatorAppendProduct(h, op, make_cuDoubleComplex(-0.3, 0.2), 1, nm1, mp3, nullptr, dat, &cid);
    cutensornetStateExpectation_t ex;
    cutensornetCreateExpectation(h, st, op, &ex);
    WS w;
    cutensornetExpectationPrepare(h, ex, (size_t)1 << 30, w.d, 0);
    w.alloc();
    calls();
    cutensornetExpectationCompute(h, ex, w.d, &r.expectation, &r.exp_norm, 0);
    r.expectation_calls = data_calls();
    cutensornetDestroyExpectation(ex);
    cutensornetDestroyNetworkOperator(op);
    cudaFree(dz);
  }
  {  // the reduced density matrix of two qubits
    int32_t mm[2] = {0, 1};
    cutensornetStateMarginal_t mg;
    cutensornetCreateMarginal(h, st, 2, mm, 0, nullptr, nullptr, &mg);
    WS w;
    cutensornetMarginalPrepare(h, mg, (size_t)1 << 30, w.d, 0);
    w.alloc();
    void* mt;
    cudaMalloc(&mt, 16 * 16 + 256);
    calls();
    cutensornetMarginalCompute(h, mg, nullptr, w.d, mt, 0);
    r.marginal_calls = data_calls();
    r.marginal.resize(16);
    cudaDeviceSynchronize();
    cudaMemcpy(r.marginal.data(), mt, 256, cudaMemcpyDeviceToHost);
    cudaFree(mt);
    cutensornetDestroyMarginal(mg);
  }
  cutensornetDestroyState(st);
  for (void* p : keep) cudaFree(p);
  return r;
}

// An expectation value of Z on qubit c for each of T terms with the coefficient (c + 1) * (rank + 1).
static cd expectation_of_terms(cutensornetState_t st, int T, int only, double rank_scale, std::string* seen) {
  int64_t ext4[4] = {2, 2, 2, 2};
  cutensornetNetworkOperator_t op;
  cutensornetCreateNetworkOperator(h, 4, ext4, CUDA_C_64F, &op);
  std::vector<cd> Z = {1, 0, 0, -1};
  void* dz;
  cudaMalloc(&dz, 64);
  cudaMemcpy(dz, Z.data(), 64, cudaMemcpyHostToDevice);
  for (int c = 0; c < T; ++c) {
    if (only >= 0 && c != only) continue;
    int32_t nm1[1] = {1}, mq[1] = {c % 4};
    const int32_t* mp[1] = {mq};
    const void* dat[1] = {dz};
    int64_t cid;
    cutensornetNetworkOperatorAppendProduct(h, op, make_cuDoubleComplex((c + 1) * rank_scale, 0.0), 1, nm1, mp, nullptr, dat, &cid);
  }
  cutensornetStateExpectation_t ex;
  cutensornetCreateExpectation(h, st, op, &ex);
  WS w;
  cutensornetExpectationPrepare(h, ex, (size_t)1 << 30, w.d, 0);
  w.alloc();
  cd value, norm;
  calls();
  cutensornetExpectationCompute(h, ex, w.d, &value, &norm, 0);
  if (seen) *seen = data_calls();
  cutensornetDestroyExpectation(ex);
  cutensornetDestroyNetworkOperator(op);
  cudaFree(dz);
  return value;
}

static void test_expectation_terms() {
  report += "-- the terms of an expectation value\n";
  if (np < 2) return;
  std::vector<void*> keep;
  const cutensornetState_t st = make_state(keep);
  // each term alone, without a communicator, with the coefficient 1
  cutensornetDistributedResetConfiguration(h, nullptr, 0);
  cd single[5];
  for (int c = 0; c < 5; ++c) single[c] = expectation_of_terms(st, c + 1, c, 1.0, nullptr) / (double)(c + 1);
  set_comm();
  calls();
  std::set<int> counts = {1, 2, 3, 4, 5};  // fewer terms than ranks, as many, and more, whatever the ranks
  for (int T : counts) {
    if (T < 1 || T > 5) continue;
    std::string seen;
    const cd got = expectation_of_terms(st, T, -1, (double)(rk + 1), &seen);
    cd want = 0;
    std::string expect_calls;
    if (T >= np) {  // whole terms, to rank c % np with its coefficient; one sum
      for (int c = 0; c < T; ++c) want += (double)((c + 1) * (c % np + 1)) * single[c];
      expect_calls = "AllreduceInPlace:1:5 AllreduceInPlace:1:5";
    } else {  // each term by all the ranks, one sum for each; this rank's coefficient
      for (int c = 0; c < T; ++c) want += (double)((c + 1) * (rk + 1)) * single[c];
      for (int c = 0; c < T; ++c) expect_calls += "AllreduceInPlace:1:5 ";
      expect_calls += "AllreduceInPlace:1:5";
    }
    const std::string what = std::to_string(T) + (T == 1 ? " term" : " terms");
    if (seen != expect_calls) report += "     primitives called: " + seen + "\n";
    check(seen == expect_calls, (what + ": the sums of the primitives").c_str());
    check(std::abs(got - want) < 1e-9 * (std::abs(want) + 1e-3), (what + ": the value, with the coefficients of the rank that contracts each term").c_str());
  }
  cutensornetDestroyState(st);
  for (void* p : keep) cudaFree(p);
}

static double dist(const std::vector<cd>& a, const std::vector<cd>& b) {
  double e = 0, m = 1e-300;
  for (size_t i = 0; i < a.size(); ++i) e = std::max(e, std::abs(a[i] - b[i])), m = std::max(m, std::abs(b[i]));
  return e / m;
}

static void test_state() {
  report += "-- the state API\n";
  cutensornetDistributedResetConfiguration(h, nullptr, 0);
  const StateResults serial = state_run();
  check(serial.accessor_calls.empty() && serial.expectation_calls.empty() && serial.marginal_calls.empty(),
        "without a communicator no primitive is called");
  set_comm();
  calls();
  const StateResults par = state_run();
  check(dist(par.amps, serial.amps) < 1e-10, "the amplitudes are the same");
  check(std::abs(par.norm - serial.norm) < 1e-10 * std::abs(serial.norm), "and the norm");
  check(std::abs(par.expectation - serial.expectation) < 1e-10 * std::abs(serial.expectation), "the expectation value of the two terms");
  check(std::abs(par.exp_norm - serial.exp_norm) < 1e-10 * std::abs(serial.exp_norm), "and its norm");
  check(dist(par.marginal, serial.marginal) < 1e-10, "the reduced density matrix");
  if (np < 2) return;
  // CUDA_C_64F is 5: each output tensor and each norm is summed in place, an expectation value's terms together.
  auto same = [](const std::string& got, const char* want, const char* what) {
    if (got != want) report += "     primitives called: " + got + "\n";
    check(got == want, what);
  };
  same(par.accessor_calls, "AllreduceInPlace:16:5 AllreduceInPlace:1:5", "amplitudes, then the norm, are summed in place");
  // two terms: on two ranks the ranks share them and sum once; on more, each term is summed on its own
  same(par.expectation_calls, np <= 2 ? "AllreduceInPlace:1:5 AllreduceInPlace:1:5" : "AllreduceInPlace:1:5 AllreduceInPlace:1:5 AllreduceInPlace:1:5",
       "the value of two terms, then the norm, are summed in place");
  same(par.marginal_calls, "AllreduceInPlace:16:5", "the reduced density matrix is summed in place");
}

// ---- distributed tensors ----

static void test_distributed_tensor() {
  report += "-- distributed tensor descriptors\n";
  int64_t ex[2] = {4, 6}, nr[2] = {np, 1};
  int32_t ml[2] = {'a', 'b'};
  cutensornetTensorDescriptor_t td = nullptr;
  cutensornetDistributedResetConfiguration(h, nullptr, 0);
  IS(cutensornetCreateDistributedTensorDescriptor(h, 2, ex, nullptr, nullptr, nullptr, nr, ml, CUDA_R_64F, &td),
     CUTENSORNET_STATUS_NOT_INITIALIZED);
  IS(cutensornetCreateDistributedTensorDescriptor(nullptr, 2, ex, nullptr, nullptr, nullptr, nr, ml, CUDA_R_64F, &td),
     CUTENSORNET_STATUS_NOT_INITIALIZED);
  set_comm();
  calls();
  const cutensornetStatus_t s =
      cutensornetCreateDistributedTensorDescriptor(h, 2, ex, nullptr, nullptr, nullptr, nr, ml, CUDA_R_64F, &td);
  // NVIDIA's library loads NCCL, cuTensorMp and cuSOLVERMp here, and refuses where they are not installed.
  check(s == CUTENSORNET_STATUS_NOT_SUPPORTED || (card && s == 0), "a distributed descriptor needs the optional backends (NOT_SUPPORTED without)");
  if (s == 0 && td) cutensornetDestroyTensorDescriptor(td);
  const std::string seen = calls();
  check(seen == "getNumRanks Allgather:1:10", "after gathering one word from every rank");
}

// ---- a library that cannot be loaded ----

static void test_unloadable(const std::string& mode) {
  report += "-- " + mode + "\n";
  int32_t v = -5;
  if (mode == "nolib") {
    IS(cutensornetDistributedResetConfiguration(h, nullptr, 0), 0);
    IS(cutensornetDistributedResetConfiguration(h, nullptr, sizeof comm), 0);
    IS(cutensornetDistributedResetConfiguration(h, &comm, sizeof comm), CUTENSORNET_STATUS_DISTRIBUTED_FAILURE);
  } else {  // nosym, version: the library is found, but is not one
    IS(cutensornetDistributedResetConfiguration(h, nullptr, 0), CUTENSORNET_STATUS_DISTRIBUTED_FAILURE);
    IS(cutensornetDistributedResetConfiguration(h, nullptr, sizeof comm), CUTENSORNET_STATUS_DISTRIBUTED_FAILURE);
    IS(cutensornetDistributedResetConfiguration(h, &comm, sizeof comm), CUTENSORNET_STATUS_DISTRIBUTED_FAILURE);
  }
  IS(cutensornetDistributedResetConfiguration(h, &comm, 0), CUTENSORNET_STATUS_INVALID_VALUE);
  IS(cutensornetDistributedGetNumRanks(h, &v), 0);
  check(v == 1, "and the world stays one rank");
  IS(cutensornetDistributedSynchronize(h), 0);
  int64_t ex[2] = {4, 6}, nr[2] = {np, 1};
  int32_t ml[2] = {'a', 'b'};
  cutensornetTensorDescriptor_t td = nullptr;
  IS(cutensornetCreateDistributedTensorDescriptor(h, 2, ex, nullptr, nullptr, nullptr, nr, ml, CUDA_R_64F, &td),
     CUTENSORNET_STATUS_NOT_INITIALIZED);
  // contractions run on this rank alone
  int os = 0;
  Net n = build(2, &os);
  IS(os, 0);
  prepare(n);
  IS(cutensornetNetworkContract(h, n.net, 0, n.work, nullptr, 0), 0);
  destroy(n);
}

// ---- driver ----

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "main";
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rk);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  card = std::getenv("VGPU_E2E_REAL") != nullptr;
  int ndev = 0;
  cudaGetDeviceCount(&ndev);
  if (ndev < 1) {
    if (rk == 0) std::printf("SKIP: no device\n");
    MPI_Finalize();
    return 0;
  }
  cudaSetDevice(rk % ndev);
  if (cutensornetCreate(&h)) {
    std::printf("cutensornetCreate failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (mode == "main") {
    test_communicator();
    test_failures();
    test_slices();
    test_legacy();
    test_output_strides();
    test_state();
    test_expectation_terms();
    test_distributed_tensor();
  } else {
    test_unloadable(mode);
  }
  cutensornetDestroy(h);
  // Rank 0 prints its checks and the failures of the others, in a fixed order.
  int mine = (int)failed.size();
  std::vector<int> sizes((size_t)np);
  MPI_Gather(&mine, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  int total_failures = 0;
  MPI_Allreduce(&failures, &total_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rk == 0) {
    std::fputs(report.c_str(), stdout);
    for (int r = 1; r < np; ++r) {
      std::string s((size_t)sizes[(size_t)r], '\0');
      if (sizes[(size_t)r]) MPI_Recv(&s[0], sizes[(size_t)r], MPI_CHAR, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::fputs(s.c_str(), stdout);
    }
    std::printf("%s\n", total_failures == 0 ? "PASS" : "FAIL");
  } else if (mine) {
    MPI_Send(failed.data(), mine, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  }
  MPI_Comm_free(&comm);
  MPI_Finalize();
  return total_failures == 0 ? 0 : 1;
}
