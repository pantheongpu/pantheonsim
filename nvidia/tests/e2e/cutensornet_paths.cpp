// cuTensorNet: what a cuQuantum Python tensor-network program calls --
// networks built tensor by tensor, the contraction optimizer and its
// information, workspace sizing, slicing and slice groups, contraction --
// and the older descriptor-and-plan API, packed optimizer information,
// conjugated tensors, hyperedges, QR, SVD with truncation and partitions, and
// gate splitting.
//
// The expectations hold for NVIDIA's libcutensornet 2.14 on an RTX 3060 too:
// this program passes there. The contraction path, the slicing the
// optimizer picks, FLOP counts and workspace sizes are each library's own
// (NVIDIA's hyper-optimizer and this library's greedy search differ), so
// the checks compare contracted results against a host reference and
// decompositions by their defining properties, and take sizes only as the
// library reports them.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "../../include/vgpu_cutensornet.h"

using cd = std::complex<double>;

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

static cutensornetHandle_t h;

// ---- host tensors (column-major) ----

struct T {
  std::vector<int32_t> modes;
  std::vector<int64_t> ext;
  std::vector<cd> v;
};

static std::vector<cd> randoms(size_t n, unsigned seed, bool cx) {
  std::vector<cd> v(n);
  unsigned s = 0x9e3779b9u * (seed + 1);
  auto next = [&] {
    s ^= s << 13, s ^= s >> 17, s ^= s << 5;
    return ((double)(s % 20001) - 10000.0) / 10000.0;
  };
  for (auto& x : v) {
    const double re = next();
    x = cx ? cd(re, next()) : cd(re, 0);
  }
  return v;
}

static T make(std::vector<int32_t> modes, std::map<int32_t, int64_t>& ext, unsigned seed, bool cx) {
  T t;
  t.modes = modes;
  size_t n = 1;
  for (int32_t m : modes) t.ext.push_back(ext[m]), n *= (size_t)ext[m];
  t.v = randoms(n, seed, cx);
  return t;
}

// out[out_modes] = sum over the rest of prod_i op(in_i).
static std::vector<cd> einsum(const std::vector<const T*>& in, const std::vector<int32_t>& out,
                              std::map<int32_t, int64_t>& ext, const std::vector<bool>& conj = {}) {
  std::vector<int32_t> all = out;
  for (const T* t : in)
    for (int32_t m : t->modes)
      if (std::find(all.begin(), all.end(), m) == all.end()) all.push_back(m);
  size_t nout = 1;
  for (int32_t m : out) nout *= (size_t)ext[m];
  std::vector<cd> r(nout);
  std::vector<int64_t> idx(all.size(), 0);
  for (;;) {
    cd p = 1;
    for (size_t i = 0; i < in.size(); ++i) {
      size_t off = 0, st = 1;
      for (size_t k = 0; k < in[i]->modes.size(); ++k) {
        const size_t pos = (size_t)(std::find(all.begin(), all.end(), in[i]->modes[k]) - all.begin());
        off += (size_t)idx[pos] * st;
        st *= (size_t)in[i]->ext[k];
      }
      cd x = in[i]->v[off];
      p *= (i < conj.size() && conj[i]) ? std::conj(x) : x;
    }
    size_t o = 0, st = 1;
    for (size_t k = 0; k < out.size(); ++k) o += (size_t)idx[k] * st, st *= (size_t)ext[out[k]];
    r[o] += p;
    size_t k = 0;
    for (; k < all.size(); ++k) {
      if (++idx[k] < ext[all[k]]) break;
      idx[k] = 0;
    }
    if (k == all.size()) break;
  }
  return r;
}

static size_t esize(cudaDataType_t t) {
  return t == CUDA_R_32F ? 4 : (t == CUDA_R_64F || t == CUDA_C_32F) ? 8 : 16;
}
static bool cplx(cudaDataType_t t) { return t == CUDA_C_32F || t == CUDA_C_64F; }

static void* upload(const std::vector<cd>& v, cudaDataType_t t) {
  std::vector<unsigned char> b(v.size() * esize(t));
  for (size_t i = 0; i < v.size(); ++i) {
    unsigned char* p = b.data() + i * esize(t);
    if (t == CUDA_R_32F) { float x = (float)v[i].real(); std::memcpy(p, &x, 4); }
    else if (t == CUDA_R_64F) { double x = v[i].real(); std::memcpy(p, &x, 8); }
    else if (t == CUDA_C_32F) { float x[2] = {(float)v[i].real(), (float)v[i].imag()}; std::memcpy(p, x, 8); }
    else { double x[2] = {v[i].real(), v[i].imag()}; std::memcpy(p, x, 16); }
  }
  void* d = nullptr;
  cudaMalloc(&d, b.size() + 256);
  cudaMemcpy(d, b.data(), b.size(), cudaMemcpyHostToDevice);
  return d;
}

static std::vector<cd> download(const void* d, size_t n, cudaDataType_t t) {
  std::vector<unsigned char> b(n * esize(t));
  cudaDeviceSynchronize();
  cudaMemcpy(b.data(), d, b.size(), cudaMemcpyDeviceToHost);
  std::vector<cd> v(n);
  for (size_t i = 0; i < n; ++i) {
    const unsigned char* p = b.data() + i * esize(t);
    if (t == CUDA_R_32F) { float x; std::memcpy(&x, p, 4); v[i] = x; }
    else if (t == CUDA_R_64F) { double x; std::memcpy(&x, p, 8); v[i] = x; }
    else if (t == CUDA_C_32F) { float x[2]; std::memcpy(x, p, 8); v[i] = cd(x[0], x[1]); }
    else { double x[2]; std::memcpy(x, p, 16); v[i] = cd(x[0], x[1]); }
  }
  return v;
}

static double rel_err(const std::vector<cd>& got, const std::vector<cd>& want) {
  double num = 0, den = 0;
  for (size_t i = 0; i < want.size() && i < got.size(); ++i) num += std::norm(got[i] - want[i]), den += std::norm(want[i]);
  return got.size() == want.size() ? std::sqrt(num / (den > 0 ? den : 1)) : 1e300;
}

static double tol(cudaDataType_t t) { return (t == CUDA_R_32F || t == CUDA_C_32F) ? 1e-5 : 1e-12; }

// ---- the network API, as cuQuantum Python drives it ----

struct Net {
  cutensornetNetworkDescriptor_t net = nullptr;
  cutensornetWorkspaceDescriptor_t work = nullptr;
  cutensornetContractionOptimizerConfig_t config = nullptr;
  cutensornetContractionOptimizerInfo_t info = nullptr;
  void* scratch = nullptr;
  int64_t scratch_size = 0;
  std::vector<void*> in;
  void* out = nullptr;
};

static Net build(const std::vector<const T*>& ts, const std::vector<int32_t>& out, std::map<int32_t, int64_t>& ext,
                 cudaDataType_t type, const std::vector<int>& conj = {}) {
  Net n;
  IS(cutensornetCreateNetwork(h, &n.net), 0);
  for (size_t i = 0; i < ts.size(); ++i) {
    cutensornetTensorQualifiers_t q{0, 0, 0};
    q.isConjugate = i < conj.size() ? conj[i] : 0;
    int64_t id = -1;
    cutensornetNetworkAppendTensor(h, n.net, (int32_t)ts[i]->modes.size(), ts[i]->ext.data(), ts[i]->modes.data(), &q,
                                   type, &id);
    if (id != (int64_t)i) std::printf("     (tensor id %lld for input %zu)\n", (long long)id, i);
  }
  IS(cutensornetNetworkSetOutputTensor(h, n.net, (int32_t)out.size(), out.data(), type), 0);
  for (size_t i = 0; i < ts.size(); ++i) {
    n.in.push_back(upload(ts[i]->v, type));
    cutensornetNetworkSetInputTensorMemory(h, n.net, (int64_t)i, n.in.back(), nullptr);
  }
  size_t nout = 1;
  for (int32_t m : out) nout *= (size_t)ext[m];
  n.out = upload(std::vector<cd>(nout, cd(7, 7)), type);
  IS(cutensornetNetworkSetOutputTensorMemory(h, n.net, n.out, nullptr), 0);
  cutensornetCreateWorkspaceDescriptor(h, &n.work);
  cutensornetCreateContractionOptimizerConfig(h, &n.config);
  cutensornetCreateContractionOptimizerInfo(h, n.net, &n.info);
  return n;
}

// Sizes, scratch at the recommended size, prepare.
static bool prepare(Net& n) {
  if (cutensornetWorkspaceComputeContractionSizes(h, n.net, n.info, n.work)) return false;
  int64_t mn = -1, rec = -1, mx = -1;
  cutensornetWorkspaceGetMemorySize(h, n.work, CUTENSORNET_WORKSIZE_PREF_MIN, CUTENSORNET_MEMSPACE_DEVICE,
                                    CUTENSORNET_WORKSPACE_SCRATCH, &mn);
  cutensornetWorkspaceGetMemorySize(h, n.work, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE,
                                    CUTENSORNET_WORKSPACE_SCRATCH, &rec);
  cutensornetWorkspaceGetMemorySize(h, n.work, CUTENSORNET_WORKSIZE_PREF_MAX, CUTENSORNET_MEMSPACE_DEVICE,
                                    CUTENSORNET_WORKSPACE_SCRATCH, &mx);
  if (mn < 0 || mn > rec || rec > mx) std::printf("     (workspace sizes out of order: %lld %lld %lld)\n", (long long)mn, (long long)rec, (long long)mx);
  n.scratch_size = rec > 0 ? rec : 256;
  if (!n.scratch) cudaMalloc(&n.scratch, (size_t)n.scratch_size);
  cutensornetWorkspaceSetMemory(h, n.work, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, n.scratch,
                                n.scratch_size);
  cutensornetWorkspaceSetMemory(h, n.work, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_CACHE, nullptr, 0);
  return cutensornetNetworkPrepareContraction(h, n.net, n.work) == 0;
}

static void destroy(Net& n) {
  for (void* p : n.in) cudaFree(p);
  cudaFree(n.out);
  cudaFree(n.scratch);
  cutensornetDestroyContractionOptimizerInfo(n.info);
  cutensornetDestroyContractionOptimizerConfig(n.config);
  cutensornetDestroyWorkspaceDescriptor(n.work);
  cutensornetDestroyNetwork(n.net);
}

static int64_t num_slices(cutensornetContractionOptimizerInfo_t info) {
  int64_t ns = -1;
  cutensornetContractionOptimizerInfoGetAttribute(h, info, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES, &ns, 8);
  return ns;
}

static void network_types() {
  for (cudaDataType_t type : {CUDA_R_64F, CUDA_C_64F, CUDA_R_32F, CUDA_C_32F}) {
    const std::string tn = type == CUDA_R_64F ? "R64F" : type == CUDA_C_64F ? "C64F" : type == CUDA_R_32F ? "R32F" : "C32F";
    // ijk,kl,jlm->im
    std::map<int32_t, int64_t> ext = {{'i', 4}, {'j', 5}, {'k', 8}, {'l', 3}, {'m', 2}};
    T a = make({'i', 'j', 'k'}, ext, 1, cplx(type)), b = make({'k', 'l'}, ext, 2, cplx(type)),
      c = make({'j', 'l', 'm'}, ext, 3, cplx(type));
    const std::vector<int32_t> out = {'i', 'm'};
    Net n = build({&a, &b, &c}, out, ext, type);
    IS(cutensornetContractionOptimize(h, n.net, n.config, 1ull << 30, n.info), 0);
    check(num_slices(n.info) == 1, (tn + ": no slicing when the workspace allows").c_str());
    cutensornetNodePair_t pairs[2];
    cutensornetContractionPath_t path{2, pairs};
    IS(cutensornetContractionOptimizerInfoGetAttribute(h, n.info, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH, &path, sizeof path), 0);
    check(path.numContractions == 2, (tn + ": a path of two pairwise contractions").c_str());
    int32_t nim[2] = {-1, -1};
    IS(cutensornetContractionOptimizerInfoGetAttribute(h, n.info, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_INTERMEDIATE_MODES, nim, 8), 0);
    check(nim[1] == 2, (tn + ": the last intermediate is the output (2 modes)").c_str());
    double flops = 0, largest = 0;
    IS(cutensornetContractionOptimizerInfoGetAttribute(h, n.info, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT, &flops, 8), 0);
    IS(cutensornetContractionOptimizerInfoGetAttribute(h, n.info, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_LARGEST_TENSOR, &largest, 8), 0);
    check(flops > 0 && largest >= 8, (tn + ": FLOP count and largest tensor reported").c_str());
    IS(prepare(n), true);
    IS(cutensornetNetworkContract(h, n.net, 0, n.work, nullptr, 0), 0);
    const std::vector<cd> want = einsum({&a, &b, &c}, out, ext);
    check(rel_err(download(n.out, 8, type), want) < tol(type), (tn + ": contracted network matches the host").c_str());
    // Accumulating adds a second copy.
    IS(cutensornetNetworkContract(h, n.net, 1, n.work, nullptr, 0), 0);
    std::vector<cd> twice = want;
    for (auto& x : twice) x *= 2;
    check(rel_err(download(n.out, 8, type), twice) < tol(type), (tn + ": accumulateOutput adds").c_str());
    destroy(n);
  }
}

static void slicing() {
  const cudaDataType_t type = CUDA_R_64F;
  std::map<int32_t, int64_t> ext = {{'i', 4}, {'j', 5}, {'k', 8}, {'l', 3}, {'m', 2}};
  T a = make({'i', 'j', 'k'}, ext, 11, false), b = make({'k', 'l'}, ext, 12, false), c = make({'j', 'l', 'm'}, ext, 13, false);
  const std::vector<int32_t> out = {'i', 'm'};
  const std::vector<cd> want = einsum({&a, &b, &c}, out, ext);
  // The optimizer's own slicing, at least two slices.
  {
    Net n = build({&a, &b, &c}, out, ext, type);
    int32_t min_slices = 2;
    IS(cutensornetContractionOptimizerConfigSetAttribute(h, n.config, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES, &min_slices, 4), 0);
    int32_t back = 0;
    IS(cutensornetContractionOptimizerConfigGetAttribute(h, n.config, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES, &back, 4), 0);
    check(back == 2, "SLICER_MIN_SLICES reads back");
    IS(cutensornetContractionOptimize(h, n.net, n.config, 1ull << 30, n.info), 0);
    const int64_t ns = num_slices(n.info);
    check(ns >= 2, "the optimizer slices into at least the minimum");
    int32_t nsm = 0;
    IS(cutensornetContractionOptimizerInfoGetAttribute(h, n.info, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES, &nsm, 4), 0);
    std::vector<cutensornetSliceInfoPair_t> pairs((size_t)std::max(nsm, 1));
    cutensornetSlicingConfig_t sc{(uint32_t)nsm, pairs.data()};
    IS(cutensornetContractionOptimizerInfoGetAttribute(h, n.info, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_CONFIG, &sc, sizeof sc), 0);
    int64_t prod = 1;
    for (int32_t s = 0; s < nsm; ++s) prod *= (ext[pairs[(size_t)s].slicedMode] + pairs[(size_t)s].slicedExtent - 1) / pairs[(size_t)s].slicedExtent;
    check(nsm >= 1 && prod == ns, "NUM_SLICES is the product of ceil(extent / sliced extent)");
    IS(prepare(n), true);
    IS(cutensornetNetworkContract(h, n.net, 0, n.work, nullptr, 0), 0);
    check(rel_err(download(n.out, 8, type), want) < 1e-12, "all slices contracted give the network");
    // Two slice groups, the second accumulating, give the same.
    cutensornetSliceGroup_t g1 = nullptr, g2 = nullptr;
    IS(cutensornetCreateSliceGroupFromIDRange(h, 0, ns / 2, 1, &g1), 0);
    std::vector<int64_t> rest;
    for (int64_t i = ns - 1; i >= ns / 2; --i) rest.push_back(i);
    rest.push_back(ns - 1);  // a duplicate, which is dropped
    IS(cutensornetCreateSliceGroupFromIDs(h, rest.data(), rest.data() + rest.size(), &g2), 0);
    IS(cutensornetNetworkContract(h, n.net, 0, n.work, g1, 0), 0);
    IS(cutensornetNetworkContract(h, n.net, 1, n.work, g2, 0), 0);
    check(rel_err(download(n.out, 8, type), want) < 1e-12, "two slice groups add up to the network");
    cutensornetDestroySliceGroup(g1);
    cutensornetDestroySliceGroup(g2);
    destroy(n);
  }
  // A path and slicing the caller chose: (0,1), (0,1), k cut in pieces of 2.
  {
    Net n = build({&a, &b, &c}, out, ext, type);
    cutensornetNodePair_t pairs[2] = {{0, 1}, {0, 1}};
    cutensornetContractionPath_t path{2, pairs};
    IS(cutensornetContractionOptimizerInfoSetAttribute(h, n.info, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH, &path, sizeof path), 0);
    cutensornetSliceInfoPair_t sl[1] = {{'k', 2}};
    cutensornetSlicingConfig_t sc{1, sl};
    IS(cutensornetContractionOptimizerInfoSetAttribute(h, n.info, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_CONFIG, &sc, sizeof sc), 0);
    check(num_slices(n.info) == 4, "k (extent 8) sliced in pieces of 2 is 4 slices");
    int32_t im[5] = {0};
    IS(cutensornetContractionOptimizerInfoGetAttribute(h, n.info, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_INTERMEDIATE_MODES, im, 20), 0);
    check(im[0] == 'i' && im[1] == 'j' && im[2] == 'l' && im[3] == 'i' && im[4] == 'm',
          "the intermediates of path (0,1),(0,1) are ijl then im");
    IS(cutensornetNetworkSetOptimizerInfo(h, n.net, n.info), 0);
    IS(prepare(n), true);
    IS(cutensornetNetworkContract(h, n.net, 0, n.work, nullptr, 0), 0);
    check(rel_err(download(n.out, 8, type), want) < 1e-12, "the caller's path and slicing give the network");
    // Packed and unpacked, the info keeps its path and slicing.
    size_t sz = 0;
    IS(cutensornetContractionOptimizerInfoGetPackedSize(h, n.info, &sz), 0);
    std::vector<unsigned char> buf(sz);
    IS(cutensornetContractionOptimizerInfoPackData(h, n.info, buf.data(), sz), 0);
    cutensornetContractionOptimizerInfo_t copy = nullptr;
    IS(cutensornetCreateContractionOptimizerInfoFromPackedData(h, n.net, buf.data(), sz, &copy), 0);
    check(num_slices(copy) == 4, "an info rebuilt from its packed form keeps its slicing");
    cutensornetNodePair_t got[2] = {{9, 9}, {9, 9}};
    cutensornetContractionPath_t gp{2, got};
    cutensornetContractionOptimizerInfoGetAttribute(h, copy, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH, &gp, sizeof gp);
    check(gp.numContractions == 2 && got[0].first == 0 && got[0].second == 1, "and its path");
    IS(cutensornetUpdateContractionOptimizerInfoFromPackedData(h, buf.data(), sz, copy), 0);
    cutensornetDestroyContractionOptimizerInfo(copy);
    destroy(n);
  }
}

static void network_shapes() {
  // Conjugated inputs (complex), a hyperedge (k in three tensors), and a
  // single tensor reduced and permuted.
  const cudaDataType_t type = CUDA_C_64F;
  std::map<int32_t, int64_t> ext = {{'a', 3}, {'b', 4}, {'c', 2}, {'k', 5}, {'d', 3}};
  T x = make({'a', 'k'}, ext, 21, true), y = make({'k', 'b'}, ext, 22, true), z = make({'k', 'c', 'd'}, ext, 23, true);
  {
    const std::vector<int32_t> out = {'a', 'b', 'c', 'k'};
    Net n = build({&x, &y, &z}, out, ext, type, {1, 0, 1});
    int32_t nconj = -1;
    IS(cutensornetNetworkGetAttribute(h, n.net, CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_CONJUGATED, &nconj, 4), 0);
    check(nconj == 2, "two conjugated inputs");
    IS(cutensornetContractionOptimize(h, n.net, n.config, 1ull << 30, n.info), 0);
    IS(prepare(n), true);
    IS(cutensornetNetworkContract(h, n.net, 0, n.work, nullptr, 0), 0);
    const std::vector<cd> want = einsum({&x, &y, &z}, out, ext, {true, false, true});
    check(rel_err(download(n.out, 3 * 4 * 2 * 5, type), want) < 1e-12,
          "conjugated inputs and a mode in three tensors and the output");
    destroy(n);
  }
  {
    const std::vector<int32_t> out = {'d', 'k'};
    Net n = build({&z}, out, ext, type);
    IS(cutensornetContractionOptimize(h, n.net, n.config, 1ull << 30, n.info), 0);
    IS(prepare(n), true);
    IS(cutensornetNetworkContract(h, n.net, 0, n.work, nullptr, 0), 0);
    check(rel_err(download(n.out, 15, type), einsum({&z}, out, ext)) < 1e-12, "a one-tensor network sums and permutes");
    destroy(n);
  }
  {
    // The compute type: 32F on double data.
    const cudaDataType_t rt = CUDA_R_64F;
    std::map<int32_t, int64_t> e2 = {{'i', 6}, {'j', 7}, {'k', 5}};
    T p = make({'i', 'k'}, e2, 31, false), q = make({'k', 'j'}, e2, 32, false);
    Net n = build({&p, &q}, {'i', 'j'}, e2, rt);
    int32_t ct = 0;
    IS(cutensornetNetworkGetAttribute(h, n.net, CUTENSORNET_NETWORK_COMPUTE_TYPE, &ct, 4), 0);
    check(ct == CUTENSORNET_COMPUTE_64F, "the default compute type of R64F is 64F");
    ct = CUTENSORNET_COMPUTE_32F;
    IS(cutensornetNetworkSetAttribute(h, n.net, CUTENSORNET_NETWORK_COMPUTE_TYPE, &ct, 4), 0);
    IS(cutensornetContractionOptimize(h, n.net, n.config, 1ull << 30, n.info), 0);
    IS(prepare(n), true);
    IS(cutensornetNetworkContract(h, n.net, 0, n.work, nullptr, 0), 0);
    const double e = rel_err(download(n.out, 42, rt), einsum({&p, &q}, {'i', 'j'}, e2));
    check(e < 1e-5, "R64F computed in 32F is right to single precision");
    cutensornetTensorDescriptor_t od = nullptr;
    IS(cutensornetGetOutputTensorDescriptor(h, n.net, &od), 0);
    int32_t nm = 0;
    size_t bytes = 0;
    int32_t modes[2] = {0, 0};
    int64_t ex[2] = {0, 0}, st[2] = {0, 0};
    IS(cutensornetGetTensorDetails(h, od, &nm, &bytes, modes, ex, st), 0);
    check(nm == 2 && bytes == 42 * 8 && modes[0] == 'i' && ex[1] == 7 && st[1] == 6, "the output tensor's details");
    cutensornetDestroyTensorDescriptor(od);
    destroy(n);
  }
}

// ---- the descriptor-and-plan API ----

static void legacy() {
  const cudaDataType_t type = CUDA_R_64F;
  std::map<int32_t, int64_t> ext = {{'i', 4}, {'j', 5}, {'k', 8}, {'l', 3}, {'m', 2}};
  T a = make({'i', 'j', 'k'}, ext, 41, false), b = make({'k', 'l'}, ext, 42, false), c = make({'j', 'l', 'm'}, ext, 43, false);
  const std::vector<int32_t> out = {'m', 'i'};
  const int32_t nmi[] = {3, 2, 3};
  const int64_t* exts[] = {a.ext.data(), b.ext.data(), c.ext.data()};
  const int32_t* mds[] = {a.modes.data(), b.modes.data(), c.modes.data()};
  // A in a strided layout: i padded to 6.
  const int64_t sa[] = {1, 6, 30};
  const int64_t* strides[] = {sa, nullptr, nullptr};
  cutensornetNetworkDescriptor_t net = nullptr;
  // The output's extents are required (INVALID_VALUE without them).
  IS(cutensornetCreateNetworkDescriptor(h, 3, nmi, exts, strides, mds, nullptr, 2, nullptr, nullptr, out.data(), type,
                                        CUTENSORNET_COMPUTE_64F, &net), CUTENSORNET_STATUS_INVALID_VALUE);
  const int64_t eout[] = {2, 4};
  IS(cutensornetCreateNetworkDescriptor(h, 3, nmi, exts, strides, mds, nullptr, 2, eout, nullptr, out.data(), type,
                                        CUTENSORNET_COMPUTE_64F, &net), 0);
  std::vector<cd> ap(6 * 5 * 8, cd(99, 0));
  for (int k = 0; k < 8; ++k)
    for (int j = 0; j < 5; ++j)
      for (int i = 0; i < 4; ++i) ap[(size_t)(i + 6 * (j + 5 * k))] = a.v[(size_t)(i + 4 * (j + 5 * k))];
  void* in[3] = {upload(ap, type), upload(b.v, type), upload(c.v, type)};
  void* o = upload(std::vector<cd>(8), type);
  cutensornetContractionOptimizerConfig_t cfg = nullptr;
  cutensornetContractionOptimizerInfo_t info = nullptr;
  cutensornetWorkspaceDescriptor_t work = nullptr;
  cutensornetCreateContractionOptimizerConfig(h, &cfg);
  cutensornetCreateContractionOptimizerInfo(h, net, &info);
  cutensornetCreateWorkspaceDescriptor(h, &work);
  int32_t ms = 2;
  cutensornetContractionOptimizerConfigSetAttribute(h, cfg, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES, &ms, 4);
  IS(cutensornetContractionOptimize(h, net, cfg, 1ull << 30, info), 0);
  IS(cutensornetWorkspaceComputeContractionSizes(h, net, info, work), 0);
  int64_t need = -1;
  IS(cutensornetWorkspaceGetMemorySize(h, work, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE,
                                       CUTENSORNET_WORKSPACE_SCRATCH, &need), 0);
  void* scratch = nullptr;
  cudaMalloc(&scratch, (size_t)std::max<int64_t>(need, 256));
  IS(cutensornetWorkspaceSetMemory(h, work, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, scratch,
                                   std::max<int64_t>(need, 256)), 0);
  cutensornetContractionPlan_t plan = nullptr;
  IS(cutensornetCreateContractionPlan(h, net, info, work, &plan), 0);
  cutensornetContractionAutotunePreference_t tp = nullptr;
  IS(cutensornetCreateContractionAutotunePreference(h, &tp), 0);
  int32_t iters = 0;
  IS(cutensornetContractionAutotunePreferenceGetAttribute(h, tp, CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS, &iters, 4), 0);
  check(iters == 3, "autotune iterations default to 3");
  IS(cutensornetContractionAutotune(h, plan, in, o, work, tp, 0), 0);
  IS(cutensornetContractSlices(h, plan, in, o, 0, work, nullptr, 0), 0);
  const std::vector<cd> want = einsum({&a, &b, &c}, out, ext);
  check(rel_err(download(o, 8, type), want) < 1e-12, "ContractSlices over a strided input");
  // Slice by slice, ascending from 0.
  const int64_t ns = num_slices(info);
  check(ns >= 2, "the legacy network is sliced");
  for (int64_t s = 0; s < ns; ++s) IS(cutensornetContraction(h, plan, in, o, work, s, 0), 0);
  check(rel_err(download(o, 8, type), want) < 1e-12, "cutensornetContraction slice by slice");
  int32_t nmo = 0;
  int64_t oe[2] = {0, 0};
  int32_t om[2] = {0, 0};
  IS(cutensornetGetOutputTensorDetails(h, net, &nmo, nullptr, om, oe, nullptr), 0);
  check(nmo == 2 && om[0] == 'm' && oe[1] == 4, "the legacy network's output details");
  cutensornetDestroyContractionAutotunePreference(tp);
  IS(cutensornetDestroyContractionPlan(plan), 0);
  cutensornetDestroyWorkspaceDescriptor(work);
  cutensornetDestroyContractionOptimizerInfo(info);
  cutensornetDestroyContractionOptimizerConfig(cfg);
  IS(cutensornetDestroyNetworkDescriptor(net), 0);
}

// ---- decompositions ----

struct Desc {
  cutensornetTensorDescriptor_t d = nullptr;
  Desc(std::vector<int32_t> modes, std::vector<int64_t> ext, cudaDataType_t t) {
    cutensornetCreateTensorDescriptor(h, (int32_t)modes.size(), ext.data(), nullptr, modes.data(), t, &d);
  }
  ~Desc() { cutensornetDestroyTensorDescriptor(d); }
};

static void* scratch_for(cutensornetWorkspaceDescriptor_t w) {
  int64_t need = 0;
  cutensornetWorkspaceGetMemorySize(h, w, CUTENSORNET_WORKSIZE_PREF_MIN, CUTENSORNET_MEMSPACE_DEVICE,
                                    CUTENSORNET_WORKSPACE_SCRATCH, &need);
  void* p = nullptr;
  cudaMalloc(&p, (size_t)std::max<int64_t>(need, 256));
  cutensornetWorkspaceSetMemory(h, w, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, p,
                                std::max<int64_t>(need, 256));
  return p;
}

static void qr() {
  for (cudaDataType_t type : {CUDA_R_64F, CUDA_C_64F, CUDA_R_32F, CUDA_C_32F}) {
    const std::string tn = type == CUDA_R_64F ? "R64F" : type == CUDA_C_64F ? "C64F" : type == CUDA_R_32F ? "R32F" : "C32F";
    for (int wide = 0; wide < 2; ++wide) {
      // A[i,j,k] -> Q[i,x] R[x,j,k]: m = 6, n = 15 (wide) or A[i,j,k] -> Q[j,k,x] R[x,i]: m = 15, n = 6.
      std::map<int32_t, int64_t> ext = {{'i', 6}, {'j', 5}, {'k', 3}, {'x', 6}};
      T a = make({'i', 'j', 'k'}, ext, 51, cplx(type));
      std::vector<int32_t> qm = wide ? std::vector<int32_t>{'i', 'x'} : std::vector<int32_t>{'j', 'k', 'x'};
      std::vector<int32_t> rm = wide ? std::vector<int32_t>{'x', 'j', 'k'} : std::vector<int32_t>{'x', 'i'};
      std::vector<int64_t> qe, re;
      for (int32_t m : qm) qe.push_back(ext[m]);
      for (int32_t m : rm) re.push_back(ext[m]);
      Desc din({'i', 'j', 'k'}, {6, 5, 3}, type), dq(qm, qe, type), dr(rm, re, type);
      cutensornetWorkspaceDescriptor_t w = nullptr;
      cutensornetCreateWorkspaceDescriptor(h, &w);
      IS(cutensornetWorkspaceComputeQRSizes(h, din.d, dq.d, dr.d, w), 0);
      void* sp = scratch_for(w);
      void* pa = upload(a.v, type);
      size_t nq = 1, nr = 1;
      for (int64_t e : qe) nq *= (size_t)e;
      for (int64_t e : re) nr *= (size_t)e;
      void *pq = upload(std::vector<cd>(nq), type), *pr = upload(std::vector<cd>(nr), type);
      IS(cutensornetTensorQR(h, din.d, pa, dq.d, pq, dr.d, pr, w, 0), 0);
      T q{qm, qe, download(pq, nq, type)}, r{rm, re, download(pr, nr, type)};
      check(rel_err(einsum({&q, &r}, {'i', 'j', 'k'}, ext), a.v) < 10 * tol(type),
            (tn + (wide ? " wide" : " tall") + " QR: Q R = A").c_str());
      // Q^H Q = I over the shared mode.
      std::map<int32_t, int64_t> e2 = ext;
      e2['y'] = 6;
      T q2 = q;
      for (auto& m : q2.modes) m = m == 'x' ? 'y' : m;
      std::vector<cd> qq = einsum({&q, &q2}, {'x', 'y'}, e2, {true, false});
      double off = 0;
      for (int x = 0; x < 6; ++x)
        for (int y = 0; y < 6; ++y) off = std::fmax(off, std::abs(qq[(size_t)(x + 6 * y)] - (x == y ? 1.0 : 0.0)));
      check(off < 10 * tol(type), (tn + (wide ? " wide" : " tall") + " QR: Q has orthonormal columns").c_str());
      cudaFree(pa), cudaFree(pq), cudaFree(pr), cudaFree(sp);
      cutensornetDestroyWorkspaceDescriptor(w);
    }
  }
}

static void svd() {
  for (cudaDataType_t type : {CUDA_R_64F, CUDA_C_64F, CUDA_R_32F, CUDA_C_32F}) {
    const std::string tn = type == CUDA_R_64F ? "R64F" : type == CUDA_C_64F ? "C64F" : type == CUDA_R_32F ? "R32F" : "C32F";
    const bool dbl = type == CUDA_R_64F || type == CUDA_C_64F;
    std::map<int32_t, int64_t> ext = {{'i', 4}, {'j', 5}, {'k', 3}, {'x', 4}};
    T a = make({'i', 'j', 'k'}, ext, 61, cplx(type));
    // U[i,x] (m = 4) and V[x,j,k] (n = 15): exact first.
    Desc din({'i', 'j', 'k'}, {4, 5, 3}, type), du({'i', 'x'}, {4, 4}, type), dv({'x', 'j', 'k'}, {4, 5, 3}, type);
    cutensornetWorkspaceDescriptor_t w = nullptr;
    cutensornetCreateWorkspaceDescriptor(h, &w);
    IS(cutensornetWorkspaceComputeSVDSizes(h, din.d, du.d, dv.d, nullptr, w), 0);
    void* sp = scratch_for(w);
    void* pa = upload(a.v, type);
    void *pu = upload(std::vector<cd>(16), type), *pv = upload(std::vector<cd>(60), type);
    void* ps = upload(std::vector<cd>(4), dbl ? CUDA_R_64F : CUDA_R_32F);
    IS(cutensornetTensorSVD(h, din.d, pa, du.d, pu, ps, dv.d, pv, nullptr, nullptr, w, 0), 0);
    std::vector<cd> s = download(ps, 4, dbl ? CUDA_R_64F : CUDA_R_32F);
    T u{{'i', 'x'}, {4, 4}, download(pu, 16, type)}, v{{'x', 'j', 'k'}, {4, 5, 3}, download(pv, 60, type)};
    for (int x = 0; x < 4; ++x)
      for (int i = 0; i < 4; ++i) u.v[(size_t)(i + 4 * x)] *= s[(size_t)x].real();
    check(s[0].real() >= s[1].real() && s[1].real() >= s[2].real() && s[3].real() > 0,
          (tn + " SVD: singular values positive and descending").c_str());
    check(rel_err(einsum({&u, &v}, {'i', 'j', 'k'}, ext), a.v) < 10 * tol(type), (tn + " SVD: U S V = A").c_str());
    // A cutoff relative to the largest value, normalized (L2), absorbed into U.
    cutensornetTensorSVDConfig_t cfg = nullptr;
    cutensornetTensorSVDInfo_t info = nullptr;
    IS(cutensornetCreateTensorSVDConfig(h, &cfg), 0);
    IS(cutensornetCreateTensorSVDInfo(h, &info), 0);
    const double rel = (s[2].real() / s[0].real() + s[1].real() / s[0].real()) / 2;  // keeps two
    IS(cutensornetTensorSVDConfigSetAttribute(h, cfg, CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF, &rel, 8), 0);
    int32_t norm = CUTENSORNET_TENSOR_SVD_NORMALIZATION_L2, part = CUTENSORNET_TENSOR_SVD_PARTITION_US;
    IS(cutensornetTensorSVDConfigSetAttribute(h, cfg, CUTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION, &norm, 4), 0);
    IS(cutensornetTensorSVDConfigSetAttribute(h, cfg, CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION, &part, 4), 0);
    double back = 0;
    IS(cutensornetTensorSVDConfigGetAttribute(h, cfg, CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF, &back, 8), 0);
    check(back == rel, "REL_CUTOFF reads back");
    Desc du2({'i', 'x'}, {4, 4}, type), dv2({'x', 'j', 'k'}, {4, 5, 3}, type);
    IS(cutensornetTensorSVD(h, din.d, pa, du2.d, pu, nullptr, dv2.d, pv, cfg, info, w, 0), 0);
    int64_t full = 0, reduced = 0;
    double dw = -1;
    IS(cutensornetTensorSVDInfoGetAttribute(h, info, CUTENSORNET_TENSOR_SVD_INFO_FULL_EXTENT, &full, 8), 0);
    IS(cutensornetTensorSVDInfoGetAttribute(h, info, CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT, &reduced, 8), 0);
    IS(cutensornetTensorSVDInfoGetAttribute(h, info, CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT, &dw, 8), 0);
    const double total = std::norm(s[0]) + std::norm(s[1]) + std::norm(s[2]) + std::norm(s[3]);
    const double want_dw = (std::norm(s[2]) + std::norm(s[3])) / total;
    check(full == 4 && reduced == 2, (tn + " SVD: a relative cutoff keeps two of four").c_str());
    check(std::fabs(dw - want_dw) < 1e-4, (tn + " SVD: the discarded weight").c_str());
    int32_t nm = 0;
    int64_t ue[2] = {0, 0}, us[2] = {0, 0};
    IS(cutensornetGetTensorDetails(h, du2.d, &nm, nullptr, nullptr, ue, us), 0);
    check(ue[1] == 2 && us[1] == 4, (tn + " SVD: U's descriptor takes the reduced extent").c_str());
    // U holds U S / |S|, V holds V: their product is the rank-2 part, scaled.
    T u2{{'i', 'x'}, {4, 2}, download(pu, 8, type)}, v2{{'x', 'j', 'k'}, {2, 5, 3}, download(pv, 30, type)};
    std::map<int32_t, int64_t> e2 = ext;
    e2['x'] = 2;
    std::vector<cd> r2 = einsum({&u2, &v2}, {'i', 'j', 'k'}, e2);
    // The same from the exact factors.
    T ue2 = u, ve2 = v;
    ue2.ext = {4, 2};
    ue2.v.resize(8);
    ve2.ext = {2, 5, 3};
    std::vector<cd> vv(30);
    for (int c2 = 0; c2 < 15; ++c2)
      for (int x = 0; x < 2; ++x) vv[(size_t)(x + 2 * c2)] = v.v[(size_t)(x + 4 * c2)];
    ve2.v = vv;
    std::vector<cd> r2want = einsum({&ue2, &ve2}, {'i', 'j', 'k'}, e2);
    const double scale = std::sqrt(std::norm(s[0]) + std::norm(s[1]));
    for (auto& z : r2want) z /= scale;
    check(rel_err(r2, r2want) < 10 * tol(type), (tn + " SVD: truncated, L2-normalized, partitioned onto U").c_str());
    cutensornetDestroyTensorSVDConfig(cfg);
    cutensornetDestroyTensorSVDInfo(info);
    cudaFree(pa), cudaFree(pu), cudaFree(pv), cudaFree(ps), cudaFree(sp);
    cutensornetDestroyWorkspaceDescriptor(w);
  }
}

static void gate_split() {
  const cudaDataType_t type = CUDA_C_64F;
  // A[a,i,b] B[b,j,c] G[p,q,i,j] -> U[a,p,x] V[x,q,c], x up to min(2*2, 2*2) = 4... a=2, p=2: m = 4.
  std::map<int32_t, int64_t> ext = {{'a', 2}, {'i', 2}, {'b', 3}, {'j', 2}, {'c', 2}, {'p', 2}, {'q', 2}, {'x', 4}};
  T A = make({'a', 'i', 'b'}, ext, 71, true), B = make({'b', 'j', 'c'}, ext, 72, true), G = make({'p', 'q', 'i', 'j'}, ext, 73, true);
  Desc da({'a', 'i', 'b'}, {2, 2, 3}, type), db({'b', 'j', 'c'}, {3, 2, 2}, type), dg({'p', 'q', 'i', 'j'}, {2, 2, 2, 2}, type);
  Desc du({'a', 'p', 'x'}, {2, 2, 4}, type), dv({'x', 'q', 'c'}, {4, 2, 2}, type);
  cutensornetTensorSVDConfig_t cfg = nullptr;
  cutensornetCreateTensorSVDConfig(h, &cfg);
  int32_t part = CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL;
  cutensornetTensorSVDConfigSetAttribute(h, cfg, CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION, &part, 4);
  cutensornetWorkspaceDescriptor_t w = nullptr;
  cutensornetCreateWorkspaceDescriptor(h, &w);
  IS(cutensornetWorkspaceComputeGateSplitSizes(h, da.d, db.d, dg.d, du.d, dv.d, CUTENSORNET_GATE_SPLIT_ALGO_DIRECT, cfg,
                                               CUTENSORNET_COMPUTE_64F, w), 0);
  void* sp = scratch_for(w);
  void *pa = upload(A.v, type), *pb = upload(B.v, type), *pg = upload(G.v, type);
  void *pu = upload(std::vector<cd>(16), type), *pv = upload(std::vector<cd>(16), type);
  cutensornetTensorSVDInfo_t info = nullptr;
  cutensornetCreateTensorSVDInfo(h, &info);
  IS(cutensornetGateSplit(h, da.d, pa, db.d, pb, dg.d, pg, du.d, pu, nullptr, dv.d, pv, CUTENSORNET_GATE_SPLIT_ALGO_DIRECT,
                          cfg, CUTENSORNET_COMPUTE_64F, info, w, 0), 0);
  T u{{'a', 'p', 'x'}, {2, 2, 4}, download(pu, 16, type)}, v{{'x', 'q', 'c'}, {4, 2, 2}, download(pv, 16, type)};
  const std::vector<int32_t> out = {'a', 'p', 'q', 'c'};
  check(rel_err(einsum({&u, &v}, out, ext), einsum({&A, &B, &G}, out, ext)) < 1e-10,
        "gate split: U V reproduces A B G (singular values split evenly)");
  int64_t reduced = 0;
  cutensornetTensorSVDInfoGetAttribute(h, info, CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT, &reduced, 8);
  check(reduced == 4, "gate split: no truncation asked, none done");
  cutensornetDestroyTensorSVDInfo(info);
  cutensornetDestroyTensorSVDConfig(cfg);
  cutensornetDestroyWorkspaceDescriptor(w);
  cudaFree(sp);
}

// ---- the library and its argument checks ----

static void infrastructure() {
  check(cutensornetGetVersion() >= 20000, "version 2.x");
  check(std::strcmp(cutensornetGetErrorString(CUTENSORNET_STATUS_INVALID_VALUE), "CUTENSORNET_STATUS_INVALID_VALUE") == 0,
        "error string");
  cutensornetDeviceMemHandler_t mh;
  IS(cutensornetGetDeviceMemHandler(h, &mh), CUTENSORNET_STATUS_NO_DEVICE_ALLOCATOR);
  cutensornetTensorDescriptor_t d = nullptr;
  const int64_t e[] = {2, 3};
  const int32_t m[] = {'a', 'b'};
  IS(cutensornetCreateTensorDescriptor(h, 2, e, nullptr, m, CUDA_R_64F, &d), 0);
  int64_t strides[2] = {0, 0};
  IS(cutensornetTensorDescriptorGetAttribute(h, d, CUTENSORNET_TENSOR_DESCRIPTOR_ELEMENT_STRIDES, strides, 16), 0);
  check(strides[0] == 1 && strides[1] == 2, "a descriptor without strides is column-major");
  IS(cutensornetTensorDescriptorGetAttribute(h, d, CUTENSORNET_TENSOR_DESCRIPTOR_ELEMENT_STRIDES, strides, 8), CUTENSORNET_STATUS_INVALID_VALUE);
  IS(cutensornetDestroyTensorDescriptor(d), 0);
  cutensornetContractionOptimizerConfig_t cfg = nullptr;
  IS(cutensornetCreateContractionOptimizerConfig(h, &cfg), 0);
  int32_t v = 0;
  IS(cutensornetContractionOptimizerConfigGetAttribute(h, cfg, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES, &v, 4), 0);
  check(v == 1, "SLICER_MIN_SLICES defaults to 1");
  IS(cutensornetContractionOptimizerConfigGetAttribute(h, cfg, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES, &v, 4), 0);
  check(v == 0, "HYPER_NUM_SAMPLES defaults to 0");
  int64_t v8 = -1;  // a larger buffer is taken: the value is an int32_t
  IS(cutensornetContractionOptimizerConfigGetAttribute(h, cfg, CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES, &v8, 8), 0);
  check((int32_t)v8 == 1, "an int32_t attribute read into 8 bytes");
  IS(cutensornetDestroyContractionOptimizerConfig(cfg), 0);
}

int main() {
  if (cutensornetCreate(&h) != CUTENSORNET_STATUS_SUCCESS) {
    std::printf("FAIL: cutensornetCreate\n");
    return 1;
  }
  infrastructure();
  network_types();
  slicing();
  network_shapes();
  legacy();
  qr();
  svd();
  gate_split();
  cutensornetDestroy(h);
  if (failures) {
    std::printf("FAIL: %d checks failed\n", failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
