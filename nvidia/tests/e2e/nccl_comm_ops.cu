// NCCL's communicator management and the newer collectives, self-verifying, in
// the single-process form: one thread drives every rank, one device per rank.
// Every value is exact in its type, so the program passes against NVIDIA's
// libnccl on real GPUs and against VirtualGPU's alike; the error codes it
// asserts are the ones NCCL 2.29.7 returned on two RTX 3060s.
//
//   VGPU_NCCL_RANKS=4   ranks (default: every visible device); needs at least 2
//
// Covered: ncclCommSplit (color/key order, NCCL_SPLIT_NOCOLOR, config
// inheritance), non-blocking communicators (ncclInProgress, polling,
// ncclCommFinalize), ncclRedOpCreatePreMulSum with host and device scalars,
// ncclAlltoAll/ncclGather/ncclScatter, ncclCommShrink,
// ncclCommInitRankScalable and ncclCommWindowRegister. Anything newer than the
// nccl.h or the libnccl in use prints "skipped:" for that part (a "SKIP:" line
// means the whole program could not run). Against NVIDIA's library it needs
// libnccl 2.29 (2.28.9 crashes on the window argument checks); its one 2.29
// entry point, ncclWinGetUserPtr, is looked up rather than linked.
//
// The 2.29-and-later management calls: ncclCommGetUniqueId and ncclCommGrow
// (one rank joining a one-rank communicator, from two threads as the API
// needs; a second grow from three ranks up), ncclCommRevoke (including a
// stream released from a collective whose peer never came), ncclCommSuspend,
// ncclCommResume, ncclCommMemStats, ncclCommQueryProperties and the device
// API's host side; and, from 2.30, the nccl*Config forms of every collective
// and the ncclParam* registry. Their error codes are NCCL 2.31.2's, measured on
// two RTX 3060s. Both libraries behave as asserted here, and what differs
// between NCCL releases is checked against the release in use.
#include <nccl.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <unistd.h>
#include <chrono>
#include <future>
#include <string>
#include <thread>
#include <vector>
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0) && defined(__has_include)
#if __has_include(<nccl_device/core.h>)
#define HAVE_NCCL_DEVICE_HEADERS 1
// Only the host half is used. The handle types' headers mention device inline
// functions whose bodies live in the rest of nccl_device.h, which needs a newer
// CUDA than this program has to build with, hence the one suppression.
#pragma nv_diag_suppress 821
#include <nccl_device/core.h>
#include <nccl_device/impl/ll_a2a__types.h>
#include <nccl_device/impl/lsa_barrier__types.h>
#include <nccl_device/impl/gin_barrier__types.h>
#endif
#endif

static int g_bad = 0;
static int g_ver = 0;

static void expect(bool ok, const char* what) {
  if (!ok) ++g_bad;
  std::printf("%-58s %s\n", what, ok ? "ok" : "BAD");
}
static void expect_rc(ncclResult_t got, ncclResult_t want, const char* what) {
  char buf[160];
  std::snprintf(buf, sizeof buf, "%s -> %d", what, (int)got);
  if (got != want) std::printf("  (wanted %d)\n", (int)want);
  expect(got == want, buf);
}

#define NK2(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess && r_ != ncclInProgress) { \
  std::printf("%s -> %s\n", #x, ncclGetErrorString(r_)); ++g_bad; } } while (0)
#define NK(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess && r_ != ncclInProgress) { \
  std::printf("%s -> %s\n", #x, ncclGetErrorString(r_)); return 1; } } while (0)

static int nranks = 0;
static std::vector<cudaStream_t> streams;

// Poll a non-blocking communicator until its last operation is done.
static ncclResult_t settle(ncclComm_t c) {
  ncclResult_t e = ncclInProgress;
  for (long i = 0; i < 2000000000L && e == ncclInProgress; ++i)
    if (ncclCommGetAsyncError(c, &e) != ncclSuccess) return ncclInternalError;
  return e;
}

static void sync_all() {
  for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); cudaStreamSynchronize(streams[i]); }
}

template <class T> static T* upload(int dev, const std::vector<T>& h) {
  T* d = nullptr;
  cudaSetDevice(dev);
  cudaMalloc(&d, h.size() * sizeof(T) + 16);
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> static std::vector<T> download(int dev, const T* d, size_t n) {
  std::vector<T> h(n);
  cudaSetDevice(dev);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

// The old rank each member of `comms` had, gathered over the new communicator:
// the new communicator's order, as every member sees it.
static bool gathered_order(const std::vector<ncclComm_t>& comms, const std::vector<int>& members,
                           const std::vector<int>& want_order) {
  const int n = (int)members.size();
  std::vector<int*> in(n), out(n);
  for (int m = 0; m < n; ++m) {
    in[m] = upload(members[m], std::vector<int>{members[m]});
    out[m] = upload(members[m], std::vector<int>(n, -1));
  }
  ncclGroupStart();
  for (int m = 0; m < n; ++m)
    ncclAllGather(in[m], out[m], 1, ncclInt32, comms[members[m]], streams[members[m]]);
  ncclResult_t r = ncclGroupEnd();
  if (r == ncclInProgress)
    for (int m = 0; m < n; ++m) settle(comms[members[m]]);
  sync_all();
  bool ok = r == ncclSuccess || r == ncclInProgress;
  for (int m = 0; m < n; ++m) {
    std::vector<int> h = download(members[m], out[m], n);
    if (h != want_order) ok = false;
    cudaSetDevice(members[m]); cudaFree(in[m]); cudaFree(out[m]);
  }
  return ok;
}

static void destroy_all(std::vector<ncclComm_t>& comms) {
  for (int i = 0; i < nranks; ++i)
    if (comms[i]) { cudaSetDevice(i); ncclCommDestroy(comms[i]); comms[i] = nullptr; }
}

// A communicator of every rank to experiment on: one split of the world.
static bool new_child(const std::vector<ncclComm_t>& parent, std::vector<ncclComm_t>& child) {
  child.assign(nranks, nullptr);
  if (ncclGroupStart() != ncclSuccess) return false;
  for (int i = 0; i < nranks; ++i) {
    cudaSetDevice(i);
    if (ncclCommSplit(parent[i], 0, i, &child[i], nullptr) != ncclSuccess) { ncclGroupEnd(); return false; }
  }
  return ncclGroupEnd() == ncclSuccess;
}

// A communicator of every rank with its own id.
static bool fresh_comm(std::vector<ncclComm_t>& comm) {
  ncclUniqueId id;
  if (ncclGetUniqueId(&id) != ncclSuccess) return false;
  comm.assign(nranks, nullptr);
  if (ncclGroupStart() != ncclSuccess) return false;
  for (int i = 0; i < nranks; ++i) {
    cudaSetDevice(i);
    if (ncclCommInitRank(&comm[i], nranks, id, i) != ncclSuccess) { ncclGroupEnd(); return false; }
  }
  return ncclGroupEnd() == ncclSuccess;
}

// A little device memory per rank, for calls that need a buffer to point at.
static float* d_scratch(int dev) {
  static std::vector<float*> buf;
  if (buf.empty()) {
    buf.assign(nranks, nullptr);
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); cudaMalloc(&buf[i], 4096 * sizeof(float)); cudaMemset(buf[i], 0, 4096 * sizeof(float)); }
  }
  return buf[dev];
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  int ndev = 0;
  cudaGetDeviceCount(&ndev);
  nranks = ndev;
  if (const char* e = std::getenv("VGPU_NCCL_RANKS")) nranks = std::atoi(e);
  if (nranks < 2 || nranks > ndev) {
    std::printf("SKIP: needs 2 or more devices, one per rank (ranks=%d, devices=%d)\n", nranks, ndev);
    return 0;
  }
  ncclGetVersion(&g_ver);
  std::printf("nccl ranks=%d\n", nranks);
  streams.resize(nranks);
  for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); cudaStreamCreate(&streams[i]); }

  std::vector<ncclComm_t> world(nranks);
  std::vector<int> devs(nranks);
  for (int i = 0; i < nranks; ++i) devs[i] = i;
  NK(ncclCommInitAll(world.data(), nranks, devs.data()));
  std::vector<int> everyone(nranks);
  for (int i = 0; i < nranks; ++i) everyone[i] = i;

  /* ---- ncclCommSplit ---- */
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 18, 0)
  {
    // One color, keys descending: the new order is the old one reversed.
    std::vector<ncclComm_t> child(nranks, nullptr);
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclCommSplit(world[i], 7, -i, &child[i], nullptr)); }
    NK(ncclGroupEnd());
    bool ok = true;
    for (int i = 0; i < nranks; ++i) {
      int n = -1, r = -1;
      ncclCommCount(child[i], &n); ncclCommUserRank(child[i], &r);
      ok = ok && n == nranks && r == nranks - 1 - i;
    }
    expect(ok, "split: keys order the ranks (reversed)");
    std::vector<int> rev(nranks);
    for (int i = 0; i < nranks; ++i) rev[i] = nranks - 1 - i;
    expect(gathered_order(child, everyone, rev), "split: allgather over the child follows the keys");
    destroy_all(child);

    // Two colors (even and odd ranks), equal keys: old order within each.
    // Color -2 is an ordinary color; only NCCL_SPLIT_NOCOLOR (-1) is special.
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclCommSplit(world[i], i % 2 ? 5 : -2, 0, &child[i], nullptr)); }
    NK(ncclGroupEnd());
    ok = true;
    for (int i = 0; i < nranks; ++i) {
      int n = -1, r = -1;
      ncclCommCount(child[i], &n); ncclCommUserRank(child[i], &r);
      const int want_n = i % 2 ? nranks / 2 : (nranks + 1) / 2;
      ok = ok && n == want_n && r == i / 2;
    }
    expect(ok, "split: two colors, ties keep the old order");
    for (int parity = 0; parity < 2; ++parity) {
      std::vector<int> m;
      for (int i = parity; i < nranks; i += 2) m.push_back(i);
      if (!m.empty() && !gathered_order(child, m, m)) ok = false;
    }
    expect(ok, "split: each color's allgather sees only its members");
    destroy_all(child);

    // The last rank opts out.
    for (auto& c : child) c = reinterpret_cast<ncclComm_t>(0x1);
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) {
      cudaSetDevice(i);
      NK(ncclCommSplit(world[i], i == nranks - 1 ? NCCL_SPLIT_NOCOLOR : 0, i, &child[i], nullptr));
    }
    NK(ncclGroupEnd());
    ok = child[nranks - 1] == nullptr;
    for (int i = 0; i + 1 < nranks; ++i) { int n = -1; ncclCommCount(child[i], &n); ok = ok && n == nranks - 1; }
    expect(ok, "split: NCCL_SPLIT_NOCOLOR gets a NULL communicator");
    destroy_all(child);

    cudaSetDevice(0);
    expect_rc(ncclCommSplit(world[0], 0, 0, nullptr, nullptr), ncclInvalidArgument, "split: NULL newcomm");
    ncclComm_t x = nullptr;
    expect_rc(ncclCommSplit(nullptr, 0, 0, &x, nullptr), ncclInvalidArgument, "split: NULL comm");
    // A config NCCL_CONFIG_INITIALIZER did not make fails the call and the group.
    ncclConfig_t bad = NCCL_CONFIG_INITIALIZER;
    bad.magic = 0;
    NK(ncclGroupStart());
    ok = true;
    for (int i = 0; i < nranks; ++i) {
      cudaSetDevice(i);
      child[i] = reinterpret_cast<ncclComm_t>(0x1);
      ok = ok && ncclCommSplit(world[i], 0, i, &child[i], &bad) == ncclInvalidArgument && child[i] == nullptr;
    }
    expect(ok, "split: uninitialized config is ncclInvalidArgument, newcomm NULL");
    expect_rc(ncclGroupEnd(), ncclInvalidArgument, "split: ...and so is the ncclGroupEnd");
  }
#else
  std::printf("skipped: ncclCommSplit needs nccl.h 2.18\n");
#endif

  /* ---- non-blocking communicators ---- */
  {
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
    cfg.blocking = 0;
    std::vector<ncclComm_t> nb(nranks, nullptr);
    // One thread, no group: a blocking init would deadlock here.
    bool ok = true;
    for (int i = 0; i < nranks; ++i) {
      cudaSetDevice(i);
      ok = ok && ncclCommInitRankConfig(&nb[i], nranks, id, i, &cfg) == ncclInProgress && nb[i];
    }
    expect(ok, "nonblocking: init returns ncclInProgress");
    for (int i = 0; i < nranks; ++i) ok = ok && settle(nb[i]) == ncclSuccess;
    expect(ok, "nonblocking: init completes once every rank joined");

    std::vector<float*> buf(nranks);
    for (int i = 0; i < nranks; ++i) buf[i] = upload(i, std::vector<float>(64, 1.0f + i));
    ok = true;
    for (int i = 0; i < nranks; ++i) {
      cudaSetDevice(i);
      ok = ok && ncclAllReduce(buf[i], buf[i], 64, ncclFloat, ncclSum, nb[i], streams[i]) == ncclInProgress;
    }
    expect(ok, "nonblocking: ungrouped allreduce returns ncclInProgress");
    for (int i = 0; i < nranks; ++i) ok = ok && settle(nb[i]) == ncclSuccess;
    sync_all();
    const float sum = nranks * (nranks + 1) / 2.0f;
    for (int i = 0; i < nranks; ++i) ok = ok && download(i, buf[i], 64) == std::vector<float>(64, sum);
    expect(ok, "nonblocking: ...and completes with the right sum");

    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) NK(ncclAllReduce(buf[i], buf[i], 64, ncclFloat, ncclSum, nb[i], streams[i]));
    expect_rc(ncclGroupEnd(), ncclInProgress, "nonblocking: grouped, ncclGroupEnd");
    ok = true;
    for (int i = 0; i < nranks; ++i) ok = ok && settle(nb[i]) == ncclSuccess;
    sync_all();
    for (int i = 0; i < nranks; ++i) ok = ok && download(i, buf[i], 64) == std::vector<float>(64, sum * nranks);
    expect(ok, "nonblocking: grouped allreduce result");

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 18, 0)
    // A NULL config inherits the parent's: the child is non-blocking too. The
    // split itself returns ncclSuccess and fills newcomm once the parent is done.
    std::vector<ncclComm_t> kid(nranks, nullptr);
    ok = true;
    for (int i = 0; i < nranks; ++i) {
      cudaSetDevice(i);
      ok = ok && ncclCommSplit(nb[i], 0, i, &kid[i], nullptr) == ncclSuccess;
    }
    expect(ok, "nonblocking: ungrouped split returns ncclSuccess");
    for (int i = 0; i < nranks; ++i) ok = ok && settle(nb[i]) == ncclSuccess && kid[i] != nullptr;
    expect(ok, "nonblocking: newcomm is set when the parent settles");
    for (int i = 0; i < nranks; ++i) {
      cudaSetDevice(i);
      ok = ok && ncclAllReduce(buf[i], buf[i], 64, ncclFloat, ncclMax, kid[i], streams[i]) == ncclInProgress;
    }
    for (int i = 0; i < nranks; ++i) ok = ok && settle(kid[i]) == ncclSuccess;
    expect(ok, "nonblocking: the child inherited blocking=0");
    destroy_all(kid);

    // An explicit config wins over inheritance.
    ncclConfig_t blk = NCCL_CONFIG_INITIALIZER;
    blk.blocking = 1;
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclCommSplit(nb[i], 0, i, &kid[i], &blk)); }
    expect_rc(ncclGroupEnd(), ncclInProgress, "nonblocking: grouped split of a non-blocking parent");
    ok = true;
    for (int i = 0; i < nranks; ++i) ok = ok && settle(nb[i]) == ncclSuccess && kid[i] != nullptr;
    for (int i = 0; i < nranks; ++i) {
      ncclResult_t e = ncclInProgress;
      ncclCommGetAsyncError(kid[i], &e);
      ok = ok && e == ncclSuccess;
    }
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) NK(ncclAllReduce(buf[i], buf[i], 64, ncclFloat, ncclMax, kid[i], streams[i]));
    expect_rc(ncclGroupEnd(), ncclSuccess, "nonblocking: blocking child's group returns ncclSuccess");
    destroy_all(kid);
#endif

    ok = true;
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); ok = ok && ncclCommFinalize(nb[i]) == ncclInProgress; }
    expect(ok, "nonblocking: finalize returns ncclInProgress");
    for (int i = 0; i < nranks; ++i) ok = ok && settle(nb[i]) == ncclSuccess;
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); ok = ok && ncclCommDestroy(nb[i]) == ncclSuccess; cudaFree(buf[i]); }
    expect(ok, "nonblocking: settles, then destroys");

    // Used before its init finished (the other ranks have not joined): the
    // call fails, and the communicator keeps that error.
    ncclGetUniqueId(&id);
    cudaSetDevice(0);
    ncclComm_t early = nullptr;
    NK(ncclCommInitRankConfig(&early, nranks, id, 0, &cfg));
    int n = -1;
    expect_rc(ncclCommCount(early, &n), ncclInvalidArgument, "nonblocking: ncclCommCount before init completes");
    ncclResult_t e = ncclSuccess;
    ncclCommGetAsyncError(early, &e);
    std::printf("  (state after the early ncclCommCount: %d)\n", (int)e);
    float* scratch = upload(0, std::vector<float>(4, 0.0f));
    expect_rc(ncclAllReduce(scratch, scratch, 4, ncclFloat, ncclSum, early, streams[0]), ncclInvalidArgument,
              "nonblocking: a collective before init completes");
    ncclCommGetAsyncError(early, &e);
    expect(e == ncclInvalidArgument, "nonblocking: ...leaves ncclInvalidArgument as its state");
    cudaFree(scratch);
    ncclCommAbort(early);

    // Config errors.
    ncclComm_t x = reinterpret_cast<ncclComm_t>(0x1);
    ncclConfig_t b5 = NCCL_CONFIG_INITIALIZER;
    b5.blocking = 5;
    ncclGetUniqueId(&id);
    expect_rc(ncclCommInitRankConfig(&x, 1, id, 0, &b5), ncclInvalidArgument, "config: blocking=5");
    expect(x == nullptr, "config: ...and the comm is NULL");
    expect_rc(ncclCommInitRankConfig(&x, 1, id, 3, &cfg), ncclInvalidArgument, "config: rank 3 of 1");
  }

  /* ---- blocking lifecycle ---- */
  {
    ncclResult_t e = ncclInProgress;
    ncclCommGetAsyncError(world[0], &e);
    expect(e == ncclSuccess, "blocking: async error is ncclSuccess");
    expect_rc(ncclCommGetAsyncError(nullptr, &e), ncclInvalidArgument, "blocking: GetAsyncError(NULL)");
    expect_rc(ncclCommDestroy(nullptr), ncclSuccess, "blocking: Destroy(NULL)");
    expect_rc(ncclCommAbort(nullptr), ncclSuccess, "blocking: Abort(NULL)");
    expect_rc(ncclCommFinalize(nullptr), ncclSuccess, "blocking: Finalize(NULL)");
  }

  /* ---- pre-multiplied sums ---- */
  {
    const size_t N = 8;
    // Integers times small scalars: exact in float, and the sum is exact too.
    auto x = [](int r, size_t k) { return (float)(k + 1) + 2.0f * r; };
    auto sc = [](int r) { return r % 2 ? 0.5f : 2.0f + r; };
    std::vector<float*> s(nranks), d(nranks), dsc(nranks);
    std::vector<ncclRedOp_t> op(nranks);
    std::vector<float> hsc(nranks);
    bool ok = true;
    for (int i = 0; i < nranks; ++i) {
      std::vector<float> h(N);
      for (size_t k = 0; k < N; ++k) h[k] = x(i, k);
      s[i] = upload(i, h);
      d[i] = upload(i, std::vector<float>(N, 0.0f));
      hsc[i] = sc(i);
      cudaSetDevice(i);
      ok = ok && ncclRedOpCreatePreMulSum(&op[i], &hsc[i], ncclFloat, ncclScalarHostImmediate, world[i]) == ncclSuccess;
      hsc[i] = 1000.0f;   // read at creation: this must not matter
    }
    expect(ok, "premulsum: create (host scalar)");
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) NK(ncclAllReduce(s[i], d[i], N, ncclFloat, op[i], world[i], streams[i]));
    NK(ncclGroupEnd());
    sync_all();
    std::vector<float> want(N, 0.0f);
    for (size_t k = 0; k < N; ++k) for (int r = 0; r < nranks; ++r) want[k] += x(r, k) * sc(r);
    for (int i = 0; i < nranks; ++i) ok = ok && download(i, d[i], N) == want;
    expect(ok, "premulsum: allreduce, host scalars read at creation");

    // A device scalar is read when the collective runs, not at creation.
    std::vector<ncclRedOp_t> dop(nranks);
    for (int i = 0; i < nranks; ++i) {
      dsc[i] = upload(i, std::vector<float>{-1.0f});
      cudaSetDevice(i);
      ok = ok && ncclRedOpCreatePreMulSum(&dop[i], dsc[i], ncclFloat, ncclScalarDevice, world[i]) == ncclSuccess;
      const float later = 0.25f * (i + 1);
      cudaMemcpy(dsc[i], &later, sizeof later, cudaMemcpyHostToDevice);
    }
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) NK(ncclAllReduce(s[i], d[i], N, ncclFloat, dop[i], world[i], streams[i]));
    NK(ncclGroupEnd());
    sync_all();
    std::fill(want.begin(), want.end(), 0.0f);
    for (size_t k = 0; k < N; ++k) for (int r = 0; r < nranks; ++r) want[k] += x(r, k) * 0.25f * (r + 1);
    ok = true;
    for (int i = 0; i < nranks; ++i) ok = ok && download(i, d[i], N) == want;
    expect(ok, "premulsum: allreduce, device scalars read at run time");

    // Reduce to the last rank and reduce-scatter take the same operator.
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); cudaMemset(d[i], 0, N * sizeof(float)); }
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) NK(ncclReduce(s[i], d[i], N, ncclFloat, dop[i], nranks - 1, world[i], streams[i]));
    NK(ncclGroupEnd());
    sync_all();
    ok = download(nranks - 1, d[nranks - 1], N) == want && download(0, d[0], N) == std::vector<float>(N, 0.0f);
    expect(ok, "premulsum: reduce (root only)");
    std::vector<float*> big(nranks), part(nranks);
    for (int i = 0; i < nranks; ++i) {
      std::vector<float> h(N * nranks);
      for (size_t k = 0; k < h.size(); ++k) h[k] = x(i, k);
      big[i] = upload(i, h);
      part[i] = upload(i, std::vector<float>(N, 0.0f));
    }
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) NK(ncclReduceScatter(big[i], part[i], N, ncclFloat, dop[i], world[i], streams[i]));
    NK(ncclGroupEnd());
    sync_all();
    ok = true;
    for (int i = 0; i < nranks; ++i) {
      std::vector<float> w(N, 0.0f);
      for (size_t k = 0; k < N; ++k) for (int r = 0; r < nranks; ++r) w[k] += x(r, i * N + k) * 0.25f * (r + 1);
      ok = ok && download(i, part[i], N) == w;
    }
    expect(ok, "premulsum: reduce-scatter");

    // Integer arithmetic wraps (card: 3 * 1e9 + (1e9 + 1) * -2 = 999999998).
    {
      std::vector<int*> si(nranks), di(nranks);
      std::vector<ncclRedOp_t> iop(nranks);
      std::vector<int> isc(nranks);
      for (int i = 0; i < nranks; ++i) {
        si[i] = upload(i, std::vector<int>{1 + i, 1000000000 + i, 2147483647, -7});
        di[i] = upload(i, std::vector<int>(4, 0));
        isc[i] = i % 2 ? -2 : 3;
        cudaSetDevice(i);
        NK(ncclRedOpCreatePreMulSum(&iop[i], &isc[i], ncclInt32, ncclScalarHostImmediate, world[i]));
      }
      NK(ncclGroupStart());
      for (int i = 0; i < nranks; ++i) NK(ncclAllReduce(si[i], di[i], 4, ncclInt32, iop[i], world[i], streams[i]));
      NK(ncclGroupEnd());
      sync_all();
      std::vector<int> w(4, 0);
      for (int r = 0; r < nranks; ++r) {
        const unsigned m = r % 2 ? (unsigned)-2 : 3u;
        const unsigned v[4] = {(unsigned)(1 + r), (unsigned)(1000000000 + r), 2147483647u, (unsigned)-7};
        for (int k = 0; k < 4; ++k) w[k] = (int)((unsigned)w[k] + v[k] * m);
      }
      ok = true;
      for (int i = 0; i < nranks; ++i) ok = ok && download(i, di[i], 4) == w;
      expect(ok, "premulsum: int32 wraps");
      for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); ncclRedOpDestroy(iop[i], world[i]); cudaFree(si[i]); cudaFree(di[i]); }
    }
    // Half: each product is rounded to half before the sum (exact values here).
    {
      std::vector<__half*> sh(nranks), dh(nranks);
      std::vector<ncclRedOp_t> hop(nranks);
      std::vector<__half> hs(nranks);
      for (int i = 0; i < nranks; ++i) {
        std::vector<__half> h(N);
        for (size_t k = 0; k < N; ++k) h[k] = __float2half(x(i, k));
        sh[i] = upload(i, h);
        dh[i] = upload(i, std::vector<__half>(N, __float2half(0.0f)));
        hs[i] = __float2half(sc(i));
        cudaSetDevice(i);
        NK(ncclRedOpCreatePreMulSum(&hop[i], &hs[i], ncclHalf, ncclScalarHostImmediate, world[i]));
      }
      NK(ncclGroupStart());
      for (int i = 0; i < nranks; ++i) NK(ncclAllReduce(sh[i], dh[i], N, ncclHalf, hop[i], world[i], streams[i]));
      NK(ncclGroupEnd());
      sync_all();
      std::vector<float> w(N, 0.0f);
      for (size_t k = 0; k < N; ++k) for (int r = 0; r < nranks; ++r) w[k] += x(r, k) * sc(r);
      ok = true;
      for (int i = 0; i < nranks; ++i) {
        std::vector<__half> got = download(i, dh[i], N);
        for (size_t k = 0; k < N; ++k) ok = ok && __half2float(got[k]) == w[k];
      }
      expect(ok, "premulsum: half");
      for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); ncclRedOpDestroy(hop[i], world[i]); cudaFree(sh[i]); cudaFree(dh[i]); }
    }

    // Error codes (card: NCCL 2.29.7).
    cudaSetDevice(0);
    expect_rc(ncclAllReduce(s[0], d[0], N, ncclInt32, op[0], world[0], streams[0]), ncclInvalidArgument,
              "premulsum: used with another datatype");
    expect_rc(ncclRedOpDestroy(ncclSum, world[0]), ncclInvalidArgument, "premulsum: destroy a builtin");
    expect_rc(ncclRedOpDestroy(op[0], nullptr), ncclInvalidArgument, "premulsum: destroy, NULL comm");
    expect_rc(ncclRedOpDestroy(op[0], world[1]), ncclInvalidArgument, "premulsum: destroy on another communicator");
    expect_rc(ncclRedOpDestroy(op[0], world[0]), ncclSuccess, "premulsum: destroy");
    expect_rc(ncclRedOpDestroy(op[0], world[0]), ncclInvalidArgument, "premulsum: destroy twice");
    expect_rc(ncclAllReduce(s[0], d[0], N, ncclFloat, op[0], world[0], streams[0]), ncclInvalidArgument,
              "premulsum: used after destroy");
    ncclRedOp_t again;
    float one = 1.0f;
    expect_rc(ncclRedOpCreatePreMulSum(&again, &one, ncclFloat, ncclScalarHostImmediate, nullptr),
              ncclInvalidArgument, "premulsum: create, NULL comm");
    expect_rc(ncclRedOpCreatePreMulSum(&again, &one, (ncclDataType_t)99, ncclScalarHostImmediate, world[0]),
              ncclInternalError, "premulsum: create, invalid datatype");
    expect_rc(ncclRedOpCreatePreMulSum(&again, &one, ncclFloat, ncclScalarHostImmediate, world[0]),
              ncclSuccess, "premulsum: create after destroy");
    ncclRedOpDestroy(again, world[0]);
    for (int i = 0; i < nranks; ++i) {
      cudaSetDevice(i);
      if (i) ncclRedOpDestroy(op[i], world[i]);
      ncclRedOpDestroy(dop[i], world[i]);
      cudaFree(s[i]); cudaFree(d[i]); cudaFree(dsc[i]); cudaFree(big[i]); cudaFree(part[i]);
    }
  }

  /* ---- all-to-all, gather, scatter ---- */
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
  if (g_ver >= NCCL_VERSION(2, 28, 0)) {
    const size_t C = 3;
    auto v = [](int r, size_t k) { return 100.0f * r + (float)k; };
    std::vector<float*> s(nranks), d(nranks);
    for (int i = 0; i < nranks; ++i) {
      std::vector<float> h(C * nranks);
      for (size_t k = 0; k < h.size(); ++k) h[k] = v(i, k);
      s[i] = upload(i, h);
      d[i] = upload(i, std::vector<float>(C * nranks, -1.0f));
    }
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) NK(ncclAlltoAll(s[i], d[i], C, ncclFloat, world[i], streams[i]));
    NK(ncclGroupEnd());
    sync_all();
    bool ok = true;
    for (int j = 0; j < nranks; ++j) {
      std::vector<float> got = download(j, d[j], C * nranks);
      for (int i = 0; i < nranks; ++i)
        for (size_t k = 0; k < C; ++k) ok = ok && got[i * C + k] == v(i, j * C + k);
    }
    expect(ok, "alltoall: block j of rank i lands as block i of rank j");
    // nccl.h documents no in-place all-to-all (unlike gather and scatter), and
    // on the card one is a race: it came out right twice and wrong once.
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) NK(ncclAlltoAll(s[i], d[i], 0, ncclFloat, world[i], streams[i]));
    expect_rc(ncclGroupEnd(), ncclSuccess, "alltoall: count 0");
    // Zero elements everywhere else too, a pre-multiplied sum and an empty
    // message included: each synchronizes and moves nothing. (Added after the
    // RTX 3060 runs, while their GPU 0 was reserved; NCCL documents count 0.)
    {
      std::vector<ncclRedOp_t> zop(nranks);
      float one = 1.0f;
      for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclRedOpCreatePreMulSum(&zop[i], &one, ncclFloat, ncclScalarHostImmediate, world[i])); }
      NK(ncclGroupStart());
      for (int i = 0; i < nranks; ++i) {
        NK(ncclAllGather(s[i], d[i], 0, ncclFloat, world[i], streams[i]));
        NK(ncclAllReduce(s[i], d[i], 0, ncclFloat, zop[i], world[i], streams[i]));
        NK(ncclGather(s[i], d[i], 0, ncclFloat, 0, world[i], streams[i]));
        NK(ncclSend(s[i], 0, ncclFloat, (i + 1) % nranks, world[i], streams[i]));
        NK(ncclRecv(d[i], 0, ncclFloat, (i + nranks - 1) % nranks, world[i], streams[i]));
      }
      expect_rc(ncclGroupEnd(), ncclSuccess, "count 0: allgather, premulsum, gather, send/recv");
      for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); ncclRedOpDestroy(zop[i], world[i]); }
    }
    cudaSetDevice(0);
    expect_rc(ncclAlltoAll(s[0], d[0], C, (ncclDataType_t)99, world[0], streams[0]), ncclInvalidArgument,
              "alltoall: invalid datatype");
    expect_rc(ncclAlltoAll(s[0], d[0], C, ncclFloat, nullptr, streams[0]), ncclInvalidArgument,
              "alltoall: NULL comm");

    // Gather to the last rank; the others' buffers are left alone.
    const int root = nranks - 1;
    for (int i = 0; i < nranks; ++i) {
      std::vector<float> h(C * nranks);
      for (size_t k = 0; k < h.size(); ++k) h[k] = v(i, k);
      cudaSetDevice(i);
      cudaMemcpy(s[i], h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemset(d[i], 0, C * nranks * sizeof(float));
    }
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) NK(ncclGather(s[i], d[i], C, ncclFloat, root, world[i], streams[i]));
    NK(ncclGroupEnd());
    sync_all();
    ok = true;
    std::vector<float> got = download(root, d[root], C * nranks);
    for (int i = 0; i < nranks; ++i) for (size_t k = 0; k < C; ++k) ok = ok && got[i * C + k] == v(i, k);
    ok = ok && download(0, d[0], C * nranks) == std::vector<float>(C * nranks, 0.0f);
    expect(ok, "gather: the root gets every rank's block");
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); cudaMemset(d[i], 0, C * nranks * sizeof(float)); }
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) NK(ncclScatter(s[i], d[i], C, ncclFloat, root, world[i], streams[i]));
    NK(ncclGroupEnd());
    sync_all();
    ok = true;
    for (int i = 0; i < nranks; ++i) {
      std::vector<float> g = download(i, d[i], C);
      for (size_t k = 0; k < C; ++k) ok = ok && g[k] == v(root, i * C + k);
    }
    expect(ok, "scatter: rank i gets the root's block i");
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); cudaFree(s[i]); cudaFree(d[i]); }
  } else {
    std::printf("skipped: ncclAlltoAll/Gather/Scatter need libnccl 2.28 (this is %d)\n", g_ver);
  }
#else
  std::printf("skipped: ncclAlltoAll needs nccl.h 2.28\n");
#endif

  /* ---- ncclCommShrink ---- */
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
  if (g_ver >= NCCL_VERSION(2, 27, 0)) {
    // The last rank leaves; only the survivors call.
    const int gone = nranks - 1;
    int ex[1] = {gone};
    std::vector<ncclComm_t> small(nranks, nullptr);
    NK(ncclGroupStart());
    for (int i = 0; i < gone; ++i) { cudaSetDevice(i); NK(ncclCommShrink(world[i], ex, 1, &small[i], nullptr, NCCL_SHRINK_DEFAULT)); }
    NK(ncclGroupEnd());
    bool ok = true;
    for (int i = 0; i < gone; ++i) {
      int n = -1, r = -1;
      ncclCommCount(small[i], &n); ncclCommUserRank(small[i], &r);
      ok = ok && n == nranks - 1 && r == i;
    }
    expect(ok, "shrink: survivors keep their order");
    std::vector<int> m(gone);
    for (int i = 0; i < gone; ++i) m[i] = i;
    expect(gathered_order(small, m, m), "shrink: allgather over the survivors");
    destroy_all(small);
    if (nranks > 2) {
      // Rank 0 leaves: everyone moves down one.
      int ex0[1] = {0};
      NK(ncclGroupStart());
      for (int i = 1; i < nranks; ++i) { cudaSetDevice(i); NK(ncclCommShrink(world[i], ex0, 1, &small[i], nullptr, NCCL_SHRINK_DEFAULT)); }
      NK(ncclGroupEnd());
      ok = true;
      for (int i = 1; i < nranks; ++i) { int r = -1; ncclCommUserRank(small[i], &r); ok = ok && r == i - 1; }
      expect(ok, "shrink: excluding rank 0 renumbers from 0");
      destroy_all(small);
    }
    cudaSetDevice(0);
    ncclComm_t x = reinterpret_cast<ncclComm_t>(0x1);
    int self[1] = {0};
    expect_rc(ncclCommShrink(world[0], self, 1, &x, nullptr, 0), ncclInvalidArgument, "shrink: excluding yourself");
    expect_rc(ncclCommShrink(world[0], ex, 0, &x, nullptr, 0), ncclInvalidArgument, "shrink: count 0");
    expect_rc(ncclCommShrink(world[0], nullptr, 1, &x, nullptr, 0), ncclInvalidArgument, "shrink: NULL list");
    expect_rc(ncclCommShrink(world[0], ex, 1, nullptr, nullptr, 0), ncclInvalidArgument, "shrink: NULL newcomm");
  } else {
    std::printf("skipped: ncclCommShrink needs libnccl 2.27 (this is %d)\n", g_ver);
  }
#else
  std::printf("skipped: ncclCommShrink needs nccl.h 2.27\n");
#endif

  /* ---- ncclCommInitRankScalable ---- */
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 23, 0)
  if (g_ver >= NCCL_VERSION(2, 23, 0)) {
    std::vector<ncclUniqueId> ids(2);
    ncclGetUniqueId(&ids[0]);
    ncclGetUniqueId(&ids[1]);
    std::vector<ncclComm_t> sc(nranks, nullptr);
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclCommInitRankScalable(&sc[i], nranks, i, 2, ids.data(), nullptr)); }
    NK(ncclGroupEnd());
    bool ok = true;
    for (int i = 0; i < nranks; ++i) { int n = -1, r = -1; ncclCommCount(sc[i], &n); ncclCommUserRank(sc[i], &r); ok = ok && n == nranks && r == i; }
    expect(ok, "scalable: two ids, every rank joins one communicator");
    expect(gathered_order(sc, everyone, everyone), "scalable: allgather");
    destroy_all(sc);
    ncclGetUniqueId(&ids[0]);
    ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
    cfg.blocking = 0;
    ok = true;
    for (int i = 0; i < nranks; ++i) {
      cudaSetDevice(i);
      ok = ok && ncclCommInitRankScalable(&sc[i], nranks, i, 1, ids.data(), &cfg) == ncclInProgress;
    }
    for (int i = 0; i < nranks; ++i) ok = ok && settle(sc[i]) == ncclSuccess;
    expect(ok, "scalable: non-blocking returns ncclInProgress, then settles");
    destroy_all(sc);
    ncclComm_t x = reinterpret_cast<ncclComm_t>(0x1);
    cudaSetDevice(0);
    expect_rc(ncclCommInitRankScalable(&x, 1, 3, 1, ids.data(), nullptr), ncclInvalidArgument, "scalable: rank 3 of 1");
    expect(x == nullptr, "scalable: ...and the comm is NULL");
  } else {
    std::printf("skipped: ncclCommInitRankScalable needs libnccl 2.23 (this is %d)\n", g_ver);
  }
#else
  std::printf("skipped: ncclCommInitRankScalable needs nccl.h 2.23\n");
#endif

  /* ---- symmetric memory windows ---- */
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
  if (g_ver >= NCCL_VERSION(2, 27, 0)) {
    // Without peer mappings (the RTX 3060 pair, VirtualGPU) registration
    // succeeds with a NULL window; with them it is a real one. Either way a
    // collective on the buffer works.
    std::vector<void*> mem(nranks);
    std::vector<ncclWindow_t> win(nranks, nullptr);
    bool ok = true;
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); ok = ok && ncclMemAlloc(&mem[i], 1 << 20) == ncclSuccess; }
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) {
      cudaSetDevice(i);
      ok = ok && ncclCommWindowRegister(world[i], mem[i], 1 << 20, &win[i], NCCL_WIN_COLL_SYMMETRIC) == ncclSuccess;
    }
    NK(ncclGroupEnd());
    expect(ok, "window: register an ncclMemAlloc buffer");
    for (int i = 0; i < nranks; ++i) {
      std::vector<float> h(4, 1.0f + i);
      cudaSetDevice(i);
      cudaMemcpy(mem[i], h.data(), 16, cudaMemcpyHostToDevice);
    }
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i)
      NK(ncclAllReduce(mem[i], (char*)mem[i] + 4096, 4, ncclFloat, ncclSum, world[i], streams[i]));
    NK(ncclGroupEnd());
    sync_all();
    const float sum = nranks * (nranks + 1) / 2.0f;
    for (int i = 0; i < nranks; ++i)
      ok = ok && download(i, (const float*)((char*)mem[i] + 4096), 4) == std::vector<float>(4, sum);
    expect(ok, "window: allreduce on the registered buffer");
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
    using GetUserPtr = ncclResult_t (*)(ncclComm_t, ncclWindow_t, void**);
    if (auto get = (GetUserPtr)dlsym(RTLD_DEFAULT, "ncclWinGetUserPtr")) {
      void* user = nullptr;
      // card: 2.29.7 refuses a NULL window; 2.31.2 and 2.32.3 answer ncclSuccess
      // and NULL. A communicator without windows has no user pointer to give.
      const ncclResult_t rc = get(world[0], win[0], &user);
      ok = win[0] ? rc == ncclSuccess && user == mem[0]
                  : g_ver >= NCCL_VERSION(2, 31, 0) ? rc == ncclSuccess && user == nullptr
                  : g_ver < NCCL_VERSION(2, 30, 0) ? rc == ncclInvalidArgument
                  : (rc == ncclInvalidArgument || (rc == ncclSuccess && user == nullptr));
      expect(ok, win[0] ? "window: user pointer" : "window: NULL window has no user pointer");
    } else {
      std::printf("skipped: ncclWinGetUserPtr needs libnccl 2.29\n");
    }
#endif
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclCommWindowDeregister(world[i], win[i])); }
    NK(ncclGroupEnd());
    cudaSetDevice(0);
    ncclWindow_t w = nullptr;
    expect_rc(ncclCommWindowRegister(world[0], mem[0], 1 << 20, nullptr, 0), ncclInvalidArgument, "window: NULL win");
    expect_rc(ncclCommWindowRegister(nullptr, mem[0], 1 << 20, &w, 0), ncclInvalidArgument, "window: NULL comm");
    expect_rc(ncclCommWindowDeregister(world[0], nullptr), ncclSuccess, "window: deregister NULL");
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); ncclMemFree(mem[i]); }
  } else {
    std::printf("skipped: windows need libnccl 2.27 (this is %d)\n", g_ver);
  }
#else
  std::printf("skipped: windows need nccl.h 2.27\n");
#endif

  /* ---- unique ids, memory statistics, suspend and resume, revoke ---- */
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
  if (g_ver >= NCCL_VERSION(2, 29, 0)) {
    // card (NCCL 2.31.2): ncclCommGetUniqueId gives a fresh id per call.
    ncclUniqueId a, b;
    NK(ncclCommGetUniqueId(world[0], &a));
    NK(ncclCommGetUniqueId(world[1], &b));
    expect(std::memcmp(&a, &b, sizeof a) != 0, "uniqueid: every call returns a different id");
    expect_rc(ncclCommGetUniqueId(nullptr, &a), ncclInvalidArgument, "uniqueid: NULL comm");
    expect_rc(ncclCommGetUniqueId(world[0], nullptr), ncclInvalidArgument, "uniqueid: NULL id");

    std::vector<ncclComm_t> ch;
    if (!new_child(world, ch)) { std::printf("cannot split a child to experiment on\n"); return 1; }

    // Statistics: four of them, and a sum that adds up. Sizes differ (NCCL
    // holds memory for the communicator, this transport none), so only the
    // relations are asserted.
    {
      uint64_t v[4] = {9, 9, 9, 9};
      bool ok = true;
      for (int k = 0; k < 4; ++k) ok = ok && ncclCommMemStats(ch[0], (ncclCommMemStat_t)k, &v[k]) == ncclSuccess;
      expect(ok && v[ncclStatGpuMemSuspended] == 0 &&
             v[ncclStatGpuMemSuspend] + v[ncclStatGpuMemPersist] == v[ncclStatGpuMemTotal],
             "memstats: suspendable + persistent = total, not suspended");
      uint64_t x = 5;
      // The statistic is an enum, and these values are no member of it: they go
      // through a pointer that takes an int, which is the same call at the ABI.
      using MemStatsInt = ncclResult_t (*)(ncclComm_t, int, uint64_t*);
      if (auto stats = (MemStatsInt)dlsym(RTLD_DEFAULT, "ncclCommMemStats")) {
        expect_rc(stats(ch[0], 4, &x), ncclInvalidArgument, "memstats: statistic 4");
        expect_rc(stats(ch[0], -1, &x), ncclInvalidArgument, "memstats: statistic -1");
      }
      expect_rc(ncclCommMemStats(ch[0], ncclStatGpuMemTotal, nullptr), ncclInvalidArgument, "memstats: NULL value");
      expect_rc(ncclCommMemStats(nullptr, ncclStatGpuMemTotal, &x), ncclInvalidArgument, "memstats: NULL comm");
    }

    // card: suspend and resume are collective; with one thread they go in a
    // group. Bit 0 suspends, anything else is a successful no-op; suspending
    // twice or resuming what is running is ncclInvalidUsage; a suspended
    // communicator still answers queries and splits.
    // NCCL 2.29.7 (RTX 3060 pair) hangs when one thread suspends or resumes the
    // ranks in a group, which 2.31.2 accepts (2.30 was not measured), so this part
    // runs from 2.30.
    if (g_ver >= NCCL_VERSION(2, 30, 0)) {
      auto suspended = [&](int i) { uint64_t x = 9; ncclCommMemStats(ch[i], ncclStatGpuMemSuspended, &x); return x; };
      NK(ncclGroupStart());
      for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclCommSuspend(ch[i], 0)); }
      NK(ncclGroupEnd());
      expect(suspended(0) == 0 && suspended(nranks - 1) == 0, "suspend: flags 0 do nothing");
      NK(ncclGroupStart());
      for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclCommSuspend(ch[i], NCCL_SUSPEND_MEM)); }
      NK(ncclGroupEnd());
      expect(suspended(0) == 1 && suspended(nranks - 1) == 1, "suspend: NCCL_SUSPEND_MEM suspends every rank");
      cudaSetDevice(0);
      expect_rc(ncclCommSuspend(ch[0], NCCL_SUSPEND_MEM), ncclInvalidUsage, "suspend: twice");
      {
        int n = -1;
        expect(ncclCommCount(ch[0], &n) == ncclSuccess && n == nranks, "suspend: still answers ncclCommCount");
        ncclUniqueId u;
        expect_rc(ncclCommGetUniqueId(ch[0], &u), ncclSuccess, "suspend: still gives unique ids");
      }
      NK(ncclGroupStart());
      for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclCommResume(ch[i])); }
      NK(ncclGroupEnd());
      expect(suspended(0) == 0 && suspended(nranks - 1) == 0, "resume: back to running");
      cudaSetDevice(0);
      expect_rc(ncclCommResume(ch[0]), ncclInvalidUsage, "resume: not suspended");
      expect_rc(ncclCommSuspend(nullptr, NCCL_SUSPEND_MEM), ncclInvalidArgument, "suspend: NULL comm");
      expect_rc(ncclCommResume(nullptr), ncclInvalidArgument, "resume: NULL comm");
      expect(gathered_order(ch, everyone, everyone), "resume: the communicator works again");
    } else {
      std::printf("skipped: suspending and resuming in a group needs libnccl 2.30 (this is %d)\n", g_ver);
    }
    destroy_all(ch);

    // card: revoke is local. It refuses new work with ncclInvalidUsage, can be
    // done once (ncclInvalidArgument after), takes only flag 0, makes
    // ncclCommFinalize invalid, accepts NULL, and leaves splitting and
    // destroying open.
    expect_rc(ncclCommRevoke(nullptr, 0), ncclSuccess, "revoke: NULL comm");
    if (!new_child(world, ch)) return 1;
    cudaSetDevice(0);
    expect_rc(ncclCommRevoke(ch[0], 1), ncclInvalidArgument, "revoke: flag 1");
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclCommRevoke(ch[i], 0)); }
    {
      ncclResult_t e = ncclInProgress;
      for (int i = 0; i < nranks; ++i) { ncclCommGetAsyncError(ch[i], &e); if (e != ncclSuccess) break; }
      expect(e == ncclSuccess, "revoke: an idle communicator is quiescent at once");
    }
    cudaSetDevice(0);
    expect_rc(ncclAllReduce(d_scratch(0), d_scratch(0), 4, ncclFloat, ncclSum, ch[0], streams[0]),
              ncclInvalidUsage, "revoke: a collective is refused");
    expect_rc(ncclSend(d_scratch(0), 4, ncclFloat, 1, ch[0], streams[0]), ncclInvalidUsage, "revoke: so is a send");
    expect_rc(ncclCommRevoke(ch[0], 0), ncclInvalidArgument, "revoke: only once");
    expect_rc(ncclCommFinalize(ch[0]), ncclInvalidArgument, "revoke: finalize is invalid");
    {
      std::vector<ncclComm_t> again;
      expect(new_child(ch, again) && gathered_order(again, everyone, everyone),
             "revoke: a revoked communicator still splits, and the child works");
      destroy_all(again);
    }
    destroy_all(ch);

    // A collective whose peers never join blocks a stream; revoke releases it.
    // The call that issues it runs in its own thread: NCCL queues it and the
    // wait is on the stream, here the call itself waits, and both are the same
    // thread's problem. Rank 0 alone issues an all-reduce; after 300 ms it is
    // revoked from this thread.
    // A communicator of its own, not a split of the world: a split shares its
    // parent's proxy, and NCCL's parent then hangs in ncclCommDestroy on the
    // operation that was torn down.
    if (!fresh_comm(ch)) return 1;
    // NCCL connects its transports on a communicator's first collective, which
    // needs every rank; revoking in the middle of that is an error of its own
    // (ncclInternalError), so the communicator is used once first.
    expect(gathered_order(ch, everyone, everyone), "revoke: a fresh communicator, connected");
    {
      auto stuck = std::async(std::launch::async, [&] {
        cudaSetDevice(0);
        const ncclResult_t r = ncclAllReduce(d_scratch(0), d_scratch(0), 4, ncclFloat, ncclSum, ch[0], streams[0]);
        const cudaError_t c = cudaStreamSynchronize(streams[0]);
        if (r != ncclSuccess || c != cudaSuccess)
          std::printf("revoke: the issuing call gave %d, the stream %d (%s)\n", (int)r, (int)c, cudaGetErrorString(c));
        return r == ncclSuccess && c == cudaSuccess;
      });
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
      cudaSetDevice(0);
      const ncclResult_t r = ncclCommRevoke(ch[0], 0);
      if (stuck.wait_for(std::chrono::seconds(60)) != std::future_status::ready) {
        std::printf("revoke: the stream is still stuck after 60 s\nRESULT: FAIL\n");
        std::fflush(stdout);
        std::_Exit(1);
      }
      expect(r == ncclSuccess && stuck.get(), "revoke: releases a collective waiting for peers");
    }
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); ncclCommRevoke(ch[i], 0); }   // rank 0's already is: the error is expected
    destroy_all(ch);

    // Non-blocking: ncclInProgress, then settles.
    {
      ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
      cfg.blocking = 0;
      std::vector<ncclComm_t> nb(nranks, nullptr);
      NK(ncclGroupStart());
      for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclCommSplit(world[i], 0, i, &nb[i], &cfg)); }
      NK(ncclGroupEnd());
      bool ok = true;
      for (int i = 0; i < nranks; ++i) ok = ok && settle(nb[i]) == ncclSuccess;
      for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); ok = ok && ncclCommRevoke(nb[i], 0) == ncclInProgress; }
      for (int i = 0; i < nranks; ++i) ok = ok && settle(nb[i]) == ncclSuccess;
      expect(ok, "revoke: non-blocking returns ncclInProgress, then settles");
      destroy_all(nb);
    }

    /* ---- ncclCommGrow ---- */
    // card: a one-rank communicator and a rank that is new to it become two.
    // The calls come from two threads, the existing rank first.
    {
      std::vector<ncclComm_t> one(nranks, nullptr);
      NK(ncclGroupStart());
      for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclCommSplit(world[i], i, 0, &one[i], nullptr)); }
      NK(ncclGroupEnd());
      ncclUniqueId id;
      cudaSetDevice(0);
      NK(ncclCommGetUniqueId(one[0], &id));
      ncclComm_t g0 = nullptr, g1 = nullptr;
      ncclResult_t r0 = ncclInternalError, r1 = ncclInternalError;
      std::thread existing([&] { cudaSetDevice(0); r0 = ncclCommGrow(one[0], 2, &id, -1, &g0, nullptr); });
      std::thread joiner([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));   // NCCL wants the existing rank first
        cudaSetDevice(1);
        r1 = ncclCommGrow(nullptr, 2, &id, 1, &g1, nullptr);
      });
      existing.join();
      joiner.join();
      expect(r0 == ncclSuccess && r1 == ncclSuccess && g0 && g1, "grow: one rank + one new rank");
      int n0 = -1, k0 = -1, n1 = -1, k1 = -1, dv = -1;
      if (g0 && g1) {
        ncclCommCount(g0, &n0); ncclCommUserRank(g0, &k0);
        ncclCommCount(g1, &n1); ncclCommUserRank(g1, &k1); ncclCommCuDevice(g1, &dv);
      }
      expect(n0 == 2 && n1 == 2 && k0 == 0 && k1 == 1 && dv == 1, "grow: the existing rank keeps 0, the new one is 1, on its own device");
      std::vector<ncclComm_t> two(nranks, nullptr);
      two[0] = g0; two[1] = g1;
      expect(g0 && g1 && gathered_order(two, std::vector<int>{0, 1}, std::vector<int>{0, 1}), "grow: the grown communicator works");
      // The old one is still a one-rank communicator.
      int n = -1;
      cudaSetDevice(0);
      expect(ncclCommCount(one[0], &n) == ncclSuccess && n == 1, "grow: the parent is untouched");

      // Errors, all local. card: nRanks no larger than the communicator, a rank
      // given by an existing rank, a rank outside 0..nRanks-1 or no id for a
      // new rank, and a NULL newcomm.
      ncclUniqueId id2;
      NK(ncclCommGetUniqueId(one[0], &id2));
      ncclComm_t x = reinterpret_cast<ncclComm_t>(0x1);
      expect_rc(ncclCommGrow(one[0], 1, &id2, -1, &x, nullptr), ncclInvalidArgument, "grow: nRanks equal to the size");
      expect_rc(ncclCommGrow(one[0], 2, &id2, 3, &x, nullptr), ncclInvalidArgument, "grow: an existing rank names a rank");
      expect_rc(ncclCommGrow(one[0], 2, &id2, -1, nullptr, nullptr), ncclInvalidArgument, "grow: NULL newcomm");
      expect_rc(ncclCommGrow(nullptr, 2, nullptr, 1, &x, nullptr), ncclInvalidArgument, "grow: a new rank without an id");
      expect_rc(ncclCommGrow(nullptr, 2, &id2, 5, &x, nullptr), ncclInvalidArgument, "grow: a new rank outside the size");
      expect_rc(ncclCommGrow(nullptr, 2, &id2, -1, &x, nullptr), ncclInvalidArgument, "grow: a new rank of -1");
      if (g0 && g1) { two[0] = g0; two[1] = g1; for (int i = 0; i < 2; ++i) { cudaSetDevice(i); ncclCommDestroy(two[i]); } }

      if (nranks >= 3 && g0 && g1) {
        // Three ranks, which the two-GPU card cannot do: the grown pair takes a
        // third. The root passes the id, the other existing rank passes NULL.
        // Rebuild the pair, since it was just destroyed.
        ncclUniqueId i1;
        NK(ncclCommGetUniqueId(one[0], &i1));
        ncclComm_t a0 = nullptr, a1 = nullptr, h0 = nullptr, h1 = nullptr, h2 = nullptr;
        ncclUniqueId j;
        // a second one-rank communicator on device 1 to be the pair's other half
        std::thread e0([&] { cudaSetDevice(0); ncclCommGrow(one[0], 2, &i1, -1, &a0, nullptr); });
        std::thread e1([&] { std::this_thread::sleep_for(std::chrono::milliseconds(500)); cudaSetDevice(1); ncclCommGrow(nullptr, 2, &i1, 1, &a1, nullptr); });
        e0.join(); e1.join();
        cudaSetDevice(0);
        NK(ncclCommGetUniqueId(a0, &j));
        ncclResult_t q0 = ncclInternalError, q1 = ncclInternalError, q2 = ncclInternalError;
        std::thread t0([&] { cudaSetDevice(0); q0 = ncclCommGrow(a0, 3, &j, -1, &h0, nullptr); });
        std::thread t1([&] { cudaSetDevice(1); q1 = ncclCommGrow(a1, 3, nullptr, -1, &h1, nullptr); });
        std::thread t2([&] { std::this_thread::sleep_for(std::chrono::milliseconds(500)); cudaSetDevice(2); q2 = ncclCommGrow(nullptr, 3, &j, 2, &h2, nullptr); });
        t0.join(); t1.join(); t2.join();
        expect(q0 == ncclSuccess && q1 == ncclSuccess && q2 == ncclSuccess, "grow: three ranks, the non-root form");
        std::vector<ncclComm_t> three(nranks, nullptr);
        three[0] = h0; three[1] = h1; three[2] = h2;
        expect(h0 && h1 && h2 && gathered_order(three, std::vector<int>{0, 1, 2}, std::vector<int>{0, 1, 2}), "grow: all three agree");
        for (int i = 0; i < 3; ++i) if (three[i]) { cudaSetDevice(i); ncclCommDestroy(three[i]); }
        cudaSetDevice(0); ncclCommDestroy(a0);
        cudaSetDevice(1); ncclCommDestroy(a1);
      }
      destroy_all(one);
    }
  } else {
    std::printf("skipped: unique ids, grow, revoke, suspend and memory statistics need libnccl 2.29 (this is %d)\n", g_ver);
  }
#else
  std::printf("skipped: unique ids, grow, revoke, suspend and memory statistics need nccl.h 2.29\n");
#endif

  /* ---- the device API's host side ---- */
#if defined(HAVE_NCCL_DEVICE_HEADERS)
  // NCCL refuses a properties struct built by a newer nccl.h than itself.
  if (g_ver >= NCCL_VERSION_CODE) {
    // No communicator of this transport supports the device API; the RTX 3060
    // pair does not either. Where one does, only the refusal is skipped.
    ncclCommProperties_t pr = NCCL_COMM_PROPERTIES_INITIALIZER;
    cudaSetDevice(1);
    const ncclResult_t rq = ncclCommQueryProperties(world[1], &pr);
    expect(rq == ncclSuccess && pr.rank == 1 && pr.nRanks == nranks && pr.cudaDev == 1
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
               && pr.devCommRuntimeVersionSize > 0  // nccl.h 2.29's struct stops before this
#endif
               ,
           "properties: rank, size, device");
    ncclCommProperties_t pr0 = NCCL_COMM_PROPERTIES_INITIALIZER;
    cudaSetDevice(0);
    ncclCommQueryProperties(world[0], &pr0);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
    expect(pr0.commHash == pr.commHash, "properties: every rank sees one communicator hash");
#endif
    ncclCommProperties_t junk;
    std::memset(&junk, 0, sizeof junk);
    expect_rc(ncclCommQueryProperties(world[0], &junk), ncclInvalidUsage, "properties: not initialised");
    expect_rc(ncclCommQueryProperties(world[0], nullptr), ncclInvalidArgument, "properties: NULL");
    expect_rc(ncclCommQueryProperties(nullptr, &pr), ncclInvalidArgument, "properties: NULL comm");
    if (!pr.deviceApiSupport) {
      ncclDevCommRequirements_t req = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      char buf[1024];
      std::memset(buf, 0, sizeof buf);
      expect_rc(ncclDevCommCreate(world[0], &req, reinterpret_cast<ncclDevComm_t*>(buf)), ncclInvalidUsage,
                "devcomm: refused where the device API is unsupported");
      expect_rc(ncclDevCommCreate(nullptr, &req, reinterpret_cast<ncclDevComm_t*>(buf)), ncclInvalidArgument, "devcomm: NULL comm");
      expect_rc(ncclDevCommCreate(world[0], nullptr, reinterpret_cast<ncclDevComm_t*>(buf)), ncclInvalidArgument, "devcomm: NULL requirements");
      ncclDevCommRequirements_t zero;
      std::memset(&zero, 0, sizeof zero);
      expect_rc(ncclDevCommCreate(world[0], &zero, reinterpret_cast<ncclDevComm_t*>(buf)), ncclInvalidUsage, "devcomm: uninitialised requirements");
      expect_rc(ncclDevCommDestroy(world[0], reinterpret_cast<ncclDevComm_t*>(buf)), ncclSuccess, "devcomm: destroying nothing");
      expect_rc(ncclDevCommDestroy(nullptr, reinterpret_cast<ncclDevComm_t*>(buf)), ncclInvalidArgument, "devcomm: destroy NULL comm");
      void* p = nullptr;
      expect_rc(ncclGetLsaDevicePointer(nullptr, 0, 0, &p), ncclInvalidArgument, "devcomm: LSA pointer of a NULL window");
      expect_rc(ncclGetPeerDevicePointer(nullptr, 0, 0, &p), ncclInvalidArgument, "devcomm: peer pointer of a NULL window");
      expect_rc(ncclGetLsaMultimemDevicePointer(nullptr, 0, &p), ncclInvalidArgument, "devcomm: multimem pointer of a NULL window");
    } else {
      std::printf("skipped: ncclDevCommCreate is supported here\n");
    }
    // One-sided operations need host RMA, which the properties say is absent
    // here; NCCL then fails all three with ncclInvalidArgument.
    if (!pr.hostRmaSupport) {
      cudaSetDevice(0);
      ncclWaitSignalDesc_t wd = {1, 1, 0, 0};
      expect_rc(ncclSignal(1, 0, 0, 0, world[0], streams[0]), ncclInvalidArgument, "rma: signal, without host RMA");
      expect_rc(ncclWaitSignal(1, &wd, world[0], streams[0]), ncclInvalidArgument, "rma: wait for a signal, without host RMA");
      expect_rc(ncclPutSignal(d_scratch(0), 4, ncclFloat, 1, nullptr, 0, 0, 0, 0, world[0], streams[0]), ncclInvalidArgument,
                "rma: put, without host RMA");
    } else {
      std::printf("skipped: host RMA is supported here\n");
    }
    // Teams. The world team is the communicator; ranks translate by stride.
    cudaSetDevice(1);
    const ncclTeam_t tw = ncclTeamWorld(world[1]);
    expect(tw.nRanks == nranks && tw.rank == 1 && tw.stride == 1, "team: world");
    expect(ncclTeamRankToWorld(world[1], tw, 0) == 0 && ncclTeamRankToWorld(world[1], tw, nranks - 1) == nranks - 1,
           "team: world rank to world rank");
    const ncclTeam_t tl = ncclTeamLsa(world[1]);
    expect(tl.nRanks >= 1 && tl.nRanks <= nranks && tl.stride == 1, "team: lsa is part of the world");
    const ncclTeam_t tr = ncclTeamRail(world[1]);
    expect(tr.nRanks >= 1 && tr.nRanks * tl.nRanks <= nranks, "team: rail times lsa fits the world");
    // The requirement helpers are host arithmetic, with the same answers on the
    // card: a barrier's buffer is nBarriers * (12 + 4 * nRanks) bytes.
    {
      ncclLsaBarrierHandle_t h;
      std::memset(&h, 0xcd, sizeof h);
      ncclDevResourceRequirements_t rq2;
      ncclTeam_t t4{4, 1, 1};
      const ncclResult_t r = ncclLsaBarrierCreateRequirement(t4, 2, &h, &rq2);
      expect(r == ncclSuccess && rq2.bufferSize == 56 && rq2.bufferAlign == 4 && rq2.outBufferHandle == &h.bufHandle &&
             h.nBarriers == 2 && rq2.next == nullptr, "requirement: lsa barrier of 4 ranks x 2 = 56 bytes");
      ncclLLA2AHandle_t lh;
      std::memset(&lh, 0xcd, sizeof lh);
      const ncclResult_t r2 = ncclLLA2ACreateRequirement(2, 8, &lh, &rq2);
      expect(r2 == ncclSuccess && rq2.bufferSize == 544 && rq2.bufferAlign == 16 && lh.nSlots == 8,
             "requirement: ll all-to-all of 2 blocks x 8 slots = 544 bytes");
      expect(ncclLLA2ACalcSlots(4, 100) == 52 && ncclLLA2ACalcSlots(4, 8) == 4 && ncclLLA2ACalcSlots(3, 16) == 6,
             "requirement: slots = elements * ceil(size / 8)");
      ncclGinBarrierHandle_t gh;
      std::memset(&gh, 0xcd, sizeof gh);
      ncclDevResourceRequirements_t rq3;
      ncclTeam_t t2{2, 0, 1};
      const ncclResult_t r3 = ncclGinBarrierCreateRequirement(world[0], t2, 4, &gh, &rq3);
      expect(r3 == ncclSuccess && rq3.ginSignalCount == 8 && rq3.bufferSize == 0 && rq3.outGinSignalStart == &gh.signal0,
             "requirement: gin barrier of 2 ranks x 4 = 8 signals");
    }
  }
#endif

  /* ---- per-collective configuration: nccl*Config ---- */
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
  if (g_ver >= NCCL_VERSION(2, 30, 0)) {
    std::vector<float*> sb(nranks), rb(nranks);
    const size_t C = 4;
    for (int i = 0; i < nranks; ++i) {
      cudaSetDevice(i);
      cudaMalloc(&sb[i], C * nranks * sizeof(float));
      cudaMalloc(&rb[i], C * nranks * sizeof(float));
      std::vector<float> h(C * nranks);
      for (size_t k = 0; k < h.size(); ++k) h[k] = 10.0f * (i + 1) + (float)k;
      cudaMemcpy(sb[i], h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    auto v = [&](int rank, size_t k) { return 10.0f * (rank + 1) + (float)k; };
    auto clear = [&] { for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); cudaMemset(rb[i], 0, C * nranks * sizeof(float)); } };
    ncclCollConfig_t good = NCCL_COLLCONFIG_INITIALIZER;
    bool ok;

    clear();
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclAllReduceConfig(sb[i], rb[i], C, ncclFloat, ncclSum, world[i], streams[i], &good)); }
    NK(ncclGroupEnd());
    sync_all();
    ok = true;
    for (int i = 0; i < nranks; ++i) {
      std::vector<float> g = download(i, rb[i], C);
      for (size_t k = 0; k < C; ++k) { float want = 0; for (int r = 0; r < nranks; ++r) want += v(r, k); ok = ok && g[k] == want; }
    }
    expect(ok, "config: AllReduce");
    clear();
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclBroadcastConfig(sb[i], rb[i], C, ncclFloat, 1, world[i], streams[i], &good)); }
    NK(ncclGroupEnd());
    sync_all();
    ok = true;
    for (int i = 0; i < nranks; ++i) { std::vector<float> g = download(i, rb[i], C); for (size_t k = 0; k < C; ++k) ok = ok && g[k] == v(1, k); }
    expect(ok, "config: Broadcast");
    clear();
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclReduceConfig(sb[i], rb[i], C, ncclFloat, ncclSum, 0, world[i], streams[i], &good)); }
    NK(ncclGroupEnd());
    sync_all();
    {
      std::vector<float> g = download(0, rb[0], C);
      ok = true;
      for (size_t k = 0; k < C; ++k) { float want = 0; for (int r = 0; r < nranks; ++r) want += v(r, k); ok = ok && g[k] == want; }
      expect(ok, "config: Reduce");
    }
    clear();
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclAllGatherConfig(sb[i], rb[i], C, ncclFloat, world[i], streams[i], &good)); }
    NK(ncclGroupEnd());
    sync_all();
    ok = true;
    for (int i = 0; i < nranks; ++i) { std::vector<float> g = download(i, rb[i], C * nranks); for (int r = 0; r < nranks; ++r) for (size_t k = 0; k < C; ++k) ok = ok && g[r * C + k] == v(r, k); }
    expect(ok, "config: AllGather");
    clear();
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclReduceScatterConfig(sb[i], rb[i], C, ncclFloat, ncclSum, world[i], streams[i], &good)); }
    NK(ncclGroupEnd());
    sync_all();
    ok = true;
    for (int i = 0; i < nranks; ++i) { std::vector<float> g = download(i, rb[i], C); for (size_t k = 0; k < C; ++k) { float want = 0; for (int r = 0; r < nranks; ++r) want += v(r, i * C + k); ok = ok && g[k] == want; } }
    expect(ok, "config: ReduceScatter");
    clear();
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclAlltoAllConfig(sb[i], rb[i], C, ncclFloat, world[i], streams[i], &good)); }
    NK(ncclGroupEnd());
    sync_all();
    ok = true;
    for (int i = 0; i < nranks; ++i) { std::vector<float> g = download(i, rb[i], C * nranks); for (int r = 0; r < nranks; ++r) for (size_t k = 0; k < C; ++k) ok = ok && g[r * C + k] == v(r, i * C + k); }
    expect(ok, "config: AlltoAll");
    clear();
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclGatherConfig(sb[i], rb[i], C, ncclFloat, 1, world[i], streams[i], &good)); }
    NK(ncclGroupEnd());
    sync_all();
    {
      std::vector<float> g = download(1, rb[1], C * nranks);
      ok = true;
      for (int r = 0; r < nranks; ++r) for (size_t k = 0; k < C; ++k) ok = ok && g[r * C + k] == v(r, k);
      expect(ok, "config: Gather");
    }
    clear();
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclScatterConfig(sb[i], rb[i], C, ncclFloat, 0, world[i], streams[i], &good)); }
    NK(ncclGroupEnd());
    sync_all();
    ok = true;
    for (int i = 0; i < nranks; ++i) { std::vector<float> g = download(i, rb[i], C); for (size_t k = 0; k < C; ++k) ok = ok && g[k] == v(0, i * C + k); }
    expect(ok, "config: Scatter");
    clear();
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclAllReduceConfig(sb[i], rb[i], C, ncclFloat, ncclSum, world[i], streams[i], nullptr)); }
    NK(ncclGroupEnd());
    sync_all();
    ok = download(0, rb[0], 1)[0] != 0.0f;
    expect(ok, "config: NULL is the plain collective");

    // card: argument errors come back from the call, before anything is queued
    // and before the communicator is looked at; the size is checked first, then
    // the magic, the forced-algorithm flag, the CTA policy and the algorithm
    // selection. These configs are rejected on rank 0 alone.
    cudaSetDevice(0);
    auto bad = [&](const char* what, const ncclCollConfig_t& cfg, ncclResult_t want, int collective = 0) {
      ncclResult_t r;
      switch (collective) {
        case 1: r = ncclBroadcastConfig(sb[0], rb[0], C, ncclFloat, 0, world[0], streams[0], &cfg); break;
        case 2: r = ncclAlltoAllConfig(sb[0], rb[0], C, ncclFloat, world[0], streams[0], &cfg); break;
        default: r = ncclAllReduceConfig(sb[0], rb[0], C, ncclFloat, ncclSum, world[0], streams[0], &cfg); break;
      }
      expect_rc(r, want, what);
    };
    ncclCollConfig_t c = NCCL_COLLCONFIG_INITIALIZER;
    c.magic = 0x1234;
    bad("config: wrong magic", c, ncclInvalidArgument);
    c = NCCL_COLLCONFIG_INITIALIZER; c.size = 8;
    bad("config: size 8", c, ncclInvalidArgument);
    c = NCCL_COLLCONFIG_INITIALIZER; c.size = 56;
    bad("config: size 56 (smaller than 2.31's struct)", c, ncclInvalidArgument);
    c = NCCL_COLLCONFIG_INITIALIZER; c.forceAlgSelection = 2;
    bad("config: forceAlgSelection 2", c, ncclInvalidArgument);
    c = NCCL_COLLCONFIG_INITIALIZER; c.forceAlgSelection = -1;
    bad("config: forceAlgSelection -1", c, ncclInvalidArgument);
    c = NCCL_COLLCONFIG_INITIALIZER; c.CTAPolicy = 4;
    bad("config: CTAPolicy 4", c, ncclInvalidArgument);
    c = NCCL_COLLCONFIG_INITIALIZER; c.CTAPolicy = -5;
    bad("config: CTAPolicy -5", c, ncclInvalidArgument);
    c = NCCL_COLLCONFIG_INITIALIZER; c.magic = 0x1234;
    expect_rc(ncclAllReduceConfig(sb[0], rb[0], C, ncclFloat, ncclSum, nullptr, streams[0], &c), ncclInvalidArgument,
              "config: a bad config with a NULL communicator");
    // Accepted whatever they hold: sizes of CTAs, a newer struct, any profiler tag, the extension list.
    clear();
    ncclCollConfig_t odd = NCCL_COLLCONFIG_INITIALIZER;
    odd.minCTAs = 4; odd.maxCTAs = 2; odd.nvlsCTAs = -4; odd.cgaClusterSize = 100; odd.CTAPolicy = NCCL_CTA_POLICY_ZERO;
    odd.userProfilerTag = 0x8000000000000001ull; odd.version = 99999; odd.size += 8;
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclAllReduceConfig(sb[i], rb[i], C, ncclFloat, ncclSum, world[i], streams[i], &odd)); }
    NK(ncclGroupEnd());
    sync_all();
    expect(download(0, rb[0], 1)[0] != 0.0f, "config: unvalidated fields do not matter");

    // Algorithm selection. AllReduce has ring and tree here; Broadcast, Reduce,
    // AllGather and ReduceScatter ring; AlltoAll, Gather and Scatter nothing to
    // choose. With forceAlgSelection = 1 (the default) the impossible is an
    // error; with 0 it falls back. A selection is a comma-separated list of
    // names in any case, "^" in front of a name meaning every other one.
    cudaSetDevice(0);
    auto alg = [](const char* sel, int force) {
      ncclCollConfig_t k = NCCL_COLLCONFIG_INITIALIZER;
      k.algSelection = sel;
      k.forceAlgSelection = force;
      return k;
    };
    bad("config: AllReduce, nvls (no NVLS here)", alg("nvls", 1), ncclInvalidArgument, 0);
    bad("config: AllReduce, pat", alg("pat", 1), ncclInvalidArgument, 0);
    bad("config: AllReduce, collnetdirect", alg("collnetdirect", 1), ncclInvalidArgument, 0);
    bad("config: AllReduce, an unknown name", alg("bogus", 1), ncclInvalidArgument, 0);
    bad("config: AllReduce, ring plus an unknown name", alg("ring,bogus", 1), ncclInvalidArgument, 0);
    bad("config: AllReduce, a bare ^", alg("^", 1), ncclInvalidArgument, 0);
    bad("config: AllReduce, ring;tree", alg("ring;tree", 1), ncclInvalidArgument, 0);
    bad("config: Broadcast, tree (ring only)", alg("tree", 1), ncclInvalidArgument, 1);
    bad("config: Broadcast, everything but ring", alg("^ring", 1), ncclInvalidArgument, 1);
    bad("config: AlltoAll, even ring", alg("ring", 1), ncclInvalidArgument, 2);
    // What is accepted, issued by every rank.
    auto accepted = [&](const char* what, const ncclCollConfig_t& cfg, int collective) {
      clear();
      bool ok2 = true;
      NK2(ncclGroupStart());
      for (int i = 0; i < nranks; ++i) {
        cudaSetDevice(i);
        ncclResult_t r;
        switch (collective) {
          case 1: r = ncclBroadcastConfig(sb[i], rb[i], C, ncclFloat, 0, world[i], streams[i], &cfg); break;
          case 2: r = ncclAlltoAllConfig(sb[i], rb[i], C, ncclFloat, world[i], streams[i], &cfg); break;
          default: r = ncclAllReduceConfig(sb[i], rb[i], C, ncclFloat, ncclSum, world[i], streams[i], &cfg); break;
        }
        ok2 = ok2 && r == ncclSuccess;
      }
      NK2(ncclGroupEnd());
      sync_all();
      ok2 = ok2 && download(0, rb[0], 1)[0] != 0.0f;
      expect(ok2, what);
    };
    accepted("config: AllReduce, ring", alg("ring", 1), 0);
    accepted("config: AllReduce, tree", alg("tree", 1), 0);
    accepted("config: AllReduce, RING,Tree", alg("RING,Tree", 1), 0);
    accepted("config: AllReduce, everything but ring (tree is left)", alg("^ring", 1), 0);
    accepted("config: AllReduce, blanks and empty items", alg(" ring , ,", 1), 0);
    accepted("config: AllReduce, nvls with a fall back", alg("nvls", 0), 0);
    accepted("config: AllReduce, an unknown name with a fall back", alg("bogus", 0), 0);
    accepted("config: Broadcast, ring", alg("ring", 1), 1);
    accepted("config: Broadcast, everything but tree", alg("^tree", 1), 1);
    accepted("config: AlltoAll, an empty selection", alg("", 1), 2);
    accepted("config: AlltoAll, ring with a fall back", alg("ring", 0), 2);

    // card: a refused config does not spoil its group; the rest of it runs.
    clear();
    {
      ncclCollConfig_t spoiled = NCCL_COLLCONFIG_INITIALIZER;
      spoiled.magic = 7;
      ok = true;
      NK(ncclGroupStart());
      for (int i = 0; i < nranks; ++i) {
        cudaSetDevice(i);
        if (i == 0) ok = ok && ncclAllReduceConfig(sb[i], rb[i], C, ncclFloat, ncclSum, world[i], streams[i], &spoiled) == ncclInvalidArgument;
        NK(ncclAllReduce(sb[i], rb[i], C, ncclFloat, ncclSum, world[i], streams[i]));
      }
      ok = ok && ncclGroupEnd() == ncclSuccess;
      sync_all();
      expect(ok && download(0, rb[0], 1)[0] != 0.0f, "config: a refused config leaves its group alone");
    }
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); cudaFree(sb[i]); cudaFree(rb[i]); }
  } else {
    std::printf("skipped: the nccl*Config forms need libnccl 2.30 (this is %d)\n", g_ver);
  }
#else
  std::printf("skipped: the nccl*Config forms need nccl.h 2.30\n");
#endif


  /* ---- the ncclParam registry ---- */
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
  if (g_ver >= NCCL_VERSION(2, 30, 0)) {
    // card (NCCL 2.31.2): six public parameters, all about debug output; three
    // more are private and show only with NCCL_PARAM_DUMP_ALL. Values asserted
    // against their defaults only where the environment leaves them alone.
    const char** keys = nullptr;
    int nk = -1;
    expect(ncclParamGetAllParameterKeys(&keys, &nk) == ncclSuccess && nk >= 6, "param: the public keys");
    const char* want_keys[] = {"NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_DEBUG_FILE", "NCCL_DEBUG_TIMESTAMP_FORMAT",
                               "NCCL_DEBUG_TIMESTAMP_LEVELS", "NCCL_SET_THREAD_NAME"};
    bool all_listed = keys != nullptr;
    for (const char* w : want_keys) {
      int hits = 0;
      for (int i = 0; keys && i < nk; ++i) hits += std::strcmp(keys[i], w) == 0;
      all_listed = all_listed && hits == 1;
    }
    expect(all_listed, "param: every public key is listed exactly once");
    expect_rc(ncclParamGetAllParameterKeys(nullptr, &nk), ncclInvalidArgument, "param: NULL table");

    ncclParamHandle_t h = nullptr, hs = nullptr, hu32 = nullptr, hu64 = nullptr, hb = nullptr, hd = nullptr;
    expect(ncclParamBind(&hs, "NCCL_DEBUG_TIMESTAMP_FORMAT") == ncclSuccess && ncclParamBind(&hu32, "NCCL_DEBUG_TIMESTAMP_LEVELS") == ncclSuccess &&
           ncclParamBind(&hu64, "NCCL_DEBUG_SUBSYS") == ncclSuccess && ncclParamBind(&hb, "NCCL_SET_THREAD_NAME") == ncclSuccess &&
           ncclParamBind(&hd, "NCCL_DEBUG") == ncclSuccess, "param: bind");
    ncclParamHandle_t again = nullptr;
    ncclParamBind(&again, "NCCL_DEBUG_SUBSYS");
    expect(again == hu64, "param: a key binds to the same handle");
    h = reinterpret_cast<ncclParamHandle_t>(0x1);
    expect_rc(ncclParamBind(&h, "NCCL_NO_SUCH_PARAMETER"), ncclInvalidArgument, "param: unknown key");
    expect(h == reinterpret_cast<ncclParamHandle_t>(0x1), "param: ...leaves the handle alone");
    expect_rc(ncclParamBind(&h, "nccl_debug"), ncclInvalidArgument, "param: keys are case-sensitive");
    expect_rc(ncclParamBind(&h, nullptr), ncclInvalidArgument, "param: NULL key");
    expect_rc(ncclParamBind(nullptr, "NCCL_DEBUG"), ncclInvalidArgument, "param: NULL handle out");

    // A getter takes the parameter's own type.
    const char* str = nullptr;
    uint32_t u32 = 0;
    uint64_t u64 = 0;
    int32_t i32 = 0;
    expect_rc(ncclParamGetStr(hs, &str), ncclSuccess, "param: string parameter, GetStr");
    if (!std::getenv("NCCL_DEBUG_TIMESTAMP_FORMAT")) expect(str && std::strcmp(str, "[%F %T] ") == 0, "param: ...its default");
    expect_rc(ncclParamGetU32(hs, &u32), ncclInvalidArgument, "param: string parameter, GetU32");
    expect_rc(ncclParamGetU32(hu32, &u32), ncclSuccess, "param: uint32 parameter, GetU32");
    if (!std::getenv("NCCL_DEBUG_TIMESTAMP_LEVELS")) expect(u32 == 0x4, "param: ...its default (WARN)");
    expect_rc(ncclParamGetStr(hu32, &str), ncclInvalidArgument, "param: uint32 parameter, GetStr");
    expect_rc(ncclParamGetU64(hu64, &u64), ncclSuccess, "param: uint64 parameter, GetU64");
    if (!std::getenv("NCCL_DEBUG_SUBSYS")) expect(u64 == 0x1081, "param: ...its default (INIT,ENV,BOOTSTRAP)");
    expect_rc(ncclParamGetU32(hu64, &u32), ncclInvalidArgument, "param: uint64 parameter, GetU32");
    expect_rc(ncclParamGetI32(hb, &i32), ncclInvalidArgument, "param: boolean parameter has no integer getter");
    expect_rc(ncclParamGetI32(hd, &i32), ncclInvalidArgument, "param: the debug level has none either");
    expect_rc(ncclParamGetU64(nullptr, &u64), ncclInvalidArgument, "param: NULL handle");
    expect_rc(ncclParamGetU64(hu64, nullptr), ncclInvalidArgument, "param: NULL out");

    // Raw bytes: a boolean one, a uint64 eight, a string with its NUL.
    unsigned char raw[64];
    int len = -7;
    expect(ncclParamGet(hb, raw, sizeof raw, &len) == ncclSuccess && len == 1, "param: raw boolean is one byte");
    if (!std::getenv("NCCL_SET_THREAD_NAME")) expect(raw[0] == 0, "param: ...false");
    expect(ncclParamGet(hu64, raw, sizeof raw, &len) == ncclSuccess && len == 8, "param: raw uint64 is eight bytes");
    len = -7;
    expect(ncclParamGet(hs, raw, sizeof raw, &len) == ncclSuccess && len >= 1 && raw[len - 1] == 0, "param: raw string ends in its NUL");
    len = -7;
    expect(ncclParamGet(hu64, raw, 4, &len) == ncclInvalidArgument && len == 0, "param: a buffer too small is an error and len 0");
    len = -7;
    expect(ncclParamGet(hu64, raw, 0, &len) == ncclInvalidArgument && len == -7, "param: maxLen 0 leaves len alone");

    // By key, as text.
    const char* val = nullptr;
    int vl = -7;
    expect_rc(ncclParamGetParameter("NCCL_SET_THREAD_NAME", &val, &vl), ncclSuccess, "param: GetParameter");
    if (!std::getenv("NCCL_SET_THREAD_NAME")) expect(val && std::strcmp(val, "FALSE") == 0 && vl == 5, "param: ...FALSE, length 5");
    if (!std::getenv("NCCL_DEBUG_SUBSYS")) {
      ncclParamGetParameter("NCCL_DEBUG_SUBSYS", &val, &vl);
      expect(val && std::strcmp(val, "INIT,ENV,BOOTSTRAP") == 0 && vl == 18, "param: a flag list prints its names");
    }
    val = reinterpret_cast<const char*>(0x1);
    expect(ncclParamGetParameter("NCCL_NO_SUCH", &val, &vl) == ncclInvalidArgument && val == nullptr && vl == 0, "param: unknown key by name");
    expect_rc(ncclParamGetParameter(nullptr, &val, &vl), ncclInvalidArgument, "param: NULL name");

    // The dump goes to stdout; catch it.
    {
      std::fflush(stdout);
      FILE* tmp = std::tmpfile();
      const int saved = dup(1);
      bool ok = tmp && saved >= 0;
      if (ok) {
        dup2(fileno(tmp), 1);
        ncclParamDumpAll();
        std::fflush(stdout);
        dup2(saved, 1);
        std::rewind(tmp);
        std::string text;
        char chunk[512];
        size_t n;
        while ((n = std::fread(chunk, 1, sizeof chunk, tmp)) > 0) text.append(chunk, n);
        ok = text.find("=== ncclParam Registry Dump ===") == 0 && text.find("NCCL_DEBUG_SUBSYS (uint64_t)") != std::string::npos &&
             text.find("Current value=") != std::string::npos && text.find("default=") != std::string::npos;
      }
      if (saved >= 0) close(saved);
      if (tmp) std::fclose(tmp);
      expect(ok, "param: ncclParamDumpAll prints the registry");
    }
  } else {
    std::printf("skipped: the ncclParam registry needs libnccl 2.30 (this is %d)\n", g_ver);
  }
#else
  std::printf("skipped: the ncclParam registry needs nccl.h 2.30\n");
#endif

  // card: a blocking communicator finalizes once; the second is an error.
  // NVIDIA's libnccl (2.31.2, an RTX 3060 pair) hangs when one thread
  // finalizes the ranks one after another outside a group, so they go in one.
  NK(ncclGroupStart());
  for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclCommFinalize(world[i])); }
  NK(ncclGroupEnd());
  cudaSetDevice(0);
  expect_rc(ncclCommFinalize(world[0]), ncclInvalidArgument, "blocking: second finalize");
  for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); ncclCommDestroy(world[i]); cudaStreamDestroy(streams[i]); }
  std::printf("%s\n", g_bad ? "RESULT: FAIL" : "RESULT: all communicator operations behave as NCCL's");
  return g_bad ? 1 : 0;
}
