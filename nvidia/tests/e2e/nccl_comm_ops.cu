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
#include <nccl.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

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
      ok = win[0] ? get(world[0], win[0], &user) == ncclSuccess && user == mem[0]
                  : get(world[0], win[0], &user) == ncclInvalidArgument;
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

  // card: a blocking communicator finalizes once; the second is an error.
  for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); NK(ncclCommFinalize(world[i])); }
  cudaSetDevice(0);
  expect_rc(ncclCommFinalize(world[0]), ncclInvalidArgument, "blocking: second finalize");
  for (int i = 0; i < nranks; ++i) { cudaSetDevice(i); ncclCommDestroy(world[i]); cudaStreamDestroy(streams[i]); }
  std::printf("%s\n", g_bad ? "RESULT: FAIL" : "RESULT: all communicator operations behave as NCCL's");
  return g_bad ? 1 : 0;
}
