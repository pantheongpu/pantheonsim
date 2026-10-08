// One rank per process, the way torchrun and mpirun launch NCCL jobs. The ranks
// share no address space, so this exercises the file-backed transport end to
// end -- the rendezvous, the barriers, and the peer reads -- in a way the
// single-process test cannot.
//
//   nccl_multiproc <rank> <nranks> <idfile>
//
// Rank 0 publishes the unique id through <idfile>; the others wait for it,
// exactly as a launcher would broadcast it over its own out-of-band channel.
//
// After the basic collectives it does what torch.distributed does with
// sub-groups: ncclCommSplit (one color per parity, keys reversed), a
// non-blocking child, a pre-multiplied sum with a device scalar, an all-to-all,
// and ncclCommShrink without the last rank -- the last rank does not call it,
// as NCCL requires -- and then ncclCommGetUniqueId + ncclCommGrow bringing the
// last rank back in. Each part works in this process alone and must still agree
// with every other process's.
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unistd.h>

#define NK(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) { \
  std::fprintf(stderr, "rank %d: %s -> %s\n", rank, #x, ncclGetErrorString(r_)); return 2; } } while (0)

static float contrib(int rank, int i) { return 1.0f + rank + 0.25f * (i % 7); }

// Polls a non-blocking communicator until its last operation finishes.
static ncclResult_t settle(ncclComm_t c) {
  ncclResult_t e = ncclInProgress;
  while (e == ncclInProgress)
    if (ncclCommGetAsyncError(c, &e) != ncclSuccess) return ncclInternalError;
  return e;
}

int main(int argc, char** argv) {
  if (argc < 4) { std::fprintf(stderr, "usage: %s <rank> <nranks> <idfile>\n", argv[0]); return 2; }
  const int rank = std::atoi(argv[1]);
  const int nranks = std::atoi(argv[2]);
  const std::string idfile = argv[3];

  ncclUniqueId id;
  if (rank == 0) {
    if (ncclGetUniqueId(&id) != ncclSuccess) return 2;
    const std::string tmp = idfile + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return 2;
    std::fwrite(&id, sizeof(id), 1, f);
    std::fclose(f);
    std::rename(tmp.c_str(), idfile.c_str());   // atomic: readers never see a partial id
  } else {
    for (int i = 0; i < 60000; ++i) {
      FILE* f = std::fopen(idfile.c_str(), "rb");
      if (f) {
        const bool ok = std::fread(&id, sizeof(id), 1, f) == 1;
        std::fclose(f);
        if (ok) break;
      }
      usleep(1000);
    }
  }

  cudaSetDevice(0);   // each process sees its own device, as CUDA_VISIBLE_DEVICES arranges
  ncclComm_t comm;
  NK(ncclCommInitRank(&comm, nranks, id, rank));

  const size_t N = 4096;
  float *send = nullptr, *recv = nullptr, *gather = nullptr;
  cudaMalloc(&send, N * sizeof(float));
  cudaMalloc(&recv, N * sizeof(float));
  cudaMalloc(&gather, N * nranks * sizeof(float));
  std::vector<float> h(N);
  for (size_t k = 0; k < N; ++k) h[k] = contrib(rank, (int)k);
  cudaMemcpy(send, h.data(), N * sizeof(float), cudaMemcpyHostToDevice);

  cudaStream_t s;
  cudaStreamCreate(&s);
  int bad = 0;

  NK(ncclAllReduce(send, recv, N, ncclFloat, ncclSum, comm, s));
  cudaStreamSynchronize(s);
  cudaMemcpy(h.data(), recv, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (size_t k = 0; k < N; ++k) {
    float want = 0;
    for (int i = 0; i < nranks; ++i) want += contrib(i, (int)k);
    if (std::fabs(h[k] - want) > 1e-4f) { ++bad; break; }
  }

  NK(ncclAllGather(send, gather, N, ncclFloat, comm, s));
  cudaStreamSynchronize(s);
  std::vector<float> g(N * nranks);
  cudaMemcpy(g.data(), gather, g.size() * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < nranks; ++i)
    for (size_t k = 0; k < N; ++k)
      if (std::fabs(g[i * N + k] - contrib(i, (int)k)) > 1e-4f) { ++bad; i = nranks; break; }

  if (nranks > 1) {  // ring: send to the next rank, receive from the previous one
    const int next = (rank + 1) % nranks, prev = (rank + nranks - 1) % nranks;
    NK(ncclGroupStart());
    NK(ncclSend(send, N, ncclFloat, next, comm, s));
    NK(ncclRecv(recv, N, ncclFloat, prev, comm, s));
    NK(ncclGroupEnd());
    cudaStreamSynchronize(s);
    cudaMemcpy(h.data(), recv, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t k = 0; k < N; ++k)
      if (std::fabs(h[k] - contrib(prev, (int)k)) > 1e-4f) { ++bad; break; }
  }

  int version = 0;
  ncclGetVersion(&version);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 18, 0)
  {  // Split by parity, keys reversed: new rank = members above me of my parity.
    ncclComm_t half = nullptr;
    NK(ncclCommSplit(comm, rank % 2, -rank, &half, nullptr));
    int hn = -1, hr = -1;
    NK(ncclCommCount(half, &hn));
    NK(ncclCommUserRank(half, &hr));
    const int members = (nranks - rank % 2 + 1) / 2;
    int above = 0;
    for (int r = rank + 2; r < nranks; r += 2) ++above;
    if (hn != members || hr != above) { ++bad; std::fprintf(stderr, "rank %d: split gave %d/%d\n", rank, hr, hn); }
    // Sum of the old ranks of my parity, over the child.
    int* v = nullptr;
    cudaMalloc(&v, sizeof(int));
    cudaMemcpy(v, &rank, sizeof(int), cudaMemcpyHostToDevice);
    NK(ncclAllReduce(v, v, 1, ncclInt32, ncclSum, half, s));
    cudaStreamSynchronize(s);
    int got = -1, want = 0;
    cudaMemcpy(&got, v, sizeof(int), cudaMemcpyDeviceToHost);
    for (int r = rank % 2; r < nranks; r += 2) want += r;
    if (got != want) { ++bad; std::fprintf(stderr, "rank %d: split allreduce %d, want %d\n", rank, got, want); }
    cudaFree(v);
    NK(ncclCommDestroy(half));

    // A non-blocking child: calls return ncclInProgress and are polled.
    ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
    cfg.blocking = 0;
    ncclComm_t nb = nullptr;
    NK(ncclCommSplit(comm, 0, rank, &nb, &cfg));
    if (settle(nb) != ncclSuccess) ++bad;
    const ncclResult_t ar = ncclAllReduce(send, recv, N, ncclFloat, ncclMax, nb, s);
    if (ar != ncclInProgress || settle(nb) != ncclSuccess) { ++bad; std::fprintf(stderr, "rank %d: non-blocking allreduce returned %d\n", rank, (int)ar); }
    cudaStreamSynchronize(s);
    cudaMemcpy(h.data(), recv, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t k = 0; k < N; ++k)
      if (h[k] != contrib(nranks - 1, (int)k)) { ++bad; break; }
    if (ncclCommFinalize(nb) != ncclInProgress || settle(nb) != ncclSuccess) ++bad;
    NK(ncclCommDestroy(nb));
  }
#endif
  {  // A pre-multiplied sum whose scalar lives on the device: rank r weighs r+1.
    float* scal = nullptr;
    cudaMalloc(&scal, sizeof(float));
    const float w = (float)(rank + 1);
    cudaMemcpy(scal, &w, sizeof w, cudaMemcpyHostToDevice);
    ncclRedOp_t op;
    NK(ncclRedOpCreatePreMulSum(&op, scal, ncclFloat, ncclScalarDevice, comm));
    std::vector<float> ints(N);
    for (size_t k = 0; k < N; ++k) ints[k] = (float)(k % 13 + rank);
    cudaMemcpy(send, ints.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    NK(ncclAllReduce(send, recv, N, ncclFloat, op, comm, s));
    cudaStreamSynchronize(s);
    cudaMemcpy(h.data(), recv, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t k = 0; k < N; ++k) {
      float want = 0;
      for (int r = 0; r < nranks; ++r) want += (float)(k % 13 + r) * (float)(r + 1);
      if (h[k] != want) { ++bad; std::fprintf(stderr, "rank %d: premulsum[%zu] %g want %g\n", rank, k, h[k], want); break; }
    }
    NK(ncclRedOpDestroy(op, comm));
    cudaFree(scal);
  }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
  if (version >= NCCL_VERSION(2, 28, 0)) {  // all-to-all, 5 elements a block
    const size_t C = 5;
    std::vector<float> a(C * nranks);
    for (size_t k = 0; k < a.size(); ++k) a[k] = 1000.0f * rank + (float)k;
    cudaMemcpy(send, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
    NK(ncclAlltoAll(send, gather, C, ncclFloat, comm, s));
    cudaStreamSynchronize(s);
    cudaMemcpy(a.data(), gather, a.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nranks; ++i)
      for (size_t k = 0; k < C; ++k)
        if (a[i * C + k] != 1000.0f * i + (float)(rank * C + k)) { ++bad; i = nranks; break; }
  }
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
  if (version >= NCCL_VERSION(2, 27, 0) && nranks > 1 && rank != nranks - 1) {
    // The last rank leaves; it does not take part.
    int gone[1] = {nranks - 1};
    ncclComm_t small = nullptr;
    NK(ncclCommShrink(comm, gone, 1, &small, nullptr, NCCL_SHRINK_DEFAULT));
    int sn = -1, sr = -1;
    NK(ncclCommCount(small, &sn));
    NK(ncclCommUserRank(small, &sr));
    if (sn != nranks - 1 || sr != rank) ++bad;
    int* v = nullptr;
    cudaMalloc(&v, sizeof(int));
    const int one = 1;
    cudaMemcpy(v, &one, sizeof(int), cudaMemcpyHostToDevice);
    NK(ncclAllReduce(v, v, 1, ncclInt32, ncclSum, small, s));
    cudaStreamSynchronize(s);
    int got = -1;
    cudaMemcpy(&got, v, sizeof(int), cudaMemcpyDeviceToHost);
    if (got != nranks - 1) { ++bad; std::fprintf(stderr, "rank %d: shrunk allreduce %d\n", rank, got); }
    cudaFree(v);
    NK(ncclCommDestroy(small));
  }
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
  if (version >= NCCL_VERSION(2, 29, 0) && nranks > 1) {
    // Grow the survivors of a shrink back to full size: the last rank, which
    // left, comes back as a new rank. The root of the survivors hands the id
    // over through a file, as a launcher would; the other survivors pass NULL.
    const std::string gidfile = idfile + ".grow";
    ncclComm_t big = nullptr;
    if (rank != nranks - 1) {
      int gone[1] = {nranks - 1};
      ncclComm_t small = nullptr;
      NK(ncclCommShrink(comm, gone, 1, &small, nullptr, NCCL_SHRINK_DEFAULT));
      ncclUniqueId gid;
      if (rank == 0) {
        NK(ncclCommGetUniqueId(small, &gid));
        const std::string tmp = gidfile + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) return 2;
        std::fwrite(&gid, sizeof(gid), 1, f);
        std::fclose(f);
        std::rename(tmp.c_str(), gidfile.c_str());
      }
      NK(ncclCommGrow(small, nranks, rank == 0 ? &gid : nullptr, -1, &big, nullptr));
      NK(ncclCommDestroy(small));
    } else {
      ncclUniqueId gid;
      for (int i = 0; i < 60000; ++i) {
        FILE* f = std::fopen(gidfile.c_str(), "rb");
        if (f) {
          const bool ok = std::fread(&gid, sizeof(gid), 1, f) == 1;
          std::fclose(f);
          if (ok) break;
        }
        usleep(1000);
      }
      usleep(1000000);   // NCCL wants the survivors to start before a new rank joins
      NK(ncclCommGrow(nullptr, nranks, &gid, nranks - 1, &big, nullptr));
    }
    int bn = -1, br = -1;
    NK(ncclCommCount(big, &bn));
    NK(ncclCommUserRank(big, &br));
    if (bn != nranks || br != rank) { ++bad; std::fprintf(stderr, "rank %d: grow gave %d/%d\n", rank, br, bn); }
    int* v = nullptr;
    cudaMalloc(&v, sizeof(int));
    const int mine = rank + 1;
    cudaMemcpy(v, &mine, sizeof(int), cudaMemcpyHostToDevice);
    NK(ncclAllReduce(v, v, 1, ncclInt32, ncclSum, big, s));
    cudaStreamSynchronize(s);
    int got = -1;
    cudaMemcpy(&got, v, sizeof(int), cudaMemcpyDeviceToHost);
    if (got != nranks * (nranks + 1) / 2) { ++bad; std::fprintf(stderr, "rank %d: grown allreduce %d\n", rank, got); }
    cudaFree(v);
    NK(ncclCommDestroy(big));
  }
#endif

  cudaStreamDestroy(s);
  cudaFree(send); cudaFree(recv); cudaFree(gather);
  ncclCommDestroy(comm);
  std::printf("rank %d of %d: %s\n", rank, nranks, bad ? "FAIL" : "ok");
  return bad ? 1 : 0;
}
