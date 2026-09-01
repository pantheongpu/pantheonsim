// One rank per process, the way torchrun and mpirun launch NCCL jobs. The ranks
// share no address space, so this exercises the file-backed transport end to
// end -- the rendezvous, the barriers, and the peer reads -- in a way the
// single-process test cannot.
//
//   nccl_multiproc <rank> <nranks> <idfile>
//
// Rank 0 publishes the unique id through <idfile>; the others wait for it,
// exactly as a launcher would broadcast it over its own out-of-band channel.
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

  cudaStreamDestroy(s);
  cudaFree(send); cudaFree(recv); cudaFree(gather);
  ncclCommDestroy(comm);
  std::printf("rank %d of %d: %s\n", rank, nranks, bad ? "FAIL" : "ok");
  return bad ? 1 : 0;
}
