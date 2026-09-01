// NCCL collectives, self-verifying.
//
// Unlike the other conformance tests this one cannot use a physical GPU as a
// full oracle: NCCL wants one device per rank, and the reference machine has a
// single GPU. So the test does two things at once. Every collective is checked
// against its analytic result, which works at any rank count; and the whole
// program is deterministic, so running it at ranks=1 against NVIDIA's libnccl
// and against VirtualGPU's still gives a byte-for-byte differential check of
// the API plumbing.
//
//   VGPU_NCCL_RANKS=4   number of ranks (default: all visible devices, max 8)
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define NK(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) { \
  printf("%s -> %s\n", #x, ncclGetErrorString(r_)); return 1; } } while (0)

static int g_bad = 0;

// Each rank contributes a distinct, reproducible pattern.
static float contrib(int rank, int i) { return 1.0f + rank + 0.25f * (i % 7); }

static void check(const char* tag, const std::vector<float>& got, const std::vector<float>& want) {
  double err = 0;
  size_t n = got.size() < want.size() ? got.size() : want.size();
  for (size_t i = 0; i < n; ++i) err = std::fmax(err, std::fabs(got[i] - want[i]));
  const bool ok = got.size() == want.size() && err <= 1e-5;
  if (!ok) ++g_bad;
  double sum = 0; for (float v : got) sum += v;
  printf("%-26s n=%zu sum=%.4f %s\n", tag, got.size(), sum, ok ? "ok" : "BAD");
}

struct Rank {
  int dev;
  float* send = nullptr;
  float* recv = nullptr;
  cudaStream_t stream = nullptr;
};

int main() {
  int ndev = 0;
  cudaGetDeviceCount(&ndev);
  int nranks = ndev > 8 ? 8 : ndev;
  if (const char* e = std::getenv("VGPU_NCCL_RANKS")) nranks = std::atoi(e);
  if (nranks < 1) nranks = 1;
  if (nranks > ndev) { printf("SKIP: %d ranks requested, %d devices visible\n", nranks, ndev); return 0; }
  printf("nccl ranks=%d\n", nranks);

  const size_t N = 1024;  // divisible by every rank count used here
  std::vector<ncclComm_t> comms(nranks);
  std::vector<int> devs(nranks);
  for (int i = 0; i < nranks; ++i) devs[i] = i;
  NK(ncclCommInitAll(comms.data(), nranks, devs.data()));

  std::vector<Rank> r(nranks);
  for (int i = 0; i < nranks; ++i) {
    r[i].dev = devs[i];
    cudaSetDevice(devs[i]);
    cudaStreamCreate(&r[i].stream);
    cudaMalloc(&r[i].send, N * nranks * sizeof(float));
    cudaMalloc(&r[i].recv, N * nranks * sizeof(float));
  }
  auto load = [&](int i, size_t n) {
    std::vector<float> h(n);
    for (size_t k = 0; k < n; ++k) h[k] = contrib(i, (int)k);
    cudaSetDevice(r[i].dev);
    cudaMemcpy(r[i].send, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(r[i].recv, 0, n * sizeof(float));
  };
  auto grab = [&](int i, size_t n) {
    std::vector<float> h(n);
    cudaSetDevice(r[i].dev);
    cudaStreamSynchronize(r[i].stream);
    cudaMemcpy(h.data(), r[i].recv, n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
  };

  struct { const char* name; ncclRedOp_t op; } reds[] = {
      {"allreduce sum", ncclSum}, {"allreduce prod", ncclProd},
      {"allreduce max", ncclMax}, {"allreduce min", ncclMin},
      {"allreduce avg", ncclAvg},
  };
  for (auto& t : reds) {
    for (int i = 0; i < nranks; ++i) load(i, N);
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i)
      NK(ncclAllReduce(r[i].send, r[i].recv, N, ncclFloat, t.op, comms[i], r[i].stream));
    NK(ncclGroupEnd());
    std::vector<float> want(N);
    for (size_t k = 0; k < N; ++k) {
      double acc = contrib(0, (int)k);
      for (int i = 1; i < nranks; ++i) {
        const double v = contrib(i, (int)k);
        switch (t.op) {
          case ncclProd: acc *= v; break;
          case ncclMax: acc = v > acc ? v : acc; break;
          case ncclMin: acc = v < acc ? v : acc; break;
          default: acc += v; break;
        }
      }
      want[k] = (float)(t.op == ncclAvg ? acc / nranks : acc);
    }
    // Every rank must land on the same answer, so check them all.
    for (int i = 0; i < nranks; ++i) {
      char tag[64]; snprintf(tag, sizeof(tag), "%s r%d", t.name, i);
      check(tag, grab(i, N), want);
    }
  }

  {  // Broadcast from the last rank.
    const int root = nranks - 1;
    for (int i = 0; i < nranks; ++i) load(i, N);
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i)
      NK(ncclBroadcast(r[i].send, r[i].recv, N, ncclFloat, root, comms[i], r[i].stream));
    NK(ncclGroupEnd());
    std::vector<float> want(N);
    for (size_t k = 0; k < N; ++k) want[k] = contrib(root, (int)k);
    for (int i = 0; i < nranks; ++i) {
      char tag[64]; snprintf(tag, sizeof(tag), "broadcast r%d", i);
      check(tag, grab(i, N), want);
    }
  }

  {  // Reduce to rank 0 only.
    for (int i = 0; i < nranks; ++i) load(i, N);
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i)
      NK(ncclReduce(r[i].send, r[i].recv, N, ncclFloat, ncclSum, 0, comms[i], r[i].stream));
    NK(ncclGroupEnd());
    std::vector<float> want(N, 0.0f);
    for (size_t k = 0; k < N; ++k)
      for (int i = 0; i < nranks; ++i) want[k] += contrib(i, (int)k);
    check("reduce root", grab(0, N), want);
    if (nranks > 1) {
      std::vector<float> zero(N, 0.0f);
      check("reduce non-root untouched", grab(1, N), zero);
    }
  }

  {  // AllGather: N elements per rank, N*nranks out.
    for (int i = 0; i < nranks; ++i) load(i, N * nranks);
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i)
      NK(ncclAllGather(r[i].send, r[i].recv, N, ncclFloat, comms[i], r[i].stream));
    NK(ncclGroupEnd());
    std::vector<float> want(N * nranks);
    for (int i = 0; i < nranks; ++i)
      for (size_t k = 0; k < N; ++k) want[i * N + k] = contrib(i, (int)k);
    for (int i = 0; i < nranks; ++i) {
      char tag[64]; snprintf(tag, sizeof(tag), "allgather r%d", i);
      check(tag, grab(i, N * nranks), want);
    }
  }

  {  // ReduceScatter: N*nranks in, N out, rank i keeps slice i.
    const size_t per = N;
    for (int i = 0; i < nranks; ++i) load(i, per * nranks);
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i)
      NK(ncclReduceScatter(r[i].send, r[i].recv, per, ncclFloat, ncclSum, comms[i], r[i].stream));
    NK(ncclGroupEnd());
    for (int i = 0; i < nranks; ++i) {
      std::vector<float> want(per, 0.0f);
      for (size_t k = 0; k < per; ++k)
        for (int j = 0; j < nranks; ++j) want[k] += contrib(j, (int)(i * per + k));
      char tag[64]; snprintf(tag, sizeof(tag), "reducescatter r%d", i);
      check(tag, grab(i, per), want);
    }
  }

  if (nranks > 1) {  // Point-to-point ring: rank i sends to i+1.
    for (int i = 0; i < nranks; ++i) load(i, N);
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i) {
      const int next = (i + 1) % nranks, prev = (i + nranks - 1) % nranks;
      NK(ncclSend(r[i].send, N, ncclFloat, next, comms[i], r[i].stream));
      NK(ncclRecv(r[i].recv, N, ncclFloat, prev, comms[i], r[i].stream));
    }
    NK(ncclGroupEnd());
    for (int i = 0; i < nranks; ++i) {
      const int prev = (i + nranks - 1) % nranks;
      std::vector<float> want(N);
      for (size_t k = 0; k < N; ++k) want[k] = contrib(prev, (int)k);
      char tag[64]; snprintf(tag, sizeof(tag), "sendrecv ring r%d", i);
      check(tag, grab(i, N), want);
    }
  }

  {  // Integer and reduced-precision payloads go down the same path.
    std::vector<int*> si(nranks), ri(nranks);
    for (int i = 0; i < nranks; ++i) {
      cudaSetDevice(r[i].dev);
      cudaMalloc(&si[i], N * sizeof(int));
      cudaMalloc(&ri[i], N * sizeof(int));
      std::vector<int> h(N);
      for (size_t k = 0; k < N; ++k) h[k] = (int)(i + 1) * (int)(k % 11);
      cudaMemcpy(si[i], h.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    }
    NK(ncclGroupStart());
    for (int i = 0; i < nranks; ++i)
      NK(ncclAllReduce(si[i], ri[i], N, ncclInt32, ncclSum, comms[i], r[i].stream));
    NK(ncclGroupEnd());
    cudaSetDevice(r[0].dev);
    cudaStreamSynchronize(r[0].stream);
    std::vector<int> got(N);
    cudaMemcpy(got.data(), ri[0], N * sizeof(int), cudaMemcpyDeviceToHost);
    long long sum = 0, want = 0;
    for (size_t k = 0; k < N; ++k) {
      sum += got[k];
      for (int i = 0; i < nranks; ++i) want += (long long)(i + 1) * (int)(k % 11);
    }
    if (sum != want) ++g_bad;
    printf("%-26s sum=%lld %s\n", "allreduce int32", sum, sum == want ? "ok" : "BAD");
    for (int i = 0; i < nranks; ++i) { cudaSetDevice(r[i].dev); cudaFree(si[i]); cudaFree(ri[i]); }
  }

  int v = 0, cnt = 0, myrank = -1, mydev = -1;
  NK(ncclGetVersion(&v));
  NK(ncclCommCount(comms[0], &cnt));
  NK(ncclCommUserRank(comms[0], &myrank));
  NK(ncclCommCuDevice(comms[0], &mydev));
  printf("comm major=%d count=%d rank=%d dev=%d\n", v / 10000, cnt, myrank, mydev);

  for (int i = 0; i < nranks; ++i) {
    cudaSetDevice(r[i].dev);
    cudaFree(r[i].send); cudaFree(r[i].recv);
    cudaStreamDestroy(r[i].stream);
    ncclCommDestroy(comms[i]);
  }
  printf("%s\n", g_bad ? "RESULT: FAIL" : "RESULT: all collectives match their analytic value");
  return g_bad ? 1 : 0;
}
