// SASS forms that once ran wrong, each in the shape ptxas emits it for some
// architecture (nvidia/tests/e2e/run_sass_archs.sh builds this for every
// one and compares the SASS run against the PTX one):
//   - a 64-bit compare against a negative constant (ISETP.GT.S64 R, -0x1),
//   - a 64-bit negate whose low word is zero (LEA Rd, P, -Ra: the carry),
//   - a return taken by some lanes of a reduction (EXIT Pn),
//   - float and integer block reductions ending in one thread's store,
//   - an mbarrier pipeline between warps (SYNCS from sm_90),
//   - thread-block clusters: their special registers, barrier and
//     distributed shared memory (sm_90+).
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void compare64(const long long* in, int* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long v = in[i];
  out[i] = (v > -1 ? 1 : 0) | (v >= -128 ? 2 : 0) | (v >= -2147483648LL ? 4 : 0) | (v < 4294967296LL ? 8 : 0);
}

// -(i * 8) in 64 bits, as triu_indices computes its offsets.
__global__ void negate64(long long base, long long* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long x = base + i;
  out[i] = -(x * 8);
}

template <class T>
__global__ void block_sum(const T* in, T* out, int n) {
  __shared__ T part[256];
  T s = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s += in[blockIdx.x * n + i];
  part[threadIdx.x] = s;
  __syncthreads();
  for (int w = blockDim.x / 2; w > 0; w /= 2) {
    if (threadIdx.x < w) part[threadIdx.x] += part[threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x != 0 || blockIdx.y != 0) return;
  out[blockIdx.x] = part[0];
}

// PyTorch's reduction epilogue: with `mode` set only thread 0 stores, and
// ptxas folds the two returns into EXIT Pn (the lanes with the guard and Pn).
__global__ void reduce_rows(const float* in, float* out, int n, int mode, int rows) {
  __shared__ float part[256];
  float s = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s += in[i];
  part[threadIdx.x] = s;
  __syncthreads();
  if (mode) {
    for (int w = 16; w > 0; w /= 2) s += __shfl_down_sync(0xffffffff, s, w);
  }
  if (static_cast<int>(blockIdx.x) >= rows) return;
  if (mode && threadIdx.x != 0) return;
  out[blockIdx.x * (mode ? 1 : blockDim.x) + (mode ? 0 : threadIdx.x)] = s + part[0];
}

// A search loop left early by some lanes: the first index at or after each
// thread's start whose value exceeds a limit.
__global__ void find_first(const int* in, int n, int limit, int* out) {
  const int t = threadIdx.x;
  int at = -1;
  for (int i = t; i < n; ++i) {
    if (in[i] > limit) {
      at = i;
      break;
    }
  }
  out[t] = at;
}

// mbarriers (sm_80; transaction counts and try_wait from sm_90): warps 1-3
// produce a round of data and arrive on `full`; warp 0 waits on it, sums the
// round and arrives on `empty`, which the producers wait on before the next
// round. From sm_90 one producer also declares bytes (arrive.expect_tx) that
// another completes (complete_tx), so a round needs its bytes as well as its
// arrivals; in the last round warp 3 leaves for good (arrive_drop), and a
// round after that completes without it. out: the sums, then the pending
// count right after init, then whether a token's test_wait saw its phase end.
__device__ __forceinline__ unsigned smem(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void wait_parity(unsigned bar, unsigned parity) {
#if __CUDA_ARCH__ >= 900
  unsigned done = 0;
  while (!done)
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(done) : "r"(bar), "r"(parity) : "memory");
#elif __CUDA_ARCH__ >= 800
  unsigned done = 0;
  while (!done)
    asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(done) : "r"(bar), "r"(parity) : "memory");
#endif
}

__global__ void mbar_pipeline(int* out, int rounds) {
#if __CUDA_ARCH__ >= 800
  __shared__ alignas(8) unsigned long long full, empty;
  __shared__ int data[96];
  const int t = threadIdx.x, warp = t / 32;
  const unsigned f = smem(&full), e = smem(&empty);
  if (t == 0) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(f), "r"(96));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(e), "r"(32));
  }
  __syncthreads();
  if (t == 0) {
    unsigned long long st;
    unsigned pending;
    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(st) : "r"(e) : "memory");   // 1 of warp 0's 32
    asm volatile("mbarrier.pending_count.b64 %0, %1;" : "=r"(pending) : "l"(st));
    out[rounds] = static_cast<int>(pending);
    int ended;
    asm volatile("{ .reg .pred p; mbarrier.test_wait.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ended) : "r"(e), "l"(st) : "memory");
    out[rounds + 1] = ended;   // the phase needs 31 more
  }
  __syncthreads();
  if (warp == 0 && t != 0) asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(e) : "memory");   // phase 0 of empty done
  for (int r = 0; r < rounds; ++r) {
    const bool leaving = r == rounds - 2 && warp == 3;
    if (warp > 0) {
      if (warp == 3 && r > rounds - 2) break;   // gone
      wait_parity(e, r & 1);                    // the consumer has emptied the buffer
      data[t - 32] = r * 1000 + t;
#if __CUDA_ARCH__ >= 900
      if (t == 32 && r < rounds - 1) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(f), "r"(64) : "memory");
      } else if (t == 33 && r < rounds - 1) {
        asm volatile("mbarrier.complete_tx.shared::cta.b64 [%0], %1;" :: "r"(f), "r"(64) : "memory");
        asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(f) : "memory");
      } else
#endif
      if (leaving) {
        asm volatile("mbarrier.arrive_drop.shared.b64 _, [%0];" :: "r"(f) : "memory");
      } else {
        asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(f) : "memory");
      }
    } else {
      wait_parity(f, r & 1);
      int s = 0;
      const int n = r == rounds - 1 ? 64 : 96;   // warp 3 left after the round before
      for (int i = t; i < n; i += 32) s += data[i];
      for (int o = 16; o > 0; o /= 2) s += __shfl_down_sync(0xffffffff, s, o);
      if (t == 0) out[r] = s;
      __syncwarp();
      asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(e) : "memory");
    }
  }
#else
  if (threadIdx.x == 0) out[0] = -1;   // no mbarriers before sm_80
#endif
}

// Thread-block clusters (sm_90+): every cluster special register, and a value
// passed around the cluster through distributed shared memory. Launched as a
// 2x2x1 cluster over a 4x4 grid. Each block writes 13 words.
__global__ void cluster_ids(unsigned* out) {
#if __CUDA_ARCH__ >= 900
  __shared__ unsigned mine;
  unsigned v[12];
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(v[0]));
  asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(v[1]));
  asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(v[2]));
  asm volatile("mov.u32 %0, %%cluster_ctaid.y;" : "=r"(v[3]));
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(v[4]));
  asm volatile("mov.u32 %0, %%cluster_nctaid.y;" : "=r"(v[5]));
  asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(v[6]));
  asm volatile("mov.u32 %0, %%clusterid.y;" : "=r"(v[7]));
  asm volatile("mov.u32 %0, %%nclusterid.x;" : "=r"(v[8]));
  asm volatile("mov.u32 %0, %%nclusterid.y;" : "=r"(v[9]));
  asm volatile("{ .reg .pred p; mov.pred p, %%is_explicit_cluster; selp.u32 %0, 1, 0, p; }" : "=r"(v[10]));
  if (threadIdx.x == 0) mine = 100 * (blockIdx.y * gridDim.x + blockIdx.x) + v[0];
  asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::: "memory");
  // The next rank's value, read through its shared memory.
  unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(&mine)), r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"((v[0] + 1) % v[1]));
  asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v[11]) : "r"(r));
  asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::: "memory");
  if (threadIdx.x == 0) {
    unsigned* o = out + 13 * (blockIdx.y * gridDim.x + blockIdx.x);
    for (int i = 0; i < 12; ++i) o[i] = v[i];
    o[12] = 1;
  }
#endif
}

static int g_fail = 0;
#define CHECK(c, ...)                        \
  do {                                       \
    if (!(c)) {                              \
      std::printf("FAIL: " __VA_ARGS__);     \
      std::printf("\n");                     \
      ++g_fail;                              \
    }                                        \
  } while (0)

int main() {
  const int n = 256;
  // 64-bit compares.
  long long hv[8] = {-1, 0, 5, -128, -129, -2147483648LL, -2147483649LL, 4294967296LL};
  long long* dv;
  int* dflags;
  cudaMalloc(&dv, sizeof hv);
  cudaMalloc(&dflags, 8 * sizeof(int));
  cudaMemcpy(dv, hv, sizeof hv, cudaMemcpyHostToDevice);
  compare64<<<1, 8>>>(dv, dflags, 8);
  int flags[8];
  cudaMemcpy(flags, dflags, sizeof flags, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 8; ++i) {
    const long long v = hv[i];
    const int want = (v > -1 ? 1 : 0) | (v >= -128 ? 2 : 0) | (v >= -2147483648LL ? 4 : 0) | (v < 4294967296LL ? 8 : 0);
    CHECK(flags[i] == want, "compare64(%lld) = %d, want %d", v, flags[i], want);
  }

  // 64-bit negates, from 0 (whose low word is zero) up.
  long long* dneg;
  cudaMalloc(&dneg, n * sizeof(long long));
  long long hneg[n];
  for (long long base : {0LL, 0x100000000LL - 3, -5LL}) {
    negate64<<<2, 128>>>(base, dneg, n);
    cudaMemcpy(hneg, dneg, sizeof hneg, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
      CHECK(hneg[i] == -((base + i) * 8), "negate64(%lld + %d) = %lld", base, i, hneg[i]);
  }

  // Block reductions: every block's result is written by its thread 0 alone.
  const int blocks = 4, per = 1000;
  float hf[blocks * per];
  int hi[blocks * per];
  for (int i = 0; i < blocks * per; ++i) {
    hf[i] = static_cast<float>(i % 7) * 0.5f;
    hi[i] = (i * 37) % 101 - 50;
  }
  float *df, *dfo;
  int *di, *dio;
  cudaMalloc(&df, sizeof hf);
  cudaMalloc(&di, sizeof hi);
  cudaMalloc(&dfo, blocks * sizeof(float));
  cudaMalloc(&dio, blocks * sizeof(int));
  cudaMemcpy(df, hf, sizeof hf, cudaMemcpyHostToDevice);
  cudaMemcpy(di, hi, sizeof hi, cudaMemcpyHostToDevice);
  cudaMemset(dfo, 0xff, blocks * sizeof(float));
  cudaMemset(dio, 0xff, blocks * sizeof(int));
  block_sum<float><<<blocks, 256>>>(df, dfo, per);
  block_sum<int><<<blocks, 256>>>(di, dio, per);
  float fo[blocks];
  int io[blocks];
  cudaMemcpy(fo, dfo, sizeof fo, cudaMemcpyDeviceToHost);
  cudaMemcpy(io, dio, sizeof io, cudaMemcpyDeviceToHost);
  for (int b = 0; b < blocks; ++b) {
    float fs = 0;
    int is = 0;
    for (int i = 0; i < per; ++i) {
      fs += hf[b * per + i];
      is += hi[b * per + i];
    }
    CHECK(fo[b] > fs - 0.01f && fo[b] < fs + 0.01f, "float block_sum[%d] = %g, want %g", b, fo[b], fs);
    CHECK(io[b] == is, "int block_sum[%d] = %d, want %d", b, io[b], is);
  }

  // The epilogue: one store per block with mode set, one per thread without.
  {
    float hin2[64], hout2[2 * 32];
    for (int i = 0; i < 64; ++i) hin2[i] = static_cast<float>(i % 5);
    float *din2, *dout2;
    cudaMalloc(&din2, sizeof hin2);
    cudaMalloc(&dout2, sizeof hout2);
    cudaMemcpy(din2, hin2, sizeof hin2, cudaMemcpyHostToDevice);
    for (int mode = 0; mode < 2; ++mode) {
      cudaMemset(dout2, 0, sizeof hout2);
      reduce_rows<<<2, 32>>>(din2, dout2, 64, mode, 2);
      cudaMemcpy(hout2, dout2, sizeof hout2, cudaMemcpyDeviceToHost);
      float t0 = hin2[0] + hin2[32];   // thread 0's own sum, and part[0]
      if (mode) {
        float all = 0;
        for (int i = 0; i < 64; ++i) all += hin2[i];
        for (int b = 0; b < 2; ++b) CHECK(hout2[b] == all + t0, "reduce_rows mode 1 [%d] = %g, want %g", b, hout2[b], all + t0);
        for (int i = 2; i < 64; ++i) CHECK(hout2[i] == 0, "reduce_rows mode 1 wrote [%d]", i);
      } else {
        for (int b = 0; b < 2; ++b)
          for (int t = 0; t < 32; ++t) {
            const float want = hin2[t] + hin2[t + 32] + t0;
            CHECK(hout2[b * 32 + t] == want, "reduce_rows mode 0 [%d][%d] = %g, want %g", b, t, hout2[b * 32 + t], want);
          }
      }
    }
  }

  // Loops some lanes leave early.
  int hin[64], hout[32];
  for (int i = 0; i < 64; ++i) hin[i] = (i * 13) % 29;
  int *din, *dout;
  cudaMalloc(&din, sizeof hin);
  cudaMalloc(&dout, sizeof hout);
  cudaMemcpy(din, hin, sizeof hin, cudaMemcpyHostToDevice);
  find_first<<<1, 32>>>(din, 64, 25, dout);
  cudaMemcpy(hout, dout, sizeof hout, cudaMemcpyDeviceToHost);
  for (int t = 0; t < 32; ++t) {
    int want = -1;
    for (int i = t; i < 64; ++i)
      if (hin[i] > 25) {
        want = i;
        break;
      }
    CHECK(hout[t] == want, "find_first[%d] = %d, want %d", t, hout[t], want);
  }

  // mbarriers.
  {
    const int rounds = 5;
    int* dm;
    cudaMalloc(&dm, (rounds + 2) * sizeof(int));
    cudaMemset(dm, 0, (rounds + 2) * sizeof(int));
    mbar_pipeline<<<1, 128>>>(dm, rounds);
    int hm[rounds + 2];
    cudaMemcpy(hm, dm, sizeof hm, cudaMemcpyDeviceToHost);
    if (hm[0] != -1) {
      for (int r = 0; r < rounds; ++r) {
        const int n = r == rounds - 1 ? 64 : 96;
        int want = 0;
        for (int i = 0; i < n; ++i) want += r * 1000 + 32 + i;
        CHECK(hm[r] == want, "mbarrier round %d summed %d, want %d", r, hm[r], want);
      }
      CHECK(hm[rounds] == 32, "mbarrier pending count after init = %d, want 32", hm[rounds]);
      CHECK(hm[rounds + 1] == 0, "a test_wait before the phase ended said it had");
    }
  }

  // Clusters.
  {
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    if (major >= 9) {
      unsigned* dc;
      const int nb = 16;
      cudaMalloc(&dc, 13 * nb * sizeof(unsigned));
      cudaMemset(dc, 0, 13 * nb * sizeof(unsigned));
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3(4, 4, 1);
      cfg.blockDim = dim3(64, 1, 1);
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeClusterDimension;
      attr[0].val.clusterDim.x = 2;
      attr[0].val.clusterDim.y = 2;
      attr[0].val.clusterDim.z = 1;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      CHECK(cudaLaunchKernelEx(&cfg, cluster_ids, dc) == cudaSuccess, "cluster launch: %s", cudaGetErrorString(cudaGetLastError()));
      unsigned hc[13 * nb];
      cudaMemcpy(hc, dc, sizeof hc, cudaMemcpyDeviceToHost);
      for (int by = 0; by < 4; ++by)
        for (int bx = 0; bx < 4; ++bx) {
          const unsigned* o = hc + 13 * (by * 4 + bx);
          const unsigned cx = bx % 2, cy = by % 2, rank = cx + 2 * cy;
          const unsigned next = (rank + 1) % 4;
          const unsigned nbx = bx - cx + next % 2, nby = by - cy + next / 2;
          const unsigned want[13] = {rank, 4, cx, cy, 2, 2, bx / 2u, by / 2u, 2, 2, 1, 100 * (nby * 4 + nbx) + next, 1};
          for (int i = 0; i < 13; ++i)
            CHECK(o[i] == want[i], "cluster block (%d,%d) word %d = %u, want %u", bx, by, i, o[i], want[i]);
        }
    }
  }

  const cudaError_t e = cudaDeviceSynchronize();
  CHECK(e == cudaSuccess, "%s", cudaGetErrorString(e));
  std::printf(g_fail ? "FAIL (%d)\n" : "PASS\n", g_fail);
  return g_fail != 0;
}
