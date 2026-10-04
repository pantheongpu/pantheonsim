// 64-bit atomics on shared memory, the way CUDA Samples' jacobiCudaGraphs
// uses them. atomicAdd of a double in shared memory compiles (sm_86) to a
// loop around ATOMS.CAST.SPIN.64, which reports in a register pair whether it
// stored, and the loop tests both registers: the simulator once left the old
// value's high word in the second, and the loop never ended.
//
// 1. Every thread of a block adds to one shared double (values that add up
//    exactly in any order), then to several, by index, as the sample does.
// 2. A tile of 8 threads (cooperative_groups::tiled_partition<8>) reducing
//    with shfl_down, then one atomicAdd per tile into global memory.
// 3. atomicCAS and atomicExch on a shared unsigned long long, and atomicMax
//    and atomicMin on a shared long long.
// 4. atomicAdd of a float in shared memory (the 32-bit form).
//
// Every result is checked exactly. Prints PASS on the last line.
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

__global__ void shared_doubles(double* out) {
  __shared__ double one;
  __shared__ double many[9];
  if (threadIdx.x == 0) one = 0.0;
  if (threadIdx.x < 9) many[threadIdx.x] = 0.0;
  __syncthreads();
  atomicAdd(&one, 0.5);
  atomicAdd(&many[threadIdx.x % 9], -static_cast<double>(threadIdx.x) * 0.25);
  __syncthreads();
  if (threadIdx.x == 0) out[0] = one;
  if (threadIdx.x < 9) out[1 + threadIdx.x] = many[threadIdx.x];
}

__global__ void tile8_sums(const double* in, double* sum) {
  cg::thread_block cta = cg::this_thread_block();
  if (threadIdx.x < 8) {
    cg::thread_block_tile<8> tile8 = cg::tiled_partition<8>(cta);
    double v = in[blockIdx.x * 8 + threadIdx.x];
    for (int offset = tile8.size() / 2; offset > 0; offset /= 2) v += tile8.shfl_down(v, offset);
    if (tile8.thread_rank() == 0) atomicAdd(sum, v);
  }
}

__global__ void shared_u64(unsigned long long* out) {
  __shared__ unsigned long long cas, xchg;
  __shared__ long long hi, lo;
  if (threadIdx.x == 0) {
    cas = 0;
    xchg = 0;
    hi = -(1ll << 40);
    lo = 1ll << 40;
  }
  __syncthreads();
  // Each thread adds its id plus 2^32 through a CAS loop.
  unsigned long long old = cas, seen;
  do {
    seen = old;
    old = atomicCAS(&cas, seen, seen + (1ull << 32) + threadIdx.x);
  } while (old != seen);
  atomicExch(&xchg, 0x0123456789abcdefull);
  const long long v = (static_cast<long long>(threadIdx.x) - 64) << 33;
  atomicMax(&hi, v);
  atomicMin(&lo, v);
  __syncthreads();
  if (threadIdx.x == 0) {
    out[0] = cas;
    out[1] = xchg;
    out[2] = static_cast<unsigned long long>(hi);
    out[3] = static_cast<unsigned long long>(lo);
  }
}

__global__ void shared_float(float* out) {
  __shared__ float f;
  if (threadIdx.x == 0) f = 0.0f;
  __syncthreads();
  atomicAdd(&f, 0.25f);
  __syncthreads();
  if (threadIdx.x == 0) *out = f;
}

int main() {
  bool ok = true;
  {
    double* d;
    double h[10];
    cudaMalloc(&d, sizeof h);
    shared_doubles<<<1, 256>>>(d);
    const cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    bool good = e == cudaSuccess && h[0] == 128.0;
    for (int k = 0; k < 9; ++k) {
      double want = 0;
      for (int t = k; t < 256; t += 9) want -= t * 0.25;
      good = good && h[1 + k] == want;
    }
    std::printf("shared double atomicAdd: %g, %g ... %g (%s)\n", h[0], h[1], h[9], good ? "exact" : "WRONG");
    ok = ok && good;
    cudaFree(d);
  }
  {
    double h[64], *in, *sum, s = 0;
    for (int i = 0; i < 64; ++i) h[i] = i * 0.125;
    cudaMalloc(&in, sizeof h);
    cudaMalloc(&sum, sizeof(double));
    cudaMemcpy(in, h, sizeof h, cudaMemcpyHostToDevice);
    cudaMemset(sum, 0, sizeof(double));
    tile8_sums<<<8, 64>>>(in, sum);
    const cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(&s, sum, sizeof s, cudaMemcpyDeviceToHost);
    const bool good = e == cudaSuccess && s == 63 * 64 / 2 * 0.125;
    std::printf("tile of 8 shfl_down, then atomicAdd: %g (%s)\n", s, good ? "exact" : "WRONG");
    ok = ok && good;
    cudaFree(in);
    cudaFree(sum);
  }
  {
    unsigned long long* d;
    unsigned long long h[4];
    cudaMalloc(&d, sizeof h);
    shared_u64<<<1, 128>>>(d);
    const cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    const bool good = e == cudaSuccess && h[0] == (128ull << 32) + 127 * 128 / 2 && h[1] == 0x0123456789abcdefull &&
                      static_cast<long long>(h[2]) == 63ll << 33 && static_cast<long long>(h[3]) == -(64ll << 33);
    std::printf("shared 64-bit CAS, exchange, max, min: %llx %llx %lld %lld (%s)\n", h[0], h[1],
                static_cast<long long>(h[2]), static_cast<long long>(h[3]), good ? "exact" : "WRONG");
    ok = ok && good;
    cudaFree(d);
  }
  {
    float* d;
    float h = 0;
    cudaMalloc(&d, sizeof(float));
    shared_float<<<1, 512>>>(d);
    const cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(&h, d, sizeof h, cudaMemcpyDeviceToHost);
    const bool good = e == cudaSuccess && h == 128.0f;
    std::printf("shared float atomicAdd: %g (%s)\n", h, good ? "exact" : "WRONG");
    ok = ok && good;
    cudaFree(d);
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
