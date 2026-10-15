// CUDA Samples' quasirandomGenerator kernel, and the variations of it that tell where a wrong answer comes from.
//
// The sample copies a 3 x 31 table of unsigned ints into a __constant__ array, and a block of 128 x 3 threads
// builds each number by XORing the row threadIdx.y's entries picked by the bits of the position, then turns it
// into a float. On an RTX 3060 its GPU numbers agree with the CPU's to an L1 norm of 9.4e-8; here the norm was
// 0.55. Every variant below computes the same table values on the host, so a variant that differs says which
// part (a pointer into the constant array, indexing it directly, a table in global memory, __umul24, the
// conversion to float) the SASS interpreter gets wrong.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define DIMS 3
#define RES 31
#define INT_SCALE (1.0f / (float)0x80000001U)
#define MUL(a, b) __umul24(a, b)

static __constant__ unsigned int c_Table[DIMS][RES];
static __device__ unsigned int g_Table[DIMS][RES];

// The sample's kernel.
static __global__ void sample(float* out, unsigned int seed, unsigned int n) {
  unsigned int* dimBase = &c_Table[threadIdx.y][0];
  unsigned int tid = MUL(blockDim.x, blockIdx.x) + threadIdx.x;
  unsigned int threadN = MUL(blockDim.x, gridDim.x);
  for (unsigned int pos = tid; pos < n; pos += threadN) {
    unsigned int result = 0;
    unsigned int data = seed + pos;
    for (int bit = 0; bit < RES; bit++, data >>= 1)
      if (data & 1) result ^= dimBase[bit];
    out[MUL(threadIdx.y, n) + pos] = (float)(result + 1) * INT_SCALE;
  }
}

// The same, indexing the constant array directly instead of through a pointer.
static __global__ void direct(float* out, unsigned int seed, unsigned int n) {
  unsigned int tid = MUL(blockDim.x, blockIdx.x) + threadIdx.x;
  unsigned int threadN = MUL(blockDim.x, gridDim.x);
  for (unsigned int pos = tid; pos < n; pos += threadN) {
    unsigned int result = 0;
    unsigned int data = seed + pos;
    for (int bit = 0; bit < RES; bit++, data >>= 1)
      if (data & 1) result ^= c_Table[threadIdx.y][bit];
    out[MUL(threadIdx.y, n) + pos] = (float)(result + 1) * INT_SCALE;
  }
}

// The same through a pointer, writing the integer: no conversion to float.
static __global__ void integers(unsigned int* out, unsigned int seed, unsigned int n) {
  unsigned int* dimBase = &c_Table[threadIdx.y][0];
  unsigned int tid = MUL(blockDim.x, blockIdx.x) + threadIdx.x;
  unsigned int threadN = MUL(blockDim.x, gridDim.x);
  for (unsigned int pos = tid; pos < n; pos += threadN) {
    unsigned int result = 0;
    unsigned int data = seed + pos;
    for (int bit = 0; bit < RES; bit++, data >>= 1)
      if (data & 1) result ^= dimBase[bit];
    out[MUL(threadIdx.y, n) + pos] = result;
  }
}

// The same with the table in global memory.
static __global__ void in_global(unsigned int* out, unsigned int seed, unsigned int n) {
  unsigned int tid = MUL(blockDim.x, blockIdx.x) + threadIdx.x;
  unsigned int threadN = MUL(blockDim.x, gridDim.x);
  for (unsigned int pos = tid; pos < n; pos += threadN) {
    unsigned int result = 0;
    unsigned int data = seed + pos;
    for (int bit = 0; bit < RES; bit++, data >>= 1)
      if (data & 1) result ^= g_Table[threadIdx.y][bit];
    out[threadIdx.y * n + pos] = result;
  }
}

// The same with plain multiplications in place of __umul24.
static __global__ void plain_mul(unsigned int* out, unsigned int seed, unsigned int n) {
  unsigned int* dimBase = &c_Table[threadIdx.y][0];
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int threadN = blockDim.x * gridDim.x;
  for (unsigned int pos = tid; pos < n; pos += threadN) {
    unsigned int result = 0;
    unsigned int data = seed + pos;
    for (int bit = 0; bit < RES; bit++, data >>= 1)
      if (data & 1) result ^= dimBase[bit];
    out[threadIdx.y * n + pos] = result;
  }
}

static int failures = 0;
static void report(const char* what, long bad, long of) {
  std::printf("%-4s %s: %ld of %ld differ\n", bad ? "FAIL" : "ok", what, bad, of);
  if (bad) ++failures;
}

int main() {
  const unsigned n = 1u << 14;
  unsigned int table[DIMS][RES];
  uint64_t s = 0x9E3779B97F4A7C15ull;
  for (int d = 0; d < DIMS; ++d)
    for (int b = 0; b < RES; ++b) {
      s = s * 6364136223846793005ull + 1442695040888963407ull;
      table[d][b] = static_cast<unsigned>(s >> 33) & 0x7FFFFFFF;   // 31 bits, as the sample's are
    }
  cudaMemcpyToSymbol(c_Table, table, sizeof table);
  cudaMemcpyToSymbol(g_Table, table, sizeof table);

  static unsigned int want[DIMS * (1u << 14)];
  static float want_f[DIMS * (1u << 14)];
  for (int d = 0; d < DIMS; ++d)
    for (unsigned pos = 0; pos < n; ++pos) {
      unsigned r = 0, data = pos;
      for (int bit = 0; bit < RES; bit++, data >>= 1)
        if (data & 1) r ^= table[d][bit];
      want[d * n + pos] = r;
      want_f[d * n + pos] = (float)(r + 1) * INT_SCALE;
    }

  float* df;
  unsigned int* du;
  cudaMalloc(&df, DIMS * n * sizeof(float));
  cudaMalloc(&du, DIMS * n * sizeof(unsigned));
  static float got_f[DIMS * (1u << 14)];
  static unsigned int got_u[DIMS * (1u << 14)];
  const dim3 threads(128, DIMS);

  auto bad_f = [&](const char* what) {
    long bad = 0;
    for (unsigned i = 0; i < DIMS * n; ++i) bad += got_f[i] != want_f[i];
    report(what, bad, DIMS * n);
  };
  auto bad_u = [&](const char* what) {
    long bad = 0;
    for (unsigned i = 0; i < DIMS * n; ++i) bad += got_u[i] != want[i];
    report(what, bad, DIMS * n);
  };

  cudaMemset(df, 0, DIMS * n * sizeof(float));
  sample<<<128, threads>>>(df, 0, n);
  cudaMemcpy(got_f, df, sizeof got_f, cudaMemcpyDeviceToHost);
  bad_f("the sample's kernel (a pointer into the __constant__ table, __umul24, float result)");

  cudaMemset(df, 0, DIMS * n * sizeof(float));
  direct<<<128, threads>>>(df, 0, n);
  cudaMemcpy(got_f, df, sizeof got_f, cudaMemcpyDeviceToHost);
  bad_f("indexing the __constant__ table directly");

  cudaMemset(du, 0, DIMS * n * sizeof(unsigned));
  integers<<<128, threads>>>(du, 0, n);
  cudaMemcpy(got_u, du, sizeof got_u, cudaMemcpyDeviceToHost);
  bad_u("the integers, before the conversion to float");

  cudaMemset(du, 0, DIMS * n * sizeof(unsigned));
  in_global<<<128, threads>>>(du, 0, n);
  cudaMemcpy(got_u, du, sizeof got_u, cudaMemcpyDeviceToHost);
  bad_u("the table in global memory");

  cudaMemset(du, 0, DIMS * n * sizeof(unsigned));
  plain_mul<<<128, threads>>>(du, 0, n);
  cudaMemcpy(got_u, du, sizeof got_u, cudaMemcpyDeviceToHost);
  bad_u("plain multiplications instead of __umul24");

  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
