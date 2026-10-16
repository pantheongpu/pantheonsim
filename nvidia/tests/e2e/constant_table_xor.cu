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


// Narrower probes, all writing the integer, to find which ingredient of the loop above goes wrong: the 2D block,
// the grid-stride loop, the shift-and-test loop, the table, or the XOR.
static __global__ void p_index(unsigned* out, unsigned n) {             // the output indexing and the 2D block alone
  unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n) out[threadIdx.y * n + tid] = threadIdx.y * n + tid;
}
static __global__ void p_stride(unsigned* out, unsigned n) {            // plus the grid-stride loop
  unsigned tid = blockDim.x * blockIdx.x + threadIdx.x, step = blockDim.x * gridDim.x;
  for (unsigned pos = tid; pos < n; pos += step) out[threadIdx.y * n + pos] = pos ^ 0x5a5a5a5au;
}
static __global__ void p_bits(unsigned* out, unsigned n) {              // the shift-and-test loop, no table
  unsigned tid = blockDim.x * blockIdx.x + threadIdx.x, step = blockDim.x * gridDim.x;
  for (unsigned pos = tid; pos < n; pos += step) {
    unsigned result = 0, data = pos;
    for (int bit = 0; bit < RES; bit++, data >>= 1)
      if (data & 1) result ^= 0x9E3779B9u * (bit + 1);
    out[threadIdx.y * n + pos] = result;
  }
}
static __global__ void p_bits_nostride(unsigned* out, unsigned n) {     // the same, one position per thread
  unsigned pos = blockDim.x * blockIdx.x + threadIdx.x;
  if (pos >= n) return;
  unsigned result = 0, data = pos;
  for (int bit = 0; bit < RES; bit++, data >>= 1)
    if (data & 1) result ^= 0x9E3779B9u * (bit + 1);
  out[threadIdx.y * n + pos] = result;
}
static __global__ void p_bits_1d(unsigned* out, unsigned n) {           // the same, 1D blocks, y never read
  unsigned pos = blockDim.x * blockIdx.x + threadIdx.x;
  if (pos >= n) return;
  unsigned result = 0, data = pos;
  for (int bit = 0; bit < RES; bit++, data >>= 1)
    if (data & 1) result ^= 0x9E3779B9u * (bit + 1);
  out[pos] = result;
}
static __global__ void p_bits_select(unsigned* out, unsigned n) {       // the bits picked without the shifting loop
  unsigned pos = blockDim.x * blockIdx.x + threadIdx.x;
  if (pos >= n) return;
  unsigned result = 0;
  for (int bit = 0; bit < RES; bit++)
    if ((pos >> bit) & 1) result ^= 0x9E3779B9u * (bit + 1);
  out[pos] = result;
}
static __global__ void p_table_1d(unsigned* out, unsigned n) {          // the global table, 1D, no stride
  unsigned pos = blockDim.x * blockIdx.x + threadIdx.x;
  if (pos >= n) return;
  unsigned result = 0, data = pos;
  for (int bit = 0; bit < RES; bit++, data >>= 1)
    if (data & 1) result ^= g_Table[0][bit];
  out[pos] = result;
}

// Round two: the probes above pass, so what is left of the failing kernels is the seed added to the position
// and the table row picked by threadIdx.y.
static __global__ void q_seed(unsigned* out, unsigned seed, unsigned n) {          // no table, seed + pos
  unsigned tid = blockDim.x * blockIdx.x + threadIdx.x, step = blockDim.x * gridDim.x;
  for (unsigned pos = tid; pos < n; pos += step) {
    unsigned result = 0, data = seed + pos;
    for (int bit = 0; bit < RES; bit++, data >>= 1)
      if (data & 1) result ^= 0x9E3779B9u * (bit + 1);
    out[threadIdx.y * n + pos] = result;
  }
}
static __global__ void q_row_nostride(unsigned* out, unsigned n) {                 // row = threadIdx.y, no stride, no seed
  unsigned pos = blockDim.x * blockIdx.x + threadIdx.x;
  if (pos >= n) return;
  unsigned result = 0, data = pos;
  for (int bit = 0; bit < RES; bit++, data >>= 1)
    if (data & 1) result ^= g_Table[threadIdx.y][bit];
  out[threadIdx.y * n + pos] = result;
}
static __global__ void q_row_stride(unsigned* out, unsigned n) {                   // row = threadIdx.y, stride, no seed
  unsigned tid = blockDim.x * blockIdx.x + threadIdx.x, step = blockDim.x * gridDim.x;
  for (unsigned pos = tid; pos < n; pos += step) {
    unsigned result = 0, data = pos;
    for (int bit = 0; bit < RES; bit++, data >>= 1)
      if (data & 1) result ^= g_Table[threadIdx.y][bit];
    out[threadIdx.y * n + pos] = result;
  }
}
static __global__ void q_row0(unsigned* out, unsigned n) {                         // 2D block, but row 0 always
  unsigned tid = blockDim.x * blockIdx.x + threadIdx.x, step = blockDim.x * gridDim.x;
  for (unsigned pos = tid; pos < n; pos += step) {
    unsigned result = 0, data = pos;
    for (int bit = 0; bit < RES; bit++, data >>= 1)
      if (data & 1) result ^= g_Table[0][bit];
    out[threadIdx.y * n + pos] = result;
  }
}
static __global__ void q_flat(unsigned* out, unsigned n) {                         // the row as an offset into a flat pointer
  const unsigned* base = &g_Table[0][0];
  unsigned tid = blockDim.x * blockIdx.x + threadIdx.x, step = blockDim.x * gridDim.x;
  for (unsigned pos = tid; pos < n; pos += step) {
    unsigned result = 0, data = pos;
    for (int bit = 0; bit < RES; bit++, data >>= 1)
      if (data & 1) result ^= base[threadIdx.y * RES + bit];
    out[threadIdx.y * n + pos] = result;
  }
}
static __global__ void q_row_load_only(unsigned* out, unsigned n) {                // just the loads: out = table[y][pos % RES]
  unsigned pos = blockDim.x * blockIdx.x + threadIdx.x;
  if (pos >= n) return;
  out[threadIdx.y * n + pos] = g_Table[threadIdx.y][pos % RES];
}

static unsigned* g_probe_out;
static unsigned g_probe_got[DIMS * (1u << 14)], g_probe_want[DIMS * (1u << 14)];
static void probe_report(const char* what, unsigned count) {
  long bad = 0;
  int shown = 0;
  for (unsigned i = 0; i < count; ++i)
    if (g_probe_got[i] != g_probe_want[i]) {
      if (shown++ < 3) std::printf("       [%u] got %08x want %08x\n", i, g_probe_got[i], g_probe_want[i]);
      ++bad;
    }
  std::printf("%-4s %s: %ld of %u differ\n", bad ? "FAIL" : "ok", what, bad, count);
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


  // Narrower probes (see above).
  unsigned* po = du;
  const unsigned total3 = DIMS * n;
  auto bits_want = [&](unsigned pos) {
    unsigned r = 0, data = pos;
    for (int bit = 0; bit < RES; bit++, data >>= 1)
      if (data & 1) r ^= 0x9E3779B9u * (bit + 1);
    return r;
  };
  auto run = [&](const char* what, unsigned count, auto want_of) {
    cudaMemcpy(g_probe_got, po, count * sizeof(unsigned), cudaMemcpyDeviceToHost);
    for (unsigned i = 0; i < count; ++i) g_probe_want[i] = want_of(i);
    probe_report(what, count);
  };
  cudaMemset(po, 0, total3 * 4);
  p_index<<<(n + 127) / 128, threads>>>(po, n);
  run("2D block, indexing only", total3, [&](unsigned i) { return i; });
  cudaMemset(po, 0, total3 * 4);
  p_stride<<<128, threads>>>(po, n);
  run("plus the grid-stride loop", total3, [&](unsigned i) { return (i % n) ^ 0x5a5a5a5au; });
  cudaMemset(po, 0, total3 * 4);
  p_bits<<<128, threads>>>(po, n);
  run("shift-and-test loop with no table, 2D block, grid-stride", total3, [&](unsigned i) { return bits_want(i % n); });
  cudaMemset(po, 0, total3 * 4);
  p_bits_nostride<<<(n + 127) / 128, threads>>>(po, n);
  run("the same without the grid-stride loop", total3, [&](unsigned i) { return bits_want(i % n); });
  cudaMemset(po, 0, total3 * 4);
  p_bits_1d<<<(n + 127) / 128, 128>>>(po, n);
  run("the same with 1D blocks", n, [&](unsigned i) { return bits_want(i); });
  cudaMemset(po, 0, total3 * 4);
  p_bits_select<<<(n + 127) / 128, 128>>>(po, n);
  run("bits picked by (pos >> bit) & 1", n, [&](unsigned i) { return bits_want(i); });
  cudaMemset(po, 0, total3 * 4);
  p_table_1d<<<(n + 127) / 128, 128>>>(po, n);
  run("a global table, 1D blocks", n, [&](unsigned i) {
    unsigned r = 0, data = i;
    for (int bit = 0; bit < RES; bit++, data >>= 1)
      if (data & 1) r ^= table[0][bit];
    return r;
  });


  // Round two.
  auto table_want = [&](unsigned dim, unsigned pos) {
    unsigned r = 0, data = pos;
    for (int bit = 0; bit < RES; bit++, data >>= 1)
      if (data & 1) r ^= table[dim][bit];
    return r;
  };
  cudaMemset(po, 0, total3 * 4);
  q_seed<<<128, threads>>>(po, 0, n);
  run("seed + pos (seed 0), no table", total3, [&](unsigned i) { return bits_want(i % n); });
  cudaMemset(po, 0, total3 * 4);
  q_seed<<<128, threads>>>(po, 5, n);
  run("seed + pos (seed 5), no table", total3, [&](unsigned i) { return bits_want(5 + i % n); });
  cudaMemset(po, 0, total3 * 4);
  q_row_load_only<<<(n + 127) / 128, threads>>>(po, n);
  run("table[threadIdx.y][pos % 31], loads only", total3, [&](unsigned i) { return table[i / n][(i % n) % RES]; });
  cudaMemset(po, 0, total3 * 4);
  q_row_nostride<<<(n + 127) / 128, threads>>>(po, n);
  run("row = threadIdx.y, no stride, no seed", total3, [&](unsigned i) { return table_want(i / n, i % n); });
  cudaMemset(po, 0, total3 * 4);
  q_row_stride<<<128, threads>>>(po, n);
  run("row = threadIdx.y, grid-stride, no seed", total3, [&](unsigned i) { return table_want(i / n, i % n); });
  cudaMemset(po, 0, total3 * 4);
  q_row0<<<128, threads>>>(po, n);
  run("2D block, row 0 always, grid-stride", total3, [&](unsigned i) { return table_want(0, i % n); });
  cudaMemset(po, 0, total3 * 4);
  q_flat<<<128, threads>>>(po, n);
  run("row as an offset into a flat pointer", total3, [&](unsigned i) { return table_want(i / n, i % n); });

  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
