// Every WMMA shape and element type, through CUDA's own mma.h API, so the PTX
// is what nvcc emits for real code: f16 with f16 and f32 accumulators, bf16,
// tf32, s8/u8, f64, s4/u4 and b1, in the square and rectangular shapes, every
// layout each allows. Each is loaded, multiplied, stored and compared with the
// same product worked out on the host.
//
// The inputs are small integers, so every product and every partial sum is
// exact in every one of these types: the comparison can be exact without
// depending on the order a tensor core accumulates in, which the ISA leaves
// unspecified. Prints PASS on the last line.
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace nvcuda;

static int g_fail = 0;

// A small integer for element (i, j) of matrix `which`, in [-2, 2] (or [0, 3]
// when the type is unsigned).
__host__ __device__ inline int val(int which, int i, int j, bool unsigned_type) {
  const int v = (i * 7 + j * 3 + which * 5) % 5;
  return unsigned_type ? v % 4 : v - 2;
}

template <int M, int N, int K, class TA, class Storage, class TC, class LA, class LB>
__global__ void mma_kernel(const Storage* a, const Storage* b, const TC* c, TC* d) {
  wmma::fragment<wmma::matrix_a, M, N, K, TA, LA> fa;
  wmma::fragment<wmma::matrix_b, M, N, K, TA, LB> fb;
  wmma::fragment<wmma::accumulator, M, N, K, TC> fc;
  constexpr bool a_row = std::is_same_v<LA, wmma::row_major>;
  constexpr bool b_row = std::is_same_v<LB, wmma::row_major>;
  wmma::load_matrix_sync(fa, a, a_row ? K : M);
  wmma::load_matrix_sync(fb, b, b_row ? N : K);
  wmma::load_matrix_sync(fc, c, N, wmma::mem_row_major);
  wmma::mma_sync(fc, fa, fb, fc);
  wmma::store_matrix_sync(d, fc, N, wmma::mem_row_major);
}

template <class T> T from_int(int v) { return static_cast<T>(v); }
template <> __half from_int<__half>(int v) { return __int2half_rn(v); }
template <> __nv_bfloat16 from_int<__nv_bfloat16>(int v) { return __int2bfloat16_rn(v); }
template <class T> double to_double(T v) { return static_cast<double>(v); }
template <> double to_double<__half>(__half v) { return __half2float(v); }

// Runs one configuration. `Storage` is the element type in memory (TA for
// most; float for tf32, whose fragments load from floats).
template <int M, int N, int K, class TA, class Storage, class TC, class LA, class LB>
void run(const char* name, bool unsigned_ab = false) {
  constexpr bool a_row = std::is_same_v<LA, wmma::row_major>;
  constexpr bool b_row = std::is_same_v<LB, wmma::row_major>;
  std::vector<Storage> ha(M * K), hb(K * N);
  std::vector<TC> hc(M * N), hd(M * N);
  std::vector<double> want(M * N);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k) ha[a_row ? i * K + k : k * M + i] = from_int<Storage>(val(0, i, k, unsigned_ab));
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j) hb[b_row ? k * N + j : j * K + k] = from_int<Storage>(val(1, k, j, unsigned_ab));
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      hc[i * N + j] = from_int<TC>(val(2, i, j, false));
      double s = val(2, i, j, false);
      for (int k = 0; k < K; ++k) s += double(val(0, i, k, unsigned_ab)) * val(1, k, j, unsigned_ab);
      want[i * N + j] = s;
    }
  Storage *da, *db;
  TC *dc, *dd;
  cudaMalloc(&da, ha.size() * sizeof(Storage));
  cudaMalloc(&db, hb.size() * sizeof(Storage));
  cudaMalloc(&dc, hc.size() * sizeof(TC));
  cudaMalloc(&dd, hd.size() * sizeof(TC));
  cudaMemcpy(da, ha.data(), ha.size() * sizeof(Storage), cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb.data(), hb.size() * sizeof(Storage), cudaMemcpyHostToDevice);
  cudaMemcpy(dc, hc.data(), hc.size() * sizeof(TC), cudaMemcpyHostToDevice);
  mma_kernel<M, N, K, TA, Storage, TC, LA, LB><<<1, 32>>>(da, db, dc, dd);
  const cudaError_t e = cudaDeviceSynchronize();
  cudaMemcpy(hd.data(), dd, hd.size() * sizeof(TC), cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int i = 0; i < M * N; ++i) bad += to_double(hd[i]) != want[i];
  std::printf("%-44s %s, %d of %d wrong\n", name, cudaGetErrorString(e), bad, M * N);
  g_fail += bad || e != cudaSuccess;
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  cudaFree(dd);
}

// Sub-byte and single-bit fragments pack their elements, low bits first.
template <class Precision, int K>
__global__ void subbyte_kernel(const void* a, const void* b, const int* c, int* d, bool and_op) {
  wmma::fragment<wmma::matrix_a, 8, 8, K, Precision, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 8, 8, K, Precision, wmma::col_major> fb;
  wmma::fragment<wmma::accumulator, 8, 8, K, int> fc;
  wmma::load_matrix_sync(fa, a, K);
  wmma::load_matrix_sync(fb, b, K);
  wmma::load_matrix_sync(fc, c, 8, wmma::mem_row_major);
  if constexpr (std::is_same_v<Precision, wmma::experimental::precision::b1>) {
    if (and_op) wmma::bmma_sync(fc, fa, fb, fc, wmma::experimental::bmmaBitOpAND);
    else wmma::bmma_sync(fc, fa, fb, fc, wmma::experimental::bmmaBitOpXOR);
  } else {
    wmma::mma_sync(fc, fa, fb, fc);
  }
  wmma::store_matrix_sync(d, fc, 8, wmma::mem_row_major);
}

template <class Precision, int K, int Bits>
void run_subbyte(const char* name, bool is_signed, bool and_op = false) {
  // A is 8 x K row-major, B is K x 8 column-major: both are 8 rows of K
  // packed elements.
  std::vector<uint8_t> ha(8 * K * Bits / 8, 0), hb(8 * K * Bits / 8, 0);
  std::vector<int> av(8 * K), bv(8 * K);
  for (int r = 0; r < 8; ++r)
    for (int k = 0; k < K; ++k) {
      const int x = Bits == 1 ? (r * 3 + k * 5) % 7 < 3 : val(0, r, k, !is_signed);
      const int y = Bits == 1 ? (r * 5 + k * 3) % 5 < 2 : val(1, k, r, !is_signed);
      av[r * K + k] = x;
      bv[r * K + k] = y;
      const int bit = (r * K + k) * Bits;
      ha[bit / 8] |= uint8_t((x & ((1 << Bits) - 1)) << (bit % 8));
      hb[bit / 8] |= uint8_t((y & ((1 << Bits) - 1)) << (bit % 8));
    }
  std::vector<int> hc(64), hd(64), want(64);
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) {
      hc[i * 8 + j] = val(2, i, j, false);
      int s = hc[i * 8 + j];
      for (int k = 0; k < K; ++k) {
        const int a = av[i * K + k], b = bv[j * K + k];
        s += Bits == 1 ? (and_op ? (a & b) : (a ^ b)) : a * b;
      }
      want[i * 8 + j] = s;
    }
  void *da, *db;
  int *dc, *dd;
  cudaMalloc(&da, ha.size());
  cudaMalloc(&db, hb.size());
  cudaMalloc(&dc, 256);
  cudaMalloc(&dd, 256);
  cudaMemcpy(da, ha.data(), ha.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb.data(), hb.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dc, hc.data(), 256, cudaMemcpyHostToDevice);
  subbyte_kernel<Precision, K><<<1, 32>>>(da, db, dc, dd, and_op);
  const cudaError_t e = cudaDeviceSynchronize();
  cudaMemcpy(hd.data(), dd, 256, cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int i = 0; i < 64; ++i) bad += hd[i] != want[i];
  std::printf("%-44s %s, %d of 64 wrong\n", name, cudaGetErrorString(e), bad);
  g_fail += bad || e != cudaSuccess;
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  cudaFree(dd);
}

int main() {
  using R = wmma::row_major;
  using C = wmma::col_major;
  run<16, 16, 16, __half, __half, float, R, C>("f16, f32 acc, 16x16x16 row.col");
  run<16, 16, 16, __half, __half, __half, R, R>("f16, f16 acc, 16x16x16 row.row");
  run<16, 16, 16, __half, __half, __half, C, C>("f16, f16 acc, 16x16x16 col.col");
  run<32, 8, 16, __half, __half, float, R, C>("f16, f32 acc, 32x8x16 row.col");
  run<8, 32, 16, __half, __half, float, C, R>("f16, f32 acc, 8x32x16 col.row");
  run<32, 8, 16, __half, __half, __half, R, R>("f16, f16 acc, 32x8x16 row.row");
  run<8, 32, 16, __half, __half, __half, C, C>("f16, f16 acc, 8x32x16 col.col");
  run<16, 16, 16, __nv_bfloat16, __nv_bfloat16, float, R, C>("bf16, 16x16x16 row.col");
  run<32, 8, 16, __nv_bfloat16, __nv_bfloat16, float, C, R>("bf16, 32x8x16 col.row");
  run<8, 32, 16, __nv_bfloat16, __nv_bfloat16, float, R, R>("bf16, 8x32x16 row.row");
  run<16, 16, 8, wmma::precision::tf32, float, float, R, C>("tf32, 16x16x8 row.col");
  run<16, 16, 16, signed char, signed char, int, R, C>("s8, 16x16x16 row.col");
  run<32, 8, 16, signed char, signed char, int, C, R>("s8, 32x8x16 col.row");
  run<8, 32, 16, unsigned char, unsigned char, int, R, R>("u8, 8x32x16 row.row", true);
  run<16, 16, 16, unsigned char, unsigned char, int, C, C>("u8, 16x16x16 col.col", true);
  run<8, 8, 4, double, double, double, R, C>("f64, 8x8x4 row.col");
  run<8, 8, 4, double, double, double, C, R>("f64, 8x8x4 col.row");
  run_subbyte<wmma::experimental::precision::s4, 32, 4>("s4, 8x8x32 row.col", true);
  run_subbyte<wmma::experimental::precision::u4, 32, 4>("u4, 8x8x32 row.col", false);
  run_subbyte<wmma::experimental::precision::b1, 128, 1>("b1 xor.popc, 8x8x128 row.col", false, false);
  run_subbyte<wmma::experimental::precision::b1, 128, 1>("b1 and.popc, 8x8x128 row.col", false, true);
  std::printf("%s\n", g_fail ? "FAIL" : "PASS");
  return g_fail ? 1 : 0;
}
