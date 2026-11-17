// Runs the test vectors of a tensor-core validation set through wmma on whatever device the CUDA runtime gives it (a card, or the
// simulator) and reports how many results are bit-equal to the recorded ones.
//
// The vectors are the ones published with Khattak and Mikaitis, "Accurate Models of NVIDIA Tensor Cores", arXiv 2512.07004
// (github.com/north-numerical-computing/MATLAB-tensor-core, BSD-2-Clause, model_validation/): 5,000 dot products a card and format,
// recorded on V100, A100, A2, Ada, L40S, H100, H200 and B200 cards. tools/tcpapers/khattak-fetch.sh downloads them and writes them in
// the one-line-a-vector form this program reads. Our own program; none of the paper's code is used.
//
//   khattak_vectors <fmt: fp16|bf16|tf32> <K> <out: fp32|fp16> <file> [--show N]
//
// A line of <file>:   a_0 .. a_{K-1}  b_0 .. b_{K-1}  c  d      (every token 8 hex digits, the bits of the float32 holding the value;
//                     d is the card's result, c the accumulator, both as float32; for out=fp16 the result is read as a float32 that
//                     holds a binary16 value, and c is rounded to binary16 on the way in, as the paper's own program does).
// The dot product is a wmma 16x16xk with the vector in row 0 of A and column 0 of B, zero elsewhere, and c in the accumulator.
// Prints  <fmt> K=<K> out=<out> vectors=<n> mismatches=<m>  and the first mismatches. Exit status 0 whatever the count.
// Build: nvcc -std=c++17 -arch=sm_<NN> -cudart shared khattak_vectors.cu -o khattak_vectors
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

template <typename T> __device__ T to_t(float x);
template <> __device__ half to_t<half>(float x) { return __float2half(x); }
template <> __device__ __nv_bfloat16 to_t<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// ---- f16 / bf16 inputs, f32 accumulator ----
template <typename T>
__global__ void k16(const float* a, const float* b, const float* c, float* d, int K, int N) {
  extern __shared__ __align__(128) unsigned char sm[];
  T* A = (T*)sm; T* B = A + 256; float* C = (float*)(B + 256);
  int p = blockIdx.x; if (p >= N) return;
  for (int i = threadIdx.x; i < 256; i += 32) { A[i] = to_t<T>(0.f); B[i] = to_t<T>(0.f); C[i] = 0.f; }
  __syncwarp();
  if (threadIdx.x < K) { A[threadIdx.x] = to_t<T>(a[(size_t)p * K + threadIdx.x]); B[threadIdx.x] = to_t<T>(b[(size_t)p * K + threadIdx.x]); }
  if (threadIdx.x == 0) C[0] = c[p];
  __syncwarp();
  wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> fb;   // col-major: B[col * 16 + k]: column 0 is B[0..15]
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
  wmma::load_matrix_sync(fa, A, 16); wmma::load_matrix_sync(fb, B, 16); wmma::load_matrix_sync(fc, C, 16, wmma::mem_row_major);
  wmma::mma_sync(fc, fa, fb, fc);
  wmma::store_matrix_sync(C, fc, 16, wmma::mem_row_major);
  __syncwarp();
  if (threadIdx.x == 0) d[p] = C[0];
}
// ---- f16 inputs, f16 accumulator ----
__global__ void k16h(const float* a, const float* b, const float* c, float* d, int K, int N) {
  extern __shared__ __align__(128) unsigned char sm[];
  half* A = (half*)sm; half* B = A + 256; half* C = B + 256;
  int p = blockIdx.x; if (p >= N) return;
  for (int i = threadIdx.x; i < 256; i += 32) { A[i] = __float2half(0.f); B[i] = __float2half(0.f); C[i] = __float2half(0.f); }
  __syncwarp();
  if (threadIdx.x < K) { A[threadIdx.x] = __float2half(a[(size_t)p * K + threadIdx.x]); B[threadIdx.x] = __float2half(b[(size_t)p * K + threadIdx.x]); }
  if (threadIdx.x == 0) C[0] = __float2half(c[p]);
  __syncwarp();
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> fc;
  wmma::load_matrix_sync(fa, A, 16); wmma::load_matrix_sync(fb, B, 16); wmma::load_matrix_sync(fc, C, 16, wmma::mem_row_major);
  wmma::mma_sync(fc, fa, fb, fc);
  wmma::store_matrix_sync(C, fc, 16, wmma::mem_row_major);
  __syncwarp();
  if (threadIdx.x == 0) d[p] = __half2float(C[0]);
}
// ---- tf32 inputs (wmma 16x16x8), f32 accumulator ----
__global__ void kt(const float* a, const float* b, const float* c, float* d, int K, int N) {
  extern __shared__ __align__(128) unsigned char sm[];
  float* A = (float*)sm; float* B = A + 256; float* C = B + 256;
  int p = blockIdx.x; if (p >= N) return;
  for (int i = threadIdx.x; i < 256; i += 32) { A[i] = 0.f; B[i] = 0.f; C[i] = 0.f; }
  __syncwarp();
  if (threadIdx.x < K) { A[threadIdx.x] = a[(size_t)p * K + threadIdx.x]; B[threadIdx.x] = b[(size_t)p * K + threadIdx.x]; }
  if (threadIdx.x == 0) C[0] = c[p];
  __syncwarp();
  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> fc;
  wmma::load_matrix_sync(fa, A, 16); wmma::load_matrix_sync(fb, B, 16); wmma::load_matrix_sync(fc, C, 16, wmma::mem_row_major);
  for (int i = 0; i < fa.num_elements; ++i) fa.x[i] = wmma::__float_to_tf32(fa.x[i]);
  for (int i = 0; i < fb.num_elements; ++i) fb.x[i] = wmma::__float_to_tf32(fb.x[i]);
  wmma::mma_sync(fc, fa, fb, fc);
  wmma::store_matrix_sync(C, fc, 16, wmma::mem_row_major);
  __syncwarp();
  if (threadIdx.x == 0) d[p] = C[0];
}

static float bits2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: %s fp16|bf16|tf32 K fp32|fp16 file [--show N]\n", argv[0]); return 2; }
  std::string fmt = argv[1]; int K = atoi(argv[2]); std::string out = argv[3];
  int show = 3; if (argc > 6 && !strcmp(argv[5], "--show")) show = atoi(argv[6]);
  if (K < 1 || K > 16 || (fmt == "tf32" && K > 8)) { fprintf(stderr, "bad K\n"); return 2; }
  FILE* f = fopen(argv[4], "r"); if (!f) { perror(argv[4]); return 2; }
  std::vector<float> a, b, c, d; char tok[64];
  std::vector<uint32_t> row;
  int ntok = 2 * K + 2;
  while (fscanf(f, "%63s", tok) == 1) {
    row.push_back((uint32_t)strtoul(tok, nullptr, 16));
    if ((int)row.size() == ntok) {
      for (int k = 0; k < K; ++k) a.push_back(bits2f(row[k]));
      for (int k = 0; k < K; ++k) b.push_back(bits2f(row[K + k]));
      c.push_back(bits2f(row[2 * K])); d.push_back(bits2f(row[2 * K + 1])); row.clear();
    }
  }
  fclose(f);
  int N = (int)d.size();
  if (!row.empty() || N == 0) { fprintf(stderr, "file does not hold whole vectors of %d tokens\n", ntok); return 2; }
  float *da, *db, *dc, *dd; CK(cudaMalloc(&da, a.size() * 4)); CK(cudaMalloc(&db, b.size() * 4)); CK(cudaMalloc(&dc, N * 4)); CK(cudaMalloc(&dd, N * 4));
  CK(cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dc, c.data(), N * 4, cudaMemcpyHostToDevice)); CK(cudaMemset(dd, 0, N * 4));
  if (fmt == "fp16" && out == "fp32") k16<half><<<N, 32, 2 * 256 * 2 + 1024>>>(da, db, dc, dd, K, N);
  else if (fmt == "bf16" && out == "fp32") k16<__nv_bfloat16><<<N, 32, 2 * 256 * 2 + 1024>>>(da, db, dc, dd, K, N);
  else if (fmt == "fp16" && out == "fp16") k16h<<<N, 32, 3 * 256 * 2>>>(da, db, dc, dd, K, N);
  else if (fmt == "tf32" && out == "fp32") kt<<<N, 32, 3 * 256 * 4>>>(da, db, dc, dd, K, N);
  else { fprintf(stderr, "unsupported combination\n"); return 2; }
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<float> r(N); CK(cudaMemcpy(r.data(), dd, N * 4, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < N; ++i) {
    if (f2bits(r[i]) == f2bits(d[i])) continue;
    if (bad < show) printf("  vector %d: card %08x  here %08x\n", i, f2bits(d[i]), f2bits(r[i]));
    ++bad;
  }
  printf("%s K=%d out=%s vectors=%d mismatches=%d\n", fmt.c_str(), K, out.c_str(), N, bad);
  return 0;
}
