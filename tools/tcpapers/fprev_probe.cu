// Reveals the order in which a GPU kernel adds a dot product's terms (its summation tree), by the method of
//   P. Xie, Y. Gao, Y. Wang, J. Xue, "Revealing Floating-Point Accumulation Orders in Software/Hardware Implementations",
//   USENIX ATC 2025, arXiv 2411.00442 (FPRev, github.com/peichenxie/FPRev, MIT).
// This is our own program for the published method; none of FPRev's code is used.
//
// The method. n terms are added. For a pair (i, j) set term i to +M and term j to -M, every other term to 1, with M so large that a
// 1 added into a sum that holds M is lost (swamped). The cancellation M - M leaves the count of ones that were added after the two
// met, so n - 16 * result = L(i, j) is the number of terms in the smallest sub-sum that contains both i and j. The matrix L fixes the
// tree: for a node over a set S the children are the connected pieces of S under "L(i, j) < |S|". A fused multi-term addition
// (a tensor core adding 8 or 16 products in one step) comes out as one node with more than two children, which is how the
// method sees it.
//
// A term is a product a_k * b_k. A "one" is 2^-2 * 2^-2 = 2^-4 (the unit of the count below), M is 65504 * 65504 (+M) and -65504 *
// 65504 (-M): 65504 is the largest binary16, so M is about 2^32 for every format here. M has to be large enough that the running
// count of ones is lost beside it: a tensor core that keeps G bits below the largest term's leading bit loses what is below
// 2^(top - G), top being 31 for a binary16 M, and the count (at most (n - 2) * 2^-4) must stay under that. With ones of 1 and
// M = 2^15 * 2^15 (our first choice) a count of 32 or more survived on a G = 26 adder and drew a wrong tree for n = 64.
//
// Kernels: plain f32 sums (a loop, a warp shuffle tree, a halving tree in shared memory), wmma f16 / bf16 / tf32 with an f32
// accumulator, and cuBLAS (sgemm, GemmEx f16 and bf16 and tf32 with f32 compute).
//
//   fprev_probe [--n N] [name ...]   one line a kernel (--n: the number of terms of the wmma and cuBLAS kernels, a multiple of 16):   <name> n=<n> <tree>   (or  ... NONINTEGER <count> when a result was not a count)
// Build: nvcc -std=c++17 -arch=sm_86 -cudart shared fprev_probe.cu -o fprev_probe -lcublas   (-DNO_CUBLAS leaves cuBLAS out)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#ifndef NO_CUBLAS
#include <cublas_v2.h>
#endif
using namespace nvcuda;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

// ---- kernels: probe p reads xa[p*n + k], xb[p*n + k] and writes out[p] ----
__global__ void k_loop(const float* xa, const float* xb, float* out, int n, int P) {
  int p = blockIdx.x * blockDim.x + threadIdx.x; if (p >= P) return;
  float acc = 0.f;
  for (int k = 0; k < n; ++k) acc += xa[p * n + k] * xb[p * n + k];
  out[p] = acc;
}
__global__ void k_fma_loop(const float* xa, const float* xb, float* out, int n, int P) {
  int p = blockIdx.x * blockDim.x + threadIdx.x; if (p >= P) return;
  float acc = 0.f;
  for (int k = 0; k < n; ++k) acc = fmaf(xa[p * n + k], xb[p * n + k], acc);
  out[p] = acc;
}
__global__ void k_shfl(const float* xa, const float* xb, float* out, int n, int P) {   // n = 32, one warp a probe
  int p = blockIdx.x; int t = threadIdx.x;
  float v = xa[p * n + t] * xb[p * n + t];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if (t == 0) out[p] = v;
}
__global__ void k_halving(const float* xa, const float* xb, float* out, int n, int P) {   // n threads, shared memory halving
  extern __shared__ float s[];
  int p = blockIdx.x; int t = threadIdx.x;
  s[t] = xa[p * n + t] * xb[p * n + t];
  __syncthreads();
  for (int st = n / 2; st > 0; st >>= 1) { if (t < st) s[t] += s[t + st]; __syncthreads(); }
  if (t == 0) out[p] = s[0];
}
template <typename T> __device__ T cvt(float);
template <> __device__ half cvt<half>(float x) { return __float2half(x); }
template <> __device__ __nv_bfloat16 cvt<__nv_bfloat16>(float x) { return __float2bfloat16(x); }
// wmma m16n16k16 (f16, bf16): every row of A is the probe's a vector, every column of B its b vector; D[0][0] is the sum.
template <typename T>
__global__ void k_wmma16(const float* xa, const float* xb, float* out, int n, int P) {   // one warp a probe
  extern __shared__ __align__(128) unsigned char smraw[];
  T* A = (T*)smraw; T* B = A + 16 * n; float* C = (float*)(B + 16 * n);
  int p = blockIdx.x;
  for (int i = threadIdx.x; i < 16 * n; i += 32) {
    int r = i / n, k = i % n;
    A[i] = cvt<T>(xa[p * n + k]);                 // row-major, ld n
    int c = i / n; B[c * n + k] = cvt<T>(xb[p * n + k]);   // col-major, ld n
    (void)r;
  }
  __syncwarp();
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc; wmma::fill_fragment(acc, 0.f);
  for (int k0 = 0; k0 < n; k0 += 16) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> fb;
    wmma::load_matrix_sync(fa, A + k0, n); wmma::load_matrix_sync(fb, B + k0, n);
    wmma::mma_sync(acc, fa, fb, acc);
  }
  wmma::store_matrix_sync(C, acc, 16, wmma::mem_row_major);
  __syncwarp();
  if (threadIdx.x == 0) out[p] = C[0];
}
__global__ void k_wmma_tf32(const float* xa, const float* xb, float* out, int n, int P) {
  extern __shared__ __align__(128) unsigned char smraw[];
  float* A = (float*)smraw; float* B = A + 16 * n; float* C = B + 16 * n;
  int p = blockIdx.x;
  for (int i = threadIdx.x; i < 16 * n; i += 32) {
    int k = i % n, c = i / n;
    A[i] = xa[p * n + k]; B[c * n + k] = xb[p * n + k];
  }
  __syncwarp();
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc; wmma::fill_fragment(acc, 0.f);
  for (int k0 = 0; k0 < n; k0 += 8) {
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> fb;
    wmma::load_matrix_sync(fa, A + k0, n); wmma::load_matrix_sync(fb, B + k0, n);
    for (int i = 0; i < fa.num_elements; ++i) fa.x[i] = wmma::__float_to_tf32(fa.x[i]);
    for (int i = 0; i < fb.num_elements; ++i) fb.x[i] = wmma::__float_to_tf32(fb.x[i]);
    wmma::mma_sync(acc, fa, fb, acc);
  }
  wmma::store_matrix_sync(C, acc, 16, wmma::mem_row_major);
  __syncwarp();
  if (threadIdx.x == 0) out[p] = C[0];
}
// gemm inputs: P batches of A (16 x n, column-major, every row the a vector) and B (n x 16, column-major, every column the b vector)
__global__ void k_fill_gemm(const float* xa, const float* xb, float* A, float* B, int n, int P) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x; long tot = (long)P * 16 * n; if (i >= tot) return;
  int p = i / (16 * n); int r = i % (16 * n);
  int k = r / 16;            // A[k * 16 + row]
  A[i] = xa[p * n + k];
  int c = r / n, kk = r % n; // B[c * n + k]
  B[i] = xb[p * n + kk]; (void)c;
}
__global__ void k_fill_gemm_h(const float* xa, const float* xb, __half* A, __half* B, int n, int P) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x; long tot = (long)P * 16 * n; if (i >= tot) return;
  int p = i / (16 * n); int r = i % (16 * n);
  A[i] = __float2half(xa[p * n + r / 16]); B[i] = __float2half(xb[p * n + r % n]);
}
__global__ void k_fill_gemm_b(const float* xa, const float* xb, __nv_bfloat16* A, __nv_bfloat16* B, int n, int P) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x; long tot = (long)P * 16 * n; if (i >= tot) return;
  int p = i / (16 * n); int r = i % (16 * n);
  A[i] = __float2bfloat16(xa[p * n + r / 16]); B[i] = __float2bfloat16(xb[p * n + r % n]);
}
__global__ void k_pick(const float* C, float* out, int P) { int p = blockIdx.x * blockDim.x + threadIdx.x; if (p < P) out[p] = C[(long)p * 256]; }

// ---- the tree, from L ----
static std::string tree(const std::vector<std::vector<int>>& L, std::vector<int> S) {
  if (S.size() == 1) return std::to_string(S[0]);
  int s = (int)S.size();
  std::vector<int> comp(s, -1); int nc = 0;
  for (int a = 0; a < s; ++a) {
    if (comp[a] >= 0) continue;
    std::vector<int> st{a}; comp[a] = nc;
    while (!st.empty()) {
      int u = st.back(); st.pop_back();
      for (int v = 0; v < s; ++v) if (comp[v] < 0 && L[S[u]][S[v]] < s) { comp[v] = nc; st.push_back(v); }
    }
    ++nc;
  }
  if (nc == 1) return "?";   // inconsistent L: no pair has L = |S|
  std::vector<std::vector<int>> kids(nc);
  for (int a = 0; a < s; ++a) kids[comp[a]].push_back(S[a]);
  std::vector<std::string> parts;
  for (auto& k : kids) parts.push_back(tree(L, k));
  // consecutive single leaves are written a-b
  std::string r = "(";
  for (size_t i = 0; i < kids.size();) {
    if (kids[i].size() == 1) {
      size_t j = i; while (j + 1 < kids.size() && kids[j + 1].size() == 1 && kids[j + 1][0] == kids[j][0] + 1) ++j;
      if (!r.empty() && r.back() != '(') r += " ";
      r += (j > i) ? (std::to_string(kids[i][0]) + "-" + std::to_string(kids[j][0])) : parts[i];
      i = j + 1;
    } else { if (r.back() != '(') r += " "; r += parts[i]; ++i; }
  }
  return r + ")";
}

using Runner = void (*)(const float*, const float*, float*, int, int);
struct Kern { const char* name; int n; Runner run; };
static void run_loop(const float* a, const float* b, float* o, int n, int P) { k_loop<<<(P + 63) / 64, 64>>>(a, b, o, n, P); }
static void run_fmaloop(const float* a, const float* b, float* o, int n, int P) { k_fma_loop<<<(P + 63) / 64, 64>>>(a, b, o, n, P); }
static void run_shfl(const float* a, const float* b, float* o, int n, int P) { k_shfl<<<P, 32>>>(a, b, o, n, P); }
static void run_halv(const float* a, const float* b, float* o, int n, int P) { k_halving<<<P, n, n * sizeof(float)>>>(a, b, o, n, P); }
static void run_wh(const float* a, const float* b, float* o, int n, int P) { k_wmma16<half><<<P, 32, 2 * 16 * n * 2 + 1024>>>(a, b, o, n, P); }
static void run_wb(const float* a, const float* b, float* o, int n, int P) { k_wmma16<__nv_bfloat16><<<P, 32, 2 * 16 * n * 2 + 1024>>>(a, b, o, n, P); }
static void run_wt(const float* a, const float* b, float* o, int n, int P) { k_wmma_tf32<<<P, 32, 2 * 16 * n * 4 + 1024>>>(a, b, o, n, P); }
#ifndef NO_CUBLAS
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s:%d: cublas status %d\n", __FILE__, __LINE__, (int)s_); exit(3); } } while (0)
static void gemm_common(const float* a, const float* b, float* o, int n, int P, int mode) {
  cublasHandle_t h; CB(cublasCreate(&h));
  long na = (long)P * 16 * n; float* C; CK(cudaMalloc(&C, (long)P * 256 * sizeof(float)));
  const float one = 1.f, zero = 0.f;
  if (mode == 0 || mode == 3) {
    float *A, *B; CK(cudaMalloc(&A, na * 4)); CK(cudaMalloc(&B, na * 4));
    k_fill_gemm<<<(na + 255) / 256, 256>>>(a, b, A, B, n, P);
    if (mode == 0) CB(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, 16, 16, n, &one, A, 16, 16L * n, B, n, 16L * n, &zero, C, 16, 256, P));
    else CB(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, 16, 16, n, &one, A, CUDA_R_32F, 16, 16L * n, B, CUDA_R_32F, n, 16L * n, &zero, C, CUDA_R_32F, 16, 256, P, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT));
    CK(cudaDeviceSynchronize()); cudaFree(A); cudaFree(B);
  } else if (mode == 1) {
    __half *A, *B; CK(cudaMalloc(&A, na * 2)); CK(cudaMalloc(&B, na * 2));
    k_fill_gemm_h<<<(na + 255) / 256, 256>>>(a, b, A, B, n, P);
    CB(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, 16, 16, n, &one, A, CUDA_R_16F, 16, 16L * n, B, CUDA_R_16F, n, 16L * n, &zero, C, CUDA_R_32F, 16, 256, P, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CK(cudaDeviceSynchronize()); cudaFree(A); cudaFree(B);
  } else {
    __nv_bfloat16 *A, *B; CK(cudaMalloc(&A, na * 2)); CK(cudaMalloc(&B, na * 2));
    k_fill_gemm_b<<<(na + 255) / 256, 256>>>(a, b, A, B, n, P);
    CB(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, 16, 16, n, &one, A, CUDA_R_16BF, 16, 16L * n, B, CUDA_R_16BF, n, 16L * n, &zero, C, CUDA_R_32F, 16, 256, P, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CK(cudaDeviceSynchronize()); cudaFree(A); cudaFree(B);
  }
  k_pick<<<(P + 63) / 64, 64>>>(C, o, P);
  CK(cudaDeviceSynchronize()); cudaFree(C); cublasDestroy(h);
}
static void run_sgemm(const float* a, const float* b, float* o, int n, int P) { gemm_common(a, b, o, n, P, 0); }
static void run_gh(const float* a, const float* b, float* o, int n, int P) { gemm_common(a, b, o, n, P, 1); }
static void run_gb(const float* a, const float* b, float* o, int n, int P) { gemm_common(a, b, o, n, P, 2); }
static void run_gt(const float* a, const float* b, float* o, int n, int P) { gemm_common(a, b, o, n, P, 3); }
#endif

int main(int argc, char** argv) {
  std::vector<Kern> ks = {
    {"f32_loop_sum", 32, run_loop}, {"f32_fma_loop_dot", 32, run_fmaloop}, {"f32_shfl_down_tree", 32, run_shfl},
    {"f32_smem_halving_tree", 64, run_halv},
    {"wmma_f16_f32", 64, run_wh}, {"wmma_bf16_f32", 64, run_wb}, {"wmma_tf32_f32", 32, run_wt},
#ifndef NO_CUBLAS
    {"cublas_sgemm", 64, run_sgemm}, {"cublas_gemmex_f16_f32", 64, run_gh}, {"cublas_gemmex_bf16_f32", 64, run_gb},
    {"cublas_gemmex_tf32", 64, run_gt},
#endif
  };
  int nover = 0, first = 1;
  if (argc > 2 && !strcmp(argv[1], "--n")) { nover = atoi(argv[2]); first = 3; }
  for (const Kern& kn : ks) {
    if (argc > first) { bool want = false; for (int i = first; i < argc; ++i) want |= !strcmp(argv[i], kn.name); if (!want) continue; }
    int n = kn.n;
    if (nover && (!strncmp(kn.name, "wmma_", 5) || !strncmp(kn.name, "cublas_", 7))) n = nover;
    std::vector<std::pair<int, int>> pairs;
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) pairs.push_back({i, j});
    int P = (int)pairs.size() + 1;
    std::vector<float> xa((size_t)P * n, 0.25f), xb((size_t)P * n, 0.25f);   // a one is 2^-4 (see above)
    for (size_t q = 0; q < pairs.size(); ++q) {
      int i = pairs[q].first, j = pairs[q].second;
      xa[(q + 1) * n + i] = 65504.f; xb[(q + 1) * n + i] = 65504.f;
      xa[(q + 1) * n + j] = -65504.f; xb[(q + 1) * n + j] = 65504.f;
    }
    float *da, *db, *dout; CK(cudaMalloc(&da, xa.size() * 4)); CK(cudaMalloc(&db, xb.size() * 4)); CK(cudaMalloc(&dout, P * 4));
    CK(cudaMemcpy(da, xa.data(), xa.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db, xb.data(), xb.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dout, 0, P * 4));
    kn.run(da, db, dout, n, P);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> out(P); CK(cudaMemcpy(out.data(), dout, P * 4, cudaMemcpyDeviceToHost));
    cudaFree(da); cudaFree(db); cudaFree(dout);
    std::vector<std::vector<int>> L(n, std::vector<int>(n, 0));
    int bad = 0; float firstbad = 0;
    if (out[0] * 16.f != (float)n) { printf("%s n=%d ALL_ONES_SUM %g\n", kn.name, n, out[0] * 16.f); continue; }
    for (size_t q = 0; q < pairs.size(); ++q) {
      float r = out[q + 1] * 16.f; int c = (int)lrintf(r);
      if (r != (float)c || c < 0 || c > n - 2) { if (!bad++) firstbad = r; c = 0; }
      L[pairs[q].first][pairs[q].second] = L[pairs[q].second][pairs[q].first] = n - c;
    }
    if (bad) { printf("%s n=%d NONINTEGER %d probes, first %g\n", kn.name, n, bad, firstbad); continue; }
    std::vector<int> all(n); for (int i = 0; i < n; ++i) all[i] = i;
    printf("%s n=%d %s\n", kn.name, n, tree(L, all).c_str());
  }
  return 0;
}
