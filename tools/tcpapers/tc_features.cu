// The numerical features of a matrix multiplier that the tensor-core papers list, measured by our own probes on whatever device the
// CUDA runtime gives it (a card, or the simulator). It prints one line a measurement:
//
//     <format> <feature> <value>
//
// Formats: f16 (f32 accumulator), f16h (f16 accumulator), bf16, tf32. Features, with the paper that names them:
//   in_subnormal        a subnormal input is read (Fasi et al. 2021, A1-A3: not flushed)              value: the result's bits
//   acc_subnormal32     an f32 subnormal accumulator survives an addition of zero products            (Fasi A3)
//   out_subnormal       a subnormal result is returned (f32: bf16 / tf32; f16h: binary16)             (Fasi A4, A5)
//   product_exact       (1 + 2^-m)^2, m the format's fraction bits, comes out exact                   (Fasi B1)
//   acc_wide            a sum beyond binary16's range is not rounded to binary16 in f32 mode          (Fasi B3)
//   align_max_j         the largest j in 20..40 for which 1 + 2^-j - 1 is not lost: bits kept below   (Fasi D1, Khattak and Mikaitis 2025:
//                       the leading bit of the largest term. 24 means one extra bit, 25 two.           extra alignment bits)
//   align_trunc         (1 + 1.5 * 2^-24) - 1: 2^-24 if the alignment truncates, 2^-23 if it rounds up, 1.5 * 2^-24 if it keeps the bit
//   fma_size            the number of products added in one step: n minus the ones that survive when +M and -M are terms 0 and 1
//                       in a dot product of 32 ones (16 for tf32); 2 is a sequential sum (Khattak and Mikaitis: effective FMA size)
//   round_pos, round_neg  the final rounding of 1 + 2^-23 + 2^-24 (+ or -): the next float toward zero (truncation) or the even one
//   carry_sum           sixteen equal products (1 + 2^-m) * (1 - 2^-(m+1)) summed: what a wide adder with its carry bits makes of them
//                       (Fasi D4: extra bits for the carry out)
//
// "value" is bits in hex. Every product's two inputs are exactly representable in the format.
//   tc_features [--format f16|f16h|bf16|tf32]...
// Build: nvcc -std=c++17 -arch=sm_<NN> -cudart shared tc_features.cu -o tc_features   (an sm_75 build: -DNO80 leaves out bf16 and tf32)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

template <typename T> __device__ T to_t(float x);
template <> __device__ half to_t<half>(float x) { return __float2half(x); }
#ifndef NO80
template <> __device__ __nv_bfloat16 to_t<__nv_bfloat16>(float x) { return __float2bfloat16(x); }
#endif

// n products (n a multiple of 16; row 0 of A, column 0 of B), accumulator c; result in out[0] (f32) or as a float holding binary16.
template <typename T, bool HACC>
__global__ void kdot(const float* a, const float* b, float c, int n, float* out) {
  extern __shared__ __align__(128) unsigned char sm[];
  T* A = (T*)sm; T* B = A + 16 * n;
  typedef typename std::conditional<HACC, half, float>::type ACC;
  ACC* C = (ACC*)(B + 16 * n);
  for (int i = threadIdx.x; i < 16 * n; i += 32) { A[i] = to_t<T>(0.f); B[i] = to_t<T>(0.f); }
  for (int i = threadIdx.x; i < 256; i += 32) C[i] = (ACC)0;
  __syncwarp();
  for (int k = threadIdx.x; k < n; k += 32) { A[k] = to_t<T>(a[k]); B[k] = to_t<T>(b[k]); }   // A[0][k] row-major ld n; B col 0 = B[k]
  if (threadIdx.x == 0) C[0] = (ACC)c;
  __syncwarp();
  wmma::fragment<wmma::accumulator, 16, 16, 16, ACC> fc;
  wmma::load_matrix_sync(fc, C, 16, wmma::mem_row_major);
  for (int k0 = 0; k0 < n; k0 += 16) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> fb;
    wmma::load_matrix_sync(fa, A + k0, n); wmma::load_matrix_sync(fb, B + k0, n);
    wmma::mma_sync(fc, fa, fb, fc);
  }
  wmma::store_matrix_sync(C, fc, 16, wmma::mem_row_major);
  __syncwarp();
  if (threadIdx.x == 0) out[0] = (float)C[0];
}
#ifndef NO80
__global__ void kdot_tf32(const float* a, const float* b, float c, int n, float* out) {
  extern __shared__ __align__(128) unsigned char sm[];
  float* A = (float*)sm; float* B = A + 16 * n; float* C = B + 16 * n;
  for (int i = threadIdx.x; i < 16 * n; i += 32) { A[i] = 0.f; B[i] = 0.f; }
  for (int i = threadIdx.x; i < 256; i += 32) C[i] = 0.f;
  __syncwarp();
  for (int k = threadIdx.x; k < n; k += 32) { A[k] = a[k]; B[k] = b[k]; }
  if (threadIdx.x == 0) C[0] = c;
  __syncwarp();
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> fc;
  wmma::load_matrix_sync(fc, C, 16, wmma::mem_row_major);
  for (int k0 = 0; k0 < n; k0 += 8) {
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> fb;
    wmma::load_matrix_sync(fa, A + k0, n); wmma::load_matrix_sync(fb, B + k0, n);
    for (int i = 0; i < fa.num_elements; ++i) fa.x[i] = wmma::__float_to_tf32(fa.x[i]);
    for (int i = 0; i < fb.num_elements; ++i) fb.x[i] = wmma::__float_to_tf32(fb.x[i]);
    wmma::mma_sync(fc, fa, fb, fc);
  }
  wmma::store_matrix_sync(C, fc, 16, wmma::mem_row_major);
  __syncwarp();
  if (threadIdx.x == 0) out[0] = C[0];
}
#endif

enum Fmt { F16, F16H, BF16, TF32 };
static float *da, *db, *dout;
// One dot product: products pa[i] * pb[i] (i < np), the rest zero, n total slots, accumulator c.
static float dot(Fmt f, const std::vector<float>& pa, const std::vector<float>& pb, float c, int n) {
  std::vector<float> ha(n, 0.f), hb(n, 0.f);
  for (size_t i = 0; i < pa.size(); ++i) { ha[i] = pa[i]; hb[i] = pb[i]; }
  CK(cudaMemcpy(da, ha.data(), n * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db, hb.data(), n * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(dout, 0, 4));
  switch (f) {
    case F16: kdot<half, false><<<1, 32, 2 * 16 * n * 2 + 1024>>>(da, db, c, n, dout); break;
    case F16H: kdot<half, true><<<1, 32, 2 * 16 * n * 2 + 512>>>(da, db, c, n, dout); break;
#ifndef NO80
    case BF16: kdot<__nv_bfloat16, false><<<1, 32, 2 * 16 * n * 2 + 1024>>>(da, db, c, n, dout); break;
    case TF32: kdot_tf32<<<1, 32, 2 * 16 * n * 4 + 1024>>>(da, db, c, n, dout); break;
#else
    default: return NAN;
#endif
  }
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  float r; CK(cudaMemcpy(&r, dout, 4, cudaMemcpyDeviceToHost)); return r;
}
static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static void line(const char* fmt, const char* feat, float v) { printf("%s %s %08x\n", fmt, feat, bits(v)); }
static void linei(const char* fmt, const char* feat, int v) { printf("%s %s %d\n", fmt, feat, v); }

int main(int argc, char** argv) {
  std::vector<std::string> want;
  for (int i = 1; i + 1 < argc; i += 2) if (!strcmp(argv[i], "--format")) want.push_back(argv[i + 1]);
  CK(cudaMalloc(&da, 64 * 4)); CK(cudaMalloc(&db, 64 * 4)); CK(cudaMalloc(&dout, 4));
  struct FI { Fmt f; const char* name; int mant; float minsub_a, minsub_b; int n; };
  // minsub: two inputs whose product is the smallest subnormal the format holds on the input side (as one input times one)
  FI fs[] = { {F16, "f16", 10, ldexpf(1, -24), 1.f, 32}, {F16H, "f16h", 10, ldexpf(1, -24), 1.f, 32},
              {BF16, "bf16", 7, ldexpf(1, -133), 1.f, 32}, {TF32, "tf32", 10, ldexpf(1, -136), 1.f, 16} };
  for (const FI& fi : fs) {
    if (!want.empty()) { bool w = false; for (auto& s : want) w |= s == fi.name; if (!w) continue; }
#ifdef NO80
    if (fi.f == BF16 || fi.f == TF32) continue;
#endif
    const char* nm = fi.name; const int n = fi.n; const bool h = fi.f == F16H;
    // subnormals
    line(nm, "in_subnormal", dot(fi.f, {fi.minsub_a}, {fi.minsub_b}, 0.f, n));
    if (!h) line(nm, "acc_subnormal32", dot(fi.f, {0.f}, {0.f}, ldexpf(1, -149), n));
    if (h) line(nm, "out_subnormal", dot(fi.f, {ldexpf(1, -14)}, {ldexpf(1, -10)}, 0.f, n));
    else if (fi.f != F16) line(nm, "out_subnormal", dot(fi.f, {ldexpf(1, -126)}, {-1.f}, 1.5f * ldexpf(1, -126), n));
    // products
    { float e = 1.f + ldexpf(1, -fi.mant); if (!h) line(nm, "product_exact", dot(fi.f, {e}, {e}, 0.f, n)); }
    if (fi.f == F16) line(nm, "acc_wide", dot(fi.f, {65504.f, 65504.f}, {65504.f, 65504.f}, 0.f, n));
    if (h) { /* f16 accumulator: the extra-bit probes below need an f32 one */ }
    // extra alignment bits: the largest j for which (1 + 2^-j) - 1 is not lost
    if (!h) {
      int jmax = 19;
      for (int j = 20; j <= 40; ++j) {
        int ja = (j + 1) / 2, jb = j / 2;        // 2^-ja * 2^-jb = 2^-j, each inside the format's range
        if (fi.f == F16 && ja > 14) break;       // f16 normal range: the smallest normal input is 2^-14
        float r = dot(fi.f, {1.f, ldexpf(1, -ja)}, {1.f, ldexpf(1, -jb)}, -1.f, n);
        if (r != 0.f) jmax = j; else break;
      }
      linei(nm, "align_max_j", jmax);
      line(nm, "align_trunc", dot(fi.f, {1.f, 1.5f * ldexpf(1, -12)}, {1.f, ldexpf(1, -12)}, -1.f, n));
      // the size of one step: +M term 0, -M term 1, ones elsewhere
      std::vector<float> pa(n, 1.f), pb(n, 1.f); pa[0] = 32768.f; pb[0] = 32768.f; pa[1] = -32768.f; pb[1] = 32768.f;
      int fma = n - (int)lrintf(dot(fi.f, pa, pb, 0.f, n));
      linei(nm, "fma_size", fma);
      // final rounding: 1 + 2^-23 + 2^-24 is half way between two floats
      line(nm, "round_pos", dot(fi.f, {1.f, ldexpf(1, -12), ldexpf(1, -12)}, {1.f, ldexpf(1, -11), ldexpf(1, -12)}, 0.f, n));
      line(nm, "round_neg", dot(fi.f, {-1.f, -ldexpf(1, -12), -ldexpf(1, -12)}, {1.f, ldexpf(1, -11), ldexpf(1, -12)}, 0.f, n));
      // carry: sixteen equal products: exact in any adder with enough carry bits
      std::vector<float> ca(16, 1.f), cb(16, 1.f);
      for (int i = 0; i < 16 && i < n; ++i) { ca[i] = 1.f + ldexpf(1, -fi.mant); cb[i] = 1.f - ldexpf(1, -fi.mant - 1); }
      line(nm, "carry_sum", dot(fi.f, ca, cb, 0.f, n));
    } else {
      // f16 result: 1 + 2^-11 + 2^-12 is above half way between two binary16 values (round to nearest: 1 + 2^-10; truncation: 1)
      line(nm, "round_pos", dot(fi.f, {1.f, ldexpf(1, -6), ldexpf(1, -6)}, {1.f, ldexpf(1, -5), ldexpf(1, -6)}, 0.f, n));
      line(nm, "round_neg", dot(fi.f, {-1.f, -ldexpf(1, -6), -ldexpf(1, -6)}, {1.f, ldexpf(1, -5), ldexpf(1, -6)}, 0.f, n));
    }
  }
  return 0;
}
