// TMA's im2col mode, checked against the convolution it exists for.
//
// The descriptor is made the way CUTLASS's Hopper convolutions make theirs
// (cutlass/conv/collective/detail.hpp and cute's copy_traits_sm90_im2col.hpp):
// the activation as a flat (C, W, H, N) or (C, W, N) tensor, the box's lower
// corner at minus the padding and its upper corner at padding - (filter - 1)
// * dilation, the traversal stride the convolution's stride, one pixel per
// row of the tile and the channels along it. A load then starts at the pixel
// that output position m's filter window begins at -- lower + q * stride,
// with p and n likewise -- and its im2col offsets are the filter tap times
// the dilation, exactly as CuTe computes them.
//
// Each tile is compared with the im2col matrix worked out from the
// definition of a convolution: row m = (n, p, q), q fastest, holds the input
// pixel that filter tap (r, s) of output (p, q) reads, or zero where that
// falls in the padding. Rows past the last output are zero too (the batch
// runs out). Prints PASS on the last line.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#define CK(x)                                                                        \
  do {                                                                               \
    auto e_ = (x);                                                                   \
    if (e_ != 0) {                                                                   \
      std::printf("FAIL: %s returned %d (line %d)\n", #x, int(e_), __LINE__);        \
      std::exit(1);                                                                  \
    }                                                                                \
  } while (0)

constexpr int kC = 16;       // channels per pixel: 64 bytes of float
constexpr int kPixels = 32;  // pixels per column: the tile's rows

struct Conv {
  const char* name;
  int n, h, w;          // input: N x H x W x C (h = 1 for the 1D case)
  int r, s;             // filter
  int pad_h, pad_w, stride_h, stride_w, dil_h, dil_w;
  bool one_d;           // NWC, a rank-3 map
  __host__ __device__ int p() const { return (h + 2 * pad_h - (r - 1) * dil_h - 1) / stride_h + 1; }
  __host__ __device__ int q() const { return (w + 2 * pad_w - (s - 1) * dil_w - 1) / stride_w + 1; }
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// One block per (tile, r, s): a 32-pixel column of the im2col matrix for one
// filter tap, as a convolution mainloop loads it.
__global__ void load_columns(const __grid_constant__ CUtensorMap map, Conv cv, float* out) {
  __shared__ alignas(128) float tile[kPixels * kC];
  __shared__ alignas(8) uint64_t bar;
  const int tiles = (cv.n * cv.p() * cv.q() + kPixels - 1) / kPixels;
  const int t = blockIdx.x % tiles, tap = blockIdx.x / tiles;
  const int rr = tap / cv.s, ss = tap % cv.s;
  if (threadIdx.x == 0) {
    const int m0 = t * kPixels;
    const int q0 = m0 % cv.q(), p0 = (m0 / cv.q()) % cv.p(), n0 = m0 / (cv.q() * cv.p());
    const int c = 0;
    const int w = -cv.pad_w + q0 * cv.stride_w, h = -cv.pad_h + p0 * cv.stride_h;
    const uint16_t off_w = uint16_t(ss * cv.dil_w), off_h = uint16_t(rr * cv.dil_h);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&bar)));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&bar)),
                 "r"(kPixels * kC * 4));
    if (cv.one_d)
      asm volatile(
          "cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes"
          " [%0], [%1, {%3, %4, %5}], [%2], {%6};" ::"r"(smem_u32(tile)),
          "l"(&map), "r"(smem_u32(&bar)), "r"(c), "r"(w), "r"(n0), "h"(off_w)
          : "memory");
    else
      asm volatile(
          "cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes"
          " [%0], [%1, {%3, %4, %5, %6}], [%2], {%7, %8};" ::"r"(smem_u32(tile)),
          "l"(&map), "r"(smem_u32(&bar)), "r"(c), "r"(w), "r"(h), "r"(n0), "h"(off_w), "h"(off_h)
          : "memory");
  }
  __syncthreads();
  uint32_t done = 0;
  while (!done)
    asm volatile(
        "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
        : "=r"(done)
        : "r"(smem_u32(&bar)));
  for (int i = threadIdx.x; i < kPixels * kC; i += blockDim.x)
    out[size_t(blockIdx.x) * kPixels * kC + i] = tile[i];
}

bool run(const Conv& cv) {
  const int P = cv.p(), Q = cv.q(), M = cv.n * P * Q;
  const int tiles = (M + kPixels - 1) / kPixels, taps = cv.r * cv.s;
  std::vector<float> in(size_t(cv.n) * cv.h * cv.w * kC);
  for (size_t i = 0; i < in.size(); ++i) in[i] = float(i % 1009) + 1.0f;   // never zero
  float *d_in, *d_out;
  CK(cudaMalloc(&d_in, in.size() * 4));
  CK(cudaMalloc(&d_out, size_t(tiles) * taps * kPixels * kC * 4));
  CK(cudaMemcpy(d_in, in.data(), in.size() * 4, cudaMemcpyHostToDevice));

  CUtensorMap map;
  const unsigned rank = cv.one_d ? 3 : 4;
  cuuint64_t dims[4], strides[3];
  int lower[2], upper[2];
  cuuint32_t estr[4] = {1, 1, 1, 1};
  dims[0] = kC;
  dims[1] = cv.w;
  strides[0] = uint64_t(kC) * 4;
  lower[0] = -cv.pad_w;
  upper[0] = cv.pad_w - (cv.s - 1) * cv.dil_w;
  estr[1] = cv.stride_w;
  if (cv.one_d) {
    dims[2] = cv.n;
    strides[1] = uint64_t(cv.w) * kC * 4;
  } else {
    dims[2] = cv.h;
    dims[3] = cv.n;
    strides[1] = uint64_t(cv.w) * kC * 4;
    strides[2] = uint64_t(cv.h) * cv.w * kC * 4;
    lower[1] = -cv.pad_h;
    upper[1] = cv.pad_h - (cv.r - 1) * cv.dil_h;
    estr[2] = cv.stride_h;
  }
  CK(cuTensorMapEncodeIm2col(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, rank, d_in, dims, strides, lower,
                             upper, kC, kPixels, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  load_columns<<<tiles * taps, 128>>>(map, cv, d_out);
  CK(cudaDeviceSynchronize());
  std::vector<float> got(size_t(tiles) * taps * kPixels * kC);
  CK(cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost));

  int bad = 0;
  for (int tap = 0; tap < taps; ++tap) {
    const int rr = tap / cv.s, ss = tap % cv.s;
    for (int t = 0; t < tiles; ++t)
      for (int row = 0; row < kPixels; ++row) {
        const int m = t * kPixels + row;
        const int q = m % Q, p = (m / Q) % P, n = m / (Q * P);
        const int y = p * cv.stride_h - cv.pad_h + rr * cv.dil_h;
        const int x = q * cv.stride_w - cv.pad_w + ss * cv.dil_w;
        const bool inside = n < cv.n && y >= 0 && y < cv.h && x >= 0 && x < cv.w;
        for (int c = 0; c < kC; ++c) {
          const float want = inside ? in[((size_t(n) * cv.h + y) * cv.w + x) * kC + c] : 0.0f;
          const float have = got[(size_t(tap * tiles + t) * kPixels + row) * kC + c];
          if (have != want && bad++ < 4)
            std::printf("  %s: tap (%d,%d) m=%d c=%d: %g, want %g\n", cv.name, rr, ss, m, c, have, want);
        }
      }
  }
  std::printf("%s: %d of %d elements wrong\n", cv.name, bad, tiles * taps * kPixels * kC);
  cudaFree(d_in);
  cudaFree(d_out);
  return bad == 0;
}

int main() {
  bool ok = true;
  //            name                                   n  h  w  r  s  ph pw sh sw dh dw  1d
  ok &= run({"2D 3x3, padding 1, stride 2",           2, 7, 9, 3, 3, 1, 1, 2, 2, 1, 1, false});
  ok &= run({"2D 3x3, padding 2, dilation 2",         2, 6, 7, 3, 3, 2, 2, 1, 1, 2, 2, false});
  ok &= run({"2D 2x3, padding 0/1, stride 1/2",       3, 5, 8, 2, 3, 0, 1, 1, 2, 1, 1, false});
  ok &= run({"1D 5-tap, padding 2, stride 3",         3, 1, 20, 1, 5, 0, 2, 1, 3, 1, 1, true});
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
