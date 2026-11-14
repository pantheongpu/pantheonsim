// A copy kind that is none of cudaMemcpyHostToHost..cudaMemcpyDefault is
// cudaErrorInvalidMemcpyDirection (21) on an RTX 3060 (CUDA 13.0), for
// cudaMemcpy, cudaMemcpyAsync, cudaMemcpy2D, cudaMemcpy2DAsync and cudaMemcpy3D.
// The simulator answered cudaErrorInvalidValue. A copy of nothing still
// succeeds whatever the kind, and a 2D copy's pitches are checked before its
// kind.
#include <cuda_runtime.h>

#include <cstdio>

static int fails = 0;
#define WANT(x, want) do { const int got_ = static_cast<int>(x); if (got_ != static_cast<int>(want)) { \
  std::printf("FAIL %s -> %d, expected %d\n", #x, got_, static_cast<int>(want)); ++fails; } } while (0)

int main() {
  char *d = nullptr, *d2 = nullptr;
  WANT(cudaMalloc(&d, 256), cudaSuccess);
  WANT(cudaMalloc(&d2, 256), cudaSuccess);
  const cudaMemcpyKind five = static_cast<cudaMemcpyKind>(5), minus = static_cast<cudaMemcpyKind>(-1);
  WANT(cudaMemcpy(d2, d, 8, five), cudaErrorInvalidMemcpyDirection);
  WANT(cudaGetLastError(), cudaErrorInvalidMemcpyDirection);
  WANT(cudaMemcpy(d2, d, 8, static_cast<cudaMemcpyKind>(7)), cudaErrorInvalidMemcpyDirection);
  WANT(cudaMemcpy(d2, d, 8, minus), cudaErrorInvalidMemcpyDirection);
  WANT(cudaMemcpy(d2, d, 0, five), cudaSuccess);
  WANT(cudaMemcpyAsync(d2, d, 8, five, 0), cudaErrorInvalidMemcpyDirection);
  WANT(cudaMemcpy2D(d2, 8, d, 8, 8, 2, five), cudaErrorInvalidMemcpyDirection);
  WANT(cudaMemcpy2D(d2, 8, d, 8, 0, 2, five), cudaSuccess);
  WANT(cudaMemcpy2D(d2, 4, d, 8, 8, 2, five), cudaErrorInvalidPitchValue);
  WANT(cudaMemcpy2DAsync(d2, 8, d, 8, 8, 2, five, 0), cudaErrorInvalidMemcpyDirection);
  cudaMemcpy3DParms p = {};
  p.srcPtr = make_cudaPitchedPtr(d, 8, 8, 2);
  p.dstPtr = make_cudaPitchedPtr(d2, 8, 8, 2);
  p.extent = make_cudaExtent(8, 2, 1);
  p.kind = five;
  WANT(cudaMemcpy3D(&p), cudaErrorInvalidMemcpyDirection);
  WANT(cudaGetLastError(), cudaErrorInvalidMemcpyDirection);
  WANT(cudaDeviceSynchronize(), cudaSuccess);
  WANT(cudaFree(d), cudaSuccess);
  WANT(cudaFree(d2), cudaSuccess);
  std::printf("%s\n", fails ? "FAILED" : "PASS");
  return fails ? 1 : 0;
}
