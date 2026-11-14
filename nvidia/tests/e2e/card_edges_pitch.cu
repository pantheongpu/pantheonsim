// The pitches of 2D copies and fills, as an RTX 3060 (CUDA 13.0) checks them.
//
//   - cudaMemcpy2D of more than one row: a pitch narrower than the row is
//     cudaErrorInvalidPitchValue (the simulator had this).
//   - Of a single row: a pitch of 0 is accepted and the row is copied; a pitch
//     that is set and narrower than the row is cudaErrorInvalidValue. The
//     simulator refused both with cudaErrorInvalidPitchValue.
//   - cudaMemset2D of more than one row with a pitch narrower than the row is
//     cudaErrorInvalidValue and writes nothing. The simulator filled the
//     overlapping rows and answered success. A single row is filled whatever
//     the pitch.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

static int fails = 0;
#define WANT(x, want) do { const int got_ = static_cast<int>(x); if (got_ != static_cast<int>(want)) { \
  std::printf("FAIL %s -> %d, expected %d\n", #x, got_, static_cast<int>(want)); ++fails; } } while (0)

static unsigned char* d = nullptr;
static unsigned char* d2 = nullptr;
// d2's first 24 bytes, against `want` (the rest of the 24 is zero).
static void expect(const char* what, const unsigned char* want, int n) {
  unsigned char got[24] = {}, w[24] = {};
  std::memcpy(w, want, n);
  cudaMemcpy(got, d2, 24, cudaMemcpyDeviceToHost);
  if (std::memcmp(got, w, 24) != 0) {
    std::printf("FAIL %s:", what);
    for (unsigned char c : got) std::printf(" %d", c);
    std::printf("\n");
    ++fails;
  }
}

int main() {
  WANT(cudaMalloc(&d, 256), cudaSuccess);
  WANT(cudaMalloc(&d2, 256), cudaSuccess);
  unsigned char h[256];
  for (int i = 0; i < 256; ++i) h[i] = static_cast<unsigned char>(i);
  WANT(cudaMemcpy(d, h, 256, cudaMemcpyHostToDevice), cudaSuccess);
  const unsigned char row[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  const unsigned char sevens[16] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
  const cudaMemcpyKind dd = cudaMemcpyDeviceToDevice;

  // One row.
  const size_t pitches[] = {0, 4, 8, 16};
  for (size_t dp : pitches)
    for (size_t sp : pitches) {
      const bool ok = (dp == 0 || dp >= 8) && (sp == 0 || sp >= 8);
      for (int async = 0; async < 2; ++async) {
        WANT(cudaMemset(d2, 0, 64), cudaSuccess);
        const cudaError_t e = async ? cudaMemcpy2DAsync(d2, dp, d, sp, 8, 1, dd, 0) : cudaMemcpy2D(d2, dp, d, sp, 8, 1, dd);
        WANT(cudaDeviceSynchronize(), cudaSuccess);
        if (e != (ok ? cudaSuccess : cudaErrorInvalidValue)) {
          std::printf("FAIL cudaMemcpy2D%s(dpitch %zu, spitch %zu, 8 x 1) -> %d\n", async ? "Async" : "", dp, sp, (int)e);
          ++fails;
        }
        cudaGetLastError();
        expect("one row", row, ok ? 8 : 0);
      }
    }
  // Host to device and back, one row with a pitch of 0.
  unsigned char back[8] = {};
  WANT(cudaMemcpy2D(d2, 0, h, 0, 8, 1, cudaMemcpyHostToDevice), cudaSuccess);
  WANT(cudaMemcpy2D(back, 0, d2, 0, 8, 1, cudaMemcpyDeviceToHost), cudaSuccess);
  WANT(std::memcmp(back, row, 8), 0);
  WANT(cudaMemcpy2D(back, 4, d2, 8, 8, 1, cudaMemcpyDeviceToHost), cudaErrorInvalidValue);
  WANT(cudaGetLastError(), cudaErrorInvalidValue);
  // Two rows: a pitch below the width, 0 included, is a pitch error.
  WANT(cudaMemcpy2D(d2, 0, d, 0, 8, 2, dd), cudaErrorInvalidPitchValue);
  WANT(cudaMemcpy2D(d2, 8, d, 4, 8, 2, dd), cudaErrorInvalidPitchValue);
  WANT(cudaGetLastError(), cudaErrorInvalidPitchValue);

  // Fills.
  for (size_t p : pitches) {
    WANT(cudaMemset(d2, 0, 64), cudaSuccess);
    WANT(cudaMemset2D(d2, p, 7, 8, 1), cudaSuccess);
    expect("one-row fill", sevens, 8);
    WANT(cudaMemset(d2, 0, 64), cudaSuccess);
    WANT(cudaMemset2DAsync(d2, p, 7, 8, 1, 0), cudaSuccess);
    WANT(cudaDeviceSynchronize(), cudaSuccess);
    expect("one-row async fill", sevens, 8);
  }
  for (size_t p : {size_t(0), size_t(4)}) {
    WANT(cudaMemset(d2, 0, 64), cudaSuccess);
    WANT(cudaMemset2D(d2, p, 7, 8, 2), cudaErrorInvalidValue);
    WANT(cudaGetLastError(), cudaErrorInvalidValue);
    WANT(cudaMemset2DAsync(d2, p, 7, 8, 2, 0), cudaErrorInvalidValue);
    WANT(cudaDeviceSynchronize(), cudaSuccess);
    cudaGetLastError();
    expect("refused fill", sevens, 0);
  }
  WANT(cudaMemset(d2, 0, 64), cudaSuccess);
  WANT(cudaMemset2D(d2, 8, 7, 8, 2), cudaSuccess);
  expect("two-row fill", sevens, 16);
  // cudaMemset3D's slices are 2D fills: a box narrower than its pitch is still refused.
  cudaPitchedPtr pp = make_cudaPitchedPtr(d2, 4, 8, 2);
  WANT(cudaMemset3D(pp, 7, make_cudaExtent(8, 2, 1)), cudaErrorInvalidValue);
  cudaGetLastError();

  WANT(cudaFree(d), cudaSuccess);
  WANT(cudaFree(d2), cudaSuccess);
  std::printf("%s\n", fails ? "FAILED" : "PASS");
  return fails ? 1 : 0;
}
