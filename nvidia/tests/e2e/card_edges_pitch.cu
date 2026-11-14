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
//   - cudaMemcpy3D between pitched pointers, measured the same way:
//       - one row of one slice: a pitch of 0 is accepted and the row copied; a pitch that is set and narrower than
//         the row (from pos.x) is cudaErrorInvalidValue (the simulator answered cudaErrorInvalidPitchValue);
//       - more than one row or slice: a pitch narrower than the row is cudaErrorInvalidPitchValue;
//       - more than one slice: a ysize below the rows, 0 included, is cudaErrorInvalidPitchValue (the simulator
//         took 0 for the rows and copied);
//       - a ysize that is set and does not hold the rows from pos.y is cudaErrorInvalidValue (it copied);
//       - an extent with a side of 0 copies nothing and succeeds (the simulator took 0 rows or slices for 1).
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

  // cudaMemcpy3D between pitched pointers. d holds 0, 1, 2, ...
  auto copy3d = [&](size_t dp, size_t dys, size_t sp, size_t sys, cudaPos spos, size_t w, size_t hgt, size_t dep,
                    int async) {
    cudaMemcpy3DParms c{};
    c.srcPtr = make_cudaPitchedPtr(d, sp, 8, sys);
    c.dstPtr = make_cudaPitchedPtr(d2, dp, 8, dys);
    c.srcPos = spos;
    c.extent = make_cudaExtent(w, hgt, dep);
    c.kind = cudaMemcpyDeviceToDevice;
    const cudaError_t e = async ? cudaMemcpy3DAsync(&c, 0) : cudaMemcpy3D(&c);
    WANT(cudaDeviceSynchronize(), cudaSuccess);
    cudaGetLastError();
    return e;
  };
  const cudaPos at0 = make_cudaPos(0, 0, 0);
  for (size_t dp : pitches)
    for (size_t sp : pitches)
      for (int async = 0; async < 2; ++async) {
        const bool ok = (dp == 0 || dp >= 8) && (sp == 0 || sp >= 8);
        WANT(cudaMemset(d2, 0, 64), cudaSuccess);
        const cudaError_t e = copy3d(dp, 0, sp, 0, at0, 8, 1, 1, async);
        if (e != (ok ? cudaSuccess : cudaErrorInvalidValue)) {
          std::printf("FAIL cudaMemcpy3D%s(dpitch %zu, spitch %zu, 8 x 1 x 1) -> %d\n", async ? "Async" : "", dp, sp,
                      (int)e);
          ++fails;
        }
        expect("one-row 3D copy", row, ok ? 8 : 0);
      }
  const unsigned char rows2[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  const struct { size_t dp, dys, sp, sys, x, y, w, h, dep; cudaError_t want; } box[] = {
      {0, 0, 0, 0, 5, 0, 4, 1, 1, cudaSuccess},                   // pitch 0, from pos.x 5
      {8, 0, 8, 0, 5, 0, 4, 1, 1, cudaErrorInvalidValue},         // 5 + 4 > 8, one row
      {8, 0, 4, 0, 0, 0, 8, 2, 1, cudaErrorInvalidPitchValue},    // two rows, pitch 4
      {8, 0, 0, 0, 0, 0, 8, 2, 1, cudaErrorInvalidPitchValue},    // two rows, pitch 0
      {8, 8, 8, 0, 0, 0, 8, 1, 2, cudaErrorInvalidPitchValue},    // two slices, source ysize 0
      {8, 0, 8, 1, 0, 0, 8, 1, 2, cudaErrorInvalidPitchValue},    // two slices, destination ysize 0
      {8, 1, 8, 1, 0, 0, 8, 2, 2, cudaErrorInvalidPitchValue},    // two slices, ysize 1 below 2 rows
      {8, 1, 8, 1, 0, 0, 8, 1, 2, cudaSuccess},                   // two slices of one row
      {8, 0, 8, 1, 0, 0, 8, 2, 1, cudaErrorInvalidValue},         // one slice, ysize 1 below 2 rows
      {8, 2, 8, 2, 0, 1, 4, 2, 1, cudaErrorInvalidValue},         // rows 1..2 of a ysize of 2
      {8, 3, 8, 3, 0, 1, 4, 2, 1, cudaSuccess},
      {8, 0, 8, 0, 0, 0, 8, 2, 1, cudaSuccess},                   // ysize 0 is fine for one slice
      {8, 0, 8, 0, 0, 0, 8, 0, 1, cudaSuccess},                   // nothing to copy
      {8, 0, 8, 0, 0, 0, 8, 1, 0, cudaSuccess},
      {4, 0, 4, 0, 0, 0, 8, 0, 0, cudaSuccess},                   // nothing to copy, so the pitch is not checked
  };
  for (const auto& b : box)
    for (int async = 0; async < 2; ++async) {
      WANT(cudaMemset(d2, 0, 64), cudaSuccess);
      const cudaError_t e = copy3d(b.dp, b.dys, b.sp, b.sys, make_cudaPos(b.x, b.y, 0), b.w, b.h, b.dep, async);
      if (e != b.want) {
        std::printf("FAIL cudaMemcpy3D%s(dpitch %zu ysize %zu, spitch %zu ysize %zu, pos %zu,%zu, %zu x %zu x %zu) -> %d,"
                    " expected %d\n", async ? "Async" : "", b.dp, b.dys, b.sp, b.sys, b.x, b.y, b.w, b.h, b.dep, (int)e,
                    (int)b.want);
        ++fails;
      }
      const bool empty = b.w == 0 || b.h == 0 || b.dep == 0;
      if (b.want != cudaSuccess || empty) expect("refused or empty 3D copy", rows2, 0);
      else if (b.x == 5) expect("3D copy from pos.x 5", rows2 + 5, 4);
      else if (b.y == 1) {
        const unsigned char want[12] = {8, 9, 10, 11, 0, 0, 0, 0, 16, 17, 18, 19};
        expect("3D copy from pos.y 1", want, 12);
      } else
        expect("3D copy", rows2, 16);
    }

  WANT(cudaFree(d), cudaSuccess);
  WANT(cudaFree(d2), cudaSuccess);
  std::printf("%s\n", fails ? "FAILED" : "PASS");
  return fails ? 1 : 0;
}
