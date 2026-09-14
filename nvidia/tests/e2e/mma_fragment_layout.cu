// Where each lane's mma.sync fragment element lives in the tile.
//
// A permuted row<->lane mapping cancels out almost everywhere: a kernel that
// writes a fragment back through the same mapping it read it with gets the
// right answer either way. It stops cancelling the moment a value is indexed
// by its logical row -- which is what ggml's flash-attention stream-k meta
// write does, and why this is worth pinning separately from "the matmul is
// right".
//
// Checked against a physical A10 through nvidia/tools/run-on-hardware.sh: 128/128 on
// the device and 128/128 here.
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void k(const __half* A, const __half* B, float* D) {
  const int lane = threadIdx.x;
  // A: 16x16 row-major, B: 16x8 col-major, per the mma.sync fragment layout.
  unsigned a[4], b[2];
  float d[4] = {0, 0, 0, 0};
  // A fragment: lane holds rows (lane/4, lane/4+8), cols 2*(lane%4)+{0,1} and +8
  const int ar = lane / 4, ac = 2 * (lane % 4);
  __half av[8] = {A[(ar) * 16 + ac], A[(ar) * 16 + ac + 1],
                  A[(ar + 8) * 16 + ac], A[(ar + 8) * 16 + ac + 1],
                  A[(ar) * 16 + ac + 8], A[(ar) * 16 + ac + 9],
                  A[(ar + 8) * 16 + ac + 8], A[(ar + 8) * 16 + ac + 9]};
  for (int i = 0; i < 4; ++i) a[i] = (__half_as_ushort(av[2 * i + 1]) << 16) | __half_as_ushort(av[2 * i]);
  // B fragment: lane holds col lane%4... use the standard k-major layout.
  // B is 16x8 col-major: the lane's column is lane/4 and its rows are
  // 2*(lane%4) plus the +8 half. Writing these the other way round is the easy
  // mistake, and produces a wrong answer that looks like an engine bug.
  const int bc = lane / 4, br = 2 * (lane % 4);
  __half bv[4] = {B[bc * 16 + br], B[bc * 16 + br + 1], B[bc * 16 + br + 8], B[bc * 16 + br + 9]};
  for (int i = 0; i < 2; ++i) b[i] = (__half_as_ushort(bv[2 * i + 1]) << 16) | __half_as_ushort(bv[2 * i]);

  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));

  // Write each register to the (row, col) the ISA says it holds.
  for (int i = 0; i < 4; ++i) {
    const int row = (i < 2 ? 0 : 8) + lane / 4;
    const int col = 2 * (lane % 4) + (i % 2);
    D[row * 8 + col] = d[i];
  }
}

int main() {
  __half hA[256], hB[128];
  for (int r = 0; r < 16; ++r)
    for (int c = 0; c < 16; ++c) hA[r * 16 + c] = __float2half(((r * 16 + c) % 7) - 3.0f);
  for (int c = 0; c < 8; ++c)
    for (int r = 0; r < 16; ++r) hB[c * 16 + r] = __float2half(((c * 16 + r) % 5) - 2.0f);
  __half *dA, *dB; float* dD;
  cudaMalloc(&dA, sizeof hA); cudaMalloc(&dB, sizeof hB); cudaMalloc(&dD, 128 * sizeof(float));
  cudaMemcpy(dA, hA, sizeof hA, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, sizeof hB, cudaMemcpyHostToDevice);
  cudaMemset(dD, 0, 128 * sizeof(float));
  k<<<1, 32>>>(dA, dB, dD);
  cudaError_t e = cudaDeviceSynchronize();
  if (e) { printf("FAIL sync: %s\n", cudaGetErrorString(e)); return 1; }
  float hD[128];
  cudaMemcpy(hD, dD, sizeof hD, cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int r = 0; r < 16; ++r)
    for (int c = 0; c < 8; ++c) {
      double want = 0;
      for (int kk = 0; kk < 16; ++kk)
        want += (double)__half2float(hA[r * 16 + kk]) * (double)__half2float(hB[c * 16 + kk]);
      if (fabs(hD[r * 8 + c] - want) > 1e-3) {
        if (bad < 6) printf("D[%2d,%d] = %g want %g\n", r, c, hD[r * 8 + c], want);
        ++bad;
      }
    }
  if (bad) {
    printf("FAIL: %d of 128 elements in the wrong place\n", bad);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
