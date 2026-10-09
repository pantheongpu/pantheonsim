// A byte or halfword read from constant memory at an offset that is not a
// multiple of 4: ptxas loads a uniform one with ULDC.U8 / .S8 / .U16 / .S16
// (for example `ULDC.U8 UR4, c[0x3][0x59]`). The simulator read the whole 32-bit
// word at that offset, which is not aligned, and refused the load as a
// misaligned access; Kokkos' CUDA kernels hit it on a bool kept in constant
// memory. The value is the byte or halfword out of the aligned word, extended
// as the suffix says. Every value here is checked on the host, and the check
// passes on an RTX 3080 Ti as well.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

__constant__ uint8_t  cu8[16];
__constant__ int8_t   cs8[16];
__constant__ uint16_t cu16[16];
__constant__ int16_t  cs16[16];
__constant__ bool     flag[8];

// Uniform: the index is a compile-time constant, so the load is the same for
// every thread. The offsets 1, 5, 9, 13 are one byte past a word; 2, 6, 10
// are halfwords in the high half of a word.
__global__ void k(int32_t* out) {
  int i = 0;
  out[i++] = cu8[1];
  out[i++] = cu8[5];
  out[i++] = cu8[9];
  out[i++] = cu8[13];
  out[i++] = cs8[3];
  out[i++] = cs8[7];
  out[i++] = cs8[10];
  out[i++] = cu16[1];
  out[i++] = cu16[3];
  out[i++] = cu16[5];
  out[i++] = cs16[1];
  out[i++] = cs16[3];
  out[i++] = cs16[7];
}


// Uniform uses -- a loop bound, a branch every thread takes the same way -- are
// what make ptxas load the value into a uniform register with ULDC.
__global__ void uni(int32_t* out, int n) {
  int acc = 0;
  for (int j = 0; j < (cu8[1] & 7); ++j) acc += n + j;          // ULDC.U8 as the loop bound
  out[0] = acc;
  acc = 0;
  for (int j = 0; j < (cs8[3] + 100) % 5 + 1; ++j) acc += n;    // ULDC.S8
  out[1] = acc;
  acc = 0;
  for (int j = 0; j < (cu16[3] & 7); ++j) acc += 3 * n;          // ULDC.U16
  out[2] = acc;
  acc = 0;
  for (int j = 0; j < (cs16[3] & 7) + 1; ++j) acc += 5 * n;      // ULDC.S16
  out[3] = acc;
  if (cs8[7] < 0) out[4] = 1; else out[4] = 2;                  // a uniform branch on a signed byte
  if (cs16[1] < 0) out[5] = 1; else out[5] = 2;
  // a bool kept in constant memory, tested uniformly (Kokkos' case), and the
  // unsigned byte and halfword compared whole
  out[6] = flag[5] ? 7 : 8;
  out[7] = flag[2] ? 7 : 8;
  acc = 0;
  for (int j = 0; j < cu8[5] - 120; ++j) acc += n;            // ULDC.U8
  out[8] = acc;
  acc = 0;
  for (int j = 0; j < cu16[5] - 0x8000 - 1500; ++j) acc += n; // ULDC.U16
  out[9] = acc;
}

// Kernel arguments are constant-bank reads as well: a byte and a halfword that
// share a word with their neighbours.
__global__ void args(int32_t* out, uint8_t a, int8_t b, uint16_t c, int16_t d, uint8_t e, int8_t f) {
  out[0] = a;
  out[1] = b;
  out[2] = c;
  out[3] = d;
  out[4] = e;
  out[5] = f;
}

int main() {
  uint8_t hu8[16];
  int8_t hs8[16];
  uint16_t hu16[16];
  int16_t hs16[16];
  for (int i = 0; i < 16; ++i) {
    hu8[i] = static_cast<uint8_t>(0x80 + 7 * i);    // above 0x7f, so a wrong extension shows
    hs8[i] = static_cast<int8_t>(-100 + 13 * i);    // negative for the low indices, positive above
    hu16[i] = static_cast<uint16_t>(0x8000 + 321 * i);
    hs16[i] = static_cast<int16_t>(-20000 + 1700 * i);
  }
  cudaMemcpyToSymbol(cu8, hu8, sizeof hu8);
  cudaMemcpyToSymbol(cs8, hs8, sizeof hs8);
  cudaMemcpyToSymbol(cu16, hu16, sizeof hu16);
  cudaMemcpyToSymbol(cs16, hs16, sizeof hs16);
  const bool hflag[8] = {false, true, false, false, true, true, false, true};
  cudaMemcpyToSymbol(flag, hflag, sizeof hflag);
  int32_t* d;
  cudaMalloc(&d, 64 * sizeof(int32_t));
  k<<<1, 1>>>(d);
  int32_t got[13], want[13] = {hu8[1], hu8[5], hu8[9], hu8[13], hs8[3], hs8[7], hs8[10],
                               hu16[1], hu16[3], hu16[5], hs16[1], hs16[3], hs16[7]};
  if (cudaMemcpy(got, d, sizeof got, cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::printf("FAIL: the constant-memory kernel did not run\n");
    return 1;
  }
  int bad = 0;
  for (int i = 0; i < 13; ++i)
    if (got[i] != want[i]) {
      std::printf("FAIL: constant read %d: got %d want %d\n", i, got[i], want[i]);
      ++bad;
    }
  uni<<<1, 1>>>(d, 3);
  int32_t g3[10];
  if (cudaMemcpy(g3, d, sizeof g3, cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::printf("FAIL: the uniform kernel did not run\n");
    return 1;
  }
  {
    int w3[10], acc;
    acc = 0; for (int j = 0; j < (hu8[1] & 7); ++j) acc += 3 + j;            w3[0] = acc;
    acc = 0; for (int j = 0; j < (hs8[3] + 100) % 5 + 1; ++j) acc += 3;      w3[1] = acc;
    acc = 0; for (int j = 0; j < (hu16[3] & 7); ++j) acc += 9;               w3[2] = acc;
    acc = 0; for (int j = 0; j < (hs16[3] & 7) + 1; ++j) acc += 15;          w3[3] = acc;
    w3[4] = hs8[7] < 0 ? 1 : 2;
    w3[5] = hs16[1] < 0 ? 1 : 2;
    w3[6] = hflag[5] ? 7 : 8;
    w3[7] = hflag[2] ? 7 : 8;
    acc = 0; for (int j = 0; j < hu8[5] - 120; ++j) acc += 3; w3[8] = acc;
    acc = 0; for (int j = 0; j < hu16[5] - 0x8000 - 1500; ++j) acc += 3; w3[9] = acc;
    for (int i = 0; i < 10; ++i)
      if (g3[i] != w3[i]) {
        std::printf("FAIL: uniform use %d: got %d want %d\n", i, g3[i], w3[i]);
        ++bad;
      }
  }
  args<<<1, 1>>>(d, 0xf1, -5, 0xfff2, -300, 0x83, -77);
  int32_t g2[6], w2[6] = {0xf1, -5, 0xfff2, -300, 0x83, -77};
  if (cudaMemcpy(g2, d, sizeof g2, cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::printf("FAIL: the argument kernel did not run\n");
    return 1;
  }
  for (int i = 0; i < 6; ++i)
    if (g2[i] != w2[i]) {
      std::printf("FAIL: argument %d: got %d want %d\n", i, g2[i], w2[i]);
      ++bad;
    }
  std::printf(bad ? "FAIL: %d wrong\n" : "PASS: every narrow constant read\n", bad);
  return bad != 0;
}
