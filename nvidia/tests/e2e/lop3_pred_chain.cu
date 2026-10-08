// A 64-bit "(a & b) != 0" is two LOP3s in a chain: the first tests the low
// words into a predicate, the second tests the high words and takes that
// predicate as its last operand, so the pair answers "either half overlaps".
//   LOP3.LUT P0, RZ, R_lo_a, R_lo_b, RZ, 0xc0, !PT
//   LOP3.LUT P0, RZ, R_hi_a, R_hi_b, RZ, 0xc0, P0
// The simulator read that last operand and ignored it, so a pair whose low
// words overlapped but whose high words did not came out false. kernel_float's
// bool tests (equals(T(x*y), c)) hit it: an assertion fired on an input that
// passes on an RTX 3080 Ti. Every case here is computed on the host and passes
// on the card as well.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

struct Case { uint64_t a, b, c; };

__global__ void k(const Case* in, uint32_t* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint64_t a = in[i].a, b = in[i].b, c = in[i].c;
  uint32_t r = 0;
  if ((a & b) != 0) r |= 1;                 // the pair of LOP3s
  if ((a | b) != 0) r |= 2;
  if ((a & b & c) != 0) r |= 4;
  if (((a & b) != 0) && ((b & c) != 0)) r |= 8;
  if (((a & b) != 0) || ((b & c) != 0)) r |= 16;
  if ((a & b) == 0) r |= 32;
  out[i] = r;
}

static uint32_t want(const Case& t) {
  uint32_t r = 0;
  if ((t.a & t.b) != 0) r |= 1;
  if ((t.a | t.b) != 0) r |= 2;
  if ((t.a & t.b & t.c) != 0) r |= 4;
  if (((t.a & t.b) != 0) && ((t.b & t.c) != 0)) r |= 8;
  if (((t.a & t.b) != 0) || ((t.b & t.c) != 0)) r |= 16;
  if ((t.a & t.b) == 0) r |= 32;
  return r;
}

int main() {
  // every mix of "this half overlaps" for the three operands
  const uint64_t pat[] = {0, 1ull, 1ull << 32, (1ull << 32) | 1, 0x8000000000000000ull, 0xffffffffull,
                          0xffffffff00000000ull, ~0ull};
  const int np = sizeof(pat) / sizeof(pat[0]);
  Case h[np * np * np];
  int n = 0;
  for (int i = 0; i < np; ++i)
    for (int j = 0; j < np; ++j)
      for (int l = 0; l < np; ++l) h[n++] = {pat[i], pat[j], pat[l]};
  Case* d;
  uint32_t* o;
  cudaMalloc(&d, sizeof h);
  cudaMalloc(&o, n * sizeof(uint32_t));
  cudaMemcpy(d, h, n * sizeof(Case), cudaMemcpyHostToDevice);
  k<<<(n + 63) / 64, 64>>>(d, o, n);
  uint32_t got[np * np * np];
  if (cudaMemcpy(got, o, n * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::printf("FAIL: the kernel did not run\n");
    return 1;
  }
  int bad = 0;
  for (int i = 0; i < n; ++i)
    if (got[i] != want(h[i])) {
      if (bad++ < 5)
        std::printf("FAIL: a=%016llx b=%016llx c=%016llx got %02x want %02x\n", (unsigned long long)h[i].a,
                    (unsigned long long)h[i].b, (unsigned long long)h[i].c, got[i], want(h[i]));
    }
  std::printf(bad ? "FAIL: %d of %d\n" : "PASS: %d cases\n", bad ? bad : n, n);
  return bad != 0;
}
