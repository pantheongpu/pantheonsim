// Device atomics from many blocks at once, on both roads the engines take for them (the striped locks on a process with fewer than four
// host threads, the CPU's own swap on each word from four up: run this under VGPU_THREADS=1, 2, 4 and 8, SASS and PTX). Every check is of
// a result no schedule can change: a fetch-add hands out each old value once, and the commutative operations leave the serial value.
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

__global__ void tickets(unsigned* ctr, unsigned long long* ctr64, float* fctr, unsigned* taken, unsigned long long* taken64, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  taken[atomicAdd(ctr, 1u)] = 1;                                    // (a ticket handed out twice would leave a hole)
  taken64[atomicAdd(ctr64, 1ull)] = 1;
  atomicAdd(fctr, 1.0f);                                            // (integers below 2^24: exact in any order)
}
__global__ void spread(unsigned* w, unsigned* mx, unsigned* mn, unsigned* an, unsigned* orr, unsigned* xr, int n, int words) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int k = (i * 7) % words;
  atomicAdd(&w[k], (unsigned)(i + 1));
  atomicMax(&mx[k], (unsigned)(i * 2654435761u));
  atomicMin(&mn[k], (unsigned)(i * 2654435761u));
  atomicAnd(&an[k], ~(1u << (i % 32)) | (unsigned)(i % 3 ? 0xffffffffu : 0));
  atomicOr(&orr[k], 1u << ((i / 3) % 32));
  atomicXor(&xr[k], (unsigned)(i * 40503u));
}
__global__ void cas_sum(unsigned long long* v, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long long old = *(volatile unsigned long long*)v;
  for (;;) {
    const unsigned long long got = atomicCAS(v, old, old + (unsigned long long)(i + 1));
    if (got == old) break;
    old = got;
  }
}
__global__ void wrap(unsigned* v, int n) {
  if (blockIdx.x * blockDim.x + threadIdx.x < n) atomicInc(v, 99u);   // (counts 0..99 and rolls over: n % 100 at the end)
}

int main() {
  const int n = 1 << 17, B = 256, words = 61;
  int bad = 0;
  auto chk = [&](bool ok, const char* what) { if (!ok) { printf("FAIL %s\n", what); ++bad; } };
  unsigned *ctr, *taken, *w, *mx, *mn, *an, *orr, *xr, *wr;
  unsigned long long *ctr64, *taken64, *cs;
  float* fctr;
  CK(cudaMalloc(&ctr, 4)); CK(cudaMalloc(&ctr64, 8)); CK(cudaMalloc(&fctr, 4)); CK(cudaMalloc(&cs, 8)); CK(cudaMalloc(&wr, 4));
  CK(cudaMalloc(&taken, 4 * n)); CK(cudaMalloc(&taken64, 8 * n));
  CK(cudaMemset(ctr, 0, 4)); CK(cudaMemset(ctr64, 0, 8)); CK(cudaMemset(fctr, 0, 4)); CK(cudaMemset(cs, 0, 8)); CK(cudaMemset(wr, 0, 4));
  CK(cudaMemset(taken, 0, 4 * n)); CK(cudaMemset(taken64, 0, 8 * n));
  tickets<<<n / B, B>>>(ctr, ctr64, fctr, taken, taken64, n);
  CK(cudaDeviceSynchronize());
  unsigned c32; unsigned long long c64; float cf;
  CK(cudaMemcpy(&c32, ctr, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&c64, ctr64, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&cf, fctr, 4, cudaMemcpyDeviceToHost));
  chk(c32 == (unsigned)n, "u32 counter"); chk(c64 == (unsigned long long)n, "u64 counter"); chk(cf == (float)n, "f32 counter");
  std::vector<unsigned> t(n); std::vector<unsigned long long> t64(n);
  CK(cudaMemcpy(t.data(), taken, 4 * n, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(t64.data(), taken64, 8 * n, cudaMemcpyDeviceToHost));
  int holes = 0, holes64 = 0;
  for (int i = 0; i < n; ++i) holes += t[i] != 1, holes64 += t64[i] != 1;
  chk(holes == 0, "every u32 ticket handed out once"); chk(holes64 == 0, "every u64 ticket handed out once");

  CK(cudaMalloc(&w, 4 * words)); CK(cudaMalloc(&mx, 4 * words)); CK(cudaMalloc(&mn, 4 * words));
  CK(cudaMalloc(&an, 4 * words)); CK(cudaMalloc(&orr, 4 * words)); CK(cudaMalloc(&xr, 4 * words));
  std::vector<unsigned> zero(words, 0), ones(words, 0xffffffffu);
  CK(cudaMemcpy(w, zero.data(), 4 * words, cudaMemcpyHostToDevice)); CK(cudaMemcpy(mx, zero.data(), 4 * words, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(mn, ones.data(), 4 * words, cudaMemcpyHostToDevice)); CK(cudaMemcpy(an, ones.data(), 4 * words, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(orr, zero.data(), 4 * words, cudaMemcpyHostToDevice)); CK(cudaMemcpy(xr, zero.data(), 4 * words, cudaMemcpyHostToDevice));
  spread<<<n / B, B>>>(w, mx, mn, an, orr, xr, n, words);
  CK(cudaDeviceSynchronize());
  std::vector<unsigned> hw(words), hmx(words), hmn(words), han(words), hor(words), hxr(words);
  CK(cudaMemcpy(hw.data(), w, 4 * words, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hmx.data(), mx, 4 * words, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hmn.data(), mn, 4 * words, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(han.data(), an, 4 * words, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hor.data(), orr, 4 * words, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hxr.data(), xr, 4 * words, cudaMemcpyDeviceToHost));
  std::vector<unsigned> ew(words, 0), emx(words, 0), emn(words, 0xffffffffu), ean(words, 0xffffffffu), eor(words, 0), exr(words, 0);
  for (int i = 0; i < n; ++i) {
    const int k = (int)(((long long)i * 7) % words);
    ew[k] += (unsigned)(i + 1);
    const unsigned x = (unsigned)i * 2654435761u;
    if (x > emx[k]) emx[k] = x;
    if (x < emn[k]) emn[k] = x;
    ean[k] &= ~(1u << (i % 32)) | (unsigned)(i % 3 ? 0xffffffffu : 0);
    eor[k] |= 1u << ((i / 3) % 32);
    exr[k] ^= (unsigned)i * 40503u;
  }
  int wrong = 0;
  for (int k = 0; k < words; ++k)
    wrong += hw[k] != ew[k] || hmx[k] != emx[k] || hmn[k] != emn[k] || han[k] != ean[k] || hor[k] != eor[k] || hxr[k] != exr[k];
  chk(wrong == 0, "add, max, min, and, or, xor on 61 words");

  cas_sum<<<n / B / 8, B>>>(cs, n / 8);
  wrap<<<n / B, B>>>(wr, n);
  CK(cudaDeviceSynchronize());
  unsigned long long hcs; unsigned hwr;
  CK(cudaMemcpy(&hcs, cs, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&hwr, wr, 4, cudaMemcpyDeviceToHost));
  const unsigned long long m = n / 8;
  chk(hcs == m * (m + 1) / 2, "compare-and-swap sum");
  chk(hwr == (unsigned)(n % 100), "atomicInc wraps");
  if (bad) return 1;
  printf("PASS\n");
  return 0;
}
