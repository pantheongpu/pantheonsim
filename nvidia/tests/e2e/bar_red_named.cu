// bar.red on the CTA's other fifteen barriers, with and without a thread
// count, with the barrier and count in registers, guarded, and in a loop.
// Each thread's results are checked against the ISA's definition on the host;
// the same program, run on an RTX 3060, is what says the definition was read
// right.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

constexpr int kThreads = 128;   // four warps
constexpr int kLoops = 4;
constexpr int kOut = 8 + kLoops;

__device__ __forceinline__ unsigned red_popc_counted(unsigned id, unsigned count, bool p) {
  unsigned d;
  asm volatile("{ .reg .pred q; setp.ne.u32 q, %3, 0; bar.red.popc.u32 %0, %1, %2, q; }"
               : "=r"(d) : "r"(id), "r"(count), "r"(unsigned(p)) : "memory");
  return d;
}

__global__ void kernel(unsigned* out) {
  const unsigned t = threadIdx.x, warp = t / 32;
  unsigned* o = out + t * kOut;

  // Barriers 1 and 2, 64 threads each: two independent reductions at once.
  unsigned a;
  if (warp < 2)
    asm volatile("{ .reg .pred p; setp.eq.u32 p, %1, 0; bar.red.popc.u32 %0, 1, 64, p; }"
                 : "=r"(a) : "r"(t % 3) : "memory");
  else
    asm volatile("{ .reg .pred p; setp.eq.u32 p, %1, 0; bar.red.popc.u32 %0, 2, 64, p; }"
                 : "=r"(a) : "r"(t % 5) : "memory");
  o[0] = a;

  // .and and .or on counted barriers, negated source.
  unsigned an, orr;
  if (warp < 2)
    asm volatile("{ .reg .pred p, q; setp.lt.u32 p, %2, 60;"
                 " bar.red.and.pred q, 3, 64, p; selp.u32 %0, 1, 0, q;"
                 " bar.red.or.pred q, 4, 64, !p; selp.u32 %1, 1, 0, q; }"
                 : "=r"(an), "=r"(orr) : "r"(t) : "memory");
  else
    asm volatile("{ .reg .pred p, q; setp.lt.u32 p, %2, 200;"
                 " bar.red.and.pred q, 5, 64, p; selp.u32 %0, 1, 0, q;"
                 " bar.red.or.pred q, 6, 64, !p; selp.u32 %1, 1, 0, q; }"
                 : "=r"(an), "=r"(orr) : "r"(t) : "memory");
  o[1] = an;
  o[2] = orr;

  // Barrier 7 without a count: every thread of the CTA.
  unsigned all;
  asm volatile("{ .reg .pred p; setp.ne.u32 p, %1, 0; bar.red.popc.u32 %0, 7, p; }"
               : "=r"(all) : "r"(t & 1u) : "memory");
  o[3] = all;

  // Barrier and count in registers.
  o[4] = red_popc_counted(13 + warp / 2, 64, (t & 7u) == 0);

  // A guarded bar.red: warp 3 skips it, and the other three warps' 96
  // threads complete barrier 9.
  unsigned g = 0xdeadbeef;
  asm volatile("{ .reg .pred take, p; setp.lt.u32 take, %1, 3; setp.lt.u32 p, %2, 50;"
               " @take bar.red.popc.u32 %0, 9, 96, p; }"
               : "+r"(g) : "r"(warp), "r"(t) : "memory");
  o[5] = g;

  // Barrier 0 with a count, which __syncthreads_count never produces.
  unsigned z;
  if (warp == 1 || warp == 2)
    asm volatile("{ .reg .pred p; setp.ge.u32 p, %1, 48; bar.red.popc.u32 %0, 0, 64, p; }"
                 : "=r"(z) : "r"(t) : "memory");
  else
    z = 0;
  o[6] = z;

  // Warps 0 and 3 meanwhile meet on barrier 10 while 1 and 2 are on 0.
  unsigned y = 0;
  if (warp == 0 || warp == 3)
    asm volatile("{ .reg .pred p; setp.ge.u32 p, %1, 16; bar.red.or.pred p, 10, 64, p; selp.u32 %0, 1, 0, p; }"
                 : "=r"(y) : "r"(t) : "memory");
  o[7] = y;

  // Rounds on one counted barrier, back to back: no round's votes may leak
  // into the next.
  for (unsigned i = 0; i < kLoops; ++i) {
    unsigned r;
    if (warp < 2)
      asm volatile("{ .reg .pred p; setp.le.u32 p, %1, %2; bar.red.popc.u32 %0, 11, 64, p; }"
                   : "=r"(r) : "r"(t % 32), "r"(i * 7) : "memory");
    else
      asm volatile("{ .reg .pred p; setp.le.u32 p, %1, %2; bar.red.popc.u32 %0, 12, 64, p; }"
                   : "=r"(r) : "r"(t % 32), "r"(i * 11) : "memory");
    o[8 + i] = r;
  }
}

static unsigned count_if(unsigned lo, unsigned hi, bool (*f)(unsigned, unsigned), unsigned arg) {
  unsigned n = 0;
  for (unsigned t = lo; t < hi; ++t) n += f(t, arg);
  return n;
}

int main() {
  unsigned* d;
  cudaMalloc(&d, kThreads * kOut * sizeof(unsigned));
  cudaMemset(d, 0xff, kThreads * kOut * sizeof(unsigned));
  kernel<<<1, kThreads>>>(d);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    std::printf("kernel failed: %s\n", cudaGetErrorString(e));
    return 1;
  }
  unsigned h[kThreads * kOut];
  cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);

  int bad = 0;
  auto check = [&](unsigned t, int slot, unsigned want, const char* what) {
    const unsigned got = h[t * kOut + slot];
    if (got != want && bad++ < 20)
      std::printf("thread %u, %s: got %u, want %u\n", t, what, got, want);
  };
  for (unsigned t = 0; t < kThreads; ++t) {
    const unsigned warp = t / 32;
    const bool lo = warp < 2;
    const unsigned b0 = lo ? 0 : 64;
    check(t, 0, lo ? count_if(b0, b0 + 64, [](unsigned x, unsigned) { return x % 3 == 0; }, 0)
                   : count_if(b0, b0 + 64, [](unsigned x, unsigned) { return x % 5 == 0; }, 0),
          "popc on barriers 1 and 2");
    check(t, 1, lo ? 0 : 1, "and on barriers 3 and 5");
    check(t, 2, lo ? 1 : 0, "or of the negation on barriers 4 and 6");
    check(t, 3, kThreads / 2, "popc on barrier 7 without a count");
    check(t, 4, 8, "popc with barrier and count in registers");
    check(t, 5, warp < 3 ? 50 : 0xdeadbeef, "guarded popc on barrier 9");
    check(t, 6, (warp == 1 || warp == 2) ? 64 - 16 : 0, "popc on barrier 0 with a count");
    check(t, 7, (warp == 0 || warp == 3) ? 1 : 0, "or on barrier 10");
    for (unsigned i = 0; i < kLoops; ++i) {
      const unsigned lim = lo ? i * 7 : i * 11;
      check(t, 8 + static_cast<int>(i), 2 * (lim + 1 < 32 ? lim + 1 : 32), "popc round on barriers 11 and 12");
    }
  }
  if (bad) {
    std::printf("%d mismatches\nFAIL\n", bad);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
