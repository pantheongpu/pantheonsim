// A __noinline__ function that returns early when two cells are equal. On sm_90 ptxas ends it with a
// RET that has a condition predicate beside its guard, and only the lanes where both hold return. The
// simulator ignored the second predicate, so lanes whose cell had changed in one coordinate returned
// without registering. The same code runs on the host over a copy of the particles; the counts must match.
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

typedef unsigned long long u64;
#define HD __host__ __device__
static constexpr int cap = 128, blocks = 144;

struct Cell { int v[2]; };
struct Particle { Cell cell; double delta[2], vel[2]; };
struct Tracker { Cell cell; unsigned tile; };

HD inline bool equals(Cell const& a, Cell const& b) {
  for (int d = 0; d < 2; ++d)
    if (a.v[d] != b.v[d]) return false;
  return true;
}

HD inline u64 fetch_add(u64& t) {
#ifdef __CUDA_ARCH__
  return atomicAdd(&t, u64{1});
#else
  return t++;
#endif
}

struct Pusher {
  u64 *counts, *slots;
  Particle* particles;

  HD __attribute__((noinline)) void step(Particle& p, Tracker& pt, u64& idx) {
    for (int d = 0; d < 2; ++d) {
      double delta = p.delta[d] + 0.25 * p.vel[d];
      p.cell.v[d] += (int)floor(delta);
      p.delta[d] = delta - floor(delta);
    }
    if (equals(p.cell, pt.cell)) return;
    int old = ((pt.cell.v[0] % 12) * 12 + pt.cell.v[1] % 12 + 144) % 144;
    slots[old * cap + fetch_add(counts[old])] = idx;
  }
  HD void one(Particle& p, Tracker&& pt, u64& idx) { step(p, pt, idx); }
  HD void run(unsigned block, unsigned lane) {
    for (u64 i = lane; i < 6; i += 32) {
      auto& p = particles[block * cap + i];
      one(p, Tracker{p.cell, block % 2}, i);
    }
  }
};

__global__ void k(Pusher p) { p.run(blockIdx.x, threadIdx.x); }

Pusher make() {
  Pusher p;
  cudaMallocManaged(&p.counts, blocks * sizeof(u64));
  cudaMallocManaged(&p.slots, blocks * cap * sizeof(u64));
  cudaMallocManaged(&p.particles, blocks * cap * sizeof(Particle));
  for (int b = 0; b < blocks; ++b) {
    p.counts[b] = 0;
    for (int i = 0; i < 6; ++i)
      p.particles[b * cap + i] = Particle{{{b % 12, b / 12}}, {0.5, 0.5}, {double((b + 3 * i) % 5 - 2), double((b * i) % 7 - 3)}};
  }
  return p;
}

int main() {
  Pusher host = make(), dev = make();
  for (int b = 0; b < blocks; ++b)
    for (int lane = 0; lane < 32; ++lane) host.run(b, lane);
  k<<<blocks, 32>>>(dev);
  cudaDeviceSynchronize();
  u64 want = 0, got = 0;
  for (int b = 0; b < blocks; ++b) want += host.counts[b], got += dev.counts[b];
  std::printf("registered %llu, host %llu: %s\n", got, want, got == want ? "PASS" : "FAIL");
  return got != want;
}
