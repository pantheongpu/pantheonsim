// Indirect calls where the lanes of a warp call different functions: virtual
// methods over a mix of object types, and a table of function pointers
// indexed per lane. Each lane must run its own target's body and get its own
// return value back, and the warp must carry on together afterwards.
#include <cstdio>
#include <cuda_runtime.h>
#include <new>

struct Shape {
  __device__ virtual float area() const = 0;
  __device__ virtual int sides() const { return 0; }
  __device__ virtual ~Shape() {}
};
struct Square : Shape {
  float s;
  __device__ explicit Square(float v) : s(v) {}
  __device__ float area() const override { return s * s; }
  __device__ int sides() const override { return 4; }
};
struct Triangle : Shape {
  float b, h;
  __device__ Triangle(float x, float y) : b(x), h(y) {}
  __device__ float area() const override { return 0.5f * b * h; }
  __device__ int sides() const override { return 3; }
};
struct Circle : Shape {
  float r;
  __device__ explicit Circle(float v) : r(v) {}
  __device__ float area() const override { return 3.0f * r * r; }
};

typedef int (*op_t)(int, int);
__device__ __noinline__ int add(int a, int b) { return a + b; }
__device__ __noinline__ int sub(int a, int b) { return a - b; }
__device__ __noinline__ int mul(int a, int b) { return a * b; }
__device__ __noinline__ int mix(int a, int b) { return (a << 4) ^ b; }
__device__ op_t table[4] = {add, sub, mul, mix};

constexpr int kThreads = 128;

__global__ void kernel(float* areas, int* sides, int* ops, int* after) {
  const int t = threadIdx.x;
  // Each thread builds one object in its own storage; the type cycles with
  // a period that does not divide the warp, so every warp mixes all three.
  alignas(16) unsigned char buf[32];
  Shape* s;
  switch (t % 3) {
    case 0: s = new (buf) Square(float(t)); break;
    case 1: s = new (buf) Triangle(float(t), 2.0f); break;
    default: s = new (buf) Circle(float(t)); break;
  }
  areas[t] = s->area();
  sides[t] = s->sides();
  s->~Shape();

  // A pointer table indexed per lane, called from diverged code too.
  op_t f = table[(t * 7) % 4];
  int r = f(t, 3);
  if (t & 1) r = table[t % 4](r, 1);
  ops[t] = r;

  // The warp is back together: a vote sees every lane.
  after[t] = __popc(__ballot_sync(0xffffffffu, true));
}

static int host_op(int k, int a, int b) {
  switch (k) {
    case 0: return a + b;
    case 1: return a - b;
    case 2: return a * b;
    default: return (a << 4) ^ b;
  }
}

int main() {
  float* da;
  int *ds, *dop, *daf;
  cudaMalloc(&da, kThreads * sizeof(float));
  cudaMalloc(&ds, kThreads * sizeof(int));
  cudaMalloc(&dop, kThreads * sizeof(int));
  cudaMalloc(&daf, kThreads * sizeof(int));
  kernel<<<1, kThreads>>>(da, ds, dop, daf);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    std::printf("kernel failed: %s\n", cudaGetErrorString(e));
    return 1;
  }
  float a[kThreads];
  int s[kThreads], o[kThreads], af[kThreads];
  cudaMemcpy(a, da, sizeof a, cudaMemcpyDeviceToHost);
  cudaMemcpy(s, ds, sizeof s, cudaMemcpyDeviceToHost);
  cudaMemcpy(o, dop, sizeof o, cudaMemcpyDeviceToHost);
  cudaMemcpy(af, daf, sizeof af, cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int t = 0; t < kThreads; ++t) {
    const float ft = float(t);
    const float wa = t % 3 == 0 ? ft * ft : t % 3 == 1 ? 0.5f * ft * 2.0f : 3.0f * ft * ft;
    const int ws = t % 3 == 0 ? 4 : t % 3 == 1 ? 3 : 0;
    int wo = host_op((t * 7) % 4, t, 3);
    if (t & 1) wo = host_op(t % 4, wo, 1);
    if (a[t] != wa && bad++ < 20) std::printf("thread %d area: got %g, want %g\n", t, a[t], wa);
    if (s[t] != ws && bad++ < 20) std::printf("thread %d sides: got %d, want %d\n", t, s[t], ws);
    if (o[t] != wo && bad++ < 20) std::printf("thread %d op: got %d, want %d\n", t, o[t], wo);
    if (af[t] != 32 && bad++ < 20) std::printf("thread %d ballot after: got %d\n", t, af[t]);
  }
  if (bad) {
    std::printf("%d mismatches\nFAIL\n", bad);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
