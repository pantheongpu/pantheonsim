// setmaxnreg: a warpgroup takes more registers than the kernel was compiled
// for. Launched with 384 threads, the kernel gets 168 registers a thread; the
// third warpgroup gives registers back (USETMAXREG.DEALLOC 40) and the first
// two take 232 (USETMAXREG.TRY_ALLOC), after which ptxas keeps 160 live
// values in registers up to R229. CUTLASS's warp-specialized Hopper GEMMs are
// this shape (consumers at 232, the producer at 40).
#include <cstdio>

constexpr int kThreads = 384, kLive = 160, kIn = 512;

__global__ void __launch_bounds__(kThreads, 1) grow(const float* in, float* out) {
  const int t = threadIdx.x;
  if (t >= 256) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)   // setmaxnreg is sm_90a's
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
#endif
    return;
  }
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
#endif
  float acc[kLive];
#pragma unroll
  for (int i = 0; i < kLive; ++i) acc[i] = in[(t + i * 7) % kIn];
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int i = 0; i < kLive; ++i) acc[i] = acc[i] * 1.0009765625f + acc[(i + 1) % kLive];
  float s = 0;
#pragma unroll
  for (int i = 0; i < kLive; ++i) s += acc[i] * (i + 1);
  out[t] = s;
}

int main() {
  float h_in[kIn], h_out[256], *d_in, *d_out;
  for (int i = 0; i < kIn; ++i) h_in[i] = static_cast<float>((i * 37) % 101) / 64.0f;
  cudaMalloc(&d_in, sizeof h_in);
  cudaMalloc(&d_out, sizeof h_out);
  cudaMemcpy(d_in, h_in, sizeof h_in, cudaMemcpyHostToDevice);
  grow<<<1, kThreads>>>(d_in, d_out);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::printf("FAIL %s\n", cudaGetErrorString(cudaGetLastError()));
    return 1;
  }
  cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int t = 0; t < 256; ++t) {
    float acc[kLive];
    for (int i = 0; i < kLive; ++i) acc[i] = h_in[(t + i * 7) % kIn];
    for (int r = 0; r < 4; ++r)
      for (int i = 0; i < kLive; ++i) acc[i] = acc[i] * 1.0009765625f + acc[(i + 1) % kLive];
    float s = 0;
    for (int i = 0; i < kLive; ++i) s += acc[i] * (i + 1);
    const float tol = 1e-4f * (s < 0 ? -s : s) + 1e-3f;
    if (!(h_out[t] - s <= tol && s - h_out[t] <= tol)) {
      if (bad++ < 4) std::printf("thread %d: %g, want %g\n", t, h_out[t], s);
    }
  }
  std::printf(bad ? "FAIL %d threads\n" : "PASS\n", bad);
  return bad ? 1 : 0;
}
