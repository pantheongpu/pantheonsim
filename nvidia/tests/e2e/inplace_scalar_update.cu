// Every thread of a block adds one to the same word, in place: the shape of the kernel Inductor writes
// for `step += 1` on a PyTorch optimizer's step counter (a one-element tensor, a block of 1024 lanes
// that all load it, add and store it back).
//
// By the letter of the CUDA memory model the warps race. On a card they do not: every warp of a block
// starts within a few cycles of the others, so each has issued its load long before the first store
// lands, and every store writes the same value. The simulator used to run a warp to the end before the
// next one began, so warp k loaded what warps 0..k-1 had stored and the counter went up by the number
// of warps (8 for 256 threads); a compiled Adam step then used step 8 for its bias corrections and
// differed from the CPU's by 0.0085. Prints PASS on the last line.
#include <cstdio>

__global__ void bump(float* s) { s[0] = s[0] + 1.0f; }
__global__ void bump_int(int* s) { s[0] += 1; }
// The same through a scalar the thread loads into a register first, with other work in between.
__global__ void bump_scaled(float* s, float* out) {
  const float v = s[0];
  const float w = v * 2.0f + threadIdx.x * 0.0f;
  s[0] = w * 0.5f + 1.0f;
  out[threadIdx.x] = w;
}

int main() {
  int bad = 0;
  float* f;
  int* n;
  float* out;
  cudaMalloc(&f, sizeof(float));
  cudaMalloc(&n, sizeof(int));
  cudaMalloc(&out, 1024 * sizeof(float));
  for (int threads : {32, 64, 256, 1024}) {
    const float f0 = 3.0f;
    const int n0 = 10;
    cudaMemcpy(f, &f0, sizeof f0, cudaMemcpyHostToDevice);
    cudaMemcpy(n, &n0, sizeof n0, cudaMemcpyHostToDevice);
    bump<<<1, threads>>>(f);
    bump_int<<<1, threads>>>(n);
    float hf = 0;
    int hn = 0;
    cudaMemcpy(&hf, f, sizeof hf, cudaMemcpyDeviceToHost);
    cudaMemcpy(&hn, n, sizeof hn, cudaMemcpyDeviceToHost);
    if (hf != 4.0f) { std::printf("FAIL float counter with %d threads: %g, want 4\n", threads, hf); ++bad; }
    if (hn != 11) { std::printf("FAIL int counter with %d threads: %d, want 11\n", threads, hn); ++bad; }

    cudaMemcpy(f, &f0, sizeof f0, cudaMemcpyHostToDevice);
    bump_scaled<<<1, threads>>>(f, out);
    float ho = 0;
    cudaMemcpy(&hf, f, sizeof hf, cudaMemcpyDeviceToHost);
    cudaMemcpy(&ho, out + threads - 1, sizeof ho, cudaMemcpyDeviceToHost);
    if (hf != 4.0f || ho != 6.0f) { std::printf("FAIL scaled counter with %d threads: %g and %g, want 4 and 6\n", threads, hf, ho); ++bad; }
  }
  if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("FAIL %s\n", cudaGetErrorString(cudaGetLastError())); ++bad; }
  std::printf(bad ? "FAILED\n" : "PASS\n");
  return bad ? 1 : 0;
}
