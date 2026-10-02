// A kernel that reads %pm0, a performance-monitor counter. On a GPU that is
// whatever a profiler set it counting; VirtualGPU has no timing model, so it
// refuses the register by name, from the PTX and from the SASS (SR_PM0)
// alike. run_sass_path.sh checks both refusals.
#include <cstdio>

__global__ void read_pm0(unsigned* out) {
  unsigned v;
  asm volatile("mov.u32 %0, %%pm0;" : "=r"(v));
  *out = v;
}

int main() {
  unsigned* d;
  cudaMalloc(&d, sizeof(unsigned));
  read_pm0<<<1, 1>>>(d);
  const cudaError_t e = cudaDeviceSynchronize();
  std::printf("%s\n", cudaGetErrorString(e));
  return e == cudaSuccess ? 0 : 1;
}
