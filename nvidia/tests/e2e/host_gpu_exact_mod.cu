// The second translation unit of host_gpu_exact.cu: its module has a __device__ variable, so the mode keeps its kernel in the simulator.
// A module of its own, because the refusal is per module: one variable keeps every kernel of its translation unit out.
__device__ int g_counter;
__global__ void uses_global(int* p) {
  if (threadIdx.x == 0) p[blockIdx.x] = atomicAdd(&g_counter, 1) >= 0;
}
void launch_uses_global(int* p) { uses_global<<<4, 32>>>(p); }
