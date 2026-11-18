// The second translation unit of host_gpu_exact.cu: its module has a __device__ variable, and its kernel calls assert, which keeps it in the
// simulator in exact mode (turbo mode sends it since section 20: a failed assert on the card ends the context as it ends the simulator's; it does not fail here). Until Round sp66 the variable alone refused every kernel of the module;
// now the card is given the simulator's address of the variable (vgpu/host_gpu_image.hpp) and only what a kernel itself calls decides.
#include <cassert>

__device__ int g_counter;
__global__ void uses_global(int* p) {
  assert(p != nullptr);
  if (threadIdx.x == 0) p[blockIdx.x] = atomicAdd(&g_counter, 1) >= 0;
}
void launch_uses_global(int* p) { uses_global<<<4, 32>>>(p); }
