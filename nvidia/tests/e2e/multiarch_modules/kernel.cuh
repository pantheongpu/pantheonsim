// One template kernel, instantiated by translation units built for different architectures. The instantiation's
// host stub is a weak symbol, so every unit that instantiates it registers the same stub with its own fatbin.
#pragma once
template <int V>
__global__ void tier_kernel(int* out) {
  out[threadIdx.x] = V * 100 + static_cast<int>(threadIdx.x);
}

// Each returns the address of an instantiation's host stub (what <<<>>>, cudaLaunchKernel and cudaGetFuncBySymbol
// take); the unit's own code is the only thing that sees the kernel.
extern "C" const void* base_stub_shared();     // tier_kernel<7>, in a unit built for sm_80 and sm_90
extern "C" const void* hopper_stub_shared();   // tier_kernel<7>, also in the sm_90-only unit
extern "C" const void* hopper_stub_only();     // tier_kernel<9>, in the sm_90-only unit alone
