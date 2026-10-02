// The device half of managed_module_load: __managed__ variables in a module
// the driver API loads. run_managed_module.sh builds it three ways -- a cubin
// (nvcc -cubin -arch=sm_86), a fatbin holding sm_86 SASS and compute_86 PTX,
// and bare PTX -- and the program loads each with cuModuleLoadData.
__managed__ int counter = 5;
__managed__ float scaled[4] = {1, 2, 3, 4};
__device__ int plain = 3;

extern "C" __global__ void bump(int by) {
  counter += by + plain;
  for (int i = 0; i < 4; ++i) scaled[i] *= 2;
}
