// Two kernels built to a cubin (nvcc -cubin) for cupti_resource.cu to load
// through the driver API: the module whose code its callbacks hand over.
extern "C" __global__ void module_fill(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 1.f;
}
extern "C" __global__ void module_scale(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] *= 3.f;
}
