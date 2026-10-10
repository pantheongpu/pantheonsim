// The kernels the driver sweep's attribute cases ask about (exports_sweep_driver.cpp reads the cubin as libf.cubin):
// static shared memory of several sizes, dynamic shared memory, launch bounds, enough registers to limit the block, and
// module constants. The same shapes as the runtime sweep's (exports_sweep_runtime.cu).
extern "C" {
__global__ void fa_dyn(int* p) {
  extern __shared__ char dsm[];
  dsm[threadIdx.x] = 1;
  __syncthreads();
  p[0] = dsm[0];
}
__global__ void fa_dyn2(int* p) {
  extern __shared__ char dsm[];
  dsm[threadIdx.x] = 1;
  __syncthreads();
  p[0] = dsm[0];
}
__global__ void fa_s256(int* p) {
  __shared__ char s[256];
  s[threadIdx.x % 256] = static_cast<char>(threadIdx.x);
  __syncthreads();
  p[0] = s[1];
}
__global__ void fa_s40000(int* p) {
  __shared__ char s[40000];
  s[threadIdx.x % 40000] = static_cast<char>(threadIdx.x);
  __syncthreads();
  p[0] = s[1];
}
__global__ void fa_s49152(int* p) {
  __shared__ char s[49152];
  s[threadIdx.x % 49152] = static_cast<char>(threadIdx.x);
  __syncthreads();
  p[0] = s[1];
}
__global__ void fa_three(int* a, float b, double* c) { a[0] = static_cast<int>(b); c[0] = 1.0; }
__global__ void fa_none() {}
__global__ __launch_bounds__(128) void fa_lb128(int* p) { p[0] = 1; }
__global__ __launch_bounds__(512, 2) void fa_lb512(int* p) { p[0] = 1; }
}

template <int N> __device__ void heavy(float* o) {
  float a[N];
#pragma unroll
  for (int i = 0; i < N; i++) a[i] = o[i * 32 + threadIdx.x];
  float s = 0;
#pragma unroll
  for (int i = 0; i < N; i++) s += a[i] * a[(i * 7 + 1) % N];
#pragma unroll
  for (int i = 0; i < N; i++) o[i * 32 + threadIdx.x] = a[i] * s + a[(i + 1) % N];
}
extern "C" __global__ void fa_heavy400(float* o) { heavy<400>(o); }

__constant__ int ctab[100];
__constant__ char cflag[5];
__constant__ double cwide[3];
extern "C" __global__ void fa_const(int* p) { p[0] = ctab[threadIdx.x] + cflag[1] + static_cast<int>(cwide[1]); }
