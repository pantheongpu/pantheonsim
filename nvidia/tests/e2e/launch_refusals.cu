// Which error each refused launch configuration reports, in the runtime API (cudaLaunchKernel, cudaLaunchKernelExC) and the driver
// API (cuLaunchKernel, cuLaunchKernelEx), on an RTX 3060's CUDA 13.0 (nvidia/tests/data/launch_refusals.rtx3060.txt). A block or grid
// the device cannot run, a zero dimension, dynamic shared memory past the limit and a block past __launch_bounds__ are
// cudaErrorInvalidValue (1); a block whose registers do not fit the multiprocessor is cudaErrorLaunchOutOfResources (701); any
// cluster launch on a part without clusters is cudaErrorInvalidClusterSize (912). The same codes in the driver. The simulator
// answered cudaErrorInvalidConfiguration (9) for the first group. run_launch_refusals.sh compares, or --card [--update].
#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void trivial(int* p) { if (p && threadIdx.x == 12345) p[0] = 1; }
__global__ void __launch_bounds__(128) bounded(int* p) { if (p && threadIdx.x == 12345) p[0] = 1; }
__global__ void heavy(const float* in, float* out) {
  float v[160];
#pragma unroll
  for (int i = 0; i < 160; ++i) v[i] = in[(threadIdx.x + i * 7) & 1023] * (i + 1);
  float s = 0;
#pragma unroll
  for (int i = 0; i < 160; ++i) s += v[i] * v[(i * 13 + threadIdx.x) % 160 == i ? i : (i + 1) % 160];
  out[threadIdx.x] = s;
}
__global__ void statsm(int* p) {
  __shared__ char s[40000];
  extern __shared__ char d[];
  s[threadIdx.x] = (char)threadIdx.x; d[threadIdx.x] = s[threadIdx.x];
  if (p && d[0] == 123 && threadIdx.x == 777) p[0] = d[1];
}
__global__ void dynsm(int* p) {
  extern __shared__ char d[];
  d[threadIdx.x] = (char)threadIdx.x;
  if (p && d[0] == 123 && threadIdx.x == 777) p[0] = d[1];
}

static void show(const char* name, cudaError_t rt, CUresult dr) {
  std::printf("%-44s runtime %4d %-34s driver %4d\n", name, (int)rt, cudaGetErrorName(rt), (int)dr);
}

struct Case {
  const char* name;
  const void* fn;
  dim3 grid, block;
  size_t smem;
  int cluster[3];   // 0 = none
};

int main() {
  int* d; cudaMalloc(&d, 4096);
  float* fin; cudaMalloc(&fin, 4096 * 4);
  const void* T = (const void*)trivial;
  const Case cases[] = {
    {"ok 1x1", T, dim3(1), dim3(1), 0, {0, 0, 0}},
    {"block.x 1025", T, dim3(1), dim3(1025), 0, {0, 0, 0}},
    {"block.x 2000", T, dim3(1), dim3(2000), 0, {0, 0, 0}},
    {"block.x 1u<<20", T, dim3(1), dim3(1u << 20), 0, {0, 0, 0}},
    {"block.y 1025", T, dim3(1), dim3(1, 1025, 1), 0, {0, 0, 0}},
    {"block.z 65", T, dim3(1), dim3(1, 1, 65), 0, {0, 0, 0}},
    {"block 32x32x2 (2048 threads)", T, dim3(1), dim3(32, 32, 2), 0, {0, 0, 0}},
    {"block.x 0", T, dim3(1), dim3(0, 1, 1), 0, {0, 0, 0}},
    {"block.y 0", T, dim3(1), dim3(1, 0, 1), 0, {0, 0, 0}},
    {"block.z 0", T, dim3(1), dim3(1, 1, 0), 0, {0, 0, 0}},
    {"grid.x 0", T, dim3(0, 1, 1), dim3(1), 0, {0, 0, 0}},
    {"grid.y 0", T, dim3(1, 0, 1), dim3(1), 0, {0, 0, 0}},
    {"grid.z 0", T, dim3(1, 1, 0), dim3(1), 0, {0, 0, 0}},
    {"grid.x 2^31", T, dim3(1u << 31, 1, 1), dim3(1), 0, {0, 0, 0}},
    {"grid.x 0xffffffff", T, dim3(0xffffffffu, 1, 1), dim3(1), 0, {0, 0, 0}},
    {"grid.y 65536", T, dim3(1, 65536, 1), dim3(1), 0, {0, 0, 0}},
    {"grid.z 65536", T, dim3(1, 1, 65536), dim3(1), 0, {0, 0, 0}},
    {"dynamic smem 49153 (kernel w/o static)", (const void*)dynsm, dim3(1), dim3(32), 49153, {0, 0, 0}},
    {"dynamic smem 1<<20", (const void*)dynsm, dim3(1), dim3(32), 1u << 20, {0, 0, 0}},
    {"dynamic smem 0xffffffff", (const void*)dynsm, dim3(1), dim3(32), 0xffffffffu, {0, 0, 0}},
    {"static 40000 + dynamic 16000", (const void*)statsm, dim3(1), dim3(32), 16000, {0, 0, 0}},
    {"heavy regs x 1024 threads", (const void*)heavy, dim3(1), dim3(1024), 0, {0, 0, 0}},
    {"heavy regs x 512 threads", (const void*)heavy, dim3(1), dim3(512), 0, {0, 0, 0}},
    {"launch_bounds(128) launched with 256", (const void*)bounded, dim3(1), dim3(256), 0, {0, 0, 0}},
    {"launch_bounds(128) launched with 129", (const void*)bounded, dim3(1), dim3(129), 0, {0, 0, 0}},
    {"cluster 2 grid 4", T, dim3(4), dim3(32), 0, {2, 1, 1}},
    {"cluster 3 grid 4 (not a divisor)", T, dim3(4), dim3(32), 0, {3, 1, 1}},
    {"cluster 0 dims", T, dim3(4), dim3(32), 0, {0, 1, 1}},
    {"cluster 9 grid 9 (past portable 8)", T, dim3(9), dim3(32), 0, {9, 1, 1}},
    {"cluster 16 grid 16", T, dim3(16), dim3(32), 0, {16, 1, 1}},
  };
  for (const Case& c : cases) {
    void* args1[] = {&d};
    void* args2[] = {&fin, &d};
    void** args = (c.fn == (const void*)heavy) ? args2 : args1;
    cudaError_t rt;
    cudaGetLastError();
    if (c.cluster[0] || c.cluster[1] || c.cluster[2] ||
        (!c.cluster[0] && c.name[0] == 'c' && c.name[1] == 'l')) {
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = c.grid; cfg.blockDim = c.block; cfg.dynamicSmemBytes = c.smem;
      cudaLaunchAttribute at[1];
      at[0].id = cudaLaunchAttributeClusterDimension;
      at[0].val.clusterDim.x = c.cluster[0]; at[0].val.clusterDim.y = c.cluster[1]; at[0].val.clusterDim.z = c.cluster[2];
      cfg.attrs = at; cfg.numAttrs = 1;
      rt = cudaLaunchKernelExC(&cfg, c.fn, args);
      CUresult dr = CUDA_SUCCESS;
      CUfunction f; cudaGetFuncBySymbol(&f, c.fn);
      CUlaunchConfig lc = {};
      lc.gridDimX = c.grid.x; lc.gridDimY = c.grid.y; lc.gridDimZ = c.grid.z;
      lc.blockDimX = c.block.x; lc.blockDimY = c.block.y; lc.blockDimZ = c.block.z; lc.sharedMemBytes = c.smem;
      CUlaunchAttribute la[1] = {};
      la[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      la[0].value.clusterDim.x = c.cluster[0]; la[0].value.clusterDim.y = c.cluster[1]; la[0].value.clusterDim.z = c.cluster[2];
      lc.attrs = la; lc.numAttrs = 1;
      dr = cuLaunchKernelEx(&lc, f, args, nullptr);
      show(c.name, rt, dr);
    } else {
      rt = cudaLaunchKernel(c.fn, c.grid, c.block, args, c.smem, 0);
      CUfunction f; cudaGetFuncBySymbol(&f, c.fn);
      CUresult dr = cuLaunchKernel(f, c.grid.x, c.grid.y, c.grid.z, c.block.x, c.block.y, c.block.z, (unsigned)c.smem, 0, args, nullptr);
      show(c.name, rt, dr);
    }
    cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) { std::printf("  sync -> %d %s\n", (int)sync, cudaGetErrorName(sync)); }
    cudaGetLastError();
  }
  // <<<>>> syntax
  trivial<<<1, 2000>>>(d);
  cudaError_t e = cudaGetLastError();
  std::printf("%-44s runtime %4d %s\n", "<<<1,2000>>> then cudaGetLastError", (int)e, cudaGetErrorName(e));
  trivial<<<1, 2000>>>(d);
  e = cudaPeekAtLastError();
  std::printf("%-44s runtime %4d %s\n", "<<<1,2000>>> then cudaPeekAtLastError", (int)e, cudaGetErrorName(e));
  e = cudaGetLastError();
  std::printf("%-44s runtime %4d %s\n", "  ...then cudaGetLastError", (int)e, cudaGetErrorName(e));
  e = cudaGetLastError();
  std::printf("%-44s runtime %4d %s\n", "  ...then cudaGetLastError again", (int)e, cudaGetErrorName(e));
  return 0;
}
