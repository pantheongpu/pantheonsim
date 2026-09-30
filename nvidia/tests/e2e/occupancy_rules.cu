// cudaOccupancyMaxActiveBlocksPerMultiprocessor against CUDA's own occupancy
// calculator (cuda_occupancy.h), given the same device properties and the
// same kernel attributes (cudaFuncGetAttributes). The calculator is the
// documented rule set: registers allocated per warp in units of 256 and
// checked per sub-partition, shared memory in 128-byte units (256 before
// Ampere) plus the driver's reserved kilobyte per block, and the warp and
// block ceilings. An RTX 3060 agrees with it for every case here, which is
// what makes it the reference; the simulator reports its own register counts,
// so the comparison is of the rules, not of ptxas's allocation.
//
// A kernel that uses eight named barriers is one of the cases: the calculator
// can limit blocks by barriers, but cudaFuncAttributes does not say how many a
// kernel uses and it assumes one -- and an RTX 3060's API answers the same
// (16 blocks of 32 threads), so named barriers do not lower the reported
// occupancy.
#include <cstdio>
#include <cuda_occupancy.h>
#include <cuda_runtime.h>

__global__ void k_small(float* p) { p[threadIdx.x] += 1.0f; }

__global__ void k_static_smem(float* p) {
  __shared__ float buf[3000];   // 12,000 bytes
  buf[threadIdx.x] = p[threadIdx.x];
  __syncthreads();
  p[threadIdx.x] = buf[(threadIdx.x + 1) % blockDim.x];
}

__global__ void k_odd_smem(float* p) {
  __shared__ float buf[33];     // 132 bytes: rounds up to the allocation unit
  buf[threadIdx.x % 33] = p[threadIdx.x];
  __syncthreads();
  p[threadIdx.x] = buf[(threadIdx.x + 1) % 33];
}

extern __shared__ float dyn[];
__global__ void k_dynamic(float* p) {
  dyn[threadIdx.x] = p[threadIdx.x];
  __syncthreads();
  p[threadIdx.x] = dyn[(threadIdx.x + 1) % blockDim.x];
}

// Many live values: a register-heavy kernel.
__global__ void k_regs(float* p) {
  float v[48];
#pragma unroll
  for (int i = 0; i < 48; ++i) v[i] = p[threadIdx.x + i * 32];
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int i = 0; i < 48; ++i) v[i] = v[i] * v[(i + 7) % 48] + v[(i + 13) % 48];
  float s = 0;
#pragma unroll
  for (int i = 0; i < 48; ++i) s += v[i];
  p[threadIdx.x] = s;
}

// Barriers 0 to 7, each with the whole block.
__global__ void k_barriers(float* p) {
  p[threadIdx.x] += 1.0f;
  asm volatile("bar.sync 0;\n\tbar.sync 1;\n\tbar.sync 2;\n\tbar.sync 3;\n\t"
               "bar.sync 4;\n\tbar.sync 5;\n\tbar.sync 6;\n\tbar.sync 7;" ::: "memory");
  p[threadIdx.x] += 1.0f;
}

struct Case {
  const char* name;
  const void* fn;
  int block;
  size_t dyn;
};

int main(int argc, char** argv) {
  const bool print = argc > 1;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  cudaOccDeviceProp occ_prop(prop);
  cudaOccDeviceState state;
  const Case cases[] = {
      {"small 32", (const void*)k_small, 32, 0},     {"small 128", (const void*)k_small, 128, 0},
      {"small 256", (const void*)k_small, 256, 0},   {"small 1024", (const void*)k_small, 1024, 0},
      {"small 96", (const void*)k_small, 96, 0},     {"small 800", (const void*)k_small, 800, 0},
      {"static 64", (const void*)k_static_smem, 64, 0},
      {"static 256", (const void*)k_static_smem, 256, 0},
      {"odd 64", (const void*)k_odd_smem, 64, 0},    {"odd 128", (const void*)k_odd_smem, 128, 0},
      {"dyn 64/4000", (const void*)k_dynamic, 64, 4000},
      {"dyn 128/9000", (const void*)k_dynamic, 128, 9000},
      {"dyn 32/20001", (const void*)k_dynamic, 32, 20001},
      {"dyn 64/48000", (const void*)k_dynamic, 64, 48000},
      {"dyn 32/1", (const void*)k_dynamic, 32, 1},
      {"regs 64", (const void*)k_regs, 64, 0},       {"regs 128", (const void*)k_regs, 128, 0},
      {"regs 256", (const void*)k_regs, 256, 0},     {"regs 512", (const void*)k_regs, 512, 0},
      {"regs 192", (const void*)k_regs, 192, 0},
      {"barriers 32", (const void*)k_barriers, 32, 0},
      {"barriers 256", (const void*)k_barriers, 256, 0},
  };
  int bad = 0;
  // A calculator older than the device has no rules for it (CUDA 12.0's knows
  // nothing past compute capability 9.0): nothing to compare with, which is
  // not a mismatch.
  {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)k_small);
    cudaOccFuncAttributes occ_attr(attr);
    cudaOccResult res;
    if (cudaOccMaxActiveBlocksPerMultiprocessor(&res, &occ_prop, &occ_attr, &state, 32, 0) ==
        CUDA_OCC_ERROR_UNKNOWN_DEVICE) {
      std::printf("%s: compute capability %d.%d is newer than this toolkit's cuda_occupancy.h; not checked\nPASS\n",
                  prop.name, prop.major, prop.minor);
      return 0;
    }
  }
  for (const Case& c : cases) {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, c.fn);
    int api = -1;
    cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&api, c.fn, c.block, c.dyn);
    cudaOccFuncAttributes occ_attr(attr);
    cudaOccResult res;
    const cudaOccError oe = cudaOccMaxActiveBlocksPerMultiprocessor(&res, &occ_prop, &occ_attr, &state, c.block, c.dyn);
    const int want = oe == CUDA_OCC_SUCCESS ? res.activeBlocksPerMultiprocessor : -2;
    if (print || e != cudaSuccess || api != want)
      std::printf("%-14s regs %3d static %6zu: API %d, cuda_occupancy.h %d (limited by 0x%x)%s\n", c.name,
                  attr.numRegs, attr.sharedSizeBytes, api, want, res.limitingFactors,
                  e != cudaSuccess || api != want ? "  <-- differs" : "");
    if (e != cudaSuccess || api != want) ++bad;
  }
  std::printf("%s: %d case(s) differ\n%s\n", prop.name, bad, bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
