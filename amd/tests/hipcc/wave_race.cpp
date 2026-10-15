// The waves of a work-group on a card leave a barrier within a few cycles of one another, so a word every wave
// reads right after it is read by all of them before anything one wave computes from it lands. Run to its next
// barrier one wave at a time, wave 0 had overwritten the word before wave 1 read it.
//
// rocSOLVER's Cholesky kernel (potf2_kernel_small, ROCm 7.1) has this shape in every iteration: all the threads read
// the diagonal entry, thread 0 overwrites it with its square root after a double-precision sqrt, and the threads then
// scale the column with the square root each of them took of what it read. By the letter of the memory model that is a
// race, but it is exact on a card, and on the simulator it gave a wrong factor for every matrix of order 66 or more
// (the order at which thread 64, the first thread of the second wave, has a column to scale): hipsolverDnDpotrf
// returned status 0 and a Cholesky factor that was wrong in the third decimal place.
//
// The program has two kernels, each run with 1 to 16 waves of 64:
//   - one iteration of that shape;
//   - four of them, a barrier apart, each square-rooting what the last one left, so that the order is also right in a
//     round that is not the kernel's first.
// Every thread must take the square root of the value as it was before thread 0 overwrote it.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>

__global__ void one_iteration(double* out) {
  extern __shared__ double s[];
  if (threadIdx.x == 0) s[0] = 16.0;
  __syncthreads();
  const double akk = s[0];
  const double lkk = sqrt(akk);
  if (threadIdx.x == 0) s[0] = lkk;
  __syncthreads();
  out[threadIdx.x] = lkk;
}

__global__ void four_iterations(double* out) {
  extern __shared__ double s[];
  if (threadIdx.x == 0) s[0] = 65536.0;
  __syncthreads();
  for (int it = 0; it < 4; ++it) {
    const double akk = s[0];
    const double lkk = sqrt(akk);
    if (threadIdx.x == 0) s[0] = lkk;
    __syncthreads();
    out[it * blockDim.x + threadIdx.x] = lkk;
    __syncthreads();
  }
}

int main() {
  double* d;
  if (hipMalloc(&d, 4 * 1024 * sizeof(double)) != hipSuccess) return 2;
  int checks = 0, holding = 0;
  for (int threads : {64, 128, 256, 512, 1024}) {
    static double o[4 * 1024];

    one_iteration<<<1, threads, sizeof(double)>>>(d);
    hipDeviceSynchronize();
    hipMemcpy(o, d, threads * sizeof(double), hipMemcpyDeviceToHost);
    int wrong = 0;
    for (int t = 0; t < threads; ++t) wrong += o[t] != 4.0;
    ++checks, holding += wrong == 0;
    std::printf("one iteration, %4d threads (%2d waves): %d threads took the root of an overwritten value\n", threads,
                threads / 64, wrong);

    four_iterations<<<1, threads, sizeof(double)>>>(d);
    hipDeviceSynchronize();
    hipMemcpy(o, d, 4 * threads * sizeof(double), hipMemcpyDeviceToHost);
    const double want[4] = {256.0, 16.0, 4.0, 2.0};
    wrong = 0;
    for (int it = 0; it < 4; ++it)
      for (int t = 0; t < threads; ++t) wrong += o[it * threads + t] != want[it];
    ++checks, holding += wrong == 0;
    std::printf("four iterations, %4d threads (%2d waves): %d values wrong\n", threads, threads / 64, wrong);
  }
  hipFree(d);
  std::printf("%d of %d wave-order checks hold\n", holding, checks);
  return holding == checks ? 0 : 1;
}
