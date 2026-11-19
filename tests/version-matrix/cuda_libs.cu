// Version-matrix check: a kernel and three of the libraries the simulator stands in for (cuBLAS, cuFFT, cuRAND),
// each with an answer that is known without a GPU. Built with `nvcc -cudart shared` against whichever CUDA
// toolkit the matrix leg installed, and run with `vgpu run`. Exits 0 and prints PASS, or says what differed.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand.h>

#include <cmath>
#include <cstdio>
#include <vector>

#define CK(call)                                                                                   \
  do {                                                                                             \
    if ((call) != 0) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #call); return 1; }     \
  } while (0)

__global__ void iota(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 3 * i + 1;
}

int main() {
  int rt = 0, drv = 0;
  CK(cudaRuntimeGetVersion(&rt));
  CK(cudaDriverGetVersion(&drv));
  std::printf("runtime %d driver %d\n", rt, drv);

  // Kernel.
  const int n = 1000;
  int* di;
  CK(cudaMalloc(&di, n * sizeof(int)));
  iota<<<(n + 127) / 128, 128>>>(di, n);
  CK(cudaDeviceSynchronize());
  std::vector<int> hi(n);
  CK(cudaMemcpy(hi.data(), di, n * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i)
    if (hi[i] != 3 * i + 1) { std::printf("FAIL kernel at %d\n", i); return 1; }

  // cuBLAS: C = A * B for column-major 2x2 matrices, every product exact in float.
  const float a[4] = {1, 2, 3, 4}, b[4] = {5, 6, 7, 8};  // A = [1 3; 2 4], B = [5 7; 6 8]
  const float want[4] = {23, 34, 31, 46};                // A*B = [23 31; 34 46]
  float *da, *db, *dc;
  CK(cudaMalloc(&da, sizeof a));
  CK(cudaMalloc(&db, sizeof b));
  CK(cudaMalloc(&dc, sizeof a));
  CK(cudaMemcpy(da, a, sizeof a, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db, b, sizeof b, cudaMemcpyHostToDevice));
  cublasHandle_t h;
  CK(cublasCreate(&h));
  const float one = 1, zero = 0;
  CK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &one, da, 2, db, 2, &zero, dc, 2));
  float c[4];
  CK(cudaMemcpy(c, dc, sizeof c, cudaMemcpyDeviceToHost));
  for (int i = 0; i < 4; ++i)
    if (c[i] != want[i]) { std::printf("FAIL sgemm %d: %g want %g\n", i, c[i], want[i]); return 1; }
  CK(cublasDestroy(h));

  // cuFFT: the transform of a unit impulse is all ones.
  const int m = 64;
  std::vector<cufftComplex> x(m, cufftComplex{0.f, 0.f});
  x[0].x = 1.f;
  cufftComplex* dx;
  CK(cudaMalloc(&dx, m * sizeof(cufftComplex)));
  CK(cudaMemcpy(dx, x.data(), m * sizeof(cufftComplex), cudaMemcpyHostToDevice));
  cufftHandle plan;
  CK(cufftPlan1d(&plan, m, CUFFT_C2C, 1));
  CK(cufftExecC2C(plan, dx, dx, CUFFT_FORWARD));
  CK(cudaMemcpy(x.data(), dx, m * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
  for (int i = 0; i < m; ++i)
    if (std::fabs(x[i].x - 1.f) > 1e-5f || std::fabs(x[i].y) > 1e-5f) { std::printf("FAIL fft bin %d\n", i); return 1; }
  CK(cufftDestroy(plan));

  // cuRAND: uniform floats are in (0, 1], and the same seed gives the same numbers twice.
  float *dr;
  CK(cudaMalloc(&dr, 2 * 256 * sizeof(float)));
  curandGenerator_t g;
  CK(curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT));
  CK(curandSetPseudoRandomGeneratorSeed(g, 1234));
  CK(curandGenerateUniform(g, dr, 256));
  CK(curandSetPseudoRandomGeneratorSeed(g, 1234));
  CK(curandGenerateUniform(g, dr + 256, 256));
  std::vector<float> r(512);
  CK(cudaMemcpy(r.data(), dr, 512 * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 256; ++i) {
    if (!(r[i] > 0.f && r[i] <= 1.f)) { std::printf("FAIL curand range %d: %g\n", i, r[i]); return 1; }
    if (r[i] != r[256 + i]) { std::printf("FAIL curand repeat %d\n", i); return 1; }
  }
  CK(curandDestroyGenerator(g));
  std::printf("PASS\n");
  return 0;
}
