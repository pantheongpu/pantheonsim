// Summation-tree probe of a cuDNN forward convolution (the same idea as split_probe.cu): a convolution whose filter covers its whole input (no padding,
// stride 1) is a matrix product y[n,k] = sum over p = (c, r, s) of x[n,p] * w[k,p] with K = C * R * R, and the output element (t, t) is probed with the big
// products +2^30 at p = a and -2^30 at p = b and every other product 2^-8. The count of ones that survive is K minus the size of the smallest sub-sum
// that holds both a and b (see split_probe.cu). FP32 tensors, the default math mode (TF32 is allowed on Ampere and later, as PyTorch's default has it),
// the algorithm cuDNN's own heuristic ranks first.
//   conv_probe <C> <R> <probes> <a> <b-first> <b-last> [fma]     (the optional word: CUDNN_FMA_MATH, no TF32)
//   conv_probe algo <C> <R> <probes>                             prints the ranked algorithms and their math types
#include <cudnn.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudnnStatus_t s_ = (x); if (s_ != CUDNN_STATUS_SUCCESS) { std::fprintf(stderr, "%s: %s\n", #x, cudnnGetErrorString(s_)); std::exit(1); } } while (0)

int main(int argc, char** argv) {
  const bool algo_mode = argc >= 2 && !std::strcmp(argv[1], "algo");
  const int o = algo_mode ? 2 : 1;
  if (argc < o + 3) { std::fprintf(stderr, "usage\n"); return 2; }
  const int C = std::atoi(argv[o]), R = std::atoi(argv[o + 1]), P = std::atoi(argv[o + 2]);
  const int K = C * R * R;
  const bool fma = !algo_mode && argc > 7 && !std::strcmp(argv[7], "fma");
  cudnnHandle_t h;
  CK(cudnnCreate(&h));
  cudnnTensorDescriptor_t xd, yd;
  cudnnFilterDescriptor_t wd;
  cudnnConvolutionDescriptor_t cd;
  CK(cudnnCreateTensorDescriptor(&xd)); CK(cudnnCreateTensorDescriptor(&yd)); CK(cudnnCreateFilterDescriptor(&wd)); CK(cudnnCreateConvolutionDescriptor(&cd));
  CK(cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, P, C, R, R));
  CK(cudnnSetFilter4dDescriptor(wd, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, P, C, R, R));
  CK(cudnnSetConvolution2dDescriptor(cd, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CK(cudnnSetConvolutionMathType(cd, fma ? CUDNN_FMA_MATH : CUDNN_DEFAULT_MATH));
  CK(cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, P, P, 1, 1));
  int n_res = 0;
  cudnnConvolutionFwdAlgoPerf_t perf[16];
  CK(cudnnGetConvolutionForwardAlgorithm_v7(h, xd, wd, cd, yd, 16, &n_res, perf));
  if (algo_mode) {
    for (int i = 0; i < n_res; ++i) std::printf("algo %d status %d math %d\n", (int)perf[i].algo, (int)perf[i].status, (int)perf[i].mathType);
    return 0;
  }
  const int a = std::atoi(argv[4]), b0 = std::atoi(argv[5]), b1 = std::atoi(argv[6]);
  const float kOne = 0.00390625f, big = 32768.f;
  std::vector<float> X((size_t)P * K, 1.f), W((size_t)P * K, kOne), Y((size_t)P * P);
  float *dX, *dW, *dY, *ws;
  size_t wsz = 0;
  cudaMalloc(&dX, X.size() * 4); cudaMalloc(&dW, W.size() * 4); cudaMalloc(&dY, Y.size() * 4);
  CK(cudnnGetConvolutionForwardWorkspaceSize(h, xd, wd, cd, yd, perf[0].algo, &wsz));
  cudaMalloc(&ws, wsz + 1);
  for (int b = b0; b <= b1;) {
    std::vector<int> bs;
    for (; b <= b1 && (int)bs.size() < P; ++b) if (b != a) bs.push_back(b);
    if (bs.empty()) break;
    for (size_t t = 0; t < bs.size(); ++t) {
      X[t * K + a] = big; X[t * K + bs[t]] = big; W[t * K + a] = big; W[t * K + bs[t]] = -big;
    }
    cudaMemcpy(dX, X.data(), X.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, W.data(), W.size() * 4, cudaMemcpyHostToDevice);
    const float al = 1.f, be = 0.f;
    CK(cudnnConvolutionForward(h, &al, xd, dX, wd, dW, cd, perf[0].algo, ws, wsz, &be, yd, dY));
    cudaDeviceSynchronize();
    cudaMemcpy(Y.data(), dY, Y.size() * 4, cudaMemcpyDeviceToHost);
    for (size_t t = 0; t < bs.size(); ++t) {
      const float v = Y[t * P + t] / kOne;
      const long r = std::lround(v);
      std::printf("%d %d %ld\n", a, bs[t], std::fabs(v - (float)r) < 1e-3f ? r : -1L);
      X[t * K + a] = 1.f; X[t * K + bs[t]] = 1.f; W[t * K + a] = kOne; W[t * K + bs[t]] = kOne;
    }
  }
  return 0;
}
