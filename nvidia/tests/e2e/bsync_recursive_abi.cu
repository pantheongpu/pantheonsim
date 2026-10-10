// A recursive device function called from a four-way unrolled loop whose lanes take different branches --
// the shape of PyTorch's bessel_j1_forward_vectorized4 kernel (bessel_j1(x) calls itself for x < 0, then
// takes a polynomial branch for small x, with its coefficient tables run through Horner loops, and an
// asymptotic branch with sin, cos and sqrt for large x).
//
// A recursive function has to follow the call ABI, so ptxas wraps it: the prologue saves the caller's
// convergence barriers (BMOV.32.CLEAR R26, B7 ; BMOV.32.CLEAR R25, B6), the epilogue puts them back
// (BMOV.32 B6, R25 ; BMOV.32 B7, R26), and the function calls itself between a BSSY B7 and its BSYNC.
// Lanes of one warp reach that one epilogue from different call levels, so they restore different masks and
// park at different BSYNCs, each waiting for a mask the other level owns. The simulator reported "every warp
// ... is waiting (a barrier some threads never reach)" until lanes parked at a BSYNC whose barrier no longer
// named them went on once nothing else could move (Runner::break_bsync_standoff, src/sass/exec.cpp). The
// float form is the one that stood off (built for sm_75: the BMOV pairs above); the double form does not on
// this ptxas and is here because the same code in double is what PyTorch's double kernel runs.
//
// The inputs mix negative and positive values of both magnitudes in every warp, so the lanes split at each
// branch. The results are computed on the host in double and compared to a tolerance: sinf and cosf of an
// argument of a few hundred differ between a card and a host by about 1e-6. The program passes on an RTX 3060.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("FAIL: %s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)

// depth bounds the recursion: the real function recurses once, for a negative argument.
template <typename T>
__host__ __device__ __noinline__ T bessel_j1(T x, int depth) {
  static const T PP[] = {7.62125616208173112003e-04, 7.31397056940917570436e-02, 1.12719608129684925192e+00, 5.11207951146807644818e+00, 8.42404590141772420927e+00, 5.21451598682361504063e+00, 1.00000000000000000254e+00};
  static const T PQ[] = {5.71323128072548699714e-04, 6.88455908754495404082e-02, 1.10514232634061696926e+00, 5.07386386128601488557e+00, 8.39985554327604159757e+00, 5.20982848682361821619e+00, 9.99999999999999997461e-01};
  static const T QP[] = {5.10862594750176621635e-02, 4.98213872951233449420e+00, 7.58238284132545283818e+01, 3.66779609360150777800e+02, 7.10856304998926107277e+02, 5.97489612400613639965e+02, 2.11688757100572135698e+02, 2.52070311549965906120e+01};
  static const T QQ[] = {1.00000000000000000000e+00, 7.42373277035675149943e+01, 1.05644886038262816351e+03, 4.98641058337653607651e+03, 9.56231892404756170795e+03, 7.99704160447350683650e+03, 2.82619278517639096600e+03, 3.36093750000000000000e+02};
  static const T RP[] = {-8.99971225705559398224e+08, 4.52228297998194034323e+11, -7.27494245221818276015e+13, 3.68295732863852883286e+15};
  static const T RQ[] = {1.00000000000000000000e+00, 6.20836478118054335476e+02, 2.56987256757748830383e+05, 8.35146791431949253037e+07, 2.21511595479792499675e+10, 4.74914122079991414898e+12, 7.84369607876235854894e+14, 8.95222336184627338078e+16};
  if (x < T(0.0) && depth > 0) return -bessel_j1(-x, depth - 1);
  if (x <= T(5.0)) {
    if (x == T(0.0)) return T(0.0);
    T rp = T(0.0);
    for (uint8_t index = 0; index <= 3; index++) rp = rp * (x * x) + RP[index];
    T rq = T(0.0);
    for (uint8_t index = 0; index <= 7; index++) rq = rq * (x * x) + RQ[index];
    return rp / rq * x * (x * x - T(1.46819706421238932572e+01)) * (x * x - T(4.92184563216946036703e+01));
  }
  T pp = T(0.0);
  for (uint8_t index = 0; index <= 6; index++) pp = pp * (T(25.0) / (x * x)) + PP[index];
  T pq = T(0.0);
  for (uint8_t index = 0; index <= 6; index++) pq = pq * (T(25.0) / (x * x)) + PQ[index];
  T qp = T(0.0);
  for (uint8_t index = 0; index <= 7; index++) qp = qp * (T(25.0) / (x * x)) + QP[index];
  T qq = T(0.0);
  for (uint8_t index = 0; index <= 7; index++) qq = qq * (T(25.0) / (x * x)) + QQ[index];
  return (pp / pq * cos(x - T(2.356194490192344928846982537459627163)) -
          T(5.0) / x * (qp / qq) * sin(x - T(2.356194490192344928846982537459627163))) *
         T(0.797884560802865355879892119868763737) / sqrt(x);
}

template <typename T>
__global__ void vec4(const T* in, T* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int e = i * 4 + j;
    if (e < n) out[e] = bessel_j1<T>(in[e], 1);
  }
}

static uint64_t lcg(uint64_t& s) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return s >> 33; }

template <typename T>
static int run(const char* name, const double* h, int n) {
  static T in[4096], got[4096];
  for (int i = 0; i < n; ++i) in[i] = static_cast<T>(h[i]);
  T *din = nullptr, *dout = nullptr;
  CK(cudaMalloc(&din, n * sizeof(T)));
  CK(cudaMalloc(&dout, n * sizeof(T)));
  CK(cudaMemcpy(din, in, n * sizeof(T), cudaMemcpyHostToDevice));
  vec4<T><<<(n / 4 + 127) / 128, 128>>>(din, dout, n);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(got, dout, n * sizeof(T), cudaMemcpyDeviceToHost));
  cudaFree(din);
  cudaFree(dout);
  int bad = 0;
  for (int i = 0; i < n; ++i) {
    const double want = bessel_j1<double>(static_cast<double>(in[i]), 1);
    if (std::fabs(static_cast<double>(got[i]) - want) > 1e-3 * (1 + std::fabs(want))) {
      if (bad++ < 5) std::printf("FAIL: %s element %d of %.9g got %.9g want %.9g\n", name, i, static_cast<double>(in[i]), static_cast<double>(got[i]), want);
    }
  }
  if (bad) { std::printf("FAIL: %s: %d of %d\n", name, bad, n); return 1; }
  return 0;
}

int main() {
  const int n = 4096;
  static double h[n];
  uint64_t s = 12345;
  for (int i = 0; i < n; ++i) {
    // negative or positive, small (<= 5) or large, with a few zeros: every warp holds all of them
    const uint64_t r = lcg(s);
    const double mag = (r & 1) ? 0.1 + double((r >> 1) % 49) / 10.0 : 5.5 + double((r >> 1) % 4000) / 7.0;
    h[i] = (r >> 20) % 17 == 0 ? 0.0 : ((r >> 12) & 1 ? -mag : mag);
  }
  if (run<float>("float", h, n)) return 1;
  if (run<double>("double", h, n)) return 1;
  std::printf("PASS: %d elements, float and double\n", n);
  return 0;
}
