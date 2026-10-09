// A small network trained across several processes, one simulated GPU each, the
// four ways large models are split over GPUs. Each rank is its own process (the
// way torchrun and mpirun launch jobs), so every collective crosses the file-backed
// NCCL transport, and the matrix products are cuBLAS's.
//
//   parallel_training <rank> <nranks> <idfile>
//
// The network is x(16) -> 48 ReLU -> 8, squared error, plain SGD for a few
// steps on a fixed batch of 96, with no bias. Every mode starts from the same
// weights and must end where a single process training on the whole batch
// ends, which the program works out itself in double precision:
//
//   data parallel   each rank a share of the batch; the gradients are summed
//                   with ncclAllReduce
//   tensor parallel the hidden layer split by columns and the output layer by
//                   rows (Megatron-style); one all-reduce of the output
//   pipeline        two stages, one layer each, activations and their
//                   gradients passed with ncclSend / ncclRecv (two ranks)
//   sharded         parameters and gradients in one flat vector, each rank
//                   owning a slice: ncclAllGather before the step,
//                   ncclReduceScatter after it (ZeRO / FSDP)
//
// Prints "ok <mode>" or "FAIL <mode>: <why>" per mode; exits non-zero on a FAIL.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
constexpr int B = 96, I = 16, H = 48, O = 8, STEPS = 5;
constexpr double LR = 0.05;

int rank_ = 0, nranks_ = 1;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  std::fprintf(stderr, "rank %d: %s -> %s\n", rank_, #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)
#define NK(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) { \
  std::fprintf(stderr, "rank %d: %s -> %s\n", rank_, #x, ncclGetErrorString(r_)); std::exit(2); } } while (0)
#define BK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  std::fprintf(stderr, "rank %d: %s -> %d\n", rank_, #x, (int)s_); std::exit(2); } } while (0)

// Reproducible numbers, the same on every rank.
struct Lcg {
  uint32_t s;
  explicit Lcg(uint32_t seed) : s(seed) {}
  float next() {
    s = s * 1664525u + 1013904223u;
    return (float)((s >> 8) & 0xffff) / 65536.0f - 0.5f;
  }
};

std::vector<float> make(int n, uint32_t seed, float scale) {
  Lcg g(seed);
  std::vector<float> v(n);
  for (auto& x : v) x = g.next() * scale;
  return v;
}

// Row-major host matrices.
std::vector<float> X, Y, W1_0, W2_0;   // (B,I) (B,O) (I,H) (H,O)

// ---- the reference: the whole batch in one process, in double ---------------

void reference(std::vector<double>& w1, std::vector<double>& w2, double& loss_first, double& loss_last) {
  w1.assign(W1_0.begin(), W1_0.end());
  w2.assign(W2_0.begin(), W2_0.end());
  for (int step = 0; step < STEPS; ++step) {
    std::vector<double> z(B * H), a(B * H), out(B * O), dout(B * O), dz(B * H);
    for (int b = 0; b < B; ++b)
      for (int h = 0; h < H; ++h) {
        double s = 0;
        for (int i = 0; i < I; ++i) s += (double)X[b * I + i] * w1[i * H + h];
        z[b * H + h] = s;
        a[b * H + h] = s > 0 ? s : 0;
      }
    double loss = 0;
    for (int b = 0; b < B; ++b)
      for (int o = 0; o < O; ++o) {
        double s = 0;
        for (int h = 0; h < H; ++h) s += a[b * H + h] * w2[h * O + o];
        out[b * O + o] = s;
        dout[b * O + o] = (s - Y[b * O + o]) / B;
        loss += 0.5 * (s - Y[b * O + o]) * (s - Y[b * O + o]) / B;
      }
    if (step == 0) loss_first = loss;
    loss_last = loss;
    for (int b = 0; b < B; ++b)
      for (int h = 0; h < H; ++h) {
        double s = 0;
        for (int o = 0; o < O; ++o) s += dout[b * O + o] * w2[h * O + o];
        dz[b * H + h] = z[b * H + h] > 0 ? s : 0;
      }
    for (int h = 0; h < H; ++h)
      for (int o = 0; o < O; ++o) {
        double g = 0;
        for (int b = 0; b < B; ++b) g += a[b * H + h] * dout[b * O + o];
        w2[h * O + o] -= LR * g;
      }
    for (int i = 0; i < I; ++i)
      for (int h = 0; h < H; ++h) {
        double g = 0;
        for (int b = 0; b < B; ++b) g += (double)X[b * I + i] * dz[b * H + h];
        w1[i * H + h] -= LR * g;
      }
  }
}

// ---- device pieces ----------------------------------------------------------

cublasHandle_t blas;
cudaStream_t stream;
ncclComm_t comm;

float* dev(int n) { float* p = nullptr; CK(cudaMalloc(&p, n * sizeof(float))); return p; }
float* to_dev(const std::vector<float>& v) {
  float* p = dev((int)v.size());
  CK(cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
  return p;
}
std::vector<float> to_host(const float* p, int n) {
  std::vector<float> v(n);
  CK(cudaMemcpy(v.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
  return v;
}

// C(m,n) = op(A) * op(B) for row-major matrices: a column-major GEMM of the
// transposes, with the operands swapped. lda, ldb, ldc are the row lengths.
void gemm(bool ta, bool tb, int m, int n, int k, const float* A, int lda, const float* Bm, int ldb, float* C, int ldc) {
  const float one = 1.0f, zero = 0.0f;
  BK(cublasSgemm(blas, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, &one, Bm, ldb, A, lda, &zero, C,
                 ldc));
}

__global__ void relu_k(const float* z, float* a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = z[i] > 0 ? z[i] : 0;
}
// dz = da where z > 0.
__global__ void relu_back_k(const float* z, const float* da, float* dz, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dz[i] = z[i] > 0 ? da[i] : 0;
}
// d = (out - y) / scale
__global__ void loss_grad_k(const float* out, const float* y, float* d, int n, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = (out[i] - y[i]) / scale;
}
__global__ void sgd_k(float* w, const float* g, int n, float lr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) w[i] -= lr * g[i];
}

#define LAUNCH(k, n, ...) k<<<((n) + 127) / 128, 128, 0, stream>>>(__VA_ARGS__)

void sync() { CK(cudaStreamSynchronize(stream)); }

// The largest difference from the reference, relative to the reference's size.
double worst(const std::vector<float>& got, const std::vector<double>& want, int want_off, int count, int stride_got = 0) {
  (void)stride_got;
  double top = 1e-6, diff = 0;
  for (int i = 0; i < count; ++i) top = std::fmax(top, std::fabs(want[want_off + i]));
  for (int i = 0; i < count; ++i) diff = std::fmax(diff, std::fabs((double)got[i] - want[want_off + i]));
  return diff / top;
}

constexpr double TOL = 2e-4;
int failures = 0;

void report(const char* mode, double err, const char* extra = "") {
  if (rank_ == 0 || err > TOL) {
    if (err > TOL) {
      std::printf("FAIL %s: rank %d is %g from a single process on the whole batch%s\n", mode, rank_, err, extra);
    } else {
      std::printf("ok %s (nranks %d, max relative difference %.2g)\n", mode, nranks_, err);
    }
    std::fflush(stdout);
  }
  failures += err > TOL;
}

// ---- data parallel ----------------------------------------------------------

void data_parallel(const std::vector<double>& r1, const std::vector<double>& r2) {
  const int bl = B / nranks_, b0 = rank_ * bl;
  float *x = to_dev(std::vector<float>(X.begin() + b0 * I, X.begin() + (b0 + bl) * I));
  float *y = to_dev(std::vector<float>(Y.begin() + b0 * O, Y.begin() + (b0 + bl) * O));
  float *w1 = to_dev(W1_0), *w2 = to_dev(W2_0);
  float *z = dev(bl * H), *a = dev(bl * H), *out = dev(bl * O), *dout = dev(bl * O), *da = dev(bl * H), *dz = dev(bl * H);
  float *g1 = dev(I * H), *g2 = dev(H * O);
  for (int step = 0; step < STEPS; ++step) {
    gemm(false, false, bl, H, I, x, I, w1, H, z, H);
    LAUNCH(relu_k, bl * H, z, a, bl * H);
    gemm(false, false, bl, O, H, a, H, w2, O, out, O);
    LAUNCH(loss_grad_k, bl * O, out, y, dout, bl * O, (float)B);   // the mean is over the whole batch
    gemm(true, false, H, O, bl, a, H, dout, O, g2, O);
    gemm(false, true, bl, H, O, dout, O, w2, O, da, H);
    LAUNCH(relu_back_k, bl * H, z, da, dz, bl * H);
    gemm(true, false, I, H, bl, x, I, dz, H, g1, H);
    NK(ncclGroupStart());
    NK(ncclAllReduce(g1, g1, I * H, ncclFloat, ncclSum, comm, stream));
    NK(ncclAllReduce(g2, g2, H * O, ncclFloat, ncclSum, comm, stream));
    NK(ncclGroupEnd());
    LAUNCH(sgd_k, I * H, w1, g1, I * H, (float)LR);
    LAUNCH(sgd_k, H * O, w2, g2, H * O, (float)LR);
  }
  sync();
  report("data parallel", std::fmax(worst(to_host(w1, I * H), r1, 0, I * H), worst(to_host(w2, H * O), r2, 0, H * O)));
  for (float* p : {x, y, w1, w2, z, a, out, dout, da, dz, g1, g2}) cudaFree(p);
}

// ---- tensor parallel --------------------------------------------------------

void tensor_parallel(const std::vector<double>& r1, const std::vector<double>& r2) {
  const int hs = H / nranks_, h0 = rank_ * hs;
  // This rank's columns of W1 and rows of W2.
  std::vector<float> w1s(I * hs), w2s(hs * O);
  for (int i = 0; i < I; ++i)
    for (int h = 0; h < hs; ++h) w1s[i * hs + h] = W1_0[i * H + h0 + h];
  for (int h = 0; h < hs; ++h)
    for (int o = 0; o < O; ++o) w2s[h * O + o] = W2_0[(h0 + h) * O + o];
  float *x = to_dev(X), *y = to_dev(Y), *w1 = to_dev(w1s), *w2 = to_dev(w2s);
  float *z = dev(B * hs), *a = dev(B * hs), *out = dev(B * O), *dout = dev(B * O), *da = dev(B * hs), *dz = dev(B * hs);
  float *g1 = dev(I * hs), *g2 = dev(hs * O);
  for (int step = 0; step < STEPS; ++step) {
    gemm(false, false, B, hs, I, x, I, w1, hs, z, hs);
    LAUNCH(relu_k, B * hs, z, a, B * hs);
    gemm(false, false, B, O, hs, a, hs, w2, O, out, O);                         // this rank's part of the output
    NK(ncclAllReduce(out, out, B * O, ncclFloat, ncclSum, comm, stream));      // the sum of the parts
    LAUNCH(loss_grad_k, B * O, out, y, dout, B * O, (float)B);
    gemm(true, false, hs, O, B, a, hs, dout, O, g2, O);
    gemm(false, true, B, hs, O, dout, O, w2, O, da, hs);
    LAUNCH(relu_back_k, B * hs, z, da, dz, B * hs);
    gemm(true, false, I, hs, B, x, I, dz, hs, g1, hs);
    LAUNCH(sgd_k, I * hs, w1, g1, I * hs, (float)LR);
    LAUNCH(sgd_k, hs * O, w2, g2, hs * O, (float)LR);
  }
  sync();
  const std::vector<float> h1 = to_host(w1, I * hs), h2 = to_host(w2, hs * O);
  double err = 0;
  for (int i = 0; i < I; ++i) {   // W1's columns h0.. against the reference's
    std::vector<float> row(h1.begin() + i * hs, h1.begin() + (i + 1) * hs);
    err = std::fmax(err, worst(row, r1, i * H + h0, hs));
  }
  err = std::fmax(err, worst(h2, r2, h0 * O, hs * O));
  report("tensor parallel", err);
  for (float* p : {x, y, w1, w2, z, a, out, dout, da, dz, g1, g2}) cudaFree(p);
}

// ---- pipeline parallel (two ranks) ------------------------------------------

void pipeline_parallel(const std::vector<double>& r1, const std::vector<double>& r2) {
  float *x = to_dev(X), *y = to_dev(Y);
  float *act = dev(B * H);   // a1: stage 0 makes it, stage 1 receives it
  float *dact = dev(B * H);  // its gradient: stage 1 makes it, stage 0 receives it
  double err = 0;
  if (rank_ == 0) {
    float *w1 = to_dev(W1_0), *z = dev(B * H), *dz = dev(B * H), *g1 = dev(I * H);
    for (int step = 0; step < STEPS; ++step) {
      gemm(false, false, B, H, I, x, I, w1, H, z, H);
      LAUNCH(relu_k, B * H, z, act, B * H);
      NK(ncclSend(act, B * H, ncclFloat, 1, comm, stream));
      NK(ncclRecv(dact, B * H, ncclFloat, 1, comm, stream));
      LAUNCH(relu_back_k, B * H, z, dact, dz, B * H);
      gemm(true, false, I, H, B, x, I, dz, H, g1, H);
      LAUNCH(sgd_k, I * H, w1, g1, I * H, (float)LR);
    }
    sync();
    err = worst(to_host(w1, I * H), r1, 0, I * H);
    for (float* p : {w1, z, dz, g1}) cudaFree(p);
  } else {
    float *w2 = to_dev(W2_0), *out = dev(B * O), *dout = dev(B * O), *g2 = dev(H * O);
    for (int step = 0; step < STEPS; ++step) {
      NK(ncclRecv(act, B * H, ncclFloat, 0, comm, stream));
      gemm(false, false, B, O, H, act, H, w2, O, out, O);
      LAUNCH(loss_grad_k, B * O, out, y, dout, B * O, (float)B);
      gemm(true, false, H, O, B, act, H, dout, O, g2, O);
      gemm(false, true, B, H, O, dout, O, w2, O, dact, H);
      NK(ncclSend(dact, B * H, ncclFloat, 0, comm, stream));
      LAUNCH(sgd_k, H * O, w2, g2, H * O, (float)LR);
    }
    sync();
    err = worst(to_host(w2, H * O), r2, 0, H * O);
    for (float* p : {w2, out, dout, g2}) cudaFree(p);
  }
  report("pipeline parallel", err);
  for (float* p : {x, y, act, dact}) cudaFree(p);
}

// ---- sharded parameters and gradients (ZeRO / FSDP) -------------------------

void sharded(const std::vector<double>& r1, const std::vector<double>& r2) {
  constexpr int S = I * H + H * O;
  const int ss = S / nranks_, bl = B / nranks_, b0 = rank_ * bl;
  std::vector<float> flat(S);
  for (int k = 0; k < I * H; ++k) flat[k] = W1_0[k];
  for (int k = 0; k < H * O; ++k) flat[I * H + k] = W2_0[k];
  float *shard = to_dev(std::vector<float>(flat.begin() + rank_ * ss, flat.begin() + (rank_ + 1) * ss));
  float *full = dev(S), *grads = dev(S), *gshard = dev(ss);
  float *x = to_dev(std::vector<float>(X.begin() + b0 * I, X.begin() + (b0 + bl) * I));
  float *y = to_dev(std::vector<float>(Y.begin() + b0 * O, Y.begin() + (b0 + bl) * O));
  float *z = dev(bl * H), *a = dev(bl * H), *out = dev(bl * O), *dout = dev(bl * O), *da = dev(bl * H), *dz = dev(bl * H);
  for (int step = 0; step < STEPS; ++step) {
    NK(ncclAllGather(shard, full, ss, ncclFloat, comm, stream));   // every rank's slice -> the whole vector
    float *w1 = full, *w2 = full + I * H;
    gemm(false, false, bl, H, I, x, I, w1, H, z, H);
    LAUNCH(relu_k, bl * H, z, a, bl * H);
    gemm(false, false, bl, O, H, a, H, w2, O, out, O);
    LAUNCH(loss_grad_k, bl * O, out, y, dout, bl * O, (float)B);
    gemm(true, false, H, O, bl, a, H, dout, O, grads + I * H, O);
    gemm(false, true, bl, H, O, dout, O, w2, O, da, H);
    LAUNCH(relu_back_k, bl * H, z, da, dz, bl * H);
    gemm(true, false, I, H, bl, x, I, dz, H, grads, H);
    NK(ncclReduceScatter(grads, gshard, ss, ncclFloat, ncclSum, comm, stream));   // summed, each rank keeps its slice
    LAUNCH(sgd_k, ss, shard, gshard, ss, (float)LR);
  }
  NK(ncclAllGather(shard, full, ss, ncclFloat, comm, stream));
  sync();
  const std::vector<float> h = to_host(full, S);
  const std::vector<float> h1(h.begin(), h.begin() + I * H), h2(h.begin() + I * H, h.end());
  report("sharded (all-gather + reduce-scatter)", std::fmax(worst(h1, r1, 0, I * H), worst(h2, r2, 0, H * O)));
  for (float* p : {shard, full, grads, gshard, x, y, z, a, out, dout, da, dz}) cudaFree(p);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) { std::fprintf(stderr, "usage: %s <rank> <nranks> <idfile>\n", argv[0]); return 2; }
  rank_ = std::atoi(argv[1]);
  nranks_ = std::atoi(argv[2]);
  const std::string idfile = argv[3];
  if (nranks_ < 1 || B % nranks_ || H % nranks_ || (I * H + H * O) % nranks_) {
    std::fprintf(stderr, "nranks %d must divide %d, %d and %d\n", nranks_, B, H, I * H + H * O);
    return 2;
  }
  X = make(B * I, 1, 2.0f);
  Y = make(B * O, 2, 2.0f);
  W1_0 = make(I * H, 3, 0.8f);
  W2_0 = make(H * O, 4, 0.8f);

  ncclUniqueId id;
  if (rank_ == 0) {
    NK(ncclGetUniqueId(&id));
    const std::string tmp = idfile + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return 2;
    std::fwrite(&id, sizeof(id), 1, f);
    std::fclose(f);
    std::rename(tmp.c_str(), idfile.c_str());   // atomic: readers never see a partial id
  } else {
    for (int i = 0; i < 60000; ++i) {
      FILE* f = std::fopen(idfile.c_str(), "rb");
      if (f) {
        const bool ok = std::fread(&id, sizeof(id), 1, f) == 1;
        std::fclose(f);
        if (ok) break;
      }
      usleep(1000);
    }
  }
  CK(cudaSetDevice(0));   // each process sees its own device
  CK(cudaStreamCreate(&stream));
  BK(cublasCreate(&blas));
  BK(cublasSetStream(blas, stream));
  NK(ncclCommInitRank(&comm, nranks_, id, rank_));

  std::vector<double> r1, r2;
  double first = 0, last = 0;
  reference(r1, r2, first, last);
  if (rank_ == 0) {
    if (!(last < first)) { std::printf("FAIL the reference's loss did not fall (%g -> %g)\n", first, last); ++failures; }
    else std::printf("ok the loss falls over %d steps (%.4g -> %.4g)\n", STEPS, first, last);
  }

  data_parallel(r1, r2);
  tensor_parallel(r1, r2);
  if (nranks_ == 2) pipeline_parallel(r1, r2);
  sharded(r1, r2);

  NK(ncclCommDestroy(comm));
  return failures ? 1 : 0;
}
