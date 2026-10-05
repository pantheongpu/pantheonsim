// The first device runtime (CDP1), still in CUDA 12 and 13 for parts before Hopper and built with
// -DCUDA_FORCE_CDP1_IF_SUPPORTED: its calls are the unprefixed cuda* ones, and cudaDeviceSynchronize
// from a kernel is back. It waits for the grids launched by the threads of the calling block (RTX 3060,
// CUDA 13.0 and 12.0: a thread saw the child another thread of its block launched, after a
// __syncthreads, and a block did not wait for another block's), and a module that calls it does not load
// on sm_90 and later.
//
// Built -rdc=true with cudadevrt (run_sass_archs.sh, sm_75 to sm_89); every result is checked exactly
// and the program prints PASS on its last line.
#include <cstdio>

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                           \
      std::printf("FAIL: %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);   \
      return false;                                                                    \
    }                                                                                  \
  } while (0)

__global__ void set_value(int* p, int v) {
  for (int i = 0; i < 4; ++i) __nanosleep(250000);   // long enough that a missing wait shows
  atomicExch(p, v);
}
__global__ void read_into(const int* p, int* out) { *out = atomicAdd(const_cast<int*>(p), 0); }

// A thread waits for its own child.
__global__ void own_child(int* flag, int* out) {
  set_value<<<1, 1>>>(flag, 42);
  out[0] = cudaDeviceSynchronize();
  out[1] = atomicAdd(flag, 0);
}
// A thread waits for a child another thread of its block launched, once that launch has happened.
__global__ void block_child(int* flag, int* out) {
  if (threadIdx.x == 1) set_value<<<1, 1>>>(flag, 7);
  __syncthreads();
  if (threadIdx.x == 0) {
    out[2] = cudaDeviceSynchronize();
    out[3] = atomicAdd(flag, 0);
  }
}
// Nothing launched: nothing to wait for.
__global__ void nothing(int* out) { out[4] = cudaDeviceSynchronize(); }
// A grid's children wait for theirs too, and the parent waits for the child that waited.
__global__ void inner(int* flag, int* out) {
  set_value<<<1, 1>>>(flag, 9);
  cudaDeviceSynchronize();
  out[5] = atomicAdd(flag, 0);
}
__global__ void nested(int* flag, int* out) {
  inner<<<1, 1>>>(flag, out);
  cudaDeviceSynchronize();
  out[6] = atomicAdd(flag, 0);
}
// The calls CDP1 names without a prefix: memory from the device heap, copies, streams, queries.
__global__ void unprefixed(int* out) {
  int* p = nullptr;
  out[7] = cudaMalloc(reinterpret_cast<void**>(&p), 64);
  p[0] = 5;
  p[1] = 6;
  cudaStream_t s;
  out[8] = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  out[9] = cudaMemcpyAsync(p + 2, p, 8, cudaMemcpyDeviceToDevice, s);
  out[10] = cudaStreamDestroy(s);
  out[11] = cudaDeviceSynchronize();
  out[12] = p[2] * 10 + p[3];
  out[13] = cudaFree(p);
  int dev = -1;
  out[14] = cudaGetDevice(&dev);
  out[15] = dev;
  out[16] = cudaGetLastError();
}

bool run() {
  int *flag, *out;
  CK(cudaMalloc(&flag, sizeof(int)));
  CK(cudaMalloc(&out, 32 * sizeof(int)));
  CK(cudaMemset(out, 0xff, 32 * sizeof(int)));
  int h[32];
  bool ok = true;
  const auto reset = [&] { return cudaMemset(flag, 0, sizeof(int)) == cudaSuccess; };

  reset();
  own_child<<<1, 1>>>(flag, out);
  CK(cudaDeviceSynchronize());
  reset();
  block_child<<<1, 2>>>(flag, out);
  CK(cudaDeviceSynchronize());
  nothing<<<1, 1>>>(out);
  CK(cudaDeviceSynchronize());
  reset();
  nested<<<1, 1>>>(flag, out);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, out, sizeof h, cudaMemcpyDeviceToHost));
  bool good = h[0] == 0 && h[1] == 42 && h[2] == 0 && h[3] == 7 && h[4] == 0 && h[5] == 9 && h[6] == 9;
  std::printf("cudaDeviceSynchronize waits for the block's launches: %d (call %d), %d (call %d), call %d, %d and %d%s\n",
              h[1], h[0], h[3], h[2], h[4], h[5], h[6], good ? "" : " (want 42, 7, 9 and 9)");
  ok = ok && good;

  unprefixed<<<1, 1>>>(out);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, out, sizeof h, cudaMemcpyDeviceToHost));
  int dev = -1;
  CK(cudaGetDevice(&dev));
  good = h[7] == 0 && h[8] == 0 && h[9] == 0 && h[10] == 0 && h[11] == 0 && h[12] == 56 && h[13] == 0 && h[14] == 0 &&
         h[15] == dev && h[16] == 0;
  std::printf("the unprefixed device runtime calls: %s\n", good ? "as expected" : "WRONG");
  ok = ok && good;
  CK(cudaFree(flag));
  CK(cudaFree(out));
  return ok;
}

int main() {
  const bool ok = run();
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
