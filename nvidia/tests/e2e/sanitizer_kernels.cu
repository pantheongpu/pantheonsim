// Small kernels with a known defect, one per case, for the checks a card gives only under compute-sanitizer. The question each asks
// is what compute-sanitizer would report for the same kernel, in its terms (memcheck: an invalid or misaligned global or shared access;
// racecheck: a hazard between warps on shared memory), and whether the simulator reports the same thing: the same kind of error, from
// the same instruction, with the access's direction, size and address, and the same error code from the runtime afterwards (700
// cudaErrorIllegalAddress for an invalid access, 716 cudaErrorMisalignedAddress for a misaligned one). Cases that must run clean are
// here too, so that a check that reports everything does not pass.
//
// A fault kills the context, so each case is a process of its own: run_sanitizer_kernels.sh runs this once per case and per engine
// (SASS, which nvcc's binary has, and PTX, VGPU_SASS=0) and compares what it prints with the expectations in its table.
//
// Prints "ERR <code>" for what cudaDeviceSynchronize returned and "OUT <n>" for the one value a clean kernel computes.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

// ---- memcheck: global memory ----

__global__ void k_global_write(int* p, int last) {   // thread `last` is one element past a buffer of `last` ints
  const int t = threadIdx.x;
  if (t <= last) p[t] = t;
}
__global__ void k_global_read_byte(const volatile char* p, int off, int* out) { out[0] = p[off]; }
__global__ void k_global_read8(const volatile long long* p, long long* out) { out[0] = p[1]; }   // (all 8 bytes used: ptxas would narrow a load whose upper half nobody reads)
__global__ void k_global_store(volatile int* p, int v) { p[0] = v; }
__global__ void k_misaligned_load(const volatile char* p, int off, int* out) { out[0] = *reinterpret_cast<const volatile int*>(p + off); }
__global__ void k_misaligned_store8(volatile char* p, int off) { *reinterpret_cast<volatile long long*>(p + off) = 1; }
__global__ void k_misaligned_atomic(char* p, int off) { atomicAdd(reinterpret_cast<int*>(p + off), 1); }

// ---- memcheck: shared memory ----

__global__ void k_shared_write(int* out, int idx) {
  __shared__ int s[32];
  s[threadIdx.x] = 1;
  __syncthreads();
  volatile int* v = s;
  v[idx] = 7;
  out[0] = s[0];
}
__global__ void k_shared_read(int* out, int idx) {
  __shared__ int s[32];
  s[threadIdx.x] = 1;
  __syncthreads();
  volatile int* v = s;
  out[0] = v[idx];
}
__global__ void k_shared_dynamic(int* out, int idx) {
  extern __shared__ int d[];
  d[threadIdx.x] = 1;
  __syncthreads();
  volatile int* v = d;
  v[idx] = 7;
  out[0] = d[0];
}
__global__ void k_shared_misaligned(int* out, int off) {
  __shared__ int s[32];
  s[threadIdx.x] = 1;
  __syncthreads();
  out[0] = *reinterpret_cast<volatile int*>(reinterpret_cast<volatile char*>(s) + off);
}

// ---- racecheck: shared memory between warps ----

__global__ void k_race_raw(int* out) {   // warp 0 writes the word, warp 1 reads it, nothing orders them
  __shared__ volatile int s[1];
  s[0] = 0;
  __syncthreads();
  if (threadIdx.x == 0) s[0] = 5;
  if (threadIdx.x == 32) out[0] = s[0];
}
__global__ void k_race_waw(int* out) {
  __shared__ volatile int s[1];
  if (threadIdx.x == 0) s[0] = 1;
  if (threadIdx.x == 32) s[0] = 2;
  __syncthreads();
  out[0] = s[0];
}
__global__ void k_race_reduce_missing_sync(const int* in, int* out) {   // the textbook mistake: no barrier between the steps
  __shared__ volatile int s[128];
  const int t = threadIdx.x;
  s[t] = in[t];
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (t < stride) s[t] += s[t + stride];   // __syncthreads() belongs here
  }
  if (t == 0) out[0] = s[0];
}
__global__ void k_ordered(int* out) {
  __shared__ int s[1];
  if (threadIdx.x == 0) s[0] = 5;
  __syncthreads();
  if (threadIdx.x == 32) out[0] = s[0];
}
__global__ void k_reduce_ok(const int* in, int* out) {
  __shared__ int s[128];
  const int t = threadIdx.x;
  s[t] = in[t];
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (t < stride) s[t] += s[t + stride];
    __syncthreads();
  }
  if (t == 0) out[0] = s[0];
}
__global__ void k_atomics_ok(int* out) {   // atomics from every warp on one word are ordered by the hardware
  __shared__ int s[1];
  if (threadIdx.x == 0) s[0] = 0;
  __syncthreads();
  atomicAdd(&s[0], 1);
  __syncthreads();
  if (threadIdx.x == 0) out[0] = s[0];
}
__global__ void k_read_read_ok(int* out) {   // every warp reads the same word: not a conflict
  __shared__ int s[1];
  if (threadIdx.x == 0) s[0] = 9;
  __syncthreads();
  const int v = s[0];
  if (threadIdx.x == 33) out[0] = v;
}

static int g_err = 0;
static void finish(const int* d_out) {
  g_err = static_cast<int>(cudaDeviceSynchronize());
  std::printf("ERR %d\n", g_err);
  if (g_err == 0 && d_out) {
    int h = -1;
    cudaMemcpy(&h, d_out, sizeof h, cudaMemcpyDeviceToHost);
    std::printf("OUT %d\n", h);
  }
}

int main(int argc, char** argv) {
  const char* c = argc > 1 ? argv[1] : "";
  auto is = [&](const char* n) { return std::strcmp(c, n) == 0; };
  int *buf = nullptr, *out = nullptr, *in = nullptr;
  cudaMalloc(&out, 64);
  cudaMemset(out, 0, 64);
  cudaMalloc(&in, 128 * sizeof(int));
  {
    int h[128];
    for (int i = 0; i < 128; ++i) h[i] = i + 1;
    cudaMemcpy(in, h, sizeof h, cudaMemcpyHostToDevice);
  }
  if (is("global-write-past-end")) {
    cudaMalloc(&buf, 32 * sizeof(int));
    k_global_write<<<1, 33>>>(buf, 32);
    finish(out);
  } else if (is("global-write-in-bounds")) {
    cudaMalloc(&buf, 32 * sizeof(int));
    k_global_write<<<1, 32>>>(buf, 31);
    finish(out);
  } else if (is("global-read-one-byte-past-end")) {
    cudaMalloc(&buf, 100);   // the allocator rounds up; a read of byte 100 is still outside what was asked for
    cudaMemset(buf, 0, 100);
    k_global_read_byte<<<1, 1>>>(reinterpret_cast<char*>(buf), 100, out);
    finish(out);
  } else if (is("global-read-last-byte")) {
    cudaMalloc(&buf, 100);
    cudaMemset(buf, 0, 100);
    k_global_read_byte<<<1, 1>>>(reinterpret_cast<char*>(buf), 99, out);
    finish(out);
  } else if (is("global-read-straddling-the-end")) {
    cudaMalloc(&buf, 12);   // an 8-byte read at byte 8 covers bytes 8..15 of 12
    cudaMemset(buf, 0, 12);
    k_global_read8<<<1, 1>>>(reinterpret_cast<long long*>(buf), reinterpret_cast<long long*>(out));
    finish(out);
  } else if (is("global-use-after-free")) {
    cudaMalloc(&buf, 64);
    cudaFree(buf);
    k_global_store<<<1, 1>>>(buf, 1);
    finish(out);
  } else if (is("global-null")) {
    k_global_store<<<1, 1>>>(nullptr, 1);
    finish(out);
  } else if (is("global-unmapped")) {
    k_global_store<<<1, 1>>>(reinterpret_cast<int*>(0x10000000), 1);
    finish(out);
  } else if (is("global-misaligned-load")) {
    cudaMalloc(&buf, 64);
    cudaMemset(buf, 0, 64);
    k_misaligned_load<<<1, 1>>>(reinterpret_cast<char*>(buf), 1, out);
    finish(out);
  } else if (is("global-misaligned-store8")) {
    cudaMalloc(&buf, 64);
    k_misaligned_store8<<<1, 1>>>(reinterpret_cast<char*>(buf), 4);
    finish(out);
  } else if (is("global-misaligned-atomic")) {
    cudaMalloc(&buf, 64);
    k_misaligned_atomic<<<1, 1>>>(reinterpret_cast<char*>(buf), 2);
    finish(out);
  } else if (is("shared-write-far-past-end")) {
    k_shared_write<<<1, 32>>>(out, 100000);
    finish(out);
  } else if (is("shared-read-far-past-end")) {
    k_shared_read<<<1, 32>>>(out, 100000);
    finish(out);
  } else if (is("shared-write-one-past-end")) {
    k_shared_write<<<1, 32>>>(out, 32);
    finish(out);
  } else if (is("shared-write-in-bounds")) {
    k_shared_write<<<1, 32>>>(out, 31);
    finish(out);
  } else if (is("shared-dynamic-far-past-end")) {
    k_shared_dynamic<<<1, 32, 32 * sizeof(int)>>>(out, 100000);
    finish(out);
  } else if (is("shared-dynamic-one-past-end")) {
    k_shared_dynamic<<<1, 32, 32 * sizeof(int)>>>(out, 32);
    finish(out);
  } else if (is("shared-misaligned")) {
    k_shared_misaligned<<<1, 32>>>(out, 2);
    finish(out);
  } else if (is("race-write-then-read")) {
    k_race_raw<<<1, 64>>>(out);
    finish(out);
  } else if (is("race-write-write")) {
    k_race_waw<<<1, 64>>>(out);
    finish(out);
  } else if (is("race-reduction-missing-barrier")) {
    k_race_reduce_missing_sync<<<1, 128>>>(in, out);
    finish(out);
  } else if (is("ordered-by-barrier")) {
    k_ordered<<<1, 64>>>(out);
    finish(out);
  } else if (is("reduction-with-barriers")) {
    k_reduce_ok<<<1, 128>>>(in, out);
    finish(out);
  } else if (is("atomics-from-every-warp")) {
    k_atomics_ok<<<1, 128>>>(out);
    finish(out);
  } else if (is("every-warp-reads-one-word")) {
    k_read_read_ok<<<1, 64>>>(out);
    finish(out);
  } else {
    std::printf("unknown case %s\n", c);
    return 2;
  }
  return 0;
}
