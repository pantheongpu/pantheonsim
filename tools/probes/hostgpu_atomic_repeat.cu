// Card run for turbo host-GPU mode (docs/perf-design/12-host-gpu-turbo.md): does the card agree with ITSELF on kernels built on atomics? Each kernel
// runs `runs` times from the same memory, natively (no simulator), and every run is compared with the first, bit for bit. A kind the card repeats is one
// whose only difference from the simulator can be numeric; a kind it does not repeat has no single right answer, and a compare of the card with the
// simulator says nothing about either (which is why float atomics with a returned value, compare-and-swap and exchange are behind VGPU_HOST_GPU_ATOMICS=1).
//
//   usage: hostgpu_atomic_repeat [runs=8] [log2_threads=22]
//   build: nvcc -std=c++17 -O2 -arch=sm_86 tools/probes/hostgpu_atomic_repeat.cu -o hostgpu_atomic_repeat
//
// Kinds (the SASS in brackets is what nvcc 12/13 emits for sm_86; check with VGPU_HOST_GPU_LOG=2 under the simulator if a card or toolkit differs):
//   imax   atomicMax on ints, result unused                  [RED.MAX: the exact set]                     control: must repeat
//   iadd   atomicAdd on ints, old value stored per thread    [ATOMG.ADD: turbo class atomic-int]        the sums repeat, the old values need not
//   fred   atomicAdd on floats, result unused                [RED.ADD.F32: turbo class float-red]       sums in arrival order
//   fatom  atomicAdd on floats, old value stored per thread  [ATOMG.ADD.F32: atomic-float, opt-in]
//   dadd   atomicAdd on doubles, result unused               [RED.ADD.F64 or a compare-and-swap loop]
//   cas    atomicCAS, first thread to arrive claims a slot   [ATOMG.CAS: atomic-swap, opt-in]
//   exch   atomicExch, last thread to arrive stays           [ATOMG.EXCH: atomic-swap, opt-in]
//
// Output, one line per kind and part of memory ("sums": the words the atomics target; "seen": what each thread got back):
//   RESULT kind=<k> part=<sums|seen> runs=N repeats=<runs equal to the first, of N-1> words=W max_words_different=D max_ulp=U
// max_ulp is the largest distance of two differing words read as binary32 (binary64 for dadd), UINT32_MAX when signs differ; it means something for
// fred, fatom and dadd sums only. Then "VERDICT kind=<k> repeats|varies". The exit status is 0 when the control repeats, 1 when it does not (a broken run).
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

constexpr int kBins = 256;

__device__ inline uint32_t mix(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return x;
}
// A float in [2^-8, 2^8) with a full mantissa: sums of these round differently in a different order.
__device__ inline float value_of(uint32_t i) {
  const uint32_t h = mix(i * 2654435761u + 12345u);
  return __uint_as_float(((119u + (h >> 28)) << 23) | (h & 0x7fffffu));
}

__global__ void k_imax(int* sums, uint32_t* seen, uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  atomicMax(&sums[i % kBins], (int)(mix(i) >> 1));
  seen[i] = 0;
}
__global__ void k_iadd(int* sums, uint32_t* seen, uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  seen[i] = (uint32_t)atomicAdd(&sums[i % kBins], 1);
}
__global__ void k_fred(float* sums, uint32_t* seen, uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  atomicAdd(&sums[i % kBins], value_of(i));
  seen[i] = 0;
}
__global__ void k_fatom(float* sums, uint32_t* seen, uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  seen[i] = __float_as_uint(atomicAdd(&sums[i % kBins], value_of(i)));
}
__global__ void k_dadd(double* sums, uint32_t* seen, uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  atomicAdd(&sums[i % kBins], (double)value_of(i) * 1.0000001192092896);
  seen[i] = 0;
}
__global__ void k_cas(int* sums, uint32_t* seen, uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  seen[i] = (uint32_t)atomicCAS(&sums[i % kBins], 0, (int)(i + 1));
}
__global__ void k_exch(int* sums, uint32_t* seen, uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  seen[i] = (uint32_t)atomicExch(&sums[i % kBins], (int)(i + 1));
}

static uint32_t ulp32(uint32_t a, uint32_t b) {
  if ((a ^ b) >> 31) return UINT32_MAX;
  return a > b ? a - b : b - a;
}
static uint32_t ulp64(uint64_t a, uint64_t b) {
  if ((a ^ b) >> 63) return UINT32_MAX;
  const uint64_t d = a > b ? a - b : b - a;
  return d > UINT32_MAX ? UINT32_MAX : (uint32_t)d;
}

struct Part {
  unsigned repeats = 0;
  uint64_t max_words = 0;
  uint32_t max_ulp = 0;
};
static void compare(const std::vector<uint8_t>& first, const std::vector<uint8_t>& now, bool wide, Part& p) {
  uint64_t bad = 0;
  const size_t step = wide ? 8 : 4;
  for (size_t at = 0; at + step <= first.size(); at += step) {
    if (!std::memcmp(&first[at], &now[at], step)) continue;
    ++bad;
    uint32_t u;
    if (wide) { uint64_t a, b; std::memcpy(&a, &first[at], 8); std::memcpy(&b, &now[at], 8); u = ulp64(a, b); }
    else { uint32_t a, b; std::memcpy(&a, &first[at], 4); std::memcpy(&b, &now[at], 4); u = ulp32(a, b); }
    if (u > p.max_ulp) p.max_ulp = u;
  }
  if (!bad) ++p.repeats;
  if (bad > p.max_words) p.max_words = bad;
}

int main(int argc, char** argv) {
  const int runs = argc > 1 ? std::atoi(argv[1]) : 8;
  const int lg = argc > 2 ? std::atoi(argv[2]) : 22;
  if (runs < 2 || lg < 8 || lg > 28) { std::fprintf(stderr, "usage: %s [runs>=2] [log2_threads 8..28]\n", argv[0]); return 2; }
  const uint32_t n = 1u << lg;
  void* sums = nullptr;
  uint32_t* seen = nullptr;
  CK(cudaMalloc(&sums, kBins * 8));
  CK(cudaMalloc(&seen, (size_t)n * 4));
  const char* kinds[] = {"imax", "iadd", "fred", "fatom", "dadd", "cas", "exch"};
  bool control_ok = true;
  for (int k = 0; k < 7; ++k) {
    const bool wide = k == 4;
    std::vector<uint8_t> first_sums, first_seen, now_sums(kBins * 8), now_seen((size_t)n * 4);
    Part ps, pn;
    for (int r = 0; r < runs; ++r) {
      CK(cudaMemset(sums, 0, kBins * 8));
      CK(cudaMemset(seen, 0, (size_t)n * 4));
      const unsigned grid = (n + 255) / 256;
      switch (k) {
        case 0: k_imax<<<grid, 256>>>((int*)sums, seen, n); break;
        case 1: k_iadd<<<grid, 256>>>((int*)sums, seen, n); break;
        case 2: k_fred<<<grid, 256>>>((float*)sums, seen, n); break;
        case 3: k_fatom<<<grid, 256>>>((float*)sums, seen, n); break;
        case 4: k_dadd<<<grid, 256>>>((double*)sums, seen, n); break;
        case 5: k_cas<<<grid, 256>>>((int*)sums, seen, n); break;
        default: k_exch<<<grid, 256>>>((int*)sums, seen, n); break;
      }
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(now_sums.data(), sums, now_sums.size(), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(now_seen.data(), seen, now_seen.size(), cudaMemcpyDeviceToHost));
      if (r == 0) { first_sums = now_sums; first_seen = now_seen; continue; }
      compare(first_sums, now_sums, wide, ps);
      compare(first_seen, now_seen, false, pn);
    }
    std::printf("RESULT kind=%s part=sums runs=%d repeats=%u words=%d max_words_different=%llu max_ulp=%u\n", kinds[k], runs, ps.repeats, wide ? kBins : kBins * 2,
                (unsigned long long)ps.max_words, ps.max_ulp);
    std::printf("RESULT kind=%s part=seen runs=%d repeats=%u words=%u max_words_different=%llu max_ulp=%u\n", kinds[k], runs, pn.repeats, n, (unsigned long long)pn.max_words,
                pn.max_ulp);
    const bool repeats = ps.repeats == (unsigned)(runs - 1) && pn.repeats == (unsigned)(runs - 1);
    std::printf("VERDICT kind=%s %s\n", kinds[k], repeats ? "repeats" : "varies");
    std::fflush(stdout);
    if (k == 0 && !repeats) control_ok = false;
  }
  CK(cudaFree(sums));
  CK(cudaFree(seen));
  if (!control_ok) std::printf("FAIL: the control (integer max, no value returned) did not repeat: the run itself is broken\n");
  return control_ok ? 0 : 1;
}
