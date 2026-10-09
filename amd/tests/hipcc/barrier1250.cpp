// gfx1250 (CDNA 5, MI455X) named barriers, each against the host (barrier1250.gfx1250, wave32). The oracle is AMD's CDNA5
// ISA (section 5.6): a named barrier (1 to 16) completes when as many signals have arrived as its member count; a wave
// joins one to hear of its completion, and waits on it. Four waves, in pairs on two barriers: the first wave of a pair
// signals, then waits, and what it reads after the wait is what the second wave wrote before it signalled -- which it
// reads only if the wait held it until then.
#include <hip/hip_runtime.h>

// The barrier instructions take their number (and member count) from M0, which the compiler would warn about.
#pragma clang diagnostic ignored "-Winline-asm"

#include <cstdint>
#include <cstdio>
#include <vector>

#define CHECK(x)                                                                        \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

__global__ void k_pairs(int* out) {
  __shared__ int lds[4];
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 32), lane = threadIdx.x % 32;
  uint32_t state = 0, first = 0;
  lds[wave] = 0;
  if (wave == 0) {
    // Barrier 1 wants two signals; set up before any wave uses it, and look at it.
    asm volatile("s_mov_b32 m0, 0x20001\n\ts_barrier_init m0\n\ts_get_barrier_state %0, 1" : "=s"(state) : : "m0");
  }
  __syncthreads();
  if (wave == 0) {
    lds[0] = 111;
    asm volatile("s_barrier_join 1\n\ts_barrier_signal_isfirst 1\n\ts_cselect_b32 %0, 1, 0\n\ts_barrier_wait 1" : "=s"(first));
    if (lane == 0) out[0] = lds[1], out[2] = first, out[7] = state;
  } else if (wave == 1) {
    lds[1] = 222;
    asm volatile("s_barrier_join 1\n\ts_barrier_signal_isfirst 1\n\ts_cselect_b32 %0, 1, 0\n\ts_barrier_wait 1" : "=s"(first));
    if (lane == 0) out[1] = lds[0], out[3] = first;
  } else if (wave == 2) {
    lds[2] = 333;
    // Barrier 2's member count comes with the signal, in M0.
    asm volatile("s_mov_b32 m0, 0x20002\n\ts_barrier_join 2\n\ts_barrier_signal m0\n\ts_barrier_wait 2" : : : "m0");
    if (lane == 0) out[4] = lds[3];
  } else {
    lds[3] = 444;
    asm volatile("s_mov_b32 m0, 0x20002\n\ts_barrier_join 2\n\ts_barrier_signal m0\n\ts_barrier_wait 2" : : : "m0");
    if (lane == 0) out[5] = lds[2];
  }
  __syncthreads();
  if (threadIdx.x == 0) out[6] = lds[0] + lds[1] + lds[2] + lds[3];
}

// A wave waits on a barrier no one will ever signal.
__global__ void k_stuck(int* out) {
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 32);
  if (wave == 0) {
    asm volatile("s_mov_b32 m0, 0x30003\n\ts_barrier_init m0\n\ts_barrier_join 3\n\ts_barrier_signal 3\n\ts_barrier_wait 3" : : : "m0");
  }
  if (threadIdx.x == 0) out[0] = 1;
}

int main() {
  int failed = 0;
  int* d = nullptr;
  CHECK(hipMalloc(&d, 8 * sizeof(int)));
  CHECK(hipMemset(d, 0, 8 * sizeof(int)));
  k_pairs<<<1, 128>>>(d);
  CHECK(hipDeviceSynchronize());
  int out[8];
  CHECK(hipMemcpy(out, d, sizeof out, hipMemcpyDeviceToHost));
  int wrong = 0;
  wrong += out[0] != 222;   // wave 0 read what wave 1 wrote before signalling
  wrong += out[1] != 111;
  wrong += out[4] != 444;
  wrong += out[5] != 333;
  wrong += out[2] != 1 || out[3] != 0;   // wave 0 signalled first
  wrong += out[6] != 111 + 222 + 333 + 444;
  wrong += ((out[7] >> 4) & 0x7F) != 2 || ((out[7] >> 16) & 0x7F) != 0 || (out[7] & 1) != 1;   // member count 2, none signalled, valid
  std::printf("named barriers: %d of 7 wrong\n", wrong);
  failed += wrong != 0;
  // A wait nothing can end is reported, not waited on for ever.
  k_stuck<<<1, 64>>>(d);
  const hipError_t e = hipDeviceSynchronize();
  std::printf("a wait on a barrier no wave can complete is an error: %s\n", e != hipSuccess ? "yes" : "no");
  failed += e == hipSuccess;
  std::printf("barrier1250: 2 checks, %d failed\n", failed);
  return failed != 0;
}
