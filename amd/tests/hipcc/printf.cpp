// Device-side printf, built by hipcc: each call goes to the host through the
// hostcall buffer, one packet or several, and comes out on stdout. Three lanes
// print different values -- one of them a string long enough to need more
// than one packet -- and each keeps what printf returned, which the host
// compares with what its own printf would have printed. The kernel also reads a
// __constant__ table, which lives in the program's code object, reached the
// way the format strings are: relative to the code itself.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>

#define CHECK(x)                                                                    \
  do {                                                                              \
    hipError_t e_ = (x);                                                            \
    if (e_ != hipSuccess) {                                                         \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                      \
      return 1;                                                                     \
    }                                                                               \
  } while (0)

#define FORMAT "lane %d of %u: %s, %x, %5.2f, %e, %c, %lld, 100%%, prime %d\n"
#define LONG_TEXT "a string long enough that it takes more than one packet to carry it"

__constant__ int kPrimes[3] = {7, 11, 13};

__global__ void say(int* returned) {
  const int t = threadIdx.x;
  if (t < 3) {
    returned[t] = printf(FORMAT, t, blockDim.x, t == 1 ? LONG_TEXT : "short", 0xbeef + t, 1.5f * t, 2.5e-3,
                         'a' + t, 1ll << 40, kPrimes[t]);
  }
}

int main() {
  int* d = nullptr;
  CHECK(hipMalloc(&d, 3 * sizeof(int)));
  say<<<1, 64>>>(d);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  int returned[3] = {};
  CHECK(hipMemcpy(returned, d, sizeof returned, hipMemcpyDeviceToHost));
  int right = 0;
  for (int t = 0; t < 3; ++t) {
    char want[256];
    const int primes[3] = {7, 11, 13};
    const int n = std::snprintf(want, sizeof want, FORMAT, t, 64u, t == 1 ? LONG_TEXT : "short", 0xbeef + t,
                                1.5 * t, 2.5e-3, 'a' + t, 1ll << 40, primes[t]);
    right += returned[t] == n;
  }
  std::printf("printf returned what it printed for %d of 3 lanes\n", right);
  CHECK(hipFree(d));
  return 0;
}
