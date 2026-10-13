// gfx1250 (CDNA 5, MI455X) clusters (cluster1250.gfx1250, wave32). The kernel (cluster1250_kernel.cpp, built for a cluster of 4 x 2
// work-groups) is a code object embedded here and launched as a module; no ROCm 7.2 runtime consumes the cluster dimensions the code
// object carries, so the simulator takes them from it. The oracle is AMD's CDNA5 ISA (2.3 and 5.6): a work-group finds its place in its
// cluster in TTMP6, TTMP7, TTMP9 and IB_STS2, and the cluster barrier (s_barrier_signal -3 and s_barrier_wait -3) completes when every
// work-group of the cluster has signalled -- so each work-group reads, after it, the flags every other one wrote before it.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "cluster1250_kernel.h"

#define CHECK(x)                                                                        \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

int main() {
  hipModule_t module;
  hipFunction_t fn;
  CHECK(hipModuleLoadData(&module, cluster1250_kernel_hsaco));
  CHECK(hipModuleGetFunction(&fn, module, "k_cluster"));
  int failed = 0;
  for (const auto shape : {std::pair<int, int>{8, 4}, std::pair<int, int>{4, 2}, std::pair<int, int>{12, 6}}) {
    const int gx = shape.first, gy = shape.second, groups = gx * gy;
    unsigned *d_o = nullptr, *d_flags = nullptr;
    CHECK(hipMalloc(&d_o, groups * 8 * 4));
    CHECK(hipMalloc(&d_flags, groups * 4));
    CHECK(hipMemset(d_o, 0xFF, groups * 8 * 4));
    CHECK(hipMemset(d_flags, 0, groups * 4));
    void* args[] = {&d_o, &d_flags};
    CHECK(hipModuleLaunchKernel(fn, gx, gy, 1, 64, 1, 1, 0, nullptr, args, nullptr));
    CHECK(hipDeviceSynchronize());
    std::vector<unsigned> o(groups * 8);
    CHECK(hipMemcpy(o.data(), d_o, o.size() * 4, hipMemcpyDeviceToHost));
    int wrong = 0;
    for (int y = 0; y < gy; ++y)
      for (int x = 0; x < gx; ++x) {
        const int group = y * gx + x;
        const unsigned cx = x / 4, cy = y / 2, wx = x % 4, wy = y % 2;
        unsigned sum = 0;
        for (int yy = 0; yy < 2; ++yy)
          for (int xx = 0; xx < 4; ++xx) sum += (cy * 2 + yy) * gx + cx * 4 + xx + 1;
        const unsigned want[8] = {static_cast<unsigned>(x), static_cast<unsigned>(y), cx, cy, wx, wy, wy * 4 + wx, sum};
        for (int k = 0; k < 8; ++k) wrong += o[group * 8 + k] != want[k];
      }
    std::printf("a %d x %d grid of clusters of 4 x 2: %d of %d words wrong\n", gx, gy, wrong, groups * 8);
    failed += wrong != 0;
    CHECK(hipFree(d_o));
    CHECK(hipFree(d_flags));
  }
  // A grid that is not whole clusters is refused, as a launch of a kernel the hardware cannot place.
  {
    unsigned *d_o = nullptr, *d_flags = nullptr;
    CHECK(hipMalloc(&d_o, 6 * 2 * 8 * 4));
    CHECK(hipMalloc(&d_flags, 6 * 2 * 4));
    void* args[] = {&d_o, &d_flags};
    const hipError_t e = hipModuleLaunchKernel(fn, 6, 2, 1, 64, 1, 1, 0, nullptr, args, nullptr);
    const hipError_t s = e == hipSuccess ? hipDeviceSynchronize() : e;
    std::printf("a 6 x 2 grid (not whole clusters) is refused: %s\n", s != hipSuccess ? "yes" : "NO");
    failed += s == hipSuccess;
  }
  std::printf("cluster1250: %d failed\n", failed);
  return failed != 0;
}
