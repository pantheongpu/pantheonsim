// A cooperative launch, built by hipcc: every work-group of the grid waits at
// a grid barrier (cooperative_groups' this_grid().sync()) until all have
// published a value, then reads all of them -- which is only right if no
// group read before the last one wrote. It passes the barrier twice, so the
// barrier's count has to come round again. Then a grid one work-group larger
// than the device holds at once is refused, as HIP refuses it.
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <cstdio>

namespace cg = cooperative_groups;

#define CHECK(x)                                                                    \
  do {                                                                              \
    hipError_t e_ = (x);                                                            \
    if (e_ != hipSuccess) {                                                         \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                      \
      return 1;                                                                     \
    }                                                                               \
  } while (0)

__global__ void two_phases(int* slots, int* sums) {
  cg::grid_group grid = cg::this_grid();
  const int g = blockIdx.x;
  if (threadIdx.x == 0) slots[g] = g + 1;
  grid.sync();
  if (threadIdx.x == 0) {
    int s = 0;
    for (unsigned i = 0; i < gridDim.x; ++i) s += slots[i];
    sums[g] = s;
  }
  grid.sync();
  if (threadIdx.x == 0) slots[g] = 0;   // after every group has summed
}

int main() {
  int coop = 0;
  CHECK(hipDeviceGetAttribute(&coop, hipDeviceAttributeCooperativeLaunch, 0));
  int per_cu = 0, cus = 0;
  CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, two_phases, 128, 0));
  CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, 0));
  std::printf("cooperative launch %d, %d work-groups of 128 a compute unit\n", coop, per_cu);

  const int groups = 16;
  int *slots = nullptr, *sums = nullptr;
  CHECK(hipMalloc(&slots, groups * sizeof(int)));
  CHECK(hipMalloc(&sums, groups * sizeof(int)));
  CHECK(hipMemset(slots, 0, groups * sizeof(int)));
  void* args[] = {&slots, &sums};
  CHECK(hipLaunchCooperativeKernel(reinterpret_cast<const void*>(two_phases), dim3(groups), dim3(128), args, 0, 0));
  CHECK(hipDeviceSynchronize());
  int got[groups] = {};
  CHECK(hipMemcpy(got, sums, sizeof got, hipMemcpyDeviceToHost));
  int right = 0;
  for (int g = 0; g < groups; ++g) right += got[g] == groups * (groups + 1) / 2;
  std::printf("after the grid barrier, %d of %d groups saw every group's value\n", right, groups);

  // One more work-group than the device holds at once.
  const hipError_t e = hipLaunchCooperativeKernel(reinterpret_cast<const void*>(two_phases),
                                                  dim3(per_cu * cus + 1), dim3(128), args, 0, 0);
  std::printf("a grid too large to be resident at once: %s\n", hipGetErrorName(e));
  return 0;
}
