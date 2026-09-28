// The grid sizes HIP reports for a simulated AMD GPU (VGPU_GPU): the
// profile's, capped at INT_MAX as AMD's runtime caps them, since HIP's are
// ints. MI325X's profile, read from a card's HSA agent, holds the agent's
// 4294967295; passed through as an int it was -1, and PyTorch refused every
// grid sized against it. ctest runs this once for each AMD profile.
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>

#include "vgpu/hip_abi.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

extern "C" {
int hipGetDevicePropertiesR0600(vgpu::amd::abi::DevicePropR0600*, int);
int hipDeviceGetAttribute(int*, int, int);
}

VTEST(grid_sizes_are_the_profiles_as_ints) {
  const char* gpu = std::getenv("VGPU_GPU");
  VCHECK(gpu != nullptr);
  if (!gpu) return;
  const vgpu::DeviceProfile p = vgpu::load_gpu(gpu);
  vgpu::amd::abi::DevicePropR0600 props{};
  VCHECK_EQ(hipGetDevicePropertiesR0600(&props, 0), 0);
  using A = vgpu::amd::abi::DeviceAttribute;
  const A attrs[3] = {A::kMaxGridDimX, A::kMaxGridDimY, A::kMaxGridDimZ};
  for (int i = 0; i < 3; ++i) {
    const int64_t want = std::min<int64_t>(static_cast<int64_t>(p.limits.max_grid_dim[i]), INT_MAX);
    VCHECK(props.maxGridSize[i] > 0);
    VCHECK_EQ(int64_t{props.maxGridSize[i]}, want);
    int value = 0;
    VCHECK_EQ(hipDeviceGetAttribute(&value, static_cast<int>(attrs[i]), 0), 0);
    VCHECK_EQ(value, props.maxGridSize[i]);
  }
}

VTEST_MAIN
