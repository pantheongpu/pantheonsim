// The structures a hipcc-built program hands the runtime, held to the layout
// the HIP headers give them. A program reads a field at the offset its own
// header says, so a field one int out of place here is a wrong answer there.
//
// The numbers below were read from ROCm 7.1's hip_runtime_api.h (and are the
// same in 6.4). amd/tests/e2e/run_hip_abi.sh compares every field against the
// header itself wherever ROCm is installed; this keeps the ones the pantheon
// workloads read pinned everywhere else, including CI.
#include <cstddef>

#include "vgpu/hip_abi.hpp"
#include "vtest.hpp"

using vgpu::amd::abi::DevicePropR0600;

static_assert(sizeof(vgpu::amd::abi::Dim3) == 12, "dim3 is three unsigned ints");
static_assert(sizeof(vgpu::amd::abi::FatbinWrapper) == 24, "the wrapper hipcc emits is 24 bytes");
static_assert(sizeof(vgpu::amd::abi::DeviceArch) == 4, "the architecture flags fit one word");
static_assert(sizeof(DevicePropR0600) == 1472, "hipDeviceProp_tR0600 is 1472 bytes");
static_assert(offsetof(DevicePropR0600, totalGlobalMem) == 288, "");
static_assert(offsetof(DevicePropR0600, warpSize) == 308, "");
static_assert(offsetof(DevicePropR0600, maxGridSize) == 336, "");
static_assert(offsetof(DevicePropR0600, multiProcessorCount) == 388, "");
static_assert(offsetof(DevicePropR0600, l2CacheSize) == 616, "");
static_assert(offsetof(DevicePropR0600, maxThreadsPerMultiProcessor) == 624, "");
static_assert(offsetof(DevicePropR0600, gcnArchName) == 1160, "");

VTEST(the_device_properties_are_laid_out_as_the_hip_headers_lay_them_out) {
  // The static_asserts above are the test; this says so when it runs.
  VCHECK_EQ(sizeof(DevicePropR0600), 1472u);
}

VTEST_MAIN
