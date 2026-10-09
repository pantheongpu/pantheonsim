// What driver_graph.cpp (the driver API's graphs) needs from driver_api.cpp: running a kernel from a copy
// of its parameters, and nothing else. Internal to libvgpucuda.
#pragma once

#include <cstdint>
#include <vector>

namespace vgpu_driver __attribute__((visibility("hidden"))) {

// Copies the parameters of one launch of `func` (a CUfunction, or a CUkernel) out of the caller's
// kernelParams array -- one pointer per parameter -- as cuLaunchKernel does. Returns 0, or a CUresult.
int copy_kernel_args(void* func, void** kernel_params, void** extra, std::vector<std::vector<uint8_t>>* out);

// Launches `func` now from copied parameters, as cuLaunchKernel on the legacy stream.
int run_kernel(void* func, const unsigned grid[3], const unsigned block[3], unsigned shared,
               const std::vector<std::vector<uint8_t>>& args);

// The runtime event that stands for a driver event in a graph (made on first use; an event the
// runtime made is its own), and the driver event a runtime event stands for (null if none).
void* runtime_event_for(void* driver_event);
void* driver_event_of(void* runtime_event);

}  // namespace vgpu_driver
