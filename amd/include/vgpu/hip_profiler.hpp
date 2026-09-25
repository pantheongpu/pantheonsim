// What libamdhip64 tells a profiler: the devices it has, and each thing it does
// as it does it -- a HIP call, a code object placed on a device, a kernel run
// with what it counted, a copy, an allocation.
//
// VirtualGPU's librocprofiler-sdk attaches here, the way a CUPTI front end
// reads vgpu/profiling.hpp on the NVIDIA side. It is a separate library, so
// this is a C interface of plain structures: the two are built together and
// never mixed across versions, and nothing outside VirtualGPU uses it.
//
// With nothing attached a HIP call pays one load of a pointer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <time.h>

#include "vgpu/amd_exec.hpp"

namespace vgpu::amd::hipprof {

// A simulated device, as its profile describes it.
struct Device {
  int ordinal = 0;
  const char* gfx = "";          // "gfx942"
  const char* model = "";        // "AMD Instinct MI300X"
  uint16_t vendor_id = 0, device_id = 0;
  uint32_t compute_units = 0;
  uint32_t wave_size = 0;
  uint32_t lds_bytes = 0;         // per work-group
  uint32_t max_threads_per_cu = 0;
  uint32_t max_workgroup = 0;
  uint32_t max_block[3] = {0, 0, 0};
  uint32_t max_grid[3] = {0, 0, 0};
  uint32_t clock_mhz = 0;
  uint64_t vram_bytes = 0;
  uint32_t pci_bus = 0;           // where it sits: 0000:<bus>:00.0
  uint8_t uuid[16] = {};
};

// A kernel in a code object that was placed on a device.
struct KernelSymbol {
  uint64_t kernel_id = 0;
  const char* name = "";          // as the code object names it (mangled)
  uint64_t entry = 0;             // where its code starts
  uint32_t kernarg_size = 0, kernarg_alignment = 0;
  uint32_t group_segment = 0, private_segment = 0;
  uint32_t sgprs = 0, vgprs = 0, agprs = 0;
};

// A code object placed on a device.
struct CodeObject {
  uint64_t id = 0;
  int device = 0;
  const char* uri = "";
  const void* image = nullptr;     // the code object itself, in host memory
  uint64_t image_size = 0;
  uint64_t load_base = 0, load_size = 0;
};

// A kernel launch.
struct Launch {
  int device = 0;
  uint64_t kernel_id = 0;
  uint64_t dispatch_id = 0;
  uint64_t stream = 0;
  uint32_t groups[3] = {1, 1, 1};
  uint32_t group_size[3] = {1, 1, 1};
  uint32_t group_segment = 0, private_segment = 0;
};

enum class Copy : uint32_t { HostToHost = 1, HostToDevice, DeviceToHost, DeviceToDevice };

struct Hooks {
  size_t size = sizeof(Hooks);
  // A HIP call begins; what it returns is handed back when the call ends.
  // Calls one HIP function makes through another are the outer call's.
  uint64_t (*api_enter)(const char* name) = nullptr;
  void (*api_exit)(uint64_t token) = nullptr;
  void (*code_object_loaded)(const CodeObject& object, const KernelSymbol* kernels, size_t count) = nullptr;
  // A launch is about to run; what this returns comes back with its end,
  // with what the dispatch counted -- or with nullptr, when it failed and
  // there is nothing to count.
  void* (*launching)(const Launch& launch) = nullptr;
  void (*launched)(const Launch& launch, const DispatchStats* stats, uint64_t start_ns, uint64_t end_ns,
                   void* token) = nullptr;
  void (*copied)(Copy kind, int src_device, int dst_device, uint64_t bytes, uint64_t dst, uint64_t src,
                 uint64_t start_ns, uint64_t end_ns) = nullptr;
  void (*allocated)(int device, uint64_t address, uint64_t bytes, bool freed, uint64_t start_ns,
                    uint64_t end_ns) = nullptr;
};

// The clock every time above is on: CLOCK_BOOTTIME in nanoseconds, which is
// the one /proc/<pid>/stat's start time is counted on.
inline uint64_t now_ns() {
  timespec ts{};
  ::clock_gettime(CLOCK_BOOTTIME, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

}  // namespace vgpu::amd::hipprof

extern "C" {
// Attaches a profiler, or with nullptr detaches it. Returns the number of
// devices, having brought the runtime up to know them (-1 if it cannot).
int vgpu_hip_profiler_attach(const vgpu::amd::hipprof::Hooks* hooks);
// The device with that ordinal. Returns 0, or -1 for one that does not exist.
int vgpu_hip_profiler_device(int ordinal, vgpu::amd::hipprof::Device* out);
}
