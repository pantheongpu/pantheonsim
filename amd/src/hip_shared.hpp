// What VirtualGPU's HSA runtime (hsa_api.cpp) takes from its HIP runtime
// (hip_api.cpp), which it shares a library with: the devices, and loading
// and running code on them. A program that uses both -- as ROCm's own
// libraries do -- sees one set of devices and one memory, as it would with
// ROCm's runtimes, where HIP is built on HSA.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/profile.hpp"

namespace vgpu::amd::shared {

struct Loaded;   // a code object loaded onto a device

// Brings the devices up, as the first HIP call does; false, and why, where
// the environment names no AMD GPU.
bool start(std::string* why);
int device_count();
const DeviceProfile& profile(int ordinal);
MemoryManager& memory(int ordinal);

// Loads a code object (an ELF; a bundle's is chosen by the caller) onto a
// device, as hipModuleLoadData does. Null, and why, where it cannot.
const Loaded* load(int ordinal, const void* bytes, size_t size, std::string* why);
void unload(const Loaded* m);
const CodeObject& object(const Loaded* m);
// A linked object's image as it was loaded, kept on the host: what ROCm's
// loader answers hsa_ven_amd_loader_query_host_address with.
const std::vector<uint8_t>& host_image(const Loaded* m);
// Where a linked object's image is on its device; a kernel's descriptor is
// this plus Kernel::descriptor.
uint64_t code_base(const Loaded* m);

// Runs a kernel of `m` on its device, on the calling thread, over a grid of
// `grid` work-items in work-groups of `group_size` (the last one in a
// dimension short where the grid does not divide), with arguments the caller
// has placed at `kernarg` (an address the device reaches). False, and why,
// if the kernel faulted or could not start.
// A cooperative dispatch has every work-group resident at once, so they may
// wait on one another; the kernarg segment is used as the caller wrote it,
// hidden arguments and all.
bool run(int ordinal, const Loaded* m, const Kernel& k, const uint32_t grid[3], const uint32_t group_size[3],
         uint32_t dynamic_lds, uint64_t kernarg, bool cooperative, std::string* why);

// Lets device `from`'s kernels reach device `to`'s memory, as
// hipDeviceEnablePeerAccess does.
void allow_peer(int from, int to);
// The device whose memory holds `address`, or -1.
int owner(uint64_t address);

// Host memory every device's kernels reach at its own address, and no longer.
void map_host(void* p, size_t n);
void unmap_host(void* p);

// Copies between two addresses, each host memory or any device's, as
// hipMemcpyDefault does.
bool copy(void* dst, const void* src, size_t n, std::string* why);

}  // namespace vgpu::amd::shared
