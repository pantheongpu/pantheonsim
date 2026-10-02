// Device-memory queries for the vendor-library shims.
//
// A library that moves bytes on the host side of the simulator (cuFile's file
// I/O, nvCOMP's codecs, NVSHMEM's symmetric heap) has to know where an
// allocation begins and ends to answer what NVIDIA's library answers -- "the
// range runs past the allocation" is a status of its own there -- and to refuse
// a bad pointer before a copy through the runtime fails on it. libcudart
// exports this; the shims link against it.
#ifndef VGPU_RUNTIME_SHIM_MEMORY_HPP
#define VGPU_RUNTIME_SHIM_MEMORY_HPP

#include <cstddef>

// True when `p` lies in a device allocation (cudaMalloc and its relatives) of
// any simulated device, with that allocation's base and size. False for host
// memory -- pinned and managed memory included -- for a device-heap block, and
// for an address no allocation covers.
bool vgpu_device_allocation(const void* p, void** base, std::size_t* size);

#endif  // VGPU_RUNTIME_SHIM_MEMORY_HPP
