// Running a CDNA kernel: work-groups of wavefronts over the decoded
// instructions of a code object (vgpu/amd_gcn.hpp).
//
// A wavefront is 64 lanes that share one instruction stream. What the PTX
// interpreter has no model for, and what this adds, is the rest of the
// machine the ISA exposes: the scalar registers a wave keeps beside its
// vector ones, and the EXEC mask that says which lanes an instruction writes.
// Divergence here is not a path stack -- it is code saving EXEC, narrowing
// it, and putting it back.
//
// One kernel launch at a time, on a device's memory. This is the layer a HIP
// runtime will sit on.
#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/memory.hpp"

namespace vgpu::amd {

class Hostcall;
class DecodeCache;

// A launch, as the packet a HIP runtime writes describes one.
struct Dispatch {
  const CodeObject* object = nullptr;
  const Kernel* kernel = nullptr;
  uint64_t kernarg = 0;        // where the kernel's arguments are in device memory
  uint32_t groups[3] = {1, 1, 1};    // work-groups
  uint32_t group_size[3] = {1, 1, 1};   // work-items in each
  uint32_t wave_size = 64;
  // The grid in work-items, where it is not a whole number of work-groups
  // (an HSA dispatch packet may ask for that; HIP never does): the last
  // work-group in a dimension then has only what is left over. Zero is a
  // grid of `groups` whole work-groups.
  uint32_t grid_items[3] = {0, 0, 0};
  // Whether a work-group larger than the kernel's metadata allows
  // (max_flat_workgroup_size) is refused, as a HIP runtime refuses the launch.
  // An AQL packet goes to the hardware as it is, and the hardware knows only
  // its own limit, 1024 work-items: an HSA runtime turns this off (ROCm's own
  // copy kernels are launched past what their metadata says).
  bool kernel_limits = true;
  // Whether the hidden arguments are written into the kernarg segment, as a
  // HIP runtime does. An HSA runtime passes the segment to the hardware as the
  // program wrote it, and the runtime above (ROCm's HIP) fills them itself.
  bool fill_hidden = true;
  // LDS the launch adds to what the kernel reserves, which is what a HIP
  // program passes as its third launch parameter.
  uint32_t dynamic_lds = 0;
  // Where a linked code object's image was placed on the device (its
  // `image`), so the program counter runs in device addresses and the code
  // reaches its own constants and variables. Zero for an object not yet
  // linked, whose code is run where its .text says it is.
  uint64_t code_base = 0;
  // The object's instructions as already decoded (vgpu/amd_decode_cache.hpp),
  // for the code where code_base puts it: a runtime keeps one a module, which
  // every launch of its kernels shares. Null makes each launch decode afresh.
  DecodeCache* decoded = nullptr;
  // Where the kernel calls the host (device-side printf), if the runtime gave
  // it anywhere: its hidden_hostcall_buffer argument, and what answers when
  // the kernel raises the doorbell.
  Hostcall* hostcall = nullptr;
  // Other devices' memory the kernel may reach, by device ordinal: those the
  // program enabled peer access to (hipDeviceEnablePeerAccess), null for the
  // rest. Empty for none.
  std::vector<MemoryManager*> peers;
  // A cooperative launch (hipLaunchCooperativeKernel): every work-group is
  // resident at once, so they can wait on one another, and grid_sync is where
  // the device library's grid barrier keeps its count -- ROCm's mg_info, which
  // the kernel finds through its hidden_multigrid_sync_arg argument.
  bool cooperative = false;
  uint64_t grid_sync = 0;
};

// What the GPU's performance counters would count for a dispatch: every
// instruction a wave issues, by the unit that takes it -- the sequencer's
// SQ_INSTS_* counters -- and the flat instructions the texture addresser
// takes (TA_FLAT_*). Each is exact, because every instruction a wave issues
// passes through the interpreter once. What a counter measures in cycles,
// stalls or cache hits has no model here and is not counted at all.
struct InstructionCounts {
  uint64_t valu = 0;      // vector ALU, matrix instructions included
  uint64_t mfma = 0;      // matrix fused multiply-add
  uint64_t salu = 0;      // scalar ALU (SOP1, SOP2, SOPK, SOPC)
  uint64_t smem = 0;      // scalar memory
  uint64_t vmem = 0;      // vector memory: buffer, and every flat, global and scratch access
  uint64_t flat = 0;      // flat, global and scratch
  uint64_t lds = 0;       // LDS, and a flat access that reached it
  uint64_t branch = 0;    // s_branch and s_cbranch_*
  uint64_t sendmsg = 0;
  uint64_t gds = 0;       // never issued: GDS is not modelled, and a kernel using it is refused
  uint64_t flat_read = 0, flat_write = 0, flat_atomic = 0;
};

// What a dispatch did, which is what a launch reports and what the tests
// check the work by.
struct DispatchStats {
  uint64_t waves = 0;
  uint64_t instructions = 0;
  uint64_t barriers = 0;
  InstructionCounts counts;
  // Waves by how many of their lanes were work-items when they started: a
  // group whose size is not a multiple of 64 ends in a wave with fewer.
  uint64_t waves_lt16 = 0, waves_lt32 = 0, waves_lt48 = 0, waves_lt64 = 0, waves_eq64 = 0;

  // What another share of the same dispatch did, added in.
  void add(const DispatchStats& o) {
    waves += o.waves;
    instructions += o.instructions;
    barriers += o.barriers;
    InstructionCounts& c = counts;
    const InstructionCounts& a = o.counts;
    c.valu += a.valu, c.mfma += a.mfma, c.salu += a.salu, c.smem += a.smem, c.vmem += a.vmem;
    c.flat += a.flat, c.lds += a.lds, c.branch += a.branch, c.sendmsg += a.sendmsg, c.gds += a.gds;
    c.flat_read += a.flat_read, c.flat_write += a.flat_write, c.flat_atomic += a.flat_atomic;
    waves_lt16 += o.waves_lt16, waves_lt32 += o.waves_lt32, waves_lt48 += o.waves_lt48;
    waves_lt64 += o.waves_lt64, waves_eq64 += o.waves_eq64;
  }
};

// Runs the dispatch to completion. Throws Err::Unsupported naming the
// instruction where one is decoded but not implemented, and
// Err::InvalidValue for a dispatch the kernel cannot take (a work-group
// larger than it allows, or more LDS than the device has).
DispatchStats execute(const Dispatch& d, MemoryManager& mem);

// The lock a device read-modify-write at this address takes while work-groups
// run on several threads -- what the host takes too when it changes memory a
// kernel changes with atomics (vgpu/amd_hostcall.hpp).
std::mutex& memory_atomic_lock(uint64_t addr);

}  // namespace vgpu::amd
