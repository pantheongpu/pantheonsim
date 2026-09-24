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
#include <string>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/memory.hpp"

namespace vgpu::amd {

// A launch, as the packet a HIP runtime writes describes one.
struct Dispatch {
  const CodeObject* object = nullptr;
  const Kernel* kernel = nullptr;
  uint64_t kernarg = 0;        // where the kernel's arguments are in device memory
  uint32_t groups[3] = {1, 1, 1};    // work-groups
  uint32_t group_size[3] = {1, 1, 1};   // work-items in each
  uint32_t wave_size = 64;
  // LDS the launch adds to what the kernel reserves, which is what a HIP
  // program passes as its third launch parameter.
  uint32_t dynamic_lds = 0;
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
};

// Runs the dispatch to completion. Throws Err::Unsupported naming the
// instruction where one is decoded but not implemented, and
// Err::InvalidValue for a dispatch the kernel cannot take (a work-group
// larger than it allows, or more LDS than the device has).
DispatchStats execute(const Dispatch& d, MemoryManager& mem);

}  // namespace vgpu::amd
