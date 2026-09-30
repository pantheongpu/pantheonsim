// Register pressure analysis.
//
// PTX declares *virtual* registers in SSA-ish form, so counting `.reg`
// declarations wildly overstates what a thread actually occupies: ptxas
// allocates physical registers by live range, and many virtual registers
// share one. This does the same thing -- liveness over the instruction
// stream, then a linear scan for the peak -- to estimate the physical
// register footprint of a kernel.
//
// The result drives real, functional behavior:
//   * a launch whose block needs more registers than the device allows fails
//     with "too many resources requested for launch", exactly as on hardware;
//   * occupancy (resident warps and blocks per SM) is reported to tools.
//
// It is an estimate, not ptxas's answer: VirtualGPU does not run NVIDIA's
// allocator, and its scheduling and rematerialization choices are its own.
// The count is deterministic and monotonic in real pressure, which is what
// makes it useful for catching resource errors in CI.
#pragma once

#include <cstdint>

#include "vgpu/ptx/ast.hpp"

namespace vgpu::ptx {

struct RegisterUsage {
  // Physical 32-bit registers per thread. A 64-bit value occupies a pair, as
  // it does on the hardware.
  uint32_t regs_per_thread = 0;
  // Predicate registers are a separate, smaller file on NVIDIA parts.
  uint32_t pred_regs = 0;
  // Bytes of .local frame per thread (spill/stack space).
  uint32_t local_bytes = 0;
  // Peak before rounding to the allocation granularity, for diagnostics.
  uint32_t peak_live = 0;
  // Registers past the architectural maximum, which the real compiler spills to
  // local memory rather than failing over. Non-zero means regs_per_thread has
  // been clamped and local_bytes includes the spill.
  uint32_t spilled_regs = 0;
};

// Computes the register footprint of a kernel. Deterministic.
RegisterUsage analyze_registers(const EntryFn& fn);

// The registers an instruction writes and reads, each as id * 2 + wide (the
// 32- and 64-bit files number their registers separately, so an id alone
// does not name one).
void instr_registers(const Instr& ins, std::vector<uint32_t>& defs, std::vector<uint32_t>& uses);

// Resident warps and blocks per SM for a given block size, given a device's
// register file and warp/block ceilings -- the standard occupancy calculation.
struct Occupancy {
  uint32_t warps_per_block = 0;
  uint32_t blocks_per_sm = 0;
  uint32_t warps_per_sm = 0;
  // Which ceiling bound the result: "registers", "warps", "blocks", "shared".
  const char* limited_by = "warps";
};

// How a device hands out registers and shared memory, as CUDA's occupancy
// calculator (cuda_occupancy.h) states it for NVIDIA parts. The defaults --
// everything in units of one, nothing reserved -- are the plain arithmetic.
struct OccupancyRules {
  uint32_t reg_alloc_unit = 1;    // registers per warp are allocated in these
  uint32_t sub_partitions = 1;    // a warp's registers come from one of these
  uint32_t regs_per_block = 0;    // the per-block limit, checked as the hardware
                                  // does (warps rounded up to sub_partitions); 0: none
  uint32_t smem_alloc_unit = 1;   // shared memory per block is allocated in these
  uint32_t reserved_smem = 0;     // the driver's own shared memory per block
};

Occupancy compute_occupancy(uint32_t regs_per_thread, uint32_t threads_per_block,
                            uint32_t static_shared_bytes, uint32_t dynamic_shared_bytes,
                            uint32_t regs_per_sm, uint32_t max_threads_per_sm,
                            uint32_t max_blocks_per_sm, uint32_t shared_per_sm,
                            uint32_t warp_size, const OccupancyRules& rules = {});

}  // namespace vgpu::ptx
