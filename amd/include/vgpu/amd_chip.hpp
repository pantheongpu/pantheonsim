// What AMD's runtime and tools report of a chip beyond a device profile, by
// the profile's architecture: its ISA target, its compute units to each of
// HIP's multiprocessors (RDNA's workgroup processor is two), SIMDs to a
// compute unit, shader engines and the arrays in each, compute dies (XCCs),
// the vector L1 and last-level (Infinity) cache, and its memory. From AMD's
// published specifications of each part. The HSA runtime (rocminfo), rocm-smi
// and amd-smi all answer from here, so they agree.
#pragma once

#include <cstdint>
#include <cstring>

namespace vgpu::amd {

struct Chip {
  const char* gfx;        // the ISA target, as rocm_agent_enumerator lists it
  const char* memory;     // the memory type amd-smi names
  uint32_t cus_per_mp;    // compute units to one of HIP's multiprocessors
  uint32_t simds;         // SIMDs to a compute unit
  uint32_t engines;       // shader engines
  uint32_t arrays;        // shader arrays to an engine
  uint32_t xccs;          // compute dies
  uint32_t l1_kb;         // vector L1 per compute unit
  uint32_t l2_kb;         // L2 (per compute die on a multi-die part)
  uint32_t l3_mb;         // last-level cache (0: none)
  uint32_t mem_bits;      // memory interface width
  uint32_t mem_mhz;       // memory clock
};

inline Chip chip(const char* architecture) {
  const auto is = [&](const char* a) { return std::strcmp(architecture, a) == 0; };
  if (is("cdna4")) return {"gfx950", "HBM3E", 1, 4, 32, 1, 8, 32, 4096, 256, 8192, 2000};   // MI350X: 8 XCDs of 4 engines
  if (is("cdna2")) return {"gfx90a", "HBM2E", 1, 4, 8, 1, 1, 16, 8192, 0, 4096, 1600};      // MI250X, one die
  if (is("rdna3")) return {"gfx1100", "GDDR6", 2, 2, 6, 2, 1, 32, 6144, 96, 384, 2500};     // RX 7900 XTX, Navi 31
  if (is("rdna4")) return {"gfx1201", "GDDR6", 2, 2, 4, 2, 1, 32, 8192, 64, 256, 2518};     // RX 9070 XT, Navi 48
  if (is("rdna2")) return {"gfx1030", "GDDR6", 2, 2, 4, 2, 1, 16, 4096, 128, 256, 2000};    // RX 6900 XT, Navi 21
  return {"gfx942", "HBM3", 1, 4, 32, 1, 8, 32, 4096, 256, 8192, 1300};                     // MI300X, MI325X
}

}  // namespace vgpu::amd
