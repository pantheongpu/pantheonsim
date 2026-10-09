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
  // The compute microengine firmware (RSMI_FW_BLOCK_MEC) a card of this chip reports: the lowest version
  // RCCL 2.27 accepts without HSA_NO_SCRATCH_RECLAIM (rccl_wrap.cc: 177 for gfx942, 24 for gfx950). No
  // profile records firmware, so this is chosen, not read from a card; 0 where nothing asks.
  uint32_t mec_fw = 0;
};

// `pci_device` is the card's PCI device id (the low 16 bits are used), for the architectures that
// have more than one chip: RDNA2's Navi 21 and Navi 22.
inline Chip chip(const char* architecture, uint32_t pci_device = 0) {
  const auto is = [&](const char* a) { return std::strcmp(architecture, a) == 0; };
  if (is("cdna4")) return {"gfx950", "HBM3E", 1, 4, 32, 1, 8, 32, 4096, 256, 8192, 2000, 24};  // MI350X: 8 XCDs of 4 engines
  // MI455X: 8 compute dies of 32 workgroup processors, 12 HBM4 stacks of 2048 bits, 192 MB of cache. The rest
  // (L1, L2, engine counts, firmware) AMD has not published; chosen like gfx950's.
  if (is("cdna5")) return {"gfx1250", "HBM4", 2, 2, 8, 1, 8, 32, 4096, 192, 24576, 3200, 0};
  if (is("cdna2")) return {"gfx90a", "HBM2E", 1, 4, 8, 1, 1, 16, 8192, 0, 4096, 1600};      // MI250X, one die
  if (is("rdna3")) return {"gfx1100", "GDDR6", 2, 2, 6, 2, 1, 32, 6144, 96, 384, 2500};     // RX 7900 XTX, Navi 31
  if (is("rdna4")) return {"gfx1201", "GDDR6", 2, 2, 4, 2, 1, 32, 8192, 64, 256, 2518};     // RX 9070 XT, Navi 48
  // RX 6700 XT, Navi 22 (0x73df), read from a card: two shader engines of two arrays, 3 MB of L2, 96 MB of
  // Infinity Cache, a 192-bit bus of the same 16 Gbps GDDR6; 40 compute units in four shader arrays.
  if (is("rdna2") && (pci_device & 0xffff) == 0x73df) return {"gfx1031", "GDDR6", 2, 2, 2, 2, 1, 16, 3072, 96, 192, 2000};
  if (is("rdna2")) return {"gfx1030", "GDDR6", 2, 2, 4, 2, 1, 16, 4096, 128, 256, 2000};    // RX 6800 and 6900 XT, Navi 21
  return {"gfx942", "HBM3", 1, 4, 32, 1, 8, 32, 4096, 256, 8192, 1300, 177};                 // MI300X, MI325X
}

}  // namespace vgpu::amd
