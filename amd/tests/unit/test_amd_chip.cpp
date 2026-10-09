// Every AMD profile has its chip's facts (vgpu/amd_chip.hpp), not another
// chip's: rocminfo, rocm-smi, amd-smi and rocm_agent_enumerator all answer
// from them, and an architecture the table does not know falls back to the
// MI300X's -- which is how every simulated GPU once reported MI300X's
// target, chip ID and shader engines. A new profile fails here until its
// chip is added.
#include <string>

#include "vgpu/amd_chip.hpp"
#include "vgpu/profile.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

VTEST(every_amd_profile_has_its_own_chip) {
  int amd = 0;
  for (const auto& id : vgpu::available_gpus()) {
    const vgpu::DeviceProfile p = vgpu::load_gpu(id);
    if (p.vendor != "amd") continue;
    ++amd;
    const vgpu::amd::Chip c = vgpu::amd::chip(p.architecture.c_str());
    // The target the chip table gives is the profile's own.
    VCHECK_EQ(std::string(c.gfx), p.gcn_arch);
    // gfx10 and later (gfx1250 too) have no sramecc or xnack settings in their target ID: the compiler rejects
    // "gfx1250:sramecc+:xnack-" ("invalid target ID"), which is what hiprtc then reported for a device named so.
    if (p.gcn_arch.rfind("gfx1", 0) == 0) VCHECK_EQ(p.gcn_arch_full, p.gcn_arch);
    // RDNA counts a workgroup processor as HIP's multiprocessor, two compute
    // units, and so does CDNA 5 (gfx1250, built on gfx12); CDNA one; and each has a memory interface and an L1 and L2.
    VCHECK_EQ(c.cus_per_mp, p.architecture.rfind("rdna", 0) == 0 || p.architecture == "cdna5" ? 2u : 1u);
    VCHECK(c.mem_bits > 0 && c.l1_kb > 0 && c.l2_kb > 0 && c.engines > 0 && c.simds > 0);
    // The L2 is the profile's too, per compute die.
    VCHECK_EQ(uint64_t{c.l2_kb} * 1024, p.limits.l2_cache_bytes);
  }
  VCHECK(amd >= 8);
}

VTEST_MAIN
