// E1: Intel in the engine's profile layer (intel/profiles/*.yaml), and the limits a SPIR-V launch is checked against.
//
// The profiles are spec-derived, not measured. What these tests pin down is the plumbing: the three profiles load
// and are found by load_gpu() but not listed by available_gpus() (every other test that walks that list assumes a
// register file and telemetry the Intel parts do not have); the resource limits come from the profile (shared local
// memory 384 KiB on Crescent Island, 128 KiB on the Battlemage parts, the register file and thread counts derived
// from the Xe core); a work-group at exactly the profile's SLM runs and one byte more is refused by name; a launch
// at a sub-group width the profile does not list is refused; and the loader refuses a malformed Intel profile.
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/profile.hpp"
#include "vgpu/ptx/builder.hpp"
#include "vgpu/registry.hpp"
#include "ast_ops.hpp"
#include "vtest.hpp"

using namespace vgpu;
using namespace astops;
using vgpu::exec::LaunchConfig;

namespace {

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

// out[0] = the launch's sub-group width, written by the one thread there is.
ptx::Module build_width() {
  ptx::KernelBuilder kb("width");
  const size_t po = kb.add_param("out", u64());
  const Reg out = kb.reg("%out", u64()), v = kb.reg("%v", u32());
  kb.emit(ld(Space::Param, u64(), out, kb.param_addr(po)), "ld");
  kb.emit(mov(u32(), v, SR(Sreg::WarpSize)), "mov");
  kb.emit(st(Space::Global, u32(), ptx::KernelBuilder::reg_addr(out, 0), RO(v)), "st");
  kb.emit(OpRet{}, "ret");
  ptx::ModuleBuilder mb;
  mb.add_entry(kb.finish());
  return mb.finish();
}

struct Part {
  const char* id;
  uint32_t slm;       // bytes of shared local memory a work-group may have
  uint32_t xe_cores;
};
const Part kParts[] = {{"intel/crescent-island", 384u * 1024, 32}, {"intel/arc-b580", 128u * 1024, 20},
                       {"intel/arc-pro-b60", 128u * 1024, 20}};

}  // namespace

VTEST(the_intel_profiles_load_and_state_what_they_are) {
  for (const Part& part : kParts) {
    const DeviceProfile p = load_gpu(part.id);
    VCHECK_EQ(p.id, std::string(part.id));
    VCHECK_EQ(p.vendor, "intel");
    VCHECK(!p.architecture.empty());
    VCHECK(!p.verified);                       // spec-derived, no card to compare with
    VCHECK_EQ(p.cc_major, 0);                  // an Xe part has no compute capability
    VCHECK(p.gcn_arch.empty());
    VCHECK_EQ(p.limits.shared_mem_per_block, part.slm);
    VCHECK_EQ(p.limits.shared_mem_per_block_optin, part.slm);
    VCHECK_EQ(p.limits.multiprocessors, part.xe_cores);
    VCHECK_EQ(p.reserved_smem_per_block(), 0u);
    VCHECK(p.xe.known());
    VCHECK(p.xe.has_sub_group_size(p.warp_size));
    VCHECK(!p.placeholders.empty());
    VCHECK(p.is_placeholder("xe.sub_group_sizes"));
    VCHECK(!p.is_placeholder("limits.shared_mem_per_block_bytes"));   // sourced, not a stand-in
  }
  // Crescent Island's 384 KiB is the NEO source note (maxProgrammableSlmSize); its EU count is a stand-in.
  const DeviceProfile cri = load_gpu("intel/crescent-island");
  VCHECK_EQ(cri.limits.shared_mem_per_block, 393216u);
  VCHECK(cri.is_placeholder("xe.eus_per_xe_core"));
  VCHECK_EQ(cri.architecture, "xe3p-xpc");
  VCHECK_EQ(cri.xe.threads_per_eu, 8u);
  // Ids are case-insensitive like every other.
  VCHECK_EQ(load_gpu("Intel/Arc-B580").id, "intel/arc-b580");
}

VTEST(the_register_file_and_thread_ceilings_are_derived_from_the_xe_core) {
  for (const Part& part : kParts) {
    const DeviceProfile p = load_gpu(part.id);
    const uint32_t threads = p.xe.eus_per_xe_core * p.xe.threads_per_eu;
    const uint64_t words = uint64_t{threads} * p.xe.grf_per_thread * p.xe.grf_bytes / 4;
    VCHECK_EQ(uint64_t{p.limits.registers_per_sm}, words);
    VCHECK_EQ(p.limits.registers_per_block, p.limits.registers_per_sm);   // a work-group lives in one Xe core
    VCHECK_EQ(p.limits.max_blocks_per_sm, threads);
    VCHECK_EQ(p.limits.max_threads_per_sm, threads * 32u);                // the widest sub-group fills every thread
    VCHECK_EQ(p.limits.max_registers_per_thread, p.xe.grf_per_thread * p.xe.grf_bytes / 4 / p.warp_size);
    VCHECK_EQ(p.limits.shared_mem_per_sm, p.limits.shared_mem_per_block_optin);
  }
}

VTEST(intel_profiles_are_found_by_the_registry_but_not_listed_with_the_others) {
  const auto listed = available_gpus();
  const auto engine = available_engine_gpus();
  VCHECK_EQ(engine.size(), size_t{3});
  for (const Part& part : kParts) {
    VCHECK(std::find(engine.begin(), engine.end(), part.id) != engine.end());
    VCHECK(std::find(listed.begin(), listed.end(), part.id) == listed.end());
  }
}

// The point of the profile: a work-group's shared local memory is the profile's, not an NVIDIA part's 227 KiB.
VTEST(a_work_group_at_the_profiles_slm_runs_and_one_byte_more_is_refused_by_name) {
  const ptx::Module m = build_width();
  for (const Part& part : kParts) {
    MemoryManager mem{1 << 20};
    const DeviceProfile prof = load_gpu(part.id);
    const uint64_t out = mem.alloc(8);
    LaunchConfig ok;
    ok.shared_bytes = part.slm;
    exec::launch(m.entries[0], ok, {arg_u64(out)}, mem, prof);
    VCHECK_EQ(static_cast<uint32_t>(mem.load_scalar(out, 4)), prof.warp_size);   // the default sub-group width

    LaunchConfig bad;
    bad.shared_bytes = part.slm + 1;
    auto err = VCAPTURE(Error, exec::launch(m.entries[0], bad, {arg_u64(out)}, mem, prof));
    VCHECK(err.code() == Err::LaunchResources || err.code() == Err::LaunchConfig);
    VCHECK_CONTAINS(err.what(), part.id);
    VCHECK_CONTAINS(err.what(), "shared");
    // validate_launch says the same without running anything.
    auto v = VCAPTURE(Error, exec::validate_launch(m.entries[0], bad, prof));
    VCHECK_CONTAINS(v.what(), part.id);
  }
  // NVIDIA's cap is untouched: 384 KiB is above an H100's 227 KiB.
  MemoryManager mem{1 << 20};
  const DeviceProfile h100 = load_gpu("nvidia/h100");
  LaunchConfig big;
  big.shared_bytes = 384 * 1024;
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], big, {arg_u64(mem.alloc(8))}, mem, h100));
  VCHECK_CONTAINS(err.what(), "nvidia/h100");
}

VTEST(occupancy_follows_the_profiles_shared_memory_and_register_file) {
  const ptx::Module m = build_width();
  const DeviceProfile cri = load_gpu("intel/crescent-island");
  // 192 KiB a work-group: two fit the Xe core's 384 KiB, three do not.
  auto r = exec::kernel_resources(m.entries[0], cri, 64, 192u * 1024);
  VCHECK_EQ(r.occupancy.blocks_per_sm, 2u);
  VCHECK_EQ(std::string(r.occupancy.limited_by), "shared");
  // Nothing shared: bound by the hardware threads. 64 invocations at 16 lanes are 4 threads, 64 threads fit.
  auto s = exec::kernel_resources(m.entries[0], cri, 64, 0);
  VCHECK(s.occupancy.blocks_per_sm >= 1u);
}

VTEST(a_sub_group_width_the_profile_does_not_list_is_refused) {
  const ptx::Module m = build_width();
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(8);
  const DeviceProfile prof = load_gpu("intel/crescent-island");
  for (uint32_t w : {8u, 16u, 32u}) {
    LaunchConfig cfg;
    cfg.warp_size = w;
    exec::launch(m.entries[0], cfg, {arg_u64(out)}, mem, prof);
    VCHECK_EQ(static_cast<uint32_t>(mem.load_scalar(out, 4)), w);
  }
  LaunchConfig wide;
  wide.warp_size = 64;   // the engine can run 64 lanes; an Intel part does not
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], wide, {arg_u64(out)}, mem, prof));
  VCHECK(err.code() == Err::LaunchConfig);
  VCHECK_CONTAINS(err.what(), "sub-group size 64");
  VCHECK_CONTAINS(err.what(), "intel/crescent-island");
}

namespace {
std::string xe_profile(const std::string& xe_body, const std::string& warp = "16") {
  return "id: intel/test\nvendor: intel\nmodel: \"T\"\narchitecture: xe2-hpg\nwarp_size: " + warp +
         "\nvram_bytes: 1048576\nverified: false\nlimits:\n  max_threads_per_block: 1024\n"
         "  max_block_dim: [1024, 1024, 1024]\n  max_grid_dim: [65535, 65535, 65535]\n"
         "  shared_mem_per_block_bytes: 65536\n  shared_mem_per_block_optin_bytes: 65536\n  multiprocessors: 4\n" +
         xe_body;
}
const char* kGoodXe =
    "xe:\n  eus_per_xe_core: 8\n  threads_per_eu: 8\n  grf_per_thread: 128\n  grf_bytes: 32\n  sub_group_sizes: [16, 32]\n";
}  // namespace

VTEST(the_loader_checks_an_intel_profile) {
  const DeviceProfile ok = DeviceProfile::from_yaml(xe_profile(kGoodXe), "good");
  VCHECK_EQ(ok.limits.registers_per_sm, 65536u);
  VCHECK_EQ(ok.limits.max_registers_per_thread, 64u);
  // No xe section.
  auto none = VCAPTURE(Error, DeviceProfile::from_yaml(xe_profile(""), "none"));
  VCHECK(none.code() == Err::ProfileParse);
  VCHECK_CONTAINS(none.what(), "xe");
  // A width that is not 8, 16 or 32.
  auto bad = VCAPTURE(Error, DeviceProfile::from_yaml(
      xe_profile("xe:\n  eus_per_xe_core: 8\n  threads_per_eu: 8\n  grf_per_thread: 128\n  grf_bytes: 32\n  sub_group_sizes: [16, 64]\n"), "bad"));
  VCHECK(bad.code() == Err::ProfileParse);
  VCHECK_CONTAINS(bad.what(), "sub_group_sizes");
  // The default width must be one of the listed ones.
  auto off = VCAPTURE(Error, DeviceProfile::from_yaml(xe_profile(kGoodXe, "8"), "off"));
  VCHECK(off.code() == Err::ProfileParse);
  VCHECK_CONTAINS(off.what(), "warp_size 8");
  // A zero count would make every derived ceiling zero.
  auto zero = VCAPTURE(Error, DeviceProfile::from_yaml(
      xe_profile("xe:\n  eus_per_xe_core: 0\n  threads_per_eu: 8\n  grf_per_thread: 128\n  grf_bytes: 32\n  sub_group_sizes: [16]\n"), "zero"));
  VCHECK(zero.code() == Err::ProfileParse);
}

VTEST_MAIN
