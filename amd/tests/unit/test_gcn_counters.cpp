// The counts a profiler reads (SQ_INSTS_*, SQ_WAVES*, TA_FLAT_*), checked
// against ROCm's own listing of the kernel.
//
// The kernel, amd/tests/data/counters.c, has no branches: every wave issues
// each instruction up to s_endpgm exactly once. So what each counter should
// read is the number of instructions of its kind in llvm-objdump's listing,
// times the number of waves -- sorted here by the assembler's mnemonics, not
// by the decoder's encodings, so the two are independent.
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "counters_oracle.hpp"
#include "vgpu/amd_exec.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

const std::string kData = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/";

amd::CodeObject object() {
  const std::string path = kData + "counters.gfx942.o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

uint64_t kernargs(MemoryManager& mem, const amd::Kernel& k, const std::vector<uint64_t>& values) {
  std::vector<uint8_t> buf(k.kernarg_size, 0);
  for (size_t i = 0; i < values.size() && i < k.args.size(); ++i) {
    const amd::KernelArg& a = k.args[i];
    for (uint32_t b = 0; b < a.size && b < 8; ++b) buf[a.offset + b] = static_cast<uint8_t>(values[i] >> (8 * b));
  }
  const uint64_t p = mem.alloc(buf.size());
  mem.write(p, buf.data(), buf.size());
  return p;
}

amd::DispatchStats run(const amd::CodeObject& o, const char* name, uint32_t groups, uint32_t group_size,
                       bool use_lds) {
  MemoryManager mem(64ull << 20);
  const size_t n = size_t{groups} * 128;
  std::vector<float> in(n, 1.5f);
  const uint64_t pin = mem.alloc(n * 4), pout = mem.alloc(n * 4), phits = mem.alloc(n * 4), pg = mem.alloc(n * 4);
  mem.write(pin, in.data(), n * 4);
  const amd::Kernel* k = amd::find_kernel(o, name);
  if (!k) throw vtest::Failure(std::string("no kernel named ") + name);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = kernargs(mem, *k, {pin, pout, phits, pg, use_lds ? 1u : 0u});
  d.groups[0] = groups;
  d.group_size[0] = group_size;
  return amd::execute(d, mem);
}

}  // namespace

VTEST(the_counting_kernel_has_no_branches) {
  // What makes a count the listing's times the waves.
  const Expected e = from_listing(kData);
  VCHECK_EQ(e.branch, 0u);
  VCHECK(e.valu > 0 && e.mfma == 1 && e.salu > 0 && e.smem > 0 && e.lds == 2 && e.generic == 1);
  VCHECK(e.reads > 0 && e.writes > 0 && e.atomics == 1);
}

VTEST(each_count_is_the_listing_times_the_waves) {
  const amd::CodeObject o = object();
  const Expected e = from_listing(kData);
  const amd::DispatchStats s = run(o, "counted_mix", 3, 64, false);
  const uint64_t w = 3;
  VCHECK_EQ(s.waves, w);
  VCHECK_EQ(s.instructions, e.all * w);
  VCHECK_EQ(s.counts.valu, e.valu * w);
  VCHECK_EQ(s.counts.mfma, e.mfma * w);
  VCHECK_EQ(s.counts.salu, e.salu * w);
  VCHECK_EQ(s.counts.smem, e.smem * w);
  VCHECK_EQ(s.counts.vmem, e.vmem * w);
  VCHECK_EQ(s.counts.flat, e.flat * w);
  VCHECK_EQ(s.counts.lds, e.lds * w);   // the generic pointer reached device memory
  VCHECK_EQ(s.counts.branch, 0u);
  VCHECK_EQ(s.counts.sendmsg, 0u);
  VCHECK_EQ(s.counts.gds, 0u);
  VCHECK_EQ(s.counts.flat_read, e.reads * w);
  VCHECK_EQ(s.counts.flat_write, e.writes * w);
  VCHECK_EQ(s.counts.flat_atomic, e.atomics * w);
  // The instruction mix: each class the listing has, and matrix work.
  using amd::gcn::Mix;
  using amd::gcn::MopsType;
  VCHECK(e.add_f32 && e.cvt && e.int32 && e.int64 && e.mops_f16);
  VCHECK_EQ(s.counts.mix[static_cast<int>(Mix::AddF32)], e.add_f32 * w);
  VCHECK_EQ(s.counts.mix[static_cast<int>(Mix::Cvt)], e.cvt * w);
  VCHECK_EQ(s.counts.mix[static_cast<int>(Mix::Int32)], e.int32 * w);
  VCHECK_EQ(s.counts.mix[static_cast<int>(Mix::Int64)], e.int64 * w);
  VCHECK_EQ(s.counts.mops[static_cast<int>(MopsType::F16)], e.mops_f16 * w);
  VCHECK_EQ(s.counts.vmem_rd, e.vmem_rd * w);
  VCHECK_EQ(s.counts.vmem_wr, e.vmem_wr * w);
}

VTEST(bytes_are_the_lanes_times_what_each_instruction_moves) {
  const amd::CodeObject o = object();
  const Expected e = from_listing(kData);
  // The listing has a dword load, a dword atomic, a flat and a global dword store, and an LDS dword write and read: a non-zero
  // number of each kind, so a count that ignored one would show.
  VCHECK(e.vmem_lane_bytes_rd == 4 && e.vmem_lane_bytes_wr == 12 && e.lds_lane_bytes_rd == 4 && e.lds_lane_bytes_wr == 4);
  const amd::DispatchStats s = run(o, "counted_mix", 3, 64, false);
  const uint64_t lanes = 3 * 64;   // every lane of every wave is active
  VCHECK_EQ(s.counts.vmem_bytes_rd, e.vmem_lane_bytes_rd * lanes);
  VCHECK_EQ(s.counts.vmem_bytes_wr, e.vmem_lane_bytes_wr * lanes);
  VCHECK_EQ(s.counts.lds_bytes_rd, e.lds_lane_bytes_rd * lanes);
  VCHECK_EQ(s.counts.lds_bytes_wr, e.lds_lane_bytes_wr * lanes);
  // The flat store reaching LDS is still a vector memory instruction, counted in full as vector memory bytes.
  const amd::DispatchStats t = run(o, "counted_mix", 2, 64, true);
  VCHECK_EQ(t.counts.vmem_bytes_rd, e.vmem_lane_bytes_rd * 2 * 64);
  VCHECK_EQ(t.counts.vmem_bytes_wr, e.vmem_lane_bytes_wr * 2 * 64);
  VCHECK_EQ(t.counts.lds_bytes_wr, e.lds_lane_bytes_wr * 2 * 64);
}

VTEST(bytes_count_only_the_lanes_a_wave_started_with) {
  const amd::CodeObject o = object();
  // counted_copy moves one dword in and one out a work-item. 2 groups of 100: 200 work-items, in waves of 64 and 36.
  const amd::DispatchStats s = run(o, "counted_copy", 2, 100, false);
  VCHECK_EQ(s.counts.vmem_bytes_rd, 4u * 200);
  VCHECK_EQ(s.counts.vmem_bytes_wr, 4u * 200);
  VCHECK_EQ(s.counts.lds_bytes_rd + s.counts.lds_bytes_wr, 0u);
  // One wave of 20 lanes, and one of 8.
  VCHECK_EQ(run(o, "counted_copy", 1, 20, false).counts.vmem_bytes_rd, 4u * 20);
  VCHECK_EQ(run(o, "counted_copy", 1, 8, false).counts.vmem_bytes_wr, 4u * 8);
}

VTEST(bytes_add_up_and_scale_like_the_other_counts) {
  amd::DispatchStats a, b;
  a.counts.vmem_bytes_rd = 3, a.counts.vmem_bytes_wr = 5, a.counts.lds_bytes_rd = 7, a.counts.lds_bytes_wr = 11;
  b.counts.vmem_bytes_rd = 100, b.counts.vmem_bytes_wr = 200, b.counts.lds_bytes_rd = 300, b.counts.lds_bytes_wr = 400;
  a.add(b);
  VCHECK_EQ(a.counts.vmem_bytes_rd, 103u);
  VCHECK_EQ(a.counts.vmem_bytes_wr, 205u);
  VCHECK_EQ(a.counts.lds_bytes_rd, 307u);
  VCHECK_EQ(a.counts.lds_bytes_wr, 411u);
  a.scale(2);
  VCHECK_EQ(a.counts.vmem_bytes_rd, 206u);
  VCHECK_EQ(a.counts.lds_bytes_wr, 822u);
}

VTEST(a_flat_access_that_reaches_lds_counts_as_lds_too) {
  const amd::CodeObject o = object();
  const Expected e = from_listing(kData);
  const amd::DispatchStats s = run(o, "counted_mix", 2, 64, true);
  VCHECK_EQ(s.counts.lds, (e.lds + e.generic) * 2);
  VCHECK_EQ(s.counts.flat, e.flat * 2);   // and is still a flat instruction
  VCHECK_EQ(s.counts.vmem, e.vmem * 2);
}

VTEST(waves_are_counted_by_the_lanes_they_start_with) {
  const amd::CodeObject o = object();
  // 100 work-items a group: a full wave and one of 36.
  const amd::DispatchStats s = run(o, "counted_copy", 2, 100, false);
  VCHECK_EQ(s.waves, 4u);
  VCHECK_EQ(s.waves_eq64, 2u);
  VCHECK_EQ(s.waves_lt64, 2u);
  VCHECK_EQ(s.waves_lt48, 2u);
  VCHECK_EQ(s.waves_lt32, 0u);
  VCHECK_EQ(s.waves_lt16, 0u);
  // A short wave issues what a full one does.
  VCHECK_EQ(s.counts.flat_write, 4u);
  // 20 work-items: one wave of 20; 8: one of 8.
  const amd::DispatchStats t = run(o, "counted_copy", 1, 20, false);
  VCHECK_EQ(t.waves_lt32, 1u);
  VCHECK_EQ(t.waves_lt16, 0u);
  VCHECK_EQ(t.waves_eq64, 0u);
  VCHECK_EQ(run(o, "counted_copy", 1, 8, false).waves_lt16, 1u);
}

VTEST_MAIN
