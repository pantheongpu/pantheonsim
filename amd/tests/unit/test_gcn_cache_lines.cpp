// The cache lines a vector memory access touches (vgpu/amd_memtraffic.hpp). MODELLED, not checked against a card, so the
// golden values here are worked out by hand from the rule (distinct lines of the target's line size) and from the line sizes the
// documents give; they pin the rule, they do not confirm it.
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vgpu/amd_memtraffic.hpp"
#include "vtest.hpp"

using namespace vgpu;
using amd::gcn::cache_lines;
using amd::gcn::Target;

namespace {

constexpr uint64_t kAll = ~0ull;

uint32_t lines(uint64_t base, uint64_t stride, uint32_t bytes, uint64_t exec, uint32_t line) {
  uint64_t addr[64];
  for (uint32_t l = 0; l < 64; ++l) addr[l] = base + l * stride;
  return cache_lines(addr, exec, bytes, line);
}

amd::CodeObject object() {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/counters.gfx942.o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

// counted_copy: each work-item reads one dword at i and writes one at i, i = group * 128 + id.
amd::InstructionCounts copy(uint32_t groups, uint32_t group_size, bool count) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const uint64_t pin = mem.alloc(groups * 512), pout = mem.alloc(groups * 512);
  const amd::Kernel* k = amd::find_kernel(o, "counted_copy");
  if (!k) throw vtest::Failure("no kernel counted_copy");
  std::vector<uint8_t> buf(k->kernarg_size, 0);
  for (int b = 0; b < 8; ++b) buf[k->args[0].offset + b] = static_cast<uint8_t>(pin >> (8 * b)), buf[k->args[1].offset + b] = static_cast<uint8_t>(pout >> (8 * b));
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(buf.size());
  mem.write(d.kernarg, buf.data(), buf.size());
  d.groups[0] = groups;
  d.group_size[0] = group_size;
  d.count_lines = count;
  return amd::execute(d, mem).counts;
}

}  // namespace

VTEST(a_wave_counts_the_distinct_lines_its_lanes_touch) {
  for (const uint32_t line : {64u, 128u}) {
    const bool narrow = line == 64;
    VCHECK_EQ(lines(0, 4, 4, kAll, line), narrow ? 4u : 2u);        // unit stride, wave64: 256 bytes
    VCHECK_EQ(lines(0, 128, 4, kAll, line), 64u);                    // stride 128 bytes: a line a lane
    VCHECK_EQ(lines(0, 64, 4, kAll, line), narrow ? 64u : 32u);      // stride 64 bytes
    VCHECK_EQ(lines(0, 16, 16, kAll, line), narrow ? 16u : 8u);      // 16 bytes a lane, 1024 bytes
    VCHECK_EQ(lines(0x1000, 0, 4, kAll, line), 1u);                  // atomics (or loads) all on one address
    VCHECK_EQ(lines(0, 4, 4, 0xFFFFF, line), narrow ? 2u : 1u);      // 20 lanes: 80 bytes
    VCHECK_EQ(lines(60, 0, 8, 1, line), narrow ? 2u : 1u);           // one lane across a line boundary at 64
    VCHECK_EQ(lines(0, 4, 4, 0, line), 0u);                          // no lane
  }
  uint64_t scattered[64];
  for (uint32_t l = 0; l < 64; ++l) scattered[l] = (l % 5) * 200;    // 0, 200, 400, 600, 800
  VCHECK_EQ(cache_lines(scattered, kAll, 4, 64), 5u);                // lines 0, 3, 6, 9, 12
  VCHECK_EQ(cache_lines(scattered, kAll, 4, 128), 5u);               // lines 0, 1, 3, 4, 6
}

VTEST(each_target_has_the_documented_line_size) {
  for (const Target t : {Target::Gfx90a, Target::Gfx942, Target::Gfx950}) VCHECK_EQ(amd::gcn::vl1d_line_bytes(t), 64u);
  for (const Target t : {Target::Gfx1030, Target::Gfx1100, Target::Gfx1200, Target::Gfx1250}) VCHECK_EQ(amd::gcn::vl1d_line_bytes(t), 128u);
}

VTEST(the_executor_counts_lines_of_a_unit_stride_copy) {
  // gfx942, 64-byte lines. 64 lanes of a dword: 256 bytes, 4 lines each way.
  const amd::InstructionCounts one = copy(1, 64, true);
  VCHECK_EQ(one.vmem_lines_rd, 4u);
  VCHECK_EQ(one.vmem_lines_wr, 4u);
  // 20 lanes: 80 bytes, 2 lines.
  VCHECK_EQ(copy(1, 20, true).vmem_lines_rd, 2u);
  // 2 groups of 100 work-items: each group's waves of 64 and 36 lanes (4 lines and 3), the second group 512 bytes on.
  const amd::InstructionCounts two = copy(2, 100, true);
  VCHECK_EQ(two.vmem_lines_rd, 14u);
  VCHECK_EQ(two.vmem_lines_wr, 14u);
}

VTEST(counting_is_off_unless_asked_for_and_changes_no_other_count) {
  const amd::InstructionCounts on = copy(2, 100, true), off = copy(2, 100, false);
  VCHECK_EQ(off.vmem_lines_rd + off.vmem_lines_wr, 0u);
  VCHECK_EQ(on.vmem_bytes_rd, off.vmem_bytes_rd);
  VCHECK_EQ(on.vmem_bytes_wr, off.vmem_bytes_wr);
  VCHECK_EQ(on.vmem, off.vmem);
  VCHECK_EQ(on.valu, off.valu);
}

VTEST(lines_add_up_and_scale_like_the_other_counts) {
  amd::DispatchStats a, b;
  a.counts.vmem_lines_rd = 3, a.counts.vmem_lines_wr = 5;
  b.counts.vmem_lines_rd = 100, b.counts.vmem_lines_wr = 200;
  a.add(b);
  VCHECK_EQ(a.counts.vmem_lines_rd, 103u);
  VCHECK_EQ(a.counts.vmem_lines_wr, 205u);
  a.scale(2);
  VCHECK_EQ(a.counts.vmem_lines_rd, 206u);
}

VTEST_MAIN
