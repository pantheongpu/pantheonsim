// A predicate source operand may be complemented in place: `and.pred %p2, !%p0, %p1;`
// and `xor.pred %p10, %p7, !%p9;` (Dr.Jit emits both for boolean operations). The
// complement is taken per lane, before the operation, on both execution paths
// (the decoded micro-op one and the general interpreter).
#include <cstring>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

namespace {
const char* kHeader = ".version 8.0\n.target sm_80\n.address_size 64\n";
std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}
struct Env {
  MemoryManager mem{1 << 20};
  DeviceProfile prof = load_gpu("nvidia/a100");
};

// bit0 of the result of `<line>` (it sets %p2) for lane tid, where %p0 = (tid & 1) and %p1 = (tid & 2).
uint32_t run_lane(const std::string& line, uint32_t tid) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 po){
 .reg .pred %p<4>; .reg .b32 %r<6>; .reg .b64 %rd<4>;
 ld.param.u64 %rd1,[po]; cvta.to.global.u64 %rd1,%rd1;
 mov.u32 %r1,%tid.x;
 and.b32 %r2,%r1,1; setp.ne.u32 %p0,%r2,0;
 and.b32 %r3,%r1,2; setp.ne.u32 %p1,%r3,0;
 )" + line + R"(
 selp.u32 %r4,1,0,%p2;
 mul.wide.u32 %rd2,%r1,4; add.u64 %rd3,%rd1,%rd2;
 st.global.u32 [%rd3],%r4; ret; }
)";
  Env e;
  auto m = ptx::parse(ptx);
  const uint64_t out = e.mem.alloc(32 * 4);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  return static_cast<uint32_t>(e.mem.load_scalar(out + tid * 4, 4));
}
}  // namespace

VTEST(negated_first_source_of_and_pred) {
  for (uint32_t t = 0; t < 4; ++t) {
    const bool p0 = t & 1, p1 = t & 2;
    VCHECK_EQ(run_lane("and.pred %p2, !%p0, %p1;", t), uint32_t(!p0 && p1));
  }
}

VTEST(negated_second_source_of_xor_pred) {
  for (uint32_t t = 0; t < 4; ++t) {
    const bool p0 = t & 1, p1 = t & 2;
    VCHECK_EQ(run_lane("xor.pred %p2, %p0, !%p1;", t), uint32_t(p0 != !p1));
  }
}

VTEST(both_sources_negated_and_or_pred) {
  for (uint32_t t = 0; t < 4; ++t) {
    const bool p0 = t & 1, p1 = t & 2;
    VCHECK_EQ(run_lane("or.pred %p2, !%p0, !%p1;", t), uint32_t(!p0 || !p1));
    VCHECK_EQ(run_lane("and.pred %p2, !%p0, !%p1;", t), uint32_t(!p0 && !p1));
  }
}

// The destination may be a negated source's own register, and a plain operand still means what it did.
VTEST(negated_source_in_place_and_plain_forms_unchanged) {
  for (uint32_t t = 0; t < 4; ++t) {
    const bool p0 = t & 1, p1 = t & 2;
    VCHECK_EQ(run_lane("and.pred %p2, %p0, %p1;", t), uint32_t(p0 && p1));
    VCHECK_EQ(run_lane("mov.pred %p2, %p0; and.pred %p2, !%p2, %p1;", t), uint32_t(!p0 && p1));
  }
}

VTEST(module_of_the_report_loads) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 p)
{
    .reg .pred %p<4>;
    mov.pred %p0, 1;
    mov.pred %p1, 0;
    and.pred %p2, !%p0, %p1;
    xor.pred %p3, %p2, !%p1;
    ret;
}
)";
  auto m = ptx::parse(ptx);
  VCHECK(m.entries.size() == 1);
}

VTEST_MAIN
