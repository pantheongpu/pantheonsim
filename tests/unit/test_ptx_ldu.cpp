// ldu: a load from an address that is the same for every thread of the warp.
// It reads what ld.global reads; the uniformity is a promise, not a different
// access. Only the .global state space has it, so any other is refused by name.
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
}  // namespace

// The kernel of the report: load one word, add one, store it next door.
VTEST(ldu_global_u32_reads_like_ld) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 po){
 .reg .b64 %rd<3>; .reg .b32 %r<3>;
 ld.param.u64 %rd1,[po]; cvta.to.global.u64 %rd1,%rd1;
 ldu.global.u32 %r1,[%rd1];
 add.u32 %r1,%r1,1;
 st.global.u32 [%rd1+4],%r1; ret; }
)";
  Env e;
  auto m = ptx::parse(ptx);
  const uint64_t buf = e.mem.alloc(8);
  e.mem.store_scalar(buf, 4, 41);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(buf)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(buf + 4, 4), 42ull);
}

// Every lane of a warp reads the same word, a 64-bit one, and a vector one.
VTEST(ldu_global_wide_and_vector_forms_across_a_warp) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 po, .param .u64 pi){
 .reg .b64 %rd<8>; .reg .b32 %r<8>;
 ld.param.u64 %rd1,[po]; ld.param.u64 %rd2,[pi];
 cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
 ldu.global.u64 %rd3,[%rd2];
 ldu.global.v2.u32 {%r1,%r2},[%rd2+8];
 mov.u32 %r3,%tid.x;
 mul.wide.u32 %rd4,%r3,16;
 add.u64 %rd5,%rd1,%rd4;
 st.global.u64 [%rd5],%rd3;
 st.global.v2.u32 [%rd5+8],{%r1,%r2};
 ret; }
)";
  Env e;
  auto m = ptx::parse(ptx);
  const uint64_t out = e.mem.alloc(32 * 16);
  const uint64_t in = e.mem.alloc(16);
  e.mem.store_scalar(in, 8, 0x1122334455667788ull);
  e.mem.store_scalar(in + 8, 4, 7);
  e.mem.store_scalar(in + 12, 4, 9);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out), arg_u64(in)}, e.mem, e.prof);
  for (uint32_t t = 0; t < 32; ++t) {
    VCHECK_EQ(e.mem.load_scalar(out + t * 16, 8), 0x1122334455667788ull);
    VCHECK_EQ(e.mem.load_scalar(out + t * 16 + 8, 4), 7ull);
    VCHECK_EQ(e.mem.load_scalar(out + t * 16 + 12, 4), 9ull);
  }
}

// ldu has only the .global space: another one is refused, and says so.
VTEST(ldu_outside_global_is_refused_by_name) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 po){
 .reg .b64 %rd<3>; .reg .b32 %r<3>;
 ld.param.u64 %rd1,[po];
 ldu.shared.u32 %r1,[%rd1];
 ret; }
)";
  const auto err = VCAPTURE(vgpu::Error, ptx::parse(ptx));
  VCHECK_CONTAINS(err.what(), "ldu");
}

VTEST_MAIN
