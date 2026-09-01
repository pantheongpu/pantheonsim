// Regression tests for defects found by the differential and fuzz passes.
// Each one crashed, hung, leaked, or silently produced wrong answers before.
#include <cstring>
#include <vector>

#include "fatbin.hpp"
#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

namespace {
const char* kHeader = ".version 8.3\n.target sm_86\n.address_size 64\n";
std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}
struct Env {
  MemoryManager mem{1 << 22};
  DeviceProfile prof = load_gpu("nvidia/a10");
};

// Builds a fatbin container with caller-chosen (possibly malformed) fields.
std::vector<uint8_t> fatbin(uint64_t container_size, uint32_t entry_hdr, uint64_t padded,
                            uint32_t payload_size) {
  std::vector<uint8_t> b(4096, 0);
  uint32_t magic = 0xBA55ED50;
  std::memcpy(b.data(), &magic, 4);
  uint16_t ver = 1, hsz = 16;
  std::memcpy(b.data() + 4, &ver, 2);
  std::memcpy(b.data() + 6, &hsz, 2);
  std::memcpy(b.data() + 8, &container_size, 8);
  uint16_t kind = 1, ever = 257;
  std::memcpy(b.data() + 16, &kind, 2);
  std::memcpy(b.data() + 18, &ever, 2);
  std::memcpy(b.data() + 20, &entry_hdr, 4);
  std::memcpy(b.data() + 24, &padded, 8);
  std::memcpy(b.data() + 32, &payload_size, 4);
  return b;
}
}  // namespace

VTEST(malformed_fatbin_fails_cleanly) {
  // These used to hang or read far out of bounds. The loader APIs hand us a
  // bare pointer with no length, so every offset must be self-consistent.
  struct Case { const char* what; std::vector<uint8_t> data; };
  std::vector<Case> cases = {
      {"container larger than any real image", fatbin(0xFFFFFFFF, 80, 16, 16)},
      {"payload_size beyond the container", fatbin(4000, 80, 16, 0x7FFFFFFF)},
      {"zero header and payload (would not advance)", fatbin(4000, 0, 0, 0)},
      {"padded payload overflowing the image", fatbin(4000, 80, 0xFFFFFFFFFFFFull, 8)},
  };
  for (auto& c : cases) {
    auto err = VCAPTURE(Error, vgpu::cuda::extract_ptx(c.data.data()));
    VCHECK(err.code() == Err::InvalidValue);
  }
  // A wrapper with a NULL inner pointer must be rejected, not dereferenced.
  std::vector<uint8_t> wrapper(32, 0);
  uint32_t wmagic = 0x466243B1;
  std::memcpy(wrapper.data(), &wmagic, 4);
  auto err = VCAPTURE(Error, vgpu::cuda::extract_ptx(wrapper.data()));
  VCHECK(err.code() == Err::InvalidValue);
}

VTEST(freed_allocation_tracking_is_bounded) {
  // Every free used to retain a record forever, so alloc/free loops grew
  // without limit. The quarantine keeps recent frees diagnosable but bounded.
  MemoryManager mm(1ull << 40);
  for (int i = 0; i < 50000; ++i) {
    uint64_t p = mm.alloc(4096);
    mm.free(p);
  }
  VCHECK_EQ(mm.used(), 0ull);
  VCHECK_EQ(mm.live_allocations(), size_t{0});

  // A recent free is still reported precisely as a double free...
  uint64_t recent = mm.alloc(64);
  mm.free(recent);
  auto err = VCAPTURE(Error, mm.free(recent));
  VCHECK(err.code() == Err::DoubleFree);

  // ...and an address retired long ago is still an error, not silence.
  uint8_t byte = 0;
  auto stale = VCAPTURE(Error, mm.read(kDeviceVaBase, &byte, 1));
  VCHECK(stale.code() == Err::UseAfterFree);
  VCHECK_CONTAINS(stale.what(), "retired");
}

VTEST(barrier_after_divergent_region) {
  // The reconvergence model used to reject any bar.sync reached while another
  // path was parked, which is the shape of nearly every real reduction:
  //   if (cond) { ... }  __syncthreads();
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .shared .align 4 .b8 buf[512];
    .reg .pred %p<3>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    mov.u64 %rd3, buf;
    mul.wide.u32 %rd4, %r1, 4;
    add.s64 %rd5, %rd3, %rd4;
    // Divergent region: only even lanes write.
    and.b32 %r2, %r1, 1;
    setp.ne.s32 %p1, %r2, 0;
    @%p1 bra SKIP;
    st.shared.u32 [%rd5], %r1;
    bra JOIN;
SKIP:
    mov.u32 %r3, 0;
    st.shared.u32 [%rd5], %r3;
JOIN:
    bar.sync 0;
    // Lane 0 sums the block: 0+2+4+...+62 for 64 threads.
    setp.ne.s32 %p2, %r1, 0;
    @%p2 bra DONE;
    mov.u32 %r4, 0;
    mov.u32 %r5, 0;
    mov.u32 %r6, %ntid.x;
LOOP:
    setp.ge.u32 %p1, %r5, %r6;
    @%p1 bra STORE;
    mul.wide.u32 %rd6, %r5, 4;
    add.s64 %rd7, %rd3, %rd6;
    ld.shared.u32 %r7, [%rd7];
    add.s32 %r4, %r4, %r7;
    add.s32 %r5, %r5, 1;
    bra LOOP;
STORE:
    st.global.u32 [%rd2], %r4;
DONE:
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(4);
  LaunchConfig cfg;
  cfg.block = {64, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  uint32_t expect = 0;
  for (uint32_t i = 0; i < 64; i += 2) expect += i;
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{expect});
}

VTEST(mul_hi_matches_hardware) {
  // Compilers emit mul.hi to turn division by a constant into a multiply, so
  // rejecting it blocked ordinary kernels. Values cross-checked on a real GPU.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, -1840700269;
    mov.u32 %r2, 7;
    mul.hi.s32 %r3, %r2, %r1;
    st.global.u32 [%rd2], %r3;
    mov.u32 %r4, -1;
    mul.hi.u32 %r5, %r4, %r4;
    st.global.u32 [%rd2+4], %r5;
    mov.u64 %rd3, -1;
    mul.hi.u64 %rd4, %rd3, %rd3;
    st.global.u64 [%rd2+8], %rd4;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(16);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out)}, e.mem, e.prof);
  // 7 * -1840700269 = -12884901883; high 32 bits of the 64-bit product = -3.
  VCHECK_EQ(static_cast<int32_t>(e.mem.load_scalar(out, 4)), -3);
  // 0xFFFFFFFF^2 = 0xFFFFFFFE00000001; high half = 0xFFFFFFFE.
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), 0xFFFFFFFEull);
  VCHECK_EQ(e.mem.load_scalar(out + 8, 8), 0xFFFFFFFFFFFFFFFEull);
}

VTEST(variable_named_directly_as_an_address) {
  // PTX may address a variable by name, without materializing a pointer first.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .shared .align 4 .b8 tally[64];
    .reg .b32 %r<5>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, 41;
    st.shared.u32 [tally+8], %r1;
    ld.shared.u32 %r2, [tally+8];
    add.s32 %r3, %r2, 1;
    st.global.u32 [%rd2], %r3;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(4);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), 42ull);
}

VTEST_MAIN
