// The CTA's sixteen barriers (PTX ISA 9.7.15.1): bar.sync with a barrier
// number and a thread count, and bar.arrive, which warp-specialized kernels
// use to hand work between producer and consumer warps -- CUTLASS's Hopper
// GEMMs and convolutions among them, which would not load without these.
#include <cstring>
#include <string>
#include <vector>

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

void run(const std::string& body, uint32_t threads, std::vector<std::vector<uint8_t>> args, MemoryManager& mem) {
  DeviceProfile prof = load_gpu("nvidia/a100");
  auto m = ptx::parse(std::string(kHeader) + body);
  LaunchConfig c;
  c.block = {threads, 1, 1};
  exec::launch(m.entries[0], c, args, mem, prof);
}

std::string parse_error(const std::string& line) {
  try {
    ptx::parse(std::string(kHeader) + ".visible .entry k()\n{\n    .reg .b32 %r<4>;\n    " + line +
               "\n    ret;\n}\n");
  } catch (const Error& e) {
    return e.message();
  }
  return "";
}

}  // namespace

// Two pairs of warps, each synchronizing on its own barrier with a count of
// 64 threads, while the other pair is still busy: warp 0 writes, warp 1 reads
// after barrier 1; warp 2 writes, warp 3 reads after barrier 2. Warps 2 and
// 3 loop first, so they are nowhere near barrier 1 when it completes.
VTEST(counted_barriers_synchronize_only_their_warps) {
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(128 * 4);
  run(R"(
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<4>;
    .shared .align 4 .b32 buf[128];
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, %tid.x;
    shr.u32 %r2, %r1, 5;          // warp
    and.b32 %r3, %r1, 31;         // lane
    shr.u32 %r4, %r2, 1;          // pair: 0 or 1
    and.b32 %r5, %r2, 1;          // 0 writes, 1 reads
    add.u32 %r6, %r4, 1;          // barrier 1 or 2
    setp.eq.u32 %p1, %r4, 1;
    @!%p1 bra WORK;
    mov.u32 %r7, 0;
SPIN:
    add.u32 %r7, %r7, 1;
    setp.lt.u32 %p2, %r7, 500;
    @%p2 bra SPIN;
WORK:
    shl.b32 %r8, %r4, 5;
    add.u32 %r8, %r8, %r3;        // slot: pair * 32 + lane
    shl.b32 %r9, %r8, 2;
    mov.u32 %r10, buf;
    add.u32 %r10, %r10, %r9;
    setp.eq.u32 %p3, %r5, 0;
    @!%p3 bra READ;
    mad.lo.u32 %r11, %r4, 1000, %r3;
    st.shared.u32 [%r10], %r11;
    bar.sync %r6, 64;
    bra DONE;
READ:
    bar.sync %r6, 64;
    ld.shared.u32 %r12, [%r10];
    mul.wide.u32 %rd2, %r8, 4;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r12;
DONE:
    bar.sync 0;
    ret;
}
)", 128, {arg_u64(out)}, mem);
  for (uint32_t pair = 0; pair < 2; ++pair)
    for (uint32_t lane = 0; lane < 32; ++lane)
      VCHECK_EQ(mem.load_scalar(out + (pair * 32 + lane) * 4, 4), uint64_t{pair * 1000 + lane});
}

// Producer and consumer, as the ISA describes them: the producer announces a
// filled buffer with bar.arrive and goes on, the consumer waits with bar.sync;
// then the roles reverse on a second barrier for the buffer's return. Three
// rounds, so each barrier completes and is reused.
VTEST(arrive_and_sync_hand_a_buffer_back_and_forth) {
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(3 * 32 * 4);
  run(R"(
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<4>;
    .shared .align 4 .b32 buf[32];
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, %tid.x;
    shr.u32 %r2, %r1, 5;
    and.b32 %r3, %r1, 31;
    shl.b32 %r4, %r3, 2;
    mov.u32 %r5, buf;
    add.u32 %r5, %r5, %r4;
    mov.u32 %r6, 0;               // round
    setp.eq.u32 %p1, %r2, 0;
    @!%p1 bra CONSUMER;
PRODUCER:
    setp.eq.u32 %p2, %r6, 0;
    @!%p2 bar.sync 4, 64;         // wait for the buffer back (not before the first round)
    mad.lo.u32 %r7, %r6, 100, %r3;
    st.shared.u32 [%r5], %r7;
    bar.arrive 3, 64;             // filled
    add.u32 %r6, %r6, 1;
    setp.lt.u32 %p3, %r6, 3;
    @%p3 bra PRODUCER;
    ret;
CONSUMER:
    bar.sync 3, 64;               // wait for it to be filled
    ld.shared.u32 %r8, [%r5];
    mad.lo.u32 %r9, %r6, 32, %r3;
    mul.wide.u32 %rd2, %r9, 4;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r8;
    add.u32 %r6, %r6, 1;
    setp.lt.u32 %p3, %r6, 3;
    @!%p3 bra END;
    bar.arrive 4, 64;             // hand it back
    bra CONSUMER;
END:
    ret;
}
)", 64, {arg_u64(out)}, mem);
  for (uint32_t round = 0; round < 3; ++round)
    for (uint32_t lane = 0; lane < 32; ++lane)
      VCHECK_EQ(mem.load_scalar(out + (round * 32 + lane) * 4, 4), uint64_t{round * 100 + lane});
}

// A barrier other than 0 without a count is the whole CTA, like bar.sync 0.
VTEST(an_uncounted_barrier_is_the_whole_cta) {
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(96 * 4);
  run(R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    .shared .align 4 .b32 buf[96];
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, %tid.x;
    shl.b32 %r2, %r1, 2;
    mov.u32 %r3, buf;
    add.u32 %r4, %r3, %r2;
    st.shared.u32 [%r4], %r1;
    barrier.sync.aligned 7;
    sub.u32 %r5, 95, %r1;
    shl.b32 %r5, %r5, 2;
    add.u32 %r6, %r3, %r5;
    ld.shared.u32 %r7, [%r6];
    mul.wide.u32 %rd2, %r1, 4;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r7;
    ret;
}
)", 96, {arg_u64(out)}, mem);
  for (uint32_t t = 0; t < 96; ++t) VCHECK_EQ(mem.load_scalar(out + t * 4, 4), uint64_t{95 - t});
}

// A barrier whose count can no longer be reached is a hang on hardware; here
// it is an error that says which barrier and how far it got, where before the
// block quietly ended with the warp still waiting.
VTEST(a_barrier_that_cannot_complete_is_reported) {
  MemoryManager mem{1 << 20};
  auto err = VCAPTURE(Error, run(R"(
.visible .entry k()
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    mov.u32 %r1, %tid.x;
    setp.ge.u32 %p1, %r1, 32;
    @%p1 bra GONE;
    bar.sync 5, 64;
GONE:
    ret;
}
)", 64, {}, mem));
  VCHECK_CONTAINS(err.message(), "deadlock");
  VCHECK_CONTAINS(err.message(), "barrier 5: 1 warp(s) waiting, 32 thread(s) arrived");
}

VTEST(barrier_operands_are_checked) {
  MemoryManager mem{1 << 20};
  auto err = VCAPTURE(Error, run(".visible .entry k()\n{\n    bar.sync 1, 48;\n    ret;\n}\n", 64, {}, mem));
  VCHECK_CONTAINS(err.message(), "multiple of the warp size");
  VCHECK_CONTAINS(parse_error("bar.sync 16;"), "barriers 0 to 15");
  VCHECK_CONTAINS(parse_error("bar.arrive 1;"), "bar.arrive needs a thread count");
}

VTEST_MAIN
