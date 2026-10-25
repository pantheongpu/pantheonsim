// The PTX interpreter's loads and stores of a run of lanes (coalesced global accesses moved with MemoryManager::load_run and
// store_run, consecutive shared-memory accesses moved as one block) against its general path.
//
// Every shape of access that kernels make -- a full warp, a tail, a run with a gap, a reversed or strided one, one byte to
// sixteen, scalar and vector, global and shared, aligned and not, inside a chunk and across a chunk boundary, off the end of an
// allocation -- is run twice, once with the fast paths and once with VGPU_FASTPATH=0, which sends every access down the lane
// loop. The two must leave the same bytes in memory, the same values in the registers (dumped to memory), the same counters
// (sectors, requests, bank conflicts, every field of the launch's statistics) and, where the access faults, the same error.
#include <cstdio>
#include <cstdlib>
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

constexpr int kThreads = 64;                 // two warps
constexpr uint32_t kBuf = 3 * static_cast<uint32_t>(kChunkSize);    // the buffers: three chunks, so a run can cross a boundary

struct Shape {
  uint32_t off = 0;      // byte offset of lane 0's access in the buffer
  uint32_t stride = 4;   // bytes between consecutive lanes' accesses
  uint32_t lo = 0, hi = kThreads;   // the lanes that make the access: lo <= lane < hi ...
  uint32_t skip = 0;                // ... except the lanes whose bit of this is set (of the first 32)
};

struct Outcome {
  std::vector<uint8_t> in, out, regs;
  std::vector<uint64_t> counters;
  std::string error;   // empty when the launch ran
};

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}
std::vector<uint8_t> arg_u32(uint32_t v) {
  std::vector<uint8_t> b(4);
  std::memcpy(b.data(), &v, 4);
  return b;
}

// The kernel: %rd6 is lane's address in `in`, %rd7 in `out`, %r3 in shared memory `sm` (all at off + lane * stride); %p1 is the
// lanes that make the access. `body` runs, then registers %r10-%r13, %rd10-%rd11 and %h1 are dumped per lane.
std::string kernel(const std::string& body) {
  return R"(.version 8.3
.target sm_86
.address_size 64
.visible .entry k(.param .u64 pin, .param .u64 pout, .param .u64 pres, .param .u32 poff, .param .u32 pstride,
                  .param .u32 plo, .param .u32 phi, .param .u32 pskip)
{
    .reg .pred %p<8>;
    .reg .b32 %r<32>;
    .reg .b16 %h<4>;
    .reg .b64 %rd<32>;
    .shared .align 16 .b8 sm[8192];
    ld.param.u64 %rd1, [pin];
    ld.param.u64 %rd2, [pout];
    ld.param.u64 %rd3, [pres];
    ld.param.u32 %r20, [poff];
    ld.param.u32 %r21, [pstride];
    ld.param.u32 %r22, [plo];
    ld.param.u32 %r23, [phi];
    ld.param.u32 %r24, [pskip];
    mov.u32 %r1, %tid.x;
    mad.lo.u32 %r2, %r1, %r21, %r20;
    cvt.u64.u32 %rd5, %r2;
    add.u64 %rd6, %rd1, %rd5;
    add.u64 %rd7, %rd2, %rd5;
    mov.u32 %r3, sm;
    add.u32 %r3, %r3, %r2;
    // shared memory starts as the input's first 8 KiB: four rounds of 32 bytes a thread
    shl.b32 %r4, %r1, 5;
    cvt.u64.u32 %rd8, %r4;
    add.u64 %rd9, %rd1, %rd8;
    mov.u32 %r5, sm;
    add.u32 %r5, %r5, %r4;
    mov.u32 %r30, 0;
$FILL:
    ld.global.u32 %r6, [%rd9];
    st.shared.u32 [%r5], %r6;
    ld.global.u32 %r6, [%rd9+4];
    st.shared.u32 [%r5+4], %r6;
    ld.global.u32 %r6, [%rd9+8];
    st.shared.u32 [%r5+8], %r6;
    ld.global.u32 %r6, [%rd9+12];
    st.shared.u32 [%r5+12], %r6;
    ld.global.u32 %r6, [%rd9+16];
    st.shared.u32 [%r5+16], %r6;
    ld.global.u32 %r6, [%rd9+20];
    st.shared.u32 [%r5+20], %r6;
    ld.global.u32 %r6, [%rd9+24];
    st.shared.u32 [%r5+24], %r6;
    ld.global.u32 %r6, [%rd9+28];
    st.shared.u32 [%r5+28], %r6;
    add.u64 %rd9, %rd9, 2048;
    add.u32 %r5, %r5, 2048;
    add.u32 %r30, %r30, 1;
    setp.lt.u32 %p7, %r30, 4;
    @%p7 bra $FILL;
    bar.sync 0;
    mov.u32 %r10, 0x11111111;
    mov.u32 %r11, 0x22222222;
    mov.u32 %r12, 0x33333333;
    mov.u32 %r13, 0x44444444;
    mov.u64 %rd10, 0x5555555555555555;
    mov.u64 %rd11, 0x6666666666666666;
    mov.b16 %h1, 0x7777;
    setp.ge.u32 %p1, %r1, %r22;
    setp.lt.u32 %p2, %r1, %r23;
    and.pred %p1, %p1, %p2;
    shr.u32 %r25, %r24, %r1;
    and.b32 %r25, %r25, 1;
    setp.eq.u32 %p3, %r25, 0;
    setp.lt.u32 %p4, %r1, 32;
    not.pred %p5, %p4;
    or.pred %p3, %p3, %p5;
    and.pred %p1, %p1, %p3;
)" + body + R"(
    bar.sync 0;
    shl.b32 %r26, %r1, 6;
    cvt.u64.u32 %rd12, %r26;
    add.u64 %rd13, %rd3, %rd12;
    st.global.u32 [%rd13], %r10;
    st.global.u32 [%rd13+4], %r11;
    st.global.u32 [%rd13+8], %r12;
    st.global.u32 [%rd13+12], %r13;
    st.global.u64 [%rd13+16], %rd10;
    st.global.u64 [%rd13+24], %rd11;
    st.global.b16 [%rd13+32], %h1;
    // the shared memory the body left, first 4 KiB
    shl.b32 %r27, %r1, 6;
    mov.u32 %r28, sm;
    add.u32 %r28, %r28, %r27;
    ld.shared.u32 %r29, [%r28];
    st.global.u32 [%rd13+36], %r29;
    ld.shared.u32 %r29, [%r28+32];
    st.global.u32 [%rd13+40], %r29;
    ld.shared.u32 %r29, [%r28+60];
    st.global.u32 [%rd13+44], %r29;
    ret;
}
)";
}

Outcome run(const std::string& body, const Shape& s, bool fast, bool uniform_out) {
  setenv("VGPU_FASTPATH", fast ? "1" : "0", 1);
  Outcome o;
  MemoryManager mem{1 << 20};
  DeviceProfile prof = load_gpu("nvidia/a10");
  auto m = ptx::parse(kernel(body));
  const uint64_t pin = mem.alloc(kBuf), pout = mem.alloc(kBuf), pres = mem.alloc(kThreads * 64);
  std::vector<uint8_t> data(kBuf);
  uint64_t x = 0x9E3779B97F4A7C15ull;
  for (size_t i = 0; i < data.size(); ++i) {
    x = x * 6364136223846793005ull + 1442695040888963407ull;
    data[i] = static_cast<uint8_t>(x >> 56);
  }
  mem.write(pin, data.data(), data.size());
  if (uniform_out) {
    const uint8_t pat = 0x5A;
    mem.fill(pout, &pat, 1, kBuf);
  }
  LaunchConfig cfg;
  cfg.block = {kThreads, 1, 1};
  try {
    exec::LaunchStats st = exec::launch(m.entries[0], cfg,
                                        {arg_u64(pin), arg_u64(pout), arg_u64(pres), arg_u32(s.off), arg_u32(s.stride),
                                         arg_u32(s.lo), arg_u32(s.hi), arg_u32(s.skip)},
                                        mem, prof);
    const auto* w = reinterpret_cast<const uint64_t*>(&st);
    o.counters.assign(w, w + exec::kCounterWords);
    o.counters.insert(o.counters.end(), st.inst_by_opcode.begin(), st.inst_by_opcode.end());
  } catch (const Error& e) {
    o.error = e.what();
  }
  unsetenv("VGPU_FASTPATH");
  o.in.resize(kBuf);
  o.out.resize(kBuf);
  o.regs.resize(kThreads * 64);
  mem.read(pin, o.in.data(), kBuf);
  mem.read(pout, o.out.data(), kBuf);
  mem.read(pres, o.regs.data(), o.regs.size());
  return o;
}

// How many launches ran to the end and how many faulted (the same way in both, but a test in which every launch faults checks little).
int g_ran = 0, g_faulted = 0;

// Both ways; everything the launch leaves behind must be the same.
void same(const char* what, const std::string& body, const Shape& s) {
  for (bool uniform_out : {false, true}) {
    const Outcome a = run(body, s, true, uniform_out), b = run(body, s, false, uniform_out);
    const std::string where = std::string(what) + " off=" + std::to_string(s.off) + " stride=" + std::to_string(s.stride) +
                              " lanes=[" + std::to_string(s.lo) + "," + std::to_string(s.hi) + ") skip=" +
                              std::to_string(s.skip) + (uniform_out ? " (out uniform)" : "");
    if (a.error != b.error) throw vtest::Failure(where + ": error differs\n  fast:    " + a.error + "\n  general: " + b.error);
    ++(a.error.empty() ? g_ran : g_faulted);
    if (a.in != b.in) throw vtest::Failure(where + ": the input buffer differs");
    if (a.out != b.out) throw vtest::Failure(where + ": the output buffer differs");
    if (a.regs != b.regs) throw vtest::Failure(where + ": the registers differ");
    if (a.counters != b.counters) {
      std::string d;
      for (size_t i = 0; i < a.counters.size() && i < b.counters.size(); ++i)
        if (a.counters[i] != b.counters[i])
          d += " [" + std::to_string(i) + "]: " + std::to_string(a.counters[i]) + " vs " + std::to_string(b.counters[i]);
      throw vtest::Failure(where + ": the counters differ:" + d);
    }
  }
}

// The shapes: a full warp, a tail, an interior run, a gap, a single lane, the second warp only, strides.
std::vector<Shape> shapes(uint32_t elem) {
  std::vector<Shape> v;
  const uint32_t e = elem;
  v.push_back({0, e, 0, kThreads, 0});                 // both warps full, consecutive
  v.push_back({64, e, 0, kThreads, 0});
  v.push_back({e, e, 0, kThreads, 0});                 // (misaligned for a vector wider than e)
  v.push_back({0, e, 5, 23, 0});                       // an interior run
  v.push_back({0, e, 0, 32, 0});
  v.push_back({0, e, 3, 4, 0});                        // one lane
  v.push_back({0, e, 32, 64, 0});                      // the second warp
  v.push_back({0, e, 0, kThreads, 0x00F0F000u});       // gaps
  v.push_back({0, e, 0, kThreads, 1u});                // lane 0 off
  v.push_back({0, e, 0, kThreads, 0x80000000u});       // lane 31 off
  v.push_back({0, 2 * e, 0, kThreads, 0});             // strided: not a run
  v.push_back({0, 0, 0, kThreads, 0});                 // every lane on one address
  v.push_back({0, e, 0, kThreads, 0});
  v.push_back({static_cast<uint32_t>(kChunkSize) - 8 * e, e, 0, kThreads, 0});            // a run across a chunk boundary
  v.push_back({static_cast<uint32_t>(kChunkSize) - 8 * e, e, 3, 20, 0});
  v.push_back({kBuf - 32 * e, e, 0, 32, 0});                       // ends exactly at the end of the buffer
  v.push_back({kBuf - 31 * e, e, 0, 32, 0});                       // runs one element off the end
  v.push_back({kBuf - 31 * e, e, 0, 31, 0});
  v.push_back({0, e, 0, kThreads, 0xAAAAAAAAu});                   // alternate lanes
  return v;
}

struct Access {
  const char* name;
  uint32_t elem;       // bytes of one lane's access (the stride of a run)
  const char* body;    // global
  const char* sbody;   // the same through shared memory
};

// In each, %rd6 / %rd7 / %r3 are the lane's addresses in `in`, `out` and shared memory; loaded values go to %r10-%r13,
// %rd10-%rd11 and %h1, and a store writes them (or their constants) to `out`.
const Access kAccesses[] = {
    {"u32", 4,
     "@%p1 ld.global.u32 %r10, [%rd6]; @%p1 st.global.u32 [%rd7], %r10;",
     "@%p1 ld.shared.u32 %r10, [%r3]; @%p1 st.shared.u32 [%r3+4096], %r10;"},
    {"f32 and offset", 4,
     "@%p1 ld.global.f32 %r10, [%rd6+8]; @%p1 st.global.f32 [%rd7+4], %r10;",
     "@%p1 ld.shared.f32 %r10, [%r3+8]; @%p1 st.shared.f32 [%r3+4100], %r10;"},
    {"u64", 8,
     "@%p1 ld.global.u64 %rd10, [%rd6]; @%p1 st.global.u64 [%rd7], %rd10;",
     "@%p1 ld.shared.u64 %rd10, [%r3]; @%p1 st.shared.u64 [%r3+2048], %rd10;"},
    {"u8", 1,
     "@%p1 ld.global.u8 %r10, [%rd6]; @%p1 st.global.u8 [%rd7], %r10;",
     "@%p1 ld.shared.u8 %r10, [%r3]; @%p1 st.shared.u8 [%r3+4096], %r10;"},
    {"s8 sign-extends", 1,
     "@%p1 ld.global.s8 %r10, [%rd6]; @%p1 st.global.u32 [%rd7], %r10;",
     "@%p1 ld.shared.s8 %r10, [%r3]; @%p1 st.shared.u32 [%r3+4096], %r10;"},
    {"u16", 2,
     "@%p1 ld.global.u16 %r10, [%rd6]; @%p1 st.global.u16 [%rd7], %r10;",
     "@%p1 ld.shared.u16 %r10, [%r3]; @%p1 st.shared.u16 [%r3+4096], %r10;"},
    {"s16 sign-extends", 2,
     "@%p1 ld.global.s16 %r10, [%rd6]; @%p1 st.global.u32 [%rd7], %r10;",
     "@%p1 ld.shared.s16 %r10, [%r3]; @%p1 st.shared.u32 [%r3+4096], %r10;"},
    {"b16 into a half register", 2,
     "@%p1 ld.global.b16 %h1, [%rd6]; @%p1 st.global.b16 [%rd7], %h1;",
     "@%p1 ld.shared.b16 %h1, [%r3]; @%p1 st.shared.b16 [%r3+4096], %h1;"},
    {"u32 into a 64-bit register", 4,
     "@%p1 ld.global.u32 %rd10, [%rd6]; @%p1 st.global.u64 [%rd7], %rd10;",
     "@%p1 ld.shared.u32 %rd10, [%r3]; @%p1 st.shared.u64 [%r3+2048], %rd10;"},
    {"s32 into a 64-bit register", 4,
     "@%p1 ld.global.s32 %rd10, [%rd6]; @%p1 st.global.u64 [%rd7], %rd10;",
     "@%p1 ld.shared.s32 %rd10, [%r3]; @%p1 st.shared.u64 [%r3+2048], %rd10;"},
    {"v2.u32", 8,
     "@%p1 ld.global.v2.u32 {%r10, %r11}, [%rd6]; @%p1 st.global.v2.u32 [%rd7], {%r11, %r10};",
     "@%p1 ld.shared.v2.u32 {%r10, %r11}, [%r3]; @%p1 st.shared.v2.u32 [%r3+2048], {%r11, %r10};"},
    {"v4.u32", 16,
     "@%p1 ld.global.v4.u32 {%r10, %r11, %r12, %r13}, [%rd6]; @%p1 st.global.v4.u32 [%rd7], {%r13, %r12, %r11, %r10};",
     "@%p1 ld.shared.v4.u32 {%r10, %r11, %r12, %r13}, [%r3]; @%p1 st.shared.v4.u32 [%r3+2048], {%r13, %r12, %r11, %r10};"},
    {"v2.u64", 16,
     "@%p1 ld.global.v2.u64 {%rd10, %rd11}, [%rd6]; @%p1 st.global.v2.u64 [%rd7], {%rd11, %rd10};",
     "@%p1 ld.shared.v2.u64 {%rd10, %rd11}, [%r3]; @%p1 st.shared.v2.u64 [%r3+2048], {%rd11, %rd10};"},
    {"v2.u16", 4,
     "@%p1 ld.global.v2.u16 {%r10, %r11}, [%rd6]; @%p1 st.global.v2.u16 [%rd7], {%r11, %r10};",
     "@%p1 ld.shared.v2.u16 {%r10, %r11}, [%r3]; @%p1 st.shared.v2.u16 [%r3+4096], {%r11, %r10};"},
    {"store of a constant and of a value nothing wrote", 4,
     "@%p1 st.global.u32 [%rd7], 305419896; @%p1 ld.global.u32 %r10, [%rd6]; @%p1 st.global.u32 [%rd7+4], %r11;",
     "@%p1 st.shared.u32 [%r3], 305419896; @%p1 ld.shared.u32 %r10, [%r3+4]; @%p1 st.shared.u32 [%r3+4096], %r11;"},
    {"store of the byte the buffer is filled with", 4,
     "@%p1 mov.u32 %r10, 0x5A5A5A5A; @%p1 st.global.u32 [%rd7], %r10; @%p1 ld.global.u32 %r11, [%rd7];",
     "@%p1 mov.u32 %r10, 0x5A5A5A5A; @%p1 st.shared.u32 [%r3], %r10; @%p1 ld.shared.u32 %r11, [%r3];"},
    {"generic address of global memory", 4,
     "@%p1 ld.u32 %r10, [%rd6]; @%p1 st.u32 [%rd7], %r10;",
     "@%p1 ld.u32 %r10, [%rd6+4]; @%p1 st.u32 [%rd7+4], %r10;"},
};

// Atomics: the lanes of a warp read-modify-write one after another, whichever words they name (one, a few, each its own).
// %r11 / %rd11 are the operand (the data pattern's bits for the lane), %r12 / %rd10 the compare and exchange values.
const Access kAtomics[] = {
    {"add.u32", 4, "@%p1 ld.global.u32 %r11, [%rd6]; @%p1 atom.global.add.u32 %r10, [%rd7], %r11;",
     "@%p1 ld.global.u32 %r11, [%rd6]; @%p1 atom.shared.add.u32 %r10, [%r3], %r11;"},
    {"red.add.u32", 4, "@%p1 ld.global.u32 %r11, [%rd6]; @%p1 red.global.add.u32 [%rd7], %r11;",
     "@%p1 ld.global.u32 %r11, [%rd6]; @%p1 red.shared.add.u32 [%r3], %r11;"},
    {"add.f32", 4, "@%p1 ld.global.f32 %r11, [%rd6]; @%p1 atom.global.add.f32 %r10, [%rd7], %r11;",
     "@%p1 ld.global.f32 %r11, [%rd6]; @%p1 atom.shared.add.f32 %r10, [%r3], %r11;"},
    {"add.u64", 8, "@%p1 ld.global.u64 %rd11, [%rd6]; @%p1 atom.global.add.u64 %rd10, [%rd7], %rd11;",
     "@%p1 ld.global.u64 %rd11, [%rd6]; @%p1 atom.shared.add.u64 %rd10, [%r3], %rd11;"},
    {"add.f64", 8, "@%p1 ld.global.f64 %rd11, [%rd6]; @%p1 atom.global.add.f64 %rd10, [%rd7], %rd11;",
     "@%p1 ld.global.f64 %rd11, [%rd6]; @%p1 atom.shared.add.f64 %rd10, [%r3], %rd11;"},
    {"min.s32", 4, "@%p1 ld.global.u32 %r11, [%rd6]; @%p1 atom.global.min.s32 %r10, [%rd7], %r11;",
     "@%p1 ld.global.u32 %r11, [%rd6]; @%p1 atom.shared.min.s32 %r10, [%r3], %r11;"},
    {"max.u32", 4, "@%p1 ld.global.u32 %r11, [%rd6]; @%p1 atom.global.max.u32 %r10, [%rd7], %r11;",
     "@%p1 ld.global.u32 %r11, [%rd6]; @%p1 atom.shared.max.u32 %r10, [%r3], %r11;"},
    {"max.s64", 8, "@%p1 ld.global.u64 %rd11, [%rd6]; @%p1 atom.global.max.s64 %rd10, [%rd7], %rd11;",
     "@%p1 ld.global.u64 %rd11, [%rd6]; @%p1 atom.shared.max.s64 %rd10, [%r3], %rd11;"},
    {"and/or/xor", 4,
     "@%p1 ld.global.u32 %r11, [%rd6]; @%p1 atom.global.and.b32 %r10, [%rd7], %r11; @%p1 atom.global.or.b32 %r12, [%rd7], %r11; "
     "@%p1 atom.global.xor.b32 %r13, [%rd7], %r11;",
     "@%p1 ld.global.u32 %r11, [%rd6]; @%p1 atom.shared.and.b32 %r10, [%r3], %r11; @%p1 atom.shared.or.b32 %r12, [%r3], %r11; "
     "@%p1 atom.shared.xor.b32 %r13, [%r3], %r11;"},
    {"exch.b32", 4, "@%p1 ld.global.u32 %r11, [%rd6]; @%p1 atom.global.exch.b32 %r10, [%rd7], %r11;",
     "@%p1 ld.global.u32 %r11, [%rd6]; @%p1 atom.shared.exch.b32 %r10, [%r3], %r11;"},
    {"cas.b32", 4,
     "@%p1 ld.global.u32 %r11, [%rd6]; @%p1 ld.global.u32 %r12, [%rd7]; @%p1 atom.global.cas.b32 %r10, [%rd7], %r12, %r11; "
     "@%p1 atom.global.cas.b32 %r13, [%rd7], %r11, %r12;",
     "@%p1 ld.global.u32 %r11, [%rd6]; @%p1 ld.shared.u32 %r12, [%r3]; @%p1 atom.shared.cas.b32 %r10, [%r3], %r12, %r11;"},
    {"inc and dec", 4,
     "@%p1 ld.global.u32 %r11, [%rd6]; and.b32 %r11, %r11, 15; @%p1 atom.global.inc.u32 %r10, [%rd7], %r11; "
     "@%p1 atom.global.dec.u32 %r12, [%rd7], %r11;",
     "@%p1 ld.global.u32 %r11, [%rd6]; and.b32 %r11, %r11, 15; @%p1 atom.shared.inc.u32 %r10, [%r3], %r11; "
     "@%p1 atom.shared.dec.u32 %r12, [%r3], %r11;"},
    {"add.noftz.f16x2", 4,
     "@%p1 ld.global.u32 %r11, [%rd6]; @%p1 atom.global.add.noftz.f16x2 %r10, [%rd7], %r11;",
     "@%p1 ld.global.u32 %r11, [%rd6]; @%p1 atom.shared.add.noftz.f16x2 %r10, [%r3], %r11;"},
    {"generic add.u32", 4, "@%p1 ld.global.u32 %r11, [%rd6]; @%p1 atom.add.u32 %r10, [%rd7], %r11;",
     "@%p1 ld.global.u32 %r11, [%rd6]; @%p1 atom.add.u32 %r10, [%rd7+4], %r11;"},
};

}  // namespace

VTEST(global_loads_and_stores_of_a_run_match_the_lane_loop) {
  for (const Access& a : kAccesses)
    for (const Shape& s : shapes(a.elem)) same(a.name, a.body, s);
  std::printf("  %d launches ran, %d faulted\n", g_ran, g_faulted);
  VCHECK(g_ran > 2 * g_faulted);
  g_ran = g_faulted = 0;
}

VTEST(shared_loads_and_stores_of_a_run_match_the_lane_loop) {
  for (const Access& a : kAccesses) {
    // Shared memory is 8 KiB, and the accesses (and the stores 2 or 4 KiB above) must fit or fault the same way.
    for (Shape s : shapes(a.elem)) {
      s.off %= 1000;
      same(a.name, a.sbody, s);
    }
    Shape off_end{8192 - 16 * a.elem, a.elem, 0, kThreads, 0};   // past the end of shared memory
    same(a.name, a.sbody, off_end);
  }
  std::printf("  %d launches ran, %d faulted\n", g_ran, g_faulted);
  VCHECK(g_ran > 2 * g_faulted);
  g_ran = g_faulted = 0;
}

VTEST(the_stats_of_a_run_are_the_same_both_ways_for_a_warp_of_narrow_registers_stored_and_reloaded) {
  // A loop's worth of runs: load a word, add to it, store it back one word further on.
  const std::string body = R"(
    mov.u32 %r14, 0;
$L:
    @%p1 ld.global.u32 %r10, [%rd6];
    add.u32 %r10, %r10, %r14;
    @%p1 st.global.u32 [%rd6+4], %r10;
    @%p1 ld.global.u32 %r11, [%rd6+4];
    @%p1 st.global.u32 [%rd7], %r11;
    add.u32 %r14, %r14, 1;
    setp.lt.u32 %p6, %r14, 4;
    @%p6 bra $L;
)";
  for (const Shape& s : shapes(4)) same("a loop over one run", body, s);
}

VTEST(atomics_of_a_warp_match_the_lane_loop) {
  for (const Access& a : kAtomics) {
    for (const Shape& s : shapes(a.elem)) same(a.name, a.body, s);
    // The words each lane names are few: every lane on one word, and two words.
    same(a.name, a.body, {0, 0, 0, kThreads, 0});
    same(a.name, a.body, {0, a.elem, 0, kThreads, 0xF0F0F0F0u});
    // A misaligned atomic faults where the lane loop faults.
    same(a.name, a.body, {2, a.elem, 0, kThreads, 0});
  }
  std::printf("  %d launches ran, %d faulted\n", g_ran, g_faulted);
  VCHECK(g_ran > 2 * g_faulted);
  g_ran = g_faulted = 0;
}

VTEST(atomics_on_shared_memory_match_the_lane_loop) {
  for (const Access& a : kAtomics)
    for (Shape s : shapes(a.elem)) {
      s.off %= 1000;
      same(a.name, a.sbody, s);
    }
  std::printf("  %d launches ran, %d faulted\n", g_ran, g_faulted);
  VCHECK(g_ran > 2 * g_faulted);
  g_ran = g_faulted = 0;
}

VTEST_MAIN
