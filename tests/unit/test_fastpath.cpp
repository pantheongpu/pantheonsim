// The interpreter's fast paths against its general path.
//
// The fast paths (Interpreter::fast_path) are the same arithmetic written a
// second time, for speed: 32-bit integer ops, mad.lo, mov, setp, f32 arithmetic
// and f32/f64 fma. A second copy of arithmetic is a second place for it to be
// wrong, so every form they take is run here twice -- once through them and
// once with VGPU_FASTPATH=0, which sends everything down the general path --
// over the values that break arithmetic: zero of both signs, the extremes,
// NaNs quiet and signalling, infinities, subnormals, and shift counts past the
// width. The two runs must agree to the bit, in a warp with half its lanes
// predicated off and in one with all of them on, and with a destination that
// is also a source.
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
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

constexpr int kThreads = 64;   // warp 0 half predicated off, warp 1 all on

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

uint32_t fbits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
uint64_t dbits(double d) {
  uint64_t u;
  std::memcpy(&u, &d, 8);
  return u;
}

std::vector<uint64_t> int_values() {
  const uint32_t v[] = {0, 1, 2, 3, 7, 31, 32, 33, 40, 0xFFFFFFFFu, 0xFFFFFFFEu, 0x7FFFFFFFu,
                        0x80000000u, 0x80000001u, 0xFFFFu, 0x10000u, 0xDEADBEEFu, 1103515245u,
                        12345u, 0xAAAAAAAAu, 0x55555555u, 100, 0xFFFFFF9Cu /* -100 */};
  std::vector<uint64_t> out;
  for (int i = 0; i < kThreads; ++i) out.push_back(v[(i * 7 + i / 23) % (sizeof v / sizeof v[0])]);
  return out;
}

std::vector<uint64_t> f32_values() {
  const float inf = std::numeric_limits<float>::infinity();
  const uint32_t v[] = {fbits(0.0f), fbits(-0.0f), fbits(1.0f), fbits(-1.0f), fbits(1.5f), fbits(3.1415926f),
                        fbits(inf), fbits(-inf), 0x7FC00000u, 0x7F800001u /* sNaN */, 0xFFC00001u,
                        1u /* smallest subnormal */, 0x807FFFFFu, fbits(std::numeric_limits<float>::max()),
                        fbits(std::numeric_limits<float>::min()), fbits(1e-30f), fbits(-7.25f),
                        fbits(65504.0f), fbits(0.1f), fbits(3.0f)};
  std::vector<uint64_t> out;
  for (int i = 0; i < kThreads; ++i) out.push_back(v[(i * 5 + i / 20) % (sizeof v / sizeof v[0])]);
  return out;
}

std::vector<uint64_t> f64_values() {
  const double inf = std::numeric_limits<double>::infinity();
  const uint64_t v[] = {dbits(0.0), dbits(-0.0), dbits(1.0), dbits(-2.5), dbits(inf), dbits(-inf),
                        0x7FF8000000000000ull, 0x7FF0000000000001ull, 1ull,
                        dbits(std::numeric_limits<double>::max()), dbits(1e-300), dbits(0.1),
                        dbits(1.0 + 1e-15), dbits(-1e308), dbits(3.0)};
  std::vector<uint64_t> out;
  for (int i = 0; i < kThreads; ++i) out.push_back(v[(i * 3 + i / 15) % (sizeof v / sizeof v[0])]);
  return out;
}

// Rotations of one set, so a, b and c are different values in each thread.
std::vector<uint64_t> rotate(const std::vector<uint64_t>& v, int by) {
  std::vector<uint64_t> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = v[(i + by) % v.size()];
  return out;
}

// Runs `body` -- instructions reading %a, %b, %c and writing %d -- in every
// thread, guarded so warp 0's odd lanes skip it, and returns each thread's %d.
// `wide` selects 64-bit registers (%fda.. for f64); otherwise 32-bit.
std::vector<uint64_t> run(const std::string& body, bool wide, const std::vector<uint64_t>& a,
                          const std::vector<uint64_t>& b, const std::vector<uint64_t>& c) {
  const std::string R = wide ? "rd" : "r";
  std::string ptx = R"(.version 8.3
.target sm_86
.address_size 64
.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u64 po)
{
    .reg .pred %p<8>;
    .reg .pred %q;
    .reg .b32 %r<16>;
    .reg .b16 %rs<4>;
    .reg .b64 %rd<24>;
    .reg .b32 %a, %b, %c, %d;
    .reg .b64 %A, %B, %C, %D;
    ld.param.u64 %rd1, [pa];
    ld.param.u64 %rd2, [pb];
    ld.param.u64 %rd3, [pc];
    ld.param.u64 %rd4, [po];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd5, %r1, 8;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.u64 %A, [%rd6];
    add.u64 %rd6, %rd2, %rd5;
    ld.global.u64 %B, [%rd6];
    add.u64 %rd6, %rd3, %rd5;
    ld.global.u64 %C, [%rd6];
    cvt.u32.u64 %a, %A;
    cvt.u32.u64 %b, %B;
    cvt.u32.u64 %c, %C;
    mov.u64 %D, 0x5A5A5A5A5A5A5A5A;
    mov.u32 %d, 0xA5A5A5A5;
    and.b32 %r2, %r1, 1;
    setp.eq.u32 %p1, %r2, 0;
    setp.ge.u32 %p2, %r1, 32;
    or.pred %q, %p1, %p2;
)";
  // Every instruction of the body runs under the guard.
  size_t pos = 0;
  while (pos < body.size()) {
    size_t end = body.find(';', pos);
    std::string one = body.substr(pos, end - pos + 1);
    while (!one.empty() && (one[0] == ' ' || one[0] == '\n')) one.erase(0, 1);
    if (!one.empty()) ptx += "    @%q " + one + "\n";
    pos = end + 1;
  }
  ptx += wide ? "    mov.b64 %rd8, %D;\n" : "    cvt.u64.u32 %rd8, %d;\n";
  ptx += R"(    add.u64 %rd9, %rd4, %rd5;
    st.global.u64 [%rd9], %rd8;
    ret;
}
)";
  (void)R;
  MemoryManager mem{1 << 20};
  DeviceProfile prof = load_gpu("nvidia/a10");
  auto m = ptx::parse(ptx);
  auto put = [&](const std::vector<uint64_t>& v) {
    const uint64_t va = mem.alloc(v.size() * 8);
    mem.write(va, v.data(), v.size() * 8);
    return va;
  };
  const uint64_t pa = put(a), pb = put(b), pc = put(c), po = mem.alloc(kThreads * 8);
  LaunchConfig cfg;
  cfg.block = {kThreads, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(pa), arg_u64(pb), arg_u64(pc), arg_u64(po)}, mem, prof);
  std::vector<uint64_t> out(kThreads);
  mem.read(po, out.data(), out.size() * 8);
  return out;
}

// Both ways, and the answers must be the same bits.
void same(const std::string& body, bool wide, const std::vector<uint64_t>& a,
          const std::vector<uint64_t>& b, const std::vector<uint64_t>& c) {
  setenv("VGPU_FASTPATH", "1", 1);
  const auto fast = run(body, wide, a, b, c);
  setenv("VGPU_FASTPATH", "0", 1);
  const auto slow = run(body, wide, a, b, c);
  unsetenv("VGPU_FASTPATH");
  for (int i = 0; i < kThreads; ++i)
    if (fast[i] != slow[i]) {
      char buf[256];
      std::snprintf(buf, sizeof buf, "'%s' thread %d: fast 0x%llx, general 0x%llx (a=0x%llx b=0x%llx c=0x%llx)",
                    body.c_str(), i, (unsigned long long)fast[i], (unsigned long long)slow[i],
                    (unsigned long long)a[i], (unsigned long long)b[i], (unsigned long long)c[i]);
      throw vtest::Failure(buf);
    }
}

}  // namespace

VTEST(fast_int_arithmetic_matches_the_general_path) {
  const auto a = int_values(), b = rotate(a, 5), c = rotate(a, 11);
  for (const char* ty : {"s32", "u32"})
    for (const char* op : {"add", "sub", "mul.lo", "min", "max"}) {
      const std::string o = std::string(op) + "." + ty;
      same(o + " %d, %a, %b;", false, a, b, c);
      same(o + " %d, %a, -7;", false, a, b, c);
      same(o + " %a, %a, %b; mov.b32 %d, %a;", false, a, b, c);   // destination is a source
    }
  for (const char* op : {"and.b32", "or.b32", "xor.b32"}) {
    same(std::string(op) + " %d, %a, %b;", false, a, b, c);
    same(std::string(op) + " %d, %a, -559038737;", false, a, b, c);
  }
  // Shift counts from the table include 31, 32, 33 and 40: past the width.
  for (const char* op : {"shl.b32", "shr.b32", "shr.u32", "shr.s32"}) {
    same(std::string(op) + " %d, %a, %b;", false, a, b, c);
    same(std::string(op) + " %d, %a, 31;", false, a, b, c);
    same(std::string(op) + " %d, %a, 32;", false, a, b, c);
  }
  // Forms the fast path declines -- division, 16- and 64-bit -- still agree.
  same("div.s32 %d, %a, %b;", false, a, b, c);
  same("rem.u32 %d, %a, %b;", false, a, b, c);
}

VTEST(fast_mad_mov_and_setp_match_the_general_path) {
  const auto a = int_values(), b = rotate(a, 3), c = rotate(a, 9);
  same("mad.lo.s32 %d, %a, %b, %c;", false, a, b, c);
  same("mad.lo.u32 %d, %a, 1103515245, 12345;", false, a, b, c);
  same("mov.u32 %d, %a;", false, a, b, c);
  same("mov.b32 %d, -1431655766;", false, a, b, c);
  for (const char* ty : {"s32", "u32", "b32"})
    for (const char* cmp : {"eq", "ne", "lt", "le", "gt", "ge"}) {
      if (std::string(ty) == "b32" && std::string(cmp) != "eq" && std::string(cmp) != "ne") continue;
      same(std::string("setp.") + cmp + "." + ty + " %p3, %a, %b; selp.u32 %d, 1, 2, %p3;", false, a,
           b, c);
      same(std::string("setp.") + cmp + "." + ty + " %p3, %a, 100; selp.u32 %d, 1, 2, %p3;", false,
           a, b, c);
    }
  const auto f = f32_values(), g = rotate(f, 7);
  for (const char* cmp : {"eq", "ne", "lt", "le", "gt", "ge", "equ", "neu", "ltu", "leu", "gtu",
                          "geu", "num", "nan"})
    same(std::string("setp.") + cmp + ".f32 %p3, %a, %b; selp.u32 %d, 1, 2, %p3;", false, f, g, c);
}

VTEST(fast_float_arithmetic_matches_the_general_path) {
  const auto a = f32_values(), b = rotate(a, 3), c = rotate(a, 8);
  for (const char* op : {"add.f32", "sub.f32", "mul.f32", "div.rn.f32", "min.f32", "max.f32",
                         "add.rn.f32", "mul.rz.f32", "min.NaN.f32"})
    same(std::string(op) + " %d, %a, %b;", false, a, b, c);
  same("fma.rn.f32 %d, %a, %b, %c;", false, a, b, c);
  same("fma.rn.f32 %d, %a, 0f3F800000, %c;", false, a, b, c);
  same("fma.rn.f32 %a, %a, %b, %a; mov.b32 %d, %a;", false, a, b, c);
  const auto x = f64_values(), y = rotate(x, 4), z = rotate(x, 9);
  same("fma.rn.f64 %D, %A, %B, %C;", true, x, y, z);
  same("fma.rn.f64 %D, %A, 0d3FF8000000000000, %C;", true, x, y, z);
  same("fma.rn.f64 %A, %A, %B, %A; mov.b64 %D, %A;", true, x, y, z);
}

// Half precision: the decode table and the integer rounding of results,
// over values whose f16 results land everywhere -- normal, subnormal,
// overflowing to infinity, rounding up into the next exponent, and NaN.
VTEST(fast_half_precision_matches_the_general_path) {
  const uint16_t h[] = {0x0000, 0x8000, 0x3C00, 0xBC00, 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0x0001,
                        0x03FF, 0x0400, 0x7BFF, 0x3555, 0x4248, 0x5A00, 0xC248, 0x3BFF, 0x0401,
                        0x1400, 0x77FF, 0x6BFF, 0x2E66};
  std::vector<uint64_t> a;
  for (int i = 0; i < kThreads; ++i) {
    const uint16_t lo = h[(i * 7) % (sizeof h / 2)], hi = h[(i * 5 + 3) % (sizeof h / 2)];
    a.push_back(uint64_t{lo} | (uint64_t{hi} << 16));
  }
  const auto b = rotate(a, 5), c = rotate(a, 13);
  for (const char* op : {"fma.rn.f16x2 %d, %a, %b, %c;", "fma.rn.bf16x2 %d, %a, %b, %c;",
                         "add.f16x2 %d, %a, %b;", "mul.f16x2 %d, %a, %b;", "sub.rn.f16x2 %d, %a, %b;",
                         "add.rz.f16x2 %d, %a, %b;", "mul.rm.f16x2 %d, %a, %b;"})
    same(op, false, a, b, c);
  // Conversions into f16 from f32 values whose halves over- and underflow.
  const auto f = f32_values();
  for (const char* op : {"cvt.rn.f16.f32 %rs1, %a; cvt.u32.u16 %d, %rs1;",
                         "cvt.rz.f16.f32 %rs1, %a; cvt.u32.u16 %d, %rs1;",
                         "cvt.f32.f16 %d, %a;"})
    same(op, false, f, b, c);
}

// Rounding a double to f16, on the values where rounding is decided: exactly
// halfway between two halves (ties to even, both parities), a hair either side
// of halfway, the top of each binade where rounding carries into the
// exponent, the overflow threshold, the smallest normals, and random bits.
VTEST(fast_f16_rounding_matches_the_general_path_at_every_tie) {
  std::vector<uint64_t> vals;
  uint64_t seed = 0x9E3779B97F4A7C15ull;
  auto rnd = [&] { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
  for (int e = -16; e <= 16; ++e)
    for (uint64_t mant10 : {0ull, 1ull, 2ull, 511ull, 1022ull, 1023ull}) {
      const uint64_t base = (uint64_t(e + 1023) << 52) | (mant10 << 42);
      for (uint64_t low : {0ull, 1ull << 41, (1ull << 41) - 1, (1ull << 41) + 1, (1ull << 42) - 1})
        for (uint64_t sign : {0ull, 1ull << 63}) vals.push_back(sign | base | low);
    }
  for (double d : {65504.0, 65519.999, 65520.0, 65536.0, 6.103515625e-05, 6.0975e-05, 5.96e-08})
    vals.push_back(dbits(d));
  while (vals.size() % kThreads) vals.push_back(dbits(double(int64_t(rnd() % 2000001) - 1000000) / 777.0));
  for (int i = 0; i < 64; ++i) vals.push_back(rnd());
  while (vals.size() % kThreads) vals.push_back(rnd());
  for (size_t at = 0; at < vals.size(); at += kThreads) {
    const std::vector<uint64_t> a(vals.begin() + at, vals.begin() + at + kThreads);
    same("cvt.rn.f16.f64 %rs1, %A; cvt.u32.u16 %d, %rs1;", false, a, a, a);
  }
}

// 64-bit integer arithmetic, integer conversions of every width pairing,
// and mul.wide: the address arithmetic around memory accesses.
VTEST(fast_address_arithmetic_matches_the_general_path) {
  std::vector<uint64_t> a;
  const uint64_t v[] = {0, 1, 63, 64, 65, 100, ~0ull, 0x8000000000000000ull, 0x7FFFFFFFFFFFFFFFull,
                        0xFFFFFFFFull, 0x80000000ull, 0x7FFFFFFFull, 0x123456789ABCDEF0ull,
                        0xFFFFFFFF80000000ull, 0x10000ull, 0xFFFFull, 0x8000ull, 0x80ull, 32};
  for (int i = 0; i < kThreads; ++i) a.push_back(v[(i * 7 + i / 19) % (sizeof v / sizeof v[0])]);
  const auto b = rotate(a, 3), c = rotate(a, 10);
  for (const char* op : {"add.s64", "sub.s64", "mul.lo.s64", "and.b64", "or.b64", "xor.b64",
                         "min.s64", "max.s64", "min.u64", "max.u64", "shl.b64", "shr.u64", "shr.s64"}) {
    same(std::string(op) + " %D, %A, %B;", true, a, b, c);
    same(std::string(op) + " %A, %A, 5; mov.b64 %D, %A;", true, a, b, c);
  }
  same("shl.b64 %D, %A, %b;", true, a, b, c);   // a 32-bit shift count
  same("div.u64 %D, %A, %B;", true, a, b, c);   // declined: general path both times
  for (const char* cvt : {"cvt.u64.u32 %D, %a;", "cvt.s64.s32 %D, %a;", "cvt.u64.s32 %D, %a;",
                          "cvt.s64.u32 %D, %a;", "cvt.u32.u64 %d, %A;", "cvt.s32.s64 %d, %A;",
                          "cvt.u16.u32 %rs1, %a; cvt.u32.u16 %d, %rs1;",
                          "cvt.u16.u32 %rs1, %a; cvt.s32.s16 %d, %rs1;",
                          "cvt.u16.u32 %rs1, %a; cvt.s64.s16 %D, %rs1;",
                          "cvt.u8.u32 %rs1, %a; cvt.s32.s8 %d, %rs1;"})
    same(cvt, std::string(cvt).find("%D") != std::string::npos, a, b, c);
  same("mul.wide.u32 %D, %a, %b;", true, a, b, c);
  same("mul.wide.s32 %D, %a, %b;", true, a, b, c);
  same("mul.wide.s32 %D, %a, -3;", true, a, b, c);
}

// wmma.mma from registers loaded with values that make every rounding in the
// f32 accumulation matter -- thirds and tenths, large and small, signed zeros,
// NaN and infinity -- in f16 and bf16, all four layout pairings, and with the
// result written over the accumulator it read.
namespace {
std::vector<uint32_t> run_wmma(const std::string& shape, bool alias, const std::vector<uint32_t>& frag) {
  std::string dregs = alias ? "%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7"
                            : "%x0, %x1, %x2, %x3, %x4, %x5, %x6, %x7";
  std::string loads, stores;
  for (int r = 0; r < 8; ++r) {
    loads += "    ld.global.u32 %a" + std::to_string(r) + ", [%rd3+" + std::to_string(r * 4) + "];\n";
    loads += "    ld.global.u32 %b" + std::to_string(r) + ", [%rd3+" + std::to_string(32 + r * 4) + "];\n";
    loads += "    ld.global.u32 %c" + std::to_string(r) + ", [%rd3+" + std::to_string(64 + r * 4) + "];\n";
    stores += "    st.global.u32 [%rd4+" + std::to_string(r * 4) + "], " + (alias ? "%c" : "%x") +
              std::to_string(r) + ";\n";
  }
  const bool bf = shape.find("bf16") != std::string::npos;
  const std::string areg = bf ? "{%a0, %a1, %a2, %a3}" : "{%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7}";
  const std::string breg = bf ? "{%b0, %b1, %b2, %b3}" : "{%b0, %b1, %b2, %b3, %b4, %b5, %b6, %b7}";
  const std::string ptx = std::string(".version 8.3\n.target sm_86\n.address_size 64\n") + R"(
.visible .entry k(.param .u64 in, .param .u64 out)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<8>;
    .reg .b32 %a<8>, %b<8>, %c<8>, %x<8>;
    ld.param.u64 %rd1, [in];
    ld.param.u64 %rd2, [out];
    mov.u32 %r1, %laneid;
    mul.wide.u32 %rd5, %r1, 96;
    add.u64 %rd3, %rd1, %rd5;
    mul.wide.u32 %rd6, %r1, 32;
    add.u64 %rd4, %rd2, %rd6;
)" + loads + "    wmma.mma.sync.aligned." + shape + " {" + dregs + "}, " + areg + ", " + breg +
                          ", {%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7};\n" + stores + "    ret;\n}\n";
  MemoryManager mem{1 << 20};
  DeviceProfile prof = load_gpu("nvidia/a10");
  auto m = ptx::parse(ptx);
  const uint64_t in = mem.alloc(frag.size() * 4), out = mem.alloc(32 * 32);
  mem.write(in, frag.data(), frag.size() * 4);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(in), arg_u64(out)}, mem, prof);
  std::vector<uint32_t> res(32 * 8);
  mem.read(out, res.data(), res.size() * 4);
  return res;
}
}  // namespace

VTEST(fast_wmma_matches_the_general_path) {
  const uint16_t h16[] = {0x3555, 0x2E66, 0xBC00, 0x3C00, 0x7BFF, 0x0001, 0x8000, 0x4248, 0xC0A0,
                          0x1400, 0x5A00, 0x3800, 0x7C00, 0x7E00, 0x0400, 0xB555};
  const uint16_t hbf[] = {0x3EAB, 0x3DCD, 0xBF80, 0x3F80, 0x7F7F, 0x0001, 0x8000, 0x4049, 0xC0A0,
                          0x3A00, 0x4700, 0x3F00, 0x7F80, 0x7FC0, 0x0080, 0xBEAB};
  const float cv[] = {0.0f, -0.0f, 1.0f / 3, 1e-8f, -2.5f, 1e30f, 0.1f, 7.0f};
  for (bool bf : {false, true}) {
    std::vector<uint32_t> frag;
    for (int lane = 0; lane < 32; ++lane) {
      for (int r = 0; r < 16; ++r) {   // 8 A words then 8 B words
        const uint16_t* t = bf ? hbf : h16;
        const uint16_t lo = t[(lane * 3 + r * 5) % 16], hi = t[(lane * 7 + r * 3 + 1) % 16];
        frag.push_back(uint32_t{lo} | (uint32_t{hi} << 16));
      }
      for (int r = 0; r < 8; ++r) frag.push_back(fbits(cv[(lane + r) % 8] * float(1 + lane % 5)));
    }
    for (const char* lay : {"row.row", "row.col", "col.row", "col.col"})
      for (bool alias : {false, true}) {
        const std::string shape = std::string(lay) + ".m16n16k16.f32" + (bf ? ".bf16.bf16" : ".f32");
        setenv("VGPU_FASTPATH", "1", 1);
        const auto fast = run_wmma(shape, alias, frag);
        setenv("VGPU_FASTPATH", "0", 1);
        const auto slow = run_wmma(shape, alias, frag);
        unsetenv("VGPU_FASTPATH");
        for (size_t i = 0; i < fast.size(); ++i)
          if (fast[i] != slow[i])
            throw vtest::Failure("wmma " + shape + (alias ? " (D over C)" : "") + ": word " +
                                 std::to_string(i) + " differs");
      }
  }
}

// ---------------------------------------------------------------------------
// Memory accesses: the counters and the data, fast against general.
//
// A warp's load or store is counted from its lanes' addresses (sectors for
// global and local, bank passes for shared), and the shared case is read and
// written without building the general path's bookkeeping. Both are the same
// answer by a different route, so every pattern below runs twice -- fast, and
// with VGPU_FASTPATH=0 -- and the loaded values, the memory left behind and
// every counter must agree.
// ---------------------------------------------------------------------------
namespace {

struct Access {
  const char* space;   // "shared" or "global"
  const char* ty;      // "u8", "u16", "u32", "u64"
  int bytes;           // the type's size
  int vec;             // 1, 2 or 4 elements
};

struct MemRun {
  std::vector<uint32_t> out;      // what each thread loaded, folded, then the buffer
  exec::LaunchStats stats;
  std::string error;              // empty when the launch succeeded
};

// One kernel: thread t touches element (t * stride + shift) mod N of a 16 KiB
// region of `space`, loading (or storing) `vec` elements of `ty` there, with
// every lane or only the even ones. Loads fold what they read into out[t];
// the buffer is dumped to out afterwards either way, so stores are seen.
MemRun run_access(const Access& a, int stride, int shift, bool even_only, bool store,
                  bool is_shared) {
  const int width = a.bytes * a.vec;                // bytes one lane moves
  const int nelem = 16384 / width;                  // elements in the region
  std::string regs;
  for (int i = 0; i < a.vec; ++i) regs += std::string(i ? ", " : "") + "%v" + std::to_string(i);
  const std::string vecq = a.vec == 1 ? "" : (a.vec == 2 ? ".v2" : ".v4");
  const std::string tyq = std::string(a.ty);
  const std::string wide = a.bytes == 8 ? "b64" : "b32";
  std::string ptx = ".version 8.3\n.target sm_86\n.address_size 64\n";
  ptx += ".visible .entry k(.param .u64 pbuf, .param .u64 pout)\n{\n";
  ptx += "    .reg .pred %p<4>;\n    .reg .b32 %r<24>;\n    .reg .b64 %rd<16>;\n";
  ptx += "    .reg .b32 %v0, %v1, %v2, %v3;\n    .reg .b64 %w0, %w1, %w2, %w3;\n";
  if (is_shared) ptx += "    .shared .align 16 .b8 sm[16384];\n";
  ptx += "    ld.param.u64 %rd1, [pbuf];\n    ld.param.u64 %rd2, [pout];\n    mov.u32 %r1, %tid.x;\n";
  // Fill: thread t writes words t + 64 j, j < 64, with a value that names the word.
  if (is_shared) ptx += "    mov.u32 %r10, sm;\n";
  for (int j = 0; j < 64; ++j) {
    ptx += "    add.u32 %r5, %r1, " + std::to_string(64 * j) + ";\n";
    ptx += "    mul.lo.u32 %r6, %r5, 2654435761;\n    xor.b32 %r6, %r6, 305419896;\n";
    if (is_shared) {
      ptx += "    shl.b32 %r7, %r5, 2;\n    add.u32 %r7, %r7, %r10;\n    st.shared.u32 [%r7], %r6;\n";
    } else {
      ptx += "    mul.wide.u32 %rd3, %r5, 4;\n    add.u64 %rd3, %rd1, %rd3;\n    st.global.u32 [%rd3], %r6;\n";
    }
  }
  if (is_shared) ptx += "    bar.sync 0;\n";
  // The element this thread touches.
  ptx += "    mul.lo.u32 %r11, %r1, " + std::to_string(stride) + ";\n";
  ptx += "    add.u32 %r11, %r11, " + std::to_string(shift) + ";\n";
  ptx += "    and.b32 %r11, %r11, " + std::to_string(nelem - 1) + ";\n";
  ptx += "    mul.lo.u32 %r12, %r11, " + std::to_string(width) + ";\n";
  if (is_shared) ptx += "    add.u32 %r12, %r12, %r10;\n";
  else ptx += "    cvt.u64.u32 %rd4, %r12;\n    add.u64 %rd4, %rd1, %rd4;\n";
  ptx += "    and.b32 %r13, %r1, 1;\n    setp.eq.u32 %p1, %r13, 0;\n";
  const std::string guard = even_only ? "@%p1 " : "";
  const std::string addr = is_shared ? "[%r12]" : "[%rd4]";
  const std::string sp = is_shared ? "shared" : "global";
  // Each element is a 32- or 64-bit register; narrower types load into 32-bit ones.
  std::string list;
  for (int i = 0; i < a.vec; ++i)
    list += std::string(i ? ", " : "") + (a.bytes == 8 ? "%w" : "%v") + std::to_string(i);
  const std::string ldty = a.bytes == 8 ? "u64" : (a.bytes == 4 ? "u32" : (a.bytes == 2 ? "u16" : "u8"));
  const std::string brace_list = a.vec == 1 ? list : "{" + list + "}";
  if (!store) {
    ptx += "    mov.b32 %r20, 0;\n";
    for (int i = 0; i < a.vec; ++i) ptx += std::string("    mov.b32 %v") + std::to_string(i) + ", 0;\n    mov.b64 %w" + std::to_string(i) + ", 0;\n";
    ptx += "    " + guard + "ld." + sp + vecq + "." + ldty + " " + brace_list + ", " + addr + ";\n";
    for (int i = 0; i < a.vec; ++i) {
      if (a.bytes == 8) {
        ptx += "    cvt.u32.u64 %r21, %w" + std::to_string(i) + ";\n    xor.b32 %r20, %r20, %r21;\n";
        ptx += "    shr.u64 %rd5, %w" + std::to_string(i) + ", 32;\n    cvt.u32.u64 %r21, %rd5;\n    add.u32 %r20, %r20, %r21;\n";
      } else {
        ptx += "    add.u32 %r20, %r20, %v" + std::to_string(i) + ";\n    xor.b32 %r20, %r20, " + std::to_string(i + 1) + ";\n";
      }
    }
  } else {
    for (int i = 0; i < a.vec; ++i) {
      if (a.bytes == 8)
        ptx += "    mul.wide.u32 %w" + std::to_string(i) + ", %r1, " + std::to_string(0x01000193 + 2 * i) + ";\n";
      else
        ptx += "    mad.lo.u32 %v" + std::to_string(i) + ", %r1, " + std::to_string(0x01000193 + 2 * i) + ", " + std::to_string(i + 7) + ";\n";
    }
    ptx += "    " + guard + "st." + sp + vecq + "." + ldty + " " + addr + ", " + brace_list + ";\n";
    ptx += "    mov.b32 %r20, 0;\n";
  }
  if (is_shared) ptx += "    bar.sync 0;\n";
  // Out: out[t] = folded value; then the region, 64 words a thread.
  ptx += "    mul.wide.u32 %rd6, %r1, 4;\n    add.u64 %rd7, %rd2, %rd6;\n    st.global.u32 [%rd7], %r20;\n";
  for (int j = 0; j < 64; ++j) {
    ptx += "    add.u32 %r5, %r1, " + std::to_string(64 * j) + ";\n";
    if (is_shared) {
      ptx += "    shl.b32 %r7, %r5, 2;\n    add.u32 %r7, %r7, %r10;\n    ld.shared.u32 %r8, [%r7];\n";
    } else {
      ptx += "    mul.wide.u32 %rd3, %r5, 4;\n    add.u64 %rd3, %rd1, %rd3;\n    ld.global.u32 %r8, [%rd3];\n";
    }
    ptx += "    add.u32 %r9, %r5, 64;\n    mul.wide.u32 %rd8, %r9, 4;\n    add.u64 %rd8, %rd2, %rd8;\n    st.global.u32 [%rd8], %r8;\n";
  }
  ptx += "    ret;\n}\n";

  MemRun r;
  MemoryManager mem{1 << 22};
  DeviceProfile prof = load_gpu("nvidia/a10");
  auto m = ptx::parse(ptx);
  const uint64_t buf = mem.alloc(16384), out = mem.alloc((64 + 4096 + 64) * 4);
  std::vector<uint8_t> zero(16384, 0);
  mem.write(buf, zero.data(), zero.size());
  LaunchConfig cfg;
  cfg.block = {64, 1, 1};
  try {
    r.stats = exec::launch(m.entries[0], cfg, {arg_u64(buf), arg_u64(out)}, mem, prof);
  } catch (const std::exception& e) {
    r.error = e.what();
  }
  r.out.resize(64 + 4096 + 64);
  mem.read(out, r.out.data(), r.out.size() * 4);
  return r;
}

bool same_stats(const exec::LaunchStats& a, const exec::LaunchStats& b) {
  return std::memcmp(&a, &b, exec::kCounterWords * sizeof(uint64_t)) == 0 &&
         a.inst_by_opcode == b.inst_by_opcode;
}

}  // namespace

VTEST(memory_access_counters_and_data_match_the_general_path) {
  const Access accesses[] = {{"u8", "u8", 1, 1},   {"u16", "u16", 2, 1},  {"u32", "u32", 4, 1},
                             {"u64", "u64", 8, 1}, {"v2.u32", "u32", 4, 2}, {"v4.u32", "u32", 4, 4},
                             {"v2.u64", "u64", 8, 2}};
  const int strides[] = {0, 1, 2, 3, 4, 5, 8, 16, 17, 32, 33, 64};
  int runs = 0;
  for (bool is_shared : {true, false})
    for (const Access& a : accesses)
      for (int stride : strides)
        for (bool even : {false, true})
          for (bool store : {false, true}) {
            const int shift = stride == 3 ? 5 : 0;
            setenv("VGPU_FASTPATH", "1", 1);
            const MemRun fast = run_access(a, stride, shift, even, store, is_shared);
            setenv("VGPU_FASTPATH", "0", 1);
            const MemRun slow = run_access(a, stride, shift, even, store, is_shared);
            unsetenv("VGPU_FASTPATH");
            const std::string what = std::string(is_shared ? "shared " : "global ") + (store ? "st." : "ld.") +
                                     a.space + " stride " + std::to_string(stride) + (even ? " even lanes" : " all lanes");
            if (fast.error != slow.error)
              throw vtest::Failure(what + ": fast error '" + fast.error + "', general '" + slow.error + "'");
            if (fast.out != slow.out) throw vtest::Failure(what + ": data differs");
            if (!same_stats(fast.stats, slow.stats)) throw vtest::Failure(what + ": counters differ");
            ++runs;
          }
  VCHECK(runs > 300);
}

// The counters are not just equal to each other: shared bank passes and global
// sectors come out at the figures the hardware's definition gives.
VTEST(access_counters_have_the_documented_values) {
  setenv("VGPU_FASTPATH", "1", 1);
  // 4-byte shared loads, stride 1: conflict-free. Stride 2: two lanes a bank in
  // each half-warp's worth -> 2 passes. Stride 32: all 32 lanes of a warp in
  // one bank on different words -> 32 passes (31 conflicts). Stride 0: one word
  // for all lanes, a broadcast, no conflict.
  const Access u32{"u32", "u32", 4, 1};
  auto conflicts = [&](int stride) {
    const MemRun r = run_access(u32, stride, 0, false, false, true);
    return r.stats.shared_bank_conflicts_ld;
  };
  const uint64_t per_warp_fill = 0;   // the fill and dump are conflict-free stores and loads
  (void)per_warp_fill;
  const uint64_t c1 = conflicts(1), c2 = conflicts(2), c32 = conflicts(32), c0 = conflicts(0);
  unsetenv("VGPU_FASTPATH");
  VCHECK(c2 > c1);      // stride 2 conflicts, stride 1 does not
  VCHECK(c32 > c2);     // stride 32 conflicts the most
  VCHECK(c0 == c1);     // a broadcast costs what a conflict-free access does
  // Two warps; stride 32 is 31 conflicts each over the stride-1 baseline.
  VCHECK_EQ(c32 - c1, 2u * 31u);
}

// ld.param by position gives what ld.param by name does -- on a kernel whose
// parameters differ in size and alignment.
VTEST(ld_param_by_position_matches_by_name) {
  const std::string ptx = R"(.version 8.3
.target sm_86
.address_size 64
.visible .entry k(.param .u8 a, .param .u64 out, .param .u32 b, .param .u16 c, .param .u64 d)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .reg .b16 %rs<4>;
    ld.param.u8 %r1, [a];
    ld.param.u32 %r2, [b];
    ld.param.u16 %r3, [c];
    ld.param.u64 %rd2, [d];
    ld.param.u64 %rd1, [out];
    add.u32 %r4, %r1, %r2;
    add.u32 %r4, %r4, %r3;
    cvt.u64.u32 %rd3, %r4;
    add.u64 %rd3, %rd3, %rd2;
    st.global.u64 [%rd1], %rd3;
    ret;
}
)";
  auto run_once = [&](const char* fp) {
    setenv("VGPU_FASTPATH", fp, 1);
    MemoryManager mem{1 << 20};
    DeviceProfile prof = load_gpu("nvidia/a10");
    auto m = ptx::parse(ptx);
    const uint64_t out = mem.alloc(8);
    LaunchConfig cfg;
    cfg.block = {32, 1, 1};
    const std::vector<std::vector<uint8_t>> args = {{200}, arg_u64(out), {0x10, 0x20, 0x30, 0x40},
                                                    {0x00, 0x01}, arg_u64(0x1122334455ull)};
    exec::launch(m.entries[0], cfg, args, mem, prof);
    uint64_t v = 0;
    mem.read(out, &v, 8);
    unsetenv("VGPU_FASTPATH");
    return v;
  };
  const uint64_t expect = (200u + 0x40302010u + 0x100u) + 0x1122334455ull;
  VCHECK_EQ(run_once("1"), expect);
  VCHECK_EQ(run_once("0"), expect);
}

// A load or store the general path refuses is refused with the same words by
// the fast one: out of bounds, misaligned, and a register nothing wrote.
VTEST(refused_accesses_give_the_same_error) {
  auto run_bad = [&](const std::string& body, const char* fp) {
    setenv("VGPU_FASTPATH", fp, 1);
    const std::string ptx = ".version 8.3\n.target sm_86\n.address_size 64\n"
        ".visible .entry k(.param .u64 pout)\n{\n    .reg .b32 %r<8>;\n    .reg .b64 %rd<8>;\n"
        "    .shared .align 16 .b8 sm[256];\n    mov.u32 %r1, %tid.x;\n    mov.u32 %r2, sm;\n" + body + "    ret;\n}\n";
    MemoryManager mem{1 << 20};
    DeviceProfile prof = load_gpu("nvidia/a10");
    auto m = ptx::parse(ptx);
    const uint64_t out = mem.alloc(64);
    LaunchConfig cfg;
    cfg.block = {32, 1, 1};
    std::string err;
    try {
      exec::launch(m.entries[0], cfg, {arg_u64(out)}, mem, prof);
    } catch (const std::exception& e) {
      err = e.what();
    }
    unsetenv("VGPU_FASTPATH");
    return err;
  };
  const char* bodies[] = {
      // past the end of the 256-byte allocation, in the last lane only
      "    shl.b32 %r3, %r1, 3;\n    add.u32 %r3, %r3, %r2;\n    ld.shared.u32 %r4, [%r3+8];\n",
      "    shl.b32 %r3, %r1, 3;\n    add.u32 %r3, %r3, %r2;\n    st.shared.u32 [%r3+8], %r1;\n",
      // a vector access that is not aligned to its width
      "    shl.b32 %r3, %r1, 4;\n    add.u32 %r3, %r3, %r2;\n    ld.shared.v4.u32 {%r4, %r5, %r6, %r7}, [%r3+4];\n",
      "    shl.b32 %r3, %r1, 4;\n    add.u32 %r3, %r3, %r2;\n    st.shared.v2.u32 [%r3+4], {%r1, %r1};\n",
      // a store through an address in no allocation
      "    mov.u64 %rd2, 16;\n    st.global.u32 [%rd2], %r1;\n",
  };
  for (const char* body : bodies) {
    const std::string fast = run_bad(body, "1"), slow = run_bad(body, "0");
    VCHECK(!slow.empty());
    VCHECK_EQ(fast, slow);
  }
}

// Thread ids come out as (lin % nx, lin / nx % ny, lin / (nx * ny)) for the
// block shapes that wrap in x, in y, in both, and at a warp boundary.
VTEST(thread_ids_are_the_linear_index_split_by_the_block_shape) {
  const std::string ptx = R"(.version 8.3
.target sm_86
.address_size 64
.visible .entry k(.param .u64 po)
{
    .reg .b32 %r<12>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [po];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %tid.y;
    mov.u32 %r3, %tid.z;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %ntid.y;
    mad.lo.u32 %r6, %r3, %r5, %r2;
    mad.lo.u32 %r6, %r6, %r4, %r1;
    mul.wide.u32 %rd2, %r6, 12;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r1;
    st.global.u32 [%rd3+4], %r2;
    st.global.u32 [%rd3+8], %r3;
    ret;
}
)";
  const uint32_t shapes[][3] = {{105, 1, 1}, {5, 3, 7}, {33, 2, 1}, {1, 1, 64}, {7, 1, 9},
                                {32, 4, 2}, {1, 33, 1}, {3, 11, 3}, {64, 1, 1}, {2, 2, 17}};
  for (const auto& sh : shapes) {
    MemoryManager mem{1 << 20};
    DeviceProfile prof = load_gpu("nvidia/a10");
    auto m = ptx::parse(ptx);
    const uint32_t n = sh[0] * sh[1] * sh[2];
    const uint64_t out = mem.alloc(n * 12);
    LaunchConfig cfg;
    cfg.block = {sh[0], sh[1], sh[2]};
    exec::launch(m.entries[0], cfg, {arg_u64(out)}, mem, prof);
    std::vector<uint32_t> got(n * 3);
    mem.read(out, got.data(), got.size() * 4);
    for (uint32_t lin = 0; lin < n; ++lin) {
      VCHECK_EQ(got[lin * 3], lin % sh[0]);
      VCHECK_EQ(got[lin * 3 + 1], lin / sh[0] % sh[1]);
      VCHECK_EQ(got[lin * 3 + 2], lin / (sh[0] * sh[1]));
    }
  }
}

VTEST_MAIN
