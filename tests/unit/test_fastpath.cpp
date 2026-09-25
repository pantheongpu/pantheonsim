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

VTEST_MAIN
