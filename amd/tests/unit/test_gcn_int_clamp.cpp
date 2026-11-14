// The clamp bit of the integer instructions. Clamped, an integer result saturates to the range of its type: the CDNA3 and CDNA4 guides
// (6.5: "For integer operations, it clamps the result to the largest and smallest representable value", and "Supports saturation" on
// each add and subtract, v_add_co_u32 and its kin among them), and an RX 6800 (gfx1030) running asm_int_clamp's kernel, agree. The carry
// out of the VOP3b instructions is the unclamped instruction's: the carry or borrow of an add or subtract, and for the 64-bit
// multiply-adds bit 64 of the 65-bit result (the guides' "{ D1.u1, D0.u64 } = 65'B(...)"; for v_mad_i64_i32 that is the sign of the
// exact result, not whether it overflowed). The executor dropped the clamp of the VOP3b instructions (the gfx9 decoder did not read the
// bit, and neither the interpreter nor the tier applied it), of the 24-bit multiplies and multiply-adds, and applied some others on gfx942
// and gfx950 only. The kernel is amd/tests/data/asm_int_clamp.s; the expected values are the guides' and the card's, not the simulator's.
// Run with the tier off and on, for gfx90a, gfx942 and gfx1030.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vgpu/amd_gcn.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

// --- what the instructions give (the model the card's results were checked against) ---
struct In {
  uint32_t a, b, c32, cin;
  uint64_t c64;
};
struct Want {
  uint32_t lo, hi, carry;
  bool low16;   // a 16-bit result: only the low half is compared
};
constexpr int kOps = 44;
constexpr int64_t kU32 = 0xFFFFFFFFll;
int64_t sat(int64_t v, int64_t lo, int64_t hi) { return std::clamp(v, lo, hi); }
int64_t s32(uint32_t v) { return static_cast<int32_t>(v); }
int64_t s16(uint32_t v) { return static_cast<int16_t>(static_cast<uint16_t>(v)); }
int64_t u16(uint32_t v) { return v & 0xFFFF; }
int64_t u24(uint32_t v) { return v & 0xFFFFFF; }
int64_t i24(uint32_t v) { return static_cast<int32_t>(v << 8) >> 8; }
uint32_t sad8(uint32_t a, uint32_t b, bool masked) {
  uint32_t s = 0;
  for (int k = 0; k < 4; ++k) {
    const int x = (a >> 8 * k) & 255, y = (b >> 8 * k) & 255;
    if (!masked || y) s += static_cast<uint32_t>(std::abs(x - y));
  }
  return s;
}
Want model(int op, const In& in, bool gfx10) {
  const int64_t a = in.a, b = in.b, c = in.c32, ci = in.cin;
  const auto w32 = [](int64_t v) { return Want{static_cast<uint32_t>(v), 0, 0, false}; };
  const auto w16 = [](int64_t v) { return Want{static_cast<uint32_t>(v) & 0xFFFF, 0, 0, true}; };
  const auto co = [](int64_t v, bool carry) { return Want{static_cast<uint32_t>(v), 0, carry ? 1u : 0u, false}; };
  const auto pk = [&](auto f) {
    return Want{static_cast<uint32_t>(f(in.a & 0xFFFF, in.b & 0xFFFF, in.c32 & 0xFFFF) & 0xFFFF) |
                    static_cast<uint32_t>(f(in.a >> 16, in.b >> 16, in.c32 >> 16) & 0xFFFF) << 16, 0, 0, false};
  };
  const auto mad_u64 = [&](bool clamp) {
    const unsigned __int128 p = static_cast<unsigned __int128>(in.a) * in.b + in.c64;
    const bool over = (p >> 64) != 0;
    const uint64_t r = clamp && over ? ~0ull : static_cast<uint64_t>(p);
    return Want{static_cast<uint32_t>(r), static_cast<uint32_t>(r >> 32), over ? 1u : 0u, false};
  };
  const auto mad_i64 = [&](bool clamp) {
    const __int128 p = static_cast<__int128>(s32(in.a)) * s32(in.b) + static_cast<int64_t>(in.c64);
    const __int128 lo = INT64_MIN, hi = INT64_MAX;
    const uint64_t r = static_cast<uint64_t>(clamp ? (p < lo ? lo : p > hi ? hi : p) : p);
    return Want{static_cast<uint32_t>(r), static_cast<uint32_t>(r >> 32), static_cast<uint32_t>(p >> 64) & 1, false};
  };
  switch (op) {
    case 0: return co(std::min(a + b, kU32), a + b > kU32);
    case 1: return co(std::max(a - b, int64_t{0}), a < b);
    case 2: return co(std::max(b - a, int64_t{0}), b < a);
    case 3: return co(std::min(a + b + ci, kU32), a + b + ci > kU32);
    case 4: return co(std::max(a - b - ci, int64_t{0}), a < b + ci);
    case 5: return co(std::max(b - a - ci, int64_t{0}), b < a + ci);
    case 6: return mad_u64(true);
    case 7: return mad_i64(true);
    case 8: return mad_u64(false);
    case 9: return mad_i64(false);
    case 10: return co(a + b, a + b > kU32);
    case 11: return co(a - b, a < b);
    case 12: return w32(std::min(a + b, kU32));
    case 13: return w32(std::max(a - b, int64_t{0}));
    case 14: return w32(std::max(b - a, int64_t{0}));
    case 15: return w32(sat(s32(in.a) + s32(in.b), INT32_MIN, INT32_MAX));
    case 16: return w32(sat(s32(in.a) - s32(in.b), INT32_MIN, INT32_MAX));
    case 17: return w16(std::min(u16(in.a) + u16(in.b), int64_t{65535}));
    case 18: return w16(std::max(u16(in.a) - u16(in.b), int64_t{0}));
    case 19: return w16(std::max(u16(in.b) - u16(in.a), int64_t{0}));
    case 20: return w16(sat(s16(in.a) + s16(in.b), -32768, 32767));
    case 21: return w16(sat(s16(in.a) - s16(in.b), -32768, 32767));
    case 22: return pk([](int64_t x, int64_t y, int64_t) { return std::min(x + y, int64_t{65535}); });
    case 23: return pk([](int64_t x, int64_t y, int64_t) { return std::max(x - y, int64_t{0}); });
    case 24: return pk([](int64_t x, int64_t y, int64_t) { return sat(s16(x) + s16(y), -32768, 32767); });
    case 25: return pk([](int64_t x, int64_t y, int64_t) { return sat(s16(x) - s16(y), -32768, 32767); });
    case 26: return pk([](int64_t x, int64_t y, int64_t z) { return std::min(x * y + z, int64_t{65535}); });
    case 27: return pk([](int64_t x, int64_t y, int64_t z) { return sat(s16(x) * s16(y) + s16(z), -32768, 32767); });
    case 28: return w32(std::min(u24(in.a) * u24(in.b) + c, kU32));
    case 29: return w32(sat(i24(in.a) * i24(in.b) + s32(in.c32), INT32_MIN, INT32_MAX));
    case 30: case 42: return w16(std::min(u16(in.a) * u16(in.b) + u16(in.c32), int64_t{65535}));
    case 31: case 43: return w16(sat(s16(in.a) * s16(in.b) + s16(in.c32), -32768, 32767));
    case 32: return w32(std::min(u16(in.a) * u16(in.b) + c, kU32));
    case 33: return w32(sat(s16(in.a) * s16(in.b) + s32(in.c32), INT32_MIN, INT32_MAX));
    case 34: return w32(std::min(u24(in.a) * u24(in.b), kU32));
    case 35: return w32(sat(i24(in.a) * i24(in.b), INT32_MIN, INT32_MAX));
    case 36: return w32(std::min((a > b ? a - b : b - a) + c, kU32));
    case 37: return w32(std::min(std::abs(u16(in.a) - u16(in.b)) + std::abs(u16(in.a >> 16) - u16(in.b >> 16)) + c, kU32));
    case 38: return w32(std::min(int64_t{sad8(in.a, in.b, false)} + c, kU32));
    case 39: return w32(std::min(int64_t{sad8(in.a, in.b, true)} + c, kU32));
    case 40: return w32(std::min((int64_t{sad8(in.a, in.b, false)} << 16) + c, kU32));
    case 41: return w16(gfx10 ? u16(in.a) * u16(in.b) : std::min(u16(in.a) * u16(in.b), int64_t{65535}));   // (gfx10: no clamp)
  }
  return {};
}
constexpr uint32_t kV[16] = {0u, 1u, 2u, 0x7ffeu, 0x7fffu, 0x8000u, 0xffffu, 0x10000u, 0x7fffffffu, 0x80000000u, 0x80000001u,
                             0xfffffffeu, 0xffffffffu, 0x12345678u, 0xdeadbeefu, 0x40000000u};
In input(int t) {
  In in;
  in.a = kV[t & 15];
  in.b = kV[(t >> 4) & 15];
  in.cin = (t >> 8) & 1;
  in.c64 = uint64_t{kV[(t * 7 + 3) & 15]} << 32 | kV[(t * 5 + (t >> 8)) & 15];
  in.c32 = kV[(t * 7 + 3) & 15];
  return in;
}
// --- end of the model ---

amd::CodeObject object(const char* target) {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/asm_int_clamp." + target + ".o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

int run(const char* target, bool tier) {
  constexpr int n = 512;
  const amd::CodeObject o = object(target);
  MemoryManager mem(16ull << 20);
  std::vector<uint32_t> words(n * 8, 0);
  for (int t = 0; t < n; ++t) {
    const In in = input(t);
    const uint32_t w[6] = {in.a, in.b, static_cast<uint32_t>(in.c64), static_cast<uint32_t>(in.c64 >> 32), in.c32, in.cin};
    std::memcpy(&words[t * 8], w, sizeof w);
  }
  const uint64_t pin = mem.alloc(words.size() * 4), pout = mem.alloc(kOps * n * 12);
  mem.write(pin, words.data(), words.size() * 4);
  const amd::Kernel* k = amd::find_kernel(o, "int_clamp");
  if (!k) throw vtest::Failure("no kernel named int_clamp");
  std::vector<uint8_t> buf(k->kernarg_size, 0);
  const uint64_t args[2] = {pin, pout};
  std::memcpy(buf.data(), args, sizeof args);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(buf.size());
  mem.write(d.kernarg, buf.data(), buf.size());
  d.group_size[0] = 64;
  d.groups[0] = n / 64;
  if (std::string(target) == "gfx1030") d.wave_size = 32;
  amd::gcn::tier::set_enabled(tier);
  amd::execute(d, mem);
  amd::gcn::tier::set_enabled(true);
  std::vector<uint32_t> got(kOps * n * 3);
  mem.read(pout, got.data(), got.size() * 4);
  const bool gfx10 = std::string(target) == "gfx1030";
  int wrong = 0;
  for (int op = 0; op < kOps; ++op)
    for (int t = 0, shown = 0; t < n; ++t) {
      const Want e = model(op, input(t), gfx10);
      const uint32_t* g = &got[(op * n + t) * 3];
      const uint32_t mask = e.low16 ? 0xFFFFu : ~0u;
      if ((g[0] & mask) == e.lo && g[1] == e.hi && g[2] == e.carry) continue;
      ++wrong;
      if (shown++ < 2) {   // (two lanes of each instruction that is wrong)
        const In in = input(t);
        std::fprintf(stderr, "  %s tier %d op %d t %d (a %x b %x c32 %x c64 %llx cin %u): %x %x %x, expected %x %x %x\n", target, tier, op, t,
                     in.a, in.b, in.c32, (unsigned long long)in.c64, in.cin, g[0] & mask, g[1], g[2], e.lo, e.hi, e.carry);
      }
    }
  return wrong;
}

}  // namespace

VTEST(clamped_integer_instructions_saturate_and_keep_their_carry_out) {
  int wrong = 0;
  for (const char* target : {"gfx90a", "gfx942", "gfx1030"})
    for (bool tier : {false, true}) wrong += run(target, tier);
  VCHECK_EQ(wrong, 0);
}

VTEST_MAIN
