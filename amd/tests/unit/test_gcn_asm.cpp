// Instructions chosen by hand, in kernels written in assembly
// (amd/tests/data/asm_*.s): the ones a compiler emits only now and then, so a
// C fixture could not be counted on to contain them. Each is checked against
// the same arithmetic done in 64 bits on the host.
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

amd::CodeObject object(const char* name) {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/" + name + ".gfx942.o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

// sopk(out, x): x + 32767 and whether it overflowed, x - 32768 and whether it
// overflowed, x * -3.
std::vector<int32_t> sopk(const amd::CodeObject& o, int32_t x) {
  MemoryManager mem(16ull << 20);
  const uint64_t out = mem.alloc(5 * 4);
  const amd::Kernel* k = amd::find_kernel(o, "sopk");
  if (!k) throw vtest::Failure("no kernel named sopk");
  std::vector<uint8_t> args(k->kernarg_size, 0);
  for (int b = 0; b < 8; ++b) args[b] = static_cast<uint8_t>(out >> (8 * b));
  for (int b = 0; b < 4; ++b) args[8 + b] = static_cast<uint8_t>(static_cast<uint32_t>(x) >> (8 * b));
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(args.size());
  mem.write(d.kernarg, args.data(), args.size());
  d.group_size[0] = 64;
  amd::execute(d, mem);
  std::vector<int32_t> r(5);
  mem.read(out, r.data(), 5 * 4);
  return r;
}

}  // namespace

VTEST(a_16_bit_constant_is_added_signed_and_scc_says_when_the_sum_overflowed) {
  const amd::CodeObject o = object("asm_sopk");
  for (int32_t x : {1, -5, INT_MAX, INT_MIN, 0x7fff8000}) {
    const std::vector<int32_t> r = sopk(o, x);
    const int64_t up = int64_t{x} + 32767, down = int64_t{x} - 32768;
    VCHECK_EQ(r[0], static_cast<int32_t>(static_cast<uint32_t>(up)));
    VCHECK_EQ(r[1], up > INT_MAX ? 1 : 0);
    VCHECK_EQ(r[2], static_cast<int32_t>(static_cast<uint32_t>(down)));
    VCHECK_EQ(r[3], down < INT_MIN ? 1 : 0);
    VCHECK_EQ(r[4], static_cast<int32_t>(static_cast<uint32_t>(int64_t{x} * -3)));
  }
}

// Runs one of the hand-written kernels over one wave, with its arguments
// laid out as the kernel's metadata says, and returns what it wrote to out.
std::vector<uint32_t> run(const amd::CodeObject& o, const char* name, MemoryManager& mem, uint64_t out, size_t words,
                          const std::vector<uint64_t>& args) {
  const amd::Kernel* k = amd::find_kernel(o, name);
  if (!k) throw vtest::Failure(std::string("no kernel named ") + name);
  std::vector<uint8_t> buf(k->kernarg_size, 0);
  for (size_t i = 0; i < args.size() && i < k->args.size(); ++i)
    for (uint32_t b = 0; b < k->args[i].size; ++b)
      buf[k->args[i].offset + b] = static_cast<uint8_t>(args[i] >> (8 * b));
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(buf.size());
  mem.write(d.kernarg, buf.data(), buf.size());
  d.group_size[0] = 64;
  amd::execute(d, mem);
  std::vector<uint32_t> r(words);
  mem.read(out, r.data(), words * 4);
  return r;
}

uint32_t u(int64_t v) { return static_cast<uint32_t>(v); }
uint32_t f(float v) {
  uint32_t b;
  std::memcpy(&b, &v, 4);
  return b;
}
uint32_t halves(float lo, float hi) {
  const _Float16 l = static_cast<_Float16>(lo), h = static_cast<_Float16>(hi);
  uint16_t a, b;
  std::memcpy(&a, &l, 2);
  std::memcpy(&b, &h, 2);
  return a | static_cast<uint32_t>(b) << 16;
}

VTEST(scalar_bit_fields_shifts_and_comparisons_give_what_the_isa_says) {
  const amd::CodeObject o = object("asm_scalar");
  for (auto [x, y] : std::vector<std::pair<int32_t, int32_t>>{{0x12345678, -7}, {-40000, 40000}, {33, 33}, {INT_MIN, 3}}) {
    MemoryManager mem(16ull << 20);
    const uint64_t out = mem.alloc(28 * 4);
    const std::vector<uint32_t> r =
        run(o, "scalar", mem, out, 28, {out, static_cast<uint32_t>(x), static_cast<uint32_t>(y)});
    const uint32_t ux = static_cast<uint32_t>(x), uy = static_cast<uint32_t>(y);
    const uint64_t pair = ux | static_cast<uint64_t>(uy) << 32;
    const int64_t field = static_cast<int64_t>(pair << (64 - 32)) >> (64 - 12);   // bits 20..31, signed
    const int64_t shifted = static_cast<int64_t>(pair) >> 7;
    const std::vector<uint32_t> want = {
        (ux >> 4) & 0xFF,
        u(field), u(field >> 32),
        u(shifted), u(shifted >> 32),
        u(std::min(x, y)), x < y,
        std::min(ux, uy), ux < uy,
        u(std::max(x, y)), x > y,
        (ux | 8u) & ~0x80000000u,
        (ux >> 5) & 1, !((ux >> 5) & 1),
        ux > 0x8000u, x < -32768,
        ux <= uy, x == y,
        x < 0 ? 0u - ux : ux,
        u((static_cast<int64_t>(x) * y) >> 32),
        (ux & 0xFFFF) | (uy & 0xFFFF) << 16,
        ~ux, ux & ~uy,
        x != y ? ux : 7u,
        ~ux, ~uy,
        0u, 0x3FF00000u};   // s_mov_b64 of 1.0: the double
    for (size_t i = 0; i < want.size(); ++i) VCHECK_EQ(r[i], want[i]);
  }
}

VTEST(a_kernel_finds_only_the_group_ids_it_asked_for_one_after_another) {
  const amd::CodeObject o = object("asm_scalar");
  const amd::Kernel* k = amd::find_kernel(o, "group_z");
  VCHECK(k != nullptr);
  VCHECK(k->group_id_x && !k->group_id_y && k->group_id_z);
  MemoryManager mem(16ull << 20);
  const uint64_t out = mem.alloc(3 * 4);
  const std::vector<uint32_t> none = {99, 99, 99};
  mem.write(out, none.data(), 3 * 4);
  std::vector<uint8_t> args(8);
  for (int b = 0; b < 8; ++b) args[b] = static_cast<uint8_t>(out >> (8 * b));
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(8);
  mem.write(d.kernarg, args.data(), 8);
  d.group_size[0] = 64;
  d.groups[2] = 3;   // three groups along z, as three batches
  amd::execute(d, mem);
  std::vector<uint32_t> r(3);
  mem.read(out, r.data(), 3 * 4);
  VCHECK_EQ(r[0], 0u);
  VCHECK_EQ(r[1], 1u);
  VCHECK_EQ(r[2], 2u);
}

VTEST(buffers_lds_and_private_memory_are_reached_as_a_card_reaches_them) {
  const amd::CodeObject o = object("asm_memory");
  MemoryManager mem(16ull << 20);
  const std::vector<uint32_t> init = {100, 101, 102, 103, 104, 105, 106, 107};
  const uint64_t buf = mem.alloc(8 * 4), out = mem.alloc(19 * 4);
  mem.write(buf, init.data(), 8 * 4);
  const std::vector<uint32_t> r = run(o, "memory", mem, out, 19, {buf, out});
  VCHECK_EQ(r[0], 630u);   // lane 0 is sent lane 63's ten times 63
  VCHECK_EQ(r[1], 101u);   // in range
  VCHECK_EQ(r[2], 0u);     // past the end
  VCHECK_EQ(r[3], 103u);   // a pair: its first word in range
  VCHECK_EQ(r[4], 0u);     // and its second past the end
  VCHECK_EQ(r[5], 103u);   // the scalar offset counts against the bounds: 4 + 8 is in range
  VCHECK_EQ(r[14], 0u);    // and 12 + 4 is not
  VCHECK_EQ(r[6], 106u);   // a scalar load's offset from a register
  VCHECK_EQ(r[7], 0x11u);  // two LDS pairs, 512 bytes apart
  VCHECK_EQ(r[8], 0x22u);
  VCHECK_EQ(r[9], 0x33u);
  VCHECK_EQ(r[10], 0x44u);
  VCHECK_EQ(r[11], 0u);    // far past LDS: zero, not what the register held
  VCHECK_EQ(r[12], 0x1234u);   // private memory, by offset alone
  VCHECK_EQ(r[13], 0x1234u);   // and from a scalar register's offset
  VCHECK_EQ(r[15], 101u);        // a short into the low half, the high one cleared
  VCHECK_EQ(r[16], 103u << 16);  // one into the high half, the low one cleared
  VCHECK_EQ(r[17], 0x11u << 16); // and from LDS
  VCHECK_EQ(r[18], 0u);          // past the end: all of it zero
  std::vector<uint32_t> after(8);
  mem.read(buf, after.data(), 8 * 4);
  VCHECK_EQ(after[2], 0x77u);   // the store in range landed
  VCHECK_EQ(after[4], 104u);    // the one past the end did not
}

VTEST(vector_comparisons_packed_math_and_mixed_precision_give_what_the_isa_says) {
  const amd::CodeObject o = object("asm_vector");
  MemoryManager mem(16ull << 20);
  const uint64_t out = mem.alloc(31 * 4);
  const int32_t x = 10, y = 7;
  const std::vector<uint32_t> r = run(o, "vector", mem, out, 31, {out, x, y});
  VCHECK_EQ(r[0], 10u);    // EXEC narrowed to the lanes below x
  VCHECK_EQ(r[1], 10u);    // and VCC written with it
  VCHECK_EQ(r[2], 1u);     // then to lane 3 alone
  VCHECK_EQ(r[3], 3u);
  VCHECK_EQ(r[19], 8u);    // the long form writes its pair too
  VCHECK_EQ(r[20], 0u);
  const __int128 sum = static_cast<__int128>(x) * y + 0x7ffffffffffffff0ll;
  VCHECK_EQ(r[4], u(static_cast<int64_t>(sum)));
  VCHECK_EQ(r[5], u(static_cast<int64_t>(sum) >> 32));
  VCHECK_EQ(r[6], 1u);     // and it overflowed
  VCHECK_EQ(r[21], 0u);
  VCHECK_EQ(r[7], halves(-(-2.0f) + 0.25f, 1.5f + 3.0f));   // (-hi(v6) + lo(v7), lo(v6) + hi(v7))
  VCHECK_EQ(r[8], halves(1.5f, -2.0f));                      // max(1.5, 0.25), max(-2, -3)
  VCHECK_EQ(r[9], halves(1.5f * 0.25f + 1.5f, -2.0f * 3.0f + 1.5f));
  VCHECK_EQ(r[10], f(5.0f * 3.0f));    // high of s[8:9] times low of v[12:13]
  VCHECK_EQ(r[11], f(2.0f * 7.0f));    // low of s[8:9] times high of v[12:13]
  VCHECK_EQ(r[12], f(3.0f + 2.0f));
  VCHECK_EQ(r[13], f(7.0f - 5.0f));    // the high result's second source negated
  VCHECK_EQ(r[14], f(7.0f));           // v_pk_mov_b32: the high of the first
  VCHECK_EQ(r[15], f(5.0f));           // and the high of the second
  VCHECK_EQ(r[16], f(-2.0f * 3.0f + 7.0f));
  VCHECK_EQ(r[17], 1u);                // the high word of 0x50003 is 5
  VCHECK_EQ(r[18], 0u);                // its low word is not
  VCHECK_EQ(r[22], 3u * 63u);          // lane 63 multiplied by 3, as it was before any carry was written
  VCHECK_EQ(r[23], 0u);                // and no lane carried
  VCHECK_EQ(r[24], f(-2.0f));          // a select of a negated source
  VCHECK_EQ(r[25], f(4.0f));           // and of a source's absolute value
  VCHECK_EQ(r[26], f(3.0f * 2.0f));    // (3, 7) times (2, 5), written over the (3, 7)
  VCHECK_EQ(r[27], f(3.0f * 5.0f));    // whose high result reads the 3 as it was
  VCHECK_EQ(r[28], 0xff0000ffu);       // the signs of bytes 1 (0x80), 3 (0), 5 (0x7f) and 7 (0x80)
  VCHECK_EQ(r[29], 0xff008002u);       // byte 4, byte 7, a zero byte, a byte of ones
  VCHECK_EQ(r[30], halves(1.5f + 1.0f, -2.0f));   // a float constant is the half in the low 16 bits, 0 above
}

VTEST(what_pytorchs_rocm_libraries_use_gives_what_the_isa_says) {
  const amd::CodeObject o = object("asm_libs");
  MemoryManager mem(16ull << 20);
  const uint64_t out = mem.alloc(59 * 4);
  const std::vector<uint32_t> r = run(o, "libs", mem, out, 59, {out});
  const auto h = [](float v) {
    const _Float16 x = static_cast<_Float16>(v);
    uint16_t b;
    std::memcpy(&b, &x, 2);
    return static_cast<uint32_t>(b);
  };
  VCHECK_EQ(r[0], 4u);            // s_ff1 of 0xf0
  VCHECK_EQ(r[1], 24u);           // s_flbit of 0xf0
  VCHECK_EQ(r[2], 29u);           // the first bit of -8 unlike its sign
  VCHECK_EQ(r[3], 0xffffffffu);   // bits 8 to 11 of 0xf00, signed: -1
  VCHECK_EQ(r[4], 27u);           // 5 << 2 + 7
  VCHECK_EQ(r[5], 0x9abc1234u);   // the two high halves
  VCHECK_EQ(r[6], 0xfffffffeu);   // 0xfffe sign-extended
  VCHECK_EQ(r[7], 0x55u);         // vcc_hi, kept when vcc_lo was written after it
  VCHECK_EQ(r[8], 0x66u);
  VCHECK_EQ(r[9], 1u);            // MODE's IEEE bit, as a dispatch starts
  VCHECK_EQ(r[10], 0u);           // and after it was cleared
  VCHECK_EQ(r[11], 77u);          // what the called function set
  VCHECK_EQ(r[12], 28u);          // v_ffbh_i32 of 0xfffffff0
  VCHECK_EQ(r[13], 4u);           // v_ffbl_b32
  VCHECK_EQ(r[14], h(-3.0f));     // -3 as a half
  VCHECK_EQ(r[15], 20u);          // the half 20 as an integer
  VCHECK_EQ(r[16], h(1.0f / 20.0f));
  VCHECK_EQ(r[17], f(4.0f));      // 5 - 1
  VCHECK_EQ(r[18], static_cast<uint32_t>((0xffffffull * 0xffffffull) >> 32));
  VCHECK_EQ(r[19], 0xffe0u);      // -128 >> 2 in 16 bits
  VCHECK_EQ(r[20], 5u);           // max(-128, 5)
  VCHECK_EQ(r[21], 0xff80u);      // min(-128, 5)
  VCHECK_EQ(r[22], 0xf00ff00fu);  // xnor
  VCHECK_EQ(r[23], 7u);           // the middle of 10, 3 and 7
  VCHECK_EQ(r[24], 0xffffu * 3u + 100u);
  VCHECK_EQ(r[25], 5u);           // max(-128, 5, -1)
  VCHECK_EQ(r[26], 0xf00u);       // four ones, eight up
  VCHECK_EQ(r[27], 0x0005ffffu);  // 70000 held to 0xffff, and 5
  VCHECK_EQ(r[28], 0x7fffffffu);  // INT_MAX + 1, clamped
  VCHECK_EQ(r[29], 0x00070005u);  // (1, 2) + (4, 5)
  VCHECK_EQ(r[30], 0xfffc0008u);  // (16, -8) >> 1, the constant in both
  VCHECK_EQ(r[31], 0x00120011u);  // 16 - 0xffff (-2's high half), 16 - 0xfffe
  VCHECK_EQ(r[32], f(11.5f));     // 1 * 3 + 2 * 4 + 0.5
  const double two_over_pi = std::ldexp(static_cast<double>(0xa2f9836e4e441529ull >> 11), -53);
  uint64_t bits;
  std::memcpy(&bits, &two_over_pi, 8);
  VCHECK_EQ(r[33], static_cast<uint32_t>(bits));   // 2/pi's first 53 bits
  VCHECK_EQ(r[34], static_cast<uint32_t>(bits >> 32));
  VCHECK_EQ(r[35], 2u);           // swapped
  VCHECK_EQ(r[36], 1u);
  VCHECK_EQ(r[37], 2u);           // through two accumulation registers
  VCHECK_EQ(r[38], f(4.0f));      // (3 + 5) / 2
  VCHECK_EQ(r[39], f(30.0f));     // 3 * 5 * 2
  VCHECK_EQ(r[40], 0x44004000u);  // the high halves of (1, 2) and (3, 4)
  VCHECK_EQ(r[41], 0x0007abcdu);  // max(5, 7, 7) into the high half, the low kept
  VCHECK_EQ(r[42], 0x11220344u);  // 1 + 2 into byte 1, the rest kept
  VCHECK_EQ(r[43], 0u);           // v70, untouched
  VCHECK_EQ(r[44], 99u);          // v71, which the indexed move reached
  VCHECK_EQ(r[45], 0x70u);        // three ones, four up
  VCHECK_EQ(r[50], 9u);           // max(5, 9) as 64 bits
  VCHECK_EQ(r[51], 0u);
  VCHECK_EQ(r[52], halves(1.5f, 2.25f));   // (1, 2) + (0.5, 0.25)
  VCHECK_EQ(r[53], 0x40403fc0u);  // bfloat16s (1, 2) + (0.5, 1)
  VCHECK_EQ(r[54], 0xfffffffdu);  // min(10, -3) in LDS
  VCHECK_EQ(r[55], 42u);          // -3 found, so 42 stored
  VCHECK_EQ(r[56], 0xfffffffdu);  // and -3 handed back
  VCHECK_EQ(r[57], 12u);          // 7 + 5 as 64 bits
  VCHECK_EQ(r[58], 0x4840beefu);  // fp8 1 and 2 into the high half
}

VTEST_MAIN
