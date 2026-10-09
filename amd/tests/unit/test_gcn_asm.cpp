// Instructions chosen by hand, in kernels written in assembly
// (amd/tests/data/asm_*.s): the ones a compiler emits only now and then, so a
// C fixture could not be counted on to contain them. Each is checked against
// the same arithmetic done in 64 bits on the host.
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vgpu/amd_image.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

amd::CodeObject object(const char* name, const char* target = "gfx942") {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/" + name + "." + target + ".o";
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

VTEST(the_rest_of_the_scalar_bitwise_family_gives_what_the_isa_says) {
  const amd::CodeObject o = object("asm_logic");
  const std::vector<std::pair<uint64_t, uint64_t>> cases = {
      {0x123456789abcdef0ull, 0x0f0f0f0f00ff00ffull},
      {~0ull, ~0ull},                                 // nand and xnor give no bits: SCC clear
      {0, 0},                                         // nor gives every bit
      {0xffffffff80000000ull, 0x0000000000000001ull},  // the most negative low half, less one
      {0x00000000deadbe05ull, 0x00000000000c0008ull}}; // a 12-bit field from bit 8; an 8-bit mask at bit 5
  for (auto [a, b] : cases) {
    MemoryManager mem(16ull << 20);
    const uint64_t out = mem.alloc(24 * 4);
    const std::vector<uint32_t> r = run(o, "logic", mem, out, 24, {out, a, b});
    const uint32_t x = static_cast<uint32_t>(a), y = static_cast<uint32_t>(b);
    const uint64_t nand = ~(a & b), nor = ~(a | b), xnor = ~(a ^ b);
    const uint64_t bfm = ((uint64_t{1} << (y & 63)) - 1) << (x & 63);
    const uint32_t start = y & 63, width = (y >> 16) & 0x7F;
    const uint64_t bfe = width == 0 ? 0 : width >= 64 ? a >> start : (a >> start) & ((uint64_t{1} << width) - 1);
    const int64_t d = int64_t{static_cast<int32_t>(x)} - static_cast<int32_t>(y);
    const uint32_t absdiff = static_cast<uint32_t>(d < 0 ? -d : d);
    const std::vector<uint32_t> want = {
        x | ~y, (x | ~y) != 0,
        ~(x & y), ~(x & y) != 0,
        u(nand), u(nand >> 32), nand != 0,
        ~(x | y), ~(x | y) != 0,
        u(nor), u(nor >> 32), nor != 0,
        ~(x ^ y), ~(x ^ y) != 0,
        u(xnor), u(xnor >> 32), xnor != 0,
        u(bfm), u(bfm >> 32),
        u(bfe), u(bfe >> 32), bfe != 0,
        absdiff, absdiff != 0};
    for (size_t i = 0; i < want.size(); ++i) VCHECK_EQ(r[i], want[i]);
  }
}

VTEST(buffer_atomics_change_the_word_and_return_what_it_was) {
  const amd::CodeObject o = object("asm_atomics");
  const std::vector<std::pair<uint32_t, uint32_t>> cases = {  // (every word before, x)
      {100, 7}, {7, 100}, {0, 5}, {0xFFFFFFF0u, 3}, {5, 5}, {0x80000000u, 0x7FFFFFFFu}};
  for (auto [before, x] : cases) {
    MemoryManager mem(16ull << 20);
    const uint64_t buf = mem.alloc(10 * 4), out = mem.alloc(10 * 4);
    const std::vector<uint32_t> init(10, before);
    mem.write(buf, init.data(), 10 * 4);
    const std::vector<uint32_t> old = run(o, "atomics", mem, out, 10, {buf, out, x});
    std::vector<uint32_t> now(10);
    mem.read(buf, now.data(), 10 * 4);
    const int32_t sb = static_cast<int32_t>(before), sx = static_cast<int32_t>(x);
    const std::vector<uint32_t> want = {
        before - x,
        u(std::min(sb, sx)), std::min(before, x),
        u(std::max(sb, sx)), std::max(before, x),
        before & x, before | x, before ^ x,
        before >= x ? 0u : before + 1,
        before == 0 || before > x ? x : before - 1};
    for (size_t i = 0; i < want.size(); ++i) {
      VCHECK_EQ(old[i], before);
      VCHECK_EQ(now[i], want[i]);
    }
  }
}

VTEST(a_matrix_instruction_broadcasts_a_blocks_a_to_its_group) {
  // v_mfma_f32_4x4x4_16b_f16: sixteen blocks, four lanes each; a lane's A
  // is two words, four halves. With CBSZ 4 every block takes block 0's A;
  // with CBSZ 2 and ABID 1, each group of four blocks takes its second's.
  // The same products without broadcast, from A as the broadcast should
  // have read it, must agree bit for bit.
  const amd::CodeObject o = object("asm_bcast");
  MemoryManager mem(16ull << 20);
  std::vector<uint32_t> a(128), a4(128), a2(128), b(128);
  uint32_t seed = 12345;
  const auto half = [&]() -> uint32_t {   // a small float16, to keep sums exact
    seed = seed * 1103515245u + 12345u;
    return 0x3c00u + ((seed >> 16) & 0x3FF);   // in [1, 2)
  };
  for (auto* v : {&a, &b})
    for (auto& x : *v) x = half() | half() << 16;
  for (uint32_t lane = 0; lane < 64; ++lane)
    for (uint32_t k = 0; k < 2; ++k) {
      a4[lane * 2 + k] = a[(lane % 4) * 2 + k];
      a2[lane * 2 + k] = a[((lane / 16) * 16 + 4 + lane % 4) * 2 + k];
    }
  uint64_t in[4];
  const std::vector<uint32_t>* src[4] = {&a, &a4, &a2, &b};
  for (int i = 0; i < 4; ++i) {
    in[i] = mem.alloc(128 * 4);
    mem.write(in[i], src[i]->data(), 128 * 4);
  }
  const uint64_t out = mem.alloc(4 * 256 * 4);
  const std::vector<uint32_t> r = run(o, "bcast", mem, out, 4 * 256, {in[0], in[1], in[2], in[3], out});
  int differ4 = 0, differ2 = 0, changed = 0;
  for (uint32_t i = 0; i < 256; ++i) {
    differ4 += r[i] != r[512 + i];
    differ2 += r[256 + i] != r[768 + i];
    changed += r[i] != r[256 + i];
  }
  VCHECK_EQ(differ4, 0);
  VCHECK_EQ(differ2, 0);
  VCHECK(changed > 0);   // the two broadcasts took different blocks' A
}

VTEST(a_permute_or_swizzle_that_reads_a_switched_off_lane_gets_zero) {
  // The MI300 ISA guide: DS_BPERMUTE_B32 returns zero when src_lane is disabled, and DS_SWIZZLE_B32
  // reads "thread_valid[j] ? thread_in[j] : 0". Lanes that are off keep what they held. rocPRIM's
  // warp shuffles (test_intrinsics) are written to that, with half the lanes off.
  const amd::CodeObject o = object("asm_permute_exec");
  MemoryManager mem(16ull << 20);
  const uint64_t active = 0x0123456789ABCDEFull;
  std::vector<uint32_t> in(64);
  for (uint32_t i = 0; i < 64; ++i) in[i] = 1000 + i;
  const uint64_t pin = mem.alloc(64 * 4), out = mem.alloc(128 * 4);
  mem.write(pin, in.data(), 64 * 4);
  const std::vector<uint32_t> r = run(o, "permute", mem, out, 128, {pin, out});
  int wrong = 0, zeros = 0;
  for (uint32_t lane = 0; lane < 64; ++lane) {
    const bool on = active >> lane & 1;
    const uint32_t from = (lane + 3) & 63, mate = lane ^ 1;
    const uint32_t want_b = !on ? 0xAAu : (active >> from & 1) ? in[from] : 0u;
    const uint32_t want_s = !on ? 0xAAu : (active >> mate & 1) ? in[mate] : 0u;
    wrong += r[lane] != want_b;
    wrong += r[64 + lane] != want_s;
    zeros += on && !(active >> from & 1);
  }
  VCHECK_EQ(wrong, 0);
  VCHECK(zeros > 10);   // the pattern does read disabled lanes
}

VTEST(xf32_matrix_instructions_cut_a_floats_mantissa_to_ten_bits) {
  // v_mfma_f32_16x16x8_xf32 and v_mfma_f32_32x32x4_xf32 take floats and multiply them with the
  // mantissa truncated to 10 bits (the MI300 ISA guide, 7.1), accumulating into a float. Inputs are
  // full-mantissa floats so the truncation shows; both forms put A[i][k] in item k % K_L of lane
  // i + M * (k / K_L), B the same by column, and the output in the layout of any MFMA of that block.
  const amd::CodeObject o = object("asm_xf32");
  MemoryManager mem(16ull << 20);
  std::vector<float> a(128), b(128);
  uint32_t seed = 4242;
  const auto next = [&]() -> float {
    seed = seed * 1103515245u + 12345u;
    return static_cast<float>(static_cast<int32_t>(seed >> 8) - (1 << 23)) / static_cast<float>(1 << 22);
  };
  for (auto* v : {&a, &b})
    for (auto& x : *v) x = next();
  const uint64_t ina = mem.alloc(128 * 4), inb = mem.alloc(128 * 4), out = mem.alloc(1280 * 4);
  mem.write(ina, a.data(), 128 * 4);
  mem.write(inb, b.data(), 128 * 4);
  const std::vector<uint32_t> r = run(o, "xf32", mem, out, 1280, {ina, inb, out});
  const auto trunc = [](float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, 4);
    bits &= ~0x1FFFu;
    std::memcpy(&x, &bits, 4);
    return static_cast<double>(x);
  };
  int wrong = 0, differs_from_full = 0;
  struct Form { uint32_t m, regs, base; };
  for (const Form f : {Form{16, 4, 0}, Form{32, 16, 256}}) {
    const uint32_t m = f.m, k_total = f.m == 16 ? 8 : 4, k_l = k_total / (64 / m);
    for (uint32_t lane = 0; lane < 64; ++lane)
      for (uint32_t reg = 0; reg < f.regs; ++reg) {
        const uint32_t row = m == 16 ? 4 * (lane / 16) + reg : 8 * (reg / 4) + 4 * (lane / 32) + reg % 4;
        const uint32_t col = lane % m;
        double sum = 0, full = 0;
        for (uint32_t k = 0; k < k_total; ++k) {
          const float av = a[(row + m * (k / k_l)) * 2 + k % k_l], bv = b[(col + m * (k / k_l)) * 2 + k % k_l];
          sum += trunc(av) * trunc(bv);
          full += static_cast<double>(av) * static_cast<double>(bv);
        }
        const float want = static_cast<float>(sum);
        uint32_t want_bits;
        std::memcpy(&want_bits, &want, 4);
        wrong += r[f.base + lane * f.regs + reg] != want_bits;
        differs_from_full += static_cast<float>(full) != want;
      }
  }
  VCHECK_EQ(wrong, 0);
  VCHECK(differs_from_full > 200);   // the cut mantissa changes most results
}

VTEST(global_loads_into_lds_land_where_m0_the_offset_and_the_lane_say) {
  const amd::CodeObject o = object("asm_lds_dma", "gfx950");
  MemoryManager mem(16ull << 20);
  std::vector<uint32_t> src(256);
  for (uint32_t i = 0; i < src.size(); ++i) src[i] = i * 7 + 1;
  const uint64_t in = mem.alloc(256 * 4), out = mem.alloc(768 * 4);
  mem.write(in, src.data(), 256 * 4);
  const std::vector<uint32_t> r = run(o, "lds_dma", mem, out, 768, {in, out});
  std::vector<uint32_t> want(768, 0);
  for (uint32_t lane = 0; lane < 64; ++lane) {
    for (uint32_t k = 0; k < 4; ++k) want[lane * 4 + k] = src[lane * 4 + k];            // x4: M0 0
    for (uint32_t k = 0; k < 3; ++k) want[257 + lane * 3 + k] = src[lane * 4 + 1 + k];  // x3: M0 1024, offset 4
    want[514 + lane] = src[lane * 4 + 2];                                                // x1: M0 2048, offset 8
  }
  int wrong = 0;
  for (size_t i = 0; i < want.size(); ++i) wrong += r[i] != want[i];
  VCHECK_EQ(wrong, 0);
}

VTEST(transposing_lds_reads_hand_each_lane_its_column) {
  // Destination lane l's element r is element l mod tile of lane A's read,
  // A = lane bits t..t+b-1, then r, then lane bits t+b and up (t = log2 tile;
  // b = 2, 1, 0 for 16-, 8-, 4-bit elements): the map Triton's AMD backend
  // lowers gfx950's ds_read_b64_tr_* by. For b16 that is the ISA's "each
  // lane holds 4 consecutive M values": lane l's four halves are the
  // (l mod 4)-th value of four lanes that differ only in the two bits r moves.
  const amd::CodeObject o = object("asm_ds_tr", "gfx950");
  MemoryManager mem(16ull << 20);
  std::vector<uint64_t> pattern(64);
  for (uint64_t i = 0; i < 64; ++i) pattern[i] = 0x0123456789abcdefull * (i + 1) ^ (i << 56);
  const uint64_t in = mem.alloc(64 * 8), out = mem.alloc(3 * 64 * 8);
  mem.write(in, pattern.data(), 64 * 8);
  const std::vector<uint32_t> r32 = run(o, "tr", mem, out, 3 * 64 * 2, {in, out});
  int wrong = 0;
  for (uint32_t form = 0; form < 3; ++form) {
    const uint32_t bits = form == 0 ? 16 : form == 1 ? 8 : 4, tile = 64 / bits;
    const uint32_t t = form == 0 ? 2 : form == 1 ? 3 : 4, b = form == 0 ? 2 : form == 1 ? 1 : 0;
    const uint64_t mask = (uint64_t{1} << bits) - 1;
    for (uint32_t l = 0; l < 64; ++l) {
      uint64_t want = 0;
      for (uint32_t r = 0; r < tile; ++r) {
        const uint32_t a = ((l >> t) & ((1u << b) - 1)) | (r << b) | ((l >> (t + b)) << (b + t));
        want |= ((pattern[a] >> ((l % tile) * bits)) & mask) << (r * bits);
      }
      const size_t at = (form * 64 + l) * 2;
      const uint64_t got = r32[at] | uint64_t{r32[at + 1]} << 32;
      wrong += got != want;
    }
  }
  VCHECK_EQ(wrong, 0);
}

VTEST(the_instructions_hip_tests_device_library_runs_compute_what_the_isa_says) {
  const amd::CodeObject o = object("asm_isa_gaps");
  MemoryManager mem(16ull << 20);
  std::vector<uint32_t> in(64 * 4), counters(64 * 2);
  uint32_t seed = 2024;
  const auto next = [&] {
    seed = seed * 1103515245u + 12345u;
    return seed;
  };
  for (uint32_t l = 0; l < 64; ++l) {
    in[4 * l + 0] = l < 8 ? 0xFFFFFFF0u + l : next();   // some sums that saturate
    in[4 * l + 1] = l < 8 ? 0x40u : next() >> (l & 7);
    in[4 * l + 2] = f(static_cast<float>(l) * 0.37f - 7.0f);
    in[4 * l + 3] = next();
    counters[2 * l] = l % 5;         // at, below and past the limit
    counters[2 * l + 1] = l % 3 == 0 ? 0 : next() % 200;
  }
  const uint64_t in_d = mem.alloc(in.size() * 4), out = mem.alloc(64 * 20 * 4), cnt = mem.alloc(counters.size() * 4);
  mem.write(in_d, in.data(), in.size() * 4);
  mem.write(cnt, counters.data(), counters.size() * 4);
  const std::vector<uint32_t> r = run(o, "gaps", mem, out, 64 * 20, {in_d, out, cnt});
  std::vector<uint32_t> after(counters.size());
  mem.read(cnt, after.data(), after.size() * 4);
  const auto s16 = [](uint32_t x, int h) { return static_cast<int64_t>(static_cast<int16_t>(x >> (16 * h))); };
  const auto s4 = [](uint32_t x, int k) {
    int64_t v = (x >> (4 * k)) & 15;
    return v & 8 ? v - 16 : v;
  };
  int wrong = 0;
  for (uint32_t l = 0; l < 64; ++l) {
    const uint32_t a = in[4 * l], b = in[4 * l + 1], c = in[4 * l + 2], d = in[4 * l + 3];
    const uint32_t* got = &r[20 * l];
    float cf;
    std::memcpy(&cf, &c, 4);
    int64_t dot2i = static_cast<int32_t>(d), dot2u = d, dot4u = d, dot8i = static_cast<int32_t>(d), dot8u = d;
    for (int h = 0; h < 2; ++h) {
      dot2i += s16(a, h) * s16(b, h);
      dot2u += int64_t{(a >> (16 * h)) & 0xFFFF} * ((b >> (16 * h)) & 0xFFFF);
    }
    for (int k = 0; k < 4; ++k) dot4u += int64_t{(a >> (8 * k)) & 0xFF} * ((b >> (8 * k)) & 0xFF);
    for (int k = 0; k < 8; ++k) {
      dot8i += s4(a, k) * s4(b, k);
      dot8u += int64_t{(a >> (4 * k)) & 15} * ((b >> (4 * k)) & 15);
    }
    const uint32_t inc_after = counters[2 * l] >= b ? 0 : counters[2 * l] + 1;
    const uint32_t dec_before = counters[2 * l + 1];
    const uint32_t dec_after = dec_before == 0 || dec_before > b ? b : dec_before - 1;
    const uint32_t lds_inc = a >= b ? 0 : a + 1;
    const uint32_t want[20] = {
        uint64_t{a} + b > UINT32_MAX ? UINT32_MAX : a + b,
        b > a ? 0u : a - b,
        f(cf - std::floor(cf)),
        u(dot2i), u(dot2u), u(dot4u), u(dot8i), u(dot8u),
        u(dot2i), u(dot8i),   // the VOP2 forms add into d the same way (no clamp, no overflow here)
        l == 63 ? 0xFFFFu : l + 1, l == 0 ? 0xFFFFu : l - 1, (l + 1) % 64, (l + 63) % 64,
        counters[2 * l], dec_before,
        a, lds_inc, c, 0};
    for (int i = 0; i < 20; ++i) wrong += got[i] != want[i];
    wrong += after[2 * l] != inc_after;
    wrong += after[2 * l + 1] != dec_after;
  }
  VCHECK_EQ(wrong, 0);
}

VTEST(a_work_group_waiting_for_another_on_the_same_thread_is_released) {
  // One host thread, two groups, group 0 first and waiting for group 1.
  setenv("VGPU_THREADS", "1", 1);
  const amd::CodeObject o = object("asm_wait");
  MemoryManager mem(16ull << 20);
  const uint64_t flag = mem.alloc(4), out = mem.alloc(4);
  const uint32_t zero = 0;
  mem.write(flag, &zero, 4);
  mem.write(out, &zero, 4);
  const amd::Kernel* k = amd::find_kernel(o, "wait");
  VCHECK(k != nullptr);
  if (!k) return;
  std::vector<uint8_t> args(k->kernarg_size, 0);
  for (int b = 0; b < 8; ++b) args[b] = static_cast<uint8_t>(flag >> (8 * b)), args[8 + b] = static_cast<uint8_t>(out >> (8 * b));
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(args.size());
  mem.write(d.kernarg, args.data(), args.size());
  d.group_size[0] = 64;
  d.groups[0] = 2;
  amd::execute(d, mem);
  unsetenv("VGPU_THREADS");
  uint32_t got = 0;
  mem.read(out, &got, 4);
  VCHECK_EQ(got, 42u);
}

VTEST(rdna_permlanex16_takes_op_sel_as_fi_and_bound_ctrl) {
  const amd::CodeObject o = object("asm_permlane", "gfx1100");
  MemoryManager mem(16ull << 20);
  const uint64_t out = mem.alloc(4 * 64 * 4);
  const std::vector<uint32_t> r = run(o, "permlane", mem, out, 4 * 64, {out});
  int wrong = 0;
  for (uint32_t item = 0; item < 64; ++item) {
    const uint32_t lane = item % 32, from = 0xff800000u | (lane ^ 16);
    const bool on = lane < 16;
    wrong += r[item] != from;                          // FI, every lane on: all 32 bits
    wrong += r[64 + item] != 7u;                       // no bits, source off: kept
    wrong += r[128 + item] != (on ? from : 7u);        // FI: read anyway
    wrong += r[192 + item] != (on ? 0u : 7u);          // BOUND_CTRL: zero
  }
  VCHECK_EQ(wrong, 0);
}

VTEST(rdna4_gives_each_wave_its_number_in_ttmp8) {
  const amd::CodeObject o = object("asm_ttmp", "gfx1201");
  MemoryManager mem(16ull << 20);
  const uint64_t out = mem.alloc(128 * 4);
  const amd::Kernel* k = amd::find_kernel(o, "wave_id");
  VCHECK(k != nullptr);
  if (!k) return;
  std::vector<uint8_t> args(k->kernarg_size, 0);
  for (int b = 0; b < 8; ++b) args[b] = static_cast<uint8_t>(out >> (8 * b));
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(args.size());
  mem.write(d.kernarg, args.data(), args.size());
  d.group_size[0] = 128;   // four waves of 32
  amd::execute(d, mem);
  std::vector<uint32_t> r(128);
  mem.read(out, r.data(), 128 * 4);
  int wrong = 0;
  for (uint32_t i = 0; i < 128; ++i) wrong += r[i] != i / 32;
  VCHECK_EQ(wrong, 0);
}

// A wave has scratch for every lane, work-item or not: a function saving
// whole-wave registers stores with every lane on, and hip-tests' double
// math, one work-item calling such a function, did just that.
VTEST(rdna_every_lane_of_a_wave_has_scratch_in_a_group_of_one) {
  const amd::CodeObject o = object("asm_wave", "gfx1100");
  MemoryManager mem(16ull << 20);
  const uint64_t out = mem.alloc(8);
  const amd::Kernel* k = amd::find_kernel(o, "whole_wave_scratch");
  VCHECK(k != nullptr);
  if (!k) return;
  std::vector<uint8_t> args(k->kernarg_size, 0);
  for (int b = 0; b < 8; ++b) args[b] = static_cast<uint8_t>(out >> (8 * b));
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(args.size());
  mem.write(d.kernarg, args.data(), args.size());
  d.group_size[0] = 1;
  amd::execute(d, mem);
  uint32_t r[2] = {};
  mem.read(out, r, 8);
  VCHECK_EQ(r[0], 0x5eed0000u);
  VCHECK_EQ(r[1], 0x5eed001fu);
}

// Runs a `where` kernel (asm_hwid*.s) over `groups` work-groups of one wave
// each, on a device laid out as `layout`, and returns the `words` it wrote
// for each group.
std::vector<uint32_t> where(const amd::CodeObject& o, uint32_t groups, amd::Dispatch::Layout layout, uint32_t lanes,
                            size_t words) {
  MemoryManager mem(16ull << 20);
  const uint64_t out = mem.alloc(groups * words * 4);
  const amd::Kernel* k = amd::find_kernel(o, "where");
  if (!k) throw vtest::Failure("no kernel named where");
  std::vector<uint8_t> args(k->kernarg_size, 0);
  for (int b = 0; b < 8; ++b) args[b] = static_cast<uint8_t>(out >> (8 * b));
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(args.size());
  mem.write(d.kernarg, args.data(), args.size());
  d.groups[0] = groups;
  d.group_size[0] = lanes;
  d.layout = layout;
  amd::execute(d, mem);
  std::vector<uint32_t> r(groups * words);
  mem.read(out, r.data(), r.size() * 4);
  return r;
}

// An MI300X: 8 compute dies of 4 shader engines, 38 compute units to a die.
// Work-groups go to the dies in turn, then to each die's engines in turn.
// HW_ID says the compute unit (11:8) and engine (14:13), XCC_ID the die, and
// HIP's __smid, which puts them together, tells all 304 apart.
VTEST(cdna3_hw_id_and_xcc_id_say_which_compute_unit_a_wave_runs_on) {
  const std::vector<uint32_t> r = where(object("asm_hwid"), 608, {8, 4, 1, 38}, 64, 2);
  int wrong = 0;
  std::set<uint32_t> smid;
  for (uint32_t g = 0; g < 608; ++g) {
    const uint32_t hw = r[2 * g], xcc = r[2 * g + 1], unit = g / 8 % 38;
    wrong += xcc != g % 8;
    wrong += (hw >> 8 & 0xF) != unit / 4 || (hw >> 13 & 3) != unit % 4 || (hw >> 12 & 1) != 0 || (hw & 0xF) != 0;
    smid.insert((xcc << 2 | (hw >> 13 & 3)) << 4 | (hw >> 8 & 0xF));
  }
  VCHECK_EQ(wrong, 0);
  VCHECK_EQ(smid.size(), size_t{304});
}

// An RX 7900 XTX: 6 shader engines of 2 arrays, 48 workgroup processors.
// HW_ID1 says the processor (13:10), array (16) and engine (20:18), and HIP's
// __smid tells all 48 apart.
VTEST(rdna3_hw_id1_says_which_workgroup_processor_a_wave_runs_on) {
  const std::vector<uint32_t> r = where(object("asm_hwid11", "gfx1100"), 96, {1, 6, 2, 48}, 32, 1);
  int wrong = 0;
  std::set<uint32_t> smid;
  for (uint32_t g = 0; g < 96; ++g) {
    const uint32_t hw = r[g], unit = g % 48;
    wrong += (hw >> 18 & 7) != unit % 6 || (hw >> 16 & 1) != unit / 6 % 2 || (hw >> 10 & 0xF) != unit / 12;
    wrong += (hw & 0x1F) != 0;
    smid.insert(((hw >> 18 & 7) << 1 | (hw >> 16 & 1)) << 4 | (hw >> 10 & 0xF));
  }
  VCHECK_EQ(wrong, 0);
  VCHECK_EQ(smid.size(), size_t{48});
}

// An RX 9070 XT: 4 engines of 2 arrays, 32 workgroup processors, the same
// HW_ID1. Register 4 is STATE_PRIV on gfx12, whose bit 9 is SCC.
VTEST(rdna4_hw_id1_says_where_a_wave_runs_and_state_priv_holds_scc) {
  const std::vector<uint32_t> r = where(object("asm_hwid12", "gfx1201"), 64, {1, 4, 2, 32}, 32, 3);
  int wrong = 0;
  std::set<uint32_t> units;
  for (uint32_t g = 0; g < 64; ++g) {
    const uint32_t hw = r[3 * g], unit = g % 32;
    wrong += (hw >> 18 & 7) != unit % 4 || (hw >> 16 & 1) != unit / 4 % 2 || (hw >> 10 & 0xF) != unit / 8;
    wrong += r[3 * g + 1] != 1u << 9 || r[3 * g + 2] != 0;
    units.insert(hw >> 10);
  }
  VCHECK_EQ(wrong, 0);
  VCHECK_EQ(units.size(), size_t{32});
}

// Sixteen bytes stored with global_store_dwordx4 and loaded with
// global_load_dwordx4 by work-groups on other host threads: each load sees
// all four words old or all four new. Moved as two 8-byte halves, about one
// load in a thousand took one half old and one new, and rocPRIM's look-back,
// which packs a tile's flag and 64-bit prefix this way, then read a prefix
// several tiles short (hipCUB's DeviceSelect::If). The memory has a fault
// hook attached and none armed, as the runtime leaves every device's.
struct UnarmedFaults : MemoryManager::AccessFault {
  uint64_t none = 0;
  UnarmedFaults() { copy_pending = stuck_pending = alu_pending = load_pending = store_pending = shared_pending = &none; }
  uint64_t on_load(uint64_t, uint32_t, uint64_t v) override { return v; }
  uint64_t on_store(uint64_t, uint32_t, uint64_t v) override { return v; }
  uint64_t on_shared_load(uint64_t, uint32_t, uint64_t v) override { return v; }
  void on_read(uint64_t, uint8_t*, uint64_t) override {}
  uint64_t on_alu(uint64_t v, uint32_t) override { return v; }
  void on_copy(uint64_t, uint8_t*, uint64_t) override {}
};

VTEST(a_16_byte_store_and_load_are_each_one_access_across_threads) {
  const amd::CodeObject o = object("asm_quad");
  MemoryManager mem(16ull << 20);
  UnarmedFaults faults;
  mem.set_access_fault(&faults);
  const uint64_t data = mem.alloc(16), torn = mem.alloc(16);
  const amd::Kernel* k = amd::find_kernel(o, "quads");
  VCHECK(k != nullptr);
  if (!k) return;
  std::vector<uint8_t> args(k->kernarg_size, 0);
  for (int b = 0; b < 8; ++b) args[b] = static_cast<uint8_t>(data >> (8 * b)), args[8 + b] = static_cast<uint8_t>(torn >> (8 * b));
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(args.size());
  mem.write(d.kernarg, args.data(), args.size());
  d.groups[0] = 4;   // a writer and three readers, each on a host thread of its own
  d.group_size[0] = 64;
  const char* was = std::getenv("VGPU_THREADS");
  const std::string saved = was ? was : "";
  setenv("VGPU_THREADS", "4", 1);
  amd::execute(d, mem);
  if (was) setenv("VGPU_THREADS", saved.c_str(), 1);
  else unsetenv("VGPU_THREADS");
  uint32_t r[4] = {}, q[4] = {};
  mem.read(torn, r, 16);
  mem.read(data, q, 16);
  VCHECK_EQ(r[0], 200000u);   // every store made
  VCHECK_EQ(q[0], 200000u);
  VCHECK_EQ(r[1] + r[2] + r[3], 0u);
}

// The shader clock as clock() reads it on RDNA: SHADER_CYCLES' 20 bits on
// gfx11, and on gfx12 a low and a high word, read high, low, high.
VTEST(rdna_shader_cycles_count_up) {
  MemoryManager mem(16ull << 20);
  const uint64_t out = mem.alloc(16);
  const std::vector<uint32_t> r = run(object("asm_wave", "gfx1100"), "shader_cycles", mem, out, 2, {out});
  VCHECK(r[0] <= 0xFFFFF && r[1] <= 0xFFFFF);
  const uint32_t ticks = (r[1] - r[0]) & 0xFFFFF;
  VCHECK(ticks > 0 && ticks < 64);
  const std::vector<uint32_t> q = run(object("asm_cycles", "gfx1201"), "shader_cycles", mem, out, 4, {out});
  VCHECK_EQ(q[0], q[2]);   // the high word held still
  VCHECK(q[3] - q[1] > 0 && q[3] - q[1] < 64);
}

// RDNA4's scalar half instructions read an inline float constant as the
// half's own encoding (2.0 is 0x4000): hip-tests' __halfMath on gfx12 got
// 3.0 for 1.0 * 2.0 + 3.0 when it was read as a float's bits.
VTEST(rdna4_scalar_half_instructions_read_inline_constants_as_halves) {
  MemoryManager mem(16ull << 20);
  const uint64_t out = mem.alloc(16);
  const std::vector<uint32_t> r = run(object("asm_salu_f16", "gfx1201"), "salu_half", mem, out, 4, {out});
  VCHECK_EQ(r[0], 0x4500u);   // 5.0
  VCHECK_EQ(r[1], 0x4000u);   // 2.0
  VCHECK_EQ(r[2], 0x3800u);   // 0.5
  VCHECK_EQ(r[3], 1u);
}

// RDNA's float atomic min and max in global memory (gfx11's
// global_atomic_min_f32, gfx12's global_atomic_min_num_f32), returning what
// they found: hip-tests' float atomicMin/atomicMax compile to them on gfx12.
VTEST(rdna_float_atomic_min_and_max_in_global_memory) {
  for (const auto& [name, target] : {std::pair{"asm_fminmax11", "gfx1100"}, std::pair{"asm_fminmax12", "gfx1201"}}) {
    MemoryManager mem(16ull << 20);
    const uint64_t out = mem.alloc(16);
    const float three[2] = {3.0f, 3.0f};
    mem.write(out, three, 8);
    const std::vector<uint32_t> r = run(object(name, target), "fminmax", mem, out, 4, {out});
    VCHECK_EQ(r[0], f(1.0f));
    VCHECK_EQ(r[1], f(5.0f));
    VCHECK_EQ(r[2], f(3.0f));
    VCHECK_EQ(r[3], f(3.0f));
  }
}

// RDNA's packed 16-bit instructions take a literal as the whole pair: the
// high result reads its top 16 bits (hipRTC's fp16 header test builds
// v_dot2_f32_f16 with 0x42004200, 3.0 in both halves).
VTEST(rdna_packed_literals_give_each_half_its_own_16_bits) {
  MemoryManager mem(16ull << 20);
  const uint64_t out = mem.alloc(8);
  const std::vector<uint32_t> r = run(object("asm_literal", "gfx1100"), "packed_literals", mem, out, 2, {out});
  VCHECK_EQ(r[0], f(9.0f));
  VCHECK_EQ(r[1], 0x46004400u);
}

// Code built for an RDNA generic target (gfx11-generic, gfx12-generic) runs
// as its family's: hip-tests' hipModuleLoadFatBinary loads such bundles, and
// they ran as CDNA code before.
VTEST(rdna_generic_code_objects_run_as_their_family) {
  for (const char* target : {"gfx11-generic", "gfx12-generic"}) {
    const amd::CodeObject o = object("vector_add", target);
    VCHECK(amd::gcn::is_rdna(amd::gcn::target_of_mach(o.mach)));
    MemoryManager mem(16ull << 20);
    std::vector<float> a(64), b(64);
    for (int i = 0; i < 64; ++i) a[i] = static_cast<float>(i), b[i] = 0.5f * static_cast<float>(i);
    const uint64_t pa = mem.alloc(256), pb = mem.alloc(256), out = mem.alloc(256);
    mem.write(pa, a.data(), 256);
    mem.write(pb, b.data(), 256);
    const std::vector<uint32_t> r = run(o, "vector_add", mem, out, 64, {pa, pb, out, 64});
    int wrong = 0;
    for (int i = 0; i < 64; ++i) wrong += r[i] != f(1.5f * static_cast<float>(i));
    VCHECK_EQ(wrong, 0);
  }
}

// A kernel that ends where it begins, launched over 64 million work-groups,
// as hip-tests launches its NOPKernel over the largest grids there are: one
// group runs, and the dispatch counts what it did once for each group.
VTEST(an_empty_kernel_over_a_huge_grid_counts_every_wave_without_running_each) {
  const amd::CodeObject o = object("asm_wave", "gfx1100");
  MemoryManager mem(16ull << 20);
  const amd::Kernel* k = amd::find_kernel(o, "empty");
  VCHECK(k != nullptr);
  if (!k) return;
  const auto launch = [&](uint32_t groups) {
    amd::Dispatch d;
    d.object = &o;
    d.kernel = k;
    d.groups[0] = groups;
    d.group_size[0] = 64;
    d.wave_size = 32;
    return amd::execute(d, mem);
  };
  const amd::DispatchStats one = launch(1);
  const auto start = std::chrono::steady_clock::now();
  const amd::DispatchStats many = launch(1u << 26);
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  VCHECK_EQ(one.waves, 2u);
  VCHECK_EQ(many.waves, one.waves << 26);
  VCHECK_EQ(many.instructions, one.instructions << 26);
  VCHECK_EQ(many.counts.salu, one.counts.salu << 26);
  VCHECK_EQ(many.waves_lt64, one.waves_lt64 << 26);
  VCHECK(seconds < 5.0);   // 64 million groups, each run, take minutes
}

VTEST(lds_64_bit_read_modify_writes_and_the_scalar_bit_operations_compute_what_the_isa_says) {
  const amd::CodeObject o = object("asm_lds64");
  MemoryManager mem(16ull << 20);
  // a and b per lane: doubles, so the floating-point forms see numbers, whose
  // bits the integer forms take as they are; some lanes a == b, some a > b.
  std::vector<uint64_t> in(128);
  for (uint32_t l = 0; l < 64; ++l) {
    const double a = static_cast<double>(l) * 1.25 - 20.0, b = l % 7 == 0 ? a : static_cast<double>(63 - l) * 0.5;
    std::memcpy(&in[2 * l], &a, 8);
    std::memcpy(&in[2 * l + 1], &b, 8);
  }
  const uint64_t in_d = mem.alloc(in.size() * 8), out = mem.alloc(64 * 68 * 8);
  mem.write(in_d, in.data(), in.size() * 8);
  const std::vector<uint32_t> r = run(o, "lds64", mem, out, 64 * 68 * 2, {in_d, out});
  const auto dbl = [](uint64_t x) {
    double d;
    std::memcpy(&d, &x, 8);
    return d;
  };
  const auto bits = [](double d) {
    uint64_t x;
    std::memcpy(&x, &d, 8);
    return x;
  };
  const char* names[] = {"add_u64", "sub_u64", "rsub_u64", "inc_u64", "dec_u64", "min_i64", "max_i64", "min_u64",
                         "max_u64", "and_b64", "or_b64", "xor_b64", "add_f64", "min_f64", "max_f64"};
  int wrong = 0;
  for (uint32_t l = 0; l < 64; ++l) {
    const uint64_t a = in[2 * l], b = in[2 * l + 1];
    const auto at = [&](uint32_t word) { return uint64_t{r[(68 * l + word) * 2]} | uint64_t{r[(68 * l + word) * 2 + 1]} << 32; };
    for (uint32_t k = 0; k < 30; ++k) {
      const std::string what = k < 15 ? names[k] : k == 27 ? "wrxchg_b64" : k < 27 ? names[k - 15] : k == 28 ? "min_f64" : "max_f64";
      uint64_t now = a;
      if (what == "add_u64") now = a + b;
      else if (what == "sub_u64") now = a - b;
      else if (what == "rsub_u64") now = b - a;
      else if (what == "inc_u64") now = a >= b ? 0 : a + 1;
      else if (what == "dec_u64") now = a == 0 || a > b ? b : a - 1;
      else if (what == "min_i64") now = static_cast<uint64_t>(std::min(static_cast<int64_t>(a), static_cast<int64_t>(b)));
      else if (what == "max_i64") now = static_cast<uint64_t>(std::max(static_cast<int64_t>(a), static_cast<int64_t>(b)));
      else if (what == "min_u64") now = std::min(a, b);
      else if (what == "max_u64") now = std::max(a, b);
      else if (what == "and_b64") now = a & b;
      else if (what == "or_b64") now = a | b;
      else if (what == "xor_b64") now = a ^ b;
      else if (what == "wrxchg_b64") now = b;
      else if (what == "add_f64") now = bits(dbl(a) + dbl(b));
      else if (what == "min_f64") now = bits(std::fmin(dbl(a), dbl(b)));
      else if (what == "max_f64") now = bits(std::fmax(dbl(a), dbl(b)));
      const uint64_t handed_back = k >= 15 ? a : 0;
      wrong += at(2 * k) != handed_back;
      wrong += at(2 * k + 1) != now;
    }
    // The scalar results, on lane 0's a, the same in every lane.
    const uint64_t s = in[0];
    uint64_t brev = 0, wqm64 = 0;
    uint32_t wqm32 = 0;
    for (int i = 0; i < 64; ++i) brev |= ((s >> i) & 1) << (63 - i);
    for (int q = 0; q < 16; ++q)
      if ((s >> (4 * q)) & 0xF) wqm64 |= uint64_t{0xF} << (4 * q);
    for (int q = 0; q < 8; ++q)
      if ((s >> (4 * q)) & 0xF) wqm32 |= 0xFu << (4 * q);
    const uint32_t lo = static_cast<uint32_t>(s);
    const uint32_t want[14] = {static_cast<uint32_t>(brev), static_cast<uint32_t>(brev >> 32),
                               static_cast<uint32_t>(64 - __builtin_popcountll(s)), static_cast<uint32_t>(32 - __builtin_popcount(lo)),
                               ~lo ? static_cast<uint32_t>(__builtin_ctz(~lo)) : 0xFFFFFFFFu,
                               ~s ? static_cast<uint32_t>(__builtin_ctzll(~s)) : 0xFFFFFFFFu,
                               static_cast<uint32_t>(wqm64), static_cast<uint32_t>(wqm64 >> 32), wqm32,
                               lo, static_cast<uint32_t>(s >> 32), 0, 0, 0};
    for (int i = 0; i < 14; ++i) wrong += r[(68 * l + 60) * 2 + i] != want[i];
  }
  VCHECK_EQ(wrong, 0);
}

VTEST(lds_float_atomics_and_compare_and_stores_compute_what_the_isa_says) {
  const amd::CodeObject o = object("asm_ldsf32");
  MemoryManager mem(16ull << 20);
  std::vector<uint32_t> in(128);
  for (uint32_t l = 0; l < 64; ++l) {
    in[2 * l] = f(static_cast<float>(l) * 0.75f - 12.0f);
    in[2 * l + 1] = f(l % 5 == 0 ? static_cast<float>(l) * 0.75f - 12.0f : static_cast<float>(40 - l) * 0.5f);
  }
  const uint64_t in_d = mem.alloc(in.size() * 4), out = mem.alloc(64 * 24 * 4);
  mem.write(in_d, in.data(), in.size() * 4);
  const std::vector<uint32_t> r = run(o, "ldsf32", mem, out, 64 * 24, {in_d, out});
  const auto fl = [](uint32_t x) {
    float v;
    std::memcpy(&v, &x, 4);
    return v;
  };
  int wrong = 0;
  for (uint32_t l = 0; l < 64; ++l) {
    const uint32_t a = in[2 * l], b = in[2 * l + 1];
    const uint32_t sum = f(fl(a) + fl(b)), lo = f(std::fmin(fl(a), fl(b))), hi = f(std::fmax(fl(a), fl(b)));
    const uint32_t keep_or_b = fl(b) == fl(b) && fl(a) == fl(b) ? b : a;   // cmpst_f32 against b
    const uint32_t want[18] = {0, sum, a, sum, 0, lo, a, lo, 0, hi, a, hi,
                               0, b,                // cmpst_b32 against a: b stored
                               0, keep_or_b,        // cmpst_f32 against b: a kept unless a == b
                               a, b};               // cmpst_rtn_f32 against a: a handed back, b stored
    const uint32_t* got = &r[24 * l];
    for (int i = 0; i < 18; ++i) wrong += got[i] != want[i];
    uint64_t pair = uint64_t{a} | uint64_t{b} << 32;
    double d;
    std::memcpy(&d, &pair, 8);
    d += d;
    uint64_t twice;
    std::memcpy(&twice, &d, 8);
    wrong += got[18] != a || got[19] != b;   // ds_add_rtn_f64 hands back the pair
    wrong += got[20] != static_cast<uint32_t>(twice) || got[21] != static_cast<uint32_t>(twice >> 32);
  }
  VCHECK_EQ(wrong, 0);
}

VTEST(each_byte_of_a_word_converts_to_its_own_float) {
  // v_cvt_f32_ubyte1 and 2 read byte 3 until the byte was taken from the
  // right place in the name: uchar2's divide came out with a zero .y.
  const amd::CodeObject o = object("asm_cvt_ubyte");
  MemoryManager mem(16ull << 20);
  std::vector<uint32_t> in(64);
  for (uint32_t l = 0; l < 64; ++l) in[l] = 0x01020304u * (l + 1) ^ (l << 11);
  const uint64_t in_d = mem.alloc(64 * 4), out = mem.alloc(64 * 32);
  mem.write(in_d, in.data(), 64 * 4);
  const std::vector<uint32_t> r = run(o, "cvt_ubyte", mem, out, 64 * 8, {in_d, out});
  int wrong = 0;
  for (uint32_t l = 0; l < 64; ++l)
    for (uint32_t k = 0; k < 4; ++k) {
      wrong += r[8 * l + k] != f(static_cast<float>((in[l] >> (8 * k)) & 0xFF));
      wrong += r[8 * l + 4 + k] != f(static_cast<float>((in[0] >> (8 * k)) & 0xFF));
    }
  VCHECK_EQ(wrong, 0);
}

VTEST(a_half_dot_product_into_a_float_is_not_held_to_one_by_its_clamp) {
  // amd_mixed_dot({1, 3}, {3, 3}, 2, true) is 14 on a card; the clamp that
  // holds other float results to [0, 1] does not hold this one.
  const amd::CodeObject o = object("asm_dot_clamp");
  MemoryManager mem(16ull << 20);
  const auto h2 = [](float lo, float hi) {
    const _Float16 a = static_cast<_Float16>(lo), b = static_cast<_Float16>(hi);
    uint16_t x, y;
    std::memcpy(&x, &a, 2);
    std::memcpy(&y, &b, 2);
    return uint32_t{x} | uint32_t{y} << 16;
  };
  std::vector<uint32_t> in(64 * 3);
  std::vector<float> al(64), ah(64), bl(64), bh(64), c(64);
  for (uint32_t l = 0; l < 64; ++l) {
    al[l] = static_cast<float>(l % 7) - 2.0f, ah[l] = static_cast<float>(l % 5) + 0.5f;
    bl[l] = static_cast<float>(l % 3) + 1.0f, bh[l] = static_cast<float>(l % 11) - 4.0f;
    c[l] = static_cast<float>(l) * 0.25f - 3.0f;
    in[3 * l] = h2(al[l], ah[l]);
    in[3 * l + 1] = h2(bl[l], bh[l]);
    in[3 * l + 2] = f(c[l]);
  }
  const uint64_t in_d = mem.alloc(in.size() * 4), out = mem.alloc(64 * 32);
  mem.write(in_d, in.data(), in.size() * 4);
  const std::vector<uint32_t> r = run(o, "dot_clamp", mem, out, 64 * 8, {in_d, out});
  int wrong = 0;
  for (uint32_t l = 0; l < 64; ++l) {
    const auto dot = [&](float a0, float a1, float b0, float b1, float cc) {
      return f(static_cast<float>(double(a0) * b0 + double(a1) * b1 + cc));
    };
    const uint32_t want[8] = {
        dot(al[l], ah[l], bl[l], bh[l], c[l]),  dot(al[l], ah[l], bl[l], bh[l], c[l]),
        dot(-al[l], ah[l], bl[l], bh[l], c[l]), dot(al[l], ah[l], bl[l], -bh[l], c[l]),
        dot(al[l], ah[l], bl[l], bh[l], -c[l]), dot(-al[l], ah[l], bl[l], bh[l], c[l]),
        dot(al[l], ah[l], bl[l], bh[l], c[l]),  dot(al[l], -ah[l], -bl[l], bh[l], c[l])};
    for (int i = 0; i < 8; ++i) wrong += r[8 * l + i] != want[i];
  }
  VCHECK_EQ(wrong, 0);
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
  const uint64_t out = mem.alloc(64 * 4);
  const std::vector<uint32_t> r = run(o, "libs", mem, out, 64, {out});
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
  VCHECK_EQ(r[59], 5u);           // a store through a resource with no data format does not land
  VCHECK_EQ(r[60], 0u);           // and a load through it reads 0
  VCHECK_EQ(r[61], 7u);           // 5 + 3, then decremented
  VCHECK_EQ(r[62], 5u);           // what the add found
  VCHECK_EQ(r[63], 8u);           // and what the decrement found
}

// s_memrealtime is the real-time clock at 100 MHz, the rate the runtime
// reports as hipDeviceAttributeWallClockRate: a kernel that spins on
// wall_clock64() for some milliseconds, as hip-tests' delay kernels do, has to
// see time pass at that rate. It read the instruction count before, which
// ran far slower than the rate said. s_memtime stays the instruction count,
// this model's shader clock.
VTEST(the_real_time_clock_counts_at_the_wall_clock_rate) {
  const amd::CodeObject o = object("asm_realtime");
  const auto now = [] {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count()) /
           10;
  };
  MemoryManager mem(16ull << 20);
  const uint64_t out = mem.alloc(4 * 4);
  const uint64_t before = now();
  const std::vector<uint32_t> r = run(o, "realtime", mem, out, 4, {out});
  const uint64_t after = now();
  const uint64_t realtime = r[0] | uint64_t{r[1]} << 32, shader = r[2] | uint64_t{r[3]} << 32;
  VCHECK(realtime >= before);
  VCHECK(realtime <= after);
  // The instructions the thread running the wave has retired: a count far
  // below a clock that has been going since the host started.
  VCHECK(shader < realtime / 1000);
}

// RDNA's image instructions and formatted buffer loads (asm_images.s) on
// resources written here as the runtime writes a texture's: each result
// against the texel arithmetic done on the host.
VTEST(rdna_images_load_store_sample_gather_and_query_what_their_resources_describe) {
  namespace im = amd::image;
  const amd::CodeObject o = object("asm_images", "gfx1100");
  constexpr im::Gen G = im::Gen::Gfx11;
  MemoryManager mem(16ull << 20);
  const auto t0 = [](int64_t x, int64_t y) {
    x = std::clamp<int64_t>(x, 0, 7), y = std::clamp<int64_t>(y, 0, 3);
    return static_cast<float>(x + 10 * y) + 0.25f;
  };
  std::vector<float> f0(32);
  for (uint32_t i = 0; i < 32; ++i) f0[i] = t0(i % 8, i / 8);
  std::vector<uint8_t> rgba(4 * 8), elems(4 * 32);
  for (uint32_t i = 0; i < 8; ++i) {
    const uint8_t b[4] = {static_cast<uint8_t>((i % 4) * 40), static_cast<uint8_t>((i / 4) * 100), 7, 255};
    std::memcpy(&rgba[4 * i], b, 4);
  }
  for (uint32_t i = 0; i < 32; ++i) {
    const uint8_t b[4] = {static_cast<uint8_t>(i), static_cast<uint8_t>(2 * i), static_cast<uint8_t>(3 * i),
                          static_cast<uint8_t>(255 - i)};
    std::memcpy(&elems[4 * i], b, 4);
  }
  std::vector<float> mips(16 + 4);
  for (uint32_t i = 0; i < 20; ++i) mips[i] = i < 16 ? 1.0f : 2.0f;
  const uint64_t d0 = mem.alloc(32 * 4), d1 = mem.alloc(rgba.size()), d2 = mem.alloc(32 * 4),
                 db = mem.alloc(elems.size()), d3 = mem.alloc(mips.size() * 4);
  mem.write(d0, f0.data(), 32 * 4);
  mem.write(d1, rgba.data(), rgba.size());
  const std::vector<uint32_t> zero(32, 0);
  mem.write(d2, zero.data(), 32 * 4);
  mem.write(db, elems.data(), elems.size());
  mem.write(d3, mips.data(), mips.size() * 4);

  uint32_t desc[48] = {};
  im::Image i0;
  i0.base = d0, i0.width = 8, i0.height = 4, i0.format = {im::Data::D32, im::Num::Float};
  im::encode(i0, G, &desc[0]);
  im::Image i1;
  i1.base = d1, i1.width = 4, i1.height = 2, i1.format = {im::Data::D8_8_8_8, im::Num::Unorm};
  im::encode(i1, G, &desc[8]);
  im::Image i2 = i0;
  i2.base = d2, i2.format = {im::Data::D32, im::Num::Uint};
  im::encode(i2, G, &desc[16]);
  im::Sampler point, linear, mip;
  im::encode(point, G, &desc[24]);
  linear.mag_linear = linear.min_linear = true;
  im::encode(linear, G, &desc[28]);
  im::encode_buffer(db, 4, 32, {im::Data::D8_8_8_8, im::Num::Unorm}, G, &desc[32]);
  im::Image i3;
  i3.base = d3, i3.width = 4, i3.height = 4, i3.last_level = 1, i3.format = {im::Data::D32, im::Num::Float};
  im::encode(i3, G, &desc[36]);
  mip.mip_filter = 2;
  im::encode(mip, G, &desc[44]);
  // The resources round-trip, and T3's level 1 sits right after level 0.
  VCHECK_EQ(im::decode_image(&desc[8], G).width, 4u);
  VCHECK(im::decode_image(&desc[8], G).format.data == im::Data::D8_8_8_8);
  VCHECK_EQ(im::texel_offset(i3, 1, 0, 0, 0), uint64_t{64});
  VCHECK(im::decode_sampler(&desc[28], G).mag_linear);
  const uint64_t ddesc = mem.alloc(sizeof(desc)), out = mem.alloc(800 * 4);
  mem.write(ddesc, desc, sizeof(desc));

  const amd::Kernel* k = amd::find_kernel(o, "images");
  VCHECK(k != nullptr);
  if (!k) return;
  std::vector<uint8_t> args(k->kernarg_size, 0);
  std::memcpy(&args[0], &out, 8);
  std::memcpy(&args[8], &ddesc, 8);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(args.size());
  mem.write(d.kernarg, args.data(), args.size());
  d.group_size[0] = 32;
  amd::execute(d, mem);
  std::vector<uint32_t> r(800);
  mem.read(out, r.data(), r.size() * 4);
  const auto f = [&](uint32_t at) {
    float v;
    std::memcpy(&v, &r[at], 4);
    return v;
  };
  const auto half = [](float v) {
    const _Float16 h = static_cast<_Float16>(v);
    uint16_t b;
    std::memcpy(&b, &h, 2);
    return uint32_t{b};
  };
  int wrong[13] = {};
  for (uint32_t l = 0; l < 32; ++l) {
    const int64_t x = l % 8, y = l / 8;
    wrong[0] += f(l) != t0(x, y);
    wrong[1] += f(32 + l) != t0(x, y);
    wrong[2] += f(64 + l) != (x == 0 ? t0(0, y) : t0(x, y) - 0.5f);
    const uint32_t t = (l % 4) + 4 * ((l / 4) % 2);
    for (uint32_t c = 0; c < 4; ++c) wrong[3] += std::fabs(f(96 + 4 * l + c) - rgba[4 * t + c] / 255.0f) > 1e-6f;
    const uint32_t level = l % 2;
    wrong[4] += r[224 + 4 * l] != (4u >> level) || r[225 + 4 * l] != (4u >> level) || r[226 + 4 * l] != 1 ||
                r[227 + 4 * l] != 2;
    const float g[4] = {t0(x, y + 1), t0(x + 1, y + 1), t0(x + 1, y), t0(x, y)};
    for (uint32_t c = 0; c < 4; ++c) wrong[5] += f(352 + 4 * l + c) != g[c];
    for (uint32_t c = 0; c < 4; ++c) wrong[6] += std::fabs(f(480 + 4 * l + c) - elems[4 * l + c] / 255.0f) > 1e-6f;
    wrong[7] += r[608 + l] != 0;
    wrong[8] += f(640 + l) != t0(x, y);
    const double lod = std::min(1.0, static_cast<double>(static_cast<float>(l) * 0.1f));
    wrong[9] += std::fabs(f(672 + l) - static_cast<float>(1.0 + lod)) > 1e-6f;
    wrong[10] += r[704 + l] != (half(rgba[4 * t] / 255.0f) | half(rgba[4 * t + 1] / 255.0f) << 16);
    wrong[11] += r[736 + l] != 3 * l;
    float e;
    std::memcpy(&e, &elems[4 * l], 4);
    wrong[12] += std::memcmp(&r[768 + l], &e, 4) != 0;
  }
  for (int c = 0; c < 13; ++c) VCHECK_EQ(wrong[c], 0);
  std::vector<uint32_t> stored(32);
  mem.read(d2, stored.data(), 32 * 4);
  int bad = 0;
  for (uint32_t l = 0; l < 32; ++l) bad += stored[l] != 3 * l + 5;
  VCHECK_EQ(bad, 0);
}

VTEST(the_24_bit_multiplies_take_the_low_24_bits_signed_or_not_and_give_the_low_or_the_high_half) {
  // v_mul_hi_i32_i24 (VOP2 0x07) was not decoded: llama.cpp's mul_mat_vec_q for Q6_K stopped on it.
  const amd::CodeObject o = object("asm_mulhi");
  MemoryManager mem(16ull << 20);
  std::vector<uint32_t> in(64 * 2);
  uint32_t seed = 77;
  const auto next = [&] {
    seed = seed * 1103515245u + 12345u;
    return seed;
  };
  for (uint32_t l = 0; l < 64; ++l) {
    in[2 * l] = l == 0 ? 0x7fffffu : l == 1 ? 0x800000u : l == 2 ? 0xffffffffu : next();   // the extremes, then noise
    in[2 * l + 1] = l == 0 ? 0x7fffffu : l == 1 ? 0x800000u : l == 2 ? 0x00000002u : next();
  }
  const uint64_t in_d = mem.alloc(in.size() * 4), out = mem.alloc(64 * 5 * 4);
  mem.write(in_d, in.data(), in.size() * 4);
  const std::vector<uint32_t> r = run(o, "mulhi", mem, out, 64 * 5, {in_d, out});
  const auto i24 = [](uint32_t v) { return static_cast<int64_t>(static_cast<int32_t>(v << 8) >> 8); };
  const auto u24 = [](uint32_t v) { return static_cast<uint64_t>(v & 0xffffff); };
  int wrong = 0;
  for (uint32_t l = 0; l < 64; ++l) {
    const uint32_t a = in[2 * l], b = in[2 * l + 1];
    const int64_t sp = i24(a) * i24(b);
    const uint64_t up = u24(a) * u24(b);
    wrong += r[5 * l + 0] != u(sp) + 0;
    wrong += r[5 * l + 1] != u(sp >> 32);
    wrong += r[5 * l + 2] != static_cast<uint32_t>(up);
    wrong += r[5 * l + 3] != static_cast<uint32_t>(up >> 32);
    wrong += r[5 * l + 4] != u(sp >> 32);
  }
  VCHECK_EQ(wrong, 0);
}

VTEST(vop3_instructions_a_compiler_emits_give_what_the_isa_says) {
  // rocPRIM's merge sort stopped on v_med3_u16 (VOP3 0x1fc); the whole range around it was missing: the
  // 16-bit median, three-way and multiply-add, v_max3_u32, v_alignbyte_b32, the sums of absolute
  // differences (and the quad forms over a 64-bit window), v_lerp_u8, and the packed conversions.
  const amd::CodeObject o = object("asm_vop3_gaps");
  MemoryManager mem(16ull << 20);
  std::vector<uint32_t> in(64 * 8);
  std::vector<float> fin(64 * 3);
  uint32_t seed = 31337;
  const auto next = [&] {
    seed = seed * 1103515245u + 12345u;
    return seed;
  };
  // Halves that are exact, so the float and half arithmetic agree; infinity and zero for the DX9 multiply.
  const float table[] = {0.25f, -0.5f, 1.5f, 3.0f, -2.0f, 0.0f, 300.0f, -7.0f, 0.75f, 100.0f, 2.0f, 1.0f, -1.0f};
  const float inf = std::numeric_limits<float>::infinity();
  for (uint32_t l = 0; l < 64; ++l) {
    for (uint32_t k = 0; k < 8; ++k) in[8 * l + k] = next();
    for (uint32_t k = 0; k < 3; ++k) fin[3 * l + k] = table[(next() >> 8) % 13];
  }
  in[0] = 0x00FF0080; in[1] = 0x7FFF8000; in[2] = 0xFFFF0001;   // the edges of the 16-bit orders
  in[8] = 0; in[9] = 0xFFFFFFFF; in[10] = 0;
  fin[3 * 5] = inf; fin[3 * 5 + 1] = 0.0f;                        // zero times infinity
  fin[3 * 6] = 0.0f; fin[3 * 6 + 1] = inf;
  fin[3 * 7] = 5.0f; fin[3 * 7 + 1] = -3.0f;                      // beyond the normalized range
  const uint64_t in_d = mem.alloc(in.size() * 4), fin_d = mem.alloc(fin.size() * 4), out = mem.alloc(64 * 34 * 4);
  mem.write(in_d, in.data(), in.size() * 4);
  mem.write(fin_d, fin.data(), fin.size() * 4);
  const std::vector<uint32_t> r = run(o, "gaps", mem, out, 64 * 34, {in_d, fin_d, out});
  const auto half_bits = [](float f) {
    const _Float16 h = static_cast<_Float16>(f);
    uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
  };
  const auto half_val = [](float f) { return static_cast<float>(static_cast<_Float16>(f)); };
  const auto med = [](auto x, auto y, auto z) { return std::max(std::min(x, y), std::min(std::max(x, y), z)); };
  const auto sad = [](uint32_t a, uint32_t b, bool masked) {
    uint32_t sum = 0;
    for (int k = 0; k < 4; ++k) {
      const int x = a >> 8 * k & 0xFF, y = b >> 8 * k & 0xFF;
      if (!(masked && y == 0)) sum += std::abs(x - y);
    }
    return sum;
  };
  const auto snorm = [](double x) -> uint16_t {
    return std::isnan(x) ? 0 : static_cast<uint16_t>(static_cast<int16_t>(std::nearbyint(std::clamp(x, -1.0, 1.0) * 32767.0)));
  };
  const auto unorm = [](double x) -> uint16_t {
    return std::isnan(x) ? 0 : static_cast<uint16_t>(std::nearbyint(std::clamp(x, 0.0, 1.0) * 65535.0));
  };
  const auto both_nan = [](uint32_t a, uint32_t b) {
    const auto nan = [](uint32_t v) { return (v & 0x7C00) == 0x7C00 && (v & 0x3FF); };
    return nan(a) && nan(b);
  };
  std::string first;
  int wrong = 0;
  const auto check = [&](uint32_t lane, int slot, uint32_t got, uint32_t want, const char* what) {
    if (got == want) return;
    if (++wrong == 1) {
      char buf[160];
      std::snprintf(buf, sizeof buf, "%s: lane %u got 0x%08x, wanted 0x%08x", what, lane, got, want);
      first = buf;
    }
    (void)slot;
  };
  for (uint32_t l = 0; l < 64; ++l) {
    const uint32_t* w = &in[8 * l];
    const float f0 = fin[3 * l], f1 = fin[3 * l + 1], f2 = fin[3 * l + 2];
    const float h0 = half_val(f0), h1 = half_val(f1), h2 = half_val(f2);
    const auto at = [&](int k) { return r[34 * l + k]; };
    const uint16_t u0 = w[0], u1 = w[1], u2 = w[2];
    const int16_t s0 = static_cast<int16_t>(u0), s1 = static_cast<int16_t>(u1), s2 = static_cast<int16_t>(u2);
    check(l, 0, at(0), med(u0, u1, u2), "v_med3_u16");
    check(l, 1, at(1), static_cast<uint16_t>(med(s0, s1, s2)), "v_med3_i16");
    check(l, 2, at(2), half_bits(med(h0, h1, h2)), "v_med3_f16");
    check(l, 3, at(3), half_bits(std::min({h0, h1, h2})), "v_min3_f16");
    check(l, 4, at(4), half_bits(std::max({h0, h1, h2})), "v_max3_f16");
    {
      const uint32_t want = half_bits(h0 * h1 + h2);
      if (!both_nan(at(5), want)) check(l, 5, at(5), want, "v_mad_f16");
    }
    check(l, 6, at(6), static_cast<uint16_t>(uint32_t{u0} * u1 + u2), "v_mad_u16");   // unsigned, or two uint16_t promote to a signed int that overflows
    check(l, 7, at(7), static_cast<uint16_t>(s0 * s1 + s2), "v_mad_i16");
    check(l, 8, at(8), std::max({w[0], w[1], w[2]}), "v_max3_u32");
    check(l, 9, at(9), static_cast<uint32_t>((uint64_t{w[0]} << 32 | w[1]) >> 8 * (w[2] & 3)), "v_alignbyte_b32");
    {
      uint32_t want = 0;
      for (int k = 0; k < 4; ++k) want |= (((w[0] >> 8 * k & 0xFF) + (w[1] >> 8 * k & 0xFF) + (w[2] >> 8 * k & 1)) >> 1) << 8 * k;
      check(l, 10, at(10), want, "v_lerp_u8");
    }
    check(l, 11, at(11), sad(w[0], w[1], false) + w[2], "v_sad_u8");
    check(l, 12, at(12), (sad(w[0], w[1], false) << 16) + w[2], "v_sad_hi_u8");
    {
      const auto d = [](uint32_t x, uint32_t y) { return x > y ? x - y : y - x; };
      check(l, 13, at(13), d(w[0] & 0xFFFF, w[1] & 0xFFFF) + d(w[0] >> 16, w[1] >> 16) + w[2], "v_sad_u16");
      check(l, 14, at(14), d(w[0], w[1]) + w[2], "v_sad_u32");
    }
    check(l, 15, at(15), sad(w[0], w[1], true) + w[2], "v_msad_u8");
    {
      const uint32_t byte = std::isnan(f0) || f0 <= 0 ? 0 : f0 >= 255 ? 255 : static_cast<uint32_t>(f0);
      const uint32_t shift = 8 * (w[1] & 3);
      check(l, 16, at(16), (w[2] & ~(0xFFu << shift)) | byte << shift, "v_cvt_pk_u8_f32");
    }
    check(l, 17, at(17), u(int32_t{s0} * s1 + static_cast<int32_t>(w[2])), "v_mad_i32_i16");
    {
      const auto sat = [](int32_t v) { return static_cast<uint16_t>(std::clamp<int32_t>(v, INT16_MIN, INT16_MAX)); };
      check(l, 18, at(18), sat(static_cast<int32_t>(w[0])) | uint32_t{sat(static_cast<int32_t>(w[1]))} << 16, "v_cvt_pk_i16_i32");
    }
    check(l, 19, at(19), snorm(f0) | uint32_t{snorm(f1)} << 16, "v_cvt_pknorm_i16_f32");
    check(l, 20, at(20), unorm(f0) | uint32_t{unorm(f1)} << 16, "v_cvt_pknorm_u16_f32");
    check(l, 21, at(21), snorm(h0) | uint32_t{snorm(h1)} << 16, "v_cvt_pknorm_i16_f16");
    check(l, 22, at(22), unorm(h0) | uint32_t{unorm(h1)} << 16, "v_cvt_pknorm_u16_f16");
    check(l, 23, at(23), static_cast<uint16_t>(s0 + s1), "v_add_i16");
    check(l, 24, at(24), static_cast<uint16_t>(s0 - s1), "v_sub_i16");
    check(l, 25, at(25), f0 == 0 || f1 == 0 ? 0u : [&] { const float p = f0 * f1; uint32_t b; std::memcpy(&b, &p, 4); return b; }(),
          "v_mul_legacy_f32");
    {
      const uint64_t a = uint64_t{w[1]} << 32 | w[0], acc = uint64_t{w[5]} << 32 | w[4];
      uint64_t q = 0, m = 0;
      uint32_t mq[4];
      for (int k = 0; k < 4; ++k) {
        q |= uint64_t{(sad(static_cast<uint32_t>(a >> 8 * k), w[2], false) + static_cast<uint32_t>(acc >> 16 * k)) & 0xFFFF} << 16 * k;
        m |= uint64_t{(sad(static_cast<uint32_t>(a >> 8 * k), w[2], true) + static_cast<uint32_t>(acc >> 16 * k)) & 0xFFFF} << 16 * k;
        mq[k] = sad(static_cast<uint32_t>(a >> 8 * k), w[2], true) + w[4 + k];
      }
      check(l, 26, at(26), static_cast<uint32_t>(q), "v_qsad_pk_u16_u8 (low)");
      check(l, 27, at(27), static_cast<uint32_t>(q >> 32), "v_qsad_pk_u16_u8 (high)");
      check(l, 28, at(28), static_cast<uint32_t>(m), "v_mqsad_pk_u16_u8 (low)");
      check(l, 29, at(29), static_cast<uint32_t>(m >> 32), "v_mqsad_pk_u16_u8 (high)");
      for (int k = 0; k < 4; ++k) check(l, 30 + k, at(30 + k), mq[k], "v_mqsad_u32_u8");
    }
  }
  if (wrong) std::fprintf(stderr, "%d wrong; first: %s\n", wrong, first.c_str());
  VCHECK_EQ(wrong, 0);
}

VTEST_MAIN
