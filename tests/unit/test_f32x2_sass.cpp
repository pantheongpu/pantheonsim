// FFMA2, FMUL2 and FADD2 (sm_100's packed fp32) in the SASS decoder and executor's gate, and the arithmetic both engines share.
//
//  * Every FFMA2/FMUL2/FADD2 line of the sm_100, sm_100a and sm_103a corpora (nvdisasm's own text for what nvcc emitted for
//    sm_100a, and for what CUTLASS's 77_blackwell_fmha uses) decodes, prints as nvdisasm printed it, and is one the executor
//    runs; a cubin of them has nothing sass::unsupported refuses -- before the decoder knew them it said
//    "SASS: unknown opcode 0xc49 (sm_100)" and 77_blackwell_fmha ran on its PTX.
//  * The forms whose meaning nothing here knows (.FMZ, both swizzle bits, an immediate that is not one float for both lanes)
//    decode and print but are refused by the executor, so the kernel stays on PTX rather than guessing.
//  * sm_120 has no such instructions: the same low opcode bits stay UVIMNMX there (0x4a).
//  * The arithmetic of vgpu/exec/f32x2.hpp, which the PTX interpreter's f32x2 and the SASS executor's FFMA2 both use, against
//    IEEE 754 worked out by hand: ties, the four rounding modes, overflow, flushing with .ftz, zeros.
// Executing the instructions (the swizzles, signs and operand forms) is nvidia/tests/e2e/packed_f32x2.cu, which needs nvcc.
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <cfenv>

#include "vgpu/error.hpp"
#include "vgpu/exec/f32x2.hpp"
#include "vgpu/sass/exec.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace vgpu::sass {
bool runs_instr(const Instr& ins);   // src/sass/exec.cpp
}

namespace {

struct Line {
  sass::Word w;
  uint64_t pc;
  std::string text;
};

bool is_packed(const std::string& text) {
  std::string t = text;
  if (!t.empty() && t[0] == '@') t = t.substr(t.find(' ') + 1);
  const std::string op = t.substr(0, t.find_first_of(" ."));
  return op == "FFMA2" || op == "FMUL2" || op == "FADD2";
}

std::vector<Line> packed_lines(const char* corpus) {
  std::ifstream in(std::string(VGPU_SOURCE_DIR) + "/nvidia/tests/data/sass/" + corpus + ".txt");
  if (!in) throw vtest::Failure(std::string("no corpus ") + corpus);
  std::vector<Line> out;
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    const size_t t1 = line.find('\t'), t2 = line.find('\t', t1 + 1);
    Line l;
    l.w.hi = std::stoull(line.substr(0, 16), nullptr, 16);
    l.w.lo = std::stoull(line.substr(16, 16), nullptr, 16);
    l.pc = std::stoull(line.substr(t1 + 1, t2 - t1 - 1), nullptr, 16);
    l.text = line.substr(t2 + 1);
    if (is_packed(l.text)) out.push_back(l);
  }
  return out;
}

template <class T>
void put(std::string& s, T v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof v);
}

// A cubin for sm_<sm> whose only code is .text.k holding `words` (as test_unsupported_memo builds them).
std::string make_cubin(int sm, const std::vector<sass::Word>& words) {
  const std::string names = std::string("\0.shstrtab\0.text.k\0", 19);
  std::string out(64, '\0');
  const uint64_t names_off = out.size();
  out += names;
  while (out.size() % 16) out.push_back('\0');
  const uint64_t text_off = out.size();
  for (const sass::Word& w : words) {
    put(out, w.lo);
    put(out, w.hi);
  }
  const uint64_t shoff = out.size();
  const auto shdr = [&](uint32_t name, uint32_t type, uint64_t off, uint64_t size) {
    put<uint32_t>(out, name);
    put<uint32_t>(out, type);
    put<uint64_t>(out, 0);
    put<uint64_t>(out, 0);
    put<uint64_t>(out, off);
    put<uint64_t>(out, size);
    put<uint32_t>(out, 0);
    put<uint32_t>(out, 0);
    put<uint64_t>(out, 16);
    put<uint64_t>(out, 0);
  };
  shdr(0, 0, 0, 0);
  shdr(1, 3, names_off, names.size());
  shdr(11, 1, text_off, 16 * words.size());
  out[0] = 0x7f;
  out[1] = 'E';
  out[2] = 'L';
  out[3] = 'F';
  out[4] = 2;
  out[5] = 1;
  out[6] = 1;
  out[7] = 0x41;
  out[8] = 8;
  const uint16_t type = 2, machine = 190, shentsize = 64, shnum = 3, shstrndx = 1;
  const uint32_t flags = 0x06000002u | (static_cast<uint32_t>(sm) << 8);
  std::memcpy(&out[16], &type, 2);
  std::memcpy(&out[18], &machine, 2);
  std::memcpy(&out[0x28], &shoff, 8);
  std::memcpy(&out[0x30], &flags, 4);
  std::memcpy(&out[0x3a], &shentsize, 2);
  std::memcpy(&out[0x3c], &shnum, 2);
  std::memcpy(&out[0x3e], &shstrndx, 2);
  return out;
}

std::string unsupported(int sm, const std::vector<sass::Word>& words) {
  const std::string c = make_cubin(sm, words);
  return sass::unsupported(reinterpret_cast<const uint8_t*>(c.data()), c.size());
}

bool mentions(const std::string& text, const std::string& what) { return text.find(what) != std::string::npos; }

// What the executor refuses on purpose, by the text nvdisasm printed (the other kind of refusal, below, prints like a run form).
bool unknown_meaning(const std::string& text) { return mentions(text, "FMZ") || mentions(text, "INVALID3"); }

}  // namespace

VTEST(the_corpora_hold_the_packed_forms_ptxas_emits) {
  for (const char* c : {"sm_100", "sm_100a", "sm_103a"}) {
    const std::vector<Line> v = packed_lines(c);
    size_t ffma2 = 0, fmul2 = 0, fadd2 = 0, ur = 0;
    for (const Line& l : v) {
      ffma2 += mentions(l.text, "FFMA2");
      fmul2 += mentions(l.text, "FMUL2");
      fadd2 += mentions(l.text, "FADD2");
      ur += mentions(l.text, "UR");
    }
    VCHECK(ffma2 >= 10);
    VCHECK(fmul2 >= 10);
    VCHECK(fadd2 >= 10);
    VCHECK(ur >= 3);
  }
}

VTEST(every_packed_line_decodes_prints_as_nvdisasm_did_and_is_run) {
  for (const char* c : {"sm_100", "sm_100a", "sm_103a"}) {
    const int sm = std::atoi(c + 3);
    size_t n = 0;
    for (const Line& l : packed_lines(c)) {
      const sass::Instr ins = sass::decode(l.w, l.pc, sm);
      VCHECK_EQ(sass::to_text(ins), l.text);
      if (!unknown_meaning(l.text) && ins.f[4] == 0) VCHECK(sass::runs_instr(ins));
      ++n;
    }
    VCHECK(n > 30);
  }
}

VTEST(a_cubin_of_the_packed_instructions_has_nothing_unsupported) {
  for (const char* c : {"sm_100", "sm_100a", "sm_103a"}) {
    const int sm = std::atoi(c + 3);
    std::vector<sass::Word> words;
    for (const Line& l : packed_lines(c))
      if (!unknown_meaning(l.text) && sass::runs_instr(sass::decode(l.w, l.pc, sm))) words.push_back(l.w);
    VCHECK(words.size() > 30);
    VCHECK_EQ(unsupported(sm, words), std::string());
  }
}

VTEST(forms_of_unknown_meaning_are_refused_by_name) {
  size_t refused = 0;
  for (const Line& l : packed_lines("sm_100a")) {
    if (!unknown_meaning(l.text)) continue;
    const sass::Instr ins = sass::decode(l.w, l.pc, 100);   // decodes and prints ...
    VCHECK_EQ(sass::to_text(ins), l.text);
    VCHECK(!sass::runs_instr(ins));                          // ... but is not run
    const std::string why = unsupported(100, {l.w});
    VCHECK(mentions(why, "sm_100 k+0x0: "));
    VCHECK(mentions(why, l.text));
    ++refused;
  }
  VCHECK(refused >= 2);
}

VTEST(an_immediate_that_is_not_one_float_for_both_lanes_is_refused) {
  // ptxas sets the .F32 bit of an immediate (85 for the operand in the third place, 88 for the second); nvdisasm prints the float
  // alone either way, so only the bit tells, and a word without it is a form nobody has seen.
  size_t tried = 0;
  for (const Line& l : packed_lines("sm_100a")) {
    const sass::Instr ins = sass::decode(l.w, l.pc, 100);
    if (!sass::runs_instr(ins)) continue;   // (the corpus holds variants of unknown meaning too)
    for (size_t i = 0; i < ins.src.size(); ++i) {
      if (ins.src[i].kind != sass::Kind::FImm) continue;
      const unsigned bit = (ins.op == sass::Op::FADD2 || i == 2) ? 85 : 88;
      sass::Word w = l.w;
      w.hi &= ~(uint64_t{1} << (bit - 64));
      const sass::Instr odd = sass::decode(w, l.pc, 100);
      VCHECK_EQ(sass::to_text(odd), sass::to_text(ins));
      VCHECK(sass::runs_instr(ins));
      VCHECK(!sass::runs_instr(odd));
      ++tried;
    }
  }
  VCHECK(tried >= 4);
}

VTEST(sm_120_keeps_its_uvimnmx_at_the_same_low_opcode) {
  size_t seen = 0;
  for (const Line& l : packed_lines("sm_100a")) {
    if (!mentions(l.text, "FMUL2")) continue;
    // The same word read as sm_120's is not a packed multiply (UVIMNMX, or a refusal), whatever else it is.
    try {
      const sass::Instr ins = sass::decode(l.w, l.pc, 120);
      VCHECK(ins.op != sass::Op::FMUL2);
    } catch (const Error&) {
    }
    ++seen;
  }
  VCHECK(seen > 10);
}

namespace {

constexpr float kInf = __builtin_inff();

uint32_t of(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}

uint64_t pair(uint32_t lo, uint32_t hi) { return uint64_t(lo) | uint64_t(hi) << 32; }

// The shared arithmetic under a rounding mode, as the PTX interpreter and the SASS executor call it: the operands are read after
// the mode is set (volatile loads behind a compiler barrier), as the engines read theirs from the register file -- a compiler
// is free to fold or reuse an operation on constants across fesetround, and did.
[[gnu::noipa]] uint64_t under(int mode, exec::F32x2Op op, bool ftz, uint64_t a, uint64_t b, uint64_t c) {
  volatile uint64_t va = a, vb = b, vc = c;
  const int old = std::fegetround();
  std::fesetround(mode);
  asm volatile("" ::: "memory");
  const uint64_t r = exec::f32x2_pair(op, ftz, va, vb, vc);
  asm volatile("" ::: "memory");
  std::fesetround(old);
  return r;
}

}  // namespace

VTEST(the_shared_arithmetic_rounds_each_lane_as_ieee_says) {
  using exec::F32x2Op;
  const uint32_t one = 0x3f800000, tie = 0x33800000 /* 2^-24 */, one_up = 0x3f800001, two = 0x40000000, max = 0x7f7fffff,
                 min_sub = 1, half = 0x3f000000;
  // 1 + 2^-24 is a tie between 1 and 1 + 2^-23, and 2 + 2^-24 is not one (2's spacing is 2^-22).
  const uint64_t x = pair(one, two), y = pair(tie, tie);
  VCHECK_EQ(under(FE_TONEAREST, F32x2Op::Add, false, x, y, 0), pair(one, two));
  VCHECK_EQ(under(FE_TOWARDZERO, F32x2Op::Add, false, x, y, 0), pair(one, two));
  VCHECK_EQ(under(FE_DOWNWARD, F32x2Op::Add, false, x, y, 0), pair(one, two));
  VCHECK_EQ(under(FE_UPWARD, F32x2Op::Add, false, x, y, 0), pair(one_up, two + 1));
  // The largest float times two: infinity, or the largest, by mode and sign.
  const uint64_t big = pair(max, max | 0x80000000u), twos = pair(two, two);
  VCHECK_EQ(under(FE_TONEAREST, F32x2Op::Mul, false, big, twos, 0), pair(0x7f800000, 0xff800000));
  VCHECK_EQ(under(FE_TOWARDZERO, F32x2Op::Mul, false, big, twos, 0), pair(max, max | 0x80000000u));
  VCHECK_EQ(under(FE_DOWNWARD, F32x2Op::Mul, false, big, twos, 0), pair(max, 0xff800000));
  VCHECK_EQ(under(FE_UPWARD, F32x2Op::Mul, false, big, twos, 0), pair(0x7f800000, max | 0x80000000u));
  // The smallest subnormal halved: a tie to even (zero, with the sign); .ftz flushes the input first.
  const uint64_t tiny = pair(min_sub, min_sub | 0x80000000u), halves = pair(half, half);
  VCHECK_EQ(under(FE_TONEAREST, F32x2Op::Mul, false, tiny, halves, 0), pair(0, 0x80000000u));
  VCHECK_EQ(under(FE_UPWARD, F32x2Op::Mul, false, tiny, halves, 0), pair(min_sub, 0x80000000u));
  VCHECK_EQ(under(FE_DOWNWARD, F32x2Op::Mul, false, tiny, halves, 0), pair(0, min_sub | 0x80000000u));
  VCHECK_EQ(under(FE_UPWARD, F32x2Op::Mul, true, tiny, halves, 0), pair(0, 0x80000000u));
  // A subnormal result stays, or is flushed to zero with its sign.
  VCHECK_EQ(under(FE_TONEAREST, F32x2Op::Add, false, pair(1, 0x80000001u), pair(1, 0x80000001u), 0), pair(2, 0x80000002u));
  VCHECK_EQ(under(FE_TONEAREST, F32x2Op::Add, true, pair(1, 0x80000001u), pair(1, 0x80000001u), 0), pair(0, 0x80000000u));
  // Zeros of opposite signs: +0, or -0 rounding down.
  VCHECK_EQ(under(FE_TONEAREST, F32x2Op::Add, false, pair(0x80000000u, 0), pair(0, 0x80000000u), 0), pair(0, 0));
  VCHECK_EQ(under(FE_DOWNWARD, F32x2Op::Add, false, pair(0x80000000u, 0), pair(0, 0x80000000u), 0), pair(0x80000000u, 0x80000000u));
  // The fused multiply-add rounds once: (1 + 2^-12)^2 - (1 + 2^-11) is exactly 2^-24; and plus 2^-30 it is in the four modes.
  const uint64_t s = pair(0x3f800800, 0x3f800800), c1 = pair(0xbf801000, 0xbf801000), c2 = pair(0x30800000, 0x30800000);
  VCHECK_EQ(under(FE_TONEAREST, F32x2Op::Fma, false, s, s, c1), pair(tie, tie));
  VCHECK_EQ(under(FE_TONEAREST, F32x2Op::Fma, false, s, s, c2), pair(0x3f801001, 0x3f801001));
  VCHECK_EQ(under(FE_TOWARDZERO, F32x2Op::Fma, false, s, s, c2), pair(0x3f801000, 0x3f801000));
  VCHECK_EQ(under(FE_DOWNWARD, F32x2Op::Fma, false, s, s, c2), pair(0x3f801000, 0x3f801000));
  VCHECK_EQ(under(FE_UPWARD, F32x2Op::Fma, false, s, s, c2), pair(0x3f801001, 0x3f801001));
  // The two lanes are independent, and an infinity minus itself is a NaN.
  const uint64_t r = under(FE_TONEAREST, F32x2Op::Add, false, pair(of(kInf), one), pair(of(-kInf), one), 0);
  VCHECK_EQ(uint32_t(r), 0x7fffffffu);   // the canonical NaN, as the scalar operations give
  // ... whatever the NaN operands' payloads and signs, and whichever lane or operand held them
  for (uint32_t nan : {0x7fc00000u, 0xffc00000u, 0x7f800001u, 0xffa00000u, 0x7fc12345u}) {
    VCHECK_EQ(uint32_t(under(FE_TONEAREST, F32x2Op::Mul, false, pair(nan, one), twos, 0)), 0x7fffffffu);
    VCHECK_EQ(uint32_t(under(FE_TONEAREST, F32x2Op::Mul, false, twos, pair(nan, one), 0)), 0x7fffffffu);
    VCHECK_EQ(uint32_t(under(FE_TOWARDZERO, F32x2Op::Fma, false, pair(one, nan), twos, twos) >> 32), 0x7fffffffu);
    VCHECK_EQ(uint32_t(under(FE_UPWARD, F32x2Op::Fma, false, twos, twos, pair(one, nan)) >> 32), 0x7fffffffu);
  }
  VCHECK_EQ(uint32_t(r >> 32), two);
}

VTEST_MAIN
