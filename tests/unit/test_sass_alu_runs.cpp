// The SASS tier issues a stretch of register-only instructions back to back (Program::run, tier_issue in tier_ops.inc) and lets the
// kernels of a full warp write the destination register themselves (tier_out). Neither may change a thing a kernel or a counter shows: the
// same kernels run with the tier and with VGPU_FASTPATH=0 (the interpreter's general path, instruction by instruction) must leave the same
// registers in memory and the same counters.
//
// The kernels are SASS for sm_86 written out word by word by the small assembler below (nvdisasm's text for each word is checked against
// the decoder's, so a mistaken encoding fails here and not as a strange result). They are long straight stretches -- longer than the 256
// instructions the run loop takes at a time -- of integer, float and conversion instructions on registers that are also their own sources,
// with a guarded one now and then (a stretch ends at it) and a compare that sets the predicate it uses; every thread stores its registers.
// Block sizes of whole warps, of a warp and one lane, and of less than a warp make the full-warp and the partial-warp paths both run.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/sass/exec.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

struct Insn {
  sass::Word w;
  std::string text;
};

std::string reg(unsigned r) { return r == 255 ? "RZ" : "R" + std::to_string(r); }
std::string hex(uint32_t v) {
  char b[16];
  std::snprintf(b, sizeof b, "0x%x", v);
  return b;
}
Insn make(uint64_t hi, uint64_t lo, std::string text) {
  Insn i;
  i.w.hi = hi, i.w.lo = lo;
  i.text = std::move(text);
  return i;
}
// lo: opcode | rd << 16 | ra << 24 | rb or immediate << 32; hi's low byte is the third register of the forms that have one.
uint64_t lo_of(uint64_t op, unsigned rd, unsigned ra, uint64_t b) { return op | uint64_t{rd} << 16 | uint64_t{ra} << 24 | b << 32; }

Insn iadd3(unsigned d, unsigned a, unsigned b, unsigned c) {
  return make(0x000fc80007ffe000ull | c, lo_of(0x7210, d, a, b), "IADD3 " + reg(d) + ", " + reg(a) + ", " + reg(b) + ", " + reg(c));
}
// guard: 7 none (PT), 0 @P0, 8 @!P0
Insn iadd3i(unsigned d, unsigned a, uint32_t imm, unsigned c, unsigned guard = 7) {
  return make(0x000fc80007ffe000ull | c, lo_of(0x0810 | (guard == 7 ? 0x7000 : guard << 12), d, a, imm),
              std::string(guard == 7 ? "" : guard == 8 ? "@!P0 " : "@P0 ") + "IADD3 " + reg(d) + ", " + reg(a) + ", " + hex(imm) + ", " + reg(c));
}
Insn imad(unsigned d, unsigned a, unsigned b, unsigned c) {
  return make(0x000fca00078e0200ull | c, lo_of(0x7224, d, a, b), "IMAD " + reg(d) + ", " + reg(a) + ", " + reg(b) + ", " + reg(c));
}
Insn imadi(unsigned d, unsigned a, uint32_t imm, unsigned c) {
  return make(0x000fe400078e0200ull | c, lo_of(0x7824, d, a, imm), "IMAD " + reg(d) + ", " + reg(a) + ", " + hex(imm) + ", " + reg(c));
}
Insn lop3(unsigned d, unsigned a, unsigned b, unsigned c, unsigned lut) {
  return make(0x000fe200078e0000ull | lut << 8 | c, lo_of(0x7212, d, a, b),
              "LOP3.LUT " + reg(d) + ", " + reg(a) + ", " + reg(b) + ", " + reg(c) + ", " + hex(lut) + ", !PT");
}
Insn shf_l(unsigned d, unsigned a, unsigned b) {
  return make(0x000fca00000006ffull, lo_of(0x7219, d, a, b), "SHF.L.U32 " + reg(d) + ", " + reg(a) + ", " + reg(b) + ", RZ");
}
Insn shf_li(unsigned d, unsigned a, unsigned imm) {
  return make(0x000fe200000006ffull, lo_of(0x7819, d, a, imm), "SHF.L.U32 " + reg(d) + ", " + reg(a) + ", " + hex(imm) + ", RZ");
}
Insn popc(unsigned d, unsigned b) { return make(0x000e220000000000ull, lo_of(0x7309, d, 0, b), "POPC " + reg(d) + ", " + reg(b)); }
Insn ffma(unsigned d, unsigned a, unsigned b, unsigned c) {
  return make(0x000fe20000000000ull | c, lo_of(0x7223, d, a, b), "FFMA " + reg(d) + ", " + reg(a) + ", " + reg(b) + ", " + reg(c));
}
Insn fadd(unsigned d, unsigned a, unsigned b) {
  return make(0x000fc80000000000ull, lo_of(0x7221, d, a, b), "FADD " + reg(d) + ", " + reg(a) + ", " + reg(b));
}
Insn i2fp(unsigned d, unsigned b) { return make(0x000fe40000201000ull, lo_of(0x7245, d, 0, b), "I2FP.F32.U32 " + reg(d) + ", " + reg(b)); }
Insn imnmx(unsigned d, unsigned a, unsigned b) {
  return make(0x000fc80003800200ull, lo_of(0x7217, d, a, b), "IMNMX " + reg(d) + ", " + reg(a) + ", " + reg(b) + ", PT");
}
Insn movi(unsigned d, uint32_t imm) { return make(0x000fe20000000f00ull, lo_of(0x7802, d, 0, imm), "MOV " + reg(d) + ", " + hex(imm)); }
Insn isetp_eq(unsigned a, unsigned b) {   // P0 = (Ra == Rb)
  return make(0x000fe40003f02270ull, lo_of(0x720c, 0, a, b), "ISETP.EQ.AND P0, PT, " + reg(a) + ", " + reg(b) + ", PT");
}
Insn isetp_ge(unsigned a, uint32_t imm) {   // P0 = (Ra >= imm), signed
  return make(0x000fc40003f06270ull, lo_of(0x780c, 0, a, imm), "ISETP.GE.AND P0, PT, " + reg(a) + ", " + hex(imm) + ", PT");
}
Insn s2r_tid(unsigned d) { return make(0x000e220000002100ull, lo_of(0x7919, d, 0, 0), "S2R " + reg(d) + ", SR_TID.X"); }
Insn exit_() { return make(0x000fea0003800000ull, 0x000000000000794dull, "EXIT"); }
// STG.E [Ra.64 + off], Rk
Insn stg(unsigned a, unsigned k, unsigned off) {
  return make(0x000fe2000c101904ull, lo_of(0x7986, 0, a, k) | uint64_t{off} << 40,
              "STG.E [" + reg(a) + ".64" + (off ? "+" + hex(off) : "") + "], " + reg(k));
}
// IMAD.WIDE.U32 Rd, Ra, imm, Rc
Insn imad_wide_i(unsigned d, unsigned a, uint32_t imm, unsigned c) {
  return make(0x000fe200078e0000ull | c, lo_of(0x7825, d, a, imm),
              "IMAD.WIDE.U32 " + reg(d) + ", " + reg(a) + ", " + hex(imm) + ", " + reg(c));
}

template <class T>
void put(std::string& s, T v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof v);
}

// A cubin for sm_86 with one kernel taking one 8-byte pointer.
std::string make_cubin(const std::vector<Insn>& code) {
  const std::string names("\0.shstrtab\0.text.k\0.nv.info.k\0", 31);
  std::string info;   // KPARAM_INFO: index, ordinal | offset << 16, size << 18
  put<uint8_t>(info, 4), put<uint8_t>(info, 0x17), put<uint16_t>(info, 12);
  put<uint32_t>(info, 0), put<uint32_t>(info, 0), put<uint32_t>(info, 8u << 18);
  std::string out(64, '\0');
  const uint64_t names_off = out.size();
  out += names;
  while (out.size() % 16) out.push_back('\0');
  const uint64_t text_off = out.size();
  for (const Insn& i : code) put(out, i.w.lo), put(out, i.w.hi);
  const uint64_t info_off = out.size();
  out += info;
  while (out.size() % 8) out.push_back('\0');
  const uint64_t shoff = out.size();
  const auto shdr = [&](uint32_t name, uint32_t type, uint64_t off, uint64_t size) {
    put<uint32_t>(out, name), put<uint32_t>(out, type);
    put<uint64_t>(out, 0), put<uint64_t>(out, 0), put<uint64_t>(out, off), put<uint64_t>(out, size);
    put<uint32_t>(out, 0), put<uint32_t>(out, 0), put<uint64_t>(out, 16), put<uint64_t>(out, 0);
  };
  shdr(0, 0, 0, 0);
  shdr(1, 3, names_off, names.size());
  shdr(11, 1, text_off, 16 * code.size());
  shdr(19, 0x70000000, info_off, info.size());
  out[0] = 0x7f, out[1] = 'E', out[2] = 'L', out[3] = 'F', out[4] = 2, out[5] = 1, out[6] = 1, out[7] = 0x41, out[8] = 8;
  const uint16_t type = 2, machine = 190, shentsize = 64, shnum = 4, shstrndx = 1;
  const uint32_t flags = 0x06000002u | (86u << 8);
  std::memcpy(&out[16], &type, 2), std::memcpy(&out[18], &machine, 2);
  std::memcpy(&out[0x28], &shoff, 8), std::memcpy(&out[0x30], &flags, 4);
  std::memcpy(&out[0x3a], &shentsize, 2), std::memcpy(&out[0x3c], &shnum, 2), std::memcpy(&out[0x3e], &shstrndx, 2);
  return out;
}

// A small deterministic generator for the choice of instructions.
struct Rng {
  uint64_t s = 0x9e3779b97f4a7c15ull;
  unsigned next(unsigned n) {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    return static_cast<unsigned>((s >> 11) % n);
  }
};

constexpr unsigned kRegs = 10;   // R1 .. R9 hold the values the kernel works on, R0 is the thread number, R12:R13 the pointer, R14:R15 the address
constexpr unsigned kRZ = 255;
constexpr uint32_t kStride = 0x40;   // bytes a thread's results take

// The kernel: `body` instructions of a mix of forms on R1..R9, after seeding them from the thread number.
std::vector<Insn> kernel(unsigned body, bool guards, uint64_t seed) {
  Rng rng;
  rng.s ^= seed * 0xff51afd7ed558ccdull;
  std::vector<Insn> code;
  code.push_back(s2r_tid(0));
  // The pointer: R12:R13 from the parameter bank (c[0x0][0x160], c[0x0][0x164]).
  code.push_back(make(0x000fc60000000f00ull, 0x0000580000007a02ull | uint64_t{12} << 16, "MOV R12, c[0x0][0x160]"));
  code.push_back(make(0x000fc60000000f00ull, 0x0000590000007a02ull | uint64_t{13} << 16, "MOV R13, c[0x0][0x164]"));
  code.push_back(imad_wide_i(14, 0, kStride, 12));
  for (unsigned r = 1; r < kRegs; ++r) {   // seeds that differ from lane to lane and register to register
    code.push_back(imadi(r, 0, (0x9e3779b1u * r + 17) & 0x7fffffffu, kRZ));
    code.push_back(iadd3i(r, r, 0x1234567u * r, 0));
  }
  const auto any = [&](bool zero) { return zero && rng.next(8) == 0 ? kRZ : rng.next(kRegs); };
  for (unsigned i = 0; i < body; ++i) {
    const unsigned d = 1 + rng.next(kRegs - 1);   // (a destination is often one of the sources: the choice is free)
    const unsigned a = any(true), b = any(false), c = any(true);
    switch (rng.next(guards ? 17 : 14)) {
      case 0: code.push_back(iadd3(d, a, b, c)); break;
      case 1: code.push_back(imad(d, a, b, c)); break;
      case 2: code.push_back(imadi(d, a, rng.next(1u << 20) + 1, c)); break;
      case 3: code.push_back(lop3(d, a, b, c, rng.next(2) ? 0x96 : 0xe8)); break;
      case 4: code.push_back(lop3(d, a, b, kRZ, rng.next(2) ? 0xc0 : 0x3c)); break;
      case 5: code.push_back(shf_l(d, a, b)); break;
      case 6: code.push_back(shf_li(d, a, rng.next(31) + 1)); break;
      case 7: code.push_back(popc(d, b)); break;
      case 8: code.push_back(i2fp(d, b)); break;
      case 9: code.push_back(fadd(d, a, b)); break;
      case 10: code.push_back(ffma(d, a, b, c)); break;
      case 11: code.push_back(imnmx(d, a, b)); break;
      case 12: code.push_back(movi(d, rng.next(1u << 30))); break;
      case 13: code.push_back(iadd3i(d, a, rng.next(1000), b)); break;
      case 14: code.push_back(isetp_ge(a, 0x40000000u)); break;
      case 15:   // the low half of the warp, then the high half, runs an instruction
        code.push_back(isetp_ge(0, 16));
        code.push_back(iadd3i(d, a, rng.next(100) + 1, c, rng.next(2) ? 0 : 8));
        break;
      default:   // a guarded instruction, on the predicate a compare just set from this very thread's data
        code.push_back(isetp_eq(a, b));
        code.push_back(iadd3i(d, a, rng.next(100) + 1, c, 0));
        break;
    }
  }
  for (unsigned r = 1; r < kRegs; ++r) code.push_back(stg(14, r, 4 * r));
  code.push_back(stg(14, 0, 0));
  code.push_back(exit_());
  return code;
}

struct Result {
  std::vector<uint32_t> out;
  exec::LaunchStats stats;
};

struct Env {
  MemoryManager mem{1 << 24};
  DeviceProfile prof = load_gpu("nvidia/a10");   // sm_86
  std::shared_ptr<sass::Module> load(const std::vector<Insn>& code) {
    for (size_t i = 0; i < code.size(); ++i) {
      const sass::Instr ins = sass::decode(code[i].w, 16 * i, 86);
      if (!code[i].text.empty() && sass::to_text(ins) != code[i].text)
        throw vtest::Failure("word " + std::to_string(i) + " is \"" + sass::to_text(ins) + "\", not \"" + code[i].text + "\"");
    }
    const std::string c = make_cubin(code);
    return sass::load(reinterpret_cast<const uint8_t*>(c.data()), c.size(), mem, prof);
  }
  Result run(sass::Module& m, uint32_t blocks, uint32_t threads) {
    const uint32_t words = blocks * threads * kStride / 4;
    const uint64_t out = mem.alloc(words * 4);
    const std::vector<uint32_t> poison(words, 0x5A5A5A5Au);
    mem.write(out, poison.data(), words * 4);
    exec::LaunchConfig cfg;
    cfg.grid = {blocks, 1, 1};
    cfg.block = {threads, 1, 1};
    std::vector<uint8_t> arg(8);
    // (each block's threads write from the start of the array: the block number is not in the address, so blocks overwrite one another
    // with the same values; one block is the check, the others are more launches of the same warps)
    std::memcpy(arg.data(), &out, 8);
    Result r;
    r.stats = sass::launch(m, "k", cfg, {arg}, mem, prof);
    r.out.resize(words);
    mem.read(out, r.out.data(), words * 4);
    mem.free(out);
    return r;
  }
};

struct Scoped {
  Scoped() { ::setenv("VGPU_THREADS", "1", 1); }
  ~Scoped() {
    ::unsetenv("VGPU_THREADS");
    ::unsetenv("VGPU_FASTPATH");
  }
};

void expect_same(const Result& tier, const Result& ref, const std::string& what) {
  if (tier.out != ref.out) {
    size_t i = 0;
    while (i < tier.out.size() && tier.out[i] == ref.out[i]) ++i;
    std::string more;   // (the next few that differ: which thread, which register)
    for (size_t j = i + 1, n = 0; j < tier.out.size() && n < 6; ++j)
      if (tier.out[j] != ref.out[j]) more += " " + std::to_string(j / (kStride / 4)) + ":" + std::to_string(j % (kStride / 4)), ++n;
    throw vtest::Failure(what + ": word " + std::to_string(i) + " (thread " + std::to_string(i / (kStride / 4)) + ", register " + std::to_string(i % (kStride / 4)) +
                         ") is " + hex(tier.out[i]) + " with the tier and " + hex(ref.out[i]) + " without; also" + more);
  }
  if (std::memcmp(&tier.stats, &ref.stats, exec::kCounterWords * sizeof(uint64_t)) != 0) {
    const uint64_t* a = reinterpret_cast<const uint64_t*>(&tier.stats);
    const uint64_t* b = reinterpret_cast<const uint64_t*>(&ref.stats);
    size_t i = 0;
    while (a[i] == b[i]) ++i;
    throw vtest::Failure(what + ": counter word " + std::to_string(i) + " is " + std::to_string(a[i]) + " with the tier and " + std::to_string(b[i]) +
                         " without");
  }
}

// The kernel run by the tier and by the interpreter, over the block sizes given.
void compare(const std::vector<Insn>& code, std::initializer_list<uint32_t> sizes, const std::string& name) {
  Scoped scoped;
  Env e;
  const auto m = e.load(code);
  for (uint32_t threads : sizes) {
    const Result tier = e.run(*m, 3, threads);
    ::setenv("VGPU_FASTPATH", "0", 1);
    const Result ref = e.run(*m, 3, threads);
    ::unsetenv("VGPU_FASTPATH");
    expect_same(tier, ref, name + ", " + std::to_string(threads) + " threads");
  }
}


// ---- the conversions between f32 and f64 (cvt64 and fltinterp forms of the tier, fast_float's kernels) ------------------------------------------------
// F2F.F64.F32 Rd, [|-]Rb ; F2F.F32.F64[.rnd] Rd, [|-]Rb ; I2F.F64.U32 / I2F.F64 (S32) Rd, Rb ; MUFU.RCP64H / RSQ64H Rd, Rb.
// guard: 7 none, 0 @P0, 8 @!P0.
std::string guard_text(unsigned guard) { return guard == 7 ? "" : guard == 8 ? "@!P0 " : "@P0 "; }
uint64_t gop(uint64_t op, unsigned guard) { return (op & 0x0fff) | (guard == 7 ? 0x7000 : uint64_t{guard} << 12); }
std::string src_text(unsigned b, bool ab, bool ng) { return std::string(ng ? "-" : "") + (ab ? "|" : "") + reg(b) + (ab ? "|" : ""); }
Insn f2f_wide(unsigned d, unsigned b, bool ab = false, bool ng = false, unsigned guard = 7) {
  return make(0x000fe40000201800ull, lo_of(gop(0x7310, guard), d, 0, b) | (ab ? 1ull << 62 : 0) | (ng ? 1ull << 63 : 0),
              guard_text(guard) + "F2F.F64.F32 " + reg(d) + ", " + src_text(b, ab, ng));
}
Insn f2f_narrow(unsigned d, unsigned b, unsigned rnd = 0, bool ab = false, bool ng = false, unsigned guard = 7) {
  static const char* const rn[] = {"", ".RM", ".RP", ".RZ"};
  return make(0x000fe40000301000ull | uint64_t{rnd} << 14, lo_of(gop(0x7310, guard), d, 0, b) | (ab ? 1ull << 62 : 0) | (ng ? 1ull << 63 : 0),
              guard_text(guard) + "F2F.F32.F64" + rn[rnd] + " " + reg(d) + ", " + src_text(b, ab, ng));
}
Insn i2f_d(unsigned d, unsigned b, bool sgn, unsigned guard = 7) {
  return make(0x000fe40000201800ull | (sgn ? 0x400ull : 0), lo_of(gop(0x7312, guard), d, 0, b),
              guard_text(guard) + (sgn ? "I2F.F64 " : "I2F.F64.U32 ") + reg(d) + ", " + reg(b));
}
Insn mufu64h(unsigned d, unsigned b, bool rsq) {
  return make(0x000fe40000001800ull | (rsq ? 0x400ull : 0), lo_of(0x7308, d, 0, b), std::string(rsq ? "MUFU.RSQ64H " : "MUFU.RCP64H ") + reg(d) + ", " + reg(b));
}

// R `r` shaped to a class of values: (r & and_m) | or_m, with the thread number added at bit `tid_shift` (an exponent that grows with the lane), if >= 0.
// (The scratch registers are R5 and R6; a kernel here has R0 .. R15 only.)
void shape(std::vector<Insn>& c, unsigned r, uint32_t and_m, uint32_t or_m, int tid_shift) {
  c.push_back(movi(5, and_m));
  c.push_back(lop3(r, r, 5, kRZ, 0xc0));
  c.push_back(movi(5, or_m));
  c.push_back(lop3(r, r, 5, kRZ, 0xfc));
  if (tid_shift >= 0) {
    c.push_back(shf_li(6, 0, static_cast<unsigned>(tid_shift)));
    c.push_back(iadd3(r, r, 6, kRZ));
  }
}
// A word that differs from lane to lane and from register to register without much structure.
void scramble(std::vector<Insn>& c, unsigned r, uint32_t k) {
  c.push_back(imadi(r, 0, ((k * 0x9e3779b1u) | 1u) & 0x7fffffffu, kRZ));
  c.push_back(iadd3i(r, r, (k * 0x85ebca6bu + 7) & 0x7fffffffu, 0));
  c.push_back(imad(r, r, r, 0));
  c.push_back(shf_li(6, r, 13));
  c.push_back(lop3(r, r, 6, kRZ, 0x96));
  c.push_back(imadi(r, r, 0x2545f491u, 0));
}

// One kernel: R1 an f32, R2:R3 an f64, R4 an integer, each shaped to a class of values chosen by the seed (zeros, subnormals, infinities and NaNs with
// every payload, exponents around the f32 edges, rounding ties), then the conversions of the family chosen by `part` (0 to 7) into R6 .. R11: with |x|, -x
// and the four rounding modes, results used as sources, a destination that is a source, a guarded form for half the warp, and MUFU.RCP64H / RSQ64H.
std::vector<Insn> cvt_kernel(uint64_t seed, unsigned part) {
  Rng rng;
  rng.s ^= seed * 0xff51afd7ed558ccdull;
  std::vector<Insn> c;
  c.push_back(s2r_tid(0));
  c.push_back(make(0x000fc60000000f00ull, 0x0000580000007a02ull | uint64_t{12} << 16, "MOV R12, c[0x0][0x160]"));
  c.push_back(make(0x000fc60000000f00ull, 0x0000590000007a02ull | uint64_t{13} << 16, "MOV R13, c[0x0][0x164]"));
  c.push_back(imad_wide_i(14, 0, kStride, 12));
  for (unsigned r = 1; r <= 4; ++r) scramble(c, r, static_cast<uint32_t>(seed) * 8 + r);
  // R1, the f32
  switch (rng.next(7)) {
    case 0: break;                                                     // any bits
    case 1: shape(c, 1, 0x807fffffu, 0, -1); break;                    // zero and subnormals
    case 2: shape(c, 1, 0x807fffffu, 0x7f800000u, -1); break;          // infinities and NaNs of every payload (signalling too)
    case 3: shape(c, 1, 0x807fffffu, 0x7fc00000u, -1); break;          // quiet NaNs
    case 4: shape(c, 1, 0x807fffffu, (0x70u + rng.next(16)) << 23, 23); break;   // exponents around 1 and growing with the lane
    case 5: shape(c, 1, 0x80000000u | (0xffffu << 7), (0xfeu - rng.next(4)) << 23, -1); break;   // large
    default: shape(c, 1, 0xff800000u, 0, -1); break;                   // exact powers of two and zero
  }
  // R2:R3, the f64
  switch (rng.next(5)) {
    case 0: break;
    case 1: shape(c, 2, 0xe0000000u, 0, -1); break;                    // few low bits: exactly representable, or ties
    case 2: shape(c, 2, 0, 0x10000000u, -1); break;                    // the half-way bit alone in the low word
    case 3: shape(c, 2, 0x1fffffffu, 0, -1); break;                    // below the half-way bit
    default: shape(c, 2, 0xffffffffu, 0x10000000u, -1); break;
  }
  switch (rng.next(8)) {
    case 0: break;
    case 1: shape(c, 3, 0x800fffffu, 0, -1); break;                    // zero and subnormals
    case 2: shape(c, 3, 0x800fffffu, 0x7ff00000u, -1); break;          // infinities and NaNs
    case 3: shape(c, 3, 0x800fffffu, 0x7ff80000u, -1); break;
    case 4: shape(c, 3, 0x800fffffu, (0x37cu + rng.next(16)) << 20, 20); break;   // the f32 subnormal edge, growing with the lane
    case 5: shape(c, 3, 0x800fffffu, (0x47cu + rng.next(8)) << 20, 20); break;    // the f32 overflow edge
    case 6: shape(c, 3, 0x800fffffu, (0x3f0u + rng.next(32)) << 20, 20); break;   // around 1
    default: shape(c, 3, 0xfff00000u, 0x00000000u, -1); break;         // exact powers of two
  }
  switch (rng.next(4)) {
    case 0: break;
    case 1: shape(c, 4, 0xffu, 0, -1); break;
    case 2: shape(c, 4, 0xffffffffu, 0x80000000u, -1); break;
    default: shape(c, 4, 0x0000ffffu, 0, 16); break;
  }
  switch (part) {
    case 0:
      c.push_back(f2f_wide(6, 1));
      c.push_back(f2f_wide(8, 1, true));
      c.push_back(f2f_wide(10, 1, false, true));
      break;
    case 1:
      c.push_back(f2f_wide(6, 1, true, true));
      c.push_back(f2f_wide(8, 4));            // an integer's bits as an f32
      c.push_back(f2f_narrow(10, 2, 0, false, true));
      break;
    case 2:
      for (unsigned m = 0; m < 4; ++m) c.push_back(f2f_narrow(6 + m, 2, m));
      c.push_back(f2f_narrow(10, 2, 0, true));
      c.push_back(f2f_narrow(11, 2, 3, false, true));
      break;
    case 3:
      c.push_back(f2f_narrow(6, 2, 1, true));
      c.push_back(f2f_narrow(7, 2, 2, true, true));
      c.push_back(f2f_narrow(8, 2, 0, true, true));
      c.push_back(f2f_narrow(9, 2, 3, true, true));
      c.push_back(i2f_d(10, 4, false));
      break;
    case 4:
      c.push_back(i2f_d(6, 4, false));
      c.push_back(i2f_d(8, 4, true));
      c.push_back(f2f_narrow(10, 6, 3));      // a result as the source
      break;
    case 5:
      c.push_back(f2f_wide(6, 1));
      c.push_back(f2f_narrow(8, 6, 2));       // a chain: f32 to f64 to f32 to f64
      c.push_back(f2f_wide(10, 8));
      break;
    case 6:
      c.push_back(f2f_wide(6, 4));            // the destination is also the source, the low word and the high one
      c.push_back(f2f_narrow(6, 6, 1));
      c.push_back(iadd3(8, 4, kRZ, kRZ));
      c.push_back(i2f_d(8, 8, rng.next(2) != 0));
      c.push_back(iadd3(10, 3, kRZ, kRZ));
      c.push_back(iadd3(11, 2, kRZ, kRZ));
      c.push_back(f2f_narrow(11, 10, 2, true));
      break;
    default:
      c.push_back(mufu64h(6, 3, false));
      c.push_back(mufu64h(7, 3, true));
      c.push_back(isetp_ge(0, 16));                   // P0: the upper lanes of a full warp
      c.push_back(f2f_wide(8, 1, false, false, 0));   // guarded: only those lanes write
      c.push_back(i2f_d(8, 4, true, 8));              // the other lanes, over it
      c.push_back(f2f_narrow(10, 2, 3, false, true, 8));
      c.push_back(f2f_narrow(11, 2, 1, true, false, 0));
      break;
  }
  for (unsigned r = 1; r < 12; ++r) c.push_back(stg(14, r, 4 * r));
  c.push_back(stg(14, 0, 0));
  c.push_back(exit_());
  return c;
}

}  // namespace

VTEST(a_stretch_of_register_instructions_gives_the_interpreters_registers_and_counters) {
  compare(kernel(40, false, 1), {32, 64, 100}, "40 instructions");
}

VTEST(a_stretch_longer_than_the_run_loops_chunk_is_issued_in_pieces) {
  // 700 straight instructions: three chunks of 256 and a rest; a restart in the middle of a stretch must find the same registers.
  compare(kernel(700, false, 2), {32, 96}, "700 instructions");
}

VTEST(a_warp_of_part_of_its_lanes_writes_only_those_lanes) {
  // 33 threads: a second warp of one lane; 5 and 31: part of a warp. The kernels write a lane's registers only for the lanes that run.
  compare(kernel(150, false, 3), {1, 5, 31, 33, 65}, "partial warps");
}

VTEST(a_guarded_instruction_ends_a_stretch_and_runs_for_its_lanes_only) {
  compare(kernel(200, true, 4), {32, 33, 64}, "guarded instructions");
  compare(kernel(200, true, 5), {31, 96}, "guarded instructions, another mix");
}


VTEST(the_conversions_between_f32_and_f64_give_the_general_rules_registers_and_counters) {
  // Whole warps, a warp and a lane, and part of a warp, over many mixes of special values.
  for (uint64_t seed = 1; seed <= 40; ++seed)
    for (unsigned part = 0; part < 8; ++part)
      compare(cvt_kernel(seed, part), {32, 100, 33, 5}, "conversions, seed " + std::to_string(seed) + ", part " + std::to_string(part));
}

VTEST_MAIN
