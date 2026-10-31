// The SASS tier runs CALL.REL, RET, WARPSYNC and FCHK itself (src/sass/tier_ops.inc, tier::Form::Call, Ret, Warpsync, Fchk) where the interpreter's
// exec_control and exec_float did. None of that may change a register or a counter: the same kernels run with the tier and with VGPU_FASTPATH=0
// (the interpreter, instruction by instruction) must leave the same words in memory and the same counters.
//
// The kernels are SASS for sm_86 written out word by word by the small assembler below (the decoder's reading of each word is checked, so a mistaken
// encoding fails here and not as a strange result). A loop calls a function with the return address in a register pair (the division's slow path
// of nvcc's code is such a call: FCHK, @P CALL.REL.NOINC, ..., RET.REL.NODEC), by every lane, by the lanes a predicate names (a diverged warp: the
// callee and the code after the call both run, for different lanes), with a RET some lanes do not run (they fall through to an EXIT), and
// WARPSYNC with the full mask and with a mask of the lane's own half warp (register), in warps of 32 lanes and in partial warps.
//
// The tier's shared-memory loads and stores of a whole warp work their 32 offsets out as 32-bit words and check the window once (h_shared): shared-memory kernels
// with sequential, strided (.X4, .X8, .X16), scrambled, byte-wise and broadcast addresses run the same way.
#include <cstdint>
#include <cstdio>
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
#ifndef SASS_CALL_RET_NO_TIER_FORMS
#include "sass/tier.hpp"
#endif
#include "vtest.hpp"

using namespace vgpu;

namespace vgpu::sass {
void annotate_instr(Instr& ins);   // src/sass/exec.cpp
}

namespace {

struct Insn {
  sass::Word w;
  std::string text;
  sass::Op op = sass::Op::Unknown;   // when set, the op the decoder must read the word as
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
Insn ffma(unsigned d, unsigned a, unsigned b, unsigned c) {
  return make(0x000fe20000000000ull | c, lo_of(0x7223, d, a, b), "FFMA " + reg(d) + ", " + reg(a) + ", " + reg(b) + ", " + reg(c));
}
Insn movi(unsigned d, uint32_t imm) { return make(0x000fe20000000f00ull, lo_of(0x7802, d, 0, imm), "MOV " + reg(d) + ", " + hex(imm)); }
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

constexpr unsigned kRZ = 255;
constexpr uint32_t kStride = 0x40;   // bytes a thread's results take
constexpr unsigned kRegs = 7;        // R1 .. R6 hold the values the kernel works on, R0 is the thread number, R7 a scratch, R8 the loop count, R9 a mask, R10:R11 the return address, R12:R13 the pointer, R14:R15 the address

unsigned guard_bits(unsigned p, bool neg) { return p | (neg ? 8u : 0u); }   // @Pp, @!Pp; 7 is PT
std::string guard_text(unsigned p, bool neg) { return p == 7 && !neg ? "" : std::string("@") + (neg ? "!" : "") + "P" + std::to_string(p) + " "; }

// A relative branch target: the signed distance in words of 4 bytes from the next instruction, in bits 34-81.
void put_offset(Insn& i, size_t from, size_t to) {
  const int64_t bytes = (static_cast<int64_t>(to) - static_cast<int64_t>(from) - 1) * 16;
  const uint64_t v = static_cast<uint64_t>(bytes / 4) & ((uint64_t{1} << 48) - 1);
  i.w.lo |= (v & 0x3fffffffull) << 34;
  i.w.hi |= v >> 30;
}
Insn bra(unsigned gp, bool gn, size_t from, size_t to, sass::Op op = sass::Op::BRA) {
  Insn i = make(0x000fea0003800000ull, 0x947 | uint64_t{guard_bits(gp, gn)} << 12, "");
  put_offset(i, from, to);
  i.op = op;
  return i;
}
Insn call(unsigned gp, bool gn, size_t from, size_t to) {
  Insn i = make(0x000fea0003800000ull, 0x944 | uint64_t{guard_bits(gp, gn)} << 12, "");
  put_offset(i, from, to);
  i.op = sass::Op::CALL;
  return i;
}
Insn ret(unsigned gp, bool gn, unsigned r) {
  Insn i = make(0x000fea0003800000ull, 0x950 | uint64_t{guard_bits(gp, gn)} << 12 | uint64_t{r} << 24, "");
  i.op = sass::Op::RET;
  return i;
}
Insn fchk(unsigned dp, unsigned a, unsigned b) {
  Insn i = make(0x000fe20000000000ull | uint64_t{dp} << 17, 0x7302 | uint64_t{a} << 24 | uint64_t{b} << 32, "");
  i.op = sass::Op::FCHK;
  return i;
}
Insn warpsync_imm(uint32_t mask) {
  Insn i = make(0x000fe20003800000ull, 0x7948 | uint64_t{mask} << 32, "");
  i.op = sass::Op::WARPSYNC;
  return i;
}
Insn warpsync_reg(unsigned r) {
  Insn i = make(0x000fe20003800000ull, 0x7348 | uint64_t{r} << 32, "");
  i.op = sass::Op::WARPSYNC;
  return i;
}
Insn isetp_ge_p(unsigned p, unsigned a, uint32_t imm) {   // Pp = (Ra >= imm), signed
  return make(0x000fc40003f06270ull | uint64_t{p} << 17, lo_of(0x780c, 0, a, imm),
              "ISETP.GE.AND P" + std::to_string(p) + ", PT, " + reg(a) + ", " + hex(imm) + ", PT");
}
Insn iadd3i_g(unsigned d, unsigned a, uint32_t imm, unsigned c, unsigned gp, bool gn) {
  return make(0x000fc80007ffe000ull | c, lo_of(0x0810 | uint64_t{guard_bits(gp, gn)} << 12, d, a, imm),
              guard_text(gp, gn) + "IADD3 " + reg(d) + ", " + reg(a) + ", " + hex(imm) + ", " + reg(c));
}

enum class RetKind { Plain, Guarded };

// The kernel. `callee` is where the function is (the main part is the same size for any value, so it is built twice).
std::vector<Insn> build(unsigned loops, bool call_guarded, RetKind rk, unsigned sync, size_t callee) {
  std::vector<Insn> code;
  code.push_back(s2r_tid(0));
  code.push_back(make(0x000fc60000000f00ull, 0x0000580000007a02ull | uint64_t{12} << 16, "MOV R12, c[0x0][0x160]"));
  code.push_back(make(0x000fc60000000f00ull, 0x0000590000007a02ull | uint64_t{13} << 16, "MOV R13, c[0x0][0x164]"));
  code.push_back(imad_wide_i(14, 0, kStride, 12));
  for (unsigned r = 1; r < kRegs; ++r) {
    code.push_back(imadi(r, 0, (0x9e3779b1u * r + 17) & 0x7fffffffu, kRZ));
    code.push_back(iadd3i(r, r, 0x1234567u * r, 0));
  }
  code.push_back(movi(8, 0));
  const size_t top = code.size();
  code.push_back(Insn{});                       // MOV R10, the return address (set below: the call is three instructions on)
  code.push_back(movi(11, 0));                  // (the high word of the pair: not a code address's, so the low word is section-relative)
  code.push_back(isetp_ge_p(1, 0, 16));         // P1: the upper half warp
  const size_t at_call = code.size();
  code.push_back(call(call_guarded ? 1 : 7, false, at_call, callee));
  const size_t after = code.size();             // where the function returns to
  code[top] = movi(10, static_cast<uint32_t>(16 * after));
  code.push_back(iadd3(1, 1, 2, 3));
  code.push_back(ffma(4, 1, 5, 4));
  if (sync == 1) code.push_back(warpsync_imm(0xffffffffu));
  if (sync == 2) {   // a mask of the lane's own half warp, from a register
    code.push_back(movi(9, 0xffffu));
    code.push_back(movi(7, 16));
    code.push_back(lop3(7, 0, 7, kRZ, 0xc0));   // R7 = tid & 16: 16 in the upper half of a warp (a warp's first thread number is a multiple of 32)
    code.push_back(shf_l(9, 9, 7));
    code.push_back(warpsync_reg(9));
  }
  code.push_back(iadd3i(8, 8, 1, kRZ));
  code.push_back(isetp_ge_p(0, 8, loops));
  code.push_back(bra(0, true, code.size(), top));
  for (unsigned r = 1; r < kRegs; ++r) code.push_back(stg(14, r, 4 * r));
  code.push_back(stg(14, 0, 0));
  code.push_back(stg(14, 8, 40));
  code.push_back(exit_());
  if (code.size() != callee) return code;       // (the first pass: the real position of the function is not known yet)
  // The function: the division's slow path in miniature.
  code.push_back(ffma(6, 6, 1, 2));
  code.push_back(fchk(3, 4, 5));
  code.push_back(iadd3i_g(6, 6, 0x11, 7, 3, false));    // @P3 (FCHK says yes on every lane)
  code.push_back(iadd3i_g(6, 6, 0x1000, 7, 3, true));   // @!P3: no lane
  code.push_back(iadd3(2, 2, 3, kRZ));
  if (rk == RetKind::Guarded) {
    code.push_back(isetp_ge_p(4, 0, 24));
    code.push_back(ret(4, false, 10));
    code.push_back(exit_());                    // the lanes that do not return leave here
  } else {
    code.push_back(ret(7, false, 10));
  }
  return code;
}

std::vector<Insn> kernel(unsigned loops, bool call_guarded, RetKind rk, unsigned sync) {
  const size_t callee = build(loops, call_guarded, rk, sync, 0).size();
  return build(loops, call_guarded, rk, sync, callee);
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
      if (code[i].op != sass::Op::Unknown && ins.op != code[i].op)
        throw vtest::Failure("word " + std::to_string(i) + " is \"" + sass::to_text(ins) + "\", not the op the test means");
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
    cfg.shared_bytes = 4096;
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

// The answer of each run as one number (FNV-1a over the words in memory and the counters). The numbers in kGolden were taken from the tree before the
// tier ran these instructions (the interpreter only: main of 2026-10-29, 9c1d6ffd); the tier and the interpreter both share Runner::release_warpsync,
// so comparing the two cannot show a mistake in what they share. VGPU_PRINT_DIGESTS=1 prints the table instead of checking it.
uint64_t digest(const Result& r) {
  uint64_t h = 0xcbf29ce484222325ull;
  const auto mix = [&](uint64_t v) {
    for (unsigned i = 0; i < 8; ++i) h = (h ^ ((v >> (8 * i)) & 0xff)) * 0x100000001b3ull;
  };
  for (uint32_t w : r.out) mix(w);
  const uint64_t* c = reinterpret_cast<const uint64_t*>(&r.stats);
  for (size_t i = 0; i < exec::kCounterWords; ++i) mix(c[i]);
  return h;
}

struct Golden {
  const char* key;
  uint64_t digest;
};
const Golden kGolden[] = {
    {"unguarded call, 32 threads", 0xc974d6cfb4b779aaull},
    {"unguarded call, 64 threads", 0xa947db22e3a5c769ull},
    {"unguarded call, 100 threads", 0x27e26a457563a759ull},
    {"guarded call, 32 threads", 0xd2b4464d6894e9daull},
    {"guarded call, 33 threads", 0xa997419dad72bd12ull},
    {"guarded call, 64 threads", 0x5a544518d86d407bull},
    {"guarded call, 100 threads", 0x36c5d26613798537ull},
    {"guarded call, 5 threads", 0xa0faddb9077e649aull},
    {"guarded call, 31 threads", 0x21a5f020ca4c2eccull},
    {"guarded return, 32 threads", 0x277fcbbf0959abbeull},
    {"guarded return, 64 threads", 0x78841b5c427475a5ull},
    {"guarded return, 100 threads", 0x6ee4492186bb0ba7ull},
    {"guarded return, 20 threads", 0xb40aa9d244a6d7b4ull},
    {"unguarded call, guarded return, 32 threads", 0x392824ba49120b57ull},
    {"unguarded call, guarded return, 96 threads", 0x86716ae689d4ada8ull},
    {"WARPSYNC full mask, 32 threads", 0x33939f5472259133ull},
    {"WARPSYNC full mask, 64 threads", 0xe8bfb078b0859678ull},
    {"WARPSYNC full mask, 33 threads", 0x10a2704712e58d5aull},
    {"WARPSYNC full mask, 5 threads", 0xcb920cbd5b7a6e2aull},
    {"WARPSYNC half-warp mask, 32 threads", 0x7fff4459318b592cull},
    {"WARPSYNC half-warp mask, 64 threads", 0x5ce884bf08d7f00dull},
    {"WARPSYNC half-warp mask, 33 threads", 0x566bd00f768bb842ull},
    {"WARPSYNC half-warp mask, 100 threads", 0x568bb8bfd8998663ull},
    {"WARPSYNC half-warp mask, unguarded call, 32 threads", 0x55a4f8ef91aa6d68ull},
    {"WARPSYNC half-warp mask, unguarded call, 96 threads", 0x629e3e426dee440cull},
    {"WARPSYNC after exits, full mask, 32 threads", 0xb823e7f0ac61ce89ull},
    {"WARPSYNC after exits, full mask, 64 threads", 0x5afee81ac3037582ull},
    {"WARPSYNC after exits, full mask, 100 threads", 0x7321745416921a54ull},
    {"WARPSYNC after exits, half-warp mask, 32 threads", 0x0bccd1903c8a151eull},
    {"WARPSYNC after exits, half-warp mask, 96 threads", 0xf7e1d7e6266f46c0ull},
    {"shared memory, shape 0, 32 threads", 0x318f560b2f96f501ull},
    {"shared memory, shape 0, 64 threads", 0x51bf71258ab808e9ull},
    {"shared memory, shape 0, 33 threads", 0xe766196c4d2c6c72ull},
    {"shared memory, shape 0, 5 threads", 0x95391eed8d5e18b7ull},
    {"shared memory, shape 0, 19 threads", 0x3affdb289d2492a1ull},
    {"shared memory, shape 1, 32 threads", 0x8b1bc614ee594f26ull},
    {"shared memory, shape 1, 64 threads", 0x7e228aa6f9b6581aull},
    {"shared memory, shape 1, 33 threads", 0x34f940dd1095dd29ull},
    {"shared memory, shape 1, 5 threads", 0x53f53a43e1271e39ull},
    {"shared memory, shape 1, 19 threads", 0x73d58d77a22e8382ull},
    {"shared memory, shape 2, 32 threads", 0x49354374d166adfbull},
    {"shared memory, shape 2, 64 threads", 0x2cdca3a4ea22a290ull},
    {"shared memory, shape 2, 33 threads", 0xcfe224058d081507ull},
    {"shared memory, shape 2, 5 threads", 0xbbafce50e8cb9a10ull},
    {"shared memory, shape 2, 19 threads", 0x767389f6da628d13ull},
    {"shared memory, shape 3, 32 threads", 0xb16714b0a71464a7ull},
    {"shared memory, shape 3, 64 threads", 0x8867c4c13c88cb8full},
    {"shared memory, shape 3, 33 threads", 0x99f995e3a1545be2ull},
    {"shared memory, shape 3, 5 threads", 0x601919b590ab584full},
    {"shared memory, shape 3, 19 threads", 0xb4d05aebc224a474ull},
    {"shared memory, shape 4, 32 threads", 0x7307cee53ea95f68ull},
    {"shared memory, shape 4, 64 threads", 0xcad3147a430bb438ull},
    {"shared memory, shape 4, 33 threads", 0x1bd52d33b35f568full},
    {"shared memory, shape 4, 5 threads", 0x4d88de7ae0ebb9b3ull},
    {"shared memory, shape 4, 19 threads", 0x2719fe590fda2480ull},
    {"shared memory, shape 5, 32 threads", 0x318f560b2f96f501ull},
    {"shared memory, shape 5, 64 threads", 0x51bf71258ab808e9ull},
    {"shared memory, shape 5, 33 threads", 0xe766196c4d2c6c72ull},
    {"shared memory, shape 5, 5 threads", 0x95391eed8d5e18b7ull},
    {"shared memory, shape 5, 19 threads", 0x3affdb289d2492a1ull},
    {"shared memory, shape 6, 32 threads", 0x6c1cf7c5dcfc1c2dull},
    {"shared memory, shape 6, 64 threads", 0x235656b0d2d6f0f6ull},
    {"shared memory, shape 6, 33 threads", 0x63a42e138ad494b7ull},
    {"shared memory, shape 6, 5 threads", 0xf82390066bff6451ull},
    {"shared memory, shape 6, 19 threads", 0xd49ab0418090f0a7ull},
    {"shared memory, shape 7, 32 threads", 0x50bd2589d566c7daull},
    {"shared memory, shape 7, 64 threads", 0x44dc43c16229321cull},
    {"shared memory, shape 7, 33 threads", 0x75b7429c6af42c9bull},
    {"shared memory, shape 7, 5 threads", 0xf686b948bc956795ull},
    {"shared memory, shape 7, 19 threads", 0xc011a2f229c740b2ull},
    {"shared memory, shape 8, 32 threads", 0xd78850e87964a44bull},
    {"shared memory, shape 8, 64 threads", 0x3cb182e00391971aull},
    {"shared memory, shape 8, 33 threads", 0xcb388d7774c15823ull},
    {"shared memory, shape 8, 5 threads", 0x01cad6db1508988cull},
    {"shared memory, shape 8, 19 threads", 0x47845e300956ba1aull},
    {"shared memory, shape 9, 32 threads", 0x8b7033574fcd833dull},
    {"shared memory, shape 9, 64 threads", 0x5b1e9254b573c3efull},
    {"shared memory, shape 9, 33 threads", 0x88f53d51d0efcddeull},
    {"shared memory, shape 9, 5 threads", 0xdee5b4e3393f5368ull},
    {"shared memory, shape 9, 19 threads", 0xf2205c545b88e010ull},
    {"shared memory, shape 10, 32 threads", 0xca67102f180ad703ull},
    {"shared memory, shape 10, 64 threads", 0xde009c0217eb8f9eull},
    {"shared memory, shape 10, 33 threads", 0xdea00bb839b1a696ull},
    {"shared memory, shape 10, 5 threads", 0x29ce1abf40625708ull},
    {"shared memory, shape 10, 19 threads", 0xa8187fd6233eddefull},
    {"shared memory, shape 11, 32 threads", 0x4ada5a229045abd0ull},
    {"shared memory, shape 11, 64 threads", 0x480388cc9438ede1ull},
    {"shared memory, shape 11, 33 threads", 0x760095857e61571aull},
    {"shared memory, shape 11, 5 threads", 0xe22829018291dd17ull},
    {"shared memory, shape 11, 19 threads", 0x3199a5486e5401dcull},
};

void check_golden(const Result& r, const std::string& key) {
  const uint64_t d = digest(r);
  if (std::getenv("VGPU_PRINT_DIGESTS")) {
    std::printf("    {\"%s\", 0x%016llxull},\n", key.c_str(), static_cast<unsigned long long>(d));
    return;
  }
  for (const Golden& g : kGolden)
    if (key == g.key) {
      if (g.digest != d)
        throw vtest::Failure(key + ": the answer is " + hex(static_cast<uint32_t>(d)) + "... where the tree before the tier ran these instructions gave " +
                             hex(static_cast<uint32_t>(g.digest)) + "...");
      return;
    }
  throw vtest::Failure(key + ": no recorded answer");
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
    check_golden(tier, name + ", " + std::to_string(threads) + " threads");
  }
}


// LDS / STS: Rd (or the data register) with the address Ra + imm, `size` the width code (0 U8, 4 32 bits, 5 64 bits), `stride` 0..3 for none, .X4, .X8, .X16.
Insn lds(unsigned d, unsigned a, uint32_t imm, unsigned size, unsigned stride) {
  Insn i = make(0x000fe20000000800ull | uint64_t{size} << 9 | uint64_t{stride} << 14, 0x7984 | uint64_t{d} << 16 | uint64_t{a} << 24 | uint64_t{imm & 0xffffffu} << 40, "");
  i.op = sass::Op::LDS;
  return i;
}
Insn sts(unsigned a, unsigned b, uint32_t imm, unsigned size, unsigned stride) {
  Insn i = make(0x000fe20000000800ull | uint64_t{size} << 9 | uint64_t{stride} << 14, 0x7388 | uint64_t{a} << 24 | uint64_t{b} << 32 | uint64_t{imm & 0xffffffu} << 40, "");
  i.op = sass::Op::STS;
  return i;
}

// Shared-memory accesses of a whole warp in every shape the tier tells apart: R7 is each lane's address, worked out from the thread number.
std::vector<Insn> shared_kernel(unsigned shape) {
  std::vector<Insn> code;
  code.push_back(s2r_tid(0));
  code.push_back(make(0x000fc60000000f00ull, 0x0000580000007a02ull | uint64_t{12} << 16, "MOV R12, c[0x0][0x160]"));
  code.push_back(make(0x000fc60000000f00ull, 0x0000590000007a02ull | uint64_t{13} << 16, "MOV R13, c[0x0][0x164]"));
  code.push_back(imad_wide_i(14, 0, kStride, 12));
  for (unsigned r = 1; r < kRegs; ++r) {
    code.push_back(imadi(r, 0, (0x9e3779b1u * r + 17) & 0x7fffffffu, kRZ));
    code.push_back(iadd3i(r, r, 0x1234567u * r, 0));
  }
  code.push_back(movi(8, 0x3fc));
  switch (shape) {
    case 0: code.push_back(imadi(7, 0, 4, kRZ)); break;                                   // a word to a lane, in order
    case 1: code.push_back(imadi(7, 0, 8, kRZ)); break;                                   // every other word
    case 2: code.push_back(imadi(7, 0, 37, kRZ)); code.push_back(lop3(7, 7, 8, kRZ, 0xc0)); break;   // scrambled, word-aligned
    case 3: code.push_back(imadi(7, 0, 3, kRZ)); break;                                   // bytes, three apart
    case 4: code.push_back(movi(7, 0)); break;                                            // one address for every lane
    case 5: code.push_back(iadd3i(7, 0, 0, kRZ)); break;                                  // the index; the .X4 forms scale it
    case 6: code.push_back(imadi(7, 0, 2, kRZ)); break;                                   // (.X8 scales to 16 bytes a lane)
    case 8:   // rows of eight words, a gap between the rows
      code.push_back(movi(8, 0x18)), code.push_back(lop3(9, 0, 8, kRZ, 0xc0)), code.push_back(imadi(7, 0, 4, kRZ)), code.push_back(imadi(7, 9, 24, 7));
      break;
    case 9:   // the same, each row starting two bytes into a word
      code.push_back(movi(8, 0x18)), code.push_back(lop3(9, 0, 8, kRZ, 0xc0)), code.push_back(imadi(7, 0, 4, kRZ)), code.push_back(imadi(7, 9, 24, 7));
      code.push_back(iadd3i(7, 7, 2, kRZ));
      break;
    case 10:  // runs of four words only: no group of eight is one run
      code.push_back(movi(8, 4)), code.push_back(lop3(9, 0, 8, kRZ, 0xc0)), code.push_back(imadi(7, 0, 4, kRZ)), code.push_back(imadi(7, 9, 20, 7));
      break;
    case 11:  // four rows on the same eight words (the later lane's value is the one stored)
      code.push_back(movi(8, 7)), code.push_back(lop3(9, 0, 8, kRZ, 0xc0)), code.push_back(imadi(7, 9, 4, kRZ));
      break;
    default: code.push_back(imadi(7, 0, 4, kRZ)); break;
  }
  const unsigned size = shape == 3 ? 0 : shape == 7 ? 5 : 4;   // U8, 64 bits, 32 bits
  const unsigned stride = shape == 5 ? 1 : shape == 6 ? 2 : 0;
  code.push_back(sts(7, 1, shape == 7 ? 8 : 0x20, size, stride));
  code.push_back(lds(2, 7, shape == 7 ? 8 : 0x20, size, stride));
  code.push_back(lds(3, 7, 0x44, 4, stride));   // (a word past it, or the one the lane's neighbour stored)
  code.push_back(imadi(5, 0, 4, kRZ));
  code.push_back(lds(5, 5, 0x20, 4, 0));   // (the words in order: where the stores put each lane's value, which a store and a load that agree on a wrong place would hide)
  code.push_back(iadd3(4, 2, 3, kRZ));
  for (unsigned r = 1; r < kRegs; ++r) code.push_back(stg(14, r, 4 * r));
  code.push_back(stg(14, 0, 0));
  code.push_back(exit_());
  return code;
}

}  // namespace

VTEST(a_call_by_every_lane_and_its_return_give_the_interpreters_registers_and_counters) {
  compare(kernel(3, false, RetKind::Plain, 0), {32, 64, 100}, "unguarded call");
}

VTEST(a_call_by_part_of_the_warp_runs_the_callee_for_those_lanes_only) {
  compare(kernel(3, true, RetKind::Plain, 0), {32, 33, 64, 100, 5, 31}, "guarded call");
}

VTEST(a_return_that_some_lanes_do_not_run_lets_them_fall_through) {
  compare(kernel(2, true, RetKind::Guarded, 0), {32, 64, 100, 20}, "guarded return");
  compare(kernel(2, false, RetKind::Guarded, 0), {32, 96}, "unguarded call, guarded return");
}

VTEST(warpsync_with_the_full_mask_and_with_a_register_mask) {
  compare(kernel(3, true, RetKind::Plain, 1), {32, 64, 33, 5}, "WARPSYNC full mask");
  compare(kernel(3, true, RetKind::Plain, 2), {32, 64, 33, 100}, "WARPSYNC half-warp mask");
  compare(kernel(2, false, RetKind::Plain, 2), {32, 96}, "WARPSYNC half-warp mask, unguarded call");
}

VTEST(warpsync_does_not_wait_for_lanes_that_have_exited) {
  // The lanes 16 to 23 leave in the callee (a RET they do not run is followed by an EXIT); a mask that names them is met by the lanes that remain.
  compare(kernel(2, true, RetKind::Guarded, 1), {32, 64, 100}, "WARPSYNC after exits, full mask");
  compare(kernel(2, true, RetKind::Guarded, 2), {32, 96}, "WARPSYNC after exits, half-warp mask");
}

VTEST(shared_memory_of_a_whole_warp_gives_the_interpreters_registers_and_counters) {
  for (unsigned shape = 0; shape < 12; ++shape) compare(shared_kernel(shape), {32, 64, 33, 5, 19}, "shared memory, shape " + std::to_string(shape));
}

#ifndef SASS_CALL_RET_NO_TIER_FORMS   // (not in the tree the answers were taken from)
VTEST(the_tier_runs_these_instructions_itself) {
  // Not vacuous: the tier's analysis takes each of the four forms (so the comparisons above compare the tier's code with the interpreter's).
  const std::vector<Insn> code = kernel(2, true, RetKind::Guarded, 2);
  bool call = false, ret = false, fchk = false, wsync = false;
  for (size_t i = 0; i < code.size(); ++i) {
    sass::Instr ins = sass::decode(code[i].w, 16 * i, 86);
    sass::annotate_instr(ins);
    sass::tier::Program prog;
    prog.ops.resize(1);
    sass::tier::analyze(ins, 86, prog, prog.ops[0]);
    call = call || prog.ops[0].form == sass::tier::Form::Call;
    ret = ret || prog.ops[0].form == sass::tier::Form::Ret;
    fchk = fchk || prog.ops[0].form == sass::tier::Form::Fchk;
    wsync = wsync || prog.ops[0].form == sass::tier::Form::Warpsync;
  }
  VCHECK(call && ret && fchk && wsync);
}

#endif

VTEST_MAIN
