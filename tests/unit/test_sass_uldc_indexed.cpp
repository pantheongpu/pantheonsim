// ULDC with a uniform base register (c[bank][URa + offset]) takes its offset as a signed 16-bit number and wraps the address in the 64 KiB
// bank, as LDC does. ptxas reaches byte 0x8d20 of a bank as `ULDC.64 UR6, c[0x3][UR8+-0x72e0]` with UR8 holding 0x10000; adding the offset
// as 0x8d20 without the wrap read 0x18d20, past the bank, and gave 0 (pantheonsim #499: an inverse NTT that multiplied by a constant it read
// as zero). The words are the ones nvcc 13.0 -arch=sm_80 emits for that kernel; one cubin holds them, a bank 3 of 0x9000 bytes and nothing else.
#include <cstdint>
#include <cstdio>
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

Insn make(uint64_t hi, uint64_t lo, std::string text) {
  Insn i;
  i.w.hi = hi, i.w.lo = lo;
  i.text = std::move(text);
  return i;
}

constexpr uint64_t kBank3Size = 0x9000;
uint32_t bank3_word(uint32_t at) { return 0xa0000000u + at; }   // (the bytes of a word that start at `at`: distinct everywhere)

// ULDC URd, c[bank][URa + off], or with `ua` < 0 ULDC URd, c[bank][off]; hi's bits 9-11 hold the size (0 U8, 4 32-bit, 5 64-bit).
Insn uldc(unsigned d, int ua, int off, unsigned size, const std::string& mod) {
  const uint64_t lo = (ua < 0 ? 0x7ab9ull : 0x7abbull) | uint64_t{d} << 16 | (ua < 0 ? 0ull : uint64_t(ua) << 24) |
                      (uint64_t(off) & 0xffff) << 38 | uint64_t{3} << 54;
  const uint64_t hi = 0x000fe40008000000ull | uint64_t{size} << 9;
  char buf[64];
  const std::string o = off < 0 ? "-0x" : "0x";
  const unsigned mag = static_cast<unsigned>(off < 0 ? -off : off);
  if (ua < 0)
    buf[0] = 0;   // (the plain form's spelling is not what this test is about)
  else
    std::snprintf(buf, sizeof buf, "ULDC%s UR%u, c[0x3][UR%d+%s%x]", mod.c_str(), d, ua, o.c_str(), mag);
  return make(hi, lo, buf);
}
Insn umov(unsigned d, uint32_t imm) {
  char buf[48];
  std::snprintf(buf, sizeof buf, "UMOV UR%u, 0x%x", d, imm);
  return make(0x000fe20000000000ull, 0x7882ull | uint64_t{d} << 16 | uint64_t{imm} << 32, buf);
}
Insn mov_r_ur(unsigned d, unsigned ur) {
  return make(0x000fc80008000f00ull, 0x7c02ull | uint64_t{d} << 16 | uint64_t{ur} << 32, "MOV R" + std::to_string(d) + ", UR" + std::to_string(ur));
}
Insn mov_param(unsigned d, unsigned off) {   // MOV Rd, c[0x0][off]
  char buf[48];
  std::snprintf(buf, sizeof buf, "MOV R%u, c[0x0][0x%x]", d, off);
  return make(0x000fc60000000f00ull, 0x7a02ull | uint64_t{d} << 16 | uint64_t{off} << 38, buf);
}
// STG.E [R12.64+off], Rk
Insn stg(unsigned k, unsigned off) {
  char buf[48];
  if (off)
    std::snprintf(buf, sizeof buf, "STG.E [R12.64+0x%x], R%u", off, k);
  else
    std::snprintf(buf, sizeof buf, "STG.E [R12.64], R%u", k);
  return make(0x000fe2000c101904ull, 0x7986ull | uint64_t{12} << 24 | uint64_t{k} << 32 | uint64_t{off} << 40, buf);
}
Insn exit_() { return make(0x000fea0003800000ull, 0x000000000000794dull, "EXIT"); }

template <class T>
void put(std::string& s, T v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof v);
}

// A cubin for sm_86: kernel k takes one 8-byte pointer; the module has a .nv.constant3 of kBank3Size bytes.
std::string make_cubin(const std::vector<Insn>& code) {
  const std::string names("\0.shstrtab\0.text.k\0.nv.info.k\0.nv.constant3\0", 45);
  std::string info;   // KPARAM_INFO: index, ordinal | offset << 16, size << 18
  put<uint8_t>(info, 4), put<uint8_t>(info, 0x17), put<uint16_t>(info, 12);
  put<uint32_t>(info, 0), put<uint32_t>(info, 0), put<uint32_t>(info, 8u << 18);
  std::string bank;
  for (uint32_t at = 0; at < kBank3Size; at += 4) put<uint32_t>(bank, bank3_word(at));
  std::string out(64, '\0');
  const uint64_t names_off = out.size();
  out += names;
  while (out.size() % 16) out.push_back('\0');
  const uint64_t text_off = out.size();
  for (const Insn& i : code) put(out, i.w.lo), put(out, i.w.hi);
  const uint64_t info_off = out.size();
  out += info;
  while (out.size() % 8) out.push_back('\0');
  const uint64_t bank_off = out.size();
  out += bank;
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
  shdr(30, 1, bank_off, bank.size());
  out[0] = 0x7f, out[1] = 'E', out[2] = 'L', out[3] = 'F', out[4] = 2, out[5] = 1, out[6] = 1, out[7] = 0x41, out[8] = 8;
  const uint16_t type = 2, machine = 190, shentsize = 64, shnum = 5, shstrndx = 1;
  const uint32_t flags = 0x06000002u | (86u << 8);
  std::memcpy(&out[16], &type, 2), std::memcpy(&out[18], &machine, 2);
  std::memcpy(&out[0x28], &shoff, 8), std::memcpy(&out[0x30], &flags, 4);
  std::memcpy(&out[0x3a], &shentsize, 2), std::memcpy(&out[0x3c], &shnum, 2), std::memcpy(&out[0x3e], &shstrndx, 2);
  return out;
}

std::vector<uint32_t> run(const std::vector<Insn>& code, unsigned words) {
  MemoryManager mem{1 << 24};
  DeviceProfile prof = load_gpu("nvidia/a10");   // sm_86
  for (size_t i = 0; i < code.size(); ++i) {
    const sass::Instr ins = sass::decode(code[i].w, 16 * i, 86);
    if (!code[i].text.empty() && sass::to_text(ins) != code[i].text)
      throw vtest::Failure("word " + std::to_string(i) + " is \"" + sass::to_text(ins) + "\", not \"" + code[i].text + "\"");
  }
  const std::string c = make_cubin(code);
  std::shared_ptr<sass::Module> m = sass::load(reinterpret_cast<const uint8_t*>(c.data()), c.size(), mem, prof);
  const uint64_t out = mem.alloc(words * 4);
  const std::vector<uint32_t> zero(words, 0);
  mem.write(out, zero.data(), words * 4);
  exec::LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {1, 1, 1};
  std::vector<uint8_t> arg(8);
  std::memcpy(arg.data(), &out, 8);
  sass::launch(*m, "k", cfg, {arg}, mem, prof);
  std::vector<uint32_t> r(words);
  mem.read(out, r.data(), words * 4);
  return r;
}

}  // namespace

VTEST(an_indexed_uldc_with_a_negative_offset_wraps_in_the_bank) {
  std::vector<Insn> c;
  c.push_back(mov_param(12, 0x160));
  c.push_back(mov_param(13, 0x164));
  c.push_back(umov(8, 0x10000));                         // (ptxas's compensation for the negative offset)
  c.push_back(uldc(6, 8, -0x72e0, 5, ".64"));            // the issue's word: bytes 0x8d20 and 0x8d24
  c.push_back(uldc(10, 8, -0x72e0, 4, ""));              // the 32-bit form
  c.push_back(uldc(11, 8, -0x72df, 0, ".U8"));           // a byte of the same word
  c.push_back(uldc(12, -1, 0x8d20, 4, ""));              // the plain form, whose offset is unsigned
  c.push_back(uldc(13, 8, 0x20, 4, ""));                 // a positive offset past the wrap: 0x10020 is 0x20
  c.push_back(umov(9, 0x7ff0));
  c.push_back(uldc(14, 9, 0x10, 4, ""));                 // no wrap involved: 0x8000
  const unsigned srcs[] = {6, 7, 10, 11, 12, 13, 14};
  for (unsigned i = 0; i < 7; ++i) c.push_back(mov_r_ur(2 + i, srcs[i]));
  for (unsigned i = 0; i < 7; ++i) c.push_back(stg(2 + i, 4 * i));
  c.push_back(exit_());
  const std::vector<uint32_t> got = run(c, 7);
  VCHECK_EQ(got[0], bank3_word(0x8d20));
  VCHECK_EQ(got[1], bank3_word(0x8d24));
  VCHECK_EQ(got[2], bank3_word(0x8d20));
  VCHECK_EQ(got[3], (bank3_word(0x8d20) >> 8) & 0xff);
  VCHECK_EQ(got[4], bank3_word(0x8d20));
  VCHECK_EQ(got[5], bank3_word(0x20));
  VCHECK_EQ(got[6], bank3_word(0x8000));
}

VTEST_MAIN
