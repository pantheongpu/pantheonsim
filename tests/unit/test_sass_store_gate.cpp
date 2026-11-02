// The SASS tier's store gate: a warp's first store to global memory ends its turn, once, so that the block's other warps issue the loads
// their prologues make before the store lands (Runner::step_warp; tier_run_t in src/sass/tier_ops.inc must stop where it does). On a card
// every warp of a block starts within a few cycles of the others and a store posts only after a load's round trip, so no warp sees the
// store of a warp-mate that started with it: Triton's `step += 1` on a one-element tensor has every warp load, add and store the same
// word, and the word ends as 1, not as the number of warps.
//
// The kernels are SASS for sm_86 written out word by word (nvdisasm's text of each word is checked against the decoder's). They run with
// the tier and with VGPU_FASTPATH=0 (the interpreter's own loop), and must leave the same word; the one-round kernel must leave 1 (the card's
// answer) whatever the number of warps. Shapes: the store a few instructions into the run (the lanes' addresses are then set to the store),
// and more than the run loop's 256 straight instructions before it (the store then starts a run of its own).
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

constexpr unsigned kRZ = 255;
uint64_t lo_of(uint64_t op, unsigned rd, unsigned ra, uint64_t b) { return op | uint64_t{rd} << 16 | uint64_t{ra} << 24 | b << 32; }
// guard: 7 none (PT), 0 @P0, 8 @!P0
Insn iadd3i(unsigned d, unsigned a, uint32_t imm, unsigned c, unsigned guard = 7) {
  return make(0x000fc80007ffe000ull | c, lo_of(0x0810 | (guard == 7 ? 0x7000 : guard << 12), d, a, imm),
              std::string(guard == 7 ? "" : guard == 8 ? "@!P0 " : "@P0 ") + "IADD3 " + reg(d) + ", " + reg(a) + ", " + hex(imm) + ", " + reg(c));
}
Insn exit_() { return make(0x000fea0003800000ull, 0x000000000000794dull, "EXIT"); }
// STG.E [Ra.64 + off], Rk
Insn stg(unsigned a, unsigned k, unsigned off) {
  return make(0x000fe2000c101904ull, lo_of(0x7986, 0, a, k) | uint64_t{off} << 40,
              "STG.E [" + reg(a) + ".64" + (off ? "+" + hex(off) : "") + "], " + reg(k));
}
// LDG.E Rd, [Ra.64]
Insn ldg(unsigned d, unsigned a) {
  return make(0x000ea2000c1e1900ull, lo_of(0x7981, d, a, 4), "LDG.E " + reg(d) + ", [" + reg(a) + ".64]");
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


// The kernel: R12:R13 = the parameter (one word); `pad` straight instructions on R2; then `rounds` times: load the word, add one, store it.
std::vector<Insn> kernel(unsigned pad, unsigned rounds) {
  std::vector<Insn> code;
  code.push_back(make(0x000fc60000000f00ull, 0x0000580000007a02ull | uint64_t{12} << 16, "MOV R12, c[0x0][0x160]"));
  code.push_back(make(0x000fc60000000f00ull, 0x0000590000007a02ull | uint64_t{13} << 16, "MOV R13, c[0x0][0x164]"));
  for (unsigned i = 0; i < pad; ++i) code.push_back(iadd3i(2, 2, 1, kRZ));
  for (unsigned r = 0; r < rounds; ++r) {
    code.push_back(ldg(1, 12));
    code.push_back(iadd3i(1, 1, 1, kRZ));
    code.push_back(stg(12, 1, 0));
  }
  code.push_back(exit_());
  return code;
}

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
  // The word after one block of `threads` threads has run.
  uint32_t run(sass::Module& m, uint32_t threads) {
    const uint64_t out = mem.alloc(16);
    const uint32_t zero[4] = {0, 0, 0, 0};
    mem.write(out, zero, 16);
    exec::LaunchConfig cfg;
    cfg.grid = {1, 1, 1};
    cfg.block = {threads, 1, 1};
    std::vector<uint8_t> arg(8);
    std::memcpy(arg.data(), &out, 8);
    sass::launch(m, "k", cfg, {arg}, mem, prof);
    uint32_t v = 0;
    mem.read(out, &v, 4);
    mem.free(out);
    return v;
  }
};

struct Scoped {
  Scoped() { ::setenv("VGPU_THREADS", "1", 1); }
  ~Scoped() {
    ::unsetenv("VGPU_THREADS");
    ::unsetenv("VGPU_FASTPATH");
  }
};

// Tier and interpreter on the same kernel; the tier's word is returned.
uint32_t both(const std::vector<Insn>& code, uint32_t threads, const std::string& name) {
  Scoped scoped;
  Env e;
  const auto m = e.load(code);
  const uint32_t tier = e.run(*m, threads);
  ::setenv("VGPU_FASTPATH", "0", 1);
  const uint32_t ref = e.run(*m, threads);
  ::unsetenv("VGPU_FASTPATH");
  if (tier != ref)
    throw vtest::Failure(name + ", " + std::to_string(threads) + " threads: the word is " + std::to_string(tier) + " with the tier and " + std::to_string(ref) +
                         " without");
  return tier;
}

}  // namespace

VTEST(every_warp_loads_before_the_first_store_lands) {
  // Four warps, one word: each loads 0, adds one and stores 1. Without the gate warp k would load what warps 0..k-1 stored and the word would end as 4.
  VCHECK_EQ(both(kernel(0, 1), 128, "one round"), 1u);
  VCHECK_EQ(both(kernel(0, 1), 256, "one round"), 1u);
  // One warp has no warp-mates to wait for, and part of a warp is the same.
  VCHECK_EQ(both(kernel(0, 1), 32, "one round"), 1u);
  VCHECK_EQ(both(kernel(0, 1), 5, "one round"), 1u);
}

VTEST(the_gate_holds_when_the_store_starts_a_run_of_its_own) {
  // 300 straight instructions first: the run loop takes 256 of them at a time, so the store is not in the run that began the pass.
  VCHECK_EQ(both(kernel(300, 1), 128, "300 instructions before"), 1u);
  VCHECK_EQ(both(kernel(256, 1), 128, "256 instructions before"), 1u);
  VCHECK_EQ(both(kernel(257, 1), 96, "257 instructions before"), 1u);
}

VTEST(only_the_first_store_of_a_warp_is_gated) {
  // Two rounds: the second store of each warp goes through at once. The tier and the interpreter agree on where the word ends.
  const uint32_t w = both(kernel(0, 2), 128, "two rounds");
  VCHECK(w >= 2u);
  both(kernel(3, 3), 96, "three rounds");
}

VTEST_MAIN
