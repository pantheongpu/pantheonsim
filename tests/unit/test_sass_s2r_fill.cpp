// S2R of SR_TID.X/Y/Z, SR_LANEID, SR_CTAID.X/Y/Z and SR_NTID (the special registers the SASS tier fills a warp with in one pass, and
// Runner::sreg gives one lane at a time from a table built per launch) must give each thread what the hardware does: x = tid % bx,
// y = (tid / bx) % by, z = tid / (bx * by), the lane, the block's coordinates and its thread count. The expected values are computed
// here from those formulas, not by the executor, so a wrong table (or a wrong fill of a partial warp) fails; the kernels run with the
// tier and with VGPU_FASTPATH=0 (the interpreter, one lane at a time), for block shapes that are not powers of two, partial warps and
// 3D blocks, and with guarded S2Rs, whose masked lanes must keep the value they had.
//
// The kernels are SASS for sm_86 written out word by word (the encodings are not checked against nvdisasm text here: a wrong one fails
// as a wrong value).
#include <array>
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
};

Insn make(uint64_t hi, uint64_t lo) {
  Insn i;
  i.w.hi = hi, i.w.lo = lo;
  return i;
}
// lo: opcode | rd << 16 | ra << 24 | rb or immediate << 32.
uint64_t lo_of(uint64_t op, unsigned rd, unsigned ra, uint64_t b) { return op | uint64_t{rd} << 16 | uint64_t{ra} << 24 | b << 32; }

constexpr unsigned kRZ = 255;
constexpr uint32_t kStride = 0x40;   // bytes a thread's results take
constexpr uint32_t kPoison = 0x5A5A5A5Au;

// guard: 7 none (PT), 0 @P0, 8 @!P0
Insn s2r(unsigned d, unsigned sr, unsigned guard = 7) { return make(0x000e220000000000ull | uint64_t{sr} << 8, lo_of(0x0919 | guard << 12, d, 0, 0)); }
Insn imadi(unsigned d, unsigned a, uint32_t imm, unsigned c) { return make(0x000fe400078e0200ull | c, lo_of(0x7824, d, a, imm)); }
Insn movi(unsigned d, uint32_t imm) { return make(0x000fe20000000f00ull, lo_of(0x7802, d, 0, imm)); }
Insn isetp_ge(unsigned a, uint32_t imm) { return make(0x000fc40003f06270ull, lo_of(0x780c, 0, a, imm)); }   // P0 = (Ra >= imm), signed
Insn imad_wide_i(unsigned d, unsigned a, uint32_t imm, unsigned c) { return make(0x000fe200078e0000ull | c, lo_of(0x7825, d, a, imm)); }
Insn mov_param(unsigned d, unsigned word) { return make(0x000fc60000000f00ull, 0x7a02ull | uint64_t{d} << 16 | uint64_t{word} << 40); }   // MOV Rd, c[0x0][word * 4]
Insn stg(unsigned a, unsigned k, unsigned off) { return make(0x000fe2000c101904ull, lo_of(0x7986, 0, a, k) | uint64_t{off} << 40); }
Insn exit_() { return make(0x000fea0003800000ull, 0x000000000000794dull); }

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

// Registers: R1 x, R2 y, R3 z, R4 lane, R5..R7 ctaid, R8 ntid, R9 x under @P0, R10 z under @!P0, R11 lane under @P0 (P0 = x >= 16),
// R12:R13 the pointer, R14:R15 the address, R0 the thread's slot, R15 (before the address) the block's.
std::vector<Insn> kernel(const uint32_t block[3], const uint32_t grid[3]) {
  std::vector<Insn> code;
  code.push_back(s2r(1, 0x21));
  code.push_back(s2r(2, 0x22));
  code.push_back(s2r(3, 0x23));
  code.push_back(s2r(4, 0x00));
  code.push_back(s2r(5, 0x25));
  code.push_back(s2r(6, 0x26));
  code.push_back(s2r(7, 0x27));
  code.push_back(s2r(8, 0x28));
  code.push_back(movi(9, kPoison));
  code.push_back(movi(10, kPoison));
  code.push_back(movi(11, kPoison));
  code.push_back(isetp_ge(1, 16));
  code.push_back(s2r(9, 0x21, 0));
  code.push_back(s2r(10, 0x23, 8));
  code.push_back(s2r(11, 0x00, 0));
  code.push_back(mov_param(12, 0x58));
  code.push_back(mov_param(13, 0x59));
  code.push_back(imadi(0, 3, block[1], 2));    // slot = (z * by + y) * bx + x
  code.push_back(imadi(0, 0, block[0], 1));
  code.push_back(imadi(15, 7, grid[1], 6));     // block = (cz * gy + cy) * gx + cx
  code.push_back(imadi(15, 15, grid[0], 5));
  code.push_back(imadi(0, 15, block[0] * block[1] * block[2], 0));
  code.push_back(imad_wide_i(14, 0, kStride, 12));
  for (unsigned r = 1; r <= 11; ++r) code.push_back(stg(14, r, 4 * r));
  code.push_back(exit_());
  (void)kRZ;
  return code;
}

struct Env {
  MemoryManager mem{1 << 24};
  DeviceProfile prof = load_gpu("nvidia/a10");   // sm_86
};

struct Scoped {
  Scoped() { ::setenv("VGPU_THREADS", "1", 1); }
  ~Scoped() {
    ::unsetenv("VGPU_THREADS");
    ::unsetenv("VGPU_FASTPATH");
  }
};

void check(Env& e, const uint32_t block[3], const uint32_t grid[3], const std::string& what) {
  const std::string c = make_cubin(kernel(block, grid));
  const auto m = sass::load(reinterpret_cast<const uint8_t*>(c.data()), c.size(), e.mem, e.prof);
  const uint32_t threads = block[0] * block[1] * block[2];
  const uint32_t blocks = grid[0] * grid[1] * grid[2];
  const uint32_t words = blocks * threads * kStride / 4;
  for (int interp = 0; interp < 2; ++interp) {
    if (interp) ::setenv("VGPU_FASTPATH", "0", 1);
    const uint64_t out = e.mem.alloc(words * 4);
    const std::vector<uint32_t> poison(words, kPoison);
    e.mem.write(out, poison.data(), words * 4);
    exec::LaunchConfig cfg;
    cfg.grid = {grid[0], grid[1], grid[2]};
    cfg.block = {block[0], block[1], block[2]};
    std::vector<uint8_t> arg(8);
    std::memcpy(arg.data(), &out, 8);
    sass::launch(*m, "k", cfg, {arg}, e.mem, e.prof);
    std::vector<uint32_t> got(words);
    e.mem.read(out, got.data(), words * 4);
    e.mem.free(out);
    ::unsetenv("VGPU_FASTPATH");
    for (uint32_t b = 0; b < blocks; ++b)
      for (uint32_t t = 0; t < threads; ++t) {
        const uint32_t x = t % block[0], y = (t / block[0]) % block[1], z = t / (block[0] * block[1]);
        const uint32_t cx = b % grid[0], cy = (b / grid[0]) % grid[1], cz = b / (grid[0] * grid[1]);
        const uint32_t want[11] = {x, y, z, t % 32, cx, cy, cz, threads, x >= 16 ? x : kPoison, x >= 16 ? kPoison : z, x >= 16 ? t % 32 : kPoison};
        for (unsigned r = 0; r < 11; ++r) {
          const uint32_t v = got[(size_t{b} * threads + t) * (kStride / 4) + 1 + r];
          if (v != want[r])
            throw vtest::Failure(what + (interp ? " (interpreter)" : " (tier)") + ": block " + std::to_string(b) + " thread " + std::to_string(t) +
                                 " register R" + std::to_string(1 + r) + " is " + std::to_string(v) + ", not " + std::to_string(want[r]));
        }
      }
  }
}

void shapes(std::initializer_list<std::array<uint32_t, 3>> list, const uint32_t grid[3], const std::string& name) {
  Scoped scoped;
  Env e;
  for (const auto& s : list) {
    const uint32_t b[3] = {s[0], s[1], s[2]};
    check(e, b, grid, name + " block " + std::to_string(s[0]) + "x" + std::to_string(s[1]) + "x" + std::to_string(s[2]));
  }
}

}  // namespace

VTEST(special_registers_of_one_dimensional_blocks) {
  const uint32_t grid[3] = {2, 1, 1};
  shapes({{32, 1, 1}, {1, 1, 1}, {5, 1, 1}, {31, 1, 1}, {33, 1, 1}, {96, 1, 1}, {100, 1, 1}, {1024, 1, 1}}, grid, "1D");
}

VTEST(special_registers_of_two_and_three_dimensional_blocks) {
  const uint32_t grid[3] = {2, 3, 2};
  shapes({{8, 4, 1}, {16, 16, 1}, {7, 5, 1}, {33, 3, 1}, {4, 4, 4}, {5, 3, 2}, {3, 5, 7}, {2, 3, 11}, {1, 1, 40}, {1, 33, 1}, {17, 2, 3}}, grid, "3D");
}

VTEST(special_registers_of_a_block_of_the_full_thousand_and_twenty_four_threads) {
  const uint32_t grid[3] = {1, 1, 2};
  shapes({{16, 16, 4}, {8, 8, 16}, {1, 1, 1024}, {32, 32, 1}}, grid, "1024");
}

VTEST_MAIN
