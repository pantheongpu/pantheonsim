// LDGSTS (cp.async) of 4, 8 and 16 bytes, run by the SASS tier's own handler (h_ldgsts): each executing lane reads global memory and writes the bytes at its
// shared address, a lane whose predicate operand is false writes zeros, and lanes on one shared address leave the last lane's bytes. The expected shared
// memory is worked out here from that rule (not by the executor), read back through LDS and stored to global memory; the kernel runs with the tier and with
// VGPU_FASTPATH=0 (the interpreter), whose results and counters must be the same, for strides that are one run, gather, one address, and misaligned
// (which the memory manager's batch declines, so exec_mem finishes them), with a guard, a predicate operand, both or neither, and with .ZFILL (exec_mem's).
//
// The kernels are SASS for sm_86 written out word by word; the LDGSTS words are the encoding of a word of the sm_86 corpus with its fields replaced.
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
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

// guard: 7 none (PT), 0 @P0, 8 @!P0
Insn s2r(unsigned d, unsigned sr, unsigned guard = 7) { return make(0x000e220000000000ull | uint64_t{sr} << 8, lo_of(0x0919 | guard << 12, d, 0, 0)); }
Insn imadi(unsigned d, unsigned a, uint32_t imm, unsigned c) { return make(0x000fe400078e0200ull | c, lo_of(0x7824, d, a, imm)); }
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

constexpr uint32_t kIn = 8192, kShared = 4096;

Insn lds(unsigned d, unsigned a, uint32_t imm) { return make(0x000e220000000800ull, 0x7984ull | uint64_t{d} << 16 | uint64_t{a} << 24 | uint64_t{imm} << 40); }

// guard: 7 none, 0 @P0, 8 @!P0. size: 4 (4 bytes), 5 (8), 6 (16). pred: -1 none, 0 P0, 1 !P0 (the predicate operand).
Insn ldgsts(unsigned size, unsigned rs, unsigned rg, uint32_t soff, uint32_t goff, int pred, bool zfill, unsigned guard) {
  uint64_t hi = 0x0003e2000b901d5aull;
  hi = (hi & ~(uint64_t{0xf} << 23) & ~(uint64_t{7} << 9) & ~(uint64_t{1} << 18)) | uint64_t{size} << 9;
  hi |= (pred < 0 ? uint64_t{7} : uint64_t{0}) << 23;
  if (pred == 1) hi |= uint64_t{1} << 26;
  if (zfill) hi |= uint64_t{1} << 18;
  const uint64_t lo = (0x7faeull & ~uint64_t{0xf000}) | uint64_t{guard} << 12 | uint64_t{rs} << 16 | uint64_t{rg} << 24 | uint64_t{goff & 0xfff} << 32 |
                      uint64_t{soff & 0xfffff} << 44;
  return make(hi, lo);
}

std::vector<Insn> kernel(unsigned size, uint32_t S, uint32_t D, uint32_t soff, uint32_t goff, int pred, bool zfill, unsigned guard) {
  std::vector<Insn> code;
  code.push_back(s2r(1, 0x21));
  code.push_back(mov_param(6, 0x58));
  code.push_back(mov_param(7, 0x59));
  code.push_back(isetp_ge(1, 16));              // P0 = (tid >= 16)
  code.push_back(imadi(2, 1, D, kRZ));          // R2 = tid * D, the shared address
  code.push_back(imad_wide_i(4, 1, S, 6));      // R4:R5 = base + tid * S, the global one
  code.push_back(imadi(3, 1, 4, kRZ));
  code.push_back(ldgsts(size, 2, 4, soff, goff, pred, zfill, guard));
  code.push_back(imad_wide_i(8, 1, 32, 6));
  for (unsigned j = 0; j < 8; ++j) {
    code.push_back(lds(10, 3, 128 * j));   // word tid + 32 j of the shared memory
    code.push_back(stg(8, 10, kIn + 4 * j));
  }
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
};

struct Scoped {
  Scoped() { ::setenv("VGPU_THREADS", "1", 1); }
  ~Scoped() {
    ::unsetenv("VGPU_THREADS");
    ::unsetenv("VGPU_FASTPATH");
  }
};

Result run(Env& e, const std::vector<Insn>& code, uint32_t threads, const std::vector<uint8_t>& input, bool interp) {
  for (size_t i = 0; i < code.size(); ++i) {
    const sass::Instr ins = sass::decode(code[i].w, 16 * i, 86);
    if (i == 7 && ins.op != sass::Op::LDGSTS) throw vtest::Failure("word 7 is \"" + sass::to_text(ins) + "\", not an LDGSTS");
    if (i >= 9 && i < 25 && (i % 2) == 1 && ins.op != sass::Op::LDS) throw vtest::Failure("word " + std::to_string(i) + " is \"" + sass::to_text(ins) + "\", not an LDS");
  }
  const std::string c = make_cubin(code);
  const auto m = sass::load(reinterpret_cast<const uint8_t*>(c.data()), c.size(), e.mem, e.prof);
  if (interp) ::setenv("VGPU_FASTPATH", "0", 1);
  const uint32_t out_bytes = threads * 32;
  const uint64_t base = e.mem.alloc(kIn + out_bytes);
  e.mem.write(base, input.data(), kIn);
  exec::LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {threads, 1, 1};
  cfg.shared_bytes = kShared;
  std::vector<uint8_t> arg(8);
  std::memcpy(arg.data(), &base, 8);
  Result r;
  r.stats = sass::launch(*m, "k", cfg, {arg}, e.mem, e.prof);
  ::unsetenv("VGPU_FASTPATH");
  r.out.resize(out_bytes / 4);
  e.mem.read(base + kIn, r.out.data(), out_bytes);
  e.mem.free(base);
  return r;
}

struct Case {
  unsigned size, bytes;
  uint32_t S, D, soff, goff;
  int pred;
  bool zfill;
  unsigned guard;
  uint32_t threads;
  std::string name() const {
    return "LDGSTS." + std::to_string(bytes * 8) + " S=" + std::to_string(S) + " D=" + std::to_string(D) + " soff=" + std::to_string(soff) + " goff=" + std::to_string(goff) +
           " pred=" + std::to_string(pred) + (zfill ? " zfill" : "") + " guard=" + std::to_string(guard) + " threads=" + std::to_string(threads);
  }
};

// The shared memory the rule gives, read back as the kernel does (word tid + 32 j of thread tid), for the threads in order.
std::vector<uint32_t> expected(const Case& k, const std::vector<uint8_t>& in) {
  std::vector<uint8_t> sh(kShared, 0);
  for (uint32_t t = 0; t < k.threads; ++t) {
    const bool p0 = t >= 16;
    if (k.guard == 0 && !p0) continue;
    if (k.guard == 8 && p0) continue;
    const bool read = k.pred < 0 || (k.pred == 0 ? p0 : !p0);
    const uint64_t g = uint64_t{k.S} * t + k.goff;
    uint8_t buf[16] = {};
    uint32_t keep = read ? k.bytes : 0;
    if (read && k.zfill) {
      const uint32_t fill = static_cast<uint32_t>(g & (k.bytes - 1));
      const uint64_t g2 = g - fill;
      keep = k.bytes - fill;
      std::memcpy(buf, &in[g2], keep);
    } else if (keep) {
      std::memcpy(buf, &in[g], keep);
    }
    std::memcpy(&sh[uint64_t{k.D} * t + k.soff], buf, k.bytes);
  }
  std::vector<uint32_t> out(k.threads * 8);
  for (uint32_t t = 0; t < k.threads; ++t)
    for (uint32_t j = 0; j < 8; ++j) std::memcpy(&out[t * 8 + j], &sh[4 * t + 128 * j], 4);
  return out;
}

void check(Env& e, const Case& k, const std::vector<uint8_t>& input) {
  const auto code = kernel(k.size, k.S, k.D, k.soff, k.goff, k.pred, k.zfill, k.guard);
  const Result tier = run(e, code, k.threads, input, false), ref = run(e, code, k.threads, input, true);
  const std::vector<uint32_t> want = expected(k, input);
  for (size_t i = 0; i < want.size(); ++i) {
    if (ref.out[i] != want[i])
      throw vtest::Failure(k.name() + " (interpreter): thread " + std::to_string(i / 8) + " word " + std::to_string(i % 8) + " is " + std::to_string(ref.out[i]) + ", not " + std::to_string(want[i]));
    if (tier.out[i] != want[i])
      throw vtest::Failure(k.name() + " (tier): thread " + std::to_string(i / 8) + " word " + std::to_string(i % 8) + " is " + std::to_string(tier.out[i]) + ", not " + std::to_string(want[i]));
  }
  if (std::memcmp(&tier.stats, &ref.stats, exec::kCounterWords * sizeof(uint64_t)) != 0) {
    const uint64_t* a = reinterpret_cast<const uint64_t*>(&tier.stats);
    const uint64_t* b = reinterpret_cast<const uint64_t*>(&ref.stats);
    size_t i = 0;
    while (a[i] == b[i]) ++i;
    throw vtest::Failure(k.name() + ": counter word " + std::to_string(i) + " is " + std::to_string(a[i]) + " with the tier and " + std::to_string(b[i]) + " without");
  }
}

std::vector<uint8_t> random_input() {
  std::mt19937 rng(7);
  std::vector<uint8_t> v(kIn);
  for (auto& b : v) b = static_cast<uint8_t>(rng());
  return v;
}

void sweep(const std::vector<std::array<uint32_t, 2>>& sd, std::initializer_list<uint32_t> threads, bool zfill) {
  Scoped scoped;
  Env e;
  const std::vector<uint8_t> input = random_input();
  for (unsigned size = 4; size <= 6; ++size)
    for (int pred = -1; pred <= 1; ++pred)
      for (unsigned guard : {7u, 0u, 8u})
        for (const auto& p : sd)
          for (uint32_t t : threads) {
            Case k{size, size == 4 ? 4u : size == 5 ? 8u : 16u, p[0], p[1], 0x40, 0x20, pred, zfill, guard, t};
            check(e, k, input);
          }
}

}  // namespace

VTEST(copies_on_a_run_or_a_stride_one_warp) { sweep({{16, 16}, {32, 16}, {64, 24}, {16, 20}}, {32, 20}, false); }

VTEST(copies_of_lanes_on_one_address_or_overlapping_shared_words) { sweep({{0, 8}, {16, 0}, {4, 4}, {8, 16}, {16, 12}}, {32}, false); }

VTEST(copies_of_two_warps) { sweep({{16, 16}, {32, 16}}, {64}, false); }

VTEST(zfill_copies_are_the_interpreters) { sweep({{16, 16}, {20, 16}, {36, 24}}, {32}, true); }

VTEST_MAIN
