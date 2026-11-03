// (Mode 3, below, is LDG and STG on the same address rows: the tier's h_global, whose full-warp path reads a warp's address rows once for the
// addresses, a lane in the constant window, a run of consecutive elements and a stride, and counts a stride's sectors from its ends.)
// LD and ST with generic 64-bit addresses (the tier's h_generic): a warp whose lanes all name the shared window moves like LDS/STS and is counted as a
// shared access of the offsets; a warp with no lane in any window is plain global memory (LDG/STG's batches); a warp with lanes in both is exec_mem's,
// which moves each lane by its window and counts the whole warp by the LAST lane's. Bytes, registers, the shared array, the global buffer and every
// counter are checked against the rule written out here (not the executor) and against the interpreter (VGPU_FASTPATH=0), for 1 to 16 byte accesses
// with sign and zero extension, a guard on each instruction, strides that are one run, a gather, one address and misaligned, and lanes on one address.
//
// The kernels are SASS for sm_86 written out word by word; the LD and ST words are the encodings of words of the sm_86 corpus with their fields replaced.
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

Insn movi(unsigned d, uint32_t imm) { return make(0x000fe20000000f00ull, lo_of(0x7802, d, 0, imm)); }
Insn imad_wide_g(unsigned d, unsigned a, uint32_t imm, unsigned c, unsigned guard) { return make(0x000fe200078e0000ull | c, lo_of(0x0825 | guard << 12, d, a, imm)); }
// IMAD.WIDE.U32 Rd, P0, Ra, imm, Rc: the carry out of the 64-bit sum goes to P0 (bits 81..83 of the word, PT in the others)
Insn imad_wide_p0(unsigned d, unsigned a, uint32_t imm, unsigned c) { return make(0x000fe20007800000ull | c, lo_of(0x7825, d, a, imm)); }
Insn lds(unsigned d, unsigned a, uint32_t imm) { return make(0x000e220000000800ull, 0x7984ull | uint64_t{d} << 16 | uint64_t{a} << 24 | uint64_t{imm} << 40); }

// size codes of the instruction: 0 U8, 1 S8, 2 U16, 3 S16, 4 32 bits, 5 64, 6 128. guard: 7 none, 0 @P0, 8 @!P0.
Insn ld_e(unsigned size, unsigned d, unsigned a, uint32_t off, unsigned guard) {
  const uint64_t hi = (0x000ea8000c101900ull & ~(uint64_t{7} << 9)) | uint64_t{size} << 9;
  return make(hi, 0x0980ull | uint64_t{guard} << 12 | uint64_t{d} << 16 | uint64_t{a} << 24 | uint64_t{4} << 32 | uint64_t{off & 0xffffff} << 40);
}
Insn st_e(unsigned size, unsigned a, unsigned data, uint32_t off, unsigned guard) {
  const uint64_t hi = (0x0001e2000c101904ull & ~(uint64_t{7} << 9)) | uint64_t{size} << 9;
  return make(hi, 0x0985ull | uint64_t{guard} << 12 | uint64_t{a} << 24 | uint64_t{data} << 32 | uint64_t{off & 0xffffff} << 40);
}

// LDG and STG: the same operands as LD and ST (opcodes 0x981 and 0x986 with the guard in bits 12..15, as the sm_86 corpus has them).
Insn ldg_e(unsigned size, unsigned d, unsigned a, uint32_t off, unsigned guard) {
  const uint64_t hi = (0x000ea8000c1e1900ull & ~(uint64_t{7} << 9)) | uint64_t{size} << 9;
  return make(hi, 0x0981ull | uint64_t{guard} << 12 | uint64_t{d} << 16 | uint64_t{a} << 24 | uint64_t{4} << 32 | uint64_t{off & 0xffffff} << 40);
}
Insn stg_e(unsigned size, unsigned a, unsigned data, uint32_t off, unsigned guard) {
  const uint64_t hi = (0x0001e2000c101904ull & ~(uint64_t{7} << 9)) | uint64_t{size} << 9;
  return make(hi, 0x0986ull | uint64_t{guard} << 12 | uint64_t{a} << 24 | uint64_t{data} << 32 | uint64_t{off & 0xffffff} << 40);
}

constexpr uint32_t kK[4] = {0x9e3779b1u, 0x85ebca6bu, 0xc2b2ae35u, 0x27d4eb2fu}, kC = 0x01234567u;

// Registers: R1 tid, R3 tid * 4, R4:R5 the address, R6:R7 the buffer, R8:R9 the output, R10 a word read back, R11 a constant, R12..R15 the data.
// mode 0: every lane's address is in the shared window (tid * D), 1: global (base + tid * S), 2: shared for tid < 16 and global for the rest, 3: as 1 with LDG and STG, 4: as 3 with the lanes in reverse order (lane l at base + (31 - l) * S: a warp that goes DOWN memory), 5: as 3 with the stride doubling from lane 16 (a full warp in global memory that is NOT on a stride), 6: as 3 with every lane of the first warp at the one address (P0 is tid >= 32 here) and the lanes of the others at base + tid * S (the shape of a full warp changes from one warp of the block to the next).
std::vector<Insn> kernel(unsigned mode, uint32_t S, uint32_t D, unsigned st_size, unsigned ld_size, uint32_t soff, uint32_t loff, unsigned st_guard, unsigned ld_guard) {
  std::vector<Insn> code;
  code.push_back(s2r(1, 0x21));
  code.push_back(mov_param(6, 0x58));
  code.push_back(mov_param(7, 0x59));
  code.push_back(isetp_ge(1, mode == 6 ? 32 : 16));   // P0 = (tid >= 16), (tid >= 32) for mode 6
  code.push_back(movi(11, kC));
  for (unsigned i = 0; i < 4; ++i) code.push_back(imadi(12 + i, 1, kK[i], 11));
  code.push_back(imadi(3, 1, 4, kRZ));
  if (mode == 4) {
    code.push_back(movi(11, 31));
    code.push_back(imadi(2, 1, 0xffffffffu, 11));   // R2 = 31 - tid
    code.push_back(imad_wide_i(4, 2, S, 6));
  } else if (mode == 6) {
    code.push_back(imad_wide_i(4, 1, 0, 6));   // (every lane at the base)
    code.push_back(imad_wide_g(4, 1, S, 6, 0));   // (@P0, the lanes of the later warps: base + tid * S)
  } else if (mode == 1 || mode == 3 || mode == 5) {
    code.push_back(imad_wide_i(4, 1, S, 6));
    if (mode == 5) code.push_back(imad_wide_g(4, 1, 2 * S, 6, 0));   // (@P0, lanes 16 and up: twice the stride)
  } else {
    code.push_back(imadi(4, 1, D, kRZ));
    code.push_back(movi(5, 0x6ffe));
    if (mode == 2) code.push_back(imad_wide_g(4, 1, S, 6, 0));
  }
  code.push_back((mode >= 3 ? stg_e : st_e)(st_size, 4, 12, soff, st_guard));
  code.push_back((mode >= 3 ? ldg_e : ld_e)(ld_size, 12, 4, loff, ld_guard));
  code.push_back(imad_wide_i(8, 1, 32, 6));
  for (unsigned i = 0; i < 4; ++i) code.push_back(stg(8, 12 + i, kIn + 4 * i));
  for (unsigned j = 0; j < 4; ++j) {
    code.push_back(lds(10, 3, 128 * j));
    code.push_back(stg(8, 10, kIn + 16 + 4 * j));
  }
  code.push_back(exit_());
  return code;
}

struct Result {
  std::vector<uint32_t> out;
  std::vector<uint8_t> global;
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

struct Case {
  unsigned mode;
  uint32_t S, D;
  unsigned st_size, ld_size;
  uint32_t soff, loff;
  unsigned st_guard, ld_guard;
  uint32_t threads;
  std::string name() const {
    return "mode " + std::to_string(mode) + " S=" + std::to_string(S) + " D=" + std::to_string(D) + " st=" + std::to_string(st_size) + " ld=" + std::to_string(ld_size) + " soff=" +
           std::to_string(soff) + " loff=" + std::to_string(loff) + " stg=" + std::to_string(st_guard) + " ldg=" + std::to_string(ld_guard) + " threads=" + std::to_string(threads);
  }
};

unsigned bytes_of(unsigned size) { return size <= 1 ? 1 : size <= 3 ? 2 : size == 4 ? 4 : size == 5 ? 8 : 16; }

Result run(Env& e, const Case& k, const std::vector<uint8_t>& input, bool interp) {
  const auto code = kernel(k.mode, k.S, k.D, k.st_size, k.ld_size, k.soff, k.loff, k.st_guard, k.ld_guard);
  unsigned lds = 0, sts = 0;
  for (size_t i = 0; i < code.size(); ++i) {
    const sass::Instr ins = sass::decode(code[i].w, 16 * i, 86);
    lds += ins.op == (k.mode >= 3 ? sass::Op::LDG : sass::Op::LD), sts += ins.op == (k.mode >= 3 ? sass::Op::STG : sass::Op::ST);
  }
  if (lds != 1 || sts != (k.mode >= 3 ? 9u : 1u)) throw vtest::Failure("the kernel does not hold the one load and the one store under test");
  const std::string c = make_cubin(code);
  const auto m = sass::load(reinterpret_cast<const uint8_t*>(c.data()), c.size(), e.mem, e.prof);
  if (interp) ::setenv("VGPU_FASTPATH", "0", 1);
  const uint32_t out_bytes = k.threads * 32;
  const uint64_t base = e.mem.alloc(kIn + out_bytes);
  e.mem.write(base, input.data(), kIn);
  exec::LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {k.threads, 1, 1};
  cfg.shared_bytes = kShared;
  std::vector<uint8_t> arg(8);
  std::memcpy(arg.data(), &base, 8);
  Result r;
  r.stats = sass::launch(*m, "k", cfg, {arg}, e.mem, e.prof);
  ::unsetenv("VGPU_FASTPATH");
  r.out.resize(out_bytes / 4);
  e.mem.read(base + kIn, r.out.data(), out_bytes);
  r.global.resize(kIn);
  e.mem.read(base, r.global.data(), kIn);
  e.mem.free(base);
  return r;
}

// The rule: the lanes in order store, then load; a lane that does not run keeps the data it holds. Memory is a shared array (zeros) and the global buffer.
Result expected(const Case& k, const std::vector<uint8_t>& in) {
  std::vector<uint8_t> sh(kShared, 0), gl = in;
  const uint32_t half = k.mode == 6 ? 32 : 16;   // (where P0 turns on)
  const auto active = [&](unsigned guard, uint32_t t) { return guard == 7 || (guard == 0 ? t >= half : t < half); };
  const auto where = [&](uint32_t t, uint32_t off, std::vector<uint8_t>*& mem) -> uint64_t {
    const bool shared = k.mode == 0 || (k.mode == 2 && t < 16);
    mem = shared ? &sh : &gl;
    const uint32_t lane = k.mode == 4 ? 31 - t : t;   // (mode 4: the lanes go down memory)
    const uint64_t s = k.mode == 5 && t >= 16 ? 2 * uint64_t{k.S} : k.mode == 6 && t < 32 ? 0 : k.S;   // (mode 5: the stride doubles from lane 16; mode 6: the first warp's lanes all at the base)
    return shared ? uint64_t{k.D} * t + off : s * lane + off;
  };
  std::vector<std::array<uint32_t, 4>> reg(k.threads);
  for (uint32_t t = 0; t < k.threads; ++t)
    for (unsigned i = 0; i < 4; ++i) reg[t][i] = t * kK[i] + kC;
  for (uint32_t t = 0; t < k.threads; ++t) {
    if (!active(k.st_guard, t)) continue;
    std::vector<uint8_t>* mem;
    const uint64_t a = where(t, k.soff, mem);
    std::memcpy(&(*mem)[a], reg[t].data(), bytes_of(k.st_size));
  }
  for (uint32_t t = 0; t < k.threads; ++t) {
    if (!active(k.ld_guard, t)) continue;
    std::vector<uint8_t>* mem;
    const uint64_t a = where(t, k.loff, mem);
    uint32_t v[4] = {};
    const unsigned n = bytes_of(k.ld_size);
    std::memcpy(v, &(*mem)[a], n);
    if (k.ld_size == 1) v[0] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(v[0])));
    if (k.ld_size == 3) v[0] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(v[0])));
    for (unsigned i = 0; i < std::max(1u, n / 4); ++i) reg[t][i] = v[i];
  }
  Result r;
  r.out.assign(k.threads * 8, 0);
  for (uint32_t t = 0; t < k.threads; ++t) {
    for (unsigned i = 0; i < 4; ++i) r.out[t * 8 + i] = reg[t][i];
    for (unsigned j = 0; j < 4; ++j) std::memcpy(&r.out[t * 8 + 4 + j], &sh[4 * t + 128 * j], 4);
  }
  r.global = gl;
  return r;
}

void check(Env& e, const Case& k, const std::vector<uint8_t>& input) {
  const Result tier = run(e, k, input, false), ref = run(e, k, input, true), want = expected(k, input);
  for (size_t i = 0; i < want.out.size(); ++i) {
    if (ref.out[i] != want.out[i])
      throw vtest::Failure(k.name() + " (interpreter): thread " + std::to_string(i / 8) + " word " + std::to_string(i % 8) + " is " + std::to_string(ref.out[i]) + ", not " + std::to_string(want.out[i]));
    if (tier.out[i] != want.out[i])
      throw vtest::Failure(k.name() + " (tier): thread " + std::to_string(i / 8) + " word " + std::to_string(i % 8) + " is " + std::to_string(tier.out[i]) + ", not " + std::to_string(want.out[i]));
  }
  if (ref.global != want.global) throw vtest::Failure(k.name() + " (interpreter): the global buffer is not what the stores leave");
  if (tier.global != want.global) throw vtest::Failure(k.name() + " (tier): the global buffer is not what the stores leave");
  if (std::memcmp(&tier.stats, &ref.stats, exec::kCounterWords * sizeof(uint64_t)) != 0) {
    const uint64_t* a = reinterpret_cast<const uint64_t*>(&tier.stats);
    const uint64_t* b = reinterpret_cast<const uint64_t*>(&ref.stats);
    size_t i = 0;
    while (a[i] == b[i]) ++i;
    throw vtest::Failure(k.name() + ": counter word " + std::to_string(i) + " is " + std::to_string(a[i]) + " with the tier and " + std::to_string(b[i]) + " without");
  }
}

// The tier against the interpreter alone (words, bytes, counters): for the cases whose rule `expected` does not model (an access that reaches past the buffer
// under test into the output area).
void check_against_interpreter(Env& e, const Case& k, const std::vector<uint8_t>& input) {
  const Result tier = run(e, k, input, false), ref = run(e, k, input, true);
  if (tier.out != ref.out) throw vtest::Failure(k.name() + ": the registers differ from the interpreter's");
  if (tier.global != ref.global) throw vtest::Failure(k.name() + ": the global buffer differs from the interpreter's");
  if (std::memcmp(&tier.stats, &ref.stats, exec::kCounterWords * sizeof(uint64_t)) != 0) throw vtest::Failure(k.name() + ": the counters differ from the interpreter's");
}

std::vector<uint8_t> random_input() {
  std::mt19937 rng(11);
  std::vector<uint8_t> v(kIn);
  for (auto& b : v) b = static_cast<uint8_t>(rng());
  return v;
}

void sweep(std::initializer_list<unsigned> modes, std::initializer_list<std::array<uint32_t, 2>> sd, std::initializer_list<uint32_t> threads) {
  Scoped scoped;
  Env e;
  const std::vector<uint8_t> input = random_input();
  const unsigned sizes[][2] = {{0, 0}, {0, 1}, {2, 2}, {2, 3}, {4, 4}, {5, 5}, {6, 6}, {6, 4}, {4, 0}};
  for (unsigned mode : modes)
    for (const auto& z : sizes)
      for (unsigned sg : {7u, 0u})
        for (unsigned lg : {7u, 8u})
          for (const auto& p : sd)
            for (uint32_t t : threads) check(e, Case{mode, p[0], p[1], z[0], z[1], 0x20, z[0] == 6 && z[1] == 4 ? 0x24u : 0x20u, sg, lg, t}, input);
}

// As `sweep`, for blocks of several warps, against the interpreter alone: the rule written out (`expected`) takes the lanes of the whole block as one
// warp's, and warps of one block run one after the other, so what a warp reads of another's stores would differ. The stores go to a region of their own
// (at 4 KiB), the loads read the input (at 32 bytes), so what each lane reads does not depend on the order of the warps; the words the lanes store and
// the counters are still the interpreter's.
void sweep_against_interpreter(std::initializer_list<unsigned> modes, std::initializer_list<std::array<uint32_t, 2>> sd, std::initializer_list<uint32_t> threads) {
  Scoped scoped;
  Env e;
  const std::vector<uint8_t> input = random_input();
  const unsigned sizes[][2] = {{0, 0}, {0, 1}, {2, 2}, {2, 3}, {4, 4}, {5, 5}, {6, 6}, {6, 4}, {4, 0}};
  for (unsigned mode : modes)
    for (const auto& z : sizes)
      for (unsigned sg : {7u, 0u})
        for (unsigned lg : {7u, 8u})
          for (const auto& p : sd)
            for (uint32_t t : threads) check_against_interpreter(e, Case{mode, p[0], p[1], z[0], z[1], 0x1000, 0x20, sg, lg, t}, input);
}

// A launch's error text, or "" when it ran.
std::string fault_of(Env& e, const Case& k, const std::vector<uint8_t>& input, bool interp) {
  try {
    run(e, k, input, interp);
  } catch (const std::exception& x) {
    return x.what();
  }
  return "";
}

}  // namespace

VTEST(generic_access_past_the_shared_array_faults_as_the_interpreter_does) {
  Scoped scoped;
  Env e;
  const std::vector<uint8_t> input = random_input();
  for (unsigned size : {4u, 6u}) {
    const Case k{0, 16, 300, size, size, 0x20, 0x20, 7, 7, 32};   // 31 * 300 is past the 4096 bytes
    const std::string tier = fault_of(e, k, input, false), ref = fault_of(e, k, input, true);
    VCHECK(!ref.empty());
    VCHECK(tier == ref);
  }
}

VTEST(generic_accesses_all_in_the_shared_window) { sweep({0}, {{16, 16}, {4, 4}, {16, 0}, {32, 24}}, {32, 20}); }

VTEST(generic_accesses_all_in_global_memory) { sweep({1}, {{16, 16}, {4, 4}, {0, 16}, {32, 24}}, {32, 20}); }

VTEST(generic_accesses_in_both_windows_are_the_interpreters) { sweep({2}, {{16, 16}, {4, 4}, {32, 24}}, {32, 20}); }

// LDG and STG over the shapes of a warp's addresses: every lane the same (0), consecutive elements (the access size, 1 to 16 bytes), a stride a sector
// or more wide (32, 128, 200), strides between (12, 36, misaligned ones among them) -- for a full warp, for 20 lanes, and for the lanes a guard
// leaves (the low half, the high half); bytes, registers and every counter against the rule and against the interpreter.
VTEST(global_accesses_on_a_stride_or_not) {
  sweep({3}, {{0, 0}, {1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {12, 0}, {32, 0}, {36, 0}, {128, 0}, {200, 0}}, {32, 20});
}

// Warps whose lanes go DOWN memory (mode 4: lane l at base + (31 - l) * S, strides of 4 to 200 bytes): a strided load or store that took the stride
// as unsigned, or counted the lanes from the wrong end, would read and write the wrong words.
VTEST(global_accesses_on_a_descending_stride) {
  sweep({4}, {{4, 0}, {8, 0}, {16, 0}, {12, 0}, {32, 0}, {36, 0}, {200, 0}}, {32});
}

// A full warp whose lanes are in global memory but NOT on one stride (the stride doubles at lane 16): the strided load and store must leave it to the batch.
VTEST(global_accesses_off_a_stride_over_a_full_warp) {
  sweep({5}, {{4, 0}, {8, 0}, {16, 0}, {12, 0}, {36, 0}, {100, 0}}, {32, 20});
}

// A full warp's run (lane l at base + l * bytes) that ends at the end of the allocation, a byte short of it, and one byte and one element past it (mode 3
// at S = the access size, the offset chosen so that lane 31's last byte lands there). The tier takes the run from the registers' rows alone; where the
// memory manager declines it (past the end) the addresses are made and the access goes the way any full warp's does, to the interpreter's fault. Whatever
// the tier does must be the interpreter's: the same words, bytes and counters when the access ran, the same error text when it did not.
VTEST(full_warp_runs_at_the_end_of_the_allocation) {
  Scoped scoped;
  const std::vector<uint8_t> input = random_input();
  const unsigned sizes[][2] = {{0, 0}, {2, 2}, {2, 3}, {4, 4}, {5, 5}, {6, 6}, {4, 0}, {0, 4}};
  unsigned ran = 0, faulted = 0;
  for (const auto& z : sizes)
    for (int d : {-32, -8, -4, -1, 0, 1, 4, 32}) {
      const unsigned sb = bytes_of(z[0]), lb = bytes_of(z[1]);
      const unsigned elem = std::max(sb, lb);
      const uint32_t total = kIn + 32 * 32;
      const uint32_t off = total - 32 * elem + static_cast<uint32_t>(d);
      for (unsigned guard : {7u, 8u}) {
        const Case k{3, elem, 0, z[0], z[1], off, off, 7, guard, 32};
        // (a world of its own for each run: the error text names the address, and the buffer is allocated at the same place in both)
        Env et, er;
        const std::string tier = fault_of(et, k, input, false), ref = fault_of(er, k, input, true);
        if (tier != ref) throw vtest::Failure(k.name() + ": the tier's error is \"" + tier + "\", the interpreter's \"" + ref + "\"");
        if (ref.empty()) {
          Env e;
          check_against_interpreter(e, k, input);
          ++ran;
        } else {
          ++faulted;
        }
      }
    }
  VCHECK(ran > 10 && faulted > 10);
}

// A block of several warps over the same load and store: the shape of the lanes' addresses decides which test the tier makes first, from the warp before
// (a run, one address, neither), so the warps after the first take the path the first one chose; the strides here include the access size (a run) and 0
// (one address), with guards that leave half warps (the later warps of a block of 64 or 128 are full).
VTEST(global_accesses_over_several_warps) {
  sweep_against_interpreter({3}, {{0, 0}, {1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {12, 0}}, {64, 128});
}

// The first warp's lanes all at one address and the later warps' lanes on a stride (mode 6), and blocks of 64, 96 and 128 threads: the hint set by the first warp
// is wrong for the second, whose lanes are a run or a stride, and right again for the third.
VTEST(global_accesses_whose_shape_changes_from_warp_to_warp) {
  sweep_against_interpreter({6}, {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {12, 0}, {20, 0}}, {64, 96, 128});
}

// IMAD.WIDE.U32 with a carry-out predicate: the tier finds the carry only when an instruction names a predicate for it. Lane t adds 4 * t to
// 0xfffffffffffffff0, so the sum wraps past 2^64 (the carry is set) for t >= 4; a store guarded by that predicate is made by those lanes only,
// and the two words of the sum are stored by all. One warp, two warps and a part of one, the tier against the interpreter and the sums written out.
VTEST(imad_wide_carry_out_predicate_guards_a_store) {
  Scoped scoped;
  Env e;
  std::vector<Insn> code;
  code.push_back(s2r(1, 0x21));
  code.push_back(mov_param(6, 0x58));
  code.push_back(mov_param(7, 0x59));
  code.push_back(movi(2, 0xfffffff0u));
  code.push_back(movi(3, 0xffffffffu));
  code.push_back(movi(11, kC));
  code.push_back(imadi(12, 1, kK[0], 11));
  code.push_back(imad_wide_p0(10, 1, 4, 2));
  code.push_back(imad_wide_i(8, 1, 32, 6));
  code.push_back(stg_e(4, 8, 12, 0, 0));   // @P0 STG.E [R8.64], R12
  code.push_back(stg(8, 10, 4));
  code.push_back(stg(8, 11, 8));
  code.push_back(exit_());
  const std::string cubin = make_cubin(code);
  const auto m = sass::load(reinterpret_cast<const uint8_t*>(cubin.data()), cubin.size(), e.mem, e.prof);
  for (uint32_t threads : {32u, 20u, 64u, 96u}) {
    std::vector<uint32_t> results[2];
    exec::LaunchStats stats[2];
    for (int interp = 0; interp < 2; ++interp) {
      if (interp) ::setenv("VGPU_FASTPATH", "0", 1);
      const uint32_t bytes = threads * 32;
      const uint64_t base = e.mem.alloc(bytes);
      const std::vector<uint8_t> zeros(bytes, 0);
      e.mem.write(base, zeros.data(), bytes);
      exec::LaunchConfig cfg;
      cfg.grid = {1, 1, 1};
      cfg.block = {threads, 1, 1};
      std::vector<uint8_t> arg(8);
      std::memcpy(arg.data(), &base, 8);
      stats[interp] = sass::launch(*m, "k", cfg, {arg}, e.mem, e.prof);
      ::unsetenv("VGPU_FASTPATH");
      results[interp].resize(bytes / 4);
      e.mem.read(base, results[interp].data(), bytes);
      e.mem.free(base);
    }
    for (int interp = 0; interp < 2; ++interp)
      for (uint32_t t = 0; t < threads; ++t) {
        const uint64_t sum = 0xfffffffffffffff0ull + 4ull * t;   // (modulo 2^64)
        const uint32_t* w = &results[interp][t * 8];
        if (w[0] != (t >= 4 ? t * kK[0] + kC : 0u) || w[1] != static_cast<uint32_t>(sum) || w[2] != static_cast<uint32_t>(sum >> 32))
          throw vtest::Failure("threads=" + std::to_string(threads) + (interp ? " (interpreter)" : " (tier)") + ": thread " + std::to_string(t) + " stored " + std::to_string(w[0]) + ", " +
                               std::to_string(w[1]) + ", " + std::to_string(w[2]));
      }
    if (std::memcmp(&stats[0], &stats[1], exec::kCounterWords * sizeof(uint64_t)) != 0) throw vtest::Failure("the counters differ from the interpreter's");
  }
}

VTEST_MAIN
