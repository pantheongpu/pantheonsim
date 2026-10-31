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

constexpr uint32_t kK[4] = {0x9e3779b1u, 0x85ebca6bu, 0xc2b2ae35u, 0x27d4eb2fu}, kC = 0x01234567u;

// Registers: R1 tid, R3 tid * 4, R4:R5 the address, R6:R7 the buffer, R8:R9 the output, R10 a word read back, R11 a constant, R12..R15 the data.
// mode 0: every lane's address is in the shared window (tid * D), 1: global (base + tid * S), 2: shared for tid < 16 and global for the rest.
std::vector<Insn> kernel(unsigned mode, uint32_t S, uint32_t D, unsigned st_size, unsigned ld_size, uint32_t soff, uint32_t loff, unsigned st_guard, unsigned ld_guard) {
  std::vector<Insn> code;
  code.push_back(s2r(1, 0x21));
  code.push_back(mov_param(6, 0x58));
  code.push_back(mov_param(7, 0x59));
  code.push_back(isetp_ge(1, 16));              // P0 = (tid >= 16)
  code.push_back(movi(11, kC));
  for (unsigned i = 0; i < 4; ++i) code.push_back(imadi(12 + i, 1, kK[i], 11));
  code.push_back(imadi(3, 1, 4, kRZ));
  if (mode == 1) {
    code.push_back(imad_wide_i(4, 1, S, 6));
  } else {
    code.push_back(imadi(4, 1, D, kRZ));
    code.push_back(movi(5, 0x6ffe));
    if (mode == 2) code.push_back(imad_wide_g(4, 1, S, 6, 0));
  }
  code.push_back(st_e(st_size, 4, 12, soff, st_guard));
  code.push_back(ld_e(ld_size, 12, 4, loff, ld_guard));
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
    lds += ins.op == sass::Op::LD, sts += ins.op == sass::Op::ST;
  }
  if (lds != 1 || sts != 1) throw vtest::Failure("the kernel does not hold one LD and one ST");
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
  const auto active = [&](unsigned guard, uint32_t t) { return guard == 7 || (guard == 0 ? t >= 16 : t < 16); };
  const auto where = [&](uint32_t t, uint32_t off, std::vector<uint8_t>*& mem) -> uint64_t {
    const bool shared = k.mode == 0 || (k.mode == 2 && t < 16);
    mem = shared ? &sh : &gl;
    return shared ? uint64_t{k.D} * t + off : uint64_t{k.S} * t + off;
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

VTEST_MAIN
