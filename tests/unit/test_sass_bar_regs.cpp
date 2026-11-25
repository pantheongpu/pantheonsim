// BAR with its barrier or its thread count in a register, as ptxas writes `bar.sync %a, %b` when the numbers are not known at compile time (cub's
// DeviceScan on sm_120, #758: "unknown opcode 0x31d"). Four forms, the same words on every architecture from sm_80 (compiled with distinct
// registers and read back with cuobjdump): both operands immediate (0xb1d), the barrier in a register and an immediate count (0x51d), an immediate
// barrier and the count in a register (0x91d), and both in registers (0x31d), which ptxas packs into the one register the word names: the barrier
// in its low four bits and the count from bit 16 up.
//
// The kernels are SASS written out word by word (nvdisasm's text for each word is checked, so a mistaken encoding fails here and not as a strange
// result). Four warps run one program: warps 0 and 1 wait on barrier 1 for 64 threads and warps 2 and 3 on barrier 2 for 64, each number read from
// a register. A count read as zero would make a barrier wait for all 128 threads, and a barrier number read as 0 would put all four warps on one
// barrier; either leaves warps waiting for ever, which the executor reports as a deadlock.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
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
  const char* text;
  Insn(uint64_t hi, uint64_t lo, const char* t) : text(t) { w.hi = hi, w.lo = lo; }   // (the corpora print the high half first)
};

template <class T>
void put(std::string& s, T v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof v);
}

const Insn kTid = {0x000e220000002100ull, 0x0000000000007919ull, "S2R R0, SR_TID.X"};
const Insn kPtrLo = {0x000fc60000000f00ull, 0x0000580000067a02ull, "MOV R6, c[0x0][0x160]"};
const Insn kPtrHi = {0x000fc60000000f00ull, 0x0000590000077a02ull, "MOV R7, c[0x0][0x164]"};
const Insn kAddr = {0x000fe200078e0006ull, 0x00000004000a7825ull, "IMAD.WIDE.U32 R10, R0, 0x4, R6"};
const Insn kSetR9One = {0x000fc60000000f00ull, 0x0000000100097802ull, "MOV R9, 0x1"};
const Insn kStoreR9 = {0x000fe2000c101904ull, 0x000000090a007986ull, "STG.E [R10.64], R9"};
const Insn kExit = {0x000fea0003800000ull, 0x000000000000794dull, "EXIT"};
const Insn kP0TidGe64 = {0x000fe40003f06070ull, 0x000000400000780cull, "ISETP.GE.U32.AND P0, PT, R0, 0x40, PT"};
const Insn kP0TidGe8 = {0x000fe40003f06070ull, 0x000000080000780cull, "ISETP.GE.U32.AND P0, PT, R0, 0x8, PT"};
const Insn kB2rR9 = {0x000e6200000e4000ull, 0x000000000009731cull, "B2R.RESULT R9, PT"};

// Texts live as long as the program.
const char* keep(std::string s) {
  static std::deque<std::string> all;
  all.push_back(std::move(s));
  return all.back().c_str();
}
std::string hex(uint32_t v) {
  char b[16];
  std::snprintf(b, sizeof b, "0x%x", v);
  return b;
}
// MOV Rn, imm
Insn mov(unsigned r, uint32_t imm) {
  return {0x000fc60000000f00ull, 0x0000000000007802ull | uint64_t{r} << 16 | uint64_t{imm} << 32, keep("MOV R" + std::to_string(r) + ", " + hex(imm))};
}

enum Guard { kAlways = 7, kP0 = 0, kNotP0 = 8 };
std::string at(Guard g) { return g == kAlways ? "" : g == kP0 ? "@P0 " : "@!P0 "; }
const uint64_t kSyncHi = 0x000fe20000010000ull;   // .DEFER_BLOCKING

// 0x31d: BAR.SYNC R, R (the barrier in the low four bits of R, the count from bit 16).
Insn bar_sync_rr(Guard g, unsigned r) {
  return {kSyncHi, 0x000000000000031dull | uint64_t{g} << 12 | uint64_t{r} << 32,
          keep(at(g) + "BAR.SYNC.DEFER_BLOCKING R" + std::to_string(r) + ", R" + std::to_string(r))};
}
// 0x51d: BAR.SYNC R, count
Insn bar_sync_ri(Guard g, unsigned r, uint32_t count) {
  return {kSyncHi, 0x000000000000051dull | uint64_t{g} << 12 | uint64_t{r} << 32 | uint64_t{count} << 42,
          keep(at(g) + "BAR.SYNC.DEFER_BLOCKING R" + std::to_string(r) + ", " + hex(count))};
}
// 0x91d: BAR.SYNC id, R
Insn bar_sync_ir(Guard g, unsigned id, unsigned r) {
  return {kSyncHi, 0x000000000000091dull | uint64_t{g} << 12 | uint64_t{r} << 32 | uint64_t{id} << 54,
          keep(at(g) + "BAR.SYNC.DEFER_BLOCKING " + hex(id) + ", R" + std::to_string(r))};
}

struct Kernel {
  std::string name;
  std::vector<Insn> code;
};

// A cubin for sm_86 (or `sm`) with the kernels given, each taking one 8-byte pointer.
std::string make_cubin(const std::vector<Kernel>& kernels, int sm = 86) {
  std::string names("\0.shstrtab\0", 11);
  std::vector<uint32_t> text_name, info_name;
  for (const Kernel& k : kernels) {
    text_name.push_back(static_cast<uint32_t>(names.size()));
    names += ".text." + k.name + '\0';
    info_name.push_back(static_cast<uint32_t>(names.size()));
    names += ".nv.info." + k.name + '\0';
  }
  std::string info;   // KPARAM_INFO: index, ordinal | offset << 16, size << 18
  put<uint8_t>(info, 4), put<uint8_t>(info, 0x17), put<uint16_t>(info, 12);
  put<uint32_t>(info, 0), put<uint32_t>(info, 0), put<uint32_t>(info, 8u << 18);
  std::string out(64, '\0');
  const uint64_t names_off = out.size();
  out += names;
  struct Place {
    uint64_t text_off, text_size, info_off;
  };
  std::vector<Place> places;
  for (const Kernel& k : kernels) {
    while (out.size() % 16) out.push_back('\0');
    Place p{out.size(), 16 * k.code.size(), 0};
    for (const Insn& i : k.code) put(out, i.w.lo), put(out, i.w.hi);
    p.info_off = out.size();
    out += info;
    places.push_back(p);
  }
  while (out.size() % 8) out.push_back('\0');
  const uint64_t shoff = out.size();
  const auto shdr = [&](uint32_t name, uint32_t type, uint64_t off, uint64_t size) {
    put<uint32_t>(out, name), put<uint32_t>(out, type);
    put<uint64_t>(out, 0), put<uint64_t>(out, 0), put<uint64_t>(out, off), put<uint64_t>(out, size);
    put<uint32_t>(out, 0), put<uint32_t>(out, 0), put<uint64_t>(out, 16), put<uint64_t>(out, 0);
  };
  shdr(0, 0, 0, 0);
  shdr(1, 3, names_off, names.size());
  for (size_t i = 0; i < kernels.size(); ++i) {
    shdr(text_name[i], 1, places[i].text_off, places[i].text_size);
    shdr(info_name[i], 0x70000000, places[i].info_off, info.size());
  }
  out[0] = 0x7f, out[1] = 'E', out[2] = 'L', out[3] = 'F', out[4] = 2, out[5] = 1, out[6] = 1, out[7] = 0x41, out[8] = 8;
  const uint16_t type = 2, machine = 190, shentsize = 64, shnum = static_cast<uint16_t>(2 + 2 * kernels.size()), shstrndx = 1;
  const uint32_t flags = 0x06000002u | (static_cast<uint32_t>(sm) << 8);
  std::memcpy(&out[16], &type, 2), std::memcpy(&out[18], &machine, 2);
  std::memcpy(&out[0x28], &shoff, 8), std::memcpy(&out[0x30], &flags, 4);
  std::memcpy(&out[0x3a], &shentsize, 2), std::memcpy(&out[0x3c], &shnum, 2), std::memcpy(&out[0x3e], &shstrndx, 2);
  return out;
}


struct Env {
  int sm;
  MemoryManager mem{1 << 24};
  DeviceProfile prof;
  explicit Env(int sm_) : sm(sm_), prof(load_gpu(sm_ == 80 ? "nvidia/a100" : sm_ == 90 ? "nvidia/h100" : sm_ == 100 ? "nvidia/b200" : "nvidia/rtx5090")) {}
  std::shared_ptr<sass::Module> load(const std::vector<Insn>& code) {
    for (size_t i = 0; i < code.size(); ++i) {
      const sass::Instr ins = sass::decode(code[i].w, 16 * i, sm);
      std::string want = code[i].text;
      if (sm >= 90 && want.rfind("STG.E [", 0) == 0) want.insert(6, "desc[UR4]");   // (from sm_90 a store names its cache-policy descriptor)
      if (sass::to_text(ins) != want)
        throw vtest::Failure("sm_" + std::to_string(sm) + " word " + std::to_string(i) + " is \"" + sass::to_text(ins) + "\", not \"" + want + "\"");
    }
    const std::string c = make_cubin({{"k", code}}, sm);
    return sass::load(reinterpret_cast<const uint8_t*>(c.data()), c.size(), mem, prof);
  }
  // One block of `threads`, each thread storing a word to out[threadIdx.x]; returns out.
  std::vector<uint32_t> run(sass::Module& m, uint32_t threads) {
    const uint64_t out = mem.alloc(threads * 4);
    const std::vector<uint32_t> poison(threads, 0x5A5A5A5Au);
    mem.write(out, poison.data(), threads * 4);
    exec::LaunchConfig cfg;
    cfg.grid = {1, 1, 1};
    cfg.block = {threads, 1, 1};
    std::vector<uint8_t> arg(8);
    std::memcpy(arg.data(), &out, 8);
    sass::launch(m, "k", cfg, {arg}, mem, prof);
    std::vector<uint32_t> got(threads);
    mem.read(out, got.data(), threads * 4);
    mem.free(out);
    return got;
  }
};

const std::vector<Insn> kHead = {kTid, kPtrLo, kPtrHi, kAddr};
std::vector<Insn> cat(std::vector<Insn> a, const std::vector<Insn>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

}  // namespace

// Each form, on every architecture whose ptxas writes it: warps 0 and 1 wait on barrier 1 for 64 threads, warps 2 and 3 on barrier 2.
VTEST(a_barrier_and_a_count_in_registers_are_read_from_the_register) {
  for (int sm : {80, 90, 100, 120}) {
    Env e(sm);
    // both in one register: the barrier in the low four bits, the count (64) from bit 16 up
    const auto packed = e.load(cat(kHead, {kSetR9One, kP0TidGe64, mov(12, 1 | 64 << 16), mov(13, 2 | 64 << 16), bar_sync_rr(kNotP0, 12),
                                           bar_sync_rr(kP0, 13), kStoreR9, kExit}));
    for (uint32_t v : e.run(*packed, 128)) VCHECK_EQ(v, 1u);
    // the barrier in a register, the count immediate
    const auto reg_imm = e.load(cat(kHead, {kSetR9One, kP0TidGe64, mov(12, 1), mov(13, 2), bar_sync_ri(kNotP0, 12, 64), bar_sync_ri(kP0, 13, 64), kStoreR9, kExit}));
    for (uint32_t v : e.run(*reg_imm, 128)) VCHECK_EQ(v, 1u);
    // the barrier immediate, the count in a register
    const auto imm_reg = e.load(cat(kHead, {kSetR9One, kP0TidGe64, mov(12, 64), bar_sync_ir(kNotP0, 1, 12), bar_sync_ir(kP0, 2, 12), kStoreR9, kExit}));
    for (uint32_t v : e.run(*imm_reg, 128)) VCHECK_EQ(v, 1u);
  }
}

// The barrier number is the register's low four bits only: 0x30 | 1 is barrier 1, and the count in the high half is not part of it.
VTEST(a_barrier_register_is_masked_to_its_low_four_bits) {
  Env e(120);
  const auto m = e.load(cat(kHead, {kSetR9One, kP0TidGe64, mov(12, 0x31 | 64 << 16), mov(13, 0xf2 | 64 << 16), bar_sync_rr(kNotP0, 12), bar_sync_rr(kP0, 13), kStoreR9, kExit}));
  for (uint32_t v : e.run(*m, 128)) VCHECK_EQ(v, 1u);
}

// A reduction with the barrier and the count in one register: BAR.RED.POPC over two warps, 56 of whose 64 threads hold the predicate.
VTEST(a_reduction_barrier_in_registers_counts_the_threads_with_the_predicate) {
  for (int sm : {90, 120}) {
    Env e(sm);
    const Insn red = {0x0001e40000014000ull, 0x000000000000031dull | uint64_t{kAlways} << 12 | uint64_t{12} << 32, "BAR.RED.POPC.DEFER_BLOCKING R12, R12, P0"};
    const auto m = e.load(cat(kHead, {kP0TidGe8, mov(12, 1 | 64 << 16), red, kB2rR9, kStoreR9, kExit}));
    for (uint32_t v : e.run(*m, 64)) VCHECK_EQ(v, 56u);
  }
}

// The words as ptxas wrote them (decoded to nvdisasm's text), including the arrive form and the reduction with a barrier in a register.
VTEST(the_register_forms_of_bar_decode_as_nvdisasm_prints_them) {
  const Insn words[] = {
      {0x0001e40000010000ull, 0x000000070000731dull, "BAR.SYNC.DEFER_BLOCKING R7, R7"},
      {0x0201f00000010000ull, 0x000200060000751dull, "BAR.SYNC.DEFER_BLOCKING R6, 0x80"},
      {0x0045e20000010000ull, 0x00c000090000791dull, "BAR.SYNC.DEFER_BLOCKING 0x3, R9"},
      {0x0001ec0000002000ull, 0x000000060000731dull, "BAR.ARV R6, R6"},
      {0x000fec0000002000ull, 0x000100000000751dull, "BAR.ARV R0, 0x40"},
      {0x0003ec0000002000ull, 0x008000050000791dull, "BAR.ARV 0x2, R5"},
      {0x0001e40000014000ull, 0x000000090000731dull, "BAR.RED.POPC.DEFER_BLOCKING R9, R9, P0"},
      {0x0003e40000014400ull, 0x000000070000731dull, "BAR.RED.AND.DEFER_BLOCKING R7, R7, P0"},
      {0x000fec0000014400ull, 0x000000060000751dull, "BAR.RED.AND.DEFER_BLOCKING R6, P0"},
  };
  for (int sm : {80, 90, 100, 120}) {
    for (const Insn& i : words) {
      const sass::Instr ins = sass::decode(i.w, 0, sm);
      VCHECK_EQ(sass::to_text(ins), std::string(i.text));
    }
  }
}

VTEST_MAIN
