// The SASS executor keeps a thread's Block (the warps' register files, the shared and local memory, the tensor memory) from block to
// block and from launch to launch, and sets it up again for each block (Runner::init_block). Every block must find what a new one
// does: a warp that is neither finished nor waiting, and registers that read zero.
//
// The kernels are SASS written out word by word (sm_86; nvdisasm's text for each word is checked, so a mistaken encoding fails here
// and not as a strange result): each stores one register to out[blockIdx.x] from every thread. One stores a register it never wrote,
// one a register it set to a number first; run one after the other on this thread (VGPU_THREADS=1), the second gets the first's block.
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
  const char* text;
  Insn(uint64_t hi, uint64_t lo, const char* t) : text(t) { w.hi = hi, w.lo = lo; }   // (the corpora print the high half first)
};

// S2R R0, SR_CTAID.X; MOV R6/R7, the out pointer (bank 0, 0x160); IMAD.WIDE.U32 R10, R0, 0x4, R6 (the pair R10:R11 is the address, R9 stays free).
const Insn kCtaid = {0x000ea20000002500ull, 0x0000000000007919ull, "S2R R0, SR_CTAID.X"};
const Insn kPtrLo = {0x000fc60000000f00ull, 0x0000580000067a02ull, "MOV R6, c[0x0][0x160]"};
const Insn kPtrHi = {0x000fc60000000f00ull, 0x0000590000077a02ull, "MOV R7, c[0x0][0x164]"};
const Insn kAddr = {0x000fe200078e0006ull, 0x00000004000a7825ull, "IMAD.WIDE.U32 R10, R0, 0x4, R6"};
const Insn kSetR9 = {0x000fc80000000f00ull, 0x1234567800097802ull, "MOV R9, 0x12345678"};
const Insn kStoreR9 = {0x000fe2000c101904ull, 0x000000090a007986ull, "STG.E [R10.64], R9"};
const Insn kStoreR0 = {0x000fe2000c101904ull, 0x000000000a007986ull, "STG.E [R10.64], R0"};
const Insn kExit = {0x000fea0003800000ull, 0x000000000000794dull, "EXIT"};

template <class T>
void put(std::string& s, T v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof v);
}

struct Kernel {
  std::string name;
  std::vector<Insn> code;
};

// A cubin for sm_86 with the kernels given, each taking one 8-byte pointer.
std::string make_cubin(const std::vector<Kernel>& kernels) {
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
  const uint32_t flags = 0x06000002u | (86u << 8);
  std::memcpy(&out[16], &type, 2), std::memcpy(&out[18], &machine, 2);
  std::memcpy(&out[0x28], &shoff, 8), std::memcpy(&out[0x30], &flags, 4);
  std::memcpy(&out[0x3a], &shentsize, 2), std::memcpy(&out[0x3c], &shnum, 2), std::memcpy(&out[0x3e], &shstrndx, 2);
  return out;
}

struct Env {
  MemoryManager mem{1 << 24};
  DeviceProfile prof = load_gpu("nvidia/a10");   // sm_86
  std::shared_ptr<sass::Module> load(const std::vector<Kernel>& kernels) {
    for (const Kernel& k : kernels)
      for (size_t i = 0; i < k.code.size(); ++i) {
        const sass::Instr ins = sass::decode(k.code[i].w, 16 * i, 86);
        if (sass::to_text(ins) != k.code[i].text)
          throw vtest::Failure(k.name + " word " + std::to_string(i) + " is \"" + sass::to_text(ins) + "\", not \"" + k.code[i].text + "\"");
      }
    const std::string c = make_cubin(kernels);
    return sass::load(reinterpret_cast<const uint8_t*>(c.data()), c.size(), mem, prof);
  }
  std::shared_ptr<sass::Module> load(const std::vector<Insn>& code) { return load(std::vector<Kernel>{{"k", code}}); }
  // Runs the module's kernel over `blocks` blocks of `threads`, each storing to out[block]; returns out.
  std::vector<uint32_t> run(sass::Module& m, uint32_t blocks, uint32_t threads, const char* kernel = "k") {
    const uint64_t out = mem.alloc(blocks * 4);
    const std::vector<uint32_t> poison(blocks, 0x5A5A5A5Au);
    mem.write(out, poison.data(), blocks * 4);
    exec::LaunchConfig cfg;
    cfg.grid = {blocks, 1, 1};
    cfg.block = {threads, 1, 1};
    std::vector<uint8_t> arg(8);
    std::memcpy(arg.data(), &out, 8);
    sass::launch(m, kernel, cfg, {arg}, mem, prof);
    std::vector<uint32_t> got(blocks);
    mem.read(out, got.data(), blocks * 4);
    return got;
  }
};

struct OneThread {
  OneThread() { ::setenv("VGPU_THREADS", "1", 1); }
  ~OneThread() { ::unsetenv("VGPU_THREADS"); }
};

}  // namespace

VTEST(every_block_of_a_grid_runs_on_a_warp_that_has_not_finished) {
  OneThread one;
  Env e;
  const auto m = e.load({kCtaid, kPtrLo, kPtrHi, kAddr, kStoreR0, kExit});
  for (uint32_t threads : {1u, 32u, 64u, 100u}) {
    const std::vector<uint32_t> got = e.run(*m, 5, threads);
    for (uint32_t b = 0; b < 5; ++b) VCHECK_EQ(got[b], b);
  }
}

VTEST(a_register_a_kernel_sets_reads_back_as_set) {
  OneThread one;
  Env e;
  const auto m = e.load({kCtaid, kPtrLo, kPtrHi, kAddr, kSetR9, kStoreR9, kExit});
  for (uint32_t v : e.run(*m, 3, 64)) VCHECK_EQ(v, 0x12345678u);
}

VTEST(registers_start_at_zero_in_the_next_block_and_the_next_launch) {
  OneThread one;
  Env e;
  const auto sets = e.load({kCtaid, kPtrLo, kPtrHi, kAddr, kSetR9, kStoreR9, kExit});
  const auto reads = e.load({kCtaid, kPtrLo, kPtrHi, kAddr, kStoreR9, kExit});
  for (int round = 0; round < 3; ++round) {
    for (uint32_t v : e.run(*sets, 4, 128)) VCHECK_EQ(v, 0x12345678u);
    for (uint32_t v : e.run(*reads, 4, 128)) VCHECK_EQ(v, 0u);
    for (uint32_t v : e.run(*reads, 4, 32)) VCHECK_EQ(v, 0u);
  }
}

VTEST(each_kernel_of_a_module_starts_at_its_own_code_and_keeps_its_constant_banks) {
  OneThread one;
  Env e;
  // j ends at once; k is the second kernel of the module, so its code does not start at the section's base.
  const auto m = e.load({{"j", {kExit}}, {"k", {kCtaid, kPtrLo, kPtrHi, kAddr, kStoreR0, kExit}}});
  for (int round = 0; round < 2; ++round) {
    for (uint32_t v : e.run(*m, 4, 32, "j")) VCHECK_EQ(v, 0x5A5A5A5Au);
    const std::vector<uint32_t> got = e.run(*m, 4, 32, "k");
    for (uint32_t b = 0; b < 4; ++b) VCHECK_EQ(got[b], b);
  }
}

VTEST_MAIN
