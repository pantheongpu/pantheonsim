// Issue #621: a load into accumulation registers that the wave had not used yet, made for the whole wave at once (Machine::wave_scalars and
// the wide forms beside it), wrote all but the last register into a copy of the register file that the growth of the file had just freed.
// The kernels are amd/tests/data/asm_issue621.s; each loads into AGPRs in a fresh wave and copies them out with v_accvgpr_read_b32. The
// expected values are written from the ISA guide (the loaded dwords land in the named registers in order), not taken from the simulator.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vgpu/error.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

amd::CodeObject object(const char* target) {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/asm_issue621." + target + ".o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

constexpr uint32_t kSentinel = 0xDEADBEEFu;

// in: 64 * 8 distinct words; out: 64 * 8 words of the sentinel. Runs `name` as one wave of 64 and returns out.
std::vector<uint32_t> run(const amd::CodeObject& o, const char* name, const std::vector<uint32_t>& in) {
  MemoryManager mem(16ull << 20);
  const uint64_t pin = mem.alloc(in.size() * 4);
  mem.write(pin, in.data(), in.size() * 4);
  const std::vector<uint32_t> blank(64 * 8, kSentinel);
  const uint64_t pout = mem.alloc(blank.size() * 4);
  mem.write(pout, blank.data(), blank.size() * 4);
  const amd::Kernel* k = amd::find_kernel(o, name);
  if (!k) throw vtest::Failure(std::string("no kernel named ") + name);
  std::vector<uint8_t> args(k->kernarg_size, 0);
  std::memcpy(&args[0], &pin, 8);
  std::memcpy(&args[8], &pout, 8);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(args.size());
  mem.write(d.kernarg, args.data(), args.size());
  d.group_size[0] = 64;
  amd::execute(d, mem);
  std::vector<uint32_t> out(blank.size());
  mem.read(pout, out.data(), out.size() * 4);
  return out;
}

std::vector<uint32_t> input() {
  std::vector<uint32_t> in(64 * 8);
  for (uint32_t i = 0; i < in.size(); ++i) in[i] = 0x9E3779B9u * (i + 1) ^ 0x00C0FFEEu;
  return in;
}

// out[w] against want(w) for every word; prints the first few that differ.
template <typename Want>
int compare(const char* target, const char* name, const std::vector<uint32_t>& out, Want want) {
  int wrong = 0;
  for (uint32_t w = 0; w < out.size(); ++w) {
    const uint32_t expected = want(w);
    if (out[w] == expected) continue;
    if (wrong++ < 4) std::fprintf(stderr, "%s %s: out[%u] = %08x, want %08x\n", target, name, w, out[w], expected);
  }
  return wrong;
}

}  // namespace

// The exadis shape: every lane loads the same pair (a pointer in a structure) into a[38:39].
VTEST(a_wave_uniform_load_into_new_accumulation_registers_fills_both) {
  const std::vector<uint32_t> in = input();
  for (const char* target : {"gfx90a", "gfx942"}) {
    const std::vector<uint32_t> out = run(object(target), "agpr_uniform", in);
    // lane L's pair at word 4 L: in[50], in[51]; the rest untouched
    VCHECK_EQ(compare(target, "agpr_uniform", out, [&](uint32_t w) { return w >= 256 || w % 4 > 1 ? kSentinel : in[50 + w % 4]; }), 0);
  }
}

// Every lane its own pair, 16 bytes apart (not a run of consecutive lanes).
VTEST(a_gathered_load_into_new_accumulation_registers_fills_both) {
  const std::vector<uint32_t> in = input();
  for (const char* target : {"gfx90a", "gfx942"}) {
    const std::vector<uint32_t> out = run(object(target), "agpr_gather", in);
    VCHECK_EQ(compare(target, "agpr_gather", out, [&](uint32_t w) { return w >= 256 || w % 4 > 1 ? kSentinel : in[w]; }), 0);
  }
}

// The same with lanes 32 to 63 off (the lanes are gathered from EXEC rather than taken as a whole row).
VTEST(a_gathered_load_of_some_lanes_into_new_accumulation_registers_fills_both) {
  const std::vector<uint32_t> in = input();
  for (const char* target : {"gfx90a", "gfx942"}) {
    const std::vector<uint32_t> out = run(object(target), "agpr_half", in);
    VCHECK_EQ(compare(target, "agpr_half", out, [&](uint32_t w) { return w >= 128 || w % 4 > 1 ? kSentinel : in[w]; }), 0);
  }
}

// Four dwords a lane into a[36:39], 32 bytes apart.
VTEST(a_gathered_load_of_four_into_new_accumulation_registers_fills_all_four) {
  const std::vector<uint32_t> in = input();
  for (const char* target : {"gfx90a", "gfx942"}) {
    const std::vector<uint32_t> out = run(object(target), "agpr_gather4", in);
    VCHECK_EQ(compare(target, "agpr_gather4", out, [&](uint32_t w) { return w % 8 > 3 ? kSentinel : in[w]; }), 0);
  }
}

VTEST_MAIN
