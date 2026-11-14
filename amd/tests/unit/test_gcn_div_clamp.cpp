// Issue #613: v_div_fmas and v_div_fixup are VOP3 float results like any other, so the clamp bit holds them to [0, 1] and the output modifier
// scales them. clang folds fmin(fmax(a / b, 0), 1) into the clamp of the division's last instruction ("v_div_fixup_f64 ... clamp"); the
// executor -- the interpreter and the tier both -- wrote the quotient unclamped. The kernel is amd/tests/data/asm_div_clamp.s; the
// expected values come from the ISA (the clamp maps a NaN or anything <= 0 to +0 and anything above 1 to 1), not from the simulator.
// Run with the tier off and on.
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vgpu/amd_gcn.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

amd::CodeObject object(const char* target) {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/asm_div_clamp." + target + ".o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

template <typename T>
uint64_t put(MemoryManager& mem, const std::vector<T>& v) {
  const uint64_t p = mem.alloc(v.size() * sizeof(T));
  mem.write(p, v.data(), v.size() * sizeof(T));
  return p;
}

template <typename T>
T clamp01(T v) { return std::isnan(v) || v <= 0 ? T(0) : v > 1 ? T(1) : v; }

int run(const char* target, bool tier) {
  const amd::CodeObject o = object(target);
  MemoryManager mem(16ull << 20);
  std::vector<float> x32(64);
  std::vector<double> x64(64);
  for (int l = 0; l < 64; ++l) {   // normal numbers on both sides of 0 and of 1, and 1 itself
    x32[l] = l == 5 ? 1.0f : (l % 3 == 0 ? -1.0f : 1.0f) * (0.0625f + 0.1875f * static_cast<float>(l));
    x64[l] = l == 9 ? 1.0 : (l % 4 == 1 ? -1.0 : 1.0) * (0.03125 + 0.15625 * l + 1e-9);
  }
  const uint64_t p32 = put(mem, x32), p64 = put(mem, x64), pout = mem.alloc(2304);
  const amd::Kernel* k = amd::find_kernel(o, "div_clamp");
  if (!k) throw vtest::Failure("no kernel named div_clamp");
  std::vector<uint8_t> buf(k->kernarg_size, 0);
  const uint64_t args[3] = {p32, p64, pout};
  std::memcpy(buf.data(), args, sizeof args);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(buf.size());
  mem.write(d.kernarg, buf.data(), buf.size());
  d.group_size[0] = 64;
  amd::gcn::tier::set_enabled(tier);
  amd::execute(d, mem);
  amd::gcn::tier::set_enabled(true);
  std::vector<uint32_t> f(320);
  std::vector<uint64_t> g(128);
  mem.read(pout, f.data(), f.size() * 4);
  mem.read(pout + 1280, g.data(), g.size() * 8);
  int wrong = 0;
  const auto want = [&](const char* what, int l, uint64_t got, uint64_t expected) {
    if (got == expected) return;
    if (wrong++ < 8) std::fprintf(stderr, "  %s tier %d lane %d %s: %llx, expected %llx\n", target, tier, l, what, (unsigned long long)got, (unsigned long long)expected);
  };
  for (int l = 0; l < 64; ++l) {
    const float x = x32[l];
    want("v_div_fixup_f32 clamp", l, f[l], std::bit_cast<uint32_t>(clamp01(x)));
    want("v_div_fmas_f32 clamp", l, f[64 + l], std::bit_cast<uint32_t>(clamp01(x)));
    // The output modifier: ignored with IEEE mode or output denormals on (CDNA2 and CDNA3 guides, 6.5; an RX 6800 too, test_amd_gcn_mods),
    // which a code object's defaults turn on -- the same as for v_mul_f32's.
    want("v_div_fixup_f32 mul:2", l, f[128 + l], f[192 + l]);
    want("v_mul_f32 mul:2", l, f[192 + l], std::bit_cast<uint32_t>(x));
    want("v_div_fixup_f32", l, f[256 + l], std::bit_cast<uint32_t>(x));
    want("v_div_fixup_f64 clamp", l, g[l], std::bit_cast<uint64_t>(clamp01(x64[l])));
    want("v_div_fmas_f64 clamp", l, g[64 + l], std::bit_cast<uint64_t>(clamp01(x64[l])));
  }
  return wrong;
}

}  // namespace

VTEST(div_fmas_and_div_fixup_honour_the_clamp_bit_and_the_output_modifier) {
  for (const char* target : {"gfx90a", "gfx942"})
    for (bool tier : {false, true}) VCHECK_EQ(run(target, tier), 0);
}

VTEST_MAIN
