// What a 16-bit vector result leaves in the high half of its destination register. gfx9 zeroes it for most instructions and keeps it
// for the VOP3 multiply-adds that have op_sel (CDNA2 and CDNA3 guides, V_MAD_F16, V_MAD_U16, V_MAD_I16, V_FMA_F16, V_DIV_FIXUP_F16: "If
// op_sel[3] is 0 Result is written to 16 LSBs of destination VGPR and hi 16 bits are preserved"; the _LEGACY forms: "hi 16 bits are written
// as 0"). gfx10 keeps it for every 16-bit VOP1, VOP2 and VOP3 result, in every encoding (e32, e64, DPP, DPP8), but v_sat_pk_u8_i16, whose
// result is "{16'b0, sat8(...), sat8(...)}": an RX 6800 and an RX 6700 XT ran the gfx1030 object (ROCm 7.2) and gave exactly the table
// below. (LLVM's GCNSubtarget::zeroesHigh16BitsOfDest says the same: "On gfx10, all 16-bit instructions preserve the high bits.")
//
// The kernel is amd/tests/data/asm_hi16.s; SDWA (dst_sel and dst_unused say what happens) and the packed instructions are not in it.
// gfx9: only the bits a rule names are compared. gfx1030: every dword is also compared with what the cards wrote
// (amd/tests/data/asm_hi16.gfx1030.card.txt), NaNs and all (test_amd_gcn_nan has the NaN rules).
// Run with the tier off and on.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
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

constexpr int kItems = 64, kOps = 75;
constexpr uint16_t kH[16] = {0x3c00, 0xbc00, 0x4000, 0x3800, 0x0001, 0x7c00, 0x7e00, 0x8000,
                             0x0000, 0x4248, 0xc500, 0x5640, 0x0005, 0x7bff, 0x3555, 0x00ff};
uint32_t word(int i) { return uint32_t{kH[(i + 5) & 15]} << 16 | kH[i & 15]; }
uint32_t old_of(int t) { return uint32_t(0x5a00 | t) << 16 | kH[(t * 3 + 2) & 15]; }

// Per slot of asm_hi16.s (the instruction list is there): P the high half kept, Z zeroed, H op_sel's high result (the low half kept),
// N no instruction (the dword kept), F a 32-bit result (not compared), X not established (not compared).
//                        0         1         2         3         4         5         6         7
//                        012345678901234567890123456789012345678901234567890123456789012345678901234
constexpr char kGfx9[] = "ZZZXXZZZZZZZZZZZZZZZXXZZZZZZZXXXZZZZZZZXXZZZZZZZZPPPPXXXXXXXXXHHHXXFFPZZZXX";
constexpr char kGfx10[] = "PPPPPPPPPPPPPPPPPPPZPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPHHHPPFFPNNNNN";
static_assert(sizeof kGfx9 == kOps + 1 && sizeof kGfx10 == kOps + 1);

amd::CodeObject object(const char* target) {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/asm_hi16." + target + ".o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

// The cards' dwords, slot by slot (empty where there is no such file).
std::vector<uint32_t> card(const char* target) {
  std::ifstream in(std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/asm_hi16." + target + ".card.txt");
  std::vector<uint32_t> v;
  for (int op, t = 0; in >> op; t = 0)
    for (std::string w; t < kItems && in >> w; ++t) v.push_back(static_cast<uint32_t>(std::stoul(w, nullptr, 16)));
  return v;
}

int run(const char* target, const char* rules, bool wave32, bool tier) {
  const std::vector<uint32_t> want = card(target);
  if (!want.empty() && want.size() != size_t{kOps} * kItems) throw vtest::Failure(std::string("a short card file for ") + target);
  const amd::CodeObject o = object(target);
  MemoryManager mem(4ull << 20);
  std::vector<uint32_t> words(kItems * 4);
  for (int t = 0; t < kItems; ++t) {
    words[4 * t] = word(t);
    words[4 * t + 1] = word(t * 5 + 3);
    words[4 * t + 2] = word(t * 7 + 1);
    words[4 * t + 3] = old_of(t);
  }
  const uint64_t pin = mem.alloc(words.size() * 4), pout = mem.alloc(kOps * kItems * 4);
  mem.write(pin, words.data(), words.size() * 4);
  const amd::Kernel* k = amd::find_kernel(o, "hi16");
  if (!k) throw vtest::Failure("no kernel named hi16");
  std::vector<uint8_t> buf(k->kernarg_size, 0);
  const uint64_t args[2] = {pin, pout};
  std::memcpy(buf.data(), args, sizeof args);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(buf.size());
  mem.write(d.kernarg, buf.data(), buf.size());
  d.group_size[0] = kItems;
  d.groups[0] = 1;
  if (wave32) d.wave_size = 32;
  amd::gcn::tier::set_enabled(tier);
  amd::execute(d, mem);
  amd::gcn::tier::set_enabled(true);
  std::vector<uint32_t> got(kOps * kItems);
  mem.read(pout, got.data(), got.size() * 4);
  int wrong = 0;
  for (int op = 0; op < kOps; ++op)
    for (int t = 0, shown = 0; t < kItems; ++t) {
      const uint32_t g = got[op * kItems + t], old = old_of(t);
      bool ok = true;
      switch (rules[op]) {
        case 'P': ok = g >> 16 == old >> 16; break;
        case 'Z': ok = g >> 16 == 0; break;
        case 'H': ok = (g & 0xFFFFu) == (old & 0xFFFFu); break;
        case 'N': ok = g == old; break;
        default: break;
      }
      if (ok && !want.empty()) ok = g == want[op * kItems + t];
      if (ok) continue;
      ++wrong;
      if (shown++ < 2)
        std::fprintf(stderr, "  %s tier %d slot %d (%c) t %d: %08x, old %08x, card %08x\n", target, tier, op, rules[op], t, g, old,
                     want.empty() ? 0u : want[op * kItems + t]);
    }
  return wrong;
}

int run_both(const char* target, const char* rules, bool wave32) {
  int wrong = 0;
  for (bool tier : {false, true}) {
    try {
      wrong += run(target, rules, wave32, tier);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "  %s tier %d: %s\n", target, tier, e.what());
      ++wrong;
    }
  }
  return wrong;
}

}  // namespace

VTEST(gfx9_16_bit_results_zero_the_high_half_but_the_vop3_multiply_adds) {
  VCHECK_EQ(run_both("gfx90a", kGfx9, false) + run_both("gfx942", kGfx9, false), 0);
}

VTEST(gfx10_16_bit_results_keep_the_high_half) {
  VCHECK_EQ(run_both("gfx1030", kGfx10, true), 0);
}

VTEST_MAIN
