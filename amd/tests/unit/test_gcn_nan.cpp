// Which NaN a gfx1030 float instruction gives, against an RX 6800 and an RX 6700 XT (which agree): every dword the cards wrote for
// amd/tests/data/asm_nan.s (half and single-precision arithmetic, multiply-adds, min/max/med3, div_fixup, conversions and
// transcendentals over pairs of eleven special values, with IEEE mode on and off) is compared, bit for bit, with the executor's.
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

constexpr int kItems = 128, kOps = 65;
constexpr uint16_t kH[11] = {0x7e00, 0xfe00, 0x7d55, 0x7c01, 0xfc01, 0x7c00, 0xfc00, 0x0000, 0x8000, 0x3c00, 0xbc00};
constexpr uint32_t kF[11] = {0x7fc00000u, 0xffc00000u, 0x7fd55555u, 0x7f800001u, 0xff800001u, 0x7f800000u,
                             0xff800000u, 0u,          0x80000000u, 0x3f800000u, 0xbf800000u};

std::string data(const std::string& name) { return std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/" + name; }

// The cards' dwords for a kernel: kOps lines of "kernel slot" and kItems words.
std::vector<uint32_t> card(const std::string& kernel) {
  std::ifstream in(data("asm_nan.gfx1030.card.txt"));
  std::vector<uint32_t> v;
  std::string k, w;
  for (int op; in >> k >> op;)
    for (int t = 0; t < kItems && in >> w; ++t)
      if (k == kernel) v.push_back(static_cast<uint32_t>(std::stoul(w, nullptr, 16)));
  if (v.size() != size_t{kOps} * kItems) throw vtest::Failure("asm_nan.gfx1030.card.txt has no full " + kernel);
  return v;
}

int run(const char* kernel, bool tier) {
  const std::string path = data("asm_nan.gfx1030.o");
  std::ifstream f(path, std::ios::binary);
  if (!f) throw vtest::Failure("no code object at " + path);
  const amd::CodeObject o = amd::load_code_object(std::string((std::istreambuf_iterator<char>(f)), {}), path);
  const std::vector<uint32_t> want = card(kernel);
  MemoryManager mem(4ull << 20);
  std::vector<uint32_t> words(kItems * 8, 0);
  for (int t = 0; t < kItems; ++t) {
    const int i = t % 11, j = (t / 11) % 11, k = (t * 7 + 3) % 11;
    const uint32_t w[6] = {kH[i], kH[j], kH[k], kF[i], kF[j], kF[k]};
    std::memcpy(&words[t * 8], w, sizeof w);
  }
  const uint64_t pin = mem.alloc(words.size() * 4), pout = mem.alloc(kOps * kItems * 4);
  mem.write(pin, words.data(), words.size() * 4);
  const amd::Kernel* k = amd::find_kernel(o, kernel);
  if (!k) throw vtest::Failure(std::string("no kernel named ") + kernel);
  std::vector<uint8_t> buf(k->kernarg_size, 0);
  const uint64_t args[2] = {pin, pout};
  std::memcpy(buf.data(), args, sizeof args);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(buf.size());
  mem.write(d.kernarg, buf.data(), buf.size());
  d.group_size[0] = 64;
  d.groups[0] = kItems / 64;
  d.wave_size = 32;
  amd::gcn::tier::set_enabled(tier);
  amd::execute(d, mem);
  amd::gcn::tier::set_enabled(true);
  std::vector<uint32_t> got(kOps * kItems);
  mem.read(pout, got.data(), got.size() * 4);
  int wrong = 0;
  for (int op = 0; op < kOps; ++op) {
    int bad = 0;
    for (int t = 0; t < kItems; ++t) {
      const uint32_t g = got[op * kItems + t], c = want[op * kItems + t];
      if (g == c) continue;
      if (bad++ < 3) {
        const int i = t % 11, j = (t / 11) % 11, kk = (t * 7 + 3) % 11;
        std::fprintf(stderr, "  %s tier %d slot %d t %d (a %04x/%08x b %04x/%08x c %04x/%08x): %08x, card %08x\n", kernel, tier, op, t,
                     kH[i], kF[i], kH[j], kF[j], kH[kk], kF[kk], g, c);
      }
    }
    if (bad) std::fprintf(stderr, "  %s tier %d slot %d: %d of %d differ\n", kernel, tier, op, bad, kItems);
    wrong += bad;
  }
  return wrong;
}

int run_both(const char* kernel) {
  int wrong = 0;
  for (bool tier : {false, true}) {
    try {
      wrong += run(kernel, tier);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "  %s tier %d: %s\n", kernel, tier, e.what());
      ++wrong;
    }
  }
  return wrong;
}

}  // namespace

VTEST(gfx1030_nans_with_ieee_mode_match_the_cards) { VCHECK_EQ(run_both("nan_k"), 0); }

VTEST(gfx1030_nans_without_ieee_mode_match_the_cards) { VCHECK_EQ(run_both("nan_n"), 0); }

VTEST_MAIN
