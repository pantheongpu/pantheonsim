// gfx1030's NaNs in the forms test_amd_gcn_nan leaves out, against an RX 6800 and an RX 6700 XT (which agree): every dword pair the
// cards wrote for amd/tests/data/asm_nan2.s (SDWA, DPP and DPP8 forms, v_fmamk/v_fmaak, the legacy fma, packed halves, doubles, the
// dot products' clamp and denormals, min/max/med3 of opposite zeros and denormals; IEEE mode on, off, and on with denormals flushed) is
// compared, bit for bit, with the executor's, the high half of each 16-bit result included. Run with the tier off and on.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vgpu/amd_gcn.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

constexpr int kItems = 192, kOps = 117, kN = 12, kSqrtF64 = 83;
constexpr uint16_t kH[kN] = {0x7e00, 0xfe00, 0x7f55, 0x7d55, 0x7c01, 0xfc01, 0x7c00, 0xfc00, 0x0000, 0x8000, 0x3c00, 0xbc00};
constexpr uint32_t kF[kN] = {0x7fc00000u, 0xffc00000u, 0x7fd55555u, 0x7f955555u, 0x7f800001u, 0xff800001u,
                             0x7f800000u, 0xff800000u, 0u,          0x80000000u, 0x3f800000u, 0xbf800000u};
constexpr uint64_t kD[kN] = {0x7ff8000000000000ull, 0xfff8000000000000ull, 0x7ffd555555555555ull, 0x7ff5555555555555ull,
                             0x7ff0000000000001ull, 0xfff0000000000001ull, 0x7ff0000000000000ull, 0xfff0000000000000ull,
                             0ull,                  0x8000000000000000ull, 0x3ff0000000000000ull, 0xbff0000000000000ull};
constexpr uint32_t kDot[16] = {0u,          0x80000000u, 0x00000001u, 0x807fffffu, 0x00010001u, 0x83ff03ffu, 0x3c003c00u, 0xbc00bc00u,
                               0x7bff7bffu, 0x7c00fc00u, 0x7e00fe00u, 0x7fffffffu, 0xffffffffu, 0x7f7f7f7fu, 0x80808080u, 0x7f7fffffu};

std::string data(const std::string& name) { return std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/" + name; }

// The work-items' sixteen dwords (the scheme asm_nan2.s describes).
std::vector<uint32_t> inputs() {
  std::vector<uint32_t> w(kItems * 16, 0);
  for (int t = 0; t < kItems; ++t) {
    const int i = t % kN, j = (t / kN) % kN, k = (t * 7 + 3) % kN;
    const int p = t % 16, q = (t * 3 + t / 16) % 16, r = (t * 5 + 7) % 16;
    uint32_t* x = &w[t * 16];
    x[0] = kH[i], x[1] = kH[j], x[2] = kH[k];
    x[3] = kF[i], x[4] = kF[j], x[5] = kF[k];
    std::memcpy(x + 6, &kD[i], 8), std::memcpy(x + 8, &kD[j], 8), std::memcpy(x + 10, &kD[k], 8);
    x[12] = kDot[p], x[13] = kDot[q], x[14] = kDot[r];
  }
  return w;
}

// The cards' results for a kernel: kOps lines of "kernel slot" and kItems words, v11 then v10 in 16 hex digits, or v10 alone in 8 where
// v11 kept its 0xa5a5a5a5.
std::vector<uint64_t> card(const std::string& kernel) {
  std::ifstream in(data("asm_nan2.gfx1030.card.txt"));
  std::vector<uint64_t> v;
  std::string k, w;
  for (int op; in >> k >> op;)
    for (int t = 0; t < kItems && in >> w; ++t)
      if (k == kernel) v.push_back(std::stoull(w, nullptr, 16) | (w.size() == 8 ? 0xa5a5a5a5ull << 32 : 0));
  if (v.size() != size_t{kOps} * kItems) throw vtest::Failure("asm_nan2.gfx1030.card.txt has no full " + kernel);
  return v;
}

// The RX 6700 XT (gfx1031) ran the same programs, assembled for gfx1031: amd/tests/data/asm_nan.gfx1031.card.digests.txt has the FNV-1a 64 of
// its results of each kernel, which must be the RX 6800's (the card file above): one rule for gfx1030 and gfx1031 (the executor maps both to one target).
std::map<std::string, uint64_t> digests_1031() {
  std::ifstream in(data("asm_nan.gfx1031.card.digests.txt"));
  std::map<std::string, uint64_t> m;
  for (std::string line; std::getline(in, line);) {
    char k[16];
    unsigned long long h;
    if (line.empty() || line[0] == '#' || std::sscanf(line.c_str(), "%15s %llx", k, &h) != 2) continue;
    m[k] = h;
  }
  return m;
}

template <class W>
uint64_t fnv(const std::vector<W>& words) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (W w : words)
    for (unsigned b = 0; b < sizeof(W); ++b) h = (h ^ (static_cast<uint64_t>(w) >> (8 * b) & 0xff)) * 0x100000001b3ull;
  return h;
}

int run(const char* kernel, bool tier) {
  const std::string path = data("asm_nan2.gfx1030.o");
  std::ifstream f(path, std::ios::binary);
  if (!f) throw vtest::Failure("no code object at " + path);
  const amd::CodeObject o = amd::load_code_object(std::string((std::istreambuf_iterator<char>(f)), {}), path);
  const std::vector<uint64_t> want = card(kernel);
  MemoryManager mem(4ull << 20);
  const std::vector<uint32_t> words = inputs();
  const uint64_t pin = mem.alloc(words.size() * 4), pout = mem.alloc(size_t{kOps} * kItems * 8);
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
  std::vector<uint64_t> got(size_t{kOps} * kItems);
  mem.read(pout, got.data(), got.size() * 8);
  const std::vector<uint32_t> in = inputs();
  int wrong = 0;
  for (int op = 0; op < kOps; ++op) {
    int bad = 0;
    for (int t = 0; t < kItems; ++t) {
      const uint64_t g = got[op * kItems + t], c = want[op * kItems + t];
      if (g == c) continue;
      // v_sqrt_f64 of a positive number: the card's is an approximation (sqrt(1) is 0x3ff0000004000000), which this does not model; its
      // NaNs, zeros and infinities are compared.
      if (op == kSqrtF64 && (in[t * 16 + 7] & 0x7ff00000u) != 0x7ff00000u && !(in[t * 16 + 7] >> 31) && (in[t * 16 + 7] | in[t * 16 + 6])) continue;
      if (bad++ < 4) {
        const uint32_t* x = &in[t * 16];
        std::fprintf(stderr, "  %s tier %d slot %d t %d (h %04x %04x %04x f %08x %08x %08x d %08x%08x %08x%08x %08x%08x dot %08x %08x %08x): "
                     "%016llx, card %016llx\n", kernel, tier, op, t, x[0], x[1], x[2], x[3], x[4], x[5], x[7], x[6], x[9], x[8], x[11], x[10],
                     x[12], x[13], x[14], static_cast<unsigned long long>(g), static_cast<unsigned long long>(c));
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

VTEST(gfx1030_nans_with_ieee_mode_match_the_cards) { VCHECK_EQ(run_both("n2_k"), 0); }

VTEST(gfx1030_nans_without_ieee_mode_match_the_cards) { VCHECK_EQ(run_both("n2_n"), 0); }

VTEST(gfx1030_nans_with_denormals_flushed_match_the_cards) { VCHECK_EQ(run_both("n2_f"), 0); }

VTEST(gfx1031_nans_are_gfx1030s_on_every_result_of_asm_nan2) {
  const auto want = digests_1031();
  VCHECK_EQ(fnv(card("n2_k")), want.at("n2_k"));
  VCHECK_EQ(fnv(card("n2_n")), want.at("n2_n"));
  VCHECK_EQ(fnv(card("n2_f")), want.at("n2_f"));
}

VTEST_MAIN
