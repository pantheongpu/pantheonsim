// v_rcp_f32, v_rsq_f32 and v_sqrt_f32 of gfx1030 and gfx1031 (amd/src/rdna2_trans.cpp) against the RX 6800's results over all 2^32 operands
// (issues #729, #741): tools/numerics/amd/gfx1030.hashes.txt holds, for each operation, 64 hashes of 2^26 consecutive operands' results
// (the definition is in tools/numerics/amd_trans_probe.cpp: 4096-result blocks, FNV-1a 64 over the 32-bit results, a chunk is FNV-1a over
// its blocks' hashes). Every chunk must have the card's hash; an RX 6700 XT gave the same hashes. A NaN operand is read as the card
// returns it (the operand with the quiet bit set, test_amd_gcn_cross_arch's vectors); the function under test does not make NaNs of
// payloads, the instruction's NaN rule does (rdna2_nan in gcn_exec.cpp).
#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "../../src/rdna2_trans.hpp"
#include "vtest.hpp"

namespace {

constexpr uint64_t kBasis = 0xcbf29ce484222325ull, kPrime = 0x100000001b3ull;

using Fn = float (*)(float);

uint32_t result_bits(Fn fn, uint32_t in) {
  const float r = fn(std::bit_cast<float>(in));
  if ((in & 0x7f800000u) == 0x7f800000u && (in & 0x7fffffu)) return in | 0x400000u;   // a NaN operand: the card's rule, not this function's
  return std::bit_cast<uint32_t>(r);
}

uint64_t chunk_hash(Fn fn, uint64_t first) {
  uint64_t chunk = kBasis;
  for (uint64_t block = first; block < first + (1ull << 26); block += 4096) {
    uint64_t h = kBasis;
    for (uint32_t i = 0; i < 4096; ++i) h = (h ^ result_bits(fn, static_cast<uint32_t>(block + i))) * kPrime;
    chunk = (chunk ^ h) * kPrime;
  }
  return chunk;
}

std::map<std::string, std::map<uint32_t, uint64_t>> card_hashes() {
  std::ifstream f(std::string(VGPU_SOURCE_DIR) + "/tools/numerics/amd/gfx1030.hashes.txt");
  if (!f) throw vtest::Failure("no hashes file");
  std::map<std::string, std::map<uint32_t, uint64_t>> out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    char op[16], at[16];
    unsigned long long h;
    if (std::sscanf(line.c_str(), "%15s %15s %llx", op, at, &h) != 3 || std::string(at) == "all") continue;
    out[op][static_cast<uint32_t>(std::stoul(at, nullptr, 16))] = h;
  }
  return out;
}

// (Under a sanitizer one chunk in eight: the whole is a minute of one core natively.)
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
constexpr unsigned kStride = 8;
#else
constexpr unsigned kStride = 1;
#endif

int check_operation(const char* name, Fn fn) {
  static const auto cards = card_hashes();
  const auto& want = cards.at(name);
  VCHECK_EQ(want.size(), size_t{64});
  std::vector<uint64_t> got(64);
  std::atomic<unsigned> next{0};
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < std::max(1u, std::min(16u, std::thread::hardware_concurrency())); ++t)
    pool.emplace_back([&] {
      for (unsigned c; (c = next++ * kStride) < 64;) got[c] = chunk_hash(fn, uint64_t{c} << 26);
    });
  for (auto& t : pool) t.join();
  int wrong = 0;
  for (unsigned c = 0; c < 64; c += kStride)
    if (got[c] != want.at(c << 26) && wrong++ < 8) std::fprintf(stderr, "  %s: operands from %08x: %016llx, the card %016llx\n", name, c << 26, static_cast<unsigned long long>(got[c]), static_cast<unsigned long long>(want.at(c << 26)));
  return wrong;
}

}  // namespace

VTEST(rcp_is_the_rx6800s_on_all_2_to_the_32_operands) { VCHECK_EQ(check_operation("RCP", vgpu::amd::rdna2::rcp), 0); }
VTEST(rsq_is_the_rx6800s_on_all_2_to_the_32_operands) { VCHECK_EQ(check_operation("RSQ", vgpu::amd::rdna2::rsq), 0); }
VTEST(sqrt_is_the_rx6800s_on_all_2_to_the_32_operands) { VCHECK_EQ(check_operation("SQRT", vgpu::amd::rdna2::sqrt), 0); }

VTEST_MAIN
