// A card's dump of wmma results (tools/fttn/tc_dump.cu; the files are described in fit_tc_model.py) against the simulator's own
// tensor-core sum, include/vgpu/exec/tc_ampere.hpp, by a named rule: every result word of every form must be the card's.
//
//   g++ -std=c++20 -O2 -I include tools/fttn/tc_dump_check.cpp build/libvgpu.a -lpthread -ldl -o tc_dump_check    (from the root, after a build)
//   tc_dump_check <dump-dir> ampere|hopper
//
// Prints one line a form and style and "TOTAL same/all"; exit 0 only if every word is the same. A rule goes into
// tc86::device_sum_model for an architecture only when this says 100 % for a card of it.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/exec/tc_ampere.hpp"

namespace tc = vgpu::exec::tc86;

static std::vector<uint32_t> load(const std::string& path) {
  std::vector<uint32_t> v;
  if (FILE* f = std::fopen(path.c_str(), "rb")) {
    std::fseek(f, 0, SEEK_END);
    v.resize(static_cast<size_t>(std::ftell(f)) / 4);
    std::fseek(f, 0, SEEK_SET);
    if (std::fread(v.data(), 4, v.size(), f) != v.size()) v.clear();
    std::fclose(f);
  }
  return v;
}
static float f32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static uint16_t half_at(const std::vector<uint32_t>& w, size_t i) { return static_cast<uint16_t>(w[i / 2] >> (16 * (i % 2))); }

int main(int argc, char** argv) {
  if (argc < 3) return std::fprintf(stderr, "usage: %s <dump-dir> ampere|hopper\n", argv[0]), 2;
  const std::string dir = argv[1];
  const tc::SumRule& rule = std::string(argv[2]) == "hopper" ? tc::kHopperRule : tc::kAmpereRule;
  std::printf("rule: %s\n", rule.name);
  long all = 0, same = 0;
  for (const char* style : {"random", "edges"}) {
    const std::string s = style;
    const auto ha = load(dir + "/in.ha." + s + ".bin"), hb = load(dir + "/in.hb." + s + ".bin"), hc = load(dir + "/in.hc." + s + ".bin");
    const auto fc = load(dir + "/in.fc." + s + ".bin"), bfa = load(dir + "/in.bfa." + s + ".bin"), bfb = load(dir + "/in.bfb." + s + ".bin");
    struct Form { const char* name; int kind; bool acc16; };
    for (const Form& fm : {Form{"wmma_f16_f32acc", 0, false}, Form{"wmma_f16_f16acc", 0, true}, Form{"wmma_bf16_f32acc", 1, false}, Form{"wmma_tf32_f32acc", 2, false}}) {
      const auto out = load(dir + "/out." + fm.name + "." + s + ".bin");
      if (out.empty() || fc.empty()) return std::fprintf(stderr, "no dump for %s %s in %s\n", fm.name, style, dir.c_str()), 2;
      const size_t tiles = out.size() * (fm.acc16 ? 2 : 1) / 256;
      const unsigned K = fm.kind == 2 ? 8 : 16;
      long n = 0, ok = 0;
      for (size_t t = 0; t < tiles; ++t) {
        float A[16 * 16], B[16 * 16], C[256], D[256];
        for (unsigned i = 0; i < 16; ++i)
          for (unsigned k = 0; k < K; ++k) {
            if (fm.kind == 2) {
              A[i * K + k] = f32(fc[t * 128 + i * 8 + k] & 0xffffe000u);            // (the tensor core reads the top 19 bits)
              B[k * 16 + i] = f32(fc[1024 + t * 128 + i * 8 + k] & 0xffffe000u);     // B is stored by column: [column][k]
            } else if (fm.kind == 1) {
              A[i * K + k] = f32(static_cast<uint32_t>(half_at(bfa, t * 256 + i * 16 + k)) << 16);
              B[k * 16 + i] = f32(static_cast<uint32_t>(half_at(bfb, t * 256 + i * 16 + k)) << 16);
            } else {
              A[i * K + k] = vgpu::simd::f16_to_f32(half_at(ha, t * 256 + i * 16 + k));
              B[k * 16 + i] = vgpu::simd::f16_to_f32(half_at(hb, t * 256 + i * 16 + k));
            }
          }
        for (unsigned e = 0; e < 256; ++e)
          C[e] = fm.acc16 ? vgpu::simd::f16_to_f32(half_at(hc, t * 256 + e)) : f32(fm.kind == 2 ? fc[2048 + t * 256 + e] : fc[t * 256 + e]);
        // (a wmma tf32 fragment is two m16n8k4 on a card: chunks of at most 4)
        tc::mma(rule, fm.kind == 2, 16, 16, K, fm.kind == 0 ? tc::kEminF16 : tc::kEminWide, fm.acc16, A, K, B, 16, C, 16, D, 16, fm.kind == 2 ? 4u : 16u);
        for (unsigned e = 0; e < 256; ++e, ++n) {
          uint32_t got, card;
          if (fm.acc16) {
            got = D[e] != D[e] ? 0x7fffu : vgpu::simd::f64_to_f16_bits(static_cast<double>(D[e]));
            card = half_at(out, t * 256 + e);
          } else {
            got = bits(D[e]);
            card = out[t * 256 + e];
          }
          if (got == card) ++ok;
          else if (n - ok <= 3) std::printf("  %s %s tile %zu element %u: card %08x, rule %08x\n", fm.name, style, t, e, card, got);
        }
      }
      std::printf("%-18s %-6s %ld of %ld\n", fm.name, style, ok, n);
      all += n, same += ok;
    }
  }
  std::printf("TOTAL %ld of %ld (%s)\n", same, all, same == all ? "every result the card's" : "NOT the card's rule");
  return same == all ? 0 : 1;
}
