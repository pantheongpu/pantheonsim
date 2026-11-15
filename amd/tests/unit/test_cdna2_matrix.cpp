// The binary16 and bfloat16 matrix instructions of gfx90a (an MI250X) against what the FTTN paper reports of that card (Li, Li,
// Fang, Swirydowicz, Laguna, Gopalakrishnan, CCGrid 2024, arXiv 2403.00232; the MI250X rows of its results table): subnormal inputs
// and outputs are not handled, three extra bits and round to nearest even, "FMA unit width 1" -- a binary32 rounding after every
// product. The inputs are the ones the paper's ten tests feed element (1,1) of D = A x B + C (rewritten here as fixed vectors; the
// FTTN programs themselves are not in this repository, tools/fttn fetches and runs them), plus its motivating GEMM, D = C - A x B
// with K = 2^13, whose answer the paper gives per card: 0 on an MI250X.
//
// Each vector has two answers: `paper`, what an MI250X gives by the paper's description, and `before`, what the simulator gave
// by default (the rule is opt-in, VGPU_MFMA_CDNA2=fttn) and gives on every other target (the K products and C summed in double,
// rounded once).
// The kernels are rocWMMA's (amd/tests/hipcc/mfma_numerics.cpp, built for gfx90a and gfx942 by amd/tests/hipcc/build.sh), so the
// layout of the operands is AMD's own; each vector runs on the tile path and on the reference path.
//
// No CDNA2 card was available: this holds the opt-in rule to the paper's summary, not to a card, and the paper's summary does not
// fix a rule: the published bit-level model of the same card (MMA-Sim, arXiv 2511.10909: products added in pairs, in fours, then to the
// accumulator, each sum rounded and flushed) gives the paper's summary row too, but not its motivating example's 0. Where the paper
// does not decide a bit (a NaN's payload, when a subnormal sum is flushed) the vector says "chosen".
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

amd::CodeObject object(const char* arch) {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/hipcc/mfma_numerics." + arch + ".o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

template <typename T>
uint64_t upload(MemoryManager& mem, const std::vector<T>& v) {
  const uint64_t p = mem.alloc(v.size() * sizeof(T));
  mem.write(p, v.data(), v.size() * sizeof(T));
  return p;
}

void run(const amd::CodeObject& o, const char* name, MemoryManager& mem, const std::vector<uint64_t>& values) {
  const amd::Kernel* k = amd::find_kernel(o, name);
  if (!k) throw vtest::Failure(std::string("no kernel named ") + name);
  std::vector<uint8_t> buf(k->kernarg_size, 0);
  for (size_t i = 0; i < values.size() && i < k->args.size(); ++i) {
    const amd::KernelArg& a = k->args[i];
    for (uint32_t b = 0; b < a.size && b < 8; ++b) buf[a.offset + b] = static_cast<uint8_t>(values[i] >> (8 * b));
  }
  const uint64_t p = mem.alloc(buf.empty() ? 1 : buf.size());
  if (!buf.empty()) mem.write(p, buf.data(), buf.size());
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = p;
  d.groups[0] = 1;
  d.group_size[0] = 64;
  amd::execute(d, mem);
}

uint32_t bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
float from_bits(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
// A value as the 16 bits of its input format: 'h' binary16, 'b' bfloat16 (the top half of the float). Every value used is exact.
uint16_t pack(char type, float v) {
  if (type == 'b') {
    if ((bits(v) & 0xffffu) != 0) throw vtest::Failure("a test value is not a bfloat16");
    return static_cast<uint16_t>(bits(v) >> 16);
  }
  const _Float16 h = static_cast<_Float16>(v);
  if (!(static_cast<float>(h) == v) && v == v) throw vtest::Failure("a test value is not a binary16");
  uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}

struct Term {
  int k;
  float a, b;
};
struct Vec {
  const char* name;   // the paper's test, and what it asks
  char type;          // 'h' or 'b'
  std::vector<Term> terms;
  float c;
  uint32_t paper;     // an MI250X, by the paper
  uint32_t before;    // the double sum
};

const float kSubH = 0x1p-24f, kMinH = 0x1p-14f;        // binary16: smallest subnormal, smallest normal
const float kSubB = 0x1p-133f, kMinB = 0x1p-126f;      // bfloat16
const uint32_t kOne = 0x3f800000u, kOneUlp = 0x3f800001u, kTwoUlp = 0x40000001u, kNan = 0x7fc00000u;

std::vector<Vec> vectors() {
  std::vector<Vec> v = {
      // ---- binary16 (fp16in.cpp) ----
      {"T_si_no: a subnormal A", 'h', {{0, kSubH, 1.0f}}, 0.0f, 0u, bits(0x1p-24f)},
      {"T_si_no: a subnormal B", 'h', {{0, 1.0f, kSubH}}, 0.0f, 0u, bits(0x1p-24f)},
      {"T_si_no: the largest subnormal A", 'h', {{0, 0x1p-14f - 0x1p-24f, 0x1p10f}}, 0.0f, 0u, bits((0x1p-14f - 0x1p-24f) * 0x1p10f)},
      {"T_sa: a subnormal C", 'h', {}, 0x1p-127f, 0u, bits(0x1p-127f)},
      {"T_sa: the smallest normal C is kept", 'h', {}, 0x1p-126f, bits(0x1p-126f), bits(0x1p-126f)},
      {"the smallest normal A is kept", 'h', {{0, kMinH, 1.0f}}, 0.0f, bits(0x1p-14f), bits(0x1p-14f)},
      {"T_rnd_dir: products, positive", 'h', {{0, kMinH, 0x1p-9f + 0x1p-10f}, {1, 1.0f, 2.0f}}, 0.0f, kTwoUlp, kTwoUlp},
      {"T_rnd_dir: products, negative", 'h', {{0, kMinH, -0x1p-9f - 0x1p-10f}, {1, 1.0f, -2.0f}}, 0.0f, kTwoUlp | 0x80000000u, kTwoUlp | 0x80000000u},
      {"T_rnd_dir: accumulator, positive", 'h', {{0, kMinH, 0x1p-9f + 0x1p-10f}}, 2.0f, kTwoUlp, kTwoUlp},
      {"T_rnd_dir: accumulator, negative", 'h', {{0, 1.0f, -2.0f}}, -0x1p-23f - 0x1p-24f, kTwoUlp | 0x80000000u, kTwoUlp | 0x80000000u},
      {"T_1_bit", 'h', {{0, 1.0f, 1.0f}}, -std::nextafterf(1.0f, 0.0f), bits(0x1p-24f), bits(0x1p-24f)},
      {"T_3_bits_fin_rnd: three extra bits", 'h', {{0, 1.0f, 1.25f}}, -(0x1p-3f + 0x1p-24f + 0x1p-26f), 0x3f8fffffu, 0x3f8fffffu},
      {"T_3_bits_fin_rnd: ties to even", 'h', {{0, 0x1p-10f, kMinH}}, 1.0f, kOne, kOne},
      {"T_3_bits_fin_rnd: sticky", 'h', {{0, 1.0f, 1.0f}}, 0x1p-24f + 0x1p-27f, kOneUlp, kOneUlp},
      // T_pres_extra_acc: 1 + three half-ulps. Rounded after each: 1. Kept to the end: 1 + 1.5 ulp, a tie, to even: 1 + 2 ulp.
      {"T_pres_extra_acc", 'h', {{0, 0x1p-10f, kMinH}, {1, 0x1p-10f, kMinH}, {2, 0x1p-10f, kMinH}}, 1.0f, kOne, 0x3f800002u},
      // T_acc_order: the big product first or second, with a half-ulp accumulator: the order matters only if each step rounds.
      {"T_acc_order: big product at k = 0", 'h', {{0, 1.0f, 1.0f}, {1, 0x1p-10f, kMinH}}, 0x1p-24f, kOne, kOneUlp},
      {"T_acc_order: big product at k = 1", 'h', {{0, 0x1p-10f, kMinH}, {1, 1.0f, 1.0f}}, 0x1p-24f, kOneUlp, kOneUlp},
      // ---- bfloat16 (bf16in.cpp) ----
      {"bf16 T_si_no: a subnormal A", 'b', {{0, kSubB, 1.0f}}, 0.0f, 0u, bits(kSubB)},
      {"bf16 T_si_no: a subnormal B", 'b', {{0, 1.0f, kSubB}}, 0.0f, 0u, bits(kSubB)},
      {"bf16 T_no_so: normals whose product is subnormal", 'b', {{0, kMinB, 0.5f}}, 0.0f, 0u, bits(0x1p-127f)},
      {"bf16 T_sa: a subnormal C", 'b', {}, 0x1p-127f, 0u, bits(0x1p-127f)},
      {"bf16 T_rnd_dir: products, positive", 'b', {{0, 1.0f, 0x1p-23f + 0x1p-24f}, {1, 1.0f, 2.0f}}, 0.0f, kTwoUlp, kTwoUlp},
      {"bf16 T_1_bit", 'b', {{0, 1.0f, 1.0f}}, -std::nextafterf(1.0f, 0.0f), bits(0x1p-24f), bits(0x1p-24f)},
      {"bf16 T_3_bits_fin_rnd: three extra bits", 'b', {{0, 1.0f, 1.25f}}, -(0x1p-3f + 0x1p-24f + 0x1p-26f), 0x3f8fffffu, 0x3f8fffffu},
      {"bf16 T_pres_extra_acc", 'b', {{0, 0x1p-12f, 0x1p-12f}, {1, 0x1p-12f, 0x1p-12f}, {2, 0x1p-12f, 0x1p-12f}}, 1.0f, kOne, 0x3f800002u},
      // ---- chosen, not in the paper ----
      {"a negative subnormal A with C = -0: +0 (the fifteen other products are +0)", 'h', {{0, -kSubH, 1.0f}}, -0.0f, 0u, bits(-0x1p-24f)},
      {"chosen: a flushed subnormal times an infinity is 0 * inf", 'h', {{0, kSubH, INFINITY}}, 0.0f, kNan, 0x7f800000u},
      {"chosen: a subnormal running sum is flushed at once (2^-127 then + 2^-127)", 'b', {{0, kMinB, 0.5f}, {1, kMinB, 0.5f}}, 0.0f, 0u, bits(0x1p-126f)},
      {"chosen: a subnormal C is read as zero before the first product (2^-126 + 2^-127)", 'b', {{0, kMinB, 1.0f}}, 0x1p-127f, bits(0x1p-126f), 0x00c00000u},
      {"an infinity", 'h', {{0, INFINITY, 1.0f}, {1, 1.0f, 1.0f}}, 3.0f, 0x7f800000u, 0x7f800000u},
  };
  // T_blk_fma_width: 1 in C, a half ulp at k = 0 and another moved along K. Rounded once for the block the two make an ulp; rounded
  // after each product both are lost. On an MI250X the paper finds them lost at the first step: width 1.
  for (int j = 1; j < 16; ++j) {
    v.push_back({"T_blk_fma_width", 'h', {{0, 0x1p-10f, kMinH}, {j, 0x1p-10f, kMinH}}, 1.0f, kOne, kOneUlp});
    v.push_back({"bf16 T_blk_fma_width", 'b', {{0, 0x1p-12f, 0x1p-12f}, {j, 0x1p-12f, 0x1p-12f}}, 1.0f, kOne, kOneUlp});
  }
  return v;
}

// Element (1,1) of D for a vector; the other elements of A, B and C are zero (and must come out zero).
uint32_t d11(const amd::CodeObject& o, const Vec& v) {
  MemoryManager mem(8ull << 20);
  std::vector<uint16_t> a(256, 0), b(256, 0);
  std::vector<float> c(256, 0.0f);
  for (const Term& t : v.terms) {
    a[t.k] = pack(v.type, t.a);   // row 0 of A, row-major
    b[t.k] = pack(v.type, t.b);   // column 0 of B, column-major
  }
  c[0] = v.c;
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pc = upload(mem, c), pd = mem.alloc(256 * 4);
  run(o, v.type == 'h' ? "_Z7mma_f16PKDF16_S0_PKfPf" : "_Z8mma_bf16PK12hip_bfloat16S1_PKfPf", mem, {pa, pb, pc, pd});
  std::vector<float> d(256);
  mem.read(pd, d.data(), 256 * 4);
  bool finite = true;
  for (const Term& t : v.terms) finite = finite && std::isfinite(t.a) && std::isfinite(t.b);
  for (int e = 1; e < 256 && finite; ++e)   // (an infinity in row 1 of A or column 1 of B meets the zeros of the other rows and columns)
    if (bits(d[e]) != 0) throw vtest::Failure(std::string(v.name) + ": an element that should be +0 is not");
  return bits(d[0]);
}

// Runs every vector on a target and holds it to one of its two answers, on the tile path and on the reference path.
void check(const char* arch, int model, bool want_paper) {
  const amd::CodeObject o = object(arch);
  amd::set_cdna2_matrix_model(model);
  for (const bool fast : {true, false}) {
    amd::set_matrix_fast_path(fast);
    for (const Vec& v : vectors()) {
      const uint32_t got = d11(o, v), want = want_paper ? v.paper : v.before;
      const bool same = got == want || (want == kNan && from_bits(got) != from_bits(got));   // (a NaN: any payload)
      if (!same) {
        amd::set_matrix_fast_path(true);
        amd::set_cdna2_matrix_model(-1);
        char buf[160];
        std::snprintf(buf, sizeof buf, " on %s (%s path): 0x%08x, want 0x%08x", arch, fast ? "tile" : "reference", got, want);
        throw vtest::Failure(std::string(v.name) + buf);
      }
    }
  }
  amd::set_matrix_fast_path(true);
  amd::set_cdna2_matrix_model(-1);
}

// The paper's motivating GEMM (section V): C = 2^20, A = 2^10 at k = 0, 2^-2 at odd k and 2^-3 at even k, B = 2^10 at k = 0 and
// -2^-3 elsewhere, K = 2^13, D = C - A x B. The real-number answer is 191.984375.
uint32_t motivating(const char* arch, int model, bool fast) {
  const amd::CodeObject o = object(arch);
  amd::set_cdna2_matrix_model(model);
  amd::set_matrix_fast_path(fast);
  const int kt = 512, K = 16 * kt;
  MemoryManager mem(16ull << 20);
  std::vector<uint16_t> a(16 * K), b(16 * K);
  for (int i = 0; i < 16; ++i)
    for (int k = 0; k < K; ++k) {
      a[i * K + k] = pack('h', k == 0 ? 0x1p10f : k % 2 ? 0x1p-2f : 0x1p-3f);
      b[i * K + k] = pack('h', k == 0 ? 0x1p10f : -0x1p-3f);
    }
  std::vector<float> c(256, 0x1p20f);
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pc = upload(mem, c), pd = mem.alloc(256 * 4);
  run(o, "_Z14gemm_minus_f16PKDF16_S0_PKfPfi", mem, {pa, pb, pc, pd, static_cast<uint64_t>(kt)});
  std::vector<float> d(256);
  mem.read(pd, d.data(), 256 * 4);
  amd::set_matrix_fast_path(true);
  amd::set_cdna2_matrix_model(-1);
  for (int e = 1; e < 256; ++e)
    if (bits(d[e]) != bits(d[0])) throw vtest::Failure("the motivating example's elements differ from one another");
  return bits(d[0]);
}

}  // namespace

VTEST(an_mi250x_sums_binary16_and_bfloat16_as_the_fttn_paper_reports) { check("gfx90a", 1, true); }

VTEST(the_switch_restores_the_double_sum_on_gfx90a) { check("gfx90a", 0, false); }

// gfx942 (MI300X, MI325X) is not the paper's card and nobody has measured one: it keeps the double sum whatever the switch says.
VTEST(gfx942_keeps_the_double_sum) {
  check("gfx942", 1, false);
  check("gfx942", 0, false);
}

// And the two answers differ where the paper's MI250X differs from an exact block: the subnormals, the blocks, the order.
VTEST(the_vectors_tell_the_two_sums_apart) {
  size_t differ = 0, width = 0;
  for (const Vec& v : vectors()) {
    differ += v.paper != v.before;
    width += std::string(v.name).find("T_blk_fma_width") != std::string::npos && v.paper != v.before;
  }
  VCHECK(differ >= 40);
  VCHECK_EQ(width, size_t{30});
}

VTEST(the_motivating_example_is_0_on_an_mi250x_as_the_paper_reports) {
  for (const bool fast : {true, false}) {
    VCHECK_EQ(motivating("gfx90a", 1, fast), 0u);                     // the paper: 0 on an MI250X
    VCHECK_EQ(motivating("gfx90a", 0, fast), bits(192.0f));           // the double sum of 16 products: 192, on no card of the paper
    VCHECK_EQ(motivating("gfx942", 1, fast), bits(192.0f));           // not measured: unchanged
  }
}

// The rule is off unless VGPU_MFMA_CDNA2=fttn is in the environment (ctest runs this binary a second time with it set).
VTEST(the_rule_is_off_unless_the_environment_asks_for_it) {
  const char* e = std::getenv("VGPU_MFMA_CDNA2");
  const bool off = !(e && e[0] == 'f');
  amd::set_cdna2_matrix_model(-1);
  VCHECK_EQ(amd::cdna2_matrix_model(), !off);
  const amd::CodeObject o = object("gfx90a");
  const Vec v{"T_si_no: a subnormal A", 'h', {{0, kSubH, 1.0f}}, 0.0f, 0u, bits(0x1p-24f)};
  VCHECK_EQ(d11(o, v), off ? v.before : v.paper);
}

VTEST_MAIN
