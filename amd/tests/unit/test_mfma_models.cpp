// The matrix-core models of Khattak, Mikaitis and Graziani (arXiv 2609.14845; amd/src/mfma_models.hpp), opt-in through
// VGPU_MFMA_MODEL=khattak.
//
// Three checks, none of them against a card or against the paper's own MATLAB code (neither is at hand):
//  1. the C++ port against tools/mfma-models/model_ref.py, a second writing of the MATLAB files in exact rational arithmetic, on
//     the vectors in amd/tests/data/mfma_models/vectors.txt (VGPU_MFMA_MODEL_VECTORS names another file, for a larger sweep);
//  2. a few vectors worked out by hand from the models' parameters;
//  3. the instructions: rocWMMA's 16x16x16 binary16 and bfloat16 forms (amd/tests/hipcc/mfma_numerics.cpp) on gfx90a and gfx942,
//     on the tile path and on the reference path, with the switch on and off.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "mfma_models.hpp"
#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vtest.hpp"

using namespace vgpu;
namespace kh = vgpu::amd::khattak;

namespace {

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
bool same(uint32_t got, uint32_t want) { return got == want || (from_bits(got) != from_bits(got) && from_bits(want) != from_bits(want)); }

// A value from its format's code (the vector file holds codes): 'h' binary16, 'b' bfloat16, 'F' xf32 (19 bits), 'e' fp8 e4m3 FNUZ,
// 'g' fp8 e5m2 FNUZ.
double decode(char t, uint32_t c) {
  switch (t) {
    case 'h': {
      const uint16_t u = static_cast<uint16_t>(c);
      _Float16 h;
      std::memcpy(&h, &u, 2);
      return static_cast<double>(h);
    }
    case 'b': return from_bits(c << 16);
    case 'F': return from_bits((c >> 18 & 1) << 31 | (c >> 10 & 255) << 23 | (c & 1023) << 13);
    case 'e': {
      const double s = c & 0x80 ? -1 : 1;
      const uint32_t e = c >> 3 & 15, m = c & 7;
      return s * (e == 0 ? m / 8.0 * 0x1p-7 : (1 + m / 8.0) * std::ldexp(1.0, static_cast<int>(e) - 8));
    }
    default: {
      const double s = c & 0x80 ? -1 : 1;
      const uint32_t e = c >> 2 & 31, m = c & 3;
      return s * (e == 0 ? m / 4.0 * 0x1p-15 : (1 + m / 4.0) * std::ldexp(1.0, static_cast<int>(e) - 16));
    }
  }
}

struct Vec {
  std::string model;
  char ta, tb;
  uint32_t k;
  std::vector<uint32_t> a, b;
  uint32_t c, d;
  size_t line;
};

std::vector<uint32_t> hex_list(const std::string& s) {
  std::vector<uint32_t> v;
  std::istringstream in(s);
  std::string w;
  while (in >> w) v.push_back(static_cast<uint32_t>(std::stoul(w, nullptr, 16)));
  return v;
}

std::vector<Vec> load() {
  const char* env = std::getenv("VGPU_MFMA_MODEL_VECTORS");
  const std::string path = env ? env : std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/mfma_models/vectors.txt";
  std::ifstream in(path);
  if (!in) throw vtest::Failure("no vectors at " + path);
  std::vector<Vec> out;
  std::string line;
  for (size_t n = 1; std::getline(in, line); ++n) {
    if (line.empty() || line[0] == '#') continue;
    std::vector<std::string> part;
    std::istringstream ls(line);
    std::string p;
    while (std::getline(ls, p, '|')) part.push_back(p);
    if (part.size() != 5) throw vtest::Failure("bad vector line " + std::to_string(n));
    Vec v;
    std::istringstream head(part[0]);
    std::string ta, tb;
    head >> v.model >> ta >> tb >> v.k;
    v.ta = ta[0], v.tb = tb[0], v.a = hex_list(part[1]), v.b = hex_list(part[2]), v.c = hex_list(part[3])[0], v.d = hex_list(part[4])[0], v.line = n;
    if (v.a.size() != v.k || v.b.size() != v.k) throw vtest::Failure("a vector of the wrong length, line " + std::to_string(n));
    out.push_back(std::move(v));
  }
  return out;
}

kh::Model model_of(const std::string& m) { return m == "cdna1" ? kh::Model::Cdna1 : m == "cdna2" ? kh::Model::Cdna2 : kh::Model::Cdna3; }

uint32_t run_model(const Vec& v) {
  std::vector<double> a, b;
  for (uint32_t i = 0; i < v.k; ++i) a.push_back(decode(v.ta, v.a[i])), b.push_back(decode(v.tb, v.b[i]));
  return bits(static_cast<float>(kh::dot(model_of(v.model), v.ta, v.tb, a.data(), b.data(), 1, v.k, from_bits(v.c))));
}

// ---- the instructions --------------------------------------------------------------------------------------------------------
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

// Element (0,0) of D = A x B + C for one K = 16 vector (row 0 of A, column 0 of B, C(0,0); everything else zero).
uint32_t d00(const amd::CodeObject& o, char type, const std::vector<uint32_t>& ac, const std::vector<uint32_t>& bc, uint32_t cbits) {
  MemoryManager mem(8ull << 20);
  std::vector<uint16_t> a(256, 0), b(256, 0);
  std::vector<float> c(256, 0.0f);
  for (size_t k = 0; k < 16; ++k) a[k] = static_cast<uint16_t>(ac[k]), b[k] = static_cast<uint16_t>(bc[k]);
  c[0] = from_bits(cbits);
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pc = upload(mem, c), pd = mem.alloc(256 * 4);
  run(o, type == 'h' ? "_Z7mma_f16PKDF16_S0_PKfPf" : "_Z8mma_bf16PK12hip_bfloat16S1_PKfPf", mem, {pa, pb, pc, pd});
  std::vector<float> d(256);
  mem.read(pd, d.data(), 256 * 4);
  return bits(d[0]);
}

struct Guard {   // puts the switches back
  ~Guard() {
    amd::set_khattak_matrix_model(-1);
    amd::set_cdna2_matrix_model(-1);
    amd::set_matrix_fast_path(true);
  }
};

// The vectors of the file that are one rocWMMA 16x16x16 instruction: K = 16, binary16 or bfloat16 on both sides.
void through_instruction(const char* arch, const char* model) {
  const amd::CodeObject o = object(arch);
  Guard g;
  amd::set_khattak_matrix_model(1);
  size_t n = 0;
  for (const bool fast : {true, false}) {
    amd::set_matrix_fast_path(fast);
    for (const Vec& v : load()) {
      if (v.model != model || v.k != 16 || v.ta != v.tb || (v.ta != 'h' && v.ta != 'b')) continue;
      const uint32_t got = d00(o, v.ta, v.a, v.b, v.c);
      if (!same(got, v.d)) {
        char buf[200];
        std::snprintf(buf, sizeof buf, "%s %c line %zu on the %s path: 0x%08x, the model says 0x%08x", arch, v.ta, v.line, fast ? "tile" : "reference", got, v.d);
        throw vtest::Failure(buf);
      }
      ++n;
    }
  }
  VCHECK(n >= 100);
}

}  // namespace

// 1.
VTEST(the_port_agrees_with_the_second_writing_on_every_vector) {
  const std::vector<Vec> vs = load();
  VCHECK(vs.size() >= 600);
  size_t seen[3] = {0, 0, 0};
  for (const Vec& v : vs) {
    ++seen[static_cast<int>(model_of(v.model))];
    const uint32_t got = run_model(v);
    if (!same(got, v.d)) {
      char buf[160];
      std::snprintf(buf, sizeof buf, "%s %c%c K=%u, line %zu: 0x%08x, the reference says 0x%08x", v.model.c_str(), v.ta, v.tb, v.k, v.line, got, v.d);
      throw vtest::Failure(buf);
    }
  }
  for (size_t s : seen) VCHECK(s >= 100);
}

// 2. Worked by hand. binary16: a = 1.75 and b = 2^-24 (the smallest subnormal), 1 x 1.
//  CDNA1: exact: 1 + 1.75 * 2^-24 = 1 + 0.875 ulp, rounds up: 1 + 2^-23.
//  CDNA2: b is a subnormal input, read as zero: 1.
//  CDNA3: the second product is aligned to the first's exponent with one extra bit and the bit lost is cut: 2^24 + 1 in units of
//         2^-24, a tie between 1 and 1 + 2^-23, to even: 1. With c = 1 instead of the first product the sum is the bigger one, eight
//         more bits are kept (1.75 * 2^-24 = 448 * 2^-32 exactly) and it rounds up: 1 + 2^-23.
VTEST(the_models_part_where_the_arithmetic_says_they_do) {
  const double a1[4] = {1.75, 1.0, 0, 0}, b1[4] = {0x1p-24, 1.0, 0, 0}, a2[4] = {1.75, 0, 0, 0}, b2[4] = {0x1p-24, 0, 0, 0};
  const uint32_t one = bits(1.0f), up = bits(1.0f + 0x1p-23f);
  const auto f = [](kh::Model m, const double* a, const double* b, double c) { return bits(static_cast<float>(kh::dot(m, 'h', 'h', a, b, 1, 4, c))); };
  VCHECK_EQ(f(kh::Model::Cdna1, a1, b1, 0.0), up);
  VCHECK_EQ(f(kh::Model::Cdna2, a1, b1, 0.0), one);
  VCHECK_EQ(f(kh::Model::Cdna3, a1, b1, 0.0), one);
  VCHECK_EQ(f(kh::Model::Cdna1, a2, b2, 1.0), up);
  VCHECK_EQ(f(kh::Model::Cdna2, a2, b2, 1.0), one);
  VCHECK_EQ(f(kh::Model::Cdna3, a2, b2, 1.0), up);
  // bfloat16: 2^-100 * 2^-30 = 2^-130, a subnormal binary32. CDNA1 and CDNA3 keep it; CDNA2 flushes the product.
  const double a3[2] = {0x1p-100, 0}, b3[2] = {0x1p-30, 0};
  const auto g = [&](kh::Model m) { return bits(static_cast<float>(kh::dot(m, 'b', 'b', a3, b3, 1, 2, 0.0))); };
  VCHECK_EQ(g(kh::Model::Cdna1), 0x00080000u);
  VCHECK_EQ(g(kh::Model::Cdna2), 0u);
  VCHECK_EQ(g(kh::Model::Cdna3), 0x00080000u);
  // Infinities and NaN are settled over the whole row, in every model.
  const double ai[2] = {INFINITY, 1}, ni[2] = {-INFINITY, 1}, o2[2] = {1, 1};
  for (const kh::Model m : {kh::Model::Cdna1, kh::Model::Cdna2, kh::Model::Cdna3}) {
    VCHECK_EQ(bits(static_cast<float>(kh::dot(m, 'h', 'h', ai, o2, 1, 2, 3.0))), 0x7f800000u);
    VCHECK_EQ(bits(static_cast<float>(kh::dot(m, 'h', 'h', ni, o2, 1, 2, 3.0))), 0xff800000u);
    VCHECK(std::isnan(kh::dot(m, 'h', 'h', ai, o2, 1, 2, -INFINITY)));
  }
}

// 3.
VTEST(an_mi250x_sums_as_the_cdna2_model_through_the_instruction) { through_instruction("gfx90a", "cdna2"); }
VTEST(an_mi300x_sums_as_the_cdna3_model_through_the_instruction) { through_instruction("gfx942", "cdna3"); }

// The switch: off, both targets give what they always gave (the double sum); on, gfx90a gives the CDNA2 model and gfx942 the CDNA3
// model, and on gfx90a the model takes the place of the FTTN rule.
VTEST(the_switch_changes_the_sum_and_nothing_else) {
  Guard g;
  // The hand vector of the test above, as binary16 codes: 1.75 (0x3f00) x 2^-24 (0x0001), then 1 (0x3c00) x 1, zeros.
  std::vector<uint32_t> ac(16, 0), bc(16, 0);
  ac[0] = 0x3f00, bc[0] = 0x0001, ac[1] = 0x3c00, bc[1] = 0x3c00;
  const uint32_t one = bits(1.0f), up = bits(1.0f + 0x1p-23f);
  for (const bool fast : {true, false}) {
    amd::set_matrix_fast_path(fast);
    for (const char* arch : {"gfx90a", "gfx942"}) {
      const amd::CodeObject o = object(arch);
      amd::set_khattak_matrix_model(0);
      VCHECK_EQ(d00(o, 'h', ac, bc, 0), up);   // the double sum: 1 + 0.875 ulp rounds up
      amd::set_khattak_matrix_model(1);
      VCHECK_EQ(d00(o, 'h', ac, bc, 0), one);  // the models: CDNA2 flushes the subnormal, CDNA3 cuts the bit it aligns away
      amd::set_cdna2_matrix_model(1);          // FTTN's rule is on too: the model wins
      VCHECK_EQ(d00(o, 'h', ac, bc, 0), one);
      amd::set_cdna2_matrix_model(-1);
    }
  }
}

// Off unless the environment asks (ctest runs this binary a second time with VGPU_MFMA_MODEL=khattak).
VTEST(the_model_is_off_unless_the_environment_asks_for_it) {
  Guard g;
  const char* e = std::getenv("VGPU_MFMA_MODEL");
  const bool on = e && e[0] == 'k';
  amd::set_khattak_matrix_model(-1);
  VCHECK_EQ(amd::khattak_matrix_model(), on);
  std::vector<uint32_t> ac(16, 0), bc(16, 0);
  ac[0] = 0x3f00, bc[0] = 0x0001, ac[1] = 0x3c00, bc[1] = 0x3c00;
  VCHECK_EQ(d00(object("gfx942"), 'h', ac, bc, 0), on ? bits(1.0f) : bits(1.0f + 0x1p-23f));
}
