// The same lane-wise kernels (amd/tests/data/xarch.c: one builtin each, so one instruction) on every AMD target, on the same edge-case
// operands, to find where the targets disagree and which side is right (issues #721 to #730, docs/amd-cross-arch.md). Each test below
// is one finding: a rule every target must follow (with the card or the guide that says so), or a difference of the hardware with the
// value each architecture must give. What no card or guide settles is not asserted: it is in the issues, left open.
//
// VGPU_XARCH_DUMP=<dir> writes each target's results of every kernel (one file a target, one line a lane) for diffing by hand.
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

const char* const kTargets[] = {"gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1100", "gfx1201", "gfx1250"};

amd::CodeObject object(const std::string& target) {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/xarch." + target + ".o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

// The operands: floats of every kind, then halves in the low 16 bits.
std::vector<uint32_t> edges() {
  std::vector<uint32_t> e = {
      0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x3f000000, 0x40000000, 0x40400000, 0x40490fdb, 0x3dcccccd,
      0x00000001, 0x80000001, 0x007fffff, 0x00800000, 0x807fffff, 0x00400000, 0x7f7fffff, 0xff7fffff, 0x7f800000,
      0xff800000, 0x7fc00000, 0xffc00000, 0x7fc12345, 0xffc54321, 0x7fa00001, 0xffa00001, 0x7f800001, 0x7fffffff,
      0xffffffff, 0x4b000000, 0x4b800001, 0x4f000000, 0x4f800000, 0xcf000000, 0xcf000001, 0x5f800000, 0x1e3ce508,
      0x7149f2ca, 0x33800000, 0x38800000, 0x477fe000, 0x477ff000, 0x47800000, 0x387fc000, 0x33000000, 0x3f7fffff,
      0x3f800001, 0x42b17218, 0xc2fc0000, 0xc3150000, 0xc3000000, 0x43000000,
      // halves (and the same bits as denormal floats)
      0x00007e00, 0x00007c01, 0x0000fe01, 0x00007c00, 0x0000fc00, 0x00000001, 0x000003ff, 0x00000400, 0x00007bff,
      0x00003c00, 0x0000bc00, 0x00000200, 0x0000ffff, 0x00004000, 0x00005640};
  return e;
}

struct Inputs {
  std::vector<uint32_t> a, b, c;
};

Inputs inputs() {
  const std::vector<uint32_t> e = edges();
  const size_t n = e.size();
  Inputs in;
  const size_t total = (n * n + 63) / 64 * 64;
  for (size_t i = 0; i < total; ++i) {
    in.a.push_back(e[i % n]);
    in.b.push_back(e[(i / n) % n]);
    in.c.push_back(e[(i * 5 + (i / n) * 11) % n]);
  }
  return in;
}

uint64_t upload(MemoryManager& mem, const std::vector<uint32_t>& v) {
  const uint64_t p = mem.alloc(v.size() * 4);
  mem.write(p, v.data(), v.size() * 4);
  return p;
}

using Results = std::map<std::string, std::vector<uint32_t>>;   // kernel -> 2 words a lane

Results run_target(const std::string& target, const Inputs& in, const std::set<std::string>& only = {}) {
  const amd::CodeObject o = object(target);
  Results out;
  const uint32_t n = static_cast<uint32_t>(in.a.size());
  for (const amd::Kernel& k : o.kernels) {
    if (!only.empty() && !only.count(k.name)) continue;
    MemoryManager mem(16ull << 20);
    const uint64_t pa = upload(mem, in.a), pb = upload(mem, in.b), pc = upload(mem, in.c);
    const uint64_t pout = mem.alloc(n * 8);
    std::vector<uint8_t> buf(k.kernarg_size, 0);
    const uint64_t args[4] = {pa, pb, pc, pout};
    std::memcpy(buf.data(), args, sizeof args);
    std::memcpy(buf.data() + 32, &n, 4);
    amd::Dispatch d;
    d.object = &o;
    d.kernel = &k;
    d.kernarg = mem.alloc(buf.size());
    mem.write(d.kernarg, buf.data(), buf.size());
    d.group_size[0] = 64;
    d.groups[0] = n / 64;
    d.wave_size = k.wavefront_size;
    std::vector<uint32_t> got(n * 2);
    try {
      amd::execute(d, mem);
      mem.read(pout, got.data(), got.size() * 4);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "%s %s: %s\n", target.c_str(), k.name.c_str(), e.what());
      got.assign(n * 2, 0xdeadbeefu);   // (refused: the census lists it as such)
    }
    out[k.name] = std::move(got);
  }
  return out;
}

}  // namespace

// ---- gfx1030's transcendentals on the RX 6800's recorded operands (tools/numerics/amd/gfx1030.vectors.txt: 77 selected operands, the
// card's result of each of v_rcp, v_rsq, v_sqrt, v_exp, v_log, v_sin and v_cos; gfx1030.trig64ths.txt: v_sin and v_cos of the multiples of
// 1/64 turn). Where the operand or the card's result is not an ordinary number (zero, denormal, infinity, NaN) the result must be the
// card's, bit for bit; where both are ordinary the simulator computes the function at host precision, which the card does not (it is
// off by up to one unit in the last place on 1 to 50 percent of operands, and by an absolute 2^-22 or so near the zeros of the sine and
// cosine; docs/numerics/math-functions.md), so one unit, or 4e-7 absolute for the sine and cosine, is allowed.
namespace {

struct Vec { uint32_t x, want; };

std::vector<Vec> column(const char* file, int col, int columns) {
  std::ifstream f(std::string(VGPU_SOURCE_DIR) + "/tools/numerics/amd/" + file);
  if (!f) throw vtest::Failure(std::string("no vectors file ") + file);
  std::vector<Vec> rows;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    unsigned v[8];
    const int n = std::sscanf(line.c_str(), "%x %x %x %x %x %x %x %x", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]);
    if (n < columns) continue;
    rows.push_back({v[0], v[col]});
  }
  return rows;
}

float as_f(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }

int check_column(const char* kernel, const std::vector<Vec>& rows, bool trig, const char* target = "gfx1030", bool exact = false) {
  Inputs in;
  for (size_t i = 0; i < (rows.size() + 63) / 64 * 64; ++i) {
    in.a.push_back(i < rows.size() ? rows[i].x : 0u);
    in.b.push_back(0);
    in.c.push_back(0);
  }
  const Results got = run_target(target, in, {kernel});
  const auto ordinary = [](uint32_t b) { const uint32_t e = b >> 23 & 0xff; return e != 0 && e != 255; };
  int wrong = 0;
  for (size_t i = 0; i < rows.size(); ++i) {
    const uint32_t g = got.at(kernel)[i], w = rows[i].want;
    bool ok = g == w;
    if (!ok && !exact && ordinary(rows[i].x) && ordinary(w) && ordinary(g) && (g >> 31) == (w >> 31)) ok = (g > w ? g - w : w - g) <= 1;
    if (!ok && trig && ordinary(g) && ordinary(w)) ok = std::fabs(as_f(g) - as_f(w)) <= 4e-7f;
    if (!ok && wrong++ < 12) std::fprintf(stderr, "  %s %s(%08x): %08x, the card gave %08x\n", target, kernel, rows[i].x, g, w);
  }
  return wrong;
}

}  // namespace

VTEST(gfx1030_transcendentals_on_the_cards_special_operands) {
  const char* const names[7] = {"k_rcp", "k_rsq", "k_sqrt", "k_exp", "k_log", "k_sin", "k_cos"};
  int wrong = 0;
  // (v_rcp_f32, v_rsq_f32 and v_sqrt_f32 are the card's bits on every operand, rdna2_trans.cpp, test_amd_rdna2_trans; the other four are the host's
  // result, within one unit)
  for (int op = 0; op < 7; ++op) wrong += check_column(names[op], column("gfx1030.vectors.txt", op + 1, 8), op >= 5, "gfx1030", op < 3);
  VCHECK_EQ(wrong, 0);
}

VTEST(gfx1030_sine_and_cosine_of_the_multiples_of_a_sixty_fourth_of_a_turn) {
  int wrong = check_column("k_sin", column("gfx1030.trig64ths.txt", 1, 3), true);
  wrong += check_column("k_cos", column("gfx1030.trig64ths.txt", 2, 3), true);
  VCHECK_EQ(wrong, 0);
}

// ---- the same rules on every target (issues #721, #722: measured on gfx1030, inferred for the others). A NaN operand's result is each
// target's own (issue #727), so the rows whose operand is a NaN are left out for the targets but gfx1030.
namespace {

bool is_nan_bits(uint32_t b) { return (b & 0x7f800000u) == 0x7f800000u && (b & 0x7fffffu) != 0; }

std::vector<Vec> without_nans(std::vector<Vec> rows) {
  std::vector<Vec> out;
  for (const Vec& v : rows)
    if (!is_nan_bits(v.x) && !is_nan_bits(v.want)) out.push_back(v);
  return out;
}

}  // namespace

VTEST(every_target_flushes_denormals_and_counts_turns_as_the_rx6800_does) {
  const char* const names[7] = {"k_rcp", "k_rsq", "k_sqrt", "k_exp", "k_log", "k_sin", "k_cos"};
  int wrong = 0;
  for (const char* t : kTargets) {
    if (std::string(t) == "gfx1030") continue;   // (the two tests above)
    for (int op = 0; op < 7; ++op)
      wrong += check_column(names[op], without_nans(column("gfx1030.vectors.txt", op + 1, 8)), op >= 5, t);
    wrong += check_column("k_sin", without_nans(column("gfx1030.trig64ths.txt", 1, 3)), true, t);
    wrong += check_column("k_cos", without_nans(column("gfx1030.trig64ths.txt", 2, 3)), true, t);
  }
  VCHECK_EQ(wrong, 0);
}

// ---- running one kernel on chosen operands

namespace {

using Case = std::array<uint32_t, 3>;

bool two_words(const std::string& kernel) { return kernel.rfind("k_d_", 0) == 0 || kernel.rfind("k_w_", 0) == 0; }

bool has_kernel(const std::string& target, const char* kernel) { return amd::find_kernel(object(target), kernel) != nullptr; }

// The result of each case: the output word, or the pair of words (the high one above) for the double and wave kernels.
std::vector<uint64_t> run_cases(const std::string& target, const char* kernel, const std::vector<Case>& cases) {
  Inputs in;
  for (size_t i = 0; i < (cases.size() + 63) / 64 * 64; ++i) {
    const Case& c = cases[i < cases.size() ? i : cases.size() - 1];
    in.a.push_back(c[0]);
    in.b.push_back(c[1]);
    in.c.push_back(c[2]);
  }
  const Results r = run_target(target, in, {kernel});
  const auto it = r.find(kernel);
  if (it == r.end()) throw vtest::Failure(target + " has no kernel " + kernel);
  std::vector<uint64_t> out;
  for (size_t i = 0; i < cases.size(); ++i)
    out.push_back(two_words(kernel) ? (static_cast<uint64_t>(it->second[2 * i + 1]) << 32 | it->second[2 * i]) : it->second[i]);
  return out;
}

uint64_t run_one(const std::string& target, const char* kernel, uint32_t a, uint32_t b = 0, uint32_t c = 0) {
  return run_cases(target, kernel, {{a, b, c}})[0];
}

bool wave64(const std::string& target) { return target.rfind("gfx9", 0) == 0; }

std::string fail_text(const std::string& target, const char* kernel, const char* what, uint64_t got, uint64_t want) {
  char buf[200];
  std::snprintf(buf, sizeof buf, "%s %s %s: %llx, expected %llx", target.c_str(), kernel, what, static_cast<unsigned long long>(got),
                static_cast<unsigned long long>(want));
  return buf;
}

}  // namespace

// ---- v_rcp_f64 and v_rsq_f64 on the RX 6800's recorded operands (issue #728; tools/numerics/amd/gfx1030.vectors64.txt: 327 selected operands,
// gfx1030.sample64.txt: 1024 pseudo-random ones; the RX 6700 XT gave the same results). Where the operand or the card's result is not an ordinary
// number (zero, infinity, NaN) the result must be the card's, bit for bit. Where both are ordinary (a denormal operand or result too: 2^-1024 comes
// back from the card as 2^-1074 away from the exact one) the card's result has 22 to 26
// significant bits (its low bits are a table's; measured over 2^27 operands, docs/numerics/math-functions.md) and the simulator's is the correctly
// rounded double on every target, so the two must agree to 2^-22 relative: what is pinned is the size of the difference, not its bits (a model of
// them is not known, issue #728 stays open).
namespace {

struct Vec64 { uint64_t x, rcp, rsq, sqrt; };

std::vector<Vec64> vectors64(const char* file) {
  std::ifstream f(std::string(VGPU_SOURCE_DIR) + "/tools/numerics/amd/" + file);
  if (!f) throw vtest::Failure(std::string("no vectors file ") + file);
  std::vector<Vec64> rows;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    unsigned long long v[4];
    if (std::sscanf(line.c_str(), "%llx %llx %llx %llx", &v[0], &v[1], &v[2], &v[3]) == 4) rows.push_back({v[0], v[1], v[2], v[3]});
  }
  return rows;
}

bool ordinary64(uint64_t b) { return (b & 0x7fffffffffffffffull) != 0 && (b >> 52 & 0x7ff) != 0x7ff; }   // (finite, not zero: a denormal counts)
double as_d(uint64_t b) { double d; std::memcpy(&d, &b, 8); return d; }

int check64(const char* target, const char* kernel, const std::vector<Vec64>& rows, uint64_t Vec64::*column) {
  std::vector<Case> cases;
  for (const Vec64& r : rows) cases.push_back({static_cast<uint32_t>(r.x), static_cast<uint32_t>(r.x >> 32), 0});
  const auto got = run_cases(target, kernel, cases);
  int wrong = 0;
  for (size_t i = 0; i < rows.size(); ++i) {
    const uint64_t w = rows[i].*column;
    bool ok = got[i] == w;
    if (!ok && ordinary64(rows[i].x) && ordinary64(w) && ordinary64(got[i]) && (got[i] >> 63) == (w >> 63))
      ok = std::fabs(as_d(got[i]) - as_d(w)) <= std::ldexp(std::fabs(as_d(w)), -22);
    if (!ok && wrong++ < 8) std::fprintf(stderr, "  %s %s(%016llx): %016llx, the card gave %016llx\n", target, kernel, static_cast<unsigned long long>(rows[i].x), static_cast<unsigned long long>(got[i]), static_cast<unsigned long long>(w));
  }
  return wrong;
}

}  // namespace

VTEST(gfx1030_double_reciprocal_and_reciprocal_square_root_on_the_cards_operands) {
  int wrong = 0;
  for (const char* file : {"gfx1030.vectors64.txt", "gfx1030.sample64.txt"}) {
    const auto rows = vectors64(file);
    VCHECK(rows.size() >= 300);
    wrong += check64("gfx1030", "k_d_rcp", rows, &Vec64::rcp);
    wrong += check64("gfx1030", "k_d_rsq", rows, &Vec64::rsq);
  }
  VCHECK_EQ(wrong, 0);
}

// ---- issue #724: the minimum and maximum. Every target orders -0 below +0; the result of v_max_f32(2, a NaN) is the 2; RDNA4's and
// gfx1250's "_num" and IEEE forms (RDNA4 ISA, V_MAX_NUM_F32: "IEEE maximumNumber()", V_MAXIMUM_F32: "IEEE maximum()") are one set of
// rules, and gfx1201 gives gfx1250's results on every operand pair.
VTEST(minimum_and_maximum_order_the_zeros_and_pass_a_number_over_a_nan_on_every_target) {
  for (const char* t : kTargets) {
    VCHECK_EQ(run_one(t, "k_max", 0x00000000, 0x80000000), uint64_t{0x00000000});
    VCHECK_EQ(run_one(t, "k_max", 0x80000000, 0x00000000), uint64_t{0x00000000});
    VCHECK_EQ(run_one(t, "k_min", 0x00000000, 0x80000000), uint64_t{0x80000000});
    VCHECK_EQ(run_one(t, "k_min", 0x80000000, 0x00000000), uint64_t{0x80000000});
    VCHECK_EQ(run_one(t, "k_max", 0x40000000, 0x7fc00000), uint64_t{0x40000000});
    VCHECK_EQ(run_one(t, "k_min", 0x7fc00000, 0x40000000), uint64_t{0x40000000});
  }
}

VTEST(rdna4_runs_the_num_and_ieee_minimum_and_maximum_as_gfx1250_does) {
  const Inputs in = inputs();
  for (const char* k : {"k_max", "k_min", "k_med3", "k_max_num", "k_min_num", "k_max3_num", "k_min3_num", "k_med3_num", "k_maximum", "k_minimum"}) {
    const Results a = run_target("gfx1201", in, {k}), b = run_target("gfx1250", in, {k});
    int different = 0;
    for (size_t i = 0; i < in.a.size(); ++i) {
      const uint32_t x = a.at(k)[i], y = b.at(k)[i];
      if (x != y && !(is_nan_bits(x) && is_nan_bits(y))) ++different;   // (a NaN's payload is the NaN issue's)
    }
    if (different) throw vtest::Failure(std::string(k) + ": gfx1201 differs from gfx1250 on " + std::to_string(different) + " operand pairs");
  }
  for (const char* t : {"gfx1201", "gfx1250"}) {
    VCHECK_EQ(run_one(t, "k_max_num", 0x00000000, 0x80000000), uint64_t{0x00000000});   // maximumNumber(+0, -0) is +0
    VCHECK_EQ(run_one(t, "k_min_num", 0x00000000, 0x80000000), uint64_t{0x80000000});
    VCHECK_EQ(run_one(t, "k_max_num", 0x40000000, 0x7fa00001), uint64_t{0x40000000});   // a number beats a signalling NaN
    VCHECK_EQ(run_one(t, "k_maximum", 0x40000000, 0x7fc00000), uint64_t{0x7fc00000});   // maximum() carries a NaN
    VCHECK_EQ(run_one(t, "k_minimum", 0x40000000, 0x7fa00001), uint64_t{0x7fe00001});   // ... quieted
    VCHECK_EQ(run_one(t, "k_maximum", 0x00000000, 0x80000000), uint64_t{0x00000000});
    VCHECK_EQ(run_one(t, "k_minimum", 0x00000000, 0x80000000), uint64_t{0x80000000});
    VCHECK_EQ(run_one(t, "k_med3_num", 0x40000000, 0x3f800000, 0x40400000), uint64_t{0x40000000});   // median of 2, 1, 3
  }
}

// ---- issue #723: instructions the ISA documents on every RDNA target and on CDNA. The cube-map instructions give one answer on all
// targets; v_sin_f16 and v_cos_f16 give one on all targets but for the payload of a NaN.
VTEST(the_cube_map_instructions_run_on_every_target_with_one_answer) {
  const Inputs in = inputs();
  const Results ref = run_target("gfx942", in, {"k_cubeid", "k_cubesc", "k_cubetc", "k_cubema"});
  for (const char* t : kTargets)
    for (const char* k : {"k_cubeid", "k_cubesc", "k_cubetc", "k_cubema"}) {
      const Results got = run_target(t, in, {k});
      VCHECK(got.at(k) == ref.at(k));
    }
  const auto f = [](float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; };
  // The face a direction points at: +x 0, -x 1, +y 2, -y 3, +z 4, -z 5 (the ISA's table).
  for (const char* t : kTargets) {
    VCHECK_EQ(run_one(t, "k_cubeid", f(1), f(0), f(0)), uint64_t{f(0)});
    VCHECK_EQ(run_one(t, "k_cubeid", f(-1), f(0), f(0)), uint64_t{f(1)});
    VCHECK_EQ(run_one(t, "k_cubeid", f(0), f(1), f(0)), uint64_t{f(2)});
    VCHECK_EQ(run_one(t, "k_cubeid", f(0), f(-1), f(0)), uint64_t{f(3)});
    VCHECK_EQ(run_one(t, "k_cubeid", f(0), f(0), f(1)), uint64_t{f(4)});
    VCHECK_EQ(run_one(t, "k_cubeid", f(0), f(0), f(-1)), uint64_t{f(5)});
    VCHECK_EQ(run_one(t, "k_cubema", f(0), f(0), f(3)), uint64_t{f(6)});   // twice the major axis
  }
}

VTEST(half_precision_sine_and_cosine_run_on_every_target_with_one_answer) {
  const Inputs in = inputs();
  for (const char* k : {"k_h_sin", "k_h_cos"}) {
    const Results ref = run_target("gfx1100", in, {k});
    for (const char* t : kTargets) {
      const Results got = run_target(t, in, {k});
      int different = 0;
      for (size_t i = 0; i < in.a.size(); ++i) {
        const uint32_t x = ref.at(k)[i] & 0xffff, y = got.at(k)[i] & 0xffff;
        const auto nan16 = [](uint32_t h) { return (h & 0x7c00) == 0x7c00 && (h & 0x3ff) != 0; };
        if (x != y && !(nan16(x) && nan16(y))) ++different;
      }
      if (different) throw vtest::Failure(std::string(t) + " " + k + " differs from gfx1100 on " + std::to_string(different) + " operands");
    }
  }
}

// ---- issue #726: the real differences, each with the value an architecture must give.
// 8-bit floats: gfx942's are the FNUZ forms (CDNA3 ISA: bias 8, no infinity, 0x80 the NaN); gfx950's, RDNA4's and gfx1250's the OCP ones
// (bias 7 for e4m3, 0x7f/0xff the NaN; e5m2 has infinities). The byte 0x40 is 1.0 in e4m3 FNUZ and 2.0 in e4m3 OCP, 1.0 and 2.0 for e5m2
// (bias 16 against 15), and 0xff is -240 and a NaN.
VTEST(eight_bit_floats_are_fnuz_on_gfx942_and_ocp_on_gfx950_and_rdna4_and_gfx1250) {
  const auto f = [](float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; };
  for (const char* t : {"gfx942", "gfx950", "gfx1201", "gfx1250"}) {
    const bool fnuz = std::string(t) == "gfx942";
    VCHECK_EQ(run_one(t, "k_fp8_to_f32", 0x40), uint64_t{f(fnuz ? 1.0f : 2.0f)});
    VCHECK_EQ(run_one(t, "k_fp8_to_f32", 0x45), uint64_t{f(fnuz ? 1.625f : 3.25f)});
    VCHECK_EQ(run_one(t, "k_bf8_to_f32", 0x40), uint64_t{f(fnuz ? 1.0f : 2.0f)});
    VCHECK_EQ(run_one(t, "k_bf8_to_f32", 0xca), uint64_t{f(fnuz ? -6.0f : -12.0f)});
    const uint64_t ff = run_one(t, "k_fp8_to_f32", 0xff);
    if (fnuz) VCHECK_EQ(ff, uint64_t{f(-240.0f)});
    else VCHECK(is_nan_bits(static_cast<uint32_t>(ff)));
    // and the other way: 1.0 is the byte 0x40 (FNUZ) or 0x38 (OCP)
    VCHECK_EQ(run_one(t, "k_f32_to_fp8", f(1.0f), f(0.0f)), uint64_t{fnuz ? 0x0040u : 0x0038u});
  }
  for (const char* t : {"gfx90a", "gfx1030", "gfx1100"}) VCHECK(!has_kernel(t, "k_fp8_to_f32"));   // (no such instruction)
}

// v_dot2_f32_f16: gfx942 and gfx950 flush a denormal addend and denormal halves (CDNA3 and CDNA4 guides, 6.4); gfx90a does not.
VTEST(dot2_flushes_denormals_on_gfx942_and_gfx950_only) {
  for (const char* t : {"gfx90a", "gfx942", "gfx950"}) {
    const bool flush = std::string(t) != "gfx90a";
    VCHECK_EQ(run_one(t, "k_dot2_f32_f16", 0x3f800000, 0x00000000, 0x80000001), uint64_t{flush ? 0x00000000u : 0x80000001u});
    VCHECK_EQ(run_one(t, "k_dot2_f32_f16", 0x3f800000, 0x00000000, 0x00400000), uint64_t{flush ? 0x00000000u : 0x00400000u});
  }
}

// Lane-wide instructions follow the wave's width (64 on CDNA, 32 on RDNA and gfx1250).
VTEST(wave_wide_instructions_follow_the_width_of_the_wave_on_every_target) {
  std::vector<Case> cases;
  for (uint32_t i = 0; i < 256; ++i) {
    const uint32_t a = i * 2654435761u ^ 0x9e3779b9u, b = (i * 40503u + 7u) ^ (i << 3);
    cases.push_back({a, b, 0});
  }
  for (const char* t : kTargets) {
    const uint32_t W = wave64(t) ? 64 : 32;
    int wrong = 0;
    const auto check = [&](const char* kernel, const auto& expect) {
      const std::vector<uint64_t> got = run_cases(t, kernel, cases);
      for (uint32_t i = 0; i < cases.size(); ++i) {
        const uint32_t base = i / W * W, lane = i - base;
        uint32_t A[64], B[64];
        for (uint32_t l = 0; l < W; ++l) A[l] = cases[base + l][0], B[l] = cases[base + l][1];
        const uint64_t want = expect(A, B, lane);
        const uint64_t g = got[i];
        if ((std::string(kernel).find("ballot") != std::string::npos || std::string(kernel) == "k_w_size" ? g : (g & 0xffffffffu)) != want &&
            wrong++ < 6)
          std::fprintf(stderr, "  %s\n", fail_text(t, kernel, "lane", g, want).c_str());
      }
    };
    check("k_w_size", [&](const uint32_t*, const uint32_t*, uint32_t) { return uint64_t{W}; });
    check("k_w_ballot", [&](const uint32_t* A, const uint32_t*, uint32_t) {
      uint64_t m = 0;
      for (uint32_t l = 0; l < W; ++l) m |= uint64_t{(A[l] & 3) != 0} << l;
      return m;
    });
    check("k_w_ballot_true", [&](const uint32_t*, const uint32_t*, uint32_t) { return W == 64 ? ~uint64_t{0} : (uint64_t{1} << W) - 1; });
    check("k_w_readfirstlane", [&](const uint32_t* A, const uint32_t*, uint32_t) { return uint64_t{A[0]}; });
    check("k_w_readlane", [&](const uint32_t* A, const uint32_t* B, uint32_t) { return uint64_t{A[(B[0] & 63) % W]}; });
    check("k_w_readlane31", [&](const uint32_t* A, const uint32_t*, uint32_t) { return uint64_t{A[31]}; });
    check("k_w_mbcnt", [&](const uint32_t*, const uint32_t*, uint32_t lane) { return uint64_t{lane}; });
    check("k_w_mbcnt_lo_only", [&](const uint32_t*, const uint32_t*, uint32_t lane) { return uint64_t{lane < 32 ? lane : 32}; });
    check("k_w_mbcnt_hi_only", [&](const uint32_t*, const uint32_t*, uint32_t lane) { return uint64_t{lane > 32 ? lane - 32 : 0}; });
    check("k_w_bpermute", [&](const uint32_t* A, const uint32_t* B, uint32_t lane) { return uint64_t{A[B[lane] & (W - 1)]}; });
    check("k_w_dpp_quad_rev", [&](const uint32_t* A, const uint32_t*, uint32_t lane) { return uint64_t{A[lane / 4 * 4 + 3 - lane % 4]}; });
    check("k_w_dpp_row_mirror", [&](const uint32_t* A, const uint32_t*, uint32_t lane) { return uint64_t{A[lane / 16 * 16 + 15 - lane % 16]}; });
    check("k_w_dpp_row_half_mirror", [&](const uint32_t* A, const uint32_t*, uint32_t lane) {
      const uint32_t q = lane % 16;
      return uint64_t{A[lane / 16 * 16 + q / 8 * 8 + 7 - q % 8]};
    });
    // (row_shr:1 leaves a row's first lane at the old value, 0x55555555, or at zero with bound_ctrl)
    check("k_w_dpp_row_shr1", [&](const uint32_t* A, const uint32_t*, uint32_t lane) { return uint64_t{lane % 16 ? A[lane - 1] : 0x55555555u}; });
    check("k_w_dpp_row_shr1_bc", [&](const uint32_t* A, const uint32_t*, uint32_t lane) { return uint64_t{lane % 16 ? A[lane - 1] : 0u}; });
    if (wave64(t)) {   // (the CDNA-only controls: wave_shr:1, row_bcast:15 and row_bcast:31)
      check("k_w_dpp_wave_shr1", [&](const uint32_t* A, const uint32_t*, uint32_t lane) { return uint64_t{lane ? A[lane - 1] : 0x55555555u}; });
      check("k_w_dpp_row_bcast15", [&](const uint32_t* A, const uint32_t*, uint32_t lane) {   // rows 1 and 3 take the last lane of the row before
        const uint32_t row = lane / 16;
        return uint64_t{(row == 1 || row == 3) ? A[row * 16 - 1] : 0x55555555u};
      });
      check("k_w_dpp_row_bcast31", [&](const uint32_t* A, const uint32_t*, uint32_t lane) { return uint64_t{lane >= 32 ? A[31] : 0x55555555u}; });
    } else {
      VCHECK(!has_kernel(t, "k_w_dpp_wave_shr1"));
      check("k_w_permlane16", [&](const uint32_t* A, const uint32_t*, uint32_t lane) { return uint64_t{A[lane]}; });
      check("k_w_permlanex16", [&](const uint32_t* A, const uint32_t*, uint32_t lane) { return uint64_t{A[lane ^ 16]}; });
    }
    if (wrong) throw vtest::Failure(std::string(t) + ": " + std::to_string(wrong) + " lanes wrong");
  }
}

// ---- issue #730: no card says what any AMD matrix core sums; the simulator's default is one rule for gfx90a, gfx942 and gfx950. This
// records that, so changing one target's rule (when a card says) changes this test with it.
VTEST(the_default_matrix_rule_is_one_unmeasured_rule_for_the_cdna_targets) {
  const Inputs in = inputs();
  const Results a = run_target("gfx90a", in, {"k_mfma_f16"}), b = run_target("gfx942", in, {"k_mfma_f16"}), c = run_target("gfx950", in, {"k_mfma_f16"});
  VCHECK(a.at("k_mfma_f16") == b.at("k_mfma_f16"));
  VCHECK(a.at("k_mfma_f16") == c.at("k_mfma_f16"));
}

VTEST(dump_the_census) {
  const char* dir = std::getenv("VGPU_XARCH_DUMP");
  if (!dir || !*dir) return;
  const Inputs in = inputs();
  for (const char* t : kTargets) {
    const Results r = run_target(t, in);
    std::ofstream f(std::string(dir) + "/" + t + ".txt");
    for (const auto& [name, words] : r)
      for (size_t i = 0; i < in.a.size(); ++i) {
        char line[200];
        const bool dbl = name.rfind("k_d_", 0) == 0 || name.rfind("k_w_", 0) == 0;
        std::snprintf(line, sizeof line, "%s %zu %08x %08x %08x %08x %08x\n", name.c_str(), i, in.a[i], in.b[i], in.c[i],
                      dbl ? words[2 * i] : words[i], dbl ? words[2 * i + 1] : 0u);
        f << line;
      }
  }
}

VTEST_MAIN
