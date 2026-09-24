// What the pantheon workloads' kernels compile to under ROCm 7, run and
// checked against the same computation in C.
//
// The kernels are amd/tests/hipcc/ops.hip, compiled by hipcc 7.1 for gfx942:
// the compiler the workloads are built with emits instructions the clang the
// other fixtures use does not, and these kernels are written in the idioms
// the workloads use so that it emits them. The float inputs are chosen so
// every result is exact however the compiler fuses the arithmetic, which is
// what lets the answers be compared to the last bit.
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

amd::CodeObject object() {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/hipcc/ops.gfx942.o";
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

template <typename T>
std::vector<T> download(MemoryManager& mem, uint64_t p, size_t n) {
  std::vector<T> out(n);
  mem.read(p, out.data(), n * sizeof(T));
  return out;
}

uint64_t kernargs(MemoryManager& mem, const amd::Kernel& k, const std::vector<uint64_t>& values) {
  std::vector<uint8_t> buf(k.kernarg_size, 0);
  for (size_t i = 0; i < values.size() && i < k.args.size(); ++i) {
    const amd::KernelArg& a = k.args[i];
    for (uint32_t b = 0; b < a.size && b < 8; ++b) buf[a.offset + b] = static_cast<uint8_t>(values[i] >> (8 * b));
  }
  const uint64_t p = mem.alloc(buf.empty() ? 1 : buf.size());
  if (!buf.empty()) mem.write(p, buf.data(), buf.size());
  return p;
}

void run(const amd::CodeObject& o, const char* name, MemoryManager& mem, const std::vector<uint64_t>& args) {
  const amd::Kernel* k = amd::find_kernel(o, name);
  if (!k) throw vtest::Failure(std::string("no kernel named ") + name);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = kernargs(mem, *k, args);
  d.groups[0] = 1;
  d.group_size[0] = 64;
  amd::execute(d, mem);
}

const int kN = 64;

uint64_t bits(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  return u;
}

uint16_t half_bits(float f) {
  const _Float16 h = static_cast<_Float16>(f);
  uint16_t u = 0;
  std::memcpy(&u, &h, 2);
  return u;
}

}  // namespace

VTEST(a_wave_adds_one_value_for_all_its_active_lanes) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int32_t> in(kN);
  for (int i = 0; i < kN; ++i) in[i] = (i % 3 == 0) ? -i : i;   // some lanes take part, some do not
  const std::vector<uint32_t> zero = {0};
  const uint64_t pc = upload(mem, zero), pin = upload(mem, in);
  run(o, "_Z12count_activePjPKi", mem, {pc, pin});
  int want = 0;
  for (int i = 0; i < kN; ++i) want += in[i] > 0;
  VCHECK_EQ(download<uint32_t>(mem, pc, 1)[0], static_cast<uint32_t>(want));
}

VTEST(sixty_four_bit_scalar_arithmetic_and_a_vector_shift) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const uint64_t a = 0x123456789ABCDEF0ull, b = 0x0FEDCBA987654321ull;
  std::vector<uint64_t> v(kN);
  for (int i = 0; i < kN; ++i) v[i] = 0xF0E1D2C3B4A59687ull * static_cast<uint64_t>(i + 1);
  const uint64_t pout = mem.alloc(kN * 8), pv = upload(mem, v);
  run(o, "_Z4wideyyPyPKy", mem, {a, b, pout, pv});
  const std::vector<uint64_t> out = download<uint64_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) VCHECK_EQ(out[i], (a - b) + (v[i] >> (i & 63)) + (a >> 8));
}

VTEST(bytes_to_floats_a_reciprocal_square_root_and_24_bit_multiplies) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<uint32_t> a(kN);
  std::vector<float> f(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = 0x01020304u * static_cast<uint32_t>(i + 3) ^ 0x5A5A5A5Au;
    f[i] = std::ldexp(1.0f, 2 * (i % 5));   // powers of four: every root is exact
  }
  const uint64_t pa = upload(mem, a), pf = upload(mem, f), pout = mem.alloc(kN * 4), po2 = mem.alloc(kN * 4);
  run(o, "_Z4miscPKjPKfPfPj", mem, {pa, pf, pout, po2});
  const std::vector<float> out = download<float>(mem, pout, kN);
  const std::vector<uint32_t> o2 = download<uint32_t>(mem, po2, kN);
  for (int i = 0; i < kN; ++i) {
    const float want = static_cast<float>(a[i] & 0xff) + 1.0f / std::sqrt(f[i]) + f[i] * f[(i + 1) & 63] + 0.001f;
    if (std::fabs(out[i] - want) > std::fabs(want) * 1e-6f)
      throw vtest::Failure("misc[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
    const uint32_t mul24 = (a[i] & 0xFFFFFFu) * (a[(i + 1) & 63] & 0xFFFFFFu);
    VCHECK_EQ(o2[i], ~a[i] + mul24 + ((a[i] + a[(i + 2) & 63]) << 3) + ((a[i] & 0xf0f0f0f0u) | 32u));
  }
}

VTEST(a_64_bit_compare_and_swap_and_three_words_at_a_time) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<uint64_t> p(kN);
  std::vector<float> in(kN * 3);
  for (int i = 0; i < kN; ++i) p[i] = 0x100000000ull * static_cast<uint64_t>(i) + 7;
  for (int i = 0; i < kN * 3; ++i) in[i] = static_cast<float>(i) * 0.5f;
  const uint64_t pp = upload(mem, p), pin = upload(mem, in), pout = mem.alloc(kN * 12);
  run(o, "_Z5cas64PyPK15HIP_vector_typeIfLj3EEPS1_", mem, {pp, pin, pout});
  const std::vector<uint64_t> after = download<uint64_t>(mem, pp, kN);
  const std::vector<float> out = download<float>(mem, pout, kN * 3);
  for (int i = 0; i < kN; ++i) VCHECK_EQ(after[i], p[i] * 3 + 1);
  for (int i = 0; i < kN * 3; ++i) VCHECK_EQ(out[i], in[i] * 2.0f);
}

VTEST(halves_built_from_floats) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<float> a(kN), b(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<float>(i) * 0.75f - 20.0f;
    b[i] = static_cast<float>(i) * -1.25f + 3.0f;
  }
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), ps = mem.alloc(kN * 4), pk = mem.alloc(kN * 4),
                 ph = mem.alloc(kN * 8);
  run(o, "_Z5splitPKfP7__half2", mem, {pa, ps});
  run(o, "_Z4packPKfS0_P7__half2", mem, {pa, pb, pk});
  run(o, "_Z6halvesPKfS0_P7__half2", mem, {pa, pb, ph});
  const std::vector<uint32_t> split = download<uint32_t>(mem, ps, kN), pack = download<uint32_t>(mem, pk, kN),
                              hv = download<uint32_t>(mem, ph, kN * 2);
  for (int i = 0; i < kN; ++i) {
    const uint32_t h = half_bits(a[i] * 0.5f);
    VCHECK_EQ(split[i], h | h << 16);
    VCHECK_EQ(pack[i], static_cast<uint32_t>(half_bits(a[i])) | static_cast<uint32_t>(half_bits(b[i])) << 16);
    VCHECK_EQ(hv[i], static_cast<uint32_t>(half_bits(a[i] * 0.5f)) | static_cast<uint32_t>(half_bits(b[i] * 0.25f)) << 16);
    VCHECK_EQ(hv[kN + i], static_cast<uint32_t>(half_bits(a[i])) | static_cast<uint32_t>(half_bits(b[i] + 1.0f)) << 16);
  }
}

VTEST(the_comparisons_a_nan_answers_yes_to) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<float> a(kN), b(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<float>(i % 7) - 3.0f;
    b[i] = static_cast<float>(i % 5) - 2.0f;
  }
  a[5] = std::nanf("");
  b[17] = std::nanf("");
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(kN * 4);
  run(o, "_Z7nan_cmpPKfS0_Pi", mem, {pa, pb, pout});
  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i)
    VCHECK_EQ(out[i], !(a[i] > b[i]) + 2 * !(a[i] < b[i]) + 4 * (a[i] != b[i]) + 8 * (a[i] == b[i]));
}

VTEST(a_fence_between_two_stores) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const uint64_t pp = mem.alloc(128 * 4);
  run(o, "_Z6fencedPi", mem, {pp});
  const std::vector<int32_t> out = download<int32_t>(mem, pp, 128);
  for (int i = 0; i < kN; ++i) {
    VCHECK_EQ(out[i], i);
    VCHECK_EQ(out[i + 64], i);
  }
}

VTEST(scalar_comparisons_and_a_constant_of_the_instructions_own) {
  const amd::CodeObject o = object();
  std::vector<int32_t> a(kN);
  for (int i = 0; i < kN; ++i) a[i] = i % 9;
  for (const int k : {0, 3, 5, 8, 10, 500, -4}) {
    MemoryManager mem(64ull << 20);
    const uint64_t pa = upload(mem, a), pout = mem.alloc(kN * 4);
    run(o, "_Z8branchesPKiPii", mem, {pa, pout, static_cast<uint64_t>(static_cast<uint32_t>(k))});
    const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
    for (int i = 0; i < kN; ++i) {
      int r = 0;
      if (k > 7) r += 1;
      if (k >= 3) r += 2;
      if (k != 5) r += 4;
      if (k == 500) r += 8;
      if (static_cast<unsigned>(k) > 9u) r += 16;
      VCHECK_EQ(out[i], r + (a[i] < 4 ? 32 : 0));
    }
  }
}

VTEST(the_rest_of_the_idioms) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<uint32_t> a(kN), b(kN);
  std::vector<float> f(kN), g(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = 0x9E3779B9u * static_cast<uint32_t>(i + 1);
    b[i] = 0x85EBCA6Bu ^ static_cast<uint32_t>(i * 131);
    f[i] = static_cast<float>(i) * 0.5f - 16.0f;
    g[i] = static_cast<float>(i % 7) - 3.0f;
  }
  const uint32_t k = 0x0FF00FF0u;
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pf = upload(mem, f), pg = upload(mem, g),
                 po = mem.alloc(320 * 4), pof = mem.alloc(256 * 4);
  mem.fill(pof, reinterpret_cast<const uint8_t*>("\0\0\0\0"), 4, 256 * 4);
  for (const auto& [x, y] : {std::pair<uint64_t, uint64_t>{5, 5}, {5, 6}}) {
    run(o, "_Z4morePKjS0_PKfS2_PjPfjyy", mem, {pa, pb, pf, pg, po, pof, k, x, y});
    const std::vector<uint32_t> out = download<uint32_t>(mem, po, 320);
    const std::vector<float> of = download<float>(mem, pof, 256);
    for (int i = 0; i < kN; ++i) {
      VCHECK_EQ(out[i], ~a[i]);
      VCHECK_EQ(out[i + 64], (a[i] & 0xFFFFFFu) * (b[i] & 0xFFFFFFu) + b[(i + 1) & 63]);
      VCHECK_EQ(out[i + 128], 1u << (a[i] & 31));
      VCHECK_EQ(out[i + 192], (a[i] & k) | b[i]);
      VCHECK_EQ(out[i + 256], static_cast<uint32_t>(((static_cast<uint64_t>(k) << 32) | k) - a[i]));
      VCHECK_EQ(of[i], f[i] * g[i] + 0.123f);
      VCHECK_EQ(of[i + 64], std::fabs(f[i]) * g[i]);
      if (x == y) VCHECK_EQ(of[i + 128], 1.0f);
      if ((k << 2) >= static_cast<unsigned>(i)) VCHECK_EQ(of[i + 192], 2.0f);
    }
  }
}

VTEST(a_condition_every_lane_agrees_on_worked_out_by_the_vector_unit) {
  const amd::CodeObject o = object();
  for (const auto& [fa, fb] : {std::pair<float, float>{2.0f, 1.0f}, {1.0f, 4.0f}, {1.0f, 1.5f}, {3.0f, 3.0f}}) {
    MemoryManager mem(64ull << 20);
    const uint64_t pout = mem.alloc(kN * 4);
    run(o, "_Z13uniform_floatffPi", mem, {bits(fa), bits(fb), pout});
    const int want = fa > fb ? 1 : (fa < fb * 0.5f ? 2 : 0);
    const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
    for (int i = 0; i < kN; ++i) VCHECK_EQ(out[i], want);
  }
}

VTEST(work_in_both_arms_of_a_divergent_branch) {
  const amd::CodeObject o = object();
  std::vector<int32_t> a(kN);
  for (int i = 0; i < kN; ++i) a[i] = i * 37 - 1000;
  for (const auto& [k, m, fa, fb] : {std::tuple<int, int, float, float>{4, 3, 1.0f, 2.0f}, {1, 3, 5.0f, 2.0f}}) {
    MemoryManager mem(64ull << 20);
    const uint64_t base = 0x0000123400000000ull;
    const uint64_t pa = upload(mem, a), pout = mem.alloc(kN * 4 + kN * 8 + 256);
    run(o, "_Z6shapesPKiPiiiffy", mem,
        {pa, pout, static_cast<uint64_t>(static_cast<uint32_t>(k)), static_cast<uint64_t>(static_cast<uint32_t>(m)),
         bits(fa), bits(fb), base});
    const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
    const std::vector<uint64_t> wide = download<uint64_t>(mem, pout + 64 * 8, kN);
    for (int i = 0; i < kN; ++i) {
      int r = 0;
      if (k >= m) r += 1;
      if (!(fa >= fb)) r += 2;
      if (a[i] & 1) r += a[i] * 3;
      else r -= a[i] / 7;
      VCHECK_EQ(out[i], r);
      VCHECK_EQ(wide[i], base - static_cast<uint64_t>(static_cast<int64_t>(a[i])));
    }
  }
}

VTEST(a_loop_each_lane_leaves_at_its_own_time) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int32_t> next(kN);
  // Every entry is a lane or a negative end: some walks end, some reach 13,
  // and some go round until the step limit stops them.
  for (int i = 0; i < kN; ++i) next[i] = (i * 7 + 3) % 69 - 5;
  const uint64_t pn = upload(mem, next), pout = mem.alloc(kN * 4);
  run(o, "_Z4walkPKiPi", mem, {pn, pout});
  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  for (int t = 0; t < kN; ++t) {
    int at = t, steps = 0;
    while (at >= 0 && steps < 64) {
      at = next[at];
      ++steps;
      if (at == 13) break;
    }
    VCHECK_EQ(out[t], steps);
  }
}

VTEST_MAIN
