// The idioms a kernel writes out by hand, run and checked against answers
// worked out in C: a clamp, a rotate, a bitfield insert, frexp, the dot
// products, and two floats packed into halves rounded toward zero.
//
// The kernels are amd/tests/data/idioms.c, compiled for gfx942. The packing
// is the one worth naming: rounding toward zero is checked against the
// definition itself -- of every finite half, the largest in magnitude that is
// not past the float -- found by searching all of them, so the check shares
// nothing with how the model computes it.
#include <algorithm>
#include <cmath>
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

amd::CodeObject object() {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/idioms.gfx942.o";
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

}  // namespace

namespace {

// The half a float rounds to toward zero, by the definition: the finite half
// of the float's sign that is largest in magnitude without exceeding it.
uint16_t rtz_by_search(float x) {
  if (std::isnan(x)) return 0x7E00;
  uint16_t best = std::signbit(x) ? 0x8000 : 0x0000;
  double best_mag = 0;
  for (uint32_t m = 0; m < 0x7C00; ++m) {   // every finite magnitude
    _Float16 h;
    const uint16_t bits = static_cast<uint16_t>(m);
    std::memcpy(&h, &bits, 2);
    const double mag = static_cast<double>(h);
    if (mag <= std::fabs(static_cast<double>(x)) && mag >= best_mag) {
      best_mag = mag;
      best = static_cast<uint16_t>((std::signbit(x) ? 0x8000 : 0) | m);
    }
  }
  return best;
}

}  // namespace

VTEST(a_value_held_between_two_bounds) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int32_t> a(kN);
  for (int i = 0; i < kN; ++i) a[i] = i * 7 - 220;
  const uint64_t pa = upload(mem, a), pout = mem.alloc(kN * 4);
  run(o, "clampi", mem, {pa, pout, static_cast<uint64_t>(kN)});
  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) VCHECK_EQ(out[i], std::min(100, std::max(-100, a[i])));
}

VTEST(a_rotate_and_a_bitfield_insert) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<uint32_t> a(kN), b(kN), s(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = 0x9E3779B9u * static_cast<uint32_t>(i + 1);
    b[i] = 0x85EBCA6Bu ^ static_cast<uint32_t>(i * 977);
    s[i] = static_cast<uint32_t>(i);   // every amount from 0 to 31, twice
  }
  {
    const uint64_t pa = upload(mem, a), ps = upload(mem, s), pout = mem.alloc(kN * 4);
    run(o, "rotl", mem, {pa, ps, pout, static_cast<uint64_t>(kN)});
    const std::vector<uint32_t> out = download<uint32_t>(mem, pout, kN);
    for (int i = 0; i < kN; ++i) {
      const uint32_t k = s[i] & 31;
      VCHECK_EQ(out[i], k ? (a[i] << k) | (a[i] >> (32 - k)) : a[i]);
    }
  }
  {
    const uint32_t m = 0x0FF0F00Fu;
    const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(kN * 4);
    run(o, "insert", mem, {pa, pb, pout, m, static_cast<uint64_t>(kN)});
    const std::vector<uint32_t> out = download<uint32_t>(mem, pout, kN);
    for (int i = 0; i < kN; ++i) VCHECK_EQ(out[i], (a[i] & m) | (b[i] & ~m));
  }
}

VTEST(thirty_two_bits_out_of_two_registers) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<uint32_t> hi(kN), lo(kN), s(kN);
  for (int i = 0; i < kN; ++i) {
    hi[i] = 0xDEADBEEFu ^ static_cast<uint32_t>(i * 0x01010101);
    lo[i] = 0x12345678u + static_cast<uint32_t>(i * 0x11111);
    s[i] = static_cast<uint32_t>(i);
  }
  const uint64_t ph = upload(mem, hi), pl = upload(mem, lo), ps = upload(mem, s), pout = mem.alloc(kN * 4);
  run(o, "funnel", mem, {ph, pl, ps, pout, static_cast<uint64_t>(kN)});
  const std::vector<uint32_t> out = download<uint32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i)
    VCHECK_EQ(out[i], static_cast<uint32_t>(((static_cast<uint64_t>(hi[i]) << 32) | lo[i]) >> (s[i] & 31)));
}

VTEST(a_float_taken_apart) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<float> a(kN);
  for (int i = 0; i < kN; ++i) a[i] = (static_cast<float>(i) - 32.0f) * 37.125f;   // zero among them
  const uint64_t pa = upload(mem, a), pfr = mem.alloc(kN * 4), pex = mem.alloc(kN * 4), pm = mem.alloc(kN * 4);
  run(o, "parts", mem, {pa, pfr, pex, pm, static_cast<uint64_t>(kN)});
  const std::vector<float> fr = download<float>(mem, pfr, kN), mant = download<float>(mem, pm, kN);
  const std::vector<int32_t> ex = download<int32_t>(mem, pex, kN);
  for (int i = 0; i < kN; ++i) {
    int e = 0;
    const float m = std::frexp(a[i], &e);
    VCHECK_EQ(mant[i], m);
    VCHECK_EQ(ex[i], e);
    VCHECK_EQ(fr[i], a[i] - std::floor(a[i]));
  }
}

VTEST(the_dot_products) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<uint32_t> a(kN), b(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = 0x7F80017Fu ^ (static_cast<uint32_t>(i) * 0x01030507u);   // bytes of both signs, extremes among them
    b[i] = 0x80FF7F01u + static_cast<uint32_t>(i) * 0x02040608u;
  }
  // For the halves, small whole numbers and halves: every product and sum
  // is exact, so how the sum is rounded cannot matter to the answer.
  std::vector<uint32_t> ha(kN), hb(kN);
  const auto pack = [](float lo, float hi) {
    const _Float16 l = static_cast<_Float16>(lo), h = static_cast<_Float16>(hi);
    uint16_t bl, bh;
    std::memcpy(&bl, &l, 2);
    std::memcpy(&bh, &h, 2);
    return static_cast<uint32_t>(bl) | static_cast<uint32_t>(bh) << 16;
  };
  for (int i = 0; i < kN; ++i) {
    ha[i] = pack(static_cast<float>(i % 9) - 4.0f, static_cast<float>(i % 5) * 0.5f);
    hb[i] = pack(static_cast<float>(i % 7) * 0.25f, static_cast<float>(i % 11) - 5.0f);
  }
  {
    const uint64_t pa = upload(mem, a), pb = upload(mem, b), p4 = mem.alloc(kN * 4), p2 = mem.alloc(kN * 4);
    run(o, "dots", mem, {pa, pb, p4, p2, static_cast<uint64_t>(kN)});
    const std::vector<int32_t> o4 = download<int32_t>(mem, p4, kN);
    for (int i = 0; i < kN; ++i) {
      int32_t want = 7;
      for (int k = 0; k < 4; ++k)
        want += static_cast<int8_t>(a[i] >> (8 * k)) * static_cast<int8_t>(b[i] >> (8 * k));
      VCHECK_EQ(o4[i], want);
    }
  }
  {
    const uint64_t pa = upload(mem, ha), pb = upload(mem, hb), p4 = mem.alloc(kN * 4), p2 = mem.alloc(kN * 4);
    run(o, "dots", mem, {pa, pb, p4, p2, static_cast<uint64_t>(kN)});
    const std::vector<float> o2 = download<float>(mem, p2, kN);
    for (int i = 0; i < kN; ++i) {
      const auto half = [](uint32_t v, int k) {
        _Float16 h;
        const uint16_t bits = static_cast<uint16_t>(v >> (16 * k));
        std::memcpy(&h, &bits, 2);
        return static_cast<float>(h);
      };
      const float want = half(ha[i], 0) * half(hb[i], 0) + half(ha[i], 1) * half(hb[i], 1) + 1.0f;
      VCHECK_EQ(o2[i], want);
    }
  }
}

VTEST(two_floats_packed_into_halves_rounded_toward_zero) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<float> a(kN), b(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = (static_cast<float>(i) - 32.0f) * 1.0009765f;     // between halves, so the direction shows
    b[i] = static_cast<float>(i) * 1234.567f - 20000.0f;
  }
  a[0] = 70000.0f;      // past the largest half: toward zero is the largest half, not an infinity
  a[1] = -70000.0f;
  a[2] = 1e-7f;         // below the smallest half: zero, with its sign
  a[3] = -1e-7f;
  b[0] = 65504.0f;      // exactly the largest half
  b[1] = 0.33333334f;
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(kN * 4);
  run(o, "pkrtz", mem, {pa, pb, pout, static_cast<uint64_t>(kN)});
  const std::vector<uint32_t> out = download<uint32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const uint32_t want = rtz_by_search(a[i]) | static_cast<uint32_t>(rtz_by_search(b[i])) << 16;
    if (out[i] != want)
      throw vtest::Failure("pkrtz[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST_MAIN
