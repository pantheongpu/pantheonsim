// The rest of what a kernel does with numbers, run and checked against what
// the C means: doubles, packed half precision, the absolute value and min/max
// modifiers, the transcendentals, bit counting, and the atomics beyond an add.
//
// The kernels are amd/tests/data/math.c, compiled for gfx942. Where the
// arithmetic is exact -- doubles, halves, min and max, the bit counts -- the
// results must match exactly. The transcendentals are the one place this
// model differs from a card on purpose: the hardware's are tables good to
// about one unit in the last place, and these are the host's, so the test
// allows that much.
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
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/math.gfx942.o";
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

// One work-group of 64, which is what these kernels index by.
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

}  // namespace

VTEST(doubles_are_kept_in_register_pairs_and_come_out_exact) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const int n = 64;
  std::vector<double> a(n), b(n);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<double>(i) * 1.25 - 20.0;
    b[i] = static_cast<double>(i % 9) + 1.5;
  }
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(n * 8);
  run(o, "dbl_math", mem, {pa, pb, pout, static_cast<uint64_t>(n)});

  const std::vector<double> out = download<double>(mem, pout, n);
  for (int i = 0; i < n; ++i) {
    const double want = a[i] * b[i] + a[i] / b[i] + std::fma(a[i], b[i], 1.0);
    if (out[i] != want)
      throw vtest::Failure("dbl_math[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want) + " (double arithmetic is exact, so these must match)");
  }
}

VTEST(an_absolute_value_and_a_min_and_max_are_the_modifiers_the_compiler_uses) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const int n = 64;
  std::vector<float> a(n), b(n);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i) - 32.0f;      // both signs, so the absolute value shows
    b[i] = static_cast<float>(32 - i) * 0.5f;
  }
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(n * 4);
  run(o, "mods", mem, {pa, pb, pout, static_cast<uint64_t>(n)});

  const std::vector<float> out = download<float>(mem, pout, n);
  for (int i = 0; i < n; ++i) {
    const float want = std::fabs(a[i]) + std::fmin(a[i], b[i]) + std::fmax(-a[i], b[i]);
    if (out[i] != want)
      throw vtest::Failure("mods[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(two_halves_are_multiplied_and_added_in_one_register) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const int n = 64;   // each element is a pair of halves
  std::vector<uint32_t> a(n), b(n);
  const auto pack = [](float lo, float hi) {
    const _Float16 l = static_cast<_Float16>(lo), h = static_cast<_Float16>(hi);
    uint16_t bl, bh;
    std::memcpy(&bl, &l, 2);
    std::memcpy(&bh, &h, 2);
    return static_cast<uint32_t>(bl) | static_cast<uint32_t>(bh) << 16;
  };
  const auto unpack = [](uint32_t v, int which) {
    uint16_t bits = static_cast<uint16_t>(which ? v >> 16 : v);
    _Float16 h;
    std::memcpy(&h, &bits, 2);
    return static_cast<float>(h);
  };
  for (int i = 0; i < n; ++i) {
    a[i] = pack(static_cast<float>(i) * 0.5f, static_cast<float>(i) - 8.0f);
    b[i] = pack(static_cast<float>(i % 5) + 1.0f, 0.25f * static_cast<float>(i % 7));
  }
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(n * 4);
  run(o, "half_math", mem, {pa, pb, pout, static_cast<uint64_t>(n)});

  const std::vector<uint32_t> out = download<uint32_t>(mem, pout, n);
  for (int i = 0; i < n; ++i)
    for (int half = 0; half < 2; ++half) {
      const _Float16 x = static_cast<_Float16>(unpack(a[i], half)), y = static_cast<_Float16>(unpack(b[i], half));
      const float want = static_cast<float>(static_cast<_Float16>(static_cast<float>(x) * static_cast<float>(y) +
                                                                 static_cast<float>(x)));
      if (unpack(out[i], half) != want)
        throw vtest::Failure("half_math[" + std::to_string(i) + "] half " + std::to_string(half) + " is " +
                             std::to_string(unpack(out[i], half)) + ", not " + std::to_string(want));
    }
}

VTEST(the_transcendentals_are_the_hosts_rather_than_the_cards_tables) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const int n = 64;
  std::vector<float> a(n);
  for (int i = 0; i < n; ++i) a[i] = 0.5f + static_cast<float>(i) * 0.75f;   // positive: the C takes a log of it
  const uint64_t pa = upload(mem, a), pout = mem.alloc(n * 4);
  run(o, "transcendental", mem, {pa, pout, static_cast<uint64_t>(n)});

  const std::vector<float> out = download<float>(mem, pout, n);
  for (int i = 0; i < n; ++i) {
    const float want = std::sqrt(a[i]) + std::exp2(a[i]) + std::log2(a[i]);
    // A card's answer would differ by about a unit in the last place; this
    // model's is the host's, so it lands exactly, and the tolerance is only
    // there to say the model is a model.
    if (std::fabs(out[i] - want) > std::fabs(want) * 1e-6f)
      throw vtest::Failure("transcendental[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(counting_bits_and_taking_an_unsigned_maximum) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const int n = 64;
  std::vector<uint32_t> a(n);
  for (int i = 0; i < n; ++i) a[i] = static_cast<uint32_t>(i) * 2654435761u;
  const uint64_t pa = upload(mem, a), pout = mem.alloc(n * 4);
  run(o, "bits", mem, {pa, pout, static_cast<uint64_t>(n)});

  const std::vector<uint32_t> out = download<uint32_t>(mem, pout, n);
  for (int i = 0; i < n; ++i) {
    const uint32_t x = a[i];
    const uint32_t want = static_cast<uint32_t>(__builtin_popcount(x)) +
                          static_cast<uint32_t>(__builtin_clz(x | 1u)) + (x > 100u ? x : 100u);
    if (out[i] != want)
      throw vtest::Failure("bits[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(the_atomics_beyond_an_add_land_on_every_lane) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const int n = 64;
  std::vector<int> in(n);
  for (int i = 0; i < n; ++i) in[i] = (i * 7919) | 1;
  std::vector<int> start = {0, -1, 0};
  const uint64_t pin = upload(mem, in), p = upload(mem, start);
  run(o, "atomics2", mem, {p, pin, static_cast<uint64_t>(n)});

  int want_or = 0, want_and = -1;
  for (int i = 0; i < n; ++i) {
    want_or |= in[i];
    want_and &= in[i];
  }
  const std::vector<int> out = download<int>(mem, p, 3);
  VCHECK_EQ(out[0], want_or);
  VCHECK_EQ(out[1], want_and);
  // The compare-and-swap writes only where it finds what it expected, which
  // one lane does; every other lane leaves it alone.
  bool found = false;
  for (int i = 0; i < n; ++i) found = found || out[2] == in[i];
  VCHECK(found);
}

VTEST_MAIN
