// What a kernel gets for the library functions it calls, run and checked
// against the same calls in C.
//
// The kernels are amd/tests/data/builtins.c, compiled for gfx942. Two of
// these are worth naming. The machine's sine and cosine take their argument
// in turns rather than radians -- a whole turn is 1.0 -- which is why the
// compiler multiplies by one over two pi first, and getting that wrong would
// give an answer that is wrong by a factor rather than by a rounding. And the
// counter a wave reads to time itself is this model's own: it counts the
// instructions the dispatch has retired, so the test asks only that it goes
// up, which is what a program timing a stretch of its own code depends on.
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
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/builtins.gfx942.o";
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

VTEST(the_four_ways_of_rounding_a_float) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<float> a(kN);
  for (int i = 0; i < kN; ++i) a[i] = static_cast<float>(i) * 0.5f - 16.0f;   // halves, where the tie-break shows
  const uint64_t pa = upload(mem, a), pout = mem.alloc(kN * 4);
  run(o, "rounding", mem, {pa, pout, static_cast<uint64_t>(kN)});

  const std::vector<float> out = download<float>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const float want = std::floor(a[i]) + std::ceil(a[i]) * 2.0f + std::rint(a[i]) * 4.0f +
                       std::trunc(a[i]) * 8.0f;
    if (out[i] != want)
      throw vtest::Failure("rounding[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(the_sine_and_the_cosine_of_an_angle_given_in_radians) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<float> a(kN);
  for (int i = 0; i < kN; ++i) a[i] = static_cast<float>(i) * 0.196349540849f - 6.0f;   // a little over two turns
  const uint64_t pa = upload(mem, a), pout = mem.alloc(kN * 4);
  run(o, "turns", mem, {pa, pout, static_cast<uint64_t>(kN)});

  const std::vector<float> out = download<float>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const float want = std::sin(a[i]) + std::cos(a[i]);
    // The compiler's own rounding is in here: it multiplies by one over two
    // pi as a float before the instruction sees the argument. A card's tables
    // would differ by about as much again.
    if (std::fabs(out[i] - want) > 1e-5f)
      throw vtest::Failure("turns[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(a_power_of_two_applied_to_a_float) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<float> a(kN);
  std::vector<int32_t> e(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<float>(i) * 1.25f - 40.0f;
    e[i] = i % 21 - 10;
  }
  const uint64_t pa = upload(mem, a), pe = upload(mem, e), pout = mem.alloc(kN * 4);
  run(o, "scaled", mem, {pa, pe, pout, static_cast<uint64_t>(kN)});

  const std::vector<float> out = download<float>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const float want = std::ldexp(a[i], e[i]);
    if (out[i] != want)
      throw vtest::Failure("scaled[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(the_smallest_and_the_largest_of_three) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int32_t> a(kN), b(kN), c(kN);
  std::vector<float> fa(kN), fb(kN), fc(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = i * 37 - 900;
    b[i] = (i % 7) * 500 - 1000;
    c[i] = 250 - i * 11;
    fa[i] = static_cast<float>(a[i]) * 0.5f;
    fb[i] = static_cast<float>(b[i]) * 0.25f;
    fc[i] = static_cast<float>(c[i]);
  }
  {
    const uint64_t pa = upload(mem, a), pb = upload(mem, b), pc = upload(mem, c), pout = mem.alloc(kN * 4);
    run(o, "three", mem, {pa, pb, pc, pout, static_cast<uint64_t>(kN)});
    const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
    for (int i = 0; i < kN; ++i) {
      const int32_t lo = std::min(std::min(a[i], b[i]), c[i]), hi = std::max(std::max(a[i], b[i]), c[i]);
      VCHECK_EQ(out[i], lo + hi * 2);
    }
  }
  {
    const uint64_t pa = upload(mem, fa), pb = upload(mem, fb), pc = upload(mem, fc), pout = mem.alloc(kN * 4);
    run(o, "three_f", mem, {pa, pb, pc, pout, static_cast<uint64_t>(kN)});
    const std::vector<float> out = download<float>(mem, pout, kN);
    for (int i = 0; i < kN; ++i) {
      const float want = std::fmin(std::fmin(fa[i], fb[i]), fc[i]) + std::fmax(std::fmax(fa[i], fb[i]), fc[i]);
      VCHECK_EQ(out[i], want);
    }
  }
}

VTEST(four_bytes_put_back_in_the_other_order) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<uint32_t> a(kN);
  for (int i = 0; i < kN; ++i) a[i] = 0x01020304u * static_cast<uint32_t>(i + 1) + 0x0A0B0C0Du;
  const uint64_t pa = upload(mem, a), pout = mem.alloc(kN * 4);
  run(o, "swapped", mem, {pa, pout, static_cast<uint64_t>(kN)});

  const std::vector<uint32_t> out = download<uint32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const uint32_t x = a[i];
    const uint32_t want = (x & 0xffu) << 24 | ((x >> 8) & 0xffu) << 16 | ((x >> 16) & 0xffu) << 8 | x >> 24;
    VCHECK_EQ(out[i], want);
  }
}

VTEST(four_words_loaded_and_stored_at_once) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<uint32_t> a(kN * 4);
  for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<uint32_t>(i) * 2654435761u;
  const uint64_t pa = upload(mem, a), pout = mem.alloc(kN * 16);
  run(o, "four_at_a_time", mem, {pa, pout, static_cast<uint64_t>(kN)});

  const std::vector<uint32_t> out = download<uint32_t>(mem, pout, a.size());
  for (size_t i = 0; i < a.size(); ++i) VCHECK_EQ(out[i], a[i] * 3u + 7u);
}

VTEST(the_counter_a_wave_times_itself_by_only_goes_up) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const uint64_t pout = mem.alloc(2 * 8);
  run(o, "timed", mem, {pout});
  const std::vector<uint64_t> out = download<uint64_t>(mem, pout, 2);
  VCHECK(out[0] > 0);
  VCHECK(out[1] > out[0]);
}

VTEST_MAIN
