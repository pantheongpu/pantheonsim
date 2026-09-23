// The instructions a real kernel uses, run and checked against what the C
// means: integer and float math, division, conversions, a loop, a
// two-dimensional grid, an atomic, and 64-bit values.
//
// The kernels are amd/tests/data/ops.c, compiled for gfx942. Each test runs
// one and compares every element with the same arithmetic done here. The
// decoding of the same object is checked separately against the assembler
// (test_gcn.cpp), so a pass here means the instructions were both read and
// carried out as the compiler intended.
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
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/ops.gfx942.o";
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

uint64_t bits_of(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}

// Runs one kernel of the fixture over `groups` work-groups of 64.
amd::DispatchStats run(const amd::CodeObject& o, const char* name, MemoryManager& mem,
                       const std::vector<uint64_t>& args, uint32_t groups, uint32_t gy = 1, uint32_t by = 1) {
  const amd::Kernel* k = amd::find_kernel(o, name);
  if (!k) throw vtest::Failure(std::string("no kernel named ") + name);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = kernargs(mem, *k, args);
  d.groups[0] = groups;
  d.groups[1] = gy;
  d.group_size[0] = by > 1 ? 8 : 64;
  d.group_size[1] = by;
  return amd::execute(d, mem);
}

}  // namespace

VTEST(integer_math_comes_out_as_the_c_does) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const int n = 128;
  std::vector<int> a(n), b(n);
  for (int i = 0; i < n; ++i) {
    a[i] = i * 37 - 500;       // negatives too, so the signed paths run
    b[i] = (i % 11) + 1;
  }
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(n * 4);
  run(o, "int_math", mem, {pa, pb, pout, static_cast<uint64_t>(n)}, n / 64);

  const std::vector<int> out = download<int>(mem, pout, n);
  for (int i = 0; i < n; ++i) {
    const int x = a[i], y = b[i];
    const int want = (x * y) + (x / 3) + (y % 5) - (x & y) + (x | y) ^ (x >> 2);
    if (out[i] != want)
      throw vtest::Failure("int_math[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(float_math_including_a_division_comes_out_as_the_c_does) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const int n = 128;
  std::vector<float> a(n), b(n);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i) * 0.75f - 40.0f;
    b[i] = static_cast<float>(i % 13) + 1.5f;   // never zero: the C would divide by it
  }
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(n * 4);
  run(o, "float_math", mem, {pa, pb, pout, static_cast<uint64_t>(n)}, n / 64);

  const std::vector<float> out = download<float>(mem, pout, n);
  for (int i = 0; i < n; ++i) {
    const float x = a[i], y = b[i];
    const float want = x * y + x / y + (x < y ? x : y) + std::fma(x, y, 1.0f);
    if (out[i] != want)
      throw vtest::Failure("float_math[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want) + " (division and fma are exact, so these must match bit for bit)");
  }
}

VTEST(a_loop_with_a_scalar_counter_runs_to_its_end) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const int n = 64;
  std::vector<float> in(n);
  for (int i = 0; i < n; ++i) in[i] = static_cast<float>(i % 5) * 0.25f;
  const uint64_t pin = upload(mem, in), pout = mem.alloc(n * 4);
  const amd::DispatchStats stats = run(o, "loop_sum", mem, {pin, pout, static_cast<uint64_t>(n)}, 1);
  // A loop of 64 rounds over 64 lanes retires far more than a straight line.
  VCHECK(stats.instructions > 500);

  const std::vector<float> out = download<float>(mem, pout, n);
  for (int i = 0; i < n; ++i) {
    float want = 0;
    for (int k = 0; k < n; ++k) want += in[k] * static_cast<float>(k + i);
    if (out[i] != want)
      throw vtest::Failure("loop_sum[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(a_two_dimensional_grid_numbers_its_work_items) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const int w = 16;
  std::vector<float> in(w * w);
  for (int i = 0; i < w * w; ++i) in[i] = static_cast<float>(i) * 0.5f;
  const uint64_t pin = upload(mem, in), pout = mem.alloc(w * w * 4);
  // Two work-groups each way, eight work-items each way.
  run(o, "grid2d", mem, {pin, pout, static_cast<uint64_t>(w)}, 2, 2, 8);

  const std::vector<float> out = download<float>(mem, pout, w * w);
  for (int y = 0; y < w; ++y)
    for (int x = 0; x < w; ++x) {
      const float want = in[x * w + y] + static_cast<float>(x + y);
      if (out[y * w + x] != want)
        throw vtest::Failure("grid2d[" + std::to_string(y) + "][" + std::to_string(x) + "] is " +
                             std::to_string(out[y * w + x]) + ", not " + std::to_string(want));
    }
}

VTEST(an_atomic_add_keeps_every_lanes_contribution) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const int n = 256;
  std::vector<int> in(n);
  int want = 0;
  for (int i = 0; i < n; ++i) {
    in[i] = (i % 7) - 2;               // some are negative, and the kernel skips those
    if (in[i] > 0) want += in[i];
  }
  const uint64_t pin = upload(mem, in), pcounter = mem.alloc(4);
  mem.fill(pcounter, reinterpret_cast<const uint8_t*>("\0"), 1, 4);
  run(o, "atomics", mem, {pcounter, pin, static_cast<uint64_t>(n)}, n / 64);
  VCHECK_EQ(download<int>(mem, pcounter, 1)[0], want);
}

VTEST(sixty_four_bit_values_are_kept_in_register_pairs) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const int n = 64;
  std::vector<int64_t> a(n);
  for (int i = 0; i < n; ++i) a[i] = (static_cast<int64_t>(i) << 40) - 12345678901LL * i;
  const uint64_t pa = upload(mem, a), pout = mem.alloc(n * 8);
  run(o, "long_math", mem, {pa, pout, static_cast<uint64_t>(n)}, 1);

  const std::vector<int64_t> out = download<int64_t>(mem, pout, n);
  for (int i = 0; i < n; ++i) {
    const int64_t want = a[i] * 3 + (a[i] >> 7);
    if (out[i] != want)
      throw vtest::Failure("long_math[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST_MAIN
