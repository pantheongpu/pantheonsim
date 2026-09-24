// Half precision one value at a time, run and checked against what the C
// means.
//
// The kernels are amd/tests/data/half.c, compiled for gfx942. A half is the
// low sixteen bits of a register, and the arithmetic on it rounds to eleven
// significant bits at every step, so these tests compute the same thing in
// _Float16 on the host and expect the answers to match exactly: a half
// computed in float and rounded at the end would differ in the last bits,
// which is the mistake worth catching.
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
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/half.gfx942.o";
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

uint64_t upload_halves(MemoryManager& mem, const std::vector<_Float16>& v) {
  const uint64_t p = mem.alloc(v.size() * 2);
  mem.write(p, v.data(), v.size() * 2);
  return p;
}

std::vector<_Float16> download_halves(MemoryManager& mem, uint64_t p, size_t n) {
  std::vector<_Float16> out(n);
  mem.read(p, out.data(), n * 2);
  return out;
}

std::string shown(_Float16 v) { return std::to_string(static_cast<double>(v)); }

}  // namespace

VTEST(the_arithmetic_rounds_at_every_step_as_a_half_does) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<_Float16> a(kN), b(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<_Float16>(static_cast<float>(i) * 0.75f - 24.0f);
    b[i] = static_cast<_Float16>(static_cast<float>(i % 7) + 0.125f);
  }
  const uint64_t pa = upload_halves(mem, a), pb = upload_halves(mem, b), pout = mem.alloc(kN * 2);
  run(o, "h_math", mem, {pa, pb, pout, static_cast<uint64_t>(kN)});

  const std::vector<_Float16> out = download_halves(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    // The compiler folds the multiply and the add into one rounding.
    const _Float16 want = static_cast<_Float16>(
        static_cast<_Float16>(std::fma(static_cast<float>(a[i]), static_cast<float>(b[i]),
                                       static_cast<float>(a[i]))) - b[i]);
    if (!(out[i] == want))
      throw vtest::Failure("h_math[" + std::to_string(i) + "] is " + shown(out[i]) + ", not " + shown(want));
  }
}

VTEST(a_plain_multiply_of_two_halves) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<_Float16> a(kN), b(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<_Float16>(static_cast<float>(i) * 1.5f - 40.0f);
    b[i] = static_cast<_Float16>(1.0f / static_cast<float>(i + 1));
  }
  const uint64_t pa = upload_halves(mem, a), pb = upload_halves(mem, b), pout = mem.alloc(kN * 2);
  run(o, "h_scale", mem, {pa, pb, pout, static_cast<uint64_t>(kN)});

  const std::vector<_Float16> out = download_halves(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const _Float16 want = static_cast<_Float16>(a[i] * b[i]);
    if (!(out[i] == want))
      throw vtest::Failure("h_scale[" + std::to_string(i) + "] is " + shown(out[i]) + ", not " + shown(want));
  }
}

VTEST(the_smaller_and_the_larger_of_two_halves) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<_Float16> a(kN), b(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<_Float16>(static_cast<float>(i) - 32.0f);
    b[i] = static_cast<_Float16>(static_cast<float>(32 - i) * 0.5f);
  }
  const uint64_t pa = upload_halves(mem, a), pb = upload_halves(mem, b), pout = mem.alloc(kN * 2);
  run(o, "h_choose", mem, {pa, pb, pout, static_cast<uint64_t>(kN)});

  const std::vector<_Float16> out = download_halves(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const _Float16 m = a[i] < b[i] ? a[i] : b[i], M = a[i] > b[i] ? a[i] : b[i];
    const _Float16 want = static_cast<_Float16>(
        std::fma(static_cast<float>(static_cast<_Float16>(a[i] + b[i])),
                 static_cast<float>(static_cast<_Float16>(m - M)), 0.5f));
    if (!(out[i] == want))
      throw vtest::Failure("h_choose[" + std::to_string(i) + "] is " + shown(out[i]) + ", not " + shown(want));
  }
}

VTEST(between_a_half_and_a_float_in_both_directions) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<_Float16> a(kN);
  std::vector<float> g(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<_Float16>(static_cast<float>(i) * 0.3125f - 10.0f);
    g[i] = static_cast<float>(i) * 1.1f - 30.0f;   // not a half exactly, so the narrowing rounds
  }
  const uint64_t pa = upload_halves(mem, a), pg = upload(mem, g), pf = mem.alloc(kN * 4),
                 pout = mem.alloc(kN * 2);
  run(o, "h_convert", mem, {pa, pg, pf, pout, static_cast<uint64_t>(kN)});

  const std::vector<float> f = download<float>(mem, pf, kN);
  const std::vector<_Float16> out = download_halves(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    VCHECK_EQ(f[i], static_cast<float>(a[i]) * 2.0f);
    const _Float16 want = static_cast<_Float16>(g[i] + 1.0f);
    if (!(out[i] == want))
      throw vtest::Failure("h_convert[" + std::to_string(i) + "] is " + shown(out[i]) + ", not " + shown(want));
  }
}

VTEST(the_comparisons_including_the_ones_a_nan_answers_differently) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<_Float16> a(kN), b(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<_Float16>(static_cast<float>(i % 8) - 3.0f);
    b[i] = static_cast<_Float16>(static_cast<float>(i % 5) - 2.0f);
  }
  // A NaN in both halves of one pair: every ordered comparison answers no,
  // and the two that are written as negations answer yes.
  const uint16_t nan_bits = 0x7E00;
  std::memcpy(&a[7], &nan_bits, 2);
  std::memcpy(&b[9], &nan_bits, 2);
  const uint64_t pa = upload_halves(mem, a), pb = upload_halves(mem, b), pout = mem.alloc(kN * 4);
  run(o, "h_cmp", mem, {pa, pb, pout, static_cast<uint64_t>(kN)});

  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const int32_t want = (a[i] < b[i]) + 2 * (a[i] >= b[i]) + 4 * !(a[i] == b[i]) + 8 * (a[i] == b[i]) +
                         16 * (a[i] > b[i] ? 1 : 0);
    if (out[i] != want)
      throw vtest::Failure("h_cmp[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST_MAIN
