// 64-bit integers and the conversions a double needs, run and checked
// against what the C means.
//
// The kernels are amd/tests/data/int64.c, compiled for gfx942. A 64-bit
// value lives in a register pair, and the machine has no instruction that
// adds one: the compiler adds the low halves, keeps a mask of the lanes that
// carried, and adds that back into the high halves. These tests pick numbers
// whose low halves carry for some lanes and not others, so a carry that went
// to the wrong lane would show.
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
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/int64.gfx942.o";
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

VTEST(a_64_bit_multiply_shift_division_and_remainder) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int64_t> a(kN), b(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<int64_t>(i - 32) * 1000000007ll;   // both signs, and wider than 32 bits
    b[i] = static_cast<int64_t>(i % 11) * 65536 - 5;
  }
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(kN * 8);
  run(o, "i64_math", mem, {pa, pb, pout, static_cast<uint64_t>(kN)});

  const std::vector<int64_t> out = download<int64_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const int64_t y = b[i] | 1;
    const int64_t want = a[i] * y + (a[i] >> 3) - (a[i] / y) + (a[i] % y);
    if (out[i] != want)
      throw vtest::Failure("i64_math[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(an_add_and_a_subtract_thread_a_carry_through_the_pair) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<uint64_t> a(kN), b(kN);
  for (int i = 0; i < kN; ++i) {
    // Half the lanes carry out of the low half and half do not, so a carry
    // that reached the wrong lane would change the answer.
    a[i] = (uint64_t{i} << 32) | (i % 2 ? 0xFFFFFFF0ull : 0x10ull);
    b[i] = (uint64_t{i * 3} << 32) | 0x20ull;
  }
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(kN * 8);
  run(o, "u64_addsub", mem, {pa, pb, pout, static_cast<uint64_t>(kN)});

  const std::vector<uint64_t> out = download<uint64_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const uint64_t want = (a[i] + b[i]) ^ (a[i] - b[i]);
    if (out[i] != want)
      throw vtest::Failure("u64_addsub[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(counting_the_bits_of_a_64_bit_value) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<uint64_t> a(kN);
  for (int i = 0; i < kN; ++i) a[i] = (uint64_t{0x9E3779B9ull} * static_cast<uint64_t>(i + 1)) << (i % 17);
  const uint64_t pa = upload(mem, a), pout = mem.alloc(kN * 8);
  run(o, "u64_bits", mem, {pa, pout, static_cast<uint64_t>(kN)});

  const std::vector<uint64_t> out = download<uint64_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const uint64_t x = a[i] | 1;
    const uint64_t want = static_cast<uint64_t>(__builtin_popcountll(x)) +
                          static_cast<uint64_t>(__builtin_clzll(x)) + static_cast<uint64_t>(__builtin_ctzll(x));
    if (out[i] != want)
      throw vtest::Failure("u64_bits[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(comparing_two_64_bit_values_is_one_instruction_over_the_pair) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<uint64_t> a(kN), b(kN);
  for (int i = 0; i < kN; ++i) {
    // The halves disagree on purpose: a comparison that only looked at one of
    // them would answer some of these the other way round.
    a[i] = (uint64_t{i % 8} << 32) | static_cast<uint64_t>(63 - i);
    b[i] = (uint64_t{i / 8} << 32) | static_cast<uint64_t>(i);
  }
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(kN * 4);
  run(o, "u64_cmp", mem, {pa, pb, pout, static_cast<uint64_t>(kN)});

  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const int32_t want = (a[i] == b[i]) + 2 * (a[i] < b[i]);
    if (out[i] != want)
      throw vtest::Failure("u64_cmp[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(a_double_turned_into_an_integer_and_into_a_float) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<double> a(kN);
  for (int i = 0; i < kN; ++i) a[i] = (static_cast<double>(i) - 32.0) * 1234.5678;
  const uint64_t pa = upload(mem, a), pi = mem.alloc(kN * 4), pf = mem.alloc(kN * 4);
  run(o, "dbl_convert", mem, {pa, pi, pf, static_cast<uint64_t>(kN)});

  const std::vector<int32_t> oi = download<int32_t>(mem, pi, kN);
  const std::vector<float> of = download<float>(mem, pf, kN);
  for (int i = 0; i < kN; ++i) {
    const int32_t want_i = static_cast<int32_t>(a[i]);
    VCHECK_EQ(oi[i], want_i);
    const float want_f = static_cast<float>(a[i]) + static_cast<float>(static_cast<unsigned>(want_i));
    if (of[i] != want_f)
      throw vtest::Failure("dbl_convert[" + std::to_string(i) + "] is " + std::to_string(of[i]) + ", not " +
                           std::to_string(want_f));
  }
}

VTEST_MAIN
