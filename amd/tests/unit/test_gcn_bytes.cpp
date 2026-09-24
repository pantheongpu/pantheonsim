// Values narrower than a register: bytes and shorts, signed and unsigned,
// loaded, computed on and stored back, run and checked against what the C
// means.
//
// The kernels are amd/tests/data/bytes.c, compiled for gfx942. Two things
// they exercise are worth naming. A load that is narrower than a register
// either carries the sign into the rest of it or does not, and which one is
// in the instruction's name rather than in the type the kernel wrote. And
// 16-bit arithmetic writes the low half of a register and zeroes the high
// half: the compiler leaves out the mask a widening would otherwise need
// after one of these, which is how a kernel that stores a whole register
// after a 16-bit add still stores the right number.
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
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/bytes.gfx942.o";
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

VTEST(a_byte_is_loaded_and_stored_as_a_byte_and_its_neighbours_are_left_alone) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<uint8_t> in(kN);
  for (int i = 0; i < kN; ++i) in[i] = static_cast<uint8_t>(i * 5 + 1);
  const uint64_t pin = upload(mem, in), pout = mem.alloc(kN);
  // The output is filled first, so a store wider than a byte would show.
  const uint8_t marker = 0xAB;
  mem.fill(pout, &marker, 1, kN);
  run(o, "bytes_u", mem, {pin, pout, static_cast<uint64_t>(kN)});

  const std::vector<uint8_t> out = download<uint8_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const uint8_t want = static_cast<uint8_t>(in[i] * 3 + 7);
    if (out[i] != want)
      throw vtest::Failure("bytes_u[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(a_signed_byte_carries_its_sign_into_the_rest_of_the_register) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int8_t> in(kN);
  for (int i = 0; i < kN; ++i) in[i] = static_cast<int8_t>(i * 4 - 128);   // both signs
  const uint64_t pin = upload(mem, in), pout = mem.alloc(kN * 4);
  run(o, "bytes_i", mem, {pin, pout, static_cast<uint64_t>(kN)});

  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const int32_t want = static_cast<int32_t>(in[i]) - 5;
    if (out[i] != want)
      throw vtest::Failure("bytes_i[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(shorts_are_computed_on_and_stored_as_shorts) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<uint16_t> a(kN), b(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<uint16_t>(i * 1013 + 7);
    b[i] = static_cast<uint16_t>(60000 - i * 900);   // the sum wraps, as a short's does
  }
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(kN * 2);
  run(o, "shorts_u", mem, {pa, pb, pout, static_cast<uint64_t>(kN)});

  const std::vector<uint16_t> out = download<uint16_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const uint16_t want = static_cast<uint16_t>(a[i] + b[i] * 2);
    if (out[i] != want)
      throw vtest::Failure("shorts_u[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(a_short_that_is_narrowed_keeps_the_sign_of_what_is_left) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int16_t> a(kN);
  for (int i = 0; i < kN; ++i) a[i] = static_cast<int16_t>(i * 1100 - 32000);
  const uint64_t pa = upload(mem, a), pout = mem.alloc(kN * 4);
  run(o, "shorts_i", mem, {pa, pout, static_cast<uint64_t>(kN)});

  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const int32_t want = static_cast<int16_t>(a[i] - 1000);
    if (out[i] != want)
      throw vtest::Failure("shorts_i[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(a_signed_load_of_each_width_widens_where_it_is_read) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int16_t> a(kN);
  std::vector<int8_t> b(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<int16_t>(i * 900 - 30000);
    b[i] = static_cast<int8_t>(i * 4 - 128);
  }
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(kN * 4);
  run(o, "widen_signed", mem, {pa, pb, pout, static_cast<uint64_t>(kN)});

  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const int32_t want = static_cast<int32_t>(a[i]) + static_cast<int32_t>(b[i]);
    if (out[i] != want)
      throw vtest::Failure("widen_signed[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(a_short_widened_after_16_bit_arithmetic_leaves_no_high_half_behind) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<uint16_t> a(kN);
  for (int i = 0; i < kN; ++i) a[i] = static_cast<uint16_t>(65529 + i);   // the sum wraps for most of them
  const uint64_t pa = upload(mem, a), pout = mem.alloc(kN * 4);
  run(o, "widen", mem, {pa, pout, static_cast<uint64_t>(kN)});

  const std::vector<uint32_t> out = download<uint32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const uint32_t want = static_cast<uint16_t>(a[i] + 7u);
    if (out[i] != want)
      throw vtest::Failure("widen[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST_MAIN
