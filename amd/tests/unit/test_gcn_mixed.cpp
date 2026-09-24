// Arithmetic that mixes widths, run and checked against what the C means.
//
// The kernels are amd/tests/data/mixed.c, compiled for gfx942. Where a
// kernel adds two bytes or two shorts and then widens the result, the
// compiler does not narrow the registers first: it uses the sub-dword form of
// a 32-bit instruction, which reads a named byte or half of each source and
// takes it into 32 bits with its sign or without. Getting the sign wrong
// would show here as a number too large by 65536 or 256.
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
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/mixed.gfx942.o";
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

VTEST(a_half_of_each_source_taken_with_its_sign) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int16_t> a(kN);
  for (int i = 0; i < kN; ++i) a[i] = static_cast<int16_t>(i * 1100 - 32000);   // both signs, and it wraps
  const uint64_t pa = upload(mem, a), pout = mem.alloc(kN * 4);
  run(o, "word_sext", mem, {pa, pout, static_cast<uint64_t>(kN)});

  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const int32_t want = static_cast<int16_t>(a[i] - 1000) + static_cast<int16_t>(a[i] << 1);
    if (out[i] != want)
      throw vtest::Failure("word_sext[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(a_byte_of_each_source_taken_without_one) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<uint8_t> a(kN), b(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<uint8_t>(i * 5 + 200);   // above 127, where a sign would show
    b[i] = static_cast<uint8_t>(255 - i * 3);
  }
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(kN * 4);
  run(o, "byte_zext", mem, {pa, pb, pout, static_cast<uint64_t>(kN)});

  const std::vector<uint32_t> out = download<uint32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const uint32_t want = static_cast<uint8_t>(a[i] + b[i]) + static_cast<uint32_t>(static_cast<uint8_t>(a[i] * 3));
    if (out[i] != want)
      throw vtest::Failure("byte_zext[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(a_byte_from_one_source_and_a_half_from_the_other) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int8_t> a(kN);
  std::vector<int16_t> b(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<int8_t>(i * 4 - 128);
    b[i] = static_cast<int16_t>(i * 900 - 30000);
  }
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(kN * 4);
  run(o, "mixed_widths", mem, {pa, pb, pout, static_cast<uint64_t>(kN)});

  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const int32_t want = static_cast<int32_t>(static_cast<int8_t>(a[i] + 1)) *
                         static_cast<int32_t>(static_cast<int16_t>(b[i] - 2));
    if (out[i] != want)
      throw vtest::Failure("mixed_widths[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST_MAIN
