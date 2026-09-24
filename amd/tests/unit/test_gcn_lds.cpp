// The rest of LDS, run and checked against answers worked out in C: values
// wider and narrower than a word, a float atomic, and the swizzle, where
// lanes trade values through the LDS unit without touching LDS.
//
// The kernels are amd/tests/data/lds.c, compiled for gfx942. The swizzle is
// pinned down the way the cross-lane form was: a butterfly -- each lane
// adding the lane whose number differs from its own in one bit, for each of
// five bits -- leaves every lane holding the sum of its group of 32, which
// the test works out for itself. And the last kernel reads another lane twice,
// the second time into the register it reads from, which is where a lane
// further on must still see what was there before an earlier lane wrote.
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
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/lds.gfx942.o";
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

VTEST(two_words_at_a_time) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<double> in(kN);
  for (int i = 0; i < kN; ++i) in[i] = static_cast<double>(i) * 1.5 - 20.0;
  const uint64_t pin = upload(mem, in), pout = mem.alloc(kN * 8);
  run(o, "doubles_in_lds", mem, {pin, pout, static_cast<uint64_t>(kN)});
  const std::vector<double> out = download<double>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) VCHECK_EQ(out[i], in[(i + 1) & 63] * 2.0 + in[(i + 5) & 63] * 2.0);
}

VTEST(four_words_at_a_time) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<float> in(kN * 4);
  for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i) * 0.25f - 7.0f;
  const uint64_t pin = upload(mem, in), pout = mem.alloc(kN * 16);
  run(o, "quads_in_lds", mem, {pin, pout, static_cast<uint64_t>(kN)});
  const std::vector<float> out = download<float>(mem, pout, in.size());
  for (int i = 0; i < kN; ++i)
    for (int k = 0; k < 4; ++k) VCHECK_EQ(out[i * 4 + k], in[((i + 1) & 63) * 4 + k]);
}

VTEST(a_byte_a_half_and_a_signed_byte) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<uint8_t> a(kN);
  std::vector<uint16_t> b(kN);
  std::vector<int8_t> c(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<uint8_t>(i * 5 + 130);      // above 127, where a sign would show if one were taken
    b[i] = static_cast<uint16_t>(i * 1000 + 3);
    c[i] = static_cast<int8_t>(i * 4 - 128);       // both signs
  }
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pc = upload(mem, c), pout = mem.alloc(kN * 4);
  run(o, "narrow_in_lds", mem, {pa, pb, pc, pout, static_cast<uint64_t>(kN)});
  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const int32_t want = a[(i + 1) & 63] + b[(i + 2) & 63] + c[(i + 3) & 63];
    if (out[i] != want)
      throw vtest::Failure("narrow_in_lds[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(a_float_atomic_collects_every_lane) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<float> in(kN);
  for (int i = 0; i < kN; ++i) in[i] = static_cast<float>(i % 10) * 0.5f;   // exact in a float, whatever the order
  const uint64_t pin = upload(mem, in), pout = mem.alloc(kN * 4);
  run(o, "float_sums_in_lds", mem, {pin, pout, static_cast<uint64_t>(kN)});
  float want[8] = {};
  for (int i = 0; i < kN; ++i) want[i & 7] += in[i];
  const std::vector<float> out = download<float>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) VCHECK_EQ(out[i], want[i & 7]);
}

VTEST(a_butterfly_leaves_every_lane_with_its_groups_sum) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int32_t> in(kN);
  for (int i = 0; i < kN; ++i) in[i] = i * 13 - 300;
  const uint64_t pin = upload(mem, in), pout = mem.alloc(kN * 4);
  run(o, "butterfly", mem, {pin, pout, static_cast<uint64_t>(kN)});
  int32_t sums[2] = {};
  for (int i = 0; i < kN; ++i) sums[i / 32] += in[i];
  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i)
    if (out[i] != sums[i / 32])
      throw vtest::Failure("butterfly[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(sums[i / 32]) + " (the sum of its group of 32)");
}

VTEST(a_group_reversed_one_lane_handed_to_eight_and_the_first_of_each_four) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int32_t> in(kN);
  for (int i = 0; i < kN; ++i) in[i] = 1000 + i;
  const uint64_t pin = upload(mem, in), pr = mem.alloc(kN * 4), pb = mem.alloc(kN * 4), pq = mem.alloc(kN * 4);
  run(o, "swizzle_shapes", mem, {pin, pr, pb, pq, static_cast<uint64_t>(kN)});
  const std::vector<int32_t> reversed = download<int32_t>(mem, pr, kN), broadcast = download<int32_t>(mem, pb, kN),
                             quad = download<int32_t>(mem, pq, kN);
  for (int i = 0; i < kN; ++i) {
    VCHECK_EQ(reversed[i], in[(i & ~31) + (31 - (i & 31))]);
    VCHECK_EQ(broadcast[i], in[(i & ~7) + 5]);
    VCHECK_EQ(quad[i], in[i & ~3]);
  }
}

VTEST(a_lane_reads_another_into_the_register_it_reads_from) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int32_t> in(kN);
  for (int i = 0; i < kN; ++i) in[i] = 1000 + i;
  const uint64_t pin = upload(mem, in), pout = mem.alloc(kN * 4);
  run(o, "permute_twice", mem, {pin, pout, static_cast<uint64_t>(kN)});
  // Lane t took lane t+1's value, then lane t+2's copy of what that lane had
  // taken: in[t + 3], all modulo the wave.
  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i)
    if (out[i] != in[(i + 3) & 63])
      throw vtest::Failure("permute_twice[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(in[(i + 3) & 63]));
}

VTEST_MAIN
