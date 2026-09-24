// The atomics: the ones beyond an add, the ones on a float and on a pair,
// the ones in LDS, and what a lane is told was there before its turn.
//
// The kernels are amd/tests/data/atomics.c, compiled for gfx942. An atomic
// that is asked for the value it replaced is where the order of the lanes
// becomes visible, and the compiler does not leave that order to the machine:
// for an add it works out the wave's total, has one lane add it, and gives
// each lane the part of the total that belongs to the lanes before it; for an
// exchange it walks the lanes one at a time. Either way the answer is the one
// a program would get by running the lanes in order, which is what these
// tests compare against.
#include <algorithm>
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
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/atomics.gfx942.o";
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

VTEST(an_atomic_tells_each_lane_what_was_there_before_its_turn) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int32_t> in(kN);
  for (int i = 0; i < kN; ++i) in[i] = (i * 7919) | 1;
  const std::vector<int32_t> start = {100, 5000, 0x5A5A5A5A, -7};
  const uint64_t pin = upload(mem, in), p = upload(mem, start), pout = mem.alloc(kN * 4);
  run(o, "returning", mem, {p, pout, pin, static_cast<uint64_t>(kN)});

  // Lane by lane, in order: each is told what the one before it left.
  std::vector<int32_t> want(kN, 0);
  int32_t added = start[0], subtracted = start[1], xored = start[2], swapped = start[3];
  for (int i = 0; i < kN; ++i) {
    want[i] = added + subtracted + xored + swapped;
    added += in[i];
    subtracted -= in[i];
    xored ^= in[i];
    swapped = in[i];
  }
  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    if (out[i] != want[i])
      throw vtest::Failure("returning[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want[i]));
  }
  // And what they left behind.
  const std::vector<int32_t> left = download<int32_t>(mem, p, 4);
  VCHECK_EQ(left[0], added);
  VCHECK_EQ(left[1], subtracted);
  VCHECK_EQ(left[2], xored);
  VCHECK_EQ(left[3], swapped);
}

VTEST(an_atomic_add_on_a_float_and_on_a_64_bit_value) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int32_t> in(kN);
  for (int i = 0; i < kN; ++i) in[i] = i * 3 - 90;
  const std::vector<float> f0 = {0.5f};
  const std::vector<uint64_t> q0 = {uint64_t{1} << 40};
  const uint64_t pin = upload(mem, in), pf = upload(mem, f0), pq = upload(mem, q0);
  run(o, "wide", mem, {pf, pq, pin, static_cast<uint64_t>(kN)});

  float want_f = f0[0];
  uint64_t want_q = q0[0];
  for (int i = 0; i < kN; ++i) {
    want_f += static_cast<float>(in[i]);
    want_q += static_cast<uint64_t>(in[i]);
  }
  VCHECK_EQ(download<float>(mem, pf, 1)[0], want_f);
  VCHECK_EQ(download<uint64_t>(mem, pq, 1)[0], want_q);
}

VTEST(the_atomics_a_work_group_shares_in_lds) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int32_t> in(kN);
  for (int i = 0; i < kN; ++i) in[i] = (i % 3 ? 1 : -1) * (i * 104729 % 100000);   // both signs, for the maximum
  const uint64_t pin = upload(mem, in), pout = mem.alloc(kN * 4);
  run(o, "shared", mem, {pout, pin, static_cast<uint64_t>(kN)});

  // Eight slots of each kind, and eight lanes share each one.
  int32_t xored[8] = {}, largest[8];
  for (int i = 0; i < 8; ++i) largest[i] = -1000000;
  for (int i = 0; i < kN; ++i) {
    xored[i & 7] ^= in[i];
    largest[i & 7] = std::max(largest[i & 7], in[i]);
  }
  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const int32_t want = xored[i & 7] + largest[i & 7];
    if (out[i] != want)
      throw vtest::Failure("shared[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST_MAIN
