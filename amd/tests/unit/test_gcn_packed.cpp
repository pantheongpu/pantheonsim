// Two floats in a register pair, both computed at once, run and checked
// against the same arithmetic in C.
//
// The kernels are amd/tests/data/packed.c, compiled for gfx942. Where a
// kernel works on a vector of two or four floats, the compiler pairs them up
// and uses one instruction for both. Which register of a pair feeds which
// half of the result is in the instruction, and a constant is written as one
// value that serves both halves, so a model that read the second register of
// a constant would take whatever happened to be beside it.
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
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/packed.gfx942.o";
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

// Each work-item handles one pair, so the buffers are 2 floats per lane.
std::vector<float> pairs(int seed) {
  std::vector<float> v(kN * 2);
  for (size_t i = 0; i < v.size(); ++i)
    v[i] = static_cast<float>(static_cast<int>(i) * (seed + 1) % 97) * 0.375f - 12.0f;
  return v;
}

}  // namespace

VTEST(a_multiply_and_an_add_over_both_halves) {
  const amd::CodeObject o = object();
  const std::vector<float> a = pairs(1), b = pairs(2);
  {
    MemoryManager mem(64ull << 20);
    const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(kN * 8);
    run(o, "pk_mul", mem, {pa, pb, pout, static_cast<uint64_t>(kN)});
    const std::vector<float> out = download<float>(mem, pout, a.size());
    for (size_t i = 0; i < a.size(); ++i) VCHECK_EQ(out[i], a[i] * b[i]);
  }
  {
    MemoryManager mem(64ull << 20);
    const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(kN * 8);
    run(o, "pk_add", mem, {pa, pb, pout, static_cast<uint64_t>(kN)});
    const std::vector<float> out = download<float>(mem, pout, a.size());
    for (size_t i = 0; i < a.size(); ++i) VCHECK_EQ(out[i], a[i] + b[i]);
  }
}

VTEST(a_multiply_add_over_both_halves) {
  const amd::CodeObject o = object();
  const std::vector<float> a = pairs(1), b = pairs(2), c = pairs(3);
  MemoryManager mem(64ull << 20);
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pc = upload(mem, c), pout = mem.alloc(kN * 8);
  run(o, "pk_mix", mem, {pa, pb, pc, pout, static_cast<uint64_t>(kN)});
  const std::vector<float> out = download<float>(mem, pout, a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    const float want = std::fma(a[i], b[i], c[i]);
    if (out[i] != want)
      throw vtest::Failure("pk_mix[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(a_constant_serves_both_halves) {
  const amd::CodeObject o = object();
  const std::vector<float> a = pairs(5);
  MemoryManager mem(64ull << 20);
  const uint64_t pa = upload(mem, a), pout = mem.alloc(kN * 8);
  run(o, "pk_scale", mem, {pa, pout, static_cast<uint64_t>(kN)});
  const std::vector<float> out = download<float>(mem, pout, a.size());
  // Both halves are scaled and shifted by the same numbers: a high half that
  // read the register beside the constant would come out as something else
  // entirely, and the odd elements are where that would show.
  for (size_t i = 0; i < a.size(); ++i) {
    const float want = std::fma(a[i], 2.0f, 1.0f);
    if (out[i] != want)
      throw vtest::Failure("pk_scale[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST_MAIN
