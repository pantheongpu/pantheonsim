// Where a kernel's values live besides registers, run and checked against
// what the C means: a private array too big to keep in registers (scratch),
// LDS with an atomic, a value read from another lane, and a pointer that may
// be either LDS or device memory.
//
// The kernels are amd/tests/data/memory.c, compiled for gfx942.
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
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/memory.gfx942.o";
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

const amd::Kernel& kernel(const amd::CodeObject& o, const char* name) {
  const amd::Kernel* k = amd::find_kernel(o, name);
  if (!k) throw vtest::Failure(std::string("no kernel named ") + name);
  return *k;
}

void run(const amd::CodeObject& o, const char* name, MemoryManager& mem, const std::vector<uint64_t>& args) {
  amd::Dispatch d;
  d.object = &o;
  d.kernel = &kernel(o, name);
  d.kernarg = kernargs(mem, *d.kernel, args);
  d.groups[0] = 1;
  d.group_size[0] = 64;
  amd::execute(d, mem);
}

}  // namespace

VTEST(a_private_array_too_big_for_registers_spills_to_scratch) {
  const amd::CodeObject o = object();
  // The kernel keeps 32 floats per work-item and indexes them at run time, so
  // the compiler puts them in the work-item's own memory.
  VCHECK(kernel(o, "scratch").private_segment > 0);

  MemoryManager mem(64ull << 20);
  const int n = 64;
  std::vector<int> idx(n);
  for (int i = 0; i < n; ++i) idx[i] = i * 13 + 5;
  const uint64_t pidx = upload(mem, idx), pout = mem.alloc(n * 4);
  run(o, "scratch", mem, {pidx, pout, static_cast<uint64_t>(n)});

  const std::vector<float> out = download<float>(mem, pout, n);
  for (int i = 0; i < n; ++i) {
    float want = 0;
    for (int k = 0; k < 8; ++k) want += static_cast<float>((idx[(i + k) % n] & 31) * i);
    if (out[i] != want)
      throw vtest::Failure("scratch[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(an_atomic_in_lds_collects_every_lane_that_shares_a_slot) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const int n = 64;
  std::vector<float> in(n);
  for (int i = 0; i < n; ++i) in[i] = static_cast<float>(i % 10);
  const uint64_t pin = upload(mem, in), pout = mem.alloc(n * 4);
  run(o, "lds_atomic", mem, {pin, pout, static_cast<uint64_t>(n)});

  // Eight slots, each the sum of the eight lanes that share it.
  int want[8] = {};
  for (int i = 0; i < n; ++i) want[i & 7] += static_cast<int>(in[i]);
  const std::vector<float> out = download<float>(mem, pout, n);
  for (int i = 0; i < n; ++i) {
    // The kernel reads the slot back as a float, and what the atomic added
    // was an integer, so the bits are an integer's.
    int seen;
    std::memcpy(&seen, &out[i], 4);
    if (seen != want[i & 7])
      throw vtest::Failure("lds slot " + std::to_string(i & 7) + " holds " + std::to_string(seen) + ", not " +
                           std::to_string(want[i & 7]));
  }
}

VTEST(a_lane_reads_what_another_lane_holds) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const int n = 64;
  std::vector<int> in(n);
  for (int i = 0; i < n; ++i) in[i] = 1000 + i;
  const uint64_t pin = upload(mem, in), pout = mem.alloc(n * 4);
  run(o, "shuffle", mem, {pin, pout, static_cast<uint64_t>(n)});

  const std::vector<int> out = download<int>(mem, pout, n);
  for (int i = 0; i < n; ++i) {
    // Each lane takes its neighbour's value, and adds the first lane's.
    const int want = in[i ^ 1] + in[0];
    if (out[i] != want)
      throw vtest::Failure("shuffle[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(a_pointer_that_may_be_either_memory_reaches_the_right_one) {
  const amd::CodeObject o = object();
  const int n = 64;

  // Through device memory: the kernel writes each work-item's number and
  // reads it back.
  {
    MemoryManager mem(64ull << 20);
    const uint64_t pg = mem.alloc(n * 4), pout = mem.alloc(n * 4);
    run(o, "generic_ptr", mem, {pg, 0, pout});
    const std::vector<float> g = download<float>(mem, pg, n), out = download<float>(mem, pout, n);
    for (int i = 0; i < n; ++i) {
      VCHECK_EQ(g[i], static_cast<float>(i));
      VCHECK_EQ(out[i], static_cast<float>(i) * 2.0f);
    }
  }
  // Through LDS: the same pointer, in the aperture that says it is shared.
  // Sixteen slots, so the last work-item to write a slot is what it holds.
  {
    MemoryManager mem(64ull << 20);
    const uint64_t pg = mem.alloc(n * 4), pout = mem.alloc(n * 4);
    mem.fill(pg, reinterpret_cast<const uint8_t*>("\xff"), 1, n * 4);
    run(o, "generic_ptr", mem, {pg, 1, pout});
    const std::vector<float> out = download<float>(mem, pout, n);
    for (int i = 0; i < n; ++i) VCHECK_EQ(out[i], static_cast<float>(i) * 2.0f);
    // Nothing reached device memory this time.
    const std::vector<uint32_t> g = download<uint32_t>(mem, pg, n);
    for (int i = 0; i < n; ++i) VCHECK_EQ(g[i], 0xFFFFFFFFu);
  }
}

VTEST_MAIN
