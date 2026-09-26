// Running CDNA kernels: the machine code of the fixture's two kernels, on a
// wavefront of 64 lanes, against device memory.
//
// These are the compiler's instructions, not a transcription of the source:
// what is checked is that the arithmetic, the divergence (EXEC), the LDS and
// the barrier come out with the answers the C the kernels were built from
// would give.
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vgpu/error.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

amd::CodeObject object() {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/vector_add.gfx942.o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

// Device memory holding `values`, and its address.
uint64_t upload(MemoryManager& mem, const std::vector<float>& values) {
  const uint64_t p = mem.alloc(values.size() * 4);
  mem.write(p, values.data(), values.size() * 4);
  return p;
}

std::vector<float> download(MemoryManager& mem, uint64_t p, size_t n) {
  std::vector<float> out(n);
  mem.read(p, out.data(), n * 4);
  return out;
}

// The kernarg segment: the arguments packed at the offsets the metadata gives.
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

}  // namespace

VTEST(a_kernel_adds_two_vectors_on_the_wavefront) {
  const amd::CodeObject o = object();
  const amd::Kernel* k = amd::find_kernel(o, "vector_add");
  MemoryManager mem(64ull << 20);

  // 1000 elements: not a multiple of the 256-work-item group, so the last
  // group runs with some lanes off and the kernel's bounds check has to hold.
  const int n = 1000;
  std::vector<float> a(n), b(n);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(2 * i + 1);
  }
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(n * 4);
  mem.fill(pout, reinterpret_cast<const uint8_t*>("\0\0\0\0"), 4, n * 4);

  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = kernargs(mem, *k, {pa, pb, pout, static_cast<uint64_t>(n)});
  d.groups[0] = (n + 255) / 256;
  d.group_size[0] = 256;
  const amd::DispatchStats stats = amd::execute(d, mem);

  // Four work-groups of 256 work-items: four waves each.
  VCHECK_EQ(stats.waves, 16u);
  VCHECK(stats.instructions > 100);
  const std::vector<float> out = download(mem, pout, n);
  for (int i = 0; i < n; ++i)
    if (out[i] != a[i] + b[i])
      throw vtest::Failure("out[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(a[i] + b[i]));
}

VTEST(the_lanes_past_the_end_write_nothing) {
  const amd::CodeObject o = object();
  const amd::Kernel* k = amd::find_kernel(o, "vector_add");
  MemoryManager mem(64ull << 20);

  // One group of 256 for 10 elements: 246 lanes must write nothing, which is
  // EXEC doing its work.
  const int n = 10, room = 256;
  std::vector<float> a(room, 1.0f), b(room, 2.0f), guard(room, -7.0f);
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = upload(mem, guard);

  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = kernargs(mem, *k, {pa, pb, pout, static_cast<uint64_t>(n)});
  d.groups[0] = 1;
  d.group_size[0] = 256;
  amd::execute(d, mem);

  const std::vector<float> out = download(mem, pout, room);
  for (int i = 0; i < room; ++i) {
    const float want = i < n ? 3.0f : -7.0f;
    if (out[i] != want)
      throw vtest::Failure("out[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(a_reduction_through_lds_and_a_barrier_sums_its_group) {
  const amd::CodeObject o = object();
  const amd::Kernel* k = amd::find_kernel(o, "reduce_sum");
  VCHECK(k != nullptr);
  VCHECK_EQ(k->group_segment, 1024u);
  MemoryManager mem(64ull << 20);

  const int n = 512;
  std::vector<float> in(n);
  for (int i = 0; i < n; ++i) in[i] = static_cast<float>(i % 7) + 1.0f;
  const float scale = 0.5f;
  const uint64_t pin = upload(mem, in), pout = mem.alloc(2 * 4);

  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = kernargs(mem, *k, {pin, pout, static_cast<uint64_t>(n), bits_of(scale)});
  d.groups[0] = 2;            // 512 elements, 256 per group
  d.group_size[0] = 256;
  const amd::DispatchStats stats = amd::execute(d, mem);
  // Every wave of every group reaches each barrier: the reduction has one
  // before the loop and one in each of its nine rounds.
  VCHECK(stats.barriers >= 2u * 4u * 9u);

  const std::vector<float> out = download(mem, pout, 2);
  for (int g = 0; g < 2; ++g) {
    float want = 0;
    for (int i = 0; i < 256; ++i) want += in[g * 256 + i] * scale;
    if (std::fabs(out[g] - want) > 1e-3f)
      throw vtest::Failure("group " + std::to_string(g) + " summed to " + std::to_string(out[g]) + ", not " +
                           std::to_string(want));
  }
}

// A work-group's LDS: 64 KB on gfx942, CDNA4's 160 KB on gfx950, as the
// code object's target says.
VTEST(a_gfx950_work_group_may_have_160_kb_of_lds_and_a_gfx942_one_64) {
  amd::CodeObject o = object();
  const amd::Kernel* k = amd::find_kernel(o, "vector_add");
  VCHECK(k != nullptr);
  MemoryManager mem(8ull << 20);
  const auto run = [&](uint32_t mach, uint32_t lds) -> std::string {
    o.mach = mach;
    amd::Dispatch d;
    d.object = &o;
    d.kernel = k;
    d.group_size[0] = 64;
    d.dynamic_lds = lds;
    // Arguments all zero: n is 0, and the kernel does nothing.
    const std::vector<uint8_t> zeros(k->kernarg_size + 64, 0);
    d.kernarg = mem.alloc(zeros.size());
    mem.write(d.kernarg, zeros.data(), zeros.size());
    try {
      amd::execute(d, mem);
    } catch (const std::exception& e) {
      return e.what();
    }
    return "ran";
  };
  VCHECK_EQ(run(0x4f, 100 * 1024), std::string("ran"));
  VCHECK_CONTAINS(run(0x4c, 100 * 1024), "past the 65536 a work-group has on gfx942");
  VCHECK_CONTAINS(run(0x4f, 161 * 1024), "past the 163840 a work-group has on gfx950");
}

VTEST(a_dispatch_a_kernel_cannot_take_is_refused) {
  const amd::CodeObject o = object();
  amd::Dispatch d;
  d.object = &o;
  d.kernel = amd::find_kernel(o, "vector_add");
  MemoryManager mem(1ull << 20);
  std::string what;
  try {
    d.group_size[0] = 4096;   // past max_flat_workgroup_size
    amd::execute(d, mem);
  } catch (const std::exception& e) {
    what = e.what();
  }
  VCHECK_CONTAINS(what, "this kernel allows");

  // An HSA packet is held only to the hardware's limit: past the kernel's
  // metadata is run, past the 1024 a work-group has is not.
  what.clear();
  try {
    d.kernel_limits = false;
    d.group_size[0] = 2048;
    amd::execute(d, mem);
  } catch (const std::exception& e) {
    what = e.what();
  }
  VCHECK_CONTAINS(what, "1024 a CDNA work-group has");
  d.kernel_limits = true;

  what.clear();
  try {
    amd::Dispatch none;
    amd::execute(none, mem);
  } catch (const std::exception& e) {
    what = e.what();
  }
  VCHECK_CONTAINS(what, "needs a kernel");
}

VTEST_MAIN
