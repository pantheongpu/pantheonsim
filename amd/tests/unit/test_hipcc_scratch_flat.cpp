// A stack slot reached through the private segment buffer and through a flat pointer is one slot.
//
// Unoptimized code keeps its locals on the stack. It stores them through the private segment buffer
// (`buffer_store_dword v, off, s[0:3], s33 offset:N`, the stack pointer scaled by the wave's size) and, when it passes the
// address of one to a function that is not inlined, makes a flat pointer from the same stack pointer
// (`src_private_base + (s33 >> 5) + N` in a wave32 kernel, `>> 6` in a wave64 one) that the function loads through. The
// buffer's address is unswizzled with the wave's size as its index stride; with 64 for a wave32 kernel the store landed at
// half the offset the pointer names, and the function read another slot: an RX 6800 kernel built -O0 returned wrong sums or
// stopped on an address that is no allocation's (found by tools/numerics/varity_cases.cu built -O0 for gfx1030).
//
// The kernel is amd/tests/hipcc/scratch_flat.cpp, built -O0 by hipcc 6.4 for gfx1030 (wave32). (A wave64 kernel's stride was
// right and is what it was: the digests of test_gcn_memory_same_bits.cpp hold every scratch instruction of the gfx942 corpus.)
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

amd::CodeObject object(const char* arch) {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/hipcc/scratch_flat.O0." + arch + ".o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

std::vector<float> run(const char* arch, uint32_t lanes) {
  const amd::CodeObject o = object(arch);
  const amd::Kernel* k = amd::find_kernel(o, "_Z12scratch_flatPKfPf");
  if (!k) throw vtest::Failure("no kernel scratch_flat");
  MemoryManager mem(64ull << 20);
  std::vector<float> in(lanes), out(lanes, -1.0f);
  for (uint32_t i = 0; i < lanes; ++i) in[i] = 1.0f + 0.5f * static_cast<float>(i);
  const uint64_t pin = mem.alloc(lanes * 4), pout = mem.alloc(lanes * 4);
  mem.write(pin, in.data(), lanes * 4);
  mem.write(pout, out.data(), lanes * 4);
  std::vector<uint8_t> buf(k->kernarg_size, 0);
  const uint64_t values[2] = {pin, pout};
  for (size_t i = 0; i < 2 && i < k->args.size(); ++i) std::memcpy(buf.data() + k->args[i].offset, &values[i], 8);
  const uint64_t kernarg = mem.alloc(buf.size());
  mem.write(kernarg, buf.data(), buf.size());
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = kernarg;
  d.groups[0] = 1;
  d.group_size[0] = lanes;
  amd::execute(d, mem);
  mem.read(pout, out.data(), lanes * 4);
  return out;
}

// weigh(a, b, c, t) is a + 2 b + 4 c + 8 t[3]. first = x + 2 (x + 1) + 4 (x + 2) + 8 (x + 3) = 15 x + 34;
// second = (x + 1) + 2 (x + 2) + 4 x + 8 (first + 3) = 127 x + 301. Every term is a small multiple of 0.5: exact.
float expected(uint32_t i) { return 142.0f * (1.0f + 0.5f * static_cast<float>(i)) + 335.0f; }

}  // namespace

VTEST(a_wave32_kernel_reads_its_stack_slots_through_a_flat_pointer) {
  for (uint32_t lanes : {32u, 64u, 1u, 33u}) {
    const std::vector<float> out = run("gfx1030", lanes);
    for (uint32_t i = 0; i < lanes; ++i) VCHECK_EQ(out[i], expected(i));
  }
}


VTEST_MAIN
