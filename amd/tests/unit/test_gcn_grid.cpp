// How a kernel learns the shape of the grid it is part of.
//
// A work-item knows its own number from a register, but the size of the grid
// comes from the runtime: either in the arguments the compiler adds after the
// kernel's own (the metadata names them hidden_block_count_x and the rest),
// or in the packet a dispatch is described by. A kernel compiled from HIP
// reads blockDim and gridDim through the first of those, so a runtime that
// leaves them empty gives every such kernel zeros.
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
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/grid.gfx942.o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

std::vector<uint32_t> download(MemoryManager& mem, uint64_t p, size_t n) {
  std::vector<uint32_t> out(n);
  mem.read(p, out.data(), n * 4);
  return out;
}

// Runs one kernel over a grid of `groups` work-groups of `group` work-items,
// and returns what it wrote. These kernels index by the work-item's number
// within its group, so every group writes the same slots and what comes back
// is one group's worth.
std::vector<uint32_t> run(const amd::CodeObject& o, const char* name, MemoryManager& mem, uint32_t groups,
                          uint32_t group) {
  const amd::Kernel* k = amd::find_kernel(o, name);
  if (!k) throw vtest::Failure(std::string("no kernel named ") + name);
  const uint64_t out = mem.alloc(groups * group * 4);
  std::vector<uint8_t> args(k->kernarg_size, 0);
  for (uint32_t b = 0; b < 8; ++b) args[b] = static_cast<uint8_t>(out >> (8 * b));
  const uint64_t kernarg = mem.alloc(args.size());
  mem.write(kernarg, args.data(), args.size());

  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = kernarg;
  d.groups[0] = groups;
  d.group_size[0] = group;
  amd::execute(d, mem);
  return download(mem, out, group);
}

}  // namespace

VTEST(a_kernel_reads_the_grid_from_the_arguments_the_compiler_added) {
  const amd::CodeObject o = object();
  const amd::Kernel* k = amd::find_kernel(o, "from_implicit");
  VCHECK(k != nullptr);
  // The kernel takes one pointer; everything after it the compiler added.
  bool has_block_count = false, has_group_size = false;
  for (const amd::KernelArg& a : k->args) {
    has_block_count = has_block_count || a.kind == "hidden_block_count_x";
    has_group_size = has_group_size || a.kind == "hidden_group_size_x";
  }
  VCHECK(has_block_count);
  VCHECK(has_group_size);

  MemoryManager mem(64ull << 20);
  // out[t] = block_count_x * 1000 + group_size_x
  const std::vector<uint32_t> out = run(o, "from_implicit", mem, 3, 64);
  for (uint32_t v : out) VCHECK_EQ(v, 3u * 1000u + 64u);

  MemoryManager other(64ull << 20);
  const std::vector<uint32_t> again = run(o, "from_implicit", other, 1, 128);
  for (uint32_t v : again) VCHECK_EQ(v, 1u * 1000u + 128u);
}

VTEST(a_kernel_reads_the_grid_from_the_packet_the_dispatch_is_described_by) {
  const amd::CodeObject o = object();
  const amd::Kernel* k = amd::find_kernel(o, "from_packet");
  VCHECK(k != nullptr);
  VCHECK(k->dispatch_ptr);   // it asked for the packet, so the launch must give it one

  MemoryManager mem(64ull << 20);
  // out[t] = workgroup_size_x * 1000 + grid_size_x, and a grid's size is in
  // work-items, not work-groups.
  const std::vector<uint32_t> out = run(o, "from_packet", mem, 2, 64);
  for (uint32_t v : out) VCHECK_EQ(v, 64u * 1000u + 128u);

  MemoryManager other(64ull << 20);
  const std::vector<uint32_t> again = run(o, "from_packet", other, 5, 32);
  for (uint32_t v : again) VCHECK_EQ(v, 32u * 1000u + 160u);
}

VTEST_MAIN
