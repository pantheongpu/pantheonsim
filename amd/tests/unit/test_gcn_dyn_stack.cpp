// Issue #585: a kernel built with .amdhsa_uses_dynamic_stack 1 (kernel_code_properties bit 11) says so in the code object, and a launch
// that gives it more scratch than its static size (Dispatch::private_segment, which a HIP launch and an AQL packet set) has that much: a
// work-item's access past the static size but within what the launch gave is its own memory, and one past both is still refused. The
// kernels are amd/tests/data/asm_dyn_stack.s (gfx942).
#include <cstdint>
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
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/asm_dyn_stack.gfx942.o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

// Runs `name` writing at byte `at` of each work-item's scratch with `reserve` bytes asked for by the launch; the error text, or "" and
// out[L] == 1000 + L checked.
std::string run(const amd::CodeObject& o, const char* name, uint32_t at, uint32_t reserve) {
  const amd::Kernel* k = amd::find_kernel(o, name);
  if (!k) throw vtest::Failure(std::string("no kernel named ") + name);
  MemoryManager mem(16ull << 20);
  const uint64_t out = mem.alloc(256);
  std::vector<uint8_t> args(16, 0);
  std::memcpy(&args[0], &out, 8);
  std::memcpy(&args[8], &at, 4);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(args.size());
  mem.write(d.kernarg, args.data(), args.size());
  d.group_size[0] = 64;
  d.private_segment = reserve;
  try {
    amd::execute(d, mem);
  } catch (const Error& e) {
    return e.what();
  }
  std::vector<uint32_t> got(64);
  mem.read(out, got.data(), 256);
  for (uint32_t l = 0; l < 64; ++l)
    if (got[l] != 1000 + l) return "lane " + std::to_string(l) + " read back " + std::to_string(got[l]);
  return "";
}

}  // namespace

VTEST(the_code_object_says_which_kernels_use_a_dynamic_stack) {
  const amd::CodeObject o = object();
  const amd::Kernel* dyn = amd::find_kernel(o, "dyn_stack");
  const amd::Kernel* fixed = amd::find_kernel(o, "fixed_stack");
  VCHECK(dyn && fixed);
  VCHECK(dyn->uses_dynamic_stack);
  VCHECK(!fixed->uses_dynamic_stack);
  VCHECK_EQ(dyn->private_segment, 16u);
}

VTEST(a_launch_that_reserves_more_scratch_than_the_static_size_has_it) {
  const amd::CodeObject o = object();
  VCHECK_EQ(run(o, "dyn_stack", 8, 0), std::string());              // within the static 16 bytes
  VCHECK(run(o, "dyn_stack", 1000, 0).find("past the") != std::string::npos);   // past it, with nothing more reserved
  VCHECK_EQ(run(o, "dyn_stack", 1000, 1024), std::string());        // the 1024 bytes ROCm's HIP gives by default
  VCHECK(run(o, "dyn_stack", 1024, 1024).find("past the") != std::string::npos);
  VCHECK_EQ(run(o, "fixed_stack", 1000, 1024), std::string());      // (the reserve is the launch's, whatever the kernel says)
  VCHECK_EQ(run(o, "dyn_stack", 8, 4), std::string());              // less than the static size: the static size
}

VTEST_MAIN
