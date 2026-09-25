// Instructions chosen by hand, in kernels written in assembly
// (amd/tests/data/asm_*.s): the ones a compiler emits only now and then, so a
// C fixture could not be counted on to contain them. Each is checked against
// the same arithmetic done in 64 bits on the host.
#include <climits>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

amd::CodeObject object(const char* name) {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/" + name + ".gfx942.o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

// sopk(out, x): x + 32767 and whether it overflowed, x - 32768 and whether it
// overflowed, x * -3.
std::vector<int32_t> sopk(const amd::CodeObject& o, int32_t x) {
  MemoryManager mem(16ull << 20);
  const uint64_t out = mem.alloc(5 * 4);
  const amd::Kernel* k = amd::find_kernel(o, "sopk");
  if (!k) throw vtest::Failure("no kernel named sopk");
  std::vector<uint8_t> args(k->kernarg_size, 0);
  for (int b = 0; b < 8; ++b) args[b] = static_cast<uint8_t>(out >> (8 * b));
  for (int b = 0; b < 4; ++b) args[8 + b] = static_cast<uint8_t>(static_cast<uint32_t>(x) >> (8 * b));
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(args.size());
  mem.write(d.kernarg, args.data(), args.size());
  d.group_size[0] = 64;
  amd::execute(d, mem);
  std::vector<int32_t> r(5);
  mem.read(out, r.data(), 5 * 4);
  return r;
}

}  // namespace

VTEST(a_16_bit_constant_is_added_signed_and_scc_says_when_the_sum_overflowed) {
  const amd::CodeObject o = object("asm_sopk");
  for (int32_t x : {1, -5, INT_MAX, INT_MIN, 0x7fff8000}) {
    const std::vector<int32_t> r = sopk(o, x);
    const int64_t up = int64_t{x} + 32767, down = int64_t{x} - 32768;
    VCHECK_EQ(r[0], static_cast<int32_t>(static_cast<uint32_t>(up)));
    VCHECK_EQ(r[1], up > INT_MAX ? 1 : 0);
    VCHECK_EQ(r[2], static_cast<int32_t>(static_cast<uint32_t>(down)));
    VCHECK_EQ(r[3], down < INT_MIN ? 1 : 0);
    VCHECK_EQ(r[4], static_cast<int32_t>(static_cast<uint32_t>(int64_t{x} * -3)));
  }
}

VTEST_MAIN
