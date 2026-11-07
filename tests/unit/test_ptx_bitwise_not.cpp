// Unary '~' on an integer constant in an operand: "and.b64 %rd3, %rd3, ~0x2;".
// ptxas takes the complement of the 64-bit constant and the instruction keeps
// the low bits of its type, so ~0x2 is 0xfffffffd as .b32 and
// 0xfffffffffffffffd as .b64 (checked by assembling with ptxas 13.0 for sm_80).
#include <cstring>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

namespace {
const char* kHeader = ".version 8.0\n.target sm_80\n.address_size 64\n";
std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}
struct Env {
  MemoryManager mem{1 << 20};
  DeviceProfile prof = load_gpu("nvidia/a100");
};

// Runs "<op> %b, %a, <imm>" with %a loaded from the buffer and returns %b.
// width is 32 or 64; type is the PTX type suffix of the operation.
uint64_t run(const std::string& op, const std::string& type, int width, const std::string& imm, uint64_t a) {
  const std::string r = width == 64 ? ".b64" : ".b32";
  const std::string ptx = std::string(kHeader) + ".visible .entry k(.param .u64 po){\n .reg .b64 %rd1; .reg ." +
                          type + " %a, %b;\n ld.param.u64 %rd1,[po]; cvta.to.global.u64 %rd1,%rd1;\n ld.global" +
                          r + " %a,[%rd1];\n " + op + "." + type + " %b, %a, " + imm + ";\n st.global" + r +
                          " [%rd1+8],%b; ret; }\n";
  Env e;
  auto m = ptx::parse(ptx);
  const uint64_t buf = e.mem.alloc(16);
  e.mem.store_scalar(buf, width / 8, a);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(buf)}, e.mem, e.prof);
  return e.mem.load_scalar(buf + 8, width / 8);
}
}  // namespace

// The line of the report.
VTEST(not_of_hex_literal_b64_is_the_64_bit_complement) {
  VCHECK_EQ(run("and", "b64", 64, "~0x2", 0xFFFFFFFFFFFFFFFFull), 0xFFFFFFFFFFFFFFFDull);
}

VTEST(not_of_hex_literal_b32_is_the_32_bit_complement) {
  VCHECK_EQ(run("and", "b32", 32, "~0x2", 0xFFFFFFFFull), 0xFFFFFFFDull);
  VCHECK_EQ(run("and", "b32", 32, "~0", 0x12345678ull), 0x12345678ull);
}

VTEST(not_nests_and_mixes_with_minus_and_parentheses) {
  VCHECK_EQ(run("and", "b32", 32, "~~5", 0xFFFFFFFFull), 5ull);
  VCHECK_EQ(run("and", "b32", 32, "~-1", 0xFFFFFFFFull), 0ull);
  VCHECK_EQ(run("and", "b64", 64, "~(0x2)", 0xFFFFFFFFFFFFFFFFull), 0xFFFFFFFFFFFFFFFDull);
  VCHECK_EQ(run("and", "b32", 32, "~5U", 0xFFFFFFFFull), 0xFFFFFFFAull);
}

// A 64-bit operation sees the whole 64-bit complement of a 32-bit-looking literal.
VTEST(not_of_32_bit_literal_in_b64_sets_the_high_word) {
  VCHECK_EQ(run("and", "b64", 64, "~0x80000000", 0xFFFFFFFFFFFFFFFFull), 0xFFFFFFFF7FFFFFFFull);
}

// Signed arithmetic: ~2 is -3.
VTEST(not_in_a_signed_add) {
  VCHECK_EQ(run("add", "s32", 32, "~0x2", 10), 7ull);
}

VTEST(not_of_a_float_or_a_register_is_refused) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 po){
 .reg .b32 %r<3>;
 and.b32 %r1, %r2, ~0f3F800000;
 ret; }
)";
  const auto err = VCAPTURE(vgpu::Error, ptx::parse(ptx));
  VCHECK_CONTAINS(err.what(), "'~' needs an integer constant");
}

VTEST_MAIN
