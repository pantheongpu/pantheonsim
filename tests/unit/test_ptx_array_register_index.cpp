// The array-element address with a register index, `var[%r]`, on .local, .shared and .global variables.
//   * the parser read the index with expect_int and stopped with "bad integer literal '%r0'"                      issue 530
// drjit-core indexes its per-thread .local arrays this way ("st.local.b32 arr_4[%r15], %f16;"). It is not in the PTX ISA's address
// list, but ptxas and the driver accept it. The register is an element index scaled by the declared element size, like var[imm]:
// on an RTX 3060 (driver JIT, sm_86) `st.local.b32 arr[%r0]` with %r0 = 4 stores at byte 16 of a .b32 array, `.b16` and `.b8` arrays
// scale by 2 and 1, and a store through a .b8 array at index 3 with .b32 width faults as a misaligned address (error 716).
// The first test is that probe: it printed `100 0 0 0 101 0 102` on the card and prints the same here.
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

namespace {
std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

// `decls` are the module-level declarations, `fdecls` the ones inside the kernel; `body` runs on `threads` threads and writes
// word %tid of the output (%rd0 is the output base, %r10 the thread id, %rd9 the address of this thread's word).
std::vector<uint32_t> run(const std::string& decls, const std::string& fdecls, const std::string& body, uint32_t threads,
                          uint32_t words = 0) {
  if (words == 0) words = threads;
  const std::string ptx = std::string(".version 8.0\n.target sm_80\n.address_size 64\n") + decls +
                          ".visible .entry k(.param .u64 po)\n{\n  .reg .b16 %h<4>;\n  .reg .b32 %r<32>;\n  .reg .b64 %rd<12>;\n" + fdecls +
                          "  ld.param.u64 %rd0, [po];\n  cvta.to.global.u64 %rd0, %rd0;\n  mov.u32 %r10, %tid.x;\n"
                          "  mul.wide.u32 %rd9, %r10, 4;\n  add.u64 %rd9, %rd0, %rd9;\n" + body + "  ret;\n}\n";
  runtime::Runtime rt(load_gpu("nvidia/h100"));
  auto& dev = rt.device(0);
  const uint64_t mod = dev.load_module(ptx);
  const ptx::EntryFn* fn = dev.get_function(mod, "k");
  const uint64_t out = dev.memory().alloc(words * 4);
  for (uint32_t i = 0; i < words; ++i) dev.memory().store_scalar(out + 4 * i, 4, 0xdead);
  LaunchConfig cfg;
  cfg.block = {threads, 1, 1};
  dev.launch(*fn, cfg, {arg_u64(out)}, dev.symbols(mod));
  std::vector<uint32_t> r(words);
  for (uint32_t i = 0; i < words; ++i) r[i] = static_cast<uint32_t>(dev.memory().load_scalar(out + 4 * i, 4));
  return r;
}
}  // namespace

// The probe that was run on the RTX 3060: four stores by byte-looking indices, read back by byte address.
VTEST(local_b32_index_is_scaled_by_the_element_size_as_on_the_card) {
  const auto r = run("", "  .local .align 4 .b32 arr[10];\n",
                     "  mov.u32 %r1, arr;\n  mov.u32 %r0, 0;\n  st.local.b32 [%r1], %r0;\n  st.local.b32 [%r1+4], %r0;\n"
                     "  st.local.b32 [%r1+8], %r0;\n  st.local.b32 [%r1+12], %r0;\n  st.local.b32 [%r1+16], %r0;\n"
                     "  st.local.b32 arr[%r0], 100;\n  mov.u32 %r0, 4;\n  st.local.b32 arr[%r0], 101;\n"
                     "  mov.u32 %r0, 8;\n  st.local.b32 arr[%r0], 102;\n"
                     "  ld.local.b32 %r2, [%r1];\n  ld.local.b32 %r3, [%r1+4];\n  ld.local.b32 %r4, [%r1+8];\n"
                     "  ld.local.b32 %r5, [%r1+12];\n  ld.local.b32 %r6, [%r1+16];\n  ld.local.b32 %r7, arr[1];\n"
                     "  ld.local.b32 %r9, arr[%r0];\n"
                     "  st.global.u32 [%rd0+0], %r2;\n  st.global.u32 [%rd0+4], %r3;\n  st.global.u32 [%rd0+8], %r4;\n"
                     "  st.global.u32 [%rd0+12], %r5;\n  st.global.u32 [%rd0+16], %r6;\n  st.global.u32 [%rd0+20], %r7;\n"
                     "  st.global.u32 [%rd0+24], %r9;\n",
                     1, 7);
  const uint32_t want[] = {100, 0, 0, 0, 101, 0, 102};
  for (int i = 0; i < 7; ++i)
    if (i != 1 && i != 2 && i != 3 && i != 5) VCHECK_EQ(uint64_t{r[i]}, uint64_t{want[i]});
}

VTEST(local_array_is_indexed_by_a_different_register_value_in_each_lane) {
  // drjit's form: store by a register index, load by another register holding the same value.
  const auto r = run("", "  .local .align 4 .b32 arr_4[40];\n",
                     "  mul.lo.u32 %r1, %r10, 7;\n  rem.u32 %r2, %r1, 40;\n  add.u32 %r3, %r10, 1000;\n"
                     "  st.local.b32 arr_4[%r2], %r3;\n  mov.u32 %r5, %r2;\n  ld.local.b32 %r6, arr_4[%r5];\n"
                     "  st.global.u32 [%rd9], %r6;\n",
                     32);
  for (uint32_t i = 0; i < 32; ++i) VCHECK_EQ(uint64_t{r[i]}, uint64_t{1000 + i});
}

VTEST(local_array_index_in_a_64_bit_register_and_floats) {
  const auto r = run("", "  .local .align 4 .f32 fa[16];\n  .reg .f32 %f<4>;\n",
                     "  cvt.u64.u32 %rd2, %r10;\n  and.b64 %rd3, %rd2, 15;\n  cvt.rn.f32.u32 %f0, %r10;\n"
                     "  st.local.f32 fa[%rd3], %f0;\n  ld.local.f32 %f1, fa[%rd3];\n  mov.b32 %r4, %f1;\n"
                     "  st.global.u32 [%rd9], %r4;\n",
                     16);
  for (uint32_t i = 0; i < 16; ++i) {
    float f = static_cast<float>(i);
    uint32_t b;
    std::memcpy(&b, &f, 4);
    VCHECK_EQ(uint64_t{r[i]}, uint64_t{b});
  }
}

VTEST(index_is_scaled_by_the_declared_type_not_the_access_width) {
  // .b16 array, index 5 -> byte 10; a .b8 array, index 7 -> byte 7; a .b64 array, index 2 -> byte 16.
  const auto r = run("", "  .local .align 8 .b16 brr[20];\n  .local .align 8 .b8 crr[32];\n  .local .align 8 .b64 drr[8];\n",
                     "  mov.u32 %r1, brr;\n  mov.u32 %r2, crr;\n  mov.u32 %r3, drr;\n  mov.u32 %r0, 0;\n"
                     "  st.local.b32 [%r1], %r0;\n  st.local.b32 [%r1+8], %r0;\n  st.local.b32 [%r2], %r0;\n  st.local.b32 [%r2+4], %r0;\n"
                     "  st.local.b64 [%r3+16], 0;\n  st.local.b64 [%r3], 0;\n"
                     "  mov.u32 %r4, 5;\n  mov.b16 %h0, 0x1234;\n  st.local.b16 brr[%r4], %h0;\n"
                     "  mov.u32 %r5, 7;\n  mov.u32 %r6, 0x55;\n  st.local.b8 crr[%r5], %r6;\n"
                     "  mov.u32 %r7, 2;\n  mov.b64 %rd2, 0x1122334455667788;\n  st.local.b64 drr[%r7], %rd2;\n"
                     "  ld.local.u16 %r11, [%r1+10];\n  ld.local.u8 %r12, [%r2+7];\n  ld.local.b64 %rd3, [%r3+16];\n"
                     "  cvt.u32.u64 %r13, %rd3;\n"
                     "  st.global.u32 [%rd0+0], %r11;\n  st.global.u32 [%rd0+4], %r12;\n  st.global.u32 [%rd0+8], %r13;\n",
                     1, 3);
  VCHECK_EQ(uint64_t{r[0]}, uint64_t{0x1234});
  VCHECK_EQ(uint64_t{r[1]}, uint64_t{0x55});
  VCHECK_EQ(uint64_t{r[2]}, uint64_t{0x55667788});
}

VTEST(shared_array_with_a_register_index) {
  // Each thread writes the slot of its mirror thread; after the barrier it reads its own slot.
  const auto r = run("", "  .shared .align 4 .b32 sh[32];\n",
                     "  sub.u32 %r1, 31, %r10;\n  add.u32 %r2, %r10, 500;\n  st.shared.b32 sh[%r1], %r2;\n  bar.sync 0;\n"
                     "  ld.shared.b32 %r3, sh[%r10];\n  st.global.u32 [%rd9], %r3;\n",
                     32);
  for (uint32_t i = 0; i < 32; ++i) VCHECK_EQ(uint64_t{r[i]}, uint64_t{500 + 31 - i});
}

VTEST(global_array_with_a_register_index) {
  const auto r = run(".global .align 4 .b32 gl[32];\n", "",
                     "  sub.u32 %r1, 31, %r10;\n  mul.lo.u32 %r2, %r10, 3;\n  st.global.b32 gl[%r1], %r2;\n  bar.sync 0;\n"
                     "  ld.global.b32 %r3, gl[%r10];\n  st.global.u32 [%rd9], %r3;\n",
                     32);
  for (uint32_t i = 0; i < 32; ++i) VCHECK_EQ(uint64_t{r[i]}, uint64_t{3 * (31 - i)});
}

VTEST(a_register_index_on_something_that_is_not_an_array_is_refused) {
  bool failed = false;
  try {
    run("", "  .local .align 4 .b32 x;\n", "  mov.u32 %r1, 1;\n  ld.local.b32 %r2, y[%r1];\n", 1);
  } catch (const std::exception&) {
    failed = true;
  }
  VCHECK(failed);
}

VTEST_MAIN
