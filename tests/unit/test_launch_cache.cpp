// What a device remembers between launches (the table of every loaded kernel that a launch hands its device-side
// launches, and the module and function lookups): the answers must be what a device that remembers nothing gives
// (VGPU_LAUNCH_CACHE=0), through loads, unloads, lazily parsed modules and modules with kernels of the same name.
//
// The check that matters is a kernel in a module loaded AFTER a launch has built the table: its child launch only
// works if loading the module dropped the table. Each scenario runs on a device that remembers and on one that does
// not, and the two must agree with each other and with the values the kernels write.
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

const char* kDecls = R"(
.extern .func (.param .b64 func_retval0) __cudaCDP2GetParameterBufferV2
(.param .b64 f, .param .align 4 .b8 g[12], .param .align 4 .b8 b[12], .param .b32 s);
.extern .func (.param .b32 func_retval0) __cudaCDP2LaunchDeviceV2
(.param .b64 buf, .param .b64 stream);
)";

// A module whose `parent` has each of four threads launch an 8-thread `child` that writes index + k into
// out[index] (the dynamic-parallelism pattern of test_dynpar.cpp), and whose `same` writes k2 in out[0]. Every
// module has the same kernel names with different constants, so a lookup that found the wrong module's kernel
// writes the wrong numbers.
// `child_ref` is what the parent names as the kernel to launch: its own child, or a literal address (of a
// kernel in another module).
std::string module_text(unsigned k, unsigned k2, const std::string& child_ref = "child") {
  const std::string K = std::to_string(k), K2 = std::to_string(k2);
  return ".version 8.3\n.target sm_90\n.address_size 64\n" + std::string(kDecls) + R"(
.visible .entry child(.param .u64 out, .param .u32 base)
{
    .reg .b32 %r<5>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    ld.param.u32 %r1, [base];
    mov.u32 %r2, %tid.x;
    add.u32 %r3, %r1, %r2;
    add.u32 %r4, %r3, )" + K + R"(;
    mul.wide.u32 %rd2, %r3, 4;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r4;
    ret;
}
.visible .entry same(.param .u64 out)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, )" + K2 + R"(;
    st.global.u32 [%rd1], %r1;
    ret;
}
.visible .entry parent(.param .u64 out)
{
    .reg .pred %p<2>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<24>;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, %tid.x;
    setp.ge.u32 %p1, %r1, 4;
    @%p1 bra DONE;
    mov.u32 %r20, 1;
    mov.u32 %r21, 8;
    mov.u64 %rd19, )" + child_ref + R"(;
    {
    .param .b64 param0;
    st.param.b64 [param0+0], %rd19;
    .param .align 4 .b8 param1[12];
    st.param.b32 [param1+0], %r20;
    st.param.b32 [param1+4], 1;
    st.param.b32 [param1+8], 1;
    .param .align 4 .b8 param2[12];
    st.param.b32 [param2+0], %r21;
    st.param.b32 [param2+4], 1;
    st.param.b32 [param2+8], 1;
    .param .b32 param3;
    st.param.b32 [param3+0], 0;
    .param .b64 retval0;
    call.uni (retval0), __cudaCDP2GetParameterBufferV2, (param0, param1, param2, param3);
    ld.param.b64 %rd20, [retval0+0];
    }
    st.u64 [%rd20], %rd1;
    shl.b32 %r2, %r1, 3;
    st.u32 [%rd20+8], %r2;
    {
    .param .b64 param0;
    st.param.b64 [param0+0], %rd20;
    .param .b64 param1;
    st.param.b64 [param1+0], 0;
    .param .b32 retval0;
    call.uni (retval0), __cudaCDP2LaunchDeviceV2, (param0, param1);
    ld.param.b32 %r22, [retval0+0];
    }
DONE:
    ret;
}
)";
}

// What a launch of `name` in `mod` wrote into a fresh 32-word buffer, as text; or the error it raised.
std::string launch(runtime::Device& dev, uint64_t mod, const char* name) {
  try {
    const ptx::EntryFn* fn = dev.get_function(mod, name);
    const uint64_t out = dev.memory().alloc(32 * 4);
    for (uint32_t i = 0; i < 32; ++i) dev.memory().store_scalar(out + i * 4, 4, 0);
    exec::LaunchConfig cfg;
    cfg.block = {32, 1, 1};
    dev.launch(*fn, cfg, {arg_u64(out)}, dev.symbols(mod));
    std::string s;
    for (uint32_t i = 0; i < 32; ++i) s += std::to_string(dev.memory().load_scalar(out + i * 4, 4)) + " ";
    dev.memory().free(out);
    return s;
  } catch (const Error& e) {
    return std::string("error: ") + e.what();
  }
}

// The numbers module_text(k, k2)'s parent writes: 32 words, word i = i + k.
std::string parent_writes(unsigned k) {
  std::string s;
  for (unsigned i = 0; i < 32; ++i) s += std::to_string(i + k) + " ";
  return s;
}
std::string same_writes(unsigned k2) {
  std::string s = std::to_string(k2) + " ";
  for (unsigned i = 1; i < 32; ++i) s += "0 ";
  return s;
}

// One sequence of loads, launches and unloads; returns what every step answered.
std::vector<std::string> scenario(bool cache) {
  runtime::Device dev(load_gpu("nvidia/h100"), 0, nullptr);
  dev.set_launch_cache(cache);
  std::vector<std::string> log;
  const uint64_t a = dev.load_module(module_text(100, 1001));
  log.push_back(launch(dev, a, "parent"));    // builds the table
  log.push_back(launch(dev, a, "same"));
  // A module loaded after a launch: its parent's child launch needs it in the table.
  const uint64_t b = dev.load_module(module_text(200, 2002));
  log.push_back(launch(dev, b, "parent"));
  log.push_back(launch(dev, b, "same"));
  log.push_back(launch(dev, a, "parent"));    // the first module still gets its own child
  log.push_back(launch(dev, a, "same"));
  // Unloaded: asking for its kernels is an error, and the other module is unaffected.
  dev.unload_module(a);
  log.push_back(launch(dev, a, "parent"));
  log.push_back(launch(dev, b, "parent"));
  // A lazily parsed module (its kernels are parsed when first asked for, and are only then in the table).
  ::setenv("VGPU_LAZY_MODULE_BYTES", "0", 1);
  const uint64_t c = dev.load_module(module_text(300, 3003));
  ::unsetenv("VGPU_LAZY_MODULE_BYTES");
  log.push_back(launch(dev, b, "parent"));    // builds the table while c has parsed nothing
  log.push_back(launch(dev, c, "parent"));    // parses c's parent and child: the table must change
  log.push_back(launch(dev, c, "same"));
  log.push_back(launch(dev, b, "parent"));
  // The same text again, a module of its own with its own kernels.
  const uint64_t d = dev.load_module(module_text(100, 1001));
  log.push_back(launch(dev, d, "parent"));
  log.push_back(launch(dev, d, "same"));
  dev.unload_module(b);
  dev.unload_module(c);
  log.push_back(launch(dev, d, "parent"));
  dev.reset();
  log.push_back(launch(dev, d, "parent"));    // everything unloaded
  const uint64_t e = dev.load_module(module_text(400, 4004));
  log.push_back(launch(dev, e, "parent"));
  // A kernel launching a kernel of another module by its address: fine while that module is loaded, and a
  // refused launch once it is not (a table that still held the unloaded module's kernels would run them).
  const uint64_t g = dev.load_module(module_text(600, 6006));
  const std::string g_child = std::to_string(dev.symbols(g)->at("child"));
  const uint64_t x = dev.load_module(module_text(500, 5005, g_child));
  log.push_back(launch(dev, x, "parent"));
  dev.unload_module(g);
  log.push_back(launch(dev, x, "parent"));
  return log;
}

}  // namespace

VTEST(a_device_that_remembers_answers_as_one_that_does_not) {
  const std::vector<std::string> with = scenario(true);
  const std::vector<std::string> without = scenario(false);
  VCHECK_EQ(with.size(), without.size());
  for (size_t i = 0; i < with.size(); ++i) VCHECK_EQ(with[i], without[i]);
}

VTEST(each_launch_runs_the_kernel_of_the_module_it_names) {
  const std::vector<std::string> log = scenario(true);
  VCHECK_EQ(log[0], parent_writes(100));
  VCHECK_EQ(log[1], same_writes(1001));
  VCHECK_EQ(log[2], parent_writes(200));   // loaded after the table was built
  VCHECK_EQ(log[3], same_writes(2002));
  VCHECK_EQ(log[4], parent_writes(100));
  VCHECK_EQ(log[5], same_writes(1001));
  VCHECK_CONTAINS(log[6], "is not loaded on device");   // unloaded
  VCHECK_EQ(log[7], parent_writes(200));
  VCHECK_EQ(log[8], parent_writes(200));
  VCHECK_EQ(log[9], parent_writes(300));   // lazy
  VCHECK_EQ(log[10], same_writes(3003));
  VCHECK_EQ(log[11], parent_writes(200));
  VCHECK_EQ(log[12], parent_writes(100));
  VCHECK_EQ(log[13], same_writes(1001));
  VCHECK_EQ(log[14], parent_writes(100));
  VCHECK_CONTAINS(log[15], "is not loaded on device");   // after reset
  VCHECK_EQ(log[16], parent_writes(400));
  VCHECK_EQ(log[17], parent_writes(600));   // x's parent launched g's child, by address
  VCHECK_CONTAINS(log[18], "not the address of a kernel loaded");   // g is gone
}

// Looking a kernel up by name finds the first of that name, and the module it is asked of.
VTEST(function_lookup_is_per_module_and_by_name) {
  runtime::Device dev(load_gpu("nvidia/h100"), 0, nullptr);
  const uint64_t a = dev.load_module(module_text(1, 1));
  const uint64_t b = dev.load_module(module_text(2, 2));
  const ptx::EntryFn* pa = dev.get_function(a, "parent");
  const ptx::EntryFn* pb = dev.get_function(b, "parent");
  VCHECK(pa != nullptr && pb != nullptr && pa != pb);
  VCHECK(dev.get_function(a, "parent") == pa);
  VCHECK(dev.get_function(b, "parent") == pb);
  VCHECK(dev.symbols(a) != dev.symbols(b));
  VCHECK(dev.symbols(a) != nullptr && dev.symbols(a)->count("child") == 1);
  const uint64_t before = dev.unload_epoch();
  dev.unload_module(a);
  VCHECK(dev.unload_epoch() != before);
  VCHECK(dev.get_function(b, "parent") == pb);   // the other module did not move
  auto gone = VCAPTURE(Error, dev.get_function(a, "parent"));
  VCHECK(gone.code() == Err::NotFound);
  auto gone2 = VCAPTURE(Error, dev.symbols(a));
  VCHECK(gone2.code() == Err::NotFound);
  auto none = VCAPTURE(Error, dev.get_function(b, "nothing"));
  VCHECK(none.code() == Err::NotFound);
  VCHECK_CONTAINS(none.what(), "parent");
}

VTEST_MAIN
