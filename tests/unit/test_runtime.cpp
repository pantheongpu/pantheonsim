// Tests for the runtime facade: devices, modules, function lookup.
#include "vgpu/runtime/runtime.hpp"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/exec/launch.hpp"

#include "vgpu/error.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;

VTEST(multi_device_enumeration) {
  runtime::Runtime rt(load_gpu("nvidia/h100"), 4);
  VCHECK_EQ(rt.device_count(), 4);
  VCHECK_EQ(rt.device(3).ordinal(), 3);
  auto err = VCAPTURE(Error, rt.device(4));
  VCHECK(err.code() == Err::InvalidValue);
}

VTEST(module_lifecycle_and_lookup) {
  runtime::Runtime rt(load_gpu("nvidia/h100"));
  auto& dev = rt.device(0);
  uint64_t mod = dev.load_module(
      ".version 8.3\n.target sm_90\n.address_size 64\n.visible .entry mykernel() { ret; }\n");
  VCHECK(dev.get_function(mod, "mykernel") != nullptr);

  auto err = VCAPTURE(Error, dev.get_function(mod, "other"));
  VCHECK(err.code() == Err::NotFound);
  VCHECK_CONTAINS(err.what(), "mykernel");  // suggests what IS in the module

  dev.unload_module(mod);
  auto err2 = VCAPTURE(Error, dev.get_function(mod, "mykernel"));
  VCHECK(err2.code() == Err::NotFound);
}

VTEST(unsupported_ptx_error_names_profile) {
  runtime::Runtime rt(load_gpu("nvidia/b200"));
  // Any instruction outside the implemented subset will do; tcgen05.mma's
  // .ashift is refused because the ISA leaves it undefined. cp.async, then
  // wgmma, then tcgen05.fence, then tcgen05.shift, then tcgen05.ld.red used to
  // stand here and each had to be replaced once it was implemented -- an
  // example of something unsupported has to actually still be unsupported.
  auto err = VCAPTURE(Error, rt.device(0).load_module(
                                 ".version 8.7\n.target sm_100a\n.address_size 64\n"
                                 ".visible .entry k() { .reg .b32 %r<2>; "
                                 "tcgen05.mma.cta_group::1.kind::f16.ashift [%r0], [%r1], %r0, %r0, %r0; ret; }\n"));
  VCHECK(err.code() == Err::UnsupportedPtx);
  VCHECK_CONTAINS(err.what(), "GPU profile: nvidia/b200");
}

// sm_90a code runs on compute capability 9.0 and nowhere else -- not on a
// newer Blackwell, which forward compatibility would otherwise allow -- and
// family code (sm_100f) within its major version only.
VTEST(arch_specific_targets_load_only_where_they_run) {
  const std::string body = "\n.address_size 64\n.visible .entry k() { ret; }\n";
  runtime::Runtime h100(load_gpu("nvidia/h100"));
  runtime::Runtime b200(load_gpu("nvidia/b200"));
  (void)h100.device(0).load_module(".version 8.3\n.target sm_90a" + body);
  (void)b200.device(0).load_module(".version 8.3\n.target sm_90" + body);   // plain: forward
  auto err = VCAPTURE(Error, b200.device(0).load_module(".version 8.3\n.target sm_90a" + body));
  VCHECK(err.code() == Err::PtxParse);
  VCHECK_CONTAINS(err.what(), "specific to compute capability 9.0");
  (void)b200.device(0).load_module(".version 8.8\n.target sm_100f" + body);
  (void)b200.device(0).load_module(".version 8.7\n.target sm_100a" + body);
  auto err2 = VCAPTURE(Error, h100.device(0).load_module(".version 8.8\n.target sm_100f" + body));
  VCHECK(err2.code() == Err::PtxParse);   // newer than the device, the plain rule
}

VTEST(devices_have_independent_memory) {
  runtime::Runtime rt(load_gpu("nvidia/a10"), 2);
  uint64_t p0 = rt.device(0).memory().alloc(64);
  uint32_t v = 42;
  rt.device(0).memory().write(p0, &v, 4);
  // Same VA on device 1 is not a valid pointer there.
  uint32_t out = 0;
  auto err = VCAPTURE(Error, rt.device(1).memory().read(p0, &out, 4));
  VCHECK(err.code() == Err::InvalidPointer);
}

VTEST(peer_copy_between_virtual_devices) {
  // Device pointers are per-device; a peer copy moves bytes between two
  // devices' memories (the path cudaMemcpyPeer takes).
  runtime::Runtime rt(load_gpu("nvidia/a10"), 2);
  uint64_t p0 = rt.device(0).memory().alloc(256);
  uint64_t p1 = rt.device(1).memory().alloc(256);
  std::vector<uint32_t> src(64);
  for (uint32_t i = 0; i < src.size(); ++i) src[i] = i * 3 + 1;
  rt.device(0).memory().write(p0, src.data(), src.size() * 4);

  std::vector<uint8_t> staging(256);
  rt.device(0).memory().read(p0, staging.data(), staging.size());
  rt.device(1).memory().write(p1, staging.data(), staging.size());

  std::vector<uint32_t> got(64);
  rt.device(1).memory().read(p1, got.data(), got.size() * 4);
  VCHECK(src == got);
  // The source device's pointer is still invalid on the destination device.
  uint32_t scratch = 0;
  auto err = VCAPTURE(Error, rt.device(1).memory().read(p0 + 4096, &scratch, 4));
  VCHECK(err.code() == Err::InvalidPointer);
}

// A large module's kernels are parsed one at a time, when first looked up
// (runtime.cpp, "lazy modules"); VGPU_LAZY_MODULE_BYTES=0 makes even a small
// one lazy. Its kernels still share the module's globals, reach its device
// functions, see function addresses in the module's order, and an unsupported
// instruction in a kernel nothing launches no longer stops the module loading.
VTEST(large_modules_parse_kernels_when_first_used) {
  setenv("VGPU_LAZY_MODULE_BYTES", "0", 1);
  runtime::Runtime rt(load_gpu("nvidia/b200"));
  auto& dev = rt.device(0);
  const uint64_t mod = dev.load_module(R"(.version 8.7
.target sm_100a
.address_size 64
.global .align 4 .u32 g;
.func (.param .b32 r) twice(.param .b32 x)
{
  .reg .b32 %a;
  ld.param.b32 %a, [x];
  add.s32 %a, %a, %a;
  st.param.b32 [r], %a;
  ret;
}
.visible .entry set_g()
{
  .reg .b64 %p;
  .reg .b32 %v;
  mov.u64 %p, g;
  mov.u32 %v, 21;
  st.global.u32 [%p], %v;
  ret;
}
.visible .entry unused()
{
  .reg .b32 %r<2>;
  tcgen05.mma.cta_group::1.kind::f16.ashift [%r0], [%r1], %r0, %r0, %r0;
  ret;
}
.visible .entry read_g(.param .u64 out)
{
  .reg .b64 %p<3>;
  .reg .b32 %v<2>;
  ld.param.u64 %p0, [out];
  mov.u64 %p1, g;
  ld.global.u32 %v0, [%p1];
  {
  .param .b32 param0;
  st.param.b32 [param0], %v0;
  .param .b32 retval0;
  call.uni (retval0), twice, (param0);
  ld.param.b32 %v1, [retval0];
  }
  st.global.u32 [%p0], %v1;
  mov.u64 %p2, twice;
  st.global.u64 [%p0+8], %p2;
  ret;
}
)");
  unsetenv("VGPU_LAZY_MODULE_BYTES");
  const uint64_t out = dev.memory().alloc(16);
  auto arg = [](uint64_t v) {
    std::vector<uint8_t> b(8);
    std::memcpy(b.data(), &v, 8);
    return b;
  };
  dev.launch(*dev.get_function(mod, "set_g"), exec::LaunchConfig{}, {}, dev.symbols(mod));
  dev.launch(*dev.get_function(mod, "read_g"), exec::LaunchConfig{}, {arg(out)}, dev.symbols(mod));
  VCHECK_EQ(dev.memory().load_scalar(out, 4), 42ull);   // set_g's global, doubled by twice
  VCHECK_EQ(dev.memory().load_scalar(out + 8, 8), dev.symbols(mod)->at("twice"));
  auto err = VCAPTURE(Error, dev.get_function(mod, "unused"));
  VCHECK(err.code() == Err::UnsupportedPtx);
}

VTEST_MAIN
