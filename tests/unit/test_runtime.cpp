// Tests for the runtime facade: devices, modules, function lookup.
#include "vgpu/runtime/runtime.hpp"

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
  // Any instruction outside the implemented subset will do; wgmma is Hopper's
  // warpgroup matrix multiply, which this does not implement. cp.async used to
  // stand here and had to be replaced once it was implemented -- an example of
  // something unsupported has to actually still be unsupported.
  auto err = VCAPTURE(Error, rt.device(0).load_module(
                                 ".version 8.3\n.target sm_100\n.address_size 64\n"
                                 ".visible .entry k() { wgmma.fence.sync.aligned; ret; }\n"));
  VCHECK(err.code() == Err::UnsupportedPtx);
  VCHECK_CONTAINS(err.what(), "GPU profile: nvidia/b200");
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

VTEST_MAIN
