// vgpu demo vectoradd — the M6 milestone as a user-visible command:
// runs the reference vectorAdd PTX through the full runtime on a chosen
// virtual GPU and verifies every output element.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"

namespace {

// The same reference kernel as tests/kernels/vector_add.ptx, embedded so the
// demo works from any directory.
const char* kVectorAddPtx = R"(
.version 8.3
.target sm_90
.address_size 64
.visible .entry vecAdd(
    .param .u64 vecAdd_param_0,
    .param .u64 vecAdd_param_1,
    .param .u64 vecAdd_param_2,
    .param .u32 vecAdd_param_3
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<4>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<11>;
    ld.param.u64 %rd1, [vecAdd_param_0];
    ld.param.u64 %rd2, [vecAdd_param_1];
    ld.param.u64 %rd3, [vecAdd_param_2];
    ld.param.u32 %r2, [vecAdd_param_3];
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r1, %r3, %r4, %r5;
    setp.ge.s32 %p1, %r1, %r2;
    @%p1 bra $L__BB0_2;
    cvta.to.global.u64 %rd4, %rd1;
    mul.wide.s32 %rd5, %r1, 4;
    add.s64 %rd6, %rd4, %rd5;
    cvta.to.global.u64 %rd7, %rd2;
    add.s64 %rd8, %rd7, %rd5;
    ld.global.f32 %f1, [%rd8];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f2, %f1;
    cvta.to.global.u64 %rd9, %rd3;
    add.s64 %rd10, %rd9, %rd5;
    st.global.f32 [%rd10], %f3;
$L__BB0_2:
    ret;
}
)";

std::vector<uint8_t> arg_bytes(const void* p, size_t n) {
  std::vector<uint8_t> b(n);
  std::memcpy(b.data(), p, n);
  return b;
}

}  // namespace

int demo_vectoradd(const std::string& gpu, long long n_ll) {
  using clock = std::chrono::steady_clock;
  if (n_ll <= 0 || n_ll > (1ll << 31)) {
    std::fprintf(stderr, "vgpu demo: -n must be in [1, 2^31]\n");
    return 2;
  }
  const uint32_t n = static_cast<uint32_t>(n_ll);

  vgpu::DeviceProfile profile = vgpu::load_gpu(gpu);
  vgpu::runtime::Runtime rt(profile);
  vgpu::runtime::Device& dev = rt.device(0);

  std::printf("vectorAdd on virtual %s (%s), n=%u\n", profile.id.c_str(), profile.model.c_str(), n);

  std::vector<float> a(n), b(n), c(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = 0.5f * static_cast<float>(i % 4096);
    b[i] = 0.25f * static_cast<float>(i % 4096);
  }

  auto t0 = clock::now();
  uint64_t da = dev.memory().alloc(uint64_t{n} * 4);
  uint64_t db = dev.memory().alloc(uint64_t{n} * 4);
  uint64_t dc = dev.memory().alloc(uint64_t{n} * 4);
  dev.memory().write(da, a.data(), uint64_t{n} * 4);
  dev.memory().write(db, b.data(), uint64_t{n} * 4);

  uint64_t mod = dev.load_module(kVectorAddPtx);
  const vgpu::ptx::EntryFn* fn = dev.get_function(mod, "vecAdd");

  vgpu::exec::LaunchConfig cfg;
  cfg.block = {256, 1, 1};
  cfg.grid = {(n + 255) / 256, 1, 1};
  dev.launch(*fn, cfg, {arg_bytes(&da, 8), arg_bytes(&db, 8), arg_bytes(&dc, 8), arg_bytes(&n, 4)});

  dev.memory().read(dc, c.data(), uint64_t{n} * 4);
  dev.memory().free(da);
  dev.memory().free(db);
  dev.memory().free(dc);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();

  for (uint32_t i = 0; i < n; ++i) {
    float expect = a[i] + b[i];
    if (c[i] != expect) {
      std::printf("FAIL at element %u: got %g, expected %g\n", i, c[i], expect);
      return 1;
    }
  }
  std::printf("PASS  (%u elements verified, %lld ms)\n", n, static_cast<long long>(ms));
  return 0;
}
