// Unit tests for the two environment overrides every CUDA entry point and
// nvidia-smi read: the declared CUDA version and the VRAM size. Both used to
// be parsed in several places with several answers; these pin the one
// function each is now read through, including what it does with rubbish.
#include <cstdlib>
#include <string>

#include "vgpu/driver_version.hpp"
#include "vgpu/profile.hpp"
#include "vtest.hpp"

namespace {
// Sets or clears a variable for the duration of one check.
struct Env {
  const char* name;
  Env(const char* n, const char* v) : name(n) {
    if (v) ::setenv(n, v, 1); else ::unsetenv(n);
  }
  ~Env() { ::unsetenv(name); }
};
}  // namespace

VTEST(driver_version_defaults_to_13_0_outside_a_session) {
  Env e("VGPU_CUDA_VERSION", nullptr);
  VCHECK_EQ(vgpu::driver_version(), 13000);
  VCHECK_EQ(vgpu::driver_version_string(), std::string("13.0"));
}

VTEST(driver_version_follows_the_session_declaration) {
  // CUDA's own encoding: major * 1000 + minor * 10.
  { Env e("VGPU_CUDA_VERSION", "12.4"); VCHECK_EQ(vgpu::driver_version(), 12040); }
  { Env e("VGPU_CUDA_VERSION", "12.6"); VCHECK_EQ(vgpu::driver_version(), 12060); }
  { Env e("VGPU_CUDA_VERSION", "11.8"); VCHECK_EQ(vgpu::driver_version(), 11080); }
  { Env e("VGPU_CUDA_VERSION", "13.0"); VCHECK_EQ(vgpu::driver_version(), 13000); }
}

VTEST(driver_version_string_is_derived_not_echoed) {
  // nvidia-smi prints this, so it must come from the number the driver API
  // reports -- never copied from the environment, which could show a version
  // the driver refused.
  { Env e("VGPU_CUDA_VERSION", "12.4"); VCHECK_EQ(vgpu::driver_version_string(), std::string("12.4")); }
  { Env e("VGPU_CUDA_VERSION", "junk"); VCHECK_EQ(vgpu::driver_version_string(), std::string("13.0")); }
}

VTEST(driver_version_rejects_anything_that_is_not_a_version) {
  for (const char* bad : {"junk", "12", "", "12.", ".4", "12.4.1", "12.4x", "-1.0", "100.0",
                          "12.100", "twelve.four"}) {
    Env e("VGPU_CUDA_VERSION", bad);
    VCHECK_EQ(vgpu::driver_version(), vgpu::kDefaultDriverVersion);
  }
}

namespace {
vgpu::DeviceProfile card(uint64_t vram) {
  vgpu::DeviceProfile p;
  p.vram_bytes = vram;
  return p;
}
constexpr uint64_t kMiB = 1024ull * 1024ull;
}  // namespace

VTEST(vram_override_is_ignored_when_unset) {
  Env e("VGPU_VRAM_MB", nullptr);
  auto p = card(81089 * kMiB);
  vgpu::apply_vram_override(p);
  VCHECK_EQ(p.vram_bytes, 81089 * kMiB);
}

VTEST(vram_override_resizes_the_card) {
  Env e("VGPU_VRAM_MB", "4096");
  auto p = card(81089 * kMiB);
  vgpu::apply_vram_override(p);
  VCHECK_EQ(p.vram_bytes, 4096 * kMiB);
}

VTEST(vram_override_that_is_not_a_number_is_ignored_not_zero) {
  // strtoull("junk") is 0, and the old code assigned it: a card with no
  // memory at all. Anything but a positive whole number now leaves it alone.
  for (const char* bad : {"junk", "0", "", "4096x", "-5", "+5", " 4096", "4.5",
                          "99999999999999999999"}) {
    Env e("VGPU_VRAM_MB", bad);
    auto p = card(81089 * kMiB);
    vgpu::apply_vram_override(p);
    VCHECK_EQ(p.vram_bytes, 81089 * kMiB);
  }
}

VTEST_MAIN
