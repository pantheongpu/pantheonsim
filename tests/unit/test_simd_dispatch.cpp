// Which kernel table the engines run, for each value of VGPU_SIMD, on this host: the best level the CPU has when the variable is unset (the AVX-512
// tile kernels on a host with AVX-512 F/BW/DQ/VL and VNNI), a named level capped to what the host can run, base the baseline kernels, off no kernels.
// CMake runs this executable once per setting (test_simd_dispatch, _simd_avx2, _simd_avx512, _simd_base, _simd_off), so that a change of the default
// is a failing test on every host class that has the instruction sets.
#include <cstdlib>
#include <cstring>
#include <string>

#include "vgpu/simd/simd.hpp"
#include "vtest.hpp"

using namespace vgpu;
using simd::Level;

namespace {

Level best_supported() {
  if (simd::supported(Level::Avx512)) return Level::Avx512;
  if (simd::supported(Level::Avx2)) return Level::Avx2;
  return Level::Base;
}

Level capped(Level want) {
  if (want == Level::Avx512) return best_supported();
  if (want == Level::Avx2) return simd::supported(Level::Avx2) ? Level::Avx2 : Level::Base;
  return Level::Base;
}

std::string name(Level l) { return simd::level_name(l); }

}  // namespace

VTEST(the_table_in_use_follows_VGPU_SIMD_and_the_host) {
  const char* v = std::getenv("VGPU_SIMD");
  const Level got = simd::kernels().level;
  if (!v || !*v) {
    VCHECK_EQ(name(got), name(best_supported()));
  } else if (!std::strcmp(v, "off")) {
    VCHECK(!simd::enabled());
  } else if (!std::strcmp(v, "base")) {
    VCHECK_EQ(name(got), name(Level::Base));
  } else if (!std::strcmp(v, "avx2")) {
    VCHECK_EQ(name(got), name(capped(Level::Avx2)));
  } else if (!std::strcmp(v, "avx512")) {
    VCHECK_EQ(name(got), name(capped(Level::Avx512)));
  }
}

VTEST(level_names_the_lane_kernels_and_never_the_tile_table) {
  // level() is the family of the lane kernels (Base or Avx2); the AVX-512 table is the AVX2 one with the tile kernels replaced.
  VCHECK(simd::level() != Level::Avx512);
  if (simd::kernels().level == Level::Avx512) VCHECK(simd::level() == Level::Avx2);
}

VTEST(avx512_is_only_chosen_where_the_host_has_it) {
  if (simd::kernels().level == Level::Avx512) VCHECK(simd::supported(Level::Avx512));
}

VTEST_MAIN
