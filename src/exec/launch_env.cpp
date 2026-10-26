#include "vgpu/exec/launch_env.hpp"

#include <cstdlib>
#include <cstring>

#if defined(__linux__)
extern "C" char** environ;
#endif

namespace vgpu::exec {

LaunchEnv read_launch_env() {
  LaunchEnv e;
#if defined(__linux__)
  const auto is = [](const char* name, const char* want) {   // `name` is "NAME=value": the whole name, then '='
    const size_t n = std::strlen(want);
    return std::strncmp(name, want, n) == 0 && name[n] == '=' ? name + n + 1 : static_cast<const char*>(nullptr);
  };
  const auto take = [](const char*& field, const char* value) {
    if (value && !field) field = value;
  };
  if (!environ) return e;
  for (char** p = environ; *p; ++p) {
    const char* s = *p;
    if (s[0] != 'V' || s[1] != 'G' || s[2] != 'P' || s[3] != 'U' || s[4] != '_') continue;
    s += 5;
    switch (s[0]) {
      case 'F':
        if (s[1] == 'A' && s[2] == 'S') take(e.fastpath, is(s, "FASTPATH"));
        else {
          take(e.fail_alloc, is(s, "FAIL_ALLOC"));
          take(e.fail_launch, is(s, "FAIL_LAUNCH"));
          take(e.fail_memcpy, is(s, "FAIL_MEMCPY"));
        }
        break;
      case 'T':
        if (s[1] == 'C') take(e.tc_exact, is(s, "TC_EXACT"));
        else if (s[1] == 'I' && s[2] == 'E' && s[3] == 'R' && s[4] == '_') {
          take(e.tier_stats, is(s, "TIER_STATS"));
          take(e.tier_chain, is(s, "TIER_CHAIN"));
        } else {
          take(e.threads, is(s, "THREADS"));
          take(e.trace, is(s, "TRACE"));
        }
        break;
      case 'S':
        if (s[1] == 'A') take(e.sass_trace, is(s, "SASS_TRACE"));
        else if (s[1] == 'T') take(e.strict, is(s, "STRICT"));
        else {
          take(e.scheduler, is(s, "SCHEDULER"));
          take(e.scheduler_seed, is(s, "SCHEDULER_SEED"));
        }
        break;
      case 'R': take(e.race, is(s, "RACE")); break;
      case 'M': take(e.max_steps, is(s, "MAX_STEPS")); break;
      case 'L': take(e.lockstep, is(s, "LOCKSTEP")); break;
      default: break;
    }
  }
#else
  e.fastpath = std::getenv("VGPU_FASTPATH");
  e.tc_exact = std::getenv("VGPU_TC_EXACT");
  e.sass_trace = std::getenv("VGPU_SASS_TRACE");
  e.tier_stats = std::getenv("VGPU_TIER_STATS");
  e.tier_chain = std::getenv("VGPU_TIER_CHAIN");
  e.threads = std::getenv("VGPU_THREADS");
  e.lockstep = std::getenv("VGPU_LOCKSTEP");
  e.trace = std::getenv("VGPU_TRACE");
  e.strict = std::getenv("VGPU_STRICT");
  e.race = std::getenv("VGPU_RACE");
  e.max_steps = std::getenv("VGPU_MAX_STEPS");
  e.scheduler = std::getenv("VGPU_SCHEDULER");
  e.scheduler_seed = std::getenv("VGPU_SCHEDULER_SEED");
  e.fail_alloc = std::getenv("VGPU_FAIL_ALLOC");
  e.fail_launch = std::getenv("VGPU_FAIL_LAUNCH");
  e.fail_memcpy = std::getenv("VGPU_FAIL_MEMCPY");
#endif
  return e;
}

}  // namespace vgpu::exec
