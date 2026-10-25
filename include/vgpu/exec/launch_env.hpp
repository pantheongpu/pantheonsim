// The environment variables a SASS launch reads, found in one pass over the environment instead of one getenv (a scan of all of it) for each.
// Read at every launch, as before: a test may set any of them between two launches of one process.
#pragma once

#include <cstdlib>
#include <cstring>

#if defined(__linux__)
extern "C" char** environ;
#endif

namespace vgpu::exec {

// Each field points at the variable's value (getenv's answer: the first entry of that name), or is null when it is not set.
struct LaunchEnv {
  const char* fastpath = nullptr;     // VGPU_FASTPATH
  const char* tc_exact = nullptr;     // VGPU_TC_EXACT
  const char* sass_trace = nullptr;   // VGPU_SASS_TRACE
  const char* tier_stats = nullptr;   // VGPU_TIER_STATS
  const char* tier_chain = nullptr;   // VGPU_TIER_CHAIN
  const char* threads = nullptr;      // VGPU_THREADS
  const char* lockstep = nullptr;     // VGPU_LOCKSTEP
};

inline LaunchEnv read_launch_env() {
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
      case 'F': take(e.fastpath, is(s, "FASTPATH")); break;
      case 'T':
        if (s[1] == 'C') take(e.tc_exact, is(s, "TC_EXACT"));
        else if (s[1] == 'I' && s[2] == 'E' && s[3] == 'R' && s[4] == '_') {
          take(e.tier_stats, is(s, "TIER_STATS"));
          take(e.tier_chain, is(s, "TIER_CHAIN"));
        } else take(e.threads, is(s, "THREADS"));
        break;
      case 'S': take(e.sass_trace, is(s, "SASS_TRACE")); break;
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
#endif
  return e;
}

}  // namespace vgpu::exec
