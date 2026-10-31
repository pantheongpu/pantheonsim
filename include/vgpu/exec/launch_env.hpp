// The environment variables a launch reads (the runtime's, the PTX interpreter's and the SASS executor's), found in one pass over the environment instead of one getenv (a scan of all of it) for each.
// Read at every launch, as before: a test may set any of them between two launches of one process.
#pragma once


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
  const char* trace = nullptr;        // VGPU_TRACE
  const char* strict = nullptr;       // VGPU_STRICT
  const char* race = nullptr;         // VGPU_RACE
  const char* max_steps = nullptr;    // VGPU_MAX_STEPS
  const char* scheduler = nullptr;    // VGPU_SCHEDULER
  const char* scheduler_seed = nullptr;   // VGPU_SCHEDULER_SEED
  const char* fail_alloc = nullptr;   // VGPU_FAIL_ALLOC
  const char* fail_launch = nullptr;  // VGPU_FAIL_LAUNCH
  const char* fail_memcpy = nullptr;  // VGPU_FAIL_MEMCPY
};

LaunchEnv read_launch_env();   // (src/exec/launch_env.cpp: out of line, so the launch paths that call it carry a call and not the scan)

// The pass a caller already made (LaunchConfig::env), or one made now into `own`: a launch made without the runtime's pass (a test, a
// tool) reads for itself, as before.
inline const LaunchEnv& launch_env_of(const LaunchEnv* given, LaunchEnv& own) {
  if (given) return *given;
  own = read_launch_env();
  return own;
}

}  // namespace vgpu::exec
