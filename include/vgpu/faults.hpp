// Fault injection.
//
// Make an allocation fail, a launch fail, a copy fail -- on demand, at a chosen
// occurrence. The point is the error paths nobody exercises: a `cudaMalloc`
// whose return value is checked but never non-zero in testing, a launch failure
// the application logs and then continues past anyway. On real hardware you
// cannot provoke those without exhausting a device or corrupting one. Here the
// allocator is ours, so it costs an if.
//
// This is diagnostics rather than emulation. Nothing here models a fault a real
// GPU would produce spontaneously; it produces the documented failure the API
// is allowed to return, at a moment you choose, so the handling can be tested.
//
// Off unless asked for. Every knob is an environment variable, because the
// program under test must not have to be rebuilt to be tested:
//
//   VGPU_FAIL_ALLOC=3        the 3rd allocation returns out-of-memory
//   VGPU_FAIL_ALLOC=2,5,9    those three do
//   VGPU_FAIL_ALLOC=all      every one does
//   VGPU_FAIL_LAUNCH=1       the 1st kernel launch fails
//   VGPU_FAIL_MEMCPY=4       the 4th copy fails
//
// Occurrences are 1-based and counted per process, per operation.
#pragma once

#include <cstdint>

namespace vgpu::faults {

enum class Op { Alloc, Launch, Memcpy, Count };

// True when this occurrence of `op` has been selected to fail. Counts the
// occurrence either way, so the numbering does not depend on which ones fail.
bool should_fail(Op op);

// True when any injection is configured at all -- lets a caller skip the
// bookkeeping entirely in the overwhelmingly common case.
bool enabled();

// Re-reads the environment. Called once per launch alongside the other mode
// flags; exposed so tests can change a variable and have it take effect
// without a new process.
void refresh();

const char* op_name(Op op);

}  // namespace vgpu::faults
