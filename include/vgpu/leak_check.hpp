#pragma once
// A resource check for the process as a whole: what the simulator and the program together still hold when the machine goes away,
// against what they held when it was made.
//
//   VGPU_LEAK_CHECK=1        one line on stderr when the process exits
//   VGPU_LEAK_CHECK=strict   the same, and the process exits with status 70 if the line says LEAK
//   VGPU_LEAK_CHECK_FILE=    append the line to this file instead of stderr
//
// A line reads
//   vgpu-leak-check: device_bytes=0 device_allocs=0 pinned_allocs=0 pinned_bytes=0 fds=+0 threads=+0 maps=+0 rss_kb=81234 -> clean
// or ends "-> LEAK device_bytes,fds" naming each resource that grew. device_bytes and device_allocs are the allocator's own
// accounting (what cudaMemGetInfo shows missing), summed over the devices; pinned_* are cudaMallocHost/cudaHostAlloc blocks; fds and
// threads are the growth since the machine was made, with the worker pool's threads left out. maps (lines of /proc/self/maps) is
// shown, not judged: a library loaded on first use maps a dozen segments and that is not a leak.
// A program that frees everything it allocates, closes what it opens and joins what it starts must read clean: a library handle
// destroyed with memory still allocated for it, a stream whose thread stays, a file the simulator keeps open, each show here.
// A program that leaks on purpose reads LEAK, which is how the check is shown to bite (nvidia/tests/e2e/leak_probe.cu).
//
// rss_kb is information only: the host's resident memory is not a count of anything the program holds (freed memory stays
// resident in the allocator's arenas), so growth in it is judged over many repeats of one workload instead (leak_soak.cu).
#include <string>

namespace vgpu::runtime {
class Runtime;

// Records the baseline. Called once, when the machine is made. Does nothing unless VGPU_LEAK_CHECK is set.
void leak_check_arm();

// Reports, and with `strict` exits with status 70 on a leak. Called with the machine still alive, as it is destroyed.
void leak_check_finish(Runtime* rt);

// The line for the current state against the baseline `leak_check_arm` took (empty if not armed); `leaked` says whether it ends LEAK.
std::string leak_check_line(Runtime* rt, bool* leaked);
}  // namespace vgpu::runtime
