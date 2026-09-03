#!/usr/bin/env bash
# Runs a few real pantheon workloads against VirtualGPU.
#
# The unit tests cover instructions and API calls one at a time. These cover the
# thing that actually matters: an unmodified GPU diagnostic, built by its own
# Makefile with its own nvcc flags, running end to end. Every serious bug this
# simulator has had showed up here first and in the unit tests second.
#
# Pantheon is a separate public repository. Point VGPU_PANTHEON_DIR at a
# checkout, or let this find one next to the build. Skips cleanly when absent so
# a contributor without it still gets a green run.
set -uo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
build="${VGPU_BUILD_DIR:-$root/build}"

pantheon=""
for cand in "${VGPU_PANTHEON_DIR:-}" "$root/../pantheon" "$HOME/pantheon"; do
  [[ -n "$cand" && -f "$cand/Makefile" && -d "$cand/kernels" ]] && { pantheon="$(cd "$cand" && pwd)"; break; }
done
[[ -n "$pantheon" ]] || { echo "SKIP: no pantheon checkout (set VGPU_PANTHEON_DIR)"; exit 0; }
command -v nvcc >/dev/null || { echo "SKIP: nvcc not found (the workloads are compiled from source)"; exit 0; }
[[ -x "$build/vgpu" ]] || { echo "SKIP: $build/vgpu not built"; exit 0; }

# Small, fast, and between them they exercise the paths that have broken before:
# integer and float arithmetic, global traffic, shared memory and atomics.
WORKLOADS=("${VGPU_WORKLOADS:-compute_virus int_virus cache_latency}")
GPU="${VGPU_WORKLOAD_GPU:-nvidia/a10}"
VRAM_MB="${VGPU_WORKLOAD_VRAM_MB:-1024}"
DURATION="${VGPU_WORKLOAD_SECONDS:-2}"
MEM_PCT="${VGPU_WORKLOAD_MEM_PCT:-2}"

echo "pantheon: $pantheon"
echo "workloads: ${WORKLOADS[*]}  (gpu=$GPU vram=${VRAM_MB}MB ${DURATION}s ${MEM_PCT}%)"

# Build and run inside the shell: it supplies an nvcc that links the shared CUDA
# runtime, without which the binaries link a static one that cannot run against
# a simulated driver.
inner=$(cat <<INNER
set -e
cd "$pantheon"
# Targets are named by their output path: a bare name hits make's builtin
# rule and compiles the .cpp with g++, which cannot take CUDA flags.
targets=""; for w in ${WORKLOADS[*]}; do targets="\$targets build/\$w"; done
make -k PLATFORM=CUDA BUILD_DIR=build -j"\$(nproc)" \$targets >/tmp/vgpu_wl_build.log 2>&1 || {
  echo "BUILD FAILED"; tail -20 /tmp/vgpu_wl_build.log; exit 1; }
rc=0
for w in ${WORKLOADS[*]}; do
  bin="build/\$w"
  [ -x "\$bin" ] || { echo "MISSING: \$w did not build"; rc=1; continue; }
  out=\$("\$bin" 0 $DURATION $MEM_PCT 2>&1) || { echo "FAIL: \$w exited nonzero"; echo "\$out" | tail -5; rc=1; continue; }
  # A workload that ran must say something about what it measured. A silent
  # success is the failure this project cares about most.
  if ! printf '%s' "\$out" | grep -qiE "throughput|bandwidth|latency|ops/s|GB/s|PANTHEON"; then
    echo "FAIL: \$w produced no measurement"; printf '%s\n' "\$out" | tail -5; rc=1; continue
  fi
  if printf '%s' "\$out" | grep -qiE "PANTHEON ERROR|CUDA error|integrity"; then
    echo "FAIL: \$w reported an error"; printf '%s' "\$out" | grep -iE "PANTHEON ERROR|CUDA error|integrity" | head -3; rc=1; continue
  fi
  echo "ok   \$w"
done
exit \$rc
INNER
)

# vgpu shell isolates itself with an unprivileged user namespace. Hardened
# kernels forbid that (Ubuntu 24.04 sets
# kernel.apparmor_restrict_unprivileged_userns=1, and CI runners and most
# containers restrict it too), so fall back to running without isolation. The
# shim still takes precedence through LD_LIBRARY_PATH; what is lost is the
# hiding of a real GPU, which matters only on a host that has one.
isolate=()
if ! unshare --user --map-root-user true >/dev/null 2>&1; then
  echo "note: unprivileged user namespaces unavailable; running with --no-isolate"
  isolate=(--no-isolate)
fi

"$build/vgpu" shell --gpu "$GPU" --vram-mb "$VRAM_MB" "${isolate[@]}" -y -c "$inner"
status=$?
[[ $status -eq 0 ]] && echo "pantheon workloads: all passed"
exit $status
