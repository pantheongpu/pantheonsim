#!/usr/bin/env bash
# ctest entry for the benchmark workloads' instruction counters.
#
# Compiles tools/bench/kernels/bench.cu and runs its sixteen workloads on the
# SASS engine and on the PTX interpreter at the quick size, then compares the
# VGPU_COUNTERS totals (instructions, memory operations, atomics, barriers, the
# class mix) with tools/bench/expected/cuda-<release>.json. Only COUNTS are
# compared, never times: a change to the simulator that speeds it up must not
# change what it counts, and this is the quick test that says so. Each workload
# also checks its own results against a host reference (it prints PASS only
# when they match), so a speed-up that breaks a result fails here too.
#
# Skips (exit 0, "SKIP:" in the output) when there is no nvcc for the shim's
# toolkit, the shim is sanitizer-built (the Release job runs this), or there is
# no expected file for the toolkit release in use. To regenerate after an
# intended change:  tools/bench/bench.sh --update-expected --build <build dir>
#
# The harness hands the programs a clean environment (no VGPU_* of the caller's), so VGPU_TIER (the pre-decoded
# execution tier, src/sass/tier.hpp) is forwarded by name: VGPU_TIER=1 tools/bench/run_bench_check.sh checks that the
# tier counts exactly what the interpreter does.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="${VGPU_BUILD_DIR:-$root/build}"
shim="$build/shim"
shopt -s nullglob
cudart=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#cudart[@]} )) || { echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0; }
command -v python3 >/dev/null 2>&1 || { echo "SKIP: python3 not found"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
cache="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-bench.XXXXXX")"
trap 'rm -rf "$cache"' EXIT
extra=()
[[ -n "${VGPU_TIER:-}" ]] && extra+=(--env "VGPU_TIER=$VGPU_TIER")
python3 -I "$root/tools/bench/bench.py" --check-counters --build "$build" --nvcc "$nvcc_bin" --cache "$cache" --quiet "${extra[@]}"
