#!/usr/bin/env bash
# ctest entry for the AMD benchmark workloads' instruction counters.
#
# Compiles tools/bench-amd/kernels/bench_amd.hip with hipcc for gfx942 and gfx1030, runs its workloads at the quick
# size on one host thread under rocprofv3 inside `vgpu shell`, and compares the SQ_INSTS_* totals with
# tools/bench-amd/expected/amd-hip-<release>.json. Only COUNTS are compared, never times: a change to the
# interpreter that makes it faster must not change what it counts, and this is the quick test that says so. Each
# workload also checks its own results against a host reference (it prints PASS only when they match).
#
# Skips (exit 0, "SKIP:" in the output) where there is no ROCm with hipcc and rocprofv3, the shim is
# sanitizer-built (ROCm's profiler tool is not), or there is no expected file for the hipcc release in use.
# To regenerate after an intended change:  tools/bench-amd/bench.sh --update-expected --build <build dir>
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" && pwd)"
[[ -x "$build/vgpu" && -e "$build/shim/libamdhip64.so.7" && -e "$build/shim/librocprofiler-sdk.so.1" ]] ||
  { echo "SKIP: build vgpu, vgpuhip and vgpurocprof first"; exit 0; }
rocm=""
for c in "${VGPU_ROCM_PATH:-}" "${ROCM_PATH:-}" /opt/rocm $(ls -d "$HOME"/.local/share/rocm-*/opt/rocm-* 2>/dev/null | sort -rV); do
  [[ -n "$c" && -x "$c/bin/hipcc" && -x "$c/bin/rocprofv3" && -e "$c/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so" ]] && { rocm=$c; break; }
done
[[ -n "$rocm" ]] || { echo "SKIP: no ROCm with hipcc and rocprofiler-sdk found (set VGPU_ROCM_PATH)"; exit 0; }
command -v python3 >/dev/null 2>&1 || { echo "SKIP: python3 not found"; exit 0; }
if objdump -p "$build/shim/libamdhip64.so.7" 2>/dev/null | awk '/NEEDED/ {print $2}' | grep -qE 'lib(a|t)san'; then
  echo "SKIP: the shim is built with a sanitizer, and ROCm's rocprofv3 tool is not"; exit 0
fi
cache="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-bench-amd.XXXXXX")"
trap 'rm -rf "$cache"' EXIT
export ROCM_PATH="$rocm"
python3 -I "$root/tools/bench-amd/bench.py" --check-counters --build "$build" --hipcc "$rocm/bin/hipcc" --cache "$cache" --quiet
