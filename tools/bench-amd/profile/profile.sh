#!/usr/bin/env bash
# Profile one workload on the simulated GPU without perf (which needs perf_event_paranoid <= 2): a SIGPROF
# sampler preloaded into the program (sigprof.c), then addr2line (report.py).
#
#   profile.sh <build dir> <workload> <mi300x|rx6800> [report.py options ...]
#
# Use a build made with -g (cmake -DCMAKE_CXX_FLAGS=-g; the same optimisation level, so the same code) for
# function names of inlined helpers and for --lines. Environment: BENCH_BIN (the compiled bench_amd for the
# target; default: the newest in ~/.cache/vgpu-bench-amd), SIGPROF_HZ (default 1000), SIGPROF_STACK=1
# (record call chains, for --inclusive), SIZE (quick|full, default full), THREADS (default 1), KEEP=file.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "$1" && pwd)"; wl="$2"; tg="$3"; shift 3
case "$tg" in mi300x) arch=gfx942; gpu=amd/mi300x ;; rx6800) arch=gfx1030; gpu=amd/rx6800 ;; *) echo "target?"; exit 2 ;; esac
bin="${BENCH_BIN:-$(ls -t "${HOME}"/.cache/vgpu-bench-amd/bench_amd.$arch.* | head -1)}"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
gcc -O2 -shared -fPIC -o "$tmp/libsigprof.so" "$here/sigprof.c" -ldl
out="${KEEP:-$tmp/samples.txt}"
LD_PRELOAD="$tmp/libsigprof.so" SIGPROF_OUT="$out" SIGPROF_HZ="${SIGPROF_HZ:-1000}" \
  VGPU_GPU="$gpu" LD_LIBRARY_PATH="$build/shim" VGPU_THREADS="${THREADS:-1}" VGPU_QUIET=1 \
  VGPU_TELEMETRY_PATH="$tmp/t" VGPU_STATE_DIR="$tmp/s" "$bin" "$wl" "${SIZE:-full}" | tail -2
python3 -I "$here/report.py" "$out" --exe "$bin" "$@"
