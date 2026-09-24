#!/usr/bin/env bash
# AMD's own profiler, rocprofv3, unmodified, on a simulated MI300X: inside
# `vgpu shell`, a program built by hipcc is profiled the way it would be on a
# card, and rocprofv3 writes its own files.
#
# rocprofv3 is ROCm's; what it talks to is VirtualGPU's librocprofiler-sdk
# (amd/src/rocprofiler_sdk.cpp), which the session puts in place of ROCm's.
# What each count should be comes from the program, not the simulator:
# scale_add runs 3000 work-items in groups of 256 -- 12 groups of 4 waves,
# 48 waves, the last of which starts with every lane past the end and skips
# both loads and the store; bump is one wave, three times over from a graph.
#
# Skips where ROCm's rocprofiler-sdk is not installed (set VGPU_ROCM_PATH).
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" && pwd)"
vgpu="$build/vgpu"
[[ -x "$vgpu" && -e "$build/shim/librocprofiler-sdk.so.1" ]] || { echo "SKIP: build vgpu and vgpurocprof first"; exit 0; }
rocm=""
for c in "${VGPU_ROCM_PATH:-}" "${ROCM_PATH:-}" /opt/rocm "$HOME"/.local/share/rocm-7*/opt/rocm-*; do
  [[ -n "$c" && -x "$c/bin/rocprofv3" && -e "$c/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so" ]] && { rocm=$c; break; }
done
[[ -n "$rocm" ]] || { echo "SKIP: no ROCm with rocprofiler-sdk found (set VGPU_ROCM_PATH)"; exit 0; }
command -v python3 >/dev/null || { echo "SKIP: rocprofv3 is a python3 script"; exit 0; }
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_rocprofv3.XXXXXX")"
trap 'rm -rf "$work"' EXIT
unset VGPU_GPU VGPU_DEVICE_COUNT VGPU_SESSION
export VGPU_TELEMETRY_PATH="$work/telemetry" ROCM_PATH="$rocm"
mkdir -p "$VGPU_TELEMETRY_PATH"
exe="$root/amd/tests/hipcc/chevron.gfx942"
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}

# The asan and tsan builds' runtimes have to come first, as for any program
# run against an instrumented shim (run_hipcc.sh).
preload=""
for lib in $(objdump -p "$build/shim/libamdhip64.so.7" 2>/dev/null | awk '/NEEDED/ && /lib(a|t)san/ {print $2}'); do
  path=$(ldconfig -p | awk -v l="$lib" '$1 == l {print $NF; exit}')
  preload="${preload:+$preload:}${path:-$lib}"
done

counters="SQ_WAVES SQ_INSTS_SMEM TA_FLAT_READ_WAVEFRONTS TA_FLAT_WRITE_WAVEFRONTS GRBM_COUNT"
timeout 300 "$vgpu" shell -y --gpu amd/mi300x --count 1 -c \
  "command -v rocprofv3; rocprofv3 ${preload:+--preload $preload} --pmc $counters --kernel-trace --memory-copy-trace \
     --output-format csv -d '$work/out' -- '$exe'" </dev/null >"$work/log" 2>&1
status=$?
out=$(find "$work/out" -name '*_counter_collection.csv' | head -1)
trace=$(find "$work/out" -name '*_kernel_trace.csv' | head -1)
copies=$(find "$work/out" -name '*_memory_copy_trace.csv' | head -1)
agents=$(find "$work/out" -name '*_agent_info.csv' | head -1)

expect "rocprofv3 runs the program to the end" "0" "$status"
expect "the session's rocprofv3 is the one found" "yes" \
  "$(head -1 "$work/log" | grep -q 'vgpu-session-.*/bin/rocprofv3$' && echo yes || echo no)"
expect "the program's own check passes under it" "chevron launch wrong 0 of 3000" \
  "$(grep -o 'chevron launch wrong 0 of 3000' "$work/log")"
expect "it writes its counters, kernel trace, copies and agents" "yes" \
  "$([[ -s "$out" && -s "$trace" && -s "$copies" && -s "$agents" ]] && echo yes || echo no)"
[[ -s "$out" ]] || { sed 's/^/      /' "$work/log" | tail -20; exit 1; }

count() {  # count <kernel prefix> <counter> -> the values, one per dispatch
  python3 - "$out" "$1" "$2" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['Kernel_Name'].startswith(sys.argv[2]) and r['Counter_Name'] == sys.argv[3]]
print(' '.join(str(int(float(r['Counter_Value']))) for r in rows))
PY
}
expect "scale_add runs 48 waves" "48" "$(count scale_add SQ_WAVES)"
expect "47 of them load a and b; the one past the end does not" "94" "$(count scale_add TA_FLAT_READ_WAVEFRONTS)"
expect "and 47 store" "47" "$(count scale_add TA_FLAT_WRITE_WAVEFRONTS)"
expect "bump is one wave, each of the three times the graph runs it" "1 1 1" "$(count bump SQ_WAVES)"
expect "a counter that needs a model of time is not made up" "" "$(count scale_add GRBM_COUNT)"
expect "and rocprofv3 says the device does not have it" "yes" \
  "$(grep -q 'Missing: \[GRBM_COUNT\]' "$work/log" && echo yes || echo no)"
expect "the kernel trace names each dispatch, demangled" \
  "scale_add(float const*, float const*, float*, float, int)|bump(int*, int)|bump(int*, int)|bump(int*, int)" \
  "$(python3 -c "import csv,sys; print('|'.join(r['Kernel_Name'] for r in csv.DictReader(open(sys.argv[1]))))" "$trace")"
expect "the copies go the way the program sends them" "MEMORY_COPY_HOST_TO_DEVICE MEMORY_COPY_HOST_TO_DEVICE MEMORY_COPY_DEVICE_TO_HOST" \
  "$(python3 -c "import csv,sys; print(' '.join([r['Direction'] for r in csv.DictReader(open(sys.argv[1]))][:3]))" "$copies")"
expect "the GPU agent is the MI300X the session simulates" "gfx942 AMD Instinct MI300X 304" \
  "$(python3 -c "import csv,sys; r=[r for r in csv.DictReader(open(sys.argv[1])) if r['Agent_Type']=='GPU'][0]; print(r['Name'], r['Product_Name'], r['Cu_Count'])" "$agents")"

# rocprofv3-avail lists what a device counts, which is how a tool learns what
# to ask for; it finds VirtualGPU's library through the session's own path.
avail=$(timeout 120 "$vgpu" shell -y --gpu amd/mi300x --count 1 -c "rocprofv3-avail list --pmc" </dev/null 2>/dev/null)
expect "rocprofv3-avail lists the counters that are counted" "yes" \
  "$(grep -q 'SQ_INSTS_VALU' <<< "$avail" && grep -q 'TA_FLAT_READ_WAVEFRONTS' <<< "$avail" && echo yes || echo no)"
expect "and not the ones that are not" "no" "$(grep -q 'GRBM_COUNT' <<< "$avail" && echo yes || echo no)"

exit $fail
