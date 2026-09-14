#!/usr/bin/env bash
# nvidia-smi's table is a parsed interface: people scrape these columns with
# awk and cut, and tools that drive a fleet read them positionally. So the
# layout is pinned here against the geometry of a real driver's output --
# 91-column frame, fields of 41, 24 and 22 between the bars, and the process
# block's seven columns at fixed offsets.
#
# The reference offsets below were measured from `nvidia-smi` on a physical
# NVIDIA machine, which is the same "hardware as oracle" rule the conformance
# suite follows. If a future driver moves a column, this test is the place to
# re-measure rather than the place to relax.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "no vgpu at $vgpu"; exit 1; }

out=$(VGPU_GPU=nvidia/h100 VGPU_DEVICE_COUNT=2 VGPU_DRIVER_VERSION=550.54.15 \
      VGPU_CUDA_VERSION=12.4 VGPU_TELEMETRY_PATH=/nonexistent-so-idle \
      "$vgpu" smi 2>&1) || { echo "vgpu smi failed"; exit 1; }

fail=0
note() { echo "FAIL: $1"; fail=1; }

# Every framed line is exactly 91 columns. The timestamp line is 31.
n=0
while IFS= read -r line; do
  n=$((n + 1))
  [[ -z "$line" ]] && continue
  if [[ $n -eq 1 ]]; then
    [[ ${#line} -eq 31 ]] || note "line 1 (timestamp) is ${#line} columns, expected 31"
    continue
  fi
  [[ ${#line} -eq 91 ]] || note "line $n is ${#line} columns, expected 91: $line"
done <<< "$out"

# The header a real driver prints. An empty "Driver Version:" was the original
# bug and is the one thing here that must never come back.
grep -q '^| NVIDIA-SMI 550\.54\.15 .*Driver Version: 550\.54\.15 .*CUDA Version: 12\.4 *|$' <<< "$out" \
  || note "the NVIDIA-SMI header line does not match the real format"
grep -q 'Driver Version: *|' <<< "$out" && note "Driver Version is empty"

# Column names, verbatim and in order.
grep -q '^| GPU  Name                 Persistence-M | Bus-Id          Disp\.A | Volatile Uncorr\. ECC |$' <<< "$out" \
  || note "the GPU/Name/Persistence-M header row does not match"
grep -q '^| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M\. |$' <<< "$out" \
  || note "the Fan/Temp/Perf header row does not match"

# The bars sit at columns 0, 42, 67 and 90 on every device row.
while IFS= read -r line; do
  for col in 0 42 67 90; do
    [[ "${line:$col:1}" == "|" ]] || note "device row has no bar at column $col: $line"
  done
done < <(grep -E '^\|( *[0-9]+  [A-Z]| *[0-9]+%)' <<< "$out")

# nvidia-smi always prints the process block, and "none" is a normal answer.
grep -q '^| Processes:' <<< "$out" || note "the Processes block is missing"
grep -q '^|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |$' <<< "$out" \
  || note "the Processes header row does not match"
grep -q 'No running processes found' <<< "$out" || note "an idle machine should report no processes"

# The virtual-device block is ours, not nvidia-smi's, so it must stay off the
# default output and stay reachable behind the flag.
grep -q 'Virtual device details' <<< "$out" && note "the virtual block must not appear by default"
VGPU_GPU=nvidia/h100 VGPU_TELEMETRY_PATH=/nonexistent-so-idle "$vgpu" smi --details 2>&1 \
  | grep -q 'Virtual device details' || note "--details no longer shows the virtual block"

[[ $fail -eq 0 ]] && echo "smi format matches the real nvidia-smi layout"
exit $fail
