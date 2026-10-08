#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER, as the driver reads them: which
# devices the runtime and the driver show a program, in what order, and what
# every call says when none is shown.
#
# The table below is what two RTX 3060s did under each value (driver 13.0),
# and a machine of two simulated GPUs must do the same. With --card the same
# table runs against the real driver, on a machine of two NVIDIA GPUs of
# one model -- which is how the table was checked, and how it can be again.
# Skips if nvcc is unavailable.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
card=0
[[ "${1:-}" == "--card" ]] && card=1
src="$root/nvidia/tests/e2e/device_attributes.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_cuda_visible_devices_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
if (( card )); then
  nvcc -std=c++14 -cudart shared -arch=native -Wno-deprecated-gpu-targets "$src" -o "$out" -lcuda
else
  shopt -s nullglob
  cudart_libs=("$shim"/libcudart.so.[0-9]*)
  shopt -u nullglob
  if (( ${#cudart_libs[@]} == 0 )); then
    echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
  fi
  nvcc -std=c++14 -cudart shared --gpu-architecture=sm_86 -Wno-deprecated-gpu-targets \
       $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" -lcuda
  if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
  if [[ -n "$(shim_sanitizer "$shim")" ]]; then rm -f "$out"; echo "SKIP: a sanitizer build loads two copies of the core"; exit 0; fi
fi
trap 'rm -f "$out"' EXIT

# One run of the program under an environment, its one line of output.
run() {   # run <CUDA_VISIBLE_DEVICES or ::unset> <CUDA_DEVICE_ORDER or ::unset> <mode>
  local env=(env -u CUDA_VISIBLE_DEVICES -u CUDA_DEVICE_ORDER)
  [[ "$1" != "::unset" ]] && env+=("CUDA_VISIBLE_DEVICES=$1")
  if [[ "$2" != "::unset" ]]; then env+=("CUDA_DEVICE_ORDER=$2")
  elif (( card )); then env+=("CUDA_DEVICE_ORDER=PCI_BUS_ID")   # so the card's numbers follow its PCI order, as here
  fi
  if (( card )); then
    "${env[@]}" LD_LIBRARY_PATH="${CUDA_RT_LIB:-/usr/local/cuda-13.0/targets/x86_64-linux/lib}" "$out" "$3" 2>&1
  else
    "${env[@]}" VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_DEVICE_COUNT=2 LD_LIBRARY_PATH="$shim" "$out" "$3" 2>&1
  fi
}

# The machine: its devices' UUIDs, in PCI order.
uuids=($(run ::unset ::unset --uuids | sed 's/.*| *//'))
(( ${#uuids[@]} == 2 )) || { echo "FAIL  expected two devices, got: ${uuids[*]:-none}"; exit 1; }
dash() { local h="${1#GPU-}"; echo "GPU-${h:0:8}-${h:8:4}-${h:12:4}-${h:16:4}-${h:20:12}"; }
u0="$(dash "${uuids[0]}")"; u1="$(dash "${uuids[1]}")"

# value | the devices shown (their PCI order numbers), or the error every call answers
cases=(
  "::unset|0 1"            "0|0"                  "1|1"                  "0,1|0 1"            "1,0|1 0"
  " 1|1"                   "1 |1"                 "0, 1|0 1"             "01|1"               "0x1|0"
  "1.0|1"                  "0,5|0"                "0,-1,1|0"             "0,abc,1|0"
  "|error 100"             "5|error 100"          "5,0|error 100"        "-1|error 100"       "abc|error 100"
  "NoDevFiles|error 100"   "GPU-|error 100"       "GPU-00000000|error 100"
  "0,0|error 101"          "1,1,0|error 101"
  "$u1|1"                  "${u1:0:12}|1"         "$u0,$u1|0 1"          "$u1,$u0|1 0"        "$u1,0|1"          "0,$u1|0"
  "0,$u0|0"                "$u0,0|0"              "1,$u1,0|1"
  "$u0,$u0|error 101"      "${u0:0:12},$u0|error 101"
)
# Each device the program lists is "[position bus-id uuid-prefix]"; the bus ids
# are the machine's own (a card's is wherever it sits), so only the position
# and the first eight digits of the UUID are compared.
bare() { sed -E 's/ +$//; s/\[([0-9]+) [0-9a-f:.]+ ([0-9a-f]{8})\]/[\1 \2]/g' <<< "$1"; }
fail=0
for c in "${cases[@]}"; do
  value="${c%|*}"; want="${c##*|}"
  got="$(bare "$(run "$value" ::unset --shown)")"
  if [[ "$want" == error* ]]; then
    e="${want#error }"
    expect="runtime $e count -1 | driver cuInit $e count-rc 3 count -1 |"
  else
    n=$(wc -w <<< "$want")
    expect="runtime 0 count $n | driver cuInit 0 count-rc 0 count $n |"
    pos=0
    for d in $want; do
      id="${uuids[$d]#GPU-}"
      expect+=" [$pos ${id:0:8}]"
      pos=$((pos + 1))
    done
  fi
  if [[ "$got" == "$expect" ]]; then
    echo "ok    CUDA_VISIBLE_DEVICES=\"$value\": ${want}"
  else
    echo "FAIL  CUDA_VISIBLE_DEVICES=\"$value\""; echo "      expected: $expect"; echo "      actual:   $got"; fail=1
  fi
done

# CUDA_DEVICE_ORDER: the two orders it knows agree on identical devices; another value is an invalid device.
for order in FASTEST_FIRST PCI_BUS_ID; do
  got="$(run "1,0" "$order" --shown | sed -E 's/ \[0 .*//')"
  [[ "$got" == "runtime 0 count 2 | driver cuInit 0 count-rc 0 count 2 |" ]] && echo "ok    CUDA_DEVICE_ORDER=$order" || { echo "FAIL  CUDA_DEVICE_ORDER=$order: $got"; fail=1; }
done
got="$(run ::unset bogus --shown)"
if [[ "$got" == "runtime 101 count -1 | driver cuInit 101 count-rc 3 count -1 |" ]]; then echo "ok    CUDA_DEVICE_ORDER=bogus"; else echo "FAIL  CUDA_DEVICE_ORDER=bogus: $got"; fail=1; fi

# The device a program calls 0 is the machine's other one under
# CUDA_VISIBLE_DEVICES=1, and its memory is that device's in NVML (and so in
# nvidia-smi): work is attributed to the machine's slot, not the program's index.
if (( ! card )); then
  # A private telemetry directory: NVML reads whatever publishers share the
  # default one, and a machine running other simulated programs (an AMD session,
  # say) would be the machine it describes.
  tele="$(mktemp -d)"; trap 'rm -rf "$tele" "$out"' EXIT
  for c in "::unset|0" "1|1" "1,0|1" "0,1|0"; do
    value="${c%|*}"; machine_device="${c##*|}"
    report="$(VGPU_TELEMETRY_PATH="$tele" run "$value" ::unset --where)"
    got="$(grep -c "device $machine_device holds 6[0-9] MiB" <<<"$report" || true)"
    other="$(grep -c "holds 6[0-9] MiB" <<<"$report" || true)"
    if [[ "$got" == 1 && "$other" == 1 ]]; then echo "ok    CUDA_VISIBLE_DEVICES=\"$value\": a program's device 0 is the machine's device $machine_device, in NVML"
    else echo "FAIL  CUDA_VISIBLE_DEVICES=\"$value\": device 0's memory should be on the machine's device $machine_device"; printf "%s\n" "$report"; fail=1; fi
  done
fi
exit $fail
