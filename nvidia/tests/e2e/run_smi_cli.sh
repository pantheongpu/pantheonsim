#!/usr/bin/env bash
# The nvidia-smi and rocm-smi forms scripts actually type, beyond the queries
# run_smi_queries.sh covers.
#
# Each check is a way the drop-ins once disagreed with the real tools: -d was
# skipped over, so `-q -d MEMORY` printed everything; -x without -q printed the
# table; --csv and rocm-smi ignored the device selection; an unknown query field
# was [N/A] with exit 0; --query-gpu without --format, and --format alone,
# printed the table; -h, -l and bare `topo` were rejected; the timestamp field
# was empty; an index past any integer crashed with "stoul"; the drop-in
# nvidia-smi dropped every argument after -L; and rocm-smi's documented flags
# were rejected in a session and ignored outside one.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" && pwd)"
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "no vgpu at $vgpu"; exit 1; }
unset VGPU_QUIET VGPU_CUDA_VERSION VGPU_DRIVER_VERSION VGPU_NVML_VERSION
idle=(env VGPU_GPU=nvidia/t4 VGPU_DEVICE_COUNT=2 VGPU_TELEMETRY_PATH=/nonexistent-so-idle)
smi() { "${idle[@]}" "$vgpu" smi "$@" 2>&1; }
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
yes_if() { if "$@"; then echo yes; else echo no; fi; }

# --- -q -d and -x ---
q=$(smi -q -d MEMORY -i 0)
expect "-q -d MEMORY prints the memory block and nothing else" "1 0 0 0" \
  "$(grep -c 'FB Memory Usage' <<< "$q") $(grep -c 'Product Name' <<< "$q") $(grep -c 'Temperature' <<< "$q") $(grep -c 'Utilization' <<< "$q")"
q=$(smi -q -d memory,TEMPERATURE,PIDS)
expect "-d takes a comma-separated list, for every GPU" "2 2 2" \
  "$(grep -c 'FB Memory Usage' <<< "$q") $(grep -c '^    Temperature$' <<< "$q") $(grep -c 'Processes' <<< "$q")"
smi -q -d BOGUS >/dev/null; expect "an unknown -d section is refused" "2" "$?"
smi -d MEMORY >/dev/null; expect "-d without -q is refused" "2" "$?"
q=$(smi -q -i 0)
expect "the full -q report has compute mode, ECC mode and processes" "1 1 1" \
  "$(grep -c 'Compute Mode' <<< "$q") $(grep -c 'ECC Mode' <<< "$q") $(grep -c 'Processes' <<< "$q")"
expect "-x alone is the XML report" '<?xml version="1.0" ?>' "$(smi -x | head -1)"

# --- device selection everywhere ---
csv=$(smi --csv -i 1)
expect "--csv honours -i" "2 1," "$(wc -l <<< "$csv" | tr -d ' ') $(sed -n 2p <<< "$csv" | cut -c1-2)"
out=$(smi -i 99999999999999999999999 --query-gpu=index --format=csv); rc=$?
expect "an index past any integer is 'No devices were found'" "No devices were found / 6" "$out / $rc"

# --- the query forms ---
out=$(smi --query-gpu=index,bogus.field --format=csv); rc=$?
expect "an unknown query field is refused, by name" "2 yes" "$rc $(yes_if grep -q 'bogus.field' <<< "$out")"
smi --query-gpu= --format=csv >/dev/null; expect "--query-gpu= with no fields is refused" "2" "$?"
smi --query-gpu=index >/dev/null; expect "--query-gpu without --format is refused" "2" "$?"
smi --format=csv >/dev/null; expect "--format with no query is refused" "2" "$?"
smi --query-gpu=index --format=xml >/dev/null; expect "an unknown --format option is refused" "2" "$?"
ts=$(smi -i 0 --query-gpu=timestamp,index --format=csv,noheader)
expect "the timestamp field is filled" "yes" \
  "$([[ "$ts" =~ ^[0-9]{4}/[0-9]{2}/[0-9]{2}\ [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3},\ 0$ ]] && echo yes || echo no)"
expect "documented pci fields answer" "0x00, 0x02" "$(smi -i 1 --query-gpu=pci.domain,pci.bus --format=csv,noheader | sed 's/0x0000/0x00/')"

# --- -h, topo, loops ---
out=$(smi -h); rc=$?
expect "-h prints the usage and succeeds" "0 yes" "$rc $(yes_if grep -q -- '--query-gpu' <<< "$out")"
smi --help >/dev/null; expect "--help succeeds" "0" "$?"
out=$(smi topo); rc=$?
expect "bare topo prints its usage and succeeds" "0 yes" "$rc $(yes_if grep -q 'topo -m' <<< "$out")"
smi topo -p2p r >/dev/null; expect "an unsupported topo form is refused" "2" "$?"
if timeout --preserve-status 1 true 2>/dev/null; then
  rows=$("${idle[@]}" timeout --preserve-status -s INT 1.6 "$vgpu" smi -lms 300 -i 0 --query-gpu=index --format=csv 2>&1); rc=$?
  n=$(grep -c '^0$' <<< "$rows")
  expect "-lms repeats a query with one header until SIGINT, then exits 0" "0 yes 1" \
    "$rc $([[ $n -ge 3 ]] && echo yes || echo "no ($n rows)") $(grep -c '^index$' <<< "$rows")"
  tables=$("${idle[@]}" timeout --preserve-status -s TERM 2.5 "$vgpu" smi -l 1 2>&1); rc=$?
  expect "-l 1 repeats the table until SIGTERM, then exits 0" "0 yes" \
    "$rc $([[ $(grep -c '^| NVIDIA-SMI' <<< "$tables") -ge 2 ]] && echo yes || echo no)"
  "${idle[@]}" timeout --preserve-status -s INT 0.5 "$vgpu" smi --loop=1 --query-gpu=index --format=csv >/dev/null 2>&1
  expect "--loop=N is accepted" "0" "$?"
else
  echo "skip  loops: no timeout --preserve-status"
fi
smi -l 0 >/dev/null; expect "-l 0 is refused" "2" "$?"

# --- the drop-in nvidia-smi script ---
ns() { "${idle[@]}" "$build/bin/nvidia-smi" "$@" 2>&1; }
expect "nvidia-smi -L -i 1 lists only GPU 1" "GPU 1" "$(ns -L -i 1 | cut -d: -f1)"
expect "--version is four numeric lines" \
  $'NVIDIA-SMI version  : 550.54.15\nNVML version        : 12.550.54.15\nDRIVER version      : 550.54.15\nCUDA Version        : 12.4' \
  "$(VGPU_DRIVER_VERSION=550.54.15 VGPU_CUDA_VERSION=12.4 "$build/bin/nvidia-smi" --version)"
expect "the session's nvidia-smi --version is the same text" \
  "$(VGPU_DRIVER_VERSION=560.35.03 VGPU_CUDA_VERSION=12.6 "$build/bin/nvidia-smi" --version)" \
  "$(VGPU_QUIET=1 timeout 60 "$vgpu" shell -y --no-isolate --driver 560.35.03 --cuda 12.6 -c 'nvidia-smi --version' </dev/null 2>&1)"

# --- rocm-smi, outside and inside a session ---
amd=(env VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=2 VGPU_TELEMETRY_PATH=/nonexistent-so-idle)
rs() { "${amd[@]}" "$build/bin/rocm-smi" "$@" 2>&1; }
out=$(rs --showproductname); rc=$?
expect "rocm-smi --showproductname" "0 2" "$rc $(grep -c 'Card Series' <<< "$out")"
out=$(rs --showmeminfo vram -d 1); rc=$?
expect "rocm-smi --showmeminfo vram -d 1 reports GPU 1 only" "0 GPU[1]" \
  "$rc $(grep 'VRAM Total Memory (B)' <<< "$out" | cut -f1)"
out=$(rs -a)
expect "rocm-smi -a is the ROCm report, not the nvidia-smi table" "yes no" \
  "$(yes_if grep -q 'ROCm System Management Interface' <<< "$out") $(yes_if grep -q 'NVIDIA-SMI' <<< "$out")"
out=$(rs -d 1)
expect "the concise rocm-smi table honours -d" "1" "$(grep -E '^[0-9]+ ' <<< "$out" | cut -d' ' -f1)"
rs --bogus >/dev/null; expect "rocm-smi refuses an unknown flag" "2" "$?"
rs --showmeminfo bogus >/dev/null; expect "rocm-smi refuses an unknown memory type" "2" "$?"
# The forms Pantheon runs on an AMD machine, exactly as it runs them. Each used
# to be refused, which left it with no AMD telemetry, an empty GPU list and no
# RAS data.
rs -v -P -t -c -f --json >/dev/null; expect "Pantheon's AMD telemetry poll is accepted" "0" "$?"
rs --showproductname --showmeminfo vram --showmaxpower --showserial --showuniqueid --showmemvendor \
  --json >/dev/null
expect "Pantheon's AMD inventory query is accepted" "0" "$?"
rs --showrasinfo --json >/dev/null; expect "rocm-smi --showrasinfo is accepted" "0" "$?"
rs --showrasinfo bogus >/dev/null; expect "rocm-smi refuses an unknown RAS block" "2" "$?"
if command -v python3 >/dev/null; then
  rsj() { "${amd[@]}" "$build/bin/rocm-smi" "$@" 2>/dev/null; }
  inv='import json,sys; c=json.load(sys.stdin)["card0"]
print(c["Max Graphics Package Power (W)"], c["Unique ID"][:2], c["GPU memory vendor"])'
  expect "the inventory carries power cap, unique ID and memory vendor" "750.0 0x unknown" \
    "$(rsj --showproductname --showmaxpower --showuniqueid --showmemvendor --json | python3 -c "$inv" 2>&1)"
  ras='import json,sys; c=json.load(sys.stdin)["card0"]
print(c["UMC RAS status"], c["UMC correctable errors"], c["UMC uncorrectable errors"])'
  expect "RAS info reports ECC on and no errors" "ENABLED 0 0" "$(rsj --showrasinfo --json | python3 -c "$ras" 2>&1)"
  tmp='import json,sys; c=json.load(sys.stdin)["card0"]
print(" ".join(sorted(k.split("Sensor ")[1].split(")")[0] for k in c if k.startswith("Temperature"))))'
  expect "edge, junction and memory sensors are reported" "edge junction memory" \
    "$(rsj -t --json | python3 -c "$tmp" 2>&1)"
fi
if command -v python3 >/dev/null; then
  check='import json, sys; d = json.load(sys.stdin); print(sorted(d), bool(d["card1"]["Card Series"]), "VRAM Total Memory (B)" in d["card0"])'
  expect "rocm-smi --json parses" "['card0', 'card1'] True True" \
    "$("${amd[@]}" "$build/bin/rocm-smi" --showproductname --showmeminfo vram --json 2>/dev/null | python3 -c "$check" 2>&1)"
  expect "and so does the session's rocm-smi, from the same flags" "['card0', 'card1'] True True" \
    "$(VGPU_QUIET=1 timeout 60 "$vgpu" shell -y --no-isolate --gpu amd/mi300x --count 2 \
         -c 'rocm-smi --showproductname --showmeminfo vram --json' </dev/null 2>/dev/null | python3 -c "$check" 2>&1)"
fi

# --- rocm_agent_enumerator outside a session ---
ag=$("${amd[@]}" "$build/bin/rocm_agent_enumerator" 2>/dev/null)
expect "rocm_agent_enumerator follows VGPU_GPU" "3 gfx000" "$(wc -l <<< "$ag" | tr -d ' ') $(head -1 <<< "$ag")"
err=$(env -u VGPU_GPU VGPU_TELEMETRY_PATH=/nonexistent-so-idle "$build/bin/rocm_agent_enumerator" 2>&1 >/dev/null)
expect "with no AMD GPU configured it says how to configure one" "yes" "$(yes_if grep -q 'VGPU_GPU' <<< "$err")"

exit $fail
