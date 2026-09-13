#!/usr/bin/env bash
# nvidia-smi's query forms, the ones scripts and monitoring agents drive.
#
# Every check here is a way the tool once disagreed with the real one:
#   -i/--id was read and thrown away, so `-i 1` printed every GPU and `-i 7`
#   printed both and exited 0 -- a per-GPU query (Pantheon runs one per card)
#   got the whole rack; -L was only understood as the first argument; CSV
#   headers had no units, though real headers always carry them; -q said CUDA
#   13.0 whatever the session was; --query-compute-apps was an unknown option.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "no vgpu at $vgpu"; exit 1; }
smi() {
  VGPU_GPU=nvidia/t4 VGPU_DEVICE_COUNT=2 VGPU_DRIVER_VERSION=550.54.15 VGPU_CUDA_VERSION=12.4 \
    VGPU_TELEMETRY_PATH=/nonexistent-so-idle "$vgpu" smi "$@" 2>&1
}
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}

expect "-i selects one GPU" "1, Tesla T4" "$(smi -i 1 --query-gpu=index,name --format=csv,noheader)"
expect "--id=N form" "1" "$(smi --id=1 --query-gpu=index --format=csv,noheader)"
expect "a list of ids" $'0\n1' "$(smi -i 0,1 --query-gpu=index --format=csv,noheader)"
uuid1=$(smi -i 1 --query-gpu=uuid --format=csv,noheader)
expect "-i by UUID, with -L after it" "GPU 1: Tesla T4 (UUID: $uuid1)" "$(smi -i "$uuid1" -L)"
expect "-L before -i" "GPU 0" "$(smi -L -i 0 | cut -d: -f1)"
expect "-i by short bus id" "1" "$(smi -i 02:00.0 --query-gpu=index --format=csv,noheader)"
out=$(smi -i 7 --query-gpu=index --format=csv); rc=$?
expect "a GPU that does not exist is an error" "No devices were found / 6" "$out / $rc"
table=$(smi -i 1)
expect "-i narrows the table to that GPU" "1 0" \
  "$(grep -cE '^\| +1  Tesla T4' <<< "$table") $(grep -cE '^\| +0  Tesla T4' <<< "$table")"

expect "CSV headers carry units, with or without nounits" \
  "name, memory.total [MiB], utilization.gpu [%], power.draw [W], temperature.gpu" \
  "$(smi --query-gpu=name,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,nounits | head -1)"
expect "clocks.gr is the graphics clock" "yes" \
  "$( [[ "$(smi --query-gpu=clocks.gr --format=csv,noheader,nounits)" =~ ^[0-9]+$'\n'[0-9]+$ ]] && echo yes || echo no)"
expect "nothing throttles" "0x0000000000000000" \
  "$(smi -i 0 --query-gpu=clocks_throttle_reasons.active --format=csv,noheader)"

expect "--query-compute-apps on an idle machine is a header and no rows" \
  "pid, process_name, used_gpu_memory [MiB] / 0" \
  "$(smi --query-compute-apps=pid,process_name,used_memory --format=csv) / $?"

q=$(smi -q -i 1)
expect "-q reports the session's CUDA version" "CUDA Version : 12.4" \
  "$(grep -m1 '^CUDA Version' <<< "$q" | tr -s ' ')"
expect "-q names the brand" "Product Brand : NVIDIA" "$(grep -m1 'Product Brand' <<< "$q" | tr -s ' ' | sed 's/^ //')"
expect "-q names the architecture" "Product Architecture : Turing" \
  "$(grep -m1 'Product Architecture' <<< "$q" | tr -s ' ' | sed 's/^ //')"

exit $fail
