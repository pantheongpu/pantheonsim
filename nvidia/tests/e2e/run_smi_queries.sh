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
expect "an idle card reports only the idle reason, as a real one does" "0x0000000000000001" \
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

# topo -m: the matrix scripts split on tabs. Links go through the host, so
# every pair is PHB; no NVLink is claimed that no profile measured.
topo=$(smi topo -m)
expect "topo -m header names each GPU and the affinity columns" \
  $'\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID' "$(head -1 <<< "$topo")"
expect "topo -m rows: self is X, peers are PHB" "GPU0| X |PHB GPU1|PHB| X " \
  "$(sed -n 2,3p <<< "$topo" | awk -F'\t' '{printf "%s|%s|%s ", $1, $2, $3}' | sed 's/ $//')"
expect "topo -m has its legend" "yes" "$(grep -q '^  PHB  = ' <<< "$topo" && echo yes || echo no)"
smi topo >/dev/null 2>&1; expect "bare topo prints its usage, as nvidia-smi does" "0" "$?"
smi topo -p2p r >/dev/null 2>&1; expect "a topo form without link data is refused" "2" "$?"

# -q -x: well-formed XML under the real element names.
if command -v python3 >/dev/null; then
  xml=$(smi -q -x -i 1)
  got=$(python3 -c '
import sys, xml.dom.minidom as m
d = m.parseString(sys.stdin.read())
t = lambda n, e=d: e.getElementsByTagName(n)[0].firstChild.data
gpus = d.getElementsByTagName("gpu")
print(t("driver_version"), t("cuda_version"), t("attached_gpus"), len(gpus),
      gpus[0].getAttribute("id"), t("product_name", gpus[0]), t("product_architecture", gpus[0]),
      t("total", gpus[0].getElementsByTagName("fb_memory_usage")[0]), t("gpu_util", gpus[0]))
' <<< "$xml" 2>&1)
  expect "-q -x parses and carries the -q values" \
    "550.54.15 12.4 2 1 00000000:02:00.0 Tesla T4 Turing 15360 MiB 0 %" "$got"
fi

# Reliability, link and clock-event fields, answered the way real cards answer
# them. smi() is a T4: ECC on, GDDR so pages are retired, no memory sensor, and
# the Gen3 x8 link clouds attach it at.
q() { smi -i 0 --query-gpu="$1" --format=csv,noheader; }
expect "ECC is on for a card that ships with it" "Enabled, Enabled" "$(q ecc.mode.current,ecc.mode.pending)"
expect "ECC counters are zero" "0, 0" \
  "$(q ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.aggregate.device_memory)"
expect "a GDDR card retires pages and does not remap rows" "0, 0, No, [N/A]" \
  "$(q retired_pages.sbe,retired_pages.double_bit.count,retired_pages.pending,remapped_rows.correctable)"
expect "no memory sensor, printed as the real driver prints it" "N/A" "$(q temperature.memory)"
expect "the link real T4s run at" "3, 8, 3, 8" \
  "$(q pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max)"
expect "an idle card reports the idle clock-event reason" \
  "0x00000000000001FF, 0x0000000000000001, Active, Not Active" \
  "$(q clocks_event_reasons.supported,clocks_event_reasons.active,clocks_event_reasons.gpu_idle,clocks_event_reasons.hw_slowdown)"
expect "the throttle spelling is accepted too" "0x0000000000000001" "$(q clocks_throttle_reasons.active)"
expect "the table's ECC column agrees" "yes" \
  "$(smi | grep -q 'Off |                    0 |' && echo yes || echo no)"
# Pantheon's RAS snapshot asks for all of these in one query. One name missing
# failed the whole query, and Pantheon read that as RAS unavailable.
ras="ecc.mode.current"
for sev in corrected uncorrected; do for win in volatile aggregate; do
  for loc in device_memory dram register_file l1_cache l2_cache texture_memory cbu sram total; do
    ras="$ras,ecc.errors.$sev.$win.$loc"
  done
done; done
ras="$ras,ecc.errors.uncorrected.volatile.sram.parity,ecc.errors.uncorrected.aggregate.sram.thresholdExceeded"
ras="$ras,retired_pages.sbe,retired_pages.dbe,retired_pages.pending"
smi -i 0 --query-gpu="$ras" --format=csv,noheader,nounits >/dev/null
expect "Pantheon's whole RAS query is accepted" "0" "$?"
# A GeForce card has no ECC but answers the SRAM breakdown, as a real RTX 3060
# does; an HBM card remaps rows instead of retiring pages.
card() { VGPU_GPU="$1" VGPU_DEVICE_COUNT=1 VGPU_TELEMETRY_PATH=/nonexistent-so-idle \
           "$vgpu" smi --query-gpu="$2" --format=csv,noheader 2>&1; }
expect "no ECC on a GeForce card, SRAM breakdown still answered" "[N/A], [N/A], 0, No" \
  "$(card nvidia/rtx3060 ecc.mode.current,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.sram.parity,ecc.errors.uncorrected.aggregate.sram.thresholdExceeded)"
expect "an HBM card remaps rows and does not retire pages" "[N/A], 0, No" \
  "$(card nvidia/h100 retired_pages.sbe,remapped_rows.correctable,remapped_rows.pending)"
expect "a memory sensor where real cards report one" "yes" \
  "$(card nvidia/b200 temperature.memory | grep -qE '^[0-9]+$' && echo yes || echo no)"

exit $fail
