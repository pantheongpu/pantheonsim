#!/usr/bin/env bash
# The system tools on a simulated AMD machine: lspci and rocminfo say what
# they would on the real one.
#
#   - lspci names each card as the current PCI ID database does, with a
#     Radeon's class (VGA controller) and revision, whether or not the host's
#     database knows the card; and where the host has no lspci at all,
#     `vgpu smi --lspci` prints the same as the real one would, form by form.
#   - rocminfo: VirtualGPU's own prints what ROCm's prints on the same
#     simulated machine, line for line, where ROCm's is installed to compare;
#     and each card's chip ID, compute units and caches are the card's.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" && pwd)"
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "SKIP: no vgpu at $vgpu"; exit 0; }
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_amd_tools.XXXXXX")"
trap 'rm -rf "$work"' EXIT
unset VGPU_GPU VGPU_DEVICE_COUNT VGPU_SESSION
export VGPU_QUIET=1 VGPU_TELEMETRY_PATH="$work/telemetry"
mkdir -p "$VGPU_TELEMETRY_PATH"

fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
sess() { local g=$1; shift; timeout 120 "$vgpu" shell -y --gpu "$g" --count 2 "$@" </dev/null 2>/dev/null; }

# --- lspci ---
expect "a Radeon is a VGA controller, of its chip's revision" \
  "01:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M] [1002:744c] (rev c8)" \
  "$(sess amd/rx7900xtx -c "$vgpu smi --lspci -nn -s 01:00.0")"
expect "an Instinct card is a processing accelerator" \
  "01:00.0 Processing accelerators [1200]: Advanced Micro Devices, Inc. [AMD/ATI] Aqua Vanjaram [Instinct MI300X] [1002:74a1]" \
  "$(sess amd/mi300x -c "$vgpu smi --lspci -nn -s 01:00.0")"
expect "a card the host's PCI ID database predates is named" \
  "01:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 48 [Radeon RX 9070/9070 XT/9070 GRE] (rev c0)" \
  "$(sess amd/rx9070xt -c 'lspci -s 01:00.0')"
expect "lspci -k names the driver" "Kernel driver in use: amdgpu" \
  "$(sess amd/mi300x -c "$vgpu smi --lspci -k -s 01:00.0" | grep -o 'Kernel driver in use: .*')"
expect "lspci -vmm keeps its record form" "Slot:	01:00.0|Class:	VGA compatible controller|Rev:	c8|ProgIf:	00" \
  "$(sess amd/rx7900xtx -c "$vgpu smi --lspci -vmm -s 01:00.0" | grep -E '^(Slot|Class|Rev|ProgIf):' | paste -sd'|')"

# Form by form, VirtualGPU's lspci against the host's real one on the same
# machine. The real one names a subsystem from udev's hardware database where
# it has one, and "Kernel modules" from the host kernel's module aliases; both
# differ by host, so both are left out of the comparison.
if command -v lspci >/dev/null; then
  for g in amd/rx7900xtx amd/mi300x; do
    diffs=$(sess "$g" --count 3 -c '
      norm() { sed -e "/Kernel modules:/d; /^Module:/d; /Subsystem:/d; /^SDevice:/d" -e "s/\"[^\"]*\"$//"; }
      for f in "" -nn -n -D -mm -mmnn -mmn -k -v -vnn -vn -vmm -vmmk -kmm "-d 1002:" "-d ::0300" "-s 02:00.0" -t -x -xxx -xxxx; do
        a=$(lspci $f 2>&1 | norm); b=$('"$vgpu"' smi --lspci $f 2>&1 | norm)
        [ "$a" = "$b" ] || echo "lspci $f"
      done')
    expect "VirtualGPU's lspci prints what the real one does, every form ($g)" "" "$diffs"
  done
else
  echo "skip  no lspci on this host to compare against"
fi

# --- rocminfo ---
ours="$build/vgpu-rocminfo"
if [[ -x "$ours" ]]; then
  expect "rocminfo: an RX 7900 XTX's chip, compute units, SIMDs and engines" \
    "29772(0x744c)|96|2|6" \
    "$(VGPU_GPU=amd/rx7900xtx LD_LIBRARY_PATH="$build/shim" "$ours" | awk -F: '/Device Type: *GPU/{g=1} g&&/Chip ID|Compute Unit|SIMDs per CU|Shader Engines/{gsub(/ /,"",$2); print $2}' | head -4 | paste -sd'|')"
  expect "rocminfo: an MI300X's caches, L3 included" "32(0x20)KB|4096(0x1000)KB|262144(0x40000)KB" \
    "$(VGPU_GPU=amd/mi300x LD_LIBRARY_PATH="$build/shim" "$ours" | awk -F: '/^    L[123]:/{gsub(/ /,"",$2); print $2}' | head -3 | paste -sd'|')"
  expect "the session's rocminfo lists both cards" "2" \
    "$(sess amd/mi300x -c 'rocminfo' | grep -c '^  Marketing Name: *AMD Instinct MI300X')"
  real="${VGPU_ROCMINFO:-}"
  if [[ -z "$real" ]]; then
    for f in /opt/rocm/bin/rocminfo $(ls "$HOME"/.local/share/rocm-*/opt/rocm-*/bin/rocminfo 2>/dev/null | sort -V | tail -1); do
      [[ -x "$f" ]] && { real=$f; break; }
    done
  fi
  if [[ -n "$real" ]]; then
    for g in amd/mi300x amd/rx7900xtx amd/mi250x; do
      # The first line is what each finds of the host's kernel driver.
      a=$(VGPU_GPU=$g VGPU_DEVICE_COUNT=2 LD_LIBRARY_PATH="$build/shim" timeout 120 "$real" 2>&1 | tail -n +2)
      b=$(VGPU_GPU=$g VGPU_DEVICE_COUNT=2 LD_LIBRARY_PATH="$build/shim" timeout 120 "$ours" 2>&1 | tail -n +2)
      expect "VirtualGPU's rocminfo prints what ROCm's does ($g)" "0" "$(diff <(echo "$a") <(echo "$b") | grep -c '^[<>]')"
    done
  else
    echo "skip  no ROCm rocminfo to compare against"
  fi
else
  echo "FAIL  no vgpu-rocminfo in $build"
  fail=1
fi
exit $fail
