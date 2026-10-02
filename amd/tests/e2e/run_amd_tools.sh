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
. "$root/tests/session_guard.sh"
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
# A session without isolation, which every host can start (GitHub's runners
# restrict unprivileged user namespaces); isolated() is the isolated form, for
# the checks that need the session's own /sys.
sess() { local g=$1; shift; timeout 120 "$vgpu" shell -y --no-isolate --gpu "$g" --count 2 "$@" </dev/null 2>/dev/null; }
isolated() { local g=$1; shift; timeout 120 "$vgpu" shell -y --gpu "$g" --count 2 "$@" </dev/null 2>/dev/null; }

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
# With no PCI ID database at all (a container without pciutils), the names
# are VirtualGPU's own: the class, the vendor, the card.
expect "with no PCI ID database, a Radeon is still a VGA controller, named" \
  "01:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M] (rev c8)" \
  "$(sess amd/rx7900xtx -c "VGPU_PCI_IDS=/nonexistent $vgpu smi --lspci -s 01:00.0")"
expect "and an Instinct card a processing accelerator" \
  "01:00.0 Processing accelerators: Advanced Micro Devices, Inc. [AMD/ATI] Aqua Vanjaram [Instinct MI300X]" \
  "$(sess amd/mi300x -c "VGPU_PCI_IDS=/nonexistent $vgpu smi --lspci -s 01:00.0")"
expect "lspci -k names the driver" "Kernel driver in use: amdgpu" \
  "$(sess amd/mi300x -c "$vgpu smi --lspci -k -s 01:00.0" | grep -o 'Kernel driver in use: .*')"
if command -v lspci >/dev/null && session_isolation_ok; then
  expect "isolated, the real lspci reads the session's /sys: the bound driver" "Kernel driver in use: amdgpu" \
    "$(isolated amd/mi300x -c 'lspci -k -s 01:00.0' | grep -o 'Kernel driver in use: .*')"
else
  echo "skip  isolated lspci: no lspci, or unprivileged user namespaces are unavailable here"
fi
expect "lspci -vmm keeps its record form" "Slot:	01:00.0|Class:	VGA compatible controller|Rev:	c8|ProgIf:	00" \
  "$(sess amd/rx7900xtx -c "$vgpu smi --lspci -vmm -s 01:00.0" | grep -E '^(Slot|Class|Rev|ProgIf):' | paste -sd'|')"

# Form by form, VirtualGPU's lspci against the host's real one on the same
# machine, reading the isolated session's /sys as it would a real machine's
# (without isolation it reads a dump of config space, which has no BAR sizes,
# interrupt or driver). The real one names a subsystem from udev's hardware
# database where it has one, and "Kernel modules" from the host kernel's module
# aliases; both differ by host, so both are left out of the comparison -- as
# is the warning it prints where the host has no module files to read (a
# container's "Unable to load libkmod resources").
if command -v lspci >/dev/null && session_isolation_ok; then
  for g in amd/rx7900xtx amd/mi300x; do
    diffs=$(timeout 300 "$vgpu" shell -y --gpu "$g" --count 3 </dev/null 2>/dev/null -c '
      norm() { sed -e "/Kernel modules:/d; /^Module:/d; /Subsystem:/d; /^SDevice:/d; /Unable to load libkmod resources/d" -e "s/\"[^\"]*\"$//"; }
      for f in "" -nn -n -D -mm -mmnn -mmn -k -v -vnn -vn -vmm -vmmk -kmm "-d 1002:" "-d ::0300" "-s 02:00.0" -t -x -xxx -xxxx; do
        a=$(lspci $f 2>&1 | norm); b=$('"$vgpu"' smi --lspci $f 2>&1 | norm)
        # (A filter may rightly match nothing; the plain listing never.)
        [ "$a" = "$b" ] && { [ -n "$a" ] || [ -n "$f" ]; } || echo "lspci $f"
      done; echo checked')
    expect "VirtualGPU's lspci prints what the real one does, every form ($g)" "checked" "$diffs"
  done
else
  echo "skip  lspci form by form: no lspci, or unprivileged user namespaces are unavailable here"
fi

# --- rocm-smi and amd-smi: every command a monitoring or inventory script
# runs answers, on an Instinct and a Radeon card ---
for g in amd/mi300x amd/rx7900xtx; do
  failed=$(sess "$g" -c '
    for c in "rocm-smi" "rocm-smi -a" "rocm-smi --showbus" "rocm-smi --showhw" "rocm-smi --showdriverversion" \
             "rocm-smi --showpids" "rocm-smi --showtopo" "rocm-smi --showmemuse" "rocm-smi --showperflevel" \
             "rocm-smi --showvoltage" "rocm-smi --showenergycounter" "rocm-smi --showcomputepartition" \
             "rocm-smi --showmemorypartition" "rocm-smi --showpagesinfo" "rocm-smi --showxgmierr" "rocm-smi -s" \
             "rocm-smi --showfwinfo" "rocm-smi --showrasinfo all" "rocm-smi --json" "rocm-smi --csv" \
             "amd-smi list" "amd-smi static" "amd-smi metric" "amd-smi process" "amd-smi topology" \
             "amd-smi monitor" "amd-smi firmware" "amd-smi partition" "amd-smi bad-pages" "amd-smi xgmi" \
             "amd-smi version" "amd-smi static --json" "amd-smi metric --json" "amd-smi topology --json"; do
      out=$($c 2>&1) && [ -n "$out" ] || echo "$c"
    done; echo checked')
  expect "every rocm-smi and amd-smi command answers ($g)" "checked" "$failed"
done
expect "rocm_agent_enumerator names a Radeon's own target" "gfx1100" \
  "$(sess amd/rx7900xtx -c 'rocm_agent_enumerator -t GPU' | sort -u)"
expect "amd-smi static: a Radeon's compute units, target, memory and slot" "96|gfx1100|PCIE|GDDR6|384" \
  "$(sess amd/rx7900xtx -c 'amd-smi static -g 0' | awk -F': ' '/NUM_COMPUTE_UNITS|TARGET_GRAPHICS_VERSION|^        TYPE|BIT_WIDTH|SLOT_TYPE/{print $2}' | paste -sd'|')"
expect "amd-smi static --json is JSON, a unit with each measurement" "750 W" \
  "$(sess amd/mi300x -c 'amd-smi static -l --json -g 0' | python3 -c 'import json,sys; l=json.load(sys.stdin)[0]["limit"]["max_power"]; print(l["value"], l["unit"])')"
expect "amd-smi topology: Instinct cards are one XGMI hop apart" "XGMI" \
  "$(sess amd/mi300x -c 'amd-smi topology' | awk '/LINK TYPE TABLE/{f=1; next} f&&/^0000:01/{print $3; exit}')"
expect "rocm-smi --showtopo: Radeon cards reach each other over PCIe" "PCIE" \
  "$(sess amd/rx7900xtx -c 'rocm-smi --showtopo' | awk '/Link Type/{f=1; next} f&&/^GPU0/{print $3; exit}')"

# --- The AMD kernel driver's interface: KFD's topology, /dev/kfd, /dev/dri ---
# Every session stages KFD's topology; an isolated one shows it at
# /sys/class/kfd, where tools look for AMD GPUs without a runtime.
expect "KFD's topology: a CPU node and a node per GPU, each with the GPU's target" "0|90402|90402" \
  "$(sess amd/mi300x -c 'for n in 0 1 2; do awk "/^gfx_target_version/{print \$2}" $VGPU_SESSION/root/sys/class/kfd/kfd/topology/nodes/$n/properties; done' | paste -sd'|')"
expect "KFD's topology: Radeon GPUs reach each other through the host" "1|1|2 40" \
  "$(sess amd/rx7900xtx -c 't=$VGPU_SESSION/root/sys/class/kfd/kfd/topology/nodes/1; awk "/^(io_links_count|p2p_links_count) /{print \$2}" $t/properties; awk "/^(type|weight) /{print \$2}" $t/p2p_links/0/properties | paste -sd" "' | paste -sd'|')"
enum=$(ls /opt/rocm/bin/rocm_agent_enumerator "$HOME"/.local/share/rocm-*/opt/rocm-*/bin/rocm_agent_enumerator 2>/dev/null | sort -V | tail -1)
if ! session_isolation_ok; then
  echo "skip  the isolated kernel-driver checks: unprivileged user namespaces are unavailable here"
else
  host_dev=$(ls -A /dev | sort | paste -sd' ')
  expect "an isolated session has no physical slots of the host's" "0" \
    "$(isolated amd/mi300x -c 'ls /sys/bus/pci/slots 2>/dev/null | wc -l')"
  expect "an isolated AMD session has /dev/kfd and a card and render node per GPU" "c|card0 card1 renderD128 renderD129" \
    "$(isolated amd/mi300x -c '[ -c /dev/kfd ] && echo c; ls /dev/dri | paste -sd" "' | paste -sd'|')"
  expect "its /dev is still the host's otherwise: process substitution, /dev/shm, ptys" "ok|shm|pts" \
    "$(isolated amd/mi300x -c 'cat <(echo ok); python3 -c "import multiprocessing as m; m.Lock(); print(\"shm\")"; [ -d /dev/pts ] && echo pts' | paste -sd'|')"
  expect "its /sys/class is still the host's otherwise" "yes" \
    "$(isolated amd/mi300x -c '[ -e /sys/class/net/lo ] && [ -e /sys/class/kfd/kfd/topology/nodes/2/properties ] && echo yes')"
  # The session mounts the host's /dev inside its directory; removing the
  # directory at the end once deleted through that mount, and as root took
  # the host's device nodes with it.
  expect "an isolated session leaves the host's /dev as it found it" "$host_dev" "$(ls -A /dev | sort | paste -sd' ')"
  if [[ -n "$enum" ]]; then
    for g in amd/mi300x amd/rx7900xtx amd/mi250x; do
      want=$(sess "$g" -c 'rocm_agent_enumerator -t GPU' | sort -u)
      expect "ROCm's rocm_agent_enumerator, unmodified, reads KFD's topology ($g)" "$want $want" \
        "$(isolated "$g" -c "python3 '$enum'" | paste -sd' ')"
    done
  else
    echo "skip  ROCm's rocm_agent_enumerator: no ROCm here"
  fi
fi

# --- ROCm's own rocm-smi, on the simulator's librocm_smi64 ---
# An isolated session has /sys/module/amdgpu loaded, as an AMD machine does,
# and points ROCm SMI at the simulator's library, so ROCm's rocm-smi runs
# unmodified and must say what VirtualGPU's own rocm-smi says.
rsmi=""
for f in /opt/rocm/libexec/rocm_smi/rocm_smi.py $(ls "$HOME"/.local/share/*/root/opt/rocm-*/libexec/rocm_smi/rocm_smi.py \
         "$HOME"/.local/share/rocm-*/opt/rocm-*/libexec/rocm_smi/rocm_smi.py 2>/dev/null | sort -V | tail -1); do
  [[ -f "$f" ]] && { rsmi=$f; break; }
done
# One line per GPU and reading, as either tool prints it.
readings() {
  sed -nE 's/^(GPU\[[0-9]+\]).*Card Series:[[:space:]]*(.*)$/\1 series \2/p
            s/^(GPU\[[0-9]+\]).*VRAM Total Memory \(B\):[[:space:]]*([0-9]+)$/\1 vram \2/p
            s/^(GPU\[[0-9]+\]).*(mclk|sclk) clock .*\(([0-9]+)Mhz\)$/\1 \2 \3/p' | sort
}
if [[ -z "$rsmi" ]]; then
  echo "skip  ROCm's rocm-smi: no ROCm here"
elif ! session_isolation_ok; then
  echo "skip  ROCm's rocm-smi: unprivileged user namespaces are unavailable here"
elif objdump -p "$build/shim/librocm_smi64.so.1" 2>/dev/null | grep -q 'NEEDED.*lib[at]san'; then
  echo "skip  ROCm's rocm-smi: a sanitizer build, loaded by a Python that is not built with one"
else
  expect "an isolated AMD session has the amdgpu kernel module loaded" "live" \
    "$(isolated amd/mi300x -c 'cat /sys/module/amdgpu/initstate')"
  expect "the session's other kernel modules are still there" "yes" \
    "$(isolated amd/mi300x -c '[ $(ls /sys/module | wc -l) -gt 1 ] && echo yes')"
  for g in amd/mi300x amd/rx7900xtx amd/rx6900xt; do
    out=$(isolated "$g" -c "python3 '$rsmi' --showproductname --showmeminfo vram --showclocks; echo ====; rocm-smi --showproductname --showmeminfo vram --showclocks")
    theirs=$(sed '/^====$/,$d' <<<"$out" | readings)
    mine=$(sed '1,/^====$/d' <<<"$out" | readings)
    expect "ROCm's rocm-smi reads every card's name, memory and clocks ($g)" "8" "$(grep -c . <<<"$theirs")"
    expect "ROCm's rocm-smi says what VirtualGPU's does ($g)" "$mine" "$theirs"
  done
  expect "ROCm's rocm-smi summary: a row per card" "2" \
    "$(isolated amd/mi300x -c "python3 '$rsmi'" | grep -cE '^[01] +[12] +0x74a1')"
  real_ri=$(ls /opt/rocm/bin/rocminfo "$HOME"/.local/share/rocm-*/opt/rocm-*/bin/rocminfo 2>/dev/null | sort -V | tail -1)
  if [[ -n "$real_ri" ]]; then
    expect "ROCm's rocminfo, unmodified, in an isolated session: both cards" "2" \
      "$(isolated amd/mi300x -c "'$real_ri'" | grep -c '^  Marketing Name: *AMD Instinct MI300X')"
  fi
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
  # ROCm's rocminfo stops before HSA without the amdgpu kernel module, except
  # under WSL; there it describes the host, not the simulator.
  if [[ -n "$real" && ! -e /sys/module/amdgpu/initstate && ! -e /dev/dxg ]]; then
    echo "skip  ROCm's rocminfo: no amdgpu kernel module here, which it asks for before HSA"
  elif [[ -n "$real" ]] && objdump -p "$build/shim/libhsa-runtime64.so.1" 2>/dev/null | grep -q 'NEEDED.*lib[at]san'; then
    echo "skip  ROCm's rocminfo: a sanitizer build, and ROCm's binary is not built with one"
  elif [[ -n "$real" ]]; then
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
