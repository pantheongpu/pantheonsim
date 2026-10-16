#!/usr/bin/env bash
# AMD's ROCm Validation Suite (RVS), unmodified, on a simulated MI300X: inside
# `vgpu shell` its inventory, PCIe, BAR, memory-bandwidth and GEMM-correctness
# modules run and pass, through the simulator's KFD topology and sysfs, AMD SMI,
# HIP, rocBLAS and PCI configuration space.
#
# What does NOT run here, by design: the modules whose pass mark is a rate of
# work against time -- gst/perf's GFLOPS target, iet's power target, tst's
# temperature target -- because the simulator is functional-only, not a model of
# a card's speed, power or heat; and mem, which passes but needs minutes a GPU
# (amd/README.md, "ROCm Validation Suite"). gst is run with a target low enough
# to be met anywhere, for its correctness checks: it compares every SGEMM with
# a host computation (accuracy_check), and with itself run to run (self_check),
# and error_inject adds a wrong element to see them catch it.
#
# Skips where RVS is not installed (VGPU_RVS names its rvs binary; ROCm's
# rocm-validation-suite package puts it in <rocm>/bin) or where the session
# cannot be isolated (RVS reads /sys/class/kfd, which the isolated session lays
# over the host's).
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" 2>/dev/null && pwd)" || { echo "SKIP: no build directory"; exit 0; }
vgpu="$build/vgpu"
[[ -x "$vgpu" && -e "$build/shim/libamd_smi.so.26" && -e "$build/shim/librocm_smi64.so.7" ]] ||
  { echo "SKIP: build vgpu, vgpuamdsmi and vgpursmi first"; exit 0; }
rvs=""
for c in "${VGPU_RVS:-}" "${VGPU_ROCM_PATH:-}/bin/rvs" "${ROCM_PATH:-}/bin/rvs" /opt/rocm/bin/rvs \
         $(ls -d "$HOME"/.local/share/rocm-*/opt/rocm-*/bin/rvs 2>/dev/null | sort -rV) "$(command -v rvs || true)"; do
  [[ -n "$c" && -x "$c" ]] && { rvs=$(readlink -f "$c"); break; }
done
[[ -n "$rvs" ]] || { echo "SKIP: no rvs found (install rocm-validation-suite, or set VGPU_RVS)"; exit 0; }
prefix=$(dirname "$(dirname "$rvs")")
conf="$prefix/share/rocm-validation-suite/conf"
[[ -d "$conf" ]] || { echo "SKIP: $conf is missing"; exit 0; }
if [[ $(id -u) != 0 ]] && ! unshare --user --map-root-user true 2>/dev/null; then
  echo "SKIP: no isolated session here (user namespaces are restricted)"; exit 0
fi
if objdump -p "$build/shim/libamdhip64.so.7" 2>/dev/null | awk '/NEEDED/ {print $2}' | grep -qE 'lib(a|t)san'; then
  echo "SKIP: the shim is built with a sanitizer, and RVS is not"; exit 0
fi
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_rvs.XXXXXX")"
trap 'rm -rf "$work"' EXIT
unset VGPU_GPU VGPU_DEVICE_COUNT VGPU_SESSION
export VGPU_TELEMETRY_PATH="$work/telemetry" VGPU_QUIET=1
mkdir -p "$VGPU_TELEMETRY_PATH"
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}

# Configurations of ours: what the simulator can answer for. The BARs are the
# MI300X's, as the PCI configuration space models them (256 GB, 2 MB and 512 KB);
# RVS's own smqt_single.conf expects an older part's 256 KB register BAR.
cat > "$work/smqt.conf" <<Y
actions:
- name: bars
  device: all
  module: smqt
  bar1_req_size: 17179869184
  bar1_base_addr_min: 0
  bar1_base_addr_max: 140737488355328
  bar2_req_size: 2097152
  bar2_base_addr_min: 0
  bar2_base_addr_max: 140737488355328
  bar4_req_size: 524288
  bar4_base_addr_min: 0
  bar4_base_addr_max: 140737488355328
  bar5_req_size: 131072
Y
gst() {  # gst <name> <error_inject>
cat <<Y
actions:
- name: $1
  device: all
  module: gst
  parallel: false
  log_interval: 1000
  ramp_interval: 2000
  duration: 3000
  hot_calls: 1
  copy_matrix: false
  target_stress: 0.001
  matrix_size_a: 128
  matrix_size_b: 128
  matrix_size_c: 128
  matrix_init: rand
  ops_type: sgemm
  lda: 128
  ldb: 128
  ldc: 128
  transa: 1
  transb: 0
  alpha: 1
  beta: 0
  self_check: true
  accuracy_check: true
  error_inject: $2
  error_freq: 2
  error_count: 1
Y
}
gst clean false > "$work/gst_clean.conf"
gst injected true > "$work/gst_injected.conf"
cat > "$work/babel.conf" <<Y
actions:
- name: babel
  device: all
  module: babel
  parallel: false
  count: 1
  num_iter: 2
  array_size: 4194304
  test_type: 1
  mibibytes: true
  o/p_csv: false
  copy: true
  mul: true
  add: true
  dot: true
  triad: true
Y

# One session runs them all; each writes its own output file.
cat > "$work/inside.sh" <<EOS
export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:$prefix/lib:$prefix/lib/rvs"
cd "$work"
"$rvs" -g > list.out 2>&1
for c in gpup_single:$conf/gpup_single.conf peqt_single:$conf/peqt_single.conf pesm_1:$conf/pesm_1.conf \
         smqt:$work/smqt.conf babel:$work/babel.conf gst_clean:$work/gst_clean.conf gst_injected:$work/gst_injected.conf; do
  timeout 120 "$rvs" -c "\${c#*:}" > "\${c%%:*}.out" 2>&1
  echo "\${c%%:*} \$?" >> status
done
EOS
timeout 900 "$vgpu" shell -y --gpu amd/mi300x --count 1 -c "bash $work/inside.sh" </dev/null >"$work/session.log" 2>&1
[[ -s "$work/list.out" ]] || { echo "FAIL  the session ran nothing"; sed 's/^/      /' "$work/session.log" | tail -10; exit 1; }
for f in "$work"/*.out; do sed -i 's/\x1b\[[0-9;]*m//g' "$f"; done

passes() {  # passes <output> -> "<n> PASS <m> FAIL" counted from RVS's summary table
  local p f
  p=$(grep -cE '^\| .* \| (PASS) +\|' "$1"); f=$(grep -cE '^\| .* \| (FAIL) +\|' "$1")
  echo "$p PASS $f FAIL"
}
expect "RVS lists the simulated MI300X, with the device id the PCI table gives it" "yes" \
  "$(grep -qE '^0000:01:00\.0 - GPU\[ *1 - *[0-9]+\] AMD Instinct MI300X \(Device 29857\)$' "$work/list.out" && echo yes || echo no)"
for m in gpup_single peqt_single pesm_1; do
  n=$(grep -cE '^\| .* \| (PASS|FAIL) +\|' "$work/$m.out")
  expect "$m: every action passes ($n of them)" "$n PASS 0 FAIL" "$(passes "$work/$m.out")"
  expect "$m: and there are some" "yes" "$([[ $n -gt 0 ]] && echo yes || echo no)"
done
# gpup prints the KFD node's properties: the ones the profile fixes (every action prints them).
for kv in "simd_count 1216" "wave_front_size 64" "gfx_target_version 90402" "num_xcc 8" "vendor_id 4098" "device_id 29857"; do
  expect "gpup reads $kv from the topology" "yes" \
    "$(grep -qE "gpup [0-9]+ ${kv}\$" "$work/gpup_single.out" && echo yes || echo no)"
done
expect "smqt: the BARs are the MI300X's" "1 PASS 0 FAIL" "$(passes "$work/smqt.out")"
expect "babel runs its five kernels" "1 PASS 0 FAIL" "$(passes "$work/babel.out")"
expect "and times each (copy, mul, add, triad, dot)" "5" "$(grep -cE '^[0-9]+ +(Copy|Mul|Add|Triad|Dot) +[0-9.]+ ' "$work/babel.out")"
expect "gst: SGEMM matches the host's and itself, run to run" "1 PASS 0 FAIL" "$(passes "$work/gst_clean.out")"
expect "and no check reports an error" "0" "$(grep -cE 'self-check error|accu-check error' "$work/gst_clean.out" | head -1)"
expect "gst: an element made wrong on purpose is caught by both checks" "yes" \
  "$(grep -q 'self-check error' "$work/gst_injected.out" && grep -q 'accu-check error' "$work/gst_injected.out" && echo yes || echo no)"
exit $fail
