#!/usr/bin/env bash
# `vgpu ncu`, the session's ncu: Nsight Compute's command line answered from the
# simulator's counters. Checks the values against kernels whose answers follow
# from their addresses alone, that only exactly-counted metrics are listed, that
# the rest are named rather than invented, and that inside `vgpu shell` `ncu` is
# this tool. Skips if nvcc is unavailable.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="${VGPU_BUILD_DIR:-$root/build}"
shim="$build/shim"
src="$root/nvidia/tests/e2e/ncu_access.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_ncu_$$"
command -v nvcc >/dev/null 2>&1 || { echo "SKIP: nvcc not found"; exit 0; }
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#cudart_libs[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
mkdir -p "$out"
trap 'rm -rf "$out"' EXIT
nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets \
     $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out/access"
if ! require_shim_libs "$shim" "$out/access"; then exit 0; fi
export VGPU_QUIET=1 VGPU_GPU=nvidia/a10 LD_LIBRARY_PATH="$shim"
fail() { echo "FAIL: $*"; exit 1; }

# Only metrics counted exactly are listed; one that needs a DRAM model is not.
"$build/vgpu" ncu --query-metrics --query-metrics-mode all --devices 0 >"$out/query.txt"
listed=$(grep -cE '^(smsp|l1tex)__' "$out/query.txt" || true)
[[ "$listed" == 12 ]] || fail "expected 12 listed metrics, got $listed"
grep -q '^dram__' "$out/query.txt" && fail "a DRAM metric was listed"

# The values. A metric asked for that is not counted is named on stderr and
# left out of the report, not given a number.
metrics=l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
metrics+=,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
metrics+=,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,smsp__inst_executed_op_shared_st.sum
metrics+=,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
metrics+=,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,dram__bytes_read.sum
"$build/vgpu" ncu --csv --set full --metrics "$metrics" --log-file "$out/report.csv" \
    "$out/access" >"$out/stdout.txt" 2>"$out/stderr.txt" || fail "ncu exited $?"
grep -q 'app done' "$out/stdout.txt" || fail "the application's output did not come through"
grep -q 'dram__bytes_read.sum' "$out/stderr.txt" || fail "the uncounted metric was not named"
grep -q 'dram__' "$out/report.csv" && fail "an uncounted metric was reported"

value() {  # kernel metric
  awk -F'","' -v k="$1" -v m="$2" 'index($5, k"(")==1 && $13==m { v=$15; sub(/"$/, "", v); print v }' \
      "$out/report.csv"
}
check() {  # kernel metric expected
  local got; got=$(value "$1" "$2")
  [[ "$got" == "$3" ]] || fail "$1 $2 = '$got', expected $3"
}
check coalesced l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum 4
check coalesced l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum 1
check coalesced smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct 100.00
check coalesced l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum 4
check strided l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum 32
check strided smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct 12.50
check vector4 l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum 16
check vector4 smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct 100.00
check vector4 l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum 16
check banks smsp__inst_executed_op_shared_st.sum 1
check banks l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum 1
check banks l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum 0

# An .ncu-rep is Nsight Compute's own format; asking for one is refused.
if "$build/vgpu" ncu -o "$out/rep" "$out/access" >/dev/null 2>&1; then fail "--export was accepted"; fi

# Inside a session, `ncu` is this tool rather than NVIDIA's, which cannot attach.
iso=()
unshare --user --map-root-user true >/dev/null 2>&1 || iso=(--no-isolate)
ver=$("$build/vgpu" shell --gpu nvidia/a10 --count 1 "${iso[@]}" -y -c "ncu --version" 2>/dev/null | tail -1)
[[ "$ver" == VirtualGPU\ ncu* ]] || fail "in a session, ncu --version said '$ver'"
echo "ncu over simulated counters: PASS"
