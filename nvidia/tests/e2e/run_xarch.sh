#!/usr/bin/env bash
# Cross-architecture comparison: one program, every NVIDIA generation.
#
# A program must not care which profile it runs on unless the hardware does. Each program below is run on the simulated t4 (sm_75), a100
# (sm_80), a10g (sm_86), l4 (sm_89), h100 and h200 (sm_90), b200 (sm_100) and rtx5090 (sm_120), and what it prints and what each launch
# leaves in every allocation (VGPU_KERNEL_DIGEST, the addresses dropped) are compared between the profiles.
#
#   VGPU_XARCH_MODE=ptx      (default) built once per virtual architecture, as PTX only (nvcc -gencode arch=compute_NN,code=compute_NN, NN =
#                            75, 80, 90, 100 and the a-suffixed 90a, 100a); the one binary is run by the PTX interpreter on every profile
#                            whose compute capability is at least NN. What differs is the profile, not the compiler.
#   VGPU_XARCH_MODE=native   built for each generation (nvcc -arch=sm_NN: its SASS beside its PTX) and run by the SASS engine on the profile
#                            of that generation: what a user who compiles for the card in front of them gets.
#
# Profiles of one generation (h100 and h200) must agree always. Across generations a difference is real hardware (the card's own rule),
# a program that asks the device what it is (SM count, memory, limits), or a defect of a profile or of an engine, and only the first
# two are allowed: xarch_expect.sh names them, with the evidence. Anything else fails.
#
#   VGPU_NIGHTLY=1           every standalone program of nvidia/tests/e2e (not only the short list)
#   VGPU_XARCH_PROFILES      the profiles (default as above); VGPU_XARCH_PROGRAMS names programs; VGPU_DIFF_JOBS caps the builds
#   VGPU_XARCH_KEEP=<dir>    copies the binaries, outputs and digests there (for looking at a failure)
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
mode="${VGPU_XARCH_MODE:-ptx}"
case "$mode" in ptx|native) ;; *) echo "usage: VGPU_XARCH_MODE=ptx|native $0"; exit 2 ;; esac
if [[ $mode == native ]] && ! grep -rq VGPU_SASS_LOG "$root/nvidia/src" 2>/dev/null; then echo "SKIP: this simulator has no SASS path"; exit 0; fi
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-xarch.XXXXXX")"
trap '[[ -n ${VGPU_XARCH_KEEP:-} ]] && { mkdir -p "$VGPU_XARCH_KEEP"; cp -r "$work"/. "$VGPU_XARCH_KEEP"/; }; rm -rf "$work"' EXIT

# profile -> its compute capability times ten
declare -A cc=([t4]=75 [a100]=80 [a10g]=86 [l4]=89 [h100]=90 [h200]=90 [b200]=100 [rtx5090]=120)
profiles=(${VGPU_XARCH_PROFILES:-t4 a100 a10g l4 h100 h200 b200 rtx5090})

# The short list (a pull request): numerics, memory and warp behaviour that print nothing about the device.
short="vector_add device_functions float_half_forms packed_int_forms simd_lane_forms mma_forms wmma_gemm wmma_types ldmatrix_forms stmatrix
  atomic_cas atomics_threads group_ballot_barrier div_approx mul_add_contraction texture_filtering surface_formatted named_barriers
  ptx_forms ptx_warp_mem constant_table_xor lop3_pred_chain"
if [[ -n ${VGPU_NIGHTLY:-} ]]; then
  needs_lib='cublas|cudnn|cufft|cusolver|cusparse|curand|nvrtc|nvml|cupti|npp|nvjpeg|nccl|cutensor|cudss|nvcomp|nvshmem|cufile|nvJitLink|nvfatbin|cutlass|nvcuvid|nvenc|cuvid|mpi'
  progs=(vector_add device_intrinsics device_functions div_approx)
  for f in "$root"/nvidia/tests/e2e/*.cu; do
    n=$(basename "$f" .cu)
    grep -q -E "$needs_lib" "$f" || progs+=("$n")
  done
else
  progs=($short)
fi
[[ -n ${VGPU_XARCH_PROGRAMS:-} ]] && progs=(${VGPU_XARCH_PROGRAMS})
rdc=" dynamic_parallelism cdp_device_api rdc_device_api "
two_devices=" managed_vars runtime_conformance p2p_managed "
# Not meaningful as one program on every profile: they name an architecture or a profile themselves, or link what nvcc here cannot.
skip=" arch_specific_cubin arch_specific_cubins card_edges_module_arch managed_module jitlink_lib sass_link_lib sass_tier sass_path sass_fastpath "

# What may differ, from xarch_expect.sh: xarch_groups[<program>] = the profiles that must agree with each other, groups separated by "|"
# (a program not there: all of them; <program>@ptx or <program>@native for one mode); xarch_nodigest = programs whose memory digest
# depends on host threads.
declare -A xarch_groups=()
xarch_nodigest=" "
if [[ -f "$root/nvidia/tests/e2e/xarch_expect.sh" ]]; then . "$root/nvidia/tests/e2e/xarch_expect.sh"; fi

supported="$("$nvcc_bin" --list-gpu-code 2>/dev/null)"
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
if [[ $mode == native ]]; then passes="75 80 86 89 90 100 120"; else passes="75 80 90 100 90a 100a"; fi
export nvcc_bin root work rdc skip mode passes
export san="${san_flags[*]}" supported="${supported//$'\n'/ }"
printf '%s\n' "${progs[@]}" | sort -u | xargs -P "${VGPU_DIFF_JOBS:-$("$root/ci/jobs.sh" -p sass)}" -L 1 bash -c '
  [[ $skip == *" $0 "* || ! -f "$root/nvidia/tests/e2e/$0.cu" ]] && exit 0
  sep=(); [[ $rdc == *" $0 "* ]] && sep=(-rdc=true -lcudadevrt)
  for a in $passes; do
    [[ " $supported " == *" sm_${a%a} "* ]] || continue
    if [[ $mode == native ]]; then gen=(-arch=sm_$a); else gen=(-gencode arch=compute_$a,code=compute_$a); fi
    if "$nvcc_bin" -std=c++17 "${gen[@]}" -cudart shared -w -Wno-deprecated-gpu-targets -Xlinker --as-needed $san \
         -I"$root/nvidia/tests/e2e" "$root/nvidia/tests/e2e/$0.cu" -o "$work/$0.v$a" -lcuda "${sep[@]}" 2> "$work/$0.v$a.build"; then
      touch "$work/$0.v$a.ok"
    fi
  done'

strip_addr() { sed -E 's/ [0-9a-f]+:/ /g' "$1"; }
fails=0; compared=0; excused=0; nobuild=0

# run_profile <program> <binary> <profile>: the signature of one run in <binary>.<profile>.sig
run_profile() {
  local p=$1 bin=$2 prof=$3 env
  env=(VGPU_QUIET=1 VGPU_GPU="nvidia/$prof" VGPU_THREADS=1 LD_LIBRARY_PATH="$shim" $(both_shims_env "$shim") VGPU_KERNEL_DIGEST="$bin.$prof.dig")
  [[ $mode == ptx ]] && env+=(VGPU_SASS=0)
  [[ $two_devices == *" $p "* ]] && env+=(VGPU_DEVICE_COUNT=2)
  rm -f "$bin.$prof.dig"
  (cd "$work" && env "${env[@]}" timeout 900 "$bin" > "$bin.$prof.raw" 2>&1); echo $? > "$bin.$prof.rc"
  touch "$bin.$prof.dig"
  { cat "$bin.$prof.rc"; grep -v '^\[vgpu\]' "$bin.$prof.raw"
    if [[ $xarch_nodigest != *" $p "* ]]; then echo "--digest"; strip_addr "$bin.$prof.dig"; fi; } > "$bin.$prof.sig"
}

# compare <program> <tag> <profile>...: every group of profiles that must agree does (sigfile <profile> names a profile's signature)
compare() {
  local p=$1 tag=$2; shift 2
  local ran=("$@") groups prof first r bad="" g members present
  local want="${xarch_groups[$p@$mode]:-${xarch_groups[$p]:-}}"
  if [[ -n $want ]]; then groups="$want"; else groups="${ran[*]}"; fi
  local oldifs=$IFS
  IFS='|'; local garr=($groups); IFS=$oldifs
  for g in "${garr[@]}"; do
    read -r -a members <<< "$g"
    present=()
    for prof in "${members[@]}"; do for r in "${ran[@]}"; do [[ $r == "$prof" ]] && present+=("$prof"); done; done
    (( ${#present[@]} >= 2 )) || continue
    first=${present[0]}
    for prof in "${present[@]:1}"; do
      if ! cmp -s "$(sigfile "$first")" "$(sigfile "$prof")"; then
        bad+="$first/$prof "
        echo "    $first < > $prof:"
        diff "$(sigfile "$first")" "$(sigfile "$prof")" | head -6 | cut -c1-200 | sed 's/^/      /'
      fi
    done
  done
  compared=$((compared + 1))
  if [[ -n $bad ]]; then echo "FAIL $p@$tag: profiles that must agree do not: $bad"; fails=1
  elif [[ -n $want ]]; then excused=$((excused + 1)); echo "ok   $p@$tag (agrees within its groups: $want)"
  else echo "ok   $p@$tag (${#ran[@]} profiles identical)"; fi
}

for p in "${progs[@]}"; do
  [[ $skip == *" $p "* || ! -f "$root/nvidia/tests/e2e/$p.cu" ]] && continue
  built=0
  if [[ $mode == native ]]; then
    ran=()
    for prof in "${profiles[@]}"; do
      c=${cc[$prof]:-}; [[ -n $c ]] || { echo "unknown profile $prof"; exit 2; }
      bin="$work/$p.v$c"
      [[ -f $bin.ok ]] || continue
      built=1
      require_shim_libs "$shim" "$bin" >/dev/null || continue
      run_profile "$p" "$bin" "$prof"; ran+=("$prof")
    done
    sigfile() { echo "$work/$p.v${cc[$1]}.$1.sig"; }
    (( ${#ran[@]} >= 2 )) && compare "$p" native "${ran[@]}"
  else
    for virt in $passes; do
      bin="$work/$p.v$virt"
      [[ -f $bin.ok ]] || continue
      built=1
      require_shim_libs "$shim" "$bin" >/dev/null || break
      ran=()
      for prof in "${profiles[@]}"; do
        c=${cc[$prof]:-}; [[ -n $c ]] || { echo "unknown profile $prof"; exit 2; }
        base=${virt%a}
        if [[ $virt == *a ]]; then (( c == base )) || continue; else (( c >= base )) || continue; fi
        run_profile "$p" "$bin" "$prof"; ran+=("$prof")
      done
      sigfile() { echo "$bin.$1.sig"; }
      (( ${#ran[@]} >= 2 )) && compare "$p" "$virt" "${ran[@]}"
    done
  fi
  (( built )) || { nobuild=$((nobuild + 1)); echo "nobuild $p: $(grep -h -m1 -E 'error' "$work/$p".v*.build 2>/dev/null | head -1 | cut -c1-140)"; }
done
echo "xarch ($mode): $compared comparisons of ${#progs[@]} programs on ${#profiles[@]} profiles; $excused only within their groups (xarch_expect.sh); $nobuild did not build"
if (( compared == 0 )); then echo "FAIL: nothing was compared"; fails=1; fi
[[ $fails == 0 ]] && echo PASS || echo "FAIL: xarch"
exit $fails
