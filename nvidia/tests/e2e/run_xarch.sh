#!/usr/bin/env bash
# Cross-architecture comparison: one program, one binary, every NVIDIA profile.
#
# Every program below is built once, as PTX only (the lowest compute_NN nvcc accepts it for), and the same binary is run on the
# simulated GPU of each generation (VGPU_GPU=nvidia/<profile>, the PTX interpreter, one host thread). What it prints and what each
# launch leaves in every allocation (VGPU_KERNEL_DIGEST, the addresses dropped) are then compared between the profiles. Two
# profiles of one generation must agree bit for bit; across generations a difference is either real hardware (the card's own
# rule: it belongs in the table `expect` below with the evidence for each side) or a defect of a profile, and one that is not in
# the table fails.
#
#   run_xarch.sh            the short list (a pull request gate)
#   VGPU_NIGHTLY=1          every standalone program of nvidia/tests/e2e
#
# VGPU_XARCH_PROFILES (default below), VGPU_XARCH_PROGRAMS (names), VGPU_XARCH_KEEP (directory to copy the outputs to),
# VGPU_DIFF_JOBS (builds at once).
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
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-xarch.XXXXXX")"
trap '[[ -n ${VGPU_XARCH_KEEP:-} ]] && { mkdir -p "$VGPU_XARCH_KEEP"; cp -r "$work"/. "$VGPU_XARCH_KEEP"/; }; rm -rf "$work"' EXIT

# profile -> its compute capability times ten, the generation it stands for
declare -A cc=([t4]=75 [a100]=80 [a10g]=86 [l4]=89 [h100]=90 [h200]=90 [b200]=100 [rtx5090]=120)
profiles=(${VGPU_XARCH_PROFILES:-t4 a100 a10g l4 h100 h200 b200 rtx5090})

# Programs that fit the pull request gate: they print nothing about the device and exercise numerics, memory and warp behaviour.
short="vector_add device_intrinsics device_functions float_half_forms minmax_forms packed_int_forms simd_lane_forms mma_forms
  wmma_gemm wmma_types ldmatrix_forms stmatrix atomic_cas atomics_threads group_ballot_barrier div_approx mul_add_contraction
  texture_filtering surface_formatted shared_max named_barriers"
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
# Not runnable or not meaningful as one binary on every profile: they build per architecture or name a profile themselves.
skip=" arch_specific_cubin arch_specific_cubins card_edges_module_arch managed_module jitlink_lib sass_link_lib sass_tier sass_path sass_fastpath "

supported="$("$nvcc_bin" --list-gpu-code 2>/dev/null)"
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
export nvcc_bin root work rdc skip
export san="${san_flags[*]}" supported="${supported//$'\n'/ }"
# One build per virtual architecture, PTX only: a program that gates code on __CUDA_ARCH__ (the matrix instructions, wgmma, clusters)
# runs that code on the profiles whose compute capability is at least the build's, and the others are not run on it. The a-suffixed
# builds run on exactly their own generation (the PTX of an "a" target is not forward compatible).
# VGPU_XARCH_MODE=native instead builds each program for each generation (SASS and PTX, nvcc -arch=sm_NN) and runs every profile on
# the build for its own generation, on the SASS engine: what a user who compiles for the card in front of them gets.
mode="${VGPU_XARCH_MODE:-ptx}"
if [[ $mode == native ]]; then passes="75 80 86 89 90 100 120"; else passes="75 80 90 100 90a 100a"; fi
export passes mode
printf '%s\n' "${progs[@]}" | sort -u | xargs -P "${VGPU_DIFF_JOBS:-$("$root/ci/jobs.sh" -p sass)}" -L 1 bash -c '
  [[ $skip == *" $0 "* || ! -f "$root/nvidia/tests/e2e/$0.cu" ]] && exit 0
  sep=(); [[ $rdc == *" $0 "* ]] && sep=(-rdc=true -lcudadevrt)
  for a in $passes; do
    [[ " $supported " == *" sm_$a "* ]] || continue
    if [[ $mode == native ]]; then gen=(-arch=sm_$a); else gen=(-gencode arch=compute_$a,code=compute_$a); fi
    if "$nvcc_bin" -std=c++17 "${gen[@]}" -cudart shared -w -Wno-deprecated-gpu-targets -Xlinker --as-needed $san \
         -I"$root/nvidia/tests/e2e" "$root/nvidia/tests/e2e/$0.cu" -o "$work/$0.v$a" -lcuda "${sep[@]}" 2> "$work/$0.v$a.build"; then
      touch "$work/$0.v$a.ok"
    fi
  done'

strip_addr() { sed -E 's/ [0-9a-f]+:/ /g' "$1"; }
fails=0; compared=0; known_n=0; nobuild=0

# What differs between architectures on purpose. <program> -> the profiles (space separated groups, '|' between the groups) that
# must give one answer within a group and may differ between groups, and why. A program not here must give one answer everywhere.
declare -A expect=()
if [[ -f "$root/nvidia/tests/e2e/xarch_expect.sh" ]]; then . "$root/nvidia/tests/e2e/xarch_expect.sh"; fi

if [[ $mode == native ]]; then
  for p in "${progs[@]}"; do
    [[ $skip == *" $p "* || ! -f "$root/nvidia/tests/e2e/$p.cu" ]] && continue
    ran=()
    for prof in "${profiles[@]}"; do
      c=${cc[$prof]}; bin="$work/$p.v$((c / 10 * 10 + c % 10))"
      bin="$work/$p.v$c"
      [[ -f $bin.ok ]] || continue
      require_shim_libs "$shim" "$bin" >/dev/null || continue
      env=(VGPU_QUIET=1 VGPU_GPU="nvidia/$prof" VGPU_THREADS=1 LD_LIBRARY_PATH="$shim" $(both_shims_env "$shim") VGPU_KERNEL_DIGEST="$bin.$prof.dig")
      [[ $two_devices == *" $p "* ]] && env+=(VGPU_DEVICE_COUNT=2)
      rm -f "$bin.$prof.dig"
      (cd "$work" && env "${env[@]}" timeout 900 "$bin" > "$bin.$prof.raw" 2>&1); echo $? > "$bin.$prof.rc"
      touch "$bin.$prof.dig"
      { cat "$bin.$prof.rc"; grep -v '^\[vgpu\]' "$bin.$prof.raw"; echo "--digest"; strip_addr "$bin.$prof.dig"; } > "$bin.$prof.sig"
      ran+=("$prof")
    done
    (( ${#ran[@]} >= 2 )) || continue
    declare -A grp=(); order=()
    for prof in "${ran[@]}"; do
      h=$(sha1sum < "$work/$p.v${cc[$prof]}.$prof.sig" | cut -c1-12)
      [[ -n ${grp[$h]:-} ]] || order+=("$h")
      grp[$h]="${grp[$h]:-} $prof"
    done
    compared=$((compared + 1))
    if (( ${#order[@]} == 1 )); then echo "ok   $p@native (${#ran[@]} profiles identical)"; else
      layout=""; for h in "${order[@]}"; do layout+="${layout:+ | }$(echo ${grp[$h]})"; done
      want="${expect[$p@native]:-}"
      if [[ -n $want && "$layout" == "$want" ]]; then known_n=$((known_n + 1)); echo "known $p@native: $layout"; else
        fails=1; echo "FAIL $p@native: the profiles do not agree: $layout"
        first=${order[0]}; r0=${grp[$first]# }; r0=${r0%% *}
        for h in "${order[@]:1}"; do
          r1=${grp[$h]# }; r1=${r1%% *}
          echo "    $r0 < > $r1:"
          diff "$work/$p.v${cc[$r0]}.$r0.sig" "$work/$p.v${cc[$r1]}.$r1.sig" | head -6 | cut -c1-200 | sed 's/^/      /'
        done
      fi
    fi
    unset grp
  done
  profiles=()   # the loop below is for the shared-binary mode
fi

for p in "${progs[@]}"; do
  [[ $skip == *" $p "* || ! -f "$root/nvidia/tests/e2e/$p.cu" ]] && continue
  built=0
  for virt in $passes; do
    bin="$work/$p.v$virt"
    [[ -f $bin.ok ]] || continue
    built=1
    require_shim_libs "$shim" "$bin" >/dev/null || { echo "SKIP $p: needs a library this shim lacks"; break; }
    ran=()
    for prof in "${profiles[@]}"; do
      c=${cc[$prof]:-}; [[ -n $c ]] || { echo "unknown profile $prof"; exit 2; }
      base=${virt%a}
      if [[ $virt == *a ]]; then (( c == base )) || continue; else (( c >= base )) || continue; fi
      env=(VGPU_QUIET=1 VGPU_GPU="nvidia/$prof" VGPU_THREADS=1 VGPU_SASS=0 LD_LIBRARY_PATH="$shim" $(both_shims_env "$shim")
           VGPU_KERNEL_DIGEST="$bin.$prof.dig")
      [[ $two_devices == *" $p "* ]] && env+=(VGPU_DEVICE_COUNT=2)
      rm -f "$bin.$prof.dig"
      (cd "$work" && env "${env[@]}" timeout 900 "$bin" > "$bin.$prof.raw" 2>&1); echo $? > "$bin.$prof.rc"
      touch "$bin.$prof.dig"
      { cat "$bin.$prof.rc"; grep -v '^\[vgpu\]' "$bin.$prof.raw"; echo "--digest"; strip_addr "$bin.$prof.dig"; } > "$bin.$prof.sig"
      ran+=("$prof")
    done
    (( ${#ran[@]} >= 2 )) || continue
    declare -A grp=(); order=()
    for prof in "${ran[@]}"; do
      h=$(sha1sum < "$bin.$prof.sig" | cut -c1-12)
      [[ -n ${grp[$h]:-} ]] || order+=("$h")
      grp[$h]="${grp[$h]:-} $prof"
    done
    compared=$((compared + 1))
    if (( ${#order[@]} == 1 )); then
      echo "ok   $p@$virt (${#ran[@]} profiles identical)"
    else
      layout=""; for h in "${order[@]}"; do layout+="${layout:+ | }$(echo ${grp[$h]})"; done
      want="${expect[$p@$virt]:-${expect[$p]:-}}"
      if [[ -n $want && "$layout" == "$want" ]]; then
        known_n=$((known_n + 1)); echo "known $p@$virt: $layout"
      else
        fails=1; echo "FAIL $p@$virt: the profiles do not agree: $layout"
        first=${order[0]}; r0=${grp[$first]# }; r0=${r0%% *}
        for h in "${order[@]:1}"; do
          r1=${grp[$h]# }; r1=${r1%% *}
          echo "    $r0 < > $r1:"
          diff "$bin.$r0.sig" "$bin.$r1.sig" | head -6 | cut -c1-200 | sed 's/^/      /'
        done
      fi
    fi
    unset grp
  done
  (( built )) || { nobuild=$((nobuild + 1)); echo "nobuild $p: $(grep -h -m1 -E 'error' "$work/$p".v*.build 2>/dev/null | head -1 | cut -c1-140)"; }
done
echo "xarch: $compared programs compared on ${#profiles[@]} profiles; $nobuild did not build; $known_n known differences"
if (( compared == 0 )); then echo "FAIL: nothing was compared"; fails=1; fi
[[ $fails == 0 ]] && echo PASS || echo "FAIL: xarch"
exit $fails
