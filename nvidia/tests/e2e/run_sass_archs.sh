#!/usr/bin/env bash
# SASS on every architecture (nvidia/docs/sass.md). A program built with
# -arch=sm_XX carries that generation's SASS and its PTX, and the simulator
# runs the SASS by default. Each program below, built so for each generation
# from Turing to Blackwell, runs on that generation's simulated GPU twice: by
# default, where the log must say its SASS ran, and on its PTX (VGPU_SASS=0).
# Both runs must pass and print the same.
#
# Hopper's own programs -- warpgroup MMA, TMA, tensor maps rewritten in place,
# stmatrix -- are built for sm_90a and run on the H100 the same way, and
# Blackwell's tensor core (tcgen05_gemm) for sm_100a on the B200; the ones
# written with CuTe need CUTLASS's headers (cutlass_fetch.sh), and skip,
# saying so, without them.
#
# An nvcc too old for an architecture (sm_100 and sm_120 need CUDA 12.8) skips
# it and says so. VGPU_SASS_ARCHS and VGPU_SASS_PROGRAMS narrow the lists;
# VGPU_SASS_JOBS caps the builds run at once (default: one per CPU).
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
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-sass-archs.XXXXXX")"
trap 'rm -rf "$work"' EXIT

declare -A gpu=([sm_75]=nvidia/t4 [sm_80]=nvidia/a100 [sm_86]=nvidia/rtx3060 [sm_89]=nvidia/l4
                [sm_90]=nvidia/h100 [sm_90a]=nvidia/h100 [sm_100]=nvidia/b200 [sm_100a]=nvidia/b200
                [sm_120]=nvidia/rtx5090 [sm_120a]=nvidia/rtx5090)
archs=(${VGPU_SASS_ARCHS:-sm_75 sm_80 sm_86 sm_89 sm_90 sm_90a sm_100 sm_100a sm_120 sm_120a})
# program:first architecture it builds for (the MMA programs need sm_80)
# [:last one it runs on (runtime_conformance checks a T4's properties;
# sync_masks' values are what an RTX 3060's ptxas code did).
progs=(${VGPU_SASS_PROGRAMS:-sass_archs:75 vector_add:75 device_functions:75 device_intrinsics:75 video_forms:75
       runtime_conformance:75:75 symbols:75 surface_oob:75 textures:75 texture_filtering:75 texture_gather:75
       texture_layers:75 texture_forms:75 surface_formatted:75 texture_mipmaps:75 texture_mip_layers:75 texture_srgb:75 texture_int_coords:75 border_colour:75 block_semaphore:75 cooperative_grid:75 alloca_stack:75 managed_vars:75 shared_max:75 named_barriers:75 smid:75 smem_size_regs:75
       sync_masks:86:86 dynamic_parallelism:75 cdp_device_api:75 cdp1_device_sync:75:89 rdc_device_api:75 large_params:75 shared_atomics64:75 waterfall:90 mma_forms:80
       mma_fragment_layout:80 modern_dtypes:80 wmma_gemm:80 wmma_types:80 dsmem_cluster:90 cooperative_cluster:90
       wgmma_cute:90a tma_gemm_cute:90a tma_reduce_cute:90a tma_im2col:90a tensormap_replace_cute:90a
       stmatrix:90a setmaxnreg:90a setmaxnreg:100a bulk_copy:90a tcgen05_gemm:100a tmem_alloc_pair:100a mma_blockscale:120a ldmatrix_forms:100a ldmatrix_forms:120a narrow_cvt:100a narrow_cvt:120a})
cute=" wgmma_cute tma_gemm_cute tma_reduce_cute tensormap_replace_cute "
# Built the way a program that uses dynamic parallelism is (-rdc=true, linked
# with cudadevrt): the linked cubin carries the device runtime library and the
# relocations that come with it.
rdc=" dynamic_parallelism cdp_device_api cdp1_device_sync rdc_device_api "
# The first device runtime, which CUDA 12 and 13 still build for parts before Hopper
# on request: cudaDeviceSynchronize from a kernel is among its calls.
cdp1=" cdp1_device_sync "

supported="$("$nvcc_bin" --list-gpu-code 2>/dev/null)"
# Hopper's own programs use PTX newer than CUDA 12.0's (tensormap.replace is
# 8.3); they are built with 12.8 and later.
release="$("$nvcc_bin" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')"
read -r rmaj rmin <<< "$release"
if (( ${rmaj:-0} * 100 + ${rmin:-0} < 1208 )); then
  echo "SKIP sm_90a: its programs need CUDA 12.8 or later (this nvcc is $rmaj.$rmin)"
  kept=()
  for a in "${archs[@]}"; do [[ $a == sm_90a ]] || kept+=("$a"); done
  archs=("${kept[@]}")
fi
# mma_forms' sparse forms are more than CUDA 12.0's ptxas assembles; CUDA 13's
# does.
if (( ${rmaj:-0} < 13 )); then
  echo "SKIP mma_forms: its SASS needs CUDA 13's ptxas (this nvcc is $rmaj.$rmin)"
  kept=()
  for p in "${progs[@]}"; do [[ ${p%%:*} == mma_forms ]] || kept+=("$p"); done
  progs=("${kept[@]}")
fi
# sync_masks' expected values are what an RTX 3060 ran from CUDA 13.0's ptxas
# (the code it generates for a mask it cannot see through); CUDA 12's lowers
# some of those cases differently, so its values are not comparable.
if (( ${rmaj:-0} < 13 )); then
  echo "SKIP sync_masks: its expected values are CUDA 13's ptxas code (this nvcc is $rmaj.$rmin)"
  kept=()
  for p in "${progs[@]}"; do [[ ${p%%:*} == sync_masks ]] || kept+=("$p"); done
  progs=("${kept[@]}")
fi
if (( ${#progs[@]} == 0 )); then
  echo "SKIP: no program left to build for this nvcc"
  exit 0
fi
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
jobs=()
for arch in "${archs[@]}"; do
  if ! grep -qx "${arch%a}" <<< "$supported"; then echo "SKIP $arch: this nvcc cannot target it"; continue; fi
  for p in "${progs[@]}"; do
    IFS=: read -r name first last <<< "$p"
    # "90a": that architecture's own image only; otherwise a range.
    if [[ $first == *a || $arch == *a ]]; then
      [[ $first == "${arch#sm_}" ]] && jobs+=("$name $arch")
    elif (( ${arch#sm_} >= first && ${arch#sm_} <= ${last:-999} )); then
      jobs+=("$name $arch")
    fi
  done
done
(( ${#jobs[@]} )) || { echo "SKIP: no architecture this nvcc can target"; exit 0; }

# CUTLASS's headers, for the CuTe programs: fetched (once) when one is listed.
cutlass=""
for j in "${jobs[@]}"; do
  if [[ $cute == *" ${j%% *} "* ]]; then
    cutlass="$(build="${VGPU_BUILD_DIR:-$root/build}"; . "$root/nvidia/tests/e2e/cutlass_fetch.sh" >/dev/null 2>&1; echo "$cutlass")"
    [[ -d "$cutlass/include" ]] || { echo "SKIP the CuTe programs: CUTLASS could not be fetched"; cutlass=""; }
    break
  fi
done
if [[ -z $cutlass ]]; then
  kept=()
  for j in "${jobs[@]}"; do [[ $cute == *" ${j%% *} "* ]] || kept+=("$j"); done
  jobs=("${kept[@]}")
fi

# Builds, several at once. --as-needed keeps libcuda out of the programs that
# never call the driver API, so a sanitizer build can still run them (below).
export nvcc_bin root work cutlass rdc cdp1
export san="${san_flags[*]}"
printf '%s\n' "${jobs[@]}" | xargs -P "${VGPU_SASS_JOBS:-$(nproc)}" -L 1 bash -c '
  sep=(); [[ $rdc == *" $0 "* ]] && sep=(-rdc=true -lcudadevrt)
  [[ $cdp1 == *" $0 "* ]] && sep+=(-DCUDA_FORCE_CDP1_IF_SUPPORTED -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING)
  "$nvcc_bin" -std=c++17 -arch="$1" -cudart shared -w -Wno-deprecated-gpu-targets -Xlinker --as-needed $san \
    ${cutlass:+-O1 --expt-relaxed-constexpr -I"$cutlass/include"} \
    -I"$root/nvidia/tests/e2e" "$root/nvidia/tests/e2e/$0.cu" -o "$work/$0.$1" -lcuda "${sep[@]}" 2> "$work/$0.$1.build" ||
    echo "build failed" >> "$work/$0.$1.build"'

fails=0
check() {
  local p=$1 arch=$2 bin="$work/$1.$2" sass ptx rs rp
  if [[ ! -x $bin ]]; then
    echo "FAIL $arch $p: does not build"; sed 's/^/    /' "$bin.build" | tail -5; fails=1; return
  fi
  require_shim_libs "$shim" "$bin" >/dev/null || return
  # Some programs load libcuda beside libcudart (linked, or fetched as
  # runtime_conformance fetches a driver function): both_shims_env is what a
  # sanitizer build needs for that.
  local env=(VGPU_QUIET=1 VGPU_GPU=${gpu[$arch]} LD_LIBRARY_PATH="$shim" $(both_shims_env "$shim"))
  [[ $p == runtime_conformance ]] && env+=(VGPU_DEVICE_COUNT=2)   # as run_runtime_conformance.sh runs it
  [[ $p == managed_vars ]] && env+=(VGPU_DEVICE_COUNT=2)          # as its own test runs it: two devices
  sass="$(cd "$work" && env "${env[@]}" VGPU_SASS_LOG=1 timeout 600 "$bin" 2>&1)"; rs=$?
  ptx="$(cd "$work" && env "${env[@]}" VGPU_SASS=0 timeout 600 "$bin" 2>&1)"; rp=$?
  local why=""
  grep -q "running SASS" <<< "$sass" || why="its SASS did not run"
  grep -q "running PTX instead of SASS" <<< "$sass" && why="fell back to PTX: $(grep -m1 'running PTX instead' <<< "$sass")"
  [[ $rs == 0 ]] || why="${why:+$why; }the SASS run failed (exit $rs)"
  [[ $rp == 0 ]] || why="${why:+$why; }the PTX run failed (exit $rp)"
  [[ "$(grep -v '^\[vgpu\] running' <<< "$sass")" == "$ptx" ]] || why="${why:+$why; }the SASS and PTX runs print differently"
  if [[ -n $why ]]; then
    echo "FAIL $arch $p: $why"
    diff <(echo "$ptx") <(grep -v '^\[vgpu\] running' <<< "$sass") | head -8 | sed 's/^/    /'
    fails=1
  else
    echo "ok   $arch $p"
  fi
}
for j in "${jobs[@]}"; do check $j; done
[[ $fails == 0 ]] && echo "PASS" || echo "FAIL: SASS on every architecture"
exit $fails
