#!/usr/bin/env bash
# SASS against PTX, kernel by kernel: each program below is built for each
# architecture's SASS (with its PTX beside it), run once on SASS and once on
# its PTX (VGPU_SASS=0), both with VGPU_KERNEL_DIGEST, and the digests --
# per launch, a hash of every allocation the kernel's arguments reach -- must
# be the same. e2e_sass_archs compares what the programs print; this compares
# the memory they leave behind, after every kernel, and names the first kernel
# that differs.
#
#   nvidia/tools/sass_ptx_digest.sh [build-dir]
#
# VGPU_DIGEST_ARCHS and VGPU_DIGEST_PROGRAMS narrow the lists; logs and
# digests go to VGPU_DIGEST_OUT (default: a temporary directory, kept on a
# difference). On a simulator without the SASS path (before nvidia/docs/sass.md
# existed) there is nothing to compare: it says so and passes.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="${1:-${VGPU_BUILD_DIR:-$root/build}}"
shim="$build/shim"

if ! grep -q VGPU_SASS_LOG "$root/nvidia/src/fatbin.cpp" 2>/dev/null; then
  echo "SKIP: this simulator has no SASS path (no VGPU_SASS_LOG in nvidia/src/fatbin.cpp); nothing to compare"
  exit 0
fi
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "FAIL: libvgpucudart is not built in $build"; exit 1; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "FAIL: no nvcc matching this shim's toolkit"; exit 1; }
nvcc_host_compiler_fix

declare -A gpu=([sm_75]=nvidia/t4 [sm_80]=nvidia/a100 [sm_86]=nvidia/rtx3060 [sm_89]=nvidia/l4
                [sm_90]=nvidia/h100 [sm_100]=nvidia/b200 [sm_120]=nvidia/rtx5090)
archs=(${VGPU_DIGEST_ARCHS:-sm_75 sm_86 sm_90 sm_100 sm_120})
# program:first architecture[:last]. Programs that print a verdict and take
# no arguments, spread over the ALU, memory, textures, atomics, cooperative
# launch, tensor cores and clusters.
progs=(${VGPU_DIGEST_PROGRAMS:-vector_add:75 device_functions:75 device_intrinsics:75 video_forms:75
       symbols:75 surface_oob:75 textures:75 texture_filtering:75 block_semaphore:75 cooperative_grid:75
       alloca_stack:75 atomic_cas:75 mma_fragment_layout:80 modern_dtypes:80 wmma_gemm:80 wmma_types:80
       dsmem_cluster:90})
supported="$("$nvcc_bin" --list-gpu-code 2>/dev/null)"
out="${VGPU_DIGEST_OUT:-$(mktemp -d "${TMPDIR:-/tmp}/vgpu-digest.XXXXXX")}"
mkdir -p "$out"
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"

jobs=()
for arch in "${archs[@]}"; do
  if ! grep -qx "$arch" <<< "$supported"; then echo "SKIP $arch: $nvcc_bin cannot target it"; continue; fi
  for p in "${progs[@]}"; do
    IFS=: read -r name first last <<< "$p"
    [[ -f "$root/nvidia/tests/e2e/$name.cu" ]] || { echo "SKIP $name: no nvidia/tests/e2e/$name.cu"; continue; }
    (( ${arch#sm_} >= first && ${arch#sm_} <= ${last:-999} )) && jobs+=("$name $arch")
  done
done
(( ${#jobs[@]} )) || { echo "FAIL: nothing to build"; exit 1; }

export nvcc_bin root out
export san="${san_flags[*]}"
printf '%s\n' "${jobs[@]}" | xargs -P "${VGPU_DIGEST_JOBS:-$(nproc)}" -L 1 bash -c '
  "$nvcc_bin" -std=c++17 -arch="$1" -cudart shared -w -Wno-deprecated-gpu-targets -Xlinker --as-needed $san \
    -I"$root/nvidia/tests/e2e" "$root/nvidia/tests/e2e/$0.cu" -o "$out/$0.$1" -lcuda > "$out/$0.$1.build" 2>&1 ||
    echo "build failed" >> "$out/$0.$1.build"'

fails=0 compared=0 fellback=0
for j in "${jobs[@]}"; do
  read -r p arch <<< "$j"
  bin="$out/$p.$arch"
  if [[ ! -x $bin ]]; then
    echo "FAIL $arch $p: does not build"; tail -5 "$bin.build" | sed 's/^/    /'; fails=1; continue
  fi
  require_shim_libs "$shim" "$bin" >/dev/null || { echo "SKIP $arch $p: needs a library this shim lacks"; continue; }
  rm -f "$bin.sass.digest" "$bin.ptx.digest"
  # One host thread: blocks that race on atomics would otherwise leave
  # different (equally right) memory from run to run.
  env=(VGPU_QUIET=1 VGPU_THREADS=1 VGPU_GPU="${gpu[$arch]}" LD_LIBRARY_PATH="$shim")
  [[ $p == runtime_conformance || $p == managed_vars ]] && env+=(VGPU_DEVICE_COUNT=2)
  (cd "$out" && env "${env[@]}" VGPU_SASS_LOG=1 VGPU_KERNEL_DIGEST="$bin.sass.digest" timeout 900 "$bin" > "$bin.sass.log" 2>&1); rs=$?
  (cd "$out" && env "${env[@]}" VGPU_SASS=0 VGPU_KERNEL_DIGEST="$bin.ptx.digest" timeout 900 "$bin" > "$bin.ptx.log" 2>&1); rp=$?
  if [[ $rs != 0 || $rp != 0 ]]; then
    echo "FAIL $arch $p: the SASS run exited $rs, the PTX run $rp"; fails=1; continue
  fi
  if ! grep -q "running SASS" "$bin.sass.log" || grep -q "running PTX instead of SASS" "$bin.sass.log"; then
    # Not a difference: the executor passed this binary's SASS over (or there
    # was none), so both runs were PTX. Counted, so a run that compared
    # nothing cannot pass unnoticed.
    echo "note $arch $p: ran PTX both times ($(grep -m1 -o 'running PTX instead of SASS.*' "$bin.sass.log" || echo 'no SASS ran'))"
    fellback=$((fellback + 1)); continue
  fi
  compared=$((compared + 1))
  if [[ ! -s "$bin.ptx.digest" ]]; then echo "FAIL $arch $p: no digest written (VGPU_KERNEL_DIGEST unsupported?)"; fails=1; continue; fi
  if cmp -s "$bin.sass.digest" "$bin.ptx.digest"; then
    echo "ok   $arch $p ($(wc -l < "$bin.ptx.digest") launches)"
  else
    first=$(diff "$bin.ptx.digest" "$bin.sass.digest" | grep -m1 '^<' | cut -c3-)
    echo "FAIL $arch $p: SASS and PTX leave different memory, first after launch: ${first%% *} $(cut -d' ' -f2 <<< "$first")"
    diff "$bin.ptx.digest" "$bin.sass.digest" | head -6 | cut -c1-200 | sed 's/^/    /'
    fails=1
  fi
done
echo "compared $compared program/architecture pairs on SASS and PTX; $fellback ran PTX both times"
if (( compared == 0 )); then echo "FAIL: no program ran its SASS, so nothing was compared"; fails=1; fi
[[ $fails == 0 ]] && { echo PASS; [[ -n ${VGPU_DIGEST_OUT:-} ]] || rm -rf "$out"; } || echo "FAIL (logs and digests in $out)"
exit $fails
