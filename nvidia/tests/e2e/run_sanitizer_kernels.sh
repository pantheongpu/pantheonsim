#!/usr/bin/env bash
# What compute-sanitizer reports for a kernel with a known defect, and the simulator's answer to the same kernel (sanitizer_kernels.cu):
# memcheck's invalid and misaligned accesses (global and shared), racecheck's hazards between warps on shared memory, and the
# kernels that must run clean. Each case is a process (a fault kills the context), run on both engines -- SASS, which the nvcc binary
# carries, and PTX (VGPU_SASS=0) -- for an sm_86 card and an sm_75 one. The two engines and the two cards must agree on the error
# code the runtime gives afterwards (700 an invalid access, 716 a misaligned one; 719, the code this simulator gives a race, which a
# card never reports) and on the kind of error and the access's direction and place the diagnostic names.
#
# One difference is known and checked as a difference: on sm_80 and later the driver keeps 1 KiB of shared memory behind each block's
# own, and the SASS engine models it, so a kernel that overruns its shared array by a word lands in that region and runs, as on a card
# without the sanitizer (compute-sanitizer reports it; the PTX engine, which has no such region, does too). sm_75 has no such
# region and both engines report it.
#
# Not covered, because the simulator does not do it: initcheck (a read of device memory nothing has written; the simulator hands out
# zeroed memory and says nothing) and synccheck. See docs/diagnostics.md.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/sanitizer_kernels.cu"
tmp="${TMPDIR:-/tmp}/vgpu_e2e_sanitizer_kernels_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
trap 'rm -f "$tmp".*' EXIT

# <case>@<error code>@<regex the diagnostic must match, or - for none>@<output value, or ->@<flag>
# r: also run with the race detector (VGPU_RACE=1); R: only with it.
cases=(
  "global-write-past-end@700@\[out-of-bounds\].*device memory write.* 0 bytes past the end of the 128-byte allocation@-@"
  "global-write-in-bounds@0@-@0@"
  "global-read-one-byte-past-end@700@\[out-of-bounds\].*device memory read.* 0 bytes past the end of the 100-byte allocation@-@"
  "global-read-last-byte@0@-@0@"
  "global-read-straddling-the-end@700@\[out-of-bounds\].*device memory read of 8 bytes.*runs past the end of the 12-byte allocation@-@"
  "global-use-after-free@700@\[use-after-free\].*device memory write.*freed allocation@-@"
  "global-null@700@\[invalid-pointer\].*device memory write at 0x0:@-@"
  "global-unmapped@700@\[invalid-pointer\].*device memory write at 0x10000000:@-@"
  "global-misaligned-load@716@\[misaligned-access\]@-@"
  "global-misaligned-store8@716@\[misaligned-access\]@-@"
  "global-misaligned-atomic@716@\[misaligned-access\]@-@"
  "shared-write-far-past-end@700@\[out-of-bounds\].*shared@-@"
  "shared-read-far-past-end@700@\[out-of-bounds\].*shared@-@"
  "shared-dynamic-far-past-end@700@\[out-of-bounds\].*shared@-@"
  "shared-write-one-past-end@700@\[out-of-bounds\].*shared@-@"
  "shared-dynamic-one-past-end@700@\[out-of-bounds\].*shared@-@"
  "shared-write-in-bounds@0@-@1@"
  "shared-misaligned@716@\[misaligned-access\]@-@"
  "race-write-then-read@719@\[data-race\].*(write-read|read-write) race on shared memory at byte offset 0:@-@R"
  "race-write-write@719@\[data-race\].*write-write race on shared memory at byte offset 0:@-@R"
  "race-reduction-missing-barrier@719@\[data-race\].*(write-read|read-write|write-write) race on shared memory@-@R"
  "ordered-by-barrier@0@-@5@r"
  "reduction-with-barriers@0@-@8256@r"
  "atomics-from-every-warp@0@-@128@r"
  "every-warp-reads-one-word@0@-@9@r"
)

status=0
# <arch for nvcc> <card> <the SASS engine runs the 1 KiB reserved region: yes|no>
run_arch() {
  local arch="$1" card="$2" reserved="$3" bin="$tmp.$1"
  nvcc -std=c++17 -cudart shared --gpu-architecture="$arch" -Wno-deprecated-gpu-targets \
       $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$bin"
  if ! require_shim_libs "$shim" "$bin"; then exit 0; fi
  local engine row name err re val flag want_err out code race_modes m extra
  for engine in sass ptx; do
    for row in "${cases[@]}"; do
      IFS='@' read -r name err re val flag <<<"$row"
      want_err="$err"
      # sm_80 and later, SASS: a one-word overrun of the shared array lands in the driver's reserved region.
      if [[ "$engine" == sass && "$reserved" == yes && "$name" == shared-*one-past-end ]]; then want_err=0; fi
      case "$flag" in R) race_modes="1" ;; r) race_modes="0 1" ;; *) race_modes="0" ;; esac
      for m in $race_modes; do
        extra=""
        [[ "$m" == 1 ]] && extra="VGPU_RACE=1"
        [[ "$engine" == ptx ]] && extra="$extra VGPU_SASS=0"
        out="$(env $(both_shims_env "$shim") $extra VGPU_GPU="$card" LD_LIBRARY_PATH="$shim" "$bin" "$name" 2>&1 || true)"
        code="$(sed -n 's/^ERR //p' <<<"$out" | head -1)"
        local ok=1 why=""
        [[ "$code" == "$want_err" ]] || { ok=0; why="error code ${code:-none}, wanted $want_err"; }
        if [[ "$re" != "-" && "$want_err" != 0 ]]; then
          grep -Eq -- "$re" <<<"$out" || { ok=0; why="$why; the diagnostic does not match /$re/"; }
        fi
        if [[ "$want_err" == 0 ]]; then
          if grep -q 'VirtualGPU error' <<<"$out"; then ok=0; why="$why; a diagnostic was printed for a kernel that is fine"; fi
          if [[ "$val" != "-" && "$(sed -n 's/^OUT //p' <<<"$out" | head -1)" != "$val" ]]; then
            ok=0; why="$why; output $(sed -n 's/^OUT //p' <<<"$out" | head -1), wanted $val"
          fi
        fi
        if (( ok )); then
          echo "ok   $arch $engine $name${extra:+ [$extra]}: $code"
        else
          echo "FAIL $arch $engine $name${extra:+ [$extra]}: $why"
          sed 's/^/     | /' <<<"$out" | head -12
          status=1
        fi
      done
    done
  done
}
run_arch sm_86 nvidia/rtx3060 yes
run_arch sm_75 nvidia/t4 no
[[ "$status" == 0 ]] && echo PASS
exit "$status"
