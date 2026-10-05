#!/usr/bin/env bash
# VGPU_QUIET=1 silences diagnostics; any other value, or none, does not. That is
# what nvidia/include/vgpu_cuda.h documents and what the hand-written shims do, but the
# generated library stubs tested for the variable's *presence* -- so
# VGPU_QUIET=0 silenced "X is not implemented by VirtualGPU" in cuFFT, cuBLAS,
# cuSPARSE, cuSOLVER, cuDNN and NCCL, and nowhere else. That message is the only
# thing telling a user their call was refused rather than answered wrongly.
set -uo pipefail

# The reporter is the code in each generated stub file (nvidia/src/generated/
# *_stubs.cpp). Whether a library still has a stub to call changes as its API
# is written (NCCL's last went with ncclCommShrink, cuSOLVER's and cuBLAS's
# files hold the reporter alone), so this compiles each file with one more
# entry point that reports itself, and calls that from Python. The probe is
# plain code, so a sanitizer build of the shims does not matter here.
root="$(cd "$(dirname "$0")/../../.." && pwd)"
command -v python3 >/dev/null || { echo "SKIP: no python3"; exit 0; }
command -v g++ >/dev/null || { echo "SKIP: no g++"; exit 0; }
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
fail=0
tested=0
for stubs in "$root"/nvidia/src/generated/*_stubs.cpp; do
  [[ -e "$stubs" ]] || continue
  grep -q vgpu_report_unimplemented "$stubs" || continue
  lib="$tmp/$(basename "${stubs%.cpp}").so"
  printf '#include "%s"\nVGPU_EXPORT int vgpu_probe() { vgpu_report_unimplemented("vgpu_probe"); return 0; }\n' \
    "$stubs" > "$tmp/probe.cpp"
  g++ -std=c++20 -shared -fPIC -o "$lib" "$tmp/probe.cpp" || { echo "FAIL  could not build a probe from $stubs"; fail=1; continue; }
  tested=1
  name="$(basename "$stubs")"
  for case in "unset:yes" "0:yes" "1:no" "true:yes"; do
    q="${case%%:*}"; want="${case##*:}"
    if [[ "$q" == unset ]]; then run=(env -u VGPU_QUIET); else run=(env "VGPU_QUIET=$q"); fi
    out=$("${run[@]}" python3 -c "import ctypes; ctypes.CDLL('$lib').vgpu_probe()" 2>&1)
    if grep -q "not implemented by VirtualGPU" <<< "$out"; then got=yes; else got=no; fi
    if [[ "$got" == "$want" ]]; then echo "ok    $name VGPU_QUIET=$q -> $([[ $got == yes ]] && echo "diagnostic printed" || echo silent)"
    else echo "FAIL  $name VGPU_QUIET=$q -> printed=$got, expected $want"; fail=1; fi
  done
done
[[ $tested == 1 ]] || { echo "FAIL  no generated stub file carries the reporter"; exit 1; }
exit $fail
