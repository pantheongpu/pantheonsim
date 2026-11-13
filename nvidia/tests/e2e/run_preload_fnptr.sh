#!/usr/bin/env bash
# `vgpu run --preload` and a library that keeps the driver's and NVRTC's functions in exported pointers named like them (ThrustRTC,
# issue #597): the program must run, not crash. preload_fnptr_lib.cpp says why it crashed.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
build="${VGPU_BUILD_DIR:-$root/build}"
vgpu="$build/vgpu"
shim="$build/shim"
[[ -x $vgpu ]] || { echo "SKIP: vgpu not built"; exit 0; }
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
nvrtcs=("$shim"/libnvrtc.so.[0-9]*)
shopt -u nullglob
[[ -e $shim/libcuda.so.1 && ${#carts[@]} -gt 0 && ${#nvrtcs[@]} -gt 0 ]] || { echo "SKIP: the driver, runtime or NVRTC shim is not built"; exit 0; }
. "$root/tests/shim_guard.sh"
if [[ -n "$(shim_sanitizer "$shim")" ]]; then
  # An instrumented shim preloaded ahead of the program's sanitizer runtime does not start (run_vgpu_run.sh says the same).
  echo "SKIP: sanitizer-instrumented shim (load order)"; exit 0
fi
cxx="${CXX:-c++}"
command -v "$cxx" >/dev/null 2>&1 || { echo "SKIP: no C++ compiler"; exit 0; }
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-preload-fnptr.XXXXXX")"
trap 'rm -rf "$work"' EXIT
"$cxx" -std=c++17 -O1 -fPIC -shared "$root/nvidia/tests/e2e/preload_fnptr_lib.cpp" -o "$work/libfnptr.so" -ldl || { echo "FAIL: the library does not compile"; exit 1; }
"$cxx" -std=c++17 -O1 "$root/nvidia/tests/e2e/preload_fnptr_main.cpp" -o "$work/main" -ldl || { echo "FAIL: the program does not compile"; exit 1; }
want="cuInit 0, cuDriverGetVersion 0 (a version)
nvrtcVersion 0 (a version)
ok"
fails=0
for mode in plain preload; do
  args=(); [[ $mode == preload ]] && args=(--preload)
  got="$(cd "$work" && timeout 120 "$vgpu" run --quiet --gpu nvidia/h100 "${args[@]}" ./main "$work/libfnptr.so" 2>&1)"; rc=$?
  if (( rc != 0 )) || [[ "$got" != "$want" ]]; then
    echo "FAIL: $mode: exit status $rc"; echo "$got" | sed 's/^/    /'; fails=1
  else
    echo "ok   $mode"
  fi
done
# And the private loads are what --preload asks for: the driver and NVRTC are not in LD_PRELOAD.
pre_env="$("$vgpu" run --gpu nvidia/h100 --preload --print-env /bin/true 2>&1)"
if grep -q '^LD_PRELOAD=.*/libcuda\.so\.1' <<< "$pre_env" || ! grep -q '^VGPU_PRELOAD_PRIVATE=.*/libcuda\.so\.1' <<< "$pre_env"; then
  echo "FAIL: --preload puts the driver in LD_PRELOAD, or not in VGPU_PRELOAD_PRIVATE"; grep -E '^(LD_PRELOAD|VGPU_PRELOAD_PRIVATE)=' <<< "$pre_env" | sed 's/^/    /'; fails=1
fi
(( fails == 0 )) || exit 1
echo "PASS"
