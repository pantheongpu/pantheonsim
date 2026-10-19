#!/usr/bin/env bash
# The driver API used after thread_local destruction -- from a thread_local destroyed later than the driver's own
# state, and from a static destructor at exit() -- must still see the context the thread made current
# (exit_teardown.cu). It ran on freed memory before, and a process like StringZilla's ended in a glibc heap abort.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-exit-teardown.XXXXXX")"
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
trap 'rm -rf "$out"' EXIT
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
"$nvcc_bin" -std=c++17 -w -cudart shared "${san_flags[@]}" "$root/nvidia/tests/e2e/exit_teardown.cu" -lcuda -o "$out/t" ||
  { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$out/t"; then exit 0; fi
fails=0
for perturb in 0 165; do
  # MALLOC_PERTURB_ fills freed memory, so a stale read of it is garbage on every allocator.
  log="$(MALLOC_PERTURB_=$perturb VGPU_QUIET=1 VGPU_GPU=nvidia/a100 LD_LIBRARY_PATH="$shim" "$out/t" 2>&1)"
  rc=$?
  if [[ $rc != 0 ]] || ! grep -q '^PASS$' <<< "$log" || ! grep -q '^exit-time: context kept$' <<< "$log" ||
     grep -qi 'malloc\|free()\|corrupt\|double' <<< "$log"; then
    echo "$log" | sed "s|^|    perturb=$perturb: |"
    echo "FAIL: exit teardown (exit $rc, MALLOC_PERTURB_=$perturb)"; fails=1
  else
    echo "ok   MALLOC_PERTURB_=$perturb"
  fi
done
[[ $fails == 0 ]] && echo "PASS" || echo "FAIL: exit teardown"
exit $fails
