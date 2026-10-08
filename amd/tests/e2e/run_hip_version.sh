#!/usr/bin/env bash
# hipRuntimeGetVersion answers for the ROCm release a program was built for: 6.4.43483 for one that
# asks for libamdhip64.so.6 (ROCm 6.4's name) and no .so.7, ROCm 7.2's 7.2.53211 for the rest.
#
# RCCL 2.27 (ROCm 7.1 and 7.2, and the PyTorch wheels that carry it) refuses to start on a runtime
# older than 6.4.43484 unless HSA_NO_SCRATCH_RECLAIM=1 is set: a runtime that says 6.4.43483 to a
# program built by ROCm 7 made every RCCL program fail to initialize.
#
# The programs are linked against stand-ins that carry the two names, so their DT_NEEDED is what
# hipcc's would be; at run time the simulator's library answers to both names.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
shim="$build/shim"
[[ -e "$shim/libamdhip64.so.7" ]] || { echo "SKIP: no HIP shim in $shim"; exit 0; }
command -v gcc >/dev/null || { echo "SKIP: no C compiler"; exit 0; }
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
echo 'void hip_stand_in(void) {}' > "$tmp/stand_in.c"
for n in 6 7; do
  mkdir -p "$tmp/lib$n"
  gcc -shared -fPIC -Wl,-soname,libamdhip64.so.$n -o "$tmp/lib$n/libamdhip64.so.$n" "$tmp/stand_in.c"
done
# A sanitizer-instrumented shim needs its runtime loaded first.
preload=""
if command -v objdump >/dev/null; then
  preload=$(objdump -p "$shim/libamdhip64.so.7" 2>/dev/null | awk '/NEEDED/ && /lib(asan|tsan)\.so/ {print $2}')
fi
run() { LD_PRELOAD="$preload" VGPU_QUIET=1 VGPU_GPU=amd/mi300x LD_LIBRARY_PATH="$shim" "$1"; }

gcc -o "$tmp/six" "$root/amd/tests/e2e/hip_version.c" -Wl,--no-as-needed "$tmp/lib6/libamdhip64.so.6" -Wl,-rpath,"$shim" \
  && expect "a program that needs libamdhip64.so.6 is on ROCm 6.4.43483" "runtime 60443483 driver 60443483" "$(run "$tmp/six")"
gcc -o "$tmp/seven" "$root/amd/tests/e2e/hip_version.c" -Wl,--no-as-needed "$tmp/lib7/libamdhip64.so.7" -Wl,-rpath,"$shim" \
  && expect "one that needs .so.7 is on ROCm 7.2's 7.2.53211" "runtime 70253211 driver 70253211" "$(run "$tmp/seven")"
gcc -o "$tmp/both" "$root/amd/tests/e2e/hip_version.c" -Wl,--no-as-needed "$tmp/lib6/libamdhip64.so.6" "$tmp/lib7/libamdhip64.so.7" \
  && expect "one that needs both (a ROCm 7 program with a ROCm 6 library) is on 7.2" "runtime 70253211 driver 70253211" "$(run "$tmp/both")"
# And one that loads the library by name at run time, as Python's ctypes does, needing neither.
cat > "$tmp/loads.c" <<'C'
#include <dlfcn.h>
#include <stdio.h>
int main(void) {
  void* h = dlopen("libamdhip64.so", RTLD_NOW);
  if (!h) return 2;
  int (*get)(int*) = (int (*)(int*))dlsym(h, "hipRuntimeGetVersion");
  int v = 0;
  get(&v);
  printf("runtime %d\n", v);
  return 0;
}
C
gcc -o "$tmp/loads" "$tmp/loads.c" -ldl && expect "one that loads the library by name is on 7.2" "runtime 70253211" "$(run "$tmp/loads")"
exit $fail
