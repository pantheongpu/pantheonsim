#!/usr/bin/env bash
# `vgpu run` needs only the vendor's own shim (#647): an AMD run starts from a
# build that has the HIP shim and no CUDA driver shim, and says what is missing
# only when it is something the AMD run needs; an NVIDIA run still stops on a
# missing libcuda.so.1 with the message it always gave.
#
# The launcher finds its shim beside its own binary, so the test copies the vgpu
# binary into build directories of its own whose shim holds only what each case
# wants, and leaves the real build alone.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
build="$(cd "$build" && pwd)"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
[[ -x "$build/vgpu" ]] || { echo "SKIP: $build/vgpu not built"; exit 0; }
[[ -e "$build/shim/libamdhip64.so.6" ]] || { echo "SKIP: no HIP shim in $build/shim"; exit 0; }
command -v gcc >/dev/null || { echo "SKIP: no C compiler"; exit 0; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0
expect() {  # expect <name> <expected-substring> <actual>
  if [[ "$3" == *"$2"* ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected to see: $2"; echo "      got:"; sed 's/^/        /' <<<"$3"; fail=1; fi
}

hip="$(readlink -f "$build/shim/libamdhip64.so.6")"
# hip: the HIP shim alone, as `cmake --build <dir> --target vgpuhip` leaves it. none: an empty shim.
for d in hip none; do
  mkdir -p "$tmp/$d/shim"
  cp "$build/vgpu" "$tmp/$d/vgpu"
done
for n in libamdhip64.so.6 libamdhip64.so.7 libamdhip64.so libhsa-runtime64.so.1 libhsa-runtime64.so; do
  ln -s "$hip" "$tmp/hip/shim/$n"
done

# A HIP program, linked against a stand-in with the real library's name and the
# two functions it calls, so its DT_NEEDED is what hipcc's would be; at run time
# the simulator's library answers.
echo 'int hipRuntimeGetVersion(int* v) { *v = -1; return 1; } int hipDriverGetVersion(int* v) { *v = -1; return 1; }' > "$tmp/stand_in.c"
mkdir -p "$tmp/lib6"
gcc -shared -fPIC -Wl,-soname,libamdhip64.so.6 -o "$tmp/lib6/libamdhip64.so.6" "$tmp/stand_in.c"
gcc -o "$tmp/six" "$root/amd/tests/e2e/hip_version.c" -Wl,--no-as-needed "$tmp/lib6/libamdhip64.so.6" \
  || { echo "FAIL  could not link the HIP program"; exit 1; }
# A sanitizer-instrumented shim needs its runtime loaded first.
preload=""
if command -v objdump >/dev/null; then
  preload=$(objdump -p "$hip" 2>/dev/null | awk '/NEEDED/ && /lib(asan|tsan)\.so/ {print $2}')
fi

out=$(LD_PRELOAD="$preload" "$tmp/hip/vgpu" run --quiet --gpu amd/mi300x "$tmp/six" 2>&1)
expect "an AMD run starts with the HIP shim and no CUDA shim" "runtime 60443483" "$out"
expect "... with the build's shim on LD_LIBRARY_PATH" "LD_LIBRARY_PATH=$tmp/hip/shim" \
  "$("$tmp/hip/vgpu" run --gpu amd/mi300x --print-env /bin/true 2>&1)"

out=$("$tmp/none/vgpu" run --gpu amd/mi300x /bin/true 2>&1); rc=$?
expect "an AMD run without the HIP shim names it" "no HIP shim at $tmp/none/shim/libamdhip64.so.6" "$out"
expect "... and the target that builds it" "--target vgpuhip" "$out"
[[ $rc -eq 2 ]] && echo "ok    ... and exits 2" || { echo "FAIL  ... exit code $rc, expected 2"; fail=1; }

# A program that needs an AMD library this build does not have (ROCm SMI is its own target).
echo 'int main(void){return 0;}' > "$tmp/prog.c"
gcc -shared -fPIC -Wl,-soname,librocm_smi64.so.7 -o "$tmp/lib6/librocm_smi64.so.7" "$tmp/stand_in.c"
gcc -o "$tmp/smi" "$tmp/prog.c" -Wl,--no-as-needed "$tmp/lib6/librocm_smi64.so.7"
out=$("$tmp/hip/vgpu" run --gpu amd/mi300x "$tmp/smi" 2>&1); rc=$?
expect "an AMD library the build lacks is named, with its target" "needs librocm_smi64.so.7" "$out"
expect "... and the target that builds it" "--target vgpursmi" "$out"
[[ $rc -eq 1 ]] && echo "ok    ... and exits 1" || { echo "FAIL  ... exit code $rc, expected 1"; fail=1; }

# NVIDIA runs are unchanged: the CUDA driver shim is required, with the same message.
out=$("$tmp/hip/vgpu" run --gpu nvidia/a10 /bin/true 2>&1); rc=$?
expect "an NVIDIA run without libcuda.so.1 still stops" "no simulator libraries at $tmp/hip/shim" "$out"
[[ $rc -eq 2 ]] && echo "ok    ... and exits 2" || { echo "FAIL  ... exit code $rc, expected 2"; fail=1; }
out=$(VGPU_GPU= "$tmp/hip/vgpu" run /bin/true 2>&1)
expect "... so does the default GPU (nvidia/h100)" "no simulator libraries at" "$out"
exit $fail
