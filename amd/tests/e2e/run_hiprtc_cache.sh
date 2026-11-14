#!/usr/bin/env bash
# The hiprtc compile cache (amd/src/hiprtc_cache.cpp) in front of ROCm's own libhiprtc: what a program gets back from a
# cold cache, from memory and from a file in a later process is, byte for byte, what ROCm's library returns with the cache
# off; a warm cache does no real compile; another simulated GPU is another compile; and the simulator's library defines every
# entry point ROCm's exports.
#
#   run_hiprtc_cache.sh <hiprtc_probe>
#
# Needs a ROCm with libhiprtc, the compiler library and its LLVM (hip-runtime-amd, comgr, rocm-llvm): found through
# VGPU_ROCM_PATH, ROCM_PATH or /opt/rocm; skips elsewhere. The probe (tests/hiprtc/hiprtc_probe.cpp) links the simulator's
# libhiprtc; the HIP runtime it compiles for is the simulator's, loaded by name as ROCm's libhiprtc loads it.
set -uo pipefail
probe="$1"
build="${VGPU_BUILD_DIR:-build}"
shim="$build/shim"
[[ -e "$shim/libhiprtc.so.6" && -e "$shim/libamdhip64.so" ]] || { echo "SKIP: no hiprtc or HIP shim in $shim"; exit 0; }
rocm=""
for c in "${VGPU_ROCM_PATH:-}" "${ROCM_PATH:-}" /opt/rocm $(ls -d "$HOME"/.local/share/rocm-*/opt/rocm-* 2>/dev/null | sort -rV); do
  [[ -n "$c" ]] && ls "$c"/lib/libhiprtc.so.[0-9] >/dev/null 2>&1 && ls "$c"/lib/libamd_comgr.so.[0-9] >/dev/null 2>&1 && { rocm=$c; break; }
done
[[ -n "$rocm" ]] || { echo "SKIP: no ROCm with libhiprtc and libamd_comgr found (set VGPU_ROCM_PATH)"; exit 0; }
real=$(ls "$rocm"/lib/libhiprtc.so.[0-9] | sort -V | tail -1)
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -m 700 "$tmp/cache"

# The programs: a plain kernel; VkFFT's shape (the runtime header included, a name of its own, -ffp-contract=off); a template
# with a header that is given and two name expressions; one that warns.
cat > "$tmp/plain.hip" <<'SRC'
extern "C" __global__ void scale(float* a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = a[i] * 2.0f + 1.0f;
}
SRC
cat > "$tmp/vkfft.hip" <<'SRC'
#include <hip/hip_runtime.h>
extern "C" __global__ void VkFFT_main(double* in, double* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * in[i] - 0.5 * in[(i + 1) % n];
}
SRC
cat > "$tmp/template.hip" <<'SRC'
#include "factor.h"
template <typename T> __global__ void axpy(T* x, T* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += FACTOR * x[i];
}
SRC
cat > "$tmp/warn.hip" <<'SRC'
extern "C" __global__ void warns(float* a) { int unused; a[threadIdx.x] = 1.0f; }
SRC
programs=(--src-file "$tmp/plain.hip" --name plain.hip
  --next --src-file "$tmp/vkfft.hip" --name VkFFT.hip --opt -ffp-contract=off
  --next --src-file "$tmp/template.hip" --name template.hip --header "factor.h=#define FACTOR 3" --expr "axpy<float>" --expr "axpy<double>" --opt -O2
  --next --src-file "$tmp/warn.hip" --name warn.hip --opt -Wall)
count=4

# The simulator's libraries first, then ROCm's (where the real libhiprtc and the compiler library are found).
run() {  # run <gpu> <cache mode> <probe arguments...>: standard output, the statistics line in $tmp/stats
  local gpu=$1 mode=$2
  shift 2
  env VGPU_GPU="$gpu" VGPU_QUIET=1 VGPU_HIPRTC_CACHE="$mode" VGPU_HIPRTC_CACHE_DIR="$tmp/cache" VGPU_HIPRTC_CACHE_STATS=1 \
    LD_LIBRARY_PATH="$shim:$rocm/lib:${LD_LIBRARY_PATH:-}" "$probe" child "$@" 2> "$tmp/stderr"
  grep "hiprtc-cache:" "$tmp/stderr" > "$tmp/stats" || true
}
# ROCm's library with nothing in front of it: the probe asks for libhiprtc.so.6 (or .so.7) by name, and a directory that holds
# only ROCm's under that name comes first. (The HIP runtime is still the simulator's, and the compiler library ROCm's.)
mkdir "$tmp/alone"
ln -s "$real" "$tmp/alone/libhiprtc.so.6"
ln -s "$real" "$tmp/alone/libhiprtc.so.7"
alone() {  # alone <gpu> <probe arguments...>
  local gpu=$1
  shift
  env VGPU_GPU="$gpu" VGPU_QUIET=1 LD_LIBRARY_PATH="$tmp/alone:$shim:$rocm/lib:${LD_LIBRARY_PATH:-}" "$probe" child "$@" 2> /dev/null
}
stat() { sed -n "s/.*hiprtc-cache: \([0-9]*\) compiles: \([0-9]*\) from memory, \([0-9]*\) from [^,]*, \([0-9]*\) by the real library.* \([0-9]*\) stored.*/\\$1/p" "$tmp/stats"; }

# ROCm's own output is not the same twice for a program with two name expressions (the order it instantiates them in varies: the
# template's code object has one size and, from run to run, one of several contents). So against a run with the cache off, that
# program's code is compared by size; what the cache hands back is compared exactly with what the real library returned to the
# same cache (cold against warm, and the first compile of a process against the second).
mask() { sed -E '/addname=.*addname=/ s/(code:[0-9]+:[0-9]+:)[0-9a-f]+/\1(varies)/'; }

rocm_alone=$(alone amd/mi300x "${programs[@]}")
want=$(run amd/mi300x 0 "${programs[@]}")
expect "with the cache off a program sees what it sees with ROCm's library alone: every result code, size and byte" "$(mask <<< "$rocm_alone")" "$(mask <<< "$want")"
expect "ROCm's libhiprtc compiles the four programs (cache off)" "$count" "$(grep -c 'compile=0(HIPRTC_SUCCESS)' <<< "$want")"
expect "... and lowers the template's two names" "2" "$(grep -o 'lowered:_Z[A-Za-z0-9_]*' <<< "$want" | sort -u | wc -l)"
expect "... and leaves a warning in the fourth program's log" "yes" "$(tail -1 <<< "$want" | grep -q 'log:[1-9]' && echo yes || echo no)"
expect "the cache off is ROCm's library alone: $count real compiles, nothing kept" "$count 0" "$(stat 4) $(ls "$tmp/cache" | wc -l)"

cold=$(run amd/mi300x 1 "${programs[@]}")
expect "a cold cache hands back what ROCm's library does: results, code, log, lowered names" "$(mask <<< "$want")" "$(mask <<< "$cold")"
expect "... having compiled each program once and kept it" "$count $count $count" "$(stat 4) $(stat 5) $(ls "$tmp/cache" | grep -c '\.rtc$')"
warm=$(run amd/mi300x 1 "${programs[@]}")
expect "a later process gets the bytes ROCm's library returned to the first, every one" "$cold" "$warm"
expect "... from the files, with no real compile" "$count 0" "$(stat 3) $(stat 4)"
twice=$(run amd/mi300x mem "${programs[@]}" --repeat 2)
expect "in one process the second compile of each program gets the bytes of the first, which ROCm's library made" "$(head -$count <<< "$twice")" "$(tail -$count <<< "$twice")"
expect "... and they are what the library returns with the cache off" "$(mask <<< "$want")" "$(head -$count <<< "$twice" | mask)"
expect "... from memory" "$count $count" "$(stat 2) $(stat 4)"

# Another simulated GPU is another architecture: its own compiles, its own bytes.
other_want=$(run amd/rx7900xtx 0 "${programs[@]}")
other=$(run amd/rx7900xtx 1 "${programs[@]}")
expect "another simulated GPU is compiled for again, with ROCm's results for it" "$(mask <<< "$other_want")|$count" "$(mask <<< "$other")|$(stat 4)"
expect "... which are not the MI300X's" "differ" "$([[ "$other_want" != "$want" ]] && echo differ || echo same)"

# The orders of calls the cache has no answer for (the bitcode, a second compile, a name expression after the compile, a name that
# was not added, a size asked for before any compile): ROCm's library alone, then behind the cache when it is off, cold and warm.
for how in bitcode twice late-expr unknown-name early; do
  args=(--src-file "$tmp/plain.hip" --name "$how.hip" --expr scale --how "$how")
  a=$(alone amd/mi300x "${args[@]}")
  expect "\"$how\": ROCm's answers with the cache off, cold and warm" "$a|$a|$a" "$(run amd/mi300x 0 "${args[@]}")|$(run amd/mi300x 1 "${args[@]}")|$(run amd/mi300x 1 "${args[@]}")"
done

# Every entry point ROCm's library exports is one the simulator's defines.
if command -v nm >/dev/null; then
  theirs=$(nm -D --defined-only "$real" | awk '$2 ~ /^[TW]$/ && $3 ~ /^hiprtc/ {print $3}' | sort -u)
  ours=$(nm -D --defined-only "$shim/libhiprtc.so.6" | awk '$2 ~ /^[TW]$/ && $3 ~ /^hiprtc/ {print $3}' | sort -u)
  expect "the simulator's libhiprtc defines every hiprtc entry point of $(basename "$real")" "" "$(comm -23 <(echo "$theirs") <(echo "$ours") | tr '\n' ' ')"
fi
exit $fail
