#!/usr/bin/env bash
# AMD's own PC sampling tool -- the pc_sampling sample that ships with ROCm's
# rocprofiler-sdk, unmodified -- on a simulated MI300X: it lists the agent's
# configurations, picks the host-trap method, samples a kernel's waves and
# prints what the sampling service delivered.
#
# The sample asks for ten thousand "micro seconds"; here the unit the agent
# offers is instructions (the simulator has no clock), so it is ten thousand
# instructions a wave. pcsampling.gfx942's kernel (spin, from pcsampling.cpp)
# issues, from llvm-objdump's listing of it:
#   5 instructions to get going         0x1900 .. 0x1914 (one is skipped)
#   4 in its loop, 20000 times          0x1918 s_add_i32, 0x191c s_cmp_eq_u32,
#                                       0x1920 v_mad_u64_u32, 0x1928 s_cbranch_scc0
#   10 to store and end                 0x192c .. 0x1964
# = 80015 a wave. A wave is sampled before its 1st, 10001st ... instruction,
# i.e. at index 0, 10000, ..., 80000: the first is 0x1900, and every later one
# is index 5 + 4 j + 3 of the loop, so the branch at 0x1928. Four waves (two
# work-groups of 128 work-items), nine samples each: 4 at 0x1900, 32 at 0x1928.
#
# Skips where ROCm's rocprofiler-sdk and its samples are not installed.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" 2>/dev/null && pwd)" || { echo "SKIP: no build directory"; exit 0; }
shim="$build/shim"
[[ -e "$shim/librocprofiler-sdk.so.1" && -e "$shim/libamdhip64.so.7" ]] || { echo "SKIP: build vgpuhip and vgpurocprof first"; exit 0; }
sdk=""
for c in "${VGPU_ROCPROFILER_SDK_PATH:-}" "${VGPU_ROCM_PATH:-}" "${ROCM_PATH:-}" /opt/rocm \
         $(ls -d "$HOME"/.local/share/rocm-*/opt/rocm-* 2>/dev/null | sort -rV) "$HOME"/.local/share/rocprofiler-sdk*/opt/rocm-*; do
  [[ -n "$c" && -e "$c/include/rocprofiler-sdk/pc_sampling.h" && -e "$c/share/rocprofiler-sdk/samples/pc_sampling/client.cpp" ]] &&
    { sdk=$c; break; }
done
[[ -n "$sdk" ]] || { echo "SKIP: no rocprofiler-sdk with its pc_sampling sample found (set VGPU_ROCPROFILER_SDK_PATH)"; exit 0; }
cxx=${CXX:-c++}
command -v "$cxx" >/dev/null || { echo "SKIP: no C++ compiler"; exit 0; }
# A tool preloaded into a program run against a sanitizer-built shim ends in a
# crash that says nothing about either; the sanitizer builds cover PC sampling
# through test_amd_rocprofiler_pcsampling.
if objdump -p "$shim/libamdhip64.so.7" 2>/dev/null | awk '/NEEDED/ {print $2}' | grep -qE 'lib(a|t)san'; then
  echo "SKIP: the shim is built with a sanitizer, and ROCm's sample tool is not"; exit 0
fi
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_pcsampling.XXXXXX")"
trap 'rm -rf "$work"' EXIT
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}

s="$sdk/share/rocprofiler-sdk/samples/pc_sampling"
if ! "$cxx" -std=c++17 -shared -fPIC -D__HIP_PLATFORM_AMD__ -I"$sdk/include" -I"$sdk/share/rocprofiler-sdk/samples" \
     "$s/client.cpp" "$s/pcs.cpp" "$s/utils.cpp" -o "$work/libpc-sampling-client.so" -L"$shim" -lrocprofiler-sdk -lpthread \
     2>"$work/cc.log"; then
  echo "SKIP: this ROCm's pc_sampling sample does not build here"; head -5 "$work/cc.log" | sed 's/^/      /'; exit 0
fi

exe="$root/amd/tests/hipcc/pcsampling.gfx942"
out=$(cd "$work" && ROCPROFILER_SAMPLE_OUTPUT_FILE=stdout ROCP_TOOL_LIBRARIES="$work/libpc-sampling-client.so" \
  LD_PRELOAD="$work/libpc-sampling-client.so" VGPU_QUIET=1 VGPU_GPU=amd/mi300x LD_LIBRARY_PATH="$shim" \
  timeout 300 "$exe" 2>&1)
status=$?
echo "$out" | grep -v '^(code_obj_id' | sed 's/^/      /' | head -20
expect "the program runs to the end with the tool loaded (an assertion in the tool's finalize would abort it)" "0" "$status"
expect "and its own check passes" "spin wrong 0 of 256" "$(grep -o 'spin wrong .*' <<< "$out")"
expect "the tool is told of one configuration: host trap (2), in instructions (1), from 1, as many as can be asked" \
  "(1.) method: 2, unit: 1, min_interval: 1, max_interval: 18446744073709551615, flags: 0" \
  "$(grep -m1 -F '(1.) method' <<< "$out")"
expect "it chooses host trap" "yes" "$(grep -q 'We chose Host-Trap PC sampling with the interval: 10000' <<< "$out" && echo yes || echo no)"
expect "nothing is dropped" "yes" "$(grep -q 'while the number of dropped samples is: 0' <<< "$out" && echo yes || echo no)"
samples=$(grep '^(code_obj_id' <<< "$out")
expect "four waves are sampled nine times each" "36" "$(grep -c . <<< "$samples")"
expect "the first instruction once a wave, the loop's branch for the other eight" "4 0x1900|32 0x1928" \
  "$(sed -n 's/^(code_obj_id, offset): ([0-9]*, \(0x[0-9a-f]*\)).*/\1/p' <<< "$samples" | sort | uniq -c | sed 's/^ *//' | paste -sd'|')"
expect "every sample is in the one code object, with every lane on" "36" \
  "$(grep -c 'code_obj_id, offset): (1, 0x[0-9a-f]*), .*exec: ffffffffffffffff' <<< "$samples")"
expect "two work-groups of two waves, nine samples each" "9 9 9 9" \
  "$(sed -n 's/.*workgroup_id_(x= *\([0-9]*\),.*wave_in_group: *\([0-9]*\),.*/\1.\2/p' <<< "$samples" | sort | uniq -c | awk '{printf "%s ", $1}' | sed 's/ $//')"
exit $fail
