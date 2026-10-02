#!/usr/bin/env bash
# AMD's hip-tests (ROCm/hip-tests rocm-7.1.0), unmodified, built with ROCm
# 7.1's hipcc for one GPU's architecture and run case by case on the
# simulator, against a per-GPU baseline (hip-tests/baseline-<gpu>.tsv) of the
# cases known not to pass: hip-tests' own bugs, timing tests, environment
# cases, and sweeps too slow for a functional simulator (timeouts, which are
# skipped rather than run). Made by the GPU simulator AMD session, which ran
# them on the pantheonsim.com host (its recipe: ht-run.v2, ht-baseline.py).
#
#   hip-tests.sh fetch <dir>
#   hip-tests.sh build <dir> <arch> <suite>...        e.g. gfx942 AtomicsTest
#   hip-tests.sh run   <dir> <gpu> <shim-dir> <report-dir> <suite>...
#
# `run` fails when a case outside the baseline fails, crashes or times out; a
# baseline case that passes now is printed as good news. ROCM_PATH
# (/opt/rocm-7.1.0) is used only to build: at run time the simulator's shim
# stands in for libamdhip64 (LD_LIBRARY_PATH). Keep the shim's libraries as
# links to one file each (no cp -L): programs that dlopen libamdhip64.so would
# otherwise load a second runtime and crash.
#
# HT_TIMEOUT (180 s per case), HT_JOBS (2 parallel compiles: each wants GBs).
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cmd="${1:?fetch|build|run}"; dir="${2:?dir}"; shift 2
rocm="${ROCM_PATH:-/opt/rocm-7.1.0}"
commit=afe4069e2c338ea336dda26f7786440af6c1898a   # tag rocm-7.1.0

arch_of() {
  case "$1" in
    amd/mi300x|amd/mi325x) echo gfx942 ;; amd/mi250x) echo gfx90a ;; amd/mi350x) echo gfx950 ;;
    amd/rx6900xt) echo gfx1030 ;; amd/rx7900xtx) echo gfx1100 ;; amd/rx9070xt) echo gfx1201 ;;
    *) echo "unknown GPU $1" >&2; return 1 ;;
  esac
}

case $cmd in
fetch)
  mkdir -p "$dir" && cd "$dir" || exit 1
  git init -q . && git remote add origin https://github.com/ROCm/hip-tests 2>/dev/null
  git fetch -q --depth 1 origin "$commit" && git -c advice.detachedHead=false checkout -q FETCH_HEAD
  echo "hip-tests at $(git rev-parse --short HEAD)" ;;

build)
  arch="${1:?arch}"; shift
  b="$dir/build-$arch"
  export ROCM_PATH="$rocm" PATH="$rocm/bin:$PATH"
  cmake -G Ninja -S "$dir/catch" -B "$b" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$rocm/bin/hipcc" \
    -DHIP_PLATFORM=amd -DOFFLOAD_ARCH_STR=--offload-arch="$arch" -DROCM_PATH="$rocm" \
    -DCMAKE_PREFIX_PATH="$rocm" > "$b.configure.log" 2>&1 || { tail -30 "$b.configure.log"; exit 1; }
  # What the suites load at run time: code objects, child programs, headers.
  # Missing, they look like simulator failures ("no code object at X.code").
  helpers=$(ninja -C "$b" -t targets all | sed -nE 's/^([^:]*(\.code|_exe|_Exe|copyRtcHeaders|hipMatMul|copyKernel\.s)): .*/\1/p' | sort -u)
  # Each suite its own target, so one that fails to build costs only itself.
  for t in "$@" $helpers; do
    nice -n 15 ninja -C "$b" -k 0 -j"${HT_JOBS:-2}" "$t" > "$b.$(basename "$t").build.log" 2>&1 ||
      { echo "::warning::hip-tests $arch: $t did not build"; tail -5 "$b.$(basename "$t").build.log"; }
  done ;;

run)
  gpu="${1:?gpu}"; shim="$(cd "${2:?shim}" && pwd)"; report="$(mkdir -p "${3:?report}" && cd "$3" && pwd)"; shift 3
  arch=$(arch_of "$gpu") || exit 2
  b="$dir/build-$arch"
  base="$here/hip-tests/baseline-${gpu//\//-}.tsv"
  disabled="$here/hip-tests/disabled-$arch.txt"
  [[ -f $base ]] || { echo "no baseline $base"; exit 2; }
  mkdir -p "$report/logs"; : > "$report/results.tsv"; : > "$report/new.txt"
  new=0
  for suite in "$@"; do
    exe=$(find "$b/catch_tests" -name "$suite" -type f -executable | head -1)
    if [[ -z $exe ]]; then echo "NEW  $suite: not built" | tee -a "$report/new.txt"; new=$((new + 1)); continue; fi
    d=$(dirname "$exe"); t0=$SECONDS; counts=""
    while IFS= read -r t; do
      [[ -n $t ]] || continue
      if grep -qxF -- "$t" "$disabled"; then st=disabled
      elif awk -F'\t' -v t="$t" '$1 == t && $2 == "timeout" {f=1} END {exit !f}' "$base"; then st=skipped-slow
      else
        spec=$(sed -e 's/\\/\\\\/g; s/,/\\,/g; s/\[/\\[/g; s/\]/\\]/g' <<<"$t")
        log="$report/logs/$suite.$(md5sum <<<"$t" | cut -c1-10).log"
        (cd "$d" && LD_LIBRARY_PATH="$shim" VGPU_GPU="$gpu" VGPU_DEVICE_COUNT=2 VGPU_VRAM_MB=4096 \
           timeout -k 5 "${HT_TIMEOUT:-180}" ./"$suite" "$spec" > "$log" 2>&1 < /dev/null)
        rc=$?
        if [[ $rc == 0 ]]; then
          if grep -qE "No tests ran|test cases: +0" "$log"; then st=skip
          elif grep -qiE "^HIP_SKIP|skipped" "$log" && ! grep -q "assertions:.*passed" "$log"; then st=skip
          else st=pass; fi
        elif [[ $rc == 124 || $rc == 137 ]]; then st=timeout
        elif grep -qE "test cases:.*failed" "$log"; then st=fail
        else st="crash($rc)"; fi
        known=$(awk -F'\t' -v t="$t" '$1 == t {print $2}' "$base")
        case $st in
          pass|skip) [[ -n $known ]] && echo "now passes (baseline: $known): $suite $t" ;;
          *) if [[ -z $known ]]; then
               new=$((new + 1))
               echo "NEW  $st: $suite $t" | tee -a "$report/new.txt"
               grep -m1 -E "VirtualGPU error|with expansion:|FAILED|error:|Assertion|Segmentation|terminate" -A1 "$log" | head -2 | sed 's/^/       /'
             fi ;;
        esac
      fi
      printf '%s\t%s\t%s\n' "$suite" "$t" "$st" >> "$report/results.tsv"
    done < <(cd "$d" && ./"$suite" --list-test-names-only 2>/dev/null | tr -d '\r')
    counts=$(awk -F'\t' -v s="$suite" '$1 == s {print $3}' "$report/results.tsv" | sed 's/(.*//' | sort | uniq -c | tr -s ' \n' ' ')
    echo "$suite on $gpu, $((SECONDS - t0)) s:$counts"
  done
  {
    echo "### hip-tests on $gpu ($arch): $new new failures"
    echo
    [[ -s $report/new.txt ]] && { echo '```'; cat "$report/new.txt"; echo '```'; echo; }
    echo "Cases by result (skipped-slow: a baseline timeout, not run):"
    echo '```'
    awk -F'\t' '{k=$3; sub(/\(.*/, "", k); print $1, k}' "$report/results.tsv" | sort | uniq -c
    echo '```'
  } > "$report/summary.md"
  (( new == 0 )) ;;

*) echo "unknown command $cmd"; exit 2 ;;
esac
