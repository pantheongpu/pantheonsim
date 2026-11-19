#!/usr/bin/env bash
# Runs one libFuzzer target for a time.   run_fuzz.sh BUILD_DIR TARGET SECONDS [CORPUS_DIR]
#
# BUILD_DIR is a build made with -DVGPU_FUZZ=ON (clang, -fsanitize=fuzzer-no-link,address,undefined; .github/workflows/fuzz.yml).
# TARGET is ptx, sass or driver. The run starts from the seed corpus (BUILD_DIR/fuzz-seeds/TARGET, made by make_seeds.py) and from
# tests/fuzz/regress/TARGET, the input of every crash found so far, and keeps what it learns in CORPUS_DIR (default
# BUILD_DIR/fuzz-corpus/TARGET). A crash, a sanitizer report, a hang or an out-of-memory leaves its input in fuzz-artifacts/
# (crash-*, timeout-*, oom-*) and fails the run: copy it to tests/fuzz/regress/TARGET with a fix.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
build=${1:?build dir}
target=${2:?target: ptx, sass or driver}
secs=${3:?seconds}
corpus=${4:-$build/fuzz-corpus/$target}
exe=$build/fuzz_$target
[[ -x $exe ]] || { echo "no $exe: build it with -DVGPU_FUZZ=ON" >&2; exit 2; }
mkdir -p "$corpus" fuzz-artifacts
inputs=("$corpus" "$build/fuzz-seeds/$target")
[[ -d $here/regress/$target ]] && inputs+=("$here/regress/$target")
args=(-max_total_time="$secs" -timeout=30 -rss_limit_mb=3072 -max_len=65536 -print_final_stats=1
      -artifact_prefix=fuzz-artifacts/"$target"-)
[[ $target == driver ]] && args+=(-timeout=60)
[[ $target != sass ]] && args+=(-dict="$here/ptx.dict")
export ASAN_OPTIONS=${ASAN_OPTIONS:-detect_leaks=0:detect_odr_violation=0:abort_on_error=0:handle_abort=1}
export UBSAN_OPTIONS=${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1}
echo "== fuzz_$target for ${secs}s =="
"$exe" "${args[@]}" "${inputs[@]}"
