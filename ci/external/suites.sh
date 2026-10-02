#!/usr/bin/env bash
# Outside test suites, unmodified, on the simulator, checked against what an
# RTX 3060 printed for the same binaries (the lists next to this script):
#
#   cuda-samples  NVIDIA's CUDA Samples v13.0            cuda-samples.txt
#   hecbench      HeCBench's self-checking benchmarks    hecbench.txt
#   rodinia       Rodinia apps, output byte for byte     rodinia.txt
#   polybench     PolyBench/GPU, CPU-checked counts      polybench.txt
#
#   suites.sh fetch <suite> <dir>    the suite's source at its pinned commit
#   suites.sh build <suite> <dir>    the listed programs, for sm_86, -cudart shared
#   suites.sh run   <suite> <dir> <shim-dir> <report-dir>
#
# The builds use nvcc from PATH (CUDA 13.0, as the card's did) and do not
# depend on the simulator, so CI caches them. `run` exits non-zero when any
# program's result differs from the card's; report-dir gets one line per
# program (results.tsv), each one's output, and a summary.md.
#
# Environment: SUITE_GPU (nvidia/rtx3060), SUITE_TIMEOUT (seconds per
# program, 600), SUITE_JOBS (parallel builds, nproc), SUITE_ONLY (a regex of
# program names to keep).
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cmd="${1:?fetch|build|run}"; suite="${2:?suite}"; dir="${3:?dir}"
gpu="${SUITE_GPU:-nvidia/rtx3060}"
timeout_s="${SUITE_TIMEOUT:-600}"
jobs="${SUITE_JOBS:-$(nproc)}"
only="${SUITE_ONLY:-.}"

declare -A repo=([cuda-samples]=https://github.com/NVIDIA/cuda-samples
                 [hecbench]=https://github.com/zjin-lcf/HeCBench
                 [rodinia]=https://github.com/yuhc/gpu-rodinia
                 [polybench]=https://github.com/sgrauerg/polybenchGpu)
declare -A commit=([cuda-samples]=3f1c50965017932fc81e6d94a3fc9e04c105b312   # tag v13.0
                   [hecbench]=7d2d3c567be522a2104065165de0a4a233a6ea1a
                   [rodinia]=9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4
                   [polybench]=5584aaa7d0be810ff5eb0b61c49fb64ecc81ba4c)
[[ -n "${repo[$suite]:-}" ]] || { echo "unknown suite $suite"; exit 2; }

# The list's entries, comments and blank lines dropped, narrowed by SUITE_ONLY.
entries() { grep -vE '^(#|$)' "$here/$suite.txt" | while IFS= read -r l; do
  [[ "${l%%|*}" =~ $only ]] && printf '%s\n' "$l"; done; }

fetch() {
  mkdir -p "$dir" && cd "$dir" || exit 1
  git init -q . && git remote add origin "${repo[$suite]}" 2>/dev/null
  if [[ $suite == hecbench ]]; then
    # Some 500 benchmarks and their data: only the listed ones are checked out.
    git config core.sparseCheckout true
    git sparse-checkout init --cone >/dev/null 2>&1
    git sparse-checkout set $(entries | cut -d'|' -f1 | sed 's#^#src/#') >/dev/null
    git fetch -q --depth 1 --filter=blob:none origin "${commit[$suite]}"
  else
    git fetch -q --depth 1 origin "${commit[$suite]}"
  fi
  git -c advice.detachedHead=false checkout -q FETCH_HEAD
  echo "$suite at $(git rev-parse --short HEAD)"
}

build() {
  cd "$dir" || exit 1
  export compat="$here/compat.h"
  nvcc --version | tail -1
  case $suite in
    cuda-samples)
      # One architecture, the card's, rather than the nine the samples list.
      find Samples -name CMakeLists.txt -exec sed -i -E 's/set\(CMAKE_CUDA_ARCHITECTURES [^)]*\)/set(CMAKE_CUDA_ARCHITECTURES 86)/' {} +
      cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared > build-configure.log 2>&1 || { tail -30 build-configure.log; exit 1; }
      targets=$(entries | cut -d'|' -f1 | xargs -n1 basename)
      # -k 0: one sample that fails to build is reported by `run`, not fatal here.
      cmake --build build -j"$jobs" --target $targets -- -k 0 > build.log 2>&1 || tail -20 build.log
      ;;
    hecbench)
      entries | cut -d'|' -f1 | xargs -P "$jobs" -I{} bash -c '
        cd src/{} && make -s CC="nvcc -cudart shared" ARCH=sm_86 EXTRA_CFLAGS="-w" > build.log 2>&1 || echo "{}: build failed" ' ;;
    rodinia)
      entries | while IFS='|' read -r app src flags args file sum; do
        (cd "$src" && nvcc -arch=sm_86 -cudart shared -w -include "$compat" -I../util $flags -o "$app" > "$app.build.log" 2>&1) ||
          { echo "$app: build failed"; tail -5 "$src/$app.build.log"; }
      done ;;
    polybench)
      entries | while IFS='|' read -r name src card; do
        (cd "CUDA/$src" && nvcc -O3 -arch=sm_86 -cudart shared -w -DMINI_DATASET -include "$compat" ./*.cu -o "$name" > build.log 2>&1) ||
          { echo "$name: build failed"; tail -5 "CUDA/$src/build.log"; }
      done ;;
  esac
  return 0
}

# One program under the simulator: run_one <name> <workdir> <log> <cmd...>;
# sets rc and secs.
run_one() {
  local name=$1 wd=$2 log=$3; shift 3
  local t0=$SECONDS
  (cd "$wd" && env VGPU_QUIET=1 VGPU_GPU="$gpu" LD_LIBRARY_PATH="$shim" \
     timeout -k 10 "$timeout_s" "$@" > "$log" 2>&1 < /dev/null)
  rc=$?; secs=$(( SECONDS - t0 ))
}

run() {
  shim="$(cd "${4:?shim dir}" && pwd)"; report="$(mkdir -p "${5:?report dir}" && cd "$5" && pwd)"
  cd "$dir" || exit 1
  : > "$report/results.tsv"; mkdir -p "$report/logs"
  local fails=0 total=0
  result() {  # result <name> <ok|FAIL> <secs> <detail>
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$report/results.tsv"
    printf '%-4s %-36s %5ss  %s\n' "$2" "$1" "$3" "$4"
    total=$((total + 1)); [[ $2 == ok ]] || fails=$((fails + 1))
  }
  case $suite in
    cuda-samples)
      while IFS='|' read -r path marker; do
        name=$(basename "$path"); exe="build/$path"; log="$report/logs/$name.log"
        [[ -x $exe ]] || { result "$name" FAIL 0 "did not build"; continue; }
        run_one "$name" "$(dirname "$exe")" "$log" "./$name"
        if [[ $rc != 0 ]]; then result "$name" FAIL "$secs" "exit $rc (the card's: 0)"
        elif [[ -n $marker ]] && ! grep -qF -- "$marker" "$log"; then result "$name" FAIL "$secs" "no \"$marker\""
        else result "$name" ok "$secs" ""; fi
      done < <(entries) ;;
    hecbench)
      while IFS='|' read -r d args card; do
        log="$report/logs/$d.log"
        mk="src/$d/Makefile"; prog=$(awk -F'= *' '/^program/{print $2; exit}' "$mk" 2>/dev/null)
        exe=$(ls "src/$d" 2>/dev/null | grep -xE "main|${prog:-main}" | head -1)
        [[ -n $exe && -x src/$d/$exe ]] || { result "$d" FAIL 0 "did not build"; continue; }
        # shellcheck disable=SC2086
        run_one "$d" "src/$d" "$log" "./$exe" $args
        # Counted as the card's were: PASS(ED) and FAIL(ED), any case.
        got="rc=$rc $(grep -ioE '\b(PASS(ED)?|FAIL(ED)?)\b' "$log" | tr 'a-z' 'A-Z' | sed 's/ED$//' | sort | uniq -c | tr '\n' ' ' | sed -E 's/ +/ /g; s/^ //; s/ $//')"
        got="${got% }"
        if [[ "$got" == "$card" ]]; then result "$d" ok "$secs" "$got"
        else result "$d" FAIL "$secs" "$got (the card's: $card)"; fi
      done < <(entries) ;;
    rodinia)
      while IFS='|' read -r app src flags args file sum; do
        log="$report/logs/$app.log"; wd="$dir/$src/run-$app"; rm -rf "$wd"; mkdir -p "$wd"
        [[ -x $src/$app ]] || { result "$app" FAIL 0 "did not build"; continue; }
        # shellcheck disable=SC2086
        run_one "$app" "$wd" "$log" "../$app" $args
        if [[ $file == stdout ]]; then got=$(grep -v '^Time consumed' "$log" | sha256sum | cut -c1-64)
        else got=$(sha256sum < "$wd/$file" 2>/dev/null | cut -c1-64); fi
        if [[ $rc != 0 ]]; then result "$app" FAIL "$secs" "exit $rc"
        elif [[ "$got" == "$sum" ]]; then result "$app" ok "$secs" "$file as the card's"
        else result "$app" FAIL "$secs" "$file differs from the card's"; fi
      done < <(entries) ;;
    polybench)
      while IFS='|' read -r name src card; do
        log="$report/logs/$name.log"
        [[ -x CUDA/$src/$name ]] || { result "$name" FAIL 0 "did not build"; continue; }
        run_one "$name" "CUDA/$src" "$log" "./$name"
        got=$(grep -oE 'Percent: [0-9]+' "$log" | awk '{print $2}' | paste -sd,)
        if [[ $rc != 0 ]]; then result "$name" FAIL "$secs" "exit $rc"
        elif [[ "$got" == "$card" ]]; then result "$name" ok "$secs" "non-matching outputs: $got"
        else result "$name" FAIL "$secs" "non-matching outputs: ${got:-none printed} (the card's: $card)"; fi
      done < <(entries) ;;
  esac
  {
    echo "### $suite on $gpu: $((total - fails)) of $total as on the RTX 3060"
    echo
    if (( fails )); then
      echo "| program | seconds | result |"; echo "|---|---|---|"
      awk -F'\t' '$2 != "ok" {printf "| %s | %s | %s |\n", $1, $3, $4}' "$report/results.tsv"
      echo
    fi
    echo "Slowest: $(sort -t$'\t' -k3,3nr "$report/results.tsv" | head -5 | awk -F'\t' '{printf "%s %ss, ", $1, $3}' | sed 's/, $//')"
  } > "$report/summary.md"
  echo "$suite: $((total - fails)) of $total match the card"
  (( total > 0 && fails == 0 ))
}

case $cmd in
  fetch) fetch ;;
  build) build ;;
  run) run "$@" ;;
  *) echo "unknown command $cmd"; exit 2 ;;
esac
