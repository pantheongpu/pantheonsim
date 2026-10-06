#!/usr/bin/env bash
# Outside test suites, unmodified, on the simulator, checked against what an
# RTX 3060 printed for the same binaries (the lists next to this script):
#
#   cuda-samples  NVIDIA's CUDA Samples v13.0            cuda-samples.txt
#   hecbench      HeCBench's self-checking benchmarks    hecbench.txt
#   rodinia       Rodinia apps, output byte for byte     rodinia.txt
#   polybench     PolyBench/GPU, CPU-checked counts      polybench.txt
#
# Two open-source GPU memory and burn-in testers are run the same way, but there
# is no card's output to compare with: each program must pass by its own
# account (exit status, its own marker line, no error reported), and
# cuda-memtest is also run against a stuck memory cell armed with `vgpu fault`,
# which it must find:
#
#   cuda-memtest  cuda_memtest (ComputationalRadiationPhysics)   cuda-memtest.txt
#   gpu-burn      gpu-burn (wilicc)                              gpu-burn.txt
#
#   suites.sh fetch <suite> <dir>    the suite's source at its pinned commit
#   suites.sh build <suite> <dir>    the listed programs, for sm_86, -cudart shared
#   suites.sh run   <suite> <dir> <shim-dir> <report-dir>
#   suites.sh digest <suite> <dir> <shim-dir> <report-dir>
#
# `digest` runs each program on the SASS (the default) and on its PTX
# (VGPU_SASS=0), one host thread each, with VGPU_KERNEL_DIGEST, and fails when
# the device memory the two leave after any kernel holds different bytes, as
# nvidia/tools/sass_ptx_digest.sh does for the e2e programs.
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
cmd="${1:?fetch|build|run|digest}"; suite="${2:?suite}"; dir="${3:?dir}"
gpu="${SUITE_GPU:-nvidia/rtx3060}"
timeout_s="${SUITE_TIMEOUT:-600}"
jobs="${SUITE_JOBS:-$(nproc)}"
only="${SUITE_ONLY:-.}"

declare -A repo=([cuda-samples]=https://github.com/NVIDIA/cuda-samples
                 [hecbench]=https://github.com/zjin-lcf/HeCBench
                 [rodinia]=https://github.com/yuhc/gpu-rodinia
                 [polybench]=https://github.com/sgrauerg/polybenchGpu
                 [cuda-memtest]=https://github.com/ComputationalRadiationPhysics/cuda_memtest
                 [gpu-burn]=https://github.com/wilicc/gpu-burn)
declare -A commit=([cuda-samples]=3f1c50965017932fc81e6d94a3fc9e04c105b312   # tag v13.0
                   [hecbench]=7d2d3c567be522a2104065165de0a4a233a6ea1a
                   [rodinia]=9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4
                   [polybench]=5584aaa7d0be810ff5eb0b61c49fb64ecc81ba4c
                   [cuda-memtest]=e94e1ee54e0689c9f154a8a08202554452552ec6   # dev, version 1.2.3
                   [gpu-burn]=0d19c94dcb2e9b6858b7586a4c327674cd67113b)      # master
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
  if [[ $suite == hecbench ]]; then
    # Some benchmarks build or run from a sibling's files (boxfilter-cuda
    # takes its headers and image from ../boxfilter-sycl): add every
    # directory a listed one's Makefile or arguments name.
    sibs=$( { entries | cut -d'|' -f2; entries | cut -d'|' -f1 | sed 's#^#src/#; s#$#/Makefile#' | xargs cat 2>/dev/null; } |
            grep -oE '\.\./[A-Za-z0-9_.+-]+' | sed 's#^\.\./#src/#' | sort -u)
    [[ -n $sibs ]] && git sparse-checkout add $sibs >/dev/null
  fi
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
      # Only the listed samples' directories (each is a CMake project of its
      # own): the others need libraries a toolchain may not have (the CI
      # image drops the static ones that simpleCUFFT_callback links), and
      # configuring them all is slow.
      if [[ ! -e CMakeLists.txt.all ]]; then
        mv CMakeLists.txt CMakeLists.txt.all
        { sed '/add_subdirectory(Samples)/d' CMakeLists.txt.all
          entries | cut -d'|' -f1 | xargs -n1 dirname | sort -u | sed 's/.*/add_subdirectory(&)/'; } > CMakeLists.txt
      fi
      # The nvcc on PATH, named: CMake otherwise finds /usr/bin/nvcc first where
      # Ubuntu's CUDA 12.0 is installed too (the pantheonsim.com runner).
      cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
        -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared > build-configure.log 2>&1 || { tail -30 build-configure.log; exit 1; }
      targets=$(entries | cut -d'|' -f1 | xargs -n1 basename)
      # -k 0: one sample that fails to build is reported by `run`, not fatal here.
      cmake --build build -j"$jobs" --target $targets -- -k 0 > build.log 2>&1 || tail -20 build.log
      ;;
    hecbench)
      entries | cut -d'|' -f1 | xargs -P "$jobs" -I{} bash -c '
        cd src/{} && make -s CC="nvcc -cudart shared" ARCH=sm_86 EXTRA_CFLAGS="-w" > build.log 2>&1 ||
          { echo "{}: build failed"; tail -8 build.log | sed "s/^/    /"; }' ;;
    rodinia)
      entries | while IFS='|' read -r app src flags args file sum; do
        (cd "$src" && nvcc -arch=sm_86 -cudart shared -w -include "$compat" -I../util $flags -o "$app" > "$app.build.log" 2>&1) ||
          { echo "$app: build failed"; tail -5 "$src/$app.build.log"; }
      done ;;
    cuda-memtest)
      # Built as its README says (CMake, CUDA language), for the card's sm_86
      # and with the shared cudart, so the simulator's is the one it loads.
      cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
        > build-configure.log 2>&1 || { tail -30 build-configure.log; exit 1; }
      cmake --build build -j"$jobs" > build.log 2>&1 || { tail -20 build.log; exit 1; } ;;
    gpu-burn)
      # Its Makefile, with the toolkit of the nvcc on PATH and the card's sm_86;
      # new dtags, so LD_LIBRARY_PATH (the shim) is searched before the rpath
      # the Makefile adds for that toolkit's libraries.
      root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
      make -j"$jobs" CUDAPATH="$root" COMPUTE=86 LDFLAGS="-Wl,--enable-new-dtags" > build.log 2>&1 ||
        { echo "gpu_burn: build failed"; tail -10 build.log; exit 1; } ;;
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
  if [[ $mode == digest ]]; then
    # SASS, then PTX; the PTX run's output beside the SASS run's.
    rm -f "$report/digest/$name".*
    (cd "$wd" && env VGPU_QUIET=1 VGPU_GPU="$gpu" LD_LIBRARY_PATH="$shim" VGPU_THREADS=1 VGPU_SASS_LOG=1 \
       VGPU_KERNEL_DIGEST="$report/digest/$name.sass" timeout -k 10 "$timeout_s" "$@" > "$log" 2>&1 < /dev/null)
    rc=$?
    (cd "$wd" && env VGPU_QUIET=1 VGPU_GPU="$gpu" LD_LIBRARY_PATH="$shim" VGPU_THREADS=1 VGPU_SASS=0 \
       VGPU_KERNEL_DIGEST="$report/digest/$name.ptx" timeout -k 10 "$timeout_s" "$@" > "$log.ptx" 2>&1 < /dev/null)
    rc_ptx=$?
  else
    (cd "$wd" && env VGPU_QUIET=1 VGPU_GPU="$gpu" LD_LIBRARY_PATH="$shim" \
       timeout -k 10 "$timeout_s" "$@" > "$log" 2>&1 < /dev/null)
    rc=$?
  fi
  secs=$(( SECONDS - t0 ))
}

# digest mode: the verdict for the program run_one just ran twice.
digest_result() {
  local name=$1 d="$report/digest/$1" strip='s/ [0-9a-f]+:/ /g'
  if [[ $rc != 0 || $rc_ptx != 0 ]]; then result "$name" FAIL "$secs" "exit $rc on SASS, $rc_ptx on PTX"; return; fi
  if ! grep -q "running SASS" "$report/logs/$name.log" || grep -q "running PTX instead of SASS" "$report/logs/$name.log"; then
    result "$name" ok "$secs" "ran PTX both times: $(grep -m1 -o 'running PTX instead of SASS.*' "$report/logs/$name.log" || echo 'no SASS ran')"
    return
  fi
  if [[ ! -s $d.ptx ]]; then result "$name" FAIL "$secs" "no digest written"; return; fi
  # A known difference (digest-known.txt) is reported, not failed.
  local known
  known=$(awk -F'|' -v s="$suite" -v n="$name" '$1 == s && $2 == n {print $3}' "$here/digest-known.txt" 2>/dev/null)
  if cmp -s <(sed -E "$strip" "$d.sass") <(sed -E "$strip" "$d.ptx"); then
    result "$name" ok "$secs" "$(wc -l < "$d.ptx") launches, the same memory on SASS and PTX${known:+ (listed as known to differ: it no longer does, take it off digest-known.txt)}"
  else
    local first
    first=$(diff <(sed -E "$strip" "$d.ptx") <(sed -E "$strip" "$d.sass") | grep -m1 '^<' | cut -c3-)
    if [[ -n $known ]]; then
      result "$name" ok "$secs" "known difference: SASS and PTX differ after launch ${first%% *} ($(cut -d' ' -f2 <<< "$first")); listed: $known"
    else
      result "$name" FAIL "$secs" "SASS and PTX differ after launch ${first%% *} ($(cut -d' ' -f2 <<< "$first"))"
    fi
  fi
}

run() {
  shim="$(cd "${4:?shim dir}" && pwd)"; report="$(mkdir -p "${5:?report dir}" && cd "$5" && pwd)"
  cd "$dir" || exit 1
  : > "$report/results.tsv"; mkdir -p "$report/logs" "$report/digest"
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
        [[ $mode == digest ]] && { digest_result "$name"; continue; }
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
        [[ $mode == digest ]] && { digest_result "$d"; continue; }
        # Counted as the card's were: PASS(ED) and FAIL(ED), any case.
        got="rc=$rc $(grep -ioE '\b(PASS(ED)?|FAIL(ED)?)\b' "$log" | tr 'a-z' 'A-Z' | sed 's/ED$//' | sort | uniq -c | tr '\n' ' ' | sed -E 's/ +/ /g; s/^ //; s/ $//')"
        got="${got% }"
        if [[ "$got" == "$card" ]]; then result "$d" ok "$secs" "$got"
        else result "$d" FAIL "$secs" "$got (the card's: $card)"; fi
      done < <(entries) ;;
    rodinia)
      while IFS='|' read -r app src flags args file sum; do
        log="$report/logs/$app.log"; wd="$src/run-$app"; rm -rf "$wd"; mkdir -p "$wd"
        [[ -x $src/$app ]] || { result "$app" FAIL 0 "did not build"; continue; }
        # shellcheck disable=SC2086
        run_one "$app" "$wd" "$log" "../$app" $args
        [[ $mode == digest ]] && { digest_result "$app"; continue; }
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
        [[ $mode == digest ]] && { digest_result "$name"; continue; }
        got=$(grep -oE 'Percent: [0-9]+' "$log" | awk '{print $2}' | paste -sd,)
        if [[ $rc != 0 ]]; then result "$name" FAIL "$secs" "exit $rc"
        elif [[ "$got" == "$card" ]]; then result "$name" ok "$secs" "non-matching outputs: $got"
        else result "$name" FAIL "$secs" "non-matching outputs: ${got:-none printed} (the card's: $card)"; fi
      done < <(entries) ;;
    cuda-memtest)
      # name | arguments | fault | the line it prints when its tests are done.
      # fault: "clean", or "stuck:<offset>": a memory cell stuck at 1, armed
      # with `vgpu fault`, that cuda_memtest must report (with --exit_on_error
      # it then exits non-zero).
      verdict="as expected"
      vgpu="$shim/../vgpu"
      while IFS='|' read -r name args fault marker; do
        log="$report/logs/$name.log"
        exe=build/cuda_memtest
        [[ -x $exe ]] || { result "$name" FAIL 0 "did not build"; continue; }
        if [[ $fault == stuck:* ]]; then
          st="$report/state-$name"; rm -rf "$st"; mkdir -p "$st/session"
          export VGPU_STATE_DIR="$st/state" VGPU_SESSION="$st/session" VGPU_TELEMETRY_PATH="$st/run" VGPU_DEVICE_COUNT=1
          VGPU_GPU="$gpu" "$vgpu" fault stuck --offset "${fault#stuck:}" --bit 3 --value 1 > /dev/null
        fi
        # shellcheck disable=SC2086
        run_one "$name" . "$log" "./$exe" $args
        unset VGPU_STATE_DIR VGPU_SESSION VGPU_TELEMETRY_PATH VGPU_DEVICE_COUNT
        if [[ $fault == stuck:* ]]; then
          if [[ $rc == 0 ]]; then result "$name" FAIL "$secs" "exit 0 with a stuck cell armed"
          elif grep -q 'errors found in block' "$log"; then result "$name" ok "$secs" "found the stuck cell"
          else result "$name" FAIL "$secs" "exit $rc, no error reported"; fi
        elif [[ $rc != 0 ]]; then result "$name" FAIL "$secs" "exit $rc"
        elif grep -q 'ERROR' "$log"; then result "$name" FAIL "$secs" "reported errors"
        elif ! grep -qF -- "$marker" "$log"; then result "$name" FAIL "$secs" "no \"$marker\""
        else result "$name" ok "$secs" ""; fi
      done < <(entries) ;;
    gpu-burn)
      # name | arguments | the line it prints when every GPU passed.
      verdict="as expected"
      # It runs nvidia-smi for temperatures: the simulator's, beside the shim.
      export PATH="$shim/../bin:$PATH"
      while IFS='|' read -r name args marker; do
        log="$report/logs/$name.log"
        [[ -x ./gpu_burn ]] || { result "$name" FAIL 0 "did not build"; continue; }
        # shellcheck disable=SC2086
        run_one "$name" . "$log" ./gpu_burn $args
        if [[ $rc != 0 ]]; then result "$name" FAIL "$secs" "exit $rc"
        elif grep -qE 'FAULTY|errors' "$log"; then result "$name" FAIL "$secs" "reported errors"
        elif ! grep -qF -- "$marker" "$log"; then result "$name" FAIL "$secs" "no \"$marker\""
        else result "$name" ok "$secs" ""; fi
      done < <(entries) ;;
  esac
  {
    if [[ $mode == digest ]]; then echo "### $suite on $gpu, SASS against PTX: $((total - fails)) of $total leave the same memory"
    else echo "### $suite on $gpu: $((total - fails)) of $total ${verdict:-as on the RTX 3060}"; fi
    echo
    if (( fails )); then
      echo "| program | seconds | result |"; echo "|---|---|---|"
      awk -F'\t' '$2 != "ok" {printf "| %s | %s | %s |\n", $1, $3, $4}' "$report/results.tsv"
      echo
    fi
    echo "Slowest: $(sort -t$'\t' -k3,3nr "$report/results.tsv" | head -5 | awk -F'\t' '{printf "%s %ss, ", $1, $3}' | sed 's/, $//')"
  } > "$report/summary.md"
  if [[ $mode == digest ]]; then echo "$suite: $((total - fails)) of $total leave the same memory on SASS and PTX"
  else echo "$suite: $((total - fails)) of $total ${verdict:-match the card}"; fi
  (( total > 0 && fails == 0 ))
}

mode=$cmd
case $cmd in
  fetch) fetch ;;
  build) build ;;
  run|digest) run "$@" ;;
  *) echo "unknown command $cmd"; exit 2 ;;
esac
