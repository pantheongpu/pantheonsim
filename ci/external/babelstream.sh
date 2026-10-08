#!/usr/bin/env bash
# BabelStream (UoB-HPC, HIP model), unmodified, on the simulated AMD GPUs. It is
# a memory-bandwidth benchmark whose every run ends in a check of its three
# arrays against the values the kernels must have left (Copy, Mul, Add, Triad
# and Dot, repeated), exiting non-zero with "FAILED validation" when memory
# holds anything else: so it is also a memory test, and passes or fails by its
# own exit status. There is no card's output to compare with (bandwidth is not
# something the simulator models), so the numbers it prints are not checked.
#
#   babelstream.sh fetch <dir>
#   babelstream.sh build <dir> <arch>            e.g. gfx942, with ROCM_PATH's hipcc
#   babelstream.sh run   <dir> <gpu> <shim-dir> <report-dir>
#
# `run` checks, on one GPU: double and float precision and the one-kernel mode (--only Triad),
# which pass; then, against a `vgpu fault stuck` cell in the first array
# (BabelStream allocates it first, so it starts device memory), that the
# validation FAILS with exit status 1, and passes again once the cell is
# cleared. ROCM_PATH (/opt/rocm-7.1.0) is used only to build. BS_ELEMENTS
# (262144) is the array length: small, the simulator is a CPU. The vgpu CLI is
# found next to the shim directory (build/vgpu), or in $VGPU.
set -uo pipefail
cmd="${1:?fetch|build|run}"; dir="${2:?dir}"; shift 2
rocm="${ROCM_PATH:-/opt/rocm-7.1.0}"
commit=17ab377b0e919e14fd3df2b67268761fdac8abb3   # BabelStream main, 2026-06-15 (version 5.0)
n="${BS_ELEMENTS:-262144}"

case $cmd in
fetch)
  mkdir -p "$dir" && cd "$dir" || exit 1
  git init -q . && git remote add origin https://github.com/UoB-HPC/BabelStream 2>/dev/null
  git fetch -q --depth 1 origin "$commit" && git -c advice.detachedHead=false checkout -q FETCH_HEAD
  echo "BabelStream at $(git rev-parse --short HEAD)" ;;

build)
  arch="${1:?arch}"; b="$dir/build-$arch"
  export PATH="$rocm/bin:$PATH"   # hipcc finds its own ROCm from where it is
  cmake -G Ninja -S "$dir" -B "$b" -DMODEL=hip -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$rocm/bin/hipcc" \
    -DCMAKE_CXX_FLAGS="--offload-arch=$arch" > "$b.configure.log" 2>&1 || { tail -30 "$b.configure.log"; exit 1; }
  nice -n 15 ninja -C "$b" -j"${BS_JOBS:-2}" > "$b.build.log" 2>&1 || { tail -30 "$b.build.log"; exit 1; }
  echo "built $b/hip-stream" ;;

run)
  gpu="${1:?gpu}"; shim="$(cd "${2:?shim}" && pwd)"; report="$(mkdir -p "${3:?report}" && cd "$3" && pwd)"
  case "$gpu" in
    amd/mi300x|amd/mi325x) arch=gfx942 ;; amd/mi250x) arch=gfx90a ;; amd/mi350x) arch=gfx950 ;;
    amd/rx6900xt) arch=gfx1030 ;; amd/rx7900xtx) arch=gfx1100 ;; amd/rx9070xt) arch=gfx1201 ;;
    *) echo "unknown GPU $gpu" >&2; exit 2 ;;
  esac
  exe="$dir/build-$arch/hip-stream"
  [[ -x $exe ]] || { echo "not built: $exe"; exit 2; }
  vgpu="${VGPU:-$(dirname "$shim")/vgpu}"
  [[ -x $vgpu ]] || { echo "no vgpu at $vgpu"; exit 2; }
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT; mkdir -p "$tmp/session"
  export VGPU_TELEMETRY_PATH="$tmp/run" VGPU_STATE_DIR="$tmp/state" VGPU_SESSION="$tmp/session"
  export VGPU_GPU="$gpu" VGPU_DEVICE_COUNT=1 VGPU_QUIET=1
  fail=0; : > "$report/results.tsv"
  # step <name> <expected exit status> <text the output must hold> <babelstream args...>
  step() {
    local name=$1 want=$2 marker=$3; shift 3
    local log="$report/$(tr ' /,' '___' <<<"$gpu.$name").log" got
    LD_LIBRARY_PATH="$shim" timeout -k 5 "${BS_TIMEOUT:-600}" "$exe" --arraysize "$n" --numtimes 3 "$@" > "$log" 2>&1
    got=$?
    if [[ $got == "$want" ]] && grep -qF -- "$marker" "$log"; then
      echo "ok    $name"; printf '%s\tpass\n' "$name" >> "$report/results.tsv"
    else
      echo "FAIL  $name (exit $got, wanted $want, looking for '$marker')"; tail -5 "$log" | sed 's/^/      /'
      printf '%s\tFAIL\n' "$name" >> "$report/results.tsv"; fail=1
    fi
  }
  "$vgpu" fault stuck --clear > /dev/null 2>&1
  step "double precision, all five kernels" 0 "Precision: double"
  step "single precision" 0 "Precision: float" --float
  step "triad only" 0 "Triad" --only Triad
  # A stuck cell in the first array: the top exponent bits of one double.
  "$vgpu" fault stuck --offset 0x107 --bit 6 --value 1 > /dev/null
  step "a stuck cell fails the validation" 1 "FAILED validation of a"
  "$vgpu" fault stuck --clear > /dev/null
  step "cleared, it validates again" 0 "Precision: double"
  { echo "### BabelStream (HIP) on $gpu"; echo; sed 's/^/    /' "$report/results.tsv"; } > "$report/summary.md"
  exit $fail ;;
*) echo "usage: $0 fetch|build|run" >&2; exit 2 ;;
esac
