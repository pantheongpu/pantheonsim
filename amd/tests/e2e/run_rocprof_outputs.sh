#!/usr/bin/env bash
# The files rocprofv3 writes besides results.json -- csv, otf2, pftrace, rocpd, --stats, --summary, --sys-trace and
# --kernel-rename -- for the vecadd program (amd/tests/rocprof/vecadd.cpp), compared with what the same
# commands wrote on a real Radeon card: the same files, the same csv columns, the same function names and call
# counts, the same rocpd tables and the same summary rows. Times are left out, and so is what the simulator does not
# record yet (amd/docs/rocprofiler.md); amd/tests/rocprof_outputs.py says what.
#
#   run_rocprof_outputs.sh [--gpu rx6800|rx6700xt]          the simulated card, compared with the expected file
#   run_rocprof_outputs.sh --card [--update] [--gpu ...]    the real card (ROCm's own libraries), compared with it
#                                                           or, with --update, rewriting amd/tests/data/rocprof_outputs[.rx6700xt].expected
#
# SKIPs where there is no ROCm with rocprofv3 and hipcc, or where rocprofv3 is not the release the file was recorded with
# (ROCPROF_ANY_VERSION=1 compares anyway).
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
usage="usage: $0 [--gpu rx6800|rx6700xt] [--card [--update]]"
gpu=rx6800; card=0; update=0
while (( $# )); do
  case "$1" in
    --card) card=1 ;;
    --update) update=1 ;;
    --gpu) shift; gpu="${1:-}" ;;
    *) echo "$usage" >&2; exit 2 ;;
  esac
  shift
done
case "$gpu" in rx6800|rx6700xt) ;; *) echo "$usage" >&2; exit 2 ;; esac
suffix=""; [[ "$gpu" == rx6700xt ]] && suffix=".rx6700xt"
expected="$root/amd/tests/data/rocprof_outputs${suffix}.expected"
if (( update )) && (( ! card )); then echo "--update needs --card: the expected files come from a real card" >&2; exit 2; fi
command -v python3 >/dev/null || { echo "SKIP: python3 needed"; exit 0; }
rocm=""
for c in "${VGPU_ROCM_PATH:-}" "${ROCM_PATH:-}" /opt/rocm $(ls -d "$HOME"/.local/share/rocm-*/opt/rocm-* 2>/dev/null | sort -rV); do
  [[ -n "$c" && -x "$c/bin/rocprofv3" && -x "$c/bin/hipcc" && -e "$c/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so" ]] && { rocm=$c; break; }
done
[[ -n "$rocm" ]] || { echo "SKIP: no ROCm with rocprofv3 and hipcc found (set VGPU_ROCM_PATH)"; exit 0; }
rocprof_version=$("$rocm/bin/rocprofv3" --version 2>&1 | sed -n 's/^ *version *: *\([0-9][0-9.]*\).*/\1/p' | head -1)
if (( ! update )); then
  [[ -f "$expected" ]] || { echo "FAIL: no expected file $expected"; exit 1; }
  recorded=$(sed -n 's/^# recorded on:.* rocprofv3 \([0-9][0-9.]*\).*/\1/p' "$expected" | head -1)
  if [[ -n "$recorded" && "$recorded" != "$rocprof_version" && -z "${ROCPROF_ANY_VERSION:-}" ]]; then
    echo "SKIP: the expected files are from rocprofv3 $recorded, this ROCm's is ${rocprof_version:-unknown} (ROCPROF_ANY_VERSION=1 compares anyway)"; exit 0
  fi
fi
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_rocprof_outputs.XXXXXX")"
trap 'rm -rf "$work"' EXIT
export ROCM_PATH="$rocm"
exe="$work/vecadd"
if ! "$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx1030 -w "$root/amd/tests/rocprof/vecadd.cpp" -o "$exe" >"$work/build.log" 2>&1; then
  cat "$work/build.log"; echo "FAIL: hipcc could not build vecadd"; exit 1
fi
traces=(--hip-trace --kernel-trace --memory-copy-trace --memory-allocation-trace --hsa-trace --marker-trace --scratch-memory-trace)
# label|rocprofv3 options
runs=(
  "all-formats|${traces[*]} --stats --summary --output-format csv json otf2 pftrace rocpd"
  "sys-trace|--sys-trace --output-format csv"
  "kernel-rename|--kernel-trace --kernel-rename --marker-trace --stats --output-format csv"
)
if (( ! card )); then
  vgpu="${VGPU_BUILD_DIR:-$root/build}/vgpu"
  build="$(dirname "$vgpu")"
  [[ -x "$vgpu" && -e "$build/shim/librocprofiler-sdk.so.1" ]] || { echo "SKIP: build vgpu and vgpurocprof first"; exit 0; }
  if objdump -p "$build/shim/libamdhip64.so.7" 2>/dev/null | awk '/NEEDED/ {print $2}' | grep -qE 'lib(a|t)san'; then
    echo "SKIP: the shim is built with a sanitizer, and ROCm's rocprofv3 tool is not"; exit 0
  fi
  unset VGPU_GPU VGPU_DEVICE_COUNT VGPU_SESSION
  [[ "$gpu" == rx6700xt ]] && export HSA_OVERRIDE_GFX_VERSION=10.3.0
  export VGPU_TELEMETRY_PATH="$work/telemetry"; mkdir -p "$VGPU_TELEMETRY_PATH"
else
  [[ -e /dev/kfd ]] || { echo "SKIP: no AMD GPU (/dev/kfd) for --card"; exit 0; }
  dev="${ROCPROF_CARD_DEVICE:-0}"; override=""
  if [[ "$gpu" == rx6700xt ]]; then dev="${ROCPROF_CARD_DEVICE:-1}"; override="10.3.0"; fi
fi
: >"$work/trace.txt"
for run in "${runs[@]}"; do
  label="${run%%|*}"; opts="${run#*|}"
  if (( card )); then
    env ${override:+HSA_OVERRIDE_GFX_VERSION=$override} HIP_VISIBLE_DEVICES="$dev" \
      timeout -k 5 300 "$rocm/bin/rocprofv3" $opts -d "$work/$label" -- "$exe" >"$work/$label.log" 2>&1
  else
    timeout -k 5 600 "$vgpu" shell -y --no-isolate --gpu "amd/$gpu" --count 1 -c \
      "rocprofv3 $opts -d '$work/$label' -- '$exe'" </dev/null >"$work/$label.log" 2>&1
  fi
  status=$?
  if (( status != 0 )) || ! grep -q "^vecadd wrong 0 " "$work/$label.log"; then
    sed 's/^/      /' "$work/$label.log" | tail -30; echo "FAIL: the $label run exited with $status"; exit 1
  fi
  python3 -I "$root/amd/tests/rocprof_outputs.py" "$label" "$work/$label" "$work/$label.log" >>"$work/trace.txt" || { echo "FAIL: could not reduce the $label output"; exit 1; }
done
if (( update )); then
  { echo "# what rocprofv3 writes besides results.json for the vecadd case, reduced by amd/tests/rocprof_outputs.py"
    echo "# recorded on: the $gpu card, $(basename "$(readlink -f "$rocm")"), rocprofv3 ${rocprof_version:-?}"
    echo "# regenerate: amd/tests/e2e/run_rocprof_outputs.sh --card --update${suffix:+ --gpu $gpu}"
    cat "$work/trace.txt"; } >"$expected"
  echo "wrote $expected"; exit 0
fi
grep -v '^#' "$expected" >"$work/expected.txt"
who="the simulated $gpu"; (( card )) && who="the real $gpu"
if diff -u --label "expected (card)" --label "$who" "$work/expected.txt" "$work/trace.txt"; then
  echo "rocprofv3's output files of $who match the card's"
else
  echo "FAIL: the files rocprofv3 wrote for $who differ from what it wrote on the card"; exit 1
fi
