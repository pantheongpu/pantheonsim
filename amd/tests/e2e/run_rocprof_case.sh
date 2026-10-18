#!/usr/bin/env bash
# What AMD's profiler, rocprofv3, records for a small HIP program, compared with
# what it recorded for the same program on a real Radeon card.
#
#   run_rocprof_case.sh <case> [--gpu rx6800|rx6700xt]
#         run the case on the simulated GPU under rocprofv3 (inside `vgpu shell`),
#         normalise the trace (amd/tests/rocprof_normalize.py: no times, ids,
#         pointers or HSA records) and compare it with
#         amd/tests/data/rocprof_<case>.expected  (rx6700xt: ...<case>.rx6700xt.expected)
#   run_rocprof_case.sh <case> --card [--gpu ...]
#         run it on the real card instead, against ROCm's own libraries, and
#         compare with the same file: the check that the expected output is what
#         hardware prints
#   run_rocprof_case.sh <case> --card --update [--gpu ...]
#         rewrite the expected file from the card
#
# Cases (amd/tests/rocprof/<case>.cpp):
#   vecadd          allocate, copy in, launch, synchronize, copy out, free
#   memset_d2d      fills and device-to-device copies, blocking and on a stream
#   copies          1 MiB copies to and from pinned memory (DMA copy records)
#   streams_events  two streams, ordered by an event
#   graph           a captured stream, instantiated and launched twice
#   async_alloc     hipMallocAsync / hipFreeAsync around a copy and a kernel
#   markers         ROCTx marks and ranges
#   markers_control ROCTx profiler pause and resume, a named stream, nested ranges, marks from another thread
#   geometry        3D grid, static plus dynamic LDS, scratch
#   args            one call per kind of HIP parameter: how null and live pointers, handles, strings,
#                   enums, integers and structs by value print
#
# The simulator has no timing model, so nothing in the comparison is a time.
# The card's expected files were recorded on an RX 6800 (gfx1030) and, for
# vecadd, an RX 6700 XT, with ROCm 7.2 and rocprofv3 1.1.0 (amd/docs/rocprofiler.md).
#
# SKIPs where there is no ROCm with rocprofv3 and hipcc to build the case with, or where its rocprofv3 is
# not the release the expected files were recorded with (ROCPROF_ANY_VERSION=1 compares anyway).
# With --card it also SKIPs where there is no AMD GPU (/dev/kfd). On the card,
# ROCPROF_CARD_DEVICE picks the HIP device (default 0 for rx6800, 1 for rx6700xt).
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
usage="usage: $0 <vecadd|memset_d2d|copies|streams_events|graph|async_alloc|markers|markers_control|geometry|args> [--gpu rx6800|rx6700xt] [--card [--update]]"
case_name="${1:?$usage}"
shift
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
src="$root/amd/tests/rocprof/${case_name}.cpp"
[[ -f "$src" ]] || { echo "no such case: $case_name" >&2; exit 2; }
suffix=""; [[ "$gpu" == rx6700xt ]] && suffix=".rx6700xt"
expected="$root/amd/tests/data/rocprof_${case_name}${suffix}.expected"
if (( update )) && (( ! card )); then echo "--update needs --card: the expected files come from a real card" >&2; exit 2; fi

command -v python3 >/dev/null || { echo "SKIP: the normaliser is a python3 script"; exit 0; }
rocm=""
for c in "${VGPU_ROCM_PATH:-}" "${ROCM_PATH:-}" /opt/rocm $(ls -d "$HOME"/.local/share/rocm-*/opt/rocm-* 2>/dev/null | sort -rV); do
  [[ -n "$c" && -x "$c/bin/rocprofv3" && -x "$c/bin/hipcc" && -e "$c/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so" ]] && { rocm=$c; break; }
done
[[ -n "$rocm" ]] || { echo "SKIP: no ROCm with rocprofv3 and hipcc found (set VGPU_ROCM_PATH)"; exit 0; }

# The expected files are what ROCm 7.2's rocprofv3 (1.1.0) printed for what its hipcc built; another release
# can print a different trace of the same program (register counts, record kinds, names). The files say which
# in their header: SKIP where this ROCm is another, unless ROCPROF_ANY_VERSION=1.
rocprof_version=$("$rocm/bin/rocprofv3" --version 2>&1 | sed -n 's/^ *version *: *\([0-9][0-9.]*\).*/\1/p' | head -1)
if (( ! update )); then
  [[ -f "$expected" ]] || { echo "FAIL: no expected file $expected"; exit 1; }
  recorded=$(sed -n 's/^# recorded on:.* rocprofv3 \([0-9][0-9.]*\).*/\1/p' "$expected" | head -1)
  if [[ -n "$recorded" && "$recorded" != "$rocprof_version" && -z "${ROCPROF_ANY_VERSION:-}" ]]; then
    echo "SKIP: the expected trace is from rocprofv3 $recorded, this ROCm's is ${rocprof_version:-unknown} (ROCPROF_ANY_VERSION=1 compares anyway)"; exit 0
  fi
fi

work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_rocprof_${case_name}.XXXXXX")"
trap 'rm -rf "$work"' EXIT
exe="$work/$case_name"
export ROCM_PATH="$rocm"

# The cases are built for gfx1030, the RX 6800; the RX 6700 XT (gfx1031) runs
# the same code object (HSA_OVERRIDE_GFX_VERSION=10.3.0 on the card).
libs=""; [[ "$case_name" == markers* ]] && libs="-lrocprofiler-sdk-roctx"
if ! "$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx1030 -w "$src" -o "$exe" $libs >"$work/build.log" 2>&1; then
  if grep -q "rocprofiler-sdk-roctx" "$work/build.log" && [[ "$case_name" == markers* ]] && grep -qE "cannot find|not found|No such file" "$work/build.log"; then
    echo "SKIP: this ROCm has no ROCTx (rocprofiler-sdk-roctx)"; exit 0
  fi
  cat "$work/build.log"; echo "FAIL: hipcc could not build the $case_name case"; exit 1
fi

flags=(--hip-trace --kernel-trace --memory-copy-trace --memory-allocation-trace --hsa-trace --marker-trace --scratch-memory-trace
       --stats --output-format json)
if (( card )); then
  [[ -e /dev/kfd ]] || { echo "SKIP: no AMD GPU (/dev/kfd) for --card"; exit 0; }
  dev="${ROCPROF_CARD_DEVICE:-0}"; override=""
  if [[ "$gpu" == rx6700xt ]]; then dev="${ROCPROF_CARD_DEVICE:-1}"; override="10.3.0"; fi
  env ${override:+HSA_OVERRIDE_GFX_VERSION=$override} HIP_VISIBLE_DEVICES="$dev" \
      timeout 300 "$rocm/bin/rocprofv3" "${flags[@]}" -d "$work/out" -- "$exe" >"$work/run.log" 2>&1
  status=$?
else
  vgpu="${VGPU_BUILD_DIR:-$root/build}/vgpu"
  build="$(dirname "$vgpu")"
  [[ -x "$vgpu" && -e "$build/shim/librocprofiler-sdk.so.1" ]] || { echo "SKIP: build vgpu and vgpurocprof first"; exit 0; }
  # ROCm's rocprofv3 tool is built without a sanitizer: see run_rocprofv3.sh.
  if objdump -p "$build/shim/libamdhip64.so.7" 2>/dev/null | awk '/NEEDED/ {print $2}' | grep -qE 'lib(a|t)san'; then
    echo "SKIP: the shim is built with a sanitizer, and ROCm's rocprofv3 tool is not"; exit 0
  fi
  unset VGPU_GPU VGPU_DEVICE_COUNT VGPU_SESSION
  # The RX 6700 XT (gfx1031) runs gfx1030 code the way the card does.
  [[ "$gpu" == rx6700xt ]] && export HSA_OVERRIDE_GFX_VERSION=10.3.0
  export VGPU_TELEMETRY_PATH="$work/telemetry"
  mkdir -p "$VGPU_TELEMETRY_PATH"
  timeout 600 "$vgpu" shell -y --no-isolate --gpu "amd/$gpu" --count 1 -c \
    "rocprofv3 ${flags[*]} -d '$work/out' -- '$exe'" </dev/null >"$work/run.log" 2>&1
  status=$?
fi
if (( status != 0 )); then
  sed 's/^/      /' "$work/run.log" | tail -30
  echo "FAIL: the $case_name case under rocprofv3 exited with $status"; exit 1
fi
if ! grep -q "^$case_name " "$work/run.log"; then
  sed 's/^/      /' "$work/run.log" | tail -30
  echo "FAIL: the $case_name program did not print its own result"; exit 1
fi
results=$(find "$work/out" -name '*_results.json' | head -1)
[[ -s "$results" ]] || { sed 's/^/      /' "$work/run.log" | tail -30; echo "FAIL: rocprofv3 wrote no results.json"; exit 1; }
# ROCPROF_KEEP=<dir> keeps the raw result, to see what the normaliser left out.
[[ -n "${ROCPROF_KEEP:-}" ]] && cp "$results" "$ROCPROF_KEEP/${case_name}.$(( card ))card.json"
python3 -I "$root/amd/tests/rocprof_normalize.py" "$results" >"$work/trace.txt" || { echo "FAIL: could not normalise $results"; exit 1; }

if (( update )); then
  product=$(sed -n 's/^gpu .*product_name=\(AMD Radeon [^=]*\) vendor_name=.*/\1/p' "$work/trace.txt" | head -1)
  isa=$(sed -n 's/^gpu name=\([^ ]*\) .*/\1/p' "$work/trace.txt" | head -1)
  {
    echo "# rocprofv3 trace of the $case_name case, normalised by amd/tests/rocprof_normalize.py"
    echo "# recorded on: $product ($isa), $(basename "$(readlink -f "$rocm")"), rocprofv3 ${rocprof_version:-?}"
    echo "# regenerate: amd/tests/e2e/run_rocprof_case.sh $case_name --card --update${suffix:+ --gpu $gpu}"
    cat "$work/trace.txt"
  } >"$expected"
  echo "wrote $expected"
  exit 0
fi

grep -v '^#' "$expected" >"$work/expected.txt"
who="the simulated $gpu"; (( card )) && who="the real $gpu"
if (( ! card )); then
  # What the simulator does not record yet (amd/docs/rocprofiler.md) is left out of both sides.
  for f in expected trace; do
    python3 -I "$root/amd/tests/rocprof_normalize.py" --strip-known-gaps <"$work/$f.txt" >"$work/$f.cmp" || exit 1
    mv "$work/$f.cmp" "$work/$f.txt"
  done
fi
if diff -u --label "expected (card)" --label "$who" "$work/expected.txt" "$work/trace.txt"; then
  echo "rocprofv3 $case_name trace of $who matches the card's"
else
  echo "FAIL: the $case_name trace of $who differs from what rocprofv3 recorded on the card"
  exit 1
fi
