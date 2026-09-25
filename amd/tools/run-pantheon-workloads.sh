#!/usr/bin/env bash
# Builds the pantheon GPU workloads for AMD and runs them on a simulated
# MI300X, each one checking its own results.
#
# The workload sources are UNMODIFIED, and they are built exactly as
# pantheon's Makefile builds them for HIP: hipcc, the same flags, gfx942. The
# executables link libamdhip64 as they would against ROCm, and VirtualGPU's
# takes its place at run time. Two things differ from a run on a card:
#   1. the workloads' own knobs are set to a CPU-appropriate intensity --
#      fewer kernel loops, a smaller grid, a small slice of a small device --
#      which runs the same code over a smaller working set; and
#   2. every workload runs with --verify, so a pass means the workload checked
#      its results, not only that it ran.
# A workload that skips itself on a CDNA part (it needs ray-tracing units,
# RDNA's matrix instructions, or NVIDIA's encoder) reports SKIP with its own
# reason, as it would on a card.
#
# Needs ROCm's hipcc (set VGPU_ROCM_PATH; /opt/rocm and the newest ~/.local/share/rocm-*
# are searched) and a pantheongpu checkout.
#
#   amd/tools/run-pantheon-workloads.sh [pantheongpu-repo] [outdir]
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
repo="${1:-$here/../pantheongpu}"
out="${2:-${TMPDIR:-/tmp}/vgpu-pantheon-amd}"
build="${VGPU_BUILD_DIR:-$here/build}"
shim="$build/shim"
bin="$out/bin"; logs="$out/logs"; code="$out/code"
mkdir -p "$bin" "$logs" "$code"

: "${VGPU_WL_GPU:=amd/mi300x}"
: "${VGPU_WL_VRAM_MB:=64}"
: "${VGPU_WL_DURATION:=5}"
: "${VGPU_WL_TIMEOUT:=300}"
: "${VGPU_WL_LOOPS:=2}"
: "${VGPU_WL_GRID:=4}"
: "${VGPU_WL_MEMPCT:=2}"
: "${VGPU_WL_DEVICES:=2}"
vram_for() { case "$1" in pcie_bandwidth|p2p_thrasher|all_reduce) echo 1024 ;; *) echo "$VGPU_WL_VRAM_MB" ;; esac; }

rocm=""
for c in "${VGPU_ROCM_PATH:-}" "${ROCM_PATH:-}" /opt/rocm $(ls -d "$HOME"/.local/share/rocm-*/opt/rocm-* 2>/dev/null | sort -rV); do
  [[ -n "$c" && -x "$c/bin/hipcc" ]] && { rocm=$c; break; }
done
[[ -n "$rocm" ]] || { echo "SKIP: no ROCm hipcc found (set VGPU_ROCM_PATH)"; exit 0; }
[[ -d "$repo/kernels" ]] || { echo "SKIP: no pantheongpu checkout at $repo"; exit 0; }
[[ -e "$shim/libamdhip64.so.7" ]] || { echo "error: build VirtualGPU first (no $shim/libamdhip64.so.7)" >&2; exit 1; }
# ROCm 5 kept the compiler in llvm/ and the device libraries under clang's
# own directory; later releases in lib/llvm/ and amdgcn/.
llvm=$([[ -d $rocm/lib/llvm/bin ]] && echo "$rocm/lib/llvm" || echo "$rocm/llvm")
bitcode=$([[ -d $rocm/amdgcn/bitcode ]] && echo "$rocm/amdgcn/bitcode" ||
          ls -d "$llvm"/lib/clang/*/lib/amdgcn/bitcode 2>/dev/null | head -1)
export ROCM_PATH=$rocm HIP_PATH=$rocm HIP_CLANG_PATH=$llvm/bin HIP_DEVICE_LIB_PATH=$bitcode

srcs=$(cd "$repo" && find kernels -name '*.cpp' ! -path 'kernels/common/*' | sort)
echo "Building $(wc -w <<<"$srcs") workloads with $rocm/bin/hipcc (unmodified sources)..."
for src in $srcs; do
  w=$(basename "$src" .cpp)
  ( cd "$repo" && "$rocm/bin/hipcc" -O3 -std=c++14 -DNDEBUG -Ikernels/common -std=c++17 --offload-arch=gfx942 \
      "$src" -o "$bin/$w" ) >"$logs/build_$w.log" 2>&1 &
  while [[ $(jobs -r | wc -l) -ge $(nproc) ]]; do wait -n; done
done
wait

# Every instruction the workloads' kernels contain, decoded and compared with
# ROCm's own listing -- including the paths a normal run never takes.
decode_fail=0
if [[ -x "$build/test_amd_gcn" && -x "$llvm/bin/llvm-objdump" ]]; then
  for exe in "$bin"/*; do
    w=$(basename "$exe")
    python3 - "$exe" "$code/$w.gfx942.o" <<'PY' || continue
import struct, subprocess, sys
exe, dest = sys.argv[1], sys.argv[2]
out = subprocess.run(['readelf', '-S', '--wide', exe], capture_output=True, text=True).stdout
for line in out.splitlines():
    if '] .hip_fatbin' in line:
        f = line.split(']', 1)[1].split(); off, size = int(f[3], 16), int(f[4], 16); break
else:
    sys.exit(1)   # no device code: a workload with no kernels
b = open(exe, 'rb').read()[off:off + size]
n = struct.unpack_from('<Q', b, 24)[0]; at = 32
for _ in range(n):
    eo, es, tl = struct.unpack_from('<QQQ', b, at); at += 24
    triple = b[at:at + tl].decode(); at += tl
    if triple.endswith('gfx942') or '--gfx942:' in triple:
        open(dest, 'wb').write(b[eo:eo + es]); sys.exit(0)
sys.exit(1)
PY
    "$llvm/bin/llvm-objdump" -d --mcpu=gfx942 "$code/$w.gfx942.o" |
      sed -n 's/^\t\(.*\)\/\/ .*/\1/p' | sed 's/[[:space:]]*$//; s/  */ /g' > "$code/$w.gfx942.dis"
    res=$(VGPU_GCN_OBJECT="$code/$w.gfx942.o" VGPU_GCN_LISTING="$code/$w.gfx942.dis" "$build/test_amd_gcn" 2>&1)
    if ! grep -q '^\[ PASS \] every_instruction_decodes' <<<"$res"; then
      echo "decode: $w: $(grep -m1 -v '^\[' <<<"$res" | cut -c1-120)"; decode_fail=1
    fi
  done
  [[ $decode_fail -eq 0 ]] && echo "Every workload's device code decodes as ROCm's llvm-objdump prints it."
fi

pass=0; fail=0; skip=0
printf '\n%-26s %-6s %s\n' WORKLOAD RESULT DETAIL
printf '%-26s %-6s %s\n' "--------------------------" "------" "------"
for src in $srcs; do
  w=$(basename "$src" .cpp)
  if [[ ! -x "$bin/$w" ]]; then
    why=$(grep -m1 'error:' "$logs/build_$w.log" | sed 's/^.*error: //' | cut -c1-70)
    printf '%-26s %-6s %s\n' "$w" SKIP "does not build for gfx942: $why"; ((skip++)); continue
  fi
  start=$SECONDS
  VGPU_QUIET=1 VGPU_GPU="$VGPU_WL_GPU" VGPU_VRAM_MB="$(vram_for "$w")" VGPU_DEVICE_COUNT="$VGPU_WL_DEVICES" \
    LD_LIBRARY_PATH="$shim" timeout "$VGPU_WL_TIMEOUT" \
    "$bin/$w" 0 "$VGPU_WL_DURATION" "$VGPU_WL_MEMPCT" \
    --kernel_loops "$VGPU_WL_LOOPS" --warmup_iters 1 --grid_size "$VGPU_WL_GRID" --verify \
    >"$logs/$w.log" 2>&1
  rc=$?; el=$((SECONDS - start))
  skipped=$(grep -m1 -E 'Skipping' "$logs/$w.log" | sed 's/.*Skipping *//' | cut -c1-70)
  verdict=$(grep -m1 -E 'Verification: (PASS|FAIL)|\[PASS\]' "$logs/$w.log" | cut -c1-60)
  if [[ $rc -eq 0 && -n "$skipped" ]]; then
    printf '%-26s %-6s %s\n' "$w" SKIP "the workload skips itself: $skipped"; ((skip++))
  elif [[ $rc -eq 0 ]]; then
    printf '%-26s %-6s %3ss  %s\n' "$w" PASS "$el" "${verdict:-ran its duration (no kernels to verify)}"; ((pass++))
  else
    why=$(grep -m1 -iE 'error|FAIL' "$logs/$w.log" | cut -c1-70)
    printf '%-26s %-6s %3ss  rc=%s %s\n' "$w" FAIL "$el" "$rc" "$why"; ((fail++))
  fi
done
printf '\n%d passed, %d failed, %d skipped. Logs in %s\n' "$pass" "$fail" "$skip" "$logs"
[[ $fail -eq 0 && $decode_fail -eq 0 ]]
