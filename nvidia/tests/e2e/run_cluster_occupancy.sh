#!/usr/bin/env bash
# cudaOccupancyMaxActiveClusters, cudaOccupancyMaxPotentialClusterSize and the driver's twins
# (cluster_occupancy.cu, nvidia/docs/clusters.md).
#
#   run_cluster_occupancy.sh                   against the shims
#   run_cluster_occupancy.sh --card            against NVIDIA's libraries on a real GPU
#   run_cluster_occupancy.sh --card --update   rewrite the expected file from the card
#   CLUSTER_OCC_SLUG=h100 CLUSTER_OCC_ARCH=sm_90a run_cluster_occupancy.sh --card --update
#                                              the same on another card (a Hopper: its own expected file
#                                              cluster_occupancy.h100.expected, built for its own SASS)
#   run_cluster_occupancy.sh h100              the shim's h100 profile against that file (once a card wrote it)
#
# An RTX 3060 has no clusters; what it answers (measured) is
# nvidia/tests/data/cluster_occupancy.rtx3060.expected, and the shim's rtx3060 profile must print the same.
# For a part with clusters there is no card here: the answers are checked against the layout NVIDIA
# publishes for the part (GPCs and TPCs; the spread of the SMs over the GPCs is derived, evenly) by the
# rules of the CUDA documentation, computed again below independently of the library.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/cluster_occupancy.cu"
expected="$root/nvidia/tests/data/cluster_occupancy.rtx3060.expected"
out="${TMPDIR:-/tmp}/vgpu-cluster-occupancy.$$"
card=0; update=0; slug_arg=""
slug="${CLUSTER_OCC_SLUG:-rtx3060}"; arch="${CLUSTER_OCC_ARCH:-sm_86}"
for a in "$@"; do
  case "$a" in
    --card) card=1 ;;
    --update) update=1 ;;
    [a-z]*[0-9a-z]) slug_arg="$a" ;;
    *) echo "usage: $0 [--card [--update]]" >&2; exit 2 ;;
  esac
done
[[ -n "$slug_arg" ]] && slug="$slug_arg"
expected="$root/nvidia/tests/data/cluster_occupancy.$slug.expected"
command -v nvcc >/dev/null 2>&1 || { echo "SKIP: nvcc not found"; exit 0; }
trap 'rm -f "$out" "$out".*' EXIT

if (( card )); then
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  libs=""
  for d in "$cuda_root/targets/x86_64-linux/lib" "$cuda_root/lib64"; do
    [[ -e "$d/libcudart.so" || -e "$d"/libcudart.so.[0-9]* ]] && libs="$d" && break
  done
  [[ -n "$libs" && -e "$libs/stubs/libcuda.so" ]] || { echo "SKIP: no CUDA libraries beside nvcc"; exit 0; }
  nvcc -std=c++17 -cudart shared -arch="$arch" -Wno-deprecated-gpu-targets "$src" -o "$out" -lcuda -L"$libs/stubs" || { echo "FAIL: does not compile"; exit 1; }
  LD_LIBRARY_PATH="$libs" "$out" > "$out.txt" || { echo "FAIL: the program failed on the card"; exit 1; }
  if (( update )); then cp "$out.txt" "$expected"; echo "wrote $expected"; exit 0; fi
  diff -u "$expected" "$out.txt" && echo "cluster occupancy matches the card's" || { echo "FAIL: differs from what the card printed"; exit 1; }
  exit 0
fi

shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
"$nvcc_bin" -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets "${san_flags[@]}" "$src" -o "$out" -lcuda -L"$shim" \
  || { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$out"; then exit 0; fi

run() {   # profile -> output file
  env $(both_shims_env "$shim") VGPU_QUIET=1 VGPU_GPU="$1" LD_LIBRARY_PATH="$shim" "$out" > "$2" 2>"$2.err" || { echo "FAIL: $1: the program failed"; cat "$2.err"; exit 1; }
}

run nvidia/rtx3060 "$out.3060"
diff -u "$root/nvidia/tests/data/cluster_occupancy.rtx3060.expected" "$out.3060" || { echo "FAIL: the rtx3060 profile does not print what the card printed"; exit 1; }
echo "rtx3060: the card's answers"

# Every other card that wrote an expected file (cluster_occupancy.<slug>.expected, from --card --update on that card):
# the profile of the same name must print what the card printed. Hopper's file is the one that decides the GPC layout.
for f in "$root"/nvidia/tests/data/cluster_occupancy.*.expected; do
  s="${f##*/cluster_occupancy.}"; s="${s%.expected}"
  [[ "$s" == rtx3060 ]] && continue
  run "nvidia/$s" "$out.$s"
  diff -u "$f" "$out.$s" || { echo "FAIL: the $s profile does not print what the card printed (see the diff: a GPC layout the card contradicts)"; exit 1; }
  echo "$s: the card's answers"
done

# Parts with clusters: GPCs as the profile has them (SMs per GPC, derived spread of NVIDIA's published
# counts), checked by the rules of vgpu/exec/cluster.hpp written out again.
python3 - "$shim" "$out" "$root" <<'PY'
import os, re, subprocess, sys
shim, prog, root = sys.argv[1:4]
def spread(gpcs, tpcs):   # the derived spread: even, larger first, two SMs a TPC
    return [(tpcs // gpcs + (1 if g < tpcs % gpcs else 0)) * 2 for g in range(gpcs)]
# profile -> (SM list per GPC or None when NVIDIA does not publish the GPC count, SMs)
parts = {
    "nvidia/h100": (spread(8, 66), 132),          # Hopper In-Depth: 8 GPCs, 66 TPCs
    "nvidia/h200": (spread(8, 66), 132),          # derived from the H100 SXM5
    "nvidia/rtx5090": (spread(11, 85), 170),      # RTX Blackwell whitepaper: 11 GPCs, 85 TPCs
    "nvidia/rtx-pro-6000": (spread(12, 94), 188), # 12 GPCs, 94 TPCs
    "nvidia/h100-pcie": (None, 114),              # '7 or 8 GPCs': not published
    "nvidia/b200": (None, 148),                   # no published GPC layout
}
bad = 0
for prof, (gpc, sms) in parts.items():
    env = dict(os.environ, VGPU_QUIET="1", VGPU_GPU=prof, LD_LIBRARY_PATH=shim)
    r = subprocess.run([prog], env=env, capture_output=True, text=True)
    if r.returncode:
        print(f"FAIL: {prof}: the program failed\n{r.stderr}"); bad += 1; continue
    lines = r.stdout.splitlines()
    per_sm = int(next(l for l in lines if l.startswith("blocks per sm")).split()[-1])
    nonportable = False
    ok = lambda code: code in ("cudaSuccess", "CUDA_SUCCESS")
    for l in lines:
        if l.startswith("non-portable sizes"):
            nonportable = "allowed" in l and "not" not in l
            continue
        m = re.match(r"(active|potential), (cluster\s*(\d+)(x(\d+))?|no dimension|1 MiB of shared memory): (\w+) (-?\d+)(?: \| driver (\w+) (-?\d+))?", l)
        if not m:
            continue
        kind, what, size, _, y, code, val, dcode, dval = m.groups()
        limit = 16 if nonportable else 8
        if what.startswith("cluster"):
            size = int(size) * (int(y) if y else 1)
        if kind == "active":
            if what == "no dimension":
                want = ("cudaErrorInvalidValue", "CUDA_ERROR_INVALID_VALUE", -7)
            elif size > limit:
                want = ("cudaErrorInvalidClusterSize", "CUDA_ERROR_INVALID_CLUSTER_SIZE", -7)
            elif size == 1:
                want = ("cudaSuccess", "CUDA_SUCCESS", per_sm * sms)
            elif gpc is None:
                want = ("cudaErrorNotSupported", "CUDA_ERROR_NOT_SUPPORTED", -7)
            else:
                want = ("cudaSuccess", "CUDA_SUCCESS", sum(g * per_sm // size for g in gpc))
        else:
            if what.startswith("1 MiB"):
                want = ("cudaSuccess", "CUDA_SUCCESS", 0)
            elif limit <= 8:
                want = ("cudaSuccess", "CUDA_SUCCESS", 8)
            elif gpc is None:
                want = ("cudaErrorNotSupported", "CUDA_ERROR_NOT_SUPPORTED", -7)
            else:
                want = ("cudaSuccess", "CUDA_SUCCESS", min(16, max(gpc) * per_sm))
        got = (code, int(val))
        if got != (want[0], want[2]) or (dcode is not None and (dcode, int(dval)) != (want[1], want[2])):
            print(f"FAIL: {prof} {'non-portable' if nonportable else 'portable'} {l}  (expected {want})"); bad += 1
    print(f"{prof}: checked against the layout" + (" (derived spread)" if gpc else " (no published GPC layout: refused)"))
sys.exit(1 if bad else 0)
PY
rc=$?
(( rc == 0 )) || { echo "FAIL: cluster occupancy of the parts with clusters"; exit 1; }
echo "PASS"
