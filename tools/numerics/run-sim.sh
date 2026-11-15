#!/usr/bin/env bash
# Runs the numerics programs on the simulator, one output directory a configuration, ready for compare.py:
#
#   tools/numerics/run-sim.sh <bin-dir> <build-dir> <out-dir> <config> [more configs ...]
#
#   <bin-dir>     what tools/numerics/build.sh wrote (sm_75/, sm_86/, ..., gfx942/, ...), and, if the Varity programs are wanted,
#                 varity-gen/ (gen.py) and varity-bin-sm_<sm>/ (or varity-bin/, any architecture's) or varity-bin-<gfx>/ (varity/build.sh)
#   <config>      nvidia/<profile>:<sm>:<sass|ptx>     e.g. nvidia/t4:75:sass   nvidia/h100:90:ptx
#                 amd/<profile>:<gfx>                  e.g. amd/mi300x:gfx942   amd/rx6800:gfx1030
#   <out-dir>/<vendor>.<profile>.<arch>[.<engine>]/    mathfn.txt mathfn_fast.txt fp_examples.txt fp_examples_nofma.txt
#                                                      varity_cases.O0.txt varity_cases.O3.txt [varity.<variant>.txt]
# Environment: MATHFN_N (results per random class, default 131072), MATHFN_DUMP=1 (also write mathfn.dump/ and mathfn_fast.dump/:
# every result, for compare.py's ulp counts), VARITY=0 (leave the Varity programs out), JOBS (Varity programs at once, default 4),
# VGPU_* pass through. The AMD runs need ROCm's runtime libraries on LD_LIBRARY_PATH after the shim.
# No file name carries the architecture, so two directories compare whatever they were built for:
#   tools/numerics/compare.py <out>/nvidia.t4.75.sass <out>/nvidia.t4.75.ptx
#   tools/numerics/compare.py <out>/nvidia.rtx3060.86.sass <card-out>/sm_86
set -uo pipefail
bin="$(cd "${1:?usage: $0 <bin-dir> <build-dir> <out-dir> <config> ...}" && pwd)"; shim="$(cd "${2:?}" && pwd)/shim"; mkdir -p "${3:?}"; out="$(cd "$3" && pwd)"; shift 3
here="$(cd "$(dirname "$0")" && pwd)"
n="${MATHFN_N:-131072}"
for config in "$@"; do
  IFS=: read -r gpu arch engine <<< "$config"
  vendor="${gpu%%/*}"; profile="${gpu##*/}"
  if [[ $vendor == nvidia ]]; then
    d="$bin/sm_$arch"; o="$out/nvidia.$profile.$arch.$engine"; vbin="$bin/varity-bin-sm_$arch"; [[ -d "$vbin" ]] || vbin="$bin/varity-bin"
    envs=(VGPU_QUIET=1 VGPU_GPU="$gpu" VGPU_SASS=$([[ $engine == sass ]] && echo 1 || echo 0) LD_LIBRARY_PATH="$shim")
  else
    d="$bin/$arch"; o="$out/amd.$profile.$arch"; vbin="$bin/varity-bin-$arch"
    envs=(VGPU_QUIET=1 VGPU_GPU="$gpu" LD_LIBRARY_PATH="$shim:${LD_LIBRARY_PATH:-}")
  fi
  [[ -d "$d" ]] || { echo "$config: no programs in $d"; continue; }
  mkdir -p "$o"
  for p in mathfn mathfn_fast; do
    [[ -x "$d/$p" ]] || continue
    dump=(); [[ "${MATHFN_DUMP:-0}" == 1 ]] && { mkdir -p "$o/$p.dump"; dump=(--dump "$o/$p.dump"); }
    env "${envs[@]}" timeout "${MATHFN_TIMEOUT:-7200}" "$d/$p" --n "$n" --published "$here/published.tsv" "${dump[@]}" > "$o/$p.txt" 2> "$o/$p.err"
    echo "$config $p rc=$? $(grep -c ' hash=' "$o/$p.txt") lines"
  done
  for p in fp_examples fp_examples_nofma varity_cases.O0 varity_cases.O3; do
    [[ -x "$d/$p" ]] || continue
    env "${envs[@]}" timeout 600 "$d/$p" > "$o/$p.txt" 2> "$o/$p.err"
    echo "$config $p rc=$?"
  done
  if [[ "${VARITY:-1}" != 0 && -d "$vbin" && -d "$bin/varity-gen" ]]; then
    "$here/varity/run.sh" "$bin/varity-gen" "$vbin" "$o" "${envs[@]}" | sed "s|^|$config |"
  fi
  find "$o" -name '*.err' -empty -delete
done
