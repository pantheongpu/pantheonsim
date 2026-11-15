#!/usr/bin/env bash
# Everything a real card is asked for the math-function numerics (docs/numerics/math-functions.md), in one run: builds the
# programs of this directory with the machine's own compiler and runs them on the card. Needs the card, its toolkit (an NVIDIA
# card: nvcc; an AMD card: hipcc) and this directory; no simulator build. Makes no cloud call and no network call unless the
# Varity programs are wanted and <out-dir>/varity-src is not there yet (then one git clone of github.com/LLNL/Varity; VARITY=0
# leaves them out; python3 is needed to generate them).
#
#   tools/numerics/card-run.sh <out-dir>
#
# What it leaves in <out-dir> (copy the whole directory back; about 300 MB with the dumps, 2 MB with MATHFN_DUMP=0):
#   gpu.txt                         the card, driver and toolkit
#   NVIDIA: sm_<own>/               the programs built for the card's own architecture: the card runs the SASS nvcc made, which is
#                                   what the simulator's SASS executor runs. The directory to compare.
#           sm_75/                  (a card later than Turing) the sm_75 build: the card cannot run that SASS, so its driver compiles
#                                   the build's PTX. Lines only; they say whether the driver's compiler gives what nvcc's does.
#   AMD:    native/                 hipcc's choice for the card
#   in each: mathfn.txt mathfn_fast.txt fp_examples.txt fp_examples_nofma.txt varity_cases.O0.txt varity_cases.O3.txt (line files);
#            in sm_<own> or native also mathfn.dump/ mathfn_fast.dump/ (every result) and varity.<O0|O3|O0_fm|O3_fm>.txt
#   rc.txt                          the exit status of every program
# Then, anywhere with python3 and a simulator run of the same programs (tools/numerics/run-sim.sh with MATHFN_DUMP=1 and, for the
# Varity lines, the programs of varity/build.sh nvidia <gen> <bin>/varity-bin-sm_<own> <own>):
#   tools/numerics/compare.py <sim-out>/nvidia.<profile>.<own>.sass <out-dir>/sm_<own> --truth
#
# Cards wanted (nothing here launches one):
#   RTX 3060 (local, sm_86), Tesla T4 (sm_75), A10G (sm_86), L4 (sm_89), H100 (sm_90); a Blackwell card (sm_100 / sm_120) if one is had
#   RX 6800 and RX 6700 XT (the local AMD box; gfx1030 / gfx1031)
# Environment: MATHFN_N (default 131072 results a random class), MATHFN_DUMP=0, VARITY=0, VARITY_PROGRAMS (default 150 of each
# precision), JOBS (compiles and Varity programs at once, default 4), NVCC / HIPCC (the compiler to use).
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="$(mkdir -p "${1:?usage: $0 <out-dir>}" && cd "$1" && pwd)"
n="${MATHFN_N:-131072}"
: > "$out/rc.txt"
if command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null 2>&1; then
  kind=nvidia
  nvcc_bin="${NVCC:-$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | tail -1)}"
  [[ -n "$nvcc_bin" ]] || { echo "no nvcc"; exit 1; }
  export PATH="$(dirname "$nvcc_bin"):$PATH"
  cuda_root="$(dirname "$(dirname "$(readlink -f "$nvcc_bin")")")"
  export LD_LIBRARY_PATH="$cuda_root/lib64:${LD_LIBRARY_PATH:-}"
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')"
  { nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1; "$nvcc_bin" --version | tail -2; date -u +%Y-%m-%d; } > "$out/gpu.txt"
  archs=("$cap"); (( cap != 75 )) && archs+=(75)
  common="sm_$cap"; dirs=(); for a in "${archs[@]}"; do dirs+=("sm_$a"); done
else
  kind=amd
  hipcc_bin="${HIPCC:-$(command -v hipcc || ls /opt/rocm*/bin/hipcc 2>/dev/null | tail -1)}"
  [[ -n "$hipcc_bin" ]] || { echo "no NVIDIA card with nvidia-smi and no hipcc"; exit 1; }
  export PATH="$(dirname "$hipcc_bin"):$PATH"
  rocm_root="$(dirname "$(dirname "$(readlink -f "$hipcc_bin")")")"
  export LD_LIBRARY_PATH="$rocm_root/lib:${LD_LIBRARY_PATH:-}"
  { "$rocm_root/bin/rocminfo" 2> /dev/null | grep -E 'Marketing Name|Name: +gfx' | sort -u; cat "$rocm_root/.info/version" 2> /dev/null; "$hipcc_bin" --version | head -2; date -u +%Y-%m-%d; } > "$out/gpu.txt"
  archs=(); common=native; dirs=(native)
fi
cat "$out/gpu.txt"
"$here/build.sh" "$kind" "$out/bin" "${archs[@]}" || echo "some programs did not build (see above); the rest run"
for d in "${dirs[@]}"; do
  mkdir -p "$out/$d"
  for p in mathfn mathfn_fast; do
    [[ -x "$out/bin/$d/$p" ]] || continue
    dump=(); [[ "$d" == "$common" && "${MATHFN_DUMP:-1}" == 1 ]] && { mkdir -p "$out/$d/$p.dump"; dump=(--dump "$out/$d/$p.dump"); }
    timeout 3600 "$out/bin/$d/$p" --n "$n" --published "$here/published.tsv" "${dump[@]}" > "$out/$d/$p.txt" 2> "$out/$d/$p.err"
    echo "$d $p rc=$? $(grep -c ' hash=' "$out/$d/$p.txt") lines" | tee -a "$out/rc.txt"
  done
  for p in fp_examples fp_examples_nofma varity_cases.O0 varity_cases.O3; do
    [[ -x "$out/bin/$d/$p" ]] || continue
    timeout 600 "$out/bin/$d/$p" > "$out/$d/$p.txt" 2> "$out/$d/$p.err"
    echo "$d $p rc=$?" | tee -a "$out/rc.txt"
  done
  find "$out/$d" -name '*.err' -empty -delete
done
if [[ "${VARITY:-1}" != 0 ]]; then
  if command -v python3 > /dev/null && "$here/varity/fetch.sh" "$out/varity-src" > /dev/null; then
    python3 "$here/varity/gen.py" "$out/varity-src" "$out/varity-gen" --programs "${VARITY_PROGRAMS:-150}"
    if [[ $kind == nvidia ]]; then "$here/varity/build.sh" nvidia "$out/varity-gen" "$out/varity-bin" "$cap"; else "$here/varity/build.sh" amd "$out/varity-gen" "$out/varity-bin"; fi
    "$here/varity/run.sh" "$out/varity-gen" "$out/varity-bin" "$out/$common" | tee -a "$out/rc.txt"
  else
    echo "Varity not run (python3 or the clone of LLNL/Varity is missing)" | tee -a "$out/rc.txt"
  fi
fi
echo "done: $out"
