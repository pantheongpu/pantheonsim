#!/usr/bin/env bash
# The body of the card-host-gpu job (.github/workflows/card-host-gpu.yml), on a machine with an NVIDIA card: build the simulator, run the host-GPU tests that
# need a card, and check the card against the simulator. Also runnable by hand on any card machine (a profile for the card must exist under nvidia/profiles).
#
#   tools/card-ci/run-card-ci.sh [build-dir]
#   env: CARD_CI_STRIDE (default 257)  every Nth float pattern in the SFU/division exhaustive checks; 1 = all 2^32 (about 10 minutes on an A10G)
#        CARD_CI_JOBS   (default 4)    compile processes
#
# What it checks, and what a failure means:
#   1. e2e_host_gpu runs and does NOT skip (on a card machine a skip is a failure: the test must have run).
#   2. unit tests host_gpu_logic, host_gpu_ref and card_gate pass.
#   3. e2e_host_gpu_stub passes (the start-up logic, card or not).
#   4. hostgpu_sfu_exhaustive div: the card's IEEE division equals the correctly rounded quotient (and so the simulator's) for every sampled divisor and eight
#      dividends: the evidence for the division idiom (VGPU_HOST_GPU_DIV). Any difference fails the job.
#   5. hostgpu_sfu_exhaustive rcp/rsq/sqrt/ex2/lg2: REPORTED, never failing: the card's approximations are expected to differ from the simulator's (that is why
#      MUFU is not in the exact set). The lines are in the log so that a driver or architecture that changes them is seen.
#   6. the verify run of the e2e program (VGPU_HOST_GPU_VERIFY=1) found no kernel of the exact set that differs (the machine-readable line
#      "host_gpu_verify: in_set_identical=N in_set_different=0 ..."), and ran at least two on the card.
# Negative control: with HOSTGPU_SFU_MUTATE=1 the script runs ONLY step 4 with the card side nudged by one ulp at a few inputs, and exits 0 exactly when the
# check notices (differ>0), 1 when it does not: the check is shown able to fail on this machine. The workflow's `mutation` input runs it.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
build="${1:-$root/build}"
stride="${CARD_CI_STRIDE:-257}"
jobs="${CARD_CI_JOBS:-4}"
fails=0
say() { echo "== $*"; }
command -v nvidia-smi > /dev/null && nvidia-smi -L || { echo "FAIL: no card on this machine"; exit 1; }

if [[ -n "${HOSTGPU_SFU_MUTATE:-}" ]]; then
  work="$(mktemp -d "${TMPDIR:-/tmp}/card-ci.XXXXXX")"
  trap 'rm -rf "$work"' EXIT
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
  nvcc_bin="${NVCC:-$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | tail -1)}"
  "$nvcc_bin" -std=c++20 -O2 -arch="sm_$cc" -cudart shared -w -I "$root/include" "$root/tools/probes/hostgpu_sfu_exhaustive.cu" -o "$work/sfu" || { echo "FAIL: compile of the probe"; exit 1; }
  if "$work/sfu" div "$stride"; then echo "FAIL: the nudged card result was not noticed"; exit 1; fi
  echo "PASS: the nudged card result was noticed"
  exit 0
fi

say "build"
mkdir -p "$build"
cmake -S "$root" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVGPU_LTO=OFF > "$build.cfg.log" 2>&1 || { tail -20 "$build.cfg.log"; exit 1; }
nice cmake --build "$build" -j"$jobs" --target vgpucudart test_host_gpu_logic test_host_gpu_ref test_card_gate > "$build.build.log" 2>&1 || { tail -30 "$build.build.log"; exit 1; }

say "unit tests"
for t in test_host_gpu_logic test_host_gpu_ref test_card_gate; do "$build/$t" > "$build.$t.log" 2>&1 || { echo "FAIL: $t"; tail -5 "$build.$t.log"; fails=1; }; done

say "e2e (card present: a skip is a failure)"
for t in e2e_host_gpu e2e_host_gpu_stub; do
  out="$(cd "$build" && VGPU_BUILD_DIR="$build" ctest -R "^$t\$" --output-on-failure 2>&1)"
  echo "$out" | tail -4
  echo "$out" | grep -q "100% tests passed" || { echo "FAIL: $t did not pass"; fails=1; }
  echo "$out" | grep -q "SKIP" && { echo "FAIL: $t skipped on a card machine"; fails=1; }
done

say "verify run, machine-readable summary"
name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
id="$(echo "$name" | tr 'A-Z' 'a-z' | sed -e 's/^nvidia //' -e 's/^geforce //' -e 's/ //g')"
work="$(mktemp -d "${TMPDIR:-/tmp}/card-ci.XXXXXX")"
trap 'rm -rf "$work"' EXIT
cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
nvcc_bin="${NVCC:-$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | tail -1)}"
"$nvcc_bin" -std=c++17 -arch="sm_$cc" -cudart shared -w "$root/nvidia/tests/e2e/host_gpu_exact.cu" "$root/nvidia/tests/e2e/host_gpu_exact_mod.cu" -o "$work/exact" || { echo "FAIL: compile"; fails=1; }
if [[ -x "$work/exact" ]]; then
  VGPU_GPU="nvidia/$id" VGPU_THREADS=4 LD_LIBRARY_PATH="$build/shim" VGPU_HOST_GPU=exact VGPU_HOST_GPU_VERIFY=1 "$work/exact" > "$work/v.out" 2> "$work/v.err"
  line="$(grep '^host_gpu_verify:' "$work/v.err" | tail -1)"
  echo "$line"
  same="$(echo "$line" | sed -n 's/.*in_set_identical=\([0-9]*\).*/\1/p')"
  diff="$(echo "$line" | sed -n 's/.*in_set_different=\([0-9]*\).*/\1/p')"
  (( ${same:-0} >= 2 )) || { echo "FAIL: fewer than 2 kernels of the exact set compared"; fails=1; }
  (( ${diff:-1} == 0 )) || { echo "FAIL: a kernel of the exact set differs on the card"; fails=1; }
fi

say "SFU and division exhaustive checks (stride $stride)"
"$nvcc_bin" -std=c++20 -O2 -arch="sm_$cc" -cudart shared -w -I "$root/include" "$root/tools/probes/hostgpu_sfu_exhaustive.cu" -o "$work/sfu" || { echo "FAIL: compile of the probe"; fails=1; }
if [[ -x "$work/sfu" ]]; then
  "$work/sfu" div "$stride" | tee "$work/div.txt"
  [[ "${PIPESTATUS[0]}" == 0 ]] || { echo "FAIL: the card's division differs from the correctly rounded quotient"; fails=1; }
  for op in rcp rsq sqrt ex2 lg2; do "$work/sfu" "$op" "$stride" | grep '^RESULT' || true; done   # reported, not judged
fi
(( fails == 0 )) && { echo "PASS: card CI on $name"; exit 0; }
echo "FAIL: card CI on $name"
exit 1
