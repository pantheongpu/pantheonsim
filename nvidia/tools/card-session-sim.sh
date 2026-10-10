#!/usr/bin/env bash
# The simulator's side of an AWS card session, run ON the instance beside the card (tools/aws-gpu-session.md):
# builds the repository (CUDA 13.2, which the shims need for cuBLASLt's emulation types), then runs the programs the
# card ran, on the simulated GPU of the same profile, so that the two outputs can be compared in place.
#
#   nvidia/tools/card-session-sim.sh build                  build (about 8-15 minutes); start it when the instance is up
#   nvidia/tools/card-session-sim.sh run <slug> <out-dir>   after card-session-remote.sh has written the expected files
#
# `run` writes <out-dir>/sim-summary.txt (one line per program: rc and the last line) and every log beside it.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
cuda=""
for d in ${VGPU_SESSION_CUDA:-} /usr/local/cuda-13.2 /usr/local/cuda-13.0 /usr/local/cuda; do
  [[ -x "$d/bin/nvcc" ]] && { cuda="$d"; break; }
done
export PATH="$cuda/bin:$PATH"
case "${1:-}" in
  build)
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > build-configure.log 2>&1 \
      && cmake --build build -j"$(nproc)" -- -k > build.log 2>&1
    echo "build finished rc=$?" >> build.log
    ;;
  run)
    slug="${2:?slug}"; out="$(mkdir -p "${3:?out-dir}" && cd "$3" && pwd)"
    export VGPU_BUILD_DIR="$root/build"
    summary="$out/sim-summary.txt"; : > "$summary"
    sim() {   # sim <name> <command...>
      local n="$1"; shift
      local t0=$SECONDS
      "$@" > "$out/sim-$n.log" 2>&1
      local rc=$?
      echo "$n rc=$rc $((SECONDS - t0))s: $(tail -n 1 "$out/sim-$n.log" | cut -c1-140)" | tee -a "$summary"
    }
    e2e="$root/nvidia/tests/e2e"
    for p in lt sparselt cvt; do sim "lowprec-$p" "$e2e/run_lowprec.sh" "$p" "$slug"; done
    sim cluster-occupancy "$e2e/run_cluster_occupancy.sh"
    # the programs the card ran natively, on the simulator (their expected last line is PASS), by ctest name
    (cd build && ctest -R 'wgmma_cute|tma_|dsmem_cluster|cooperative_cluster|setmaxnreg|stmatrix|vector_atomics|mma_forms|mma_fragment|wmma_|uldc|lt_paths|lt_epilogue|lt_blockscaled|sparselt_paths|dnn_fp8|dnn_frontend_ops|dnn_attention$|dnn_sdpa_mask|cluster_occupancy' \
        --output-on-failure -j4 > "$out/sim-ctest.log" 2>&1; echo "ctest rc=$?" ) | tee -a "$summary"
    grep -E "Test +#|tests passed|Failed|\*\*\*" "$out/sim-ctest.log" | tee -a "$summary" > /dev/null
    # characterization: the simulator's own answers, compared with the card's profile.yaml
    if [[ -f "$out/profile.yaml" && -f "$root/nvidia/profiles/$slug.yaml" ]]; then
      python3 nvidia/tools/compare-profile.py "$out/profile.yaml" "nvidia/profiles/$slug.yaml" > "$out/profile-compare.txt" 2>&1
      echo "profile compare rc=$?: $(tail -n 1 "$out/profile-compare.txt")" | tee -a "$summary"
    fi
    for ref in ptx_semantics control_flow; do
      [[ -f "$out/$ref.ref.txt" ]] || continue
      mkdir -p "$out/refs" && cp "$out/$ref.ref.txt" "$out/refs/$slug.$ref.ref.txt"
    done
    [[ -d "$out/refs" ]] && sim verify-profile nvidia/tools/verify-profile.sh "nvidia/$slug" "$out/refs"
    ;;
  *) echo "usage: $0 build | run <slug> <out-dir>" >&2; exit 2 ;;
esac
