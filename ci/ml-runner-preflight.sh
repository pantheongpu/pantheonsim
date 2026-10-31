#!/usr/bin/env bash
# Fail early, and say why, when a job landed on a runner without the toolchain it needs.
#
#   ci/ml-runner-preflight.sh <check>...      checks: cuda13 rocm71 shim13 torch13
#
# The nightly workflows that use ML_RUNS_ON (external-suites, pytorch-sweep-nightly) need the
# pantheonsim.com host: CUDA 13.0, ROCm 7.1 under /opt/rocm-7.1.0 and a PyTorch for CUDA 13 under
# ~/.local/share/torch-cu13*. When another runner carries the same label (server1-sim, the
# server2 boxes) the symptoms are far from the cause: every NVIDIA program exits 127 in 0 s
# (libcudart.so.13 not found, because the shim was built as libcudart.so.12), CMake cannot find
# /opt/rocm-7.1.0/bin/hipcc, or the sweep "skips" and the job calls that a failure.
# Remedy: take the label off that runner (repository variable ML_RUNS_ON names the labels).
set -uo pipefail
bad=0
fail() { echo "::error::runner ${RUNNER_NAME:-$(hostname)}: $*"; bad=1; }
for check in "$@"; do
  case $check in
    cuda13)
      v=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p')
      [[ $v == 13.* ]] || fail "needs CUDA 13 (nvcc says '${v:-no nvcc}'); this runner is not the pantheonsim.com host. Remove the ML_RUNS_ON label from it." ;;
    rocm71)
      [[ -x ${ROCM_PATH:-/opt/rocm-7.1.0}/bin/hipcc ]] || fail "needs ROCm 7.1 at ${ROCM_PATH:-/opt/rocm-7.1.0} (no hipcc there); this runner is not the pantheonsim.com host. Remove the ML_RUNS_ON label from it." ;;
    shim13)
      [[ -e build/shim/libcudart.so.13 ]] || fail "build/shim has no libcudart.so.13 ($(ls build/shim 2>/dev/null | grep '^libcudart' | tr '\n' ' ')): the simulator was built against another CUDA major than the suites (13.0). Check the sim job's runner." ;;
    torch13)
      ok=""
      for c in "${VGPU_TORCH_CUDA_PYTHON:-}" $(ls -d "$HOME"/.local/share/torch-cu13*/bin/python 2>/dev/null); do
        [[ -n $c && -x $c ]] && "$c" -c 'import torch, sys; sys.exit(0 if (torch.version.cuda or "").startswith("13") else 1)' 2>/dev/null && { ok=$c; break; }
      done
      [[ -n $ok ]] || fail "no Python with PyTorch for CUDA 13 under ~/.local/share/torch-cu13*; this runner is not the pantheonsim.com host (the sweep would only skip)." ;;
    *) echo "unknown check $check"; exit 2 ;;
  esac
done
exit $bad
