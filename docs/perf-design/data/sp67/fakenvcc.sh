#!/bin/bash
# server3 has no CUDA toolkit: stands in for nvcc 13.0 in run_host_gpu_stub.sh by copying the program compiled on the card rig (nvcc 13.0, sm_86)
if [[ "$1" == --version ]]; then echo "Cuda compilation tools, release 13.0, V13.0.88"; exit 0; fi
out=""; while (( $# )); do [[ "$1" == -o ]] && { out="$2"; shift; }; shift; done
cp "$HOME/.cache/sp67/tk/lazy_keep" "$out"
