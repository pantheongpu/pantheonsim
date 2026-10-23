#!/bin/bash
# on the rig host (nvcc 13.2): builds the throwaway probes into ~/results/probes
set -e; mkdir -p ~/results/probes; cd "$(dirname "$0")"
nvcc -arch=sm_86 -cudart shared probe_props.cu -o ~/results/probes/probe_props
nvcc -arch=sm_86 -O2 -cudart shared e5_launch.cu -o ~/results/probes/e5_launch
nvcc -arch=sm_86 probe_va.cu -o ~/results/probes/probe_va -lcuda
