#!/usr/bin/env bash
# What vLLM for ROCm needs on a machine without an AMD GPU, for the simulator
# to run it: ROCm 7.2.3's libraries from AMD's apt repository (vLLM's
# PyTorch is linked against /opt/rocm-7.2.3/lib) and OpenMPI, then vLLM's
# wheels in a venv. The HIP, HSA, ROCm SMI and AMD SMI libraries that are
# used are the simulator's, put in front on the library path by
# run_vllm_amd.sh.
#
#   amd/tests/vllm/install.sh VENV
set -euo pipefail
venv=$1
vllm_version=0.30.0+rocm723
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends wget gnupg ca-certificates python3-venv libopenmpi3t64 \
  libnuma1 libdrm2 libdrm-amdgpu1 libelf1
sudo mkdir -p /etc/apt/keyrings
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg >/dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.3 noble main" |
  sudo tee /etc/apt/sources.list.d/rocm723.list >/dev/null
printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 600\n' | sudo tee /etc/apt/preferences.d/rocm-pin >/dev/null
sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends hip-runtime-amd hipblas hipfft hiprand hipsparse hipsolver \
  rocsolver hipsparselt rccl miopen-hip rocprofiler-sdk hipblaslt rocblas
ls -d /opt/rocm-7.2.3/lib >/dev/null
python3 -m venv "$venv"
"$venv/bin/pip" install -q uv
"$venv/bin/uv" pip install -q --python "$venv/bin/python" --index-strategy unsafe-best-match \
  "vllm==$vllm_version" --extra-index-url https://wheels.vllm.ai/rocm/
