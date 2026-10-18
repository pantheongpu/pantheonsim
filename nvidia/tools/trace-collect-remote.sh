#!/usr/bin/env bash
# Runs on a rented GPU: builds each CUPTI e2e case (nvidia/tests/e2e/cupti_*.cu)
# for the card in the machine, runs it against NVIDIA's own libcupti, and leaves
#   out/<case>.txt   what the card printed
#   out/<case>.bin   the binary that printed it (-cudart shared), so the same bytes
#                    can be run on the simulator later with no CUDA toolkit
#   out/info.txt     device name, compute capability, driver and toolkit
# Cases that need two GPUs are skipped on a one-GPU machine and say so.
set -uo pipefail
src="${1:?usage: trace-collect-remote.sh <dir with the cupti_*.cu files> }"
out="$src/out"; mkdir -p "$out"
export PATH=$PATH:/usr/local/cuda/bin
# Toolkit. TRACE_CUDA=13-0 installs and uses that toolkit instead of the image's default, to tell
# a toolkit difference from a driver or platform one. With no nvcc on the machine (Lambda, the
# marketplaces) one is installed from NVIDIA's repo: 13.0 when the driver can run it (the version
# the simulator's expected files come from), else the newest the driver supports.
SUDO=""; [[ "$(id -u)" != 0 ]] && SUDO="sudo"
want="${TRACE_CUDA:-}"
if [[ -z "$want" ]] && ! command -v nvcc >/dev/null 2>&1; then
  read -r dmaj dmin < <(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1 \2/p' | head -1)
  if (( ${dmaj:-0} >= 13 )); then want=13-0; else want="${dmaj:-12}-${dmin:-8}"; fi
fi
if [[ -n "$want" ]]; then
  if ! apt-cache policy "cuda-nvcc-$want" 2>/dev/null | grep -q 'Candidate: [0-9]'; then
    . /etc/os-release; arch="$(uname -m)"; [[ "$arch" == aarch64 ]] && arch=sbsa
    $SUDO apt-get update -qq >/dev/null 2>&1; $SUDO apt-get install -y -qq wget ca-certificates >/dev/null 2>&1
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${ID}${VERSION_ID//./}/$arch/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb \
      && $SUDO dpkg -i /tmp/cuda-keyring.deb >/dev/null 2>&1
  fi
  $SUDO apt-get update -qq >/dev/null 2>&1
  $SUDO apt-get install -y -qq "cuda-nvcc-$want" "cuda-cudart-dev-$want" "cuda-cupti-dev-$want" "cuda-nvtx-$want" >"$out/apt-pin.log" 2>&1 \
    || { echo "cannot install toolkit $want" > "$out/trace.txt"; exit 1; }
  export PATH="/usr/local/cuda-${want/-/.}/bin:$PATH"
fi
cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
find_cupti() {
  # The header and library must come from the toolkit nvcc is (an AMI can carry
  # several): <root>/extras/CUPTI/{include,lib64}. Nsight Compute has a private
  # libcupti that is not the one to link.
  inc=""; libs=""
  for r in "$cuda_root" /usr/local/cuda; do
    if [[ -f "$r/extras/CUPTI/include/cupti.h" && -d "$r/extras/CUPTI/lib64" ]]; then
      inc="$r/extras/CUPTI/include"; libs="$r/extras/CUPTI/lib64"; return
    fi
  done
  for r in "$cuda_root/include" /usr/include; do
    [[ -f "$r/cupti.h" ]] && inc="$r" && break
  done
  for r in "$cuda_root/lib64" "$cuda_root"/targets/*/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu; do
    compgen -G "$r/libcupti.so*" >/dev/null && libs="$r" && break
  done
}
find_cupti
if [[ -z "$libs" || -z "$inc" ]]; then
  # The base image ships the toolkit without CUPTI: add it from NVIDIA's repo.
  ver="$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1-\2/p')"
  $SUDO apt-get update -qq >/dev/null 2>&1
  $SUDO apt-get install -y -qq "cuda-cupti-dev-$ver" "cuda-nvtx-$ver" >"$out/apt.log" 2>&1 || $SUDO apt-get install -y -qq libcupti-dev >>"$out/apt.log" 2>&1
  find_cupti
fi
stubs="$libs/stubs"; [[ -e "$stubs/libcuda.so" ]] || stubs="$(dirname "$(find "$cuda_root" -name libcuda.so -path '*stubs*' | head -1)")"
arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)"
ngpu="$(nvidia-smi -L | grep -c GPU)"
{
  nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,uuid --format=csv
  nvcc --version | tail -2
  echo "ngpu=$ngpu arch=sm_$arch libs=$libs inc=$inc"
} > "$out/info.txt"
cubin="$out/resource_module.cubin"
nvcc -cubin -arch=sm_$arch -Wno-deprecated-gpu-targets "$src/cupti_resource_module.cu" -o "$cubin" 2>"$out/resource_module.err" || cubin=""
for f in "$src"/cupti_*.cu; do
  c="$(basename "$f" .cu)"; c="${c#cupti_}"
  [[ "$c" == resource_module || "$c" == activity ]] && continue
  [[ -n "${TRACE_ONLY:-}" && " $TRACE_ONLY " != *" $c "* ]] && continue
  if [[ "$c" == misc || "$c" == peer ]] && (( ngpu < 2 )); then echo "SKIP: needs two GPUs" > "$out/$c.txt"; continue; fi
  if [[ "$c" == trace_driver ]]; then
    # A driver-API program, built as the CI runner builds it (run_cupti_trace_driver.sh): no CUDA
    # runtime in the process, since loading one changes what initializes the driver when.
    nvcc -x c++ -std=c++17 -cudart none -Wno-deprecated-gpu-targets -I"$src" -I"$inc" "$f" -o "$out/$c.bin" -lcupti -L"$libs" -lcuda -L"$stubs" 2>"$out/$c.build.err" \
      || { echo "BUILD FAILED" > "$out/$c.txt"; continue; }
  else
  nvcc -std=c++17 -cudart shared -gencode arch=compute_$arch,code=sm_$arch -gencode arch=compute_$arch,code=compute_$arch \
         -Wno-deprecated-gpu-targets -I"$src" -I"$inc" "$f" -o "$out/$c.bin" -lcupti -L"$libs" -lcuda -L"$stubs" 2>"$out/$c.build.err" \
      || { echo "BUILD FAILED" > "$out/$c.txt"; continue; }
  fi
  CUPTI_TEST_CUBIN="$([[ $c == resource ]] && echo "$cubin")" NVTX_INJECTION64_PATH="$(ls "$libs"/libcupti.so.[0-9]* | head -1)" \
    LD_LIBRARY_PATH="$libs:$(dirname "$libs")/../lib64:$cuda_root/lib64:$cuda_root/targets/x86_64-linux/lib" timeout 300 "$out/$c.bin" > "$out/$c.txt" 2>"$out/$c.run.err" || echo "EXIT $?" >> "$out/$c.txt"
  echo "$c: $(wc -l < "$out/$c.txt") lines"
done
