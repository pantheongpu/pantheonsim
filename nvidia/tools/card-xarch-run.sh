#!/usr/bin/env bash
# Runs on a rented NVIDIA card (one GPU): every probe the Hopper and Blackwell cross-architecture issues (#733, #736, #737, #752, #753, #742's
# follow-ups) ask of a card, built for the card's own architecture with the CUDA 13.0 toolkit, one session, nothing else installed or kept.
# It leaves out/ (default ~/out) with one file a probe; each probe that is deterministic runs twice and says whether the runs agree.
#
#   nvidia/tools/card-xarch-run.sh <repo dir> [out dir]
#
# The repo dir holds tools/tcpapers, nvidia/tests/e2e, nvidia/tests/pytorch and nvidia/tests/data/tcpapers (a tarball of those is enough).
# A failing probe is logged and the run goes on: the card is billed by the hour, so one broken probe must not cost the others.
set -uo pipefail
repo="$(cd "${1:?usage: $0 <repo dir> [out dir]}" && pwd)"
out="${2:-$HOME/out}"; mkdir -p "$out"
exec > >(tee -a "$out/run.log") 2>&1
step() { echo; echo "=== $(date -u +%H:%M:%S) $*"; }
SUDO=""; [[ "$(id -u)" != 0 ]] && SUDO="sudo"
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')"
name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "card: $name, compute capability $cap"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,pci.device_id,clocks.max.sm,ecc.mode.current --format=csv > "$out/gpu.csv"
nvidia-smi -q > "$out/nvidia-smi-q.txt" 2>&1
uname -a > "$out/system.txt"

# 1. PyTorch in the background (the largest download), the toolkit meanwhile.
step "torch install (background)"
( python3 -m venv "$HOME/torch" && "$HOME/torch/bin/pip" install -q numpy "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple > "$out/torch-install.log" 2>&1; echo "torch install exit $?" >> "$out/torch-install.log" ) &
torch_pid=$!

step "CUDA 13.0 toolkit"
. /etc/os-release
if ! apt-cache policy cuda-nvcc-13-0 2>/dev/null | grep -q 'Candidate: [0-9]'; then
  $SUDO apt-get update -qq >/dev/null 2>&1
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${ID}${VERSION_ID//./}/$(uname -m)/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb && $SUDO dpkg -i /tmp/cuda-keyring.deb >/dev/null 2>&1
fi
$SUDO apt-get update -qq >/dev/null 2>&1
$SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-nvcc-13-0 cuda-cudart-dev-13-0 libcublas-dev-13-0 cuda-cuobjdump-13-0 cuda-nvdisasm-13-0 cuda-crt-13-0 cuda-cccl-13-0 > "$out/apt.log" 2>&1 \
  || { echo "toolkit install failed"; tail -5 "$out/apt.log"; }
export PATH="/usr/local/cuda-13.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:/usr/local/cuda-13.0/targets/$(uname -m)-linux/lib:${LD_LIBRARY_PATH:-}"
nvcc --version | tail -2 | tee "$out/nvcc.txt"
b() { nvcc -std=c++17 -O1 -cudart shared -w "$@"; }
arch="sm_$cap"; archa="$arch"
(( cap == 90 )) && archa=sm_90a
(( cap >= 120 )) && archa="sm_${cap}a"
bin="$out/bin"; mkdir -p "$bin"

twice() {   # <label> <cmd...>: runs a command twice, keeps the first output as out/<label>.txt, says whether both agree
  local label="$1"; shift
  "$@" > "$out/$label.txt" 2> "$out/$label.err"; local rc=$?
  "$@" > "$out/$label.2.txt" 2> "$out/$label.2.err"
  if cmp -s "$out/$label.txt" "$out/$label.2.txt"; then echo "$label: exit $rc, two runs identical"; rm -f "$out/$label.2.txt" "$out/$label.2.err"
  else echo "$label: exit $rc, TWO RUNS DIFFER"; fi
}

# 2. #752 and #753 and #750: the one-card questions of the e2e programs.
step "cg_multi_warp_tiles, texture_grad, cluster_occupancy, bulk_copy (-arch=$arch)"
e2e="$repo/nvidia/tests/e2e"
b -arch="$arch" -O3 "$e2e/cg_multi_warp_tiles.cu" -o "$bin/cg_multi_warp_tiles" && twice cg_multi_warp_tiles "$bin/cg_multi_warp_tiles"
b -arch="$arch" "$e2e/texture_grad.cu" -o "$bin/texture_grad" && { twice texture_grad.check "$bin/texture_grad"; twice texture_grad.print "$bin/texture_grad" --print; }
b -arch="$arch" "$e2e/cluster_occupancy.cu" -o "$bin/cluster_occupancy" -lcuda && twice cluster_occupancy "$bin/cluster_occupancy"
b -arch="$arch" "$e2e/bulk_copy.cu" -o "$bin/bulk_copy" && { twice bulk_copy "$bin/bulk_copy"; cuobjdump -sass "$bin/bulk_copy" > "$out/bulk_copy.sass.txt" 2>&1; }
cuobjdump -sass "$bin/cg_multi_warp_tiles" > "$out/cg_multi_warp_tiles.sass.txt" 2>&1
cuobjdump -sass "$bin/texture_grad" > "$out/texture_grad.sass.txt" 2>&1

# 3. #733: cuBLAS tensor-core GEMMs, the pinned cases and the K sweep.
step "cublas_tc_gemm"
b -arch="$arch" "$e2e/cublas_tc_gemm.cu" -o "$bin/cublas_tc_gemm" -lcublas \
  && { twice cublas_tc_gemm "$bin/cublas_tc_gemm"; twice cublas_tc_gemm.sweep "$bin/cublas_tc_gemm" --sweep; }
ldd "$bin/cublas_tc_gemm" | grep cublas > "$out/cublas-lib.txt"
# which kernels cuBLAS runs (the instruction a wgmma or mma.sync kernel is named by)
if command -v nsys >/dev/null 2>&1; then nsys profile -o "$out/cublas" --stats=true "$bin/cublas_tc_gemm" > "$out/cublas.nsys.txt" 2>&1; fi

# 4. #742's follow-ups and the numerics probes on the card.
step "tcpapers: fprev, tc_features, valpey, khattak mma.sync vectors"
tool="$repo/tools/tcpapers"; data="$repo/nvidia/tests/data/tcpapers"
b -arch="$arch" -DNO_CUBLAS "$tool/fprev_probe.cu" -o "$bin/fprev_probe" && twice fprev "$bin/fprev_probe"
b -arch="$arch" "$tool/tc_features.cu" -o "$bin/tc_features" && twice tc_features "$bin/tc_features"
b -arch="$arch" "$tool/khattak_vectors.cu" -o "$bin/khattak_vectors"
step "khattak vectors: fetch the full sets"
vec="$HOME/khattak"
( bash "$tool/khattak-fetch.sh" "$vec" > "$out/khattak-fetch.log" 2>&1 ) || echo "fetch failed (the 24-vector samples are used)"
vdir="$vec/vectors"; [[ -d "$vdir" ]] || vdir="$data"
echo "vector dir: $vdir"; ls "$vdir" | head -40 > "$out/vector-files.txt"
: > "$out/khattak_mma.txt"
for f in "$vdir"/*.txt; do
  base="$(basename "$f" .txt)"; IFS=. read -r card fmt o <<< "$base"
  case "$fmt" in fp16|bf16|tf32) ;; *) continue ;; esac
  case "$o" in fp32|fp16) ;; *) continue ;; esac
  k=$(( ($(head -1 "$f" | wc -w) - 2) / 2 ))
  echo "$base $("$bin/khattak_vectors" "$fmt" "$k" "$o" "$f" --show 0 2>&1 | tail -1)" >> "$out/khattak_mma.txt"
done
cat "$out/khattak_mma.txt"

if (( cap == 90 )); then
  step "wgmma vectors (sm_90a): the Khattak sets through a real wgmma"
  b -gencode "arch=compute_90a,code=sm_90a" "$tool/khattak_wgmma_vectors.cu" -o "$bin/khattak_wgmma_vectors" || echo "wgmma build failed"
  : > "$out/wgmma_counts.txt"
  mkdir -p "$out/wgmma"
  for f in "$vdir"/*.txt; do
    base="$(basename "$f" .txt)"; IFS=. read -r card fmt o <<< "$base"
    case "$fmt.$o" in fp16.fp32) mode=f16;; bf16.fp32) mode=bf16;; tf32.fp32) mode=tf32;; fp16.fp16) mode=f16h;; *) continue;; esac
    echo "$base as $mode: $("$bin/khattak_wgmma_vectors" $mode "$f" --show 0 2>&1 | tail -1)" | tee -a "$out/wgmma_counts.txt"
  done
  # the card's wgmma results on the H100 inputs: every 16th vector of each full set (and the 24-vector samples whole), a file a test can replay
  for pair in f16:fp16.fp32 bf16:bf16.fp32 tf32:tf32.fp32 f16h:fp16.fp16; do
    mode="${pair%%:*}"; sfx="${pair#*:}"
    src="$vdir/H100.$sfx.txt"; [[ -f "$src" ]] || continue
    awk 'NR % 16 == 1' "$src" > "$out/wgmma/in.$mode.txt"
    "$bin/khattak_wgmma_vectors" $mode "$out/wgmma/in.$mode.txt" --dump > "$out/wgmma/H100.wgmma.$mode.txt" 2> "$out/wgmma/H100.wgmma.$mode.err"
    "$bin/khattak_wgmma_vectors" $mode "$out/wgmma/in.$mode.txt" --dump > "$out/wgmma/again.$mode.txt" 2>/dev/null
    cmp -s "$out/wgmma/H100.wgmma.$mode.txt" "$out/wgmma/again.$mode.txt" && echo "wgmma $mode dump: two runs identical" || echo "wgmma $mode dump: TWO RUNS DIFFER"
    rm -f "$out/wgmma/again.$mode.txt" "$out/wgmma/in.$mode.txt"
  done
fi

step "m8n8k4 and mma.sp on the card (the T4's and A10G's inputs)"
if (( cap >= 80 )); then
  b -arch="$arch" "$tool/m8n8k4_vectors.cu" -o "$bin/m8n8k4_vectors" && for o in fp32 fp16; do
    "$bin/m8n8k4_vectors" $o "$data/m8n8k4.a10g.$o.txt" --dump > "$out/m8n8k4.card.$o.txt" 2> "$out/m8n8k4.card.$o.err"
    "$bin/m8n8k4_vectors" $o "$data/m8n8k4.a10g.$o.txt" --dump > "$out/m8n8k4.card.$o.2.txt" 2>/dev/null
    cmp -s "$out/m8n8k4.card.$o.txt" "$out/m8n8k4.card.$o.2.txt" && echo "m8n8k4 $o: two runs identical" || echo "m8n8k4 $o: TWO RUNS DIFFER"; rm -f "$out/m8n8k4.card.$o.2.txt"
    # a card's own file differs from the A10G's by the d column only: how many results differ
    echo "m8n8k4 $o vs the A10G: $(paste -d' ' <(awk '{print $NF}' "$data/m8n8k4.a10g.$o.txt") <(awk '{print $NF}' "$out/m8n8k4.card.$o.txt") | awk '$1!=$2{n++} END{print n+0, "of", NR, "results differ"}')"
  done
  b -arch="$arch" "$tool/sparse_vectors.cu" -o "$bin/sparse_vectors" && mkdir -p "$out/sparse" && for f in "$data"/sparse/a10g.*.txt; do
    mode="$(basename "$f" .txt)"; mode="${mode#a10g.}"
    "$bin/sparse_vectors" "$mode" "$f" --dump > "$out/sparse/card.$mode.txt" 2> "$out/sparse/card.$mode.err"
    "$bin/sparse_vectors" "$mode" "$f" --dump > "$out/sparse/card.$mode.2.txt" 2>/dev/null
    cmp -s "$out/sparse/card.$mode.txt" "$out/sparse/card.$mode.2.txt" && same=identical || same=DIFFER; rm -f "$out/sparse/card.$mode.2.txt"
    echo "sparse $mode: two runs $same; vs the A10G: $(paste -d' ' <(awk '{print $NF}' "$f") <(awk '{print $NF}' "$out/sparse/card.$mode.txt") | awk '$1!=$2{n++} END{print n+0, "of", NR, "results differ"}')"
  done
fi

# 5. #751 and #758 (sm_120): the SASS of a cub scan and the multicast bulk copy, with this toolkit, and that they run.
if (( cap >= 120 )); then
  step "cub scan on sm_120 (SASS for #758)"
  cat > "$out/scan.cu" <<'EOF'
#include <cub/device/device_scan.cuh>
#include <cstdio>
#include <vector>
int main() {
  int const n = 1 << 20;
  std::vector<unsigned> h(n, 1);
  unsigned *in = nullptr, *out = nullptr;
  cudaMalloc(&in, n * sizeof(unsigned)); cudaMalloc(&out, n * sizeof(unsigned));
  cudaMemcpy(in, h.data(), n * sizeof(unsigned), cudaMemcpyHostToDevice);
  void* tmp = nullptr; size_t bytes = 0;
  cub::DeviceScan::ExclusiveSum(tmp, bytes, in, out, n);
  cudaMalloc(&tmp, bytes);
  cub::DeviceScan::ExclusiveSum(tmp, bytes, in, out, n);
  cudaError_t const e = cudaDeviceSynchronize();
  unsigned last = 0; cudaMemcpy(&last, out + n - 1, sizeof(unsigned), cudaMemcpyDeviceToHost);
  std::printf("%s, last %u (expect %d)\n", cudaGetErrorString(e), last, n - 1);
  cudaFree(in); cudaFree(out); cudaFree(tmp);
  return e == cudaSuccess && last == unsigned(n - 1) ? 0 : 1;
}
EOF
  b -arch="$arch" "$out/scan.cu" -o "$bin/scan" && { "$bin/scan" > "$out/scan.txt" 2>&1; cat "$out/scan.txt"; cuobjdump -sass "$bin/scan" > "$out/scan.sass.txt" 2>&1; grep -c "BAR.SYNC" "$out/scan.sass.txt"; grep "BAR.SYNC" "$out/scan.sass.txt" | sort | uniq -c | head -20 > "$out/scan.barsync.txt"; }
fi

# 6. #736 and #737: the PyTorch application smoke recorded on the card, and which SDPA kernels run.
step "torch (wait for the install)"
wait "$torch_pid"; tail -2 "$out/torch-install.log"
py="$HOME/torch/bin/python"
if "$py" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())' > "$out/torch-version.txt" 2>&1; then
  cat "$out/torch-version.txt"
  cd "$repo/nvidia/tests/pytorch" || true
  slug="$(echo "$name" | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-')"
  step "app_smoke --write (each workload twice on the card)"
  "$py" app_smoke.py --write "$out/pytorch_app_smoke.card.ref" > "$out/app_smoke.write.log" 2>&1; echo "app_smoke --write exit $?"; tail -25 "$out/app_smoke.write.log"
  step "SDPA backends"
  cp "$repo/nvidia/tests/pytorch/sdpa_backend_probe.py" "$out/sdpa_backend_probe.py" 2>/dev/null
  "$py" "$repo/nvidia/tests/pytorch/sdpa_backend_probe.py" > "$out/sdpa_backends.txt" 2>&1; echo "sdpa probe exit $?"; head -60 "$out/sdpa_backends.txt"
else
  echo "torch unusable"; cat "$out/torch-version.txt" | tail -5
fi
step "done"
ls -la "$out" | head -80
