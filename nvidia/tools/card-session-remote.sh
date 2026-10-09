#!/usr/bin/env bash
# Everything the AWS verification session runs on the rented GPU, in one pass.
# Runs ON the instance, from the repository tarball's root:
#
#   nvidia/tools/card-session-remote.sh <slug> <out-dir>
#
#   <slug>  the profile this card is (h100, rtx-pro-6000, l40s ...); it names the
#           expected files nvidia/tests/data/lowprec/<probe>.<slug>.txt
#
# Nothing here is hidden: every step logs to <out-dir>/<step>.log and the exit
# status of each program goes into <out-dir>/summary.txt, so a step that failed
# reads as failed. A failing step does not stop the session, because the
# instance costs money by the hour and every other step's data is still wanted.
# docs/aws-gpu-session.md (tools/aws-gpu-session.md) says how it is launched.
set -uo pipefail
slug="${1:?usage: card-session-remote.sh <slug> <out-dir>}"
out="$(mkdir -p "${2:?usage: card-session-remote.sh <slug> <out-dir>}" && cd "$2" && pwd)"
root="$(cd "$(dirname "$0")/../.." && pwd)"
e2e="$root/nvidia/tests/e2e"
cd "$root"
summary="$out/summary.txt"; : > "$summary"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$summary"; }

# ---- the toolkit: the newest /usr/local/cuda-13.x, else /usr/local/cuda
cuda=""
for d in ${VGPU_SESSION_CUDA:-} $(ls -d /usr/local/cuda-13.* 2>/dev/null | sort -V | tac) /usr/local/cuda; do
  [[ -x "$d/bin/nvcc" ]] && { cuda="$d"; break; }
done
[[ -n "$cuda" ]] || { say "FATAL no CUDA toolkit under /usr/local"; exit 1; }
export PATH="$cuda/bin:$PATH"
export LD_LIBRARY_PATH="$cuda/lib64:${LD_LIBRARY_PATH:-}"
cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. \r')"
case "$cc" in
  90) arch=sm_90a ;;     # Hopper
  100) arch=sm_100a ;;
  120) arch=sm_120a ;;   # Blackwell workstation / server (RTX PRO 6000)
  *) arch="sm_$cc" ;;
esac
say "slug=$slug compute_cap=$cc arch=$arch toolkit=$cuda"
{ nvidia-smi; nvcc --version; } > "$out/environment.txt" 2>&1
nvidia-smi -q > "$out/nvidia-smi-q.txt" 2>&1          # redact serial, PDI and UUID before committing
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,power.max_limit,clocks.max.sm,clocks.max.memory,pci.device_id,pci.sub_device_id --format=csv > "$out/gpu-query.csv" 2>&1

# step <name> <command...>: run, log, record the status
step() {
  local name="$1"; shift
  local t0=$SECONDS
  "$@" > "$out/$name.log" 2>&1
  local rc=$?
  say "$name rc=$rc $((SECONDS - t0))s: $(tail -n 1 "$out/$name.log" | cut -c1-120)"
  return 0
}

# ---- NVIDIA's other libraries from pip wheels (cuSPARSELt, cuDNN; cuBLAS 13.3 as the L4 transcripts used)
wheels="${VGPU_SESSION_WHEELS:-/tmp/wheels}"
if [[ ! -d "$wheels" && -z "${VGPU_SESSION_NO_PIP:-}" ]]; then
  python3 -m pip install -q --target "$wheels" nvidia-cusparselt-cu13 nvidia-cudnn-cu13 > "$out/pip.log" 2>&1 \
    || say "pip: cu13 wheels failed (see pip.log)"
  python3 -m pip install -q --target "$wheels/cublas133" "nvidia-cublas==13.3.*" >> "$out/pip.log" 2>&1 \
    || say "pip: no nvidia-cublas 13.3 wheel; cuBLASLt will be the toolkit's"
fi
findlib() { find "$wheels" -name "$1" 2>/dev/null | head -1; }
sparselt_so="$(findlib 'libcusparseLt.so*')"
cudnn_so="$(findlib 'libcudnn.so.9')"
cublaslt_so="$(find "$wheels/cublas133" -name 'libcublasLt.so.13' 2>/dev/null | head -1)"
sparselt_lib="${sparselt_so:+$(dirname "$sparselt_so")}"
sparselt_inc="$(dirname "$(findlib 'cusparseLt.h')" 2>/dev/null)"
cudnn_lib="${cudnn_so:+$(dirname "$cudnn_so")}"
cudnn_inc="$(dirname "$(findlib 'cudnn.h')" 2>/dev/null)"
cublas_lib="${cublaslt_so:+$(dirname "$cublaslt_so")}"
# A wheel ships libX.so.N without the unversioned name -lX needs.
for d in "$sparselt_lib" "$cudnn_lib" "$cublas_lib"; do
  [[ -n "$d" ]] || continue
  for f in "$d"/lib*.so.[0-9]*; do
    b="${f##*/}"; base="${b%%.so.*}"
    [[ -e "$d/$base.so" ]] || ln -s "$b" "$d/$base.so" 2>/dev/null
  done
done
{ echo "sparselt_lib=$sparselt_lib"; echo "sparselt_inc=$sparselt_inc"; echo "cudnn_lib=$cudnn_lib"
  echo "cudnn_inc=$cudnn_inc"; echo "cublas_lib=$cublas_lib"; } > "$out/wheel-dirs.txt"

# ---- 1. the narrow-precision transcripts (lowprec.md): cuBLASLt, cuSPARSELt, cvt, ptx120
# cuBLAS: the 13.3 wheel when it came, as the L4's transcripts are 13.3's (the version is the first
# line of each transcript; a different cuBLAS is a different answer, not a failure).
step lowprec-lt env ${cublas_lib:+LOWPREC_LIB_DIR="$cublas_lib"} "$e2e/run_lowprec.sh" lt --card "$slug" --update
step lowprec-sparselt env LOWPREC_LIB_DIR="$sparselt_lib" LOWPREC_INC_DIR="$root/nvidia/include" \
     "$e2e/run_lowprec.sh" sparselt --card "$slug" --update
step lowprec-cvt "$e2e/run_lowprec.sh" cvt --card "$slug" --update
[[ "$cc" == 120 ]] && step lowprec-ptx120 "$e2e/run_lowprec.sh" ptx120 --card "$slug" --update
mkdir -p "$out/transcripts"; cp "$root"/nvidia/tests/data/lowprec/*."$slug".txt "$out/transcripts/" 2>/dev/null

# ---- 2. the two FP8 attention tests, against NVIDIA's cuDNN (documented formulas; Hopper and later)
if [[ -n "$cudnn_lib" ]]; then
  fe="$("$e2e/fetch_cudnn_frontend.sh" 2>"$out/fetch-frontend.log")"
  step dnn-fp8-attention bash -c "nvcc -std=c++17 -cudart shared -arch=$arch -w -I'$cudnn_inc' '$e2e/dnn_fp8_attention.cu' -o /tmp/dnn_fp8_attention -L'$cudnn_lib' -lcudnn \
      && LD_LIBRARY_PATH='$cudnn_lib':\$LD_LIBRARY_PATH /tmp/dnn_fp8_attention"
  [[ -n "$fe" ]] && step dnn-fp8-attention-frontend bash -c "nvcc -std=c++17 -cudart shared -arch=$arch -w -I'$cudnn_inc' -I'$fe' '$e2e/dnn_fp8_attention_frontend.cpp' -o /tmp/dnn_fp8_attention_frontend -L'$cudnn_lib' -lcudnn -lcuda \
      && LD_LIBRARY_PATH='$cudnn_lib':\$LD_LIBRARY_PATH /tmp/dnn_fp8_attention_frontend"
else
  say "no cuDNN wheel: the FP8 attention tests were not run"
fi

# ---- 3. the self-checking programs written from documentation, run natively on the card.
# Each prints PASS on its last line (or SKIP); built for the card's own SASS, no simulator.
# The same programs run on the simulator in CI, so a native FAIL is a test that encodes a wrong assumption.
mkdir -p "$out/native"
cutlass=""
if grep -lq "cute/\|cutlass/" "$e2e"/wgmma_cute.cu; then
  cl="/tmp/cutlass-v4.8.0"
  if [[ ! -d "$cl/include" ]]; then
    mkdir -p "$cl" && curl -fsSL https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.8.0.tar.gz \
      | tar -xz -C "$cl" --strip-components=1 --wildcards '*/include' '*/tools/util/include' 2>/dev/null
  fi
  [[ -d "$cl/include" ]] && cutlass="$cl"
fi
native() {   # native <program> [extra nvcc args...]   (source .cu or .cpp in nvidia/tests/e2e)
  local p="$1"; shift
  local src="$e2e/$p.cu"; [[ -f "$src" ]] || src="$e2e/$p.cpp"
  local exe="/tmp/native_$p"
  local t0=$SECONDS
  { nvcc -std=c++17 -O1 -cudart shared -arch="$arch" -w --expt-relaxed-constexpr \
      ${cutlass:+-I$cutlass/include -I$cutlass/tools/util/include} -I"$e2e" -I"$root/nvidia/include" "$@" "$src" -o "$exe" -lcuda \
      && LD_LIBRARY_PATH="${sparselt_lib:+$sparselt_lib:}${cublas_lib:+$cublas_lib:}$LD_LIBRARY_PATH" timeout 600 "$exe"; } > "$out/native/$p.txt" 2>&1
  local rc=$?
  say "native $p rc=$rc $((SECONDS - t0))s: $(tail -n 1 "$out/native/$p.txt" | cut -c1-100)"
}
case "$cc" in
  90)  progs="dsmem_cluster cooperative_cluster setmaxnreg stmatrix vector_atomics wgmma_cute tma_gemm_cute tma_im2col tma_reduce_cute mma_forms mma_fragment_layout wmma_gemm wmma_types uldc_narrow" ;;
  120) progs="mma_blockscale narrow_cvt ldmatrix_forms dsmem_cluster cooperative_cluster stmatrix vector_atomics mma_forms mma_fragment_layout wmma_gemm wmma_types uldc_narrow" ;;
  89)  progs="mma_forms mma_fragment_layout wmma_gemm wmma_types uldc_narrow stmatrix" ;;
  *)   progs="" ;;
esac
progs="${VGPU_SESSION_PROGS:-$progs}"
for p in $progs; do native "$p"; done
# tcgen05_gemm needs sm_100a (a data-center Blackwell); not buildable here.
# library programs on the real libraries: cuBLASLt with and without block scales, cuSPARSELt paths
if [[ "$cc" == 90 || "$cc" == 120 ]]; then
  for p in lt_paths lt_epilogue_paths lt_blockscaled_paths; do native "$p" -lcublasLt ${cublas_lib:+-L$cublas_lib}; done
  [[ -n "$sparselt_lib" ]] && native sparselt_paths -I"$sparselt_inc" -L"$sparselt_lib" -lcusparseLt
fi

# ---- 4. the characterization: profile values, conformance references, attributes, counters
tools="$root/nvidia/tools"
nvcc -std=c++14 -Wno-deprecated-gpu-targets "$tools/characterize.cu" -o /tmp/characterize \
     -L"$cuda/lib64/stubs" -lcuda > "$out/characterize-build.log" 2>&1
{ /tmp/characterize 0 2>/dev/null | sed '/^telemetry:/,$d'; bash "$tools/characterize-telemetry.sh" 0; } > "$out/profile.yaml" 2> "$out/characterize.err"
say "characterize: $(grep -c . "$out/profile.yaml") lines of profile"
for t in ptx_semantics control_flow; do
  nvcc -std=c++14 -arch="sm_$cc" -Wno-deprecated-gpu-targets "$root/nvidia/tests/conformance/$t.cu" -o "/tmp/$t" 2>/dev/null \
    && "/tmp/$t" > "$out/$t.ref.txt" 2>&1
done
nvcc -std=c++14 --gpu-architecture="sm_$cc" -Wno-deprecated-gpu-targets "$e2e/device_attributes.cu" -o /tmp/device_attributes -lcuda \
  > "$out/device_attributes-build.log" 2>&1 && /tmp/device_attributes --dump > "$out/cuda_attributes_$slug.card.txt" 2>&1
say "device_attributes --dump: $(wc -l < "$out/cuda_attributes_$slug.card.txt" 2>/dev/null) lines"
(sudo -n ncu --query-metrics 2>/dev/null || ncu --query-metrics 2>/dev/null || true) > "$out/metrics.txt"
say "done"
