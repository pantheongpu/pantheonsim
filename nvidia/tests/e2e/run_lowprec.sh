#!/usr/bin/env bash
# The narrow-precision library probes (FP8, MXFP8, NVFP4; nvidia/docs/lowprec.md),
# each a program that prints one line per case, run against the shims on a
# simulated GPU and compared with what NVIDIA's own libraries printed for the
# same program on a real GPU.
#
#   run_lowprec.sh <probe> [<slug>...]            run on the simulated GPU of each
#                                                 expected file (or the slugs named)
#                                                 and compare
#   run_lowprec.sh <probe> --card <slug> [--update] [-- probe args]
#                                                 run against NVIDIA's libraries on the
#                                                 real GPU and compare with (or, with
#                                                 --update, rewrite) the same file
#
# Probes (nvidia/tests/e2e/lowprec_<probe>.{cu,cpp}), libraries they link:
#   lt        cuBLASLt: FP8 and block-scaled matmuls          cublasLt
#   sparselt  cuSPARSELt: FP8/FP4 structured-sparse matmuls   cusparseLt
#   cvt       the packed FP8 conversions of sm_89 and later (PTX)
#   ptx120    sm_120's block-scaled mma.sync forms, fp4/fp6/fp8 conversions and ldmatrix
#             expansions (mma_blockscale.cu, narrow_cvt.cu, ldmatrix_forms.cu), run on SASS
#             and on PTX: both must print what the card printed
#
# The expected files are nvidia/tests/data/lowprec/<probe>.<slug>.txt, where <slug>
# names the profile (nvidia/<slug>): l4, h100, rtx-pro-6000, rtx3060 ...
# Lines starting with '#' (the device, the library version) are not compared.
# VGPU_LOWPREC_EXTRA_LIBS names libraries to add to the link line; PROBE_DUMP=<dir>
# makes the probe write every case's inputs and outputs there.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
probe="${1:?usage: $0 <probe> [slug...|--card slug [--update]]}"
shift
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
data="$root/nvidia/tests/data/lowprec"
src="$root/nvidia/tests/e2e/lowprec_${probe}.cu"
[[ -f "$src" ]] || src="${src%.cu}.cpp"
[[ "$probe" == ptx120 ]] || [[ -f "$src" ]] || { echo "no such probe: $probe" >&2; exit 2; }
case "$probe" in
  lt) libs=(cublasLt) ;;
  sparselt) libs=(cusparseLt) ;;
  cvt) libs=() ;;
  ptx120) libs=() ;;
  *) echo "unknown probe $probe" >&2; exit 2 ;;
esac

card=0; update=0; slugs=(); pargs=()
while (( $# )); do
  case "$1" in
    --card) card=1 ;;
    --update) update=1 ;;
    --) shift; pargs=("$@"); break ;;
    *) slugs+=("$1") ;;
  esac
  shift
done

if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
if [[ "$probe" == ptx120 ]]; then
  # Three programs that print what sm_120a's tensor-core and conversion instructions compute.
  programs=(mma_blockscale narrow_cvt ldmatrix_forms)
  work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_lowprec_ptx120.XXXXXX")"
  trap 'rm -rf "$work"' EXIT
  arch="${LOWPREC_ARCH:-sm_120a}"
  build() { nvcc -std=c++17 -cudart shared -arch="$arch" -w -Wno-deprecated-gpu-targets "$root/nvidia/tests/e2e/$1.cu" -o "$work/$1" "${@:2}"; }
  if (( card )); then
    [[ ${#slugs[@]} == 1 ]] || { echo "--card takes exactly one slug" >&2; exit 2; }
    slug="${slugs[0]}"
    cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
    : > "$work/card.txt"
    echo "# device $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)" >> "$work/card.txt"
    for p in "${programs[@]}"; do
      build "$p"
      { echo "== $p"; LD_LIBRARY_PATH="${LOWPREC_LIB_DIR:-$cuda_root/lib64}" "$work/$p"; } >> "$work/card.txt"
    done
    if (( update )); then
      mkdir -p "$data"; cp "$work/card.txt" "$data/$probe.$slug.txt"; echo "wrote $data/$probe.$slug.txt"; exit 0
    fi
    diff <(grep -v '^#' "$data/$probe.$slug.txt") <(grep -v '^#' "$work/card.txt") && { echo "PASS (card $slug)"; exit 0; }
    echo "FAIL: the card disagrees with $data/$probe.$slug.txt"; exit 1
  fi
  if (( ${#slugs[@]} == 0 )); then
    shopt -s nullglob
    for f in "$data/$probe".*.txt; do b="${f##*/}"; b="${b#"$probe".}"; slugs+=("${b%.txt}"); done
    shopt -u nullglob
  fi
  (( ${#slugs[@]} )) || { echo "SKIP: no expected files for $probe"; exit 0; }
  shopt -s nullglob; carts=("$shim"/libcudart.so.[0-9]*); shopt -u nullglob
  (( ${#carts[@]} )) || { echo "SKIP: the CUDA runtime shim is not built"; exit 0; }
  if ! nvcc -arch=compute_120a -code=compute_120a -c -x cu /dev/null -o /dev/null 2>/dev/null; then
    echo "SKIP: this nvcc cannot target sm_120a (needs CUDA 12.8 or later)"; exit 0
  fi
  for p in "${programs[@]}"; do build "$p" $(shim_sanitizer_nvcc_flags "$shim"); done
  require_shim_libs "$shim" "$work/${programs[0]}" || exit 0
  fails=0
  for slug in "${slugs[@]}"; do
    for engine in sass ptx; do
      : > "$work/sim.txt"
      for p in "${programs[@]}"; do
        { echo "== $p"
          VGPU_QUIET=1 VGPU_GPU="nvidia/$slug" VGPU_SASS=$([[ $engine == sass ]] && echo 1 || echo 0) LD_LIBRARY_PATH="$shim" "$work/$p"; } >> "$work/sim.txt" 2>&1 || true
      done
      if diff <(grep -v '^#' "$data/$probe.$slug.txt") <(grep -v '^#' "$work/sim.txt") > "$work/d.txt"; then
        echo "ok   $slug ($engine)"
      else
        echo "FAIL $slug ($engine): differs from the card's transcript"; head -10 "$work/d.txt"; fails=1
      fi
    done
  done
  (( fails == 0 )) && echo "PASS" || { echo "FAIL"; exit 1; }
  exit 0
fi

out="${TMPDIR:-/tmp}/vgpu_lowprec_${probe}_$$"

# compare <expected> <actual>: the lines of <actual> for the cases <expected> lists (the lt and sparselt probes name
# each case in its first field; a case added since a transcript was taken is not compared on that profile), then diff.
compare() {
  case "$probe" in
    lt|sparselt)
      awk 'NR==FNR { if ($0 !~ /^#/) want[$1] = 1; next } /^#/ { next } ($1 in want)' "$1" "$2" > "$out.filtered" || return 2
      diff <(grep -v '^#' "$1") "$out.filtered" ;;
    *) diff <(grep -v '^#' "$1") <(grep -v '^#' "$2") ;;
  esac
}
trap 'rm -f "$out" "$out".*' EXIT

if (( card )); then
  [[ ${#slugs[@]} == 1 ]] || { echo "--card takes exactly one slug" >&2; exit 2; }
  slug="${slugs[0]}"
  # NVIDIA's libraries: the toolkit's own, or those LOWPREC_LIB_DIR names
  # (pip wheels), and the driver's. Cases are compared with the same file.
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  libdir="${LOWPREC_LIB_DIR:-$cuda_root/lib64}"
  inc=()
  [[ -n "${LOWPREC_INC_DIR:-}" ]] && inc=("-I$LOWPREC_INC_DIR")
  links=()
  for l in "${libs[@]}"; do links+=("-l$l"); done
  nvcc -std=c++17 -cudart shared -arch="${LOWPREC_ARCH:-native}" -Wno-deprecated-gpu-targets "${inc[@]}" \
       "$src" -o "$out" -L"$libdir" "${links[@]}" ${VGPU_LOWPREC_EXTRA_LIBS:-}
  LD_LIBRARY_PATH="$libdir:$cuda_root/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$out" "${pargs[@]}" > "$out.card.txt"
  if (( update )); then
    mkdir -p "$data"
    cp "$out.card.txt" "$data/$probe.$slug.txt"
    echo "wrote $data/$probe.$slug.txt"
    exit 0
  fi
  compare "$data/$probe.$slug.txt" "$out.card.txt" >"$out.diff.txt" && { echo "PASS (card $slug)"; exit 0; }
  head -20 "$out.diff.txt"; echo "FAIL: the card disagrees with $data/$probe.$slug.txt"; exit 1
fi

# The simulated GPU.
if (( ${#slugs[@]} == 0 )); then
  shopt -s nullglob
  for f in "$data/$probe".*.txt; do b="${f##*/}"; b="${b#"$probe".}"; slugs+=("${b%.txt}"); done
  shopt -u nullglob
fi
(( ${#slugs[@]} )) || { echo "SKIP: no expected files for $probe"; exit 0; }
links=(); cudart=shared
for lib in "${libs[@]}"; do
  shopt -s nullglob; have=("$shim"/lib"$lib".so.[0-9]*); shopt -u nullglob
  if (( ${#have[@]} == 0 )); then echo "SKIP: the $lib shim is not built"; exit 0; fi
  [[ "$lib" == cusparseLt || "$lib" == cudnn ]] && links+=("-L$shim")
  if [[ "$lib" == cusparseLt ]]; then links+=("-I$root/nvidia/include"); fi
  if [[ "$lib" == cudnn ]]; then
    cudnn_inc="$(cudnn_include_dir)" || { echo "SKIP: no cuDNN headers"; exit 0; }
    links+=("-I$cudnn_inc")
  fi
  links+=("-l$lib")
done
if grep -q '#include <cudnn_frontend.h>' "$src"; then
  fe="$("$root/nvidia/tests/e2e/fetch_cudnn_frontend.sh")" || { echo "SKIP: cudnn-frontend could not be fetched"; exit 0; }
  links+=("-I$fe")
fi
sim_arch=compute_80; [[ "$probe" == cvt ]] && sim_arch=compute_89
nvcc -std=c++17 -cudart "$cudart" -arch=$sim_arch -code=$sim_arch -Wno-deprecated-gpu-targets \
     -Xcompiler -Wno-deprecated-declarations $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" "${links[@]}"
require_shim_libs "$shim" "$out" || exit 0
fails=0
for slug in "${slugs[@]}"; do
  expected="$data/$probe.$slug.txt"
  [[ -f "$expected" ]] || { echo "no expected file $expected" >&2; fails=1; continue; }
  set +e
  VGPU_QUIET=1 VGPU_GPU="nvidia/$slug" VGPU_E2E_DATA="$root/nvidia/tests/data" LD_LIBRARY_PATH="$shim" "$out" "${pargs[@]}" \
      > "$out.$slug.txt" 2> "$out.$slug.err"
  rc=$?
  set -e
  if [[ $rc != 0 ]]; then echo "FAIL $slug: exit $rc"; tail -5 "$out.$slug.err"; fails=1; continue; fi
  if compare "$expected" "$out.$slug.txt" > "$out.$slug.diff"; then
    echo "ok   $slug ($(grep -vc '^#' "$expected") lines)"
  else
    echo "FAIL $slug: differs from the card's transcript ($(grep -c '^[<>]' "$out.$slug.diff") lines differ)"
    head -12 "$out.$slug.diff"
    fails=1
  fi
  rm -f "$out.$slug.diff" "$out.$slug.err"
done
(( fails == 0 )) && echo "PASS" || { echo "FAIL"; exit 1; }
