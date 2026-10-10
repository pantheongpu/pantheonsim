#!/usr/bin/env bash
# cuTensorNet's distributed execution: an MPI job (Open MPI), one process per
# rank, each rank on its own simulated GPU.
#
#   run_cutensornet_mpi.sh                   run against the shims
#   run_cutensornet_mpi.sh --card            run against NVIDIA's libraries on the
#                                            GPUs of this machine (two RTX 3060s
#                                            made the expected file): the same
#                                            output must come out
#   run_cutensornet_mpi.sh --card --update   rewrite the expected file from the card
#
# cutensornet_mpi_paths.cpp is run in a job of two ranks and in a job of three
# (the checks print the same text for both), and in four more jobs whose
# communication library cannot be used: none named, a path that holds no
# library, a library without the table, and one with another version of it.
# nvidia/tests/data/cutensornet_mpi_paths.expected is what NVIDIA's library
# printed for the checks. $CUTENSORNET_COMM_LIB is a build of cutn_comm_mpi.c.
#
# Two things about running MPI here:
#  * HWLOC_COMPONENTS=-gl: Open MPI's topology discovery opens an X display
#    for every GPU through hwloc's OpenGL component; where a connection to a
#    display that is not there never fails (a firewall that drops instead of
#    refusing, as on WSL), mpirun hangs before it starts the job;
#  * the job's processes share the machine's two GPUs when there are three ranks.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
expected="$root/nvidia/tests/data/cutensornet_mpi_paths.expected"
card=0; update=0
for a in "$@"; do
  case "$a" in
    --card) card=1 ;;
    --update) update=1 ;;
    *) echo "usage: $0 [--card [--update]]" >&2; exit 2 ;;
  esac
done
for tool in nvcc mpicc mpicxx mpirun; do
  command -v "$tool" >/dev/null 2>&1 || { echo "SKIP: $tool not found (this test needs the CUDA toolkit and Open MPI)"; exit 0; }
done
if ! mpirun --version 2>&1 | grep -q "Open MPI"; then
  echo "SKIP: the runner uses Open MPI's mpirun"; exit 0
fi
if ! echo '#include <mpi.h>' | mpicc -x c -fsyntax-only - >/dev/null 2>&1; then
  echo "SKIP: mpi.h not found (install the Open MPI development files)"; exit 0
fi
cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
inc="$cuda_root/include"
[[ -e "$inc/cuda_runtime_api.h" ]] || inc="$cuda_root/targets/x86_64-linux/include"
[[ -e "$inc/cuda_runtime_api.h" ]] || { echo "SKIP: no CUDA headers beside nvcc"; exit 0; }
if (( ! card )); then
  if [[ -n "$(shim_sanitizer "$shim")" ]]; then
    echo "SKIP: a sanitizer build (Open MPI itself is not clean under the leak checker)"; exit 0
  fi
  have=("$shim"/libcutensornet.so.[0-9]*)
  [[ -e "${have[0]}" ]] || { echo "SKIP: the cuTensorNet shim is not built"; exit 0; }
fi

tmp="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_cutn_mpi.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

mpicc -shared -fPIC -std=gnu99 -I"$inc" "$root/nvidia/tests/e2e/cutn_comm_mpi.c" -o "$tmp/libcutn_comm_mpi.so" -ldl

if (( card )); then
  # NVIDIA's libcutensornet (cuQuantum) and libcutensor, from the pip wheels or CUQUANTUM_LIBS / CUTENSOR_LIBS.
  find_lib() {  # find_lib <name> <var> <dirs...>
    local name="$1" var="$2"; shift 2
    local d="${!var:-}"
    [[ -n "$d" && -e "$d/$name" ]] && { echo "$d"; return 0; }
    for d in "$@"; do [[ -e "$d/$name" ]] && { echo "$d"; return 0; }; done
    return 1
  }
  cutn="$(find_lib libcutensornet.so.2 CUQUANTUM_LIBS "$HOME/.cache/cutensor-dl/x/cuquantum/lib" /usr/lib/x86_64-linux-gnu)" ||
    { echo "SKIP: no NVIDIA libcutensornet.so.2 (set CUQUANTUM_LIBS)"; exit 0; }
  ctnr="$(find_lib libcutensor.so.2 CUTENSOR_LIBS "$HOME/.cache/cutensor-dl/x/cutensor/lib" /usr/lib/x86_64-linux-gnu)" ||
    { echo "SKIP: no NVIDIA libcutensor.so.2 (set CUTENSOR_LIBS)"; exit 0; }
  mkdir "$tmp/nv"
  ln -s "$cutn/libcutensornet.so.2" "$tmp/nv/libcutensornet.so.2"
  ln -s libcutensornet.so.2 "$tmp/nv/libcutensornet.so"
  ln -s "$ctnr/libcutensor.so.2" "$tmp/nv/libcutensor.so.2"
  libdir="$tmp/nv:$cuda_root/lib64"
  mpicxx -std=c++17 -O1 -I"$inc" "$root/nvidia/tests/e2e/cutensornet_mpi_paths.cpp" -o "$tmp/t" \
      -L"$tmp/nv" -lcutensornet -L"$cuda_root/lib64" -lcudart -ldl
  runenv=(VGPU_E2E_REAL=1)
else
  libdir="$shim"
  mpicxx -std=c++17 -O1 -I"$inc" "$root/nvidia/tests/e2e/cutensornet_mpi_paths.cpp" -o "$tmp/t" \
      -L"$shim" -lcutensornet -lcudart -ldl
  require_shim_libs "$shim" "$tmp/t" || exit 0
  runenv=(VGPU_GPU=nvidia/a100 VGPU_DEVICE_COUNT=2 VGPU_QUIET=1)
fi

# run_job <ranks> <mode> [VAR=value ...] : the job's output on stdout; fails unless it ends with PASS
run_job() {
  local ranks="$1" mode="$2"; shift 2
  local asroot=() vars=(HWLOC_COMPONENTS=-gl OMPI_MCA_btl=self,vader OMPI_MCA_btl_vader_single_copy_mechanism=none
                        LD_LIBRARY_PATH="$libdir" "${runenv[@]}" "$@") xs=() v
  [[ "$(id -u)" == 0 ]] && asroot=(--allow-run-as-root)
  for v in "${vars[@]}"; do xs+=(-x "${v%%=*}"); done
  local out="$tmp/out.$ranks.$mode"
  env "${vars[@]}" timeout -s KILL 280 mpirun "${asroot[@]}" --oversubscribe -np "$ranks" "${xs[@]}" \
      "$tmp/t" "$mode" > "$out" 2> "$out.err" ||
    { cat "$out" "$out.err"; echo "FAIL: the $ranks-rank job in mode $mode did not pass"; return 1; }
  [[ "$(tail -1 "$out")" == PASS ]] || { cat "$out" "$out.err"; echo "FAIL: the $ranks-rank job in mode $mode did not end with PASS"; return 1; }
  if grep -qE 'VirtualGPU error \[|is not implemented by VirtualGPU' "$out" "$out.err"; then
    cat "$out" "$out.err"; echo "FAIL: the job reached an unimplemented entry point"; return 1
  fi
  cat "$out"
}

comm="$tmp/libcutn_comm_mpi.so"
if (( card && update )); then
  run_job 2 main CUTENSORNET_COMM_LIB="$comm" > "$expected"
  echo "wrote $expected"
  exit 0
fi
status=0
for ranks in 2 3; do
  run_job "$ranks" main CUTENSORNET_COMM_LIB="$comm" > "$tmp/main.$ranks" || { cat "$tmp/main.$ranks"; status=1; continue; }
  if ! diff -u "$expected" "$tmp/main.$ranks"; then
    echo "FAIL: the $ranks-rank job's checks differ from what NVIDIA's library passed"; status=1
  fi
done
run_job 2 nolib > /dev/null || status=1
run_job 2 nolib CUTENSORNET_COMM_LIB="$tmp/no-such-library.so" > /dev/null || status=1
run_job 2 nosym CUTENSORNET_COMM_LIB="$(gcc -print-file-name=libm.so.6)" > /dev/null || status=1
run_job 2 nosym CUTENSORNET_COMM_LIB="$comm" CUTN_COMM_VERSION=1 > /dev/null || status=1
(( status == 0 )) && echo "cuTensorNet distributed execution passed (2 and 3 ranks, and four unusable communication libraries)"
exit "$status"
