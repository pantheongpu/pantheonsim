#!/usr/bin/env bash
# NVSHMEM's bootstraps through a launcher: MPI (a communicator, NVSHMEM_BOOTSTRAP=MPI, the MPI
# plugin by name), PMIx (NVSHMEM_BOOTSTRAP_PMI=PMIX), PMI and PMI-2 without their libraries,
# OpenSHMEM (a flag and NVSHMEM_BOOTSTRAP=SHMEM), and the settings NVSHMEM refuses -- programs
# built with NVIDIA's NVSHMEM headers and device library and run as jobs of two PEs.
#
#   run_nvshmem_bootstrap.sh                   run against the shims and compare with
#                                              nvidia/tests/data/nvshmem_bootstrap_paths.expected
#   run_nvshmem_bootstrap.sh --card            run against NVIDIA's NVSHMEM on the GPUs of this
#                                              machine (an RTX 3060 made the expected file)
#   run_nvshmem_bootstrap.sh --card --update   rewrite the expected file from the card
#
# Both PEs use GPU 0: NVIDIA's NVSHMEM refuses PEs on two GeForce GPUs that have no peer access
# ("Peer GPU 1 is not accessible", NVSHMEMX_ERROR_NOT_SUPPORTED), and PEs on one GPU make its
# multiple-processes-per-GPU mode (status 3). The scenario two_gpus puts PE r on GPU r: on the card's two
# RTX 3060s that is the refusal (exit status 255), and the simulated rtx3060 profile, whose devices
# have no peer path either, refuses it the same way; on a data-centre profile (a100) the job starts, which
# is checked below (the card has no such pair). It needs two GPUs on the card. A scenario whose launcher or library is missing is
# skipped here and in the expected file. Open MPI's OpenSHMEM crashes now and then in UCX on its
# own (one run in four of a plain shmem_init program on the machine this was made on, with or
# without NVSHMEM), so an OpenSHMEM job that died of a segmentation fault or failed to start is run
# again, up to eight times.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="${VGPU_BUILD_DIR:-$root/build}"
shim="$build/shim"
expected="$root/nvidia/tests/data/nvshmem_bootstrap_paths.expected"
card=0; update=0
for a in "$@"; do
  case "$a" in
    --card) card=1 ;;
    --update) update=1 ;;
    *) echo "usage: $0 [--card [--update]]" >&2; exit 2 ;;
  esac
done
for tool in nvcc mpicxx mpirun; do
  command -v "$tool" >/dev/null 2>&1 || { echo "SKIP: $tool not found (this test needs the CUDA toolkit and Open MPI)"; exit 0; }
done
mpirun --version 2>&1 | grep -q "Open MPI" || { echo "SKIP: the runner uses Open MPI's mpirun"; exit 0; }
echo '#include <mpi.h>' | mpicxx -x c++ -fsyntax-only - >/dev/null 2>&1 || { echo "SKIP: mpi.h not found"; exit 0; }
if (( ! card )); then
  [[ -e "$shim/libnvshmem_host.so.3" ]] || { echo "SKIP: the NVSHMEM shim is not built"; exit 0; }
  [[ -z "$(shim_sanitizer "$shim")" ]] || { echo "SKIP: a sanitizer build (Open MPI itself is not clean under the leak checker)"; exit 0; }
fi
nvs="${NVSHMEM_HOME:-}"
if [[ -z "$nvs" ]]; then
  nvs="$(python3 -c 'import os, nvidia.nvshmem as m; print(os.path.dirname(m.__file__))' 2>/dev/null || true)"
  [[ -n "$nvs" && -d "$nvs" ]] || nvs="$(python3 -c 'import os, nvidia; print(os.path.join(list(nvidia.__path__)[0], "nvshmem"))' 2>/dev/null || true)"
fi
if [[ -z "$nvs" || ! -e "$nvs/include/nvshmem.h" || ! -e "$nvs/lib/libnvshmem_device.a" ]]; then
  echo "SKIP: NVIDIA's NVSHMEM headers and device library are not installed (set NVSHMEM_HOME)"; exit 0
fi
cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"

tmp="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_nvshmem_boot.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

if (( card )); then
  # NVIDIA's host library with its bootstrap plugins beside it, where the library looks for them.
  mkdir "$tmp/nv"
  for f in "$nvs"/lib/libnvshmem_host.so.3 "$nvs"/lib/nvshmem_bootstrap_*.so.3 "$nvs"/lib/nvshmem_transport_*.so.*; do
    [[ -e "$f" ]] && ln -s "$f" "$tmp/nv/"
  done
  ln -s libnvshmem_host.so.3 "$tmp/nv/libnvshmem_host.so"
  libdir="$tmp/nv:$cuda_root/lib64"
  linkdir="$tmp/nv"
  arch=sm_86
  runenv=(NVSHMEM_REMOTE_TRANSPORT=none)  # no InfiniBand here, and its probe only prints a warning
else
  libdir="$shim"
  linkdir="$shim"
  arch=sm_80
  runenv=(VGPU_GPU=nvidia/a100 VGPU_DEVICE_COUNT=2 VGPU_QUIET=1)
fi

compile() {  # compile <launcher compiler> <source> <output>
  nvcc -std=c++17 -rdc=true -cudart shared -arch="$arch" -Wno-deprecated-gpu-targets -ccbin "$1" \
       -I"$nvs/include" "$root/nvidia/tests/e2e/$2" -o "$tmp/$3" "$nvs/lib/libnvshmem_device.a" -L"$linkdir" -lnvshmem_host \
       > "$tmp/$3.log" 2>&1 || { cat "$tmp/$3.log"; echo "SKIP: $2 does not build here (NVSHMEM's device library is for another CUDA major?)"; exit 0; }
}
compile mpicxx nvshmem_bootstrap_paths.cu mpiprog
have_shmem=0
if command -v oshrun >/dev/null 2>&1 && command -v oshc++ >/dev/null 2>&1 && echo '#include <shmem.h>' | oshc++ -x c++ -fsyntax-only - >/dev/null 2>&1; then
  compile oshc++ nvshmem_bootstrap_shmem.cu shmemprog
  have_shmem=1
fi
if (( card )) && [[ ! -e "$tmp/mpiprog" ]]; then exit 0; fi
(( card )) || require_shim_libs "$shim" "$tmp/mpiprog" || exit 0   # the card run uses NVIDIA's libraries, not the shim's

# PMIx: libpmix.so.2 for the card's plugin, and pmix.h at build time for the shim as well.
has_pmix_lib() { /sbin/ldconfig -p 2>/dev/null | grep -q 'libpmix\.so\.2' || compgen -G '/usr/lib/*/libpmix.so.2' >/dev/null; }
have_pmix=0
if (( card )); then
  has_pmix_lib && have_pmix=1
else
  grep -q '^VGPU_PMIX_INCLUDE_DIR:PATH=/' "$build/CMakeCache.txt" 2>/dev/null && has_pmix_lib && have_pmix=1
fi

summary="$tmp/summary"
: > "$summary"
skipped=()

# scenario <launcher> <name> <program> <arg> [VAR=value ...]
scenario() {
  local launcher="$1" name="$2" prog="$3" arg="$4"; shift 4
  local asroot=() vars=(HWLOC_COMPONENTS=-gl OMPI_MCA_btl=self,vader
                        LD_LIBRARY_PATH="$libdir" "${runenv[@]}" NVSHMEM_BOOT_OUT="$tmp/out.$name" "$@") xs=() v rc=0 tries=1
  [[ "$(id -u)" == 0 ]] && asroot=(--allow-run-as-root)
  for v in "${vars[@]}"; do xs+=(-x "${v%%=*}"); done
  [[ "$launcher" == oshrun ]] && tries=8
  local t
  for ((t = 0; t < tries; ++t)); do
    rm -f "$tmp/out.$name".*
    rc=0
    env "${vars[@]}" timeout -s KILL 120 "$launcher" "${asroot[@]}" --oversubscribe -np 2 "${xs[@]}" \
        "$tmp/$prog" "$arg" > "$tmp/log.$name" 2>&1 || rc=$?
    [[ "$launcher" == oshrun && ( $rc == 139 || $rc == 255 ) ]] || break
  done
  # A job that failed to start is one exit status for the launcher: 0, or 255 for a PE that aborted.
  { echo "== $name (exit status $rc)"; cat "$tmp/out.$name".0 "$tmp/out.$name".1 2>/dev/null || true; } >> "$summary"
}

scenario mpirun mpi_attr mpiprog attr
scenario mpirun mpi_attr_reversed mpiprog attr_reversed
scenario mpirun mpi_attr_null mpiprog attr_null
two_gpus=1
if (( card )); then
  (( $(nvidia-smi -L 2>/dev/null | grep -c '^GPU') >= 2 )) || two_gpus=0
fi
if (( two_gpus )); then
  scenario mpirun two_gpus mpiprog two_gpus VGPU_GPU=nvidia/rtx3060
else
  skipped+=(two_gpus)
fi
scenario mpirun env_mpi mpiprog env NVSHMEM_BOOTSTRAP=MPI
scenario mpirun plugin_mpi mpiprog env NVSHMEM_BOOTSTRAP=plugin NVSHMEM_BOOTSTRAP_PLUGIN=nvshmem_bootstrap_mpi.so.3
if (( have_pmix )); then
  scenario mpirun pmix mpiprog env NVSHMEM_BOOTSTRAP=PMI NVSHMEM_BOOTSTRAP_PMI=PMIX
else
  skipped+=(pmix)
fi
scenario mpirun pmi_without_library mpiprog env NVSHMEM_BOOTSTRAP=PMI
scenario mpirun pmi2_without_library mpiprog env NVSHMEM_BOOTSTRAP=PMI NVSHMEM_BOOTSTRAP_PMI=PMI-2
scenario mpirun bogus_bootstrap mpiprog env NVSHMEM_BOOTSTRAP=bogus
scenario mpirun uid_without_flags mpiprog env NVSHMEM_BOOTSTRAP=UID
scenario mpirun bogus_pmi mpiprog env NVSHMEM_BOOTSTRAP=PMI NVSHMEM_BOOTSTRAP_PMI=bogus
scenario mpirun plugin_not_named mpiprog env NVSHMEM_BOOTSTRAP=plugin
scenario mpirun plugin_missing mpiprog env NVSHMEM_BOOTSTRAP=plugin NVSHMEM_BOOTSTRAP_PLUGIN=nonexistent.so
if (( have_shmem )); then
  scenario oshrun shmem_flag shmemprog flag
  scenario oshrun shmem_env shmemprog env NVSHMEM_BOOTSTRAP=SHMEM
else
  skipped+=(shmem_flag shmem_env)
fi

# What was not run is left out of the expected file too.
want="$tmp/expected"
cp "$expected" "$want" 2>/dev/null || : > "$want"
for s in "${skipped[@]:-}"; do
  [[ -n "$s" ]] || continue
  awk -v s="$s" '/^== /{skip = ($2 == s)} !skip' "$want" > "$want.2" && mv "$want.2" "$want"
done

if (( card && update )); then
  (( ${#skipped[@]} == 0 )) || { echo "not rewriting the expected file: skipped ${skipped[*]}"; exit 1; }
  cp "$summary" "$expected"
  echo "wrote $expected"
  exit 0
fi
if diff -u "$want" "$summary"; then
  echo "NVSHMEM bootstraps match the card's${skipped[*]:+ (skipped: ${skipped[*]})}"
else
  echo "FAIL: the bootstraps differ from what NVIDIA's NVSHMEM did"
  exit 1
fi
