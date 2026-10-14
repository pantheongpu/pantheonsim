#!/usr/bin/env bash
# The training run for a profile-guided build (VGPU_PGO=generate in
# CMakeLists.txt). Run it between the two builds:
#
#   cmake -S . -B build-pgo -DVGPU_PGO=generate && cmake --build build-pgo --target vgpucuda vgpucudart
#   scripts/pgo-train.sh build-pgo /path/to/pantheon/build-vgpu-86
#   cmake -S . -B build-pgo -DVGPU_PGO=use      && cmake --build build-pgo
#
# The second argument is a directory of built pantheon workloads (the binaries
# `make PLATFORM=CUDA BUILD_DIR=...` leaves in a pantheon checkout, built for
# the GPU in VGPU_GPU). The programs run at CPU-sized settings against the
# instrumented shims; each one writes its counts into build-pgo/pgo when it
# exits. An instrumented build runs about ten times slower, so each workload
# gets a two-second duration and one thread.
#
# The profile is only valid for the compiler version and the sources it was
# taken from, and only for a build directory at the same path: do not keep it.
set -uo pipefail
build="${1:?usage: pgo-train.sh BUILD_DIR WORKLOAD_BIN_DIR}"
bins="${2:?usage: pgo-train.sh BUILD_DIR WORKLOAD_BIN_DIR}"
build="$(cd "$build" && pwd)"
bins="$(cd "$bins" && pwd)"
[[ -d "$build/shim" ]] || { echo "pgo-train: $build/shim not found; build the shims first" >&2; exit 2; }
[[ -d "$build/pgo" || -n "$(find "$build" -maxdepth 1 -name 'libcudart.so.*' | head -1)" ]] ||
  { echo "pgo-train: $build does not look like a built tree" >&2; exit 2; }

export LD_LIBRARY_PATH="$build/shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VGPU_GPU="${VGPU_GPU:-nvidia/a10}" VGPU_VRAM_MB="${VGPU_VRAM_MB:-1024}" VGPU_THREADS=1
export VGPU_MAX_STEPS="${VGPU_MAX_STEPS:-17179869184}"

# name | extra environment (or -) | arguments after "<gpu> <seconds> <mem%>"
# Each runs both the SASS engine (the default) and, for int_virus, the PTX one.
runs=(
  "compute_virus|-|--grid_size 8"
  "pulse_virus|-|--grid_size 1"
  "memory_bank_thrash|-|--grid_size 1"
  "mma_virus|-|--grid_size 1 --kernel_loops 100"
  "fp64_virus|-|--grid_size 1 --kernel_loops 100"
  "cache_latency|-|--grid_size 4"
  "int_virus|-|--grid_size 4"
  "int_virus|VGPU_SASS=0|--grid_size 1"
)
rc=0
for spec in "${runs[@]}"; do
  IFS='|' read -r name senv args <<<"$spec"
  [[ -x "$bins/$name" ]] || { echo "pgo-train: skip $name (not in $bins)"; continue; }
  envs=(); [[ "$senv" != "-" ]] && envs=("$senv")
  echo "pgo-train: $name ${envs[*]:-} $args"
  # shellcheck disable=SC2086
  if ! env "${envs[@]}" timeout 900 "$bins/$name" 0 2 2 $args >/dev/null 2>&1; then
    echo "pgo-train: $name failed; its counts may be missing" >&2; rc=1
  fi
done
echo "pgo-train: profile files: $(find "$build" -name '*.gcda' | wc -l)"
exit $rc
