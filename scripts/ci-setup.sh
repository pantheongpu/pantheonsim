#!/usr/bin/env bash
# Set up PantheonSim on a CI runner that is not GitHub Actions (GitLab CI,
# CircleCI, Forgejo, Jenkins, Buildkite, a plain shell). It does what action.yml
# does: install the build tools and a CUDA or ROCm compiler, build the simulator
# (or reuse a build from an earlier run), and put the simulated GPUs on PATH.
#
# Usage, from a checkout of this repository:
#
#     PSIM_GPU=nvidia/t4 PSIM_COUNT=2 scripts/ci-setup.sh
#     . "$PSIM_ENV_FILE"      # or: . .pantheonsim-build.env
#     nvcc -arch=compute_75 my_test.cu -o my_test && vgpu run ./my_test
#
# The script exports the environment for the rest of its own run and writes it
# to an env file, because a child process cannot change its caller's shell. Each
# CI step is a new shell, so source the file in every step that needs the GPUs.
#
# Inputs, as environment variables (names follow the action's inputs):
#   PSIM_GPU             GPU profile id, see `vgpu list-gpus`. Default nvidia/t4.
#   PSIM_COUNT           How many of them. Default 1.
#   PSIM_CUDA_TOOLKIT    NVIDIA GPU: `apt` installs Ubuntu's nvidia-cuda-toolkit,
#                        a version like 12.6 or 13.0 installs that nvcc from
#                        NVIDIA's repository, `none` uses a toolkit already on the
#                        runner. Empty means apt (NVIDIA GPU, Ubuntu 24.04), 12.6
#                        (NVIDIA GPU, older Ubuntu) or none (AMD GPU).
#   PSIM_ROCM            AMD GPU: a ROCm version like 7.1 whose hipcc is installed
#                        from AMD's repository, or `none` for a ROCm already on the
#                        runner. Empty means 7.1 for an AMD GPU, none otherwise.
#   PSIM_LIBRARY_PATH    true (default) puts the simulator's libraries on
#                        LD_LIBRARY_PATH, so programs run directly as well as under
#                        `vgpu run`. false leaves it alone (see docs/github-action.md
#                        for when that is needed).
#   PSIM_BUILD_DIR       Where to build. Default $PWD/.pantheonsim-build. Cache this
#                        directory between runs.
#   PSIM_ENV_FILE        Where to write the env file. Default $PSIM_BUILD_DIR.env,
#                        next to the build directory, not inside it.
#
# Outputs: the env file (PATH, VGPU_GPU, VGPU_DEVICE_COUNT, VGPU_SHIM_DIR,
# LD_LIBRARY_PATH, ROCM_PATH, CMAKE_TOOLCHAIN_FILE, NVCC_PREPEND_FLAGS as needed).
# When GITHUB_PATH and GITHUB_ENV are set, the same changes are appended to them
# too; nothing else here depends on GitHub.
#
# Re-running is safe. A build directory that was built from the same sources with
# the same toolkit on the same OS release is reused, and only the (fast) steps
# that put the simulated GPUs in place run again. The script prints the cache key
# it computed, for CI systems whose cache is keyed by a string.
#
# Requirements: Linux x86_64, Ubuntu 22.04 or 24.04 (the NVIDIA and AMD package
# repositories are addressed by the Ubuntu release), root or sudo, network access
# to the Ubuntu mirrors and, depending on the inputs, developer.download.nvidia.com,
# repo.radeon.com and PyPI (the cuDNN headers are fetched at configure time, see
# THIRD_PARTY_NOTICES.md). No GPU and no privileged container are needed.
#
# Nothing NVIDIA or AMD proprietary is bundled here. The toolkit is installed
# from Ubuntu's or the vendor's own package repository, on the runner, at job time,
# exactly as action.yml does.
set -euo pipefail

err() { echo "pantheonsim: error: $*" >&2; exit 1; }
warn() { echo "pantheonsim: warning: $*" >&2; }

GPU=${PSIM_GPU:-nvidia/t4}
COUNT=${PSIM_COUNT:-1}
TOOLKIT=${PSIM_CUDA_TOOLKIT:-}
ROCM=${PSIM_ROCM:-}
LIBPATH=${PSIM_LIBRARY_PATH:-true}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD=${PSIM_BUILD_DIR:-$PWD/.pantheonsim-build}
mkdir -p "$BUILD"; BUILD=$(cd "$BUILD" && pwd)
ENVF=${PSIM_ENV_FILE:-$BUILD.env}
TMPD=$(mktemp -d "${TMPDIR:-/tmp}/pantheonsim.XXXXXX")
trap 'rm -rf "$TMPD"' EXIT

# Check the runner.
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || err "PantheonSim needs a Linux x86_64 runner"
[[ -r /etc/os-release ]] || err "no /etc/os-release; only Ubuntu is supported"
# shellcheck disable=SC1091
. /etc/os-release
[[ "$ID" == ubuntu ]] || err "only Ubuntu is supported (this is ${ID:-unknown}); use an ubuntu:24.04 image"

# Work out the toolkits (the same rules as action.yml).
case "$GPU" in amd/*) amd=1 ;; *) amd= ;; esac
# Ubuntu's own toolkit is CUDA 12.0 on 24.04 but 11.5 on 22.04, and the
# simulator needs CUDA 12 or newer. Older releases get NVIDIA's 12.6.
old_ubuntu=; [[ "${VERSION_ID%%.*}" -lt 24 ]] && old_ubuntu=1
[[ -n "$TOOLKIT" ]] || TOOLKIT=$([[ $amd ]] && echo none || { [[ $old_ubuntu ]] && echo 12.6 || echo apt; })
if [[ "$TOOLKIT" == apt && $old_ubuntu ]]; then
  err "Ubuntu $VERSION_ID's CUDA toolkit is too old for the simulator, which needs CUDA 12 or newer. Use PSIM_CUDA_TOOLKIT=12.6"
fi
[[ -n "$ROCM" ]] || ROCM=$([[ $amd ]] && echo 7.1 || echo none)
# Checked before anything reaches a shell command or a URL.
[[ "$TOOLKIT" =~ ^(apt|none|[0-9]+\.[0-9]+)$ ]] || err "PSIM_CUDA_TOOLKIT must be apt, none or a version like 12.6, not '$TOOLKIT'"
[[ "$ROCM" =~ ^(none|[0-9]+\.[0-9]+(\.[0-9]+)?)$ ]] || err "PSIM_ROCM must be none or a version like 7.1, not '$ROCM'"
[[ "$COUNT" =~ ^[1-9][0-9]*$ ]] || err "PSIM_COUNT must be a positive number, not '$COUNT'"
case "$LIBPATH" in true|false) ;; *) err "PSIM_LIBRARY_PATH must be true or false, not '$LIBPATH'" ;; esac

# A container job runs as root with no sudo; a VM is the other way round.
sudo=sudo; [[ "$(id -u)" == 0 ]] && sudo=
if [[ -n "$sudo" ]] && ! command -v sudo >/dev/null; then
  err "not root and no sudo; run as root or install sudo"
fi
export DEBIAN_FRONTEND=noninteractive
extra_path=(); extra_env=()

# Install build tools. The download tools are not in a bare container image.
# python3 runs scripts/fetch-cudnn-headers.py at configure time.
pkgs="cmake ninja-build g++ ca-certificates wget gnupg python3"
# Ubuntu's toolkit is CUDA 12.0, whose nvcc accepts gcc 12 at most.
[[ "$TOOLKIT" == apt ]] && pkgs="$pkgs nvidia-cuda-toolkit g++-12"
# The simulator is C++20 and needs gcc 12 or newer; 22.04's default is 11.
[[ $old_ubuntu ]] && pkgs="$pkgs g++-12"
$sudo apt-get update -qq
# Unpacking the NVIDIA runtime libraries that nvidia-cuda-toolkit pulls in has hung a
# hosted runner for 45 minutes. Bound each attempt, and finish what dpkg left
# half-configured before the next.
for attempt in 1 2 3; do
  # shellcheck disable=SC2086  # $pkgs is a word list
  $sudo timeout 900 apt-get install -y -qq --no-install-recommends $pkgs && break
  [[ $attempt == 3 ]] && err "apt-get install of $pkgs did not finish in three attempts"
  warn "apt-get install failed or timed out (attempt $attempt of 3); retrying"
  $sudo dpkg --configure -a || true
done

if [[ "$TOOLKIT" != apt && "$TOOLKIT" != none ]]; then
  repo="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64"
  wget -qO "$TMPD/cuda-keyring.deb" "$repo/cuda-keyring_1.1-1_all.deb"
  $sudo dpkg -i "$TMPD/cuda-keyring.deb"
  $sudo apt-get update -qq
  v="${TOOLKIT//./-}"
  # CUDA 13 renamed the NVVM package.
  nvvm="cuda-nvvm-$v"; [[ "${TOOLKIT%%.*}" -ge 13 ]] && nvvm="libnvvm-$v"
  # nvcc, plus the headers of every library the simulator stands in for (it
  # builds each one against the toolkit's own declarations), and nvJitLink,
  # which the toolkit's cuSPARSE needs to link a program against it.
  $sudo apt-get install -y -qq --no-install-recommends \
    "cuda-nvcc-$v" "cuda-cudart-dev-$v" "cuda-crt-$v" "$nvvm" \
    "cuda-nvml-dev-$v" "cuda-cupti-dev-$v" "cuda-nvrtc-dev-$v" "libcublas-dev-$v" \
    "libcufft-dev-$v" "libcurand-dev-$v" "libcusparse-dev-$v" "libnvjpeg-dev-$v" \
    "libcusolver-dev-$v" "libnpp-dev-$v" "libnvjitlink-dev-$v" "cuda-profiler-api-$v" \
    || err "could not install CUDA $TOOLKIT from NVIDIA's repository for Ubuntu $VERSION_ID"
  extra_path+=("/usr/local/cuda-$TOOLKIT/bin")
  export PATH="/usr/local/cuda-$TOOLKIT/bin:$PATH"
fi

# hipcc and the pieces it needs to compile and link, not the ROCm runtime: the
# simulator is the runtime. Extracted rather than installed, because the full
# packages pull in the kernel driver and the whole runtime stack.
if [[ "$ROCM" != none ]]; then
  # Ubuntu 24.04 renamed libelf1; 22.04 knows the new name but has no package for it.
  elf=libelf1t64; apt-cache policy libelf1t64 2>/dev/null | grep -q 'Candidate: [0-9]' || elf=libelf1
  $sudo apt-get install -y -qq --no-install-recommends libxml2 libnuma1 "$elf"
  $sudo mkdir -p /etc/apt/keyrings
  wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor | $sudo tee /etc/apt/keyrings/rocm.gpg >/dev/null
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$ROCM $VERSION_CODENAME main" \
    | $sudo tee /etc/apt/sources.list.d/rocm.list >/dev/null
  # AMD's packages, not Ubuntu's hipcc of the same name.
  printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 1001\n' | $sudo tee /etc/apt/preferences.d/rocm >/dev/null
  $sudo apt-get update -qq
  dir="$TMPD/rocm-debs"; mkdir -p "$dir"
  (cd "$dir" && apt-get download hipcc hip-dev rocm-llvm rocm-device-libs rocm-core hsa-rocr-dev) \
    || err "could not download ROCm $ROCM for Ubuntu $VERSION_ID from repo.radeon.com"
  for d in "$dir"/*.deb; do $sudo dpkg-deb -x "$d" /; done
  rm -rf "$dir"
  root=$(ls -d /opt/rocm-"$ROCM"* | sort -V | tail -1)
  $sudo ln -sfn "$root" /opt/rocm
  extra_path+=(/opt/rocm/bin); extra_env+=("ROCM_PATH=/opt/rocm"); export ROCM_PATH=/opt/rocm
  /opt/rocm/bin/hipcc --version | head -3
fi

# Work out the build. The key is the source itself, the toolkit version (the CUDA
# libraries are built against its headers) and the OS release (a build from 24.04
# does not load on 22.04).
srchash=$(cd "$SRC" && find CMakeLists.txt src include nvidia amd tools scripts -type f -print0 \
          | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-20)
nvcc_version=$( { nvcc --version 2>/dev/null || true; } | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)
key="pantheonsim-$(uname -s)-$(uname -m)-${ID}${VERSION_ID}-${srchash}-cuda${nvcc_version:-none}"
echo "pantheonsim: cache key $key"

if [[ -x "$BUILD/vgpu" && "$(cat "$BUILD/.ci-setup-key" 2>/dev/null || true)" == "$key" ]]; then
  echo "pantheonsim: using the cached build in $BUILD"
else
  cxx=g++; cc=gcc
  [[ "$(g++ -dumpversion | cut -d. -f1)" -lt 12 ]] && { cxx=g++-12; cc=gcc-12; }
  # The C compiler matches the C++ one: with link-time optimization the two must
  # be the same GCC, or the link rejects the C objects' LTO version.
  cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$cxx" -DCMAKE_C_COMPILER="$cc"
  cmake --build "$BUILD" -j"$("$SRC/ci/jobs.sh" -p release)"
  echo "$key" > "$BUILD/.ci-setup-key"
fi

# Put the simulated GPUs in place.
# An unknown profile fails here, by name, rather than in the job's tests.
"$BUILD/vgpu" info --gpu "$GPU" >/dev/null || err "no GPU profile '$GPU' (see vgpu list-gpus)"
# vgpu lives one level up and finds its shim beside itself; the build links it
# into bin, but a cached build from before that did not.
[[ -e "$BUILD/bin/vgpu" ]] || ln -s ../vgpu "$BUILD/bin/vgpu"
# A program hipcc built looks for libamdhip64 in /opt/rocm/lib first (RUNPATH).
# Pointing that at the simulator lets it run directly as well as under vgpu run.
if [[ "$ROCM" != none && -d /opt/rocm/lib ]]; then
  hip=$(readlink -f "$BUILD/shim/libamdhip64.so.6")
  for n in libamdhip64.so libamdhip64.so.6 libamdhip64.so.7; do $sudo ln -sfn "$hip" "/opt/rocm/lib/$n"; done
  # ROCm's CMake packages name the runtime by its full version, which only the
  # real runtime package installs.
  for n in $(grep -rhoE 'libamdhip64\.so\.[0-9.]+[0-9]' /opt/rocm/lib/cmake 2>/dev/null | sort -u); do
    $sudo ln -sfn "$hip" "/opt/rocm/lib/$n"
  done
  # hipcc with no --offload-arch asks this which GPUs are present.
  $sudo ln -sfn "$BUILD/bin/rocm_agent_enumerator" /opt/rocm/bin/rocm_agent_enumerator
fi

# The env file. build/bin goes first on PATH: nvidia-smi and nvcc there are the
# simulator's drop-ins, and vgpu runs programs against it.
path="$BUILD/bin"; for p in "${extra_path[@]}"; do path="$path:$p"; done
{
  echo "# Source this file: . $ENVF"
  # shellcheck disable=SC2016  # the ${...} is for the shell that sources the file
  printf 'export PATH=%q${PATH:+:$PATH}\n' "$path"
  printf 'export VGPU_GPU=%q VGPU_DEVICE_COUNT=%q VGPU_SHIM_DIR=%q\n' "$GPU" "$COUNT" "$BUILD/shim"
  for e in "${extra_env[@]}"; do echo "export $e"; done
  if [[ "$TOOLKIT" == apt ]] && command -v g++-12 >/dev/null; then
    echo 'export NVCC_PREPEND_FLAGS="-ccbin g++-12"'
  fi
  # shellcheck disable=SC2016  # the ${...} is for the shell that sources the file
  if [[ "$LIBPATH" == true ]]; then printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$BUILD/shim"; fi
  # CMake links CUDA programs with the static runtime unless told otherwise, and
  # the static runtime cannot talk to a simulated driver. A toolchain file (read
  # from the environment by CMake 3.21 and later) makes the shared one the default.
  if [[ -z "$amd" ]]; then
    tc="$BUILD.toolchain.cmake"
    # A value left by this script's own earlier run (the env file sourced) is not the caller's.
    if [[ -n "${CMAKE_TOOLCHAIN_FILE:-}" && "${CMAKE_TOOLCHAIN_FILE}" != "$tc" ]]; then
      warn "CMAKE_TOOLCHAIN_FILE is already set, so CMake keeps its default static CUDA runtime, which cannot run on the simulator. Add set(CMAKE_CUDA_RUNTIME_LIBRARY Shared) to that toolchain file."
    else
      printf '%s\n' '# Written by scripts/ci-setup.sh: link CUDA programs with the shared runtime.' \
        'set(CMAKE_CUDA_RUNTIME_LIBRARY Shared CACHE STRING "CUDA runtime library")' > "$tc"
      printf 'export CMAKE_TOOLCHAIN_FILE=%q\n' "$tc"
    fi
  fi
} > "$ENVF"
# shellcheck disable=SC1090
. "$ENVF"

# On GitHub Actions the same changes go where the runner reads them.
if [[ -n "${GITHUB_PATH:-}" ]]; then
  echo "$BUILD/bin" >> "$GITHUB_PATH"
  for p in "${extra_path[@]}"; do echo "$p" >> "$GITHUB_PATH"; done
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "VGPU_GPU=$GPU"; echo "VGPU_DEVICE_COUNT=$COUNT"; echo "VGPU_SHIM_DIR=$BUILD/shim"
    for e in "${extra_env[@]}"; do echo "$e"; done
    if [[ -n "${NVCC_PREPEND_FLAGS:-}" ]]; then echo "NVCC_PREPEND_FLAGS=$NVCC_PREPEND_FLAGS"; fi
    if [[ "$LIBPATH" == true ]]; then echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"; fi
    if [[ -z "$amd" && -n "${CMAKE_TOOLCHAIN_FILE:-}" ]]; then echo "CMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE"; fi
  } >> "$GITHUB_ENV"
fi

if [[ $amd ]]; then rocm-smi; else nvidia-smi -L; fi
echo "pantheonsim: ready; source $ENVF to use it"
