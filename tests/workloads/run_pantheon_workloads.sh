#!/usr/bin/env bash
# Runs real pantheon workloads against VirtualGPU.
#
# The unit tests cover instructions and API calls one at a time. These cover the
# thing that actually matters: an unmodified GPU diagnostic, built by its own
# Makefile with its own nvcc flags, running end to end. Every serious bug this
# simulator has had showed up here first and in the unit tests second.
#
# Pantheon is a separate public repository. Point VGPU_PANTHEON_DIR at a
# checkout, or let this find one next to the build. Skips cleanly when absent so
# a contributor without it still gets a green run.
#
# By default it runs three quick workloads on one simulated A10, which is what
# ctest and every pull request run. The nightly workflow runs all of them on a
# spread of machines:
#
#   VGPU_WORKLOADS=all             every workload in kernels/ (or a list of names)
#   VGPU_WORKLOAD_GPU=nvidia/h100  the profile; the Makefile builds for its arch
#   VGPU_WORKLOAD_COUNT=8          how many GPUs (all_reduce and p2p_thrasher need 2)
#   VGPU_WORKLOAD_ARGS="..."       extra workload flags, e.g. a CPU-sized --grid_size
#   VGPU_WORKLOAD_TIMEOUT=300      seconds before one workload counts as hung
#   VGPU_WORKLOAD_REPORT=dir       write results.tsv, summary.md and each log there
set -uo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
build="${VGPU_BUILD_DIR:-$root/build}"

pantheon=""
for cand in "${VGPU_PANTHEON_DIR:-}" "$root/../pantheon" "$HOME/pantheon"; do
  [[ -n "$cand" && -f "$cand/Makefile" && -d "$cand/kernels" ]] && { pantheon="$(cd "$cand" && pwd)"; break; }
done
[[ -n "$pantheon" ]] || { echo "SKIP: no pantheon checkout (set VGPU_PANTHEON_DIR)"; exit 0; }
command -v nvcc >/dev/null || { echo "SKIP: nvcc not found (the workloads are compiled from source)"; exit 0; }
[[ -x "$build/vgpu" ]] || { echo "SKIP: $build/vgpu not built"; exit 0; }

# The shim's soname major follows the toolkit it was built against, and nvcc
# stamps these binaries with the major *it* ships. A CUDA 12 build directory
# next to a CUDA 13 nvcc is an unbuildable pairing rather than a defect: the
# workloads would link the real libcudart, reach the real driver, and fail with
# "integrity checks failed" -- which reads as a simulator bug and is not one.
shopt -s nullglob
_cudart=("$build"/shim/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#_cudart[@]} )); then
  _shim_major="${_cudart[0]##*.}"
  _nvcc_major="$(nvcc --version | sed -n 's/.*release \([0-9]*\).*/\1/p' | head -1)"
  if [[ -n "$_nvcc_major" && "$_shim_major" != "$_nvcc_major" ]]; then
    echo "SKIP: shim is CUDA $_shim_major but nvcc is CUDA $_nvcc_major;" \
         "the workloads would link the real runtime"
    exit 0
  fi
fi

# Not CUDA, so not simulated: OptiX is NVIDIA's ray-tracing library and NVENC its
# hardware video encoder. Reported as skipped, never as passed.
OUT_OF_SCOPE="rt_virus media_enc_virus"

# Small, fast, and between them they exercise the paths that have broken before:
# integer and float arithmetic, global traffic, shared memory and atomics.
WORKLOADS="${VGPU_WORKLOADS:-compute_virus int_virus cache_latency}"
if [[ "$WORKLOADS" == all ]]; then
  WORKLOADS=$(cd "$pantheon/kernels" && for d in */; do d="${d%/}"; [[ "$d" == common ]] || printf '%s ' "$d"; done)
fi
GPU="${VGPU_WORKLOAD_GPU:-nvidia/a10}"
COUNT="${VGPU_WORKLOAD_COUNT:-1}"
VRAM_MB="${VGPU_WORKLOAD_VRAM_MB:-1024}"
DURATION="${VGPU_WORKLOAD_SECONDS:-2}"
MEM_PCT="${VGPU_WORKLOAD_MEM_PCT:-2}"
ARGS="${VGPU_WORKLOAD_ARGS:-}"
TIMEOUT="${VGPU_WORKLOAD_TIMEOUT:-0}"
REPORT="${VGPU_WORKLOAD_REPORT:-}"
# These are soak kernels: they are built to keep a GPU busy for a duration, so
# their launches are legitimately long and the default step budget -- which
# exists to catch a kernel looping forever -- cuts them off. int_virus trips it.
# Raise it rather than pick easier workloads, since being long-running is the
# property that makes them worth testing. Still finite, so a real runaway stops.
export VGPU_MAX_STEPS="${VGPU_MAX_STEPS:-17179869184}"

logs="${REPORT:-$(mktemp -d)}"
mkdir -p "$logs"
logs="$(cd "$logs" && pwd)"

echo "pantheon: $pantheon ($(git -C "$pantheon" rev-parse --short HEAD 2>/dev/null || echo unknown))"
echo "machine:  $COUNT x $GPU, ${VRAM_MB} MB each, $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/CUDA \1/p' | head -1)"
echo "run:      ${DURATION}s ${MEM_PCT}% ${ARGS:+$ARGS }timeout=${TIMEOUT}s max_steps=$VGPU_MAX_STEPS"
echo "workloads: $(wc -w <<<"$WORKLOADS")"

# Build and run inside the shell: it supplies an nvcc that links the shared CUDA
# runtime, without which the binaries link a static one that cannot run against
# a simulated driver, and an nvidia-smi that reports the simulated card, which
# is what the Makefile reads to pick the architecture to build for.
inner=$(cat <<'INNER'
set -u
cd "$VGPU_WL_PANTHEON"
# Targets are named by their output path: a bare name hits make's builtin
# rule and compiles the .cpp with g++, which cannot take CUDA flags.
# A separate build directory per architecture: the Makefile's signature check
# would otherwise rebuild everything each time the matrix changes card.
bdir="build-vgpu-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)"
targets=""; for w in $VGPU_WL_NAMES; do case " $VGPU_WL_SKIP " in *" $w "*) ;; *) targets="$targets $bdir/$w" ;; esac; done
make -k PLATFORM=CUDA BUILD_DIR="$bdir" -j"$(nproc)" $targets >"$VGPU_WL_LOGS/build.log" 2>&1 || true
rc=0
: >"$VGPU_WL_LOGS/results.tsv"
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >>"$VGPU_WL_LOGS/results.tsv"; printf '%-26s %-8s %5ss  %s\n' "$1" "$2" "$3" "$4"; }
printf '\n%-26s %-8s %6s  %s\n' WORKLOAD RESULT TIME DETAIL
for w in $VGPU_WL_NAMES; do
  case " $VGPU_WL_SKIP " in *" $w "*) record "$w" SKIP 0 "needs OptiX or NVENC, which are not CUDA"; continue ;; esac
  bin="$bdir/$w"
  log="$VGPU_WL_LOGS/$w.log"
  if [ ! -x "$bin" ]; then
    grep -E "$w" "$VGPU_WL_LOGS/build.log" | grep -iE "error" | head -3 >"$log"
    record "$w" MISSING 0 "did not build: $(head -1 "$log" | cut -c1-80)"; rc=1; continue
  fi
  start=$(date +%s)
  if [ "$VGPU_WL_TIMEOUT" -gt 0 ]; then
    timeout "$VGPU_WL_TIMEOUT" "$bin" 0 "$VGPU_WL_DURATION" "$VGPU_WL_MEMPCT" $VGPU_WL_ARGS >"$log" 2>&1
  else
    "$bin" 0 "$VGPU_WL_DURATION" "$VGPU_WL_MEMPCT" $VGPU_WL_ARGS >"$log" 2>&1
  fi
  status=$?
  secs=$(( $(date +%s) - start ))
  why=$(grep -iE "PANTHEON ERROR|CUDA error|integrity|VirtualGPU error|unsupported PTX|Verification: FAIL" "$log" | head -1 | cut -c1-100)
  if [ "$status" -eq 124 ]; then
    record "$w" TIMEOUT "$secs" "still running after ${VGPU_WL_TIMEOUT}s"; rc=1
  elif [ "$status" -ne 0 ]; then
    record "$w" FAIL "$secs" "exit $status${why:+: $why}"; rc=1
  elif [ -n "$why" ]; then
    record "$w" FAIL "$secs" "$why"; rc=1
  # A workload that ran must say something about what it measured. A silent
  # success is the failure this project cares about most.
  elif ! grep -qiE "throughput|bandwidth|latency|ops/s|GB/s|PANTHEON|Verification: PASS" "$log"; then
    record "$w" FAIL "$secs" "exited 0 but reported no measurement"; rc=1
  else
    record "$w" PASS "$secs" "$(grep -iE 'throughput|ops/s|GB/s|Verification' "$log" | head -1 | tr -s ' ' | cut -c1-80)"
  fi
done
exit $rc
INNER
)

# vgpu shell isolates itself with an unprivileged user namespace. Hardened
# kernels forbid that (Ubuntu 24.04 sets
# kernel.apparmor_restrict_unprivileged_userns=1, and CI runners and most
# containers restrict it too), so fall back to running without isolation. The
# shim still takes precedence through LD_LIBRARY_PATH; what is lost is the
# hiding of a real GPU, which matters only on a host that has one.
isolate=()
if ! unshare --user --map-root-user true >/dev/null 2>&1; then
  echo "note: unprivileged user namespaces unavailable; running with --no-isolate"
  isolate=(--no-isolate)
fi

VGPU_WL_PANTHEON="$pantheon" VGPU_WL_NAMES="$WORKLOADS" VGPU_WL_SKIP="$OUT_OF_SCOPE" \
VGPU_WL_LOGS="$logs" VGPU_WL_DURATION="$DURATION" VGPU_WL_MEMPCT="$MEM_PCT" \
VGPU_WL_ARGS="$ARGS" VGPU_WL_TIMEOUT="$TIMEOUT" \
  "$build/vgpu" shell --gpu "$GPU" --count "$COUNT" --vram-mb "$VRAM_MB" "${isolate[@]}" -y -c "$inner"
status=$?

results="$logs/results.tsv"
if [[ -s "$results" ]]; then
  pass=$(grep -c $'\tPASS\t' "$results"); skip=$(grep -c $'\tSKIP\t' "$results")
  bad=$(grep -cvE $'\t(PASS|SKIP)\t' "$results")
  echo
  echo "pantheon workloads on $COUNT x $GPU: $pass passed, $bad failed, $skip skipped"
  if [[ -n "$REPORT" ]]; then
    echo "$COUNT x ${GPU#nvidia/}, CUDA $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)" >"$logs/machine.txt"
    {
      echo "### $COUNT x $GPU, $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/CUDA \1/p' | head -1)"
      echo
      echo "$pass passed, $bad failed, $skip skipped (pantheon $(git -C "$pantheon" rev-parse --short HEAD 2>/dev/null))"
      echo
      if (( bad )); then
        echo "| Workload | Result | Seconds | Detail |"
        echo "| --- | --- | ---: | --- |"
        grep -vE $'\t(PASS|SKIP)\t' "$results" | awk -F'\t' '{ gsub(/\|/, "\\|", $4); printf "| %s | **%s** | %s | %s |\n", $1, $2, $3, $4 }'
        echo
      fi
      echo "<details><summary>All results</summary>"
      echo
      echo "| Workload | Result | Seconds | Detail |"
      echo "| --- | --- | ---: | --- |"
      awk -F'\t' '{ gsub(/\|/, "\\|", $4); printf "| %s | %s | %s | %s |\n", $1, $2, $3, $4 }' "$results"
      echo
      echo "</details>"
    } >"$logs/summary.md"
  fi
fi
[[ $status -eq 0 ]] && echo "pantheon workloads: all passed"
exit $status
