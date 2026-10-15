#!/usr/bin/env bash
# Runs the external program corpora (HeCBench, Rodinia, PolyBench/GPU, CUDA Samples and the
# pantheon workloads) on two simulator builds, one program at a time, and keeps every run's
# stdout, stderr, exit code, per-launch counters and output files for tools/verify/summarize.py.
#
#   tools/verify/run_corpora.sh --a <build dir A> --b <build dir B> --out <results dir> [options]
#
# A is the baseline (for example a fresh build of origin/main), B the build under test. Each
# program is run once per configuration, all configurations back to back, so a run-to-run
# difference shows up between A and A2 (the same build run twice) rather than being blamed on B.
# See tools/verify/README.md for the corpus set-up, the configurations and how to read the report.
#
# Options:
#   --corpora LIST   comma list of hecbench,rodinia,polybench,samples,pantheon (default: all five)
#   --configs LIST   comma list of NAME=BUILD[+VAR=value...] (default below); BUILD is A or B
#   --only REGEX     only programs whose "corpus/name" matches (extended regex)
#   --force          rerun runs that already have an rc file
#   --list           print the programs and exit
#   --dry-run        print the command for each run, run nothing
#   --max-load N     wait (checking every --pause seconds) while the 1-minute load average is above N
#   --pause SECS     see --max-load (default 60)
#   --skip-timeouts CFG  do not run the other configurations of a program whose CFG run hit the timeout
#                    (it would cost a full timeout each time); run them later without this option
#
# Default configurations:
#   A   = A                      baseline, SASS engine
#   A2  = A                      the same build again: finds programs that differ from themselves
#   B   = B                      build under test, SASS engine
#   Bt  = B+VGPU_TIER=1+VGPU_TIER_STATS=1   the opt-in pre-decoded tier
#   Ap  = A+VGPU_SASS=0          baseline, PTX engine
#   Bp  = B+VGPU_SASS=0          build under test, PTX engine
#
# Every run: VGPU_THREADS=1 (strictly ordered blocks), VGPU_COUNTERS=1, VGPU_COUNTERS_FILE, the RTX 3060
# profile, nice -n 10, ulimit -v 12582912, timeout (per corpus, scaled by TIMEOUT_SCALE).
# Nothing is run in parallel. Environment overrides: HECBENCH_DIR, HECBENCH_LIST, RODINIA_DIR,
# POLYBENCH_DIR, SAMPLES_DIR, SAMPLES_LIST, SAMPLES_TIMEOUT (seconds, default 120), PANTHEON_BIN (directory of built workloads), TIMEOUT_SCALE,
# WL_SECONDS (workload duration, default 2), WL_ARGS.
set -uo pipefail

A_BUILD=""; B_BUILD=""; OUT=""; CORPORA="hecbench,rodinia,polybench,samples,pantheon"
CONFIGS="A=A,A2=A,B=B,Bt=B+VGPU_TIER=1+VGPU_TIER_STATS=1,Ap=A+VGPU_SASS=0,Bp=B+VGPU_SASS=0"
ONLY=""; SKIPTO=""; FORCE=0; LIST=0; DRY=0; MAXLOAD=0; PAUSE=60
while [[ $# -gt 0 ]]; do
  case "$1" in
    --a) A_BUILD="$2"; shift 2;;       --b) B_BUILD="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;         --corpora) CORPORA="$2"; shift 2;;
    --configs) CONFIGS="$2"; shift 2;; --only) ONLY="$2"; shift 2;;
    --force) FORCE=1; shift;;          --list) LIST=1; shift;;
    --dry-run) DRY=1; shift;;          --max-load) MAXLOAD="$2"; shift 2;;
    --pause) PAUSE="$2"; shift 2;;
    --skip-timeouts) SKIPTO="$2"; shift 2;;
    -h|--help) sed -n '2,38p' "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
[[ -n "$OUT" ]] || { echo "--out is required" >&2; exit 2; }
if (( ! LIST )); then
  [[ -d "$A_BUILD/shim" && -d "$B_BUILD/shim" ]] || { echo "--a and --b must be build directories with a shim/ directory" >&2; exit 2; }
  A_BUILD="$(cd "$A_BUILD" && pwd)"; B_BUILD="$(cd "$B_BUILD" && pwd)"
fi
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"

HECBENCH_DIR="${HECBENCH_DIR:-$HOME/.cache/HeCBench}"
HECBENCH_LIST="${HECBENCH_LIST:-$HECBENCH_DIR/reduced.txt}"
RODINIA_DIR="${RODINIA_DIR:-$HOME/.cache/gpu-rodinia/bin}"
POLYBENCH_DIR="${POLYBENCH_DIR:-$HOME/.cache/polybenchGpu/bin}"
SAMPLES_DIR="${SAMPLES_DIR:-$HOME/.cache/cuda-samples/build}"
SAMPLES_LIST="${SAMPLES_LIST:-$SAMPLES_DIR/exes.txt}"
PANTHEON_BIN="${PANTHEON_BIN:-$OUT/pantheon-bin}"
TIMEOUT_SCALE="${TIMEOUT_SCALE:-1}"
WL_SECONDS="${WL_SECONDS:-2}"
WL_ARGS="${WL_ARGS:---kernel_loops 2 --warmup_iters 1 --grid_size 4}"

# --- program lists: one line per program: name <TAB> mode <TAB> cwd <TAB> timeout <TAB> command
# mode: inplace = run in cwd (files created there are hashed, then removed)
#       fresh   = run in an empty directory made for the run (commands use absolute paths)
#       timed   = a time-based workload (fresh directory; the summary compares kernels and
#                 verification lines, not totals)
list_hecbench() {
  local src="$HECBENCH_DIR/src" d args exe prog
  while IFS='|' read -r d args; do
    [[ -n "$d" ]] || continue
    # programs whose full-size default runs for minutes: use the small variants made for the simulator
    [[ "$d" == aidw-cuda ]] && args="4 1 1"
    if [[ "$d" == aligned-types-cuda && -x "$src/aligned-types-cuda/small" ]]; then
      exe=small
    elif [[ "$d" == binomial-cuda && -x "$src/binomial-small/main" ]]; then
      printf '%s\t%s\t%s\t%s\t%s\n' "binomial-small" inplace "$src/binomial-small" $((120*TIMEOUT_SCALE)) "./main $args"; continue
    else
      prog=$(awk -F'= *' '/^program/{print $2; exit}' "$src/$d/Makefile" 2>/dev/null)
      exe=$(ls "$src/$d" 2>/dev/null | grep -xE "main|${prog:-main}" | head -1)
    fi
    [[ -n "$exe" && -x "$src/$d/$exe" ]] || { echo "# missing binary: $d" >&2; continue; }
    printf '%s\t%s\t%s\t%s\t%s\n' "$d" inplace "$src/$d" $((120*TIMEOUT_SCALE)) "./$exe $args"
  done < "$HECBENCH_LIST"
}
list_rodinia() {
  local R="$RODINIA_DIR" D="$RODINIA_DIR/data" t=$((600*TIMEOUT_SCALE))
  local rows=(
    "backprop|$R/backprop 16384"
    "gaussian|$R/gaussian -s 64"
    "lud|$R/lud -s 256 -v"
    "needle|$R/needle 256 10"
    "pathfinder|$R/pathfinder 1000 50 20"
    "srad_v2|$R/srad_v2 64 64 0 31 0 31 0.5 2"
    "particlefilter_float|$R/particlefilter_float -x 64 -y 64 -z 5 -np 1000"
    "sc_gpu|$R/sc_gpu 10 20 16 1024 1024 1000 none out.txt 1"
    "hotspot|$R/hotspot 64 2 20 $D/temp_64 $D/power_64 output.out"
    "bfs|$R/bfs $D/graph4096.txt"
    "nn|$R/nn $D/filelist4 -r 10 -lat 30 -lng 90"
    "lavaMD|$R/lavaMD -boxes1d 4"
  )
  local r
  for r in "${rows[@]}"; do printf '%s\t%s\t%s\t%s\t%s\n' "${r%%|*}" fresh "$R" "$t" "${r#*|}"; done
}
list_polybench() {
  local p name
  for p in "$POLYBENCH_DIR"/*; do
    [[ -f "$p" && -x "$p" ]] || continue
    name="$(basename "$p")"; [[ "$name" == *.* ]] && continue
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" fresh "$POLYBENCH_DIR" $((300*TIMEOUT_SCALE)) "$p"
  done
}
list_samples() {
  local exe name
  while read -r exe; do
    [[ -n "$exe" ]] || continue
    name="$(basename "$exe")"
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" inplace "$SAMPLES_DIR/$(dirname "$exe")" $((${SAMPLES_TIMEOUT:-120}*TIMEOUT_SCALE)) "./$name"
  done < "$SAMPLES_LIST"
}
list_pantheon() {
  local b name extra
  [[ -d "$PANTHEON_BIN" ]] || { echo "# no pantheon workload binaries in $PANTHEON_BIN (see README)" >&2; return; }
  for b in "$PANTHEON_BIN"/*; do
    [[ -f "$b" && -x "$b" ]] || continue
    name="$(basename "$b")"; extra=""
    # not CUDA (OptiX) and the NVENC shim target, which this verification does not build: as in run_pantheon_workloads.sh
    case "$name" in rt_virus|media_enc_virus) continue;; esac
    case "$name" in mma_virus|omni_virus) extra="--gemm_size 256";; esac
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" timed "$PANTHEON_BIN" $((120*TIMEOUT_SCALE)) "$b 0 $WL_SECONDS 2 $WL_ARGS $extra"
  done
}

all_programs() {
  local c line
  for c in ${CORPORA//,/ }; do
    "list_$c" | while IFS= read -r line; do printf '%s\t%s\n' "$c" "$line"; done
  done
}

if (( LIST )); then all_programs | awk -F'\t' '{printf "%s/%s\t%s\ttimeout=%s\t%s\n", $1, $2, $3, $5, $6}'; exit 0; fi

# --- one run
build_of() { case "$1" in A) echo "$A_BUILD";; B) echo "$B_BUILD";; *) echo "bad build $1" >&2; return 1;; esac; }

run_one() {   # corpus name mode cwd timeout cmd cfgname cfgspec
  local corpus="$1" name="$2" mode="$3" cwd="$4" to="$5" cmd="$6" cfg="$7" spec="$8"
  local rdir="$OUT/$corpus/$name/$cfg"
  if [[ -f "$rdir/rc" && $FORCE -eq 0 ]]; then return 0; fi
  local build="${spec%%+*}" extra=""; [[ "$spec" == *+* ]] && extra="${spec#*+}"
  local bdir; bdir="$(build_of "$build")" || return 1
  local envs=(PATH="$PATH" HOME="$HOME" LANG=C LC_ALL=C LD_LIBRARY_PATH="$bdir/shim"
              VGPU_GPU=nvidia/rtx3060 VGPU_THREADS=1 VGPU_COUNTERS=1 VGPU_COUNTERS_FILE="$rdir/counters.jsonl")
  if [[ "$corpus" == pantheon ]]; then
    envs+=(VGPU_DEVICE_COUNT=2 VGPU_VRAM_MB=1024 VGPU_MAX_STEPS=17179869184)
  fi
  local kv; for kv in ${extra//+/ }; do envs+=("$kv"); done
  if (( DRY )); then echo "[$corpus/$name/$cfg] (cd $cwd && env ${envs[*]} timeout $to $cmd)"; return 0; fi
  if (( MAXLOAD > 0 )); then
    while awk -v m="$MAXLOAD" '{exit !($1 > m)}' /proc/loadavg; do sleep "$PAUSE"; done
  fi
  rm -rf "${rdir:?}"; mkdir -p "$rdir"
  local work="$cwd" pre="$rdir/.pre" post="$rdir/.post"
  if [[ "$mode" != inplace ]]; then work="$rdir/work"; mkdir -p "$work"; fi
  ( cd "$work" && find . -type f -printf '%p\t%s\t%T@\n' | sort > "$pre" )
  local load; load="$(cut -d' ' -f1-3 /proc/loadavg)"
  local t0 t1; t0=$(date +%s.%N)
  ( cd "$work" && env -i "${envs[@]}" nice -n 10 timeout -k 10 "$to" \
      bash -c "ulimit -v 12582912; exec $cmd" > "$rdir/stdout" 2> "$rdir/stderr" < /dev/null )
  local rc=$?
  t1=$(date +%s.%N)
  # files the program created or changed: hashed, and kept when small
  ( cd "$work" && find . -type f -printf '%p\t%s\t%T@\n' | sort > "$post" )
  mkdir -p "$rdir/files"
  local f
  while IFS= read -r f; do
    [[ -n "$f" ]] || continue
    ( cd "$work" && md5sum "$f" ) >> "$rdir/files.md5"
    if (( $(stat -c %s "${work:?}/$f") <= 2097152 )); then
      mkdir -p "$rdir/files/$(dirname "$f")"; cp -p "${work:?}/$f" "$rdir/files/$f"
    fi
    # in-place runs: remove what the run created, so every configuration starts from the same directory
    if [[ "$mode" == inplace ]] && ! cut -f1 "$pre" | grep -q -F -x -e "$f"; then rm -f "${work:?}/${f:?}"; fi
  done < <(comm -13 "$pre" "$post" | cut -f1)
  [[ -f "$rdir/files.md5" ]] && sort -k2 -o "$rdir/files.md5" "$rdir/files.md5"
  rm -f "$pre" "$post"
  # fresh and timed runs used an empty directory of their own: its outputs are the copies in files/
  [[ "$mode" != inplace ]] && rm -rf "${rdir:?}/work"
  local secs; secs=$(echo "$t1 - $t0" | bc -l)
  { echo "corpus=$corpus"; echo "program=$name"; echo "mode=$mode"; echo "config=$cfg"; echo "spec=$spec"; echo "cmd=$cmd"
    echo "cwd=$cwd"; echo "timeout=$to"; echo "rc=$rc"; printf 'secs=%.2f\n' "$secs"
    echo "load_at_start=$load"; echo "load_at_end=$(cut -d' ' -f1-3 /proc/loadavg)"
    echo "date=$(date -Is)"; } > "$rdir/meta"
  echo "$rc" > "$rdir/rc"
  printf '%-10s %-34s %-4s rc=%-3s %7.1fs load=%s\n' "$corpus" "$name" "$cfg" "$rc" "$secs" "$load"
}

IFS=',' read -r -a CFGS <<< "$CONFIGS"
echo "# run_corpora: A=$A_BUILD B=$B_BUILD out=$OUT start=$(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg)"
all_programs > "$OUT/programs.tsv"
while IFS=$'\t' read -r corpus name mode cwd to cmd; do
  [[ -z "$ONLY" || "$corpus/$name" =~ $ONLY ]] || continue
  for c in "${CFGS[@]}"; do
    if [[ -n "$SKIPTO" && "${c%%=*}" != "$SKIPTO" && -f "$OUT/$corpus/$name/$SKIPTO/rc" ]] \
       && [[ "$(cat "$OUT/$corpus/$name/$SKIPTO/rc")" == 124 ]]; then continue; fi
    run_one "$corpus" "$name" "$mode" "$cwd" "$to" "$cmd" "${c%%=*}" "${c#*=}"
  done
done < "$OUT/programs.tsv"
echo "# done $(date -Is)"
