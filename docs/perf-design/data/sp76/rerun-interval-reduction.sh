#!/usr/bin/env bash
# Round sp76 follow-up, the card part: why did samples/interval and samples/reduction read 9 % and 7 % slower in the one-run `rest` pass of
# card-run.sh with the same decisions in both forms? Run on the rig (one A10G), from the root of the source tree, corpora in ~/corp, the build
# of card-run.sh (SP76_BUILD). Not run when written. About 15 minutes (two programs of 7 to 10 s, three forms, 15 runs a form).
#   docs/perf-design/data/sp76/rerun-interval-reduction.sh [out dir]
# Three forms of turbo mode, interleaved run by run in a rotating order (so no form always follows the same other):
#   new  the default (the rule of Round sp76), environment as the `turbo` form has it
#   d0   VGPU_HOST_GPU_SAMPLE_STARTED=0 (the rule before it)
#   new1 VGPU_HOST_GPU_SAMPLE_STARTED=1: the rule of `new` (the switch reads only "0"), but an environment of d0's size. If new1 is as slow
#        as new, the 9 % is not the environment (the size of the environment tilted particlefilter's race in Round sp76); if new1 is as fast as
#        d0, it is.
# What to read: out/summary.txt (median, min and max of each form's wall time, and each form over d0), out/decisions.txt (the mode's decision
# lines of one logged run of each form, timings cut: they must be the same for new and d0, apart from the switch's own line), out/cpu.txt (user + sys
# time of every run: a form whose wall time is longer with the same CPU time waited, one with more CPU time worked).
# Verdict: a form over d0 by more than the spread of d0 in 15 runs, in the same direction in both programs, is a cost of the rule even with
# identical decisions (then read the logged run for a sample that the decision line of a kept launch does not show, and the page faults after a
# fork: `host_gpu_sample:` lines with fork_s); within the spread, the single-run reading was noise, the same as FDTD-2D's and
# segmentationTreeThrust's in rerun.md.
set -u
out=${1:-$HOME/sp76-out-interval}
build=${SP76_BUILD:-$HOME/sp76-build}
corp=${SP76_CORP:-$HOME/corp}
runs=${SP76_RUNS:-15}
mkdir -p "$out"
export PATH=/usr/local/cuda-13.0/bin:$PATH
S=$corp/samples
: > "$out/times.txt"; : > "$out/cpu.txt"; : > "$out/decisions.txt"
formenv() {   # <form> -> the variables of the form
  case $1 in
    new) echo "VGPU_HOST_GPU=turbo" ;;
    d0) echo "VGPU_HOST_GPU=turbo VGPU_HOST_GPU_SAMPLE_STARTED=0" ;;
    new1) echo "VGPU_HOST_GPU=turbo VGPU_HOST_GPU_SAMPLE_STARTED=1" ;;
  esac
}
one() {   # <program> <exe> <form> <extra env...>: one timed run, wall and cpu seconds to times.txt and cpu.txt, output compared with the first
  local prog=$1 exe=$2 form=$3; shift 3
  local dir; dir=$(dirname "$exe")
  local t; t=$( cd "$dir" && /usr/bin/time -f "%e %U %S" env VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH="$build/shim" $(formenv "$form") "$@" timeout -s KILL 300 "$exe" 2>&1 >"$out/last.out" | tail -1 )
  set -- $t
  echo "$prog $form $1" >> "$out/times.txt"
  echo "$prog $form wall=$1 user=$2 sys=$3" >> "$out/cpu.txt"
  [[ -f "$out/ref-$prog.out" ]] || cp "$out/last.out" "$out/ref-$prog.out"
  cmp -s "$out/last.out" "$out/ref-$prog.out" || echo "$prog $form: OUTPUT DIFFERS from the first run" >> "$out/cpu.txt"
}
for prog in interval reduction; do
  exe=$(find "$S" -type f -name "$prog" -perm -u+x | head -1)
  [[ -n "$exe" ]] || { echo "no $prog under $S"; exit 1; }
  # one logged run of each form: the decisions (timings cut), to compare
  for f in new d0 new1; do
    ( cd "$(dirname "$exe")" && env VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH="$build/shim" $(formenv $f) VGPU_HOST_GPU_LOG=1 timeout -s KILL 300 "$exe" 2>&1 >/dev/null \
      | grep -E '^host_gpu' | sed -E 's/[0-9]+\.[0-9]+ ?(s|ms|us)\b/T/g; s/(card_s|sim_s|fork_s|wall_s|[a-z_]*_s)=[0-9.e+-]+/\1=T/g' > "$out/dec-$prog-$f.txt" )
  done
  { echo "== $prog: decision lines with timings cut, new against d0, then new1 against new"; diff "$out/dec-$prog-new.txt" "$out/dec-$prog-d0.txt" | head -20; echo "-- new1:"; diff "$out/dec-$prog-new.txt" "$out/dec-$prog-new1.txt" | head -20; } >> "$out/decisions.txt"
  forms=(new d0 new1)
  for i in $(seq 0 $((runs - 1))); do
    for k in 0 1 2; do one "$prog" "$exe" "${forms[$(( (i + k) % 3 ))]}"; done
  done
done
python3 -I - "$out" <<'PY' | tee "$out/summary.txt"
import statistics, sys, collections
t = collections.defaultdict(list)
for line in open(sys.argv[1] + "/times.txt"):
    p, f, w = line.split(); t[(p, f)].append(float(w))
for p in ("interval", "reduction"):
    base = statistics.median(t[(p, "d0")])
    for f in ("d0", "new", "new1"):
        v = t[(p, f)]
        print("%-10s %-5s n=%d median %.3f s  min %.3f  max %.3f  over d0 %.3f" % (p, f, len(v), statistics.median(v), min(v), max(v), statistics.median(v) / base))
PY
echo "out: $out"
