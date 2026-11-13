#!/usr/bin/env bash
# Round sp65: the offline tables, from the traces of trace_corpus.py (server3, no card).
#   docs/perf-design/data/sp65/analyze.sh <trace dir> <out dir>
set -u
T=${1:?traces}
O=${2:?out}
here=$(cd "$(dirname "$0")" && pwd)
R="python3 $here/residency_replay.py --fit $here/../sp59/bench.json"
mkdir -p "$O"
turbo=$(ls "$T"/*.turbo.trace)
# exact mode: the exact traces where there is one, else the turbo trace of a program whose kernels both modes admit alike (trace_corpus.py's list)
exact=""
for f in $turbo; do
  e=${f%.turbo.trace}.exact.trace
  [[ -f $e ]] && exact="$exact $e"
done
for m in turbo exact; do
  files=$turbo; [[ $m == exact ]] && files=$exact
  $R --rule pull $files --json "$O/replay-$m.json" > "$O/replay-$m.md"
  $R --q1 $files --json "$O/q1-$m.json" > "$O/q1-$m.md"
  $R --why $files > "$O/why-$m.md"
  $R --changes --rule pull $files > "$O/changes-$m.md"
  for v in pullpush pullnf; do $R --rule $v $files > "$O/replay-$m-$v.md"; done
  $R --rule pull --kdefault 10e-6 $files > "$O/replay-$m-k10us.md"
done
python3 "$here/q2_card_time.py" --bench "$here/../sp59/bench.json" --traces "$T" > "$O/q2-turbo.md"
python3 "$here/sim_scale.py" --bench "$here/../sp59/bench.json" --traces "$T" > "$O/sim-scale.txt"
