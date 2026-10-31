#!/bin/bash
# The PGO A/B (inside drx.sh, repo root /w): plain build vs PGO build, alternating, pinned to CPU 6. Output in /results/small-pgo/.
L=/results/small-pgo; cd /w
A=${1:-build-plain} B=${2:-build-pgo} tag=${3:-a}
uptime > $L/ab-$tag.start
tools/bench/bench.sh --ab $A $B --engines sass,ptx --runs 9 --cpus 6 --json $L/ab-$tag-tier0.json > $L/ab-$tag-tier0.txt 2>&1
tools/bench/bench.sh --ab $A $B --engines sass --runs 9 --cpus 6 --env VGPU_TIER=1 --json $L/ab-$tag-tier1.json > $L/ab-$tag-tier1.txt 2>&1
tools/bench/bench.sh --check-counters --build $B > $L/counters-$tag.txt 2>&1
tools/bench/bench.sh --check-counters --build $B --env VGPU_TIER=1 >> $L/counters-$tag.txt 2>&1
bash docs/perf-design/data/round10/r10small-nn-ab.sh 6 ${NNRUNS:-5} $A $B nn > $L/nn-$tag.txt 2>&1
uptime > $L/ab-$tag.end; echo DONE > $L/ab-$tag.done
