#!/usr/bin/env bash
# Interpreter throughput benchmark. Reports instructions/second so changes are
# comparable across machines and kernel shapes.
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
for n in 2000000; do
  out=$("$root/build/vgpu" demo vectoradd -n $n 2>&1 | tail -1)
  ms=$(sed -E 's/.*, ([0-9]+) ms\)/\1/' <<<"$out")
  echo "vectoradd n=$n: ${ms} ms"
done
