#!/usr/bin/env bash
# `vgpu info` shows a profile's numerics and what they rest on (nvidia/docs/profiles.md, "Numerics and provenance"): the
# tensor-core sum with the class of each value, the other keys a profile tags, the evidence, and the same in --json.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build="${VGPU_BUILD_DIR:-$root/build}"
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "no vgpu at $vgpu"; exit 1; }

# Nothing from the caller's environment may decide the answers.
for v in $(env | sed -n 's/^\(VGPU_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export HOME="$tmp/home"
mkdir -p "$HOME" "$tmp/work"
cd "$tmp/work"

fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
contains() {  # contains <name> <needle> <haystack>
  if [[ "$3" == *"$2"* ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected to contain: $2"; echo "      actual: $3"; fail=1; fi
}
lacks() {  # lacks <name> <needle> <haystack>
  if [[ "$3" != *"$2"* ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected not to contain: $2"; fail=1; fi
}
line() { grep -E "^ *$1" | head -1 | tr -s ' ' | sed 's/^ //'; }   # the first line that starts with a key, single-spaced

# ---- a measured card: each numerics value with its class, and the card in the evidence ------------------
out=$("$vgpu" info --gpu nvidia/t4 2>&1); rc=$?
expect "info runs" "0" "$rc"
expect "t4: the model and its class" "tensor_sum: ampere [measured]" "$(line 'tensor_sum:' <<<"$out")"
expect "t4: chunk" "tensor_sum_chunk: 8 [measured]" "$(line 'tensor_sum_chunk:' <<<"$out")"
expect "t4: the tf32 chunk is not its own measurement" "tensor_sum_chunk_tf32: 4 [inferred]" "$(line 'tensor_sum_chunk_tf32:' <<<"$out")"
expect "t4: grid" "tensor_sum_grid: 25 [measured]" "$(line 'tensor_sum_grid:' <<<"$out")"
contains "t4: the evidence names the card, driver, toolkit and date" \
  "numerics: card: Tesla T4 (this model), AWS; driver: 595.112.07; toolkit: CUDA 13.0; date: 2026-11-15." "$out"

# ---- nvidia/h100 (the default profile of a run): an H100 SXM5, inferred from the H100 PCIe ------------------------------------------
out=$("$vgpu" info --gpu nvidia/h100 2>&1)
expect "the profile a run uses when none is named" "VirtualGPU device: nvidia/h100" "$(head -1 <<<"$out")"
expect "h100: hopper, inferred" "tensor_sum: hopper [inferred]" "$(line 'tensor_sum:' <<<"$out")"
expect "h100: chunk 16" "tensor_sum_chunk: 16 [inferred]" "$(line 'tensor_sum_chunk:' <<<"$out")"
expect "h100: tf32 chunk 8" "tensor_sum_chunk_tf32: 8 [inferred]" "$(line 'tensor_sum_chunk_tf32:' <<<"$out")"
expect "h100: grid 26" "tensor_sum_grid: 26 [inferred]" "$(line 'tensor_sum_grid:' <<<"$out")"
contains "h100: inferred from the card that was run" "an H100 PCIe (nvidia/profiles/h100-pcie.yaml)" "$out"
expect "h100-pcie is the measured one" "tensor_sum: hopper [measured]" "$("$vgpu" info --gpu nvidia/h100-pcie 2>&1 | line 'tensor_sum:')"

# ---- Blackwell: b200 and gb200 sum by the Hopper rule, inferred from data others recorded on a B200 (RE-RECORDED 2026-11-16; before: ------
# ---- sequential, unknown, with the published model named as not in force); the other Blackwell profiles stay sequential, unknown ---------
for gpu in b200 gb200; do
  out=$("$vgpu" info --gpu nvidia/$gpu 2>&1)
  expect "$gpu: hopper, inferred" "tensor_sum: hopper [inferred]" "$(line 'tensor_sum:' <<<"$out")"
  expect "$gpu: chunk 16" "tensor_sum_chunk: 16 [inferred]" "$(line 'tensor_sum_chunk:' <<<"$out")"
  expect "$gpu: tf32 chunk 8" "tensor_sum_chunk_tf32: 8 [inferred]" "$(line 'tensor_sum_chunk_tf32:' <<<"$out")"
  expect "$gpu: grid 26" "tensor_sum_grid: 26 [inferred]" "$(line 'tensor_sum_grid:' <<<"$out")"
  lacks "$gpu: nothing left to opt in to" "tensor_sum_published" "$out"
  contains "$gpu: the evidence says whose data it is" "NOT run by us on a card" "$out"
done
for gpu in b300 vr200 thor gb10 rtx5090 rtx-pro-6000; do
  out=$("$vgpu" info --gpu nvidia/$gpu 2>&1)
  expect "$gpu: sequential, unknown" "tensor_sum: sequential [unknown]" "$(line 'tensor_sum:' <<<"$out")"
  lacks "$gpu: no chunk is printed for the sequential sum" "tensor_sum_chunk" "$out"
done
out=$("$vgpu" info --gpu nvidia/rtx5090 2>&1)
expect "rtx5090: the published model is still opt-in" "tensor_sum_published: hopper [inferred] (not in force: VGPU_TC_EXACT=published)" \
  "$(line 'tensor_sum_published:' <<<"$out")"
out=$("$vgpu" info --gpu nvidia/b300 2>&1)
lacks "b300: nothing published" "tensor_sum_published" "$out"

# ---- AMD: tagged limits on the two cards that were read, no numerics anywhere --------------------------------
out=$("$vgpu" info --gpu amd/rx6800 2>&1)
lacks "rx6800: no numerics" "numerics" "$out"
expect "rx6800: a limit read from the card" "limits.multiprocessors [measured]" "$(line 'limits.multiprocessors' <<<"$out")"
expect "rx6800: a limit no tool reports" "limits.max_blocks_per_sm [inferred]" "$(line 'limits.max_blocks_per_sm' <<<"$out")"
contains "rx6800: where the card's files are" "amd/registers/measurements/rx6800/characterize-hip.txt" "$out"
out=$("$vgpu" info --gpu amd/mi300x 2>&1)
lacks "mi300x: no numerics" "numerics" "$out"
lacks "mi300x: nothing tagged" "provenance" "$out"
lacks "mi300x: no evidence" "evidence" "$out"

# ---- every NVIDIA profile prints a class for its sum, and it is one of the four -------------------------------
bad=""; n=0
for gpu in $("$vgpu" list-gpus | grep '^nvidia/'); do
  n=$((n + 1))
  l=$("$vgpu" info --gpu "$gpu" 2>&1 | line 'tensor_sum:')
  [[ "$l" =~ ^tensor_sum:\ (sequential|ampere|hopper)\ \[(measured|documented|inferred|unknown)\]$ ]] || bad+=" $gpu"
  [[ "$l" == "tensor_sum: sequential [unknown]" || "$l" != *sequential* ]] || bad+=" $gpu(sequential-not-unknown)"
done
expect "38 NVIDIA profiles" "38" "$n"
expect "each prints its tensor sum and a class" "" "$bad"

# ---- --json -----------------------------------------------------------------
js=$("$vgpu" info --gpu nvidia/t4 --json 2>&1)
contains "json: numerics" '"tensor_sum": "ampere"' "$js"
contains "json: a number" '"tensor_sum_chunk_tf32": 4' "$js"
contains "json: provenance" '"numerics.tensor_sum_chunk_tf32": "inferred"' "$js"
contains "json: evidence" '"numerics": "card: Tesla T4 (this model)' "$js"
contains "json: a profile without the sections has empty objects" '"numerics": {},' "$("$vgpu" info --gpu amd/mi300x --json 2>&1)"
contains "json: ... provenance too" '"provenance": {},' "$("$vgpu" info --gpu amd/mi300x --json 2>&1)"
if command -v python3 >/dev/null; then
  bad=""
  for gpu in $("$vgpu" list-gpus); do
    "$vgpu" info --gpu "$gpu" --json 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
n, p, e = d["numerics"], d["provenance"], d["evidence"]
assert all("numerics." + k in p for k in n), "a numerics key without a class"
assert all(v in ("measured", "documented", "inferred", "unknown") for v in p.values()), "a class that is not one of the four"
assert (d["vendor"] == "nvidia") == ("tensor_sum" in n), "numerics on the wrong vendor"
for k, v in p.items():
    assert v == "unknown" or k in e or k.split(".")[0] in e, "a claim without evidence: " + k
' 2>/dev/null || bad+=" $gpu"
  done
  expect "json: every profile parses, and every numerics key has a class and every claim its evidence" "" "$bad"
else
  echo "skip  json parse (no python3)"
fi

exit $fail
