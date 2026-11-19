#!/usr/bin/env bash
# check_leak_sweep.py finds what it is for: a clean log passes; a leak that is not in the known list fails and names the resource and
# the figure; a listed leak passes; one that grew or shrank fails; a second resource beside a listed one fails; a listed entry whose test
# no longer leaks fails as stale; a test with no log is not judged; a malformed list is refused; --update writes a list that passes.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
lint="$here/check_leak_sweep.py"
command -v python3 >/dev/null || { echo "SKIP: no python3"; exit 0; }
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
fail=0
expect() {  # expect <name> <wanted exit status> <got>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else echo "FAIL  $1 (exit $3, wanted $2)"; fail=1; fi
}
clean='vgpu-leak-check: device_bytes=0 device_allocs=0 pinned_allocs=0 pinned_bytes=0 fds=+0 threads=+0 maps=+6 rss_kb=14000 exe=app -> clean'
leak() {   # leak <device bytes> <device allocs> <the names the line ends with>
  echo "vgpu-leak-check: device_bytes=$1 device_allocs=$2 pinned_allocs=0 pinned_bytes=0 fds=+1 threads=+0 maps=+6 rss_kb=14000 exe=app -> LEAK $3"
}
run() { python3 "$lint" "$tmp/logs" "$tmp/known.txt" >"$tmp/out" 2>&1; echo $?; }
reset() { rm -rf "$tmp/logs"; mkdir -p "$tmp/logs"; : > "$tmp/known.txt"; }

reset; echo "$clean" > "$tmp/logs/e2e_a.log"
expect "a clean log passes" 0 "$(run)"

reset; leak 4096 1 device_bytes,device_allocs > "$tmp/logs/e2e_a.log"
expect "an unlisted leak fails" 1 "$(run)"
grep -q "NEW LEAK in e2e_a: device_bytes=4096, not listed" "$tmp/out" && echo "ok    ... and names the resource and the figure" || { echo "FAIL  the failure does not name the resource"; fail=1; }

reset; leak 4096 1 device_bytes,device_allocs > "$tmp/logs/e2e_a.log"; echo "e2e_a device_bytes=4096,device_allocs=1  # the program never frees its buffer" > "$tmp/known.txt"
expect "a listed leak passes" 0 "$(run)"

reset; leak 8192 2 device_bytes,device_allocs > "$tmp/logs/e2e_a.log"; echo "e2e_a device_bytes=4096,device_allocs=1  # the program never frees its buffer" > "$tmp/known.txt"
expect "a listed test that now leaks more fails" 1 "$(run)"
grep -q "listed for 4096" "$tmp/out" && echo "ok    ... and says what it was listed for" || { echo "FAIL  no listed figure"; fail=1; }

reset; leak 2048 1 device_bytes,device_allocs > "$tmp/logs/e2e_a.log"; echo "e2e_a device_bytes=4096,device_allocs=1  # the program never frees its buffer" > "$tmp/known.txt"
expect "a listed test that now leaks less fails (the figure only goes down)" 1 "$(run)"

reset; leak 4096 1 device_bytes,fds > "$tmp/logs/e2e_a.log"; echo "e2e_a device_bytes=4096  # the program never frees its buffer" > "$tmp/known.txt"
expect "a second resource beside a listed one fails" 1 "$(run)"

reset; echo "$clean" > "$tmp/logs/e2e_a.log"; echo "e2e_a device_bytes=4096  # the program never frees its buffer" > "$tmp/known.txt"
expect "a listed test that no longer leaks fails as stale" 1 "$(run)"
grep -q "STALE entry for e2e_a" "$tmp/out" && echo "ok    ... and says so" || { echo "FAIL  no stale message"; fail=1; }

reset; echo "e2e_gone device_bytes=4096  # skipped here" > "$tmp/known.txt"
expect "a listed test with no log is not judged" 0 "$(run)"

reset; { echo "$clean"; leak 4096 1 device_bytes; } > "$tmp/logs/e2e_a.log"
expect "one leaking process among several fails" 1 "$(run)"

reset; echo "e2e_a device_bytes=4096" > "$tmp/known.txt"
expect "a list line without a reason is refused" 1 "$(run)"

reset; leak 4096 1 device_bytes,device_allocs > "$tmp/logs/e2e_a.log"
python3 "$lint" --update "$tmp/logs" > "$tmp/known.txt"
expect "--update writes a list the check accepts" 0 "$(run)"
exit "$fail"
