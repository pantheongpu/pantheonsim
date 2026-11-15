#!/usr/bin/env bash
# The math-function numerics programs (tools/numerics: mathfn, fp_examples, varity_cases) on the simulator: what the
# simulator prints is pinned, its two engines are held to each other, and two profiles of one architecture to each other.
# docs/numerics/math-functions.md says what the programs are and what a card is expected to print instead.
#
#   nvidia/tests/data/numerics_mathfn.txt        mathfn's lines (function, precision, class, count, hash), --n 4096, from the SASS of an
#   nvidia/tests/data/numerics_mathfn_fast.txt   sm_86 build on the rtx3060 profile: the default build and the -use_fast_math one.
#                                                The simulator's own bits, NOT a card's: the MUFU instructions are computed at host
#                                                precision here and from tables on a card, so a card differs on most lines. They are
#                                                pinned so that a change of any math result is seen and decided, not so that it is forbidden.
#   nvidia/tests/data/numerics_cases.txt         fp_examples', fp_examples_nofma's and varity_cases' lines (the same for -O0 and -O3)
#   nvidia/tests/data/numerics_engines.txt       the lines on which the PTX interpreter may differ from the SASS executor, each with its
#                                                reason. Every other line must be the same from both engines.
#
# Checked: the pinned lines (sm_86 on rtx3060, SASS); PTX against SASS line by line; an sm_75 build on t4 and on rtx2080ti, and the
# sm_86 build on a10g, against the same pins (the results do not depend on the profile or on the cubin's architecture); fp_examples
# exits 0 (every answer the paper documents); compare.py says "0 differ" for a run against itself and counts one result of one ulp
# in a doctored copy.
#
#   run_numerics_mathfn.sh             the checks above
#   run_numerics_mathfn.sh --update    rewrite the pinned files from this build's SASS run (numerics_engines.txt is edited by hand)
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
data="$root/nvidia/tests/data"
num="$root/tools/numerics"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
out="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_numerics_mathfn.XXXXXX")"
trap 'rm -rf "$out"' EXIT
NUMERICS_FLAGS="$(shim_sanitizer_nvcc_flags "$shim")" "$num/build.sh" nvidia "$out/bin" 86 75 > "$out/build.txt" 2>&1 || { echo "FAIL: the numerics programs do not build"; cat "$out/build.txt"; exit 1; }
if ! require_shim_libs "$shim" "$out/bin/sm_86/mathfn"; then exit 0; fi
MATHFN_N=4096 VARITY=0 "$num/run-sim.sh" "$out/bin" "${VGPU_BUILD_DIR:-$root/build}" "$out/run" \
  nvidia/rtx3060:86:sass nvidia/rtx3060:86:ptx nvidia/a10g:86:sass nvidia/t4:75:sass nvidia/t4:75:ptx nvidia/rtx2080ti:75:sass > "$out/run.txt" 2>&1
fails=0
if grep -v -e ' rc=0' "$out/run.txt" | grep -q .; then echo "FAIL: a program did not exit 0:"; grep -v -e ' rc=0' "$out/run.txt"; fails=1; fi
# A mathfn line without the results it prints: function, precision, class, count, hash.
keys() { grep ' hash=' "$1" | cut -d' ' -f1-5; }
cases() { for p in fp_examples fp_examples_nofma varity_cases.O0 varity_cases.O3; do sed "s/^/$p /" "$1/$p.txt"; done; }
ref="$out/run/nvidia.rtx3060.86.sass"
if [[ "${1:-}" == --update ]]; then
  keys "$ref/mathfn.txt" > "$data/numerics_mathfn.txt"
  keys "$ref/mathfn_fast.txt" > "$data/numerics_mathfn_fast.txt"
  cases "$ref" > "$data/numerics_cases.txt"
  echo "wrote numerics_mathfn.txt ($(wc -l < "$data/numerics_mathfn.txt") lines), numerics_mathfn_fast.txt, numerics_cases.txt"; exit 0
fi
# 1. The pinned lines, from every SASS run: the profile and the cubin's architecture do not matter.
for d in nvidia.rtx3060.86.sass nvidia.a10g.86.sass nvidia.t4.75.sass nvidia.rtx2080ti.75.sass; do
  for p in mathfn mathfn_fast; do
    if diff "$data/numerics_$p.txt" <(keys "$out/run/$d/$p.txt") > "$out/d.txt"; then echo "ok   $d $p ($(wc -l < "$data/numerics_$p.txt") lines)"
    else echo "FAIL $d $p: differs from numerics_$p.txt"; head -6 "$out/d.txt"; fails=1; fi
  done
  if diff "$data/numerics_cases.txt" <(cases "$out/run/$d") > "$out/d.txt"; then echo "ok   $d fp_examples, varity_cases"
  else echo "FAIL $d: differs from numerics_cases.txt"; head -6 "$out/d.txt"; fails=1; fi
done
# 2. The PTX interpreter against the SASS executor: the same lines, but for the ones numerics_engines.txt lists.
for a in rtx3060.86 t4.75; do
  for p in mathfn mathfn_fast; do
    diff <(keys "$out/run/nvidia.$a.sass/$p.txt") <(keys "$out/run/nvidia.$a.ptx/$p.txt") | grep '^<' | cut -d' ' -f2-4 | sed "s/^/$p /" > "$out/differ.txt"
    grep -v '^#' "$data/numerics_engines.txt" | grep "^$p " | cut -d' ' -f1-4 > "$out/allowed.txt"
    if diff "$out/allowed.txt" "$out/differ.txt" > "$out/d.txt"; then echo "ok   $a $p: PTX is SASS but for $(wc -l < "$out/allowed.txt") listed lines"
    else echo "FAIL $a $p: the lines on which PTX differs from SASS are not the listed ones (< listed, > differ now):"; head -10 "$out/d.txt"; fails=1; fi
  done
  if diff <(cases "$out/run/nvidia.$a.sass") <(cases "$out/run/nvidia.$a.ptx") > "$out/d.txt"; then echo "ok   $a fp_examples, varity_cases: PTX is SASS"
  else echo "FAIL $a: fp_examples or varity_cases differ between the engines"; head -6 "$out/d.txt"; fails=1; fi
done
# 3. compare.py: a run against itself, and against a copy with one result moved by one ulp.
if command -v python3 >/dev/null 2>&1; then
  if python3 "$num/compare.py" "$ref" "$ref" > "$out/c.txt" 2>&1 && ! grep -v ' 0 differ' "$out/c.txt" | grep -q 'differ'; then echo "ok   compare.py: a run against itself"
  else echo "FAIL compare.py: a run differs from itself"; head -5 "$out/c.txt"; fails=1; fi
  mkdir -p "$out/doctored"; cp "$ref"/*.txt "$out/doctored/"
  # (sin f32 edges prints sin(1) third: 3f800000=<bits>; the copy's last hex digit is changed by one and its hash by a digit)
  python3 - "$out/doctored/mathfn.txt" <<'EOF'
import re, sys
path = sys.argv[1]
lines = open(path).read().split('\n')
for i, l in enumerate(lines):
    if l.startswith('sin f32 edges '):
        m = re.search(r' 3f800000=([0-9a-f]{8})', l)
        l = l.replace(' 3f800000=' + m.group(1), ' 3f800000=%08x' % (int(m.group(1), 16) ^ 1))
        h = re.search(r'hash=([0-9a-f]{16})', l).group(1)
        lines[i] = l.replace(h, '%016x' % (int(h, 16) ^ 1))
open(path, 'w').write('\n'.join(lines))
EOF
  python3 "$num/compare.py" "$ref" "$out/doctored" > "$out/c.txt" 2>&1; rc=$?
  if (( rc == 1 )) && grep -q 'sin        f32 edges     1 of 4 differ: 1 ulp 1, 2 ulp 0, more 0 (max 1 ulp)' "$out/c.txt"; then echo "ok   compare.py: one result of one ulp is counted"
  else echo "FAIL compare.py: the doctored result is not reported (rc=$rc)"; grep sin "$out/c.txt" | head -3; fails=1; fi
else
  echo "skip compare.py (no python3)"
fi
(( fails == 0 )) && echo PASS || { echo FAIL; exit 1; }
