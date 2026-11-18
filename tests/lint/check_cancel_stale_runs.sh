#!/usr/bin/env bash
# ci/cancel-stale-runs.sh against a fake gh: a run of a merged or superseded commit is cancelled, a run of an open
# PR's head is kept, a failed listing cancels nothing, and a dry run only prints.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
script="$root/ci/cancel-stale-runs.sh"
[[ -x $script ]] || { echo "FAIL: $script is not executable"; exit 1; }
d="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_cancel_stale.XXXXXX")"
trap 'rm -rf "$d"' EXIT
mkdir -p "$d/bin"
cat > "$d/bin/gh" <<'EOF'
#!/usr/bin/env bash
# the canned answers: runs_<status> hold "id branch sha" lines, prs holds "branch sha" lines
case "$1 $2" in
  "run list") for a in "$@"; do [[ ${prev:-} == --status ]] && st=$a; prev=$a; done
              [[ -e $FAKE_DIR/fail_runs ]] && exit 1
              cat "$FAKE_DIR/runs_$st" 2>/dev/null; exit 0 ;;
  "pr list")  [[ -e $FAKE_DIR/fail_prs ]] && exit 1; cat "$FAKE_DIR/prs"; exit 0 ;;
  "run cancel") echo "$3" >> "$FAKE_DIR/cancelled"; exit 0 ;;
esac
echo "fake gh: unexpected $*" >&2; exit 1
EOF
chmod +x "$d/bin/gh"
fails=0
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }
run() { env -i HOME="$d" PATH="$d/bin:/usr/bin:/bin" FAKE_DIR="$d" GH_REPO=acme/sim "$@" "$script"; }
reset() { rm -f "$d"/runs_* "$d"/prs "$d"/cancelled "$d"/fail_*; : > "$d/prs"; }

reset
printf 'feat aaaa1111\nmerged bbbb2222\n' > "$d/prs"; sed -i '/^merged /d' "$d/prs"   # only "feat" is open, at aaaa1111
printf '1 feat aaaa1111\n2 merged bbbb2222\n3 feat oldold00\n' > "$d/runs_in_progress"
printf '4 feat aaaa1111\n5 gone cccc3333\n' > "$d/runs_queued"
out="$(run)"; rc=$?
[[ $rc == 0 ]] || fail "exit $rc: $out"
expect="2 3 5"; got="$(sort -n "$d/cancelled" | tr '\n' ' ' | sed 's/ $//')"
[[ $got == "$expect" ]] || fail "cancelled '$got', wanted '$expect': $out"
[[ $out == *"3 stale pull_request run(s) cancelled, 2 kept"* ]] || fail "the summary: $out"

reset; printf 'feat aaaa1111\n' > "$d/prs"; printf '2 merged bbbb2222\n' > "$d/runs_queued"
out="$(run DRY_RUN=true)"
[[ ! -e $d/cancelled && $out == *"would cancel run 2"* ]] || fail "dry run: $out"

reset; printf '2 merged bbbb2222\n' > "$d/runs_queued"; touch "$d/fail_prs"
out="$(run)"; [[ ! -e $d/cancelled && $out == *"nothing done"* ]] || fail "a failed PR listing cancelled runs: $out"
reset; printf '2 merged bbbb2222\n' > "$d/runs_queued"; touch "$d/fail_runs"
out="$(run)"; [[ ! -e $d/cancelled && $out == *"nothing done"* ]] || fail "a failed run listing cancelled runs: $out"

reset; out="$(run)"; [[ ! -e $d/cancelled ]] || fail "nothing queued, yet something was cancelled"
(( fails == 0 )) && echo "PASS: ci/cancel-stale-runs.sh cancels the runs of merged and superseded commits only"
exit $(( fails > 0 ))
