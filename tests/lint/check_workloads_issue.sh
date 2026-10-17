#!/usr/bin/env bash
# ci/track-workloads-issue.sh: the "workloads failing on main" issue.
#
# Runs the script against a fake `gh` that records every call and answers from
# canned state, so nothing touches the network. What it pins:
#   success    closes the open issue; with none open it does nothing
#   failure    opens an issue (none open), or comments on the open one
#   skipped    counts as a failure (the machine jobs never ran)
#   cancelled  touches nothing, whether or not an issue is open: a run a later
#              push cancelled says nothing about main (pantheonsim issue #376
#              was opened by one)
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
script="$root/ci/track-workloads-issue.sh"
[[ -x $script ]] || { echo "FAIL: $script is not executable"; exit 1; }
d="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_workloads_issue.XXXXXX")"
trap 'rm -rf "$d"' EXIT
fails=0
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }

mkdir -p "$d/bin"
cat > "$d/bin/gh" <<'EOS'
#!/usr/bin/env bash
echo "$*" >> "$FAKE_DIR/gh.log"
case "$1 $2" in
  "label create") ;;
  "issue list") cat "$FAKE_DIR/open" 2>/dev/null || true ;;
  "issue close"|"issue comment"|"issue create") ;;
  *) echo "fake gh: unexpected $*" >&2; exit 1 ;;
esac
EOS
chmod +x "$d/bin/gh"

# run NAME RESULT OPEN: the script as a run with that machine result, where OPEN
# is the number of the open issue ("" for none). Sets calls (gh's verbs, one per
# line: "issue close 7"), rc.
run() {
  local name=$1 result=$2 open=$3
  export FAKE_DIR="$d/$name"
  rm -rf "$FAKE_DIR"; mkdir -p "$FAKE_DIR"
  [[ -n $open ]] && echo "$open" > "$FAKE_DIR/open"
  echo "| machine | result |" > "$FAKE_DIR/matrix.md"
  (cd "$FAKE_DIR" && env -i HOME="$d" PATH="$d/bin:/usr/bin:/bin" FAKE_DIR="$FAKE_DIR" \
    RESULT="$result" RUN_URL=https://example.test/run/1 GITHUB_EVENT_NAME=push \
    GITHUB_SHA=0123456789abcdef bash "$script" > "$FAKE_DIR/out" 2>&1)
  rc=$?
  calls="$(grep -E '^issue (close|comment|create)' "$FAKE_DIR/gh.log" 2>/dev/null | cut -d' ' -f1-3)"
}
expect() { [[ "$3" == "$2" ]] || fail "$1: wanted '$2', got '$3'"; }

run ok_open success 7;      expect "success, issue open: exit" 0 "$rc"; expect "success, issue open: calls" "issue close 7" "$calls"
run ok_none success "";     expect "success, none open: exit" 0 "$rc";   expect "success, none open: calls" "" "$calls"
run bad_none failure "";    expect "failure, none open: exit" 0 "$rc";   expect "failure, none open: calls" "issue create --title" "$calls"
run bad_open failure 7;     expect "failure, issue open: exit" 0 "$rc";  expect "failure, issue open: calls" "issue comment 7" "$calls"
run skip_none skipped "";   expect "skipped, none open: calls" "issue create --title" "$calls"
run can_none cancelled "";  expect "cancelled, none open: exit" 0 "$rc"; expect "cancelled, none open: calls" "" "$calls"
run can_open cancelled 7;   expect "cancelled, issue open: exit" 0 "$rc"; expect "cancelled, issue open: calls" "" "$calls"
for n in can_none can_open; do
  [[ -s $d/$n/gh.log ]] && fail "$n: gh was called at all: $(cat "$d/$n/gh.log")"
done
grep -q "^Workloads are failing on main (push, 0123456): https://example.test/run/1" "$d/bad_none/body.md" || fail "the issue body names the run and commit"

if (( fails )); then echo "$fails check(s) failed"; exit 1; fi
echo "workloads issue tracker: all checks passed"
