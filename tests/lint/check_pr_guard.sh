#!/usr/bin/env bash
# The merge-conflict guards for CI: ci/pr-guard.sh (the first step of a heavy
# job: cancel the run when the pull request conflicts) and
# ci/cancel-conflicting-prs.sh (cancel the runs of every conflicting PR after
# main moves). Both run here against a fake `gh` (and a fake `curl`) that
# answers from canned files, so nothing touches the network or a real run.
#
# pr-guard: a clean PR passes; `dirty` cancels the run (the fake records the
# call) and fails the job; `mergeable: null` is asked again until GitHub has an
# answer; one it never gives lets the job continue; `behind`, `blocked`,
# `unstable` are not conflicts; a push, schedule or dispatch does nothing; an
# API error, a rate limit, an unreadable answer, a missing token or neither gh
# nor curl never fails the job; the curl path (a container without gh) does the
# same as the gh path.
# canceller: only `dirty` PRs lose their pull_request runs and gain the label;
# a PR that stopped conflicting loses the label; a dry run changes nothing.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
guard="$root/ci/pr-guard.sh"
canceller="$root/ci/cancel-conflicting-prs.sh"
for f in "$guard" "$canceller"; do [[ -x $f ]] || { echo "FAIL: $f is not executable"; exit 1; }; done
d="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_pr_guard.XXXXXX")"
trap 'rm -rf "$d"' EXIT

fails=0
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }
expect() { [[ "$3" == "$2" ]] || fail "$1: wanted '$2', got '$3'"; }

# ---- the fakes ---------------------------------------------------------------
mkdir -p "$d/bin" "$d/min"
# pop N: the next canned answer for PR N, as the API's JSON (the last repeats)
cat > "$d/bin/pop" <<'EOF'
#!/usr/bin/env bash
n=$1
echo "poll $n" >> "$FAKE_DIR/polls.log"
q="$FAKE_DIR/q.$n"
[[ -f $q ]] || { echo "no PR $n" >&2; exit 1; }
i=$(cat "$FAKE_DIR/i.$n" 2>/dev/null || echo 0)
total=$(wc -l < "$q")
(( i >= total )) && i=$((total - 1))
echo $((i + 1)) > "$FAKE_DIR/i.$n"
read -r mergeable state < <(sed -n "$((i + 1))p" "$q")
case "$mergeable" in
  ERROR) echo "HTTP 403: API rate limit exceeded" >&2; exit 1 ;;
  GARBAGE) echo "<html>not json</html>" ;;
  COMPACT) m=true; [[ $state == dirty ]] && m=false
           printf '{"number":%s,"state":"open","mergeable":%s,"mergeable_state":"%s","merged":false}\n' "$n" "$m" "$state" ;;
  *) printf '{\n  "number": %s,\n  "state": "open",\n  "mergeable": %s,\n  "mergeable_state": "%s",\n  "merged": false\n}\n' "$n" "$mergeable" "$state" ;;
esac
EOF
cat > "$d/bin/gh" <<'EOF'
#!/usr/bin/env bash
echo "$*" >> "$FAKE_DIR/gh.log"
case "$1 $2" in
  "api repos/"*) exec "$FAKE_DIR/bin/pop" "${2##*/}" ;;
  "run cancel")
    [[ -n ${FAKE_CANCEL_FAIL:-} ]] && { echo "HTTP 403: Resource not accessible by integration" >&2; exit 1; }
    echo "cancel $3" >> "$FAKE_DIR/cancel.log" ;;
  "pr list") cat "$FAKE_DIR/prs.tsv" ;;
  "run list")
    sha="" status=""
    while (($#)); do
      case "$1" in --commit) sha=$2; shift ;; --status) status=$2; shift ;; esac
      shift
    done
    cat "$FAKE_DIR/runs.$sha.$status" 2>/dev/null ;;
  "pr edit") echo "edit $*" >> "$FAKE_DIR/edit.log" ;;
  "label create") echo "create $*" >> "$FAKE_DIR/label.log" ;;
  *) echo "fake gh: unexpected $*" >&2; exit 1 ;;
esac
EOF
cat > "$d/bin/curl" <<'EOF'
#!/usr/bin/env bash
echo "$*" >> "$FAKE_DIR/curl.log"
url="${@: -1}"
case " $* " in
  *" -X POST "*) echo "POST ${url#*/repos/}" >> "$FAKE_DIR/cancel.log"; [[ -n ${FAKE_CANCEL_FAIL:-} ]] && exit 22; exit 0 ;;
esac
exec "$FAKE_DIR/bin/pop" "${url##*/}"
EOF
chmod +x "$d/bin/"*
# a PATH with the tools the script needs and neither gh nor curl
for t in sed head sleep timeout printf cat wc bash env; do
  p="$(command -v "$t" 2>/dev/null)"; [[ -n $p && -x $p ]] && ln -sf "$p" "$d/min/$t"
done

# ---- pr-guard ------------------------------------------------------------------
# guard NAME EVENT QUEUELINES...: pr-guard.sh as a job of that event on PR 7, whose
# answers are the queue (lines "<mergeable> <state>", or ERROR / GARBAGE / COMPACT <state>).
# Sets rc, out; the fake's logs are in $d/$NAME.
guard() {
  local name=$1 event=$2; shift 2
  export FAKE_DIR="$d/$name"
  rm -rf "$FAKE_DIR"; mkdir -p "$FAKE_DIR"
  ln -s "$d/bin" "$FAKE_DIR/bin"
  printf '%s\n' "$@" > "$FAKE_DIR/q.7"
  out="$(env -i HOME="$d" PATH="${GUARD_PATH:-$d/bin:/usr/bin:/bin}" FAKE_DIR="$FAKE_DIR" \
    FAKE_CANCEL_FAIL="${FAKE_CANCEL_FAIL:-}" PR_GUARD_NO_GH="${PR_GUARD_NO_GH:-}" \
    GITHUB_EVENT_NAME="$event" GITHUB_REPOSITORY=acme/sim GITHUB_REF="${REF:-refs/pull/7/merge}" \
    GITHUB_RUN_ID=4242 GITHUB_BASE_REF=main GITHUB_EVENT_PATH="${EVENT_PATH:-}" \
    GH_TOKEN="${TOKEN-secret}" PR_GUARD_STEP=0 PR_GUARD_CANCEL_WAIT=0 PR_GUARD_TRIES="${TRIES:-9}" \
    bash "$guard" 2>&1)"
  rc=$?
}
count() { local c; c="$(grep -c "$1" "$2" 2>/dev/null)"; echo "${c:-0}"; }   # matching lines, 0 when no file
polls() { count '^poll' "$d/$1/polls.log"; }
cancels() { count 'cancel' "$d/$1/cancel.log"; }

guard clean pull_request "true clean"
expect "clean: exit" 0 "$rc"; expect "clean: cancels" 0 "$(cancels clean)"; expect "clean: polls" 1 "$(polls clean)"

guard dirty pull_request "false dirty"
expect "dirty: exit" 1 "$rc"
expect "dirty: the run is cancelled" "cancel 4242" "$(cat "$d/dirty/cancel.log" 2>/dev/null)"
[[ $out == *"PR #7 has a merge conflict with main: cancelling this run, rebase and push"* ]] || fail "dirty: message: $out"
grep -q -- "--repo acme/sim" "$d/dirty/gh.log" || fail "dirty: cancel names the repository"

guard late pull_request "null unknown" "null unknown" "false dirty"
expect "null, null, dirty: exit" 1 "$rc"; expect "null, null, dirty: polls" 3 "$(polls late)"
expect "null, null, dirty: cancels" 1 "$(cancels late)"

guard late_clean pull_request "null unknown" "true clean"
expect "null then clean: exit" 0 "$rc"; expect "null then clean: polls" 2 "$(polls late_clean)"

TRIES=4 guard never pull_request "null unknown"
expect "unknown forever: the job continues" 0 "$rc"
expect "unknown forever: asked the tries it was given" 4 "$(polls never)"
expect "unknown forever: cancels" 0 "$(cancels never)"

for st in behind blocked unstable has_hooks draft unknown; do
  guard "s_$st" pull_request "true $st"
  expect "$st is not a conflict: exit" 0 "$rc"; expect "$st: cancels" 0 "$(cancels "s_$st")"
done

for ev in push schedule workflow_dispatch merge_group repository_dispatch; do
  guard "e_$ev" "$ev" "false dirty"
  expect "$ev does nothing: exit" 0 "$rc"; expect "$ev: polls" 0 "$(polls "e_$ev")"; expect "$ev: cancels" 0 "$(cancels "e_$ev")"
done

guard api_err pull_request "ERROR x"
expect "an API error does not fail the job" 0 "$rc"; expect "API error: cancels" 0 "$(cancels api_err)"
[[ $out == *"continuing"* ]] || fail "API error: no notice: $out"
guard garbage pull_request "GARBAGE x"
expect "an unreadable answer does not fail the job" 0 "$rc"
guard rate_after_null pull_request "null unknown" "ERROR x"
expect "an error after a null does not fail the job" 0 "$rc"
guard compact pull_request "COMPACT dirty"
expect "compact JSON is read: exit" 1 "$rc"
guard compact_clean pull_request "COMPACT clean"
expect "compact JSON, not dirty: exit" 0 "$rc"
FAKE_CANCEL_FAIL=1 guard cancel_fail pull_request "false dirty"
expect "a dirty PR whose cancel is refused still stops the job" 1 "$rc"
[[ $out == *"could not cancel"* ]] || fail "cancel refused: no warning: $out"

# the PR number from the event payload (pull_request_target runs on the base ref)
printf '{\n  "action": "synchronize",\n  "number": 7,\n  "pull_request": {"number": 7}\n}\n' > "$d/event.json"
REF=refs/heads/main EVENT_PATH="$d/event.json" guard target pull_request_target "false dirty"
expect "pull_request_target, number from the payload: exit" 1 "$rc"
REF=refs/heads/main guard nonum pull_request "false dirty"
expect "no PR number: the job continues" 0 "$rc"; expect "no PR number: cancels" 0 "$(cancels nonum)"
TOKEN="" guard notoken pull_request "false dirty"
expect "no token: the job continues" 0 "$rc"; expect "no token: polls" 0 "$(polls notoken)"

# the curl path (a container job without gh)
PR_GUARD_NO_GH=1 guard curl_clean pull_request "true clean"
expect "curl, clean: exit" 0 "$rc"; expect "curl, clean: cancels" 0 "$(cancels curl_clean)"
PR_GUARD_NO_GH=1 guard curl_dirty pull_request "null unknown" "false dirty"
expect "curl, dirty: exit" 1 "$rc"
expect "curl, dirty: the cancel is the REST call" "POST acme/sim/actions/runs/4242/cancel" "$(cat "$d/curl_dirty/cancel.log" 2>/dev/null)"
grep -q "Authorization: Bearer secret" "$d/curl_dirty/curl.log" || fail "curl sends the token"
PR_GUARD_NO_GH=1 guard curl_err pull_request "ERROR x"
expect "curl, API error: exit" 0 "$rc"
GUARD_PATH="$d/min" guard nothing pull_request "false dirty"
expect "neither gh nor curl: the job continues" 0 "$rc"

# ---- the canceller -------------------------------------------------------------
# PRs: 1 conflicts (two live runs, one a push run that must be left alone), 2 is
# clean but still labelled, 3 conflicts and is labelled already, 4 is behind, 5
# never gets an answer, 6 cannot be read.
canceller_setup() {
  export FAKE_DIR="$d/canceller"
  rm -rf "$FAKE_DIR"; mkdir -p "$FAKE_DIR"; ln -s "$d/bin" "$FAKE_DIR/bin"
  printf '%s\t%s\t%s\t%s\n' \
    1 sha1 false feat-a  2 sha2 true feat-b  3 sha3 true feat-c  4 sha4 false feat-d \
    5 sha5 false feat-e  6 sha6 false feat-f > "$FAKE_DIR/prs.tsv"
  printf 'false dirty\n' > "$FAKE_DIR/q.1"
  printf 'true clean\n'  > "$FAKE_DIR/q.2"
  printf 'null unknown\nfalse dirty\n' > "$FAKE_DIR/q.3"
  printf 'true behind\n' > "$FAKE_DIR/q.4"
  printf 'null unknown\n' > "$FAKE_DIR/q.5"
  printf 'ERROR x\n' > "$FAKE_DIR/q.6"
  printf '100\n' > "$FAKE_DIR/runs.sha1.in_progress"
  printf '101\n102\n' > "$FAKE_DIR/runs.sha1.queued"
  printf '300\n' > "$FAKE_DIR/runs.sha3.waiting"
  # what `gh run list --jq` leaves after dropping non-pull_request events is all the fake knows:
  # a push run (200) is never in the lists it returns
}
run_canceller() {
  out="$(env -i HOME="$d" PATH="$d/bin:/usr/bin:/bin" FAKE_DIR="$FAKE_DIR" GH_REPO=acme/sim \
    GH_TOKEN=secret DRY_RUN="$1" CC_STEP=0 CC_ROUNDS=2 CC_PAUSE=0 bash "$canceller" 2>&1)"
  rc=$?
}

canceller_setup
run_canceller false
expect "canceller: exit" 0 "$rc"
expect "canceller: runs cancelled" $'cancel 100\ncancel 101\ncancel 102\ncancel 300' "$(sort -k2 -n "$FAKE_DIR/cancel.log" 2>/dev/null)"
[[ $out == *"#1 (feat-a): CONFLICT, cancelled 3 run(s), labelled"* ]] || fail "canceller: PR 1 line: $out"
[[ $out == *"#2 (feat-b): clean, removed the merge-conflict label"* ]] || fail "canceller: PR 2 line: $out"
[[ $out == *"#3 (feat-c): CONFLICT, cancelled 1 run(s), already labelled"* ]] || fail "canceller: PR 3 line: $out"
[[ $out == *"#4 (feat-d): behind"* ]] || fail "canceller: PR 4 line: $out"
[[ $out == *"#5 (feat-e): mergeability not computed yet, skipped"* ]] || fail "canceller: PR 5 line: $out"
[[ $out == *"#6 (feat-f): could not read it, skipped"* ]] || fail "canceller: PR 6 line: $out"
[[ $out == *"checked 6 open PR(s): 2 conflicting, 4 run(s) cancelled, 1 labelled, 1 unlabelled, 2 skipped"* ]] || fail "canceller: summary: $out"
expect "canceller: label edits" $'edit pr edit 1 --repo acme/sim --add-label merge-conflict\nedit pr edit 2 --repo acme/sim --remove-label merge-conflict' "$(cat "$FAKE_DIR/edit.log")"
grep -q -- "--color d93f0b" "$FAKE_DIR/label.log" || fail "canceller: the label is created with its colour"
expect "canceller: the label is created once" 1 "$(wc -l < "$FAKE_DIR/label.log")"
expect "canceller: the null PR is asked again, not forever" 3 "$(grep -c '^poll 5' "$FAKE_DIR/polls.log")"
if grep -q "run list.*--status" "$FAKE_DIR/gh.log"; then
  for s in in_progress queued waiting; do grep -q -- "--status $s" "$FAKE_DIR/gh.log" || fail "canceller: never asked for $s runs"; done
fi
grep -q -- "--commit sha2" "$FAKE_DIR/gh.log" && fail "canceller: listed runs of a PR that does not conflict"

canceller_setup
run_canceller true
expect "dry run: exit" 0 "$rc"
[[ -e $FAKE_DIR/cancel.log || -e $FAKE_DIR/edit.log || -e $FAKE_DIR/label.log ]] && fail "dry run changed something"
[[ $out == *"#1 (feat-a): CONFLICT, would cancel 3 run(s), would add the merge-conflict label"* ]] || fail "dry run: PR 1 line: $out"
[[ $out == *"#2 (feat-b): clean, would remove the merge-conflict label"* ]] || fail "dry run: PR 2 line: $out"

(( fails )) && { echo "FAIL: $fails check(s) failed"; exit 1; }
echo "PASS: the merge-conflict guard cancels a conflicting PR's run and nothing else; the canceller labels and cancels only dirty PRs"
