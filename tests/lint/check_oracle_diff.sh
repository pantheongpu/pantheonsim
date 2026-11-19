#!/usr/bin/env bash
# The card-oracle diff (tools/oracle-diff, .github/workflows/oracle-diff.yml): the manifest names real scripts that support --card and
# have a pinned card file, the driver passes only when every script prints PASS (a skip, a failure, a missing script or an empty
# manifest are failures), and the workflow is scheduled and has a card job. No build, no GPU, no nvcc: stub scripts stand in.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
drv="$root/tools/oracle-diff/oracle-diff.sh"
man="$root/tools/oracle-diff/manifest.txt"
wf="$root/.github/workflows/oracle-diff.yml"
fail=0
bad() { echo "FAIL: $*"; fail=1; }

# 1. the real manifest
n=0
while IFS= read -r l; do
  l="${l%%#*}"; l="${l//[[:space:]]/}"; [[ -n $l ]] || continue
  n=$((n + 1))
  s="$root/nvidia/tests/e2e/$l"
  [[ -x $s ]] || { bad "$l is not an executable script in nvidia/tests/e2e"; continue; }
  grep -q -- '--card' "$s" || bad "$l has no --card mode"
  grep -q 'VGPU_CARD_OUT' "$s" || bad "$l does not export the card's lines (VGPU_CARD_OUT)"
  grep -Eq 'nvidia/tests/data|\$data/' "$s" || bad "$l names no pinned file under nvidia/tests/data"
done < "$man"
(( n >= 2 )) || bad "manifest has $n scripts, want at least 2"

# 2. the driver, on stubs
d="$(mktemp -d "${TMPDIR:-/tmp}/oracle_selftest.XXXXXX")"; trap 'rm -rf "$d"' EXIT
mk() { printf '#!/usr/bin/env bash\n%s\n' "$2" > "$d/$1"; chmod +x "$d/$1"; }
mk ok.sh 'echo "PASS (stub, 3 lines)"'
mk skip.sh 'echo "SKIP: no nvcc"; exit 0'
mk fail.sh 'echo "FAIL: differs"; exit 1'
mk silent.sh 'exit 0'
run() { # run <expected rc> <label> <manifest lines...>
  local want="$1" label="$2"; shift 2
  printf '%s\n' "$@" > "$d/m.txt"
  ORACLE_E2E_DIR="$d" ORACLE_MANIFEST="$d/m.txt" ORACLE_DIFF_OUT="$d/out" GITHUB_STEP_SUMMARY= "$drv" > "$d/log" 2>&1
  local rc=$?
  (( rc == want )) || { bad "driver: $label: exit $rc, want $want"; sed 's/^/    /' "$d/log"; }
}
run 0 "all pass" ok.sh "# comment" ok.sh
run 1 "a failure" ok.sh fail.sh
run 1 "a skip is a failure" ok.sh skip.sh
run 1 "no PASS line" silent.sh
run 1 "missing script" nonexistent.sh
run 1 "empty manifest" "# nothing"
[[ -s $d/out/summary.md ]] || bad "driver wrote no summary.md"

# 3. the workflow
[[ -f $wf ]] || bad "$wf is missing"
if [[ -f $wf ]]; then
  grep -Eq '^\s*-\s*cron:' "$wf" || bad "workflow has no schedule"
  grep -q 'workflow_dispatch' "$wf" || bad "workflow has no workflow_dispatch"
  grep -q 'oracle-diff.sh --card' "$wf" || bad "workflow never runs the card mode"
  grep -q 'oracle-diff.sh$\|oracle-diff.sh *$' "$wf" || bad "workflow never runs the simulator mode"
  grep -q 'ORACLE_CARD_RUNNER' "$wf" || bad "card job is not gated on ORACLE_CARD_RUNNER"
  # the card job must never run for a pull request (a fork's code on a card host)
  grep -Eq "github.event_name != 'pull_request'" "$wf" || bad "card job can run on pull_request"
fi
(( fail )) && exit 1
echo "PASS: oracle-diff manifest ($n scripts), driver verdicts, workflow"
