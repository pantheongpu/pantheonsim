# How a finished pantheon workload is judged. Read by run_pantheon_workloads.sh, inside
# `vgpu shell`, and by verdict_test.sh. POSIX sh: no arrays, no here-strings, no $'..'.
#
#   workload_verdict <exit status> <log file>
#
# prints one line, "RESULT<TAB>detail", where RESULT is PASS, SKIP or FAIL. A run that hit
# the timeout (status 124) is TIMEOUT, which the caller decides before asking.
#
# A workload that deliberately does not run (not enough GPUs, no library, started without
# the environment its pantheon runner gives it) exits 0 and says so on a line of its own,
# "Skipping: <reason>", with no throughput line. That is a skip. Counted as a pass it would
# make a workload that never ran look like one that ran, which is the failure this runner
# exists to catch: its output still holds the word PANTHEON, which the check for "said
# something about what it measured" accepts.
workload_verdict() {
  status="$1"
  log="$2"
  why=$(grep -iE "PANTHEON ERROR|CUDA error|integrity|VirtualGPU error|unsupported PTX|Verification: FAIL" "$log" | head -1 | cut -c1-100)
  if [ "$status" -ne 0 ]; then
    printf 'FAIL\texit %s%s\n' "$status" "${why:+: $why}"
  elif [ -n "$why" ]; then
    printf 'FAIL\t%s\n' "$why"
  elif reason=$(grep -m1 '^Skipping: ' "$log"); then
    printf 'SKIP\t%s\n' "$(printf '%s\n' "$reason" | cut -c11-100)"
  elif ! grep -qiE "throughput|bandwidth|latency|ops/s|GB/s|PANTHEON|Verification: PASS" "$log"; then
    printf 'FAIL\texited 0 but reported no measurement\n'
  else
    printf 'PASS\t%s\n' "$(grep -iE 'throughput|ops/s|GB/s|Verification' "$log" | head -1 | tr -s ' ' | cut -c1-80)"
  fi
}
