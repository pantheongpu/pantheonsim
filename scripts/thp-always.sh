#!/usr/bin/env bash
# GitHub-hosted Ubuntu runners ship transparent huge pages as `madvise`: only memory a program asks for with
# madvise(MADV_HUGEPAGE) gets 2 MiB pages, and everything else, a program's own heap and the large buffers glibc maps for
# it among them, is faulted in 4 KiB at a time. A simulated GPU runs the whole program on the CPU, so that program's own
# memory is as hot as the simulator's: SCS's max_ent took 1432 s with `madvise` and 696 s with `always` on the same
# runner and build (#551). On a GitHub-hosted runner whose setting is `madvise`, this sets `always` for the rest of the
# job (the runner is thrown away after it) and says so. It changes nothing on a self-hosted runner (the machine is
# someone's to configure), when the setting is `always` or `never` already, or when it cannot write the setting (no
# passwordless sudo, a read-only /sys in a container); it says which, and it always exits 0.
#
#   THP_ENABLED         the setting's file (default /sys/kernel/mm/transparent_hugepage/enabled; the test points it
#                       at a temporary one)
#   RUNNER_ENVIRONMENT  github-hosted or self-hosted, as GitHub sets it
#   SUDO                `sudo` to write through sudo (never prompting), empty to write directly (a root container job)
set -u
f=${THP_ENABLED:-/sys/kernel/mm/transparent_hugepage/enabled}
if [[ ! -r $f ]]; then
  echo "transparent huge pages: $f is not there; nothing to do"
  exit 0
fi
mode=$(sed -n 's/.*\[\([a-z]*\)\].*/\1/p' "$f" 2>/dev/null)
if [[ ${RUNNER_ENVIRONMENT:-} != github-hosted ]]; then
  echo "transparent huge pages: '${mode:-unknown}', left as it is (not a GitHub-hosted runner)"
  exit 0
fi
if [[ $mode != madvise ]]; then
  echo "transparent huge pages: '${mode:-unknown}', left as it is"
  exit 0
fi
writer=(tee "$f")
if [[ -n ${SUDO:-} ]]; then
  if ! "$SUDO" -n true 2>/dev/null; then
    echo "transparent huge pages: 'madvise', left as it is (no passwordless sudo to change it)"
    exit 0
  fi
  writer=("$SUDO" -n tee "$f")
fi
if echo always | "${writer[@]}" >/dev/null 2>&1; then
  echo "transparent huge pages: 'madvise' -> 'always' for this job (the simulated program's own memory gets 2 MiB pages)"
  exit 0
fi
echo "transparent huge pages: 'madvise', left as it is (could not write $f)"
exit 0
