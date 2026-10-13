#!/usr/bin/env bash
# The number of parallel build jobs for a CI job: the CPU count, but no more than the
# memory that is free allows at MEM_PER_JOB_GB gigabytes a job (default 3; an
# AddressSanitizer Debug compile of this tree needs several). A hosted runner (4 CPUs,
# 16 GB) gets the CPU count, as before; a big self-hosted runner (32 CPUs, 29 GB) no
# longer starts 32 compiles and gets its runner process killed by the OOM killer.
set -euo pipefail
per="${1:-${MEM_PER_JOB_GB:-3}}"
cpus="$(nproc)"
mem_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || true)"
if [[ -z "$mem_kb" ]]; then echo "$cpus"; exit 0; fi
by_mem=$(( mem_kb / 1024 / 1024 / per ))
(( by_mem < 1 )) && by_mem=1
echo $(( by_mem < cpus ? by_mem : cpus ))
