#!/usr/bin/env python3
"""The version matrix covers every CUDA toolkit from 12.4 to 13.2 and every ROCm from 6.4 to 7.2.

The range is the one docs/install.md and docs/status.md promise. A version is covered when a workflow installs it:
version-matrix.yml (the nightly matrix of small checks), cuda-toolkits.yml and ci.yml (the full suites) for CUDA,
version-matrix.yml and rocm-libraries.yml for ROCm. Also checked:

  - version-matrix.yml runs on a schedule and by hand, never on a push, and on a pull request only when the PR
    changes the workflow's own files (a `pull_request:` with `paths:`);
  - every program the matrix builds exists, and the matrix names the same rows the lint expects.

Reads the workflows as text (the standard library only). usage: check_version_matrix.py
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
WF = ROOT / ".github" / "workflows"
CUDA = ["12.4", "12.5", "12.6", "12.8", "12.9", "13.0", "13.1", "13.2"]  # 12.7 never shipped
ROCM = ["6.4", "7.0", "7.1", "7.2"]

fail = False


def bad(msg):
    global fail
    print("FAIL  " + msg)
    fail = True


def text(name):
    return (WF / name).read_text()


matrix = text("version-matrix.yml")

# Triggers.
on = matrix.split("\njobs:")[0]
if re.search(r"^  push:", on, re.M):
    bad("version-matrix.yml runs on push; it is nightly, not per change")
if not re.search(r"^  schedule:", on, re.M):
    bad("version-matrix.yml has no schedule")
if not re.search(r"^  workflow_dispatch:", on, re.M):
    bad("version-matrix.yml cannot be run by hand")
pr = re.search(r"^  pull_request:\n((?:    .*\n)*)", on, re.M)
if pr and "paths:" not in pr.group(1):
    bad("version-matrix.yml runs on every pull request; give pull_request a paths: list of its own files")

# Versions the matrix installs.
cuda_rows = re.findall(r"cuda: '([0-9.]+)'", matrix)
rocm_rows = re.findall(r"rocm: '([0-9.]+)'", matrix)

# Versions the full suites install.
suites_cuda = set(re.findall(r"cuda: '([0-9.]+)'", text("cuda-toolkits.yml")))
ci = text("ci.yml")
suites_cuda |= set(re.findall(r"cuda-?(1[23]\.\d+)", ci))
suites_rocm = set(re.findall(r"rocm(\d+\.\d+)", text("rocm-libraries.yml")))

for v in CUDA:
    if v not in cuda_rows and v not in suites_cuda:
        bad(f"CUDA {v} is in no workflow")
for v in ROCM:
    if v not in rocm_rows and v not in suites_rocm:
        bad(f"ROCm {v} is in no workflow")
for v in cuda_rows:
    if v not in CUDA:
        bad(f"version-matrix.yml has CUDA {v}, outside the range this lint knows ({CUDA[0]} to {CUDA[-1]})")
for v in rocm_rows:
    if v not in ROCM:
        bad(f"version-matrix.yml has ROCm {v}, outside the range this lint knows ({ROCM[0]} to {ROCM[-1]})")
if sorted(cuda_rows) != sorted(CUDA):
    bad(f"version-matrix.yml's CUDA rows are {cuda_rows}, not {CUDA}")
if sorted(rocm_rows) != sorted(ROCM):
    bad(f"version-matrix.yml's ROCm rows are {rocm_rows}, not {ROCM}")

# The programs and the pages that state the range.
for f in re.findall(r"tests/version-matrix/[\w.]+", matrix):
    if not (ROOT / f).is_file():
        bad(f"version-matrix.yml builds {f}, which does not exist")
install = (ROOT / "docs" / "install.md").read_text()
if "12.4 to 13.2" not in install or "6.4, 7.0, 7.1 and 7.2" not in install:
    bad("docs/install.md does not state the range the matrix checks (CUDA 12.4 to 13.2, ROCm 6.4 to 7.2)")

if not fail:
    print(f"ok   CUDA {', '.join(CUDA)} and ROCm {', '.join(ROCM)} covered; matrix is scheduled")
sys.exit(1 if fail else 0)
