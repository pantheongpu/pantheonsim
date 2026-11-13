#!/usr/bin/env python3
"""Integration of rounds sp64 to sp68: mutants of the start-share rule (include/vgpu/host_gpu.hpp start_share_saved, should_offload_with_start,
first_launch_offload and Costs::start_launches / start_launches_static). Each is a copy of the header with one change, put in a directory outside
the tree that comes first on the include path; tests/unit/test_host_gpu_logic.cpp is compiled against it and must fail. (The runner is round
sp66's data/sp66/mutants.py with this list.)

  mutants.py --src <tree> --work <dir outside the tree> [--only N] [--lock /tmp/pantheonsim-builder.lock]
"""
import argparse
import fcntl
import os
import subprocess
import time

IM = "vgpu/host_gpu_image.hpp"
RP = "vgpu/sass/reloc_patch.hpp"
HG = "vgpu/host_gpu.hpp"
ACC = "  return eligible_sim_spent_s >= c.start_cost_s && start_share_saved(e, c);"
SHARE = "  return e.sim_s - card_path_seconds(e, c) >= c.start_cost_s / c.start_launches;"
M = [
    (HG, "share: not asked (the account alone)", ACC, "  return eligible_sim_spent_s >= c.start_cost_s;"),
    (HG, "share: instead of the account", ACC, "  return start_share_saved(e, c);"),
    (HG, "share: or the account", ACC, "  return eligible_sim_spent_s >= c.start_cost_s || start_share_saved(e, c);"),
    (HG, "share: the card path not subtracted", SHARE, "  return e.sim_s >= c.start_cost_s / c.start_launches;"),
    (HG, "share: more than, not at least", SHARE, "  return e.sim_s - card_path_seconds(e, c) > c.start_cost_s / c.start_launches;"),
    (HG, "share: the whole start", SHARE, "  return e.sim_s - card_path_seconds(e, c) >= c.start_cost_s;"),
    (HG, "share: not the start's (a fixed 0.25 s)", SHARE, "  return e.sim_s - card_path_seconds(e, c) >= 0.25 / c.start_launches;"),
    (HG, "share: the card path with the start", "&& start_share_saved(e, c);", "&& start_share_saved(cold, c);"),
    (HG, "share: no guard for 0", "  if (!(c.start_launches > 0)) return true;\n", ""),
    (HG, "default: 25 launches", "  double start_launches = 50;", "  double start_launches = 25;"),
    (HG, "default: 100 launches", "  double start_launches = 50;", "  double start_launches = 100;"),
    (HG, "default: static share of 50", "  double start_launches_static = 20;", "  double start_launches_static = 50;"),
    (HG, "first launch: the history's share", "  if (c.start_launches > 0) fc.start_launches = c.start_launches_static;\n", ""),
    (HG, "first launch: the static share even with 0", "  if (c.start_launches > 0) fc.start_launches = c.start_launches_static;", "  fc.start_launches = c.start_launches_static;"),
    (HG, "first launch: not the copy", "  return should_offload_with_start(e, fc, driver_started, eligible_sim_spent_s);", "  return should_offload_with_start(e, c, driver_started, eligible_sim_spent_s);"),
    (HG, "cold: a launch never repays the start alone", "  if (should_offload(cold, c)) return true;                     // the launch repays the start by itself\n", ""),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", type=int, default=-1)
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    ap.add_argument("--hold", type=float, default=200)
    a = ap.parse_args()
    src = os.path.abspath(a.src)
    work = os.path.abspath(a.work)
    assert not work.startswith(src + os.sep), "the mutants go outside the tree"
    os.makedirs(os.path.join(work, "inc/vgpu/sass"), exist_ok=True)
    text = {h: open(os.path.join(src, "include", h)).read() for h in (IM, RP, HG)}
    pre = ["nice", "-n", "19"]
    fd = os.open(a.lock, os.O_RDWR | os.O_CREAT, 0o666) if a.lock else None
    since = None
    killed = 0
    todo = [(i, m) for i, m in enumerate(M) if a.only < 0 or i == a.only]
    for i, (h, name, old, new) in todo:
        if text[h].count(old) != 1:
            print("%2d NOT APPLIED (%d matches): %s" % (i, text[h].count(old), name), flush=True)
            continue
        if fd is not None and since is not None and time.monotonic() - since > a.hold:   # the machine's build lock: given up now and then
            fcntl.flock(fd, fcntl.LOCK_UN)
            since = None
            time.sleep(1)
        if fd is not None and since is None:
            fcntl.flock(fd, fcntl.LOCK_EX)
            since = time.monotonic()
        for other in (IM, RP, HG):
            p = os.path.join(work, "inc", other)
            if os.path.exists(p):
                os.remove(p)
        open(os.path.join(work, "inc", h), "w").write(text[h].replace(old, new))
        exe = os.path.join(work, "mut%d" % i)
        c = subprocess.run(pre + ["g++", "-std=c++20", "-O1", "-I", os.path.join(work, "inc"), "-I", os.path.join(src, "include"), "-I", os.path.join(src, "tests/framework"),
                                  os.path.join(src, "tests/unit/test_host_gpu_logic.cpp"), "-o", exe], capture_output=True, text=True)
        if c.returncode != 0:
            print("%2d killed (does not compile): %s" % (i, name), flush=True)
            killed += 1
            continue
        r = subprocess.run([exe], capture_output=True, text=True)
        fails = [l.split("]", 1)[1].strip() for l in r.stdout.splitlines() if l.startswith("[ FAIL ]")]
        os.remove(exe)
        if r.returncode != 0:
            killed += 1
            print("%2d killed by %s: %s" % (i, ", ".join(fails[:3]) or "exit %d" % r.returncode, name), flush=True)
        else:
            print("%2d SURVIVED: %s" % (i, name), flush=True)
    if fd is not None and since is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
    print("%d of %d killed" % (killed, len(todo)))


if __name__ == "__main__":
    main()
