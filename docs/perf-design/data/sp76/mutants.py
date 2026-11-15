#!/usr/bin/env python3
"""Round sp76: mutants of the rule for a sample once the driver is up (include/vgpu/host_gpu.hpp sample_allowed, sample_card_path_seconds,
sample_short_before_seconds, sample_floor_estimate, sample_refusal). Each is a copy of the header with one change, put in a directory outside the tree that comes
first on the include path; tests/unit/test_host_gpu_logic.cpp is compiled against it and must fail. (The runner is round sp66's
data/sp66/mutants.py with this list.)

  mutants.py --src <tree> --work <dir outside the tree> [--only N] [--lock /tmp/pantheonsim-builder.lock]
"""
import argparse
import fcntl
import os
import subprocess
import time

LP = "vgpu/exec/launch.hpp"
HG = "vgpu/host_gpu.hpp"
UP = "  return after_start && start_ready;"
START = "  e.start_s = driver_up ? 0.0 : std::max(c.start_cost_s, 0.0);"
BAR = "  return std::max(c.margin * card_path_s, w.driver_up ? sample_floor(w) : 0.0);"
M = [
    (HG, "allowed: a driver that failed is not asked", "  if (start_failed) return false;\n  if (!start_begun) return true;", "  if (!start_begun) return true;"),
    (HG, "allowed: before the start by the switch", "  if (!start_begun) return true;", "  if (!start_begun) return after_start;"),
    (HG, "allowed: never with the driver up (the rule before)", UP, "  return false;"),
    (HG, "allowed: the switch not asked", UP, "  return start_ready;"),
    (HG, "allowed: a start that is not ready", UP, "  return after_start;"),
    (HG, "allowed: always", UP, "  return true;"),
    (HG, "card path: the start charged with the driver up", START, "  e.start_s = std::max(c.start_cost_s, 0.0);"),
    (HG, "card path: the start never charged", START, "  e.start_s = 0.0;"),
    (HG, "card path: a negative start cost is a credit", START, "  e.start_s = driver_up ? 0.0 : c.start_cost_s;"),
    (HG, "card path: the memory one way", "  e.push_bytes = e.pull_bytes = live_bytes;\n  e.start_s = driver_up", "  e.push_bytes = live_bytes;\n  e.start_s = driver_up"),
    (HG, "short before: the floor never asked", BAR, "  return c.margin * card_path_s;"),
    (HG, "short before: the floor asked before the start too", BAR, "  return std::max(c.margin * card_path_s, sample_floor(w));"),
    (HG, "short before: the floor alone with the driver up", BAR, "  return w.driver_up ? sample_floor(w) : c.margin * card_path_s;"),
    (HG, "short before: the margin not applied", BAR, "  return std::max(card_path_s, w.driver_up ? sample_floor(w) : 0.0);"),
    (HG, "short before: 30 ms whatever the live memory", BAR, "  return std::max(c.margin * card_path_s, w.driver_up ? kSampleStaticFloorSeconds : 0.0);"),
    (HG, "short before: a launch at the bar is sampled", "  if (w.whole_s && *w.whole_s <= sample_short_before_seconds(w, c, card_path_s)) return NoSample::ShortBefore;",
     "  if (w.whole_s && *w.whole_s < sample_short_before_seconds(w, c, card_path_s)) return NoSample::ShortBefore;"),
    (HG, "floor: the whole launch never stands in", "  return w.driver_up && w.whole_s ? std::max(counted, *w.whole_s) : counted;", "  return counted;"),
    (HG, "floor: the whole launch stands in before the start too", "  return w.driver_up && w.whole_s ? std::max(counted, *w.whole_s) : counted;", "  return w.whole_s ? std::max(counted, *w.whole_s) : counted;"),
    (HG, "floor: the whole launch in the count's place even when it is under it", "  return w.driver_up && w.whole_s ? std::max(counted, *w.whole_s) : counted;", "  return w.driver_up && w.whole_s ? *w.whole_s : counted;"),
    (HG, "floor: the rule asks the count alone", "  if (!(sample_floor_estimate(w) >= sample_floor(w) ||", "  if (!(sample_static_estimate(w) >= sample_floor(w) ||"),
    (HG, "short before: not asked", "  if (w.whole_s && *w.whole_s <= sample_short_before_seconds(w, c, card_path_s)) return NoSample::ShortBefore;\n", ""),
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
    os.makedirs(os.path.join(work, "inc/vgpu/exec"), exist_ok=True)
    text = {h: open(os.path.join(src, "include", h)).read() for h in (LP, HG)}
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
        for other in (LP, HG):
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
