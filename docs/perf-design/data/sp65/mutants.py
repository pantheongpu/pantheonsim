#!/usr/bin/env python3
"""Round sp65: mutants of the rule for a launch while the card holds the live memory (include/vgpu/host_gpu.hpp streak_offload and the two sides it
weighs). Each is a copy of the header with one change, put in a directory outside the tree that comes first on the include path;
tests/unit/test_host_gpu_logic.cpp is compiled against it and must fail. (The runner is sp61's.)

  mutants.py --src <tree> --work <dir outside the tree> [--only N] [--lock /tmp/pantheonsim-builder.lock]
"""
import argparse
import fcntl
import os
import subprocess
import time

BP = "vgpu/exec/block_progress.hpp"
HG = "vgpu/host_gpu.hpp"
M = [
    (HG, "no driver needed", "  if (!driver_started || s.card_only_bytes == 0 || !(s.sim_s > 0)) return false;", "  if (s.card_only_bytes == 0 || !(s.sim_s > 0)) return false;"),
    (HG, "no card data needed", "  if (!driver_started || s.card_only_bytes == 0 || !(s.sim_s > 0)) return false;", "  if (!driver_started || !(s.sim_s > 0)) return false;"),
    (HG, "no estimate needed", "  if (!driver_started || s.card_only_bytes == 0 || !(s.sim_s > 0)) return false;", "  if (!driver_started || s.card_only_bytes == 0) return false;"),
    (HG, "a negative estimate passes", "  if (!driver_started || s.card_only_bytes == 0 || !(s.sim_s > 0)) return false;", "  if (!driver_started || s.card_only_bytes == 0 || s.sim_s == 0) return false;"),
    (HG, "the driver inverted", "  if (!driver_started || s.card_only_bytes == 0 || !(s.sim_s > 0)) return false;", "  if (driver_started || s.card_only_bytes == 0 || !(s.sim_s > 0)) return false;"),
    (HG, "no floor", "  return keep >= c.min_sim_s && keep > c.margin * streak_card_seconds(s, c);", "  return keep > c.margin * streak_card_seconds(s, c);"),
    (HG, "floor or margin", "  return keep >= c.min_sim_s && keep > c.margin * streak_card_seconds(s, c);", "  return keep >= c.min_sim_s || keep > c.margin * streak_card_seconds(s, c);"),
    (HG, "the floor strict", "  return keep >= c.min_sim_s && keep > c.margin * streak_card_seconds(s, c);", "  return keep > c.min_sim_s && keep > c.margin * streak_card_seconds(s, c);"),
    (HG, "the floor on the simulator's time alone", "  return keep >= c.min_sim_s && keep > c.margin * streak_card_seconds(s, c);", "  return s.sim_s >= c.min_sim_s && keep > c.margin * streak_card_seconds(s, c);"),
    (HG, "no margin", "  return keep >= c.min_sim_s && keep > c.margin * streak_card_seconds(s, c);", "  return keep >= c.min_sim_s && keep > streak_card_seconds(s, c);"),
    (HG, "the margin not strict", "  return keep >= c.min_sim_s && keep > c.margin * streak_card_seconds(s, c);", "  return keep >= c.min_sim_s && keep >= c.margin * streak_card_seconds(s, c);"),
    (HG, "the margin doubled", "  return keep >= c.min_sim_s && keep > c.margin * streak_card_seconds(s, c);", "  return keep >= c.min_sim_s && keep > 2 * c.margin * streak_card_seconds(s, c);"),
    (HG, "the pull not charged", "  return s.sim_s + static_cast<double>(s.card_only_bytes) / c.pull_bytes_per_s;", "  return s.sim_s;"),
    (HG, "the pull at the push rate", "  return s.sim_s + static_cast<double>(s.card_only_bytes) / c.pull_bytes_per_s;", "  return s.sim_s + static_cast<double>(s.card_only_bytes) / c.push_bytes_per_s;"),
    (HG, "the pull twice (and the push back)", "  return s.sim_s + static_cast<double>(s.card_only_bytes) / c.pull_bytes_per_s;", "  return s.sim_s + 2 * static_cast<double>(s.card_only_bytes) / c.pull_bytes_per_s;"),
    (HG, "the simulator's time left out", "  return s.sim_s + static_cast<double>(s.card_only_bytes) / c.pull_bytes_per_s;", "  return static_cast<double>(s.card_only_bytes) / c.pull_bytes_per_s;"),
    (HG, "no launch cost", "  return c.launch_s + s.card_kernel_s.value_or(c.kernel_s_default) + static_cast<double>(s.push_bytes) / c.push_bytes_per_s;", "  return s.card_kernel_s.value_or(c.kernel_s_default) + static_cast<double>(s.push_bytes) / c.push_bytes_per_s;"),
    (HG, "no kernel time", "  return c.launch_s + s.card_kernel_s.value_or(c.kernel_s_default) + static_cast<double>(s.push_bytes) / c.push_bytes_per_s;", "  return c.launch_s + static_cast<double>(s.push_bytes) / c.push_bytes_per_s;"),
    (HG, "an unknown kernel time is free", "  return c.launch_s + s.card_kernel_s.value_or(c.kernel_s_default) + static_cast<double>(s.push_bytes) / c.push_bytes_per_s;", "  return c.launch_s + s.card_kernel_s.value_or(0.0) + static_cast<double>(s.push_bytes) / c.push_bytes_per_s;"),
    (HG, "the default always", "  return c.launch_s + s.card_kernel_s.value_or(c.kernel_s_default) + static_cast<double>(s.push_bytes) / c.push_bytes_per_s;", "  return c.launch_s + c.kernel_s_default + static_cast<double>(s.push_bytes) / c.push_bytes_per_s;"),
    (HG, "the push not charged", "  return c.launch_s + s.card_kernel_s.value_or(c.kernel_s_default) + static_cast<double>(s.push_bytes) / c.push_bytes_per_s;", "  return c.launch_s + s.card_kernel_s.value_or(c.kernel_s_default);"),
    (HG, "the push at the pull rate", "  return c.launch_s + s.card_kernel_s.value_or(c.kernel_s_default) + static_cast<double>(s.push_bytes) / c.push_bytes_per_s;", "  return c.launch_s + s.card_kernel_s.value_or(c.kernel_s_default) + static_cast<double>(s.push_bytes) / c.pull_bytes_per_s;"),
    (HG, "the card's side priced with the card-only bytes", "  return c.launch_s + s.card_kernel_s.value_or(c.kernel_s_default) + static_cast<double>(s.push_bytes) / c.push_bytes_per_s;", "  return c.launch_s + s.card_kernel_s.value_or(c.kernel_s_default) + static_cast<double>(s.push_bytes + s.card_only_bytes) / c.push_bytes_per_s;"),
    # the bans' reference (Round sp65, D)
    (HG, "ban: the pull never counted", "  return streak_rule && s.card_only_bytes > 0 ? streak_keep_seconds(s, c) : s.sim_s;", "  return s.sim_s;"),
    (HG, "ban: the pull always counted", "  return streak_rule && s.card_only_bytes > 0 ? streak_keep_seconds(s, c) : s.sim_s;", "  return streak_keep_seconds(s, c);"),
    (HG, "ban: the switch ignored", "  return streak_rule && s.card_only_bytes > 0 ? streak_keep_seconds(s, c) : s.sim_s;", "  return s.card_only_bytes > 0 ? streak_keep_seconds(s, c) : s.sim_s;"),
    (HG, "ban: the card's side added", "  return streak_rule && s.card_only_bytes > 0 ? streak_keep_seconds(s, c) : s.sim_s;", "  return streak_rule && s.card_only_bytes > 0 ? streak_keep_seconds(s, c) + streak_card_seconds(s, c) : s.sim_s;"),
    (HG, "ban: the switch inverted", "  return streak_rule && s.card_only_bytes > 0 ? streak_keep_seconds(s, c) : s.sim_s;", "  return !streak_rule && s.card_only_bytes > 0 ? streak_keep_seconds(s, c) : s.sim_s;"),
    # the zero-grid check
    (HG, "shape: never empty", "  return !grid[0] || !grid[1] || !grid[2] || !block[0] || !block[1] || !block[2];", "  return false;"),
    (HG, "shape: grid x only", "  return !grid[0] || !grid[1] || !grid[2] || !block[0] || !block[1] || !block[2];", "  return !grid[0];"),
    (HG, "shape: the grid only", "  return !grid[0] || !grid[1] || !grid[2] || !block[0] || !block[1] || !block[2];", "  return !grid[0] || !grid[1] || !grid[2];"),
    (HG, "shape: the block only", "  return !grid[0] || !grid[1] || !grid[2] || !block[0] || !block[1] || !block[2];", "  return !block[0] || !block[1] || !block[2];"),
    (HG, "shape: z ignored", "  return !grid[0] || !grid[1] || !grid[2] || !block[0] || !block[1] || !block[2];", "  return !grid[0] || !grid[1] || !block[0] || !block[1];"),
    (HG, "shape: all must be zero", "  return !grid[0] || !grid[1] || !grid[2] || !block[0] || !block[1] || !block[2];", "  return !grid[0] && !grid[1] && !grid[2] && !block[0] && !block[1] && !block[2];"),
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
    text = {h: open(os.path.join(src, "include", h)).read() for h in (BP, HG)}
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
        for other in (BP, HG):
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
