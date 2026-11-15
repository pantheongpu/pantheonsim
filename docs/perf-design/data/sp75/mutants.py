#!/usr/bin/env python3
"""Round sp75: mutants of the bounded-block rule (include/vgpu/exec/launch.hpp part_block_limit; include/vgpu/host_gpu.hpp kSampleBlockSeconds,
kSampleBlockCapSeconds, bounded_blocks, bounded_timeout_seconds, sample_need_seconds, sample_bounded_small, sample_floor, sample_refusal). Each is
a copy of one header with one change, put in a directory outside the tree that comes first on the include path;
tests/unit/test_host_gpu_logic.cpp is compiled against it and must fail. (The runner is round sp66's data/sp66/mutants.py with this list.)

  mutants.py --src <tree> --work <dir outside the tree> [--only N] [--lock /tmp/pantheonsim-builder.lock]
"""
import argparse
import fcntl
import os
import subprocess
import time

LP = "vgpu/exec/launch.hpp"
HG = "vgpu/host_gpu.hpp"
WANT = "  const double want = kPartBlockSlack * p.need_s * static_cast<double>(threads ? threads : 1) / static_cast<double>(units ? units : 1);"
SMALL = "inline bool sample_bounded_small(const SampleCandidate& w) { return w.bounded && sample_blocks(w.blocks) == 0 && !sample_inside(w) && bounded_blocks(w.blocks) != 0; }"
FLOOR = "  return sample_bounded_small(w) ? kSampleStaticFloorSeconds : sample_static_floor(w.live_bytes);"
GRID = "  if (sample_blocks(w.blocks) == 0 && !sample_inside(w) && !sample_bounded_small(w)) return NoSample::SmallGrid;"
M = [
    (LP, "limit: no slack", "inline constexpr double kPartBlockSlack = 1.4;", "inline constexpr double kPartBlockSlack = 1.0;"),
    (LP, "limit: a slack of 1.25", "inline constexpr double kPartBlockSlack = 1.4;", "inline constexpr double kPartBlockSlack = 1.25;"),
    (LP, "limit: a slack of 2", "inline constexpr double kPartBlockSlack = 1.4;", "inline constexpr double kPartBlockSlack = 2.0;"),
    (LP, "limit: the host threads not counted", WANT, "  const double want = kPartBlockSlack * p.need_s / static_cast<double>(units ? units : 1);"),
    (LP, "limit: the grid not counted", WANT, "  const double want = kPartBlockSlack * p.need_s * static_cast<double>(threads ? threads : 1);"),
    (LP, "limit: the cap not asked", "  if (want > p.block_cap_s) return 0;\n", ""),
    (LP, "limit: the cap instead of no block", "  if (want > p.block_cap_s) return 0;", "  if (want > p.block_cap_s) return p.block_cap_s;"),
    (LP, "limit: at the cap is over it", "  if (want > p.block_cap_s) return 0;", "  if (want >= p.block_cap_s * 0.8) return 0;"),
    (LP, "limit: the need not asked", "  if (!(p.need_s > 0)) return p.block_limit_s;", "  if (true) return p.block_limit_s;"),
    (LP, "limit: under the limit of a sampled grid", "  return want > p.block_limit_s ? want : p.block_limit_s;", "  return want;"),
    (LP, "limit: on when off", "  if (!(p.block_limit_s > 0)) return 0;", "  if (!(p.block_limit_s > 0)) return 0.018;"),
    (HG, "constant: a block of 18 ms (under 20 ms to spare)", "inline constexpr double kSampleBlockSeconds = 0.015;", "inline constexpr double kSampleBlockSeconds = 0.018;"),
    (HG, "constant: a block of 30 ms (two runs outlast the wait)", "inline constexpr double kSampleBlockSeconds = 0.015;", "inline constexpr double kSampleBlockSeconds = 0.030;"),
    (HG, "constant: a block of 5 ms", "inline constexpr double kSampleBlockSeconds = 0.015;", "inline constexpr double kSampleBlockSeconds = 0.005;"),
    (HG, "constant: a cap of 40 ms", "inline constexpr double kSampleBlockCapSeconds = 0.060;", "inline constexpr double kSampleBlockCapSeconds = 0.040;"),
    (HG, "constant: a cap of 200 ms", "inline constexpr double kSampleBlockCapSeconds = 0.060;", "inline constexpr double kSampleBlockCapSeconds = 0.200;"),
    (HG, "blocks: a grid of one block", "inline constexpr uint64_t kBoundedMinBlocks = 2;", "inline constexpr uint64_t kBoundedMinBlocks = 1;"),
    (HG, "blocks: a sampled grid too", "  if (blocks >= kSampleMinBlocks || blocks < kBoundedMinBlocks) return 0;", "  if (blocks < kBoundedMinBlocks) return 0;"),
    (HG, "blocks: every block", "  return std::min<uint64_t>(blocks, kSampleMaxBlocks);\n}\ninline uint64_t bounded_stride", "  return blocks;\n}\ninline uint64_t bounded_stride"),
    (HG, "wait: not extended", "inline double bounded_timeout_seconds(uint64_t live_bytes) { return sample_timeout_seconds(live_bytes) + 2 * kSampleBlockCapSeconds; }",
     "inline double bounded_timeout_seconds(uint64_t live_bytes) { return sample_timeout_seconds(live_bytes); }"),
    (HG, "wait: one run of a block", "inline double bounded_timeout_seconds(uint64_t live_bytes) { return sample_timeout_seconds(live_bytes) + 2 * kSampleBlockCapSeconds; }",
     "inline double bounded_timeout_seconds(uint64_t live_bytes) { return sample_timeout_seconds(live_bytes) + kSampleBlockCapSeconds; }"),
    (HG, "need: the margin not applied", "  return std::max(c.margin * fixed / (1 - share), c.min_sim_s);", "  return std::max(fixed / (1 - share), c.min_sim_s);"),
    (HG, "need: the card's share not counted", "  return std::max(c.margin * fixed / (1 - share), c.min_sim_s);", "  return std::max(c.margin * fixed, c.min_sim_s);"),
    (HG, "need: always the short kernel", "  if (short_kernel < kSampleCardSpeedup * c.kernel_s_default) return std::max(short_kernel, c.min_sim_s);", "  if (true) return std::max(short_kernel, c.min_sim_s);"),
    (HG, "need: never the short kernel", "  if (short_kernel < kSampleCardSpeedup * c.kernel_s_default) return std::max(short_kernel, c.min_sim_s);\n", ""),
    (HG, "need: no card too slow", "  if (!(share < 1)) return 1e30;", "  if (!(share < 1)) return 1.0;"),
    (HG, "need: without the floor of the size rule", "  if (short_kernel < kSampleCardSpeedup * c.kernel_s_default) return std::max(short_kernel, c.min_sim_s);",
     "  if (short_kernel < kSampleCardSpeedup * c.kernel_s_default) return short_kernel;"),
    (HG, "small grid: never", SMALL, "inline bool sample_bounded_small(const SampleCandidate&) { return false; }"),
    (HG, "small grid: without the switch", SMALL, SMALL.replace("return w.bounded && ", "return ")),
    (HG, "small grid: a followed loop too", SMALL, SMALL.replace(" && !sample_inside(w)", "")),
    (HG, "small grid: a grid of one block", SMALL, SMALL.replace(" && bounded_blocks(w.blocks) != 0", "")),
    (HG, "floor: the live memory's", FLOOR, "  return sample_static_floor(w.live_bytes);"),
    (HG, "floor: none for a small grid", FLOOR, "  return sample_bounded_small(w) ? 0.0 : sample_static_floor(w.live_bytes);"),
    (HG, "floor: 30 ms for every grid", FLOOR, "  return kSampleStaticFloorSeconds;"),
    (HG, "refusal: a small grid never refused", GRID, "  if (sample_blocks(w.blocks) == 0 && !sample_inside(w) && !w.bounded) return NoSample::SmallGrid;"),
    (HG, "refusal: a small grid always refused", GRID, "  if (sample_blocks(w.blocks) == 0 && !sample_inside(w)) return NoSample::SmallGrid;"),
    (HG, "refusal: the floor not asked", "  if (!(sample_static_estimate(w) >= sample_floor(w) || sample_affordable(w.age_s, w.spent_s, sample_expected_seconds(w)))) return NoSample::Floor;\n", ""),
    (HG, "afford: the two runs at the cap not counted", "  return std::max(w.expected_s, 0.0) + (sample_bounded_small(w) ? 2 * kSampleBlockCapSeconds : 0.0);", "  return std::max(w.expected_s, 0.0);"),
    (HG, "afford: one run at the cap", "(sample_bounded_small(w) ? 2 * kSampleBlockCapSeconds : 0.0);", "(sample_bounded_small(w) ? kSampleBlockCapSeconds : 0.0);"),
    (HG, "afford: counted for every grid", "(sample_bounded_small(w) ? 2 * kSampleBlockCapSeconds : 0.0);", "2 * kSampleBlockCapSeconds;"),
    (HG, "afford: the rule asks the guess alone", "sample_affordable(w.age_s, w.spent_s, sample_expected_seconds(w))))", "sample_affordable(w.age_s, w.spent_s, w.expected_s)))"),
    (HG, "refusal: two reasons with one text", '    case NoSample::ShortBefore: return "its earlier whole launch was short";', '    case NoSample::ShortBefore: return "the kernel has no loop";'),
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
