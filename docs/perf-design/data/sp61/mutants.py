#!/usr/bin/env python3
"""Round sp61: mutants of the in-block estimator (include/vgpu/exec/block_progress.hpp) and of the loop finder and the in-block rule
(include/vgpu/host_gpu.hpp). Each is a copy of one header with one change, put in a directory outside the tree that comes first on the include path;
tests/unit/test_host_gpu_logic.cpp is compiled against it and must fail.

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
    # lane_turns_left
    (BP, "turns: no step gives turns", "  if (l.step == 0) return 0;\n  const int64_t x = (l.is_unsigned", "  const int64_t x = (l.is_unsigned"),
    (BP, "turns: the offset is ignored", "static_cast<int64_t>(static_cast<int32_t>(iv_value))) + l.iv_offset;", "static_cast<int64_t>(static_cast<int32_t>(iv_value)));"),
    (BP, "turns: the offset is taken off", "static_cast<int64_t>(static_cast<int32_t>(iv_value))) + l.iv_offset;", "static_cast<int64_t>(static_cast<int32_t>(iv_value))) - l.iv_offset;"),
    (BP, "turns: always unsigned", "const int64_t x = (l.is_unsigned ? static_cast<int64_t>(iv_value)", "const int64_t x = (true ? static_cast<int64_t>(iv_value)"),
    (BP, "turns: the bound always unsigned", "const int64_t b = l.is_unsigned ? static_cast<int64_t>(bound_value)", "const int64_t b = true ? static_cast<int64_t>(bound_value)"),
    (BP, "turns: never unsigned", "const int64_t x = (l.is_unsigned ? static_cast<int64_t>(iv_value)", "const int64_t x = (false ? static_cast<int64_t>(iv_value)"),
    (BP, "turns: a lane past the bound has turns", "  if ((d > 0) != (l.step > 0)) return 0;   // (at the bound", "  if (false) return 0;   // (at the bound"),
    (BP, "turns: rounded down", "return static_cast<double>((a + s - 1) / s);", "return static_cast<double>(a / s);"),
    (BP, "turns: one more", "return static_cast<double>((a + s - 1) / s);", "return static_cast<double>((a + s) / s);"),
    (BP, "turns: the step not divided out", "return static_cast<double>((a + s - 1) / s);", "return static_cast<double>(a);"),
    (BP, "turns: a negative step is taken as is", "  const int64_t s = l.step > 0 ? l.step : -l.step;\n  const int64_t a = d > 0 ? d : -d;", "  const int64_t s = l.step;\n  const int64_t a = d > 0 ? d : -d;"),
    (BP, "turns: distance from the lane, not to the bound", "  const int64_t d = b - x;\n", "  const int64_t d = x - b;\n"),
    # turns_between
    (BP, "between: backwards counts", "  if ((d > 0) != (l.step > 0)) return 0;\n  const int64_t s = l.step > 0 ? l.step : -l.step;\n  return", "  if (d == 0) return 0;\n  const int64_t s = l.step > 0 ? l.step : -l.step;\n  return"),
    (BP, "between: part turns round up", "return static_cast<double>((d > 0 ? d : -d) / s);", "return static_cast<double>(((d > 0 ? d : -d) + s - 1) / s);"),
    (BP, "between: always signed", "  const int64_t x0 = l.is_unsigned ? static_cast<int64_t>(before)", "  const int64_t x0 = false ? static_cast<int64_t>(before)"),
    (BP, "between: no step guard", "  if (l.step == 0) return 0;\n  const int64_t x0", "  const int64_t x0"),
    # the measure
    (BP, "per turn: no completed turn gives none", "return m.turns >= 1 ? static_cast<double>(m.steps) / m.turns : static_cast<double>(m.steps);", "return m.turns >= 1 ? static_cast<double>(m.steps) / m.turns : 0.0;"),
    (BP, "per turn: a part turn divides", "return m.turns >= 1 ? static_cast<double>(m.steps) / m.turns", "return m.turns > 0 ? static_cast<double>(m.steps) / m.turns"),
    (BP, "per turn: the steps whatever the turns", "return m.turns >= 1 ? static_cast<double>(m.steps) / m.turns : static_cast<double>(m.steps);", "return static_cast<double>(m.steps);"),
    (BP, "per step: no time is a measure", "if (m.steps == 0 || !(m.seconds > 0)) return 0;", "if (m.steps == 0) return 0;"),
    (BP, "per step: no steps divides by zero", "if (m.steps == 0 || !(m.seconds > 0)) return 0;", "if (!(m.seconds > 0)) return 0;"),
    (BP, "per step: inverted", "return m.seconds / static_cast<double>(m.steps);", "return static_cast<double>(m.steps) / m.seconds;"),
    (BP, "block: the steps taken left out", "return b.steps + b.turns * std::max(per_turn, 0.0);", "return b.turns * std::max(per_turn, 0.0);"),
    (BP, "block: the turns left out", "return b.steps + b.turns * std::max(per_turn, 0.0);", "return b.steps;"),
    (BP, "block: a negative cost", "return b.steps + b.turns * std::max(per_turn, 0.0);", "return b.steps + b.turns * per_turn;"),
    # the launch from the blocks read
    (HG, "launch: no block read is an estimate", "if (blocks.empty() || total == 0 || !(per_step > 0) || !(per_turn > 0)) return std::nullopt;", "if (total == 0 || !(per_step > 0) || !(per_turn > 0)) return std::nullopt;"),
    (HG, "launch: no grid is an estimate", "if (blocks.empty() || total == 0 || !(per_step > 0) || !(per_turn > 0)) return std::nullopt;", "if (blocks.empty() || !(per_step > 0) || !(per_turn > 0)) return std::nullopt;"),
    (HG, "launch: no measure is an estimate", "if (blocks.empty() || total == 0 || !(per_step > 0) || !(per_turn > 0)) return std::nullopt;", "if (blocks.empty() || total == 0) return std::nullopt;"),
    (HG, "launch: not shared among threads", "const double steps = std::max(sum / static_cast<double>(blocks.size()) * static_cast<double>(total) / lanes, most);", "const double steps = std::max(sum / static_cast<double>(blocks.size()) * static_cast<double>(total), most);"),
    (HG, "launch: the longest block not a floor", "const double steps = std::max(sum / static_cast<double>(blocks.size()) * static_cast<double>(total) / lanes, most);", "const double steps = sum / static_cast<double>(blocks.size()) * static_cast<double>(total) / lanes;"),
    (HG, "launch: the blocks read are the grid", "const double steps = std::max(sum / static_cast<double>(blocks.size()) * static_cast<double>(total) / lanes, most);", "const double steps = std::max(sum / lanes, most);"),
    (HG, "launch: zero threads", "static_cast<double>(std::max<unsigned>(threads, 1));", "static_cast<double>(threads);"),
    (HG, "launch: the shortest block", "    most = std::max(most, s);", "    most = most == 0 ? s : std::min(most, s);"),
    # which blocks
    (HG, "inblock: the minimum grid is read inside", "if (blocks >= kSampleMinBlocks || blocks < 1) return 0;", "if (blocks > kSampleMinBlocks || blocks < 1) return 0;"),
    (HG, "inblock: any grid is read inside", "if (blocks >= kSampleMinBlocks || blocks < 1) return 0;", "if (blocks < 1) return 0;"),
    (HG, "inblock: no cap", "return std::min<uint64_t>(blocks, kSampleMaxBlocks);\n}\ninline uint64_t inblock_stride", "return blocks;\n}\ninline uint64_t inblock_stride"),
    (HG, "inblock: half the blocks", "return std::min<uint64_t>(blocks, kSampleMaxBlocks);\n}\ninline uint64_t inblock_stride", "return std::min<uint64_t>(blocks / 2, kSampleMaxBlocks);\n}\ninline uint64_t inblock_stride"),
    (HG, "inblock stride: always 1", "  const uint64_t n = inblock_blocks(blocks);\n  return n ? blocks / n : 1;", "  return 1;"),
    (HG, "inblock: the enter time doubled", "inline constexpr double kInBlockEnterSeconds = 0.001;", "inline constexpr double kInBlockEnterSeconds = 0.002;"),
    (HG, "inblock: the measure halved", "inline constexpr double kInBlockMeasureSeconds = 0.004;", "inline constexpr double kInBlockMeasureSeconds = 0.002;"),
    (HG, "inblock: the budget doubled", "inline constexpr double kInBlockBudgetSeconds = 0.012;", "inline constexpr double kInBlockBudgetSeconds = 0.024;"),
    # the rule
    (HG, "inside: without a followed loop", "return sample_blocks(w.blocks) == 0 && w.followed && inblock_blocks(w.blocks) != 0;", "return sample_blocks(w.blocks) == 0 && inblock_blocks(w.blocks) != 0;"),
    (HG, "inside: also for whole-block grids", "return sample_blocks(w.blocks) == 0 && w.followed && inblock_blocks(w.blocks) != 0;", "return w.followed && w.blocks != 0;"),
    (HG, "worth: small grids without the inside test", "if (!w.loop || (sample_blocks(w.blocks) == 0 && !sample_inside(w))) return false;", "if (!w.loop) return false;"),
    (HG, "worth: small grids never", "if (!w.loop || (sample_blocks(w.blocks) == 0 && !sample_inside(w))) return false;", "if (!w.loop || sample_blocks(w.blocks) == 0) return false;"),
    (HG, "worth: no loop test inside", "if (!w.loop || (sample_blocks(w.blocks) == 0 && !sample_inside(w))) return false;", "if ((!w.loop && !sample_inside(w)) || (sample_blocks(w.blocks) == 0 && !sample_inside(w))) return false;"),
    (HG, "sampled offload: the start is never charged", "e.start_s = driver_started ? 0 : std::max(c.start_cost_s, 0.0);", "e.start_s = 0;"),
    (HG, "sampled offload: the guessed card time is the default", "std::max(c.kernel_s_default, e.sim_s / kSampleCardSpeedup);", "c.kernel_s_default;"),
    (HG, "floor: the bound not counted", "return sample_inside(w) ? w.static_s * std::max(w.bound_turns, 1.0) : w.static_s;", "return w.static_s;"),
    (HG, "floor: the bound counted for whole-block grids", "return sample_inside(w) ? w.static_s * std::max(w.bound_turns, 1.0) : w.static_s;", "return w.static_s * std::max(w.bound_turns, 1.0);"),
    (HG, "floor: a bound of 0 zeroes the count", "return sample_inside(w) ? w.static_s * std::max(w.bound_turns, 1.0) : w.static_s;", "return sample_inside(w) ? w.static_s * w.bound_turns : w.static_s;"),
    (HG, "floor: the plain count in worth_sampling", "  return sample_static_estimate(w) >= sample_static_floor(w.live_bytes) ||", "  return w.static_s >= sample_static_floor(w.live_bytes) ||"),
    (HG, "bound turns: the step ignored", "  return std::floor(b / s);", "  return std::floor(b);"),
    (HG, "bound turns: rounded up", "  return std::floor(b / s);", "  return std::ceil(b / s);"),
    (HG, "bound turns: a negative bound", "static_cast<double>(bound < 0 ? -bound : bound), s", "static_cast<double>(bound), s"),
    (HG, "bound turns: no step guard", "  if (step == 0) return 0;\n  const double b", "  const double b"),
    # the loop finder
    (HG, "finder: an indirect jump is ignored", "    if (c.flow.indirect) return out;   // a jump", "    if (false) return out;   // a jump"),
    (HG, "finder: unreachable back edges count", "if (!reach[i] || t < 0 || static_cast<size_t>(t) > i || !reach[static_cast<size_t>(t)]) continue;", "if (t < 0 || static_cast<size_t>(t) > i) continue;"),
    (HG, "finder: a forward branch is a back edge", "if (!reach[i] || t < 0 || static_cast<size_t>(t) > i || !reach[static_cast<size_t>(t)]) continue;", "if (!reach[i] || t < 0 || !reach[static_cast<size_t>(t)]) continue;"),
    (HG, "finder: the first back edge, not the last", "    hi = std::max(hi, i);", "    if (!hi) hi = i;"),
    (HG, "finder: inner loops too", "    if (!inner && !crossed) top.emplace_back(lo, hi);", "    if (!crossed) top.emplace_back(lo, hi);"),
    (HG, "finder: crossed ranges too", "    if (!inner && !crossed) top.emplace_back(lo, hi);", "    if (!inner) top.emplace_back(lo, hi);"),
    (HG, "finder: no cap", "    if (out.size() >= exec::kProbeLoops) break;\n    const auto writes_in_loop", "    const auto writes_in_loop"),
    (HG, "finder: the bound may be written in the loop", "if (yr >= 0 && writes_in_loop(yr, nullptr) != 0) continue;", "(void)yr;"),
    (HG, "finder: the counter may be written twice", "if (writes_in_loop(xr, &at) != 1 || code[at].kind", "if (writes_in_loop(xr, &at) < 1 || code[at].kind"),
    (HG, "finder: any write is a counter", "if (writes_in_loop(xr, &at) != 1 || code[at].kind != LoopCodeInsn::Kind::AddImm || code[at].add_dst != xr) continue;", "if (writes_in_loop(xr, &at) != 1) continue;"),
    (HG, "finder: no step test", "        if (step == 0) continue;\n", ""),
    (HG, "finder: no bound test", "        if (y.kind == exec::ProbeOperand::None) continue;\n", ""),
    (HG, "finder: the through-register's counter need not be one", "if (iv < 0 || writes_in_loop(iv, &at2) != 1 || code[at2].kind != LoopCodeInsn::Kind::AddImm || code[at2].add_dst != iv || code[at2].add_src != iv) continue;", "if (iv < 0) continue;"),
    (HG, "finder: the through-register offset dropped", "          offset = code[at].add_imm;\n", ""),
    (HG, "finder: any last writer of the predicate", "if (!have || code[j].kind != LoopCodeInsn::Kind::Compare) continue;", "if (!have) continue;"),
    (HG, "finder: the compare may be before the loop", "      while (j-- > lo) {", "      while (j-- > 0) {"),
    (HG, "finder: the compare need not write the branch's predicate", "      if (cmp.cmp_pred != p && cmp.cmp_pred2 != p) continue;\n", ""),
    (HG, "finder: the bound side only", "      for (int side = 0; side < 2 && !found; ++side) {", "      for (int side = 0; side < 1 && !found; ++side) {"),
    (HG, "finder: unconditional back edges decide", "      if (code[i].branch_pred >= 0 && code[i].flow.target >= static_cast<int64_t>(lo) && static_cast<size_t>(code[i].flow.target) <= i) branches.push_back(i);", "      if (code[i].flow.target >= static_cast<int64_t>(lo) && static_cast<size_t>(code[i].flow.target) <= i) branches.push_back(i);"),
    (HG, "finder: exit branches do not decide", "          (static_cast<size_t>(code[i].flow.target) < lo || static_cast<size_t>(code[i].flow.target) > hi))\n        branches.push_back(i);", "          false)\n        branches.push_back(i);"),
    (HG, "finder: signedness lost", "        f.loop.is_unsigned = cmp.cmp_unsigned;\n", ""),
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
