#!/usr/bin/env python3
"""Round sp72: mutants of the block runner (Machine::run_block and what is around it). Each is one edit of amd/src/gcn_exec.cpp in a
build tree (put back afterwards); test_amd_gcn_blocks_same_bits is rebuilt, run, and must fail.

  mutants.py --src <tree> --build <tree>/build [--only N,M] [--lock /tmp/pantheonsim-builder.lock]

The tree is best one configured with -DVGPU_LTO=OFF (a mutant is then one compile and a link).
"""
import argparse
import os
import subprocess

EX = "amd/src/gcn_exec.cpp"
M = [
    ("a whole block counts nothing", "      for (const gcn::TierBlock::Add& a : b.adds) *reinterpret_cast<uint64_t*>(counters + a.at) += a.n;\n", ""),
    ("a whole block does not count its instructions", "      add(stats.instructions, static_cast<uint32_t>(fresh->insts.size()));\n", ""),
    ("a whole block does not count by unit", "      for (int u = 1; u < 5; ++u) add(*unit_slot[u], unit[u]);\n", ""),
    ("a whole block does not count the mix", "      for (int i = 0; i < static_cast<int>(gcn::Mix::Count); ++i) add(stats.counts.mix[i], mix[i]);\n", ""),
    ("a whole block does not count the matrix operations", "      for (int i = 0; i < static_cast<int>(gcn::MopsType::Count); ++i) add(stats.counts.mops[i], mops[i]);\n", ""),
    ("a whole block counts each unit one too many", "      ++unit[cur->fast.unit];", "      unit[cur->fast.unit] += 2;"),
    ("a block runs past the turn's room", "const size_t len = b.insts.size(), n = len < room ? len : static_cast<size_t>(room);", "const size_t len = b.insts.size(), n = len;"),
    ("the room is one instruction too long", "n + 1 < slice ? slice - 1 - n : 0;", "n < slice ? slice - n : 0;"),
    ("a handed-back instruction leaves the program counter past it", "    w.pc -= in.size;\n    if (i) count_block_part(b, i);\n    w.last = &in;", "    if (i) count_block_part(b, i);\n    w.last = &in;"),
    ("a handed-back instruction is counted", "    w.pc -= in.size;\n    if (i) count_block_part(b, i);\n    w.last = &in;", "    w.pc -= in.size;\n    count_block_part(b, i + 1);\n    w.last = &in;"),
    ("an instruction that gives up the turn after it is done is a step", "      count_block_part(b, i + 1);\n      w.last = &in;\n      *end = kBlockStop;\n      return i;",
     "      count_block_part(b, i + 1);\n      w.last = &in;\n      *end = kBlockStop;\n      return i + 1;"),
    ("an instruction that gives up the turn after it is done does not end the turn", "      count_block_part(b, i + 1);\n      w.last = &in;\n      *end = kBlockStop;\n      return i;",
     "      count_block_part(b, i + 1);\n      w.last = &in;\n      *end = kBlockOn;\n      return i + 1;"),
    ("a gated store goes to the interpreter instead of ending the turn", "    *end = done == 3 ? kBlockStop : kBlockInterp;", "    *end = kBlockInterp;"),
    ("a handed-back instruction ends the turn", "    *end = done == 3 ? kBlockStop : kBlockInterp;", "    *end = kBlockStop;"),
    ("a block goes on past a branch", "      if (cur->fast.unit == gcn::tier::kUnitBranch) break;", ""),
    ("the wave's state is not asked", "  static bool block_state(const Wave& w) { return !(w.vgpr_msb || w.gpr_idx || w.vskip || !FloatMode::is_default(w.mode)); }",
     "  static bool block_state(const Wave&) { return true; }"),
    ("the Machine's instructions are taken into a block", "  static bool in_block(const Inst& in) { return in.fast.run && !in.fast.mach; }", "  static bool in_block(const Inst& in) { return in.fast.run; }"),
    ("the steps of a block are not counted in the turn", "          n += ran;\n", ""),
    ("the branch target is not handed to the handler", "      r.target = in.target;\n      w.pc += in.size;\n      done = in.fast.run(in.fast, r);", "      w.pc += in.size;\n      done = in.fast.run(in.fast, r);"),
    ("the room's end counts nothing", "    if (i == n) {   // the turn's room ended first\n      count_block_part(b, i);", "    if (i == n) {   // the turn's room ended first"),
    ("the last instruction of a block is run twice when the room ends on it", "    if (__builtin_expect(i == len, 1)) {   // the whole block", "    if (__builtin_expect(i + 1 >= len, 1)) {   // the whole block"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    a = ap.parse_args()
    only = {int(x) for x in a.only.split(",") if x}
    path = os.path.join(a.src, EX)
    original = open(path).read()
    killed = total = 0
    try:
        for i, (name, old, new) in enumerate(M):
            if only and i not in only:
                continue
            total += 1
            if original.count(old) != 1:
                print("%2d NOT APPLIED (%d matches): %s" % (i, original.count(old), name), flush=True)
                continue
            open(path, "w").write(original.replace(old, new))
            b = subprocess.run(["flock", a.lock, "nice", "-n", "19", "ninja", "-C", a.build, "-j3", "test_amd_gcn_blocks_same_bits"], capture_output=True, text=True)
            if b.returncode:
                print("%2d DOES NOT BUILD: %s\n%s" % (i, name, b.stdout[-600:]), flush=True)
                continue
            try:
                t = subprocess.run([os.path.join(a.build, "test_amd_gcn_blocks_same_bits")], capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired:   # (the unmutated test takes 15 to 40 s)
                killed += 1
                print("%2d killed: %s  [the test does not end: stopped after 300 s]" % (i, name), flush=True)
                continue
            out = (t.stdout + t.stderr)
            why = [l.strip() for l in out.splitlines() if "runs differ" in l or "VCHECK" in l][:1]
            if t.returncode:
                killed += 1
                print("%2d killed: %s  [%s]" % (i, name, why[0][:110] if why else "exit %d" % t.returncode), flush=True)
            else:
                print("%2d SURVIVED: %s" % (i, name), flush=True)
    finally:
        open(path, "w").write(original)
    print("%d of %d killed" % (killed, total))


main()
