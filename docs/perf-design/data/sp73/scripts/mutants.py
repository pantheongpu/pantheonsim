#!/usr/bin/env python3
"""Round sp73: mutants of the block runner (Machine::run_block and what is around it; the list is of its second form, with stretches,
the Machine's instructions as members and waves that go on in a block). Each is one edit of amd/src/gcn_exec.cpp (or of the file a mutant names) in a
build tree (put back afterwards); test_amd_gcn_blocks_same_bits is rebuilt, run, and must fail.

  mutants.py --src <tree> --build <tree>/build [--only N,M] [--lock /tmp/pantheonsim-builder.lock]

The tree is best one configured with -DVGPU_LTO=OFF (a mutant is then one compile and a link).
"""
import argparse
import os
import subprocess

EX = "amd/src/gcn_exec.cpp"
M = [
    ("a whole stretch counts nothing", "        for (const gcn::TierBlock::Add *a = b.adds.data() + s->first_add, *e = a + s->n_adds; a != e; ++a) *reinterpret_cast<uint64_t*>(counters + a->at) += a->n;\n", ""),
    ("a whole stretch does not count its instructions", "        add(stats.instructions, end - begin);\n", ""),
    ("a whole stretch does not count by unit", "        for (int u = 1; u < 5; ++u) add(*unit_slot[u], unit[u]);\n", ""),
    ("a whole stretch does not count the mix", "        for (int i = 0; i < static_cast<int>(gcn::Mix::Count); ++i) add(stats.counts.mix[i], mix[i]);\n", ""),
    ("a whole stretch does not count the matrix operations", "        for (int i = 0; i < static_cast<int>(gcn::MopsType::Count); ++i) add(stats.counts.mops[i], mops[i]);\n", ""),
    ("a whole stretch counts each unit one too many", "++unit[x.fast.unit];", "unit[x.fast.unit] += 2;"),
    ("a stretch that ran in part counts nothing", "      } else if (i != first) {\n        count_range(b, first, i);\n      }", "      }"),
    ("a block runs past the turn's room", "const uint32_t limit = room < len - from ? from + static_cast<uint32_t>(room) : len;", "const uint32_t limit = len;"),
    ("the room is one instruction too long", "n + 1 < slice ? slice - 1 - n : 0;", "n < slice ? slice - n : 0;"),
    ("a handed-back instruction leaves the program counter past it", "          w.pc -= in.size;\n          count_range(b, first, i);", "          count_range(b, first, i);"),
    ("a handed-back instruction is counted", "          w.pc -= in.size;\n          count_range(b, first, i);", "          w.pc -= in.size;\n          count_range(b, first, i + 1);"),
    ("the instructions before a handed-back one are not counted", "          w.pc -= in.size;\n          count_range(b, first, i);", "          w.pc -= in.size;"),
    ("a wait that ends the turn after it ran is a step", "            count_one(in);\n            return leave_block(w, b, from, i + 1, i - from, kBlockStop, end);",
     "            count_one(in);\n            return leave_block(w, b, from, i + 1, i + 1 - from, kBlockStop, end);"),
    ("a wait that ends the turn after it ran does not end it", "            count_one(in);\n            return leave_block(w, b, from, i + 1, i - from, kBlockStop, end);",
     "            count_one(in);\n            return leave_block(w, b, from, i + 1, i + 1 - from, kBlockOn, end);"),
    ("a wait that ends the turn is not counted", "            count_one(in);\n            return leave_block(w, b, from, i + 1, i - from, kBlockStop, end);",
     "            return leave_block(w, b, from, i + 1, i - from, kBlockStop, end);"),
    ("a gated store goes to the interpreter instead of ending the turn", "t == 3 ? kBlockStop : kBlockInterp, end);", "kBlockInterp, end);"),
    ("a handed-back store or wait ends the turn", "t == 3 ? kBlockStop : kBlockInterp, end);", "kBlockStop, end);"),
    ("a store or a wait that ran inside a block is not counted", "          return leave_block(w, b, from, i, i - from, t == 3 ? kBlockStop : kBlockInterp, end);\n        }\n        count_one(in);", "          return leave_block(w, b, from, i, i - from, t == 3 ? kBlockStop : kBlockInterp, end);\n        }"),
    ("a gated store leaves the program counter past it", "          w.pc -= in.size;\n          return leave_block(w, b, from, i, i - from, t == 3 ? kBlockStop : kBlockInterp, end);", "          return leave_block(w, b, from, i, i - from, t == 3 ? kBlockStop : kBlockInterp, end);"),
    ("a Machine instruction that gives up the turn or hands back always ends the turn", "i - from, t ? kBlockStop : kBlockInterp, end);", "i - from, kBlockStop, end);"),
    ("a Machine instruction that gives up the turn goes to the interpreter", "i - from, t ? kBlockStop : kBlockInterp, end);", "i - from, kBlockInterp, end);"),
    ("an error of a Machine instruction is not told where it was", "        block_pc = w.pc;\n        const int t = tier_mach(w, g, in);", "        const int t = tier_mach(w, g, in);"),
    ("a block goes on past a branch", "      if (cur->fast.unit == gcn::tier::kUnitBranch) break;", ""),
    ("the wave's state is not asked", "  static bool block_state(const Wave& w) { return !(w.vgpr_msb || w.gpr_idx || w.vskip || !FloatMode::is_default(w.mode)); }",
     "  static bool block_state(const Wave&) { return true; }"),
    ("the steps of a block are not counted in the turn", "          n += ran;\n", ""),
    ("the branch target is not handed to the handler", "        r.target = in.target;\n        w.pc += in.size;\n        if (__builtin_expect(in.fast.run(in.fast, r) != 1, 0)) {", "        w.pc += in.size;\n        if (__builtin_expect(in.fast.run(in.fast, r) != 1, 0)) {"),
    ("a wave goes on in the block it left without asking where it is", "    if (b && at < b->insts.size() && b->insts[at]->pc == w.pc) {", "    if (b && at < b->insts.size()) {"),
    ("a wave that ran the instruction after the one it left at goes on one instruction early", "      in = b->insts[++at];\n      w.blk_at = at;", "      in = b->insts[at];"),
    ("a store is never taken for one that can end the turn", "gcn::tier::gives_up_turn(cur->fast) || cur->fast.store ? gcn::TierBlock::kTurn", "gcn::tier::gives_up_turn(cur->fast) ? gcn::TierBlock::kTurn"),
    ("a wait is never taken for one that can end the turn", "gcn::tier::gives_up_turn(cur->fast) || cur->fast.store ? gcn::TierBlock::kTurn", "cur->fast.store ? gcn::TierBlock::kTurn"),
    ("an error in a wave's turn is told the address the turn began at", "        throw m.at_instruction(e, m.block_pc);", "        throw m.at_instruction(e, w.pc - 4);"),
    ("a bound run is used with lanes off", "if (run.end <= limit && (lanes == 64 ? ex == ~uint64_t{0} : static_cast<uint32_t>(ex) == 0xFFFFFFFFu)) {", "if (run.end <= limit) {"),
    ("a bound run of a wave of 64 lanes asks only the first 32", "(lanes == 64 ? ex == ~uint64_t{0} : static_cast<uint32_t>(ex) == 0xFFFFFFFFu)", "(static_cast<uint32_t>(ex) == 0xFFFFFFFFu)"),
    ("a bound run runs past the turn's room", "if (run.end <= limit && (lanes == 64", "if ((lanes == 64"),
    ("a bound run leaves the program counter where it was", "          w.pc = insts[run.end - 1]->pc + insts[run.end - 1]->size;\n", ""),
    ("a whole bound run counts nothing", "            for (const gcn::TierBlock::Add *a = b.adds.data() + run.first_add, *e = a + run.n_adds; a != e; ++a) *reinterpret_cast<uint64_t*>(counters + a->at) += a->n;\n", ""),
    ("a bound run entered in its middle counts nothing", "          } else {\n            count_range(b, i, run.end);\n          }", "          }"),
    ("a bound run entered in its middle starts at its first operation", "b.bound_ops.data() + run.first_op + (i - run.begin), run.end - i,", "b.bound_ops.data() + run.first_op, run.end - i,"),
    ("an instruction handed back inside a bound run leaves the program counter past it", "              w.pc -= x.size;\n              count_range(b, begin, i);", "              count_range(b, begin, i);"),
    ("the instructions of a bound run that ran one at a time are not counted", "          count_range(b, begin, i);\n          if (i != run.end)", "          if (i != run.end)"),
    ("every constant of a block reads the first constant's row", "if (rows[i][0] == value) return ~static_cast<int32_t>(i);", "if (true) return ~static_cast<int32_t>(i);"),
    ("a bound operation's missing source is register 0, not its first source", "out->src[k] = !s.kind ? out->src[0] :", "out->src[k] = !s.kind ? 0 :", "amd/src/gcn_tier.cpp"),
    ("a bound operation with a modifier on a source is bound", "if (s.neg || s.abs || (s.kind != kVgpr && s.kind != kConst)", "if ((s.kind != kVgpr && s.kind != kConst)", "amd/src/gcn_tier.cpp"),
    ("a scalar register source is bound as a vector register", "(s.kind != kVgpr && s.kind != kConst) || (s.kind == kVgpr && s.index >= kVgprs)) return false;", "(s.kind != kVgpr && s.kind != kConst && s.kind != kSgpr) || (s.kind == kVgpr && s.index >= kVgprs)) return false;", "amd/src/gcn_tier.cpp"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    a = ap.parse_args()
    only = {int(x) for x in a.only.split(",") if x}
    killed = total = 0
    for i, m in enumerate(M):
        name, old, new = m[0], m[1], m[2]
        path = os.path.join(a.src, m[3] if len(m) > 3 else EX)
        if only and i not in only:
            continue
        total += 1
        original = open(path).read()
        if original.count(old) != 1:
            print("%2d NOT APPLIED (%d matches): %s" % (i, original.count(old), name), flush=True)
            continue
        try:
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
            why = [l.strip() for l in out.splitlines() if "runs differ" in l or "differ with" in l or "VCHECK" in l][:1]
            if t.returncode:
                killed += 1
                print("%2d killed: %s  [%s]" % (i, name, why[0][:110] if why else "exit %d" % t.returncode), flush=True)
            else:
                print("%2d SURVIVED: %s" % (i, name), flush=True)
        finally:
            open(path, "w").write(original)
    print("%d of %d killed" % (killed, total))


main()
