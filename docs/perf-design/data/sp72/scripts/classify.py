#!/usr/bin/env python3
"""Where the AMD engine's CPU goes, by kind of host work (round sp72; after round sp48's classify.py, without perf).

  python3 -I classify.py <libamdhip64.so> <samples.txt>...     (samples: tools/bench-amd/profile/sigprof.c, "<count> <object>+<offset>")

Every sample whose instruction is inside the simulator's library is put in one class, by the function it is in and,
inside the tier's handlers and lane kernels, by the host instruction at the sampled address:
  loop     run_round, run_wave_blocks, Machine::step / step_at / tier_step / run_block / fetch_*: dispatch and the counters
  decode   scalar host instructions of the tier's handlers: operand kinds, the EXEC tests, the calls of the lane kernels
  rows     vector moves with a memory operand in the handlers and the lane kernels: a register row (or a temporary row) loaded
           or stored, a constant spread
  lanes    every other instruction of the lane kernels and every other vector instruction of the handlers: the guest
           instruction's own arithmetic (a load folded into it and the kernel's loop control included)
  mem      the Machine's memory instructions (tier_global, memory_run, global_access, the LDS paths) and the memory manager
  interp   the interpreter's own code for an instruction (step_interp, vector_alu, scalar_alu ...)
  other    everything else in the library (launch, group set-up, the HIP API)
Samples outside the library (the program's host code, libc, the kernel's return path) are reported as "outside".
A SIGPROF sample names the instruction that was about to run, so a slow instruction is charged to the one after it:
the function classes are solid, the decode/rows/lanes split is approximate.
"""
import bisect
import collections
import re
import subprocess
import sys

LOOP = ("run_round", "run_wave_blocks", "run_groups", "Machine::step(", "Machine::step_at", "Machine::tier_step", "Machine::run_block",
        "Machine::fetch", "Machine::block_at", "Machine::count_block_part")
MEM = ("tier_global", "tier_narrow", "tier_mach", "memory_run", "wave_scalars", "global_access", "MemoryManager", "narrow_run",
       "load_quad", "load_scalar", "scalar_location", "lookup_host", "Machine::load", "Machine::store", "tier_lds", "ds_", "memcpy",
       "scalar_load", "tier_smem", "mem::", "Memory")
VEC_MOVE = re.compile(r"^v?(mov[au]p[sd]|movdq[au]\w*|movd|movq|pbroadcast\w+|broadcasts[sd]|broadcast[fi]128|pmovzx\w+|pmovsx\w+|maskmov\w+|pmaskmov\w+|insert[fi]128|extract[fi]128)$")


def main():
    lib = sys.argv[1]
    counts = collections.Counter()
    libname = lib.rsplit("/", 1)[-1].split(".so")[0]
    outside = 0
    for path in sys.argv[2:]:
        for line in open(path):
            if line.startswith("#"):
                continue
            parts = line.split()
            obj, off = parts[1].rsplit("+", 1)
            if libname in obj:
                counts[int(off, 16)] += int(parts[0])
            else:
                outside += int(parts[0])
    # function symbols
    syms = []
    for line in subprocess.run(["nm", "-C", "-n", "--defined-only", lib], capture_output=True, text=True).stdout.splitlines():
        p = line.split(" ", 2)
        if len(p) == 3 and p[1] in "tTwW":
            syms.append((int(p[0], 16), p[2]))
    starts = [s[0] for s in syms]
    # instructions at the sampled addresses
    insn = {}
    lo, hi = min(counts), max(counts)
    dis = subprocess.run(["objdump", "-d", "--no-show-raw-insn", "--start-address=%#x" % lo, "--stop-address=%#x" % (hi + 16), lib],
                         capture_output=True, text=True).stdout
    for line in dis.splitlines():
        m = re.match(r"^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$", line)
        if m:
            a = int(m.group(1), 16)
            if a in counts:
                insn[a] = (m.group(2), m.group(3))
    cls = collections.Counter()
    by_fn = collections.Counter()
    for a, n in counts.items():
        i = bisect.bisect_right(starts, a) - 1
        fn = syms[i][1] if i >= 0 else "?"
        by_fn[fn] += n
        if any(k in fn for k in LOOP):
            c = "loop"
        elif "_impl::" in fn or "gcn::(anonymous namespace)::" in fn or "gcn::tier" in fn or "packed_math" in fn or "simd::" in fn:
            mn, ops = insn.get(a, ("?", ""))
            vec = "ymm" in ops or "xmm" in ops or "zmm" in ops
            if vec and "(" in ops and VEC_MOVE.match(mn):
                c = "rows"
            elif vec or "_impl::" in fn:   # (a lane kernel the compiler left scalar -- popcount, a 64-bit multiply -- is arithmetic all the same)
                c = "lanes"
            else:
                c = "decode"
        elif any(k in fn for k in MEM):
            c = "mem"
        elif "step_interp" in fn or "vector_alu" in fn or "scalar_alu" in fn or "Machine::" in fn:
            c = "interp"
        else:
            c = "other"
        cls[c] += n
    total = sum(cls.values())
    print("samples in the library: %d (outside it: %d)" % (total, outside))
    for c in ("loop", "decode", "rows", "lanes", "mem", "interp", "other"):
        print("  %-7s %6.1f%%" % (c, 100.0 * cls[c] / total))
    print("top functions:")
    for fn, n in by_fn.most_common(14):
        print("  %5.1f%%  %s" % (100.0 * n / total, fn[:150]))


main()
