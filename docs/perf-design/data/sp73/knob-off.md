# The knob off: how its costs were found (round sp73)

Device CPU seconds of a bench row (`scripts/quick.sh`: best of n single runs, the builds taking turns, pinned to cpus 14,15), knob off in
every build. A ratio is the base's time over the build's (below 1: slower than the base). `noop` is the control.

## 1. valu_f64, round sp72 as committed (`work`)

```
valu_f64   rx6800  base=0.16455  work=0.213347 p2=0.164454 noop=0.165983      work 0.771 of the base
valu_f64   mi300x  base=0.119292 work=0.13686  p2=0.118218 noop=0.120018      work 0.872
valu_f32   rx6800  base=0.124358 work=0.125811 p2=0.125633 noop=0.125354
valu_f32   mi300x  base=0.072978 work=0.073473 p2=0.074086 noop=0.073538
valu_int   rx6800  base=0.13988  work=0.140007 p2=0.141138 noop=0.139954
valu_int   mi300x  base=0.089267 work=0.087968 p2=0.089031 noop=0.08967
```

`p2` is the build with `Machine::block_pc`. Not the stack's place: `scripts/stackshift.sh valu_f64 gfx1030 amd/rx6800 base work noop`,
every thread's stack moved by 16 to 3,856 bytes in steps of 256 (best of 5 each): base 0.157 to 0.168 s, noop 0.162 to 0.168, work
0.203 to 0.216 at every one of the 16 offsets. Where the time went (instruction-pointer samples, 20 runs at 10 kHz, in the library):
3,229 samples for the base, 4,241 for `work`; `h_f64r` 953 to 1,506, the dispatch loop 517 to 929, `b_scc0` 29 to 153, the lane kernels
unchanged (1,617 to 1,565). The loop's disassembly: `mov 0x238(%rbx),%r15` in the base, `mov 0x238(%rbx),%rax; mov %rax,-0x68(%rbp)` in
`work` (the wave's program counter kept for the error's address, in a register and on the stack).

## 2. The Machine's instructions as block members (`p2`, then the first builds of `p3`)

`tier_mach` in the base: in line in the dispatch loop (no symbol). With the block runner as a second caller (`p2`): a function of 0x521
bytes. Knob off against the base (9 runs; the control within 1.3 %): buf 0.977 (MI300X) and 0.983 (RX 6800), ldsx 0.981 and 0.990, mem
0.988 and 0.984, lds, narrow, valu_f64 and p_perm within 1 %. With `tier_mach` always in line (5 and 11 runs): buf 0.980 and 0.984, once
0.952; p_i64 on the MI300X 0.966; mfma 0.981. With the members blocks keep moved behind the Machine's others (9 runs): buf 0.969 and
0.976, flat on the RX 6800 0.963, p_cmp 0.973 and 0.980, p_i64 on the MI300X 0.978, p_dpp_arch 0.978; p_f64x and p_lane, which were at
0.96 the build before, at 1.04 and 0.99. (Each build re-rolls a few rows by 2 to 4 %; buf is low in all of them.)

## 3. A flag test and a second loop inside run_round

The same build, a second pass with the base grown by the size blocks add to `Wave` (16 bytes) and `Inst` (8) as a second control
(`basewi`), 9 runs:

| row | target | control | larger Wave and Inst | knob off |
|---|---|---:|---:|---:|
| buf | mi300x | 0.998 | 0.993 | 0.971 |
| buf | rx6800 | 1.001 | 0.990 | 0.985 |
| p_cmp | mi300x | 1.002 | 1.008 | 0.986 |
| p_i64 | mi300x | 0.991 | 0.980 | 0.947 |
| p_dpp_arch | mi300x | 0.982 | 0.989 | 0.945 |
| p_lane | mi300x | 1.012 | 1.001 | 0.962 |
| p_lane | rx6800 | 0.998 | 0.984 | 0.946 |
| valu_int | mi300x | 0.990 | 1.002 | 1.000 |
| mfma | mi300x | 0.997 | 0.999 | 0.988 |

Geometric mean of 14 rows: control 0.999, larger structures 0.998, knob off 0.979. The dispatch loop of that build, around the handler
call: the base's is one run of instructions from the wave's state tests to the turn's test; the build's has the state tests and the fetch
elsewhere and `jmp` after the counters.

`run_round` as the base's text, a function of its own (the blocks' twin beside it, `run_groups<false>` calling it): geometric mean of 16
rows 1.002 (control 1.002), but rows from 0.958 (valu_int on the RX 6800) to 1.034 (flat on the MI300X): the loop was no longer in line in
`run_groups`, as it is in the base. `run_round` always in line: 20 rows, geometric mean 0.992 (control 0.998), all within 1.6 % but
p_dpp_arch on the MI300X (0.946; 0.977 to 1.000 in the passes around it). The final table is `steps.md`.

## 4. The API rows (no kernel), `bench.py`, 15 runs a side, pinned

Base against this branch with the knob off: 20 rows, 0.971 (launch on the RX 6800) to 1.048 (apicheap on the MI300X), geometric mean 0.992;
startup's wall time 0.965 and 0.986 of the base's. Base against the control: 0.903 to 1.090 (both apicheap), geometric mean 0.997.
