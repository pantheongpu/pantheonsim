#!/usr/bin/env python3
# sp68: mutants of include/vgpu/host_gpu_idiom.hpp against tests/unit/test_host_gpu_idiom.cpp. Each mutant is a copy of the header with one textual change,
# compiled outside the tree (its own include directory first on the path) with the unit test and the tree's libvgpu.a, then run. KILLED: the test fails
# (or the mutant does not compile, said separately); SURVIVED: it passes.
#   usage (server3): python3 mutants.py <source tree> <build dir> <scratch dir>
import os, subprocess, sys, shutil
src, build, work = sys.argv[1], sys.argv[2], sys.argv[3]
hdr = open(os.path.join(src, 'include/vgpu/host_gpu_idiom.hpp')).read()
M = [
 ('no fork: an undecided guard is taken as true', 'if (++made > max_paths) return fail("too many paths");\n            Path other = s;\n            other.cond.emplace_back(g, false);\n            work.push_back(std::move(other));', 'if (++made > max_paths) return fail("too many paths");'),
 ('liveness never reads', 'inline bool may_be_read(const Code& c, size_t from, bool pred, unsigned n) {', 'inline bool may_be_read(const Code& c, size_t from, bool pred, unsigned n) {\n  if (from < c.ins.size() + 1) return false;'),
 ('a predicated write covers reads of either polarity', 'const unsigned gbit = guarded ? 2 * ins.guard + (ins.guard_not ? 1 : 0) : 0;', 'const unsigned gbit = guarded ? 2 * ins.guard : 0;'),
 ('a predicated branch kills on both edges', 'const bool taken_written = guarded && (s.covered >> gbit & 1), fall_written = guarded && (s.covered >> (gbit ^ 1) & 1);', 'const bool taken_written = guarded && ((s.covered >> gbit & 1) || (s.covered >> (gbit ^ 1) & 1)), fall_written = taken_written;'),
 ('a rewritten predicate keeps its cover', 'if (wr.p.test(q)) s.covered &= static_cast<uint16_t>(~(3u << (2 * q)));', 'if (wr.p.test(q) && false) s.covered &= static_cast<uint16_t>(~(3u << (2 * q)));'),
 ('no check of branches into the idiom', 'if (t > static_cast<int64_t>(lo) && t < static_cast<int64_t>(hi)) return fail("a branch from outside lands inside the idiom");', ''),
 ('no check of code falling into the slow path', 'if (falls) return fail("code before the slow path falls into it");', ''),
 ('the RET return address not checked', 'if (reg_value(s, ins.src[0].reg) != "#" + std::to_string(back) || reg_value(s, ins.src[0].reg + 1) != "#0")', 'if (false)'),
 ('any MUFU form inside the idiom', 'if (ins.mods.size() != 1 || ins.mods[0] != "RSQ") return fail("an SFU instruction other than MUFU.RSQ in the idiom");', ''),
 ('memory and barriers allowed inside the idiom', 'return fail("memory, a barrier, another thread or control flow inside the idiom");', ';'),
 ('a MUFU value may flow through any instruction', 'return fail("a MUFU value used by an instruction that is not the refinement");', ';'),
 ('the result on one path is enough', 'for (const auto& [k, v] : ref) all = all && renamed(reg_value(*by_key[k], n), x) == v;', 'all = false;\n    for (const auto& [k, v] : ref) all = all || renamed(reg_value(*by_key[k], n), x) == v;'),
 ('the path keys not compared', 'if (!by_key.count(k)) return fail("not the paths of the reference sqrt.rn");', ';'),
 ('predicates from a MUFU not checked', 'st.live = 256 + static_cast<int>(n);\n        return fail("a predicate computed from a MUFU value is read after the idiom");', 'st.live = 256 + static_cast<int>(n);'),
 ('MUFU values left in registers not checked', 'st.live = static_cast<int>(n);\n        return fail("a register holding an unrefined MUFU value is read after the idiom");', 'st.live = static_cast<int>(n);'),
 ('IMAD.WIDE addend read as one register', 'return k == 2 ? 2 : 1;   // a 32 x 32 product', 'return 1;   // a 32 x 32 product'),
 ('a 64-bit address reads one register', 'if (o.reg + 1 < 255 && o.text.find(".64") != std::string::npos) g.r.set(o.reg + 1);', ''),
 ('a 128-bit store reads one register', 'if (any_mod_part(ins.mods, [](const std::string& p) { return p.find("128") != std::string::npos; })) return 4;', ''),
 ('a predicated MUFU starts a site', 'if (!unconditional(mu) || mu.src.size() != 1', 'if (mu.src.size() != 1'),
 ('the subroutine entry may be reached by any branch', '!(mn == "CALL" && t == entry))', '!(t == entry))'),
 ('a call of the division idiom into a sqrt slow path allowed', 'v.why = "a call into a square root\'s slow path from outside its site";\n            return v;', 'v.why = "";'),
 ('a backward branch allowed in the idiom', 'if (t < 0 || c.ins[static_cast<size_t>(t)].pc <= ins.pc) return fail("a branch that is not forward");', 'if (t < 0) return fail("a branch that is not forward");'),
 ('the argument-independent guard check drops the X test', 'else if (!control && !mentions_x(g) && std::none_of(', 'else if (!control && std::none_of('),
 ('the switch ignored: sqrt always on', 'inline KernelVerdict idioms_only(const std::vector<Instr>& code, bool div, bool sqrt) {\n  KernelVerdict v;', 'inline KernelVerdict idioms_only(const std::vector<Instr>& code, bool div, bool sqrt) {\n  sqrt = true;\n  KernelVerdict v;'),
]
LOCK = ['flock', '/tmp/pantheonsim-builder.lock', 'nice', '-n', '19']   # the machine's build lock, one hold per compile and per run
os.makedirs(work, exist_ok=True)
lib = os.path.join(build, 'libvgpu.a')
res = []
for name, old, new in M:
    if hdr.count(old) != 1:
        res.append((name, 'NOT-APPLIED (%d matches)' % hdr.count(old))); continue
    d = os.path.join(work, 'inc', 'vgpu'); shutil.rmtree(os.path.join(work, 'inc'), ignore_errors=True); os.makedirs(d)
    open(os.path.join(d, 'host_gpu_idiom.hpp'), 'w').write(hdr.replace(old, new))
    exe = os.path.join(work, 't')
    c = subprocess.run(LOCK + ['g++', '-std=c++20', '-O1', '-I', os.path.join(work, 'inc'), '-I', os.path.join(src, 'include'), '-I', os.path.join(src, 'tests/framework'),
                        '-I', os.path.join(src, 'tests/unit'), os.path.join(src, 'tests/unit/test_host_gpu_idiom.cpp'), lib, '-lpthread', '-ldl', '-o', exe], capture_output=True, text=True)
    if c.returncode:
        res.append((name, 'KILLED (does not compile)')); continue
    r = subprocess.run(LOCK + [exe], capture_output=True, text=True, timeout=600)
    fails = [l[9:] for l in r.stdout.splitlines() + r.stderr.splitlines() if l.startswith('[ FAIL ] ')]
    res.append((name, ('KILLED by ' + ', '.join(fails)) if r.returncode else 'SURVIVED'))
    print(name, '->', res[-1][1], flush=True)
k = sum(1 for _, v in res if v.startswith('KILLED'))
print('\n| mutant | result |\n|---|---|')
for n, v in res: print('| %s | %s |' % (n, v))
print('\n%d of %d killed' % (k, len(res)))
