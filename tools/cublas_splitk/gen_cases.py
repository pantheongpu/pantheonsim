#!/usr/bin/env python3
"""Pick the products nvidia/tests/e2e/cublas_tc_gemm.cu runs besides its fixed ones: from the survey of both cards (the lines of `split_probe survey`,
"in out m n k ta tb alpha beta L <L> slices <s> ok|odd"), per type a spread of products the slice model describes ("ok") on every card that runs the type
(f16 on both, bf16 and tf32 on the A10G) of which at least one splits, with a spread of transposes and slice counts, m n k <= 2^25. Prints the C++
initializers. A product counts for a card when its survey line is "ok" and it is either measured unsplit or a row of the card's table (rows the classification dropped
do not). Usage: gen_cases.py <a10g.inc> <t4.inc> <a10g survey> <t4 survey>"""
import random, sys
random.seed(734)
code = {('f16', 'f32'): ('F16', 'F32', 'C32'), ('f16', 'f16'): ('F16', 'F16', 'C32'), ('bf16', 'f32'): ('BF16', 'F32', 'C32'), ('tf32', 'f32'): ('F32', 'F32', 'CT')}
def load(files):
    d = {}
    for f in files:
        for line in open(f):
            p = line.split()
            if 'L' not in p or p[-1] not in ('ok', 'odd'):
                continue
            i, o, m, n, k, ta, tb = p[0], p[1], int(p[2]), int(p[3]), int(p[4]), int(p[5]), int(p[6])
            d[(i, o, m, n, k, ta, tb)] = (int(p[p.index('L') + 1]), p[-1])
    return d
def table(path):
    rows = set()
    for line in open(path):
        if line.startswith('{'):
            v = list(map(int, line.strip().strip('{},').split(',')))
            rows.add(tuple(v[:6]))
    return rows
tcode = {('f16', 'f32'): 0, ('f16', 'f16'): 1, ('bf16', 'f32'): 2, ('bf16', 'f16'): 3, ('tf32', 'f32'): 4}
def good(survey, tab, key):
    i, o, m, n, k, ta, tb = key
    if key not in survey or survey[key][1] != 'ok':
        return False
    return survey[key][0] >= k or (tcode[(i, o)], ta, tb, m, n, k) in tab
a10g_s, t4_s = load([sys.argv[3]]), load([sys.argv[4]])
a10g_t, t4_t = table(sys.argv[1]), table(sys.argv[2])
a10g = {k: v for k, v in a10g_s.items() if good(a10g_s, a10g_t, k)}
t4 = {k: v for k, v in t4_s.items() if good(t4_s, t4_t, k)}
cands = {}
for key, (L, st) in a10g.items():
    i, o, m, n, k, ta, tb = key
    if st != 'ok' or m < 64 or n < 64 or m * n * k > 1 << 24 or (i == 'tf32' and m * n * k < 1 << 21):
        continue
    if i in ('f16',):
        if key not in t4 or t4[key][1] != 'ok':
            continue
        split = L < k or t4[key][0] < k
    else:
        split = L < k
    if split:
        cands.setdefault((i, o), []).append((key, L, t4.get(key, (k, ''))[0]))
picks = []
for t, pool in sorted(cands.items()):
    pool.sort()
    chosen = []
    for tr in ((0, 0), (1, 0), (0, 1), (1, 1)):
        sub = [r for r in pool if (r[0][5], r[0][6]) == tr]
        if sub:
            chosen.append(random.choice(sub))
    # the 2-slice, 3-slice and many-slice products (the epilogue is not the same for 2 slices), and the extremes
    by = lambda r: -(-r[0][4] // r[1])
    for want in (2, 3):
        sub = [r for r in pool if by(r) == want]
        if sub:
            chosen += random.sample(sub, min(2, len(sub)))
    chosen += [max(pool, key=by)]
    picks += [(t, r) for r in dict.fromkeys(chosen)]
out = []
AB = [('', '1.f', '0.f'), (' alpha beta', '0.5f', '0.25f'), (' alpha beta 1.3 0.7', '1.3f', '0.7f'), (' alpha 1.3', '1.3f', '0.f'), (' beta 0.7', '1.f', '0.7f')]
def emit(name_t, key, kinds):
    i, o, m, n, k, ta, tb = key
    ins, outs, comp = code[(i, o)]
    for kd in kinds:
        suffix, al, be = AB[kd]
        name = '%s %s %s %dx%dx%d %s%s%s' % (name_t, i, o, m, n, k, 'T' if ta else 'N', 'T' if tb else 'N', suffix)
        out.append('      {"%s", %s, %s, %s, %d, %d, %d, %s, %s, %s, %s},' % (name, ins, outs, comp, m, n, k, 'true' if ta else 'false', 'true' if tb else 'false', al, be))
chosen_sorted = sorted(set((t, r) for t, r in picks), key=lambda x: (x[0], x[1][0]))
for idx, (t, (key, L, L4)) in enumerate(chosen_sorted):
    nslices = -(-key[4] // L)
    # the epilogue differs with the number of slices (two, or more): every kind on those, one kind on the others
    emit('splitk', key, [0, 1, 2, 3, 4] if nslices in (2, 3) else [idx % 3])
# products that are not split (the chain), with the epilogue kinds the fixed group does not have
for key in (('f16', 'f32', 128, 96, 256, 0, 0), ('f16', 'f16', 128, 96, 256, 0, 0), ('bf16', 'f32', 128, 96, 256, 0, 0), ('tf32', 'f32', 128, 128, 128, 0, 0)):
    emit('chain', key, [2, 3, 4])
print('\n'.join(out))
