#!/usr/bin/env python3
"""Writes the per-profile table of docs/numerics/per-profile.md from `vgpu info --gpu <id> --json` of every profile: for each device, which
matrix-instruction arithmetic is measured on a card, taken from a paper or recorded vectors, inferred, or unknown.

  profile_table.py <dir of <vendor>.<name>.json files>      (a loop of `vgpu info --gpu <id> --json > <dir>/<vendor>.<name>.json` over `vgpu list-gpus`)

Our own script. Run it with python3 -I.
"""
import json, os, sys

CLASS = {"measured": "measured", "documented": "documented", "inferred": "inferred", "unknown": "unknown"}

def cls(d, key):
    return d["provenance"].get(key, "unstated")

def f16_cell(d):
    n = d["numerics"]
    sm = n["tensor_sum"]
    if sm == "sequential":
        return "sequential f32 sum, **unknown** (no card, no published model, no recorded data)"
    chunks = f"chunks of {n['tensor_sum_chunk']} (tf32 {n['tensor_sum_chunk_tf32']}), grid {n['tensor_sum_grid']}"
    c = cls(d, "numerics.tensor_sum")
    tail = {"measured": "a card of that model was run",
            "inferred": "from a card of the same compute capability, or from vectors others recorded"}.get(c, c)
    return f"`{sm}`: {chunks}; **{c}** ({tail})"

def ccv(d):
    return tuple(int(x) for x in d["compute_capability"].split("."))

def fp8_cell(d):
    n = d["numerics"]
    cc = d["compute_capability"]
    sm = n.get("tensor_sum_fp8")
    if ccv(d) < (8, 9):
        return "no FP8 form on this architecture"
    if sm == "ada":
        return f"`ada`: chunks of {n['tensor_sum_fp8_chunk']}, grid {n['tensor_sum_fp8_grid']}, result cut to {n['tensor_sum_fp8_out_bits']} bits; **{cls(d, 'numerics.tensor_sum_fp8')}** (vectors recorded by others, reproduced bit for bit)"
    if sm == "via_f16":
        return f"`via_f16`: ptxas's f16 `HMMA` chain, `FADD` of C after; **{cls(d, 'numerics.tensor_sum_fp8')}** (vectors recorded by others, reproduced bit for bit)"
    if cc.startswith("12.0"):
        who = "measured" if d["id"].endswith("rtx-pro-6000-server") else "inferred"
        return (f"exact sum of the products and C cut toward zero (`mma_narrow_sum`); **{who}**"
                + (" (18 of 27 forms bit for bit on the card, 9 wide ones differ)" if who == "measured" else " (from the server edition)"))
    return "sequential f32 sum, **unknown**"

def other_cell(d):
    v = ccv(d)
    if v == (7, 5):
        return "`m8n8k4`: sequential f32 sum, **unknown**"
    if v in ((9, 0), (10, 0)):
        if v == (9, 0):
            n = d["numerics"]
            if n.get("tensor_sum_wgmma_fp8") == "hopper":
                return (f"`wgmma` e4m3, e5m2 (dense): `hopper`, one chunk of {n['tensor_sum_wgmma_fp8_chunk']}, grid {n['tensor_sum_wgmma_fp8_grid']}, result cut to "
                        f"{n['tensor_sum_wgmma_fp8_out_bits']} bits; **{cls(d, 'numerics.tensor_sum_wgmma_fp8')}** (vectors recorded by others on an H100 and an H200, zero accumulator). "
                        "`wgmma` f16, bf16, tf32, FP8 sparse: sequential f32 sum, **unknown**")
            return "`wgmma`, sparse: sequential f32 sum, **unknown**"
        return "`wgmma`, `tcgen05.mma`, sparse: sequential f32 sum, **unknown**"
    if v[0] in (10, 11) or v == (12, 1):
        return "all of them: sequential f32 sum, **unknown**"
    return "sparse (`mma.sp`): sequential f32 sum, **unknown**"

def main():
    rows = []
    for f in sorted(os.listdir(sys.argv[1])):
        if not f.endswith(".json") or not f.startswith("nvidia."):
            continue
        d = json.load(open(os.path.join(sys.argv[1], f)))
        if not d.get("numerics"):
            continue
        rows.append(d)
    rows.sort(key=lambda d: (tuple(int(x) for x in d["compute_capability"].split(".")), d["id"]))
    print("| Profile | CC | f16, bf16, tf32 (`wmma`, `mma.sync`, `HMMA`) | e4m3, e5m2 `mma.sync` | Other matrix forms |")
    print("|---|---|---|---|---|")
    for d in rows:
        print(f"| {d['id'].split('/')[1]} | {d['compute_capability']} | {f16_cell(d)} | {fp8_cell(d)} | {other_cell(d)} |")

main()
