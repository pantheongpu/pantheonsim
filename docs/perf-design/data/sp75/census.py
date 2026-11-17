#!/usr/bin/env python3
"""Round sp75: the census of what keeps a launch in the simulator, per launch, against the stand-in driver.

Runs programs of the corpora (tools/bench/real/real.py's manifest) with VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1 against the stand-in driver
(nvidia/tests/e2e/host_gpu_stub_driver.c, which refuses every module: a launch the mode sends "fails on the card" and the simulator runs it),
once per form, the forms one after another for each program, and keeps what the mode said: the summary counts, the host_gpu_kept reasons, the
samples (taken, sent, lost, their cost), the host_gpu_nosample lines and the decision lines of sampled launches. A form is NAME=BUILD[,ENV=VALUE...].
With --cut N a run is ended once N decisions have been logged (a long launch's decision is made before it runs).

  census.py --corp CORP --stub libstubcuda.so --out DIR --form old=/path/build-old --form new=/path/build --form knob=/path/build,VGPU_HOST_GPU_SAMPLE_BLOCK=0
            [--programs a,b] [--skip a,b] [--timeout 150] [--runs 1] [--cut 0] [--threads 3] [--gpu nvidia/a10g --sm 86 --mp 80]

Round r9 (pantheonsim #656, the loop-weighted static count): a mode that changes no decision. With --weighted a program is run once, in the
first form, with VGPU_HOST_GPU_LOG=2 (the mode's own text of every kernel's code), and kept per kernel is the plain static count beside the
count that weighs a loop's body by its turns where the bound is an immediate (a nest of loops multiplies), with the launches the floor kept in
the simulator ("static estimate under the floor and the process too young"). --report DIR prints the table of a census made so:

  census.py --weighted --corp CORP --stub libstubcuda.so --out DIR --form new=/path/build [--programs a,b] [--timeout 150]
  census.py --report DIR
"""
import argparse, fcntl, json, os, re, shutil, signal, subprocess, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "tools", "bench", "real"))
import real  # noqa: E402

DECISION = re.compile(r"host_gpu_nosample:|host_gpu: \S+ sampled|stays in the simulator \(sim|starts the driver")


def run(argv, cwd, env, timeout, cut, lock, keep_code=False):
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        t0 = time.monotonic()
        p = subprocess.Popen(["nice", "-n", "19"] + argv, cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
        state = {"timeout": False}

        def killer():
            if p.wait_done.wait(timeout):
                return
            state["timeout"] = True
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        p.wait_done = threading.Event()
        threading.Thread(target=killer, daemon=True).start()
        lines, n, was_cut = [], 0, False
        for raw in p.stderr:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if "host_gpu" not in line or (line.startswith("host_gpu:   ") and not keep_code):
                continue
            lines.append(line)
            if DECISION.search(line):
                n += 1
                if cut and n >= cut and not was_cut:
                    was_cut = True
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        rc = p.wait()
        p.wait_done.set()
        return {"rc": rc, "wall_s": round(time.monotonic() - t0, 4), "timeout": state["timeout"], "cut": was_cut, "lines": lines}
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def digest(r):
    """What a run decided, from its lines."""
    d = {"rc": r["rc"], "wall_s": r["wall_s"], "timeout": r["timeout"], "cut": r["cut"], "kept": [], "nosample": [], "sampled_lines": [], "starts": []}
    for line in r["lines"]:
        m = re.match(r"host_gpu: (\d+) launches ran on the card, (\d+) in the simulator \(.*?(\d+) too small, (\d+) unsupported launch shapes, (\d+) failed on the card\)", line)
        if m:
            d["launches_sim"], d["too_small"], d["failed"] = int(m.group(2)), int(m.group(3)), int(m.group(5))
        m = re.match(r"host_gpu_sampled: sampled=(\d+) sent=(\d+) timeouts=(\d+) cost_s=(\S+)", line)
        if m:
            d["sampled"], d["sample_sent"], d["sample_timeouts"], d["sample_cost_s"] = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
        m = re.match(r"host_gpu_kept: kernel=(\S+) launches=(\d+) sim_s=(\S+) reason=(.*)$", line)
        if m:
            d["kept"].append({"kernel": m.group(1), "launches": int(m.group(2)), "sim_s": float(m.group(3)), "reason": m.group(4)})
        if line.startswith("host_gpu_nosample: "):
            d["nosample"].append(dict(kv.split("=", 1) for kv in line[len("host_gpu_nosample: "):].split(" reason=")[0].split()) | {"reason": line.split(" reason=")[1]})
        m = re.match(r"host_gpu: (\S+) sampled(.*)$", line)
        if m:
            d["sampled_lines"].append(m.group(1) + m.group(2))
        m = re.match(r"host_gpu: (\S+) starts the driver: (.*)$", line)
        if m:
            d["starts"].append(m.group(1) + ": " + m.group(2))
    return d


# ---- Round r9: the loop-weighted static count (pantheonsim #656), as a census; no decision of the mode is touched -------------------------------

CODE_LINE = re.compile(r"^host_gpu:   (\S+): ([0-9a-f]+) (.*)$")
NOSAMPLE_FLOOR = "static estimate under the floor and the process too young"
WARP_INSTR_PER_S = 100e6   # include/vgpu/host_gpu.hpp Costs::sim_warp_instr_per_s, the rate every static estimate here is made with
ADD_MNEMONICS = ("IADD3", "VIADD", "IADD", "UIADD3")


def reg_number(o):
    """R12 -> 12, UR4 -> 260, RZ / URZ / anything else -> -1 (the numbering of vgpu/host_gpu.hpp LoopCodeInsn)."""
    m = re.fullmatch(r"(U?)R(\d+)", o)
    if not m:
        return -1
    n = int(m.group(2))
    if m.group(1):
        return 256 + n if n < 63 else -1
    return n if n < 255 else -1


def pred_number(o):
    m = re.fullmatch(r"(U?)P(\d)", o)
    return -1 if not m else (8 if m.group(1) else 0) + int(m.group(2))


def imm_value(o):
    return int(o, 16) if re.fullmatch(r"-?0x[0-9a-f]+", o) else None


def parse_insn(text):
    """The text sass::to_text prints -> dict (guard, mnemonic, mods, operands)."""
    guard = None
    m = re.match(r"^@(!?)(U?P)(\d|T) ", text)
    if m:
        guard = (m.group(1) == "!", m.group(2) == "UP", None if m.group(3) == "T" else int(m.group(3)))
        text = text[m.end():]
    head, _, ops = text.partition(" ")
    mn, *mods = head.split(".")
    # (nvdisasm's register-reuse hints, R4.reuse, say nothing of the value)
    return {"guard": guard, "mn": mn, "mods": mods, "ops": [re.sub(r"\.reuse$", "", x.strip()) for x in ops.split(", ")] if ops else []}


def bound_turns(bound, step):
    """include/vgpu/host_gpu.hpp bound_turns: the turns a loop has at most when it counts from 0 to `bound` by `step`."""
    return 0 if step == 0 else abs(bound) // abs(step)


def analyse(code_lines):
    """code_lines: [(offset, text)] of one kernel -> the plain count, the weighted count and what it is made of."""
    n = len(code_lines)
    ins = [parse_insn(t) for _, t in code_lines]
    at = {off: i for i, (off, _) in enumerate(code_lines)}
    base = 0
    out = {"plain": n, "weighted": n, "loops": [], "calls": 0, "indirect": False, "unresolved": 0}
    target = [-1] * n
    for i, x in enumerate(ins):
        if x["mn"] == "BRA" and x["ops"]:
            t = imm_value(x["ops"][-1])
            if t is not None and (t - base) in at:
                target[i] = at[t - base]
            else:
                out["unresolved"] += 1
        elif x["mn"] == "CALL":
            out["calls"] += 1
        elif x["mn"] in ("BRX", "JMP"):   # (a RET ends a called routine of the kernel's own code: no flow of the kernel's loops goes through it)
            out["indirect"] = True
    if out["indirect"] or out["unresolved"]:
        return out
    # the loops: one range per head, [head, last back edge]; a range crossing another is left out (as find_probe_loops does)
    rng = {}
    for i in range(n):
        if 0 <= target[i] <= i:
            rng[target[i]] = max(rng.get(target[i], 0), i)
    loops = []
    for lo, hi in sorted(rng.items()):
        crossed = any((lo2 < lo <= hi2 < hi) or (lo < lo2 <= hi < hi2) for lo2, hi2 in rng.items())
        if not crossed:
            loops.append((lo, hi))

    # registers and predicates each instruction writes: the destination operands (the leading ones that are registers or predicates, before the sources)
    def dsts(x):
        if x["mn"] in ("BRA", "EXIT", "STG", "STS", "ST", "STL", "RED", "BAR", "NOP", "CALL", "RET", "MEMBAR", "ATOMS", "ATOMG", "ATOM", "SYNCS", "WARPSYNC"):
            return []
        mn = x["mn"]
        k = 2 if mn in ("ISETP", "UISETP", "FSETP", "DSETP", "HSETP2", "PSETP", "R2P", "VOTE", "SHFL", "ULOP3", "LOP3") and x["ops"] and pred_number(x["ops"][0]) >= 0 and len(x["ops"]) > 1 and pred_number(x["ops"][1]) >= 0 else 1
        res = x["ops"][:k]
        return res

    def reg_writes(i):
        r = set()
        for o in dsts(ins[i]):
            v = reg_number(o)
            if v >= 0:
                r.add(v)
                m = ins[i]["mods"]
                if "128" in m:
                    r.update(v + k for k in range(1, 4))
                elif "64" in m or "WIDE" in m:   # (a wide result also writes the register above)
                    r.add(v + 1)
        return r

    def pred_writes(i):
        return {pred_number(o) for o in dsts(ins[i]) if pred_number(o) >= 0}

    def add_imm(i):
        """-> (dst, src, imm) when instruction i is dst = src + constant (IADD3 / VIADD / IADD / UIADD3, one register source), else None."""
        x = ins[i]
        if x["mn"] not in ADD_MNEMONICS or x["guard"] is not None or x["mods"] or not x["ops"]:
            return None
        d = reg_number(x["ops"][0])
        if d < 0:
            return None
        src, regs, imm = -1, 0, 0
        for o in x["ops"][1:]:
            if pred_number(o) >= 0 or o in ("PT", "UPT"):
                continue
            if o in ("RZ", "URZ"):
                continue
            v = imm_value(o)
            if v is not None:
                imm += v
                continue
            r = reg_number(o)
            if r >= 0:
                src = r
                regs += 1
                continue
            return None
        if regs != 1 or imm == 0:
            return None
        imm = imm & 0xffffffff
        return d, src, imm - (1 << 32) if imm >= 1 << 31 else imm

    def compare(i):
        """-> (predicates written, value a, value b, unsigned) for an ISETP / UISETP that is a counter alone, else None."""
        x = ins[i]
        if x["mn"] not in ("ISETP", "UISETP") or x["guard"] is not None or "EX" in x["mods"]:
            return None
        ops = x["ops"]
        preds = [pred_number(o) for o in ops[:2] if pred_number(o) >= 0]
        vals = [o for o in ops[2:] if pred_number(o) < 0 and o not in ("PT", "UPT")]
        for o in ops[2:]:   # the predicate it combines with must be PT
            if pred_number(o) >= 0 or o.startswith("!"):
                return None
        if len(vals) != 2:
            return None
        return preds, vals[0], vals[1], "U32" in x["mods"]

    for lo, hi in loops:
        info = {"lo": 16 * lo, "hi": 16 * hi, "turns": None, "bound": None}
        wr = {}
        for j in range(lo, hi + 1):
            for r in reg_writes(j):
                wr.setdefault(r, []).append(j)
        branches = [i for i in range(hi, lo - 1, -1) if ins[i]["mn"] == "BRA" and ins[i]["guard"] is not None and lo <= target[i] <= i]
        branches += [i for i in range(hi, lo - 1, -1) if ins[i]["mn"] == "BRA" and ins[i]["guard"] is not None and target[i] >= 0 and not lo <= target[i] <= hi]
        branches += [i for i in range(hi, lo - 1, -1) if ins[i]["mn"] == "CALL" and ins[i]["guard"] is not None]   # (left through a call: step 1 of the design)
        for b in branches:
            if info["bound"] is not None:   # (the first branch that gives a counter decides, whatever its bound is, as find_probe_loops does)
                break
            g = ins[b]["guard"]
            p = (8 if g[1] else 0) + g[2] if g[2] is not None else -1
            if p < 0:
                continue
            j = b - 1
            while j >= lo and p not in pred_writes(j):
                j -= 1
            if j < lo:
                continue
            c = compare(j)
            if not c or p not in c[0]:
                continue
            for xo, yo in ((c[1], c[2]), (c[2], c[1])):
                xr = reg_number(xo)
                if xr < 0 or len(wr.get(xr, [])) != 1:
                    continue
                a = add_imm(wr[xr][0])
                if not a or a[0] != xr:
                    continue
                step = a[2]
                if a[1] != xr:   # x = iv + offset, iv = iv + step
                    iv = a[1]
                    if iv < 0 or len(wr.get(iv, [])) != 1:
                        continue
                    a2 = add_imm(wr[iv][0])
                    if not a2 or a2[0] != iv or a2[1] != iv:
                        continue
                    step = a2[2]
                if step == 0:
                    continue
                v = imm_value(yo)
                if v is not None:
                    v = ((v + (1 << 31)) & 0xffffffff) - (1 << 31)   # (the signed 32-bit word, as consider_sample reads an immediate bound)
                    info["turns"] = bound_turns(v, step)
                    info["bound"] = "imm"
                else:
                    yr = reg_number(yo)
                    if yo.startswith("c["):
                        info["bound"] = "param"
                    elif yo in ("RZ", "URZ") or (yr >= 0 and yr not in wr):
                        info["bound"] = "reg"
                    else:
                        continue   # (a register the loop writes, or an operand that is no value: not a bound)
                break
        out["loops"].append(info)
    w = [1.0] * n
    for info in out["loops"]:
        if info["turns"]:
            for j in range(info["lo"] // 16, info["hi"] // 16 + 1):
                w[j] *= info["turns"]
    out["weighted"] = int(sum(w))
    return out


def parse_weighted_run(lines):
    """The lines of a LOG=2 run -> {kernel: analyse(...)} and the floor-refused launches (host_gpu_nosample lines whose reason is the floor)."""
    code = {}
    for line in lines:
        m = CODE_LINE.match(line)
        if m:
            code.setdefault(m.group(1), []).append((int(m.group(2), 16), m.group(3)))
    kernels = {k: analyse(v) for k, v in code.items()}
    followed = {}
    for line in lines:
        m = re.match(r"^host_gpu: (\S+): followed loop \[0x([0-9a-f]+), 0x([0-9a-f]+)\]: iv \S+ step (-?\d+) bound kind (\d+) reg (\d+) imm 0x([0-9a-f]+)", line)
        if m:
            followed.setdefault(m.group(1), []).append((int(m.group(2), 16), int(m.group(3), 16), int(m.group(4)), int(m.group(5)), int(m.group(7), 16)))
    # the simulator's own finder (top-level loops, no call in the kernel) against this census's: the same loop, and for an immediate bound the same turns
    check = {"followed": 0, "match": 0, "mismatch": []}
    for k, fl in followed.items():
        mine = {(l["lo"], l["hi"]): l for l in kernels[k]["loops"]} if k in kernels else {}
        for lo, hi, step, kind, imm in fl:
            check["followed"] += 1
            l = mine.get((lo, hi))
            want = None
            if kind == 3:   # Imm
                v = ((imm + (1 << 31)) & 0xffffffff) - (1 << 31)
                want = bound_turns(v, step)
            if l is not None and (kind != 3 or l["turns"] == want) and (kind == 3 or l["bound"] in ("param", "reg")):
                check["match"] += 1
            else:
                check["mismatch"].append("%s [0x%x,0x%x] step %d kind %d: census %s" % (k, lo, hi, step, kind, None if l is None else (l["turns"], l["bound"])))
    floor = []
    for line in lines:
        if line.startswith("host_gpu_nosample: ") and line.endswith(" reason=" + NOSAMPLE_FLOOR):
            d = dict(kv.split("=", 1) for kv in line[len("host_gpu_nosample: "):].split(" reason=")[0].split())
            floor.append({k: (d[k] if k == "kernel" else float(d[k])) for k in d})
    samples = []   # the samples the run took (host_gpu_sample lines): what each cost, and what it said
    for line in lines:
        if line.startswith("host_gpu_sample: "):
            d = dict(kv.split("=", 1) for kv in line[len("host_gpu_sample: "):].split())
            samples.append({k: (d[k] if k == "kernel" else float(d[k])) for k in ("kernel", "blocks", "cost_s", "fork_s", "estimate_s", "would_send", "launch_s", "static_s") if k in d})
    return kernels, floor, check, samples


def weighted_records(a, forms):
    name, build, extra = forms[0]
    want = set(x for x in a.programs.split(",") if x)
    skip = set(x for x in a.skip.split(",") if x)
    for corpus, pname, cwd, argv, copy_cwd in real.manifest(a.corp, a.corpora.split(",")):
        key = corpus + "/" + pname
        if (want and key not in want and pname not in want) or key in skip or pname in skip:
            continue
        dest = os.path.join(a.out, corpus + "__" + pname.replace("/", "_") + ".json")
        if os.path.exists(dest):
            continue
        env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_") and not k.startswith("STUB_")}
        env.update(VGPU_GPU=a.gpu, LD_LIBRARY_PATH=os.path.join(build, "shim"), VGPU_THREADS=a.threads, VGPU_HOST_GPU="turbo", VGPU_HOST_GPU_LOG="2",
                   VGPU_HOST_GPU_LIBCUDA=a.stub, STUB_SM=a.sm, STUB_MP=a.mp)
        env.update(extra)
        scratch, rcwd = None, cwd
        if copy_cwd:
            scratch = tempfile.mkdtemp(prefix="census-")
            dd = os.path.join(os.path.dirname(argv[0]), "data")
            with open(os.path.join(scratch, "filelist"), "w") as f:
                f.write("".join(os.path.join(dd, "cane4_%d.db\n" % j) for j in range(4)))
            rcwd = scratch
        r = run(argv, rcwd, env, a.timeout, a.cut, a.lock, keep_code=True)
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)
        kernels, floor, check, samples = parse_weighted_run(r["lines"])
        json.dump({"program": key, "rc": r["rc"], "timeout": r["timeout"], "cut": r["cut"], "kernels": kernels, "floor": floor, "check": check, "samples": samples}, open(dest, "w"), sort_keys=True, separators=(",", ":"))
        print("%-45s rc=%s%s kernels=%d floor_refused=%d" % (key, r["rc"], " T" if r["timeout"] else "", len(kernels), len(floor)), flush=True)


def weighted_report(d, cost_dir=""):
    """The table of a weighted census (deterministic: nothing time-dependent is printed)."""
    recs = [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.endswith(".json")]
    kernels = {}   # name -> {program: analysis}
    for r in recs:
        for k, v in r["kernels"].items():
            kernels.setdefault(k, {})[r["program"]] = v
    out = []
    out.append("programs: %d (%d cut at the time limit: %s)" % (len(recs), sum(1 for r in recs if r["timeout"]), ", ".join(r["program"] for r in recs if r["timeout"]) or "none"))
    nk = sum(len(r["kernels"]) for r in recs)
    rise = sum(1 for k in kernels.values() for v in k.values() if v["weighted"] > v["plain"])
    out.append("kernels with code logged (a kernel the mode admitted and that has a loop): %d (%d distinct names); weighted count above the plain count: %d" % (nk, len(kernels), rise))
    chk = [sum(r["check"][k] if k != "mismatch" else len(r["check"][k]) for r in recs) for k in ("followed", "match", "mismatch")]
    out.append("cross-check with the simulator's own finder (top-level loops of kernels with no call): %d followed loops, the census finds the same loop and turns for %d, differs for %d%s" % (
        chk[0], chk[1], chk[2], "".join("\n  " + x for r in recs for x in r["check"]["mismatch"])))
    unres = sorted({p + ":" + k for r in recs for k, v in r["kernels"].items() for p in [r["program"]] if v["indirect"] or v["unresolved"]})
    out.append("kernels whose flow the census could not read (a register jump, BRX or JMP, or an unresolved branch; weighted = plain): %d" % len(unres))
    rows = []
    for r in recs:
        for f in r["floor"]:
            v = r["kernels"].get(f["kernel"])
            if not v:
                continue
            wps = f["blocks"] * ((f["threads_per_block"] + 31) // 32) * v["weighted"] / WARP_INSTR_PER_S
            rows.append((r["program"], f["kernel"], f, v, wps))
    out.append("")
    out.append("Launches the floor kept in the simulator (the first two of each kernel are said): %d, of %d kernels" % (len(rows), len({(p, k) for p, k, *_ in rows})))
    out.append("")
    cost = {}   # (program, kernel) -> the sample a run with VGPU_HOST_GPU_SAMPLE=always took of it
    if cost_dir:
        for f in sorted(os.listdir(cost_dir)):
            if f.endswith(".json"):
                r = json.load(open(os.path.join(cost_dir, f)))
                for sm in r.get("samples", []):
                    cost.setdefault((r["program"], sm["kernel"]), sm)
    out.append("| program | kernel | grid x block | plain count (instrs) | weighted count | today's estimate ms | weighted estimate ms | floor ms | rises over the floor | loops (turns; bound) |" + (" sample cost ms (fork ms) | sample says ms, would send |" if cost_dir else ""))
    out.append("|---|---|---|---:|---:|---:|---:|---:|---|---|" + ("---:|---|" if cost_dir else ""))
    seen = set()
    for p, k, f, v, wps in sorted(rows, key=lambda t: (t[0], t[1], t[2]["blocks"])):
        key = (p, k, f["blocks"], f["threads_per_block"])
        if key in seen:
            continue
        seen.add(key)
        loops = "; ".join("%s;%s" % (l["turns"] if l["turns"] is not None else "?", l["bound"] or "none") for l in v["loops"]) or "-"
        if v["calls"]:
            loops += " (%d call%s)" % (v["calls"], "" if v["calls"] == 1 else "s")
        out.append("| %s | `%s` | %d x %d | %d | %d | %.1f | %.1f | %.1f | %s | %s |" % (
            p, k, f["blocks"], f["threads_per_block"], v["plain"], v["weighted"], f["static_s"] * 1e3, wps * 1e3, f["floor_s"] * 1e3,
            "**yes**" if wps >= f["floor_s"] else "no", loops) + (
            " %s | %s |" % (("%.1f (%.1f)" % (cost[(p, k)]["cost_s"] * 1e3, cost[(p, k)]["fork_s"] * 1e3), "%.0f, %s" % (cost[(p, k)]["estimate_s"] * 1e3, "yes" if cost[(p, k)]["would_send"] else "no"))
                             if (p, k) in cost else ("-", "-")) if cost_dir else ""))
    return "\n".join(out)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corp")
    ap.add_argument("--stub")
    ap.add_argument("--out")
    ap.add_argument("--form", action="append")
    ap.add_argument("--corpora", default="polybench,rodinia,hecbench,samples")
    ap.add_argument("--programs", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--timeout", type=float, default=150)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--cut", type=int, default=0)
    ap.add_argument("--gpu", default="nvidia/a10g")
    ap.add_argument("--sm", default="86")
    ap.add_argument("--mp", default="80")
    ap.add_argument("--threads", default="3")
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    ap.add_argument("--weighted", action="store_true", help="Round r9: per kernel, the plain static count beside the loop-weighted one (changes no decision)")
    ap.add_argument("--cost-from", default="", help="with --report: a weighted census made with a form that sets VGPU_HOST_GPU_SAMPLE=always, for what each kernel's sample cost")
    ap.add_argument("--report", default="", help="print the table of a weighted census kept in this directory, and exit")
    a = ap.parse_args()
    if a.report:
        print(weighted_report(a.report, a.cost_from))
        return
    for need in ("corp", "stub", "out", "form"):
        if not getattr(a, need):
            ap.error("--%s is required" % need)
    os.makedirs(a.out, exist_ok=True)
    forms = []
    for f in a.form:
        name, _, rest = f.partition("=")
        parts = rest.split(",")
        forms.append((name, parts[0], dict(kv.split("=", 1) for kv in parts[1:])))
    if a.weighted:
        weighted_records(a, forms)
        return
    want = set(x for x in a.programs.split(",") if x)
    skip = set(x for x in a.skip.split(",") if x)
    for corpus, name, cwd, argv, copy_cwd in real.manifest(a.corp, a.corpora.split(",")):
        key = corpus + "/" + name
        if (want and key not in want and name not in want) or key in skip or name in skip:
            continue
        dest = os.path.join(a.out, corpus + "__" + name.replace("/", "_") + ".json")
        if os.path.exists(dest):
            continue
        rec = {"program": key, "forms": {n: [] for n, _, _ in forms}}
        for i in range(a.runs):
            for fname, build, extra in (forms if i % 2 == 0 else forms[::-1]):
                env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_") and not k.startswith("STUB_")}
                env.update(VGPU_GPU=a.gpu, LD_LIBRARY_PATH=os.path.join(build, "shim"), VGPU_THREADS=a.threads, VGPU_HOST_GPU="turbo", VGPU_HOST_GPU_LOG="1",
                           VGPU_HOST_GPU_LIBCUDA=a.stub, STUB_SM=a.sm, STUB_MP=a.mp)
                env.update(extra)
                scratch, rcwd = None, cwd
                if copy_cwd:
                    scratch = tempfile.mkdtemp(prefix="census-")
                    dd = os.path.join(os.path.dirname(argv[0]), "data")
                    with open(os.path.join(scratch, "filelist"), "w") as f:
                        f.write("".join(os.path.join(dd, "cane4_%d.db\n" % j) for j in range(4)))
                    rcwd = scratch
                r = run(argv, rcwd, env, a.timeout, a.cut, a.lock)
                if scratch:
                    shutil.rmtree(scratch, ignore_errors=True)
                rec["forms"][fname].append(digest(r))
        json.dump(rec, open(dest, "w"), separators=(",", ":"))
        print("%-45s %s" % (key, "  ".join("%s %.2fs%s" % (n, min(x["wall_s"] for x in rec["forms"][n]), " T" if any(x["timeout"] for x in rec["forms"][n]) else "") for n, _, _ in forms)), flush=True)


if __name__ == "__main__":
    main()
