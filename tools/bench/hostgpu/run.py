#!/usr/bin/env python3
"""Host-GPU benchmark: run prebuilt CUDA programs (PolyBench/GPU, Rodinia, the
reduced HeCBench list) on the simulator in three forms -- sim (the mode off),
exact (VGPU_HOST_GPU=exact) and turbo (VGPU_HOST_GPU=turbo) -- and
optionally once more as turbo with VGPU_HOST_GPU_VERIFY=all, and report per
program: wall time, speed-up over sim, launches on the card, seconds on the
card and in the simulator, why launches stayed in the simulator, what verify
found, and how the program's output compares with the sim run's.

Needs a machine with the NVIDIA card the simulator is told to be (--gpu) and
the corpora of tools/bench/real (pack_corpora.sh; the same layout, arguments
and timeouts' meaning).  See docs/perf-design/12-host-gpu-turbo.md.

  run.py --build /path/to/build --corp /path/to/corp --gpu nvidia/a10g \\
         --runs 3 --forms sim,exact,turbo,verify --timeout 600 \\
         --json out.json --md out.md
  run.py --dry-run          # self-test, no card, no simulator (wired into ctest)

Forms.  sim, exact, exactdiv (exact with VGPU_HOST_GPU_DIV=1), turbo, exact0 and turbo0
(exact and turbo with VGPU_HOST_GPU_SAMPLE=0: no launch is sampled), and exactw and turbow (with
VGPU_HOST_GPU_SAMPLE=whole: whole blocks only, the rule of Round sp60b), and exactn and turbon (with
VGPU_HOST_GPU_STREAK=0: no launch is sent because the card holds the memory, the rule of Round sp61) are timed
VGPU_HOST_GPU_WIDER=0: the safety rules before Round sp66) are timed
--runs times each, interleaved; a program whose first sim run takes longer
than --long-secs gets --long-runs runs of each.
exact prints its machine-readable lines only with VGPU_HOST_GPU_LOG, which
also prints a line per launch, so the timed exact runs go without it and ONE
extra, untimed exact run with the log collects card_s, sim_s and the refusal
classes (--no-exact-stats skips it; a program cut to --long-runs does not
get it either).  verify is one untimed run.

Output comparison, against the first sim run: stdout and stderr with the
lines that carry timings and the simulator's own lines dropped, and for
Rodinia the files the program wrote.  "identical": equal text and files.
"numeric": the same words, and every number within atol + rtol * |sim|
(--atol/--rtol; NaN matches NaN, an infinity only itself).  "different":
anything else.  The largest absolute and relative difference of two numbers
at the same place is reported in every case where the words line up.

A program that hangs is killed after --timeout seconds (the process group
the harness created for it, nothing else) and recorded as TIMEOUT."""
import argparse
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "real"))
import real  # noqa: E402  (tools/bench/real/real.py: the corpus manifest and the timing-line filter)

FORMS = ("sim", "exact", "exact0", "exactw", "exactn", "exactdiv", "turbo", "turbo0", "turbow", "turbon", "verify")
CARD_FORMS = ("exact", "exact0", "exactw", "exactn", "exactdiv", "turbo", "turbo0", "turbow", "turbon")   # the timed forms that use the card, in the table's order
# lines the simulator or the mode prints: never part of the program's output
OURS = re.compile(r"^(host_gpu[a-z_]*:|\[vgpu\]|vgpu:)")
# A word of the output is a number when all of it is one and it has a fraction or an exponent: an integer (a count, an index, a checksum printed in
# decimal) and anything glued to letters (a hash in hex) are words, compared exactly.
SPLIT = re.compile(r"[^\s,;:=()\[\]{}|]+|[,;:=()\[\]{}|]")
FLOAT = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
NAN = re.compile(r"[-+]?nan(?:\([^)]*\))?", re.I)

# ---- the mode's lines --------------------------------------------------------

def _num(text):
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _fields(rest):
    out = {}
    for part in rest.split():
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = _num(v)
    return out


def parse_stderr(text):
    """What the host-GPU mode said at the end of a run -> dict."""
    r = {"mode_off": None, "summary": None, "turbo": None, "refusals": None, "verify": None,
         "refused_kernels": [], "outside_kernels": [], "moved_mib": None, "timeout_line": None}
    for line in text.splitlines():
        m = re.match(r"host_gpu: (\d+) launches ran on the card, (\d+) in the simulator \((\d+) kernels not "
                     r"(?:in the exact set|admitted by turbo mode), (\d+) too small, (\d+) unsupported launch shapes, "
                     r"(\d+) failed on the card\)", line)
        if m:
            g = [int(x) for x in m.groups()]
            r["summary"] = {"offloaded": g[0], "simulated": g[1], "refused_code": g[2], "too_small": g[3],
                            "refused_safety": g[4], "failed": g[5]}
            continue
        m = re.match(r"host_gpu: ([0-9.]+) MiB went to the card, ([0-9.]+) MiB came back", line)
        if m:
            r["moved_mib"] = [float(m.group(1)), float(m.group(2))]
            continue
        if line.startswith("host_gpu_turbo: "):
            r["turbo"] = _fields(line.split(": ", 1)[1])
        elif line.startswith("host_gpu_refusals: "):
            r["refusals"] = _fields(line.split(": ", 1)[1])
        elif line.startswith("host_gpu_sampled: "):   # launches sampled in a forked copy, how many were sent, what the samples cost
            r["sampled"] = _fields(line.split(": ", 1)[1])
        elif line.startswith("host_gpu_streak: "):   # launches sent because the card held the memory (Round sp65)
            r["streak"] = _fields(line.split(": ", 1)[1])
        elif line.startswith("host_gpu_verify: "):
            r["verify"] = _fields(line.split(": ", 1)[1])
        elif line.startswith("host_gpu_refused: "):
            m = re.match(r"host_gpu_refused: kernel=(\S+) class=(\S+) insn=(.*)", line)
            if m:
                r["refused_kernels"].append({"kernel": m.group(1), "class": m.group(2), "insn": m.group(3)})
        elif line.startswith("host_gpu_outside: "):
            r["outside_kernels"].append(_fields(line.split(": ", 1)[1]))
        elif line.startswith("host_gpu: ") and "mode off" in line and r["mode_off"] is None:
            r["mode_off"] = line[len("host_gpu: "):]
        elif line.startswith("host_gpu: ") and "VGPU_HOST_GPU_TIMEOUT" in line:
            r["timeout_line"] = line[len("host_gpu: "):]
    return r


# ---- comparing a program's output with the sim run's --------------------------

def program_text(stdout, stderr):
    """The program's own output: both streams, without the simulator's lines and
    without the lines that carry timings (real.py's filter)."""
    keep = []
    prev_noise = False
    for stream in (stdout, stderr):
        for l in stream.splitlines():
            if OURS.match(l):
                continue
            n = bool(real.NOISE.search(l)) or bool(re.search(r"\bs,\s.*%", l))
            if not n and not (prev_noise and re.fullmatch(r"\s*[0-9.eE+-]+\s*(s|ms|sec)?\s*", l)):
                keep.append(l)
            prev_noise = n
    return "\n".join(keep)


def close(card, sim, atol, rtol):
    """-> (inside tolerance, abs difference, relative difference); the rule of include/vgpu/host_gpu_tol.hpp."""
    if math.isnan(card) or math.isnan(sim):
        return math.isnan(card) and math.isnan(sim), 0.0, 0.0
    if math.isinf(card) or math.isinf(sim):
        return card == sim, 0.0, 0.0
    d = abs(card - sim)
    return d <= atol + rtol * abs(sim), d, (d / abs(sim) if sim != 0 else 0.0)


def _is_number(tok):
    return bool(FLOAT.fullmatch(tok)) and any(c in tok for c in ".eE")


def compare_text(sim, other, atol, rtol):
    """-> {"class": identical|numeric|different, "max_abs", "max_rel", "numbers", "detail"}"""
    if sim == other:
        return {"class": "identical", "max_abs": 0.0, "max_rel": 0.0, "numbers": 0, "detail": ""}
    ts, to = SPLIT.findall(sim), SPLIT.findall(other)
    res = {"class": "numeric", "max_abs": 0.0, "max_rel": 0.0, "numbers": 0, "detail": ""}
    if len(ts) != len(to):
        res["class"] = "different"
        res["detail"] = "%d words against %d" % (len(to), len(ts))
    for a, b in zip(ts, to):
        if a == b or (NAN.fullmatch(a) and NAN.fullmatch(b)):   # (a NaN prints with either sign)
            continue
        if _is_number(a) and _is_number(b):
            ok, d, rel = close(float(b), float(a), atol, rtol)
            res["numbers"] += 1
            res["max_abs"] = max(res["max_abs"], d)
            res["max_rel"] = max(res["max_rel"], rel)
            if not ok and res["class"] != "different":
                res["class"] = "different"
                res["detail"] = "%s against %s" % (b, a)
        elif res["class"] != "different" or not res["detail"]:
            res["class"] = "different"
            res["detail"] = "'%s' against '%s'" % (b[:24], a[:24])
    if res["class"] == "numeric" and res["numbers"] == 0:
        res["class"] = "different"   # the texts differ, yet no number does: white space
        res["detail"] = res["detail"] or "layout"
    return res


def compare_outputs(sim, other, atol, rtol):
    """sim, other: (text, {file name: bytes}).  The worst class of the text and of each file."""
    order = {"identical": 0, "numeric": 1, "different": 2}
    res = compare_text(sim[0], other[0], atol, rtol)
    for name in sorted(set(sim[1]) | set(other[1])):
        a, b = sim[1].get(name), other[1].get(name)
        if a == b:
            continue
        if a is None or b is None:
            f = {"class": "different", "max_abs": 0.0, "max_rel": 0.0, "numbers": 0, "detail": "file %s on one side only" % name}
        else:
            try:
                f = compare_text(a.decode("ascii"), b.decode("ascii"), atol, rtol)
            except UnicodeDecodeError:
                f = {"class": "different", "max_abs": 0.0, "max_rel": 0.0, "numbers": 0, "detail": "binary"}
            f["detail"] = ("file %s: %s" % (name, f["detail"])).rstrip(": ")
        res["max_abs"] = max(res["max_abs"], f["max_abs"])
        res["max_rel"] = max(res["max_rel"], f["max_rel"])
        res["numbers"] += f["numbers"]
        if order[f["class"]] > order[res["class"]]:
            res["class"], res["detail"] = f["class"], f["detail"]
    return res


# ---- running -------------------------------------------------------------------

def form_env(form, a):
    env = {}
    if form in ("exact", "exact0", "exactw", "exactn", "exactdiv"):
        env["VGPU_HOST_GPU"] = "exact"
        if form == "exactdiv":
            env["VGPU_HOST_GPU_DIV"] = "1"   # the IEEE division sequence joins the exact set (docs/cli.md)
    elif form in ("turbo", "turbo0", "turbow", "turbon", "verify"):
        env["VGPU_HOST_GPU"] = "turbo"
        env["VGPU_HOST_GPU_ATOL"] = repr(a.atol)
        env["VGPU_HOST_GPU_RTOL"] = repr(a.rtol)
    if form == "verify":
        env["VGPU_HOST_GPU_VERIFY"] = "all"
    if form in ("exact0", "turbo0"):
        env["VGPU_HOST_GPU_SAMPLE"] = "0"   # no launch is sampled: the size rule as it was before Round sp60b, for A/B runs
    if form in ("exactw", "turbow"):
        env["VGPU_HOST_GPU_SAMPLE"] = "whole"   # whole blocks only, never inside them: the rule of Round sp60b, for A/B runs of Round sp61
    if form in ("exactn", "turbon"):
        env["VGPU_HOST_GPU_STREAK"] = "0"   # no launch is sent because the card holds the memory: the rule of Round sp61, for A/B runs of Round sp65
        env["VGPU_HOST_GPU_WIDER"] = "0"   # the safety rules before Round sp66 (module-wide refusals, the texture refusal, the atomics opt-in), for A/B runs
    if form != "sim":
        for kv in a.env:
            k, v = kv.split("=", 1)
            env[k] = v
    return env


def run_program(argv, cwd, copy_cwd, env_extra, base_env, timeout):
    """-> dict(wall, rc, timeout, stdout, stderr, files).  The child gets a process group of
    its own; on a timeout that group, and only it, is killed."""
    env = dict(base_env)
    env.update(env_extra)
    scratch = None
    if copy_cwd:   # Rodinia: run in a scratch directory, as real.py does (nn reads a list of absolute db paths)
        scratch = tempfile.mkdtemp(prefix="hostgpu-")
        dd = os.path.join(os.path.dirname(argv[0]), "data")
        with open(os.path.join(scratch, "filelist"), "w") as f:
            f.write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
        cwd = scratch
    t0 = time.monotonic()
    p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    timed_out = False
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(p.pid, signal.SIGKILL)   # (start_new_session: the group's id is the child's pid)
        except ProcessLookupError:
            pass
        out, err = p.communicate()
    wall = time.monotonic() - t0
    files = {}
    if scratch:
        for f in sorted(x for x in os.listdir(scratch) if x != "filelist"):
            fp = os.path.join(scratch, f)
            # (backprop's output.dat is left out, as in real.py: its tail is uninitialised memory)
            if os.path.isfile(fp) and f != "output.dat":
                files[f] = open(fp, "rb").read()
        shutil.rmtree(scratch, ignore_errors=True)
    return {"wall": wall, "rc": p.returncode, "timeout": timed_out, "stdout": out.decode("utf-8", "replace"),
            "stderr": err.decode("utf-8", "replace"), "files": files}


def summarise(form, runs, ref, a):
    """runs: the run dicts of one form of one program -> the record that goes to JSON."""
    last = runs[-1]
    info = parse_stderr(last["stderr"])
    rec = {"wall": [round(r["wall"], 4) for r in runs], "rc": [r["rc"] for r in runs],
           "timeout": any(r["timeout"] for r in runs), "median_s": round(statistics.median(r["wall"] for r in runs), 4),
           "mode_off": info["mode_off"], "refused_kernels": info["refused_kernels"], "moved_mib": info["moved_mib"]}
    s, rel = info["summary"], info["turbo"]
    if rel:
        rec.update(launches_card=rel.get("launches_card"), launches_sim=rel.get("launches_sim"), card_s=rel.get("card_s"),
                   sim_s=rel.get("sim_s"))
    elif s:
        rec.update(launches_card=s["offloaded"], launches_sim=s["simulated"], card_s=None, sim_s=None)
    if s:
        # every launch is either offloaded or counted in "simulated" (a verify run counts its launches there)
        rec["launches_total"] = s["offloaded"] + s["simulated"]
    if info.get("sampled"):
        rec["sampled"] = info["sampled"]
    if info.get("streak"):
        rec["streak"] = info["streak"]
    if info["refusals"]:
        rec["refusals"] = info["refusals"]
    elif s:
        rec["refusals"] = {"refused_code": s["refused_code"], "refused_safety": s["refused_safety"], "too_small": s["too_small"],
                           "failed": s["failed"]}
    if form == "verify":
        rec["verify"] = {"launches": info["verify"], "outside_kernels": info["outside_kernels"]}
        if rel:
            for k in ("words_compared", "words_different", "words_outside", "words_subnormal", "kernels_outside", "max_abs", "max_rel"):
                rec["verify"][k] = rel.get(k)
    if info["timeout_line"]:
        rec["card_timeout"] = info["timeout_line"]
    if ref is not None and not rec["timeout"]:
        outs = [compare_outputs(ref, (program_text(r["stdout"], r["stderr"]), r["files"]), a.atol, a.rtol) for r in runs]
        order = {"identical": 0, "numeric": 1, "different": 2}
        rec["output"] = max(outs, key=lambda o: (order[o["class"]], o["max_rel"], o["max_abs"]))
    return rec


def bench(a):
    corp = os.path.abspath(a.corp)
    progs = real.manifest(corp, a.corpora.split(","))
    if a.programs:
        want = set(a.programs.split(","))
        progs = [p for p in progs if p[0] + "/" + p[1] in want or p[1] in want]
    if a.skip:
        sk = set(a.skip.split(","))
        progs = [p for p in progs if p[0] + "/" + p[1] not in sk and p[1] not in sk]
    forms = [f for f in a.forms.split(",") if f]
    for f in forms:
        if f not in FORMS:
            sys.exit("unknown form %s (of %s)" % (f, ", ".join(FORMS)))
    if "sim" not in forms:
        sys.exit("the sim form is the reference of every comparison: --forms must include it")
    base_env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
    base_env.update(VGPU_GPU=a.gpu, LD_LIBRARY_PATH=os.path.join(os.path.abspath(a.build), "shim"))
    if a.threads != "default":
        base_env["VGPU_THREADS"] = a.threads
    timed = [f for f in forms if f != "verify"]
    res = {"meta": {"gpu": a.gpu, "runs": a.runs, "forms": forms, "timeout": a.timeout, "atol": a.atol, "rtol": a.rtol,
                    "threads": a.threads, "env": a.env, "build": os.path.abspath(a.build)}, "programs": {}}
    for corpus, name, cwd, argv, copy_cwd in progs:
        key = corpus + "/" + name
        runs = {f: [] for f in forms}
        dead = set()   # forms that timed out: not run again
        nruns = a.runs
        i = -1
        while i + 1 < nruns:
            i += 1
            for f in (timed if i % 2 == 0 else timed[::-1]):
                if f in dead:
                    continue
                while a.max_load and os.getloadavg()[0] > a.max_load:
                    time.sleep(15)
                r = run_program(argv, cwd, copy_cwd, form_env(f, a), base_env, a.timeout)
                runs[f].append(r)
                if r["timeout"]:
                    dead.add(f)
            if i == 0 and runs["sim"] and runs["sim"][0]["wall"] > a.long_secs:
                nruns = min(nruns, a.long_runs)   # a program this slow in the simulator gets fewer runs of every form
        ref = None
        if runs["sim"] and not runs["sim"][0]["timeout"]:
            s0 = runs["sim"][0]
            ref = (program_text(s0["stdout"], s0["stderr"]), s0["files"])
        rec = {}
        for f in timed:
            rec[f] = summarise(f, runs[f], None if f == "sim" else ref, a)
        if "exact" in forms and not a.no_exact_stats and "exact" not in dead and nruns == a.runs:   # (not for a long program: one more run of it)
            env = form_env("exact", a)
            env["VGPU_HOST_GPU_LOG"] = "1"
            r = run_program(argv, cwd, copy_cwd, env, base_env, a.timeout)
            st = summarise("exact", [r], None, a)
            for k in ("card_s", "sim_s", "refusals", "launches_card", "launches_sim", "sampled", "streak"):
                if st.get(k) is not None:
                    rec["exact"][k] = st[k]
        if "verify" in forms:
            r = run_program(argv, cwd, copy_cwd, form_env("verify", a), base_env, a.verify_timeout or a.timeout)
            rec["verify"] = summarise("verify", [r], ref, a)
        if len(runs["sim"]) > 1 and ref is not None:   # does the simulator repeat itself? (a program that does not cannot be compared)
            again = [compare_outputs(ref, (program_text(r["stdout"], r["stderr"]), r["files"]), a.atol, a.rtol) for r in runs["sim"][1:] if not r["timeout"]]
            rec["sim"]["repeats"] = all(o["class"] == "identical" for o in again)
        sim_s = rec["sim"]["median_s"]
        for f in timed:
            if f != "sim" and not rec[f]["timeout"] and not rec["sim"]["timeout"] and rec[f]["median_s"] > 0:
                rec[f]["speedup"] = round(sim_s / rec[f]["median_s"], 3)
        res["programs"][key] = rec
        print(progress_line(key, rec, timed), flush=True)
        if a.json:
            with open(a.json, "w") as fh:
                json.dump(res, fh, indent=1)
    md = markdown(res)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(md)
    print()
    print(md)
    return 0


def progress_line(key, rec, timed):
    line = key.ljust(36)
    for f in timed:
        d = rec[f]
        line += " %s %s" % (f, "TIMEOUT" if d["timeout"] else "%.2fs" % d["median_s"])
        if d.get("mode_off"):
            line += " (mode off)"
    return line


# ---- the table -----------------------------------------------------------------

def _g(v, fmt="%.3g"):
    return "-" if v is None else (fmt % v)


def _time(d):
    if d is None:
        return "-"
    if d["timeout"]:
        return "TIMEOUT"
    bad = sorted(set(r for r in d["rc"] if r != 0))
    return "%.2f" % d["median_s"] + (" rc%s" % bad if bad else "")


def _speed(d):
    if d is None or d.get("speedup") is None:
        return "-"
    return "%.2fx" % d["speedup"]


def _launches(d):
    if d is None or d.get("launches_card") is None:
        return "off" if d is not None and d.get("mode_off") else "-"
    total = d.get("launches_total")
    return "%d / %s" % (d["launches_card"], "?" if total is None else total)


def _refusals(d):
    if d is None or not d.get("refusals"):
        return "-"
    r = d["refusals"]
    if "refused_never" in r:
        return "%d/%d/%d/%d/%d/%d" % (r["refused_never"], r["refused_atomics"], r["refused_off"], r["refused_exact"], r["refused_safety"], r["too_small"])
    return "code %d, safety %d, small %d" % (r["refused_code"], r["refused_safety"], r["too_small"])


def _output(d):
    if d is None or "output" not in d:
        return "-"
    o = d["output"]
    if o["class"] == "identical":
        return "identical"
    text = o["class"]
    if o["numbers"]:
        text += " (abs %.3g, rel %.3g)" % (o["max_abs"], o["max_rel"])
    elif o["detail"]:
        text += " (%s)" % o["detail"].replace("|", "/")[:40]
    return text


def _verify(d):
    if d is None:
        return "-"
    if d["timeout"]:
        return "TIMEOUT"
    v = d.get("verify") or {}
    if v.get("words_compared") is None:
        return "off" if d.get("mode_off") else "-"
    return "%d / %d (abs %s, rel %s)" % (v["words_different"], v["words_outside"], _g(v.get("max_abs")), _g(v.get("max_rel")))


def markdown(res):
    m = res["meta"]
    forms = m["forms"]
    out = ["Host-GPU benchmark: %s, %d runs (median wall seconds), timeout %g s, atol %g, rtol %g, threads %s%s." % (
        m["gpu"], m["runs"], m["timeout"], m["atol"], m["rtol"], m["threads"], (", " + " ".join(m["env"])) if m["env"] else ""), ""]
    cols = ["program", "sim s"]
    for f in CARD_FORMS:
        if f in forms:
            cols += ["%s s" % f, "x sim", "%s card / all launches" % f, "%s card s / sim s" % f, "%s output" % f]
    if "turbo" in forms:
        cols.append("turbo kept in sim: never/atomics/off/exact/safety/small")
    if "verify" in forms:
        cols += ["verify words different / outside (largest)", "verify kernels outside", "verify output"]
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|")
    speed = {f: [] for f in CARD_FORMS}
    notes = []
    for key, rec in res["programs"].items():
        row = [key, _time(rec.get("sim"))]
        for f in CARD_FORMS:
            if f not in forms:
                continue
            d = rec.get(f)
            row += [_time(d), _speed(d), _launches(d),
                    "-" if d is None or d.get("card_s") is None else "%.2f / %.2f" % (d["card_s"], d["sim_s"]), _output(d)]
            if d and d.get("speedup"):
                speed[f].append(d["speedup"])
            if d and d.get("mode_off"):
                notes.append("%s %s: %s" % (key, f, d["mode_off"]))
            if d and d.get("card_timeout"):
                notes.append("%s %s: %s" % (key, f, d["card_timeout"]))
        if "turbo" in forms:
            row.append(_refusals(rec.get("turbo")))
        if "verify" in forms:
            d = rec.get("verify")
            v = (d or {}).get("verify") or {}
            row += [_verify(d), "-" if v.get("kernels_outside") is None else str(v["kernels_outside"]), _output(d)]
        if rec.get("sim", {}).get("repeats") is False:
            notes.append("%s: the sim runs do not print the same output twice: its comparisons mean little" % key)
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    for f in CARD_FORMS:
        if speed[f]:
            out.append("%s: geometric mean speed-up over sim %.2fx (%d programs)." % (f, real.geomean(speed[f]), len(speed[f])))
    missing = {}
    for key, rec in res["programs"].items():
        for k in (rec.get("turbo") or {}).get("refused_kernels", []):
            missing.setdefault((k["class"], k["insn"].split(" ")[0]), []).append(key)
    if missing:
        out += ["", "Kernels turbo mode kept in the simulator, by class and first refused instruction:", ""]
        for (cls, mn), keys in sorted(missing.items()):
            out.append("* %s `%s`: %d kernels (%s)" % (cls, mn, len(keys), ", ".join(sorted(set(keys))[:8])))
    if notes:
        out += ["", "Notes:", ""] + ["* " + n for n in notes]
    return "\n".join(out) + "\n"


# ---- self-test (--dry-run): no card, no simulator -----------------------------------

CANNED_TURBO = """[vgpu] device nvidia/a10g
host_gpu_refused: kernel=_Z9tex_fetchPfi class=never insn=TEX.SCR.LL R4, R5, R2, R2, 0x0, 0x58, 2D, 0x1
host_gpu_refused: kernel=_Z4lockPi class=atomics insn=ATOMG.E.CAS.STRONG.GPU P0, R6, [R2.64], R4, R5
host_gpu: verify _Z8use_expfPKfPfi (admitted): 1048576 words compared, 4711 different, 0 outside tolerance, 0 zero or denormal on both sides; largest difference 2.38418579e-07, relative 1.19209275e-07
host_gpu: 37 launches ran on the card, 12 in the simulator (5 kernels not admitted by turbo mode, 3 too small, 1 unsupported launch shapes, 0 failed on the card)
host_gpu: 96.0 MiB went to the card, 12.5 MiB came back
host_gpu_verify: in_set_identical=3 in_set_different=34 other_identical=0 other_different=5
host_gpu_turbo: mode=turbo launches_card=37 launches_sim=12 card_s=0.412345 sim_s=3.250000 words_compared=9437184 words_different=4711 words_outside=2 words_subnormal=0 kernels_outside=1 max_abs=2.38418579e-07 max_rel=0.5
host_gpu_refusals: refused_never=3 refused_atomics=2 refused_off=0 refused_exact=0 refused_safety=1 too_small=3 failed=0
host_gpu_sampled: sampled=2 sent=1 timeouts=0 cost_s=0.006100 after_driver=0 copies_running_at_exit=0 copies_left=0
host_gpu_streak: sent=7 card_only_mib=48.5 rule=1
host_gpu_outside: kernel=_Z4histPiS_i words_compared=262144 words_different=2 words_outside=2 max_abs=1 max_rel=0.5 first=0x7f4a00001040:card=0x00000003:sim=0x00000002
"""
CANNED_EXACT = """host_gpu: 8 launches ran on the card, 41 in the simulator (30 kernels not in the exact set, 2 too small, 4 unsupported launch shapes, 0 failed on the card)
host_gpu: 16.0 MiB went to the card, 4.0 MiB came back
"""
CANNED_OFF = "host_gpu: cannot load the real driver /usr/lib/x86_64-linux-gnu/libcuda.so.1; mode off\n"


def selftest():
    fails = []

    def check(cond, what):
        if not cond:
            fails.append(what)

    # the mode's lines
    r = parse_stderr(CANNED_TURBO)
    check(r["summary"] == {"offloaded": 37, "simulated": 12, "refused_code": 5, "too_small": 3, "refused_safety": 1, "failed": 0}, "turbo prose summary")
    check(r["turbo"]["mode"] == "turbo" and r["turbo"]["launches_card"] == 37 and r["turbo"]["launches_sim"] == 12, "host_gpu_turbo launches")
    check(r["turbo"]["card_s"] == 0.412345 and r["turbo"]["sim_s"] == 3.25, "host_gpu_turbo seconds")
    check(r["turbo"]["words_compared"] == 9437184 and r["turbo"]["words_different"] == 4711 and r["turbo"]["words_outside"] == 2, "host_gpu_turbo words")
    check(r["turbo"]["kernels_outside"] == 1 and abs(r["turbo"]["max_abs"] - 2.38418579e-07) < 1e-15 and r["turbo"]["max_rel"] == 0.5, "host_gpu_turbo maxima")
    check(r["sampled"]["sampled"] == 2 and r["sampled"]["sent"] == 1 and r["sampled"]["after_driver"] == 0 and r["sampled"]["copies_left"] == 0, "host_gpu_sampled")
    check(r["streak"] == {"sent": 7, "card_only_mib": 48.5, "rule": 1}, "host_gpu_streak")
    check(r["refusals"] == {"refused_never": 3, "refused_atomics": 2, "refused_off": 0, "refused_exact": 0, "refused_safety": 1, "too_small": 3, "failed": 0}, "host_gpu_refusals")
    check(r["verify"] == {"in_set_identical": 3, "in_set_different": 34, "other_identical": 0, "other_different": 5}, "host_gpu_verify")
    check(len(r["refused_kernels"]) == 2 and r["refused_kernels"][0] == {"kernel": "_Z9tex_fetchPfi", "class": "never", "insn": "TEX.SCR.LL R4, R5, R2, R2, 0x0, 0x58, 2D, 0x1"}, "host_gpu_refused")
    check(r["refused_kernels"][1]["class"] == "atomics" and r["refused_kernels"][1]["insn"].startswith("ATOMG.E.CAS"), "host_gpu_refused atomics")
    check(len(r["outside_kernels"]) == 1 and r["outside_kernels"][0]["kernel"] == "_Z4histPiS_i" and r["outside_kernels"][0]["words_outside"] == 2, "host_gpu_outside")
    check(r["outside_kernels"][0]["first"] == "0x7f4a00001040:card=0x00000003:sim=0x00000002", "host_gpu_outside first")
    check(r["moved_mib"] == [96.0, 12.5] and r["mode_off"] is None, "MiB line")
    e = parse_stderr(CANNED_EXACT)
    check(e["summary"]["offloaded"] == 8 and e["summary"]["refused_code"] == 30 and e["turbo"] is None and e["refusals"] is None, "exact prose summary")
    o = parse_stderr(CANNED_OFF)
    check(o["mode_off"] and o["mode_off"].startswith("cannot load the real driver") and o["summary"] is None, "mode off line")
    check(parse_stderr("")["summary"] is None and parse_stderr("host_gpu_turbo: garbage\n")["turbo"] == {}, "empty and malformed lines")
    t = parse_stderr("host_gpu: k has been on the card for more than 600 s (VGPU_HOST_GPU_TIMEOUT): a kernel ...\n")
    check(t["timeout_line"] is not None and t["mode_off"] is None, "card timeout line")

    # the program's own text
    txt = program_text("result 1.5\nTime: 3.2 s\nhost_gpu: x\n", "host_gpu_turbo: mode=turbo\n[vgpu] hello\nwarning: y\n")
    check(txt == "result 1.5\nwarning: y", "program_text drops timings and our lines: %r" % txt)

    # numbers
    check(close(1.0 + 5e-6, 1.0, 1e-8, 1e-5)[0] and not close(1.0 + 2e-5, 1.0, 1e-8, 1e-5)[0], "close: relative")
    check(close(5e-9, 0.0, 1e-8, 1e-5)[0] and not close(2e-8, 0.0, 1e-8, 1e-5)[0], "close: absolute")
    check(close(float("nan"), float("nan"), 0, 0)[0] and not close(float("nan"), 1.0, 1e9, 1e9)[0], "close: nan")
    check(close(float("inf"), float("inf"), 0, 0)[0] and not close(float("inf"), 1e308, 1e9, 1e9)[0] and not close(float("inf"), float("-inf"), 1e9, 1e9)[0], "close: inf")
    check(close(2.0, 4.0, 0, 0.5)[0] and not close(4.0, 2.0, 0, 0.5)[0], "close: relative to sim, not to the card")

    # output classes
    c = compare_text("sum = 1.000000 ok\n", "sum = 1.000000 ok\n", 1e-8, 1e-5)
    check(c["class"] == "identical", "identical text")
    c = compare_text("sum = 1.000000, max 2.5e+03 ok\n", "sum = 1.000001, max 2.5e+03 ok\n", 1e-8, 1e-5)
    check(c["class"] == "numeric" and c["numbers"] == 1 and abs(c["max_abs"] - 1e-6) < 1e-12 and abs(c["max_rel"] - 1e-6) < 1e-12, "numeric text: %r" % c)
    c = compare_text("sum = 1.000000\n", "sum = 1.100000\n", 1e-8, 1e-5)
    check(c["class"] == "different" and abs(c["max_abs"] - 0.1) < 1e-9 and "1.100000 against 1.000000" in c["detail"], "different number: %r" % c)
    c = compare_text("checksum 3f9a11c2\n", "checksum 3f9a11c3\n", 1e-8, 1e-5)
    check(c["class"] == "different", "a hash that differs is different: %r" % c)
    c = compare_text("PASS 1.0\n", "FAIL 1.0\n", 1e-8, 1e-5)
    check(c["class"] == "different" and "FAIL" in c["detail"], "a word that differs: %r" % c)
    c = compare_text("a 1.0\nb 2.0\n", "a 1.0\n", 1e-8, 1e-5)
    check(c["class"] == "different" and "words" in c["detail"], "a missing line: %r" % c)
    c = compare_text("v nan inf -inf\n", "v nan inf inf\n", 1e-8, 1e-5)
    check(c["class"] == "different", "an infinity of the other sign: %r" % c)
    c = compare_text("v nan 1.5\n", "v -nan 1.5000001\n", 1e-8, 1e-5)
    check(c["class"] == "numeric" and c["numbers"] == 1, "a NaN of the other sign is a NaN: %r" % c)
    c = compare_text("nodes 1000000 hash 12345678\n", "nodes 1000001 hash 12345678\n", 1e-8, 1e-5)
    check(c["class"] == "different", "an integer that differs is different, however close: %r" % c)
    c = compare_text("hash 12345678\n", "hash 12345679\n", 1.0, 1.0)
    check(c["class"] == "different", "a decimal checksum is a word: %r" % c)
    c = compare_text("e 1e5 3.\n", "e 1.000001e5 3.00000001\n", 1e-8, 1e-5)
    check(c["class"] == "numeric" and c["numbers"] == 2, "exponents and a bare point: %r" % c)
    c = compare_text("x=-1.5e-3;y=7\n", "x=-1.5000001e-3;y=7\n", 1e-8, 1e-5)
    check(c["class"] == "numeric" and c["numbers"] == 1, "numbers next to punctuation: %r" % c)
    c = compare_text("a  1.0\n", "a 1.0\n", 1e-8, 1e-5)
    check(c["class"] == "different" and c["detail"] == "layout", "white space only: %r" % c)
    c = compare_outputs(("x 1.0", {"out.txt": b"1.0 2.0\n"}), ("x 1.0", {"out.txt": b"1.0 2.000001\n"}), 1e-8, 1e-5)
    check(c["class"] == "numeric" and c["numbers"] == 1, "a file within tolerance: %r" % c)
    c = compare_outputs(("x 1.0", {"out.bin": b"\xff\x00\x01"}), ("x 1.0", {"out.bin": b"\xff\x00\x02"}), 1e-8, 1e-5)
    check(c["class"] == "different" and "out.bin" in c["detail"] and "binary" in c["detail"], "a binary file that differs: %r" % c)
    c = compare_outputs(("x 1.0", {"a": b"1"}), ("x 1.0", {}), 1e-8, 1e-5)
    check(c["class"] == "different" and "one side" in c["detail"], "a file on one side: %r" % c)
    c = compare_outputs(("x 1.0", {"a": b"1"}), ("x 1.0", {"a": b"1"}), 1e-8, 1e-5)
    check(c["class"] == "identical", "identical with files")

    # a real child: output, files' absence, and a hang that is killed (its whole group, nothing else)
    env = dict(os.environ)
    r = run_program([sys.executable, "-c", "import sys; print('value 2.5'); sys.stderr.write('host_gpu: 1 launches ran\\n')"], None, False, {"X": "1"}, env, 30)
    check(r["rc"] == 0 and not r["timeout"] and r["stdout"] == "value 2.5\n" and r["stderr"].startswith("host_gpu:"), "a child's output")
    t0 = time.monotonic()
    code = ("import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "print('started', flush=True)\ntime.sleep(60)\n")
    r = run_program([sys.executable, "-c", code], None, False, {}, env, 1.0)
    took = time.monotonic() - t0
    check(r["timeout"] and r["rc"] == -signal.SIGKILL and took < 20, "a hang is killed: rc %r in %.1f s" % (r["rc"], took))
    check("started" in r["stdout"], "the output before the hang is kept")

    # summarise and the table, from canned runs
    class A:
        atol, rtol = 1e-8, 1e-5

    def fake(wall, out, err, timeout=False, rc=0):
        return {"wall": wall, "rc": rc, "timeout": timeout, "stdout": out, "stderr": err, "files": {}}
    ref = (program_text("sum 1.000000\n", ""), {})
    sim = summarise("sim", [fake(10.0, "sum 1.000000\n", ""), fake(12.0, "sum 1.000000\n", "")], None, A)
    exact = summarise("exact", [fake(5.0, "sum 1.000000\n", CANNED_EXACT)], ref, A)
    turbo = summarise("turbo", [fake(1.0, "sum 1.000002\n", CANNED_TURBO), fake(1.2, "sum 1.000002\n", CANNED_TURBO)], ref, A)
    verify = summarise("verify", [fake(30.0, "sum 1.000000\n", CANNED_TURBO)], ref, A)
    hung = summarise("turbo", [fake(600.0, "", "", timeout=True, rc=-9)], ref, A)
    off = summarise("turbo", [fake(10.5, "sum 1.000000\n", CANNED_OFF)], ref, A)
    check(sim["median_s"] == 11.0 and "output" not in sim, "sim record")
    check(exact["launches_card"] == 8 and exact["launches_total"] == 49 and exact["card_s"] is None and exact["output"]["class"] == "identical", "exact record: %r" % exact)
    check(exact["refusals"] == {"refused_code": 30, "refused_safety": 4, "too_small": 2, "failed": 0}, "exact refusals from the prose line")
    check(turbo["launches_card"] == 37 and turbo["launches_total"] == 49 and turbo["card_s"] == 0.412345 and turbo["sim_s"] == 3.25, "turbo record")
    check(turbo["output"]["class"] == "numeric" and abs(turbo["output"]["max_abs"] - 2e-6) < 1e-10 and turbo["median_s"] == 1.1, "turbo output class: %r" % turbo["output"])
    check(turbo["refusals"]["refused_never"] == 3 and len(turbo["refused_kernels"]) == 2, "turbo refusals")
    check(verify["verify"]["words_outside"] == 2 and verify["verify"]["kernels_outside"] == 1 and verify["verify"]["launches"]["in_set_different"] == 34, "verify record")
    check(verify["output"]["class"] == "identical", "verify output")
    check(hung["timeout"] and "output" not in hung, "a timed-out form has no output class")
    check(off["mode_off"].startswith("cannot load") and off.get("launches_card") is None, "mode off record")
    exact["speedup"], turbo["speedup"] = 2.2, 10.0
    res = {"meta": {"gpu": "nvidia/a10g", "runs": 2, "forms": list(FORMS), "timeout": 600, "atol": 1e-8, "rtol": 1e-5, "threads": "default", "env": []},
           "programs": {"polybench/GEMM": {"sim": sim, "exact": exact, "turbo": turbo, "verify": verify},
                        "hecbench/hang": {"sim": sim, "exact": exact, "turbo": hung},
                        "rodinia/nocard": {"sim": sim, "turbo": off}}}
    md = markdown(res)
    rows = [l for l in md.splitlines() if l.startswith("| ")]
    check(len(rows) == 4 and len(set(l.count("|") for l in rows)) == 1, "every table row has the header's columns:\n" + md)
    gemm = [l for l in rows if l.startswith("| polybench/GEMM")][0]
    for want in ("| 11.00 |", "| 5.00 | 2.20x | 8 / 49 |", "| 1.10 | 10.00x | 37 / 49 | 0.41 / 3.25 | numeric (abs 2e-06, rel 2e-06) |", "| 3/2/0/0/1/3 |",
                 "| 4711 / 2 (abs 2.38e-07, rel 0.5) | 1 | identical |"):
        check(want in gemm, "GEMM row lacks %r:\n%s" % (want, gemm))
    check("| TIMEOUT | - | - | - | - |" in [l for l in rows if l.startswith("| hecbench/hang")][0], "the hung program's row")
    check("| off |" in [l for l in rows if l.startswith("| rodinia/nocard")][0], "the no-card row")
    check("turbo: geometric mean speed-up over sim 10.00x (1 programs)." in md and "exact: geometric mean speed-up over sim 2.20x (2 programs)." in md, "geometric means")
    check("* never `TEX.SCR.LL`: 1 kernels (polybench/GEMM)" in md and "* atomics `ATOMG.E.CAS.STRONG.GPU`: 1 kernels" in md, "the list of refused kernels")
    check("rodinia/nocard turbo: cannot load the real driver" in md, "the mode-off note")
    json.dumps(res)   # everything that goes to the JSON file can be written

    # the forms' environments
    class B:
        atol, rtol, env = 1e-8, 1e-5, ["VGPU_HOST_GPU_ATOMICS=1"]
    check(form_env("sim", B) == {}, "sim sets nothing")
    check(form_env("exact", B) == {"VGPU_HOST_GPU": "exact", "VGPU_HOST_GPU_ATOMICS": "1"}, "exact env")
    check(form_env("exactdiv", B) == {"VGPU_HOST_GPU": "exact", "VGPU_HOST_GPU_DIV": "1", "VGPU_HOST_GPU_ATOMICS": "1"}, "exactdiv env")
    v = form_env("verify", B)
    check(v["VGPU_HOST_GPU"] == "turbo" and v["VGPU_HOST_GPU_VERIFY"] == "all" and float(v["VGPU_HOST_GPU_RTOL"]) == 1e-5 and v["VGPU_HOST_GPU_ATOMICS"] == "1", "verify env")
    check("VGPU_HOST_GPU_VERIFY" not in form_env("turbo", B), "turbo env")
    check(form_env("exact0", B) == {"VGPU_HOST_GPU": "exact", "VGPU_HOST_GPU_SAMPLE": "0", "VGPU_HOST_GPU_ATOMICS": "1"}, "exact0 env")
    check(form_env("turbo0", B)["VGPU_HOST_GPU"] == "turbo" and form_env("turbo0", B)["VGPU_HOST_GPU_SAMPLE"] == "0" and
          "VGPU_HOST_GPU_SAMPLE" not in form_env("turbo", B) and "VGPU_HOST_GPU_SAMPLE" not in form_env("exact", B), "turbo0 env")
    check(form_env("exactw", B) == {"VGPU_HOST_GPU": "exact", "VGPU_HOST_GPU_SAMPLE": "whole", "VGPU_HOST_GPU_ATOMICS": "1"}, "exactw env")
    check(form_env("turbow", B)["VGPU_HOST_GPU"] == "turbo" and form_env("turbow", B)["VGPU_HOST_GPU_SAMPLE"] == "whole", "turbow env")
    check(form_env("exactn", B) == {"VGPU_HOST_GPU": "exact", "VGPU_HOST_GPU_STREAK": "0", "VGPU_HOST_GPU_ATOMICS": "1"}, "exactn env")
    check(form_env("turbon", B)["VGPU_HOST_GPU"] == "turbo" and form_env("turbon", B)["VGPU_HOST_GPU_STREAK"] == "0" and
          "VGPU_HOST_GPU_STREAK" not in form_env("turbo", B) and "VGPU_HOST_GPU_STREAK" not in form_env("exact", B), "turbon env")
    check(form_env("exactn", B) == {"VGPU_HOST_GPU": "exact", "VGPU_HOST_GPU_WIDER": "0", "VGPU_HOST_GPU_ATOMICS": "1"}, "exactn env")
    check(form_env("turbon", B)["VGPU_HOST_GPU"] == "turbo" and form_env("turbon", B)["VGPU_HOST_GPU_WIDER"] == "0" and
          "VGPU_HOST_GPU_WIDER" not in form_env("turbo", B) and "VGPU_HOST_GPU_WIDER" not in form_env("exact", B), "turbon env")

    for f in fails:
        print("FAIL: " + f)
    print("%s: %d checks failed" % ("FAIL" if fails else "PASS", len(fails)))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="self-test: parsing, comparison, the table and the timeout, with no card and no simulator")
    ap.add_argument("--build", help="the simulator's build directory (its shim/ is put on LD_LIBRARY_PATH)")
    ap.add_argument("--corp", help="the corpora, in tools/bench/real/pack_corpora.sh's layout")
    ap.add_argument("--gpu", default="nvidia/a10g", help="VGPU_GPU: the simulated device, which must be the machine's card")
    ap.add_argument("--corpora", default="polybench,rodinia,hecbench")
    ap.add_argument("--programs", default="", help="comma list of corpus/name or name; default all")
    ap.add_argument("--skip", default="", help="comma list of corpus/name or name to leave out")
    ap.add_argument("--forms", default="sim,exact,turbo", help="of sim,exact,exact0,exactw,exactn,exactdiv,turbo,turbo0,turbow,turbon,verify (exactdiv: exact with VGPU_HOST_GPU_DIV=1; exact0, turbo0: with VGPU_HOST_GPU_SAMPLE=0; exactw, turbow: with VGPU_HOST_GPU_SAMPLE=whole; exactn, turbon: with VGPU_HOST_GPU_STREAK=0; verify: one untimed turbo run with VGPU_HOST_GPU_VERIFY=all)")
    ap.add_argument("--forms", default="sim,exact,turbo", help="of sim,exact,exact0,exactw,exactdiv,turbo,turbo0,turbow,verify (exactdiv: exact with VGPU_HOST_GPU_DIV=1; exact0, turbo0: with VGPU_HOST_GPU_SAMPLE=0; exactw, turbow: with VGPU_HOST_GPU_SAMPLE=whole; exactn, turbon: with VGPU_HOST_GPU_WIDER=0, the safety rules before Round sp66; verify: one untimed turbo run with VGPU_HOST_GPU_VERIFY=all)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--long-secs", type=float, default=120, help="a program whose first sim run takes longer gets --long-runs runs")
    ap.add_argument("--long-runs", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600, help="seconds per run; a run past it is killed and recorded")
    ap.add_argument("--verify-timeout", type=float, default=0, help="seconds for the verify run (default: --timeout)")
    ap.add_argument("--threads", default="default", help="VGPU_THREADS, or 'default' to leave it unset")
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--env", action="append", default=[], metavar="K=V", help="added to every form but sim (e.g. VGPU_HOST_GPU_ATOMICS=1); repeatable")
    ap.add_argument("--no-exact-stats", action="store_true", help="skip the extra exact run with VGPU_HOST_GPU_LOG=1")
    ap.add_argument("--max-load", type=float, default=0, help="wait while the 1-minute load average is above this (0: do not wait)")
    ap.add_argument("--json", default="")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    if a.dry_run:
        return selftest()
    if not a.build or not a.corp:
        ap.error("--build and --corp are needed (or --dry-run)")
    for kv in a.env:
        if "=" not in kv:
            ap.error("--env takes K=V")
    return bench(a)


if __name__ == "__main__":
    sys.exit(main())
