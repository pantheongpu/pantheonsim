#!/usr/bin/env python3
"""Discover a real GPU's registers, to map them into VirtualGPU's register database.

Runs on the machine with the GPU. Python 3 standard library only, plus
mmio_read.c (compiled on first use) for BAR reads. Every step is read-only.

  capture  OUT [--bdf B]   what the system says about each GPU: configuration
                           space (all 4096 bytes as root, the 64-byte header
                           otherwise), lspci -vvv, the sysfs files, nvidia-smi
                           -q or rocm-smi, and on AMD the IP discovery tree
  mmio     OUT --bdf B --range START-END [--bar N] [--stride S]
                           read a BAR's registers (root). Each value is flushed
                           to OUT/mmio-*.tsv as it is read, so a hang keeps them
  correlate OUT --bdf B --range START-END [--seconds S] [--load CMD]
                           sample registers alongside nvidia-smi's readings --
                           idle, then while CMD runs -- and rank the offsets
                           whose values follow temperature, clocks, power and
                           utilization: candidates for what each register is
  compare  CAPTURE --profile ID [--vgpu PATH]
                           (at home) diff a captured configuration space
                           against VirtualGPU's model of that profile

A sweep over a region nothing documents can hang a GPU or the host: reading
some registers has side effects, and protected ones may stall. Sweep a card
that nobody else is using, starting small.
"""
import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
                           universal_newlines=True)
        return r.stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return "<%s: %s>" % (" ".join(cmd), e)


def gpus():
    """PCI addresses of display and accelerator devices from NVIDIA and AMD."""
    out = []
    base = "/sys/bus/pci/devices"
    for bdf in sorted(os.listdir(base)):
        d = os.path.join(base, bdf)
        try:
            vendor = open(os.path.join(d, "vendor")).read().strip()
            cls = int(open(os.path.join(d, "class")).read().strip(), 16)
        except OSError:
            continue
        if vendor in ("0x10de", "0x1002") and cls >> 16 in (0x03, 0x12):
            out.append(bdf)
    return out


def read_bytes(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError as e:
        return None


def copy_tree(src, dst):
    """Copies a sysfs tree's readable files, following no symlinks out of it."""
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            data = read_bytes(os.path.join(root, f))
            if data is not None:
                with open(os.path.join(dst, rel, f), "wb") as out:
                    out.write(data)


def cmd_capture(args):
    os.makedirs(args.out, exist_ok=True)
    meta = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kernel": platform.release(),
        "root": os.geteuid() == 0,
        "gpus": [],
    }
    for bdf in [args.bdf] if args.bdf else gpus():
        d = "/sys/bus/pci/devices/" + bdf
        g = os.path.join(args.out, bdf)
        os.makedirs(g, exist_ok=True)
        cfg = read_bytes(d + "/config") or b""
        with open(os.path.join(g, "config.bin"), "wb") as f:
            f.write(cfg)
        info = {"bdf": bdf, "config_bytes": len(cfg)}
        for name in ("vendor", "device", "class", "revision", "subsystem_vendor", "subsystem_device",
                     "resource", "current_link_speed", "current_link_width", "max_link_speed",
                     "max_link_width", "numa_node", "aer_dev_correctable", "aer_dev_nonfatal",
                     "aer_dev_fatal", "driver_override"):
            data = read_bytes(os.path.join(d, name))
            if data is not None:
                info[name] = data.decode(errors="replace").strip()
        driver = os.path.join(d, "driver")
        info["driver"] = os.path.basename(os.path.realpath(driver)) if os.path.exists(driver) else None
        with open(os.path.join(g, "lspci-vvv.txt"), "w") as f:
            f.write(run(["lspci", "-vvv", "-nn", "-s", bdf]))
        with open(os.path.join(g, "lspci-xxxx.txt"), "w") as f:
            f.write(run(["lspci", "-xxxx", "-s", bdf]))
        # AMD: where the driver found each IP block, per die and instance.
        disc = d + "/ip_discovery"
        if os.path.isdir(disc):
            copy_tree(disc, os.path.join(g, "ip_discovery"))
            info["ip_discovery"] = True
        meta["gpus"].append(info)
    if shutil.which("nvidia-smi"):
        with open(os.path.join(args.out, "nvidia-smi-q.txt"), "w") as f:
            f.write(run(["nvidia-smi", "-q"]))
        meta["driver"] = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).strip()
    if shutil.which("rocm-smi"):
        with open(os.path.join(args.out, "rocm-smi.txt"), "w") as f:
            f.write(run(["rocm-smi", "--showallinfo"]))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    for info in meta["gpus"]:
        print("%s: %d bytes of configuration space%s" % (
            info["bdf"], info["config_bytes"],
            "" if info["config_bytes"] >= 4096 else " (run as root for all 4096)"))
    print("wrote", args.out)


def reader():
    """mmio_read, compiled next to this script the first time it is needed."""
    exe = os.path.join(HERE, "mmio_read")
    src = os.path.join(HERE, "mmio_read.c")
    if not os.path.exists(exe) or os.path.getmtime(exe) < os.path.getmtime(src):
        subprocess.check_call(["cc", "-O2", "-o", exe, src])
    return exe


def parse_range(s):
    a, b = s.split("-")
    return int(a, 0), int(b, 0)


def read_range(bdf, bar, start, end, stride, out_path=None):
    """{offset: value} for the range, written to out_path line by line."""
    res = "/sys/bus/pci/devices/%s/resource%d" % (bdf, bar)
    cmd = [reader(), res, hex(start), hex(end), str(stride)]
    values = {}
    sink = open(out_path, "a") if out_path else None
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for line in p.stdout:
        off, val = line.split()
        values[int(off, 16)] = int(val, 16)
        if sink:
            sink.write(line)
            sink.flush()
    if p.wait() != 0:
        raise SystemExit("mmio_read failed; see its message above")
    if sink:
        sink.close()
    return values


def cmd_mmio(args):
    if os.geteuid() != 0:
        raise SystemExit("reading a BAR needs root")
    os.makedirs(args.out, exist_ok=True)
    start, end = parse_range(args.range)
    path = os.path.join(args.out, "mmio-%s-bar%d-%x-%x.tsv" % (args.bdf, args.bar, start, end))
    values = read_range(args.bdf, args.bar, start, end, args.stride, path)
    nonzero = sum(1 for v in values.values() if v not in (0, 0xFFFFFFFF))
    print("%d registers read, %d neither 0 nor all ones; wrote %s" % (len(values), nonzero, path))


SMI_FIELDS = ["temperature.gpu", "clocks.sm", "clocks.mem", "power.draw", "utilization.gpu", "fan.speed"]


def smi_sample():
    out = run(["nvidia-smi", "--query-gpu=" + ",".join(SMI_FIELDS), "--format=csv,noheader,nounits"])
    vals = []
    for x in out.strip().splitlines()[0].split(","):
        try:
            vals.append(float(x))
        except ValueError:
            vals.append(float("nan"))
    return dict(zip(SMI_FIELDS, vals))


def pearson(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if not (math.isnan(x) or math.isnan(y))]
    if len(pts) < 3:
        return 0.0
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    syy = sum((p[1] - my) ** 2 for p in pts)
    if sxx == 0 or syy == 0:
        return 0.0
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / math.sqrt(sxx * syy)


def cmd_correlate(args):
    if os.geteuid() != 0:
        raise SystemExit("reading a BAR needs root")
    os.makedirs(args.out, exist_ok=True)
    start, end = parse_range(args.range)
    samples = []   # (phase, smi, {offset: value})
    load = None
    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            phase = "load" if load else "idle"
            if not load and args.load and time.time() - t0 > args.seconds / 2:
                load = subprocess.Popen(args.load, shell=True)
                phase = "load"
            smi = smi_sample()
            regs = read_range(args.bdf, args.bar, start, end, args.stride)
            samples.append((phase, smi, regs))
            time.sleep(args.interval)
    finally:
        if load:
            load.terminate()
            load.wait()
    with open(os.path.join(args.out, "correlate-samples.json"), "w") as f:
        json.dump([{"phase": p, "smi": s, "regs": {hex(k): v for k, v in r.items()}}
                   for p, s, r in samples], f)
    # Rank the offsets whose values change, by how closely each follows a
    # reading, both as a raw word and as its low 16 bits.
    offsets = sorted(samples[0][2])
    report = []
    for off in offsets:
        series = [s[2].get(off, 0) for s in samples]
        if len(set(series)) < 2:
            continue
        for field in SMI_FIELDS:
            ys = [s[1][field] for s in samples]
            for label, xs in (("word", series), ("low16", [v & 0xFFFF for v in series])):
                r = pearson([float(x) for x in xs], ys)
                if abs(r) >= args.threshold:
                    report.append((abs(r), hex(off), field, label, round(r, 3)))
    report.sort(reverse=True)
    with open(os.path.join(args.out, "correlate-report.tsv"), "w") as f:
        f.write("offset\treading\tas\tr\n")
        for _, off, field, label, r in report:
            f.write("%s\t%s\t%s\t%s\n" % (off, field, label, r))
    print("%d samples; %d candidate (offset, reading) pairs at |r| >= %.2f; top:" %
          (len(samples), len(report), args.threshold))
    for _, off, field, label, r in report[:15]:
        print("  %-10s %-18s %-6s r=%+.3f" % (off, field, label, r))


def cmd_compare(args):
    cfg_path = None
    for root, _, files in os.walk(args.capture):
        if "config.bin" in files:
            cfg_path = os.path.join(root, "config.bin")
            break
    if not cfg_path:
        raise SystemExit("no config.bin under " + args.capture)
    real = open(cfg_path, "rb").read()
    env = dict(os.environ, VGPU_GPU=args.profile, VGPU_DEVICE_COUNT="1", VGPU_QUIET="1")
    dump = subprocess.run([args.vgpu, "smi", "--lspci-dump"], stdout=subprocess.PIPE, env=env,
                          universal_newlines=True, check=True).stdout
    # The first device's hex rows: "00: 10 de ...", sixteen bytes each.
    model = bytearray()
    for line in dump.splitlines():
        if not line.strip():
            if model:
                break
            continue
        m = re.match(r"^[0-9a-f]+:((?: [0-9a-f]{2}){16})$", line)
        if m:
            model += bytes(int(b, 16) for b in m.group(1).split())
    listing = subprocess.run([args.vgpu, "regs", "list"], stdout=subprocess.PIPE, env=env,
                             universal_newlines=True, check=True).stdout.splitlines()[1:]
    n = min(len(real), len(model))
    print("comparing %d bytes (%d captured, %d modelled)" % (n, len(real), len(model)))
    diffs = 0
    for row in listing:
        cols = row.split()
        off, width, name = int(cols[0], 16), int(cols[1]), cols[3]
        size = width // 8
        if off + size > n:
            continue
        r = int.from_bytes(real[off:off + size], "little")
        m = int.from_bytes(model[off:off + size], "little")
        if r != m:
            diffs += 1
            print("  0x%03x %-28s real 0x%0*x  model 0x%0*x" % (off, name, size * 2, r, size * 2, m))
    print("%d of the registers in reach differ" % diffs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("out")
    c.add_argument("--bdf")
    m = sub.add_parser("mmio")
    m.add_argument("out")
    m.add_argument("--bdf", required=True)
    m.add_argument("--range", required=True)
    m.add_argument("--bar", type=int, default=0)
    m.add_argument("--stride", type=int, default=4)
    r = sub.add_parser("correlate")
    r.add_argument("out")
    r.add_argument("--bdf", required=True)
    r.add_argument("--range", required=True)
    r.add_argument("--bar", type=int, default=0)
    r.add_argument("--stride", type=int, default=4)
    r.add_argument("--seconds", type=float, default=60)
    r.add_argument("--interval", type=float, default=1.0)
    r.add_argument("--load", help="a command that loads the GPU, started halfway through")
    r.add_argument("--threshold", type=float, default=0.8)
    p = sub.add_parser("compare")
    p.add_argument("capture")
    p.add_argument("--profile", required=True)
    p.add_argument("--vgpu", default="vgpu")
    args = ap.parse_args()
    {"capture": cmd_capture, "mmio": cmd_mmio, "correlate": cmd_correlate, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
