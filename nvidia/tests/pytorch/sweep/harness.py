"""The sweep's harness: a registry of checks, each run on the simulated GPU and
on the CPU from the same inputs and compared; one line each, "ok <name>" or
"FAIL <name>: <why>", and "skip <name>: <why>" for a check whose optional
package is not installed (those are not counted).

known_failures.txt lists the checks that are known not to match the CPU yet
(name, a tab, a one-line reason). Such a check prints "XFAIL <name>: <why>
-- known: <reason>" when it fails, with the numbers, and the run does not
fail for it; when it starts passing it prints "XPASS <name>" and the run
does fail, so the entry is removed and the list stays honest. A reason that
starts with "[sim-error]" says the failure also prints a "VirtualGPU error ["
line, which run_pytorch_sweep.sh tolerates only then.

Selection (environment): VGPU_SWEEP_GROUPS (comma list, default every group),
VGPU_SWEEP_TIER (quick: the time-bounded subset; full: everything),
VGPU_SWEEP_ONLY (a regular expression on the check's name),
VGPU_SWEEP_TIMEOUT (seconds a check may take before it is called a hang),
VGPU_SWEEP_DEPS (directories added to sys.path, for optional packages).
"""
import importlib.util
import os
import re
import sys
import threading
import time
import traceback

import torch

for _p in filter(None, os.environ.get('VGPU_SWEEP_DEPS', '').split(':')):
    sys.path.insert(0, _p)

GPU = 'cuda'
# A card's TF32 convolutions are not what the simulator computes (it rounds like fp32);
# compare in fp32 so a difference means a bug, not the precision mode.
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
KNOWN = {}  # check name -> reason, from known_failures.txt
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'known_failures.txt')) as _f:
        for _line in _f:
            if _line.strip() and not _line.startswith('#'):
                _name, _, _reason = _line.rstrip('\n').partition('\t')
                KNOWN[_name] = _reason.strip()
except OSError:
    pass
REGISTRY = []  # dicts: group, name, fn, tol, tier, needs


def have(mod):
    if mod == 'cuda2':  # a second device, as VGPU_DEVICE_COUNT=2 gives
        return torch.cuda.device_count() >= 2
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def check(group, name, tol=1e-4, tier='quick', needs=()):
    """Registers fn(device) -> tensor, or a list/tuple/dict of them, as a check."""
    def deco(fn):
        REGISTRY.append(dict(group=group, name=name, fn=fn, tol=tol, tier=tier, needs=tuple(needs)))
        return fn
    return deco


def cases(group, name, table, tol=1e-4, tier='quick', needs=()):
    """One check made of named sub-cases (each fn(device)), compared one by one:
    it fails naming every sub-case that differs or raises."""
    def run(d):
        return {k: f(d) for k, f in table.items()}
    REGISTRY.append(dict(group=group, name=name, fn=run, tol=tol, tier=tier, needs=tuple(needs), table=table))


# ---- inputs: always built on the CPU from a seed, so both devices see the same

def rnd(*shape, seed=1, dtype=torch.float32, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).to(dtype)


def rint(low, high, *shape, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(low, high, shape, generator=g)


def flat(x, out=None):
    out = [] if out is None else out
    if isinstance(x, torch.Tensor):
        out.append(x)
    elif isinstance(x, dict):
        for k in sorted(x, key=str):
            flat(x[k], out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            flat(v, out)
    elif x is None:
        pass
    else:
        out.append(torch.tensor(x))
    return out


def lowp(x, dtype):
    """x rounded through dtype and back: what the GPU sees, held in float32."""
    return x.to(dtype).float() if x.is_floating_point() else x


def diff(g, c):
    """Largest scaled difference between two tensors; inf for a shape mismatch."""
    g, c = g.detach().cpu(), c.detach()
    if g.shape != c.shape:
        return float('inf'), f"shape {tuple(g.shape)} against the CPU's {tuple(c.shape)}"
    if g.is_complex() or c.is_complex():
        g, c = torch.view_as_real(g.to(torch.complex128)), torch.view_as_real(c.to(torch.complex128))
    if g.dtype == torch.bool or not g.is_floating_point() or not c.is_floating_point():
        g, c = g.to(torch.float64), c.to(torch.float64)
    else:
        g, c = g.double(), c.double()
    nan_g, nan_c = torch.isnan(g), torch.isnan(c)
    if not torch.equal(nan_g, nan_c):
        return float('inf'), 'NaNs where the CPU has none (or the reverse)'
    inf_g, inf_c = torch.isinf(g), torch.isinf(c)
    if not torch.equal(inf_g, inf_c) or not torch.equal(g[inf_g], c[inf_c]):
        return float('inf'), 'infinities differ'
    ok = ~(nan_g | inf_g)
    if not bool(ok.any()):
        return 0.0, ''
    g, c = g[ok], c[ok]
    return float((g - c).abs().max()) / max(1.0, float(c.abs().max())), ''


def compare(g, c, tol):
    gs, cs = flat(g), flat(c)
    if len(gs) != len(cs):
        return f"{len(gs)} tensors against the CPU's {len(cs)}"
    worst = 0.0
    for i, (a, b) in enumerate(zip(gs, cs)):
        d, why = diff(a, b)
        if why:
            return f"tensor {i}: {why}"
        worst = max(worst, d)
    return None if worst <= tol else f"max scaled difference {worst:g}, allowed {tol:g}"


def short(e):
    return f"{type(e).__name__}: {str(e).splitlines()[0][:220] if str(e) else ''}"


def run_one(c, timeout):
    name = c['name']
    print(f"# start {name}", flush=True)
    t0 = time.time()
    stop = threading.Event()

    def watchdog():
        if not stop.wait(timeout):
            print(f"FAIL {name}: hang, no result after {timeout:g} s", flush=True)
            os._exit(3)
    threading.Thread(target=watchdog, daemon=True).start()
    why = None
    try:
        if 'table' in c:
            bad = []
            for k, f in c['table'].items():
                try:
                    w = compare(f(GPU), f('cpu'), c['tol'])
                except Exception as e:
                    w = short(e)
                if w:
                    bad.append(f"{k} ({w})")
            why = '; '.join(bad) if bad else None
        else:
            torch.manual_seed(0)
            g = c['fn'](GPU)
            torch.manual_seed(0)
            why = compare(g, c['fn']('cpu'), c['tol'])
    except Exception as e:
        why = short(e)
        if os.environ.get('VGPU_SWEEP_TRACE'):
            traceback.print_exc()
    stop.set()
    secs = time.time() - t0
    known = KNOWN.get(name)
    if known is not None:
        if why is None:
            print(f"XPASS {name}: it matches the CPU now; remove it from sweep/known_failures.txt  ({secs:.1f} s)", flush=True)
            return 'xpass'
        print(f"XFAIL {name}: {why}  -- known: {known}  ({secs:.1f} s)", flush=True)
        return 'xfail'
    print(f"ok {name}  ({secs:.1f} s)" if why is None else f"FAIL {name}: {why}  ({secs:.1f} s)", flush=True)
    return 'ok' if why is None else 'fail'


def selected():
    groups = [g for g in os.environ.get('VGPU_SWEEP_GROUPS', '').split(',') if g]
    tier = os.environ.get('VGPU_SWEEP_TIER', 'quick')
    only = re.compile(os.environ['VGPU_SWEEP_ONLY']) if os.environ.get('VGPU_SWEEP_ONLY') else None
    out, skipped = [], []
    for c in REGISTRY:
        if groups and c['group'] not in groups:
            continue
        if tier == 'quick' and c['tier'] != 'quick':
            continue
        if only and not only.search(c['name']):
            continue
        missing = [m for m in c['needs'] if not have(m)]
        (skipped if missing else out).append((c, missing))
    return [c for c, _ in out], skipped


def main(argv, loaders):
    """loaders: {group: module name}; imports the chosen groups, lists or runs."""
    chosen = [g for g in os.environ.get('VGPU_SWEEP_GROUPS', '').split(',') if g] or list(loaders)
    for g in chosen:
        importlib.import_module(loaders[g])
    todo, skipped = selected()
    if '--list' in argv:
        for c in todo:
            print(f"{c['group']}\t{c['name']}")
        return 0
    for c, missing in skipped:
        print(f"skip {c['name']}: needs {', '.join(missing)}", flush=True)
    timeout = float(os.environ.get('VGPU_SWEEP_TIMEOUT', '900'))
    results = [run_one(c, timeout) for c in todo]
    print(f"# {results.count('ok')} of {len(todo)} checks ok, {results.count('xfail')} known failures", flush=True)
    return 0


def fwd_bwd(ctor, inputs, d, dtype=None, train=True, seed=0, params=True):
    """Builds ctor() on the CPU (same weights for both devices), runs it forward
    on `inputs` and backward through a fixed random weighting of the outputs;
    returns [outputs, input gradients, parameter gradients, floating buffers].
    With dtype set (half, bfloat16) the GPU runs in it and the CPU in float32
    from the same rounded values."""
    torch.manual_seed(seed)
    m = ctor()
    low = dtype is not None
    if low and d == 'cpu':
        for p in m.parameters():
            p.data = lowp(p.data, dtype)
        for b in m.buffers():
            b.data = lowp(b.data, dtype)
    m = m.to(d, dtype) if (low and d != 'cpu') else m.to(d)
    m.train(train)
    xs = []
    for x in inputs:
        if x.is_floating_point():
            x = (lowp(x, dtype) if (low and d == 'cpu') else x).detach().clone().to(d, dtype if (low and d != 'cpu') else x.dtype)
            x.requires_grad_()
        else:
            x = x.to(d)
        xs.append(x)
    out = m(*xs)
    outs = [t for t in flat(out) if t.is_floating_point()]
    loss = sum((o.float() * rnd(*o.shape, seed=100 + i).to(d)).sum() for i, o in enumerate(outs))
    loss.backward()
    res = [outs, [x.grad for x in xs if x.is_floating_point()]]
    if params:
        res.append([p.grad for p in m.parameters() if p.grad is not None])
    res.append([b for b in m.buffers() if b.is_floating_point()])
    return res


def layer(group, name, ctor, shapes, tol=1e-4, dtype=None, train=True, tier='quick', needs=()):
    """A check: ctor() forward and backward on random inputs of the given shapes
    (an int tuple is a float input; ('i', hi, shape...) an integer one)."""
    def make():
        xs = []
        for i, s in enumerate(shapes):
            xs.append(rint(s[1], s[2], *s[3:], seed=10 + i) if s and s[0] == 'i' else rnd(*s, seed=10 + i))
        return xs

    def run(d):
        return fwd_bwd(ctor, make(), d, dtype=dtype, train=train)
    REGISTRY.append(dict(group=group, name=name, fn=run, tol=tol, tier=tier, needs=tuple(needs)))
