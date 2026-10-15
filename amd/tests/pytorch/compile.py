"""PyTorch's compiler and Triton, unmodified, on a simulated AMD GPU.

torch.compile turns a function into Triton kernels that Triton compiles for
the device and loads through HIP at run time; a Triton kernel written by
hand goes the same way. Each is checked against the CPU: reductions, scans,
sorts, random numbers, dynamic shapes, a training step's forward and
backward, and the "reduce-overhead" mode that records the compiled kernels
into a HIP graph and replays it (what vLLM does to its model). One line each:
"ok <name>", or "FAIL <name>: <why>".
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

torch.manual_seed(0)


def check(name, got, want, tol):
    try:
        diff = float((got.detach().float().cpu() - want.detach().float()).abs().max())
        print(f"ok {name}" if diff <= tol else f"FAIL {name}: max difference {diff:g}, allowed {tol:g}", flush=True)
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)


def fused(x, y):
    return torch.relu(x @ y + 1.0).sum(dim=1) * torch.sigmoid(x[:, :4].sum())


x, y = torch.randn(64, 32), torch.randn(32, 48)
try:
    check('torch.compile of a matmul, reduction and pointwise', torch.compile(fused)(x.cuda(), y.cuda()),
          fused(x, y), 1e-3)
except Exception as e:
    print(f"FAIL torch.compile of a matmul, reduction and pointwise: {type(e).__name__}: "
          f"{str(e).splitlines()[0][:200]}", flush=True)


def softmax_rows(t):
    return torch.softmax(t * 0.5, dim=-1) + t.mean()


t = torch.randn(33, 100)
try:
    check('torch.compile of a softmax, rows not a power of two', torch.compile(softmax_rows)(t.cuda()),
          softmax_rows(t), 1e-5)
except Exception as e:
    print(f"FAIL torch.compile of a softmax, rows not a power of two: {type(e).__name__}: "
          f"{str(e).splitlines()[0][:200]}", flush=True)


@triton.jit
def add(a, b, c, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(c + i, tl.load(a + i, mask=m) + tl.load(b + i, mask=m), mask=m)


a, b = torch.randn(1000), torch.randn(1000)
c = torch.empty(1000, device='cuda')
try:
    add[(8,)](a.cuda(), b.cuda(), c, 1000, BLOCK=128)
    check('a Triton kernel written by hand', c, a + b, 0.0)
except Exception as e:
    print(f"FAIL a Triton kernel written by hand: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)


def run(name, f):
    try:
        why = f()
        print(f"ok {name}" if why is None else f"FAIL {name}: {why}", flush=True)
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)


def diff(got, want, tol):
    d = float((got.detach().float().cpu() - want.detach().float()).abs().max())
    return None if d <= tol else f"max difference {d:g}, allowed {tol:g}"


def reduce_overhead():
    # Dynamo, Inductor and cudagraph trees: warm-up runs, then the kernels
    # recorded into a graph, which later calls replay with the inputs copied
    # in. The inputs change between calls, so a stale replay would be seen.
    def f(x, w):
        return torch.relu(x @ w).sum(1)
    g = torch.compile(f, mode="reduce-overhead")
    for i in range(4):
        x, w = torch.randn(32, 32), torch.randn(32, 16)
        if (why := diff(g(x.cuda(), w.cuda()), f(x, w), 1e-3)):
            return f"call {i}: {why}"


def scan_and_sort():
    def f(t):
        return torch.cumsum(t, -1) + torch.cumprod(torch.sigmoid(t), -1) + torch.sort(t, -1)[0]
    t = torch.randn(16, 300)
    return diff(torch.compile(f)(t.cuda()), f(t), 1e-3)


def dynamic_shapes():
    f = torch.compile(lambda t: torch.softmax(t, -1) * t.shape[0], dynamic=True)
    for n in (5, 9, 33):
        t = torch.randn(n, 70)
        if (why := diff(f(t.cuda()), torch.softmax(t, -1) * n, 1e-5)):
            return f"{n} rows: {why}"


def random_numbers():
    f = torch.compile(lambda t: F.dropout(t, 0.5, True) + torch.rand_like(t))
    t = torch.ones(4096, device='cuda')
    a, b = f(t), f(t)
    if torch.equal(a, b):
        return "two calls drew the same numbers"
    return None if float(a.min()) >= 0 and float(a.max()) <= 3 else "values outside [0, 3]"


def training_step():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.LayerNorm(64), torch.nn.Linear(64, 8))
    x, y = torch.randn(16, 32), torch.randint(0, 8, (16,))
    want = m(x)
    F.cross_entropy(want, y).backward()
    grads = [p.grad.clone() for p in m.parameters()]
    for p in m.parameters():
        p.grad = None
    out = torch.compile(m.cuda())(x.cuda())
    F.cross_entropy(out, y.cuda()).backward()
    for p, g in zip(m.parameters(), grads):
        if (why := diff(p.grad, g, 2e-3)):
            return f"gradient: {why}"
    return diff(out, want, 1e-3)


def nan_propagation():
    # Inductor's relu, maximum, minimum and clamp keep a NaN (what torch's CPU does), each way
    # its Triton lowers them (compare and select, or an IEEE minimum or maximum instruction).
    # The CPU's answers, with NaNs and signed zeros in the inputs, are the reference.
    def f(x, y):
        return torch.relu(x), torch.maximum(x, y), torch.minimum(x, y), torch.clamp(y, -0.5, 0.5)
    x = torch.randn(257)
    y = torch.randn(257)
    x[::7], y[1::7] = float('nan'), float('nan')
    x[2::11], y[2::11] = 0.0, -0.0
    for dtype in (torch.float32, torch.float16):
        a, b = x.to(dtype), y.to(dtype)
        got = torch.compile(f)(a.cuda(), b.cuda())
        for g, w in zip(got, f(a, b)):
            g = g.cpu()
            if not torch.equal(torch.isnan(g), torch.isnan(w)):
                return f"{dtype}: NaNs in different places"
            if not torch.equal(torch.nan_to_num(g, nan=7.0), torch.nan_to_num(w, nan=7.0)):
                return f"{dtype}: values differ"


run('torch.compile with mode="reduce-overhead" (compiled kernels in a HIP graph)', reduce_overhead)
run('torch.compile of a cumulative sum, product and sort', scan_and_sort)
run('torch.compile with dynamic shapes', dynamic_shapes)
run('torch.compile of dropout and rand_like draws new numbers per call', random_numbers)
run('torch.compile of a training step, forward and backward', training_step)
run('torch.compile of relu, maximum, minimum and clamp with NaNs in', nan_propagation)
