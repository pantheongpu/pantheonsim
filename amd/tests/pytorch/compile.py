"""PyTorch's compiler and Triton, unmodified, on a simulated AMD GPU.

torch.compile turns a function into Triton kernels that Triton compiles for
the device and loads through HIP at run time; a Triton kernel written by
hand goes the same way. Each is checked against the CPU. One line each:
"ok <name>", or "FAIL <name>: <why>".
"""
import torch
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
