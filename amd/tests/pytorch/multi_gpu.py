"""PyTorch's ROCm build, unmodified, on two simulated MI300Xs.

RCCL's collectives between the two devices, called as torch.cuda.nccl calls
them, and DataParallel training a small model across both, each checked
against what the CPU computes. They pass only if the two devices' kernels run
at the same time: a collective's kernels wait on each other's writes. One line
each: "ok <name>", or "FAIL <name>: <why>".
"""
import torch
import torch.cuda.nccl as nccl
import torch.nn as nn

N = 2


def check(name, f):
    try:
        why = f()
        print(f"ok {name}" if why is None else f"FAIL {name}: {why}", flush=True)
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)


def same(got, want):
    got = [g.cpu() for g in got]
    bad = [i for i, g in enumerate(got) if not torch.equal(g, want[i] if isinstance(want, list) else want)]
    return None if not bad else f"device {bad[0]} differs"


def all_reduce(dtype, size):
    def run():
        xs = [torch.arange(size, device=f"cuda:{i}").to(dtype) * (i + 1) for i in range(N)]
        nccl.all_reduce(xs)
        return same(xs, torch.arange(size).to(dtype) * 3)
    return run


def broadcast():
    xs = [torch.full((5000,), 7.0 if i == 0 else 0.0, device=f"cuda:{i}") for i in range(N)]
    nccl.broadcast(xs, root=0)
    return same(xs, torch.full((5000,), 7.0))


def reduce():
    xs = [torch.full((3000,), float(i + 2), device=f"cuda:{i}") for i in range(N)]
    out = torch.empty(3000, device="cuda:1")
    nccl.reduce(xs, output=out, root=1)
    return same([out], torch.full((3000,), 5.0))


def all_gather():
    xs = [torch.full((100,), float(i), device=f"cuda:{i}") for i in range(N)]
    outs = [torch.empty(100 * N, device=f"cuda:{i}") for i in range(N)]
    nccl.all_gather(xs, outs)
    return same(outs, torch.cat([torch.full((100,), float(i)) for i in range(N)]))


def reduce_scatter():
    xs = [torch.arange(100 * N, device=f"cuda:{i}").float() for i in range(N)]
    outs = [torch.empty(100, device=f"cuda:{i}") for i in range(N)]
    nccl.reduce_scatter(xs, outs)
    want = torch.arange(100 * N).float() * N
    return same(outs, [want[i * 100:(i + 1) * 100] for i in range(N)])


def data_parallel():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 8))
    x = torch.randn(16, 32)
    want = model(x)
    want.sum().backward()
    want_grad = model[0].weight.grad.clone()
    model.zero_grad()
    parallel = nn.DataParallel(model.cuda(0), device_ids=list(range(N)))
    got = parallel(x.cuda(0))
    got.sum().backward()
    diff = max(float((got.cpu() - want).abs().max()), float((model[0].weight.grad.cpu() - want_grad).abs().max()))
    return None if diff <= 1e-4 else f"max difference {diff:g}"


if torch.cuda.device_count() < N:
    print(f"FAIL devices: {torch.cuda.device_count()} of {N}", flush=True)
check("all_reduce float 1M", all_reduce(torch.float32, 1 << 20))
check("all_reduce bfloat16", all_reduce(torch.bfloat16, 1 << 16))
check("all_reduce half", all_reduce(torch.float16, 1 << 16))
check("all_reduce int32", all_reduce(torch.int32, 4096))
check("broadcast", broadcast)
check("reduce", reduce)
check("all_gather", all_gather)
check("reduce_scatter", reduce_scatter)
check("DataParallel forward and backward", data_parallel)
