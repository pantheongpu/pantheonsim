"""RCCL collectives recorded into HIP graphs, across two processes.

vLLM with tensor parallelism captures its decode step with the all-reduces of
each layer inside it, and replays that per token. Run twice at once
(amd/tests/e2e/run_pytorch_distributed.sh distributed_graphs.py 6), as
torchrun would: RANK and WORLD_SIZE say which; each process is its own
simulated MI300X and RCCL's kernels reach the other's memory through HIP's IPC
handles. Each check captures, then changes its inputs and replays, so a
replay that only repeated the captured values would fail. One line a check
from each rank: "ok <name>", or "FAIL <name>: <why>".
"""
import os

import torch
import torch.distributed as dist

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
total = float(sum(range(1, world + 1)))


def check(name, f):
    try:
        why = f()
        print(f"ok rank {rank}: {name}" if why is None else f"FAIL rank {rank}: {name}: {why}", flush=True)
    except Exception as e:
        print(f"FAIL rank {rank}: {name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)


def capture(fn):
    fn()  # the communicator is set up outside the capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


def all_reduce():
    x = torch.zeros(4096, device="cuda")

    def f():
        dist.all_reduce(x)
        return x * 2
    g, y = capture(f)
    for scale in (1.0, 10.0):
        x.fill_((rank + 1) * scale)
        g.replay()
        torch.cuda.synchronize()
        if not bool((y == total * scale * 2).all()):
            return f"scale {scale}: got {y[0].item()}, want {total * scale * 2}"


def all_gather():
    x = torch.arange(8, device="cuda", dtype=torch.float32)
    outs = [torch.empty_like(x) for _ in range(world)]
    g, _ = capture(lambda: dist.all_gather(outs, x))
    for scale in (1.0, 2.0):
        x.copy_(torch.arange(8.) * (rank + 1) * scale)
        g.replay()
        torch.cuda.synchronize()
        if not all(torch.equal(o.cpu(), torch.arange(8.) * (i + 1) * scale) for i, o in enumerate(outs)):
            return f"scale {scale}: a rank's piece differs"


def two_layers():
    # A tensor-parallel block: a matmul on this rank's share of the weights,
    # an all-reduce of the partial sums, twice; the result is the full-size
    # product, as on one device.
    torch.manual_seed(7)
    w1, w2 = torch.randn(16, 32), torch.randn(32, 16)
    x = torch.zeros(4, 16, device="cuda")
    half = 32 // world
    a = w1[:, rank * half:(rank + 1) * half].cuda()
    b = w2[rank * half:(rank + 1) * half].cuda()

    def f():
        h = torch.relu(x @ a) @ b  # relu of this rank's columns, then the partial sum
        dist.all_reduce(h)
        h = h + x
        h = torch.relu(h @ a) @ b
        dist.all_reduce(h)
        return h
    g, y = capture(f)
    for seed in (1, 2):
        v = torch.randn(4, 16, generator=torch.Generator().manual_seed(seed))
        x.copy_(v)
        g.replay()
        torch.cuda.synchronize()
        h = v + torch.relu(v @ w1) @ w2
        want = torch.relu(h @ w1) @ w2
        d = float((y.cpu() - want).abs().max())
        if d > 1e-3:
            return f"seed {seed}: max difference {d:g}"


check("all_reduce recorded in a graph and replayed", all_reduce)
check("all_gather recorded in a graph and replayed", all_gather)
check("a tensor-parallel block, two all-reduces in one graph", two_layers)
dist.destroy_process_group()
