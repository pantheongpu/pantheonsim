"""torch.distributed across two processes, each its own simulated NVIDIA GPU.

Run twice at once (nvidia/tests/e2e/run_pytorch_distributed.sh), as torchrun
would: RANK and WORLD_SIZE say which. PyTorch's NCCL backend runs on
VirtualGPU's libnccl, whose ranks in separate processes meet through its
file-backed transport. The collectives first, then a small network trained the
ways large models are split over GPUs -- DistributedDataParallel, tensor
parallel, pipeline parallel, FullyShardedDataParallel -- each of which must end
where one process training on the whole batch (on the CPU) ends. One line a
check from each rank: "ok <name>", or "FAIL <name>: <why>".
"""
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dev = torch.device("cuda", rank)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)

LR, STEPS, PER_RANK = 0.1, 3, 8
IN, HID, OUT = 16, 32, 4


def check(name, f):
    try:
        why = f()
        print(f"ok rank {rank}: {name}" if why is None else f"FAIL rank {rank}: {name}: {why}", flush=True)
    except Exception as e:
        print(f"FAIL rank {rank}: {name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)


# ---- the collectives ----------------------------------------------------------

def all_reduce():
    x = torch.full((4096,), float(rank + 1), device=dev)
    dist.all_reduce(x)
    want = float(sum(range(1, world + 1)))
    return None if bool((x == want).all()) else f"got {x[0].item()}, want {want}"


def all_gather():
    y = torch.arange(8, device=dev, dtype=torch.float32) * (rank + 1)
    out = [torch.empty_like(y) for _ in range(world)]
    dist.all_gather(out, y)
    ok = all(torch.equal(o.cpu(), torch.arange(8.) * (i + 1)) for i, o in enumerate(out))
    return None if ok else "a rank's piece differs"


def broadcast():
    z = torch.full((1000,), 7.0 if rank == 0 else 0.0, device=dev)
    dist.broadcast(z, src=0)
    return None if bool((z == 7).all()) else "not what rank 0 sent"


def reduce_scatter():
    inp = torch.arange(world * 4, dtype=torch.float32, device=dev) + rank
    out = torch.empty(4, device=dev)
    dist.reduce_scatter_tensor(out, inp)
    want = sum(torch.arange(world * 4, dtype=torch.float32)[rank * 4:(rank + 1) * 4] + r for r in range(world))
    return None if torch.equal(out.cpu(), want) else f"got {out.tolist()}, want {want.tolist()}"


def all_to_all():
    inp = torch.arange(world * 2, dtype=torch.float32, device=dev) + 100 * rank
    out = torch.empty_like(inp)
    dist.all_to_all_single(out, inp)
    want = torch.cat([torch.arange(rank * 2, rank * 2 + 2, dtype=torch.float32) + 100 * r for r in range(world)])
    return None if torch.equal(out.cpu(), want) else f"got {out.tolist()}, want {want.tolist()}"


def send_recv():
    t = torch.full((64,), float(rank + 5), device=dev)
    peer = (rank + 1) % world
    prev = (rank - 1) % world
    got = torch.empty_like(t)
    ops = [dist.P2POp(dist.isend, t, peer), dist.P2POp(dist.irecv, got, prev)]
    for req in dist.batch_isend_irecv(ops):
        req.wait()
    return None if bool((got == prev + 5).all()) else f"got {got[0].item()}, want {prev + 5}"


# ---- a network, trained the ways large models are split -----------------------

def data():
    g = torch.Generator().manual_seed(1)
    return torch.randn(PER_RANK * world, IN, generator=g), torch.randn(PER_RANK * world, OUT, generator=g)


def mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(IN, HID), nn.ReLU(), nn.Linear(HID, OUT))


def reference():
    """One process, the CPU, the whole batch: the weights to end at."""
    model = mlp()
    x, y = data()
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    for _ in range(STEPS):
        opt.zero_grad()
        F.mse_loss(model(x), y).backward()
        opt.step()
    return [p.detach().clone() for p in model.parameters()]


def worst(got, want):
    return max(float((g.detach().cpu() - w).abs().max()) for g, w in zip(got, want))


def ddp():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    batch = torch.randn(8 * world, 16)
    # The CPU's gradients for the whole batch, the mean over it.
    model(batch).pow(2).mean().backward()
    want = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    ddp_model = nn.parallel.DistributedDataParallel(model.to(dev), device_ids=[rank])
    ddp_model(batch[8 * rank:8 * (rank + 1)].to(dev)).pow(2).mean().backward()
    diff = worst([p.grad for p in ddp_model.parameters()], want)
    return None if diff <= 1e-5 else f"max gradient difference {diff:g}"


def ddp_training():
    want = reference()
    x, y = data()
    model = nn.parallel.DistributedDataParallel(mlp().to(dev), device_ids=[rank])
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    xs, ys = x[PER_RANK * rank:PER_RANK * (rank + 1)].to(dev), y[PER_RANK * rank:PER_RANK * (rank + 1)].to(dev)
    for _ in range(STEPS):
        opt.zero_grad()
        F.mse_loss(model(xs), ys).backward()
        opt.step()
    diff = worst(model.parameters(), want)
    return None if diff <= 1e-4 else f"max weight difference from one process on the whole batch {diff:g}"


class _SumAcrossRanks(torch.autograd.Function):
    """All-reduce forward; the gradient of a replicated sum goes back unchanged."""

    @staticmethod
    def forward(ctx, t):
        t = t.clone()
        dist.all_reduce(t)
        return t

    @staticmethod
    def backward(ctx, g):
        return g


def tensor_parallel():
    want = reference()
    x, y = (t.to(dev) for t in data())
    full = mlp()
    s = HID // world
    sl = slice(rank * s, (rank + 1) * s)
    # Hidden layer split by output features, output layer by input features.
    w1 = full[0].weight.detach()[sl].clone().to(dev).requires_grad_()
    b1 = full[0].bias.detach()[sl].clone().to(dev).requires_grad_()
    w2 = full[2].weight.detach()[:, sl].clone().to(dev).requires_grad_()
    b2 = full[2].bias.detach().clone().to(dev).requires_grad_()
    params = [w1, b1, w2, b2]
    for _ in range(STEPS):
        out = _SumAcrossRanks.apply(F.relu(x @ w1.t() + b1) @ w2.t()) + b2
        F.mse_loss(out, y).backward()
        with torch.no_grad():
            for p in params:
                p -= LR * p.grad
                p.grad = None
    diff = max(float((w1.cpu() - want[0][sl]).abs().max()), float((b1.cpu() - want[1][sl]).abs().max()),
               float((w2.cpu() - want[2][:, sl]).abs().max()), float((b2.cpu() - want[3]).abs().max()))
    return None if diff <= 1e-4 else f"max weight difference from the unsplit network {diff:g}"


def pipeline_parallel():
    if world != 2:
        return None
    want = reference()
    x, y = (t.to(dev) for t in data())
    full = mlp()
    if rank == 0:   # stage 0: the hidden layer
        w, b = full[0].weight.detach().to(dev).requires_grad_(), full[0].bias.detach().to(dev).requires_grad_()
        for _ in range(STEPS):
            h = F.relu(x @ w.t() + b)
            dist.send(h.detach().contiguous(), dst=1)
            g = torch.empty_like(h)
            dist.recv(g, src=1)
            h.backward(g)
            with torch.no_grad():
                for p in (w, b):
                    p -= LR * p.grad
                    p.grad = None
        diff = max(float((w.cpu() - want[0]).abs().max()), float((b.cpu() - want[1]).abs().max()))
    else:           # stage 1: the output layer and the loss
        w, b = full[2].weight.detach().to(dev).requires_grad_(), full[2].bias.detach().to(dev).requires_grad_()
        for _ in range(STEPS):
            h = torch.empty(x.shape[0], HID, device=dev)
            dist.recv(h, src=0)
            h.requires_grad_()
            F.mse_loss(h @ w.t() + b, y).backward()
            dist.send(h.grad.contiguous(), dst=0)
            with torch.no_grad():
                for p in (w, b):
                    p -= LR * p.grad
                    p.grad = None
        diff = max(float((w.cpu() - want[2]).abs().max()), float((b.cpu() - want[3]).abs().max()))
    return None if diff <= 1e-4 else f"max weight difference from the unsplit network {diff:g}"


def fsdp():
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    want = reference()
    x, y = data()
    model = FSDP(mlp().to(dev), device_id=dev, use_orig_params=True)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    xs, ys = x[PER_RANK * rank:PER_RANK * (rank + 1)].to(dev), y[PER_RANK * rank:PER_RANK * (rank + 1)].to(dev)
    for _ in range(STEPS):
        opt.zero_grad()
        F.mse_loss(model(xs), ys).backward()
        opt.step()
    with FSDP.summon_full_params(model):
        diff = worst(list(model.parameters()), want)
    return None if diff <= 1e-4 else f"max weight difference from one process on the whole batch {diff:g}"


CHECKS = [("all_reduce", all_reduce), ("all_gather", all_gather), ("broadcast", broadcast),
          ("reduce_scatter", reduce_scatter), ("all_to_all", all_to_all), ("send/recv", send_recv),
          ("DistributedDataParallel's averaged gradients", ddp),
          ("DistributedDataParallel training matches one process on the whole batch", ddp_training),
          ("tensor-parallel training matches the unsplit network", tensor_parallel),
          ("pipeline-parallel training matches the unsplit network", pipeline_parallel),
          ("FullyShardedDataParallel training matches one process on the whole batch", fsdp)]
for name, f in CHECKS:
    check(name, f)
    dist.barrier()
dist.destroy_process_group()
