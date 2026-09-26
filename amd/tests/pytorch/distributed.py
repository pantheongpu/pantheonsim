"""torch.distributed across two processes, each its own simulated MI300X.

Run twice at once (amd/tests/e2e/run_pytorch_distributed.sh), as torchrun
would: RANK and WORLD_SIZE say which. The NCCL backend is RCCL, whose
processes reach each other's device memory through HIP's IPC handles --
each process's simulator maps the other's memory from a shared file. Checks
the collectives, and DistributedDataParallel training, whose averaged
gradients must be the gradients of the whole batch on the CPU. One line a
check from each rank: "ok <name>", or "FAIL <name>: <why>".
"""
import os

import torch
import torch.distributed as dist
import torch.nn as nn

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))


def check(name, f):
    try:
        why = f()
        print(f"ok rank {rank}: {name}" if why is None else f"FAIL rank {rank}: {name}: {why}", flush=True)
    except Exception as e:
        print(f"FAIL rank {rank}: {name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)


def all_reduce():
    x = torch.full((4096,), float(rank + 1), device="cuda")
    dist.all_reduce(x)
    want = float(sum(range(1, world + 1)))
    return None if bool((x == want).all()) else f"got {x[0].item()}, want {want}"


def all_gather():
    y = torch.arange(8, device="cuda", dtype=torch.float32) * (rank + 1)
    out = [torch.empty_like(y) for _ in range(world)]
    dist.all_gather(out, y)
    ok = all(torch.equal(o.cpu(), torch.arange(8.) * (i + 1)) for i, o in enumerate(out))
    return None if ok else "a rank's piece differs"


def broadcast():
    z = torch.full((1000,), 7.0 if rank == 0 else 0.0, device="cuda")
    dist.broadcast(z, src=0)
    return None if bool((z == 7).all()) else "not what rank 0 sent"


def ddp():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    batch = torch.randn(8 * world, 16)
    # The CPU's gradients for the whole batch, the mean over it.
    model(batch).pow(2).mean().backward()
    want = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    ddp_model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[rank])
    ddp_model(batch[8 * rank:8 * (rank + 1)].cuda()).pow(2).mean().backward()
    diff = max(float((p.grad.cpu() - w).abs().max()) for p, w in zip(ddp_model.parameters(), want))
    return None if diff <= 1e-5 else f"max gradient difference {diff:g}"


check("all_reduce", all_reduce)
check("all_gather", all_gather)
check("broadcast", broadcast)
check("DistributedDataParallel's averaged gradients", ddp)
dist.destroy_process_group()
