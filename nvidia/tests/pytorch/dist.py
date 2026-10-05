"""torch.distributed with the NCCL backend across two processes, each with its
own simulated NVIDIA GPU (nvidia/tests/e2e/run_pytorch_dist.sh launches the two
the way torchrun does: RANK and WORLD_SIZE say which). The ranks reach each
other's device memory through NCCL's file-backed transport. Collectives in
several dtypes, point to point, sub-groups, Python objects, and
DistributedDataParallel / FullyShardedDataParallel training, whose gradients
must be those of the whole batch on the CPU. One line a check from each rank:
"ok rank <r>: <name>" or "FAIL rank <r>: <name>: <why>".
"""
import datetime
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', rank=rank, world_size=world, device_id=torch.device('cuda', rank),
                        timeout=datetime.timedelta(seconds=300))
FULL = os.environ.get('VGPU_SWEEP_TIER', 'quick') == 'full'
ONLY = os.environ.get('VGPU_SWEEP_ONLY')


def check(name, f, full=False):
    if (full and not FULL) or (ONLY and ONLY not in name):
        return
    try:
        why = f()
        print(f"ok rank {rank}: {name}" if why is None else f"FAIL rank {rank}: {name}: {why}", flush=True)
    except Exception as e:
        print(f"FAIL rank {rank}: {name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)


def same(got, want, tol=0.0):
    d = float((got.detach().float().cpu() - want.float()).abs().max())
    return None if d <= tol else f"max difference {d:g}"


def per_rank(r, n=1000, dtype=torch.float32):
    return (torch.arange(n, dtype=torch.float32) % 13 * (r + 1) - 5).to(dtype)


def all_reduce(op, dtype, expect):
    def run():
        x = per_rank(rank, dtype=dtype).cuda()
        dist.all_reduce(x, op=op)
        return same(x, expect(torch.stack([per_rank(r) for r in range(world)])).to(dtype).float(), 1e-2 if dtype != torch.float32 else 0)
    return run


check('all_reduce sum, float32', all_reduce(dist.ReduceOp.SUM, torch.float32, lambda s: s.sum(0)))
check('all_reduce sum, half', all_reduce(dist.ReduceOp.SUM, torch.float16, lambda s: s.sum(0)))
check('all_reduce sum, bfloat16', all_reduce(dist.ReduceOp.SUM, torch.bfloat16, lambda s: s.sum(0)))
check('all_reduce max', all_reduce(dist.ReduceOp.MAX, torch.float32, lambda s: s.amax(0)))
check('all_reduce min, int64', all_reduce(dist.ReduceOp.MIN, torch.int64, lambda s: s.amin(0)), full=True)
check('all_reduce product', all_reduce(dist.ReduceOp.PRODUCT, torch.float32, lambda s: s.prod(0)), full=True)
check('all_reduce average', all_reduce(dist.ReduceOp.AVG, torch.float32, lambda s: s.mean(0)))


def broadcast():
    z = torch.full((5000,), 7.0 if rank == 0 else 0.0, device='cuda')
    dist.broadcast(z, src=0)
    return same(z, torch.full((5000,), 7.0))


def all_gather():
    y = per_rank(rank, 64).cuda()
    out = [torch.empty_like(y) for _ in range(world)]
    dist.all_gather(out, y)
    for r, o in enumerate(out):
        w = same(o, per_rank(r, 64))
        if w:
            return f"rank {r}'s piece: {w}"


def all_gather_into_tensor():
    y = per_rank(rank, 64).cuda()
    out = torch.empty(world * 64, device='cuda')
    dist.all_gather_into_tensor(out, y)
    return same(out, torch.cat([per_rank(r, 64) for r in range(world)]))


def reduce_scatter():
    xs = per_rank(rank, world * 32).cuda()
    out = torch.empty(32, device='cuda')
    dist.reduce_scatter_tensor(out, xs)
    full = sum(per_rank(r, world * 32) for r in range(world))
    return same(out, full[rank * 32:(rank + 1) * 32])


def all_to_all():
    x = (torch.arange(world * 4, dtype=torch.float32) + 100 * rank).cuda()
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x)
    want = torch.cat([(torch.arange(world * 4, dtype=torch.float32) + 100 * r)[rank * 4:(rank + 1) * 4] for r in range(world)])
    return same(out, want)


def reduce_():
    x = per_rank(rank, 200).cuda()
    dist.reduce(x, dst=1, op=dist.ReduceOp.SUM)
    if rank == 1:
        return same(x, sum(per_rank(r, 200) for r in range(world)))


def gather_scatter():
    x = per_rank(rank, 50).cuda()
    gl = [torch.empty_like(x) for _ in range(world)] if rank == 0 else None
    dist.gather(x, gl, dst=0)
    if rank == 0:
        for r, g in enumerate(gl):
            w = same(g, per_rank(r, 50))
            if w:
                return w
    out = torch.empty(50, device='cuda')
    dist.scatter(out, [per_rank(r, 50).cuda() + 1 for r in range(world)] if rank == 0 else None, src=0)
    return same(out, per_rank(rank, 50) + 1)


def send_recv():
    if rank == 0:
        dist.send(per_rank(0, 300).cuda(), dst=1)
        back = torch.empty(300, device='cuda')
        dist.recv(back, src=1)
        return same(back, per_rank(1, 300))
    got = torch.empty(300, device='cuda')
    dist.recv(got, src=0)
    dist.send(per_rank(1, 300).cuda(), dst=0)
    return same(got, per_rank(0, 300))


def isend_irecv():
    peer = 1 - rank
    x, y = per_rank(rank, 128).cuda(), torch.empty(128, device='cuda')
    ops = [dist.P2POp(dist.isend, x, peer), dist.P2POp(dist.irecv, y, peer)]
    for req in dist.batch_isend_irecv(ops):
        req.wait()
    return same(y, per_rank(peer, 128))


def objects():
    out = [None] * world
    dist.all_gather_object(out, {'rank': rank, 'text': 'x' * (rank + 3)})
    return None if all(o == {'rank': r, 'text': 'x' * (r + 3)} for r, o in enumerate(out)) else f"got {out}"


def barrier_async():
    x = torch.ones(10, device='cuda')
    w = dist.all_reduce(x, async_op=True)
    w.wait()
    dist.barrier()
    return same(x, torch.full((10,), float(world)))


def new_group():
    g = dist.new_group([0, 1])
    x = per_rank(rank, 100).cuda()
    dist.all_reduce(x, group=g)
    return same(x, sum(per_rank(r, 100) for r in range(world)))


def coalesced():
    ts = [per_rank(rank, n).cuda() for n in (33, 100, 7)]
    dist.all_reduce_coalesced(ts) if hasattr(dist, 'all_reduce_coalesced') else [dist.all_reduce(t) for t in ts]
    for t, n in zip(ts, (33, 100, 7)):
        w = same(t, sum(per_rank(r, n) for r in range(world)))
        if w:
            return w


def large_message():
    x = torch.full((4_000_000,), float(rank + 1), device='cuda')
    dist.all_reduce(x)
    return None if bool((x == sum(range(1, world + 1))).all()) else 'a 16 MB all_reduce differs'


check('broadcast', broadcast)
check('all_gather', all_gather)
check('all_gather_into_tensor', all_gather_into_tensor)
check('reduce_scatter_tensor', reduce_scatter)
check('all_to_all_single', all_to_all)
check('reduce to rank 1', reduce_)
check('gather and scatter', gather_scatter)
check('send and recv', send_recv)
check('batch_isend_irecv', isend_irecv, full=True)
check('all_gather_object', objects)
check('asynchronous all_reduce, wait and barrier', barrier_async)
check('new_group over both ranks', new_group, full=True)
check('all_reduce on several tensors', coalesced, full=True)
check('a 16 MB all_reduce', large_message, full=True)


def make_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))


def whole_batch_grads():
    batch = torch.randn(8 * world, 16, generator=torch.Generator().manual_seed(5))
    model = make_model()
    model(batch).pow(2).mean().backward()
    return batch, [p.grad.clone() for p in model.parameters()]


def ddp():
    batch, want = whole_batch_grads()
    m = nn.parallel.DistributedDataParallel(make_model().cuda(), device_ids=[rank])
    m(batch[8 * rank:8 * (rank + 1)].cuda()).pow(2).mean().backward()
    diff = max(float((p.grad.cpu() - w).abs().max()) for p, w in zip(m.parameters(), want))
    return None if diff <= 1e-5 else f"max gradient difference {diff:g}"


def ddp_train():
    # Three SGD steps: the replicas must stay identical and equal the CPU's whole-batch training.
    batch = torch.randn(8 * world, 16, generator=torch.Generator().manual_seed(5))
    cpu = make_model()
    opt = torch.optim.SGD(cpu.parameters(), 0.1, momentum=0.9)
    for _ in range(3):
        opt.zero_grad(); cpu(batch).pow(2).mean().backward(); opt.step()
    m = nn.parallel.DistributedDataParallel(make_model().cuda(), device_ids=[rank], bucket_cap_mb=1)
    opt = torch.optim.SGD(m.parameters(), 0.1, momentum=0.9)
    for _ in range(3):
        opt.zero_grad(); m(batch[8 * rank:8 * (rank + 1)].cuda()).pow(2).mean().backward(); opt.step()
    diff = max(float((p.cpu() - c).abs().max()) for p, c in zip(m.parameters(), cpu.parameters()))
    return None if diff <= 1e-4 else f"max parameter difference {diff:g}"


def ddp_no_sync_and_unused():
    batch, _ = whole_batch_grads()
    net = nn.ModuleDict({'a': nn.Linear(16, 8), 'unused': nn.Linear(16, 8)})
    torch.manual_seed(0)
    m = nn.parallel.DistributedDataParallel(net.cuda(), device_ids=[rank], find_unused_parameters=True)
    x = batch[8 * rank:8 * (rank + 1)].cuda()
    with m.no_sync():
        m.module['a'](x).sum().backward()
    m.module['a'](x).sum().backward()
    g = m.module['a'].weight.grad.clone()
    gs = [torch.empty_like(g) for _ in range(world)]
    dist.all_gather(gs, g)
    return same(gs[0], gs[1].cpu())


def ddp_half_hook():
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
    batch, want = whole_batch_grads()
    m = nn.parallel.DistributedDataParallel(make_model().cuda(), device_ids=[rank])
    m.register_comm_hook(None, default_hooks.fp16_compress_hook)
    m(batch[8 * rank:8 * (rank + 1)].cuda()).pow(2).mean().backward()
    diff = max(float((p.grad.cpu() - w).abs().max()) for p, w in zip(m.parameters(), want))
    return None if diff <= 2e-3 else f"max gradient difference {diff:g}"


def ddp_bn():
    # SyncBatchNorm: the batch statistics are those of the whole batch.
    batch = torch.randn(8 * world, 6, 5, 5, generator=torch.Generator().manual_seed(5))
    ref = nn.BatchNorm2d(6)
    want = ref(batch)[8 * rank:8 * (rank + 1)]
    sb = nn.SyncBatchNorm(6).cuda()
    got = sb(batch[8 * rank:8 * (rank + 1)].cuda())
    return same(got, want, 1e-4)


def fsdp():
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    batch, want = whole_batch_grads()
    m = FSDP(make_model().cuda(), device_id=rank)
    out = m(batch[8 * rank:8 * (rank + 1)].cuda())
    out.pow(2).mean().backward()
    with FSDP.summon_full_params(m, with_grads=True):
        diff = max(float((p.grad.cpu() - w).abs().max()) for p, w in zip(m.parameters(), want))
    return None if diff <= 1e-5 else f"max gradient difference {diff:g}"


def fsdp2():
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor
    batch, want = whole_batch_grads()
    net = make_model().cuda()
    fully_shard(net)
    net(batch[8 * rank:8 * (rank + 1)].cuda()).pow(2).mean().backward()
    diff = max(float((p.grad.full_tensor().cpu() - w).abs().max()) for p, w in zip(net.parameters(), want))
    return None if diff <= 1e-5 else f"max gradient difference {diff:g}"


def dtensor():
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor
    mesh = init_device_mesh('cuda', (world,))
    full = per_rank(0, 8 * 6).reshape(8, 6).cuda()
    dt = distribute_tensor(full, mesh, [Shard(0)])
    res = (dt * 2 + 1).sum(dim=0).full_tensor()
    return same(res, (per_rank(0, 48).reshape(8, 6) * 2 + 1).sum(0))


check('DistributedDataParallel: averaged gradients are the whole batch\'s', ddp)
check('DistributedDataParallel: three SGD steps with small buckets match the CPU', ddp_train)
check('DistributedDataParallel: no_sync and find_unused_parameters', ddp_no_sync_and_unused, full=True)
check('DistributedDataParallel: fp16 gradient compression hook', ddp_half_hook, full=True)
check('SyncBatchNorm across the two ranks', ddp_bn, full=True)
check('FullyShardedDataParallel: full-parameter gradients are the whole batch\'s', fsdp, full=True)
check('fully_shard (FSDP2) with DTensor gradients', fsdp2, full=True)
check('DeviceMesh and a sharded DTensor reduction', dtensor, full=True)
dist.barrier()
dist.destroy_process_group()
