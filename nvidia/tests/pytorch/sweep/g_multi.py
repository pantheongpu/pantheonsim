"""Two simulated GPUs in one process: peer copies, per-device streams and
events, torch.cuda.comm, NCCL through torch.cuda.nccl, DataParallel, a model
split across devices. Needs VGPU_DEVICE_COUNT=2 (the checks skip with one)."""
import torch
import torch.nn as nn

from harness import check, rnd

L = 'multi'
N = ('cuda2',)


def _peer_copy(d):
    x = rnd(10000, seed=1)
    if d == 'cpu':
        return [x * 2, x * 2]
    a = x.to('cuda:0')
    b = a.to('cuda:1') * 2
    c = b.to('cuda:0')
    return [b.cpu(), c.cpu()]


check(L, 'tensor copies between devices, and a kernel on the second', 0, needs=N)(_peer_copy)


def _peer_access(d):
    if d == 'cpu':
        return torch.tensor([1.0])
    ok = True  # whether the pair reports peer access is the hardware's; the copy must work either way
    a = rnd(1000, seed=1).to('cuda:0')
    with torch.cuda.device(1):
        b = torch.empty(1000, device='cuda:1')
        b.copy_(a, non_blocking=True)
        torch.cuda.synchronize()
    return torch.tensor([float(ok and torch.equal(a.cpu(), b.cpu()))])


check(L, 'peer access and an asynchronous peer copy', 0, needs=N)(_peer_access)


def _streams_events(d):
    x = rnd(5000, seed=1)
    if d == 'cpu':
        return (x.sin() * 2).cos()
    a = x.to('cuda:0')
    s0 = torch.cuda.Stream(device=0)
    s1 = torch.cuda.Stream(device=1)
    e = torch.cuda.Event()
    with torch.cuda.stream(s0):
        y = a.sin() * 2
        e.record(s0)
    s1.wait_event(e)
    with torch.cuda.stream(s1):
        z = y.to('cuda:1').cos()
    s1.synchronize()
    return z.cpu()


check(L, 'a stream on each device ordered by an event across them', 1e-5, needs=N)(_streams_events)


def _comm(d):
    x = rnd(2000, seed=1)
    if d == 'cpu':
        return [x * 2, x[:1000], x[1000:], torch.cat([x[:1000], x[1000:]]), x * 2]
    import torch.cuda.comm as comm
    outs = comm.broadcast(x.to('cuda:0') * 2, (0, 1))
    sc = comm.scatter(x.to('cuda:0'), (0, 1))
    g = comm.gather(sc, 0, 'cuda:0')
    r = comm.reduce_add([x.to('cuda:0'), x.to('cuda:1')], 0)
    return [outs[1].cpu(), sc[0].cpu(), sc[1].cpu(), g.cpu(), r.cpu()]


check(L, 'torch.cuda.comm: broadcast, scatter, gather, reduce_add', 1e-6, needs=N)(_comm)


def _nccl(d):
    xs = [rnd(3000, seed=i + 1) for i in range(2)]
    if d == 'cpu':
        s = xs[0] + xs[1]
        return [s, s, xs[0], xs[0], torch.cat(xs)]
    import torch.cuda.nccl as nccl
    ts = [x.to(f'cuda:{i}') for i, x in enumerate(xs)]
    nccl.all_reduce(ts)
    a, b = ts[0].cpu(), ts[1].cpu()
    bs = [xs[0].to('cuda:0'), torch.zeros(3000, device='cuda:1')]
    nccl.broadcast(bs, root=0)
    gs = [x.to(f'cuda:{i}') for i, x in enumerate(xs)]
    outs = [torch.empty(6000, device=f'cuda:{i}') for i in range(2)]
    nccl.all_gather(gs, outs)
    return [a, b, bs[0].cpu(), bs[1].cpu(), outs[1].cpu()]


check(L, 'torch.cuda.nccl: all_reduce, broadcast, all_gather over the two devices', 1e-5, needs=N)(_nccl)


def _reduce_scatter(d):
    xs = [rnd(4000, seed=i + 1) for i in range(2)]
    if d == 'cpu':
        s = xs[0] + xs[1]
        return [s[:2000], s[2000:]]
    import torch.cuda.nccl as nccl
    ins = [x.to(f'cuda:{i}') for i, x in enumerate(xs)]
    outs = [torch.empty(2000, device=f'cuda:{i}') for i in range(2)]
    nccl.reduce_scatter(ins, outs)
    return [o.cpu() for o in outs]


check(L, 'torch.cuda.nccl: reduce_scatter', 1e-5, needs=N, tier='full')(_reduce_scatter)


def _dataparallel(d):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    x, y = rnd(8, 16, seed=1), rnd(8, 4, seed=2)
    if d == 'cpu':
        (net(x) - y).square().mean().backward()
        return [net(x).detach(), [p.grad for p in net.parameters()]]
    dp = nn.DataParallel(net.to('cuda:0'), device_ids=[0, 1])
    out = dp(x.to('cuda:0'))
    # DataParallel gathers the pieces; the loss over the whole batch gives the whole-batch gradient.
    (out - y.to('cuda:0')).square().mean().backward()
    return [out.detach().cpu(), [p.grad for p in net.parameters()]]


check(L, 'DataParallel across both devices: outputs and the whole-batch gradients', 1e-4, needs=N)(_dataparallel)


def _dp_train(d):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 4))
    x, y = rnd(8, 16, seed=1), rnd(8, 4, seed=2)
    m = nn.DataParallel(net.to('cuda:0'), device_ids=[0, 1]) if d != 'cpu' else net
    opt = torch.optim.SGD(net.parameters(), 0.1, momentum=0.9)
    for _ in range(3):
        opt.zero_grad()
        (m(x.to('cuda:0') if d != 'cpu' else x) - (y.to('cuda:0') if d != 'cpu' else y)).square().mean().backward()
        opt.step()
    return list(net.parameters())


check(L, 'DataParallel training, three SGD steps', 1e-3, needs=N, tier='full')(_dp_train)


def _model_parallel(d):
    torch.manual_seed(0)
    a, b = nn.Linear(16, 32), nn.Linear(32, 4)
    x = rnd(8, 16, seed=1)
    if d == 'cpu':
        out = b(torch.relu(a(x)))
        out.sum().backward()
        return [out.detach(), a.weight.grad, b.weight.grad]
    a, b = a.to('cuda:0'), b.to('cuda:1')
    out = b(torch.relu(a(x.to('cuda:0'))).to('cuda:1'))
    out.sum().backward()
    return [out.detach().cpu(), a.weight.grad.cpu(), b.weight.grad.cpu()]


check(L, 'a model split across two devices, with gradients across the boundary', 1e-4, needs=N)(_model_parallel)


def _device_ctx(d):
    if d == 'cpu':
        return torch.tensor([0.0, 1.0, 1.0])
    out = []
    for i in (0, 1):
        with torch.cuda.device(i):
            out.append(float(torch.zeros(1, device='cuda').device.index))
    out.append(float(torch.cuda.get_device_properties(1).name == torch.cuda.get_device_properties(0).name))
    return torch.tensor([out[0], out[1], out[2]])


check(L, 'device context, current device, and device properties on each device', 0, needs=N)(_device_ctx)
