"""torch.compile: Inductor's Triton kernels (compiled to PTX, run by the
simulator) for forward and backward, training steps, dynamic shapes, CUDA
graphs through mode="reduce-overhead", and flex_attention."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from harness import check, rnd

L = 'compile'


def comp(f, **kw):
    fast = torch.compile(f, **kw)
    return lambda d, *a: (fast if d != 'cpu' else f)(*[t.to(d) for t in a])


def reg(name, f, args, tol=1e-3, tier='quick', **kw):
    g = comp(f, **kw)
    check(L, 'torch.compile: ' + name, tol, tier=tier)(lambda d: g(d, *args))


x2 = rnd(32, 48, seed=1)
x4 = rnd(2, 4, 16, 32, seed=2)

reg('softmax, log_softmax and cross entropy', lambda t: F.cross_entropy(t, torch.arange(32, device=t.device) % 48), [x2])
reg('RMSNorm and rotary-style pointwise math', lambda t: (t * torch.rsqrt(t.square().mean(-1, keepdim=True) + 1e-6)) * torch.cos(t) + torch.sin(t.roll(1, -1)), [x2])
reg('cumsum, sort and argmax', lambda t: [t.cumsum(1), t.sort(1).values, t.argmax(1)], [x2])
reg('gather, scatter_add and index_select', lambda t: [t.gather(1, (torch.arange(48, device=t.device) % 48).expand(32, 48)), torch.zeros_like(t).scatter_add(0, (torch.arange(32, device=t.device) % 5).unsqueeze(1).expand(32, 48), t), t.index_select(0, torch.tensor([3, 1, 4], device=t.device))], [x2])
reg('half and bfloat16 pointwise chains', lambda t: [(t.half() * 2 + 1).sin().float(), (t.bfloat16().exp() - 1).float()], [x2], 2e-2)
reg('integer and boolean math', lambda t: [((t * 100).long() % 7 + (t > 0).long()).float(), torch.where(t > 0, t, -t * 2)], [x2])
reg('where, clamp, masked_fill and max-reductions with dims', lambda t: [t.masked_fill(t > 1, 0).amax(0), t.clamp(-0.5, 0.5).sum(1), t.max(dim=1)[1]], [x2])
reg('matmul with a fused bias, GELU and residual', lambda a, w, b: F.gelu(a @ w + b) * 2, [x2, rnd(48, 24, seed=3), rnd(24, seed=4)], 2e-3)
reg('conv2d, batch norm (eval) and ReLU', lambda t, w: F.relu(F.batch_norm(F.conv2d(t, w, padding=1), torch.zeros(6, device=t.device), torch.ones(6, device=t.device))), [rnd(2, 3, 12, 12, seed=2), rnd(6, 3, 3, 3, seed=3)], 2e-3)
reg('layer norm backward (AOTAutograd)', lambda t: _grad(lambda u: F.layer_norm(u, (48,)).square().mul(torch.arange(48, device=u.device)).sum(), t), [x2], 2e-3)
reg('attention block with a causal mask', lambda q, k, v: _grad_multi(lambda a, b, c: (torch.softmax((a @ b.transpose(-1, -2)) / 5.6 + torch.triu(torch.full((16, 16), float('-inf'), device=a.device), 1), -1) @ c).square().sum(), q, k, v), [x4, x4.flip(-1), x4 * 0.5], 3e-2)
reg('dynamic shapes: the same function at three lengths', lambda t: _dyn(t), [x2], 1e-3, tier='full')
reg('graph breaks and a data-dependent branch', lambda t: t.sum() if t.sum() > 0 else -t.sum(), [x2], 1e-3, tier='full')
reg('max-autotune-no-cudagraphs: a matmul template', lambda a, b: a @ b + 1, [rnd(64, 64, seed=1), rnd(64, 64, seed=2)], 5e-3, tier='full', mode='max-autotune-no-cudagraphs')


def _grad(f, t):
    t = t.detach().clone().requires_grad_()
    y = f(t)
    y.backward()
    return t.grad


def _grad_multi(f, *ts):
    ts = [t.detach().clone().requires_grad_() for t in ts]
    f(*ts).backward()
    return [t.grad for t in ts]


def _dyn(t):
    return torch.stack([(t[:n] * 2).sum() for n in (8, 16, 32)])


def _train(d):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.LayerNorm(32), nn.Linear(32, 4)).to(d)
    fwd = torch.compile(net) if d != 'cpu' else net
    opt = torch.optim.AdamW(net.parameters(), 1e-2)
    xs, ys = rnd(12, 16, seed=1).to(d), rnd(12, 4, seed=2).to(d)
    ls = []
    for _ in range(3):
        opt.zero_grad()
        l = (fwd(xs) - ys).square().mean()
        l.backward()
        opt.step()
        ls.append(l.detach())
    return [torch.stack(ls), list(net.parameters())]


check(L, 'torch.compile: a model trained three AdamW steps (forward and backward kernels)', 3e-3)(_train)


def _compiled_opt(d):
    torch.manual_seed(0)
    net = nn.Linear(16, 4).to(d)
    opt = torch.optim.Adam(net.parameters(), 1e-2, capturable=d != 'cpu')
    step = torch.compile(opt.step) if d != 'cpu' else opt.step
    xs = rnd(12, 16, seed=1).to(d)
    for _ in range(2):
        opt.zero_grad()
        net(xs).square().mean().backward()
        step()
    return list(net.parameters())


check(L, 'torch.compile: the optimizer step (fused foreach kernels)', 3e-3, tier='full')(_compiled_opt)


def _reduce_overhead(d):
    f = lambda t: (t.sin() * 3).cumsum(0)
    g = torch.compile(f, mode='reduce-overhead') if d != 'cpu' else f
    outs = []
    for i in range(3):
        outs.append(g(rnd(500, seed=i).to(d)).clone())
    return outs


check(L, 'torch.compile: mode="reduce-overhead" (CUDA graphs under Inductor)', 1e-3)(_reduce_overhead)


def _flex(d):
    from torch.nn.attention.flex_attention import flex_attention
    q, k, v = (rnd(1, 2, 64, 32, seed=s).to(d) for s in (1, 2, 3))
    if d == 'cpu':
        causal = torch.tril(torch.ones(64, 64, dtype=torch.bool))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
    fa = torch.compile(flex_attention)
    return fa(q, k, v, score_mod=lambda s, b, h, qi, ki: torch.where(qi >= ki, s, -float('inf')))


check(L, 'flex_attention with a causal score_mod (a templated Triton kernel)', 5e-3, tier='full')(_flex)
