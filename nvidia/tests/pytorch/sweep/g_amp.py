"""Automatic mixed precision: autocast in half and bfloat16 across layer
kinds, GradScaler (scale, unscale, inf detection, step skipping, growth)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from harness import check, rnd, lowp

L = 'amp'


def amp(dtype, mk, shapes, tol, name, tier='quick'):
    def run(d):
        torch.manual_seed(0)
        m = mk()
        if d == 'cpu':
            for p in m.parameters():
                p.data = lowp(p.data, dtype)
        m = m.to(d)
        xs = [rnd(*s, seed=3 + i).to(d) for i, s in enumerate(shapes)]
        if d == 'cpu':
            xs = [lowp(x, dtype) for x in xs]
        xs = [x.requires_grad_() for x in xs]
        with torch.autocast('cuda', dtype=dtype, enabled=d != 'cpu'):
            y = m(*xs)
        y = y[0] if isinstance(y, tuple) else y
        y.float().square().sum().backward()
        return [y.float(), [x.grad for x in xs], [p.grad for p in m.parameters()]]
    check(L, name, tol, tier=tier)(run)


for dt, tag, tol in ((torch.float16, 'half', 3e-2), (torch.bfloat16, 'bfloat16', 2e-1)):
    amp(dt, lambda: nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8)), [(6, 16)], tol, f'autocast {tag}: MLP')
    amp(dt, lambda: nn.Sequential(nn.Conv2d(3, 6, 3, padding=1), nn.BatchNorm2d(6), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(6, 4)),
        [(4, 3, 8, 8)], tol, f'autocast {tag}: conv, BatchNorm, pooling, linear')
    amp(dt, lambda: nn.LSTM(8, 12, 2), [(5, 3, 8)], tol, f'autocast {tag}: LSTM', 'full' if dt == torch.bfloat16 else 'quick')
    amp(dt, lambda: nn.TransformerEncoderLayer(16, 4, 32, dropout=0.0, batch_first=True), [(3, 6, 16)], tol, f'autocast {tag}: transformer layer')
    amp(dt, lambda: nn.Sequential(nn.LayerNorm(16), nn.Linear(16, 16), nn.Softmax(-1)), [(6, 16)], tol, f'autocast {tag}: LayerNorm and softmax stay in float32', 'full')


def _scaler(d):
    torch.manual_seed(0)
    m = nn.Linear(10, 3).to(d)
    opt = torch.optim.SGD(m.parameters(), 0.1)
    sc = torch.amp.GradScaler('cpu' if d == 'cpu' else 'cuda', init_scale=1024.0, growth_interval=2)
    x, y = rnd(8, 10, seed=1).to(d), rnd(8, 3, seed=2).to(d)
    scales = []
    for i in range(5):
        opt.zero_grad()
        with torch.autocast('cuda', dtype=torch.float16, enabled=d != 'cpu'):
            loss = F.mse_loss(m(x).float(), y)
        if i == 2:
            loss = loss * float('inf')  # a step with infinite gradients: skipped, scale halved
        sc.scale(loss).backward()
        sc.unscale_(opt)
        sc.step(opt)
        sc.update()
        scales.append(sc.get_scale())
    return [list(m.parameters()), torch.tensor(scales)]


check(L, 'GradScaler: scale, unscale, an infinite step skipped, growth', 5e-2)(_scaler)


def _foreach_unscale(d):
    ts = [rnd(7, 5, seed=1).to(d), rnd(100, seed=2).to(d), rnd(3, 3, seed=3).to(d).half()]
    inv = torch.tensor(0.25, device=d)
    found = torch.zeros(1, device=d)
    if d == 'cpu':
        return [[t * 0.25 for t in ts], found]
    torch._amp_foreach_non_finite_check_and_unscale_(ts, found, inv)
    return [ts, found]


check(L, '_amp_foreach_non_finite_check_and_unscale_ on mixed dtypes', 1e-3)(_foreach_unscale)


def _amp_update_scale(d):
    if d == 'cpu':
        return [torch.tensor([512.0]), torch.tensor([0], dtype=torch.int32)]
    scale = torch.tensor([1024.0], device=d)
    tracker = torch.tensor([3], dtype=torch.int32, device=d)
    found = torch.tensor([1.0], device=d)
    torch._amp_update_scale_(scale, tracker, found, 2.0, 0.5, 4)
    return [scale, tracker]


check(L, '_amp_update_scale_ backs the scale off on an infinity', 0)(_amp_update_scale)


def _bf16_train(d):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(12, 24), nn.ReLU(), nn.Linear(24, 4))
    if d == 'cpu':
        for p in m.parameters():
            p.data = lowp(p.data, torch.bfloat16)
    m = m.to(d)
    opt = torch.optim.AdamW(m.parameters(), 1e-2)
    x, y = lowp(rnd(16, 12, seed=1), torch.bfloat16).to(d), rnd(16, 4, seed=2).to(d)
    for _ in range(3):
        opt.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=d != 'cpu'):
            loss = F.mse_loss(m(x).float(), y)
        loss.backward()
        opt.step()
    return list(m.parameters())


check(L, 'bfloat16 autocast training, three AdamW steps', 5e-2)(_bf16_train)


def _pure_half(d):
    # A model kept entirely in half, trained with a float32 master copy of the loss.
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(12, 24), nn.GELU(), nn.Linear(24, 4))
    if d == 'cpu':
        for p in m.parameters():
            p.data = lowp(p.data, torch.float16)
    else:
        m = m.half()
    m = m.to(d)
    x = lowp(rnd(16, 12, seed=1), torch.float16).to(d)
    x = x.half() if d != 'cpu' else x
    out = m(x)
    out.float().square().mean().backward()
    return [out.float(), [p.grad.float() for p in m.parameters()]]


check(L, 'a model held in half (parameters and gradients)', 3e-2)(_pure_half)
