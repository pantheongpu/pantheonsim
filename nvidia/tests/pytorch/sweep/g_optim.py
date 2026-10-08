"""Optimizers (single-tensor, foreach and fused implementations), schedulers
and gradient clipping: several steps on a small model, parameters compared."""
import torch
import torch.nn as nn

from harness import check, rnd

L = 'optim'


def model(d):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(12, 24), nn.Tanh(), nn.Linear(24, 5), nn.LayerNorm(5))
    return m.to(d)


def train(make_opt, steps=4, sched=None, clip=None, loss='mse'):
    def run(d):
        m = model(d)
        opt = make_opt(m.parameters())
        s = sched(opt) if sched else None
        x, y = rnd(16, 12, seed=1).to(d), rnd(16, 5, seed=2).to(d)
        for _ in range(steps):
            def closure():
                opt.zero_grad()
                l = (m(x) - y).square().mean()
                l.backward()
                return l
            if isinstance(opt, torch.optim.LBFGS):
                opt.step(closure)
            else:
                closure()
                if clip:
                    nn.utils.clip_grad_norm_(m.parameters(), clip)
                opt.step()
            if s:
                s.step()
        return [list(m.parameters()), [t for st in opt.state.values() for t in st.values() if isinstance(t, torch.Tensor) and t.is_floating_point()]]
    return run


O = torch.optim


def _cap(p):
    p = list(p)
    return O.AdamW(p, 1e-2, capturable=p[0].is_cuda)


OPTS = {
    'SGD': lambda p: O.SGD(p, 0.05),
    'SGD momentum, nesterov, weight decay': lambda p: O.SGD(p, 0.05, momentum=0.9, nesterov=True, weight_decay=1e-3),
    'SGD dampening, maximize': lambda p: O.SGD(p, 0.05, momentum=0.9, dampening=0.1, maximize=True),
    'Adam': lambda p: O.Adam(p, 1e-2),
    'Adam amsgrad, weight decay': lambda p: O.Adam(p, 1e-2, amsgrad=True, weight_decay=1e-2),
    'Adam foreach': lambda p: O.Adam(p, 1e-2, foreach=True),
    'Adam, single tensor': lambda p: O.Adam(p, 1e-2, foreach=False),
    'Adam fused': lambda p: O.Adam(p, 1e-2, fused=True),
    'AdamW': lambda p: O.AdamW(p, 1e-2, weight_decay=0.05),
    'AdamW fused': lambda p: O.AdamW(p, 1e-2, weight_decay=0.05, fused=True),
    'AdamW capturable': lambda p: _cap(p),
    'Adamax': lambda p: O.Adamax(p, 1e-2),
    'NAdam': lambda p: O.NAdam(p, 1e-2, decoupled_weight_decay=True, weight_decay=1e-2),
    'RAdam': lambda p: O.RAdam(p, 1e-2),
    'RMSprop': lambda p: O.RMSprop(p, 1e-3, momentum=0.5, centered=True),
    'Adagrad': lambda p: O.Adagrad(p, 1e-2, lr_decay=1e-3),
    'Adadelta': lambda p: O.Adadelta(p, 1.0),
    'ASGD': lambda p: O.ASGD(p, 1e-2),
    'Rprop': lambda p: O.Rprop(p, 1e-2),
    'SGD fused': lambda p: O.SGD(p, 0.05, momentum=0.9, fused=True),
    'LBFGS': lambda p: O.LBFGS(p, 0.5, max_iter=5),
}
for name, mk in OPTS.items():
    check(L, f'optimizer {name}', 2e-3, tier='full' if name in ('SGD dampening, maximize', 'Adam, single tensor', 'ASGD', 'Adadelta', 'Rprop', 'RAdam', 'Adamax', 'AdamW capturable') else 'quick')(train(mk))


class Lion(torch.optim.Optimizer):
    """Lion (sign of an interpolated momentum), written in torch ops, as users do:
    foreach lerp, sign and add kernels."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), wd=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, wd=wd))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            ps = [p for p in g['params'] if p.grad is not None]
            ms = [self.state[p].setdefault('m', torch.zeros_like(p)) for p in ps]
            torch._foreach_mul_(ps, 1 - g['lr'] * g['wd'])
            upd = torch._foreach_mul(ms, g['betas'][0])
            torch._foreach_add_(upd, [p.grad for p in ps], alpha=1 - g['betas'][0])
            torch._foreach_sign_(upd)
            torch._foreach_add_(ps, upd, alpha=-g['lr'])
            torch._foreach_mul_(ms, g['betas'][1])
            torch._foreach_add_(ms, [p.grad for p in ps], alpha=1 - g['betas'][1])


check(L, 'optimizer Lion (foreach kernels)', 2e-3)(train(lambda p: Lion(p, 3e-3, wd=0.01)))
check(L, 'Adam with a StepLR schedule', 2e-3)(train(lambda p: O.Adam(p, 1e-2), 5, lambda o: O.lr_scheduler.StepLR(o, 2, 0.5)))
check(L, 'SGD with OneCycleLR and gradient clipping', 2e-3)(train(lambda p: O.SGD(p, 0.05, momentum=0.9), 5, lambda o: O.lr_scheduler.OneCycleLR(o, 0.1, total_steps=5), clip=0.5))
check(L, 'AdamW with cosine warm restarts', 2e-3, tier='full')(train(lambda p: O.AdamW(p, 1e-2), 6, lambda o: O.lr_scheduler.CosineAnnealingWarmRestarts(o, 2)))


def _clip(d):
    m = model(d)
    x = rnd(16, 12, seed=1).to(d)
    m(x).square().sum().backward()
    n1 = nn.utils.clip_grad_norm_(m.parameters(), 0.5)
    n2 = nn.utils.clip_grad_norm_(m.parameters(), 0.5, norm_type=float('inf'), foreach=False)
    nn.utils.clip_grad_value_(m.parameters(), 0.01)
    return [n1, n2, [p.grad for p in m.parameters()], nn.utils.parameters_to_vector(m.parameters())]


check(L, 'clip_grad_norm_ (foreach), clip_grad_value_, parameters_to_vector', 1e-4)(_clip)


def _ema(d):
    m = model(d)
    ema = torch.optim.swa_utils.AveragedModel(m, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9))
    opt = O.SGD(m.parameters(), 0.1)
    x = rnd(16, 12, seed=1).to(d)
    for _ in range(3):
        opt.zero_grad()
        m(x).square().mean().backward()
        opt.step()
        ema.update_parameters(m)
    return [list(ema.module.parameters())]


check(L, 'exponential moving average of weights (AveragedModel, foreach lerp)', 1e-4, tier='full')(_ema)


def _state_dict(d):
    # Save the optimizer's state, load it into a fresh one, and continue: the state moves between devices.
    m = model(d)
    opt = O.Adam(m.parameters(), 1e-2)
    x = rnd(16, 12, seed=1).to(d)
    for _ in range(2):
        opt.zero_grad(); m(x).square().mean().backward(); opt.step()
    sd = opt.state_dict()
    m2 = model(d)
    m2.load_state_dict(m.state_dict())
    opt2 = O.Adam(m2.parameters(), 1e-2)
    opt2.load_state_dict(sd)
    opt2.zero_grad(); m2(x).square().mean().backward(); opt2.step()
    return list(m2.parameters())


check(L, 'optimizer state_dict round trip', 1e-3, tier='full')(_state_dict)
