"""Debug probe (temporary): where the compiled capturable Adam step differs from the CPU's."""
import sys
import torch
import torch.nn as nn

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
print('torch', torch.__version__, 'triton', __import__('triton').__version__, flush=True)


def rnd(*shape, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


xs = rnd(12, 16, seed=1)


def run(d, compiled, grads=None):
    torch.manual_seed(0)
    net = nn.Linear(16, 4).to(d)
    opt = torch.optim.Adam(net.parameters(), 1e-2, capturable=d != 'cpu')
    step = torch.compile(opt.step) if compiled and d != 'cpu' else opt.step
    x = xs.to(d)
    log = []
    for i in range(2):
        opt.zero_grad()
        if grads is None:
            net(x).square().mean().backward()
        else:
            for p, g in zip(net.parameters(), grads[i]):
                p.grad = g.to(d).clone()
        gr = [p.grad.detach().cpu().clone() for p in net.parameters()]
        step()
        st = opt.state[net.weight]
        log.append(dict(g=gr[0], p=net.weight.detach().cpu().clone(), m=st['exp_avg'].detach().cpu().clone(),
                        v=st['exp_avg_sq'].detach().cpu().clone(), t=st['step'].detach().cpu().clone(),
                        pb=net.bias.detach().cpu().clone()))
    return log


def show(title, logs, names):
    print('=====', title, flush=True)
    ref = logs[0]
    for i in range(2):
        for k in ('g', 'p', 'm', 'v', 'pb'):
            row = []
            for n, lg in zip(names, logs):
                if lg is ref:
                    continue
                row.append('%s maxdiff %.3e' % (n, float((lg[i][k].double() - ref[i][k].double()).abs().max())))
            print('step', i + 1, k, ' | '.join(row))
        print('step', i + 1, 'steps', [float(lg[i]['t']) for lg in logs])
        for n, lg in zip(names, logs):
            e = lg[i]
            print('  %-10s [1,10] g=%.9g m=%.9g v=%.9g p=%.9g' % (n, e['g'][1, 10], e['m'][1, 10], e['v'][1, 10], e['p'][1, 10]))
    sys.stdout.flush()


cpu = run('cpu', False)
eager = run('cuda', False)
comp = run('cuda', True)
show('natural gradients', [cpu, eager, comp], ['cpu', 'eager', 'compiled'])

# Same gradients for all three: the CPU run's own, so only the optimizer step differs.
gb = [torch.zeros(4) + 0.01 * (i + 1) for i in range(2)]
fixed = [[cpu[i]['g'], gb[i]] for i in range(2)]
cpu2 = run('cpu', False, fixed)
eager2 = run('cuda', False, fixed)
comp2 = run('cuda', True, fixed)
show('fixed gradients', [cpu2, eager2, comp2], ['cpu', 'eager', 'compiled'])

# Pieces, compiled on the GPU against float64 on the CPU.
def probe(name, f, *args):
    try:
        ref = f(*[a.double() for a in args])
        g = torch.compile(f)(*[a.cuda() for a in args]).cpu()
        e = f(*[a.cuda() for a in args]).cpu()
        print('probe %-28s compiled-vs-f64 %.3e eager-vs-f64 %.3e' % (name, float((g.double() - ref).abs().max()), float((e.double() - ref).abs().max())), flush=True)
        print('   compiled', g.flatten()[:4].tolist(), 'eager', e.flatten()[:4].tolist(), 'f64', ref.flatten()[:4].tolist(), flush=True)
    except Exception as ex:
        print('probe', name, 'raised', repr(ex)[:300], flush=True)


t = torch.tensor([1.0, 2.0, 3.0])
v = torch.rand(8) * 1e-3 + 1e-6
probe('pow(0.9, step)', lambda s: torch.pow(0.9, s), t)
probe('pow(0.999, step)', lambda s: torch.pow(0.999, s), t)
probe('sqrt(1-0.999**t)', lambda s: torch.sqrt(1 - torch.pow(0.999, s)), t)
probe('sqrt(v)', lambda a: a.sqrt(), v)
probe('sqrt(v)/b+eps', lambda a: a.sqrt() / 0.0447 + 1e-8, v)
probe('reciprocal', lambda a: a.reciprocal(), v)
probe('a/b', lambda a: a / (a * 3.7 + 0.1), v)
probe('lerp', lambda a: torch.lerp(a, a * 2 - 1, 0.1), v)
print('probes done', flush=True)
