"""CUDA graphs through PyTorch: capture and replay, make_graphed_callables,
a whole training step in a graph, graphs sharing a memory pool, RNG in a graph."""
import torch
import torch.nn as nn

from harness import check, rnd

L = 'graphs'


def _capture(d, fn, static, warm=True, **kw):
    if d == 'cpu':
        return None
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, **kw):
        out = fn()
    return g, out


def _elementwise(d):
    x = rnd(1000, seed=1).to(d)
    fn = lambda: torch.sin(x) * 2 + x.exp().clamp(max=5)
    if d == 'cpu':
        return [fn(), (lambda: (x.copy_(rnd(1000, seed=2)), fn())[1])()]
    g, out = _capture(d, fn, x)
    g.replay()
    first = out.clone()
    x.copy_(rnd(1000, seed=2).to(d))
    g.replay()
    return [first, out.clone()]


check(L, 'capture and replay: pointwise kernels, new input between replays', 1e-5)(_elementwise)


def _matmul_conv(d):
    x = rnd(2, 3, 12, 12, seed=1).to(d)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 6, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(6 * 144, 10)).to(d)
    with torch.no_grad():
        fn = lambda: net(x)
        if d == 'cpu':
            first = fn()
            x.copy_(rnd(2, 3, 12, 12, seed=2))
            return [first, fn()]
        g, out = _capture(d, fn, x)
        g.replay()
        first = out.clone()
        x.copy_(rnd(2, 3, 12, 12, seed=2).to(d))
        g.replay()
    return [first, out.clone()]


check(L, 'capture and replay: convolution (cuDNN) and linear (cuBLAS)', 1e-4)(_matmul_conv)


def _train_step(d):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 3)).to(d)
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    x, y = rnd(8, 10, seed=1).to(d), rnd(8, 3, seed=2).to(d)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = (net(x) - y).square().mean()
        loss.backward()
        opt.step()
        return loss
    if d == 'cpu':
        for _ in range(3 + 2):
            step()
        return [list(net.parameters())]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        step()
    for _ in range(2 - 1):
        g.replay()
    g.replay()
    # three warm-up steps + the captured one's capture (not executed) + two replays = 5 steps
    return [list(net.parameters())]


check(L, 'a training step captured whole (forward, backward, SGD with momentum) and replayed', 2e-3)(_train_step)


def _adam_capturable(d):
    torch.manual_seed(0)
    net = nn.Linear(10, 3).to(d)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, capturable=d != 'cpu')
    x, y = rnd(8, 10, seed=1).to(d), rnd(8, 3, seed=2).to(d)

    def step():
        opt.zero_grad(set_to_none=True)
        (net(x) - y).square().mean().backward()
        opt.step()
    if d == 'cpu':
        for _ in range(5):
            step()
        return list(net.parameters())
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        step()
    g.replay()
    g.replay()
    return list(net.parameters())


check(L, 'Adam with capturable=True inside a graph', 2e-3)(_adam_capturable)


def _graphed_callables(d):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(10, 20), nn.GELU(), nn.Linear(20, 3)).to(d)
    x = rnd(8, 10, seed=1).to(d).requires_grad_()
    if d != 'cpu':
        net = torch.cuda.make_graphed_callables(net, (x,))
    outs = []
    for i in range(2):
        xi = rnd(8, 10, seed=3 + i).to(d).requires_grad_()
        y = net(xi)
        (y * rnd(8, 3, seed=7).to(d)).sum().backward()
        outs += [y.detach().clone(), xi.grad.clone()]
    return [outs]


check(L, 'make_graphed_callables: forward and backward graphs', 1e-4)(_graphed_callables)


def _shared_pool(d):
    x = rnd(500, seed=1).to(d)
    f1, f2 = (lambda: x * 2), None
    if d == 'cpu':
        return [x * 2, x * 2 + 1]
    g1 = torch.cuda.CUDAGraph()
    g2 = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g1):
        a = x * 2
    with torch.cuda.graph(g2, pool=g1.pool()):
        b = a + 1
    g1.replay()
    g2.replay()
    return [a.clone(), b.clone()]


check(L, 'two graphs sharing a memory pool, one reading the other\'s output', 1e-6)(_shared_pool)


def _rng(d):
    # Dropout in a graph draws a new mask on every replay (the generator's offset
    # is a graph input); compare the kept fraction, not the mask.
    x = torch.ones(20000, device=d)
    if d == 'cpu':
        return torch.tensor([1.0])
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        y = torch.nn.functional.dropout(x, 0.5)
    g.replay()
    a = y.clone()
    g.replay()
    b = y.clone()
    ok = abs(float((a > 0).float().mean()) - 0.5) < 0.03 and abs(float((b > 0).float().mean()) - 0.5) < 0.03 and not torch.equal(a, b)
    return torch.tensor([float(ok)])


check(L, 'dropout in a graph: a new mask on every replay', 0)(_rng)


def _multi_stream(d):
    x = rnd(4000, seed=1).to(d)

    def fn():
        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
        s1.wait_stream(torch.cuda.current_stream())
        s2.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s1):
            a = x.sin()
        with torch.cuda.stream(s2):
            b = x.cos()
        torch.cuda.current_stream().wait_stream(s1)
        torch.cuda.current_stream().wait_stream(s2)
        return a * b
    if d == 'cpu':
        return x.sin() * x.cos()
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        out = fn()
    g.replay()
    return out.clone()


check(L, 'a graph forking into two streams and joining', 1e-5)(_multi_stream)


def _graph_sort_reduce(d):
    x = rnd(3, 2000, seed=1).to(d)
    fn = lambda: [x.sort(1).values, x.sum(1), x.softmax(1), x.cumsum(1), x.topk(5).values]
    if d == 'cpu':
        return fn()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = fn()
    g.replay()
    return [o.clone() for o in outs]


check(L, 'a graph of sort, reductions, softmax, scan and topk (library kernels)', 1e-3, tier='full')(_graph_sort_reduce)
