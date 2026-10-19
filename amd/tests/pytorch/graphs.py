"""CUDA graphs (HIP graphs) in PyTorch's ROCm build, unmodified, on a simulated AMD GPU.

vLLM's decode step, and every "reduce-overhead" torch.compile, is a stream
capture: PyTorch records a model's launches, copies and collectives into a
graph once and replays it. The kernels, copies and event waits recorded go
through hipStreamBeginCapture / hipStreamEndCapture and hipGraphLaunch, and
the allocator hands out the graph's memory from a pool of its own. Each check
captures something, changes the inputs it was captured with, replays, and
compares with what the CPU computes (a replay that merely re-ran stale values
would fail, the inputs change between replays). One line each: "ok <name>",
or "FAIL <name>: <why>".
"""
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def check(name, f):
    try:
        why = f()
        print(f"ok {name}" if why is None else f"FAIL {name}: {why}", flush=True)
    except Exception as e:  # a refused call or kernel shows up here, and says which
        print(f"FAIL {name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)


def close(got, want, tol):
    d = float((got.detach().float().cpu() - want.detach().float().cpu()).abs().max())
    return None if d <= tol else f"max difference {d:g}, allowed {tol:g}"


def capture(fn, warm=True, **kw):
    """fn recorded into a graph (after a warm-up on a side stream, as PyTorch's
    documentation does it); returns the graph and what fn returned."""
    if warm:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, **kw):
        out = fn()
    return g, out


def replay(g):
    g.replay()
    torch.cuda.synchronize()


def pointwise():
    x = torch.randn(4096, device='cuda')
    g, y = capture(lambda: x * 2 + 1)
    for _ in range(2):  # a kernel node reads the argument it was captured with
        v = torch.randn(4096)
        x.copy_(v)
        replay(g)
        if (why := close(y, v * 2 + 1, 1e-6)):
            return why


def matmul():
    a = torch.randn(64, 64, device='cuda')
    b = torch.randn(64, 64, device='cuda')
    g, c = capture(lambda: torch.relu(a @ b))
    v = torch.randn(64, 64)
    a.copy_(v)
    replay(g)
    return close(c, torch.relu(v @ b.cpu()), 1e-3)


def half_linear():
    a = torch.randn(48, 96, device='cuda', dtype=torch.float16)
    w = torch.randn(40, 96, device='cuda', dtype=torch.float16)
    bias = torch.randn(40, device='cuda', dtype=torch.float16)
    g, c = capture(lambda: F.linear(a, w, bias))
    v = torch.randn(48, 96).half()
    a.copy_(v)
    replay(g)
    return close(c, v.float() @ w.cpu().float().T + bias.cpu().float(), 0.25)


def shared_pool():
    x = torch.randn(1024, device='cuda')
    pool = torch.cuda.graph_pool_handle()
    graphs = []
    for n in (1, 2, 4, 8):
        def f(n=n):
            y = x
            for _ in range(n):
                y = y * 1.01 + 0.5
            return y
        graphs.append((n,) + capture(f, pool=pool))
    v = torch.randn(1024)
    x.copy_(v)
    for n, g, y in graphs:
        replay(g)
        want = v
        for _ in range(n):
            want = want * 1.01 + 0.5
        if (why := close(y, want, 1e-4)):
            return f"graph of {n}: {why}"


def random_numbers():
    x = torch.ones(4096, device='cuda')
    g, y = capture(lambda: F.dropout(x, 0.5, True))
    replay(g)
    a = y.clone()
    replay(g)
    b = y.clone()
    kept = float((a > 0).float().mean())
    if torch.equal(a, b):
        return "two replays drew the same numbers"
    if not 0.4 < kept < 0.6 or not set(a.unique().tolist()) <= {0.0, 2.0}:
        return f"dropout kept {kept:.2f} of the elements, values {sorted(set(a.unique().tolist()))}"


def streams():
    x = torch.randn(1024, device='cuda')

    def f():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        a = x * 2
        with torch.cuda.stream(side):
            b = x + 3
        torch.cuda.current_stream().wait_stream(side)
        return a + b
    g, y = capture(f, warm=False)
    v = torch.randn(1024)
    x.copy_(v)
    replay(g)
    return close(y, v * 3 + 3, 1e-5)


def events():
    x = torch.randn(1024, device='cuda')
    ev = torch.cuda.Event()

    def f():
        side = torch.cuda.Stream()
        a = x * 2
        ev.record()
        with torch.cuda.stream(side):
            side.wait_event(ev)
            b = a + 1
        torch.cuda.current_stream().wait_stream(side)
        return b
    g, y = capture(f, warm=False)
    v = torch.randn(1024)
    x.copy_(v)
    replay(g)
    return close(y, v * 2 + 1, 1e-6)


def host_copies():
    src = torch.arange(256, dtype=torch.float32).pin_memory()
    dst = torch.empty(256, device='cuda')
    back = torch.empty(256, dtype=torch.float32).pin_memory()

    def f():
        dst.copy_(src, non_blocking=True)
        dst.mul_(2)
        back.copy_(dst, non_blocking=True)
        return dst
    g, _ = capture(f)
    src.fill_(3)  # a copy node reads the host memory at replay, not at capture
    replay(g)
    return None if bool((back == 6).all()) else f"the host got {back[:3].tolist()}, wanted 6"


def allocations():
    x = torch.randn(1 << 16, device='cuda')

    def f():
        t = x * 2
        u = t + 1
        del t  # freed and reusable inside the capture
        v = torch.empty(1 << 18, device='cuda')
        v[:1 << 16] = u
        return v
    g, out = capture(f)
    v = torch.randn(1 << 16)
    x.copy_(v)
    replay(g)
    return close(out[:1 << 16], v * 2 + 1, 1e-6)


def error_modes():
    x = torch.randn(64, device='cuda')
    for mode in ("global", "thread_local", "relaxed"):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, capture_error_mode=mode):
            y = x + 1
        replay(g)
        if (why := close(y, x + 1, 0)):
            return f"{mode}: {why}"


def sync_in_capture():
    x = torch.randn(64, device='cuda')
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            x.cpu()  # a device-to-host copy that waits is illegal during capture
    except Exception:
        torch.cuda.synchronize()
        return close(x + 1, x.cpu() + 1, 0)  # and the device carries on afterwards
    return "x.cpu() during capture did not raise"


def operators():
    x = torch.randn(8, 128, device='cuda', dtype=torch.float16)
    idx = torch.randint(0, 1000, (8, 16), device='cuda')
    emb = nn.Embedding(1000, 128).cuda().half()
    ln = nn.LayerNorm(128).cuda().half()

    def f(x, idx, emb, ln):
        h = ln(emb(idx).sum(1) + x).float()
        p = F.softmax(h, -1)
        v, i = torch.topk(p, 5, dim=-1)
        return torch.stack([v.sum(), i.sum().float(), torch.sort(h, -1)[0].sum(), torch.cumsum(p, -1).sum(),
                            h.argmax(-1).sum().float()])
    g, out = capture(lambda: f(x, idx, emb, ln))
    v = torch.randn(8, 128).half()
    x.copy_(v)
    replay(g)
    want = f(v, idx.cpu(), emb.cpu().float(), ln.cpu().float())
    return close(out, want, 0.05 * max(1.0, float(want.abs().max())))


def attention():
    q = torch.randn(1, 4, 32, 64, device='cuda', dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    g, o = capture(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
    nq = torch.randn(1, 4, 32, 64).half()
    q.copy_(nq)
    replay(g)
    return close(o, F.scaled_dot_product_attention(nq.float(), k.cpu().float(), v.cpu().float(), is_causal=True), 5e-3)


def convolution():
    c = nn.Conv2d(3, 8, 3, padding=1).cuda()
    x = torch.randn(2, 3, 16, 16, device='cuda')
    g, y = capture(lambda: c(x))
    v = torch.randn(2, 3, 16, 16)
    x.copy_(v)
    replay(g)
    return close(y, c.cpu()(v), 1e-4)


def graphed_callables():
    # Forward and backward each captured, and the autograd engine run on the
    # replays: gradients of a Linear against the CPU's.
    torch.manual_seed(1)
    m = nn.Linear(32, 16).cuda()
    x = torch.randn(8, 32, device='cuda', requires_grad=True)
    gm = torch.cuda.make_graphed_callables(m, (x,))
    xv = torch.randn(8, 32)
    out = gm(xv.cuda().requires_grad_())
    out.pow(2).sum().backward()
    w_grad = m.weight.grad.clone()
    mc = nn.Linear(32, 16)
    mc.load_state_dict({k: v.cpu() for k, v in m.state_dict().items()})
    mc(xv).pow(2).sum().backward()
    return close(out, mc(xv), 1e-4) or close(w_grad, mc.weight.grad, 1e-3)


class Decoder(nn.Module):
    """A small GPT with a KV cache of fixed size: one decode step is the same
    kernels every token, the position and the token the only things that change."""

    def __init__(self, vocab=96, d=64, heads=4, layers=2, ctx=16):
        super().__init__()
        self.h, self.d, self.ctx = heads, d, ctx
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.qkv = nn.ModuleList(nn.Linear(d, 3 * d) for _ in range(layers))
        self.proj = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for _ in range(layers))
        self.n1 = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.n2 = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.out = nn.Linear(d, vocab)

    def step(self, token, pos, kcache, vcache):
        """token [1], pos [1] (int64 tensors): logits [vocab]; writes the caches at pos."""
        x = self.emb(token) + self.pos(pos)
        mask = (torch.arange(self.ctx, device=token.device) <= pos).view(1, 1, -1)
        for i in range(len(self.qkv)):
            q, k, v = self.qkv[i](self.n1[i](x)).chunk(3, -1)
            kcache[i].index_copy_(0, pos, k.view(1, self.h, -1))
            vcache[i].index_copy_(0, pos, v.view(1, self.h, -1))
            s = torch.einsum('hd,thd->ht', q.view(self.h, -1), kcache[i]) * (self.d // self.h) ** -0.5
            s = s.masked_fill(~mask.view(1, -1), float('-inf'))
            a = torch.einsum('ht,thd->hd', torch.softmax(s, -1), vcache[i]).reshape(1, -1)
            x = x + self.proj[i](a)
            x = x + self.mlp[i](self.n2[i](x))
        return self.out(x)[0]


def decode_loop():
    torch.manual_seed(2)
    model = Decoder().eval()
    prompt, steps = [5, 17, 42], 10

    def caches(dev):
        shape = (model.ctx, model.h, model.d // model.h)
        return [torch.zeros(shape, device=dev) for _ in range(2)], [torch.zeros(shape, device=dev) for _ in range(2)]

    def generate(step):
        """Greedy: the prompt fed in, then each token the model chose."""
        toks, logits = list(prompt), []
        for p in range(steps):
            lg = step(toks[p], p)
            if p >= len(prompt) - 1:
                logits.append(lg)
                toks.append(int(lg.argmax()))
        return toks, torch.stack(logits)

    with torch.no_grad():
        kc, vc = caches('cpu')
        toks, want = generate(lambda t, p: model.step(torch.tensor([t]), torch.tensor([p]), kc, vc))
        # One decode step captured, then replayed for each token with the
        # token and the position written into its inputs.
        model.cuda()
        kg, vg = caches('cuda')
        token = torch.zeros(1, dtype=torch.long, device='cuda')
        pos = torch.zeros(1, dtype=torch.long, device='cuda')
        g, logits = capture(lambda: model.step(token, pos, kg, vg))
        for t in kg + vg:  # the warm-up and the capture itself wrote the caches
            t.zero_()

        def replayed(t, p):
            token.fill_(t)
            pos.fill_(p)
            replay(g)
            return logits.clone()
        gtoks, got = generate(replayed)
    if gtoks != toks:
        return f"tokens {gtoks}, the CPU's {toks}"
    return close(got, want, 2e-3)


def kept_graph():
    # keep_graph=True: the graph stays for the program to look at (a node
    # count, a dot dump) and is instantiated on request.
    x = torch.randn(256, device='cuda')
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        y = (x * 2 + 1).relu()
    raw = g.raw_cuda_graph()
    if not raw:
        return "no raw graph handle"
    g.instantiate()
    path = os.path.join(tempfile.mkdtemp(), "graph.dot")
    g.enable_debug_mode()
    g.debug_dump(path)
    if not os.path.exists(path) or "digraph" not in open(path).read():
        return "debug_dump wrote no graph"
    v = torch.randn(256)
    x.copy_(v)
    replay(g)
    return close(y, (v * 2 + 1).relu(), 1e-6)


for name, fn in [
    ("a kernel graph replayed with new inputs", pointwise),
    ("a hipBLASLt GEMM in a graph", matmul),
    ("an fp16 linear with bias in a graph", half_linear),
    ("four graphs sharing one memory pool", shared_pool),
    ("random numbers in a graph differ per replay", random_numbers),
    ("a side stream forked and joined in a capture", streams),
    ("an event recorded and waited on in a capture", events),
    ("host-to-device and device-to-host copy nodes", host_copies),
    ("allocations and frees inside a capture", allocations),
    ("every capture error mode", error_modes),
    ("a synchronising call during capture raises", sync_in_capture),
    ("embedding, layernorm, softmax, topk, sort, cumsum, argmax", operators),
    ("causal attention in a graph", attention),
    ("a convolution (MIOpen) in a graph", convolution),
    ("make_graphed_callables, forward and backward", graphed_callables),
    ("a KV-cache decode step replayed per token", decode_loop),
    ("a kept graph: raw handle, instantiate, dot dump", kept_graph),
]:
    check(name, fn)
