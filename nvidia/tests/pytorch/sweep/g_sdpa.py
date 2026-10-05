"""scaled_dot_product_attention, each backend forced in turn (flash, memory
efficient, cuDNN, math), forward and backward, over dtypes, masks, causal,
grouped queries, odd lengths and head sizes. Unsupported combinations are not
listed: a backend that refuses a configuration is PyTorch's choice, the same
on a real GPU."""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from harness import cases, check, rnd, have

L = 'sdpa'
B = SDPBackend


def attn(backend, dtype, b=2, h=4, hk=None, sq=16, sk=None, dim=32, causal=False, mask=None, scale=None):
    sk = sk or sq
    hk = hk or h
    q, k, v = rnd(b, h, sq, dim, seed=1), rnd(b, hk, sk, dim, seed=2), rnd(b, hk, sk, dim, seed=3)
    m = None
    if mask == 'bool':
        m = rnd(b, 1, sq, sk, seed=4) > -0.5
    elif mask == 'float':
        m = rnd(b, h, sq, sk, seed=4)

    def run(d):
        ts = [t.to(dtype).to(d) if d != 'cpu' else t.to(dtype).float() for t in (q, k, v)]
        for t in ts:
            t.requires_grad_()
        kw = dict(is_causal=causal, scale=scale)
        if hk != h:
            kw['enable_gqa'] = True
        mm = None if m is None else (m.to(d) if m.dtype == torch.bool else (m.to(dtype).to(d) if d != 'cpu' else m.to(dtype).float()))
        if d == 'cpu':
            out = F.scaled_dot_product_attention(*ts, attn_mask=mm, **kw)
        else:
            with sdpa_kernel(backend):
                out = F.scaled_dot_product_attention(*ts, attn_mask=mm, **kw)
        (out.float() * rnd(*out.shape, seed=9).to(d)).sum().backward()
        return [out.float(), [t.grad.float() for t in ts]]
    return run


H, BF = torch.float16, torch.bfloat16
CFG = {
    'causal': dict(causal=True),
    'not causal': dict(),
    'odd length 37': dict(sq=37, causal=True),
    'cross attention (3 queries, 21 keys)': dict(sq=3, sk=21),
    'head size 64': dict(dim=64, causal=True),
    'head size 128': dict(dim=128, sq=8, causal=True),
    'grouped queries (8 query heads, 2 key heads)': dict(h=8, hk=2, causal=True),
    'custom scale': dict(scale=0.3, causal=True),
}
for backend, name, tol, extra in (
    (B.FLASH_ATTENTION, 'flash', 2e-2, {}),
    (B.EFFICIENT_ATTENTION, 'efficient', 2e-2, {'bool mask': dict(mask='bool'), 'float bias': dict(mask='float'), 'head size 40 (not a multiple of 8... of 16)': dict(dim=40, causal=True)}),
    (B.CUDNN_ATTENTION, 'cudnn', 2e-2, {}),
    (B.MATH, 'math', 2e-2, {'bool mask': dict(mask='bool'), 'float bias': dict(mask='float')}),
):
    for dt, tag, t in ((H, 'half', tol), (BF, 'bfloat16', 8e-2)):
        table = {k: attn(backend, dt, **v) for k, v in {**CFG, **extra}.items()
                 if not (name in ('efficient', 'cudnn') and v.get('hk'))}
        cases(L, f'sdpa {name} backend, {tag}, forward and backward', table, t, tier='quick' if (dt is H) else 'full')
    if name in ('efficient', 'math'):
        cases(L, f'sdpa {name} backend, float32', {k: attn(backend, torch.float32, **CFG[k]) for k in ('causal', 'not causal')}, 2e-3, tier='full')


def _lse(d):
    # The flash kernel's logsumexp output, which training with checkpointing reads.
    q, k, v = (rnd(2, 4, 16, 32, seed=s).half().to(d) for s in (1, 2, 3))
    if d == 'cpu':
        s = (q.float() @ k.float().transpose(-1, -2)) / (32 ** 0.5)
        s = s.masked_fill(torch.triu(torch.ones(16, 16, dtype=torch.bool), 1), float('-inf'))
        return [torch.logsumexp(s, -1)]
    res = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, True, False)
    return [res[1][..., :16]]


check(L, 'flash attention: the logsumexp it saves for backward', 1e-2, tier='full')(_lse)


def _dropout(d):
    # Dropout inside attention: the output is a mix we cannot compare to the CPU's mask;
    # with dropout 0.5 the output's mean over many heads stays near the undropped one.
    q, k, v = (rnd(1, 32, 64, 32, seed=s).half().to(d) for s in (1, 2, 3))
    if d == 'cpu':
        return torch.tensor([1.0])
    outs = []
    for be in (B.FLASH_ATTENTION, B.EFFICIENT_ATTENTION, B.MATH):
        with sdpa_kernel(be):
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.5)
            ref = F.scaled_dot_product_attention(q, k, v)
        outs.append(float((y.float().mean() - ref.float().mean()).abs() < 0.05) * float(not torch.equal(y, ref)))
    return torch.tensor([min(outs)])


check(L, 'attention with dropout_p=0.5 (flash, efficient, math)', 0, tier='full')(_dropout)


def _jagged(d):
    # A nested (jagged) batch through the fused kernels.
    ts = [rnd(n, 4, 32, seed=n).half() for n in (5, 9, 16)]
    if d == 'cpu':
        return [F.scaled_dot_product_attention(t.float().transpose(0, 1), t.float().transpose(0, 1), t.float().transpose(0, 1)) for t in ts]
    nt = torch.nested.nested_tensor([t.to(d) for t in ts], layout=torch.jagged).transpose(1, 2)
    out = F.scaled_dot_product_attention(nt, nt, nt)
    return [o.float() for o in out.unbind()]


check(L, 'jagged nested tensors through scaled_dot_product_attention', 2e-2, tier='full')(_jagged)

check(L, 'xformers memory_efficient_attention', 2e-2, needs=('xformers',), tier='full')(lambda d: _xformers(d))
check(L, 'flash_attn flash_attn_func, causal, forward and backward', 2e-2, needs=('flash_attn',), tier='full')(lambda d: _flash_attn(d))


def _xformers(d):
    import xformers.ops as xo
    q, k, v = (rnd(2, 16, 4, 32, seed=s).half() for s in (1, 2, 3))
    if d == 'cpu':
        qq, kk, vv = (t.float().transpose(1, 2) for t in (q, k, v))
        return F.scaled_dot_product_attention(qq, kk, vv, is_causal=True).transpose(1, 2)
    return xo.memory_efficient_attention(q.to(d), k.to(d), v.to(d), attn_bias=xo.LowerTriangularMask())


def _flash_attn(d):
    from flash_attn import flash_attn_func
    q, k, v = (rnd(2, 16, 4, 32, seed=s).half() for s in (1, 2, 3))
    if d == 'cpu':
        qq, kk, vv = (t.float().transpose(1, 2) for t in (q, k, v))
        return F.scaled_dot_product_attention(qq, kk, vv, is_causal=True).transpose(1, 2)
    return flash_attn_func(q.to(d), k.to(d), v.to(d), causal=True)
