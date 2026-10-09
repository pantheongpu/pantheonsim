"""Tensor operators: indexing, scatter, sorting, scans, statistics, shape and
type manipulation, and random numbers."""
import torch
import torch.nn.functional as F

from harness import cases, check, rnd, rint

L = 'ops'
v = rnd(1000, seed=1)
m = rnd(32, 40, seed=2)
t3 = rnd(4, 5, 6, seed=3)
iv = rint(-50, 50, 1000, seed=4)
im = rint(0, 8, 12, 10, seed=5)

cases(L, 'indexing and gather/scatter', {
    'index_select': lambda d: m.to(d).index_select(1, torch.tensor([3, 1, 39, 7], device=d)),
    'advanced indexing': lambda d: t3.to(d)[torch.tensor([0, 2, 3], device=d)][:, torch.tensor([1, 4], device=d)],
    'boolean mask select': lambda d: m.to(d)[m.to(d) > 0.5],
    'masked_fill and where': lambda d: torch.where(m.to(d) > 0, m.to(d).masked_fill(m.to(d) > 1, -1), torch.zeros_like(m.to(d))),
    'masked_scatter': lambda d: torch.zeros(32, 40, device=d).masked_scatter(m.to(d) > 0, v.to(d)),
    'index_put accumulate': lambda d: torch.zeros(10, device=d).index_put_((im.to(d)[:, 0],), v.to(d)[:12], accumulate=True),
    'index_add and index_copy': lambda d: torch.zeros(10, 4, device=d).index_add_(0, im.to(d)[:, 0], rnd(12, 4).to(d)).index_copy_(0, torch.arange(3, device=d), rnd(3, 4, seed=9).to(d)),
    'index_fill': lambda d: torch.zeros(10, 4, device=d).index_fill_(0, torch.tensor([1, 5], device=d), 3.0),
    'gather': lambda d: torch.gather(m.to(d), 1, rint(0, 40, 32, 7, seed=6).to(d)),
    'scatter': lambda d: torch.zeros(32, 40, device=d).scatter(1, torch.arange(32 * 7).reshape(32, 7).to(d) % 40, 2.5),
    'scatter_reduce sum, prod, mean, amax, amin': lambda d: [torch.ones(8, device=d).scatter_reduce(0, (iv.to(d) % 8).abs(), v.to(d), r, include_self=False) for r in ('sum', 'prod', 'mean', 'amax', 'amin')],
    'scatter_add 2d': lambda d: torch.zeros(10, 6, device=d).scatter_add(0, im.to(d)[:, :6], rnd(12, 6, seed=8).to(d)),
    'take, put': lambda d: [m.to(d).take(torch.tensor([5, 77, 1000], device=d)), torch.zeros(20, device=d).put_(torch.tensor([3, 4, 3], device=d), torch.tensor([1., 2., 3.], device=d), accumulate=True)],
    'take_along_dim': lambda d: torch.take_along_dim(m.to(d), m.to(d).argsort(1)[:, :5], 1),
    'nonzero and argwhere': lambda d: [(m.to(d) > 1).nonzero(), torch.argwhere(m.to(d) > 1.5)],
    'masked_select': lambda d: v.to(d).masked_select(v.to(d) > 0),
    'bucketize and searchsorted': lambda d: [torch.bucketize(v.to(d), torch.linspace(-2, 2, 9, device=d)), torch.searchsorted(torch.linspace(-3, 3, 50, device=d), v.to(d), right=True)],
    'embedding lookup (index_select backward path)': lambda d: F.embedding(im.to(d), rnd(8, 5, seed=3).to(d)),
}, 1e-6)

cases(L, 'sorting and selection', {
    'sort stable, both directions': lambda d: [torch.sort(iv.to(d), stable=True), torch.sort(iv.to(d), descending=True, stable=True)],
    'sort 2d along each dim': lambda d: [torch.sort(m.to(d), dim=0).values, torch.sort(m.to(d), dim=1, descending=True).indices],
    'sort a large row (segmented sort)': lambda d: torch.sort(rnd(4, 5000, seed=3).to(d), dim=1).values,
    'argsort': lambda d: iv.to(d).argsort(stable=True),
    'topk largest and smallest': lambda d: [torch.topk(m.to(d), 5, dim=1), torch.topk(m.to(d), 3, dim=0, largest=False)],
    'kthvalue': lambda d: torch.kthvalue(m.to(d), 7, dim=1),
    'median and nanmedian': lambda d: [torch.median(m.to(d), dim=1), torch.nanmedian(v.to(d))],
    'mode': lambda d: torch.mode(torch.tensor([[1, 2, 2, 3], [5, 5, 5, 1], [7, 4, 4, 4]]).to(d), dim=1),
    'msort, unique, unique_consecutive': lambda d: [torch.msort(m.to(d)), torch.unique(iv.to(d), return_inverse=True, return_counts=True), torch.unique_consecutive(torch.sort(iv.to(d)).values, return_counts=True)],
    'unique with dim': lambda d: torch.unique(im.to(d), dim=0),
    'cummax, cummin': lambda d: [torch.cummax(m.to(d), 1), torch.cummin(m.to(d), 0)],
    'logcumsumexp, cumprod': lambda d: [torch.logcumsumexp(m.to(d), 1), torch.cumprod(rnd(8, 20, seed=2, scale=0.1).to(d) + 1, 1)],
    'cumsum on integers': lambda d: torch.cumsum(iv.to(d), 0),
    'histc, bincount, histogram': lambda d: [torch.histc(v.to(d), 20, -3, 3), torch.bincount(iv.to(d).abs(), minlength=60), torch.histogram(v, bins=10)[0].to(d) if d == 'cpu' else torch.histc(v.to(d), 10, v.min().item(), v.max().item())],
    'diff, flip, roll, rot90': lambda d: [torch.diff(m.to(d), dim=1), torch.flip(t3.to(d), (0, 2)), torch.roll(m.to(d), (3, -5), (0, 1)), torch.rot90(t3.to(d), 1, (1, 2))],
    'tril, triu, diag, diag_embed': lambda d: [torch.tril(m.to(d), 2), torch.triu(m.to(d), -1), torch.diag(m.to(d)[0, :5]), torch.diag_embed(t3.to(d))],
}, 1e-5)

cases(L, 'reductions and statistics', {
    'sum, prod, mean along dims': lambda d: [t3.to(d).sum((0, 2)), (t3.to(d) * 0.3 + 1).prod(1), t3.to(d).mean(-1, keepdim=True)],
    'var, std, var_mean, std_mean': lambda d: [t3.to(d).var(1), t3.to(d).std(2, correction=0), torch.var_mean(t3.to(d), (0, 1)), torch.std_mean(t3.to(d))],
    'amax, amin, aminmax, max with indices': lambda d: [t3.to(d).amax((0, 2)), t3.to(d).amin(1), torch.aminmax(t3.to(d)), t3.to(d).max(1)],
    'norm, vector_norm orders': lambda d: [torch.linalg.vector_norm(t3.to(d), o, dim=1) for o in (0, 1, 2, 3.5, float('inf'), -float('inf'))],
    'logsumexp, softmax along dims': lambda d: [torch.logsumexp(t3.to(d), (0, 1)), torch.softmax(t3.to(d), 0)],
    'nansum, nanmean, nan_to_num': lambda d: [torch.nansum(torch.where(v.to(d) > 1, torch.nan, v.to(d))), torch.nanmean(torch.where(v.to(d) > 1, torch.nan, v.to(d))), torch.nan_to_num(torch.where(v.to(d) > 1, torch.nan, v.to(d)))],
    'any, all, count_nonzero': lambda d: [(t3.to(d) > 0).any(1), (t3.to(d) > -5).all(), torch.count_nonzero(t3.to(d) > 0, dim=1)],
    'quantile and nanquantile': lambda d: [torch.quantile(v.to(d), torch.tensor([0.1, 0.5, 0.9], device=d)), torch.nanquantile(m.to(d), 0.25, dim=1)],
    'corrcoef and cov': lambda d: [torch.corrcoef(m.to(d)[:6]), torch.cov(m.to(d)[:6])],
    'sum of integers and bools': lambda d: [iv.to(d).sum(), (iv.to(d) > 0).sum(), iv.to(d).to(torch.int8).sum(dtype=torch.int64), iv.to(d).prod(dtype=torch.int64) % 1000],
    'large reduction (multi-block)': lambda d: [rnd(300000, seed=2).to(d).sum(), rnd(300000, seed=2).to(d).amax()],
    'dist, renorm, trace': lambda d: [torch.dist(m.to(d), m.to(d) * 1.1), torch.renorm(m.to(d), 2, 0, 5.0), torch.trace(m.to(d))],
}, 1e-4)

cases(L, 'products and contractions', {
    'einsum (several)': lambda d: [torch.einsum('bij,bjk->bik', t3.to(d), rnd(4, 6, 3, seed=2).to(d)), torch.einsum('ij,ij->', m.to(d), m.to(d)), torch.einsum('bij->bji', t3.to(d)), torch.einsum('i,j->ij', v.to(d)[:7], v.to(d)[:9])],
    'tensordot, kron, outer, cross': lambda d: [torch.tensordot(t3.to(d), rnd(5, 6, 3, seed=4).to(d), dims=([1, 2], [0, 1])), torch.kron(m.to(d)[:3, :3], m.to(d)[:2, :4]), torch.outer(v.to(d)[:6], v.to(d)[:9]), torch.linalg.cross(rnd(5, 3, seed=2).to(d), rnd(5, 3, seed=3).to(d))],
    'addmm, baddbmm, addbmm, addmv, addr': lambda d: [torch.addmm(m.to(d)[:4, :5], m.to(d)[:4, :6], m.to(d)[:6, :5], beta=0.5, alpha=2), torch.baddbmm(t3.to(d)[:, :, :3], t3.to(d), rnd(4, 6, 3, seed=2).to(d)), torch.addbmm(rnd(5, 3, seed=2).to(d), t3.to(d), rnd(4, 6, 3, seed=2).to(d)), torch.addmv(v.to(d)[:5], t3.to(d)[0], v.to(d)[:6]), torch.addr(m.to(d)[:3, :4], v.to(d)[:3], v.to(d)[:4])],
    'matmul broadcasting': lambda d: [torch.matmul(t3.to(d), rnd(6, 2, seed=2).to(d)), torch.matmul(rnd(3, 1, 5, 6, seed=2).to(d), rnd(4, 6, 2, seed=5).to(d)), torch.matmul(v.to(d)[:6], t3.to(d).transpose(1, 2))],
    'integer matmul (torch._int_mm)': lambda d: (torch._int_mm(rint(-5, 5, 32, 32, seed=2).to(d).to(torch.int8), rint(-5, 5, 32, 32, seed=3).to(d).to(torch.int8)) if d != 'cpu' else (rint(-5, 5, 32, 32, seed=2) @ rint(-5, 5, 32, 32, seed=3)).int()),
    'chain_matmul and multi_dot': lambda d: torch.linalg.multi_dot([m.to(d)[:5, :8], m.to(d)[:8, :6], m.to(d)[:6, :3]]),
    'bilinear and trapezoid': lambda d: [F.bilinear(m.to(d)[:4, :5], m.to(d)[:4, 5:11], rnd(3, 5, 6, seed=2).to(d)), torch.trapezoid(m.to(d), dim=1), torch.cumulative_trapezoid(m.to(d), dim=1)],
}, 1e-4)

cases(L, 'elementwise math', {
    'exp, log, sqrt family': lambda d: [torch.exp(v.to(d)), torch.log(v.to(d).abs() + 1e-3), torch.sqrt(v.to(d).abs()), torch.rsqrt(v.to(d).abs() + 0.1), torch.expm1(v.to(d)), torch.log1p(v.to(d).abs()), torch.log2(v.to(d).abs() + 1e-3), torch.log10(v.to(d).abs() + 1e-3), torch.exp2(v.to(d))],
    'trig and hyperbolic': lambda d: [f(v.to(d)) for f in (torch.sin, torch.cos, torch.tan, torch.tanh, torch.sinh, torch.cosh, torch.atan, torch.asinh)] + [torch.asin(torch.tanh(v.to(d))), torch.acos(torch.tanh(v.to(d))), torch.atanh(torch.tanh(v.to(d)) * 0.9), torch.atan2(v.to(d), v.to(d).flip(0)), torch.acosh(v.to(d).abs() + 1.1)],
    'special functions': lambda d: [torch.erf(v.to(d)), torch.erfc(v.to(d)), torch.erfinv(torch.tanh(v.to(d)) * 0.9), torch.lgamma(v.to(d).abs() + 0.1), torch.digamma(v.to(d).abs() + 0.5), torch.sigmoid(v.to(d)), torch.special.i0(v.to(d)), torch.special.i1(v.to(d)), torch.special.ndtri(torch.sigmoid(v.to(d)) * 0.98 + 0.01), torch.special.log_ndtr(v.to(d)), torch.special.xlogy(v.to(d).abs(), v.to(d).abs() + 1), torch.special.entr(torch.sigmoid(v.to(d))), torch.special.sinc(v.to(d)), torch.special.bessel_j0(v.to(d)), torch.special.bessel_j1(v.to(d)), torch.special.erfcx(v.to(d))],
    'zeta, polygamma, igamma': lambda d: [torch.special.zeta(v.to(d).abs() + 2, torch.full_like(v.to(d), 1.5)), torch.polygamma(1, v.to(d).abs() + 0.5), torch.igamma(v.to(d).abs() + 0.5, v.to(d).abs() + 1), torch.igammac(v.to(d).abs() + 0.5, v.to(d).abs() + 1)],
    'power, pow, float_power, hypot': lambda d: [torch.pow(v.to(d).abs(), 2.5), torch.pow(2.0, v.to(d)), torch.float_power(v.to(d).abs(), 1.5), torch.hypot(v.to(d), v.to(d).flip(0)), torch.pow(v.to(d), 3), torch.pow(v.to(d).abs() + 0.1, -0.5)],
    'rounding': lambda d: [torch.round(v.to(d) * 10) / 10, torch.round(v.to(d), decimals=1), torch.floor(v.to(d)), torch.ceil(v.to(d)), torch.trunc(v.to(d)), torch.frac(v.to(d)), torch.sign(v.to(d)), torch.sgn(v.to(d))],
    'fmod, remainder, floor_divide, div modes': lambda d: [torch.fmod(v.to(d), 0.7), torch.remainder(v.to(d), 0.7), torch.floor_divide(v.to(d), 0.7), torch.div(v.to(d), 0.7, rounding_mode='trunc'), torch.div(v.to(d), 0.7, rounding_mode='floor'), torch.fmod(iv.to(d), 7), torch.remainder(iv.to(d), -7), torch.floor_divide(iv.to(d), -7)],
    'clamp, lerp, addcmul, addcdiv': lambda d: [torch.clamp(v.to(d), -0.5, 0.7), torch.clamp(v.to(d), min=0.1), torch.lerp(v.to(d), v.to(d).flip(0), 0.3), torch.addcmul(v.to(d), v.to(d), v.to(d).flip(0), value=0.5), torch.addcdiv(v.to(d), v.to(d), v.to(d).abs() + 1, value=0.5)],
    'maximum, minimum, fmax, fmin, copysign, nextafter, ldexp, frexp': lambda d: [torch.maximum(v.to(d), v.to(d).flip(0)), torch.minimum(v.to(d), v.to(d).flip(0)), torch.fmax(v.to(d), v.to(d).flip(0)), torch.fmin(v.to(d), v.to(d).flip(0)), torch.copysign(v.to(d), v.to(d).flip(0)), torch.nextafter(v.to(d), v.to(d).flip(0)), torch.ldexp(v.to(d), torch.arange(1000, device=d) % 5), torch.frexp(v.to(d))],
    'isnan, isinf, isfinite, signbit, logical ops': lambda d: [torch.isnan(v.to(d) / 0 * 0), torch.isinf(v.to(d) / 0), torch.isfinite(v.to(d) / (v.to(d) > 0)), torch.signbit(v.to(d)), torch.logical_xor(v.to(d) > 0, v.to(d).flip(0) > 0), torch.logical_not(v.to(d) > 0)],
    'comparisons and isclose': lambda d: [v.to(d) < 0.1, v.to(d) >= v.to(d).flip(0), torch.isclose(v.to(d), v.to(d) + 1e-9), torch.allclose(v.to(d), v.to(d)) * torch.tensor(1)],
    'complex arithmetic': lambda d: [torch.view_as_real(f(torch.complex(v.to(d), v.to(d).flip(0)))) if f.__name__ != 'abs' else f(torch.complex(v.to(d), v.to(d).flip(0))) for f in (torch.exp, torch.log, torch.sqrt, torch.sin, torch.abs, torch.conj_physical, torch.reciprocal, torch.square)] + [torch.angle(torch.complex(v.to(d), v.to(d).flip(0))), torch.view_as_real(torch.polar(v.to(d).abs(), v.to(d)))],
}, 1e-3)

cases(L, 'integer, bool and bit operations', {
    'int arithmetic in every width': lambda d: [(iv.to(d).to(t) * 3 + 7 - iv.to(d).flip(0).to(t)) for t in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)],
    'bitwise and shifts': lambda d: [iv.to(d) & 0xF0, iv.to(d) | 5, iv.to(d) ^ 0x55, ~iv.to(d), iv.to(d) << 3, iv.to(d) >> 2, torch.bitwise_left_shift(iv.to(d).int(), 4), torch.bitwise_right_shift(iv.to(d).to(torch.int16), 1)],
    'integer division and modulo by variable': lambda d: [iv.to(d) // (iv.to(d).flip(0).abs() + 1), iv.to(d) % (iv.to(d).flip(0).abs() + 1)],
    'type promotion': lambda d: [iv.to(d) + 1.5, iv.to(d).to(torch.int8) + torch.tensor(300, device=d, dtype=torch.int16), v.to(d).half() + iv.to(d).to(torch.int32), v.to(d).bfloat16() * v.to(d).half(), (v.to(d) > 0) + (v.to(d).flip(0) > 0)],
    'casts between every dtype': lambda d: [v.to(d).to(t).to(torch.float32) for t in (torch.float16, torch.bfloat16, torch.float64, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool, torch.float8_e4m3fn, torch.float8_e5m2)],
    'gcd, lcm, popcount-style ops': lambda d: [torch.gcd(iv.to(d).abs() + 1, iv.to(d).flip(0).abs() + 1), torch.lcm(iv.to(d).abs() % 50 + 1, iv.to(d).flip(0).abs() % 50 + 1)],
    'arange, linspace, logspace, eye, full, tri': lambda d: [torch.arange(0, 10, 0.37, device=d), torch.linspace(-3, 5, 77, device=d), torch.logspace(0, 3, 9, device=d), torch.eye(5, 7, device=d), torch.full((3, 4), 2.5, device=d), torch.tril_indices(5, 6, device=d), torch.arange(1000, dtype=torch.int64, device=d) * 3],
    'cat, stack, split, chunk, repeat, tile, expand': lambda d: [torch.cat([m.to(d), m.to(d) * 2], 1), torch.stack([m.to(d)] * 3, 2), torch.cat(torch.split(m.to(d), 8, 1), 0), torch.cat(torch.chunk(m.to(d), 4, 0), 1), m.to(d).repeat(2, 3), torch.tile(m.to(d)[:3, :3], (2, 2)), m.to(d)[:1].expand(5, 40), torch.repeat_interleave(v.to(d)[:5], torch.tensor([1, 2, 0, 3, 1], device=d))],
    'strided views, transposes, contiguous': lambda d: [m.to(d).t().contiguous(), m.to(d)[::3, 1::2].contiguous() + 1, t3.to(d).permute(2, 0, 1).reshape(6, -1), t3.to(d).movedim(0, 2).flatten(1), t3.to(d).unfold(2, 3, 2), m.to(d).as_strided((5, 5), (41, 2))],
    'meshgrid, cartesian_prod, combinations': lambda d: [torch.stack(torch.meshgrid(v.to(d)[:4], v.to(d)[:5], indexing='ij')), torch.cartesian_prod(v.to(d)[:3], v.to(d)[:4]), torch.combinations(v.to(d)[:5], 2)],
    'nested tensors': lambda d: torch.nested.nested_tensor([v.to(d)[:5], v.to(d)[:9]]).to_padded_tensor(0.0),
}, 1e-4)


def _stat(f, lo, hi, n=20000):
    # Statistical: the generator differs from the CPU's, so compare a moment to its expectation.
    def run(d):
        x = f(d, n).double()
        return torch.tensor(float(lo <= float(x.mean()) <= hi))
    return run


def gen(d, seed=1):
    return torch.Generator(d).manual_seed(seed)


cases(L, 'random numbers (distribution checks)', {
    'rand': _stat(lambda d, n: torch.rand(n, device=d, generator=gen(d)), 0.49, 0.51),
    'randn': _stat(lambda d, n: torch.randn(n, device=d, generator=gen(d)).square(), 0.97, 1.03),
    'randint': _stat(lambda d, n: torch.randint(0, 10, (n,), device=d, generator=gen(d)), 4.4, 4.6),
    'bernoulli': _stat(lambda d, n: torch.bernoulli(torch.full((n,), 0.3, device=d), generator=gen(d)), 0.29, 0.31),
    'normal_ with a tensor of means': _stat(lambda d, n: torch.normal(torch.full((n,), 2.0, device=d), 1.0, generator=gen(d)), 1.97, 2.03),
    'exponential_': _stat(lambda d, n: torch.empty(n, device=d).exponential_(2.0, generator=gen(d)), 0.48, 0.52),
    'geometric_ and cauchy_ and log_normal_': _stat(lambda d, n: torch.empty(n, device=d).geometric_(0.5, generator=gen(d)), 1.95, 2.05),
    'poisson': _stat(lambda d, n: torch.poisson(torch.full((n,), 3.0, device=d), generator=gen(d)), 2.95, 3.05),
    'binomial': _stat(lambda d, n: torch.binomial(torch.full((n,), 10.0, device=d), torch.full((n,), 0.4, device=d), generator=gen(d)), 3.9, 4.1),
    'multinomial': _stat(lambda d, n: torch.multinomial(torch.tensor([0.1, 0.2, 0.7], device=d), n, replacement=True, generator=gen(d)).float(), 1.55, 1.65),
    'gamma and dirichlet': _stat(lambda d, n: torch._standard_gamma(torch.full((n,), 3.0, device=d), generator=gen(d)), 2.9, 3.1),
    'dropout keeps its fraction': _stat(lambda d, n: (F.dropout(torch.ones(n, device=d), 0.3, True) > 0).float(), 0.68, 0.72),
    'randperm is a permutation': lambda d: torch.randperm(1000, device=d, generator=gen(d)).sort().values,
    'randperm large': lambda d: torch.randperm(200000, device=d, generator=gen(d)).sort().values,
    'reproducible: same seed, same numbers': lambda d: torch.tensor(float(torch.equal(torch.rand(500, device=d, generator=gen(d, 5)), torch.rand(500, device=d, generator=gen(d, 5))))),
    'manual_seed and get_rng_state round trip': lambda d: _rng_roundtrip(d),
}, 0)


def _rng_roundtrip(d):
    torch.manual_seed(3)
    s = torch.cuda.get_rng_state() if d != 'cpu' else torch.get_rng_state()
    a = torch.rand(100, device=d)
    (torch.cuda.set_rng_state if d != 'cpu' else torch.set_rng_state)(s)
    return torch.tensor(float(torch.equal(a, torch.rand(100, device=d))))
