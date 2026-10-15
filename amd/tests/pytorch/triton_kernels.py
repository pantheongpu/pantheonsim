"""Triton's AMD backend, unmodified, on a simulated AMD GPU.

Kernels written by hand, compiled by Triton for the device when they are first
launched and loaded through HIP: what Inductor generates for torch.compile and
what vLLM's attention, quantization and sampling kernels are made of, without
PyTorch's operators. Matrix multiplies on MFMA (CDNA) or WMMA (RDNA) in every
input type the device has, a flash-attention loop (two dots and an online
softmax) at several warp counts and pipeline depths, reductions, scans, sorts
and histograms, atomics, the math library, inline assembly, block pointers
and the device's print and assert. Each is checked against the CPU. One line
each: "ok <name>", or "FAIL <name>: <why>".

fp8 matrix products (gfx942's fnuz formats and gfx950's OCP ones) and MX-scaled
products (gfx950) run on the devices that have them, so the number of lines
depends on the device: 18, and 2 more on gfx942, 6 more on gfx950.
"""
import torch
import triton
import triton.language as tl

torch.manual_seed(0)
arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
dev = 'cuda'


def check(name, f):
    try:
        why = f()
        print(f"ok {name}" if why is None else f"FAIL {name}: {why}", flush=True)
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)


def close(got, want, tol):
    d = float((got.detach().float().cpu() - want.detach().float().cpu()).abs().max())
    return None if d <= tol else f"max difference {d:g}, allowed {tol:g}"


@triton.jit
def attention_kernel(q, k, v, o, N, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    Q = tl.load(q + rm[:, None] * D + rd[None, :])
    m = tl.full((BM,), float('-inf'), tl.float32)
    l = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for n0 in range(0, N, BN):
        rn = n0 + tl.arange(0, BN)
        K = tl.load(k + rn[None, :] * D + rd[:, None])   # D x BN
        s = tl.dot(Q, K) * SCALE
        s = tl.where(rm[:, None] >= rn[None, :], s, float('-inf'))
        mn = tl.maximum(m, tl.max(s, 1))
        p = tl.exp(s - mn[:, None])
        a = tl.exp(m - mn)
        l = l * a + tl.sum(p, 1)
        acc = acc * a[:, None]
        V = tl.load(v + rn[:, None] * D + rd[None, :])
        acc += tl.dot(p.to(tl.float16), V)
        m = mn
    tl.store(o + rm[:, None] * D + rd[None, :], (acc / l[:, None]).to(tl.float16))


def attention(warps, stages):
    def run():
        n, d = 128, 64
        q = torch.randn(n, d, device=dev).half()
        k, v, o = torch.randn_like(q), torch.randn_like(q), torch.empty_like(q)
        attention_kernel[(n // 32,)](q, k, v, o, n, D=d, BM=32, BN=32, SCALE=d ** -0.5, num_warps=warps, num_stages=stages)
        s = (q.float() @ k.float().T) * d ** -0.5
        s = s.masked_fill(torch.ones(n, n, device=dev).triu(1) > 0, float('-inf'))
        return close(o, torch.softmax(s, -1) @ v.float(), 2e-2)
    return run


@triton.jit
def dot_kernel(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, OUT: tl.constexpr):
    rm, rn, rk = tl.arange(0, M), tl.arange(0, N), tl.arange(0, K)
    A = tl.load(a + rm[:, None] * K + rk[None, :])
    B = tl.load(b + rk[:, None] * N + rn[None, :])
    tl.store(c + rm[:, None] * N + rn[None, :], tl.dot(A, B, out_dtype=OUT))


def dot(dtype, out, m=64, n=64, k=64, tol=1e-2, gen=torch.randn):
    def run():
        a, b = gen(m, k, device=dev).to(dtype), gen(k, n, device=dev).to(dtype)
        c = torch.empty(m, n, device=dev, dtype=torch.int32 if out == tl.int32 else torch.float32)
        dot_kernel[(1,)](a, b, c, m, n, k, OUT=out)
        return close(c, a.float() @ b.float(), tol)
    return run


def small_ints(*shape, device):
    return torch.randint(-5, 5, shape, device=device)


def small_floats(*shape, device):
    return torch.randn(*shape, device=device).clamp(-2, 2)


@triton.jit
def reduce_kernel(x, o, n, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(x + i, mask=i < n, other=0.0)
    tl.store(o + 0, tl.sum(v, 0))
    tl.store(o + 1, tl.max(v, 0))
    tl.store(o + 2, tl.argmax(v, 0).to(tl.float32))
    tl.store(o + 3, tl.sum(tl.cumsum(v, 0), 0))
    tl.store(o + 4, tl.sum(tl.sort(v, 0) * i.to(tl.float32), 0))
    tl.store(o + 8 + tl.arange(0, 8), tl.histogram((v * 3).to(tl.int32) & 7, 8).to(tl.float32))


def reductions():
    n = 200
    x = torch.rand(n, device=dev)
    o = torch.zeros(16, device=dev)
    reduce_kernel[(1,)](x, o, n, BLOCK=256)
    v = torch.cat([x.cpu(), torch.zeros(56)])   # the lanes past n read 0
    want = torch.stack([v.sum(), v.max(), v.argmax().float(), v.cumsum(0).sum(), (torch.sort(v)[0] * torch.arange(256)).sum()])
    hist = torch.bincount((v * 3).int() & 7, minlength=8).float()
    return close(o[:5], want, 0.05) or close(o[8:], hist, 0)


@triton.jit
def atomic_kernel(x, y, z, c, n, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    m = i < n
    tl.atomic_add(x + (i % 4), tl.full((BLOCK,), 1.0, tl.float32), mask=m)
    tl.atomic_add(y + (i % 4), tl.full((BLOCK,), 1.0, tl.float16), mask=m)
    tl.atomic_max(z + (i % 4), i, mask=m)
    tl.atomic_add(c + i * 0, tl.full((BLOCK,), 1, tl.int32), mask=m)


def atomics():
    # 100 of each block's 128 lanes, in two blocks: 50 adds into each of four slots.
    x, y = torch.zeros(4, device=dev), torch.zeros(4, device=dev, dtype=torch.float16)
    z, c = torch.zeros(4, device=dev, dtype=torch.int32), torch.zeros(1, device=dev, dtype=torch.int32)
    atomic_kernel[(2,)](x, y, z, c, 100, BLOCK=128)
    got = (x.tolist(), y.float().tolist(), z.tolist(), c.tolist())
    want = ([50.0] * 4, [50.0] * 4, [96, 97, 98, 99], [200])
    return None if got == want else f"{got}, wanted {want}"


@triton.jit
def math_kernel(x, o, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(x + i)
    r = (tl.exp(v) + tl.log(v + 3) + tl.sqrt(v + 3) + tl.sin(v) + tl.cos(v) + tl.math.erf(v) + tl.extra.libdevice.tanh(v) +
         tl.math.rsqrt(v + 3) + tl.math.exp2(v) + tl.math.log2(v + 3) + tl.math.floor(v * 3) + tl.math.fma(v, v, v) +
         tl.abs(v) + tl.extra.libdevice.pow(v + 3, 1.5))
    tl.store(o + i, r)


def maths():
    x = torch.rand(256, device=dev) * 2 - 1.0   # v + 3 is positive
    o = torch.empty_like(x)
    math_kernel[(1,)](x, o, BLOCK=256)
    want = (torch.exp(x) + torch.log(x + 3) + torch.sqrt(x + 3) + torch.sin(x) + torch.cos(x) + torch.erf(x) + torch.tanh(x) +
            torch.rsqrt(x + 3) + torch.exp2(x) + torch.log2(x + 3) + torch.floor(x * 3) + x * x + x + x.abs() + (x + 3) ** 1.5)
    return close(o, want, 1e-2)


@triton.jit
def nan_kernel(a, b, c, o, n, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    m = i < n
    x, y, z = tl.load(a + i, mask=m), tl.load(b + i, mask=m), tl.load(c + i, mask=m)
    nan = tl.PropagateNan.ALL
    tl.store(o + i, tl.maximum(tl.maximum(x, y, propagate_nan=nan), z, propagate_nan=nan))
    tl.store(o + n + i, tl.minimum(tl.minimum(x, y, propagate_nan=nan), z, propagate_nan=nan))


def nan_propagation(dtype):
    # Three-way maximum and minimum that pass a NaN through: gfx950's v_maximum3_f32 and
    # v_pk_maximum3_f16 (and the minimum forms). The CPU's torch.maximum and minimum also pass a
    # NaN through, and treat -0 and +0 as equal.
    def run():
        n = 128
        a, b, c = (torch.randn(n).to(dtype) for _ in range(3))
        a[::5] = b[1::7] = float('nan')
        a[2::11], b[2::11], c[2::11] = 0.0, -0.0, -0.0
        o = torch.empty(2 * n, device=dev, dtype=dtype)
        nan_kernel[(1,)](a.to(dev), b.to(dev), c.to(dev), o, n, BLOCK=128)
        want = torch.cat([torch.maximum(torch.maximum(a, b), c), torch.minimum(torch.minimum(a, b), c)])
        got = o.cpu()
        if not torch.equal(torch.isnan(got), torch.isnan(want)):
            return "NaNs in different places"
        return close(torch.nan_to_num(got, nan=7.0), torch.nan_to_num(want, nan=7.0), 0)
    return run


@triton.jit
def asm_kernel(x, o, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(o + i, tl.inline_asm_elementwise("v_add_u32 $0, $1, 7", "=v,v", [tl.load(x + i)], dtype=tl.int32,
                                              is_pure=True, pack=1))


def inline_asm():
    x = torch.arange(64, device=dev, dtype=torch.int32)
    o = torch.empty_like(x)
    asm_kernel[(1,)](x, o, BLOCK=64)
    return close(o, x + 7, 0)


@triton.jit
def block_kernel(x, o, N, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK
    t = tl.load(tl.make_block_ptr(x, (N, N), (N, 1), (row, 0), (BLOCK, BLOCK), (1, 0)), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(o, (N, N), (N, 1), (row, 0), (BLOCK, BLOCK), (1, 0)), tl.trans(t) * 2, boundary_check=(0, 1))


def block_pointers():
    # 40 x 40 in 32 x 32 blocks. The first block is whole: out[r, c] = 2 x[c, r].
    # The second has 8 rows: its tile is read with zeros below them,
    # transposed, and stored with the rows and columns that do not exist cut
    # off, so out[32 + r, c] = 2 x[32 + c, r] for r, c < 8, and 0 elsewhere.
    n = 40
    x = torch.randn(n, n, device=dev)
    o = torch.zeros(n, n, device=dev)
    block_kernel[(2,)](x, o, n, BLOCK=32)
    xc = x.cpu()
    want = torch.zeros(n, n)
    want[:32, :32] = xc[:32, :32].T * 2
    want[32:, :8] = xc[32:, :8].T * 2
    return close(o, want, 1e-6)


@triton.jit
def print_kernel(x):
    tl.device_print("v", tl.load(x + tl.arange(0, 4)))


def device_print():
    print_kernel[(1,)](torch.arange(4, device=dev).float())
    torch.cuda.synchronize()


@triton.jit
def assert_kernel(x):
    tl.device_assert(tl.load(x) > 0, "must be positive")


def device_assert():
    assert_kernel[(1,)](torch.ones(1, device=dev))
    torch.cuda.synchronize()


@triton.jit
def scaled_dot_kernel(a, a_scale, b, b_scale, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm, rn, rk, rs = tl.arange(0, M), tl.arange(0, N), tl.arange(0, K), tl.arange(0, K // 32)
    A = tl.load(a + rm[:, None] * K + rk[None, :])
    B = tl.load(b + rk[:, None] * N + rn[None, :])
    As = tl.load(a_scale + rm[:, None] * (K // 32) + rs[None, :])
    Bs = tl.load(b_scale + rn[:, None] * (K // 32) + rs[None, :])
    tl.store(c + rm[:, None] * N + rn[None, :], tl.dot_scaled(A, As, "e4m3", B, Bs, "e4m3"))


def scaled_dot():
    # E8M0 scales of 127 are 2^0: the product is the plain fp8 one.
    m = n = 32
    k = 64
    a = small_floats(m, k, device=dev).to(torch.float8_e4m3fn)
    b = small_floats(k, n, device=dev).to(torch.float8_e4m3fn)
    sa = torch.full((m, k // 32), 127, device=dev, dtype=torch.uint8)
    sb = torch.full((n, k // 32), 127, device=dev, dtype=torch.uint8)
    c = torch.empty(m, n, device=dev)
    scaled_dot_kernel[(1,)](a, sa, b, sb, c, m, n, k)
    return close(c, a.float() @ b.float(), 0.05)


@triton.jit
def mx_kernel(a, a_scale, b, b_scale, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, KA: tl.constexpr,
              KB: tl.constexpr, FA: tl.constexpr, FB: tl.constexpr):
    rm, rn, rka, rkb, rs = tl.arange(0, M), tl.arange(0, N), tl.arange(0, KA), tl.arange(0, KB), tl.arange(0, K // 32)
    A = tl.load(a + rm[:, None] * KA + rka[None, :])
    B = tl.load(b + rkb[:, None] * N + rn[None, :])
    As = tl.load(a_scale + rm[:, None] * (K // 32) + rs[None, :])
    Bs = tl.load(b_scale + rn[:, None] * (K // 32) + rs[None, :])
    tl.store(c + rm[:, None] * N + rn[None, :], tl.dot_scaled(A, As, FA, B, Bs, FB))


E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])


def mx_operand(fmt, rows, k):
    """(what the kernel loads, the values it stands for) for a rows x k operand of an MX format, K packed
    two to a byte for E2M1 (low nibble first)."""
    if fmt == "e2m1":
        raw = torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8)
        nibbles = torch.stack([raw & 0xF, raw >> 4], -1).flatten(-2).long()
        return raw, torch.where(nibbles & 8 > 0, -1.0, 1.0) * E2M1[nibbles & 7]
    x = torch.randn(rows, k).clamp(-3, 3).to({"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[fmt])
    return x, x.float()


def mx_dot(fa, fb):
    # One 32-wide block of K and E8M0 scales (2^-2 .. 2^2) of each row of A and column of B: Triton
    # widens the operands (v_cvt_scalef32_pk_bf16_*) and multiplies in bfloat16. The reference decodes
    # the formats here and scales on the CPU.
    def run():
        m = n = k = 32
        (ra, da), (rb, db) = mx_operand(fa, m, k), mx_operand(fb, n, k)
        rb = rb.view(n, -1).t().contiguous()   # K x N
        sa = torch.randint(125, 130, (m, k // 32), dtype=torch.uint8)
        sb = torch.randint(125, 130, (n, k // 32), dtype=torch.uint8)
        c = torch.empty(m, n, device=dev)
        mx_kernel[(1,)](ra.to(dev), sa.to(dev), rb.to(dev), sb.to(dev), c, m, n, k, ra.shape[1], rb.shape[0], fa, fb)
        want = (da * torch.pow(2.0, sa.float() - 127).repeat_interleave(32, 1)) @ \
               (db * torch.pow(2.0, sb.float() - 127).repeat_interleave(32, 1)).t()
        return close(c, want, 0.1)
    return run


check("a flash-attention loop, 4 warps and 2 stages", attention(4, 2))
check("a flash-attention loop, 8 warps and 3 stages", attention(8, 3))
check("a flash-attention loop, one warp", attention(1, 1))
check("tl.dot of fp16", dot(torch.float16, tl.float32))
check("tl.dot of bf16", dot(torch.bfloat16, tl.float32, tol=0.1))
check("tl.dot of int8 into int32", dot(torch.int8, tl.int32, tol=0, gen=small_ints))
check("tl.dot of fp32", dot(torch.float32, tl.float32, tol=0.5))
check("tl.dot of 128 x 128 x 64", dot(torch.float16, tl.float32, 128, 128, 64, tol=0.05))
check("tl.dot of 16 x 16 x 16", dot(torch.float16, tl.float32, 16, 16, 16))
for name, dtype in {"gfx942": [("e4m3fnuz", torch.float8_e4m3fnuz), ("e5m2fnuz", torch.float8_e5m2fnuz)],
                    "gfx950": [("e4m3fn", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2)]}.get(arch, []):
    check(f"tl.dot of fp8 {name}", dot(dtype, tl.float32, 32, 32, 32, tol=0.05, gen=small_floats))
if arch == "gfx950":
    check("tl.dot_scaled of fp8 with E8M0 scales", scaled_dot)
    for fa, fb in [("e2m1", "e2m1"), ("e4m3", "e2m1"), ("e5m2", "e5m2")]:
        check(f"tl.dot_scaled of {fa} by {fb}, one scale block of K", mx_dot(fa, fb))
check("sum, max, argmax, cumsum, sort and histogram", reductions)
check("atomic add (fp32, fp16, int32) and max", atomics)
check("exp, log, sqrt, trigonometry, erf, tanh and libdevice pow", maths)
check("three-way maximum and minimum passing NaNs, fp32", nan_propagation(torch.float32))
check("three-way maximum and minimum passing NaNs, fp16", nan_propagation(torch.float16))
check("inline assembly", inline_asm)
check("block pointers with a boundary check, and a transpose", block_pointers)
check("tl.device_print", device_print)
check("tl.device_assert, passing", device_assert)
