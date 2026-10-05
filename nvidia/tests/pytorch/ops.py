"""PyTorch's CUDA build, unmodified, on a simulated NVIDIA GPU: one operator
at a time, each on the simulated GPU and on the CPU from the same inputs --
PyTorch's own kernels and the libraries it ships (cuBLAS, cuBLASLt, cuDNN,
cuFFT, cuRAND, cuSOLVER, cuSPARSE), attention through each SDPA backend
(cuDNN's attention graphs, forward and backward, among them),
cuDNN's training paths (convolution gradients, half and double LSTMs, CTC
loss), and autocast. One line each: "ok <name>", or "FAIL <name>: <why>".
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
GPU = 'cuda'


def report(name, gpu, cpu, tol):
    if isinstance(gpu, (tuple, list)):
        diff = max(float((g.detach().float().cpu() - c.detach().float()).abs().max()) for g, c in zip(gpu, cpu))
    else:
        diff = float((gpu.detach().float().cpu() - cpu.detach().float()).abs().max())
    print(f"ok {name}" if diff <= tol else f"FAIL {name}: max difference {diff:g}, allowed {tol:g}", flush=True)


def check(name, f, tol=1e-4):
    try:
        report(name, f(GPU), f('cpu'), tol)
    except Exception as e:  # a refused kernel or library call shows up here, and says which
        print(f"FAIL {name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)


g = torch.Generator().manual_seed(1)
a, b, v = torch.randn(64, 48, generator=g), torch.randn(48, 32, generator=g), torch.randn(1000, generator=g)
x4 = torch.randn(2, 3, 16, 16, generator=g)
sq = torch.randn(12, 12, generator=g)
spd = sq @ sq.t() + 12 * torch.eye(12)

check('elementwise', lambda d: torch.relu(v.to(d) * 1.5 + 2).exp().log1p())
check('reductions', lambda d: torch.stack([v.to(d).sum(), v.to(d).amax(), v.to(d).std(), v.to(d).norm()]))
check('matmul float', lambda d: a.to(d) @ b.to(d))
check('matmul half', lambda d: (a.half().to(d) @ b.half().to(d)) if d != 'cpu' else a.half().float() @ b.half().float(), 1e-2)
check('matmul bfloat16', lambda d: (a.bfloat16().to(d) @ b.bfloat16().to(d)).float() if d != 'cpu'
      else a.bfloat16().float() @ b.bfloat16().float(), 0.15)
check('batched matmul', lambda d: torch.bmm(a.reshape(4, 16, 48).to(d), b.expand(4, 48, 32).contiguous().to(d)))
check('softmax and log_softmax', lambda d: torch.stack([torch.softmax(a.to(d), 1), torch.log_softmax(a.to(d), 1)]))
check('layer_norm and group_norm', lambda d: torch.cat([F.layer_norm(a.to(d), (48,)).flatten(),
                                                    F.group_norm(x4.to(d), 3).flatten()]))
check('conv2d, pooling, upsampling', lambda d: F.interpolate(F.max_pool2d(F.conv2d(x4.to(d), torch.ones(8, 3, 3, 3).to(d) / 9,
                                                                                    padding=1), 2), scale_factor=2, mode='bilinear'))
check('sort', lambda d: torch.sort(v.to(d)).values)
check('topk', lambda d: torch.topk(v.to(d), 7).values)
check('cumsum', lambda d: torch.cumsum(v.to(d), 0), 1e-3)
check('argmax, unique', lambda d: torch.cat([v.to(d).argmax().reshape(1).float(),
                                          torch.unique((v * 4).round().to(d)).float()]))
check('scatter_add and index_select', lambda d: torch.zeros(10, device=d).scatter_add(
    0, (torch.arange(1000) % 10).to(d), v.to(d)).index_select(0, torch.tensor([3, 1, 4], device=d)), 1e-3)
check('embedding bag', lambda d: F.embedding_bag(torch.arange(12).to(d), a[:20].to(d), torch.tensor([0, 5, 9]).to(d)))
check('random numbers (cuRAND), statistics', lambda d: torch.tensor(float(abs(torch.randn(8192, device=d,
      generator=torch.Generator(d).manual_seed(1)).std() - 1) < 0.05)))
check('fft (cuFFT)', lambda d: torch.view_as_real(torch.fft.fft(torch.complex(v, v.flip(0)).to(d))), 1e-3)
check('rfft2 and irfft2', lambda d: torch.fft.irfft2(torch.fft.rfft2(x4.to(d)), s=(16, 16)), 1e-4)
check('inverse (cuSOLVER)', lambda d: torch.linalg.inv(spd.to(d)), 1e-4)
check('cholesky', lambda d: torch.linalg.cholesky(spd.to(d)), 1e-4)
check('solve', lambda d: torch.linalg.solve(spd.to(d), a[:12, :5].to(d)), 1e-4)
check('qr', lambda d: torch.linalg.qr(spd.to(d)).R.abs(), 1e-3)
check('svd, singular values', lambda d: torch.linalg.svdvals(sq.to(d)), 1e-3)
check('eigh, eigenvalues', lambda d: torch.linalg.eigvalsh(spd.to(d)), 1e-3)
check('sparse matmul (cuSPARSE)', lambda d: torch.sparse.mm(a.relu().to_sparse().to(d), b.to(d)), 1e-4)

# torch.linalg's batched and general paths: cuSOLVER's 64-bit and Jacobi
# APIs, cuBLAS's batched LU and triangular solves, and det/slogdet, whose
# reductions are jiterator kernels compiled through NVRTC.
bs = torch.randn(4, 6, 6, generator=g)
bspd = bs @ bs.transpose(1, 2) + 6 * torch.eye(6)
gen = torch.randn(5, 5, generator=g)
check('batched eigh', lambda d: torch.linalg.eigvalsh(bspd.to(d)), 1e-3)
check('batched svdvals', lambda d: torch.linalg.svdvals(bs.to(d)), 1e-3)
check('batched inverse', lambda d: torch.linalg.inv(bspd.to(d)), 1e-4)
check('batched solve', lambda d: torch.linalg.solve(bs.to(d), bs[:, :, :2].to(d)), 1e-3)
check('det, slogdet', lambda d: torch.stack([torch.linalg.det(gen.to(d)), torch.linalg.slogdet(gen.to(d)).logabsdet]), 1e-3)
check('lu_factor', lambda d: torch.linalg.lu_factor(gen.to(d)).LU, 1e-4)
check('lstsq', lambda d: torch.linalg.lstsq(a[:20, :6].to(d), a[:20, 6:8].to(d)).solution, 1e-3)
check('eig (general), eigenvalue magnitudes', lambda d: torch.linalg.eigvals(gen.to(d)).abs().sort().values, 1e-3)
check('pinv', lambda d: torch.linalg.pinv(a[:10, :6].to(d)), 1e-3)
check('matrix_exp', lambda d: torch.linalg.matrix_exp(0.1 * gen.to(d)), 1e-4)
check('cholesky_solve', lambda d: torch.cholesky_solve(a[:12, :3].to(d), torch.linalg.cholesky(spd.to(d))), 1e-3)
check('triangular_solve', lambda d: torch.linalg.solve_triangular(torch.tril(spd).to(d), a[:12, :3].to(d), upper=False), 1e-4)
check('householder_product (orgqr)', lambda d: torch.linalg.householder_product(*torch.geqrf(sq.to(d))).abs(), 1e-3)
check('svd 40x30, full', lambda d: torch.linalg.svd(torch.randn(40, 30, generator=torch.Generator().manual_seed(3)).to(d)).S, 1e-3)
check('eigh 40x40', lambda d: torch.linalg.eigvalsh((lambda m: m @ m.T)(torch.randn(40, 40, generator=torch.Generator().manual_seed(4))).to(d)), 1e-2)
q, k, vv = (torch.randn(2, 4, 16, 32, generator=g) for _ in range(3))
check('attention', lambda d: F.scaled_dot_product_attention(q.to(d), k.to(d), vv.to(d), is_causal=True))
from torch.nn.attention import SDPBackend, sdpa_kernel
for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH):
    def run(d, backend=backend):
        if d == 'cpu':
            return F.scaled_dot_product_attention(q.half().float(), k.half().float(), vv.half().float(), is_causal=True)
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q.half().to(d), k.half().to(d), vv.half().to(d), is_causal=True)
    check(f'attention, {backend.name.lower()} backend, half', run, 1e-2)


def cudnn_attention_grads(d):
    # Training through cuDNN's attention graph (forward with statistics, then
    # the backward graph); the CPU's math path in float is the reference.
    qq, kk, vvv = (t.half().float().clone().to(d) if d == 'cpu' else t.half().to(d).clone() for t in (q, k, vv))
    for t in (qq, kk, vvv):
        t.requires_grad_(True)
    if d == 'cpu':
        out = F.scaled_dot_product_attention(qq, kk, vvv, is_causal=True)
    else:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            out = F.scaled_dot_product_attention(qq, kk, vvv, is_causal=True)
    out.float().square().sum().backward()
    return [qq.grad, kk.grad, vvv.grad]


check('attention gradients, cudnn_attention backend, half', cudnn_attention_grads, 3e-2)


def autocast_mlp(d):
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(48, 64), torch.nn.GELU(), torch.nn.Linear(64, 10)).to(d)
    with torch.autocast(d, dtype=torch.bfloat16, enabled=d != 'cpu'):
        out = m(a.to(d))
    return out.float()


check('autocast bfloat16 MLP', autocast_mlp, 5e-2)
x8, w8 = (torch.randn(32, 64, generator=g) * 2).to(torch.float8_e4m3fn), (torch.randn(48, 64, generator=g) * 2).to(torch.float8_e4m3fn)


def fp8(d):
    if d == 'cpu':
        return (x8.float() * 0.5) @ (w8.float() * 2).t()
    return torch._scaled_mm(x8.to(d), w8.to(d).t(), scale_a=torch.tensor(0.5, device=d),
                            scale_b=torch.tensor(2.0, device=d), out_dtype=torch.float32)


check('fp8 matmul (torch._scaled_mm)', fp8, 1e-3)


# cuDNN's training paths, through PyTorch: convolution gradients (channels
# last, groups, dilation; and double), an LSTM in half under autocast and in
# double, and CTC loss with its gradient (PyTorch takes cuDNN's when the
# targets are int32 on the CPU and every input is full length).
def conv_grads(d, dtype=torch.float32, channels_last=True):
    xg = x4.to(d, dtype).repeat(1, 2, 1, 1)
    if channels_last:
        xg = xg.contiguous(memory_format=torch.channels_last)
    xg.requires_grad_()
    wg = (torch.arange(6 * 3 * 3 * 3, dtype=torch.float32).reshape(6, 3, 3, 3).sin() / 4).to(d, dtype).requires_grad_()
    out = F.conv2d(xg, wg, padding=2, dilation=2, groups=2)
    (out * out.detach().flip(-1)).sum().backward()
    return torch.cat([out.flatten(), xg.grad.flatten(), wg.grad.flatten()])


def lstm(d, autocast=False, dtype=torch.float32):
    torch.manual_seed(0)
    m = torch.nn.LSTM(12, 16, num_layers=2).to(d, dtype)
    xs = a[:30, :12].reshape(5, 6, 12).to(d, dtype)
    with torch.autocast(d, dtype=torch.float16, enabled=autocast and d != 'cpu'):
        out, _ = m(xs)
    return out.float()


def ctc(d):
    logits = a[:60, :5].reshape(10, 6, 5).clone().to(d).requires_grad_()
    targets = torch.tensor([1, 2, 2, 3, 4, 1, 3, 2, 4, 1, 1, 2], dtype=torch.int32)
    loss = F.ctc_loss(F.log_softmax(logits, 2), targets if d != 'cpu' else targets.long(), [10] * 6, [2, 2, 2, 2, 2, 2],
                      reduction='sum')
    loss.backward()
    return torch.cat([loss.reshape(1), logits.grad.flatten()])


check('conv2d gradients (cuDNN), channels last, groups, dilation', conv_grads, 1e-3)
check('conv2d gradients (cuDNN), double', lambda d: conv_grads(d, torch.float64, False), 1e-10)
check('LSTM under autocast (cuDNN half RNN)', lambda d: lstm(d, autocast=True), 1e-2)
check('LSTM in double (cuDNN double RNN)', lambda d: lstm(d, dtype=torch.float64), 1e-10)
check('ctc_loss and its gradient (cuDNN)', ctc, 1e-3)


# Complex tensors: cuBLAS's and cuSOLVER's C/Z entry points.
ca = torch.complex(a[:16, :16], a[16:32, :16]); cb = torch.complex(b[:16, :8], b[16:32, :8])
cs = ca + 8 * torch.eye(16)
ch = ca @ ca.mH + 16 * torch.eye(16)
R = torch.view_as_real
check('complex matmul', lambda d: R(ca.to(d) @ cb.to(d)), 1e-4)
check('complex bmm', lambda d: R(torch.bmm(ca.expand(3, 16, 16).to(d), cb.expand(3, 16, 8).to(d))), 1e-4)
check('complex double matmul', lambda d: R(ca.cdouble().to(d) @ cb.cdouble().to(d)), 1e-10)
check('complex inverse', lambda d: R(torch.linalg.inv(cs.to(d))), 1e-4)
check('complex solve', lambda d: R(torch.linalg.solve(cs.to(d), cb.to(d))), 1e-4)
check('complex det', lambda d: R(torch.linalg.det(cs.to(d) / 8).reshape(1)), 1e-4)
check('complex cholesky', lambda d: R(torch.linalg.cholesky(ch.to(d))), 1e-4)
check('complex qr (|R|)', lambda d: torch.linalg.qr(ca.to(d)).R.abs(), 1e-3)
check('complex svdvals', lambda d: torch.linalg.svdvals(ca.to(d)), 1e-3)
check('complex eigvalsh', lambda d: torch.linalg.eigvalsh(ch.to(d)), 1e-2)
check('complex lstsq', lambda d: R(torch.linalg.lstsq(torch.complex(a[:20, :6], a[20:40, :6]).to(d), cb[:, :2].repeat(2, 1)[:20].to(d)).solution), 1e-3)
check('complex pinv', lambda d: R(torch.linalg.pinv(ca[:10, :6].to(d))), 1e-3)
check('complex batched inverse', lambda d: R(torch.linalg.inv(cs.expand(3, 16, 16).to(d))), 1e-4)
check('complex solve_triangular', lambda d: R(torch.linalg.solve_triangular(torch.tril(cs).to(d), cb.to(d), upper=False)), 1e-4)
check('complex fft round trip', lambda d: R(torch.fft.ifft(torch.fft.fft(ca.to(d)))), 1e-5)

# torch.compile: Inductor's Triton kernels, compiled by Triton and handed to the
# simulator as PTX (vgpu run points Triton's ptxas at vgpu-ptxas). The CPU side
# runs the same function eagerly.
def compiled(f):
    fast = torch.compile(f)
    return lambda d, *args: (fast if d != 'cpu' else f)(*[t.to(d) for t in args])


mlp = torch.nn.Sequential(torch.nn.Linear(48, 64), torch.nn.GELU(), torch.nn.LayerNorm(64), torch.nn.Linear(64, 8))
mlp_c = compiled(lambda t: mlp.to(t.device)(t))
fused = compiled(lambda t: torch.sin(t) * 2 + t.cos().sum())
attn = compiled(lambda q, k, v: torch.softmax(q @ k.transpose(-1, -2) / 8, -1) @ v)
q4 = torch.randn(2, 4, 16, 32, generator=g)
check('torch.compile: fused pointwise and reduction', lambda d: fused(d, v))
check('torch.compile: MLP with LayerNorm and GELU', lambda d: mlp_c(d, a), 1e-3)
check('torch.compile: attention', lambda d: attn(d, q4, q4.flip(-1), q4 * 0.5), 1e-3)
