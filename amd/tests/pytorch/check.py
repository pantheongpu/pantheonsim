"""PyTorch's ROCm build, unmodified, on a simulated MI300X.

Each check runs on the simulated GPU and on the CPU from the same inputs and
compares: operators whose kernels are PyTorch's own and ones from the ROCm
libraries it ships (hipBLASLt, MIOpen, rocPRIM, rocRAND, rocFFT, rocSOLVER),
a small CNN and a small transformer trained a few steps, attention, and an
FP8 GEMM. One line each: "ok <name>", or "FAIL <name>: <why>".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
GPU = 'cuda'


def report(name, gpu, cpu, tol):
    diff = float((gpu.detach().float().cpu() - cpu.detach().float()).abs().max())
    print(f"ok {name}" if diff <= tol else f"FAIL {name}: max difference {diff:g}, allowed {tol:g}", flush=True)


def check(name, f, tol=1e-4):
    try:
        report(name, f(GPU), f('cpu'), tol)
    except Exception as e:  # a refused kernel shows up here, and says which
        print(f"FAIL {name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)


a, b, v = torch.randn(64, 48), torch.randn(48, 32), torch.randn(1000)
x4 = torch.randn(2, 3, 16, 16)
check('elementwise', lambda d: torch.relu(v.to(d) * 1.5 + 2))
check('reductions', lambda d: torch.stack([v.to(d).sum(), v.to(d).amax(), v.to(d).std()]))
check('matmul float', lambda d: a.to(d) @ b.to(d))
check('matmul half', lambda d: (a.half().to(d) @ b.half().to(d)) if d != 'cpu' else a.half().float() @ b.half().float(), 1e-2)
# The product is rounded to a bfloat16 on the GPU, eight bits of mantissa.
check('matmul bfloat16', lambda d: (a.bfloat16().to(d) @ b.bfloat16().to(d)).float() if d != 'cpu'
      else a.bfloat16().float() @ b.bfloat16().float(), 0.15)
check('matmul, A transposed', lambda d: b.t().to(d) @ a.t().to(d))
check('softmax', lambda d: torch.softmax(a.to(d), 1))
check('conv2d', lambda d: F.conv2d(x4.to(d), torch.ones(8, 3, 3, 3).to(d) / 9, padding=1))
check('sort', lambda d: torch.sort(v.to(d)).values)
check('cumsum', lambda d: torch.cumsum(v.to(d), 0), 1e-3)
check('topk', lambda d: torch.topk(v.to(d), 7).values)
check('random numbers', lambda d: torch.tensor(float(abs(torch.randn(4096, device=d, generator=torch.Generator(d).manual_seed(1)).std() - 1) < 0.1)))
check('fft', lambda d: torch.fft.fft(torch.complex(v, v.flip(0)).to(d)), 1e-3)
check('rfft', lambda d: torch.fft.rfft(v.reshape(10, 100).to(d)), 1e-3)
check('inverse', lambda d: torch.linalg.inv(torch.eye(8).to(d) * 2 + a[:8, :8].to(d) * 0.1))
check('layernorm', lambda d: F.layer_norm(a.to(d), (48,)))
q, k, vv = (torch.randn(2, 4, 16, 32) for _ in range(3))
check('attention', lambda d: F.scaled_dot_product_attention(q.to(d), k.to(d), vv.to(d), is_causal=True))


torch.manual_seed(1)
# The 8-bit float the device's hardware has: gfx942's is FNUZ (no negative
# zero, one NaN), gfx950's the OCP format every other vendor uses.
F8 = torch.float8_e4m3fn if torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx950') \
    else torch.float8_e4m3fnuz
x8, w8 = (torch.randn(32, 64) * 2).to(F8), (torch.randn(48, 64) * 2).to(F8)


def fp8(d):
    if d == 'cpu':
        return (x8.float() * 0.5) @ (w8.float() * 2).t()
    return torch._scaled_mm(x8.to(d), w8.to(d).t(), scale_a=torch.tensor(0.5, device=d),
                            scale_b=torch.tensor(2.0, device=d), out_dtype=torch.float32)


check('fp8 matmul', fp8)


def train(model_fn, inputs, loss_fn, steps):
    def run(d):
        torch.manual_seed(0)
        model = model_fn().to(d)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        xs = [t.to(d) for t in inputs]
        losses = []
        for _ in range(steps):
            opt.zero_grad()
            loss = loss_fn(model, *xs)
            loss.backward()
            opt.step()
            losses.append(loss.detach())
        return torch.stack(losses)
    return run


g = torch.Generator().manual_seed(2)
img, lab = torch.randn(16, 1, 12, 12, generator=g), torch.randint(0, 10, (16,), generator=g)
check('cnn training', train(lambda: nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                                                  nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                                                  nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(64, 10)),
                            [img, lab], lambda m, x, y: F.cross_entropy(m(x), y), 3), 1e-4)
tok = torch.randint(0, 50, (4, 12), generator=g)
check('transformer training', train(lambda: nn.Sequential(nn.Embedding(50, 32), nn.TransformerEncoder(
    nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True), 1), nn.Linear(32, 50)),
    [tok], lambda m, t: F.cross_entropy(m(t).reshape(-1, 50), t.reshape(-1)), 3), 1e-4)
