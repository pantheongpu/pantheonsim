"""PyTorch's own SASS for a Turing card (sm_75, a T4: no PTX path for it in the wheels, so the SASS executor
runs every kernel). One line each: "ok <name>", or "FAIL <name>: <why>".

The first check is a bug that shipped: a bf16 convolution on a T4 has no cuDNN path, so PyTorch lowers it to its
im2col kernel, whose index arithmetic is 64-bit and contains `LEA Rd, P0, Ra, -c[pad], 0x1` followed by
`IADD3.X Rd, Rhi, ~c[pad+4], RZ, P0`: the pad 0 negated is ~0 + 1, which carries out of the low word. The LEA lost
that carry, the high word came out 0xffffffff and the kernel read 2^32 elements before the image
("invalid-pointer"). Patch embedding (kernel = stride = 4, no padding) hits it. A failure leaves the CUDA context in error, so the checks after the first one fail too."""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
g = torch.Generator().manual_seed(1)


def check(name, f, tol):
    try:
        gpu, cpu = f('cuda'), f('cpu')
        diff = float((gpu.float().cpu() - cpu.float()).abs().max())
        print(f"ok {name}" if diff <= tol else f"FAIL {name}: max difference {diff:g}, allowed {tol:g}", flush=True)
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)


img = torch.randn(4, 3, 16, 16, generator=g)
w44 = torch.randn(64, 3, 4, 4, generator=g) * 0.1
w33 = torch.randn(16, 3, 3, 3, generator=g) * 0.1
check('bf16 patch-embedding conv (kernel = stride = 4, no padding)',
      lambda d: F.conv2d(img.bfloat16().to(d), w44.bfloat16().to(d), stride=4), 0.1)
check('bf16 conv, padding 1', lambda d: F.conv2d(img.bfloat16().to(d), w33.bfloat16().to(d), padding=1), 0.1)
check('bf16 strided conv, kernel 2 stride 2', lambda d: F.conv2d(img.bfloat16().to(d), w44[:, :, :2, :2].bfloat16().to(d), stride=2), 0.1)
check('float conv, kernel = stride = 4', lambda d: F.conv2d(img.to(d), w44.to(d), stride=4), 1e-4)
