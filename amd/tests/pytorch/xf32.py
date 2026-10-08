"""hipBLASLt's xf32 GEMM, as PyTorch's ROCm build runs it on an MI300X when told a float32 matmul may
use reduced precision (torch.set_float32_matmul_precision("high") with HIPBLASLT_ALLOW_TF32=1): floats
multiplied with the mantissa cut to 10 bits by v_mfma_f32_16x16x8_xf32 and its 32x32x4 sibling. The
product must be near the exact one, and visibly less exact than a float GEMM, which proves the xf32
kernel ran. One line: "ok <name>", or "FAIL <name>: <why>".
"""
import os

os.environ.setdefault("HIPBLASLT_ALLOW_TF32", "1")
import torch

torch.manual_seed(0)
torch.set_float32_matmul_precision("high")
name = "float GEMM at reduced precision (xf32)"
try:
    a, b = torch.randn(256, 192), torch.randn(192, 128)
    exact = a.double() @ b.double()
    cpu = float((a @ b).double().sub(exact).abs().max())
    gpu = float((a.cuda() @ b.cuda()).cpu().double().sub(exact).abs().max())
    # A 10-bit mantissa is 2^-10 relative on each factor: the error of a sum of 192 products of
    # magnitude ~1 is some hundredths, thousands of times a float GEMM's.
    ok = 100 * cpu < gpu < 0.5
    print(f"ok {name}" if ok else f"FAIL {name}: error {gpu:g} against a float GEMM's {cpu:g}", flush=True)
except Exception as e:
    print(f"FAIL {name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)
