"""Composable Kernel's GEMMs, as PyTorch's ROCm build runs them when told to
prefer CK over hipBLASLt (torch.backends.cuda.preferred_blas_library("ck")),
each against the CPU. Half is not among them: PyTorch 2.10 compiles its CK
half GEMM out (ck_gemm_half.hip's dispatch is "#if 0"), so on a card too it
runs nothing. One line each: "ok <name>", or "FAIL <name>: <why>".
"""
import torch

torch.manual_seed(0)
torch.backends.cuda.preferred_blas_library("ck")
for name, dt, tol in (("CK's bfloat16 GEMM", torch.bfloat16, 0.5), ("CK's float GEMM", torch.float32, 1e-3),
                      ("CK's float GEMM, B transposed", torch.float32, 1e-3)):
    try:
        a, b = torch.randn(128, 64).to(dt), torch.randn(64, 96).to(dt)
        if "transposed" in name:
            got = (a.cuda() @ b.t().contiguous().cuda().t()).float().cpu()
        else:
            got = (a.cuda() @ b.cuda()).float().cpu()
        diff = float((got - a.float() @ b.float()).abs().max())
        print(f"ok {name}" if diff <= tol else f"FAIL {name}: max difference {diff:g}, allowed {tol:g}", flush=True)
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)
