# Triton on VirtualGPU, end to end.
#
# Triton compiles all the way to PTX on its own and then shells out to ptxas for
# a cubin -- the one artifact in its pipeline VirtualGPU cannot load, since
# there is no SASS decoder here. tools/vgpu_triton.py ends the pipeline at PTX
# through Triton's own documented stage hook, and this checks the result against
# NumPy. The tl.dot case is the interesting one: it goes through real mma.sync
# and ldmatrix, in the .shared form that names the space in the instruction
# rather than converting the address first.
#
# Device memory comes from Numba because Triton ships no allocator of its own
# and torch is not a dependency here; a Triton kernel takes anything with
# .data_ptr() and .dtype.
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))

import numpy as np, triton, triton.language as tl
import vgpu_triton
from numba import cuda  # only for device allocations; Triton brings no allocator

if not vgpu_triton.available():
    # An older Triton without the stage hook. Nothing here can run on it, and
    # that is a version limit rather than a simulator failure.
    print("SKIP: this Triton has no knobs.runtime.add_stages_inspection_hook")
    sys.exit(0)
vgpu_triton.install()

class DT:
    def __init__(self, n): self.n = n
    def __str__(self): return "torch." + self.n
class P:
    def __init__(self, d, n="float32"): self.d = d; self.dtype = DT(n)
    def data_ptr(self): return self.d.__cuda_array_interface__['data'][0]

fails = []
def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name, extra); ok or fails.append(name)

# --- 1. fused softmax over rows ---
@triton.jit
def softmax_kernel(out_ptr, in_ptr, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * stride + cols, mask=mask, other=-float('inf'))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(out_ptr + row * stride + cols, e / tl.sum(e, axis=0), mask=mask)

rs = np.random.RandomState(0)
X = rs.randn(16, 100).astype(np.float32)
dX = cuda.to_device(X); dO = cuda.device_array_like(X)
softmax_kernel[(16,)](P(dO), P(dX), 100, 100, BLOCK=128)
got = dO.copy_to_host()
e = np.exp(X - X.max(1, keepdims=True)); ref = e / e.sum(1, keepdims=True)
check("softmax", np.allclose(got, ref, atol=1e-6), f"maxerr={np.abs(got-ref).max():.2e}")

# --- 2. tiled matmul ---
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        offs_k = k + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

M = N = K = 128
A = rs.rand(M, K).astype(np.float32); B = rs.rand(K, N).astype(np.float32)
dA, dB = cuda.to_device(A), cuda.to_device(B)
dC = cuda.device_array((M, N), dtype=np.float32)
matmul_kernel[(4, 4)](P(dA), P(dB), P(dC), M, N, K, BM=32, BN=32, BK=32)
C = dC.copy_to_host()
check("tl.dot matmul", np.allclose(C, A @ B, atol=2e-2), f"maxerr={np.abs(C-A@B).max():.2e}")

# --- 3. atomics ---
@triton.jit
def sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    tl.atomic_add(out_ptr, tl.sum(v, axis=0))

xs = np.arange(1024, dtype=np.float32)
dx = cuda.to_device(xs); ds = cuda.to_device(np.zeros(1, dtype=np.float32))
sum_kernel[(8,)](P(dx), P(ds), 1024, BLOCK=128)
tot = ds.copy_to_host()[0]
check("atomic_add reduction", abs(tot - xs.sum()) < 1.0, f"{tot} vs {xs.sum()}")

if fails:
    print(f"FAIL: {len(fails)} of the Triton checks failed: {fails}")
    sys.exit(1)
print("RESULT: Triton compiles to PTX and runs on VirtualGPU")
