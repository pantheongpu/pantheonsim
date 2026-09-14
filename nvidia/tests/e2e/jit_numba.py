# Numba on VirtualGPU, end to end: Python in, PTX through cuLinkCreate /
# cuLinkAddData / cuLinkComplete, kernels on the interpreter, answers checked
# against NumPy.
#
# Numba is worth a test of its own because it uses none of the paths the rest of
# the suite covers: no cudart, no fatbin, no nvcc. It resolves libcuda symbols
# by name at import (which is why the cuIpc* family has to exist even though it
# refuses every call), compiles to PTX itself, and assembles that PTX through
# the JIT link API.
import numpy as np, math
from numba import cuda, float32, int32

fails = []
def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name, extra)
    if not ok: fails.append(name)

# 1. shared memory + block reduction
@cuda.jit
def blocksum(x, out):
    sm = cuda.shared.array(128, float32)
    t = cuda.threadIdx.x
    i = cuda.grid(1)
    sm[t] = x[i] if i < x.size else 0.0
    cuda.syncthreads()
    s = 64
    while s > 0:
        if t < s: sm[t] += sm[t + s]
        cuda.syncthreads()
        s //= 2
    if t == 0: out[cuda.blockIdx.x] = sm[0]

x = np.arange(1024, dtype=np.float32)
out = cuda.device_array(8, dtype=np.float32)
blocksum[8, 128](cuda.to_device(x), out)
r = out.copy_to_host()
check("shared+reduction", abs(r.sum() - x.sum()) < 1e-1, f"{r.sum()} vs {x.sum()}")

# 2. atomics
@cuda.jit
def atomic_hist(x, h):
    i = cuda.grid(1)
    if i < x.size:
        cuda.atomic.add(h, x[i] % 16, 1)
xi = (np.arange(2048) % 37).astype(np.int32)
h = cuda.to_device(np.zeros(16, dtype=np.int32))
atomic_hist[16, 128](cuda.to_device(xi), h)
hh = h.copy_to_host()
exp = np.bincount(xi % 16, minlength=16)
check("atomic.add", np.array_equal(hh, exp), f"{hh[:4]} vs {exp[:4]}")

# 3. math intrinsics + 2D grid
@cuda.jit
def mathk(a, b):
    i, j = cuda.grid(2)
    if i < a.shape[0] and j < a.shape[1]:
        b[i, j] = math.sqrt(a[i, j]) + math.exp(-a[i, j]) + math.sin(a[i, j])
a = np.abs(np.random.RandomState(0).randn(64, 64)).astype(np.float32)
b = cuda.device_array_like(a)
mathk[(4, 4), (16, 16)](cuda.to_device(a), b)
ref = np.sqrt(a) + np.exp(-a) + np.sin(a)
got = b.copy_to_host()
check("math intrinsics 2D", np.allclose(got, ref, atol=2e-5), f"maxerr={np.abs(got-ref).max():.2e}")

# 4. matmul with shared tiles (numba docs example)
TPB = 16
@cuda.jit
def fast_matmul(A, B, C):
    sA = cuda.shared.array((TPB, TPB), float32)
    sB = cuda.shared.array((TPB, TPB), float32)
    x, y = cuda.grid(2)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    if x >= C.shape[0] and y >= C.shape[1]: return
    tmp = float32(0.)
    for k in range(int32(A.shape[1] / TPB)):
        sA[tx, ty] = A[x, ty + k * TPB]
        sB[tx, ty] = B[tx + k * TPB, y]
        cuda.syncthreads()
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]
        cuda.syncthreads()
    C[x, y] = tmp
rs = np.random.RandomState(1)
A = rs.rand(64, 64).astype(np.float32); B = rs.rand(64, 64).astype(np.float32)
C = cuda.device_array((64, 64), dtype=np.float32)
fast_matmul[(4, 4), (TPB, TPB)](cuda.to_device(A), cuda.to_device(B), C)
cc = C.copy_to_host()
check("tiled matmul", np.allclose(cc, A @ B, atol=1e-3), f"maxerr={np.abs(cc-A@B).max():.2e}")

# 5. warp-level: cuda.syncwarp + shfl
@cuda.jit
def warp_prefix(x, out):
    i = cuda.grid(1)
    lane = cuda.laneid
    v = x[i]
    for d in (1, 2, 4, 8, 16):
        n = cuda.shfl_up_sync(0xFFFFFFFF, v, d)
        if lane >= d: v += n
    out[i] = v
xf = np.ones(64, dtype=np.float32)
o = cuda.device_array(64, dtype=np.float32)
warp_prefix[2, 32](cuda.to_device(xf), o)
oo = o.copy_to_host()
check("shfl_up_sync scan", np.array_equal(oo[:32], np.arange(1, 33, dtype=np.float32)), f"{oo[:5]}")

# 6. streams + async copies
s = cuda.stream()
d = cuda.to_device(np.arange(256, dtype=np.float32), stream=s)
@cuda.jit
def scale(x):
    i = cuda.grid(1)
    if i < x.size: x[i] *= 3.0
scale[2, 128, s](d)
res = d.copy_to_host(stream=s); s.synchronize()
check("streams", abs(res[10] - 30.0) < 1e-5, f"{res[10]}")

import sys
if fails:
    print(f"FAIL: {len(fails)} of the Numba checks failed: {fails}")
    sys.exit(1)
print("RESULT: Numba compiles, links and runs on VirtualGPU")
