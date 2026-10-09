"""torch.sparse: COO, CSR, CSC and BSR tensors (cuSPARSE and PyTorch's own
kernels), compared with the CPU's results as dense matrices."""
import torch

from harness import cases, rnd, rint

L = 'sparse'
dense = rnd(24, 32, seed=1)
dense = dense * (rnd(24, 32, seed=2) > 0.6)
B = rnd(32, 10, seed=3)
v = rnd(32, seed=4)
sq = rnd(16, 16, seed=5) * (rnd(16, 16, seed=6) > 0.5) + 4 * torch.eye(16)


def dn(t):
    return t.to_dense() if t.layout != torch.strided else t


cases(L, 'COO', {
    'to_sparse, to_dense round trip': lambda d: dn(dense.to(d).to_sparse()),
    'coalesce sums duplicate entries': lambda d: dn(torch.sparse_coo_tensor(rint(0, 6, 2, 40, seed=2), rnd(40, seed=3), (6, 6)).to(d).coalesce()),
    'sparse @ dense (torch.sparse.mm)': lambda d: torch.sparse.mm(dense.to(d).to_sparse(), B.to(d)),
    'sparse @ dense, backward to the dense matrix': lambda d: _mm_grad(d),
    'addmm with a sparse matrix': lambda d: torch.sparse.addmm(rnd(24, 10, seed=7).to(d), dense.to(d).to_sparse(), B.to(d), beta=0.5, alpha=2),
    'sparse + sparse and sparse * dense': lambda d: [dn(dense.to(d).to_sparse() + dense.flip(0).to(d).to_sparse()), dn(dense.to(d).to_sparse() * 3.0)],
    'sparse.sum, over a dimension': lambda d: [torch.sparse.sum(dense.to(d).to_sparse()), dn(torch.sparse.sum(dense.to(d).to_sparse(), 1))],
    'sparse softmax': lambda d: dn(torch.sparse.softmax(dense.to(d).to_sparse(), 1)),
    'sparse transpose and index_select': lambda d: [dn(dense.to(d).to_sparse().t()), dn(dense.to(d).to_sparse().index_select(1, torch.tensor([0, 5, 9], device=d)))],
    'sparse_mask': lambda d: dn(rnd(24, 32, seed=9).to(d).sparse_mask(dense.to(d).to_sparse())),
    'hybrid sparse tensor (dense values)': lambda d: dn(torch.sparse_coo_tensor(torch.tensor([[0, 2, 3]]), rnd(3, 4, seed=3), (5, 4)).to(d).coalesce()),
    'sparse mv (matrix times vector)': lambda d: torch.mv(dense.to(d).to_sparse(), v.to(d)),
    'bmm of a sparse batch': lambda d: torch.bmm(torch.stack([dense, dense.flip(0)]).to(d).to_sparse(), torch.stack([B, B * 2]).to(d)),
}, 1e-4)

cases(L, 'CSR, CSC, BSR (cuSPARSE generic API)', {
    'to_sparse_csr and back': lambda d: dn(dense.to(d).to_sparse_csr()),
    'CSR @ dense (SpMM)': lambda d: dense.to(d).to_sparse_csr() @ B.to(d),
    'CSR @ vector (SpMV)': lambda d: dense.to(d).to_sparse_csr() @ v.to(d),
    'CSR addmm': lambda d: torch.addmm(rnd(24, 10, seed=7).to(d), dense.to(d).to_sparse_csr(), B.to(d), beta=0.5, alpha=2),
    'CSR @ CSR (SpGEMM)': lambda d: dn(dense.to(d).to_sparse_csr() @ dense.t().contiguous().to(d).to_sparse_csr()),
    'CSC @ dense': lambda d: dense.to(d).to_sparse_csc() @ B.to(d),
    'BSR @ dense, 4x4 blocks': lambda d: dense.to(d).to_sparse_bsr((4, 4)) @ B.to(d),
    'BSR to dense, block sizes 2 and 8': lambda d: [dn(dense.to(d).to_sparse_bsr((2, 2))), dn(dense.to(d).to_sparse_bsr((8, 8)))],
    'CSR to COO and back': lambda d: dn(dense.to(d).to_sparse_csr().to_sparse().to_sparse_csr()),
    'CSR sparse_sampled_addmm (SDDMM)': lambda d: dn(torch.sparse.sampled_addmm(dense.to(d).to_sparse_csr(), rnd(24, 8, seed=2).to(d), rnd(8, 32, seed=3).to(d), beta=0.5, alpha=2)),
    'CSR in half': lambda d: (dense.half().to(d).to_sparse_csr() @ B.half().to(d)).float() if d != 'cpu' else (dense.half().float() @ B.half().float()),
    'CSR in double': lambda d: dense.double().to(d).to_sparse_csr() @ B.double().to(d),
    'CSR with int32 indices': lambda d: dn(torch.sparse_csr_tensor(*(lambda c: (c.crow_indices().int(), c.col_indices().int(), c.values()))(dense.to_sparse_csr()), size=dense.shape).to(d)),
    'sparse CSR add and mul by scalar': lambda d: [dn(dense.to(d).to_sparse_csr() + dense.to(d).to_sparse_csr()), dn(dense.to(d).to_sparse_csr() * 2.5)],
}, 1e-3)


def _mm_grad(d):
    b = B.to(d).requires_grad_()
    y = torch.sparse.mm(dense.to(d).to_sparse(), b)
    (y * rnd(*y.shape, seed=8).to(d)).sum().backward()
    return [y, b.grad]


def _csr_grad(d):
    b = B.to(d).requires_grad_()
    y = dense.to(d).to_sparse_csr() @ b
    (y * rnd(*y.shape, seed=8).to(d)).sum().backward()
    return [y, b.grad]


def _semi(d):
    # 2:4 structured sparsity (cuSPARSELt): sm_80 and newer.
    from torch.sparse import to_sparse_semi_structured
    w = torch.tensor([1, 1, 0, 0] * 256, dtype=torch.float16).reshape(32, 32) * rnd(32, 32, seed=2).half().abs().add(0.1)
    x = rnd(32, 32, seed=3).half()
    if d == 'cpu':
        return (w.float() @ x.float())
    sw = to_sparse_semi_structured(w.to(d))
    return (sw @ x.to(d)).float()


cases(L, '2:4 semi-structured sparsity (cuSPARSELt)', {'to_sparse_semi_structured @ dense': _semi}, 3e-2, tier='full')
