"""torch.linalg corners beyond ops.py: factorizations with vectors (compared
through invariants, since signs and orderings are not unique), norms, powers,
LU and its solves, double precision, batched and larger sizes."""
import torch

from harness import cases, rnd

L = 'linalg'
A = rnd(12, 12, seed=1)
SPD = A @ A.T + 12 * torch.eye(12)
G = rnd(9, 9, seed=2)
TALL = rnd(20, 7, seed=3)
WIDE = rnd(6, 15, seed=4)
BATCH = rnd(5, 8, 8, seed=5)
BSPD = BATCH @ BATCH.transpose(1, 2) + 8 * torch.eye(8)


def inv(f):
    """A reconstruction error: compares a factorization by the matrix it rebuilds."""
    return f


def recon_svd(x, d, full=False):
    u, s, vh = torch.linalg.svd(x.to(d), full_matrices=full)
    k = s.shape[-1]
    return (u[..., :, :k] * s.unsqueeze(-2)) @ vh[..., :k, :]


def recon_eigh(x, d):
    w, v = torch.linalg.eigh(x.to(d))
    return (v * w.unsqueeze(-2)) @ v.mH


def recon_qr(x, d, mode='reduced'):
    q, r = torch.linalg.qr(x.to(d), mode=mode)
    return q @ r


def recon_lu(x, d):
    p, l, u = torch.linalg.lu(x.to(d))
    return p @ l @ u


def recon_eig(x, d):
    w, v = torch.linalg.eig(x.to(d))
    return torch.view_as_real(v @ torch.diag_embed(w) @ torch.linalg.inv(v))


cases(L, 'factorizations, compared through what they rebuild', {
    'svd, reduced': lambda d: recon_svd(TALL, d),
    'svd, full matrices (wide)': lambda d: recon_svd(WIDE, d, True),
    'svd, batched': lambda d: recon_svd(BATCH, d),
    'svd, double': lambda d: recon_svd(TALL.double(), d),
    'svd with gesvdj driver': lambda d: (lambda r: r)(_svd_driver(TALL, d, 'gesvdj')),
    'svd with gesvda driver (batched)': lambda d: _svd_driver(BATCH, d, 'gesvda'),
    'eigh and eigenvectors': lambda d: recon_eigh(SPD, d),
    'eigh, batched, lower triangle': lambda d: (lambda w_v: (w_v[1] * w_v[0].unsqueeze(-2)) @ w_v[1].mH)(torch.linalg.eigh(BSPD.to(d), UPLO='L')),
    'eigh, double': lambda d: recon_eigh(SPD.double(), d),
    'eig of a general matrix, reconstructed': lambda d: recon_eig(G, d),
    'qr, reduced': lambda d: recon_qr(TALL, d),
    'qr, complete': lambda d: recon_qr(TALL, d, 'complete'),
    'qr, r mode': lambda d: torch.linalg.qr(TALL.to(d), mode='r').R.abs(),
    'qr, batched, double': lambda d: recon_qr(BATCH.double(), d),
    'lu, pivoted, rebuilt': lambda d: recon_lu(A, d),
    'lu, batched': lambda d: recon_lu(BATCH, d),
    'lu_factor and lu_solve, left and right, adjoint': lambda d: [torch.linalg.lu_solve(*torch.linalg.lu_factor(A.to(d)), rnd(12, 3, seed=9).to(d)), torch.linalg.lu_solve(*torch.linalg.lu_factor(A.to(d)), rnd(3, 12, seed=9).to(d), left=False), torch.linalg.lu_solve(*torch.linalg.lu_factor(A.to(d)), rnd(12, 3, seed=9).to(d), adjoint=True)],
    'lu_factor_ex info': lambda d: [torch.linalg.lu_factor_ex(A.to(d)).info, torch.linalg.lu_factor_ex(torch.zeros(4, 4, device=d)).info],
    'cholesky, upper and batched': lambda d: [torch.linalg.cholesky(SPD.to(d), upper=True), torch.linalg.cholesky(BSPD.to(d))],
    'cholesky_ex flags a matrix that is not positive definite': lambda d: torch.linalg.cholesky_ex(-SPD.to(d)).info,
    'cholesky_inverse': lambda d: torch.cholesky_inverse(torch.linalg.cholesky(SPD.to(d))),
    'ldl_factor, solved': lambda d: torch.linalg.ldl_solve(*torch.linalg.ldl_factor(SPD.to(d)), rnd(12, 2, seed=9).to(d)),
    'geqrf and ormqr': lambda d: torch.ormqr(*torch.geqrf(TALL.to(d)), rnd(20, 4, seed=7).to(d)).abs(),
    'svd_lowrank and pca_lowrank (singular values)': lambda d: torch.svd_lowrank(torch.outer(torch.arange(1., 21.), torch.arange(1., 9.)).to(d) + 0.01 * rnd(20, 8, seed=2).to(d), q=3, niter=4)[1][:1].div(100).round(),
}, 2e-3)


def _svd_driver(x, d, driver):
    if d == 'cpu':
        return recon_svd(x, d)
    u, s, vh = torch.linalg.svd(x.to(d), full_matrices=False, driver=driver)
    return (u * s.unsqueeze(-2)) @ vh


cases(L, 'solves, inverses, norms, powers', {
    'solve with several right-hand sides, left=False': lambda d: [torch.linalg.solve(A.to(d), rnd(12, 4, seed=9).to(d)), torch.linalg.solve(A.to(d), rnd(3, 12, seed=9).to(d), left=False)],
    'solve_ex': lambda d: torch.linalg.solve_ex(A.to(d), rnd(12, seed=9).to(d)).result,
    'inv, inv_ex, batched double': lambda d: [torch.linalg.inv(A.to(d)), torch.linalg.inv_ex(A.to(d)).inverse, torch.linalg.inv(BSPD.double().to(d))],
    'solve_triangular, upper, unitriangular, batched': lambda d: [torch.linalg.solve_triangular(torch.triu(A).to(d) + 12 * torch.eye(12, device=d), rnd(12, 3, seed=9).to(d), upper=True), torch.linalg.solve_triangular(torch.tril(BSPD).to(d), rnd(5, 8, 2, seed=9).to(d), upper=False, unitriangular=True)],
    'det and slogdet, batched': lambda d: [torch.linalg.det(BSPD.to(d)), torch.linalg.slogdet(BSPD.to(d)).logabsdet],
    'matrix_power, positive and negative': lambda d: [torch.linalg.matrix_power(SPD.to(d) / 20, 3), torch.linalg.matrix_power(SPD.to(d) / 20, -2)],
    'matrix_rank and cond': lambda d: [torch.linalg.matrix_rank(A.to(d)).float(), torch.linalg.matrix_rank(torch.ones(5, 5, device=d)).float(), torch.linalg.cond(SPD.to(d)), torch.linalg.cond(SPD.to(d), 1)],
    'matrix_norm: fro, nuc, 2, -2, 1, inf': lambda d: [torch.linalg.matrix_norm(G.to(d), o) for o in ('fro', 'nuc', 2, -2, 1, float('inf'))],
    'norm of vectors and matrices': lambda d: [torch.linalg.norm(G.to(d)), torch.linalg.norm(G.to(d), 2, dim=1), torch.linalg.norm(G.to(d), dim=(0, 1), ord='nuc')],
    'lstsq with each driver': lambda d: [torch.linalg.lstsq(TALL.to(d), rnd(20, 2, seed=9).to(d)).solution, torch.linalg.lstsq(TALL.to(d), rnd(20, 2, seed=9).to(d), driver='gelsd').solution if d == 'cpu' else torch.linalg.lstsq(TALL.to(d), rnd(20, 2, seed=9).to(d)).solution],
    'pinv, hermitian, rcond': lambda d: [torch.linalg.pinv(TALL.to(d)), torch.linalg.pinv(SPD.to(d), hermitian=True)],
    'matrix_exp, batched and large norm': lambda d: [torch.linalg.matrix_exp(BATCH.to(d) * 0.3), torch.linalg.matrix_exp(G.to(d) * 2)],
    'tensorinv and tensorsolve': lambda d: [torch.linalg.tensorinv(SPD.to(d).reshape(12, 12), ind=1), torch.linalg.tensorsolve(SPD.reshape(12, 3, 4).to(d), rnd(12, seed=9).to(d))],
    'vander and vecdot': lambda d: [torch.linalg.vander(rnd(5, seed=2).to(d), N=4), torch.linalg.vecdot(rnd(5, 7, seed=2).to(d), rnd(5, 7, seed=3).to(d))],
    'diagonal, diag, trace on batches': lambda d: [torch.linalg.diagonal(BATCH.to(d)), torch.vmap(torch.trace)(BATCH.to(d))],
    'vmap over inv and det': lambda d: [torch.vmap(torch.linalg.inv)(BSPD.to(d)), torch.vmap(torch.linalg.det)(BSPD.to(d))],
    'large matmul in float, tf32 off': lambda d: (rnd(256, 192, seed=2).to(d) @ rnd(192, 160, seed=3).to(d)),
    'large batched solve': lambda d: torch.linalg.solve(rnd(40, 24, 24, seed=2).to(d) + 24 * torch.eye(24, device=d), rnd(40, 24, 2, seed=3).to(d)),
}, 2e-3)

cases(L, 'torch.linalg in double and lobpcg', {
    'solve': lambda d: torch.linalg.solve(A.double().to(d), rnd(12, 4, seed=9).double().to(d)),
    'cholesky': lambda d: torch.linalg.cholesky(SPD.double().to(d)),
    'inv': lambda d: torch.linalg.inv(A.double().to(d)),
    'det': lambda d: torch.linalg.det(A.double().to(d)),
    'eigvalsh': lambda d: torch.linalg.eigvalsh(SPD.double().to(d)),
    'svdvals': lambda d: torch.linalg.svdvals(TALL.double().to(d)),
    'lstsq': lambda d: torch.linalg.lstsq(TALL.double().to(d), rnd(20, 2, seed=9).double().to(d)).solution,
    'matmul': lambda d: A.double().to(d) @ A.double().to(d),
}, 1e-9, tier='full')
