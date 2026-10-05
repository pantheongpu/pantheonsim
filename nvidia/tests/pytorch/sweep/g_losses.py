"""Loss functions, forward and backward."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from harness import cases, check, rnd, rint, flat

L = 'losses'


def _grads(d, loss_fn, *tensors, ints=()):
    xs = [t.detach().clone().to(d).requires_grad_() for t in tensors]
    extra = [t.to(d) for t in ints]
    loss = loss_fn(*xs, *extra)
    loss.sum().backward()
    return [loss, [x.grad for x in xs]]


def sc(f, *tensors, ints=()):
    return lambda d: _grads(d, f, *tensors, ints=ints)


logits = rnd(6, 5, seed=1)
tgt = rint(0, 5, 6, seed=2)
probs = torch.sigmoid(rnd(6, 5, seed=3))
bin_t = (rnd(6, 5, seed=4) > 0).float()
a, b, c = rnd(6, 5, seed=5), rnd(6, 5, seed=6), rnd(6, 5, seed=7)
sign = torch.where(rnd(6, seed=8) > 0, 1.0, -1.0)
w5 = torch.tensor([0.5, 1.0, 2.0, 1.5, 0.7])
img = rnd(3, 4, 5, 5, seed=9)
img_t = rint(0, 4, 3, 5, 5, seed=10)

cases(L, 'classification losses', {
    'cross_entropy': sc(lambda x, t: F.cross_entropy(x, t), logits, ints=(tgt,)),
    'cross_entropy, label smoothing': sc(lambda x, t: F.cross_entropy(x, t, label_smoothing=0.1), logits, ints=(tgt,)),
    'cross_entropy, weights and ignore_index': sc(lambda x, t, w: F.cross_entropy(x, t, weight=w, ignore_index=2), logits, ints=(tgt, w5)),
    'cross_entropy, probabilities': sc(lambda x, p: F.cross_entropy(x, p), logits, probs / probs.sum(1, keepdim=True), ints=()),
    'cross_entropy, 2d': sc(lambda x, t: F.cross_entropy(x, t), img, ints=(img_t,)),
    'nll_loss': sc(lambda x, t: F.nll_loss(F.log_softmax(x, 1), t), logits, ints=(tgt,)),
    'nll_loss 2d, weights': sc(lambda x, t, w: F.nll_loss(F.log_softmax(x, 1), t, weight=w[:4]), img, ints=(img_t, w5)),
    'binary_cross_entropy': sc(lambda p, t: F.binary_cross_entropy(p, t), probs, bin_t),
    'binary_cross_entropy_with_logits, pos_weight': sc(lambda x, t, w: F.binary_cross_entropy_with_logits(x, t, pos_weight=w), logits, ints=(bin_t, w5)),
    'kl_div': sc(lambda x, p: F.kl_div(F.log_softmax(x, 1), p, reduction='batchmean'), logits, probs / probs.sum(1, keepdim=True)),
    'multilabel_soft_margin': sc(lambda x, t: F.multilabel_soft_margin_loss(x, t), logits, ints=(bin_t,)),
    'multi_margin': sc(lambda x, t: F.multi_margin_loss(x, t), logits, ints=(tgt,)),
    'multilabel_margin': sc(lambda x, t: F.multilabel_margin_loss(x, t), logits, ints=(torch.tensor([[3, 0, -1, 1, 2]] * 6),)),
    'poisson_nll_loss': sc(lambda x, t: F.poisson_nll_loss(x, t), logits, probs * 4),
    'gaussian_nll_loss': sc(lambda x, t, v: F.gaussian_nll_loss(x, t, v), logits, a, b.abs() + 0.5),
}, 1e-4)

cases(L, 'regression and embedding losses', {
    'mse_loss': sc(F.mse_loss, a, b),
    'l1_loss': sc(F.l1_loss, a, b),
    'smooth_l1_loss': sc(lambda x, y: F.smooth_l1_loss(x, y, beta=0.5), a, b),
    'huber_loss': sc(lambda x, y: F.huber_loss(x, y, delta=0.7), a, b),
    'hinge_embedding_loss': sc(lambda x, t: F.hinge_embedding_loss(x, t), a, ints=(torch.where(rnd(6, 5, seed=11) > 0, 1.0, -1.0),)),
    'cosine_embedding_loss': sc(lambda x, y, t: F.cosine_embedding_loss(x, y, t), a, b, ints=(sign,)),
    'margin_ranking_loss': sc(lambda x, y, t: F.margin_ranking_loss(x[:, 0], y[:, 0], t, margin=0.1), a, b, ints=(sign,)),
    'triplet_margin_loss': sc(lambda x, y, z: F.triplet_margin_loss(x, y, z), a, b, c),
    'triplet_margin_with_distance_loss': sc(lambda x, y, z: F.triplet_margin_with_distance_loss(x, y, z, distance_function=lambda p, q: 1 - F.cosine_similarity(p, q)), a, b, c),
    'soft_margin_loss': sc(lambda x, t: F.soft_margin_loss(x, t), a, ints=(torch.where(rnd(6, 5, seed=12) > 0, 1.0, -1.0),)),
    'reduction sum and none': lambda d: _grads(d, lambda x, y: F.mse_loss(x, y, reduction='none') + F.l1_loss(x, y, reduction='sum'), a, b),
}, 1e-4)

cases(L, 'losses in half', {
    'cross_entropy': lambda d: _lowp(d, lambda x, t: F.cross_entropy(x, t), logits, ints=(tgt,)),
    'mse_loss': lambda d: _lowp(d, F.mse_loss, a, b),
    'binary_cross_entropy_with_logits': lambda d: _lowp(d, lambda x, t: F.binary_cross_entropy_with_logits(x, t), a, bin_t),
}, 3e-2, tier='full')


def _lowp(d, f, *tensors, ints=()):
    dt = torch.float16
    if d == 'cpu':
        return _grads(d, f, *[t.to(dt).float() for t in tensors], ints=ints)
    xs = [t.detach().clone().to(d, dt).requires_grad_() for t in tensors]
    loss = f(*xs, *[t.to(d) for t in ints])
    loss.float().sum().backward()
    return [loss.float(), [x.grad.float() for x in xs]]


def _ctc(d):
    lp = F.log_softmax(rnd(12, 4, 6, seed=2).to(d), 2).detach().requires_grad_()
    tg = rint(1, 6, 4, 3, seed=3).to(d)
    loss = F.ctc_loss(lp, tg, torch.full((4,), 12), torch.tensor([3, 2, 3, 1]), reduction='mean', zero_infinity=True)
    loss.backward()
    return [loss, lp.grad]


check(L, 'ctc_loss with int64 targets and variable lengths (native kernel)', 1e-3)(_ctc)


def _label_smooth_seq(d):
    x = rnd(2, 7, 11, seed=1).to(d).requires_grad_()
    t = rint(0, 11, 2, 7, seed=2).to(d)
    loss = F.cross_entropy(x.transpose(1, 2), t, label_smoothing=0.2, ignore_index=0)
    loss.backward()
    return [loss, x.grad]


check(L, 'token cross entropy with label smoothing and ignore_index', 1e-4)(_label_smooth_seq)


def _adaptive_softmax(d):
    torch.manual_seed(0)
    m = nn.AdaptiveLogSoftmaxWithLoss(16, 40, cutoffs=[5, 15], div_value=2.0)
    m = m.to(d)
    x = rnd(10, 16, seed=1).to(d).requires_grad_()
    t = rint(0, 40, 10, seed=2).to(d)
    out = m(x, t)
    out.loss.backward()
    return [out.output, out.loss, x.grad]


check(L, 'AdaptiveLogSoftmaxWithLoss', 1e-4, tier='full')(_adaptive_softmax)
