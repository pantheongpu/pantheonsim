"""torch.nn layers, forward and backward: convolutions, pooling, padding,
resampling, normalization, activations, embeddings."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from harness import cases, check, layer, rnd, rint, fwd_bwd

L = 'nn'
H, B = torch.float16, torch.bfloat16

# ---- convolutions (cuDNN where PyTorch takes it, its own kernels otherwise)
layer(L, 'Conv1d stride 2, padding 1', lambda: nn.Conv1d(4, 6, 3, stride=2, padding=1), [(2, 4, 16)], 1e-4)
layer(L, 'Conv2d groups 2, padding 1', lambda: nn.Conv2d(4, 6, 3, padding=1, groups=2), [(2, 4, 9, 9)], 1e-4)
layer(L, 'Conv2d dilation 2, stride 2, reflect padding', lambda: nn.Conv2d(3, 5, 3, stride=2, dilation=2, padding=2, padding_mode='reflect'),
      [(2, 3, 12, 12)], 1e-4)
layer(L, 'Conv2d 1x1 without bias', lambda: nn.Conv2d(6, 4, 1, bias=False), [(2, 6, 7, 7)], 1e-4)
layer(L, 'Conv2d depthwise 3x3', lambda: nn.Conv2d(8, 8, 3, padding=1, groups=8), [(2, 8, 10, 10)], 1e-4)
layer(L, 'Conv2d half', lambda: nn.Conv2d(4, 6, 3, padding=1), [(2, 4, 12, 12)], 3e-2, dtype=H)
layer(L, 'Conv2d bfloat16', lambda: nn.Conv2d(4, 6, 3, padding=1), [(2, 4, 12, 12)], 1.5e-1, dtype=B)
layer(L, 'Conv3d padding 1', lambda: nn.Conv3d(2, 4, 3, padding=1), [(1, 2, 6, 6, 6)], 2e-3)
layer(L, 'ConvTranspose1d stride 2', lambda: nn.ConvTranspose1d(4, 3, 4, stride=2, padding=1), [(2, 4, 8)], 1e-4)
layer(L, 'ConvTranspose2d stride 2, output padding', lambda: nn.ConvTranspose2d(4, 3, 3, stride=2, padding=1, output_padding=1),
      [(2, 4, 6, 6)], 1e-4)
layer(L, 'ConvTranspose2d groups 2', lambda: nn.ConvTranspose2d(4, 6, 3, groups=2), [(2, 4, 6, 6)], 1e-4)
layer(L, 'ConvTranspose3d stride 2', lambda: nn.ConvTranspose3d(2, 2, 3, stride=2), [(1, 2, 4, 4, 4)], 2e-3)
layer(L, 'Conv2d, cuDNN off (native kernels)', lambda: nn.Sequential(nn.Conv2d(3, 4, 3, padding=1)), [(2, 3, 8, 8)], 1e-4, tier='full')


def _no_cudnn_conv(d):
    with torch.backends.cudnn.flags(enabled=False):
        return fwd_bwd(lambda: nn.Conv2d(3, 4, 3, stride=2, padding=1, groups=1), [rnd(2, 3, 9, 9, seed=10)], d)


check(L, 'Conv2d with cuDNN disabled (PyTorch\'s native kernels)', 1e-4)(_no_cudnn_conv)


def _no_cudnn_conv3d(d):
    with torch.backends.cudnn.flags(enabled=False):
        return fwd_bwd(lambda: nn.Conv3d(2, 3, 3, padding=1), [rnd(1, 2, 5, 5, 5, seed=10)], d)


check(L, 'Conv3d with cuDNN disabled', 1e-4, tier='full')(_no_cudnn_conv3d)


def _channels_last(d):
    def mk():
        return nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
    torch.manual_seed(0)
    m = mk().to(d).to(memory_format=torch.channels_last)
    x = rnd(2, 4, 8, 8, seed=3).to(d).contiguous(memory_format=torch.channels_last).requires_grad_()
    y = m(x)
    (y * rnd(*y.shape, seed=5).to(d)).sum().backward()
    return [y, x.grad, [p.grad for p in m.parameters()]]


check(L, 'Conv + BatchNorm + ReLU, channels last', 1e-4)(_channels_last)

# ---- unfold / fold / shuffle / padding
layer(L, 'Unfold and Fold', lambda: nn.Sequential(nn.Unfold(3, padding=1, stride=2), nn.Fold((8, 8), 3, padding=1, stride=2)),
      [(2, 3, 8, 8)], 1e-4)
layer(L, 'PixelShuffle and PixelUnshuffle', lambda: nn.Sequential(nn.PixelShuffle(2), nn.PixelUnshuffle(2)), [(2, 8, 5, 5)], 1e-6)
layer(L, 'ChannelShuffle', lambda: nn.ChannelShuffle(2), [(2, 6, 4, 4)], 1e-6)
for mode, mk in (('reflect', nn.ReflectionPad2d), ('replicate', nn.ReplicationPad2d), ('circular', nn.CircularPad2d), ('zero', nn.ZeroPad2d)):
    layer(L, f'{mode} padding 2d', (lambda mk=mk: mk((1, 2, 2, 1))), [(2, 3, 6, 6)], 1e-6)
layer(L, 'reflect padding 1d', lambda: nn.ReflectionPad1d((2, 3)), [(2, 3, 9)], 1e-6)
layer(L, 'replicate padding 3d', lambda: nn.ReplicationPad3d((1, 2, 0, 1, 1, 1)), [(1, 2, 4, 4, 4)], 1e-6)
layer(L, 'reflect padding 3d', lambda: nn.ReflectionPad3d(1), [(1, 2, 4, 4, 4)], 1e-6, tier='full')
layer(L, 'constant padding 2d', lambda: nn.ConstantPad2d((1, 1, 2, 0), 3.5), [(2, 3, 4, 4)], 1e-6, tier='full')

# ---- pooling
layer(L, 'MaxPool1d', lambda: nn.MaxPool1d(3, stride=2, padding=1), [(2, 3, 17)], 1e-6)
layer(L, 'MaxPool2d ceil mode, dilation', lambda: nn.MaxPool2d(3, stride=2, padding=1, dilation=2, ceil_mode=True), [(2, 3, 13, 13)], 1e-6)
layer(L, 'MaxPool3d', lambda: nn.MaxPool3d(2), [(1, 2, 6, 6, 6)], 1e-6)
layer(L, 'AvgPool1d', lambda: nn.AvgPool1d(3, stride=2, padding=1, count_include_pad=False), [(2, 3, 17)], 1e-5)
layer(L, 'AvgPool2d divisor override', lambda: nn.AvgPool2d(3, stride=2, padding=1, divisor_override=5), [(2, 3, 9, 9)], 1e-5)
layer(L, 'AvgPool3d', lambda: nn.AvgPool3d(2, stride=1), [(1, 2, 5, 5, 5)], 1e-5)
layer(L, 'AdaptiveAvgPool1d', lambda: nn.AdaptiveAvgPool1d(5), [(2, 3, 17)], 1e-5)
layer(L, 'AdaptiveAvgPool2d', lambda: nn.AdaptiveAvgPool2d((3, 5)), [(2, 3, 11, 13)], 1e-5)
layer(L, 'AdaptiveAvgPool3d', lambda: nn.AdaptiveAvgPool3d((2, 3, 2)), [(1, 2, 5, 7, 4)], 1e-5)
layer(L, 'AdaptiveMaxPool2d', lambda: nn.AdaptiveMaxPool2d((3, 4)), [(2, 3, 11, 13)], 1e-6)
layer(L, 'AdaptiveMaxPool3d', lambda: nn.AdaptiveMaxPool3d(2), [(1, 2, 5, 7, 4)], 1e-6, tier='full')
layer(L, 'LPPool2d', lambda: nn.LPPool2d(2, 3, stride=2), [(2, 3, 9, 9)], 1e-4)
layer(L, 'LPPool1d', lambda: nn.LPPool1d(3, 3, stride=2), [(2, 3, 15)], 1e-4, tier='full')


def _unpool(d):
    x = rnd(2, 3, 8, 8, seed=2).to(d).requires_grad_()
    y, idx = F.max_pool2d(x, 2, return_indices=True)
    z = F.max_unpool2d(y, idx, 2)
    z.mul(rnd(*z.shape, seed=3).to(d)).sum().backward()
    return [y, idx, z, x.grad]


check(L, 'max_pool2d with indices, then max_unpool2d', 1e-6)(_unpool)


def _frac(d):
    x = rnd(2, 3, 12, 12, seed=2).to(d).requires_grad_()
    samples = torch.rand(2, 3, 2, generator=torch.Generator().manual_seed(4)).to(d)
    y = F.fractional_max_pool2d(x, 3, output_size=(5, 5), _random_samples=samples)
    y.mul(rnd(*y.shape, seed=3).to(d)).sum().backward()
    return [y, x.grad]


check(L, 'fractional_max_pool2d with fixed samples', 1e-6)(_frac)

# ---- resampling
for mode, kw in (('nearest', {}), ('nearest-exact', {}), ('bilinear', {'align_corners': False}), ('bilinear', {'align_corners': True}),
                 ('bicubic', {'align_corners': False}), ('area', {})):
    nm = f"interpolate {mode}" + (f", align_corners={kw['align_corners']}" if kw else '')
    layer(L, nm, (lambda mode=mode, kw=kw: nn.Upsample(size=(11, 13), mode=mode, **kw)), [(2, 3, 7, 8)], 1e-4)
layer(L, 'interpolate trilinear', lambda: nn.Upsample(scale_factor=2, mode='trilinear'), [(1, 2, 4, 4, 4)], 1e-4)
layer(L, 'interpolate linear (1d)', lambda: nn.Upsample(scale_factor=2.5, mode='linear'), [(2, 3, 9)], 1e-4)


def _antialias(d):
    x = rnd(2, 3, 20, 20, seed=2).to(d).requires_grad_()
    y = F.interpolate(x, size=(7, 7), mode='bilinear', antialias=True)
    y.mul(rnd(*y.shape, seed=3).to(d)).sum().backward()
    return [y, x.grad]


check(L, 'interpolate bilinear with antialias', 1e-4)(_antialias)


def _grid_sample(d):
    x = rnd(2, 3, 8, 8, seed=2).to(d).requires_grad_()
    theta = (torch.tensor([[[0.9, 0.2, 0.05], [-0.1, 0.8, 0.0]]]).repeat(2, 1, 1)).to(d).requires_grad_()
    grid = F.affine_grid(theta, (2, 3, 6, 6), align_corners=False)
    outs = []
    for mode in ('bilinear', 'nearest', 'bicubic'):
        for pad in ('zeros', 'border', 'reflection'):
            outs.append(F.grid_sample(x, grid, mode=mode, padding_mode=pad, align_corners=False))
    loss = sum((o * rnd(*o.shape, seed=3 + i).to(d)).sum() for i, o in enumerate(outs))
    loss.backward()
    return [outs, x.grad, theta.grad]


check(L, 'affine_grid and grid_sample, every mode and padding, with gradients', 1e-4)(_grid_sample)


def _grid_sample3d(d):
    x = rnd(1, 2, 5, 5, 5, seed=2).to(d).requires_grad_()
    grid = (rnd(1, 3, 3, 3, 3, seed=4) * 0.5).to(d).requires_grad_()
    y = F.grid_sample(x, grid, align_corners=True)
    y.mul(rnd(*y.shape, seed=3).to(d)).sum().backward()
    return [y, x.grad, grid.grad]


check(L, 'grid_sample 3d with gradients', 1e-4, tier='full')(_grid_sample3d)

# ---- normalization
layer(L, 'BatchNorm1d training', lambda: nn.BatchNorm1d(6), [(8, 6, 5)], 1e-4)
layer(L, 'BatchNorm2d training (running statistics)', lambda: nn.BatchNorm2d(4, momentum=0.3), [(4, 4, 6, 6)], 1e-4)
layer(L, 'BatchNorm3d training', lambda: nn.BatchNorm3d(3), [(2, 3, 4, 4, 4)], 1e-4)
layer(L, 'BatchNorm2d eval', lambda: nn.BatchNorm2d(4), [(4, 4, 6, 6)], 1e-4, train=False)
layer(L, 'BatchNorm2d half', lambda: nn.BatchNorm2d(4), [(4, 4, 6, 6)], 3e-2, dtype=H)
layer(L, 'BatchNorm2d without statistics', lambda: nn.BatchNorm2d(4, track_running_stats=False), [(4, 4, 6, 6)], 1e-4)
layer(L, 'LayerNorm', lambda: nn.LayerNorm((5, 6)), [(4, 5, 6)], 1e-4)
layer(L, 'LayerNorm without affine', lambda: nn.LayerNorm(7, elementwise_affine=False), [(4, 3, 7)], 1e-4)
layer(L, 'LayerNorm half', lambda: nn.LayerNorm(32), [(4, 3, 32)], 3e-2, dtype=H)
layer(L, 'LayerNorm bfloat16', lambda: nn.LayerNorm(32), [(4, 3, 32)], 1.5e-1, dtype=B)
layer(L, 'GroupNorm', lambda: nn.GroupNorm(2, 6), [(3, 6, 5, 5)], 1e-4)
layer(L, 'GroupNorm half', lambda: nn.GroupNorm(2, 6), [(3, 6, 5, 5)], 3e-2, dtype=H)
layer(L, 'InstanceNorm1d affine', lambda: nn.InstanceNorm1d(4, affine=True), [(3, 4, 9)], 1e-4)
layer(L, 'InstanceNorm2d with running statistics', lambda: nn.InstanceNorm2d(4, track_running_stats=True), [(3, 4, 6, 6)], 1e-4)
layer(L, 'LocalResponseNorm', lambda: nn.LocalResponseNorm(3), [(2, 6, 5, 5)], 1e-4)
layer(L, 'RMSNorm', lambda: nn.RMSNorm(16), [(4, 3, 16)], 1e-4)
layer(L, 'RMSNorm half', lambda: nn.RMSNorm(16), [(4, 3, 16)], 3e-2, dtype=H)
layer(L, 'CrossMapLRN2d', lambda: nn.CrossMapLRN2d(3), [(2, 6, 5, 5)], 1e-4, tier='full')


def _batchnorm_stats(d):
    # Training updates the running statistics; a second pass in eval uses them.
    torch.manual_seed(0)
    m = nn.BatchNorm2d(5, momentum=0.2).to(d)
    for i in range(3):
        m(rnd(4, 5, 6, 6, seed=20 + i, scale=1 + i).to(d))
    m.eval()
    y = m(rnd(4, 5, 6, 6, seed=30).to(d))
    return [y, m.running_mean, m.running_var, m.num_batches_tracked]


check(L, 'BatchNorm2d running statistics over three steps, then eval', 1e-4)(_batchnorm_stats)

# ---- activations, losses of the simplest kind: elementwise, one case each
ACT = {
    'ReLU6': nn.ReLU6, 'LeakyReLU': lambda: nn.LeakyReLU(0.2), 'ELU': lambda: nn.ELU(1.3), 'SELU': nn.SELU, 'CELU': lambda: nn.CELU(0.7),
    'GELU': nn.GELU, 'GELU tanh': lambda: nn.GELU('tanh'), 'SiLU': nn.SiLU, 'Mish': nn.Mish, 'Softplus': lambda: nn.Softplus(2, 5),
    'Softshrink': lambda: nn.Softshrink(0.4), 'Hardshrink': lambda: nn.Hardshrink(0.4), 'Hardtanh': lambda: nn.Hardtanh(-0.8, 0.9),
    'Hardsigmoid': nn.Hardsigmoid, 'Hardswish': nn.Hardswish, 'LogSigmoid': nn.LogSigmoid, 'Softsign': nn.Softsign,
    'Tanhshrink': nn.Tanhshrink, 'Threshold': lambda: nn.Threshold(0.3, -2.0), 'Sigmoid': nn.Sigmoid, 'Tanh': nn.Tanh,
    'PReLU (per channel)': lambda: nn.PReLU(4), 'RReLU (eval)': nn.RReLU, 'GLU': lambda: nn.GLU(1),
    'Softmin': lambda: nn.Softmin(1), 'Softmax': lambda: nn.Softmax(1), 'LogSoftmax': lambda: nn.LogSoftmax(1), 'Softmax2d': nn.Softmax2d,
}


def _act(mk, dtype=None):
    def f(d):
        return fwd_bwd(mk, [rnd(3, 4, 5, 5, seed=2, scale=2)], d, dtype=dtype, train=False)
    return f


cases(L, 'activations (28 functions)', {k: _act(v) for k, v in ACT.items()}, 1e-4)
cases(L, 'activations in half', {k: _act(v, H) for k, v in ACT.items() if k not in ('RReLU (eval)',)}, 3e-2, tier='full')
cases(L, 'activations in bfloat16', {k: _act(v, B) for k, v in ACT.items() if k not in ('RReLU (eval)',)}, 1.5e-1, tier='full')

# ---- linear layers and embeddings
layer(L, 'Linear', lambda: nn.Linear(24, 10), [(5, 24)], 1e-4)
layer(L, 'Linear, three dimensions, no bias', lambda: nn.Linear(24, 10, bias=False), [(3, 5, 24)], 1e-4)
layer(L, 'Linear half', lambda: nn.Linear(32, 16), [(8, 32)], 3e-2, dtype=H)
layer(L, 'Bilinear', lambda: nn.Bilinear(6, 5, 4), [(7, 6), (7, 5)], 1e-4)
layer(L, 'Embedding with padding_idx', lambda: nn.Embedding(20, 8, padding_idx=3), [('i', 0, 20, 4, 6)], 1e-4)
layer(L, 'Embedding with max_norm', lambda: nn.Embedding(20, 8, max_norm=0.9), [('i', 0, 20, 4, 6)], 1e-4, tier='full')
layer(L, 'EmbeddingBag sum', lambda: nn.EmbeddingBag(20, 8, mode='sum'), [('i', 0, 20, 4, 6)], 1e-4)
layer(L, 'EmbeddingBag mean', lambda: nn.EmbeddingBag(20, 8, mode='mean'), [('i', 0, 20, 4, 6)], 1e-4)
layer(L, 'EmbeddingBag max', lambda: nn.EmbeddingBag(20, 8, mode='max'), [('i', 0, 20, 4, 6)], 1e-4)
layer(L, 'Flatten and Unflatten', lambda: nn.Sequential(nn.Flatten(), nn.Unflatten(1, (2, 24))), [(3, 4, 12)], 1e-6, tier='full')


def _emb_bag_offsets(d):
    torch.manual_seed(0)
    m = nn.EmbeddingBag(30, 6, mode='sum').to(d)
    idx = rint(0, 30, 17, seed=3).to(d)
    off = torch.tensor([0, 4, 4, 9, 13]).to(d)
    psw = rnd(17, seed=5).to(d)
    y = F.embedding_bag(idx, m.weight, off, mode='sum', per_sample_weights=psw)
    y.mul(rnd(*y.shape, seed=3).to(d)).sum().backward()
    return [y, m.weight.grad]


check(L, 'embedding_bag with offsets, empty bag and per-sample weights', 1e-4)(_emb_bag_offsets)


def _sparse_embedding(d):
    torch.manual_seed(0)
    m = nn.Embedding(50, 6, sparse=True).to(d)
    idx = rint(0, 50, 4, 5, seed=3).to(d)
    y = m(idx)
    y.mul(rnd(*y.shape, seed=3).to(d)).sum().backward()
    g = m.weight.grad
    return [y, g.coalesce().indices().float(), g.coalesce().values()]


check(L, 'Embedding with sparse gradients', 1e-4)(_sparse_embedding)


def _onehot(d):
    return [F.one_hot(rint(0, 7, 12, seed=3).to(d), 7), F.one_hot(rint(0, 5, 3, 4, seed=3).to(d))]


check(L, 'one_hot', 0)(_onehot)

# ---- dropout and friends in eval, and the shape-only layers
layer(L, 'Dropout variants in eval mode', lambda: nn.Sequential(nn.Dropout(0.5), nn.AlphaDropout(0.5), nn.Dropout2d(0.5), nn.FeatureAlphaDropout(0.5)),
      [(2, 3, 5, 5)], 1e-6, train=False)


def _distance(d):
    a = rnd(5, 6, seed=2).to(d).requires_grad_()
    b = rnd(5, 6, seed=3).to(d).requires_grad_()
    outs = [F.cosine_similarity(a, b), F.pairwise_distance(a, b), F.pdist(a), torch.cdist(a, b), torch.cdist(a, b, p=1.0)]
    sum((o * rnd(*o.shape, seed=7 + i).to(d)).sum() for i, o in enumerate(outs)).backward()
    return [outs, a.grad, b.grad]


check(L, 'cosine_similarity, pairwise_distance, pdist, cdist', 1e-4)(_distance)


def _normalize(d):
    x = rnd(4, 9, seed=2).to(d).requires_grad_()
    outs = [F.normalize(x, dim=1), F.normalize(x, p=1.0, dim=0), F.softmax(x, 0, dtype=torch.float64), F.gumbel_softmax(x, hard=False) * 0]
    sum((o * rnd(*o.shape, seed=7 + i, dtype=o.dtype).to(d)).sum() for i, o in enumerate(outs[:3])).backward()
    return [outs[:3], x.grad]


check(L, 'normalize and softmax with a dtype', 1e-5)(_normalize)
