"""torchvision's CUDA operators (nms, roi_align, roi_pool, deform_conv2d,
ps_roi_align, box ops), a ResNet block and a detection-style head: needs
torchvision (the sweep installs nothing; the caller's environment has it or
the checks are skipped)."""
import torch

from harness import cases, check, rnd, rint

L = 'vision'
V = ('torchvision',)


def boxes(n, seed, size=64.0):
    p = rnd(n, 4, seed=seed).abs() * size / 3
    p[:, 2:] = p[:, :2] + p[:, 2:] + 4
    return p


def _tv():
    import torchvision
    return torchvision


bx = boxes(60, 1)
sc = rnd(60, seed=2)
feat = rnd(2, 8, 24, 24, seed=3)
rois = torch.cat([torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0]]), boxes(5, 4, 20.0)], 1)


def ops():
    from torchvision import ops as o
    return o


cases(L, 'detection operators', {
    'nms': lambda d: ops().nms(bx.to(d), sc.to(d), 0.4),
    'batched_nms': lambda d: ops().batched_nms(bx.to(d), sc.to(d), (torch.arange(60) % 3).to(d), 0.4),
    'roi_align (aligned, sampling 2)': lambda d: ops().roi_align(feat.to(d), rois.to(d), (5, 5), 0.5, 2, True),
    'roi_align backward': lambda d: _grad(d, lambda f: ops().roi_align(f, rois.to(d), (5, 5), 0.5, 2, False)),
    'roi_pool': lambda d: ops().roi_pool(feat.to(d), rois.to(d), (4, 4), 0.5),
    'roi_pool backward': lambda d: _grad(d, lambda f: ops().roi_pool(f, rois.to(d), (4, 4), 0.5)),
    'ps_roi_align': lambda d: ops().ps_roi_align(rnd(2, 16, 24, 24, seed=3).to(d), rois.to(d), 2, 0.5, 2),
    'ps_roi_pool': lambda d: ops().ps_roi_pool(rnd(2, 16, 24, 24, seed=3).to(d), rois.to(d), 2, 0.5),
    'box_iou, generalized_box_iou, box_area': lambda d: [ops().box_iou(bx.to(d), bx.to(d)), ops().generalized_box_iou(bx.to(d), bx.flip(0).to(d)), ops().box_area(bx.to(d))],
    'clip_boxes_to_image, remove_small_boxes': lambda d: [ops().clip_boxes_to_image(bx.to(d), (40, 50)), ops().remove_small_boxes(bx.to(d), 8.0)],
    'sigmoid_focal_loss': lambda d: ops().sigmoid_focal_loss(rnd(20, 5, seed=2).to(d), (rnd(20, 5, seed=3) > 0).float().to(d), reduction='mean'),
    'complete_box_iou_loss and distance_box_iou_loss': lambda d: [ops().complete_box_iou_loss(bx[:20].to(d), bx[20:40].to(d)), ops().distance_box_iou_loss(bx[:20].to(d), bx[20:40].to(d))],
}, 1e-4, needs=V)


def _grad(d, f):
    x = feat.to(d).clone().requires_grad_()
    y = f(x)
    (y * rnd(*y.shape, seed=7).to(d)).sum().backward()
    return [y, x.grad]


def _deform(d):
    from torchvision import ops as o
    x = rnd(2, 4, 10, 10, seed=1).to(d).requires_grad_()
    w = rnd(6, 4, 3, 3, seed=2).to(d).requires_grad_()
    off = (rnd(2, 18, 8, 8, seed=3) * 0.5).to(d).requires_grad_()
    mask = torch.sigmoid(rnd(2, 9, 8, 8, seed=4)).to(d).requires_grad_()
    y = o.deform_conv2d(x, off, w, mask=mask)
    (y * rnd(*y.shape, seed=7).to(d)).sum().backward()
    return [y, x.grad, w.grad, off.grad, mask.grad]


check(L, 'deform_conv2d (v2, with a mask): forward and backward', 2e-3, needs=V)(_deform)


def _deform_groups(d):
    from torchvision import ops as o
    x = rnd(1, 4, 9, 9, seed=1).to(d).requires_grad_()
    w = rnd(4, 2, 3, 3, seed=2).to(d).requires_grad_()
    off = (rnd(1, 2 * 2 * 9, 9, 9, seed=3) * 0.4).to(d).requires_grad_()
    y = o.deform_conv2d(x, off, w, padding=1, stride=1)
    y.square().sum().backward()
    return [y, x.grad, w.grad, off.grad]


check(L, 'deform_conv2d with groups and two offset groups', 2e-3, needs=V, tier='full')(_deform_groups)


def _resnet(d):
    import torchvision
    torch.manual_seed(0)
    m = torchvision.models.resnet18(weights=None, num_classes=10)
    x = rnd(2, 3, 32, 32, seed=1)
    from harness import fwd_bwd
    r = fwd_bwd(lambda: m, [x], d, train=False)
    return [r[0], r[1], r[2][:4]]


check(L, 'resnet18 forward (eval) and the gradients of its first layers', 2e-3, needs=V)(_resnet)


def _mobilenet(d):
    import torchvision
    from harness import fwd_bwd
    x = rnd(2, 3, 32, 32, seed=1)
    return fwd_bwd(lambda: torchvision.models.mobilenet_v3_small(weights=None, num_classes=10), [x], d, train=False)[:2]


check(L, 'mobilenet_v3_small forward and input gradient (depthwise convolutions, hardswish, squeeze-excite)', 2e-3, needs=V, tier='full')(_mobilenet)


def _transforms(d):
    from torchvision.transforms import v2
    img = (rnd(3, 40, 48, seed=1).abs() * 80).clamp(0, 255).to(torch.uint8)
    f = lambda t: [v2.functional.resize(t, [20, 24], antialias=True), v2.functional.rotate(t.float(), 15.0), v2.functional.adjust_brightness(t, 1.3), v2.functional.gaussian_blur(t.float(), [5, 5]), v2.functional.horizontal_flip(t), v2.functional.normalize(t.float(), [0.5] * 3, [0.2] * 3), v2.functional.rgb_to_grayscale(t)]
    return [o.float() for o in f(img.to(d))]


check(L, 'transforms v2: resize with antialias, rotate, brightness, blur, flip, normalize, grayscale', 2.0, needs=V, tier='full')(_transforms)
