"""torch.fft beyond ops.py: every transform and norm mode, sizes that are not
powers of two, double and half precision, strided input, stft/istft, windows."""
import torch

from harness import cases, rnd

L = 'fft'
x = rnd(4, 30, seed=1)
xc = torch.complex(rnd(4, 30, seed=1), rnd(4, 30, seed=2))
img = rnd(2, 3, 12, 10, seed=3)
vol = rnd(1, 2, 6, 8, 5, seed=4)
R = torch.view_as_real


def c(t):
    return R(t.resolve_conj()) if t.is_complex() else t


def many(fs):
    return lambda d: [c(f(d)) for f in fs]


cases(L, 'one-dimensional transforms, every norm', {
    f'{fn} norm={nm}': (lambda d, fn=fn, nm=nm: c(getattr(torch.fft, fn)((xc if fn in ('fft', 'ifft', 'hfft') else x).to(d), norm=nm)))
    for fn in ('fft', 'ifft', 'rfft', 'ihfft', 'hfft') for nm in ('backward', 'ortho', 'forward')
}, 2e-4)

cases(L, 'sizes: padded, trimmed, primes and odd lengths', {
    'fft n=17 (prime)': lambda d: c(torch.fft.fft(xc.to(d), n=17)),
    'fft n=64 (padded)': lambda d: c(torch.fft.fft(xc.to(d), n=64)),
    'fft n=127': lambda d: c(torch.fft.fft(rnd(3, 127, seed=5).to(d))),
    'rfft n=45': lambda d: c(torch.fft.rfft(x.to(d), n=45)),
    'irfft n=29 (odd)': lambda d: torch.fft.irfft(torch.fft.rfft(x.to(d)), n=29),
    'irfft round trip n=30': lambda d: torch.fft.irfft(torch.fft.rfft(x.to(d)), n=30),
    'fft n=1000 (2^3 5^3)': lambda d: c(torch.fft.fft(rnd(2, 1000, seed=6).to(d))),
    'fft n=4096': lambda d: c(torch.fft.fft(rnd(2, 4096, seed=6).to(d))),
    'fft along dim 0': lambda d: c(torch.fft.fft(xc.to(d), dim=0)),
    'fft of a transposed (strided) input': lambda d: c(torch.fft.fft(xc.to(d).t(), dim=0)),
    'fft of a sliced (strided) input': lambda d: c(torch.fft.fft(xc.to(d)[:, 3::2])),
}, 2e-4)

cases(L, 'multi-dimensional transforms', {
    'fft2': lambda d: c(torch.fft.fft2(img.to(d))),
    'ifft2 ortho': lambda d: c(torch.fft.ifft2(torch.complex(img, img.flip(-1)).to(d), norm='ortho')),
    'rfft2 and irfft2 with s': lambda d: [c(torch.fft.rfft2(img.to(d), s=(8, 14))), torch.fft.irfft2(torch.fft.rfft2(img.to(d)), s=(12, 10))],
    'hfft2 and ihfft2': lambda d: [torch.fft.hfft2(torch.complex(img, img.flip(-1)).to(d)), c(torch.fft.ihfft2(img.to(d)))],
    'fftn over three dimensions': lambda d: c(torch.fft.fftn(vol.to(d), dim=(-3, -2, -1))),
    'rfftn and irfftn over three dimensions': lambda d: [c(torch.fft.rfftn(vol.to(d), dim=(1, 2, 3))), torch.fft.irfftn(torch.fft.rfftn(vol.to(d), dim=(1, 2, 3)), s=(6, 8, 5), dim=(1, 2, 3))],
    'fftn over non-adjacent dimensions': lambda d: c(torch.fft.fftn(vol.to(d), dim=(1, 3))),
    'fft2 of a channels-last image': lambda d: c(torch.fft.fft2(img.to(d).contiguous(memory_format=torch.channels_last))),
}, 5e-4)

cases(L, 'double precision, half precision and helpers', {
    'fft double': lambda d: c(torch.fft.fft(xc.double().to(d))),
    'rfft2 double': lambda d: c(torch.fft.rfft2(img.double().to(d))),
    'fft complex double 3d': lambda d: c(torch.fft.fftn(torch.complex(vol, vol).double().to(d))),
    'rfft half (power of two)': lambda d: c(torch.fft.rfft(x[:, :16].half().to(d)).to(torch.complex64)) if d != 'cpu' else c(torch.fft.rfft(x[:, :16].half().float())),
    'fftshift, ifftshift, fftfreq, rfftfreq': lambda d: [torch.fft.fftshift(x.to(d)), torch.fft.ifftshift(x.to(d), dim=1), torch.fft.fftfreq(11, 0.5).to(d), torch.fft.rfftfreq(11).to(d)],
    'convolution by FFT': lambda d: torch.fft.irfft(torch.fft.rfft(x.to(d), n=64) * torch.fft.rfft(x.flip(1).to(d), n=64), n=64),
    'spectrogram: stft magnitude': lambda d: torch.stft(rnd(2, 400, seed=2).to(d), 64, 16, window=torch.hann_window(64, device=d), return_complex=True).abs(),
    'stft with center and reflect padding': lambda d: c(torch.stft(rnd(1, 300, seed=2).to(d), 32, 8, 32, torch.hamming_window(32, device=d), center=True, pad_mode='reflect', normalized=True, return_complex=True)),
    'istft round trip': lambda d: torch.istft(torch.stft(rnd(2, 400, seed=2).to(d), 64, 16, window=torch.hann_window(64, device=d), return_complex=True), 64, 16, window=torch.hann_window(64, device=d), length=400),
    'windows (hann, hamming, blackman, bartlett, kaiser)': lambda d: [f(33, device=d) for f in (torch.hann_window, torch.hamming_window, torch.blackman_window, torch.bartlett_window, torch.kaiser_window)],
    'gradient through fft and ifft': lambda d: _fft_grad(d),
    'gradient through rfft2': lambda d: _rfft_grad(d),
}, 2e-3)


def _fft_grad(d):
    z = torch.complex(rnd(3, 16, seed=1), rnd(3, 16, seed=2)).to(d).requires_grad_()
    y = torch.fft.ifft(torch.fft.fft(z) * 2)
    (y.abs().square().sum() + (y.real * 3).sum()).backward()
    return [c(y), c(z.grad)]


def _rfft_grad(d):
    z = rnd(2, 12, 10, seed=1).to(d).requires_grad_()
    y = torch.fft.irfft2(torch.fft.rfft2(z) * 0.5, s=(12, 10))
    (y * rnd(2, 12, 10, seed=3).to(d)).sum().backward()
    return [y, z.grad]
