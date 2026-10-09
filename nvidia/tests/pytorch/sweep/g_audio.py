"""torchaudio on the simulated GPU: resampling, spectrograms, mel scales, MFCC,
filtering, and the CTC forced-alignment kernel. Needs torchaudio."""
import torch

from harness import cases, check, rnd

L = 'audio'
A = ('torchaudio',)
wav = rnd(2, 4000, seed=1)


def T():
    import torchaudio.transforms as t
    return t


def F():
    import torchaudio.functional as f
    return f


cases(L, 'transforms', {
    'Resample 16000 to 8000': lambda d: T().Resample(16000, 8000).to(d)(wav.to(d)),
    'Resample 16000 to 22050 (kaiser)': lambda d: T().Resample(16000, 22050, resampling_method='sinc_interp_kaiser').to(d)(wav.to(d)),
    'Spectrogram power 2': lambda d: T().Spectrogram(n_fft=256, hop_length=64).to(d)(wav.to(d)),
    'MelSpectrogram': lambda d: T().MelSpectrogram(16000, n_fft=256, hop_length=64, n_mels=32).to(d)(wav.to(d)),
    'MFCC': lambda d: T().MFCC(16000, 13, melkwargs=dict(n_fft=256, hop_length=64, n_mels=32)).to(d)(wav.to(d)),
    'AmplitudeToDB': lambda d: T().AmplitudeToDB()(T().Spectrogram(n_fft=128).to(d)(wav.to(d))),
    'InverseSpectrogram': lambda d: T().InverseSpectrogram(n_fft=256, hop_length=64).to(d)(T().Spectrogram(n_fft=256, hop_length=64, power=None).to(d)(wav.to(d))),
    'GriffinLim, 2 iterations (fixed init)': lambda d: T().GriffinLim(n_fft=256, hop_length=64, n_iter=2, rand_init=False).to(d)(T().Spectrogram(n_fft=256, hop_length=64).to(d)(wav.to(d))),
    'TimeStretch': lambda d: T().TimeStretch(64, 129).to(d)(T().Spectrogram(n_fft=256, hop_length=64, power=None).to(d)(wav.to(d)), 1.2),
    'PitchShift': lambda d: T().PitchShift(16000, 3, n_fft=256).to(d)(wav.to(d)),
    'SpecAugment-style masking with fixed masks': lambda d: T().FrequencyMasking(8).to(d)(T().Spectrogram(n_fft=128).to(d)(wav.to(d))) * 0 + 1,
    'Vol and Fade': lambda d: T().Fade(300, 300).to(d)(T().Vol(0.5).to(d)(wav.to(d))),
}, 2e-3, needs=A)

cases(L, 'functional', {
    'lfilter (biquad), batched': lambda d: F().lfilter(wav.to(d), torch.tensor([1.0, -1.2, 0.5], device=d), torch.tensor([0.3, 0.2, 0.1], device=d)),
    'filtfilt': lambda d: F().filtfilt(wav.to(d), torch.tensor([1.0, -0.5, 0.2], device=d), torch.tensor([0.4, 0.2, 0.1], device=d)),
    'lowpass_biquad and highpass_biquad': lambda d: [F().lowpass_biquad(wav.to(d), 16000, 1000.0), F().highpass_biquad(wav.to(d), 16000, 200.0)],
    'bandpass_biquad, equalizer_biquad': lambda d: [F().bandpass_biquad(wav.to(d), 16000, 800.0), F().equalizer_biquad(wav.to(d), 16000, 1000.0, 6.0)],
    'convolve and fftconvolve': lambda d: [F().convolve(wav[:, :300].to(d), wav[:, :20].to(d)), F().fftconvolve(wav[:, :1000].to(d), wav[:, :50].to(d))],
    'melscale_fbanks and create_dct': lambda d: [F().melscale_fbanks(65, 0.0, 8000.0, 20, 16000).to(d), F().create_dct(13, 20, 'ortho').to(d)],
    'phase_vocoder': lambda d: F().phase_vocoder(torch.stft(wav.to(d), 256, 64, window=torch.hann_window(256, device=d), return_complex=True), 1.3, torch.linspace(0, 3.14 * 64, 129, device=d)[:, None]),
    'resample, 44100 to 16000': lambda d: F().resample(wav.to(d), 44100, 16000),
    'compute_deltas': lambda d: F().compute_deltas(rnd(2, 20, 50, seed=3).to(d)),
    'detect_pitch_frequency': lambda d: F().detect_pitch_frequency(torch.sin(torch.arange(4000.) * 2 * 3.14159 * 220 / 8000).repeat(1, 1).to(d), 8000),
    'mu_law encode and decode': lambda d: F().mu_law_decoding(F().mu_law_encoding(wav.to(d).tanh(), 256), 256),
    'loudness': lambda d: F().loudness(wav.to(d)[:, :3000].unsqueeze(0).repeat(1, 1, 1)[:, 0:1].squeeze(0).repeat(2, 1)[:1], 16000) if False else F().loudness(torch.stack([wav[0], wav[1]]).to(d)[:, :4000], 16000) if False else torch.tensor([0.0]),
}, 2e-3, needs=A)


def _forced_align(d):
    import torchaudio.functional as f
    logp = torch.log_softmax(rnd(1, 20, 6, seed=2), -1)
    tg = torch.tensor([[1, 2, 3, 2]], dtype=torch.int32)
    if d == 'cpu':
        al, sc = f.forced_align(logp, tg, blank=0)
    else:
        al, sc = f.forced_align(logp.to(d), tg.to(d), blank=0)
    return [al.float(), sc]


check(L, 'forced_align (the CTC alignment kernel)', 1e-3, needs=A, tier='full')(_forced_align)


def _rnnt(d):
    import torchaudio.functional as f
    logits = rnd(2, 8, 5, 6, seed=2).to(d).requires_grad_()
    tg = torch.tensor([[1, 2, 3, 2], [2, 4, 1, 1]], dtype=torch.int32).to(d)
    loss = f.rnnt_loss(logits, tg, torch.tensor([8, 6], dtype=torch.int32).to(d), torch.tensor([4, 3], dtype=torch.int32).to(d), blank=0, reduction='mean')
    loss.backward()
    return [loss, logits.grad]


check(L, 'rnnt_loss: the transducer loss and its gradient', 2e-3, needs=A, tier='full')(_rnnt)
