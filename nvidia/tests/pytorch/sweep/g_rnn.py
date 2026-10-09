"""Recurrent layers, forward and backward: cuDNN's RNN API and the native path."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from harness import check, fwd_bwd, layer, rnd, flat

L = 'rnn'
H = torch.float16
X = [(7, 3, 6)]  # (time, batch, features)

layer(L, 'RNN tanh', lambda: nn.RNN(6, 8), X, 1e-4)
layer(L, 'RNN relu, two layers', lambda: nn.RNN(6, 8, 2, nonlinearity='relu'), X, 1e-4)
layer(L, 'LSTM', lambda: nn.LSTM(6, 8), X, 1e-4)
layer(L, 'LSTM two layers, bidirectional', lambda: nn.LSTM(6, 8, 2, bidirectional=True), X, 1e-4)
layer(L, 'LSTM batch_first with projections', lambda: nn.LSTM(6, 8, 2, batch_first=True, proj_size=4), [(3, 7, 6)], 1e-4)
layer(L, 'GRU', lambda: nn.GRU(6, 8), X, 1e-4)
layer(L, 'GRU two layers, bidirectional, batch_first', lambda: nn.GRU(6, 8, 2, bidirectional=True, batch_first=True), [(3, 7, 6)], 1e-4)
layer(L, 'LSTM half', lambda: nn.LSTM(6, 8, 2), X, 3e-2, dtype=H)
layer(L, 'GRU half', lambda: nn.GRU(6, 8), X, 3e-2, dtype=H)
layer(L, 'LSTM bfloat16', lambda: nn.LSTM(6, 8), X, 2e-1, dtype=torch.bfloat16, tier='full')
layer(L, 'RNNCell', lambda: nn.RNNCell(6, 8), [(3, 6)], 1e-4)
layer(L, 'LSTMCell', lambda: nn.LSTMCell(6, 8), [(3, 6)], 1e-4)
layer(L, 'GRUCell', lambda: nn.GRUCell(6, 8), [(3, 6)], 1e-4)


def _no_cudnn(mk):
    def f(d):
        with torch.backends.cudnn.flags(enabled=False):
            return fwd_bwd(mk, [rnd(7, 3, 6, seed=10)], d)
    return f


check(L, 'LSTM with cuDNN disabled (native fused cell kernels)', 1e-4)(_no_cudnn(lambda: nn.LSTM(6, 8, 2)))
check(L, 'GRU with cuDNN disabled', 1e-4, tier='full')(_no_cudnn(lambda: nn.GRU(6, 8, 2, bidirectional=True)))


def _packed(d):
    torch.manual_seed(0)
    m = nn.LSTM(6, 8, 2, bidirectional=True, batch_first=True).to(d)
    x = rnd(4, 9, 6, seed=3).to(d).requires_grad_()
    lens = torch.tensor([9, 7, 4, 2])
    out, (h, c) = m(pack_padded_sequence(x, lens, batch_first=True))
    y, _ = pad_packed_sequence(out, batch_first=True)
    (y * rnd(*y.shape, seed=4).to(d)).sum().backward()
    return [y, h, c, x.grad, [p.grad for p in m.parameters()]]


check(L, 'packed sequences through a bidirectional LSTM', 1e-4)(_packed)


def _packed_gru_unsorted(d):
    torch.manual_seed(0)
    m = nn.GRU(6, 5, batch_first=True).to(d)
    seqs = [rnd(n, 6, seed=20 + n).to(d) for n in (3, 8, 5)]
    padded = pad_sequence(seqs, batch_first=True).requires_grad_()
    out, h = m(pack_padded_sequence(padded, torch.tensor([3, 8, 5]), batch_first=True, enforce_sorted=False))
    y, _ = pad_packed_sequence(out, batch_first=True)
    (y.square()).sum().backward()
    return [y, h, padded.grad]


check(L, 'packed unsorted sequences through a GRU', 1e-4, tier='full')(_packed_gru_unsorted)


def _lstm_state(d):
    torch.manual_seed(0)
    m = nn.LSTM(6, 8, 2).to(d)
    x = rnd(5, 3, 6, seed=3).to(d)
    h0, c0 = rnd(2, 3, 8, seed=4).to(d).requires_grad_(), rnd(2, 3, 8, seed=5).to(d).requires_grad_()
    out, (h, c) = m(x, (h0, c0))
    (out.sum() + h.square().sum() + c.sum()).backward()
    return [out, h, c, h0.grad, c0.grad]


check(L, 'LSTM with an initial state, gradients to the state', 1e-4)(_lstm_state)


def _lstm_dropout_eval(d):
    torch.manual_seed(0)
    m = nn.LSTM(6, 8, 3, dropout=0.5).to(d).eval()
    return m(rnd(5, 3, 6, seed=3).to(d))[0]


check(L, 'LSTM with inter-layer dropout in eval', 1e-4, tier='full')(_lstm_dropout_eval)


def _seq2seq(d):
    # An encoder-decoder with teacher forcing, two optimizer steps.
    torch.manual_seed(0)
    enc, dec, head = nn.GRU(8, 16, batch_first=True), nn.GRU(8, 16, batch_first=True), nn.Linear(16, 11)
    mods = nn.ModuleList([enc, dec, head]).to(d)
    opt = torch.optim.Adam(mods.parameters(), lr=1e-2)
    src, tgt = rnd(4, 6, 8, seed=1).to(d), rnd(4, 5, 8, seed=2).to(d)
    lab = (torch.arange(20).reshape(4, 5) % 11).to(d)
    losses = []
    for _ in range(2):
        opt.zero_grad()
        _, h = enc(src)
        out, _ = dec(tgt, h)
        loss = nn.functional.cross_entropy(head(out).reshape(-1, 11), lab.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.detach())
    return [torch.stack(losses), [p for p in mods.parameters()]]


check(L, 'GRU encoder-decoder, two Adam steps', 1e-3)(_seq2seq)
