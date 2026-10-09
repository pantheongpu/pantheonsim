"""MultiheadAttention and the Transformer layers, forward and backward (their
fused fast paths in eval, the math and fused-SDPA paths in training)."""
import torch
import torch.nn as nn

from harness import check, fwd_bwd, layer, rnd, flat

L = 'attention'
H = torch.float16
T = [(3, 7, 16)]  # (batch, sequence, model)

layer(L, 'MultiheadAttention (self, batch_first)', lambda: _MHA(nn.MultiheadAttention(16, 4, batch_first=True)), T, 1e-4)
layer(L, 'MultiheadAttention with key padding mask and attention weights',
      lambda: _MHA(nn.MultiheadAttention(16, 4, batch_first=True), mask=True), T, 1e-4)
layer(L, 'MultiheadAttention with kdim and vdim', lambda: _MHAkv(), [(3, 7, 16), (3, 9, 10)], 1e-4)
layer(L, 'MultiheadAttention with bias_kv and add_zero_attn', lambda: _MHA(nn.MultiheadAttention(16, 4, add_bias_kv=True, add_zero_attn=True,
                                                                                                     batch_first=True)), T, 1e-4, tier='full')
layer(L, 'MultiheadAttention half', lambda: _MHA(nn.MultiheadAttention(16, 4, batch_first=True)), T, 3e-2, dtype=H)


class _MHA(nn.Module):
    def __init__(self, mha, mask=False):
        super().__init__()
        self.mha, self.mask = mha, mask

    def forward(self, x):
        kpm = None
        if self.mask:
            kpm = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
            kpm[:, -2:] = True
        causal = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), 1)
        out, w = self.mha(x, x, x, key_padding_mask=kpm, attn_mask=causal, need_weights=True)
        return out, w


class _MHAkv(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(16, 2, kdim=10, vdim=10, batch_first=True)

    def forward(self, q, kv):
        return self.mha(q, kv, kv)[0]


def enc(norm_first=False, act='relu', **kw):
    return lambda: nn.TransformerEncoderLayer(16, 4, 32, dropout=0.0, batch_first=True, norm_first=norm_first, activation=act, **kw)


layer(L, 'TransformerEncoderLayer', enc(), T, 1e-4)
layer(L, 'TransformerEncoderLayer, pre-norm, GELU', enc(True, 'gelu'), T, 1e-4)
layer(L, 'TransformerEncoderLayer half', enc(), T, 3e-2, dtype=H)
layer(L, 'TransformerDecoderLayer', lambda: _Dec(), [(3, 5, 16), (3, 7, 16)], 1e-4)
layer(L, 'nn.Transformer (2 encoder, 2 decoder layers)', lambda: _Tr(), [(3, 7, 16), (3, 5, 16)], 1e-4)


class _Dec(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.TransformerDecoderLayer(16, 4, 32, dropout=0.0, batch_first=True)

    def forward(self, tgt, mem):
        return self.l(tgt, mem, tgt_mask=nn.Transformer.generate_square_subsequent_mask(tgt.shape[1], device=tgt.device), tgt_is_causal=True)


class _Tr(nn.Module):
    def __init__(self):
        super().__init__()
        self.t = nn.Transformer(16, 4, 2, 2, 32, dropout=0.0, batch_first=True)

    def forward(self, src, tgt):
        return self.t(src, tgt, tgt_mask=self.t.generate_square_subsequent_mask(tgt.shape[1], device=src.device))


def _fast_path(d):
    # Eval, no grad: the fused encoder-layer kernels (torch._transformer_encoder_layer_fwd),
    # with a padding mask so the nested-tensor path is taken.
    torch.manual_seed(0)
    m = nn.TransformerEncoder(nn.TransformerEncoderLayer(16, 4, 32, dropout=0.0, batch_first=True), 2, enable_nested_tensor=True).to(d).eval()
    x = rnd(3, 7, 16, seed=2).to(d)
    kpm = torch.zeros(3, 7, dtype=torch.bool, device=d)
    kpm[0, 5:] = True
    kpm[2, 3:] = True
    with torch.no_grad():
        y = m(x, src_key_padding_mask=kpm)
    return y.masked_fill(kpm.unsqueeze(-1), 0)


check(L, 'TransformerEncoder fast path with nested tensors (eval)', 1e-4)(_fast_path)


def _fast_path_mha(d):
    torch.manual_seed(0)
    m = nn.MultiheadAttention(16, 4, batch_first=True).to(d).eval()
    x = rnd(3, 7, 16, seed=2).to(d)
    with torch.no_grad():
        return m(x, x, x, need_weights=False)[0]


check(L, 'MultiheadAttention fast path (eval, no weights)', 1e-4, tier='full')(_fast_path_mha)


def _train_step(d):
    # A tiny encoder-only language model, three AdamW steps.
    torch.manual_seed(0)
    emb = nn.Embedding(32, 16)
    enc_ = nn.TransformerEncoder(nn.TransformerEncoderLayer(16, 4, 32, dropout=0.0, batch_first=True, norm_first=True), 2)
    head = nn.Linear(16, 32)
    m = nn.ModuleList([emb, enc_, head]).to(d)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    ids = (torch.arange(24).reshape(3, 8) * 5 % 32).to(d)
    mask = nn.Transformer.generate_square_subsequent_mask(8, device=d)
    losses = []
    for _ in range(3):
        opt.zero_grad()
        logits = head(enc_(emb(ids), mask=mask, is_causal=True))
        loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 32), ids[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.detach())
    return [torch.stack(losses), list(m.parameters())]


check(L, 'tiny causal transformer, three AdamW steps', 2e-3)(_train_step)
