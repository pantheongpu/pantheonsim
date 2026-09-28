"""Whole models in PyTorch's ROCm build, unmodified, on a simulated AMD GPU.

Where check.py runs single operators and two toy models three steps each,
these are the shapes real workloads have, small enough for a simulator: a
GPT-style decoder trained with a schedule, clipping and weight decay, then
generating with a KV cache; a ResNet; an LSTM; a U-Net block; a vision
transformer; mixture-of-experts routing; a DLRM-style recommender; mixed
precision with a gradient scaler; activation checkpointing; and a checkpoint
saved on the GPU and loaded on the CPU. Each runs on the simulated GPU and on
the CPU from the same weights and inputs, and the two are compared. One line
each: "ok <name>", or "FAIL <name>: <why>".
"""
import io
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

GPU = 'cuda'


def report(name, gpu, cpu, tol):
    gpu, cpu = gpu.detach().float().cpu(), cpu.detach().float()
    if gpu.shape != cpu.shape:
        print(f"FAIL {name}: shape {tuple(gpu.shape)}, the CPU's {tuple(cpu.shape)}", flush=True)
        return
    diff = float((gpu - cpu).abs().max())
    print(f"ok {name}" if diff <= tol else f"FAIL {name}: max difference {diff:g}, allowed {tol:g}", flush=True)


def check(name, f, tol=1e-4):
    try:
        report(name, f(GPU), f('cpu'), tol)
    except Exception as e:  # a refused kernel shows up here, and says which
        print(f"FAIL {name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)


def fit(model_fn, batches, loss_fn, lr=3e-3, clip=None, schedule=False, weight_decay=0.01):
    """Trains a fresh model_fn() on device d, one step per batch; returns the losses."""
    def run(d):
        torch.manual_seed(0)
        model = model_fn().to(d)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=len(batches)) if schedule else None
        losses = []
        for batch in batches:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model, *(t.to(d) for t in batch))
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if sched:
                sched.step()
            losses.append(loss.detach())
        return torch.stack(losses)
    return run


g = torch.Generator().manual_seed(7)

# --- A GPT-style decoder: pre-norm blocks, causal attention, GELU MLP, tied
# embeddings, AdamW with weight decay, a one-cycle schedule and clipping.
VOCAB, CTX, DIM, HEADS, LAYERS = 64, 16, 64, 4, 2


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(DIM), nn.LayerNorm(DIM)
        self.qkv, self.proj = nn.Linear(DIM, 3 * DIM), nn.Linear(DIM, DIM)
        self.mlp = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(), nn.Linear(4 * DIM, DIM))

    def forward(self, x, cache=None):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, HEADS, DIM // HEADS).permute(2, 0, 3, 1, 4)
        if cache is not None:
            if 'k' in cache:
                k, v = torch.cat([cache['k'], k], 2), torch.cat([cache['v'], v], 2)
            cache['k'], cache['v'] = k, v
        # Causal only over the new positions: with a cache, each new query sees
        # every earlier key.
        a = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None or t > 1)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, DIM))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(VOCAB, DIM), nn.Embedding(CTX, DIM)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.ln = nn.LayerNorm(DIM)

    def forward(self, idx, caches=None, start=0):
        x = self.tok(idx) + self.pos(torch.arange(start, start + idx.shape[1], device=idx.device))
        for i, blk in enumerate(self.blocks):
            x = blk(x, None if caches is None else caches[i])
        return self.ln(x) @ self.tok.weight.t()


# Text with structure to learn: each sequence counts up from a random start.
seqs = (torch.randint(0, VOCAB, (8, 1), generator=g) + torch.arange(CTX + 1)) % VOCAB
gpt_batches = [(seqs[i:i + 4],) for i in (0, 4, 0, 4, 0, 4)]


def lm_loss(m, s):
    return F.cross_entropy(m(s[:, :-1]).reshape(-1, VOCAB), s[:, 1:].reshape(-1))


check('gpt training (AdamW, one-cycle schedule, clipping)',
      fit(GPT, gpt_batches, lm_loss, lr=3e-3, clip=1.0, schedule=True), 2e-4)


def generate(d):
    """Greedy decoding with a KV cache after the same training, and the logits of every step."""
    torch.manual_seed(0)
    model = GPT().to(d)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for (s,) in gpt_batches:
        opt.zero_grad()
        lm_loss(model, s.to(d)).backward()
        opt.step()
    model.eval()
    caches = [{} for _ in range(LAYERS)]
    idx = seqs[:2, :4].to(d)
    with torch.no_grad():
        logits = [model(idx, caches)[:, -1]]
        out = [logits[-1].argmax(-1)]
        for step in range(4, CTX - 1):
            logits.append(model(out[-1][:, None], caches, start=step)[:, -1])
            out.append(logits[-1].argmax(-1))
    return torch.cat([torch.stack(out, 1).float(), torch.stack(logits, 1).flatten(1)], 1)


check('gpt generation with a KV cache', generate, 2e-3)


# --- A ResNet: basic blocks with batch norm and strided shortcuts, trained,
# then evaluated with the running statistics it kept.
class Basic(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout)
        self.c2, self.b2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout)
        self.short = nn.Sequential() if stride == 1 and cin == cout else \
            nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.short(x))


def resnet():
    return nn.Sequential(nn.Conv2d(3, 8, 3, 1, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
                         Basic(8, 8, 1), Basic(8, 16, 2), Basic(16, 32, 2),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10))


imgs, labels = torch.randn(8, 3, 16, 16, generator=g), torch.randint(0, 10, (8,), generator=g)


def resnet_run(d):
    losses = fit(resnet, [(imgs, labels)] * 3, lambda m, x, y: F.cross_entropy(m(x), y))(d)
    torch.manual_seed(0)
    model = resnet().to(d)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    for _ in range(2):
        opt.zero_grad()
        F.cross_entropy(model(imgs.to(d)), labels.to(d)).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return torch.cat([losses, model(imgs.to(d)).flatten()])


check('resnet training, then inference on running statistics', resnet_run, 5e-4)


# --- An LSTM language model, two layers, packed variable-length batches.
class Lstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.rnn, self.out = nn.Embedding(40, 24), nn.LSTM(24, 32, 2, batch_first=True), nn.Linear(32, 40)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        y, _ = self.rnn(packed)
        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=x.shape[1])
        return self.out(y)


words = torch.randint(1, 40, (6, 10), generator=g)
lengths = torch.tensor([10, 7, 9, 4, 10, 6])


def lstm_loss(m, x, n):
    mask = torch.arange(x.shape[1], device=x.device)[None, :-1] < (n[:, None] - 1)
    logits = m(x, n)[:, :-1]
    return F.cross_entropy(logits[mask], x[:, 1:][mask])


check('lstm training on packed sequences', fit(Lstm, [(words, lengths)] * 3, lstm_loss), 2e-4)


# --- A U-Net block, as a diffusion model has: group norm, SiLU, a timestep
# embedding, a strided down path, a transposed-convolution up path, skip
# concatenation and bilinear upsampling.
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.t = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 16))
        self.inp = nn.Conv2d(3, 16, 3, padding=1)
        self.down = nn.Sequential(nn.GroupNorm(4, 16), nn.SiLU(), nn.Conv2d(16, 32, 3, 2, 1))
        self.mid = nn.Sequential(nn.GroupNorm(8, 32), nn.SiLU(), nn.Conv2d(32, 32, 3, padding=1))
        self.up = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.outp = nn.Sequential(nn.GroupNorm(4, 32), nn.SiLU(), nn.Conv2d(32, 3, 3, padding=1))

    def forward(self, x, t):
        half = torch.exp(-math.log(1000) * torch.arange(8, device=x.device) / 8)
        emb = self.t(torch.cat([torch.sin(t[:, None] * half), torch.cos(t[:, None] * half)], 1))
        h = self.inp(x) + emb[:, :, None, None]
        u = self.up(self.mid(self.down(h)))
        u = u + F.interpolate(self.down(h), scale_factor=2, mode='bilinear', align_corners=False)[:, :16]
        return self.outp(torch.cat([h, u], 1))


noisy, steps, noise = torch.randn(4, 3, 16, 16, generator=g), torch.rand(4, generator=g) * 1000, \
    torch.randn(4, 3, 16, 16, generator=g)
check('u-net denoising training (diffusion)',
      fit(UNet, [(noisy, steps, noise)] * 3, lambda m, x, t, n: F.mse_loss(m(x, t), n)), 2e-4)


# --- A vision transformer: patches by a strided convolution, a class token,
# attention over every patch, and dropout paths off.
class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 48, 4, 4)
        self.cls, self.pos = nn.Parameter(torch.zeros(1, 1, 48)), nn.Parameter(torch.randn(1, 17, 48) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(48, 4, 96, dropout=0.0, activation='gelu',
                                                                    batch_first=True, norm_first=True), 2,
                                 enable_nested_tensor=False)
        self.head = nn.Linear(48, 10)

    def forward(self, x):
        p = self.patch(x).flatten(2).transpose(1, 2)
        z = self.enc(torch.cat([self.cls.expand(len(x), -1, -1), p], 1) + self.pos)
        return self.head(z[:, 0])


check('vision transformer training', fit(ViT, [(imgs, labels)] * 3, lambda m, x, y: F.cross_entropy(m(x), y)), 2e-4)


# --- Mixture of experts: a router's top-2 choice per token, tokens gathered
# to their experts, weighted and scattered back, with a load-balancing loss.
class MoE(nn.Module):
    def __init__(self, experts=4):
        super().__init__()
        self.router = nn.Linear(32, experts)
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 32))
                                     for _ in range(experts))

    def forward(self, x):
        probs = F.softmax(self.router(x), -1)
        w, idx = probs.topk(2, -1)
        w = w / w.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if len(tok):
                out.index_add_(0, tok, expert(x[tok]) * w[tok, slot, None])
        balance = (probs.mean(0) * F.one_hot(idx[:, 0], len(self.experts)).float().mean(0)).sum()
        return out, balance


tokens, targets = torch.randn(48, 32, generator=g), torch.randn(48, 32, generator=g)


def moe_loss(m, x, y):
    out, balance = m(x)
    return F.mse_loss(out, y) + 0.01 * balance


check('mixture of experts training (top-2 routing)', fit(MoE, [(tokens, targets)] * 3, moe_loss), 2e-4)


# --- A DLRM-style recommender: sparse features through embedding bags, dense
# ones through an MLP, their pairwise interactions, a click prediction.
class DLRM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bags = nn.ModuleList(nn.EmbeddingBag(n, 16, mode='sum') for n in (100, 50, 30))
        self.bottom = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 16))
        self.top = nn.Sequential(nn.Linear(16 + 6, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, dense, ids, offsets):
        feats = [self.bottom(dense)] + [bag(ids[i], offsets[i]) for i, bag in enumerate(self.bags)]
        z = torch.stack(feats, 1)
        inter = z @ z.transpose(1, 2)
        i, j = torch.triu_indices(4, 4, 1, device=z.device)
        return self.top(torch.cat([feats[0], inter[:, i, j]], 1)).squeeze(1)


dense = torch.randn(8, 8, generator=g)
ids = torch.stack([torch.randint(0, n, (20,), generator=g) for n in (30, 30, 30)])
offsets = torch.tensor([[0, 2, 5, 7, 10, 12, 15, 18]] * 3)
clicks = torch.randint(0, 2, (8,), generator=g).float()
check('dlrm recommender training (embedding bags, interactions)',
      fit(DLRM, [(dense, ids, offsets, clicks)] * 3,
          lambda m, x, i, o, y: F.binary_cross_entropy_with_logits(m(x, i, o), y)), 2e-4)


# --- Mixed precision: the GPU trains under autocast (half, with a gradient
# scaler, and bfloat16), the CPU in float32 from the same start; the losses
# agree to what 16-bit arithmetic allows.
def mlp():
    return nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, 10))


feats, cls = torch.randn(64, 32, generator=g), torch.randint(0, 10, (64,), generator=g)


def amp(dtype):
    def run(d):
        torch.manual_seed(0)
        model = mlp().to(d)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler(d, enabled=d != 'cpu' and dtype == torch.half)
        losses = []
        for _ in range(4):
            opt.zero_grad()
            with torch.autocast(d, dtype=dtype, enabled=d != 'cpu'):
                loss = F.cross_entropy(model(feats.to(d)), cls.to(d))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.detach().float())
        return torch.stack(losses)
    return run


check('mixed precision training, half with a gradient scaler', amp(torch.half), 1e-2)
check('mixed precision training, bfloat16', amp(torch.bfloat16), 5e-2)


# --- Activation checkpointing: the backward pass recomputes each block's
# forward pass; the gradients match those kept the ordinary way.
def checkpointed(d):
    torch.manual_seed(0)
    model = GPT().to(d)
    s = seqs[:4].to(d)

    def loss(recompute):
        model.zero_grad()
        x = model.tok(s[:, :-1]) + model.pos(torch.arange(CTX, device=d))
        for blk in model.blocks:
            x = checkpoint(blk, x, use_reentrant=False) if recompute else blk(x)
        F.cross_entropy((model.ln(x) @ model.tok.weight.t()).reshape(-1, VOCAB), s[:, 1:].reshape(-1)).backward()
        return torch.cat([p.grad.flatten() for p in model.parameters()])

    kept, redone = loss(False), loss(True)
    return torch.stack([(kept - redone).abs().max(), kept.norm()])


check('activation checkpointing', checkpointed, 1e-4)


# --- A checkpoint: trained on the GPU, saved, loaded on the CPU, it gives the
# GPU's outputs.
def saved(d):
    torch.manual_seed(0)
    model = resnet().to(d)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    F.cross_entropy(model(imgs.to(d)), labels.to(d)).backward()
    opt.step()
    buf = io.BytesIO()
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict()}, buf)
    buf.seek(0)
    state = torch.load(buf, map_location='cpu')
    cpu = resnet()
    cpu.load_state_dict(state['model'])
    cpu.eval()
    with torch.no_grad():
        return cpu(imgs).flatten()


check('checkpoint saved on the device, loaded on the CPU', saved, 5e-4)
