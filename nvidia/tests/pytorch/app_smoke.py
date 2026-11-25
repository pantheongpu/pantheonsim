#!/usr/bin/env python3
"""Application smoke checks: small PyTorch programs whose outputs are compared with a real card's.

models.py and sweep.py compare the simulated GPU with PyTorch's CPU backend. The CPU is not what an
application meets on a card: cuBLAS and cuDNN choose their own kernels, and the sum order, the rounding
of float16 results and the TF32 cut all differ. This script runs programs of the shapes real workloads
have (gemm in four precisions, convolution forward and backward, a cuDNN LSTM, an MLP, a CNN with batch
norm and a decoder trained a few steps, with logits, losses and greedy tokens) and compares what they
produce with the tensors a real card produced, recorded in nvidia/tests/data/pytorch_app_smoke.<card>.ref (rtx3060, a10g, t4;
the file of the device's compute capability, rtx3060 where none was recorded: app_smoke_ref.card_for; APP_SMOKE_REF_CARD overrides).
app_smoke_ref.py says, per workload, what is allowed to differ and why.

  app_smoke.py             run on the simulated GPU (run it with `vgpu run --preload`, as run_pytorch.sh
                           does) and check against the card's tensors; "ok <name>" or "FAIL <name>: <why>"
  app_smoke.py --card      the same check on a real GPU (does the reference still hold on the card?)
  app_smoke.py --write F   record the tensors on a real GPU into F. Each workload runs twice there; a
                           bit-for-bit entry that differs between the two runs, or a tolerance below four
                           times the card's own run-to-run difference, refuses the recording.

Prints "SKIP: ..." and exits 0 where PyTorch or a CUDA device is missing.
"""
import ctypes
import datetime
import os
import sys

# Before torch is imported: cuBLAS's workspace setting that deterministic mode needs.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import app_smoke_ref as ref  # noqa: E402

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:  # not installed, or a build that cannot load here
    print(f"SKIP: PyTorch is not usable ({type(e).__name__}: {e})")
    sys.exit(0)
if not torch.version.cuda:
    print("SKIP: this PyTorch is not a CUDA build")
    sys.exit(0)

# The card the reference was recorded on: the one of the device's compute capability where there is one (app_smoke_ref.card_for: a T4
# for sm_75, an A10G for sm_86, the RTX 3060 otherwise); APP_SMOKE_REF_CARD=<slug> reads pytorch_app_smoke.<slug>.ref instead.
REF_CARD = os.environ.get('APP_SMOKE_REF_CARD') or ref.card_for(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (8, 6))[0]
REF_PATH = os.path.join(os.environ.get('VGPU_E2E_DATA', os.path.join(HERE, '..', 'data')),
                        f'pytorch_app_smoke.{REF_CARD}.ref')
GPU = 'cuda'
MIN_MARGIN = 0.02  # the top two logits of a recorded greedy token differ by at least this
MARGINS = []


def data(*shape, seed=1, scale=1.0):
    """The same numbers on every machine: drawn on the CPU from a seeded generator."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * scale


def flat(t):
    t = t.detach().float().cpu()
    return list(t.shape), t.reshape(-1).tolist()


def cat(*tensors):
    parts = [t.detach().float().cpu().reshape(-1) for t in tensors]
    v = torch.cat(parts)
    return [v.numel()], v.tolist()


# --- one workload each; every one takes the device and returns (shape, flat floats) ---

def elementwise_f32(d):
    x = data(1024, seed=2).to(d)
    y = torch.clamp(torch.relu(x * 2.0 + 1.0) - 0.5 * x, max=3.0) + x * x
    return flat(y)


def int_ops(d):
    g = torch.Generator().manual_seed(3)
    x = torch.randint(-1000, 1000, (4, 129), generator=g).to(d)
    srt = torch.sort(x, dim=1, stable=True)
    out = torch.cat([x.cumsum(1)[:, -5:].reshape(-1), srt.indices[:, :16].reshape(-1),
                     x.argmax(1), x.gather(1, srt.indices[:, :4]).reshape(-1)])
    return flat(out.float())


def gemm_f32(d):
    torch.backends.cuda.matmul.allow_tf32 = False
    return flat(data(32, 48, seed=4).to(d) @ data(48, 40, seed=5).to(d))


def gemm_f16(d):
    return flat((data(32, 48, seed=4).to(d).half() @ data(48, 40, seed=5).to(d).half()))


def gemm_bf16(d):
    return flat((data(32, 48, seed=4).to(d).bfloat16() @ data(48, 40, seed=5).to(d).bfloat16()))


def gemm_tf32(d):
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        return flat(data(32, 48, seed=4).to(d) @ data(48, 40, seed=5).to(d))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False


def bmm_f32(d):
    return flat(torch.bmm(data(4, 16, 24, seed=6).to(d), data(4, 24, 20, seed=7).to(d)))


def softmax_layernorm(d):
    x = data(8, 64, seed=8, scale=3.0).to(d)
    w, b = data(64, seed=9).to(d), data(64, seed=10).to(d)
    return cat(F.softmax(x, dim=-1), F.layer_norm(x, (64,), w, b), F.gelu(x))


def conv2d_f32(d):
    torch.backends.cudnn.allow_tf32 = False
    return flat(F.conv2d(data(2, 4, 12, 12, seed=11).to(d), data(8, 4, 3, 3, seed=12, scale=0.2).to(d), padding=1))


def conv2d_f16(d):
    x = data(2, 4, 12, 12, seed=11).to(d).half().contiguous(memory_format=torch.channels_last)
    w = data(8, 4, 3, 3, seed=12, scale=0.2).to(d).half().contiguous(memory_format=torch.channels_last)
    return flat(F.conv2d(x, w, padding=1))


def conv2d_grads(d):
    torch.backends.cudnn.allow_tf32 = False
    x = data(2, 4, 12, 12, seed=11).to(d).requires_grad_()
    w = data(8, 4, 3, 3, seed=12, scale=0.2).to(d).requires_grad_()
    y = F.conv2d(x, w, stride=2, padding=1)
    y.backward(data(*y.shape, seed=13).to(d))
    return cat(x.grad, w.grad)


def lstm_cudnn(d):
    torch.manual_seed(0)
    m = nn.LSTM(16, 32, num_layers=2, batch_first=True).to(d)
    out, (h, c) = m(data(2, 6, 16, seed=14).to(d))
    return cat(out, h, c)


def mlp_train(d):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 10)).to(d)
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
    x, y = data(16, 32, seed=15).to(d), torch.randint(0, 10, (16,), generator=torch.Generator().manual_seed(16)).to(d)
    losses = []
    for _ in range(4):
        opt.zero_grad()
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.detach())
    return cat(torch.stack(losses), m(x))


def cnn_train(d):
    torch.manual_seed(0)
    torch.backends.cudnn.allow_tf32 = False
    m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
                      nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                      nn.Flatten(), nn.Linear(16, 5)).to(d)
    opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
    x, y = data(8, 3, 16, 16, seed=17).to(d), torch.randint(0, 5, (8,), generator=torch.Generator().manual_seed(18)).to(d)
    losses = []
    for _ in range(3):
        opt.zero_grad()
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.detach())
    m.eval()
    return cat(torch.stack(losses), m(x))


VOCAB, CTX, DIM, HEADS = 64, 16, 64, 4


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(DIM), nn.LayerNorm(DIM)
        self.qkv, self.proj = nn.Linear(DIM, 3 * DIM), nn.Linear(DIM, DIM)
        self.mlp = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(), nn.Linear(4 * DIM, DIM))

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, HEADS, DIM // HEADS).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, DIM)
        x = x + self.proj(a)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(VOCAB, DIM), nn.Embedding(CTX, DIM)
        self.blocks = nn.Sequential(Block(), Block())
        self.ln = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB, bias=False)
        with torch.no_grad():
            self.head.weight.mul_(4.0)  # spread the logits so the greedy tokens are well separated

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        return self.head(self.ln(self.blocks(x)))


def gpt_model(d):
    torch.manual_seed(0)
    return GPT().to(d)


def prompt(d, rows=2, n=8, seed=19):
    return torch.randint(0, VOCAB, (rows, n), generator=torch.Generator().manual_seed(seed)).to(d)


def gpt_logits(d):
    with torch.no_grad():
        return flat(gpt_model(d)(prompt(d)))


def gpt_sdpa_f16(d):
    q, k, v = (data(1, 2, 16, 32, seed=s).to(d).half() for s in (20, 21, 22))
    return flat(F.scaled_dot_product_attention(q, k, v, is_causal=True))


def gpt_greedy_tokens(d):
    m = gpt_model(d)
    idx = prompt(d, rows=2, n=4)
    with torch.no_grad():
        for _ in range(8):
            logits = m(idx)[:, -1]
            top2 = logits.float().topk(2, dim=-1).values.cpu()
            MARGINS.append(float((top2[:, 0] - top2[:, 1]).min()))
            idx = torch.cat([idx, logits.argmax(-1, keepdim=True)], dim=1)
    return flat(idx.float())


def gpt_train(d):
    m = gpt_model(d)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
    losses = []
    for step in range(3):
        idx = prompt(d, rows=4, n=CTX, seed=30 + step)
        opt.zero_grad(set_to_none=True)
        logits = m(idx[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), idx[:, 1:].reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        losses.append(loss.detach())
    return flat(torch.stack(losses))


WORKLOADS = [elementwise_f32, int_ops, gemm_f32, gemm_f16, gemm_bf16, gemm_tf32, bmm_f32, softmax_layernorm,
             conv2d_f32, conv2d_f16, conv2d_grads, lstm_cudnn, mlp_train, cnn_train, gpt_logits, gpt_sdpa_f16,
             gpt_greedy_tokens, gpt_train]
assert [w.__name__ for w in WORKLOADS] == list(ref.TOLERANCES), 'app_smoke_ref.TOLERANCES and WORKLOADS differ'


def run_all():
    """{name: (shape, floats) or an exception string}; the workloads are independent."""
    out = {}
    for w in WORKLOADS:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            shape, vals = w(GPU)
            torch.cuda.synchronize()
            out[w.__name__] = (shape, vals)
        except Exception as e:  # a refused kernel or library call shows up here, and says which
            out[w.__name__] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    return out


def record(path):
    MARGINS.clear()
    first, second = run_all(), run_all()
    bad = []
    for name, (tol, _) in ref.TOLERANCES.items():
        a, b = first[name], second[name]
        if isinstance(a, str) or isinstance(b, str):
            bad.append(f"{name}: {a if isinstance(a, str) else b}")
            continue
        entry = {'name': name, 'values': a[1], 'shape': a[0]}
        ok, msg = ref.compare(entry, b[0], b[1])
        if tol == 0.0 and not ok:
            bad.append(f"{name} is listed bit for bit but the card's two runs differ ({msg})")
        elif tol > 0.0:
            scale = max(abs(v) for v in a[1]) or 1.0
            noise = max(abs(x - y) for x, y in zip(a[1], b[1])) / scale
            if tol < 4 * noise:
                bad.append(f"{name}: tolerance {tol:g} is under four times the card's run-to-run difference {noise:g}")
    if not MARGINS:
        bad.append("gpt_greedy_tokens: no top-two logit gap was measured")
    elif min(MARGINS) < MIN_MARGIN:
        bad.append(f"gpt_greedy_tokens: top-two logit gap {min(MARGINS):g} is under {MIN_MARGIN:g}; a token could flip")
    if bad:
        print("refusing to record:")
        for b in bad:
            print("  " + b)
        return 1
    gpu = torch.cuda.get_device_name(0)
    ref.write_ref(path, [
        f"torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}, on {gpu}"
        f" (compute capability {'.'.join(map(str, torch.cuda.get_device_capability(0)))}), recorded {datetime.date.today()}",
        "Tensors nvidia/tests/pytorch/app_smoke.py produced on a real card; each workload ran twice and agreed as its tolerance says.",
        f"Smallest top-two logit gap of the greedy tokens: {min(MARGINS):.3f}",
        "Regenerate on a real GPU: app_smoke.py --write <this file>. Tolerances and their reasons: app_smoke_ref.py",
    ], {n: first[n] for n in ref.TOLERANCES})
    print("wrote", path)
    return 0


def check():
    header, entries = ref.read_ref(REF_PATH)
    results = run_all()
    failed = False
    for name in ref.TOLERANCES:
        r = results[name]
        if name not in entries:
            print(f"FAIL {name}: the reference file has no entry for it")
        elif isinstance(r, str):
            print(f"FAIL {name}: {r}")
        else:
            ok, msg = ref.compare(entries[name], r[0], r[1])
            print(f"ok {name} ({msg})" if ok else f"FAIL {name}: {msg}")
            if ok:
                continue
        failed = True
    if failed:
        print(f"      (the card's tensors are from: {header[0] if header else '?'}; this run is torch {torch.__version__}."
              " If PyTorch's kernels changed, regenerate with --write on a real GPU)")
    return 1 if failed else 0


def main():
    args = sys.argv[1:]
    write = None
    while args:
        a = args.pop(0)
        if a == '--write':
            write = args.pop(0)
        elif a == '--card':
            pass  # the check is the same; this says the process is on a real GPU
        else:
            print(f"unknown argument {a}", file=sys.stderr)
            return 2
    if not torch.cuda.is_available():
        print("SKIP: PyTorch sees no CUDA device")
        return 0
    # The simulator's driver shim wants cuInit first (see profiler_check.py); idempotent against NVIDIA's.
    ctypes.CDLL('libcuda.so.1').cuInit(0)
    return record(write) if write else check()


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    code = main()
    sys.stdout.flush()
    os._exit(code)  # as sweep.py: PyTorch's autograd thread can abort interpreter finalisation
