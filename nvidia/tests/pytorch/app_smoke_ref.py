"""Reference file and comparison for the application smoke checks (app_smoke.py).

No torch, no simulator: this module only reads and writes the reference file and decides whether a result
matches an entry, so test_app_smoke_ref.py can check the drift detection anywhere.

A reference entry is the tensor a real card produced for one workload, stored as the exact bits of the
float32 values (an int64 or a half is converted to float32 first, which is exact for the small values here):

    {"name": "gemm_f32", "tol": 1e-06, "shape": [64, 80], "hex": "3f800000..."}

`tol` is the allowed normwise error, max|got - ref| / max|ref|, and 0 means bit for bit. Which workload gets
which tolerance, and why, is the table TOLERANCES below: bit for bit where the card's arithmetic is a fixed
sequence of correctly rounded operations (elementwise float32, integers, an argmax of well-separated logits),
a stated bound where the order of a sum or the choice of a library algorithm is the card's own (cuBLAS and
cuDNN pick kernels by heuristics that differ between a card and the simulator, so a K-long float32 sum may be
associated differently: a few ulps; a float16 or bfloat16 result is rounded to 11 or 8 bits first, so the
bound is a fraction of that rounding step).
"""
import json
import math
import struct

# name -> (tolerance, why). Order is the order app_smoke.py runs and prints them in.
TOLERANCES = {
    'elementwise_f32': (0.0, 'add, mul, relu, clamp: one correctly rounded operation each'),
    'int_ops': (0.0, 'integer cumsum, sort, argmax, gather'),
    'gemm_f32': (1e-6, 'cuBLAS sgemm, TF32 off: K=48 float32 sum, association is the library kernel'),
    'gemm_f16': (2e-3, 'cuBLAS hgemm on tensor cores, float16 result (11 bits: half an ulp is 5e-4)'),
    'gemm_bf16': (1e-2, 'cuBLAS bf16 gemm on tensor cores, bfloat16 result (8 bits: half an ulp is 4e-3)'),
    'gemm_tf32': (2e-3, 'cuBLAS sgemm with TF32 on: inputs cut to 10 bits, float32 accumulate'),
    'bmm_f32': (1e-6, 'batched float32 gemm'),
    'softmax_layernorm': (2e-6, 'softmax and layer norm: expf, a reduction, a division'),
    'conv2d_f32': (1e-5, 'cuDNN float32 convolution forward, TF32 off'),
    'conv2d_f16': (2e-3, 'cuDNN float16 convolution forward on tensor cores'),
    'conv2d_grads': (1e-4, 'cuDNN backward data and filter, deterministic algorithms forced'),
    'lstm_cudnn': (2e-5, "cuDNN's fused LSTM, two layers, forward"),
    'mlp_train': (1e-4, 'four SGD steps of a three-layer MLP: losses and logits'),
    'cnn_train': (1e-3, 'three steps of conv, batch norm, ReLU, pool, linear: losses and logits'),
    'gpt_logits': (1e-4, 'a two-layer decoder, float32, SDPA: logits of a fixed prompt'),
    'gpt_sdpa_f16': (5e-3, 'scaled_dot_product_attention in float16'),
    'gpt_greedy_tokens': (0.0, 'eight greedy tokens (argmax); recording refuses a run whose top two logits are close'),
    'gpt_train': (1e-3, 'three AdamW steps of the decoder: losses'),
}


# The card whose tensors a device is held to: one of the same compute capability where one was recorded (cuBLAS and cuDNN pick kernels by
# architecture, so a T4's tensors are not an A10G's: gemm_f32 differs in 999 of 1280 values between them), else the RTX 3060's, which
# makes the comparison a bound on another architecture's numbers and not that architecture's. Every compute capability that has a card in
# nvidia/tests/data (pytorch_app_smoke.<slug>.ref) is here.
CARD_OF_CC = {(7, 5): 't4', (8, 6): 'a10g'}
FALLBACK_CARD = 'rtx3060'


def card_for(cc):
    """-> (slug of the reference file, True when a card of this compute capability was recorded)."""
    slug = CARD_OF_CC.get(tuple(cc))
    return (slug, True) if slug else (FALLBACK_CARD, tuple(cc) == (8, 6))


def bits_of(values):
    """Hex of a flat list of Python floats as float32 bits."""
    return ''.join('%08x' % struct.unpack('<I', struct.pack('<f', v))[0] for v in values)


def floats_of(hexstr):
    return list(struct.unpack('>%df' % (len(hexstr) // 8), bytes.fromhex(hexstr)))


def write_ref(path, header, entries):
    """entries: {name: (shape, list of floats)}; tolerances come from TOLERANCES."""
    with open(path, 'w') as f:
        for line in header:
            f.write('# ' + line + '\n')
        for name in TOLERANCES:
            shape, vals = entries[name]
            f.write(json.dumps({'name': name, 'tol': TOLERANCES[name][0], 'shape': list(shape),
                                'hex': bits_of(vals)}, sort_keys=True) + '\n')


def read_ref(path):
    """-> (header lines, {name: entry}) with entry['values'] the floats."""
    header, entries = [], {}
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith('#'):
                header.append(line[1:].strip())
                continue
            e = json.loads(line)
            if e['name'] in entries:
                raise ValueError('duplicate entry ' + e['name'])
            n = 1
            for d in e['shape']:
                n *= d
            if len(e['hex']) != 8 * n:
                raise ValueError('%s: %d values for shape %s' % (e['name'], len(e['hex']) // 8, e['shape']))
            e['values'] = floats_of(e['hex'])
            entries[e['name']] = e
    return header, entries


def compare(entry, shape, got):
    """-> (ok, message) for a result (shape, flat floats) against a reference entry."""
    ref = entry['values']
    tol = TOLERANCES[entry['name']][0]
    if list(shape) != list(entry['shape']):
        return False, 'shape %s, the card\'s %s' % (list(shape), entry['shape'])
    if len(got) != len(ref):
        return False, '%d values, the card\'s %d' % (len(got), len(ref))
    if tol == 0.0:
        bad = [i for i, (a, b) in enumerate(zip(got, ref)) if bits_of([a]) != bits_of([b])]
        if bad:
            i = bad[0]
            return False, 'not bit for bit: %d of %d differ, first at %d (%r, the card\'s %r)' % (len(bad), len(ref), i, got[i], ref[i])
        return True, 'bit for bit (%d values)' % len(ref)
    scale = max((abs(b) for b in ref if math.isfinite(b)), default=0.0)
    if any(math.isnan(a) != math.isnan(b) or (math.isinf(a) and a != b) for a, b in zip(got, ref)):
        return False, 'a NaN or an infinity where the card has none (or the reverse)'
    err = max((abs(a - b) for a, b in zip(got, ref) if math.isfinite(a) and math.isfinite(b)), default=0.0)
    rel = err / scale if scale else err
    if rel <= tol:
        return True, 'normwise error %.3g, allowed %.3g' % (rel, tol)
    return False, 'normwise error %.3g, allowed %.3g' % (rel, tol)
