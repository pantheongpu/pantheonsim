#!/usr/bin/env python3
"""Which scaled_dot_product_attention kernels PyTorch runs for app_smoke's gpt_sdpa_f16 on this device, and what each backend prints.

A card-side probe (issue "PyTorch SDPA f16: h100 and b200 profiles differ from the sm_86 reference"): PyTorch picks the SDPA backend
(flash, memory-efficient, cuDNN, math) by compute capability and library version, so the same program runs different kernels on
different cards. For the inputs of gpt_sdpa_f16 this prints, for the default choice and for each backend forced, the kernel names the
profiler saw and the float32 bits of the output, and the normwise error of each against the RTX 3060's recorded tensor.

  sdpa_backend_probe.py [reference file]      (default: ../data/pytorch_app_smoke.rtx3060.ref)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402
import app_smoke_ref as ref  # noqa: E402


def data(*shape, seed):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def inputs():
    return [data(1, 2, 16, 32, seed=s).cuda().half() for s in (20, 21, 22)]


def run(backend):
    q, k, v = inputs()
    ctx = sdpa_kernel(backend) if backend is not None else sdpa_kernel(list(SDPBackend))
    try:
        with ctx:
            F.scaled_dot_product_attention(q, k, v, is_causal=True)  # warm up
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                torch.cuda.synchronize()
    except Exception as e:  # a backend that does not support this device or shape
        return None, [f'unavailable: {type(e).__name__}: {str(e)[:100]}']
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return o.float().cpu().reshape(-1).tolist(), names


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'data', 'pytorch_app_smoke.rtx3060.ref')
    want = None
    if os.path.exists(path):
        _, entries = ref.read_ref(path)
        want = entries['gpt_sdpa_f16']['values']
    print(f'device {torch.cuda.get_device_name(0)} cc {torch.cuda.get_device_capability(0)} torch {torch.__version__} cuda {torch.version.cuda}'
          f' cudnn {torch.backends.cudnn.version()}')
    for label, backend in [('default', None), ('flash', SDPBackend.FLASH_ATTENTION), ('efficient', SDPBackend.EFFICIENT_ATTENTION),
                           ('cudnn', SDPBackend.CUDNN_ATTENTION), ('math', SDPBackend.MATH)]:
        vals, names = run(backend)
        print(f'--- {label}')
        for n in names:
            print('  kernel', n[:160])
        if vals is None:
            continue
        err = ''
        if want:
            scale = max(abs(x) for x in want)
            err = max(abs(a - b) for a, b in zip(vals, want)) / scale
        print(f'  normwise error against {os.path.basename(path)}: {err}')
        print('  bits', ref.bits_of(vals))


if __name__ == '__main__':
    main()
